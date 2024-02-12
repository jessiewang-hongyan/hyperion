from model import PHOLID
import torch
from torch import nn

class ScoringModel(nn.Module):
    def __init__(self, pholidModel, n_lang, use_pho=True, use_tac=False):
        super().__init__()
        self.pholidModel = pholidModel
        self.n_lang = n_lang
        self.use_pho = use_pho
        self.use_pho = use_pho
        self.use_tac = use_tac

        self.d_model = self.pholidModel.d_model
        # print(f"d_model: {self.d_model}")
        # if self.use_pho is True and self.use_tac is True:
        self.scoring = nn.Linear(self.d_model*2, self.n_lang, bias=True)
        # else:
        #     self.scoring = nn.Linear(self.d_model, self.n_lang, bias=True)

        self.softmax = nn.Softmax(dim=1)

        # self.clf = nn.Sequential(
        #     nn.Linear(self.d_model*2, int(self.d_model / 4), bias=True),
        #     nn.Tanh(),
        #     nn.Linear(int(self.d_model / 4), 1, bias=True)
        # )

    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
                atten_mask=None, eps=1e-5):

        # freeze all params in pholidModel
        for param in self.pholidModel.parameters(): 
            param.requires_grad = False
        # for param in self.pholidModel.attention_block2.parameters():
        #     param.requires_grad = True
        # for param in self.pholidModel.attention_block1.parameters():
        #     param.requires_grad = True
        # for param in self.pholidModel.conv_1.parameters():
        #     param.requires_grad = True
        for param in self.pholidModel.lid_clf.parameters():
            param.requires_grad = True

        # print(f"x: {type(x)}, {type(seq_len)}, {type(atten_mask)}")

        # print(f"x: {x.shape}")
        batch_size, max_seq_len, _, feat_dim = x.shape
        # print(f"batch_size: {batch_size}, max_seq_len: {max_seq_len}, feat_dim: {feat_dim}")

        # h_pho = self.pholidModel.get_pho_embeddings(x, seq_len, atten_mask)
        h_cat = self.pholidModel.get_embeddings(x, seq_len, atten_mask)
        # print(f"h_cat shape: {h_cat.shape}, split: {[x.shape for x in torch.split(h_cat, 2, dim=-1)]}")
        # h_pho, h_tac = torch.split(h_cat, self.d_model, dim=-1)
        # print(f"h_cat: {h_cat.shape}")

        # if self.use_pho is True and self.use_tac is False:
        #     s = self.scoring(h_pho.reshape(-1, self.d_model))
        # elif self.use_tac is True and self.use_pho is False:
        s = self.scoring(h_cat.reshape(-1, self.d_model*2))
        # else:
            # s = self.scoring(h_tac.reshape(-1, self.d_model*2))
        # print(f"s: {s.shape}")

        s = s.reshape(batch_size, max_seq_len, -1)

        # s = s.reshape(batch_size, max_seq_len, self.n_lang)
        # print(f"s reshape: {s.shape}")

        s = self.softmax(s).unsqueeze(-1)
        # print(f"s softmax: {s.shape}")

        # s = map(torch.diag_embed, s)
        h_repr = s * h_cat.unsqueeze(-2)
        # print(f"h_repr: {h_repr.shape}")

        # h_repr = h_repr.reshape(batch_size, max_seq_len, n_lang, -1)
        # print(f"h_repr reshape: {h_repr.shape}")

        outputs = self.pholidModel.lid_clf(h_repr)
        # print(f"outputs: {outputs.shape}")

        return outputs

    def get_scoring_reg(self, p=2):
        reg = torch.tensor(0., requires_grad=True)

        for param in self.scoring.named_parameters():
            if "weight" in param[0]:
                reg = reg + torch.square(param[1]).sum()

        return reg
        