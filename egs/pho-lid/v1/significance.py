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
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # freeze params in pholidModel
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


    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
                atten_mask=None, eps=1e-5, return_score=False):
        batch_size, max_seq_len, _, feat_dim = x.shape
        h_cat = self.pholidModel.get_embeddings(x, seq_len, atten_mask)
        # print(f"h_cat: {h_cat.shape}")

        s = self.scoring(h_cat.reshape(-1, self.d_model*2))
        raw_scores = s
        # print(f"without softmax: {s}\n")
        s = s.reshape(batch_size, max_seq_len, -1)
        # print(f"no softmax s mean:{s.mean()}, std: {s.std()}")

        s = self.softmax(s)
        # s = self.sigmoid(s)
        # print(f"with softmax: {s}\n\n")
        # s = self.softmax(s)
        # s = s.unsqueeze(-1)
        # print(f"s softmax: {s.shape}")

        # s = map(torch.diag_embed, s)
        # h_repr = s.unsqueeze(-1) * h_cat.unsqueeze(-2)
        h_repr = s[:, :, 0].unsqueeze(-1) * h_cat
        # print(f"h_repr: {h_repr.shape}")
        # print(f"s mean:{s.mean()}, std: {s.std()}")

        # h_repr = h_repr.reshape(batch_size, max_seq_len, n_lang, -1)
        # print(f"h_repr reshape: {h_repr.shape}")

        # outputs = self.pholidModel.lid_clf(h_cat) * s
        outputs = self.pholidModel.lid_clf(h_repr)
        # print(f"outputs: {outputs.shape}, s: {s.shape}")
        # output = s * outputs

        if return_score:
            return outputs, raw_scores, s, h_cat, h_repr
        else:
            return outputs

    def get_scoring_reg(self, p=2):
        reg = torch.tensor(0., requires_grad=True)

        for param in self.scoring.named_parameters():
            if "weight" in param[0]:
                reg = reg + torch.square(param[1]).sum()

        return reg
        # for param in self.scoring.named_parameters():
        #     if "weight" in param[0]:
        #         # print(f"{torch.square(param[1]).sum()}, {param}")
        #         return torch.square(param[1]).sum()

    def get_embeddings(self, x, seq_len, atten_mask=None):
        batch_size, max_seq_len, _, feat_dim = x.shape
        h_cat = self.pholidModel.get_embeddings(x, seq_len, atten_mask)
        s = self.scoring(h_cat.reshape(-1, self.d_model*2))
        s = s.reshape(batch_size, max_seq_len, -1)
        s = self.softmax(s).unsqueeze(-1)

        h_repr = s * h_cat.unsqueeze(-2)
        return h_repr

    def bf_check(self, vec, seq_len,  mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None):
        return self.pholidModel.lid_clf(vec)