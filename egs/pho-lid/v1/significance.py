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
        print(f"d_model: {self.d_model}")
        if self.use_pho is True and self.use_tac is True:
            self.scoring = nn.Linear(self.d_model*2, self.n_lang, bias=True)
        else:
            self.scoring = nn.Linear(self.d_model, self.n_lang, bias=True)

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
            if param not in self.pholidModel.clf.parameters():
                param.requires_grad = False
        # print(f"x: {type(x)}, {type(seq_len)}, {type(atten_mask)}")

        print(f"x: {x.shape}")
        batch_size, max_seq_len, _, feat_dim = x.shape
        print(f"batch_size: {batch_size}, max_seq_len: {max_seq_len}, feat_dim: {feat_dim}")

        h_pho = self.pholidModel.get_pho_embeddings(x, seq_len, atten_mask)
        h_tac = self.pholidModel.get_embeddings(x, seq_len, atten_mask)
        h_cat = torch.cat((h_tac, h_pho), dim=-1)
        print(f"h_tac: {h_tac.shape}, h_pho: {h_pho.shape}, h_cat: {h_cat.shape}")

        if self.use_pho is True and self.use_tac is False:
            s = self.scoring(h_pho.reshape(-1, self.d_model))
        elif self.use_tac is True:
            s = self.scoring(h_cat.reshape(-1, self.d_model*2))
        else:
            s = self.scoring(h_tac.reshape(-1, self.d_model))
        print(f"s: {s.shape}")

        s = s.reshape(batch_size, max_seq_len, -1)

        # s = s.reshape(batch_size, max_seq_len, self.n_lang)
        print(f"s reshape: {s.shape}")

        s = self.softmax(s).unsqueeze(-1)
        print(f"s softmax: {s.shape}")

        # s = map(torch.diag_embed, s)
        h_repr = s * h_cat.unsqueeze(-2)
        print(f"h_repr: {h_repr.shape}")

        # h_repr = h_repr.reshape(batch_size, max_seq_len, n_lang, -1)
        # print(f"h_repr reshape: {h_repr.shape}")

        outputs = self.pholidModel.clf(h_repr)
        print(f"outputs: {outputs.shape}")

        return outputs
        