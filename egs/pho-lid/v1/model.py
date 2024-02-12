from transformer import *


class CNN_Trans_LID(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=8,
                 dropout=0.1, n_lang=3, max_seq_len=10000):
        super(CNN_Trans_LID, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.shared_TDNN = nn.Sequential(nn.Dropout(p=dropout),
                                         nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=True),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=True),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=True),
                                         )
        self.fc_xv = nn.Linear(1024, feat_dim)

        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)

        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

    def mean_std_pooling(self, x, batchsize, seq_lens, mask_mean, weight_mean, mask_std, weight_unb):
        max_len = seq_lens[0]
        feat_dim = x.size(-1)
        if mask_mean is not None:
            assert mask_mean.size() == x.size()
            x.masked_fill_(mask_mean, 0)
        correct_mean = x.mean(dim=1).transpose(0, 1) * weight_mean
        correct_mean = correct_mean.transpose(0, 1)
        center_seq = x - correct_mean.repeat(1, 1, max_len).view(batchsize, -1, feat_dim)
        variance = torch.mean(torch.mul(torch.abs(center_seq) ** 2, mask_std), dim=1).transpose(0,1) \
                   * weight_unb * weight_mean
        std = torch.sqrt(variance.transpose(0, 1))
        return torch.cat((correct_mean, std), dim=1)

    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
                atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = self.dropout(x)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)

        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x += noise * eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        if std_mask_ is not None:
            stats = self.mean_std_pooling(output, batch_size, seq_len, mean_mask_, weight_mean,
                                          std_mask_, weight_unbaised)
        else:
            stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

class PHOLID(nn.Module):
    def __init__(self,input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=8,
                 dropout=0.1, n_lang=3, max_seq_len=10000):
        super(PHOLID, self).__init__()
        self.input_dim = input_dim

        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        self.shared_TDNN = nn.Sequential(nn.Dropout(p=dropout),
                                         nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1))
        self.phoneme_proj = nn.Linear(512, 64)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.fc_xv = nn.Linear(1024, feat_dim)
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.lid_clf = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                     nn.ReLU(),
                                     nn.Linear(self.d_model, self.d_model),
                                     nn.ReLU(),
                                     nn.Linear(self.d_model, n_lang))

    def mean_std_pooling(self, x, batchsize, seq_lens, mask_mean, weight_mean, mask_std, weight_unb):
        max_len = seq_lens[0]
        feat_dim = x.size(-1)
        if mask_mean is not None:
            assert mask_mean.size() == x.size()
            x.masked_fill_(mask_mean, 0)
        correct_mean = x.mean(dim=1).transpose(0, 1) * weight_mean
        correct_mean = correct_mean.transpose(0, 1)
        center_seq = x - correct_mean.repeat(1, 1, max_len).view(batchsize, -1, feat_dim)
        variance = torch.mean(torch.mul(torch.abs(center_seq) ** 2, mask_std), dim=1).transpose(0, 1) \
                   * weight_unb * weight_mean
        std = torch.sqrt(variance.transpose(0, 1))
        return torch.cat((correct_mean, std), dim=1)

    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
                atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)

        pho_x = x.transpose(-1, -2)
        pho_out = self.phoneme_proj(pho_x)
        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x += noise * eps

        seg_stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(seg_stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)

        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        if std_mask_ is not None:
            stats = self.mean_std_pooling(output, batch_size, seq_len, mean_mask_, weight_mean,
                                          std_mask_, weight_unbaised)
        else:
            stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)

        output = self.lid_clf(stats)

        return output, pho_out.reshape(batch_size, T_len, -1, 64)

    
    def get_embeddings(self, x, seq_len, atten_mask=None):
        # self.eval()
        batch_size = x.size(0)
        T_len = x.size(1)

        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)
        seg_stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(seg_stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)

        # print(f'The dim of attention output: {output.shape}')
        return output

    def bf_check(self, vec, seq_len,  mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None):
        batch_size = vec.size(0)
        vec = vec.squeeze().repeat(1, 1, 2)
        output = self.lid_clf(vec)

        return output


class PHOLID_conv(PHOLID):
    def __init__(self,input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=8,
                 dropout=0.1, n_lang=3, max_seq_len=10000, conv_kernel_size=5):
        super().__init__(input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads,
                 dropout, n_lang, max_seq_len)
        self.conv_1 = nn.Conv1d(self.d_model, self.d_model, tuple([conv_kernel_size]),
                                 padding=tuple([int((conv_kernel_size)/2)]),
                                 stride=1)
        # for param in self.conv_1.parameters():
        #     param.requires_grad = False

    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
                atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)

        pho_x = x.transpose(-1, -2)
        pho_out = self.phoneme_proj(pho_x)
        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x = x + noise * eps

        seg_stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(seg_stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)

        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # add conv layer 1
        output = output.transpose(1, 2)
        output = self.conv_1(output)
        output = output.transpose(1, 2)

        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)

        if std_mask_ is not None:
            stats = self.mean_std_pooling(output, batch_size, seq_len, mean_mask_, weight_mean,
                                          std_mask_, weight_unbaised)
        else:
            stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        
        output = self.lid_clf(stats)

        return output, pho_out.reshape(batch_size, T_len, -1, 64)


    def get_embeddings(self, x, seq_len, atten_mask=None):
        batch_size = x.size(0)
        T_len = x.size(1)

        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)
        seg_stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(seg_stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # add conv layer 1
        output = output.transpose(1, 2)
        output = self.conv_1(output)
        output = output.transpose(1, 2)

        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)

        return output

    def bf_check(self, vec, seq_len,  mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None):
        batch_size = vec.size(0)
        vec = vec.squeeze().repeat(1, 1, 2)
        output = self.lid_clf(vec)

        return output

class PHOLID_conv_pho(PHOLID_conv):
    def __init__(self,input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=8,
                 dropout=0.1, n_lang=3, max_seq_len=10000, conv_kernel_size=5):
        super().__init__(input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads,
                 dropout, n_lang, max_seq_len)

    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
            atten_mask=None, eps=1e-5, norm_tac=True):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)

        pho_x = x.transpose(-1, -2)
        pho_out = self.phoneme_proj(pho_x)

        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x = x + noise * eps

        seg_stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(seg_stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)

        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # add conv layer 1
        output = output.transpose(1, 2)
        output = self.conv_1(output)
        output = output.transpose(1, 2)

        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)

        if norm_tac == True:
            output = torch.nn.functional.normalize(output, dim=1)

        if std_mask_ is not None:
            stats = self.mean_std_pooling(output, batch_size, seq_len, mean_mask_, weight_mean,
                                          std_mask_, weight_unbaised)
        else:
            stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        
        output = self.lid_clf(stats)

        return output, pho_out.reshape(batch_size, T_len, -1, 64)

    def get_embeddings(self, x, seq_len, atten_mask=None, norm_pho=True, norm_tac=True):
        # self.eval()
        batch_size = x.size(0)
        T_len = x.size(1)

        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)
        seg_stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(seg_stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # add conv layer 1
        output = output.transpose(1, 2)

        output = self.conv_1(output)
        output = output.transpose(1, 2)

        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)

        x = x.reshape(batch_size, T_len, x.shape[-2], x.shape[-1])
        x_mean = torch.mean(x, dim=-1)
        # print(f"bf: phonotactic: {output.shape}\n\t{output.mean()}\n, phoneme: {x_mean.shape}\n\t{x_mean.mean()}\n")
        
        if norm_pho == True:
            x_mean = torch.nn.functional.normalize(x_mean, dim=1)
        if norm_tac == True:
            output = torch.nn.functional.normalize(output, dim=1)
        
        output = torch.cat((output, x_mean), dim=-1)
        # print(f"bf: cat: {output.shape}")
        
        # output = self.lid_clf(stats)

        # print(f'The dim of attention output: {output.shape}')
        return output

    def bf_check(self, vec, seq_len,  mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None):
        batch_size = vec.size(0)
        # vec = vec.squeeze().repeat(1, 1, 2)
        # print(f"bf: vec: {vec.shape}")
        output = self.lid_clf(vec)

        return output

    def get_pho_embeddings(self, x, seq_len, atten_mask=None, norm_pho=True):
        batch_size = x.size(0)
        T_len = x.size(1)

        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)
        x = x.reshape(batch_size, T_len, x.shape[-2], x.shape[-1])
        x_mean = torch.mean(x, dim=-1)

        if norm_pho == True:
            x_mean = torch.nn.functional.normalize(x_mean, dim=-1)

        return x_mean


class LD_classifier(nn.Module):
    def __init__(self, in_dim:int, kernel_size:int, lang_lab=0):
        super(LD_classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, int(in_dim/4)),
            nn.ReLU(),
            nn.Linear(int(in_dim/4), int(in_dim/16)),
            nn.ReLU(),
            nn.Linear(int(in_dim/16), 2))
        
        self.conv = nn.Conv1d(2, 2, tuple([kernel_size]),
                                 padding=tuple([int((kernel_size)/2)]),
                                 stride=1)
        
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.lang_lab = lang_lab

    def forward(self, x):
        output = self.linear(x)
        output = output.transpose(1, 2)
        output = self.conv(output)
        output = output.squeeze()
        output = self.softmax(output)
        return output
    
    def convert_lab(self, y, ignore_idx = 100):
        convert = lambda y: int(y == self.lang_lab) if not y == ignore_idx else int(y)
        new_labels = [[convert(lab) for lab in row] for row in y]
        return torch.tensor(new_labels)

    def convert_lab_from_lid(self, y, max_seq_len):
        convert = lambda y: int(y == self.lang_lab)
        new_labels = [[convert(lab)]*max_seq_len for lab in y]
        return torch.tensor(new_labels)

# class ld_e2e(nn.Module):
#     def __init__(self, pconv, clf0, clf1):
#         super(ld_e2e, self).__init__()
#         self.pconv = pconv
#         self.clf0 = clf0
#         self.clf1 = clf1

#     def forward(self, x, seq_len, atten_mask=None):
#         embd = self.pconv.get_embeddings(x, seq_len, atten_mask)
#         output0 = self.clf0(embd)
#         output1 = self.clf1(embd)
#         return output0, output1


