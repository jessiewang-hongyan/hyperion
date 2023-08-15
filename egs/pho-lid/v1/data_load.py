import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from torchvision.transforms import Lambda

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    # label_stack = []
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0)
    # return data, torch.tensor(label_stack), seq_length
    return data, label, seq_length


def collate_fn_atten(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, labels, seq_length = zip(*batch)
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    if not isinstance(labels[0], int):
        # for LD
        labels = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=100)
    else:
        # for LID
        labels = torch.LongTensor(labels)
    # with open('./data_collate_log.txt', 'a') as f:
    #     f.write("data shape: {}\t labels shape: {} \t seq_len:{} \t seq_len sum: {}\n".format(data.shape, label_padded.shape, seq_length, sum(seq_length)))

    return data, labels, seq_length




# class RawFeatures(data.Dataset):
#     def __init__(self, txt_path):
#         with open(txt_path, 'r') as f:
#             lines = f.readlines()
#             self.feature_list = [i.split()[0] for i in lines]
#             self.label_list = [i.split()[1] for i in lines]
#             self.seq_len_list = [i.split()[2].strip() for i in lines]

#     def __getitem__(self, index):
#         feature_path = self.feature_list[index]
#         # feature = torch.from_numpy(np.load(feature_path, allow_pickle=True))
#         feature = torch.tensor(np.load(feature_path, allow_pickle=True).tolist())
#         label = int(self.label_list[index])
#         seq_len = int(self.seq_len_list[index])
#         return feature, label, seq_len

#     def __len__(self):
#         return len(self.label_list)

class RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split('\t')[0] for i in lines]
            self.label_list = [i.split('\t')[1] for i in lines]
            self.seq_len_list = [i.split('\t')[2].strip() for i in lines]
            self.is_seq_lab = False

        # for seame data
        if '[' in self.label_list[0]:
            self.is_seq_lab = True
        
        if self.is_seq_lab:
            label_list = []

            label_text = [x.strip().replace('[', '').replace(']', '') for x in self.label_list]
            label_text = [x.split() for x in label_text]

            for row in label_text:
                label_list.append(torch.tensor([torch.tensor(float(x.replace(',', ''))) for x in row]))

            self.label_list = label_list



    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        feature = torch.tensor(np.load(feature_path, allow_pickle=True).tolist())
        if self.is_seq_lab:
            label = torch.FloatTensor(self.label_list[index])
        else:
            label = int(self.label_list[index])
            
        seq_len = int(self.seq_len_list[index])
        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)


def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()

def get_atten_mask_frame(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len*20, max_len*20])
    for i in range(batch_size):
        length = seq_lens[i]*20
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()


def std_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = torch.tensor(seq_lens) / (torch.tensor(seq_lens) - 1)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, length:, :] = 1e-9
    return atten_mask, weight_unbaised

def mean_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = seq_lens[0] / torch.tensor(seq_lens)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, :length, :] = 0
    return atten_mask.bool(), weight_unbaised


