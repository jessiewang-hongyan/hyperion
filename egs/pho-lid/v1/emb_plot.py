import json
import scoring
import subprocess
from ssl_sampler import *
from model import *
from data_load import *
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from scoring_ld import draw_roc
import scoring_ld as sld

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    # parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='xsa_config.json')
    # parser.add_argument('--gpu', type=str, default="0")
    args = parser.parse_args()
    with open(args.json, 'r') as json_obj:
        config_proj = json.load(json_obj)
    seed = config_proj["optim_config"]["seed"]
    if seed == -1:
        pass
    else:
        print("Random seed is {}".format(seed))
        setup_seed(seed)
    device = torch.device('cuda:{}'.format(config_proj["optim_config"]["device"])
                          if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:{}'.format(args.gpu)
    #                       if torch.cuda.is_available() else 'cpu')
    print(f'device:{device}')
    feat_dim = config_proj["model_config"]["d_k"]
    n_heads = config_proj["model_config"]["n_heads"]
    model = PHOLID_conv_pho(input_dim=config_proj["model_config"]["feat_dim"],
                   feat_dim=config_proj["model_config"]["d_k"],
                   d_k=config_proj["model_config"]["d_k"],
                   d_v=config_proj["model_config"]["d_k"],
                   d_ff=config_proj["model_config"]["d_ff"],
                   n_heads=config_proj["model_config"]["n_heads"],
                   dropout=0.1,
                   n_lang=config_proj["model_config"]["n_language"],
                   max_seq_len=10000)

    if config_proj["Input"]["load_path"] is not None and not config_proj["Input"]["load_path"] == "":
        model.load_state_dict(torch.load(config_proj["Input"]["load_path"]))
    model.to(device)
    model_name = config_proj["model_name"]
    print("model name: {}".format(model_name))
    log_dir = config_proj["Input"]["userroot"] + config_proj["Input"]["log"]
    # kaldi_root = config_proj["Input"]["userroot"] + config_proj["kaldi"]
    kaldi_root = config_proj["kaldi"]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    train_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["train"]
    train_set = RawFeatures(train_txt)
    train_data = DataLoader(dataset=train_set,
                            batch_size=config_proj["optim_config"]["batch"],
                            pin_memory=True,
                            num_workers=config_proj["optim_config"]["num_work"],
                            shuffle=True,
                            collate_fn=collate_fn_atten)

    if config_proj["Input"]["valid"] != "none":
        print("Validation is True")
        valid_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["valid"]
    else:
        valid_txt = None
    if config_proj["Input"]["test"] != "none":
        print("Test is True")
        test_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["test"]
    else:
        test_txt = None

    loss_func_lid = nn.CrossEntropyLoss(ignore_index=100).to(device)
    num_nega_samples = config_proj["optim_config"]["nega_frames"]
    print("Compute phoneme SSL over segments with {} negative samples".format(num_nega_samples))
    # loss_func_phn = Phoneme_SSL_loss(num_frames=20, num_sample=num_nega_samples).to(device)
    total_step = len(train_data)
    total_epochs = config_proj["optim_config"]["epochs"]
    valid_epochs = config_proj["optim_config"]["valid_epochs"]
    weight_lid = config_proj["optim_config"]["weight_lid"]
    weight_ssl = config_proj["optim_config"]["weight_ssl"]
    print("weights: LID {} SSL {}".format(weight_lid, weight_ssl))
    optimizer = torch.optim.Adam(model.parameters(), lr=config_proj["optim_config"]["learning_rate"])
    SSL_epochs = config_proj["optim_config"]["SSL_epochs"]
    SSL_steps = SSL_epochs * total_step

    # save ckpt in a folder
    model_save_path = './models/'+model_name
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    labels_new = []
    embeddings_new = []

    for step, (utt, labels, seq_len) in enumerate(train_data):
        utt_ = utt.to(device=device)
        atten_mask = get_atten_mask(seq_len, utt_.size(0))
        atten_mask = atten_mask.to(device=device)

        batch_size = utt.shape[0]
        frame_size = utt.shape[1]
        new_seq_len = [1]* (batch_size*frame_size)

        labels = labels.type(torch.LongTensor) 
        if labels.shape[-1] > 25:
            labels = labels[:, :25]
        labels = labels.reshape(batch_size*frame_size)

        # get embeddings
        embeddings = model.get_embeddings(utt_, seq_len, atten_mask, norm_pho=True, norm_tac=True)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1]).detach().cpu().numpy()

        embeddings_no_norm = model.get_embeddings(utt_, seq_len, atten_mask, norm_pho=False, norm_tac=False)
        embeddings_no_norm = embeddings_no_norm.reshape(-1, embeddings.shape[-1]).detach().cpu().numpy()

        # remove paddings
        for emb, emb_no, lab in zip(embeddings, embeddings_no_norm, labels):
            if lab != 100:
                embeddings_new.append(emb)
                embeddings_no_norm_new.append(emb_no)
                labels_new.append(lab)
                
        if step > 5:
            break
        
    labels_new = np.array(labels_new)
    print(f'labels_new.shape: {labels_new.shape}')

    # pca
    pca = PCA(n_components=20)
    emb_pca = pca.fit_transform(embeddings_new)
    emb_pca_no_norm = pca.fit_transform(embeddings_new)
    print(f'pca emb.shape: {emb_pca.shape}')

    # t-SNE
    time_start = time.time()
    tsne = TSNE(n_components=2, learning_rate='auto', perplexity=40, n_iter=300)
    emb_tsne = tsne.fit_transform(emb_pca)
    emb_tsne_no_norm = tsne.fit_transform(emb_pca_no_norm)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    print(f'tsne emb.shape: {emb_tsne.shape}')

    emb_dim1 = emb_tsne[:, 0]
    emb_dim2 = emb_tsne[:, 1]

    emb_no_norm_dim1 = emb_tsne_no_norm[:, 0]
    emb_no_norm_dim2 = emb_tsne_no_norm[:, 1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=emb_dim1, y=emb_dim2,
        hue=labels_new,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.3
    )
    plt.savefig("./emb_with_norm_seame.png")

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=emb_no_norm_dim1, y=emb_no_norm_dim2,
        hue=labels_new,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.3
    )
    plt.savefig("./emb_without_norm_seame.png")
    
if __name__ == "__main__":
    main()
