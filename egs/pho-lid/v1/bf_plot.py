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
from significance import ScoringModel

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


def validation(valid_txt, model, model_name, device, kaldi, log_dir, num_lang, ignore_idx=100, verbose=False):
    valid_set = RawFeatures(valid_txt)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    model.eval()
    correct = 0
    total = 0
    scores = 0
    score_pos = []
    preds = 0
    truths = 0
    with torch.no_grad():
        for step, (utt, labels, seq_len) in enumerate(valid_data):
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device)

            # Forward pass
            embeddings = model.get_embeddings(utt, seq_len)
            embeddings = embeddings.reshape(-1, 1, embeddings.shape[-1])

            batch_size = utt.shape[0]
            frame_size = utt.shape[1]
            new_seq_len = [1]* (batch_size*frame_size)

            outputs = model.bf_check(embeddings, new_seq_len)
            predicted = torch.argmax(outputs, -1)

            labels = labels.squeeze()
            predicted = predicted.squeeze()
            outputs = outputs.squeeze()
            if verbose:
                print(f'outputs: {outputs}')
                print(f'predicted: {predicted}')
                print(f'labels: {labels}')
            # trancate to accuracy len
            correct_len = min(len(labels), len(predicted))
            labels = labels[:correct_len]
            predicted = predicted[:correct_len]

            if ignore_idx in labels:
                idx = labels.detach().cpu().numpy().tolist().index(ignore_idx)
                labels = labels[:idx]
                predicted = predicted[:idx]

            total += labels.size(-1)
            correct += (predicted == labels).sum().item()

            # labels = labels.flatten().cpu().tolist()
            # predicted = predicted.flatten().cpu().tolist()

            if ignore_idx in labels:
                padding_start = labels.index(ignore_idx)
                labels = labels[:padding_start]
                predicted = predicted[:padding_start]

            if step == 0:
                scores = [outputs.cpu()]
                score_pos.append(outputs[:,:1].cpu().numpy().tolist())
                preds = [predicted.cpu().numpy().astype(int).tolist()]
                truths = [labels.cpu().numpy().astype(int).tolist()]
            else:
                scores.append(outputs.cpu())
                score_pos.append(outputs[:,:1].cpu().numpy().tolist())
                preds.append(predicted.cpu().numpy().astype(int).tolist())
                truths.append(labels.cpu().numpy().astype(int).tolist())
            # if step > 10:
            #     break

    # print(f"preds: {preds}\ntruths: {truths}")

    acc = correct / total
    print('Current Acc.: {:.4f} %'.format(100 * acc))
    # scores = scores.squeeze().cpu().numpy()
    # print(scores.shape)
    trial_txt = log_dir + '/trial_{}.txt'.format(model_name)
    score_txt = log_dir + '/score_{}.txt'.format(model_name)
    output_txt = log_dir + '/output_{}.txt'.format(model_name)
    sld.get_trials(valid_txt, num_lang, trial_txt)
    sld.get_score(valid_txt, scores, num_lang, score_txt)
    eer_txt = trial_txt.replace('trial', 'eer')
    subprocess.call(f"{kaldi}/egs/ywspeech/sre/subtools/computeEER.sh "
                    f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)
    cavg = sld.compute_cavg(trial_txt, score_txt)
    wacc,accs, weights = sld.compute_wacc(preds, truths, num_lang)
    bacc = sld.get_bacc(truths, preds)

    # truths = truths.flatten()
    # draw_roc(truths, score_pos, fname=model_name.replace('.ckpt', '.png'))
    print("Cavg:{}".format(cavg))
    print(f"Balanced Acc:{bacc}, Weighted Acc: {wacc}, Acc by class:{accs}, class weights:{weights}")
    with open(output_txt, 'a') as f:
        f.write(f'Valid set: {valid_txt}:\n')
        f.write("ACC:{}% Cavg:{} BACC: {} WACC:{} ACC_class:{} Weight_class {}\n".format(round(acc*100, 4), cavg, bacc, wacc, accs, weights))
    return cavg


def plot(x, label, name):
    # print(x.shape)
    # print(label.shape)
    # print(x[:, 0].shape)
    # print(x[:, 1].shape)

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x[:, 0], y=x[:, 1],
        hue=label,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.3
    )
    plt.savefig(name)

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

    model.eval()
    
    plot_path = "./plots/ppho_seame_bf_1e-5tune_seame_640/"
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # brute force predict each vector in embeddings
    for epoch in tqdm(range(total_epochs)):
        # model_name = "./models/sig/ppho_seame_bf_0205_clf_reg0_sigonseame/ppho_seame_bf_0205_clf_reg0_sigonseame_reg0_lr1e-05_epoch_{}.ckpt".format(str(epoch))        
        model_name = "./models/ppho_seame_bf_1e-5tune/ppho_seame_bf_1e-5tune_epoch_{}.ckpt".format(str(epoch))
        model.load_state_dict(torch.load(model_name))
        model.eval()

        output_samples = []
        h_cat_samples = []
        label_samples = []

        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)

            batch_size = utt.shape[0]
            frame_size = utt.shape[1]
            new_seq_len = [1]* (batch_size*frame_size)

            if labels.shape[-1] > 25:
                labels = labels[:, :25]

            outputs, _ = model(utt_, new_seq_len, atten_mask=atten_mask)
            h_cat = model.get_embeddings(utt_, new_seq_len, atten_mask=atten_mask)
           
            labels = labels.reshape(batch_size*frame_size)
            outputs = outputs.reshape(-1, outputs.shape[-1]).detach().cpu().numpy()
            h_cat = h_cat.reshape(-1, h_cat.shape[-1]).detach().cpu().numpy()

            # remove paddings
            for o, hc, lab in zip(outputs, h_cat, labels):
                if lab != 100:
                    output_samples.append(o)
                    h_cat_samples.append(hc)
                    label_samples.append(lab)
            if step > 5*128:
                break

        output_samples = np.array(output_samples)
        h_cat_samples = np.array(h_cat_samples)
        label_samples = np.array(label_samples)

        # pca
        pca = PCA(n_components=20)
        h_cat_pca = pca.fit_transform(h_cat_samples)

        # t-SNE
        tsne = TSNE(n_components=2, learning_rate='auto', perplexity=40, n_iter=300)
        h_cat_tsne = tsne.fit_transform(h_cat_pca)

        plot_path_e = plot_path + "/epoch{}".format(epoch)
        if not os.path.exists(plot_path_e):
            os.mkdir(plot_path_e)


        plot(output_samples, label_samples, plot_path_e + "/{}.png".format("outputs"))
        plot(h_cat_tsne, label_samples, plot_path_e + "/{}.png".format("h_cat"))

if __name__ == "__main__":
    main()
