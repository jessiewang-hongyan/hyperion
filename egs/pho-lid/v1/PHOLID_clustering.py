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
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime
import torch.nn.functional as F

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

def validation(valid_txt, model, model_name, device, kaldi, log_dir, num_lang, lang_vecs, ignore_idx=100, verbose=False):
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
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

            score_0 = F.cosine_similarity(embeddings, lang_vecs[0, :], dim=0)
            score_1 = F.cosine_similarity(embeddings, lang_vecs[1, :], dim=0)
            # print(f'embeddings shape: {embeddings.shape} lang_vec shape: {lang_vecs.shape} score_0: {score_0.shape}')

            
            outputs = torch.stack((score_0, score_1))
            outputs = torch.transpose(outputs, 0, 1)
            # print(outputs.shape)

            predicted = torch.argmax(outputs, -1)
            # print(f'embeddings shape: {embeddings.shape} outputs shape: {outputs.shape} lang_vec shape: {lang_vecs.shape} predicted shape: {predicted.shape}')

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


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    # parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='xsa_config.json')
    args = parser.parse_args()
    with open(args.json, 'r') as json_obj:
        config_proj = json.load(json_obj)
    seed = config_proj["optim_config"]["seed"]
    if seed == -1:
        pass
    else:
        print("Random seed is {}".format(seed))
        setup_seed(seed)
    # device = torch.device('cuda:{}'.format(config_proj["optim_config"]["device"])
    #                       if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:{}'.format(0)
                          if torch.cuda.is_available() else 'cpu')
    print(f'device:{device}')
    feat_dim = config_proj["model_config"]["d_k"]
    n_heads = config_proj["model_config"]["n_heads"]
    model = PHOLID_conv(input_dim=config_proj["model_config"]["feat_dim"],
                   feat_dim=config_proj["model_config"]["d_k"],
                   d_k=config_proj["model_config"]["d_k"],
                   d_v=config_proj["model_config"]["d_k"],
                   d_ff=config_proj["model_config"]["d_ff"],
                   n_heads=config_proj["model_config"]["n_heads"],
                   dropout=0.1,
                   n_lang=config_proj["model_config"]["n_language"],
                   max_seq_len=10000)

    if config_proj["Input"]["load_path"] is not None:
        model.load_state_dict(torch.load(config_proj["Input"]["load_path"]))
    # model.load_state_dict(torch.load('./models/pconv_seame_bf/pconv_seame_bf_epoch_4.ckpt'))
    model.to(device)
    model_name = config_proj["model_name"]
    print("model name: {}".format(model_name))
    log_dir = config_proj["Input"]["userroot"] + config_proj["Input"]["log"]
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

    ahc = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
    model.eval()

    total = 0
    correct = 0
    conf_matrix = np.zeros((2, 2))

    # calculate lang_vectors
    batch_sizes = []
    centroids = [None, None]

    for step, (utt, labels, seq_len) in enumerate(train_data):
        utt_ = utt.to(device=device)
        atten_mask = get_atten_mask(seq_len, utt_.size(0))
        atten_mask = atten_mask.to(device=device)

        embeddings = model.get_embeddings(utt_, seq_len, atten_mask)
        # print(f'embeddings: {embeddings.shape}')
        # embeddings = embeddings.reshape(-1, embeddings.shape[-1]).cpu().numpy()

        features = None
        valid_labels = None

        labels = labels.detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        ignore_idx = 100
        
        # remove attention paddings
        for lab, slen, emb in zip(labels, seq_len, embeddings):
            feats = emb[:slen, :]

            # remove label paddings
            if ignore_idx in lab:
                valid_label_len = lab.tolist().index(ignore_idx)
                feats = feats[:valid_label_len, :]
                lab = lab[:valid_label_len]

            min_len = min(feats.shape[0], lab.shape[0])
            feats = feats[:min_len]
            lab = lab[:min_len]

            if features is None:
                features = feats
                valid_labels = lab
            elif len(lab) > 0:
                features = np.concatenate((features, feats), axis=0)
                valid_labels = np.concatenate((valid_labels, lab), axis=0)


        if len(valid_labels) > 0:
            clustering = ahc.fit_predict(features)

            total += len(valid_labels)
            # handle permutation
            correct1 = sum(valid_labels.astype(int)==clustering)
            correct2 = len(valid_labels) - correct1

            # clustering label matches truth label
            if correct1 > correct2:
                correct += correct1
            # clustering label is the inverse of truth label
            else:
                correct += correct2
                clustering = 1 - clustering

            # save centroids
            class_0_mask = (np.array(clustering) == 0)
            class_1_mask = (np.array(clustering) == 1)

            centroid_0 = np.mean(features[class_0_mask], axis=0)
            centroid_1 = np.mean(features[class_1_mask], axis=0)

            if centroids[0] is None:
                centroids[0] = centroid_0
                centroids[1] = centroid_1
            else:
                print(f'centroids[0]: {centroids[0].shape}, controid_0: {centroid_0.shape}')
                centroids[0] = np.vstack((centroids[0], centroid_0))
                centroids[1] = np.vstack((centroids[1], centroid_1))

            batch_sizes.append(len(valid_labels))
            
            conf_matrix = conf_matrix + confusion_matrix(valid_labels.astype(int), clustering)


    accuracy = correct/total
    print(f'total amount of data: {total}. AHC training clustering acc: {accuracy}.')
    print(f'training confusion matrix: \nrow: Truth\n{conf_matrix}')

    # calculate whole-batch vectors
    lang_vecs = np.zeros((2, centroids[0].shape[-1]))
    weights = np.array(batch_sizes) / total

    for idx, cen in enumerate(centroids):
        print(f'cen.shape: {cen.shape}, weights: {weights.shape}')
        lang_vecs[idx, :] = np.average(cen, axis=0, weights=weights)

    # save centroids
    save_dir = './models/clustering/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y%m%d%H%M%S")
    new_folder = save_dir + model_name + '_' + current_time_str + '/'
    os.mkdir(new_folder)
    with open(new_folder + 'param.txt', 'w') as f:
        f.write(f'Train set: {train_txt}')
    
    filepath = new_folder + f"centroids.npy"
    np.save(filepath, lang_vecs)

    lang_vecs = torch.from_numpy(lang_vecs).to(device)

    # validation
    if valid_txt is not None:
        print(f'Val set: {valid_txt}')
        validation(valid_txt, model, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
                    num_lang=config_proj["model_config"]["n_language"], lang_vecs=lang_vecs)
    if test_txt is not None:
        print(f'Test set: {test_txt}')
        validation(test_txt, model, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
                    num_lang=config_proj["model_config"]["n_language"], lang_vecs=lang_vecs)



if __name__ == "__main__":
    main()
