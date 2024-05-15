from der import DER
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
    der = DER()
  
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

            if correct_len < 1:
                continue

            if ignore_idx in labels:
                padding_start = labels.detach().cpu().numpy().tolist().index(ignore_idx)
                labels = labels[:padding_start]
                predicted = predicted[:padding_start]

            if len(labels)< 1:
                continue


            der.get_DER(labels, predicted)

            # if step > 10:
            #     break

    global_value, mean, lower, upper = der.get_global_DER()
    print(f"DER evaluation: global: {round(global_value, 4)}, mean: {round(mean, 4)}, lower: {round(lower, 4)}, upper: {round(upper)}")

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
    model = PHOLID_pho(input_dim=config_proj["model_config"]["feat_dim"],
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
        
    if valid_txt is not None:
        print('On val set:')
        validation(valid_txt, model, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
                    num_lang=config_proj["model_config"]["n_language"])
    if test_txt is not None:
        print('On test set:')
        validation(test_txt, model, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
                    num_lang=config_proj["model_config"]["n_language"])


if __name__ == "__main__":
    main()
