import json
import scoring_ld as scoring
import subprocess
from ssl_sampler import *
from model import *
from data_load import *
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def validation_clf(valid_txt, pho_model, clf, model_name, device, kaldi, log_dir, num_lang, ignore_idx=100, clf_idx=0):
    valid_set = RawFeatures(valid_txt)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    pho_model.eval()
    clf.eval()

    correct = 0
    total = 0
    scores = []
    confidences = []
    all_labels = []
    truths = []
    preds = []

    with torch.no_grad():
        for step, (utt, labels, seq_len) in enumerate(valid_data):
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            # Forward pass
            embeddings = pho_model.get_embeddings(utt, seq_len)
            # print(f'embeddings shape: {embeddings.shape}, label size: {labels.shape}')
            outputs = clf(embeddings)
            predicted = torch.argmax(outputs, 0)

            # transform labels
            if isinstance(labels[0], int):
                transformed_label = clf.convert_lab_from_lid(labels, max_seq_len=max(seq_len))
            else:
                transformed_label = clf.convert_lab(labels, ignore_idx=100)

            correct_len = embeddings.shape[1]
            transformed_label = transformed_label[0][:correct_len]
            predicted = predicted[:correct_len]
            # print(f'truth: {transformed_label.shape}: {transformed_label}')
            # print(f'preds: {predicted.shape}: {predicted}')
            total += transformed_label.size(-1) - (transformed_label == ignore_idx).sum().item()

            correct += (predicted.cpu() == transformed_label.cpu()).sum().item()

            transformed_label = transformed_label.flatten().cpu().tolist()
            predicted = predicted.flatten().cpu().tolist()

            # if contain paddings, remove it
            if ignore_idx in transformed_label:
                padding_start = transformed_label.index(ignore_idx)
                transformed_label = transformed_label[:padding_start]
                predicted = predicted[:padding_start]

            truths = truths + transformed_label
            preds = preds + predicted[:len(transformed_label)]
            scores.append(outputs.cpu().tolist())
            all_labels = all_labels + transformed_label
         
            confidences = confidences + scores[0][1][:len(transformed_label)]
            # print(f'labels:{transformed_label}\npred: {predicted}')

            if step == 0:
                _,accs, weights = scoring.compute_wacc(predicted, transformed_label, num_lang)

            else:
                _,new_accs, new_weights = scoring.compute_wacc(predicted, transformed_label, num_lang)

                for i in range(num_lang):
                    accs[i] += new_accs[i]
                    weights[i] += new_weights[i]

    wacc, accs, weights = scoring.compute_wacc(preds, truths, num_lang)
    wacc = round(wacc, 4)

    bacc = scoring.get_bacc(truths, preds)

    acc = correct / total
    print('Current Acc.: {:.4f} %'.format(100 * acc))

    trial_txt = log_dir + '/trial_{}.txt'.format(model_name)
    score_txt = log_dir + '/score_{}.txt'.format(model_name)
    output_txt = log_dir + '/output_{}.txt'.format(model_name)
    scoring.get_trials(valid_txt, num_lang, trial_txt)
    scoring.get_score(valid_txt, scores, num_lang, score_txt)
    eer_txt = trial_txt.replace('trial', 'eer')
    subprocess.call(f"{kaldi}/egs/ywspeech/sre/subtools/computeEER.sh "
                    f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)

    cavg = scoring.compute_cavg(trial_txt, score_txt)
    
    scoring.draw_roc(all_labels, confidences, fname='clf{}'.format(clf_idx))

    print("Cavg:{}".format(cavg))
    print(f"Balanced Acc:{bacc}, Weighted Acc: {wacc}, Acc by class:{accs}, class weights:{weights}")
    with open(output_txt, 'a') as f:
        f.write(f'Valid set: {valid_txt}:\n')
        f.write("ACC:{} Cavg:{} BACC: {} WACC:{} ACC_class:{} Weight_class".format(acc, cavg, bacc, wacc, accs, weights))
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
    
    model = PHOLID_conv(input_dim=config_proj["model_config"]["feat_dim"],
                   feat_dim=config_proj["model_config"]["d_k"],
                   d_k=config_proj["model_config"]["d_k"],
                   d_v=config_proj["model_config"]["d_k"],
                   d_ff=config_proj["model_config"]["d_ff"],
                   n_heads=config_proj["model_config"]["n_heads"],
                   dropout=0.1,
                   n_lang=config_proj["model_config"]["n_language"],
                   max_seq_len=10000)
    model.load_state_dict(torch.load(config_proj["clf_config"]["model_load_path"], map_location=device))
    model.to(device)
    # model.eval()

    # add classifiers
    clfs = []
    for i in range(2):
        clf = LD_classifier(in_dim=model.d_model, kernel_size=5, lang_lab=i)
        # clf.load_state_dict(torch.load(f'./models/clfs/pconv_seame/train_seame/clf{i}/clf{i}_epoch_31.ckpt', map_location=device))

        clfs.append(clf)
        clf.to(device)

    model_name = config_proj["model_name"]
    print("model name: {}".format(model_name))
    log_dir = config_proj["Input"]["userroot"] + config_proj["Input"]["log"]
    kaldi_root = config_proj["kaldi"]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    train_txt = config_proj["Input"]["userroot"] + config_proj["clf_config"]["train"]
    train_set = RawFeatures(train_txt)
    train_data = DataLoader(dataset=train_set,
                            batch_size=config_proj["optim_config"]["batch"],
                            pin_memory=True,
                            num_workers=config_proj["optim_config"]["num_work"],
                            shuffle=True,
                            collate_fn=collate_fn_atten)

    if config_proj["Input"]["valid"] != "none":
        print("Validation is True")
        valid_txt = config_proj["Input"]["userroot"] + config_proj["clf_config"]["valid"]
    else:
        valid_txt = None
    if config_proj["Input"]["test"] != "none":
        print("Test is True")
        test_txt = config_proj["Input"]["userroot"] + config_proj["clf_config"]["test"]
    else:
        test_txt = None

    # loss_func_lid = nn.CrossEntropyLoss().to(device)
    # num_nega_samples = config_proj["optim_config"]["nega_frames"]
    # print("Compute phoneme SSL over segments with {} negative samples".format(num_nega_samples))
    # loss_func_phn = Phoneme_SSL_loss(num_frames=20, num_sample=num_nega_samples).to(device)
    total_step = len(train_data)
    total_epochs = config_proj["clf_config"]["epochs"]
    valid_epochs = config_proj["optim_config"]["valid_epochs"]
    # weight_lid = config_proj["optim_config"]["weight_lid"]
    # weight_ssl = config_proj["optim_config"]["weight_ssl"]
    # print("weights: LID {} SSL {}".format(weight_lid, weight_ssl))
    # optimizer = torch.optim.Adam(model.parameters(), lr=config_proj["clf_config"]["learning_rate"])
    # SSL_epochs = config_proj["optim_config"]["SSL_epochs"]
    # SSL_steps = SSL_epochs * total_step
    # if config_proj["optim_config"]["warmup_step"] == -1:
    #     warmup = total_step * 3
    # else:
    #     warmup = config_proj["optim_config"]["warmup_step"]
    # if config_proj["optim_config"]["warmup_step"] == -1:
    #     warmup = total_step * 3
    # else:
    #     warmup = config_proj["optim_config"]["warmup_step"]
    # warm_up_with_cosine_lr = lambda step: 1 if step <= SSL_steps else (
    #     (step - SSL_steps) / warmup if step < SSL_steps + warmup else 0.5 * (
    #             math.cos((step - SSL_steps - warmup) / (total_epochs * total_step - SSL_steps - warmup) * math.pi) + 1))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    # train clfs
    print(f'Learning rate: {config_proj["clf_config"]["learning_rate"]}')
    e2e_model = ld_e2e(model, clfs[0], clfs[1])
    optimizer = torch.optim.Adam(e2e_model.parameters(), lr=config_proj["clf_config"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, eps=1e-12)
    save_path = './models/e2e/seame/'

    for epoch in tqdm(range(total_epochs)):
        loss_func_clf0 = nn.CrossEntropyLoss(ignore_index=100).to(device)
        loss_func_clf1 = nn.CrossEntropyLoss(ignore_index=100).to(device)
        loss_sum = 0
        loss_sum0 = 0
        loss_sum1 = 0

        for step, (utt, labels, seq_len) in enumerate(train_data):
            optimizer.zero_grad()

            # print(f'labels size: {labels.shape}')

            utt_ = utt.to(device=device)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)

            transformed_labels = []

            if isinstance(labels[0], int):
                transformed_labels.append(e2e_model.clf0.convert_lab_from_lid(labels, max_seq_len=max(seq_len)).to(device=device))
                transformed_labels.append(e2e_model.clf1.convert_lab_from_lid(labels, max_seq_len=max(seq_len)).to(device=device))
            else:
                transformed_labels.append(e2e_model.clf0.convert_lab(labels, ignore_idx=100).to(device=device))
                transformed_labels.append(e2e_model.clf1.convert_lab(labels, ignore_idx=100).to(device=device))

            # print(f'transformed_labels[0]: {transformed_labels[0].shape}')
            output0, output1 = e2e_model(utt_, seq_len, atten_mask)

            weight0 = 0.5
            loss_clf0 = loss_func_clf0(output0, transformed_labels[0])
            loss_clf1 = loss_func_clf1(output1, transformed_labels[1])
            total_loss = weight0 * loss_clf0 + (1 - weight0) * loss_clf1

            total_loss.backward()
            optimizer.step()
            loss_sum = loss_sum + total_loss
            loss_sum0 = loss_sum0 + loss_clf0
            loss_sum1 = loss_sum1 + loss_clf1
           
        scheduler.step(loss_sum)
        print("Epoch [{}/{}], total Loss: {:.4f}, loss0: {:.4f}, loss1: {:.4f}".
            format(epoch + 1, total_epochs, loss_sum.item(), loss_sum0.item(), loss_sum1.item()))
        
        torch.save(e2e_model.state_dict(), '{}_epoch_{}.ckpt'.format(save_path+'/e2e', epoch))
            
            # if epoch >= total_epochs - valid_epochs - 1:
            #     for clf in clfs:
            #         if valid_txt is not None:
            #             validation_clf(valid_txt, model, clf, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
            #                     num_lang=config_proj["model_config"]["n_language"])
            #         if test_txt is not None:
            #             validation_clf(test_txt, model, clf, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
            #                     num_lang=config_proj["model_config"]["n_language"])
    clfs = []
    model = e2e_model.pconv
    clfs.append(e2e_model.clf0)
    clfs.append(e2e_model.clf1)
    for idx, clf in enumerate(clfs):
        if valid_txt is not None:
            validation_clf(valid_txt, model, clf, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
                    num_lang=config_proj["model_config"]["n_language"], clf_idx=idx)

if __name__ == "__main__":
    main()
