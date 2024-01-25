import sys
sys.path.append('./inference/')
from inference import preprocess
from preprocess import Preprocess
from model import PHOLID_conv_pho, PHOLID_conv
import torch
import numpy as np


if __name__ == "__main__":
    # my_model = PHOLID_conv_pho(input_dim=1024, feat_dim=64, d_k=64, d_v=64, d_ff=2048, n_heads=8, n_lang=2)
    # my_model.load_state_dict(torch.load("./models/pholid_seame_bf_pho/pholid_seame_bf_pho_epoch_4.ckpt"))
    my_model = PHOLID_conv(input_dim=1024, feat_dim=64, d_k=64, d_v=64, d_ff=2048, n_heads=8, n_lang=2)
    my_model.load_state_dict(torch.load("models/pconv_seame_bf2/pconv_seame_bf2_epoch_4.ckpt"))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    test_file =  "./local/seame_split/cat/feat2lang_test_final.txt"
    export_file = "./inference/outputs/seame_pred_conv.rttm"

    lang_map = ["English", "Mandarin"]

    my_model = my_model.to(device)
    my_model.eval()

    with open(test_file, "r") as f:
        lines = f.readlines()

    for i, row in enumerate(lines):
        audio, labels, seq_len = row.strip().split("\t")
        audio_name = audio.replace(".npy", "")
        labels = labels.replace("[", "").replace("]", "").split(", ")
        labels = [int(float(x)) for x in labels]
        
        start = 0
        duration = 0.4 * int(seq_len)
        
        feat = torch.tensor(np.load(audio)).unsqueeze(0).to(device)
        seq_len = [int(seq_len)]
        embeddings = my_model.get_embeddings(feat, seq_len)
        embeddings = embeddings.reshape(-1, 1, embeddings.shape[-1])

        batch_size = feat.shape[0]
        frame_size = feat.shape[1]
        new_seq_len = [1]* (batch_size*frame_size)

        outputs = my_model.bf_check(embeddings, new_seq_len)
        predicted = torch.argmax(outputs, -1).squeeze()

        # print(f"feat.shape: {feat.shape}, seq_len: {seq_len}")
        # print(f"original_file: {original_file}, start: {start}, duration: {duration}, result: {outputs}, pred: {predicted}")
        
        with open(export_file, "a") as f:
            if seq_len[0] > 1:
                s = start
                lapse = duration / seq_len[0]
                for pred in predicted:
                    lang = lang_map[pred]
                    fields = ['SPEAKER', audio_name, '0', str(round(s, 4)), str(round(lapse, 4)), '<NA>', '<NA>', lang, '<NA>', '<NA>']
                    line = ' '.join(fields) + "\n"
                    f.write(line)
                    s += lapse
            else:
                lang = lang_map[predicted.tolist()]
                fields = ['SPEAKER', audio_name, '0', str(round(start, 4)), str(round(duration, 4)), '<NA>', '<NA>', lang, '<NA>', '<NA>']
                line = ' '.join(fields) + "\n"
                f.write(line)
