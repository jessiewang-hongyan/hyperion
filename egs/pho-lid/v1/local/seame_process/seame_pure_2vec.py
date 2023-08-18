import os
import glob
import torch
# import librosa
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
# import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from s3prl.nn import S3PRLUpstream
import torch.nn.functional as F
import torchaudio

# def upsampling_lre(audio, save_dir):
#     # if audio.endswith('.sph'):
#     #     data, sr = librosa.load(audio, sr=None)
#     #     new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.sph', '.wav')
#     #     sf.write(new_name, data, 8000, subtype='PCM_16')
#     #     subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)
#     # el
#     if audio.endswith('.wav') or audio.endswith('.WAV'):
#         new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.WAV', '.wav')
#         subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)

class preprocess():
    def __init__(self, lredir, label_encoder, model='xlsr_53', device=0, layer=16, 
                 seglen=10, overlap=1, savedir=None, audiodir=None):
        self.lredir = lredir
        self.seglen = seglen
        self.overlap = overlap
        self.savedir = savedir
        self.audiodir = audiodir
        self.layer = layer
        self.le = label_encoder

        self.device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
        self.model = S3PRLUpstream(model)
        self.model.to(device)
        self.model.eval()

        if not os.path.exists(savedir):
            os.mkdir(savedir)

    # step 0
    def make_wav2lang(self):
        audio_list = []
        labels = []
        key_file = self.lredir + '/data_label_list.txt'
        with open(key_file, 'r') as f:
            lines = f.readlines()
        audio_list = [self.lredir + '{}'.format(x.split()[0]) for x in lines]
        labels = [x.split()[1].strip().replace('-', '') for x in lines]

        audio2lang_txt = self.savedir + '/wav2lang.txt'
        # clear old version
        if os.path.exists(audio2lang_txt):
            os.remove(audio2lang_txt)

        with open(audio2lang_txt, 'a') as f:
            for i in tqdm(range(len(audio_list))):
                audio = audio_list[i]
                if labels[i].isnumeric():
                    f.write("{}\t{}\n".format(audio, labels[i]))
                else:
                    f.write("{}\t{}\n".format(audio, self.le.transform([labels[i]])[0]))

    # step 1
    def cut_wav_lab(self, lab_time=400):
        print('Segment long utterances to {} secs'.format(self.seglen))
        audio2lang_txt = self.savedir + '/wav2lang.txt'
        with open(audio2lang_txt, 'r') as f:
            lines = f.readlines()

        audio_list = []
        labels_list = []
        for x in lines:
            if len(x.split('\t')) > 1:
                audio_list.append(x.split(sep='\t')[0])
                labels_list.append(x.split(sep='\t')[1])

        audio2lang_seg_txt = self.savedir + '/segment2lang.txt'
        # clear old version
        if os.path.exists(audio2lang_seg_txt):
            os.remove(audio2lang_seg_txt)

        save_seg_dir = self.savedir
        if not os.path.exists(save_seg_dir):
            os.mkdir(save_seg_dir)

        with open(audio2lang_seg_txt, 'a') as f:
            for i in tqdm(range(len(audio_list))):
                audio = audio_list[i]
                main_name = os.path.split(audio)[-1]
                label = labels_list[i]
                
                # audio_length = librosa.get_duration(path=audio)
                waveform, sr = torchaudio.load(audio)
                audio_length = waveform.shape[-1] / sr
                data_ = AudioSegment.from_file(audio, "wav")
                num_segs = (audio_length - self.overlap) // (self.seglen - self.overlap)
                remainder = (audio_length - self.overlap) % (self.seglen - self.overlap)

                if num_segs >= 1:
                    start = 0
                    for ii in range(int(num_segs)):
                        end = start + self.seglen
                        start_ = start * 1000
                        end_ = end * 1000
                        data_seg = data_[start_:end_]
                        save_name = save_seg_dir + main_name.replace('.wav', '_{}.wav'.format(ii))
                        data_seg.export(save_name, format='wav')
                        start = end - self.overlap
                        f.write("{}\t{}\n".format(save_name, label))
                    if remainder >= 3: # if remainder longer than 3s, keep it, otherwise throw away
                        start_ = start * 1000
                        data_seg = data_[start_:]
                        save_name = save_seg_dir + main_name.replace('.wav', '_{}.wav'.format(num_segs))
                        data_seg.export(save_name, format='wav')

                        f.write("{}\t{}\n".format(save_name, label))
    # step 2
    def extract_wav2vec(self, pad_idx=100):
        print("Extracting wav2vec features from layer {} of pretrained {}".
              format(self.layer, type(self.model)))
        audio2lang_seg_txt = self.savedir + '/segment2lang.txt'
        feat2lang_txt = self.savedir + '/feat2lang.txt'
        # clear old version
        if os.path.exists(feat2lang_txt):
            os.remove(feat2lang_txt)

        with open(audio2lang_seg_txt, 'r') as f:
            lines = f.readlines()

        if len(lines) > 0:
            audio_list = []
            labels_list = []

            for x in lines:
                if len(x.split(sep='\t')) > 1:
                    audio_list.append(x.split(sep='\t')[0])
                    labels_list.append(x.split(sep='\t')[1])

            with open(feat2lang_txt, 'a') as f:
                for i in tqdm(range(len(audio_list))):
                    audio = audio_list[i]
                    label = labels_list[i]
                    # print(f'label length in step2: {len(label)}')
                    # data, sr = librosa.load(audio, sr=None)
                    data, sr = torchaudio.load(audio)
                    data_ = data.to(device=self.device, dtype=torch.float).reshape(-1, 1)
                    data_wav_len = torch.tensor([data_.shape[1]])
                    self.model.eval()
                    features = self.model(data_, wavs_len=data_wav_len)
                    # print(features)
                
                    features = features[0][self.layer].squeeze(0)
                    save_name = audio.replace(self.audiodir, self.savedir).replace('.wav', '.npy')

                    # print(save_name)

                    feat_shape = features.shape
                    if feat_shape[0]%20 == 0:
                        new_dim0 = int(feat_shape[0]/20)
                        np.save(save_name, features.cpu().detach().numpy().reshape((new_dim0, 20, feat_shape[1])))
                        # f.write(f"{save_name}\t{label.strip()}\t{new_dim0}\n").strip()
                    else:
                        new_dim0 = int(feat_shape[0]/20) + 1
                        padding_size = 20 - feat_shape[0]%20
                        features = F.pad(features, [0, 0, 1, padding_size - 1], "constant", 0)
                        
                        # print(f"padding size: {padding_size}, new_dim0: {new_dim0}, features size: {features.shape}")
                        
                        np.save(save_name, features.cpu().detach().numpy().reshape((new_dim0, 20, feat_shape[1])))
                        # f.write(f"{save_name}\t{label.strip()}\t{new_dim0}\n")

                    if len(label) < new_dim0:
                        label = label + [pad_idx]*(new_dim0 - len(label))

                    f.write(f"{save_name}\t{label.strip()}\t{new_dim0}\n")

                # f.write(f"{save_name}\t{label}\t{int(features.squeeze(0).shape[0]/20)}\n")

if __name__ == "__main__":
    lang_list = ['EN', 'ZH']
    le = LabelEncoder()
    le.fit(lang_list)

    for dir_name in os.listdir("data/seame/"):
        file_path = "/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/seame/" + dir_name

        if dir_name != 'merge' and 'data_label_list.txt' in os.listdir(file_path+'/pure/'):
            if os.path.exists(file_path+'/pure/feat2lang.txt'):
                need_2vec = True
                for fname in os.listdir(file_path+'/pure/'):
                    if fname.endswith('.npy'):
                        need_2vec = True
                if need_2vec:
                    preprocess_pipe = preprocess(
                        label_encoder=le,
                        seglen=10,
                        overlap=1,
                        lredir= file_path + '/pure/',
                        savedir= file_path + '/pure/',
                        audiodir= file_path + '/pure/',
                    )

                    # preprocess_pipe.make_wav2lang()
                    # preprocess_pipe.cut_wav_lab()
                    preprocess_pipe.extract_wav2vec()
                    print(file_path + ' Done.')
                else:
                    print(f'No changes to {file_path}')

    print('Pure seg2vec process done.')
    