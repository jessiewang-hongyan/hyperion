import os
import math
import numpy as np
import csv
import torch
import torchaudio
import librosa
# import soundfile as sf
import subprocess
import random
from sklearn.preprocessing import LabelEncoder

class cs_concat():
    def __init__(self, dir_name, langs=['English', 'Mandarin'], audio_dir='/seg/', data_label_list='/data_label_list.txt', save_dir='/cat/'):
        self.data_label_list = data_label_list
        self.dir = dir_name
        self.audio_dir = audio_dir
        self.save_dir = save_dir
        self.le = LabelEncoder()
        self.langs = self.le.fit_transform(langs)

        # self.langs = self.le.transform(langs)
        self.lists_by_rec = self.get_lists_by_rec()


    def get_lists_by_rec(self):
        lists = dict()

        with open(self.dir+self.data_label_list, 'r') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            
            for row in csv_reader:
                (audio, lang, length) = tuple(row)
                if lang.isnumeric():
                    lang = int(lang)
                elif lang not in ['English', 'Mandarin']:
                    continue
                else:
                    lang = self.le.transform([lang])[0]

                parts = audio.split('_')
                rec_name = "_".join(parts[:-1]) + '.wav'
                # print(parts)
                # print(rec_name)
                utt_code = int(parts[-1].replace('.wav', '').replace('a', ''))
                audio = self.dir + self.audio_dir + audio

                if not rec_name in lists.keys():
                    lists[rec_name] = []
                lists[rec_name].append((utt_code, audio, lang, length))
        return lists
    
    def gen_lab_by_time(self, pure_audio_len, label, lapse=400):
        # print(f"pure audio length: {pure_audio_len}, multiplier: {math.floor(pure_audio_len/lapse)}")
        num_labels = math.floor(pure_audio_len/lapse)
        # if pure_audio_len % lapse > lapse /2:
        #     num_labels = num_labels + 1
        result = [label]*int(num_labels)
        return result

    def concatenate_audio_segments(self, audio_segments, labels):
        con_audio = torch.tensor([])
        con_labels = torch.tensor([])
        for audio, label in zip(audio_segments, labels):
            con_audio = torch.cat((con_audio, audio), dim = -1)
            con_labels = torch.cat((con_labels, torch.tensor(label)), dim = -1)
        return con_audio, con_labels

    def cut_wav_by_sec(self, waveform, time_len, sample_rate):
        sameple_len = int(time_len * sample_rate / 1000)
        seg_wav = waveform[:, 0 : sameple_len]
        return seg_wav
    
    def concat_by_rec(self, merged_audio_name, selected_segs, label_time=400):
        audio_lists = []
        label_lists = []
        
        selected_segs = sorted(selected_segs, key=lambda x: x[0])

        # read audio into waveforms
        sr = 16000
        for utt, audio_name, lang, length in selected_segs:
            length = int(length)

            waveform, sr = torchaudio.load(audio_name)

            # cut to closest 400ms multipliers
            after_cut_len = math.floor(length / label_time) * label_time
            waveform = self.cut_wav_by_sec(waveform, after_cut_len, sr)
            # print(f'after cut len: {after_cut_len}, waveform: {waveform.shape} = {waveform.shape[-1]/sr*1000}ms')
            
            audio_lists.append(waveform)
            label_lists.append(self.gen_lab_by_time(after_cut_len, lang, label_time))

        # concatenate and store
        cat_waveform, cat_lab = self.concatenate_audio_segments(audio_lists, label_lists)
        cat_save_path = self.dir + self.save_dir + '/' + merged_audio_name

        print(cat_waveform.shape)

        torchaudio.save(cat_save_path, cat_waveform, 16000, bits_per_sample=16)

        # write in the file
        # save_data_lab_path = self.dir + self.save_dir + '/' + self.data_label_list
        save_data_lab_path = self.dir + self.save_dir + '/' + 'data_label_list.txt'
        
        # print(cat_lab.tolist())

        with open(save_data_lab_path, 'a') as file:
            file.write(cat_save_path + '\t' + str(cat_lab.tolist()) + '\t' + str(length) + '\n')



if __name__ == '__main__':    
    file_path = "/export/fs05/ywang793/merlion/train/"

    if not os.path.exists(file_path +'/cat/'):
        os.mkdir(file_path +'/cat/')

    concat = cs_concat(dir_name=file_path)
                
    for k in concat.lists_by_rec.keys():
        # print(concat.lists_by_rec[k])
        concat.concat_by_rec(k, concat.lists_by_rec[k])

    print('concatenate process done.')