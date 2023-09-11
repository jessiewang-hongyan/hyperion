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
    def __init__(self, dir_name, langs=['EN', 'ZH'], audio_dir='/seg/', data_label_list='/data_label_list.txt', save_dir='/cat/', pure_dir='/pure/'):
        self.data_label_list = data_label_list
        self.dir = dir_name
        self.audio_dir = audio_dir
        self.save_dir = save_dir
        self.pure_dir = pure_dir
        self.le = LabelEncoder()
        self.langs = self.le.fit_transform(langs)

        # self.langs = self.le.transform(langs)
        self.lists_by_lang = self.get_lists_by_lang()


    def get_lists_by_lang(self):
        lists = dict()
        for i in self.langs:
            lists[i] = list()

        with open(self.dir+self.data_label_list, 'r') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            # print(f'self.dir:{self.dir}')
            for row in csv_reader:
                (audio, lang, length) = tuple(row)
                if lang.isnumeric():
                    lang = int(lang)
                else:
                    lang = self.le.transform([lang])[0]
                utt_code = int(audio.replace('.wav', '').split('_')[-1])
                lists[lang].append((utt_code, audio, length, lang))
        return lists
    
    def gen_lab_by_time(self, pure_audio_len, label, lapse=400):
        # print(f"pure audio length: {pure_audio_len}, multiplier: {math.floor(pure_audio_len/lapse)}")
        result = [label]*int(math.floor(pure_audio_len/lapse))
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
    
    def concat_pure(self, min_time_len=30000, lang_lab=0):
        selected_segs = []
        audio_lists = []
        dummy_label_list = []
        time_len = 0

        while time_len < min_time_len:  
            audio_selection = random.choices(self.lists_by_lang[lang_lab])[0]
            selected_segs.append(audio_selection)

            _, _, length, _ = audio_selection
            time_len += int(length)
        selected_segs = sorted(selected_segs, key=lambda x: x[0])

        # read audio into waveforms
        utt_code = ''
        for utt, audio_name, length, lang in selected_segs:
            utt_code = utt_code + '_' + str(utt) 
            length = int(length)

            waveform, sr = torchaudio.load(self.dir+self.audio_dir+audio_name)
            
            audio_lists.append(waveform)
            dummy_label_list.append([])
        
        # concatenate and store
        cat_waveform, _ = self.concatenate_audio_segments(audio_lists, dummy_label_list)
        filename = self.dir.split('/')[-1]
        cat_save_path = self.dir + self.pure_dir + '/' + filename + '_pure' + utt_code + '.wav'
        torchaudio.save(cat_save_path, cat_waveform, sr, bits_per_sample=16)

        # write in the file
        save_data_lab_path = self.dir + self.pure_dir + '/data_label_list.txt'
        # print(cat_lab.tolist())

        with open(save_data_lab_path, 'a') as file:
            file.write(filename + '_pure' + utt_code + '.wav' + '\t' + str(lang_lab) + '\t' + str(length) + '\n')

    def concat_random(self, min_time_len=30000, ratio=[0.5, 0.5], label_time=400):
        selected_segs = []
        audio_lists = []
        label_lists = []
        time_len = 0

        while time_len < min_time_len:  
            lang_selection = random.choices(self.langs, ratio)[0]
            audio_selection = random.choices(self.lists_by_lang[lang_selection])[0]
            selected_segs.append(audio_selection)

            _, _, length, _ = audio_selection
            time_len += int(length)
        
        selected_segs = sorted(selected_segs, key=lambda x: x[0])

        # read audio into waveforms
        utt_code = ''
        for utt, audio_name, length, lang in selected_segs:
            utt_code = utt_code + '_' + str(utt) 
            length = int(length)

            waveform, sr = torchaudio.load(self.dir+self.audio_dir+audio_name)
            # if >10s, preserve the first 10s; else, preserve to multiply of 400ms
            largest_len = 10000
            if length > largest_len:
                waveform = self.cut_wav_by_sec(waveform, largest_len, sr)
                length = largest_len
            else:
                length = int((length // label_time) * label_time)
                waveform = self.cut_wav_by_sec(waveform, length, sr)
            
            audio_lists.append(waveform)
            label_lists.append(self.gen_lab_by_time(length, lang, label_time))
        # print(f'label lists: {[len(x) for x in label_lists]}')

        # concatenate and store
        cat_waveform, cat_lab = self.concatenate_audio_segments(audio_lists, label_lists)
        filename = self.dir.split('/')[-1]
        cat_save_path = self.dir + self.save_dir + '/' + filename + '_cat' + utt_code + '.wav'
        torchaudio.save(cat_save_path, cat_waveform, sr, bits_per_sample=16)

        # write in the file
        # save_data_lab_path = self.dir + self.save_dir + '/' + self.data_label_list
        save_data_lab_path = self.dir + self.save_dir + '/' + 'data_label_list.txt'
        
        # print(cat_lab.tolist())

        with open(save_data_lab_path, 'a') as file:
            file.write(filename + '_cat' + utt_code + '.wav' + '\t' + str(cat_lab.tolist()) + '\t' + str(length) + '\n')



if __name__ == '__main__':
    mix_concat_num = 50
    pure_concat_num = 10
    
    for dir_name in os.listdir("data/seame/"):
        file_path = "data/seame/" + dir_name

        if os.path.exists(file_path +'/data_label_list.txt'):
            concat = cs_concat(dir_name=file_path)
            # check if both lang speech exists
            can_do_cs = True
            for lang in concat.lists_by_lang.keys():
                if len(concat.lists_by_lang[lang]) < 1:
                    can_do_cs = False
            if can_do_cs:
                # remove old label_file
                # save_data_lab_path = concat.dir + concat.save_dir + '/' + concat.data_label_list
                # if os.path.exists(save_data_lab_path):
                #     os.remove(save_data_lab_path)

                # generate longer pure audio
                # if not os.path.exists(concat.dir + "/data_label_list_old.txt"):
                
                for i in range(mix_concat_num):
                    concat.concat_random(ratio=[0.5, 0.5])

                # lang_list = concat.lists_by_lang
                # num_lang_0 = len(lang_list[0])
                # num_lang_1 = len(lang_list[1])

                # gen_lang_0_samples = pure_concat_num if num_lang_0 > 15 else 2
                # gen_lang_1_samples = pure_concat_num if num_lang_1 > 15 else 2
                # for i in range(gen_lang_0_samples):
                #     concat.concat_pure(min_time_len=30000, lang_lab=0)
                # for i in range(gen_lang_1_samples):
                #     concat.concat_pure(min_time_len=30000, lang_lab=1)

                print(file_path + 'concatenate process done.')

    print('concatenate process done.')