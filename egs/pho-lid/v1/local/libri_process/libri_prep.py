'''This preparation is for librispeech training data (one speech, one language label):
1. sampling data in 16kHz
2. assign each seg with a label
3. generate a list of segment-labels pairs
'''

import os
import math
import numpy as np
import csv
import torch
import torchaudio
import librosa
import soundfile
import subprocess

class label_reader(object):
    def __init__(self, 
                 audio_path:str,
                 save_root='/export/fs05/ywang793/mini_libri/',
                 default_lang='EN'):
        super().__init__()
        self.audio_path = audio_path
        self.save_root = save_root
        self.lab = default_lang

        if not os.path.exists(self.save_root):
            os.mkdir(self.save_root)

    def upsampling_lre(self, audio, save_dir):
        # if audio.endswith('.sph'):
        #     data, sr = librosa.load(audio, sr=None)
        #     new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.sph', '.wav')
        #     sf.write(new_name, data, 8000, subtype='PCM_16')
        #     subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)
        # el
        if audio.endswith('.wav') or audio.endswith('.WAV'):
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.WAV', '.wav')
            # subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)
            data, sr = librosa.load(audio, sr=None)
            soundfile.write(new_name, data, 16000, subtype='PCM_16')
        elif audio.endswith('.flac'):
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.flac', '.wav')
            # subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)
            data, sr = librosa.load(audio, sr=None)
            soundfile.write(new_name, data, 16000, subtype='PCM_16')

        return new_name

   
    def read_audio_seg(self):
        for spk in os.listdir(self.audio_path):
            spk_level_path = self.save_root + '/' + spk
            if not os.path.exists(spk_level_path):
                os.mkdir(spk_level_path)
            
            # mapping file
            file_path = spk_level_path + '/data_label_list.txt'

            print(f'write into: {file_path}')
            for passage in os.listdir(self.audio_path+'/'+spk):
                psg_level_path = spk_level_path + '/' + passage 
                if not os.path.exists(psg_level_path):
                    os.mkdir(psg_level_path)
                    
                for utt in os.listdir(self.audio_path+'/'+spk+'/'+passage):
                    if not utt.endswith('.txt'):
                        audio_path = self.audio_path+'/'+spk+'/'+passage+'/'+utt
                        print(f'Start writing: {audio_path}')

                        # resampling and force mono-channel
                        saved_audio = self.upsampling_lre(audio_path, psg_level_path)

                        # write in the file
                        with open(file_path, 'a') as file:
                            file.write(saved_audio+ '\t' + self.lab + '\n')

                        print(f'Finish writing: {saved_audio}')



if __name__ == '__main__':
    train_set_root = '/export/corpora5/LibriSpeech/train-clean-100'
    reader = label_reader(train_set_root)
    reader.read_audio_seg()

    

    