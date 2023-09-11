'''This preparation is for merlion training data (one speech, multiple language labels):
1. sampling data in 16kHz
2. segment data based on provided labels
3. generate a list of segment-labels pairs
'''

import os
import math
import numpy as np
import csv
import torch
import torchaudio
import librosa
import soundfile as sf
import subprocess

class label_reader(object):
    def __init__(self, 
                 audio_path:str, 
                 audio_label_list:str, 
                 audio_save_path='/audio/',
                 seg_save_path='/seg/', 
                 list_save_path='/data_label_list.txt', 
                 silence='NON_SPEECH', 
                 absolute_path=False, 
                 save_root='/export/fs05/ywang793/merlion/train/'):
        super().__init__()
        self.audio_path = audio_path
        self.audio_label_list = audio_label_list

        self.audio_save_path = audio_save_path if absolute_path else save_root+audio_save_path
        self.seg_save_path = seg_save_path if absolute_path else save_root+seg_save_path
        self.list_save_path = list_save_path if absolute_path else save_root+list_save_path
        self.silence = silence
        self.audio_segs = dict()

    def upsampling_lre(self, audio, save_dir):
        if audio.endswith('.wav') or audio.endswith('.WAV'):
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.WAV', '.wav')
            # subprocess.call(f"sox \"{audio}\" -r {self.sample_rate} \"{new_name}\"", shell=True)
            data, sr = librosa.load(audio, sr=None)
            sf.write(new_name, data, 16000, subtype='PCM_16')

        elif audio.endswith('.flac'):
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.flac', '.wav')
            # subprocess.call(f"sox \"{audio}\" -r {self.sample_rate} \"{new_name}\"", shell=True)
            data, sr = librosa.load(audio, sr=None)
            sf.write(new_name, data, 16000, subtype='PCM_16')

        return new_name

    def get_seg(self, skip_head=True):
        with open(self.audio_label_list, 'r') as file:
            csv_reader = csv.reader(file)

            # Skips the heading
            if skip_head:
                heading = next(csv_reader)

            for row in csv_reader:
                (audio, utt, start, end, length, lang, overlap, dev) = tuple(row)
                
                temp = dict()
                temp['seg'] = (int(start), int(end))
                temp['lab'] = lang
                temp['utt'] = utt
                temp['length'] = int(length)
                if audio not in self.audio_segs.keys(): 
                    self.audio_segs[audio] = list()
                
                if overlap == 'FALSE' and lang != self.silence: 
                    self.audio_segs[audio].append(temp)
                           
    def read_audio_seg(self):
        # mapping file
        file_path = self.list_save_path

        # clear old version
        if os.path.exists(file_path):
            os.remove(file_path)

        # read each wav file
        for recording in self.audio_segs.keys():
            segments = self.audio_segs[recording]
            audio_path = self.audio_path + '/' + recording

            # resampling and force mono-channel
            save_dir = self.audio_save_path
            saved_audio = self.upsampling_lre(audio_path, save_dir)
            waveform, sample_rate = torchaudio.load(saved_audio)
            # print(f'loaded waveform: {waveform.shape}')

            # do segmentation for each recorded segment
            for s in segments:
                t_start, t_end = s["seg"]
                utt = s["utt"]
                lab = s['lab']
                w_start = int(t_start * sample_rate / 1000)
                w_end = int(t_end * sample_rate / 1000)


                # segment speech
                seg_wav = waveform[:, w_start : w_end]

                # print(f"t_start: {t_start}, t_end: {t_end}, w_start: {w_start}, w_end: {w_end}, seg_len: {seg_wav.shape}")

                seg_name = recording.replace('.wav', '') + '_' + utt +'.wav'
                segment_path = self.seg_save_path + seg_name
                torchaudio.save(segment_path, seg_wav, sample_rate, bits_per_sample=16)
                seg_len = t_end-t_start

                # write in the file
                with open(file_path, 'a') as file:
                    file.write(seg_name+ '\t' + lab + '\t' + str(seg_len) + '\n')


if __name__ == '__main__':
    train_set_root = '/export/fs05/ywang793//merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL'
    audio_path = '/_audio'
    audio_label_list = '/_labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv'

    reader = label_reader(train_set_root+audio_path, train_set_root+audio_label_list)
    reader.get_seg()
    reader.read_audio_seg()

    

    