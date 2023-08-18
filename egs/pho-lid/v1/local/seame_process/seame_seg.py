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

class label_reader(object):
    def __init__(self, 
                 audio_path:str, 
                 audio_label_list:str, 
                 audio_save_path='/audio/',
                 seg_save_path='/seg/', 
                 list_save_path='/data_label_list.txt', 
                 silence='NON_SPEECH', 
                 absolute_path=False, 
                 save_root='/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/seame_new/',
                 sample_rate=16000):
        super().__init__()
        self.audio_path = audio_path
        self.audio_label_list = audio_label_list

        self.audio_save_path = audio_save_path if absolute_path else save_root+audio_save_path
        self.seg_save_path = seg_save_path if absolute_path else save_root+seg_save_path
        self.list_save_path = list_save_path if absolute_path else save_root+list_save_path
        self.silence = silence
        self.audio_segs = dict()

        self.sample_rate = sample_rate

    def upsampling_lre(self, audio, save_dir):
        # if audio.endswith('.sph'):
        #     data, sr = librosa.load(audio, sr=None)
        #     new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.sph', '.wav')
        #     sf.write(new_name, data, 8000, subtype='PCM_16')
        #     subprocess.call(f"sox {audio} -r {self.sample_rate} {new_name}", shell=True)
        # el
        if audio.endswith('.wav') or audio.endswith('.WAV'):
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.WAV', '.wav')
            subprocess.call(f"sox \"{audio}\" -r {self.sample_rate} \"{new_name}\"", shell=True)
        elif audio.endswith('.flac'):
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.flac', '.wav')
            subprocess.call(f"sox \"{audio}\" -r {self.sample_rate} \"{new_name}\"", shell=True)
        
        print(audio)
        print(new_name)

        return new_name

    def get_seg(self):
        with open(self.audio_label_list, 'r') as file:
            csv_reader = csv.reader(file, delimiter='\t')

            for row in csv_reader:
                (audio, start, end, lang, _) = tuple(row)
                
                temp = dict()
                temp['seg'] = (int(start), int(end))
                temp['lab'] = lang
                if audio not in self.audio_segs.keys(): 
                    self.audio_segs[audio] = list()
                
                if lang != 'CS': 
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
            audio_path = self.audio_path + '/' + recording + '.flac'
            # print(audio_path)

            # resampling and force mono-channel
            save_dir = self.audio_save_path
            saved_audio = self.upsampling_lre(audio_path, save_dir)
            waveform, sample_rate = torchaudio.load(saved_audio)

            # do segmentation for each recorded segment
            for idx, s in enumerate(segments):
                t_start, t_end = s["seg"]
                w_start = int(t_start * sample_rate / 1000)
                w_end = int(t_end * sample_rate / 1000)
                utt = str(idx)
                lab = s['lab']
                length = t_end - t_start

                # segment speech
                seg_wav = waveform[:, w_start : w_end]
                seg_name = recording.replace('.wav', '') + '_' + utt +'.wav'
                segment_path = self.seg_save_path + seg_name
                torchaudio.save(segment_path, seg_wav, sample_rate)

                # write in the file
                with open(file_path, 'a') as file:
                    file.write(seg_name+ '\t' + lab + '\t' + str(length) + '\n')


if __name__ == '__main__':
    dataset_root = "/export/corpora5/LDC/LDC2015S04/data"
    data_list = ['/interview', '/conversation']
    label_list = '/transcript/phaseII/'
    audio_list = '/audio/'

    for recording_type in data_list:
        audio_label_path = dataset_root + recording_type + label_list

        save_parent = '/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/seame_new/'

        for file in os.listdir(audio_label_path):
            filename = file.replace('.txt', '')
            audio_label_list = dataset_root + recording_type + label_list+filename+'.txt'
            audio_path = dataset_root + recording_type + audio_list
            
            if filename not in os.listdir(save_parent):
                os.mkdir(save_parent+filename)
                os.mkdir(save_parent+filename+'/audio')
                os.mkdir(save_parent+filename+'/seg')
                os.mkdir(save_parent+filename+'/processed')
                os.mkdir(save_parent+filename+'/cat')
                os.mkdir(save_parent+filename+'/pure')

                reader = label_reader(
                    audio_path, 
                    audio_label_list,
                    save_root=save_parent+filename+'/')
                reader.get_seg()
                reader.read_audio_seg()

    print('SEAME_new seg done.')