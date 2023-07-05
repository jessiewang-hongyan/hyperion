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

# class feature_extract(object):
#     def __init__(self, save_path):
#         super().__init__()
#         model_name = "facebook/wav2vec2-large-xlsr-53"
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
#         self.save_path = save_path

#     def read_wav(self, filepath:str):
#         waveform, sample_rate = torchaudio.load(filepath)
#         waveform = waveform.flatten()
#         print(f'shape of speech: {waveform.shape}')

#         # Resample if necessary
#         xlsr_rate = self.feature_extractor.sampling_rate
#         if sample_rate != xlsr_rate:
#             waveform = torchaudio.transforms.Resample(sample_rate, xlsr_rate)(waveform)
#         input_values = self.feature_extractor(waveform, sampling_rate=xlsr_rate, return_tensors="np", return_attention_mask=False).input_values
        
#         with torch.no_grad():
#             features = self.feature_extractor(input_values)['input_values']
#             np.save(filepath.replace('.wav', '')+'.npy', features)


class label_reader(object):
    def __init__(self, 
                 audio_path:str, 
                 audio_label_list:str, 
                 audio_save_path='/audio/',
                 seg_save_path='/seg/', 
                 list_save_path='/data_label_list.txt', 
                 silence='NON_SPEECH', 
                 absolute_path=False, 
                 save_root='/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train/'):
        super().__init__()
        self.audio_path = audio_path
        self.audio_label_list = audio_label_list

        self.audio_save_path = audio_save_path if absolute_path else save_root+audio_save_path
        self.seg_save_path = seg_save_path if absolute_path else save_root+seg_save_path
        self.list_save_path = list_save_path if absolute_path else save_root+list_save_path
        self.silence = silence
        self.audio_segs = dict()

    def upsampling_lre(self, audio, save_dir):
        if audio.endswith('.sph'):
            data, sr = librosa.load(audio, sr=None)
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.sph', '.wav')
            sf.write(new_name, data, 8000, subtype='PCM_16')
            subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)
        elif audio.endswith('.wav') or audio.endswith('.WAV'):
            new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.WAV', '.wav')
            subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)

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
                temp['seg'] = (int(start)*1000, int(end)*1000)
                temp['lab'] = lang
                temp['utt'] = utt
                temp['length'] = int(length)*1000
                if audio not in self.audio_segs.keys(): #and audio == 'TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav':
                    self.audio_segs[audio] = list()
                
                if overlap == 'FALSE' and lang != self.silence: #and audio == 'TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav':
                    self.audio_segs[audio].append(temp)
                           
    def read_audio_seg(self):
        # mapping file
        file_path = self.list_save_path

        # read each wav file
        for recording in self.audio_segs.keys():
            segments = self.audio_segs[recording]
            audio_path = self.audio_path + '/' + recording

            # resampling and force mono-channel
            save_dir = self.audio_save_path
            saved_audio = self.upsampling_lre(audio_path, save_dir)
            waveform, sample_rate = torchaudio.load(saved_audio)

            # do segmentation for each recorded segment
            for s in segments:
                start, end = s["seg"]
                utt = s["utt"]
                lab = s['lab']

                # segment speech
                seg_wav = waveform[:, int(start) : int(end)]
                seg_name = recording.replace('.wav', '') + '_' + utt +'.wav'
                segment_path = self.seg_save_path + seg_name
                torchaudio.save(segment_path, seg_wav, sample_rate)

                # write in the file
                with open(file_path, 'a') as file:
                    file.write(seg_name+ ' ' + lab + '\n')


if __name__ == '__main__':
    train_set_root = '/export/fs05/ywang793//merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL'
    audio_path = '/_audio'
    audio_label_list = '/_labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv'

    reader = label_reader(train_set_root+audio_path, train_set_root+audio_label_list)
    reader.get_seg()
    reader.read_audio_seg()

    

    