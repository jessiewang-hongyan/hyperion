import os
import math
import numpy as np
import csv
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Model

# class audio_reader(object):
#     def __init__(self, save_path):
#         super.__init__(super)
#         self.save_path = save_path

#     def read_wav_by_seg(self, filepath:str, start:int, end:int):
#         waveform, sample_rate = torchaudio.load(filepath)
#         waveform1 = waveform1[:, frame_offset : frame_offset + num_frames]
#         metadata = torchaudio.info(waveform)
#         print(metadata)


class feature_extract(object):
    def __init__(self, save_path):
        super().__init__()
        model_name = "facebook/wav2vec2-large-xlsr-53"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.save_path = save_path

    def read_wav(self, filepath:str):
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.flatten()
        print(f'shape of speech: {waveform.shape}')

        # Resample if necessary
        xlsr_rate = self.feature_extractor.sampling_rate
        if sample_rate != xlsr_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, xlsr_rate)(waveform)
        # extract features
        # input_values = self.feature_extractor(waveform, sampling_rate=xlsr_rate, return_tensors="pt").input_values
        input_values = self.feature_extractor(waveform, sampling_rate=xlsr_rate, return_tensors="np").input_values
        
        with torch.no_grad():
            features = self.feature_extractor(input_values)
            np.save(filepath.replace('.wav', '')+'.npy', features)


class label_reader(object):
    def __init__(self, save_path:str, audio_path:str):
        super().__init__()
        self.save_path = save_path
        self.audio_path = audio_path
        self.audio_segs = dict()

    def get_seg(self, csv_path:str):
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                audio = row[0]
                utt = row[1]
                start = row[2]
                end = row[3]
                length = row[4]
                lang = row[5]
                overlap = row[6]

                temp = dict()
                temp['seg'] = (start, end)
                temp['lab'] = lang
                temp['utt'] = utt
                temp['length'] = length
                
                if audio != 'audio_name':
                    if audio not in self.audio_segs.keys() and audio == 'TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav':
                        self.audio_segs[audio] = list()
                    
                    if audio == 'TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav':
                        self.audio_segs[audio].append(temp)
                           
    def read_audio_seg(self):
        # mapping file
        file_path = os.path.join(self.save_path, 'data_label_list.txt')
        extractor = feature_extract(self.save_path)

        for recording in self.audio_segs.keys():
            segments = self.audio_segs[recording]
            audio_path = self.audio_path + '/' + recording
            waveform, sample_rate = torchaudio.load(audio_path)

            for s in segments:
                start, end = s["seg"]
                utt = s["utt"]
                lab = s['lab']
                length = s['length']

                # segment speech
                seg_wav = waveform[:, int(start) : int(end)]
                seg_name = 'seg/' + recording.replace('.wav', '') + '_' + utt +'.wav'
                segment_path = os.path.join(self.save_path, seg_name)
                torchaudio.save(segment_path, seg_wav, sample_rate)

                # write in the file
                with open(file_path, 'a') as file:
                    T_prime = math.ceil(int(length) / 20)
                    file.write(segment_path.replace('.wav', '.npy')+ ' ' + lab + ' ' + str(T_prime) + '\n')

                # convert
                extractor.read_wav(segment_path)


if __name__ == '__main__':
    audio_path = '/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/wav'
    save_path = './data'
    csv_path = '/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv'
    
    reader = label_reader(save_path, audio_path)
    reader.get_seg(csv_path)
    reader.read_audio_seg()

    

    