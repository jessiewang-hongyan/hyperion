import os
import numpy as np
import csv
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Model

# class xlsr_reader(object):
#     def __init__(self, save_path):
#         super.__init__(super)
#         self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
#         self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
#         self.save_path = save_path

#     def read_wav(self, filepath:str):
#         waveform, sample_rate = torchaudio.load(filepath)
#         inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

#         with torch.no_grad():
#             features = self.model.extract_features(inputs.input_values)
#             array = features.numpy()
#             np.save(self.save_path+filepath.replace('.wav', '')+'.npy', array)

# class audio_reader(object):
#     def __init__(self, save_path):
#         super.__init__(super)
#         self.save_path = save_path

#     def read_wav_by_seg(self, filepath:str, start:int, end:int):
#         waveform, sample_rate = torchaudio.load(filepath)
#         waveform1 = waveform1[:, frame_offset : frame_offset + num_frames]
#         metadata = torchaudio.info(waveform)
#         print(metadata)




class label_reader(object):
    def __init__(self, save_path:str, audio_path:str):
        super.__init__(super)
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
                
                if audio not in self.audio_segs.keys() and audio == 'TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav':
                    self.audio_segs[audio] = list()
                
                if audio == 'TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav':
                    self.audio_segs[audio].append(temp)
                           
    def read_audio_seg(self):
        for recording in self.audio_segs.keys()[:1]:
            segments = self.audio_segs[recording]
            audio_path = self.audio_path + '/' + recording
            waveform, sample_rate = torchaudio.load(audio_path)

            for s in segments:
                start, end = s["seg"]
                utt = s["utt"]

                seg_wav = waveform[:, start : end]
                seg_name = recording + '_' + utt +'.wav'
                segment_path = os.path.join(self.save_path, seg_name)
                torchaudio.save(segment_path, seg_wav, sample_rate)

        
if __name__ == '__main__':
    audio_path = '/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_audio'
    save_path = './data/seg'
    csv_path = '/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv'
    
    reader = label_reader(save_path, audio_path)
    reader.get_seg(csv_path)
    reader.read_audio_seg()
    