import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Model

class xlsr_reader(object):
    def __init__(self, save_path):
        super.__init__(self)
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.save_path = save_path

    def read_wav(self, filepath:str):
        waveform, sample_rate = torchaudio.load(filepath)
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            features = self.model.extract_features(inputs.input_values)
            array = features.numpy()
            np.save(self.save_path+filepath.replace('.wav', '')+'.npy', array)

class label_reader(object):
    def __init__(self, save_path):
        super.__init__(self)
        self.save_path = save_path

    def load_labels_from_csv(self, csv_path):


        
if __name__=='__main__':
    filepath='/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_audio/TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav'
    save_path='./data/npy'
    reader = xlsr_reader(save_path)
    reader.read_wav(filepath)
    