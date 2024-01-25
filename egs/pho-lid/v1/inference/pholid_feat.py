# from scipy.signal import stft
import os
import glob
import torch
import librosa
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf
import s3prl.upstream.wav2vec2.hubconf as hubconf
from sklearn.preprocessing import LabelEncoder
from s3prl.nn import S3PRLUpstream
import torch.nn.functional as F
from lhotse.features import FeatureExtractor
from dataclasses import dataclass
from lhotse.utils import Seconds
from typing import Any, Dict, List, Optional, Sequence, Union
import torch.nn as nn


class PholidExtractor():
    def __init__(self, model_name="xlsr_53", layer=16):
        self.model = S3PRLUpstream(model_name)
        self.layer = layer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def extract(self, samples):
        self.model.eval()
        device = self.device

        print(f"input sameple shape: {samples.shape}")
        # data, sr = librosa.load(audio, sr=None)
        data_ = torch.tensor(samples).to(device=device, dtype=torch.float).unsqueeze(0)
        
        data_wav_len = torch.tensor([data_.shape[-1]])
        features = self.model(data_, wavs_len=data_wav_len)
        features = features[0][self.layer].squeeze(0)

        feat_shape = features.shape
        if feat_shape[0]%20 == 0:
            new_dim0 = int(feat_shape[0]/20)
            features = features.reshape(new_dim0, 20, feat_shape[1])
            # np.save(save_name, features.cpu().detach().numpy().reshape((new_dim0, 20, feat_shape[1])))
            # f.write(f"{save_name} {label} {new_dim0}\n")
        else:
            new_dim0 = int(feat_shape[0]/20) + 1
            padding_size = 20 - feat_shape[0] % 20
            features = F.pad(features, [0, 0, 1, padding_size - 1], "constant", 0)
            features = features.reshape(new_dim0, 20, feat_shape[1])

        print(f"feature size: {features.shape}")

        return features
