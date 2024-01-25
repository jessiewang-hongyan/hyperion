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

from lhotse.features.kaldi.layers import (
    Wav2LogFilterBank,
    Wav2LogSpec,
    Wav2MFCC,
    Wav2Spec,
)

@dataclass
class PholidFeatureExtractorConfig:
    frame_len: Seconds = 0.400
    frame_shift: Seconds = 0.01
    model: S3PRLUpstream = S3PRLUpstream('xlsr_53')
    device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    layer: int = 16

class PholidFeatureExtractor(FeatureExtractor):
    """
    A minimal class example, showing how to implement a custom feature extractor in Lhotse.
    """
    name = 'pholid-feature-extractor'
    config_type = PholidFeatureExtractorConfig

    def __init__(self, config: Optional[PholidFeatureExtractorConfig] = None):
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2LogFilterBank(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def to(self, device: str):
        self.config.device = device
        self.extractor.to(device)



    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()

        print(f"input sameple shape: {samples.shape}")
        # data, sr = librosa.load(audio, sr=None)
        data_ = torch.tensor(sample).to(device=device, dtype=torch.float).unsqueeze(0)
        
        data_wav_len = torch.tensor([data_.shape[-1]])
        features = self.model(data_, wavs_len=data_wav_len)
        features = features[0][self.layer].squeeze(0)

        feat_shape = features.shape
        if feat_shape[0]%20 == 0:
            new_dim0 = int(feat_shape[0]/20)
            np.save(save_name, features.cpu().detach().numpy().reshape((new_dim0, 20, feat_shape[1])))
            f.write(f"{save_name} {label} {new_dim0}\n")
        else:
            new_dim0 = int(feat_shape[0]/20) + 1
            padding_size = 20 - feat_shape[0]%20
            features = F.pad(features, [0, 0, 1, padding_size - 1], "constant", 0)

        print(f"feature size: {featuers.shape}")

        return features
        # f, t, Zxx = stft(
        #     samples,
        #     sampling_rate,
        #     nperseg=round(self.config.frame_len * sampling_rate),
        #     noverlap=round(self.frame_shift * sampling_rate)
        # )
        # # Note: returning a magnitude of the STFT might interact badly with lilcom compression,
        # # as it performs quantization of the float values and works best with log-scale quantities.
        # # It's advised to turn lilcom compression off, or use log-scale, in such cases.
        # return np.abs(Zxx)

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return (sampling_rate * self.config.frame_len) / 2 + 1