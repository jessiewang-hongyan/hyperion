import torch
import torchaudio
from hyperion import feats


if __name__ == "__main__":
    source_folder="/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train/audio/"
    file_name="TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav"

    waveform, sr = torchaudio.load(source_folder+file_name)
    vad = feats.energy_vad.EnergyVAD(sample_frequency=sr)

    vad_result = vad.compute(waveform)
    print(vad_result)

