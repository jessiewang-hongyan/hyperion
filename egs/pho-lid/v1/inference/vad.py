import torch
import torchaudio
# from hyperion import feats

def ground_truth_vad(rttm_file, vad_result_file, vad_slience_file, slience_label="NON_SPEECH", speech_label="SPEECH"):
    result = []
    slience = []
    
    with open(rttm_file, "r") as f:
        for row in f.readlines():
            new_line = row.strip().split()

            # find the lang label
            lang = new_line[7]
            if lang != slience_label:
                new_line[7] = speech_label
                result.append(" ".join(new_line))
            else:
                slience.append(new_line)

    with open(vad_result_file, "wb") as f:
        for line in result:
            f.write(line.encode('utf-8'))
            f.write(b'\n')

    with open(vad_slience_file, "wb") as f:
        for line in slience:
            f.write(line.encode('utf-8'))
            f.write(b'\n')
                


if __name__ == "__main__":
    rttm_file = "inference/outputs/merlion_ground_truth_channel0.rttm"
    vad_result_file = "inference/vad/merlion_truth_vad_channel0.rttm"
    vad_slience_file = "inference/vad/merlion_slience_channel0.rttm"
    ground_truth_vad(rttm_file, vad_result_file, vad_slience_file, slience_label="NON_SPEECH", speech_label="SPEECH")
