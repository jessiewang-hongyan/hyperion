import os
import librosa
import soundfile as sf
from lhotse import RecordingSet, Recording, AudioSource, SupervisionSet, SupervisionSegment, CutSet
from lhotse.dataset.diarization import DiarizationDataset
from lhotse.features.io import NumpyFilesWriter
from pholid_feat import PholidExtractor
import json
import torch

class Preprocess():
    def __init__(self, audio_dir, save_dir, audio_pattern, rttm_file, vad_file):
        self.audio_dir = audio_dir
        self.save_dir = save_dir
        self.audio_pattern = audio_pattern
        self.rttm_file = rttm_file
        self.vad_file = vad_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            for file in os.listdir(self.audio_dir):
                self.upsampling_lre(self.audio_dir + file, self.save_dir)
            
        self.make_sets()
        self.make_cutset(self.vad_file)
    
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

    def make_sets(self):
        self.recordingset = RecordingSet.from_dir(path=self.save_dir, pattern=self.audio_pattern)
        print(f"recording set length: {len(self.recordingset)}")
        self.ground_truth = SupervisionSet.from_rttm(self.rttm_file)
        print(f"supervision set length: {len(self.ground_truth)}")

        # rec_folder = "inference/rec_set/"
        # # os.mkdir(rec_folder)
        # for rec in self.recordingset.to_dicts():
        #     with open(rec_folder + "rec.json", "a") as fp:
        #         json.dump(rec, fp)
        #     break

        # sup_folder = "inference/sup_set/"
        # # os.mkdir(sup_folder)
        # for sup in self.ground_truth.to_dicts():
        #     with open(sup_folder + "sup.json", "a") as fp:
        #         json.dump(sup, fp)
        #         fp.write("\n")
        #     break


    # TODO: 
    # 1. make cuts based on vad result
    # 2. extract audio from a cut, process it to pholid input format
    # 3. store the preprocessed result and make labels
    def make_cutset(self, vad_file):
        vad_supervision = SupervisionSet.from_rttm(vad_file)

        for sup in vad_supervision.to_dicts():
            print(sup)
            break

        cuts = CutSet.from_manifests(
            recordings=self.recordingset,
            supervisions=vad_supervision
        )

        print(f"cut set length: {len(cuts)}")
        # cuts.describe()

        # trim as VAD indicates
        cuts = cuts.trim_to_supervisions()
        # print(f"cut set length after trim: {len(cuts)}")

        # truncate to at most 10s segments
        cuts = cuts.truncate(max_duration=10, offset_type="start", preserve_id=False)
        # print(f"cut set length after trucation: {len(cuts)}")
        # cuts = cuts.drop_supervisions()
        # padding to 10s
        # cuts = cuts.pad(duration=10, preserve_id=True)

        # print(f"cut set length after trucation and padding: {len(cuts)}")

        # # compute the features
        # feature_folder = "inference/extracted/"
        # if not os.path.exists("inference/extracted/"):
        #     os.mkdir("inference/extracted/")
        # cuts = cuts.compute_and_store_features(extractor=PholidFeatureExtractor, storage_path="inference/extracted/", storage_type=NumpyFilesWriter)
        # print(f"final cut set length: {len(cuts)}")

        self.vad_cuts = cuts
        print(f"final cut set length: {len(cuts)}")

    def process(self, model, lang_map, export_file):
        preprocessor = PholidExtractor()
        model = model.to(self.device)
        model.eval()

        for i, cut in enumerate(self.vad_cuts):
            # print(cut.load_audio().squeeze().shape)
            # print(cut)
            sup = cut.supervisions[0]
            start = cut.start
            duration = cut.duration
            original_file = sup.recording_id
            # feat = preprocessor.extract(cut.load_audio().squeeze()).unsqueeze(0)
            # seq_len = [feat.shape[1]]

            
            feat = preprocessor.extract(cut.load_audio().squeeze()).unsqueeze(0)
            seq_len = [feat.shape[1]]
            embeddings = model.get_embeddings(feat, seq_len)
            embeddings = embeddings.reshape(-1, 1, embeddings.shape[-1])

            batch_size = feat.shape[0]
            frame_size = feat.shape[1]
            new_seq_len = [1]* (batch_size*frame_size)

            outputs = model.bf_check(embeddings, new_seq_len)
            predicted = torch.argmax(outputs, -1).squeeze()

            print(f"feat.shape: {feat.shape}, seq_len: {seq_len}")
            print(f"original_file: {original_file}, start: {start}, duration: {duration}, result: {outputs}, pred: {predicted}")
            
            with open(export_file, "a") as f:
                if seq_len[0] > 1:
                    s = start
                    lapse = duration / seq_len[0]
                    for pred in predicted:
                        lang = lang_map[pred]
                        fields = ['SPEAKER', original_file, '0', str(round(s, 4)), str(round(lapse, 4)), '<NA>', '<NA>', lang, '<NA>', '<NA>']
                        line = ' '.join(fields) + "\n"
                        f.write(line)
                        s += lapse
                else:
                    lang = lang_map[predicted.tolist()]
                    fields = ['SPEAKER', original_file, '0', str(round(start, 4)), str(round(duration, 4)), '<NA>', '<NA>', lang, '<NA>', '<NA>']
                    line = ' '.join(fields) + "\n"
                    f.write(line)
        print(f"Processed {i} cuts!")
            # if i > 10:
            #     break
            


            

if __name__ == "__main__":

    my_model = model.PHOLID_conv_pho()
    my_model.load("../models/pholid_pho_seame/pholid_pho_seame_epoch_12.ckpt")
    cutset.process(my_model)

    model = model.to(self.device)
    model.eval()

    for i, cut in enumerate(self.vad_cuts):
        # print(cut.load_audio().squeeze().shape)
        # print(cut)
        sup = cut.supervisions[0]
        start = cut.start
        duration = cut.duration
        original_file = sup.recording_id
        # feat = preprocessor.extract(cut.load_audio().squeeze()).unsqueeze(0)
        # seq_len = [feat.shape[1]]

        
        feat = preprocessor.extract(cut.load_audio().squeeze()).unsqueeze(0)
        seq_len = [feat.shape[1]]
        embeddings = model.get_embeddings(feat, seq_len)
        embeddings = embeddings.reshape(-1, 1, embeddings.shape[-1])

        batch_size = feat.shape[0]
        frame_size = feat.shape[1]
        new_seq_len = [1]* (batch_size*frame_size)

        outputs = model.bf_check(embeddings, new_seq_len)
        predicted = torch.argmax(outputs, -1).squeeze()

        print(f"feat.shape: {feat.shape}, seq_len: {seq_len}")
        print(f"original_file: {original_file}, start: {start}, duration: {duration}, result: {outputs}, pred: {predicted}")
        
        with open(export_file, "a") as f:
            if seq_len[0] > 1:
                s = start
                lapse = duration / seq_len[0]
                for pred in predicted:
                    lang = lang_map[pred]
                    fields = ['SPEAKER', original_file, '0', str(round(s, 4)), str(round(lapse, 4)), '<NA>', '<NA>', lang, '<NA>', '<NA>']
                    line = ' '.join(fields) + "\n"
                    f.write(line)
                    s += lapse
            else:
                lang = lang_map[predicted.tolist()]
                fields = ['SPEAKER', original_file, '0', str(round(start, 4)), str(round(duration, 4)), '<NA>', '<NA>', lang, '<NA>', '<NA>']
                line = ' '.join(fields) + "\n"
                f.write(line)
