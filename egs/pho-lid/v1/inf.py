import sys
sys.path.append('./inference/')
from inference import preprocess
from preprocess import Preprocess
from model import *
from significance import *
import torch


if __name__ == "__main__":

    audio_dir = '/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_audio/'
    save_dir = 'inference/data'
    audio_pattern ='*.wav'
    rttm_file = "inference/outputs/merlion_ground_truth_channel0.rttm"
    vad_file = "inference/vad/merlion_truth_vad_channel0.rttm"

    pocessor = Preprocess(audio_dir, save_dir, audio_pattern, rttm_file, vad_file)
    # cutset.describe()
    # cutset_dict = cutset.to_dicts()
    # cut_folder = "inference/cuts/"
    # os.mkdir(cut_folder)
    # for cut in cutset_dict:
    #     with open(cut_folder + "cut{}.json".format(cut[id])) as fp:
    #         json.dump(cut, fp)
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # my_model = PHOLID(input_dim=1024, feat_dim=64, d_k=64, d_v=64, d_ff=2048, n_heads=8, n_lang=2).to(device)
    # my_model.load_state_dict(torch.load("./models/pholid_seame_bf2/pholid_seame_bf2_epoch_4.ckpt"))
    # my_model = PHOLID_conv_pho(input_dim=1024, feat_dim=64, d_k=64, d_v=64, d_ff=2048, n_heads=8, n_lang=2)
    # my_model.load_state_dict(torch.load("./models/ppho_norm_seame_bf_0205_lre5/ppho_norm_seame_bf_0205_lre5_epoch_5.ckpt"))
    # my_model = PHOLID_pho(input_dim=1024, feat_dim=64, d_k=64, d_v=64, d_ff=2048, n_heads=8, n_lang=2)
    # my_model.load_state_dict(torch.load("./models/ppho_noconv_seame_bf_1e-5/ppho_noconv_seame_bf_1e-5_epoch_4.ckpt"))
    
    my_model = PHOLID_conv_pho(input_dim=1024, feat_dim=64, d_k=64, d_v=64, d_ff=2048, n_heads=8, n_lang=2)
    
    # my_model = ScoringModel(model, n_lang=2)
    # my_model.load_state_dict(torch.load("./models/sig/ppho_seame_sig_freeze_till_conv/ppho_seame_sig_freeze_till_conv_reg10_lr1e-05_epoch_4.ckpt"))
    my_model.load_state_dict(torch.load("./models/ppho_seame_bf_1e-5tune/ppho_seame_bf_1e-5tune_epoch_4.ckpt"))
    
    my_model.eval()
    
    lang_map = ["English", "Mandarin"]
    export_file = "./inference/outputs/merlion_pred_ppho_tune.rttm"
    pocessor.process(my_model, lang_map, export_file)