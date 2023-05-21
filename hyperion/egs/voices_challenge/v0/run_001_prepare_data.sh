#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e


stage=1

. parse_options.sh || exit 1;

ldc_root=/export/corpora/LDC
sitw_root=/export/corpora/SRI/sitw
voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
voices_root=/export/corpora/SRI/VOiCES_2019_challenge
voices_dev_listdir=$voices_root/Development_Data/Speaker_Recognition/sid_dev_lists_and_keys
voices_eval_listdir=$voices_root/Evaluation_Data/Speaker_Recognition/sid_eval_lists
voices_eval_keydir=$voices_root/VOiCES_challenge_2019_post-eval-release

if [ $stage -le 1 ]; then
    # Prepare telephone and microphone speech from Mixer6.
    local/make_mx6.sh $ldc_root/LDC2013S03 16 data
    grep -v "trim 0.000 =0.000" data/mx6_mic/wav.scp > data/mx6_mic/wav.scp.tmp
    mv data/mx6_mic/wav.scp.tmp data/mx6_mic/wav.scp
    fix_data_dir.sh data/mx6_mic
    exit
fi

if [ $stage -le 2 ];then
    # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
    # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
    # contains the speakers that overlap between VoxCeleb1 and our evaluation
    # set SITW.  The script removes these overlapping speakers from VoxCeleb1.
    local/make_voxceleb1cat.pl $voxceleb1_root 16 data

    # Prepare the dev portion of the VoxCeleb2 dataset.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
fi

if [ $stage -le 3 ];then
  # Prepare SITW dev to train x-vector
    local/make_sitw_train.sh $sitw_root dev 16 data/sitw_train_dev
    local/make_sitw_train.sh $sitw_root eval 16 data/sitw_train_eval
    utils/combine_data.sh data/sitw_train data/sitw_train_dev data/sitw_train_eval
fi

if [ $stage -le 4 ];then
    # Prepare voices
    local/make_voices19_challenge_dev.sh $voices_root $voices_dev_listdir ./data
    local/make_voices19_challenge_eval.sh $voices_root $voices_eval_listdir $voices_eval_keydir ./data
fi

exit
