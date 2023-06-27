#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
#$ -N pholid_prepare
#$ -j y -o /export/c12/ywang793/logs/log.pholid_prepare
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c0*|c1[0123456789]
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1
# Submit to GPU
#$ -q g.q
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 

# if [ $stage -le 1 ];then

#     # Prepare the VoxCeleb1 dataset for training.
#     local/make_voxceleb1cat.pl $voxceleb1_root 16 data

#     # Prepare the VoxCeleb2 dataset for training.
#     local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
#     utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train
# fi

# if [ $stage -le 2 ];then
#     # prepare Dihard2019
#     local/make_dihard2019.sh $dihard2019_dev data/dihard2019_dev
#     local/make_dihard2019.sh $dihard2019_eval data/dihard2019_eval
# fi
# local/make_dihard2019.sh  [where data in] [information/output of data]

# chmod +x local/make_pholid.sh
local/make_pholid.sh

source ~/.bashrc
conda activate merlion

python ./data_prepare.py
