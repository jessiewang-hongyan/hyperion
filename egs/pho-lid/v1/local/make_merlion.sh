#!/bin/bash
# Copyright 2020   Johns Hopkins Universiy (Jesus Villalba)
# Apache 2.0.
#
# Creates the DIHARD 2019 data directories.

# data_root_dir="/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_audio"
processed_data_dir="/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train"

# echo "data dir $data_root_dir"
# echo "making data dir $processed_data_dir"

mkdir -p $processed_data_dir
mkdir -p $processed_data_dir/audio
mkdir -p $processed_data_dir/seg
mkdir -p $processed_data_dir/processed

    
source ~/.bashrc
conda activate merlion

# python ./local/merlion_process/merlion_seg.py
# python ./local/merlion_process/merlion_2vec.py --step 0 \
#     --lredir /export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train \
#     --model xlsr_53 \
#     --device 0 \
#     --layer 16 \
#     --seglen 10\
#     --overlap 1\
#     --savedir /export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train/processed \
#     --audiodir /export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train/audio
# python ./local/merlion_process/merlion_spk_split.py
python ./local/merlion_process/merlion_merge.py

