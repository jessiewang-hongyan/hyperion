#!/bin/bash
# Copyright 2020   Johns Hopkins Universiy (Jesus Villalba)
# Apache 2.0.
#
# Creates the DIHARD 2019 data directories.

# data_root_dir="/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_audio"
# processed_data_dir="/export/fs05/ywang793/libri"

# echo "data dir $data_root_dir"
# echo "making data dir $processed_data_dir"

# mkdir -p $processed_data_dir
    
source ~/.bashrc
conda activate merlion
# conda activate merlion4d01
# conda activate python3_9
# conda activate cuda11

# step 1: downsampling
# preserve only pure language segments
# python ./local/libri_process/libri_prep.py

# step 2: concatenate pure languages to simulate CS data
# python ./local/seame_process/seame_concat.py

# step 3: convert pure-lang segs into vectors for pholid-conv training
python ./local/libri_process/libri_2vec.py

# step 4: convert mixed-lang segs into vectors for clf training
# python ./local/seame_process/seame_mix_2vec.py

# step 5: split into train and test according to speakers
# python ./local/seame_process/seame_spk_split.py
# python ./local/seame_process/seame_merge.py