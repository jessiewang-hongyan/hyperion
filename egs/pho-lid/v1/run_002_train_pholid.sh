#!/bin/bash

#$ -N exp_pholid
#$ -j y -o /export/c12/ywang793/logs/log.exp_pholid_pho
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1
# Submit to GPU (c0*|c1[0123456789])
#$ -q g.q

source /home/gqin2/scripts/acquire-gpu
# export CUDA_VISIBLE_DEVICES=$(free-gpu)

echo "cuda device: $CUDA_VISIBLE_DEVICES"

. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 

mkdir -p ./models

source ~/.bashrc
conda activate merlion
# conda activate python3_9
# conda activate cuda11
# conda activate merlion4d01

echo "train on merlion:"
python train_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_merlion_pholid.json

echo "train on seame:"
python train_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_pholid.json
# python test_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_pconv.json
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_pconv.json
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_merlion_pconv.json

