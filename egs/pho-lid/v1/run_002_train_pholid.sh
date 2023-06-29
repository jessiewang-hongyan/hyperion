#!/bin/bash

#$ -N pholid_train
#$ -j y -o /export/c12/ywang793/logs/log.pholid_train
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c0*|c1[0123456789]
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1
# Submit to GPU
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

source ~/.bashrc
conda activate merlion

python train_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/config_example.json