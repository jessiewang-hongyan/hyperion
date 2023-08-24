#!/bin/bash

#$ -N clf_train_seame
#$ -j y -o /export/c12/ywang793/logs/log.clf_train_seame
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=d01
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1 
# Submit to GPU c0*|c1[0123456789]
#$ -q p.q

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
# for b and c machines, cuda version 10.2
# conda activate merlion
# the env for d01 machine
conda activate merlion4d01


python train_clf.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_pconv.json