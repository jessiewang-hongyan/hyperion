#!/bin/bash
#$ -N pholid_clear
#$ -j y -o /export/c12/ywang793/logs/log.pholid_clear
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=d01
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1
# Submit to GPU, for d01 use p.q, otherwise g.q (c0*|c1[0123456789])
#$ -q p.q

source /home/gqin2/scripts/acquire-gpu
# export CUDA_VISIBLE_DEVICES=$(free-gpu)

echo "cuda device: $CUDA_VISIBLE_DEVICES"

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 

conda activate merlion4d01
python ./local/delete_bydir.py
