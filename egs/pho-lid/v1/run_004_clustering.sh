#!/bin/bash

#$ -N exp_ahc_s
#$ -j y -o /export/c12/ywang793/logs/log.exp_ahc_s
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1 
# Submit to GPU c0*|c1[0123456789]
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
# for b and c machines, cuda version 10.2
conda activate merlion
# the env for d01 machine
# conda activate merlion4d01
# conda activate python3_9

echo "Result with SEAME pure -> SEAME mix"
python PHOLID_clustering.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_merlion_clustering.json

# echo "Result with SEAME pure -> MERLIon mix"
# python PHOLID_clustering.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_merlion_clustering2.json

# echo "Result with SEAME pure -> SEAME mix -> MERLIon mix"
# python PHOLID_clustering.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_merlion_clustering3.json