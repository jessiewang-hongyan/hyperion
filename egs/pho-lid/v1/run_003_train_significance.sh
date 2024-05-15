#!/bin/bash

#$ -N sig_eng
#$ -j y -o /export/fs05/ywang793/logs/log.sig_eng
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
echo "----------------------"
echo "Double check. Scoring on outputs, free on norm 0205, reg=10, lr=1e-5, epoch=10. Squared version."
echo "----------------------"

# python bf_significance.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_significance.json
# python bf_sig_tune.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_sig_tune.json
python bf_sig_plot.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_significance.json
# python bf_plot.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_significance.json
