#!/bin/bash

#$ -N emb_plot
#$ -j y -o /export/c12/ywang793/logs/log.emb_plot
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=d01
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1 
# Submit to GPU c0*|c1[0123456789]
#$ -q p.q

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 

source ~/.bashrc
# conda activate merlion
conda activate python3_9
LD_LIBRARY_PATH=/export/fs05/ywang793/miniconda3/lib

python emb_plot_umap.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json
# python emb_plot.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json