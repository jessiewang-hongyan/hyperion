#!/bin/bash
#$ -N merlion_cat
#$ -j y -o /export/c12/ywang793/logs/log.merlion_cat
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1
# Submit to GPU, for d01 use p.q, otherwise g.q (c0*|c1[0123456789])
#$ -q g.q

source /home/gqin2/scripts/acquire-gpu
# export CUDA_VISIBLE_DEVICES=$(free-gpu)

echo "cuda device: $CUDA_VISIBLE_DEVICES"

# . ./cmd.sh
# . ./path.sh
# set -e

# stage=1
# config_file=default_config.sh

# . parse_options.sh || exit 1;
# . datapath.sh 


. local/make_merlion.sh
# . local/make_seame.sh
# . local/make_libri.sh

