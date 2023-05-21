#!/bin/bash

#$ -N malawi_prepare
#$ -j y -o log.malawi_prepare
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c0*|c1[0123456789]
#$ -wd /export/c12/ywang793/vad_malawi
# Submit to GPU
#$ -q g.q

# Modified from Copyright 2020   Johns Hopkins Universiy (Jesus Villalba) DIHARD

# Assign a free GPU to your program (make sure -n matches the requested number of GPUs above)
#source /home/gqin2/scripts/acquire-gpu -n 1
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh

# activate python
#cd /export/c12/ywang793/miniconda3/bin || exit
#source ~/.bashrc

# activate environment
#conda activate merlion

# go to work directory
#cd /export/c12/ywang793/vad_malawi || exit


# malawi data dir
#malawi_dir=/export/fs05/leibny/CCWD-Fe62023/langdev
#data_dir=/export/fs05/ywang793/malawi_data

echo "making data dir $data_dir"

rm $data_dir/wav.scp

mkdir -p $data_dir

find $malawi_dir -name "*.mp3" | \
    awk '
{ bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
  print bn, "ffmpeg -i "$1" "$1".wav - |" }' | sort -k1,1 > $data_dir/wav.scp