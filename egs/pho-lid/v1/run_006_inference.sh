#!/bin/bash

#$ -N inference
#$ -j y -o /export/c12/ywang793/logs/log.inference
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1 
# Submit to GPU c0*|c1[0123456789]
#$ -q g.q

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
conda activate hyperion
LD_LIBRARY_PATH=/export/fs05/ywang793/miniconda3/lib

. activate_cuda_10.2.89.sh
chmod +x inference/vad.py

source_folder=/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train/audio/
file_name=TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav
python inference/vad.py 

