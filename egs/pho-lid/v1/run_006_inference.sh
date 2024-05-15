#!/bin/bash

#$ -N inference
#$ -j y -o /export/c12/ywang793/logs/log.der_ppho_tune_results
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
conda activate merlion
# conda activate python3_9
LD_LIBRARY_PATH=/export/fs05/ywang793/miniconda3/lib

# . activate_cuda_10.2.89.sh
# chmod +x inference/vad.py

# do it separately and store each step

source_folder=/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/merlion/train/audio/
file_name=TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav
# python inference/ground_truth.py 
# python inference/vad.py  
# python inference/preprocess.py 
# python inference/predict.py 
# python inference/eval.py 

echo "DER result for PHOLID"

python inf.py
python inf_seame.py
# python inf_seame_2.py
# python inference/seame_to_rttm.py

. ./inference/eval.sh