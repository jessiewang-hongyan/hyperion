#!/bin/bash

#$ -N ppho_norm_bf
#$ -j y -o /export/c12/ywang793/logs/log.ppho_norm_0205
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c0*|c1[0123456789]
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1
# Submit to GPU (c0*|c1[0123456789], g.q; d01, p.q)
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
conda activate merlion
# conda activate python3_9
# conda activate cuda11
# conda activate merlion4d01

echo "-------------------------"
echo "for ppho bf lr=1e-5"
echo "-------------------------"

# echo "BF on SEAME pure:"
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf.json

echo "BF on SEAME pure -> SEAME mix:"
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2.json
# python train_PHOLID_pho.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf_pho.json #--gpu $CUDA_VISIBLE_DEVICES
python bf_PHOLID_pho.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json #--gpu $CUDA_VISIBLE_DEVICES

# echo "BF on SEAME pure -> MERLIon mix:"
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf3.json

# echo "BF on SEAME pure -> SEAME mix -> MERLIon mix:"
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf4.json
