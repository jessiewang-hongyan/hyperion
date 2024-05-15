#!/bin/bash

#$ -N dummy_test1
#$ -j y -o /export/fs05/ywang793/logs/log.dummy_test1
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
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
echo "dummy test label=1"
echo "-------------------------"

python bf_dummy.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json

# echo "tdnn z3 result (epoch 9):"
# python train_ppho_tdnn.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_tdnn.json
# python bf_ppho_tdnn.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json

# echo "BF on SEAME pure:"
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf.json

# echo "BF on SEAME pure -> SEAME mix lr=1e-5:"
# echo "BF ppho no scoring tuned on MELRIon lr=1e-5"
# python bf_PHOLID_pho.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json

# python cal_der.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json
# python train_PHOLID_pho.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf_pho.json #--gpu $CUDA_VISIBLE_DEVICES
# python bf_PHOLID_pho.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf2_pho.json #--gpu $CUDA_VISIBLE_DEVICES

# echo "BF on SEAME pure -> MERLIon mix:"
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf3.json

# echo "BF on SEAME pure -> SEAME mix -> MERLIon mix:"
# python bf_PHOLID.py --json /export/fs05/ywang793/hyperion/egs/pho-lid/v1/cfgs/cfg_seame_bf4.json
