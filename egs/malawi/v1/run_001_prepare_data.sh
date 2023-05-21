#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

#$ -N malawi_prepare
#$ -j y -o log.malawi_prepare
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c0*|c1[0123456789]
#$ -wd /export/c12/ywang793/vad_malawi
# Submit to GPU
#$ -q g.q

log_file='/export/c12/ywang793/logs/log.dihard2019_prepare'
echo "------------
working directory: $(pwd)
---------------" >> "$log_file"


. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. $HYP_ROOT/hyp_utils/parse_options.sh || exit 1;
. datapath.sh 

if [ $stage -le 1 ];then

    # Prepare the VoxCeleb1 dataset for training.
    local/make_voxceleb1cat.pl $voxceleb1_root 16 data

    # Prepare the VoxCeleb2 dataset for training.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
    utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train
fi

if [ $stage -le 2 ];then
    # prepare Dihard2019
#    local/make_dihard2019.sh $dihard2019_dev data/dihard2019_dev
#    local/make_dihard2019.sh $dihard2019_eval data/dihard2019_eval
    local/make_dihard2019.sh
fi
