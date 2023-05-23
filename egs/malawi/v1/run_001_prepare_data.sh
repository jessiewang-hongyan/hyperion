#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

#$ -N malawi_prepare
#$ -j y -o /export/c12/ywang793/logs/log.malawi_prepare
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c0*|c1[0123456789]
#$ -wd /export/fs05/ywang793/hyperion/egs/malawi/v1
# Submit to GPU
#$ -q g.q

#log_file='/export/c12/ywang793/logs/log.dihard2019_prepare'
#echo "------------
#working directory: $(pwd)
#---------------" >> "$log_file"


. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. $HYP_ROOT/hyp_utils/parse_options.sh || exit 1;
. datapath.sh 

#if [ $stage -le 1 ];then
#
#    # Prepare the VoxCeleb1 dataset for training.
#    local/make_voxceleb1cat.pl $voxceleb1_root 16 data
#
#    # Prepare the VoxCeleb2 dataset for training.
#    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
#    utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train
#fi

#if [ $stage -le 2 ];then
#    # prepare Dihard2019
##    local/make_dihard2019.sh $dihard2019_dev data/dihard2019_dev
##    local/make_dihard2019.sh $dihard2019_eval data/dihard2019_eval
#    chmod +x local/make_malawi.sh
#    local/make_malawi.sh
#fi
malawi_dir=/export/fs05/leibny/CCWD-Fe62023/langdev
data_dir=/export/fs05/ywang793/malawi_data

echo "making data dir $data_dir"

mkdir -p $data_dir

find $malawi_dir -name "*.mp3" | \
    awk '
{ bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
  print bn, "ffmpeg -i "$1" "$1".wav - |" }' | sort -k1,1 > $data_dir/wav.scp

awk '{ print $1,$1}' $data_dir/wav.scp  > $data_dir/utt2spk
cat $data_dir/utt2spk > $data_dir/spk2utt

for f in $(find $malawi_dir -name "*.lab" | sort)
do
    awk '{ bn=FILENAME; sub(/.*\//,"",bn); sub(/\.lab$/,"",bn);
           printf "%s-%010d-%010d %s %f %f\n", bn, $1*1000, $2*1000, bn, $1, $2}' $f
done > $data_dir/vad.segments

rm -f $data_dir/reco2num_spks
for f in $(find $malawi_dir -name "*.rttm" | sort)
do
    cat $f
    awk '{ print $2, $8}' $f | sort -u | awk '{ f=$1; count++}END{ print f, count}' >> $data_dir/reco2num_spks

done > $data_dir/diarization.rttm

for f in $(find $malawi_dir -name "*.uem" | sort)
do
    cat $f
done > $data_dir/diarization.uem

utils/validate_data_dir.sh --no-feats --no-text $data_dir