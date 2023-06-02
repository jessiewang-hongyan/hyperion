#!/bin/bash
# Copyright 2020   Johns Hopkins Universiy (Jesus Villalba)
# Apache 2.0.
#
# Creates the DIHARD 2019 data directories.

# if [ $# != 2 ]; then
#   echo "Usage: $0 <dihard-dir> <data-dir>"
#   echo " e.g.: $0 /export/corpora/LDC/LDC2019E31 data/dihard2019_dev"
# fi

dihard_dir=/export/fs05/leibny/CCWD-Fe62023/langdev
data_dir=/export/fs05/ywang793/hyperion/egs/malawi/v1/data

echo "making data dir $data_dir"

mkdir -p $data_dir

# find $dihard_dir -name "*.flac" | \
#     awk '
# { bn=$1; sub(/.*\//,"",bn); sub(/\.flac$/,"",bn); 
#   print bn, "sox "$1" -t wav -b 16 -e signed-integer - |" }' | sort -k1,1 > $data_dir/wav.scp

# awk '{ print $1,$1}' $data_dir/wav.scp  > $data_dir/utt2spk
# cat $data_dir/utt2spk > $data_dir/spk2utt

# for f in $(find $dihard_dir -name "*.lab" | sort)
# do
#     awk '{ bn=FILENAME; sub(/.*\//,"",bn); sub(/\.lab$/,"",bn); 
#            printf "%s-%010d-%010d %s %f %f\n", bn, $1*1000, $2*1000, bn, $1, $2}' $f
# done > $data_dir/vad.segments


find $dihard_dir -name "*.mp3" | \
    awk '
{ bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
  split(bn, parts, "_");
  print parts[1], "_", parts[2], "ffmpeg -i "$1" "$1".wav - |" }' | sort -k1,1 > $data_dir/wav.scp

awk '{bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
      split(bn, parts, "_");
      printf "%s_%s %s\n", parts[1], parts[2], parts[3]}' $data_dir/wav.scp  > $data_dir/utt2spk
cat $data_dir/utt2spk > $data_dir/spk2utt

for f in $(find $dihard_dir -name "*.mp3" | sort)
do
    awk '{ bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
          split(bn, parts, "_"); spk=parts[1] parts[2]; utt=parts[3];
          printf "%s-%010d-%010d %s %f %f\n", bn, spk*1000, utt*1000, bn, spk, utt}' $f
done > $data_dir/vad.segments

rm -f $data_dir/reco2num_spks
for f in $(find $dihard_dir -name "*.rttm" | sort)
do
    cat $f
    awk '{ print $2, $8}' $f | sort -u | awk '{ f=$1; count++}END{ print f, count}' >> $data_dir/reco2num_spks

done > $data_dir/diarization.rttm

for f in $(find $dihard_dir -name "*.uem" | sort)
do
    cat $f
done > $data_dir/diarization.uem

utils/validate_data_dir.sh --no-feats --no-text $data_dir
    
