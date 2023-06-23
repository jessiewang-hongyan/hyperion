#!/bin/bash
# Copyright 2020   Johns Hopkins Universiy (Jesus Villalba)
# Apache 2.0.
#
# Creates the DIHARD 2019 data directories.

# if [ $# != 2 ]; then
#   echo "Usage: $0 <dihard-dir> <data-dir>"
#   echo " e.g.: $0 /export/corpora/LDC/LDC2019E31 data/dihard2019_dev"
# fi

data_root_dir=/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL
processed_data_dir=/export/fs05/ywang793/hyperion/egs/pholid/v1/data

echo "making data dir $processed_data_dir"

mkdir -p $processed_data_dir
mkdir -p $processed_data_dir/wav

file_name=TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav

find $data_root_dir/_audio -name $file_name | \
 awk '
{ bn=$1; sub(/.*\//,"",bn); sub(/\.wav$/,"",bn);
    split(bn, parts, "_");
  print parts[2], $processed_data_dir"/wav/"parts[2]"_"parts[5]".wav" }' | sort -k1,1 > $data_dir/wav.scp
awk '{ print $1,$1}' $data_dir/wav.scp  > $data_dir/utt2spk
cat $data_dir/utt2spk > $data_dir/spk2utt


files=$(find $data_root_dir/_audio -name $filename)

for f in $files
do
    # Extract the base name of the file
    base_name=$(basename "$f")
    # Remove the extension from the base name
    file_name=$(echo "${base_name%.*}" | cut -d '_' -f 2,5)
    
    # Construct the output file path with the desired format
    output_file="$processed_data_dir/wav/$file_name.wav"
    # Convert the file using FFmpeg
    ffmpeg -i "$f"  -ar 16000 -ac 1 -f wav "$output_file"

    # Display a message with the filename
    echo "filename: $file_name"
done

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
    
