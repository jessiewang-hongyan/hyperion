#!/bin/bash
# Copyright 2020   Johns Hopkins Universiy (Jesus Villalba)
# Apache 2.0.
#
# Creates the DIHARD 2019 data directories.

# if [ $# != 2 ]; then
#   echo "Usage: $0 <dihard-dir> <data-dir>"
#   echo " e.g.: $0 /export/corpora/LDC/LDC2019E31 data/dihard2019_dev"
# fi

libri_dir=/export/corpora5/LibriSpeech/train-clean-100

data_dir=/export/fs05/ywang793/mini-libri

# my_data_dir=/export/fs05/ywang793/hyperion/egs/malawi/v1/data

echo "making data dir $data_dir"

mkdir -p $data_dir
mkdir -p $data_dir/wav

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

# find $libri_dir -name "*.flac" | \
#  awk '
# { bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
#   split(bn, parts, "_");
#   print bn, "/export/fs05/ywang793/mini-libri/wav/"bn".wav" }' | sort -k1,1 > $data_dir/wav.scp
# awk '{bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
#       split(bn, parts, "_");
#       printf "%s %s\n", bn, parts[3]}' $data_dir/wav.scp  > $data_dir/utt2spk
# cat $data_dir/utt2spk > $data_dir/spk2utt


filename="*.flac"
files=$(find "$libri_dir" -name "$filename" | sort)

for f in $files
do
    # Extract the base name of the file
    base_name=$(basename "$f")
    # Remove the extension from the base name
    file_name="${base_name%.*}"
    # Construct the output file path with the desired format
    output_no_path="$file_name.wav"
    output_file="$data_dir/wav/$output_no_path"
    # Convert the file using FFmpeg
    ffmpeg -i "$f"  -ar 16000 -ac 1 -f wav "$output_file"

    awk '{print $output_no_path $output_file }' | sort -k1,1 > $data_dir/wav.scp
    awk '{printf "%s %s\n", $output_no_path, $last_dir_name}' $data_dir/wav.scp  > $data_dir/utt2spk
    awk '{ spk=$last_dir_name; utt=parts[3];
          printf "%s-%010d-%010d %s %f %f\n", bn, spk*1000, utt*1000, bn, spk, utt}'

    # Display a message with the filename
    echo "filename: $file_name"
done
cat $data_dir/utt2spk > $data_dir/spk2utt

# find $libri_dir -name "*.flac" | \
# awk '{bn=$1; sub(/.*\//,"",bn); sub(/\.mp3$/,"",bn);
#       split(bn, parts, "_");
#       printf "%s %s\n", bn, parts[3]}' $data_dir/wav.scp  > $data_dir/utt2spk
# cat $data_dir/utt2spk > $data_dir/spk2utt

for f in $(find $libri_dir -name "*.flac" | sort)
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
    
