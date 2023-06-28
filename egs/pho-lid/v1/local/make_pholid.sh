#!/bin/bash
# Copyright 2020   Johns Hopkins Universiy (Jesus Villalba)
# Apache 2.0.
#
# Creates the DIHARD 2019 data directories.

# if [ $# != 2 ]; then
#   echo "Usage: $0 <dihard-dir> <data-dir>"
#   echo " e.g.: $0 /export/corpora/LDC/LDC2019E31 data/dihard2019_dev"
# fi

data_root_dir="/export/fs05/ywang793/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_audio"
processed_data_dir="/export/fs05/ywang793/hyperion/egs/pholid/v1/data"

echo "data dir $data_root_dir"
echo "making data dir $processed_data_dir"

mkdir -p $processed_data_dir
mkdir -p $processed_data_dir/wav
mkdir -p $processed_data_dir/seg

file_name="TTS_P10040TT_VCST_ECxxx_01_AO_35259847_v001_R004_CRR_MERLIon-CCS.wav"

files=$(find "$data_root_dir" -name "$file_name")
echo "find $data_root_dir -name $file_name"

if [[ -n "$files" ]]; then
    for f in $files
    do
    # Extract the base name of the file
    base_name=$(basename "$f")
    # Remove the extension from the base name
    # file_name=$(echo "${base_name%.*}" | cut -d '_' -f 2,5)
    file_name="${base_name%.*}"
    
    # Construct the output file path with the desired format
    output_file="$processed_data_dir/wav/$file_name.wav"
    # Convert the file using FFmpeg
    ffmpeg -i "$f" -ar 16000 -ac 1 -f wav "$output_file"

    # Display a message with the filename
    echo "filename: $file_name"
    done
else
    echo "No files found matching the specified pattern."
fi
    
# source ~/.bashrc
# conda activate merlion

# python ./data_prepare.py