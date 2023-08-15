#!/bin/bash
#$ -N dummy_job
#$ -j y -o /export/c12/ywang793/logs/log.dummy_job
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=d01
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1
# Submit to GPU
#$ -q p.q

source /home/gqin2/scripts/acquire-gpu

echo "cuda device: $CUDA_VISIBLE_DEVICES"