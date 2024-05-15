#!/bin/bash

#$ -N inf_eval
#$ -j y -o /export/c12/ywang793/logs/log.inf_eval_seame1
#$ -M ywang793@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/fs05/ywang793/hyperion/egs/pho-lid/v1 
# Submit to GPU c0*|c1[0123456789]
#$ -q g.q

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate merlion
# conda activate hyperion
LD_LIBRARY_PATH=/export/fs05/ywang793/miniconda3/lib

# ppho
# python ./dscore/score.py -r ./inference/outputs/merlion_ground_truth_channel0.rttm -s ./inference/outputs/merlion_pred_ppho.rttm 

# pconv
# python ./dscore/score.py -r ./inference/outputs/merlion_ground_truth_channel0.rttm -s ./inference/outputs/merlion_pred_pconv.rttm 
# python ./dscore/score.py -r ./inference/outputs/merlion_ground_truth_channel0.rttm -s ./inference/outputs/merlion_pred.rttm 

# python ./dscore/score.py -r ./inference/outputs/seame_ground_truth_channel0.rttm -s ./inference/outputs/seame_pred_pho.rttm 
python ./dscore/score.py -r ./inference/outputs/seame_ground_truth_channel0.rttm -s ./inference/outputs/seame_pred_ppho_tune.rttm 
python ./dscore/score.py -r ./inference/outputs/merlion_ground_truth_channel0.rttm -s ./inference/outputs/merlion_pred_ppho_tune.rttm 
