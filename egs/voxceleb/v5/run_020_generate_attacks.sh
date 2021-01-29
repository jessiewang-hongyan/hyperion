#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=7
config_file=default_config.sh
use_gpu=false
xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

attack_dir=exp/attacks/$nnet_name/
class2int=data/${nnet_data}_proc_audio_no_sil/lists_xvec/class2int

if [ $stage -le 1 ]; then
    # create fgsm attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 25 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type fgsm --attacks-min-eps 3e-6 --attacks-max-eps 0.03 --p-attack 0.25 --random-seed 1000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int fgsm \
    	    $attack_dir/fgsm/${name}
    done
fi

if [ $stage -le 2 ]; then
    # create rand fgsm attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 25 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type rand-fgsm --attacks-min-eps 3e-6 --attacks-max-eps 0.03 --attacks-min-alpha 1e-6 --attacks-max-alpha 0.005 --p-attack 0.25 --random-seed 2000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int rand-fgsm \
    	    $attack_dir/rand-fgsm/${name}
    done
fi

if [ $stage -le 3 ]; then
    # create iter fgsm attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type iter-fgsm --attacks-min-eps 3e-6 --attacks-max-eps 0.03 --attacks-min-alpha 1e-6 --attacks-max-alpha 0.005 --p-attack 0.25 --random-seed 3000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int iter-fgsm \
    	    $attack_dir/iter-fgsm/${name}
    done
fi

if [ $stage -le 4 ]; then
    # create pgd attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type pgd --norms inf --attacks-min-eps 3e-6 --attacks-max-eps 0.03 --attacks-min-alpha 1e-6 --attacks-max-alpha 0.005 --p-attack 0.25 --min-num-random-init 0 --max-num-random-init 5 --random-seed 4000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int pgd-linf \
    	    $attack_dir/pgd-linf/${name}
    done
fi

if [ $stage -le 5 ]; then
    # create pgd attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type pgd --norms 1 --attacks-min-eps 3e-6 --attacks-max-eps 0.03 --attacks-min-alpha 1e-6 --attacks-max-alpha 0.005 --p-attack 0.25 --min-num-random-init 0 --max-num-random-init 5 --random-seed 5000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int pgd-l1 \
    	    $attack_dir/pgd-l1/${name}
    done
fi

if [ $stage -le 6 ]; then
    # create pgd attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type pgd --norms 2 --attacks-min-eps 3e-6 --attacks-max-eps 0.03 --attacks-min-alpha 1e-6 --attacks-max-alpha 0.005 --p-attack 0.25 --min-num-random-init 0 --max-num-random-init 5 --random-seed 6000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int pgd-l2 \
    	    $attack_dir/pgd-l2/${name}
    done
fi


if [ $stage -le 7 ]; then
    # create CW-L2 attacks
    # for name in voxceleb2cat
    # do
    # 	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 500 ${xvec_args} \
    # 	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
    # 	    --feat-config $feat_config \
    # 	    --attacks-opts "--attacks-attack-type cw-l2 --attacks-min-confidence 0 --attacks-max-confidence 3 --attacks-min-lr 1e-3 --attacks-max-lr 1e-2 --attacks-min-iter 10 --attacks-max-iter 200 --attacks-norm-time --p-attack 0.25 --random-seed 7000" \
    # 	    $nnet data/${name}_proc_audio_no_sil \
    # 	    $class2int cw-l2 \
    # 	    $attack_dir/cw-l2/${name}
    # done
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 500 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type cw-l2 --attacks-min-confidence 0 --attacks-max-confidence 3 --attacks-min-lr 1e-5 --attacks-max-lr 1e-2 --attacks-min-iter 10 --attacks-max-iter 100 --attacks-norm-time --p-attack 0.25 --random-seed 7000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int cw-l2 \
    	    $attack_dir/cw-l2/${name}
    done

fi

exit

if [ $stage -le 8 ]; then
    # create CW-Linf attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 500 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type cw-linf --attacks-min-confidence 0 --attacks-max-confidence 3 --attacks-min-lr 1e-3 --attacks-max-lr 1e-2 --attacks-min-iter 10 --attacks-max-iter 200 --p-attack 0.25 --random-seed 8000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int cw-linf \
    	    $attack_dir/cw-linf/${name}
    done
fi

if [ $stage -le 9 ]; then
    # create CW-L0 attacks
    for name in voxceleb2cat
    do
    	steps_attacks/generate_adv_attacks_xvector_classif.sh  --cmd "$xvec_cmd --mem 12G" --nj 1000 ${xvec_args} \
	    --random-utt-length true --min-utt-length 4 --max-utt-length 60 --use-bin-vad false \
	    --feat-config $feat_config \
	    --attacks-opts "--attacks-attack-type cw-l0 --attacks-min-confidence 0 --attacks-max-confidence 3 --attacks-min-lr 1e-3 --attacks-max-lr 1e-2 --attacks-min-iter 10 --attacks-max-iter 100 --p-attack 0.25 --random-seed 9000" \
    	    $nnet data/${name}_proc_audio_no_sil \
	    $class2int cw-l0 \
    	    $attack_dir/cw-l0/${name}
    done
fi



exit
