#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

#spk det back-end
lda_tel_dim=150
ncoh_tel=300

w_mu1=1
w_B1=0.5
w_W1=0.75
w_mu2=0
w_B2=0.5
w_W2=0.5
num_spks=600

plda_tel_y_dim=125
plda_tel_z_dim=150

plda_tel_type=splda
plda_tel_data=sre_tel
plda_adapt_data=sre18_cmn2_adapt_lab
coh_tel_data=sre18_dev_unlabeled

adapted=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

if [ "$adapted" == "true" ];then
    nnet_name=${nnet_name}_adapt_cmn2
fi

plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt_v2_a1_mu${w_mu1}B${w_B1}W${w_W1}_a2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}
be_tel_name=lda${lda_tel_dim}_${plda_tel_label}_${plda_tel_data}_${plda_adapt_data}

xvector_dir=exp/xvectors/$nnet_name
be_tel_dir=exp/be/$nnet_name/$be_tel_name
score_dir=exp/scores/$nnet_name/${be_tel_name}
score_plda_dir=$score_dir/plda

if [ ! -d scoring_software/sre19-cmn2 ];then
    local/download_sre19cmn2_scoring_tool.sh 
fi

if [ $stage -le 1 ]; then

    echo "Train BE"
    steps_be/train_tel_be_v2.sh --cmd "$train_cmd" \
    	--lda_dim $lda_tel_dim \
    	--plda_type $plda_tel_type \
    	--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    	--w_mu1 $w_mu1 --w_B1 $w_B1 --w_W1 $w_W1 \
    	--w_mu2 $w_mu2 --w_B2 $w_B2 --w_W2 $w_W2 --num_spks_unlab $num_spks \
    	$xvector_dir/$plda_tel_data/xvector.scp \
    	data/$plda_tel_data \
	$xvector_dir/${plda_adapt_data}/xvector.scp \
	data/${plda_adapt_data} \
	$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    	$sre18_dev_meta $be_tel_dir 


fi

if [ $stage -le 2 ]; then

    #SRE18
    echo "eval SRE18 without S-Norm"
    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type \
			       data/sre18_eval40_test_cmn2/trials \
			       data/sre18_eval40_enroll_cmn2/utt2spk \
			       $xvector_dir/sre18_eval40_cmn2/xvector.scp \
			       $be_tel_dir/lda_lnorm_adapt.h5 \
			       $be_tel_dir/plda_adapt2.h5 \
			       $score_plda_dir/sre18_eval40_cmn2_scores &

    echo "eval SRE19 without S-Norm"
    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type \
			       data/sre19_eval_test_cmn2/trials \
			       data/sre19_eval_enroll_cmn2/utt2spk \
			       $xvector_dir/sre19_eval_cmn2/xvector.scp \
			       $be_tel_dir/lda_lnorm_adapt.h5 \
			       $be_tel_dir/plda_adapt2.h5 \
			       $score_plda_dir/sre19_eval_cmn2_scores &

    wait
    local/score_sre18cmn2.sh data/sre18_eval40_test_cmn2 eval40 $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $score_plda_dir
fi

if [ $stage -le 3 ];then
    local/calibrate_sre19cmn2_v1.sh --cmd "$train_cmd" eval40 $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_eval40_test_cmn2 eval40 ${score_plda_dir}_cal_v1eval40
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1eval40
fi


score_plda_dir=$score_dir/plda_snorm${ncoh_tel}

if [ $stage -le 4 ]; then

    #SRE18
    echo "SRE18 S-Norm"
    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type --ncoh $ncoh_tel \
				     data/sre18_eval40_test_cmn2/trials \
				     data/sre18_eval40_enroll_cmn2/utt2spk \
				     $xvector_dir/sre18_eval40_cmn2/xvector.scp \
				     data/${coh_tel_data}/utt2spk \
				     $xvector_dir/${coh_tel_data}/xvector.scp \
				     $be_tel_dir/lda_lnorm_adapt.h5 \
				     $be_tel_dir/plda_adapt2.h5 \
				     $score_plda_dir/sre18_eval40_cmn2_scores &

    echo "SRE19 S-Norm"
    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type --ncoh $ncoh_tel \
				     data/sre19_eval_test_cmn2/trials \
				     data/sre19_eval_enroll_cmn2/utt2spk \
				     $xvector_dir/sre19_eval_cmn2/xvector.scp \
				     data/${coh_tel_data}/utt2spk \
				     $xvector_dir/${coh_tel_data}/xvector.scp \
				     $be_tel_dir/lda_lnorm_adapt.h5 \
				     $be_tel_dir/plda_adapt2.h5 \
				     $score_plda_dir/sre19_eval_cmn2_scores &

    wait
    local/score_sre18cmn2.sh data/sre18_eval40_test_cmn2 eval40 $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $score_plda_dir
fi

if [ $stage -le 5 ];then
    local/calibrate_sre19cmn2_v1.sh --cmd "$train_cmd" eval40 $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_eval40_test_cmn2 eval40 ${score_plda_dir}_cal_v1eval40
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1eval40
fi

    
exit
