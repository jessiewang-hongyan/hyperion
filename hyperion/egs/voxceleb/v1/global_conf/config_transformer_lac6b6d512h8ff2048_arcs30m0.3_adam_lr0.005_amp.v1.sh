# x-Vector using Transformer Encoder as x-Vector Encoder
# Transformer Encoder uses 6 Transformer blocks with 
# model_d=512 ff_d=2048, heads=8
# Self attention context is limited to 6 frames around the current frame
# input is downsampled x4 by conv network

#xvector training 
nnet_data=voxceleb2cat_train_combined
nnet_type=transformer
batch_size_1gpu=64
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.005
dropout=0
embed_dim=256
s=30
margin_warmup=20
margin=0.3
blocks=6
d_model=512
heads=8
d_ff=2048
att_context=6 # 250 ms
nnet_opt="--in-feats 80 --num-enc-blocks $blocks --enc-d-model $d_model --num-enc-heads $heads --enc-ff-type linear --enc-d-ff $d_ff --in-layer-type conv2d-sub --enc-att-type local-scaled-dot-prod-v1 --enc-att-context $att_context"
opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 12000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
nnet_name=transformer_csub_lac${att_context}b${blocks}d${d_model}h${heads}linff${d_ff}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=80
num_augs=5
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0080.pth


#xvector finetuning
ft_batch_size_1gpu=32
ft_eff_batch_size=512 # effective batch size
ft_min_chunk=400
ft_max_chunk=400
ft_ipe=1
ft_lr=0.01
ft_nnet_num_epochs=40
ft_margin_warmup=3
ft_opt_opt="--optim.opt-type sgd --optim.lr $ft_lr --optim.momentum 0.9 --optim.weight-decay 1e-5 --use-amp --var-batch-size"
#ft_lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 100 --lrsched.update-lr-on-opt-step"
ft_lrs_opt="--lrsched.lrsch-type cos_lr --lrsched.t 2500 --lrsched.t-mul 2 --lrsched.warm-restarts --lrsched.gamma 0.75 --lrsched.min-lr 1e-4 --lrsched.warmup-steps 100 --lrsched.update-lr-on-opt-step"
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v2
# ft_opt_opt="--optim.opt-type adam --optim.lr $ft_lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
# ft_lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
# ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_adam_lr${ft_lr}_b${ft_eff_batch_size}_amp.v2
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0007.pth


#back-end
lda_dim=200
plda_y_dim=150
plda_z_dim=200

plda_data=voxceleb2cat_train_combined
plda_type=splda
