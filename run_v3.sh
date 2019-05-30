#CUDA_VISIBLE_DEVICES=1 \
#    python train_semi_3.py \
#    --normalization ZCA_v3 \
#    --name v3_baseline
#CUDA_VISIBLE_DEVICES=0 \
#    python train_semi_3.py \
#    --normalization ZCA_v3 \
#    --gaussian-noise-level 0.0 \
#    --name v3_baseline_no_gaussian

CUDA_VISIBLE_DEVICES=0 \
    python train_semi_3.py \
    --normalization ZCA_v3 \
    --l1-reg 0.0 \
    --l2-reg 0.0 \
    --name v3_baseline_no_l1_l2_2 >> v3_baseline_no_l1_l2_2.log &

#CUDA_VISIBLE_DEVICES=0 \
#    python train_semi_3.py \
#    --normalization ZCA_v3 \
#    --l1-reg 0.0 \
#    --name v3_baseline_no_l1

#CUDA_VISIBLE_DEVICES=2,3 \
#    python train_semi_3.py \
#    --TSA linear \
#    --UDA \
#    --normalization ZCA_v3 \
#    --name v3_UDA_linear

#CUDA_VISIBLE_DEVICES=4,5 \
#    python train_semi_3.py \
#    --TSA linear \
#    --UDA \
#    --normalization ZCA_v3 \
#    --name v3_UDA_linear_no_l1

#CUDA_VISIBLE_DEVICES=6,7 \
#    python train_semi_3.py \
#    --TSA linear \
#    --UDA \
#    --normalization ZCA_v3 \
#    --batch-size 32 \
#    --name v3_UDA_linear_sup_bs_32

#CUDA_VISIBLE_DEVICES=5 \
#    python train_semi_3.py \
#    --TSA linear \
#    --UDA \
#    --normalization ZCA_v3 \
#    --batch-size 32 --lr 0.001 \
#    --name v3_UDA_linear_sup_bs_32_lr_0.001
