
# BASELINES
#CUDA_VISIBLE_DEVICES=2 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine \
#    --name baseline_no_wd \
#    >> baseline_no_wd.log &
#CUDA_VISIBLE_DEVICES=3 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine \
#    --name baseline_wd_0.0005 \
#    >> baseline_wd_0.0005.log &
#CUDA_VISIBLE_DEVICES=3 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine \
#    --nesterov \
#    --name baseline_wd_0.0005_nesterov \
#    >> baseline_wd_0.0005_nesterov.log &

CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine \
    --nesterov \
    --AutoAugment-cutout-only \
    --name baseline_wd_0.0005_nesterov_Cutout \
    >> baseline_wd_0.0005_nesterov_Cutout.log &
