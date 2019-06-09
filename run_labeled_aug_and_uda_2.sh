
CUDA_VISIBLE_DEVICES=6,7 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --AutoAugment --cifar10-policy-all \
    --UDA \
    --name Labeled_AutoAugment_FULL_UDA_AutoAugment_FULL \
    >> Labeled_AutoAugment_FULL_UDA_AutoAugment_FULL.log &

CUDA_VISIBLE_DEVICES=6,7 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --AutoAugment-all --cifar10-policy-all \
    --UDA --UDA-CUTOUT \
    --name Labeled_AutoAugment_FULL_Cutout_UDA_AutoAugment_FULL_Cutout \
    >> Labeled_AutoAugment_FULL_Cutout_UDA_AutoAugment_FULL_Cutout.log &
