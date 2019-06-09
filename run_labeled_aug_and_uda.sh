
CUDA_VISIBLE_DEVICES=4,5 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --AutoAugment \
    --UDA \
    --name Labeled_AutoAugment_UDA_AutoAugment \
    >> Labeled_AutoAugment_UDA_AutoAugment.log &

CUDA_VISIBLE_DEVICES=4,5 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --AutoAugment-all \
    --UDA --UDA-CUTOUT \
    --name Labeled_AutoAugment_Cutout_UDA_AutoAugment_Cutout \
    >> Labeled_AutoAugment_Cutout_UDA_AutoAugment_Cutout.log &

