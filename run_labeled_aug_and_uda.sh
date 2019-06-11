
#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --AutoAugment \
#    --UDA \
#    --name Labeled_AutoAugment_UDA_AutoAugment \
#    >> Labeled_AutoAugment_UDA_AutoAugment.log &

#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --AutoAugment-all \
#    --UDA --UDA-CUTOUT \
#    --name Labeled_AutoAugment_Cutout_UDA_AutoAugment_Cutout \
#    >> Labeled_AutoAugment_Cutout_UDA_AutoAugment_Cutout.log &


#CUDA_VISIBLE_DEVICES=2,3 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --AutoAugment-cutout-only \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name Labeled_Cutout_UDA_AutoAugment_FULL_Cutout \
#    >> Labeled_Cutout_UDA_AutoAugment_FULL_Cutout.log &

#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 200000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --AutoAugment-cutout-only \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name Labeled_Cutout_UDA_AutoAugment_FULL_Cutout_200K \
#    >> Labeled_Cutout_UDA_AutoAugment_FULL_Cutout_200K.log &

CUDA_VISIBLE_DEVICES=0,1 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --AutoAugment-cutout-only \
    --UDA --cifar10-policy-all --UDA-CUTOUT \
    --dropout-rate 0.0 \
    --name Labeled_Cutout_UDA_AutoAugment_FULL_Cutout_no_dropout \
    >> Labeled_Cutout_UDA_AutoAugment_FULL_Cutout_no_dropout.log &

CUDA_VISIBLE_DEVICES=2,3 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --AutoAugment-all \
    --UDA --cifar10-policy-all --UDA-CUTOUT \
    --dropout-rate 0.0 \
    --name Labeled_AutoAugment_FULL_Cutout_UDA_AutoAugment_FULL_Cutout_no_dropout \
    >> Labeled_AutoAugment_FULL_Cutout_UDA_AutoAugment_FULL_Cutout_no_dropout.log &
