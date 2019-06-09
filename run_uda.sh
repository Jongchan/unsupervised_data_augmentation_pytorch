

#CUDA_VISIBLE_DEVICES=0,1 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA \
#    --name UDA_AutoAugment \
#    >> UDA_AutoAugment.log &

#CUDA_VISIBLE_DEVICES=0,1 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --UDA-CUTOUT \
#    --name UDA_AutoAugment_Cutout \
#    >> UDA_AutoAugment_Cutout.log &

#CUDA_VISIBLE_DEVICES=2,3 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all \
#    --name UDA_AutoAugment_FULL \
#    >> UDA_AutoAugment_FULL.log &

#CUDA_VISIBLE_DEVICES=2,3 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout \
#    >> UDA_AutoAugment_FULL_Cutout.log &

CUDA_VISIBLE_DEVICES=6,7 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine \
    --UDA --cifar10-policy-all --UDA-CUTOUT \
    --name UDA_AutoAugment_FULL_Cutout_no_nesterov \
    >> UDA_AutoAugment_FULL_Cutout_no_nesterov.log &

CUDA_VISIBLE_DEVICES=6,7 \
    python main.py \
    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 200000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --UDA --cifar10-policy-all --UDA-CUTOUT \
    --name UDA_AutoAugment_FULL_Cutout_200K \
    >> UDA_AutoAugment_FULL_Cutout_200K.log &
