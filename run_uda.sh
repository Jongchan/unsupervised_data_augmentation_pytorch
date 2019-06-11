

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

#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_no_nesterov \
#    >> UDA_AutoAugment_FULL_Cutout_no_nesterov.log &

#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 200000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_200K \
#    >> UDA_AutoAugment_FULL_Cutout_200K.log &

#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.003 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_init_lr_0.003 \
#    >> UDA_AutoAugment_FULL_Cutout_init_lr_0.003.log &

#CUDA_VISIBLE_DEVICES=2,3 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l2-reg 0.0001 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_l2_reg_0.0001 \
#    >> UDA_AutoAugment_FULL_Cutout_l2_reg_0.0001.log &

#CUDA_VISIBLE_DEVICES=2,3 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l1-reg 0.001 --l2-reg 0.0001 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_l2_reg_0.0001_l1_reg_0.000001 \
#    >> UDA_AutoAugment_FULL_Cutout_l2_reg_0.0001_l1_reg_0.000001.log &
#
#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l1-reg 0.001 --l2-reg 0.0001 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.000001 \
#    >> UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.000001.log &

#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l1-reg 0.0001 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.0001 \
#    >> UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.0001.log &

#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l1-reg 0.00001 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.00001 \
#    >> UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.00001.log &

#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l1-reg 0.00001 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 200000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.00001_200K \
#    >> UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.00001_200K.log &

#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l1-reg 0.000001 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.000001 \
#    >> UDA_AutoAugment_FULL_Cutout_l2_reg_0.0005_l1_reg_0.000001.log &

#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN --batch-size 32 --l1-reg 0 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
#    --dropout-rate 0.0 \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_no_dropout \
#    >> UDA_AutoAugment_FULL_Cutout_l2_no_dropout.log &

CUDA_VISIBLE_DEVICES=4,5 \
    python main.py \
    --normalization GCN --batch-size 32 --l1-reg 0 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --dropout-rate 0.1 \
    --UDA --cifar10-policy-all --UDA-CUTOUT \
    --name UDA_AutoAugment_FULL_Cutout_dropout_0.1 \
    >> UDA_AutoAugment_FULL_Cutout_dropout_0.1.log &

CUDA_VISIBLE_DEVICES=4,5 \
    python main.py \
    --normalization GCN --batch-size 32 --l1-reg 0 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 100000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov \
    --dropout-rate 0.2 \
    --UDA --cifar10-policy-all --UDA-CUTOUT \
    --name UDA_AutoAugment_FULL_Cutout_dropout_0.2 \
    >> UDA_AutoAugment_FULL_Cutout_dropout_0.2.log &
