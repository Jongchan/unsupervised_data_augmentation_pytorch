

#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN --batch-size 64 --batch-size-unsup 320 --l1-reg 0 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 400000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov --warmup-steps 20000 \
#    --dropout-rate 0.0 \
#    --UDA --cifar10-policy-all --UDA-CUTOUT \
#    --name UDA_AutoAugment_FULL_Cutout_no_dropout_400K_warmup_larger_batch \
#    >> UDA_AutoAugment_FULL_Cutout_l2_no_dropout_400K_warmup_larger_batch.log &
CUDA_VISIBLE_DEVICES=3 \
    python main.py \
    --normalization GCN --batch-size 64 --batch-size-unsup 320 --l1-reg 0 --l2-reg 0.0005 --final-lr 0.00012 --max-iter 400000 --lr 0.03 --optimizer SGD --lr-decay cosine --nesterov --warmup-steps 20000 \
    --dropout-rate 0.0 \
    --UDA --cifar10-policy-all --UDA-CUTOUT \
    --name UDA_AutoAugment_FULL_Cutout_no_dropout_400K_warmup_larger_batch_single_gpu \
    >> UDA_AutoAugment_FULL_Cutout_l2_no_dropout_400K_warmup_larger_batch_single_gpu.log &
