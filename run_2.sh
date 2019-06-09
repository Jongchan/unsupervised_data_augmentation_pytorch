
#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN \
#    --l1-reg 0.0 \
#    --batch-si#ze 32 \
#    --lr 0.03 \
#    --final-lr 0.00012 \
#    --max-iter 100000 \
#    --UDA \
#    --UDA-CUTOUT \
#    --AutoAugment \
#    --optimizer SGD \
#    --lr-decay cosine \
#    --name autoaugment_code_base_UDA_SGD_COSINELR_AutoAugment_wd \
#    >> autoaugment_code_base_UDA_SGD_COSINELR_AutoAugment_wd.log &

#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN \
#    --l1-reg 0.0 \
#    --batch-size 32 \
#    --lr 0.03 \
#    --final-lr 0.00012 \
#    --max-iter 100000 \
#    --UDA \
#    --UDA-CUTOUT \
#    --optimizer SGD \
#    --lr-decay cosine \
#    --name autoaugment_code_base_UDA_SGD_COSINELR_wd \
#    >> autoaugment_code_base_UDA_SGD_COSINELR_wd.log &

#CUDA_VISIBLE_DEVICES=0,1 \
#    python main.py \
#    --normalization GCN \
#    --l1-reg 0.0 \
#    --l2-reg 0.0005 \
#    --batch-size 32 \
#    --lr 0.03 \
#    --final-lr 0.00012 \
#    --max-iter 100000 \
#    --UDA \
#    --UDA-CUTOUT \
#    --AutoAugment \
#    --optimizer SGD \
#    --lr-decay cosine \
#    --name autoaugment_code_base_UDA_SGD_COSINELR_AutoAugment_wd_0.0005 \
#    >> autoaugment_code_base_UDA_SGD_COSINELR_AutoAugment_wd_0.0005.log &
#CUDA_VISIBLE_DEVICES=2,3 \
#    python main.py \
#    --normalization GCN \
#    --l1-reg 0.0 \
#    --l2-reg 0.0005 \
#    --batch-size 32 \
#    --lr 0.03 \
#    --final-lr 0.00012 \
#    --max-iter 100000 \
#    --UDA \
#    --UDA-CUTOUT \
#    --optimizer SGD \
#    --lr-decay cosine \
#    --name autoaugment_code_base_UDA_SGD_COSINELR_wd_0.0005 \
#    >> autoaugment_code_base_UDA_SGD_COSINELR_wd_0.0005.log &

#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN \
#    --l1-reg 0.0 \
#    --l2-reg 0.0005 \
#    --batch-size 32 \
#    --lr 0.03 \
#    --final-lr 0.00012 \
#    --max-iter 100000 \
#    --UDA \
#    --UDA-CUTOUT \
#    --AutoAugment \
#    --optimizer SGD \
#    --nesterov \
#    --lr-decay cosine \
#    --name autoaugment_code_base_UDA_SGD_COSINELR_AutoAugment_wd_0.0005_nesterov \
#    >> autoaugment_code_base_UDA_SGD_COSINELR_AutoAugment_wd_0.0005_nesterov.log &
#CUDA_VISIBLE_DEVICES=6,7 \
#    python main.py \
#    --normalization GCN \
#    --l1-reg 0.0 \
#    --l2-reg 0.0005 \
#    --batch-size 32 \
#    --lr 0.03 \
#    --final-lr 0.00012 \
#    --max-iter 100000 \
#    --UDA \
#    --UDA-CUTOUT \
#    --optimizer SGD \
#    --nesterov \
#    --lr-decay cosine \
#    --name autoaugment_code_base_UDA_SGD_COSINELR_wd_0.0005_nesterov \
#    >> autoaugment_code_base_UDA_SGD_COSINELR_wd_0.0005_nesterov.log &

#CUDA_VISIBLE_DEVICES=4,5 \
#    python main.py \
#    --normalization GCN \
#    --l1-reg 0.0 \
#    --l2-reg 0.0005 \
#    --batch-size 32 \
#    --lr 0.03 \
#    --final-lr 0.00012 \
#    --max-iter 100000 \
#    --UDA \
#    --AutoAugment \
#    --optimizer SGD \
#    --nesterov \
#    --cifar10_policy_all \
#    --lr-decay cosine \
#    --name autoaugment_code_base_UDA_no_CUTOUT_SGD_COSINELR_AutoAugment_wd_0.0005_nesterov_all \
#    >> autoaugment_code_base_UDA_no_CUTOUT_SGD_COSINELR_AutoAugment_wd_0.0005_nesterov_all.log &




