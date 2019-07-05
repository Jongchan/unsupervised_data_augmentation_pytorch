
# Settings from S4L paper. 200 epochs, base LR 0.1, LR decay at 140, 160, 180. Batch size unknown. weight decay 0.001
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet18 \
    --workers 20 \
    --batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.001 \
    --max-iter 100000 \
    --lr-drop-iter 70000 80000 90000 \
    --warmup --warmup-iter 2500 \
    --save_dir checkpoint/baseline_resnet18_S4L_bs256 \
    2>&1 | tee logs/baseline_resnet18_S4L_bs256 &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet18 \
    --workers 20 \
    --batch-size 512 \
    --lr 0.2 \
    --weight-decay 0.001 \
    --max-iter 100000 \
    --lr-drop-iter 70000 80000 90000 \
    --warmup --warmup-iter 2500 \
    --save_dir checkpoint/baseline_resnet18_S4L_bs512 \
    2>&1 | tee logs/baseline_resnet18_S4L_bs512 &
