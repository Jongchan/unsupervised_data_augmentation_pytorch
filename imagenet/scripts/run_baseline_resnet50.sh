#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#    python train_imagenet.py ./ImageNet/ \
#    --arch resnet50 \
#    --workers 20 \
#    --batch-size 512 \
#    --lr 0.1 \
#    --save_dir checkpoint/baseline_resnet50_bs512_lr_0.1

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#    python train_imagenet.py ./ImageNet/ \
#    --arch resnet50 \
#    --workers 20 \
#    --batch-size 512 \
#    --lr 0.2 \
#    --save_dir checkpoint/baseline_resnet50_bs512_lr_0.2

# Settings from S4L paper. 200 epochs, base LR 0.1, LR decay at 140, 160, 180. Batch size unknown. weight decay 0.001
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet50 \
    --workers 20 \
    --batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.001 \
    --max-iter 100000 \
    --lr-drop-iter 70000 80000 90000 \
    --warmup --warmup-iter 2500 \
    --save_dir checkpoint/baseline_resnet50_S4L_bs256 \
    >> logs/baseline_resnet50_S4L_bs256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet50 \
    --workers 20 \
    --batch-size 512 \
    --lr 0.1 \
    --weight-decay 0.001 \
    --max-iter 100000 \
    --lr-drop-iter 70000 80000 90000 \
    --warmup --warmup-iter 2500 \
    --save_dir checkpoint/baseline_resnet50_S4L_bs512 \
    >> logs/baseline_resnet50_S4L_bs512

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet50 \
    --workers 20 \
    --batch-size 512 \
    --lr 0.1 \
    --weight-decay 0.001 \
    --max-iter 50000 \
    --lr-drop-iter 35000 40000 45000 \
    --warmup --warmup-iter 2500 \
    --save_dir checkpoint/baseline_resnet50_S4L_bs512_50K \
    >> logs/baseline_resnet50_S4L_bs512_50K
