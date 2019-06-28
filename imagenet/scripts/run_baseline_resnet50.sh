CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet50 \
    --workers 20 \
    --batch-size 128 \
    --lr 0.05 \
    --save_dir checkpoint/baseline_resnet50_bs128 \
    >> logs/baseline_resnet50_bs128.log &&

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet50 \
    --workers 20 \
    --batch-size 64 \
    --lr 0.025 \
    --save_dir checkpoint/baseline_resnet50_bs64 \
    >> logs/baseline_resnet50_bs64.log &&

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet50 \
    --workers 20 \
    --batch-size 32 \
    --lr 0.0125 \
    --save_dir checkpoint/baseline_resnet50_bs32 \
    >> logs/baseline_resnet50_bs32.log &
