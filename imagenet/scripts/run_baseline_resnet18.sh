
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet18 \
    --workers 20 \
    --batch-size 256 \
    --lr 0.1 \
    >> logs/baseline_resnet18_bs256.log &&

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet18 \
    --workers 20 \
    --batch-size 128 \
    --lr 0.05 \
    >> logs/baseline_resnet18_bs128.log &&

CUDA_VISIBLE_DEVICES=0,1 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet18 \
    --workers 20 \
    --batch-size 64 \
    --lr 0.025 \
    >> logs/baseline_resnet18_bs64.log &

CUDA_VISIBLE_DEVICES=2,3 \
    python train_imagenet.py ./ImageNet/ \
    --arch resnet18 \
    --workers 20 \
    --batch-size 32 \
    --lr 0.0125 \
    >> logs/baseline_resnet18_bs32.log 
