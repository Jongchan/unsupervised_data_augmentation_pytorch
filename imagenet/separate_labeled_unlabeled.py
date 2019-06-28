import glob
import random
import os
import torch



def separate_and_save_dataset(dataset_root, labeled_portion, seed=123):
    random.seed(seed)
    dataset_cls_dirs = glob.glob( os.path.join(dataset_root, '*') )

    labeled_images = {}
    unlabeled_images = {}

    for cls_idx, dataset_cls_dir in enumerate(sorted(dataset_cls_dirs)):
        cls_key = os.path.basename(dataset_cls_dir).replace('/','')
        print ("cls {}/{} cls_key {}".format(cls_idx, len(dataset_cls_dirs), cls_key))
    
        image_paths = glob.glob( os.path.join(dataset_cls_dir, '*') )
        random.shuffle(image_paths)
        print ("total {} images".format(len(image_paths)))
        labeled_count = int( len(image_paths) * labeled_portion )
    
        labeled_paths = image_paths[:labeled_count]
        unlabeled_paths = image_paths[labeled_count:]
    
        labeled_images[cls_key] = labeled_paths
        unlabeled_images[cls_key] = unlabeled_paths

    torch.save(labeled_images, 'data_split/labeled_images_{:.2f}.pth'.format(labeled_portion))
    torch.save(unlabeled_images, 'data_split/unlabeled_images_{:.2f}.pth'.format(1-labeled_portion))

if __name__=='__main__':
    separate_and_save_dataset('./ImageNet/train', 0.1, 123)
