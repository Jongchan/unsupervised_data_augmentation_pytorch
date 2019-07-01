import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import random 
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

'''
def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images
'''


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def load_db(db_path, class_to_idx):
    db = torch.load(db_path)
    images = []
    for key in sorted(db.keys()):
        for image_path in db[key]:
            images.append( (image_path, class_to_idx[key]) )
    return images

from autoaugment import ImageNetPolicy
class ImageNet(data.Dataset):

    def __init__(self, root, args, transform=None, target_transform=None,
                 loader=default_loader, db_path='./data_split/labeled_images_0.10.pth', is_unlabeled=False):
        classes, class_to_idx = find_classes(root)
        #imgs = make_dataset(root, class_to_idx)
        imgs = load_db(db_path, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_unlabeled = is_unlabeled
        self.autoaugment = ImageNetPolicy()

        indices = [i for i in range(len(imgs))]
        self.random_indices = []
        self.total_train_count = args.batch_size * args.max_iter
        for _ in range(self.total_train_count // len(indices)):
            self.random_indices += indices
        remainder_count = self.total_train_count - len(self.random_indices)
        random.shuffle(indices)
        self.random_indices += indices[:remainder_count]
        random.shuffle(self.random_indices)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        random_index = self.random_indices[index]
        path, target = self.imgs[random_index]
        img = self.loader(path)

        if self.is_unlabeled:
            aug_img = self.autoaugment(img)
            aug_img = self.transform(aug_img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_unlabeled:
            return img, aug_img
        else:
            return img, target

    def __len__(self):
        return self.total_train_count
