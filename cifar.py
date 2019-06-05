from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import torchvision.transforms as transforms
import AutoAugment.autoaugment as autoaugment
import json

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, args, train=True, uda=False, normalize=False):

        #super(CIFAR10, self).__init__(root)
        self.normalize = normalize
        self.args= args
        self.root = os.path.expanduser('./data')
        self.uda = uda

        self.train = train

        if self.args.use_cutout:
            self.autoaugment = transforms.Compose([
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.ToPILImage(),
                ])
        elif self.args.UDA_CUTOUT:
            print ("USE UDA CUTOUT")
            self.autoaugment = transforms.Compose([
                autoaugment.CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.ToPILImage(),
                ])
        else:
            self.autoaugment = transforms.Compose([
                autoaugment.CIFAR10Policy(),
                ])

        if self.args.AutoAugment:
            self.autoaugment_labeled = transforms.Compose([
                autoaugment.CIFAR10Policy(),
                ])

        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        if self.train:
            with open('./data/cifar-10-batches-py/cifar_label_map_count_4000_index_0', 'r') as f:
                label_map_str = f.readlines()[0]
                label_map = json.loads(label_map_str)['values']
                label_map = [int(label) for label in label_map]
         
            if self.uda:
                self.data       = np.delete( self.data, label_map, axis=0 )
                self.targets    = None
            else:
                self.data       = np.take(self.data, label_map, axis=0)
                self.targets    = np.take(self.targets, label_map, axis=0)
 
        print ("loaded data count {}".format(len(self.data)))

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        '''
            Labeled data
                X_raw --> [common preproc (filp/random crop)] --> GCN --> X

            Unlabeled data
                X_raw --> [common preproc (filp/random crop)] --> GCN --> X
                X_raw --> AutoAugment --> [common preproc (filp/random crop)] --> GCN --> X_aug

            Test data
                X_raw --> GCN --> X

        '''
        index = index % len(self.data)

        if self.uda:
            # UNLABELED
            img_raw = self.data[index] # image with shape 32,32,3
            img_raw = Image.fromarray(img_raw) # PIL image

            img_uda = self.transform(self.autoaugment(img_raw)) # torch tensor shape 3,32,32
            img = self.transform(img_raw) # torch tensor shape 3,32,32
            return img.type(torch.FloatTensor), img_uda.type(torch.FloatTensor)
        else:

            img, target = self.data[index], self.targets[index] # image with shape 32,32,3
            img = Image.fromarray(img) # PIL image
            if self.train:#LABELED
                if self.args.AutoAugment:
                    img = self.autoaugment_labeled(img)
                img = self.transform(img) # torch tensor shape 3,32,32
            else:#TEST
                img = transforms.ToTensor()(img)
            return img.type(torch.FloatTensor), target


        '''
        if self.uda:
            img = self.data[index] # image with shape 32,32,3
        else:
            img, target = self.data[index], self.targets[index] # image with shape 32,32,3
        img = Image.fromarray(img) # PIL image

        if self.train:

            if self.uda:
                # apply UDA
                img_uda = self.autoaugment(img) # torch tensor shape 3,32,32
                img_uda = self.transform(img_uda)

            img = self.transform(img) # torch tensor shape 3,32,32
            #img.clamp_(0.0, 1.0) # remove overflow due to gaussian noise
            
            if self.uda:
                return img.type(torch.FloatTensor), img_uda.type(torch.FloatTensor)
        else:
            img = transforms.ToTensor()(img)

        return img.type(torch.FloatTensor), target
        '''

    def __len__(self):
        if self.train:
            if self.uda:
                return self.args.eval_iter * self.args.batch_size_unsup
            else:
                return self.args.eval_iter * self.args.batch_size
        else:
            return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
        }
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',         type=int,       default=100)
    parser.add_argument('--eval-iter',          type=int,       default=10000)
    parser.add_argument('--batch-size-unsup',   type=int,       default=960)
    parser.add_argument('--gaussian-noise-level',type=float,    default=0.15)
    args = parser.parse_args()

    cifar10_unnormalize = CIFAR10(args, False, False, normalize=False)
    cifar10_normalize = CIFAR10(args, False, False, normalize=True)



    loader_normalize = torch.utils.data.DataLoader( cifar10_normalize,   batch_size=args.batch_size, shuffle=False,   num_workers=1, pin_memory=True )
    loader_unnormalize = torch.utils.data.DataLoader( cifar10_unnormalize,   batch_size=args.batch_size, shuffle=False,   num_workers=1, pin_memory=True )

    batch_normalize = next(iter(loader_normalize))[0]
    batch_unnormalize = next(iter(loader_unnormalize))[0]

    from train_semi_3 import global_contrast_normalize, ZCA
    zca_params = torch.load('./zca_params.pth')
    zca = ZCA(zca_params)
    batch_unnormalize_gcn = global_contrast_normalize( batch_unnormalize )
    import ipdb;ipdb.set_trace()


