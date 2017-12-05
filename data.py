import os
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import numpy as np
import torch

# TODO create a class for each dataset EXCEPT MNIST (this is already built into pytorch)
# If the dataset comes pre-split into train/test we should write a separate class for each.

# For example the street view house number might look something like this...
# The cropped version of the dataset is in a weird *.mat format, see https://stackoverflow.com/questions/29185493/read-svhn-dataset-in-python for instructions to load with numpy
class SVHNDataset(Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`
    Args:
        data_dir (string): directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    filename = ""
    filepath = ""
    split_list = {
        'train': "train_32x32.mat",
        'test': "test_32x32.mat",
        'extra': "extra_32x32.mat"}

    def __init__(self, data_dir='./datasets', split='train',
                 transform=None, target_transform=None, download=False):
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.filename = self.split_list[split]
        self.filepath = os.path.join(self.data_dir, self.filename)
        
        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio
        
        
        # judge if .mat exist
        if not os.path.isfile(self.filepath):
            raise RuntimeError('Dataset not found or corrupted.' +
                ' You can use fetch_data.sh to download it')
        
        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(self.filepath)

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.data)
    
class MNIST_Transform(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, sample):
        #rint(sample.shape)
        #ample_3 = torch.cat((sample, sample, sample), 1)
        return sample 

class EmojiDataset(Dataset):
    '''
    Dataset of 1 million bitmoji images.
    start_idx - image number dataset should start at
    end_idx - data number where dataset ends
    '''
    def __init__(self, data_dir, start_idx=0, end_idx=1000000, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_len = end_idx - start_idx
    
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        """
        img_name = os.path.join(self.data_dir, 'emoji_{}.png'.format(idx))
        img = Image.open(img_name)
        img = img.convert('RGB') # b/c it's a png

        if self.transform is not None:
            img = self.transform(img)
                                   
        return img

    def __len__(self):
        return self.data_len    

class CelebADataset(Dataset):
    '''
    CelebA face image dataset. This is the aligned and cropped version. 
    data_dir - directory of image data
    ann_dir - directory of annotation data
    split - either 'train', 'eval', or 'test'
    '''
    def __init__(self, data_dir, ann_dir, split, transform=None):
                
        data_splits = ['train', 'eval', 'test']
        self.data_dir = data_dir
        self.transform = transform
        
        split = data_splits.index(split)
        split_data = []
        with open(os.path.join(ann_dir, 'list_eval_partition.txt')) as split_file:
            reader = csv.reader(split_file, delimiter=' ')
            for row in reader:
                split_data.append(row)
        bbox_data = []
        with open(os.path.join(ann_dir, 'list_bbox_celeba.txt')) as bbox_file:
            reader = csv.reader(bbox_file, delimiter=' ', skipinitialspace=True)
            test_row = next(reader) # header row
            test_row = next(reader) # header row
            for row in reader:
                bbox_data.append(row)
                
        split_data = np.array(split_data)
        bbox_data = np.array(bbox_data)
        split_inds = np.where(split_data[:,1] == str(split))[0]
        
        self.split_info = split_data[split_inds, :]
        self.bbox_info = bbox_data[split_inds, :]
        self.data_len = self.split_info.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        """
        img_name = os.path.join(self.data_dir, self.split_info[idx, 0])
        img = Image.open(img_name)
        
        if self.transform is not None:
            img = self.transform(img)
                           
        return img

    def __len__(self):
        return self.data_len
    
class MSCeleb1MDataset(Dataset):
    '''
    MS-Celeb-1M face image dataset. This is the aligned and cropped version. 
    data_dir - directory of data. This directory should contain annotation files and a subdirectory for image data.
    split - either 'train' or 'test'
    '''
    def __init__(self, data_dir, split, transform=None):
        data_splits = ['train', 'test']
        self.transform = transform
        
        split = data_splits.index(split)
        if split == 0:
            info_path = 'train_data_info.txt'
            self.data_path = os.path.join(data_dir, 'images_train/')
        elif split == 1:
            info_path = 'test_data_info.txt'
            self.data_path = os.path.join(data_dir, 'images_test/')
        
        info_data = []
        with open(os.path.join(data_dir, info_path)) as info_file:
            reader = csv.reader(info_file, delimiter=' ')
            for row in reader:
                info_data.append(row)
                
        self.info = np.array(info_data)
        self.data_len = self.info.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        """
        img_name = os.path.join(self.data_path, self.info[idx, 0])
        img = Image.open(img_name)
        
        if self.transform is not None:
            img = self.transform(img)
                       
        return img

    def __len__(self):
        return self.data_len
    
class ResizeTransform(object):
    ''' Resizes a PIL image to (size, size) to feed into OpenFace net and returns a torch tensor.'''
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):
        img = sample.resize((self.size, self.size), Image.BILINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)
    
class UnNormalize(object):
    ''' from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
