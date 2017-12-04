import os
from torch.utils.data import Dataset, DataLoader
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
    
    
class UnNormalize(object):
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
#         for i in range(tensor.shape[0]):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
