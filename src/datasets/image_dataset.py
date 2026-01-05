import numpy as np
import h5py
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from src.transforms.functional import rgb2ycbcr


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class FastImageFolder(Dataset):
    def __init__(self, datablock, transform=None, yuv420=False):
        self.datablock = datablock
        self.transform = transform
        self.yuv420 = yuv420

    def __getitem__(self, index):
        img = self.datablock[index]
        img = Image.fromarray(np.uint8(img))
        if self.transform:
            img = self.transform(img)
            if self.yuv420:
                img = rgb2ycbcr(img)
            return img
        
        return img

    def __len__(self):
        return len(self.datablock)


def LoadAllImg(h5f_name):
    print("++++++++")
    h5f = h5py.File(h5f_name, 'r')
    print("--------")
    samples = []
    cnt = 0
    for key in list(h5f.keys()):
        samples.append(np.array(h5f[key]))
        cnt += 1
        if cnt % 100 == 0:
            print(f'{cnt} images loaded')
    h5f.close()
    return samples
