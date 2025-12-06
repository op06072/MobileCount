from __future__ import annotations

import os
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
from functools import cache
from torch.utils import data
from pathlib import Path

from datasets import DataDict
from multiprocessing.managers import DictProxy

class MALL(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None):
        self.data = None
        self.img_path = Path(data_path) / 'img'
        self.gt_path = Path(data_path) / 'den'
        self.data_files = list(self.img_path.glob('*.*'))
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def setdict(self, datas: DataDict | DictProxy):
        self.datas = datas

    def __getitem__(self, index):
        fname = self.data_files[index]
        if fname not in self.datas:
            img, den = self.read_image_and_gt(fname)
            self.datas[fname] = [img, den]
        else:
            img, den = self.datas[fname]
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    @cache
    def read_image_and_gt(self, fname):
        img = Image.open(self.img_path / fname)
        if img.mode == 'L':
            img = img.convert('RGB')

        file_accuracy = 'hard' # 'soft' means that the gt value is not perfectly fit with the number of people and 'hard' means the opposite
        den = sio.loadmat(self.gt_path / f'{fname.stem}_{file_accuracy}.mat')['map']
        # den = np.loadtxt(self.gt_path /  f'{fname.stem}_{file_accuracy}.csv', delimiter=',')

        den = den.astype(np.float32, copy=False)
        if file_accuracy == 'hard':
            den /= 1e4
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
