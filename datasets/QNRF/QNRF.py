from __future__ import annotations

import os
import numpy as np
from PIL import Image
from scipy import io as sio
from torch.utils import data

import pandas as pd

from datasets import DataDict
from multiprocessing.managers import DictProxy


class QNRF(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
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

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
