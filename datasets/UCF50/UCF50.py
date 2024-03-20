from __future__ import annotations

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data

from datasets import DataDict
from multiprocessing.managers import DictProxy


class UCF50(data.Dataset):
    def __init__(self, data_path, folder, main_transform=None, img_transform=None, gt_transform=None):
        self.datas = None
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'

        self.img_files = []
        self.gt_files = []
        for i_folder in folder:
            folder_img = self.img_path + '/' + str(i_folder)
            folder_gt = self.gt_path + '/' + str(i_folder)
            for filename in os.listdir(folder_img):
                if os.path.isfile(os.path.join(folder_img, filename)):
                    self.img_files.append(folder_img + '/' + filename)
                    self.gt_files.append(folder_gt + '/' + filename.split('.')[0] + '.csv')

        self.num_samples = len(self.img_files)

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def setdict(self, datas: DataDict | DictProxy):
        self.datas = datas

    def __getitem__(self, index):
        fname = self.img_files[index]
        if fname not in self.datas:
            img, den = self.read_image_and_gt(index)
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

    def read_image_and_gt(self, index):
        img = Image.open(self.img_files[index])
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(self.gt_files[index], sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def get_num_samples(self):
        return self.num_samples
