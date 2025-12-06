from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from PIL import Image

from .dynamics import CustomDataset


class CustomSHH(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        """
        Load Custom SHH
        """
        super().__init__()
        self.subset = ''
        if '_A' in folder:
            self.subset = 'SHHA'
            self.gt_name_folder = kwargs.get('SHHA__gt_name_folder', 'den')
            self.gt_format = kwargs.get('SHHA__gt_format', '.csv')
            self.transform = kwargs.get('SHHA__transform', None)
            self.dataset_weight = kwargs.get('SHHA__dataset_weight', 1)
        elif '_B' in folder:
            self.subset = 'SHHB'
            self.gt_name_folder = kwargs.get('SHHB__gt_name_folder', 'den')
            self.gt_format = kwargs.get('SHHB__gt_format', '.csv')
            self.transform = kwargs.get('SHHB__transform', None)
            self.dataset_weight = kwargs.get('SHHB__dataset_weight', 1)
        else:
            raise ValueError('Choose a path with SHH part A or SHH part B')
        print('dataset_weight:', self.dataset_weight)
        self.folder = Path(folder)
        self.mode = mode
        self.dataset = self.read_index()

    def read_index(self):
        """
        Read all images position in SSHB Dataset
        """
        img_list = (self.folder / f'{self.mode}_data' / 'img').glob('*')
        gt_folder = self.folder / f'{self.mode}_data' / self.gt_name_folder
        json_data = []
        for im in img_list:
            if im.suffix not in ['txt', 'zip']:
                filename = Path(im).stem
                gt_count = None
                json_data.append({
                    "path_img": im,
                    "path_gt": gt_folder / (filename + self.gt_format),
                    "gt_count": gt_count,
                    "folder": self.folder,
                    "sample_weight": self.dataset_weight,
                })
        # df = pd.DataFrame.from_dict(json_data, orient='index')
        print(f'CustomSHH - subset:{self.subset} - mode:{self.mode} - df.shape:{len(json_data)}x5')
        # return df
        return json_data

    def load_gt(self, filename):
        """
        Load GT in np.array
        """
        density_map = None
        if Path(filename).suffix == '.npz':
            density_map = load_npz(filename).toarray()
        elif Path(filename).suffix == '.h5':
            gt_file = h5py.File(filename)
            density_map = np.asarray(gt_file['density'])
        elif Path(filename).suffix == '.csv':
            density_map = pd.read_csv(filename, sep=',', header=None).values
            density_map = density_map.astype(np.float32, copy=False)
            # density_map = Image.fromarray(density_map)

        self.check_density_map(density_map)
        return density_map