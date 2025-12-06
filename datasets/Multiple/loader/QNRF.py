from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import load_npz
from PIL import Image

from .dynamics import CustomDataset


class CustomQNRF(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        """
        Load Custom QNRF
        """
        super().__init__()
        self.subset = "QNRF"
        self.gt_name_folder = kwargs.get("QNRF__gt_name_folder", "den")
        self.gt_format = kwargs.get("QNRF__gt_format", ".csv")
        self.transform = kwargs.get("QNRF__transform", None)
        self.dataset_weight = kwargs.get("QNRF__dataset_weight", 1)

        print("dataset_weight:", self.dataset_weight)
        self.folder = Path(folder)
        self.mode = mode
        self.dataset = self.read_index()

    def read_index(self):
        """
        Read all images position in QNRF Dataset
        """
        # QNRF folder structure might be different, but assuming standard train/test split
        # If mode is 'train', folder is 'train_data'
        # If mode is 'test', folder is 'test_data'

        # Adjusting for QNRF specific folder names if necessary
        # Usually QNRF has 'Train' and 'Test' folders, but let's stick to the pattern or adapt
        # If the user follows the pattern:
        # root/train/images
        # root/test/images

        # However, standard QNRF is:
        # Train/
        # Test/

        # Let's assume the data has been prepared in the standard structure used by this repo:
        # train/images
        # test/images

        img_list = (self.folder / f"{self.mode}" / "img").glob("*")
        gt_folder = self.folder / f"{self.mode}" / self.gt_name_folder
        json_data = []
        for im in img_list:
            if im.suffix not in ["txt", "zip"]:
                filename = Path(im).stem
                gt_count = None
                json_data.append({
                    "path_img": im,
                    "path_gt": gt_folder / (filename + self.gt_format),
                    "gt_count": gt_count,
                    "folder": self.folder,
                    "sample_weight": self.dataset_weight,
                })
        # df = pd.DataFrame.from_dict(json_data, orient="index")
        print(
            f"CustomQNRF - subset:{self.subset} - mode:{self.mode} - df.shape:{len(json_data)}x5"
        )
        # return df
        return json_data

    def load_gt(self, filename):
        """
        Load GT in np.array
        """
        density_map = None
        if Path(filename).suffix == ".npz":
            density_map = load_npz(filename).toarray()
        elif Path(filename).suffix == ".h5":
            gt_file = h5py.File(filename)
            density_map = np.asarray(gt_file["density"])
        elif Path(filename).suffix == ".csv":
            density_map = np.loadtxt(
                filename, delimiter=","
            ).astype(np.float32, copy=False)
            # density_map = Image.fromarray(density_map)

        self.check_density_map(density_map)
        return density_map
