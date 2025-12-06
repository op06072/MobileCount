from pathlib import Path

import h5py
import scipy.io as sio
import numpy as np
from PIL import Image

from .dynamics import CustomDataset


class CustomMALL(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        """
        Load Custom MALL
        """
        super().__init__()
        self.subset = "MALL"
        self.gt_name_folder = kwargs.get("MALL__gt_name_folder", "den")
        self.gt_format = kwargs.get("MALL__gt_format", ".mat")
        self.transform = kwargs.get("MALL__transform", None)
        self.dataset_weight = kwargs.get("MALL__dataset_weight", 1)

        print("dataset_weight:", self.dataset_weight)
        self.folder = Path(folder)
        self.mode = mode
        self.dataset = self.read_index()

    def read_index(self):
        """
        Read all images position in MALL Dataset
        """
        img_list = [
            f
            for f in (self.folder / f"{self.mode}" / "img").glob("*")
            if f.suffix not in ["txt", "zip"]
        ]
        gt_folder = self.folder / f"{self.mode}" / self.gt_name_folder
        json_data = {}
        for n, im in enumerate(img_list):
            filename = Path(im).stem
            gt_count = None
            json_data[n] = {
                "path_img": im,
                "path_gt": gt_folder / (f"{filename}_soft{self.gt_format}"),
                "gt_count": gt_count,
                "folder": self.folder,
                "sample_weight": self.dataset_weight,
            }
        df = pd.DataFrame.from_dict(json_data, orient="index")
        print(
            f"CustomMALL - subset:{self.subset} - mode:{self.mode} - df.shape:{df.shape}"
        )
        return df

    def load_gt(self, filename):
        """
        Load GT in np.array
        """
        density_map = None
        if Path(filename).suffix == ".mat":
            try:
                mat_data = sio.loadmat(str(filename))
                if "map" in mat_data:
                    density_map = mat_data["map"]
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        elif Path(filename).suffix == ".csv":
            # CSV loading (much faster than MAT)
            # density_map = pd.read_csv(filename, header=None).values.astype(np.float32)
            density_map = np.loadtxt(filename, delimiter=",").astype(np.float32, copy=False)
        elif Path(filename).suffix == ".h5":
            gt_file = h5py.File(filename, "r")
            density_map = np.asarray(gt_file["density"])

        if density_map is not None:
            self.check_density_map(density_map)
        return density_map
