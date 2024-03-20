from __future__ import annotations

import os
import torch
import numpy as np

from config import cfg

from PIL.Image import Image
from datasets import DataDict
from misc.transforms import Compose
from torch.utils.data import DataLoader
from multiprocessing.managers import DictProxy
from typing import Callable, Any, Tuple, List, AnyStr

# from multiprocessing import freeze_support

# from trainer import Trainer
# from trainer_CMTL import Trainer_CMTL

if __name__ == '__main__':
    # freeze_support()

    # ------------prepare environment------------
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cfg.GPU_DEVICE == "cuda":
            torch.cuda.manual_seed(seed)
        elif cfg.GPU_DEVICE == "mps":
            torch.mps.manual_seed(seed)

    gpus = cfg.GPU_ID
    if len(gpus) == 1:
        if cfg.GPU_DEVICE == "cuda":
            torch.cuda.set_device(gpus[0])

    if cfg.GPU_DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    data_loader: Callable[
        [DataDict | DictProxy[str, List[Image]], int], Tuple[DataLoader[Any] | None, DataLoader[Any], Compose]
    ] | Callable[[], Tuple[DataLoader[Any] | None, DataLoader[Any], Compose]]

    # ------------prepare data loader------------
    data_mode = cfg.DATASET
    if data_mode == 'SHHA':
        from datasets.SHHA.loading_data import loading_data as data_loader
        from datasets.SHHA.setting import cfg_data
    elif data_mode == 'SHHB':
        from datasets.SHHB.loading_data import loading_data as data_loader
        from datasets.SHHB.setting import cfg_data
    elif data_mode == 'QNRF':
        from datasets.QNRF.loading_data import loading_data as data_loader
        from datasets.QNRF.setting import cfg_data
    elif data_mode == 'UCF50':
        from datasets.UCF50.loading_data import loading_data as data_loader
        from datasets.UCF50.setting import cfg_data
    elif data_mode == 'WE':
        from datasets.WE.loading_data import loading_data as data_loader
        from datasets.WE.setting import cfg_data
    elif data_mode == 'GCC':
        from datasets.GCC.loading_data import loading_data as data_loader
        from datasets.GCC.setting import cfg_data

    # ------------Prepare Trainer------------
    net = cfg.NET
    from trainer import Trainer

    # ------------Start Training------------
    pwd: str | bytes = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(data_loader, cfg_data, pwd)
    # cc_trainer.preload()
    cc_trainer.forward()
