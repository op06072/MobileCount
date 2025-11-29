import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
# from misc.data import DataLoader
import misc.transforms as own_transforms
from .SHHB import SHHB
from .setting import cfg_data
import torch

from typing import Tuple, Any
from datasets import DataDict
from multiprocessing.managers import DictProxy


def loading_data(
        datas: DataDict | DictProxy,
        data_workers: int = 0
) -> Tuple[DataLoader[Any] | None, DataLoader[Any], own_transforms.Compose]:
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    train_main_transform = own_transforms.Compose([
    	own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = SHHB(
        cfg_data.DATA_PATH+'/train_data', main_transform=train_main_transform,
        img_transform=img_transform, gt_transform=gt_transform
    )
    train_set.setdict(datas)
    train_loader = DataLoader(
        train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=data_workers,
        shuffle=True, drop_last=True, persistent_workers=(data_workers != 0),
    )
    

    val_set = SHHB(
        cfg_data.DATA_PATH+'/test_data', main_transform=None,
        img_transform=img_transform, gt_transform=gt_transform
    )
    val_set.setdict(datas)
    val_loader = DataLoader(
        val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=data_workers,
        shuffle=True, drop_last=False, persistent_workers=(data_workers != 0),
    )

    return train_loader, val_loader, restore_transform
