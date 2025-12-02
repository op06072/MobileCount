import torch
import torch.nn as nn
from config import cfg
from .layer import NormalizedEuclideanLoss, LSALoss


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        if model_name == "MobileCountx1_25":
            from .MobileCountx1_25 import MobileCount as net
        elif model_name == "MobileCountx2":
            from .MobileCountx2 import MobileCount as net
        elif model_name == "LSANet":
            from .LSANet import LSANet as net
        elif model_name == "MobileCountV3Large":
            from .MobileCountV3 import MobileCountV3
            from functools import partial

            net = partial(MobileCountV3, backbone="large")
        elif model_name == "MobileCountV3Small":
            from .MobileCountV3 import MobileCountV3
            from functools import partial

            net = partial(MobileCountV3, backbone="small")
        elif model_name == "MobileCountV3Lite":
            from .MobileCountV3Lite import MobileCountV3Lite as net
        elif model_name == "MobileCountV4":
            from .MobileCountTimm import MobileCountTimm

            # Using partial to pass arguments since net() is called without args below
            from functools import partial

            net = partial(
                MobileCountTimm, model_name="mobilenetv4_conv_small.e2400_r224_in1k"
            )
        elif model_name == "MobileCountV5":
            from .MobileCountTimm import MobileCountTimm
            from functools import partial

            # Using a likely available V5 model name from timm
            net = partial(MobileCountTimm, model_name="mobilenetv5_300m.gemma3n")
        else:
            from .MobileCount import MobileCount as net

        self.cfg = cfg

        if model_name == "LSANet":
            self.CCN = net(bn=cfg.NET_BN, act=cfg.NET_ACT)
            net.apply(self, fn=self.init_weights)
        else:
            self.CCN = net()

        self.dev = cfg.DEVICE
        self.dev_acc = self.dev != torch.device("cpu")

        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus)
        else:
            self.CCN = self.CCN
        if self.dev_acc:
            self.CCN = self.CCN.to(self.dev)

        if model_name == "LSANet":
            # self.loss_mse_fn = NormalizedEuclideanLoss().to(cfg.DEVICE)
            self.loss_mse_fn = LSALoss()
        else:
            self.loss_mse_fn = nn.MSELoss(reduction="none")
        if self.dev_acc:
            self.loss_mse_fn = self.loss_mse_fn.to(self.dev)

    def compute_lc_loss(self, output, target, sizes=(1, 2, 4)) -> torch.Tensor:
        # criterion_L1 = torch.nn.L1Loss(reduction=self.cfg.L1_LOSS_REDUCTION)
        # self.cfg.L1_LOSS_REDUCTION not used anymore
        criterion_L1 = torch.nn.L1Loss(reduction="none")
        if self.dev_acc:
            criterion_L1 = criterion_L1.to(self.dev)
        lc_loss = None
        for s in sizes:
            pool = torch.nn.AdaptiveAvgPool2d(s)
            if self.dev_acc:
                pool = pool.to(self.dev)
            est = pool(output.unsqueeze(0))
            gt = pool(target.unsqueeze(0))
            c = criterion_L1(est, gt).squeeze(0)
            if c.ndim == 3:
                c_mean = c.mean(dim=(1, 2)) / s**2
            else:
                c_mean = c.mean() / s**2
            if lc_loss is not None:
                lc_loss += c_mean
                # lc_loss += criterion_L1(est, gt) / s**2
            else:
                lc_loss = c_mean
        return lc_loss

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)

    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse

    def forward(self, img, gt_map=None, sample_weight=None):
        density_map = self.test_forward(img)
        if gt_map is not None:
            self.loss_mse = self.build_loss(
                density_map.squeeze(), gt_map.squeeze(), sample_weight
            )
        return density_map

    def build_loss(self, density_map, gt_data, sample_weight=None):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        if loss_mse.ndim == 3:
            loss_mse = loss_mse.mean(dim=(1, 2))
        else:
            loss_mse = loss_mse.mean()
        self.lc_loss = 0
        computing_lc_loss = True
        if self.dev.type == "mps":
            h, w = density_map.shape[-2:]
            if any(h % s != 0 or w % s != 0 for s in self.cfg.CUSTOM_LOSS_SIZES):
                computing_lc_loss = False
        if self.cfg.CUSTOM_LOSS and computing_lc_loss:
            lc_loss = self.compute_lc_loss(
                density_map, gt_data, sizes=self.cfg.CUSTOM_LOSS_SIZES
            )
            self.lc_loss = lc_loss.sum()
            loss_mse += self.cfg.CUSTOM_LOSS_LAMBDA * lc_loss
        if sample_weight is not None:
            a = loss_mse * sample_weight
            loss_mse = a.sum() / sample_weight.sum()
        else:
            loss_mse = loss_mse.mean()
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        if density_map.dtype not in (torch.bfloat16, torch.float, torch.double):
            density_map = density_map.float()
        return density_map
