import torch
import torch.nn as nn
from config import cfg
from .layer import NormalizedEuclideanLoss, LSALoss


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        if model_name == 'MobileCountx1_25':
            from .MobileCountx1_25 import MobileCount as net
        elif model_name == 'MobileCountx2':
            from .MobileCountx2 import MobileCount as net
        elif model_name == 'LSANet':
            from .LSANet import LSANet as net
        else:
            from .MobileCount import MobileCount as net

        if model_name == 'LSANet':
            self.CCN = net(bn=cfg.NET_BN, act=cfg.NET_ACT)
            net.apply(self, fn=self.init_weights)
        else:
            self.CCN = net()

        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).to(cfg.DEVICE)
        else:
            self.CCN = self.CCN.to(cfg.DEVICE)

        if model_name == 'LSANet':
            # self.loss_mse_fn = NormalizedEuclideanLoss().to(cfg.DEVICE)
            self.loss_mse_fn = LSALoss().to(cfg.DEVICE)
        else:
            self.loss_mse_fn = nn.MSELoss().to(cfg.DEVICE)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)

    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
