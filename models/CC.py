import torch
import torch.nn as nn
from config import cfg


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()        
        
        if model_name == 'MobileCount':
            from MobileCount import MobileCount as net
        elif model_name == 'MobileCountx1_25':
            from MobileCountx1_25 import MobileCount as net
        elif model_name == 'MobileCountx2':
            from MobileCountx2 import MobileCount as net
        elif model_name == 'LSANet':
            from .LSANet import LSANet as net

        self.CCN = net()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).to(cfg.DEVICE)
        else:
            self.CCN=self.CCN.to(cfg.DEVICE)
        self.loss_mse_fn = nn.MSELoss().to(cfg.DEVICE)
        
    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

