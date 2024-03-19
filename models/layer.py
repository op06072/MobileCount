import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False,
                 dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class NormalizedEuclideanLoss(nn.Module):
    def __init__(self):
        super(NormalizedEuclideanLoss, self).__init__()

    @staticmethod
    def forward(inputs, targets):
        dim = targets.ndim
        y_n = torch.sum(targets, dim=tuple(range(dim)[1:]) if dim > 3 else None).sqrt()
        return torch.square(inputs / y_n - targets / y_n).sum() / (inputs.shape[0] if dim > 3 else 1)


class MIoU(nn.Module):
    def __init__(self):
        super(MIoU, self).__init__()

    @staticmethod
    def forward(inputs, targets):
        m_iou = 0
        for i in range(4):
            y_min = torch.fmin(inputs, targets)
            iou = 1 - y_min.sum() / torch.sum(inputs + targets - y_min)
            m_iou += iou
            t_size = inputs.shape
            p_size = targets.shape
            sizes = [t_size[-2], t_size[-1], p_size[-2], p_size[-1]]
            for j, k in enumerate(sizes):
                if k % 2 != 0:
                    sizes[j] += 1
                sizes[j] //= 2
            inputs = nn.AdaptiveAvgPool2d((sizes[0], sizes[1]))(inputs.unsqueeze(0)).squeeze(0)
            targets = nn.AdaptiveAvgPool2d((sizes[2], sizes[3]))(targets.unsqueeze(0)).squeeze(0)
        return m_iou / 4


class LSALoss(nn.Module):
    def __init__(self):
        super(LSALoss, self).__init__()

    @staticmethod
    def forward(inputs, targets):
        return NormalizedEuclideanLoss.forward(inputs, targets) + MIoU.forward(inputs, targets)
