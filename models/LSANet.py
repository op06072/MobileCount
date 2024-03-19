import torch
from torch import nn


def conv3x3(in_planes, out_planes, dilation=1, padding=1, bn=False, act=False):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=3, dilation=dilation, padding=padding)
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    if act:
        layers.append(nn.ELU())
    return nn.Sequential(*layers)


class LFE(nn.Module):
    def __init__(self, bn=False, act=False):
        super().__init__()

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.conv1 = conv3x3(3, 16, bn=bn, act=act)
        self.conv2 = conv3x3(16, 32, bn=bn, act=act)
        self.conv3 = conv3x3(32, 48, bn=bn, act=act)
        self.conv4 = conv3x3(48, 48, bn=bn, act=act)
        self.conv5 = conv3x3(48, 64, bn=bn, act=act)
        self.conv6 = conv3x3(64, 64, bn=bn, act=act)
        self.conv7 = conv3x3(64, 64, bn=bn, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.conv7(x)


class HDC(nn.Module):
    def __init__(self, bn=False, act=False):
        super().__init__()

        self.bn = nn.BatchNorm2d(16)
        self.act = nn.ELU()

        self.conv1 = nn.Conv2d(64, 16, 1, dilation=1)
        self.conv2 = conv3x3(16, 16, 1, bn=bn, act=act)
        self.conv3 = conv3x3(16, 16, 2, 2, bn=bn, act=act)
        self.conv4 = conv3x3(16, 16, 3, 3, bn=bn, act=act)

    def forward(self, x):
        outputs = [self.conv1(x)]
        if self.bn:
            outputs[-1] = self.bn(outputs[-1])
        if self.act:
            outputs[-1] = self.act(outputs[-1])
        for c in [self.conv2, self.conv3, self.conv4]:
            outputs.append(c(outputs[-1]))
        con = torch.cat(outputs, 1)
        return con + x


class EAF(nn.Module):
    def __init__(self, bn=False, act=False):
        super().__init__()

        self.bn = bn
        self.act = act

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv1d(2, 2, 9, groups=2, padding=4)
        self.conv2 = nn.Conv2d(64, 64, 1)

        self.batnorm1d = nn.BatchNorm1d(2)
        self.batnorm2d = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.normalize = nn.functional.normalize

    def forward(self, x, y):
        w1_a = self.avgpool1(x).squeeze(-1)
        w2_a = self.avgpool2(y).squeeze(-1)
        w = torch.cat([w1_a, w2_a], 2).transpose(-1, -2)
        w = self.conv1(w)
        if self.bn:
            w = self.batnorm1d(w)
        if self.act:
            w = self.elu(w)
        # w1, w2 = torch.tensor_split(torch.sigmoid(w).reshape(w.shape[0], w.shape[-1], w.shape[1], 1), 2, dim=2)
        w1, w2 = torch.tensor_split(w.transpose(-1, -2).unsqueeze(-1), 2, dim=2)
        f = w1 * x + w2 * y
        f = self.conv2(nn.functional.normalize(f, dim=-1))
        if self.bn:
            f = self.batnorm2d(f)
        if self.act:
            f = self.elu(f)
        return f


class DMR(nn.Module):
    def __init__(self, bn=False, act=False):
        super().__init__()

        self.conv1 = conv3x3(64, 32, bn=bn, act=act)
        self.conv2 = conv3x3(16, 8, bn=bn, act=act)
        self.conv3 = conv3x3(4, 4, bn=bn, act=act)

        self.deconv1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(8, 4, 2, 2)
        self.deconv3 = nn.ConvTranspose2d(4, 1, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv1(x)
        x = self.conv2(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        return self.deconv3(x)


class LSANet(nn.Module):
    def __init__(self, bn=False, act=False):
        super().__init__()

        self.lfe = LFE(bn, act)

        self.hdc1 = HDC(bn, act)
        self.hdc2 = HDC(bn, act)
        self.hdc3 = HDC(bn, act)

        self.eaf1 = EAF(bn, act)
        self.eaf2 = EAF(bn, act)
        self.eaf3 = EAF(bn, act)

        self.dmr = DMR(bn, act)

    def forward(self, x):
        f_lfe = self.lfe(x)
        f_hdc2 = self.hdc1(self.hdc1(f_lfe))
        f_hdc4 = self.hdc2(self.hdc2(f_hdc2))
        f_hdc6 = self.hdc3(self.hdc3(f_hdc4))
        f_eaf1 = self.eaf1(f_lfe, f_hdc2)
        f_eaf2 = self.eaf2(f_eaf1, f_hdc4)
        f_eaf3 = self.eaf3(f_eaf2, f_hdc6)
        return self.dmr(f_eaf3 + f_lfe)


if __name__ == "__main__":
    import sys, os
    from torchsummary import summary

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from config import cfg
    from datasets.SHHA.setting import cfg_data

    img_size = tuple([3] + list(cfg_data.STD_SIZE))

    model = LSANet(bn=True, act=True)
    model = model.to(cfg.DEVICE)
    summary(model, img_size)

    lfe = LFE(bn=True, act=True)
    lfe = lfe.to(cfg.DEVICE)
    summary(lfe, img_size)
    data = lfe(torch.rand(img_size).to(cfg.DEVICE).unsqueeze(0))
    hdc = HDC(bn=True, act=True)
    hdc = hdc.to(cfg.DEVICE)
    summary(hdc, data.squeeze(0).shape)
    lfe_data = data
    data = hdc(data)
    eaf = EAF(bn=True, act=True)
    eaf = eaf.to(cfg.DEVICE)
    summary(eaf, [lfe_data.squeeze(0).shape, data.squeeze(0).shape])

