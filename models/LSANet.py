import torch
from torch import nn


def conv3x3(in_planes, out_planes, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, dilation=dilation, padding=padding)


class LFE(nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.conv1 = conv3x3(3, 16)
        self.conv2 = conv3x3(16, 32)
        self.conv3 = conv3x3(32, 48)
        self.conv4 = conv3x3(48, 48)
        self.conv5 = conv3x3(48, 64)
        self.conv6 = conv3x3(64, 64)
        self.conv7 = conv3x3(64, 64)

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
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 16, 1, dilation=1)
        self.conv2 = conv3x3(16, 16, 1)
        self.conv3 = conv3x3(16, 16, 2, 2)
        self.conv4 = conv3x3(16, 16, 3, 3)

    def forward(self, x):
        outputs = [self.conv1(x)]
        for c in [self.conv2, self.conv3, self.conv4]:
            outputs.append(c(outputs[-1]))
        con = torch.cat(outputs, 1)
        return con + x


class EAF(nn.Module):
    def __init__(self):
        super().__init__()

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv1d(2, 2, 9, groups=2, padding=4)
        self.conv2 = nn.Conv2d(64, 64, 1)

    def forward(self, x, y):
        w1 = self.avgpool1(x)
        w2 = self.avgpool2(y)
        w1_a = torch.reshape(w1, w1.shape[:-1])
        w2_a = torch.reshape(w2, w2.shape[:-1])
        w = torch.cat([w1_a, w2_a], 2)
        w = self.conv1(torch.reshape(w, (w.shape[0], w.shape[2], w.shape[1])))
        w1, w2 = torch.tensor_split(torch.sigmoid(w).reshape(w.shape[0], w.shape[-1], w.shape[1], 1), 2, dim=2)
        f = w1 * x + w2 * y
        return self.conv2(nn.functional.normalize(f, dim=-1))


class DMR(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = conv3x3(64, 32)
        self.conv2 = conv3x3(16, 8)
        self.conv3 = conv3x3(4, 4)

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
    def __init__(self):
        super().__init__()

        self.lfe = LFE()

        self.hdc1 = HDC()
        self.hdc2 = HDC()
        self.hdc3 = HDC()

        self.eaf1 = EAF()
        self.eaf2 = EAF()
        self.eaf3 = EAF()

        self.dmr = DMR()

    def forward(self, x):
        f_lfe = self.lfe(x)
        f_hdc2 = self.hdc1(self.hdc1(f_lfe))
        f_hdc4 = self.hdc2(self.hdc2(f_hdc2))
        f_hdc6 = self.hdc3(self.hdc3(f_hdc4))
        f_eaf1 = self.eaf1(f_lfe, f_hdc2)
        f_eaf2 = self.eaf2(f_eaf1, f_hdc4)
        f_eaf3 = self.eaf3(f_eaf2, f_hdc6)
        return self.dmr(f_eaf3)
