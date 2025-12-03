"""MobileCountV3Lite - Lightweight custom MobileNetV3-inspired architecture.
Matches the original MobileCount paper's philosophy with V3 improvements."""

import torch.nn as nn
import torch.nn.functional as F


# Helpers / wrappers
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block from MobileNetV3"""

    def __init__(self, in_chs, se_ratio=0.25):
        super(SqueezeExcite, self).__init__()
        reduced_chs = max(1, int(in_chs * se_ratio))
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * x_se.sigmoid()


class InvertedResidualV3(nn.Module):
    """MobileNetV3-style Inverted Residual block with SE"""

    def __init__(
        self, inplanes, planes, stride=1, expansion=6, se_ratio=0.25, downsample=None
    ):
        super(InvertedResidualV3, self).__init__()
        hidden_dim = inplanes * expansion
        self.use_se = se_ratio is not None and se_ratio > 0
        self.stride = stride
        self.downsample = downsample

        # Expansion
        self.conv1 = nn.Conv2d(inplanes, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim, momentum=0.05)

        # Depthwise
        self.conv2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim, momentum=0.05)

        # Squeeze-and-Excite
        if self.use_se:
            self.se = SqueezeExcite(hidden_dim, se_ratio=se_ratio)

        # Projection
        self.conv3 = nn.Conv2d(hidden_dim, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=0.05)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x

        # Expansion
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        # Depthwise
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        # SE
        if self.use_se:
            out = self.se(out)

        # Projection
        out = self.conv3(out)
        out = self.bn3(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(x)

        if self.stride == 1 and residual.shape == out.shape:
            out += residual

        return out


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes,
                    stride=1,
                    bias=False,
                ),
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = top + x
        return x


class MobileCountV3Lite(nn.Module):
    """Lightweight MobileNetV3-inspired architecture for crowd counting.

    Matches the original MobileCount paper's compact structure [1, 2, 3, 4] blocks
    with MobileNetV3 improvements (SE modules, better activations).
    """

    def __init__(self, num_classes=1, pretrained=False):
        self.inplanes = 32
        block = InvertedResidualV3
        layers = [1, 2, 3, 4]  # Same as original MobileCount
        super(MobileCountV3Lite, self).__init__()

        # Stem (same as original)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act = nn.SiLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder stages (same structure as original, but V3 blocks)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, expansion=6)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, expansion=6)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, expansion=6)

        # Decoder (RefineNet - same as original)
        self.dropout4 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl1_dimred = conv1x1(256, 64, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(64, 64, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(64, 32, bias=False)

        self.dropout3 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl2_dimred = conv1x1(128, 32, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(32, 32, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(32, 32, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(64, 32, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(32, 32, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(32, 32, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(32, 32, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(32, 32, 4)

        self.dropout_clf = nn.Dropout(p=0.5)
        self.clf_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride, expansion):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                expansion=expansion,
                downsample=downsample,
            )
        ]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        size1 = x.shape[2:]

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        # Encoder
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # Decoder (RefineNet)
        l4 = self.dropout4(l4)
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = F.silu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear")(x4)

        l3 = self.dropout3(l3)
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.silu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear")(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.silu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear")(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        x1 = self.dropout_clf(x1)
        out = self.clf_conv(x1)

        out = F.interpolate(out, size=size1, mode="bilinear")

        return out
