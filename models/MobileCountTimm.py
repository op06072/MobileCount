import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


# Helpers / wrappers
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


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


class MobileCountTimm(nn.Module):
    def __init__(
        self, model_name="mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True
    ):
        super(MobileCountTimm, self).__init__()

        if timm is None:
            raise ImportError(
                "Please install timm to use MobileCountTimm: pip install timm"
            )

        # Create backbone with features_only=True to get intermediate feature maps
        # out_indices=(1, 2, 3, 4) typically corresponds to strides 4, 8, 16, 32
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
        except RuntimeError as e:
            # Fallback for models that might have different indices
            print(
                f"Warning: Could not load {model_name} with indices (1,2,3,4). Trying default."
            )
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, features_only=True
            )

        # Get channel info
        feature_info = self.backbone.feature_info
        all_channels = [x["num_chs"] for x in feature_info]
        print(f"Model: {model_name}, Channels: {all_channels}")

        # Handle models that return 5 features (take last 4)
        if len(all_channels) == 5:
            print(f"Model returns 5 features, using last 4 for decoder")
            self.in_channels = all_channels[1:]  # Skip first, use [1,2,3,4]
        elif len(all_channels) == 4:
            self.in_channels = all_channels
        else:
            raise ValueError(
                f"Expected 4 or 5 feature levels, got {len(all_channels)}. Channels: {all_channels}"
            )

        # Decoder (RefineNet)
        # l4 -> x4
        self.dropout4 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl1_dimred = conv1x1(self.in_channels[3], 64, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(64, 64, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(64, 32, bias=False)

        # l3 -> x3
        self.dropout3 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl2_dimred = conv1x1(self.in_channels[2], 32, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(32, 32, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(32, 32, bias=False)

        # l2 -> x2
        self.p_ims1d2_outl3_dimred = conv1x1(self.in_channels[1], 32, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(32, 32, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(32, 32, bias=False)

        # l1 -> x1
        self.p_ims1d2_outl4_dimred = conv1x1(self.in_channels[0], 32, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(32, 32, 4)

        self.dropout_clf = nn.Dropout(p=0.5)
        self.clf_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # Initialize decoder weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight, std=0.01)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, x):
        size1 = x.shape[2:]

        # Extract features
        features = self.backbone(x)

        # Map to l1, l2, l3, l4 (handle 5 features by taking last 4)
        if len(features) == 5:
            l1 = features[1]
            l2 = features[2]
            l3 = features[3]
            l4 = features[4]
        else:  # len(features) == 4
            l1 = features[0]
            l2 = features[1]
            l3 = features[2]
            l4 = features[3]

        # Decoder
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
        x1 = F.silu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        x1 = self.dropout_clf(x1)
        out = self.clf_conv(x1)

        out = F.interpolate(out, size=size1, mode="bilinear")

        return out
