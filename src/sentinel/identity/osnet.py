"""OSNet architecture (Zhou et al., ICCV 2019), implemented directly.

Why not import ``torchreid``: the package on PyPI under that name is an
abandoned 2019 build, and the maintained ``deep-person-reid`` is a source
install carrying old pins that do not resolve on modern Python. For an
air-gapped system whose whole point is that it runs offline on one machine,
vendoring ~150 lines of well-documented architecture is a better trade than
inheriting a dependency tree that may not install on the demo box.

Weights are the author's own MIT-licensed release
(huggingface.co/kaiyangzhou/osnet), trained on MSMT17. Nothing is trained here.

The design idea worth knowing: each block runs four parallel streams of
depthwise-separable 3x3 convolutions at receptive-field scales 3, 5, 7 and 9,
then fuses them through a learned channel gate. That "omni-scale" fusion is
why a 2.2 M-parameter network competes with much larger re-ID backbones, and
why it is the right speed/accuracy point for real-time use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Conv + BN + ReLU."""

    def __init__(self, in_c, out_c, k, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv1x1Linear(nn.Module):
    """1x1 conv + BN with no activation (used before a residual add)."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 pointwise followed by 3x3 depthwise -- the cheap 3x3 substitute."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1,
                               bias=False, groups=out_c)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))), inplace=True)


class ChannelGate(nn.Module):
    """Aggregation gate: learns per-channel weights to fuse the four streams."""

    def __init__(self, in_c, reduction=16):
        super().__init__()
        mid = max(1, in_c // reduction)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, mid, 1, bias=True)
        self.fc2 = nn.Conv2d(mid, in_c, 1, bias=True)

    def forward(self, x):
        g = self.global_avgpool(x)
        g = F.relu(self.fc1(g), inplace=True)
        return x * torch.sigmoid(self.fc2(g))


class OSBlock(nn.Module):
    """Omni-scale residual block: four receptive-field scales, gated fusion."""

    def __init__(self, in_c, out_c, reduction=4):
        super().__init__()
        mid = out_c // reduction
        self.conv1 = Conv1x1(in_c, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_c)
        self.downsample = Conv1x1Linear(in_c, out_c) if in_c != out_c else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity, inplace=True)


class OSNet(nn.Module):
    """OSNet backbone producing an L2-normalisable appearance embedding."""

    def __init__(self, blocks=(2, 2, 2), channels=(16, 64, 96, 128),
                 feature_dim=512, num_classes=4101):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            OSBlock(c[0], c[1]), OSBlock(c[1], c[1]),
            nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2)),
        )
        self.conv3 = nn.Sequential(
            OSBlock(c[1], c[2]), OSBlock(c[2], c[2]),
            nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2)),
        )
        self.conv4 = nn.Sequential(OSBlock(c[2], c[3]), OSBlock(c[3], c[3]))
        self.conv5 = Conv1x1(c[3], c[3])

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True)
        )
        # Present so the released checkpoint loads cleanly. Never used at
        # inference: this is an open-set system, and a closed-set classifier
        # head over 4101 MSMT17 identities is exactly what we do not want.
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def featuremaps(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

    def forward(self, x):
        v = self.global_avgpool(self.featuremaps(x))
        v = v.view(v.size(0), -1)
        return self.fc(v)


def osnet_x0_25(num_classes: int = 4101) -> OSNet:
    return OSNet(blocks=(2, 2, 2), channels=(16, 64, 96, 128),
                 feature_dim=512, num_classes=num_classes)


def load_osnet(weights_path, device="cpu") -> OSNet:
    """Load the released checkpoint, tolerating its wrapper keys."""
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    n_cls = sd["classifier.weight"].shape[0] if "classifier.weight" in sd else 4101
    model = osnet_x0_25(num_classes=n_cls)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"OSNet checkpoint is missing {len(missing)} tensors "
            f"(first: {missing[:3]}). The architecture does not match the weights."
        )
    model.eval().to(device)
    return model
