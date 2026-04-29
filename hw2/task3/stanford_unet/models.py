"""Hand-written U-Net model for HW2 Task3."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two Conv-BatchNorm-ReLU blocks used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block."""
        return self.net(x)


class Down(nn.Module):
    """Downsampling block: max pooling followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one encoder downsampling step."""
        return self.net(x)


class Up(nn.Module):
    """Upsampling block with skip-connection concatenation."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            up_channels = out_channels
        self.conv = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features and concatenate encoder features."""
        x = self.up(x)
        x = _match_spatial_size(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """Final 1x1 convolution that maps features to class logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel class logits."""
        return self.conv(x)


class UNet(nn.Module):
    """Classic U-Net with four downsampling stages and skip connections."""

    def __init__(self, num_classes: int, base_channels: int = 32, bilinear: bool = False) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.inc = DoubleConv(3, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])
        self.up1 = Up(channels[4], channels[3], channels[3], bilinear)
        self.up2 = Up(channels[3], channels[2], channels[2], bilinear)
        self.up3 = Up(channels[2], channels[1], channels[1], bilinear)
        self.up4 = Up(channels[1], channels[0], channels[0], bilinear)
        self.outc = OutConv(channels[0], num_classes)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder, decoder and final classifier head."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def build_model(model_config: dict) -> nn.Module:
    """Construct the hand-written U-Net model from config."""
    if bool(model_config.get("pretrained", False)):
        raise ValueError("Task3 forbids pretrained weights.")
    name = str(model_config.get("name", "unet")).lower()
    if name != "unet":
        raise ValueError(f"Unsupported model: {name}")
    return UNet(
        num_classes=int(model_config.get("num_classes", 8)),
        base_channels=int(model_config.get("base_channels", 32)),
        bilinear=bool(model_config.get("bilinear", False)),
    )


def _match_spatial_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
