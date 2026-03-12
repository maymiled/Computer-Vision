"""
models/unet.py
──────────────────────────────────────────────────────────────────────────────
U-Net adapted for binary image classification (Pneumonia vs Normal).

Original paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
                Ronneberger et al., 2015  —  https://arxiv.org/abs/1505.04597

Architecture overview
─────────────────────
                    Input (3, H, W)
                         │
              ┌──────────▼──────────┐
              │    Encoder blocks   │  (contracting path)
              │  conv → pool × 4   │
              └──────────┬──────────┘
                         │ bottleneck
              ┌──────────▼──────────┐
              │    Decoder blocks   │  (expanding path)
              │  up-conv × 4 with  │
              │    skip connections │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Global Avg Pool    │
              │  + Classifier head  │
              └──────────┬──────────┘
                         │
                  Output (1,)  — logit

Note: for classification we do NOT produce a segmentation map.
      The decoder is used to build rich multi-scale features before pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive (Conv → BN → ReLU) layers — the basic U-Net unit."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # first convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool2d — one step down the contracting path."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        """Returns (feature_map_before_pool, pooled_output)."""
        features = self.conv(x)
        pooled   = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Upsample → concatenate skip connection → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch before concatenating
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── Main model ─────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for binary classification.

    Args:
        in_channels  : number of input channels (3 for RGB)
        base_filters : number of filters in the first encoder block;
                       doubles at each level (e.g. 64 → 128 → 256 → 512)
        dropout_rate : dropout probability before the final linear layer
    """

    def __init__(
        self,
        in_channels:  int   = 3,
        base_filters: int   = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        f = base_filters

        # Encoder (contracting path) — 4 levels
        self.enc1 = EncoderBlock(in_channels, f)
        self.enc2 = EncoderBlock(f,      f * 2)
        self.enc3 = EncoderBlock(f * 2,  f * 4)
        self.enc4 = EncoderBlock(f * 4,  f * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)

        # Decoder (expanding path) — mirrors the encoder
        self.dec4 = DecoderBlock(f * 16, f * 8)
        self.dec3 = DecoderBlock(f * 8,  f * 4)
        self.dec2 = DecoderBlock(f * 4,  f * 2)
        self.dec1 = DecoderBlock(f * 2,  f)

        # Classification head
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(f, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, 1)  — apply sigmoid externally for probabilities
        """
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Pooling + classification
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = UNet(in_channels=3, base_filters=64)
    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print("Output shape:", output.shape)  # expected: (2, 1)
