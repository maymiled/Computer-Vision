"""
models/resnet.py
──────────────────────────────────────────────────────────────────────────────
ResNet adapted for binary image classification (Pneumonia vs Normal).

Original paper: "Deep Residual Learning for Image Recognition"
                He et al., 2015  —  https://arxiv.org/abs/1512.03385

Architecture overview
─────────────────────
                    Input (3, H, W)
                         │
              ┌──────────▼──────────┐
              │  Stem (7×7 conv)    │
              │  + MaxPool          │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Layer 1  (×2)     │  64  filters, stride 1
              │   Layer 2  (×2)     │  128 filters, stride 2
              │   Layer 3  (×2)     │  256 filters, stride 2
              │   Layer 4  (×2)     │  512 filters, stride 2
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Global Avg Pool    │
              │  + Classifier head  │
              └──────────┬──────────┘
                         │
                  Output (1,)  — logit

Each residual block: x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
                                                              ↑
                                               shortcut (identity or 1×1 conv)
"""

import torch
import torch.nn as nn


# ── Building blocks ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Basic residual block (used in ResNet-18 / ResNet-34).

    Consists of two 3×3 convolutions with a shortcut connection.
    When stride > 1 or channels change, a 1×1 conv is used in the shortcut.
    """

    expansion = 1  # channel multiplier at the end of the block

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Main path: two 3×3 convolutions
        self.main = nn.Sequential(
            # conv1: 3×3, stride=stride, no bias (BN handles bias)
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            # bn1
            nn.BatchNorm2d(out_channels),
            # relu
            nn.ReLU(inplace=True),
            # conv2: 3×3, stride=1
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            # bn2
            nn.BatchNorm2d(out_channels),
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                # 1×1 conv to match dimensions
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                # bn
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = out + self.shortcut(x)
        return self.relu(out)


# ── Main model ─────────────────────────────────────────────────────────────────

class ResNet(nn.Module):
    """
    ResNet-18-style network for binary classification.

    Args:
        in_channels  : number of input channels (3 for RGB)
        base_filters : filters in layer 1 (doubles each layer)
        dropout_rate : dropout before the final linear layer
    """

    def __init__(
        self,
        in_channels:  int   = 3,
        base_filters: int   = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.in_channels = base_filters  # tracks current channel count for _make_layer

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual layers
        self.layer1 = self._make_layer(base_filters,       n_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_filters * 2,   n_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters * 4,   n_blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters * 8,   n_blocks=2, stride=2)

        # Classification head
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(base_filters * 8 * ResidualBlock.expansion, 1)

        self._init_weights()

    def _make_layer(self, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        """Stack n_blocks ResidualBlocks; first block handles stride & channel change."""
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * ResidualBlock.expansion
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, 1)
        """
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = ResNet(in_channels=3, base_filters=64)
    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print("Output shape:", output.shape)  # expected: (2, 1)
