"""
models/inception.py
──────────────────────────────────────────────────────────────────────────────
Inception-style network for binary image classification (Pneumonia vs Normal).

Inspired by: "Going Deeper with Convolutions"
             Szegedy et al., 2014  —  https://arxiv.org/abs/1409.4842

Architecture overview
─────────────────────
                    Input (3, H, W)
                         │
              ┌──────────▼──────────┐
              │      Stem           │  two conv layers to reduce spatial size
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Inception Block ×4 │  each block has 4 parallel branches:
              │                     │    1×1 conv
              │                     │    1×1 → 3×3 conv
              │                     │    1×1 → 5×5 conv
              │                     │    MaxPool → 1×1 conv
              │                     │  → concatenate along channel dim
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Global Avg Pool    │
              │  + Classifier head  │
              └──────────┬──────────┘
                         │
                  Output (1,)  — logit

Key idea: instead of choosing one kernel size, compute multiple in parallel
          and let the network learn which scales matter.
"""

import torch
import torch.nn as nn


# ── Building blocks ────────────────────────────────────────────────────────────

def conv_bn_relu(in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0) -> nn.Sequential:
    """Convenience: Conv2d → BatchNorm2d → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class InceptionBlock(nn.Module):
    """
    One Inception module with 4 parallel branches.

    Branch layout:
        branch1: 1×1 conv
        branch2: 1×1 conv  →  3×3 conv
        branch3: 1×1 conv  →  5×5 conv
        branch4: 3×3 MaxPool  →  1×1 conv

    Output channels = b1 + b2_out + b3_out + b4_out

    Args:
        in_channels : input channels
        b1          : output channels of branch 1 (1×1 only)
        b2_reduce   : bottleneck channels before 3×3 in branch 2
        b2_out      : output channels of the 3×3 in branch 2
        b3_reduce   : bottleneck channels before 5×5 in branch 3
        b3_out      : output channels of the 5×5 in branch 3
        b4_out      : output channels of the 1×1 after MaxPool in branch 4
    """

    def __init__(
        self,
        in_channels: int,
        b1:        int,
        b2_reduce: int, b2_out: int,
        b3_reduce: int, b3_out: int,
        b4_out:    int,
    ):
        super().__init__()

        # Branch 1: 1×1
        self.branch1 = conv_bn_relu(in_channels, b1, kernel=1)

        # Branch 2: 1×1 → 3×3
        self.branch2 = nn.Sequential(
            # bottleneck 1×1
            nn.Conv2d(in_channels, b2_reduce, 1),
            nn.ReLU(inplace=True),
            # 3×3 conv (padding=1 to preserve spatial size)
            nn.Conv2d(b2_reduce, b2_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1×1 → 5×5
        self.branch3 = nn.Sequential(
            # bottleneck 1×1
            nn.Conv2d(in_channels, b3_reduce, 1),
            nn.ReLU(inplace=True),
            # 5×5 conv (padding=2 to preserve spatial size)
            nn.Conv2d(b3_reduce, b3_out, 5, padding=2),
            nn.ReLU(inplace=True),
        )

        # Branch 4: MaxPool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 1×1 conv
            nn.Conv2d(in_channels, b4_out, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# ── Main model ─────────────────────────────────────────────────────────────────

class Inception(nn.Module):
    """
    Inception network for binary classification.

    Args:
        in_channels  : number of input channels (3 for RGB)
        dropout_rate : dropout probability before the final linear layer
    """

    def __init__(self, in_channels: int = 3, dropout_rate: float = 0.5):
        super().__init__()

        # Stem: two conv layers to bring 224×224 down to ~28×28
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, 32, kernel=3, stride=2, padding=1),  # 112×112
            conv_bn_relu(32,          64, kernel=3, stride=1, padding=1),  # 112×112
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),              #  56×56
        )

        # Inception blocks
        # InceptionBlock(in_ch, b1, b2_reduce, b2_out, b3_reduce, b3_out, b4_out)
        self.block1 = InceptionBlock(64,  32,  16,  32,  8,  16,  16)   # out: 96
        self.pool1  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 28×28

        self.block2 = InceptionBlock(96,  64,  32,  64, 16,  32,  32)   # out: 192
        self.pool2  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 14×14

        self.block3 = InceptionBlock(192, 96,  48,  96, 24,  48,  48)   # out: 288

        self.block4 = InceptionBlock(288, 96,  48,  96, 24,  48,  48)   # out: 288

        # Classification head
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(288, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, 1)
        """
        x = self.stem(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = Inception(in_channels=3)
    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print("Output shape:", output.shape)  # expected: (2, 1)
