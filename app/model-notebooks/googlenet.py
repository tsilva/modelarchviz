# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input feature channels.
        branch1_channels,  # Output channels for the 1x1 branch.
        branch3_reduce,  # Reduction channels before the 3x3 branch.
        branch3_channels,  # Output channels for the 3x3 branch.
        branch5_reduce,  # Reduction channels before the 5x5 branch.
        branch5_channels,  # Output channels for the 5x5 branch.
        pool_channels  # Output channels for the pooling branch.
    ):
        super().__init__()

        # Register the four parallel Inception branches.
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch1_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch3_reduce, branch3_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch5_reduce, branch5_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Evaluate parallel branches while preserving spatial size.
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = self.branch_pool(x)

        # Concatenate branch channels: list of (batch, channels, height, width) -> one feature map.
        branches = [branch1, branch3, branch5, branch_pool]
        x = torch.cat(branches, dim=1)
        return x


class GoogLeNet(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # Register stem, Inception stages, and classifier head.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Downsample the input into stem features: (batch, 3, 224, 224) -> (batch, 192, 28, 28).
        x = self.stem(x)

        # Run Inception stage 3 and downsample spatial size.
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Run Inception stage 4 and downsample spatial size.
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Run Inception stage 5 and pool to a classifier vector.
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        # Apply dropout and classify pooled features: (batch, 1024) -> (batch, num_classes).
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
model = GoogLeNet(num_classes=1000)
test_input = torch.randn(2, 3, 224, 224)
logits = model(test_input)

# logits: (2, 1000)
