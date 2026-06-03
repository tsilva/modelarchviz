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


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of widened output channels.
        stride=1,  # Spatial stride for the first convolution.
        dropout_rate=0.0  # Dropout probability inside the residual branch.
    ):
        super().__init__()

        # Register a pre-activation residual branch with an optional projection shortcut.
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout_rate = dropout_rate
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        # Preserve the shortcut path, projecting it when width or spatial size changes.
        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(x)

        # Run the widened pre-activation residual branch.
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)

        # Merge shortcut and residual features.
        out = out + shortcut
        return out


class WideNet(nn.Module):
    def __init__(
        self,
        depth=28,  # Total WRN depth; must satisfy 6n + 4.
        widen_factor=10,  # Channel multiplier for residual stages.
        dropout_rate=0.0,  # Dropout probability inside residual blocks.
        num_classes=10  # Number of output classes.
    ):
        super().__init__()

        # Configure the WRN-28-10 channel plan for CIFAR-size inputs.
        if (depth - 4) % 6 != 0:
            raise ValueError("WideNet depth must satisfy depth = 6n + 4.")
        block_count = (depth - 4) // 6
        widths = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        # Register the shallow stem, three widened residual stages, and classifier.
        self.conv1 = nn.Conv2d(
            3,
            widths[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer1 = self._make_layer(
            widths[0],
            widths[1],
            block_count,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.layer2 = self._make_layer(
            widths[1],
            widths[2],
            block_count,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.layer3 = self._make_layer(
            widths[2],
            widths[3],
            block_count,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_rate):
        # Start each stage with the only block that may widen channels or downsample.
        layers = [
            WideBasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout_rate=dropout_rate,
            )
        ]
        for _ in range(1, blocks):
            block = WideBasicBlock(
                out_channels,
                out_channels,
                dropout_rate=dropout_rate,
            )
            layers.append(block)

        stage = nn.Sequential(*layers)
        return stage

    def forward(self, x):
        # Convert image input into low-level features: (batch, 3, 32, 32) -> (batch, 16, 32, 32).
        x = self.conv1(x)

        # Run widened residual stages: 160, 320, then 640 channels for WRN-28-10.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Pool final feature maps and classify: (batch, 640, 8, 8) -> (batch, 10).
        x = self.bn(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


# Create and run a sample CIFAR-size image batch: (2, 3, 32, 32) -> (2, 10).
model = WideNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=10)
test_input = torch.randn(2, 3, 32, 32)
logits = model(test_input)

# logits: (2, 10)

# Train on a tiny synthetic CIFAR-size batch.
model = WideNet(depth=10, widen_factor=1, dropout_rate=0.0, num_classes=2)
train_images = torch.zeros(2, 3, 32, 32)
train_images[0, :, 4:16, 4:16] = 1.0
train_images[1, :, 16:28, 16:28] = 1.0
train_targets = torch.tensor([0, 1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)
    loss = criterion(logits, train_targets)
    loss.backward()
    optimizer.step()

final_loss = loss.item()
