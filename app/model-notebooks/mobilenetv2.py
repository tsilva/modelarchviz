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


# %%
class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of channels entering the block.
        out_channels,  # Number of channels after linear projection.
        stride,  # Spatial stride for the depthwise convolution.
        expand_ratio  # Channel expansion multiplier before depthwise filtering.
    ):
        super().__init__()

        # Configure the expanded hidden width and residual eligibility.
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        # Register expansion, depthwise filtering, and linear bottleneck projection.
        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # Run the expanded depthwise transformation and linear projection.
        out = self.block(x)  # (batch, in_channels, height, width) -> (batch, out_channels, out_h, out_w)

        # Add the shortcut only when input and output bottlenecks have matching shape.
        if self.use_residual:
            out = out + x  # (batch, out_channels, height, width)
        return out


# %% [notebook-only]
# Create and run one inverted residual block: (2, 16, 32, 32) -> (2, 16, 32, 32).
example_block = InvertedResidual(in_channels=16, out_channels=16, stride=1, expand_ratio=6)
example_block_input = torch.randn(2, 16, 32, 32)  # -> (2, 16, 32, 32)
example_block_output = example_block(example_block_input)  # (2, 16, 32, 32) -> (2, 16, 32, 32)
print("block_output shape:", example_block_output.shape)

# %%
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,  # Number of output classes.
        dropout=0.2  # Dropout probability before classification.
    ):
        super().__init__()

        # MobileNetV2 stage plan: expansion, output channels, repeats, first stride.
        settings = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # Register the initial stride-2 convolutional stem.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Stack inverted residual stages with linear bottleneck outputs.
        blocks = []
        in_channels = 32
        for expand_ratio, out_channels, repeats, stride in settings:
            for index in range(repeats):
                block_stride = stride if index == 0 else 1
                block = InvertedResidual(
                    in_channels,
                    out_channels,
                    block_stride,
                    expand_ratio,
                )
                blocks.append(block)
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        # Register the final expansion and classifier head.
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Convert image input into stem features: (batch, 3, 224, 224) -> (batch, 32, 112, 112).
        x = self.stem(x)  # (batch, 3, 224, 224) -> (batch, 32, 112, 112)

        # Run inverted residual stages with depthwise filters and linear bottlenecks.
        x = self.blocks(x)  # (batch, 32, 112, 112) -> (batch, 320, 7, 7)

        # Expand channels, globally pool, regularize, and classify.
        x = self.head(x)  # (batch, 320, 7, 7) -> (batch, 1280, 7, 7)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # (batch, 1280, 7, 7) -> (batch, 1280, 1, 1)
        x = torch.flatten(x, start_dim=1)  # (batch, 1280, 1, 1) -> (batch, 1280)
        x = self.dropout(x)  # (batch, 1280)
        logits = self.classifier(x)  # (batch, 1280) -> (batch, num_classes)
        return logits


# %% [notebook-only]
# Create and run a sample ImageNet-size batch: (2, 3, 224, 224) -> (2, 1000).
example_model = MobileNetV2(num_classes=1000)
example_test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
example_logits = example_model(example_test_input)  # (2, 3, 224, 224) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = MobileNetV2(num_classes=2)
train_images = torch.zeros(2, 3, 224, 224)  # -> (2, 3, 224, 224)
train_images[0, :, 32:96, 32:96] = 1.0
train_images[1, :, 128:192, 128:192] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 224, 224) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
