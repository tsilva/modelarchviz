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


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels,  # Number of channels to recalibrate.
        squeeze_channels  # Bottleneck channels in the squeeze pathway.
    ):
        super().__init__()

        # Register channel attention projections.
        self.reduce = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

    def forward(self, x):
        # Squeeze spatial dimensions into one descriptor per channel.
        scale = F.adaptive_avg_pool2d(x, output_size=1)

        # Excite channels and gate the original feature map.
        scale = self.reduce(scale)
        scale = F.silu(scale)
        scale = self.expand(scale)
        scale = torch.sigmoid(scale)
        out = x * scale
        return out


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of channels entering the block.
        out_channels,  # Number of channels after projection.
        expand_ratio,  # Expansion multiplier before depthwise convolution.
        stride,  # Spatial stride for the depthwise convolution.
        kernel_size,  # Depthwise convolution kernel size.
        se_ratio=0.25  # Squeeze-and-excitation channel ratio.
    ):
        super().__init__()

        # Configure expanded and squeezed channel widths.
        expanded_channels = in_channels * expand_ratio
        squeeze_channels = max(1, in_channels // 4)
        padding = kernel_size // 2
        self.use_residual = stride == 1 and in_channels == out_channels

        # Register optional expansion, depthwise filtering, SE, and projection.
        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True),
            )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
        )
        se_channels = int(expanded_channels * se_ratio) or squeeze_channels
        self.se = SqueezeExcite(expanded_channels, se_channels)
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Preserve the identity path when the block keeps shape unchanged.
        identity = x

        # Expand channels before depthwise spatial filtering.
        out = x
        if self.expand is not None:
            out = self.expand(out)
        out = self.depthwise(out)

        # Reweight channels with squeeze-and-excitation, then project back.
        out = self.se(out)
        out = self.project(out)

        # Add the residual shortcut only for same-shape blocks.
        if self.use_residual:
            out = out + identity
        return out


class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # EfficientNet-B0 stage plan: expand, channels, repeats, stride, kernel.
        settings = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        # Register the convolutional stem.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Register mobile inverted bottleneck stages.
        blocks = []
        in_channels = 32
        for expand_ratio, out_channels, repeats, stride, kernel_size in settings:
            for index in range(repeats):
                block_stride = stride if index == 0 else 1
                block = MBConv(
                    in_channels,
                    out_channels,
                    expand_ratio,
                    block_stride,
                    kernel_size,
                )
                blocks.append(block)
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        # Register the final expansion head and classifier.
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                1280,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Convert image input into stem features: (batch, 3, 224, 224) -> (batch, 32, 112, 112).
        x = self.stem(x)

        # Run compound-scaled MBConv stages with depthwise filters and SE gates.
        x = self.blocks(x)

        # Expand final channels, pool, and classify.
        x = self.head(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


# Create and run a sample ImageNet-size batch: (2, 3, 224, 224) -> (2, 1000).
model = EfficientNet(num_classes=1000)
test_input = torch.randn(2, 3, 224, 224)
logits = model(test_input)

# logits: (2, 1000)

# Train on a tiny synthetic image batch.
model = EfficientNet(num_classes=2)
train_images = torch.zeros(2, 3, 224, 224)
train_images[0, :, 32:96, 32:96] = 1.0
train_images[1, :, 128:192, 128:192] = 1.0
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
