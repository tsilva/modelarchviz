# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class SqueezeExcite(nn.Module):
    # @arch squeezeexcite.def-__init__:start
    def __init__(
    # @arch squeezeexcite.def-__init__:end
        self,
        channels,  # Number of channels to recalibrate.
        squeeze_channels  # Bottleneck channels in the squeeze pathway.
    ):
        super().__init__()

        # Register channel attention projections.
        # @arch squeezeexcite.self-reduce-nn-convnd-channels-squeeze_channels-kernel_size-n:start
        self.reduce = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        # @arch squeezeexcite.self-reduce-nn-convnd-channels-squeeze_channels-kernel_size-n:end
        # @arch squeezeexcite.self-expand-nn-convnd-squeeze_channels-channels-kernel_size-n:start
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)
        # @arch squeezeexcite.self-expand-nn-convnd-squeeze_channels-channels-kernel_size-n:end

    def forward(self, x):
        # Squeeze spatial dimensions into one descriptor per channel.
        scale = F.adaptive_avg_pool2d(x, output_size=1)  # (batch, channels, height, width) -> (batch, channels, 1, 1)

        # Excite channels and gate the original feature map.
        # @arch squeezeexcite.forward.scale-self-reduce-scale:start
        scale = self.reduce(scale)  # (batch, channels, 1, 1) -> (batch, squeeze_channels, 1, 1)
        # @arch squeezeexcite.forward.scale-self-reduce-scale:end
        # @arch squeezeexcite.forward.scale-f-silu-scale:start
        scale = F.silu(scale)  # (batch, squeeze_channels, 1, 1)
        # @arch squeezeexcite.forward.scale-f-silu-scale:end
        # @arch squeezeexcite.forward.scale-self-expand-scale:start
        scale = self.expand(scale)  # (batch, squeeze_channels, 1, 1) -> (batch, channels, 1, 1)
        # @arch squeezeexcite.forward.scale-self-expand-scale:end
        # @arch squeezeexcite.forward.scale-torch-sigmoid-scale:start
        scale = torch.sigmoid(scale)  # (batch, channels, 1, 1)
        # @arch squeezeexcite.forward.scale-torch-sigmoid-scale:end
        # @arch squeezeexcite.forward.out-x-scale:start
        out = x * scale  # (batch, channels, height, width)
        # @arch squeezeexcite.forward.out-x-scale:end
        # @arch squeezeexcite.forward.return-out:start
        return out
        # @arch squeezeexcite.forward.return-out:end


# %% [notebook-only]
# Create and run a squeeze-excitation gate: (2, 8, 16, 16) -> (2, 8, 16, 16).
gate = SqueezeExcite(channels=8, squeeze_channels=2)
feature_map = torch.randn(2, 8, 16, 16)  # -> (2, 8, 16, 16)
gated = gate(feature_map)  # (2, 8, 16, 16) -> (2, 8, 16, 16)
print("gated shape:", gated.shape)

# %%
class MBConv(nn.Module):
    def __init__(
        # @arch mbconv.self:start
        self,
        # @arch mbconv.self:end
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
        # @arch mbconv.squeeze_channels-max-n-in_channels-n:start
        squeeze_channels = max(1, in_channels // 4)
        # @arch mbconv.squeeze_channels-max-n-in_channels-n:end
        # @arch mbconv.padding-kernel_size-n:start
        padding = kernel_size // 2
        # @arch mbconv.padding-kernel_size-n:end
        # @arch mbconv.self-use_residual-stride-n-and-in_channels-out_channels:start
        self.use_residual = stride == 1 and in_channels == out_channels
        # @arch mbconv.self-use_residual-stride-n-and-in_channels-out_channels:end

        # Register optional expansion, depthwise filtering, SE, and projection.
        self.expand = None
        # @arch mbconv.if-expand_ratio-n:start
        if expand_ratio != 1:
        # @arch mbconv.if-expand_ratio-n:end
            self.expand = nn.Sequential(
                # @arch mbconv.nn-convnd-in_channels-expanded_channels-kernel_size-n-bias-false:start
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                # @arch mbconv.nn-convnd-in_channels-expanded_channels-kernel_size-n-bias-false:end
                # @arch mbconv.nn-batchnormnd-expanded_channels:start
                nn.BatchNorm2d(expanded_channels),
                # @arch mbconv.nn-batchnormnd-expanded_channels:end
                # @arch mbconv.nn-silu-inplace-true:start
                nn.SiLU(inplace=True),
                # @arch mbconv.nn-silu-inplace-true:end
            # @arch mbconv.code.6:start
            )
            # @arch mbconv.code.6:end
        # @arch mbconv.self-depthwise-nn-sequential:start
        self.depthwise = nn.Sequential(
        # @arch mbconv.self-depthwise-nn-sequential:end
            # @arch mbconv.nn-convnd:start
            nn.Conv2d(
            # @arch mbconv.nn-convnd:end
                # @arch mbconv.expanded_channels:start
                expanded_channels,
                # @arch mbconv.expanded_channels:end
                # @arch mbconv.expanded_channels.2:start
                expanded_channels,
                # @arch mbconv.expanded_channels.2:end
                # @arch mbconv.kernel_size-kernel_size:start
                kernel_size=kernel_size,
                # @arch mbconv.kernel_size-kernel_size:end
                # @arch mbconv.stride-stride:start
                stride=stride,
                # @arch mbconv.stride-stride:end
                # @arch mbconv.padding-padding:start
                padding=padding,
                # @arch mbconv.padding-padding:end
                # @arch mbconv.groups-expanded_channels:start
                groups=expanded_channels,
                # @arch mbconv.groups-expanded_channels:end
                # @arch mbconv.bias-false:start
                bias=False,
                # @arch mbconv.bias-false:end
            # @arch mbconv.code.7:start
            ),
            # @arch mbconv.code.7:end
            # @arch mbconv.nn-batchnormnd-expanded_channels.2:start
            nn.BatchNorm2d(expanded_channels),
            # @arch mbconv.nn-batchnormnd-expanded_channels.2:end
            # @arch mbconv.nn-silu-inplace-true.2:start
            nn.SiLU(inplace=True),
            # @arch mbconv.nn-silu-inplace-true.2:end
        # @arch mbconv.code.8:start
        )
        # @arch mbconv.code.8:end
        # @arch mbconv.se_channels-int-expanded_channels-se_ratio-or-squeeze_channels:start
        se_channels = int(expanded_channels * se_ratio) or squeeze_channels
        # @arch mbconv.se_channels-int-expanded_channels-se_ratio-or-squeeze_channels:end
        self.se = SqueezeExcite(expanded_channels, se_channels)
        # @arch mbconv.self-project-nn-sequential:start
        self.project = nn.Sequential(
        # @arch mbconv.self-project-nn-sequential:end
            # @arch mbconv.nn-convnd-expanded_channels-out_channels-kernel_size-n-bias-false:start
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            # @arch mbconv.nn-convnd-expanded_channels-out_channels-kernel_size-n-bias-false:end
            # @arch mbconv.nn-batchnormnd-out_channels:start
            nn.BatchNorm2d(out_channels),
            # @arch mbconv.nn-batchnormnd-out_channels:end
        # @arch mbconv.code.9:start
        )
        # @arch mbconv.code.9:end

    def forward(self, x):
        # Preserve the identity path when the block keeps shape unchanged.
        # @arch mbconv.forward.identity-x:start
        identity = x  # (batch, in_channels, height, width)
        # @arch mbconv.forward.identity-x:end

        # Expand channels before depthwise spatial filtering.
        out = x  # (batch, in_channels, height, width)
        # @arch mbconv.forward.if-self-expand-is-not-none:start
        if self.expand is not None:
        # @arch mbconv.forward.if-self-expand-is-not-none:end
            # @arch mbconv.forward.out-self-expand-out:start
            out = self.expand(out)  # (batch, in_channels, height, width) -> (batch, expanded_channels, height, width)
            # @arch mbconv.forward.out-self-expand-out:end
        # @arch mbconv.forward.out-self-depthwise-out:start
        out = self.depthwise(out)  # (batch, expanded_channels, height, width) -> (batch, expanded_channels, out_h, out_w)
        # @arch mbconv.forward.out-self-depthwise-out:end

        # Reweight channels with squeeze-and-excitation, then project back.
        out = self.se(out)  # (batch, expanded_channels, out_h, out_w)
        # @arch mbconv.forward.out-self-project-out:start
        out = self.project(out)  # (batch, expanded_channels, out_h, out_w) -> (batch, out_channels, out_h, out_w)
        # @arch mbconv.forward.out-self-project-out:end

        # Add the residual shortcut only for same-shape blocks.
        # @arch mbconv.forward.if-self-use_residual:start
        if self.use_residual:
        # @arch mbconv.forward.if-self-use_residual:end
            # @arch mbconv.forward.out-out-identity:start
            out = out + identity  # (batch, out_channels, height, width)
            # @arch mbconv.forward.out-out-identity:end
        # @arch mbconv.forward.return-out:start
        return out
        # @arch mbconv.forward.return-out:end


# %% [notebook-only]
# Create and run one mobile inverted bottleneck: (2, 8, 16, 16) -> (2, 8, 16, 16).
example_block = MBConv(in_channels=8, out_channels=8, expand_ratio=1, stride=1, kernel_size=3)
block_input = torch.randn(2, 8, 16, 16)  # -> (2, 8, 16, 16)
example_block_output = example_block(block_input)  # (2, 8, 16, 16) -> (2, 8, 16, 16)
print("block_output shape:", example_block_output.shape)

# %%
class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # EfficientNet-B0 stage plan: expand, channels, repeats, stride, kernel.
        settings = [
            (1, 16, 1, 1, 3),
            # @arch efficientnet.n-n-n-n-n.2:start
            (6, 24, 2, 2, 3),
            # @arch efficientnet.n-n-n-n-n.2:end
            # @arch efficientnet.n-n-n-n-n.3:start
            (6, 40, 2, 2, 5),
            # @arch efficientnet.n-n-n-n-n.3:end
            # @arch efficientnet.n-n-n-n-n.4:start
            (6, 80, 3, 2, 3),
            # @arch efficientnet.n-n-n-n-n.4:end
            # @arch efficientnet.n-n-n-n-n.5:start
            (6, 112, 3, 1, 5),
            # @arch efficientnet.n-n-n-n-n.5:end
            # @arch efficientnet.n-n-n-n-n.6:start
            (6, 192, 4, 2, 5),
            # @arch efficientnet.n-n-n-n-n.6:end
            # @arch efficientnet.n-n-n-n-n.7:start
            (6, 320, 1, 1, 3),
            # @arch efficientnet.n-n-n-n-n.7:end
        # @arch efficientnet.code.4:start
        ]
        # @arch efficientnet.code.4:end

        # Register the convolutional stem.
        self.stem = nn.Sequential(
            nn.Conv2d(
                # @arch efficientnet.n:start
                3,
                # @arch efficientnet.n:end
                # @arch efficientnet.n.2:start
                32,
                # @arch efficientnet.n.2:end
                # @arch efficientnet.kernel_size-n:start
                kernel_size=3,
                # @arch efficientnet.kernel_size-n:end
                # @arch efficientnet.stride-n:start
                stride=2,
                # @arch efficientnet.stride-n:end
                # @arch efficientnet.padding-n:start
                padding=1,
                # @arch efficientnet.padding-n:end
                # @arch efficientnet.bias-false:start
                bias=False,
                # @arch efficientnet.bias-false:end
            # @arch efficientnet.code.7:start
            ),
            # @arch efficientnet.code.7:end
            # @arch efficientnet.nn-batchnormnd-n:start
            nn.BatchNorm2d(32),
            # @arch efficientnet.nn-batchnormnd-n:end
            # @arch efficientnet.nn-silu-inplace-true:start
            nn.SiLU(inplace=True),
            # @arch efficientnet.nn-silu-inplace-true:end
        # @arch efficientnet.code.8:start
        )
        # @arch efficientnet.code.8:end

        # Register mobile inverted bottleneck stages.
        # @arch efficientnet.blocks:start
        blocks = []
        # @arch efficientnet.blocks:end
        in_channels = 32
        # @arch efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings:start
        for expand_ratio, out_channels, repeats, stride, kernel_size in settings:
        # @arch efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings:end
            # @arch efficientnet.for-index-in-range-repeats:start
            for index in range(repeats):
            # @arch efficientnet.for-index-in-range-repeats:end
                # @arch efficientnet.block_stride-stride-if-index-n-else-n:start
                block_stride = stride if index == 0 else 1
                # @arch efficientnet.block_stride-stride-if-index-n-else-n:end
                # @arch efficientnet.block-mbconv:start
                block = MBConv(
                # @arch efficientnet.block-mbconv:end
                    # @arch efficientnet.in_channels:start
                    in_channels,
                    # @arch efficientnet.in_channels:end
                    # @arch efficientnet.out_channels:start
                    out_channels,
                    # @arch efficientnet.out_channels:end
                    # @arch efficientnet.expand_ratio:start
                    expand_ratio,
                    # @arch efficientnet.expand_ratio:end
                    # @arch efficientnet.block_stride:start
                    block_stride,
                    # @arch efficientnet.block_stride:end
                    # @arch efficientnet.kernel_size:start
                    kernel_size,
                    # @arch efficientnet.kernel_size:end
                # @arch efficientnet.code.11:start
                )
                # @arch efficientnet.code.11:end
                # @arch efficientnet.blocks-append-block:start
                blocks.append(block)
                # @arch efficientnet.blocks-append-block:end
                # @arch efficientnet.in_channels-out_channels:start
                in_channels = out_channels
                # @arch efficientnet.in_channels-out_channels:end
        # @arch efficientnet.self-blocks-nn-sequential-blocks:start
        self.blocks = nn.Sequential(*blocks)
        # @arch efficientnet.self-blocks-nn-sequential-blocks:end

        # Register the final expansion head and classifier.
        self.head = nn.Sequential(
            nn.Conv2d(
                # @arch efficientnet.in_channels.2:start
                in_channels,
                # @arch efficientnet.in_channels.2:end
                # @arch efficientnet.n.3:start
                1280,
                # @arch efficientnet.n.3:end
                # @arch efficientnet.kernel_size-n.2:start
                kernel_size=1,
                # @arch efficientnet.kernel_size-n.2:end
                # @arch efficientnet.bias-false.2:start
                bias=False,
                # @arch efficientnet.bias-false.2:end
            # @arch efficientnet.code.14:start
            ),
            # @arch efficientnet.code.14:end
            # @arch efficientnet.nn-batchnormnd-n.2:start
            nn.BatchNorm2d(1280),
            # @arch efficientnet.nn-batchnormnd-n.2:end
            # @arch efficientnet.nn-silu-inplace-true.2:start
            nn.SiLU(inplace=True),
            # @arch efficientnet.nn-silu-inplace-true.2:end
        # @arch efficientnet.code.15:start
        )
        # @arch efficientnet.code.15:end
        # @arch efficientnet.self-classifier-nn-linear-n-num_classes:start
        self.classifier = nn.Linear(1280, num_classes)
        # @arch efficientnet.self-classifier-nn-linear-n-num_classes:end

    # @arch efficientnet.def-forward-self-x:start
    def forward(self, x):
    # @arch efficientnet.def-forward-self-x:end
        # Convert image input into stem features: (batch, 3, 224, 224) -> (batch, 32, 112, 112).
        x = self.stem(x)  # (batch, 3, 224, 224) -> (batch, 32, 112, 112)

        # Run compound-scaled MBConv stages with depthwise filters and SE gates.
        # @arch efficientnet.forward.x-self-blocks-x:start
        x = self.blocks(x)  # (batch, 32, 112, 112) -> (batch, 320, 7, 7)
        # @arch efficientnet.forward.x-self-blocks-x:end

        # Expand final channels, pool, and classify.
        # @arch efficientnet.forward.x-self-head-x:start
        x = self.head(x)  # (batch, 320, 7, 7) -> (batch, 1280, 7, 7)
        # @arch efficientnet.forward.x-self-head-x:end
        # @arch efficientnet.head.pool:start
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # (batch, 1280, 7, 7) -> (batch, 1280, 1, 1)
        # @arch efficientnet.head.pool:end
        # @arch efficientnet.forward.x-torch-flatten-x-n:start
        x = torch.flatten(x, 1)  # (batch, 1280, 1, 1) -> (batch, 1280)
        # @arch efficientnet.forward.x-torch-flatten-x-n:end
        # @arch efficientnet.forward.logits-self-classifier-x:start
        logits = self.classifier(x)  # (batch, 1280) -> (batch, num_classes)
        # @arch efficientnet.forward.logits-self-classifier-x:end
        return logits


# %% [notebook-only]
# Create and run a sample ImageNet-size batch: (2, 3, 224, 224) -> (2, 1000).
example_model = EfficientNet(num_classes=1000)
example_test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
example_logits = example_model(example_test_input)  # (2, 3, 224, 224) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = EfficientNet(num_classes=2)
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
final_loss = loss.item()  # scalar
