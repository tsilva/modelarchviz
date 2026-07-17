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
        # @arch invertedresidual.hidden_channels-in_channels-expand_ratio:start
        hidden_channels = in_channels * expand_ratio
        # @arch invertedresidual.hidden_channels-in_channels-expand_ratio:end
        # @arch invertedresidual.self-use_residual-stride-n-and-in_channels-out_channels:start
        self.use_residual = stride == 1 and in_channels == out_channels
        # @arch invertedresidual.self-use_residual-stride-n-and-in_channels-out_channels:end

        # Register expansion, depthwise filtering, and linear bottleneck projection.
        # @arch invertedresidual.layers:start
        layers = []
        # @arch invertedresidual.layers:end
        # @arch invertedresidual.if-expand_ratio-n:start
        if expand_ratio != 1:
        # @arch invertedresidual.if-expand_ratio-n:end
            # @arch invertedresidual.layers-extend:start
            layers.extend(
            # @arch invertedresidual.layers-extend:end
                # @arch invertedresidual.code.6:start
                [
                # @arch invertedresidual.code.6:end
                    # @arch invertedresidual.nn-convnd-in_channels-hidden_channels-kernel_size-n-bias-false:start
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                    # @arch invertedresidual.nn-convnd-in_channels-hidden_channels-kernel_size-n-bias-false:end
                    # @arch invertedresidual.nn-batchnormnd-hidden_channels:start
                    nn.BatchNorm2d(hidden_channels),
                    # @arch invertedresidual.nn-batchnormnd-hidden_channels:end
                    # @arch invertedresidual.nn-relun-inplace-true:start
                    nn.ReLU6(inplace=True),
                    # @arch invertedresidual.nn-relun-inplace-true:end
                # @arch invertedresidual.code.7:start
                ]
                # @arch invertedresidual.code.7:end
            # @arch invertedresidual.code.8:start
            )
            # @arch invertedresidual.code.8:end
        # @arch invertedresidual.layers-extend.2:start
        layers.extend(
        # @arch invertedresidual.layers-extend.2:end
            # @arch invertedresidual.code.9:start
            [
            # @arch invertedresidual.code.9:end
                # @arch invertedresidual.nn-convnd:start
                nn.Conv2d(
                # @arch invertedresidual.nn-convnd:end
                    # @arch invertedresidual.hidden_channels:start
                    hidden_channels,
                    # @arch invertedresidual.hidden_channels:end
                    # @arch invertedresidual.hidden_channels.2:start
                    hidden_channels,
                    # @arch invertedresidual.hidden_channels.2:end
                    # @arch invertedresidual.kernel_size-n:start
                    kernel_size=3,
                    # @arch invertedresidual.kernel_size-n:end
                    # @arch invertedresidual.stride-stride:start
                    stride=stride,
                    # @arch invertedresidual.stride-stride:end
                    # @arch invertedresidual.padding-n:start
                    padding=1,
                    # @arch invertedresidual.padding-n:end
                    # @arch invertedresidual.groups-hidden_channels:start
                    groups=hidden_channels,
                    # @arch invertedresidual.groups-hidden_channels:end
                    # @arch invertedresidual.bias-false:start
                    bias=False,
                    # @arch invertedresidual.bias-false:end
                # @arch invertedresidual.code.10:start
                ),
                # @arch invertedresidual.code.10:end
                # @arch invertedresidual.nn-batchnormnd-hidden_channels.2:start
                nn.BatchNorm2d(hidden_channels),
                # @arch invertedresidual.nn-batchnormnd-hidden_channels.2:end
                # @arch invertedresidual.nn-relun-inplace-true.2:start
                nn.ReLU6(inplace=True),
                # @arch invertedresidual.nn-relun-inplace-true.2:end
                # @arch invertedresidual.nn-convnd-hidden_channels-out_channels-kernel_size-n-bias-false:start
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                # @arch invertedresidual.nn-convnd-hidden_channels-out_channels-kernel_size-n-bias-false:end
                # @arch invertedresidual.nn-batchnormnd-out_channels:start
                nn.BatchNorm2d(out_channels),
                # @arch invertedresidual.nn-batchnormnd-out_channels:end
            # @arch invertedresidual.code.11:start
            ]
            # @arch invertedresidual.code.11:end
        # @arch invertedresidual.code.12:start
        )
        # @arch invertedresidual.code.12:end
        # @arch invertedresidual.self-block-nn-sequential-layers:start
        self.block = nn.Sequential(*layers)
        # @arch invertedresidual.self-block-nn-sequential-layers:end

    def forward(self, x):
        # Run the expanded depthwise transformation and linear projection.
        # @arch invertedresidual.forward.out-self-block-x:start
        out = self.block(x)  # (batch, in_channels, height, width) -> (batch, out_channels, out_h, out_w)
        # @arch invertedresidual.forward.out-self-block-x:end

        # Add the shortcut only when input and output bottlenecks have matching shape.
        # @arch invertedresidual.forward.if-self-use_residual:start
        if self.use_residual:
        # @arch invertedresidual.forward.if-self-use_residual:end
            # @arch invertedresidual.forward.out-out-x:start
            out = out + x  # (batch, out_channels, height, width)
            # @arch invertedresidual.forward.out-out-x:end
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
        # @arch mobilenetvn.dropout-n:start
        dropout=0.2  # Dropout probability before classification.
        # @arch mobilenetvn.dropout-n:end
    ):
        super().__init__()

        # MobileNetV2 stage plan: expansion, output channels, repeats, first stride.
        settings = [
            # @arch mobilenetvn.n-n-n-n:start
            (1, 16, 1, 1),
            # @arch mobilenetvn.n-n-n-n:end
            # @arch mobilenetvn.n-n-n-n.2:start
            (6, 24, 2, 2),
            # @arch mobilenetvn.n-n-n-n.2:end
            # @arch mobilenetvn.n-n-n-n.3:start
            (6, 32, 3, 2),
            # @arch mobilenetvn.n-n-n-n.3:end
            # @arch mobilenetvn.n-n-n-n.4:start
            (6, 64, 4, 2),
            # @arch mobilenetvn.n-n-n-n.4:end
            # @arch mobilenetvn.n-n-n-n.5:start
            (6, 96, 3, 1),
            # @arch mobilenetvn.n-n-n-n.5:end
            # @arch mobilenetvn.n-n-n-n.6:start
            (6, 160, 3, 2),
            # @arch mobilenetvn.n-n-n-n.6:end
            # @arch mobilenetvn.n-n-n-n.7:start
            (6, 320, 1, 1),
            # @arch mobilenetvn.n-n-n-n.7:end
        ]

        # Register the initial stride-2 convolutional stem.
        # @arch mobilenetvn.self-stem-nn-sequential:start
        self.stem = nn.Sequential(
        # @arch mobilenetvn.self-stem-nn-sequential:end
            # @arch mobilenetvn.nn-convnd-n-n-kernel_size-n-stride-n-padding-n-bias-false:start
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            # @arch mobilenetvn.nn-convnd-n-n-kernel_size-n-stride-n-padding-n-bias-false:end
            # @arch mobilenetvn.nn-batchnormnd-n:start
            nn.BatchNorm2d(32),
            # @arch mobilenetvn.nn-batchnormnd-n:end
            # @arch mobilenetvn.nn-relun-inplace-true:start
            nn.ReLU6(inplace=True),
            # @arch mobilenetvn.nn-relun-inplace-true:end
        # @arch mobilenetvn.code.7:start
        )
        # @arch mobilenetvn.code.7:end

        # Stack inverted residual stages with linear bottleneck outputs.
        blocks = []
        in_channels = 32
        # @arch mobilenetvn.for-expand_ratio-out_channels-repeats-stride-in-settings:start
        for expand_ratio, out_channels, repeats, stride in settings:
        # @arch mobilenetvn.for-expand_ratio-out_channels-repeats-stride-in-settings:end
            # @arch mobilenetvn.for-index-in-range-repeats:start
            for index in range(repeats):
            # @arch mobilenetvn.for-index-in-range-repeats:end
                # @arch mobilenetvn.block_stride-stride-if-index-n-else-n:start
                block_stride = stride if index == 0 else 1
                # @arch mobilenetvn.block_stride-stride-if-index-n-else-n:end
                # @arch mobilenetvn.block-invertedresidual:start
                block = InvertedResidual(
                # @arch mobilenetvn.block-invertedresidual:end
                    # @arch mobilenetvn.in_channels:start
                    in_channels,
                    # @arch mobilenetvn.in_channels:end
                    # @arch mobilenetvn.out_channels:start
                    out_channels,
                    # @arch mobilenetvn.out_channels:end
                    # @arch mobilenetvn.block_stride:start
                    block_stride,
                    # @arch mobilenetvn.block_stride:end
                    # @arch mobilenetvn.expand_ratio:start
                    expand_ratio,
                    # @arch mobilenetvn.expand_ratio:end
                # @arch mobilenetvn.code.10:start
                )
                # @arch mobilenetvn.code.10:end
                # @arch mobilenetvn.blocks-append-block:start
                blocks.append(block)
                # @arch mobilenetvn.blocks-append-block:end
                # @arch mobilenetvn.in_channels-out_channels:start
                in_channels = out_channels
                # @arch mobilenetvn.in_channels-out_channels:end
        self.blocks = nn.Sequential(*blocks)

        # Register the final expansion and classifier head.
        # @arch mobilenetvn.self-head-nn-sequential:start
        self.head = nn.Sequential(
        # @arch mobilenetvn.self-head-nn-sequential:end
            # @arch mobilenetvn.nn-convnd-in_channels-n-kernel_size-n-bias-false:start
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            # @arch mobilenetvn.nn-convnd-in_channels-n-kernel_size-n-bias-false:end
            # @arch mobilenetvn.nn-batchnormnd-n.2:start
            nn.BatchNorm2d(1280),
            # @arch mobilenetvn.nn-batchnormnd-n.2:end
            # @arch mobilenetvn.nn-relun-inplace-true.2:start
            nn.ReLU6(inplace=True),
            # @arch mobilenetvn.nn-relun-inplace-true.2:end
        # @arch mobilenetvn.code.13:start
        )
        # @arch mobilenetvn.code.13:end
        # @arch mobilenetvn.self-dropout-nn-dropout-p-dropout:start
        self.dropout = nn.Dropout(p=dropout)
        # @arch mobilenetvn.self-dropout-nn-dropout-p-dropout:end
        # @arch mobilenetvn.self-classifier-nn-linear-n-num_classes:start
        self.classifier = nn.Linear(1280, num_classes)
        # @arch mobilenetvn.self-classifier-nn-linear-n-num_classes:end

    def forward(self, x):
        # Convert image input into stem features: (batch, 3, 224, 224) -> (batch, 32, 112, 112).
        # @arch mobilenetvn.forward.x-self-stem-x:start
        x = self.stem(x)  # (batch, 3, 224, 224) -> (batch, 32, 112, 112)
        # @arch mobilenetvn.forward.x-self-stem-x:end

        # Run inverted residual stages with depthwise filters and linear bottlenecks.
        # @arch mobilenetvn.forward.x-self-blocks-x:start
        x = self.blocks(x)  # (batch, 32, 112, 112) -> (batch, 320, 7, 7)
        # @arch mobilenetvn.forward.x-self-blocks-x:end

        # Expand channels, globally pool, regularize, and classify.
        # @arch mobilenetvn.forward.x-self-head-x:start
        x = self.head(x)  # (batch, 320, 7, 7) -> (batch, 1280, 7, 7)
        # @arch mobilenetvn.forward.x-self-head-x:end
        # @arch mobilenetvn.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n:start
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # (batch, 1280, 7, 7) -> (batch, 1280, 1, 1)
        # @arch mobilenetvn.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n:end
        # @arch mobilenetvn.forward.x-torch-flatten-x-start_dim-n:start
        x = torch.flatten(x, start_dim=1)  # (batch, 1280, 1, 1) -> (batch, 1280)
        # @arch mobilenetvn.forward.x-torch-flatten-x-start_dim-n:end
        # @arch mobilenetvn.forward.x-self-dropout-x:start
        x = self.dropout(x)  # (batch, 1280)
        # @arch mobilenetvn.forward.x-self-dropout-x:end
        # @arch mobilenetvn.forward.logits-self-classifier-x:start
        logits = self.classifier(x)  # (batch, 1280) -> (batch, num_classes)
        # @arch mobilenetvn.forward.logits-self-classifier-x:end
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
# @arch train_images-torch-zeros-n-n-n-n:start
train_images = torch.zeros(2, 3, 224, 224)  # -> (2, 3, 224, 224)
# @arch train_images-torch-zeros-n-n-n-n:end
train_images[0, :, 32:96, 32:96] = 1.0
train_images[1, :, 128:192, 128:192] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    # @arch logits-model-train_images:start
    logits = model(train_images)  # (2, 3, 224, 224) -> (2, 2)
    # @arch logits-model-train_images:end
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
