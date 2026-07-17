# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
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
        # @arch widebasicblock.self-bnn-nn-batchnormnd-in_channels:start
        self.bn1 = nn.BatchNorm2d(in_channels)
        # @arch widebasicblock.self-bnn-nn-batchnormnd-in_channels:end
        # @arch widebasicblock.self-relu-nn-relu-inplace-true:start
        self.relu = nn.ReLU(inplace=True)
        # @arch widebasicblock.self-relu-nn-relu-inplace-true:end
        # @arch widebasicblock.self-convn-nn-convnd:start
        self.conv1 = nn.Conv2d(
        # @arch widebasicblock.self-convn-nn-convnd:end
            # @arch widebasicblock.in_channels.2:start
            in_channels,
            # @arch widebasicblock.in_channels.2:end
            # @arch widebasicblock.out_channels.2:start
            out_channels,
            # @arch widebasicblock.out_channels.2:end
            # @arch widebasicblock.kernel_size-n:start
            kernel_size=3,
            # @arch widebasicblock.kernel_size-n:end
            # @arch widebasicblock.stride-stride:start
            stride=stride,
            # @arch widebasicblock.stride-stride:end
            # @arch widebasicblock.padding-n:start
            padding=1,
            # @arch widebasicblock.padding-n:end
            # @arch widebasicblock.bias-false:start
            bias=False,
            # @arch widebasicblock.bias-false:end
        # @arch widebasicblock.code.4:start
        )
        # @arch widebasicblock.code.4:end
        # @arch widebasicblock.self-bnn-nn-batchnormnd-out_channels:start
        self.bn2 = nn.BatchNorm2d(out_channels)
        # @arch widebasicblock.self-bnn-nn-batchnormnd-out_channels:end
        # @arch widebasicblock.self-dropout_rate-dropout_rate:start
        self.dropout_rate = dropout_rate
        # @arch widebasicblock.self-dropout_rate-dropout_rate:end
        # @arch widebasicblock.self-conv2:start
        self.conv2 = nn.Conv2d(
        # @arch widebasicblock.self-conv2:end
            # @arch widebasicblock.out_channels.3:start
            out_channels,
            # @arch widebasicblock.out_channels.3:end
            # @arch widebasicblock.out_channels.4:start
            out_channels,
            # @arch widebasicblock.out_channels.4:end
            # @arch widebasicblock.kernel_size-n.2:start
            kernel_size=3,
            # @arch widebasicblock.kernel_size-n.2:end
            # @arch widebasicblock.stride-n.2:start
            stride=1,
            # @arch widebasicblock.stride-n.2:end
            # @arch widebasicblock.padding-n.2:start
            padding=1,
            # @arch widebasicblock.padding-n.2:end
            # @arch widebasicblock.bias-false.2:start
            bias=False,
            # @arch widebasicblock.bias-false.2:end
        # @arch widebasicblock.code.5:start
        )
        # @arch widebasicblock.code.5:end
        # @arch widebasicblock.self-shortcut-none:start
        self.shortcut = None
        # @arch widebasicblock.self-shortcut-none:end
        # @arch widebasicblock.if-stride-n-or-in_channels-out_channels:start
        if stride != 1 or in_channels != out_channels:
        # @arch widebasicblock.if-stride-n-or-in_channels-out_channels:end
            # @arch widebasicblock.self-shortcut-nn-convnd:start
            self.shortcut = nn.Conv2d(
            # @arch widebasicblock.self-shortcut-nn-convnd:end
                # @arch widebasicblock.in_channels.3:start
                in_channels,
                # @arch widebasicblock.in_channels.3:end
                # @arch widebasicblock.out_channels.5:start
                out_channels,
                # @arch widebasicblock.out_channels.5:end
                # @arch widebasicblock.kernel_size-n.3:start
                kernel_size=1,
                # @arch widebasicblock.kernel_size-n.3:end
                # @arch widebasicblock.stride-stride.2:start
                stride=stride,
                # @arch widebasicblock.stride-stride.2:end
                # @arch widebasicblock.bias-false.3:start
                bias=False,
                # @arch widebasicblock.bias-false.3:end
            # @arch widebasicblock.code.6:start
            )
            # @arch widebasicblock.code.6:end

    def forward(self, x):
        # Preserve the shortcut path, projecting it when width or spatial size changes.
        # @arch widebasicblock.forward.shortcut-x:start
        shortcut = x  # (batch, in_channels, height, width)
        # @arch widebasicblock.forward.shortcut-x:end
        # @arch widebasicblock.forward.if-self-shortcut-is-not-none:start
        if self.shortcut is not None:
        # @arch widebasicblock.forward.if-self-shortcut-is-not-none:end
            # @arch widebasicblock.forward.shortcut-self-shortcut-x:start
            shortcut = self.shortcut(x)  # (batch, in_channels, height, width) -> (batch, out_channels, out_h, out_w)
            # @arch widebasicblock.forward.shortcut-self-shortcut-x:end

        # Run the widened pre-activation residual branch.
        # @arch widebasicblock.forward.out-self-bnn-x:start
        out = self.bn1(x)  # (batch, in_channels, height, width)
        # @arch widebasicblock.forward.out-self-bnn-x:end
        # @arch widebasicblock.forward.out-self-relu-out:start
        out = self.relu(out)  # (batch, in_channels, height, width)
        # @arch widebasicblock.forward.out-self-relu-out:end
        # @arch widebasicblock.forward.out-self-convn-out:start
        out = self.conv1(out)  # (batch, in_channels, height, width) -> (batch, out_channels, out_h, out_w)
        # @arch widebasicblock.forward.out-self-convn-out:end
        # @arch widebasicblock.forward.out-self-bnn-out:start
        out = self.bn2(out)  # (batch, out_channels, out_h, out_w)
        # @arch widebasicblock.forward.out-self-bnn-out:end
        # @arch widebasicblock.forward.out-self-relu-out.2:start
        out = self.relu(out)  # (batch, out_channels, out_h, out_w)
        # @arch widebasicblock.forward.out-self-relu-out.2:end
        # @arch widebasicblock.forward.if-self-dropout_rate-n:start
        if self.dropout_rate > 0:
        # @arch widebasicblock.forward.if-self-dropout_rate-n:end
            out = F.dropout(out, p=self.dropout_rate, training=self.training)  # (batch, out_channels, out_h, out_w)
        # @arch widebasicblock.forward.conv2:start
        out = self.conv2(out)  # (batch, out_channels, out_h, out_w)
        # @arch widebasicblock.forward.conv2:end

        # Merge shortcut and residual features.
        # @arch widebasicblock.forward.out-out-shortcut:start
        out = out + shortcut  # (batch, out_channels, out_h, out_w)
        # @arch widebasicblock.forward.out-out-shortcut:end
        # @arch widebasicblock.forward.return-out:start
        return out
        # @arch widebasicblock.forward.return-out:end


# %% [notebook-only]
# Create and run one widened residual block: (2, 16, 32, 32) -> (2, 32, 16, 16).
example_block = WideBasicBlock(in_channels=16, out_channels=32, stride=2, dropout_rate=0.0)
block_input = torch.randn(2, 16, 32, 32)  # -> (2, 16, 32, 32)
example_block_output = example_block(block_input)  # (2, 16, 32, 32) -> (2, 32, 16, 16)
print("block_output shape:", example_block_output.shape)

# %%
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
        # @arch widenet.stem-conv:start
        self.conv1 = nn.Conv2d(
        # @arch widenet.stem-conv:end
            3,
            # @arch widenet.widths-n:start
            widths[0],
            # @arch widenet.widths-n:end
            # @arch widenet.kernel_size-n:start
            kernel_size=3,
            # @arch widenet.kernel_size-n:end
            # @arch widenet.stride-n:start
            stride=1,
            # @arch widenet.stride-n:end
            # @arch widenet.padding-n:start
            padding=1,
            # @arch widenet.padding-n:end
            # @arch widenet.bias-false:start
            bias=False,
            # @arch widenet.bias-false:end
        # @arch widenet.code.7:start
        )
        # @arch widenet.code.7:end
        # @arch widenet.self-layern-self-_make_layer:start
        self.layer1 = self._make_layer(
        # @arch widenet.self-layern-self-_make_layer:end
            # @arch widenet.widths-n.2:start
            widths[0],
            # @arch widenet.widths-n.2:end
            # @arch widenet.widths-n.3:start
            widths[1],
            # @arch widenet.widths-n.3:end
            # @arch widenet.block_count:start
            block_count,
            # @arch widenet.block_count:end
            # @arch widenet.stride-n.2:start
            stride=1,
            # @arch widenet.stride-n.2:end
            # @arch widenet.dropout_rate-dropout_rate:start
            dropout_rate=dropout_rate,
            # @arch widenet.dropout_rate-dropout_rate:end
        # @arch widenet.code.8:start
        )
        # @arch widenet.code.8:end
        # @arch widenet.self-layern-self-_make_layer.2:start
        self.layer2 = self._make_layer(
        # @arch widenet.self-layern-self-_make_layer.2:end
            # @arch widenet.widths-n.4:start
            widths[1],
            # @arch widenet.widths-n.4:end
            # @arch widenet.widths-n.5:start
            widths[2],
            # @arch widenet.widths-n.5:end
            # @arch widenet.block_count.2:start
            block_count,
            # @arch widenet.block_count.2:end
            # @arch widenet.stride-n.3:start
            stride=2,
            # @arch widenet.stride-n.3:end
            # @arch widenet.dropout_rate-dropout_rate.2:start
            dropout_rate=dropout_rate,
            # @arch widenet.dropout_rate-dropout_rate.2:end
        # @arch widenet.code.9:start
        )
        # @arch widenet.code.9:end
        # @arch widenet.self-layern-self-_make_layer.3:start
        self.layer3 = self._make_layer(
        # @arch widenet.self-layern-self-_make_layer.3:end
            # @arch widenet.widths-n.6:start
            widths[2],
            # @arch widenet.widths-n.6:end
            # @arch widenet.widths-n.7:start
            widths[3],
            # @arch widenet.widths-n.7:end
            # @arch widenet.block_count.3:start
            block_count,
            # @arch widenet.block_count.3:end
            # @arch widenet.stride-n.4:start
            stride=2,
            # @arch widenet.stride-n.4:end
            # @arch widenet.dropout_rate-dropout_rate.3:start
            dropout_rate=dropout_rate,
            # @arch widenet.dropout_rate-dropout_rate.3:end
        # @arch widenet.code.10:start
        )
        # @arch widenet.code.10:end
        # @arch widenet.self-bn-nn-batchnormnd-widths-n:start
        self.bn = nn.BatchNorm2d(widths[3])
        # @arch widenet.self-bn-nn-batchnormnd-widths-n:end
        # @arch widenet.self-relu-nn-relu-inplace-true:start
        self.relu = nn.ReLU(inplace=True)
        # @arch widenet.self-relu-nn-relu-inplace-true:end
        # @arch widenet.self-fc-nn-linear-widths-n-num_classes:start
        self.fc = nn.Linear(widths[3], num_classes)
        # @arch widenet.self-fc-nn-linear-widths-n-num_classes:end

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_rate):
        # Start each stage with the only block that may widen channels or downsample.
        # @arch widenet._make_layer.layers:start
        layers = [
        # @arch widenet._make_layer.layers:end
            # @arch widenet._make_layer.widebasicblock:start
            WideBasicBlock(
            # @arch widenet._make_layer.widebasicblock:end
                # @arch widenet._make_layer.in_channels:start
                in_channels,
                # @arch widenet._make_layer.in_channels:end
                # @arch widenet._make_layer.out_channels:start
                out_channels,
                # @arch widenet._make_layer.out_channels:end
                # @arch widenet._make_layer.stride-stride:start
                stride=stride,
                # @arch widenet._make_layer.stride-stride:end
                # @arch widenet._make_layer.dropout_rate-dropout_rate:start
                dropout_rate=dropout_rate,
                # @arch widenet._make_layer.dropout_rate-dropout_rate:end
            # @arch widenet._make_layer.code.2:start
            )
            # @arch widenet._make_layer.code.2:end
        # @arch widenet._make_layer.code.3:start
        ]
        # @arch widenet._make_layer.code.3:end
        # @arch widenet._make_layer.for-_-in-range-n-blocks:start
        for _ in range(1, blocks):
        # @arch widenet._make_layer.for-_-in-range-n-blocks:end
            block = WideBasicBlock(
                # @arch widenet._make_layer.out_channels.2:start
                out_channels,
                # @arch widenet._make_layer.out_channels.2:end
                # @arch widenet._make_layer.out_channels.3:start
                out_channels,
                # @arch widenet._make_layer.out_channels.3:end
                # @arch widenet._make_layer.dropout_rate-dropout_rate.2:start
                dropout_rate=dropout_rate,
                # @arch widenet._make_layer.dropout_rate-dropout_rate.2:end
            # @arch widenet._make_layer.code.4:start
            )
            # @arch widenet._make_layer.code.4:end
            # @arch widenet._make_layer.layers-append-block:start
            layers.append(block)
            # @arch widenet._make_layer.layers-append-block:end

        # @arch widenet._make_layer.stage-nn-sequential-layers:start
        stage = nn.Sequential(*layers)
        # @arch widenet._make_layer.stage-nn-sequential-layers:end
        return stage

    def forward(self, x):
        # Convert image input into low-level features: (batch, 3, 32, 32) -> (batch, 16, 32, 32).
        # @arch widenet.input:start
        x = self.conv1(x)  # (batch, 3, 32, 32) -> (batch, 16, 32, 32)
        # @arch widenet.input:end

        # Run widened residual stages: 160, 320, then 640 channels for WRN-28-10.
        x = self.layer1(x)  # (batch, 16, 32, 32) -> (batch, 160, 32, 32)
        x = self.layer2(x)  # (batch, 160, 32, 32) -> (batch, 320, 16, 16)
        # @arch widenet.forward.x-self-layern-x.3:start
        x = self.layer3(x)  # (batch, 320, 16, 16) -> (batch, 640, 8, 8)
        # @arch widenet.forward.x-self-layern-x.3:end

        # Pool final feature maps and classify: (batch, 640, 8, 8) -> (batch, 10).
        # @arch widenet.head.bn:start
        x = self.bn(x)  # (batch, 640, 8, 8)
        # @arch widenet.head.bn:end
        # @arch widenet.forward.x-self-relu-x:start
        x = self.relu(x)  # (batch, 640, 8, 8)
        # @arch widenet.forward.x-self-relu-x:end
        # @arch widenet.forward.x-f-avg_poolnd-x-kernel_size-n:start
        x = F.avg_pool2d(x, kernel_size=8)  # (batch, 640, 8, 8) -> (batch, 640, 1, 1)
        # @arch widenet.forward.x-f-avg_poolnd-x-kernel_size-n:end
        # @arch widenet.forward.x-torch-flatten-x-n:start
        x = torch.flatten(x, 1)  # (batch, 640, 1, 1) -> (batch, 640)
        # @arch widenet.forward.x-torch-flatten-x-n:end
        # @arch widenet.forward.logits-self-fc-x:start
        logits = self.fc(x)  # (batch, 640) -> (batch, num_classes)
        # @arch widenet.forward.logits-self-fc-x:end
        return logits


# %% [notebook-only]
# Create and run a sample CIFAR-size image batch: (2, 3, 32, 32) -> (2, 10).
example_model = WideNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=10)
example_test_input = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
example_logits = example_model(example_test_input)  # (2, 3, 32, 32) -> (2, 10)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic CIFAR-size batch.
model = WideNet(depth=10, widen_factor=1, dropout_rate=0.0, num_classes=2)
train_images = torch.zeros(2, 3, 32, 32)  # -> (2, 3, 32, 32)
train_images[0, :, 4:16, 4:16] = 1.0
train_images[1, :, 16:28, 16:28] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 32, 32) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
