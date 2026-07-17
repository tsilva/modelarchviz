# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of channels entering the dense layer.
        growth_rate,  # Number of new feature channels produced by this layer.
        bottleneck_width=4,  # Multiplier for the 1x1 bottleneck width.
        dropout_rate=0.0  # Dropout probability after the 3x3 convolution.
    ):
        super().__init__()

        # Register the bottleneck and growth convolutions.
        # @arch denselayer.bottleneck_channels-bottleneck_width-growth_rate:start
        bottleneck_channels = bottleneck_width * growth_rate
        # @arch denselayer.bottleneck_channels-bottleneck_width-growth_rate:end
        # @arch denselayer.self-normn-nn-batchnormnd-in_channels:start
        self.norm1 = nn.BatchNorm2d(in_channels)
        # @arch denselayer.self-normn-nn-batchnormnd-in_channels:end
        # @arch denselayer.self-relun-nn-relu-inplace-true:start
        self.relu1 = nn.ReLU(inplace=True)
        # @arch denselayer.self-relun-nn-relu-inplace-true:end
        self.conv1 = nn.Conv2d(
            # @arch denselayer.in_channels.2:start
            in_channels,
            # @arch denselayer.in_channels.2:end
            # @arch denselayer.bottleneck_channels:start
            bottleneck_channels,
            # @arch denselayer.bottleneck_channels:end
            # @arch denselayer.kernel_size-n:start
            kernel_size=1,
            # @arch denselayer.kernel_size-n:end
            # @arch denselayer.stride-n:start
            stride=1,
            # @arch denselayer.stride-n:end
            # @arch denselayer.bias-false:start
            bias=False,
            # @arch denselayer.bias-false:end
        # @arch denselayer.code.4:start
        )
        # @arch denselayer.code.4:end
        # @arch denselayer.self-normn-nn-batchnormnd-bottleneck_channels:start
        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        # @arch denselayer.self-normn-nn-batchnormnd-bottleneck_channels:end
        # @arch denselayer.self-relun-nn-relu-inplace-true.2:start
        self.relu2 = nn.ReLU(inplace=True)
        # @arch denselayer.self-relun-nn-relu-inplace-true.2:end
        self.conv2 = nn.Conv2d(
            # @arch denselayer.bottleneck_channels.2:start
            bottleneck_channels,
            # @arch denselayer.bottleneck_channels.2:end
            # @arch denselayer.growth_rate.2:start
            growth_rate,
            # @arch denselayer.growth_rate.2:end
            # @arch denselayer.kernel_size-n.2:start
            kernel_size=3,
            # @arch denselayer.kernel_size-n.2:end
            # @arch denselayer.stride-n.2:start
            stride=1,
            # @arch denselayer.stride-n.2:end
            # @arch denselayer.padding-n:start
            padding=1,
            # @arch denselayer.padding-n:end
            # @arch denselayer.bias-false.2:start
            bias=False,
            # @arch denselayer.bias-false.2:end
        # @arch denselayer.code.5:start
        )
        # @arch denselayer.code.5:end
        # @arch denselayer.self-dropout_rate-dropout_rate:start
        self.dropout_rate = dropout_rate
        # @arch denselayer.self-dropout_rate-dropout_rate:end

    def forward(self, x):
        # Compress existing features, then produce new growth features.
        # @arch denselayer.forward.out-self-normn-x:start
        out = self.norm1(x)  # (batch, channels, height, width)
        # @arch denselayer.forward.out-self-normn-x:end
        # @arch denselayer.forward.out-self-relun-out:start
        out = self.relu1(out)  # (batch, channels, height, width)
        # @arch denselayer.forward.out-self-relun-out:end
        # @arch denselayer.forward.out-self-convn-out:start
        out = self.conv1(out)  # (batch, channels, height, width) -> (batch, bottleneck_channels, height, width)
        # @arch denselayer.forward.out-self-convn-out:end
        # @arch denselayer.forward.out-self-normn-out:start
        out = self.norm2(out)  # (batch, bottleneck_channels, height, width)
        # @arch denselayer.forward.out-self-normn-out:end
        # @arch denselayer.forward.out-self-relun-out.2:start
        out = self.relu2(out)  # (batch, bottleneck_channels, height, width)
        # @arch denselayer.forward.out-self-relun-out.2:end
        # @arch denselayer.forward.out-self-convn-out.2:start
        out = self.conv2(out)  # (batch, bottleneck_channels, height, width) -> (batch, growth_rate, height, width)
        # @arch denselayer.forward.out-self-convn-out.2:end
        # @arch denselayer.forward.if-self-dropout_rate-n:start
        if self.dropout_rate > 0:
        # @arch denselayer.forward.if-self-dropout_rate-n:end
            out = F.dropout(out, p=self.dropout_rate, training=self.training)  # (batch, growth_rate, height, width)

        # Concatenate old and new features: (batch, channels, height, width) grows by growth_rate.
        # @arch denselayer.forward.features-x-out:start
        features = [x, out]
        # @arch denselayer.forward.features-x-out:end
        # @arch denselayer.forward.out-torch-cat-features-dim-n:start
        out = torch.cat(features, dim=1)  # (batch, channels, height, width) -> (batch, channels + growth_rate, height, width)
        # @arch denselayer.forward.out-torch-cat-features-dim-n:end
        # @arch denselayer.forward.return-out:start
        return out
        # @arch denselayer.forward.return-out:end


# %% [notebook-only]
example_dense_layer = DenseLayer(in_channels=6, growth_rate=4)
example_dense_input = torch.randn(2, 6, 8, 8)  # -> (2, 6, 8, 8)
example_dense_output = example_dense_layer(example_dense_input)  # (2, 6, 8, 8) -> (2, 10, 8, 8)
print(example_dense_output.shape)


# %%
class DenseBlock(nn.Module):
    def __init__(
        # @arch denseblock.self:start
        self,
        # @arch denseblock.self:end
        # @arch denseblock.layer_count:start
        layer_count,  # Number of dense layers in this block.
        # @arch denseblock.layer_count:end
        # @arch denseblock.in_channels:start
        in_channels,  # Number of channels entering the block.
        # @arch denseblock.in_channels:end
        # @arch denseblock.growth_rate:start
        growth_rate,  # Number of new channels added per dense layer.
        # @arch denseblock.growth_rate:end
        # @arch denseblock.dropout_rate-n:start
        dropout_rate=0.0  # Dropout probability inside dense layers.
        # @arch denseblock.dropout_rate-n:end
    # @arch denseblock.code:start
    ):
    # @arch denseblock.code:end
        # @arch denseblock.super-__init__:start
        super().__init__()
        # @arch denseblock.super-__init__:end

        # Register dense layers whose inputs grow after every concatenation.
        # @arch denseblock.layers:start
        layers = []
        # @arch denseblock.layers:end
        # @arch denseblock.current_channels-in_channels:start
        current_channels = in_channels
        # @arch denseblock.current_channels-in_channels:end
        # @arch denseblock.for-_-in-range-layer_count:start
        for _ in range(layer_count):
        # @arch denseblock.for-_-in-range-layer_count:end
            # @arch denseblock.layer-denselayer:start
            layer = DenseLayer(
            # @arch denseblock.layer-denselayer:end
                # @arch denseblock.current_channels:start
                current_channels,
                # @arch denseblock.current_channels:end
                # @arch denseblock.growth_rate.2:start
                growth_rate,
                # @arch denseblock.growth_rate.2:end
                # @arch denseblock.dropout_rate-dropout_rate:start
                dropout_rate=dropout_rate,
                # @arch denseblock.dropout_rate-dropout_rate:end
            # @arch denseblock.code.4:start
            )
            # @arch denseblock.code.4:end
            # @arch denseblock.layers-append-layer:start
            layers.append(layer)
            # @arch denseblock.layers-append-layer:end
            # @arch denseblock.current_channels-current_channels-growth_rate:start
            current_channels = current_channels + growth_rate
            # @arch denseblock.current_channels-current_channels-growth_rate:end
        # @arch denseblock.self-layers-nn-modulelist-layers:start
        self.layers = nn.ModuleList(layers)
        # @arch denseblock.self-layers-nn-modulelist-layers:end
        # @arch denseblock.self-out_channels-current_channels:start
        self.out_channels = current_channels
        # @arch denseblock.self-out_channels-current_channels:end

    # @arch denseblock.def-forward-self-x:start
    def forward(self, x):
    # @arch denseblock.def-forward-self-x:end
        # Feed every layer the full accumulated feature stack.
        # @arch denseblock.forward.out-x:start
        out = x
        # @arch denseblock.forward.out-x:end
        # @arch denseblock.forward.for-layer-in-self-layers:start
        for layer in self.layers:
        # @arch denseblock.forward.for-layer-in-self-layers:end
            # @arch denseblock.forward.out-layer-out:start
            out = layer(out)  # (batch, channels, height, width) -> (batch, channels + growth_rate, height, width)
            # @arch denseblock.forward.out-layer-out:end
        # @arch denseblock.forward.return-out:start
        return out
        # @arch denseblock.forward.return-out:end


# %% [notebook-only]
dense_block = DenseBlock(layer_count=3, in_channels=6, growth_rate=4)
block_input = torch.randn(2, 6, 8, 8)  # -> (2, 6, 8, 8)
example_block_output = dense_block(block_input)  # (2, 6, 8, 8) -> (2, 18, 8, 8)
print(example_block_output.shape)


# %%
# @arch class-transition-nn-module:start
class Transition(nn.Module):
# @arch class-transition-nn-module:end
    def __init__(
        self,
        # @arch transition.in_channels:start
        in_channels,  # Number of channels entering the transition.
        # @arch transition.in_channels:end
        out_channels  # Number of channels after compression.
    ):
        super().__init__()

        # Register channel compression before spatial downsampling.
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # @arch transition.self-conv-nn-convnd:start
        self.conv = nn.Conv2d(
        # @arch transition.self-conv-nn-convnd:end
            # @arch transition.in_channels.2:start
            in_channels,
            # @arch transition.in_channels.2:end
            out_channels,
            # @arch transition.kernel_size-n:start
            kernel_size=1,
            # @arch transition.kernel_size-n:end
            # @arch transition.stride-n:start
            stride=1,
            # @arch transition.stride-n:end
            # @arch transition.bias-false:start
            bias=False,
            # @arch transition.bias-false:end
        # @arch transition.code.4:start
        )
        # @arch transition.code.4:end

    # @arch transition.def-forward-self-x:start
    def forward(self, x):
    # @arch transition.def-forward-self-x:end
        # Compress channels and halve spatial resolution.
        out = self.norm(x)  # (batch, in_channels, height, width)
        # @arch transition.forward.out-self-relu-out:start
        out = self.relu(out)  # (batch, in_channels, height, width)
        # @arch transition.forward.out-self-relu-out:end
        # @arch transition.forward.out-self-conv-out:start
        out = self.conv(out)  # (batch, in_channels, height, width) -> (batch, out_channels, height, width)
        # @arch transition.forward.out-self-conv-out:end
        # @arch transition.forward.out-f-avg_poolnd-out-kernel_size-n-stride-n:start
        out = F.avg_pool2d(out, kernel_size=2, stride=2)  # (batch, out_channels, height, width) -> (batch, out_channels, height/2, width/2)
        # @arch transition.forward.out-f-avg_poolnd-out-kernel_size-n-stride-n:end
        # @arch transition.forward.return-out:start
        return out
        # @arch transition.forward.return-out:end


# %% [notebook-only]
example_transition = Transition(in_channels=18, out_channels=9)
transition_input = torch.randn(2, 18, 8, 8)  # -> (2, 18, 8, 8)
example_transition_output = example_transition(transition_input)  # (2, 18, 8, 8) -> (2, 9, 4, 4)
print(example_transition_output.shape)


# %%
# @arch class-densenet-nn-module:start
class DenseNet(nn.Module):
# @arch class-densenet-nn-module:end
    def __init__(
        self,
        growth_rate=32,  # New channels added by each dense layer.
        block_config=(6, 12, 24, 16),  # DenseNet-121 layer counts.
        num_init_features=64,  # Stem output channels.
        compression=0.5,  # Transition channel compression factor.
        dropout_rate=0.0,  # Dropout probability inside dense layers.
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # Register the ImageNet stem.
        # @arch densenet.stem:start
        self.stem = nn.Sequential(
        # @arch densenet.stem:end
            # @arch densenet.stem-conv:start
            nn.Conv2d(
            # @arch densenet.stem-conv:end
                3,
                # @arch densenet.num_init_features:start
                num_init_features,
                # @arch densenet.num_init_features:end
                # @arch densenet.kernel_size-n:start
                kernel_size=7,
                # @arch densenet.kernel_size-n:end
                # @arch densenet.stride-n:start
                stride=2,
                # @arch densenet.stride-n:end
                # @arch densenet.padding-n:start
                padding=3,
                # @arch densenet.padding-n:end
                # @arch densenet.bias-false:start
                bias=False,
                # @arch densenet.bias-false:end
            # @arch densenet.code.4:start
            ),
            # @arch densenet.code.4:end
            # @arch densenet.nn-batchnormnd-num_init_features:start
            nn.BatchNorm2d(num_init_features),
            # @arch densenet.nn-batchnormnd-num_init_features:end
            # @arch densenet.nn-relu-inplace-true:start
            nn.ReLU(inplace=True),
            # @arch densenet.nn-relu-inplace-true:end
            # @arch densenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n:start
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # @arch densenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n:end
        # @arch densenet.code.5:start
        )
        # @arch densenet.code.5:end

        # Register dense blocks and transition layers.
        # @arch densenet.blocks:start
        blocks = []
        # @arch densenet.blocks:end
        num_features = num_init_features
        for index, layer_count in enumerate(block_config):
            dense_block = DenseBlock(
                layer_count,
                num_features,
                growth_rate,
                # @arch densenet.dropout_rate-dropout_rate:start
                dropout_rate=dropout_rate,
                # @arch densenet.dropout_rate-dropout_rate:end
            # @arch densenet.code.8:start
            )
            # @arch densenet.code.8:end
            # @arch densenet.blocks-append-dense_block:start
            blocks.append(dense_block)
            # @arch densenet.blocks-append-dense_block:end
            # @arch densenet.num_features-dense_block-out_channels:start
            num_features = dense_block.out_channels
            # @arch densenet.num_features-dense_block-out_channels:end
            # @arch densenet.is_last_block-index-len-block_config-n:start
            is_last_block = index == len(block_config) - 1
            # @arch densenet.is_last_block-index-len-block_config-n:end
            # @arch densenet.if-not-is_last_block:start
            if not is_last_block:
            # @arch densenet.if-not-is_last_block:end
                # @arch densenet.out_features-int-num_features-compression:start
                out_features = int(num_features * compression)
                # @arch densenet.out_features-int-num_features-compression:end
                # @arch densenet.transition-transition-num_features-out_features:start
                transition = Transition(num_features, out_features)
                # @arch densenet.transition-transition-num_features-out_features:end
                blocks.append(transition)
                # @arch densenet.num_features-out_features:start
                num_features = out_features
                # @arch densenet.num_features-out_features:end
        # @arch densenet.self-features-nn-sequential-blocks:start
        self.features = nn.Sequential(*blocks)
        # @arch densenet.self-features-nn-sequential-blocks:end
        # @arch densenet.self-norm-nn-batchnormnd-num_features:start
        self.norm = nn.BatchNorm2d(num_features)
        # @arch densenet.self-norm-nn-batchnormnd-num_features:end
        # @arch densenet.self-classifier-nn-linear-num_features-num_classes:start
        self.classifier = nn.Linear(num_features, num_classes)
        # @arch densenet.self-classifier-nn-linear-num_features-num_classes:end

    def forward(self, x):
        # Convert image input into stem features: (batch, 3, 224, 224) -> (batch, 64, 56, 56).
        # @arch densenet.forward.x-self-stem-x:start
        x = self.stem(x)  # (batch, 3, 224, 224) -> (batch, 64, 56, 56)
        # @arch densenet.forward.x-self-stem-x:end

        # Grow and compress feature maps through dense blocks and transitions.
        # @arch densenet.forward.x-self-features-x:start
        x = self.features(x)  # (batch, 64, 56, 56) -> (batch, num_features, 7, 7)
        # @arch densenet.forward.x-self-features-x:end

        # Normalize, pool, and classify final dense features.
        # @arch densenet.forward.x-self-norm-x:start
        x = self.norm(x)  # (batch, num_features, 7, 7)
        # @arch densenet.forward.x-self-norm-x:end
        # @arch densenet.forward.x-f-relu-x-inplace-true:start
        x = F.relu(x, inplace=True)  # (batch, num_features, 7, 7)
        # @arch densenet.forward.x-f-relu-x-inplace-true:end
        # @arch densenet.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n:start
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # (batch, num_features, 7, 7) -> (batch, num_features, 1, 1)
        # @arch densenet.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n:end
        # @arch densenet.forward.x-torch-flatten-x-n:start
        x = torch.flatten(x, 1)  # (batch, num_features, 1, 1) -> (batch, num_features)
        # @arch densenet.forward.x-torch-flatten-x-n:end
        # @arch densenet.forward.logits-self-classifier-x:start
        logits = self.classifier(x)  # (batch, num_features) -> (batch, num_classes)
        # @arch densenet.forward.logits-self-classifier-x:end
        # @arch densenet.forward.return-logits:start
        return logits
        # @arch densenet.forward.return-logits:end


# %% [notebook-only]
# Create and run a compact image batch: (2, 3, 64, 64) -> (2, 10).
example_model = DenseNet(
    growth_rate=4,
    block_config=(1, 1, 1, 1),
    num_init_features=8,
    num_classes=10,
)
example_test_input = torch.randn(2, 3, 64, 64)  # -> (2, 3, 64, 64)
example_logits = example_model(example_test_input)  # (2, 3, 64, 64) -> (2, 10)
print(example_logits.shape)


# %%
# Train on a tiny synthetic image batch.
model = DenseNet(
    growth_rate=4,
    block_config=(1, 1, 1, 1),
    num_init_features=8,
    num_classes=2,
)
train_images = torch.zeros(2, 3, 64, 64)  # -> (2, 3, 64, 64)
train_images[0, :, 8:24, 8:24] = 1.0
train_images[1, :, 40:56, 40:56] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 64, 64) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
