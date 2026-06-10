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
        bottleneck_channels = bottleneck_width * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # Compress existing features, then produce new growth features.
        out = self.norm1(x)  # (batch, channels, height, width)
        out = self.relu1(out)  # (batch, channels, height, width)
        out = self.conv1(out)  # (batch, channels, height, width) -> (batch, bottleneck_channels, height, width)
        out = self.norm2(out)  # (batch, bottleneck_channels, height, width)
        out = self.relu2(out)  # (batch, bottleneck_channels, height, width)
        out = self.conv2(out)  # (batch, bottleneck_channels, height, width) -> (batch, growth_rate, height, width)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)  # (batch, growth_rate, height, width)

        # Concatenate old and new features: (batch, channels, height, width) grows by growth_rate.
        features = [x, out]
        out = torch.cat(features, dim=1)  # (batch, channels, height, width) -> (batch, channels + growth_rate, height, width)
        return out


# %% [notebook-only]
example_dense_layer = DenseLayer(in_channels=6, growth_rate=4)
example_dense_input = torch.randn(2, 6, 8, 8)  # -> (2, 6, 8, 8)
example_dense_output = example_dense_layer(example_dense_input)  # (2, 6, 8, 8) -> (2, 10, 8, 8)
print(example_dense_output.shape)


# %%
class DenseBlock(nn.Module):
    def __init__(
        self,
        layer_count,  # Number of dense layers in this block.
        in_channels,  # Number of channels entering the block.
        growth_rate,  # Number of new channels added per dense layer.
        dropout_rate=0.0  # Dropout probability inside dense layers.
    ):
        super().__init__()

        # Register dense layers whose inputs grow after every concatenation.
        layers = []
        current_channels = in_channels
        for _ in range(layer_count):
            layer = DenseLayer(
                current_channels,
                growth_rate,
                dropout_rate=dropout_rate,
            )
            layers.append(layer)
            current_channels = current_channels + growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = current_channels

    def forward(self, x):
        # Feed every layer the full accumulated feature stack.
        out = x
        for layer in self.layers:
            out = layer(out)  # (batch, channels, height, width) -> (batch, channels + growth_rate, height, width)
        return out


# %% [notebook-only]
dense_block = DenseBlock(layer_count=3, in_channels=6, growth_rate=4)
block_input = torch.randn(2, 6, 8, 8)  # -> (2, 6, 8, 8)
example_block_output = dense_block(block_input)  # (2, 6, 8, 8) -> (2, 18, 8, 8)
print(example_block_output.shape)


# %%
class Transition(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of channels entering the transition.
        out_channels  # Number of channels after compression.
    ):
        super().__init__()

        # Register channel compression before spatial downsampling.
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def forward(self, x):
        # Compress channels and halve spatial resolution.
        out = self.norm(x)  # (batch, in_channels, height, width)
        out = self.relu(out)  # (batch, in_channels, height, width)
        out = self.conv(out)  # (batch, in_channels, height, width) -> (batch, out_channels, height, width)
        out = F.avg_pool2d(out, kernel_size=2, stride=2)  # (batch, out_channels, height, width) -> (batch, out_channels, height/2, width/2)
        return out


# %% [notebook-only]
example_transition = Transition(in_channels=18, out_channels=9)
transition_input = torch.randn(2, 18, 8, 8)  # -> (2, 18, 8, 8)
example_transition_output = example_transition(transition_input)  # (2, 18, 8, 8) -> (2, 9, 4, 4)
print(example_transition_output.shape)


# %%
class DenseNet(nn.Module):
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
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Register dense blocks and transition layers.
        blocks = []
        num_features = num_init_features
        for index, layer_count in enumerate(block_config):
            dense_block = DenseBlock(
                layer_count,
                num_features,
                growth_rate,
                dropout_rate=dropout_rate,
            )
            blocks.append(dense_block)
            num_features = dense_block.out_channels
            is_last_block = index == len(block_config) - 1
            if not is_last_block:
                out_features = int(num_features * compression)
                transition = Transition(num_features, out_features)
                blocks.append(transition)
                num_features = out_features
        self.features = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Convert image input into stem features: (batch, 3, 224, 224) -> (batch, 64, 56, 56).
        x = self.stem(x)  # (batch, 3, 224, 224) -> (batch, 64, 56, 56)

        # Grow and compress feature maps through dense blocks and transitions.
        x = self.features(x)  # (batch, 64, 56, 56) -> (batch, num_features, 7, 7)

        # Normalize, pool, and classify final dense features.
        x = self.norm(x)  # (batch, num_features, 7, 7)
        x = F.relu(x, inplace=True)  # (batch, num_features, 7, 7)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # (batch, num_features, 7, 7) -> (batch, num_features, 1, 1)
        x = torch.flatten(x, 1)  # (batch, num_features, 1, 1) -> (batch, num_features)
        logits = self.classifier(x)  # (batch, num_features) -> (batch, num_classes)
        return logits


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
