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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        stride=1,  # Spatial stride for the first convolution.
        downsample=None  # Optional projection for the residual path.
    ):
        super().__init__()

        # Register the residual branch and optional projection shortcut.
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        # Preserve the residual path, projecting it when shape changes.
        identity = x

        # Apply the two-convolution residual branch.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual and apply final activation.
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # Register stem, residual stages, global pool, and classifier.
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        # Build the optional projection when a stage changes shape.
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # Stack residual blocks for one ResNet stage.
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            block = BasicBlock(self.in_channels, out_channels)
            layers.append(block)
        stage = nn.Sequential(*layers)
        return stage

    def forward(self, x):
        # Convert image input into stem features.
        x = self.stem(x)
        x = self.maxpool(x)

        # Run residual stages while reducing spatial size.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool final features and classify.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
model = ResNet18(num_classes=1000)
test_input = torch.randn(2, 3, 224, 224)
logits = model(test_input)


# Train on a tiny synthetic image batch.
model = ResNet18(num_classes=2)
train_images = torch.zeros(2, 3, 224, 224)
train_images[0, :, 32:96, 32:96] = 1.0
train_images[1, :, 128:192, 128:192] = 1.0
train_targets = torch.tensor([0, 1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)
    loss = criterion(logits, train_targets)
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
