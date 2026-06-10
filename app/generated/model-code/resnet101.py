import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels before expansion.
        stride=1,  # Spatial stride for the first convolution.
        downsample=None  # Optional projection for the residual path.
    ):
        super().__init__()

        # Register the two-convolution residual branch and optional projection shortcut.
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
        identity = x  # (batch, in_channels, height, width)

        # Apply the residual branch.
        out = self.conv1(x)  # (batch, in_channels, height, width) -> (batch, out_channels, out_height, out_width)
        out = self.bn1(out)  # (batch, out_channels, out_height, out_width)
        out = self.relu(out)  # (batch, out_channels, out_height, out_width)
        out = self.conv2(out)  # (batch, out_channels, out_height, out_width)
        out = self.bn2(out)  # (batch, out_channels, out_height, out_width)
        if self.downsample is not None:
            identity = self.downsample(x)  # (batch, in_channels, height, width) -> (batch, out_channels, out_height, out_width)

        # Add residual and apply final activation.
        out = out + identity  # (batch, out_channels, out_height, out_width)
        out = self.relu(out)  # (batch, out_channels, out_height, out_width)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Bottleneck width before expansion.
        stride=1,  # Spatial stride for the 3x3 convolution.
        downsample=None  # Optional projection for the residual path.
    ):
        super().__init__()

        # Register the 1x1, 3x3, 1x1 bottleneck branch and optional projection shortcut.
        expanded_channels = out_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # Preserve the residual path, projecting it when shape changes.
        identity = x  # (batch, in_channels, height, width)

        # Reduce, process, and expand channels in the residual branch.
        out = self.conv1(x)  # (batch, in_channels, height, width) -> (batch, out_channels, height, width)
        out = self.bn1(out)  # (batch, out_channels, height, width)
        out = self.relu(out)  # (batch, out_channels, height, width)
        out = self.conv2(out)  # (batch, out_channels, height, width) -> (batch, out_channels, out_height, out_width)
        out = self.bn2(out)  # (batch, out_channels, out_height, out_width)
        out = self.relu(out)  # (batch, out_channels, out_height, out_width)
        out = self.conv3(out)  # (batch, out_channels, out_height, out_width) -> (batch, expanded_channels, out_height, out_width)
        out = self.bn3(out)  # (batch, expanded_channels, out_height, out_width)
        if self.downsample is not None:
            identity = self.downsample(x)  # (batch, in_channels, height, width) -> (batch, expanded_channels, out_height, out_width)

        # Add residual and apply final activation.
        out = out + identity  # (batch, expanded_channels, out_height, out_width)
        out = self.relu(out)  # (batch, expanded_channels, out_height, out_width)
        return out

class ResNet101(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # Register stem, residual stages, global pool, and classifier.
        block = Bottleneck
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(block, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(block, 256, blocks=23, stride=2)
        self.layer4 = self._make_layer(block, 512, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        # Build the optional projection when a stage changes shape.
        expanded_channels = out_channels * block.expansion
        downsample = None
        if stride != 1 or self.in_channels != expanded_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, expanded_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_channels),
            )

        # Stack residual blocks for one ResNet stage.
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = expanded_channels
        for _ in range(1, blocks):
            current_block = block(self.in_channels, out_channels)
            layers.append(current_block)
        stage = nn.Sequential(*layers)
        return stage

    def forward(self, x):
        # Convert image input into stem features.
        x = self.stem(x)  # (batch, 3, 224, 224) -> (batch, 64, 112, 112)
        x = self.maxpool(x)  # (batch, 64, 112, 112) -> (batch, 64, 56, 56)

        # Run residual stages while reducing spatial size.
        x = self.layer1(x)  # (batch, 64, 56, 56)
        x = self.layer2(x)  # (batch, channels, 56, 56) -> (batch, channels, 28, 28)
        x = self.layer3(x)  # (batch, channels, 28, 28) -> (batch, channels, 14, 14)
        x = self.layer4(x)  # (batch, channels, 14, 14) -> (batch, channels, 7, 7)

        # Pool final features and classify.
        x = self.avgpool(x)  # (batch, channels, 7, 7) -> (batch, channels, 1, 1)
        x = torch.flatten(x, 1)  # (batch, channels, 1, 1) -> (batch, channels)
        logits = self.fc(x)  # (batch, channels) -> (batch, num_classes)
        return logits

# Train on a tiny synthetic image batch.
model = ResNet101(num_classes=2)
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
