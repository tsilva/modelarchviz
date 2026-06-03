import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Register two same-resolution convolution blocks.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Preserve spatial size while changing channel width.
        out = self.net(x)
        return out


class UNet(nn.Module):
    def __init__(
        self,
        num_classes=2  # Number of segmentation classes.
    ):
        super().__init__()

        # Register encoder, bottleneck, decoder, and segmentation head.
        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encode features while reducing spatial size at each stage.
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Process the bottleneck at the smallest spatial resolution.
        b = self.bottleneck(p4)

        # Decode and concatenate skip features back to full resolution.
        x = self.up4(b)
        x = torch.cat([x, d4], dim=1)
        x = self.dec4(x)
        x = self.up3(x)
        x = torch.cat([x, d3], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.dec1(x)

        # Project decoder features to segmentation logits.
        logits = self.out_conv(x)
        return logits


# Create and run a sample image batch: (2, 1, 572, 572) -> (2, 2, 572, 572).
model = UNet(num_classes=2)
test_input = torch.randn(2, 1, 572, 572)
logits = model(test_input)

# logits: (2, 2, 572, 572)

# Train on two synthetic segmentation masks.
model = UNet(num_classes=2)
train_images = torch.zeros(2, 1, 64, 64)
train_images[0, :, 8:32, 8:32] = 1.0
train_images[1, :, 32:56, 32:56] = 1.0
train_targets = torch.zeros(2, 64, 64, dtype=torch.long)
train_targets[0, 8:32, 8:32] = 1
train_targets[1, 32:56, 32:56] = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)
    loss = criterion(logits, train_targets)
    loss.backward()
    optimizer.step()

final_loss = loss.item()
