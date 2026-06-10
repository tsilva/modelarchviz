import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # Register the convolutional feature extractor.
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Register the dense classifier for flattened feature maps.
        flattened_features = 256 * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # Extract convolutional features: (batch, 3, 227, 227) -> (batch, 256, 6, 6).
        x = self.features(x)  # (batch, 3, 227, 227) -> (batch, 256, 6, 6)

        # Flatten feature maps for dense classification: (batch, 256, 6, 6) -> (batch, 9216).
        x = torch.flatten(x, start_dim=1)  # (batch, 256, 6, 6) -> (batch, 9216)

        # Classify flattened features: (batch, 9216) -> (batch, num_classes).
        logits = self.classifier(x)  # (batch, 9216) -> (batch, num_classes)
        return logits

# Train on a tiny synthetic image batch.
model = AlexNet(num_classes=2)
train_images = torch.zeros(2, 3, 227, 227)  # -> (2, 3, 227, 227)
train_images[0, :, 40:100, 40:100] = 1.0
train_images[1, :, 120:180, 120:180] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 227, 227) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
