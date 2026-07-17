# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class AlexNet(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # Register the convolutional feature extractor.
        self.features = nn.Sequential(
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-stride-n:start
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-stride-n:end
            # @arch alexnet.nn-relu-inplace-true:start
            nn.ReLU(inplace=True),
            # @arch alexnet.nn-relu-inplace-true:end
            # @arch alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n:start
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # @arch alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n:end
            # @arch alexnet.nn-maxpoolnd-kernel_size-n-stride-n:start
            nn.MaxPool2d(kernel_size=3, stride=2),
            # @arch alexnet.nn-maxpoolnd-kernel_size-n-stride-n:end
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n:start
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n:end
            # @arch alexnet.nn-relu-inplace-true.2:start
            nn.ReLU(inplace=True),
            # @arch alexnet.nn-relu-inplace-true.2:end
            # @arch alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n.2:start
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # @arch alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n.2:end
            # @arch alexnet.nn-maxpoolnd-kernel_size-n-stride-n.2:start
            nn.MaxPool2d(kernel_size=3, stride=2),
            # @arch alexnet.nn-maxpoolnd-kernel_size-n-stride-n.2:end
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n.2:start
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n.2:end
            # @arch alexnet.nn-relu-inplace-true.3:start
            nn.ReLU(inplace=True),
            # @arch alexnet.nn-relu-inplace-true.3:end
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n.3:start
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n.3:end
            # @arch alexnet.nn-relu-inplace-true.4:start
            nn.ReLU(inplace=True),
            # @arch alexnet.nn-relu-inplace-true.4:end
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n.4:start
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # @arch alexnet.nn-convnd-n-n-kernel_size-n-padding-n.4:end
            # @arch alexnet.nn-relu-inplace-true.5:start
            nn.ReLU(inplace=True),
            # @arch alexnet.nn-relu-inplace-true.5:end
            # @arch alexnet.nn-maxpoolnd-kernel_size-n-stride-n.3:start
            nn.MaxPool2d(kernel_size=3, stride=2),
            # @arch alexnet.nn-maxpoolnd-kernel_size-n-stride-n.3:end
        # @arch alexnet.code.4:start
        )
        # @arch alexnet.code.4:end

        # Register the dense classifier for flattened feature maps.
        flattened_features = 256 * 6 * 6
        # @arch alexnet.self-classifier-nn-sequential:start
        self.classifier = nn.Sequential(
        # @arch alexnet.self-classifier-nn-sequential:end
            # @arch alexnet.nn-dropout-n:start
            nn.Dropout(0.5),
            # @arch alexnet.nn-dropout-n:end
            # @arch alexnet.nn-linear-flattened_features-n:start
            nn.Linear(flattened_features, 4096),
            # @arch alexnet.nn-linear-flattened_features-n:end
            # @arch alexnet.nn-relu-inplace-true.6:start
            nn.ReLU(inplace=True),
            # @arch alexnet.nn-relu-inplace-true.6:end
            # @arch alexnet.nn-dropout-n.2:start
            nn.Dropout(0.5),
            # @arch alexnet.nn-dropout-n.2:end
            # @arch alexnet.nn-linear-n-n:start
            nn.Linear(4096, 4096),
            # @arch alexnet.nn-linear-n-n:end
            # @arch alexnet.nn-relu-inplace-true.7:start
            nn.ReLU(inplace=True),
            # @arch alexnet.nn-relu-inplace-true.7:end
            # @arch alexnet.nn-linear-n-num_classes:start
            nn.Linear(4096, num_classes),
            # @arch alexnet.nn-linear-n-num_classes:end
        )

    def forward(self, x):
        # Extract convolutional features: (batch, 3, 227, 227) -> (batch, 256, 6, 6).
        # @arch alexnet.forward.x-self-features-x:start
        x = self.features(x)  # (batch, 3, 227, 227) -> (batch, 256, 6, 6)
        # @arch alexnet.forward.x-self-features-x:end

        # Flatten feature maps for dense classification: (batch, 256, 6, 6) -> (batch, 9216).
        # @arch alexnet.forward.x-torch-flatten-x-start_dim-n:start
        x = torch.flatten(x, start_dim=1)  # (batch, 256, 6, 6) -> (batch, 9216)
        # @arch alexnet.forward.x-torch-flatten-x-start_dim-n:end

        # Classify flattened features: (batch, 9216) -> (batch, num_classes).
        logits = self.classifier(x)  # (batch, 9216) -> (batch, num_classes)
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 3, 227, 227) -> (2, 1000).
example_model = AlexNet(num_classes=1000)
example_test_input = torch.randn(2, 3, 227, 227)  # -> (2, 3, 227, 227)
example_logits = example_model(example_test_input)  # (2, 3, 227, 227) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
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
