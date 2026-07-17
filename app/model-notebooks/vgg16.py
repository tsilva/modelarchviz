# %%
import torch
import torch.nn as nn


# %%
class VGG16(nn.Module):
    def __init__(
        self,
        num_classes=1000  # Number of output classes.
    ):
        super().__init__()

        # Register five plain convolutional stages with only 3x3 filters.
        # @arch vggn.self-features-nn-sequential:start
        self.features = nn.Sequential(
        # @arch vggn.self-features-nn-sequential:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n:start
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n:end
            # @arch vggn.nn-relu-inplace-true:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.2:start
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.2:end
            # @arch vggn.nn-relu-inplace-true.2:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.2:end
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n:start
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.3:start
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.3:end
            # @arch vggn.nn-relu-inplace-true.3:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.3:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.4:start
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.4:end
            # @arch vggn.nn-relu-inplace-true.4:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.4:end
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.2:start
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.2:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.5:start
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.5:end
            # @arch vggn.nn-relu-inplace-true.5:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.5:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.6:start
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.6:end
            # @arch vggn.nn-relu-inplace-true.6:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.6:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.7:start
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.7:end
            # @arch vggn.nn-relu-inplace-true.7:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.7:end
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.3:start
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.3:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.8:start
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.8:end
            # @arch vggn.nn-relu-inplace-true.8:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.8:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.9:start
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.9:end
            # @arch vggn.nn-relu-inplace-true.9:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.9:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.10:start
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.10:end
            # @arch vggn.nn-relu-inplace-true.10:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.10:end
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.4:start
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.4:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.11:start
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.11:end
            # @arch vggn.nn-relu-inplace-true.11:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.11:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.12:start
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.12:end
            # @arch vggn.nn-relu-inplace-true.12:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.12:end
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.13:start
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # @arch vggn.nn-convnd-n-n-kernel_size-n-padding-n.13:end
            # @arch vggn.nn-relu-inplace-true.13:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.13:end
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.5:start
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @arch vggn.nn-maxpoolnd-kernel_size-n-stride-n.5:end
        )

        # Register the original three-layer dense classifier.
        # @arch vggn.flattened_features-n-n-n:start
        flattened_features = 512 * 7 * 7
        # @arch vggn.flattened_features-n-n-n:end
        # @arch vggn.self-classifier-nn-sequential:start
        self.classifier = nn.Sequential(
        # @arch vggn.self-classifier-nn-sequential:end
            # @arch vggn.nn-linear-flattened_features-n:start
            nn.Linear(flattened_features, 4096),
            # @arch vggn.nn-linear-flattened_features-n:end
            # @arch vggn.nn-relu-inplace-true.14:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.14:end
            # @arch vggn.nn-dropout-n:start
            nn.Dropout(0.5),
            # @arch vggn.nn-dropout-n:end
            # @arch vggn.nn-linear-n-n:start
            nn.Linear(4096, 4096),
            # @arch vggn.nn-linear-n-n:end
            # @arch vggn.nn-relu-inplace-true.15:start
            nn.ReLU(inplace=True),
            # @arch vggn.nn-relu-inplace-true.15:end
            # @arch vggn.nn-dropout-n.2:start
            nn.Dropout(0.5),
            # @arch vggn.nn-dropout-n.2:end
            # @arch vggn.nn-linear-n-num_classes:start
            nn.Linear(4096, num_classes),
            # @arch vggn.nn-linear-n-num_classes:end
        )

    def forward(self, x):
        # Extract hierarchical image features: (batch, 3, 224, 224) -> (batch, 512, 7, 7).
        # @arch vggn.forward.x-self-features-x:start
        x = self.features(x)  # (batch, 3, 224, 224) -> (batch, 512, 7, 7)
        # @arch vggn.forward.x-self-features-x:end

        # Flatten final feature maps for dense classification.
        # @arch vggn.forward.x-torch-flatten-x-start_dim-n:start
        x = torch.flatten(x, start_dim=1)  # (batch, 512, 7, 7) -> (batch, 25088)
        # @arch vggn.forward.x-torch-flatten-x-start_dim-n:end

        # Classify the flattened feature vector.
        # @arch vggn.forward.logits-self-classifier-x:start
        logits = self.classifier(x)  # (batch, 25088) -> (batch, num_classes)
        # @arch vggn.forward.logits-self-classifier-x:end
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
example_model = VGG16(num_classes=1000)
example_test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
example_logits = example_model(example_test_input)  # (2, 3, 224, 224) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = VGG16(num_classes=2)
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
