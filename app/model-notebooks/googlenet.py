# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input feature channels.
        branch1_channels,  # Output channels for the 1x1 branch.
        branch3_reduce,  # Reduction channels before the 3x3 branch.
        branch3_channels,  # Output channels for the 3x3 branch.
        branch5_reduce,  # Reduction channels before the 5x5 branch.
        branch5_channels,  # Output channels for the 5x5 branch.
        pool_channels  # Output channels for the pooling branch.
    ):
        super().__init__()

        # Register the four parallel Inception branches.
        self.branch1 = nn.Sequential(
            # @arch inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n:start
            nn.Conv2d(in_channels, branch1_channels, kernel_size=1),
            # @arch inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n:end
            # @arch inceptionblock.nn-relu-inplace-true:start
            nn.ReLU(inplace=True),
            # @arch inceptionblock.nn-relu-inplace-true:end
        )
        self.branch3 = nn.Sequential(
            # @arch inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n:start
            nn.Conv2d(in_channels, branch3_reduce, kernel_size=1),
            # @arch inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n:end
            # @arch inceptionblock.nn-relu-inplace-true.2:start
            nn.ReLU(inplace=True),
            # @arch inceptionblock.nn-relu-inplace-true.2:end
            # @arch inceptionblock.branch3-conv:start
            nn.Conv2d(branch3_reduce, branch3_channels, kernel_size=3, padding=1),
            # @arch inceptionblock.branch3-conv:end
            # @arch inceptionblock.nn-relu-inplace-true.3:start
            nn.ReLU(inplace=True),
            # @arch inceptionblock.nn-relu-inplace-true.3:end
        )
        self.branch5 = nn.Sequential(
            # @arch inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2:start
            nn.Conv2d(in_channels, branch5_reduce, kernel_size=1),
            # @arch inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2:end
            # @arch inceptionblock.nn-relu-inplace-true.4:start
            nn.ReLU(inplace=True),
            # @arch inceptionblock.nn-relu-inplace-true.4:end
            # @arch inceptionblock.branch5-conv:start
            nn.Conv2d(branch5_reduce, branch5_channels, kernel_size=5, padding=2),
            # @arch inceptionblock.branch5-conv:end
            # @arch inceptionblock.nn-relu-inplace-true.5:start
            nn.ReLU(inplace=True),
            # @arch inceptionblock.nn-relu-inplace-true.5:end
        )
        self.branch_pool = nn.Sequential(
            # @arch inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n:start
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # @arch inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n:end
            # @arch inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n:start
            nn.Conv2d(in_channels, pool_channels, kernel_size=1),
            # @arch inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n:end
            # @arch inceptionblock.nn-relu-inplace-true.6:start
            nn.ReLU(inplace=True),
            # @arch inceptionblock.nn-relu-inplace-true.6:end
        # @arch inceptionblock.code.7:start
        )
        # @arch inceptionblock.code.7:end

    def forward(self, x):
        # Evaluate parallel branches while preserving spatial size.
        branch1 = self.branch1(x)  # (batch, in_channels, height, width) -> (batch, branch1_channels, height, width)
        # @arch inceptionblock.forward.branchn-self-branchn-x.2:start
        branch3 = self.branch3(x)  # (batch, in_channels, height, width) -> (batch, branch3_channels, height, width)
        # @arch inceptionblock.forward.branchn-self-branchn-x.2:end
        # @arch inceptionblock.forward.branchn-self-branchn-x.3:start
        branch5 = self.branch5(x)  # (batch, in_channels, height, width) -> (batch, branch5_channels, height, width)
        # @arch inceptionblock.forward.branchn-self-branchn-x.3:end
        # @arch inceptionblock.forward.branch_pool-self-branch_pool-x:start
        branch_pool = self.branch_pool(x)  # (batch, in_channels, height, width) -> (batch, pool_channels, height, width)
        # @arch inceptionblock.forward.branch_pool-self-branch_pool-x:end

        # Concatenate branch channels: list of (batch, channels, height, width) -> one feature map.
        branches = [branch1, branch3, branch5, branch_pool]
        # @arch inceptionblock.forward.x-torch-cat-branches-dim-n:start
        x = torch.cat(branches, dim=1)  # list of (batch, channels, height, width) -> (batch, output_channels, height, width)
        # @arch inceptionblock.forward.x-torch-cat-branches-dim-n:end
        # @arch inceptionblock.forward.return-x:start
        return x
        # @arch inceptionblock.forward.return-x:end


# %% [notebook-only]
# Create and run one Inception block: (2, 8, 16, 16) -> (2, 16, 16, 16).
example_block = InceptionBlock(8, 4, 4, 4, 4, 4, 4)
block_input = torch.randn(2, 8, 16, 16)  # -> (2, 8, 16, 16)
example_block_output = example_block(block_input)  # (2, 8, 16, 16) -> (2, 16, 16, 16)
print("block_output shape:", example_block_output.shape)

# %%
# @arch class-googlenet-nn-module:start
class GoogLeNet(nn.Module):
# @arch class-googlenet-nn-module:end
    # @arch googlenet.def-__init__:start
    def __init__(
    # @arch googlenet.def-__init__:end
        # @arch googlenet.self:start
        self,
        # @arch googlenet.self:end
        # @arch googlenet.num_classes-n:start
        num_classes=1000  # Number of output classes.
        # @arch googlenet.num_classes-n:end
    # @arch googlenet.code:start
    ):
    # @arch googlenet.code:end
        # @arch googlenet.super-__init__:start
        super().__init__()
        # @arch googlenet.super-__init__:end

        # Register stem, Inception stages, and classifier head.
        # @arch googlenet.self-stem-nn-sequential:start
        self.stem = nn.Sequential(
        # @arch googlenet.self-stem-nn-sequential:end
            # @arch googlenet.nn-convnd-n-n-kernel_size-n-stride-n-padding-n:start
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            # @arch googlenet.nn-convnd-n-n-kernel_size-n-stride-n-padding-n:end
            # @arch googlenet.nn-relu-inplace-true:start
            nn.ReLU(inplace=True),
            # @arch googlenet.nn-relu-inplace-true:end
            # @arch googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n:start
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # @arch googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n:end
            # @arch googlenet.nn-convnd-n-n-kernel_size-n:start
            nn.Conv2d(64, 64, kernel_size=1),
            # @arch googlenet.nn-convnd-n-n-kernel_size-n:end
            # @arch googlenet.nn-relu-inplace-true.2:start
            nn.ReLU(inplace=True),
            # @arch googlenet.nn-relu-inplace-true.2:end
            # @arch googlenet.nn-convnd-n-n-kernel_size-n-padding-n:start
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            # @arch googlenet.nn-convnd-n-n-kernel_size-n-padding-n:end
            # @arch googlenet.nn-relu-inplace-true.3:start
            nn.ReLU(inplace=True),
            # @arch googlenet.nn-relu-inplace-true.3:end
            # @arch googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2:start
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # @arch googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2:end
        # @arch googlenet.code.4:start
        )
        # @arch googlenet.code.4:end
        # @arch googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n:start
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        # @arch googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n:end
        # @arch googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n:start
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        # @arch googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n:end
        # @arch googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2:start
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        # @arch googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2:end
        # @arch googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2:start
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        # @arch googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2:end
        # @arch googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n:start
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        # @arch googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n:end
        # @arch googlenet.self-inceptionnd-inceptionblock-n-n-n-n-n-n-n:start
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        # @arch googlenet.self-inceptionnd-inceptionblock-n-n-n-n-n-n-n:end
        # @arch googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n:start
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        # @arch googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n:end
        # @arch googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3:start
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        # @arch googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3:end
        # @arch googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.3:start
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        # @arch googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.3:end
        # @arch googlenet.self-avgpool-nn-adaptiveavgpoolnd-n-n:start
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # @arch googlenet.self-avgpool-nn-adaptiveavgpoolnd-n-n:end
        # @arch googlenet.self-dropout-nn-dropout-n:start
        self.dropout = nn.Dropout(0.4)
        # @arch googlenet.self-dropout-nn-dropout-n:end
        # @arch googlenet.self-fc-nn-linear-n-num_classes:start
        self.fc = nn.Linear(1024, num_classes)
        # @arch googlenet.self-fc-nn-linear-n-num_classes:end

    def forward(self, x):
        # Downsample the input into stem features: (batch, 3, 224, 224) -> (batch, 192, 28, 28).
        # @arch googlenet.input:start
        x = self.stem(x)  # (batch, 3, 224, 224) -> (batch, 192, 28, 28)
        # @arch googlenet.input:end

        # Run Inception stage 3 and downsample spatial size.
        x = self.inception3a(x)  # (batch, 192, 28, 28) -> (batch, 256, 28, 28)
        x = self.inception3b(x)  # (batch, 256, 28, 28) -> (batch, 480, 28, 28)
        # @arch googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n:start
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)  # (batch, 480, 28, 28) -> (batch, 480, 14, 14)
        # @arch googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n:end

        # Run Inception stage 4 and downsample spatial size.
        x = self.inception4a(x)  # (batch, 480, 14, 14) -> (batch, 512, 14, 14)
        x = self.inception4b(x)  # (batch, 512, 14, 14)
        # @arch googlenet.forward.x-self-inceptionnc-x:start
        x = self.inception4c(x)  # (batch, 512, 14, 14)
        # @arch googlenet.forward.x-self-inceptionnc-x:end
        # @arch googlenet.forward.x-self-inceptionnd-x:start
        x = self.inception4d(x)  # (batch, 512, 14, 14) -> (batch, 528, 14, 14)
        # @arch googlenet.forward.x-self-inceptionnd-x:end
        # @arch googlenet.forward.x-self-inceptionne-x:start
        x = self.inception4e(x)  # (batch, 528, 14, 14) -> (batch, 832, 14, 14)
        # @arch googlenet.forward.x-self-inceptionne-x:end
        # @arch googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n.2:start
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)  # (batch, 832, 14, 14) -> (batch, 832, 7, 7)
        # @arch googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n.2:end

        # Run Inception stage 5 and pool to a classifier vector.
        x = self.inception5a(x)  # (batch, 832, 7, 7)
        x = self.inception5b(x)  # (batch, 832, 7, 7) -> (batch, 1024, 7, 7)
        # @arch googlenet.forward.x-self-avgpool-x:start
        x = self.avgpool(x)  # (batch, 1024, 7, 7) -> (batch, 1024, 1, 1)
        # @arch googlenet.forward.x-self-avgpool-x:end
        # @arch googlenet.forward.x-torch-flatten-x-start_dim-n:start
        x = torch.flatten(x, start_dim=1)  # (batch, 1024, 1, 1) -> (batch, 1024)
        # @arch googlenet.forward.x-torch-flatten-x-start_dim-n:end

        # Apply dropout and classify pooled features: (batch, 1024) -> (batch, num_classes).
        # @arch googlenet.classifier.dropout:start
        x = self.dropout(x)  # (batch, 1024)
        # @arch googlenet.classifier.dropout:end
        # @arch googlenet.classifier.fc:start
        logits = self.fc(x)  # (batch, 1024) -> (batch, num_classes)
        # @arch googlenet.classifier.fc:end
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
example_model = GoogLeNet(num_classes=1000)
example_test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
example_logits = example_model(example_test_input)  # (2, 3, 224, 224) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = GoogLeNet(num_classes=2)
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
