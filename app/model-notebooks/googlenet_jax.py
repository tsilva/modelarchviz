# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-inceptionblock-nn-module:start
class InceptionBlock(nn.Module):
# @arch class-inceptionblock-nn-module:end
    # @arch inceptionblock.branchn_channels-int:start
    branch1_channels: int
    # @arch inceptionblock.branchn_channels-int:end
    # @arch inceptionblock.branchn_reduce-int:start
    branch3_reduce: int
    # @arch inceptionblock.branchn_reduce-int:end
    # @arch inceptionblock.branchn_channels-int.2:start
    branch3_channels: int
    # @arch inceptionblock.branchn_channels-int.2:end
    # @arch inceptionblock.branchn_reduce-int.2:start
    branch5_reduce: int
    # @arch inceptionblock.branchn_reduce-int.2:end
    # @arch inceptionblock.branchn_channels-int.3:start
    branch5_channels: int
    # @arch inceptionblock.branchn_channels-int.3:end
    # @arch inceptionblock.pool_channels-int:start
    pool_channels: int
    # @arch inceptionblock.pool_channels-int:end

    # @arch inceptionblock.nn-compact:start
    @nn.compact
    # @arch inceptionblock.nn-compact:end
    # @arch inceptionblock.def-__call__-self-x:start
    def __call__(self, x):
    # @arch inceptionblock.def-__call__-self-x:end
        # Evaluate parallel 1x1 and larger-kernel branches.
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x:start
        branch1 = nn.Conv(self.branch1_channels, (1, 1), name='branch1')(x)  # (batch, height, width, in_channels) -> (batch, height, width, branch1_channels)
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x:end
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn:start
        branch1 = nn.relu(branch1)  # (batch, height, width, branch1_channels)
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn:end

        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x:start
        branch3 = nn.Conv(self.branch3_reduce, (1, 1), name='branch3_reduce')(x)  # (batch, height, width, in_channels) -> (batch, height, width, branch3_reduce)
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x:end
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.2:start
        branch3 = nn.relu(branch3)  # (batch, height, width, branch3_reduce)
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.2:end
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran:start
        branch3 = nn.Conv(self.branch3_channels, (3, 3), padding='SAME', name='branch3')(branch3)  # (batch, height, width, branch3_reduce) -> (batch, height, width, branch3_channels)
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran:end
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.3:start
        branch3 = nn.relu(branch3)  # (batch, height, width, branch3_channels)
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.3:end

        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2:start
        branch5 = nn.Conv(self.branch5_reduce, (1, 1), name='branch5_reduce')(x)  # (batch, height, width, in_channels) -> (batch, height, width, branch5_reduce)
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2:end
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.4:start
        branch5 = nn.relu(branch5)  # (batch, height, width, branch5_reduce)
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.4:end
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2:start
        branch5 = nn.Conv(self.branch5_channels, (5, 5), padding='SAME', name='branch5')(branch5)  # (batch, height, width, branch5_reduce) -> (batch, height, width, branch5_channels)
        # @arch inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2:end
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.5:start
        branch5 = nn.relu(branch5)  # (batch, height, width, branch5_channels)
        # @arch inceptionblock.__call__.branchn-nn-relu-branchn.5:end

        # @arch inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same:start
        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')  # (batch, height, width, in_channels)
        # @arch inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same:end
        # @arch inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool:start
        branch_pool = nn.Conv(self.pool_channels, (1, 1), name='pool_proj')(branch_pool)  # (batch, height, width, in_channels) -> (batch, height, width, pool_channels)
        # @arch inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool:end
        # @arch inceptionblock.__call__.branch_pool-nn-relu-branch_pool:start
        branch_pool = nn.relu(branch_pool)  # (batch, height, width, pool_channels)
        # @arch inceptionblock.__call__.branch_pool-nn-relu-branch_pool:end

        # Concatenate branch channels: list of (batch, height, width, channels) -> one feature map.
        # @arch inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool:start
        branches = [branch1, branch3, branch5, branch_pool]
        # @arch inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool:end
        # @arch inceptionblock.__call__.x-jnp-concatenate-branches-axis-n:start
        x = jnp.concatenate(branches, axis=-1)  # list of (batch, height, width, channels) -> (batch, height, width, output_channels)
        # @arch inceptionblock.__call__.x-jnp-concatenate-branches-axis-n:end
        # @arch inceptionblock.__call__.return-x:start
        return x
        # @arch inceptionblock.__call__.return-x:end


# %% [notebook-only]
# Create and run one Inception block: (2, 16, 16, 8) -> (2, 16, 16, 16).
example_block = InceptionBlock(4, 4, 4, 4, 4, 4)
block_input = jnp.ones((2, 16, 16, 8))  # -> (2, 16, 16, 8)
example_params = example_block.init(jax.random.PRNGKey(0), block_input)
example_block_output = example_block.apply(example_params, block_input)  # (2, 16, 16, 8) -> (2, 16, 16, 16)
print("block_output shape:", example_block_output.shape)

# %%
class GoogLeNet(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Downsample the input into stem features: (batch, 224, 224, 3) -> stem feature maps.
        # @arch googlenet.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-name-stem_convn-x:start
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME', name='stem_conv7')(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, 64)
        # @arch googlenet.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-name-stem_convn-x:end
        # @arch googlenet.__call__.x-nn-relu-x:start
        x = nn.relu(x)  # (batch, 112, 112, 64)
        # @arch googlenet.__call__.x-nn-relu-x:end
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 112, 112, 64) -> (batch, 56, 56, 64)
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same:end
        # @arch googlenet.__call__.x-nn-conv-n-n-n-name-stem_convn-x:start
        x = nn.Conv(64, (1, 1), name='stem_conv1')(x)  # (batch, 56, 56, 64)
        # @arch googlenet.__call__.x-nn-conv-n-n-n-name-stem_convn-x:end
        # @arch googlenet.__call__.x-nn-relu-x.2:start
        x = nn.relu(x)  # (batch, 56, 56, 64)
        # @arch googlenet.__call__.x-nn-relu-x.2:end
        # @arch googlenet.__call__.x-nn-conv-n-n-n-padding-same-name-stem_convn-x:start
        x = nn.Conv(192, (3, 3), padding='SAME', name='stem_conv3')(x)  # (batch, 56, 56, 64) -> (batch, 56, 56, 192)
        # @arch googlenet.__call__.x-nn-conv-n-n-n-padding-same-name-stem_convn-x:end
        # @arch googlenet.__call__.x-nn-relu-x.3:start
        x = nn.relu(x)  # (batch, 56, 56, 192)
        # @arch googlenet.__call__.x-nn-relu-x.3:end
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.2:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 56, 56, 192) -> (batch, 28, 28, 192)
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.2:end

        # Run Inception stage 3 and downsample spatial size.
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x:start
        x = InceptionBlock(64, 96, 128, 16, 32, 32, name='inception3a')(x)  # (batch, 28, 28, 192) -> (batch, 28, 28, 256)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x:end
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x:start
        x = InceptionBlock(128, 128, 192, 32, 96, 64, name='inception3b')(x)  # (batch, 28, 28, 256) -> (batch, 28, 28, 480)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x:end
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.3:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 28, 28, 480) -> (batch, 14, 14, 480)
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.3:end

        # Run Inception stage 4 and downsample spatial size.
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.2:start
        x = InceptionBlock(192, 96, 208, 16, 48, 64, name='inception4a')(x)  # (batch, 14, 14, 480) -> (batch, 14, 14, 512)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.2:end
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x.2:start
        x = InceptionBlock(160, 112, 224, 24, 64, 64, name='inception4b')(x)  # (batch, 14, 14, 512)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x.2:end
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnc-x:start
        x = InceptionBlock(128, 128, 256, 24, 64, 64, name='inception4c')(x)  # (batch, 14, 14, 512)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnc-x:end
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnd-x:start
        x = InceptionBlock(112, 144, 288, 32, 64, 64, name='inception4d')(x)  # (batch, 14, 14, 512) -> (batch, 14, 14, 528)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnd-x:end
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionne-x:start
        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception4e')(x)  # (batch, 14, 14, 528) -> (batch, 14, 14, 832)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionne-x:end
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.4:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 14, 14, 832) -> (batch, 7, 7, 832)
        # @arch googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.4:end

        # Run Inception stage 5 and pool to a classifier vector.
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.3:start
        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception5a')(x)  # (batch, 7, 7, 832)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.3:end
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x.3:start
        x = InceptionBlock(384, 192, 384, 48, 128, 128, name='inception5b')(x)  # (batch, 7, 7, 832) -> (batch, 7, 7, 1024)
        # @arch googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x.3:end

        # Apply dropout and classify pooled features: (batch, 1024) -> (batch, num_classes).
        # @arch googlenet.__call__.x-jnp-mean-x-axis-n-n:start
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, 1024) -> (batch, 1024)
        # @arch googlenet.__call__.x-jnp-mean-x-axis-n-n:end
        # @arch googlenet.__call__.x-nn-dropout-n-deterministic-not-train-x:start
        x = nn.Dropout(0.4, deterministic=not train)(x)  # (batch, 1024)
        # @arch googlenet.__call__.x-nn-dropout-n-deterministic-not-train-x:end
        # @arch googlenet.__call__.logits-nn-dense-self-num_classes-name-fc-x:start
        logits = nn.Dense(self.num_classes, name='fc')(x)  # (batch, 1024) -> (batch, num_classes)
        # @arch googlenet.__call__.logits-nn-dense-self-num_classes-name-fc-x:end
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
example_model = GoogLeNet(num_classes=1000)
example_test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input, train=False)
example_logits = example_model.apply(example_params, example_test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = GoogLeNet(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images, train=False)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs, train=False)  # (batch, 224, 224, 3) -> (batch, num_classes)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, classes) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss
