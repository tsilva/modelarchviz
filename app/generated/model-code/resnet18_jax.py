import jax
import jax.numpy as jnp
from flax import linen as nn

class BasicBlock(nn.Module):
    out_channels: int
    stride: int = 1
    use_projection: bool = False
    expansion = 1

    @nn.compact
    def __call__(self, x, train=False):
        # Preserve the residual path, projecting it when shape changes.
        identity = x  # (batch, height, width, in_channels)

        # Apply the two-convolution residual branch.
        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding="SAME", use_bias=False, name="conv1")(x)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn1")(y)  # (batch, out_height, out_width, out_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, out_channels)
        y = nn.Conv(self.out_channels, (3, 3), padding="SAME", use_bias=False, name="conv2")(y)  # (batch, out_height, out_width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn2")(y)  # (batch, out_height, out_width, out_channels)
        if self.use_projection:
            identity = nn.Conv(self.out_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False, name="downsample_conv")(x)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, out_channels)
            identity = nn.BatchNorm(use_running_average=not train, name="downsample_bn")(identity)  # (batch, out_height, out_width, out_channels)

        # Add residual and apply final activation.
        y = y + identity  # (batch, out_height, out_width, out_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, out_channels)
        return y

class Bottleneck(nn.Module):
    out_channels: int
    stride: int = 1
    use_projection: bool = False
    expansion = 4

    @nn.compact
    def __call__(self, x, train=False):
        # Preserve the residual path, projecting it when shape changes.
        identity = x  # (batch, height, width, in_channels)
        expanded_channels = self.out_channels * self.expansion

        # Reduce, process, and expand channels in the residual branch.
        y = nn.Conv(self.out_channels, (1, 1), use_bias=False, name="conv1")(x)  # (batch, height, width, in_channels) -> (batch, height, width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn1")(y)  # (batch, height, width, out_channels)
        y = nn.relu(y)  # (batch, height, width, out_channels)
        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding="SAME", use_bias=False, name="conv2")(y)  # (batch, height, width, out_channels) -> (batch, out_height, out_width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn2")(y)  # (batch, out_height, out_width, out_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, out_channels)
        y = nn.Conv(expanded_channels, (1, 1), use_bias=False, name="conv3")(y)  # (batch, out_height, out_width, out_channels) -> (batch, out_height, out_width, expanded_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn3")(y)  # (batch, out_height, out_width, expanded_channels)
        if self.use_projection:
            identity = nn.Conv(expanded_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False, name="downsample_conv")(x)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, expanded_channels)
            identity = nn.BatchNorm(use_running_average=not train, name="downsample_bn")(identity)  # (batch, out_height, out_width, expanded_channels)

        # Add residual and apply final activation.
        y = y + identity  # (batch, out_height, out_width, expanded_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, expanded_channels)
        return y

class ResNet18(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Convert image input into stem features.
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False, name="stem_conv")(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, 64)
        x = nn.BatchNorm(use_running_average=not train, name="stem_bn")(x)  # (batch, 112, 112, 64)
        x = nn.relu(x)  # (batch, 112, 112, 64)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")  # (batch, 112, 112, 64) -> (batch, 56, 56, 64)

        # Run residual stages while reducing spatial size.
        block = BasicBlock
        x = self._stage(x, block, 64, blocks=2, stride=1, train=train)  # (batch, 56, 56, 64)
        x = self._stage(x, block, 128, blocks=2, stride=2, train=train)  # (batch, 56, 56, channels) -> (batch, 28, 28, channels)
        x = self._stage(x, block, 256, blocks=2, stride=2, train=train)  # (batch, 28, 28, channels) -> (batch, 14, 14, channels)
        x = self._stage(x, block, 512, blocks=2, stride=2, train=train)  # (batch, 14, 14, channels) -> (batch, 7, 7, channels)

        # Pool final features and classify.
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, channels) -> (batch, channels)
        logits = nn.Dense(self.num_classes, name="fc")(x)  # (batch, channels) -> (batch, num_classes)
        return logits

    def _stage(self, x, block, channels, blocks, stride, train):
        # Stack residual blocks for one ResNet stage.
        expanded_channels = channels * block.expansion
        use_projection = stride != 1 or x.shape[-1] != expanded_channels
        x = block(channels, stride, use_projection=use_projection)(x, train=train)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, channels)
        for _ in range(1, blocks):
            x = block(channels)(x, train=train)  # (batch, out_height, out_width, channels)
        return x

# Train on a tiny synthetic image batch.
model = ResNet18(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
variables = model.init(jax.random.PRNGKey(1), train_images, train=False)
params = variables["params"]
batch_stats = variables["batch_stats"]


def train_step(params, batch_stats, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        current_variables = {"params": current_params, "batch_stats": batch_stats}
        logits = model.apply(current_variables, inputs, train=False)  # (batch, 224, 224, 3) -> (batch, num_classes)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, classes) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, batch_stats, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss
