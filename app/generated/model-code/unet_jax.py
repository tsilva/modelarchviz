import jax
import jax.numpy as jnp
from flax import linen as nn


class DoubleConv(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # Preserve spatial size while changing channel width.
        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)  # (batch, height, width, channels) -> (batch, height, width, out_channels)
        x = nn.relu(x)  # (batch, height, width, out_channels)
        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)  # (batch, height, width, out_channels)
        x = nn.relu(x)  # (batch, height, width, out_channels)
        return x


class UNet(nn.Module):
    num_classes: int = 2

    @nn.compact
    def __call__(self, x):
        # Encode features while reducing spatial size at each stage.
        d1 = DoubleConv(64)(x)  # (batch, height, width, 1) -> (batch, height, width, 64)
        p1 = nn.max_pool(d1, (2, 2), (2, 2))  # (batch, height, width, 64) -> (batch, height/2, width/2, 64)
        d2 = DoubleConv(128)(p1)  # (batch, height/2, width/2, 64) -> (batch, height/2, width/2, 128)
        p2 = nn.max_pool(d2, (2, 2), (2, 2))  # (batch, height/2, width/2, 128) -> (batch, height/4, width/4, 128)
        d3 = DoubleConv(256)(p2)  # (batch, height/4, width/4, 128) -> (batch, height/4, width/4, 256)
        p3 = nn.max_pool(d3, (2, 2), (2, 2))  # (batch, height/4, width/4, 256) -> (batch, height/8, width/8, 256)
        d4 = DoubleConv(512)(p3)  # (batch, height/8, width/8, 256) -> (batch, height/8, width/8, 512)
        p4 = nn.max_pool(d4, (2, 2), (2, 2))  # (batch, height/8, width/8, 512) -> (batch, height/16, width/16, 512)

        # Process the bottleneck at the smallest spatial resolution.
        b = DoubleConv(1024)(p4)  # (batch, height/16, width/16, 512) -> (batch, height/16, width/16, 1024)

        # Decode and concatenate skip features back to full resolution.
        x = resize_like(b, d4)  # (batch, height/16, width/16, 1024) -> (batch, height/8, width/8, 1024)
        x = jnp.concatenate([x, d4], axis=-1)  # (batch, height/8, width/8, 1024) -> (batch, height/8, width/8, 1536)
        x = DoubleConv(512)(x)  # (batch, height/8, width/8, 1536) -> (batch, height/8, width/8, 512)
        x = resize_like(x, d3)  # (batch, height/8, width/8, 512) -> (batch, height/4, width/4, 512)
        x = jnp.concatenate([x, d3], axis=-1)  # (batch, height/4, width/4, 512) -> (batch, height/4, width/4, 768)
        x = DoubleConv(256)(x)  # (batch, height/4, width/4, 768) -> (batch, height/4, width/4, 256)
        x = resize_like(x, d2)  # (batch, height/4, width/4, 256) -> (batch, height/2, width/2, 256)
        x = jnp.concatenate([x, d2], axis=-1)  # (batch, height/2, width/2, 256) -> (batch, height/2, width/2, 384)
        x = DoubleConv(128)(x)  # (batch, height/2, width/2, 384) -> (batch, height/2, width/2, 128)
        x = resize_like(x, d1)  # (batch, height/2, width/2, 128) -> (batch, height, width, 128)
        x = jnp.concatenate([x, d1], axis=-1)  # (batch, height, width, 128) -> (batch, height, width, 192)
        x = DoubleConv(64)(x)  # (batch, height, width, 192) -> (batch, height, width, 64)

        # Project decoder features to segmentation logits.
        logits = nn.Conv(self.num_classes, (1, 1), name='out_conv')(x)  # (batch, height, width, 64) -> (batch, height, width, num_classes)
        return logits


def resize_like(x, skip):
    # Resize decoder features to the skip tensor spatial size.
    resize_shape = (x.shape[0], skip.shape[1], skip.shape[2], x.shape[-1])  # (batch, height, width, channels)
    resized = jax.image.resize(x, resize_shape, method='nearest')  # (batch, in_h, in_w, channels) -> (batch, skip_h, skip_w, channels)
    return resized


# Create and run a sample image batch: (2, 572, 572, 1) -> (2, 572, 572, 2).
model = UNet(num_classes=2)
test_input = jnp.ones((2, 572, 572, 1))  # -> (2, 572, 572, 1)
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)  # (2, 572, 572, 1) -> (2, 572, 572, 2)


# Train on two synthetic segmentation masks.
model = UNet(num_classes=2)
train_images = jnp.zeros((2, 64, 64, 1))  # -> (2, 64, 64, 1)
train_images = train_images.at[0, 8:32, 8:32, :].set(1.0)  # (2, 64, 64, 1)
train_images = train_images.at[1, 32:56, 32:56, :].set(1.0)  # (2, 64, 64, 1)
train_targets = jnp.zeros((2, 64, 64), dtype=jnp.int32)  # -> (2, 64, 64)
train_targets = train_targets.at[0, 8:32, 8:32].set(1)
train_targets = train_targets.at[1, 32:56, 32:56].set(1)
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # (batch, height, width, 1) -> (batch, height, width, num_classes)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch, height, width) -> (batch, height, width, num_classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, height, width, num_classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, height, width, num_classes) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
