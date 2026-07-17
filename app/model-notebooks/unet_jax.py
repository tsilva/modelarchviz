# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-doubleconv-nn-module:start
class DoubleConv(nn.Module):
# @arch class-doubleconv-nn-module:end
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # Preserve spatial size while changing channel width.
        # @arch doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x:start
        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)  # (batch, height, width, channels) -> (batch, height, width, out_channels)
        # @arch doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x:end
        # @arch doubleconv.__call__.x-nn-relu-x:start
        x = nn.relu(x)  # (batch, height, width, out_channels)
        # @arch doubleconv.__call__.x-nn-relu-x:end
        # @arch doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2:start
        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)  # (batch, height, width, out_channels)
        # @arch doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2:end
        # @arch doubleconv.__call__.x-nn-relu-x.2:start
        x = nn.relu(x)  # (batch, height, width, out_channels)
        # @arch doubleconv.__call__.x-nn-relu-x.2:end
        return x


# %% [notebook-only]
# Create and run one double convolution: (2, 32, 32, 1) -> (2, 32, 32, 8).
example_block = DoubleConv(out_channels=8)
block_input = jnp.ones((2, 32, 32, 1))  # -> (2, 32, 32, 1)
example_params = example_block.init(jax.random.PRNGKey(0), block_input)
example_block_output = example_block.apply(example_params, block_input)  # (2, 32, 32, 1) -> (2, 32, 32, 8)
print("block_output shape:", example_block_output.shape)


# %%
# @arch def-resize_like-x-skip:start
def resize_like(x, skip):
# @arch def-resize_like-x-skip:end
    # Resize decoder features to the skip tensor spatial size.
    # @arch resize_like.resize_shape-x-shape-n-skip-shape-n-skip-shape-n-x-shape-n:start
    resize_shape = (x.shape[0], skip.shape[1], skip.shape[2], x.shape[-1])  # (batch, height, width, channels)
    # @arch resize_like.resize_shape-x-shape-n-skip-shape-n-skip-shape-n-x-shape-n:end
    # @arch resize_like.resized-jax-image-resize-x-resize_shape-method-nearest:start
    resized = jax.image.resize(x, resize_shape, method='nearest')  # (batch, in_h, in_w, channels) -> (batch, skip_h, skip_w, channels)
    # @arch resize_like.resized-jax-image-resize-x-resize_shape-method-nearest:end
    # @arch resize_like.return-resized:start
    return resized
    # @arch resize_like.return-resized:end


# %%
# @arch class-unet-nn-module:start
class UNet(nn.Module):
# @arch class-unet-nn-module:end
    num_classes: int = 2

    @nn.compact
    def __call__(self, x):
        # Encode features while reducing spatial size at each stage.
        # @arch unet.__call__.dn-doubleconv-n-x:start
        d1 = DoubleConv(64)(x)  # (batch, height, width, 1) -> (batch, height, width, 64)
        # @arch unet.__call__.dn-doubleconv-n-x:end
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n:start
        p1 = nn.max_pool(d1, (2, 2), (2, 2))  # (batch, height, width, 64) -> (batch, height/2, width/2, 64)
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n:end
        # @arch unet.__call__.dn-doubleconv-n-pn:start
        d2 = DoubleConv(128)(p1)  # (batch, height/2, width/2, 64) -> (batch, height/2, width/2, 128)
        # @arch unet.__call__.dn-doubleconv-n-pn:end
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n.2:start
        p2 = nn.max_pool(d2, (2, 2), (2, 2))  # (batch, height/2, width/2, 128) -> (batch, height/4, width/4, 128)
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n.2:end
        # @arch unet.__call__.dn-doubleconv-n-pn.2:start
        d3 = DoubleConv(256)(p2)  # (batch, height/4, width/4, 128) -> (batch, height/4, width/4, 256)
        # @arch unet.__call__.dn-doubleconv-n-pn.2:end
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n.3:start
        p3 = nn.max_pool(d3, (2, 2), (2, 2))  # (batch, height/4, width/4, 256) -> (batch, height/8, width/8, 256)
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n.3:end
        # @arch unet.__call__.dn-doubleconv-n-pn.3:start
        d4 = DoubleConv(512)(p3)  # (batch, height/8, width/8, 256) -> (batch, height/8, width/8, 512)
        # @arch unet.__call__.dn-doubleconv-n-pn.3:end
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n.4:start
        p4 = nn.max_pool(d4, (2, 2), (2, 2))  # (batch, height/8, width/8, 512) -> (batch, height/16, width/16, 512)
        # @arch unet.__call__.pn-nn-max_pool-dn-n-n-n-n.4:end

        # Process the bottleneck at the smallest spatial resolution.
        # @arch unet.__call__.b-doubleconv-n-pn:start
        b = DoubleConv(1024)(p4)  # (batch, height/16, width/16, 512) -> (batch, height/16, width/16, 1024)
        # @arch unet.__call__.b-doubleconv-n-pn:end

        # Decode and concatenate skip features back to full resolution.
        # @arch unet.__call__.x-resize_like-b-dn:start
        x = resize_like(b, d4)  # (batch, height/16, width/16, 1024) -> (batch, height/8, width/8, 1024)
        # @arch unet.__call__.x-resize_like-b-dn:end
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n:start
        x = jnp.concatenate([x, d4], axis=-1)  # (batch, height/8, width/8, 1024) -> (batch, height/8, width/8, 1536)
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n:end
        # @arch unet.__call__.x-doubleconv-n-x:start
        x = DoubleConv(512)(x)  # (batch, height/8, width/8, 1536) -> (batch, height/8, width/8, 512)
        # @arch unet.__call__.x-doubleconv-n-x:end
        # @arch unet.__call__.x-resize_like-x-dn:start
        x = resize_like(x, d3)  # (batch, height/8, width/8, 512) -> (batch, height/4, width/4, 512)
        # @arch unet.__call__.x-resize_like-x-dn:end
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n.2:start
        x = jnp.concatenate([x, d3], axis=-1)  # (batch, height/4, width/4, 512) -> (batch, height/4, width/4, 768)
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n.2:end
        # @arch unet.__call__.x-doubleconv-n-x.2:start
        x = DoubleConv(256)(x)  # (batch, height/4, width/4, 768) -> (batch, height/4, width/4, 256)
        # @arch unet.__call__.x-doubleconv-n-x.2:end
        # @arch unet.__call__.x-resize_like-x-dn.2:start
        x = resize_like(x, d2)  # (batch, height/4, width/4, 256) -> (batch, height/2, width/2, 256)
        # @arch unet.__call__.x-resize_like-x-dn.2:end
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n.3:start
        x = jnp.concatenate([x, d2], axis=-1)  # (batch, height/2, width/2, 256) -> (batch, height/2, width/2, 384)
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n.3:end
        # @arch unet.__call__.x-doubleconv-n-x.3:start
        x = DoubleConv(128)(x)  # (batch, height/2, width/2, 384) -> (batch, height/2, width/2, 128)
        # @arch unet.__call__.x-doubleconv-n-x.3:end
        # @arch unet.__call__.x-resize_like-x-dn.3:start
        x = resize_like(x, d1)  # (batch, height/2, width/2, 128) -> (batch, height, width, 128)
        # @arch unet.__call__.x-resize_like-x-dn.3:end
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n.4:start
        x = jnp.concatenate([x, d1], axis=-1)  # (batch, height, width, 128) -> (batch, height, width, 192)
        # @arch unet.__call__.x-jnp-concatenate-x-dn-axis-n.4:end
        # @arch unet.__call__.x-doubleconv-n-x.4:start
        x = DoubleConv(64)(x)  # (batch, height, width, 192) -> (batch, height, width, 64)
        # @arch unet.__call__.x-doubleconv-n-x.4:end

        # Project decoder features to segmentation logits.
        # @arch unet.__call__.logits-nn-conv-self-num_classes-n-n-name-out_conv-x:start
        logits = nn.Conv(self.num_classes, (1, 1), name='out_conv')(x)  # (batch, height, width, 64) -> (batch, height, width, num_classes)
        # @arch unet.__call__.logits-nn-conv-self-num_classes-n-n-name-out_conv-x:end
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 572, 572, 1) -> (2, 572, 572, 2).
example_model = UNet(num_classes=2)
example_test_input = jnp.ones((2, 572, 572, 1))  # -> (2, 572, 572, 1)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input)
example_logits = example_model.apply(example_params, example_test_input)  # (2, 572, 572, 1) -> (2, 572, 572, 2)
print("logits shape:", example_logits.shape)

# %%
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
