import jax
import jax.numpy as jnp
from flax import linen as nn


class BasicBlock(nn.Module):
    out_channels: int
    stride: int = 1
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        # Preserve the residual path, projecting it when shape changes.
        identity = x

        # Apply the two-convolution residual branch.
        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding='SAME', use_bias=False, name='conv1')(x)
        y = nn.BatchNorm(use_running_average=not train, name='bn1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.out_channels, (3, 3), padding='SAME', use_bias=False, name='conv2')(y)
        y = nn.BatchNorm(use_running_average=not train, name='bn2')(y)
        if self.use_projection:
            identity = nn.Conv(self.out_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False, name='downsample_conv')(x)
            identity = nn.BatchNorm(use_running_average=not train, name='downsample_bn')(identity)

        # Add residual and apply final activation.
        y = y + identity
        y = nn.relu(y)
        return y


class ResNet18(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Convert image input into stem features.
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME', use_bias=False, name='stem_conv')(x)
        x = nn.BatchNorm(use_running_average=not train, name='stem_bn')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Run residual stages while reducing spatial size.
        x = self._stage(x, 64, blocks=2, stride=1, train=train)
        x = self._stage(x, 128, blocks=2, stride=2, train=train)
        x = self._stage(x, 256, blocks=2, stride=2, train=train)
        x = self._stage(x, 512, blocks=2, stride=2, train=train)

        # Pool final features and classify.
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes, name='fc')(x)
        return logits

    def _stage(self, x, channels, blocks, stride, train):
        # Stack residual blocks for one ResNet stage.
        use_projection = stride != 1
        x = BasicBlock(channels, stride, use_projection=use_projection)(x, train=train)
        for _ in range(1, blocks):
            x = BasicBlock(channels)(x, train=train)
        return x


# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
model = ResNet18(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)

# logits: (2, 1000)

# Train on a tiny synthetic image batch.
model = ResNet18(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)
train_targets = jnp.array([0, 1])
variables = model.init(jax.random.PRNGKey(1), train_images, train=False)
params = variables['params']
batch_stats = variables['batch_stats']


def train_step(params, batch_stats, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        current_variables = {'params': current_params, 'batch_stats': batch_stats}
        logits = model.apply(current_variables, inputs, train=False)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


for step in range(3):
    params, loss = train_step(params, batch_stats, train_images, train_targets)

final_loss = loss
