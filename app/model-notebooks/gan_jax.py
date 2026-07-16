# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class Generator(nn.Module):
    latent_dim: int = 100
    image_dim: int = 784
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z):
        # Transform latent vectors into synthetic images.
        x = nn.Dense(self.hidden_dim, name="fc1")(z)  # (batch, latent_dim) -> (batch, hidden_dim)
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        x = nn.Dense(self.hidden_dim, name="fc2")(x)  # (batch, hidden_dim)
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        x = nn.Dense(self.image_dim, name="fc3")(x)  # (batch, hidden_dim) -> (batch, image_dim)
        fake_images = jnp.tanh(x)  # (batch, image_dim)
        return fake_images  # (batch, image_dim)


# %% [notebook-only]
# Create and run the generator: (2, 16) -> (2, 32).
example_generator = Generator(latent_dim=16, image_dim=32, hidden_dim=24)
example_latents = jnp.ones((2, 16))  # -> (2, 16)
example_params = example_generator.init(jax.random.PRNGKey(0), example_latents)
example_fake_images = example_generator.apply(example_params, example_latents)  # (2, 16) -> (2, 32)
print("fake images shape:", example_fake_images.shape)

# %%
class Discriminator(nn.Module):
    image_dim: int = 784
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, images):
        # Convert images into real/fake logits.
        x = nn.Dense(self.hidden_dim, name="fc1")(images)  # (batch, image_dim) -> (batch, hidden_dim)
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        x = nn.Dense(self.hidden_dim, name="fc2")(x)  # (batch, hidden_dim)
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        logits = nn.Dense(1, name="fc3")(x)  # (batch, hidden_dim) -> (batch, 1)
        logits = jnp.squeeze(logits, axis=-1)  # (batch, 1) -> (batch)
        return logits  # (batch)


# %% [notebook-only]
# Create and run the discriminator: (2, 32) -> (2).
example_discriminator = Discriminator(image_dim=32, hidden_dim=24)
example_images = jnp.ones((2, 32))  # -> (2, 32)
example_params = example_discriminator.init(jax.random.PRNGKey(1), example_images)
example_logits = example_discriminator.apply(example_params, example_images)  # (2, 32) -> (2)
print("discriminator logits shape:", example_logits.shape)

# %%
def binary_cross_entropy_with_logits(logits, targets):
    # Stable binary cross-entropy from logits.
    positive_part = jnp.maximum(logits, 0.0)  # (batch)
    log_exp = jnp.log1p(jnp.exp(-jnp.abs(logits)))  # (batch)
    loss_values = positive_part - logits * targets + log_exp  # (batch)
    loss = jnp.mean(loss_values)  # (batch) -> scalar
    return loss  # scalar


# %%
class GAN(nn.Module):
    latent_dim: int = 100
    image_dim: int = 784
    hidden_dim: int = 256

    def setup(self):
        # Register the competing generator and discriminator.
        self.generator = Generator(self.latent_dim, self.image_dim, self.hidden_dim)
        self.discriminator = Discriminator(self.image_dim, self.hidden_dim)

    def generate(self, z):
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        return fake_images  # (batch, image_dim)

    def discriminate(self, images):
        logits = self.discriminator(images)  # (batch, image_dim) -> (batch)
        return logits  # (batch)

    def __call__(self, z, real_images):
        # Expose both adversarial branches for shape inspection.
        fake_images = self.generate(z)  # (batch, latent_dim) -> (batch, image_dim)
        real_logits = self.discriminate(real_images)  # (batch, image_dim) -> (batch)
        fake_logits = self.discriminate(fake_images)  # (batch, image_dim) -> (batch)
        return fake_images, real_logits, fake_logits

    def discriminator_loss(self, real_images, z):
        # Train D to classify real images as 1 and generated images as 0.
        fake_images = self.generate(z)  # (batch, latent_dim) -> (batch, image_dim)
        real_logits = self.discriminate(real_images)  # (batch, image_dim) -> (batch)
        fake_logits = self.discriminate(jax.lax.stop_gradient(fake_images))  # (batch, image_dim) -> (batch)
        real_targets = jnp.ones_like(real_logits)  # -> (batch)
        fake_targets = jnp.zeros_like(fake_logits)  # -> (batch)
        real_loss = binary_cross_entropy_with_logits(real_logits, real_targets)  # scalar
        fake_loss = binary_cross_entropy_with_logits(fake_logits, fake_targets)  # scalar
        loss = real_loss + fake_loss  # scalar
        return loss  # scalar

    def generator_loss(self, z):
        # Train G to make D classify generated images as real.
        fake_images = self.generate(z)  # (batch, latent_dim) -> (batch, image_dim)
        fake_logits = self.discriminate(fake_images)  # (batch, image_dim) -> (batch)
        real_targets = jnp.ones_like(fake_logits)  # -> (batch)
        loss = binary_cross_entropy_with_logits(fake_logits, real_targets)  # scalar
        return loss  # scalar


# %% [notebook-only]
# Create and run a small GAN forward path.
example_model = GAN(latent_dim=16, image_dim=32, hidden_dim=24)
example_z = jnp.ones((2, 16))  # -> (2, 16)
example_real_images = jnp.ones((2, 32))  # -> (2, 32)
example_params = example_model.init(jax.random.PRNGKey(2), example_z, example_real_images)
example_fake_images, example_real_logits, example_fake_logits = example_model.apply(example_params, example_z, example_real_images)  # (2, 16), (2, 32) -> (2, 32), (2), (2)
print("fake images shape:", example_fake_images.shape, "fake logits shape:", example_fake_logits.shape)

# %%
# Train on a tiny synthetic real-image batch.
model = GAN(latent_dim=16, image_dim=32, hidden_dim=24)
real_images = jnp.zeros((2, 32))  # -> (2, 32)
real_images = real_images.at[0, 4:12].set(1.0)  # (2, 32)
real_images = real_images.at[1, 20:28].set(1.0)  # (2, 32)
z = jnp.ones((2, 16))  # -> (2, 16)
params = model.init(jax.random.PRNGKey(3), z, real_images)


def discriminator_train_step(params, real_images, z, learning_rate=0.01):
    def loss_fn(current_params):
        loss = model.apply(current_params, real_images, z, method=GAN.discriminator_loss)  # (2, 32), (2, 16) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


def generator_train_step(params, z, learning_rate=0.01):
    def loss_fn(current_params):
        loss = model.apply(current_params, z, method=GAN.generator_loss)  # (2, 16) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Alternate discriminator and generator updates.
for step in range(3):
    params, d_loss = discriminator_train_step(params, real_images, z)
    params, g_loss = generator_train_step(params, z)

# Keep the final scalar losses for inspection.
final_discriminator_loss = d_loss  # scalar
final_generator_loss = g_loss  # scalar
