# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-generator-nn-module:start
class Generator(nn.Module):
# @arch class-generator-nn-module:end
    latent_dim: int = 100
    image_dim: int = 784
    hidden_dim: int = 256

    @nn.compact
    # @arch generator.def-__call__-self-z:start
    def __call__(self, z):
    # @arch generator.def-__call__-self-z:end
        # Transform latent vectors into synthetic images.
        # @arch generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-z:start
        x = nn.Dense(self.hidden_dim, name="fc1")(z)  # (batch, latent_dim) -> (batch, hidden_dim)
        # @arch generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-z:end
        # @arch generator.__call__.x-nn-leaky_relu-x-negative_slope-n:start
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        # @arch generator.__call__.x-nn-leaky_relu-x-negative_slope-n:end
        # @arch generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-x:start
        x = nn.Dense(self.hidden_dim, name="fc2")(x)  # (batch, hidden_dim)
        # @arch generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-x:end
        # @arch generator.__call__.x-nn-leaky_relu-x-negative_slope-n.2:start
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        # @arch generator.__call__.x-nn-leaky_relu-x-negative_slope-n.2:end
        # @arch generator.__call__.x-nn-dense-self-image_dim-name-fcn-x:start
        x = nn.Dense(self.image_dim, name="fc3")(x)  # (batch, hidden_dim) -> (batch, image_dim)
        # @arch generator.__call__.x-nn-dense-self-image_dim-name-fcn-x:end
        # @arch generator.__call__.fake_images-jnp-tanh-x:start
        fake_images = jnp.tanh(x)  # (batch, image_dim)
        # @arch generator.__call__.fake_images-jnp-tanh-x:end
        return fake_images  # (batch, image_dim)


# %% [notebook-only]
# Create and run the generator: (2, 16) -> (2, 32).
example_generator = Generator(latent_dim=16, image_dim=32, hidden_dim=24)
example_latents = jnp.ones((2, 16))  # -> (2, 16)
example_params = example_generator.init(jax.random.PRNGKey(0), example_latents)
example_fake_images = example_generator.apply(example_params, example_latents)  # (2, 16) -> (2, 32)
print("fake images shape:", example_fake_images.shape)

# %%
# @arch class-discriminator-nn-module:start
class Discriminator(nn.Module):
# @arch class-discriminator-nn-module:end
    image_dim: int = 784
    hidden_dim: int = 256

    @nn.compact
    # @arch discriminator.def-__call__-self-images:start
    def __call__(self, images):
    # @arch discriminator.def-__call__-self-images:end
        # Convert images into real/fake logits.
        # @arch discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-images:start
        x = nn.Dense(self.hidden_dim, name="fc1")(images)  # (batch, image_dim) -> (batch, hidden_dim)
        # @arch discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-images:end
        # @arch discriminator.__call__.x-nn-leaky_relu-x-negative_slope-n:start
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        # @arch discriminator.__call__.x-nn-leaky_relu-x-negative_slope-n:end
        # @arch discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-x:start
        x = nn.Dense(self.hidden_dim, name="fc2")(x)  # (batch, hidden_dim)
        # @arch discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-x:end
        # @arch discriminator.__call__.x-nn-leaky_relu-x-negative_slope-n.2:start
        x = nn.leaky_relu(x, negative_slope=0.2)  # (batch, hidden_dim)
        # @arch discriminator.__call__.x-nn-leaky_relu-x-negative_slope-n.2:end
        # @arch discriminator.__call__.logits-nn-dense-n-name-fcn-x:start
        logits = nn.Dense(1, name="fc3")(x)  # (batch, hidden_dim) -> (batch, 1)
        # @arch discriminator.__call__.logits-nn-dense-n-name-fcn-x:end
        # @arch discriminator.__call__.logits-jnp-squeeze-logits-axis-n:start
        logits = jnp.squeeze(logits, axis=-1)  # (batch, 1) -> (batch)
        # @arch discriminator.__call__.logits-jnp-squeeze-logits-axis-n:end
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
        # @arch gan.setup.self-generator-generator-self-latent_dim-self-image_dim-self-hidden_dim:start
        self.generator = Generator(self.latent_dim, self.image_dim, self.hidden_dim)
        # @arch gan.setup.self-generator-generator-self-latent_dim-self-image_dim-self-hidden_dim:end
        # @arch gan.setup.self-discriminator-discriminator-self-image_dim-self-hidden_dim:start
        self.discriminator = Discriminator(self.image_dim, self.hidden_dim)
        # @arch gan.setup.self-discriminator-discriminator-self-image_dim-self-hidden_dim:end

    # @arch gan.def-generate-self-z:start
    def generate(self, z):
    # @arch gan.def-generate-self-z:end
        # @arch gan.generate.fake_images-self-generator-z:start
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch gan.generate.fake_images-self-generator-z:end
        return fake_images  # (batch, image_dim)

    # @arch gan.def-discriminate-self-images:start
    def discriminate(self, images):
    # @arch gan.def-discriminate-self-images:end
        # @arch gan.discriminate.logits-self-discriminator-images:start
        logits = self.discriminator(images)  # (batch, image_dim) -> (batch)
        # @arch gan.discriminate.logits-self-discriminator-images:end
        return logits  # (batch)

    def __call__(self, z, real_images):
        # Expose both adversarial branches for shape inspection.
        # @arch gan.__call__.fake_images-self-generate-z:start
        fake_images = self.generate(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch gan.__call__.fake_images-self-generate-z:end
        # @arch gan.__call__.real_logits-self-discriminate-real_images:start
        real_logits = self.discriminate(real_images)  # (batch, image_dim) -> (batch)
        # @arch gan.__call__.real_logits-self-discriminate-real_images:end
        # @arch gan.__call__.fake_logits-self-discriminate-fake_images:start
        fake_logits = self.discriminate(fake_images)  # (batch, image_dim) -> (batch)
        # @arch gan.__call__.fake_logits-self-discriminate-fake_images:end
        return fake_images, real_logits, fake_logits

    # @arch gan.def-discriminator_loss-self-real_images-z:start
    def discriminator_loss(self, real_images, z):
    # @arch gan.def-discriminator_loss-self-real_images-z:end
        # Train D to classify real images as 1 and generated images as 0.
        # @arch gan.discriminator_loss.fake_images-self-generate-z:start
        fake_images = self.generate(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch gan.discriminator_loss.fake_images-self-generate-z:end
        # @arch gan.discriminator_loss.real_logits-self-discriminate-real_images:start
        real_logits = self.discriminate(real_images)  # (batch, image_dim) -> (batch)
        # @arch gan.discriminator_loss.real_logits-self-discriminate-real_images:end
        # @arch gan.discriminator_loss.fake_logits-self-discriminate-jax-lax-stop_gradient-fake_images:start
        fake_logits = self.discriminate(jax.lax.stop_gradient(fake_images))  # (batch, image_dim) -> (batch)
        # @arch gan.discriminator_loss.fake_logits-self-discriminate-jax-lax-stop_gradient-fake_images:end
        # @arch gan.discriminator_loss.real_targets-jnp-ones_like-real_logits:start
        real_targets = jnp.ones_like(real_logits)  # -> (batch)
        # @arch gan.discriminator_loss.real_targets-jnp-ones_like-real_logits:end
        # @arch gan.discriminator_loss.fake_targets-jnp-zeros_like-fake_logits:start
        fake_targets = jnp.zeros_like(fake_logits)  # -> (batch)
        # @arch gan.discriminator_loss.fake_targets-jnp-zeros_like-fake_logits:end
        # @arch gan.discriminator_loss.real_loss-binary_cross_entropy_with_logits-real_logits-real_targets:start
        real_loss = binary_cross_entropy_with_logits(real_logits, real_targets)  # scalar
        # @arch gan.discriminator_loss.real_loss-binary_cross_entropy_with_logits-real_logits-real_targets:end
        # @arch gan.discriminator_loss.fake_loss-binary_cross_entropy_with_logits-fake_logits-fake_targets:start
        fake_loss = binary_cross_entropy_with_logits(fake_logits, fake_targets)  # scalar
        # @arch gan.discriminator_loss.fake_loss-binary_cross_entropy_with_logits-fake_logits-fake_targets:end
        # @arch gan.discriminator_loss.loss-real_loss-fake_loss:start
        loss = real_loss + fake_loss  # scalar
        # @arch gan.discriminator_loss.loss-real_loss-fake_loss:end
        return loss  # scalar

    # @arch gan.def-generator_loss-self-z:start
    def generator_loss(self, z):
    # @arch gan.def-generator_loss-self-z:end
        # Train G to make D classify generated images as real.
        # @arch gan.generator_loss.fake_images-self-generate-z:start
        fake_images = self.generate(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch gan.generator_loss.fake_images-self-generate-z:end
        # @arch gan.generator_loss.fake_logits-self-discriminate-fake_images:start
        fake_logits = self.discriminate(fake_images)  # (batch, image_dim) -> (batch)
        # @arch gan.generator_loss.fake_logits-self-discriminate-fake_images:end
        # @arch gan.generator_loss.real_targets-jnp-ones_like-fake_logits:start
        real_targets = jnp.ones_like(fake_logits)  # -> (batch)
        # @arch gan.generator_loss.real_targets-jnp-ones_like-fake_logits:end
        # @arch gan.generator_loss.loss-binary_cross_entropy_with_logits-fake_logits-real_targets:start
        loss = binary_cross_entropy_with_logits(fake_logits, real_targets)  # scalar
        # @arch gan.generator_loss.loss-binary_cross_entropy_with_logits-fake_logits-real_targets:end
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
# @arch real_images-jnp-zeros-n-n:start
real_images = jnp.zeros((2, 32))  # -> (2, 32)
# @arch real_images-jnp-zeros-n-n:end
real_images = real_images.at[0, 4:12].set(1.0)  # (2, 32)
real_images = real_images.at[1, 20:28].set(1.0)  # (2, 32)
# @arch z-jnp-ones-n-n:start
z = jnp.ones((2, 16))  # -> (2, 16)
# @arch z-jnp-ones-n-n:end
params = model.init(jax.random.PRNGKey(3), z, real_images)


# @arch def-discriminator_train_step-params-real_images-z-learning_rate-n:start
def discriminator_train_step(params, real_images, z, learning_rate=0.01):
# @arch def-discriminator_train_step-params-real_images-z-learning_rate-n:end
    def loss_fn(current_params):
        # @arch discriminator_train_step.loss_fn.loss-model-apply-current_params-real_images-z-method-gan-discriminator_l:start
        loss = model.apply(current_params, real_images, z, method=GAN.discriminator_loss)  # (2, 32), (2, 16) -> scalar
        # @arch discriminator_train_step.loss_fn.loss-model-apply-current_params-real_images-z-method-gan-discriminator_l:end
        return loss  # scalar

    # @arch discriminator_train_step.loss-grads-jax-value_and_grad-loss_fn-params:start
    loss, grads = jax.value_and_grad(loss_fn)(params)
    # @arch discriminator_train_step.loss-grads-jax-value_and_grad-loss_fn-params:end
    # @arch discriminator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:start
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    # @arch discriminator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:end
    return params, loss


# @arch def-generator_train_step-params-z-learning_rate-n:start
def generator_train_step(params, z, learning_rate=0.01):
# @arch def-generator_train_step-params-z-learning_rate-n:end
    def loss_fn(current_params):
        # @arch generator_train_step.loss_fn.loss-model-apply-current_params-z-method-gan-generator_loss:start
        loss = model.apply(current_params, z, method=GAN.generator_loss)  # (2, 16) -> scalar
        # @arch generator_train_step.loss_fn.loss-model-apply-current_params-z-method-gan-generator_loss:end
        return loss  # scalar

    # @arch generator_train_step.loss-grads-jax-value_and_grad-loss_fn-params:start
    loss, grads = jax.value_and_grad(loss_fn)(params)
    # @arch generator_train_step.loss-grads-jax-value_and_grad-loss_fn-params:end
    # @arch generator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:start
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    # @arch generator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:end
    return params, loss


# Alternate discriminator and generator updates.
# @arch for-step-in-range-n:start
for step in range(3):
# @arch for-step-in-range-n:end
    # @arch params-d_loss-discriminator_train_step-params-real_images-z:start
    params, d_loss = discriminator_train_step(params, real_images, z)
    # @arch params-d_loss-discriminator_train_step-params-real_images-z:end
    # @arch params-g_loss-generator_train_step-params-z:start
    params, g_loss = generator_train_step(params, z)
    # @arch params-g_loss-generator_train_step-params-z:end

# Keep the final scalar losses for inspection.
final_discriminator_loss = d_loss  # scalar
final_generator_loss = g_loss  # scalar
