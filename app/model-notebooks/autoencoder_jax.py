# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class Encoder(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32

    @nn.compact
    def __call__(self, x):
        # Compress input vectors into low-dimensional latent codes.
        x = nn.Dense(self.hidden_dim, name="fc1")(x)  # (batch, input_dim) -> (batch, hidden_dim)
        x = nn.relu(x)  # (batch, hidden_dim)
        z = nn.Dense(self.latent_dim, name="fc2")(x)  # (batch, hidden_dim) -> (batch, latent_dim)
        return z  # (batch, latent_dim)


# %% [notebook-only]
# Create and run the encoder: (2, 64) -> (2, 8).
example_encoder = Encoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_params = example_encoder.init(jax.random.PRNGKey(0), example_inputs)
example_codes = example_encoder.apply(example_params, example_inputs)  # (2, 64) -> (2, 8)
print("latent codes shape:", example_codes.shape)

# %%
class Decoder(nn.Module):
    latent_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 784

    @nn.compact
    def __call__(self, z):
        # Decode latent codes back into input-shaped reconstructions.
        x = nn.Dense(self.hidden_dim, name="fc1")(z)  # (batch, latent_dim) -> (batch, hidden_dim)
        x = nn.relu(x)  # (batch, hidden_dim)
        x = nn.Dense(self.output_dim, name="fc2")(x)  # (batch, hidden_dim) -> (batch, output_dim)
        reconstruction = nn.sigmoid(x)  # (batch, output_dim)
        return reconstruction  # (batch, output_dim)


# %% [notebook-only]
# Create and run the decoder: (2, 8) -> (2, 64).
example_decoder = Decoder(latent_dim=8, hidden_dim=24, output_dim=64)
example_codes = jnp.ones((2, 8))  # -> (2, 8)
example_params = example_decoder.init(jax.random.PRNGKey(1), example_codes)
example_reconstruction = example_decoder.apply(example_params, example_codes)  # (2, 8) -> (2, 64)
print("reconstruction shape:", example_reconstruction.shape)

# %%
class Autoencoder(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32

    def setup(self):
        # Register mirrored encoder and decoder modules.
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.input_dim)

    def encode(self, x):
        z = self.encoder(x)  # (batch, input_dim) -> (batch, latent_dim)
        return z  # (batch, latent_dim)

    def decode(self, z):
        reconstruction = self.decoder(z)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction  # (batch, input_dim)

    def __call__(self, x):
        # Encode through the bottleneck, then reconstruct the original input.
        z = self.encode(x)  # (batch, input_dim) -> (batch, latent_dim)
        reconstruction = self.decode(z)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction, z  # (batch, input_dim), (batch, latent_dim)


# %% [notebook-only]
# Create and run a small autoencoder: (2, 64) -> (2, 64), (2, 8).
example_model = Autoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_params = example_model.init(jax.random.PRNGKey(2), example_inputs)
example_reconstruction, example_codes = example_model.apply(example_params, example_inputs)  # (2, 64) -> (2, 64), (2, 8)
print("reconstruction shape:", example_reconstruction.shape, "latent shape:", example_codes.shape)

# %%
# Train on a tiny synthetic reconstruction batch.
model = Autoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
inputs = jnp.zeros((2, 64))  # -> (2, 64)
inputs = inputs.at[0, 8:24].set(1.0)  # (2, 64)
inputs = inputs.at[1, 40:56].set(1.0)  # (2, 64)
params = model.init(jax.random.PRNGKey(3), inputs)


def train_step(params, inputs, learning_rate=0.05):
    def loss_fn(current_params):
        reconstruction, latent_codes = model.apply(current_params, inputs)  # (2, 64) -> (2, 64), (2, 8)
        loss = jnp.mean((reconstruction - inputs) ** 2)  # (2, 64), (2, 64) -> scalar
        return loss, latent_codes

    (loss, latent_codes), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss, latent_codes


# Fit the model to reconstruct its own inputs.
for step in range(3):
    params, loss, latent_codes = train_step(params, inputs)

# Keep the final scalar loss and latent codes for inspection.
final_loss = loss  # scalar
final_latent_codes = latent_codes  # (2, 8)
