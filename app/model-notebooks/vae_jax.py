# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class VariationalEncoder(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32

    @nn.compact
    def __call__(self, x):
        # Encode each input into a diagonal Gaussian q(z|x).
        hidden = nn.Dense(self.hidden_dim, name="fc1")(x)  # (batch, input_dim) -> (batch, hidden_dim)
        hidden = nn.relu(hidden)  # (batch, hidden_dim)
        mu = nn.Dense(self.latent_dim, name="fc_mu")(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        logvar = nn.Dense(self.latent_dim, name="fc_logvar")(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        return mu, logvar  # two (batch, latent_dim)


# %% [notebook-only]
# Create and run the variational encoder: (2, 64) -> two (2, 8).
example_encoder = VariationalEncoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_params = example_encoder.init(jax.random.PRNGKey(0), example_inputs)
example_mu, example_logvar = example_encoder.apply(example_params, example_inputs)  # (2, 64) -> two (2, 8)
print("mu shape:", example_mu.shape, "logvar shape:", example_logvar.shape)

# %%
class Decoder(nn.Module):
    latent_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 784

    @nn.compact
    def __call__(self, z):
        # Decode latent samples into reconstruction probabilities.
        x = nn.Dense(self.hidden_dim, name="fc1")(z)  # (batch, latent_dim) -> (batch, hidden_dim)
        x = nn.relu(x)  # (batch, hidden_dim)
        x = nn.Dense(self.output_dim, name="fc2")(x)  # (batch, hidden_dim) -> (batch, output_dim)
        reconstruction = nn.sigmoid(x)  # (batch, output_dim)
        return reconstruction  # (batch, output_dim)


# %% [notebook-only]
# Create and run the decoder: (2, 8) -> (2, 64).
example_decoder = Decoder(latent_dim=8, hidden_dim=24, output_dim=64)
example_z = jnp.ones((2, 8))  # -> (2, 8)
example_params = example_decoder.init(jax.random.PRNGKey(1), example_z)
example_reconstruction = example_decoder.apply(example_params, example_z)  # (2, 8) -> (2, 64)
print("reconstruction shape:", example_reconstruction.shape)

# %%
def binary_cross_entropy(reconstruction, x):
    # Bernoulli negative log-likelihood for bounded input vectors.
    eps = 1e-7  # scalar
    reconstruction = jnp.clip(reconstruction, eps, 1.0 - eps)  # (batch, input_dim)
    loss_values = x * jnp.log(reconstruction) + (1.0 - x) * jnp.log(1.0 - reconstruction)  # (batch, input_dim)
    loss = -jnp.sum(loss_values)  # scalar
    return loss  # scalar


# %% [notebook-only]
# Compute binary cross-entropy for bounded reconstructions: two (2, 4) -> scalar.
example_reconstruction = jnp.full((2, 4), 0.75)
example_targets = jnp.ones((2, 4))
example_loss = binary_cross_entropy(example_reconstruction, example_targets)
print("binary cross-entropy:", example_loss)


# %%
class VariationalAutoencoder(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32

    def setup(self):
        # Register the inference network and generative decoder.
        self.encoder = VariationalEncoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.input_dim)

    def reparameterize(self, mu, logvar, epsilon):
        # Sample z = mu + sigma * epsilon so gradients flow through mu and logvar.
        std = jnp.exp(0.5 * logvar)  # (batch, latent_dim)
        z = mu + std * epsilon  # (batch, latent_dim)
        return z  # (batch, latent_dim)

    def __call__(self, x, epsilon):
        # Infer q(z|x), sample a latent code, and decode it.
        mu, logvar = self.encoder(x)  # (batch, input_dim) -> two (batch, latent_dim)
        z = self.reparameterize(mu, logvar, epsilon)  # two (batch, latent_dim), (batch, latent_dim) -> (batch, latent_dim)
        reconstruction = self.decoder(z)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction, mu, logvar, z  # (batch, input_dim), three (batch, latent_dim)

    def loss(self, x, epsilon):
        # Optimize the negative ELBO: reconstruction loss plus KL to N(0, I).
        reconstruction, mu, logvar, z = self.__call__(x, epsilon)  # (batch, input_dim) -> reconstruction and latent stats
        reconstruction_loss = binary_cross_entropy(reconstruction, x)  # scalar
        kl_terms = 1.0 + logvar - mu ** 2 - jnp.exp(logvar)  # (batch, latent_dim)
        kl_loss = -0.5 * jnp.sum(kl_terms)  # scalar
        total_loss = reconstruction_loss + kl_loss  # scalar
        return total_loss, reconstruction_loss, kl_loss, z


# %% [notebook-only]
# Create and run a small VAE: (2, 64), (2, 8) -> (2, 64), two (2, 8), (2, 8).
example_model = VariationalAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_epsilon = jnp.ones((2, 8))  # -> (2, 8)
example_params = example_model.init(jax.random.PRNGKey(2), example_inputs, example_epsilon)
example_reconstruction, example_mu, example_logvar, example_z = example_model.apply(example_params, example_inputs, example_epsilon)  # (2, 64), (2, 8) -> (2, 64), three (2, 8)
print("reconstruction shape:", example_reconstruction.shape, "z shape:", example_z.shape)

# %%
# Train on a tiny synthetic batch with the VAE objective.
model = VariationalAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
inputs = jnp.zeros((2, 64))  # -> (2, 64)
inputs = inputs.at[0, 8:24].set(1.0)  # (2, 64)
inputs = inputs.at[1, 40:56].set(1.0)  # (2, 64)
epsilon = jnp.ones((2, 8))  # -> (2, 8)
params = model.init(jax.random.PRNGKey(3), inputs, epsilon)


def train_step(params, inputs, epsilon, learning_rate=0.05):
    def loss_fn(current_params):
        loss, reconstruction_loss, kl_loss, z = model.apply(current_params, inputs, epsilon, method=VariationalAutoencoder.loss)  # (2, 64), (2, 8) -> scalar losses and (2, 8)
        aux = (reconstruction_loss, kl_loss, z)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    reconstruction_loss, kl_loss, z = aux
    return params, loss, reconstruction_loss, kl_loss, z


# Fit reconstruction quality while regularizing q(z|x) toward the unit Gaussian prior.
for step in range(3):
    params, loss, reconstruction_loss, kl_loss, z = train_step(params, inputs, epsilon)

# Keep the final scalar losses and latent sample for inspection.
final_loss = loss  # scalar
final_reconstruction_loss = reconstruction_loss  # scalar
final_kl_loss = kl_loss  # scalar
final_latent_sample = z  # (2, 8)
