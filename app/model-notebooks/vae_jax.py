# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-variationalencoder-nn-module:start
class VariationalEncoder(nn.Module):
# @arch class-variationalencoder-nn-module:end
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32

    @nn.compact
    # @arch variationalencoder.def-__call__-self-x:start
    def __call__(self, x):
    # @arch variationalencoder.def-__call__-self-x:end
        # Encode each input into a diagonal Gaussian q(z|x).
        # @arch variationalencoder.__call__.hidden-nn-dense-self-hidden_dim-name-fcn-x:start
        hidden = nn.Dense(self.hidden_dim, name="fc1")(x)  # (batch, input_dim) -> (batch, hidden_dim)
        # @arch variationalencoder.__call__.hidden-nn-dense-self-hidden_dim-name-fcn-x:end
        # @arch variationalencoder.__call__.hidden-nn-relu-hidden:start
        hidden = nn.relu(hidden)  # (batch, hidden_dim)
        # @arch variationalencoder.__call__.hidden-nn-relu-hidden:end
        # @arch variationalencoder.__call__.mu-nn-dense-self-latent_dim-name-fc_mu-hidden:start
        mu = nn.Dense(self.latent_dim, name="fc_mu")(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        # @arch variationalencoder.__call__.mu-nn-dense-self-latent_dim-name-fc_mu-hidden:end
        # @arch variationalencoder.__call__.logvar-nn-dense-self-latent_dim-name-fc_logvar-hidden:start
        logvar = nn.Dense(self.latent_dim, name="fc_logvar")(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        # @arch variationalencoder.__call__.logvar-nn-dense-self-latent_dim-name-fc_logvar-hidden:end
        # @arch variationalencoder.__call__.return-mu-logvar:start
        return mu, logvar  # two (batch, latent_dim)
        # @arch variationalencoder.__call__.return-mu-logvar:end


# %% [notebook-only]
# Create and run the variational encoder: (2, 64) -> two (2, 8).
example_encoder = VariationalEncoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_params = example_encoder.init(jax.random.PRNGKey(0), example_inputs)
example_mu, example_logvar = example_encoder.apply(example_params, example_inputs)  # (2, 64) -> two (2, 8)
print("mu shape:", example_mu.shape, "logvar shape:", example_logvar.shape)

# %%
# @arch class-decoder-nn-module:start
class Decoder(nn.Module):
# @arch class-decoder-nn-module:end
    latent_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 784

    @nn.compact
    # @arch decoder.def-__call__-self-z:start
    def __call__(self, z):
    # @arch decoder.def-__call__-self-z:end
        # Decode latent samples into reconstruction probabilities.
        # @arch decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z:start
        x = nn.Dense(self.hidden_dim, name="fc1")(z)  # (batch, latent_dim) -> (batch, hidden_dim)
        # @arch decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z:end
        # @arch decoder.__call__.x-nn-relu-x:start
        x = nn.relu(x)  # (batch, hidden_dim)
        # @arch decoder.__call__.x-nn-relu-x:end
        # @arch decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x:start
        x = nn.Dense(self.output_dim, name="fc2")(x)  # (batch, hidden_dim) -> (batch, output_dim)
        # @arch decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x:end
        # @arch decoder.__call__.reconstruction-nn-sigmoid-x:start
        reconstruction = nn.sigmoid(x)  # (batch, output_dim)
        # @arch decoder.__call__.reconstruction-nn-sigmoid-x:end
        # @arch decoder.__call__.return-reconstruction:start
        return reconstruction  # (batch, output_dim)
        # @arch decoder.__call__.return-reconstruction:end


# %% [notebook-only]
# Create and run the decoder: (2, 8) -> (2, 64).
example_decoder = Decoder(latent_dim=8, hidden_dim=24, output_dim=64)
example_z = jnp.ones((2, 8))  # -> (2, 8)
example_params = example_decoder.init(jax.random.PRNGKey(1), example_z)
example_reconstruction = example_decoder.apply(example_params, example_z)  # (2, 8) -> (2, 64)
print("reconstruction shape:", example_reconstruction.shape)

# %%
# @arch def-binary_cross_entropy-reconstruction-x:start
def binary_cross_entropy(reconstruction, x):
# @arch def-binary_cross_entropy-reconstruction-x:end
    # Bernoulli negative log-likelihood for bounded input vectors.
    eps = 1e-7  # scalar
    # @arch binary_cross_entropy.reconstruction-jnp-clip-reconstruction-eps-n-eps:start
    reconstruction = jnp.clip(reconstruction, eps, 1.0 - eps)  # (batch, input_dim)
    # @arch binary_cross_entropy.reconstruction-jnp-clip-reconstruction-eps-n-eps:end
    # @arch binary_cross_entropy.loss_values-x-jnp-log-reconstruction-n-x-jnp-log-n-reconstruction:start
    loss_values = x * jnp.log(reconstruction) + (1.0 - x) * jnp.log(1.0 - reconstruction)  # (batch, input_dim)
    # @arch binary_cross_entropy.loss_values-x-jnp-log-reconstruction-n-x-jnp-log-n-reconstruction:end
    # @arch binary_cross_entropy.loss-jnp-sum-loss_values:start
    loss = -jnp.sum(loss_values)  # scalar
    # @arch binary_cross_entropy.loss-jnp-sum-loss_values:end
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
        # @arch variationalautoencoder.setup.self-encoder-variationalencoder-self-input_dim-self-hidden_dim-self-late:start
        self.encoder = VariationalEncoder(self.input_dim, self.hidden_dim, self.latent_dim)
        # @arch variationalautoencoder.setup.self-encoder-variationalencoder-self-input_dim-self-hidden_dim-self-late:end
        # @arch variationalautoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim:start
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.input_dim)
        # @arch variationalautoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim:end

    # @arch variationalautoencoder.def-reparameterize-self-mu-logvar-epsilon:start
    def reparameterize(self, mu, logvar, epsilon):
    # @arch variationalautoencoder.def-reparameterize-self-mu-logvar-epsilon:end
        # Sample z = mu + sigma * epsilon so gradients flow through mu and logvar.
        # @arch variationalautoencoder.reparameterize.std-jnp-exp-n-logvar:start
        std = jnp.exp(0.5 * logvar)  # (batch, latent_dim)
        # @arch variationalautoencoder.reparameterize.std-jnp-exp-n-logvar:end
        # @arch variationalautoencoder.reparameterize.z-mu-std-epsilon:start
        z = mu + std * epsilon  # (batch, latent_dim)
        # @arch variationalautoencoder.reparameterize.z-mu-std-epsilon:end
        # @arch variationalautoencoder.reparameterize.return-z:start
        return z  # (batch, latent_dim)
        # @arch variationalautoencoder.reparameterize.return-z:end

    # @arch variationalautoencoder.def-__call__-self-x-epsilon:start
    def __call__(self, x, epsilon):
    # @arch variationalautoencoder.def-__call__-self-x-epsilon:end
        # Infer q(z|x), sample a latent code, and decode it.
        # @arch variationalautoencoder.__call__.mu-logvar-self-encoder-x:start
        mu, logvar = self.encoder(x)  # (batch, input_dim) -> two (batch, latent_dim)
        # @arch variationalautoencoder.__call__.mu-logvar-self-encoder-x:end
        # @arch variationalautoencoder.__call__.z-self-reparameterize-mu-logvar-epsilon:start
        z = self.reparameterize(mu, logvar, epsilon)  # two (batch, latent_dim), (batch, latent_dim) -> (batch, latent_dim)
        # @arch variationalautoencoder.__call__.z-self-reparameterize-mu-logvar-epsilon:end
        # @arch variationalautoencoder.__call__.reconstruction-self-decoder-z:start
        reconstruction = self.decoder(z)  # (batch, latent_dim) -> (batch, input_dim)
        # @arch variationalautoencoder.__call__.reconstruction-self-decoder-z:end
        # @arch variationalautoencoder.__call__.return-reconstruction-mu-logvar-z:start
        return reconstruction, mu, logvar, z  # (batch, input_dim), three (batch, latent_dim)
        # @arch variationalautoencoder.__call__.return-reconstruction-mu-logvar-z:end

    # @arch variationalautoencoder.def-loss-self-x-epsilon:start
    def loss(self, x, epsilon):
    # @arch variationalautoencoder.def-loss-self-x-epsilon:end
        # Optimize the negative ELBO: reconstruction loss plus KL to N(0, I).
        # @arch variationalautoencoder.loss.reconstruction-mu-logvar-z-self-__call__-x-epsilon:start
        reconstruction, mu, logvar, z = self.__call__(x, epsilon)  # (batch, input_dim) -> reconstruction and latent stats
        # @arch variationalautoencoder.loss.reconstruction-mu-logvar-z-self-__call__-x-epsilon:end
        # @arch variationalautoencoder.loss.reconstruction_loss-binary_cross_entropy-reconstruction-x:start
        reconstruction_loss = binary_cross_entropy(reconstruction, x)  # scalar
        # @arch variationalautoencoder.loss.reconstruction_loss-binary_cross_entropy-reconstruction-x:end
        # @arch variationalautoencoder.loss.kl_terms-n-logvar-mu-n-jnp-exp-logvar:start
        kl_terms = 1.0 + logvar - mu ** 2 - jnp.exp(logvar)  # (batch, latent_dim)
        # @arch variationalautoencoder.loss.kl_terms-n-logvar-mu-n-jnp-exp-logvar:end
        # @arch variationalautoencoder.loss.kl_loss-n-jnp-sum-kl_terms:start
        kl_loss = -0.5 * jnp.sum(kl_terms)  # scalar
        # @arch variationalautoencoder.loss.kl_loss-n-jnp-sum-kl_terms:end
        # @arch variationalautoencoder.loss.total_loss-reconstruction_loss-kl_loss:start
        total_loss = reconstruction_loss + kl_loss  # scalar
        # @arch variationalautoencoder.loss.total_loss-reconstruction_loss-kl_loss:end
        # @arch variationalautoencoder.loss.return-total_loss-reconstruction_loss-kl_loss-z:start
        return total_loss, reconstruction_loss, kl_loss, z
        # @arch variationalautoencoder.loss.return-total_loss-reconstruction_loss-kl_loss-z:end


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
# @arch inputs-jnp-zeros-n-n:start
inputs = jnp.zeros((2, 64))  # -> (2, 64)
# @arch inputs-jnp-zeros-n-n:end
inputs = inputs.at[0, 8:24].set(1.0)  # (2, 64)
inputs = inputs.at[1, 40:56].set(1.0)  # (2, 64)
# @arch epsilon-jnp-ones-n-n:start
epsilon = jnp.ones((2, 8))  # -> (2, 8)
# @arch epsilon-jnp-ones-n-n:end
params = model.init(jax.random.PRNGKey(3), inputs, epsilon)


def train_step(params, inputs, epsilon, learning_rate=0.05):
    def loss_fn(current_params):
        # @arch train_step.loss_fn.loss-reconstruction_loss-kl_loss-z-model-apply-current_params-inputs-eps:start
        loss, reconstruction_loss, kl_loss, z = model.apply(current_params, inputs, epsilon, method=VariationalAutoencoder.loss)  # (2, 64), (2, 8) -> scalar losses and (2, 8)
        # @arch train_step.loss_fn.loss-reconstruction_loss-kl_loss-z-model-apply-current_params-inputs-eps:end
        aux = (reconstruction_loss, kl_loss, z)
        return loss, aux

    # @arch train_step.loss-aux-grads-jax-value_and_grad-loss_fn-has_aux-true-params:start
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # @arch train_step.loss-aux-grads-jax-value_and_grad-loss_fn-has_aux-true-params:end
    # @arch train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:start
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    # @arch train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:end
    reconstruction_loss, kl_loss, z = aux
    return params, loss, reconstruction_loss, kl_loss, z


# Fit reconstruction quality while regularizing q(z|x) toward the unit Gaussian prior.
for step in range(3):
    params, loss, reconstruction_loss, kl_loss, z = train_step(params, inputs, epsilon)

# Keep the final scalar losses and latent sample for inspection.
final_loss = loss  # scalar
final_reconstruction_loss = reconstruction_loss  # scalar
final_kl_loss = kl_loss  # scalar
# @arch final_latent_sample-z:start
final_latent_sample = z  # (2, 8)
# @arch final_latent_sample-z:end
