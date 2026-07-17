# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-encoder-nn-module:start
class Encoder(nn.Module):
# @arch class-encoder-nn-module:end
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32

    @nn.compact
    # @arch encoder.def-__call__-self-x:start
    def __call__(self, x):
    # @arch encoder.def-__call__-self-x:end
        # Compress input vectors into low-dimensional latent codes.
        # @arch encoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-x:start
        x = nn.Dense(self.hidden_dim, name="fc1")(x)  # (batch, input_dim) -> (batch, hidden_dim)
        # @arch encoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-x:end
        # @arch encoder.__call__.x-nn-relu-x:start
        x = nn.relu(x)  # (batch, hidden_dim)
        # @arch encoder.__call__.x-nn-relu-x:end
        # @arch encoder.__call__.z-nn-dense-self-latent_dim-name-fcn-x:start
        z = nn.Dense(self.latent_dim, name="fc2")(x)  # (batch, hidden_dim) -> (batch, latent_dim)
        # @arch encoder.__call__.z-nn-dense-self-latent_dim-name-fcn-x:end
        # @arch encoder.__call__.return-z:start
        return z  # (batch, latent_dim)
        # @arch encoder.__call__.return-z:end


# %% [notebook-only]
# Create and run the encoder: (2, 64) -> (2, 8).
example_encoder = Encoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_params = example_encoder.init(jax.random.PRNGKey(0), example_inputs)
example_codes = example_encoder.apply(example_params, example_inputs)  # (2, 64) -> (2, 8)
print("latent codes shape:", example_codes.shape)

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
        # Decode latent codes back into input-shaped reconstructions.
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
        # @arch autoencoder.setup.self-encoder-encoder-self-input_dim-self-hidden_dim-self-latent_dim:start
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        # @arch autoencoder.setup.self-encoder-encoder-self-input_dim-self-hidden_dim-self-latent_dim:end
        # @arch autoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim:start
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.input_dim)
        # @arch autoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim:end

    def encode(self, x):
        # @arch autoencoder.encode.z-self-encoder-x:start
        z = self.encoder(x)  # (batch, input_dim) -> (batch, latent_dim)
        # @arch autoencoder.encode.z-self-encoder-x:end
        return z  # (batch, latent_dim)

    def decode(self, z):
        # @arch autoencoder.decode.reconstruction-self-decoder-z:start
        reconstruction = self.decoder(z)  # (batch, latent_dim) -> (batch, input_dim)
        # @arch autoencoder.decode.reconstruction-self-decoder-z:end
        return reconstruction  # (batch, input_dim)

    # @arch autoencoder.def-__call__-self-x:start
    def __call__(self, x):
    # @arch autoencoder.def-__call__-self-x:end
        # Encode through the bottleneck, then reconstruct the original input.
        # @arch autoencoder.__call__.z-self-encode-x:start
        z = self.encode(x)  # (batch, input_dim) -> (batch, latent_dim)
        # @arch autoencoder.__call__.z-self-encode-x:end
        # @arch autoencoder.__call__.reconstruction-self-decode-z:start
        reconstruction = self.decode(z)  # (batch, latent_dim) -> (batch, input_dim)
        # @arch autoencoder.__call__.reconstruction-self-decode-z:end
        # @arch autoencoder.__call__.return-reconstruction-z:start
        return reconstruction, z  # (batch, input_dim), (batch, latent_dim)
        # @arch autoencoder.__call__.return-reconstruction-z:end


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
# @arch inputs-jnp-zeros-n-n:start
inputs = jnp.zeros((2, 64))  # -> (2, 64)
# @arch inputs-jnp-zeros-n-n:end
inputs = inputs.at[0, 8:24].set(1.0)  # (2, 64)
inputs = inputs.at[1, 40:56].set(1.0)  # (2, 64)
params = model.init(jax.random.PRNGKey(3), inputs)


# @arch def-train_step-params-inputs-learning_rate-n:start
def train_step(params, inputs, learning_rate=0.05):
# @arch def-train_step-params-inputs-learning_rate-n:end
    # @arch train_step.def-loss_fn-current_params:start
    def loss_fn(current_params):
    # @arch train_step.def-loss_fn-current_params:end
        # @arch train_step.loss_fn.reconstruction-latent_codes-model-apply-current_params-inputs:start
        reconstruction, latent_codes = model.apply(current_params, inputs)  # (2, 64) -> (2, 64), (2, 8)
        # @arch train_step.loss_fn.reconstruction-latent_codes-model-apply-current_params-inputs:end
        # @arch train_step.loss_fn.loss-jnp-mean-reconstruction-inputs-n:start
        loss = jnp.mean((reconstruction - inputs) ** 2)  # (2, 64), (2, 64) -> scalar
        # @arch train_step.loss_fn.loss-jnp-mean-reconstruction-inputs-n:end
        return loss, latent_codes

    # @arch train_step.loss-latent_codes-grads-jax-value_and_grad-loss_fn-has_aux-true-params:start
    (loss, latent_codes), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # @arch train_step.loss-latent_codes-grads-jax-value_and_grad-loss_fn-has_aux-true-params:end
    # @arch train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:start
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    # @arch train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:end
    return params, loss, latent_codes


# Fit the model to reconstruct its own inputs.
# @arch for-step-in-range-n:start
for step in range(3):
# @arch for-step-in-range-n:end
    # @arch params-loss-latent_codes-train_step-params-inputs:start
    params, loss, latent_codes = train_step(params, inputs)
    # @arch params-loss-latent_codes-train_step-params-inputs:end

# Keep the final scalar loss and latent codes for inspection.
final_loss = loss  # scalar
# @arch final_latent_codes-latent_codes:start
final_latent_codes = latent_codes  # (2, 8)
# @arch final_latent_codes-latent_codes:end
