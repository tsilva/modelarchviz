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

    # @arch encoder.nn-compact:start
    @nn.compact
    # @arch encoder.nn-compact:end
    # @arch encoder.def-__call__-self-x:start
    def __call__(self, x):
    # @arch encoder.def-__call__-self-x:end
        # Encode each input into a continuous latent vector z_e(x).
        # @arch encoder.__call__.z_e-nn-dense-self-hidden_dim-name-fcn-x:start
        z_e = nn.Dense(self.hidden_dim, name="fc1")(x)  # (batch, input_dim) -> (batch, hidden_dim)
        # @arch encoder.__call__.z_e-nn-dense-self-hidden_dim-name-fcn-x:end
        # @arch encoder.__call__.z_e-nn-relu-z_e:start
        z_e = nn.relu(z_e)  # (batch, hidden_dim)
        # @arch encoder.__call__.z_e-nn-relu-z_e:end
        # @arch encoder.__call__.z_e-nn-dense-self-latent_dim-name-fcn-z_e:start
        z_e = nn.Dense(self.latent_dim, name="fc2")(z_e)  # (batch, hidden_dim) -> (batch, latent_dim)
        # @arch encoder.__call__.z_e-nn-dense-self-latent_dim-name-fcn-z_e:end
        # @arch encoder.__call__.return-z_e:start
        return z_e  # (batch, latent_dim)
        # @arch encoder.__call__.return-z_e:end


# %% [notebook-only]
# Create and run the encoder: (2, 64) -> (2, 8).
example_encoder = Encoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_params = example_encoder.init(jax.random.PRNGKey(0), example_inputs)
example_z_e = example_encoder.apply(example_params, example_inputs)  # (2, 64) -> (2, 8)
print("encoder output shape:", example_z_e.shape)

# %%
# @arch class-vectorquantizer-nn-module:start
class VectorQuantizer(nn.Module):
# @arch class-vectorquantizer-nn-module:end
    num_codes: int = 16
    latent_dim: int = 32
    beta: float = 0.25

    # @arch vectorquantizer.nn-compact:start
    @nn.compact
    # @arch vectorquantizer.nn-compact:end
    def __call__(self, z_e):
        # Compare each encoder vector with every codebook entry by squared distance.
        # @arch vectorquantizer.__call__.codebook-self-param-codebook-nn-initializers-uniform-scale-n-self-num_co:start
        codebook = self.param("codebook", nn.initializers.uniform(scale=0.1), (self.num_codes, self.latent_dim))
        # @arch vectorquantizer.__call__.codebook-self-param-codebook-nn-initializers-uniform-scale-n-self-num_co:end
        # @arch vectorquantizer.__call__.z_squared-jnp-sum-z_e-n-axis-n-keepdims-true:start
        z_squared = jnp.sum(z_e ** 2, axis=1, keepdims=True)  # (batch, latent_dim) -> (batch, 1)
        # @arch vectorquantizer.__call__.z_squared-jnp-sum-z_e-n-axis-n-keepdims-true:end
        # @arch vectorquantizer.__call__.codebook_squared-jnp-sum-codebook-n-axis-n:start
        codebook_squared = jnp.sum(codebook ** 2, axis=1)  # (num_codes)
        # @arch vectorquantizer.__call__.codebook_squared-jnp-sum-codebook-n-axis-n:end
        # @arch vectorquantizer.__call__.dot_products-z_e-codebook-t:start
        dot_products = z_e @ codebook.T  # (batch, latent_dim) -> (batch, num_codes)
        # @arch vectorquantizer.__call__.dot_products-z_e-codebook-t:end
        # @arch vectorquantizer.__call__.distances-z_squared-codebook_squared-n-dot_products:start
        distances = z_squared + codebook_squared - 2.0 * dot_products  # (batch, num_codes)
        # @arch vectorquantizer.__call__.distances-z_squared-codebook_squared-n-dot_products:end

        # Replace each continuous vector with its nearest discrete code.
        # @arch vectorquantizer.__call__.encoding_indices-jnp-argmin-distances-axis-n:start
        encoding_indices = jnp.argmin(distances, axis=1)  # (batch)
        # @arch vectorquantizer.__call__.encoding_indices-jnp-argmin-distances-axis-n:end
        # @arch vectorquantizer.__call__.quantized-codebook-encoding_indices:start
        quantized = codebook[encoding_indices]  # (batch) -> (batch, latent_dim)
        # @arch vectorquantizer.__call__.quantized-codebook-encoding_indices:end

        # Train the codebook and commit the encoder output to the selected entries.
        # @arch vectorquantizer.__call__.codebook_error-quantized-jax-lax-stop_gradient-z_e:start
        codebook_error = quantized - jax.lax.stop_gradient(z_e)  # (batch, latent_dim)
        # @arch vectorquantizer.__call__.codebook_error-quantized-jax-lax-stop_gradient-z_e:end
        # @arch vectorquantizer.__call__.codebook_loss-jnp-mean-codebook_error-n:start
        codebook_loss = jnp.mean(codebook_error ** 2)  # scalar
        # @arch vectorquantizer.__call__.codebook_loss-jnp-mean-codebook_error-n:end
        # @arch vectorquantizer.__call__.commitment_error-z_e-jax-lax-stop_gradient-quantized:start
        commitment_error = z_e - jax.lax.stop_gradient(quantized)  # (batch, latent_dim)
        # @arch vectorquantizer.__call__.commitment_error-z_e-jax-lax-stop_gradient-quantized:end
        # @arch vectorquantizer.__call__.commitment_loss-jnp-mean-commitment_error-n:start
        commitment_loss = jnp.mean(commitment_error ** 2)  # scalar
        # @arch vectorquantizer.__call__.commitment_loss-jnp-mean-commitment_error-n:end
        # @arch vectorquantizer.__call__.vq_loss-codebook_loss-self-beta-commitment_loss:start
        vq_loss = codebook_loss + self.beta * commitment_loss  # scalar
        # @arch vectorquantizer.__call__.vq_loss-codebook_loss-self-beta-commitment_loss:end

        # Use the straight-through estimator: decoder sees quantized values, encoder gets gradients.
        # @arch vectorquantizer.__call__.quantized_st-z_e-jax-lax-stop_gradient-quantized-z_e:start
        quantized_st = z_e + jax.lax.stop_gradient(quantized - z_e)  # (batch, latent_dim)
        # @arch vectorquantizer.__call__.quantized_st-z_e-jax-lax-stop_gradient-quantized-z_e:end
        # @arch vectorquantizer.__call__.return-quantized_st-vq_loss-encoding_indices-codebook_loss-commitment_lo:start
        return quantized_st, vq_loss, encoding_indices, codebook_loss, commitment_loss
        # @arch vectorquantizer.__call__.return-quantized_st-vq_loss-encoding_indices-codebook_loss-commitment_lo:end


# %% [notebook-only]
# Quantize encoder outputs with a discrete codebook: (2, 8) -> (2, 8), (2).
example_quantizer = VectorQuantizer(num_codes=6, latent_dim=8)
example_z_e = jnp.ones((2, 8))  # -> (2, 8)
example_params = example_quantizer.init(jax.random.PRNGKey(1), example_z_e)
example_z_q, example_vq_loss, example_indices, example_codebook_loss, example_commitment_loss = example_quantizer.apply(example_params, example_z_e)
print("quantized shape:", example_z_q.shape, "indices shape:", example_indices.shape)

# %%
# @arch class-decoder-nn-module:start
class Decoder(nn.Module):
# @arch class-decoder-nn-module:end
    latent_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 784

    # @arch decoder.nn-compact:start
    @nn.compact
    # @arch decoder.nn-compact:end
    def __call__(self, z_q):
        # Decode quantized vectors into reconstruction probabilities.
        # @arch decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z_q:start
        x = nn.Dense(self.hidden_dim, name="fc1")(z_q)  # (batch, latent_dim) -> (batch, hidden_dim)
        # @arch decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z_q:end
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
example_z_q = jnp.ones((2, 8))  # -> (2, 8)
example_params = example_decoder.init(jax.random.PRNGKey(2), example_z_q)
example_reconstruction = example_decoder.apply(example_params, example_z_q)  # (2, 8) -> (2, 64)
print("reconstruction shape:", example_reconstruction.shape)

# %%
class VectorQuantizedAutoencoder(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32
    num_codes: int = 16
    beta: float = 0.25

    def setup(self):
        # Register the encoder, quantizer, and decoder that form the VQ-VAE path.
        # @arch vectorquantizedautoencoder.setup.self-encoder-encoder-self-input_dim-self-hidden_dim-self-latent_dim:start
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        # @arch vectorquantizedautoencoder.setup.self-encoder-encoder-self-input_dim-self-hidden_dim-self-latent_dim:end
        # @arch vectorquantizedautoencoder.setup.self-quantizer-vectorquantizer-self-num_codes-self-latent_dim-self-beta:start
        self.quantizer = VectorQuantizer(self.num_codes, self.latent_dim, self.beta)
        # @arch vectorquantizedautoencoder.setup.self-quantizer-vectorquantizer-self-num_codes-self-latent_dim-self-beta:end
        # @arch vectorquantizedautoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim:start
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.input_dim)
        # @arch vectorquantizedautoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim:end

    # @arch vectorquantizedautoencoder.def-__call__-self-x:start
    def __call__(self, x):
    # @arch vectorquantizedautoencoder.def-__call__-self-x:end
        # Encode, discretize with nearest-code lookup, and reconstruct.
        # @arch vectorquantizedautoencoder.__call__.z_e-self-encoder-x:start
        z_e = self.encoder(x)  # (batch, input_dim) -> (batch, latent_dim)
        # @arch vectorquantizedautoencoder.__call__.z_e-self-encoder-x:end
        # @arch vectorquantizedautoencoder.__call__.z_q-vq_loss-encoding_indices-codebook_loss-commitment_loss-self-quantize:start
        z_q, vq_loss, encoding_indices, codebook_loss, commitment_loss = self.quantizer(z_e)  # (batch, latent_dim) -> discrete latents
        # @arch vectorquantizedautoencoder.__call__.z_q-vq_loss-encoding_indices-codebook_loss-commitment_loss-self-quantize:end
        # @arch vectorquantizedautoencoder.__call__.reconstruction-self-decoder-z_q:start
        reconstruction = self.decoder(z_q)  # (batch, latent_dim) -> (batch, input_dim)
        # @arch vectorquantizedautoencoder.__call__.reconstruction-self-decoder-z_q:end
        return reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss

    # @arch vectorquantizedautoencoder.def-loss-self-x:start
    def loss(self, x):
    # @arch vectorquantizedautoencoder.def-loss-self-x:end
        # Optimize reconstruction plus vector-quantization losses.
        # @arch vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment:start
        reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss = self.__call__(x)
        # @arch vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment:end
        # @arch vectorquantizedautoencoder.loss.reconstruction_loss-jnp-sum-x-jnp-log-reconstruction-ne-n-n-x-jnp-log-n-:start
        reconstruction_loss = -jnp.sum(x * jnp.log(reconstruction + 1e-7) + (1.0 - x) * jnp.log(1.0 - reconstruction + 1e-7))
        # @arch vectorquantizedautoencoder.loss.reconstruction_loss-jnp-sum-x-jnp-log-reconstruction-ne-n-n-x-jnp-log-n-:end
        # @arch vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss:start
        total_loss = reconstruction_loss + vq_loss  # scalar
        # @arch vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss:end
        # @arch vectorquantizedautoencoder.loss.return-total_loss-reconstruction_loss-vq_loss-codebook_loss-commitment_l:start
        return total_loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices
        # @arch vectorquantizedautoencoder.loss.return-total_loss-reconstruction_loss-vq_loss-codebook_loss-commitment_l:end


# %% [notebook-only]
# Create and run a small VQ-VAE: (2, 64) -> reconstruction, continuous/quantized latents, and code ids.
example_model = VectorQuantizedAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8, num_codes=6)
example_inputs = jnp.ones((2, 64))  # -> (2, 64)
example_params = example_model.init(jax.random.PRNGKey(3), example_inputs)
example_outputs = example_model.apply(example_params, example_inputs)
example_reconstruction, example_z_e, example_z_q, example_indices, example_vq_loss, example_codebook_loss, example_commitment_loss = example_outputs
print("reconstruction shape:", example_reconstruction.shape, "indices shape:", example_indices.shape)

# %%
# Train on a tiny synthetic batch with the VQ-VAE objective.
model = VectorQuantizedAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8, num_codes=6)
# @arch inputs-jnp-zeros-n-n:start
inputs = jnp.zeros((2, 64))  # -> (2, 64)
# @arch inputs-jnp-zeros-n-n:end
inputs = inputs.at[0, 8:24].set(1.0)  # (2, 64)
inputs = inputs.at[1, 40:56].set(1.0)  # (2, 64)
params = model.init(jax.random.PRNGKey(4), inputs)


def train_step(params, inputs, learning_rate=0.05):
    def loss_fn(current_params):
        # @arch train_step.loss_fn.loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_:start
        loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = model.apply(current_params, inputs, method=VectorQuantizedAutoencoder.loss)
        # @arch train_step.loss_fn.loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_:end
        aux = (reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices)
        return loss, aux

    # @arch train_step.loss-aux-grads-jax-value_and_grad-loss_fn-has_aux-true-params:start
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # @arch train_step.loss-aux-grads-jax-value_and_grad-loss_fn-has_aux-true-params:end
    # @arch train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:start
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    # @arch train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads:end
    # @arch train_step.reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_indic:start
    reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = aux
    # @arch train_step.reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_indic:end
    return params, loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices


# Fit reconstructions while learning discrete code vectors and encoder commitment.
for step in range(3):
    # @arch params-loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-en:start
    params, loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = train_step(params, inputs)
    # @arch params-loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-en:end

# Keep the final scalar losses and selected code ids for inspection.
# @arch final_loss-loss:start
final_loss = loss  # scalar
# @arch final_loss-loss:end
# @arch final_reconstruction_loss-reconstruction_loss:start
final_reconstruction_loss = reconstruction_loss  # scalar
# @arch final_reconstruction_loss-reconstruction_loss:end
# @arch final_vq_loss-vq_loss:start
final_vq_loss = vq_loss  # scalar
# @arch final_vq_loss-vq_loss:end
# @arch final_codebook_loss-codebook_loss:start
final_codebook_loss = codebook_loss  # scalar
# @arch final_codebook_loss-codebook_loss:end
# @arch final_commitment_loss-commitment_loss:start
final_commitment_loss = commitment_loss  # scalar
# @arch final_commitment_loss-commitment_loss:end
final_encoding_indices = encoding_indices  # (2)
