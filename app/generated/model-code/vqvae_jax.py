import jax
import jax.numpy as jnp
from flax import linen as nn

class Encoder(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32

    @nn.compact
    def __call__(self, x):
        # Encode each input into a continuous latent vector z_e(x).
        z_e = nn.Dense(self.hidden_dim, name="fc1")(x)  # (batch, input_dim) -> (batch, hidden_dim)
        z_e = nn.relu(z_e)  # (batch, hidden_dim)
        z_e = nn.Dense(self.latent_dim, name="fc2")(z_e)  # (batch, hidden_dim) -> (batch, latent_dim)
        return z_e  # (batch, latent_dim)

class VectorQuantizer(nn.Module):
    num_codes: int = 16
    latent_dim: int = 32
    beta: float = 0.25

    @nn.compact
    def __call__(self, z_e):
        # Compare each encoder vector with every codebook entry by squared distance.
        codebook = self.param("codebook", nn.initializers.uniform(scale=0.1), (self.num_codes, self.latent_dim))
        z_squared = jnp.sum(z_e ** 2, axis=1, keepdims=True)  # (batch, latent_dim) -> (batch, 1)
        codebook_squared = jnp.sum(codebook ** 2, axis=1)  # (num_codes)
        dot_products = z_e @ codebook.T  # (batch, latent_dim) -> (batch, num_codes)
        distances = z_squared + codebook_squared - 2.0 * dot_products  # (batch, num_codes)

        # Replace each continuous vector with its nearest discrete code.
        encoding_indices = jnp.argmin(distances, axis=1)  # (batch)
        quantized = codebook[encoding_indices]  # (batch) -> (batch, latent_dim)

        # Train the codebook and commit the encoder output to the selected entries.
        codebook_error = quantized - jax.lax.stop_gradient(z_e)  # (batch, latent_dim)
        codebook_loss = jnp.mean(codebook_error ** 2)  # scalar
        commitment_error = z_e - jax.lax.stop_gradient(quantized)  # (batch, latent_dim)
        commitment_loss = jnp.mean(commitment_error ** 2)  # scalar
        vq_loss = codebook_loss + self.beta * commitment_loss  # scalar

        # Use the straight-through estimator: decoder sees quantized values, encoder gets gradients.
        quantized_st = z_e + jax.lax.stop_gradient(quantized - z_e)  # (batch, latent_dim)
        return quantized_st, vq_loss, encoding_indices, codebook_loss, commitment_loss

class Decoder(nn.Module):
    latent_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 784

    @nn.compact
    def __call__(self, z_q):
        # Decode quantized vectors into reconstruction probabilities.
        x = nn.Dense(self.hidden_dim, name="fc1")(z_q)  # (batch, latent_dim) -> (batch, hidden_dim)
        x = nn.relu(x)  # (batch, hidden_dim)
        x = nn.Dense(self.output_dim, name="fc2")(x)  # (batch, hidden_dim) -> (batch, output_dim)
        reconstruction = nn.sigmoid(x)  # (batch, output_dim)
        return reconstruction  # (batch, output_dim)

class VectorQuantizedAutoencoder(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    latent_dim: int = 32
    num_codes: int = 16
    beta: float = 0.25

    def setup(self):
        # Register the encoder, quantizer, and decoder that form the VQ-VAE path.
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.quantizer = VectorQuantizer(self.num_codes, self.latent_dim, self.beta)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.input_dim)

    def __call__(self, x):
        # Encode, discretize with nearest-code lookup, and reconstruct.
        z_e = self.encoder(x)  # (batch, input_dim) -> (batch, latent_dim)
        z_q, vq_loss, encoding_indices, codebook_loss, commitment_loss = self.quantizer(z_e)  # (batch, latent_dim) -> discrete latents
        reconstruction = self.decoder(z_q)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss

    def loss(self, x):
        # Optimize reconstruction plus vector-quantization losses.
        reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss = self.__call__(x)
        reconstruction_loss = -jnp.sum(x * jnp.log(reconstruction + 1e-7) + (1.0 - x) * jnp.log(1.0 - reconstruction + 1e-7))
        total_loss = reconstruction_loss + vq_loss  # scalar
        return total_loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices

# Train on a tiny synthetic batch with the VQ-VAE objective.
model = VectorQuantizedAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8, num_codes=6)
inputs = jnp.zeros((2, 64))  # -> (2, 64)
inputs = inputs.at[0, 8:24].set(1.0)  # (2, 64)
inputs = inputs.at[1, 40:56].set(1.0)  # (2, 64)
params = model.init(jax.random.PRNGKey(4), inputs)


def train_step(params, inputs, learning_rate=0.05):
    def loss_fn(current_params):
        loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = model.apply(current_params, inputs, method=VectorQuantizedAutoencoder.loss)
        aux = (reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = aux
    return params, loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices


# Fit reconstructions while learning discrete code vectors and encoder commitment.
for step in range(3):
    params, loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = train_step(params, inputs)

# Keep the final scalar losses and selected code ids for inspection.
final_loss = loss  # scalar
final_reconstruction_loss = reconstruction_loss  # scalar
final_vq_loss = vq_loss  # scalar
final_codebook_loss = codebook_loss  # scalar
final_commitment_loss = commitment_loss  # scalar
final_encoding_indices = encoding_indices  # (2)
