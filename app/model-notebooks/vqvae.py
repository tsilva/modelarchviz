# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Flattened input width.
        hidden_dim=256,  # Hidden representation width.
        latent_dim=32  # Continuous encoder output width.
    ):
        super().__init__()

        # Register a small inference network that emits continuous vectors before quantization.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        # Encode each input into a continuous latent vector z_e(x).
        z_e = self.net(x)  # (batch, input_dim) -> (batch, latent_dim)
        return z_e  # (batch, latent_dim)


# %% [notebook-only]
# Create and run the encoder: (2, 64) -> (2, 8).
example_encoder = Encoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.randn(2, 64)  # -> (2, 64)
example_z_e = example_encoder(example_inputs)  # (2, 64) -> (2, 8)
print("encoder output shape:", example_z_e.shape)

# %%
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes=16,  # Number of entries in the discrete codebook.
        latent_dim=32,  # Width of each code vector.
        beta=0.25  # Commitment-loss weight.
    ):
        super().__init__()
        self.beta = beta

        # Register a learnable table of discrete latent embeddings.
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        # Compare each encoder vector with every codebook entry by squared distance.
        z_squared = torch.sum(z_e ** 2, dim=1, keepdim=True)  # (batch, latent_dim) -> (batch, 1)
        codebook_squared = torch.sum(self.codebook.weight ** 2, dim=1)  # (num_codes)
        dot_products = z_e @ self.codebook.weight.t()  # (batch, latent_dim) -> (batch, num_codes)
        distances = z_squared + codebook_squared - 2.0 * dot_products  # (batch, num_codes)

        # Replace each continuous vector with its nearest discrete code.
        encoding_indices = torch.argmin(distances, dim=1)  # (batch)
        quantized = self.codebook(encoding_indices)  # (batch) -> (batch, latent_dim)

        # Train the codebook and commit the encoder output to the selected entries.
        codebook_loss = F.mse_loss(quantized, z_e.detach())  # scalar
        commitment_loss = F.mse_loss(z_e, quantized.detach())  # scalar
        vq_loss = codebook_loss + self.beta * commitment_loss  # scalar

        # Use the straight-through estimator: decoder sees quantized values, encoder gets gradients.
        quantized_st = z_e + (quantized - z_e).detach()  # (batch, latent_dim)
        return quantized_st, vq_loss, encoding_indices, codebook_loss, commitment_loss


# %% [notebook-only]
# Quantize encoder outputs with a discrete codebook: (2, 8) -> (2, 8), (2).
example_quantizer = VectorQuantizer(num_codes=6, latent_dim=8)
example_z_e = torch.randn(2, 8)  # -> (2, 8)
example_z_q, example_vq_loss, example_indices, example_codebook_loss, example_commitment_loss = example_quantizer(example_z_e)
print("quantized shape:", example_z_q.shape, "indices shape:", example_indices.shape)

# %%
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim=32,  # Quantized latent width.
        hidden_dim=256,  # Hidden reconstruction width.
        output_dim=784  # Flattened output width.
    ):
        super().__init__()

        # Register a decoder that maps selected code vectors back to input probabilities.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_q):
        # Decode quantized vectors into reconstruction probabilities.
        reconstruction = self.net(z_q)  # (batch, latent_dim) -> (batch, output_dim)
        return reconstruction  # (batch, output_dim)


# %% [notebook-only]
# Create and run the decoder: (2, 8) -> (2, 64).
example_decoder = Decoder(latent_dim=8, hidden_dim=24, output_dim=64)
example_z_q = torch.randn(2, 8)  # -> (2, 8)
example_reconstruction = example_decoder(example_z_q)  # (2, 8) -> (2, 64)
print("reconstruction shape:", example_reconstruction.shape)

# %%
class VectorQuantizedAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Flattened input and reconstruction width.
        hidden_dim=256,  # Encoder/decoder hidden width.
        latent_dim=32,  # Width of each continuous and quantized latent.
        num_codes=16,  # Number of discrete latent codes.
        beta=0.25  # Commitment-loss weight.
    ):
        super().__init__()

        # Register the encoder, quantizer, and decoder that form the VQ-VAE path.
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_codes, latent_dim, beta)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        # Encode, discretize with nearest-code lookup, and reconstruct.
        z_e = self.encoder(x)  # (batch, input_dim) -> (batch, latent_dim)
        z_q, vq_loss, encoding_indices, codebook_loss, commitment_loss = self.quantizer(z_e)  # (batch, latent_dim) -> discrete latents
        reconstruction = self.decoder(z_q)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss

    def loss(self, x):
        # Optimize reconstruction plus vector-quantization losses.
        reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss = self.forward(x)
        reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")  # scalar
        total_loss = reconstruction_loss + vq_loss  # scalar
        return total_loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices


# %% [notebook-only]
# Create and run a small VQ-VAE: (2, 64) -> reconstruction, continuous/quantized latents, and code ids.
example_model = VectorQuantizedAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8, num_codes=6)
example_inputs = torch.rand(2, 64)  # -> (2, 64)
example_outputs = example_model(example_inputs)
example_reconstruction, example_z_e, example_z_q, example_indices, example_vq_loss, example_codebook_loss, example_commitment_loss = example_outputs
print("reconstruction shape:", example_reconstruction.shape, "indices shape:", example_indices.shape)

# %%
# Train on a tiny synthetic batch with the VQ-VAE objective.
model = VectorQuantizedAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8, num_codes=6)
inputs = torch.zeros(2, 64)  # -> (2, 64)
inputs[0, 8:24] = 1.0  # (2, 64)
inputs[1, 40:56] = 1.0  # (2, 64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Fit reconstructions while learning discrete code vectors and encoder commitment.
for step in range(3):
    optimizer.zero_grad()
    loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = model.loss(inputs)
    loss.backward()
    optimizer.step()

# Keep the final scalar losses and selected code ids for inspection.
final_loss = loss.item()  # scalar
final_reconstruction_loss = reconstruction_loss.item()  # scalar
final_vq_loss = vq_loss.item()  # scalar
final_codebook_loss = codebook_loss.item()  # scalar
final_commitment_loss = commitment_loss.item()  # scalar
final_encoding_indices = encoding_indices.detach()  # (2)
