# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# @arch class-encoder-nn-module:start
class Encoder(nn.Module):
# @arch class-encoder-nn-module:end
    def __init__(
        self,
        input_dim=784,  # Flattened input width.
        hidden_dim=256,  # Hidden representation width.
        latent_dim=32  # Continuous encoder output width.
    ):
        super().__init__()

        # Register a small inference network that emits continuous vectors before quantization.
        # @arch encoder.self-net-nn-sequential:start
        self.net = nn.Sequential(
        # @arch encoder.self-net-nn-sequential:end
            # @arch encoder.nn-linear-input_dim-hidden_dim:start
            nn.Linear(input_dim, hidden_dim),
            # @arch encoder.nn-linear-input_dim-hidden_dim:end
            # @arch encoder.nn-relu:start
            nn.ReLU(),
            # @arch encoder.nn-relu:end
            # @arch encoder.nn-linear-hidden_dim-latent_dim:start
            nn.Linear(hidden_dim, latent_dim),
            # @arch encoder.nn-linear-hidden_dim-latent_dim:end
        )

    # @arch encoder.def-forward-self-x:start
    def forward(self, x):
    # @arch encoder.def-forward-self-x:end
        # Encode each input into a continuous latent vector z_e(x).
        # @arch encoder.forward.z_e-self-net-x:start
        z_e = self.net(x)  # (batch, input_dim) -> (batch, latent_dim)
        # @arch encoder.forward.z_e-self-net-x:end
        # @arch encoder.forward.return-z_e:start
        return z_e  # (batch, latent_dim)
        # @arch encoder.forward.return-z_e:end


# %% [notebook-only]
# Create and run the encoder: (2, 64) -> (2, 8).
example_encoder = Encoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.randn(2, 64)  # -> (2, 64)
example_z_e = example_encoder(example_inputs)  # (2, 64) -> (2, 8)
print("encoder output shape:", example_z_e.shape)

# %%
# @arch class-vectorquantizer-nn-module:start
class VectorQuantizer(nn.Module):
# @arch class-vectorquantizer-nn-module:end
    def __init__(
        self,
        num_codes=16,  # Number of entries in the discrete codebook.
        latent_dim=32,  # Width of each code vector.
        beta=0.25  # Commitment-loss weight.
    ):
        super().__init__()
        self.beta = beta

        # Register a learnable table of discrete latent embeddings.
        # @arch vectorquantizer.self-codebook-nn-embedding-num_codes-latent_dim:start
        self.codebook = nn.Embedding(num_codes, latent_dim)
        # @arch vectorquantizer.self-codebook-nn-embedding-num_codes-latent_dim:end
        # @arch vectorquantizer.self-codebook-weight-data-uniform_-n-num_codes-n-num_codes:start
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        # @arch vectorquantizer.self-codebook-weight-data-uniform_-n-num_codes-n-num_codes:end

    # @arch vectorquantizer.def-forward-self-z_e:start
    def forward(self, z_e):
    # @arch vectorquantizer.def-forward-self-z_e:end
        # Compare each encoder vector with every codebook entry by squared distance.
        # @arch vectorquantizer.forward.z_squared-torch-sum-z_e-n-dim-n-keepdim-true:start
        z_squared = torch.sum(z_e ** 2, dim=1, keepdim=True)  # (batch, latent_dim) -> (batch, 1)
        # @arch vectorquantizer.forward.z_squared-torch-sum-z_e-n-dim-n-keepdim-true:end
        # @arch vectorquantizer.forward.codebook_squared-torch-sum-self-codebook-weight-n-dim-n:start
        codebook_squared = torch.sum(self.codebook.weight ** 2, dim=1)  # (num_codes)
        # @arch vectorquantizer.forward.codebook_squared-torch-sum-self-codebook-weight-n-dim-n:end
        # @arch vectorquantizer.forward.dot_products-z_e-self-codebook-weight-t:start
        dot_products = z_e @ self.codebook.weight.t()  # (batch, latent_dim) -> (batch, num_codes)
        # @arch vectorquantizer.forward.dot_products-z_e-self-codebook-weight-t:end
        # @arch vectorquantizer.forward.distances-z_squared-codebook_squared-n-dot_products:start
        distances = z_squared + codebook_squared - 2.0 * dot_products  # (batch, num_codes)
        # @arch vectorquantizer.forward.distances-z_squared-codebook_squared-n-dot_products:end

        # Replace each continuous vector with its nearest discrete code.
        # @arch vectorquantizer.forward.encoding_indices-torch-argmin-distances-dim-n:start
        encoding_indices = torch.argmin(distances, dim=1)  # (batch)
        # @arch vectorquantizer.forward.encoding_indices-torch-argmin-distances-dim-n:end
        # @arch vectorquantizer.forward.quantized-self-codebook-encoding_indices:start
        quantized = self.codebook(encoding_indices)  # (batch) -> (batch, latent_dim)
        # @arch vectorquantizer.forward.quantized-self-codebook-encoding_indices:end

        # Train the codebook and commit the encoder output to the selected entries.
        # @arch vectorquantizer.forward.codebook_loss-f-mse_loss-quantized-z_e-detach:start
        codebook_loss = F.mse_loss(quantized, z_e.detach())  # scalar
        # @arch vectorquantizer.forward.codebook_loss-f-mse_loss-quantized-z_e-detach:end
        # @arch vectorquantizer.forward.commitment_loss-f-mse_loss-z_e-quantized-detach:start
        commitment_loss = F.mse_loss(z_e, quantized.detach())  # scalar
        # @arch vectorquantizer.forward.commitment_loss-f-mse_loss-z_e-quantized-detach:end
        # @arch vectorquantizer.forward.vq_loss-codebook_loss-self-beta-commitment_loss:start
        vq_loss = codebook_loss + self.beta * commitment_loss  # scalar
        # @arch vectorquantizer.forward.vq_loss-codebook_loss-self-beta-commitment_loss:end

        # Use the straight-through estimator: decoder sees quantized values, encoder gets gradients.
        # @arch vectorquantizer.forward.quantized_st-z_e-quantized-z_e-detach:start
        quantized_st = z_e + (quantized - z_e).detach()  # (batch, latent_dim)
        # @arch vectorquantizer.forward.quantized_st-z_e-quantized-z_e-detach:end
        # @arch vectorquantizer.forward.return-quantized_st-vq_loss-encoding_indices-codebook_loss-commitment_lo:start
        return quantized_st, vq_loss, encoding_indices, codebook_loss, commitment_loss
        # @arch vectorquantizer.forward.return-quantized_st-vq_loss-encoding_indices-codebook_loss-commitment_lo:end


# %% [notebook-only]
# Quantize encoder outputs with a discrete codebook: (2, 8) -> (2, 8), (2).
example_quantizer = VectorQuantizer(num_codes=6, latent_dim=8)
example_z_e = torch.randn(2, 8)  # -> (2, 8)
example_z_q, example_vq_loss, example_indices, example_codebook_loss, example_commitment_loss = example_quantizer(example_z_e)
print("quantized shape:", example_z_q.shape, "indices shape:", example_indices.shape)

# %%
# @arch class-decoder-nn-module:start
class Decoder(nn.Module):
# @arch class-decoder-nn-module:end
    def __init__(
        self,
        latent_dim=32,  # Quantized latent width.
        hidden_dim=256,  # Hidden reconstruction width.
        output_dim=784  # Flattened output width.
    ):
        super().__init__()

        # Register a decoder that maps selected code vectors back to input probabilities.
        # @arch decoder.self-net-nn-sequential:start
        self.net = nn.Sequential(
        # @arch decoder.self-net-nn-sequential:end
            # @arch decoder.nn-linear-latent_dim-hidden_dim:start
            nn.Linear(latent_dim, hidden_dim),
            # @arch decoder.nn-linear-latent_dim-hidden_dim:end
            # @arch decoder.nn-relu:start
            nn.ReLU(),
            # @arch decoder.nn-relu:end
            # @arch decoder.nn-linear-hidden_dim-output_dim:start
            nn.Linear(hidden_dim, output_dim),
            # @arch decoder.nn-linear-hidden_dim-output_dim:end
            # @arch decoder.nn-sigmoid:start
            nn.Sigmoid(),
            # @arch decoder.nn-sigmoid:end
        )

    # @arch decoder.def-forward-self-z_q:start
    def forward(self, z_q):
    # @arch decoder.def-forward-self-z_q:end
        # Decode quantized vectors into reconstruction probabilities.
        # @arch decoder.forward.reconstruction-self-net-z_q:start
        reconstruction = self.net(z_q)  # (batch, latent_dim) -> (batch, output_dim)
        # @arch decoder.forward.reconstruction-self-net-z_q:end
        # @arch decoder.forward.return-reconstruction:start
        return reconstruction  # (batch, output_dim)
        # @arch decoder.forward.return-reconstruction:end


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
        # @arch vectorquantizedautoencoder.self-encoder-encoder-input_dim-hidden_dim-latent_dim:start
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        # @arch vectorquantizedautoencoder.self-encoder-encoder-input_dim-hidden_dim-latent_dim:end
        # @arch vectorquantizedautoencoder.self-quantizer-vectorquantizer-num_codes-latent_dim-beta:start
        self.quantizer = VectorQuantizer(num_codes, latent_dim, beta)
        # @arch vectorquantizedautoencoder.self-quantizer-vectorquantizer-num_codes-latent_dim-beta:end
        # @arch vectorquantizedautoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim:start
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        # @arch vectorquantizedautoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim:end

    # @arch vectorquantizedautoencoder.def-forward-self-x:start
    def forward(self, x):
    # @arch vectorquantizedautoencoder.def-forward-self-x:end
        # Encode, discretize with nearest-code lookup, and reconstruct.
        # @arch vectorquantizedautoencoder.forward.z_e-self-encoder-x:start
        z_e = self.encoder(x)  # (batch, input_dim) -> (batch, latent_dim)
        # @arch vectorquantizedautoencoder.forward.z_e-self-encoder-x:end
        # @arch vectorquantizedautoencoder.forward.z_q-vq_loss-encoding_indices-codebook_loss-commitment_loss-self-quantize:start
        z_q, vq_loss, encoding_indices, codebook_loss, commitment_loss = self.quantizer(z_e)  # (batch, latent_dim) -> discrete latents
        # @arch vectorquantizedautoencoder.forward.z_q-vq_loss-encoding_indices-codebook_loss-commitment_loss-self-quantize:end
        # @arch vectorquantizedautoencoder.forward.reconstruction-self-decoder-z_q:start
        reconstruction = self.decoder(z_q)  # (batch, latent_dim) -> (batch, input_dim)
        # @arch vectorquantizedautoencoder.forward.reconstruction-self-decoder-z_q:end
        return reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss

    # @arch vectorquantizedautoencoder.def-loss-self-x:start
    def loss(self, x):
    # @arch vectorquantizedautoencoder.def-loss-self-x:end
        # Optimize reconstruction plus vector-quantization losses.
        # @arch vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment:start
        reconstruction, z_e, z_q, encoding_indices, vq_loss, codebook_loss, commitment_loss = self.forward(x)
        # @arch vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment:end
        # @arch vectorquantizedautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su:start
        reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")  # scalar
        # @arch vectorquantizedautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su:end
        # @arch vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss:start
        total_loss = reconstruction_loss + vq_loss  # scalar
        # @arch vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss:end
        # @arch vectorquantizedautoencoder.loss.return-total_loss-reconstruction_loss-vq_loss-codebook_loss-commitment_l:start
        return total_loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices
        # @arch vectorquantizedautoencoder.loss.return-total_loss-reconstruction_loss-vq_loss-codebook_loss-commitment_l:end


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
# @arch inputs-torch-zeros-n-n:start
inputs = torch.zeros(2, 64)  # -> (2, 64)
# @arch inputs-torch-zeros-n-n:end
inputs[0, 8:24] = 1.0  # (2, 64)
inputs[1, 40:56] = 1.0  # (2, 64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Fit reconstructions while learning discrete code vectors and encoder commitment.
for step in range(3):
    optimizer.zero_grad()
    # @arch loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_:start
    loss, reconstruction_loss, vq_loss, codebook_loss, commitment_loss, encoding_indices = model.loss(inputs)
    # @arch loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_:end
    loss.backward()
    optimizer.step()

# Keep the final scalar losses and selected code ids for inspection.
# @arch final_loss-loss-item:start
final_loss = loss.item()  # scalar
# @arch final_loss-loss-item:end
# @arch final_reconstruction_loss-reconstruction_loss-item:start
final_reconstruction_loss = reconstruction_loss.item()  # scalar
# @arch final_reconstruction_loss-reconstruction_loss-item:end
# @arch final_vq_loss-vq_loss-item:start
final_vq_loss = vq_loss.item()  # scalar
# @arch final_vq_loss-vq_loss-item:end
# @arch final_codebook_loss-codebook_loss-item:start
final_codebook_loss = codebook_loss.item()  # scalar
# @arch final_codebook_loss-codebook_loss-item:end
# @arch final_commitment_loss-commitment_loss-item:start
final_commitment_loss = commitment_loss.item()  # scalar
# @arch final_commitment_loss-commitment_loss-item:end
final_encoding_indices = encoding_indices.detach()  # (2)
