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
        latent_dim=32  # Bottleneck code width.
    ):
        super().__init__()

        # Register a feed-forward compression network.
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
        # Compress input vectors into low-dimensional latent codes.
        # @arch encoder.forward.z-self-net-x:start
        z = self.net(x)  # (batch, input_dim) -> (batch, latent_dim)
        # @arch encoder.forward.z-self-net-x:end
        # @arch encoder.forward.return-z:start
        return z  # (batch, latent_dim)
        # @arch encoder.forward.return-z:end


# %% [notebook-only]
# Create and run the encoder: (2, 64) -> (2, 8).
example_encoder = Encoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.randn(2, 64)  # -> (2, 64)
example_codes = example_encoder(example_inputs)  # (2, 64) -> (2, 8)
print("latent codes shape:", example_codes.shape)

# %%
# @arch class-decoder-nn-module:start
class Decoder(nn.Module):
# @arch class-decoder-nn-module:end
    def __init__(
        self,
        latent_dim=32,  # Bottleneck code width.
        hidden_dim=256,  # Hidden reconstruction width.
        output_dim=784  # Flattened output width.
    ):
        super().__init__()

        # Register a feed-forward reconstruction network.
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

    # @arch decoder.def-forward-self-z:start
    def forward(self, z):
    # @arch decoder.def-forward-self-z:end
        # Decode latent codes back into input-shaped reconstructions.
        # @arch decoder.forward.reconstruction-self-net-z:start
        reconstruction = self.net(z)  # (batch, latent_dim) -> (batch, output_dim)
        # @arch decoder.forward.reconstruction-self-net-z:end
        # @arch decoder.forward.return-reconstruction:start
        return reconstruction  # (batch, output_dim)
        # @arch decoder.forward.return-reconstruction:end


# %% [notebook-only]
# Create and run the decoder: (2, 8) -> (2, 64).
example_decoder = Decoder(latent_dim=8, hidden_dim=24, output_dim=64)
example_codes = torch.randn(2, 8)  # -> (2, 8)
example_reconstruction = example_decoder(example_codes)  # (2, 8) -> (2, 64)
print("reconstruction shape:", example_reconstruction.shape)

# %%
class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Flattened input and reconstruction width.
        hidden_dim=256,  # Encoder/decoder hidden width.
        latent_dim=32  # Bottleneck code width.
    ):
        super().__init__()

        # Register mirrored encoder and decoder modules.
        # @arch autoencoder.self-encoder-encoder-input_dim-hidden_dim-latent_dim:start
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        # @arch autoencoder.self-encoder-encoder-input_dim-hidden_dim-latent_dim:end
        # @arch autoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim:start
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        # @arch autoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim:end

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

    # @arch autoencoder.def-forward-self-x:start
    def forward(self, x):
    # @arch autoencoder.def-forward-self-x:end
        # Encode through the bottleneck, then reconstruct the original input.
        # @arch autoencoder.forward.z-self-encode-x:start
        z = self.encode(x)  # (batch, input_dim) -> (batch, latent_dim)
        # @arch autoencoder.forward.z-self-encode-x:end
        # @arch autoencoder.forward.reconstruction-self-decode-z:start
        reconstruction = self.decode(z)  # (batch, latent_dim) -> (batch, input_dim)
        # @arch autoencoder.forward.reconstruction-self-decode-z:end
        # @arch autoencoder.forward.return-reconstruction-z:start
        return reconstruction, z  # (batch, input_dim), (batch, latent_dim)
        # @arch autoencoder.forward.return-reconstruction-z:end


# %% [notebook-only]
# Create and run a small autoencoder: (2, 64) -> (2, 64), (2, 8).
example_model = Autoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.randn(2, 64)  # -> (2, 64)
example_reconstruction, example_codes = example_model(example_inputs)  # (2, 64) -> (2, 64), (2, 8)
print("reconstruction shape:", example_reconstruction.shape, "latent shape:", example_codes.shape)

# %%
# Train on a tiny synthetic reconstruction batch.
model = Autoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
# @arch inputs-torch-zeros-n-n:start
inputs = torch.zeros(2, 64)  # -> (2, 64)
# @arch inputs-torch-zeros-n-n:end
inputs[0, 8:24] = 1.0  # (2, 64)
inputs[1, 40:56] = 1.0  # (2, 64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Fit the model to reconstruct its own inputs.
# @arch for-step-in-range-n:start
for step in range(3):
# @arch for-step-in-range-n:end
    # @arch optimizer-zero_grad:start
    optimizer.zero_grad()
    # @arch optimizer-zero_grad:end
    # @arch reconstruction-latent_codes-model-inputs:start
    reconstruction, latent_codes = model(inputs)  # (2, 64) -> (2, 64), (2, 8)
    # @arch reconstruction-latent_codes-model-inputs:end
    # @arch loss-f-mse_loss-reconstruction-inputs:start
    loss = F.mse_loss(reconstruction, inputs)  # (2, 64), (2, 64) -> scalar
    # @arch loss-f-mse_loss-reconstruction-inputs:end
    # @arch loss-backward:start
    loss.backward()
    # @arch loss-backward:end
    # @arch optimizer-step:start
    optimizer.step()
    # @arch optimizer-step:end

# Keep the final scalar loss and latent codes for inspection.
final_loss = loss.item()  # scalar
# @arch final_latent_codes-latent_codes-detach:start
final_latent_codes = latent_codes.detach()  # (2, 8)
# @arch final_latent_codes-latent_codes-detach:end
