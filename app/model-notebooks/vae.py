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
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class VariationalEncoder(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Flattened input width.
        hidden_dim=256,  # Hidden representation width.
        latent_dim=32  # Gaussian latent width.
    ):
        super().__init__()

        # Register a shared trunk with separate Gaussian parameter heads.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Encode each input into a diagonal Gaussian q(z|x).
        hidden = self.fc1(x)  # (batch, input_dim) -> (batch, hidden_dim)
        hidden = F.relu(hidden)  # (batch, hidden_dim)
        mu = self.fc_mu(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        logvar = self.fc_logvar(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        return mu, logvar  # two (batch, latent_dim)


# %% [notebook-only]
# Create and run the variational encoder: (2, 64) -> two (2, 8).
example_encoder = VariationalEncoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.randn(2, 64)  # -> (2, 64)
example_mu, example_logvar = example_encoder(example_inputs)  # (2, 64) -> two (2, 8)
print("mu shape:", example_mu.shape, "logvar shape:", example_logvar.shape)

# %%
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim=32,  # Gaussian latent width.
        hidden_dim=256,  # Hidden reconstruction width.
        output_dim=784  # Flattened output width.
    ):
        super().__init__()

        # Register a decoder that maps latent samples to Bernoulli probabilities.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # Decode latent samples into reconstruction probabilities.
        reconstruction = self.net(z)  # (batch, latent_dim) -> (batch, output_dim)
        return reconstruction  # (batch, output_dim)


# %% [notebook-only]
# Create and run the decoder: (2, 8) -> (2, 64).
example_decoder = Decoder(latent_dim=8, hidden_dim=24, output_dim=64)
example_z = torch.randn(2, 8)  # -> (2, 8)
example_reconstruction = example_decoder(example_z)  # (2, 8) -> (2, 64)
print("reconstruction shape:", example_reconstruction.shape)

# %%
class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Flattened input and reconstruction width.
        hidden_dim=256,  # Encoder/decoder hidden width.
        latent_dim=32  # Gaussian latent width.
    ):
        super().__init__()

        # Register the inference network and generative decoder.
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        # Sample z = mu + sigma * epsilon so gradients flow through mu and logvar.
        std = torch.exp(0.5 * logvar)  # (batch, latent_dim)
        epsilon = torch.randn_like(std)  # (batch, latent_dim)
        z = mu + std * epsilon  # (batch, latent_dim)
        return z  # (batch, latent_dim)

    def forward(self, x):
        # Infer q(z|x), sample a latent code, and decode it.
        mu, logvar = self.encoder(x)  # (batch, input_dim) -> two (batch, latent_dim)
        z = self.reparameterize(mu, logvar)  # two (batch, latent_dim) -> (batch, latent_dim)
        reconstruction = self.decoder(z)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction, mu, logvar, z  # (batch, input_dim), three (batch, latent_dim)

    def loss(self, x):
        # Optimize the negative ELBO: reconstruction loss plus KL to N(0, I).
        reconstruction, mu, logvar, z = self.forward(x)  # (batch, input_dim) -> reconstruction and latent stats
        reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")  # scalar
        kl_terms = 1 + logvar - mu.pow(2) - logvar.exp()  # (batch, latent_dim)
        kl_loss = -0.5 * torch.sum(kl_terms)  # scalar
        total_loss = reconstruction_loss + kl_loss  # scalar
        return total_loss, reconstruction_loss, kl_loss, z


# %% [notebook-only]
# Create and run a small VAE: (2, 64) -> (2, 64), two (2, 8), (2, 8).
example_model = VariationalAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.rand(2, 64)  # -> (2, 64)
example_reconstruction, example_mu, example_logvar, example_z = example_model(example_inputs)  # (2, 64) -> (2, 64), three (2, 8)
print("reconstruction shape:", example_reconstruction.shape, "z shape:", example_z.shape)

# %%
# Train on a tiny synthetic batch with the VAE objective.
model = VariationalAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
inputs = torch.zeros(2, 64)  # -> (2, 64)
inputs[0, 8:24] = 1.0  # (2, 64)
inputs[1, 40:56] = 1.0  # (2, 64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Fit reconstruction quality while regularizing q(z|x) toward the unit Gaussian prior.
for step in range(3):
    optimizer.zero_grad()
    loss, reconstruction_loss, kl_loss, z = model.loss(inputs)  # (2, 64) -> scalar losses and (2, 8)
    loss.backward()
    optimizer.step()

# Keep the final scalar losses and latent sample for inspection.
final_loss = loss.item()  # scalar
final_reconstruction_loss = reconstruction_loss.item()  # scalar
final_kl_loss = kl_loss.item()  # scalar
final_latent_sample = z.detach()  # (2, 8)
