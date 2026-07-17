# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# @arch class-variationalencoder-nn-module:start
class VariationalEncoder(nn.Module):
# @arch class-variationalencoder-nn-module:end
    def __init__(
        self,
        input_dim=784,  # Flattened input width.
        hidden_dim=256,  # Hidden representation width.
        latent_dim=32  # Gaussian latent width.
    ):
        super().__init__()

        # Register a shared trunk with separate Gaussian parameter heads.
        # @arch variationalencoder.self-fcn-nn-linear-input_dim-hidden_dim:start
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # @arch variationalencoder.self-fcn-nn-linear-input_dim-hidden_dim:end
        # @arch variationalencoder.self-fc_mu-nn-linear-hidden_dim-latent_dim:start
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # @arch variationalencoder.self-fc_mu-nn-linear-hidden_dim-latent_dim:end
        # @arch variationalencoder.self-fc_logvar-nn-linear-hidden_dim-latent_dim:start
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # @arch variationalencoder.self-fc_logvar-nn-linear-hidden_dim-latent_dim:end

    # @arch variationalencoder.def-forward-self-x:start
    def forward(self, x):
    # @arch variationalencoder.def-forward-self-x:end
        # Encode each input into a diagonal Gaussian q(z|x).
        # @arch variationalencoder.forward.hidden-self-fcn-x:start
        hidden = self.fc1(x)  # (batch, input_dim) -> (batch, hidden_dim)
        # @arch variationalencoder.forward.hidden-self-fcn-x:end
        # @arch variationalencoder.forward.hidden-f-relu-hidden:start
        hidden = F.relu(hidden)  # (batch, hidden_dim)
        # @arch variationalencoder.forward.hidden-f-relu-hidden:end
        # @arch variationalencoder.forward.mu-self-fc_mu-hidden:start
        mu = self.fc_mu(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        # @arch variationalencoder.forward.mu-self-fc_mu-hidden:end
        # @arch variationalencoder.forward.logvar-self-fc_logvar-hidden:start
        logvar = self.fc_logvar(hidden)  # (batch, hidden_dim) -> (batch, latent_dim)
        # @arch variationalencoder.forward.logvar-self-fc_logvar-hidden:end
        # @arch variationalencoder.forward.return-mu-logvar:start
        return mu, logvar  # two (batch, latent_dim)
        # @arch variationalencoder.forward.return-mu-logvar:end


# %% [notebook-only]
# Create and run the variational encoder: (2, 64) -> two (2, 8).
example_encoder = VariationalEncoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.randn(2, 64)  # -> (2, 64)
example_mu, example_logvar = example_encoder(example_inputs)  # (2, 64) -> two (2, 8)
print("mu shape:", example_mu.shape, "logvar shape:", example_logvar.shape)

# %%
# @arch class-decoder-nn-module:start
class Decoder(nn.Module):
# @arch class-decoder-nn-module:end
    def __init__(
        self,
        latent_dim=32,  # Gaussian latent width.
        hidden_dim=256,  # Hidden reconstruction width.
        output_dim=784  # Flattened output width.
    ):
        super().__init__()

        # Register a decoder that maps latent samples to Bernoulli probabilities.
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
        # Decode latent samples into reconstruction probabilities.
        # @arch decoder.forward.reconstruction-self-net-z:start
        reconstruction = self.net(z)  # (batch, latent_dim) -> (batch, output_dim)
        # @arch decoder.forward.reconstruction-self-net-z:end
        # @arch decoder.forward.return-reconstruction:start
        return reconstruction  # (batch, output_dim)
        # @arch decoder.forward.return-reconstruction:end


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
        # @arch variationalautoencoder.self-encoder-variationalencoder-input_dim-hidden_dim-latent_dim:start
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        # @arch variationalautoencoder.self-encoder-variationalencoder-input_dim-hidden_dim-latent_dim:end
        # @arch variationalautoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim:start
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        # @arch variationalautoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim:end

    # @arch variationalautoencoder.def-reparameterize-self-mu-logvar:start
    def reparameterize(self, mu, logvar):
    # @arch variationalautoencoder.def-reparameterize-self-mu-logvar:end
        # Sample z = mu + sigma * epsilon so gradients flow through mu and logvar.
        # @arch variationalautoencoder.reparameterize.std-torch-exp-n-logvar:start
        std = torch.exp(0.5 * logvar)  # (batch, latent_dim)
        # @arch variationalautoencoder.reparameterize.std-torch-exp-n-logvar:end
        # @arch variationalautoencoder.reparameterize.epsilon-torch-randn_like-std:start
        epsilon = torch.randn_like(std)  # (batch, latent_dim)
        # @arch variationalautoencoder.reparameterize.epsilon-torch-randn_like-std:end
        # @arch variationalautoencoder.reparameterize.z-mu-std-epsilon:start
        z = mu + std * epsilon  # (batch, latent_dim)
        # @arch variationalautoencoder.reparameterize.z-mu-std-epsilon:end
        # @arch variationalautoencoder.reparameterize.return-z:start
        return z  # (batch, latent_dim)
        # @arch variationalautoencoder.reparameterize.return-z:end

    # @arch variationalautoencoder.def-forward-self-x:start
    def forward(self, x):
    # @arch variationalautoencoder.def-forward-self-x:end
        # Infer q(z|x), sample a latent code, and decode it.
        # @arch variationalautoencoder.forward.mu-logvar-self-encoder-x:start
        mu, logvar = self.encoder(x)  # (batch, input_dim) -> two (batch, latent_dim)
        # @arch variationalautoencoder.forward.mu-logvar-self-encoder-x:end
        # @arch variationalautoencoder.forward.z-self-reparameterize-mu-logvar:start
        z = self.reparameterize(mu, logvar)  # two (batch, latent_dim) -> (batch, latent_dim)
        # @arch variationalautoencoder.forward.z-self-reparameterize-mu-logvar:end
        # @arch variationalautoencoder.forward.reconstruction-self-decoder-z:start
        reconstruction = self.decoder(z)  # (batch, latent_dim) -> (batch, input_dim)
        # @arch variationalautoencoder.forward.reconstruction-self-decoder-z:end
        # @arch variationalautoencoder.forward.return-reconstruction-mu-logvar-z:start
        return reconstruction, mu, logvar, z  # (batch, input_dim), three (batch, latent_dim)
        # @arch variationalautoencoder.forward.return-reconstruction-mu-logvar-z:end

    # @arch variationalautoencoder.def-loss-self-x:start
    def loss(self, x):
    # @arch variationalautoencoder.def-loss-self-x:end
        # Optimize the negative ELBO: reconstruction loss plus KL to N(0, I).
        # @arch variationalautoencoder.loss.reconstruction-mu-logvar-z-self-forward-x:start
        reconstruction, mu, logvar, z = self.forward(x)  # (batch, input_dim) -> reconstruction and latent stats
        # @arch variationalautoencoder.loss.reconstruction-mu-logvar-z-self-forward-x:end
        # @arch variationalautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su:start
        reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")  # scalar
        # @arch variationalautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su:end
        # @arch variationalautoencoder.loss.kl_terms-n-logvar-mu-pow-n-logvar-exp:start
        kl_terms = 1 + logvar - mu.pow(2) - logvar.exp()  # (batch, latent_dim)
        # @arch variationalautoencoder.loss.kl_terms-n-logvar-mu-pow-n-logvar-exp:end
        # @arch variationalautoencoder.loss.kl_loss-n-torch-sum-kl_terms:start
        kl_loss = -0.5 * torch.sum(kl_terms)  # scalar
        # @arch variationalautoencoder.loss.kl_loss-n-torch-sum-kl_terms:end
        # @arch variationalautoencoder.loss.total_loss-reconstruction_loss-kl_loss:start
        total_loss = reconstruction_loss + kl_loss  # scalar
        # @arch variationalautoencoder.loss.total_loss-reconstruction_loss-kl_loss:end
        # @arch variationalautoencoder.loss.return-total_loss-reconstruction_loss-kl_loss-z:start
        return total_loss, reconstruction_loss, kl_loss, z
        # @arch variationalautoencoder.loss.return-total_loss-reconstruction_loss-kl_loss-z:end


# %% [notebook-only]
# Create and run a small VAE: (2, 64) -> (2, 64), two (2, 8), (2, 8).
example_model = VariationalAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
example_inputs = torch.rand(2, 64)  # -> (2, 64)
example_reconstruction, example_mu, example_logvar, example_z = example_model(example_inputs)  # (2, 64) -> (2, 64), three (2, 8)
print("reconstruction shape:", example_reconstruction.shape, "z shape:", example_z.shape)

# %%
# Train on a tiny synthetic batch with the VAE objective.
model = VariationalAutoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
# @arch inputs-torch-zeros-n-n:start
inputs = torch.zeros(2, 64)  # -> (2, 64)
# @arch inputs-torch-zeros-n-n:end
inputs[0, 8:24] = 1.0  # (2, 64)
inputs[1, 40:56] = 1.0  # (2, 64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Fit reconstruction quality while regularizing q(z|x) toward the unit Gaussian prior.
for step in range(3):
    optimizer.zero_grad()
    # @arch loss-reconstruction_loss-kl_loss-z-model-loss-inputs:start
    loss, reconstruction_loss, kl_loss, z = model.loss(inputs)  # (2, 64) -> scalar losses and (2, 8)
    # @arch loss-reconstruction_loss-kl_loss-z-model-loss-inputs:end
    # @arch loss-backward:start
    loss.backward()
    # @arch loss-backward:end
    # @arch optimizer-step:start
    optimizer.step()
    # @arch optimizer-step:end

# Keep the final scalar losses and latent sample for inspection.
final_loss = loss.item()  # scalar
final_reconstruction_loss = reconstruction_loss.item()  # scalar
final_kl_loss = kl_loss.item()  # scalar
# @arch final_latent_sample-z-detach:start
final_latent_sample = z.detach()  # (2, 8)
# @arch final_latent_sample-z-detach:end
