import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Flattened input width.
        hidden_dim=256,  # Hidden representation width.
        latent_dim=32  # Bottleneck code width.
    ):
        super().__init__()

        # Register a feed-forward compression network.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        # Compress input vectors into low-dimensional latent codes.
        z = self.net(x)  # (batch, input_dim) -> (batch, latent_dim)
        return z  # (batch, latent_dim)

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim=32,  # Bottleneck code width.
        hidden_dim=256,  # Hidden reconstruction width.
        output_dim=784  # Flattened output width.
    ):
        super().__init__()

        # Register a feed-forward reconstruction network.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # Decode latent codes back into input-shaped reconstructions.
        reconstruction = self.net(z)  # (batch, latent_dim) -> (batch, output_dim)
        return reconstruction  # (batch, output_dim)

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Flattened input and reconstruction width.
        hidden_dim=256,  # Encoder/decoder hidden width.
        latent_dim=32  # Bottleneck code width.
    ):
        super().__init__()

        # Register mirrored encoder and decoder modules.
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def encode(self, x):
        z = self.encoder(x)  # (batch, input_dim) -> (batch, latent_dim)
        return z  # (batch, latent_dim)

    def decode(self, z):
        reconstruction = self.decoder(z)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction  # (batch, input_dim)

    def forward(self, x):
        # Encode through the bottleneck, then reconstruct the original input.
        z = self.encode(x)  # (batch, input_dim) -> (batch, latent_dim)
        reconstruction = self.decode(z)  # (batch, latent_dim) -> (batch, input_dim)
        return reconstruction, z  # (batch, input_dim), (batch, latent_dim)

# Train on a tiny synthetic reconstruction batch.
model = Autoencoder(input_dim=64, hidden_dim=24, latent_dim=8)
inputs = torch.zeros(2, 64)  # -> (2, 64)
inputs[0, 8:24] = 1.0  # (2, 64)
inputs[1, 40:56] = 1.0  # (2, 64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Fit the model to reconstruct its own inputs.
for step in range(3):
    optimizer.zero_grad()
    reconstruction, latent_codes = model(inputs)  # (2, 64) -> (2, 64), (2, 8)
    loss = F.mse_loss(reconstruction, inputs)  # (2, 64), (2, 64) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss and latent codes for inspection.
final_loss = loss.item()  # scalar
final_latent_codes = latent_codes.detach()  # (2, 8)
