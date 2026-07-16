# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=100,  # Noise vector width.
        image_dim=784,  # Flattened image size.
        hidden_dim=256  # Hidden layer width.
    ):
        super().__init__()

        # Register an MLP that maps latent noise into image space.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        # Transform latent vectors into synthetic images.
        fake_images = self.net(z)  # (batch, latent_dim) -> (batch, image_dim)
        return fake_images  # (batch, image_dim)


# %% [notebook-only]
# Create and run the generator: (2, 16) -> (2, 32).
example_generator = Generator(latent_dim=16, image_dim=32, hidden_dim=24)
example_latents = torch.randn(2, 16)  # -> (2, 16)
example_fake_images = example_generator(example_latents)  # (2, 16) -> (2, 32)
print("fake images shape:", example_fake_images.shape)

# %%
class Discriminator(nn.Module):
    def __init__(
        self,
        image_dim=784,  # Flattened image size.
        hidden_dim=256  # Hidden layer width.
    ):
        super().__init__()

        # Register an MLP that scores real and generated images.
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images):
        # Convert images into real/fake logits.
        logits = self.net(images)  # (batch, image_dim) -> (batch, 1)
        logits = logits.squeeze(-1)  # (batch, 1) -> (batch)
        return logits  # (batch)


# %% [notebook-only]
# Create and run the discriminator: (2, 32) -> (2).
example_discriminator = Discriminator(image_dim=32, hidden_dim=24)
example_images = torch.randn(2, 32)  # -> (2, 32)
example_logits = example_discriminator(example_images)  # (2, 32) -> (2)
print("discriminator logits shape:", example_logits.shape)

# %%
class GAN(nn.Module):
    def __init__(
        self,
        latent_dim=100,  # Noise vector width.
        image_dim=784,  # Flattened image size.
        hidden_dim=256  # Shared hidden width for both networks.
    ):
        super().__init__()

        # Register the competing generator and discriminator.
        self.generator = Generator(latent_dim, image_dim, hidden_dim)
        self.discriminator = Discriminator(image_dim, hidden_dim)

    def generate(self, z):
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        return fake_images  # (batch, image_dim)

    def discriminate(self, images):
        logits = self.discriminator(images)  # (batch, image_dim) -> (batch)
        return logits  # (batch)

    def discriminator_loss(self, real_images, z):
        # Train D to classify real images as 1 and generated images as 0.
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        real_logits = self.discriminator(real_images)  # (batch, image_dim) -> (batch)
        fake_logits = self.discriminator(fake_images.detach())  # (batch, image_dim) -> (batch)
        real_targets = torch.ones_like(real_logits)  # -> (batch)
        fake_targets = torch.zeros_like(fake_logits)  # -> (batch)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)  # (batch), (batch) -> scalar
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)  # (batch), (batch) -> scalar
        loss = real_loss + fake_loss  # scalar
        return loss  # scalar

    def generator_loss(self, z):
        # Train G to make D classify generated images as real.
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        fake_logits = self.discriminator(fake_images)  # (batch, image_dim) -> (batch)
        real_targets = torch.ones_like(fake_logits)  # -> (batch)
        loss = F.binary_cross_entropy_with_logits(fake_logits, real_targets)  # (batch), (batch) -> scalar
        return loss  # scalar


# %% [notebook-only]
# Create and run a small GAN forward path.
example_model = GAN(latent_dim=16, image_dim=32, hidden_dim=24)
example_z = torch.randn(2, 16)  # -> (2, 16)
example_real_images = torch.randn(2, 32)  # -> (2, 32)
example_fake_images = example_model.generate(example_z)  # (2, 16) -> (2, 32)
example_real_logits = example_model.discriminate(example_real_images)  # (2, 32) -> (2)
example_fake_logits = example_model.discriminate(example_fake_images)  # (2, 32) -> (2)
print("fake images shape:", example_fake_images.shape, "fake logits shape:", example_fake_logits.shape)

# %%
# Train on a tiny synthetic real-image batch.
model = GAN(latent_dim=16, image_dim=32, hidden_dim=24)
real_images = torch.zeros(2, 32)  # -> (2, 32)
real_images[0, 4:12] = 1.0  # (2, 32)
real_images[1, 20:28] = 1.0  # (2, 32)
z = torch.randn(2, 16)  # -> (2, 16)
generator_optimizer = torch.optim.SGD(model.generator.parameters(), lr=0.01)
discriminator_optimizer = torch.optim.SGD(model.discriminator.parameters(), lr=0.01)

# Alternate discriminator and generator updates.
for step in range(3):
    discriminator_optimizer.zero_grad()
    d_loss = model.discriminator_loss(real_images, z)  # (2, 32), (2, 16) -> scalar
    d_loss.backward()
    discriminator_optimizer.step()

    generator_optimizer.zero_grad()
    g_loss = model.generator_loss(z)  # (2, 16) -> scalar
    g_loss.backward()
    generator_optimizer.step()

# Keep the final scalar losses for inspection.
final_discriminator_loss = d_loss.item()  # scalar
final_generator_loss = g_loss.item()  # scalar
