# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# @arch class-generator-nn-module:start
class Generator(nn.Module):
# @arch class-generator-nn-module:end
    def __init__(
        self,
        latent_dim=100,  # Noise vector width.
        image_dim=784,  # Flattened image size.
        hidden_dim=256  # Hidden layer width.
    ):
        super().__init__()

        # Register an MLP that maps latent noise into image space.
        # @arch generator.self-net-nn-sequential:start
        self.net = nn.Sequential(
        # @arch generator.self-net-nn-sequential:end
            # @arch generator.nn-linear-latent_dim-hidden_dim:start
            nn.Linear(latent_dim, hidden_dim),
            # @arch generator.nn-linear-latent_dim-hidden_dim:end
            # @arch generator.nn-leakyrelu-n:start
            nn.LeakyReLU(0.2),
            # @arch generator.nn-leakyrelu-n:end
            # @arch generator.nn-linear-hidden_dim-hidden_dim:start
            nn.Linear(hidden_dim, hidden_dim),
            # @arch generator.nn-linear-hidden_dim-hidden_dim:end
            # @arch generator.nn-leakyrelu-n.2:start
            nn.LeakyReLU(0.2),
            # @arch generator.nn-leakyrelu-n.2:end
            # @arch generator.nn-linear-hidden_dim-image_dim:start
            nn.Linear(hidden_dim, image_dim),
            # @arch generator.nn-linear-hidden_dim-image_dim:end
            # @arch generator.nn-tanh:start
            nn.Tanh(),
            # @arch generator.nn-tanh:end
        )

    # @arch generator.def-forward-self-z:start
    def forward(self, z):
    # @arch generator.def-forward-self-z:end
        # Transform latent vectors into synthetic images.
        # @arch generator.forward.fake_images-self-net-z:start
        fake_images = self.net(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch generator.forward.fake_images-self-net-z:end
        return fake_images  # (batch, image_dim)


# %% [notebook-only]
# Create and run the generator: (2, 16) -> (2, 32).
example_generator = Generator(latent_dim=16, image_dim=32, hidden_dim=24)
example_latents = torch.randn(2, 16)  # -> (2, 16)
example_fake_images = example_generator(example_latents)  # (2, 16) -> (2, 32)
print("fake images shape:", example_fake_images.shape)

# %%
# @arch class-discriminator-nn-module:start
class Discriminator(nn.Module):
# @arch class-discriminator-nn-module:end
    def __init__(
        self,
        image_dim=784,  # Flattened image size.
        hidden_dim=256  # Hidden layer width.
    ):
        super().__init__()

        # Register an MLP that scores real and generated images.
        # @arch discriminator.self-net-nn-sequential:start
        self.net = nn.Sequential(
        # @arch discriminator.self-net-nn-sequential:end
            # @arch discriminator.nn-linear-image_dim-hidden_dim:start
            nn.Linear(image_dim, hidden_dim),
            # @arch discriminator.nn-linear-image_dim-hidden_dim:end
            # @arch discriminator.nn-leakyrelu-n:start
            nn.LeakyReLU(0.2),
            # @arch discriminator.nn-leakyrelu-n:end
            # @arch discriminator.nn-linear-hidden_dim-hidden_dim:start
            nn.Linear(hidden_dim, hidden_dim),
            # @arch discriminator.nn-linear-hidden_dim-hidden_dim:end
            # @arch discriminator.nn-leakyrelu-n.2:start
            nn.LeakyReLU(0.2),
            # @arch discriminator.nn-leakyrelu-n.2:end
            # @arch discriminator.nn-linear-hidden_dim-n:start
            nn.Linear(hidden_dim, 1),
            # @arch discriminator.nn-linear-hidden_dim-n:end
        )

    # @arch discriminator.def-forward-self-images:start
    def forward(self, images):
    # @arch discriminator.def-forward-self-images:end
        # Convert images into real/fake logits.
        # @arch discriminator.forward.logits-self-net-images:start
        logits = self.net(images)  # (batch, image_dim) -> (batch, 1)
        # @arch discriminator.forward.logits-self-net-images:end
        # @arch discriminator.forward.logits-logits-squeeze-n:start
        logits = logits.squeeze(-1)  # (batch, 1) -> (batch)
        # @arch discriminator.forward.logits-logits-squeeze-n:end
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
        # @arch gan.self-generator-generator-latent_dim-image_dim-hidden_dim:start
        self.generator = Generator(latent_dim, image_dim, hidden_dim)
        # @arch gan.self-generator-generator-latent_dim-image_dim-hidden_dim:end
        # @arch gan.self-discriminator-discriminator-image_dim-hidden_dim:start
        self.discriminator = Discriminator(image_dim, hidden_dim)
        # @arch gan.self-discriminator-discriminator-image_dim-hidden_dim:end

    # @arch gan.def-generate-self-z:start
    def generate(self, z):
    # @arch gan.def-generate-self-z:end
        # @arch gan.generate.fake_images-self-generator-z:start
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch gan.generate.fake_images-self-generator-z:end
        return fake_images  # (batch, image_dim)

    # @arch gan.def-discriminate-self-images:start
    def discriminate(self, images):
    # @arch gan.def-discriminate-self-images:end
        # @arch gan.discriminate.logits-self-discriminator-images:start
        logits = self.discriminator(images)  # (batch, image_dim) -> (batch)
        # @arch gan.discriminate.logits-self-discriminator-images:end
        return logits  # (batch)

    # @arch gan.def-discriminator_loss-self-real_images-z:start
    def discriminator_loss(self, real_images, z):
    # @arch gan.def-discriminator_loss-self-real_images-z:end
        # Train D to classify real images as 1 and generated images as 0.
        # @arch gan.discriminator_loss.fake_images-self-generator-z:start
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch gan.discriminator_loss.fake_images-self-generator-z:end
        # @arch gan.discriminator_loss.real_logits-self-discriminator-real_images:start
        real_logits = self.discriminator(real_images)  # (batch, image_dim) -> (batch)
        # @arch gan.discriminator_loss.real_logits-self-discriminator-real_images:end
        # @arch gan.discriminator_loss.fake_logits-self-discriminator-fake_images-detach:start
        fake_logits = self.discriminator(fake_images.detach())  # (batch, image_dim) -> (batch)
        # @arch gan.discriminator_loss.fake_logits-self-discriminator-fake_images-detach:end
        # @arch gan.discriminator_loss.real_targets-torch-ones_like-real_logits:start
        real_targets = torch.ones_like(real_logits)  # -> (batch)
        # @arch gan.discriminator_loss.real_targets-torch-ones_like-real_logits:end
        # @arch gan.discriminator_loss.fake_targets-torch-zeros_like-fake_logits:start
        fake_targets = torch.zeros_like(fake_logits)  # -> (batch)
        # @arch gan.discriminator_loss.fake_targets-torch-zeros_like-fake_logits:end
        # @arch gan.discriminator_loss.real_loss-f-binary_cross_entropy_with_logits-real_logits-real_targets:start
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)  # (batch), (batch) -> scalar
        # @arch gan.discriminator_loss.real_loss-f-binary_cross_entropy_with_logits-real_logits-real_targets:end
        # @arch gan.discriminator_loss.fake_loss-f-binary_cross_entropy_with_logits-fake_logits-fake_targets:start
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)  # (batch), (batch) -> scalar
        # @arch gan.discriminator_loss.fake_loss-f-binary_cross_entropy_with_logits-fake_logits-fake_targets:end
        # @arch gan.discriminator_loss.loss-real_loss-fake_loss:start
        loss = real_loss + fake_loss  # scalar
        # @arch gan.discriminator_loss.loss-real_loss-fake_loss:end
        return loss  # scalar

    # @arch gan.def-generator_loss-self-z:start
    def generator_loss(self, z):
    # @arch gan.def-generator_loss-self-z:end
        # Train G to make D classify generated images as real.
        # @arch gan.generator_loss.fake_images-self-generator-z:start
        fake_images = self.generator(z)  # (batch, latent_dim) -> (batch, image_dim)
        # @arch gan.generator_loss.fake_images-self-generator-z:end
        # @arch gan.generator_loss.fake_logits-self-discriminator-fake_images:start
        fake_logits = self.discriminator(fake_images)  # (batch, image_dim) -> (batch)
        # @arch gan.generator_loss.fake_logits-self-discriminator-fake_images:end
        # @arch gan.generator_loss.real_targets-torch-ones_like-fake_logits:start
        real_targets = torch.ones_like(fake_logits)  # -> (batch)
        # @arch gan.generator_loss.real_targets-torch-ones_like-fake_logits:end
        # @arch gan.generator_loss.loss-f-binary_cross_entropy_with_logits-fake_logits-real_targets:start
        loss = F.binary_cross_entropy_with_logits(fake_logits, real_targets)  # (batch), (batch) -> scalar
        # @arch gan.generator_loss.loss-f-binary_cross_entropy_with_logits-fake_logits-real_targets:end
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
# @arch real_images-torch-zeros-n-n:start
real_images = torch.zeros(2, 32)  # -> (2, 32)
# @arch real_images-torch-zeros-n-n:end
real_images[0, 4:12] = 1.0  # (2, 32)
real_images[1, 20:28] = 1.0  # (2, 32)
# @arch z-torch-randn-n-n:start
z = torch.randn(2, 16)  # -> (2, 16)
# @arch z-torch-randn-n-n:end
# @arch generator_optimizer-torch-optim-sgd-model-generator-parameters-lr-n:start
generator_optimizer = torch.optim.SGD(model.generator.parameters(), lr=0.01)
# @arch generator_optimizer-torch-optim-sgd-model-generator-parameters-lr-n:end
# @arch discriminator_optimizer-torch-optim-sgd-model-discriminator-parameters-l:start
discriminator_optimizer = torch.optim.SGD(model.discriminator.parameters(), lr=0.01)
# @arch discriminator_optimizer-torch-optim-sgd-model-discriminator-parameters-l:end

# Alternate discriminator and generator updates.
# @arch for-step-in-range-n:start
for step in range(3):
# @arch for-step-in-range-n:end
    # @arch discriminator_optimizer-zero_grad:start
    discriminator_optimizer.zero_grad()
    # @arch discriminator_optimizer-zero_grad:end
    # @arch d_loss-model-discriminator_loss-real_images-z:start
    d_loss = model.discriminator_loss(real_images, z)  # (2, 32), (2, 16) -> scalar
    # @arch d_loss-model-discriminator_loss-real_images-z:end
    # @arch d_loss-backward:start
    d_loss.backward()
    # @arch d_loss-backward:end
    # @arch discriminator_optimizer-step:start
    discriminator_optimizer.step()
    # @arch discriminator_optimizer-step:end

    # @arch generator_optimizer-zero_grad:start
    generator_optimizer.zero_grad()
    # @arch generator_optimizer-zero_grad:end
    # @arch g_loss-model-generator_loss-z:start
    g_loss = model.generator_loss(z)  # (2, 16) -> scalar
    # @arch g_loss-model-generator_loss-z:end
    # @arch g_loss-backward:start
    g_loss.backward()
    # @arch g_loss-backward:end
    # @arch generator_optimizer-step:start
    generator_optimizer.step()
    # @arch generator_optimizer-step:end

# Keep the final scalar losses for inspection.
final_discriminator_loss = d_loss.item()  # scalar
final_generator_loss = g_loss.item()  # scalar
