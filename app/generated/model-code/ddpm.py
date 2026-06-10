import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(
        self,
        width=128  # Timestep embedding width.
    ):
        super().__init__()

        # Register the MLP that maps sinusoidal features into conditioning vectors.
        self.width = width
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, timesteps):
        # Build sinusoidal timestep features.
        half_width = self.width // 2  # scalar
        device = timesteps.device
        frequencies = torch.arange(half_width, device=device, dtype=torch.float32)  # -> (half_width)
        frequencies = frequencies / max(half_width - 1, 1)  # (half_width)
        frequencies = torch.exp(-math.log(10000.0) * frequencies)  # (half_width)
        angles = timesteps.float()[:, None] * frequencies[None, :]  # (batch), (half_width) -> (batch, half_width)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (batch, half_width) -> (batch, width)
        embedding = self.mlp(embedding)  # (batch, width)
        return embedding  # (batch, width)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # Input feature channels.
        out_channels,  # Output feature channels.
        time_width=128,  # Timestep embedding width.
        groups=8  # Number of GroupNorm groups.
    ):
        super().__init__()

        # Register a time-conditioned residual convolution block.
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_width, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_emb):
        # Apply the first normalized convolution and inject timestep conditioning.
        h = self.norm1(x)  # (batch, in_channels, height, width)
        h = F.silu(h)  # (batch, in_channels, height, width)
        h = self.conv1(h)  # (batch, in_channels, height, width) -> (batch, out_channels, height, width)
        time_bias = self.time_proj(F.silu(time_emb))  # (batch, time_width) -> (batch, out_channels)
        time_bias = time_bias[:, :, None, None]  # (batch, out_channels) -> (batch, out_channels, 1, 1)
        h = h + time_bias  # (batch, out_channels, height, width)

        # Finish the residual path and add the channel-matched skip.
        h = self.norm2(h)  # (batch, out_channels, height, width)
        h = F.silu(h)  # (batch, out_channels, height, width)
        h = self.conv2(h)  # (batch, out_channels, height, width)
        skip = self.skip(x)  # (batch, in_channels, height, width) -> (batch, out_channels, height, width)
        out = h + skip  # (batch, out_channels, height, width)
        return out  # (batch, out_channels, height, width)

class Downsample(nn.Module):
    def __init__(
        self,
        channels  # Feature channels.
    ):
        super().__init__()

        # Register a stride-2 convolution for spatial downsampling.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)  # (batch, channels, height, width) -> (batch, channels, height/2, width/2)
        return out  # (batch, channels, height/2, width/2)

class Upsample(nn.Module):
    def __init__(
        self,
        channels  # Feature channels.
    ):
        super().__init__()

        # Register a transposed convolution for spatial upsampling.
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)  # (batch, channels, height, width) -> (batch, channels, height*2, width*2)
        return out  # (batch, channels, height*2, width*2)

class UNetDenoiser(nn.Module):
    def __init__(
        self,
        image_channels=3,  # Channels in x_t and predicted noise.
        base_channels=64,  # Base U-Net feature width.
        time_width=128  # Timestep embedding width.
    ):
        super().__init__()

        # Register timestep conditioning, encoder, bottleneck, decoder, and noise head.
        self.time_embedding = TimeEmbedding(time_width)
        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_width)
        self.downsample1 = Downsample(base_channels)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_width)
        self.downsample2 = Downsample(base_channels * 2)
        self.middle = ResidualBlock(base_channels * 2, base_channels * 4, time_width)
        self.upsample2 = Upsample(base_channels * 4)
        self.up2 = ResidualBlock(base_channels * 6, base_channels * 2, time_width)
        self.upsample1 = Upsample(base_channels * 2)
        self.up1 = ResidualBlock(base_channels * 3, base_channels, time_width)
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, noisy_images, timesteps):
        # Embed timesteps once and run the encoder path.
        time_emb = self.time_embedding(timesteps)  # (batch) -> (batch, time_width)
        x = self.input_conv(noisy_images)  # (batch, channels, height, width) -> (batch, base, height, width)
        skip1 = self.down1(x, time_emb)  # (batch, base, height, width)
        x = self.downsample1(skip1)  # (batch, base, height, width) -> (batch, base, height/2, width/2)
        skip2 = self.down2(x, time_emb)  # (batch, base, height/2, width/2) -> (batch, base*2, height/2, width/2)
        x = self.downsample2(skip2)  # (batch, base*2, height/2, width/2) -> (batch, base*2, height/4, width/4)

        # Denoise at the bottleneck and decode with U-Net skip concatenations.
        x = self.middle(x, time_emb)  # (batch, base*2, height/4, width/4) -> (batch, base*4, height/4, width/4)
        x = self.upsample2(x)  # (batch, base*4, height/4, width/4) -> (batch, base*4, height/2, width/2)
        x = torch.cat([x, skip2], dim=1)  # (batch, base*4, height/2, width/2) -> (batch, base*6, height/2, width/2)
        x = self.up2(x, time_emb)  # (batch, base*6, height/2, width/2) -> (batch, base*2, height/2, width/2)
        x = self.upsample1(x)  # (batch, base*2, height/2, width/2) -> (batch, base*2, height, width)
        x = torch.cat([x, skip1], dim=1)  # (batch, base*2, height, width) -> (batch, base*3, height, width)
        x = self.up1(x, time_emb)  # (batch, base*3, height, width) -> (batch, base, height, width)

        # Predict the Gaussian noise added to the clean image.
        x = self.out_norm(x)  # (batch, base, height, width)
        x = F.silu(x)  # (batch, base, height, width)
        predicted_noise = self.out_conv(x)  # (batch, base, height, width) -> (batch, channels, height, width)
        return predicted_noise  # (batch, channels, height, width)

class DDPM(nn.Module):
    def __init__(
        self,
        image_channels=3,  # Channels in x_0 and x_t.
        base_channels=64,  # Base U-Net feature width.
        time_width=128,  # Timestep embedding width.
        timesteps=1000,  # Number of diffusion steps.
        beta_start=1e-4,  # First beta in the noise schedule.
        beta_end=0.02  # Last beta in the noise schedule.
    ):
        super().__init__()

        # Register the denoiser and fixed forward-process schedule buffers.
        self.denoiser = UNetDenoiser(image_channels, base_channels, time_width)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)

    def _extract(self, values, timesteps, target_shape):
        # Gather per-example schedule values and reshape for image broadcasting.
        batch_size = timesteps.size(0)  # (batch) -> scalar
        gathered = values.gather(0, timesteps)  # (timesteps), (batch) -> (batch)
        broadcast_shape = (batch_size,) + (1,) * (len(target_shape) - 1)  # -> (batch, 1, 1, 1)
        gathered = gathered.reshape(broadcast_shape)  # (batch) -> (batch, 1, 1, 1)
        return gathered

    def q_sample(self, clean_images, timesteps, noise):
        # Add scheduled Gaussian noise to clean images in one closed-form step.
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_images.shape)  # (batch, 1, 1, 1)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_images.shape)  # (batch, 1, 1, 1)
        noisy_images = sqrt_alpha * clean_images + sqrt_one_minus_alpha * noise  # (batch, channels, height, width)
        return noisy_images  # (batch, channels, height, width)

    def forward(self, noisy_images, timesteps):
        predicted_noise = self.denoiser(noisy_images, timesteps)  # (batch, channels, height, width), (batch) -> (batch, channels, height, width)
        return predicted_noise  # (batch, channels, height, width)

    def p_mean_variance(self, noisy_images, timesteps):
        # Convert predicted noise into the reverse-process Gaussian mean.
        predicted_noise = self(noisy_images, timesteps)  # (batch, channels, height, width)
        betas_t = self._extract(self.betas, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        model_mean = sqrt_recip_alpha * (noisy_images - betas_t * predicted_noise / sqrt_one_minus_alpha)  # (batch, channels, height, width)
        variance = self._extract(self.posterior_variance, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        return model_mean, variance  # (batch, channels, height, width), (batch, 1, 1, 1)

    def p_sample(self, noisy_images, timesteps, noise):
        # Sample one reverse diffusion step.
        model_mean, variance = self.p_mean_variance(noisy_images, timesteps)  # (batch, channels, height, width), (batch, 1, 1, 1)
        nonzero_mask = (timesteps != 0).float()[:, None, None, None]  # (batch) -> (batch, 1, 1, 1)
        sample = model_mean + nonzero_mask * torch.sqrt(variance) * noise  # (batch, channels, height, width)
        return sample  # (batch, channels, height, width)

# Train on a tiny synthetic denoising batch.
model = DDPM(image_channels=3, base_channels=16, time_width=32, timesteps=50)
clean_images = torch.zeros(2, 3, 32, 32)  # -> (2, 3, 32, 32)
clean_images[0, :, 4:16, 4:16] = 1.0  # (2, 3, 32, 32)
clean_images[1, :, 16:28, 16:28] = 1.0  # (2, 3, 32, 32)
timesteps = torch.tensor([10, 25])  # -> (2)
noise = torch.randn_like(clean_images)  # -> (2, 3, 32, 32)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model to predict the exact noise added by q_sample.
for step in range(3):
    optimizer.zero_grad()
    noisy_images = model.q_sample(clean_images, timesteps, noise)  # (2, 3, 32, 32)
    predicted_noise = model(noisy_images, timesteps)  # (2, 3, 32, 32), (2) -> (2, 3, 32, 32)
    loss = F.mse_loss(predicted_noise, noise)  # (2, 3, 32, 32), (2, 3, 32, 32) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
