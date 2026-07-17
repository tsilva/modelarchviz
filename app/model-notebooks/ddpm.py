# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# @arch class-timeembedding-nn-module:start
class TimeEmbedding(nn.Module):
# @arch class-timeembedding-nn-module:end
    def __init__(
        self,
        width=128  # Timestep embedding width.
    ):
        super().__init__()

        # Register the MLP that maps sinusoidal features into conditioning vectors.
        self.width = width
        # @arch timeembedding.self-mlp-nn-sequential:start
        self.mlp = nn.Sequential(
        # @arch timeembedding.self-mlp-nn-sequential:end
            # @arch timeembedding.nn-linear-width-width-n:start
            nn.Linear(width, width * 4),
            # @arch timeembedding.nn-linear-width-width-n:end
            # @arch timeembedding.nn-silu:start
            nn.SiLU(),
            # @arch timeembedding.nn-silu:end
            # @arch timeembedding.nn-linear-width-n-width:start
            nn.Linear(width * 4, width),
            # @arch timeembedding.nn-linear-width-n-width:end
        )

    # @arch timeembedding.def-forward-self-timesteps:start
    def forward(self, timesteps):
    # @arch timeembedding.def-forward-self-timesteps:end
        # Build sinusoidal timestep features.
        # @arch timeembedding.forward.half_width-self-width-n:start
        half_width = self.width // 2  # scalar
        # @arch timeembedding.forward.half_width-self-width-n:end
        device = timesteps.device
        # @arch timeembedding.forward.frequencies-torch-arange-half_width-device-device-dtype-torch-floatn:start
        frequencies = torch.arange(half_width, device=device, dtype=torch.float32)  # -> (half_width)
        # @arch timeembedding.forward.frequencies-torch-arange-half_width-device-device-dtype-torch-floatn:end
        frequencies = frequencies / max(half_width - 1, 1)  # (half_width)
        # @arch timeembedding.forward.frequencies-torch-exp-math-log-n-frequencies:start
        frequencies = torch.exp(-math.log(10000.0) * frequencies)  # (half_width)
        # @arch timeembedding.forward.frequencies-torch-exp-math-log-n-frequencies:end
        # @arch timeembedding.forward.angles-timesteps-float-none-frequencies-none:start
        angles = timesteps.float()[:, None] * frequencies[None, :]  # (batch), (half_width) -> (batch, half_width)
        # @arch timeembedding.forward.angles-timesteps-float-none-frequencies-none:end
        # @arch timeembedding.forward.embedding-torch-cat-angles-sin-angles-cos-dim-n:start
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (batch, half_width) -> (batch, width)
        # @arch timeembedding.forward.embedding-torch-cat-angles-sin-angles-cos-dim-n:end
        # @arch timeembedding.forward.embedding-self-mlp-embedding:start
        embedding = self.mlp(embedding)  # (batch, width)
        # @arch timeembedding.forward.embedding-self-mlp-embedding:end
        return embedding  # (batch, width)


# %% [notebook-only]
# Create and run timestep embeddings: (4) -> (4, 32).
example_embedding = TimeEmbedding(width=32)
example_timesteps = torch.tensor([0, 10, 100, 999])  # -> (4)
example_outputs = example_embedding(example_timesteps)  # (4) -> (4, 32)
print("time embedding shape:", example_outputs.shape)

# %%
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


# %% [notebook-only]
# Create and run one residual block: (2, 16, 32, 32), (2, 32) -> (2, 32, 32, 32).
example_block = ResidualBlock(in_channels=16, out_channels=32, time_width=32)
example_features = torch.randn(2, 16, 32, 32)  # -> (2, 16, 32, 32)
example_time = torch.randn(2, 32)  # -> (2, 32)
example_outputs = example_block(example_features, example_time)  # (2, 16, 32, 32), (2, 32) -> (2, 32, 32, 32)
print("residual output shape:", example_outputs.shape)

# %%
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


# %% [notebook-only]
# Create and run downsampling: (2, 32, 32, 32) -> (2, 32, 16, 16).
example_downsample = Downsample(channels=32)
example_features = torch.randn(2, 32, 32, 32)  # -> (2, 32, 32, 32)
example_outputs = example_downsample(example_features)  # (2, 32, 32, 32) -> (2, 32, 16, 16)
print("downsampled shape:", example_outputs.shape)

# %%
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


# %% [notebook-only]
# Create and run upsampling: (2, 32, 16, 16) -> (2, 32, 32, 32).
example_upsample = Upsample(channels=32)
example_features = torch.randn(2, 32, 16, 16)  # -> (2, 32, 16, 16)
example_outputs = example_upsample(example_features)  # (2, 32, 16, 16) -> (2, 32, 32, 32)
print("upsampled shape:", example_outputs.shape)

# %%
# @arch class-unetdenoiser-nn-module:start
class UNetDenoiser(nn.Module):
# @arch class-unetdenoiser-nn-module:end
    def __init__(
        self,
        image_channels=3,  # Channels in x_t and predicted noise.
        base_channels=64,  # Base U-Net feature width.
        time_width=128  # Timestep embedding width.
    ):
        super().__init__()

        # Register timestep conditioning, encoder, bottleneck, decoder, and noise head.
        # @arch unetdenoiser.self-time_embedding-timeembedding-time_width:start
        self.time_embedding = TimeEmbedding(time_width)
        # @arch unetdenoiser.self-time_embedding-timeembedding-time_width:end
        # @arch unetdenoiser.self-input_conv-nn-convnd-image_channels-base_channels-kernel_size-n-pad:start
        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        # @arch unetdenoiser.self-input_conv-nn-convnd-image_channels-base_channels-kernel_size-n-pad:end
        # @arch unetdenoiser.self-downn-residualblock-base_channels-base_channels-time_width:start
        self.down1 = ResidualBlock(base_channels, base_channels, time_width)
        # @arch unetdenoiser.self-downn-residualblock-base_channels-base_channels-time_width:end
        # @arch unetdenoiser.self-downsamplen-downsample-base_channels:start
        self.downsample1 = Downsample(base_channels)
        # @arch unetdenoiser.self-downsamplen-downsample-base_channels:end
        # @arch unetdenoiser.self-downn-residualblock-base_channels-base_channels-n-time_width:start
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_width)
        # @arch unetdenoiser.self-downn-residualblock-base_channels-base_channels-n-time_width:end
        # @arch unetdenoiser.self-downsamplen-downsample-base_channels-n:start
        self.downsample2 = Downsample(base_channels * 2)
        # @arch unetdenoiser.self-downsamplen-downsample-base_channels-n:end
        # @arch unetdenoiser.self-middle-residualblock-base_channels-n-base_channels-n-time_width:start
        self.middle = ResidualBlock(base_channels * 2, base_channels * 4, time_width)
        # @arch unetdenoiser.self-middle-residualblock-base_channels-n-base_channels-n-time_width:end
        # @arch unetdenoiser.self-upsamplen-upsample-base_channels-n:start
        self.upsample2 = Upsample(base_channels * 4)
        # @arch unetdenoiser.self-upsamplen-upsample-base_channels-n:end
        # @arch unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-n-time_width:start
        self.up2 = ResidualBlock(base_channels * 6, base_channels * 2, time_width)
        # @arch unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-n-time_width:end
        # @arch unetdenoiser.self-upsamplen-upsample-base_channels-n.2:start
        self.upsample1 = Upsample(base_channels * 2)
        # @arch unetdenoiser.self-upsamplen-upsample-base_channels-n.2:end
        # @arch unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-time_width:start
        self.up1 = ResidualBlock(base_channels * 3, base_channels, time_width)
        # @arch unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-time_width:end
        # @arch unetdenoiser.self-out_norm-nn-groupnorm-n-base_channels:start
        self.out_norm = nn.GroupNorm(8, base_channels)
        # @arch unetdenoiser.self-out_norm-nn-groupnorm-n-base_channels:end
        # @arch unetdenoiser.self-out_conv-nn-convnd-base_channels-image_channels-kernel_size-n-paddi:start
        self.out_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)
        # @arch unetdenoiser.self-out_conv-nn-convnd-base_channels-image_channels-kernel_size-n-paddi:end

    # @arch unetdenoiser.def-forward-self-noisy_images-timesteps:start
    def forward(self, noisy_images, timesteps):
    # @arch unetdenoiser.def-forward-self-noisy_images-timesteps:end
        # Embed timesteps once and run the encoder path.
        # @arch unetdenoiser.forward.time_emb-self-time_embedding-timesteps:start
        time_emb = self.time_embedding(timesteps)  # (batch) -> (batch, time_width)
        # @arch unetdenoiser.forward.time_emb-self-time_embedding-timesteps:end
        # @arch unetdenoiser.forward.x-self-input_conv-noisy_images:start
        x = self.input_conv(noisy_images)  # (batch, channels, height, width) -> (batch, base, height, width)
        # @arch unetdenoiser.forward.x-self-input_conv-noisy_images:end
        # @arch unetdenoiser.forward.skipn-self-downn-x-time_emb:start
        skip1 = self.down1(x, time_emb)  # (batch, base, height, width)
        # @arch unetdenoiser.forward.skipn-self-downn-x-time_emb:end
        # @arch unetdenoiser.forward.x-self-downsamplen-skipn:start
        x = self.downsample1(skip1)  # (batch, base, height, width) -> (batch, base, height/2, width/2)
        # @arch unetdenoiser.forward.x-self-downsamplen-skipn:end
        # @arch unetdenoiser.forward.skipn-self-downn-x-time_emb.2:start
        skip2 = self.down2(x, time_emb)  # (batch, base, height/2, width/2) -> (batch, base*2, height/2, width/2)
        # @arch unetdenoiser.forward.skipn-self-downn-x-time_emb.2:end
        # @arch unetdenoiser.forward.x-self-downsamplen-skipn.2:start
        x = self.downsample2(skip2)  # (batch, base*2, height/2, width/2) -> (batch, base*2, height/4, width/4)
        # @arch unetdenoiser.forward.x-self-downsamplen-skipn.2:end

        # Denoise at the bottleneck and decode with U-Net skip concatenations.
        # @arch unetdenoiser.forward.x-self-middle-x-time_emb:start
        x = self.middle(x, time_emb)  # (batch, base*2, height/4, width/4) -> (batch, base*4, height/4, width/4)
        # @arch unetdenoiser.forward.x-self-middle-x-time_emb:end
        # @arch unetdenoiser.forward.x-self-upsamplen-x:start
        x = self.upsample2(x)  # (batch, base*4, height/4, width/4) -> (batch, base*4, height/2, width/2)
        # @arch unetdenoiser.forward.x-self-upsamplen-x:end
        # @arch unetdenoiser.forward.x-torch-cat-x-skipn-dim-n:start
        x = torch.cat([x, skip2], dim=1)  # (batch, base*4, height/2, width/2) -> (batch, base*6, height/2, width/2)
        # @arch unetdenoiser.forward.x-torch-cat-x-skipn-dim-n:end
        # @arch unetdenoiser.forward.x-self-upn-x-time_emb:start
        x = self.up2(x, time_emb)  # (batch, base*6, height/2, width/2) -> (batch, base*2, height/2, width/2)
        # @arch unetdenoiser.forward.x-self-upn-x-time_emb:end
        # @arch unetdenoiser.forward.x-self-upsamplen-x.2:start
        x = self.upsample1(x)  # (batch, base*2, height/2, width/2) -> (batch, base*2, height, width)
        # @arch unetdenoiser.forward.x-self-upsamplen-x.2:end
        # @arch unetdenoiser.forward.x-torch-cat-x-skipn-dim-n.2:start
        x = torch.cat([x, skip1], dim=1)  # (batch, base*2, height, width) -> (batch, base*3, height, width)
        # @arch unetdenoiser.forward.x-torch-cat-x-skipn-dim-n.2:end
        # @arch unetdenoiser.forward.x-self-upn-x-time_emb.2:start
        x = self.up1(x, time_emb)  # (batch, base*3, height, width) -> (batch, base, height, width)
        # @arch unetdenoiser.forward.x-self-upn-x-time_emb.2:end

        # Predict the Gaussian noise added to the clean image.
        # @arch unetdenoiser.forward.x-self-out_norm-x:start
        x = self.out_norm(x)  # (batch, base, height, width)
        # @arch unetdenoiser.forward.x-self-out_norm-x:end
        # @arch unetdenoiser.forward.x-f-silu-x:start
        x = F.silu(x)  # (batch, base, height, width)
        # @arch unetdenoiser.forward.x-f-silu-x:end
        # @arch unetdenoiser.forward.predicted_noise-self-out_conv-x:start
        predicted_noise = self.out_conv(x)  # (batch, base, height, width) -> (batch, channels, height, width)
        # @arch unetdenoiser.forward.predicted_noise-self-out_conv-x:end
        return predicted_noise  # (batch, channels, height, width)


# %% [notebook-only]
# Create and run a small denoising U-Net: (2, 3, 32, 32), (2) -> (2, 3, 32, 32).
example_model = UNetDenoiser(image_channels=3, base_channels=16, time_width=32)
example_noisy = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
example_timesteps = torch.tensor([5, 25])  # -> (2)
example_outputs = example_model(example_noisy, example_timesteps)  # (2, 3, 32, 32), (2) -> (2, 3, 32, 32)
print("predicted noise shape:", example_outputs.shape)

# %%
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
        # @arch ddpm.self-denoiser-unetdenoiser-image_channels-base_channels-time_width:start
        self.denoiser = UNetDenoiser(image_channels, base_channels, time_width)
        # @arch ddpm.self-denoiser-unetdenoiser-image_channels-base_channels-time_width:end
        # @arch ddpm.betas-torch-linspace-beta_start-beta_end-timesteps:start
        betas = torch.linspace(beta_start, beta_end, timesteps)
        # @arch ddpm.betas-torch-linspace-beta_start-beta_end-timesteps:end
        # @arch ddpm.alphas-n-betas:start
        alphas = 1.0 - betas
        # @arch ddpm.alphas-n-betas:end
        # @arch ddpm.alphas_cumprod-torch-cumprod-alphas-dim-n:start
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # @arch ddpm.alphas_cumprod-torch-cumprod-alphas-dim-n:end
        # @arch ddpm.alphas_cumprod_prev-torch-cat-torch-ones-n-alphas_cumprod-n:start
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        # @arch ddpm.alphas_cumprod_prev-torch-cat-torch-ones-n-alphas_cumprod-n:end
        # @arch ddpm.posterior_variance-betas-n-alphas_cumprod_prev-n-alphas_cumprod:start
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # @arch ddpm.posterior_variance-betas-n-alphas_cumprod_prev-n-alphas_cumprod:end
        # @arch ddpm.self-register_buffer-betas-betas:start
        self.register_buffer("betas", betas)
        # @arch ddpm.self-register_buffer-betas-betas:end
        # @arch ddpm.self-register_buffer-sqrt_alphas_cumprod-torch-sqrt-alphas_cumprod:start
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        # @arch ddpm.self-register_buffer-sqrt_alphas_cumprod-torch-sqrt-alphas_cumprod:end
        # @arch ddpm.self-register_buffer-sqrt_one_minus_alphas_cumprod-torch-sqrt-n-alphas_c:start
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        # @arch ddpm.self-register_buffer-sqrt_one_minus_alphas_cumprod-torch-sqrt-n-alphas_c:end
        # @arch ddpm.self-register_buffer-sqrt_recip_alphas-torch-sqrt-n-alphas:start
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        # @arch ddpm.self-register_buffer-sqrt_recip_alphas-torch-sqrt-n-alphas:end
        # @arch ddpm.self-register_buffer-posterior_variance-posterior_variance:start
        self.register_buffer("posterior_variance", posterior_variance)
        # @arch ddpm.self-register_buffer-posterior_variance-posterior_variance:end

    # @arch ddpm.def-_extract-self-values-timesteps-target_shape:start
    def _extract(self, values, timesteps, target_shape):
    # @arch ddpm.def-_extract-self-values-timesteps-target_shape:end
        # Gather per-example schedule values and reshape for image broadcasting.
        # @arch ddpm._extract.batch_size-timesteps-size-n:start
        batch_size = timesteps.size(0)  # (batch) -> scalar
        # @arch ddpm._extract.batch_size-timesteps-size-n:end
        # @arch ddpm._extract.gathered-values-gather-n-timesteps:start
        gathered = values.gather(0, timesteps)  # (timesteps), (batch) -> (batch)
        # @arch ddpm._extract.gathered-values-gather-n-timesteps:end
        # @arch ddpm._extract.broadcast_shape-batch_size-n-len-target_shape-n:start
        broadcast_shape = (batch_size,) + (1,) * (len(target_shape) - 1)  # -> (batch, 1, 1, 1)
        # @arch ddpm._extract.broadcast_shape-batch_size-n-len-target_shape-n:end
        # @arch ddpm._extract.gathered-gathered-reshape-broadcast_shape:start
        gathered = gathered.reshape(broadcast_shape)  # (batch) -> (batch, 1, 1, 1)
        # @arch ddpm._extract.gathered-gathered-reshape-broadcast_shape:end
        return gathered

    # @arch ddpm.def-q_sample-self-clean_images-timesteps-noise:start
    def q_sample(self, clean_images, timesteps, noise):
    # @arch ddpm.def-q_sample-self-clean_images-timesteps-noise:end
        # Add scheduled Gaussian noise to clean images in one closed-form step.
        # @arch ddpm.q_sample.sqrt_alpha-self-_extract-self-sqrt_alphas_cumprod-timesteps-clean_images:start
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_images.shape)  # (batch, 1, 1, 1)
        # @arch ddpm.q_sample.sqrt_alpha-self-_extract-self-sqrt_alphas_cumprod-timesteps-clean_images:end
        # @arch ddpm.q_sample.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti:start
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_images.shape)  # (batch, 1, 1, 1)
        # @arch ddpm.q_sample.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti:end
        # @arch ddpm.q_sample.noisy_images-sqrt_alpha-clean_images-sqrt_one_minus_alpha-noise:start
        noisy_images = sqrt_alpha * clean_images + sqrt_one_minus_alpha * noise  # (batch, channels, height, width)
        # @arch ddpm.q_sample.noisy_images-sqrt_alpha-clean_images-sqrt_one_minus_alpha-noise:end
        # @arch ddpm.q_sample.return-noisy_images:start
        return noisy_images  # (batch, channels, height, width)
        # @arch ddpm.q_sample.return-noisy_images:end

    # @arch ddpm.def-forward-self-noisy_images-timesteps:start
    def forward(self, noisy_images, timesteps):
    # @arch ddpm.def-forward-self-noisy_images-timesteps:end
        # @arch ddpm.forward.predicted_noise-self-denoiser-noisy_images-timesteps:start
        predicted_noise = self.denoiser(noisy_images, timesteps)  # (batch, channels, height, width), (batch) -> (batch, channels, height, width)
        # @arch ddpm.forward.predicted_noise-self-denoiser-noisy_images-timesteps:end
        # @arch ddpm.forward.return-predicted_noise:start
        return predicted_noise  # (batch, channels, height, width)
        # @arch ddpm.forward.return-predicted_noise:end

    # @arch ddpm.def-p_mean_variance-self-noisy_images-timesteps:start
    def p_mean_variance(self, noisy_images, timesteps):
    # @arch ddpm.def-p_mean_variance-self-noisy_images-timesteps:end
        # Convert predicted noise into the reverse-process Gaussian mean.
        # @arch ddpm.p_mean_variance.predicted_noise-self-noisy_images-timesteps:start
        predicted_noise = self(noisy_images, timesteps)  # (batch, channels, height, width)
        # @arch ddpm.p_mean_variance.predicted_noise-self-noisy_images-timesteps:end
        # @arch ddpm.p_mean_variance.betas_t-self-_extract-self-betas-timesteps-noisy_images-shape:start
        betas_t = self._extract(self.betas, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        # @arch ddpm.p_mean_variance.betas_t-self-_extract-self-betas-timesteps-noisy_images-shape:end
        # @arch ddpm.p_mean_variance.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti:start
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        # @arch ddpm.p_mean_variance.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti:end
        # @arch ddpm.p_mean_variance.sqrt_recip_alpha-self-_extract-self-sqrt_recip_alphas-timesteps-noisy_im:start
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        # @arch ddpm.p_mean_variance.sqrt_recip_alpha-self-_extract-self-sqrt_recip_alphas-timesteps-noisy_im:end
        # @arch ddpm.p_mean_variance.model_mean-sqrt_recip_alpha-noisy_images-betas_t-predicted_noise-sqrt_on:start
        model_mean = sqrt_recip_alpha * (noisy_images - betas_t * predicted_noise / sqrt_one_minus_alpha)  # (batch, channels, height, width)
        # @arch ddpm.p_mean_variance.model_mean-sqrt_recip_alpha-noisy_images-betas_t-predicted_noise-sqrt_on:end
        # @arch ddpm.p_mean_variance.variance-self-_extract-self-posterior_variance-timesteps-noisy_images-sh:start
        variance = self._extract(self.posterior_variance, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        # @arch ddpm.p_mean_variance.variance-self-_extract-self-posterior_variance-timesteps-noisy_images-sh:end
        return model_mean, variance  # (batch, channels, height, width), (batch, 1, 1, 1)

    # @arch ddpm.def-p_sample-self-noisy_images-timesteps-noise:start
    def p_sample(self, noisy_images, timesteps, noise):
    # @arch ddpm.def-p_sample-self-noisy_images-timesteps-noise:end
        # Sample one reverse diffusion step.
        # @arch ddpm.p_sample.model_mean-variance-self-p_mean_variance-noisy_images-timesteps:start
        model_mean, variance = self.p_mean_variance(noisy_images, timesteps)  # (batch, channels, height, width), (batch, 1, 1, 1)
        # @arch ddpm.p_sample.model_mean-variance-self-p_mean_variance-noisy_images-timesteps:end
        # @arch ddpm.p_sample.nonzero_mask-timesteps-n-float-none-none-none:start
        nonzero_mask = (timesteps != 0).float()[:, None, None, None]  # (batch) -> (batch, 1, 1, 1)
        # @arch ddpm.p_sample.nonzero_mask-timesteps-n-float-none-none-none:end
        # @arch ddpm.p_sample.sample-model_mean-nonzero_mask-torch-sqrt-variance-noise:start
        sample = model_mean + nonzero_mask * torch.sqrt(variance) * noise  # (batch, channels, height, width)
        # @arch ddpm.p_sample.sample-model_mean-nonzero_mask-torch-sqrt-variance-noise:end
        return sample  # (batch, channels, height, width)


# %% [notebook-only]
# Create and run a small DDPM forward and reverse step.
example_model = DDPM(image_channels=3, base_channels=16, time_width=32, timesteps=50)
example_clean = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
example_noise = torch.randn_like(example_clean)  # -> (2, 3, 32, 32)
example_timesteps = torch.tensor([10, 25])  # -> (2)
example_noisy = example_model.q_sample(example_clean, example_timesteps, example_noise)  # (2, 3, 32, 32)
example_predicted_noise = example_model(example_noisy, example_timesteps)  # (2, 3, 32, 32), (2) -> (2, 3, 32, 32)
example_previous = example_model.p_sample(example_noisy, example_timesteps, example_noise)  # (2, 3, 32, 32)
print("predicted noise shape:", example_predicted_noise.shape, "sample shape:", example_previous.shape)

# %%
# Train on a tiny synthetic denoising batch.
model = DDPM(image_channels=3, base_channels=16, time_width=32, timesteps=50)
# @arch clean_images-torch-zeros-n-n-n-n:start
clean_images = torch.zeros(2, 3, 32, 32)  # -> (2, 3, 32, 32)
# @arch clean_images-torch-zeros-n-n-n-n:end
clean_images[0, :, 4:16, 4:16] = 1.0  # (2, 3, 32, 32)
clean_images[1, :, 16:28, 16:28] = 1.0  # (2, 3, 32, 32)
# @arch timesteps-torch-tensor-n-n:start
timesteps = torch.tensor([10, 25])  # -> (2)
# @arch timesteps-torch-tensor-n-n:end
# @arch noise-torch-randn_like-clean_images:start
noise = torch.randn_like(clean_images)  # -> (2, 3, 32, 32)
# @arch noise-torch-randn_like-clean_images:end
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model to predict the exact noise added by q_sample.
for step in range(3):
    optimizer.zero_grad()
    # @arch noisy_images-model-q_sample-clean_images-timesteps-noise:start
    noisy_images = model.q_sample(clean_images, timesteps, noise)  # (2, 3, 32, 32)
    # @arch noisy_images-model-q_sample-clean_images-timesteps-noise:end
    # @arch predicted_noise-model-noisy_images-timesteps:start
    predicted_noise = model(noisy_images, timesteps)  # (2, 3, 32, 32), (2) -> (2, 3, 32, 32)
    # @arch predicted_noise-model-noisy_images-timesteps:end
    # @arch loss-f-mse_loss-predicted_noise-noise:start
    loss = F.mse_loss(predicted_noise, noise)  # (2, 3, 32, 32), (2, 3, 32, 32) -> scalar
    # @arch loss-f-mse_loss-predicted_noise-noise:end
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
