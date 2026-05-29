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


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels=3,  # Number of image channels.
        embed_dim=768,  # Patch embedding width.
        patch_size=16,  # Square patch size.
        image_size=224  # Square image size.
    ):
        super().__init__()

        # Register the strided patch projection.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        patches_per_side = image_size // patch_size
        self.num_patches = patches_per_side ** 2

    def forward(self, x):
        # Project image patches: (batch, channels, height, width) -> (batch, embed_dim, grid, grid).
        x = self.proj(x)

        # Flatten patches into a token sequence: (batch, embed_dim, grid, grid) -> (batch, patches, embed_dim).
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim=768,  # Token embedding width.
        num_heads=12,  # Number of attention heads.
        mlp_dim=3072  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register pre-normalized attention and MLP sublayers.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        # Apply self-attention with a residual connection.
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + attn_output

        # Apply MLP with a residual connection.
        mlp_input = self.norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes=1000,  # Number of output classes.
        embed_dim=768,  # Token embedding width.
        depth=12,  # Number of encoder blocks.
        num_heads=12  # Number of attention heads.
    ):
        super().__init__()

        # Register patch embedding, learned tokens, encoder stack, and classifier head.
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Convert image patches into tokens and prepend CLS token.
        x = self.patch_embed(x)
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add learned positions and run the encoder stack.
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)

        # Normalize CLS output and project to class logits.
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits


# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
model = VisionTransformer(num_classes=1000)
test_input = torch.randn(2, 3, 224, 224)
logits = model(test_input)

# logits: (2, 1000)
