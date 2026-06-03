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
        x = self.proj(x)  # (batch, channels, height, width) -> (batch, embed_dim, grid, grid)

        # Flatten patches into a token sequence: (batch, embed_dim, grid, grid) -> (batch, patches, embed_dim).
        x = x.flatten(2)  # (batch, embed_dim, grid, grid) -> (batch, embed_dim, patches)
        x = x.transpose(1, 2)  # (batch, embed_dim, patches) -> (batch, patches, embed_dim)
        return x  # (batch, patches, embed_dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,  # Token embedding width.
        num_heads=12  # Number of attention heads.
    ):
        super().__init__()

        # Register explicit Q/K/V projections and the output projection.
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Project tokens into per-head query, key, and value tensors.
        batch_size = x.size(0)  # (batch, tokens, embed_dim) -> scalar
        tokens = x.size(1)  # (batch, tokens, embed_dim) -> scalar
        q = self.q_proj(x)  # (batch, tokens, embed_dim)
        k = self.k_proj(x)  # (batch, tokens, embed_dim)
        v = self.v_proj(x)  # (batch, tokens, embed_dim)

        # Split model width across heads: (batch, tokens, embed_dim) -> (batch, heads, tokens, head_dim).
        q = q.view(batch_size, tokens, self.num_heads, self.head_dim)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        q = q.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        k = k.view(batch_size, tokens, self.num_heads, self.head_dim)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        k = k.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        v = v.view(batch_size, tokens, self.num_heads, self.head_dim)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        v = v.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)

        # Compute scaled dot-product attention over all image tokens.
        key_transpose = k.transpose(-2, -1)  # (batch, heads, tokens, head_dim) -> (batch, heads, head_dim, tokens)
        scores = q @ key_transpose  # (batch, heads, tokens, head_dim), (batch, heads, head_dim, tokens) -> (batch, heads, tokens, tokens)
        scale = self.head_dim ** -0.5  # scalar
        attn_scores = scores * scale  # (batch, heads, tokens, tokens)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, heads, tokens, tokens)

        # Mix values, merge heads, and project back to embedding width.
        context = attn_weights @ v  # (batch, heads, tokens, tokens), (batch, heads, tokens, head_dim) -> (batch, heads, tokens, head_dim)
        context = context.transpose(1, 2)  # (batch, heads, tokens, head_dim) -> (batch, tokens, heads, head_dim)
        context = context.contiguous()  # (batch, tokens, heads, head_dim)
        merged = context.view(batch_size, tokens, self.num_heads * self.head_dim)  # (batch, tokens, heads, head_dim) -> (batch, tokens, embed_dim)
        out = self.out_proj(merged)  # (batch, tokens, embed_dim)
        return out  # (batch, tokens, embed_dim)


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
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        # Apply self-attention with a residual connection.
        attn_input = self.norm1(x)  # (batch, tokens, embed_dim)
        attn_output = self.attn(attn_input)  # (batch, tokens, embed_dim)
        x = x + attn_output  # (batch, tokens, embed_dim)

        # Apply MLP with a residual connection.
        mlp_input = self.norm2(x)  # (batch, tokens, embed_dim)
        mlp_output = self.mlp(mlp_input)  # (batch, tokens, embed_dim)
        x = x + mlp_output  # (batch, tokens, embed_dim)
        return x  # (batch, tokens, embed_dim)


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
        x = self.patch_embed(x)  # (batch, channels, height, width) -> (batch, patches, embed_dim)
        batch_size = x.size(0)  # (batch, patches, embed_dim) -> scalar
        cls = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, embed_dim) -> (batch, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)  # (batch, 1, embed_dim), (batch, patches, embed_dim) -> (batch, tokens, embed_dim)

        # Add learned positions and run the encoder stack.
        x = x + self.pos_embed  # (batch, tokens, embed_dim)
        for block in self.blocks:
            x = block(x)  # (batch, tokens, embed_dim)

        # Normalize CLS output and project to class logits.
        x = self.norm(x)  # (batch, tokens, embed_dim)
        cls_output = x[:, 0]  # (batch, tokens, embed_dim) -> (batch, embed_dim)
        logits = self.head(cls_output)  # (batch, embed_dim) -> (batch, num_classes)
        return logits  # (batch, num_classes)


# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
model = VisionTransformer(num_classes=1000)
test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
logits = model(test_input)  # (2, 3, 224, 224) -> (2, 1000)


# Train on a tiny synthetic image batch.
model = VisionTransformer(num_classes=2, embed_dim=48, depth=1, num_heads=4)
train_images = torch.zeros(2, 3, 224, 224)  # -> (2, 3, 224, 224)
train_images[0, :, 32:96, 32:96] = 1.0  # (2, 3, 224, 224)
train_images[1, :, 128:192, 128:192] = 1.0  # (2, 3, 224, 224)
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 224, 224) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
