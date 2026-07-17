# %%
import torch
import torch.nn as nn


# %%
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
        # @arch vit.patch_embed.proj:start
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # @arch vit.patch_embed.proj:end
        patches_per_side = image_size // patch_size
        # @arch patchembed.self-num_patches-patches_per_side-n:start
        self.num_patches = patches_per_side ** 2
        # @arch patchembed.self-num_patches-patches_per_side-n:end

    def forward(self, x):
        # Project image patches: (batch, channels, height, width) -> (batch, embed_dim, grid, grid).
        # @arch vit.patch_embed.project:start
        x = self.proj(x)  # (batch, channels, height, width) -> (batch, embed_dim, grid, grid)
        # @arch vit.patch_embed.project:end

        # Flatten patches into a token sequence: (batch, embed_dim, grid, grid) -> (batch, patches, embed_dim).
        x = x.flatten(2)  # (batch, embed_dim, grid, grid) -> (batch, embed_dim, patches)
        # @arch patchembed.forward.x-x-transpose-n-n:start
        x = x.transpose(1, 2)  # (batch, embed_dim, patches) -> (batch, patches, embed_dim)
        # @arch patchembed.forward.x-x-transpose-n-n:end
        # @arch patchembed.forward.return-x:start
        return x  # (batch, patches, embed_dim)
        # @arch patchembed.forward.return-x:end


# %% [notebook-only]
# Create and run patch embedding: (2, 3, 32, 32) -> (2, 4, 24).
patch_embed = PatchEmbed(in_channels=3, embed_dim=24, patch_size=16, image_size=32)
images = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
patch_tokens = patch_embed(images)  # (2, 3, 32, 32) -> (2, 4, 24)
print("patch_tokens shape:", patch_tokens.shape)

# %%
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        # @arch multiheadselfattention.self:start
        self,
        # @arch multiheadselfattention.self:end
        embed_dim=768,  # Token embedding width.
        num_heads=12  # Number of attention heads.
    ):
        super().__init__()

        # Register explicit Q/K/V projections and the output projection.
        self.num_heads = num_heads
        # @arch multiheadselfattention.self-head_dim-embed_dim-num_heads:start
        self.head_dim = embed_dim // num_heads
        # @arch multiheadselfattention.self-head_dim-embed_dim-num_heads:end
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        # @arch multiheadselfattention.self-v_proj-nn-linear-embed_dim-embed_dim:start
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # @arch multiheadselfattention.self-v_proj-nn-linear-embed_dim-embed_dim:end
        # @arch multiheadselfattention.self-out_proj-nn-linear-embed_dim-embed_dim:start
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # @arch multiheadselfattention.self-out_proj-nn-linear-embed_dim-embed_dim:end

    # @arch multiheadselfattention.def-forward-self-x:start
    def forward(self, x):
    # @arch multiheadselfattention.def-forward-self-x:end
        # Project tokens into per-head query, key, and value tensors.
        # @arch multiheadselfattention.forward.batch_size-x-size-n:start
        batch_size = x.size(0)  # (batch, tokens, embed_dim) -> scalar
        # @arch multiheadselfattention.forward.batch_size-x-size-n:end
        tokens = x.size(1)  # (batch, tokens, embed_dim) -> scalar
        q = self.q_proj(x)  # (batch, tokens, embed_dim)
        k = self.k_proj(x)  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.forward.v-self-v_proj-x:start
        v = self.v_proj(x)  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.forward.v-self-v_proj-x:end

        # Split model width across heads: (batch, tokens, embed_dim) -> (batch, heads, tokens, head_dim).
        q = q.view(batch_size, tokens, self.num_heads, self.head_dim)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        q = q.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        # @arch multiheadselfattention.forward.k-k-view-batch_size-tokens-self-num_heads-self-head_dim:start
        k = k.view(batch_size, tokens, self.num_heads, self.head_dim)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.forward.k-k-view-batch_size-tokens-self-num_heads-self-head_dim:end
        # @arch multiheadselfattention.forward.k-k-transpose-n-n:start
        k = k.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        # @arch multiheadselfattention.forward.k-k-transpose-n-n:end
        # @arch multiheadselfattention.forward.v-v-view-batch_size-tokens-self-num_heads-self-head_dim:start
        v = v.view(batch_size, tokens, self.num_heads, self.head_dim)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.forward.v-v-view-batch_size-tokens-self-num_heads-self-head_dim:end
        # @arch multiheadselfattention.forward.v-v-transpose-n-n:start
        v = v.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        # @arch multiheadselfattention.forward.v-v-transpose-n-n:end

        # Compute scaled dot-product attention over all image tokens.
        key_transpose = k.transpose(-2, -1)  # (batch, heads, tokens, head_dim) -> (batch, heads, head_dim, tokens)
        scores = q @ key_transpose  # (batch, heads, tokens, head_dim), (batch, heads, head_dim, tokens) -> (batch, heads, tokens, tokens)
        # @arch multiheadselfattention.forward.scale-self-head_dim-n:start
        scale = self.head_dim ** -0.5  # scalar
        # @arch multiheadselfattention.forward.scale-self-head_dim-n:end
        # @arch multiheadselfattention.forward.attn_scores-scores-scale:start
        attn_scores = scores * scale  # (batch, heads, tokens, tokens)
        # @arch multiheadselfattention.forward.attn_scores-scores-scale:end
        # @arch multiheadselfattention.forward.attn_weights-torch-softmax-attn_scores-dim-n:start
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, heads, tokens, tokens)
        # @arch multiheadselfattention.forward.attn_weights-torch-softmax-attn_scores-dim-n:end

        # Mix values, merge heads, and project back to embedding width.
        context = attn_weights @ v  # (batch, heads, tokens, tokens), (batch, heads, tokens, head_dim) -> (batch, heads, tokens, head_dim)
        context = context.transpose(1, 2)  # (batch, heads, tokens, head_dim) -> (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.forward.context-context-contiguous:start
        context = context.contiguous()  # (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.forward.context-context-contiguous:end
        # @arch multiheadselfattention.forward.merged-context-view-batch_size-tokens-self-num_heads-self-head_dim:start
        merged = context.view(batch_size, tokens, self.num_heads * self.head_dim)  # (batch, tokens, heads, head_dim) -> (batch, tokens, embed_dim)
        # @arch multiheadselfattention.forward.merged-context-view-batch_size-tokens-self-num_heads-self-head_dim:end
        out = self.out_proj(merged)  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.forward.return-out:start
        return out  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.forward.return-out:end


# %% [notebook-only]
# Create and run vision self-attention: (2, 5, 24) -> (2, 5, 24).
example_attention = MultiHeadSelfAttention(embed_dim=24, num_heads=4)
tokens = torch.randn(2, 5, 24)  # -> (2, 5, 24)
example_attended = example_attention(tokens)  # (2, 5, 24) -> (2, 5, 24)
print("attended shape:", example_attended.shape)

# %%
# @arch class-encoderblock-nn-module:start
class EncoderBlock(nn.Module):
# @arch class-encoderblock-nn-module:end
    def __init__(
        self,
        # @arch encoderblock.embed_dim-n:start
        embed_dim=768,  # Token embedding width.
        # @arch encoderblock.embed_dim-n:end
        num_heads=12,  # Number of attention heads.
        mlp_dim=3072  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register pre-normalized attention and MLP sublayers.
        # @arch vit.encoder.ln1:start
        self.norm1 = nn.LayerNorm(embed_dim)
        # @arch vit.encoder.ln1:end
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        # @arch encoderblock.self-normn-nn-layernorm-embed_dim.2:start
        self.norm2 = nn.LayerNorm(embed_dim)
        # @arch encoderblock.self-normn-nn-layernorm-embed_dim.2:end
        # @arch encoderblock.self-mlp-nn-sequential:start
        self.mlp = nn.Sequential(
        # @arch encoderblock.self-mlp-nn-sequential:end
            # @arch encoderblock.nn-linear-embed_dim-mlp_dim:start
            nn.Linear(embed_dim, mlp_dim),
            # @arch encoderblock.nn-linear-embed_dim-mlp_dim:end
            # @arch encoderblock.nn-gelu:start
            nn.GELU(),
            # @arch encoderblock.nn-gelu:end
            # @arch encoderblock.nn-linear-mlp_dim-embed_dim:start
            nn.Linear(mlp_dim, embed_dim),
            # @arch encoderblock.nn-linear-mlp_dim-embed_dim:end
        # @arch encoderblock.code.4:start
        )
        # @arch encoderblock.code.4:end

    # @arch encoderblock.def-forward-self-x:start
    def forward(self, x):
    # @arch encoderblock.def-forward-self-x:end
        # Apply self-attention with a residual connection.
        # @arch vit.encoder.ln1_call:start
        attn_input = self.norm1(x)  # (batch, tokens, embed_dim)
        # @arch vit.encoder.ln1_call:end
        # @arch encoderblock.forward.attn_output-self-attn-attn_input:start
        attn_output = self.attn(attn_input)  # (batch, tokens, embed_dim)
        # @arch encoderblock.forward.attn_output-self-attn-attn_input:end
        # @arch vit.encoder.resid1:start
        x = x + attn_output  # (batch, tokens, embed_dim)
        # @arch vit.encoder.resid1:end

        # Apply MLP with a residual connection.
        # @arch encoderblock.forward.mlp_input-self-normn-x:start
        mlp_input = self.norm2(x)  # (batch, tokens, embed_dim)
        # @arch encoderblock.forward.mlp_input-self-normn-x:end
        # @arch encoderblock.forward.mlp_output-self-mlp-mlp_input:start
        mlp_output = self.mlp(mlp_input)  # (batch, tokens, embed_dim)
        # @arch encoderblock.forward.mlp_output-self-mlp-mlp_input:end
        # @arch vit.encoder.resid2:start
        x = x + mlp_output  # (batch, tokens, embed_dim)
        # @arch vit.encoder.resid2:end
        # @arch encoderblock.forward.return-x:start
        return x  # (batch, tokens, embed_dim)
        # @arch encoderblock.forward.return-x:end


# %% [notebook-only]
# Create and run one ViT encoder block: (2, 5, 24) -> (2, 5, 24).
example_block = EncoderBlock(embed_dim=24, num_heads=4, mlp_dim=48)
tokens = torch.randn(2, 5, 24)  # -> (2, 5, 24)
encoded_tokens = example_block(tokens)  # (2, 5, 24) -> (2, 5, 24)
print("encoded_tokens shape:", encoded_tokens.shape)

# %%
# @arch class-visiontransformer-nn-module:start
class VisionTransformer(nn.Module):
# @arch class-visiontransformer-nn-module:end
    # @arch visiontransformer.def-__init__:start
    def __init__(
    # @arch visiontransformer.def-__init__:end
        self,
        num_classes=1000,  # Number of output classes.
        embed_dim=768,  # Token embedding width.
        depth=12,  # Number of encoder blocks.
        num_heads=12  # Number of attention heads.
    ):
        super().__init__()

        # Register patch embedding, learned tokens, encoder stack, and classifier head.
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        # @arch vit.tokens.cls:start
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # @arch vit.tokens.cls:end
        # @arch vit.tokens.position:start
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        # @arch vit.tokens.position:end
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads) for _ in range(depth)])
        # @arch visiontransformer.self-norm-nn-layernorm-embed_dim:start
        self.norm = nn.LayerNorm(embed_dim)
        # @arch visiontransformer.self-norm-nn-layernorm-embed_dim:end
        # @arch visiontransformer.self-head-nn-linear-embed_dim-num_classes:start
        self.head = nn.Linear(embed_dim, num_classes)
        # @arch visiontransformer.self-head-nn-linear-embed_dim-num_classes:end

    # @arch visiontransformer.def-forward-self-x:start
    def forward(self, x):
    # @arch visiontransformer.def-forward-self-x:end
        # Convert image patches into tokens and prepend CLS token.
        # @arch visiontransformer.forward.x-self-patch_embed-x:start
        x = self.patch_embed(x)  # (batch, channels, height, width) -> (batch, patches, embed_dim)
        # @arch visiontransformer.forward.x-self-patch_embed-x:end
        batch_size = x.size(0)  # (batch, patches, embed_dim) -> scalar
        cls = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, embed_dim) -> (batch, 1, embed_dim)
        # @arch vit.tokens.concat:start
        x = torch.cat([cls, x], dim=1)  # (batch, 1, embed_dim), (batch, patches, embed_dim) -> (batch, tokens, embed_dim)
        # @arch vit.tokens.concat:end

        # Add learned positions and run the encoder stack.
        # @arch visiontransformer.forward.x-x-self-pos_embed:start
        x = x + self.pos_embed  # (batch, tokens, embed_dim)
        # @arch visiontransformer.forward.x-x-self-pos_embed:end
        for block in self.blocks:
            x = block(x)  # (batch, tokens, embed_dim)

        # Normalize CLS output and project to class logits.
        # @arch visiontransformer.forward.x-self-norm-x:start
        x = self.norm(x)  # (batch, tokens, embed_dim)
        # @arch visiontransformer.forward.x-self-norm-x:end
        # @arch visiontransformer.forward.cls_output-x-n:start
        cls_output = x[:, 0]  # (batch, tokens, embed_dim) -> (batch, embed_dim)
        # @arch visiontransformer.forward.cls_output-x-n:end
        # @arch vit.head:start
        logits = self.head(cls_output)  # (batch, embed_dim) -> (batch, num_classes)
        # @arch vit.head:end
        return logits  # (batch, num_classes)


# %% [notebook-only]
# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
example_model = VisionTransformer(num_classes=1000)
example_test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
example_logits = example_model(example_test_input)  # (2, 3, 224, 224) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
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
