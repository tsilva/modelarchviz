# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        width=512,  # Token embedding width.
        num_heads=8  # Number of attention heads.
    ):
        super().__init__()

        # Register explicit Q/K/V projections and output projection.
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x, mask=None):
        # Project tokens into query, key, and value tensors.
        batch_size = x.size(0)  # (batch, tokens, width) -> scalar
        token_count = x.size(1)  # (batch, tokens, width) -> scalar
        q = self.q_proj(x)  # (batch, tokens, width)
        k = self.k_proj(x)  # (batch, tokens, width)
        v = self.v_proj(x)  # (batch, tokens, width)

        # Split model width across attention heads.
        head_shape = (batch_size, token_count, self.num_heads, self.head_dim)
        q = q.view(head_shape)  # (batch, tokens, width) -> (batch, tokens, heads, head_dim)
        q = q.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        k = k.view(head_shape)  # (batch, tokens, width) -> (batch, tokens, heads, head_dim)
        k = k.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        v = v.view(head_shape)  # (batch, tokens, width) -> (batch, tokens, heads, head_dim)
        v = v.transpose(1, 2)  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)

        # Compute scaled dot-product attention, optionally with causal masking.
        key_transpose = k.transpose(-2, -1)  # (batch, heads, tokens, head_dim) -> (batch, heads, head_dim, tokens)
        scores = q @ key_transpose  # (batch, heads, tokens, tokens)
        scale = self.head_dim ** -0.5  # scalar
        scores = scores * scale  # (batch, heads, tokens, tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))  # (batch, heads, tokens, tokens)
        weights = torch.softmax(scores, dim=-1)  # (batch, heads, tokens, tokens)

        # Mix values and merge heads back to model width.
        context = weights @ v  # (batch, heads, tokens, head_dim)
        context = context.transpose(1, 2)  # (batch, heads, tokens, head_dim) -> (batch, tokens, heads, head_dim)
        context = context.contiguous()  # (batch, tokens, heads, head_dim)
        merged_shape = (batch_size, token_count, self.num_heads * self.head_dim)
        merged = context.view(merged_shape)  # (batch, tokens, heads, head_dim) -> (batch, tokens, width)
        out = self.out_proj(merged)  # (batch, tokens, width)
        return out  # (batch, tokens, width)


# %% [notebook-only]
# Create and run attention over four tokens: (2, 4, 32) -> (2, 4, 32).
example_attention = MultiHeadAttention(width=32, num_heads=4)
example_tokens = torch.randn(2, 4, 32)  # -> (2, 4, 32)
example_mask = torch.tril(torch.ones(4, 4)).view(1, 1, 4, 4)  # -> (1, 1, 4, 4)
example_attended = example_attention(example_tokens, example_mask)  # (2, 4, 32), (1, 1, 4, 4) -> (2, 4, 32)
print("attended shape:", example_attended.shape)

# %%
class TransformerBlock(nn.Module):
    def __init__(
        self,
        width=512,  # Token embedding width.
        num_heads=8,  # Number of attention heads.
        mlp_ratio=4  # Feed-forward expansion ratio.
    ):
        super().__init__()

        # Register pre-normalized attention and MLP sublayers.
        self.ln_1 = nn.LayerNorm(width)
        self.attn = MultiHeadAttention(width, num_heads)
        self.ln_2 = nn.LayerNorm(width)
        mlp_width = width * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, width),
        )

    def forward(self, x, mask=None):
        # Apply self-attention with a residual connection.
        attn_input = self.ln_1(x)  # (batch, tokens, width)
        attn_output = self.attn(attn_input, mask)  # (batch, tokens, width)
        x = x + attn_output  # (batch, tokens, width)

        # Apply MLP with a residual connection.
        mlp_input = self.ln_2(x)  # (batch, tokens, width)
        mlp_output = self.mlp(mlp_input)  # (batch, tokens, width)
        x = x + mlp_output  # (batch, tokens, width)
        return x  # (batch, tokens, width)


# %% [notebook-only]
# Create and run one transformer block: (2, 4, 32) -> (2, 4, 32).
example_block = TransformerBlock(width=32, num_heads=4)
example_tokens = torch.randn(2, 4, 32)  # -> (2, 4, 32)
example_outputs = example_block(example_tokens)  # (2, 4, 32) -> (2, 4, 32)
print("block output shape:", example_outputs.shape)

# %%
# @arch class-visionencoder-nn-module:start
class VisionEncoder(nn.Module):
# @arch class-visionencoder-nn-module:end
    def __init__(
        self,
        image_size=224,  # Square image size.
        patch_size=32,  # Square patch size.
        width=768,  # Vision transformer width.
        layers=12,  # Number of transformer blocks.
        heads=12,  # Number of attention heads.
        embed_dim=512  # Shared CLIP embedding width.
    ):
        super().__init__()

        # Register patch projection, learned tokens, transformer, and projection head.
        # @arch visionencoder.self-patch_embed-nn-convnd-n-width-kernel_size-patch_size-stride-patch_s:start
        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        # @arch visionencoder.self-patch_embed-nn-convnd-n-width-kernel_size-patch_size-stride-patch_s:end
        patch_grid = image_size // patch_size
        patch_count = patch_grid ** 2
        # @arch visionencoder.self-cls_token-nn-parameter-torch-zeros-n-n-width:start
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        # @arch visionencoder.self-cls_token-nn-parameter-torch-zeros-n-n-width:end
        # @arch visionencoder.self-pos_embed-nn-parameter-torch-zeros-n-patch_count-n-width:start
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, width))
        # @arch visionencoder.self-pos_embed-nn-parameter-torch-zeros-n-patch_count-n-width:end
        # @arch visionencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la:start
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(layers)])
        # @arch visionencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la:end
        self.ln_post = nn.LayerNorm(width)
        # @arch visionencoder.self-proj-nn-linear-width-embed_dim-bias-false:start
        self.proj = nn.Linear(width, embed_dim, bias=False)
        # @arch visionencoder.self-proj-nn-linear-width-embed_dim-bias-false:end

    # @arch visionencoder.def-forward-self-images:start
    def forward(self, images):
    # @arch visionencoder.def-forward-self-images:end
        # Convert images into patch tokens and prepend the CLS token.
        # @arch visionencoder.forward.x-self-patch_embed-images:start
        x = self.patch_embed(images)  # (batch, 3, height, width) -> (batch, vision_width, grid, grid)
        # @arch visionencoder.forward.x-self-patch_embed-images:end
        # @arch visionencoder.forward.x-x-flatten-n:start
        x = x.flatten(2)  # (batch, vision_width, grid, grid) -> (batch, vision_width, patches)
        # @arch visionencoder.forward.x-x-flatten-n:end
        # @arch visionencoder.forward.x-x-transpose-n-n:start
        x = x.transpose(1, 2)  # (batch, vision_width, patches) -> (batch, patches, vision_width)
        # @arch visionencoder.forward.x-x-transpose-n-n:end
        batch_size = x.size(0)  # (batch, patches, vision_width) -> scalar
        # @arch visionencoder.forward.cls-self-cls_token-expand-batch_size-n-n:start
        cls = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, vision_width) -> (batch, 1, vision_width)
        # @arch visionencoder.forward.cls-self-cls_token-expand-batch_size-n-n:end
        # @arch visionencoder.forward.x-torch-cat-cls-x-dim-n:start
        x = torch.cat([cls, x], dim=1)  # (batch, 1, vision_width), (batch, patches, vision_width) -> (batch, tokens, vision_width)
        # @arch visionencoder.forward.x-torch-cat-cls-x-dim-n:end

        # Add positions and run the visual transformer.
        # @arch visionencoder.forward.x-x-self-pos_embed:start
        x = x + self.pos_embed  # (batch, tokens, vision_width)
        # @arch visionencoder.forward.x-x-self-pos_embed:end
        # @arch visionencoder.forward.for-block-in-self-blocks:start
        for block in self.blocks:
        # @arch visionencoder.forward.for-block-in-self-blocks:end
            # @arch visionencoder.forward.x-block-x:start
            x = block(x)  # (batch, tokens, vision_width)
            # @arch visionencoder.forward.x-block-x:end

        # Project the CLS token into the shared embedding space.
        # @arch visionencoder.forward.x-self-ln_post-x:start
        x = self.ln_post(x)  # (batch, tokens, vision_width)
        # @arch visionencoder.forward.x-self-ln_post-x:end
        # @arch visionencoder.forward.cls_output-x-n:start
        cls_output = x[:, 0]  # (batch, tokens, vision_width) -> (batch, vision_width)
        # @arch visionencoder.forward.cls_output-x-n:end
        # @arch visionencoder.forward.image_features-self-proj-cls_output:start
        image_features = self.proj(cls_output)  # (batch, vision_width) -> (batch, embed_dim)
        # @arch visionencoder.forward.image_features-self-proj-cls_output:end
        return image_features  # (batch, embed_dim)


# %% [notebook-only]
# Create and run a small image encoder: (2, 3, 32, 32) -> (2, 32).
example_encoder = VisionEncoder(image_size=32, patch_size=16, width=32, layers=1, heads=4, embed_dim=32)
example_images = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
example_features = example_encoder(example_images)  # (2, 3, 32, 32) -> (2, 32)
print("image features shape:", example_features.shape)

# %%
# @arch class-textencoder-nn-module:start
class TextEncoder(nn.Module):
# @arch class-textencoder-nn-module:end
    def __init__(
        self,
        vocab_size=49408,  # CLIP byte-pair vocabulary size.
        context_length=77,  # Maximum text sequence length.
        width=512,  # Text transformer width.
        layers=12,  # Number of transformer blocks.
        heads=8,  # Number of attention heads.
        embed_dim=512  # Shared CLIP embedding width.
    ):
        super().__init__()

        # Register token/position embeddings, causal transformer, and projection head.
        # @arch textencoder.self-token_embedding-nn-embedding-vocab_size-width:start
        self.token_embedding = nn.Embedding(vocab_size, width)
        # @arch textencoder.self-token_embedding-nn-embedding-vocab_size-width:end
        # @arch textencoder.self-pos_embed-nn-parameter-torch-zeros-n-context_length-width:start
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, width))
        # @arch textencoder.self-pos_embed-nn-parameter-torch-zeros-n-context_length-width:end
        # @arch textencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la:start
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(layers)])
        # @arch textencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la:end
        self.ln_final = nn.LayerNorm(width)
        # @arch textencoder.self-text_projection-nn-linear-width-embed_dim-bias-false:start
        self.text_projection = nn.Linear(width, embed_dim, bias=False)
        # @arch textencoder.self-text_projection-nn-linear-width-embed_dim-bias-false:end
        # @arch textencoder.mask-torch-tril-torch-ones-context_length-context_length:start
        mask = torch.tril(torch.ones(context_length, context_length))
        # @arch textencoder.mask-torch-tril-torch-ones-context_length-context_length:end
        # @arch textencoder.self-register_buffer-causal_mask-mask-view-n-n-context_length-context_le:start
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length), persistent=False)
        # @arch textencoder.self-register_buffer-causal_mask-mask-view-n-n-context_length-context_le:end

    # @arch textencoder.def-forward-self-input_ids:start
    def forward(self, input_ids):
    # @arch textencoder.def-forward-self-input_ids:end
        # Embed text tokens and add learned positions.
        token_count = input_ids.size(1)  # (batch, tokens) -> scalar
        # @arch textencoder.forward.token_embeddings-self-token_embedding-input_ids:start
        token_embeddings = self.token_embedding(input_ids)  # (batch, tokens) -> (batch, tokens, text_width)
        # @arch textencoder.forward.token_embeddings-self-token_embedding-input_ids:end
        # @arch textencoder.forward.position_embeddings-self-pos_embed-token_count:start
        position_embeddings = self.pos_embed[:, :token_count, :]  # (1, context, text_width) -> (1, tokens, text_width)
        # @arch textencoder.forward.position_embeddings-self-pos_embed-token_count:end
        # @arch textencoder.forward.x-token_embeddings-position_embeddings:start
        x = token_embeddings + position_embeddings  # (batch, tokens, text_width)
        # @arch textencoder.forward.x-token_embeddings-position_embeddings:end

        # Run the text transformer with a causal mask.
        # @arch textencoder.forward.mask-self-causal_mask-token_count-token_count:start
        mask = self.causal_mask[:, :, :token_count, :token_count]  # (1, 1, context, context) -> (1, 1, tokens, tokens)
        # @arch textencoder.forward.mask-self-causal_mask-token_count-token_count:end
        # @arch textencoder.forward.for-block-in-self-blocks:start
        for block in self.blocks:
        # @arch textencoder.forward.for-block-in-self-blocks:end
            # @arch textencoder.forward.x-block-x-mask:start
            x = block(x, mask)  # (batch, tokens, text_width)
            # @arch textencoder.forward.x-block-x-mask:end

        # Pool at the end-of-text token and project to the shared embedding space.
        # @arch textencoder.forward.x-self-ln_final-x:start
        x = self.ln_final(x)  # (batch, tokens, text_width)
        # @arch textencoder.forward.x-self-ln_final-x:end
        # @arch textencoder.forward.eot_indices-input_ids-argmax-dim-n:start
        eot_indices = input_ids.argmax(dim=-1)  # (batch, tokens) -> (batch)
        # @arch textencoder.forward.eot_indices-input_ids-argmax-dim-n:end
        # @arch textencoder.forward.batch_indices-torch-arange-input_ids-size-n-device-input_ids-device:start
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)  # -> (batch)
        # @arch textencoder.forward.batch_indices-torch-arange-input_ids-size-n-device-input_ids-device:end
        # @arch textencoder.forward.pooled-x-batch_indices-eot_indices:start
        pooled = x[batch_indices, eot_indices]  # (batch, tokens, text_width) -> (batch, text_width)
        # @arch textencoder.forward.pooled-x-batch_indices-eot_indices:end
        # @arch textencoder.forward.text_features-self-text_projection-pooled:start
        text_features = self.text_projection(pooled)  # (batch, text_width) -> (batch, embed_dim)
        # @arch textencoder.forward.text_features-self-text_projection-pooled:end
        return text_features  # (batch, embed_dim)


# %% [notebook-only]
# Create and run a small text encoder: (2, 8) -> (2, 32).
example_encoder = TextEncoder(vocab_size=100, context_length=8, width=32, layers=1, heads=4, embed_dim=32)
example_input_ids = torch.tensor([[1, 5, 10, 99, 0, 0, 0, 0], [1, 8, 11, 12, 99, 0, 0, 0]])  # -> (2, 8)
example_features = example_encoder(example_input_ids)  # (2, 8) -> (2, 32)
print("text features shape:", example_features.shape)

# %%
class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim=512,  # Shared image-text embedding width.
        image_size=224,  # Square image size.
        patch_size=32,  # Vision patch size.
        vision_width=768,  # Vision transformer width.
        vision_layers=12,  # Vision transformer depth.
        vision_heads=12,  # Vision attention heads.
        vocab_size=49408,  # Text vocabulary size.
        context_length=77,  # Text context length.
        text_width=512,  # Text transformer width.
        text_layers=12,  # Text transformer depth.
        text_heads=8  # Text attention heads.
    ):
        super().__init__()

        # Register the two encoders and the learned contrastive temperature.
        # @arch clip.self-visual-visionencoder-image_size-patch_size-vision_width-vision_laye:start
        self.visual = VisionEncoder(image_size, patch_size, vision_width, vision_layers, vision_heads, embed_dim)
        # @arch clip.self-visual-visionencoder-image_size-patch_size-vision_width-vision_laye:end
        # @arch clip.self-text-textencoder-vocab_size-context_length-text_width-text_layers-t:start
        self.text = TextEncoder(vocab_size, context_length, text_width, text_layers, text_heads, embed_dim)
        # @arch clip.self-text-textencoder-vocab_size-context_length-text_width-text_layers-t:end
        # @arch clip.self-logit_scale-nn-parameter-torch-ones-torch-log-torch-tensor-n-n:start
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        # @arch clip.self-logit_scale-nn-parameter-torch-ones-torch-log-torch-tensor-n-n:end

    # @arch clip.def-forward-self-images-input_ids:start
    def forward(self, images, input_ids):
    # @arch clip.def-forward-self-images-input_ids:end
        # Encode each modality into the same embedding width.
        # @arch clip.forward.image_features-self-visual-images:start
        image_features = self.visual(images)  # (batch, 3, height, width) -> (batch, embed_dim)
        # @arch clip.forward.image_features-self-visual-images:end
        # @arch clip.forward.text_features-self-text-input_ids:start
        text_features = self.text(input_ids)  # (batch, tokens) -> (batch, embed_dim)
        # @arch clip.forward.text_features-self-text-input_ids:end

        # Normalize embeddings and score every image against every text.
        # @arch clip.forward.image_features-f-normalize-image_features-dim-n:start
        image_features = F.normalize(image_features, dim=-1)  # (batch, embed_dim)
        # @arch clip.forward.image_features-f-normalize-image_features-dim-n:end
        # @arch clip.forward.text_features-f-normalize-text_features-dim-n:start
        text_features = F.normalize(text_features, dim=-1)  # (batch, embed_dim)
        # @arch clip.forward.text_features-f-normalize-text_features-dim-n:end
        # @arch clip.forward.logit_scale-self-logit_scale-exp:start
        logit_scale = self.logit_scale.exp()  # scalar
        # @arch clip.forward.logit_scale-self-logit_scale-exp:end
        # @arch clip.forward.text_features_t-text_features-t:start
        text_features_t = text_features.t()  # (batch, embed_dim) -> (embed_dim, batch)
        # @arch clip.forward.text_features_t-text_features-t:end
        # @arch clip.forward.logits_per_image-logit_scale-image_features-text_features_t:start
        logits_per_image = logit_scale * image_features @ text_features_t  # (batch, embed_dim), (embed_dim, batch) -> (batch, batch)
        # @arch clip.forward.logits_per_image-logit_scale-image_features-text_features_t:end
        # @arch clip.forward.logits_per_text-logits_per_image-t:start
        logits_per_text = logits_per_image.t()  # (batch, batch)
        # @arch clip.forward.logits_per_text-logits_per_image-t:end
        return logits_per_image, logits_per_text  # two (batch, batch)


# %% [notebook-only]
# Create and run a small CLIP batch.
example_model = CLIP(
    embed_dim=32,
    image_size=32,
    patch_size=16,
    vision_width=32,
    vision_layers=1,
    vision_heads=4,
    vocab_size=100,
    context_length=8,
    text_width=32,
    text_layers=1,
    text_heads=4,
)
example_images = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
example_input_ids = torch.tensor([[1, 5, 10, 99, 0, 0, 0, 0], [1, 8, 11, 12, 99, 0, 0, 0]])  # -> (2, 8)
example_logits = example_model(example_images, example_input_ids)  # (2, 3, 32, 32), (2, 8) -> two (2, 2)
print("image logits shape:", example_logits[0].shape)

# %%
# Train on a tiny aligned image-text batch.
model = CLIP(
    embed_dim=32,
    image_size=32,
    patch_size=16,
    vision_width=32,
    vision_layers=1,
    vision_heads=4,
    vocab_size=100,
    context_length=8,
    text_width=32,
    text_layers=1,
    text_heads=4,
)
images = torch.zeros(2, 3, 32, 32)  # -> (2, 3, 32, 32)
images[0, :, 4:16, 4:16] = 1.0  # (2, 3, 32, 32)
images[1, :, 16:28, 16:28] = 1.0  # (2, 3, 32, 32)
input_ids = torch.tensor([[1, 5, 10, 99, 0, 0, 0, 0], [1, 8, 11, 12, 99, 0, 0, 0]])  # -> (2, 8)
targets = torch.arange(images.size(0))  # -> (2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit matching image-text pairs with symmetric contrastive loss.
for step in range(3):
    optimizer.zero_grad()
    logits_per_image, logits_per_text = model(images, input_ids)  # (2, 3, 32, 32), (2, 8) -> two (2, 2)
    image_loss = F.cross_entropy(logits_per_image, targets)  # (2, 2), (2) -> scalar
    text_loss = F.cross_entropy(logits_per_text, targets)  # (2, 2), (2) -> scalar
    loss = (image_loss + text_loss) / 2  # scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
