import torch
import torch.nn as nn
import torch.nn.functional as F

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

class VisionEncoder(nn.Module):
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
        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        patch_grid = image_size // patch_size
        patch_count = patch_grid ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim, bias=False)

    def forward(self, images):
        # Convert images into patch tokens and prepend the CLS token.
        x = self.patch_embed(images)  # (batch, 3, height, width) -> (batch, vision_width, grid, grid)
        x = x.flatten(2)  # (batch, vision_width, grid, grid) -> (batch, vision_width, patches)
        x = x.transpose(1, 2)  # (batch, vision_width, patches) -> (batch, patches, vision_width)
        batch_size = x.size(0)  # (batch, patches, vision_width) -> scalar
        cls = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, vision_width) -> (batch, 1, vision_width)
        x = torch.cat([cls, x], dim=1)  # (batch, 1, vision_width), (batch, patches, vision_width) -> (batch, tokens, vision_width)

        # Add positions and run the visual transformer.
        x = x + self.pos_embed  # (batch, tokens, vision_width)
        for block in self.blocks:
            x = block(x)  # (batch, tokens, vision_width)

        # Project the CLS token into the shared embedding space.
        x = self.ln_post(x)  # (batch, tokens, vision_width)
        cls_output = x[:, 0]  # (batch, tokens, vision_width) -> (batch, vision_width)
        image_features = self.proj(cls_output)  # (batch, vision_width) -> (batch, embed_dim)
        return image_features  # (batch, embed_dim)

class TextEncoder(nn.Module):
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
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(layers)])
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Linear(width, embed_dim, bias=False)
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length), persistent=False)

    def forward(self, input_ids):
        # Embed text tokens and add learned positions.
        token_count = input_ids.size(1)  # (batch, tokens) -> scalar
        token_embeddings = self.token_embedding(input_ids)  # (batch, tokens) -> (batch, tokens, text_width)
        position_embeddings = self.pos_embed[:, :token_count, :]  # (1, context, text_width) -> (1, tokens, text_width)
        x = token_embeddings + position_embeddings  # (batch, tokens, text_width)

        # Run the text transformer with a causal mask.
        mask = self.causal_mask[:, :, :token_count, :token_count]  # (1, 1, context, context) -> (1, 1, tokens, tokens)
        for block in self.blocks:
            x = block(x, mask)  # (batch, tokens, text_width)

        # Pool at the end-of-text token and project to the shared embedding space.
        x = self.ln_final(x)  # (batch, tokens, text_width)
        eot_indices = input_ids.argmax(dim=-1)  # (batch, tokens) -> (batch)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)  # -> (batch)
        pooled = x[batch_indices, eot_indices]  # (batch, tokens, text_width) -> (batch, text_width)
        text_features = self.text_projection(pooled)  # (batch, text_width) -> (batch, embed_dim)
        return text_features  # (batch, embed_dim)

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
        self.visual = VisionEncoder(image_size, patch_size, vision_width, vision_layers, vision_heads, embed_dim)
        self.text = TextEncoder(vocab_size, context_length, text_width, text_layers, text_heads, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, input_ids):
        # Encode each modality into the same embedding width.
        image_features = self.visual(images)  # (batch, 3, height, width) -> (batch, embed_dim)
        text_features = self.text(input_ids)  # (batch, tokens) -> (batch, embed_dim)

        # Normalize embeddings and score every image against every text.
        image_features = F.normalize(image_features, dim=-1)  # (batch, embed_dim)
        text_features = F.normalize(text_features, dim=-1)  # (batch, embed_dim)
        logit_scale = self.logit_scale.exp()  # scalar
        text_features_t = text_features.t()  # (batch, embed_dim) -> (embed_dim, batch)
        logits_per_image = logit_scale * image_features @ text_features_t  # (batch, embed_dim), (embed_dim, batch) -> (batch, batch)
        logits_per_text = logits_per_image.t()  # (batch, batch)
        return logits_per_image, logits_per_text  # two (batch, batch)

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
