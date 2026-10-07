# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class GraphAttentionLayer(nn.Module):
    output_dim: int = 8
    heads: int = 8
    concatenate: bool = True
    dropout: float = 0.6

    @nn.compact
    def __call__(self, features, adjacency, train=False):
        # Drop node features before computing a separate projection for every head.
        # @arch gat.layer.dropout:start
        feature_dropout = nn.Dropout(rate=self.dropout, name="feature_dropout")
        dropped_features = feature_dropout(features, deterministic=not train)
        # @arch gat.layer.dropout:end
        # @arch gat.layer.project:start
        projection = self.param("projection", nn.initializers.glorot_uniform(), (self.heads, features.shape[1], self.output_dim))
        projected = jnp.einsum("nf,hfc->nhc", dropped_features, projection)  # -> (nodes, heads, output_dim)
        # @arch gat.layer.project:end

        # Score ordered node pairs using the additive attention mechanism.
        # @arch gat.layer.score:start
        source_attention = self.param("source_attention", nn.initializers.glorot_uniform(), (self.heads, self.output_dim))
        target_attention = self.param("target_attention", nn.initializers.glorot_uniform(), (self.heads, self.output_dim))
        source_scores = jnp.einsum("nhc,hc->nh", projected, source_attention)  # -> (nodes, heads)
        target_scores = jnp.einsum("nhc,hc->nh", projected, target_attention)  # -> (nodes, heads)
        pair_scores = source_scores[:, None, :] + target_scores[None, :, :]  # -> (nodes, nodes, heads)
        scores = nn.leaky_relu(pair_scores, negative_slope=0.2)  # (nodes, nodes, heads)
        # @arch gat.layer.score:end

        # Mask non-neighbors and include self-loops so every node has a valid neighborhood.
        # @arch gat.layer.mask:start
        node_count = features.shape[0]  # (nodes, input_dim) -> scalar
        self_loops = jnp.eye(node_count, dtype=bool)
        neighbor_mask = adjacency.astype(bool) | self_loops  # (nodes, nodes)
        masked_scores = jnp.where(neighbor_mask[:, :, None], scores, -jnp.inf)
        # @arch gat.layer.mask:end

        # Normalize over neighbors, then apply attention dropout during training.
        # @arch gat.layer.softmax:start
        attention = nn.softmax(masked_scores, axis=1)  # (nodes, nodes, heads)
        # @arch gat.layer.softmax:end
        # @arch gat.layer.attention_dropout:start
        attention_dropout = nn.Dropout(rate=self.dropout, name="attention_dropout")
        attention = attention_dropout(attention, deterministic=not train)
        # @arch gat.layer.attention_dropout:end

        # Aggregate each node's neighbor features separately in every head.
        # @arch gat.layer.aggregate:start
        head_outputs = jnp.einsum("ijh,jhc->ihc", attention, projected)  # -> (nodes, heads, output_dim)
        # @arch gat.layer.aggregate:end
        if self.concatenate:
            # @arch gat.layer.concat:start
            outputs = head_outputs.reshape(node_count, self.heads * self.output_dim)
            # @arch gat.layer.concat:end
        else:
            # @arch gat.layer.average:start
            outputs = jnp.mean(head_outputs, axis=1)  # -> (nodes, output_dim)
            # @arch gat.layer.average:end
        # @arch gat.layer.bias:start
        bias = self.param("bias", nn.initializers.zeros, (outputs.shape[-1],))
        outputs = outputs + bias
        # @arch gat.layer.bias:end
        return outputs


# %% [notebook-only]
example_layer = GraphAttentionLayer(output_dim=8, heads=2, dropout=0.0)
example_features = jnp.ones((3, 4))  # -> (3, 4)
example_adjacency = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
example_params = example_layer.init(jax.random.PRNGKey(0), example_features, example_adjacency)
example_outputs = example_layer.apply(example_params, example_features, example_adjacency)  # -> (3, 16)
print("Graph attention layer:", example_features.shape, "->", example_outputs.shape)


# %%
class GAT(nn.Module):
    hidden_dim: int = 8
    heads: int = 8
    num_classes: int = 7
    dropout: float = 0.6

    @nn.compact
    def __call__(self, features, adjacency, train=False):
        # Consume node features and graph connectivity in the hidden attention layer.
        hidden_layer = GraphAttentionLayer(self.hidden_dim, self.heads, dropout=self.dropout, name="hidden")
        # @arch gat.input:start
        hidden = hidden_layer(features, adjacency, train=train)  # -> (nodes, heads * hidden_dim)
        # @arch gat.input:end
        # @arch gat.activation:start
        hidden = nn.elu(hidden)  # (nodes, heads * hidden_dim)
        # @arch gat.activation:end

        # Produce class logits for every node using an output attention head.
        # @arch gat.logits:start
        output_layer = GraphAttentionLayer(self.num_classes, heads=1, concatenate=False, dropout=self.dropout, name="output")
        logits = output_layer(hidden, adjacency, train=train)  # -> (nodes, num_classes)
        # @arch gat.logits:end
        return logits


# %% [notebook-only]
example_model = GAT(hidden_dim=8, heads=2, num_classes=2, dropout=0.0)
example_features = jnp.ones((3, 4))  # -> (3, 4)
example_adjacency = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
example_params = example_model.init(jax.random.PRNGKey(0), example_features, example_adjacency)
example_logits = example_model.apply(example_params, example_features, example_adjacency)  # -> (3, 2)
print("GAT node classifier:", example_features.shape, "->", example_logits.shape)


# %%
# Train on a small graph with two node classes.
model = GAT(hidden_dim=8, heads=2, num_classes=2, dropout=0.0)
train_features = jnp.array([[1., 0., 1., 0.], [1., 1., 0., 0.], [0., 1., 0., 1.], [0., 0., 1., 1.]])
train_adjacency = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=bool)
train_targets = jnp.array([0, 0, 1, 1])  # -> (4)
params = model.init(jax.random.PRNGKey(1), train_features, train_adjacency)


def train_step(params, features, adjacency, targets, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, features, adjacency, train=True)  # -> (nodes, num_classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
        loss = -jnp.mean(target_probs)  # -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


for step in range(3):
    params, loss = train_step(params, train_features, train_adjacency, train_targets)
final_loss = loss  # scalar
