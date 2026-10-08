# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, heads=8, concatenate=True, dropout=0.6):
        super().__init__()
        self.heads = heads
        self.output_dim = output_dim
        self.concatenate = concatenate
        self.dropout = dropout
        self.projection = nn.Parameter(torch.empty(heads, input_dim, output_dim))
        self.source_attention = nn.Parameter(torch.empty(heads, output_dim))
        self.target_attention = nn.Parameter(torch.empty(heads, output_dim))
        output_width = heads * output_dim if concatenate else output_dim
        self.bias = nn.Parameter(torch.zeros(output_width))
        nn.init.xavier_uniform_(self.projection)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, features, adjacency):
        # Drop node features before computing a separate projection for every head.
        # @arch gat.layer.dropout:start
        dropped_features = F.dropout(features, p=self.dropout, training=self.training)
        # @arch gat.layer.dropout:end
        # @arch gat.layer.project:start
        projected = torch.einsum("nf,hfc->nhc", dropped_features, self.projection)  # -> (nodes, heads, output_dim)
        # @arch gat.layer.project:end

        # Score ordered node pairs using the additive attention mechanism.
        # @arch gat.layer.score:start
        source_scores = torch.einsum("nhc,hc->nh", projected, self.source_attention)  # -> (nodes, heads)
        target_scores = torch.einsum("nhc,hc->nh", projected, self.target_attention)  # -> (nodes, heads)
        pair_scores = source_scores[:, None, :] + target_scores[None, :, :]  # -> (nodes, nodes, heads)
        scores = F.leaky_relu(pair_scores, negative_slope=0.2)  # (nodes, nodes, heads)
        # @arch gat.layer.score:end

        # Mask non-neighbors and include self-loops so every node has a valid neighborhood.
        # @arch gat.layer.mask:start
        node_count = features.shape[0]  # (nodes, input_dim) -> scalar
        self_loops = torch.eye(node_count, dtype=torch.bool, device=features.device)
        neighbor_mask = adjacency.bool() | self_loops  # (nodes, nodes)
        masked_scores = scores.masked_fill(~neighbor_mask[:, :, None], float("-inf"))
        # @arch gat.layer.mask:end

        # Normalize over neighbors, then apply attention dropout during training.
        # @arch gat.layer.softmax:start
        attention = torch.softmax(masked_scores, dim=1)  # (nodes, nodes, heads)
        # @arch gat.layer.softmax:end
        # @arch gat.layer.attention_dropout:start
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        # @arch gat.layer.attention_dropout:end

        # Aggregate each node's neighbor features separately in every head.
        # @arch gat.layer.aggregate:start
        head_outputs = torch.einsum("ijh,jhc->ihc", attention, projected)  # -> (nodes, heads, output_dim)
        # @arch gat.layer.aggregate:end
        if self.concatenate:
            # @arch gat.layer.concat:start
            outputs = head_outputs.reshape(node_count, self.heads * self.output_dim)
            # @arch gat.layer.concat:end
        else:
            # @arch gat.layer.average:start
            outputs = head_outputs.mean(dim=1)  # -> (nodes, output_dim)
            # @arch gat.layer.average:end
        # @arch gat.layer.bias:start
        outputs = outputs + self.bias
        # @arch gat.layer.bias:end
        return outputs


# %% [notebook-only]
example_layer = GraphAttentionLayer(4, 8, heads=2, dropout=0.0)
example_features = torch.randn(3, 4)  # -> (3, 4)
example_adjacency = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
example_outputs = example_layer(example_features, example_adjacency)  # -> (3, 16)
print("Graph attention layer:", example_features.shape, "->", example_outputs.shape)


# %%
class GAT(nn.Module):
    def __init__(self, input_dim=1433, hidden_dim=8, heads=8, num_classes=7, dropout=0.6):
        super().__init__()
        self.hidden = GraphAttentionLayer(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.output = GraphAttentionLayer(hidden_dim * heads, num_classes, heads=1, concatenate=False, dropout=dropout)

    def forward(self, features, adjacency):
        # Consume node features and graph connectivity in the hidden attention layer.
        # @arch gat.input:start
        hidden = self.hidden(features, adjacency)  # (nodes, input_dim) -> (nodes, heads * hidden_dim)
        # @arch gat.input:end
        # @arch gat.activation:start
        hidden = F.elu(hidden)  # (nodes, heads * hidden_dim)
        # @arch gat.activation:end

        # Produce class logits for every node using an output attention head.
        # @arch gat.logits:start
        logits = self.output(hidden, adjacency)  # -> (nodes, num_classes)
        # @arch gat.logits:end
        return logits


# %% [notebook-only]
example_model = GAT(input_dim=4, hidden_dim=8, heads=2, num_classes=2, dropout=0.0)
example_features = torch.randn(3, 4)  # -> (3, 4)
example_adjacency = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
example_logits = example_model(example_features, example_adjacency)  # -> (3, 2)
print("GAT node classifier:", example_features.shape, "->", example_logits.shape)


# %%
# Train on a small graph with two node classes.
model = GAT(input_dim=4, hidden_dim=8, heads=2, num_classes=2, dropout=0.0)
train_features = torch.tensor([[1., 0., 1., 0.], [1., 1., 0., 0.], [0., 1., 0., 1.], [0., 0., 1., 1.]])
train_adjacency = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.bool)
train_targets = torch.tensor([0, 0, 1, 1])  # -> (4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_features, train_adjacency)  # -> (4, 2)
    loss = F.cross_entropy(logits, train_targets)  # -> scalar
    loss.backward()
    optimizer.step()
final_loss = loss.item()  # scalar
