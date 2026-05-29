"use client";

import { Fragment, useEffect, useRef, useState } from "react";

type NodeKind =
  | "input"
  | "group"
  | "conv"
  | "activation"
  | "pool"
  | "reshape"
  | "linear"
  | "embedding"
  | "norm"
  | "attention"
  | "recurrent"
  | "head"
  | "residual"
  | "mlp"
  | "dropout"
  | "concat";

type ArchNode = {
  id: string;
  label: string;
  type: string;
  kind: NodeKind;
  summary?: string;
  badges?: string[];
  children?: ArchNode[];
  lazyChildren?: () => ArchNode[];
  defaultExpanded?: boolean;
  codeLines: number[];
};

type ModelSpec = {
  id: string;
  label: string;
  breadcrumb: string;
  stats: string;
  fileName: string;
  jaxFileName: string;
  paper: {
    title: string;
    authors: string;
    year: string;
    publishedLabel: string;
    publishedDate: string;
    venue: string;
    url: string;
    pdfUrl: string;
    focus: string[];
  };
  selectedId: string;
  nodes: ArchNode[];
  code: string[];
  jaxCode: string[];
};

type PaneKey = "architecture" | "paper" | "code";
type CodeLanguage = "pytorch" | "jax";

const languageLabels: Record<CodeLanguage, string> = {
  pytorch: "PyTorch",
  jax: "JAX",
};

const mlpJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "# Define a multilayer perceptron.",
  "class MLP(nn.Module):",
  "    hidden_dim: int = 128",
  "    output_dim: int = 10",
  "",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        # First hidden block: (batch, features) -> (batch, hidden_dim).",
  "        h1_pre = nn.Dense(self.hidden_dim, name='hidden1')(x)",
  "        h1 = nn.sigmoid(h1_pre)",
  "",
  "        # Second hidden block keeps the hidden shape: (batch, hidden_dim).",
  "        h2_pre = nn.Dense(self.hidden_dim, name='hidden2')(h1)",
  "        h2 = nn.sigmoid(h2_pre)",
  "",
  "        # Output block: (batch, hidden_dim) -> (batch, output_dim).",
  "        logits = nn.Dense(self.output_dim, name='output')(h2)",
  "        return logits",
  "",
  "",
  "# Create and run a sample batch: (2, 784) -> (2, 10).",
  "model = MLP(hidden_dim=128, output_dim=10)",
  "inputs = jnp.ones((2, 784))",
  "params = model.init(jax.random.PRNGKey(0), inputs)",
  "logits = model.apply(params, inputs)",
  "",
  "# logits: (2, 10)",
];

const rnnJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class ElmanRNN(nn.Module):",
  "    hidden_size: int = 64",
  "    output_size: int = 10",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        batch_size = x.shape[0]",
  "        hidden_shape = (batch_size, self.hidden_size)",
  "        h = jnp.zeros(hidden_shape)",
  "        states = []",
  "        input_to_hidden = nn.Dense(self.hidden_size, name='input_to_hidden')",
  "        hidden_to_hidden = nn.Dense(self.hidden_size, use_bias=False, name='hidden_to_hidden')",
  "        hidden_to_output = nn.Dense(self.output_size, name='hidden_to_output')",
  "        step_count = x.shape[1]",
  "        for t in range(step_count):",
  "            current_input = x[:, t]",
  "            input_hidden = input_to_hidden(current_input)",
  "            recurrent_hidden = hidden_to_hidden(h)",
  "            hidden_sum = input_hidden + recurrent_hidden",
  "            h = jnp.tanh(hidden_sum)",
  "            states.append(h)",
  "        logits = hidden_to_output(h)",
  "        state_trace = jnp.stack(states, axis=1)",
  "        outputs = (logits, state_trace)",
  "        return outputs",
  "",
  "",
  "model = ElmanRNN(hidden_size=64, output_size=10)",
  "sequence = jnp.ones((2, 8, 32))",
  "params = model.init(jax.random.PRNGKey(0), sequence)",
  "logits, states = model.apply(params, sequence)",
  "# logits: (2, 10), states: (2, 8, 64)",
];

const gruJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class GRUCell(nn.Module):",
  "    hidden_size: int = 64",
  "    @nn.compact",
  "    def __call__(self, x, h):",
  "        x_z = nn.Dense(self.hidden_size, name='x_z')(x)",
  "        h_z = nn.Dense(self.hidden_size, use_bias=False, name='h_z')(h)",
  "        z_pre = x_z + h_z",
  "        z = nn.sigmoid(z_pre)",
  "",
  "        x_r = nn.Dense(self.hidden_size, name='x_r')(x)",
  "        h_r = nn.Dense(self.hidden_size, use_bias=False, name='h_r')(h)",
  "        r_pre = x_r + h_r",
  "        r = nn.sigmoid(r_pre)",
  "",
  "        reset_h = r * h",
  "        x_n = nn.Dense(self.hidden_size, name='x_n')(x)",
  "        h_n = nn.Dense(self.hidden_size, use_bias=False, name='h_n')(reset_h)",
  "        n_pre = x_n + h_n",
  "        n = jnp.tanh(n_pre)",
  "",
  "        keep_h = z * h",
  "        candidate_h = (1.0 - z) * n",
  "        h_next = candidate_h + keep_h",
  "        return h_next",
  "",
  "",
  "class GRUSequence(nn.Module):",
  "    hidden_size: int = 64",
  "    output_size: int = 10",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        batch_size = x.shape[0]",
  "        hidden_shape = (batch_size, self.hidden_size)",
  "        h = jnp.zeros(hidden_shape)",
  "        states = []",
  "        cell = GRUCell(self.hidden_size)",
  "        step_count = x.shape[1]",
  "        for t in range(step_count):",
  "            current_input = x[:, t]",
  "            h = cell(current_input, h)",
  "            states.append(h)",
  "        logits = nn.Dense(self.output_size, name='readout')(h)",
  "        state_trace = jnp.stack(states, axis=1)",
  "        outputs = (logits, state_trace)",
  "        return outputs",
  "",
  "",
  "model = GRUSequence(hidden_size=64, output_size=10)",
  "sequence = jnp.ones((2, 8, 32))",
  "params = model.init(jax.random.PRNGKey(0), sequence)",
  "logits, states = model.apply(params, sequence)",
  "# logits: (2, 10), states: (2, 8, 64)",
];

const lstmJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class LSTMCell(nn.Module):",
  "    hidden_size: int = 64",
  "    @nn.compact",
  "    def __call__(self, x, state):",
  "        h, c = state",
  "        x_i = nn.Dense(self.hidden_size, name='x_i')(x)",
  "        h_i = nn.Dense(self.hidden_size, use_bias=False, name='h_i')(h)",
  "        i_pre = x_i + h_i",
  "        i = nn.sigmoid(i_pre)",
  "",
  "        x_f = nn.Dense(self.hidden_size, name='x_f')(x)",
  "        h_f = nn.Dense(self.hidden_size, use_bias=False, name='h_f')(h)",
  "        f_pre = x_f + h_f",
  "        f = nn.sigmoid(f_pre)",
  "",
  "        x_g = nn.Dense(self.hidden_size, name='x_g')(x)",
  "        h_g = nn.Dense(self.hidden_size, use_bias=False, name='h_g')(h)",
  "        g_pre = x_g + h_g",
  "        g = jnp.tanh(g_pre)",
  "",
  "        x_o = nn.Dense(self.hidden_size, name='x_o')(x)",
  "        h_o = nn.Dense(self.hidden_size, use_bias=False, name='h_o')(h)",
  "        o_pre = x_o + h_o",
  "        o = nn.sigmoid(o_pre)",
  "",
  "        forget_c = f * c",
  "        write_c = i * g",
  "        c_next = forget_c + write_c",
  "        c_readout = jnp.tanh(c_next)",
  "        h_next = o * c_readout",
  "        return h_next, c_next",
  "",
  "",
  "class LSTMSequence(nn.Module):",
  "    hidden_size: int = 64",
  "    output_size: int = 10",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        batch_size = x.shape[0]",
  "        hidden_shape = (batch_size, self.hidden_size)",
  "        h = jnp.zeros(hidden_shape)",
  "        c = jnp.zeros(hidden_shape)",
  "        states = []",
  "        cell = LSTMCell(self.hidden_size)",
  "        step_count = x.shape[1]",
  "        for t in range(step_count):",
  "            current_input = x[:, t]",
  "            state = (h, c)",
  "            h, c = cell(current_input, state)",
  "            states.append(h)",
  "        logits = nn.Dense(self.output_size, name='readout')(h)",
  "        state_trace = jnp.stack(states, axis=1)",
  "        outputs = (logits, state_trace)",
  "        return outputs",
  "",
  "",
  "model = LSTMSequence(hidden_size=64, output_size=10)",
  "sequence = jnp.ones((2, 8, 32))",
  "params = model.init(jax.random.PRNGKey(0), sequence)",
  "logits, states = model.apply(params, sequence)",
  "# logits: (2, 10), states: (2, 8, 64)",
];

const lenet5JaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class LeNet5(nn.Module):",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        # x: (batch, 32, 32, 1)",
  "        conv1 = nn.Conv(features=6, kernel_size=(5, 5))(x)",
  "        x = jnp.tanh(conv1)",
  "        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))",
  "        conv2 = nn.Conv(features=16, kernel_size=(5, 5))(x)",
  "        x = jnp.tanh(conv2)",
  "        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))",
  "        batch_size = x.shape[0]",
  "        x = x.reshape((batch_size, -1))",
  "",
  "        fc1 = nn.Dense(features=120)(x)",
  "        x = jnp.tanh(fc1)",
  "        fc2 = nn.Dense(features=84)(x)",
  "        x = jnp.tanh(fc2)",
  "        logits = nn.Dense(features=10)(x)",
  "        return logits",
  "",
  "",
  "model = LeNet5()",
  "test_input = jnp.ones((2, 32, 32, 1))",
  "params = model.init(jax.random.PRNGKey(0), test_input)",
  "logits = model.apply(params, test_input)",
  "# logits: (2, 10)",
];

const alexnetJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class AlexNet(nn.Module):",
  "    num_classes: int = 1000",
  "    @nn.compact",
  "    def __call__(self, x, train=False):",
  "        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4), name='conv1')(x)",
  "        x = nn.relu(x)",
  "        x = local_response_norm(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))",
  "        x = nn.Conv(features=256, kernel_size=(5, 5), padding='SAME', name='conv2')(x)",
  "        x = nn.relu(x)",
  "        x = local_response_norm(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))",
  "        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv3')(x)",
  "        x = nn.relu(x)",
  "        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv4')(x)",
  "        x = nn.relu(x)",
  "        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='conv5')(x)",
  "        x = nn.relu(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))",
  "        batch_size = x.shape[0]",
  "        x = x.reshape((batch_size, -1))",
  "        x = nn.Dropout(0.5, deterministic=not train)(x)",
  "        x = nn.Dense(features=4096, name='fc6')(x)",
  "        x = nn.relu(x)",
  "        x = nn.Dropout(0.5, deterministic=not train)(x)",
  "        x = nn.Dense(features=4096, name='fc7')(x)",
  "        x = nn.relu(x)",
  "        logits = nn.Dense(features=self.num_classes, name='fc8')(x)",
  "        return logits",
  "",
  "",
  "def local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0):",
  "    half = size // 2",
  "    squared = jnp.square(x)",
  "    padded = jnp.pad(squared, ((0, 0), (0, 0), (0, 0), (half, half)))",
  "    scale = k",
  "    for offset in range(size):",
  "        channel_end = offset + x.shape[-1]",
  "        window = padded[..., offset:channel_end]",
  "        scale_step = (alpha / size) * window",
  "        scale = scale + scale_step",
  "    denominator = jnp.power(scale, beta)",
  "    normalized = x / denominator",
  "    return normalized",
  "",
  "",
  "model = AlexNet(num_classes=1000)",
  "test_input = jnp.ones((2, 227, 227, 3))",
  "params = model.init(jax.random.PRNGKey(0), test_input, train=False)",
  "logits = model.apply(params, test_input, train=False)",
  "# logits: (2, 1000)",
];

const googlenetJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class InceptionBlock(nn.Module):",
  "    branch1_channels: int",
  "    branch3_reduce: int",
  "    branch3_channels: int",
  "    branch5_reduce: int",
  "    branch5_channels: int",
  "    pool_channels: int",
  "",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        branch1 = nn.Conv(self.branch1_channels, (1, 1), name='branch1')(x)",
  "        branch1 = nn.relu(branch1)",
  "",
  "        branch3 = nn.Conv(self.branch3_reduce, (1, 1), name='branch3_reduce')(x)",
  "        branch3 = nn.relu(branch3)",
  "        branch3 = nn.Conv(self.branch3_channels, (3, 3), padding='SAME', name='branch3')(branch3)",
  "        branch3 = nn.relu(branch3)",
  "",
  "        branch5 = nn.Conv(self.branch5_reduce, (1, 1), name='branch5_reduce')(x)",
  "        branch5 = nn.relu(branch5)",
  "        branch5 = nn.Conv(self.branch5_channels, (5, 5), padding='SAME', name='branch5')(branch5)",
  "        branch5 = nn.relu(branch5)",
  "",
  "        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')",
  "        branch_pool = nn.Conv(self.pool_channels, (1, 1), name='pool_proj')(branch_pool)",
  "        branch_pool = nn.relu(branch_pool)",
  "",
  "        branches = [branch1, branch3, branch5, branch_pool]",
  "        x = jnp.concatenate(branches, axis=-1)",
  "        return x",
  "",
  "",
  "class GoogLeNet(nn.Module):",
  "    num_classes: int = 1000",
  "",
  "    @nn.compact",
  "    def __call__(self, x, train=False):",
  "        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME', name='stem_conv7')(x)",
  "        x = nn.relu(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')",
  "        x = nn.Conv(64, (1, 1), name='stem_conv1')(x)",
  "        x = nn.relu(x)",
  "        x = nn.Conv(192, (3, 3), padding='SAME', name='stem_conv3')(x)",
  "        x = nn.relu(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')",
  "",
  "        x = InceptionBlock(64, 96, 128, 16, 32, 32, name='inception3a')(x)",
  "        x = InceptionBlock(128, 128, 192, 32, 96, 64, name='inception3b')(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')",
  "        x = InceptionBlock(192, 96, 208, 16, 48, 64, name='inception4a')(x)",
  "        x = InceptionBlock(160, 112, 224, 24, 64, 64, name='inception4b')(x)",
  "        x = InceptionBlock(128, 128, 256, 24, 64, 64, name='inception4c')(x)",
  "        x = InceptionBlock(112, 144, 288, 32, 64, 64, name='inception4d')(x)",
  "        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception4e')(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')",
  "        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception5a')(x)",
  "        x = InceptionBlock(384, 192, 384, 48, 128, 128, name='inception5b')(x)",
  "",
  "        x = jnp.mean(x, axis=(1, 2))",
  "        x = nn.Dropout(0.4, deterministic=not train)(x)",
  "        logits = nn.Dense(self.num_classes, name='fc')(x)",
  "        return logits",
  "",
  "",
  "model = GoogLeNet(num_classes=1000)",
  "test_input = jnp.ones((2, 224, 224, 3))",
  "params = model.init(jax.random.PRNGKey(0), test_input, train=False)",
  "logits = model.apply(params, test_input, train=False)",
  "# logits: (2, 1000)",
];

const unetJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class DoubleConv(nn.Module):",
  "    out_channels: int",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)",
  "        x = nn.relu(x)",
  "        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)",
  "        x = nn.relu(x)",
  "        return x",
  "",
  "",
  "class UNet(nn.Module):",
  "    num_classes: int = 2",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        d1 = DoubleConv(64)(x)",
  "        p1 = nn.max_pool(d1, (2, 2), (2, 2))",
  "        d2 = DoubleConv(128)(p1)",
  "        p2 = nn.max_pool(d2, (2, 2), (2, 2))",
  "        d3 = DoubleConv(256)(p2)",
  "        p3 = nn.max_pool(d3, (2, 2), (2, 2))",
  "        d4 = DoubleConv(512)(p3)",
  "        p4 = nn.max_pool(d4, (2, 2), (2, 2))",
  "        b = DoubleConv(1024)(p4)",
  "        x = resize_like(b, d4)",
  "        x = jnp.concatenate([x, d4], axis=-1)",
  "        x = DoubleConv(512)(x)",
  "        x = resize_like(x, d3)",
  "        x = jnp.concatenate([x, d3], axis=-1)",
  "        x = DoubleConv(256)(x)",
  "        x = resize_like(x, d2)",
  "        x = jnp.concatenate([x, d2], axis=-1)",
  "        x = DoubleConv(128)(x)",
  "        x = resize_like(x, d1)",
  "        x = jnp.concatenate([x, d1], axis=-1)",
  "        x = DoubleConv(64)(x)",
  "        logits = nn.Conv(self.num_classes, (1, 1), name='out_conv')(x)",
  "        return logits",
  "",
  "",
  "def resize_like(x, skip):",
  "    resize_shape = (x.shape[0], skip.shape[1], skip.shape[2], x.shape[-1])",
  "    resized = jax.image.resize(x, resize_shape, method='nearest')",
  "    return resized",
  "",
  "",
  "model = UNet(num_classes=2)",
  "test_input = jnp.ones((2, 572, 572, 1))",
  "params = model.init(jax.random.PRNGKey(0), test_input)",
  "logits = model.apply(params, test_input)",
  "# logits: (2, 572, 572, 2)",
];

const transformerJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class PositionalEncoding(nn.Module):",
  "    d_model: int = 512",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        seq_len = x.shape[1]",
  "        positions = jnp.arange(seq_len)",
  "        position = positions[:, None]",
  "        even_indices = jnp.arange(0, self.d_model, 2)",
  "        scale = -jnp.log(10000.0) / self.d_model",
  "        div_term = jnp.exp(even_indices * scale)",
  "        pe = jnp.zeros((seq_len, self.d_model))",
  "        sin_values = jnp.sin(position * div_term)",
  "        cos_values = jnp.cos(position * div_term)",
  "        pe = pe.at[:, 0::2].set(sin_values)",
  "        pe = pe.at[:, 1::2].set(cos_values)",
  "        batch_pe = pe[None, :, :]",
  "        encoded = x + batch_pe",
  "        return encoded",
  "",
  "",
  "class EncoderLayer(nn.Module):",
  "    d_model: int = 512",
  "    nhead: int = 8",
  "    d_ff: int = 2048",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        attn = nn.MultiHeadDotProductAttention(num_heads=self.nhead)(x, x)",
  "        attn_residual = x + attn",
  "        x = nn.LayerNorm()(attn_residual)",
  "        ffn_layers = [nn.Dense(self.d_ff), nn.relu, nn.Dense(self.d_model)]",
  "        ffn = nn.Sequential(ffn_layers)(x)",
  "        ffn_residual = x + ffn",
  "        out = nn.LayerNorm()(ffn_residual)",
  "        return out",
  "",
  "",
  "class DecoderLayer(nn.Module):",
  "    d_model: int = 512",
  "    nhead: int = 8",
  "    d_ff: int = 2048",
  "    @nn.compact",
  "    def __call__(self, x, memory, mask):",
  "        masked = nn.MultiHeadDotProductAttention(num_heads=self.nhead)(x, x, mask=mask)",
  "        masked_residual = x + masked",
  "        x = nn.LayerNorm()(masked_residual)",
  "        cross = nn.MultiHeadDotProductAttention(num_heads=self.nhead)(x, memory)",
  "        cross_residual = x + cross",
  "        x = nn.LayerNorm()(cross_residual)",
  "        ffn_layers = [nn.Dense(self.d_ff), nn.relu, nn.Dense(self.d_model)]",
  "        ffn = nn.Sequential(ffn_layers)(x)",
  "        ffn_residual = x + ffn",
  "        out = nn.LayerNorm()(ffn_residual)",
  "        return out",
  "",
  "",
  "class Transformer(nn.Module):",
  "    vocab_size: int = 37000",
  "    d_model: int = 512",
  "    nhead: int = 8",
  "    num_layers: int = 6",
  "    @nn.compact",
  "    def __call__(self, src_ids, tgt_ids, tgt_mask):",
  "        src_embedding = nn.Embed(self.vocab_size, self.d_model)(src_ids)",
  "        memory = PositionalEncoding(self.d_model)(src_embedding)",
  "        for _ in range(self.num_layers):",
  "            memory = EncoderLayer(self.d_model, self.nhead)(memory)",
  "        tgt_embedding = nn.Embed(self.vocab_size, self.d_model)(tgt_ids)",
  "        x = PositionalEncoding(self.d_model)(tgt_embedding)",
  "        for _ in range(self.num_layers):",
  "            x = DecoderLayer(self.d_model, self.nhead)(x, memory, tgt_mask)",
  "        logits = nn.Dense(self.vocab_size)(x)",
  "        return logits",
  "",
  "",
  "model = Transformer(vocab_size=37000)",
  "src_ids = jnp.ones((2, 16), dtype=jnp.int32)",
  "tgt_ids = jnp.ones((2, 16), dtype=jnp.int32)",
  "mask_values = jnp.ones((1, 1, 16, 16))",
  "tgt_mask = jnp.tril(mask_values)",
  "params = model.init(jax.random.PRNGKey(0), src_ids, tgt_ids, tgt_mask)",
  "logits = model.apply(params, src_ids, tgt_ids, tgt_mask)",
  "# logits: (2, 16, 37000)",
];

const bertJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class BertEmbeddings(nn.Module):",
  "    vocab_size: int = 30522",
  "    hidden_size: int = 768",
  "    max_position: int = 512",
  "    type_vocab_size: int = 2",
  "    @nn.compact",
  "    def __call__(self, input_ids, token_type_ids, train=False):",
  "        positions = jnp.arange(input_ids.shape[1])",
  "        x = nn.Embed(self.vocab_size, self.hidden_size, name='word_embeddings')(input_ids)",
  "        position_embeddings = nn.Embed(self.max_position, self.hidden_size, name='position_embeddings')(positions)",
  "        position_embeddings = position_embeddings[None, :, :]",
  "        x = x + position_embeddings",
  "        token_type_embeddings = nn.Embed(self.type_vocab_size, self.hidden_size, name='token_type_embeddings')(token_type_ids)",
  "        x = x + token_type_embeddings",
  "        x = nn.LayerNorm(name='LayerNorm')(x)",
  "        x = nn.Dropout(0.1, deterministic=not train)(x)",
  "        return x",
  "",
  "",
  "class BertLayer(nn.Module):",
  "    hidden_size: int = 768",
  "    num_heads: int = 12",
  "    intermediate_size: int = 3072",
  "    @nn.compact",
  "    def __call__(self, x, attention_mask=None, train=False):",
  "        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, name='attention')(x, x, mask=attention_mask)",
  "        attn = nn.Dropout(0.1, deterministic=not train)(attn)",
  "        attn_residual = x + attn",
  "        x = nn.LayerNorm(name='attention_norm')(attn_residual)",
  "        ffn = nn.Dense(self.intermediate_size, name='intermediate')(x)",
  "        ffn = nn.gelu(ffn)",
  "        ffn = nn.Dense(self.hidden_size, name='output_dense')(ffn)",
  "        ffn = nn.Dropout(0.1, deterministic=not train)(ffn)",
  "        ffn_residual = x + ffn",
  "        out = nn.LayerNorm(name='output_norm')(ffn_residual)",
  "        return out",
  "",
  "",
  "class BERTBase(nn.Module):",
  "    vocab_size: int = 30522",
  "    hidden_size: int = 768",
  "    num_layers: int = 12",
  "    @nn.compact",
  "    def __call__(self, input_ids, token_type_ids, attention_mask=None, train=False):",
  "        x = BertEmbeddings(self.vocab_size, self.hidden_size)(input_ids, token_type_ids, train=train)",
  "        for _ in range(self.num_layers):",
  "            x = BertLayer(self.hidden_size)(x, attention_mask, train=train)",
  "        cls_token = x[:, 0]",
  "        pooled_projection = nn.Dense(self.hidden_size, name='pooler')(cls_token)",
  "        pooled = jnp.tanh(pooled_projection)",
  "        mlm_logits = nn.Dense(self.vocab_size, name='mlm_head')(x)",
  "        return mlm_logits, pooled",
  "",
  "",
  "model = BERTBase(vocab_size=30522)",
  "input_ids = jnp.ones((2, 16), dtype=jnp.int32)",
  "token_type_ids = jnp.zeros((2, 16), dtype=jnp.int32)",
  "attention_mask = jnp.ones((2, 1, 1, 16), dtype=jnp.bool_)",
  "params = model.init(jax.random.PRNGKey(0), input_ids, token_type_ids, attention_mask)",
  "mlm_logits, pooled = model.apply(params, input_ids, token_type_ids, attention_mask)",
  "# mlm_logits: (2, 16, 30522), pooled: (2, 768)",
];

const vitJaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class PatchEmbed(nn.Module):",
  "    embed_dim: int = 768",
  "    patch_size: int = 16",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        projection = nn.Conv(self.embed_dim, (self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), name='proj')",
  "        x = projection(x)",
  "        batch_size = x.shape[0]",
  "        x = x.reshape((batch_size, -1, self.embed_dim))",
  "        return x",
  "",
  "",
  "class EncoderBlock(nn.Module):",
  "    embed_dim: int = 768",
  "    num_heads: int = 12",
  "    mlp_dim: int = 3072",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        y = nn.LayerNorm(name='ln_1')(x)",
  "        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, name='attn')(y, y)",
  "        x = x + y",
  "        y = nn.LayerNorm(name='ln_2')(x)",
  "        y = nn.Dense(self.mlp_dim, name='mlp_fc1')(y)",
  "        y = nn.gelu(y)",
  "        y = nn.Dense(self.embed_dim, name='mlp_fc2')(y)",
  "        out = x + y",
  "        return out",
  "",
  "",
  "class VisionTransformer(nn.Module):",
  "    num_classes: int = 1000",
  "    embed_dim: int = 768",
  "    depth: int = 12",
  "    num_heads: int = 12",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        x = PatchEmbed(self.embed_dim)(x)",
  "        cls = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))",
  "        batch_size = x.shape[0]",
  "        cls_shape = (batch_size, 1, 1)",
  "        cls = jnp.tile(cls, cls_shape)",
  "        x = jnp.concatenate([cls, x], axis=1)",
  "        pos_init = nn.initializers.normal(0.02)",
  "        pos_shape = (1, x.shape[1], self.embed_dim)",
  "        pos = self.param('pos_embed', pos_init, pos_shape)",
  "        x = x + pos",
  "        for _ in range(self.depth):",
  "            x = EncoderBlock(self.embed_dim, self.num_heads)(x)",
  "        x = nn.LayerNorm(name='encoder_norm')(x)",
  "        cls_output = x[:, 0]",
  "        logits = nn.Dense(self.num_classes, name='head')(cls_output)",
  "        return logits",
  "",
  "",
  "model = VisionTransformer(num_classes=1000)",
  "test_input = jnp.ones((2, 224, 224, 3))",
  "params = model.init(jax.random.PRNGKey(0), test_input)",
  "logits = model.apply(params, test_input)",
  "# logits: (2, 1000)",
];

const gpt2JaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class GPT2Small(nn.Module):",
  "    vocab_size: int",
  "    n_ctx: int = 1024",
  "    n_embd: int = 768",
  "    n_head: int = 12",
  "    n_layer: int = 12",
  "    @nn.compact",
  "    def __call__(self, input_ids, mask):",
  "        B, T = input_ids.shape",
  "        positions = jnp.arange(T)",
  "        token_embeddings = nn.Embed(self.vocab_size, self.n_embd, name='wte')(input_ids)",
  "        position_embeddings = nn.Embed(self.n_ctx, self.n_embd, name='wpe')(positions)",
  "        position_embeddings = position_embeddings[None, :, :]",
  "        x = token_embeddings + position_embeddings",
  "        x = nn.Dropout(0.1, deterministic=True, name='drop')(x)",
  "        for _ in range(self.n_layer):",
  "            x = Block()(x, mask)",
  "        x = nn.LayerNorm(name='ln_f')(x)",
  "        logits = nn.Dense(self.vocab_size, name='lm_head')(x)",
  "        return logits",
  "",
  "class CausalSelfAttention(nn.Module):",
  "    n_embd: int = 768",
  "    n_head: int = 12",
  "    def __call__(self, x, mask):",
  "        B, T, C = x.shape",
  "        qkv_dim = 3 * C",
  "        qkv = nn.Dense(qkv_dim, name='c_attn')(x)",
  "        q, k, v = jnp.split(qkv, 3, axis=2)",
  "        head_dim = C // self.n_head",
  "        q = q.reshape(B, T, self.n_head, head_dim)",
  "        q = q.transpose(0, 2, 1, 3)",
  "        k = k.reshape(B, T, self.n_head, head_dim)",
  "        k = k.transpose(0, 2, 1, 3)",
  "        v = v.reshape(B, T, self.n_head, head_dim)",
  "        v = v.transpose(0, 2, 1, 3)",
  "        key_transpose = jnp.swapaxes(k, -2, -1)",
  "        scores = q @ key_transpose",
  "        scale = k.shape[-1] ** -0.5",
  "        att = scores * scale",
  "        mask_window = mask[:, :, :T, :T]",
  "        att = jnp.where(mask_window == 0, -jnp.inf, att)",
  "        weights = nn.softmax(att, axis=-1)",
  "        y = weights @ v",
  "        y = y.transpose(0, 2, 1, 3)",
  "        y = y.reshape(B, T, C)",
  "        out = nn.Dense(C, name='c_proj')(y)",
  "        return out",
  "",
  "class Block(nn.Module):",
  "    def __call__(self, x, mask):",
  "        attn_input = nn.LayerNorm(name='ln_1')(x)",
  "        attn = CausalSelfAttention()(attn_input, mask)",
  "        x = x + attn",
  "        mlp_input = nn.LayerNorm(name='ln_2')(x)",
  "        mlp_out = MLP()(mlp_input)",
  "        x = x + mlp_out",
  "        return x",
  "",
  "",
  "model = GPT2Small(vocab_size=50257)",
  "test_input = jnp.ones((2, 16), dtype=jnp.int32)",
  "mask_values = jnp.ones((16, 16))",
  "mask = jnp.tril(mask_values)",
  "mask = mask.reshape(1, 1, 16, 16)",
  "params = model.init(jax.random.PRNGKey(0), test_input, mask)",
  "logits = model.apply(params, test_input, mask)",
  "# logits: (2, 16, 50257)",
];

const resnet18JaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class BasicBlock(nn.Module):",
  "    out_channels: int",
  "    stride: int = 1",
  "    use_projection: bool = False",
  "    @nn.compact",
  "    def __call__(self, x, train=False):",
  "        identity = x",
  "        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding='SAME', use_bias=False, name='conv1')(x)",
  "        y = nn.BatchNorm(use_running_average=not train, name='bn1')(y)",
  "        y = nn.relu(y)",
  "        y = nn.Conv(self.out_channels, (3, 3), padding='SAME', use_bias=False, name='conv2')(y)",
  "        y = nn.BatchNorm(use_running_average=not train, name='bn2')(y)",
  "        if self.use_projection:",
  "            identity = nn.Conv(self.out_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False, name='downsample_conv')(x)",
  "            identity = nn.BatchNorm(use_running_average=not train, name='downsample_bn')(identity)",
  "        y = y + identity",
  "        y = nn.relu(y)",
  "        return y",
  "",
  "",
  "class ResNet18(nn.Module):",
  "    num_classes: int = 1000",
  "    @nn.compact",
  "    def __call__(self, x, train=False):",
  "        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME', use_bias=False, name='stem_conv')(x)",
  "        x = nn.BatchNorm(use_running_average=not train, name='stem_bn')(x)",
  "        x = nn.relu(x)",
  "        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')",
  "        x = self._stage(x, 64, blocks=2, stride=1, train=train)",
  "        x = self._stage(x, 128, blocks=2, stride=2, train=train)",
  "        x = self._stage(x, 256, blocks=2, stride=2, train=train)",
  "        x = self._stage(x, 512, blocks=2, stride=2, train=train)",
  "        x = jnp.mean(x, axis=(1, 2))",
  "        logits = nn.Dense(self.num_classes, name='fc')(x)",
  "        return logits",
  "",
  "    def _stage(self, x, channels, blocks, stride, train):",
  "        use_projection = stride != 1",
  "        x = BasicBlock(channels, stride, use_projection=use_projection)(x, train=train)",
  "        for _ in range(1, blocks):",
  "            x = BasicBlock(channels)(x, train=train)",
  "        return x",
  "",
  "",
  "model = ResNet18(num_classes=1000)",
  "test_input = jnp.ones((2, 224, 224, 3))",
  "params = model.init(jax.random.PRNGKey(0), test_input, train=False)",
  "logits = model.apply(params, test_input, train=False)",
  "# logits: (2, 1000)",
];

function makeTransformerEncoderBlock(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `encoder.${index}`,
    label: `encoder.${index}`,
    type: "EncoderLayer",
    kind: "group",
    summary: "self-attn + ffn",
    defaultExpanded,
    codeLines: [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 91, 92],
    lazyChildren: () => [
      {
        id: `encoder.${index}.self_attn`,
        label: "self_attn",
        type: "MultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "d=512"],
        codeLines: [32, 42],
      },
      {
        id: `encoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [38, 43, 44],
      },
      {
        id: `encoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        codeLines: [33, 34, 35, 36, 37, 45],
      },
      {
        id: `encoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [39, 46, 47, 48],
      },
    ],
  };
}

function makeRnnStep(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `step.${index}`,
    label: `step.${index}`,
    type: "RecurrentCell",
    kind: "group",
    summary: index === 0 ? "x_t + h_{t-1}" : "same weights",
    defaultExpanded,
    codeLines: [38, 40, 42, 44, 46, 48, 50],
    lazyChildren: () => [
      {
        id: `step.${index}.input_to_hidden`,
        label: "input_to_hidden",
        type: "Linear",
        kind: "linear",
        badges: ["32->64"],
        codeLines: [13, 15, 42],
      },
      {
        id: `step.${index}.hidden_to_hidden`,
        label: "hidden_to_hidden",
        type: "RecurrentLinear",
        kind: "recurrent",
        badges: ["64->64", "shared"],
        codeLines: [17, 19, 44],
      },
      {
        id: `step.${index}.update`,
        label: "add + tanh",
        type: "StateUpdate",
        kind: "activation",
        badges: ["h_t"],
        codeLines: [46, 48],
      },
      {
        id: `step.${index}.state`,
        label: "state history",
        type: "AppendHidden",
        kind: "recurrent",
        badges: ["store h_t"],
        codeLines: [34, 50, 54],
      },
    ],
  };
}

function makeGruStep(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `step.${index}`,
    label: `step.${index}`,
    type: "GRUCell",
    kind: "group",
    summary: index === 0 ? "z/r/n gates" : "same gates",
    defaultExpanded,
    codeLines: [15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 51, 52, 53, 54],
    lazyChildren: () => [
      {
        id: `step.${index}.update_gate`,
        label: "update gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["z_t"],
        codeLines: [8, 9, 16, 17, 18, 19],
      },
      {
        id: `step.${index}.reset_gate`,
        label: "reset gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["r_t"],
        codeLines: [10, 11, 21, 22, 23, 24],
      },
      {
        id: `step.${index}.candidate`,
        label: "candidate",
        type: "TanhState",
        kind: "activation",
        badges: ["n_t"],
        codeLines: [12, 13, 26, 27, 28, 29, 30],
      },
      {
        id: `step.${index}.mix`,
        label: "state mix",
        type: "GatedInterpolation",
        kind: "recurrent",
        badges: ["h_t"],
        codeLines: [32, 33, 34, 35],
      },
    ],
  };
}

function makeTransformerDecoderBlock(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `decoder.${index}`,
    label: `decoder.${index}`,
    type: "DecoderLayer",
    kind: "group",
    summary: "masked + cross + ffn",
    defaultExpanded,
    codeLines: [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 95, 96],
    lazyChildren: () => [
      {
        id: `decoder.${index}.masked_self_attn`,
        label: "masked self_attn",
        type: "CausalMultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "causal"],
        codeLines: [54, 66, 104, 105, 106],
      },
      {
        id: `decoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [61, 67, 68],
      },
      {
        id: `decoder.${index}.cross_attn`,
        label: "cross_attn",
        type: "EncoderDecoderAttention",
        kind: "attention",
        badges: ["Q=decoder", "K,V=encoder"],
        codeLines: [55, 69],
      },
      {
        id: `decoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [62, 70, 71],
      },
      {
        id: `decoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        codeLines: [56, 57, 58, 59, 60, 72],
      },
      {
        id: `decoder.${index}.norm3`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [63, 73, 74, 75],
      },
    ],
  };
}

function makeBertLayer(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `encoder.layer.${index}`,
    label: `layer.${index}`,
    type: "BertLayer",
    kind: "group",
    summary: "self-attn + ffn",
    defaultExpanded,
    codeLines: [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 62, 63],
    lazyChildren: () => [
      {
        id: `encoder.layer.${index}.self_attn`,
        label: "self_attn",
        type: "BidirectionalSelfAttention",
        kind: "attention",
        badges: ["12 heads", "768"],
        codeLines: [30, 41],
      },
      {
        id: `encoder.layer.${index}.attn_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [31, 42, 43, 44],
      },
      {
        id: `encoder.layer.${index}.intermediate`,
        label: "intermediate",
        type: "Dense + GELU",
        kind: "mlp",
        badges: ["768->3072"],
        codeLines: [32, 33, 34, 35, 36, 45],
      },
      {
        id: `encoder.layer.${index}.output`,
        label: "output",
        type: "Dense",
        kind: "mlp",
        badges: ["3072->768"],
        codeLines: [32, 33, 34, 35, 36, 45],
      },
      {
        id: `encoder.layer.${index}.output_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [37, 46, 47, 48, 49],
      },
    ],
  };
}

function makeGpt2Block(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `block.${index}`,
    label: `block.${index}`,
    type: "TransformerBlock",
    kind: "group",
    summary: "ln + attn + mlp",
    defaultExpanded,
    codeLines: [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 24, 25],
    lazyChildren: () => [
      {
        id: `block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        codeLines: [65, 75],
      },
      {
        id: `block.${index}.attn`,
        label: "attn",
        type: "CausalSelfAttention",
        kind: "attention",
        summary: "12 heads",
        codeLines: [30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 76],
        lazyChildren: () => [
          {
            id: `block.${index}.attn.c_attn`,
            label: "c_attn",
            type: "QKV Projection",
            kind: "attention",
            badges: ["768->2304"],
            codeLines: [34, 39, 40],
          },
          {
            id: `block.${index}.attn.heads`,
            label: "heads",
            type: "Head grid",
            kind: "group",
            summary: "12 x dim 64",
            codeLines: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
            lazyChildren: () =>
              Array.from({ length: 12 }, (_, headIndex) => ({
                id: `block.${index}.attn.head.${headIndex}`,
                label: `head.${headIndex}`,
                type: "AttentionHead",
                kind: "head" as NodeKind,
                badges: ["q,k,v", "dim 64"],
                codeLines: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
              })),
          },
          {
            id: `block.${index}.attn.merge`,
            label: "merge",
            type: "Concat heads",
            kind: "attention",
            badges: ["12 x 64 -> 768"],
            codeLines: [56, 57, 58],
          },
          {
            id: `block.${index}.attn.c_proj`,
            label: "c_proj",
            type: "Output Projection",
            kind: "attention",
            badges: ["768->768"],
            codeLines: [35, 59, 60],
          },
        ],
      },
      {
        id: `block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        codeLines: [77],
      },
      {
        id: `block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        codeLines: [67, 78],
      },
      {
        id: `block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        summary: "3072 hidden",
        codeLines: [68, 69, 70, 71, 72, 79],
      },
      {
        id: `block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        codeLines: [80, 81],
      },
    ],
  };
}

type InceptionNodeConfig = {
  id: string;
  label: string;
  inputChannels: number;
  branch1Channels: number;
  branch3Reduce: number;
  branch3Channels: number;
  branch5Reduce: number;
  branch5Channels: number;
  poolChannels: number;
  callLine: number;
  forwardLine: number;
  defaultExpanded?: boolean;
};

function makeInceptionNode(config: InceptionNodeConfig): ArchNode {
  const outputChannels =
    config.branch1Channels +
    config.branch3Channels +
    config.branch5Channels +
    config.poolChannels;

  return {
    id: config.id,
    label: config.label,
    type: "InceptionBlock",
    kind: "group",
    summary: "parallel branches",
    badges: [`${config.inputChannels}->${outputChannels}`],
    defaultExpanded: config.defaultExpanded,
    codeLines: [8, 9, 12, 13, 15, 18, 19, 21, 24, 25, 26, 31, 32, 33, 34, 35, 36, config.callLine, config.forwardLine],
    lazyChildren: () => [
      {
        id: `${config.id}.branch1`,
        label: "branch1",
        type: "1x1 Conv",
        kind: "conv",
        badges: [`${config.inputChannels}->${config.branch1Channels}`],
        codeLines: [8, 9, 31, config.callLine, config.forwardLine],
      },
      {
        id: `${config.id}.branch3`,
        label: "branch3",
        type: "1x1 reduce + 3x3",
        kind: "group",
        summary: "medium receptive field",
        codeLines: [12, 13, 15, 16, 32, config.callLine, config.forwardLine],
        children: [
          {
            id: `${config.id}.branch3.reduce`,
            label: "reduce",
            type: "1x1 Conv",
            kind: "conv",
            badges: [`${config.inputChannels}->${config.branch3Reduce}`],
            codeLines: [12, 13],
          },
          {
            id: `${config.id}.branch3.conv`,
            label: "conv3x3",
            type: "3x3 Conv",
            kind: "conv",
            badges: [`${config.branch3Reduce}->${config.branch3Channels}`],
            codeLines: [15, 16],
          },
        ],
      },
      {
        id: `${config.id}.branch5`,
        label: "branch5",
        type: "1x1 reduce + 5x5",
        kind: "group",
        summary: "wide receptive field",
        codeLines: [18, 19, 21, 22, 33, config.callLine, config.forwardLine],
        children: [
          {
            id: `${config.id}.branch5.reduce`,
            label: "reduce",
            type: "1x1 Conv",
            kind: "conv",
            badges: [`${config.inputChannels}->${config.branch5Reduce}`],
            codeLines: [18, 19],
          },
          {
            id: `${config.id}.branch5.conv`,
            label: "conv5x5",
            type: "5x5 Conv",
            kind: "conv",
            badges: [`${config.branch5Reduce}->${config.branch5Channels}`],
            codeLines: [21, 22],
          },
        ],
      },
      {
        id: `${config.id}.pool_proj`,
        label: "pool_proj",
        type: "3x3 Pool + 1x1 Conv",
        kind: "pool",
        badges: [`${config.inputChannels}->${config.poolChannels}`],
        codeLines: [24, 25, 26, 27, 34, config.callLine, config.forwardLine],
      },
      {
        id: `${config.id}.concat`,
        label: "concat",
        type: "ChannelConcat",
        kind: "concat",
        badges: [`${outputChannels} channels`],
        codeLines: [35, 36],
      },
    ],
  };
}

function makeVitBlock(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `encoder.block.${index}`,
    label: `block.${index}`,
    type: "EncoderBlock",
    kind: "group",
    summary: "self-attn + mlp",
    defaultExpanded,
    codeLines: [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 57, 58],
    lazyChildren: () => [
      {
        id: `encoder.block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        codeLines: [22, 32],
      },
      {
        id: `encoder.block.${index}.attn`,
        label: "attn",
        type: "MultiHeadSelfAttention",
        kind: "attention",
        badges: ["12 heads", "197 tokens"],
        codeLines: [23, 33],
      },
      {
        id: `encoder.block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        codeLines: [34],
      },
      {
        id: `encoder.block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        codeLines: [24, 35],
      },
      {
        id: `encoder.block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        badges: ["768->3072->768"],
        codeLines: [25, 26, 27, 28, 29, 36],
      },
      {
        id: `encoder.block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        codeLines: [37],
      },
    ],
  };
}

const models: ModelSpec[] = [
  {
    id: "mlp",
    label: "MLP",
    breadcrumb: "MLP / hidden.1 / dense",
    stats: "2 hidden layers · sigmoid activations · backprop",
    fileName: "mlp.py",
    jaxFileName: "mlp_jax.py",
    paper: {
      title: "Learning representations by back-propagating errors",
      authors: "David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams",
      year: "1986",
      publishedLabel: "Oct 9, 1986",
      publishedDate: "1986-10-09",
      venue: "Nature",
      url: "https://www.nature.com/articles/323533a0",
      pdfUrl: "/papers/mlp.pdf",
      focus: ["backpropagation", "hidden representations", "multilayer perceptrons"],
    },
    selectedId: "hidden.1.dense",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "FlatVector",
        kind: "input",
        badges: ["784 features"],
        codeLines: [9, 34, 36, 37],
      },
      {
        id: "hidden.1",
        label: "hidden.1",
        type: "HiddenLayer",
        kind: "group",
        summary: "dense + sigmoid",
        badges: ["784->128"],
        codeLines: [10, 15, 16, 21, 22, 23],
        children: [
          {
            id: "hidden.1.dense",
            label: "dense",
            type: "Linear",
            kind: "linear",
            badges: ["784->128"],
            codeLines: [10, 15, 16, 22],
          },
          {
            id: "hidden.1.sigmoid",
            label: "sigmoid",
            type: "Activation",
            kind: "activation",
            codeLines: [21, 23],
          },
        ],
      },
      {
        id: "hidden.2",
        label: "hidden.2",
        type: "HiddenLayer",
        kind: "group",
        summary: "dense + sigmoid",
        badges: ["128->128"],
        codeLines: [10, 15, 17, 25, 26, 27],
        children: [
          {
            id: "hidden.2.dense",
            label: "dense",
            type: "Linear",
            kind: "linear",
            badges: ["128->128"],
            codeLines: [10, 15, 17, 26],
          },
          {
            id: "hidden.2.sigmoid",
            label: "sigmoid",
            type: "Activation",
            kind: "activation",
            codeLines: [25, 27],
          },
        ],
      },
      {
        id: "output",
        label: "output",
        type: "Linear",
        kind: "linear",
        badges: ["128->10"],
        codeLines: [11, 15, 18, 29, 30],
      },
      {
        id: "logits",
        label: "logits",
        type: "ClassScores",
        kind: "head",
        badges: ["10 classes"],
        codeLines: [30, 31, 34, 37, 39],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "",
      "",
      "# Define a multilayer perceptron.",
      "class MLP(nn.Module):",
      "    def __init__(",
      "        self,",
      "        input_dim=784,  # Number of input dimensions.",
      "        hidden_dim=128,  # Number of hidden dimensions.",
      "        output_dim=10  # Number of output dimensions.",
      "    ):",
      "        super().__init__()",
      "",
      "        # Register affine projections; initialization does not transform tensors.",
      "        self.hidden1 = nn.Linear(input_dim, hidden_dim)",
      "        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)",
      "        self.output = nn.Linear(hidden_dim, output_dim)",
      "",
      "    def forward(self, x):",
      "        # First hidden block: (batch, input_dim) -> (batch, hidden_dim).",
      "        h1_pre = self.hidden1(x)",
      "        h1 = torch.sigmoid(h1_pre)",
      "",
      "        # Second hidden block keeps the hidden shape: (batch, hidden_dim).",
      "        h2_pre = self.hidden2(h1)",
      "        h2 = torch.sigmoid(h2_pre)",
      "",
      "        # Output block: (batch, hidden_dim) -> (batch, output_dim).",
      "        logits = self.output(h2)",
      "        return logits",
      "",
      "",
      "# Create and run a sample batch: (2, 784) -> (2, 10).",
      "model = MLP(input_dim=784, hidden_dim=128, output_dim=10)",
      "inputs = torch.randn(2, 784)",
      "logits = model(inputs)",
      "",
      "# logits: (2, 10)",
    ],
    jaxCode: mlpJaxCode,
  },
  {
    id: "rnn",
    label: "RNN (Elman)",
    breadcrumb: "RNN / recurrent loop / step.0 / hidden_to_hidden",
    stats: "8 time steps · 64 hidden units · shared recurrent cell",
    fileName: "elman_rnn.py",
    jaxFileName: "elman_rnn_jax.py",
    paper: {
      title: "Finding Structure in Time",
      authors: "Jeffrey L. Elman",
      year: "1990",
      publishedLabel: "Mar 1990",
      publishedDate: "1990-03-01",
      venue: "Cognitive Science",
      url: "https://doi.org/10.1207/s15516709cog1402_1",
      pdfUrl: "/papers/rnn.pdf",
      focus: ["recurrent hidden state", "dynamic memory", "sequence structure"],
    },
    selectedId: "step.0.hidden_to_hidden",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        codeLines: [26, 28, 36, 40, 64, 66],
      },
      {
        id: "recurrent_loop",
        label: "Recurrent Loop",
        type: "UnrolledRNN",
        kind: "group",
        summary: "shared cell over time",
        badges: ["tanh"],
        defaultExpanded: true,
        codeLines: [32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 54],
        children: [
          {
            id: "h0",
            label: "h0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["64 hidden"],
            codeLines: [28, 30, 32],
          },
          ...Array.from({ length: 8 }, (_, index) => makeRnnStep(index, index === 0)),
        ],
      },
      {
        id: "readout",
        label: "readout",
        type: "Linear",
        kind: "linear",
        badges: ["64->10", "last h"],
        codeLines: [21, 23, 52],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "all states"],
        codeLines: [52, 54, 56, 58, 66, 68, 70, 71],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "",
      "",
      "# Define an Elman recurrent model.",
      "class ElmanRNN(nn.Module):",
      "    # Build the recurrent layers.",
      "    def __init__(self, input_size=32, hidden_size=64, output_size=10):",
      "        super().__init__()",
      "        # Store hidden width: scalar -> scalar.",
      "        self.hidden_size = hidden_size",
      "        # Create input projection: (batch, input_size) -> (batch, hidden_size).",
      "        input_to_hidden = nn.Linear(input_size, hidden_size)",
      "        # Attach input projection to the module.",
      "        self.input_to_hidden = input_to_hidden",
      "        # Create recurrent projection: (batch, hidden_size) -> (batch, hidden_size).",
      "        hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)",
      "        # Attach recurrent projection to the module.",
      "        self.hidden_to_hidden = hidden_to_hidden",
      "        # Create output projection: (batch, hidden_size) -> (batch, output_size).",
      "        hidden_to_output = nn.Linear(hidden_size, output_size)",
      "        # Attach output projection to the module.",
      "        self.hidden_to_output = hidden_to_output",
      "",
      "    # Run the sequence through the recurrent cell.",
      "    def forward(self, x):",
      "        # Read batch size: (batch, steps, input_size) -> scalar.",
      "        batch_size = x.size(0)",
      "        # Build hidden shape: scalar -> (batch, hidden_size).",
      "        hidden_shape = (batch_size, self.hidden_size)",
      "        # Create initial hidden state: (batch, hidden_size).",
      "        h = torch.zeros(hidden_shape, device=x.device)",
      "        # Create state history: list of (batch, hidden_size).",
      "        states = []",
      "        # Read step count: (batch, steps, input_size) -> scalar.",
      "        step_count = x.size(1)",
      "        # Iterate over time steps.",
      "        for t in range(step_count):",
      "            # Select current input: (batch, steps, input_size) -> (batch, input_size).",
      "            current_input = x[:, t]",
      "            # Project current input: (batch, input_size) -> (batch, hidden_size).",
      "            input_hidden = self.input_to_hidden(current_input)",
      "            # Project previous state: (batch, hidden_size) -> (batch, hidden_size).",
      "            recurrent_hidden = self.hidden_to_hidden(h)",
      "            # Add projections: (batch, hidden_size) -> (batch, hidden_size).",
      "            hidden_sum = input_hidden + recurrent_hidden",
      "            # Apply tanh: (batch, hidden_size) -> (batch, hidden_size).",
      "            h = torch.tanh(hidden_sum)",
      "            # Append current state: (batch, hidden_size) -> list.",
      "            states.append(h)",
      "        # Project final state: (batch, hidden_size) -> (batch, output_size).",
      "        logits = self.hidden_to_output(h)",
      "        # Stack state history: list -> (batch, steps, hidden_size).",
      "        state_trace = torch.stack(states, dim=1)",
      "        # Pack outputs: logits (batch, output_size), states (batch, steps, hidden_size).",
      "        outputs = (logits, state_trace)",
      "        # Return model outputs.",
      "        return outputs",
      "",
      "",
      "# Create the model.",
      "model = ElmanRNN(input_size=32, hidden_size=64, output_size=10)",
      "# Create an input batch: (batch, steps, input_size).",
      "sequence = torch.randn(2, 8, 32)",
      "# Run the model: (2, 8, 32) -> tuple.",
      "outputs = model(sequence)",
      "# Select logits: tuple -> (2, 10).",
      "logits = outputs[0]",
      "# Select states: tuple -> (2, 8, 64).",
      "states = outputs[1]",
      "# logits: (2, 10), states: (2, 8, 64)",
    ],
    jaxCode: rnnJaxCode,
  },
  {
    id: "gru",
    label: "GRU",
    breadcrumb: "GRU / recurrent loop / step.0 / update gate",
    stats: "8 time steps · update/reset gates · 64 hidden units",
    fileName: "gru.py",
    jaxFileName: "gru_jax.py",
    paper: {
      title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
      authors: "Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio",
      year: "2014",
      publishedLabel: "Jun 3, 2014",
      publishedDate: "2014-06-03",
      venue: "arXiv / EMNLP 2014",
      url: "https://arxiv.org/abs/1406.1078",
      pdfUrl: "/papers/gru.pdf",
      focus: ["update gate", "reset gate", "encoder-decoder sequence modeling"],
    },
    selectedId: "step.0.update_gate",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        codeLines: [29, 38, 39],
      },
      {
        id: "cell_params",
        label: "GRU Cell Params",
        type: "GatedRecurrentCell",
        kind: "group",
        summary: "6 affine projections",
        badges: ["z", "r", "n"],
        defaultExpanded: true,
        codeLines: [5, 6, 8, 9, 10, 11, 12, 13, 25],
        children: [
          {
            id: "cell_params.update",
            label: "update params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_z", "h_z"],
            codeLines: [8, 9, 16],
          },
          {
            id: "cell_params.reset",
            label: "reset params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_r", "h_r"],
            codeLines: [10, 11, 17],
          },
          {
            id: "cell_params.candidate",
            label: "candidate params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_n", "h_n"],
            codeLines: [12, 13, 18],
          },
        ],
      },
      {
        id: "recurrent_loop",
        label: "Recurrent Loop",
        type: "UnrolledGRU",
        kind: "group",
        summary: "shared gated cell",
        defaultExpanded: true,
        codeLines: [28, 29, 30, 31, 32, 33, 35],
        children: [
          {
            id: "h0",
            label: "h0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["64 hidden"],
            codeLines: [29],
          },
          ...Array.from({ length: 8 }, (_, index) => makeGruStep(index, index === 0)),
        ],
      },
      {
        id: "readout",
        label: "readout",
        type: "Linear",
        kind: "linear",
        badges: ["64->10", "last h"],
        codeLines: [26, 34],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "all states"],
        codeLines: [34, 35, 40, 41],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "",
      "",
      "class GRUCell(nn.Module):",
      "    def __init__(self, input_size=32, hidden_size=64):",
      "        super().__init__()",
      "        self.x_z = nn.Linear(input_size, hidden_size)",
      "        self.h_z = nn.Linear(hidden_size, hidden_size, bias=False)",
      "        self.x_r = nn.Linear(input_size, hidden_size)",
      "        self.h_r = nn.Linear(hidden_size, hidden_size, bias=False)",
      "        self.x_n = nn.Linear(input_size, hidden_size)",
      "        self.h_n = nn.Linear(hidden_size, hidden_size, bias=False)",
      "",
      "    def forward(self, x, h):",
      "        x_z = self.x_z(x)",
      "        h_z = self.h_z(h)",
      "        z_pre = x_z + h_z",
      "        z = torch.sigmoid(z_pre)",
      "",
      "        x_r = self.x_r(x)",
      "        h_r = self.h_r(h)",
      "        r_pre = x_r + h_r",
      "        r = torch.sigmoid(r_pre)",
      "",
      "        reset_h = r * h",
      "        x_n = self.x_n(x)",
      "        h_n = self.h_n(reset_h)",
      "        n_pre = x_n + h_n",
      "        n = torch.tanh(n_pre)",
      "",
      "        keep_h = z * h",
      "        candidate_h = (1 - z) * n",
      "        h_next = candidate_h + keep_h",
      "        return h_next",
      "",
      "",
      "class GRUSequence(nn.Module):",
      "    def __init__(self, input_size=32, hidden_size=64, output_size=10):",
      "        super().__init__()",
      "        self.hidden_size = hidden_size",
      "        self.cell = GRUCell(input_size, hidden_size)",
      "        self.readout = nn.Linear(hidden_size, output_size)",
      "",
      "    def forward(self, x):",
      "        batch_size = x.size(0)",
      "        hidden_shape = (batch_size, self.hidden_size)",
      "        h = torch.zeros(hidden_shape, device=x.device)",
      "        states = []",
      "        step_count = x.size(1)",
      "        for t in range(step_count):",
      "            current_input = x[:, t]",
      "            h = self.cell(current_input, h)",
      "            states.append(h)",
      "        logits = self.readout(h)",
      "        state_trace = torch.stack(states, dim=1)",
      "        outputs = (logits, state_trace)",
      "        return outputs",
      "",
      "",
      "model = GRUSequence(input_size=32, hidden_size=64, output_size=10)",
      "sequence = torch.randn(2, 8, 32)",
      "logits, states = model(sequence)",
      "# logits: (2, 10), states: (2, 8, 64)",
    ],
    jaxCode: gruJaxCode,
  },
  {
    id: "lstm",
    label: "LSTM",
    breadcrumb: "LSTM / recurrent loop / step.0 / forget gate",
    stats: "8 time steps · input/forget/output gates · cell state",
    fileName: "lstm.py",
    jaxFileName: "lstm_jax.py",
    paper: {
      title: "Long Short-Term Memory",
      authors: "Sepp Hochreiter, Jurgen Schmidhuber",
      year: "1997",
      publishedLabel: "Nov 1, 1997",
      publishedDate: "1997-11-01",
      venue: "Neural Computation",
      url: "https://doi.org/10.1162/neco.1997.9.8.1735",
      pdfUrl: "/papers/lstm.pdf",
      focus: ["cell state memory", "input/forget/output gates", "long-range dependencies"],
    },
    selectedId: "step.0.forget_gate",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        codeLines: [36, 45, 46],
      },
      {
        id: "cell_params",
        label: "LSTM Cell Params",
        type: "GatedMemoryCell",
        kind: "group",
        summary: "8 affine projections",
        badges: ["i", "f", "g", "o"],
        defaultExpanded: true,
        codeLines: [5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 32],
        children: [
          {
            id: "cell_params.input",
            label: "input params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_i", "h_i"],
            codeLines: [8, 9, 19],
          },
          {
            id: "cell_params.forget",
            label: "forget params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_f", "h_f"],
            codeLines: [10, 11, 20],
          },
          {
            id: "cell_params.candidate",
            label: "candidate params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_g", "h_g"],
            codeLines: [12, 13, 21],
          },
          {
            id: "cell_params.output",
            label: "output params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_o", "h_o"],
            codeLines: [14, 15, 22],
          },
        ],
      },
      {
        id: "recurrent_loop",
        label: "Recurrent Loop",
        type: "UnrolledLSTM",
        kind: "group",
        summary: "hidden + cell state",
        defaultExpanded: true,
        codeLines: [35, 36, 37, 38, 39, 40, 42],
        children: [
          {
            id: "state0",
            label: "h0 + c0",
            type: "ZeroStates",
            kind: "recurrent",
            badges: ["64 hidden", "64 cell"],
            codeLines: [36, 37],
          },
          {
            id: "step.0",
            label: "step.0",
            type: "LSTMCell",
            kind: "group",
            summary: "i/f/g/o gates",
            defaultExpanded: true,
            codeLines: [17, 18, 19, 20, 21, 22, 23, 24, 25, 40],
            children: [
              {
                id: "step.0.input_gate",
                label: "input gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["i_t"],
                codeLines: [8, 9, 19],
              },
              {
                id: "step.0.forget_gate",
                label: "forget gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["f_t"],
                codeLines: [10, 11, 20],
              },
              {
                id: "step.0.candidate",
                label: "candidate",
                type: "TanhState",
                kind: "activation",
                badges: ["g_t"],
                codeLines: [12, 13, 21],
              },
              {
                id: "step.0.cell_update",
                label: "cell update",
                type: "MemoryUpdate",
                kind: "recurrent",
                badges: ["c_t"],
                codeLines: [23],
              },
              {
                id: "step.0.output_gate",
                label: "output gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["o_t"],
                codeLines: [14, 15, 22],
              },
              {
                id: "step.0.hidden_update",
                label: "hidden update",
                type: "GatedReadout",
                kind: "recurrent",
                badges: ["h_t"],
                codeLines: [24, 25],
              },
            ],
          },
          ...[1, 2, 3, 4, 5, 6, 7].map((index) => ({
            id: `step.${index}`,
            label: `step.${index}`,
            type: "LSTMCell",
            kind: "recurrent" as NodeKind,
            summary: "same gates",
            codeLines: [39, 40],
          })),
        ],
      },
      {
        id: "readout",
        label: "readout",
        type: "Linear",
        kind: "linear",
        badges: ["64->10", "last h"],
        codeLines: [33, 42],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "hidden states"],
        codeLines: [42, 43, 47, 48],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "",
      "",
      "class LSTMCell(nn.Module):",
      "    def __init__(self, input_size=32, hidden_size=64):",
      "        super().__init__()",
      "        self.x_i = nn.Linear(input_size, hidden_size)",
      "        self.h_i = nn.Linear(hidden_size, hidden_size, bias=False)",
      "        self.x_f = nn.Linear(input_size, hidden_size)",
      "        self.h_f = nn.Linear(hidden_size, hidden_size, bias=False)",
      "        self.x_g = nn.Linear(input_size, hidden_size)",
      "        self.h_g = nn.Linear(hidden_size, hidden_size, bias=False)",
      "        self.x_o = nn.Linear(input_size, hidden_size)",
      "        self.h_o = nn.Linear(hidden_size, hidden_size, bias=False)",
      "",
      "    def forward(self, x, state):",
      "        h, c = state",
      "        x_i = self.x_i(x)",
      "        h_i = self.h_i(h)",
      "        i_pre = x_i + h_i",
      "        i = torch.sigmoid(i_pre)",
      "",
      "        x_f = self.x_f(x)",
      "        h_f = self.h_f(h)",
      "        f_pre = x_f + h_f",
      "        f = torch.sigmoid(f_pre)",
      "",
      "        x_g = self.x_g(x)",
      "        h_g = self.h_g(h)",
      "        g_pre = x_g + h_g",
      "        g = torch.tanh(g_pre)",
      "",
      "        x_o = self.x_o(x)",
      "        h_o = self.h_o(h)",
      "        o_pre = x_o + h_o",
      "        o = torch.sigmoid(o_pre)",
      "",
      "        forget_c = f * c",
      "        write_c = i * g",
      "        c_next = forget_c + write_c",
      "        c_readout = torch.tanh(c_next)",
      "        h_next = o * c_readout",
      "        return h_next, c_next",
      "",
      "",
      "class LSTMSequence(nn.Module):",
      "    def __init__(self, input_size=32, hidden_size=64, output_size=10):",
      "        super().__init__()",
      "        self.hidden_size = hidden_size",
      "        self.cell = LSTMCell(input_size, hidden_size)",
      "        self.readout = nn.Linear(hidden_size, output_size)",
      "",
      "    def forward(self, x):",
      "        batch_size = x.size(0)",
      "        hidden_shape = (batch_size, self.hidden_size)",
      "        h = torch.zeros(hidden_shape, device=x.device)",
      "        c = torch.zeros(hidden_shape, device=x.device)",
      "        states = []",
      "        step_count = x.size(1)",
      "        for t in range(step_count):",
      "            current_input = x[:, t]",
      "            state = (h, c)",
      "            h, c = self.cell(current_input, state)",
      "            states.append(h)",
      "        logits = self.readout(h)",
      "        state_trace = torch.stack(states, dim=1)",
      "        outputs = (logits, state_trace)",
      "        return outputs",
      "",
      "",
      "model = LSTMSequence(input_size=32, hidden_size=64, output_size=10)",
      "sequence = torch.randn(2, 8, 32)",
      "logits, states = model(sequence)",
      "# logits: (2, 10), states: (2, 8, 64)",
    ],
    jaxCode: lstmJaxCode,
  },
  {
    id: "lenet5",
    label: "LeNet-5",
    breadcrumb: "LeNet-5 / Feature Extractor / conv1",
    stats: "3 groups · 11 ops",
    fileName: "lenet5.py",
    jaxFileName: "lenet5_jax.py",
    paper: {
      title: "Gradient-Based Learning Applied to Document Recognition",
      authors: "Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner",
      year: "1998",
      publishedLabel: "Nov 1998",
      publishedDate: "1998-11-01",
      venue: "Proceedings of the IEEE",
      url: "https://ieeexplore.ieee.org/document/726791",
      pdfUrl: "/papers/lenet5.pdf",
      focus: ["convolutional feature maps", "subsampling", "document recognition"],
    },
    selectedId: "features.conv1",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["1 x 32 x 32"],
        codeLines: [17],
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "Group",
        kind: "group",
        summary: "6 ops",
        defaultExpanded: true,
        codeLines: [9, 10, 16, 17, 18, 19, 20, 21],
        children: [
          {
            id: "features.conv1",
            label: "conv1",
            type: "Conv2d",
            kind: "conv",
            badges: ["1->6", "k=5", "out 6x28x28"],
            codeLines: [9, 18],
          },
          {
            id: "features.tanh1",
            label: "tanh1",
            type: "Tanh",
            kind: "activation",
            codeLines: [18],
          },
          {
            id: "features.pool1",
            label: "pool1",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["k=2", "out 6x14x14"],
            codeLines: [19],
          },
          {
            id: "features.conv2",
            label: "conv2",
            type: "Conv2d",
            kind: "conv",
            badges: ["6->16", "k=5", "out 16x10x10"],
            codeLines: [10, 20],
          },
          {
            id: "features.tanh2",
            label: "tanh2",
            type: "Tanh",
            kind: "activation",
            codeLines: [20],
          },
          {
            id: "features.pool2",
            label: "pool2",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["k=2", "out 16x5x5"],
            codeLines: [21],
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["400"],
        codeLines: [22],
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "Group",
        kind: "group",
        summary: "4 ops",
        defaultExpanded: true,
        codeLines: [12, 13, 14, 24, 25, 26],
        children: [
          {
            id: "classifier.fc1",
            label: "fc1",
            type: "Linear",
            kind: "linear",
            badges: ["400->120"],
            codeLines: [11, 24],
          },
          {
            id: "classifier.tanh3",
            label: "tanh3",
            type: "Tanh",
            kind: "activation",
            codeLines: [24],
          },
          {
            id: "classifier.fc2",
            label: "fc2",
            type: "Linear",
            kind: "linear",
            badges: ["120->84"],
            codeLines: [12, 25],
          },
          {
            id: "classifier.output",
            label: "output",
            type: "Linear",
            kind: "linear",
            badges: ["84->10"],
            codeLines: [14, 26],
          },
        ],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      "",
      "class LeNet5(nn.Module):",
      "    def __init__(self):",
      "        super().__init__()",
      "        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)",
      "        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)",
      "",
      "        flattened_features = 16 * 5 * 5",
      "        self.fc1 = nn.Linear(flattened_features, 120)",
      "        self.fc2 = nn.Linear(120, 84)",
      "        self.output = nn.Linear(84, 10)",
      "",
      "    def forward(self, x):",
      "        # x: (batch, 1, 32, 32)",
      "        conv1 = self.conv1(x)",
      "        x = torch.tanh(conv1)",
      "        x = F.avg_pool2d(x, kernel_size=2)",
      "        conv2 = self.conv2(x)",
      "        x = torch.tanh(conv2)",
      "        x = F.avg_pool2d(x, kernel_size=2)",
      "        x = torch.flatten(x, start_dim=1)",
      "",
      "        fc1 = self.fc1(x)",
      "        x = torch.tanh(fc1)",
      "        fc2 = self.fc2(x)",
      "        x = torch.tanh(fc2)",
      "        logits = self.output(x)",
      "        return logits",
      "",
      "",
      "model = LeNet5()",
      "test_input = torch.randn(2, 1, 32, 32)",
      "logits = model(test_input)",
      "# logits: (2, 10)",
    ],
    jaxCode: lenet5JaxCode,
  },
  {
    id: "alexnet",
    label: "AlexNet",
    breadcrumb: "AlexNet / features / conv1",
    stats: "5 conv layers · 3 FC layers · 60M params",
    fileName: "alexnet.py",
    jaxFileName: "alexnet_jax.py",
    paper: {
      title: "ImageNet Classification with Deep Convolutional Neural Networks",
      authors: "Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton",
      year: "2012",
      publishedLabel: "Dec 2012",
      publishedDate: "2012-12-03",
      venue: "NeurIPS 2012",
      url: "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks",
      pdfUrl: "/papers/alexnet.pdf",
      focus: ["large-scale CNNs", "ReLU activations", "GPU training"],
    },
    selectedId: "features.conv1",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["3 x 227 x 227"],
        codeLines: [43],
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "Group",
        kind: "group",
        summary: "13 ops",
        defaultExpanded: true,
        codeLines: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 37],
        children: [
          {
            id: "features.conv1",
            label: "conv1",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->96", "k=11", "s=4", "55x55"],
            codeLines: [10, 37],
          },
          {
            id: "features.relu1",
            label: "relu1",
            type: "ReLU",
            kind: "activation",
            codeLines: [11],
          },
          {
            id: "features.lrn1",
            label: "lrn1",
            type: "LocalResponseNorm",
            kind: "norm",
            badges: ["size=5"],
            codeLines: [12],
          },
          {
            id: "features.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "27x27"],
            codeLines: [13],
          },
          {
            id: "features.conv2",
            label: "conv2",
            type: "Conv2d",
            kind: "conv",
            badges: ["96->256", "k=5", "27x27"],
            codeLines: [14],
          },
          {
            id: "features.relu2",
            label: "relu2",
            type: "ReLU",
            kind: "activation",
            codeLines: [15],
          },
          {
            id: "features.lrn2",
            label: "lrn2",
            type: "LocalResponseNorm",
            kind: "norm",
            badges: ["size=5"],
            codeLines: [16],
          },
          {
            id: "features.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "13x13"],
            codeLines: [17],
          },
          {
            id: "features.conv3",
            label: "conv3",
            type: "Conv2d",
            kind: "conv",
            badges: ["256->384", "k=3"],
            codeLines: [18],
          },
          {
            id: "features.relu3",
            label: "relu3",
            type: "ReLU",
            kind: "activation",
            codeLines: [19],
          },
          {
            id: "features.conv4",
            label: "conv4",
            type: "Conv2d",
            kind: "conv",
            badges: ["384->384", "k=3"],
            codeLines: [20],
          },
          {
            id: "features.relu4",
            label: "relu4",
            type: "ReLU",
            kind: "activation",
            codeLines: [21],
          },
          {
            id: "features.conv5",
            label: "conv5",
            type: "Conv2d",
            kind: "conv",
            badges: ["384->256", "k=3"],
            codeLines: [22],
          },
          {
            id: "features.relu5",
            label: "relu5",
            type: "ReLU",
            kind: "activation",
            codeLines: [23],
          },
          {
            id: "features.pool5",
            label: "pool5",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "6x6"],
            codeLines: [24],
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["9216"],
        codeLines: [38],
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "Group",
        kind: "group",
        summary: "7 ops",
        defaultExpanded: true,
        codeLines: [26, 27, 28, 29, 30, 31, 32, 33, 39],
        children: [
          {
            id: "classifier.drop1",
            label: "dropout1",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            codeLines: [27],
          },
          {
            id: "classifier.fc6",
            label: "fc6",
            type: "Linear",
            kind: "linear",
            badges: ["9216->4096"],
            codeLines: [28, 39],
          },
          {
            id: "classifier.relu6",
            label: "relu6",
            type: "ReLU",
            kind: "activation",
            codeLines: [29],
          },
          {
            id: "classifier.drop2",
            label: "dropout2",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            codeLines: [30],
          },
          {
            id: "classifier.fc7",
            label: "fc7",
            type: "Linear",
            kind: "linear",
            badges: ["4096->4096"],
            codeLines: [31],
          },
          {
            id: "classifier.relu7",
            label: "relu7",
            type: "ReLU",
            kind: "activation",
            codeLines: [32],
          },
          {
            id: "classifier.fc8",
            label: "fc8",
            type: "Linear",
            kind: "linear",
            badges: ["4096->1000"],
            codeLines: [33],
          },
        ],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      "",
      "class AlexNet(nn.Module):",
      "    def __init__(self, num_classes=1000):",
      "        super().__init__()",
      "        self.features = nn.Sequential(",
      "            nn.Conv2d(3, 96, kernel_size=11, stride=4),",
      "            nn.ReLU(inplace=True),",
      "            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),",
      "            nn.MaxPool2d(kernel_size=3, stride=2),",
      "            nn.Conv2d(96, 256, kernel_size=5, padding=2),",
      "            nn.ReLU(inplace=True),",
      "            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),",
      "            nn.MaxPool2d(kernel_size=3, stride=2),",
      "            nn.Conv2d(256, 384, kernel_size=3, padding=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.Conv2d(384, 384, kernel_size=3, padding=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.Conv2d(384, 256, kernel_size=3, padding=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.MaxPool2d(kernel_size=3, stride=2),",
      "        )",
      "        flattened_features = 256 * 6 * 6",
      "        self.classifier = nn.Sequential(",
      "            nn.Dropout(0.5),",
      "            nn.Linear(flattened_features, 4096),",
      "            nn.ReLU(inplace=True),",
      "            nn.Dropout(0.5),",
      "            nn.Linear(4096, 4096),",
      "            nn.ReLU(inplace=True),",
      "            nn.Linear(4096, num_classes),",
      "        )",
      "",
      "    def forward(self, x):",
      "        x = self.features(x)",
      "        x = torch.flatten(x, start_dim=1)",
      "        logits = self.classifier(x)",
      "        return logits",
      "",
      "",
      "model = AlexNet(num_classes=1000)",
      "test_input = torch.randn(2, 3, 227, 227)",
      "logits = model(test_input)",
      "# logits: (2, 1000)",
    ],
    jaxCode: alexnetJaxCode,
  },
  {
    id: "googlenet",
    label: "GoogLeNet / Inception v1",
    breadcrumb: "GoogLeNet / inception3a / concat",
    stats: "9 Inception blocks · parallel conv branches · 22 layers",
    fileName: "googlenet.py",
    jaxFileName: "googlenet_jax.py",
    paper: {
      title: "Going Deeper with Convolutions",
      authors: "Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich",
      year: "2014",
      publishedLabel: "Sep 17, 2014",
      publishedDate: "2014-09-17",
      venue: "arXiv / CVPR 2015",
      url: "https://arxiv.org/abs/1409.4842",
      pdfUrl: "/papers/googlenet.pdf",
      focus: ["Inception modules", "parallel convolutions", "channel concatenation"],
    },
    selectedId: "inception3a.concat",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [87],
      },
      {
        id: "stem",
        label: "stem",
        type: "ConvPoolStem",
        kind: "group",
        summary: "7x7 + 1x1 + 3x3",
        defaultExpanded: true,
        codeLines: [43, 44, 45, 46, 47, 48, 49, 50, 51, 67],
        children: [
          {
            id: "stem.conv7",
            label: "conv7",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->64", "k=7", "s=2"],
            codeLines: [44, 67],
          },
          {
            id: "stem.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            codeLines: [46, 67],
          },
          {
            id: "stem.conv1",
            label: "conv1x1",
            type: "Conv2d",
            kind: "conv",
            badges: ["64->64"],
            codeLines: [47, 67],
          },
          {
            id: "stem.conv3",
            label: "conv3x3",
            type: "Conv2d",
            kind: "conv",
            badges: ["64->192"],
            codeLines: [49, 67],
          },
          {
            id: "stem.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            codeLines: [51, 67],
          },
        ],
      },
      {
        id: "stage3",
        label: "stage3",
        type: "InceptionStage",
        kind: "group",
        summary: "2 blocks",
        badges: ["28x28"],
        defaultExpanded: true,
        codeLines: [53, 54, 68, 69, 70],
        children: [
          makeInceptionNode({
            id: "inception3a",
            label: "inception3a",
            inputChannels: 192,
            branch1Channels: 64,
            branch3Reduce: 96,
            branch3Channels: 128,
            branch5Reduce: 16,
            branch5Channels: 32,
            poolChannels: 32,
            callLine: 53,
            forwardLine: 68,
            defaultExpanded: true,
          }),
          makeInceptionNode({
            id: "inception3b",
            label: "inception3b",
            inputChannels: 256,
            branch1Channels: 128,
            branch3Reduce: 128,
            branch3Channels: 192,
            branch5Reduce: 32,
            branch5Channels: 96,
            poolChannels: 64,
            callLine: 54,
            forwardLine: 69,
          }),
          {
            id: "stage3.pool",
            label: "maxpool",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            codeLines: [70],
          },
        ],
      },
      {
        id: "stage4",
        label: "stage4",
        type: "InceptionStage",
        kind: "group",
        summary: "5 blocks",
        badges: ["14x14"],
        codeLines: [55, 56, 57, 58, 59, 71, 72, 73, 74, 75, 76],
        children: [
          makeInceptionNode({
            id: "inception4a",
            label: "inception4a",
            inputChannels: 480,
            branch1Channels: 192,
            branch3Reduce: 96,
            branch3Channels: 208,
            branch5Reduce: 16,
            branch5Channels: 48,
            poolChannels: 64,
            callLine: 55,
            forwardLine: 71,
          }),
          makeInceptionNode({
            id: "inception4b",
            label: "inception4b",
            inputChannels: 512,
            branch1Channels: 160,
            branch3Reduce: 112,
            branch3Channels: 224,
            branch5Reduce: 24,
            branch5Channels: 64,
            poolChannels: 64,
            callLine: 56,
            forwardLine: 72,
          }),
          makeInceptionNode({
            id: "inception4c",
            label: "inception4c",
            inputChannels: 512,
            branch1Channels: 128,
            branch3Reduce: 128,
            branch3Channels: 256,
            branch5Reduce: 24,
            branch5Channels: 64,
            poolChannels: 64,
            callLine: 57,
            forwardLine: 73,
          }),
          makeInceptionNode({
            id: "inception4d",
            label: "inception4d",
            inputChannels: 512,
            branch1Channels: 112,
            branch3Reduce: 144,
            branch3Channels: 288,
            branch5Reduce: 32,
            branch5Channels: 64,
            poolChannels: 64,
            callLine: 58,
            forwardLine: 74,
          }),
          makeInceptionNode({
            id: "inception4e",
            label: "inception4e",
            inputChannels: 528,
            branch1Channels: 256,
            branch3Reduce: 160,
            branch3Channels: 320,
            branch5Reduce: 32,
            branch5Channels: 128,
            poolChannels: 128,
            callLine: 59,
            forwardLine: 75,
          }),
          {
            id: "stage4.pool",
            label: "maxpool",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            codeLines: [76],
          },
        ],
      },
      {
        id: "stage5",
        label: "stage5",
        type: "InceptionStage",
        kind: "group",
        summary: "2 blocks",
        badges: ["7x7"],
        codeLines: [60, 61, 77, 78],
        children: [
          makeInceptionNode({
            id: "inception5a",
            label: "inception5a",
            inputChannels: 832,
            branch1Channels: 256,
            branch3Reduce: 160,
            branch3Channels: 320,
            branch5Reduce: 32,
            branch5Channels: 128,
            poolChannels: 128,
            callLine: 60,
            forwardLine: 77,
          }),
          makeInceptionNode({
            id: "inception5b",
            label: "inception5b",
            inputChannels: 832,
            branch1Channels: 384,
            branch3Reduce: 192,
            branch3Channels: 384,
            branch5Reduce: 48,
            branch5Channels: 128,
            poolChannels: 128,
            callLine: 61,
            forwardLine: 78,
          }),
        ],
      },
      {
        id: "classifier",
        label: "classifier",
        type: "GlobalPoolHead",
        kind: "group",
        summary: "avgpool + fc",
        codeLines: [62, 63, 64, 79, 80, 81, 82],
        children: [
          {
            id: "classifier.avgpool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            codeLines: [62, 79],
          },
          {
            id: "classifier.flatten",
            label: "flatten",
            type: "Flatten",
            kind: "reshape",
            badges: ["1024"],
            codeLines: [80],
          },
          {
            id: "classifier.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.4"],
            codeLines: [63, 81],
          },
          {
            id: "classifier.fc",
            label: "fc",
            type: "Linear",
            kind: "linear",
            badges: ["1024->1000"],
            codeLines: [64, 82],
          },
        ],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      "class InceptionBlock(nn.Module):",
      "    def __init__(self, in_channels, branch1_channels, branch3_reduce, branch3_channels, branch5_reduce, branch5_channels, pool_channels):",
      "        super().__init__()",
      "        self.branch1 = nn.Sequential(",
      "            nn.Conv2d(in_channels, branch1_channels, kernel_size=1),",
      "            nn.ReLU(inplace=True),",
      "        )",
      "        self.branch3 = nn.Sequential(",
      "            nn.Conv2d(in_channels, branch3_reduce, kernel_size=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.Conv2d(branch3_reduce, branch3_channels, kernel_size=3, padding=1),",
      "            nn.ReLU(inplace=True),",
      "        )",
      "        self.branch5 = nn.Sequential(",
      "            nn.Conv2d(in_channels, branch5_reduce, kernel_size=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.Conv2d(branch5_reduce, branch5_channels, kernel_size=5, padding=2),",
      "            nn.ReLU(inplace=True),",
      "        )",
      "        self.branch_pool = nn.Sequential(",
      "            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),",
      "            nn.Conv2d(in_channels, pool_channels, kernel_size=1),",
      "            nn.ReLU(inplace=True),",
      "        )",
      "",
      "    def forward(self, x):",
      "        branch1 = self.branch1(x)",
      "        branch3 = self.branch3(x)",
      "        branch5 = self.branch5(x)",
      "        branch_pool = self.branch_pool(x)",
      "        branches = [branch1, branch3, branch5, branch_pool]",
      "        x = torch.cat(branches, dim=1)",
      "        return x",
      "",
      "",
      "class GoogLeNet(nn.Module):",
      "    def __init__(self, num_classes=1000):",
      "        super().__init__()",
      "        self.stem = nn.Sequential(",
      "            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),",
      "            nn.ReLU(inplace=True),",
      "            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),",
      "            nn.Conv2d(64, 64, kernel_size=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.Conv2d(64, 192, kernel_size=3, padding=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),",
      "        )",
      "        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)",
      "        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)",
      "        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)",
      "        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)",
      "        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)",
      "        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)",
      "        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)",
      "        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)",
      "        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)",
      "        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))",
      "        self.dropout = nn.Dropout(0.4)",
      "        self.fc = nn.Linear(1024, num_classes)",
      "",
      "    def forward(self, x):",
      "        x = self.stem(x)",
      "        x = self.inception3a(x)",
      "        x = self.inception3b(x)",
      "        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)",
      "        x = self.inception4a(x)",
      "        x = self.inception4b(x)",
      "        x = self.inception4c(x)",
      "        x = self.inception4d(x)",
      "        x = self.inception4e(x)",
      "        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)",
      "        x = self.inception5a(x)",
      "        x = self.inception5b(x)",
      "        x = self.avgpool(x)",
      "        x = torch.flatten(x, start_dim=1)",
      "        x = self.dropout(x)",
      "        logits = self.fc(x)",
      "        return logits",
      "",
      "",
      "model = GoogLeNet(num_classes=1000)",
      "test_input = torch.randn(2, 3, 224, 224)",
      "logits = model(test_input)",
      "# logits: (2, 1000)",
    ],
    jaxCode: googlenetJaxCode,
  },
  {
    id: "unet",
    label: "U-Net",
    breadcrumb: "U-Net / expansive path / up4 / skip d4",
    stats: "4 down blocks · bottleneck · 4 up blocks",
    fileName: "unet.py",
    jaxFileName: "unet_jax.py",
    paper: {
      title: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
      authors: "Olaf Ronneberger, Philipp Fischer, Thomas Brox",
      year: "2015",
      publishedLabel: "May 18, 2015",
      publishedDate: "2015-05-18",
      venue: "arXiv / MICCAI 2015",
      url: "https://arxiv.org/abs/1505.04597",
      pdfUrl: "/papers/unet.pdf",
      focus: ["encoder-decoder segmentation", "skip concatenations", "biomedical images"],
    },
    selectedId: "expansive.up4.skip",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["1 x 572 x 572"],
        codeLines: [61],
      },
      {
        id: "contracting",
        label: "Contracting Path",
        type: "Encoder",
        kind: "group",
        summary: "4 DoubleConv blocks",
        defaultExpanded: true,
        codeLines: [21, 22, 23, 24, 25, 26, 27, 28, 41, 42, 43, 44],
        children: [
          {
            id: "contracting.down1",
            label: "down1",
            type: "DoubleConv",
            kind: "conv",
            badges: ["1->64", "572x572"],
            codeLines: [8, 9, 10, 11, 12, 16, 21, 41],
          },
          {
            id: "contracting.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [22, 42],
          },
          {
            id: "contracting.down2",
            label: "down2",
            type: "DoubleConv",
            kind: "conv",
            badges: ["64->128"],
            codeLines: [23, 42],
          },
          {
            id: "contracting.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [24, 43],
          },
          {
            id: "contracting.down3",
            label: "down3",
            type: "DoubleConv",
            kind: "conv",
            badges: ["128->256"],
            codeLines: [25, 43],
          },
          {
            id: "contracting.pool3",
            label: "pool3",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [26, 44],
          },
          {
            id: "contracting.down4",
            label: "down4",
            type: "DoubleConv",
            kind: "conv",
            badges: ["256->512"],
            codeLines: [27, 44],
          },
          {
            id: "contracting.pool4",
            label: "pool4",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [28, 45],
          },
        ],
      },
      {
        id: "bottleneck",
        label: "bottleneck",
        type: "DoubleConv",
        kind: "conv",
        badges: ["512->1024"],
        codeLines: [29, 45],
      },
      {
        id: "expansive",
        label: "Expansive Path",
        type: "Decoder",
        kind: "group",
        summary: "upsample + concat skips",
        defaultExpanded: true,
        codeLines: [30, 31, 32, 33, 34, 35, 36, 37, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
        children: [
          {
            id: "expansive.up4",
            label: "up4",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["1024->512", "x2"],
            codeLines: [30, 46],
          },
          {
            id: "expansive.up4.skip",
            label: "skip d4",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [47],
          },
          {
            id: "expansive.dec4",
            label: "dec4",
            type: "DoubleConv",
            kind: "conv",
            badges: ["1024->512"],
            codeLines: [31, 48],
          },
          {
            id: "expansive.up3",
            label: "up3",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["512->256", "x2"],
            codeLines: [32, 49],
          },
          {
            id: "expansive.up3.skip",
            label: "skip d3",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [50],
          },
          {
            id: "expansive.dec3",
            label: "dec3",
            type: "DoubleConv",
            kind: "conv",
            badges: ["512->256"],
            codeLines: [33, 51],
          },
          {
            id: "expansive.up2",
            label: "up2",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["256->128", "x2"],
            codeLines: [34, 52],
          },
          {
            id: "expansive.up2.skip",
            label: "skip d2",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [53],
          },
          {
            id: "expansive.dec2",
            label: "dec2",
            type: "DoubleConv",
            kind: "conv",
            badges: ["256->128"],
            codeLines: [35, 54],
          },
          {
            id: "expansive.up1",
            label: "up1",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["128->64", "x2"],
            codeLines: [36, 55],
          },
          {
            id: "expansive.up1.skip",
            label: "skip d1",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [56],
          },
          {
            id: "expansive.dec1",
            label: "dec1",
            type: "DoubleConv",
            kind: "conv",
            badges: ["128->64"],
            codeLines: [37, 57],
          },
        ],
      },
      {
        id: "out_conv",
        label: "out_conv",
        type: "Conv2d",
        kind: "conv",
        badges: ["64->2", "1x1"],
        codeLines: [38, 58],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "",
      "",
      "class DoubleConv(nn.Module):",
      "    def __init__(self, in_channels, out_channels):",
      "        super().__init__()",
      "        self.net = nn.Sequential(",
      "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),",
      "            nn.ReLU(inplace=True),",
      "            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),",
      "            nn.ReLU(inplace=True),",
      "        )",
      "",
      "    def forward(self, x):",
      "        out = self.net(x)",
      "        return out",
      "",
      "",
      "class UNet(nn.Module):",
      "    def __init__(self, num_classes=2):",
      "        super().__init__()",
      "        self.down1 = DoubleConv(1, 64)",
      "        self.pool1 = nn.MaxPool2d(2)",
      "        self.down2 = DoubleConv(64, 128)",
      "        self.pool2 = nn.MaxPool2d(2)",
      "        self.down3 = DoubleConv(128, 256)",
      "        self.pool3 = nn.MaxPool2d(2)",
      "        self.down4 = DoubleConv(256, 512)",
      "        self.pool4 = nn.MaxPool2d(2)",
      "        self.bottleneck = DoubleConv(512, 1024)",
      "        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)",
      "        self.dec4 = DoubleConv(1024, 512)",
      "        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)",
      "        self.dec3 = DoubleConv(512, 256)",
      "        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)",
      "        self.dec2 = DoubleConv(256, 128)",
      "        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)",
      "        self.dec1 = DoubleConv(128, 64)",
      "        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)",
      "",
      "    def forward(self, x):",
      "        d1 = self.down1(x)",
      "        p1 = self.pool1(d1)",
      "        d2 = self.down2(p1)",
      "        p2 = self.pool2(d2)",
      "        d3 = self.down3(p2)",
      "        p3 = self.pool3(d3)",
      "        d4 = self.down4(p3)",
      "        p4 = self.pool4(d4)",
      "        b = self.bottleneck(p4)",
      "        x = self.up4(b)",
      "        x = torch.cat([x, d4], dim=1)",
      "        x = self.dec4(x)",
      "        x = self.up3(x)",
      "        x = torch.cat([x, d3], dim=1)",
      "        x = self.dec3(x)",
      "        x = self.up2(x)",
      "        x = torch.cat([x, d2], dim=1)",
      "        x = self.dec2(x)",
      "        x = self.up1(x)",
      "        x = torch.cat([x, d1], dim=1)",
      "        x = self.dec1(x)",
      "        logits = self.out_conv(x)",
      "        return logits",
      "",
      "",
      "model = UNet(num_classes=2)",
      "test_input = torch.randn(2, 1, 572, 572)",
      "logits = model(test_input)",
      "# logits: (2, 2, 572, 572)",
    ],
    jaxCode: unetJaxCode,
  },
  {
    id: "transformer",
    label: "Transformer",
    breadcrumb: "Transformer / decoder.0 / cross_attn",
    stats: "6 encoder layers · 6 decoder layers · 8 heads",
    fileName: "transformer.py",
    jaxFileName: "transformer_jax.py",
    paper: {
      title: "Attention Is All You Need",
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
      year: "2017",
      publishedLabel: "Jun 12, 2017",
      publishedDate: "2017-06-12",
      venue: "NeurIPS 2017",
      url: "https://arxiv.org/abs/1706.03762",
      pdfUrl: "/papers/transformer.pdf",
      focus: ["scaled dot-product attention", "encoder-decoder stacks", "positional encoding"],
    },
    selectedId: "decoder.0.cross_attn",
    nodes: [
      {
        id: "src.input",
        label: "src input",
        type: "TokenIds",
        kind: "input",
        badges: ["source", "16 tokens"],
        codeLines: [73],
      },
      {
        id: "tgt.input",
        label: "target input",
        type: "TokenIds",
        kind: "input",
        badges: ["target", "shifted"],
        codeLines: [74],
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position",
        defaultExpanded: true,
        codeLines: [55, 56, 57, 63, 66],
        children: [
          {
            id: "src_embed",
            label: "src_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            codeLines: [55, 63],
          },
          {
            id: "tgt_embed",
            label: "tgt_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            codeLines: [56, 66],
          },
          {
            id: "positional_encoding",
            label: "positional",
            type: "SinusoidalEncoding",
            kind: "embedding",
            badges: ["absolute"],
            codeLines: [6, 7, 9, 10, 11, 12, 13, 14, 57, 63, 66],
          },
        ],
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "Stack",
        kind: "group",
        summary: "6 layers",
        badges: ["bidirectional"],
        defaultExpanded: true,
        codeLines: [58, 64, 65],
        children: Array.from({ length: 6 }, (_, index) => makeTransformerEncoderBlock(index, index === 0)),
      },
      {
        id: "decoder",
        label: "Decoder",
        type: "Stack",
        kind: "group",
        summary: "6 layers",
        badges: ["causal", "cross-attn"],
        defaultExpanded: true,
        codeLines: [59, 67, 68],
        children: Array.from({ length: 6 }, (_, index) => makeTransformerDecoderBlock(index, index === 0)),
      },
      {
        id: "generator",
        label: "generator",
        type: "Linear",
        kind: "linear",
        badges: ["512->vocab"],
        codeLines: [60, 69],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      "",
      "class PositionalEncoding(nn.Module):",
      "    def __init__(self, d_model=512, max_len=5000):",
      "        super().__init__()",
      "        positions = torch.arange(max_len)",
      "        position = positions.unsqueeze(1)",
      "        even_indices = torch.arange(0, d_model, 2)",
      "        log_base = torch.log(torch.tensor(10000.0))",
      "        scale = -log_base / d_model",
      "        div_term = torch.exp(even_indices * scale)",
      "        pe = torch.zeros(max_len, d_model)",
      "        sin_values = torch.sin(position * div_term)",
      "        cos_values = torch.cos(position * div_term)",
      "        pe[:, 0::2] = sin_values",
      "        pe[:, 1::2] = cos_values",
      "        self.register_buffer('pe', pe)",
      "",
      "    def forward(self, x):",
      "        seq_len = x.size(1)",
      "        position_encoding = self.pe[:seq_len]",
      "        encoded = x + position_encoding",
      "        return encoded",
      "",
      "",
      "class EncoderLayer(nn.Module):",
      "    def __init__(self, d_model=512, nhead=8, d_ff=2048):",
      "        super().__init__()",
      "        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)",
      "        self.ffn = nn.Sequential(",
      "            nn.Linear(d_model, d_ff),",
      "            nn.ReLU(),",
      "            nn.Linear(d_ff, d_model),",
      "        )",
      "        self.norm1 = nn.LayerNorm(d_model)",
      "        self.norm2 = nn.LayerNorm(d_model)",
      "",
      "    def forward(self, x, src_mask=None):",
      "        attn, _ = self.self_attn(x, x, x, attn_mask=src_mask)",
      "        attn_residual = x + attn",
      "        x = self.norm1(attn_residual)",
      "        ffn = self.ffn(x)",
      "        ffn_residual = x + ffn",
      "        out = self.norm2(ffn_residual)",
      "        return out",
      "",
      "",
      "class DecoderLayer(nn.Module):",
      "    def __init__(self, d_model=512, nhead=8, d_ff=2048):",
      "        super().__init__()",
      "        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)",
      "        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)",
      "        self.ffn = nn.Sequential(",
      "            nn.Linear(d_model, d_ff),",
      "            nn.ReLU(),",
      "            nn.Linear(d_ff, d_model),",
      "        )",
      "        self.norm1 = nn.LayerNorm(d_model)",
      "        self.norm2 = nn.LayerNorm(d_model)",
      "        self.norm3 = nn.LayerNorm(d_model)",
      "",
      "    def forward(self, x, memory, tgt_mask=None):",
      "        masked, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)",
      "        masked_residual = x + masked",
      "        x = self.norm1(masked_residual)",
      "        cross, _ = self.cross_attn(x, memory, memory)",
      "        cross_residual = x + cross",
      "        x = self.norm2(cross_residual)",
      "        ffn = self.ffn(x)",
      "        ffn_residual = x + ffn",
      "        out = self.norm3(ffn_residual)",
      "        return out",
      "",
      "",
      "class Transformer(nn.Module):",
      "    def __init__(self, vocab_size=37000, d_model=512, nhead=8, num_layers=6):",
      "        super().__init__()",
      "        self.src_embed = nn.Embedding(vocab_size, d_model)",
      "        self.tgt_embed = nn.Embedding(vocab_size, d_model)",
      "        self.pos = PositionalEncoding(d_model)",
      "        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead) for _ in range(num_layers)])",
      "        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead) for _ in range(num_layers)])",
      "        self.generator = nn.Linear(d_model, vocab_size)",
      "",
      "    def forward(self, src_ids, tgt_ids, tgt_mask):",
      "        src_embeddings = self.src_embed(src_ids)",
      "        memory = self.pos(src_embeddings)",
      "        for layer in self.encoder:",
      "            memory = layer(memory)",
      "        tgt_embeddings = self.tgt_embed(tgt_ids)",
      "        x = self.pos(tgt_embeddings)",
      "        for layer in self.decoder:",
      "            x = layer(x, memory, tgt_mask)",
      "        logits = self.generator(x)",
      "        return logits",
      "",
      "",
      "model = Transformer(vocab_size=37000)",
      "src_ids = torch.randint(0, 37000, (2, 16))",
      "tgt_ids = torch.randint(0, 37000, (2, 16))",
      "mask_values = torch.ones(16, 16)",
      "mask_values = mask_values * float('-inf')",
      "tgt_mask = torch.triu(mask_values, diagonal=1)",
      "logits = model(src_ids, tgt_ids, tgt_mask)",
      "# logits: (2, 16, 37000)",
    ],
    jaxCode: transformerJaxCode,
  },
  {
    id: "bert",
    label: "BERT base",
    breadcrumb: "BERT / encoder.layer.3 / self_attn",
    stats: "12 encoder layers · 12 heads/layer · 110M params",
    fileName: "bert_base.py",
    jaxFileName: "bert_base_jax.py",
    paper: {
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      authors: "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
      year: "2018",
      publishedLabel: "Oct 11, 2018",
      publishedDate: "2018-10-11",
      venue: "arXiv / NAACL 2019",
      url: "https://arxiv.org/abs/1810.04805",
      pdfUrl: "/papers/bert.pdf",
      focus: ["masked language modeling", "bidirectional encoders", "fine-tuning"],
    },
    selectedId: "encoder.layer.3.self_attn",
    nodes: [
      {
        id: "input_ids",
        label: "input_ids",
        type: "TokenIds",
        kind: "input",
        badges: ["WordPiece", "16 tokens"],
        codeLines: [55],
      },
      {
        id: "token_type_ids",
        label: "token_type_ids",
        type: "SegmentIds",
        kind: "input",
        badges: ["sentence A/B"],
        codeLines: [56],
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position + segment",
        defaultExpanded: true,
        codeLines: [9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 41, 47],
        children: [
          {
            id: "embeddings.word",
            label: "word",
            type: "WordPieceEmbedding",
            kind: "embedding",
            badges: ["30522", "768"],
            codeLines: [9, 17],
          },
          {
            id: "embeddings.position",
            label: "position",
            type: "PositionEmbedding",
            kind: "embedding",
            badges: ["512", "768"],
            codeLines: [10, 16, 18],
          },
          {
            id: "embeddings.segment",
            label: "segment",
            type: "TokenTypeEmbedding",
            kind: "embedding",
            badges: ["2", "768"],
            codeLines: [11, 19],
          },
          {
            id: "embeddings.norm",
            label: "norm",
            type: "LayerNorm",
            kind: "norm",
            codeLines: [12, 20],
          },
          {
            id: "embeddings.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.1"],
            codeLines: [13, 20],
          },
        ],
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "Stack",
        kind: "group",
        summary: "12 bidirectional layers",
        badges: ["no causal mask"],
        defaultExpanded: true,
        codeLines: [42, 48, 49],
        children: Array.from({ length: 12 }, (_, index) => makeBertLayer(index, index === 3)),
      },
      {
        id: "pooler",
        label: "pooler",
        type: "CLSProjection",
        kind: "linear",
        badges: ["CLS", "768->768"],
        codeLines: [43, 50, 58],
      },
      {
        id: "mlm_head",
        label: "mlm_head",
        type: "MaskedLMHead",
        kind: "head",
        badges: ["768->30522"],
        codeLines: [44, 51, 58],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      "",
      "class BertEmbeddings(nn.Module):",
      "    def __init__(self, vocab_size=30522, hidden_size=768, max_position=512):",
      "        super().__init__()",
      "        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)",
      "        self.position_embeddings = nn.Embedding(max_position, hidden_size)",
      "        self.token_type_embeddings = nn.Embedding(2, hidden_size)",
      "        self.norm = nn.LayerNorm(hidden_size)",
      "        self.dropout = nn.Dropout(0.1)",
      "",
      "    def forward(self, input_ids, token_type_ids):",
      "        positions = torch.arange(input_ids.size(1), device=input_ids.device)",
      "        x = self.word_embeddings(input_ids)",
      "        position_embeddings = self.position_embeddings(positions)",
      "        position_embeddings = position_embeddings[None, :, :]",
      "        x = x + position_embeddings",
      "        x = x + self.token_type_embeddings(token_type_ids)",
      "        x = self.norm(x)",
      "        x = self.dropout(x)",
      "        return x",
      "",
      "",
      "class BertLayer(nn.Module):",
      "    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):",
      "        super().__init__()",
      "        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)",
      "        self.attn_norm = nn.LayerNorm(hidden_size)",
      "        self.ffn = nn.Sequential(",
      "            nn.Linear(hidden_size, intermediate_size),",
      "            nn.GELU(),",
      "            nn.Linear(intermediate_size, hidden_size),",
      "        )",
      "        self.ffn_norm = nn.LayerNorm(hidden_size)",
      "        self.dropout = nn.Dropout(0.1)",
      "",
      "    def forward(self, x, attention_mask=None):",
      "        attn, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)",
      "        attn = self.dropout(attn)",
      "        attn_residual = x + attn",
      "        x = self.attn_norm(attn_residual)",
      "        ffn = self.ffn(x)",
      "        ffn = self.dropout(ffn)",
      "        ffn_residual = x + ffn",
      "        out = self.ffn_norm(ffn_residual)",
      "        return out",
      "",
      "",
      "class BERTBase(nn.Module):",
      "    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12):",
      "        super().__init__()",
      "        self.embeddings = BertEmbeddings(vocab_size, hidden_size)",
      "        self.layers = nn.ModuleList([BertLayer(hidden_size) for _ in range(num_layers)])",
      "        self.pooler = nn.Linear(hidden_size, hidden_size)",
      "        self.mlm = nn.Linear(hidden_size, vocab_size)",
      "",
      "    def forward(self, input_ids, token_type_ids, attention_mask=None):",
      "        x = self.embeddings(input_ids, token_type_ids)",
      "        for layer in self.layers:",
      "            x = layer(x, attention_mask)",
      "        cls_token = x[:, 0]",
      "        pooled_projection = self.pooler(cls_token)",
      "        pooled = torch.tanh(pooled_projection)",
      "        mlm_logits = self.mlm(x)",
      "        return mlm_logits, pooled",
      "",
      "",
      "model = BERTBase(vocab_size=30522)",
      "input_ids = torch.randint(0, 30522, (2, 16))",
      "token_type_ids = torch.zeros((2, 16), dtype=torch.long)",
      "attention_mask = torch.zeros((2, 16), dtype=torch.bool)",
      "mlm_logits, pooled = model(input_ids, token_type_ids, attention_mask)",
      "# mlm_logits: (2, 16, 30522), pooled: (2, 768)",
    ],
    jaxCode: bertJaxCode,
  },
  {
    id: "gpt2",
    label: "GPT-2 small",
    breadcrumb: "GPT-2 / block.3 / attn / head.2",
    stats: "12 blocks · 12 heads/block · virtualized",
    fileName: "gpt2_attention.py",
    jaxFileName: "gpt2_attention_jax.py",
    paper: {
      title: "Language Models are Unsupervised Multitask Learners",
      authors: "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever",
      year: "2019",
      publishedLabel: "Feb 14, 2019",
      publishedDate: "2019-02-14",
      venue: "OpenAI technical report",
      url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      pdfUrl: "/papers/gpt2.pdf",
      focus: ["decoder-only transformers", "causal language modeling", "zero-shot transfer"],
    },
    selectedId: "block.3.attn.head.2",
    nodes: [
      {
        id: "wte",
        label: "wte",
        type: "TokenEmbedding",
        kind: "embedding",
        badges: ["vocab", "768"],
        codeLines: [9],
      },
      {
        id: "wpe",
        label: "wpe",
        type: "PositionEmbedding",
        kind: "embedding",
        badges: ["1024", "768"],
        codeLines: [10],
      },
      {
        id: "drop",
        label: "drop",
        type: "Dropout",
        kind: "dropout",
        codeLines: [11],
      },
      ...Array.from({ length: 12 }, (_, index) => makeGpt2Block(index, index === 3)),
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      "",
      "class GPT2Small(nn.Module):",
      "    def __init__(self, vocab_size, n_ctx=1024, n_embd=768):",
      "        super().__init__()",
      "        self.wte = nn.Embedding(vocab_size, n_embd)",
      "        self.wpe = nn.Embedding(n_ctx, n_embd)",
      "        self.drop = nn.Dropout(0.1)",
      "        self.blocks = nn.ModuleList([Block() for _ in range(12)])",
      "        self.ln_f = nn.LayerNorm(n_embd)",
      "        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)",
      "",
      "    def forward(self, input_ids, mask):",
      "        B, T = input_ids.shape",
      "        positions = torch.arange(T, device=input_ids.device)",
      "        token_embeddings = self.wte(input_ids)",
      "        position_embeddings = self.wpe(positions)",
      "        position_embeddings = position_embeddings[None, :, :]",
      "        x = token_embeddings + position_embeddings",
      "        x = self.drop(x)",
      "        for block in self.blocks:",
      "            x = block(x, mask)",
      "        x = self.ln_f(x)",
      "        logits = self.lm_head(x)",
      "        return logits",
      "",
      "class CausalSelfAttention(nn.Module):",
      "    def __init__(self, n_embd=768, n_head=12):",
      "        super().__init__()",
      "        self.n_head = n_head",
      "        self.c_attn = nn.Linear(n_embd, 3 * n_embd)",
      "        self.c_proj = nn.Linear(n_embd, n_embd)",
      "",
      "    def forward(self, x, mask):",
      "        B, T, C = x.shape",
      "        qkv = self.c_attn(x)",
      "        q, k, v = qkv.split(C, dim=2)",
      "        head_dim = C // self.n_head",
      "        q = q.view(B, T, self.n_head, head_dim)",
      "        q = q.transpose(1, 2)",
      "        k = k.view(B, T, self.n_head, head_dim)",
      "        k = k.transpose(1, 2)",
      "        v = v.view(B, T, self.n_head, head_dim)",
      "        v = v.transpose(1, 2)",
      "        key_transpose = k.transpose(-2, -1)",
      "        scores = q @ key_transpose",
      "        scale = k.size(-1) ** -0.5",
      "        att = scores * scale",
      "        mask_window = mask[:, :, :T, :T]",
      "        att = att.masked_fill(mask_window == 0, float('-inf'))",
      "        weights = F.softmax(att, dim=-1)",
      "        y = weights @ v",
      "        y = y.transpose(1, 2)",
      "        y = y.contiguous()",
      "        y = y.view(B, T, C)",
      "        out = self.c_proj(y)",
      "        return out",
      "",
      "class Block(nn.Module):",
      "    def __init__(self):",
      "        super().__init__()",
      "        self.ln_1 = nn.LayerNorm(768)",
      "        self.attn = CausalSelfAttention()",
      "        self.ln_2 = nn.LayerNorm(768)",
      "        self.mlp = nn.Sequential(",
      "            nn.Linear(768, 3072),",
      "            nn.GELU(),",
      "            nn.Linear(3072, 768),",
      "        )",
      "",
      "    def forward(self, x, mask):",
      "        attn_input = self.ln_1(x)",
      "        attn = self.attn(attn_input, mask)",
      "        x = x + attn",
      "        mlp_input = self.ln_2(x)",
      "        mlp_out = self.mlp(mlp_input)",
      "        x = x + mlp_out",
      "        return x",
      "",
      "",
      "model = GPT2Small(vocab_size=50257)",
      "test_input = torch.randint(0, 50257, (2, 16))",
      "mask_values = torch.ones(16, 16)",
      "mask = torch.tril(mask_values)",
      "mask = mask.view(1, 1, 16, 16)",
      "logits = model(test_input, mask)",
      "# logits: (2, 16, 50257)",
    ],
    jaxCode: gpt2JaxCode,
  },
  {
    id: "vit",
    label: "ViT-B/16",
    breadcrumb: "ViT-B/16 / encoder.block.3 / attn",
    stats: "196 patches · 12 encoder blocks · 12 heads",
    fileName: "vit_b16.py",
    jaxFileName: "vit_b16_jax.py",
    paper: {
      title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
      authors: "Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby",
      year: "2020",
      publishedLabel: "Oct 22, 2020",
      publishedDate: "2020-10-22",
      venue: "arXiv / ICLR 2021",
      url: "https://arxiv.org/abs/2010.11929",
      pdfUrl: "/papers/vit.pdf",
      focus: ["image patches as tokens", "class token", "Transformer encoders for vision"],
    },
    selectedId: "encoder.block.3.attn",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [50],
      },
      {
        id: "patch_embed",
        label: "patch_embed",
        type: "Conv2d projection",
        kind: "conv",
        badges: ["16x16", "196 tokens", "768"],
        defaultExpanded: true,
        codeLines: [8, 11, 12, 33, 40],
        children: [
          {
            id: "patch_embed.proj",
            label: "proj",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->768", "k=16", "s=16"],
            codeLines: [8, 11],
          },
          {
            id: "patch_embed.flatten",
            label: "flatten patches",
            type: "Flatten",
            kind: "reshape",
            badges: ["14x14 -> 196"],
            codeLines: [12],
          },
        ],
      },
      {
        id: "tokens",
        label: "Token Prep",
        type: "Group",
        kind: "group",
        summary: "class + position",
        defaultExpanded: true,
        codeLines: [34, 35, 41, 42, 43],
        children: [
          {
            id: "tokens.cls",
            label: "cls_token",
            type: "LearnedToken",
            kind: "embedding",
            badges: ["1 x 768"],
            codeLines: [34, 41, 42],
          },
          {
            id: "tokens.position",
            label: "pos_embed",
            type: "PositionEmbedding",
            kind: "embedding",
            badges: ["197 x 768"],
            codeLines: [35, 43],
          },
        ],
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "Stack",
        kind: "group",
        summary: "12 Transformer blocks",
        defaultExpanded: true,
        codeLines: [36, 44, 45],
        children: Array.from({ length: 12 }, (_, index) => makeVitBlock(index, index === 3)),
      },
      {
        id: "norm",
        label: "encoder_norm",
        type: "LayerNorm",
        kind: "norm",
        badges: ["CLS"],
        codeLines: [37, 46],
      },
      {
        id: "head",
        label: "head",
        type: "Linear",
        kind: "linear",
        badges: ["768->1000"],
        codeLines: [38, 47],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "",
      "",
      "class PatchEmbed(nn.Module):",
      "    def __init__(self, in_channels=3, embed_dim=768, patch_size=16, image_size=224):",
      "        super().__init__()",
      "        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)",
      "        patches_per_side = image_size // patch_size",
      "        self.num_patches = patches_per_side ** 2",
      "",
      "    def forward(self, x):",
      "        x = self.proj(x)",
      "        x = x.flatten(2)",
      "        x = x.transpose(1, 2)",
      "        return x",
      "",
      "",
      "class EncoderBlock(nn.Module):",
      "    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072):",
      "        super().__init__()",
      "        self.norm1 = nn.LayerNorm(embed_dim)",
      "        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)",
      "        self.norm2 = nn.LayerNorm(embed_dim)",
      "        self.mlp = nn.Sequential(",
      "            nn.Linear(embed_dim, mlp_dim),",
      "            nn.GELU(),",
      "            nn.Linear(mlp_dim, embed_dim),",
      "        )",
      "",
      "    def forward(self, x):",
      "        attn_input = self.norm1(x)",
      "        attn_output, _ = self.attn(attn_input, attn_input, attn_input)",
      "        x = x + attn_output",
      "        mlp_input = self.norm2(x)",
      "        mlp_output = self.mlp(mlp_input)",
      "        x = x + mlp_output",
      "        return x",
      "",
      "",
      "class VisionTransformer(nn.Module):",
      "    def __init__(self, num_classes=1000, embed_dim=768, depth=12, num_heads=12):",
      "        super().__init__()",
      "        self.patch_embed = PatchEmbed(embed_dim=embed_dim)",
      "        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))",
      "        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))",
      "        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads) for _ in range(depth)])",
      "        self.norm = nn.LayerNorm(embed_dim)",
      "        self.head = nn.Linear(embed_dim, num_classes)",
      "",
      "    def forward(self, x):",
      "        x = self.patch_embed(x)",
      "        batch_size = x.size(0)",
      "        cls = self.cls_token.expand(batch_size, -1, -1)",
      "        x = torch.cat([cls, x], dim=1)",
      "        x = x + self.pos_embed",
      "        for block in self.blocks:",
      "            x = block(x)",
      "        x = self.norm(x)",
      "        cls_output = x[:, 0]",
      "        logits = self.head(cls_output)",
      "        return logits",
      "",
      "",
      "model = VisionTransformer(num_classes=1000)",
      "test_input = torch.randn(2, 3, 224, 224)",
      "logits = model(test_input)",
      "# logits: (2, 1000)",
    ],
    jaxCode: vitJaxCode,
  },
  {
    id: "resnet18",
    label: "ResNet-18",
    breadcrumb: "ResNet-18 / layer2 / block.0 / conv1",
    stats: "4 residual stages · 8 BasicBlocks · 18 conv layers",
    fileName: "resnet18.py",
    jaxFileName: "resnet18_jax.py",
    paper: {
      title: "Deep Residual Learning for Image Recognition",
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
      year: "2015",
      publishedLabel: "Dec 10, 2015",
      publishedDate: "2015-12-10",
      venue: "arXiv / CVPR 2016",
      url: "https://arxiv.org/abs/1512.03385",
      pdfUrl: "/papers/resnet18.pdf",
      focus: ["identity shortcuts", "residual blocks", "very deep CNNs"],
    },
    selectedId: "layer2.0.conv1",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [57],
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-ReLU",
        kind: "group",
        summary: "7x7 stride 2",
        defaultExpanded: true,
        codeLines: [31, 32, 33, 34, 35, 58],
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->64", "k=7", "s=2"],
            codeLines: [32, 58],
          },
          {
            id: "stem.bn",
            label: "bn",
            type: "BatchNorm2d",
            kind: "norm",
            badges: ["64"],
            codeLines: [33, 58],
          },
          {
            id: "stem.relu",
            label: "relu",
            type: "ReLU",
            kind: "activation",
            codeLines: [34, 58],
          },
        ],
      },
      {
        id: "maxpool",
        label: "maxpool",
        type: "MaxPool2d",
        kind: "pool",
        badges: ["k=3", "s=2"],
        codeLines: [36, 59],
      },
      {
        id: "layer1",
        label: "layer1",
        type: "ResidualStage",
        kind: "group",
        summary: "2 BasicBlocks",
        badges: ["64 ch", "56x56"],
        codeLines: [37, 60],
        children: [
          {
            id: "layer1.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [51, 60],
          },
          {
            id: "layer1.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [53, 54, 60],
          },
        ],
      },
      {
        id: "layer2",
        label: "layer2",
        type: "ResidualStage",
        kind: "group",
        summary: "2 BasicBlocks",
        badges: ["128 ch", "28x28"],
        defaultExpanded: true,
        codeLines: [38, 61],
        children: [
          {
            id: "layer2.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            defaultExpanded: true,
            codeLines: [10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 51, 61],
            children: [
              {
                id: "layer2.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["64->128", "k=3", "s=2"],
                codeLines: [10, 19, 51],
              },
              {
                id: "layer2.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["128"],
                codeLines: [11, 19],
              },
              {
                id: "layer2.0.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [12, 19],
              },
              {
                id: "layer2.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["128->128", "k=3"],
                codeLines: [13, 20],
              },
              {
                id: "layer2.0.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["128"],
                codeLines: [14, 20],
              },
              {
                id: "layer2.0.downsample",
                label: "downsample",
                type: "ProjectionSkip",
                kind: "group",
                summary: "1x1 stride 2",
                defaultExpanded: true,
                codeLines: [15, 21, 22, 47, 48, 49, 50],
                children: [
                  {
                    id: "layer2.0.downsample.conv",
                    label: "conv1x1",
                    type: "Conv2d",
                    kind: "conv",
                    badges: ["64->128", "s=2"],
                    codeLines: [48, 22],
                  },
                  {
                    id: "layer2.0.downsample.bn",
                    label: "bn",
                    type: "BatchNorm2d",
                    kind: "norm",
                    badges: ["128"],
                    codeLines: [49, 22],
                  },
                ],
              },
              {
                id: "layer2.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [23],
              },
            ],
          },
          {
            id: "layer2.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [53, 54, 61],
          },
        ],
      },
      {
        id: "layer3",
        label: "layer3",
        type: "ResidualStage",
        kind: "group",
        summary: "2 BasicBlocks",
        badges: ["256 ch", "14x14"],
        codeLines: [39, 62],
        children: [
          {
            id: "layer3.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            codeLines: [10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 51, 62],
            children: [
              {
                id: "layer3.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["128->256", "k=3", "s=2"],
                codeLines: [10, 19, 51],
              },
              {
                id: "layer3.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [11, 19],
              },
              {
                id: "layer3.0.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [12, 19],
              },
              {
                id: "layer3.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [13, 20],
              },
              {
                id: "layer3.0.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [14, 20],
              },
              {
                id: "layer3.0.downsample",
                label: "downsample",
                type: "ProjectionSkip",
                kind: "group",
                summary: "1x1 stride 2",
                codeLines: [15, 21, 22, 47, 48, 49, 50],
                children: [
                  {
                    id: "layer3.0.downsample.conv",
                    label: "conv1x1",
                    type: "Conv2d",
                    kind: "conv",
                    badges: ["128->256", "s=2"],
                    codeLines: [48, 22],
                  },
                  {
                    id: "layer3.0.downsample.bn",
                    label: "bn",
                    type: "BatchNorm2d",
                    kind: "norm",
                    badges: ["256"],
                    codeLines: [49, 22],
                  },
                ],
              },
              {
                id: "layer3.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [23],
              },
            ],
          },
          {
            id: "layer3.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [10, 11, 12, 13, 14, 17, 18, 19, 20, 23, 53, 54, 62],
            children: [
              {
                id: "layer3.1.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [10, 19, 53],
              },
              {
                id: "layer3.1.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [11, 19],
              },
              {
                id: "layer3.1.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [12, 19],
              },
              {
                id: "layer3.1.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [13, 20],
              },
              {
                id: "layer3.1.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [14, 20],
              },
              {
                id: "layer3.1.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [23],
              },
            ],
          },
        ],
      },
      {
        id: "layer4",
        label: "layer4",
        type: "ResidualStage",
        kind: "group",
        summary: "2 BasicBlocks",
        badges: ["512 ch", "7x7"],
        codeLines: [40, 63],
        children: [
          {
            id: "layer4.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            codeLines: [10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 51, 63],
            children: [
              {
                id: "layer4.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->512", "k=3", "s=2"],
                codeLines: [10, 19, 51],
              },
              {
                id: "layer4.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [11, 19],
              },
              {
                id: "layer4.0.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [12, 19],
              },
              {
                id: "layer4.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["512->512", "k=3"],
                codeLines: [13, 20],
              },
              {
                id: "layer4.0.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [14, 20],
              },
              {
                id: "layer4.0.downsample",
                label: "downsample",
                type: "ProjectionSkip",
                kind: "group",
                summary: "1x1 stride 2",
                codeLines: [15, 21, 22, 47, 48, 49, 50],
                children: [
                  {
                    id: "layer4.0.downsample.conv",
                    label: "conv1x1",
                    type: "Conv2d",
                    kind: "conv",
                    badges: ["256->512", "s=2"],
                    codeLines: [48, 22],
                  },
                  {
                    id: "layer4.0.downsample.bn",
                    label: "bn",
                    type: "BatchNorm2d",
                    kind: "norm",
                    badges: ["512"],
                    codeLines: [49, 22],
                  },
                ],
              },
              {
                id: "layer4.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [23],
              },
            ],
          },
          {
            id: "layer4.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [10, 11, 12, 13, 14, 17, 18, 19, 20, 23, 53, 54, 63],
            children: [
              {
                id: "layer4.1.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["512->512", "k=3"],
                codeLines: [10, 19, 53],
              },
              {
                id: "layer4.1.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [11, 19],
              },
              {
                id: "layer4.1.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [12, 19],
              },
              {
                id: "layer4.1.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["512->512", "k=3"],
                codeLines: [13, 20],
              },
              {
                id: "layer4.1.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [14, 20],
              },
              {
                id: "layer4.1.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [23],
              },
            ],
          },
        ],
      },
      {
        id: "pool-flatten",
        label: "pool + flatten",
        type: "ClassifierPrep",
        kind: "group",
        summary: "global avg",
        codeLines: [41, 64],
        children: [
          {
            id: "avgpool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            codeLines: [41, 64],
          },
          {
            id: "flatten",
            label: "flatten",
            type: "Flatten",
            kind: "reshape",
            badges: ["512"],
            codeLines: [64],
          },
        ],
      },
      {
        id: "fc",
        label: "fc",
        type: "Linear",
        kind: "linear",
        badges: ["512->1000"],
        codeLines: [42, 65],
      },
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "",
      "",
      "class BasicBlock(nn.Module):",
      "    expansion = 1",
      "",
      "    def __init__(self, in_channels, out_channels, stride=1, downsample=None):",
      "        super().__init__()",
      "        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)",
      "        self.bn1 = nn.BatchNorm2d(out_channels)",
      "        self.relu = nn.ReLU(inplace=True)",
      "        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)",
      "        self.bn2 = nn.BatchNorm2d(out_channels)",
      "        self.downsample = downsample",
      "",
      "    def forward(self, x):",
      "        identity = x",
      "        out = self.conv1(x)",
      "        out = self.bn1(out)",
      "        out = self.relu(out)",
      "        out = self.conv2(out)",
      "        out = self.bn2(out)",
      "        if self.downsample is not None:",
      "            identity = self.downsample(x)",
      "        out = out + identity",
      "        out = self.relu(out)",
      "        return out",
      "",
      "",
      "class ResNet18(nn.Module):",
      "    def __init__(self, num_classes=1000):",
      "        super().__init__()",
      "        self.in_channels = 64",
      "        self.stem = nn.Sequential(",
      "            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),",
      "            nn.BatchNorm2d(64),",
      "            nn.ReLU(inplace=True),",
      "        )",
      "        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)",
      "        self.layer1 = self._make_layer(64, blocks=2, stride=1)",
      "        self.layer2 = self._make_layer(128, blocks=2, stride=2)",
      "        self.layer3 = self._make_layer(256, blocks=2, stride=2)",
      "        self.layer4 = self._make_layer(512, blocks=2, stride=2)",
      "        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))",
      "        self.fc = nn.Linear(512, num_classes)",
      "",
      "    def _make_layer(self, out_channels, blocks, stride):",
      "        downsample = None",
      "        if stride != 1 or self.in_channels != out_channels:",
      "            downsample = nn.Sequential(",
      "                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),",
      "                nn.BatchNorm2d(out_channels),",
      "            )",
      "        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]",
      "        self.in_channels = out_channels",
      "        for _ in range(1, blocks):",
      "            block = BasicBlock(self.in_channels, out_channels)",
      "            layers.append(block)",
      "        stage = nn.Sequential(*layers)",
      "        return stage",
      "",
      "    def forward(self, x):",
      "        x = self.stem(x)",
      "        x = self.maxpool(x)",
      "        x = self.layer1(x)",
      "        x = self.layer2(x)",
      "        x = self.layer3(x)",
      "        x = self.layer4(x)",
      "        x = self.avgpool(x)",
      "        x = torch.flatten(x, 1)",
      "        logits = self.fc(x)",
      "        return logits",
      "",
      "",
      "model = ResNet18(num_classes=1000)",
      "test_input = torch.randn(2, 3, 224, 224)",
      "logits = model(test_input)",
      "# logits: (2, 1000)",
    ],
    jaxCode: resnet18JaxCode,
  },
];

const modelsByPublicationDate = [...models].sort((first, second) =>
  first.paper.publishedDate.localeCompare(second.paper.publishedDate),
);

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path
        d={expanded ? "M4 6l4 4 4-4" : "M6 4l4 4-4 4"}
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function PaneIcon({ pane }: { pane: PaneKey }) {
  if (pane === "architecture") {
    return (
      <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
        <path d="M3 4h4M9 4h4M5 4v4m0 0h6m0-4v8M3 12h4m2 0h4" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.6" />
      </svg>
    );
  }

  if (pane === "paper") {
    return (
      <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
        <path d="M4 2.5h5l3 3v8H4z" fill="none" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.4" />
        <path d="M9 2.5v3h3M6 8h4M6 10.5h4" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.4" />
      </svg>
    );
  }

  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path d="M6 4L3 8l3 4M10 4l3 4-3 4" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7" />
    </svg>
  );
}

function ArrowIcon({ direction }: { direction: "left" | "right" }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path
        d={direction === "left" ? "M10 4 6 8l4 4" : "M6 4l4 4-4 4"}
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path d="M8 2.5v7M5 6.8 8 9.8l3-3M3.5 12.5h9" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7" />
    </svg>
  );
}

function flattenNodes(nodes: ArchNode[]): ArchNode[] {
  return nodes.flatMap((node) => [node, ...(node.children ? flattenNodes(node.children) : [])]);
}

function findNodeById(nodes: ArchNode[], id: string): ArchNode | null {
  for (const node of nodes) {
    if (node.id === id) {
      return node;
    }

    const children = node.children ?? node.lazyChildren?.();
    if (!children) {
      continue;
    }

    const match = findNodeById(children, id);
    if (match) {
      return match;
    }
  }

  return null;
}

function collectDefaultExpandedIds(nodes: ArchNode[]): Set<string> {
  const expandedIds = new Set<string>();

  const visit = (node: ArchNode) => {
    if (node.defaultExpanded) {
      expandedIds.add(node.id);
    }

    const children = node.children ?? node.lazyChildren?.();
    children?.forEach(visit);
  };

  nodes.forEach(visit);
  return expandedIds;
}

function ArchitectureTree({
  nodes,
  selectedId,
  onSelect,
  expanded,
  setExpanded,
  query,
}: {
  nodes: ArchNode[];
  selectedId: string | null;
  onSelect: (node: ArchNode) => void;
  expanded: Set<string>;
  setExpanded: (next: Set<string>) => void;
  query: string;
}) {
  return (
    <div className="tree">
      {nodes.map((node) => (
        <TreeNode
          key={node.id}
          node={node}
          depth={0}
          selectedId={selectedId}
          onSelect={onSelect}
          expanded={expanded}
          setExpanded={setExpanded}
          query={query}
        />
      ))}
    </div>
  );
}

function TreeNode({
  node,
  depth,
  selectedId,
  onSelect,
  expanded,
  setExpanded,
  query,
}: {
  node: ArchNode;
  depth: number;
  selectedId: string | null;
  onSelect: (node: ArchNode) => void;
  expanded: Set<string>;
  setExpanded: (next: Set<string>) => void;
  query: string;
}) {
  const hasChildren = Boolean(node.children?.length || node.lazyChildren);
  const isExpanded = expanded.has(node.id);
  const renderedChildren = isExpanded ? node.children ?? node.lazyChildren?.() : undefined;
  const isSelected = selectedId === node.id;
  const isDimmed =
    query.length > 1 &&
    !`${node.label} ${node.type} ${node.id}`.toLowerCase().includes(query.toLowerCase());

  const toggleExpanded = () => {
    const next = new Set(expanded);
    if (next.has(node.id)) {
      next.delete(node.id);
    } else {
      next.add(node.id);
    }
    setExpanded(next);
  };

  if (node.kind === "head") {
    return (
      <button
        className={`head-tile ${isSelected ? "selected" : ""} ${isDimmed ? "dimmed" : ""}`}
        data-kind={node.kind}
        onClick={() => onSelect(node)}
      >
        <span>{node.label}</span>
        {isSelected ? <small>id: {node.id}</small> : null}
      </button>
    );
  }

  const row = (
    <button
      className={`arch-row ${isSelected ? "selected" : ""} ${isDimmed ? "dimmed" : ""}`}
      data-kind={node.kind}
      style={{ "--depth": depth } as React.CSSProperties}
      onClick={() => onSelect(node)}
    >
      <span className="row-left">
        {hasChildren ? (
          <span
            className="chevron"
            role="button"
            tabIndex={0}
            aria-label={isExpanded ? "Collapse group" : "Expand group"}
            onClick={(event) => {
              event.stopPropagation();
              toggleExpanded();
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                event.stopPropagation();
                toggleExpanded();
              }
            }}
          >
            <ChevronIcon expanded={isExpanded} />
          </span>
        ) : (
          <span className="chevron spacer" />
        )}
        <span className="node-main">
          <strong>{node.label}</strong>
          <span>{node.type}</span>
        </span>
      </span>
      <span className="row-right">
        {node.summary ? <span className="summary">{node.summary}</span> : null}
        {node.badges?.map((badge) => (
          <span className="badge" key={badge}>
            {badge}
          </span>
        ))}
      </span>
    </button>
  );

  return (
    <div className="node-wrap">
      {row}
      {renderedChildren?.length ? (
        <div className={node.id.endsWith(".heads") ? "head-grid" : "children"}>
          {renderedChildren.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              onSelect={onSelect}
              expanded={expanded}
              setExpanded={setExpanded}
              query={query}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function tokenizeCode(line: string) {
  const pattern =
    /(#.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:import|from|class|def|return|for|in|range|float|super|if|not|None|False|True)\b|\b(?:self|x|q|k|v|att|y|B|T|C|identity|logits|train)\b|\b(?:torch|nn|F|jax|jnp|flax|linen)\b|\b[A-Z][A-Za-z0-9_]*\b|\b\d+(?:\.\d+)?\b|->|==|!=|\/\/|\*\*|[@=+\-*/:,.[\]()]|\s+)/g;
  const tokens: Array<{ text: string; className?: string }> = [];
  let cursor = 0;

  for (const match of line.matchAll(pattern)) {
    const text = match[0];
    const index = match.index ?? 0;

    if (index > cursor) {
      tokens.push({ text: line.slice(cursor, index) });
    }

    let className: string | undefined;
    if (text.startsWith("#")) {
      className = "syntax-comment";
    } else if (text.startsWith('"') || text.startsWith("'")) {
      className = "syntax-string";
    } else if (/^(import|from|class|def|return|for|in|range|float|super|if|not|None|False|True)$/.test(text)) {
      className = "syntax-keyword";
    } else if (/^(torch|nn|F|jax|jnp|flax|linen)$/.test(text)) {
      className = "syntax-module";
    } else if (/^(self|x|q|k|v|att|y|B|T|C|identity|logits|train)$/.test(text)) {
      className = "syntax-variable";
    } else if (/^[A-Z][A-Za-z0-9_]*$/.test(text)) {
      className = "syntax-type";
    } else if (/^\d/.test(text)) {
      className = "syntax-number";
    } else if (/^(->|==|!=|\/\/|\*\*|[@=+\-*/:,.[\]()])$/.test(text)) {
      className = "syntax-operator";
    }

    tokens.push({ text, className });
    cursor = index + text.length;
  }

  if (cursor < line.length) {
    tokens.push({ text: line.slice(cursor) });
  }

  return tokens.length > 0 ? tokens : [{ text: " " }];
}

function SyntaxLine({ line }: { line: string }) {
  return (
    <>
      {tokenizeCode(line).map((token, index) => (
        <span className={token.className} key={`${index}-${token.text}`}>
          {token.text}
        </span>
      ))}
    </>
  );
}

function CodeEditor({ model, selected }: { model: ModelSpec; selected: ArchNode | null }) {
  const [language, setLanguage] = useState<CodeLanguage>("pytorch");
  const codeFiles = {
    pytorch: [{ id: "main", fileName: model.fileName, code: model.code }],
    jax: [{ id: "main", fileName: model.jaxFileName, code: model.jaxCode }],
  } satisfies Record<CodeLanguage, Array<{ id: string; fileName: string; code: string[] }>>;
  const filesForLanguage = codeFiles[language];
  const currentFile = filesForLanguage[0];
  const selectedLines = new Set(selected?.codeLines ?? []);

  return (
    <section className="code-pane">
      <div className="pane-toolbar code-toolbar">
        <div className="tab-group file-tab-group">
          <select className="editor-select" aria-label="Select source file" value={currentFile.id} disabled>
            {filesForLanguage.map((file) => (
              <option value={file.id} key={file.id}>
                {file.fileName}
              </option>
            ))}
          </select>
        </div>
        <div className="tab-group language-tab-group">
          <select
            className="language-select"
            aria-label="Select code language"
            value={language}
            onChange={(event) => setLanguage(event.currentTarget.value as CodeLanguage)}
          >
            {(Object.keys(languageLabels) as CodeLanguage[]).map((entry) => (
              <option value={entry} key={entry}>
                {languageLabels[entry]}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="editor">
        {currentFile.code.map((line, index) => {
          const lineNumber = index + 1;
          const highlighted = selectedLines.has(lineNumber);
          return (
            <div className={`code-line ${highlighted ? "highlighted" : ""}`} key={`${lineNumber}-${line}`}>
              <span className="line-number">{lineNumber}</span>
              <code>
                <SyntaxLine line={line} />
              </code>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function PaperPane({ model }: { model: ModelSpec }) {
  return (
    <section className="paper-pane">
      <PdfViewer model={model} />
    </section>
  );
}

function PdfViewer({ model }: { model: ModelSpec }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageCount, setPageCount] = useState(0);
  const [viewerWidth, setViewerWidth] = useState(0);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    setPageNumber(1);
    setPageCount(0);
  }, [model.paper.pdfUrl]);

  useEffect(() => {
    if (!viewerRef.current || typeof ResizeObserver === "undefined") {
      return;
    }

    const updateWidth = () => setViewerWidth(viewerRef.current?.clientWidth ?? 0);
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(viewerRef.current);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    let loadingTask: any;
    let renderTask: any;

    const renderPage = async () => {
      const canvas = canvasRef.current;
      const viewer = viewerRef.current;
      if (!canvas || !viewer) {
        return;
      }

      setStatus("loading");

      try {
        const pdfjs = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        loadingTask = pdfjs.getDocument(model.paper.pdfUrl);
        const pdf = await loadingTask.promise;
        if (cancelled) {
          return;
        }

        setPageCount(pdf.numPages);
        const page = await pdf.getPage(Math.min(pageNumber, pdf.numPages));
        const baseViewport = page.getViewport({ scale: 1 });
        const maxWidth = Math.max((viewerWidth || viewer.clientWidth) - 28, 240);
        const scale = Math.min(Math.max(maxWidth / baseViewport.width, 0.45), 1.8);
        const viewport = page.getViewport({ scale });
        const context = canvas.getContext("2d");

        if (!context) {
          throw new Error("Canvas context unavailable");
        }

        const outputScale = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;
        context.setTransform(outputScale, 0, 0, outputScale, 0, 0);
        context.clearRect(0, 0, viewport.width, viewport.height);

        renderTask = page.render({ canvasContext: context, viewport });
        await renderTask.promise;

        if (!cancelled) {
          setStatus("ready");
        }
      } catch (error) {
        if (!cancelled && !(error instanceof Error && error.name === "RenderingCancelledException")) {
          console.error("PDF render failed", error);
          setStatus("error");
        }
      }
    };

    renderPage();

    return () => {
      cancelled = true;
      renderTask?.cancel();
      void loadingTask?.destroy();
    };
  }, [model.paper.pdfUrl, pageNumber, viewerWidth]);

  return (
    <div className="paper-viewer" ref={viewerRef}>
      <div className="pdf-controls">
        <button
          className="pdf-control-button"
          type="button"
          aria-label="Previous paper page"
          title="Previous page"
          disabled={pageNumber <= 1}
          onClick={() => setPageNumber((current) => Math.max(1, current - 1))}
        >
          <ArrowIcon direction="left" />
        </button>
        <span className="pdf-page-count">
          {pageCount > 0 ? `${pageNumber} / ${pageCount}` : "Loading"}
        </span>
        <button
          className="pdf-control-button"
          type="button"
          aria-label="Next paper page"
          title="Next page"
          disabled={pageCount === 0 || pageNumber >= pageCount}
          onClick={() => setPageNumber((current) => Math.min(pageCount, current + 1))}
        >
          <ArrowIcon direction="right" />
        </button>
        <a
          className="pdf-control-button"
          href={model.paper.pdfUrl}
          download={`${model.id}.pdf`}
          aria-label={`Download PDF for ${model.label}`}
          title={`Download PDF for ${model.label}`}
        >
          <DownloadIcon />
        </a>
      </div>
      <div className="pdf-canvas-wrap">
        {status !== "ready" ? (
          <div className={`pdf-status ${status === "error" ? "error" : ""}`}>
            {status === "error" ? "PDF could not be rendered" : "Rendering PDF"}
          </div>
        ) : null}
        <canvas ref={canvasRef} className="pdf-canvas" aria-label={`${model.paper.title} page ${pageNumber}`} />
      </div>
    </div>
  );
}

export default function Home() {
  const [modelId, setModelId] = useState(modelsByPublicationDate[0].id);
  const model = models.find((entry) => entry.id === modelId) ?? modelsByPublicationDate[0];
  const [expandedByModel, setExpandedByModel] = useState<Record<string, Set<string>>>({});
  const [selectedByModel, setSelectedByModel] = useState<Record<string, ArchNode | null>>({});
  const [visibleColumns, setVisibleColumns] = useState<Record<PaneKey, boolean>>({
    architecture: true,
    paper: false,
    code: true,
  });
  const [query, setQuery] = useState("");

  const paneOrder: PaneKey[] = ["architecture", "code", "paper"];
  const visiblePanes = paneOrder.filter((pane) => visibleColumns[pane]);
  const expanded = expandedByModel[model.id] ?? collectDefaultExpandedIds(model.nodes);
  const selected = selectedByModel[model.id] ?? findNodeById(model.nodes, model.selectedId);

  const updateModel = (nextModelId: string) => {
    const nextModel = models.find((entry) => entry.id === nextModelId);
    const nextExpanded = nextModel ? collectDefaultExpandedIds(nextModel.nodes) : new Set<string>();
    const nextSelected = nextModel ? findNodeById(nextModel.nodes, nextModel.selectedId) : null;

    setModelId(nextModelId);
    setExpandedByModel((current) => ({
      ...current,
      [nextModelId]: nextExpanded,
    }));
    setSelectedByModel((current) => ({
      ...current,
      [nextModelId]: nextSelected,
    }));
    setQuery("");
  };

  const toggleColumn = (pane: PaneKey) => {
    setVisibleColumns((current) => {
      const visibleCount = paneOrder.filter((entry) => current[entry]).length;
      if (current[pane] && visibleCount === 1) {
        return current;
      }

      return {
        ...current,
        [pane]: !current[pane],
      };
    });
  };

  const renderPane = (pane: PaneKey) => {
    if (pane === "architecture") {
      return (
        <section className="architecture-pane">
          <div className="pane-toolbar">
            <div>
              <h1>Architecture</h1>
              <p>{model.stats}</p>
            </div>
          </div>
          <div className="tree-scroll">
            <ArchitectureTree
              nodes={model.nodes}
              selectedId={selected?.id ?? null}
              onSelect={(node) =>
                setSelectedByModel((current) => ({
                  ...current,
                  [model.id]: node,
                }))
              }
              expanded={expanded}
              setExpanded={(next) =>
                setExpandedByModel((current) => ({
                  ...current,
                  [model.id]: next,
                }))
              }
              query={query}
            />
          </div>
        </section>
      );
    }

    if (pane === "paper") {
      return <PaperPane model={model} />;
    }

    return <CodeEditor model={model} selected={selected} />;
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">M</div>
          <span>ModelArchViz</span>
        </div>
        <select
          className="model-select"
          aria-label="Select model"
          value={model.id}
          onChange={(event) => updateModel(event.currentTarget.value)}
          onInput={(event) => updateModel(event.currentTarget.value)}
        >
          {modelsByPublicationDate.map((entry) => (
            <option value={entry.id} key={entry.id}>
              {entry.label} ({entry.paper.publishedDate})
            </option>
          ))}
        </select>
        <div className="header-spacer" />
        <div className="search-field">
          <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
            <path
              d="M7 12a5 5 0 114-2l2.5 2.5"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeWidth="1.6"
            />
          </svg>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={model.id === "gpt2" ? "Find layer or head" : "Find layer"}
          />
        </div>
        <div className="column-toggles" aria-label="Visible columns">
          {paneOrder.map((pane) => (
            <button
              className={`column-toggle ${visibleColumns[pane] ? "active" : ""}`}
              type="button"
              key={pane}
              aria-label={`${visibleColumns[pane] ? "Hide" : "Show"} ${pane} column`}
              aria-pressed={visibleColumns[pane]}
              disabled={visibleColumns[pane] && visiblePanes.length === 1}
              title={`${visibleColumns[pane] ? "Hide" : "Show"} ${pane} column`}
              onClick={() => toggleColumn(pane)}
            >
              <PaneIcon pane={pane} />
            </button>
          ))}
        </div>
      </header>

      <div className="workspace" data-pane-count={visiblePanes.length}>
        {visiblePanes.map((pane, index) => (
          <Fragment key={pane}>
            {index > 0 ? <div className="divider" aria-hidden="true" /> : null}
            {renderPane(pane)}
          </Fragment>
        ))}
      </div>
    </main>
  );
}
