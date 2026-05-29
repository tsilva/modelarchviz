"use client";

import { Fragment, useMemo, useState } from "react";

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
  | "head"
  | "residual"
  | "mlp"
  | "dropout";

type ArchNode = {
  id: string;
  label: string;
  type: string;
  kind: NodeKind;
  summary?: string;
  badges?: string[];
  children?: ArchNode[];
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

const lenet5JaxCode = [
  "import jax",
  "import jax.numpy as jnp",
  "from flax import linen as nn",
  "",
  "",
  "class LeNet5(nn.Module):",
  "    @nn.compact",
  "    def __call__(self, x):",
  "        x = nn.Conv(features=6, kernel_size=(5, 5))(x)",
  "        x = nn.Conv(features=16, kernel_size=(5, 5))(x)",
  "",
  "        x = nn.Dense(features=120)(x)",
  "        x = nn.Dense(features=84)(x)",
  "        logits = nn.Dense(features=10)(x)",
  "",
  "        # x: [batch, 32, 32, 1]",
  "        x = jnp.tanh(nn.Conv(features=6, kernel_size=(5, 5))(x))",
  "        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))",
  "        x = jnp.tanh(nn.Conv(features=16, kernel_size=(5, 5))(x))",
  "        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))",
  "        x = x.reshape((x.shape[0], -1))",
  "",
  "        x = jnp.tanh(nn.Dense(features=120)(x))",
  "        x = jnp.tanh(nn.Dense(features=84)(x))",
  "        return nn.Dense(features=10)(x)",
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
  "",
  "class CausalSelfAttention(nn.Module):",
  "    n_embd: int = 768",
  "    n_head: int = 12",
  "    def __call__(self, x, mask):",
  "        B, T, C = x.shape",
  "        qkv = nn.Dense(3 * C, name='c_attn')(x)",
  "        q, k, v = jnp.split(qkv, 3, axis=2)",
  "        q = q.reshape(B, T, 12, C // 12).transpose(0, 2, 1, 3)",
  "        k = k.reshape(B, T, 12, C // 12).transpose(0, 2, 1, 3)",
  "        v = v.reshape(B, T, 12, C // 12).transpose(0, 2, 1, 3)",
  "        att = (q @ jnp.swapaxes(k, -2, -1)) * (k.shape[-1] ** -0.5)",
  "        att = jnp.where(mask[:, :, :T, :T] == 0, -jnp.inf, att)",
  "        y = nn.softmax(att, axis=-1) @ v",
  "        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)",
  "        return nn.Dense(C, name='c_proj')(y)",
  "",
  "class Block(nn.Module):",
  "    def __call__(self, x):",
  "        x = x + CausalSelfAttention()(nn.LayerNorm(name='ln_1')(x))",
  "        x = x + MLP()(nn.LayerNorm(name='ln_2')(x))",
  "        return x",
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
  "        return nn.relu(y + identity)",
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
  "        return nn.Dense(self.num_classes, name='fc')(x)",
  "",
  "    def _stage(self, x, channels, blocks, stride, train):",
  "        x = BasicBlock(channels, stride, use_projection=stride != 1)(x, train=train)",
  "        for _ in range(1, blocks):",
  "            x = BasicBlock(channels)(x, train=train)",
  "        return x",
];

const models: ModelSpec[] = [
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
      venue: "Proceedings of the IEEE",
      url: "https://ieeexplore.ieee.org/document/726791",
      pdfUrl: "/papers/lenet5",
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
      "        self.fc1 = nn.Linear(16 * 5 * 5, 120)",
      "        self.fc2 = nn.Linear(120, 84)",
      "        self.output = nn.Linear(84, 10)",
      "",
      "    def forward(self, x):",
      "        # x: [batch, 1, 32, 32]",
      "        x = torch.tanh(self.conv1(x))",
      "        x = F.avg_pool2d(x, kernel_size=2)",
      "        x = torch.tanh(self.conv2(x))",
      "        x = F.avg_pool2d(x, kernel_size=2)",
      "        x = torch.flatten(x, start_dim=1)",
      "",
      "        x = torch.tanh(self.fc1(x))",
      "        x = torch.tanh(self.fc2(x))",
      "        return self.output(x)",
    ],
    jaxCode: lenet5JaxCode,
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
      venue: "OpenAI technical report",
      url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      pdfUrl: "/papers/gpt2",
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
        codeLines: [8],
      },
      {
        id: "wpe",
        label: "wpe",
        type: "PositionEmbedding",
        kind: "embedding",
        badges: ["1024", "768"],
        codeLines: [9],
      },
      {
        id: "drop",
        label: "drop",
        type: "Dropout",
        kind: "dropout",
        codeLines: [10],
      },
      ...[0, 1, 2].map((index) => ({
        id: `block.${index}`,
        label: `block.${index}`,
        type: "TransformerBlock",
        kind: "group" as NodeKind,
        summary: "ln + attn + mlp",
        codeLines: [32],
      })),
      {
        id: "block.3",
        label: "block.3",
        type: "TransformerBlock",
        kind: "group",
        summary: "ln + attn + mlp",
        defaultExpanded: true,
        codeLines: [32, 33],
        children: [
          {
            id: "block.3.ln1",
            label: "ln_1",
            type: "LayerNorm",
            kind: "norm",
            badges: ["768"],
            codeLines: [32],
          },
          {
            id: "block.3.attn",
            label: "attn",
            type: "CausalSelfAttention",
            kind: "attention",
            summary: "12 heads",
            defaultExpanded: true,
            codeLines: [15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            children: [
              {
                id: "block.3.attn.c_attn",
                label: "c_attn",
                type: "QKV Projection",
                kind: "attention",
                badges: ["768->2304"],
                codeLines: [15, 20],
              },
              {
                id: "block.3.attn.heads",
                label: "heads",
                type: "Head grid",
                kind: "group",
                summary: "12 x dim 64",
                defaultExpanded: true,
                codeLines: [20, 21, 22, 23, 24, 25, 26],
                children: Array.from({ length: 12 }, (_, index) => ({
                  id: `block.3.attn.head.${index}`,
                  label: `head.${index}`,
                  type: "AttentionHead",
                  kind: "head" as NodeKind,
                  badges: ["q,k,v", "dim 64"],
                  codeLines: [20, 21, 22, 23, 24, 25, 26],
                })),
              },
              {
                id: "block.3.attn.merge",
                label: "merge",
                type: "Concat heads",
                kind: "attention",
                badges: ["12 x 64 -> 768"],
                codeLines: [27],
              },
              {
                id: "block.3.attn.c_proj",
                label: "c_proj",
                type: "Output Projection",
                kind: "attention",
                badges: ["768->768"],
                codeLines: [16, 28],
              },
            ],
          },
          {
            id: "block.3.resid1",
            label: "resid_1",
            type: "Add",
            kind: "residual",
            codeLines: [32],
          },
          {
            id: "block.3.ln2",
            label: "ln_2",
            type: "LayerNorm",
            kind: "norm",
            codeLines: [33],
          },
          {
            id: "block.3.mlp",
            label: "mlp",
            type: "FeedForward",
            kind: "mlp",
            summary: "3072 hidden",
            codeLines: [33],
          },
          {
            id: "block.3.resid2",
            label: "resid_2",
            type: "Add",
            kind: "residual",
            codeLines: [33],
          },
        ],
      },
      ...[4, 5, 6, 7].map((index) => ({
        id: `block.${index}`,
        label: `block.${index}`,
        type: "TransformerBlock",
        kind: "group" as NodeKind,
        summary: "ln + attn + mlp",
        codeLines: [32],
      })),
    ],
    code: [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
      "",
      "",
      "class GPT2Small(nn.Module):",
      "    def __init__(self, vocab_size, n_ctx=1024, n_embd=768):",
      "        self.wte = nn.Embedding(vocab_size, n_embd)",
      "        self.wpe = nn.Embedding(n_ctx, n_embd)",
      "        self.drop = nn.Dropout(0.1)",
      "        self.blocks = nn.ModuleList([Block() for _ in range(12)])",
      "",
      "class CausalSelfAttention(nn.Module):",
      "    def __init__(self, n_embd=768, n_head=12):",
      "        self.c_attn = nn.Linear(n_embd, 3 * n_embd)",
      "        self.c_proj = nn.Linear(n_embd, n_embd)",
      "",
      "    def forward(self, x, mask):",
      "        B, T, C = x.shape",
      "        q, k, v = self.c_attn(x).split(C, dim=2)",
      "        q = q.view(B, T, 12, C // 12).transpose(1, 2)",
      "        k = k.view(B, T, 12, C // 12).transpose(1, 2)",
      "        v = v.view(B, T, 12, C // 12).transpose(1, 2)",
      "        att = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)",
      "        att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))",
      "        y = F.softmax(att, dim=-1) @ v",
      "        y = y.transpose(1, 2).contiguous().view(B, T, C)",
      "        return self.c_proj(y)",
      "",
      "class Block(nn.Module):",
      "    def forward(self, x):",
      "        x = x + self.attn(self.ln_1(x))",
      "        x = x + self.mlp(self.ln_2(x))",
      "        return x",
    ],
    jaxCode: gpt2JaxCode,
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
      venue: "arXiv / CVPR 2016",
      url: "https://arxiv.org/abs/1512.03385",
      pdfUrl: "/papers/resnet18",
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
      },
      {
        id: "layer4",
        label: "layer4",
        type: "ResidualStage",
        kind: "group",
        summary: "2 BasicBlocks",
        badges: ["512 ch", "7x7"],
        codeLines: [40, 63],
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
      "        out = self.relu(self.bn1(self.conv1(x)))",
      "        out = self.bn2(self.conv2(out))",
      "        if self.downsample is not None:",
      "            identity = self.downsample(x)",
      "        out = self.relu(out + identity)",
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
      "            layers.append(BasicBlock(self.in_channels, out_channels))",
      "        return nn.Sequential(*layers)",
      "",
      "    def forward(self, x):",
      "        x = self.stem(x)",
      "        x = self.maxpool(x)",
      "        x = self.layer1(x)",
      "        x = self.layer2(x)",
      "        x = self.layer3(x)",
      "        x = self.layer4(x)",
      "        x = torch.flatten(self.avgpool(x), 1)",
      "        return self.fc(x)",
    ],
    jaxCode: resnet18JaxCode,
  },
];

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

function ExternalIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path d="M6.5 4H4v8h8V9.5M8 3.5h4.5V8M12.3 3.7 7.5 8.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" />
    </svg>
  );
}

function FullscreenIcon({ active }: { active: boolean }) {
  if (active) {
    return (
      <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
        <path d="M6.5 3v3.5H3M9.5 3v3.5H13M6.5 13V9.5H3M9.5 13V9.5H13" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" />
      </svg>
    );
  }

  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path d="M3 6.5V3h3.5M13 6.5V3H9.5M3 9.5V13h3.5M13 9.5V13H9.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" />
    </svg>
  );
}

function flattenNodes(nodes: ArchNode[]): ArchNode[] {
  return nodes.flatMap((node) => [node, ...(node.children ? flattenNodes(node.children) : [])]);
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
  selectedId: string;
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
  selectedId: string;
  onSelect: (node: ArchNode) => void;
  expanded: Set<string>;
  setExpanded: (next: Set<string>) => void;
  query: string;
}) {
  const hasChildren = Boolean(node.children?.length);
  const isExpanded = expanded.has(node.id);
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
        {isSelected ? <span className="line-chip">lines {node.codeLines.join(", ")}</span> : null}
      </span>
    </button>
  );

  return (
    <div className="node-wrap">
      {row}
      {hasChildren && isExpanded ? (
        <div className={node.id.endsWith(".heads") ? "head-grid" : "children"}>
          {node.children?.map((child) => (
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

function CodeEditor({ model, selected }: { model: ModelSpec; selected: ArchNode }) {
  const [language, setLanguage] = useState<CodeLanguage>("pytorch");
  const codeFiles = {
    pytorch: [{ id: "main", fileName: model.fileName, code: model.code }],
    jax: [{ id: "main", fileName: model.jaxFileName, code: model.jaxCode }],
  } satisfies Record<CodeLanguage, Array<{ id: string; fileName: string; code: string[] }>>;
  const filesForLanguage = codeFiles[language];
  const currentFile = filesForLanguage[0];
  const selectedLines = new Set(selected.codeLines);

  return (
    <section className="code-pane">
      <div className="pane-toolbar code-toolbar">
        <div className="tab-group">
          <select className="editor-select" aria-label="Select source file" value={currentFile.id} disabled>
            {filesForLanguage.map((file) => (
              <option value={file.id} key={file.id}>
                {file.fileName}
              </option>
            ))}
          </select>
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
        <div className="selected-note">selected: {selected.id}</div>
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
  const [fullscreen, setFullscreen] = useState(false);

  return (
    <section className={`paper-pane ${fullscreen ? "fullscreen" : ""}`}>
      <div className="pane-toolbar">
        <div>
          <h1>Paper</h1>
          <p>
            {model.paper.year} · {model.paper.venue}
          </p>
        </div>
        <div className="paper-actions">
          <button
            className="paper-tool-button"
            type="button"
            aria-label={fullscreen ? "Leave paper fullscreen" : "Enter paper fullscreen"}
            title={fullscreen ? "Leave paper fullscreen" : "Enter paper fullscreen"}
            onClick={() => setFullscreen((current) => !current)}
          >
            <FullscreenIcon active={fullscreen} />
          </button>
          <a
            className="paper-tool-button"
            href={model.paper.url}
            target="_blank"
            rel="noreferrer"
            aria-label={`Open paper for ${model.label}`}
            title={`Open paper for ${model.label}`}
          >
            <ExternalIcon />
          </a>
        </div>
      </div>
      <div className="paper-viewer">
        <iframe className="paper-frame" src={model.paper.pdfUrl} title={`${model.paper.title} PDF`} />
        <a className="paper-fallback" href={model.paper.pdfUrl} target="_blank" rel="noreferrer">
          Open PDF
        </a>
      </div>
    </section>
  );
}

export default function Home() {
  const [modelId, setModelId] = useState(models[0].id);
  const model = models.find((entry) => entry.id === modelId) ?? models[0];
  const defaultExpanded = useMemo(() => {
    return new Set(flattenNodes(model.nodes).filter((node) => node.defaultExpanded).map((node) => node.id));
  }, [model]);
  const [expandedByModel, setExpandedByModel] = useState<Record<string, Set<string>>>({});
  const [selectedByModel, setSelectedByModel] = useState<Record<string, string>>({});
  const [visibleColumns, setVisibleColumns] = useState<Record<PaneKey, boolean>>({
    architecture: true,
    paper: false,
    code: true,
  });
  const [query, setQuery] = useState("");

  const paneOrder: PaneKey[] = ["architecture", "code", "paper"];
  const visiblePanes = paneOrder.filter((pane) => visibleColumns[pane]);
  const expanded = expandedByModel[model.id] ?? defaultExpanded;
  const selectedId = selectedByModel[model.id] ?? model.selectedId;
  const allNodes = flattenNodes(model.nodes);
  const selected = allNodes.find((node) => node.id === selectedId) ?? allNodes[0];

  const updateModel = (nextModelId: string) => {
    setModelId(nextModelId);
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
              selectedId={selected.id}
              onSelect={(node) =>
                setSelectedByModel((current) => ({
                  ...current,
                  [model.id]: node.id,
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
          aria-label="Select model"
          value={model.id}
          onChange={(event) => updateModel(event.currentTarget.value)}
          onInput={(event) => updateModel(event.currentTarget.value)}
        >
          {models.map((entry) => (
            <option value={entry.id} key={entry.id}>
              {entry.label}
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
