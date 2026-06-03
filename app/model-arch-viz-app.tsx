"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import mlpPythonSource from "./generated/model-code/mlp.py";
import mlpJaxPythonSource from "./generated/model-code/mlp_jax.py";
import rnnPythonSource from "./generated/model-code/elman_rnn.py";
import rnnJaxPythonSource from "./generated/model-code/elman_rnn_jax.py";
import gruPythonSource from "./generated/model-code/gru.py";
import gruJaxPythonSource from "./generated/model-code/gru_jax.py";
import lstmPythonSource from "./generated/model-code/lstm.py";
import lstmJaxPythonSource from "./generated/model-code/lstm_jax.py";
import lenet5PythonSource from "./generated/model-code/lenet5.py";
import lenet5JaxPythonSource from "./generated/model-code/lenet5_jax.py";
import alexnetPythonSource from "./generated/model-code/alexnet.py";
import alexnetJaxPythonSource from "./generated/model-code/alexnet_jax.py";
import googlenetPythonSource from "./generated/model-code/googlenet.py";
import googlenetJaxPythonSource from "./generated/model-code/googlenet_jax.py";
import unetPythonSource from "./generated/model-code/unet.py";
import unetJaxPythonSource from "./generated/model-code/unet_jax.py";
import transformerPythonSource from "./generated/model-code/transformer.py";
import transformerJaxPythonSource from "./generated/model-code/transformer_jax.py";
import bertPythonSource from "./generated/model-code/bert_base.py";
import bertJaxPythonSource from "./generated/model-code/bert_base_jax.py";
import gpt2PythonSource from "./generated/model-code/gpt2_attention.py";
import gpt2JaxPythonSource from "./generated/model-code/gpt2_attention_jax.py";
import vitPythonSource from "./generated/model-code/vit_b16.py";
import vitJaxPythonSource from "./generated/model-code/vit_b16_jax.py";
import resnet18PythonSource from "./generated/model-code/resnet18.py";
import resnet18JaxPythonSource from "./generated/model-code/resnet18_jax.py";
import widenetPythonSource from "./generated/model-code/widenet.py";
import widenetJaxPythonSource from "./generated/model-code/widenet_jax.py";
import densenetPythonSource from "./generated/model-code/densenet.py";
import densenetJaxPythonSource from "./generated/model-code/densenet_jax.py";
import efficientnetPythonSource from "./generated/model-code/efficientnet.py";
import efficientnetJaxPythonSource from "./generated/model-code/efficientnet_jax.py";

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

const githubRepository = (process.env.NEXT_PUBLIC_GITHUB_REPOSITORY ?? "tsilva/modelarchviz")
  .replace(/^https?:\/\/github\.com\//, "")
  .replace(/\/$/, "");
const githubBranch = process.env.NEXT_PUBLIC_GITHUB_BRANCH ?? "main";

function codeLines(source: string) {
  const trimmed = source.trimEnd();
  return trimmed.split("\n");
}

function notebookFileName(fileName: string) {
  return fileName.replace(/\.py$/, ".ipynb");
}

function colabUrl(notebookName: string) {
  return `https://colab.research.google.com/github/${githubRepository}/blob/${githubBranch}/public/notebooks/${notebookName}`;
}

const completedPdfPrefetches = new Set<string>();
const pendingPdfPrefetches = new Map<string, Promise<void>>();
const pdfPrefetchHints = new Set<string>();

function addPdfPrefetchHint(pdfUrl: string) {
  if (typeof document === "undefined" || pdfPrefetchHints.has(pdfUrl)) {
    return;
  }

  const hint = document.createElement("link");
  hint.rel = "prefetch";
  hint.as = "fetch";
  hint.href = pdfUrl;
  hint.type = "application/pdf";
  hint.dataset.modelarchvizPdfPrefetch = pdfUrl;
  document.head.appendChild(hint);
  pdfPrefetchHints.add(pdfUrl);
}

function prefetchPdf(pdfUrl: string, signal: AbortSignal) {
  addPdfPrefetchHint(pdfUrl);

  if (completedPdfPrefetches.has(pdfUrl) || pendingPdfPrefetches.has(pdfUrl)) {
    return;
  }

  const prefetch = fetch(pdfUrl, { cache: "force-cache", signal })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`PDF prefetch failed with ${response.status}`);
      }

      return response.blob();
    })
    .then(() => {
      completedPdfPrefetches.add(pdfUrl);
    })
    .catch((error: unknown) => {
      if (signal.aborted) {
        return;
      }

      console.warn("PDF prefetch failed", error);
    })
    .finally(() => {
      pendingPdfPrefetches.delete(pdfUrl);
    });

  pendingPdfPrefetches.set(pdfUrl, prefetch);
}

function makeTransformerEncoderBlock(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `encoder.${index}`,
    label: `encoder.${index}`,
    type: "EncoderLayer",
    kind: "group",
    summary: "self-attn + ffn",
    defaultExpanded,
    codeLines: [35, 36, 42, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 57, 58, 61, 62, 63, 64, 128, 129],
    lazyChildren: () => [
      {
        id: `encoder.${index}.self_attn`,
        label: "self_attn",
        type: "MultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "d=512"],
        codeLines: [45, 56],
      },
      {
        id: `encoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [51, 57, 58],
      },
      {
        id: `encoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        codeLines: [46, 47, 48, 49, 50, 61],
      },
      {
        id: `encoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [52, 62, 63, 64],
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
    codeLines: [29, 30, 31, 32, 33, 34, 35],
    lazyChildren: () => [
      {
        id: `step.${index}.input_to_hidden`,
        label: "input_to_hidden",
        type: "Linear",
        kind: "linear",
        badges: ["32->64"],
        codeLines: [16, 31],
      },
      {
        id: `step.${index}.hidden_to_hidden`,
        label: "hidden_to_hidden",
        type: "RecurrentLinear",
        kind: "recurrent",
        badges: ["64->64", "shared"],
        codeLines: [17, 32],
      },
      {
        id: `step.${index}.update`,
        label: "add + tanh",
        type: "StateUpdate",
        kind: "activation",
        badges: ["h_t"],
        codeLines: [33, 34],
      },
      {
        id: `step.${index}.state`,
        label: "state history",
        type: "AppendHidden",
        kind: "recurrent",
        badges: ["store h_t"],
        codeLines: [35, 39],
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
    codeLines: [21, 23, 24, 25, 26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 71, 72, 73, 74],
    lazyChildren: () => [
      {
        id: `step.${index}.update_gate`,
        label: "update gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["z_t"],
        codeLines: [14, 15, 23, 24, 25, 26],
      },
      {
        id: `step.${index}.reset_gate`,
        label: "reset gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["r_t"],
        codeLines: [16, 17, 29, 30, 31, 32],
      },
      {
        id: `step.${index}.candidate`,
        label: "candidate",
        type: "TanhState",
        kind: "activation",
        badges: ["n_t"],
        codeLines: [18, 19, 35, 36, 37, 38, 39],
      },
      {
        id: `step.${index}.mix`,
        label: "state mix",
        type: "GatedInterpolation",
        kind: "recurrent",
        badges: ["h_t"],
        codeLines: [42, 43, 44, 45],
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
    codeLines: [67, 68, 74, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 90, 91, 92, 95, 96, 97, 100, 101, 102, 103, 134, 135],
    lazyChildren: () => [
      {
        id: `decoder.${index}.masked_self_attn`,
        label: "masked self_attn",
        type: "CausalMultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "causal"],
        codeLines: [77, 90, 148, 149, 150],
      },
      {
        id: `decoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [84, 91, 92],
      },
      {
        id: `decoder.${index}.cross_attn`,
        label: "cross_attn",
        type: "EncoderDecoderAttention",
        kind: "attention",
        badges: ["Q=decoder", "K,V=encoder"],
        codeLines: [78, 95],
      },
      {
        id: `decoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [85, 96, 97],
      },
      {
        id: `decoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        codeLines: [79, 80, 81, 82, 83, 100],
      },
      {
        id: `decoder.${index}.norm3`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [86, 101, 102, 103],
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
    codeLines: [36, 37, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 58, 59, 60, 61, 64, 65, 66, 67, 68, 89, 90],
    lazyChildren: () => [
      {
        id: `encoder.layer.${index}.self_attn`,
        label: "self_attn",
        type: "BidirectionalSelfAttention",
        kind: "attention",
        badges: ["12 heads", "768"],
        codeLines: [46, 58],
      },
      {
        id: `encoder.layer.${index}.attn_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [47, 59, 60, 61],
      },
      {
        id: `encoder.layer.${index}.intermediate`,
        label: "intermediate",
        type: "Dense + GELU",
        kind: "mlp",
        badges: ["768->3072"],
        codeLines: [48, 49, 50, 51, 52, 64],
      },
      {
        id: `encoder.layer.${index}.output`,
        label: "output",
        type: "Dense",
        kind: "mlp",
        badges: ["3072->768"],
        codeLines: [48, 49, 50, 51, 52, 64],
      },
      {
        id: `encoder.layer.${index}.output_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [53, 65, 66, 67, 68],
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
    codeLines: [89, 90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 103, 105, 106, 107, 110, 111, 112, 113, 34, 35],
    lazyChildren: () => [
      {
        id: `block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        codeLines: [94, 105],
      },
      {
        id: `block.${index}.attn`,
        label: "attn",
        type: "CausalSelfAttention",
        kind: "attention",
        summary: "12 heads",
        codeLines: [43, 44, 49, 52, 53, 54, 56, 57, 59, 60, 61, 62, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 106],
        lazyChildren: () => [
          {
            id: `block.${index}.attn.c_attn`,
            label: "c_attn",
            type: "QKV Projection",
            kind: "attention",
            badges: ["768->2304"],
            codeLines: [53, 59, 60],
          },
          {
            id: `block.${index}.attn.heads`,
            label: "heads",
            type: "Head grid",
            kind: "group",
            summary: "12 x dim 64",
            codeLines: [61, 62, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 81],
            lazyChildren: () =>
              Array.from({ length: 12 }, (_, headIndex) => ({
                id: `block.${index}.attn.head.${headIndex}`,
                label: `head.${headIndex}`,
                type: "AttentionHead",
                kind: "head" as NodeKind,
                badges: ["q,k,v", "dim 64"],
                codeLines: [61, 62, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 81],
              })),
          },
          {
            id: `block.${index}.attn.merge`,
            label: "merge",
            type: "Concat heads",
            kind: "attention",
            badges: ["12 x 64 -> 768"],
            codeLines: [82, 83, 84],
          },
          {
            id: `block.${index}.attn.c_proj`,
            label: "c_proj",
            type: "Output Projection",
            kind: "attention",
            badges: ["768->768"],
            codeLines: [54, 85, 86],
          },
        ],
      },
      {
        id: `block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        codeLines: [107],
      },
      {
        id: `block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        codeLines: [96, 110],
      },
      {
        id: `block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        summary: "3072 hidden",
        codeLines: [97, 98, 99, 100, 101, 111],
      },
      {
        id: `block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        codeLines: [112, 113],
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
    codeLines: [20, 21, 24, 25, 27, 30, 31, 33, 36, 37, 38, 44, 45, 46, 47, 50, 51, config.callLine, config.forwardLine],
    lazyChildren: () => [
      {
        id: `${config.id}.branch1`,
        label: "branch1",
        type: "1x1 Conv",
        kind: "conv",
        badges: [`${config.inputChannels}->${config.branch1Channels}`],
        codeLines: [20, 21, 44, config.callLine, config.forwardLine],
      },
      {
        id: `${config.id}.branch3`,
        label: "branch3",
        type: "1x1 reduce + 3x3",
        kind: "group",
        summary: "medium receptive field",
        codeLines: [24, 25, 27, 28, 45, config.callLine, config.forwardLine],
        children: [
          {
            id: `${config.id}.branch3.reduce`,
            label: "reduce",
            type: "1x1 Conv",
            kind: "conv",
            badges: [`${config.inputChannels}->${config.branch3Reduce}`],
            codeLines: [24, 25],
          },
          {
            id: `${config.id}.branch3.conv`,
            label: "conv3x3",
            type: "3x3 Conv",
            kind: "conv",
            badges: [`${config.branch3Reduce}->${config.branch3Channels}`],
            codeLines: [27, 28],
          },
        ],
      },
      {
        id: `${config.id}.branch5`,
        label: "branch5",
        type: "1x1 reduce + 5x5",
        kind: "group",
        summary: "wide receptive field",
        codeLines: [30, 31, 33, 34, 46, config.callLine, config.forwardLine],
        children: [
          {
            id: `${config.id}.branch5.reduce`,
            label: "reduce",
            type: "1x1 Conv",
            kind: "conv",
            badges: [`${config.inputChannels}->${config.branch5Reduce}`],
            codeLines: [30, 31],
          },
          {
            id: `${config.id}.branch5.conv`,
            label: "conv5x5",
            type: "5x5 Conv",
            kind: "conv",
            badges: [`${config.branch5Reduce}->${config.branch5Channels}`],
            codeLines: [33, 34],
          },
        ],
      },
      {
        id: `${config.id}.pool_proj`,
        label: "pool_proj",
        type: "3x3 Pool + 1x1 Conv",
        kind: "pool",
        badges: [`${config.inputChannels}->${config.poolChannels}`],
        codeLines: [36, 37, 38, 39, 47, config.callLine, config.forwardLine],
      },
      {
        id: `${config.id}.concat`,
        label: "concat",
        type: "ChannelConcat",
        kind: "concat",
        badges: [`${outputChannels} channels`],
        codeLines: [50, 51],
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
    codeLines: [30, 31, 37, 40, 41, 42, 43, 44, 45, 46, 47, 49, 51, 52, 53, 56, 57, 58, 59, 89, 90],
    lazyChildren: () => [
      {
        id: `encoder.block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        codeLines: [40, 51],
      },
      {
        id: `encoder.block.${index}.attn`,
        label: "attn",
        type: "MultiHeadSelfAttention",
        kind: "attention",
        badges: ["12 heads", "197 tokens"],
        codeLines: [41, 52],
      },
      {
        id: `encoder.block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        codeLines: [53],
      },
      {
        id: `encoder.block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        codeLines: [42, 56],
      },
      {
        id: `encoder.block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        badges: ["768->3072->768"],
        codeLines: [43, 44, 45, 46, 47, 57],
      },
      {
        id: `encoder.block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        codeLines: [58],
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
        badges: ["784->784"],
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
            badges: ["128->128"],
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
            badges: ["128->128"],
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
        badges: ["10->10"],
        codeLines: [30, 31, 34, 37, 39],
      },
    ],
    code: codeLines(mlpPythonSource),
    jaxCode: codeLines(mlpJaxPythonSource),
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
        codeLines: [20, 22, 28, 30, 46, 47],
      },
      {
        id: "recurrent_loop",
        label: "Recurrent Loop",
        type: "UnrolledRNN",
        kind: "group",
        summary: "shared cell over time",
        badges: ["tanh"],
        defaultExpanded: true,
        codeLines: [20, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        children: [
          {
            id: "h0",
            label: "h0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["64 hidden"],
            codeLines: [22, 23, 24],
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
        codeLines: [18, 37, 38],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "all states"],
        codeLines: [38, 39, 40, 41, 47, 48, 49, 51],
      },
    ],
    code: codeLines(rnnPythonSource),
    jaxCode: codeLines(rnnJaxPythonSource),
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
        codeLines: [62, 64, 70, 72, 85, 86],
      },
      {
        id: "cell_params",
        label: "GRU Cell Params",
        type: "GatedRecurrentCell",
        kind: "group",
        summary: "6 affine projections",
        badges: ["z", "r", "n"],
        defaultExpanded: true,
        codeLines: [5, 6, 13, 14, 15, 16, 17, 18, 19, 21],
        children: [
          {
            id: "cell_params.update",
            label: "update params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_z", "h_z"],
            codeLines: [14, 15, 23, 24],
          },
          {
            id: "cell_params.reset",
            label: "reset params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_r", "h_r"],
            codeLines: [16, 17, 29, 30],
          },
          {
            id: "cell_params.candidate",
            label: "candidate params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_n", "h_n"],
            codeLines: [18, 19, 36, 37],
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
        codeLines: [62, 64, 65, 66, 69, 70, 71, 72, 73, 74],
        children: [
          {
            id: "h0",
            label: "h0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["64 hidden"],
            codeLines: [64, 65, 66],
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
        codeLines: [60, 76, 77],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "all states"],
        codeLines: [77, 78, 79, 80, 86, 87, 88, 90],
      },
    ],
    code: codeLines(gruPythonSource),
    jaxCode: codeLines(gruJaxPythonSource),
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
        codeLines: [77, 79, 86, 88, 104, 105],
      },
      {
        id: "cell_params",
        label: "LSTM Cell Params",
        type: "GatedMemoryCell",
        kind: "group",
        summary: "8 affine projections",
        badges: ["i", "f", "g", "o"],
        defaultExpanded: true,
        codeLines: [5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23],
        children: [
          {
            id: "cell_params.input",
            label: "input params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_i", "h_i"],
            codeLines: [14, 15, 28, 29],
          },
          {
            id: "cell_params.forget",
            label: "forget params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_f", "h_f"],
            codeLines: [16, 17, 34, 35],
          },
          {
            id: "cell_params.candidate",
            label: "candidate params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_g", "h_g"],
            codeLines: [18, 19, 40, 41],
          },
          {
            id: "cell_params.output",
            label: "output params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_o", "h_o"],
            codeLines: [20, 21, 46, 47],
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
        codeLines: [77, 79, 80, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93],
        children: [
          {
            id: "state0",
            label: "h0 + c0",
            type: "ZeroStates",
            kind: "recurrent",
            badges: ["64 hidden", "64 cell"],
            codeLines: [79, 80, 81, 82],
          },
          {
            id: "step.0",
            label: "step.0",
            type: "LSTMCell",
            kind: "group",
            summary: "i/f/g/o gates",
            defaultExpanded: true,
            codeLines: [23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 90, 91, 92, 93],
            children: [
              {
                id: "step.0.input_gate",
                label: "input gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["i_t"],
                codeLines: [14, 15, 27, 28, 29, 30, 31],
              },
              {
                id: "step.0.forget_gate",
                label: "forget gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["f_t"],
                codeLines: [16, 17, 33, 34, 35, 36, 37],
              },
              {
                id: "step.0.candidate",
                label: "candidate",
                type: "TanhState",
                kind: "activation",
                badges: ["g_t"],
                codeLines: [18, 19, 39, 40, 41, 42, 43],
              },
              {
                id: "step.0.cell_update",
                label: "cell update",
                type: "MemoryUpdate",
                kind: "recurrent",
                badges: ["c_t"],
                codeLines: [51, 52, 53, 54],
              },
              {
                id: "step.0.output_gate",
                label: "output gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["o_t"],
                codeLines: [20, 21, 45, 46, 47, 48, 49],
              },
              {
                id: "step.0.hidden_update",
                label: "hidden update",
                type: "GatedReadout",
                kind: "recurrent",
                badges: ["h_t"],
                codeLines: [56, 57, 58, 59],
              },
            ],
          },
          ...[1, 2, 3, 4, 5, 6, 7].map((index) => ({
            id: `step.${index}`,
            label: `step.${index}`,
            type: "LSTMCell",
            kind: "recurrent" as NodeKind,
            summary: "same gates",
            codeLines: [87, 88, 89, 90, 91, 92, 93],
          })),
        ],
      },
      {
        id: "readout",
        label: "readout",
        type: "Linear",
        kind: "linear",
        badges: ["64->10", "last h"],
        codeLines: [75, 95, 96],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "hidden states"],
        codeLines: [96, 97, 98, 99, 105, 106, 107, 109],
      },
    ],
    code: codeLines(lstmPythonSource),
    jaxCode: codeLines(lstmJaxPythonSource),
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
        codeLines: [18],
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "Group",
        kind: "group",
        summary: "6 ops",
        defaultExpanded: true,
        codeLines: [11, 12, 17, 18, 19, 20, 21, 22],
        children: [
          {
            id: "features.conv1",
            label: "conv1",
            type: "Conv2d",
            kind: "conv",
            badges: ["1->6", "k=5", "out 6x28x28"],
            codeLines: [11, 19],
          },
          {
            id: "features.tanh1",
            label: "tanh1",
            type: "Tanh",
            kind: "activation",
            codeLines: [19],
          },
          {
            id: "features.pool1",
            label: "pool1",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["k=2", "out 6x14x14"],
            codeLines: [20],
          },
          {
            id: "features.conv2",
            label: "conv2",
            type: "Conv2d",
            kind: "conv",
            badges: ["6->16", "k=5", "out 16x10x10"],
            codeLines: [12, 21],
          },
          {
            id: "features.tanh2",
            label: "tanh2",
            type: "Tanh",
            kind: "activation",
            codeLines: [21],
          },
          {
            id: "features.pool2",
            label: "pool2",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["k=2", "out 16x5x5"],
            codeLines: [22],
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["400"],
        codeLines: [23],
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "Group",
        kind: "group",
        summary: "4 ops",
        defaultExpanded: true,
        codeLines: [13, 14, 15, 25, 28, 29],
        children: [
          {
            id: "classifier.fc1",
            label: "fc1",
            type: "Linear",
            kind: "linear",
            badges: ["400->120"],
            codeLines: [12, 25],
          },
          {
            id: "classifier.tanh3",
            label: "tanh3",
            type: "Tanh",
            kind: "activation",
            codeLines: [25],
          },
          {
            id: "classifier.fc2",
            label: "fc2",
            type: "Linear",
            kind: "linear",
            badges: ["120->84"],
            codeLines: [13, 28],
          },
          {
            id: "classifier.output",
            label: "output",
            type: "Linear",
            kind: "linear",
            badges: ["84->10"],
            codeLines: [15, 29],
          },
        ],
      },
    ],
    code: codeLines(lenet5PythonSource),
    jaxCode: codeLines(lenet5JaxPythonSource),
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
        codeLines: [55],
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "Group",
        kind: "group",
        summary: "13 ops",
        defaultExpanded: true,
        codeLines: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 44],
        children: [
          {
            id: "features.conv1",
            label: "conv1",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->96", "k=11", "s=4", "55x55"],
            codeLines: [15, 44],
          },
          {
            id: "features.relu1",
            label: "relu1",
            type: "ReLU",
            kind: "activation",
            codeLines: [16],
          },
          {
            id: "features.lrn1",
            label: "lrn1",
            type: "LocalResponseNorm",
            kind: "norm",
            badges: ["size=5"],
            codeLines: [17],
          },
          {
            id: "features.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "27x27"],
            codeLines: [18],
          },
          {
            id: "features.conv2",
            label: "conv2",
            type: "Conv2d",
            kind: "conv",
            badges: ["96->256", "k=5", "27x27"],
            codeLines: [19],
          },
          {
            id: "features.relu2",
            label: "relu2",
            type: "ReLU",
            kind: "activation",
            codeLines: [20],
          },
          {
            id: "features.lrn2",
            label: "lrn2",
            type: "LocalResponseNorm",
            kind: "norm",
            badges: ["size=5"],
            codeLines: [21],
          },
          {
            id: "features.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "13x13"],
            codeLines: [22],
          },
          {
            id: "features.conv3",
            label: "conv3",
            type: "Conv2d",
            kind: "conv",
            badges: ["256->384", "k=3"],
            codeLines: [23],
          },
          {
            id: "features.relu3",
            label: "relu3",
            type: "ReLU",
            kind: "activation",
            codeLines: [24],
          },
          {
            id: "features.conv4",
            label: "conv4",
            type: "Conv2d",
            kind: "conv",
            badges: ["384->384", "k=3"],
            codeLines: [25],
          },
          {
            id: "features.relu4",
            label: "relu4",
            type: "ReLU",
            kind: "activation",
            codeLines: [26],
          },
          {
            id: "features.conv5",
            label: "conv5",
            type: "Conv2d",
            kind: "conv",
            badges: ["384->256", "k=3"],
            codeLines: [27],
          },
          {
            id: "features.relu5",
            label: "relu5",
            type: "ReLU",
            kind: "activation",
            codeLines: [28],
          },
          {
            id: "features.pool5",
            label: "pool5",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "6x6"],
            codeLines: [29],
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["9216"],
        codeLines: [46],
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "Group",
        kind: "group",
        summary: "7 ops",
        defaultExpanded: true,
        codeLines: [33, 34, 35, 36, 37, 38, 39, 40, 49],
        children: [
          {
            id: "classifier.drop1",
            label: "dropout1",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            codeLines: [34],
          },
          {
            id: "classifier.fc6",
            label: "fc6",
            type: "Linear",
            kind: "linear",
            badges: ["9216->4096"],
            codeLines: [35, 49],
          },
          {
            id: "classifier.relu6",
            label: "relu6",
            type: "ReLU",
            kind: "activation",
            codeLines: [36],
          },
          {
            id: "classifier.drop2",
            label: "dropout2",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            codeLines: [37],
          },
          {
            id: "classifier.fc7",
            label: "fc7",
            type: "Linear",
            kind: "linear",
            badges: ["4096->4096"],
            codeLines: [38],
          },
          {
            id: "classifier.relu7",
            label: "relu7",
            type: "ReLU",
            kind: "activation",
            codeLines: [39],
          },
          {
            id: "classifier.fc8",
            label: "fc8",
            type: "Linear",
            kind: "linear",
            badges: ["4096->1000"],
            codeLines: [40],
          },
        ],
      },
    ],
    code: codeLines(alexnetPythonSource),
    jaxCode: codeLines(alexnetJaxPythonSource),
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
        codeLines: [117],
      },
      {
        id: "stem",
        label: "stem",
        type: "ConvPoolStem",
        kind: "group",
        summary: "7x7 + 1x1 + 3x3",
        defaultExpanded: true,
        codeLines: [63, 64, 65, 66, 67, 68, 69, 70, 71, 88],
        children: [
          {
            id: "stem.conv7",
            label: "conv7",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->64", "k=7", "s=2"],
            codeLines: [64, 88],
          },
          {
            id: "stem.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            codeLines: [66, 88],
          },
          {
            id: "stem.conv1",
            label: "conv1x1",
            type: "Conv2d",
            kind: "conv",
            badges: ["64->64"],
            codeLines: [67, 88],
          },
          {
            id: "stem.conv3",
            label: "conv3x3",
            type: "Conv2d",
            kind: "conv",
            badges: ["64->192"],
            codeLines: [69, 88],
          },
          {
            id: "stem.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            codeLines: [71, 88],
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
        codeLines: [73, 74, 91, 92, 93],
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
            codeLines: [93],
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
        codeLines: [75, 76, 77, 78, 79, 96, 97, 98, 99, 100, 101],
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
            codeLines: [101],
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
        codeLines: [80, 81, 104, 105],
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
        codeLines: [82, 83, 84, 106, 107, 110, 111],
        children: [
          {
            id: "classifier.avgpool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            codeLines: [82, 106],
          },
          {
            id: "classifier.flatten",
            label: "flatten",
            type: "Flatten",
            kind: "reshape",
            badges: ["1024"],
            codeLines: [107],
          },
          {
            id: "classifier.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.4"],
            codeLines: [83, 110],
          },
          {
            id: "classifier.fc",
            label: "fc",
            type: "Linear",
            kind: "linear",
            badges: ["1024->1000"],
            codeLines: [84, 111],
          },
        ],
      },
    ],
    code: codeLines(googlenetPythonSource),
    jaxCode: codeLines(googlenetJaxPythonSource),
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
        codeLines: [74],
      },
      {
        id: "contracting",
        label: "Contracting Path",
        type: "Encoder",
        kind: "group",
        summary: "4 DoubleConv blocks",
        defaultExpanded: true,
        codeLines: [24, 28, 31, 32, 33, 34, 35, 36, 49, 50, 52, 53],
        children: [
          {
            id: "contracting.down1",
            label: "down1",
            type: "DoubleConv",
            kind: "conv",
            badges: ["1->64", "572x572"],
            codeLines: [10, 11, 12, 13, 14, 19, 24, 49],
          },
          {
            id: "contracting.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [28, 50],
          },
          {
            id: "contracting.down2",
            label: "down2",
            type: "DoubleConv",
            kind: "conv",
            badges: ["64->128"],
            codeLines: [31, 50],
          },
          {
            id: "contracting.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [32, 52],
          },
          {
            id: "contracting.down3",
            label: "down3",
            type: "DoubleConv",
            kind: "conv",
            badges: ["128->256"],
            codeLines: [33, 52],
          },
          {
            id: "contracting.pool3",
            label: "pool3",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [34, 53],
          },
          {
            id: "contracting.down4",
            label: "down4",
            type: "DoubleConv",
            kind: "conv",
            badges: ["256->512"],
            codeLines: [35, 53],
          },
          {
            id: "contracting.pool4",
            label: "pool4",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            codeLines: [36, 54],
          },
        ],
      },
      {
        id: "bottleneck",
        label: "bottleneck",
        type: "DoubleConv",
        kind: "conv",
        badges: ["512->1024"],
        codeLines: [37, 54],
      },
      {
        id: "expansive",
        label: "Expansive Path",
        type: "Decoder",
        kind: "group",
        summary: "upsample + concat skips",
        defaultExpanded: true,
        codeLines: [38, 39, 40, 41, 42, 43, 44, 45, 55, 56, 57, 58, 59, 62, 65, 66, 67, 68, 69, 70],
        children: [
          {
            id: "expansive.up4",
            label: "up4",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["1024->512", "x2"],
            codeLines: [38, 55],
          },
          {
            id: "expansive.up4.skip",
            label: "skip d4",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [56],
          },
          {
            id: "expansive.dec4",
            label: "dec4",
            type: "DoubleConv",
            kind: "conv",
            badges: ["1024->512"],
            codeLines: [39, 57],
          },
          {
            id: "expansive.up3",
            label: "up3",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["512->256", "x2"],
            codeLines: [40, 58],
          },
          {
            id: "expansive.up3.skip",
            label: "skip d3",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [59],
          },
          {
            id: "expansive.dec3",
            label: "dec3",
            type: "DoubleConv",
            kind: "conv",
            badges: ["512->256"],
            codeLines: [41, 62],
          },
          {
            id: "expansive.up2",
            label: "up2",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["256->128", "x2"],
            codeLines: [42, 65],
          },
          {
            id: "expansive.up2.skip",
            label: "skip d2",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [66],
          },
          {
            id: "expansive.dec2",
            label: "dec2",
            type: "DoubleConv",
            kind: "conv",
            badges: ["256->128"],
            codeLines: [43, 67],
          },
          {
            id: "expansive.up1",
            label: "up1",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["128->64", "x2"],
            codeLines: [44, 68],
          },
          {
            id: "expansive.up1.skip",
            label: "skip d1",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            codeLines: [69],
          },
          {
            id: "expansive.dec1",
            label: "dec1",
            type: "DoubleConv",
            kind: "conv",
            badges: ["128->64"],
            codeLines: [45, 70],
          },
        ],
      },
      {
        id: "out_conv",
        label: "out_conv",
        type: "Conv2d",
        kind: "conv",
        badges: ["64->2", "1x1"],
        codeLines: [46, 71],
      },
    ],
    code: codeLines(unetPythonSource),
    jaxCode: codeLines(unetJaxPythonSource),
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
        codeLines: [101],
      },
      {
        id: "tgt.input",
        label: "target input",
        type: "TokenIds",
        kind: "input",
        badges: ["target", "shifted"],
        codeLines: [102],
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position",
        defaultExpanded: true,
        codeLines: [78, 79, 80, 86, 90],
        children: [
          {
            id: "src_embed",
            label: "src_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            codeLines: [78, 86],
          },
          {
            id: "tgt_embed",
            label: "tgt_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            codeLines: [79, 90],
          },
          {
            id: "positional_encoding",
            label: "positional",
            type: "SinusoidalEncoding",
            kind: "embedding",
            badges: ["absolute"],
            codeLines: [5, 6, 14, 15, 16, 17, 18, 19, 80, 86, 90],
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
        codeLines: [81, 87, 88],
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
        codeLines: [82, 91, 92],
        children: Array.from({ length: 6 }, (_, index) => makeTransformerDecoderBlock(index, index === 0)),
      },
      {
        id: "generator",
        label: "generator",
        type: "Linear",
        kind: "linear",
        badges: ["512->vocab"],
        codeLines: [83, 95],
      },
    ],
    code: codeLines(transformerPythonSource),
    jaxCode: codeLines(transformerJaxPythonSource),
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
        codeLines: [81],
      },
      {
        id: "token_type_ids",
        label: "token_type_ids",
        type: "SegmentIds",
        kind: "input",
        badges: ["sentence A/B"],
        codeLines: [82],
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position + segment",
        defaultExpanded: true,
        codeLines: [15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 58, 66],
        children: [
          {
            id: "embeddings.word",
            label: "word",
            type: "WordPieceEmbedding",
            kind: "embedding",
            badges: ["30522", "768"],
            codeLines: [15, 24],
          },
          {
            id: "embeddings.position",
            label: "position",
            type: "PositionEmbedding",
            kind: "embedding",
            badges: ["512", "768"],
            codeLines: [16, 23, 25],
          },
          {
            id: "embeddings.segment",
            label: "segment",
            type: "TokenTypeEmbedding",
            kind: "embedding",
            badges: ["2", "768"],
            codeLines: [17, 26],
          },
          {
            id: "embeddings.norm",
            label: "norm",
            type: "LayerNorm",
            kind: "norm",
            codeLines: [18, 27],
          },
          {
            id: "embeddings.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.1"],
            codeLines: [19, 27],
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
        codeLines: [59, 67, 68],
        children: Array.from({ length: 12 }, (_, index) => makeBertLayer(index, index === 3)),
      },
      {
        id: "pooler",
        label: "pooler",
        type: "CLSProjection",
        kind: "linear",
        badges: ["CLS", "768->768"],
        codeLines: [60, 69, 84],
      },
      {
        id: "mlm_head",
        label: "mlm_head",
        type: "MaskedLMHead",
        kind: "head",
        badges: ["768->30522"],
        codeLines: [61, 70, 84],
      },
    ],
    code: codeLines(bertPythonSource),
    jaxCode: codeLines(bertJaxPythonSource),
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
        codeLines: [16],
      },
      {
        id: "wpe",
        label: "wpe",
        type: "PositionEmbedding",
        kind: "embedding",
        badges: ["1024", "768"],
        codeLines: [17],
      },
      {
        id: "drop",
        label: "drop",
        type: "Dropout",
        kind: "dropout",
        codeLines: [18],
      },
      ...Array.from({ length: 12 }, (_, index) => makeGpt2Block(index, index === 3)),
    ],
    code: codeLines(gpt2PythonSource),
    jaxCode: codeLines(gpt2JaxPythonSource),
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
        codeLines: [79],
      },
      {
        id: "patch_embed",
        label: "patch_embed",
        type: "Conv2d projection",
        kind: "conv",
        badges: ["16x16", "196 tokens", "768"],
        defaultExpanded: true,
        codeLines: [16, 19, 20, 52, 61],
        children: [
          {
            id: "patch_embed.proj",
            label: "proj",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->768", "k=16", "s=16"],
            codeLines: [16, 19],
          },
          {
            id: "patch_embed.flatten",
            label: "flatten patches",
            type: "Flatten",
            kind: "reshape",
            badges: ["14x14 -> 196"],
            codeLines: [20],
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
        codeLines: [53, 56, 62, 63, 70],
        children: [
          {
            id: "tokens.cls",
            label: "cls_token",
            type: "LearnedToken",
            kind: "embedding",
            badges: ["1 x 768"],
            codeLines: [53, 62, 63],
          },
          {
            id: "tokens.position",
            label: "pos_embed",
            type: "PositionEmbedding",
            kind: "embedding",
            badges: ["197 x 768"],
            codeLines: [56, 70],
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
        codeLines: [57, 73, 74],
        children: Array.from({ length: 12 }, (_, index) => makeVitBlock(index, index === 3)),
      },
      {
        id: "norm",
        label: "encoder_norm",
        type: "LayerNorm",
        kind: "norm",
        badges: ["CLS"],
        codeLines: [58, 75],
      },
      {
        id: "head",
        label: "head",
        type: "Linear",
        kind: "linear",
        badges: ["768->1000"],
        codeLines: [59, 76],
      },
    ],
    code: codeLines(vitPythonSource),
    jaxCode: codeLines(vitJaxPythonSource),
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
        codeLines: [85],
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-ReLU",
        kind: "group",
        summary: "7x7 stride 2",
        defaultExpanded: true,
        codeLines: [51, 52, 56, 59, 60, 86],
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->64", "k=7", "s=2"],
            codeLines: [52, 86],
          },
          {
            id: "stem.bn",
            label: "bn",
            type: "BatchNorm2d",
            kind: "norm",
            badges: ["64"],
            codeLines: [56, 86],
          },
          {
            id: "stem.relu",
            label: "relu",
            type: "ReLU",
            kind: "activation",
            codeLines: [59, 86],
          },
        ],
      },
      {
        id: "maxpool",
        label: "maxpool",
        type: "MaxPool2d",
        kind: "pool",
        badges: ["k=3", "s=2"],
        codeLines: [61, 87],
      },
      {
        id: "layer1",
        label: "layer1",
        type: "ResidualStage",
        kind: "group",
        summary: "2 BasicBlocks",
        badges: ["64 ch", "56x56"],
        codeLines: [62, 88],
        children: [
          {
            id: "layer1.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [77, 88],
          },
          {
            id: "layer1.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [79, 80, 88],
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
        codeLines: [63, 89],
        children: [
          {
            id: "layer2.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            defaultExpanded: true,
            codeLines: [16, 26, 27, 28, 29, 30, 32, 34, 37, 38, 39, 40, 41, 77, 89],
            children: [
              {
                id: "layer2.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["64->128", "k=3", "s=2"],
                codeLines: [16, 37, 77],
              },
              {
                id: "layer2.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["128"],
                codeLines: [26, 37],
              },
              {
                id: "layer2.0.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [27, 37],
              },
              {
                id: "layer2.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["128->128", "k=3"],
                codeLines: [28, 38],
              },
              {
                id: "layer2.0.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["128"],
                codeLines: [29, 38],
              },
              {
                id: "layer2.0.downsample",
                label: "downsample",
                type: "ProjectionSkip",
                kind: "group",
                summary: "1x1 stride 2",
                defaultExpanded: true,
                codeLines: [30, 39, 40, 72, 73, 75, 76],
                children: [
                  {
                    id: "layer2.0.downsample.conv",
                    label: "conv1x1",
                    type: "Conv2d",
                    kind: "conv",
                    badges: ["64->128", "s=2"],
                    codeLines: [73, 40],
                  },
                  {
                    id: "layer2.0.downsample.bn",
                    label: "bn",
                    type: "BatchNorm2d",
                    kind: "norm",
                    badges: ["128"],
                    codeLines: [75, 40],
                  },
                ],
              },
              {
                id: "layer2.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [41],
              },
            ],
          },
          {
            id: "layer2.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [79, 80, 89],
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
        codeLines: [64, 90],
        children: [
          {
            id: "layer3.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            codeLines: [16, 26, 27, 28, 29, 30, 32, 34, 37, 38, 39, 40, 41, 77, 90],
            children: [
              {
                id: "layer3.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["128->256", "k=3", "s=2"],
                codeLines: [16, 37, 77],
              },
              {
                id: "layer3.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [26, 37],
              },
              {
                id: "layer3.0.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [27, 37],
              },
              {
                id: "layer3.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [28, 38],
              },
              {
                id: "layer3.0.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [29, 38],
              },
              {
                id: "layer3.0.downsample",
                label: "downsample",
                type: "ProjectionSkip",
                kind: "group",
                summary: "1x1 stride 2",
                codeLines: [30, 39, 40, 72, 73, 75, 76],
                children: [
                  {
                    id: "layer3.0.downsample.conv",
                    label: "conv1x1",
                    type: "Conv2d",
                    kind: "conv",
                    badges: ["128->256", "s=2"],
                    codeLines: [73, 40],
                  },
                  {
                    id: "layer3.0.downsample.bn",
                    label: "bn",
                    type: "BatchNorm2d",
                    kind: "norm",
                    badges: ["256"],
                    codeLines: [75, 40],
                  },
                ],
              },
              {
                id: "layer3.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [41],
              },
            ],
          },
          {
            id: "layer3.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [16, 26, 27, 28, 29, 32, 34, 37, 38, 41, 79, 80, 90],
            children: [
              {
                id: "layer3.1.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [16, 37, 79],
              },
              {
                id: "layer3.1.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [26, 37],
              },
              {
                id: "layer3.1.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [27, 37],
              },
              {
                id: "layer3.1.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [28, 38],
              },
              {
                id: "layer3.1.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["256"],
                codeLines: [29, 38],
              },
              {
                id: "layer3.1.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [41],
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
        codeLines: [65, 91],
        children: [
          {
            id: "layer4.0",
            label: "block.0",
            type: "BasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            codeLines: [16, 26, 27, 28, 29, 30, 32, 34, 37, 38, 39, 40, 41, 77, 91],
            children: [
              {
                id: "layer4.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->512", "k=3", "s=2"],
                codeLines: [16, 37, 77],
              },
              {
                id: "layer4.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [26, 37],
              },
              {
                id: "layer4.0.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [27, 37],
              },
              {
                id: "layer4.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["512->512", "k=3"],
                codeLines: [28, 38],
              },
              {
                id: "layer4.0.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [29, 38],
              },
              {
                id: "layer4.0.downsample",
                label: "downsample",
                type: "ProjectionSkip",
                kind: "group",
                summary: "1x1 stride 2",
                codeLines: [30, 39, 40, 72, 73, 75, 76],
                children: [
                  {
                    id: "layer4.0.downsample.conv",
                    label: "conv1x1",
                    type: "Conv2d",
                    kind: "conv",
                    badges: ["256->512", "s=2"],
                    codeLines: [73, 40],
                  },
                  {
                    id: "layer4.0.downsample.bn",
                    label: "bn",
                    type: "BatchNorm2d",
                    kind: "norm",
                    badges: ["512"],
                    codeLines: [75, 40],
                  },
                ],
              },
              {
                id: "layer4.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [41],
              },
            ],
          },
          {
            id: "layer4.1",
            label: "block.1",
            type: "BasicBlock",
            kind: "residual",
            summary: "identity skip",
            codeLines: [16, 26, 27, 28, 29, 32, 34, 37, 38, 41, 79, 80, 91],
            children: [
              {
                id: "layer4.1.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["512->512", "k=3"],
                codeLines: [16, 37, 79],
              },
              {
                id: "layer4.1.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [26, 37],
              },
              {
                id: "layer4.1.relu1",
                label: "relu",
                type: "ReLU",
                kind: "activation",
                codeLines: [27, 37],
              },
              {
                id: "layer4.1.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["512->512", "k=3"],
                codeLines: [28, 38],
              },
              {
                id: "layer4.1.bn2",
                label: "bn2",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["512"],
                codeLines: [29, 38],
              },
              {
                id: "layer4.1.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [41],
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
        codeLines: [66, 93],
        children: [
          {
            id: "avgpool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            codeLines: [66, 93],
          },
          {
            id: "flatten",
            label: "flatten",
            type: "Flatten",
            kind: "reshape",
            badges: ["512"],
            codeLines: [93],
          },
        ],
      },
      {
        id: "fc",
        label: "fc",
        type: "Linear",
        kind: "linear",
        badges: ["512->1000"],
        codeLines: [67, 94],
      },
    ],
    code: codeLines(resnet18PythonSource),
    jaxCode: codeLines(resnet18JaxPythonSource),
  },
  {
    id: "widenet",
    label: "WideNet",
    breadcrumb: "WideNet / layer2 / block.0 / conv1",
    stats: "WRN-28-10 · width factor 10 · pre-activation residual blocks",
    fileName: "widenet.py",
    jaxFileName: "widenet_jax.py",
    paper: {
      title: "Wide Residual Networks",
      authors: "Sergey Zagoruyko, Nikos Komodakis",
      year: "2016",
      publishedLabel: "May 23, 2016",
      publishedDate: "2016-05-23",
      venue: "arXiv / BMVC 2016",
      url: "https://arxiv.org/abs/1605.07146",
      pdfUrl: "/papers/widenet.pdf",
      focus: ["widened residual blocks", "feature reuse", "CIFAR image classification"],
    },
    selectedId: "layer2.0.conv1",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "CIFARImage",
        kind: "input",
        badges: ["3 x 32 x 32"],
        codeLines: [144, 145, 162, 164],
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv2d",
        kind: "conv",
        summary: "3x3 shallow stem",
        badges: ["3->16", "32x32"],
        codeLines: [90, 91, 92, 93, 94, 95, 96, 97, 145, 146],
      },
      {
        id: "layer1",
        label: "layer1",
        type: "WideResidualStage",
        kind: "group",
        summary: "4 widened blocks",
        badges: ["160 ch", "32x32"],
        codeLines: [98, 99, 100, 101, 102, 103, 104, 123, 124, 125, 126, 127, 128, 129, 130, 131, 149],
        children: [
          {
            id: "layer1.0",
            label: "block.0",
            type: "WideBasicBlock",
            kind: "residual",
            summary: "16->160 projection",
            codeLines: [19, 20, 21, 37, 38, 39, 47, 48, 49, 50, 51, 54, 56, 63, 64, 126, 149],
          },
          {
            id: "layer1.1-3",
            label: "block.1-3",
            type: "WideBasicBlock x3",
            kind: "residual",
            summary: "identity skips",
            codeLines: [133, 134, 135, 136, 137, 138, 139, 149],
          },
        ],
      },
      {
        id: "layer2",
        label: "layer2",
        type: "WideResidualStage",
        kind: "group",
        summary: "4 widened blocks",
        badges: ["320 ch", "16x16"],
        defaultExpanded: true,
        codeLines: [105, 106, 107, 108, 109, 110, 111, 123, 124, 125, 126, 127, 128, 129, 130, 131, 150],
        children: [
          {
            id: "layer2.0",
            label: "block.0",
            type: "WideBasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            defaultExpanded: true,
            codeLines: [16, 17, 18, 19, 27, 29, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 61, 63, 64, 65, 126, 150],
            children: [
              {
                id: "layer2.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["160"],
                codeLines: [17, 54],
              },
              {
                id: "layer2.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["160->320", "k=3", "s=2"],
                codeLines: [19, 20, 21, 22, 23, 24, 25, 26, 56],
              },
              {
                id: "layer2.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["320->320", "k=3"],
                codeLines: [29, 30, 31, 32, 33, 34, 35, 36, 61],
              },
              {
                id: "layer2.0.shortcut",
                label: "shortcut",
                type: "ProjectionSkip",
                kind: "conv",
                badges: ["160->320", "s=2"],
                codeLines: [37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51],
              },
              {
                id: "layer2.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                codeLines: [63, 64],
              },
            ],
          },
          {
            id: "layer2.1-3",
            label: "block.1-3",
            type: "WideBasicBlock x3",
            kind: "residual",
            summary: "identity skips",
            codeLines: [133, 134, 135, 136, 137, 138, 139, 150],
          },
        ],
      },
      {
        id: "layer3",
        label: "layer3",
        type: "WideResidualStage",
        kind: "group",
        summary: "4 widened blocks",
        badges: ["640 ch", "8x8"],
        codeLines: [112, 113, 114, 115, 116, 117, 118, 123, 124, 125, 126, 127, 128, 129, 130, 131, 151],
        children: [
          {
            id: "layer3.0",
            label: "block.0",
            type: "WideBasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            codeLines: [19, 29, 37, 38, 39, 47, 54, 56, 61, 64, 126, 151],
          },
          {
            id: "layer3.1-3",
            label: "block.1-3",
            type: "WideBasicBlock x3",
            kind: "residual",
            summary: "identity skips",
            codeLines: [133, 134, 135, 136, 137, 138, 139, 151],
          },
        ],
      },
      {
        id: "head",
        label: "head",
        type: "BN-ReLU-Pool-FC",
        kind: "group",
        summary: "global average pool",
        codeLines: [119, 120, 121, 153, 154, 155, 156, 157, 158],
        children: [
          {
            id: "head.bn",
            label: "bn",
            type: "BatchNorm2d",
            kind: "norm",
            badges: ["640"],
            codeLines: [119, 154],
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["8x8"],
            codeLines: [156],
          },
          {
            id: "head.flatten",
            label: "flatten",
            type: "Flatten",
            kind: "reshape",
            badges: ["640"],
            codeLines: [157],
          },
          {
            id: "head.fc",
            label: "fc",
            type: "Linear",
            kind: "linear",
            badges: ["640->10"],
            codeLines: [121, 158],
          },
        ],
      },
    ],
    code: codeLines(widenetPythonSource),
    jaxCode: codeLines(widenetJaxPythonSource),
  },
  {
    id: "densenet",
    label: "DenseNet-121",
    breadcrumb: "DenseNet-121 / denseblock2 / layer.1 / concat",
    stats: "4 dense blocks · 58 dense layers · feature concatenation",
    fileName: "densenet.py",
    jaxFileName: "densenet_jax.py",
    paper: {
      title: "Densely Connected Convolutional Networks",
      authors: "Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger",
      year: "2016",
      publishedLabel: "Aug 25, 2016",
      publishedDate: "2016-08-25",
      venue: "arXiv / CVPR 2017",
      url: "https://arxiv.org/abs/1608.06993",
      pdfUrl: "/papers/densenet.pdf",
      focus: ["dense connectivity", "feature reuse", "vanishing-gradient mitigation"],
    },
    selectedId: "denseblock2.layer1.concat",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [166, 181, 183],
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-ReLU-Pool",
        kind: "group",
        summary: "7x7 stride 2",
        badges: ["3->64", "56x56"],
        codeLines: [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 166, 167],
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->64", "k=7", "s=2"],
            codeLines: [130, 131, 132, 133, 134, 135, 136, 137, 167],
          },
          {
            id: "stem.pool",
            label: "maxpool",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["56x56"],
            codeLines: [140, 167],
          },
        ],
      },
      {
        id: "denseblock1",
        label: "denseblock1",
        type: "DenseBlock",
        kind: "group",
        summary: "6 dense layers",
        badges: ["64->256", "56x56"],
        codeLines: [56, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
        children: [
          {
            id: "denseblock1.layer1",
            label: "layer.1",
            type: "DenseLayer",
            kind: "concat",
            summary: "append growth features",
            badges: ["64+32"],
            codeLines: [16, 17, 18, 20, 27, 29, 40, 41, 43, 46, 50, 51, 52, 70, 84],
          },
          {
            id: "denseblock1.layer2-6",
            label: "layer.2-6",
            type: "DenseLayer x5",
            kind: "concat",
            summary: "repeat concatenation",
            codeLines: [69, 70, 71, 72, 73, 74, 75, 76, 83, 84],
          },
        ],
      },
      {
        id: "transition1",
        label: "transition1",
        type: "Transition",
        kind: "group",
        summary: "compress + downsample",
        badges: ["256->128", "28x28"],
        codeLines: [88, 96, 97, 99, 107, 108, 109, 110, 111, 112, 156, 157, 158, 159, 160],
        children: [
          {
            id: "transition1.conv",
            label: "conv1x1",
            type: "CompressionConv",
            kind: "conv",
            badges: ["256->128"],
            codeLines: [99, 100, 101, 102, 103, 104, 105, 111, 158],
          },
          {
            id: "transition1.pool",
            label: "avgpool",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["stride 2"],
            codeLines: [112],
          },
        ],
      },
      {
        id: "denseblock2",
        label: "denseblock2",
        type: "DenseBlock",
        kind: "group",
        summary: "12 dense layers",
        badges: ["128->512", "28x28"],
        defaultExpanded: true,
        codeLines: [56, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
        children: [
          {
            id: "denseblock2.layer1",
            label: "layer.1",
            type: "DenseLayer",
            kind: "group",
            summary: "BN-ReLU-conv x2",
            defaultExpanded: true,
            codeLines: [16, 17, 18, 20, 27, 29, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 70, 84],
            children: [
              {
                id: "denseblock2.layer1.bottleneck",
                label: "bottleneck",
                type: "1x1 Conv",
                kind: "conv",
                badges: ["128->128"],
                codeLines: [17, 18, 20, 21, 22, 23, 24, 25, 26, 41, 42, 43],
              },
              {
                id: "denseblock2.layer1.growth",
                label: "growth",
                type: "3x3 Conv",
                kind: "conv",
                badges: ["128->32"],
                codeLines: [27, 29, 30, 31, 32, 33, 34, 35, 36, 44, 45, 46],
              },
              {
                id: "denseblock2.layer1.concat",
                label: "concat",
                type: "FeatureConcat",
                kind: "concat",
                badges: ["128+32"],
                codeLines: [50, 51, 52],
              },
            ],
          },
          {
            id: "denseblock2.layer2-12",
            label: "layer.2-12",
            type: "DenseLayer x11",
            kind: "concat",
            summary: "same pattern",
            codeLines: [69, 70, 71, 72, 73, 74, 75, 76, 83, 84],
          },
        ],
      },
      {
        id: "transition2",
        label: "transition2",
        type: "Transition",
        kind: "group",
        summary: "compress + downsample",
        badges: ["512->256", "14x14"],
        codeLines: [88, 96, 97, 99, 107, 108, 109, 110, 111, 112, 156, 157, 158, 159, 160],
      },
      {
        id: "denseblock3",
        label: "denseblock3",
        type: "DenseBlock",
        kind: "group",
        summary: "24 dense layers",
        badges: ["256->1024", "14x14"],
        codeLines: [56, 66, 67, 68, 69, 70, 76, 80, 83, 84, 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
        children: [
          {
            id: "denseblock3.layer1",
            label: "layer.1",
            type: "DenseLayer",
            kind: "concat",
            summary: "append growth features",
            badges: ["256+32"],
            codeLines: [40, 41, 43, 46, 50, 51, 52, 84],
          },
          {
            id: "denseblock3.layer2-24",
            label: "layer.2-24",
            type: "DenseLayer x23",
            kind: "concat",
            summary: "repeat concatenation",
            codeLines: [69, 70, 76, 83, 84],
          },
        ],
      },
      {
        id: "transition3",
        label: "transition3",
        type: "Transition",
        kind: "group",
        summary: "compress + downsample",
        badges: ["1024->512", "7x7"],
        codeLines: [88, 96, 97, 99, 107, 108, 109, 110, 111, 112, 156, 157, 158, 159, 160],
      },
      {
        id: "denseblock4",
        label: "denseblock4",
        type: "DenseBlock",
        kind: "group",
        summary: "16 dense layers",
        badges: ["512->1024", "7x7"],
        codeLines: [56, 66, 67, 68, 69, 70, 76, 80, 83, 84, 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
      },
      {
        id: "head",
        label: "head",
        type: "Norm-Pool-FC",
        kind: "group",
        summary: "global average pool",
        codeLines: [162, 163, 172, 173, 174, 175, 176, 177],
        children: [
          {
            id: "head.norm",
            label: "norm",
            type: "BatchNorm2d",
            kind: "norm",
            badges: ["1024"],
            codeLines: [162, 173],
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            codeLines: [175],
          },
          {
            id: "head.classifier",
            label: "classifier",
            type: "Linear",
            kind: "linear",
            badges: ["1024->1000"],
            codeLines: [163, 177],
          },
        ],
      },
    ],
    code: codeLines(densenetPythonSource),
    jaxCode: codeLines(densenetJaxPythonSource),
  },
  {
    id: "efficientnet",
    label: "EfficientNet-B0",
    breadcrumb: "EfficientNet-B0 / blocks / stage.2 / mbconv.0 / depthwise",
    stats: "MBConv stages · depthwise convs · squeeze-excitation · compound scaling",
    fileName: "efficientnet.py",
    jaxFileName: "efficientnet_jax.py",
    paper: {
      title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
      authors: "Mingxing Tan, Quoc V. Le",
      year: "2019",
      publishedLabel: "May 28, 2019",
      publishedDate: "2019-05-28",
      venue: "arXiv / ICML 2019",
      url: "https://arxiv.org/abs/1905.11946",
      pdfUrl: "/papers/efficientnet.pdf",
      focus: ["compound scaling", "mobile inverted bottlenecks", "squeeze-and-excitation"],
    },
    selectedId: "stage2.mbconv0.depthwise",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [159, 160, 174, 176],
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-SiLU",
        kind: "group",
        summary: "3x3 stride 2",
        badges: ["3->32", "112x112"],
        codeLines: [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 160, 161],
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->32", "k=3", "s=2"],
            codeLines: [117, 118, 119, 120, 121, 122, 123, 124, 161],
          },
          {
            id: "stem.silu",
            label: "silu",
            type: "Swish",
            kind: "activation",
            codeLines: [126, 161],
          },
        ],
      },
      {
        id: "stage1",
        label: "stage1",
        type: "MBConv1",
        kind: "group",
        summary: "1 block",
        badges: ["32->16", "112x112"],
        codeLines: [104, 105, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 163, 164],
        children: [
          {
            id: "stage1.mbconv0",
            label: "mbconv.0",
            type: "MBConv",
            kind: "group",
            summary: "depthwise + SE",
            codeLines: [31, 43, 44, 45, 46, 47, 49, 57, 70, 71, 72, 77, 81, 85, 87, 88, 89, 104, 105, 135, 164],
          },
        ],
      },
      {
        id: "stage2",
        label: "stage2",
        type: "MBConv6",
        kind: "group",
        summary: "2 blocks",
        badges: ["16->24", "56x56"],
        defaultExpanded: true,
        codeLines: [104, 106, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 163, 164],
        children: [
          {
            id: "stage2.mbconv0",
            label: "mbconv.0",
            type: "MBConv",
            kind: "group",
            summary: "stride 2",
            defaultExpanded: true,
            codeLines: [31, 43, 44, 45, 46, 47, 49, 51, 52, 57, 70, 71, 72, 77, 81, 83, 85, 87, 88, 89, 90, 104, 106, 135, 164],
            children: [
              {
                id: "stage2.mbconv0.expand",
                label: "expand",
                type: "1x1 Conv",
                kind: "conv",
                badges: ["16->96"],
                codeLines: [51, 52, 53, 54, 55, 56, 81, 82, 83, 84],
              },
              {
                id: "stage2.mbconv0.depthwise",
                label: "depthwise",
                type: "DepthwiseConv",
                kind: "conv",
                badges: ["k=3", "s=2", "groups=96"],
                codeLines: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 85],
              },
              {
                id: "stage2.mbconv0.se",
                label: "se",
                type: "SqueezeExcite",
                kind: "attention",
                badges: ["channel gate"],
                codeLines: [6, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 70, 71, 87, 88],
              },
              {
                id: "stage2.mbconv0.project",
                label: "project",
                type: "1x1 Conv",
                kind: "conv",
                badges: ["96->24"],
                codeLines: [72, 73, 74, 88, 89],
              },
            ],
          },
          {
            id: "stage2.mbconv1",
            label: "mbconv.1",
            type: "MBConv",
            kind: "residual",
            summary: "same-shape residual",
            codeLines: [47, 77, 78, 91, 92, 93, 133, 134, 135, 164],
          },
        ],
      },
      {
        id: "stage3",
        label: "stage3",
        type: "MBConv6",
        kind: "group",
        summary: "2 blocks",
        badges: ["24->40", "28x28", "k=5"],
        codeLines: [104, 107, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 164],
        children: [
          {
            id: "stage3.mbconv0",
            label: "mbconv.0",
            type: "MBConv",
            kind: "group",
            summary: "stride 2 + SE",
            codeLines: [57, 61, 62, 63, 64, 70, 71, 87, 88, 135, 164],
          },
          {
            id: "stage3.mbconv1",
            label: "mbconv.1",
            type: "MBConv",
            kind: "residual",
            summary: "identity add",
            codeLines: [91, 92, 93, 133, 134, 135, 164],
          },
        ],
      },
      {
        id: "stage4",
        label: "stage4",
        type: "MBConv6",
        kind: "group",
        summary: "3 blocks",
        badges: ["40->80", "14x14"],
        codeLines: [104, 108, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 164],
      },
      {
        id: "stage5",
        label: "stage5",
        type: "MBConv6",
        kind: "group",
        summary: "3 blocks",
        badges: ["80->112", "14x14", "k=5"],
        codeLines: [104, 109, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 164],
      },
      {
        id: "stage6",
        label: "stage6",
        type: "MBConv6",
        kind: "group",
        summary: "4 blocks",
        badges: ["112->192", "7x7", "k=5"],
        codeLines: [104, 110, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 164],
      },
      {
        id: "stage7",
        label: "stage7",
        type: "MBConv6",
        kind: "group",
        summary: "1 block",
        badges: ["192->320", "7x7"],
        codeLines: [104, 111, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 164],
      },
      {
        id: "head",
        label: "head",
        type: "Conv-Pool-FC",
        kind: "group",
        summary: "1280 classifier",
        codeLines: [146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 166, 167, 168, 169, 170],
        children: [
          {
            id: "head.conv",
            label: "conv",
            type: "1x1 Conv",
            kind: "conv",
            badges: ["320->1280"],
            codeLines: [148, 149, 150, 151, 152, 153, 167],
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            codeLines: [168],
          },
          {
            id: "head.classifier",
            label: "classifier",
            type: "Linear",
            kind: "linear",
            badges: ["1280->1000"],
            codeLines: [157, 170],
          },
        ],
      },
    ],
    code: codeLines(efficientnetPythonSource),
    jaxCode: codeLines(efficientnetJaxPythonSource),
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

function FullscreenIcon({ active }: { active: boolean }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      {active ? (
        <path
          d="M6.5 2.5v4h-4M9.5 2.5v4h4M6.5 13.5v-4h-4M9.5 13.5v-4h4"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.6"
        />
      ) : (
        <path
          d="M6.5 2.5h-4v4M9.5 2.5h4v4M6.5 13.5h-4v-4M9.5 13.5h4v-4"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.6"
        />
      )}
    </svg>
  );
}

function ColabIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path
        d="M5.1 5.1a3.4 3.4 0 0 1 5.8 0M5.1 10.9a3.4 3.4 0 0 0 5.8 0M5.5 8a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0ZM13.5 8a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0Z"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
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
        <span className="head-tile-main">
          <strong className="head-label">{node.label}</strong>
          {node.badges?.map((badge) => (
            <span className="badge" key={badge}>
              {badge}
            </span>
          ))}
        </span>
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

function isCommentOnlyCodeLine(line: string) {
  return line.trimStart().startsWith("#");
}

function CodeEditor({ model, selected }: { model: ModelSpec; selected: ArchNode | null }) {
  const [language, setLanguage] = useState<CodeLanguage>("pytorch");
  const editorRef = useRef<HTMLDivElement>(null);
  const codeFiles = {
    pytorch: [{ id: "main", fileName: model.fileName, notebookName: notebookFileName(model.fileName), code: model.code }],
    jax: [{ id: "main", fileName: model.jaxFileName, notebookName: notebookFileName(model.jaxFileName), code: model.jaxCode }],
  } satisfies Record<CodeLanguage, Array<{ id: string; fileName: string; notebookName: string; code: string[] }>>;
  const filesForLanguage = codeFiles[language];
  const currentFile = filesForLanguage[0];
  const selectedLineNumbers = selected?.codeLines ?? [];
  const selectedLines = new Set(
    selectedLineNumbers.filter((lineNumber) => {
      const line = currentFile.code[lineNumber - 1];
      return line !== undefined && !isCommentOnlyCodeLine(line);
    }),
  );
  const firstSelectedLine =
    selectedLineNumbers.find((lineNumber) => selectedLines.has(lineNumber)) ?? null;

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor || firstSelectedLine === null) {
      return;
    }

    const selectedLine = editor.querySelector<HTMLElement>(`[data-line-number="${firstSelectedLine}"]`);
    if (!selectedLine) {
      return;
    }

    const centerOffset = Math.max(24, Math.floor((editor.clientHeight - selectedLine.offsetHeight) / 2));
    const centeredTop = selectedLine.offsetTop - centerOffset;
    editor.scrollTo({
      top: Math.max(0, centeredTop),
      behavior: "smooth",
    });
  }, [firstSelectedLine, language, model.id, selected?.id]);

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
          <a
            className="colab-link"
            href={colabUrl(currentFile.notebookName)}
            target="_blank"
            rel="noreferrer"
            aria-label={`Open ${currentFile.fileName} in Google Colab`}
            title={`Open ${currentFile.fileName} in Google Colab`}
          >
            <ColabIcon />
            <span>Colab</span>
          </a>
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
      <div className="editor" ref={editorRef}>
        {currentFile.code.map((line, index) => {
          const lineNumber = index + 1;
          const highlighted = selectedLines.has(lineNumber);
          return (
            <div
              className={`code-line ${highlighted ? "highlighted" : ""}`}
              data-line-number={lineNumber}
              key={`${lineNumber}-${line}`}
            >
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
  const [isFullscreen, setIsFullscreen] = useState(false);

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
    const updateFullscreenState = () => {
      setIsFullscreen(document.fullscreenElement === viewerRef.current);
    };

    updateFullscreenState();
    document.addEventListener("fullscreenchange", updateFullscreenState);

    return () => document.removeEventListener("fullscreenchange", updateFullscreenState);
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

  const toggleFullscreen = async () => {
    const viewer = viewerRef.current;
    if (!viewer || typeof document === "undefined") {
      return;
    }

    try {
      if (document.fullscreenElement === viewer) {
        await document.exitFullscreen();
        return;
      }

      await viewer.requestFullscreen();
    } catch {
      // Fullscreen can be denied by embedded browsers or document permissions.
    }
  };

  const fullscreenSupported = typeof document !== "undefined" && document.fullscreenEnabled;

  return (
    <div className="paper-viewer" ref={viewerRef}>
      <div className="pdf-canvas-wrap">
        {status !== "ready" ? (
          <div className={`pdf-status ${status === "error" ? "error" : ""}`}>
            {status === "error" ? "PDF could not be rendered" : "Rendering PDF"}
          </div>
        ) : null}
        <canvas ref={canvasRef} className="pdf-canvas" aria-label={`${model.paper.title} page ${pageNumber}`} />
      </div>
      <div className="pdf-controls-dock">
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
          <button
            className="pdf-control-button"
            type="button"
            aria-label={isFullscreen ? "Exit fullscreen paper viewer" : "Enter fullscreen paper viewer"}
            title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            disabled={!fullscreenSupported}
            onClick={toggleFullscreen}
          >
            <FullscreenIcon active={isFullscreen} />
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
      </div>
    </div>
  );
}

type ModelArchVizAppProps = {
  initialModelId?: string;
};

export default function ModelArchVizApp({ initialModelId }: ModelArchVizAppProps) {
  const initialModel = models.find((entry) => entry.id === initialModelId) ?? modelsByPublicationDate[0];
  const router = useRouter();
  const pathname = usePathname();
  const [modelId, setModelId] = useState(initialModel.id);
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
  const expanded = expandedByModel[model.id] ?? new Set<string>();
  const selected = selectedByModel[model.id] ?? null;

  useEffect(() => {
    setModelId(initialModel.id);
    setExpandedByModel({});
    setSelectedByModel({});
    setQuery("");
  }, [initialModel.id]);

  useEffect(() => {
    const controller = new AbortController();
    prefetchPdf(model.paper.pdfUrl, controller.signal);

    return () => controller.abort();
  }, [model.paper.pdfUrl]);

  const updateModel = (nextModelId: string) => {
    setModelId(nextModelId);
    setExpandedByModel({});
    setSelectedByModel({});
    setQuery("");

    const nextPath = `/models/${nextModelId}`;
    if (pathname !== nextPath) {
      router.push(nextPath);
    }
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
