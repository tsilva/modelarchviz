"use client";

import {
  Fragment,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type KeyboardEvent,
  type ReactNode,
  type SetStateAction,
} from "react";
import { usePathname, useRouter } from "next/navigation";
import mlpPythonSource from "./generated/model-code/mlp.py";
import mlpJaxPythonSource from "./generated/model-code/mlp_jax.py";
import rnnPythonSource from "./generated/model-code/elman_rnn.py";
import rnnJaxPythonSource from "./generated/model-code/elman_rnn_jax.py";
import gruPythonSource from "./generated/model-code/gru.py";
import gruJaxPythonSource from "./generated/model-code/gru_jax.py";
import seq2seqPythonSource from "./generated/model-code/seq2seq.py";
import seq2seqJaxPythonSource from "./generated/model-code/seq2seq_jax.py";
import lstmPythonSource from "./generated/model-code/lstm.py";
import lstmJaxPythonSource from "./generated/model-code/lstm_jax.py";
import autoencoderPythonSource from "./generated/model-code/autoencoder.py";
import autoencoderJaxPythonSource from "./generated/model-code/autoencoder_jax.py";
import lenet5PythonSource from "./generated/model-code/lenet5.py";
import lenet5JaxPythonSource from "./generated/model-code/lenet5_jax.py";
import alexnetPythonSource from "./generated/model-code/alexnet.py";
import alexnetJaxPythonSource from "./generated/model-code/alexnet_jax.py";
import vgg16PythonSource from "./generated/model-code/vgg16.py";
import vgg16JaxPythonSource from "./generated/model-code/vgg16_jax.py";
import vaePythonSource from "./generated/model-code/vae.py";
import vaeJaxPythonSource from "./generated/model-code/vae_jax.py";
import ganPythonSource from "./generated/model-code/gan.py";
import ganJaxPythonSource from "./generated/model-code/gan_jax.py";
import googlenetPythonSource from "./generated/model-code/googlenet.py";
import googlenetJaxPythonSource from "./generated/model-code/googlenet_jax.py";
import unetPythonSource from "./generated/model-code/unet.py";
import unetJaxPythonSource from "./generated/model-code/unet_jax.py";
import transformerPythonSource from "./generated/model-code/transformer.py";
import transformerJaxPythonSource from "./generated/model-code/transformer_jax.py";
import vqvaePythonSource from "./generated/model-code/vqvae.py";
import vqvaeJaxPythonSource from "./generated/model-code/vqvae_jax.py";
import bertPythonSource from "./generated/model-code/bert_base.py";
import bertJaxPythonSource from "./generated/model-code/bert_base_jax.py";
import gpt2PythonSource from "./generated/model-code/gpt2_attention.py";
import gpt2JaxPythonSource from "./generated/model-code/gpt2_attention_jax.py";
import ddpmPythonSource from "./generated/model-code/ddpm.py";
import ddpmJaxPythonSource from "./generated/model-code/ddpm_jax.py";
import vitPythonSource from "./generated/model-code/vit_b16.py";
import vitJaxPythonSource from "./generated/model-code/vit_b16_jax.py";
import clipPythonSource from "./generated/model-code/clip.py";
import clipJaxPythonSource from "./generated/model-code/clip_jax.py";
import resnet18PythonSource from "./generated/model-code/resnet18.py";
import resnet18JaxPythonSource from "./generated/model-code/resnet18_jax.py";
import resnet34PythonSource from "./generated/model-code/resnet34.py";
import resnet34JaxPythonSource from "./generated/model-code/resnet34_jax.py";
import resnet50PythonSource from "./generated/model-code/resnet50.py";
import resnet50JaxPythonSource from "./generated/model-code/resnet50_jax.py";
import resnet101PythonSource from "./generated/model-code/resnet101.py";
import resnet101JaxPythonSource from "./generated/model-code/resnet101_jax.py";
import resnet152PythonSource from "./generated/model-code/resnet152.py";
import resnet152JaxPythonSource from "./generated/model-code/resnet152_jax.py";
import widenetPythonSource from "./generated/model-code/widenet.py";
import widenetJaxPythonSource from "./generated/model-code/widenet_jax.py";
import densenetPythonSource from "./generated/model-code/densenet.py";
import densenetJaxPythonSource from "./generated/model-code/densenet_jax.py";
import mobilenetv2PythonSource from "./generated/model-code/mobilenetv2.py";
import mobilenetv2JaxPythonSource from "./generated/model-code/mobilenetv2_jax.py";
import efficientnetPythonSource from "./generated/model-code/efficientnet.py";
import efficientnetJaxPythonSource from "./generated/model-code/efficientnet_jax.py";
import resnetTemplateVariants from "./model-templates/resnet.variants.json";

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
  jaxCodeLines?: number[];
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
  defaultCodeLines?: number[];
  jaxDefaultCodeLines?: number[];
  variants?: ModelVariantSpec[];
  activeVariantId?: string;
};

type ModelVariantSpec = {
  id: string;
  label: string;
  depth: number;
  blockClass: "BasicBlock" | "Bottleneck";
  blockLabel: string;
  stageBlocks: [number, number, number, number];
  expansion: 1 | 4;
  stats: string;
  fileName: string;
  jaxFileName: string;
  selectedId: string;
  nodes: ArchNode[];
  code: string[];
  jaxCode: string[];
};

type PaneKey = "architecture" | "paper" | "code" | "chat";
type CodeLanguage = "pytorch" | "jax";
type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};
type AgentCodeSelection = {
  modelId: string;
  language: CodeLanguage;
  fileName: string;
  lines: number[];
  reason?: string;
};
type UserCodeSelection = {
  modelId: string;
  language: CodeLanguage;
  fileName: string;
  lines: number[];
  text: string;
};
type MarkdownInlineDelimiter = "`" | "**" | "*";
type PaperSelectionHighlightRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};
type PaperSelection = {
  modelId: string;
  pageNumber: number;
  text: string;
  highlightRects: PaperSelectionHighlightRect[];
} | null;
type PdfSearchMatch = {
  pageNumber: number;
  pageMatchIndex: number;
};
type PdfSearchHighlightRect = PaperSelectionHighlightRect & {
  active: boolean;
};

const defaultVisibleColumns: Record<PaneKey, boolean> = {
  architecture: true,
  paper: false,
  code: true,
  chat: false,
};

let activeVisibleColumns = defaultVisibleColumns;

const paneMinWidths: Record<PaneKey, number> = {
  architecture: 320,
  paper: 300,
  code: 420,
  chat: 300,
};

const paneResizeStep = 24;

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

function lineRange(start: number, end: number) {
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}

function notebookFileName(fileName: string) {
  return fileName.replace(/\.py$/, ".ipynb");
}

function colabUrl(notebookName: string) {
  return `https://colab.research.google.com/github/${githubRepository}/blob/${githubBranch}/public/notebooks/${notebookName}`;
}

type ResNetTemplateVariant = {
  id: string;
  label: string;
  depth: number;
  blockClass: "BasicBlock" | "Bottleneck";
  blockLabel: string;
  stageBlocks: [number, number, number, number];
  expansion: 1 | 4;
};

const resnetGeneratedSources: Record<string, { code: string[]; jaxCode: string[] }> = {
  resnet18: {
    code: codeLines(resnet18PythonSource),
    jaxCode: codeLines(resnet18JaxPythonSource),
  },
  resnet34: {
    code: codeLines(resnet34PythonSource),
    jaxCode: codeLines(resnet34JaxPythonSource),
  },
  resnet50: {
    code: codeLines(resnet50PythonSource),
    jaxCode: codeLines(resnet50JaxPythonSource),
  },
  resnet101: {
    code: codeLines(resnet101PythonSource),
    jaxCode: codeLines(resnet101JaxPythonSource),
  },
  resnet152: {
    code: codeLines(resnet152PythonSource),
    jaxCode: codeLines(resnet152JaxPythonSource),
  },
};

function resnetVariantStats(variant: ResNetTemplateVariant) {
  const blockCount = variant.stageBlocks.reduce((total, blocks) => total + blocks, 0);
  const stageSummary = variant.stageBlocks.join("-");

  return `${variant.depth} layers · ${blockCount} ${variant.blockLabel}s · stages ${stageSummary}`;
}

function makeResNetBlockNode(
  variant: ResNetTemplateVariant,
  stageIndex: number,
  blockIndex: number,
  defaultExpanded = false,
): ArchNode {
  const stageChannels = [64, 128, 256, 512] as const;
  const stageSpatial = [56, 28, 14, 7] as const;
  const stageName = `layer${stageIndex + 1}`;
  const stageBaseChannels = stageChannels[stageIndex];
  const previousStageBaseChannels = stageIndex === 0 ? 64 : stageChannels[stageIndex - 1];
  const inputChannels = blockIndex === 0 ? previousStageBaseChannels * variant.expansion : stageBaseChannels * variant.expansion;
  const outputChannels = stageBaseChannels * variant.expansion;
  const stride = blockIndex === 0 && stageIndex > 0 ? 2 : 1;
  const needsProjection = stride !== 1 || inputChannels !== outputChannels;
  const blockCodeLines = variant.blockClass === "BasicBlock" ? [4, 17, 25, 27, 28, 36, 37, 38, 39, 40, 45, 46] : [49, 63, 65, 74, 75, 84, 85, 86, 87, 88, 89, 90, 91, 96, 97];
  const blockNode: ArchNode = {
    id: `${stageName}.${blockIndex}`,
    label: `block.${blockIndex}`,
    type: variant.blockClass,
    kind: "residual",
    summary: needsProjection ? `stride ${stride} + projection` : "identity skip",
    defaultExpanded,
    codeLines: [...blockCodeLines, 134, 137],
  };

  if (variant.blockClass === "BasicBlock") {
    const children: ArchNode[] = [
      {
        id: `${stageName}.${blockIndex}.conv1`,
        label: "conv1",
        type: "Conv2d",
        kind: "conv",
        badges: [`${inputChannels}->${stageBaseChannels}`, "k=3", ...(stride > 1 ? [`s=${stride}`] : [])],
        codeLines: [17, 36, 134, 137],
      },
      {
        id: `${stageName}.${blockIndex}.bn1`,
        label: "bn1",
        type: "BatchNorm2d",
        kind: "norm",
        badges: [`${stageBaseChannels}`],
        codeLines: [25, 37],
      },
      {
        id: `${stageName}.${blockIndex}.relu1`,
        label: "relu",
        type: "ReLU",
        kind: "activation",
        codeLines: [26, 38],
      },
      {
        id: `${stageName}.${blockIndex}.conv2`,
        label: "conv2",
        type: "Conv2d",
        kind: "conv",
        badges: [`${stageBaseChannels}->${outputChannels}`, "k=3"],
        codeLines: [27, 39],
      },
      {
        id: `${stageName}.${blockIndex}.bn2`,
        label: "bn2",
        type: "BatchNorm2d",
        kind: "norm",
        badges: [`${outputChannels}`],
        codeLines: [28, 40],
      },
    ];

    if (needsProjection) {
      children.push({
        id: `${stageName}.${blockIndex}.downsample`,
        label: "downsample",
        type: "ProjectionSkip",
        kind: "group",
        summary: `1x1 stride ${stride}`,
        codeLines: [127, 128, 129, 130, 134, 41, 42],
        children: [
          {
            id: `${stageName}.${blockIndex}.downsample.conv`,
            label: "conv1x1",
            type: "Conv2d",
            kind: "conv",
            badges: [`${inputChannels}->${outputChannels}`, `s=${stride}`],
            codeLines: [129, 42],
          },
          {
            id: `${stageName}.${blockIndex}.downsample.bn`,
            label: "bn",
            type: "BatchNorm2d",
            kind: "norm",
            badges: [`${outputChannels}`],
            codeLines: [130, 42],
          },
        ],
      });
    }

    children.push({
      id: `${stageName}.${blockIndex}.add`,
      label: "add",
      type: "ResidualAdd",
      kind: "residual",
      codeLines: [45],
    });

    return {
      ...blockNode,
      children,
    };
  }

  const bottleneckChildren: ArchNode[] = [
    {
      id: `${stageName}.${blockIndex}.conv1`,
      label: "conv1",
      type: "Conv2d",
      kind: "conv",
      badges: [`${inputChannels}->${stageBaseChannels}`, "k=1"],
      codeLines: [63, 84],
    },
    {
      id: `${stageName}.${blockIndex}.conv2`,
      label: "conv2",
      type: "Conv2d",
      kind: "conv",
      badges: [`${stageBaseChannels}->${stageBaseChannels}`, "k=3", ...(stride > 1 ? [`s=${stride}`] : [])],
      codeLines: [65, 87],
    },
    {
      id: `${stageName}.${blockIndex}.conv3`,
      label: "conv3",
      type: "Conv2d",
      kind: "conv",
      badges: [`${stageBaseChannels}->${outputChannels}`, "k=1"],
      codeLines: [74, 90],
    },
  ];

  if (needsProjection) {
    bottleneckChildren.push({
      id: `${stageName}.${blockIndex}.downsample`,
      label: "downsample",
      type: "ProjectionSkip",
      kind: "group",
      summary: `1x1 stride ${stride}`,
      codeLines: [127, 128, 129, 130, 134, 92, 93],
      children: [
        {
          id: `${stageName}.${blockIndex}.downsample.conv`,
          label: "conv1x1",
          type: "Conv2d",
          kind: "conv",
          badges: [`${inputChannels}->${outputChannels}`, `s=${stride}`],
          codeLines: [129, 93],
        },
        {
          id: `${stageName}.${blockIndex}.downsample.bn`,
          label: "bn",
          type: "BatchNorm2d",
          kind: "norm",
          badges: [`${outputChannels}`],
          codeLines: [130, 93],
        },
      ],
    });
  }

  bottleneckChildren.push({
    id: `${stageName}.${blockIndex}.add`,
    label: "add",
    type: "ResidualAdd",
    kind: "residual",
    codeLines: [96],
  });

  return {
    ...blockNode,
    children: bottleneckChildren,
  };
}

function makeResNetNodes(variant: ResNetTemplateVariant): ArchNode[] {
  const stageChannels = [64, 128, 256, 512] as const;
  const stageSpatial = [56, 28, 14, 7] as const;

  return [
    {
      id: "input",
      label: "input",
      type: "Input",
      kind: "input",
      badges: ["3 x 224 x 224"],
      codeLines: [144],
      jaxCodeLines: [67],
    },
    {
      id: "stem",
      label: "stem",
      type: "Conv-BN-ReLU",
      kind: "group",
      summary: "7x7 stride 2",
      defaultExpanded: true,
      codeLines: [110, 111, 112, 113, 114, 144],
      children: [
        {
          id: "stem.conv",
          label: "conv",
          type: "Conv2d",
          kind: "conv",
          badges: ["3->64", "k=7", "s=2"],
          codeLines: [111, 144],
        },
        {
          id: "stem.bn",
          label: "bn",
          type: "BatchNorm2d",
          kind: "norm",
          badges: ["64"],
          codeLines: [112, 144],
        },
        {
          id: "stem.relu",
          label: "relu",
          type: "ReLU",
          kind: "activation",
          codeLines: [113, 144],
        },
      ],
    },
    {
      id: "maxpool",
      label: "maxpool",
      type: "MaxPool2d",
      kind: "pool",
      badges: ["k=3", "s=2"],
      codeLines: [115, 145],
    },
    ...variant.stageBlocks.map((blockCount, stageIndex) => {
      const stageName = `layer${stageIndex + 1}`;
      const stageChannelsLabel = stageChannels[stageIndex] * variant.expansion;
      const stageCodeLine = 116 + stageIndex;

      return {
        id: stageName,
        label: stageName,
        type: "ResidualStage",
        kind: "group",
        summary: `${blockCount} ${variant.blockLabel}s`,
        badges: [`${stageChannelsLabel} ch`, `${stageSpatial[stageIndex]}x${stageSpatial[stageIndex]}`],
        defaultExpanded: stageIndex === 1,
        codeLines: [stageCodeLine, 123, 134, 136, 137, 148 + stageIndex],
        children: Array.from({ length: blockCount }, (_, blockIndex) =>
          makeResNetBlockNode(variant, stageIndex, blockIndex, stageIndex === 1 && blockIndex === 0),
        ),
      } satisfies ArchNode;
    }),
    {
      id: "pool-flatten",
      label: "pool + flatten",
      type: "ClassifierPrep",
      kind: "group",
      summary: "global avg",
      codeLines: [120, 154, 155],
      children: [
        {
          id: "avgpool",
          label: "avgpool",
          type: "AdaptiveAvgPool2d",
          kind: "pool",
          badges: ["1x1"],
          codeLines: [120, 154],
        },
        {
          id: "flatten",
          label: "flatten",
          type: "Flatten",
          kind: "reshape",
          badges: [`${512 * variant.expansion}`],
          codeLines: [155],
        },
      ],
    },
    {
      id: "fc",
      label: "fc",
      type: "Linear",
      kind: "linear",
      badges: [`${512 * variant.expansion}->1000`],
      codeLines: [121, 156],
    },
  ];
}

const resnetVariantDefinitions = resnetTemplateVariants as unknown as ResNetTemplateVariant[];

const resnetVariants: ModelVariantSpec[] = resnetVariantDefinitions.map((variant) => {
  const sources = resnetGeneratedSources[variant.id];

  return {
    ...variant,
    stats: resnetVariantStats(variant),
    fileName: `${variant.id}.py`,
    jaxFileName: `${variant.id}_jax.py`,
    selectedId: "",
    nodes: makeResNetNodes(variant),
    code: sources.code,
    jaxCode: sources.jaxCode,
  };
});

const completedPdfPrefetches = new Set<string>();
const pendingPdfPrefetches = new Map<string, Promise<void>>();
const pdfPrefetchHints = new Set<string>();
const paperPdfAssetVersion = "20260604";

function paperPdfUrl(modelId: string) {
  return `/papers/${modelId}.pdf?v=${paperPdfAssetVersion}`;
}

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
    codeLines: [
      35, 43, 46, 47, 48, 49, 51, 56, 57, 58, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 76, 79, 80, 82,
      83, 84, 87, 96, 97, 98, 99, 100, 101, 103, 104, 106, 108, 109, 110, 113, 114, 115, 116, 172, 173,
    ],
    lazyChildren: () => [
      {
        id: `encoder.${index}.self_attn`,
        label: "self_attn",
        type: "MultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "d=512"],
        codeLines: [
          46, 47, 48, 49, 56, 57, 58, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 74, 75, 76, 79, 80, 81, 82,
          83, 97, 108,
        ],
      },
      {
        id: `encoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [109, 110],
      },
      {
        id: `encoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        codeLines: [98, 99, 100, 101, 113],
      },
      {
        id: `encoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [114, 115, 116],
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
    codeLines: [21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 69, 70, 71, 72],
    jaxCodeLines: [10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 50, 51, 52, 53],
    lazyChildren: () => [
      {
        id: `step.${index}.update_gate`,
        label: "update gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["z_t"],
        codeLines: [13, 14, 22, 23, 24, 25],
        jaxCodeLines: [11, 12, 13, 14],
      },
      {
        id: `step.${index}.reset_gate`,
        label: "reset gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["r_t"],
        codeLines: [15, 16, 28, 29, 30, 31],
        jaxCodeLines: [17, 18, 19, 20],
      },
      {
        id: `step.${index}.candidate`,
        label: "candidate",
        type: "TanhState",
        kind: "activation",
        badges: ["n_t"],
        codeLines: [17, 18, 34, 35, 36, 37, 38],
        jaxCodeLines: [23, 24, 25, 26, 27],
      },
      {
        id: `step.${index}.mix`,
        label: "state mix",
        type: "GatedInterpolation",
        kind: "recurrent",
        badges: ["h_t"],
        codeLines: [40, 41, 42, 43, 44],
        jaxCodeLines: [29, 30, 31, 32, 33],
      },
    ],
  };
}

function makeSeq2SeqEncoderStep(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `encoder.step.${index}`,
    label: `step.${index}`,
    type: "EncoderLSTMCell",
    kind: "group",
    summary: index === 0 ? "reversed token" : "same cell",
    defaultExpanded,
    codeLines: [91, 92, 93, 94, 95],
    jaxCodeLines: [71, 72, 73, 74, 75],
    lazyChildren: () => [
      {
        id: `encoder.step.${index}.embedding`,
        label: "token embedding",
        type: "EmbeddingLookup",
        kind: "embedding",
        badges: ["128 dim"],
        codeLines: [80, 81, 91],
        jaxCodeLines: [59, 60, 71],
      },
      {
        id: `encoder.step.${index}.lstm_gates`,
        label: "lstm gates",
        type: "Input/Forget/Cell/Output",
        kind: "recurrent",
        badges: ["i", "f", "g", "o"],
        codeLines: [27, 28, 29, 30, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47, 48, 92],
        jaxCodeLines: [14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29, 32, 33, 34, 35, 72],
      },
      {
        id: `encoder.step.${index}.state`,
        label: "state update",
        type: "ContextState",
        kind: "recurrent",
        badges: ["h_t", "c_t"],
        codeLines: [51, 52, 53, 56, 57, 58, 93, 94, 95],
        jaxCodeLines: [38, 39, 40, 43, 44, 45, 73, 74, 75],
      },
    ],
  };
}

function makeSeq2SeqDecoderStep(index: number, defaultExpanded = false): ArchNode {
  return {
    id: `decoder.step.${index}`,
    label: `step.${index}`,
    type: "DecoderLSTMCell",
    kind: "group",
    summary: index === 0 ? "context + BOS" : "teacher forced",
    defaultExpanded,
    codeLines: [127, 128, 129, 130, 131, 132, 133],
    jaxCodeLines: [101, 102, 103, 104, 105, 106, 107],
    lazyChildren: () => [
      {
        id: `decoder.step.${index}.embedding`,
        label: "target embedding",
        type: "EmbeddingLookup",
        kind: "embedding",
        badges: ["shifted target"],
        codeLines: [121, 127],
        jaxCodeLines: [93, 101],
      },
      {
        id: `decoder.step.${index}.lstm_gates`,
        label: "lstm gates",
        type: "Input/Forget/Cell/Output",
        kind: "recurrent",
        badges: ["i", "f", "g", "o"],
        codeLines: [27, 28, 29, 30, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47, 48, 128],
        jaxCodeLines: [14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29, 32, 33, 34, 35, 102],
      },
      {
        id: `decoder.step.${index}.projection`,
        label: "vocab logits",
        type: "Linear",
        kind: "linear",
        badges: ["256->vocab"],
        codeLines: [115, 131, 132],
        jaxCodeLines: [99, 105, 106],
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
    codeLines: [
      35, 43, 46, 47, 48, 49, 51, 56, 57, 58, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 76, 79, 80, 82,
      83, 84, 119, 128, 129, 130, 131, 132, 133, 134, 136, 137, 138, 140, 142, 143, 144, 147, 148, 149,
      152, 153, 154, 155, 172, 173,
    ],
    lazyChildren: () => [
      {
        id: `decoder.${index}.masked_self_attn`,
        label: "masked self_attn",
        type: "CausalMultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "causal"],
        codeLines: [
          46, 47, 48, 49, 56, 57, 58, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 74, 75, 76, 79, 80, 81, 82,
          83, 129, 142, 200, 201, 202,
        ],
      },
      {
        id: `decoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [143, 144],
      },
      {
        id: `decoder.${index}.cross_attn`,
        label: "cross_attn",
        type: "EncoderDecoderAttention",
        kind: "attention",
        badges: ["Q=decoder", "K,V=encoder"],
        codeLines: [
          46, 47, 48, 49, 56, 57, 58, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 76, 79, 80, 82, 83, 130,
          147,
        ],
      },
      {
        id: `decoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [148, 149],
      },
      {
        id: `decoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        codeLines: [131, 132, 133, 134, 152],
      },
      {
        id: `decoder.${index}.norm3`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [153, 154, 155],
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
    codeLines: [
      36, 44, 47, 48, 49, 50, 52, 56, 57, 58, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 76, 79, 80, 82,
      83, 84, 87, 96, 97, 98, 99, 100, 101, 102, 104, 105, 107, 109, 110, 111, 112, 115, 116, 117, 118,
      119, 133, 140, 141,
    ],
    lazyChildren: () => [
      {
        id: `encoder.layer.${index}.self_attn`,
        label: "self_attn",
        type: "BidirectionalSelfAttention",
        kind: "attention",
        badges: ["12 heads", "768"],
        codeLines: [
          47, 48, 49, 50, 56, 57, 58, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 74, 75, 76, 79, 80, 81, 82,
          83, 97, 109,
        ],
      },
      {
        id: `encoder.layer.${index}.attn_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [110, 111, 112],
      },
      {
        id: `encoder.layer.${index}.intermediate`,
        label: "intermediate",
        type: "Dense + GELU",
        kind: "mlp",
        badges: ["768->3072"],
        codeLines: [99, 100, 101, 102, 115],
      },
      {
        id: `encoder.layer.${index}.output`,
        label: "output",
        type: "Dense",
        kind: "mlp",
        badges: ["3072->768"],
        codeLines: [99, 100, 101, 102, 115],
      },
      {
        id: `encoder.layer.${index}.output_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        codeLines: [116, 117, 118, 119],
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
    codeLines: [...lineRange(50, 74), 89, 104, 105],
    jaxCodeLines: [...lineRange(54, 66), 87, 88],
    lazyChildren: () => [
      {
        id: `block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        codeLines: [55, 66],
        jaxCodeLines: [58],
      },
      {
        id: `block.${index}.attn`,
        label: "attn",
        type: "CausalSelfAttention",
        kind: "attention",
        summary: "12 heads",
        codeLines: [...lineRange(5, 48), 56, 67],
        jaxCodeLines: [...lineRange(5, 40), 59],
        lazyChildren: () => [
          {
            id: `block.${index}.attn.c_attn`,
            label: "c_attn",
            type: "QKV Projection",
            kind: "attention",
            badges: ["768->2304"],
            codeLines: [15, 21, 22],
            jaxCodeLines: [14, 15],
          },
          {
            id: `block.${index}.attn.heads`,
            label: "heads",
            type: "Head grid",
            kind: "group",
            summary: "12 x dim 64",
            codeLines: [20, 23, ...lineRange(26, 31), ...lineRange(34, 40), 43],
            jaxCodeLines: [12, 16, ...lineRange(19, 24), ...lineRange(27, 33), 36],
            lazyChildren: () =>
              Array.from({ length: 12 }, (_, headIndex) => ({
                id: `block.${index}.attn.head.${headIndex}`,
                label: `head.${headIndex}`,
                type: "AttentionHead",
                kind: "head" as NodeKind,
                badges: ["q,k,v", "dim 64"],
                codeLines: [20, 23, ...lineRange(26, 31), ...lineRange(34, 40), 43],
                jaxCodeLines: [12, 16, ...lineRange(19, 24), ...lineRange(27, 33), 36],
              })),
          },
          {
            id: `block.${index}.attn.merge`,
            label: "merge",
            type: "Concat heads",
            kind: "attention",
            badges: ["12 x 64 -> 768"],
            codeLines: [...lineRange(43, 47)],
            jaxCodeLines: [...lineRange(36, 39)],
          },
          {
            id: `block.${index}.attn.c_proj`,
            label: "c_proj",
            type: "Output Projection",
            kind: "attention",
            badges: ["768->768"],
            codeLines: [16, 47],
            jaxCodeLines: [39],
          },
        ],
      },
      {
        id: `block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        codeLines: [68],
        jaxCodeLines: [60],
      },
      {
        id: `block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        codeLines: [57, 71],
        jaxCodeLines: [63],
      },
      {
        id: `block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        summary: "3072 hidden",
        codeLines: [...lineRange(58, 62), 72],
        jaxCodeLines: [...lineRange(42, 52), 64],
      },
      {
        id: `block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        codeLines: [73],
        jaxCodeLines: [65],
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
    codeLines: [
      30, 38, 41, 42, 43, 44, 46, 50, 51, 52, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 70, 71, 73,
      74, 75, 78, 87, 88, 89, 90, 91, 92, 93, 94, 97, 99, 100, 101, 104, 105, 106, 107, 124, 137, 138,
    ],
    lazyChildren: () => [
      {
        id: `encoder.block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        codeLines: [88, 99],
      },
      {
        id: `encoder.block.${index}.attn`,
        label: "attn",
        type: "MultiHeadSelfAttention",
        kind: "attention",
        badges: ["12 heads", "197 tokens"],
        codeLines: [
          41, 42, 43, 44, 50, 51, 52, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75,
          89, 100,
        ],
      },
      {
        id: `encoder.block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        codeLines: [101],
      },
      {
        id: `encoder.block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        codeLines: [90, 104],
      },
      {
        id: `encoder.block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        badges: ["768->3072->768"],
        codeLines: [91, 92, 93, 94, 105],
      },
      {
        id: `encoder.block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        codeLines: [106, 107],
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
      pdfUrl: paperPdfUrl("mlp"),
      focus: ["backpropagation", "hidden representations", "multilayer perceptrons"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "FlatVector",
        kind: "input",
        badges: ["784->784"],
        codeLines: [22],
        jaxCodeLines: [15],
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
        codeLines: [30],
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
      pdfUrl: paperPdfUrl("rnn"),
      focus: ["recurrent hidden state", "dynamic memory", "sequence structure"],
    },
    selectedId: "",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        codeLines: [30],
        jaxCodeLines: [26],
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
        codeLines: [38, 39, 40],
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
      pdfUrl: paperPdfUrl("gru"),
      focus: ["update gate", "reset gate", "encoder-decoder sequence modeling"],
    },
    selectedId: "",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        codeLines: [70, 71],
        jaxCodeLines: [51, 52],
      },
      {
        id: "cell_params",
        label: "GRU Cell Params",
        type: "GatedRecurrentCell",
        kind: "group",
        summary: "6 affine projections",
        badges: ["z", "r", "n"],
        defaultExpanded: true,
        codeLines: [5, 6, 13, 14, 15, 16, 17, 18, 20],
        jaxCodeLines: [5, 6, 8, 11, 12, 17, 18, 24, 25],
        children: [
          {
            id: "cell_params.update",
            label: "update params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_z", "h_z"],
            codeLines: [13, 14, 22, 23],
            jaxCodeLines: [11, 12],
          },
          {
            id: "cell_params.reset",
            label: "reset params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_r", "h_r"],
            codeLines: [15, 16, 28, 29],
            jaxCodeLines: [17, 18],
          },
          {
            id: "cell_params.candidate",
            label: "candidate params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_n", "h_n"],
            codeLines: [17, 18, 35, 36],
            jaxCodeLines: [24, 25],
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
        codeLines: [62, 63, 64, 67, 68, 69, 70, 71, 72],
        jaxCodeLines: [42, 43, 44, 47, 48, 49, 50, 51, 52, 53],
        children: [
          {
            id: "h0",
            label: "h0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["64 hidden"],
            codeLines: [62, 63, 64],
            jaxCodeLines: [42, 43, 44],
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
        codeLines: [58, 75],
        jaxCodeLines: [37, 56],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "all states"],
        codeLines: [75, 76, 77, 78],
        jaxCodeLines: [56, 57, 58, 59],
      },
    ],
    code: codeLines(gruPythonSource),
    jaxCode: codeLines(gruJaxPythonSource),
    defaultCodeLines: [80, 81, 82, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99],
    jaxDefaultCodeLines: [61, 62, 63, 68, 69, 72, 73, 74, 75, 76, 77, 78, 81, 82, 86, 87, 88, 90, 91],
  },
  {
    id: "vae",
    label: "VAE",
    breadcrumb: "VAE / reparameterization / latent sample",
    stats: "Gaussian encoder · reparameterization trick · ELBO loss",
    fileName: "vae.py",
    jaxFileName: "vae_jax.py",
    paper: {
      title: "Auto-Encoding Variational Bayes",
      authors: "Diederik P. Kingma, Max Welling",
      year: "2013",
      publishedLabel: "Dec 20, 2013",
      publishedDate: "2013-12-20",
      venue: "arXiv / ICLR 2014",
      url: "https://arxiv.org/abs/1312.6114",
      pdfUrl: paperPdfUrl("vae"),
      focus: ["variational inference", "reparameterization trick", "latent-variable generative models"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input x",
        type: "FlatVector",
        kind: "input",
        badges: ["784"],
        codeLines: [19, 21, 69, 71, 76, 78, 87, 95],
        jaxCodeLines: [11, 13, 57, 59, 64, 66, 75, 84],
      },
      {
        id: "encoder",
        label: "Variational Encoder",
        type: "q_phi(z|x)",
        kind: "group",
        summary: "input -> mu and logvar",
        badges: ["784->256", "two heads"],
        defaultExpanded: true,
        codeLines: [5, 15, 16, 17, 19, 21, 22, 23, 24, 59, 71],
        jaxCodeLines: [5, 11, 13, 14, 15, 16, 48, 59],
        children: [
          {
            id: "encoder.hidden",
            label: "shared trunk",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["784->256"],
            codeLines: [15, 21, 22],
            jaxCodeLines: [13, 14],
          },
          {
            id: "encoder.mu",
            label: "mu",
            type: "Mean head",
            kind: "linear",
            badges: ["32"],
            codeLines: [16, 23, 25, 64, 66, 80],
            jaxCodeLines: [15, 17, 53, 54, 68],
          },
          {
            id: "encoder.logvar",
            label: "logvar",
            type: "Log-variance head",
            kind: "linear",
            badges: ["32"],
            codeLines: [17, 24, 25, 64, 80],
            jaxCodeLines: [16, 17, 53, 68],
          },
        ],
      },
      {
        id: "reparameterize",
        label: "Reparameterize",
        type: "z = mu + sigma eps",
        kind: "reshape",
        summary: "differentiable sampling",
        badges: ["epsilon", "Gaussian"],
        codeLines: [62, 63, 64, 65, 66, 67, 72, 74, 103],
        jaxCodeLines: [51, 52, 53, 54, 55, 60, 62, 78, 102],
      },
      {
        id: "decoder",
        label: "Decoder",
        type: "p_theta(x|z)",
        kind: "group",
        summary: "latent sample -> reconstruction",
        badges: ["32->256->784"],
        defaultExpanded: true,
        codeLines: [27, 37, 38, 39, 40, 41, 44, 46, 60, 73],
        jaxCodeLines: [19, 25, 27, 28, 29, 30, 49, 61],
        children: [
          {
            id: "decoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["32->256"],
            codeLines: [37, 38, 39],
            jaxCodeLines: [27, 28],
          },
          {
            id: "decoder.reconstruction",
            label: "reconstruction",
            type: "Bernoulli probs",
            kind: "activation",
            badges: ["sigmoid"],
            codeLines: [40, 41, 46, 47, 73, 74],
            jaxCodeLines: [29, 30, 31, 61, 62],
          },
        ],
      },
      {
        id: "elbo_loss",
        label: "ELBO Loss",
        type: "Recon + KL",
        kind: "head",
        summary: "negative variational lower bound",
        badges: ["BCE", "KL"],
        defaultExpanded: true,
        codeLines: [76, 77, 78, 79, 80, 81, 82, 83, 95, 96, 97],
        jaxCodeLines: [33, 36, 37, 38, 64, 65, 66, 67, 68, 69, 70, 71, 84, 88, 89],
        children: [
          {
            id: "elbo_loss.reconstruction",
            label: "reconstruction",
            type: "BCE",
            kind: "head",
            codeLines: [79],
            jaxCodeLines: [33, 36, 37, 38, 67],
          },
          {
            id: "elbo_loss.kl",
            label: "KL to prior",
            type: "N(0, I)",
            kind: "head",
            codeLines: [80, 81],
            jaxCodeLines: [68, 69],
          },
        ],
      },
    ],
    code: codeLines(vaePythonSource),
    jaxCode: codeLines(vaeJaxPythonSource),
  },
  {
    id: "gan",
    label: "GAN",
    breadcrumb: "GAN / adversarial game / generator loss",
    stats: "latent generator · real/fake discriminator · minimax training",
    fileName: "gan.py",
    jaxFileName: "gan_jax.py",
    paper: {
      title: "Generative Adversarial Nets",
      authors: "Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio",
      year: "2014",
      publishedLabel: "Jun 10, 2014",
      publishedDate: "2014-06-10",
      venue: "arXiv / NeurIPS 2014",
      url: "https://arxiv.org/abs/1406.2661",
      pdfUrl: paperPdfUrl("gan"),
      focus: ["adversarial training", "generator-discriminator game", "implicit generative models"],
    },
    selectedId: "",
    nodes: [
      {
        id: "latent",
        label: "latent z",
        type: "NoiseVector",
        kind: "input",
        badges: ["100-d"],
        codeLines: [24, 26, 65, 66, 75, 87, 98],
        jaxCodeLines: [11, 13, 54, 55, 64, 71, 83, 94],
      },
      {
        id: "real_images",
        label: "real images",
        type: "Data samples",
        kind: "input",
        badges: ["784 pixels"],
        codeLines: [46, 48, 69, 70, 76, 95, 105],
        jaxCodeLines: [26, 28, 58, 59, 65, 72, 91, 100],
      },
      {
        id: "generator",
        label: "Generator G",
        type: "MLP",
        kind: "group",
        summary: "z -> fake image",
        badges: ["100->784", "tanh"],
        defaultExpanded: true,
        codeLines: [5, 15, 16, 17, 18, 19, 20, 21, 24, 26, 62, 66, 75, 87],
        jaxCodeLines: [5, 11, 13, 14, 15, 16, 17, 18, 51, 55, 64, 71, 83],
        children: [
          {
            id: "generator.hidden",
            label: "hidden MLP",
            type: "Linear stack",
            kind: "linear",
            badges: ["LeakyReLU"],
            codeLines: [15, 16, 17, 18, 19],
            jaxCodeLines: [13, 14, 15, 16],
          },
          {
            id: "generator.output",
            label: "fake image",
            type: "Tanh output",
            kind: "activation",
            badges: ["image_dim"],
            codeLines: [20, 21, 26],
            jaxCodeLines: [17, 18],
          },
        ],
      },
      {
        id: "discriminator",
        label: "Discriminator D",
        type: "MLP classifier",
        kind: "group",
        summary: "image -> real/fake logit",
        badges: ["784->1", "logit"],
        defaultExpanded: true,
        codeLines: [29, 38, 39, 40, 41, 42, 43, 46, 48, 49, 63, 70, 76, 77, 88],
        jaxCodeLines: [21, 26, 28, 29, 30, 31, 32, 33, 52, 59, 65, 66, 72, 73, 84],
        children: [
          {
            id: "discriminator.hidden",
            label: "hidden MLP",
            type: "Linear stack",
            kind: "linear",
            badges: ["LeakyReLU"],
            codeLines: [38, 39, 40, 41, 42],
            jaxCodeLines: [28, 29, 30, 31],
          },
          {
            id: "discriminator.logit",
            label: "real/fake logit",
            type: "Linear",
            kind: "head",
            badges: ["scalar"],
            codeLines: [43, 48, 49],
            jaxCodeLines: [32, 33],
          },
        ],
      },
      {
        id: "discriminator_loss",
        label: "D Loss",
        type: "Real vs fake BCE",
        kind: "head",
        summary: "real->1, fake->0",
        badges: ["detach G"],
        defaultExpanded: true,
        codeLines: [73, 75, 76, 77, 78, 79, 80, 81, 82, 105, 106, 107],
        jaxCodeLines: [69, 71, 72, 73, 74, 75, 76, 77, 78, 100, 103, 104, 120],
        children: [
          {
            id: "discriminator_loss.real",
            label: "real branch",
            type: "BCE target 1",
            kind: "head",
            codeLines: [76, 78, 80],
            jaxCodeLines: [72, 74, 76],
          },
          {
            id: "discriminator_loss.fake",
            label: "fake branch",
            type: "BCE target 0",
            kind: "head",
            codeLines: [75, 77, 79, 81],
            jaxCodeLines: [71, 73, 75, 77],
          },
        ],
      },
      {
        id: "generator_loss",
        label: "G Loss",
        type: "Fool discriminator",
        kind: "head",
        summary: "fake->real",
        badges: ["target 1"],
        codeLines: [85, 87, 88, 89, 90, 110, 111, 112],
        jaxCodeLines: [81, 83, 84, 85, 86, 110, 113, 114, 121],
      },
      {
        id: "alternating_updates",
        label: "Alternating Updates",
        type: "Two optimizers",
        kind: "group",
        summary: "update D, then update G",
        badges: ["minimax game"],
        codeLines: [99, 100, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112],
        jaxCodeLines: [98, 103, 104, 108, 113, 114, 118, 119, 120, 121],
      },
    ],
    code: codeLines(ganPythonSource),
    jaxCode: codeLines(ganJaxPythonSource),
  },
  {
    id: "seq2seq",
    label: "Seq2Seq",
    breadcrumb: "Seq2Seq / decoder / step.0 / vocab logits",
    stats: "7 source steps · 6 target steps · fixed context state",
    fileName: "seq2seq.py",
    jaxFileName: "seq2seq_jax.py",
    paper: {
      title: "Sequence to Sequence Learning with Neural Networks",
      authors: "Ilya Sutskever, Oriol Vinyals, Quoc V. Le",
      year: "2014",
      publishedLabel: "Sep 10, 2014",
      publishedDate: "2014-09-10",
      venue: "arXiv / NeurIPS 2014",
      url: "https://arxiv.org/abs/1409.3215",
      pdfUrl: paperPdfUrl("seq2seq"),
      focus: ["encoder-decoder LSTMs", "fixed-length context", "sequence transduction"],
    },
    selectedId: "",
    nodes: [
      {
        id: "source.input",
        label: "source input",
        type: "TokenIds",
        kind: "input",
        badges: ["7 tokens", "reversed"],
        codeLines: [80, 81, 157],
        jaxCodeLines: [59, 60, 130],
      },
      {
        id: "target.input",
        label: "target input",
        type: "ShiftedTokenIds",
        kind: "input",
        badges: ["6 tokens", "teacher forcing"],
        codeLines: [121, 127, 162],
        jaxCodeLines: [93, 101, 141],
      },
      {
        id: "lstm_cell",
        label: "LSTM Cell",
        type: "SharedGateCell",
        kind: "group",
        summary: "encoder and decoder cells",
        badges: ["i", "f", "g", "o"],
        codeLines: [13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47, 48, 51, 52, 53, 56, 57, 58],
        jaxCodeLines: [14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29, 32, 33, 34, 35, 38, 39, 40, 43, 44, 45],
        children: [
          {
            id: "lstm_cell.input_gate",
            label: "input gate",
            type: "SigmoidGate",
            kind: "recurrent",
            badges: ["i_t"],
            codeLines: [13, 14, 27, 28, 29, 30],
            jaxCodeLines: [14, 15, 16, 17],
          },
          {
            id: "lstm_cell.forget_gate",
            label: "forget gate",
            type: "SigmoidGate",
            kind: "recurrent",
            badges: ["f_t"],
            codeLines: [15, 16, 33, 34, 35, 36],
            jaxCodeLines: [20, 21, 22, 23],
          },
          {
            id: "lstm_cell.candidate",
            label: "candidate",
            type: "TanhMemory",
            kind: "activation",
            badges: ["g_t"],
            codeLines: [17, 18, 39, 40, 41, 42],
            jaxCodeLines: [26, 27, 28, 29],
          },
          {
            id: "lstm_cell.output_gate",
            label: "output gate",
            type: "SigmoidGate",
            kind: "recurrent",
            badges: ["o_t"],
            codeLines: [19, 20, 45, 46, 47, 48],
            jaxCodeLines: [32, 33, 34, 35],
          },
          {
            id: "lstm_cell.state_update",
            label: "state update",
            type: "CellAndHidden",
            kind: "recurrent",
            badges: ["c_t", "h_t"],
            codeLines: [51, 52, 53, 56, 57, 58],
            jaxCodeLines: [38, 39, 40, 43, 44, 45],
          },
        ],
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "RecurrentEncoder",
        kind: "group",
        summary: "compress source",
        badges: ["fixed context"],
        defaultExpanded: true,
        codeLines: [72, 73, 80, 81, 84, 85, 86, 90, 91, 92, 93, 94, 95, 98, 99, 157, 158],
        jaxCodeLines: [60, 63, 64, 65, 69, 70, 71, 72, 73, 74, 75, 78, 79, 130, 131],
        children: [
          {
            id: "encoder.reverse",
            label: "reverse source",
            type: "TokenOrder",
            kind: "reshape",
            badges: ["optimization"],
            codeLines: [79, 80],
            jaxCodeLines: [58, 59],
          },
          {
            id: "encoder.embedding",
            label: "source embedding",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab->128"],
            codeLines: [72, 81],
            jaxCodeLines: [60],
          },
          {
            id: "encoder.initial_state",
            label: "h0/c0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["256 hidden"],
            codeLines: [84, 85, 86],
            jaxCodeLines: [63, 64, 65],
          },
          ...Array.from({ length: 7 }, (_, index) => makeSeq2SeqEncoderStep(index, index === 0)),
        ],
      },
      {
        id: "context",
        label: "context",
        type: "FinalEncoderState",
        kind: "recurrent",
        badges: ["h", "c"],
        codeLines: [98, 99, 157, 158, 162],
        jaxCodeLines: [78, 79, 130, 131, 141],
      },
      {
        id: "decoder",
        label: "Decoder",
        type: "RecurrentDecoder",
        kind: "group",
        summary: "teacher-forced outputs",
        badges: ["autoregressive form"],
        defaultExpanded: true,
        codeLines: [113, 114, 115, 119, 121, 126, 127, 128, 129, 130, 131, 132, 133, 136, 137, 162, 163, 164],
        jaxCodeLines: [93, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 141, 142, 143],
        children: [
          {
            id: "decoder.embedding",
            label: "target embedding",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab->128"],
            codeLines: [113, 121],
            jaxCodeLines: [93],
          },
          ...Array.from({ length: 6 }, (_, index) => makeSeq2SeqDecoderStep(index, index === 0)),
        ],
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + Traces",
        kind: "head",
        badges: ["target vocab", "states"],
        codeLines: [115, 131, 136, 137, 163, 164, 165],
        jaxCodeLines: [99, 105, 110, 111, 142, 143, 144],
      },
    ],
    code: codeLines(seq2seqPythonSource),
    jaxCode: codeLines(seq2seqJaxPythonSource),
  },
  {
    id: "lstm",
    label: "LSTM",
    breadcrumb: "LSTM / recurrent loop / step.0 / forget gate",
    stats: "Sequence classifier · (batch, 8, 32) input · logits + state trace · PyTorch/JAX notebooks",
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
      pdfUrl: paperPdfUrl("lstm"),
      focus: ["cell state memory", "input/forget/output gates", "long-range dependencies"],
    },
    selectedId: "",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        codeLines: [88],
        jaxCodeLines: [67],
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
        codeLines: [96, 97],
      },
    ],
    code: codeLines(lstmPythonSource),
    jaxCode: codeLines(lstmJaxPythonSource),
    defaultCodeLines: [61, 62, 63, 64, 65, 66, 72, 73, 75, 77, 79, 80, 84, 85, 86, 88, 89, 90, 94, 95, 96, 97],
    jaxDefaultCodeLines: [48, 49, 50, 53, 55, 57, 58, 63, 64, 65, 67, 68, 69, 73, 74, 75, 76],
  },
  {
    id: "autoencoder",
    label: "Autoencoder",
    breadcrumb: "Autoencoder / bottleneck / latent code",
    stats: "encoder · 32-d bottleneck · decoder · reconstruction loss",
    fileName: "autoencoder.py",
    jaxFileName: "autoencoder_jax.py",
    paper: {
      title: "Reducing the Dimensionality of Data with Neural Networks",
      authors: "Geoffrey E. Hinton, Ruslan R. Salakhutdinov",
      year: "2006",
      publishedLabel: "Jul 28, 2006",
      publishedDate: "2006-07-28",
      venue: "Science",
      url: "https://www.science.org/doi/10.1126/science.1127647",
      pdfUrl: paperPdfUrl("autoencoder"),
      focus: ["dimensionality reduction", "encoder-decoder reconstruction", "bottleneck representations"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input x",
        type: "FlatVector",
        kind: "input",
        badges: ["784"],
        codeLines: [21, 23, 69, 71, 77, 85],
        jaxCodeLines: [11, 13, 50, 52, 58, 66],
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "Compression MLP",
        kind: "group",
        summary: "input -> latent code",
        badges: ["784->256->32"],
        defaultExpanded: true,
        codeLines: [5, 15, 16, 17, 18, 21, 23, 58, 62, 71],
        jaxCodeLines: [5, 11, 13, 14, 15, 39, 43, 52],
        children: [
          {
            id: "encoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["784->256"],
            codeLines: [15, 16, 17],
            jaxCodeLines: [13, 14],
          },
          {
            id: "encoder.bottleneck",
            label: "bottleneck z",
            type: "LatentCode",
            kind: "embedding",
            badges: ["32"],
            codeLines: [18, 23, 24, 62, 71, 73, 92],
            jaxCodeLines: [15, 16, 43, 52, 54, 81],
          },
        ],
      },
      {
        id: "decoder",
        label: "Decoder",
        type: "Reconstruction MLP",
        kind: "group",
        summary: "latent code -> reconstruction",
        badges: ["32->256->784"],
        defaultExpanded: true,
        codeLines: [26, 36, 37, 38, 39, 40, 43, 45, 59, 66, 72],
        jaxCodeLines: [18, 24, 26, 27, 28, 29, 40, 47, 53],
        children: [
          {
            id: "decoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["32->256"],
            codeLines: [36, 37, 38],
            jaxCodeLines: [26, 27],
          },
          {
            id: "decoder.output",
            label: "reconstruction",
            type: "Sigmoid output",
            kind: "activation",
            badges: ["784"],
            codeLines: [39, 40, 45, 46, 66, 72, 73],
            jaxCodeLines: [28, 29, 30, 47, 53, 54],
          },
        ],
      },
      {
        id: "reconstruction_loss",
        label: "Reconstruction Loss",
        type: "MSE",
        kind: "head",
        summary: "x_hat compared with x",
        badges: ["self-supervised"],
        codeLines: [82, 83, 84, 85, 86, 87, 88],
        jaxCodeLines: [64, 65, 66, 67, 70, 71, 76, 77],
      },
    ],
    code: codeLines(autoencoderPythonSource),
    jaxCode: codeLines(autoencoderJaxPythonSource),
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
      pdfUrl: paperPdfUrl("lenet5"),
      focus: ["convolutional feature maps", "subsampling", "document recognition"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["1 x 32 x 32"],
        codeLines: [20],
        jaxCodeLines: [10],
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
      pdfUrl: paperPdfUrl("alexnet"),
      focus: ["large-scale CNNs", "ReLU activations", "GPU training"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["3 x 227 x 227"],
        codeLines: [46],
        jaxCodeLines: [12],
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
    id: "vgg16",
    label: "VGG-16",
    breadcrumb: "VGG-16 / stage3 / conv3_3",
    stats: "13 conv layers · 3 FC layers · stacked 3x3 filters",
    fileName: "vgg16.py",
    jaxFileName: "vgg16_jax.py",
    paper: {
      title: "Very Deep Convolutional Networks for Large-Scale Image Recognition",
      authors: "Karen Simonyan, Andrew Zisserman",
      year: "2014",
      publishedLabel: "Sep 4, 2014",
      publishedDate: "2014-09-04",
      venue: "arXiv / ICLR 2015",
      url: "https://arxiv.org/abs/1409.1556",
      pdfUrl: paperPdfUrl("vgg16"),
      focus: ["deep plain CNNs", "3x3 convolution stacks", "ImageNet classification"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [60, 71, 81],
        jaxCodeLines: [11, 68, 77],
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "PlainConvStack",
        kind: "group",
        summary: "13 conv + 5 pools",
        defaultExpanded: true,
        codeLines: [11, 12, 13, 15, 17, 18, 20, 22, 23, 25, 27, 29, 30, 32, 34, 36, 37, 39, 41, 43, 60],
        jaxCodeLines: [10, 11, 13, 15, 18, 20, 22, 25, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49],
        children: [
          {
            id: "features.stage1",
            label: "stage1",
            type: "ConvBlock",
            kind: "group",
            summary: "2 convs",
            badges: ["64", "112x112"],
            codeLines: [13, 14, 15, 16, 17],
            jaxCodeLines: [11, 12, 13, 14, 15],
          },
          {
            id: "features.stage2",
            label: "stage2",
            type: "ConvBlock",
            kind: "group",
            summary: "2 convs",
            badges: ["128", "56x56"],
            codeLines: [18, 19, 20, 21, 22],
            jaxCodeLines: [18, 19, 20, 21, 22],
          },
          {
            id: "features.stage3",
            label: "stage3",
            type: "ConvBlock",
            kind: "group",
            summary: "3 convs",
            badges: ["256", "28x28"],
            codeLines: [23, 24, 25, 26, 27, 28, 29],
            jaxCodeLines: [25, 26, 27, 28, 29, 30, 31],
            children: [
              {
                id: "features.stage3.conv1",
                label: "conv3_1",
                type: "Conv2d",
                kind: "conv",
                badges: ["128->256", "k=3"],
                codeLines: [23],
                jaxCodeLines: [25],
              },
              {
                id: "features.stage3.conv2",
                label: "conv3_2",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [25],
                jaxCodeLines: [27],
              },
              {
                id: "features.stage3.conv3",
                label: "conv3_3",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                codeLines: [27],
                jaxCodeLines: [29],
              },
              {
                id: "features.stage3.pool",
                label: "pool3",
                type: "MaxPool2d",
                kind: "pool",
                badges: ["28x28"],
                codeLines: [29],
                jaxCodeLines: [31],
              },
            ],
          },
          {
            id: "features.stage4",
            label: "stage4",
            type: "ConvBlock",
            kind: "group",
            summary: "3 convs",
            badges: ["512", "14x14"],
            codeLines: [30, 31, 32, 33, 34, 35, 36],
            jaxCodeLines: [34, 35, 36, 37, 38, 39, 40],
          },
          {
            id: "features.stage5",
            label: "stage5",
            type: "ConvBlock",
            kind: "group",
            summary: "3 convs",
            badges: ["512", "7x7"],
            codeLines: [37, 38, 39, 40, 41, 42, 43],
            jaxCodeLines: [43, 44, 45, 46, 47, 48, 49],
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["25088"],
        codeLines: [63],
        jaxCodeLines: [52, 53, 54],
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "DenseHead",
        kind: "group",
        summary: "4096 -> 4096 -> classes",
        defaultExpanded: true,
        codeLines: [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 66],
        jaxCodeLines: [57, 58, 59, 60, 61, 62, 63],
        children: [
          {
            id: "classifier.fc6",
            label: "fc6",
            type: "Linear",
            kind: "linear",
            badges: ["25088->4096"],
            codeLines: [49, 66],
            jaxCodeLines: [57],
          },
          {
            id: "classifier.drop6",
            label: "dropout6",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            codeLines: [51],
            jaxCodeLines: [59],
          },
          {
            id: "classifier.fc7",
            label: "fc7",
            type: "Linear",
            kind: "linear",
            badges: ["4096->4096"],
            codeLines: [52],
            jaxCodeLines: [60],
          },
          {
            id: "classifier.drop7",
            label: "dropout7",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            codeLines: [54],
            jaxCodeLines: [62],
          },
          {
            id: "classifier.fc8",
            label: "fc8",
            type: "Linear",
            kind: "linear",
            badges: ["4096->1000"],
            codeLines: [55],
            jaxCodeLines: [63],
          },
        ],
      },
    ],
    code: codeLines(vgg16PythonSource),
    jaxCode: codeLines(vgg16JaxPythonSource),
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
      pdfUrl: paperPdfUrl("googlenet"),
      focus: ["Inception modules", "parallel convolutions", "channel concatenation"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [88],
        jaxCodeLines: [46],
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
        codeLines: [82, 83, 84, 106, 107, 110],
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
            codeLines: [84],
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
      pdfUrl: paperPdfUrl("unet"),
      focus: ["encoder-decoder segmentation", "skip concatenations", "biomedical images"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["1 x 572 x 572"],
        codeLines: [52],
        jaxCodeLines: [25],
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
      pdfUrl: paperPdfUrl("transformer"),
      focus: ["scaled dot-product attention", "encoder-decoder stacks", "positional encoding"],
    },
    selectedId: "",
    nodes: [
      {
        id: "src.input",
        label: "src input",
        type: "TokenIds",
        kind: "input",
        badges: ["source", "16 tokens"],
        codeLines: [173],
        jaxCodeLines: [124],
      },
      {
        id: "tgt.input",
        label: "target input",
        type: "TokenIds",
        kind: "input",
        badges: ["target", "shifted"],
        codeLines: [179],
        jaxCodeLines: [130],
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position",
        defaultExpanded: true,
        codeLines: [169, 170, 171, 178, 179, 184, 185],
        children: [
          {
            id: "src_embed",
            label: "src_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            codeLines: [169, 178],
          },
          {
            id: "tgt_embed",
            label: "tgt_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            codeLines: [170, 184],
          },
          {
            id: "positional_encoding",
            label: "positional",
            type: "SinusoidalEncoding",
            kind: "embedding",
            badges: ["absolute"],
            codeLines: [5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 171, 179, 185],
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
        codeLines: [172, 180, 181],
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
        codeLines: [173, 186],
        children: Array.from({ length: 6 }, (_, index) => makeTransformerDecoderBlock(index, index === 0)),
      },
      {
        id: "generator",
        label: "generator",
        type: "Linear",
        kind: "linear",
        badges: ["512->vocab"],
        codeLines: [174],
      },
    ],
    code: codeLines(transformerPythonSource),
    jaxCode: codeLines(transformerJaxPythonSource),
  },
  {
    id: "vqvae",
    label: "VQ-VAE",
    breadcrumb: "VQ-VAE / quantizer / nearest code",
    stats: "discrete codebook · nearest-neighbor lookup · straight-through estimator",
    fileName: "vqvae.py",
    jaxFileName: "vqvae_jax.py",
    paper: {
      title: "Neural Discrete Representation Learning",
      authors: "Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu",
      year: "2017",
      publishedLabel: "Nov 2, 2017",
      publishedDate: "2017-11-02",
      venue: "arXiv / NeurIPS 2017",
      url: "https://arxiv.org/abs/1711.00937",
      pdfUrl: paperPdfUrl("vqvae"),
      focus: ["vector quantization", "discrete latent codes", "straight-through estimator"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input x",
        type: "FlatVector",
        kind: "input",
        badges: ["784"],
        codeLines: [21, 23, 98, 100, 105, 107, 114, 122],
        jaxCodeLines: [11, 13, 74, 76, 81, 83, 90, 98],
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "z_e(x)",
        kind: "group",
        summary: "continuous latent before VQ",
        badges: ["784->256->32"],
        defaultExpanded: true,
        codeLines: [5, 15, 16, 17, 18, 21, 23, 24, 94, 100],
        jaxCodeLines: [5, 10, 13, 14, 15, 16, 70, 76],
        children: [
          {
            id: "encoder.trunk",
            label: "projection",
            type: "MLP",
            kind: "linear",
            badges: ["784->32"],
            codeLines: [15, 16, 17, 18, 23],
            jaxCodeLines: [13, 14, 15],
          },
        ],
      },
      {
        id: "quantizer",
        label: "Vector Quantizer",
        type: "nearest code",
        kind: "group",
        summary: "z_e -> codebook entry",
        badges: ["K codes", "L2"],
        defaultExpanded: true,
        codeLines: [26, 37, 38, 40, 42, 43, 44, 45, 48, 49, 52, 53, 54, 57, 58, 95, 101],
        jaxCodeLines: [18, 23, 26, 27, 28, 29, 30, 33, 34, 37, 38, 39, 40, 41, 44, 45, 71, 77],
        children: [
          {
            id: "quantizer.codebook",
            label: "codebook",
            type: "Embedding table",
            kind: "embedding",
            badges: ["K x D"],
            codeLines: [37, 38, 43, 44, 48, 49, 52],
            jaxCodeLines: [26, 28, 29, 33, 34, 37, 38],
          },
          {
            id: "quantizer.lookup",
            label: "nearest lookup",
            type: "argmin distance",
            kind: "attention",
            badges: ["argmin"],
            codeLines: [40, 42, 43, 44, 45, 48, 49, 101],
            jaxCodeLines: [27, 28, 29, 30, 33, 34, 77],
          },
          {
            id: "quantizer.straight_through",
            label: "straight-through",
            type: "Estimator",
            kind: "residual",
            badges: ["detach"],
            codeLines: [52, 53, 54, 57, 58],
            jaxCodeLines: [37, 38, 39, 40, 41, 44, 45],
          },
        ],
      },
      {
        id: "decoder",
        label: "Decoder",
        type: "p(x|z_q)",
        kind: "group",
        summary: "selected code -> reconstruction",
        badges: ["32->256->784"],
        defaultExpanded: true,
        codeLines: [60, 70, 71, 72, 73, 74, 77, 79, 80, 96, 102],
        jaxCodeLines: [47, 52, 55, 56, 57, 58, 59, 72, 78],
        children: [
          {
            id: "decoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["32->256"],
            codeLines: [70, 71, 72],
            jaxCodeLines: [55, 56],
          },
          {
            id: "decoder.reconstruction",
            label: "reconstruction",
            type: "Bernoulli probs",
            kind: "activation",
            badges: ["sigmoid"],
            codeLines: [73, 74, 79, 80, 102],
            jaxCodeLines: [57, 58, 59, 78],
          },
        ],
      },
      {
        id: "loss",
        label: "VQ-VAE Loss",
        type: "Recon + VQ",
        kind: "head",
        summary: "reconstruction plus codebook commitment",
        badges: ["BCE", "commit"],
        defaultExpanded: true,
        codeLines: [105, 107, 108, 109, 110, 122, 127, 128, 129, 130, 131],
        jaxCodeLines: [81, 83, 84, 85, 86, 98, 102, 103, 104, 110, 113, 114, 115, 116, 117],
        children: [
          {
            id: "loss.reconstruction",
            label: "reconstruction",
            type: "BCE",
            kind: "head",
            codeLines: [108],
            jaxCodeLines: [84],
          },
          {
            id: "loss.vq",
            label: "codebook + commitment",
            type: "VQ terms",
            kind: "head",
            codeLines: [52, 53, 54, 109, 129, 130, 131],
            jaxCodeLines: [37, 38, 39, 40, 41, 85, 115, 116, 117],
          },
        ],
      },
    ],
    code: codeLines(vqvaePythonSource),
    jaxCode: codeLines(vqvaeJaxPythonSource),
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
      pdfUrl: paperPdfUrl("bert"),
      focus: ["masked language modeling", "bidirectional encoders", "fine-tuning"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input_ids",
        label: "input_ids",
        type: "TokenIds",
        kind: "input",
        badges: ["WordPiece", "16 tokens"],
        codeLines: [139],
        jaxCodeLines: [101],
      },
      {
        id: "token_type_ids",
        label: "token_type_ids",
        type: "SegmentIds",
        kind: "input",
        badges: ["sentence A/B"],
        codeLines: [139],
        jaxCodeLines: [101],
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position + segment",
        defaultExpanded: true,
        codeLines: [15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 31, 32, 132, 139],
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
            codeLines: [16, 23, 25, 26, 27],
          },
          {
            id: "embeddings.segment",
            label: "segment",
            type: "TokenTypeEmbedding",
            kind: "embedding",
            badges: ["2", "768"],
            codeLines: [17, 28],
          },
          {
            id: "embeddings.norm",
            label: "norm",
            type: "LayerNorm",
            kind: "norm",
            codeLines: [18, 31],
          },
          {
            id: "embeddings.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.1"],
            codeLines: [19, 32],
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
        codeLines: [133, 140, 141],
        children: Array.from({ length: 12 }, (_, index) => makeBertLayer(index, index === 3)),
      },
      {
        id: "pooler",
        label: "pooler",
        type: "CLSProjection",
        kind: "linear",
        badges: ["CLS", "768->768"],
        codeLines: [134, 144, 145],
      },
      {
        id: "mlm_head",
        label: "mlm_head",
        type: "MaskedLMHead",
        kind: "head",
        badges: ["768->30522"],
        codeLines: [135],
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
      pdfUrl: paperPdfUrl("gpt2"),
      focus: ["decoder-only transformers", "causal language modeling", "zero-shot transfer"],
    },
    selectedId: "",
    nodes: [
      {
        id: "wte",
        label: "wte",
        type: "TokenEmbedding",
        kind: "embedding",
        badges: ["vocab", "768"],
        codeLines: [86, 97],
        jaxCodeLines: [80],
      },
      {
        id: "wpe",
        label: "wpe",
        type: "PositionEmbedding",
        kind: "embedding",
        badges: ["1024", "768"],
        codeLines: [87, 96, 98, 99],
        jaxCodeLines: [79, 81, 82],
      },
      {
        id: "drop",
        label: "drop",
        type: "Dropout",
        kind: "dropout",
        codeLines: [88, 101],
        jaxCodeLines: [84],
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
      pdfUrl: paperPdfUrl("vit"),
      focus: ["image patches as tokens", "class token", "Transformer encoders for vision"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [130],
        jaxCodeLines: [92],
      },
      {
        id: "patch_embed",
        label: "patch_embed",
        type: "Conv2d projection",
        kind: "conv",
        badges: ["16x16", "196 tokens", "768"],
        defaultExpanded: true,
        codeLines: [16, 17, 18, 20, 22, 25, 26, 121, 130],
        children: [
          {
            id: "patch_embed.proj",
            label: "proj",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->768", "k=16", "s=16"],
            codeLines: [16, 22],
          },
          {
            id: "patch_embed.flatten",
            label: "flatten patches",
            type: "Flatten",
            kind: "reshape",
            badges: ["14x14 -> 196"],
            codeLines: [25, 26],
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
        codeLines: [122, 123, 131, 132, 133, 136],
        children: [
          {
            id: "tokens.cls",
            label: "cls_token",
            type: "LearnedToken",
            kind: "embedding",
            badges: ["1 x 768"],
            codeLines: [122, 131, 132, 133],
          },
          {
            id: "tokens.position",
            label: "pos_embed",
            type: "PositionEmbedding",
            kind: "embedding",
            badges: ["197 x 768"],
            codeLines: [123, 136],
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
        codeLines: [124, 137, 138],
        children: Array.from({ length: 12 }, (_, index) => makeVitBlock(index, index === 3)),
      },
      {
        id: "norm",
        label: "encoder_norm",
        type: "LayerNorm",
        kind: "norm",
        badges: ["CLS"],
        codeLines: [125],
      },
      {
        id: "head",
        label: "head",
        type: "Linear",
        kind: "linear",
        badges: ["768->1000"],
        codeLines: [126],
      },
    ],
    code: codeLines(vitPythonSource),
    jaxCode: codeLines(vitJaxPythonSource),
  },
  {
    id: "clip",
    label: "CLIP",
    breadcrumb: "CLIP / contrastive logits / image-text similarity",
    stats: "dual encoders · shared embedding space · contrastive logits",
    fileName: "clip.py",
    jaxFileName: "clip_jax.py",
    paper: {
      title: "Learning Transferable Visual Models From Natural Language Supervision",
      authors: "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever",
      year: "2021",
      publishedLabel: "Feb 26, 2021",
      publishedDate: "2021-02-26",
      venue: "arXiv / ICML 2021",
      url: "https://arxiv.org/abs/2103.00020",
      pdfUrl: paperPdfUrl("clip"),
      focus: ["natural language supervision", "dual encoders", "contrastive image-text pretraining"],
    },
    selectedId: "",
    nodes: [
      {
        id: "image_input",
        label: "image input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [193, 195],
        jaxCodeLines: [143, 145],
      },
      {
        id: "text_input",
        label: "text input",
        type: "TokenIds",
        kind: "input",
        badges: ["77 tokens"],
        codeLines: [193, 196],
        jaxCodeLines: [143, 146],
      },
      {
        id: "vision_encoder",
        label: "Vision Encoder",
        type: "ViT-B/32 tower",
        kind: "group",
        summary: "patches -> CLS embedding",
        badges: ["224px", "32px patches", "512-d"],
        defaultExpanded: true,
        codeLines: [88, 101, 106, 108, 110, 112, 120, 121, 125, 127, 189, 195],
        jaxCodeLines: [66, 77, 78, 88, 89, 92, 94, 145],
        children: [
          {
            id: "vision_encoder.patch_embed",
            label: "patch_embed",
            type: "Conv2d projection",
            kind: "conv",
            badges: ["32x32", "49 patches"],
            codeLines: [101, 112, 113, 114],
            jaxCodeLines: [77, 78, 80],
          },
          {
            id: "vision_encoder.cls_position",
            label: "CLS + position",
            type: "Learned tokens",
            kind: "embedding",
            badges: ["50 tokens"],
            codeLines: [104, 105, 116, 117, 120],
            jaxCodeLines: [81, 82, 83, 86, 87],
          },
          {
            id: "vision_encoder.blocks",
            label: "visual blocks",
            type: "Transformer stack",
            kind: "attention",
            badges: ["12 blocks", "12 heads"],
            codeLines: [106, 121, 122],
            jaxCodeLines: [88, 89],
          },
          {
            id: "vision_encoder.projection",
            label: "image projection",
            type: "Linear",
            kind: "linear",
            badges: ["768->512"],
            codeLines: [108, 125, 126, 127],
            jaxCodeLines: [92, 93, 94],
          },
        ],
      },
      {
        id: "text_encoder",
        label: "Text Encoder",
        type: "Causal transformer tower",
        kind: "group",
        summary: "tokens -> EOT embedding",
        badges: ["BPE vocab", "77 context", "512-d"],
        defaultExpanded: true,
        codeLines: [130, 143, 145, 147, 148, 149, 151, 154, 156, 159, 160, 164, 167, 168, 190, 196],
        jaxCodeLines: [97, 109, 110, 115, 117, 118, 122, 125, 126, 146],
        children: [
          {
            id: "text_encoder.embedding",
            label: "token + position",
            type: "Embedding",
            kind: "embedding",
            badges: ["49408", "77"],
            codeLines: [143, 144, 154, 155, 156],
            jaxCodeLines: [109, 110, 111, 112],
          },
          {
            id: "text_encoder.causal_mask",
            label: "causal mask",
            type: "Lower triangle",
            kind: "attention",
            badges: ["text-only"],
            codeLines: [148, 149, 159, 160, 161],
            jaxCodeLines: [115, 116, 117, 118, 119],
          },
          {
            id: "text_encoder.eot_pool",
            label: "EOT pool",
            type: "Token gather",
            kind: "reshape",
            badges: ["end token"],
            codeLines: [164, 165, 166, 167],
            jaxCodeLines: [122, 123, 124, 125],
          },
          {
            id: "text_encoder.projection",
            label: "text projection",
            type: "Linear",
            kind: "linear",
            badges: ["512->512"],
            codeLines: [147, 168],
            jaxCodeLines: [126],
          },
        ],
      },
      {
        id: "contrastive_logits",
        label: "Contrastive Logits",
        type: "Similarity matrix",
        kind: "head",
        summary: "all image-text pairs",
        badges: ["batch x batch"],
        defaultExpanded: true,
        codeLines: [188, 189, 190, 191, 195, 196, 199, 200, 201, 202, 203, 204],
        jaxCodeLines: [145, 146, 149, 150, 151, 152, 153, 154, 155],
        children: [
          {
            id: "contrastive_logits.normalize",
            label: "normalize",
            type: "Unit vectors",
            kind: "norm",
            badges: ["cosine"],
            codeLines: [199, 200],
            jaxCodeLines: [149, 150],
          },
          {
            id: "contrastive_logits.temperature",
            label: "logit_scale",
            type: "Learned temperature",
            kind: "linear",
            badges: ["exp"],
            codeLines: [191, 201],
            jaxCodeLines: [151, 152],
          },
          {
            id: "contrastive_logits.matrix",
            label: "similarity matrix",
            type: "Matmul",
            kind: "head",
            badges: ["image @ text.T"],
            codeLines: [202, 203, 204],
            jaxCodeLines: [153, 154, 155],
          },
        ],
      },
    ],
    code: codeLines(clipPythonSource),
    jaxCode: codeLines(clipJaxPythonSource),
  },
  {
    id: "ddpm",
    label: "DDPM",
    breadcrumb: "DDPM / U-Net denoiser / predicted noise",
    stats: "forward noising · timestep-conditioned U-Net · reverse denoising",
    fileName: "ddpm.py",
    jaxFileName: "ddpm_jax.py",
    paper: {
      title: "Denoising Diffusion Probabilistic Models",
      authors: "Jonathan Ho, Ajay Jain, Pieter Abbeel",
      year: "2020",
      publishedLabel: "Jun 19, 2020",
      publishedDate: "2020-06-19",
      venue: "arXiv / NeurIPS 2020",
      url: "https://arxiv.org/abs/2006.11239",
      pdfUrl: paperPdfUrl("ddpm"),
      focus: ["forward diffusion", "noise prediction", "iterative denoising"],
    },
    selectedId: "",
    nodes: [
      {
        id: "clean_input",
        label: "clean input",
        type: "Image x0",
        kind: "input",
        badges: ["3 x 32 x 32"],
        codeLines: [180, 181, 210],
        jaxCodeLines: [124, 125, 157],
      },
      {
        id: "noise_timestep",
        label: "noise + t",
        type: "Noise schedule inputs",
        kind: "input",
        badges: ["epsilon", "t"],
        codeLines: [172, 174, 175, 180, 182, 183, 213, 214],
        jaxCodeLines: [117, 119, 124, 127, 128, 160, 161],
      },
      {
        id: "schedule",
        label: "Diffusion Schedule",
        type: "Beta schedule",
        kind: "group",
        summary: "fixed alphas and variances",
        badges: ["1000 steps", "linear beta"],
        defaultExpanded: true,
        codeLines: [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 175, 176, 177],
        jaxCodeLines: [108, 110, 111, 112, 113, 114, 117, 119, 120, 121],
        children: [
          {
            id: "schedule.betas",
            label: "betas",
            type: "Linear schedule",
            kind: "linear",
            badges: ["1e-4 -> 0.02"],
            codeLines: [161, 166],
            jaxCodeLines: [110],
          },
          {
            id: "schedule.alpha_bar",
            label: "alpha_bar",
            type: "Cumulative product",
            kind: "reshape",
            badges: ["signal power"],
            codeLines: [162, 163, 167, 168],
            jaxCodeLines: [111, 112, 127, 128],
          },
          {
            id: "schedule.posterior_variance",
            label: "posterior variance",
            type: "Reverse variance",
            kind: "linear",
            badges: ["p(x_{t-1}|x_t)"],
            codeLines: [164, 165, 170, 198],
            jaxCodeLines: [113, 114, 145],
          },
        ],
      },
      {
        id: "forward_noising",
        label: "Forward Noising",
        type: "q_sample",
        kind: "group",
        summary: "x0 + scheduled Gaussian noise",
        badges: ["closed form", "x_t"],
        defaultExpanded: true,
        codeLines: [180, 181, 182, 183, 184, 185, 220],
        jaxCodeLines: [124, 125, 126, 127, 128, 129, 130, 162, 168],
        children: [
          {
            id: "forward_noising.signal",
            label: "sqrt alpha_bar",
            type: "Signal scale",
            kind: "linear",
            codeLines: [182, 184],
            jaxCodeLines: [127, 129],
          },
          {
            id: "forward_noising.noise",
            label: "sqrt one-minus",
            type: "Noise scale",
            kind: "linear",
            codeLines: [183, 184],
            jaxCodeLines: [128, 129],
          },
        ],
      },
      {
        id: "time_embedding",
        label: "Time Embedding",
        type: "Sinusoidal MLP",
        kind: "embedding",
        badges: ["sin/cos", "MLP"],
        codeLines: [7, 16, 17, 18, 19, 22, 24, 26, 28, 29, 30, 31, 109, 125],
        jaxCodeLines: [5, 9, 11, 12, 14, 15, 16, 19, 20, 21, 78],
      },
      {
        id: "unet_denoiser",
        label: "U-Net Denoiser",
        type: "epsilon_theta(x_t, t)",
        kind: "group",
        summary: "timestep-conditioned noise predictor",
        badges: ["skip concat", "same image shape"],
        defaultExpanded: true,
        codeLines: [99, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 123, 125, 126, 127, 129, 133, 135, 138, 144, 160, 188],
        jaxCodeLines: [70, 78, 79, 80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 92, 95, 97, 134],
        children: [
          {
            id: "unet_denoiser.encoder",
            label: "encoder",
            type: "Down path",
            kind: "conv",
            badges: ["32 -> 8"],
            codeLines: [110, 111, 112, 113, 114, 126, 127, 128, 129, 130],
            jaxCodeLines: [79, 80, 81, 82, 83],
          },
          {
            id: "unet_denoiser.bottleneck",
            label: "bottleneck",
            type: "ResidualBlock",
            kind: "residual",
            badges: ["base*4"],
            codeLines: [115, 133],
            jaxCodeLines: [86],
          },
          {
            id: "unet_denoiser.decoder",
            label: "decoder + skips",
            type: "Up path",
            kind: "concat",
            badges: ["skip1", "skip2"],
            codeLines: [116, 117, 118, 119, 134, 135, 136, 137, 138, 139],
            jaxCodeLines: [87, 88, 89, 90, 91, 92],
          },
          {
            id: "unet_denoiser.noise_head",
            label: "noise head",
            type: "Conv2d",
            kind: "head",
            badges: ["3 channels"],
            codeLines: [120, 121, 142, 143, 144],
            jaxCodeLines: [95, 96, 97],
          },
        ],
      },
      {
        id: "noise_objective",
        label: "Noise Objective",
        type: "MSE epsilon target",
        kind: "head",
        summary: "predict added noise",
        badges: ["MSE"],
        codeLines: [187, 188, 189, 220, 221, 222],
        jaxCodeLines: [132, 134, 135, 168, 169, 170],
      },
      {
        id: "reverse_step",
        label: "Reverse Step",
        type: "p_sample",
        kind: "group",
        summary: "x_t -> x_{t-1}",
        badges: ["mean", "variance"],
        defaultExpanded: true,
        codeLines: [191, 193, 194, 195, 196, 197, 198, 201, 203, 204, 205],
        jaxCodeLines: [137, 139, 140, 141, 142, 143, 144, 145, 148, 150, 151, 152],
        children: [
          {
            id: "reverse_step.mean",
            label: "reverse mean",
            type: "Gaussian mean",
            kind: "linear",
            codeLines: [193, 194, 195, 196, 197],
            jaxCodeLines: [139, 140, 141, 142, 143, 144],
          },
          {
            id: "reverse_step.sample",
            label: "sample x_{t-1}",
            type: "Gaussian sample",
            kind: "head",
            codeLines: [198, 203, 204, 205],
            jaxCodeLines: [145, 150, 151, 152],
          },
        ],
      },
    ],
    code: codeLines(ddpmPythonSource),
    jaxCode: codeLines(ddpmJaxPythonSource),
  },
  {
    id: "resnet18",
    label: "ResNet",
    breadcrumb: "ResNet / layer2 / block.0 / conv1",
    stats: resnetVariants[0].stats,
    fileName: "resnet18.py",
    jaxFileName: "resnet18_jax.py",
    variants: resnetVariants,
    paper: {
      title: "Deep Residual Learning for Image Recognition",
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
      year: "2015",
      publishedLabel: "Dec 10, 2015",
      publishedDate: "2015-12-10",
      venue: "arXiv / CVPR 2016",
      url: "https://arxiv.org/abs/1512.03385",
      pdfUrl: paperPdfUrl("resnet18"),
      focus: ["identity shortcuts", "residual blocks", "very deep CNNs"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [144],
        jaxCodeLines: [67],
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
      pdfUrl: paperPdfUrl("widenet"),
      focus: ["widened residual blocks", "feature reuse", "CIFAR image classification"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "CIFARImage",
        kind: "input",
        badges: ["3 x 32 x 32"],
        codeLines: [146],
        jaxCodeLines: [
          76,
          77,
          78,
          79,
          80,
          81,
          82,
        ],
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
        codeLines: [
          119,
          120,
          121,
          153,
          154,
          155,
          156,
          157,
        ],
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
            codeLines: [121],
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
      pdfUrl: paperPdfUrl("densenet"),
      focus: ["dense connectivity", "feature reuse", "vanishing-gradient mitigation"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [163],
        jaxCodeLines: [
          73,
          74,
          75,
          76,
          77,
          78,
          79,
          80,
        ],
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
        codeLines: [...lineRange(56, 85), 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
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
        codeLines: [...lineRange(56, 85), 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
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
        codeLines: [...lineRange(56, 85), 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
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
        codeLines: [...lineRange(56, 85), 147, 148, 149, 150, 151, 152, 153, 154, 169, 170],
      },
      {
        id: "head",
        label: "head",
        type: "Norm-Pool-FC",
        kind: "group",
        summary: "global average pool",
        codeLines: [162, 163, 172, 173, 174],
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
            codeLines: [171],
          },
          {
            id: "head.classifier",
            label: "classifier",
            type: "Linear",
            kind: "linear",
            badges: ["1024->1000"],
            codeLines: [163],
          },
        ],
      },
    ],
    code: codeLines(densenetPythonSource),
    jaxCode: codeLines(densenetJaxPythonSource),
  },
  {
    id: "mobilenetv2",
    label: "MobileNetV2",
    breadcrumb: "MobileNetV2 / blocks / inverted residual / linear bottleneck",
    stats: "17 inverted residual blocks · depthwise separable convs · linear bottlenecks",
    fileName: "mobilenetv2.py",
    jaxFileName: "mobilenetv2_jax.py",
    paper: {
      title: "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
      authors: "Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen",
      year: "2018",
      publishedLabel: "Jan 13, 2018",
      publishedDate: "2018-01-13",
      venue: "arXiv / CVPR 2018",
      url: "https://arxiv.org/abs/1801.04381",
      pdfUrl: paperPdfUrl("mobilenetv2"),
      focus: ["inverted residuals", "linear bottlenecks", "mobile-efficient CNNs"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [108, 122, 132],
        jaxCodeLines: [80, 113, 125],
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-ReLU6",
        kind: "group",
        summary: "3x3 stride 2",
        badges: ["3->32", "112x112"],
        codeLines: [75, 76, 77, 78, 79, 80, 108],
        jaxCodeLines: [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->32", "k=3", "s=2"],
            codeLines: [77, 108],
            jaxCodeLines: [73, 74, 75, 76, 77, 78, 79, 80],
          },
          {
            id: "stem.relu6",
            label: "relu6",
            type: "ClippedReLU",
            kind: "activation",
            codeLines: [79, 108],
            jaxCodeLines: [82],
          },
        ],
      },
      {
        id: "inverted_residual",
        label: "Inverted Residual",
        type: "BlockTemplate",
        kind: "group",
        summary: "expand -> depthwise -> linear project",
        defaultExpanded: true,
        codeLines: [15, 16, 17, ...lineRange(19, 46), 50, 53, 54, 88, 89, 90, 91, 92, 93],
        jaxCodeLines: [12, 13, 14, 15, ...lineRange(17, 53), 90, 91, 92, 93, 94, 95],
        children: [
          {
            id: "inverted_residual.expand",
            label: "expand",
            type: "PointwiseConv",
            kind: "conv",
            badges: ["1x1", "t=6"],
            codeLines: [16, 21, 24, 25, 26],
            jaxCodeLines: [14, 19, 20, 21, 22, 23, 24, 25, 26, 27],
          },
          {
            id: "inverted_residual.depthwise",
            label: "depthwise",
            type: "DepthwiseConv",
            kind: "conv",
            badges: ["3x3", "groups=hidden"],
            codeLines: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
            jaxCodeLines: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
          },
          {
            id: "inverted_residual.project",
            label: "linear bottleneck",
            type: "PointwiseProjection",
            kind: "conv",
            badges: ["1x1", "no ReLU"],
            codeLines: [42, 43, 50],
            jaxCodeLines: [43, 44, 45, 46, 47, 48, 49],
          },
          {
            id: "inverted_residual.shortcut",
            label: "shortcut",
            type: "ResidualAdd",
            kind: "residual",
            badges: ["same shape"],
            codeLines: [17, 53, 54],
            jaxCodeLines: [15, 52, 53],
          },
        ],
      },
      {
        id: "stage1",
        label: "stage1",
        type: "InvertedResidual",
        kind: "group",
        summary: "1 block",
        badges: ["32->16", "112x112"],
        codeLines: [66, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 111],
        jaxCodeLines: [63, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
      },
      {
        id: "stage2",
        label: "stage2",
        type: "InvertedResidual x2",
        kind: "group",
        summary: "downsample then residual",
        badges: ["16->24", "56x56"],
        codeLines: [67, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 111],
        jaxCodeLines: [64, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
      },
      {
        id: "stage3",
        label: "stage3",
        type: "InvertedResidual x3",
        kind: "group",
        summary: "32-channel bottlenecks",
        badges: ["24->32", "28x28"],
        codeLines: [68, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 111],
        jaxCodeLines: [65, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
      },
      {
        id: "stage4",
        label: "stage4",
        type: "InvertedResidual x4",
        kind: "group",
        summary: "64-channel bottlenecks",
        badges: ["32->64", "14x14"],
        codeLines: [69, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 111],
        jaxCodeLines: [66, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
      },
      {
        id: "stage5_7",
        label: "stage5-7",
        type: "InvertedResidual x7",
        kind: "group",
        summary: "96/160/320 bottlenecks",
        badges: ["7x7 final"],
        codeLines: [70, 71, 72, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 111],
        jaxCodeLines: [67, 68, 69, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
      },
      {
        id: "head",
        label: "head",
        type: "Conv-Pool-FC",
        kind: "group",
        summary: "1280 expansion",
        defaultExpanded: true,
        codeLines: [98, 99, 100, 101, 102, 103, 104, 114, 115, 116, 117],
        jaxCodeLines: [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        children: [
          {
            id: "head.expand",
            label: "expand",
            type: "PointwiseConv",
            kind: "conv",
            badges: ["320->1280"],
            codeLines: [100, 114],
            jaxCodeLines: [99, 100, 101, 102, 103, 104],
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "GlobalAveragePool",
            kind: "pool",
            badges: ["1x1"],
            codeLines: [115, 116],
            jaxCodeLines: [107],
          },
          {
            id: "head.classifier",
            label: "classifier",
            type: "Linear",
            kind: "linear",
            badges: ["1280->1000"],
            codeLines: [104, 117],
            jaxCodeLines: [108],
          },
        ],
      },
    ],
    code: codeLines(mobilenetv2PythonSource),
    jaxCode: codeLines(mobilenetv2JaxPythonSource),
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
      pdfUrl: paperPdfUrl("efficientnet"),
      focus: ["compound scaling", "mobile inverted bottlenecks", "squeeze-and-excitation"],
    },
    selectedId: "",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        codeLines: [161],
        jaxCodeLines: [
          97,
          98,
          99,
          100,
          101,
          102,
          103,
          104,
        ],
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
        codeLines: [
          146,
          147,
          148,
          149,
          150,
          151,
          152,
          153,
          154,
          155,
          156,
          157,
          166,
          167,
          168,
        ],
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
            codeLines: [157],
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

  if (pane === "chat") {
    return (
      <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
        <path
          d="M3 4.5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v3.8a2 2 0 0 1-2 2H8l-3.2 2.4v-2.4H5a2 2 0 0 1-2-2Z"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.5"
        />
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

function SearchIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path
        d="M7 12a5 5 0 1 1 4-2l2.5 2.5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.6"
      />
    </svg>
  );
}

function ClearIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path d="M4.5 4.5 11.5 11.5M11.5 4.5 4.5 11.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
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

function SendIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="icon">
      <path
        d="M2.5 8h10M8.5 3.5 13 8l-4.5 4.5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
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

function getCodeForLanguage(model: ModelSpec, language: CodeLanguage) {
  if (language === "jax") {
    return {
      fileName: model.jaxFileName,
      code: model.jaxCode,
    };
  }

  return {
    fileName: model.fileName,
    code: model.code,
  };
}

function modelStateKey(model: ModelSpec) {
  return model.activeVariantId ? `${model.id}:${model.activeVariantId}` : model.id;
}

function resolveModelVariant(model: ModelSpec, variantId: string | undefined) {
  const variant = model.variants?.find((entry) => entry.id === variantId) ?? model.variants?.[0];

  if (!variant) {
    return model;
  }

  return {
    ...model,
    label: variant.label,
    stats: variant.stats,
    fileName: variant.fileName,
    jaxFileName: variant.jaxFileName,
    selectedId: variant.selectedId,
    nodes: variant.nodes,
    code: variant.code,
    jaxCode: variant.jaxCode,
    activeVariantId: variant.id,
  };
}

function classNameForNode(node: ArchNode) {
  const match = node.type.match(/^([A-Z][A-Za-z0-9_]*)(?:\s+x\d+)?$/);

  return match?.[1] ?? null;
}

function classLineRange(code: string[], className: string) {
  const startIndex = code.findIndex((line) => line.match(new RegExp(`^class\\s+${className}\\b`)));
  if (startIndex === -1) {
    return [];
  }

  let endExclusiveIndex = code.length;
  for (let index = startIndex + 1; index < code.length; index += 1) {
    const line = code[index];
    if (line.trim().length === 0) {
      continue;
    }

    const isTopLevelLine = !line.startsWith(" ") && !line.startsWith("\t");
    if (isTopLevelLine) {
      endExclusiveIndex = index;
      break;
    }
  }

  let endLineNumber = endExclusiveIndex;
  while (endLineNumber > startIndex + 1 && code[endLineNumber - 1].trim().length === 0) {
    endLineNumber -= 1;
  }

  return lineRange(startIndex + 1, endLineNumber);
}

function selectedLineNumbers(model: ModelSpec, selected: ArchNode | null, language: CodeLanguage) {
  if (!selected) {
    return [];
  }

  const currentFile = getCodeForLanguage(model, language);
  const className = classNameForNode(selected);
  const classLines = className ? classLineRange(currentFile.code, className) : [];

  const nodeLines = language === "jax" && selected.jaxCodeLines ? selected.jaxCodeLines : selected.codeLines;

  return classLines.length > 0 ? classLines : nodeLines;
}

function selectedCodeContext(model: ModelSpec, selected: ArchNode | null, language: CodeLanguage) {
  const currentFile = getCodeForLanguage(model, language);

  return selectedLineNumbers(model, selected, language)
    .filter((lineNumber) => currentFile.code[lineNumber - 1] !== undefined)
    .map((lineNumber) => ({
      lineNumber,
      text: currentFile.code[lineNumber - 1],
    }));
}

function agentSelectedCodeContext(model: ModelSpec, selection: AgentCodeSelection | null) {
  if (!selection || selection.modelId !== model.id) {
    return [];
  }

  const currentFile = getCodeForLanguage(model, selection.language);
  if (selection.fileName !== currentFile.fileName) {
    return [];
  }

  return selection.lines
    .filter((lineNumber) => currentFile.code[lineNumber - 1] !== undefined)
    .map((lineNumber) => ({
      lineNumber,
      text: currentFile.code[lineNumber - 1],
    }));
}

function userSelectedCodeContext(model: ModelSpec, selection: UserCodeSelection | null) {
  if (!selection || selection.modelId !== model.id) {
    return [];
  }

  const currentFile = getCodeForLanguage(model, selection.language);
  if (selection.fileName !== currentFile.fileName) {
    return [];
  }

  return selection.lines
    .filter((lineNumber) => currentFile.code[lineNumber - 1] !== undefined)
    .map((lineNumber) => ({
      lineNumber,
      text: currentFile.code[lineNumber - 1],
    }));
}

function coerceCodeLanguage(value: unknown, fallback: CodeLanguage): CodeLanguage {
  if (value === "jax") {
    return "jax";
  }

  if (value === "pytorch" || value === "python" || value === "torch") {
    return "pytorch";
  }

  return fallback;
}

function normalizeAgentCodeSelection(
  value: unknown,
  model: ModelSpec,
  fallbackLanguage: CodeLanguage,
): AgentCodeSelection | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const candidate = value as Partial<AgentCodeSelection>;
  const language = coerceCodeLanguage(candidate.language, fallbackLanguage);
  const currentFile = getCodeForLanguage(model, language);
  const lines = Array.isArray(candidate.lines)
    ? [...new Set(candidate.lines)]
        .filter((lineNumber) => Number.isInteger(lineNumber) && lineNumber >= 1 && lineNumber <= currentFile.code.length)
        .sort((left, right) => left - right)
    : [];

  if (lines.length === 0) {
    return null;
  }

  return {
    modelId: model.id,
    language,
    fileName: currentFile.fileName,
    lines,
    reason: typeof candidate.reason === "string" ? candidate.reason : undefined,
  };
}

function lineNumberFromCodeLine(element: Element | null) {
  if (!(element instanceof HTMLElement)) {
    return null;
  }

  const lineNumber = Number(element.dataset.lineNumber);
  return Number.isInteger(lineNumber) ? lineNumber : null;
}

function closestCodeLine(node: Node | null) {
  if (!node) {
    return null;
  }

  const element = node instanceof Element ? node : node.parentElement;
  return element?.closest("[data-line-number]") ?? null;
}

function codeSelectionFromBrowserRange(
  model: ModelSpec,
  language: CodeLanguage,
  fileName: string,
  editor: HTMLElement,
) {
  const selection = window.getSelection();
  const selectedText = selection?.toString() ?? "";
  if (!selection || selection.rangeCount === 0 || selectedText.trim().length === 0) {
    return null;
  }

  const anchorInside = selection.anchorNode ? editor.contains(selection.anchorNode) : false;
  const focusInside = selection.focusNode ? editor.contains(selection.focusNode) : false;
  if (!anchorInside || !focusInside) {
    return null;
  }

  const range = selection.getRangeAt(0);
  const lineNumbers = new Set<number>();
  editor.querySelectorAll("[data-line-number]").forEach((lineElement) => {
    try {
      if (range.intersectsNode(lineElement)) {
        const lineNumber = lineNumberFromCodeLine(lineElement);
        if (lineNumber !== null) {
          lineNumbers.add(lineNumber);
        }
      }
    } catch {
      // Ignore transient selection ranges that cannot be compared to a line node.
    }
  });

  if (lineNumbers.size === 0) {
    for (const node of [selection.anchorNode, selection.focusNode]) {
      const lineNumber = lineNumberFromCodeLine(closestCodeLine(node));
      if (lineNumber !== null) {
        lineNumbers.add(lineNumber);
      }
    }
  }

  const lines = [...lineNumbers].sort((left, right) => left - right);
  if (lines.length === 0) {
    return null;
  }

  return {
    modelId: model.id,
    language,
    fileName,
    lines,
    text: selectedText.trim(),
  } satisfies UserCodeSelection;
}

function clearBrowserSelection() {
  window.getSelection()?.removeAllRanges();
}

function previewText(text: string, maxLength = 180) {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength).trimEnd()}...`;
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

function CodeEditor({
  model,
  selected,
  language,
  setLanguage,
  agentCodeSelection,
  onAgentCodeSelectionChange,
  userCodeSelection,
  onUserCodeSelectionChange,
}: {
  model: ModelSpec;
  selected: ArchNode | null;
  language: CodeLanguage;
  setLanguage: (language: CodeLanguage) => void;
  agentCodeSelection: AgentCodeSelection | null;
  onAgentCodeSelectionChange: (selection: AgentCodeSelection | null) => void;
  userCodeSelection: UserCodeSelection | null;
  onUserCodeSelectionChange: (selection: UserCodeSelection | null) => void;
}) {
  const editorRef = useRef<HTMLDivElement>(null);
  const codeFiles = {
    pytorch: [{ id: "main", fileName: model.fileName, notebookName: notebookFileName(model.fileName), code: model.code }],
    jax: [{ id: "main", fileName: model.jaxFileName, notebookName: notebookFileName(model.jaxFileName), code: model.jaxCode }],
  } satisfies Record<CodeLanguage, Array<{ id: string; fileName: string; notebookName: string; code: string[] }>>;
  const filesForLanguage = codeFiles[language];
  const currentFile = filesForLanguage[0];
  const selectedLineNumbersForLanguage = selectedLineNumbers(model, selected, language);
  const defaultLineNumbersForLanguage =
    language === "jax" && model.jaxDefaultCodeLines ? model.jaxDefaultCodeLines : (model.defaultCodeLines ?? []);
  const activeAgentSelection =
    agentCodeSelection &&
    agentCodeSelection.modelId === model.id &&
    agentCodeSelection.language === language &&
    agentCodeSelection.fileName === currentFile.fileName
      ? agentCodeSelection
      : null;
  const activeUserSelection =
    userCodeSelection &&
    userCodeSelection.modelId === model.id &&
    userCodeSelection.language === language &&
    userCodeSelection.fileName === currentFile.fileName
      ? userCodeSelection
      : null;
  const highlightedLineNumbers = activeAgentSelection ? activeAgentSelection.lines : selectedLineNumbersForLanguage;
  const selectedLines = new Set(
    highlightedLineNumbers.filter((lineNumber) => {
      const line = currentFile.code[lineNumber - 1];
      return line !== undefined;
    }),
  );
  const userSelectedLines = new Set(
    activeUserSelection
      ? activeUserSelection.lines.filter((lineNumber) => {
          const line = currentFile.code[lineNumber - 1];
          return line !== undefined;
        })
      : [],
  );
  const scrollLineNumbers = highlightedLineNumbers.length > 0 ? highlightedLineNumbers : defaultLineNumbersForLanguage;
  const firstScrollLine =
    scrollLineNumbers.find((lineNumber) => currentFile.code[lineNumber - 1] !== undefined) ?? null;
  const scrollLineKey = scrollLineNumbers.join(",");

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor || firstScrollLine === null) {
      return;
    }

    const selectedLine = editor.querySelector<HTMLElement>(`[data-line-number="${firstScrollLine}"]`);
    if (!selectedLine) {
      return;
    }

    const centerOffset = Math.max(24, Math.floor((editor.clientHeight - selectedLine.offsetHeight) / 2));
    const centeredTop = selectedLine.offsetTop - centerOffset;
    editor.scrollTo({
      top: Math.max(0, centeredTop),
      behavior: "smooth",
    });
  }, [firstScrollLine, scrollLineKey, language, model.id, selected?.id]);

  const captureUserCodeSelection = () => {
    const editor = editorRef.current;
    if (!editor) {
      return;
    }

    const nextSelection = codeSelectionFromBrowserRange(model, language, currentFile.fileName, editor);
    if (!nextSelection) {
      return;
    }

    onAgentCodeSelectionChange(null);
    onUserCodeSelectionChange(nextSelection);
    clearBrowserSelection();
  };

  const captureClickedCodeLine = (event: React.MouseEvent<HTMLDivElement>) => {
    const lineNumber = lineNumberFromCodeLine(closestCodeLine(event.target as Node));
    if (lineNumber === null) {
      return;
    }

    const lineText = currentFile.code[lineNumber - 1];
    if (lineText === undefined) {
      return;
    }

    onAgentCodeSelectionChange(null);
    onUserCodeSelectionChange({
      modelId: model.id,
      language,
      fileName: currentFile.fileName,
      lines: [lineNumber],
      text: lineText.trim().length > 0 ? lineText.trim() : `line ${lineNumber}`,
    });
    clearBrowserSelection();
  };

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
            onChange={(event) => {
              setLanguage(event.currentTarget.value as CodeLanguage);
              onAgentCodeSelectionChange(null);
              onUserCodeSelectionChange(null);
            }}
          >
            {(Object.keys(languageLabels) as CodeLanguage[]).map((entry) => (
              <option value={entry} key={entry}>
                {languageLabels[entry]}
              </option>
            ))}
          </select>
        </div>
      </div>
      {activeUserSelection ? (
        <div className="code-selection-status user-code-selection-status">
          <span>
            User selected {activeUserSelection.lines.length} line
            {activeUserSelection.lines.length === 1 ? "" : "s"} · {previewText(activeUserSelection.text, 96)}
          </span>
          <button type="button" onClick={() => onUserCodeSelectionChange(null)}>
            Clear
          </button>
        </div>
      ) : null}
      {activeAgentSelection ? (
        <div className="code-selection-status">
          <span>
            Assistant highlighted {activeAgentSelection.lines.length} line
            {activeAgentSelection.lines.length === 1 ? "" : "s"}
            {activeAgentSelection.reason ? ` · ${activeAgentSelection.reason}` : ""}
          </span>
          <button type="button" onClick={() => onAgentCodeSelectionChange(null)}>
            Clear
          </button>
        </div>
      ) : null}
      <div
        className="editor"
        ref={editorRef}
        onKeyUp={captureUserCodeSelection}
        onMouseUp={captureUserCodeSelection}
        onDoubleClick={captureClickedCodeLine}
        onTouchEnd={captureUserCodeSelection}
      >
        {currentFile.code.map((line, index) => {
          const lineNumber = index + 1;
          const highlighted = selectedLines.has(lineNumber);
          const userSelected = userSelectedLines.has(lineNumber);
          return (
            <div
              className={`code-line ${highlighted ? "highlighted" : ""} ${userSelected ? "user-selected" : ""}`}
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

function ChatPanel({
  model,
  selected,
  language,
  query,
  paperSelection,
  agentCodeSelection,
  userCodeSelection,
  onAgentCodeSelectionChange,
  messages,
  setMessages,
}: {
  model: ModelSpec;
  selected: ArchNode | null;
  language: CodeLanguage;
  query: string;
  paperSelection: PaperSelection;
  agentCodeSelection: AgentCodeSelection | null;
  userCodeSelection: UserCodeSelection | null;
  onAgentCodeSelectionChange: (selection: AgentCodeSelection | null) => void;
  messages: ChatMessage[];
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
}) {
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const currentFile = getCodeForLanguage(model, language);
  const selectedLines = selectedCodeContext(model, selected, language);
  const agentSelectedLines = agentSelectedCodeContext(model, agentCodeSelection);
  const userSelectedLines = userSelectedCodeContext(model, userCodeSelection);
  const activeUserCodeSelection =
    userCodeSelection &&
    userCodeSelection.modelId === model.id &&
    userCodeSelection.language === language &&
    userCodeSelection.fileName === currentFile.fileName
      ? userCodeSelection
      : null;

  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) {
      return;
    }

    scroller.scrollTo({
      top: scroller.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, pending]);

  const sendMessage = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = draft.trim();
    if (!content || pending) {
      return;
    }

    const nextMessages: ChatMessage[] = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setDraft("");
    setPending(true);
    setError(null);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: nextMessages,
          context: {
            model: {
              id: model.id,
              label: model.label,
              breadcrumb: model.breadcrumb,
              stats: model.stats,
            },
            paper: {
              title: model.paper.title,
              authors: model.paper.authors,
              year: model.paper.year,
              venue: model.paper.venue,
              focus: model.paper.focus,
            },
            paperSelection:
              paperSelection && paperSelection.modelId === model.id
                ? {
                    pageNumber: paperSelection.pageNumber,
                    text: paperSelection.text,
                  }
                : null,
            selection: selected
              ? {
                  id: selected.id,
                  label: selected.label,
                  type: selected.type,
                  kind: selected.kind,
                  summary: selected.summary ?? null,
                  badges: selected.badges ?? [],
                  codeLines: selected.codeLines,
                }
              : null,
            source: {
              language,
              fileName: currentFile.fileName,
              code: currentFile.code,
              selectedLines,
              userSelectedLines,
              userSelectedText: activeUserCodeSelection?.text ?? "",
              agentSelectedLines,
            },
            searchQuery: query,
          },
        }),
      });
      const payload = (await response.json()) as {
        message?: string;
        error?: string;
        codeSelection?: unknown;
      };

      if (!response.ok || !payload.message) {
        throw new Error(payload.error ?? "Chat request failed");
      }

      if (payload.codeSelection !== null && payload.codeSelection !== undefined) {
        const nextCodeSelection = normalizeAgentCodeSelection(payload.codeSelection, model, language);
        if (nextCodeSelection) {
          onAgentCodeSelectionChange(nextCodeSelection);
        }
      }
      setMessages((current) => [...current, { role: "assistant", content: payload.message ?? "" }]);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Chat request failed";
      setError(message);
      setDraft(content);
      setMessages(messages);
    } finally {
      setPending(false);
    }
  };

  return (
    <section className="chat-pane">
      <div className="pane-toolbar chat-toolbar">
        <div>
          <h1>Chat</h1>
          <p>{selected ? `${selected.label} · ${currentFile.fileName}` : currentFile.fileName}</p>
        </div>
      </div>
      <div className="chat-body" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="chat-empty">
            <strong>Ask about the current selection.</strong>
            <span>{selected ? `${selected.type} · ${selected.id}` : model.label}</span>
            {activeUserCodeSelection ? (
              <span>Code lines {activeUserCodeSelection.lines.join(", ")} · {previewText(activeUserCodeSelection.text)}</span>
            ) : null}
            {paperSelection && paperSelection.modelId === model.id ? (
              <span>Paper page {paperSelection.pageNumber} · {previewText(paperSelection.text)}</span>
            ) : null}
          </div>
        ) : null}
        {messages.map((message, index) => (
          <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
            <span>{message.role === "user" ? "You" : "Assistant"}</span>
            {message.role === "assistant" ? (
              <MarkdownMessage content={message.content} />
            ) : (
              <p className="chat-message-content">{message.content}</p>
            )}
          </div>
        ))}
        {pending ? (
          <div className="chat-message assistant">
            <span>Assistant</span>
            <p className="chat-message-content">Thinking...</p>
          </div>
        ) : null}
      </div>
      <form className="chat-composer" onSubmit={sendMessage}>
        {error ? <div className="chat-error">{error}</div> : null}
        <div className="chat-input-row">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Ask about this selection"
            aria-label="Chat message"
            rows={3}
          />
          <button type="submit" aria-label="Send chat message" title="Send" disabled={pending || draft.trim().length === 0}>
            <SendIcon />
          </button>
        </div>
      </form>
    </section>
  );
}

function MarkdownMessage({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    const fence = trimmed.match(/^```([A-Za-z0-9_-]+)?\s*$/);
    if (fence) {
      const codeLinesBlock: string[] = [];
      index += 1;

      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLinesBlock.push(lines[index]);
        index += 1;
      }

      if (index < lines.length) {
        index += 1;
      }

      blocks.push(
        <pre className="chat-markdown-code" key={`code-${index}`}>
          <code>{codeLinesBlock.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2];
      const HeadingTag = level === 1 ? "h2" : level === 2 ? "h3" : "h4";
      blocks.push(<HeadingTag key={`heading-${index}`}>{renderMarkdownInline(text)}</HeadingTag>);
      index += 1;
      continue;
    }

    const unorderedItem = trimmed.match(/^[-*]\s+(.+)$/);
    if (unorderedItem) {
      const items: string[] = [];

      while (index < lines.length) {
        const match = lines[index].trim().match(/^[-*]\s+(.+)$/);
        if (!match) {
          break;
        }

        items.push(match[1]);
        index += 1;
      }

      blocks.push(
        <ul key={`ul-${index}`}>
          {items.map((item, itemIndex) => (
            <li key={`ul-${index}-${itemIndex}`}>{renderMarkdownInline(item)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    const orderedItem = trimmed.match(/^\d+\.\s+(.+)$/);
    if (orderedItem) {
      const items: string[] = [];

      while (index < lines.length) {
        const match = lines[index].trim().match(/^\d+\.\s+(.+)$/);
        if (!match) {
          break;
        }

        items.push(match[1]);
        index += 1;
      }

      blocks.push(
        <ol key={`ol-${index}`}>
          {items.map((item, itemIndex) => (
            <li key={`ol-${index}-${itemIndex}`}>{renderMarkdownInline(item)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length) {
      const current = lines[index].trim();
      if (
        !current ||
        current.startsWith("```") ||
        current.match(/^(#{1,3})\s+.+$/) ||
        current.match(/^[-*]\s+.+$/) ||
        current.match(/^\d+\.\s+.+$/)
      ) {
        break;
      }

      paragraphLines.push(current);
      index += 1;
    }

    blocks.push(<p key={`p-${index}`}>{renderMarkdownInline(paragraphLines.join(" "))}</p>);
  }

  return <div className="chat-message-content chat-markdown">{blocks}</div>;
}

function renderMarkdownInline(text: string): ReactNode[] {
  return renderMarkdownInlineRange(text, 0);
}

function renderMarkdownInlineRange(text: string, depth: number): ReactNode[] {
  if (depth > 8) {
    return [text];
  }

  const nodes: ReactNode[] = [];
  let index = 0;

  while (index < text.length) {
    const link = nextMarkdownLink(text, index);
    const delimiter = nextMarkdownDelimiter(text, index);

    if (link && (!delimiter || link.start <= delimiter.start)) {
      if (link.start > index) {
        nodes.push(text.slice(index, link.start));
      }

      const href = safeMarkdownHref(link.href);
      nodes.push(
        <a href={href} key={`link-${link.start}`} rel="noreferrer" target="_blank">
          {renderMarkdownInlineRange(link.text, depth + 1)}
        </a>,
      );
      index = link.end;
      continue;
    }

    if (!delimiter) {
      nodes.push(text.slice(index));
      break;
    }

    if (delimiter.start > index) {
      nodes.push(text.slice(index, delimiter.start));
    }

    const inner = text.slice(delimiter.start + delimiter.marker.length, delimiter.end);
    if (delimiter.marker === "`") {
      nodes.push(<code key={`code-${delimiter.start}`}>{inner}</code>);
    } else if (delimiter.marker === "**") {
      nodes.push(<strong key={`strong-${delimiter.start}`}>{renderMarkdownInlineRange(inner, depth + 1)}</strong>);
    } else {
      nodes.push(<em key={`em-${delimiter.start}`}>{renderMarkdownInlineRange(inner, depth + 1)}</em>);
    }

    index = delimiter.end + delimiter.marker.length;
  }

  return nodes;
}

function nextMarkdownLink(text: string, fromIndex: number) {
  const linkPattern = /\[([^\]]+)\]\(([^)\s]+)\)/g;
  linkPattern.lastIndex = fromIndex;
  const match = linkPattern.exec(text);

  if (!match) {
    return null;
  }

  return {
    start: match.index,
    end: match.index + match[0].length,
    text: match[1],
    href: match[2],
  };
}

function nextMarkdownDelimiter(text: string, fromIndex: number) {
  const candidates: { marker: MarkdownInlineDelimiter; start: number; end: number }[] = [];

  for (const marker of ["`", "**", "*"] as MarkdownInlineDelimiter[]) {
    const start = text.indexOf(marker, fromIndex);
    if (start === -1) {
      continue;
    }

    const end = text.indexOf(marker, start + marker.length);
    if (end === -1 || end === start + marker.length) {
      continue;
    }

    candidates.push({ marker, start, end });
  }

  return candidates.sort((left, right) => left.start - right.start || right.marker.length - left.marker.length)[0];
}

function safeMarkdownHref(href: string) {
  if (href.startsWith("http://") || href.startsWith("https://") || href.startsWith("mailto:")) {
    return href;
  }

  return "#";
}

function PaperPane({
  model,
  paperSelection,
  onPaperSelectionChange,
}: {
  model: ModelSpec;
  paperSelection: PaperSelection;
  onPaperSelectionChange: (selection: PaperSelection) => void;
}) {
  return (
    <section className="paper-pane">
      <PdfViewer model={model} paperSelection={paperSelection} onPaperSelectionChange={onPaperSelectionChange} />
    </section>
  );
}

function PdfViewer({
  model,
  paperSelection,
  onPaperSelectionChange,
}: {
  model: ModelSpec;
  paperSelection: PaperSelection;
  onPaperSelectionChange: (selection: PaperSelection) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);
  const [pdfDocument, setPdfDocument] = useState<any>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageCount, setPageCount] = useState(0);
  const [viewerSize, setViewerSize] = useState({ width: 0, height: 0 });
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMatches, setSearchMatches] = useState<PdfSearchMatch[]>([]);
  const [activeSearchIndex, setActiveSearchIndex] = useState(-1);
  const [isSearchPending, setIsSearchPending] = useState(false);
  const [searchHighlightRects, setSearchHighlightRects] = useState<PdfSearchHighlightRect[]>([]);

  useEffect(() => {
    setPageNumber(1);
    setPageCount(0);
    setPdfDocument(null);
    setSearchQuery("");
    setSearchMatches([]);
    setActiveSearchIndex(-1);
    setSearchHighlightRects([]);
    setStatus("loading");
  }, [model.paper.pdfUrl]);

  useEffect(() => {
    let cancelled = false;
    let loadingTask: any;

    const loadDocument = async () => {
      setStatus("loading");

      try {
        const pdfjs = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        loadingTask = pdfjs.getDocument(model.paper.pdfUrl);
        const pdf = await loadingTask.promise;
        if (cancelled) {
          void pdf.destroy();
          return;
        }

        setPdfDocument(pdf);
        setPageCount(pdf.numPages);
      } catch (error) {
        if (!cancelled) {
          console.error("PDF load failed", error);
          setStatus("error");
        }
      }
    };

    loadDocument();

    return () => {
      cancelled = true;
      void loadingTask?.destroy();
    };
  }, [model.paper.pdfUrl]);

  useEffect(() => {
    if (!viewerRef.current || typeof ResizeObserver === "undefined") {
      return;
    }

    const updateSize = () => {
      const nextWidth = viewerRef.current?.clientWidth ?? 0;
      const nextHeight = viewerRef.current?.clientHeight ?? 0;
      setViewerSize((current) => {
        if (current.width === nextWidth && current.height === nextHeight) {
          return current;
        }

        return { width: nextWidth, height: nextHeight };
      });
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
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
    let renderTask: any;
    let textLayerTask: any;

    const renderPage = async () => {
      const canvas = canvasRef.current;
      const viewer = viewerRef.current;
      const textLayer = textLayerRef.current;
      if (!canvas || !viewer || !textLayer || !pdfDocument) {
        return;
      }

      setStatus("loading");
      textLayer.replaceChildren();

      try {
        const pdfjs = await import("pdfjs-dist");
        const page = await pdfDocument.getPage(Math.min(pageNumber, pdfDocument.numPages));
        const baseViewport = page.getViewport({ scale: 1 });
        const availableWidth = Math.max(viewerSize.width || viewer.clientWidth, 240);
        const availableHeight = Math.max(viewerSize.height || viewer.clientHeight, 240);
        const fitWidthScale = availableWidth / baseViewport.width;
        const fitHeightScale = availableHeight / baseViewport.height;
        const scale = Math.min(Math.max(Math.min(fitWidthScale, fitHeightScale), 0.45), 2.4);
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

        textLayer.style.setProperty("--scale-factor", String(viewport.scale));
        textLayerTask = new pdfjs.TextLayer({
          textContentSource: page.streamTextContent({
            includeMarkedContent: true,
            disableNormalization: true,
          }),
          container: textLayer,
          viewport,
        });
        await textLayerTask.render();

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
      textLayerTask?.cancel();
    };
  }, [pageNumber, pdfDocument, viewerSize.height, viewerSize.width]);

  useEffect(() => {
    const trimmedQuery = searchQuery.trim();
    if (!pdfDocument || trimmedQuery.length === 0) {
      setIsSearchPending(false);
      setSearchMatches([]);
      setActiveSearchIndex(-1);
      return;
    }

    let cancelled = false;
    const searchDelay = window.setTimeout(async () => {
      setIsSearchPending(true);
      const nextMatches: PdfSearchMatch[] = [];
      const normalizedQuery = trimmedQuery.toLocaleLowerCase();

      try {
        for (let currentPageNumber = 1; currentPageNumber <= pdfDocument.numPages; currentPageNumber += 1) {
          if (cancelled) {
            return;
          }

          const page = await pdfDocument.getPage(currentPageNumber);
          const textContent = await page.getTextContent();
          let pageMatchIndex = 0;

          for (const item of textContent.items) {
            const itemText = "str" in item ? String(item.str) : "";
            const normalizedItemText = itemText.toLocaleLowerCase();
            let matchIndex = normalizedItemText.indexOf(normalizedQuery);

            while (matchIndex !== -1) {
              nextMatches.push({ pageNumber: currentPageNumber, pageMatchIndex });
              pageMatchIndex += 1;
              matchIndex = normalizedItemText.indexOf(normalizedQuery, matchIndex + normalizedQuery.length);
            }
          }
        }

        if (!cancelled) {
          setSearchMatches(nextMatches);
          setActiveSearchIndex(nextMatches.length > 0 ? 0 : -1);
        }
      } catch (error) {
        if (!cancelled) {
          console.error("PDF search failed", error);
          setSearchMatches([]);
          setActiveSearchIndex(-1);
        }
      } finally {
        if (!cancelled) {
          setIsSearchPending(false);
        }
      }
    }, 160);

    return () => {
      cancelled = true;
      window.clearTimeout(searchDelay);
    };
  }, [pdfDocument, searchQuery]);

  const activeSearchMatch = activeSearchIndex >= 0 ? searchMatches[activeSearchIndex] : undefined;

  useEffect(() => {
    if (!activeSearchMatch || activeSearchMatch.pageNumber === pageNumber) {
      return;
    }

    setPageNumber(activeSearchMatch.pageNumber);
  }, [activeSearchMatch, pageNumber]);

  useEffect(() => {
    const textLayer = textLayerRef.current;
    const trimmedQuery = searchQuery.trim();
    if (!textLayer || status !== "ready" || trimmedQuery.length === 0) {
      setSearchHighlightRects([]);
      return;
    }

    const textLayerRect = textLayer.getBoundingClientRect();
    const normalizedQuery = trimmedQuery.toLocaleLowerCase();
    const nextRects: PdfSearchHighlightRect[] = [];
    const walker = document.createTreeWalker(textLayer, NodeFilter.SHOW_TEXT);
    let pageMatchIndex = 0;

    while (walker.nextNode()) {
      const textNode = walker.currentNode;
      const nodeText = textNode.textContent ?? "";
      const normalizedNodeText = nodeText.toLocaleLowerCase();
      let matchIndex = normalizedNodeText.indexOf(normalizedQuery);

      while (matchIndex !== -1) {
        const range = document.createRange();
        range.setStart(textNode, matchIndex);
        range.setEnd(textNode, matchIndex + trimmedQuery.length);

        const isActive =
          activeSearchMatch?.pageNumber === pageNumber && activeSearchMatch.pageMatchIndex === pageMatchIndex;

        for (const rect of Array.from(range.getClientRects())) {
          const left = Math.max(rect.left, textLayerRect.left) - textLayerRect.left;
          const top = Math.max(rect.top, textLayerRect.top) - textLayerRect.top;
          const right = Math.min(rect.right, textLayerRect.right) - textLayerRect.left;
          const bottom = Math.min(rect.bottom, textLayerRect.bottom) - textLayerRect.top;

          nextRects.push({
            left,
            top,
            width: Math.max(right - left, 0),
            height: Math.max(bottom - top, 0),
            active: isActive,
          });
        }

        range.detach();
        pageMatchIndex += 1;
        matchIndex = normalizedNodeText.indexOf(normalizedQuery, matchIndex + normalizedQuery.length);
      }
    }

    setSearchHighlightRects(nextRects.filter((rect) => rect.width > 0 && rect.height > 0));
  }, [activeSearchMatch, pageNumber, searchQuery, status]);

  const capturePaperSelection = () => {
    const selection = window.getSelection();
    const textLayer = textLayerRef.current;
    if (!selection || !textLayer || selection.rangeCount === 0) {
      return;
    }

    const anchorInside = selection.anchorNode ? textLayer.contains(selection.anchorNode) : false;
    const focusInside = selection.focusNode ? textLayer.contains(selection.focusNode) : false;
    const selectedText = selection.toString().replace(/\s+/g, " ").trim();
    if (!anchorInside && !focusInside) {
      return;
    }

    if (!selectedText) {
      return;
    }

    const textLayerRect = textLayer.getBoundingClientRect();
    const range = selection.getRangeAt(0);
    const highlightRects = Array.from(range.getClientRects())
      .map((rect) => {
        const left = Math.max(rect.left, textLayerRect.left) - textLayerRect.left;
        const top = Math.max(rect.top, textLayerRect.top) - textLayerRect.top;
        const right = Math.min(rect.right, textLayerRect.right) - textLayerRect.left;
        const bottom = Math.min(rect.bottom, textLayerRect.bottom) - textLayerRect.top;

        return {
          left,
          top,
          width: Math.max(right - left, 0),
          height: Math.max(bottom - top, 0),
        };
      })
      .filter((rect) => rect.width > 0 && rect.height > 0);

    onPaperSelectionChange({
      modelId: model.id,
      pageNumber,
      text: selectedText.slice(0, 4000),
      highlightRects,
    });
  };

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

  const goToSearchMatch = (direction: -1 | 1) => {
    setActiveSearchIndex((current) => {
      if (searchMatches.length === 0) {
        return -1;
      }

      const normalizedIndex = current >= 0 ? current : 0;
      return (normalizedIndex + direction + searchMatches.length) % searchMatches.length;
    });
  };

  const handleSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter") {
      return;
    }

    event.preventDefault();
    goToSearchMatch(event.shiftKey ? -1 : 1);
  };

  const trimmedSearchQuery = searchQuery.trim();
  const hasSearchQuery = trimmedSearchQuery.length > 0;
  const searchCountLabel =
    !hasSearchQuery
      ? ""
      : isSearchPending
        ? "..."
        : searchMatches.length > 0
          ? `${Math.max(activeSearchIndex + 1, 1)} / ${searchMatches.length}`
          : "0 / 0";
  const hasSearchMatches = searchMatches.length > 0;
  const searchControlsDisabled = !hasSearchMatches || isSearchPending;
  const fullscreenSupported = typeof document !== "undefined" && document.fullscreenEnabled;

  return (
    <div className="paper-viewer" ref={viewerRef}>
      <div className="pdf-canvas-wrap">
        {status !== "ready" ? (
          <div className={`pdf-status ${status === "error" ? "error" : ""}`}>
            {status === "error" ? "PDF could not be rendered" : "Rendering PDF"}
          </div>
        ) : null}
        <div className="pdf-page-shell">
          <canvas ref={canvasRef} className="pdf-canvas" aria-label={`${model.paper.title} page ${pageNumber}`} />
          {paperSelection && paperSelection.modelId === model.id && paperSelection.pageNumber === pageNumber ? (
            <div className="pdf-selection-highlight" aria-hidden="true">
              {paperSelection.highlightRects.map((rect, index) => (
                <span
                  key={`${index}-${rect.left}-${rect.top}`}
                  style={{
                    left: `${rect.left}px`,
                    top: `${rect.top}px`,
                    width: `${rect.width}px`,
                    height: `${rect.height}px`,
                  }}
                />
              ))}
            </div>
          ) : null}
          {searchHighlightRects.length > 0 ? (
            <div className="pdf-search-highlight" aria-hidden="true">
              {searchHighlightRects.map((rect, index) => (
                <span
                  className={rect.active ? "active" : ""}
                  key={`${index}-${rect.left}-${rect.top}-${rect.active}`}
                  style={{
                    left: `${rect.left}px`,
                    top: `${rect.top}px`,
                    width: `${rect.width}px`,
                    height: `${rect.height}px`,
                  }}
                />
              ))}
            </div>
          ) : null}
          <div
            className="pdf-text-layer"
            ref={textLayerRef}
            onMouseUp={capturePaperSelection}
            onTouchEnd={capturePaperSelection}
          />
        </div>
        {paperSelection && paperSelection.modelId === model.id && paperSelection.pageNumber === pageNumber ? (
          <div className="pdf-selection-status">Paper text selected</div>
        ) : null}
      </div>
      <div className="pdf-controls-dock">
        <div className="pdf-controls">
          <label className="pdf-search-field">
            <SearchIcon />
            <input
              aria-label={`Search ${model.paper.title}`}
              value={searchQuery}
              disabled={!pdfDocument || status === "error"}
              onChange={(event) => setSearchQuery(event.currentTarget.value)}
              onKeyDown={handleSearchKeyDown}
              placeholder="Search paper"
            />
            {searchCountLabel ? (
              <span className={`pdf-search-count ${searchMatches.length === 0 && !isSearchPending ? "empty" : ""}`}>
                {searchCountLabel}
              </span>
            ) : null}
          </label>
          {hasSearchQuery ? (
            <div className="pdf-control-group" role="group" aria-label="Paper search match navigation">
              {hasSearchMatches ? (
                <>
                  <button
                    className="pdf-control-button"
                    type="button"
                    aria-label="Previous search match"
                    title="Previous match"
                    disabled={searchControlsDisabled}
                    onClick={() => goToSearchMatch(-1)}
                  >
                    <ArrowIcon direction="left" />
                  </button>
                  <button
                    className="pdf-control-button"
                    type="button"
                    aria-label="Next search match"
                    title="Next match"
                    disabled={searchControlsDisabled}
                    onClick={() => goToSearchMatch(1)}
                  >
                    <ArrowIcon direction="right" />
                  </button>
                </>
              ) : null}
              <button
                className="pdf-control-button"
                type="button"
                aria-label="Clear paper search"
                title="Clear search"
                onClick={() => setSearchQuery("")}
              >
                <ClearIcon />
              </button>
            </div>
          ) : null}
          <div className="pdf-control-group pdf-page-controls" role="group" aria-label="Paper page navigation">
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
          </div>
          <div className="pdf-control-group" role="group" aria-label="Paper viewer actions">
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
    </div>
  );
}

function VariantSlider({
  model,
  onVariantChange,
}: {
  model: ModelSpec;
  onVariantChange: (variantId: string) => void;
}) {
  if (!model.variants || model.variants.length <= 1) {
    return null;
  }

  const activeVariantId = model.activeVariantId ?? model.variants[0].id;
  const activeIndex = Math.max(
    0,
    model.variants.findIndex((variant) => variant.id === activeVariantId),
  );
  const activeVariant = model.variants[activeIndex];
  const updateFromSlider = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextIndex = Number(event.currentTarget.value);
    const nextVariant = model.variants?.[nextIndex];
    if (nextVariant) {
      onVariantChange(nextVariant.id);
    }
  };

  return (
    <div className="variant-slider">
      <div className="variant-slider-label">
        <span>Variant</span>
        <strong>{activeVariant.label}</strong>
      </div>
      <input
        type="range"
        min={0}
        max={model.variants.length - 1}
        step={1}
        value={activeIndex}
        aria-label="Select ResNet depth"
        onChange={updateFromSlider}
        onInput={updateFromSlider}
      />
      <div className="variant-ticks" aria-label="ResNet depth options">
        {model.variants.map((variant, index) => (
          <button
            type="button"
            key={variant.id}
            aria-label={`Select ${variant.label}`}
            aria-pressed={index === activeIndex}
            onClick={() => onVariantChange(variant.id)}
          >
            {variant.depth}
          </button>
        ))}
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
  const [variantByModel, setVariantByModel] = useState<Record<string, string>>({});
  const baseModel = models.find((entry) => entry.id === modelId) ?? modelsByPublicationDate[0];
  const activeVariantId = variantByModel[baseModel.id];
  const model = resolveModelVariant(baseModel, activeVariantId);
  const currentModelKey = modelStateKey(model);
  const [expandedByModel, setExpandedByModel] = useState<Record<string, Set<string>>>({});
  const [selectedByModel, setSelectedByModel] = useState<Record<string, ArchNode | null>>({});
  const [visibleColumns, setVisibleColumns] = useState<Record<PaneKey, boolean>>(() => activeVisibleColumns);
  const [query, setQuery] = useState("");
  const [codeLanguage, setCodeLanguage] = useState<CodeLanguage>("pytorch");
  const [agentCodeSelection, setAgentCodeSelection] = useState<AgentCodeSelection | null>(null);
  const [userCodeSelection, setUserCodeSelection] = useState<UserCodeSelection | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [paperSelection, setPaperSelection] = useState<PaperSelection>(null);
  const [customPaneWidths, setCustomPaneWidths] = useState<Record<string, number[]>>({});
  const workspaceRef = useRef<HTMLDivElement>(null);

  const paneOrder: PaneKey[] = ["architecture", "code", "paper", "chat"];
  const visiblePanes = paneOrder.filter((pane) => visibleColumns[pane]);
  const paneLayout = visiblePanes.join("-");
  const resizedPaneWidths =
    customPaneWidths[paneLayout]?.length === visiblePanes.length ? customPaneWidths[paneLayout] : null;
  const workspaceStyle = resizedPaneWidths
    ? ({
        gridTemplateColumns: resizedPaneWidths.map((width) => `${Math.round(width)}px`).join(" 8px "),
      } as React.CSSProperties)
    : undefined;
  const expanded = expandedByModel[currentModelKey] ?? new Set<string>();
  const selected = selectedByModel[currentModelKey] ?? null;

  useEffect(() => {
    setModelId(initialModel.id);
    setExpandedByModel({});
    setSelectedByModel({});
    setAgentCodeSelection(null);
    setUserCodeSelection(null);
    setPaperSelection(null);
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
    setAgentCodeSelection(null);
    setUserCodeSelection(null);
    setPaperSelection(null);
    setQuery("");

    const nextPath = `/models/${nextModelId}`;
    if (pathname !== nextPath) {
      router.push(nextPath);
    }
  };

  const updateVariant = (nextVariantId: string) => {
    setVariantByModel((current) => ({
      ...current,
      [baseModel.id]: nextVariantId,
    }));
    setAgentCodeSelection(null);
    setUserCodeSelection(null);
    setPaperSelection(null);
  };

  const updateAgentCodeSelection = (selection: AgentCodeSelection | null) => {
    setAgentCodeSelection(selection);
    if (!selection) {
      return;
    }

    setCodeLanguage(selection.language);
    setVisibleColumns((current) => {
      if (current.code) {
        return current;
      }

      const next = {
        ...current,
        code: true,
      };
      activeVisibleColumns = next;

      return next;
    });
  };

  const toggleColumn = (pane: PaneKey) => {
    setVisibleColumns((current) => {
      const visibleCount = paneOrder.filter((entry) => current[entry]).length;
      if (current[pane] && visibleCount === 1) {
        return current;
      }

      const next = {
        ...current,
        [pane]: !current[pane],
      };
      activeVisibleColumns = next;

      return next;
    });
  };

  const readCurrentPaneWidths = () => {
    const workspace = workspaceRef.current;
    if (!workspace) {
      return null;
    }

    const widths = visiblePanes.map((pane) => {
      const element = workspace.querySelector(`:scope > .${pane}-pane`);
      if (!(element instanceof HTMLElement)) {
        return 0;
      }

      return element.getBoundingClientRect().width;
    });

    if (widths.some((width) => width <= 0)) {
      return null;
    }

    return widths;
  };

  const applyPaneResize = (leftPaneIndex: number, delta: number, startingWidths?: number[]) => {
    const widths = startingWidths ?? resizedPaneWidths ?? readCurrentPaneWidths();
    const rightPaneIndex = leftPaneIndex + 1;
    if (!widths || !visiblePanes[leftPaneIndex] || !visiblePanes[rightPaneIndex]) {
      return;
    }

    const pairTotal = widths[leftPaneIndex] + widths[rightPaneIndex];
    const leftMin = paneMinWidths[visiblePanes[leftPaneIndex]];
    const rightMin = paneMinWidths[visiblePanes[rightPaneIndex]];
    if (pairTotal <= leftMin + rightMin) {
      return;
    }

    const nextLeftWidth = Math.min(Math.max(widths[leftPaneIndex] + delta, leftMin), pairTotal - rightMin);
    const nextWidths = widths.map((width) => Math.round(width));
    nextWidths[leftPaneIndex] = Math.round(nextLeftWidth);
    nextWidths[rightPaneIndex] = Math.round(pairTotal - nextLeftWidth);

    setCustomPaneWidths((current) => ({
      ...current,
      [paneLayout]: nextWidths,
    }));
  };

  const startPaneResize = (event: React.PointerEvent<HTMLDivElement>, leftPaneIndex: number) => {
    const startingWidths = readCurrentPaneWidths();
    const workspace = workspaceRef.current;
    if (!startingWidths || !workspace) {
      return;
    }

    event.preventDefault();
    workspace.dataset.resizing = "true";

    const startX = event.clientX;
    const onPointerMove = (moveEvent: PointerEvent) => {
      moveEvent.preventDefault();
      applyPaneResize(leftPaneIndex, moveEvent.clientX - startX, startingWidths);
    };
    const stopPaneResize = () => {
      delete workspace.dataset.resizing;
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", stopPaneResize);
      document.removeEventListener("pointercancel", stopPaneResize);
    };

    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", stopPaneResize, { once: true });
    document.addEventListener("pointercancel", stopPaneResize, { once: true });
  };

  const resizePaneFromKeyboard = (event: React.KeyboardEvent<HTMLDivElement>, leftPaneIndex: number) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }

    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? -paneResizeStep : paneResizeStep;
    applyPaneResize(leftPaneIndex, delta);
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
            <VariantSlider model={model} onVariantChange={updateVariant} />
          </div>
          <div className="tree-scroll">
            <ArchitectureTree
              nodes={model.nodes}
              selectedId={selected?.id ?? null}
              onSelect={(node) => {
                setAgentCodeSelection(null);
                setSelectedByModel((current) => ({
                  ...current,
                  [currentModelKey]: node,
                }));
              }}
              expanded={expanded}
              setExpanded={(next) =>
                setExpandedByModel((current) => ({
                  ...current,
                  [currentModelKey]: next,
                }))
              }
              query={query}
            />
          </div>
        </section>
      );
    }

    if (pane === "paper") {
      return <PaperPane model={model} paperSelection={paperSelection} onPaperSelectionChange={setPaperSelection} />;
    }

    if (pane === "chat") {
      return (
        <ChatPanel
          model={model}
          selected={selected}
          language={codeLanguage}
          query={query}
          paperSelection={paperSelection}
          agentCodeSelection={agentCodeSelection}
          userCodeSelection={userCodeSelection}
          onAgentCodeSelectionChange={updateAgentCodeSelection}
          messages={chatMessages}
          setMessages={setChatMessages}
        />
      );
    }

    return (
      <CodeEditor
        model={model}
        selected={selected}
        language={codeLanguage}
        setLanguage={setCodeLanguage}
        agentCodeSelection={agentCodeSelection}
        onAgentCodeSelectionChange={updateAgentCodeSelection}
        userCodeSelection={userCodeSelection}
        onUserCodeSelectionChange={setUserCodeSelection}
      />
    );
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
          value={baseModel.id}
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

      <div
        className="workspace"
        data-pane-count={visiblePanes.length}
        data-pane-layout={paneLayout}
        ref={workspaceRef}
        style={workspaceStyle}
      >
        {visiblePanes.map((pane, index) => (
          <Fragment key={pane}>
            {index > 0 ? (
              <div
                className="divider"
                role="separator"
                aria-label={`Resize ${visiblePanes[index - 1]} and ${pane} panes`}
                aria-orientation="vertical"
                tabIndex={0}
                onKeyDown={(event) => resizePaneFromKeyboard(event, index - 1)}
                onPointerDown={(event) => startPaneResize(event, index - 1)}
              />
            ) : null}
            {renderPane(pane)}
          </Fragment>
        ))}
      </div>
    </main>
  );
}
