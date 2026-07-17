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
import {
  normalizeLineNumbers,
  type ChatCodeLanguage,
  type ChatMessage,
  type ChatResponse,
} from "./chat-contract";
import {
  modelHighlightManifest,
  modelSources,
  resnetVariantDefinitions,
  type ModelHighlightAnchor,
} from "./generated/model-sources";
import { modelCatalog, modelRoutePath, type ModelId } from "./model-routes";

type CodeLanguage = ChatCodeLanguage;

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
  sourceRefs: Record<CodeLanguage, string[]>;
  focusRef?: Partial<Record<CodeLanguage, string>>;
  includeChildRefs: boolean;
};

type ModelSpec = {
  id: string;
  label: string;
  stats: string;
  fileName: string;
  jaxFileName: string;
  paper: {
    title: string;
    authors: string;
    publishedDate: string;
    venue: string;
    pdfUrl: string;
    focus: string[];
  };
  nodes: ArchNode[];
  code: string[];
  jaxCode: string[];
  highlights: Readonly<Record<string, ModelHighlightAnchor>>;
  jaxHighlights: Readonly<Record<string, ModelHighlightAnchor>>;
  variants?: ModelVariantSpec[];
  activeVariantId?: string;
};

type ModelVariantSpec = {
  id: string;
  label: string;
  depth: number;
  blockClass: "BasicBlock" | "Bottleneck";
  blockLabel: string;
  stageBlocks: readonly [number, number, number, number];
  expansion: 1 | 4;
  stats: string;
  fileName: string;
  jaxFileName: string;
  nodes: ArchNode[];
  code: string[];
  jaxCode: string[];
  highlights: Readonly<Record<string, ModelHighlightAnchor>>;
  jaxHighlights: Readonly<Record<string, ModelHighlightAnchor>>;
};

type ModelDefinition = Omit<
  ModelSpec,
  | "id"
  | "label"
  | "paper"
  | "fileName"
  | "jaxFileName"
  | "code"
  | "jaxCode"
  | "highlights"
  | "jaxHighlights"
>;

type PaneKey = "architecture" | "paper" | "code" | "chat";
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

function modelSourcePair(baseName: string) {
  const fileName = `${baseName}.py`;
  const jaxFileName = `${baseName}_jax.py`;
  const source = modelSources[fileName];
  const jaxSource = modelSources[jaxFileName];
  const highlights = modelHighlightManifest[fileName];
  const jaxHighlights = modelHighlightManifest[jaxFileName];

  if (!source || !jaxSource || !highlights || !jaxHighlights) {
    throw new Error(`Missing generated model source pair for ${baseName}`);
  }

  return {
    fileName,
    jaxFileName,
    code: codeLines(source),
    jaxCode: codeLines(jaxSource),
    highlights,
    jaxHighlights,
  };
}

function notebookFileName(fileName: string) {
  return fileName.replace(/\.py$/, ".ipynb");
}

function colabUrl(notebookName: string) {
  return `https://colab.research.google.com/github/${githubRepository}/blob/${githubBranch}/public/notebooks/${notebookName}`;
}

type ResNetTemplateVariant = (typeof resnetVariantDefinitions)[number];

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
  const stageName = `layer${stageIndex + 1}`;
  const stageBaseChannels = stageChannels[stageIndex];
  const previousStageBaseChannels = stageIndex === 0 ? 64 : stageChannels[stageIndex - 1];
  const inputChannels = blockIndex === 0 ? previousStageBaseChannels * variant.expansion : stageBaseChannels * variant.expansion;
  const outputChannels = stageBaseChannels * variant.expansion;
  const stride = blockIndex === 0 && stageIndex > 0 ? 2 : 1;
  const needsProjection = stride !== 1 || inputChannels !== outputChannels;
  const blockNode: ArchNode = {
    id: `${stageName}.${blockIndex}`,
    label: `block.${blockIndex}`,
    type: variant.blockClass,
    kind: "residual",
    summary: needsProjection ? `stride ${stride} + projection` : "identity skip",
    defaultExpanded,
    sourceRefs: variant.blockClass === "BasicBlock" ? {
      pytorch: ["class-basicblock-nn-module","basicblock.self-convn-nn-convnd","basicblock.self-bnn-nn-batchnormnd-out_channels","basicblock.self-convn-nn-convnd-out_channels-out_channels-kernel_size-n-padding-n-b","basicblock.self-bnn-nn-batchnormnd-out_channels.2","basicblock.forward.out-self-convn-x","basicblock.forward.out-self-bnn-out","basicblock.forward.out-self-relu-out","basicblock.forward.out-self-convn-out","basicblock.forward.out-self-bnn-out.2","basicblock.forward.out-out-identity","basicblock.forward.out-self-relu-out.2","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample","resnetn._make_layer.current_block-block-self-in_channels-out_channels"],
      jax: ["class-basicblock-nn-module","basicblock.__call__.y-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-padding-","basicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y","basicblock.__call__.y-nn-relu-y","basicblock.__call__.y-nn-conv-self-out_channels-n-n-padding-same-use_bias-false-name-convn-y","basicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y.2","basicblock.__call__.y-y-identity","basicblock.__call__.y-nn-relu-y.2","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train","resnetn._stage.x-block-channels-x-train-train"],
    } : {
      pytorch: ["class-bottleneck-nn-module","bottleneck.self-convn-nn-convnd-in_channels-out_channels-kernel_size-n-bias-false","bottleneck.self-convn-nn-convnd","bottleneck.self-convn-nn-convnd-out_channels-expanded_channels-kernel_size-n-bias-f","bottleneck.self-bnn-nn-batchnormnd-expanded_channels","bottleneck.forward.out-self-convn-x","bottleneck.forward.out-self-bnn-out","bottleneck.forward.out-self-relu-out","bottleneck.forward.out-self-convn-out","bottleneck.forward.out-self-bnn-out.2","bottleneck.forward.out-self-relu-out.2","bottleneck.forward.out-self-convn-out.2","bottleneck.forward.out-self-bnn-out.3","bottleneck.forward.out-out-identity","bottleneck.forward.out-self-relu-out.3","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample","resnetn._make_layer.current_block-block-self-in_channels-out_channels"],
      jax: ["class-bottleneck-nn-module","bottleneck.__call__.y-nn-conv-self-out_channels-n-n-use_bias-false-name-convn-x","bottleneck.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y","bottleneck.__call__.y-nn-relu-y","bottleneck.__call__.y-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-padding-","bottleneck.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y.2","bottleneck.__call__.y-nn-relu-y.2","bottleneck.__call__.y-nn-conv-expanded_channels-n-n-use_bias-false-name-convn-y","bottleneck.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y.3","bottleneck.__call__.y-y-identity","bottleneck.__call__.y-nn-relu-y.3","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train","resnetn._stage.x-block-channels-x-train-train"],
    },
    focusRef: variant.blockClass === "BasicBlock" ? {
      pytorch: "class-basicblock-nn-module",
      jax: "class-basicblock-nn-module",
    } : {
      pytorch: "class-bottleneck-nn-module",
      jax: "class-bottleneck-nn-module",
    },
    includeChildRefs: false,
  };

  if (variant.blockClass === "BasicBlock") {
    const children: ArchNode[] = [
      {
        id: `${stageName}.${blockIndex}.conv1`,
        label: "conv1",
        type: "Conv2d",
        kind: "conv",
        badges: [`${inputChannels}->${stageBaseChannels}`, "k=3", ...(stride > 1 ? [`s=${stride}`] : [])],
        sourceRefs: {
          pytorch: ["basicblock.self-convn-nn-convnd","basicblock.forward.out-self-convn-x","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample","resnetn._make_layer.current_block-block-self-in_channels-out_channels"],
          jax: ["basicblock.__call__.y-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-padding-","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train","resnetn._stage.x-block-channels-x-train-train"],
        },
        focusRef: {
          pytorch: "basicblock.self-convn-nn-convnd",
          jax: "basicblock.__call__.y-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-padding-",
        },
        includeChildRefs: false,
      },
      {
        id: `${stageName}.${blockIndex}.bn1`,
        label: "bn1",
        type: "BatchNorm2d",
        kind: "norm",
        badges: [`${stageBaseChannels}`],
        sourceRefs: {
          pytorch: ["basicblock.self-bnn-nn-batchnormnd-out_channels","basicblock.forward.out-self-bnn-out"],
          jax: ["basicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y"],
        },
        focusRef: {
          pytorch: "basicblock.self-bnn-nn-batchnormnd-out_channels",
          jax: "basicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y",
        },
        includeChildRefs: false,
      },
      {
        id: `${stageName}.${blockIndex}.relu1`,
        label: "relu",
        type: "ReLU",
        kind: "activation",
        sourceRefs: {
          pytorch: ["basicblock.self-relu-nn-relu-inplace-true","basicblock.forward.out-self-relu-out"],
          jax: ["basicblock.__call__.y-nn-relu-y"],
        },
        focusRef: {
          pytorch: "basicblock.self-relu-nn-relu-inplace-true",
          jax: "basicblock.__call__.y-nn-relu-y",
        },
        includeChildRefs: false,
      },
      {
        id: `${stageName}.${blockIndex}.conv2`,
        label: "conv2",
        type: "Conv2d",
        kind: "conv",
        badges: [`${stageBaseChannels}->${outputChannels}`, "k=3"],
        sourceRefs: {
          pytorch: ["basicblock.self-convn-nn-convnd-out_channels-out_channels-kernel_size-n-padding-n-b","basicblock.forward.out-self-convn-out"],
          jax: ["basicblock.__call__.y-nn-conv-self-out_channels-n-n-padding-same-use_bias-false-name-convn-y"],
        },
        focusRef: {
          pytorch: "basicblock.self-convn-nn-convnd-out_channels-out_channels-kernel_size-n-padding-n-b",
          jax: "basicblock.__call__.y-nn-conv-self-out_channels-n-n-padding-same-use_bias-false-name-convn-y",
        },
        includeChildRefs: false,
      },
      {
        id: `${stageName}.${blockIndex}.bn2`,
        label: "bn2",
        type: "BatchNorm2d",
        kind: "norm",
        badges: [`${outputChannels}`],
        sourceRefs: {
          pytorch: ["basicblock.self-bnn-nn-batchnormnd-out_channels.2","basicblock.forward.out-self-bnn-out.2"],
          jax: ["basicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y.2"],
        },
        focusRef: {
          pytorch: "basicblock.self-bnn-nn-batchnormnd-out_channels.2",
          jax: "basicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y.2",
        },
        includeChildRefs: false,
      },
    ];

    if (needsProjection) {
      children.push({
        id: `${stageName}.${blockIndex}.downsample`,
        label: "downsample",
        type: "ProjectionSkip",
        kind: "group",
        summary: `1x1 stride ${stride}`,
        sourceRefs: {
          pytorch: ["basicblock.forward.if-self-downsample-is-not-none","resnetn._make_layer.if-stride-n-or-self-in_channels-expanded_channels","resnetn._make_layer.downsample-nn-sequential","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample"],
          jax: ["basicblock.__call__.if-self-use_projection","resnetn._stage.use_projection-stride-n-or-x-shape-n-expanded_channels"],
        },
        focusRef: {
          pytorch: "resnetn._make_layer.if-stride-n-or-self-in_channels-expanded_channels",
          jax: "basicblock.__call__.if-self-use_projection",
        },
        includeChildRefs: true,
        children: [
          {
            id: `${stageName}.${blockIndex}.downsample.conv`,
            label: "conv1x1",
            type: "Conv2d",
            kind: "conv",
            badges: [`${inputChannels}->${outputChannels}`, `s=${stride}`],
            sourceRefs: {
              pytorch: ["basicblock.forward.identity-self-downsample-x","resnetn._make_layer.nn-convnd-self-in_channels-expanded_channels-kernel_size-n-stride-stride"],
              jax: ["basicblock.__call__.identity-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-u","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train"],
            },
            focusRef: {
              pytorch: "resnetn._make_layer.nn-convnd-self-in_channels-expanded_channels-kernel_size-n-stride-stride",
              jax: "basicblock.__call__.identity-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-u",
            },
            includeChildRefs: false,
          },
          {
            id: `${stageName}.${blockIndex}.downsample.bn`,
            label: "bn",
            type: "BatchNorm2d",
            kind: "norm",
            badges: [`${outputChannels}`],
            sourceRefs: {
              pytorch: ["basicblock.forward.identity-self-downsample-x","resnetn._make_layer.nn-batchnormnd-expanded_channels"],
              jax: ["basicblock.__call__.identity-nn-batchnorm-use_running_average-not-train-name-downsample_bn-i"],
            },
            focusRef: {
              pytorch: "resnetn._make_layer.nn-batchnormnd-expanded_channels",
              jax: "basicblock.__call__.identity-nn-batchnorm-use_running_average-not-train-name-downsample_bn-i",
            },
            includeChildRefs: false,
          },
        ],
      });
    }

    children.push({
      id: `${stageName}.${blockIndex}.add`,
      label: "add",
      type: "ResidualAdd",
      kind: "residual",
      sourceRefs: {
        pytorch: ["basicblock.forward.out-out-identity"],
        jax: ["basicblock.__call__.y-y-identity"],
      },
      focusRef: {
        pytorch: "basicblock.forward.out-out-identity",
        jax: "basicblock.__call__.y-y-identity",
      },
      includeChildRefs: false,
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
      sourceRefs: {
        pytorch: ["bottleneck.self-convn-nn-convnd-in_channels-out_channels-kernel_size-n-bias-false","bottleneck.forward.out-self-convn-x"],
        jax: ["bottleneck.__call__.y-nn-conv-self-out_channels-n-n-use_bias-false-name-convn-x"],
      },
      focusRef: {
        pytorch: "bottleneck.self-convn-nn-convnd-in_channels-out_channels-kernel_size-n-bias-false",
        jax: "bottleneck.__call__.y-nn-conv-self-out_channels-n-n-use_bias-false-name-convn-x",
      },
      includeChildRefs: false,
    },
    {
      id: `${stageName}.${blockIndex}.conv2`,
      label: "conv2",
      type: "Conv2d",
      kind: "conv",
      badges: [`${stageBaseChannels}->${stageBaseChannels}`, "k=3", ...(stride > 1 ? [`s=${stride}`] : [])],
      sourceRefs: {
        pytorch: ["bottleneck.self-convn-nn-convnd","bottleneck.forward.out-self-convn-out"],
        jax: ["bottleneck.__call__.y-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-padding-"],
      },
      focusRef: {
        pytorch: "bottleneck.self-convn-nn-convnd",
        jax: "bottleneck.__call__.y-nn-conv-self-out_channels-n-n-strides-self-stride-self-stride-padding-",
      },
      includeChildRefs: false,
    },
    {
      id: `${stageName}.${blockIndex}.conv3`,
      label: "conv3",
      type: "Conv2d",
      kind: "conv",
      badges: [`${stageBaseChannels}->${outputChannels}`, "k=1"],
      sourceRefs: {
        pytorch: ["bottleneck.self-convn-nn-convnd-out_channels-expanded_channels-kernel_size-n-bias-f","bottleneck.forward.out-self-convn-out.2"],
        jax: ["bottleneck.__call__.y-nn-conv-expanded_channels-n-n-use_bias-false-name-convn-y"],
      },
      focusRef: {
        pytorch: "bottleneck.self-convn-nn-convnd-out_channels-expanded_channels-kernel_size-n-bias-f",
        jax: "bottleneck.__call__.y-nn-conv-expanded_channels-n-n-use_bias-false-name-convn-y",
      },
      includeChildRefs: false,
    },
  ];

  if (needsProjection) {
    bottleneckChildren.push({
      id: `${stageName}.${blockIndex}.downsample`,
      label: "downsample",
      type: "ProjectionSkip",
      kind: "group",
      summary: `1x1 stride ${stride}`,
      sourceRefs: {
        pytorch: ["bottleneck.forward.if-self-downsample-is-not-none","resnetn._make_layer.if-stride-n-or-self-in_channels-expanded_channels","resnetn._make_layer.downsample-nn-sequential","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample"],
        jax: ["bottleneck.__call__.if-self-use_projection","resnetn._stage.use_projection-stride-n-or-x-shape-n-expanded_channels"],
      },
      focusRef: {
        pytorch: "resnetn._make_layer.if-stride-n-or-self-in_channels-expanded_channels",
        jax: "bottleneck.__call__.if-self-use_projection",
      },
      includeChildRefs: true,
      children: [
        {
          id: `${stageName}.${blockIndex}.downsample.conv`,
          label: "conv1x1",
          type: "Conv2d",
          kind: "conv",
          badges: [`${inputChannels}->${outputChannels}`, `s=${stride}`],
          sourceRefs: {
            pytorch: ["bottleneck.forward.identity-self-downsample-x","resnetn._make_layer.nn-convnd-self-in_channels-expanded_channels-kernel_size-n-stride-stride"],
            jax: ["bottleneck.__call__.identity-nn-conv-expanded_channels-n-n-strides-self-stride-self-stride-u","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train"],
          },
          focusRef: {
            pytorch: "resnetn._make_layer.nn-convnd-self-in_channels-expanded_channels-kernel_size-n-stride-stride",
            jax: "bottleneck.__call__.identity-nn-conv-expanded_channels-n-n-strides-self-stride-self-stride-u",
          },
          includeChildRefs: false,
        },
        {
          id: `${stageName}.${blockIndex}.downsample.bn`,
          label: "bn",
          type: "BatchNorm2d",
          kind: "norm",
          badges: [`${outputChannels}`],
          sourceRefs: {
            pytorch: ["bottleneck.forward.identity-self-downsample-x","resnetn._make_layer.nn-batchnormnd-expanded_channels"],
            jax: ["bottleneck.__call__.identity-nn-batchnorm-use_running_average-not-train-name-downsample_bn-i"],
          },
          focusRef: {
            pytorch: "resnetn._make_layer.nn-batchnormnd-expanded_channels",
            jax: "bottleneck.__call__.identity-nn-batchnorm-use_running_average-not-train-name-downsample_bn-i",
          },
          includeChildRefs: false,
        },
      ],
    });
  }

  bottleneckChildren.push({
    id: `${stageName}.${blockIndex}.add`,
    label: "add",
    type: "ResidualAdd",
    kind: "residual",
    sourceRefs: {
      pytorch: ["bottleneck.forward.out-out-identity"],
      jax: ["bottleneck.__call__.y-y-identity"],
    },
    focusRef: {
      pytorch: "bottleneck.forward.out-out-identity",
      jax: "bottleneck.__call__.y-y-identity",
    },
    includeChildRefs: false,
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
      sourceRefs: {
        pytorch: ["resnetn.forward.x-self-stem-x"],
        jax: ["resnetn.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-use_bias-false-name-stem_conv-x"],
      },
      focusRef: {
        pytorch: "resnetn.forward.x-self-stem-x",
        jax: "resnetn.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-use_bias-false-name-stem_conv-x",
      },
      includeChildRefs: false,
    },
    {
      id: "stem",
      label: "stem",
      type: "Conv-BN-ReLU",
      kind: "group",
      summary: "7x7 stride 2",
      defaultExpanded: true,
      sourceRefs: {
        pytorch: ["resnetn.self-stem-nn-sequential","resnetn.code.4"],
        jax: [],
      },
      focusRef: {
        pytorch: "resnetn.self-stem-nn-sequential",
        jax: "resnetn.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-use_bias-false-name-stem_conv-x",
      },
      includeChildRefs: true,
      children: [
        {
          id: "stem.conv",
          label: "conv",
          type: "Conv2d",
          kind: "conv",
          badges: ["3->64", "k=7", "s=2"],
          sourceRefs: {
            pytorch: ["resnetn.nn-convnd-n-n-kernel_size-n-stride-n-padding-n-bias-false","resnetn.forward.x-self-stem-x"],
            jax: ["resnetn.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-use_bias-false-name-stem_conv-x"],
          },
          focusRef: {
            pytorch: "resnetn.nn-convnd-n-n-kernel_size-n-stride-n-padding-n-bias-false",
            jax: "resnetn.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-use_bias-false-name-stem_conv-x",
          },
          includeChildRefs: false,
        },
        {
          id: "stem.bn",
          label: "bn",
          type: "BatchNorm2d",
          kind: "norm",
          badges: ["64"],
          sourceRefs: {
            pytorch: ["resnetn.nn-batchnormnd-n","resnetn.forward.x-self-stem-x"],
            jax: ["resnetn.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x"],
          },
          focusRef: {
            pytorch: "resnetn.nn-batchnormnd-n",
            jax: "resnetn.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x",
          },
          includeChildRefs: false,
        },
        {
          id: "stem.relu",
          label: "relu",
          type: "ReLU",
          kind: "activation",
          sourceRefs: {
            pytorch: ["resnetn.nn-relu-inplace-true","resnetn.forward.x-self-stem-x"],
            jax: ["resnetn.__call__.x-nn-relu-x"],
          },
          focusRef: {
            pytorch: "resnetn.nn-relu-inplace-true",
            jax: "resnetn.__call__.x-nn-relu-x",
          },
          includeChildRefs: false,
        },
      ],
    },
    {
      id: "maxpool",
      label: "maxpool",
      type: "MaxPool2d",
      kind: "pool",
      badges: ["k=3", "s=2"],
      sourceRefs: {
        pytorch: ["resnetn.self-maxpool-nn-maxpoolnd-kernel_size-n-stride-n-padding-n","resnetn.forward.x-self-maxpool-x"],
        jax: ["resnetn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same"],
      },
      focusRef: {
        pytorch: "resnetn.self-maxpool-nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
        jax: "resnetn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
      },
      includeChildRefs: false,
    },
    ...variant.stageBlocks.map((blockCount, stageIndex) => {
      const stageName = `layer${stageIndex + 1}`;
      const stageChannelsLabel = stageChannels[stageIndex] * variant.expansion;

      return {
        id: stageName,
        label: stageName,
        type: "ResidualStage",
        kind: "group",
        summary: `${blockCount} ${variant.blockLabel}s`,
        badges: [`${stageChannelsLabel} ch`, `${stageSpatial[stageIndex]}x${stageSpatial[stageIndex]}`],
        defaultExpanded: stageIndex === 1,
        sourceRefs: [{
          pytorch: ["resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n","resnetn.def-_make_layer-self-block-out_channels-blocks-stride","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample","resnetn._make_layer.for-_-in-range-n-blocks","resnetn._make_layer.current_block-block-self-in_channels-out_channels","resnetn.forward.x-self-layern-x"],
          jax: ["resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train","resnetn.def-_stage-self-x-block-channels-blocks-stride-train","resnetn._stage.use_projection-stride-n-or-x-shape-n-expanded_channels","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train","resnetn._stage.for-_-in-range-n-blocks","resnetn._stage.x-block-channels-x-train-train"],
        }, {
          pytorch: ["resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n.2","resnetn.def-_make_layer-self-block-out_channels-blocks-stride","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample","resnetn._make_layer.for-_-in-range-n-blocks","resnetn._make_layer.current_block-block-self-in_channels-out_channels","resnetn.forward.x-self-layern-x.2"],
          jax: ["resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train.2","resnetn.def-_stage-self-x-block-channels-blocks-stride-train","resnetn._stage.use_projection-stride-n-or-x-shape-n-expanded_channels","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train","resnetn._stage.for-_-in-range-n-blocks","resnetn._stage.x-block-channels-x-train-train"],
        }, {
          pytorch: ["resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n.3","resnetn.def-_make_layer-self-block-out_channels-blocks-stride","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample","resnetn._make_layer.for-_-in-range-n-blocks","resnetn._make_layer.current_block-block-self-in_channels-out_channels","resnetn.forward.x-self-layern-x.3"],
          jax: ["resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train.3","resnetn.def-_stage-self-x-block-channels-blocks-stride-train","resnetn._stage.use_projection-stride-n-or-x-shape-n-expanded_channels","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train","resnetn._stage.for-_-in-range-n-blocks","resnetn._stage.x-block-channels-x-train-train"],
        }, {
          pytorch: ["resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n.4","resnetn.def-_make_layer-self-block-out_channels-blocks-stride","resnetn._make_layer.layers-block-self-in_channels-out_channels-stride-downsample","resnetn._make_layer.for-_-in-range-n-blocks","resnetn._make_layer.current_block-block-self-in_channels-out_channels","resnetn.forward.x-self-layern-x.4"],
          jax: ["resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train.4","resnetn.def-_stage-self-x-block-channels-blocks-stride-train","resnetn._stage.use_projection-stride-n-or-x-shape-n-expanded_channels","resnetn._stage.x-block-channels-stride-use_projection-use_projection-x-train-train","resnetn._stage.for-_-in-range-n-blocks","resnetn._stage.x-block-channels-x-train-train"],
        }][stageIndex],
        focusRef: [{
          pytorch: "resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n",
          jax: "resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train",
        }, {
          pytorch: "resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n.2",
          jax: "resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train.2",
        }, {
          pytorch: "resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n.3",
          jax: "resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train.3",
        }, {
          pytorch: "resnetn.self-layern-self-_make_layer-block-n-blocks-n-stride-n.4",
          jax: "resnetn.__call__.x-self-_stage-x-block-n-blocks-n-stride-n-train-train.4",
        }][stageIndex],
        includeChildRefs: false,
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
      sourceRefs: {
        pytorch: [],
        jax: [],
      },
      focusRef: {
        pytorch: "resnetn.self-avgpool-nn-adaptiveavgpoolnd-n-n",
        jax: "resnetn.__call__.x-jnp-mean-x-axis-n-n",
      },
      includeChildRefs: true,
      children: [
        {
          id: "avgpool",
          label: "avgpool",
          type: "AdaptiveAvgPool2d",
          kind: "pool",
          badges: ["1x1"],
          sourceRefs: {
            pytorch: ["resnetn.self-avgpool-nn-adaptiveavgpoolnd-n-n","resnetn.forward.x-self-avgpool-x"],
            jax: ["resnetn.__call__.x-jnp-mean-x-axis-n-n"],
          },
          focusRef: {
            pytorch: "resnetn.self-avgpool-nn-adaptiveavgpoolnd-n-n",
            jax: "resnetn.__call__.x-jnp-mean-x-axis-n-n",
          },
          includeChildRefs: false,
        },
        {
          id: "flatten",
          label: "flatten",
          type: "Flatten",
          kind: "reshape",
          badges: [`${512 * variant.expansion}`],
          sourceRefs: {
            pytorch: ["resnetn.forward.x-torch-flatten-x-n"],
            jax: ["resnetn.__call__.x-jnp-mean-x-axis-n-n"],
          },
          focusRef: {
            pytorch: "resnetn.forward.x-torch-flatten-x-n",
            jax: "resnetn.__call__.x-jnp-mean-x-axis-n-n",
          },
          includeChildRefs: false,
        },
      ],
    },
    {
      id: "fc",
      label: "fc",
      type: "Linear",
      kind: "linear",
      badges: [`${512 * variant.expansion}->1000`],
      sourceRefs: {
        pytorch: ["resnetn.self-fc-nn-linear-n-block-expansion-num_classes","resnetn.forward.logits-self-fc-x"],
        jax: ["resnetn.__call__.logits-nn-dense-self-num_classes-name-fc-x"],
      },
      focusRef: {
        pytorch: "resnetn.self-fc-nn-linear-n-block-expansion-num_classes",
        jax: "resnetn.__call__.logits-nn-dense-self-num_classes-name-fc-x",
      },
      includeChildRefs: false,
    },
  ];
}

const resnetVariants: ModelVariantSpec[] = resnetVariantDefinitions.map((variant) => {
  return {
    ...variant,
    stats: resnetVariantStats(variant),
    ...modelSourcePair(variant.id),
    nodes: makeResNetNodes(variant),
  };
});

const completedPdfPrefetches = new Set<string>();
const pendingPdfPrefetches = new Map<string, Promise<void>>();
const paperPdfAssetVersion = "20260604";

function paperPdfUrl(modelId: string) {
  return `/papers/${modelId}.pdf?v=${paperPdfAssetVersion}`;
}

function prefetchPdf(pdfUrl: string, signal: AbortSignal) {
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
    sourceRefs: {
      pytorch: ["multiheadattention.self","multiheadattention.self-head_dim-d_model-nhead","multiheadattention.self-v_proj-nn-linear-d_model-d_model","multiheadattention.self-out_proj-nn-linear-d_model-d_model","multiheadattention.def-forward-self-query-key-value-attn_mask-none","multiheadattention.forward.batch_size-query-size-n","multiheadattention.forward.v-self-v_proj-value","multiheadattention.forward.k-k-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.k-k-transpose-n-n","multiheadattention.forward.v-v-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.v-v-transpose-n-n","multiheadattention.forward.scale-self-head_dim-n","multiheadattention.forward.attn_scores-scores-scale","multiheadattention.forward.if-attn_mask-is-not-none","multiheadattention.forward.mask-attn_mask-none-none","multiheadattention.forward.context-context-contiguous","multiheadattention.forward.merged-context-view-batch_size-query_steps-self-nhead-self-head_dim","multiheadattention.forward.return-out","class-encoderlayer-nn-module","encoderlayer.d_model-n","encoderlayer.nn-linear-d_model-d_ff","encoderlayer.nn-relu","encoderlayer.nn-linear-d_ff-d_model","encoderlayer.code.4","encoderlayer.self-normn-nn-layernorm-d_model","encoderlayer.self-normn-nn-layernorm-d_model.2","encoderlayer.def-forward-self-x-src_mask-none","encoderlayer.forward.attn_residual-x-attn","encoderlayer.forward.ffn-self-ffn-x","encoderlayer.forward.return-out","class-decoderlayer-nn-module","decoderlayer.def-__init__","transformer.forward.src_embeddings-self-src_embed-src_ids"],
      jax: ["class-encoderlayer-nn-module","encoderlayer.d_model-int-n","encoderlayer.nhead-int-n","encoderlayer.d_ff-int-n","encoderlayer.nn-compact","encoderlayer.def-__call__-self-x","encoderlayer.__call__.attn-multiheadattention-self-d_model-self-nhead-x-x-x","encoderlayer.__call__.attn_residual-x-attn","encoderlayer.__call__.x-nn-layernorm-attn_residual","encoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model","encoderlayer.__call__.ffn-nn-sequential-ffn_layers-x","encoderlayer.__call__.ffn_residual-x-ffn","encoderlayer.__call__.out-nn-layernorm-ffn_residual","encoderlayer.__call__.return-out"],
    },
    focusRef: {
      pytorch: "multiheadattention.self",
      jax: "class-encoderlayer-nn-module",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `encoder.${index}.self_attn`,
        label: "self_attn",
        type: "MultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "d=512"],
        sourceRefs: {
          pytorch: ["multiheadattention.self-v_proj-nn-linear-d_model-d_model","multiheadattention.self-out_proj-nn-linear-d_model-d_model","multiheadattention.def-forward-self-query-key-value-attn_mask-none","multiheadattention.forward.v-self-v_proj-value","multiheadattention.forward.k-k-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.k-k-transpose-n-n","multiheadattention.forward.v-v-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.v-v-transpose-n-n","multiheadattention.forward.scale-self-head_dim-n","multiheadattention.forward.attn_scores-scores-scale","multiheadattention.forward.if-attn_mask-is-not-none","multiheadattention.forward.mask-attn_mask-none-none","multiheadattention.forward.attn_weights-torch-softmax-attn_scores-dim-n","multiheadattention.forward.context-context-contiguous","multiheadattention.forward.merged-context-view-batch_size-query_steps-self-nhead-self-head_dim","multiheadattention.forward.out-self-out_proj-merged","multiheadattention.forward.return-out","encoderlayer.nn-relu"],
          jax: ["class-multiheadattention-nn-module","multiheadattention.d_model-int-n","multiheadattention.nhead-int-n","multiheadattention.nn-compact","multiheadattention.def-__call__-self-query-key-value-mask-none","multiheadattention.__call__.batch_size-query-shape-n","multiheadattention.__call__.query_steps-query-shape-n","multiheadattention.__call__.key_steps-key-shape-n","multiheadattention.__call__.head_dim-self-d_model-self-nhead","multiheadattention.__call__.q-nn-dense-self-d_model-query","multiheadattention.__call__.k-nn-dense-self-d_model-key","multiheadattention.__call__.v-nn-dense-self-d_model-value","multiheadattention.__call__.q-q-reshape-batch_size-query_steps-self-nhead-head_dim","multiheadattention.__call__.q-jnp-transpose-q-n-n-n-n","multiheadattention.__call__.k-k-reshape-batch_size-key_steps-self-nhead-head_dim","multiheadattention.__call__.k-jnp-transpose-k-n-n-n-n","multiheadattention.__call__.v-v-reshape-batch_size-key_steps-self-nhead-head_dim","multiheadattention.__call__.v-jnp-transpose-v-n-n-n-n","multiheadattention.__call__.key_transpose-jnp-swapaxes-k-n-n","multiheadattention.__call__.scores-q-key_transpose","multiheadattention.__call__.scale-head_dim-n","multiheadattention.__call__.attn_scores-scores-scale","multiheadattention.__call__.if-mask-is-not-none","multiheadattention.__call__.attn_scores-jnp-where-mask-n-jnp-inf-attn_scores","multiheadattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n","multiheadattention.__call__.context-attn_weights-v","multiheadattention.__call__.context-jnp-transpose-context-n-n-n-n","multiheadattention.__call__.merged-context-reshape-batch_size-query_steps-self-d_model","multiheadattention.__call__.out-nn-dense-self-d_model-merged","multiheadattention.__call__.return-out"],
        },
        focusRef: {
          pytorch: "multiheadattention.self-v_proj-nn-linear-d_model-d_model",
          jax: "class-multiheadattention-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        sourceRefs: {
          pytorch: ["encoderlayer.forward.attn_residual-x-attn","transformer.encoder.norm1"],
          jax: ["encoderlayer.__call__.attn_residual-x-attn","encoderlayer.__call__.x-nn-layernorm-attn_residual"],
        },
        focusRef: {
          pytorch: "encoderlayer.forward.attn_residual-x-attn",
          jax: "encoderlayer.__call__.attn_residual-x-attn",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        sourceRefs: {
          pytorch: ["encoderlayer.nn-linear-d_model-d_ff","encoderlayer.nn-relu","encoderlayer.nn-linear-d_ff-d_model","encoderlayer.forward.ffn-self-ffn-x"],
          jax: ["encoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model","encoderlayer.__call__.ffn-nn-sequential-ffn_layers-x"],
        },
        focusRef: {
          pytorch: "encoderlayer.nn-linear-d_ff-d_model",
          jax: "encoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        sourceRefs: {
          pytorch: ["transformer.encoder.ffn_residual","transformer.encoder.norm2"],
          jax: ["encoderlayer.__call__.ffn_residual-x-ffn","encoderlayer.__call__.out-nn-layernorm-ffn_residual"],
        },
        focusRef: {
          pytorch: "transformer.encoder.ffn_residual",
          jax: "encoderlayer.__call__.ffn_residual-x-ffn",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["elmanrnn.forward.current_input-x-t","elmanrnn.forward.input_hidden-self-input_to_hidden-current_input","elmanrnn.forward.recurrent_hidden-self-hidden_to_hidden-h","elmanrnn.forward.hidden_sum-input_hidden-recurrent_hidden","elmanrnn.forward.h-torch-tanh-hidden_sum","elmanrnn.forward.states-append-h"],
      jax: ["elmanrnn.__call__.current_input-x-t","elmanrnn.__call__.input_hidden-input_to_hidden-current_input","elmanrnn.__call__.recurrent_hidden-hidden_to_hidden-h","elmanrnn.__call__.hidden_sum-input_hidden-recurrent_hidden","elmanrnn.__call__.h-jnp-tanh-hidden_sum","elmanrnn.__call__.states-append-h"],
    },
    focusRef: {
      pytorch: "elmanrnn.forward.current_input-x-t",
      jax: "elmanrnn.__call__.current_input-x-t",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `step.${index}.input_to_hidden`,
        label: "input_to_hidden",
        type: "Linear",
        kind: "linear",
        badges: ["32->64"],
        sourceRefs: {
          pytorch: ["elmanrnn.input_to_hidden","elmanrnn.forward.input_hidden-self-input_to_hidden-current_input"],
          jax: ["elmanrnn.__call__.input_to_hidden-nn-dense-self-hidden_size-name-input_to_hidden","elmanrnn.__call__.current_input-x-t","elmanrnn.__call__.input_hidden-input_to_hidden-current_input"],
        },
        focusRef: {
          pytorch: "elmanrnn.input_to_hidden",
          jax: "elmanrnn.__call__.input_to_hidden-nn-dense-self-hidden_size-name-input_to_hidden",
        },
        includeChildRefs: false,
      },
      {
        id: `step.${index}.hidden_to_hidden`,
        label: "hidden_to_hidden",
        type: "RecurrentLinear",
        kind: "recurrent",
        badges: ["64->64", "shared"],
        sourceRefs: {
          pytorch: ["elmanrnn.self-hidden_to_hidden-nn-linear-hidden_size-hidden_size-bias-false","elmanrnn.forward.recurrent_hidden-self-hidden_to_hidden-h"],
          jax: ["elmanrnn.__call__.hidden_to_hidden-nn-dense-self-hidden_size-use_bias-false-name-hidden_to","elmanrnn.__call__.recurrent_hidden-hidden_to_hidden-h"],
        },
        focusRef: {
          pytorch: "elmanrnn.self-hidden_to_hidden-nn-linear-hidden_size-hidden_size-bias-false",
          jax: "elmanrnn.__call__.hidden_to_hidden-nn-dense-self-hidden_size-use_bias-false-name-hidden_to",
        },
        includeChildRefs: false,
      },
      {
        id: `step.${index}.update`,
        label: "add + tanh",
        type: "StateUpdate",
        kind: "activation",
        badges: ["h_t"],
        sourceRefs: {
          pytorch: ["elmanrnn.forward.h-torch-tanh-hidden_sum","elmanrnn.forward.states-append-h"],
          jax: ["elmanrnn.__call__.hidden_sum-input_hidden-recurrent_hidden","elmanrnn.__call__.h-jnp-tanh-hidden_sum"],
        },
        focusRef: {
          pytorch: "elmanrnn.forward.h-torch-tanh-hidden_sum",
          jax: "elmanrnn.__call__.hidden_sum-input_hidden-recurrent_hidden",
        },
        includeChildRefs: false,
      },
      {
        id: `step.${index}.state`,
        label: "state history",
        type: "AppendHidden",
        kind: "recurrent",
        badges: ["store h_t"],
        sourceRefs: {
          pytorch: ["elmanrnn.forward.states-append-h","elmanrnn.forward.state_trace-torch-stack-states-dim-n"],
          jax: ["elmanrnn.__call__.h-jnp-tanh-hidden_sum","elmanrnn.__call__.states-append-h","elmanrnn.__call__.state_trace-jnp-stack-states-axis-n"],
        },
        focusRef: {
          pytorch: "elmanrnn.forward.states-append-h",
          jax: "elmanrnn.__call__.h-jnp-tanh-hidden_sum",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["grucell.forward.x_z-self-x_z-x","grucell.forward.h_z-self-h_z-h","grucell.forward.z_pre-x_z-h_z","grucell.forward.z-torch-sigmoid-z_pre","grucell.forward.x_r-self-x_r-x","grucell.forward.h_r-self-h_r-h","grucell.forward.r_pre-x_r-h_r","grucell.forward.r-torch-sigmoid-r_pre","grucell.forward.reset_h-r-h","grucell.forward.x_n-self-x_n-x","grucell.forward.h_n-self-h_n-reset_h","grucell.forward.n_pre-x_n-h_n","grucell.forward.n-torch-tanh-n_pre","grucell.forward.keep_h-z-h","grucell.forward.candidate_h-n-z-n","grucell.forward.h_next-candidate_h-keep_h","grucell.forward.return-h_next","grusequence.forward.for-t-in-range-step_count","grusequence.forward.current_input-x-t","grusequence.forward.h-self-cell-current_input-h","grusequence.forward.states-append-h"],
      jax: ["grucell.__call__.x_z-nn-dense-self-hidden_size-name-x_z-x","grucell.__call__.h_z-nn-dense-self-hidden_size-use_bias-false-name-h_z-h","grucell.__call__.z_pre-x_z-h_z","grucell.__call__.z-nn-sigmoid-z_pre","grucell.__call__.x_r-nn-dense-self-hidden_size-name-x_r-x","grucell.__call__.h_r-nn-dense-self-hidden_size-use_bias-false-name-h_r-h","grucell.__call__.r_pre-x_r-h_r","grucell.__call__.r-nn-sigmoid-r_pre","grucell.__call__.reset_h-r-h","grucell.__call__.x_n-nn-dense-self-hidden_size-name-x_n-x","grucell.__call__.h_n-nn-dense-self-hidden_size-use_bias-false-name-h_n-reset_h","grucell.__call__.n_pre-x_n-h_n","grucell.__call__.n-jnp-tanh-n_pre","grucell.__call__.keep_h-z-h","grucell.__call__.candidate_h-n-z-n","grucell.__call__.h_next-candidate_h-keep_h","grucell.__call__.return-h_next","grusequence.__call__.for-t-in-range-step_count","grusequence.__call__.current_input-x-t","grusequence.__call__.h-cell-current_input-h","grusequence.__call__.states-append-h"],
    },
    focusRef: {
      pytorch: "grucell.forward.x_z-self-x_z-x",
      jax: "grucell.__call__.x_z-nn-dense-self-hidden_size-name-x_z-x",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `step.${index}.update_gate`,
        label: "update gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["z_t"],
        sourceRefs: {
          pytorch: ["grucell.self-x_z-nn-linear-input_size-hidden_size","grucell.self-h_z-nn-linear-hidden_size-hidden_size-bias-false","grucell.forward.x_z-self-x_z-x","grucell.forward.h_z-self-h_z-h","grucell.forward.z_pre-x_z-h_z","grucell.forward.z-torch-sigmoid-z_pre"],
          jax: ["grucell.__call__.x_z-nn-dense-self-hidden_size-name-x_z-x","grucell.__call__.h_z-nn-dense-self-hidden_size-use_bias-false-name-h_z-h","grucell.__call__.z_pre-x_z-h_z","grucell.__call__.z-nn-sigmoid-z_pre"],
        },
        focusRef: {
          pytorch: "grucell.self-x_z-nn-linear-input_size-hidden_size",
          jax: "grucell.__call__.x_z-nn-dense-self-hidden_size-name-x_z-x",
        },
        includeChildRefs: false,
      },
      {
        id: `step.${index}.reset_gate`,
        label: "reset gate",
        type: "SigmoidGate",
        kind: "recurrent",
        badges: ["r_t"],
        sourceRefs: {
          pytorch: ["grucell.self-x_r-nn-linear-input_size-hidden_size","grucell.self-h_r-nn-linear-hidden_size-hidden_size-bias-false","grucell.forward.x_r-self-x_r-x","grucell.forward.h_r-self-h_r-h","grucell.forward.r_pre-x_r-h_r","grucell.forward.r-torch-sigmoid-r_pre"],
          jax: ["grucell.__call__.x_r-nn-dense-self-hidden_size-name-x_r-x","grucell.__call__.h_r-nn-dense-self-hidden_size-use_bias-false-name-h_r-h","grucell.__call__.r_pre-x_r-h_r","grucell.__call__.r-nn-sigmoid-r_pre"],
        },
        focusRef: {
          pytorch: "grucell.self-x_r-nn-linear-input_size-hidden_size",
          jax: "grucell.__call__.x_r-nn-dense-self-hidden_size-name-x_r-x",
        },
        includeChildRefs: false,
      },
      {
        id: `step.${index}.candidate`,
        label: "candidate",
        type: "TanhState",
        kind: "activation",
        badges: ["n_t"],
        sourceRefs: {
          pytorch: ["grucell.self-x_n-nn-linear-input_size-hidden_size","grucell.self-h_n-nn-linear-hidden_size-hidden_size-bias-false","grucell.forward.reset_h-r-h","grucell.forward.x_n-self-x_n-x","grucell.forward.h_n-self-h_n-reset_h","grucell.forward.n_pre-x_n-h_n","grucell.forward.n-torch-tanh-n_pre"],
          jax: ["grucell.__call__.reset_h-r-h","grucell.__call__.x_n-nn-dense-self-hidden_size-name-x_n-x","grucell.__call__.h_n-nn-dense-self-hidden_size-use_bias-false-name-h_n-reset_h","grucell.__call__.n_pre-x_n-h_n","grucell.__call__.n-jnp-tanh-n_pre"],
        },
        focusRef: {
          pytorch: "grucell.self-x_n-nn-linear-input_size-hidden_size",
          jax: "grucell.__call__.reset_h-r-h",
        },
        includeChildRefs: false,
      },
      {
        id: `step.${index}.mix`,
        label: "state mix",
        type: "GatedInterpolation",
        kind: "recurrent",
        badges: ["h_t"],
        sourceRefs: {
          pytorch: ["grucell.forward.keep_h-z-h","grucell.forward.candidate_h-n-z-n","grucell.forward.h_next-candidate_h-keep_h","grucell.forward.return-h_next"],
          jax: ["grucell.__call__.keep_h-z-h","grucell.__call__.candidate_h-n-z-n","grucell.__call__.h_next-candidate_h-keep_h","grucell.__call__.return-h_next"],
        },
        focusRef: {
          pytorch: "grucell.forward.keep_h-z-h",
          jax: "grucell.__call__.keep_h-z-h",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["seqnseqencoder.forward.current_embedding-embeddings-t","seqnseqencoder.forward.state-self-cell-current_embedding-h-c","seqnseqencoder.forward.h-state-n","seqnseqencoder.forward.c-state-n","seqnseqencoder.forward.encoder_states-append-h"],
      jax: ["seqnseqencoder.__call__.current_embedding-embeddings-t","seqnseqencoder.__call__.state-cell-current_embedding-h-c","seqnseqencoder.__call__.h-state-n","seqnseqencoder.__call__.c-state-n","seqnseqencoder.__call__.encoder_states-append-h"],
    },
    focusRef: {
      pytorch: "seqnseqencoder.forward.current_embedding-embeddings-t",
      jax: "seqnseqencoder.__call__.current_embedding-embeddings-t",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `encoder.step.${index}.embedding`,
        label: "token embedding",
        type: "EmbeddingLookup",
        kind: "embedding",
        badges: ["128 dim"],
        sourceRefs: {
          pytorch: ["seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions","seqnseqencoder.forward.embeddings-self-embedding-reversed_ids","seqnseqencoder.forward.current_embedding-embeddings-t"],
          jax: ["seqnseqencoder.__call__.reversed_ids-source_ids-source_order","seqnseqencoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-source_embe","seqnseqencoder.__call__.current_embedding-embeddings-t"],
        },
        focusRef: {
          pytorch: "seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions",
          jax: "seqnseqencoder.__call__.reversed_ids-source_ids-source_order",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.step.${index}.lstm_gates`,
        label: "lstm gates",
        type: "Input/Forget/Cell/Output",
        kind: "recurrent",
        badges: ["i", "f", "g", "o"],
        sourceRefs: {
          pytorch: ["lstmcell.forward.x_i-self-x_i-x","lstmcell.forward.h_i-self-h_i-h","lstmcell.forward.i_pre-x_i-h_i","lstmcell.forward.i-torch-sigmoid-i_pre","lstmcell.forward.x_f-self-x_f-x","lstmcell.forward.h_f-self-h_f-h","lstmcell.forward.f_pre-x_f-h_f","lstmcell.forward.f-torch-sigmoid-f_pre","lstmcell.forward.x_g-self-x_g-x","lstmcell.forward.h_g-self-h_g-h","lstmcell.forward.g_pre-x_g-h_g","lstmcell.forward.g-torch-tanh-g_pre","lstmcell.forward.x_o-self-x_o-x","lstmcell.forward.h_o-self-h_o-h","lstmcell.forward.o_pre-x_o-h_o","lstmcell.forward.o-torch-sigmoid-o_pre","seqnseqencoder.forward.state-self-cell-current_embedding-h-c"],
          jax: ["lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h","lstmcell.__call__.i_pre-x_i-h_i","lstmcell.__call__.i-nn-sigmoid-i_pre","lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h","lstmcell.__call__.f_pre-x_f-h_f","lstmcell.__call__.f-nn-sigmoid-f_pre","lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h","lstmcell.__call__.g_pre-x_g-h_g","lstmcell.__call__.g-jnp-tanh-g_pre","lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h","lstmcell.__call__.o_pre-x_o-h_o","lstmcell.__call__.o-nn-sigmoid-o_pre","seqnseqencoder.__call__.state-cell-current_embedding-h-c"],
        },
        focusRef: {
          pytorch: "lstmcell.forward.x_i-self-x_i-x",
          jax: "lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.step.${index}.state`,
        label: "state update",
        type: "ContextState",
        kind: "recurrent",
        badges: ["h_t", "c_t"],
        sourceRefs: {
          pytorch: ["lstmcell.forward.forget_c-f-c","lstmcell.forward.write_c-i-g","lstmcell.forward.c_next-forget_c-write_c","lstmcell.forward.c_readout-torch-tanh-c_next","lstmcell.forward.h_next-o-c_readout","lstmcell.forward.next_state-h_next-c_next","seqnseqencoder.forward.h-state-n","seqnseqencoder.forward.c-state-n","seqnseqencoder.forward.encoder_states-append-h"],
          jax: ["lstmcell.__call__.forget_c-f-c","lstmcell.__call__.write_c-i-g","lstmcell.__call__.c_next-forget_c-write_c","lstmcell.__call__.c_readout-jnp-tanh-c_next","lstmcell.__call__.h_next-o-c_readout","lstmcell.__call__.next_state-h_next-c_next","seqnseqencoder.__call__.h-state-n","seqnseqencoder.__call__.c-state-n","seqnseqencoder.__call__.encoder_states-append-h"],
        },
        focusRef: {
          pytorch: "lstmcell.forward.forget_c-f-c",
          jax: "lstmcell.__call__.forget_c-f-c",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["seqnseqdecoder.forward.current_embedding-embeddings-t","seqnseqdecoder.forward.state-self-cell-current_embedding-h-c","seqnseqdecoder.forward.h-state-n","seqnseqdecoder.forward.c-state-n","seqnseqdecoder.forward.logits-self-output_projection-h","seqnseqdecoder.forward.logits_per_step-append-logits","seqnseqdecoder.forward.decoder_states-append-h"],
      jax: ["seqnseqdecoder.__call__.current_embedding-embeddings-t","seqnseqdecoder.__call__.state-cell-current_embedding-h-c","seqnseqdecoder.__call__.h-state-n","seqnseqdecoder.__call__.c-state-n","seqnseqdecoder.__call__.logits-output_projection-h","seqnseqdecoder.__call__.logits_per_step-append-logits","seqnseqdecoder.__call__.decoder_states-append-h"],
    },
    focusRef: {
      pytorch: "seqnseqdecoder.forward.current_embedding-embeddings-t",
      jax: "seqnseqdecoder.__call__.current_embedding-embeddings-t",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `decoder.step.${index}.embedding`,
        label: "target embedding",
        type: "EmbeddingLookup",
        kind: "embedding",
        badges: ["shifted target"],
        sourceRefs: {
          pytorch: ["seqnseqdecoder.forward.embeddings-self-embedding-decoder_input_ids","seqnseqdecoder.forward.current_embedding-embeddings-t"],
          jax: ["seqnseqdecoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-target_embe","seqnseqdecoder.__call__.current_embedding-embeddings-t"],
        },
        focusRef: {
          pytorch: "seqnseqdecoder.forward.embeddings-self-embedding-decoder_input_ids",
          jax: "seqnseqdecoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-target_embe",
        },
        includeChildRefs: false,
      },
      {
        id: `decoder.step.${index}.lstm_gates`,
        label: "lstm gates",
        type: "Input/Forget/Cell/Output",
        kind: "recurrent",
        badges: ["i", "f", "g", "o"],
        sourceRefs: {
          pytorch: ["lstmcell.forward.x_i-self-x_i-x","lstmcell.forward.h_i-self-h_i-h","lstmcell.forward.i_pre-x_i-h_i","lstmcell.forward.i-torch-sigmoid-i_pre","lstmcell.forward.x_f-self-x_f-x","lstmcell.forward.h_f-self-h_f-h","lstmcell.forward.f_pre-x_f-h_f","lstmcell.forward.f-torch-sigmoid-f_pre","lstmcell.forward.x_g-self-x_g-x","lstmcell.forward.h_g-self-h_g-h","lstmcell.forward.g_pre-x_g-h_g","lstmcell.forward.g-torch-tanh-g_pre","lstmcell.forward.x_o-self-x_o-x","lstmcell.forward.h_o-self-h_o-h","lstmcell.forward.o_pre-x_o-h_o","lstmcell.forward.o-torch-sigmoid-o_pre","seqnseqdecoder.forward.state-self-cell-current_embedding-h-c"],
          jax: ["lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h","lstmcell.__call__.i_pre-x_i-h_i","lstmcell.__call__.i-nn-sigmoid-i_pre","lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h","lstmcell.__call__.f_pre-x_f-h_f","lstmcell.__call__.f-nn-sigmoid-f_pre","lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h","lstmcell.__call__.g_pre-x_g-h_g","lstmcell.__call__.g-jnp-tanh-g_pre","lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h","lstmcell.__call__.o_pre-x_o-h_o","lstmcell.__call__.o-nn-sigmoid-o_pre","seqnseqdecoder.__call__.state-cell-current_embedding-h-c"],
        },
        focusRef: {
          pytorch: "lstmcell.forward.x_i-self-x_i-x",
          jax: "lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x",
        },
        includeChildRefs: false,
      },
      {
        id: `decoder.step.${index}.projection`,
        label: "vocab logits",
        type: "Linear",
        kind: "linear",
        badges: ["256->vocab"],
        sourceRefs: {
          pytorch: ["seqnseqdecoder.self-output_projection-nn-linear-hidden_size-vocab_size","seqnseqdecoder.forward.logits-self-output_projection-h","seqnseqdecoder.forward.logits_per_step-append-logits"],
          jax: ["seqnseqdecoder.__call__.output_projection-nn-dense-self-vocab_size-name-output_projection","seqnseqdecoder.__call__.logits-output_projection-h","seqnseqdecoder.__call__.logits_per_step-append-logits"],
        },
        focusRef: {
          pytorch: "seqnseqdecoder.self-output_projection-nn-linear-hidden_size-vocab_size",
          jax: "seqnseqdecoder.__call__.output_projection-nn-dense-self-vocab_size-name-output_projection",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["multiheadattention.self","multiheadattention.self-head_dim-d_model-nhead","multiheadattention.self-v_proj-nn-linear-d_model-d_model","multiheadattention.self-out_proj-nn-linear-d_model-d_model","multiheadattention.def-forward-self-query-key-value-attn_mask-none","multiheadattention.forward.batch_size-query-size-n","multiheadattention.forward.v-self-v_proj-value","multiheadattention.forward.k-k-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.k-k-transpose-n-n","multiheadattention.forward.v-v-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.v-v-transpose-n-n","multiheadattention.forward.scale-self-head_dim-n","multiheadattention.forward.attn_scores-scores-scale","multiheadattention.forward.if-attn_mask-is-not-none","multiheadattention.forward.mask-attn_mask-none-none","multiheadattention.forward.context-context-contiguous","multiheadattention.forward.merged-context-view-batch_size-query_steps-self-nhead-self-head_dim","multiheadattention.forward.return-out","class-encoderlayer-nn-module","decoderlayer.nhead-n","decoderlayer.nn-linear-d_model-d_ff","decoderlayer.nn-relu","decoderlayer.nn-linear-d_ff-d_model","decoderlayer.code.4","decoderlayer.self-normn-nn-layernorm-d_model","decoderlayer.self-normn-nn-layernorm-d_model.2","decoderlayer.self-normn-nn-layernorm-d_model.3","decoderlayer.def-forward-self-x-memory-tgt_mask-none","decoderlayer.forward.masked-self-self_attn-x-x-x-tgt_mask","decoderlayer.forward.x-self-normn-masked_residual","decoderlayer.forward.cross-self-cross_attn-x-memory-memory","decoderlayer.forward.cross_residual-x-cross","decoderlayer.forward.ffn-self-ffn-x","decoderlayer.forward.ffn_residual-x-ffn","class-transformer-nn-module","transformer.def-__init__","transformer.self","transformer.forward.src_embeddings-self-src_embed-src_ids"],
      jax: ["class-decoderlayer-nn-module","decoderlayer.d_model-int-n","decoderlayer.nhead-int-n","decoderlayer.d_ff-int-n","decoderlayer.nn-compact","decoderlayer.def-__call__-self-x-memory-mask","decoderlayer.__call__.masked-multiheadattention-self-d_model-self-nhead-x-x-x-mask","decoderlayer.__call__.masked_residual-x-masked","decoderlayer.__call__.x-nn-layernorm-masked_residual","decoderlayer.__call__.cross-multiheadattention-self-d_model-self-nhead-x-memory-memory","decoderlayer.__call__.cross_residual-x-cross","decoderlayer.__call__.x-nn-layernorm-cross_residual","decoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model","decoderlayer.__call__.ffn-nn-sequential-ffn_layers-x","decoderlayer.__call__.ffn_residual-x-ffn","decoderlayer.__call__.out-nn-layernorm-ffn_residual","decoderlayer.__call__.return-out"],
    },
    focusRef: {
      pytorch: "multiheadattention.self",
      jax: "class-decoderlayer-nn-module",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `decoder.${index}.masked_self_attn`,
        label: "masked self_attn",
        type: "CausalMultiHeadAttention",
        kind: "attention",
        badges: ["8 heads", "causal"],
        sourceRefs: {
          pytorch: ["multiheadattention.self-v_proj-nn-linear-d_model-d_model","multiheadattention.self-out_proj-nn-linear-d_model-d_model","multiheadattention.def-forward-self-query-key-value-attn_mask-none","multiheadattention.forward.v-self-v_proj-value","multiheadattention.forward.k-k-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.k-k-transpose-n-n","multiheadattention.forward.v-v-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.v-v-transpose-n-n","multiheadattention.forward.scale-self-head_dim-n","multiheadattention.forward.attn_scores-scores-scale","multiheadattention.forward.if-attn_mask-is-not-none","multiheadattention.forward.mask-attn_mask-none-none","multiheadattention.forward.attn_weights-torch-softmax-attn_scores-dim-n","multiheadattention.forward.context-context-contiguous","multiheadattention.forward.merged-context-view-batch_size-query_steps-self-nhead-self-head_dim","multiheadattention.forward.out-self-out_proj-merged","multiheadattention.forward.return-out","decoderlayer.nn-relu","for-step-in-range-n","optimizer-zero_grad","logits-model-src_ids-tgt_ids-tgt_mask"],
          jax: ["decoderlayer.__call__.masked-multiheadattention-self-d_model-self-nhead-x-x-x-mask","decoderlayer.__call__.masked_residual-x-masked"],
        },
        focusRef: {
          pytorch: "multiheadattention.self-v_proj-nn-linear-d_model-d_model",
          jax: "decoderlayer.__call__.masked-multiheadattention-self-d_model-self-nhead-x-x-x-mask",
        },
        includeChildRefs: false,
      },
      {
        id: `decoder.${index}.norm1`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        sourceRefs: {
          pytorch: ["decoderlayer.forward.x-self-normn-masked_residual"],
          jax: ["decoderlayer.__call__.masked_residual-x-masked","decoderlayer.__call__.x-nn-layernorm-masked_residual"],
        },
        focusRef: {
          pytorch: "decoderlayer.forward.x-self-normn-masked_residual",
          jax: "decoderlayer.__call__.masked_residual-x-masked",
        },
        includeChildRefs: false,
      },
      {
        id: `decoder.${index}.cross_attn`,
        label: "cross_attn",
        type: "EncoderDecoderAttention",
        kind: "attention",
        badges: ["Q=decoder", "K,V=encoder"],
        sourceRefs: {
          pytorch: ["multiheadattention.self-v_proj-nn-linear-d_model-d_model","multiheadattention.self-out_proj-nn-linear-d_model-d_model","multiheadattention.def-forward-self-query-key-value-attn_mask-none","multiheadattention.forward.v-self-v_proj-value","multiheadattention.forward.k-k-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.k-k-transpose-n-n","multiheadattention.forward.v-v-view-batch_size-key_steps-self-nhead-self-head_dim","multiheadattention.forward.v-v-transpose-n-n","multiheadattention.forward.scale-self-head_dim-n","multiheadattention.forward.attn_scores-scores-scale","multiheadattention.forward.if-attn_mask-is-not-none","multiheadattention.forward.mask-attn_mask-none-none","multiheadattention.forward.context-context-contiguous","multiheadattention.forward.merged-context-view-batch_size-query_steps-self-nhead-self-head_dim","multiheadattention.forward.return-out","decoderlayer.nn-linear-d_ff-d_model"],
          jax: ["decoderlayer.__call__.cross-multiheadattention-self-d_model-self-nhead-x-memory-memory","decoderlayer.__call__.cross_residual-x-cross"],
        },
        focusRef: {
          pytorch: "multiheadattention.self-v_proj-nn-linear-d_model-d_model",
          jax: "decoderlayer.__call__.cross-multiheadattention-self-d_model-self-nhead-x-memory-memory",
        },
        includeChildRefs: false,
      },
      {
        id: `decoder.${index}.norm2`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        sourceRefs: {
          pytorch: ["decoderlayer.forward.cross_residual-x-cross","transformer.decoder.norm2"],
          jax: ["decoderlayer.__call__.cross_residual-x-cross","decoderlayer.__call__.x-nn-layernorm-cross_residual"],
        },
        focusRef: {
          pytorch: "decoderlayer.forward.cross_residual-x-cross",
          jax: "decoderlayer.__call__.cross_residual-x-cross",
        },
        includeChildRefs: false,
      },
      {
        id: `decoder.${index}.ffn`,
        label: "ffn",
        type: "FeedForward",
        kind: "mlp",
        badges: ["512->2048->512"],
        sourceRefs: {
          pytorch: ["decoderlayer.nn-linear-d_model-d_ff","decoderlayer.nn-relu","decoderlayer.nn-linear-d_ff-d_model","decoderlayer.forward.ffn-self-ffn-x"],
          jax: ["decoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model","decoderlayer.__call__.ffn-nn-sequential-ffn_layers-x"],
        },
        focusRef: {
          pytorch: "decoderlayer.nn-linear-d_model-d_ff",
          jax: "decoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model",
        },
        includeChildRefs: false,
      },
      {
        id: `decoder.${index}.norm3`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        sourceRefs: {
          pytorch: ["decoderlayer.forward.ffn_residual-x-ffn","transformer.decoder.norm3"],
          jax: ["decoderlayer.__call__.ffn_residual-x-ffn","decoderlayer.__call__.out-nn-layernorm-ffn_residual"],
        },
        focusRef: {
          pytorch: "decoderlayer.forward.ffn_residual-x-ffn",
          jax: "decoderlayer.__call__.ffn_residual-x-ffn",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["class-bertlayer-nn-module","bertlayer.def-__init__","bertlayer.self","bertlayer.hidden_size-n","bertlayer.num_heads-n","bertlayer.intermediate_size-n","bertlayer.code","bertlayer.super-__init__","bertlayer.self-self_attn-bertselfattention-hidden_size-num_heads","bertlayer.self-attn_norm-nn-layernorm-hidden_size","bertlayer.self-ffn-nn-sequential","bertlayer.nn-linear-hidden_size-intermediate_size","bertlayer.nn-gelu","bertlayer.nn-linear-intermediate_size-hidden_size","bertlayer.code.4","bertlayer.self-ffn_norm-nn-layernorm-hidden_size","bertlayer.self-dropout-nn-dropout-n","bertlayer.def-forward-self-x-attention_mask-none","bertlayer.forward.attn-self-self_attn-x-attention_mask","bertlayer.forward.attn-self-dropout-attn","bertlayer.forward.attn_residual-x-attn","bertlayer.forward.x-self-attn_norm-attn_residual","bertlayer.forward.ffn-self-ffn-x","bertlayer.forward.ffn-self-dropout-ffn","bertlayer.forward.ffn_residual-x-ffn","bertlayer.forward.out-self-ffn_norm-ffn_residual","bertlayer.forward.return-out"],
      jax: ["class-bertlayer-nn-module","bertlayer.hidden_size-int-n","bertlayer.num_heads-int-n","bertlayer.intermediate_size-int-n","bertlayer.nn-compact","bertlayer.def-__call__-self-x-attention_mask-none-train-false","bertlayer.__call__.attn-bertselfattention-self-hidden_size-self-num_heads-x-attention_mask","bertlayer.__call__.attn-nn-dropout-n-deterministic-not-train-attn","bertlayer.__call__.attn_residual-x-attn","bertlayer.__call__.x-nn-layernorm-name-attention_norm-attn_residual","bertlayer.__call__.ffn-nn-dense-self-intermediate_size-name-intermediate-x","bertlayer.__call__.ffn-nn-gelu-ffn","bertlayer.__call__.ffn-nn-dense-self-hidden_size-name-output_dense-ffn","bertlayer.__call__.ffn-nn-dropout-n-deterministic-not-train-ffn","bertlayer.__call__.ffn_residual-x-ffn","bertlayer.__call__.out-nn-layernorm-name-output_norm-ffn_residual","bertlayer.__call__.return-out"],
    },
    focusRef: {
      pytorch: "class-bertlayer-nn-module",
      jax: "class-bertlayer-nn-module",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `encoder.layer.${index}.self_attn`,
        label: "self_attn",
        type: "BidirectionalSelfAttention",
        kind: "attention",
        badges: ["12 heads", "768"],
        sourceRefs: {
          pytorch: ["class-bertselfattention-nn-module","bertselfattention.self-num_heads-num_heads","bertselfattention.self-head_dim-hidden_size-num_heads","bertselfattention.self-q_proj-nn-linear-hidden_size-hidden_size","bertselfattention.self-k_proj-nn-linear-hidden_size-hidden_size","bertselfattention.self-v_proj-nn-linear-hidden_size-hidden_size","bertselfattention.self-out_proj-nn-linear-hidden_size-hidden_size","bertselfattention.def-forward-self-x-attention_mask-none","bertselfattention.forward.batch_size-x-size-n","bertselfattention.forward.steps-x-size-n","bertselfattention.forward.q-self-q_proj-x","bertselfattention.forward.k-self-k_proj-x","bertselfattention.forward.v-self-v_proj-x","bertselfattention.forward.q-q-view-batch_size-steps-self-num_heads-self-head_dim","bertselfattention.forward.q-q-transpose-n-n","bertselfattention.forward.k-k-view-batch_size-steps-self-num_heads-self-head_dim","bertselfattention.forward.k-k-transpose-n-n","bertselfattention.forward.v-v-view-batch_size-steps-self-num_heads-self-head_dim","bertselfattention.forward.v-v-transpose-n-n","bertselfattention.forward.key_transpose-k-transpose-n-n","bertselfattention.forward.scores-q-key_transpose","bertselfattention.forward.scale-self-head_dim-n","bertselfattention.forward.attn_scores-scores-scale","bertselfattention.forward.if-attention_mask-is-not-none","bertselfattention.forward.mask-attention_mask-none-none","bertselfattention.forward.attn_scores-attn_scores-masked_fill-mask-nen","bertselfattention.forward.attn_weights-torch-softmax-attn_scores-dim-n","bertselfattention.forward.context-attn_weights-v","bertselfattention.forward.context-context-transpose-n-n","bertselfattention.forward.context-context-contiguous","bertselfattention.forward.merged-context-view-batch_size-steps-self-num_heads-self-head_dim","bertselfattention.forward.out-self-out_proj-merged","bertselfattention.forward.return-out","bertlayer.self-self_attn-bertselfattention-hidden_size-num_heads","bertlayer.forward.attn-self-self_attn-x-attention_mask"],
          jax: ["class-bertselfattention-nn-module","bertselfattention.hidden_size-int-n","bertselfattention.num_heads-int-n","bertselfattention.nn-compact","bertselfattention.def-__call__-self-x-attention_mask-none","bertselfattention.__call__.batch_size-x-shape-n","bertselfattention.__call__.steps-x-shape-n","bertselfattention.__call__.head_dim-self-hidden_size-self-num_heads","bertselfattention.__call__.q-nn-dense-self-hidden_size-name-q_proj-x","bertselfattention.__call__.k-nn-dense-self-hidden_size-name-k_proj-x","bertselfattention.__call__.v-nn-dense-self-hidden_size-name-v_proj-x","bertselfattention.__call__.head_shape-batch_size-steps-self-num_heads-head_dim","bertselfattention.__call__.q-q-reshape-head_shape","bertselfattention.__call__.q-jnp-transpose-q-n-n-n-n","bertselfattention.__call__.k-k-reshape-head_shape","bertselfattention.__call__.k-jnp-transpose-k-n-n-n-n","bertselfattention.__call__.v-v-reshape-head_shape","bertselfattention.__call__.v-jnp-transpose-v-n-n-n-n","bertselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n","bertselfattention.__call__.scores-q-key_transpose","bertselfattention.__call__.scale-head_dim-n","bertselfattention.__call__.attn_scores-scores-scale","bertselfattention.__call__.if-attention_mask-is-not-none","bertselfattention.__call__.attn_scores-jnp-where-attention_mask-attn_scores-jnp-inf","bertselfattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n","bertselfattention.__call__.context-attn_weights-v","bertselfattention.__call__.context-jnp-transpose-context-n-n-n-n","bertselfattention.__call__.merged_shape-batch_size-steps-self-hidden_size","bertselfattention.__call__.merged-context-reshape-merged_shape","bertselfattention.__call__.out-nn-dense-self-hidden_size-name-out_proj-merged","bertselfattention.__call__.return-out","bertlayer.__call__.attn-bertselfattention-self-hidden_size-self-num_heads-x-attention_mask"],
        },
        focusRef: {
          pytorch: "class-bertselfattention-nn-module",
          jax: "class-bertselfattention-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.layer.${index}.attn_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        sourceRefs: {
          pytorch: ["bertlayer.self-attn_norm-nn-layernorm-hidden_size","bertlayer.forward.attn_residual-x-attn","bertlayer.forward.x-self-attn_norm-attn_residual"],
          jax: ["bertlayer.__call__.attn_residual-x-attn","bertlayer.__call__.x-nn-layernorm-name-attention_norm-attn_residual"],
        },
        focusRef: {
          pytorch: "bertlayer.self-attn_norm-nn-layernorm-hidden_size",
          jax: "bertlayer.__call__.attn_residual-x-attn",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.layer.${index}.intermediate`,
        label: "intermediate",
        type: "Dense + GELU",
        kind: "mlp",
        badges: ["768->3072"],
        sourceRefs: {
          pytorch: ["bertlayer.self-ffn-nn-sequential","bertlayer.nn-linear-hidden_size-intermediate_size","bertlayer.nn-gelu","bertlayer.forward.ffn-self-ffn-x"],
          jax: ["bertlayer.__call__.ffn-nn-dense-self-intermediate_size-name-intermediate-x","bertlayer.__call__.ffn-nn-gelu-ffn"],
        },
        focusRef: {
          pytorch: "bertlayer.self-ffn-nn-sequential",
          jax: "bertlayer.__call__.ffn-nn-dense-self-intermediate_size-name-intermediate-x",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.layer.${index}.output`,
        label: "output",
        type: "Dense",
        kind: "mlp",
        badges: ["3072->768"],
        sourceRefs: {
          pytorch: ["bertlayer.nn-linear-intermediate_size-hidden_size","bertlayer.forward.ffn-self-ffn-x"],
          jax: ["bertlayer.__call__.ffn-nn-dense-self-hidden_size-name-output_dense-ffn"],
        },
        focusRef: {
          pytorch: "bertlayer.nn-linear-intermediate_size-hidden_size",
          jax: "bertlayer.__call__.ffn-nn-dense-self-hidden_size-name-output_dense-ffn",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.layer.${index}.output_norm`,
        label: "add + norm",
        type: "ResidualLayerNorm",
        kind: "residual",
        sourceRefs: {
          pytorch: ["bertlayer.self-ffn_norm-nn-layernorm-hidden_size","bertlayer.forward.ffn_residual-x-ffn","bertlayer.forward.out-self-ffn_norm-ffn_residual"],
          jax: ["bertlayer.__call__.ffn_residual-x-ffn","bertlayer.__call__.out-nn-layernorm-name-output_norm-ffn_residual"],
        },
        focusRef: {
          pytorch: "bertlayer.self-ffn_norm-nn-layernorm-hidden_size",
          jax: "bertlayer.__call__.ffn_residual-x-ffn",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["class-block-nn-module","block.def-__init__-self","block.__init__.super-__init__","block.__init__.self-ln_n-nn-layernorm-n","block.__init__.self-attn-causalselfattention","block.__init__.self-ln_n-nn-layernorm-n.2","block.__init__.self-mlp-nn-sequential","block.__init__.nn-linear-n-n","block.__init__.nn-gelu","block.__init__.nn-linear-n-n.2","block.__init__.code.3","block.def-forward-self-x-mask","block.forward.attn_input-self-ln_n-x","block.forward.attn-self-attn-attn_input-mask","block.forward.x-x-attn","block.forward.mlp_input-self-ln_n-x","block.forward.mlp_out-self-mlp-mlp_input","block.forward.x-x-mlp_out","block.forward.return-x","gptnsmall.self-blocks-nn-modulelist-block-for-_-in-range-n","gptnsmall.forward.for-block-in-self-blocks","gptnsmall.forward.x-block-x-mask"],
      jax: ["class-block-nn-module","block.nn-compact","block.def-__call__-self-x-mask","block.__call__.attn_input-nn-layernorm-name-ln_n-x","block.__call__.attn-causalselfattention-attn_input-mask","block.__call__.x-x-attn","block.__call__.mlp_input-nn-layernorm-name-ln_n-x","block.__call__.mlp_out-mlp-mlp_input","block.__call__.x-x-mlp_out","block.__call__.return-x","gptnsmall.__call__.for-_-in-range-self-n_layer","gptnsmall.__call__.x-block-x-mask"],
    },
    focusRef: {
      pytorch: "class-block-nn-module",
      jax: "class-block-nn-module",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        sourceRefs: {
          pytorch: ["block.__init__.self-ln_n-nn-layernorm-n","block.forward.attn_input-self-ln_n-x"],
          jax: ["block.__call__.attn_input-nn-layernorm-name-ln_n-x"],
        },
        focusRef: {
          pytorch: "block.__init__.self-ln_n-nn-layernorm-n",
          jax: "block.__call__.attn_input-nn-layernorm-name-ln_n-x",
        },
        includeChildRefs: false,
      },
      {
        id: `block.${index}.attn`,
        label: "attn",
        type: "CausalSelfAttention",
        kind: "attention",
        summary: "12 heads",
        sourceRefs: {
          pytorch: ["class-causalselfattention-nn-module","causalselfattention.def-__init__","causalselfattention.self","causalselfattention.n_embd-n","causalselfattention.n_head-n","causalselfattention.code","causalselfattention.super-__init__","causalselfattention.self-n_head-n_head","causalselfattention.def-forward-self-x-mask","causalselfattention.forward.return-out","block.__init__.self-attn-causalselfattention","block.forward.attn-self-attn-attn_input-mask"],
          jax: ["class-causalselfattention-nn-module","causalselfattention.n_embd-int-n","causalselfattention.n_head-int-n","causalselfattention.nn-compact","causalselfattention.def-__call__-self-x-mask","causalselfattention.__call__.qkv_dim-n-channel_count","causalselfattention.__call__.return-out","block.__call__.attn-causalselfattention-attn_input-mask"],
        },
        focusRef: {
          pytorch: "class-causalselfattention-nn-module",
          jax: "class-causalselfattention-nn-module",
        },
        includeChildRefs: true,
        lazyChildren: () => [
          {
            id: `block.${index}.attn.c_attn`,
            label: "c_attn",
            type: "QKV Projection",
            kind: "attention",
            badges: ["768->2304"],
            sourceRefs: {
              pytorch: ["causalselfattention.self-c_attn-nn-linear-n_embd-n-n_embd","causalselfattention.forward.qkv-self-c_attn-x","causalselfattention.forward.q-k-v-qkv-split-channel_count-dim-n"],
              jax: ["causalselfattention.__call__.qkv-nn-dense-qkv_dim-name-c_attn-x","causalselfattention.__call__.q-k-v-jnp-split-qkv-n-axis-n"],
            },
            focusRef: {
              pytorch: "causalselfattention.self-c_attn-nn-linear-n_embd-n-n_embd",
              jax: "causalselfattention.__call__.qkv-nn-dense-qkv_dim-name-c_attn-x",
            },
            includeChildRefs: false,
          },
          {
            id: `block.${index}.attn.heads`,
            label: "heads",
            type: "Head grid",
            kind: "group",
            summary: "12 x dim 64",
            sourceRefs: {
              pytorch: [],
              jax: [],
            },
            focusRef: {
              pytorch: "causalselfattention.forward.batch_size-step_count-channel_count-x-shape",
              jax: "causalselfattention.__call__.batch_size-step_count-channel_count-x-shape",
            },
            includeChildRefs: true,
            lazyChildren: () =>
              Array.from({ length: 12 }, (_, headIndex) => ({
                id: `block.${index}.attn.head.${headIndex}`,
                label: `head.${headIndex}`,
                type: "AttentionHead",
                kind: "head" as NodeKind,
                badges: ["q,k,v", "dim 64"],
                sourceRefs: {
                  pytorch: ["causalselfattention.forward.batch_size-step_count-channel_count-x-shape","causalselfattention.forward.head_dim-channel_count-self-n_head","causalselfattention.forward.q-q-view-batch_size-step_count-self-n_head-head_dim","causalselfattention.forward.q-q-transpose-n-n","causalselfattention.forward.k-k-view-batch_size-step_count-self-n_head-head_dim","causalselfattention.forward.k-k-transpose-n-n","causalselfattention.forward.v-v-view-batch_size-step_count-self-n_head-head_dim","causalselfattention.forward.v-v-transpose-n-n","causalselfattention.forward.key_transpose-k-transpose-n-n","causalselfattention.forward.scores-q-key_transpose","causalselfattention.forward.scale-k-size-n-n","causalselfattention.forward.att-scores-scale","causalselfattention.forward.mask_window-mask-step_count-step_count","causalselfattention.forward.att-att-masked_fill-mask_window-n-float-inf","causalselfattention.forward.weights-f-softmax-att-dim-n","causalselfattention.forward.y-weights-v"],
                  jax: ["causalselfattention.__call__.batch_size-step_count-channel_count-x-shape","causalselfattention.__call__.head_dim-channel_count-self-n_head","causalselfattention.__call__.q-q-reshape-batch_size-step_count-self-n_head-head_dim","causalselfattention.__call__.q-q-transpose-n-n-n-n","causalselfattention.__call__.k-k-reshape-batch_size-step_count-self-n_head-head_dim","causalselfattention.__call__.k-k-transpose-n-n-n-n","causalselfattention.__call__.v-v-reshape-batch_size-step_count-self-n_head-head_dim","causalselfattention.__call__.v-v-transpose-n-n-n-n","causalselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n","causalselfattention.__call__.scores-q-key_transpose","causalselfattention.__call__.scale-k-shape-n-n","causalselfattention.__call__.att-scores-scale","causalselfattention.__call__.mask_window-mask-step_count-step_count","causalselfattention.__call__.att-jnp-where-mask_window-n-jnp-inf-att","causalselfattention.__call__.weights-nn-softmax-att-axis-n","causalselfattention.__call__.y-weights-v"],
                },
                focusRef: {
                  pytorch: "causalselfattention.forward.batch_size-step_count-channel_count-x-shape",
                  jax: "causalselfattention.__call__.batch_size-step_count-channel_count-x-shape",
                },
                includeChildRefs: false,
              })),
          },
          {
            id: `block.${index}.attn.merge`,
            label: "merge",
            type: "Concat heads",
            kind: "attention",
            badges: ["12 x 64 -> 768"],
            sourceRefs: {
              pytorch: ["causalselfattention.forward.y-weights-v","causalselfattention.forward.y-y-transpose-n-n","causalselfattention.forward.y-y-contiguous","causalselfattention.forward.y-y-view-batch_size-step_count-channel_count","causalselfattention.forward.out-self-c_proj-y"],
              jax: ["causalselfattention.__call__.y-weights-v","causalselfattention.__call__.y-y-transpose-n-n-n-n","causalselfattention.__call__.y-y-reshape-batch_size-step_count-channel_count","causalselfattention.__call__.out-nn-dense-channel_count-name-c_proj-y"],
            },
            focusRef: {
              pytorch: "causalselfattention.forward.y-weights-v",
              jax: "causalselfattention.__call__.y-weights-v",
            },
            includeChildRefs: false,
          },
          {
            id: `block.${index}.attn.c_proj`,
            label: "c_proj",
            type: "Output Projection",
            kind: "attention",
            badges: ["768->768"],
            sourceRefs: {
              pytorch: ["causalselfattention.self-c_proj-nn-linear-n_embd-n_embd","causalselfattention.forward.out-self-c_proj-y"],
              jax: ["causalselfattention.__call__.out-nn-dense-channel_count-name-c_proj-y"],
            },
            focusRef: {
              pytorch: "causalselfattention.self-c_proj-nn-linear-n_embd-n_embd",
              jax: "causalselfattention.__call__.out-nn-dense-channel_count-name-c_proj-y",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: `block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        sourceRefs: {
          pytorch: ["block.forward.x-x-attn"],
          jax: ["block.__call__.x-x-attn"],
        },
        focusRef: {
          pytorch: "block.forward.x-x-attn",
          jax: "block.__call__.x-x-attn",
        },
        includeChildRefs: false,
      },
      {
        id: `block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        sourceRefs: {
          pytorch: ["block.__init__.self-ln_n-nn-layernorm-n.2","block.forward.mlp_input-self-ln_n-x"],
          jax: ["block.__call__.mlp_input-nn-layernorm-name-ln_n-x"],
        },
        focusRef: {
          pytorch: "block.__init__.self-ln_n-nn-layernorm-n.2",
          jax: "block.__call__.mlp_input-nn-layernorm-name-ln_n-x",
        },
        includeChildRefs: false,
      },
      {
        id: `block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        summary: "3072 hidden",
        sourceRefs: {
          pytorch: ["block.__init__.self-mlp-nn-sequential","block.__init__.nn-linear-n-n","block.__init__.nn-gelu","block.__init__.nn-linear-n-n.2","block.__init__.code.3","block.forward.mlp_out-self-mlp-mlp_input"],
          jax: ["class-mlp-nn-module","mlp.n_embd-int-n","mlp.hidden_dim-int-n","mlp.nn-compact","mlp.def-__call__-self-x","mlp.__call__.hidden-nn-dense-self-hidden_dim-name-c_fc-x","mlp.__call__.hidden-nn-gelu-hidden","mlp.__call__.out-nn-dense-self-n_embd-name-c_proj-hidden","mlp.__call__.return-out","block.__call__.mlp_out-mlp-mlp_input"],
        },
        focusRef: {
          pytorch: "block.__init__.self-mlp-nn-sequential",
          jax: "class-mlp-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: `block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        sourceRefs: {
          pytorch: ["block.forward.x-x-mlp_out"],
          jax: ["block.__call__.x-x-mlp_out"],
        },
        focusRef: {
          pytorch: "block.forward.x-x-mlp_out",
          jax: "block.__call__.x-x-mlp_out",
        },
        includeChildRefs: false,
      },
    ],
  };
}

type InceptionNodeId =
  | "inception3a"
  | "inception3b"
  | "inception4a"
  | "inception4b"
  | "inception4c"
  | "inception4d"
  | "inception4e"
  | "inception5a"
  | "inception5b";

type InceptionNodeConfig = {
  id: InceptionNodeId;
  label: string;
  inputChannels: number;
  branch1Channels: number;
  branch3Reduce: number;
  branch3Channels: number;
  branch5Reduce: number;
  branch5Channels: number;
  poolChannels: number;
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
    sourceRefs: {"inception3a": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","class-googlenet-nn-module","googlenet.nn-relu-inplace-true.3"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception3b": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.def-__init__","googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception4a": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.self","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception4b": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.num_classes-n","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception4c": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.code","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception4d": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.super-__init__","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception4e": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception5a": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }, "inception5b": {
      pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2","inceptionblock.nn-relu-inplace-true.3","inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4","inceptionblock.nn-relu-inplace-true.5","inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.forward.branchn-self-branchn-x.2","inceptionblock.forward.branchn-self-branchn-x.3","inceptionblock.forward.branch_pool-self-branch_pool-x","inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x","googlenet.self-stem-nn-sequential","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3"],
      jax: ["class-inceptionblock-nn-module","inceptionblock.branchn_channels-int","inceptionblock.branchn_reduce-int","inceptionblock.branchn_channels-int.2","inceptionblock.branchn_reduce-int.2","inceptionblock.branchn_channels-int.3","inceptionblock.pool_channels-int","inceptionblock.nn-compact","inceptionblock.def-__call__-self-x","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3","inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4","inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5","inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool","inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n","inceptionblock.__call__.return-x"],
    }}[config.id],
    focusRef: {"inception3a": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception3b": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception4a": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception4b": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception4c": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception4d": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception4e": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception5a": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }, "inception5b": {
      pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
      jax: "class-inceptionblock-nn-module",
    }}[config.id],
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `${config.id}.branch1`,
        label: "branch1",
        type: "1x1 Conv",
        kind: "conv",
        badges: [`${config.inputChannels}->${config.branch1Channels}`],
        sourceRefs: {"inception3a": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","class-googlenet-nn-module","googlenet.nn-relu-inplace-true.3"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception3b": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.def-__init__","googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception4a": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.self","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception4b": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.num_classes-n","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception4c": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.code","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception4d": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.super-__init__","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception4e": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception5a": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }, "inception5b": {
          pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true","inceptionblock.forward.branchn-self-branchn-x.2","googlenet.self-stem-nn-sequential","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3"],
          jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x","inceptionblock.__call__.branchn-nn-relu-branchn"],
        }}[config.id],
        focusRef: {"inception3a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception3b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception4a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception4b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception4c": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception4d": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception4e": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception5a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }, "inception5b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_channels-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-name-branchn-x",
        }}[config.id],
        includeChildRefs: false,
      },
      {
        id: `${config.id}.branch3`,
        label: "branch3",
        type: "1x1 reduce + 3x3",
        kind: "group",
        summary: "medium receptive field",
        sourceRefs: {"inception3a": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","class-googlenet-nn-module","googlenet.nn-relu-inplace-true.3"],
          jax: [],
        }, "inception3b": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.def-__init__","googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2"],
          jax: [],
        }, "inception4a": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.self","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception4b": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.num_classes-n","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception4c": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.code","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: [],
        }, "inception4d": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.super-__init__","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: [],
        }, "inception4e": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception5a": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception5b": {
          pytorch: ["inceptionblock.forward.branchn-self-branchn-x.3","googlenet.self-stem-nn-sequential","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3"],
          jax: [],
        }}[config.id],
        focusRef: {"inception3a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception3b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception4a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception4b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception4c": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception4d": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception4e": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception5a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }, "inception5b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
        }}[config.id],
        includeChildRefs: true,
        children: [
          {
            id: `${config.id}.branch3.reduce`,
            label: "reduce",
            type: "1x1 Conv",
            kind: "conv",
            badges: [`${config.inputChannels}->${config.branch3Reduce}`],
            sourceRefs: {
              pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n","inceptionblock.nn-relu-inplace-true.2"],
              jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x","inceptionblock.__call__.branchn-nn-relu-branchn.2"],
            },
            focusRef: {
              pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n",
              jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x",
            },
            includeChildRefs: false,
          },
          {
            id: `${config.id}.branch3.conv`,
            label: "conv3x3",
            type: "3x3 Conv",
            kind: "conv",
            badges: [`${config.branch3Reduce}->${config.branch3Channels}`],
            sourceRefs: {
              pytorch: ["inceptionblock.nn-relu-inplace-true.3","inceptionblock.code.5"],
              jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran","inceptionblock.__call__.branchn-nn-relu-branchn.3"],
            },
            focusRef: {
              pytorch: "inceptionblock.nn-relu-inplace-true.3",
              jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: `${config.id}.branch5`,
        label: "branch5",
        type: "1x1 reduce + 5x5",
        kind: "group",
        summary: "wide receptive field",
        sourceRefs: {"inception3a": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","class-googlenet-nn-module","googlenet.nn-relu-inplace-true.3"],
          jax: [],
        }, "inception3b": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.def-__init__","googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2"],
          jax: [],
        }, "inception4a": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.self","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception4b": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.num_classes-n","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception4c": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.code","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: [],
        }, "inception4d": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.super-__init__","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: [],
        }, "inception4e": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception5a": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n"],
          jax: [],
        }, "inception5b": {
          pytorch: ["inceptionblock.forward.branch_pool-self-branch_pool-x","googlenet.self-stem-nn-sequential","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3"],
          jax: [],
        }}[config.id],
        focusRef: {"inception3a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception3b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception4a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception4b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception4c": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception4d": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception4e": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception5a": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }, "inception5b": {
          pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
          jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
        }}[config.id],
        includeChildRefs: true,
        children: [
          {
            id: `${config.id}.branch5.reduce`,
            label: "reduce",
            type: "1x1 Conv",
            kind: "conv",
            badges: [`${config.inputChannels}->${config.branch5Reduce}`],
            sourceRefs: {
              pytorch: ["inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2","inceptionblock.nn-relu-inplace-true.4"],
              jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2","inceptionblock.__call__.branchn-nn-relu-branchn.4"],
            },
            focusRef: {
              pytorch: "inceptionblock.nn-convnd-in_channels-branchn_reduce-kernel_size-n.2",
              jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_reduce-n-n-name-branchn_reduce-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: `${config.id}.branch5.conv`,
            label: "conv5x5",
            type: "5x5 Conv",
            kind: "conv",
            badges: [`${config.branch5Reduce}->${config.branch5Channels}`],
            sourceRefs: {
              pytorch: ["inceptionblock.nn-relu-inplace-true.5","inceptionblock.code.6"],
              jax: ["inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2","inceptionblock.__call__.branchn-nn-relu-branchn.5"],
            },
            focusRef: {
              pytorch: "inceptionblock.nn-relu-inplace-true.5",
              jax: "inceptionblock.__call__.branchn-nn-conv-self-branchn_channels-n-n-padding-same-name-branchn-bran.2",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: `${config.id}.pool_proj`,
        label: "pool_proj",
        type: "3x3 Pool + 1x1 Conv",
        kind: "pool",
        badges: [`${config.inputChannels}->${config.poolChannels}`],
        sourceRefs: {"inception3a": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","class-googlenet-nn-module","googlenet.nn-relu-inplace-true.3"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception3b": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.def-__init__","googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception4a": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.self","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception4b": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.num_classes-n","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception4c": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.code","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception4d": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.super-__init__","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception4e": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception5a": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }, "inception5b": {
          pytorch: ["inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","inceptionblock.nn-convnd-in_channels-pool_channels-kernel_size-n","inceptionblock.nn-relu-inplace-true.6","inceptionblock.code.7","googlenet.self-stem-nn-sequential","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3"],
          jax: ["inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same","inceptionblock.__call__.branch_pool-nn-conv-self-pool_channels-n-n-name-pool_proj-branch_pool","inceptionblock.__call__.branch_pool-nn-relu-branch_pool"],
        }}[config.id],
        focusRef: {"inception3a": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception3b": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception4a": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception4b": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception4c": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception4d": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception4e": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception5a": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }, "inception5b": {
          pytorch: "inceptionblock.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
          jax: "inceptionblock.__call__.branch_pool-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
        }}[config.id],
        includeChildRefs: false,
      },
      {
        id: `${config.id}.concat`,
        label: "concat",
        type: "ChannelConcat",
        kind: "concat",
        badges: [`${outputChannels} channels`],
        sourceRefs: {
          pytorch: ["inceptionblock.forward.x-torch-cat-branches-dim-n","inceptionblock.forward.return-x"],
          jax: ["inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool","inceptionblock.__call__.x-jnp-concatenate-branches-axis-n"],
        },
        focusRef: {
          pytorch: "inceptionblock.forward.x-torch-cat-branches-dim-n",
          jax: "inceptionblock.__call__.branches-branchn-branchn-branchn-branch_pool",
        },
        includeChildRefs: false,
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
    sourceRefs: {
      pytorch: ["multiheadselfattention.self","multiheadselfattention.self-head_dim-embed_dim-num_heads","multiheadselfattention.self-v_proj-nn-linear-embed_dim-embed_dim","multiheadselfattention.self-out_proj-nn-linear-embed_dim-embed_dim","multiheadselfattention.def-forward-self-x","multiheadselfattention.forward.batch_size-x-size-n","multiheadselfattention.forward.v-self-v_proj-x","multiheadselfattention.forward.k-k-view-batch_size-tokens-self-num_heads-self-head_dim","multiheadselfattention.forward.k-k-transpose-n-n","multiheadselfattention.forward.v-v-view-batch_size-tokens-self-num_heads-self-head_dim","multiheadselfattention.forward.v-v-transpose-n-n","multiheadselfattention.forward.scale-self-head_dim-n","multiheadselfattention.forward.attn_scores-scores-scale","multiheadselfattention.forward.attn_weights-torch-softmax-attn_scores-dim-n","multiheadselfattention.forward.context-context-contiguous","multiheadselfattention.forward.merged-context-view-batch_size-tokens-self-num_heads-self-head_dim","multiheadselfattention.forward.return-out","class-encoderblock-nn-module","encoderblock.embed_dim-n","encoderblock.self-normn-nn-layernorm-embed_dim.2","encoderblock.self-mlp-nn-sequential","encoderblock.nn-linear-embed_dim-mlp_dim","encoderblock.nn-gelu","encoderblock.nn-linear-mlp_dim-embed_dim","encoderblock.code.4","encoderblock.def-forward-self-x","encoderblock.forward.attn_output-self-attn-attn_input","encoderblock.forward.mlp_input-self-normn-x","encoderblock.forward.return-x","class-visiontransformer-nn-module","visiontransformer.def-__init__","visiontransformer.def-forward-self-x","visiontransformer.forward.x-self-norm-x","visiontransformer.forward.cls_output-x-n"],
      jax: ["class-encoderblock-nn-module","encoderblock.embed_dim-int-n","encoderblock.num_heads-int-n","encoderblock.mlp_dim-int-n","encoderblock.nn-compact","encoderblock.def-__call__-self-x","encoderblock.__call__.y-nn-layernorm-name-ln_n-x","encoderblock.__call__.y-multiheadselfattention-self-embed_dim-self-num_heads-y","encoderblock.__call__.x-x-y","encoderblock.__call__.y-nn-layernorm-name-ln_n-x.2","encoderblock.__call__.y-nn-dense-self-mlp_dim-name-mlp_fcn-y","encoderblock.__call__.y-nn-gelu-y","encoderblock.__call__.y-nn-dense-self-embed_dim-name-mlp_fcn-y","encoderblock.__call__.out-x-y","encoderblock.__call__.return-out"],
    },
    focusRef: {
      pytorch: "multiheadselfattention.self",
      jax: "class-encoderblock-nn-module",
    },
    includeChildRefs: false,
    lazyChildren: () => [
      {
        id: `encoder.block.${index}.ln1`,
        label: "ln_1",
        type: "LayerNorm",
        kind: "norm",
        badges: ["768"],
        sourceRefs: {
          pytorch: ["vit.encoder.ln1","vit.encoder.ln1_call"],
          jax: ["encoderblock.__call__.y-nn-layernorm-name-ln_n-x"],
        },
        focusRef: {
          pytorch: "vit.encoder.ln1",
          jax: "encoderblock.__call__.y-nn-layernorm-name-ln_n-x",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.block.${index}.attn`,
        label: "attn",
        type: "MultiHeadSelfAttention",
        kind: "attention",
        badges: ["12 heads", "197 tokens"],
        sourceRefs: {
          pytorch: ["encoderblock.forward.attn_output-self-attn-attn_input"],
          jax: ["class-multiheadselfattention-nn-module","multiheadselfattention.embed_dim-int-n","multiheadselfattention.num_heads-int-n","multiheadselfattention.nn-compact","multiheadselfattention.def-__call__-self-x","multiheadselfattention.__call__.batch_size-x-shape-n","multiheadselfattention.__call__.tokens-x-shape-n","multiheadselfattention.__call__.head_dim-self-embed_dim-self-num_heads","multiheadselfattention.__call__.q-nn-dense-self-embed_dim-name-q_proj-x","multiheadselfattention.__call__.k-nn-dense-self-embed_dim-name-k_proj-x","multiheadselfattention.__call__.v-nn-dense-self-embed_dim-name-v_proj-x","multiheadselfattention.__call__.head_shape-batch_size-tokens-self-num_heads-head_dim","multiheadselfattention.__call__.q-q-reshape-head_shape","multiheadselfattention.__call__.q-jnp-transpose-q-n-n-n-n","multiheadselfattention.__call__.k-k-reshape-head_shape","multiheadselfattention.__call__.k-jnp-transpose-k-n-n-n-n","multiheadselfattention.__call__.v-v-reshape-head_shape","multiheadselfattention.__call__.v-jnp-transpose-v-n-n-n-n","multiheadselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n","multiheadselfattention.__call__.scores-q-key_transpose","multiheadselfattention.__call__.scale-head_dim-n","multiheadselfattention.__call__.attn_scores-scores-scale","multiheadselfattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n","multiheadselfattention.__call__.context-attn_weights-v","multiheadselfattention.__call__.context-jnp-transpose-context-n-n-n-n","multiheadselfattention.__call__.merged_shape-batch_size-tokens-self-embed_dim","multiheadselfattention.__call__.merged-context-reshape-merged_shape","multiheadselfattention.__call__.out-nn-dense-self-embed_dim-name-out_proj-merged","multiheadselfattention.__call__.return-out"],
        },
        focusRef: {
          pytorch: "encoderblock.forward.attn_output-self-attn-attn_input",
          jax: "class-multiheadselfattention-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.block.${index}.resid1`,
        label: "resid_1",
        type: "Add",
        kind: "residual",
        sourceRefs: {
          pytorch: ["vit.encoder.resid1"],
          jax: ["encoderblock.__call__.x-x-y"],
        },
        focusRef: {
          pytorch: "vit.encoder.resid1",
          jax: "encoderblock.__call__.x-x-y",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.block.${index}.ln2`,
        label: "ln_2",
        type: "LayerNorm",
        kind: "norm",
        sourceRefs: {
          pytorch: ["encoderblock.self-normn-nn-layernorm-embed_dim.2","encoderblock.forward.mlp_input-self-normn-x"],
          jax: ["encoderblock.__call__.y-nn-layernorm-name-ln_n-x.2"],
        },
        focusRef: {
          pytorch: "encoderblock.self-normn-nn-layernorm-embed_dim.2",
          jax: "encoderblock.__call__.y-nn-layernorm-name-ln_n-x.2",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.block.${index}.mlp`,
        label: "mlp",
        type: "FeedForward",
        kind: "mlp",
        badges: ["768->3072->768"],
        sourceRefs: {
          pytorch: ["encoderblock.self-mlp-nn-sequential","encoderblock.nn-linear-embed_dim-mlp_dim","encoderblock.nn-gelu","encoderblock.nn-linear-mlp_dim-embed_dim","encoderblock.forward.mlp_output-self-mlp-mlp_input"],
          jax: ["encoderblock.__call__.y-nn-dense-self-mlp_dim-name-mlp_fcn-y","encoderblock.__call__.y-nn-gelu-y","encoderblock.__call__.y-nn-dense-self-embed_dim-name-mlp_fcn-y"],
        },
        focusRef: {
          pytorch: "encoderblock.self-mlp-nn-sequential",
          jax: "encoderblock.__call__.y-nn-dense-self-mlp_dim-name-mlp_fcn-y",
        },
        includeChildRefs: false,
      },
      {
        id: `encoder.block.${index}.resid2`,
        label: "resid_2",
        type: "Add",
        kind: "residual",
        sourceRefs: {
          pytorch: ["vit.encoder.resid2"],
          jax: ["encoderblock.__call__.out-x-y"],
        },
        focusRef: {
          pytorch: "vit.encoder.resid2",
          jax: "encoderblock.__call__.out-x-y",
        },
        includeChildRefs: false,
      },
    ],
  };
}

const modelDefinitions: Record<ModelId, ModelDefinition> = {
  mlp: {
    stats: "2 hidden layers · sigmoid activations · backprop",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "FlatVector",
        kind: "input",
        badges: ["784->784"],
        sourceRefs: {
          pytorch: ["train_inputs-torch-tensor","code.4","n-n-n-n","n-n-n-n.2","code.5","code.6","logits-model-train_inputs"],
          jax: ["train_inputs-jnp-array","code.4","n-n-n-n","n-n-n-n.2","code.5","code.6","params-model-init-jax-random-prngkey-n-train_inputs","train_step.loss_fn.logits-model-apply-current_params-inputs"],
        },
        focusRef: {
          pytorch: "train_inputs-torch-tensor",
          jax: "train_inputs-jnp-array",
        },
        includeChildRefs: false,
      },
      {
        id: "hidden.1",
        label: "hidden.1",
        type: "HiddenLayer",
        kind: "group",
        summary: "dense + sigmoid",
        badges: ["784->128"],
        sourceRefs: {
          pytorch: [],
          jax: [],
        },
        focusRef: {
          pytorch: "mlp.input_dim-n",
          jax: "mlp.hidden_dim-int-n",
        },
        includeChildRefs: true,
        children: [
          {
            id: "hidden.1.dense",
            label: "dense",
            type: "Linear",
            kind: "linear",
            badges: ["784->128"],
            sourceRefs: {
              pytorch: ["mlp.input_dim-n","mlp.hidden_dim-n","mlp.self-hiddenn-nn-linear-input_dim-hidden_dim","mlp.forward.hn_pre-self-hiddenn-x"],
              jax: ["mlp.hidden_dim-int-n","mlp.__call__.hiddenn-nn-dense-self-hidden_dim-name-hiddenn","mlp.__call__.hn_pre-hiddenn-x"],
            },
            focusRef: {
              pytorch: "mlp.input_dim-n",
              jax: "mlp.hidden_dim-int-n",
            },
            includeChildRefs: false,
          },
          {
            id: "hidden.1.sigmoid",
            label: "sigmoid",
            type: "Activation",
            kind: "activation",
            badges: ["128->128"],
            sourceRefs: {
              pytorch: ["mlp.forward.hn-torch-sigmoid-hn_pre"],
              jax: ["mlp.__call__.hn-nn-sigmoid-hn_pre"],
            },
            focusRef: {
              pytorch: "mlp.forward.hn-torch-sigmoid-hn_pre",
              jax: "mlp.__call__.hn-nn-sigmoid-hn_pre",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: [],
          jax: [],
        },
        focusRef: {
          pytorch: "mlp.hidden_dim-n",
          jax: "mlp.hidden_dim-int-n",
        },
        includeChildRefs: true,
        children: [
          {
            id: "hidden.2.dense",
            label: "dense",
            type: "Linear",
            kind: "linear",
            badges: ["128->128"],
            sourceRefs: {
              pytorch: ["mlp.hidden_dim-n","mlp.self-hiddenn-nn-linear-hidden_dim-hidden_dim","mlp.forward.hn_pre-self-hiddenn-hn"],
              jax: ["mlp.hidden_dim-int-n","mlp.__call__.hiddenn-nn-dense-self-hidden_dim-name-hiddenn.2","mlp.__call__.hn_pre-hiddenn-hn"],
            },
            focusRef: {
              pytorch: "mlp.hidden_dim-n",
              jax: "mlp.hidden_dim-int-n",
            },
            includeChildRefs: false,
          },
          {
            id: "hidden.2.sigmoid",
            label: "sigmoid",
            type: "Activation",
            kind: "activation",
            badges: ["128->128"],
            sourceRefs: {
              pytorch: ["mlp.forward.hn-torch-sigmoid-hn_pre.2"],
              jax: ["mlp.__call__.hn-nn-sigmoid-hn_pre.2"],
            },
            focusRef: {
              pytorch: "mlp.forward.hn-torch-sigmoid-hn_pre.2",
              jax: "mlp.__call__.hn-nn-sigmoid-hn_pre.2",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "output",
        label: "output",
        type: "Linear",
        kind: "linear",
        badges: ["128->10"],
        sourceRefs: {
          pytorch: ["mlp.output_dim-n","mlp.self-output-nn-linear-hidden_dim-output_dim","mlp.forward.logits-self-output-hn"],
          jax: ["mlp.output_dim-int-n","mlp.__call__.output-nn-dense-self-output_dim-name-output","mlp.__call__.logits-output-hn"],
        },
        focusRef: {
          pytorch: "mlp.output_dim-n",
          jax: "mlp.output_dim-int-n",
        },
        includeChildRefs: false,
      },
      {
        id: "logits",
        label: "logits",
        type: "ClassScores",
        kind: "head",
        badges: ["10->10"],
        sourceRefs: {
          pytorch: ["mlp.forward.logits-self-output-hn","mlp.forward.return-logits"],
          jax: ["mlp.__call__.logits-output-hn","mlp.__call__.return-logits"],
        },
        focusRef: {
          pytorch: "mlp.forward.logits-self-output-hn",
          jax: "mlp.__call__.logits-output-hn",
        },
        includeChildRefs: false,
      },
    ],
  },
  rnn: {
    stats: "8 time steps · 64 hidden units · shared recurrent cell",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        sourceRefs: {
          pytorch: ["elmanrnn.forward.input_hidden-self-input_to_hidden-current_input"],
          jax: ["elmanrnn.__call__.input_hidden-input_to_hidden-current_input"],
        },
        focusRef: {
          pytorch: "elmanrnn.forward.input_hidden-self-input_to_hidden-current_input",
          jax: "elmanrnn.__call__.input_hidden-input_to_hidden-current_input",
        },
        includeChildRefs: false,
      },
      {
        id: "recurrent_loop",
        label: "Recurrent Loop",
        type: "UnrolledRNN",
        kind: "group",
        summary: "shared cell over time",
        badges: ["tanh"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["elmanrnn.forward.step_count-x-size-n","elmanrnn.forward.for-t-in-range-step_count"],
          jax: ["elmanrnn.__call__.input_to_hidden-nn-dense-self-hidden_size-name-input_to_hidden","elmanrnn.__call__.hidden_to_hidden-nn-dense-self-hidden_size-use_bias-false-name-hidden_to","elmanrnn.__call__.step_count-x-shape-n","elmanrnn.__call__.for-t-in-range-step_count"],
        },
        focusRef: {
          pytorch: "elmanrnn.forward.hidden_shape-batch_size-self-hidden_size",
          jax: "elmanrnn.__call__.batch_size-x-shape-n",
        },
        includeChildRefs: true,
        children: [
          {
            id: "h0",
            label: "h0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["64 hidden"],
            sourceRefs: {
              pytorch: ["elmanrnn.forward.hidden_shape-batch_size-self-hidden_size","elmanrnn.forward.h-torch-zeros-hidden_shape-device-x-device"],
              jax: ["elmanrnn.__call__.batch_size-x-shape-n","elmanrnn.__call__.hidden_shape-batch_size-self-hidden_size","elmanrnn.__call__.h-jnp-zeros-hidden_shape"],
            },
            focusRef: {
              pytorch: "elmanrnn.forward.hidden_shape-batch_size-self-hidden_size",
              jax: "elmanrnn.__call__.batch_size-x-shape-n",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["elmanrnn.self-hidden_to_output-nn-linear-hidden_size-output_size","elmanrnn.forward.logits-self-hidden_to_output-h"],
          jax: ["elmanrnn.__call__.hidden_to_output-nn-dense-self-output_size-name-hidden_to_output","elmanrnn.__call__.logits-hidden_to_output-h"],
        },
        focusRef: {
          pytorch: "elmanrnn.self-hidden_to_output-nn-linear-hidden_size-output_size",
          jax: "elmanrnn.__call__.hidden_to_output-nn-dense-self-output_size-name-hidden_to_output",
        },
        includeChildRefs: false,
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "all states"],
        sourceRefs: {
          pytorch: ["elmanrnn.forward.state_trace-torch-stack-states-dim-n","elmanrnn.forward.outputs-logits-state_trace","elmanrnn.forward.return-outputs"],
          jax: ["elmanrnn.__call__.logits-hidden_to_output-h","elmanrnn.__call__.state_trace-jnp-stack-states-axis-n","elmanrnn.__call__.outputs-logits-state_trace","elmanrnn.__call__.return-outputs"],
        },
        focusRef: {
          pytorch: "elmanrnn.forward.state_trace-torch-stack-states-dim-n",
          jax: "elmanrnn.__call__.logits-hidden_to_output-h",
        },
        includeChildRefs: false,
      },
    ],
  },
  gru: {
    stats: "8 time steps · update/reset gates · 64 hidden units",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        sourceRefs: {
          pytorch: ["grusequence.forward.current_input-x-t","grusequence.forward.h-self-cell-current_input-h"],
          jax: ["grusequence.__call__.current_input-x-t","grusequence.__call__.h-cell-current_input-h"],
        },
        focusRef: {
          pytorch: "grusequence.forward.current_input-x-t",
          jax: "grusequence.__call__.current_input-x-t",
        },
        includeChildRefs: false,
      },
      {
        id: "cell_params",
        label: "GRU Cell Params",
        type: "GatedRecurrentCell",
        kind: "group",
        summary: "6 affine projections",
        badges: ["z", "r", "n"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["grucell.def-__init__","grucell.self","grucell.self-x_z-nn-linear-input_size-hidden_size","grucell.self-h_z-nn-linear-hidden_size-hidden_size-bias-false","grucell.self-x_r-nn-linear-input_size-hidden_size","grucell.self-h_r-nn-linear-hidden_size-hidden_size-bias-false","grucell.self-x_n-nn-linear-input_size-hidden_size","grucell.self-h_n-nn-linear-hidden_size-hidden_size-bias-false","grucell.def-forward-self-x-h"],
          jax: ["class-grucell-nn-module","grucell.hidden_size-int-n","grucell.nn-compact","grucell.__call__.x_z-nn-dense-self-hidden_size-name-x_z-x","grucell.__call__.h_z-nn-dense-self-hidden_size-use_bias-false-name-h_z-h","grucell.__call__.x_r-nn-dense-self-hidden_size-name-x_r-x","grucell.__call__.h_r-nn-dense-self-hidden_size-use_bias-false-name-h_r-h","grucell.__call__.x_n-nn-dense-self-hidden_size-name-x_n-x","grucell.__call__.h_n-nn-dense-self-hidden_size-use_bias-false-name-h_n-reset_h"],
        },
        focusRef: {
          pytorch: "grucell.def-__init__",
          jax: "class-grucell-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "cell_params.update",
            label: "update params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_z", "h_z"],
            sourceRefs: {
              pytorch: ["grucell.self-x_z-nn-linear-input_size-hidden_size","grucell.self-h_z-nn-linear-hidden_size-hidden_size-bias-false","grucell.forward.x_z-self-x_z-x","grucell.forward.h_z-self-h_z-h"],
              jax: ["grucell.__call__.x_z-nn-dense-self-hidden_size-name-x_z-x","grucell.__call__.h_z-nn-dense-self-hidden_size-use_bias-false-name-h_z-h"],
            },
            focusRef: {
              pytorch: "grucell.self-x_z-nn-linear-input_size-hidden_size",
              jax: "grucell.__call__.x_z-nn-dense-self-hidden_size-name-x_z-x",
            },
            includeChildRefs: false,
          },
          {
            id: "cell_params.reset",
            label: "reset params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_r", "h_r"],
            sourceRefs: {
              pytorch: ["grucell.self-x_r-nn-linear-input_size-hidden_size","grucell.self-h_r-nn-linear-hidden_size-hidden_size-bias-false","grucell.forward.x_r-self-x_r-x","grucell.forward.h_r-self-h_r-h"],
              jax: ["grucell.__call__.x_r-nn-dense-self-hidden_size-name-x_r-x","grucell.__call__.h_r-nn-dense-self-hidden_size-use_bias-false-name-h_r-h"],
            },
            focusRef: {
              pytorch: "grucell.self-x_r-nn-linear-input_size-hidden_size",
              jax: "grucell.__call__.x_r-nn-dense-self-hidden_size-name-x_r-x",
            },
            includeChildRefs: false,
          },
          {
            id: "cell_params.candidate",
            label: "candidate params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_n", "h_n"],
            sourceRefs: {
              pytorch: ["grucell.self-x_n-nn-linear-input_size-hidden_size","grucell.self-h_n-nn-linear-hidden_size-hidden_size-bias-false","grucell.forward.x_n-self-x_n-x","grucell.forward.h_n-self-h_n-reset_h"],
              jax: ["grucell.__call__.x_n-nn-dense-self-hidden_size-name-x_n-x","grucell.__call__.h_n-nn-dense-self-hidden_size-use_bias-false-name-h_n-reset_h"],
            },
            focusRef: {
              pytorch: "grucell.self-x_n-nn-linear-input_size-hidden_size",
              jax: "grucell.__call__.x_n-nn-dense-self-hidden_size-name-x_n-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["grusequence.forward.batch_size-x-size-n","grusequence.forward.hidden_shape-batch_size-self-hidden_size","grusequence.forward.h-torch-zeros-hidden_shape-device-x-device","grusequence.forward.states","grusequence.forward.step_count-x-size-n","grusequence.forward.for-t-in-range-step_count","grusequence.forward.current_input-x-t","grusequence.forward.h-self-cell-current_input-h","grusequence.forward.states-append-h"],
          jax: ["grusequence.__call__.batch_size-x-shape-n","grusequence.__call__.hidden_shape-batch_size-self-hidden_size","grusequence.__call__.h-jnp-zeros-hidden_shape","grusequence.__call__.states","grusequence.__call__.cell-grucell-self-hidden_size","grusequence.__call__.step_count-x-shape-n","grusequence.__call__.for-t-in-range-step_count","grusequence.__call__.current_input-x-t","grusequence.__call__.h-cell-current_input-h","grusequence.__call__.states-append-h"],
        },
        focusRef: {
          pytorch: "grusequence.forward.batch_size-x-size-n",
          jax: "grusequence.__call__.batch_size-x-shape-n",
        },
        includeChildRefs: false,
        children: [
          {
            id: "h0",
            label: "h0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["64 hidden"],
            sourceRefs: {
              pytorch: ["grusequence.forward.batch_size-x-size-n","grusequence.forward.hidden_shape-batch_size-self-hidden_size","grusequence.forward.h-torch-zeros-hidden_shape-device-x-device"],
              jax: ["grusequence.__call__.batch_size-x-shape-n","grusequence.__call__.hidden_shape-batch_size-self-hidden_size","grusequence.__call__.h-jnp-zeros-hidden_shape"],
            },
            focusRef: {
              pytorch: "grusequence.forward.batch_size-x-size-n",
              jax: "grusequence.__call__.batch_size-x-shape-n",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["grusequence.self-readout-nn-linear-hidden_size-output_size","grusequence.forward.logits-self-readout-h"],
          jax: ["grusequence.output_size-int-n","grusequence.__call__.logits-nn-dense-self-output_size-name-readout-h"],
        },
        focusRef: {
          pytorch: "grusequence.self-readout-nn-linear-hidden_size-output_size",
          jax: "grusequence.output_size-int-n",
        },
        includeChildRefs: false,
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "all states"],
        sourceRefs: {
          pytorch: ["grusequence.forward.logits-self-readout-h","grusequence.forward.state_trace-torch-stack-states-dim-n","grusequence.forward.outputs-logits-state_trace","grusequence.forward.return-outputs"],
          jax: ["grusequence.__call__.logits-nn-dense-self-output_size-name-readout-h","grusequence.__call__.state_trace-jnp-stack-states-axis-n","grusequence.__call__.outputs-logits-state_trace","grusequence.__call__.return-outputs"],
        },
        focusRef: {
          pytorch: "grusequence.forward.logits-self-readout-h",
          jax: "grusequence.__call__.logits-nn-dense-self-output_size-name-readout-h",
        },
        includeChildRefs: false,
      },
    ],
  },
  vae: {
    stats: "Gaussian encoder · reparameterization trick · ELBO loss",
    nodes: [
      {
        id: "input",
        label: "input x",
        type: "FlatVector",
        kind: "input",
        badges: ["784"],
        sourceRefs: {
          pytorch: ["variationalencoder.def-forward-self-x","variationalencoder.forward.hidden-self-fcn-x","variationalautoencoder.def-forward-self-x","variationalautoencoder.forward.mu-logvar-self-encoder-x","variationalautoencoder.def-loss-self-x","variationalautoencoder.loss.reconstruction-mu-logvar-z-self-forward-x","inputs-torch-zeros-n-n","loss-reconstruction_loss-kl_loss-z-model-loss-inputs"],
          jax: ["variationalencoder.def-__call__-self-x","variationalencoder.__call__.hidden-nn-dense-self-hidden_dim-name-fcn-x","variationalautoencoder.def-__call__-self-x-epsilon","variationalautoencoder.__call__.mu-logvar-self-encoder-x","variationalautoencoder.def-loss-self-x-epsilon","variationalautoencoder.loss.reconstruction-mu-logvar-z-self-__call__-x-epsilon","inputs-jnp-zeros-n-n","train_step.loss_fn.loss-reconstruction_loss-kl_loss-z-model-apply-current_params-inputs-eps"],
        },
        focusRef: {
          pytorch: "variationalencoder.def-forward-self-x",
          jax: "variationalencoder.def-__call__-self-x",
        },
        includeChildRefs: false,
      },
      {
        id: "encoder",
        label: "Variational Encoder",
        type: "q_phi(z|x)",
        kind: "group",
        summary: "input -> mu and logvar",
        badges: ["784->256", "two heads"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-variationalencoder-nn-module","variationalencoder.self-fcn-nn-linear-input_dim-hidden_dim","variationalencoder.self-fc_mu-nn-linear-hidden_dim-latent_dim","variationalencoder.self-fc_logvar-nn-linear-hidden_dim-latent_dim","variationalencoder.def-forward-self-x","variationalencoder.forward.hidden-self-fcn-x","variationalencoder.forward.hidden-f-relu-hidden","variationalencoder.forward.mu-self-fc_mu-hidden","variationalencoder.forward.logvar-self-fc_logvar-hidden","variationalautoencoder.self-encoder-variationalencoder-input_dim-hidden_dim-latent_dim","variationalautoencoder.forward.mu-logvar-self-encoder-x"],
          jax: ["class-variationalencoder-nn-module","variationalencoder.def-__call__-self-x","variationalencoder.__call__.hidden-nn-dense-self-hidden_dim-name-fcn-x","variationalencoder.__call__.hidden-nn-relu-hidden","variationalencoder.__call__.mu-nn-dense-self-latent_dim-name-fc_mu-hidden","variationalencoder.__call__.logvar-nn-dense-self-latent_dim-name-fc_logvar-hidden","variationalautoencoder.setup.self-encoder-variationalencoder-self-input_dim-self-hidden_dim-self-late","variationalautoencoder.__call__.mu-logvar-self-encoder-x"],
        },
        focusRef: {
          pytorch: "class-variationalencoder-nn-module",
          jax: "class-variationalencoder-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "encoder.hidden",
            label: "shared trunk",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["784->256"],
            sourceRefs: {
              pytorch: ["variationalencoder.self-fcn-nn-linear-input_dim-hidden_dim","variationalencoder.forward.hidden-self-fcn-x","variationalencoder.forward.hidden-f-relu-hidden"],
              jax: ["variationalencoder.__call__.hidden-nn-dense-self-hidden_dim-name-fcn-x","variationalencoder.__call__.hidden-nn-relu-hidden"],
            },
            focusRef: {
              pytorch: "variationalencoder.self-fcn-nn-linear-input_dim-hidden_dim",
              jax: "variationalencoder.__call__.hidden-nn-dense-self-hidden_dim-name-fcn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "encoder.mu",
            label: "mu",
            type: "Mean head",
            kind: "linear",
            badges: ["32"],
            sourceRefs: {
              pytorch: ["variationalencoder.self-fc_mu-nn-linear-hidden_dim-latent_dim","variationalencoder.forward.mu-self-fc_mu-hidden","variationalencoder.forward.return-mu-logvar","variationalautoencoder.reparameterize.std-torch-exp-n-logvar","variationalautoencoder.reparameterize.z-mu-std-epsilon","variationalautoencoder.loss.kl_terms-n-logvar-mu-pow-n-logvar-exp"],
              jax: ["variationalencoder.__call__.mu-nn-dense-self-latent_dim-name-fc_mu-hidden","variationalencoder.__call__.return-mu-logvar","variationalautoencoder.reparameterize.std-jnp-exp-n-logvar","variationalautoencoder.reparameterize.z-mu-std-epsilon","variationalautoencoder.loss.kl_terms-n-logvar-mu-n-jnp-exp-logvar"],
            },
            focusRef: {
              pytorch: "variationalencoder.self-fc_mu-nn-linear-hidden_dim-latent_dim",
              jax: "variationalencoder.__call__.mu-nn-dense-self-latent_dim-name-fc_mu-hidden",
            },
            includeChildRefs: false,
          },
          {
            id: "encoder.logvar",
            label: "logvar",
            type: "Log-variance head",
            kind: "linear",
            badges: ["32"],
            sourceRefs: {
              pytorch: ["variationalencoder.self-fc_logvar-nn-linear-hidden_dim-latent_dim","variationalencoder.forward.logvar-self-fc_logvar-hidden","variationalencoder.forward.return-mu-logvar","variationalautoencoder.reparameterize.std-torch-exp-n-logvar","variationalautoencoder.loss.kl_terms-n-logvar-mu-pow-n-logvar-exp"],
              jax: ["variationalencoder.__call__.logvar-nn-dense-self-latent_dim-name-fc_logvar-hidden","variationalencoder.__call__.return-mu-logvar","variationalautoencoder.reparameterize.std-jnp-exp-n-logvar","variationalautoencoder.loss.kl_terms-n-logvar-mu-n-jnp-exp-logvar"],
            },
            focusRef: {
              pytorch: "variationalencoder.self-fc_logvar-nn-linear-hidden_dim-latent_dim",
              jax: "variationalencoder.__call__.logvar-nn-dense-self-latent_dim-name-fc_logvar-hidden",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["variationalautoencoder.def-reparameterize-self-mu-logvar","variationalautoencoder.reparameterize.std-torch-exp-n-logvar","variationalautoencoder.reparameterize.epsilon-torch-randn_like-std","variationalautoencoder.reparameterize.z-mu-std-epsilon","variationalautoencoder.reparameterize.return-z","variationalautoencoder.forward.z-self-reparameterize-mu-logvar","variationalautoencoder.forward.return-reconstruction-mu-logvar-z","final_latent_sample-z-detach"],
          jax: ["variationalautoencoder.def-reparameterize-self-mu-logvar-epsilon","variationalautoencoder.reparameterize.std-jnp-exp-n-logvar","variationalautoencoder.reparameterize.z-mu-std-epsilon","variationalautoencoder.reparameterize.return-z","variationalautoencoder.__call__.z-self-reparameterize-mu-logvar-epsilon","variationalautoencoder.__call__.return-reconstruction-mu-logvar-z","epsilon-jnp-ones-n-n","final_latent_sample-z"],
        },
        focusRef: {
          pytorch: "variationalautoencoder.def-reparameterize-self-mu-logvar",
          jax: "variationalautoencoder.def-reparameterize-self-mu-logvar-epsilon",
        },
        includeChildRefs: false,
      },
      {
        id: "decoder",
        label: "Decoder",
        type: "p_theta(x|z)",
        kind: "group",
        summary: "latent sample -> reconstruction",
        badges: ["32->256->784"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-decoder-nn-module","decoder.self-net-nn-sequential","decoder.nn-linear-latent_dim-hidden_dim","decoder.nn-relu","decoder.nn-linear-hidden_dim-output_dim","decoder.nn-sigmoid","decoder.def-forward-self-z","decoder.forward.reconstruction-self-net-z","variationalautoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim","variationalautoencoder.forward.reconstruction-self-decoder-z"],
          jax: ["class-decoder-nn-module","decoder.def-__call__-self-z","decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z","decoder.__call__.x-nn-relu-x","decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x","decoder.__call__.reconstruction-nn-sigmoid-x","variationalautoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim","variationalautoencoder.__call__.reconstruction-self-decoder-z"],
        },
        focusRef: {
          pytorch: "class-decoder-nn-module",
          jax: "class-decoder-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "decoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["32->256"],
            sourceRefs: {
              pytorch: ["decoder.self-net-nn-sequential","decoder.nn-linear-latent_dim-hidden_dim","decoder.nn-relu"],
              jax: ["decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z","decoder.__call__.x-nn-relu-x"],
            },
            focusRef: {
              pytorch: "decoder.self-net-nn-sequential",
              jax: "decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z",
            },
            includeChildRefs: false,
          },
          {
            id: "decoder.reconstruction",
            label: "reconstruction",
            type: "Bernoulli probs",
            kind: "activation",
            badges: ["sigmoid"],
            sourceRefs: {
              pytorch: ["decoder.nn-linear-hidden_dim-output_dim","decoder.nn-sigmoid","decoder.forward.reconstruction-self-net-z","decoder.forward.return-reconstruction","variationalautoencoder.forward.reconstruction-self-decoder-z","variationalautoencoder.forward.return-reconstruction-mu-logvar-z"],
              jax: ["decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x","decoder.__call__.reconstruction-nn-sigmoid-x","decoder.__call__.return-reconstruction","variationalautoencoder.__call__.reconstruction-self-decoder-z","variationalautoencoder.__call__.return-reconstruction-mu-logvar-z"],
            },
            focusRef: {
              pytorch: "decoder.nn-linear-hidden_dim-output_dim",
              jax: "decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["variationalautoencoder.def-loss-self-x","variationalautoencoder.loss.reconstruction-mu-logvar-z-self-forward-x","variationalautoencoder.loss.total_loss-reconstruction_loss-kl_loss","variationalautoencoder.loss.return-total_loss-reconstruction_loss-kl_loss-z","loss-reconstruction_loss-kl_loss-z-model-loss-inputs","loss-backward","optimizer-step"],
          jax: ["variationalautoencoder.def-loss-self-x-epsilon","variationalautoencoder.loss.reconstruction-mu-logvar-z-self-__call__-x-epsilon","variationalautoencoder.loss.total_loss-reconstruction_loss-kl_loss","variationalautoencoder.loss.return-total_loss-reconstruction_loss-kl_loss-z","train_step.loss_fn.loss-reconstruction_loss-kl_loss-z-model-apply-current_params-inputs-eps","train_step.loss-aux-grads-jax-value_and_grad-loss_fn-has_aux-true-params","train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads"],
        },
        focusRef: {
          pytorch: "variationalautoencoder.def-loss-self-x",
          jax: "def-binary_cross_entropy-reconstruction-x",
        },
        includeChildRefs: true,
        children: [
          {
            id: "elbo_loss.reconstruction",
            label: "reconstruction",
            type: "BCE",
            kind: "head",
            sourceRefs: {
              pytorch: ["variationalautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su"],
              jax: ["def-binary_cross_entropy-reconstruction-x","binary_cross_entropy.reconstruction-jnp-clip-reconstruction-eps-n-eps","binary_cross_entropy.loss_values-x-jnp-log-reconstruction-n-x-jnp-log-n-reconstruction","binary_cross_entropy.loss-jnp-sum-loss_values","variationalautoencoder.loss.reconstruction_loss-binary_cross_entropy-reconstruction-x"],
            },
            focusRef: {
              pytorch: "variationalautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su",
              jax: "def-binary_cross_entropy-reconstruction-x",
            },
            includeChildRefs: false,
          },
          {
            id: "elbo_loss.kl",
            label: "KL to prior",
            type: "N(0, I)",
            kind: "head",
            sourceRefs: {
              pytorch: ["variationalautoencoder.loss.kl_terms-n-logvar-mu-pow-n-logvar-exp","variationalautoencoder.loss.kl_loss-n-torch-sum-kl_terms"],
              jax: ["variationalautoencoder.loss.kl_terms-n-logvar-mu-n-jnp-exp-logvar","variationalautoencoder.loss.kl_loss-n-jnp-sum-kl_terms"],
            },
            focusRef: {
              pytorch: "variationalautoencoder.loss.kl_terms-n-logvar-mu-pow-n-logvar-exp",
              jax: "variationalautoencoder.loss.kl_terms-n-logvar-mu-n-jnp-exp-logvar",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  gan: {
    stats: "latent generator · real/fake discriminator · minimax training",
    nodes: [
      {
        id: "latent",
        label: "latent z",
        type: "NoiseVector",
        kind: "input",
        badges: ["100-d"],
        sourceRefs: {
          pytorch: ["generator.def-forward-self-z","generator.forward.fake_images-self-net-z","gan.def-generate-self-z","gan.generate.fake_images-self-generator-z","gan.discriminator_loss.fake_images-self-generator-z","gan.generator_loss.fake_images-self-generator-z","z-torch-randn-n-n"],
          jax: ["generator.def-__call__-self-z","generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-z","gan.def-generate-self-z","gan.generate.fake_images-self-generator-z","gan.__call__.fake_images-self-generate-z","gan.discriminator_loss.fake_images-self-generate-z","gan.generator_loss.fake_images-self-generate-z","z-jnp-ones-n-n"],
        },
        focusRef: {
          pytorch: "generator.def-forward-self-z",
          jax: "generator.def-__call__-self-z",
        },
        includeChildRefs: false,
      },
      {
        id: "real_images",
        label: "real images",
        type: "Data samples",
        kind: "input",
        badges: ["784 pixels"],
        sourceRefs: {
          pytorch: ["discriminator.def-forward-self-images","discriminator.forward.logits-self-net-images","gan.def-discriminate-self-images","gan.discriminate.logits-self-discriminator-images","gan.discriminator_loss.real_logits-self-discriminator-real_images","real_images-torch-zeros-n-n","d_loss-model-discriminator_loss-real_images-z"],
          jax: ["discriminator.def-__call__-self-images","discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-images","gan.def-discriminate-self-images","gan.discriminate.logits-self-discriminator-images","gan.__call__.real_logits-self-discriminate-real_images","gan.discriminator_loss.real_logits-self-discriminate-real_images","real_images-jnp-zeros-n-n","discriminator_train_step.loss_fn.loss-model-apply-current_params-real_images-z-method-gan-discriminator_l"],
        },
        focusRef: {
          pytorch: "discriminator.def-forward-self-images",
          jax: "discriminator.def-__call__-self-images",
        },
        includeChildRefs: false,
      },
      {
        id: "generator",
        label: "Generator G",
        type: "MLP",
        kind: "group",
        summary: "z -> fake image",
        badges: ["100->784", "tanh"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-generator-nn-module","generator.def-forward-self-z","gan.self-generator-generator-latent_dim-image_dim-hidden_dim","gan.generate.fake_images-self-generator-z","gan.discriminator_loss.fake_images-self-generator-z","gan.generator_loss.fake_images-self-generator-z"],
          jax: ["class-generator-nn-module","generator.def-__call__-self-z","gan.setup.self-generator-generator-self-latent_dim-self-image_dim-self-hidden_dim","gan.generate.fake_images-self-generator-z","gan.__call__.fake_images-self-generate-z","gan.discriminator_loss.fake_images-self-generate-z","gan.generator_loss.fake_images-self-generate-z"],
        },
        focusRef: {
          pytorch: "class-generator-nn-module",
          jax: "class-generator-nn-module",
        },
        includeChildRefs: true,
        children: [
          {
            id: "generator.hidden",
            label: "hidden MLP",
            type: "Linear stack",
            kind: "linear",
            badges: ["LeakyReLU"],
            sourceRefs: {
              pytorch: ["generator.self-net-nn-sequential","generator.nn-linear-latent_dim-hidden_dim","generator.nn-leakyrelu-n","generator.nn-linear-hidden_dim-hidden_dim","generator.nn-leakyrelu-n.2"],
              jax: ["generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-z","generator.__call__.x-nn-leaky_relu-x-negative_slope-n","generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-x","generator.__call__.x-nn-leaky_relu-x-negative_slope-n.2"],
            },
            focusRef: {
              pytorch: "generator.self-net-nn-sequential",
              jax: "generator.__call__.x-nn-dense-self-hidden_dim-name-fcn-z",
            },
            includeChildRefs: false,
          },
          {
            id: "generator.output",
            label: "fake image",
            type: "Tanh output",
            kind: "activation",
            badges: ["image_dim"],
            sourceRefs: {
              pytorch: ["generator.nn-linear-hidden_dim-image_dim","generator.nn-tanh","generator.forward.fake_images-self-net-z"],
              jax: ["generator.__call__.x-nn-dense-self-image_dim-name-fcn-x","generator.__call__.fake_images-jnp-tanh-x"],
            },
            focusRef: {
              pytorch: "generator.nn-linear-hidden_dim-image_dim",
              jax: "generator.__call__.x-nn-dense-self-image_dim-name-fcn-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["class-discriminator-nn-module","discriminator.def-forward-self-images","gan.self-discriminator-discriminator-image_dim-hidden_dim","gan.discriminate.logits-self-discriminator-images","gan.discriminator_loss.real_logits-self-discriminator-real_images","gan.discriminator_loss.fake_logits-self-discriminator-fake_images-detach","gan.generator_loss.fake_logits-self-discriminator-fake_images"],
          jax: ["class-discriminator-nn-module","discriminator.def-__call__-self-images","gan.setup.self-discriminator-discriminator-self-image_dim-self-hidden_dim","gan.discriminate.logits-self-discriminator-images","gan.__call__.real_logits-self-discriminate-real_images","gan.__call__.fake_logits-self-discriminate-fake_images","gan.discriminator_loss.real_logits-self-discriminate-real_images","gan.discriminator_loss.fake_logits-self-discriminate-jax-lax-stop_gradient-fake_images","gan.generator_loss.fake_logits-self-discriminate-fake_images"],
        },
        focusRef: {
          pytorch: "class-discriminator-nn-module",
          jax: "class-discriminator-nn-module",
        },
        includeChildRefs: true,
        children: [
          {
            id: "discriminator.hidden",
            label: "hidden MLP",
            type: "Linear stack",
            kind: "linear",
            badges: ["LeakyReLU"],
            sourceRefs: {
              pytorch: ["discriminator.self-net-nn-sequential","discriminator.nn-linear-image_dim-hidden_dim","discriminator.nn-leakyrelu-n","discriminator.nn-linear-hidden_dim-hidden_dim","discriminator.nn-leakyrelu-n.2"],
              jax: ["discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-images","discriminator.__call__.x-nn-leaky_relu-x-negative_slope-n","discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-x","discriminator.__call__.x-nn-leaky_relu-x-negative_slope-n.2"],
            },
            focusRef: {
              pytorch: "discriminator.self-net-nn-sequential",
              jax: "discriminator.__call__.x-nn-dense-self-hidden_dim-name-fcn-images",
            },
            includeChildRefs: false,
          },
          {
            id: "discriminator.logit",
            label: "real/fake logit",
            type: "Linear",
            kind: "head",
            badges: ["scalar"],
            sourceRefs: {
              pytorch: ["discriminator.nn-linear-hidden_dim-n","discriminator.forward.logits-self-net-images","discriminator.forward.logits-logits-squeeze-n"],
              jax: ["discriminator.__call__.logits-nn-dense-n-name-fcn-x","discriminator.__call__.logits-jnp-squeeze-logits-axis-n"],
            },
            focusRef: {
              pytorch: "discriminator.nn-linear-hidden_dim-n",
              jax: "discriminator.__call__.logits-nn-dense-n-name-fcn-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["gan.def-discriminator_loss-self-real_images-z","gan.discriminator_loss.loss-real_loss-fake_loss","d_loss-model-discriminator_loss-real_images-z","d_loss-backward","discriminator_optimizer-step"],
          jax: ["gan.def-discriminator_loss-self-real_images-z","gan.discriminator_loss.loss-real_loss-fake_loss","discriminator_train_step.loss_fn.loss-model-apply-current_params-real_images-z-method-gan-discriminator_l","discriminator_train_step.loss-grads-jax-value_and_grad-loss_fn-params","discriminator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads","params-d_loss-discriminator_train_step-params-real_images-z"],
        },
        focusRef: {
          pytorch: "gan.def-discriminator_loss-self-real_images-z",
          jax: "gan.def-discriminator_loss-self-real_images-z",
        },
        includeChildRefs: true,
        children: [
          {
            id: "discriminator_loss.real",
            label: "real branch",
            type: "BCE target 1",
            kind: "head",
            sourceRefs: {
              pytorch: ["gan.discriminator_loss.real_logits-self-discriminator-real_images","gan.discriminator_loss.real_targets-torch-ones_like-real_logits","gan.discriminator_loss.real_loss-f-binary_cross_entropy_with_logits-real_logits-real_targets"],
              jax: ["gan.discriminator_loss.real_logits-self-discriminate-real_images","gan.discriminator_loss.real_targets-jnp-ones_like-real_logits","gan.discriminator_loss.real_loss-binary_cross_entropy_with_logits-real_logits-real_targets"],
            },
            focusRef: {
              pytorch: "gan.discriminator_loss.real_logits-self-discriminator-real_images",
              jax: "gan.discriminator_loss.real_logits-self-discriminate-real_images",
            },
            includeChildRefs: false,
          },
          {
            id: "discriminator_loss.fake",
            label: "fake branch",
            type: "BCE target 0",
            kind: "head",
            sourceRefs: {
              pytorch: ["gan.discriminator_loss.fake_images-self-generator-z","gan.discriminator_loss.fake_logits-self-discriminator-fake_images-detach","gan.discriminator_loss.fake_targets-torch-zeros_like-fake_logits","gan.discriminator_loss.fake_loss-f-binary_cross_entropy_with_logits-fake_logits-fake_targets"],
              jax: ["gan.discriminator_loss.fake_images-self-generate-z","gan.discriminator_loss.fake_logits-self-discriminate-jax-lax-stop_gradient-fake_images","gan.discriminator_loss.fake_targets-jnp-zeros_like-fake_logits","gan.discriminator_loss.fake_loss-binary_cross_entropy_with_logits-fake_logits-fake_targets"],
            },
            focusRef: {
              pytorch: "gan.discriminator_loss.fake_images-self-generator-z",
              jax: "gan.discriminator_loss.fake_images-self-generate-z",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["gan.def-generator_loss-self-z","gan.generator_loss.fake_images-self-generator-z","gan.generator_loss.fake_logits-self-discriminator-fake_images","gan.generator_loss.real_targets-torch-ones_like-fake_logits","gan.generator_loss.loss-f-binary_cross_entropy_with_logits-fake_logits-real_targets","g_loss-model-generator_loss-z","g_loss-backward","generator_optimizer-step"],
          jax: ["gan.def-generator_loss-self-z","gan.generator_loss.fake_images-self-generate-z","gan.generator_loss.fake_logits-self-discriminate-fake_images","gan.generator_loss.real_targets-jnp-ones_like-fake_logits","gan.generator_loss.loss-binary_cross_entropy_with_logits-fake_logits-real_targets","generator_train_step.loss_fn.loss-model-apply-current_params-z-method-gan-generator_loss","generator_train_step.loss-grads-jax-value_and_grad-loss_fn-params","generator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads","params-g_loss-generator_train_step-params-z"],
        },
        focusRef: {
          pytorch: "gan.def-generator_loss-self-z",
          jax: "gan.def-generator_loss-self-z",
        },
        includeChildRefs: false,
      },
      {
        id: "alternating_updates",
        label: "Alternating Updates",
        type: "Two optimizers",
        kind: "group",
        summary: "update D, then update G",
        badges: ["minimax game"],
        sourceRefs: {
          pytorch: ["generator_optimizer-torch-optim-sgd-model-generator-parameters-lr-n","discriminator_optimizer-torch-optim-sgd-model-discriminator-parameters-l","for-step-in-range-n","discriminator_optimizer-zero_grad","d_loss-model-discriminator_loss-real_images-z","d_loss-backward","discriminator_optimizer-step","generator_optimizer-zero_grad","g_loss-model-generator_loss-z","g_loss-backward","generator_optimizer-step"],
          jax: ["def-discriminator_train_step-params-real_images-z-learning_rate-n","discriminator_train_step.loss-grads-jax-value_and_grad-loss_fn-params","discriminator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads","def-generator_train_step-params-z-learning_rate-n","generator_train_step.loss-grads-jax-value_and_grad-loss_fn-params","generator_train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads","for-step-in-range-n","params-d_loss-discriminator_train_step-params-real_images-z","params-g_loss-generator_train_step-params-z"],
        },
        focusRef: {
          pytorch: "generator_optimizer-torch-optim-sgd-model-generator-parameters-lr-n",
          jax: "def-discriminator_train_step-params-real_images-z-learning_rate-n",
        },
        includeChildRefs: false,
      },
    ],
  },
  seq2seq: {
    stats: "7 source steps · 6 target steps · fixed context state",
    nodes: [
      {
        id: "source.input",
        label: "source input",
        type: "TokenIds",
        kind: "input",
        badges: ["7 tokens", "reversed"],
        sourceRefs: {
          pytorch: ["seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions","seqnseqencoder.forward.embeddings-self-embedding-reversed_ids","seqnseq.forward.encoder_outputs-self-encoder-source_ids"],
          jax: ["seqnseqencoder.__call__.reversed_ids-source_ids-source_order","seqnseqencoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-source_embe","seqnseq.__call__.encoder_outputs-encoder-source_ids"],
        },
        focusRef: {
          pytorch: "seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions",
          jax: "seqnseqencoder.__call__.reversed_ids-source_ids-source_order",
        },
        includeChildRefs: false,
      },
      {
        id: "target.input",
        label: "target input",
        type: "ShiftedTokenIds",
        kind: "input",
        badges: ["6 tokens", "teacher forcing"],
        sourceRefs: {
          pytorch: ["seqnseqdecoder.forward.embeddings-self-embedding-decoder_input_ids","seqnseqdecoder.forward.current_embedding-embeddings-t","seqnseq.forward.decoder_outputs-self-decoder-decoder_input_ids-context"],
          jax: ["seqnseqdecoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-target_embe","seqnseqdecoder.__call__.current_embedding-embeddings-t","seqnseq.__call__.decoder_outputs-decoder-decoder_input_ids-context"],
        },
        focusRef: {
          pytorch: "seqnseqdecoder.forward.embeddings-self-embedding-decoder_input_ids",
          jax: "seqnseqdecoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-target_embe",
        },
        includeChildRefs: false,
      },
      {
        id: "lstm_cell",
        label: "LSTM Cell",
        type: "SharedGateCell",
        kind: "group",
        summary: "encoder and decoder cells",
        badges: ["i", "f", "g", "o"],
        sourceRefs: {
          pytorch: [],
          jax: [],
        },
        focusRef: {
          pytorch: "lstmcell.self-x_i-nn-linear-input_size-hidden_size",
          jax: "lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x",
        },
        includeChildRefs: true,
        children: [
          {
            id: "lstm_cell.input_gate",
            label: "input gate",
            type: "SigmoidGate",
            kind: "recurrent",
            badges: ["i_t"],
            sourceRefs: {
              pytorch: ["lstmcell.self-x_i-nn-linear-input_size-hidden_size","lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.forward.x_i-self-x_i-x","lstmcell.forward.h_i-self-h_i-h","lstmcell.forward.i_pre-x_i-h_i","lstmcell.forward.i-torch-sigmoid-i_pre"],
              jax: ["lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h","lstmcell.__call__.i_pre-x_i-h_i","lstmcell.__call__.i-nn-sigmoid-i_pre"],
            },
            focusRef: {
              pytorch: "lstmcell.self-x_i-nn-linear-input_size-hidden_size",
              jax: "lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x",
            },
            includeChildRefs: false,
          },
          {
            id: "lstm_cell.forget_gate",
            label: "forget gate",
            type: "SigmoidGate",
            kind: "recurrent",
            badges: ["f_t"],
            sourceRefs: {
              pytorch: ["lstmcell.self-x_f-nn-linear-input_size-hidden_size","lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.forward.x_f-self-x_f-x","lstmcell.forward.h_f-self-h_f-h","lstmcell.forward.f_pre-x_f-h_f","lstmcell.forward.f-torch-sigmoid-f_pre"],
              jax: ["lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h","lstmcell.__call__.f_pre-x_f-h_f","lstmcell.__call__.f-nn-sigmoid-f_pre"],
            },
            focusRef: {
              pytorch: "lstmcell.self-x_f-nn-linear-input_size-hidden_size",
              jax: "lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x",
            },
            includeChildRefs: false,
          },
          {
            id: "lstm_cell.candidate",
            label: "candidate",
            type: "TanhMemory",
            kind: "activation",
            badges: ["g_t"],
            sourceRefs: {
              pytorch: ["lstmcell.self-x_g-nn-linear-input_size-hidden_size","lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.forward.x_g-self-x_g-x","lstmcell.forward.h_g-self-h_g-h","lstmcell.forward.g_pre-x_g-h_g","lstmcell.forward.g-torch-tanh-g_pre"],
              jax: ["lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h","lstmcell.__call__.g_pre-x_g-h_g","lstmcell.__call__.g-jnp-tanh-g_pre"],
            },
            focusRef: {
              pytorch: "lstmcell.self-x_g-nn-linear-input_size-hidden_size",
              jax: "lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x",
            },
            includeChildRefs: false,
          },
          {
            id: "lstm_cell.output_gate",
            label: "output gate",
            type: "SigmoidGate",
            kind: "recurrent",
            badges: ["o_t"],
            sourceRefs: {
              pytorch: ["lstmcell.self-x_o-nn-linear-input_size-hidden_size","lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.forward.x_o-self-x_o-x","lstmcell.forward.h_o-self-h_o-h","lstmcell.forward.o_pre-x_o-h_o","lstmcell.forward.o-torch-sigmoid-o_pre"],
              jax: ["lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h","lstmcell.__call__.o_pre-x_o-h_o","lstmcell.__call__.o-nn-sigmoid-o_pre"],
            },
            focusRef: {
              pytorch: "lstmcell.self-x_o-nn-linear-input_size-hidden_size",
              jax: "lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x",
            },
            includeChildRefs: false,
          },
          {
            id: "lstm_cell.state_update",
            label: "state update",
            type: "CellAndHidden",
            kind: "recurrent",
            badges: ["c_t", "h_t"],
            sourceRefs: {
              pytorch: ["lstmcell.forward.forget_c-f-c","lstmcell.forward.write_c-i-g","lstmcell.forward.c_next-forget_c-write_c","lstmcell.forward.c_readout-torch-tanh-c_next","lstmcell.forward.h_next-o-c_readout","lstmcell.forward.next_state-h_next-c_next"],
              jax: ["lstmcell.__call__.forget_c-f-c","lstmcell.__call__.write_c-i-g","lstmcell.__call__.c_next-forget_c-write_c","lstmcell.__call__.c_readout-jnp-tanh-c_next","lstmcell.__call__.h_next-o-c_readout","lstmcell.__call__.next_state-h_next-c_next"],
            },
            focusRef: {
              pytorch: "lstmcell.forward.forget_c-f-c",
              jax: "lstmcell.__call__.forget_c-f-c",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["seqnseqencoder.self-embedding-nn-embedding-vocab_size-embedding_size","seqnseqencoder.self-cell-lstmcell-embedding_size-hidden_size","seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions","seqnseqencoder.forward.embeddings-self-embedding-reversed_ids","seqnseqencoder.forward.state_shape-batch_size-self-hidden_size","seqnseqencoder.forward.h-torch-zeros-state_shape-device-source_ids-device","seqnseqencoder.forward.c-torch-zeros-state_shape-device-source_ids-device","seqnseqencoder.forward.for-t-in-range-source_steps","seqnseqencoder.forward.current_embedding-embeddings-t","seqnseqencoder.forward.state-self-cell-current_embedding-h-c","seqnseqencoder.forward.h-state-n","seqnseqencoder.forward.c-state-n","seqnseqencoder.forward.encoder_states-append-h","seqnseqencoder.forward.context-h-c","seqnseqencoder.forward.encoder_trace-torch-stack-encoder_states-dim-n","seqnseq.forward.encoder_outputs-self-encoder-source_ids","seqnseq.forward.context-encoder_outputs-n"],
          jax: ["seqnseqencoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-source_embe","seqnseqencoder.__call__.state_shape-batch_size-self-hidden_size","seqnseqencoder.__call__.h-jnp-zeros-state_shape","seqnseqencoder.__call__.c-jnp-zeros-state_shape","seqnseqencoder.__call__.cell-lstmcell-self-hidden_size-name-encoder_cell","seqnseqencoder.__call__.for-t-in-range-source_steps","seqnseqencoder.__call__.current_embedding-embeddings-t","seqnseqencoder.__call__.state-cell-current_embedding-h-c","seqnseqencoder.__call__.h-state-n","seqnseqencoder.__call__.c-state-n","seqnseqencoder.__call__.encoder_states-append-h","seqnseqencoder.__call__.context-h-c","seqnseqencoder.__call__.encoder_trace-jnp-stack-encoder_states-axis-n","seqnseq.__call__.encoder_outputs-encoder-source_ids","seqnseq.__call__.context-encoder_outputs-n"],
        },
        focusRef: {
          pytorch: "seqnseqencoder.self-embedding-nn-embedding-vocab_size-embedding_size",
          jax: "seqnseqencoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-source_embe",
        },
        includeChildRefs: false,
        children: [
          {
            id: "encoder.reverse",
            label: "reverse source",
            type: "TokenOrder",
            kind: "reshape",
            badges: ["optimization"],
            sourceRefs: {
              pytorch: ["seqnseqencoder.forward.source_positions-torch-arange-source_steps-n-n-n-device-source_ids-devic","seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions"],
              jax: ["seqnseqencoder.__call__.source_order-jnp-arange-source_steps-n-n-n","seqnseqencoder.__call__.reversed_ids-source_ids-source_order"],
            },
            focusRef: {
              pytorch: "seqnseqencoder.forward.source_positions-torch-arange-source_steps-n-n-n-device-source_ids-devic",
              jax: "seqnseqencoder.__call__.source_order-jnp-arange-source_steps-n-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "encoder.embedding",
            label: "source embedding",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab->128"],
            sourceRefs: {
              pytorch: ["seqnseqencoder.self-embedding-nn-embedding-vocab_size-embedding_size","seqnseqencoder.forward.embeddings-self-embedding-reversed_ids"],
              jax: ["seqnseqencoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-source_embe"],
            },
            focusRef: {
              pytorch: "seqnseqencoder.self-embedding-nn-embedding-vocab_size-embedding_size",
              jax: "seqnseqencoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-source_embe",
            },
            includeChildRefs: false,
          },
          {
            id: "encoder.initial_state",
            label: "h0/c0",
            type: "ZeroState",
            kind: "recurrent",
            badges: ["256 hidden"],
            sourceRefs: {
              pytorch: ["seqnseqencoder.forward.state_shape-batch_size-self-hidden_size","seqnseqencoder.forward.h-torch-zeros-state_shape-device-source_ids-device","seqnseqencoder.forward.c-torch-zeros-state_shape-device-source_ids-device"],
              jax: ["seqnseqencoder.__call__.state_shape-batch_size-self-hidden_size","seqnseqencoder.__call__.h-jnp-zeros-state_shape","seqnseqencoder.__call__.c-jnp-zeros-state_shape"],
            },
            focusRef: {
              pytorch: "seqnseqencoder.forward.state_shape-batch_size-self-hidden_size",
              jax: "seqnseqencoder.__call__.state_shape-batch_size-self-hidden_size",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["seqnseqencoder.forward.context-h-c","seqnseqencoder.forward.encoder_trace-torch-stack-encoder_states-dim-n","seqnseq.forward.encoder_outputs-self-encoder-source_ids","seqnseq.forward.context-encoder_outputs-n","seqnseq.forward.decoder_outputs-self-decoder-decoder_input_ids-context"],
          jax: ["seqnseqencoder.__call__.context-h-c","seqnseqencoder.__call__.encoder_trace-jnp-stack-encoder_states-axis-n","seqnseq.__call__.encoder_outputs-encoder-source_ids","seqnseq.__call__.context-encoder_outputs-n","seqnseq.__call__.decoder_outputs-decoder-decoder_input_ids-context"],
        },
        focusRef: {
          pytorch: "seqnseqencoder.forward.context-h-c",
          jax: "seqnseqencoder.__call__.context-h-c",
        },
        includeChildRefs: false,
      },
      {
        id: "decoder",
        label: "Decoder",
        type: "RecurrentDecoder",
        kind: "group",
        summary: "teacher-forced outputs",
        badges: ["autoregressive form"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["seqnseqdecoder.self-cell-lstmcell-embedding_size-hidden_size","seqnseqdecoder.self-output_projection-nn-linear-hidden_size-vocab_size","seqnseqdecoder.forward.h-c-context","seqnseqdecoder.forward.for-t-in-range-target_steps","seqnseqdecoder.forward.output_logits-torch-stack-logits_per_step-dim-n","seqnseqdecoder.forward.decoder_trace-torch-stack-decoder_states-dim-n","seqnseq.forward.decoder_outputs-self-decoder-decoder_input_ids-context","seqnseq.forward.logits-decoder_outputs-n","seqnseq.forward.decoder_trace-decoder_outputs-n"],
          jax: ["seqnseqdecoder.__call__.cell-lstmcell-self-hidden_size-name-decoder_cell","seqnseqdecoder.__call__.output_projection-nn-dense-self-vocab_size-name-output_projection","seqnseqdecoder.__call__.for-t-in-range-target_steps","seqnseqdecoder.__call__.output_logits-jnp-stack-logits_per_step-axis-n","seqnseqdecoder.__call__.decoder_trace-jnp-stack-decoder_states-axis-n","seqnseq.__call__.decoder_outputs-decoder-decoder_input_ids-context","seqnseq.__call__.logits-decoder_outputs-n","seqnseq.__call__.decoder_trace-decoder_outputs-n"],
        },
        focusRef: {
          pytorch: "seqnseqdecoder.self-embedding-nn-embedding-vocab_size-embedding_size",
          jax: "seqnseqdecoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-target_embe",
        },
        includeChildRefs: true,
        children: [
          {
            id: "decoder.embedding",
            label: "target embedding",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab->128"],
            sourceRefs: {
              pytorch: ["seqnseqdecoder.self-embedding-nn-embedding-vocab_size-embedding_size","seqnseqdecoder.forward.embeddings-self-embedding-decoder_input_ids"],
              jax: ["seqnseqdecoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-target_embe"],
            },
            focusRef: {
              pytorch: "seqnseqdecoder.self-embedding-nn-embedding-vocab_size-embedding_size",
              jax: "seqnseqdecoder.__call__.embeddings-nn-embed-self-vocab_size-self-embedding_size-name-target_embe",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["seqnseqdecoder.self-output_projection-nn-linear-hidden_size-vocab_size","seqnseqdecoder.forward.logits-self-output_projection-h","seqnseqdecoder.forward.output_logits-torch-stack-logits_per_step-dim-n","seqnseqdecoder.forward.decoder_trace-torch-stack-decoder_states-dim-n","seqnseq.forward.logits-decoder_outputs-n","seqnseq.forward.decoder_trace-decoder_outputs-n","seqnseq.forward.outputs-logits-encoder_trace-decoder_trace"],
          jax: ["seqnseqdecoder.__call__.output_projection-nn-dense-self-vocab_size-name-output_projection","seqnseqdecoder.__call__.logits-output_projection-h","seqnseqdecoder.__call__.output_logits-jnp-stack-logits_per_step-axis-n","seqnseqdecoder.__call__.decoder_trace-jnp-stack-decoder_states-axis-n","seqnseq.__call__.logits-decoder_outputs-n","seqnseq.__call__.decoder_trace-decoder_outputs-n","seqnseq.__call__.outputs-logits-encoder_trace-decoder_trace"],
        },
        focusRef: {
          pytorch: "seqnseqdecoder.self-output_projection-nn-linear-hidden_size-vocab_size",
          jax: "seqnseqdecoder.__call__.output_projection-nn-dense-self-vocab_size-name-output_projection",
        },
        includeChildRefs: false,
      },
    ],
  },
  lstm: {
    stats: "Sequence classifier · (batch, 8, 32) input · logits + state trace · PyTorch/JAX notebooks",
    nodes: [
      {
        id: "sequence",
        label: "sequence",
        type: "SequenceInput",
        kind: "input",
        badges: ["8 steps", "32 features"],
        sourceRefs: {
          pytorch: ["lstmsequence.forward.next_state-self-cell-current_input-previous_state"],
          jax: ["lstmsequence.__call__.next_state-cell-current_input-previous_state"],
        },
        focusRef: {
          pytorch: "lstmsequence.forward.next_state-self-cell-current_input-previous_state",
          jax: "lstmsequence.__call__.next_state-cell-current_input-previous_state",
        },
        includeChildRefs: false,
      },
      {
        id: "cell_params",
        label: "LSTM Cell Params",
        type: "GatedMemoryCell",
        kind: "group",
        summary: "8 affine projections",
        badges: ["i", "f", "g", "o"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["lstmcell.def-__init__","lstmcell.self","lstmcell.self-x_i-nn-linear-input_size-hidden_size","lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_f-nn-linear-input_size-hidden_size","lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_g-nn-linear-input_size-hidden_size","lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_o-nn-linear-input_size-hidden_size","lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false"],
          jax: ["class-lstmcell-nn-module","lstmcell.hidden_size-int-n","lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h","lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h","lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h","lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h"],
        },
        focusRef: {
          pytorch: "lstmcell.def-__init__",
          jax: "class-lstmcell-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "cell_params.input",
            label: "input params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_i", "h_i"],
            sourceRefs: {
              pytorch: ["lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_f-nn-linear-input_size-hidden_size","lstmcell.forward.h_i-self-h_i-h","lstmcell.forward.i_pre-x_i-h_i"],
              jax: ["lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h"],
            },
            focusRef: {
              pytorch: "lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false",
              jax: "lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x",
            },
            includeChildRefs: false,
          },
          {
            id: "cell_params.forget",
            label: "forget params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_f", "h_f"],
            sourceRefs: {
              pytorch: ["lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_g-nn-linear-input_size-hidden_size","lstmcell.forward.h_f-self-h_f-h","lstmcell.forward.f_pre-x_f-h_f"],
              jax: ["lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h"],
            },
            focusRef: {
              pytorch: "lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false",
              jax: "lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x",
            },
            includeChildRefs: false,
          },
          {
            id: "cell_params.candidate",
            label: "candidate params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_g", "h_g"],
            sourceRefs: {
              pytorch: ["lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_o-nn-linear-input_size-hidden_size","lstmcell.forward.h_g-self-h_g-h","lstmcell.forward.g_pre-x_g-h_g"],
              jax: ["lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h"],
            },
            focusRef: {
              pytorch: "lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false",
              jax: "lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x",
            },
            includeChildRefs: false,
          },
          {
            id: "cell_params.output",
            label: "output params",
            type: "Linear pairs",
            kind: "recurrent",
            badges: ["x_o", "h_o"],
            sourceRefs: {
              pytorch: ["lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.forward.h_o-self-h_o-h","lstmcell.forward.o_pre-x_o-h_o"],
              jax: ["lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h"],
            },
            focusRef: {
              pytorch: "lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false",
              jax: "lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["lstmsequence.forward.batch_size-x-size-n","lstmsequence.forward.h-torch-zeros-hidden_shape-device-x-device","lstmsequence.forward.c-torch-zeros-hidden_shape-device-x-device","lstmsequence.forward.for-t-in-range-step_count","lstmsequence.forward.current_input-x-t","lstmsequence.forward.previous_state-h-c","lstmsequence.forward.next_state-self-cell-current_input-previous_state","lstmsequence.forward.h-next_state-n","lstmsequence.forward.c-next_state-n","lstmsequence.forward.states-append-h"],
          jax: ["lstmsequence.__call__.batch_size-x-shape-n","lstmsequence.__call__.hidden_shape-batch_size-self-hidden_size","lstmsequence.__call__.h-jnp-zeros-hidden_shape","lstmsequence.__call__.c-jnp-zeros-hidden_shape","lstmsequence.__call__.states","lstmsequence.__call__.cell-lstmcell-self-hidden_size","lstmsequence.__call__.step_count-x-shape-n","lstmsequence.__call__.for-t-in-range-step_count","lstmsequence.__call__.current_input-x-t","lstmsequence.__call__.previous_state-h-c","lstmsequence.__call__.next_state-cell-current_input-previous_state","lstmsequence.__call__.h-next_state-n","lstmsequence.__call__.c-next_state-n","lstmsequence.__call__.states-append-h"],
        },
        focusRef: {
          pytorch: "lstmsequence.forward.batch_size-x-size-n",
          jax: "lstmsequence.__call__.batch_size-x-shape-n",
        },
        includeChildRefs: false,
        children: [
          {
            id: "state0",
            label: "h0 + c0",
            type: "ZeroStates",
            kind: "recurrent",
            badges: ["64 hidden", "64 cell"],
            sourceRefs: {
              pytorch: ["lstmsequence.forward.h-torch-zeros-hidden_shape-device-x-device","lstmsequence.forward.c-torch-zeros-hidden_shape-device-x-device"],
              jax: ["lstmsequence.__call__.batch_size-x-shape-n","lstmsequence.__call__.hidden_shape-batch_size-self-hidden_size","lstmsequence.__call__.h-jnp-zeros-hidden_shape","lstmsequence.__call__.c-jnp-zeros-hidden_shape"],
            },
            focusRef: {
              pytorch: "lstmsequence.forward.h-torch-zeros-hidden_shape-device-x-device",
              jax: "lstmsequence.__call__.batch_size-x-shape-n",
            },
            includeChildRefs: false,
          },
          {
            id: "step.0",
            label: "step.0",
            type: "LSTMCell",
            kind: "group",
            summary: "i/f/g/o gates",
            defaultExpanded: true,
            sourceRefs: {
              pytorch: ["lstmcell.forward.h-c-state","lstmcell.forward.x_i-self-x_i-x","lstmcell.forward.h_i-self-h_i-h","lstmcell.forward.i_pre-x_i-h_i","lstmcell.forward.i-torch-sigmoid-i_pre","lstmcell.forward.x_f-self-x_f-x","lstmcell.forward.h_f-self-h_f-h","lstmcell.forward.f_pre-x_f-h_f","lstmcell.forward.f-torch-sigmoid-f_pre","lstmcell.forward.x_g-self-x_g-x","lstmcell.forward.h_g-self-h_g-h","lstmcell.forward.g_pre-x_g-h_g","lstmcell.forward.g-torch-tanh-g_pre","lstmcell.forward.x_o-self-x_o-x","lstmcell.forward.h_o-self-h_o-h","lstmcell.forward.o_pre-x_o-h_o","lstmcell.forward.o-torch-sigmoid-o_pre","lstmcell.forward.forget_c-f-c","lstmcell.forward.write_c-i-g","lstmcell.forward.c_next-forget_c-write_c","lstmcell.forward.c_readout-torch-tanh-c_next","lstmcell.forward.h_next-o-c_readout","lstmcell.forward.next_state-h_next-c_next","lstmcell.forward.return-next_state","lstmsequence.forward.c-next_state-n","lstmsequence.forward.states-append-h"],
              jax: ["class-lstmcell-nn-module","lstmcell.hidden_size-int-n","lstmcell.nn-compact","lstmcell.def-__call__-self-x-state","lstmcell.__call__.h-c-state","lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h","lstmcell.__call__.i_pre-x_i-h_i","lstmcell.__call__.i-nn-sigmoid-i_pre","lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h","lstmcell.__call__.f_pre-x_f-h_f","lstmcell.__call__.f-nn-sigmoid-f_pre","lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h","lstmcell.__call__.g_pre-x_g-h_g","lstmcell.__call__.g-jnp-tanh-g_pre","lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h","lstmcell.__call__.o_pre-x_o-h_o","lstmcell.__call__.o-nn-sigmoid-o_pre","lstmcell.__call__.forget_c-f-c","lstmcell.__call__.write_c-i-g","lstmcell.__call__.c_next-forget_c-write_c","lstmcell.__call__.c_readout-jnp-tanh-c_next","lstmcell.__call__.h_next-o-c_readout","lstmcell.__call__.next_state-h_next-c_next","lstmcell.__call__.return-next_state"],
            },
            focusRef: {
              pytorch: "lstmcell.forward.h-c-state",
              jax: "class-lstmcell-nn-module",
            },
            includeChildRefs: false,
            children: [
              {
                id: "step.0.input_gate",
                label: "input gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["i_t"],
                sourceRefs: {
                  pytorch: ["lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_f-nn-linear-input_size-hidden_size","lstmcell.forward.x_i-self-x_i-x","lstmcell.forward.h_i-self-h_i-h","lstmcell.forward.i_pre-x_i-h_i","lstmcell.forward.i-torch-sigmoid-i_pre"],
                  jax: ["lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h","lstmcell.__call__.i_pre-x_i-h_i","lstmcell.__call__.i-nn-sigmoid-i_pre"],
                },
                focusRef: {
                  pytorch: "lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false",
                  jax: "lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x",
                },
                includeChildRefs: false,
              },
              {
                id: "step.0.forget_gate",
                label: "forget gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["f_t"],
                sourceRefs: {
                  pytorch: ["lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_g-nn-linear-input_size-hidden_size","lstmcell.forward.x_f-self-x_f-x","lstmcell.forward.h_f-self-h_f-h","lstmcell.forward.f_pre-x_f-h_f","lstmcell.forward.f-torch-sigmoid-f_pre"],
                  jax: ["lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h","lstmcell.__call__.f_pre-x_f-h_f","lstmcell.__call__.f-nn-sigmoid-f_pre"],
                },
                focusRef: {
                  pytorch: "lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false",
                  jax: "lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x",
                },
                includeChildRefs: false,
              },
              {
                id: "step.0.candidate",
                label: "candidate",
                type: "TanhState",
                kind: "activation",
                badges: ["g_t"],
                sourceRefs: {
                  pytorch: ["lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.self-x_o-nn-linear-input_size-hidden_size","lstmcell.forward.x_g-self-x_g-x","lstmcell.forward.h_g-self-h_g-h","lstmcell.forward.g_pre-x_g-h_g","lstmcell.forward.g-torch-tanh-g_pre"],
                  jax: ["lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h","lstmcell.__call__.g_pre-x_g-h_g","lstmcell.__call__.g-jnp-tanh-g_pre"],
                },
                focusRef: {
                  pytorch: "lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false",
                  jax: "lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x",
                },
                includeChildRefs: false,
              },
              {
                id: "step.0.cell_update",
                label: "cell update",
                type: "MemoryUpdate",
                kind: "recurrent",
                badges: ["c_t"],
                sourceRefs: {
                  pytorch: ["lstmcell.forward.forget_c-f-c","lstmcell.forward.write_c-i-g","lstmcell.forward.c_next-forget_c-write_c"],
                  jax: ["lstmcell.__call__.forget_c-f-c","lstmcell.__call__.write_c-i-g","lstmcell.__call__.c_next-forget_c-write_c"],
                },
                focusRef: {
                  pytorch: "lstmcell.forward.forget_c-f-c",
                  jax: "lstmcell.__call__.forget_c-f-c",
                },
                includeChildRefs: false,
              },
              {
                id: "step.0.output_gate",
                label: "output gate",
                type: "SigmoidGate",
                kind: "recurrent",
                badges: ["o_t"],
                sourceRefs: {
                  pytorch: ["lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false","lstmcell.forward.x_o-self-x_o-x","lstmcell.forward.h_o-self-h_o-h","lstmcell.forward.o_pre-x_o-h_o","lstmcell.forward.o-torch-sigmoid-o_pre"],
                  jax: ["lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h","lstmcell.__call__.o_pre-x_o-h_o","lstmcell.__call__.o-nn-sigmoid-o_pre"],
                },
                focusRef: {
                  pytorch: "lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false",
                  jax: "lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x",
                },
                includeChildRefs: false,
              },
              {
                id: "step.0.hidden_update",
                label: "hidden update",
                type: "GatedReadout",
                kind: "recurrent",
                badges: ["h_t"],
                sourceRefs: {
                  pytorch: ["lstmcell.forward.c_readout-torch-tanh-c_next","lstmcell.forward.h_next-o-c_readout","lstmcell.forward.next_state-h_next-c_next","lstmcell.forward.return-next_state"],
                  jax: ["lstmcell.__call__.c_readout-jnp-tanh-c_next","lstmcell.__call__.h_next-o-c_readout"],
                },
                focusRef: {
                  pytorch: "lstmcell.forward.c_readout-torch-tanh-c_next",
                  jax: "lstmcell.__call__.c_readout-jnp-tanh-c_next",
                },
                includeChildRefs: false,
              },
            ],
          },
          ...[1, 2, 3, 4, 5, 6, 7].map((index) => ({
            id: `step.${index}`,
            label: `step.${index}`,
            type: "LSTMCell",
            kind: "recurrent" as NodeKind,
            summary: "same gates",
            sourceRefs: {
              pytorch: ["lstmsequence.forward.previous_state-h-c","lstmsequence.forward.next_state-self-cell-current_input-previous_state","lstmsequence.forward.h-next_state-n","lstmsequence.forward.c-next_state-n","lstmsequence.forward.states-append-h"],
              jax: ["class-lstmcell-nn-module","lstmcell.hidden_size-int-n","lstmcell.nn-compact","lstmcell.def-__call__-self-x-state","lstmcell.__call__.h-c-state","lstmcell.__call__.x_i-nn-dense-self-hidden_size-name-x_i-x","lstmcell.__call__.h_i-nn-dense-self-hidden_size-use_bias-false-name-h_i-h","lstmcell.__call__.i_pre-x_i-h_i","lstmcell.__call__.i-nn-sigmoid-i_pre","lstmcell.__call__.x_f-nn-dense-self-hidden_size-name-x_f-x","lstmcell.__call__.h_f-nn-dense-self-hidden_size-use_bias-false-name-h_f-h","lstmcell.__call__.f_pre-x_f-h_f","lstmcell.__call__.f-nn-sigmoid-f_pre","lstmcell.__call__.x_g-nn-dense-self-hidden_size-name-x_g-x","lstmcell.__call__.h_g-nn-dense-self-hidden_size-use_bias-false-name-h_g-h","lstmcell.__call__.g_pre-x_g-h_g","lstmcell.__call__.g-jnp-tanh-g_pre","lstmcell.__call__.x_o-nn-dense-self-hidden_size-name-x_o-x","lstmcell.__call__.h_o-nn-dense-self-hidden_size-use_bias-false-name-h_o-h","lstmcell.__call__.o_pre-x_o-h_o","lstmcell.__call__.o-nn-sigmoid-o_pre","lstmcell.__call__.forget_c-f-c","lstmcell.__call__.write_c-i-g","lstmcell.__call__.c_next-forget_c-write_c","lstmcell.__call__.c_readout-jnp-tanh-c_next","lstmcell.__call__.h_next-o-c_readout","lstmcell.__call__.next_state-h_next-c_next","lstmcell.__call__.return-next_state"],
            },
            focusRef: {
              pytorch: "lstmsequence.forward.previous_state-h-c",
              jax: "class-lstmcell-nn-module",
            },
            includeChildRefs: false,
          })),
        ],
      },
      {
        id: "readout",
        label: "readout",
        type: "Linear",
        kind: "linear",
        badges: ["64->10", "last h"],
        sourceRefs: {
          pytorch: ["lstmsequence.def-forward-self-x","lstmsequence.forward.state_trace-torch-stack-states-dim-n","lstmsequence.forward.outputs-logits-state_trace"],
          jax: ["lstmsequence.__call__.logits-nn-dense-self-output_size-name-readout-h"],
        },
        focusRef: {
          pytorch: "lstmsequence.def-forward-self-x",
          jax: "lstmsequence.__call__.logits-nn-dense-self-output_size-name-readout-h",
        },
        includeChildRefs: false,
      },
      {
        id: "outputs",
        label: "outputs",
        type: "Logits + StateTrace",
        kind: "head",
        badges: ["classes", "hidden states"],
        sourceRefs: {
          pytorch: ["lstmsequence.forward.outputs-logits-state_trace","lstmsequence.forward.return-outputs"],
          jax: ["lstmsequence.__call__.logits-nn-dense-self-output_size-name-readout-h","lstmsequence.__call__.state_trace-jnp-stack-states-axis-n","lstmsequence.__call__.outputs-logits-state_trace","lstmsequence.__call__.return-outputs"],
        },
        focusRef: {
          pytorch: "lstmsequence.forward.outputs-logits-state_trace",
          jax: "lstmsequence.__call__.logits-nn-dense-self-output_size-name-readout-h",
        },
        includeChildRefs: false,
      },
    ],
  },
  autoencoder: {
    stats: "encoder · 32-d bottleneck · decoder · reconstruction loss",
    nodes: [
      {
        id: "input",
        label: "input x",
        type: "FlatVector",
        kind: "input",
        badges: ["784"],
        sourceRefs: {
          pytorch: ["encoder.def-forward-self-x","encoder.forward.z-self-net-x","autoencoder.def-forward-self-x","autoencoder.forward.z-self-encode-x","inputs-torch-zeros-n-n","reconstruction-latent_codes-model-inputs"],
          jax: ["encoder.def-__call__-self-x","encoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-x","autoencoder.def-__call__-self-x","autoencoder.__call__.z-self-encode-x","inputs-jnp-zeros-n-n","train_step.loss_fn.reconstruction-latent_codes-model-apply-current_params-inputs"],
        },
        focusRef: {
          pytorch: "encoder.def-forward-self-x",
          jax: "encoder.def-__call__-self-x",
        },
        includeChildRefs: false,
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "Compression MLP",
        kind: "group",
        summary: "input -> latent code",
        badges: ["784->256->32"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-encoder-nn-module","encoder.self-net-nn-sequential","encoder.nn-linear-input_dim-hidden_dim","encoder.nn-relu","encoder.nn-linear-hidden_dim-latent_dim","encoder.def-forward-self-x","encoder.forward.z-self-net-x","autoencoder.self-encoder-encoder-input_dim-hidden_dim-latent_dim","autoencoder.encode.z-self-encoder-x","autoencoder.forward.z-self-encode-x"],
          jax: ["class-encoder-nn-module","encoder.def-__call__-self-x","encoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-x","encoder.__call__.x-nn-relu-x","encoder.__call__.z-nn-dense-self-latent_dim-name-fcn-x","autoencoder.setup.self-encoder-encoder-self-input_dim-self-hidden_dim-self-latent_dim","autoencoder.encode.z-self-encoder-x","autoencoder.__call__.z-self-encode-x"],
        },
        focusRef: {
          pytorch: "class-encoder-nn-module",
          jax: "class-encoder-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "encoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["784->256"],
            sourceRefs: {
              pytorch: ["encoder.self-net-nn-sequential","encoder.nn-linear-input_dim-hidden_dim","encoder.nn-relu"],
              jax: ["encoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-x","encoder.__call__.x-nn-relu-x"],
            },
            focusRef: {
              pytorch: "encoder.self-net-nn-sequential",
              jax: "encoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "encoder.bottleneck",
            label: "bottleneck z",
            type: "LatentCode",
            kind: "embedding",
            badges: ["32"],
            sourceRefs: {
              pytorch: ["encoder.nn-linear-hidden_dim-latent_dim","encoder.forward.z-self-net-x","encoder.forward.return-z","autoencoder.encode.z-self-encoder-x","autoencoder.forward.z-self-encode-x","autoencoder.forward.return-reconstruction-z","final_latent_codes-latent_codes-detach"],
              jax: ["encoder.__call__.z-nn-dense-self-latent_dim-name-fcn-x","encoder.__call__.return-z","autoencoder.encode.z-self-encoder-x","autoencoder.__call__.z-self-encode-x","autoencoder.__call__.return-reconstruction-z","final_latent_codes-latent_codes"],
            },
            focusRef: {
              pytorch: "encoder.nn-linear-hidden_dim-latent_dim",
              jax: "encoder.__call__.z-nn-dense-self-latent_dim-name-fcn-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["class-decoder-nn-module","decoder.self-net-nn-sequential","decoder.nn-linear-latent_dim-hidden_dim","decoder.nn-relu","decoder.nn-linear-hidden_dim-output_dim","decoder.nn-sigmoid","decoder.def-forward-self-z","decoder.forward.reconstruction-self-net-z","autoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim","autoencoder.decode.reconstruction-self-decoder-z","autoencoder.forward.reconstruction-self-decode-z"],
          jax: ["class-decoder-nn-module","decoder.def-__call__-self-z","decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z","decoder.__call__.x-nn-relu-x","decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x","decoder.__call__.reconstruction-nn-sigmoid-x","autoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim","autoencoder.decode.reconstruction-self-decoder-z","autoencoder.__call__.reconstruction-self-decode-z"],
        },
        focusRef: {
          pytorch: "class-decoder-nn-module",
          jax: "class-decoder-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "decoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["32->256"],
            sourceRefs: {
              pytorch: ["decoder.self-net-nn-sequential","decoder.nn-linear-latent_dim-hidden_dim","decoder.nn-relu"],
              jax: ["decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z","decoder.__call__.x-nn-relu-x"],
            },
            focusRef: {
              pytorch: "decoder.self-net-nn-sequential",
              jax: "decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z",
            },
            includeChildRefs: false,
          },
          {
            id: "decoder.output",
            label: "reconstruction",
            type: "Sigmoid output",
            kind: "activation",
            badges: ["784"],
            sourceRefs: {
              pytorch: ["decoder.nn-linear-hidden_dim-output_dim","decoder.nn-sigmoid","decoder.forward.reconstruction-self-net-z","decoder.forward.return-reconstruction","autoencoder.decode.reconstruction-self-decoder-z","autoencoder.forward.reconstruction-self-decode-z","autoencoder.forward.return-reconstruction-z"],
              jax: ["decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x","decoder.__call__.reconstruction-nn-sigmoid-x","decoder.__call__.return-reconstruction","autoencoder.decode.reconstruction-self-decoder-z","autoencoder.__call__.reconstruction-self-decode-z","autoencoder.__call__.return-reconstruction-z"],
            },
            focusRef: {
              pytorch: "decoder.nn-linear-hidden_dim-output_dim",
              jax: "decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["for-step-in-range-n","optimizer-zero_grad","reconstruction-latent_codes-model-inputs","loss-f-mse_loss-reconstruction-inputs","loss-backward","optimizer-step"],
          jax: ["def-train_step-params-inputs-learning_rate-n","train_step.def-loss_fn-current_params","train_step.loss_fn.reconstruction-latent_codes-model-apply-current_params-inputs","train_step.loss_fn.loss-jnp-mean-reconstruction-inputs-n","train_step.loss-latent_codes-grads-jax-value_and_grad-loss_fn-has_aux-true-params","train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads","for-step-in-range-n","params-loss-latent_codes-train_step-params-inputs"],
        },
        focusRef: {
          pytorch: "for-step-in-range-n",
          jax: "def-train_step-params-inputs-learning_rate-n",
        },
        includeChildRefs: false,
      },
    ],
  },
  lenet5: {
    stats: "3 groups · 11 ops",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["1 x 32 x 32"],
        sourceRefs: {
          pytorch: ["lenetn.forward.convn-self-convn-x"],
          jax: ["lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x"],
        },
        focusRef: {
          pytorch: "lenetn.forward.convn-self-convn-x",
          jax: "lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x",
        },
        includeChildRefs: false,
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "Group",
        kind: "group",
        summary: "6 ops",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: [],
          jax: [],
        },
        focusRef: {
          pytorch: "lenetn.forward.convn-self-convn-x",
          jax: "lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x",
        },
        includeChildRefs: true,
        children: [
          {
            id: "features.conv1",
            label: "conv1",
            type: "Conv2d",
            kind: "conv",
            badges: ["1->6", "k=5", "out 6x28x28"],
            sourceRefs: {
              pytorch: ["lenetn.forward.convn-self-convn-x"],
              jax: ["lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x"],
            },
            focusRef: {
              pytorch: "lenetn.forward.convn-self-convn-x",
              jax: "lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x",
            },
            includeChildRefs: false,
          },
          {
            id: "features.tanh1",
            label: "tanh1",
            type: "Tanh",
            kind: "activation",
            sourceRefs: {
              pytorch: ["lenetn.forward.x-torch-tanh-convn"],
              jax: ["lenetn.__call__.x-jnp-tanh-convn"],
            },
            focusRef: {
              pytorch: "lenetn.forward.x-torch-tanh-convn",
              jax: "lenetn.__call__.x-jnp-tanh-convn",
            },
            includeChildRefs: false,
          },
          {
            id: "features.pool1",
            label: "pool1",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["k=2", "out 6x14x14"],
            sourceRefs: {
              pytorch: ["lenetn.forward.x-f-avg_poolnd-x-kernel_size-n"],
              jax: ["lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n"],
            },
            focusRef: {
              pytorch: "lenetn.forward.x-f-avg_poolnd-x-kernel_size-n",
              jax: "lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "features.conv2",
            label: "conv2",
            type: "Conv2d",
            kind: "conv",
            badges: ["6->16", "k=5", "out 16x10x10"],
            sourceRefs: {
              pytorch: ["lenetn.forward.convn-self-convn-x.2"],
              jax: ["lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x.2"],
            },
            focusRef: {
              pytorch: "lenetn.forward.convn-self-convn-x.2",
              jax: "lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "features.tanh2",
            label: "tanh2",
            type: "Tanh",
            kind: "activation",
            sourceRefs: {
              pytorch: ["lenetn.forward.x-torch-tanh-convn.2"],
              jax: ["lenetn.__call__.x-jnp-tanh-convn.2"],
            },
            focusRef: {
              pytorch: "lenetn.forward.x-torch-tanh-convn.2",
              jax: "lenetn.__call__.x-jnp-tanh-convn.2",
            },
            includeChildRefs: false,
          },
          {
            id: "features.pool2",
            label: "pool2",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["k=2", "out 16x5x5"],
            sourceRefs: {
              pytorch: ["lenet5.features.pool2"],
              jax: ["lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n.2"],
            },
            focusRef: {
              pytorch: "lenet5.features.pool2",
              jax: "lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n.2",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["400"],
        sourceRefs: {
          pytorch: ["lenet5.flatten"],
          jax: ["lenetn.__call__.x-x-reshape-flat_shape"],
        },
        focusRef: {
          pytorch: "lenet5.flatten",
          jax: "lenetn.__call__.x-x-reshape-flat_shape",
        },
        includeChildRefs: false,
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "Group",
        kind: "group",
        summary: "4 ops",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["lenetn.__init__.self-fcn-nn-linear-flattened_features-n","lenetn.__init__.self-fcn-nn-linear-n-n","lenetn.__init__.self-output-nn-linear-n-n"],
          jax: ["lenetn.__call__.fcn-nn-dense-features-n-x","lenetn.__call__.x-jnp-tanh-fcn","lenetn.__call__.fcn-nn-dense-features-n-x.2","lenetn.__call__.x-jnp-tanh-fcn.2","lenetn.__call__.logits-nn-dense-features-n-x"],
        },
        focusRef: {
          pytorch: "lenetn.__init__.self-fcn-nn-linear-flattened_features-n",
          jax: "lenetn.__call__.fcn-nn-dense-features-n-x",
        },
        includeChildRefs: false,
        children: [
          {
            id: "classifier.fc1",
            label: "fc1",
            type: "Linear",
            kind: "linear",
            badges: ["400->120"],
            sourceRefs: {
              pytorch: ["lenetn.__init__.self-fcn-nn-linear-flattened_features-n","lenet5.classifier.fc1"],
              jax: ["lenetn.__call__.fcn-nn-dense-features-n-x"],
            },
            focusRef: {
              pytorch: "lenetn.__init__.self-fcn-nn-linear-flattened_features-n",
              jax: "lenetn.__call__.fcn-nn-dense-features-n-x",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.tanh3",
            label: "tanh3",
            type: "Tanh",
            kind: "activation",
            sourceRefs: {
              pytorch: ["lenet5.classifier.tanh3"],
              jax: ["lenetn.__call__.x-jnp-tanh-fcn"],
            },
            focusRef: {
              pytorch: "lenet5.classifier.tanh3",
              jax: "lenetn.__call__.x-jnp-tanh-fcn",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.fc2",
            label: "fc2",
            type: "Linear",
            kind: "linear",
            badges: ["120->84"],
            sourceRefs: {
              pytorch: ["lenetn.__init__.self-fcn-nn-linear-n-n","lenet5.classifier.fc2"],
              jax: ["lenetn.__call__.fcn-nn-dense-features-n-x.2"],
            },
            focusRef: {
              pytorch: "lenetn.__init__.self-fcn-nn-linear-n-n",
              jax: "lenetn.__call__.fcn-nn-dense-features-n-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.output",
            label: "output",
            type: "Linear",
            kind: "linear",
            badges: ["84->10"],
            sourceRefs: {
              pytorch: ["lenetn.__init__.self-output-nn-linear-n-n","lenet5.classifier.output"],
              jax: ["lenetn.__call__.logits-nn-dense-features-n-x"],
            },
            focusRef: {
              pytorch: "lenetn.__init__.self-output-nn-linear-n-n",
              jax: "lenetn.__call__.logits-nn-dense-features-n-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  alexnet: {
    stats: "5 conv layers · 3 FC layers · 60M params",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Input",
        kind: "input",
        badges: ["3 x 227 x 227"],
        sourceRefs: {
          pytorch: ["alexnet.forward.x-self-features-x"],
          jax: ["alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x"],
        },
        focusRef: {
          pytorch: "alexnet.forward.x-self-features-x",
          jax: "alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x",
        },
        includeChildRefs: false,
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "Group",
        kind: "group",
        summary: "13 ops",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["alexnet.nn-convnd-n-n-kernel_size-n-stride-n","alexnet.nn-relu-inplace-true","alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n","alexnet.nn-maxpoolnd-kernel_size-n-stride-n","alexnet.nn-convnd-n-n-kernel_size-n-padding-n","alexnet.nn-relu-inplace-true.2","alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n.2","alexnet.nn-maxpoolnd-kernel_size-n-stride-n.2","alexnet.nn-convnd-n-n-kernel_size-n-padding-n.2","alexnet.nn-relu-inplace-true.3","alexnet.nn-convnd-n-n-kernel_size-n-padding-n.3","alexnet.nn-relu-inplace-true.4","alexnet.nn-convnd-n-n-kernel_size-n-padding-n.4","alexnet.nn-relu-inplace-true.5","alexnet.nn-maxpoolnd-kernel_size-n-stride-n.3","alexnet.code.4"],
          jax: ["alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x","alexnet.__call__.x-nn-relu-x","alexnet.__call__.x-local_response_norm-x","alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n","alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x","alexnet.__call__.x-nn-relu-x.2","alexnet.__call__.x-local_response_norm-x.2","alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.2","alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.2","alexnet.__call__.x-nn-relu-x.3","alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.3","alexnet.__call__.x-nn-relu-x.4","alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.4","alexnet.__call__.x-nn-relu-x.5","alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3"],
        },
        focusRef: {
          pytorch: "alexnet.nn-convnd-n-n-kernel_size-n-stride-n",
          jax: "alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x",
        },
        includeChildRefs: false,
        children: [
          {
            id: "features.conv1",
            label: "conv1",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->96", "k=11", "s=4", "55x55"],
            sourceRefs: {
              pytorch: ["alexnet.nn-relu-inplace-true"],
              jax: ["alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x"],
            },
            focusRef: {
              pytorch: "alexnet.nn-relu-inplace-true",
              jax: "alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "features.relu1",
            label: "relu1",
            type: "ReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n"],
              jax: ["alexnet.__call__.x-nn-relu-x"],
            },
            focusRef: {
              pytorch: "alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n",
              jax: "alexnet.__call__.x-nn-relu-x",
            },
            includeChildRefs: false,
          },
          {
            id: "features.lrn1",
            label: "lrn1",
            type: "LocalResponseNorm",
            kind: "norm",
            badges: ["size=5"],
            sourceRefs: {
              pytorch: ["alexnet.nn-maxpoolnd-kernel_size-n-stride-n"],
              jax: ["def-local_response_norm-x-size-n-alpha-ne-n-beta-n-k-n","local_response_norm.half-size-n","local_response_norm.squared-jnp-square-x","local_response_norm.padded-jnp-pad-squared-n-n-n-n-n-n-half-half","local_response_norm.scale-k","local_response_norm.for-offset-in-range-size","local_response_norm.channel_end-offset-x-shape-n","local_response_norm.window-padded-offset-channel_end","local_response_norm.scale_step-alpha-size-window","local_response_norm.scale-scale-scale_step","local_response_norm.denominator-jnp-power-scale-beta","local_response_norm.normalized-x-denominator","local_response_norm.return-normalized","alexnet.__call__.x-local_response_norm-x"],
            },
            focusRef: {
              pytorch: "alexnet.nn-maxpoolnd-kernel_size-n-stride-n",
              jax: "def-local_response_norm-x-size-n-alpha-ne-n-beta-n-k-n",
            },
            includeChildRefs: false,
          },
          {
            id: "features.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "27x27"],
            sourceRefs: {
              pytorch: ["alexnet.nn-convnd-n-n-kernel_size-n-padding-n"],
              jax: ["alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n"],
            },
            focusRef: {
              pytorch: "alexnet.nn-convnd-n-n-kernel_size-n-padding-n",
              jax: "alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "features.conv2",
            label: "conv2",
            type: "Conv2d",
            kind: "conv",
            badges: ["96->256", "k=5", "27x27"],
            sourceRefs: {
              pytorch: ["alexnet.nn-relu-inplace-true.2"],
              jax: ["alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x"],
            },
            focusRef: {
              pytorch: "alexnet.nn-relu-inplace-true.2",
              jax: "alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "features.relu2",
            label: "relu2",
            type: "ReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n.2"],
              jax: ["alexnet.__call__.x-nn-relu-x.2"],
            },
            focusRef: {
              pytorch: "alexnet.nn-localresponsenorm-size-n-alpha-ne-n-beta-n-k-n.2",
              jax: "alexnet.__call__.x-nn-relu-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "features.lrn2",
            label: "lrn2",
            type: "LocalResponseNorm",
            kind: "norm",
            badges: ["size=5"],
            sourceRefs: {
              pytorch: ["alexnet.nn-maxpoolnd-kernel_size-n-stride-n.2"],
              jax: ["def-local_response_norm-x-size-n-alpha-ne-n-beta-n-k-n","local_response_norm.half-size-n","local_response_norm.squared-jnp-square-x","local_response_norm.padded-jnp-pad-squared-n-n-n-n-n-n-half-half","local_response_norm.scale-k","local_response_norm.for-offset-in-range-size","local_response_norm.channel_end-offset-x-shape-n","local_response_norm.window-padded-offset-channel_end","local_response_norm.scale_step-alpha-size-window","local_response_norm.scale-scale-scale_step","local_response_norm.denominator-jnp-power-scale-beta","local_response_norm.normalized-x-denominator","local_response_norm.return-normalized","alexnet.__call__.x-local_response_norm-x.2"],
            },
            focusRef: {
              pytorch: "alexnet.nn-maxpoolnd-kernel_size-n-stride-n.2",
              jax: "def-local_response_norm-x-size-n-alpha-ne-n-beta-n-k-n",
            },
            includeChildRefs: false,
          },
          {
            id: "features.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "13x13"],
            sourceRefs: {
              pytorch: ["alexnet.nn-convnd-n-n-kernel_size-n-padding-n.2"],
              jax: ["alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.2"],
            },
            focusRef: {
              pytorch: "alexnet.nn-convnd-n-n-kernel_size-n-padding-n.2",
              jax: "alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.2",
            },
            includeChildRefs: false,
          },
          {
            id: "features.conv3",
            label: "conv3",
            type: "Conv2d",
            kind: "conv",
            badges: ["256->384", "k=3"],
            sourceRefs: {
              pytorch: ["alexnet.nn-relu-inplace-true.3"],
              jax: ["alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.2"],
            },
            focusRef: {
              pytorch: "alexnet.nn-relu-inplace-true.3",
              jax: "alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "features.relu3",
            label: "relu3",
            type: "ReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["alexnet.nn-convnd-n-n-kernel_size-n-padding-n.3"],
              jax: ["alexnet.__call__.x-nn-relu-x.3"],
            },
            focusRef: {
              pytorch: "alexnet.nn-convnd-n-n-kernel_size-n-padding-n.3",
              jax: "alexnet.__call__.x-nn-relu-x.3",
            },
            includeChildRefs: false,
          },
          {
            id: "features.conv4",
            label: "conv4",
            type: "Conv2d",
            kind: "conv",
            badges: ["384->384", "k=3"],
            sourceRefs: {
              pytorch: ["alexnet.nn-relu-inplace-true.4"],
              jax: ["alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.3"],
            },
            focusRef: {
              pytorch: "alexnet.nn-relu-inplace-true.4",
              jax: "alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.3",
            },
            includeChildRefs: false,
          },
          {
            id: "features.relu4",
            label: "relu4",
            type: "ReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["alexnet.nn-convnd-n-n-kernel_size-n-padding-n.4"],
              jax: ["alexnet.__call__.x-nn-relu-x.4"],
            },
            focusRef: {
              pytorch: "alexnet.nn-convnd-n-n-kernel_size-n-padding-n.4",
              jax: "alexnet.__call__.x-nn-relu-x.4",
            },
            includeChildRefs: false,
          },
          {
            id: "features.conv5",
            label: "conv5",
            type: "Conv2d",
            kind: "conv",
            badges: ["384->256", "k=3"],
            sourceRefs: {
              pytorch: ["alexnet.nn-relu-inplace-true.5"],
              jax: ["alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.4"],
            },
            focusRef: {
              pytorch: "alexnet.nn-relu-inplace-true.5",
              jax: "alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.4",
            },
            includeChildRefs: false,
          },
          {
            id: "features.relu5",
            label: "relu5",
            type: "ReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["alexnet.nn-maxpoolnd-kernel_size-n-stride-n.3"],
              jax: ["alexnet.__call__.x-nn-relu-x.5"],
            },
            focusRef: {
              pytorch: "alexnet.nn-maxpoolnd-kernel_size-n-stride-n.3",
              jax: "alexnet.__call__.x-nn-relu-x.5",
            },
            includeChildRefs: false,
          },
          {
            id: "features.pool5",
            label: "pool5",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["k=3", "s=2", "6x6"],
            sourceRefs: {
              pytorch: ["alexnet.nn-maxpoolnd-kernel_size-n-stride-n.3"],
              jax: ["alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3"],
            },
            focusRef: {
              pytorch: "alexnet.nn-maxpoolnd-kernel_size-n-stride-n.3",
              jax: "alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["9216"],
        sourceRefs: {
          pytorch: ["alexnet.forward.x-torch-flatten-x-start_dim-n"],
          jax: ["alexnet.__call__.x-x-reshape-flat_shape"],
        },
        focusRef: {
          pytorch: "alexnet.forward.x-torch-flatten-x-start_dim-n",
          jax: "alexnet.__call__.x-x-reshape-flat_shape",
        },
        includeChildRefs: false,
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "Group",
        kind: "group",
        summary: "7 ops",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["alexnet.self-classifier-nn-sequential"],
          jax: [],
        },
        focusRef: {
          pytorch: "alexnet.self-classifier-nn-sequential",
          jax: "alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x",
        },
        includeChildRefs: true,
        children: [
          {
            id: "classifier.drop1",
            label: "dropout1",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            sourceRefs: {
              pytorch: ["alexnet.nn-dropout-n"],
              jax: ["alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x"],
            },
            focusRef: {
              pytorch: "alexnet.nn-dropout-n",
              jax: "alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.fc6",
            label: "fc6",
            type: "Linear",
            kind: "linear",
            badges: ["9216->4096"],
            sourceRefs: {
              pytorch: ["alexnet.nn-linear-flattened_features-n"],
              jax: ["alexnet.__call__.x-nn-dense-features-n-name-fcn-x"],
            },
            focusRef: {
              pytorch: "alexnet.nn-linear-flattened_features-n",
              jax: "alexnet.__call__.x-nn-dense-features-n-name-fcn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.relu6",
            label: "relu6",
            type: "ReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["alexnet.nn-relu-inplace-true.6"],
              jax: ["alexnet.__call__.x-nn-relu-x.6"],
            },
            focusRef: {
              pytorch: "alexnet.nn-relu-inplace-true.6",
              jax: "alexnet.__call__.x-nn-relu-x.6",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.drop2",
            label: "dropout2",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            sourceRefs: {
              pytorch: ["alexnet.nn-dropout-n.2"],
              jax: ["alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x.2"],
            },
            focusRef: {
              pytorch: "alexnet.nn-dropout-n.2",
              jax: "alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.fc7",
            label: "fc7",
            type: "Linear",
            kind: "linear",
            badges: ["4096->4096"],
            sourceRefs: {
              pytorch: ["alexnet.nn-linear-n-n"],
              jax: ["alexnet.__call__.x-nn-dense-features-n-name-fcn-x.2"],
            },
            focusRef: {
              pytorch: "alexnet.nn-linear-n-n",
              jax: "alexnet.__call__.x-nn-dense-features-n-name-fcn-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.relu7",
            label: "relu7",
            type: "ReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["alexnet.nn-relu-inplace-true.7"],
              jax: ["alexnet.__call__.x-nn-relu-x.7"],
            },
            focusRef: {
              pytorch: "alexnet.nn-relu-inplace-true.7",
              jax: "alexnet.__call__.x-nn-relu-x.7",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.fc8",
            label: "fc8",
            type: "Linear",
            kind: "linear",
            badges: ["4096->1000"],
            sourceRefs: {
              pytorch: ["alexnet.nn-linear-n-num_classes"],
              jax: ["alexnet.__call__.logits-nn-dense-features-self-num_classes-name-fcn-x"],
            },
            focusRef: {
              pytorch: "alexnet.nn-linear-n-num_classes",
              jax: "alexnet.__call__.logits-nn-dense-features-self-num_classes-name-fcn-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  vgg16: {
    stats: "13 conv layers · 3 FC layers · stacked 3x3 filters",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        sourceRefs: {
          pytorch: ["vggn.forward.x-self-features-x","train_images-torch-zeros-n-n-n-n","logits-model-train_images"],
          jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x","train_images-jnp-zeros-n-n-n-n","train_step.loss_fn.logits-model-apply-current_params-inputs-train-false"],
        },
        focusRef: {
          pytorch: "vggn.forward.x-self-features-x",
          jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x",
        },
        includeChildRefs: false,
      },
      {
        id: "features",
        label: "Feature Extractor",
        type: "PlainConvStack",
        kind: "group",
        summary: "13 conv + 5 pools",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["vggn.self-features-nn-sequential","vggn.nn-convnd-n-n-kernel_size-n-padding-n","vggn.nn-convnd-n-n-kernel_size-n-padding-n.2","vggn.nn-maxpoolnd-kernel_size-n-stride-n","vggn.nn-convnd-n-n-kernel_size-n-padding-n.3","vggn.nn-convnd-n-n-kernel_size-n-padding-n.4","vggn.nn-maxpoolnd-kernel_size-n-stride-n.2","vggn.nn-convnd-n-n-kernel_size-n-padding-n.5","vggn.nn-convnd-n-n-kernel_size-n-padding-n.6","vggn.nn-convnd-n-n-kernel_size-n-padding-n.7","vggn.nn-maxpoolnd-kernel_size-n-stride-n.3","vggn.nn-convnd-n-n-kernel_size-n-padding-n.8","vggn.nn-convnd-n-n-kernel_size-n-padding-n.9","vggn.nn-convnd-n-n-kernel_size-n-padding-n.10","vggn.nn-maxpoolnd-kernel_size-n-stride-n.4","vggn.nn-convnd-n-n-kernel_size-n-padding-n.11","vggn.nn-convnd-n-n-kernel_size-n-padding-n.12","vggn.nn-convnd-n-n-kernel_size-n-padding-n.13","vggn.nn-maxpoolnd-kernel_size-n-stride-n.5","vggn.forward.x-self-features-x"],
          jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.2","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.3","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.4","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.2","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.5","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.6","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.7","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.8","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.9","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.10","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.4","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.11","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.12","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.13","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.5"],
        },
        focusRef: {
          pytorch: "vggn.self-features-nn-sequential",
          jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x",
        },
        includeChildRefs: false,
        children: [
          {
            id: "features.stage1",
            label: "stage1",
            type: "ConvBlock",
            kind: "group",
            summary: "2 convs",
            badges: ["64", "112x112"],
            sourceRefs: {
              pytorch: ["vggn.nn-convnd-n-n-kernel_size-n-padding-n","vggn.nn-relu-inplace-true","vggn.nn-convnd-n-n-kernel_size-n-padding-n.2","vggn.nn-relu-inplace-true.2","vggn.nn-maxpoolnd-kernel_size-n-stride-n"],
              jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x","vggn.__call__.x-nn-relu-x","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.2","vggn.__call__.x-nn-relu-x.2","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n"],
            },
            focusRef: {
              pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n",
              jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x",
            },
            includeChildRefs: false,
          },
          {
            id: "features.stage2",
            label: "stage2",
            type: "ConvBlock",
            kind: "group",
            summary: "2 convs",
            badges: ["128", "56x56"],
            sourceRefs: {
              pytorch: ["vggn.nn-convnd-n-n-kernel_size-n-padding-n.3","vggn.nn-relu-inplace-true.3","vggn.nn-convnd-n-n-kernel_size-n-padding-n.4","vggn.nn-relu-inplace-true.4","vggn.nn-maxpoolnd-kernel_size-n-stride-n.2"],
              jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.3","vggn.__call__.x-nn-relu-x.3","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.4","vggn.__call__.x-nn-relu-x.4","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.2"],
            },
            focusRef: {
              pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n.3",
              jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.3",
            },
            includeChildRefs: false,
          },
          {
            id: "features.stage3",
            label: "stage3",
            type: "ConvBlock",
            kind: "group",
            summary: "3 convs",
            badges: ["256", "28x28"],
            sourceRefs: {
              pytorch: ["vggn.nn-relu-inplace-true.5","vggn.nn-relu-inplace-true.6","vggn.nn-relu-inplace-true.7"],
              jax: ["vggn.__call__.x-nn-relu-x.5","vggn.__call__.x-nn-relu-x.6","vggn.__call__.x-nn-relu-x.7"],
            },
            focusRef: {
              pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n.5",
              jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.5",
            },
            includeChildRefs: true,
            children: [
              {
                id: "features.stage3.conv1",
                label: "conv3_1",
                type: "Conv2d",
                kind: "conv",
                badges: ["128->256", "k=3"],
                sourceRefs: {
                  pytorch: ["vggn.nn-convnd-n-n-kernel_size-n-padding-n.5"],
                  jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.5"],
                },
                focusRef: {
                  pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n.5",
                  jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.5",
                },
                includeChildRefs: false,
              },
              {
                id: "features.stage3.conv2",
                label: "conv3_2",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                sourceRefs: {
                  pytorch: ["vggn.nn-convnd-n-n-kernel_size-n-padding-n.6"],
                  jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.6"],
                },
                focusRef: {
                  pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n.6",
                  jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.6",
                },
                includeChildRefs: false,
              },
              {
                id: "features.stage3.conv3",
                label: "conv3_3",
                type: "Conv2d",
                kind: "conv",
                badges: ["256->256", "k=3"],
                sourceRefs: {
                  pytorch: ["vggn.nn-convnd-n-n-kernel_size-n-padding-n.7"],
                  jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.7"],
                },
                focusRef: {
                  pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n.7",
                  jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.7",
                },
                includeChildRefs: false,
              },
              {
                id: "features.stage3.pool",
                label: "pool3",
                type: "MaxPool2d",
                kind: "pool",
                badges: ["28x28"],
                sourceRefs: {
                  pytorch: ["vggn.nn-maxpoolnd-kernel_size-n-stride-n.3"],
                  jax: ["vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3"],
                },
                focusRef: {
                  pytorch: "vggn.nn-maxpoolnd-kernel_size-n-stride-n.3",
                  jax: "vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3",
                },
                includeChildRefs: false,
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
            sourceRefs: {
              pytorch: ["vggn.nn-convnd-n-n-kernel_size-n-padding-n.8","vggn.nn-relu-inplace-true.8","vggn.nn-convnd-n-n-kernel_size-n-padding-n.9","vggn.nn-relu-inplace-true.9","vggn.nn-convnd-n-n-kernel_size-n-padding-n.10","vggn.nn-relu-inplace-true.10","vggn.nn-maxpoolnd-kernel_size-n-stride-n.4"],
              jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.8","vggn.__call__.x-nn-relu-x.8","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.9","vggn.__call__.x-nn-relu-x.9","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.10","vggn.__call__.x-nn-relu-x.10","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.4"],
            },
            focusRef: {
              pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n.8",
              jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.8",
            },
            includeChildRefs: false,
          },
          {
            id: "features.stage5",
            label: "stage5",
            type: "ConvBlock",
            kind: "group",
            summary: "3 convs",
            badges: ["512", "7x7"],
            sourceRefs: {
              pytorch: ["vggn.nn-convnd-n-n-kernel_size-n-padding-n.11","vggn.nn-relu-inplace-true.11","vggn.nn-convnd-n-n-kernel_size-n-padding-n.12","vggn.nn-relu-inplace-true.12","vggn.nn-convnd-n-n-kernel_size-n-padding-n.13","vggn.nn-relu-inplace-true.13","vggn.nn-maxpoolnd-kernel_size-n-stride-n.5"],
              jax: ["vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.11","vggn.__call__.x-nn-relu-x.11","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.12","vggn.__call__.x-nn-relu-x.12","vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.13","vggn.__call__.x-nn-relu-x.13","vggn.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.5"],
            },
            focusRef: {
              pytorch: "vggn.nn-convnd-n-n-kernel_size-n-padding-n.11",
              jax: "vggn.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn_n-x.11",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "flatten",
        label: "flatten",
        type: "Flatten",
        kind: "reshape",
        badges: ["25088"],
        sourceRefs: {
          pytorch: ["vggn.forward.x-torch-flatten-x-start_dim-n"],
          jax: ["vggn.__call__.batch_size-x-shape-n","vggn.__call__.flat_shape-batch_size-n","vggn.__call__.x-x-reshape-flat_shape"],
        },
        focusRef: {
          pytorch: "vggn.forward.x-torch-flatten-x-start_dim-n",
          jax: "vggn.__call__.batch_size-x-shape-n",
        },
        includeChildRefs: false,
      },
      {
        id: "classifier",
        label: "Classifier",
        type: "DenseHead",
        kind: "group",
        summary: "4096 -> 4096 -> classes",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["vggn.flattened_features-n-n-n","vggn.self-classifier-nn-sequential","vggn.nn-relu-inplace-true.14","vggn.nn-relu-inplace-true.15"],
          jax: ["vggn.__call__.x-nn-relu-x.14","vggn.__call__.x-nn-relu-x.15"],
        },
        focusRef: {
          pytorch: "vggn.flattened_features-n-n-n",
          jax: "vggn.__call__.x-nn-dense-features-n-name-fcn-x",
        },
        includeChildRefs: true,
        children: [
          {
            id: "classifier.fc6",
            label: "fc6",
            type: "Linear",
            kind: "linear",
            badges: ["25088->4096"],
            sourceRefs: {
              pytorch: ["vggn.nn-linear-flattened_features-n","vggn.forward.logits-self-classifier-x"],
              jax: ["vggn.__call__.x-nn-dense-features-n-name-fcn-x"],
            },
            focusRef: {
              pytorch: "vggn.nn-linear-flattened_features-n",
              jax: "vggn.__call__.x-nn-dense-features-n-name-fcn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.drop6",
            label: "dropout6",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            sourceRefs: {
              pytorch: ["vggn.nn-dropout-n"],
              jax: ["vggn.__call__.x-nn-dropout-n-deterministic-not-train-x"],
            },
            focusRef: {
              pytorch: "vggn.nn-dropout-n",
              jax: "vggn.__call__.x-nn-dropout-n-deterministic-not-train-x",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.fc7",
            label: "fc7",
            type: "Linear",
            kind: "linear",
            badges: ["4096->4096"],
            sourceRefs: {
              pytorch: ["vggn.nn-linear-n-n"],
              jax: ["vggn.__call__.x-nn-dense-features-n-name-fcn-x.2"],
            },
            focusRef: {
              pytorch: "vggn.nn-linear-n-n",
              jax: "vggn.__call__.x-nn-dense-features-n-name-fcn-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.drop7",
            label: "dropout7",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.5"],
            sourceRefs: {
              pytorch: ["vggn.nn-dropout-n.2"],
              jax: ["vggn.__call__.x-nn-dropout-n-deterministic-not-train-x.2"],
            },
            focusRef: {
              pytorch: "vggn.nn-dropout-n.2",
              jax: "vggn.__call__.x-nn-dropout-n-deterministic-not-train-x.2",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.fc8",
            label: "fc8",
            type: "Linear",
            kind: "linear",
            badges: ["4096->1000"],
            sourceRefs: {
              pytorch: ["vggn.nn-linear-n-num_classes"],
              jax: ["vggn.__call__.logits-nn-dense-features-self-num_classes-name-fcn-x"],
            },
            focusRef: {
              pytorch: "vggn.nn-linear-n-num_classes",
              jax: "vggn.__call__.logits-nn-dense-features-self-num_classes-name-fcn-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  googlenet: {
    stats: "9 Inception blocks · parallel conv branches · 22 layers",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        sourceRefs: {
          pytorch: ["googlenet.input"],
          jax: ["googlenet.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-name-stem_convn-x"],
        },
        focusRef: {
          pytorch: "googlenet.input",
          jax: "googlenet.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-name-stem_convn-x",
        },
        includeChildRefs: false,
      },
      {
        id: "stem",
        label: "stem",
        type: "ConvPoolStem",
        kind: "group",
        summary: "7x7 + 1x1 + 3x3",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["googlenet.nn-relu-inplace-true","googlenet.nn-convnd-n-n-kernel_size-n","googlenet.nn-relu-inplace-true.3","googlenet.code.4"],
          jax: [],
        },
        focusRef: {
          pytorch: "googlenet.nn-relu-inplace-true",
          jax: "googlenet.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-name-stem_convn-x",
        },
        includeChildRefs: true,
        children: [
          {
            id: "stem.conv7",
            label: "conv7",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->64", "k=7", "s=2"],
            sourceRefs: {
              pytorch: ["googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n"],
              jax: ["googlenet.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-name-stem_convn-x","googlenet.__call__.x-nn-relu-x"],
            },
            focusRef: {
              pytorch: "googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n",
              jax: "googlenet.__call__.x-nn-conv-n-n-n-strides-n-n-padding-same-name-stem_convn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "stem.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            sourceRefs: {
              pytorch: ["googlenet.nn-relu-inplace-true.2"],
              jax: ["googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same"],
            },
            focusRef: {
              pytorch: "googlenet.nn-relu-inplace-true.2",
              jax: "googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
            },
            includeChildRefs: false,
          },
          {
            id: "stem.conv1",
            label: "conv1x1",
            type: "Conv2d",
            kind: "conv",
            badges: ["64->64"],
            sourceRefs: {
              pytorch: ["googlenet.nn-convnd-n-n-kernel_size-n-padding-n"],
              jax: ["googlenet.__call__.x-nn-conv-n-n-n-name-stem_convn-x","googlenet.__call__.x-nn-relu-x.2"],
            },
            focusRef: {
              pytorch: "googlenet.nn-convnd-n-n-kernel_size-n-padding-n",
              jax: "googlenet.__call__.x-nn-conv-n-n-n-name-stem_convn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "stem.conv3",
            label: "conv3x3",
            type: "Conv2d",
            kind: "conv",
            badges: ["64->192"],
            sourceRefs: {
              pytorch: ["googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2"],
              jax: ["googlenet.__call__.x-nn-conv-n-n-n-padding-same-name-stem_convn-x","googlenet.__call__.x-nn-relu-x.3"],
            },
            focusRef: {
              pytorch: "googlenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n.2",
              jax: "googlenet.__call__.x-nn-conv-n-n-n-padding-same-name-stem_convn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "stem.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            sourceRefs: {
              pytorch: ["googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n"],
              jax: ["googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.2"],
            },
            focusRef: {
              pytorch: "googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n",
              jax: "googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.2",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.2","googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n"],
          jax: ["googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x","googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x","googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.3"],
        },
        focusRef: {
          pytorch: "googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.2",
          jax: "googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x",
        },
        includeChildRefs: false,
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
          }),
          {
            id: "stage3.pool",
            label: "maxpool",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            sourceRefs: {
              pytorch: ["googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n"],
              jax: ["googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.3"],
            },
            focusRef: {
              pytorch: "googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n",
              jax: "googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.3",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n","googlenet.self-inceptionnd-inceptionblock-n-n-n-n-n-n-n","googlenet.self-inceptionne-inceptionblock-n-n-n-n-n-n-n","googlenet.self-inceptionna-inceptionblock-n-n-n-n-n-n-n.3","googlenet.self-inceptionnb-inceptionblock-n-n-n-n-n-n-n.3","googlenet.forward.x-self-inceptionnc-x","googlenet.forward.x-self-inceptionnd-x","googlenet.forward.x-self-inceptionne-x","googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n.2"],
          jax: ["googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.2","googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x.2","googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnc-x","googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnd-x","googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionne-x","googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.4"],
        },
        focusRef: {
          pytorch: "googlenet.self-inceptionnc-inceptionblock-n-n-n-n-n-n-n",
          jax: "googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.2",
        },
        includeChildRefs: false,
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
          }),
          {
            id: "stage4.pool",
            label: "maxpool",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["s=2"],
            sourceRefs: {
              pytorch: ["googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n.2"],
              jax: ["googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.4"],
            },
            focusRef: {
              pytorch: "googlenet.forward.x-f-max_poolnd-x-kernel_size-n-stride-n-padding-n.2",
              jax: "googlenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same.4",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["googlenet.self-avgpool-nn-adaptiveavgpoolnd-n-n","googlenet.self-dropout-nn-dropout-n","googlenet.forward.x-self-avgpool-x","googlenet.forward.x-torch-flatten-x-start_dim-n"],
          jax: ["googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.3","googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionnb-x.3"],
        },
        focusRef: {
          pytorch: "googlenet.self-avgpool-nn-adaptiveavgpoolnd-n-n",
          jax: "googlenet.__call__.x-inceptionblock-n-n-n-n-n-n-name-inceptionna-x.3",
        },
        includeChildRefs: false,
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
          }),
        ],
      },
      {
        id: "classifier",
        label: "classifier",
        type: "GlobalPoolHead",
        kind: "group",
        summary: "avgpool + fc",
        sourceRefs: {
          pytorch: [],
          jax: [],
        },
        focusRef: {
          pytorch: "googlenet.self-fc-nn-linear-n-num_classes",
          jax: "googlenet.__call__.x-jnp-mean-x-axis-n-n",
        },
        includeChildRefs: true,
        children: [
          {
            id: "classifier.avgpool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            sourceRefs: {
              pytorch: ["googlenet.self-avgpool-nn-adaptiveavgpoolnd-n-n","googlenet.forward.x-self-avgpool-x"],
              jax: ["googlenet.__call__.x-jnp-mean-x-axis-n-n"],
            },
            focusRef: {
              pytorch: "googlenet.self-avgpool-nn-adaptiveavgpoolnd-n-n",
              jax: "googlenet.__call__.x-jnp-mean-x-axis-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.flatten",
            label: "flatten",
            type: "Flatten",
            kind: "reshape",
            badges: ["1024"],
            sourceRefs: {
              pytorch: ["googlenet.forward.x-torch-flatten-x-start_dim-n"],
              jax: ["googlenet.__call__.x-jnp-mean-x-axis-n-n"],
            },
            focusRef: {
              pytorch: "googlenet.forward.x-torch-flatten-x-start_dim-n",
              jax: "googlenet.__call__.x-jnp-mean-x-axis-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.4"],
            sourceRefs: {
              pytorch: ["googlenet.self-dropout-nn-dropout-n","googlenet.classifier.dropout"],
              jax: ["googlenet.__call__.x-nn-dropout-n-deterministic-not-train-x"],
            },
            focusRef: {
              pytorch: "googlenet.self-dropout-nn-dropout-n",
              jax: "googlenet.__call__.x-nn-dropout-n-deterministic-not-train-x",
            },
            includeChildRefs: false,
          },
          {
            id: "classifier.fc",
            label: "fc",
            type: "Linear",
            kind: "linear",
            badges: ["1024->1000"],
            sourceRefs: {
              pytorch: ["googlenet.self-fc-nn-linear-n-num_classes","googlenet.classifier.fc"],
              jax: ["googlenet.__call__.logits-nn-dense-self-num_classes-name-fc-x"],
            },
            focusRef: {
              pytorch: "googlenet.self-fc-nn-linear-n-num_classes",
              jax: "googlenet.__call__.logits-nn-dense-self-num_classes-name-fc-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  unet: {
    stats: "4 down blocks · bottleneck · 4 up blocks",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["1 x 572 x 572"],
        sourceRefs: {
          pytorch: ["unet.forward.dn-self-downn-x"],
          jax: ["unet.__call__.dn-doubleconv-n-x"],
        },
        focusRef: {
          pytorch: "unet.forward.dn-self-downn-x",
          jax: "unet.__call__.dn-doubleconv-n-x",
        },
        includeChildRefs: false,
      },
      {
        id: "contracting",
        label: "Contracting Path",
        type: "Encoder",
        kind: "group",
        summary: "4 DoubleConv blocks",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["unet.num_classes-n","unet.self-downn-doubleconv-n-n.2","unet.self-pooln-nn-maxpoolnd-n.2","unet.self-downn-doubleconv-n-n.3","unet.self-pooln-nn-maxpoolnd-n.3","unet.self-downn-doubleconv-n-n.4","unet.self-pooln-nn-maxpoolnd-n.4","unet.forward.dn-self-downn-x","unet.forward.dn-self-downn-pn","unet.forward.pn-self-pooln-dn.2"],
          jax: ["class-unet-nn-module","unet.__call__.dn-doubleconv-n-x","unet.__call__.pn-nn-max_pool-dn-n-n-n-n","unet.__call__.dn-doubleconv-n-pn","unet.__call__.pn-nn-max_pool-dn-n-n-n-n.2","unet.__call__.dn-doubleconv-n-pn.2","unet.__call__.pn-nn-max_pool-dn-n-n-n-n.3","unet.__call__.dn-doubleconv-n-pn.3","unet.__call__.pn-nn-max_pool-dn-n-n-n-n.4"],
        },
        focusRef: {
          pytorch: "unet.num_classes-n",
          jax: "class-unet-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "contracting.down1",
            label: "down1",
            type: "DoubleConv",
            kind: "conv",
            badges: ["1->64", "572x572"],
            sourceRefs: {
              pytorch: ["doubleconv.__init__.nn-convnd-in_channels-out_channels-kernel_size-n-padding-n","doubleconv.__init__.nn-relu-inplace-true","doubleconv.__init__.nn-convnd-out_channels-out_channels-kernel_size-n-padding-n","doubleconv.__init__.nn-relu-inplace-true.2","doubleconv.__init__.code.3","doubleconv.forward.return-out","unet.num_classes-n"],
              jax: ["class-doubleconv-nn-module","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x","doubleconv.__call__.x-nn-relu-x","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2","doubleconv.__call__.x-nn-relu-x.2","unet.__call__.dn-doubleconv-n-x"],
            },
            focusRef: {
              pytorch: "doubleconv.__init__.nn-convnd-in_channels-out_channels-kernel_size-n-padding-n",
              jax: "class-doubleconv-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "contracting.pool1",
            label: "pool1",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            sourceRefs: {
              pytorch: ["unet.forward.dn-self-downn-x"],
              jax: ["unet.__call__.pn-nn-max_pool-dn-n-n-n-n"],
            },
            focusRef: {
              pytorch: "unet.forward.dn-self-downn-x",
              jax: "unet.__call__.pn-nn-max_pool-dn-n-n-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "contracting.down2",
            label: "down2",
            type: "DoubleConv",
            kind: "conv",
            badges: ["64->128"],
            sourceRefs: {
              pytorch: ["unet.self-downn-doubleconv-n-n.2","unet.forward.dn-self-downn-x"],
              jax: ["unet.__call__.dn-doubleconv-n-pn"],
            },
            focusRef: {
              pytorch: "unet.self-downn-doubleconv-n-n.2",
              jax: "unet.__call__.dn-doubleconv-n-pn",
            },
            includeChildRefs: false,
          },
          {
            id: "contracting.pool2",
            label: "pool2",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            sourceRefs: {
              pytorch: ["unet.self-pooln-nn-maxpoolnd-n.2","unet.forward.dn-self-downn-pn"],
              jax: ["unet.__call__.pn-nn-max_pool-dn-n-n-n-n.2"],
            },
            focusRef: {
              pytorch: "unet.self-pooln-nn-maxpoolnd-n.2",
              jax: "unet.__call__.pn-nn-max_pool-dn-n-n-n-n.2",
            },
            includeChildRefs: false,
          },
          {
            id: "contracting.down3",
            label: "down3",
            type: "DoubleConv",
            kind: "conv",
            badges: ["128->256"],
            sourceRefs: {
              pytorch: ["unet.self-downn-doubleconv-n-n.3","unet.forward.dn-self-downn-pn"],
              jax: ["unet.__call__.dn-doubleconv-n-pn.2"],
            },
            focusRef: {
              pytorch: "unet.self-downn-doubleconv-n-n.3",
              jax: "unet.__call__.dn-doubleconv-n-pn.2",
            },
            includeChildRefs: false,
          },
          {
            id: "contracting.pool3",
            label: "pool3",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            sourceRefs: {
              pytorch: ["unet.self-pooln-nn-maxpoolnd-n.3","unet.forward.pn-self-pooln-dn.2"],
              jax: ["unet.__call__.pn-nn-max_pool-dn-n-n-n-n.3"],
            },
            focusRef: {
              pytorch: "unet.self-pooln-nn-maxpoolnd-n.3",
              jax: "unet.__call__.pn-nn-max_pool-dn-n-n-n-n.3",
            },
            includeChildRefs: false,
          },
          {
            id: "contracting.down4",
            label: "down4",
            type: "DoubleConv",
            kind: "conv",
            badges: ["256->512"],
            sourceRefs: {
              pytorch: ["unet.self-downn-doubleconv-n-n.4","unet.forward.pn-self-pooln-dn.2"],
              jax: ["unet.__call__.dn-doubleconv-n-pn.3"],
            },
            focusRef: {
              pytorch: "unet.self-downn-doubleconv-n-n.4",
              jax: "unet.__call__.dn-doubleconv-n-pn.3",
            },
            includeChildRefs: false,
          },
          {
            id: "contracting.pool4",
            label: "pool4",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["2x2"],
            sourceRefs: {
              pytorch: ["unet.self-pooln-nn-maxpoolnd-n.4","unet.forward.dn-self-downn-pn.2"],
              jax: ["unet.__call__.pn-nn-max_pool-dn-n-n-n-n.4"],
            },
            focusRef: {
              pytorch: "unet.self-pooln-nn-maxpoolnd-n.4",
              jax: "unet.__call__.pn-nn-max_pool-dn-n-n-n-n.4",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "bottleneck",
        label: "bottleneck",
        type: "DoubleConv",
        kind: "conv",
        badges: ["512->1024"],
        sourceRefs: {
          pytorch: ["unet.self-bottleneck-doubleconv-n-n","unet.forward.dn-self-downn-pn.2"],
          jax: ["class-doubleconv-nn-module","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x","doubleconv.__call__.x-nn-relu-x","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2","doubleconv.__call__.x-nn-relu-x.2","unet.__call__.b-doubleconv-n-pn"],
        },
        focusRef: {
          pytorch: "unet.self-bottleneck-doubleconv-n-n",
          jax: "class-doubleconv-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: "expansive",
        label: "Expansive Path",
        type: "Decoder",
        kind: "group",
        summary: "upsample + concat skips",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n","unet.self-decn-doubleconv-n-n","unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.2","unet.self-decn-doubleconv-n-n.2","unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.3","unet.self-decn-doubleconv-n-n.3","unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.4","unet.self-decn-doubleconv-n-n.4","unet.forward.pn-self-pooln-dn.3","unet.forward.dn-self-downn-pn.3","unet.forward.pn-self-pooln-dn.4","unet.forward.x-self-decn-x","unet.forward.x-self-upn-x","unet.forward.x-torch-cat-x-dn-dim-n.2","unet.forward.x-self-decn-x.2","unet.forward.x-self-upn-x.2","unet.forward.x-torch-cat-x-dn-dim-n.3"],
          jax: ["class-unet-nn-module","unet.__call__.x-resize_like-b-dn","unet.__call__.x-jnp-concatenate-x-dn-axis-n","unet.__call__.x-doubleconv-n-x","unet.__call__.x-resize_like-x-dn","unet.__call__.x-jnp-concatenate-x-dn-axis-n.2","unet.__call__.x-doubleconv-n-x.2","unet.__call__.x-resize_like-x-dn.2","unet.__call__.x-jnp-concatenate-x-dn-axis-n.3","unet.__call__.x-doubleconv-n-x.3","unet.__call__.x-resize_like-x-dn.3","unet.__call__.x-jnp-concatenate-x-dn-axis-n.4","unet.__call__.x-doubleconv-n-x.4"],
        },
        focusRef: {
          pytorch: "unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n",
          jax: "class-unet-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "expansive.up4",
            label: "up4",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["1024->512", "x2"],
            sourceRefs: {
              pytorch: ["unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n","unet.forward.pn-self-pooln-dn.3"],
              jax: ["def-resize_like-x-skip","resize_like.resize_shape-x-shape-n-skip-shape-n-skip-shape-n-x-shape-n","resize_like.resized-jax-image-resize-x-resize_shape-method-nearest","resize_like.return-resized","unet.__call__.x-resize_like-b-dn"],
            },
            focusRef: {
              pytorch: "unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n",
              jax: "def-resize_like-x-skip",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.up4.skip",
            label: "skip d4",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            sourceRefs: {
              pytorch: ["unet.forward.dn-self-downn-pn.3"],
              jax: ["unet.__call__.x-jnp-concatenate-x-dn-axis-n"],
            },
            focusRef: {
              pytorch: "unet.forward.dn-self-downn-pn.3",
              jax: "unet.__call__.x-jnp-concatenate-x-dn-axis-n",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.dec4",
            label: "dec4",
            type: "DoubleConv",
            kind: "conv",
            badges: ["1024->512"],
            sourceRefs: {
              pytorch: ["unet.self-decn-doubleconv-n-n","unet.forward.pn-self-pooln-dn.4"],
              jax: ["class-doubleconv-nn-module","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x","doubleconv.__call__.x-nn-relu-x","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2","doubleconv.__call__.x-nn-relu-x.2","unet.__call__.x-doubleconv-n-x"],
            },
            focusRef: {
              pytorch: "unet.self-decn-doubleconv-n-n",
              jax: "class-doubleconv-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.up3",
            label: "up3",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["512->256", "x2"],
            sourceRefs: {
              pytorch: ["unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.2"],
              jax: ["def-resize_like-x-skip","resize_like.resize_shape-x-shape-n-skip-shape-n-skip-shape-n-x-shape-n","resize_like.resized-jax-image-resize-x-resize_shape-method-nearest","resize_like.return-resized","unet.__call__.x-resize_like-x-dn"],
            },
            focusRef: {
              pytorch: "unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.2",
              jax: "def-resize_like-x-skip",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.up3.skip",
            label: "skip d3",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            sourceRefs: {
              pytorch: ["unet.forward.x-torch-cat-x-dn-dim-n.2"],
              jax: ["unet.__call__.x-jnp-concatenate-x-dn-axis-n.2"],
            },
            focusRef: {
              pytorch: "unet.forward.x-torch-cat-x-dn-dim-n.2",
              jax: "unet.__call__.x-jnp-concatenate-x-dn-axis-n.2",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.dec3",
            label: "dec3",
            type: "DoubleConv",
            kind: "conv",
            badges: ["512->256"],
            sourceRefs: {
              pytorch: ["unet.self-decn-doubleconv-n-n.2"],
              jax: ["class-doubleconv-nn-module","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x","doubleconv.__call__.x-nn-relu-x","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2","doubleconv.__call__.x-nn-relu-x.2","unet.__call__.x-doubleconv-n-x.2"],
            },
            focusRef: {
              pytorch: "unet.self-decn-doubleconv-n-n.2",
              jax: "class-doubleconv-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.up2",
            label: "up2",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["256->128", "x2"],
            sourceRefs: {
              pytorch: ["unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.3","unet.forward.x-self-decn-x"],
              jax: ["def-resize_like-x-skip","resize_like.resize_shape-x-shape-n-skip-shape-n-skip-shape-n-x-shape-n","resize_like.resized-jax-image-resize-x-resize_shape-method-nearest","resize_like.return-resized","unet.__call__.x-resize_like-x-dn.2"],
            },
            focusRef: {
              pytorch: "unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.3",
              jax: "def-resize_like-x-skip",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.up2.skip",
            label: "skip d2",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            sourceRefs: {
              pytorch: ["unet.forward.x-self-upn-x"],
              jax: ["unet.__call__.x-jnp-concatenate-x-dn-axis-n.3"],
            },
            focusRef: {
              pytorch: "unet.forward.x-self-upn-x",
              jax: "unet.__call__.x-jnp-concatenate-x-dn-axis-n.3",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.dec2",
            label: "dec2",
            type: "DoubleConv",
            kind: "conv",
            badges: ["256->128"],
            sourceRefs: {
              pytorch: ["unet.self-decn-doubleconv-n-n.3","unet.forward.x-torch-cat-x-dn-dim-n.2"],
              jax: ["class-doubleconv-nn-module","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x","doubleconv.__call__.x-nn-relu-x","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2","doubleconv.__call__.x-nn-relu-x.2","unet.__call__.x-doubleconv-n-x.3"],
            },
            focusRef: {
              pytorch: "unet.self-decn-doubleconv-n-n.3",
              jax: "class-doubleconv-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.up1",
            label: "up1",
            type: "ConvTranspose2d",
            kind: "reshape",
            badges: ["128->64", "x2"],
            sourceRefs: {
              pytorch: ["unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.4","unet.forward.x-self-decn-x.2"],
              jax: ["def-resize_like-x-skip","resize_like.resize_shape-x-shape-n-skip-shape-n-skip-shape-n-x-shape-n","resize_like.resized-jax-image-resize-x-resize_shape-method-nearest","resize_like.return-resized","unet.__call__.x-resize_like-x-dn.3"],
            },
            focusRef: {
              pytorch: "unet.self-upn-nn-convtransposend-n-n-kernel_size-n-stride-n.4",
              jax: "def-resize_like-x-skip",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.up1.skip",
            label: "skip d1",
            type: "Concat",
            kind: "residual",
            badges: ["encoder features"],
            sourceRefs: {
              pytorch: ["unet.forward.x-self-upn-x.2"],
              jax: ["unet.__call__.x-jnp-concatenate-x-dn-axis-n.4"],
            },
            focusRef: {
              pytorch: "unet.forward.x-self-upn-x.2",
              jax: "unet.__call__.x-jnp-concatenate-x-dn-axis-n.4",
            },
            includeChildRefs: false,
          },
          {
            id: "expansive.dec1",
            label: "dec1",
            type: "DoubleConv",
            kind: "conv",
            badges: ["128->64"],
            sourceRefs: {
              pytorch: ["unet.self-decn-doubleconv-n-n.4","unet.forward.x-torch-cat-x-dn-dim-n.3"],
              jax: ["class-doubleconv-nn-module","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x","doubleconv.__call__.x-nn-relu-x","doubleconv.__call__.x-nn-conv-self-out_channels-n-n-padding-same-x.2","doubleconv.__call__.x-nn-relu-x.2","unet.__call__.x-doubleconv-n-x.4"],
            },
            focusRef: {
              pytorch: "unet.self-decn-doubleconv-n-n.4",
              jax: "class-doubleconv-nn-module",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "out_conv",
        label: "out_conv",
        type: "Conv2d",
        kind: "conv",
        badges: ["64->2", "1x1"],
        sourceRefs: {
          pytorch: ["unet.self-out_conv-nn-convnd-n-num_classes-kernel_size-n","unet.forward.x-self-decn-x.3"],
          jax: ["unet.__call__.logits-nn-conv-self-num_classes-n-n-name-out_conv-x"],
        },
        focusRef: {
          pytorch: "unet.self-out_conv-nn-convnd-n-num_classes-kernel_size-n",
          jax: "unet.__call__.logits-nn-conv-self-num_classes-n-n-name-out_conv-x",
        },
        includeChildRefs: false,
      },
    ],
  },
  transformer: {
    stats: "6 encoder layers · 6 decoder layers · 8 heads",
    nodes: [
      {
        id: "src.input",
        label: "src input",
        type: "TokenIds",
        kind: "input",
        badges: ["source", "16 tokens"],
        sourceRefs: {
          pytorch: ["transformer.forward.src_embeddings-self-src_embed-src_ids"],
          jax: ["transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids"],
        },
        focusRef: {
          pytorch: "transformer.forward.src_embeddings-self-src_embed-src_ids",
          jax: "transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids",
        },
        includeChildRefs: false,
      },
      {
        id: "tgt.input",
        label: "target input",
        type: "TokenIds",
        kind: "input",
        badges: ["target", "shifted"],
        sourceRefs: {
          pytorch: ["transformer.forward.tgt_embeddings-self-tgt_embed-tgt_ids"],
          jax: ["transformer.__call__.tgt_embedding-nn-embed-self-vocab_size-self-d_model-tgt_ids"],
        },
        focusRef: {
          pytorch: "transformer.forward.tgt_embeddings-self-tgt_embed-tgt_ids",
          jax: "transformer.__call__.tgt_embedding-nn-embed-self-vocab_size-self-d_model-tgt_ids",
        },
        includeChildRefs: false,
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["transformer.self-generator-nn-linear-d_model-vocab_size","transformer.def-forward-self-src_ids-tgt_ids-tgt_mask","transformer.forward.tgt_embeddings-self-tgt_embed-tgt_ids","transformer.forward.logits-self-generator-x"],
          jax: ["transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids","transformer.__call__.memory-positionalencoding-self-d_model-src_embedding","transformer.__call__.tgt_embedding-nn-embed-self-vocab_size-self-d_model-tgt_ids","transformer.__call__.x-positionalencoding-self-d_model-tgt_embedding"],
        },
        focusRef: {
          pytorch: "transformer.self-generator-nn-linear-d_model-vocab_size",
          jax: "transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids",
        },
        includeChildRefs: false,
        children: [
          {
            id: "src_embed",
            label: "src_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            sourceRefs: {
              pytorch: ["transformer.self-generator-nn-linear-d_model-vocab_size"],
              jax: ["transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids"],
            },
            focusRef: {
              pytorch: "transformer.self-generator-nn-linear-d_model-vocab_size",
              jax: "transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids",
            },
            includeChildRefs: false,
          },
          {
            id: "tgt_embed",
            label: "tgt_embed",
            type: "Embedding",
            kind: "embedding",
            badges: ["vocab", "512"],
            sourceRefs: {
              pytorch: ["transformer.self-tgt_embed-nn-embedding-vocab_size-d_model"],
              jax: ["transformer.__call__.tgt_embedding-nn-embed-self-vocab_size-self-d_model-tgt_ids"],
            },
            focusRef: {
              pytorch: "transformer.self-tgt_embed-nn-embedding-vocab_size-d_model",
              jax: "transformer.__call__.tgt_embedding-nn-embed-self-vocab_size-self-d_model-tgt_ids",
            },
            includeChildRefs: false,
          },
          {
            id: "positional_encoding",
            label: "positional",
            type: "SinusoidalEncoding",
            kind: "embedding",
            badges: ["absolute"],
            sourceRefs: {
              pytorch: ["positionalencoding.def-__init__","positionalencoding.self","positionalencoding.positions-torch-arange-max_len","positionalencoding.position-positions-unsqueeze-n","positionalencoding.even_indices-torch-arange-n-d_model-n","positionalencoding.log_base-torch-log-torch-tensor-n","positionalencoding.scale-log_base-d_model","positionalencoding.div_term-torch-exp-even_indices-scale","positionalencoding.pe-torch-zeros-max_len-d_model","positionalencoding.sin_values-torch-sin-position-div_term","positionalencoding.cos_values-torch-cos-position-div_term","positionalencoding.pe-n-n-sin_values","positionalencoding.pe-n-n-cos_values","positionalencoding.self-register_buffer-pe-pe","positionalencoding.forward.position_encoding-self-pe-seq_len","positionalencoding.forward.encoded-x-position_encoding","positionalencoding.forward.return-encoded","transformer.def-forward-self-src_ids-tgt_ids-tgt_mask","transformer.forward.tgt_embeddings-self-tgt_embed-tgt_ids","transformer.forward.logits-self-generator-x"],
              jax: ["positionalencoding.__call__.seq_len-x-shape-n","positionalencoding.__call__.positions-jnp-arange-seq_len","positionalencoding.__call__.position-positions-none","positionalencoding.__call__.even_indices-jnp-arange-n-self-d_model-n","positionalencoding.__call__.scale-jnp-log-n-self-d_model","positionalencoding.__call__.div_term-jnp-exp-even_indices-scale","positionalencoding.__call__.pe-jnp-zeros-seq_len-self-d_model","positionalencoding.__call__.sin_values-jnp-sin-position-div_term","positionalencoding.__call__.cos_values-jnp-cos-position-div_term","positionalencoding.__call__.pe-pe-at-n-n-set-sin_values","positionalencoding.__call__.pe-pe-at-n-n-set-cos_values","positionalencoding.__call__.batch_pe-pe-none","positionalencoding.__call__.encoded-x-batch_pe","positionalencoding.__call__.return-encoded","transformer.__call__.memory-positionalencoding-self-d_model-src_embedding","transformer.__call__.x-positionalencoding-self-d_model-tgt_embedding"],
            },
            focusRef: {
              pytorch: "positionalencoding.def-__init__",
              jax: "positionalencoding.__call__.seq_len-x-shape-n",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["transformer.forward.x-self-pos-tgt_embeddings","transformer.forward.for-layer-in-self-decoder"],
          jax: ["transformer.__call__.for-_-in-range-self-num_layers","transformer.__call__.memory-encoderlayer-self-d_model-self-nhead-memory"],
        },
        focusRef: {
          pytorch: "transformer.forward.x-self-pos-tgt_embeddings",
          jax: "transformer.__call__.for-_-in-range-self-num_layers",
        },
        includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["transformer.forward.src_embeddings-self-src_embed-src_ids","transformer.forward.return-logits"],
          jax: ["transformer.__call__.for-_-in-range-self-num_layers.2","transformer.__call__.x-decoderlayer-self-d_model-self-nhead-x-memory-tgt_mask"],
        },
        focusRef: {
          pytorch: "transformer.forward.src_embeddings-self-src_embed-src_ids",
          jax: "transformer.__call__.for-_-in-range-self-num_layers.2",
        },
        includeChildRefs: false,
        children: Array.from({ length: 6 }, (_, index) => makeTransformerDecoderBlock(index, index === 0)),
      },
      {
        id: "generator",
        label: "generator",
        type: "Linear",
        kind: "linear",
        badges: ["512->vocab"],
        sourceRefs: {
          pytorch: ["transformer.forward.memory-self-pos-src_embeddings"],
          jax: ["transformer.__call__.logits-nn-dense-self-vocab_size-x"],
        },
        focusRef: {
          pytorch: "transformer.forward.memory-self-pos-src_embeddings",
          jax: "transformer.__call__.logits-nn-dense-self-vocab_size-x",
        },
        includeChildRefs: false,
      },
    ],
  },
  vqvae: {
    stats: "discrete codebook · nearest-neighbor lookup · straight-through estimator",
    nodes: [
      {
        id: "input",
        label: "input x",
        type: "FlatVector",
        kind: "input",
        badges: ["784"],
        sourceRefs: {
          pytorch: ["encoder.def-forward-self-x","encoder.forward.z_e-self-net-x","vectorquantizedautoencoder.def-forward-self-x","vectorquantizedautoencoder.forward.z_e-self-encoder-x","vectorquantizedautoencoder.def-loss-self-x","vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment","inputs-torch-zeros-n-n","loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_"],
          jax: ["encoder.def-__call__-self-x","encoder.__call__.z_e-nn-dense-self-hidden_dim-name-fcn-x","vectorquantizedautoencoder.def-__call__-self-x","vectorquantizedautoencoder.__call__.z_e-self-encoder-x","vectorquantizedautoencoder.def-loss-self-x","vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment","inputs-jnp-zeros-n-n","train_step.loss_fn.loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_"],
        },
        focusRef: {
          pytorch: "encoder.def-forward-self-x",
          jax: "encoder.def-__call__-self-x",
        },
        includeChildRefs: false,
      },
      {
        id: "encoder",
        label: "Encoder",
        type: "z_e(x)",
        kind: "group",
        summary: "continuous latent before VQ",
        badges: ["784->256->32"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-encoder-nn-module","encoder.def-forward-self-x","encoder.forward.return-z_e","vectorquantizedautoencoder.self-encoder-encoder-input_dim-hidden_dim-latent_dim","vectorquantizedautoencoder.forward.z_e-self-encoder-x"],
          jax: ["class-encoder-nn-module","encoder.nn-compact","encoder.__call__.return-z_e","vectorquantizedautoencoder.setup.self-encoder-encoder-self-input_dim-self-hidden_dim-self-latent_dim","vectorquantizedautoencoder.__call__.z_e-self-encoder-x"],
        },
        focusRef: {
          pytorch: "class-encoder-nn-module",
          jax: "class-encoder-nn-module",
        },
        includeChildRefs: true,
        children: [
          {
            id: "encoder.trunk",
            label: "projection",
            type: "MLP",
            kind: "linear",
            badges: ["784->32"],
            sourceRefs: {
              pytorch: ["encoder.self-net-nn-sequential","encoder.nn-linear-input_dim-hidden_dim","encoder.nn-relu","encoder.nn-linear-hidden_dim-latent_dim","encoder.forward.z_e-self-net-x"],
              jax: ["encoder.__call__.z_e-nn-dense-self-hidden_dim-name-fcn-x","encoder.__call__.z_e-nn-relu-z_e","encoder.__call__.z_e-nn-dense-self-latent_dim-name-fcn-z_e"],
            },
            focusRef: {
              pytorch: "encoder.self-net-nn-sequential",
              jax: "encoder.__call__.z_e-nn-dense-self-hidden_dim-name-fcn-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["class-vectorquantizer-nn-module","vectorquantizedautoencoder.self-quantizer-vectorquantizer-num_codes-latent_dim-beta"],
          jax: ["class-vectorquantizer-nn-module","vectorquantizer.nn-compact","vectorquantizedautoencoder.setup.self-quantizer-vectorquantizer-self-num_codes-self-latent_dim-self-beta"],
        },
        focusRef: {
          pytorch: "class-vectorquantizer-nn-module",
          jax: "class-vectorquantizer-nn-module",
        },
        includeChildRefs: true,
        children: [
          {
            id: "quantizer.codebook",
            label: "codebook",
            type: "Embedding table",
            kind: "embedding",
            badges: ["K x D"],
            sourceRefs: {
              pytorch: ["vectorquantizer.self-codebook-nn-embedding-num_codes-latent_dim","vectorquantizer.self-codebook-weight-data-uniform_-n-num_codes-n-num_codes","vectorquantizer.forward.codebook_squared-torch-sum-self-codebook-weight-n-dim-n","vectorquantizer.forward.dot_products-z_e-self-codebook-weight-t","vectorquantizer.forward.encoding_indices-torch-argmin-distances-dim-n","vectorquantizer.forward.quantized-self-codebook-encoding_indices","vectorquantizer.forward.codebook_loss-f-mse_loss-quantized-z_e-detach"],
              jax: ["vectorquantizer.__call__.codebook-self-param-codebook-nn-initializers-uniform-scale-n-self-num_co","vectorquantizer.__call__.codebook_squared-jnp-sum-codebook-n-axis-n","vectorquantizer.__call__.dot_products-z_e-codebook-t","vectorquantizer.__call__.encoding_indices-jnp-argmin-distances-axis-n","vectorquantizer.__call__.quantized-codebook-encoding_indices","vectorquantizer.__call__.codebook_error-quantized-jax-lax-stop_gradient-z_e","vectorquantizer.__call__.codebook_loss-jnp-mean-codebook_error-n"],
            },
            focusRef: {
              pytorch: "vectorquantizer.self-codebook-nn-embedding-num_codes-latent_dim",
              jax: "vectorquantizer.__call__.codebook-self-param-codebook-nn-initializers-uniform-scale-n-self-num_co",
            },
            includeChildRefs: false,
          },
          {
            id: "quantizer.lookup",
            label: "nearest lookup",
            type: "argmin distance",
            kind: "attention",
            badges: ["argmin"],
            sourceRefs: {
              pytorch: ["vectorquantizer.def-forward-self-z_e","vectorquantizer.forward.z_squared-torch-sum-z_e-n-dim-n-keepdim-true","vectorquantizer.forward.codebook_squared-torch-sum-self-codebook-weight-n-dim-n","vectorquantizer.forward.dot_products-z_e-self-codebook-weight-t","vectorquantizer.forward.distances-z_squared-codebook_squared-n-dot_products","vectorquantizer.forward.encoding_indices-torch-argmin-distances-dim-n","vectorquantizer.forward.quantized-self-codebook-encoding_indices","vectorquantizedautoencoder.forward.z_q-vq_loss-encoding_indices-codebook_loss-commitment_loss-self-quantize"],
              jax: ["vectorquantizer.__call__.z_squared-jnp-sum-z_e-n-axis-n-keepdims-true","vectorquantizer.__call__.codebook_squared-jnp-sum-codebook-n-axis-n","vectorquantizer.__call__.dot_products-z_e-codebook-t","vectorquantizer.__call__.distances-z_squared-codebook_squared-n-dot_products","vectorquantizer.__call__.encoding_indices-jnp-argmin-distances-axis-n","vectorquantizer.__call__.quantized-codebook-encoding_indices","vectorquantizedautoencoder.__call__.z_q-vq_loss-encoding_indices-codebook_loss-commitment_loss-self-quantize"],
            },
            focusRef: {
              pytorch: "vectorquantizer.def-forward-self-z_e",
              jax: "vectorquantizer.__call__.z_squared-jnp-sum-z_e-n-axis-n-keepdims-true",
            },
            includeChildRefs: false,
          },
          {
            id: "quantizer.straight_through",
            label: "straight-through",
            type: "Estimator",
            kind: "residual",
            badges: ["detach"],
            sourceRefs: {
              pytorch: ["vectorquantizer.forward.codebook_loss-f-mse_loss-quantized-z_e-detach","vectorquantizer.forward.commitment_loss-f-mse_loss-z_e-quantized-detach","vectorquantizer.forward.vq_loss-codebook_loss-self-beta-commitment_loss","vectorquantizer.forward.quantized_st-z_e-quantized-z_e-detach","vectorquantizer.forward.return-quantized_st-vq_loss-encoding_indices-codebook_loss-commitment_lo"],
              jax: ["vectorquantizer.__call__.codebook_error-quantized-jax-lax-stop_gradient-z_e","vectorquantizer.__call__.codebook_loss-jnp-mean-codebook_error-n","vectorquantizer.__call__.commitment_error-z_e-jax-lax-stop_gradient-quantized","vectorquantizer.__call__.commitment_loss-jnp-mean-commitment_error-n","vectorquantizer.__call__.vq_loss-codebook_loss-self-beta-commitment_loss","vectorquantizer.__call__.quantized_st-z_e-jax-lax-stop_gradient-quantized-z_e","vectorquantizer.__call__.return-quantized_st-vq_loss-encoding_indices-codebook_loss-commitment_lo"],
            },
            focusRef: {
              pytorch: "vectorquantizer.forward.codebook_loss-f-mse_loss-quantized-z_e-detach",
              jax: "vectorquantizer.__call__.codebook_error-quantized-jax-lax-stop_gradient-z_e",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["class-decoder-nn-module","decoder.def-forward-self-z_q","vectorquantizedautoencoder.self-decoder-decoder-latent_dim-hidden_dim-input_dim"],
          jax: ["class-decoder-nn-module","decoder.nn-compact","vectorquantizedautoencoder.setup.self-decoder-decoder-self-latent_dim-self-hidden_dim-self-input_dim"],
        },
        focusRef: {
          pytorch: "class-decoder-nn-module",
          jax: "class-decoder-nn-module",
        },
        includeChildRefs: true,
        children: [
          {
            id: "decoder.hidden",
            label: "hidden",
            type: "Linear + ReLU",
            kind: "linear",
            badges: ["32->256"],
            sourceRefs: {
              pytorch: ["decoder.self-net-nn-sequential","decoder.nn-linear-latent_dim-hidden_dim","decoder.nn-relu"],
              jax: ["decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z_q","decoder.__call__.x-nn-relu-x"],
            },
            focusRef: {
              pytorch: "decoder.self-net-nn-sequential",
              jax: "decoder.__call__.x-nn-dense-self-hidden_dim-name-fcn-z_q",
            },
            includeChildRefs: false,
          },
          {
            id: "decoder.reconstruction",
            label: "reconstruction",
            type: "Bernoulli probs",
            kind: "activation",
            badges: ["sigmoid"],
            sourceRefs: {
              pytorch: ["decoder.nn-linear-hidden_dim-output_dim","decoder.nn-sigmoid","decoder.forward.reconstruction-self-net-z_q","decoder.forward.return-reconstruction","vectorquantizedautoencoder.forward.reconstruction-self-decoder-z_q"],
              jax: ["decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x","decoder.__call__.reconstruction-nn-sigmoid-x","decoder.__call__.return-reconstruction","vectorquantizedautoencoder.__call__.reconstruction-self-decoder-z_q"],
            },
            focusRef: {
              pytorch: "decoder.nn-linear-hidden_dim-output_dim",
              jax: "decoder.__call__.x-nn-dense-self-output_dim-name-fcn-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["vectorquantizedautoencoder.def-loss-self-x","vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment","vectorquantizedautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su","vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss","vectorquantizedautoencoder.loss.return-total_loss-reconstruction_loss-vq_loss-codebook_loss-commitment_l","loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_","final_loss-loss-item","final_reconstruction_loss-reconstruction_loss-item","final_vq_loss-vq_loss-item","final_codebook_loss-codebook_loss-item","final_commitment_loss-commitment_loss-item"],
          jax: ["vectorquantizedautoencoder.def-loss-self-x","vectorquantizedautoencoder.loss.reconstruction-z_e-z_q-encoding_indices-vq_loss-codebook_loss-commitment","vectorquantizedautoencoder.loss.reconstruction_loss-jnp-sum-x-jnp-log-reconstruction-ne-n-n-x-jnp-log-n-","vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss","vectorquantizedautoencoder.loss.return-total_loss-reconstruction_loss-vq_loss-codebook_loss-commitment_l","train_step.loss_fn.loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_","train_step.loss-aux-grads-jax-value_and_grad-loss_fn-has_aux-true-params","train_step.params-jax-tree_util-tree_map-lambda-p-g-p-learning_rate-g-params-grads","train_step.reconstruction_loss-vq_loss-codebook_loss-commitment_loss-encoding_indic","params-loss-reconstruction_loss-vq_loss-codebook_loss-commitment_loss-en","final_loss-loss","final_reconstruction_loss-reconstruction_loss","final_vq_loss-vq_loss","final_codebook_loss-codebook_loss","final_commitment_loss-commitment_loss"],
        },
        focusRef: {
          pytorch: "vectorquantizedautoencoder.def-loss-self-x",
          jax: "vectorquantizedautoencoder.def-loss-self-x",
        },
        includeChildRefs: false,
        children: [
          {
            id: "loss.reconstruction",
            label: "reconstruction",
            type: "BCE",
            kind: "head",
            sourceRefs: {
              pytorch: ["vectorquantizedautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su"],
              jax: ["vectorquantizedautoencoder.loss.reconstruction_loss-jnp-sum-x-jnp-log-reconstruction-ne-n-n-x-jnp-log-n-"],
            },
            focusRef: {
              pytorch: "vectorquantizedautoencoder.loss.reconstruction_loss-f-binary_cross_entropy-reconstruction-x-reduction-su",
              jax: "vectorquantizedautoencoder.loss.reconstruction_loss-jnp-sum-x-jnp-log-reconstruction-ne-n-n-x-jnp-log-n-",
            },
            includeChildRefs: false,
          },
          {
            id: "loss.vq",
            label: "codebook + commitment",
            type: "VQ terms",
            kind: "head",
            sourceRefs: {
              pytorch: ["vectorquantizer.forward.codebook_loss-f-mse_loss-quantized-z_e-detach","vectorquantizer.forward.commitment_loss-f-mse_loss-z_e-quantized-detach","vectorquantizer.forward.vq_loss-codebook_loss-self-beta-commitment_loss","vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss","final_vq_loss-vq_loss-item","final_codebook_loss-codebook_loss-item","final_commitment_loss-commitment_loss-item"],
              jax: ["vectorquantizer.__call__.codebook_error-quantized-jax-lax-stop_gradient-z_e","vectorquantizer.__call__.codebook_loss-jnp-mean-codebook_error-n","vectorquantizer.__call__.commitment_error-z_e-jax-lax-stop_gradient-quantized","vectorquantizer.__call__.commitment_loss-jnp-mean-commitment_error-n","vectorquantizer.__call__.vq_loss-codebook_loss-self-beta-commitment_loss","vectorquantizedautoencoder.loss.total_loss-reconstruction_loss-vq_loss","final_vq_loss-vq_loss","final_codebook_loss-codebook_loss","final_commitment_loss-commitment_loss"],
            },
            focusRef: {
              pytorch: "vectorquantizer.forward.codebook_loss-f-mse_loss-quantized-z_e-detach",
              jax: "vectorquantizer.__call__.codebook_error-quantized-jax-lax-stop_gradient-z_e",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  bert: {
    stats: "12 encoder layers · 12 heads/layer · 110M params",
    nodes: [
      {
        id: "input_ids",
        label: "input_ids",
        type: "TokenIds",
        kind: "input",
        badges: ["WordPiece", "16 tokens"],
        sourceRefs: {
          pytorch: ["bertembeddings.forward.x-self-word_embeddings-input_ids","bertbase.forward.x-self-embeddings-input_ids-token_type_ids","input_ids-torch-tensor-n-n-n-n-n-n-n-n","outputs-model-input_ids-token_type_ids-attention_mask"],
          jax: ["bertembeddings.__call__.x-nn-embed-self-vocab_size-self-hidden_size-name-word_embeddings-input_i","bertbase.__call__.x-bertembeddings-self-vocab_size-self-hidden_size-input_ids-token_type_i","input_ids-jnp-array-n-n-n-n-n-n-n-n-dtype-jnp-intn","params-model-init-jax-random-prngkey-n-input_ids-token_type_ids-attentio","train_step.loss_fn.outputs-model-apply-current_params-input_ids-token_type_ids-attention_ma"],
        },
        focusRef: {
          pytorch: "bertembeddings.forward.x-self-word_embeddings-input_ids",
          jax: "bertembeddings.__call__.x-nn-embed-self-vocab_size-self-hidden_size-name-word_embeddings-input_i",
        },
        includeChildRefs: false,
      },
      {
        id: "token_type_ids",
        label: "token_type_ids",
        type: "SegmentIds",
        kind: "input",
        badges: ["sentence A/B"],
        sourceRefs: {
          pytorch: ["bertembeddings.forward.x-x-self-token_type_embeddings-token_type_ids","bertbase.forward.x-self-embeddings-input_ids-token_type_ids","token_type_ids-torch-zeros-n-n-dtype-torch-long","outputs-model-input_ids-token_type_ids-attention_mask"],
          jax: ["bertembeddings.__call__.token_type_embeddings-nn-embed-self-type_vocab_size-self-hidden_size-nam","bertbase.__call__.x-bertembeddings-self-vocab_size-self-hidden_size-input_ids-token_type_i","token_type_ids-jnp-zeros-n-n-dtype-jnp-intn","params-model-init-jax-random-prngkey-n-input_ids-token_type_ids-attentio","train_step.loss_fn.outputs-model-apply-current_params-input_ids-token_type_ids-attention_ma"],
        },
        focusRef: {
          pytorch: "bertembeddings.forward.x-x-self-token_type_embeddings-token_type_ids",
          jax: "bertembeddings.__call__.token_type_embeddings-nn-embed-self-type_vocab_size-self-hidden_size-nam",
        },
        includeChildRefs: false,
      },
      {
        id: "embeddings",
        label: "Embeddings",
        type: "Group",
        kind: "group",
        summary: "token + position + segment",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-bertembeddings-nn-module","bertembeddings.def-forward-self-input_ids-token_type_ids","bertembeddings.forward.return-x","bertbase.self-embeddings-bertembeddings-vocab_size-hidden_size","bertbase.forward.x-self-embeddings-input_ids-token_type_ids"],
          jax: ["class-bertembeddings-nn-module","bertembeddings.nn-compact","bertembeddings.def-__call__-self-input_ids-token_type_ids-train-false","bertembeddings.__call__.return-x","bertbase.__call__.x-bertembeddings-self-vocab_size-self-hidden_size-input_ids-token_type_i"],
        },
        focusRef: {
          pytorch: "class-bertembeddings-nn-module",
          jax: "class-bertembeddings-nn-module",
        },
        includeChildRefs: true,
        children: [
          {
            id: "embeddings.word",
            label: "word",
            type: "WordPieceEmbedding",
            kind: "embedding",
            badges: ["30522", "768"],
            sourceRefs: {
              pytorch: ["bertembeddings.self-word_embeddings-nn-embedding-vocab_size-hidden_size","bertembeddings.forward.x-self-word_embeddings-input_ids"],
              jax: ["bertembeddings.__call__.x-nn-embed-self-vocab_size-self-hidden_size-name-word_embeddings-input_i"],
            },
            focusRef: {
              pytorch: "bertembeddings.self-word_embeddings-nn-embedding-vocab_size-hidden_size",
              jax: "bertembeddings.__call__.x-nn-embed-self-vocab_size-self-hidden_size-name-word_embeddings-input_i",
            },
            includeChildRefs: false,
          },
          {
            id: "embeddings.position",
            label: "position",
            type: "PositionEmbedding",
            kind: "embedding",
            badges: ["512", "768"],
            sourceRefs: {
              pytorch: ["bertembeddings.self-position_embeddings-nn-embedding-max_position-hidden_size","bertembeddings.forward.positions-torch-arange-input_ids-size-n-device-input_ids-device","bertembeddings.forward.position_embeddings-self-position_embeddings-positions","bertembeddings.forward.position_embeddings-position_embeddings-none","bertembeddings.forward.x-x-position_embeddings"],
              jax: ["bertembeddings.__call__.positions-jnp-arange-input_ids-shape-n","bertembeddings.__call__.position_embeddings-nn-embed-self-max_position-self-hidden_size-name-pos","bertembeddings.__call__.position_embeddings-position_embeddings-none","bertembeddings.__call__.x-x-position_embeddings"],
            },
            focusRef: {
              pytorch: "bertembeddings.self-position_embeddings-nn-embedding-max_position-hidden_size",
              jax: "bertembeddings.__call__.positions-jnp-arange-input_ids-shape-n",
            },
            includeChildRefs: false,
          },
          {
            id: "embeddings.segment",
            label: "segment",
            type: "TokenTypeEmbedding",
            kind: "embedding",
            badges: ["2", "768"],
            sourceRefs: {
              pytorch: ["bertembeddings.self-token_type_embeddings-nn-embedding-n-hidden_size","bertembeddings.forward.x-x-self-token_type_embeddings-token_type_ids"],
              jax: ["bertembeddings.__call__.token_type_embeddings-nn-embed-self-type_vocab_size-self-hidden_size-nam","bertembeddings.__call__.x-x-token_type_embeddings"],
            },
            focusRef: {
              pytorch: "bertembeddings.self-token_type_embeddings-nn-embedding-n-hidden_size",
              jax: "bertembeddings.__call__.token_type_embeddings-nn-embed-self-type_vocab_size-self-hidden_size-nam",
            },
            includeChildRefs: false,
          },
          {
            id: "embeddings.norm",
            label: "norm",
            type: "LayerNorm",
            kind: "norm",
            sourceRefs: {
              pytorch: ["bertembeddings.self-norm-nn-layernorm-hidden_size","bertembeddings.forward.x-self-norm-x"],
              jax: ["bertembeddings.__call__.x-nn-layernorm-name-layernorm-x"],
            },
            focusRef: {
              pytorch: "bertembeddings.self-norm-nn-layernorm-hidden_size",
              jax: "bertembeddings.__call__.x-nn-layernorm-name-layernorm-x",
            },
            includeChildRefs: false,
          },
          {
            id: "embeddings.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "dropout",
            badges: ["p=0.1"],
            sourceRefs: {
              pytorch: ["bertembeddings.self-dropout-nn-dropout-n","bertembeddings.forward.x-self-dropout-x"],
              jax: ["bertembeddings.__call__.x-nn-dropout-n-deterministic-not-train-x"],
            },
            focusRef: {
              pytorch: "bertembeddings.self-dropout-nn-dropout-n",
              jax: "bertembeddings.__call__.x-nn-dropout-n-deterministic-not-train-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["bertbase.self-layers-nn-modulelist-bertlayer-hidden_size-for-_-in-range-num_layer","bertbase.forward.for-layer-in-self-layers","bertbase.forward.x-layer-x-attention_mask"],
          jax: ["bertbase.__call__.for-_-in-range-self-num_layers","bertbase.__call__.x-bertlayer-self-hidden_size-x-attention_mask-train-train"],
        },
        focusRef: {
          pytorch: "bertbase.self-layers-nn-modulelist-bertlayer-hidden_size-for-_-in-range-num_layer",
          jax: "bertbase.__call__.for-_-in-range-self-num_layers",
        },
        includeChildRefs: false,
        children: Array.from({ length: 12 }, (_, index) => makeBertLayer(index, index === 3)),
      },
      {
        id: "pooler",
        label: "pooler",
        type: "CLSProjection",
        kind: "linear",
        badges: ["CLS", "768->768"],
        sourceRefs: {
          pytorch: ["bertbase.self-pooler-nn-linear-hidden_size-hidden_size","bertbase.forward.cls_token-x-n","bertbase.forward.pooled_projection-self-pooler-cls_token","bertbase.forward.pooled-torch-tanh-pooled_projection"],
          jax: ["bertbase.__call__.cls_token-x-n","bertbase.__call__.pooled_projection-nn-dense-self-hidden_size-name-pooler-cls_token","bertbase.__call__.pooled-jnp-tanh-pooled_projection"],
        },
        focusRef: {
          pytorch: "bertbase.self-pooler-nn-linear-hidden_size-hidden_size",
          jax: "bertbase.__call__.cls_token-x-n",
        },
        includeChildRefs: false,
      },
      {
        id: "mlm_head",
        label: "mlm_head",
        type: "MaskedLMHead",
        kind: "head",
        badges: ["768->30522"],
        sourceRefs: {
          pytorch: ["bertbase.self-mlm-nn-linear-hidden_size-vocab_size","bertbase.forward.mlm_logits-self-mlm-x"],
          jax: ["bertbase.__call__.mlm_logits-nn-dense-self-vocab_size-name-mlm_head-x"],
        },
        focusRef: {
          pytorch: "bertbase.self-mlm-nn-linear-hidden_size-vocab_size",
          jax: "bertbase.__call__.mlm_logits-nn-dense-self-vocab_size-name-mlm_head-x",
        },
        includeChildRefs: false,
      },
    ],
  },
  gpt2: {
    stats: "12 blocks · 12 heads/block · virtualized",
    nodes: [
      {
        id: "wte",
        label: "wte",
        type: "TokenEmbedding",
        kind: "embedding",
        badges: ["vocab", "768"],
        sourceRefs: {
          pytorch: ["gptnsmall.self-wte-nn-embedding-vocab_size-n_embd","gptnsmall.forward.token_embeddings-self-wte-input_ids"],
          jax: ["gptnsmall.__call__.token_embeddings-nn-embed-self-vocab_size-self-n_embd-name-wte-input_ids"],
        },
        focusRef: {
          pytorch: "gptnsmall.self-wte-nn-embedding-vocab_size-n_embd",
          jax: "gptnsmall.__call__.token_embeddings-nn-embed-self-vocab_size-self-n_embd-name-wte-input_ids",
        },
        includeChildRefs: false,
      },
      {
        id: "wpe",
        label: "wpe",
        type: "PositionEmbedding",
        kind: "embedding",
        badges: ["1024", "768"],
        sourceRefs: {
          pytorch: ["gptnsmall.self-wpe-nn-embedding-n_ctx-n_embd","gptnsmall.forward.positions-torch-arange-step_count-device-input_ids-device","gptnsmall.forward.position_embeddings-self-wpe-positions","gptnsmall.forward.position_embeddings-position_embeddings-none"],
          jax: ["gptnsmall.__call__.positions-jnp-arange-step_count","gptnsmall.__call__.position_embeddings-nn-embed-self-n_ctx-self-n_embd-name-wpe-positions","gptnsmall.__call__.position_embeddings-position_embeddings-none"],
        },
        focusRef: {
          pytorch: "gptnsmall.self-wpe-nn-embedding-n_ctx-n_embd",
          jax: "gptnsmall.__call__.positions-jnp-arange-step_count",
        },
        includeChildRefs: false,
      },
      {
        id: "drop",
        label: "drop",
        type: "Dropout",
        kind: "dropout",
        sourceRefs: {
          pytorch: ["gptnsmall.self-drop-nn-dropout-n","gptnsmall.forward.x-self-drop-x"],
          jax: ["gptnsmall.__call__.x-nn-dropout-n-deterministic-true-name-drop-x"],
        },
        focusRef: {
          pytorch: "gptnsmall.self-drop-nn-dropout-n",
          jax: "gptnsmall.__call__.x-nn-dropout-n-deterministic-true-name-drop-x",
        },
        includeChildRefs: false,
      },
      ...Array.from({ length: 12 }, (_, index) => makeGpt2Block(index, index === 3)),
    ],
  },
  vit: {
    stats: "196 patches · 12 encoder blocks · 12 heads",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        sourceRefs: {
          pytorch: ["visiontransformer.forward.x-self-patch_embed-x"],
          jax: ["visiontransformer.__call__.x-patchembed-self-embed_dim-x"],
        },
        focusRef: {
          pytorch: "visiontransformer.forward.x-self-patch_embed-x",
          jax: "visiontransformer.__call__.x-patchembed-self-embed_dim-x",
        },
        includeChildRefs: false,
      },
      {
        id: "patch_embed",
        label: "patch_embed",
        type: "Conv2d projection",
        kind: "conv",
        badges: ["16x16", "196 tokens", "768"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["patchembed.self-num_patches-patches_per_side-n","visiontransformer.self-norm-nn-layernorm-embed_dim"],
          jax: ["visiontransformer.__call__.x-patchembed-self-embed_dim-x"],
        },
        focusRef: {
          pytorch: "vit.patch_embed.proj",
          jax: "patchembed.__call__.projection-nn-conv-self-embed_dim-self-patch_size-self-patch_size-stride",
        },
        includeChildRefs: true,
        children: [
          {
            id: "patch_embed.proj",
            label: "proj",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->768", "k=16", "s=16"],
            sourceRefs: {
              pytorch: ["vit.patch_embed.proj","vit.patch_embed.project"],
              jax: ["patchembed.__call__.projection-nn-conv-self-embed_dim-self-patch_size-self-patch_size-stride","patchembed.__call__.x-projection-x"],
            },
            focusRef: {
              pytorch: "vit.patch_embed.proj",
              jax: "patchembed.__call__.projection-nn-conv-self-embed_dim-self-patch_size-self-patch_size-stride",
            },
            includeChildRefs: false,
          },
          {
            id: "patch_embed.flatten",
            label: "flatten patches",
            type: "Flatten",
            kind: "reshape",
            badges: ["14x14 -> 196"],
            sourceRefs: {
              pytorch: ["patchembed.forward.x-x-transpose-n-n","patchembed.forward.return-x"],
              jax: ["patchembed.__call__.batch_size-x-shape-n","patchembed.__call__.sequence_shape-batch_size-n-self-embed_dim","patchembed.__call__.x-x-reshape-sequence_shape","patchembed.__call__.return-x"],
            },
            focusRef: {
              pytorch: "patchembed.forward.x-x-transpose-n-n",
              jax: "patchembed.__call__.batch_size-x-shape-n",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: [],
          jax: [],
        },
        focusRef: {
          pytorch: "vit.tokens.cls",
          jax: "visiontransformer.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-embed_dim",
        },
        includeChildRefs: true,
        children: [
          {
            id: "tokens.cls",
            label: "cls_token",
            type: "LearnedToken",
            kind: "embedding",
            badges: ["1 x 768"],
            sourceRefs: {
              pytorch: ["vit.tokens.cls","vit.tokens.concat"],
              jax: ["visiontransformer.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-embed_dim","visiontransformer.__call__.batch_size-x-shape-n","visiontransformer.__call__.cls_shape-batch_size-n-n","visiontransformer.__call__.cls-jnp-tile-cls-cls_shape","visiontransformer.__call__.x-jnp-concatenate-cls-x-axis-n"],
            },
            focusRef: {
              pytorch: "vit.tokens.cls",
              jax: "visiontransformer.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-embed_dim",
            },
            includeChildRefs: false,
          },
          {
            id: "tokens.position",
            label: "pos_embed",
            type: "PositionEmbedding",
            kind: "embedding",
            badges: ["197 x 768"],
            sourceRefs: {
              pytorch: ["vit.tokens.position","visiontransformer.forward.x-x-self-pos_embed"],
              jax: ["visiontransformer.__call__.pos_init-nn-initializers-normal-n","visiontransformer.__call__.pos_shape-n-x-shape-n-self-embed_dim","visiontransformer.__call__.pos-self-param-pos_embed-pos_init-pos_shape","visiontransformer.__call__.x-x-pos"],
            },
            focusRef: {
              pytorch: "vit.tokens.position",
              jax: "visiontransformer.__call__.pos_init-nn-initializers-normal-n",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["visiontransformer.def-forward-self-x","visiontransformer.forward.x-self-norm-x","visiontransformer.forward.cls_output-x-n"],
          jax: ["visiontransformer.__call__.for-_-in-range-self-depth","visiontransformer.__call__.x-encoderblock-self-embed_dim-self-num_heads-x"],
        },
        focusRef: {
          pytorch: "visiontransformer.def-forward-self-x",
          jax: "visiontransformer.__call__.for-_-in-range-self-depth",
        },
        includeChildRefs: false,
        children: Array.from({ length: 12 }, (_, index) => makeVitBlock(index, index === 3)),
      },
      {
        id: "norm",
        label: "encoder_norm",
        type: "LayerNorm",
        kind: "norm",
        badges: ["CLS"],
        sourceRefs: {
          pytorch: ["visiontransformer.self-norm-nn-layernorm-embed_dim","visiontransformer.forward.x-self-norm-x"],
          jax: ["visiontransformer.__call__.x-nn-layernorm-name-encoder_norm-x","visiontransformer.__call__.cls_output-x-n"],
        },
        focusRef: {
          pytorch: "visiontransformer.self-norm-nn-layernorm-embed_dim",
          jax: "visiontransformer.__call__.x-nn-layernorm-name-encoder_norm-x",
        },
        includeChildRefs: false,
      },
      {
        id: "head",
        label: "head",
        type: "Linear",
        kind: "linear",
        badges: ["768->1000"],
        sourceRefs: {
          pytorch: ["visiontransformer.self-head-nn-linear-embed_dim-num_classes","vit.head"],
          jax: ["visiontransformer.__call__.logits-nn-dense-self-num_classes-name-head-cls_output"],
        },
        focusRef: {
          pytorch: "visiontransformer.self-head-nn-linear-embed_dim-num_classes",
          jax: "visiontransformer.__call__.logits-nn-dense-self-num_classes-name-head-cls_output",
        },
        includeChildRefs: false,
      },
    ],
  },
  clip: {
    stats: "dual encoders · shared embedding space · contrastive logits",
    nodes: [
      {
        id: "image_input",
        label: "image input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        sourceRefs: {
          pytorch: ["clip.def-forward-self-images-input_ids","clip.forward.image_features-self-visual-images"],
          jax: ["clip.def-__call__-self-images-input_ids","clip.__call__.image_features-visionencoder-self-image_size-self-patch_size-self-vision"],
        },
        focusRef: {
          pytorch: "clip.def-forward-self-images-input_ids",
          jax: "clip.def-__call__-self-images-input_ids",
        },
        includeChildRefs: false,
      },
      {
        id: "text_input",
        label: "text input",
        type: "TokenIds",
        kind: "input",
        badges: ["77 tokens"],
        sourceRefs: {
          pytorch: ["clip.def-forward-self-images-input_ids","clip.forward.text_features-self-text-input_ids"],
          jax: ["clip.def-__call__-self-images-input_ids","clip.__call__.text_features-textencoder-self-vocab_size-self-context_length-self-text_"],
        },
        focusRef: {
          pytorch: "clip.def-forward-self-images-input_ids",
          jax: "clip.def-__call__-self-images-input_ids",
        },
        includeChildRefs: false,
      },
      {
        id: "vision_encoder",
        label: "Vision Encoder",
        type: "ViT-B/32 tower",
        kind: "group",
        summary: "patches -> CLS embedding",
        badges: ["224px", "32px patches", "512-d"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-visionencoder-nn-module","visionencoder.self-patch_embed-nn-convnd-n-width-kernel_size-patch_size-stride-patch_s","visionencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la","visionencoder.self-proj-nn-linear-width-embed_dim-bias-false","visionencoder.def-forward-self-images","visionencoder.forward.x-self-patch_embed-images","visionencoder.forward.x-x-self-pos_embed","visionencoder.forward.for-block-in-self-blocks","visionencoder.forward.x-self-ln_post-x","visionencoder.forward.image_features-self-proj-cls_output","clip.self-visual-visionencoder-image_size-patch_size-vision_width-vision_laye","clip.forward.image_features-self-visual-images"],
          jax: ["class-visionencoder-nn-module","visionencoder.__call__.projection-nn-conv-self-width-self-patch_size-self-patch_size-strides-se","visionencoder.__call__.x-projection-images","visionencoder.__call__.for-_-in-range-self-layers","visionencoder.__call__.x-transformerblock-self-width-self-heads-x","visionencoder.__call__.x-nn-layernorm-name-ln_post-x","visionencoder.__call__.image_features-nn-dense-self-embed_dim-use_bias-false-name-proj-cls_outp","clip.__call__.image_features-visionencoder-self-image_size-self-patch_size-self-vision"],
        },
        focusRef: {
          pytorch: "class-visionencoder-nn-module",
          jax: "class-visionencoder-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "vision_encoder.patch_embed",
            label: "patch_embed",
            type: "Conv2d projection",
            kind: "conv",
            badges: ["32x32", "49 patches"],
            sourceRefs: {
              pytorch: ["visionencoder.self-patch_embed-nn-convnd-n-width-kernel_size-patch_size-stride-patch_s","visionencoder.forward.x-self-patch_embed-images","visionencoder.forward.x-x-flatten-n","visionencoder.forward.x-x-transpose-n-n"],
              jax: ["visionencoder.__call__.projection-nn-conv-self-width-self-patch_size-self-patch_size-strides-se","visionencoder.__call__.x-projection-images","visionencoder.__call__.x-x-reshape-batch_size-n-self-width"],
            },
            focusRef: {
              pytorch: "visionencoder.self-patch_embed-nn-convnd-n-width-kernel_size-patch_size-stride-patch_s",
              jax: "visionencoder.__call__.projection-nn-conv-self-width-self-patch_size-self-patch_size-strides-se",
            },
            includeChildRefs: false,
          },
          {
            id: "vision_encoder.cls_position",
            label: "CLS + position",
            type: "Learned tokens",
            kind: "embedding",
            badges: ["50 tokens"],
            sourceRefs: {
              pytorch: ["visionencoder.self-cls_token-nn-parameter-torch-zeros-n-n-width","visionencoder.self-pos_embed-nn-parameter-torch-zeros-n-patch_count-n-width","visionencoder.forward.cls-self-cls_token-expand-batch_size-n-n","visionencoder.forward.x-torch-cat-cls-x-dim-n","visionencoder.forward.x-x-self-pos_embed"],
              jax: ["visionencoder.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-width","visionencoder.__call__.cls-jnp-tile-cls-batch_size-n-n","visionencoder.__call__.x-jnp-concatenate-cls-x-axis-n","visionencoder.__call__.pos_embed-self-param-pos_embed-nn-initializers-zeros-n-x-shape-n-self-wi","visionencoder.__call__.x-x-pos_embed"],
            },
            focusRef: {
              pytorch: "visionencoder.self-cls_token-nn-parameter-torch-zeros-n-n-width",
              jax: "visionencoder.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-width",
            },
            includeChildRefs: false,
          },
          {
            id: "vision_encoder.blocks",
            label: "visual blocks",
            type: "Transformer stack",
            kind: "attention",
            badges: ["12 blocks", "12 heads"],
            sourceRefs: {
              pytorch: ["visionencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la","visionencoder.forward.for-block-in-self-blocks","visionencoder.forward.x-block-x"],
              jax: ["visionencoder.__call__.for-_-in-range-self-layers","visionencoder.__call__.x-transformerblock-self-width-self-heads-x"],
            },
            focusRef: {
              pytorch: "visionencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la",
              jax: "visionencoder.__call__.for-_-in-range-self-layers",
            },
            includeChildRefs: false,
          },
          {
            id: "vision_encoder.projection",
            label: "image projection",
            type: "Linear",
            kind: "linear",
            badges: ["768->512"],
            sourceRefs: {
              pytorch: ["visionencoder.self-proj-nn-linear-width-embed_dim-bias-false","visionencoder.forward.x-self-ln_post-x","visionencoder.forward.cls_output-x-n","visionencoder.forward.image_features-self-proj-cls_output"],
              jax: ["visionencoder.__call__.x-nn-layernorm-name-ln_post-x","visionencoder.__call__.cls_output-x-n","visionencoder.__call__.image_features-nn-dense-self-embed_dim-use_bias-false-name-proj-cls_outp"],
            },
            focusRef: {
              pytorch: "visionencoder.self-proj-nn-linear-width-embed_dim-bias-false",
              jax: "visionencoder.__call__.x-nn-layernorm-name-ln_post-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["class-textencoder-nn-module","textencoder.self-token_embedding-nn-embedding-vocab_size-width","textencoder.self-blocks-nn-modulelist-transformerblock-width-heads-for-_-in-range-la","textencoder.self-text_projection-nn-linear-width-embed_dim-bias-false","textencoder.mask-torch-tril-torch-ones-context_length-context_length","textencoder.self-register_buffer-causal_mask-mask-view-n-n-context_length-context_le","textencoder.def-forward-self-input_ids","textencoder.forward.token_embeddings-self-token_embedding-input_ids","textencoder.forward.x-token_embeddings-position_embeddings","textencoder.forward.mask-self-causal_mask-token_count-token_count","textencoder.forward.for-block-in-self-blocks","textencoder.forward.x-self-ln_final-x","textencoder.forward.pooled-x-batch_indices-eot_indices","textencoder.forward.text_features-self-text_projection-pooled","clip.self-text-textencoder-vocab_size-context_length-text_width-text_layers-t","clip.forward.text_features-self-text-input_ids"],
          jax: ["class-textencoder-nn-module","textencoder.__call__.token_embeddings-nn-embed-self-vocab_size-self-width-name-token_embeddin","textencoder.__call__.pos_embed-self-param-pos_embed-nn-initializers-zeros-n-self-context_leng","textencoder.__call__.full_mask-jnp-tril-jnp-ones-self-context_length-self-context_length","textencoder.__call__.mask-mask-reshape-n-n-token_count-token_count","textencoder.__call__.for-_-in-range-self-layers","textencoder.__call__.x-nn-layernorm-name-ln_final-x","textencoder.__call__.pooled-x-batch_indices-eot_indices","textencoder.__call__.text_features-nn-dense-self-embed_dim-use_bias-false-name-text_projectio","clip.__call__.text_features-textencoder-self-vocab_size-self-context_length-self-text_"],
        },
        focusRef: {
          pytorch: "class-textencoder-nn-module",
          jax: "class-textencoder-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "text_encoder.embedding",
            label: "token + position",
            type: "Embedding",
            kind: "embedding",
            badges: ["49408", "77"],
            sourceRefs: {
              pytorch: ["textencoder.self-token_embedding-nn-embedding-vocab_size-width","textencoder.self-pos_embed-nn-parameter-torch-zeros-n-context_length-width","textencoder.forward.token_embeddings-self-token_embedding-input_ids","textencoder.forward.position_embeddings-self-pos_embed-token_count","textencoder.forward.x-token_embeddings-position_embeddings"],
              jax: ["textencoder.__call__.token_embeddings-nn-embed-self-vocab_size-self-width-name-token_embeddin","textencoder.__call__.pos_embed-self-param-pos_embed-nn-initializers-zeros-n-self-context_leng","textencoder.__call__.position_embeddings-pos_embed-token_count","textencoder.__call__.x-token_embeddings-position_embeddings"],
            },
            focusRef: {
              pytorch: "textencoder.self-token_embedding-nn-embedding-vocab_size-width",
              jax: "textencoder.__call__.token_embeddings-nn-embed-self-vocab_size-self-width-name-token_embeddin",
            },
            includeChildRefs: false,
          },
          {
            id: "text_encoder.causal_mask",
            label: "causal mask",
            type: "Lower triangle",
            kind: "attention",
            badges: ["text-only"],
            sourceRefs: {
              pytorch: ["textencoder.mask-torch-tril-torch-ones-context_length-context_length","textencoder.self-register_buffer-causal_mask-mask-view-n-n-context_length-context_le","textencoder.forward.mask-self-causal_mask-token_count-token_count","textencoder.forward.for-block-in-self-blocks","textencoder.forward.x-block-x-mask"],
              jax: ["textencoder.__call__.full_mask-jnp-tril-jnp-ones-self-context_length-self-context_length","textencoder.__call__.mask-full_mask-token_count-token_count","textencoder.__call__.mask-mask-reshape-n-n-token_count-token_count","textencoder.__call__.for-_-in-range-self-layers","textencoder.__call__.x-transformerblock-self-width-self-heads-x-mask"],
            },
            focusRef: {
              pytorch: "textencoder.mask-torch-tril-torch-ones-context_length-context_length",
              jax: "textencoder.__call__.full_mask-jnp-tril-jnp-ones-self-context_length-self-context_length",
            },
            includeChildRefs: false,
          },
          {
            id: "text_encoder.eot_pool",
            label: "EOT pool",
            type: "Token gather",
            kind: "reshape",
            badges: ["end token"],
            sourceRefs: {
              pytorch: ["textencoder.forward.x-self-ln_final-x","textencoder.forward.eot_indices-input_ids-argmax-dim-n","textencoder.forward.batch_indices-torch-arange-input_ids-size-n-device-input_ids-device","textencoder.forward.pooled-x-batch_indices-eot_indices"],
              jax: ["textencoder.__call__.x-nn-layernorm-name-ln_final-x","textencoder.__call__.eot_indices-jnp-argmax-input_ids-axis-n","textencoder.__call__.batch_indices-jnp-arange-input_ids-shape-n","textencoder.__call__.pooled-x-batch_indices-eot_indices"],
            },
            focusRef: {
              pytorch: "textencoder.forward.x-self-ln_final-x",
              jax: "textencoder.__call__.x-nn-layernorm-name-ln_final-x",
            },
            includeChildRefs: false,
          },
          {
            id: "text_encoder.projection",
            label: "text projection",
            type: "Linear",
            kind: "linear",
            badges: ["512->512"],
            sourceRefs: {
              pytorch: ["textencoder.self-text_projection-nn-linear-width-embed_dim-bias-false","textencoder.forward.text_features-self-text_projection-pooled"],
              jax: ["textencoder.__call__.text_features-nn-dense-self-embed_dim-use_bias-false-name-text_projectio"],
            },
            focusRef: {
              pytorch: "textencoder.self-text_projection-nn-linear-width-embed_dim-bias-false",
              jax: "textencoder.__call__.text_features-nn-dense-self-embed_dim-use_bias-false-name-text_projectio",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["clip.self-visual-visionencoder-image_size-patch_size-vision_width-vision_laye","clip.self-text-textencoder-vocab_size-context_length-text_width-text_layers-t","clip.forward.image_features-self-visual-images","clip.forward.text_features-self-text-input_ids"],
          jax: ["clip.__call__.image_features-visionencoder-self-image_size-self-patch_size-self-vision","clip.__call__.text_features-textencoder-self-vocab_size-self-context_length-self-text_"],
        },
        focusRef: {
          pytorch: "clip.self-visual-visionencoder-image_size-patch_size-vision_width-vision_laye",
          jax: "clip.__call__.image_features-visionencoder-self-image_size-self-patch_size-self-vision",
        },
        includeChildRefs: true,
        children: [
          {
            id: "contrastive_logits.normalize",
            label: "normalize",
            type: "Unit vectors",
            kind: "norm",
            badges: ["cosine"],
            sourceRefs: {
              pytorch: ["clip.forward.image_features-f-normalize-image_features-dim-n","clip.forward.text_features-f-normalize-text_features-dim-n"],
              jax: ["clip.__call__.image_features-image_features-jnp-linalg-norm-image_features-axis-n-keep","clip.__call__.text_features-text_features-jnp-linalg-norm-text_features-axis-n-keepdim"],
            },
            focusRef: {
              pytorch: "clip.forward.image_features-f-normalize-image_features-dim-n",
              jax: "clip.__call__.image_features-image_features-jnp-linalg-norm-image_features-axis-n-keep",
            },
            includeChildRefs: false,
          },
          {
            id: "contrastive_logits.temperature",
            label: "logit_scale",
            type: "Learned temperature",
            kind: "linear",
            badges: ["exp"],
            sourceRefs: {
              pytorch: ["clip.self-logit_scale-nn-parameter-torch-ones-torch-log-torch-tensor-n-n","clip.forward.logit_scale-self-logit_scale-exp"],
              jax: ["clip.__call__.logit_scale-self-param-logit_scale-lambda-key-jnp-log-jnp-array-n-n","clip.__call__.logit_scale-jnp-exp-logit_scale"],
            },
            focusRef: {
              pytorch: "clip.self-logit_scale-nn-parameter-torch-ones-torch-log-torch-tensor-n-n",
              jax: "clip.__call__.logit_scale-self-param-logit_scale-lambda-key-jnp-log-jnp-array-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "contrastive_logits.matrix",
            label: "similarity matrix",
            type: "Matmul",
            kind: "head",
            badges: ["image @ text.T"],
            sourceRefs: {
              pytorch: ["clip.forward.text_features_t-text_features-t","clip.forward.logits_per_image-logit_scale-image_features-text_features_t","clip.forward.logits_per_text-logits_per_image-t"],
              jax: ["clip.__call__.text_features_t-jnp-swapaxes-text_features-n-n","clip.__call__.logits_per_image-logit_scale-image_features-text_features_t","clip.__call__.logits_per_text-jnp-swapaxes-logits_per_image-n-n"],
            },
            focusRef: {
              pytorch: "clip.forward.text_features_t-text_features-t",
              jax: "clip.__call__.text_features_t-jnp-swapaxes-text_features-n-n",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  ddpm: {
    stats: "forward noising · timestep-conditioned U-Net · reverse denoising",
    nodes: [
      {
        id: "clean_input",
        label: "clean input",
        type: "Image x0",
        kind: "input",
        badges: ["3 x 32 x 32"],
        sourceRefs: {
          pytorch: ["ddpm.def-q_sample-self-clean_images-timesteps-noise","clean_images-torch-zeros-n-n-n-n"],
          jax: ["ddpm.def-q_sample-self-clean_images-timesteps-noise","clean_images-jnp-zeros-n-n-n-n"],
        },
        focusRef: {
          pytorch: "ddpm.def-q_sample-self-clean_images-timesteps-noise",
          jax: "ddpm.def-q_sample-self-clean_images-timesteps-noise",
        },
        includeChildRefs: false,
      },
      {
        id: "noise_timestep",
        label: "noise + t",
        type: "Noise schedule inputs",
        kind: "input",
        badges: ["epsilon", "t"],
        sourceRefs: {
          pytorch: ["ddpm.def-_extract-self-values-timesteps-target_shape","ddpm._extract.batch_size-timesteps-size-n","ddpm._extract.gathered-values-gather-n-timesteps","ddpm.def-q_sample-self-clean_images-timesteps-noise","ddpm.q_sample.sqrt_alpha-self-_extract-self-sqrt_alphas_cumprod-timesteps-clean_images","ddpm.q_sample.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti","timesteps-torch-tensor-n-n","noise-torch-randn_like-clean_images"],
          jax: ["ddpm.def-extract-self-values-timesteps-target_shape","ddpm.extract.gathered-values-timesteps","ddpm.def-q_sample-self-clean_images-timesteps-noise","ddpm.q_sample.sqrt_alpha-self-extract-jnp-sqrt-alphas_cumprod-timesteps-clean_images-s","ddpm.q_sample.sqrt_one_minus_alpha-self-extract-jnp-sqrt-n-alphas_cumprod-timesteps-cl","timesteps-jnp-array-n-n","noise-jnp-ones_like-clean_images"],
        },
        focusRef: {
          pytorch: "ddpm.def-_extract-self-values-timesteps-target_shape",
          jax: "ddpm.def-extract-self-values-timesteps-target_shape",
        },
        includeChildRefs: false,
      },
      {
        id: "schedule",
        label: "Diffusion Schedule",
        type: "Beta schedule",
        kind: "group",
        summary: "fixed alphas and variances",
        badges: ["1000 steps", "linear beta"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["ddpm.betas-torch-linspace-beta_start-beta_end-timesteps","ddpm.alphas-n-betas","ddpm.alphas_cumprod-torch-cumprod-alphas-dim-n","ddpm.alphas_cumprod_prev-torch-cat-torch-ones-n-alphas_cumprod-n","ddpm.posterior_variance-betas-n-alphas_cumprod_prev-n-alphas_cumprod","ddpm.self-register_buffer-betas-betas","ddpm.self-register_buffer-sqrt_alphas_cumprod-torch-sqrt-alphas_cumprod","ddpm.self-register_buffer-sqrt_one_minus_alphas_cumprod-torch-sqrt-n-alphas_c","ddpm.self-register_buffer-sqrt_recip_alphas-torch-sqrt-n-alphas","ddpm.self-register_buffer-posterior_variance-posterior_variance","ddpm.def-_extract-self-values-timesteps-target_shape","ddpm._extract.gathered-values-gather-n-timesteps","ddpm._extract.broadcast_shape-batch_size-n-len-target_shape-n","ddpm._extract.gathered-gathered-reshape-broadcast_shape"],
          jax: ["ddpm.def-schedule-self","ddpm.schedule.betas-jnp-linspace-self-beta_start-self-beta_end-self-timesteps","ddpm.schedule.alphas-n-betas","ddpm.schedule.alphas_cumprod-jnp-cumprod-alphas-axis-n","ddpm.schedule.alphas_cumprod_prev-jnp-concatenate-jnp-ones-n-alphas_cumprod-n-axis-n","ddpm.schedule.posterior_variance-betas-n-alphas_cumprod_prev-n-alphas_cumprod","ddpm.def-extract-self-values-timesteps-target_shape","ddpm.extract.gathered-values-timesteps","ddpm.extract.broadcast_shape-timesteps-shape-n-n-len-target_shape-n","ddpm.extract.gathered-gathered-reshape-broadcast_shape"],
        },
        focusRef: {
          pytorch: "ddpm.betas-torch-linspace-beta_start-beta_end-timesteps",
          jax: "ddpm.def-schedule-self",
        },
        includeChildRefs: false,
        children: [
          {
            id: "schedule.betas",
            label: "betas",
            type: "Linear schedule",
            kind: "linear",
            badges: ["1e-4 -> 0.02"],
            sourceRefs: {
              pytorch: ["ddpm.betas-torch-linspace-beta_start-beta_end-timesteps","ddpm.self-register_buffer-betas-betas"],
              jax: ["ddpm.schedule.betas-jnp-linspace-self-beta_start-self-beta_end-self-timesteps"],
            },
            focusRef: {
              pytorch: "ddpm.betas-torch-linspace-beta_start-beta_end-timesteps",
              jax: "ddpm.schedule.betas-jnp-linspace-self-beta_start-self-beta_end-self-timesteps",
            },
            includeChildRefs: false,
          },
          {
            id: "schedule.alpha_bar",
            label: "alpha_bar",
            type: "Cumulative product",
            kind: "reshape",
            badges: ["signal power"],
            sourceRefs: {
              pytorch: ["ddpm.alphas-n-betas","ddpm.alphas_cumprod-torch-cumprod-alphas-dim-n","ddpm.self-register_buffer-sqrt_alphas_cumprod-torch-sqrt-alphas_cumprod","ddpm.self-register_buffer-sqrt_one_minus_alphas_cumprod-torch-sqrt-n-alphas_c"],
              jax: ["ddpm.schedule.alphas-n-betas","ddpm.schedule.alphas_cumprod-jnp-cumprod-alphas-axis-n","ddpm.q_sample.sqrt_alpha-self-extract-jnp-sqrt-alphas_cumprod-timesteps-clean_images-s","ddpm.q_sample.sqrt_one_minus_alpha-self-extract-jnp-sqrt-n-alphas_cumprod-timesteps-cl"],
            },
            focusRef: {
              pytorch: "ddpm.alphas-n-betas",
              jax: "ddpm.schedule.alphas-n-betas",
            },
            includeChildRefs: false,
          },
          {
            id: "schedule.posterior_variance",
            label: "posterior variance",
            type: "Reverse variance",
            kind: "linear",
            badges: ["p(x_{t-1}|x_t)"],
            sourceRefs: {
              pytorch: ["ddpm.alphas_cumprod_prev-torch-cat-torch-ones-n-alphas_cumprod-n","ddpm.posterior_variance-betas-n-alphas_cumprod_prev-n-alphas_cumprod","ddpm.self-register_buffer-posterior_variance-posterior_variance","ddpm.p_mean_variance.variance-self-_extract-self-posterior_variance-timesteps-noisy_images-sh"],
              jax: ["ddpm.schedule.alphas_cumprod_prev-jnp-concatenate-jnp-ones-n-alphas_cumprod-n-axis-n","ddpm.schedule.posterior_variance-betas-n-alphas_cumprod_prev-n-alphas_cumprod","ddpm.p_mean_variance.variance-self-extract-posterior_variance-timesteps-noisy_images-shape"],
            },
            focusRef: {
              pytorch: "ddpm.alphas_cumprod_prev-torch-cat-torch-ones-n-alphas_cumprod-n",
              jax: "ddpm.schedule.alphas_cumprod_prev-jnp-concatenate-jnp-ones-n-alphas_cumprod-n-axis-n",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["ddpm.def-q_sample-self-clean_images-timesteps-noise","ddpm.q_sample.return-noisy_images","noisy_images-model-q_sample-clean_images-timesteps-noise"],
          jax: ["ddpm.def-q_sample-self-clean_images-timesteps-noise","ddpm.q_sample._-_-alphas_cumprod-_-self-schedule","ddpm.q_sample.return-noisy_images","noisy_images-model-q_sample-clean_images-timesteps-noise","train_step.loss_fn.noisy_images-model-q_sample-clean_images-timesteps-noise"],
        },
        focusRef: {
          pytorch: "ddpm.def-q_sample-self-clean_images-timesteps-noise",
          jax: "ddpm.def-q_sample-self-clean_images-timesteps-noise",
        },
        includeChildRefs: true,
        children: [
          {
            id: "forward_noising.signal",
            label: "sqrt alpha_bar",
            type: "Signal scale",
            kind: "linear",
            sourceRefs: {
              pytorch: ["ddpm.q_sample.sqrt_alpha-self-_extract-self-sqrt_alphas_cumprod-timesteps-clean_images","ddpm.q_sample.noisy_images-sqrt_alpha-clean_images-sqrt_one_minus_alpha-noise"],
              jax: ["ddpm.q_sample.sqrt_alpha-self-extract-jnp-sqrt-alphas_cumprod-timesteps-clean_images-s","ddpm.q_sample.noisy_images-sqrt_alpha-clean_images-sqrt_one_minus_alpha-noise"],
            },
            focusRef: {
              pytorch: "ddpm.q_sample.sqrt_alpha-self-_extract-self-sqrt_alphas_cumprod-timesteps-clean_images",
              jax: "ddpm.q_sample.sqrt_alpha-self-extract-jnp-sqrt-alphas_cumprod-timesteps-clean_images-s",
            },
            includeChildRefs: false,
          },
          {
            id: "forward_noising.noise",
            label: "sqrt one-minus",
            type: "Noise scale",
            kind: "linear",
            sourceRefs: {
              pytorch: ["ddpm.q_sample.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti","ddpm.q_sample.noisy_images-sqrt_alpha-clean_images-sqrt_one_minus_alpha-noise"],
              jax: ["ddpm.q_sample.sqrt_one_minus_alpha-self-extract-jnp-sqrt-n-alphas_cumprod-timesteps-cl","ddpm.q_sample.noisy_images-sqrt_alpha-clean_images-sqrt_one_minus_alpha-noise"],
            },
            focusRef: {
              pytorch: "ddpm.q_sample.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti",
              jax: "ddpm.q_sample.sqrt_one_minus_alpha-self-extract-jnp-sqrt-n-alphas_cumprod-timesteps-cl",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "time_embedding",
        label: "Time Embedding",
        type: "Sinusoidal MLP",
        kind: "embedding",
        badges: ["sin/cos", "MLP"],
        sourceRefs: {
          pytorch: ["class-timeembedding-nn-module","timeembedding.self-mlp-nn-sequential","timeembedding.nn-linear-width-width-n","timeembedding.nn-silu","timeembedding.nn-linear-width-n-width","timeembedding.def-forward-self-timesteps","timeembedding.forward.half_width-self-width-n","timeembedding.forward.frequencies-torch-arange-half_width-device-device-dtype-torch-floatn","timeembedding.forward.frequencies-torch-exp-math-log-n-frequencies","timeembedding.forward.angles-timesteps-float-none-frequencies-none","timeembedding.forward.embedding-torch-cat-angles-sin-angles-cos-dim-n","timeembedding.forward.embedding-self-mlp-embedding","unetdenoiser.self-time_embedding-timeembedding-time_width","unetdenoiser.forward.time_emb-self-time_embedding-timesteps"],
          jax: ["class-timeembedding-nn-module","timeembedding.def-__call__-self-timesteps","timeembedding.__call__.half_width-self-width-n","timeembedding.__call__.frequencies-jnp-arange-half_width-dtype-jnp-floatn","timeembedding.__call__.frequencies-jnp-exp-jnp-log-n-frequencies","timeembedding.__call__.angles-timesteps-astype-jnp-floatn-none-frequencies-none","timeembedding.__call__.embedding-jnp-concatenate-jnp-sin-angles-jnp-cos-angles-axis-n","timeembedding.__call__.embedding-nn-dense-self-width-n-name-fcn-embedding","timeembedding.__call__.embedding-nn-silu-embedding","timeembedding.__call__.embedding-nn-dense-self-width-name-fcn-embedding","unetdenoiser.__call__.time_emb-timeembedding-self-time_width-timesteps"],
        },
        focusRef: {
          pytorch: "class-timeembedding-nn-module",
          jax: "class-timeembedding-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: "unet_denoiser",
        label: "U-Net Denoiser",
        type: "epsilon_theta(x_t, t)",
        kind: "group",
        summary: "timestep-conditioned noise predictor",
        badges: ["skip concat", "same image shape"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["class-unetdenoiser-nn-module","unetdenoiser.self-time_embedding-timeembedding-time_width","unetdenoiser.self-input_conv-nn-convnd-image_channels-base_channels-kernel_size-n-pad","unetdenoiser.self-downn-residualblock-base_channels-base_channels-time_width","unetdenoiser.self-downsamplen-downsample-base_channels","unetdenoiser.self-downn-residualblock-base_channels-base_channels-n-time_width","unetdenoiser.self-downsamplen-downsample-base_channels-n","unetdenoiser.self-middle-residualblock-base_channels-n-base_channels-n-time_width","unetdenoiser.self-upsamplen-upsample-base_channels-n","unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-n-time_width","unetdenoiser.self-upsamplen-upsample-base_channels-n.2","unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-time_width","unetdenoiser.self-out_norm-nn-groupnorm-n-base_channels","unetdenoiser.self-out_conv-nn-convnd-base_channels-image_channels-kernel_size-n-paddi","unetdenoiser.def-forward-self-noisy_images-timesteps","unetdenoiser.forward.time_emb-self-time_embedding-timesteps","unetdenoiser.forward.x-self-input_conv-noisy_images","unetdenoiser.forward.skipn-self-downn-x-time_emb","unetdenoiser.forward.skipn-self-downn-x-time_emb.2","unetdenoiser.forward.x-self-middle-x-time_emb","unetdenoiser.forward.x-torch-cat-x-skipn-dim-n","unetdenoiser.forward.x-torch-cat-x-skipn-dim-n.2","unetdenoiser.forward.predicted_noise-self-out_conv-x","ddpm.self-denoiser-unetdenoiser-image_channels-base_channels-time_width","ddpm.forward.predicted_noise-self-denoiser-noisy_images-timesteps"],
          jax: ["class-unetdenoiser-nn-module","unetdenoiser.__call__.time_emb-timeembedding-self-time_width-timesteps","unetdenoiser.__call__.x-nn-conv-self-base_channels-n-n-padding-same-name-input_conv-noisy_imag","unetdenoiser.__call__.skipn-residualblock-self-base_channels-self-time_width-x-time_emb","unetdenoiser.__call__.x-downsample-self-base_channels-skipn","unetdenoiser.__call__.skipn-residualblock-self-base_channels-n-self-time_width-x-time_emb","unetdenoiser.__call__.x-downsample-self-base_channels-n-skipn","unetdenoiser.__call__.x-residualblock-self-base_channels-n-self-time_width-x-time_emb","unetdenoiser.__call__.x-upsample-self-base_channels-n-x","unetdenoiser.__call__.x-jnp-concatenate-x-skipn-axis-n","unetdenoiser.__call__.x-residualblock-self-base_channels-n-self-time_width-x-time_emb.2","unetdenoiser.__call__.x-upsample-self-base_channels-n-x.2","unetdenoiser.__call__.x-jnp-concatenate-x-skipn-axis-n.2","unetdenoiser.__call__.x-residualblock-self-base_channels-self-time_width-x-time_emb","unetdenoiser.__call__.x-nn-groupnorm-num_groups-n-name-out_norm-x","unetdenoiser.__call__.predicted_noise-nn-conv-self-image_channels-n-n-padding-same-name-out_co","ddpm.__call__.predicted_noise-unetdenoiser-self-image_channels-self-base_channels-self"],
        },
        focusRef: {
          pytorch: "class-unetdenoiser-nn-module",
          jax: "class-unetdenoiser-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "unet_denoiser.encoder",
            label: "encoder",
            type: "Down path",
            kind: "conv",
            badges: ["32 -> 8"],
            sourceRefs: {
              pytorch: ["unetdenoiser.self-input_conv-nn-convnd-image_channels-base_channels-kernel_size-n-pad","unetdenoiser.self-downn-residualblock-base_channels-base_channels-time_width","unetdenoiser.self-downsamplen-downsample-base_channels","unetdenoiser.self-downn-residualblock-base_channels-base_channels-n-time_width","unetdenoiser.self-downsamplen-downsample-base_channels-n","unetdenoiser.forward.x-self-input_conv-noisy_images","unetdenoiser.forward.skipn-self-downn-x-time_emb","unetdenoiser.forward.x-self-downsamplen-skipn","unetdenoiser.forward.skipn-self-downn-x-time_emb.2","unetdenoiser.forward.x-self-downsamplen-skipn.2"],
              jax: ["unetdenoiser.__call__.x-nn-conv-self-base_channels-n-n-padding-same-name-input_conv-noisy_imag","unetdenoiser.__call__.skipn-residualblock-self-base_channels-self-time_width-x-time_emb","unetdenoiser.__call__.x-downsample-self-base_channels-skipn","unetdenoiser.__call__.skipn-residualblock-self-base_channels-n-self-time_width-x-time_emb","unetdenoiser.__call__.x-downsample-self-base_channels-n-skipn"],
            },
            focusRef: {
              pytorch: "unetdenoiser.self-input_conv-nn-convnd-image_channels-base_channels-kernel_size-n-pad",
              jax: "unetdenoiser.__call__.x-nn-conv-self-base_channels-n-n-padding-same-name-input_conv-noisy_imag",
            },
            includeChildRefs: false,
          },
          {
            id: "unet_denoiser.bottleneck",
            label: "bottleneck",
            type: "ResidualBlock",
            kind: "residual",
            badges: ["base*4"],
            sourceRefs: {
              pytorch: ["unetdenoiser.self-middle-residualblock-base_channels-n-base_channels-n-time_width","unetdenoiser.forward.x-self-middle-x-time_emb"],
              jax: ["unetdenoiser.__call__.x-residualblock-self-base_channels-n-self-time_width-x-time_emb"],
            },
            focusRef: {
              pytorch: "unetdenoiser.self-middle-residualblock-base_channels-n-base_channels-n-time_width",
              jax: "unetdenoiser.__call__.x-residualblock-self-base_channels-n-self-time_width-x-time_emb",
            },
            includeChildRefs: false,
          },
          {
            id: "unet_denoiser.decoder",
            label: "decoder + skips",
            type: "Up path",
            kind: "concat",
            badges: ["skip1", "skip2"],
            sourceRefs: {
              pytorch: ["unetdenoiser.self-upsamplen-upsample-base_channels-n","unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-n-time_width","unetdenoiser.self-upsamplen-upsample-base_channels-n.2","unetdenoiser.self-upn-residualblock-base_channels-n-base_channels-time_width","unetdenoiser.forward.x-self-upsamplen-x","unetdenoiser.forward.x-torch-cat-x-skipn-dim-n","unetdenoiser.forward.x-self-upn-x-time_emb","unetdenoiser.forward.x-self-upsamplen-x.2","unetdenoiser.forward.x-torch-cat-x-skipn-dim-n.2","unetdenoiser.forward.x-self-upn-x-time_emb.2"],
              jax: ["unetdenoiser.__call__.x-upsample-self-base_channels-n-x","unetdenoiser.__call__.x-jnp-concatenate-x-skipn-axis-n","unetdenoiser.__call__.x-residualblock-self-base_channels-n-self-time_width-x-time_emb.2","unetdenoiser.__call__.x-upsample-self-base_channels-n-x.2","unetdenoiser.__call__.x-jnp-concatenate-x-skipn-axis-n.2","unetdenoiser.__call__.x-residualblock-self-base_channels-self-time_width-x-time_emb"],
            },
            focusRef: {
              pytorch: "unetdenoiser.self-upsamplen-upsample-base_channels-n",
              jax: "unetdenoiser.__call__.x-upsample-self-base_channels-n-x",
            },
            includeChildRefs: false,
          },
          {
            id: "unet_denoiser.noise_head",
            label: "noise head",
            type: "Conv2d",
            kind: "head",
            badges: ["3 channels"],
            sourceRefs: {
              pytorch: ["unetdenoiser.self-out_norm-nn-groupnorm-n-base_channels","unetdenoiser.self-out_conv-nn-convnd-base_channels-image_channels-kernel_size-n-paddi","unetdenoiser.forward.x-self-out_norm-x","unetdenoiser.forward.x-f-silu-x","unetdenoiser.forward.predicted_noise-self-out_conv-x"],
              jax: ["unetdenoiser.__call__.x-nn-groupnorm-num_groups-n-name-out_norm-x","unetdenoiser.__call__.x-nn-silu-x","unetdenoiser.__call__.predicted_noise-nn-conv-self-image_channels-n-n-padding-same-name-out_co"],
            },
            focusRef: {
              pytorch: "unetdenoiser.self-out_norm-nn-groupnorm-n-base_channels",
              jax: "unetdenoiser.__call__.x-nn-groupnorm-num_groups-n-name-out_norm-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["ddpm.def-forward-self-noisy_images-timesteps","ddpm.forward.predicted_noise-self-denoiser-noisy_images-timesteps","ddpm.forward.return-predicted_noise","noisy_images-model-q_sample-clean_images-timesteps-noise","predicted_noise-model-noisy_images-timesteps","loss-f-mse_loss-predicted_noise-noise"],
          jax: ["ddpm.nn-compact","ddpm.__call__.predicted_noise-unetdenoiser-self-image_channels-self-base_channels-self","ddpm.__call__.return-predicted_noise","train_step.loss_fn.noisy_images-model-q_sample-clean_images-timesteps-noise","train_step.loss_fn.predicted_noise-model-apply-current_params-noisy_images-timesteps","train_step.loss_fn.loss-jnp-mean-predicted_noise-noise-n"],
        },
        focusRef: {
          pytorch: "ddpm.def-forward-self-noisy_images-timesteps",
          jax: "ddpm.nn-compact",
        },
        includeChildRefs: false,
      },
      {
        id: "reverse_step",
        label: "Reverse Step",
        type: "p_sample",
        kind: "group",
        summary: "x_t -> x_{t-1}",
        badges: ["mean", "variance"],
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["ddpm.def-p_mean_variance-self-noisy_images-timesteps","ddpm.def-p_sample-self-noisy_images-timesteps-noise"],
          jax: ["ddpm.def-p_mean_variance-self-params-noisy_images-timesteps","ddpm.def-p_sample-self-params-noisy_images-timesteps-noise"],
        },
        focusRef: {
          pytorch: "ddpm.def-p_mean_variance-self-noisy_images-timesteps",
          jax: "ddpm.def-p_mean_variance-self-params-noisy_images-timesteps",
        },
        includeChildRefs: true,
        children: [
          {
            id: "reverse_step.mean",
            label: "reverse mean",
            type: "Gaussian mean",
            kind: "linear",
            sourceRefs: {
              pytorch: ["ddpm.p_mean_variance.predicted_noise-self-noisy_images-timesteps","ddpm.p_mean_variance.betas_t-self-_extract-self-betas-timesteps-noisy_images-shape","ddpm.p_mean_variance.sqrt_one_minus_alpha-self-_extract-self-sqrt_one_minus_alphas_cumprod-ti","ddpm.p_mean_variance.sqrt_recip_alpha-self-_extract-self-sqrt_recip_alphas-timesteps-noisy_im","ddpm.p_mean_variance.model_mean-sqrt_recip_alpha-noisy_images-betas_t-predicted_noise-sqrt_on"],
              jax: ["ddpm.p_mean_variance.predicted_noise-self-apply-params-noisy_images-timesteps","ddpm.p_mean_variance.betas-alphas-alphas_cumprod-posterior_variance-self-schedule","ddpm.p_mean_variance.betas_t-self-extract-betas-timesteps-noisy_images-shape","ddpm.p_mean_variance.sqrt_one_minus_alpha-self-extract-jnp-sqrt-n-alphas_cumprod-timesteps-no","ddpm.p_mean_variance.sqrt_recip_alpha-self-extract-jnp-sqrt-n-alphas-timesteps-noisy_images-s","ddpm.p_mean_variance.model_mean-sqrt_recip_alpha-noisy_images-betas_t-predicted_noise-sqrt_on"],
            },
            focusRef: {
              pytorch: "ddpm.p_mean_variance.predicted_noise-self-noisy_images-timesteps",
              jax: "ddpm.p_mean_variance.predicted_noise-self-apply-params-noisy_images-timesteps",
            },
            includeChildRefs: false,
          },
          {
            id: "reverse_step.sample",
            label: "sample x_{t-1}",
            type: "Gaussian sample",
            kind: "head",
            sourceRefs: {
              pytorch: ["ddpm.p_mean_variance.variance-self-_extract-self-posterior_variance-timesteps-noisy_images-sh","ddpm.p_sample.model_mean-variance-self-p_mean_variance-noisy_images-timesteps","ddpm.p_sample.nonzero_mask-timesteps-n-float-none-none-none","ddpm.p_sample.sample-model_mean-nonzero_mask-torch-sqrt-variance-noise"],
              jax: ["ddpm.p_mean_variance.variance-self-extract-posterior_variance-timesteps-noisy_images-shape","ddpm.p_sample.model_mean-variance-self-p_mean_variance-params-noisy_images-timesteps","ddpm.p_sample.nonzero_mask-timesteps-n-astype-jnp-floatn-none-none-none","ddpm.p_sample.sample-model_mean-nonzero_mask-jnp-sqrt-variance-noise"],
            },
            focusRef: {
              pytorch: "ddpm.p_mean_variance.variance-self-_extract-self-posterior_variance-timesteps-noisy_images-sh",
              jax: "ddpm.p_mean_variance.variance-self-extract-posterior_variance-timesteps-noisy_images-shape",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  resnet18: {
    stats: resnetVariants[0].stats,
    variants: resnetVariants,
    nodes: resnetVariants[0].nodes,
  },
  widenet: {
    stats: "WRN-28-10 · width factor 10 · pre-activation residual blocks",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "CIFARImage",
        kind: "input",
        badges: ["3 x 32 x 32"],
        sourceRefs: {
          pytorch: ["widenet.input"],
          jax: ["widenet.__call__.n-n","widenet.__call__.padding-same","widenet.__call__.use_bias-false","widenet.__call__.name-convn","widenet.__call__.x"],
        },
        focusRef: {
          pytorch: "widenet.input",
          jax: "widenet.__call__.n-n",
        },
        includeChildRefs: false,
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv2d",
        kind: "conv",
        summary: "3x3 shallow stem",
        badges: ["3->16", "32x32"],
        sourceRefs: {
          pytorch: ["widenet.widths-n","widenet.kernel_size-n","widenet.stride-n","widenet.padding-n","widenet.bias-false","widenet.code.7","widenet.self-layern-self-_make_layer","widenet.widths-n.2"],
          jax: ["widenet.__call__.x-nn-conv","widenet.__call__.widths-n","widenet.__call__.n-n","widenet.__call__.padding-same","widenet.__call__.use_bias-false","widenet.__call__.name-convn","widenet.__call__.x"],
        },
        focusRef: {
          pytorch: "widenet.widths-n",
          jax: "widenet.__call__.x-nn-conv",
        },
        includeChildRefs: false,
      },
      {
        id: "layer1",
        label: "layer1",
        type: "WideResidualStage",
        kind: "group",
        summary: "4 widened blocks",
        badges: ["160 ch", "32x32"],
        sourceRefs: {
          pytorch: ["widenet.widths-n.3","widenet.block_count","widenet.stride-n.2","widenet.dropout_rate-dropout_rate","widenet.code.8","widenet.self-layern-self-_make_layer.2","widenet.widths-n.4","widenet._make_layer.layers","widenet._make_layer.widebasicblock","widenet._make_layer.in_channels","widenet._make_layer.out_channels","widenet._make_layer.stride-stride","widenet._make_layer.dropout_rate-dropout_rate","widenet._make_layer.code.2","widenet._make_layer.code.3","widenet._make_layer.for-_-in-range-n-blocks","widenet.forward.x-self-layern-x.3"],
          jax: ["widenet.__call__.x-self-_stage","widenet.__call__.x.2","widenet.__call__.widths-n.2","widenet.__call__.block_count","widenet.__call__.stride-n","widenet.__call__.train-train","widenet.__call__.name-layern","widenet.__call__.code.7","widenet._stage.for-index-in-range-blocks","widenet._stage.block_stride-stride-if-index-n-else-n","widenet._stage.use_projection-index-n","widenet._stage.block_name-f-name-index","widenet._stage.x-widebasicblock","widenet._stage.channels","widenet._stage.stride-block_stride","widenet._stage.dropout_rate-self-dropout_rate","widenet._stage.use_projection-use_projection","widenet._stage.name-block_name","widenet._stage.x-train-train"],
        },
        focusRef: {
          pytorch: "widenet.widths-n.3",
          jax: "widenet.__call__.x-self-_stage",
        },
        includeChildRefs: false,
        children: [
          {
            id: "layer1.0",
            label: "block.0",
            type: "WideBasicBlock",
            kind: "residual",
            summary: "16->160 projection",
            sourceRefs: {
              pytorch: ["widebasicblock.in_channels.2","widebasicblock.out_channels.2","widebasicblock.kernel_size-n","widebasicblock.if-stride-n-or-in_channels-out_channels","widebasicblock.self-shortcut-nn-convnd","widebasicblock.in_channels.3","widebasicblock.forward.shortcut-x","widebasicblock.forward.if-self-shortcut-is-not-none","widebasicblock.forward.shortcut-self-shortcut-x","widebasicblock.forward.out-self-relu-out","widebasicblock.forward.out-self-bnn-out","widebasicblock.forward.out-out-shortcut","widebasicblock.forward.return-out","widenet._make_layer.out_channels","widenet.forward.x-self-layern-x.3"],
              jax: ["class-widebasicblock-nn-module","widebasicblock.out_channels-int","widebasicblock.stride-int-n","widebasicblock.dropout_rate-float-n","widebasicblock.use_projection-bool-false","widebasicblock.nn-compact","widebasicblock.def-__call__-self-x-train-false","widebasicblock.__call__.shortcut-x","widebasicblock.__call__.if-self-use_projection","widebasicblock.__call__.shortcut-nn-conv","widebasicblock.__call__.self-out_channels","widebasicblock.__call__.n-n","widebasicblock.__call__.strides-self-stride-self-stride","widebasicblock.__call__.use_bias-false","widebasicblock.__call__.name-shortcut","widebasicblock.__call__.x","widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-x","widebasicblock.__call__.y-nn-relu-y","widebasicblock.__call__.y-nn-conv","widebasicblock.__call__.self-out_channels.2","widebasicblock.__call__.n-n.2","widebasicblock.__call__.strides-self-stride-self-stride.2","widebasicblock.__call__.padding-same","widebasicblock.__call__.use_bias-false.2","widebasicblock.__call__.name-convn","widebasicblock.__call__.y","widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y","widebasicblock.__call__.y-nn-relu-y.2","widebasicblock.__call__.if-self-dropout_rate-n","widebasicblock.__call__.y-nn-dropout","widebasicblock.__call__.rate-self-dropout_rate","widebasicblock.__call__.name-dropout","widebasicblock.__call__.y-deterministic-not-train","widebasicblock.__call__.y-nn-conv.2","widebasicblock.__call__.self-out_channels.3","widebasicblock.__call__.n-n.3","widebasicblock.__call__.padding-same.2","widebasicblock.__call__.use_bias-false.3","widebasicblock.__call__.name-convn.2","widebasicblock.__call__.y.2","widebasicblock.__call__.y-y-shortcut","widebasicblock.__call__.return-y"],
            },
            focusRef: {
              pytorch: "widebasicblock.in_channels.2",
              jax: "class-widebasicblock-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "layer1.1-3",
            label: "block.1-3",
            type: "WideBasicBlock x3",
            kind: "residual",
            summary: "identity skips",
            sourceRefs: {
              pytorch: ["widenet._make_layer.out_channels.2","widenet._make_layer.out_channels.3","widenet._make_layer.dropout_rate-dropout_rate.2","widenet._make_layer.code.4","widenet._make_layer.layers-append-block","widenet._make_layer.stage-nn-sequential-layers","widenet.forward.x-self-layern-x.3"],
              jax: ["widenet._stage.for-index-in-range-blocks","widenet._stage.block_stride-stride-if-index-n-else-n","widenet._stage.use_projection-index-n","widenet._stage.block_name-f-name-index","widenet._stage.x-widebasicblock","widenet._stage.channels","widenet._stage.stride-block_stride","widenet._stage.dropout_rate-self-dropout_rate","widenet._stage.use_projection-use_projection","widenet._stage.name-block_name","widenet._stage.x-train-train"],
            },
            focusRef: {
              pytorch: "widenet._make_layer.out_channels.2",
              jax: "widenet._stage.for-index-in-range-blocks",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["widenet.widths-n.5","widenet.block_count.2","widenet.stride-n.3","widenet.dropout_rate-dropout_rate.2","widenet.code.9","widenet.self-layern-self-_make_layer.3","widenet.widths-n.6","widenet._make_layer.layers","widenet._make_layer.widebasicblock","widenet._make_layer.in_channels","widenet._make_layer.out_channels","widenet._make_layer.stride-stride","widenet._make_layer.dropout_rate-dropout_rate","widenet._make_layer.code.2","widenet._make_layer.code.3","widenet._make_layer.for-_-in-range-n-blocks"],
          jax: ["widenet.__call__.x-self-_stage.2","widenet.__call__.x.3","widenet.__call__.widths-n.3","widenet.__call__.block_count.2","widenet.__call__.stride-n.2","widenet.__call__.train-train.2","widenet.__call__.name-layern.2","widenet.__call__.code.8","widenet._stage.for-index-in-range-blocks","widenet._stage.block_stride-stride-if-index-n-else-n","widenet._stage.use_projection-index-n","widenet._stage.block_name-f-name-index","widenet._stage.x-widebasicblock","widenet._stage.channels","widenet._stage.stride-block_stride","widenet._stage.dropout_rate-self-dropout_rate","widenet._stage.use_projection-use_projection","widenet._stage.name-block_name","widenet._stage.x-train-train"],
        },
        focusRef: {
          pytorch: "widenet.widths-n.5",
          jax: "widenet.__call__.x-self-_stage.2",
        },
        includeChildRefs: false,
        children: [
          {
            id: "layer2.0",
            label: "block.0",
            type: "WideBasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            defaultExpanded: true,
            sourceRefs: {
              pytorch: ["widebasicblock.self-bnn-nn-batchnormnd-in_channels","widebasicblock.self-relu-nn-relu-inplace-true","widebasicblock.self-convn-nn-convnd","widebasicblock.in_channels.2","widebasicblock.self-dropout_rate-dropout_rate","widebasicblock.out_channels.3","widebasicblock.if-stride-n-or-in_channels-out_channels","widebasicblock.self-shortcut-nn-convnd","widebasicblock.in_channels.3","widebasicblock.out_channels.5","widebasicblock.kernel_size-n.3","widebasicblock.stride-stride.2","widebasicblock.bias-false.3","widebasicblock.code.6","widebasicblock.forward.shortcut-x","widebasicblock.forward.if-self-shortcut-is-not-none","widebasicblock.forward.shortcut-self-shortcut-x","widebasicblock.forward.out-self-bnn-x","widebasicblock.forward.out-self-relu-out","widebasicblock.forward.out-self-convn-out","widebasicblock.forward.out-self-bnn-out","widebasicblock.forward.out-self-relu-out.2","widebasicblock.forward.if-self-dropout_rate-n","widebasicblock.forward.out-out-shortcut","widebasicblock.forward.return-out","widenet._make_layer.out_channels"],
              jax: ["class-widebasicblock-nn-module","widebasicblock.out_channels-int","widebasicblock.stride-int-n","widebasicblock.dropout_rate-float-n","widebasicblock.use_projection-bool-false","widebasicblock.nn-compact","widebasicblock.def-__call__-self-x-train-false","widebasicblock.__call__.shortcut-x","widebasicblock.__call__.if-self-use_projection","widebasicblock.__call__.shortcut-nn-conv","widebasicblock.__call__.self-out_channels","widebasicblock.__call__.n-n","widebasicblock.__call__.strides-self-stride-self-stride","widebasicblock.__call__.use_bias-false","widebasicblock.__call__.name-shortcut","widebasicblock.__call__.x","widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-x","widebasicblock.__call__.y-nn-relu-y","widebasicblock.__call__.y-nn-conv","widebasicblock.__call__.self-out_channels.2","widebasicblock.__call__.n-n.2","widebasicblock.__call__.strides-self-stride-self-stride.2","widebasicblock.__call__.padding-same","widebasicblock.__call__.use_bias-false.2","widebasicblock.__call__.name-convn","widebasicblock.__call__.y","widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y","widebasicblock.__call__.y-nn-relu-y.2","widebasicblock.__call__.if-self-dropout_rate-n","widebasicblock.__call__.y-nn-dropout","widebasicblock.__call__.rate-self-dropout_rate","widebasicblock.__call__.name-dropout","widebasicblock.__call__.y-deterministic-not-train","widebasicblock.__call__.y-nn-conv.2","widebasicblock.__call__.self-out_channels.3","widebasicblock.__call__.n-n.3","widebasicblock.__call__.padding-same.2","widebasicblock.__call__.use_bias-false.3","widebasicblock.__call__.name-convn.2","widebasicblock.__call__.y.2","widebasicblock.__call__.y-y-shortcut","widebasicblock.__call__.return-y"],
            },
            focusRef: {
              pytorch: "widebasicblock.self-bnn-nn-batchnormnd-in_channels",
              jax: "class-widebasicblock-nn-module",
            },
            includeChildRefs: false,
            children: [
              {
                id: "layer2.0.bn1",
                label: "bn1",
                type: "BatchNorm2d",
                kind: "norm",
                badges: ["160"],
                sourceRefs: {
                  pytorch: ["widebasicblock.self-relu-nn-relu-inplace-true","widebasicblock.forward.out-self-relu-out"],
                  jax: ["widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-x"],
                },
                focusRef: {
                  pytorch: "widebasicblock.self-relu-nn-relu-inplace-true",
                  jax: "widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-x",
                },
                includeChildRefs: false,
              },
              {
                id: "layer2.0.conv1",
                label: "conv1",
                type: "Conv2d",
                kind: "conv",
                badges: ["160->320", "k=3", "s=2"],
                sourceRefs: {
                  pytorch: ["widebasicblock.in_channels.2","widebasicblock.out_channels.2","widebasicblock.kernel_size-n","widebasicblock.stride-stride","widebasicblock.padding-n","widebasicblock.bias-false","widebasicblock.code.4","widebasicblock.self-bnn-nn-batchnormnd-out_channels","widebasicblock.forward.out-self-bnn-out"],
                  jax: ["widebasicblock.__call__.y-nn-conv","widebasicblock.__call__.self-out_channels.2","widebasicblock.__call__.n-n.2","widebasicblock.__call__.strides-self-stride-self-stride.2","widebasicblock.__call__.padding-same","widebasicblock.__call__.use_bias-false.2","widebasicblock.__call__.name-convn","widebasicblock.__call__.y"],
                },
                focusRef: {
                  pytorch: "widebasicblock.in_channels.2",
                  jax: "widebasicblock.__call__.y-nn-conv",
                },
                includeChildRefs: false,
              },
              {
                id: "layer2.0.conv2",
                label: "conv2",
                type: "Conv2d",
                kind: "conv",
                badges: ["320->320", "k=3"],
                sourceRefs: {
                  pytorch: ["widebasicblock.out_channels.3","widebasicblock.out_channels.4","widebasicblock.kernel_size-n.2","widebasicblock.stride-n.2","widebasicblock.padding-n.2","widebasicblock.bias-false.2","widebasicblock.code.5","widebasicblock.self-shortcut-none"],
                  jax: ["widebasicblock.__call__.y-nn-conv.2","widebasicblock.__call__.self-out_channels.3","widebasicblock.__call__.n-n.3","widebasicblock.__call__.padding-same.2","widebasicblock.__call__.use_bias-false.3","widebasicblock.__call__.name-convn.2","widebasicblock.__call__.y.2"],
                },
                focusRef: {
                  pytorch: "widebasicblock.out_channels.3",
                  jax: "widebasicblock.__call__.y-nn-conv.2",
                },
                includeChildRefs: false,
              },
              {
                id: "layer2.0.shortcut",
                label: "shortcut",
                type: "ProjectionSkip",
                kind: "conv",
                badges: ["160->320", "s=2"],
                sourceRefs: {
                  pytorch: ["widebasicblock.if-stride-n-or-in_channels-out_channels","widebasicblock.self-shortcut-nn-convnd","widebasicblock.in_channels.3","widebasicblock.out_channels.5","widebasicblock.kernel_size-n.3","widebasicblock.stride-stride.2","widebasicblock.bias-false.3","widebasicblock.code.6","widebasicblock.forward.shortcut-self-shortcut-x"],
                  jax: ["widebasicblock.__call__.shortcut-x","widebasicblock.__call__.if-self-use_projection","widebasicblock.__call__.shortcut-nn-conv","widebasicblock.__call__.self-out_channels","widebasicblock.__call__.n-n","widebasicblock.__call__.strides-self-stride-self-stride","widebasicblock.__call__.use_bias-false","widebasicblock.__call__.name-shortcut","widebasicblock.__call__.x"],
                },
                focusRef: {
                  pytorch: "widebasicblock.if-stride-n-or-in_channels-out_channels",
                  jax: "widebasicblock.__call__.shortcut-x",
                },
                includeChildRefs: false,
              },
              {
                id: "layer2.0.add",
                label: "add",
                type: "ResidualAdd",
                kind: "residual",
                sourceRefs: {
                  pytorch: ["widebasicblock.forward.out-out-shortcut","widebasicblock.forward.return-out"],
                  jax: ["widebasicblock.__call__.y-y-shortcut"],
                },
                focusRef: {
                  pytorch: "widebasicblock.forward.out-out-shortcut",
                  jax: "widebasicblock.__call__.y-y-shortcut",
                },
                includeChildRefs: false,
              },
            ],
          },
          {
            id: "layer2.1-3",
            label: "block.1-3",
            type: "WideBasicBlock x3",
            kind: "residual",
            summary: "identity skips",
            sourceRefs: {
              pytorch: ["widenet._make_layer.out_channels.2","widenet._make_layer.out_channels.3","widenet._make_layer.dropout_rate-dropout_rate.2","widenet._make_layer.code.4","widenet._make_layer.layers-append-block","widenet._make_layer.stage-nn-sequential-layers"],
              jax: ["widenet._stage.for-index-in-range-blocks","widenet._stage.block_stride-stride-if-index-n-else-n","widenet._stage.use_projection-index-n","widenet._stage.block_name-f-name-index","widenet._stage.x-widebasicblock","widenet._stage.channels","widenet._stage.stride-block_stride","widenet._stage.dropout_rate-self-dropout_rate","widenet._stage.use_projection-use_projection","widenet._stage.name-block_name","widenet._stage.x-train-train"],
            },
            focusRef: {
              pytorch: "widenet._make_layer.out_channels.2",
              jax: "widenet._stage.for-index-in-range-blocks",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["widenet.widths-n.7","widenet.block_count.3","widenet.stride-n.4","widenet.dropout_rate-dropout_rate.3","widenet.code.10","widenet.self-bn-nn-batchnormnd-widths-n","widenet.self-relu-nn-relu-inplace-true","widenet._make_layer.layers","widenet._make_layer.widebasicblock","widenet._make_layer.in_channels","widenet._make_layer.out_channels","widenet._make_layer.stride-stride","widenet._make_layer.dropout_rate-dropout_rate","widenet._make_layer.code.2","widenet._make_layer.code.3","widenet._make_layer.for-_-in-range-n-blocks"],
          jax: ["widenet.__call__.x-self-_stage.3","widenet.__call__.x.4","widenet.__call__.widths-n.4","widenet.__call__.block_count.3","widenet.__call__.stride-n.3","widenet.__call__.train-train.3","widenet.__call__.name-layern.3","widenet.__call__.code.9","widenet._stage.for-index-in-range-blocks","widenet._stage.block_stride-stride-if-index-n-else-n","widenet._stage.use_projection-index-n","widenet._stage.block_name-f-name-index","widenet._stage.x-widebasicblock","widenet._stage.channels","widenet._stage.stride-block_stride","widenet._stage.dropout_rate-self-dropout_rate","widenet._stage.use_projection-use_projection","widenet._stage.name-block_name","widenet._stage.x-train-train"],
        },
        focusRef: {
          pytorch: "widenet.widths-n.7",
          jax: "widenet.__call__.x-self-_stage.3",
        },
        includeChildRefs: false,
        children: [
          {
            id: "layer3.0",
            label: "block.0",
            type: "WideBasicBlock",
            kind: "residual",
            summary: "stride 2 + projection",
            sourceRefs: {
              pytorch: ["widebasicblock.in_channels.2","widebasicblock.out_channels.3","widebasicblock.if-stride-n-or-in_channels-out_channels","widebasicblock.self-shortcut-nn-convnd","widebasicblock.in_channels.3","widebasicblock.forward.out-self-relu-out","widebasicblock.forward.out-self-bnn-out","widebasicblock.forward.return-out","widenet._make_layer.out_channels"],
              jax: ["class-widebasicblock-nn-module","widebasicblock.out_channels-int","widebasicblock.stride-int-n","widebasicblock.dropout_rate-float-n","widebasicblock.use_projection-bool-false","widebasicblock.nn-compact","widebasicblock.def-__call__-self-x-train-false","widebasicblock.__call__.shortcut-x","widebasicblock.__call__.if-self-use_projection","widebasicblock.__call__.shortcut-nn-conv","widebasicblock.__call__.self-out_channels","widebasicblock.__call__.n-n","widebasicblock.__call__.strides-self-stride-self-stride","widebasicblock.__call__.use_bias-false","widebasicblock.__call__.name-shortcut","widebasicblock.__call__.x","widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-x","widebasicblock.__call__.y-nn-relu-y","widebasicblock.__call__.y-nn-conv","widebasicblock.__call__.self-out_channels.2","widebasicblock.__call__.n-n.2","widebasicblock.__call__.strides-self-stride-self-stride.2","widebasicblock.__call__.padding-same","widebasicblock.__call__.use_bias-false.2","widebasicblock.__call__.name-convn","widebasicblock.__call__.y","widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y","widebasicblock.__call__.y-nn-relu-y.2","widebasicblock.__call__.if-self-dropout_rate-n","widebasicblock.__call__.y-nn-dropout","widebasicblock.__call__.rate-self-dropout_rate","widebasicblock.__call__.name-dropout","widebasicblock.__call__.y-deterministic-not-train","widebasicblock.__call__.y-nn-conv.2","widebasicblock.__call__.self-out_channels.3","widebasicblock.__call__.n-n.3","widebasicblock.__call__.padding-same.2","widebasicblock.__call__.use_bias-false.3","widebasicblock.__call__.name-convn.2","widebasicblock.__call__.y.2","widebasicblock.__call__.y-y-shortcut","widebasicblock.__call__.return-y"],
            },
            focusRef: {
              pytorch: "widebasicblock.in_channels.2",
              jax: "class-widebasicblock-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "layer3.1-3",
            label: "block.1-3",
            type: "WideBasicBlock x3",
            kind: "residual",
            summary: "identity skips",
            sourceRefs: {
              pytorch: ["widenet._make_layer.out_channels.2","widenet._make_layer.out_channels.3","widenet._make_layer.dropout_rate-dropout_rate.2","widenet._make_layer.code.4","widenet._make_layer.layers-append-block","widenet._make_layer.stage-nn-sequential-layers"],
              jax: ["widenet._stage.for-index-in-range-blocks","widenet._stage.block_stride-stride-if-index-n-else-n","widenet._stage.use_projection-index-n","widenet._stage.block_name-f-name-index","widenet._stage.x-widebasicblock","widenet._stage.channels","widenet._stage.stride-block_stride","widenet._stage.dropout_rate-self-dropout_rate","widenet._stage.use_projection-use_projection","widenet._stage.name-block_name","widenet._stage.x-train-train"],
            },
            focusRef: {
              pytorch: "widenet._make_layer.out_channels.2",
              jax: "widenet._stage.for-index-in-range-blocks",
            },
            includeChildRefs: false,
          },
        ],
      },
      {
        id: "head",
        label: "head",
        type: "BN-ReLU-Pool-FC",
        kind: "group",
        summary: "global average pool",
        sourceRefs: {
          pytorch: ["widenet.forward.x-self-relu-x","widenet.forward.x-torch-flatten-x-n"],
          jax: ["widenet.__call__.x-nn-relu-x"],
        },
        focusRef: {
          pytorch: "widenet.self-fc-nn-linear-widths-n-num_classes",
          jax: "widenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-bn-x",
        },
        includeChildRefs: true,
        children: [
          {
            id: "head.bn",
            label: "bn",
            type: "BatchNorm2d",
            kind: "norm",
            badges: ["640"],
            sourceRefs: {
              pytorch: ["widenet.self-bn-nn-batchnormnd-widths-n","widenet.head.bn"],
              jax: ["widenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-bn-x"],
            },
            focusRef: {
              pytorch: "widenet.self-bn-nn-batchnormnd-widths-n",
              jax: "widenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-bn-x",
            },
            includeChildRefs: false,
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["8x8"],
            sourceRefs: {
              pytorch: ["widenet.forward.x-f-avg_poolnd-x-kernel_size-n"],
              jax: ["widenet.__call__.x-jnp-mean-x-axis-n-n"],
            },
            focusRef: {
              pytorch: "widenet.forward.x-f-avg_poolnd-x-kernel_size-n",
              jax: "widenet.__call__.x-jnp-mean-x-axis-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "head.flatten",
            label: "flatten",
            type: "Flatten",
            kind: "reshape",
            badges: ["640"],
            sourceRefs: {
              pytorch: ["widenet.forward.x-torch-flatten-x-n"],
              jax: ["widenet.__call__.x-jnp-mean-x-axis-n-n"],
            },
            focusRef: {
              pytorch: "widenet.forward.x-torch-flatten-x-n",
              jax: "widenet.__call__.x-jnp-mean-x-axis-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "head.fc",
            label: "fc",
            type: "Linear",
            kind: "linear",
            badges: ["640->10"],
            sourceRefs: {
              pytorch: ["widenet.self-fc-nn-linear-widths-n-num_classes","widenet.forward.logits-self-fc-x"],
              jax: ["widenet.__call__.logits-nn-dense-self-num_classes-name-fc-x"],
            },
            focusRef: {
              pytorch: "widenet.self-fc-nn-linear-widths-n-num_classes",
              jax: "widenet.__call__.logits-nn-dense-self-num_classes-name-fc-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  densenet: {
    stats: "4 dense blocks · 58 dense layers · feature concatenation",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        sourceRefs: {
          pytorch: ["densenet.forward.x-self-stem-x"],
          jax: ["densenet.__call__.strides-n-n","densenet.__call__.padding-same","densenet.__call__.use_bias-false","densenet.__call__.name-stem_conv","densenet.__call__.x","densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_norm-x","densenet.__call__.x-nn-relu-x","densenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same"],
        },
        focusRef: {
          pytorch: "densenet.forward.x-self-stem-x",
          jax: "densenet.__call__.strides-n-n",
        },
        includeChildRefs: false,
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-ReLU-Pool",
        kind: "group",
        summary: "7x7 stride 2",
        badges: ["3->64", "56x56"],
        sourceRefs: {
          pytorch: ["densenet.num_init_features","densenet.kernel_size-n","densenet.forward.x-self-features-x"],
          jax: ["class-densenet-nn-module","densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_norm-x","densenet.__call__.x-nn-relu-x"],
        },
        focusRef: {
          pytorch: "densenet.num_init_features",
          jax: "class-densenet-nn-module",
        },
        includeChildRefs: true,
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->64", "k=7", "s=2"],
            sourceRefs: {
              pytorch: ["densenet.stride-n","densenet.padding-n","densenet.bias-false","densenet.code.4","densenet.nn-batchnormnd-num_init_features","densenet.nn-relu-inplace-true","densenet.nn-maxpoolnd-kernel_size-n-stride-n-padding-n","densenet.code.5"],
              jax: ["densenet.__call__.x-nn-conv","densenet.__call__.self-num_init_features","densenet.__call__.n-n","densenet.__call__.strides-n-n","densenet.__call__.padding-same","densenet.__call__.use_bias-false","densenet.__call__.name-stem_conv","densenet.__call__.x"],
            },
            focusRef: {
              pytorch: "densenet.stride-n",
              jax: "densenet.__call__.x-nn-conv",
            },
            includeChildRefs: false,
          },
          {
            id: "stem.pool",
            label: "maxpool",
            type: "MaxPool2d",
            kind: "pool",
            badges: ["56x56"],
            sourceRefs: {
              pytorch: ["densenet.blocks"],
              jax: ["densenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same"],
            },
            focusRef: {
              pytorch: "densenet.blocks",
              jax: "densenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["denseblock.self","denseblock.layer_count","denseblock.in_channels","denseblock.growth_rate","denseblock.dropout_rate-n","denseblock.code","denseblock.super-__init__","denseblock.layers","denseblock.current_channels-in_channels","denseblock.for-_-in-range-layer_count","denseblock.layer-denselayer","denseblock.current_channels","denseblock.growth_rate.2","denseblock.dropout_rate-dropout_rate","denseblock.code.4","denseblock.layers-append-layer","denseblock.current_channels-current_channels-growth_rate","denseblock.self-layers-nn-modulelist-layers","denseblock.self-out_channels-current_channels","denseblock.def-forward-self-x","denseblock.forward.out-x","denseblock.forward.for-layer-in-self-layers","denseblock.forward.out-layer-out","denseblock.forward.return-out","class-transition-nn-module","densenet.dropout_rate-dropout_rate","densenet.code.8","densenet.blocks-append-dense_block","densenet.num_features-dense_block-out_channels","densenet.is_last_block-index-len-block_config-n","densenet.if-not-is_last_block","densenet.out_features-int-num_features-compression","densenet.transition-transition-num_features-out_features","densenet.forward.x-self-norm-x","densenet.forward.x-f-relu-x-inplace-true"],
          jax: ["densenet.__call__.num_features-self-num_init_features","densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
        },
        focusRef: {
          pytorch: "denseblock.self",
          jax: "densenet.__call__.num_features-self-num_init_features",
        },
        includeChildRefs: false,
        children: [
          {
            id: "denseblock1.layer1",
            label: "layer.1",
            type: "DenseLayer",
            kind: "concat",
            summary: "append growth features",
            badges: ["64+32"],
            sourceRefs: {
              pytorch: ["denselayer.bottleneck_channels-bottleneck_width-growth_rate","denselayer.self-normn-nn-batchnormnd-in_channels","denselayer.self-relun-nn-relu-inplace-true","denselayer.in_channels.2","denselayer.self-relun-nn-relu-inplace-true.2","denselayer.bottleneck_channels.2","denselayer.forward.out-self-normn-x","denselayer.forward.out-self-relun-out","denselayer.forward.out-self-normn-out","denselayer.forward.if-self-dropout_rate-n","denselayer.forward.features-x-out","denselayer.forward.out-torch-cat-features-dim-n","denselayer.forward.return-out","denseblock.growth_rate.2"],
              jax: ["class-denselayer-nn-module","denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-x","denselayer.__call__.y-nn-relu-y","denselayer.__call__.y-nn-conv","denselayer.__call__.bottleneck_channels","denselayer.__call__.n-n","denselayer.__call__.use_bias-false","denselayer.__call__.name-convn","denselayer.__call__.y","denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-y","denselayer.__call__.y-nn-relu-y.2","denselayer.__call__.y-nn-conv.2","denselayer.__call__.self-growth_rate","denselayer.__call__.n-n.2","denselayer.__call__.padding-same","denselayer.__call__.use_bias-false.2","denselayer.__call__.name-convn.2","denselayer.__call__.y.2","denselayer.__call__.features-x-y","denselayer.__call__.y-jnp-concatenate-features-axis-n","denselayer.__call__.return-y","densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
            },
            focusRef: {
              pytorch: "denselayer.bottleneck_channels-bottleneck_width-growth_rate",
              jax: "class-denselayer-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "denseblock1.layer2-6",
            label: "layer.2-6",
            type: "DenseLayer x5",
            kind: "concat",
            summary: "repeat concatenation",
            sourceRefs: {
              pytorch: ["denseblock.current_channels","denseblock.growth_rate.2","denseblock.dropout_rate-dropout_rate","denseblock.code.4","denseblock.layers-append-layer","denseblock.current_channels-current_channels-growth_rate","denseblock.self-layers-nn-modulelist-layers","denseblock.self-out_channels-current_channels","denseblock.forward.return-out"],
              jax: ["densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
            },
            focusRef: {
              pytorch: "denseblock.current_channels",
              jax: "densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["transition.in_channels","transition.self-conv-nn-convnd","transition.in_channels.2","transition.kernel_size-n","transition.forward.out-self-relu-out","transition.forward.out-self-conv-out","transition.forward.out-f-avg_poolnd-out-kernel_size-n-stride-n","transition.forward.return-out","class-densenet-nn-module","densenet.num_features-out_features","densenet.self-features-nn-sequential-blocks","densenet.self-norm-nn-batchnormnd-num_features","densenet.self-classifier-nn-linear-num_features-num_classes"],
          jax: ["class-transition-nn-module","transition.__call__.y-nn-batchnorm-use_running_average-not-train-name-norm-x","transition.__call__.y-nn-relu-y","transition.__call__.y-nn-conv","transition.__call__.self-out_channels","transition.__call__.n-n","transition.__call__.use_bias-false","transition.__call__.name-conv","transition.__call__.y","transition.__call__.y-nn-avg_pool-y-window_shape-n-n-strides-n-n-padding-valid","transition.__call__.return-y","densenet.__call__.is_last_block-block_index-len-self-block_config-n","densenet.__call__.if-not-is_last_block","densenet.__call__.num_features-int-num_features-self-compression","densenet.__call__.transition_name-f-transition-block_index-n","densenet.__call__.x-transition-num_features-name-transition_name-x-train-train"],
        },
        focusRef: {
          pytorch: "transition.in_channels",
          jax: "class-transition-nn-module",
        },
        includeChildRefs: false,
        children: [
          {
            id: "transition1.conv",
            label: "conv1x1",
            type: "CompressionConv",
            kind: "conv",
            badges: ["256->128"],
            sourceRefs: {
              pytorch: ["transition.kernel_size-n","transition.stride-n","transition.bias-false","transition.code.4","transition.def-forward-self-x","densenet.self-norm-nn-batchnormnd-num_features"],
              jax: ["transition.__call__.y-nn-conv","transition.__call__.self-out_channels","transition.__call__.n-n","transition.__call__.use_bias-false","transition.__call__.name-conv","transition.__call__.y","densenet.__call__.x-transition-num_features-name-transition_name-x-train-train"],
            },
            focusRef: {
              pytorch: "transition.kernel_size-n",
              jax: "transition.__call__.y-nn-conv",
            },
            includeChildRefs: false,
          },
          {
            id: "transition1.pool",
            label: "avgpool",
            type: "AvgPool2d",
            kind: "pool",
            badges: ["stride 2"],
            sourceRefs: {
              pytorch: ["transition.forward.out-f-avg_poolnd-out-kernel_size-n-stride-n"],
              jax: ["transition.__call__.y-nn-avg_pool-y-window_shape-n-n-strides-n-n-padding-valid"],
            },
            focusRef: {
              pytorch: "transition.forward.out-f-avg_poolnd-out-kernel_size-n-stride-n",
              jax: "transition.__call__.y-nn-avg_pool-y-window_shape-n-n-strides-n-n-padding-valid",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["denseblock.self","denseblock.layer_count","denseblock.in_channels","denseblock.growth_rate","denseblock.dropout_rate-n","denseblock.code","denseblock.super-__init__","denseblock.layers","denseblock.current_channels-in_channels","denseblock.for-_-in-range-layer_count","denseblock.layer-denselayer","denseblock.current_channels","denseblock.growth_rate.2","denseblock.dropout_rate-dropout_rate","denseblock.code.4","denseblock.layers-append-layer","denseblock.current_channels-current_channels-growth_rate","denseblock.self-layers-nn-modulelist-layers","denseblock.self-out_channels-current_channels","denseblock.def-forward-self-x","denseblock.forward.out-x","denseblock.forward.for-layer-in-self-layers","denseblock.forward.out-layer-out","denseblock.forward.return-out","class-transition-nn-module","densenet.dropout_rate-dropout_rate","densenet.code.8","densenet.blocks-append-dense_block","densenet.num_features-dense_block-out_channels","densenet.is_last_block-index-len-block_config-n","densenet.if-not-is_last_block","densenet.out_features-int-num_features-compression","densenet.transition-transition-num_features-out_features","densenet.forward.x-self-norm-x","densenet.forward.x-f-relu-x-inplace-true"],
          jax: ["densenet.__call__.num_features-self-num_init_features","densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
        },
        focusRef: {
          pytorch: "denseblock.self",
          jax: "densenet.__call__.num_features-self-num_init_features",
        },
        includeChildRefs: false,
        children: [
          {
            id: "denseblock2.layer1",
            label: "layer.1",
            type: "DenseLayer",
            kind: "group",
            summary: "BN-ReLU-conv x2",
            defaultExpanded: true,
            sourceRefs: {
              pytorch: ["denselayer.bottleneck_channels-bottleneck_width-growth_rate","denselayer.self-normn-nn-batchnormnd-in_channels","denselayer.self-relun-nn-relu-inplace-true","denselayer.in_channels.2","denselayer.self-relun-nn-relu-inplace-true.2","denselayer.bottleneck_channels.2","denselayer.forward.out-self-normn-x","denselayer.forward.out-self-relun-out","denselayer.forward.out-self-convn-out","denselayer.forward.out-self-normn-out","denselayer.forward.out-self-relun-out.2","denselayer.forward.out-self-convn-out.2","denselayer.forward.if-self-dropout_rate-n","denselayer.forward.features-x-out","denselayer.forward.out-torch-cat-features-dim-n","denselayer.forward.return-out","denseblock.growth_rate.2"],
              jax: ["class-denselayer-nn-module","denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-x","denselayer.__call__.y-nn-relu-y","denselayer.__call__.y-nn-conv","denselayer.__call__.bottleneck_channels","denselayer.__call__.n-n","denselayer.__call__.use_bias-false","denselayer.__call__.name-convn","denselayer.__call__.y","denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-y","denselayer.__call__.y-nn-relu-y.2","denselayer.__call__.y-nn-conv.2","denselayer.__call__.self-growth_rate","denselayer.__call__.n-n.2","denselayer.__call__.padding-same","denselayer.__call__.use_bias-false.2","denselayer.__call__.name-convn.2","denselayer.__call__.y.2","denselayer.__call__.features-x-y","denselayer.__call__.y-jnp-concatenate-features-axis-n","denselayer.__call__.return-y","densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
            },
            focusRef: {
              pytorch: "denselayer.bottleneck_channels-bottleneck_width-growth_rate",
              jax: "class-denselayer-nn-module",
            },
            includeChildRefs: false,
            children: [
              {
                id: "denseblock2.layer1.bottleneck",
                label: "bottleneck",
                type: "1x1 Conv",
                kind: "conv",
                badges: ["128->128"],
                sourceRefs: {
                  pytorch: ["denselayer.self-normn-nn-batchnormnd-in_channels","denselayer.self-relun-nn-relu-inplace-true","denselayer.in_channels.2","denselayer.bottleneck_channels","denselayer.kernel_size-n","denselayer.stride-n","denselayer.bias-false","denselayer.code.4","denselayer.self-normn-nn-batchnormnd-bottleneck_channels","denselayer.forward.out-self-relun-out","denselayer.forward.out-self-convn-out","denselayer.forward.out-self-normn-out"],
                  jax: ["denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-x","denselayer.__call__.y-nn-relu-y","denselayer.__call__.y-nn-conv","denselayer.__call__.bottleneck_channels","denselayer.__call__.n-n","denselayer.__call__.use_bias-false","denselayer.__call__.name-convn","denselayer.__call__.y"],
                },
                focusRef: {
                  pytorch: "denselayer.self-normn-nn-batchnormnd-in_channels",
                  jax: "denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-x",
                },
                includeChildRefs: false,
              },
              {
                id: "denseblock2.layer1.growth",
                label: "growth",
                type: "3x3 Conv",
                kind: "conv",
                badges: ["128->32"],
                sourceRefs: {
                  pytorch: ["denselayer.self-relun-nn-relu-inplace-true.2","denselayer.bottleneck_channels.2","denselayer.growth_rate.2","denselayer.kernel_size-n.2","denselayer.stride-n.2","denselayer.padding-n","denselayer.bias-false.2","denselayer.code.5","denselayer.self-dropout_rate-dropout_rate","denselayer.forward.out-self-relun-out.2","denselayer.forward.out-self-convn-out.2","denselayer.forward.if-self-dropout_rate-n"],
                  jax: ["denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-y","denselayer.__call__.y-nn-relu-y.2","denselayer.__call__.y-nn-conv.2","denselayer.__call__.self-growth_rate","denselayer.__call__.n-n.2","denselayer.__call__.padding-same","denselayer.__call__.use_bias-false.2","denselayer.__call__.name-convn.2","denselayer.__call__.y.2"],
                },
                focusRef: {
                  pytorch: "denselayer.self-relun-nn-relu-inplace-true.2",
                  jax: "denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-y",
                },
                includeChildRefs: false,
              },
              {
                id: "denseblock2.layer1.concat",
                label: "concat",
                type: "FeatureConcat",
                kind: "concat",
                badges: ["128+32"],
                sourceRefs: {
                  pytorch: ["denselayer.forward.features-x-out","denselayer.forward.out-torch-cat-features-dim-n","denselayer.forward.return-out"],
                  jax: ["denselayer.__call__.features-x-y","denselayer.__call__.y-jnp-concatenate-features-axis-n","denselayer.__call__.return-y"],
                },
                focusRef: {
                  pytorch: "denselayer.forward.features-x-out",
                  jax: "denselayer.__call__.features-x-y",
                },
                includeChildRefs: false,
              },
            ],
          },
          {
            id: "denseblock2.layer2-12",
            label: "layer.2-12",
            type: "DenseLayer x11",
            kind: "concat",
            summary: "same pattern",
            sourceRefs: {
              pytorch: ["denseblock.current_channels","denseblock.growth_rate.2","denseblock.dropout_rate-dropout_rate","denseblock.code.4","denseblock.layers-append-layer","denseblock.current_channels-current_channels-growth_rate","denseblock.self-layers-nn-modulelist-layers","denseblock.self-out_channels-current_channels","denseblock.forward.return-out"],
              jax: ["densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
            },
            focusRef: {
              pytorch: "denseblock.current_channels",
              jax: "densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["transition.in_channels","transition.self-conv-nn-convnd","transition.in_channels.2","transition.kernel_size-n","transition.forward.out-self-relu-out","transition.forward.out-self-conv-out","transition.forward.out-f-avg_poolnd-out-kernel_size-n-stride-n","transition.forward.return-out","class-densenet-nn-module","densenet.num_features-out_features","densenet.self-features-nn-sequential-blocks","densenet.self-norm-nn-batchnormnd-num_features","densenet.self-classifier-nn-linear-num_features-num_classes"],
          jax: ["class-transition-nn-module","transition.__call__.y-nn-batchnorm-use_running_average-not-train-name-norm-x","transition.__call__.y-nn-relu-y","transition.__call__.y-nn-conv","transition.__call__.self-out_channels","transition.__call__.n-n","transition.__call__.use_bias-false","transition.__call__.name-conv","transition.__call__.y","transition.__call__.y-nn-avg_pool-y-window_shape-n-n-strides-n-n-padding-valid","transition.__call__.return-y","densenet.__call__.is_last_block-block_index-len-self-block_config-n","densenet.__call__.if-not-is_last_block","densenet.__call__.num_features-int-num_features-self-compression","densenet.__call__.transition_name-f-transition-block_index-n","densenet.__call__.x-transition-num_features-name-transition_name-x-train-train"],
        },
        focusRef: {
          pytorch: "transition.in_channels",
          jax: "class-transition-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: "denseblock3",
        label: "denseblock3",
        type: "DenseBlock",
        kind: "group",
        summary: "24 dense layers",
        badges: ["256->1024", "14x14"],
        sourceRefs: {
          pytorch: ["denseblock.self","denseblock.layer_count","denseblock.in_channels","denseblock.growth_rate","denseblock.dropout_rate-n","denseblock.code","denseblock.super-__init__","denseblock.layers","denseblock.current_channels-in_channels","denseblock.for-_-in-range-layer_count","denseblock.layer-denselayer","denseblock.current_channels","denseblock.growth_rate.2","denseblock.dropout_rate-dropout_rate","denseblock.code.4","denseblock.layers-append-layer","denseblock.current_channels-current_channels-growth_rate","denseblock.self-layers-nn-modulelist-layers","denseblock.self-out_channels-current_channels","denseblock.def-forward-self-x","denseblock.forward.out-x","denseblock.forward.for-layer-in-self-layers","denseblock.forward.out-layer-out","denseblock.forward.return-out","class-transition-nn-module","densenet.dropout_rate-dropout_rate","densenet.code.8","densenet.blocks-append-dense_block","densenet.num_features-dense_block-out_channels","densenet.is_last_block-index-len-block_config-n","densenet.if-not-is_last_block","densenet.out_features-int-num_features-compression","densenet.transition-transition-num_features-out_features","densenet.forward.x-self-norm-x","densenet.forward.x-f-relu-x-inplace-true"],
          jax: ["densenet.__call__.num_features-self-num_init_features","densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
        },
        focusRef: {
          pytorch: "denseblock.self",
          jax: "densenet.__call__.num_features-self-num_init_features",
        },
        includeChildRefs: false,
        children: [
          {
            id: "denseblock3.layer1",
            label: "layer.1",
            type: "DenseLayer",
            kind: "concat",
            summary: "append growth features",
            badges: ["256+32"],
            sourceRefs: {
              pytorch: ["denselayer.forward.out-self-normn-x","denselayer.forward.out-self-relun-out","denselayer.forward.out-self-normn-out","denselayer.forward.if-self-dropout_rate-n","denselayer.forward.features-x-out","denselayer.forward.out-torch-cat-features-dim-n","denselayer.forward.return-out"],
              jax: ["class-denselayer-nn-module","denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-x","denselayer.__call__.y-nn-relu-y","denselayer.__call__.y-nn-conv","denselayer.__call__.bottleneck_channels","denselayer.__call__.n-n","denselayer.__call__.use_bias-false","denselayer.__call__.name-convn","denselayer.__call__.y","denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-y","denselayer.__call__.y-nn-relu-y.2","denselayer.__call__.y-nn-conv.2","denselayer.__call__.self-growth_rate","denselayer.__call__.n-n.2","denselayer.__call__.padding-same","denselayer.__call__.use_bias-false.2","denselayer.__call__.name-convn.2","denselayer.__call__.y.2","denselayer.__call__.features-x-y","denselayer.__call__.y-jnp-concatenate-features-axis-n","denselayer.__call__.return-y","densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
            },
            focusRef: {
              pytorch: "denselayer.forward.out-self-normn-x",
              jax: "class-denselayer-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "denseblock3.layer2-24",
            label: "layer.2-24",
            type: "DenseLayer x23",
            kind: "concat",
            summary: "repeat concatenation",
            sourceRefs: {
              pytorch: ["denseblock.current_channels","denseblock.growth_rate.2","denseblock.self-out_channels-current_channels","denseblock.forward.return-out"],
              jax: ["densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
            },
            focusRef: {
              pytorch: "denseblock.current_channels",
              jax: "densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["transition.in_channels","transition.self-conv-nn-convnd","transition.in_channels.2","transition.kernel_size-n","transition.forward.out-self-relu-out","transition.forward.out-self-conv-out","transition.forward.out-f-avg_poolnd-out-kernel_size-n-stride-n","transition.forward.return-out","class-densenet-nn-module","densenet.num_features-out_features","densenet.self-features-nn-sequential-blocks","densenet.self-norm-nn-batchnormnd-num_features","densenet.self-classifier-nn-linear-num_features-num_classes"],
          jax: ["class-transition-nn-module","transition.__call__.y-nn-batchnorm-use_running_average-not-train-name-norm-x","transition.__call__.y-nn-relu-y","transition.__call__.y-nn-conv","transition.__call__.self-out_channels","transition.__call__.n-n","transition.__call__.use_bias-false","transition.__call__.name-conv","transition.__call__.y","transition.__call__.y-nn-avg_pool-y-window_shape-n-n-strides-n-n-padding-valid","transition.__call__.return-y","densenet.__call__.is_last_block-block_index-len-self-block_config-n","densenet.__call__.if-not-is_last_block","densenet.__call__.num_features-int-num_features-self-compression","densenet.__call__.transition_name-f-transition-block_index-n","densenet.__call__.x-transition-num_features-name-transition_name-x-train-train"],
        },
        focusRef: {
          pytorch: "transition.in_channels",
          jax: "class-transition-nn-module",
        },
        includeChildRefs: false,
      },
      {
        id: "denseblock4",
        label: "denseblock4",
        type: "DenseBlock",
        kind: "group",
        summary: "16 dense layers",
        badges: ["512->1024", "7x7"],
        sourceRefs: {
          pytorch: ["denseblock.self","denseblock.layer_count","denseblock.in_channels","denseblock.growth_rate","denseblock.dropout_rate-n","denseblock.code","denseblock.super-__init__","denseblock.layers","denseblock.current_channels-in_channels","denseblock.for-_-in-range-layer_count","denseblock.layer-denselayer","denseblock.current_channels","denseblock.growth_rate.2","denseblock.dropout_rate-dropout_rate","denseblock.code.4","denseblock.layers-append-layer","denseblock.current_channels-current_channels-growth_rate","denseblock.self-layers-nn-modulelist-layers","denseblock.self-out_channels-current_channels","denseblock.def-forward-self-x","denseblock.forward.out-x","denseblock.forward.for-layer-in-self-layers","denseblock.forward.out-layer-out","denseblock.forward.return-out","class-transition-nn-module","densenet.dropout_rate-dropout_rate","densenet.code.8","densenet.blocks-append-dense_block","densenet.num_features-dense_block-out_channels","densenet.is_last_block-index-len-block_config-n","densenet.if-not-is_last_block","densenet.out_features-int-num_features-compression","densenet.transition-transition-num_features-out_features","densenet.forward.x-self-norm-x","densenet.forward.x-f-relu-x-inplace-true"],
          jax: ["densenet.__call__.num_features-self-num_init_features","densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config","densenet.__call__.for-layer_index-in-range-layer_count","densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n","densenet.__call__.x-denselayer","densenet.__call__.self-growth_rate","densenet.__call__.dropout_rate-self-dropout_rate","densenet.__call__.name-layer_name","densenet.__call__.x-train-train","densenet.__call__.num_features-num_features-self-growth_rate"],
        },
        focusRef: {
          pytorch: "denseblock.self",
          jax: "densenet.__call__.num_features-self-num_init_features",
        },
        includeChildRefs: false,
      },
      {
        id: "head",
        label: "head",
        type: "Norm-Pool-FC",
        kind: "group",
        summary: "global average pool",
        sourceRefs: {
          pytorch: ["densenet.forward.x-self-stem-x","densenet.forward.x-torch-flatten-x-n","densenet.forward.logits-self-classifier-x","densenet.forward.return-logits"],
          jax: ["densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-norm-x","densenet.__call__.x-nn-relu-x.2","densenet.__call__.x-jnp-mean-x-axis-n-n","densenet.__call__.logits-nn-dense-self-num_classes-name-classifier-x","densenet.__call__.return-logits"],
        },
        focusRef: {
          pytorch: "densenet.forward.x-self-stem-x",
          jax: "densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-norm-x",
        },
        includeChildRefs: false,
        children: [
          {
            id: "head.norm",
            label: "norm",
            type: "BatchNorm2d",
            kind: "norm",
            badges: ["1024"],
            sourceRefs: {
              pytorch: ["densenet.self-norm-nn-batchnormnd-num_features","densenet.forward.x-self-norm-x"],
              jax: ["densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-norm-x"],
            },
            focusRef: {
              pytorch: "densenet.self-norm-nn-batchnormnd-num_features",
              jax: "densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-norm-x",
            },
            includeChildRefs: false,
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            sourceRefs: {
              pytorch: ["densenet.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n"],
              jax: ["densenet.__call__.x-jnp-mean-x-axis-n-n"],
            },
            focusRef: {
              pytorch: "densenet.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n",
              jax: "densenet.__call__.x-jnp-mean-x-axis-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "head.classifier",
            label: "classifier",
            type: "Linear",
            kind: "linear",
            badges: ["1024->1000"],
            sourceRefs: {
              pytorch: ["densenet.self-classifier-nn-linear-num_features-num_classes","densenet.forward.logits-self-classifier-x"],
              jax: ["densenet.__call__.logits-nn-dense-self-num_classes-name-classifier-x"],
            },
            focusRef: {
              pytorch: "densenet.self-classifier-nn-linear-num_features-num_classes",
              jax: "densenet.__call__.logits-nn-dense-self-num_classes-name-classifier-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  mobilenetv2: {
    stats: "17 inverted residual blocks · depthwise separable convs · linear bottlenecks",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        sourceRefs: {
          pytorch: ["mobilenetvn.forward.x-self-stem-x","train_images-torch-zeros-n-n-n-n","logits-model-train_images"],
          jax: ["mobilenetvn.__call__.x","train_images-jnp-zeros-n-n-n-n","train_step.loss_fn.logits-updated_variables-model-apply","train_step.loss_fn.inputs"],
        },
        focusRef: {
          pytorch: "mobilenetvn.forward.x-self-stem-x",
          jax: "mobilenetvn.__call__.x",
        },
        includeChildRefs: false,
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-ReLU6",
        kind: "group",
        summary: "3x3 stride 2",
        badges: ["3->32", "112x112"],
        sourceRefs: {
          pytorch: ["mobilenetvn.self-stem-nn-sequential","mobilenetvn.nn-batchnormnd-n","mobilenetvn.code.7"],
          jax: ["mobilenetvn.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x"],
        },
        focusRef: {
          pytorch: "mobilenetvn.self-stem-nn-sequential",
          jax: "mobilenetvn.__call__.x-nn-conv",
        },
        includeChildRefs: true,
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->32", "k=3", "s=2"],
            sourceRefs: {
              pytorch: ["mobilenetvn.nn-convnd-n-n-kernel_size-n-stride-n-padding-n-bias-false","mobilenetvn.forward.x-self-stem-x"],
              jax: ["mobilenetvn.__call__.x-nn-conv","mobilenetvn.__call__.n","mobilenetvn.__call__.n-n","mobilenetvn.__call__.strides-n-n","mobilenetvn.__call__.padding-same","mobilenetvn.__call__.use_bias-false","mobilenetvn.__call__.name-stem_conv","mobilenetvn.__call__.x"],
            },
            focusRef: {
              pytorch: "mobilenetvn.nn-convnd-n-n-kernel_size-n-stride-n-padding-n-bias-false",
              jax: "mobilenetvn.__call__.x-nn-conv",
            },
            includeChildRefs: false,
          },
          {
            id: "stem.relu6",
            label: "relu6",
            type: "ClippedReLU",
            kind: "activation",
            sourceRefs: {
              pytorch: ["mobilenetvn.nn-relun-inplace-true","mobilenetvn.forward.x-self-stem-x"],
              jax: ["mobilenetvn.__call__.x-nn-relun-x"],
            },
            focusRef: {
              pytorch: "mobilenetvn.nn-relun-inplace-true",
              jax: "mobilenetvn.__call__.x-nn-relun-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["invertedresidual.layers","invertedresidual.layers-extend","invertedresidual.code.6","invertedresidual.code.7","invertedresidual.code.8","invertedresidual.layers-extend.2","invertedresidual.code.9","invertedresidual.code.11","invertedresidual.code.12","invertedresidual.self-block-nn-sequential-layers","mobilenetvn.block-invertedresidual","mobilenetvn.in_channels","mobilenetvn.out_channels","mobilenetvn.block_stride","mobilenetvn.expand_ratio","mobilenetvn.code.10"],
          jax: ["invertedresidual.__call__.in_channels-x-shape-n","invertedresidual.__call__.y-x","mobilenetvn.__call__.x-invertedresidual","mobilenetvn.__call__.out_channels","mobilenetvn.__call__.block_stride","mobilenetvn.__call__.expand_ratio","mobilenetvn.__call__.name-block_name","mobilenetvn.__call__.x-train-train"],
        },
        focusRef: {
          pytorch: "invertedresidual.hidden_channels-in_channels-expand_ratio",
          jax: "invertedresidual.__call__.in_channels-x-shape-n",
        },
        includeChildRefs: true,
        children: [
          {
            id: "inverted_residual.expand",
            label: "expand",
            type: "PointwiseConv",
            kind: "conv",
            badges: ["1x1", "t=6"],
            sourceRefs: {
              pytorch: ["invertedresidual.hidden_channels-in_channels-expand_ratio","invertedresidual.if-expand_ratio-n","invertedresidual.nn-convnd-in_channels-hidden_channels-kernel_size-n-bias-false","invertedresidual.nn-batchnormnd-hidden_channels","invertedresidual.nn-relun-inplace-true"],
              jax: ["invertedresidual.__call__.hidden_channels-in_channels-self-expand_ratio","invertedresidual.__call__.if-self-expand_ratio-n","invertedresidual.__call__.y-nn-conv","invertedresidual.__call__.hidden_channels","invertedresidual.__call__.n-n","invertedresidual.__call__.use_bias-false","invertedresidual.__call__.name-expand_conv","invertedresidual.__call__.y","invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y","invertedresidual.__call__.y-nn-relun-y"],
            },
            focusRef: {
              pytorch: "invertedresidual.hidden_channels-in_channels-expand_ratio",
              jax: "invertedresidual.__call__.hidden_channels-in_channels-self-expand_ratio",
            },
            includeChildRefs: false,
          },
          {
            id: "inverted_residual.depthwise",
            label: "depthwise",
            type: "DepthwiseConv",
            kind: "conv",
            badges: ["3x3", "groups=hidden"],
            sourceRefs: {
              pytorch: ["invertedresidual.nn-convnd","invertedresidual.hidden_channels","invertedresidual.hidden_channels.2","invertedresidual.kernel_size-n","invertedresidual.stride-stride","invertedresidual.padding-n","invertedresidual.groups-hidden_channels","invertedresidual.bias-false","invertedresidual.code.10","invertedresidual.nn-batchnormnd-hidden_channels.2","invertedresidual.nn-relun-inplace-true.2"],
              jax: ["invertedresidual.__call__.y-nn-conv.2","invertedresidual.__call__.hidden_channels.2","invertedresidual.__call__.n-n.2","invertedresidual.__call__.strides-self-stride-self-stride","invertedresidual.__call__.padding-same","invertedresidual.__call__.feature_group_count-hidden_channels","invertedresidual.__call__.use_bias-false.2","invertedresidual.__call__.name-depthwise_conv","invertedresidual.__call__.y.2","invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y","invertedresidual.__call__.y-nn-relun-y.2"],
            },
            focusRef: {
              pytorch: "invertedresidual.nn-convnd",
              jax: "invertedresidual.__call__.y-nn-conv.2",
            },
            includeChildRefs: false,
          },
          {
            id: "inverted_residual.project",
            label: "linear bottleneck",
            type: "PointwiseProjection",
            kind: "conv",
            badges: ["1x1", "no ReLU"],
            sourceRefs: {
              pytorch: ["invertedresidual.nn-convnd-hidden_channels-out_channels-kernel_size-n-bias-false","invertedresidual.nn-batchnormnd-out_channels","invertedresidual.forward.out-self-block-x"],
              jax: ["invertedresidual.__call__.y-nn-conv.3","invertedresidual.__call__.self-out_channels","invertedresidual.__call__.n-n.3","invertedresidual.__call__.use_bias-false.3","invertedresidual.__call__.name-project_conv","invertedresidual.__call__.y.3","invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y"],
            },
            focusRef: {
              pytorch: "invertedresidual.nn-convnd-hidden_channels-out_channels-kernel_size-n-bias-false",
              jax: "invertedresidual.__call__.y-nn-conv.3",
            },
            includeChildRefs: false,
          },
          {
            id: "inverted_residual.shortcut",
            label: "shortcut",
            type: "ResidualAdd",
            kind: "residual",
            badges: ["same shape"],
            sourceRefs: {
              pytorch: ["invertedresidual.self-use_residual-stride-n-and-in_channels-out_channels","invertedresidual.forward.if-self-use_residual","invertedresidual.forward.out-out-x"],
              jax: ["invertedresidual.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels","invertedresidual.__call__.if-use_residual","invertedresidual.__call__.y-y-x"],
            },
            focusRef: {
              pytorch: "invertedresidual.self-use_residual-stride-n-and-in_channels-out_channels",
              jax: "invertedresidual.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["mobilenetvn.n-n-n-n","mobilenetvn.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.for-index-in-range-repeats","mobilenetvn.block_stride-stride-if-index-n-else-n","mobilenetvn.block-invertedresidual","mobilenetvn.in_channels","mobilenetvn.out_channels","mobilenetvn.block_stride","mobilenetvn.expand_ratio","mobilenetvn.code.10","mobilenetvn.blocks-append-block","mobilenetvn.in_channels-out_channels","mobilenetvn.forward.x-self-blocks-x"],
          jax: ["mobilenetvn.__call__.n-n-n-n","mobilenetvn.__call__.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.__call__.for-repeat_index-in-range-repeats","mobilenetvn.__call__.block_stride-stride-if-repeat_index-n-else-n","mobilenetvn.__call__.block_name-f-blocks-block_index","mobilenetvn.__call__.x-invertedresidual","mobilenetvn.__call__.out_channels","mobilenetvn.__call__.block_stride","mobilenetvn.__call__.expand_ratio","mobilenetvn.__call__.name-block_name","mobilenetvn.__call__.x-train-train"],
        },
        focusRef: {
          pytorch: "mobilenetvn.n-n-n-n",
          jax: "mobilenetvn.__call__.n-n-n-n",
        },
        includeChildRefs: false,
      },
      {
        id: "stage2",
        label: "stage2",
        type: "InvertedResidual x2",
        kind: "group",
        summary: "downsample then residual",
        badges: ["16->24", "56x56"],
        sourceRefs: {
          pytorch: ["mobilenetvn.n-n-n-n.2","mobilenetvn.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.for-index-in-range-repeats","mobilenetvn.block_stride-stride-if-index-n-else-n","mobilenetvn.block-invertedresidual","mobilenetvn.in_channels","mobilenetvn.out_channels","mobilenetvn.block_stride","mobilenetvn.expand_ratio","mobilenetvn.code.10","mobilenetvn.blocks-append-block","mobilenetvn.in_channels-out_channels","mobilenetvn.forward.x-self-blocks-x"],
          jax: ["mobilenetvn.__call__.n-n-n-n.2","mobilenetvn.__call__.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.__call__.for-repeat_index-in-range-repeats","mobilenetvn.__call__.block_stride-stride-if-repeat_index-n-else-n","mobilenetvn.__call__.block_name-f-blocks-block_index","mobilenetvn.__call__.x-invertedresidual","mobilenetvn.__call__.out_channels","mobilenetvn.__call__.block_stride","mobilenetvn.__call__.expand_ratio","mobilenetvn.__call__.name-block_name","mobilenetvn.__call__.x-train-train"],
        },
        focusRef: {
          pytorch: "mobilenetvn.n-n-n-n.2",
          jax: "mobilenetvn.__call__.n-n-n-n.2",
        },
        includeChildRefs: false,
      },
      {
        id: "stage3",
        label: "stage3",
        type: "InvertedResidual x3",
        kind: "group",
        summary: "32-channel bottlenecks",
        badges: ["24->32", "28x28"],
        sourceRefs: {
          pytorch: ["mobilenetvn.n-n-n-n.3","mobilenetvn.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.for-index-in-range-repeats","mobilenetvn.block_stride-stride-if-index-n-else-n","mobilenetvn.block-invertedresidual","mobilenetvn.in_channels","mobilenetvn.out_channels","mobilenetvn.block_stride","mobilenetvn.expand_ratio","mobilenetvn.code.10","mobilenetvn.blocks-append-block","mobilenetvn.in_channels-out_channels","mobilenetvn.forward.x-self-blocks-x"],
          jax: ["mobilenetvn.__call__.n-n-n-n.3","mobilenetvn.__call__.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.__call__.for-repeat_index-in-range-repeats","mobilenetvn.__call__.block_stride-stride-if-repeat_index-n-else-n","mobilenetvn.__call__.block_name-f-blocks-block_index","mobilenetvn.__call__.x-invertedresidual","mobilenetvn.__call__.out_channels","mobilenetvn.__call__.block_stride","mobilenetvn.__call__.expand_ratio","mobilenetvn.__call__.name-block_name","mobilenetvn.__call__.x-train-train"],
        },
        focusRef: {
          pytorch: "mobilenetvn.n-n-n-n.3",
          jax: "mobilenetvn.__call__.n-n-n-n.3",
        },
        includeChildRefs: false,
      },
      {
        id: "stage4",
        label: "stage4",
        type: "InvertedResidual x4",
        kind: "group",
        summary: "64-channel bottlenecks",
        badges: ["32->64", "14x14"],
        sourceRefs: {
          pytorch: ["mobilenetvn.n-n-n-n.4","mobilenetvn.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.for-index-in-range-repeats","mobilenetvn.block_stride-stride-if-index-n-else-n","mobilenetvn.block-invertedresidual","mobilenetvn.in_channels","mobilenetvn.out_channels","mobilenetvn.block_stride","mobilenetvn.expand_ratio","mobilenetvn.code.10","mobilenetvn.blocks-append-block","mobilenetvn.in_channels-out_channels","mobilenetvn.forward.x-self-blocks-x"],
          jax: ["mobilenetvn.__call__.n-n-n-n.4","mobilenetvn.__call__.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.__call__.for-repeat_index-in-range-repeats","mobilenetvn.__call__.block_stride-stride-if-repeat_index-n-else-n","mobilenetvn.__call__.block_name-f-blocks-block_index","mobilenetvn.__call__.x-invertedresidual","mobilenetvn.__call__.out_channels","mobilenetvn.__call__.block_stride","mobilenetvn.__call__.expand_ratio","mobilenetvn.__call__.name-block_name","mobilenetvn.__call__.x-train-train"],
        },
        focusRef: {
          pytorch: "mobilenetvn.n-n-n-n.4",
          jax: "mobilenetvn.__call__.n-n-n-n.4",
        },
        includeChildRefs: false,
      },
      {
        id: "stage5_7",
        label: "stage5-7",
        type: "InvertedResidual x7",
        kind: "group",
        summary: "96/160/320 bottlenecks",
        badges: ["7x7 final"],
        sourceRefs: {
          pytorch: ["mobilenetvn.n-n-n-n.5","mobilenetvn.n-n-n-n.6","mobilenetvn.n-n-n-n.7","mobilenetvn.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.for-index-in-range-repeats","mobilenetvn.block_stride-stride-if-index-n-else-n","mobilenetvn.block-invertedresidual","mobilenetvn.in_channels","mobilenetvn.out_channels","mobilenetvn.block_stride","mobilenetvn.expand_ratio","mobilenetvn.code.10","mobilenetvn.blocks-append-block","mobilenetvn.in_channels-out_channels","mobilenetvn.forward.x-self-blocks-x"],
          jax: ["mobilenetvn.__call__.n-n-n-n.5","mobilenetvn.__call__.n-n-n-n.6","mobilenetvn.__call__.n-n-n-n.7","mobilenetvn.__call__.for-expand_ratio-out_channels-repeats-stride-in-settings","mobilenetvn.__call__.for-repeat_index-in-range-repeats","mobilenetvn.__call__.block_stride-stride-if-repeat_index-n-else-n","mobilenetvn.__call__.block_name-f-blocks-block_index","mobilenetvn.__call__.x-invertedresidual","mobilenetvn.__call__.out_channels","mobilenetvn.__call__.block_stride","mobilenetvn.__call__.expand_ratio","mobilenetvn.__call__.name-block_name","mobilenetvn.__call__.x-train-train"],
        },
        focusRef: {
          pytorch: "mobilenetvn.n-n-n-n.5",
          jax: "mobilenetvn.__call__.n-n-n-n.5",
        },
        includeChildRefs: false,
      },
      {
        id: "head",
        label: "head",
        type: "Conv-Pool-Dropout-FC",
        kind: "group",
        summary: "1280 expansion",
        defaultExpanded: true,
        sourceRefs: {
          pytorch: ["mobilenetvn.self-head-nn-sequential","mobilenetvn.nn-convnd-in_channels-n-kernel_size-n-bias-false","mobilenetvn.nn-batchnormnd-n.2","mobilenetvn.nn-relun-inplace-true.2","mobilenetvn.code.13","mobilenetvn.self-dropout-nn-dropout-p-dropout","mobilenetvn.self-classifier-nn-linear-n-num_classes","mobilenetvn.forward.x-self-head-x","mobilenetvn.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n","mobilenetvn.forward.x-torch-flatten-x-start_dim-n","mobilenetvn.forward.x-self-dropout-x","mobilenetvn.forward.logits-self-classifier-x"],
          jax: ["mobilenetvn.__call__.x-nn-conv.2","mobilenetvn.__call__.n.2","mobilenetvn.__call__.n-n.2","mobilenetvn.__call__.use_bias-false.2","mobilenetvn.__call__.name-head_conv","mobilenetvn.__call__.x.2","mobilenetvn.__call__.x-nn-batchnorm-use_running_average-not-train-name-head_bn-x","mobilenetvn.__call__.x-nn-relun-x.2","mobilenetvn.__call__.x-jnp-mean-x-axis-n-n","mobilenetvn.__call__.x-nn-dropout-rate-self-dropout_rate-deterministic-not-train-name-dropout","mobilenetvn.__call__.logits-nn-dense-self-num_classes-name-classifier-x"],
        },
        focusRef: {
          pytorch: "mobilenetvn.self-head-nn-sequential",
          jax: "mobilenetvn.__call__.x-nn-conv.2",
        },
        includeChildRefs: false,
        children: [
          {
            id: "head.expand",
            label: "expand",
            type: "PointwiseConv",
            kind: "conv",
            badges: ["320->1280"],
            sourceRefs: {
              pytorch: ["mobilenetvn.nn-convnd-in_channels-n-kernel_size-n-bias-false","mobilenetvn.forward.x-self-head-x"],
              jax: ["mobilenetvn.__call__.x-nn-conv.2","mobilenetvn.__call__.n.2","mobilenetvn.__call__.n-n.2","mobilenetvn.__call__.use_bias-false.2","mobilenetvn.__call__.name-head_conv","mobilenetvn.__call__.x.2"],
            },
            focusRef: {
              pytorch: "mobilenetvn.nn-convnd-in_channels-n-kernel_size-n-bias-false",
              jax: "mobilenetvn.__call__.x-nn-conv.2",
            },
            includeChildRefs: false,
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "GlobalAveragePool",
            kind: "pool",
            badges: ["1x1"],
            sourceRefs: {
              pytorch: ["mobilenetvn.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n","mobilenetvn.forward.x-torch-flatten-x-start_dim-n"],
              jax: ["mobilenetvn.__call__.x-jnp-mean-x-axis-n-n"],
            },
            focusRef: {
              pytorch: "mobilenetvn.forward.x-f-adaptive_avg_poolnd-x-output_size-n-n",
              jax: "mobilenetvn.__call__.x-jnp-mean-x-axis-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "head.dropout",
            label: "dropout",
            type: "Dropout",
            kind: "activation",
            badges: ["p=0.2"],
            sourceRefs: {
              pytorch: ["mobilenetvn.dropout-n","mobilenetvn.self-dropout-nn-dropout-p-dropout","mobilenetvn.forward.x-self-dropout-x"],
              jax: ["mobilenetvn.dropout_rate-float-n","mobilenetvn.__call__.x-nn-dropout-rate-self-dropout_rate-deterministic-not-train-name-dropout"],
            },
            focusRef: {
              pytorch: "mobilenetvn.dropout-n",
              jax: "mobilenetvn.dropout_rate-float-n",
            },
            includeChildRefs: false,
          },
          {
            id: "head.classifier",
            label: "classifier",
            type: "Linear",
            kind: "linear",
            badges: ["1280->1000"],
            sourceRefs: {
              pytorch: ["mobilenetvn.self-classifier-nn-linear-n-num_classes","mobilenetvn.forward.logits-self-classifier-x"],
              jax: ["mobilenetvn.__call__.logits-nn-dense-self-num_classes-name-classifier-x"],
            },
            focusRef: {
              pytorch: "mobilenetvn.self-classifier-nn-linear-n-num_classes",
              jax: "mobilenetvn.__call__.logits-nn-dense-self-num_classes-name-classifier-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
  efficientnet: {
    stats: "MBConv stages · depthwise convs · squeeze-excitation · compound scaling",
    nodes: [
      {
        id: "input",
        label: "input",
        type: "Image",
        kind: "input",
        badges: ["3 x 224 x 224"],
        sourceRefs: {
          pytorch: ["efficientnet.forward.x-self-blocks-x"],
          jax: ["efficientnet.__call__.strides-n-n","efficientnet.__call__.padding-same","efficientnet.__call__.use_bias-false","efficientnet.__call__.name-stem_conv","efficientnet.__call__.x","efficientnet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x","efficientnet.__call__.x-nn-silu-x"],
        },
        focusRef: {
          pytorch: "efficientnet.forward.x-self-blocks-x",
          jax: "efficientnet.__call__.strides-n-n",
        },
        includeChildRefs: false,
      },
      {
        id: "stem",
        label: "stem",
        type: "Conv-BN-SiLU",
        kind: "group",
        summary: "3x3 stride 2",
        badges: ["3->32", "112x112"],
        sourceRefs: {
          pytorch: ["efficientnet.n","efficientnet.n.2","efficientnet.blocks"],
          jax: [],
        },
        focusRef: {
          pytorch: "efficientnet.n",
          jax: "efficientnet.__call__.x-nn-conv",
        },
        includeChildRefs: true,
        children: [
          {
            id: "stem.conv",
            label: "conv",
            type: "Conv2d",
            kind: "conv",
            badges: ["3->32", "k=3", "s=2"],
            sourceRefs: {
              pytorch: ["efficientnet.kernel_size-n","efficientnet.stride-n","efficientnet.padding-n","efficientnet.bias-false","efficientnet.code.7","efficientnet.nn-batchnormnd-n","efficientnet.nn-silu-inplace-true","efficientnet.code.8","efficientnet.forward.x-self-blocks-x"],
              jax: ["efficientnet.__call__.x-nn-conv","efficientnet.__call__.n","efficientnet.__call__.n-n","efficientnet.__call__.strides-n-n","efficientnet.__call__.padding-same","efficientnet.__call__.use_bias-false","efficientnet.__call__.name-stem_conv","efficientnet.__call__.x","efficientnet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x"],
            },
            focusRef: {
              pytorch: "efficientnet.kernel_size-n",
              jax: "efficientnet.__call__.x-nn-conv",
            },
            includeChildRefs: false,
          },
          {
            id: "stem.silu",
            label: "silu",
            type: "Swish",
            kind: "activation",
            sourceRefs: {
              pytorch: ["efficientnet.forward.x-self-blocks-x"],
              jax: ["efficientnet.__call__.x-nn-silu-x"],
            },
            focusRef: {
              pytorch: "efficientnet.forward.x-self-blocks-x",
              jax: "efficientnet.__call__.x-nn-silu-x",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["efficientnet.n-n-n-n-n.2","efficientnet.n-n-n-n-n.3","efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.for-index-in-range-repeats","efficientnet.block_stride-stride-if-index-n-else-n","efficientnet.block-mbconv","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.block_stride","efficientnet.kernel_size","efficientnet.code.11","efficientnet.blocks-append-block","efficientnet.in_channels-out_channels","efficientnet.self-blocks-nn-sequential-blocks","efficientnet.forward.x-self-head-x"],
          jax: ["efficientnet.__call__.n-n-n-n-n","efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.__call__.for-repeat_index-in-range-repeats","efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n","efficientnet.__call__.block_name-f-blocks-block_index","efficientnet.__call__.x-mbconv","efficientnet.__call__.out_channels","efficientnet.__call__.expand_ratio","efficientnet.__call__.block_stride","efficientnet.__call__.kernel_size","efficientnet.__call__.name-block_name","efficientnet.__call__.x-train-train","efficientnet.__call__.block_index-block_index-n"],
        },
        focusRef: {
          pytorch: "efficientnet.n-n-n-n-n.2",
          jax: "efficientnet.__call__.n-n-n-n-n",
        },
        includeChildRefs: false,
        children: [
          {
            id: "stage1.mbconv0",
            label: "mbconv.0",
            type: "MBConv",
            kind: "group",
            summary: "depthwise + SE",
            sourceRefs: {
              pytorch: ["mbconv.self","mbconv.squeeze_channels-max-n-in_channels-n","mbconv.padding-kernel_size-n","mbconv.self-use_residual-stride-n-and-in_channels-out_channels","mbconv.if-expand_ratio-n","mbconv.expanded_channels","mbconv.self-project-nn-sequential","mbconv.nn-convnd-expanded_channels-out_channels-kernel_size-n-bias-false","mbconv.nn-batchnormnd-out_channels","mbconv.forward.identity-x","mbconv.forward.if-self-expand-is-not-none","mbconv.forward.out-self-project-out","efficientnet.n-n-n-n-n.2","efficientnet.n-n-n-n-n.3","efficientnet.expand_ratio","efficientnet.forward.x-self-head-x"],
              jax: ["class-mbconv-nn-module","mbconv.out_channels-int","mbconv.expand_ratio-int","mbconv.stride-int","mbconv.kernel_size-int","mbconv.se_ratio-float-n","mbconv.nn-compact","mbconv.def-__call__-self-x-train-false","mbconv.__call__.in_channels-x-shape-n","mbconv.__call__.expanded_channels-in_channels-self-expand_ratio","mbconv.__call__.squeeze_channels-max-n-int-expanded_channels-self-se_ratio","mbconv.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels","mbconv.__call__.y-x","mbconv.__call__.if-self-expand_ratio-n","mbconv.__call__.y-nn-conv","mbconv.__call__.expanded_channels","mbconv.__call__.n-n","mbconv.__call__.use_bias-false","mbconv.__call__.name-expand_conv","mbconv.__call__.y","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y","mbconv.__call__.y-nn-silu-y","mbconv.__call__.y-nn-conv.2","mbconv.__call__.expanded_channels.2","mbconv.__call__.self-kernel_size-self-kernel_size","mbconv.__call__.strides-self-stride-self-stride","mbconv.__call__.padding-same","mbconv.__call__.feature_group_count-expanded_channels","mbconv.__call__.use_bias-false.2","mbconv.__call__.name-depthwise_conv","mbconv.__call__.y.2","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y","mbconv.__call__.y-nn-silu-y.2","mbconv.__call__.y-squeezeexcite-squeeze_channels-name-se-y","mbconv.__call__.y-nn-conv.3","mbconv.__call__.self-out_channels","mbconv.__call__.n-n.2","mbconv.__call__.use_bias-false.3","mbconv.__call__.name-project_conv","mbconv.__call__.y.3","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y","mbconv.__call__.if-use_residual","mbconv.__call__.y-y-x","mbconv.__call__.return-y"],
            },
            focusRef: {
              pytorch: "mbconv.self",
              jax: "class-mbconv-nn-module",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["efficientnet.n-n-n-n-n.2","efficientnet.n-n-n-n-n.4","efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.for-index-in-range-repeats","efficientnet.block_stride-stride-if-index-n-else-n","efficientnet.block-mbconv","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.block_stride","efficientnet.kernel_size","efficientnet.code.11","efficientnet.blocks-append-block","efficientnet.in_channels-out_channels","efficientnet.self-blocks-nn-sequential-blocks","efficientnet.forward.x-self-head-x"],
          jax: ["efficientnet.__call__.n-n-n-n-n.2","efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.__call__.for-repeat_index-in-range-repeats","efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n","efficientnet.__call__.block_name-f-blocks-block_index","efficientnet.__call__.x-mbconv","efficientnet.__call__.out_channels","efficientnet.__call__.expand_ratio","efficientnet.__call__.block_stride","efficientnet.__call__.kernel_size","efficientnet.__call__.name-block_name","efficientnet.__call__.x-train-train","efficientnet.__call__.block_index-block_index-n"],
        },
        focusRef: {
          pytorch: "efficientnet.n-n-n-n-n.2",
          jax: "efficientnet.__call__.n-n-n-n-n.2",
        },
        includeChildRefs: false,
        children: [
          {
            id: "stage2.mbconv0",
            label: "mbconv.0",
            type: "MBConv",
            kind: "group",
            summary: "stride 2",
            defaultExpanded: true,
            sourceRefs: {
              pytorch: ["mbconv.self","mbconv.squeeze_channels-max-n-in_channels-n","mbconv.padding-kernel_size-n","mbconv.self-use_residual-stride-n-and-in_channels-out_channels","mbconv.if-expand_ratio-n","mbconv.nn-convnd-in_channels-expanded_channels-kernel_size-n-bias-false","mbconv.nn-batchnormnd-expanded_channels","mbconv.expanded_channels","mbconv.self-project-nn-sequential","mbconv.nn-convnd-expanded_channels-out_channels-kernel_size-n-bias-false","mbconv.nn-batchnormnd-out_channels","mbconv.forward.identity-x","mbconv.forward.if-self-expand-is-not-none","mbconv.forward.out-self-depthwise-out","mbconv.forward.out-self-project-out","mbconv.forward.if-self-use_residual","efficientnet.n-n-n-n-n.2","efficientnet.n-n-n-n-n.4","efficientnet.expand_ratio","efficientnet.forward.x-self-head-x"],
              jax: ["class-mbconv-nn-module","mbconv.out_channels-int","mbconv.expand_ratio-int","mbconv.stride-int","mbconv.kernel_size-int","mbconv.se_ratio-float-n","mbconv.nn-compact","mbconv.def-__call__-self-x-train-false","mbconv.__call__.in_channels-x-shape-n","mbconv.__call__.expanded_channels-in_channels-self-expand_ratio","mbconv.__call__.squeeze_channels-max-n-int-expanded_channels-self-se_ratio","mbconv.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels","mbconv.__call__.y-x","mbconv.__call__.if-self-expand_ratio-n","mbconv.__call__.y-nn-conv","mbconv.__call__.expanded_channels","mbconv.__call__.n-n","mbconv.__call__.use_bias-false","mbconv.__call__.name-expand_conv","mbconv.__call__.y","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y","mbconv.__call__.y-nn-silu-y","mbconv.__call__.y-nn-conv.2","mbconv.__call__.expanded_channels.2","mbconv.__call__.self-kernel_size-self-kernel_size","mbconv.__call__.strides-self-stride-self-stride","mbconv.__call__.padding-same","mbconv.__call__.feature_group_count-expanded_channels","mbconv.__call__.use_bias-false.2","mbconv.__call__.name-depthwise_conv","mbconv.__call__.y.2","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y","mbconv.__call__.y-nn-silu-y.2","mbconv.__call__.y-squeezeexcite-squeeze_channels-name-se-y","mbconv.__call__.y-nn-conv.3","mbconv.__call__.self-out_channels","mbconv.__call__.n-n.2","mbconv.__call__.use_bias-false.3","mbconv.__call__.name-project_conv","mbconv.__call__.y.3","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y","mbconv.__call__.if-use_residual","mbconv.__call__.y-y-x","mbconv.__call__.return-y"],
            },
            focusRef: {
              pytorch: "mbconv.self",
              jax: "class-mbconv-nn-module",
            },
            includeChildRefs: false,
            children: [
              {
                id: "stage2.mbconv0.expand",
                label: "expand",
                type: "1x1 Conv",
                kind: "conv",
                badges: ["16->96"],
                sourceRefs: {
                  pytorch: ["mbconv.nn-convnd-in_channels-expanded_channels-kernel_size-n-bias-false","mbconv.nn-batchnormnd-expanded_channels","mbconv.nn-silu-inplace-true","mbconv.code.6","mbconv.self-depthwise-nn-sequential","mbconv.nn-convnd","mbconv.forward.if-self-expand-is-not-none","mbconv.forward.out-self-expand-out","mbconv.forward.out-self-depthwise-out"],
                  jax: ["mbconv.__call__.expanded_channels-in_channels-self-expand_ratio","mbconv.__call__.if-self-expand_ratio-n","mbconv.__call__.y-nn-conv","mbconv.__call__.expanded_channels","mbconv.__call__.n-n","mbconv.__call__.use_bias-false","mbconv.__call__.name-expand_conv","mbconv.__call__.y","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y","mbconv.__call__.y-nn-silu-y"],
                },
                focusRef: {
                  pytorch: "mbconv.nn-convnd-in_channels-expanded_channels-kernel_size-n-bias-false",
                  jax: "mbconv.__call__.expanded_channels-in_channels-self-expand_ratio",
                },
                includeChildRefs: false,
              },
              {
                id: "stage2.mbconv0.depthwise",
                label: "depthwise",
                type: "DepthwiseConv",
                kind: "conv",
                badges: ["k=3", "s=2", "groups=96"],
                sourceRefs: {
                  pytorch: ["mbconv.expanded_channels","mbconv.expanded_channels.2","mbconv.kernel_size-kernel_size","mbconv.stride-stride","mbconv.padding-padding","mbconv.groups-expanded_channels","mbconv.bias-false","mbconv.code.7","mbconv.nn-batchnormnd-expanded_channels.2","mbconv.nn-silu-inplace-true.2","mbconv.code.8","mbconv.se_channels-int-expanded_channels-se_ratio-or-squeeze_channels"],
                  jax: ["mbconv.__call__.y-nn-conv.2","mbconv.__call__.expanded_channels.2","mbconv.__call__.self-kernel_size-self-kernel_size","mbconv.__call__.strides-self-stride-self-stride","mbconv.__call__.padding-same","mbconv.__call__.feature_group_count-expanded_channels","mbconv.__call__.use_bias-false.2","mbconv.__call__.name-depthwise_conv","mbconv.__call__.y.2","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y","mbconv.__call__.y-nn-silu-y.2"],
                },
                focusRef: {
                  pytorch: "mbconv.expanded_channels",
                  jax: "mbconv.__call__.y-nn-conv.2",
                },
                includeChildRefs: false,
              },
              {
                id: "stage2.mbconv0.se",
                label: "se",
                type: "SqueezeExcite",
                kind: "attention",
                badges: ["channel gate"],
                sourceRefs: {
                  pytorch: ["squeezeexcite.def-__init__","squeezeexcite.self-reduce-nn-convnd-channels-squeeze_channels-kernel_size-n","squeezeexcite.self-expand-nn-convnd-squeeze_channels-channels-kernel_size-n","squeezeexcite.forward.scale-self-reduce-scale","squeezeexcite.forward.scale-f-silu-scale","squeezeexcite.forward.scale-self-expand-scale","squeezeexcite.forward.scale-torch-sigmoid-scale","squeezeexcite.forward.out-x-scale","squeezeexcite.forward.return-out","mbconv.self-project-nn-sequential","mbconv.nn-convnd-expanded_channels-out_channels-kernel_size-n-bias-false","mbconv.forward.out-self-project-out"],
                  jax: ["class-squeezeexcite-nn-module","squeezeexcite.squeeze_channels-int","squeezeexcite.nn-compact","squeezeexcite.def-__call__-self-x","squeezeexcite.__call__.scale-jnp-mean-x-axis-n-n-keepdims-true","squeezeexcite.__call__.scale-nn-conv-self-squeeze_channels-n-n-name-reduce-scale","squeezeexcite.__call__.scale-nn-silu-scale","squeezeexcite.__call__.channel_count-x-shape-n","squeezeexcite.__call__.scale-nn-conv-channel_count-n-n-name-expand-scale","squeezeexcite.__call__.scale-nn-sigmoid-scale","squeezeexcite.__call__.y-x-scale","squeezeexcite.__call__.return-y"],
                },
                focusRef: {
                  pytorch: "squeezeexcite.def-__init__",
                  jax: "class-squeezeexcite-nn-module",
                },
                includeChildRefs: false,
              },
              {
                id: "stage2.mbconv0.project",
                label: "project",
                type: "1x1 Conv",
                kind: "conv",
                badges: ["96->24"],
                sourceRefs: {
                  pytorch: ["mbconv.nn-batchnormnd-out_channels","mbconv.code.9"],
                  jax: ["mbconv.__call__.y-nn-conv.3","mbconv.__call__.self-out_channels","mbconv.__call__.n-n.2","mbconv.__call__.use_bias-false.3","mbconv.__call__.name-project_conv","mbconv.__call__.y.3","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y"],
                },
                focusRef: {
                  pytorch: "mbconv.nn-batchnormnd-out_channels",
                  jax: "mbconv.__call__.y-nn-conv.3",
                },
                includeChildRefs: false,
              },
            ],
          },
          {
            id: "stage2.mbconv1",
            label: "mbconv.1",
            type: "MBConv",
            kind: "residual",
            summary: "same-shape residual",
            sourceRefs: {
              pytorch: ["mbconv.forward.identity-x","mbconv.forward.out-out-identity","mbconv.forward.return-out","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.forward.x-self-head-x"],
              jax: ["class-mbconv-nn-module","mbconv.out_channels-int","mbconv.expand_ratio-int","mbconv.stride-int","mbconv.kernel_size-int","mbconv.se_ratio-float-n","mbconv.nn-compact","mbconv.def-__call__-self-x-train-false","mbconv.__call__.in_channels-x-shape-n","mbconv.__call__.expanded_channels-in_channels-self-expand_ratio","mbconv.__call__.squeeze_channels-max-n-int-expanded_channels-self-se_ratio","mbconv.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels","mbconv.__call__.y-x","mbconv.__call__.if-self-expand_ratio-n","mbconv.__call__.y-nn-conv","mbconv.__call__.expanded_channels","mbconv.__call__.n-n","mbconv.__call__.use_bias-false","mbconv.__call__.name-expand_conv","mbconv.__call__.y","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y","mbconv.__call__.y-nn-silu-y","mbconv.__call__.y-nn-conv.2","mbconv.__call__.expanded_channels.2","mbconv.__call__.self-kernel_size-self-kernel_size","mbconv.__call__.strides-self-stride-self-stride","mbconv.__call__.padding-same","mbconv.__call__.feature_group_count-expanded_channels","mbconv.__call__.use_bias-false.2","mbconv.__call__.name-depthwise_conv","mbconv.__call__.y.2","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y","mbconv.__call__.y-nn-silu-y.2","mbconv.__call__.y-squeezeexcite-squeeze_channels-name-se-y","mbconv.__call__.y-nn-conv.3","mbconv.__call__.self-out_channels","mbconv.__call__.n-n.2","mbconv.__call__.use_bias-false.3","mbconv.__call__.name-project_conv","mbconv.__call__.y.3","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y","mbconv.__call__.if-use_residual","mbconv.__call__.y-y-x","mbconv.__call__.return-y"],
            },
            focusRef: {
              pytorch: "mbconv.forward.identity-x",
              jax: "class-mbconv-nn-module",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["efficientnet.n-n-n-n-n.2","efficientnet.n-n-n-n-n.5","efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.block-mbconv","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.block_stride","efficientnet.kernel_size","efficientnet.code.11","efficientnet.blocks-append-block","efficientnet.in_channels-out_channels","efficientnet.forward.x-self-head-x"],
          jax: ["efficientnet.__call__.n-n-n-n-n.3","efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.__call__.for-repeat_index-in-range-repeats","efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n","efficientnet.__call__.block_name-f-blocks-block_index","efficientnet.__call__.x-mbconv","efficientnet.__call__.out_channels","efficientnet.__call__.expand_ratio","efficientnet.__call__.block_stride","efficientnet.__call__.kernel_size","efficientnet.__call__.name-block_name","efficientnet.__call__.x-train-train","efficientnet.__call__.block_index-block_index-n"],
        },
        focusRef: {
          pytorch: "efficientnet.n-n-n-n-n.2",
          jax: "efficientnet.__call__.n-n-n-n-n.3",
        },
        includeChildRefs: false,
        children: [
          {
            id: "stage3.mbconv0",
            label: "mbconv.0",
            type: "MBConv",
            kind: "group",
            summary: "stride 2 + SE",
            sourceRefs: {
              pytorch: ["mbconv.expanded_channels","mbconv.padding-padding","mbconv.groups-expanded_channels","mbconv.bias-false","mbconv.code.7","mbconv.self-project-nn-sequential","mbconv.nn-convnd-expanded_channels-out_channels-kernel_size-n-bias-false","mbconv.forward.out-self-project-out","efficientnet.expand_ratio","efficientnet.forward.x-self-head-x"],
              jax: ["class-mbconv-nn-module","mbconv.out_channels-int","mbconv.expand_ratio-int","mbconv.stride-int","mbconv.kernel_size-int","mbconv.se_ratio-float-n","mbconv.nn-compact","mbconv.def-__call__-self-x-train-false","mbconv.__call__.in_channels-x-shape-n","mbconv.__call__.expanded_channels-in_channels-self-expand_ratio","mbconv.__call__.squeeze_channels-max-n-int-expanded_channels-self-se_ratio","mbconv.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels","mbconv.__call__.y-x","mbconv.__call__.if-self-expand_ratio-n","mbconv.__call__.y-nn-conv","mbconv.__call__.expanded_channels","mbconv.__call__.n-n","mbconv.__call__.use_bias-false","mbconv.__call__.name-expand_conv","mbconv.__call__.y","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y","mbconv.__call__.y-nn-silu-y","mbconv.__call__.y-nn-conv.2","mbconv.__call__.expanded_channels.2","mbconv.__call__.self-kernel_size-self-kernel_size","mbconv.__call__.strides-self-stride-self-stride","mbconv.__call__.padding-same","mbconv.__call__.feature_group_count-expanded_channels","mbconv.__call__.use_bias-false.2","mbconv.__call__.name-depthwise_conv","mbconv.__call__.y.2","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y","mbconv.__call__.y-nn-silu-y.2","mbconv.__call__.y-squeezeexcite-squeeze_channels-name-se-y","mbconv.__call__.y-nn-conv.3","mbconv.__call__.self-out_channels","mbconv.__call__.n-n.2","mbconv.__call__.use_bias-false.3","mbconv.__call__.name-project_conv","mbconv.__call__.y.3","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y","mbconv.__call__.if-use_residual","mbconv.__call__.y-y-x","mbconv.__call__.return-y"],
            },
            focusRef: {
              pytorch: "mbconv.expanded_channels",
              jax: "class-mbconv-nn-module",
            },
            includeChildRefs: false,
          },
          {
            id: "stage3.mbconv1",
            label: "mbconv.1",
            type: "MBConv",
            kind: "residual",
            summary: "identity add",
            sourceRefs: {
              pytorch: ["mbconv.forward.out-out-identity","mbconv.forward.return-out","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.forward.x-self-head-x"],
              jax: ["class-mbconv-nn-module","mbconv.out_channels-int","mbconv.expand_ratio-int","mbconv.stride-int","mbconv.kernel_size-int","mbconv.se_ratio-float-n","mbconv.nn-compact","mbconv.def-__call__-self-x-train-false","mbconv.__call__.in_channels-x-shape-n","mbconv.__call__.expanded_channels-in_channels-self-expand_ratio","mbconv.__call__.squeeze_channels-max-n-int-expanded_channels-self-se_ratio","mbconv.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels","mbconv.__call__.y-x","mbconv.__call__.if-self-expand_ratio-n","mbconv.__call__.y-nn-conv","mbconv.__call__.expanded_channels","mbconv.__call__.n-n","mbconv.__call__.use_bias-false","mbconv.__call__.name-expand_conv","mbconv.__call__.y","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y","mbconv.__call__.y-nn-silu-y","mbconv.__call__.y-nn-conv.2","mbconv.__call__.expanded_channels.2","mbconv.__call__.self-kernel_size-self-kernel_size","mbconv.__call__.strides-self-stride-self-stride","mbconv.__call__.padding-same","mbconv.__call__.feature_group_count-expanded_channels","mbconv.__call__.use_bias-false.2","mbconv.__call__.name-depthwise_conv","mbconv.__call__.y.2","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y","mbconv.__call__.y-nn-silu-y.2","mbconv.__call__.y-squeezeexcite-squeeze_channels-name-se-y","mbconv.__call__.y-nn-conv.3","mbconv.__call__.self-out_channels","mbconv.__call__.n-n.2","mbconv.__call__.use_bias-false.3","mbconv.__call__.name-project_conv","mbconv.__call__.y.3","mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y","mbconv.__call__.if-use_residual","mbconv.__call__.y-y-x","mbconv.__call__.return-y"],
            },
            focusRef: {
              pytorch: "mbconv.forward.out-out-identity",
              jax: "class-mbconv-nn-module",
            },
            includeChildRefs: false,
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
        sourceRefs: {
          pytorch: ["efficientnet.n-n-n-n-n.2","efficientnet.n-n-n-n-n.6","efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.block-mbconv","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.block_stride","efficientnet.kernel_size","efficientnet.code.11","efficientnet.blocks-append-block","efficientnet.in_channels-out_channels","efficientnet.forward.x-self-head-x"],
          jax: ["efficientnet.__call__.n-n-n-n-n.4","efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.__call__.for-repeat_index-in-range-repeats","efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n","efficientnet.__call__.block_name-f-blocks-block_index","efficientnet.__call__.x-mbconv","efficientnet.__call__.out_channels","efficientnet.__call__.expand_ratio","efficientnet.__call__.block_stride","efficientnet.__call__.kernel_size","efficientnet.__call__.name-block_name","efficientnet.__call__.x-train-train","efficientnet.__call__.block_index-block_index-n"],
        },
        focusRef: {
          pytorch: "efficientnet.n-n-n-n-n.2",
          jax: "efficientnet.__call__.n-n-n-n-n.4",
        },
        includeChildRefs: false,
      },
      {
        id: "stage5",
        label: "stage5",
        type: "MBConv6",
        kind: "group",
        summary: "3 blocks",
        badges: ["80->112", "14x14", "k=5"],
        sourceRefs: {
          pytorch: ["efficientnet.n-n-n-n-n.2","efficientnet.n-n-n-n-n.7","efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.block-mbconv","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.block_stride","efficientnet.kernel_size","efficientnet.code.11","efficientnet.blocks-append-block","efficientnet.in_channels-out_channels","efficientnet.forward.x-self-head-x"],
          jax: ["efficientnet.__call__.n-n-n-n-n.5","efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.__call__.for-repeat_index-in-range-repeats","efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n","efficientnet.__call__.block_name-f-blocks-block_index","efficientnet.__call__.x-mbconv","efficientnet.__call__.out_channels","efficientnet.__call__.expand_ratio","efficientnet.__call__.block_stride","efficientnet.__call__.kernel_size","efficientnet.__call__.name-block_name","efficientnet.__call__.x-train-train","efficientnet.__call__.block_index-block_index-n"],
        },
        focusRef: {
          pytorch: "efficientnet.n-n-n-n-n.2",
          jax: "efficientnet.__call__.n-n-n-n-n.5",
        },
        includeChildRefs: false,
      },
      {
        id: "stage6",
        label: "stage6",
        type: "MBConv6",
        kind: "group",
        summary: "4 blocks",
        badges: ["112->192", "7x7", "k=5"],
        sourceRefs: {
          pytorch: ["efficientnet.n-n-n-n-n.2","efficientnet.code.4","efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.block-mbconv","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.block_stride","efficientnet.kernel_size","efficientnet.code.11","efficientnet.blocks-append-block","efficientnet.in_channels-out_channels","efficientnet.forward.x-self-head-x"],
          jax: ["efficientnet.__call__.n-n-n-n-n.6","efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.__call__.for-repeat_index-in-range-repeats","efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n","efficientnet.__call__.block_name-f-blocks-block_index","efficientnet.__call__.x-mbconv","efficientnet.__call__.out_channels","efficientnet.__call__.expand_ratio","efficientnet.__call__.block_stride","efficientnet.__call__.kernel_size","efficientnet.__call__.name-block_name","efficientnet.__call__.x-train-train","efficientnet.__call__.block_index-block_index-n"],
        },
        focusRef: {
          pytorch: "efficientnet.n-n-n-n-n.2",
          jax: "efficientnet.__call__.n-n-n-n-n.6",
        },
        includeChildRefs: false,
      },
      {
        id: "stage7",
        label: "stage7",
        type: "MBConv6",
        kind: "group",
        summary: "1 block",
        badges: ["192->320", "7x7"],
        sourceRefs: {
          pytorch: ["efficientnet.n-n-n-n-n.2","efficientnet.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.block-mbconv","efficientnet.in_channels","efficientnet.out_channels","efficientnet.expand_ratio","efficientnet.block_stride","efficientnet.kernel_size","efficientnet.code.11","efficientnet.blocks-append-block","efficientnet.in_channels-out_channels","efficientnet.forward.x-self-head-x"],
          jax: ["efficientnet.__call__.n-n-n-n-n.7","efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings","efficientnet.__call__.for-repeat_index-in-range-repeats","efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n","efficientnet.__call__.block_name-f-blocks-block_index","efficientnet.__call__.x-mbconv","efficientnet.__call__.out_channels","efficientnet.__call__.expand_ratio","efficientnet.__call__.block_stride","efficientnet.__call__.kernel_size","efficientnet.__call__.name-block_name","efficientnet.__call__.x-train-train","efficientnet.__call__.block_index-block_index-n"],
        },
        focusRef: {
          pytorch: "efficientnet.n-n-n-n-n.2",
          jax: "efficientnet.__call__.n-n-n-n-n.7",
        },
        includeChildRefs: false,
      },
      {
        id: "head",
        label: "head",
        type: "Conv-Pool-FC",
        kind: "group",
        summary: "1280 classifier",
        sourceRefs: {
          pytorch: ["efficientnet.in_channels.2","efficientnet.n.3","efficientnet.self-classifier-nn-linear-n-num_classes","efficientnet.def-forward-self-x","efficientnet.forward.x-torch-flatten-x-n"],
          jax: [],
        },
        focusRef: {
          pytorch: "efficientnet.in_channels.2",
          jax: "efficientnet.__call__.x-nn-conv.2",
        },
        includeChildRefs: true,
        children: [
          {
            id: "head.conv",
            label: "conv",
            type: "1x1 Conv",
            kind: "conv",
            badges: ["320->1280"],
            sourceRefs: {
              pytorch: ["efficientnet.kernel_size-n.2","efficientnet.bias-false.2","efficientnet.code.14","efficientnet.nn-batchnormnd-n.2","efficientnet.nn-silu-inplace-true.2","efficientnet.code.15","efficientnet.forward.logits-self-classifier-x"],
              jax: ["efficientnet.__call__.x-nn-conv.2","efficientnet.__call__.n.2","efficientnet.__call__.n-n.2","efficientnet.__call__.use_bias-false.2","efficientnet.__call__.name-head_conv","efficientnet.__call__.x.2","efficientnet.__call__.x-nn-batchnorm-use_running_average-not-train-name-head_bn-x","efficientnet.__call__.x-nn-silu-x.2"],
            },
            focusRef: {
              pytorch: "efficientnet.kernel_size-n.2",
              jax: "efficientnet.__call__.x-nn-conv.2",
            },
            includeChildRefs: false,
          },
          {
            id: "head.pool",
            label: "avgpool",
            type: "AdaptiveAvgPool2d",
            kind: "pool",
            badges: ["1x1"],
            sourceRefs: {
              pytorch: ["efficientnet.head.pool"],
              jax: ["efficientnet.__call__.x-jnp-mean-x-axis-n-n"],
            },
            focusRef: {
              pytorch: "efficientnet.head.pool",
              jax: "efficientnet.__call__.x-jnp-mean-x-axis-n-n",
            },
            includeChildRefs: false,
          },
          {
            id: "head.classifier",
            label: "classifier",
            type: "Linear",
            kind: "linear",
            badges: ["1280->1000"],
            sourceRefs: {
              pytorch: ["efficientnet.self-classifier-nn-linear-n-num_classes","efficientnet.forward.logits-self-classifier-x"],
              jax: ["efficientnet.__call__.logits-nn-dense-self-num_classes-name-classifier-x"],
            },
            focusRef: {
              pytorch: "efficientnet.self-classifier-nn-linear-n-num_classes",
              jax: "efficientnet.__call__.logits-nn-dense-self-num_classes-name-classifier-x",
            },
            includeChildRefs: false,
          },
        ],
      },
    ],
  },
};

const models: ModelSpec[] = modelCatalog.map((entry) => {
  const definition = modelDefinitions[entry.id];

  return {
    ...definition,
    ...modelSourcePair("sourceBaseName" in entry ? entry.sourceBaseName : entry.id),
    id: entry.id,
    label: entry.label,
    paper: {
      ...entry.paper,
      publishedDate: entry.publishedDate,
      pdfUrl: paperPdfUrl(entry.id),
    },
  };
});

type HighlightSource = Pick<
  ModelSpec,
  "fileName" | "jaxFileName" | "code" | "jaxCode" | "highlights" | "jaxHighlights"
>;

function highlightSourceForLanguage(model: HighlightSource, language: CodeLanguage) {
  return language === "jax"
    ? { fileName: model.jaxFileName, code: model.jaxCode, manifest: model.jaxHighlights }
    : { fileName: model.fileName, code: model.code, manifest: model.highlights };
}

function architectureChildren(node: ArchNode) {
  return node.children ?? node.lazyChildren?.() ?? [];
}

function resolveArchitectureHighlight(model: HighlightSource, node: ArchNode, language: CodeLanguage) {
  const { fileName, manifest } = highlightSourceForLanguage(model, language);
  const lines = node.sourceRefs[language].flatMap((ref) => {
    const resolved = manifest[ref];
    if (!resolved) {
      throw new Error(`Missing ${language} source ref ${fileName}:${ref} for ${node.id}`);
    }
    return resolved.lines;
  });

  if (node.includeChildRefs) {
    for (const child of architectureChildren(node)) {
      lines.push(...resolveArchitectureHighlight(model, child, language).lines);
    }
  }

  const resolvedLines = [...new Set(lines)].sort((left, right) => left - right);
  const focusRef = node.focusRef?.[language];
  const focusAnchor = focusRef ? manifest[focusRef] : undefined;
  if (focusRef && !focusAnchor) {
    throw new Error(`Missing ${language} focus ref ${fileName}:${focusRef} for ${node.id}`);
  }
  const focusLine = focusAnchor?.focusLine ?? resolvedLines[0] ?? null;
  if (focusLine !== null && !resolvedLines.includes(focusLine)) {
    throw new Error(`Focus ref does not belong to ${language} selection for ${fileName}:${node.id}`);
  }

  return { lines: resolvedLines, focusLine };
}

const referencedRefsByFile = new Map<string, Set<string>>();

function validateHighlightTree(modelLabel: string, model: HighlightSource, nodes: ArchNode[]) {
  const seenNodeIds = new Set<string>();

  const validateNode = (node: ArchNode) => {
    if (seenNodeIds.has(node.id)) {
      throw new Error(`Duplicate architecture node id for ${modelLabel}: ${node.id}`);
    }
    seenNodeIds.add(node.id);

    for (const language of ["pytorch", "jax"] as const) {
      const source = highlightSourceForLanguage(model, language);
      const referencedRefs = referencedRefsByFile.get(source.fileName) ?? new Set<string>();
      node.sourceRefs[language].forEach((ref) => referencedRefs.add(ref));
      const focusRef = node.focusRef?.[language];
      if (focusRef) {
        referencedRefs.add(focusRef);
      }
      referencedRefsByFile.set(source.fileName, referencedRefs);

      const { lines } = resolveArchitectureHighlight(model, node, language);
      if (lines.length === 0) {
        throw new Error(`Missing ${language} highlight lines for ${modelLabel}:${node.id}`);
      }

      for (const line of lines) {
        if (!Number.isInteger(line) || line < 1 || line > source.code.length) {
          throw new Error(`Invalid ${language} highlight line ${line} for ${modelLabel}:${node.id}`);
        }
      }
    }

    architectureChildren(node).forEach(validateNode);
  };

  nodes.forEach(validateNode);
}

for (const model of models) {
  validateHighlightTree(model.id, model, model.nodes);

  for (const variant of model.variants ?? []) {
    validateHighlightTree(variant.id, variant, variant.nodes);
  }
}

for (const [fileName, manifest] of Object.entries(modelHighlightManifest)) {
  const referencedRefs = referencedRefsByFile.get(fileName) ?? new Set<string>();
  const resnetFamilyRefs = /^resnet\d+(_jax)?\.py$/.test(fileName)
    ? new Set(
        [...referencedRefsByFile.entries()]
          .filter(([candidate]) => {
            const sameLanguage = fileName.endsWith("_jax.py")
              ? candidate.endsWith("_jax.py")
              : !candidate.endsWith("_jax.py");
            return sameLanguage && /^resnet\d+/.test(candidate);
          })
          .flatMap(([, refs]) => [...refs]),
      )
    : null;
  for (const ref of Object.keys(manifest)) {
    if (!referencedRefs.has(ref) && !resnetFamilyRefs?.has(ref)) {
      throw new Error(`Orphaned architecture anchor in ${fileName}: ${ref}`);
    }
  }
}

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
  return model.activeVariantId ?? model.id;
}

function resolveModelVariant(model: ModelSpec, variantId: string | undefined) {
  const variant = model.variants?.find((entry) => entry.id === variantId) ?? model.variants?.[0];

  if (!variant) {
    return model;
  }

  return {
    ...model,
    id: variant.id,
    label: variant.label,
    stats: variant.stats,
    fileName: variant.fileName,
    jaxFileName: variant.jaxFileName,
    nodes: variant.nodes,
    code: variant.code,
    jaxCode: variant.jaxCode,
    highlights: variant.highlights,
    jaxHighlights: variant.jaxHighlights,
    activeVariantId: variant.id,
  };
}

function selectedHighlight(model: ModelSpec, selected: ArchNode | null, language: CodeLanguage) {
  if (!selected) {
    return { lines: [], focusLine: null };
  }

  return resolveArchitectureHighlight(model, selected, language);
}

function selectedCodeContext(model: ModelSpec, selected: ArchNode | null, language: CodeLanguage) {
  const currentFile = getCodeForLanguage(model, language);

  return selectedHighlight(model, selected, language).lines
    .filter((lineNumber) => currentFile.code[lineNumber - 1] !== undefined)
    .map((lineNumber) => ({
      lineNumber,
      text: currentFile.code[lineNumber - 1],
    }));
}

function referencedCodeContext(
  model: ModelSpec,
  selection: AgentCodeSelection | UserCodeSelection | null,
) {
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

  if (value === "pytorch") {
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
  const lines = normalizeLineNumbers(candidate, currentFile.code.length);

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
  const source = getCodeForLanguage(model, language);
  const currentFile = {
    ...source,
    notebookName: notebookFileName(source.fileName),
  };
  const selectedHighlightForLanguage = selectedHighlight(model, selected, language);
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
  const highlightedLineNumbers = activeAgentSelection ? activeAgentSelection.lines : selectedHighlightForLanguage.lines;
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
  const firstScrollLine = activeAgentSelection
    ? activeAgentSelection.lines.find((lineNumber) => currentFile.code[lineNumber - 1] !== undefined) ?? null
    : selectedHighlightForLanguage.focusLine;
  const scrollLineKey = `${firstScrollLine ?? ""}:${highlightedLineNumbers.join(",")}`;

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
          <span className="editor-file-name">{currentFile.fileName}</span>
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
  const agentSelectedLines = referencedCodeContext(model, agentCodeSelection);
  const userSelectedLines = referencedCodeContext(model, userCodeSelection);
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
              stats: model.stats,
            },
            paper: {
              title: model.paper.title,
              authors: model.paper.authors,
              year: model.paper.publishedDate.slice(0, 4),
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
      const payload = (await response.json()) as ChatResponse;

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
  const [visibleColumns, setVisibleColumns] = useState<Record<PaneKey, boolean>>(() => defaultVisibleColumns);
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

    const nextPath = modelRoutePath(nextModelId);
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
