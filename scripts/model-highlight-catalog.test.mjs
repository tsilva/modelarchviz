import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function transpile(source, fileName) {
  return ts.transpileModule(source, {
    fileName,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
  }).outputText;
}

function evaluateModule(relativePath) {
  const fileName = path.join(repoRoot, relativePath);
  const module = { exports: {} };
  new Function("require", "module", "exports", transpile(readFileSync(fileName, "utf8"), fileName))(
    require,
    module,
    module.exports,
  );
  return module.exports;
}

function loadCatalog() {
  const appPath = path.join(repoRoot, "app/model-arch-viz-app.tsx");
  const appSource = readFileSync(appPath, "utf8");
  const cutoff = appSource.indexOf("const modelsByPublicationDate");
  assert.notEqual(cutoff, -1, "catalog initialization cutoff must exist");

  const generated = evaluateModule("app/generated/model-sources.ts");
  const routes = evaluateModule("app/model-routes.ts");
  const isolatedGlobal = {};
  const module = { exports: {} };
  const requireImpl = (specifier) => {
    if (specifier === "./generated/model-sources") return generated;
    if (specifier === "./model-routes") return routes;
    if (specifier === "./chat-contract") return { normalizeLineNumbers: (lines) => lines };
    if (specifier === "react" || specifier === "next/navigation" || specifier === "react/jsx-runtime") return {};
    throw new Error(`Unexpected catalog import: ${specifier}`);
  };
  const capture = "\nglobalThis.__catalog = { models, resolveArchitectureHighlight, architectureChildren };\n";
  const compiled = transpile(`${appSource.slice(0, cutoff)}${capture}`, appPath);
  new Function("require", "module", "exports", "process", "globalThis", compiled)(
    requireImpl,
    module,
    module.exports,
    process,
    isolatedGlobal,
  );
  return isolatedGlobal.__catalog;
}

const catalog = loadCatalog();
const catalogModels = [
  ...catalog.models,
  ...catalog.models.flatMap((model) => model.variants ?? []),
].filter((model, index, models) => models.findIndex((candidate) => candidate.id === model.id) === index);

function findNode(nodes, id) {
  for (const node of nodes) {
    if (node.id === id) return node;
    const nested = findNode(catalog.architectureChildren(node), id);
    if (nested) return nested;
  }
  return null;
}

function resolvedText(modelId, nodeId, language) {
  const model = catalogModels.find((candidate) => candidate.id === modelId);
  assert.ok(model, `missing model ${modelId}`);
  const node = findNode(model.nodes, nodeId);
  assert.ok(node, `missing node ${modelId}:${nodeId}`);
  const source = language === "jax" ? model.jaxCode : model.code;
  return catalog.resolveArchitectureHighlight(model, node, language).lines
    .map((lineNumber) => source[lineNumber - 1])
    .join("\n");
}

test("every catalog node resolves concrete source in both languages", () => {
  for (const model of catalogModels) {
    const visit = (node) => {
      for (const language of ["pytorch", "jax"]) {
        const selection = catalog.resolveArchitectureHighlight(model, node, language);
        assert.ok(selection.lines.length > 0, `${model.id}:${node.id}:${language}`);
        assert.ok(selection.lines.includes(selection.focusLine), `${model.id}:${node.id}:${language}:focus`);
      }
      catalog.architectureChildren(node).forEach(visit);
    };
    model.nodes.forEach(visit);
  }
});

test("every catalog operation resolves semantically matching source", () => {
  const expectedTextByKind = {
    activation: /relu|gelu|sigmoid|tanh|softmax|silu|clamp|activation|\bact\b/i,
    attention: /attention|attn|query|key|value|qkv|score|weight|mask|softmax|head|distance|argmin|lookup|quantized|squeeze|excite|gate|sigmoid|scale/i,
    concat: /cat|concat/i,
    conv: /conv/i,
    dropout: /dropout/i,
    embedding: /embed|token|position|patch|latent|\bcode\b|quant/i,
    head: /head|classifier|logits|linear|dense|\bfc\b|output|\bout_|predicted|pool|proj|loss|criterion|optimizer|prob|sample/i,
    linear: /linear|linspace|dense|classifier|logit|weight|matmul|\bfc\b|proj|variance|sqrt_alpha|noisy_images/i,
    mlp: /mlp|feed.forward|linear|dense|\bfc\b|gelu|relu/i,
    norm: /norm|normalize|mean|variance|\bvar\b|sqrt|\bstd\b/i,
    pool: /pool|mean/i,
    recurrent: /lstm|gru|rnn|gate|hidden|cell|state|sequence/i,
    reshape: /flatten|reshape|view|permute|transpose|rearrange|squeeze|unsqueeze|shape|reparameterize|sample|epsilon|latent|reverse|index_select|arange|mean|cumprod/i,
    residual: /residual|shortcut|identity|\badd\b|\+|block/i,
  };

  for (const model of catalogModels) {
    const visit = (node) => {
      const expectedText = expectedTextByKind[node.kind];
      if (expectedText) {
        for (const language of ["pytorch", "jax"]) {
          assert.match(
            resolvedText(model.id, node.id, language),
            expectedText,
            `${model.id}:${node.id}:${language} must resolve ${node.kind} source`,
          );
        }
      }
      catalog.architectureChildren(node).forEach(visit);
    };
    model.nodes.forEach(visit);
  }
});

test("regression mappings resolve the intended architecture operations", () => {
  const alexNetFeatureOperations = [
    ["features.conv1", /nn\.Conv2d\(3, 96,/],
    ["features.relu1", /nn\.ReLU\(inplace=True\)/],
    ["features.lrn1", /nn\.LocalResponseNorm\(size=5,/],
    ["features.pool1", /nn\.MaxPool2d\(kernel_size=3, stride=2\)/],
    ["features.conv2", /nn\.Conv2d\(96, 256,/],
    ["features.relu2", /nn\.ReLU\(inplace=True\)/],
    ["features.lrn2", /nn\.LocalResponseNorm\(size=5,/],
    ["features.pool2", /nn\.MaxPool2d\(kernel_size=3, stride=2\)/],
    ["features.conv3", /nn\.Conv2d\(256, 384,/],
    ["features.relu3", /nn\.ReLU\(inplace=True\)/],
    ["features.conv4", /nn\.Conv2d\(384, 384,/],
    ["features.relu4", /nn\.ReLU\(inplace=True\)/],
    ["features.conv5", /nn\.Conv2d\(384, 256,/],
    ["features.relu5", /nn\.ReLU\(inplace=True\)/],
    ["features.pool5", /nn\.MaxPool2d\(kernel_size=3, stride=2\)/],
  ];

  for (const [nodeId, expectedOperation] of alexNetFeatureOperations) {
    assert.match(resolvedText("alexnet", nodeId, "pytorch"), expectedOperation);
  }

  assert.match(resolvedText("transformer", "encoder.0.norm2", "pytorch"), /ffn_residual = x \+ ffn/);
  assert.match(resolvedText("transformer", "encoder.0.norm2", "pytorch"), /self\.norm2\(ffn_residual\)/);
  assert.match(resolvedText("vit", "encoder.block.0.resid2", "pytorch"), /x = x \+ mlp_output/);
  assert.match(resolvedText("googlenet", "stage3.pool", "pytorch"), /max_pool2d/);
  assert.match(resolvedText("googlenet", "stage4.pool", "pytorch"), /max_pool2d/);
  assert.match(resolvedText("googlenet", "classifier.avgpool", "pytorch"), /avgpool/);
  assert.match(resolvedText("googlenet", "classifier.fc", "pytorch"), /self\.fc/);
  assert.match(resolvedText("efficientnet", "head.pool", "pytorch"), /adaptive_avg_pool2d/);
  assert.match(resolvedText("efficientnet", "head.classifier", "pytorch"), /self\.classifier/);
});
