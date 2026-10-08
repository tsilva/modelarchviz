import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const notFoundError = new Error("Route not found");
const ModelArchVizAppStub = () => null;

function evaluateModule(relativePath, requireImpl = require) {
  const fileName = new URL(relativePath, import.meta.url);
  const compiled = ts.transpileModule(readFileSync(fileName, "utf8"), {
    fileName: fileName.pathname,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", compiled)(requireImpl, module, module.exports);
  return module.exports;
}

const routes = evaluateModule("../app/model-routes.ts");
const page = evaluateModule("../app/models/[modelId]/page.tsx", (specifier) => {
  if (specifier === "../../model-routes") return routes;
  if (specifier === "../../model-arch-viz-app") return ModelArchVizAppStub;
  if (specifier === "next/navigation") {
    return { notFound: () => { throw notFoundError; } };
  }
  if (specifier === "react/jsx-runtime") return require(specifier);
  throw new Error(`Unexpected page import: ${specifier}`);
});

test("catalog routes render the requested model and metadata from promised params", async () => {
  for (const model of routes.modelCatalog) {
    const props = { params: Promise.resolve({ modelId: model.id }) };
    const rendered = await page.default(props);
    assert.equal(rendered.type, ModelArchVizAppStub);
    assert.equal(rendered.props.initialModelId, model.id);

    const metadata = await page.generateMetadata(props);
    assert.equal(metadata.title, `${model.title} | ModelArchViz`);
    assert.equal(metadata.alternates.canonical, `/models/${model.id}`);
  }
});

test("unknown model routes return not found and no model metadata", async () => {
  const props = { params: Promise.resolve({ modelId: "unknown-model" }) };
  await assert.rejects(page.default(props), (error) => error === notFoundError);
  assert.deepEqual(await page.generateMetadata(props), {});
});
