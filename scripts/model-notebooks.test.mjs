import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const notebooks = readdirSync(path.join(root, "public/notebooks"))
  .filter((name) => name.endsWith(".ipynb"))
  .map((name) => ({ name, ...JSON.parse(readFileSync(path.join(root, "public/notebooks", name), "utf8")) }));

test("every generated notebook cell contains valid Python with real line breaks", () => {
  const cells = notebooks.flatMap((notebook) => notebook.cells
    .filter((cell) => cell.cell_type === "code")
    .map((cell, index) => ({ name: `${notebook.name}:cell-${index}`, source: cell.source.join("") })));
  const result = spawnSync("python3", ["-c", "import json,sys; [compile(c['source'], c['name'], 'exec') for c in json.load(sys.stdin)]"], {
    input: JSON.stringify(cells), encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr || result.error?.message);
});

test("all JAX notebooks install the constrained environment before model imports", () => {
  const requirements = readFileSync(path.join(root, "requirements-notebooks-jax.txt"), "utf8");
  const constraints = readFileSync(path.join(root, "constraints.txt"), "utf8");
  assert.match(requirements, /^-c constraints\.txt$/m);
  const jaxNotebooks = notebooks.filter((notebook) => notebook.name.endsWith("_jax.ipynb"));
  assert.ok(jaxNotebooks.length > 0);
  for (const notebook of jaxNotebooks) {
    const [setup, imports] = notebook.cells;
    assert.ok(setup.metadata.tags.includes("environment-setup"), notebook.name);
    const source = setup.source.join("");
    assert.ok(source.includes(JSON.stringify(requirements)), notebook.name);
    assert.ok(source.includes(JSON.stringify(constraints)), notebook.name);
    assert.match(source, /'pip', 'install', '--quiet', '-r'/);
    assert.match(imports.source.join(""), /import jax/);
    assert.doesNotMatch(source, /from flax|import jax/);
  }
  for (const notebook of notebooks.filter((notebook) => !notebook.name.endsWith("_jax.ipynb"))) {
    assert.match(notebook.cells[0].source.join(""), /import torch/, notebook.name);
    assert.ok(!notebook.cells.some((cell) => cell.metadata.tags?.includes("environment-setup")), notebook.name);
  }
});
