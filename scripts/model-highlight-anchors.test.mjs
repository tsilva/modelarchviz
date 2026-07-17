import assert from "node:assert/strict";
import test from "node:test";
import { cleanedPythonAndHighlightManifest, stripNotebookAnchorMarkers } from "./model-highlight-anchors.mjs";

function codeCell(lines, tags = []) {
  return { cell_type: "code", tags, lines };
}

test("strips markers and emits generated line selections", () => {
  const cells = [codeCell([
    "# @arch encoder.norm:start",
    "x = self.norm(x)",
    "# @arch encoder.norm:end",
  ])];
  const result = cleanedPythonAndHighlightManifest(cells, "model.py");
  assert.equal(result.cleanedPython, "x = self.norm(x)\n");
  assert.deepEqual(result.highlights, { "encoder.norm": { lines: [1], focusLine: 1 } });
  assert.deepEqual(stripNotebookAnchorMarkers(cells, "model.py")[0].lines, ["x = self.norm(x)"]);
});

for (const [name, lines, message] of [
  ["malformed", ["# @arch bad"], /Malformed/],
  ["nested", ["# @arch outer:start", "# @arch inner:start"], /Nested/],
  ["duplicate", ["# @arch same:start", "x = 1", "# @arch same:end", "# @arch same:start", "y = 2", "# @arch same:end"], /Duplicate/],
  ["empty", ["# @arch empty:start", "# @arch empty:end"], /Empty/],
  ["metadata-only", ["# @arch metadata:start", "# comment", "", "# @arch metadata:end"], /Metadata-only/],
  ["missing end", ["# @arch missing:start", "x = 1"], /Orphaned.*start/],
  ["orphaned end", ["# @arch missing:end"], /Orphaned.*end/],
]) {
  test(`rejects ${name} anchors`, () => {
    assert.throws(() => cleanedPythonAndHighlightManifest([codeCell(lines)], "broken.py"), message);
  });
}

test("rejects anchors in notebook-only cells", () => {
  assert.throws(
    () => stripNotebookAnchorMarkers([codeCell(["# @arch hidden:start", "x = 1", "# @arch hidden:end"], ["notebook-only"])], "hidden.py"),
    /outside web-visible code/,
  );
});
