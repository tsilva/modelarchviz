---
name: audit-model-architecture
description: Audit a ModelArchViz model by opening the local app with the official OpenAI Browser Use plugin, clicking architecture nodes one by one, and verifying that each selected architecture part highlights the correct source code lines in the code pane. Use when checking model architecture specs, code highlight mappings, expanded/lazy architecture groups, generated model-code artifacts, PyTorch/JAX language parity, or regressions in the ModelArchViz UI.
---

# Audit Model Architecture

## Purpose

Use this skill to validate that a ModelArchViz architecture tree matches the implementation code and that every clickable architecture part selects the expected highlighted code lines in the browser UI.

## Workflow

1. Inspect the target model contract.
   - Read model identity, route, paper, and source metadata in `app/model-routes.ts`.
   - Read architecture nodes and code-highlight mappings in `app/model-arch-viz-app.tsx`.
   - Find the `ModelSpec` entry by `id`, `label`, or `fileName`.
   - Treat each `ArchNode.codeLines` value as the expected highlight set for that node in the active code language, unless the app has language-specific mappings.
   - Include eager `children` and lazy groups from `lazyChildren`.
   - For generated groups, sample representative generated children unless the user requests exhaustive coverage.

2. Inspect the corresponding implementation code.
   - Treat `app/model-notebooks/*.py` as the source of truth for model implementations.
   - Inspect `app/generated/model-sources.ts` for the exact generated strings and line numbers rendered in the UI.
   - Do not hand-edit `app/generated/model-sources.ts` or `public/notebooks/*.ipynb`; if a fix is requested, edit `app/model-notebooks/*.py` and regenerate with `pnpm generate:model-artifacts`.
   - Confirm each node label/type is conceptually correct for the code lines it references.
   - Flag code lines that are stale, shifted, missing, duplicated only by accident, or mapped to unrelated operations.
   - For PyTorch/JAX parity checks, switch the UI language selector and verify whether shared line mappings still make sense for the selected source.

3. Start or reuse the local app.
   - Prefer the repo's existing script: `pnpm dev -- --port auto`.
   - Use the currently running localhost URL when already available.
   - Keep npm/pnpm supply-chain hardening intact; do not install or update packages unless required for the user's task.

4. Use the official OpenAI Browser Use plugin for browser testing.
   - Load the Browser skill first when it is available.
   - Open the exact local URL printed by the dev-server wrapper.
   - Do not substitute shell-only checks for UI behavior. The browser audit is the source of truth for selection and highlighting behavior.

5. Select the target model in the `Select model` combobox.
   - Keep the architecture and code columns visible.
   - Use the search field only as a locator aid; clear it before final visual checks if dimmed rows could obscure the result.

6. Click architecture parts one by one.
   - Expand groups with the chevron control before checking descendants.
   - Click the row or head tile for each node under audit.
   - After each click, collect the selected architecture node, highlighted code line numbers, and highlighted code text from the DOM.
   - Compare highlighted line numbers as a set. The DOM returns highlights in source order, while `codeLines` arrays may not be sorted.
   - Expected UI selectors in this app:
     - selected architecture rows: `.arch-row.selected`, `.head-tile.selected`
     - highlighted code lines: `.code-line.highlighted .line-number`
     - model selector: `[aria-label="Select model"]`
     - language selector: `[aria-label="Select code language"]`

7. Compare browser state to the source spec.
   - The highlighted line number set must equal the clicked node's `codeLines` set.
   - Verify the highlighted text corresponds to the architecture operation, not only that line numbers match.
   - In language parity checks, inspect the highlighted text after switching languages. Shared line mappings can pass numerically for PyTorch while selecting unrelated JAX operations, blank lines, sample-run code, or no lines.
   - For lazy nodes, confirm expansion creates the intended child nodes before clicking them.
   - If a row click selects the parent while chevron click only expands/collapses, treat that as correct behavior.

8. Report findings with evidence.
   - Start with bugs and regressions, ordered by severity.
   - Include the model, node id/label, expected lines, observed lines, and why the mapping is wrong.
   - Reference local files with absolute paths and line numbers.
   - Mention the browser URL, model, and language audited.
   - If no issues are found, say that clearly and describe any sampling limits.

## Practical Browser Snippet

After Browser setup, this Playwright-side extraction pattern is useful for each clicked node:

```js
const selectedNodeText = await tab.playwright
  .locator(".arch-row.selected, .head-tile.selected")
  .allTextContents();

const highlightedLines = await tab.playwright
  .locator(".code-line.highlighted .line-number")
  .allTextContents();

const highlightedCode = await tab.playwright.evaluate(() =>
  Array.from(document.querySelectorAll(".code-line.highlighted")).map((line) => ({
    line: line.querySelector(".line-number")?.textContent?.trim() ?? "",
    code: line.querySelector("code")?.textContent ?? "",
  }))
);

console.log({ selectedNodeText, highlightedLines, highlightedCode });
```

Prefer accessible locators for model/language selects and visible text locators for rows. Use direct CSS selectors only for app-specific state that is not exposed through accessible names.

## Audit Depth

- **Smoke audit**: model selector works, default node highlights, 3-5 representative top-level and child nodes pass.
- **Targeted audit**: every node connected to the user's changed model or changed code lines.
- **Full audit**: every model, every eager node, representative generated children for repeated blocks/heads, and both PyTorch/JAX language views when requested.

When scope is ambiguous, run a targeted audit for the named model or changed files. If there is no named model, audit the model changed in the current git diff.
