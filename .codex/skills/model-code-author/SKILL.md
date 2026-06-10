---
name: model-code-author
description: "Author/edit/review/normalize neural-network model files/snippets: Python, PyTorch, JAX/Flax, app/model-notebooks sources, generated/displayed model-code, readable tensor-shape-aware teaching code."
---

# Model Code Author

Preserve behavior; expose tensor dataflow.

## Workflow

1. Identify model files/snippets and dependent metadata: `codeLines`, docs, snapshots, tests.
2. Edit only `app/model-notebooks/*.py` as source; never hand-edit `app/generated/model-code/*.py` or notebook artifacts.
3. Cells: imports first; `# %%` for implementation; one top-level model/component class per cell, never two.
4. After every top-level model/component class, immediately add small `# %% [notebook-only]` smoke test: construct, run representative input, show expected input/output shape, end with `print(...)`.
5. Component smoke tests/shape inspection/non-web examples: `# %% [notebook-only]`. Training examples meant for the web code panel: separate normal `# %%` cell after the notebook-only smoke test.
6. Split compound tensor expressions into named steps; add concise block comments; validate, update moved line mappings, regenerate artifacts.

## Rules

- Split indexing, nested calls, arithmetic, assignments, shape changes, returns where practical; avoid chained shape ops and nested-call returns.
- Prefer role names: `batch_size`, `hidden_shape`, `current_input`, `gate_pre`, `state_trace`, `logits`, `outputs`.
- Prefer low-level mechanisms over high-level modules: e.g. explicit Q/K/V, scale, mask, softmax, value mix, head merge, output projection.
- Comments: one standalone block comment before meaningful tensor blocks only: shape transform, gate, attention scores, residual, pack outputs.
- Shape comments: tensor change `# input_shape -> output_shape`; creation `# -> output_shape`; scalar metadata `# tensor_shape -> scalar`; unchanged one shape. Use symbolic parenthesized shapes, not brackets.
- Spacing: blank before standalone comments, not forced after; one blank between adjacent top-level demo/comment-led examples; two before top-level classes/functions; none between method signature and first explanatory comment.
- No comments above imports or `super().__init__()`. If parameter comments help, use multiline signatures with inline parameter comments.
- Multiple classes: repeat `# %%` class, immediate `# %% [notebook-only]` smoke test.
- Generated web previews stay implementation-focused: no notebook-only examples/smoke tests/shape inspection. Include intended training examples in normal `# %%` cells; do not hide them in notebook-only cells.
- Web-visible snippets use normal `# %%`; notebook learning/tests/examples use `# %% [notebook-only]`.
- Keep the final model smoke test and training example split: notebook-only smoke test with `example_*` names and `print(...)`, then normal web-visible training cell with `model`/`params`/`outputs`/`logits` as needed.
- No separate generated example files unless explicitly asked.
- No notebook-only examples/smoke tests/training snippets as architecture side-panel nodes unless explicitly asked.
- No active code selections or highlighted lines on page load; use non-selected/non-highlighted scroll/focus hints only.
- Notebook-only smoke tests use `example_model`, `example_params`, `example_outputs`, `example_logits`, `example_states`; when normal cells use `model`, `params`, `outputs`, `logits`, smoke tests must use distinct `example_*`.
- Every notebook-only smoke-test cell ends with `print(...)` for relevant outputs/shapes.

## Validation

- Syntax-check when ML deps are available/unneeded: `python3 -m py_compile path/to/file.py`.
- Run repo checks (typecheck/build) when app bundles/displays files.
- After regeneration inspect generated web code: intended training example is present; no notebook-only comments/tests/examples; no duplicate top-level `model = ...` except intentional implementation; only intentional normal-cell examples/training.
- Inspect `.ipynb` starts: every top-level class has own cell plus immediate expected notebook-only smoke-test cell; no duplicate training-style `model = ModelClass(...)` unless intentional and clearly named.
- Confirm smoke tests end with `print(...)` and use `example_*` when normal cells use `model`/`params`/`outputs`/`logits`.
- If web UI should load with no side-panel node selected, verify zero selected architecture nodes and zero highlighted code lines.
- Update line-highlight metadata when displayed snippet line numbers move.
