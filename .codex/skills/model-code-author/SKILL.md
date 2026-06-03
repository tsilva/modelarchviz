---
name: model-code-author
description: Author, edit, review, and normalize neural network model implementation files and model-code snippets. Use when creating or changing Python, JAX/Flax, or PyTorch model files such as mlp.py, lstm.py, transformer.py, *_jax.py, canonical notebook sources under app/model-notebooks, or generated/displayed model-code snippets that should be readable, tensor-shape aware, and consistently commented.
---

# Model Code Author

Use this skill to author model implementation code for architecture visualization, teaching, and inspection. Preserve behavior first; make the tensor dataflow explicit second.

## Workflow

1. Identify model files or snippets: PyTorch, JAX/Flax, or similar neural-network code.
2. Read surrounding metadata that may depend on line numbers, such as `codeLines`, docs, snapshots, or tests.
3. Edit `app/model-notebooks/*.py` as the source of truth. Do not hand-edit generated model-code or notebook artifacts.
4. Rewrite compound expressions into named steps where practical.
5. Add concise block-level comments for logical tensor operations.
6. Validate syntax/build checks and update any line mappings affected by inserted or moved lines.
7. Regenerate generated artifacts when canonical notebook sources change.

## Formatting Rules

- Split indexing, nested calls, arithmetic, assignments, and returns into separate named steps where practical.
- Prefer names that describe tensor roles: `batch_size`, `hidden_shape`, `current_input`, `gate_pre`, `state_trace`, `logits`, `outputs`.
- Prefer low-level implementations for the model's core mechanism. For example, implement attention with explicit Q/K/V projections, score scaling, masking, softmax, weighted value mixing, head merging, and output projection instead of using high-level attention modules.
- Avoid chaining shape-changing operations such as `reshape(...).transpose(...)`; assign each step.
- Avoid returning a nested call directly; assign `logits`, `out`, `outputs`, or another descriptive result first.
- Add one standalone comment before each meaningful logical block, especially blocks that transform tensor shape, compute a gate, form attention scores, apply a residual, or pack outputs.
- By default, add inline shape comments to tensor-producing and shape-transforming lines using `# input_shape -> output_shape`.
- For tensor creation with no tensor input, use `# -> output_shape`; for scalar shape metadata, use `# tensor_shape -> scalar`; for unchanged tensors, repeat the shape on both sides.
- Keep inline shape comments concise so the line remains readable. Prefer symbolic names such as `(batch, steps, d_model)` over concrete dimensions unless the example intentionally fixes the batch or sequence length.
- Do not comment every non-tensor line. Use inline shape comments for tensor dataflow and one block comment for several consecutive code lines that form one concept.
- Put an empty line before each standalone line comment.
- Do not force an empty line after a standalone line comment.
- Use one blank line between adjacent top-level demo blocks or comment-led examples.
- Reserve two blank lines for top-level class and function definitions.
- Do not add an empty line between a method signature and the first explanatory comment inside that method.
- Do not add comments above imports.
- Do not add comments above `super().__init__()`.
- Represent tensor shapes in comments with parentheses, for example `(batch, steps, features)`, not square brackets.
- When parameter comments are useful, use multiline method signatures and put parameter explanations inline on parameter lines.

## Comment Pattern

Use comments that describe the block's tensor purpose, not the mechanics of a single line.

Good:

```python
    def forward(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        batch_size = x.size(0)  # (batch, input_size) -> scalar
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)

        # Run the shared cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        step_count = x.size(1)
        for t in range(step_count):
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            h = self.cell(current_input, h)  # (batch, input_size), (batch, hidden_size) -> (batch, hidden_size)
            states.append(h)
```

Avoid:

```python
    # Gets the batch size.
    batch_size = x.size(0)
    # Creates a tuple.
    hidden_shape = (batch_size, self.hidden_size)
```

## Common Rewrites

Nested activation:

```python
conv1 = self.conv1(x)
x = torch.tanh(conv1)
```

Gate math:

```python
x_i = self.x_i(x)
h_i = self.h_i(h)
i_pre = x_i + h_i
i = torch.sigmoid(i_pre)
```

Packed output:

```python
state_trace = torch.stack(states, dim=1)
outputs = (logits, state_trace)
return outputs
```

Attention score calculation:

```python
key_transpose = k.transpose(-2, -1)
scores = q @ key_transpose
scale = k.size(-1) ** -0.5
attn_scores = scores * scale
```

## Validation

- Run syntax checks for Python files when imports do not require unavailable ML dependencies, for example `python3 -m py_compile path/to/files.py`.
- Run repo checks such as typecheck/build when the files are displayed or bundled by an app.
- If comments or rewrites change displayed snippet line numbers, update line-highlight metadata in the same change.
