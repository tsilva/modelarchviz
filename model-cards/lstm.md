# LSTM Sequence Classifier Architecture

An educational PyTorch and JAX/Flax LSTM sequence classifier example from ModelArchViz, paired with architecture nodes, code highlights, notebooks, and the 1997 source paper.

## At a Glance

| Item | Value |
|---|---|
| Task | Sequence classification architecture example |
| Dataset | Synthetic two-sequence toy example in the notebooks |
| Model | LSTM cell unrolled over 8 input steps with a final linear readout |
| Format | PyTorch source, JAX/Flax source, generated Colab notebooks, in-app architecture card |
| Input | `(batch, steps, input_size)`, shown as `(2, 8, 32)` in the sample notebook cell |
| Output | `(logits, state_trace)` where logits are `(batch, 10)` and state trace is `(batch, steps, 64)` |
| Test accuracy | Not reported; this is an architecture demonstration, not a benchmarked checkpoint |
| Uploaded checkpoint | None |

## Quick Start

Run the app locally and open the LSTM architecture page:

```bash
pnpm install
pnpm dev
```

Then open `/models/lstm` from the printed local dev-server URL.

To inspect the standalone notebook artifacts, use:

- `public/notebooks/lstm.ipynb` for the PyTorch walkthrough.
- `public/notebooks/lstm_jax.ipynb` for the JAX/Flax walkthrough.

## Validate the Example Shapes

The PyTorch notebook source includes a smoke test that constructs `LSTMSequence(input_size=32, hidden_size=64, output_size=10)` and runs a `(2, 8, 32)` sequence through it.

Expected shape output:

```text
logits shape: torch.Size([2, 10]) states shape: torch.Size([2, 8, 64])
```

The JAX/Flax notebook mirrors the same interface and expected shapes:

```text
logits shape: (2, 10) states shape: (2, 8, 64)
```

## Results

| Evaluation | Result |
|---|---|
| Shape smoke test | PyTorch and JAX examples produce logits and full hidden-state traces |
| Training metric | No validation or test metric reported |
| Benchmark status | Not benchmarked; no checkpoint selection was performed |

## Input / Output

The sequence model consumes a batch of fixed-width feature vectors over time. In the canonical PyTorch source, the sample sequence has shape `(batch=2, steps=8, input_size=32)`.

The model returns:

- `logits`: final hidden state projected from 64 features to 10 output classes.
- `state_trace`: the hidden state from every recurrent step, stacked as `(batch, steps, hidden_size)`.

## Architecture

The app visualizes the LSTM as:

- Sequence input with 8 steps and 32 features.
- Shared LSTM cell parameters for input, forget, candidate, and output gates.
- Recurrent loop over 8 steps with hidden state `h` and cell state `c`.
- Final linear readout from the last hidden state.
- Output bundle containing logits and the hidden-state trace.

The architecture card links each node to the concrete PyTorch and JAX lines that implement it.

## Training Recipe

The notebook-only training cell fits the example model for 3 SGD steps on two synthetic sequences with opposite labels:

- Optimizer: SGD.
- Learning rate: `0.1`.
- Loss: cross entropy.
- Training data: two hand-built synthetic sequences.
- Checkpoint selection: none.

This cell exists to show that the model is trainable end to end; it is not a meaningful evaluation recipe.

## Files

| File | Purpose |
|---|---|
| `app/model-notebooks/lstm.py` | Canonical PyTorch notebook-source file |
| `app/model-notebooks/lstm_jax.py` | Canonical JAX/Flax notebook-source file |
| `app/generated/model-code/lstm.py` | Generated PyTorch source displayed in the app |
| `app/generated/model-code/lstm_jax.py` | Generated JAX/Flax source displayed in the app |
| `public/notebooks/lstm.ipynb` | Generated PyTorch Colab notebook |
| `public/notebooks/lstm_jax.ipynb` | Generated JAX/Flax Colab notebook |
| `app/model-arch-viz-app.tsx` | Embedded LSTM architecture metadata and code-line mappings |
| `app/model-routes.ts` | `/models/lstm` SEO route metadata |
| `public/papers/lstm.pdf` | Local copy of the source paper PDF used by the app |

## Provenance

The LSTM architecture card is part of the ModelArchViz repository. The source-paper metadata points to Hochreiter and Schmidhuber's "Long Short-Term Memory", published in Neural Computation on November 1, 1997, with DOI link `https://doi.org/10.1162/neco.1997.9.8.1735`.

Generated artifacts are produced from the canonical notebook sources with:

```bash
pnpm generate:model-artifacts
```

## Limitations

- The example is intentionally small and educational.
- The toy training cell does not establish generalization performance.
- No validation split, test split, checkpoint, ONNX artifact, or model weights are published.
- The sample classifier uses fixed feature vectors and synthetic labels rather than a real sequence dataset.
