# EmbeddingGemma 2 architecture example

Open `/models/embeddinggemma2`. The architecture tree exposes the 24-block text backbone and expandable 16-block vision and 12-block audio branches. Selecting a component highlights its implementation independently in PyTorch and JAX. The source pane presents Google's official model card; this release does not use another model's paper PDF.

The canonical sources are `app/model-notebooks/embeddinggemma2.py` and `app/model-notebooks/embeddinggemma2_jax.py`. Regenerate the UI previews and runnable Colab notebooks with `pnpm generate:model-artifacts`.

## Fidelity and boundaries

Default configuration values match the [released checkpoint config](https://huggingface.co/google/embeddinggemma-2/blob/main/config.json). The implementations expose:

- Scaled token embeddings, projection-only per-layer inputs, 24 bidirectional blocks, and a five-local/one-global attention pattern. Local layers use four query heads, two KV heads, and 256-wide heads; global layers use one KV head and 512-wide heads.
- Q/K RMS normalization, unscaled V normalization, unit attention scaling, layer-specific rotary bases, pre/post residual normalization, GELU gated feed-forward layers, gated per-layer inputs, and layer scalars.
- Vision patch projection, learned x/y positions, axial rotary attention, coordinate-based spatial pooling, and modality projection into the text stream.
- Audio convolutional subsampling, half-step SiLU feed-forward layers, blocked relative attention with learned query scaling and logit soft caps, causal depthwise convolution, and modality projection.
- Padding-aware soft-token insertion, final token projection, masked mean pooling, Matryoshka truncation, and L2 normalization **after** truncation.

The checkpoint's `sliding_window=512` is a radius: local attention includes distances up to 512 in either direction. The model card describes the span as 1024 tokens. The code follows the checkpoint and released mask semantics, including the center token.

These are architecture implementations with random weights. They do not download or load the trained checkpoint, and their parameter names are not checkpoint-loader mappings. Defaults retain the released dimensions; notebook smoke tests reduce width, vocabulary, depth, and media sizes so each component can run on a CPU. Random outputs are useful for shape and dataflow inspection, not semantic retrieval.

The input boundary is processor output: token IDs with the appropriate task prefixes, flattened RGB patches in `[0, 1]` and their `(x, y)` positions (`-1, -1` for padding), and 128-bin audio features plus a validity mask. Tokenization, resampling, media decoding, patch-grid construction, and audio feature extraction are external. Media placeholder counts must equal valid soft-token counts per example. Patches can have shape `(batch, patches, patch_width)` or `(batch, media, patches, patch_width)`; each image or video frame is encoded independently before its tokens are inserted in order. `video=True` selects video placeholders for the supplied frames; a call accepts either image or video patches, together with optional audio. Use float32 or bfloat16, not float16.

PyTorch accepts `EmbeddingGemma2(TEXT_CONFIG, VISION_CONFIG, AUDIO_CONFIG)`; pass `None` for unwanted modality configs. Flax accepts the same three configs and initializes only modules exercised by `init`; include representative media inputs when initializing a multimodal model. Eager calls reject mismatched placeholders. Under JAX JIT, mismatches produce NaNs instead of silently returning a plausible embedding.

## Validation

Run the site checks with `pnpm typecheck`, `pnpm test:highlights`, `pnpm test:routes`, `pnpm test:notebooks`, and `pnpm build`.

Numerical tests require Python 3.12+ and the existing constrained JAX notebook environment, plus PyTorch:

```bash
uv venv /tmp/modelarchviz-model-check --python 3.12
uv pip install --python /tmp/modelarchviz-model-check/bin/python --exclude-newer '7 days' -r requirements-notebooks-jax.txt torch
/tmp/modelarchviz-model-check/bin/python -m unittest discover -s scripts -p test_embeddinggemma2.py -v
```

The tests transfer identical parameters into both frameworks and compare local/global/axial attention, vision pooling, audio subsampling, interleaved media, independent video frames, padding, truncation, JIT execution, and gradient propagation. Fixed reference outputs in `scripts/fixtures/embeddinggemma2-reference.json` also check the text, vision, and audio paths against the released implementation. They use deterministic sinusoidal tiny weights, without trained weights or network access during testing. The fixture records reference source hashes.

## Sources and attribution

- [Google DeepMind model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) and [release announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/).
- Hugging Face Transformers, revision `d4f69a869d66eeba7af0cc19a585552b14ddd6f2`: [EmbeddingGemma 2 implementation](https://github.com/huggingface/transformers/blob/d4f69a869d66eeba7af0cc19a585552b14ddd6f2/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py) and revision `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0` for the [Gemma 4 modality encoders](https://github.com/huggingface/transformers/blob/e0eb2810ac1b2caf6d082f247d8d37162ecf67a0/src/transformers/models/gemma4/modeling_gemma4.py).

The reference implementations are Copyright 2026 the HuggingFace Team, licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). These teaching implementations reorganize those architectures into explicit tensor operations and add a Flax translation, architecture anchors, compact smoke tests, and a pooling/normalization wrapper.
