"""Numerical PyTorch/Flax parity and embedding regressions (CPU, random tiny weights).

Run with Python 3.12+ and the constrained notebook environment plus torch:
    python -m unittest discover -s scripts -p test_embeddinggemma2.py
"""
import json
import re
import types
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import traverse_util

ROOT = Path(__file__).resolve().parents[1]


def load_source(name):
    path = ROOT / 'app' / 'model-notebooks' / name
    module = types.ModuleType(name)
    sections = re.split(r'^# %%([^\n]*)\n', path.read_text(), flags=re.M)
    code = []
    for index in range(1, len(sections), 2):
        if '[' not in sections[index]:
            code.append(sections[index + 1])
    exec(compile('\n'.join(code), str(path), 'exec'), module.__dict__)
    return module


PT = load_source('embeddinggemma2.py')
JX = load_source('embeddinggemma2_jax.py')
TEXT = dict(PT.TEXT_CONFIG, vocab=64, width=16, layers=6, hidden=32, heads=4,
            local_kv=2, local_dim=4, global_dim=8, radius=2, ple_width=8, output=12,
            image_token=61, audio_token=62, video_token=63)
VISION = dict(PT.VISION_CONFIG, width=16, layers=1, hidden=32, heads=4,
              head_dim=4, patch=2, positions=16, pool=2)
AUDIO = dict(PT.AUDIO_CONFIG, width=16, layers=1, heads=4, channels=(4, 2),
             bins=8, chunk=4, left=5, output=24)


def transfer(torch_model, flax_variables):
    """Copy actual weights, convolution layouts, constants, and clipping bounds."""
    state = torch_model.state_dict()
    result = {}
    for path, template in traverse_util.flatten_dict(flax_variables).items():
        collection, *parts, leaf = path
        parts = [re.sub(r'^(layers|convs|norms)_(\d+)$', r'\1.\2', part) for part in parts]
        torch_leaf = {'kernel': 'weight', 'embedding': 'weight', 'scale': 'weight'}.get(leaf, leaf)
        key = '.'.join(parts + [torch_leaf])
        array = state[key].detach().cpu().numpy()
        if leaf == 'kernel':
            if array.ndim == 2:
                array = array.T
            elif array.ndim == 3:
                array = array.transpose(2, 1, 0)
            elif array.ndim == 4:
                array = array.transpose(2, 3, 1, 0)
        if array.shape != template.shape:
            raise AssertionError(f'{key}: {array.shape} != {template.shape}')
        result[path] = jnp.asarray(array)
    return traverse_util.unflatten_dict(result)


def as_jax(inputs):
    return jax.tree.map(lambda value: jnp.asarray(value.detach().numpy()) if isinstance(value, torch.Tensor) else value, inputs)


class EmbeddingGemmaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def compare(self, torch_model, flax_model, inputs):
        jax_inputs = as_jax(inputs)
        variables = flax_model.init(jax.random.key(7), *jax_inputs)
        variables = transfer(torch_model, variables)
        with torch.no_grad():
            actual = torch_model(*inputs)
        expected = flax_model.apply(variables, *jax_inputs)
        torch_leaves = jax.tree.leaves(actual)
        jax_leaves = jax.tree.leaves(expected)
        self.assertEqual(len(torch_leaves), len(jax_leaves))
        for torch_value, jax_value in zip(torch_leaves, jax_leaves):
            np.testing.assert_allclose(torch_value.numpy(), np.asarray(jax_value), rtol=3e-4, atol=3e-5)
        return variables

    def test_released_reference_outputs(self):
        fixture = json.loads((ROOT / 'scripts/fixtures/embeddinggemma2-reference.json').read_text())
        def deterministic(model):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    offset = sum(name.encode()) % 97
                    values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                    values = torch.sin(values * 0.17 + offset) * 0.05
                    if name.endswith('weight') and parameter.ndim == 1:
                        values = 1 + values
                    parameter.copy_(values)
            return model
        ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
        text = deterministic(PT.TextBackbone(TEXT))
        text_output = text(text.embed(ids), ids != 0)
        patches = torch.linspace(0, 1, 2 * 16 * 12).reshape(2, 16, 12)
        positions = torch.tensor([[[x, y] for y in range(4) for x in range(4)]]).expand(2, 16, 2).clone()
        positions[1, 8:] = -1
        vision = deterministic(PT.VisionEncoder(VISION))
        vision_output, _ = vision(patches, positions)
        audio = deterministic(PT.AudioEncoder(dict(AUDIO, channels=(8, 2))))
        features = torch.linspace(-1, 1, 2 * 25 * 8).reshape(2, 25, 8)
        valid = torch.tensor([[True] * 25, [True] * 17 + [False] * 8])
        audio_output, _ = audio(features, valid)
        for label, output in [('reference text', text_output), ('reference vision', vision_output), ('reference audio', audio_output)]:
            with self.subTest(component=label):
                np.testing.assert_allclose(output.detach(), fixture[label], rtol=3e-4, atol=3e-5)

    def test_attention_local_global_and_axial(self):
        inputs = (torch.randn(2, 7, 16), torch.tensor([[True] * 7, [True] * 4 + [False] * 3]), torch.arange(7)[None].expand(2, 7))
        for kv, dim, radius in [(2, 4, 2), (1, 8, None)]:
            with self.subTest(kv=kv):
                self.compare(PT.EncoderAttention(16, 4, kv, dim, radius=radius),
                             JX.EncoderAttention(16, 4, kv, dim, radius=radius), inputs)
        positions = torch.stack((torch.arange(7) % 4, torch.arange(7) // 4), dim=-1)[None].expand(2, 7, 2)
        self.compare(PT.EncoderAttention(16, 4, 4, 4, 100.0, axial=True),
                     JX.EncoderAttention(16, 4, 4, 4, 100.0, axial=True), (*inputs[:2], positions))

    def test_vision_spatial_pooling_and_padding(self):
        patches = torch.rand(2, 16, 12)
        positions = torch.tensor([[[x, y] for y in range(4) for x in range(4)]]).expand(2, 16, 2).clone()
        positions[1, 8:] = -1
        self.compare(PT.VisionEncoder(VISION), JX.VisionEncoder(VISION), (patches, positions))
        # Permuting patch order together with positions must preserve spatial pooling.
        model = PT.VisionEncoder(VISION)
        permutation = torch.randperm(16)
        first, valid = model(patches, positions)
        second, other_valid = model(patches[:, permutation], positions[:, permutation])
        np.testing.assert_allclose(first.detach(), second.detach(), rtol=2e-5, atol=1e-5)
        self.assertTrue(torch.equal(valid, other_valid))

    def test_audio_subsampling_relative_attention_and_padding(self):
        features = torch.randn(2, 25, 8)
        valid = torch.tensor([[True] * 25, [True] * 17 + [False] * 8])
        self.compare(PT.AudioEncoder(AUDIO), JX.AudioEncoder(AUDIO), (features, valid))
        self.compare(PT.AudioAttention(AUDIO), JX.AudioAttention(AUDIO),
                     (torch.randn(2, 7, 16), valid[:, ::4]))

    def test_text_embeddings_padding_and_matryoshka(self):
        model = PT.EmbeddingGemma2(TEXT, vision=None, audio=None)
        flax = JX.EmbeddingGemma2(TEXT)
        ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
        variables = self.compare(model, flax, (ids, ids != 0))
        first = model(ids[:1], ids[:1] != 0)
        unpadded = model(ids[:1, :4], torch.ones(1, 4, dtype=torch.bool))
        np.testing.assert_allclose(first.detach(), unpadded.detach(), atol=1e-5)
        shortened = model(ids, ids != 0, output_dim=6)
        expected = torch.nn.functional.normalize(model(ids, ids != 0)[:, :6], dim=-1)
        np.testing.assert_allclose(shortened.detach(), expected.detach(), atol=1e-6)
        compiled = jax.jit(lambda variables, tokens: flax.apply(variables, tokens, tokens != 0, output_dim=6))
        np.testing.assert_allclose(shortened.detach(), compiled(variables, as_jax(ids)), rtol=3e-4, atol=3e-5)
        gradients = jax.grad(lambda params: jnp.sum(flax.apply(
            dict(variables, params=params), as_jax(ids), as_jax(ids != 0))[:, 0]))(variables['params'])
        self.assertGreater(float(jnp.linalg.norm(gradients['text']['ple']['kernel'])), 0)
        self.assertEqual(len(model.text.layers), 6)
        self.assertEqual(model.text.layers[4].attention.kv_heads, 2)
        self.assertEqual(model.text.layers[5].attention.kv_heads, 1)
        self.assertIsNone(model.text.layers[5].attention.radius)
        # The embedding loss must propagate through attention and PLE.
        model(ids, ids != 0)[:, 0].sum().backward()
        self.assertGreater(float(model.text.ple.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.text.layers[0].attention.q_proj.weight.grad.abs().sum()), 0)

    def test_soft_token_packing_per_example_and_jit_validation(self):
        ids = torch.tensor([[1, 61, 2, 61], [61, 2, 3, 4]])
        embeddings = torch.ones(2, 4, 3)
        soft = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        valid = torch.tensor([[False, True, False, True], [False, False, True, False]])
        actual = PT.merge_soft_tokens(embeddings, ids, 61, soft, valid)
        expected = embeddings.clone()
        expected[0, 1], expected[0, 3], expected[1, 0] = soft[0, 1], soft[0, 3], soft[1, 2]
        np.testing.assert_array_equal(actual, expected)
        compiled = jax.jit(lambda tokens: JX.merge_soft_tokens(as_jax(embeddings), tokens, 61, as_jax(soft), as_jax(valid)))
        np.testing.assert_array_equal(compiled(as_jax(ids)), expected)
        invalid = ids.clone()
        invalid[1, 1] = 61
        self.assertTrue(bool(jnp.isnan(compiled(as_jax(invalid))).all()))

    def test_interleaved_vision_audio_and_video(self):
        model, flax = PT.EmbeddingGemma2(TEXT, VISION, AUDIO), JX.EmbeddingGemma2(TEXT, VISION, AUDIO)
        ids = torch.tensor([[1, 61, 61, 2, 62, 62, 62, 3]])
        patches = torch.rand(1, 8, 12)
        positions = torch.tensor([[[x, y] for y in range(2) for x in range(4)]])
        features = torch.randn(1, 9, 8)
        feature_valid = torch.ones(1, 9, dtype=torch.bool)
        inputs = (ids, ids != 0, patches, positions, features, feature_valid)
        variables = self.compare(model, flax, inputs)
        video_ids = ids.clone()
        video_ids[video_ids == 61] = 63
        with torch.no_grad():
            image = model(*inputs)
            video = model(video_ids, ids != 0, patches, positions, features, feature_valid, video=True)
        # Placeholder embeddings are fully replaced by vision features.
        np.testing.assert_allclose(image, video, atol=1e-6)
        compiled = jax.jit(lambda variables, *args: flax.apply(variables, *args))
        np.testing.assert_allclose(image, compiled(variables, *as_jax(inputs)), rtol=3e-4, atol=3e-5)
        frames = torch.stack((patches, patches * 0.5), dim=1)
        frame_positions = positions[:, None].expand(-1, 2, -1, -1)
        frame_ids = torch.tensor([[1, 63, 63, 2, 63, 63, 3]])
        video_inputs = (frame_ids, frame_ids != 0, frames, frame_positions, None, None, None, True)
        self.compare(model, flax, video_inputs)
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            model(torch.tensor([[1, 61, 2]]), torch.ones(1, 3, dtype=torch.bool), patches, positions)
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            JX.merge_soft_tokens(jnp.ones((1, 3, 16)), jnp.array([[61, 61, 1]]), 61, jnp.ones((1, 1, 16)), jnp.ones((1, 1), dtype=bool))


if __name__ == '__main__':
    unittest.main()
