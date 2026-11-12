from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from desktop_detector.backend import D1Backend, detect_gpu_device, verify_model_directory
from desktop_detector.core import MODEL_ID


class D1AdapterTests(unittest.TestCase):
    def test_binary_localizes_only_after_positive(self):
        class FakeTorch:
            @staticmethod
            def inference_mode(): return nullcontext()
        class FakeModel:
            def __init__(self): self.calls = []; self.score = .1
            def system_one(self, state, questions, media):
                self.calls.append(questions)
                if "decision" in questions:
                    return {"answers": {"decision": {"noul": self.score}}}
                return {"answers": {"objects": {"boxes": [{"box": {"left": .1, "top": .2, "right": .8, "bottom": .9}}]}}}
        adapter = D1Backend(); adapter.model = FakeModel(); adapter.torch = FakeTorch(); adapter.device = "cpu"
        image = Image.new("RGB", (40, 30))
        self.assertEqual(adapter.predict_binary(image, "fish?", .2, "fish", 4), (.1, []))
        self.assertEqual(len(adapter.model.calls), 1)
        adapter.model.score = .3
        score, boxes = adapter.predict_binary(image, "fish?", .2, "fish", 4)
        self.assertEqual(score, .3)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(adapter.model.calls), 3)
    def test_choice_uses_highest_probability(self):
        class FakeTorch:
            @staticmethod
            def inference_mode(): return nullcontext()
        class FakeModel:
            def system_one(self, state, questions, media):
                assert questions["category"]["criteria"]["c0"] == "魚がある"
                return {"answers": {"category": {"choice": "none", "probabilities": {"c0": .8, "none": .2}}}}
        adapter = D1Backend(); adapter.model = FakeModel(); adapter.torch = FakeTorch(); adapter.device = "cpu"
        self.assertEqual(adapter.predict_choice(Image.new("RGB", (20, 20)), ("魚がある",))[:2], ("魚がある", .8))
    def test_calls_system_one_not_generate(self):
        class FakeTorch:
            @staticmethod
            def inference_mode(): return nullcontext()
        class FakeModel:
            called = None
            def system_one(self, state, questions, media):
                self.called = (state, questions, media)
                return {"answers": {"objects": {"boxes": [{"box": {"left": .1, "top": .2, "right": .8, "bottom": .9}, "confidence": .8}]}}}
        adapter = D1Backend(); adapter.model = FakeModel(); adapter.torch = FakeTorch(); adapter.device = "cpu"
        ds = adapter.predict(Image.new("RGB", (40, 30)), "Every fish.", 4)
        state, questions, media = adapter.model.called
        self.assertIsNone(state)
        self.assertEqual(questions["objects"]["type"], "boxes")
        self.assertEqual(questions["objects"]["max"], 4)
        self.assertEqual(media[0].size, (40, 30))
        self.assertEqual(len(ds), 1)
    def test_gpu_detection_order_and_xpu(self):
        from types import SimpleNamespace as NS
        def fake(cuda=False, xpu=False, mps=False):
            return NS(cuda=NS(is_available=lambda: cuda), xpu=NS(is_available=lambda: xpu),
                      backends=NS(mps=NS(is_available=lambda: mps)))
        self.assertEqual(detect_gpu_device(fake(xpu=True)), "xpu")
        self.assertEqual(detect_gpu_device(fake(cuda=True, xpu=True)), "cuda")
        self.assertEqual(detect_gpu_device(fake(mps=True)), "mps")
        self.assertIsNone(detect_gpu_device(fake()))
        no_xpu = fake(); del no_xpu.xpu  # a CPU/CUDA build of torch has no usable torch.xpu
        self.assertIsNone(detect_gpu_device(no_xpu))
    def test_silent_cpu_execution_is_an_error(self):
        from types import SimpleNamespace as NS
        class FakeTorch:
            @staticmethod
            def inference_mode(): return nullcontext()
        class FakeModel:
            def __init__(self): self.engine = NS(device=NS(type="cpu"))
            def system_one(self, state, questions, media):
                return {"answers": {"objects": {"boxes": []}}}
        adapter = D1Backend(); adapter.model = FakeModel(); adapter.torch = FakeTorch(); adapter.device = "xpu"
        adapter.synchronize = lambda: None
        with self.assertRaises(RuntimeError): adapter.predict(Image.new("RGB", (20, 20)), "fish")
        adapter.model.engine.device.type = "xpu"
        self.assertEqual(adapter.predict(Image.new("RGB", (20, 20)), "fish"), [])
    def test_npu_padding_is_fixed_and_bounded(self):
        import torch
        from desktop_detector.npu_backend import SEQ_LEN, pad_inputs, fixed_picture, FIXED_SIZE
        ids = torch.arange(5)[None]
        p_ids, mask, state, question = pad_inputs(ids, torch.ones_like(ids), torch.tensor([3]), torch.full((1, 5), 1))
        self.assertEqual(tuple(p_ids.shape), (1, SEQ_LEN))
        self.assertEqual(int(mask.sum()), 5)
        self.assertEqual(int(question[0, 5]), -1)
        with self.assertRaises(RuntimeError):
            pad_inputs(torch.zeros(1, SEQ_LEN + 1, dtype=torch.long), torch.ones(1, SEQ_LEN + 1), torch.tensor([3]),
                       torch.ones(1, SEQ_LEN + 1, dtype=torch.long))
        self.assertEqual(fixed_picture(Image.new("RGB", (100, 300))).size, FIXED_SIZE)
    def test_npu_export_patches_match_originals(self):
        import torch
        import torch.nn.functional as F
        from desktop_detector.npu_backend import sdpa_without_gqa, static_masked_scatter
        torch.manual_seed(0)
        q, k, v = torch.randn(1, 8, 6, 4), torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4)
        mask = torch.rand(1, 1, 6, 6) > .3
        mask[..., 0] = True
        original = F.scaled_dot_product_attention
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        with sdpa_without_gqa():
            patched = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        self.assertIs(F.scaled_dot_product_attention, original)  # restored after the block
        torch.testing.assert_close(patched, expected)
        target = torch.randn(1, 7, 3)
        token_mask = torch.tensor([[0, 1, 1, 0, 1, 0, 0]], dtype=torch.bool)[..., None].expand(1, 7, 3)
        source = torch.randn(3, 3)
        expected = target.masked_scatter(token_mask, source)
        with static_masked_scatter():
            patched = target.masked_scatter(token_mask, source)
        torch.testing.assert_close(patched, expected, rtol=0, atol=0)
    def test_unloaded_model_error(self):
        with self.assertRaises(RuntimeError): D1Backend().predict(Image.new("RGB", (20, 20)), "fish")
    def test_missing_model_fails_before_imports(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(RuntimeError): verify_model_directory(Path(path))
    def test_local_code_tamper_detection(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"fake test fixture, not weights")
            file = root / "modeling.py"; file.write_text("# fixture")
            lock = {"model_id": MODEL_ID, "revision": "fixture", "code_sha256": {"modeling.py": hashlib.sha256(file.read_bytes()).hexdigest()}}
            (root / "model.lock.json").write_text(json.dumps(lock))
            self.assertEqual(verify_model_directory(root)["revision"], "fixture")
            file.write_text("# changed")
            with self.assertRaises(RuntimeError): verify_model_directory(root)


if __name__ == "__main__":
    unittest.main()
