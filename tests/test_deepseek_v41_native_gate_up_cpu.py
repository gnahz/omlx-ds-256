"""CPU-only admission/dispatch checks; this file does not import real MLX."""

import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class Array:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.size = len(self.shape), math.prod(self.shape)


class Projection(SimpleNamespace):
    def get(self, name):
        return getattr(self, name, None)


class NativeGateUpAdmission(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.created = []
        core = ModuleType("mlx.core")
        for dtype in ("bfloat16", "float16", "float32", "uint32", "int32", "uint16"):
            setattr(core, dtype, dtype)
        core.gpu, core.cpu = "gpu", "cpu"
        core.default_device = lambda: "gpu"

        def create(**kwargs):
            self.created.append(kwargs)
            def run(**arguments):
                self.calls.append(arguments)
                return ["gate", "up"]
            return run
        core.fast = SimpleNamespace(metal_kernel=create)
        parent = ModuleType("mlx")
        parent.core = core
        path = Path(__file__).parents[1] / "omlx/patches/deepseek_v41/native_gate_up.py"
        spec = importlib.util.spec_from_file_location("native_gate_up_cpu_stub", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"mlx": parent, "mlx.core": core}), patch.dict(
                os.environ, {"DS41_NATIVE_GATE_UP": "0"}):
            spec.loader.exec_module(self.module)
        self.core = core

    def projection(self, experts=384):
        return Projection(
            mode="affine", bits=3, group_size=128, quantize_input=True,
            weight=Array((experts, 2304, 480), "uint32"),
            scales=Array((experts, 2304, 40), "bfloat16"),
            biases=Array((experts, 2304, 40), "bfloat16"))

    def expert(self, experts=384):
        return SimpleNamespace(w1=self.projection(experts), w3=self.projection(experts))

    def test_default_off_and_no_kernel_created_by_admission(self):
        self.assertFalse(self.module.ENABLED)
        self.assertFalse(self.module.supported(self.expert(), Array((1, 1, 1, 1, 5120), "bfloat16"),
                                               Array((1, 1, 6), "int32")))
        self.assertEqual(self.created, [])

    def test_all_short_rows_and_route_dtype(self):
        self.module.ENABLED = True
        for rows in range(1, 9):
            for dtype in ("int32", "uint32"):
                self.assertTrue(self.module.supported(
                    self.expert(), Array((1, rows, 1, 1, 5120), "bfloat16"),
                    Array((1, rows, 6), dtype)))
        self.assertEqual(self.created, [])

    def test_mismatched_metadata_and_unsupported_projection(self):
        self.module.ENABLED = True
        variants = [
            ("mode", "mxfp4"), ("bits", 4), ("group_size", 64), ("quantize_input", False),
            ("weight", Array((384, 2304, 480), "uint16")),
            ("weight", Array((385, 2304, 480), "uint32")),
            ("weight", Array((0, 2304, 480), "uint32")),
            ("weight", Array((384, 2304, 479), "uint32")),
            ("weight", Array((384, 2303, 480), "uint32")),
            ("weight", Array((2304, 480), "uint32")),
            ("scales", Array((384, 2304, 40), "float32")),
            ("scales", Array((384, 2304, 39), "bfloat16")),
            ("biases", None), ("biases", Array((384, 2304, 40), "float16")),
            ("biases", Array((384, 2303, 40), "bfloat16"))]
        for side in ("w1", "w3"):
            for field, value in variants:
                expert = self.expert()
                setattr(getattr(expert, side), field, value)
                self.assertFalse(self.module.supported(expert, Array((1, 5120), "bfloat16"),
                                                       Array((1, 6), "uint32")), (side, field))
        expert = self.expert()
        expert.w3 = self.projection(383)
        self.assertFalse(self.module.supported(expert, Array((1, 5120), "bfloat16"),
                                               Array((1, 6), "uint32")))
        self.assertEqual(self.created, [])

    def test_rows_width_dtype_and_device_fallback(self):
        self.module.ENABLED = True
        for x, ids in [
            (Array((0, 5120), "bfloat16"), Array((0, 6), "uint32")),
            (Array((9, 5120), "bfloat16"), Array((9, 6), "uint32")),
            (Array((1, 5120), "float32"), Array((1, 6), "uint32")),
            (Array((1, 5120), "float16"), Array((1, 6), "uint32")),
            (Array((1, 5121), "bfloat16"), Array((1, 6), "uint32")),
            (Array((2, 5120), "bfloat16"), Array((1, 6), "uint32")),
            (Array((1, 5120), "bfloat16"), Array((1, 5), "uint32")),
            (Array((1, 5120), "bfloat16"), Array((1, 6), "uint16"))]:
            self.assertFalse(self.module.supported(self.expert(), x, ids))
        self.core.default_device = lambda: "cpu"
        self.assertFalse(self.module.supported(self.expert(), Array((1, 5120), "bfloat16"),
                                               Array((1, 6), "uint32")))
        self.assertEqual(self.created, [])

    def test_one_lazy_dispatch_exact_extents_and_input_identity(self):
        expert = self.expert()
        for rows in range(1, 9):
            x, ids = Array((rows, 1, 5120), "bfloat16"), Array((rows * 6,), "uint32")
            self.assertEqual(self.module.gate_up(x, expert.w1, expert.w3, ids, 6), ["gate", "up"])
            call = self.calls[-1]
            self.assertIs(call["inputs"][0], x)
            self.assertIs(call["inputs"][1], expert.w1.weight)
            self.assertIs(call["inputs"][4], expert.w3.weight)
            self.assertIs(call["inputs"][-1], ids)
            self.assertEqual(call["grid"], (32 * rows * 6, 1152, 1))
            self.assertEqual(call["threadgroup"], (32, 2, 1))
            self.assertEqual(call["output_shapes"], [(rows * 6, 1, 2304)] * 2)
            self.assertEqual(call["output_dtypes"], ["bfloat16"] * 2)
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(len(self.created), 1)
        self.assertTrue(self.created[0]["ensure_row_contiguous"])

    def test_low_level_bad_shapes_fail_before_kernel(self):
        expert = self.expert()
        for x, ids, topk in [
            (Array((1, 5120), "bfloat16"), Array((6,), "uint32"), 6),
            (Array((1, 1, 5120), "float32"), Array((6,), "uint32"), 6),
            (Array((9, 1, 5120), "bfloat16"), Array((54,), "uint32"), 6),
            (Array((1, 1, 5120), "bfloat16"), Array((1, 6), "uint32"), 6),
            (Array((1, 1, 5120), "bfloat16"), Array((5,), "uint32"), 6),
            (Array((1, 1, 5120), "bfloat16"), Array((6,), "int32"), 6),
            (Array((1, 1, 5120), "bfloat16"), Array((6,), "uint32"), 5)]:
            with self.assertRaises(ValueError):
                self.module.gate_up(x, expert.w1, expert.w3, ids, topk)
        self.assertEqual(self.created, [])


if __name__ == "__main__":
    unittest.main()
