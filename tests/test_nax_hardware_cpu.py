"""JIT capability must not depend on stock MLX's compiled kernel inventory."""
import ast
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("extension", [False, True])
@pytest.mark.parametrize("hardware", [False, True])
@pytest.mark.parametrize("stock", [False, True])
def test_hardware_and_stock_capabilities_are_distinct(extension, hardware, stock, monkeypatch):
    fast = extract("omlx/custom_kernels/quant_matmul/fast.py",
                   {"is_nax_hardware_available", "is_nax_available"}, {
        "_EXT_HAS_NAX": extension,
        "_ext": SimpleNamespace(is_nax_available=lambda: hardware),
        "_nax_available_fallback": lambda: hardware,
        "_stock_mlx_has_nax": lambda: stock,
        "_nax_available_cache": None,
    })
    assert fast["is_nax_hardware_available"]() is hardware
    assert fast["is_nax_available"]() is (hardware and stock)
    shim = ModuleType("omlx.custom_kernels.nax")
    shim.is_nax_hardware_available = fast["is_nax_hardware_available"]
    monkeypatch.setitem(sys.modules, "omlx.custom_kernels.nax", shim)
    attention = extract("omlx/patches/deepseek_v4/sparse_attention_nax.py",
                        {"nax_sparse_attention_available"}, {"lru_cache": lru_cache})
    assert attention["nax_sparse_attention_available"]() is hardware


def test_attention_probe_failure_falls_back(monkeypatch):
    shim = ModuleType("omlx.custom_kernels.nax")
    def unavailable():
        raise RuntimeError("GPU metadata unavailable")
    shim.is_nax_hardware_available = unavailable
    monkeypatch.setitem(sys.modules, "omlx.custom_kernels.nax", shim)
    attention = extract("omlx/patches/deepseek_v4/sparse_attention_nax.py",
                        {"nax_sparse_attention_available"}, {"lru_cache": lru_cache})
    assert attention["nax_sparse_attention_available"]() is False
