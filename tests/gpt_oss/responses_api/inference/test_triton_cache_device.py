import contextlib
import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def test_get_infer_next_token_places_all_caches_on_model_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    created_devices = []

    class FakeCache:
        def __init__(self, batch_size, n_ctx, n_kv_heads, d_head=64, device=None):
            created_devices.append(device)

    fake_model_module = ModuleType("gpt_oss.triton.model")
    fake_model_module.Cache = FakeCache
    fake_model_module.ModelConfig = object
    fake_model_module.Transformer = object
    monkeypatch.setitem(sys.modules, "gpt_oss.triton.model", fake_model_module)
    sys.modules.pop("gpt_oss.responses_api.inference.triton", None)

    triton_backend = importlib.import_module("gpt_oss.responses_api.inference.triton")

    tensor = MagicMock()
    monkeypatch.setattr(triton_backend.torch, "zeros", MagicMock(return_value=tensor))
    monkeypatch.setattr(triton_backend.torch.cuda, "CUDAGraph", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        triton_backend.torch.cuda,
        "graph",
        lambda graph: contextlib.nullcontext(),
    )

    model = MagicMock()
    model.config.num_key_value_heads = 8
    model.block = [object(), object(), object()]
    model.return_value = [MagicMock()]
    device = torch.device("cuda:1")

    triton_backend.get_infer_next_token(model, device)

    assert created_devices == [device, device, device]
