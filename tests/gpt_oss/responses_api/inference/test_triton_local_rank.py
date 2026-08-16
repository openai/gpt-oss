import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def test_load_model_uses_local_rank_for_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")

    loaded = {}
    fake_model = object()

    class FakeTransformer:
        @staticmethod
        def from_checkpoint(checkpoint, device):
            loaded["checkpoint"] = checkpoint
            loaded["device"] = device
            return fake_model

    fake_model_module = ModuleType("gpt_oss.triton.model")
    fake_model_module.Cache = object
    fake_model_module.ModelConfig = object
    fake_model_module.Transformer = FakeTransformer
    monkeypatch.setitem(sys.modules, "gpt_oss.triton.model", fake_model_module)
    sys.modules.pop("gpt_oss.responses_api.inference.triton", None)

    triton_backend = importlib.import_module("gpt_oss.responses_api.inference.triton")
    set_device = MagicMock()
    monkeypatch.setattr(triton_backend.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(triton_backend.torch, "set_grad_enabled", MagicMock())

    model, device = triton_backend.load_model("checkpoint")

    assert model is fake_model
    assert device == torch.device("cuda:1")
    assert loaded == {"checkpoint": "checkpoint", "device": torch.device("cuda:1")}
    assert triton_backend.rank == 5
    assert triton_backend.local_rank == 1
    set_device.assert_called_once_with(1)
