from unittest.mock import Mock

import torch

import gpt_oss.torch.utils as torch_utils


def test_init_distributed_uses_local_rank_for_cuda_device(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")

    init_process_group = Mock()
    set_device = Mock()
    all_reduce = Mock()
    synchronize = Mock()
    suppress_output = Mock()
    warmup_tensor = object()

    monkeypatch.setattr(torch_utils.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(torch_utils.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch_utils.torch, "ones", Mock(return_value=warmup_tensor))
    monkeypatch.setattr(torch_utils.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(torch_utils.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch_utils, "suppress_output", suppress_output)

    device = torch_utils.init_distributed()

    assert device == torch.device("cuda:1")
    init_process_group.assert_called_once_with(
        backend="nccl",
        init_method="env://",
        world_size=8,
        rank=5,
    )
    set_device.assert_called_once_with(1)
    torch_utils.torch.ones.assert_called_once_with(1, device=torch.device("cuda:1"))
    all_reduce.assert_called_once_with(warmup_tensor)
    synchronize.assert_called_once_with(torch.device("cuda:1"))
    suppress_output.assert_called_once_with(5)
