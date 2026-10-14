import pytest

torch = pytest.importorskip("torch")

from gpt_oss.torch import sampling


def test_local_greedy_sampling_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(sampling.dist, "is_initialized", lambda: False)

    token = sampling.sample_next_token(torch.tensor([0.1, 0.8, 0.2]), 0.0)

    assert token == 1


def test_rank_zero_samples_and_broadcasts(monkeypatch) -> None:
    broadcasts = []
    monkeypatch.setattr(sampling.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sampling.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        sampling.torch,
        "multinomial",
        lambda probs, num_samples: torch.tensor([2], device=probs.device),
    )
    monkeypatch.setattr(
        sampling.dist,
        "broadcast",
        lambda tensor, src: broadcasts.append((int(tensor.item()), src)),
    )

    token = sampling.sample_next_token(torch.tensor([0.2, 0.3, 0.5]), 1.0)

    assert token == 2
    assert broadcasts == [(2, 0)]


def test_nonzero_rank_uses_broadcast_token_without_sampling(monkeypatch) -> None:
    monkeypatch.setattr(sampling.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sampling.dist, "get_rank", lambda: 1)

    def fail_if_sampled(*args, **kwargs):
        raise AssertionError("nonzero ranks must not sample independently")

    def receive_rank_zero_token(tensor, src):
        assert src == 0
        tensor.fill_(1)

    monkeypatch.setattr(sampling.torch, "multinomial", fail_if_sampled)
    monkeypatch.setattr(sampling.dist, "broadcast", receive_rank_zero_token)

    token = sampling.sample_next_token(torch.tensor([0.9, 0.1]), 1.0)

    assert token == 1
