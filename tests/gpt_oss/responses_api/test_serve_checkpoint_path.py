from pathlib import Path

from gpt_oss.responses_api.serve import resolve_checkpoint_path


def test_resolve_checkpoint_path_expands_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_checkpoint_path("~/model") == str(tmp_path / "model")


def test_resolve_checkpoint_path_preserves_absolute_path(tmp_path: Path) -> None:
    checkpoint = str(tmp_path / "model")

    assert resolve_checkpoint_path(checkpoint) == checkpoint
