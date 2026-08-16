import pytest
from openai_harmony import Message, Role

from gpt_oss import chat


def test_local_tool_execution_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(chat.torch.distributed, "is_initialized", lambda: False)
    expected = [Message.from_role_and_content(Role.TOOL, "local")]

    assert chat._run_tool_on_rank_zero(lambda: expected) is expected


def test_nonzero_rank_receives_rank_zero_tool_result(monkeypatch) -> None:
    expected = Message.from_role_and_content(Role.TOOL, "from rank zero")
    monkeypatch.setattr(chat.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(chat.torch.distributed, "get_rank", lambda: 1)

    def broadcast(payload_list, src):
        assert src == 0
        payload_list[0] = {"messages": [expected.to_dict()], "error": None}

    monkeypatch.setattr(chat.torch.distributed, "broadcast_object_list", broadcast)

    def fail_if_executed():
        raise AssertionError("nonzero ranks must not execute tools")

    result = chat._run_tool_on_rank_zero(fail_if_executed)

    assert [message.to_dict() for message in result] == [expected.to_dict()]


def test_rank_zero_tool_failure_is_propagated(monkeypatch) -> None:
    monkeypatch.setattr(chat.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(chat.torch.distributed, "get_rank", lambda: 1)

    def broadcast(payload_list, src):
        assert src == 0
        payload_list[0] = {"messages": [], "error": "ValueError: broken tool"}

    monkeypatch.setattr(chat.torch.distributed, "broadcast_object_list", broadcast)

    with pytest.raises(RuntimeError, match="ValueError: broken tool"):
        chat._run_tool_on_rank_zero(lambda: [])
