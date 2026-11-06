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


def test_rank_zero_executes_once_and_broadcasts_serialized_messages(monkeypatch):
    expected = Message.from_role_and_content(Role.TOOL, "leader result")
    monkeypatch.setattr(chat.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(chat.torch.distributed, "get_rank", lambda: 0)
    calls = []
    payloads = []

    def run_tool():
        calls.append(True)
        return [expected]

    def broadcast(payload_list, src):
        assert src == 0
        payloads.append(payload_list[0])

    monkeypatch.setattr(chat.torch.distributed, "broadcast_object_list", broadcast)
    result = chat._run_tool_on_rank_zero(run_tool)
    assert calls == [True]
    assert payloads == [{"messages": [expected.to_dict()], "error": None}]
    assert [message.to_dict() for message in result] == [expected.to_dict()]


def test_rank_zero_broadcasts_its_failure_before_raising(monkeypatch):
    monkeypatch.setattr(chat.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(chat.torch.distributed, "get_rank", lambda: 0)
    payloads = []
    monkeypatch.setattr(
        chat.torch.distributed,
        "broadcast_object_list",
        lambda data, src: payloads.append(data[0]),
    )

    def fail():
        raise ValueError("leader failed")

    with pytest.raises(RuntimeError, match="leader failed"):
        chat._run_tool_on_rank_zero(fail)
    assert payloads == [{"messages": [], "error": "ValueError: leader failed"}]
