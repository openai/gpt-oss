from gpt_oss.tokenizer import get_tokenizer


def test_harmony_special_token_ids_are_unique() -> None:
    tokenizer = get_tokenizer()
    special_tokens = tokenizer._special_tokens

    assert len(special_tokens.values()) == len(set(special_tokens.values()))


def test_endofprompt_keeps_its_base_token_id_without_reserved_alias() -> None:
    tokenizer = get_tokenizer()

    assert tokenizer._special_tokens["<|endofprompt|>"] == 200018
    assert "<|reserved_200018|>" not in tokenizer._special_tokens
