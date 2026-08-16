import tiktoken


def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")
    special_tokens = {
        **o200k_base._special_tokens,
        "<|startoftext|>": 199998,
        "<|endoftext|>": 199999,
        "<|reserved_200000|>": 200000,
        "<|reserved_200001|>": 200001,
        "<|return|>": 200002,
        "<|constrain|>": 200003,
        "<|reserved_200004|>": 200004,
        "<|channel|>": 200005,
        "<|start|>": 200006,
        "<|end|>": 200007,
        "<|message|>": 200008,
        "<|reserved_200009|>": 200009,
        "<|reserved_200010|>": 200010,
        "<|reserved_200011|>": 200011,
        "<|call|>": 200012,
    }
    used_token_ids = set(special_tokens.values())
    special_tokens |= {
        f"<|reserved_{i}|>": i
        for i in range(200013, 201088)
        if i not in used_token_ids
    }

    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens=special_tokens,
    )
    return tokenizer
