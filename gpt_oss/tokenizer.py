import tiktoken
import json
import os


def get_tokenizer_():
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
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
        } | {
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer


def get_tokenizer():
    if not os.path.isfile("o200k_harmony_tokenizer.json"):
        save_tokenizer()

    return load_tokenizer()


def get_custom_tokenizer_components():
    """
    Fetches the components for the custom tokenizer from the base tokenizer.
    """
    # This part will require a download the first time it's run
    o200k_base = tiktoken.get_encoding("o200k_base")

    # Define the custom special tokens
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
    } | {
        f"<|reserved_{i}|>": i for i in range(200013, 201088)
    }

    # The mergeable_ranks are bytes, which are not directly JSON serializable.
    # We need to encode them, for example, using base64.
    mergeable_ranks_b64 = {
        k.decode('latin-1'): v for k, v in o200k_base._mergeable_ranks.items()
    }


    return {
        "name": "o200k_harmony",
        "pat_str": o200k_base._pat_str,
        "mergeable_ranks": mergeable_ranks_b64,
        "special_tokens": special_tokens,
    }


def save_tokenizer(filepath="o200k_harmony_tokenizer.json"):
    """
    Saves the custom tokenizer components to a local JSON file.
    """
    components = get_custom_tokenizer_components()
    with open(filepath, "w") as f:
        json.dump(components, f, indent=2)
    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath="o200k_harmony_tokenizer.json"):
    """
    Loads the custom tokenizer from a local file.
    """
    with open(filepath, "r") as f:
        components = json.load(f)

    # The mergeable_ranks were saved with string keys, so we need to
    # convert them back to bytes.
    mergeable_ranks = {
        k.encode('latin-1'): v for k, v in components["mergeable_ranks"].items()
    }

    return tiktoken.Encoding(
        name=components["name"],
        pat_str=components["pat_str"],
        mergeable_ranks=mergeable_ranks,
        special_tokens=components["special_tokens"],
    )


# --- Main execution ---
if __name__ == "__main__":
    # 1. First, save the tokenizer to your local disk.
    #    You only need to run this once.
    save_tokenizer()

    # 2. Now, you can load the tokenizer from the local file
    #    without needing to download anything.
    tokenizer = load_tokenizer()

    # Verify that the tokenizer works as expected
    text = "Hello world! <|endoftext|>"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Original text: {text}")
    print(f"Encoded tokens: {encoded}")
    print(f"Decoded text: {decoded}")
