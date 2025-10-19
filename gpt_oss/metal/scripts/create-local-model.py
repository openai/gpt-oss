import argparse
import os
import math
import sys
import json
import itertools
import struct
from uuid import UUID

import tiktoken

import torch
from safetensors import safe_open
from tqdm import tqdm
from openai_harmony import load_harmony_encoding, HarmonyEncodingName

parser = argparse.ArgumentParser(prog='create-local-model.py', description='Convert a checkpoint directory to a local model file')
parser.add_argument('-s', '--src', metavar='DIR', type=str, required=True, help='Path to the input checkpoint directory')
parser.add_argument('-d', '--dst', metavar='FILE', type=str, required=True, help='Path to the output model file')


o200k_base = tiktoken.get_encoding("o200k_base")
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

o200k_gptoss = tiktoken.Encoding(
    name="o200k_gptoss",
    pat_str=o200k_base._pat_str,
    mergeable_ranks=o200k_base._mergeable_ranks,
    special_tokens={
        "<|reversed199998|>": 199998,  # unused
        "<|endoftext|>": 199999,
        "<|untrusted|>": 200000,
        "<|endofuntrusted|>": 200001,
        "<|return|>": 200002,
        "<|constrain|>": 200003,
        "<|reversed200004|>": 200004,  # unused
        "<|channel|>": 200005,
        "<|start|>": 200006,
        "<|end|>": 200007,
        "<|message|>": 200008,
        "<|reversed200008|>": 200008,  # unused
        "<|reversed200009|>": 200009,  # unused
        "<|reversed200010|>": 200010,  # unused
        "<|reversed200011|>": 200011,  # unused
        "<|call|>": 200012,
        "<|refusal|>": 200013,
    }
)

FILE_MAGIC = struct.pack('ccccccccccccI', b'G', b'P', b'T', b'-', b'O', b'S', b'S', b' ', b'v', b'1', b'.', b'0', 0)
SPECIAL_TOKEN_UUID = {
    '<|start|>': UUID('55a77c2f-8a01-4c54-8ac2-313bfc7e208d').bytes,
    '<|message|>': UUID('16e40431-f47f-4b22-b59b-8b278fc30a54').bytes,
    '<|end|>': UUID('fcac2f6d-4705-4f6b-b228-642accac7238').bytes,
    '<|return|>': UUID('f799ff69-1992-43c4-a3d8-d831f475dc75').bytes,
    '<|refusal|>': UUID('e15ba702-28c4-4292-ab8f-ffa434709128').bytes,
    '<|constrain|>': UUID('c0bb14c7-6022-49da-ad08-792d67e8b470').bytes,
    '<|channel|>': UUID('fd3dda11-c8ab-4033-876e-d93deb172c93').bytes,
    '<|call|>': UUID('1220f796-e388-4de5-b487-fe2eb5fe03c0').bytes,
    '<|untrusted|>': UUID('07d7da55-b346-4cff-8b37-7cefacf8a3e8').bytes,
    '<|end_untrusted|>': UUID('f265bd9c-c717-469e-a447-920687d65d90').bytes,
}

INCLUDE_SPECIAL_TOKENS = [
    "<|start|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|refusal|>",
    "<|constrain|>",
    "<|channel|>",
    "<|call|>",
    "<|untrusted|>",
    "<|end_untrusted|>",
]

GPTOSS_MODEL_UUID = UUID('df52dc86-1789-4ed0-a295-66f10508145b').bytes
APPLE_GPU_LAYOUT_UUID = UUID('229177a8-5775-4268-bfd8-d588b351c56d').bytes
TIKTOKEN_TOKENIZER_UUID = UUID('7401aded-2a95-40cb-b782-9ccebaafe72b').bytes

UE8_OFFSET = 14  # bias to MXFP4 block scales

def write_file_header(f):
    f.write(FILE_MAGIC)

def write_tokenizer_header(f,
                           num_special_tokens: int,
                           num_text_tokens: int,
                           regex_size: int,
                           tokens_size: int):
    f.write(TIKTOKEN_TOKENIZER_UUID)
    f.write(struct.pack('<I', num_special_tokens))
    f.write(struct.pack('<I', num_text_tokens))
    f.write(struct.pack('<I', regex_size))
    f.write(struct.pack('<I', tokens_size))

def write_model_header(f,
                       context_length : int,
                       num_blocks : int,
                       num_experts : int,
                       num_active_experts : int,
                       embedding_dim : int,
                       mlp_dim : int,
                       swiglu_limit : float,
                       head_dim: int,
                       num_heads : int,
                       num_kv_heads : int,
                       attention_window : int,
                       rope_theta : float,
                       interpolation_scale : float,
                       yarn_offset : float,
                       yarn_scale : float,
                       yarn_multiplier : float,
                       rmsnorm_epsilon : float):
    f.write(GPTOSS_MODEL_UUID)
    f.write(struct.pack('<I', context_length))
    f.write(struct.pack('<I', num_blocks))
    f.write(struct.pack('<I', num_experts))
    f.write(struct.pack('<I', num_active_experts))
    f.write(struct.pack('<I', embedding_dim))
    f.write(struct.pack('<I', mlp_dim))
    f.write(struct.pack('<f', swiglu_limit))
    f.write(struct.pack('<I', head_dim))
    f.write(struct.pack('<I', num_heads))
    f.write(struct.pack('<I', num_kv_heads))
    f.write(struct.pack('<I', attention_window))
    f.write(struct.pack('<f', rope_theta))
    f.write(struct.pack('<f', interpolation_scale))
    f.write(struct.pack('<f', yarn_offset))
    f.write(struct.pack('<f', yarn_scale))
    f.write(struct.pack('<f', yarn_multiplier))
    f.write(struct.pack('<f', rmsnorm_epsilon))
    f.write(APPLE_GPU_LAYOUT_UUID)


def write_padding(out_file, alignment_multiple=16384):
    offset = out_file.tell()
    alignment_size = -offset % alignment_multiple
    if alignment_size != 0:
        alignment = bytes(alignment_size)
        out_file.write(alignment)


def write_embedding_weight(out_file, weight):
    write_padding(out_file, alignment_multiple=16)

    assert weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.bfloat16
    out_file.write(weight.view(torch.uint8).numpy().tobytes())


def write_rmsnorm_gain(out_file, gain):
    write_padding(out_file, alignment_multiple=16)

    assert gain.dtype == torch.bfloat16
    out_file.write(gain.view(torch.uint8).numpy().tobytes())


def write_attn_sink(out_file, sink):
    write_padding(out_file, alignment_multiple=16)

    assert sink.dtype == torch.bfloat16
    out_file.write(sink.view(torch.uint8).numpy().tobytes())


def write_linear_weight(out_file, *args):
    write_padding(out_file, alignment_multiple=16)

    for t in args:
        out_file.write(t.view(torch.uint8).numpy().tobytes())


def main(args):
    options = parser.parse_args(args)

    with open(os.path.join(options.src, "config.json"), "r") as f:
        config = json.load(f)

    num_blocks = config["num_hidden_layers"]
    num_experts = config["num_experts"]
    num_active_experts = 4
    num_q_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    embedding_dim = config["hidden_size"]
    mlp_dim = config["intermediate_size"]
    swiglu_limit = config.get("swiglu_limit", 7.0)
    rope_theta = config["rope_theta"]
    attention_window = config["sliding_window"]
    initial_context_length = config["initial_context_length"]
    rope_scaling_factor = config["rope_scaling_factor"]
    rope_ntk_alpha = config["rope_ntk_alpha"]
    rope_ntk_beta = config["rope_ntk_beta"]

    tokens_size = 0
    num_text_tokens = 0
    # First add all text tokens
    for t in range(o200k_gptoss.n_vocab):
        if not harmony_encoding.is_special_token(t):
            token_bytes = o200k_gptoss.decode_single_token_bytes(t)
            assert len(token_bytes) > 0
            tokens_size += len(token_bytes) + 2  # uint16_t string length + string data
            num_text_tokens += 1
    # Then add all special tokens
    num_included_tokens = 200013 + 1
    print(f"Tokenizer: {num_included_tokens} tokens")

    # Read from all files ending with .safetensors in the checkpoint directory
    safetensor_files = [
        os.path.join(options.src, fname)
        for fname in os.listdir(options.src)
        if fname.endswith(".safetensors")
    ]
    # Build a mapping from tensor name to filepath
    tensor_name_to_file = {}
    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as src:
            for key in src.keys():
                tensor_name_to_file[key] = safetensor_file

    def get_tensor(name):
        with safe_open(tensor_name_to_file[name], framework="pt", device="cpu") as src:
            return src.get_tensor(name)

    with open(options.dst, "wb") as dst:
        write_file_header(dst)

        yarn_low = (
            head_dim / 2
            * math.log(initial_context_length / (rope_ntk_beta * 2 * math.pi))
            / math.log(rope_theta)
        )
        yarn_high = (
            head_dim / 2
            * math.log(initial_context_length / (rope_ntk_alpha * 2 * math.pi))
            / math.log(rope_theta)
        )

        write_model_header(dst,
                            context_length=int(initial_context_length * rope_scaling_factor),
                            num_blocks=num_blocks,
                            num_experts=num_experts,
                            num_active_experts=num_active_experts,
                            embedding_dim=embedding_dim,
                            mlp_dim=mlp_dim,
                            swiglu_limit=swiglu_limit,
                            head_dim=head_dim,
                            num_heads=num_q_heads,
                            num_kv_heads=num_kv_heads,
                            attention_window=attention_window,
                            rope_theta=rope_theta,
                            interpolation_scale=1.0 / rope_scaling_factor,
                            yarn_offset=-yarn_low / (yarn_high - yarn_low),
                            yarn_scale=1.0 / (yarn_high - yarn_low),
                            yarn_multiplier=0.1 * math.log(rope_scaling_factor) + 1.0,
                            rmsnorm_epsilon=1.0e-5)

        write_tokenizer_header(dst,
                                num_special_tokens=num_included_tokens - num_text_tokens,
                                num_text_tokens=num_text_tokens,
                                regex_size=len(o200k_gptoss._pat_str.encode("ascii")) + 1,
                                tokens_size=tokens_size)

        ### Tokenizer
        # Special tokens
        for token_idx in range(num_text_tokens, num_included_tokens):
            token = o200k_gptoss.decode_single_token_bytes(token_idx).decode('ascii')
            if token in INCLUDE_SPECIAL_TOKENS:
                dst.write(SPECIAL_TOKEN_UUID[token])
            else:
                dst.write(bytes(16))
        # Regex
        dst.write(o200k_gptoss._pat_str.encode("ascii"))
        dst.write(struct.pack('B', 0))
        # Text tokens
        tokenizer_bytes_written = 0
        for t in range(num_text_tokens):
            token_bytes = o200k_gptoss.decode_single_token_bytes(t)
            assert len(token_bytes) > 0
            dst.write(struct.pack('<H', len(token_bytes)))
            dst.write(token_bytes)
            tokenizer_bytes_written += len(token_bytes) + 2
        assert(tokenizer_bytes_written == tokens_size), (tokenizer_bytes_written, tokens_size)
        write_padding(dst)

        embedding_weight = get_tensor("embedding.weight")
        # Filter out unused tokens
        embedding_weight = embedding_weight[:num_included_tokens, :]
        write_embedding_weight(dst, embedding_weight)

        for n in tqdm(range(num_blocks)):
            write_rmsnorm_gain(dst, get_tensor(f"block.{n}.attn.norm.scale"))

            attn_qkv_weight = get_tensor(f"block.{n}.attn.qkv.weight")
            attn_qkv_bias = get_tensor(f"block.{n}.attn.qkv.bias")
            for qkv in (attn_qkv_weight, attn_qkv_bias):
                qk = qkv[:head_dim * (num_q_heads + num_kv_heads), ...].contiguous()
                v = qkv[head_dim * (num_q_heads + num_kv_heads):, ...].contiguous()
                qk = qk.view(num_q_heads + num_kv_heads, 2, head_dim // 2, -1).transpose(1, 2).reshape(num_q_heads + num_kv_heads, head_dim, -1)
                q = qk[:num_q_heads, ...]
                k = qk[num_q_heads:, ...]
                # Factor multiplication by 1/sqrt(64) = 0.125 = 0.5 * 0.25 in SDPA into Q and K projections
                assert head_dim == 64
                q *= 0.5
                k *= 0.25
                v = v.view(num_kv_heads, head_dim, -1)
                qkv.copy_(torch.cat((q, k, v), dim=0).reshape(*qkv.shape))

            write_linear_weight(dst, attn_qkv_weight, attn_qkv_bias)

            write_attn_sink(dst, get_tensor(f"block.{n}.attn.sinks"))

            write_linear_weight(dst, get_tensor(f"block.{n}.attn.out.weight"), get_tensor(f"block.{n}.attn.out.bias"))

            write_rmsnorm_gain(dst, get_tensor(f"block.{n}.mlp.norm.scale"))

            write_linear_weight(dst, get_tensor(f"block.{n}.mlp.gate.weight"), get_tensor(f"block.{n}.mlp.gate.bias"))

        write_rmsnorm_gain(dst, get_tensor("norm.scale"))

        unembedding_weight = get_tensor("unembedding.weight")
        unembedding_weight = unembedding_weight[:num_included_tokens, :]
        write_linear_weight(dst, unembedding_weight)

        for n in tqdm(range(num_blocks)):
            mlp1_blocks = get_tensor(f"block.{n}.mlp.mlp1_weight.blocks")
            mlp1_scales = get_tensor(f"block.{n}.mlp.mlp1_weight.scales")
            assert mlp1_scales.min().item() < 254 - UE8_OFFSET
            mlp1_bias = get_tensor(f"block.{n}.mlp.mlp1_bias")

            mlp2_blocks = get_tensor(f"block.{n}.mlp.mlp2_weight.blocks")
            mlp2_scales = get_tensor(f"block.{n}.mlp.mlp2_weight.scales")
            assert mlp2_scales.min().item() < 254 - UE8_OFFSET
            mlp2_bias = get_tensor(f"block.{n}.mlp.mlp2_bias")

            # Write MoE weights grouped by expert
            write_padding(dst)

            for e in range(num_experts):
                write_padding(dst, alignment_multiple=16)                    
                dst.write(mlp1_blocks[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                dst.write((mlp1_scales + UE8_OFFSET)[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                dst.write(mlp1_bias[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)                    
                dst.write(mlp2_blocks[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                dst.write((mlp2_scales + UE8_OFFSET)[e, ...].view(torch.uint8).numpy().tobytes())

                write_padding(dst, alignment_multiple=16)
                dst.write(mlp2_bias[e, ...].view(torch.uint8).numpy().tobytes())

if __name__ == "__main__":
    main(sys.argv[1:])
