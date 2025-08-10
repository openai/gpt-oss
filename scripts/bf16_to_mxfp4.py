from argparse import ArgumentParser
from typing import Optional, Tuple

from glob import glob
import json
import os
import re

import torch
from tqdm import tqdm
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

# GPT-OSS
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


from safetensors.torch import load_file, save_file


# NOTE (yiakwy) : for quick verification purpose
# from simple_py_mxfp4 import quantize_bf16_mxfp4

from gpt_oss_triton_mxfp4 import quantize_bf16_mxfp4

def has_tensor(weight_map, loaded_files, mxfp4_path, tensor_name):
    """
    Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

    Args:
        tensor_name (str): The name of the tensor to retrieve.

    Returns:
        torch.Tensor: The retrieved tensor.

    Raises:
        KeyError: If the tensor does not exist in the safetensor file.
    """
    file_name = weight_map[tensor_name]
    if file_name not in loaded_files:
        file_path = os.path.join(mxfp4_path, file_name)
        loaded_files[file_name] = load_file(file_path, device="cuda")
    return loaded_files[file_name][tensor_name]


def quantize(bf16_path, mxfp4_path, ref_weights_scale_inv_map_path=None):
    ref_weights_scale_inv_map_f = os.path.join(
        ref_weights_scale_inv_map_path, "weight_with_scale_inv_map.index.json"
    )
    with open(ref_weights_scale_inv_map_f, "r") as f:
        s_model_index = json.load(f)
    ref_weights_scale_inv_map = s_model_index["weight_with_scale_inv_map"]

    os.makedirs(mxfp4_path, exist_ok=True)

    model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    bf16_weight_names = []
    bf16_weight_scales = {}

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            block_name = f"{weight_name}_blocks"
            if (
                ref_weights_scale_inv_map is not None
                and ref_weights_scale_inv_map.get(block_name, None) is not None
            ):
                scale_name = f"{weight_name}_scales"

                bf16_weight_names.append(weight_name)
                bf16_weight_scales[scale_name] = file_name
                weight_transpose = weight.permute(0, 2, 1).contiguous()
                mxfp4_weight, mxfp4_scale = quantize_bf16_mxfp4(weight_transpose, 32)
                new_state_dict[block_name] = mxfp4_weight.view(*mxfp4_weight.shape[:-1], -1, 16).contiguous()
                new_state_dict[scale_name] = mxfp4_scale.contiguous()
            else:
                print(f"skipping {weight_name} dtype={weight.dtype}...")
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(mxfp4_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        del new_state_dict

        if len(loaded_files) > 1:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(mxfp4_path, "model.safetensors.index.json")

    for weight_name in bf16_weight_names:
        scale_name = f"{weight_name}_scales"
        block_name = f"{weight_name}_blocks"

        weight_map[scale_name] = bf16_weight_scales[scale_name]
        weight_map[block_name] = weight_map[weight_name]

        weight_map.pop(weight_name)

    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def read_mxfp4_list(bf16_path):
    model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    mxfp4_weights_inv_map = {}

    # Cache for loaded safetensor files
    loaded_files = {}
    mxfp4_weights_name = []

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("scales"):
                print(f"skipping {weight_name} dtype={weight.dtype}...")
                continue
            elif weight.element_size() == 1: # MXFP4
                scale_name = weight_name.replace("blocks", "scales")
                try:
                    weight_scales = has_tensor(
                        weight_map, loaded_files, bf16_path, scale_name
                    )
                    mxfp4_weights_name.append(weight_name)
                    mxfp4_weights_inv_map[weight_name] = weight_map[scale_name]
                except KeyError:
                    print(
                        f"Warning: Missing scales tensor for {weight_name}, skipping conversion ..."
                    )
            else:
                print(f"skipping {weight_name} dtype={weight.dtype}...")

        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    weights_with_scale_inv = os.path.join(
        bf16_path, "weight_with_scale_inv_map.index.json"
    )
    with open(weights_with_scale_inv, "w") as f:
        json.dump(
            {"metadata": {}, "weight_with_scale_inv_map": mxfp4_weights_inv_map},
            f,
            indent=2,
        )


def _verify_tokenizer_and_model(hf_tokenizer, model):
    texts = ["你是谁？"] # ["Give me a short introduction to large language model.", ]
    messages = [
        {"role": "user", "content": text} for text in texts
    ]

    prompts = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)

    model_inputs = hf_tokenizer([prompts], return_tensors="pt").to(model.device)
    outputs_ids = model.generate(**model_inputs, max_new_tokens=256)

    outputs_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs_ids)
    ]

    response = hf_tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)[0]
    print(f"response : {response}")


def verify_tokenizer_and_model(hf_tokenizer_path, model):
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)

    _verify_tokenizer_and_model(hf_tokenizer, model)


def load_and_verify_hf_model(source_model):
    model = AutoModelForCausalLM.from_pretrained(
        source_model, torch_dtype="auto", device_map="auto"
    )

    verify_tokenizer_and_model(source_model, model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source_model", default=None, type=str, required=False, help="source model."
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=False, help="Where to save the converted model."
    )
    parser.add_argument(
        "--get_scaled_weights", action="store_true", required=False, help="get scaled weights"
    )
    args = parser.parse_args()
    
    if not args.output_dir:
        if args.get_scaled_weights:
            read_mxfp4_list(args.source_model)
        else:
            load_and_verify_hf_model(args.source_model)
    else:
        quantize(args.source_model, args.output_dir, ref_weights_scale_inv_map_path="/root/models/gpt-oss-120b")
