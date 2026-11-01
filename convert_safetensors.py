import torch
import os
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_PATH = r"./gpt-oss-20b/original"
OUTPUT_PATH = r"./gpt-oss-20b/optimized"


# =================================================

class FP8Converter:
    def __init__(self, device="cuda"):
        self.device = device
        self.lut_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.bfloat16, device=self.device
        )

    def _dequantize_mxfp4(self, blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Dequantizes MXFP4 to BFloat16."""
        blocks = blocks.to(self.device)
        scales = scales.to(device=self.device).to(torch.int32) - 127

        # Input shape: [NumExperts, Rows, Groups, 16]
        original_shape = blocks.shape
        num_experts, rows, groups, b_size = original_shape

        # Reshape for processing
        blocks_flat = blocks.reshape(num_experts, -1, b_size)
        scales_flat = scales.reshape(num_experts, -1, 1)

        # Dequant
        out = torch.empty(blocks_flat.shape[0], blocks_flat.shape[1], 32, dtype=torch.bfloat16, device=self.device)
        low = blocks_flat & 0x0F
        high = (blocks_flat >> 4) & 0x0F
        out[:, :, 0::2] = self.lut_values[low.long()]
        out[:, :, 1::2] = self.lut_values[high.long()]
        out = torch.ldexp(out, scales_flat)

        # Reshape back to [NumExperts, OutDim, InDim]
        # groups * 32 is the last dimension
        return out.view(num_experts, rows, groups * 32)

    def convert(self):
        input_file = os.path.join(INPUT_PATH, "model.safetensors")
        output_file = os.path.join(OUTPUT_PATH, "model.safetensors")
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        print(f"Processing {input_file}...")
        tensors_to_save = {}

        with safe_open(input_file, framework="pt", device="cpu") as f:
            keys = f.keys()
            mlp_keys = [k for k in keys if ".mlp." in k]

            pbar = tqdm(mlp_keys)
            for key in pbar:
                pbar.set_description(f"Converting {key}")

                if key.endswith(".blocks"):
                    base_name = key.replace(".blocks", "")
                    scale_key = base_name + ".scales"

                    if scale_key not in keys:
                        continue

                    blocks = f.get_tensor(key)
                    scales = f.get_tensor(scale_key)

                    # 1. Dequantize -> Shape: [32, OutDim, InDim]
                    bf16_tensor = self._dequantize_mxfp4(blocks, scales)

                    # 2. Calculate Scale PER EXPERT
                    # Shape: [32, OutDim, InDim] -> amax over dim 1 & 2 -> [32]
                    abs_max = torch.abs(bf16_tensor.flatten(start_dim=1)).amax(dim=1)
                    abs_max = torch.clamp(abs_max, min=1e-12)
                    fp8_scale = abs_max / 448.0  # Shape: [32]

                    # 3. Quantize to FP8
                    # We must expand scale to match weight dimensions for broadcasting
                    # Weight is [32, Out, In]. Scale needs to be [32, 1, 1]
                    scale_expanded = fp8_scale.view(-1, 1, 1)

                    fp8_tensor = (bf16_tensor / scale_expanded).to(torch.float8_e4m3fn)

                    # 4. Save
                    tensors_to_save[base_name] = fp8_tensor.cpu()
                    # Save the 1D scale vector
                    tensors_to_save[base_name + "_scale"] = fp8_scale.cpu().to(torch.bfloat16)

                elif key.endswith("_bias"):
                    # Biases remain unchanged
                    tensors_to_save[key] = f.get_tensor(key)

        print(f"Saving optimized weights to {output_file}...")
        save_file(tensors_to_save, output_file)
        print("Done.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FP8Converter(device).convert()