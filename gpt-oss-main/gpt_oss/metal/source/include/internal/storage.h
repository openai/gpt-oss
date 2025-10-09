#pragma once

#include <stdbool.h>
#include <stdint.h>

struct gptoss_file_header {
    char magic[12];
    uint32_t zero;
};

struct gptoss_gptoss_model_header {
    uint32_t context_length;
    uint32_t num_blocks;
    uint32_t num_experts;
    uint32_t num_active_experts;
    uint32_t embedding_dim;
    uint32_t mlp_dim;
    float swiglu_limit;
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t attention_window;
    float rope_theta;
    float interpolation_scale;
    float yarn_offset;
    float yarn_scale;
    float yarn_multiplier;
    float rmsnorm_epsilon;
};

struct gptoss_tiktoken_tokenizer_header {
    uint32_t num_special_tokens;
    uint32_t num_text_tokens;
    uint32_t regex_size;
    uint32_t tokens_size;
};
