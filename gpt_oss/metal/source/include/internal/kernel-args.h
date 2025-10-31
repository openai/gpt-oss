#pragma once

#if !defined(__METAL_VERSION__)
#include <stdint.h>
#endif

// TODO(ibahmed): specalize using metal function constants.
#define QKV_Bm 64
#define QKV_Bn 64
#define QKV_Bk 32
#define QKV_Sg_Bm 32
#define QKV_Sg_Bn 32

#define ATTN_OUTPUT_Bm 32
#define ATTN_OUTPUT_Bn 64
#define ATTN_OUTPUT_Bk 64
#define ATTN_OUTPUT_Sg_Bm 32
#define ATTN_OUTPUT_Sg_Bn 16

#define MLP_GATE_Bm 64
#define MLP_GATE_Bn 16
#define MLP_GATE_Bk 64
#define MLP_GATE_Sg_Bm 16
#define MLP_GATE_Sg_Bn 16

#define MOE_DENSE_MATMUL_SWIGLU_Bm 32
#define MOE_DENSE_MATMUL_SWIGLU_Bn 64
#define MOE_DENSE_MATMUL_SWIGLU_Bk 16
#define MOE_DENSE_MATMUL_SWIGLU_Sg_Bm 32
#define MOE_DENSE_MATMUL_SWIGLU_Sg_Bn 16

#define MOE_DENSE_MATMUL_Bm 32
#define MOE_DENSE_MATMUL_Bn 64
#define MOE_DENSE_MATMUL_Bk 16
#define MOE_DENSE_MATMUL_Sg_Bm 32
#define MOE_DENSE_MATMUL_Sg_Bn 16

struct gptoss_expert_prediction {
    uint32_t expert_id;
    float score;
};

struct gptoss_control {
    uint32_t abort;
};

struct gptoss_topk_args {
    uint32_t num_vecs_per_token;
};

struct gptoss_sdpa_args {
    uint32_t qkv_dim;
    uint32_t num_kv_tokens;
    uint32_t kv_stride;
    uint32_t window;
};

struct gptoss_u32_fill_random_args {
    uint64_t num_vecs_per_threadgroup;
    uint64_t num_vecs;
    uint64_t offset;
    uint64_t seed;
};

struct gptoss_f32_fill_random_args {
    uint64_t num_vecs_per_threadgroup;
    uint64_t num_vecs;
    uint64_t offset;
    uint64_t seed;
    float scale;
    float bias;
};

struct gptoss_accumulate_args {
    uint32_t num_vecs_per_expert;
    uint32_t num_vecs_per_threadgroup;
    uint32_t num_vecs;
};

struct gptoss_convert_args {
    uint64_t num_vecs_per_threadgroup;
    uint64_t num_vecs;
};

struct gptoss_embeddings_args {
    uint32_t num_vecs;
};

struct gptoss_rmsnorm_args {
    uint32_t num_vecs;
    float num_channels;
    float epsilon;
};

struct gptoss_matmul_args {
    uint32_t num_column_vecs;
    uint32_t num_rows;
    uint32_t add;
};

struct gptoss_dense_matmul_args {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

// Specialize qkv matmul args as it writes kv directly to the KV cache buffer.
struct gptoss_dense_matmul_qkv_args {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t max_tokens;
    uint32_t token_offset;
};

struct gptoss_scatter_args {
    uint32_t tokens;
    uint32_t active_experts_per_token;
    uint32_t token_stride;
};

struct gptoss_moe_dense_matmul_swiglu_args {
    uint32_t k;
    uint32_t n;
    uint32_t weight_blocks_expert_stride_bytes;
    uint32_t weight_scales_expert_stride_bytes;
    uint32_t bias_expert_stride_bytes;
    float swiglu_min;
    float swiglu_max;
};
struct gptoss_moe_dense_matmul_args {
    uint32_t k;
    uint32_t n;
    uint32_t weight_blocks_expert_stride_bytes;
    uint32_t weight_scales_expert_stride_bytes;
    uint32_t bias_expert_stride_bytes;
};

struct gptoss_expert_routing_metadata_args {
uint32_t tokens;
    uint32_t num_experts;
};

struct gptoss_gather_args {
    uint32_t tokens;
    uint32_t active_experts_per_token;
    uint32_t token_stride;
};

struct gptoss_unembedding_args {
    uint32_t num_column_vecs;
    uint32_t num_rows_per_threadgroup;
    uint32_t num_rows;
};

struct gptoss_moe_matmul_swiglu_args {
    uint32_t num_column_vecs;
    uint32_t num_rows;
    uint32_t num_active_experts;
    uint32_t weight_expert_stride;  // in bytes
    uint32_t output_expert_stride;  // in elements
    float swiglu_min;
    float swiglu_max;
};

struct gptoss_moe_matmul_args {
    uint32_t num_column_vecs;
    uint32_t num_rows;
    uint32_t num_active_experts;
    uint32_t input_expert_stride;  // in blocks of 32 elements
    uint32_t weight_expert_stride;  // in bytes
    uint32_t output_expert_stride;  // in elements
};

struct gptoss_rope_args {
    uint32_t token_stride;
    uint32_t token_offset;
    uint32_t max_tokens;
    float freq_scale;
    float interpolation_scale;
    float yarn_offset;
    float yarn_scale;
    float yarn_multiplier;
};

struct gptoss_qkv_args {
    uint32_t num_column_vecs;
    uint32_t num_rows;
    uint32_t token_offset;
    float freq_scale;
    float interpolation_scale;
    float yarn_offset;
    float yarn_scale;
    float yarn_multiplier;
    uint32_t max_tokens;
};

struct gptoss_softmax_args {
    uint32_t num_vecs;
    uint32_t num_vecs_per_threadgroup;
    uint32_t max_threadgroups;
    float temperature;
};

struct gptoss_sample_args {
    uint64_t rng_seed;
    uint32_t rng_offset;
    uint32_t num_blocks;
    uint32_t num_dims;
    uint32_t num_dims_per_block;
};
