#pragma once

#ifndef __cplusplus
    #include <stdatomic.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "internal/metal.h"


struct gptoss_tokenizer {
#ifndef __cplusplus
    atomic_uint_least64_t ref_count;
#else
    uint_least64_t ref_count;
#endif

    void* mapping_ptr;
    size_t mapping_size;

    const char* regex_ptr;
    const char* tokens_ptr;

    uint32_t num_text_tokens;
    uint32_t num_special_tokens;

    uint32_t special_token_id[gptoss_special_token_max - 1];
};

struct gptoss_model {
#ifndef __cplusplus
    atomic_uint_least64_t ref_count;
#else
    uint_least64_t ref_count;
#endif

    struct gptoss_tokenizer* tokenizer;

    void* mapping_ptr;
    size_t mapping_size;

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

    uint32_t vocabulary_size;

    // Maximum number of tokens that can be processed in a single batch.
    // Once the batch size is reached, we process it to fill the KV cache.
    size_t max_batch_tokens;

    bool lock_memory;

    size_t weights_size;
    size_t allocation_size;

    // Metal objects
    struct gptoss_metal_device device;
    size_t max_threadgroups;
    struct gptoss_metal_command_queue command_queue;
    struct gptoss_metal_library library;
    struct gptoss_metal_function bf16_f32_embeddings_fn;
    struct gptoss_metal_function f32_bf16w_rmsnorm_fn;
    struct gptoss_metal_function f32_bf16w_matmul_fn;
    struct gptoss_metal_function f32_bf16w_matmul_qkv_fn;
    struct gptoss_metal_function f32_bf16w_dense_matmul_qkv_fn;
    struct gptoss_metal_function f32_bf16w_dense_matmul_attn_output_fn;
    struct gptoss_metal_function f32_bf16w_dense_matmul_mlp_gate_fn;
    struct gptoss_metal_function f32_bf16w_unembedding_fn;
    struct gptoss_metal_function f32_rope_fn;
    struct gptoss_metal_function f32_mf4w_moe_matmul_swiglu_fn;
    struct gptoss_metal_function f32_mf4w_moe_matmul_fn;
    struct gptoss_metal_function f32_accumulate_e4_fn;
    struct gptoss_metal_function f32_topk_softmax_e32_k4_fn;
    struct gptoss_metal_function f32_topk_softmax_e128_k4_fn;
    struct gptoss_metal_function f32_sdpa_q8_d64_fn;
    struct gptoss_metal_function f32_softmax_fn;
    struct gptoss_metal_function f32_sample_fn;

    size_t per_block_shared_weights_size;
    size_t per_expert_block_weight_size;

    size_t embeddings_threadgroup_size;
    size_t attn_qkv_threadgroup_size;
    size_t attn_out_threadgroup_size;
    size_t mlp_gate_threadgroup_size;
    size_t mlp_swiglu_threadgroup_size;
    size_t mlp_out_threadgroup_size;
    size_t mlp_acc_threadgroup_size;
    size_t unembedding_threadgroup_size;

    size_t attn_rmsnorm_gain_offset;
    size_t attn_qkv_weight_offset;
    size_t attn_qkv_bias_offset;
    size_t attn_sdpa_sink_offset;
    size_t attn_out_weight_offset;
    size_t attn_out_bias_offset;
    size_t mlp_rmsnorm_gain_offset;
    size_t mlp_gate_weight_offset;
    size_t mlp_gate_bias_offset;
    size_t mlp_swiglu_scale_offset;
    size_t mlp_swiglu_bias_offset;
    size_t mlp_out_block_offset;
    size_t mlp_out_scale_offset;
    size_t mlp_out_bias_offset;
    size_t rmsnorm_weight_offset;
    size_t unembedding_weight_offset;

    // Buffer with non-MoE weights. Includes MoE gates, embeddings/unembeddings.
    struct gptoss_metal_buffer shared_weight_buffer;
    // num_blocks per-block buffers with MoE weights to follow.
    struct gptoss_metal_buffer block_weight_buffers[];
};

#define GPTOSS_DEFAULT_BATCH_SIZE 128

struct gptoss_context {
#ifndef __cplusplus
    atomic_uint_least64_t ref_count;
#else
    uint_least64_t ref_count;
#endif

    struct gptoss_model* model;
    // Number of tokens processed in the context.
    size_t num_tokens;
    // Number of tokens in the KV cache.
    size_t num_kv_tokens;
    // Length of the context.
    size_t max_tokens;

    size_t kvcache_size;
    size_t allocation_size;

    // Activation buffers.
    // TODO: merge into a single buffer.
    struct gptoss_metal_buffer residual_activation_buffer;  // Residual stream
    struct gptoss_metal_buffer rmsnorm_activation_buffer;  // Both attention & MLP RMSNorm output
    struct gptoss_metal_buffer qkv_activation_buffer;  // QKV projection output
    struct gptoss_metal_buffer sdpa_activation_buffer;  // SDPA output
    struct gptoss_metal_buffer gate_activation_buffer;  // MoE gating output
    struct gptoss_metal_buffer expert_activation_buffer;  // MoE expert predictions
    struct gptoss_metal_buffer swiglu_activation_buffer;  // MLP+SwiGLU output
    struct gptoss_metal_buffer moe_activation_buffer;  // MoE MLP output (per-active expert)

    // Input/output buffers.
    struct gptoss_metal_buffer control_buffer;
    struct gptoss_metal_buffer token_buffer;  // uint32 token IDs
    struct gptoss_metal_buffer score_buffer;  // unembedding outputs
    struct gptoss_metal_buffer prob_buffer;
    struct gptoss_metal_buffer sum_buffer;
    struct gptoss_metal_buffer argmax_buffer;
    struct gptoss_metal_buffer kvcache_buffer;
};
