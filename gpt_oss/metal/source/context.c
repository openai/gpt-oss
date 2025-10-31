#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <gpt-oss.h>

#include "internal/datatype.h"
#include "internal/model.h"
#include "internal/metal.h"
#include "internal/metal-kernels.h"
#include "internal/log.h"
#include "internal/rng.h"


enum gptoss_status GPTOSS_ABI gptoss_context_create(
    gptoss_model_t model,
    size_t context_length,
    size_t max_batch_tokens,
    gptoss_context_t* context_out)
{
    *context_out = NULL;

    enum gptoss_status status = gptoss_status_success;
    struct gptoss_context* context = NULL;

    // Validate context_length
    if (context_length == 0) {
        context_length = model->context_length;
    } else if (context_length > model->context_length) {
        GPTOSS_LOG_ERROR("requested context length %zu exceeds model context length %" PRIu32,
            context_length, model->context_length);
        status = gptoss_status_invalid_argument;
        goto cleanup;
    }
    assert(context_length != 0);
    assert(context_length <= model->context_length);

    // Validate max_batch_tokens
    if (max_batch_tokens == 0) {
        max_batch_tokens = GPTOSS_DEFAULT_BATCH_SIZE;
    } else if (max_batch_tokens > context_length) {
        GPTOSS_LOG_ERROR("requested max batch tokens %zu exceeds context length %zu",
            max_batch_tokens, context_length);
        status = gptoss_status_invalid_argument;
        goto cleanup;
    }
    assert(max_batch_tokens != 0);
    assert(max_batch_tokens <= context_length);

    context = malloc(sizeof(struct gptoss_context));
    if (context == NULL) {
        GPTOSS_LOG_ERROR("failed to allocate %zu bytes for Context object",
            sizeof(struct gptoss_context));
        status = gptoss_status_insufficient_memory;
        goto cleanup;
    }
    memset(context, 0, sizeof(struct gptoss_context));

    atomic_store_explicit(&context->ref_count, 1, memory_order_relaxed);
    context->max_tokens = context_length;
    context->max_batch_tokens = max_batch_tokens;

    // Activation buffers
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->embedding_dim * sizeof(float), NULL, &context->residual_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->embedding_dim * sizeof(float), NULL, &context->rmsnorm_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->head_dim * (model->num_heads + 2 * model->num_kv_heads) * sizeof(float), NULL, &context->qkv_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->head_dim * model->num_heads * sizeof(float), NULL, &context->sdpa_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->num_experts * sizeof(float), NULL, &context->gate_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->num_experts * sizeof(struct gptoss_expert_prediction), NULL, &context->expert_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    // The last entry will hold the total number of tokens.
    status = gptoss_metal_buffer_create(&model->device, (1 + model->num_experts) * sizeof(uint32_t), NULL, &context->expert_offset_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->num_active_experts * sizeof(uint32_t), NULL, &context->token_to_expert_routing_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->num_active_experts * model->embedding_dim * sizeof(float), NULL, &context->swiglu_input_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->num_active_experts * model->mlp_dim * sizeof(float), NULL, &context->swiglu_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->num_active_experts * model->embedding_dim * sizeof(float), NULL, &context->moe_activation_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }

    // Input/output buffers
    status = gptoss_metal_buffer_create(&model->device, sizeof(struct gptoss_control), NULL, &context->control_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, context_length * sizeof(uint32_t), NULL, &context->token_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->vocabulary_size * sizeof(float), NULL, &context->score_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->vocabulary_size * sizeof(float), NULL, &context->prob_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * model->max_threadgroups * sizeof(float), NULL, &context->sum_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, max_batch_tokens * sizeof(uint64_t), NULL, &context->argmax_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_buffer_create(&model->device, model->num_blocks * context_length * 2 * model->num_kv_heads * model->head_dim * sizeof(float), NULL, &context->kvcache_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }

    context->kvcache_size = context->kvcache_buffer.size;
    context->allocation_size = 
        context->residual_activation_buffer.size + context->rmsnorm_activation_buffer.size +
        context->qkv_activation_buffer.size + context->sdpa_activation_buffer.size +
        context->gate_activation_buffer.size + context->expert_activation_buffer.size +
        context->expert_offset_buffer.size + context->token_to_expert_routing_buffer.size + context->swiglu_input_buffer.size +
        context->swiglu_activation_buffer.size + context->moe_activation_buffer.size +
        context->token_buffer.size + context->kvcache_buffer.size + context->score_buffer.size + context->argmax_buffer.size;

    context->model = model;
    gptoss_model_retain(model);
    *context_out = context;
    context = NULL;

cleanup:
    gptoss_context_release(context);
    return status;
}

enum gptoss_status GPTOSS_ABI gptoss_context_get_num_tokens(
    gptoss_context_t context,
    size_t* num_tokens_out)
{
    *num_tokens_out = context->num_tokens;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_context_get_max_tokens(
    gptoss_context_t context,
    size_t* max_tokens_out)
{
    *max_tokens_out = context->max_tokens;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_context_get_tokens(
    gptoss_context_t context,
    uint32_t* tokens_out,
    size_t max_tokens,
    size_t* num_tokens_out)
{
    *num_tokens_out = context->num_tokens;
    if (max_tokens < context->num_tokens) {
        return gptoss_status_insufficient_memory;
    }

    if (context->num_tokens != 0) {
        memcpy(tokens_out, context->token_buffer.ptr, context->num_tokens * sizeof(uint32_t));
    }
    return gptoss_status_success;
}

// Prefill: input_tokens_offset = number of tokens in KV cache, num_input_tokens > 0, num_output_tokens = 0.
// Sampling: input_tokens_offset = number of tokens in the context - 1, num_input_tokens = 1, num_output_tokens = 1.
// Perplexity: input_tokens_offset = 0, num_input_tokens > 1, num_output_tokens = num_input_tokens.
static enum gptoss_status process_tokens(
    gptoss_context_t context,
    struct gptoss_metal_command_buffer* command_buffer,
    size_t input_tokens_offset,
    size_t num_input_tokens,
    size_t num_output_tokens)
{
    assert(num_input_tokens != 0);
    assert(num_input_tokens <= context->max_batch_tokens);
    assert(num_output_tokens <= context->max_batch_tokens);
    assert(num_input_tokens >= num_output_tokens);
    const size_t min_tokens_for_dense_matmul_kernels = 64;
    const size_t min_tokens_for_dense_moe_kernels = 64;

    enum gptoss_status status = gptoss_status_success;
    const struct gptoss_model* model = context->model;

    const size_t attn_qkv_dim = model->head_dim * (model->num_heads + 2 * model->num_kv_heads);

    const size_t input_tokens_end = input_tokens_offset + num_input_tokens;
    for (size_t input_batch_start = input_tokens_offset;
        input_batch_start < input_tokens_end;
        input_batch_start += context->max_batch_tokens)
    {
        const size_t input_batch_size = math_min(context->max_batch_tokens, input_tokens_end - input_batch_start);
        const size_t input_batch_end = input_batch_start + input_batch_size;
        const size_t output_batch_size = math_sub_sat(num_output_tokens, input_tokens_end - input_batch_end);

        status = gptoss_metal_command_buffer_encode_launch_bf16_f32_embeddings(
            command_buffer,
            &model->bf16_f32_embeddings_fn,
            model->embeddings_threadgroup_size,
            &context->token_buffer,
            input_batch_start * sizeof(uint32_t),
            &model->shared_weight_buffer,
            /*weight_offset=*/0,
            &context->residual_activation_buffer,
            /*output_offset=*/0,
            &context->control_buffer,
            /*control_offset=*/0,
            /*num_tokens=*/input_batch_size,
            /*num_channels=*/model->embedding_dim);
        if (status != gptoss_status_success) {
            GPTOSS_LOG_ERROR("failed to encode bf16_f32_embeddings kernel launch");
            return status;
        }
        for (uint32_t n = 0; n < model->num_blocks; n++) {
            const bool last_block = n + 1 == model->num_blocks;
            const size_t num_block_output_tokens = last_block ? output_batch_size : input_batch_size;

            status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                command_buffer,
                &model->f32_bf16w_rmsnorm_fn,
                &context->residual_activation_buffer,
                /*input_offset=*/0,
                &model->shared_weight_buffer,
                /*weight_offset=*/model->attn_rmsnorm_gain_offset + model->per_block_shared_weights_size * n,
                &context->rmsnorm_activation_buffer,
                /*output_offset=*/0,
                &context->control_buffer,
                /*control_offset=*/0,
                /*num_tokens=*/input_batch_size,
                /*num_channels=*/model->embedding_dim,
                model->rmsnorm_epsilon);
            if (status != gptoss_status_success) {
                GPTOSS_LOG_ERROR("failed to encode f32_bf16w_rmsnorm kernel launch");
                return status;
            }

            if (input_batch_size >= min_tokens_for_dense_matmul_kernels) {
                status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_qkv(
                    command_buffer,
                    &model->f32_bf16w_dense_matmul_qkv_fn,
                    &context->rmsnorm_activation_buffer,
                    /*input_offset=*/0,
                    &model->shared_weight_buffer,
                    /*weight_offset=*/model->attn_qkv_weight_offset + model->per_block_shared_weights_size * n,
                    &model->shared_weight_buffer,
                    /*bias_offset=*/model->attn_qkv_bias_offset + model->per_block_shared_weights_size * n,
                    &context->qkv_activation_buffer,
                    /*output_offset=*/0,
                    &context->kvcache_buffer,
                    /*kv_offset=*/n * model->num_kv_heads * context->max_tokens * 2 * model->head_dim * sizeof(float),
                    &context->control_buffer,
                    /*control_offset=*/0,
                    /*num_tokens=*/input_batch_size,
                    /*num_cols=*/model->embedding_dim,
                    /*num_rows=*/attn_qkv_dim,
                    /*max_tokens=*/context->max_tokens,
                    /*token_offset=*/input_batch_start);
                if (status != gptoss_status_success) {
                    GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul_qkv kernel launch");
                    return status;
                }

                status = gptoss_metal_command_buffer_encode_launch_f32_rope(
                    command_buffer,
                    &model->f32_rope_fn,
                    /*threadgroup_size=*/32,
                    &context->qkv_activation_buffer,
                    /*input_offset=*/0,

                    &context->kvcache_buffer,
                    /*kv_offset=*/n * model->num_kv_heads * context->max_tokens * 2 * model->head_dim * sizeof(float),
                    &context->control_buffer,
                    /*control_offset=*/0,
                    model->rope_theta,
                    model->interpolation_scale,
                    model->yarn_offset,
                    model->yarn_scale,
                    model->yarn_multiplier,
                    input_batch_size,
                    model->num_heads,
                    model->num_kv_heads,
                    model->head_dim,
                    /*max_tokens=*/context->max_tokens,
                    /*token_offset=*/input_batch_start);
                if (status != gptoss_status_success) {
                    GPTOSS_LOG_ERROR("failed to encode f32_rope kernel launch");
                    return status;
                }

            } else {
                status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul_qkv(
                    command_buffer,
                    &model->f32_bf16w_matmul_qkv_fn,
                    model->attn_qkv_threadgroup_size,
                    &context->rmsnorm_activation_buffer,
                    /*input_offset=*/0,
                    &model->shared_weight_buffer,
                    /*weight_offset=*/model->attn_qkv_weight_offset + model->per_block_shared_weights_size * n,
                    &model->shared_weight_buffer,
                    /*bias_offset=*/model->attn_qkv_bias_offset + model->per_block_shared_weights_size * n,
                    &context->qkv_activation_buffer,
                    /*output_offset=*/0,
                    &context->kvcache_buffer,
                    /*kv_offset=*/n * model->num_kv_heads * context->max_tokens * 2 * model->head_dim * sizeof(float),
                    &context->control_buffer,
                    /*control_offset=*/0,
                    /*num_tokens=*/input_batch_size,
                    /*num_cols=*/model->embedding_dim,
                    /*num_q_heads=*/model->num_heads,
                    /*num_kv_heads=*/model->num_kv_heads,
                    /*attn_head_dim=*/model->head_dim,
                    /*token_offset=*/input_batch_start,
                    /*max_tokens=*/context->max_tokens,
                    /*rope_base=*/model->rope_theta,
                    /*interpolation_scale=*/model->interpolation_scale,
                    /*yarn_offset=*/model->yarn_offset,
                    /*yarn_scale=*/model->yarn_scale,
                    /*yarn_multiplier=*/model->yarn_multiplier);
                if (status != gptoss_status_success) {
                    GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch");
                    return status;
                }
            }

            if (num_block_output_tokens != 0) {
                status = gptoss_metal_command_buffer_encode_launch_f32_sdpa(
                    command_buffer,
                    &model->f32_sdpa_q8_d64_fn,
                    &context->qkv_activation_buffer,
                    /*q_offset=*/attn_qkv_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                    &context->kvcache_buffer,
                    /*kv_offset=*/n * model->num_kv_heads * context->max_tokens * 2 * model->head_dim * sizeof(float),
                    &model->shared_weight_buffer,
                    /*s_offset=*/model->attn_sdpa_sink_offset + model->per_block_shared_weights_size * n,
                    &context->sdpa_activation_buffer,
                    /*output_offset=*/0,
                    &context->control_buffer,
                    /*control_offset=*/0,
                    /*window=*/n % 2 == 0 ? model->attention_window : UINT32_MAX,
                    /*kv_stride=*/2 * context->max_tokens * model->head_dim,
                    num_block_output_tokens,
                    input_batch_start + input_batch_size - num_block_output_tokens,
                    model->num_heads, model->num_kv_heads, model->head_dim);
                if (status != gptoss_status_success) {
                    GPTOSS_LOG_ERROR("failed to encode f32_sdpa kernel launch");
                    return status;
                }

                if (input_batch_size >= min_tokens_for_dense_matmul_kernels) {
                    status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_attn_output(
                        command_buffer,
                        &model->f32_bf16w_dense_matmul_attn_output_fn,
                        &context->sdpa_activation_buffer,
                        /*input_offset=*/0,
                        &model->shared_weight_buffer,
                        /*weight_offset=*/model->attn_out_weight_offset + model->per_block_shared_weights_size * n,
                        &model->shared_weight_buffer,
                        /*bias_offset=*/model->attn_out_bias_offset + model->per_block_shared_weights_size * n,
                        &context->residual_activation_buffer,
                        /*output_offset=*/model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                        &context->control_buffer,
                        /*control_offset=*/0,
                        /*num_tokens=*/num_block_output_tokens,
                        /*num_cols=*/model->num_heads * model->head_dim,
                        /*num_rows=*/model->embedding_dim);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul_attn_output kernel launch");
                        return status;
                    }
                } else {
                    status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul_add(
                        command_buffer,
                        &model->f32_bf16w_matmul_fn,
                        model->attn_out_threadgroup_size,
                        &context->sdpa_activation_buffer,
                        /*input_offset=*/0,
                        &model->shared_weight_buffer,
                        /*weight_offset=*/model->attn_out_weight_offset + model->per_block_shared_weights_size * n,
                        &model->shared_weight_buffer,
                        /*bias_offset=*/model->attn_out_bias_offset + model->per_block_shared_weights_size * n,
                        &context->residual_activation_buffer,
                        /*output_offset=*/model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                        &context->control_buffer,
                        /*control_offset=*/0,
                        /*num_tokens=*/num_block_output_tokens,
                        /*num_cols=*/model->num_heads * model->head_dim,
                        /*num_rows=*/model->embedding_dim);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch");
                        return status;
                    }
                }
                status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                    command_buffer,
                    &model->f32_bf16w_rmsnorm_fn,
                    &context->residual_activation_buffer,
                    /*input_offset=*/model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                    &model->shared_weight_buffer,
                    /*weight_offset=*/model->mlp_rmsnorm_gain_offset + model->per_block_shared_weights_size * n,
                    &context->rmsnorm_activation_buffer,
                    /*output_offset=*/0,
                    &context->control_buffer,
                    /*control_offset=*/0,
                    num_block_output_tokens,
                    model->embedding_dim,
                    model->rmsnorm_epsilon);
                if (status != gptoss_status_success) {
                    GPTOSS_LOG_ERROR("failed to encode f32_bf16w_rmsnorm kernel launch");
                    return status;
                }
                if (input_batch_size >= min_tokens_for_dense_matmul_kernels) {
                    status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_mlp_gate(
                        command_buffer,
                        &model->f32_bf16w_dense_matmul_mlp_gate_fn,
                        &context->rmsnorm_activation_buffer,
                        /*input_offset=*/0,
                        &model->shared_weight_buffer,
                        /*weight_offset=*/model->mlp_gate_weight_offset + model->per_block_shared_weights_size * n,
                        &model->shared_weight_buffer,
                        /*bias_offset=*/model->mlp_gate_bias_offset + model->per_block_shared_weights_size * n,
                        &context->gate_activation_buffer,
                        /*output_offset=*/0,
                        &context->control_buffer,
                        /*control_offset=*/0,
                        num_block_output_tokens,
                        model->embedding_dim,
                        model->num_experts);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul_mlp_gate kernel launch");
                        return status;
                    }
                } else {
                    status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul(
                        command_buffer,
                        &model->f32_bf16w_matmul_fn,
                        model->mlp_gate_threadgroup_size,
                        &context->rmsnorm_activation_buffer,
                        /*input_offset=*/0,
                        &model->shared_weight_buffer,
                        /*weight_offset=*/model->mlp_gate_weight_offset + model->per_block_shared_weights_size * n,
                        &model->shared_weight_buffer,
                        /*bias_offset=*/model->mlp_gate_bias_offset + model->per_block_shared_weights_size * n,
                        &context->gate_activation_buffer,
                        /*output_offset=*/0,
                        &context->control_buffer,
                        /*control_offset=*/0,
                        /*num_tokens=*/num_block_output_tokens,
                        /*num_cols=*/model->embedding_dim,
                        /*num_rows=*/model->num_experts);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch");
                        return status;
                    }
                }

                const char* kernel_name = NULL;
                switch (model->num_experts) {
                    case 32:
                        kernel_name = "f32_topk_softmax_e32_k4_fn";
                        status = gptoss_metal_command_buffer_encode_launch_f32_topk(
                            command_buffer,
                            &model->f32_topk_softmax_e32_k4_fn,
                            &context->gate_activation_buffer, /*input_offset=*/0,
                            &context->expert_activation_buffer, /*output_offset=*/0,
                            &context->control_buffer, /*control_offset=*/0,
                            num_block_output_tokens,
                            model->num_experts,
                            model->num_active_experts);
                        break;
                    case 128:
                        kernel_name = "f32_topk_softmax_e128_k4_fn";
                        status = gptoss_metal_command_buffer_encode_launch_f32_topk(
                            command_buffer,
                            &model->f32_topk_softmax_e128_k4_fn,
                            &context->gate_activation_buffer, /*input_offset=*/0,
                            &context->expert_activation_buffer, /*output_offset=*/0,
                            &context->control_buffer, /*control_offset=*/0,
                            num_block_output_tokens,
                            model->num_experts,
                            model->num_active_experts);
                        break;
                    default:
                        status = gptoss_status_unsupported_argument;
                        GPTOSS_LOG_ERROR("missing Top-K kernel for %" PRIu32 " experts", model->num_experts);
                        return status;
                }
                if (status != gptoss_status_success) {
                    GPTOSS_LOG_ERROR("failed to encode %s kernel launch", kernel_name);
                    return status;
                }

                // If we have enough tokens in prefill, we will pick the prefill-optimized kernels.
                if (num_block_output_tokens >= min_tokens_for_dense_moe_kernels) {
                    status = gptoss_metal_command_buffer_encode_launch_expert_routing_metadata(
                        command_buffer,
                        &model->f32_expert_routing_metadata_fn,
                        &context->expert_activation_buffer,
                        /*expert_predictions_offset=*/0,
                        &context->expert_offset_buffer,
                        /*expert_offsets_offset=*/0,
                        &context->token_to_expert_routing_buffer,
                        /*intra_expert_offsets_offset=*/0,
                        num_block_output_tokens * model->num_active_experts,
                        model->num_experts);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_expert_routing_metadata kernel launch");
                        return status;
                    }
                    status = gptoss_metal_command_buffer_encode_launch_f32_scatter(
                        command_buffer,
                        &model->f32_scatter_e4_fn,
                        &context->rmsnorm_activation_buffer,
                        /*input_offset=*/0,
                        &context->expert_activation_buffer,
                        /*expert_predictions_offset=*/0,
                        &context->expert_offset_buffer,
                        /*expert_offsets_offset=*/0,
                        &context->token_to_expert_routing_buffer,
                        /*intra_expert_offsets_offset=*/0,
                        &context->swiglu_input_buffer,
                        /*output_offset=*/0,
                        model->embedding_dim,
                        num_block_output_tokens,
                        model->num_active_experts);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_scatter kernel launch");
                        return status;
                    } 
                    // Dense MoE SwiGLU matmul.
                    status = gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_dense_matmul_swiglu(
                        command_buffer,
                        &model->f32_mf4w_moe_dense_matmul_swiglu_fn,
                        &context->expert_offset_buffer,
                        /*expert_offsets_offset=*/0,
                        &context->swiglu_input_buffer,
                        /*input_offset=*/0,
                        &model->block_weight_buffers[n],
                        /*weight_block_offset=*/0,
                        &model->block_weight_buffers[n],
                        /*weight_scale_offset=*/model->mlp_swiglu_scale_offset,
                        &model->block_weight_buffers[n],
                        /*bias_offset=*/model->mlp_swiglu_bias_offset,
                        &context->swiglu_activation_buffer,
                        /*output_offset=*/0,
                        model->swiglu_limit,
                        /*expert_stride_bytes=*/model->per_expert_block_weight_size,
                        num_block_output_tokens,
                        model->num_experts,
                        model->embedding_dim,
                        2 * model->mlp_dim);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch");
                        return status;
                    }

                    // Dense MoE proj matmul.
                    status = gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_dense_matmul(
                        command_buffer,
                        &model->f32_mf4w_moe_dense_matmul_fn,
                        &context->expert_offset_buffer,
                        /*expert_offsets_offset=*/0,
                        &context->swiglu_activation_buffer,
                        /*input_offset=*/0,
                        &model->block_weight_buffers[n],
                        /*weight_block_offset=*/model->mlp_out_block_offset,
                        &model->block_weight_buffers[n],
                        /*weight_scale_offset=*/model->mlp_out_scale_offset,
                        &model->block_weight_buffers[n],
                        /*bias_offset=*/model->mlp_out_bias_offset,
                        &context->moe_activation_buffer,
                        /*output_offset=*/0,
                        /*expert_stride_bytes=*/model->per_expert_block_weight_size,
                        num_block_output_tokens,
                        model->num_experts,
                        model->mlp_dim,
                        model->embedding_dim);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch");
                        return status;
                    }
                    // Gather and accumulate.
                    status = gptoss_metal_command_buffer_encode_launch_f32_gather_and_accumulate_e4(
                        command_buffer,
                        &model->f32_gather_and_accumulate_e4_fn,
                        &context->moe_activation_buffer,
                        /*input_offset=*/0,
                        &context->expert_activation_buffer,
                        /*expert_predictions_offset=*/0,
                        &context->expert_offset_buffer,
                        /*expert_offsets_offset=*/0,
                        &context->token_to_expert_routing_buffer,
                        /*intra_expert_offsets_offset=*/0,
                        &context->residual_activation_buffer, 
                        /*output_offset=*/model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                        model->embedding_dim,
                        num_block_output_tokens,
                        model->num_active_experts);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_gather_and_accumulate_e4 kernel launch");
                        return status;
                    }

                } else {
                    status = gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul_swiglu(
                        command_buffer,
                        &model->f32_mf4w_moe_matmul_swiglu_fn,
                        model->mlp_swiglu_threadgroup_size,
                        &context->rmsnorm_activation_buffer,
                        /*input_offset=*/0,
                        &context->expert_activation_buffer,
                        /*expert_offset=*/0,
                        &model->block_weight_buffers[n],
                        /*weight_block_offset=*/0,
                        &model->block_weight_buffers[n],
                        /*weight_scale_offset=*/model->mlp_swiglu_scale_offset,
                        &model->block_weight_buffers[n],
                        /*bias_offset=*/model->mlp_swiglu_bias_offset,
                        &context->swiglu_activation_buffer,
                        /*output_offset=*/0,
                        &context->control_buffer,
                        /*control_offset=*/0,
                        model->swiglu_limit,
                        model->per_expert_block_weight_size,
                        num_block_output_tokens,
                        model->num_active_experts,
                        model->embedding_dim,
                        model->mlp_dim);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch");
                        return status;
                    }

                    status = gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul(
                        command_buffer,
                        &model->f32_mf4w_moe_matmul_fn,
                        model->mlp_out_threadgroup_size,
                        &context->swiglu_activation_buffer,
                        /*input_offset=*/0,
                        &context->expert_activation_buffer,
                        /*expert_offset=*/0,
                        &model->block_weight_buffers[n],
                        /*weight_block_offset=*/model->mlp_out_block_offset,
                        &model->block_weight_buffers[n],
                        /*weight_scale_offset=*/model->mlp_out_scale_offset,
                        &model->block_weight_buffers[n],
                        /*bias_offset=*/model->mlp_out_bias_offset,
                        &context->moe_activation_buffer,
                        /*output_offset=*/0,
                        &context->control_buffer,
                        /*control_offset=*/0,
                        model->per_expert_block_weight_size,
                        num_block_output_tokens,
                        model->num_active_experts,
                        model->mlp_dim,
                        model->embedding_dim);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch");
                        return status;
                    }

                    status = gptoss_metal_command_buffer_encode_launch_f32_accumulate(
                        command_buffer,
                        &model->f32_accumulate_e4_fn,
                        model->mlp_acc_threadgroup_size,
                        model->max_threadgroups,
                        &context->moe_activation_buffer,
                        /*input_offset=*/0,
                        &context->expert_activation_buffer,
                        /*expert_offset=*/0,
                        &context->residual_activation_buffer,
                        /*output_offset=*/model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                        &context->control_buffer,
                        /*control_offset=*/0,
                        model->embedding_dim,
                        num_block_output_tokens,
                        model->num_active_experts);
                    if (status != gptoss_status_success) {
                        GPTOSS_LOG_ERROR("failed to encode f32_accumulate kernel launch");
                        return status;
                    }
                }
            }
        }

        if (output_batch_size != 0) {
            status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                command_buffer,
                &model->f32_bf16w_rmsnorm_fn,
                &context->residual_activation_buffer,
                /*input_offset=*/model->embedding_dim * (input_batch_size - output_batch_size) * sizeof(float),
                &model->shared_weight_buffer,
                /*weight_offset=*/model->rmsnorm_weight_offset,
                &context->rmsnorm_activation_buffer,
                /*output_offset=*/0,
                &context->control_buffer,
                /*control_offset=*/0,
                /*num_tokens=*/output_batch_size,
                /*num_channels=*/model->embedding_dim,
                model->rmsnorm_epsilon);
            if (status != gptoss_status_success) {
                GPTOSS_LOG_ERROR("failed to encode f32_bf16w_rmsnorm kernel launch");
                return status;
            }

            status = gptoss_metal_command_buffer_encode_fill_buffer(
                command_buffer,
                &context->argmax_buffer,
                /*offset=*/0,
                /*size=*/sizeof(uint64_t) * output_batch_size,
                /*fill_value=*/0xFF);
            if (status != gptoss_status_success) {
                GPTOSS_LOG_ERROR("failed to encode fill buffer command");
                return status;
            }

            status = gptoss_metal_command_buffer_encode_launch_f32_bf16w_unembedding(
                command_buffer,
                &model->f32_bf16w_unembedding_fn,
                model->unembedding_threadgroup_size,
                model->max_threadgroups,
                &context->rmsnorm_activation_buffer,
                /*input_offset=*/0,
                &model->shared_weight_buffer,
                /*weight_offset=*/model->unembedding_weight_offset,
                &context->score_buffer,
                /*output_offset=*/0,
                &context->argmax_buffer,
                /*argmax_offset=*/0,
                &context->control_buffer,
                /*control_offset=*/0,
                /*num_tokens=*/output_batch_size,
                /*num_cols=*/model->embedding_dim,
                /*num_rows=*/model->vocabulary_size);
            if (status != gptoss_status_success) {
                GPTOSS_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch");
                return status;
            }
        }
    }
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_context_append_chars(
    gptoss_context_t context,
    const char* text,
    size_t text_length,
    size_t* num_tokens_out)
{
    enum gptoss_status status = gptoss_status_success;
    const struct gptoss_model* model = context->model;
    const struct gptoss_tokenizer* tokenizer = model->tokenizer;
    size_t num_appended_tokens = 0;
    while (text_length != 0) {
        if (context->num_tokens == context->max_tokens) {
            status = gptoss_status_context_overflow;
            break;
        }
        const char* tokens = tokenizer->tokens_ptr;
        uint32_t best_token = UINT32_MAX;
        uint32_t best_token_length = 0;
        for (size_t t = 0; t < tokenizer->num_text_tokens; t++) {
            uint16_t token_length;
            memcpy(&token_length, tokens, sizeof(uint16_t));
            tokens += sizeof(uint16_t);
            if (token_length <= text_length && token_length > best_token_length) {
                if (memcmp(text, tokens, token_length) == 0) {
                    if (token_length > best_token_length) {
                        best_token = (uint32_t) t;
                        best_token_length = token_length;
                    }
                }
            }
            tokens += token_length;
        }

        if (best_token == UINT32_MAX) {
            GPTOSS_LOG_ERROR("failed to tokenize text \"%.*s\"", (int) text_length, text);
            return gptoss_status_invalid_argument;
        }

        uint32_t* input_tokens = (uint32_t*) context->token_buffer.ptr;
        if (context->num_kv_tokens > context->num_tokens) {
            if (input_tokens[context->num_tokens] != best_token) {
                input_tokens[context->num_tokens] = best_token;

                // Invalidate the KV cache starting with the newly added token.
                context->num_kv_tokens = context->num_tokens;
            }
            context->num_tokens++;
        } else {
            input_tokens[context->num_tokens++] = best_token;
        }
        num_appended_tokens++;
        text += best_token_length;
        text_length -= best_token_length;
    }
    if (num_tokens_out != NULL) {
        *num_tokens_out = num_appended_tokens;
    }
    return status;
}

enum gptoss_status GPTOSS_ABI gptoss_context_append_tokens(
    gptoss_context_t context,
    size_t num_tokens,
    const uint32_t* tokens)
{
    const struct gptoss_model* model = context->model;

    // Validate all tokens
    for (size_t t = 0; t < num_tokens; t++) {
        const uint32_t token = tokens[t];
        if (token >= model->vocabulary_size) {
            GPTOSS_LOG_ERROR("token %" PRIu32 " at index %zu is out of bounds for vocabulary size %" PRIu32,
                token, t, context->model->vocabulary_size);
            return gptoss_status_invalid_argument;
        }
    }

    enum gptoss_status status = gptoss_status_success;
    uint32_t* input_tokens = (uint32_t*) context->token_buffer.ptr;
    while (num_tokens != 0) {
        if (context->num_tokens == context->max_tokens) {
            status = gptoss_status_context_overflow;
            break;
        }

        if (context->num_kv_tokens > context->num_tokens) {
            const size_t num_tokens_to_verify = math_min(context->num_kv_tokens - context->num_tokens, num_tokens);
            size_t num_verified_tokens = 0;
            for (; num_verified_tokens < num_tokens_to_verify; num_verified_tokens++) {
                if (input_tokens[context->num_tokens + num_verified_tokens] != tokens[num_verified_tokens]) {
                    // Invalidate the KV cache starting with the newly added tokens.
                    context->num_kv_tokens = context->num_tokens + num_verified_tokens;
                    break;
                }
            }

            context->num_tokens += num_verified_tokens;
            tokens += num_verified_tokens;
            num_tokens -= num_verified_tokens;
        } else {
            const size_t num_tokens_to_copy = math_min(context->max_tokens - context->num_tokens, num_tokens);
            memcpy(input_tokens + context->num_tokens, tokens, num_tokens_to_copy * sizeof(uint32_t));
            context->num_tokens += num_tokens_to_copy;
            tokens += num_tokens_to_copy;
            num_tokens -= num_tokens_to_copy;
        }
    }

    return status;
}

enum gptoss_status GPTOSS_ABI gptoss_context_process(
    gptoss_context_t context)
{
    if (context->num_tokens > context->num_kv_tokens) {
        struct gptoss_metal_command_buffer command_buffer = {0};

        enum gptoss_status status = gptoss_metal_command_buffer_create(&context->model->command_queue, &command_buffer);
        if (status != gptoss_status_success) {
            goto cleanup;
        }

        struct gptoss_control* control = (struct gptoss_control*) context->control_buffer.ptr;
        control->abort = 0;

        status = process_tokens(
            context,
            &command_buffer,
            /*input_tokens_offset=*/context->num_kv_tokens,
            /*num_input_tokens=*/context->num_tokens - context->num_kv_tokens,
            /*num_output_tokens=*/0);
        if (status != gptoss_status_success) {
            goto cleanup;
        }

        status = gptoss_metal_command_buffer_commit(&command_buffer);
        if (status != gptoss_status_success) {
            goto cleanup;
        }

        status = gptoss_metal_command_buffer_wait_completion(&command_buffer, NULL);
        if (status != gptoss_status_success) {
            goto cleanup;
        }

        context->num_kv_tokens = context->num_tokens;

cleanup:
        gptoss_metal_command_buffer_release(&command_buffer);
        return status;
    }
    
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_context_sample(
    gptoss_context_t context,
    float temperature,
    uint64_t seed,
    size_t max_tokens,
    uint32_t* tokens_out,
    size_t* num_tokens_out)
{
    enum gptoss_status status = gptoss_status_success;
    const struct gptoss_model* model = context->model;
    struct gptoss_metal_command_buffer command_buffer = {0};

    *num_tokens_out = 0;

    const uint32_t num_original_tokens = context->num_tokens;

    status = gptoss_metal_command_buffer_create(&context->model->command_queue, &command_buffer);
    if (status != gptoss_status_success) {
        goto cleanup;
    }

    struct gptoss_control* control = (struct gptoss_control*) context->control_buffer.ptr;
    control->abort = 0;

    for (size_t t = 0; t < max_tokens; t++) {
        if (context->num_kv_tokens < context->num_tokens) {
            status = process_tokens(
                context,
                &command_buffer,
                /*input_tokens_offset=*/context->num_kv_tokens,
                /*num_input_tokens=*/context->num_tokens - context->num_kv_tokens,
                /*num_output_tokens=*/1);
            context->num_kv_tokens = context->num_tokens;
        } else {
            status = process_tokens(
                context,
                &command_buffer,
                /*input_tokens_offset=*/context->num_tokens - 1,
                /*num_input_tokens=*/1,
                /*num_output_tokens=*/1);
        }
        if (status != gptoss_status_success) {
            goto cleanup;
        }

        if (temperature != 0.0f) {
            assert(context->num_processed_tokens != 0);
            uint32_t num_threadgroups = 0;
            uint32_t num_dims_per_threadgroup = 0;
            status = gptoss_metal_command_buffer_encode_launch_f32_softmax(
                &command_buffer,
                &model->f32_softmax_fn,
                /*threadgroup_size=*/512,
                model->max_threadgroups,
                &context->score_buffer,
                /*score_offset=*/0,
                &context->argmax_buffer,
                /*argmax_offset=*/0,
                &context->prob_buffer,
                /*prob_offset=*/0,
                &context->sum_buffer,
                /*sum_offset=*/0,
                &context->control_buffer,
                /*control_offset=*/0,
                model->vocabulary_size,
                /*num_tokens=*/1,
                temperature,
                &num_threadgroups,
                &num_dims_per_threadgroup);
            if (status != gptoss_status_success) {
                GPTOSS_LOG_ERROR("failed to encode f32_softmax kernel launch");
                goto cleanup;
            }

            status = gptoss_metal_command_buffer_encode_launch_f32_sample(
                &command_buffer,
                &model->f32_sample_fn,
                /*min_threadgroup_size=*/512,
                &context->prob_buffer,
                /*prob_offset=*/0,
                &context->sum_buffer,
                /*sum_offset=*/0,
                &context->token_buffer,
                /*token_offset=*/context->num_tokens * sizeof(uint32_t),
                &context->control_buffer,
                /*control_offset=*/0,
                /*rng_seed=*/seed + UINT64_C(0x123456789ABCDEF),
                /*rng_offset=*/context->num_tokens,
                /*num_blocks=*/num_threadgroups,
                /*num_channels=*/model->vocabulary_size,
                /*num_channels_per_block=*/num_dims_per_threadgroup);
            if (status != gptoss_status_success) {
                GPTOSS_LOG_ERROR("failed to encode f32_sample kernel launch");
                goto cleanup;
            }
        } else {
            status = gptoss_metal_command_buffer_encode_copy_buffer(
                &command_buffer,
                &context->argmax_buffer,
                /*input_offset=*/0,
                &context->token_buffer,
                /*output_offset=*/context->num_tokens * sizeof(uint32_t),
                /*size=*/sizeof(uint32_t));
            if (status != gptoss_status_success) {
                GPTOSS_LOG_ERROR("failed to encode copy buffer");
                goto cleanup;
            }
        }
        context->num_tokens += 1;
        context->num_kv_tokens = context->num_tokens;
    }

    gptoss_metal_command_buffer_commit(&command_buffer);
    gptoss_metal_command_buffer_wait_completion(&command_buffer, NULL);

    const uint32_t* token_ptr = (const uint32_t*) context->token_buffer.ptr;
    const uint32_t num_generated_tokens = context->num_tokens - num_original_tokens;
    memcpy(tokens_out, token_ptr + num_original_tokens, num_generated_tokens * sizeof(uint32_t));
    *num_tokens_out = num_generated_tokens;

cleanup:
    gptoss_metal_command_buffer_release(&command_buffer);
    return status;
}

enum gptoss_status GPTOSS_ABI gptoss_context_reset(
    gptoss_context_t context)
{
    context->num_tokens = 0;

    // Note: context->num_kv_tokens is not reset and context->input_tokens_buffer is not cleared.
    // If the subsequently added tokens match the tokens already in the KV cache, we reuse the KV cache.

    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_context_retain(
    gptoss_context_t context)
{
    atomic_fetch_add_explicit(&context->ref_count, 1, memory_order_relaxed);
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_context_release(
    gptoss_context_t context)
{
    if (context != NULL) {
        if (atomic_fetch_sub_explicit(&context->ref_count, 1, memory_order_acq_rel) == 1) {
            // Activation buffers
            gptoss_metal_buffer_release(&context->residual_activation_buffer);
            gptoss_metal_buffer_release(&context->rmsnorm_activation_buffer);
            gptoss_metal_buffer_release(&context->qkv_activation_buffer);
            gptoss_metal_buffer_release(&context->sdpa_activation_buffer);
            gptoss_metal_buffer_release(&context->gate_activation_buffer);
            gptoss_metal_buffer_release(&context->expert_activation_buffer);
            gptoss_metal_buffer_release(&context->swiglu_activation_buffer);
            gptoss_metal_buffer_release(&context->moe_activation_buffer);
            gptoss_metal_buffer_release(&context->expert_offset_buffer);
            gptoss_metal_buffer_release(&context->token_to_expert_routing_buffer);
            gptoss_metal_buffer_release(&context->swiglu_input_buffer);

            // Input/output buffers
            gptoss_metal_buffer_release(&context->control_buffer);
            gptoss_metal_buffer_release(&context->token_buffer);
            gptoss_metal_buffer_release(&context->score_buffer);
            gptoss_metal_buffer_release(&context->prob_buffer);
            gptoss_metal_buffer_release(&context->sum_buffer);
            gptoss_metal_buffer_release(&context->argmax_buffer);
            gptoss_metal_buffer_release(&context->kvcache_buffer);

            gptoss_model_release(context->model);

            memset(context, 0, sizeof(struct gptoss_context));
            free(context);
        }
    }
    return gptoss_status_success;
}
