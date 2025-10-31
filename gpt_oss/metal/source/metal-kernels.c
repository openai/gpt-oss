#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <internal/kernel-args.h>
#include <internal/log.h>
#include <internal/math.h>
#include <internal/metal.h>
#include <internal/metal-kernels.h>


enum gptoss_status gptoss_metal_command_buffer_encode_launch_u32_fill_random(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* u32_fill_random_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint64_t num_elements,
    uint64_t rng_seed,
    uint64_t rng_offset)
{
    if (command_buffer->object == NULL || u32_fill_random_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = u32_fill_random_fn->max_threadgroup_threads;
    } else if (threadgroup_size > u32_fill_random_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_u32_fill_random_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .seed = rng_seed,
        .offset = rng_offset,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, u32_fill_random_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        1, &output_buffer, &output_offset,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_fill_random(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_fill_random_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint64_t num_elements,
    uint64_t rng_seed,
    uint64_t rng_offset,
    float rng_min,
    float rng_max)
{
    if (command_buffer->object == NULL || f32_fill_random_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_fill_random_fn->max_threadgroup_threads;
    } else if (threadgroup_size > f32_fill_random_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    if (rng_min >= rng_max) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_f32_fill_random_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .seed = rng_seed,
        .offset = rng_offset,
        .scale = (rng_max - rng_min) * 0x1.0p-32f,
        .bias = (rng_min + rng_max) * 0.5f,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_fill_random_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        1, &output_buffer, &output_offset,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_bf16_fill_random(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* bf16_fill_random_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint64_t num_elements,
    uint64_t rng_seed,
    uint64_t rng_offset,
    float rng_min,
    float rng_max)
{
    if (command_buffer->object == NULL || bf16_fill_random_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = bf16_fill_random_fn->max_threadgroup_threads;
    } else if (threadgroup_size > bf16_fill_random_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    if (rng_min >= rng_max) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_f32_fill_random_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .seed = rng_seed,
        .offset = rng_offset,
        .scale = (rng_max - rng_min) * 0x1.0p-32f,
        .bias = (rng_min + rng_max) * 0.5f,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, bf16_fill_random_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        1, &output_buffer, &output_offset,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_mf4_f32_convert(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* mf4_f32_convert_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* block_buffer,
    const struct gptoss_metal_buffer* scale_buffer,
    const struct gptoss_metal_buffer* output_buffer,
    uint64_t num_elements)
{
    if (command_buffer->object == NULL || mf4_f32_convert_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_elements % 32 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = mf4_f32_convert_fn->max_threadgroup_threads;
    } else if (threadgroup_size > mf4_f32_convert_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements / 32;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_convert_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, mf4_f32_convert_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        3, (const struct gptoss_metal_buffer *[]) {block_buffer, scale_buffer, output_buffer}, NULL,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_bf16_f32_embeddings(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* bf16_f32_embeddings_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* token_buffer,
    size_t token_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_channels)
{
    if (command_buffer->object == NULL || bf16_f32_embeddings_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels % 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = bf16_f32_embeddings_fn->max_threadgroup_threads;
    } else if (threadgroup_size > bf16_f32_embeddings_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const uint32_t num_vecs = num_channels / 4;
    const struct gptoss_embeddings_args args = {
        .num_vecs = num_vecs,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, bf16_f32_embeddings_fn,
        threadgroup_size, 1, 1,
        num_tokens, 1, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {token_buffer, weight_buffer, output_buffer, control_buffer},
        (const size_t[]) {token_offset, weight_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_rmsnorm_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_channels,
    float epsilon)
{
    if (command_buffer->object == NULL || f32_bf16w_rmsnorm_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels % 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (f32_bf16w_rmsnorm_fn->max_threadgroup_threads < 1024) {
        return gptoss_status_unsupported_system;
    }

    if (f32_bf16w_rmsnorm_fn->simdgroup_threads != 32) {
        return gptoss_status_unsupported_system;
    }

    const uint32_t num_vecs = num_channels / 4;
    const struct gptoss_rmsnorm_args args = {
        .num_vecs = num_vecs,
        .num_channels = (float) num_channels,
        .epsilon = epsilon,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_rmsnorm_fn,
        /*threadgroup_size=*/1024, 1, 1,
        num_tokens, 1, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, output_buffer, control_buffer},
        (const size_t[]) {input_offset, weight_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_matmul_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_bf16w_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_bf16w_matmul_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_bf16w_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_bf16w_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 4 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: number of columns (%" PRIu32 ") is not divisible by 4",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_bf16w_matmul_fn->simdgroup_threads;
    if (num_rows % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: number of rows (%" PRIu32 ") is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_matmul_args args = {
        .num_column_vecs = num_cols / 4,
        .num_rows = num_rows,
        .add = 0,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_matmul_fn,
        threadgroup_size, 1, 1,
        num_rows / num_simdgroups, num_tokens, 1,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, bias_buffer, output_buffer, control_buffer},
        (const size_t[]) {input_offset, weight_offset, bias_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul_qkv(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_matmul_qkv_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* kv_buffer,
    size_t kv_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t attn_head_dim,
    uint32_t token_offset,
    uint32_t max_tokens,
    float rope_base,
    float interpolation_scale,
    float yarn_offset,
    float yarn_scale,
    float yarn_multiplier)
{
    if (command_buffer->object == NULL || f32_bf16w_matmul_qkv_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_bf16w_matmul_qkv_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_bf16w_matmul_qkv_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_bf16w_matmul_qkv_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 4 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch: number of columns (%" PRIu32 ") is not divisible by 4",
            num_cols);
        return gptoss_status_invalid_argument;
    }

    if (num_q_heads != 64) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch: number of Q heads (%" PRIu32 ") must be 64",
            num_q_heads);
        return gptoss_status_invalid_argument;
    }
    if (num_kv_heads != 8) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch: number of KV heads (%" PRIu32 ") must be 8",
            num_kv_heads);
        return gptoss_status_invalid_argument;
    }
    if (attn_head_dim != 64) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch: attention head dimension (%" PRIu32 ") must be 64",
            attn_head_dim);
        return gptoss_status_invalid_argument;
    }

    const size_t num_simdgroups = threadgroup_size / f32_bf16w_matmul_qkv_fn->simdgroup_threads;
    const uint32_t num_rows = (num_q_heads + 2 * num_kv_heads) * attn_head_dim;
    if (num_rows % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_qkv kernel launch: number of rows (%" PRIu32 ") is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_qkv_args args = {
        .num_column_vecs = num_cols / 4,
        .num_rows = num_rows,
        .token_offset = token_offset,
        .freq_scale = -logf(rope_base) / (float) (int32_t) (attn_head_dim / 2),
        .interpolation_scale = interpolation_scale,
        .yarn_offset = yarn_offset,
        .yarn_scale = yarn_scale,
        .yarn_multiplier = yarn_multiplier,
        .max_tokens = max_tokens,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_matmul_qkv_fn,
        threadgroup_size, 1, 1,
        num_rows / num_simdgroups, num_tokens, 1,
        sizeof(args), &args,
        6,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, bias_buffer, output_buffer, kv_buffer, control_buffer},
        (const size_t[]) {input_offset, weight_offset, bias_offset, output_offset, kv_offset, control_offset},
        /*threadgroup_buffer_size=*/num_simdgroups * sizeof(float));
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul_add(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_matmul_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_bf16w_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_bf16w_matmul_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_bf16w_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_bf16w_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 4 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: number of columns (%" PRIu32 ") is not divisible by 4",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_bf16w_matmul_fn->simdgroup_threads;
    if (num_rows % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: number of rows (%" PRIu32 ") is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_matmul_args args = {
        .num_column_vecs = num_cols / 4,
        .num_rows = num_rows,
        .add = 1,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_matmul_fn,
        threadgroup_size, 1, 1,
        num_rows / num_simdgroups, num_tokens, 1,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, bias_buffer, output_buffer, control_buffer},
        (const size_t[]) {input_offset, weight_offset, bias_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status _gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_impl(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_dense_matmul_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset, 
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t Bm,
    uint32_t Bn,
    uint32_t Bk,
    uint32_t Sg_Bm,
    uint32_t Sg_Bn)
{

    if (command_buffer->object == NULL || f32_bf16w_dense_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (num_cols % 8 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: number of columns (%" PRIu32 ") is not divisible by 8",
                         num_cols);
        return gptoss_status_invalid_argument;
    }
    if (num_rows % 8 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: number of rows (%" PRIu32 ") is not divisible by 8",
                         num_rows);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_dense_matmul_args args = {
        .m = num_tokens,
        .n = num_rows,
        .k = num_cols,
    };
    const size_t threads_per_simdgroup = f32_bf16w_dense_matmul_fn->simdgroup_threads;
    const uint32_t m = args.m;
    const uint32_t n = args.n;
    const uint32_t k = args.k;
    if (Bm % Sg_Bm != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: Bm (%" PRIu32 ") is not divisible by Sg_Bm (%" PRIu32 ")",
                         Bm, Sg_Bm);
        return gptoss_status_invalid_argument;
    }
    if (Bn % Sg_Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: Bn (%" PRIu32 ") is not divisible by Sg_Bn (%" PRIu32 ")",
                         Bn, Sg_Bn);
        return gptoss_status_invalid_argument;
    }
    const size_t threadgroup_size_x = (Bm / Sg_Bm) * (Bn / Sg_Bn) * threads_per_simdgroup;
    const size_t threadgroup_size_y = 1;
    const size_t threadgroup_size_z = 1;
    const size_t total_threadgroup_size = threadgroup_size_x * threadgroup_size_y * threadgroup_size_z;
    if (total_threadgroup_size > f32_bf16w_dense_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: total threadgroup size (%zu) exceeds supported maximum (%zu)",
                         total_threadgroup_size, f32_bf16w_dense_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }
    if (n % Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: n (%" PRIu32 ") is not divisible by Bn (%" PRIu32 ")",
                         n, Bn);
        return gptoss_status_invalid_argument;
    }
    if (k % Bk != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: k (%" PRIu32 ") is not divisible by Bk (%" PRIu32 ")",
                         k, Bk);
        return gptoss_status_invalid_argument;
    }
    const size_t grid_x = n / Bn;
    const size_t grid_y = math_ceil_div(m, Bm);
    const size_t grid_z = 1;

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_dense_matmul_fn,
        threadgroup_size_x, threadgroup_size_y, threadgroup_size_z,
        grid_x, grid_y, grid_z,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]){input_buffer, weight_buffer, bias_buffer, output_buffer, control_buffer},
        (const size_t[]){input_offset, weight_offset, bias_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_qkv(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_dense_matmul_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* kv_buffer,
    size_t kv_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t max_tokens,
    uint32_t token_offset)
{
    if (command_buffer->object == NULL || f32_bf16w_dense_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (num_cols % 8 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: number of columns (%" PRIu32 ") is not divisible by 8",
                         num_cols);
        return gptoss_status_invalid_argument;
    }
    if (num_rows % 8 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: number of rows (%" PRIu32 ") is not divisible by 8",
                         num_rows);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_dense_matmul_qkv_args args = {
        .m = num_tokens,
        .n = num_rows,
        .k = num_cols,
        .max_tokens = max_tokens,
        .token_offset = token_offset,
    };
    const size_t threads_per_simdgroup = f32_bf16w_dense_matmul_fn->simdgroup_threads;
    const uint32_t m = args.m;
    const uint32_t n = args.n;
    const uint32_t k = args.k;
    if (QKV_Bm % QKV_Sg_Bm != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: Bm (%" PRIu32 ") is not divisible by Sg_Bm (%" PRIu32 ")",
                         QKV_Bm, QKV_Sg_Bm);
        return gptoss_status_invalid_argument;
    }
    if (QKV_Bn % QKV_Sg_Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: Bn (%" PRIu32 ") is not divisible by Sg_Bn (%" PRIu32 ")",
                         QKV_Bn, QKV_Sg_Bn);
        return gptoss_status_invalid_argument;
    }
    const size_t threadgroup_size_x = (QKV_Bm / QKV_Sg_Bm) * (QKV_Bn / QKV_Sg_Bn) * threads_per_simdgroup;
    const size_t threadgroup_size_y = 1;
    const size_t threadgroup_size_z = 1;
    const size_t total_threadgroup_size = threadgroup_size_x * threadgroup_size_y * threadgroup_size_z;
    if (total_threadgroup_size > f32_bf16w_dense_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: total threadgroup size (%zu) exceeds supported maximum (%zu)",
                         total_threadgroup_size, f32_bf16w_dense_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }
    if (n % QKV_Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: n (%" PRIu32 ") is not divisible by Bn (%" PRIu32 ")",
                         n, QKV_Bn);
        return gptoss_status_invalid_argument;
    }
    if (k % QKV_Bk != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_dense_matmul kernel launch: k (%" PRIu32 ") is not divisible by Bk (%" PRIu32 ")",
                         k, QKV_Bk);
        return gptoss_status_invalid_argument;
    }
    const size_t grid_x = n / QKV_Bn;
    const size_t grid_y = math_ceil_div(m, QKV_Bm);
    const size_t grid_z = 1;

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_dense_matmul_fn,
        threadgroup_size_x, threadgroup_size_y, threadgroup_size_z,
        grid_x, grid_y, grid_z,
        sizeof(args), &args,
        6,
        (const struct gptoss_metal_buffer *[]){input_buffer, weight_buffer, bias_buffer, output_buffer, kv_buffer, control_buffer},
        (const size_t[]){input_offset, weight_offset, bias_offset, output_offset, kv_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_attn_output(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_dense_matmul_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    return _gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_impl(
        command_buffer, f32_bf16w_dense_matmul_fn, input_buffer, input_offset,
        weight_buffer, weight_offset, bias_buffer, bias_offset, output_buffer,
        output_offset, control_buffer, control_offset, num_tokens, num_cols, num_rows, ATTN_OUTPUT_Bm,
        ATTN_OUTPUT_Bn, ATTN_OUTPUT_Bk, ATTN_OUTPUT_Sg_Bm, ATTN_OUTPUT_Sg_Bn);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_mlp_gate(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_dense_matmul_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    return _gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_impl(
        command_buffer, f32_bf16w_dense_matmul_fn, input_buffer, input_offset,
        weight_buffer, weight_offset, bias_buffer, bias_offset, output_buffer,
        output_offset, control_buffer, control_offset, num_tokens, num_cols,
        num_rows, MLP_GATE_Bm, MLP_GATE_Bn, MLP_GATE_Bk, MLP_GATE_Sg_Bm,
        MLP_GATE_Sg_Bn);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_unembedding(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_unembedding_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* argmax_buffer,
    size_t argmax_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_bf16w_unembedding_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_bf16w_unembedding_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_bf16w_unembedding_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_bf16w_unembedding_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 4 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch: number of columns (%" PRIu32 ") is not divisible by 4",
            num_cols);
        return gptoss_status_invalid_argument;
    }

    const size_t num_simdgroups = threadgroup_size / f32_bf16w_unembedding_fn->simdgroup_threads;
    const size_t num_rows_per_threadgroup = math_ceil_div(num_rows, max_threadgroups * num_simdgroups) * num_simdgroups;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_rows, num_rows_per_threadgroup));
    const struct gptoss_unembedding_args args = {
        .num_column_vecs = num_cols / 4,
        .num_rows_per_threadgroup = num_rows_per_threadgroup,
        .num_rows = num_rows,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_unembedding_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, num_tokens, 1,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, output_buffer, argmax_buffer, control_buffer},
        (const size_t[]) {input_offset, weight_offset, output_offset, argmax_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul_swiglu(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_mf4w_moe_matmul_swiglu_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_buffer,
    size_t expert_offset,
    const struct gptoss_metal_buffer* weight_block_buffer,
    size_t weight_block_offset,
    const struct gptoss_metal_buffer* weight_scale_buffer,
    size_t weight_scale_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    float swiglu_limit,
    uint32_t expert_stride,
    uint32_t num_tokens,
    uint32_t num_active_experts,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_mf4w_moe_matmul_swiglu_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = 2 * f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_mf4w_moe_matmul_swiglu_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_mf4w_moe_matmul_swiglu_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    } else if (threadgroup_size % (2 * f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads)) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: threadgroup size (%zu) is not divisible by simdgroup size (%zu) multiplied by 2X",
            threadgroup_size, f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 32 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: number of columns (%" PRIu32 ") is not divisible by 32",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads;
    if ((2 * num_rows) % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: "
            "the number of rows (%" PRIu32 ") multiplied by 2X is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_moe_matmul_swiglu_args args = {
        .num_column_vecs = num_cols / 32,
        .num_rows = num_rows,
        .num_active_experts = num_active_experts,
        .weight_expert_stride = expert_stride,
        .output_expert_stride = num_rows * num_tokens,
        .swiglu_min = -swiglu_limit,
        .swiglu_max = swiglu_limit,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_mf4w_moe_matmul_swiglu_fn,
        threadgroup_size, 1, 1,
        (2 * num_rows) / num_simdgroups, num_tokens, num_active_experts,
        sizeof(args), &args,
        7,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_buffer, weight_block_buffer, weight_scale_buffer, bias_buffer, output_buffer, control_buffer},
        (const size_t[]) {input_offset, expert_offset, weight_block_offset, weight_scale_offset, bias_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_mf4w_moe_matmul_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_buffer,
    size_t expert_offset,
    const struct gptoss_metal_buffer* weight_block_buffer,
    size_t weight_block_offset,
    const struct gptoss_metal_buffer* weight_scale_buffer,
    size_t weight_scale_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t expert_stride,
    uint32_t num_tokens,
    uint32_t num_active_experts,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_mf4w_moe_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_mf4w_moe_matmul_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_mf4w_moe_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_mf4w_moe_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    } else if (threadgroup_size % f32_mf4w_moe_matmul_fn->simdgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: threadgroup size (%zu) is not divisible by simdgroup size (%zu)",
            threadgroup_size, f32_mf4w_moe_matmul_fn->simdgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 32 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: number of columns (%" PRIu32 ") is not divisible by 32",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_mf4w_moe_matmul_fn->simdgroup_threads;
    if (num_rows % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: "
            "the number of rows (%" PRIu32 ") is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_moe_matmul_args args = {
        .num_column_vecs = num_cols / 32,
        .num_rows = num_rows,
        .num_active_experts = num_active_experts,
        .input_expert_stride = num_tokens * (num_cols / 32),
        .weight_expert_stride = expert_stride,
        .output_expert_stride = num_rows * num_tokens,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_mf4w_moe_matmul_fn,
        threadgroup_size, 1, 1,
        num_rows / num_simdgroups, num_tokens, num_active_experts,
        sizeof(args), &args,
        7,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_buffer, weight_block_buffer, weight_scale_buffer, bias_buffer, output_buffer, control_buffer},
        (const size_t[]) {input_offset, expert_offset, weight_block_offset, weight_scale_offset, bias_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_rope(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_rope_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* activations_buffer,
    size_t activations_offset,
    const struct gptoss_metal_buffer* kv_buffer,
    size_t kv_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    float rope_base,
    float interpolation_scale,
    float yarn_offset,
    float yarn_scale,
    float yarn_multiplier,
    uint32_t num_tokens,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t attn_head_dim,
    uint32_t max_tokens,
    uint32_t token_offset)
{
    if (command_buffer->object == NULL || f32_rope_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_rope_fn->max_threadgroup_threads;
    } else if (threadgroup_size > f32_rope_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_simdgroups = threadgroup_size / f32_rope_fn->simdgroup_threads;
    const uint32_t num_qk_heads = num_q_heads + num_kv_heads;
    if (num_qk_heads % num_simdgroups != 0) {
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_rope_args args = {
        .token_stride = (num_q_heads + 2 * num_kv_heads) * (attn_head_dim / 2),
        .token_offset = token_offset,
        .max_tokens = max_tokens,
        .freq_scale = -logf(rope_base) / (float) (int32_t) (attn_head_dim / 2),
        .interpolation_scale = interpolation_scale,
        .yarn_offset = yarn_offset,
        .yarn_scale = yarn_scale,
        .yarn_multiplier = yarn_multiplier,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_rope_fn,
        threadgroup_size, 1, 1,
        num_qk_heads / num_simdgroups, num_tokens, 1,
        sizeof(args), &args,
        3,
        (const struct gptoss_metal_buffer *[]) {activations_buffer, kv_buffer, control_buffer},
        (const size_t[]) {activations_offset, kv_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_expert_routing_metadata(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* expert_routing_metadata_fn,
    const struct gptoss_metal_buffer* expert_predictions_buffer,
    size_t expert_predictions_offset,
    const struct gptoss_metal_buffer* expert_offsets_buffer,
    size_t expert_offsets_offset,
    const struct gptoss_metal_buffer* intra_expert_offsets_buffer,
    size_t intra_expert_offsets_offset,
    uint32_t num_tokens,
    uint32_t num_experts)
{
    if (command_buffer->object == NULL || expert_routing_metadata_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }
    
    const struct gptoss_expert_routing_metadata_args args = {
        .tokens = num_tokens,
        .num_experts = num_experts,
    };
    const uint32_t threadgroup_size = 256;
    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, expert_routing_metadata_fn,
        threadgroup_size, 1, 1,
        /*num_threadgroups_x=*/1, /*num_threadgroups_y=*/1, /*num_threadgroups_z=*/1,
        sizeof(args), &args,
        3,
        (const struct gptoss_metal_buffer *[]) {expert_predictions_buffer, expert_offsets_buffer, intra_expert_offsets_buffer},
        (const size_t[]) {expert_predictions_offset, expert_offsets_offset, intra_expert_offsets_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_scatter(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_scatter_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_predictions_buffer,
    size_t expert_predictions_offset,
    const struct gptoss_metal_buffer* expert_offsets_buffer,
    size_t expert_offsets_offset,
    const struct gptoss_metal_buffer* intra_expert_offsets_buffer,
    size_t intra_expert_offsets_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_channels,
    uint32_t num_tokens,
    uint32_t num_active_experts)
{
    if (command_buffer->object == NULL || f32_scatter_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels % 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_channels / 4;
    const size_t tgx = math_min(num_vecs, 64);
    const size_t tgy = 1;
    const size_t tgz = 1;
    const size_t grid_x = math_ceil_div(num_vecs, tgx);
    const size_t grid_y = num_tokens;
    const size_t grid_z = 1;
    const size_t total_threadgroup_size = tgx * tgy * tgz;
    if (total_threadgroup_size > f32_scatter_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }
    const struct gptoss_scatter_args args = {
        .tokens = num_tokens,
        .active_experts_per_token = num_active_experts,
        .token_stride = num_channels,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_scatter_fn,
        tgx, tgy, tgz,
        grid_x, grid_y, grid_z,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_predictions_buffer, expert_offsets_buffer, intra_expert_offsets_buffer, output_buffer},
        (const size_t[]) {input_offset, expert_predictions_offset, expert_offsets_offset, intra_expert_offsets_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_gather_and_accumulate_e4(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_gather_and_accumulate_e4_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_predictions_buffer,
    size_t expert_predictions_offset,
    const struct gptoss_metal_buffer* expert_offsets_buffer,
    size_t expert_offsets_offset,
    const struct gptoss_metal_buffer* intra_expert_offsets_buffer,
    size_t intra_expert_offsets_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_channels,
    uint32_t num_tokens,
    uint32_t num_active_experts) 
{
        if (command_buffer->object == NULL || f32_gather_and_accumulate_e4_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels % 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_channels / 4;
    const size_t tgx = math_min(num_vecs, 64);
    const size_t tgy = 1;
    const size_t tgz = 1;
    const size_t grid_x = math_ceil_div(num_vecs, tgx);
    const size_t grid_y = num_tokens;
    const size_t grid_z = 1;
    const size_t total_threadgroup_size = tgx * tgy * tgz;
    if (total_threadgroup_size > f32_gather_and_accumulate_e4_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }
    const struct gptoss_gather_args args = {
        .tokens = num_tokens,
        .active_experts_per_token = num_active_experts,
        .token_stride = num_channels,
    };
    
    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_gather_and_accumulate_e4_fn,
        tgx, tgy, tgz,
        grid_x, grid_y, grid_z,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_predictions_buffer, expert_offsets_buffer, intra_expert_offsets_buffer, output_buffer},
        (const size_t[]) {input_offset, expert_predictions_offset, expert_offsets_offset, intra_expert_offsets_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_dense_matmul_swiglu(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_mf4w_moe_dense_matmul_swiglu_fn,
    const struct gptoss_metal_buffer* expert_offsets_buffer,
    size_t expert_offsets_offset,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_block_buffer,
    size_t weight_block_offset,
    const struct gptoss_metal_buffer* weight_scale_buffer,
    size_t weight_scale_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    float swiglu_limit,
    uint32_t expert_stride_bytes,
    uint32_t num_tokens,
    uint32_t num_experts,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_mf4w_moe_dense_matmul_swiglu_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_cols % 32 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch: number of columns (%" PRIu32 ") is not divisible by 32",
            num_cols);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_moe_dense_matmul_swiglu_args args = {
        .n = num_rows,
        .k = num_cols,
        .weight_blocks_expert_stride_bytes = expert_stride_bytes,
        .weight_scales_expert_stride_bytes = expert_stride_bytes,
        .bias_expert_stride_bytes = expert_stride_bytes,
        .swiglu_min = -swiglu_limit,
        .swiglu_max = swiglu_limit,
    };
    const size_t threads_per_simdgroup = f32_mf4w_moe_dense_matmul_swiglu_fn->simdgroup_threads;
    const uint32_t m = num_tokens;
    const uint32_t n = args.n;
    const uint32_t k = args.k;
    const uint32_t Bm = MOE_DENSE_MATMUL_SWIGLU_Bm;
    const uint32_t Bn = MOE_DENSE_MATMUL_SWIGLU_Bn;
    const uint32_t Bk = MOE_DENSE_MATMUL_SWIGLU_Bk;
    const uint32_t Sg_Bm = MOE_DENSE_MATMUL_SWIGLU_Sg_Bm;
    const uint32_t Sg_Bn = MOE_DENSE_MATMUL_SWIGLU_Sg_Bn;
    if (Bm % Sg_Bm != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch: Bm (%" PRIu32 ") is not divisible by Sg_Bm (%" PRIu32 ")",
            Bm, Sg_Bm);
        return gptoss_status_invalid_argument;
    }
    if (Bn % Sg_Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch: Bn (%" PRIu32 ") is not divisible by Sg_Bn (%" PRIu32 ")",
            Bn, Sg_Bn);
        return gptoss_status_invalid_argument;
    }

    const size_t threadgroup_size_x = (Bm / Sg_Bm) * (Bn / Sg_Bn) * threads_per_simdgroup;
    const size_t threadgroup_size_y = 1;
    const size_t threadgroup_size_z = 1;
    const size_t total_threadgroup_size = threadgroup_size_x * threadgroup_size_y * threadgroup_size_z;
    if (total_threadgroup_size > f32_mf4w_moe_dense_matmul_swiglu_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch: total threadgroup size (%zu) exceeds supported maximum (%zu)",
            total_threadgroup_size, f32_mf4w_moe_dense_matmul_swiglu_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }
    if (n % Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch: n (%" PRIu32 ") is not divisible by Bn (%" PRIu32 ")",
            n, Bn);
        return gptoss_status_invalid_argument;
    }
    if (k % Bk != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul_swiglu kernel launch: k (%" PRIu32 ") is not divisible by Bk (%" PRIu32 ")",
            k, Bk);
        return gptoss_status_invalid_argument;
    }
    const size_t grid_x = n / Bn;
    const size_t grid_y = math_ceil_div(m, Bm);
    const size_t grid_z = num_experts;

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_mf4w_moe_dense_matmul_swiglu_fn,
        threadgroup_size_x, threadgroup_size_y, threadgroup_size_z,
        grid_x, grid_y, grid_z,
        sizeof(args), &args,
        6,
        (const struct gptoss_metal_buffer *[]) {expert_offsets_buffer, input_buffer, weight_block_buffer, weight_scale_buffer, bias_buffer, output_buffer},
        (const size_t[]) {expert_offsets_offset, input_offset, weight_block_offset, weight_scale_offset, bias_offset, output_offset},
        /*threadgroup_buffer_size=*/0);

    }

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_dense_matmul(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_mf4w_moe_dense_matmul_fn,
    const struct gptoss_metal_buffer* expert_offsets_buffer,
    size_t expert_offsets_offset,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_block_buffer,
    size_t weight_block_offset,
    const struct gptoss_metal_buffer* weight_scale_buffer,
    size_t weight_scale_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t expert_stride_bytes,
    uint32_t num_tokens,
    uint32_t num_experts,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_mf4w_moe_dense_matmul_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_cols % 32 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul kernel launch: number of columns (%" PRIu32 ") is not divisible by 32",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const struct gptoss_moe_dense_matmul_args args = {
        .k = num_cols,
        .n = num_rows,
        .weight_blocks_expert_stride_bytes = expert_stride_bytes,
        .weight_scales_expert_stride_bytes = expert_stride_bytes,
        .bias_expert_stride_bytes = expert_stride_bytes,
    };

    const size_t threads_per_simdgroup = f32_mf4w_moe_dense_matmul_fn->simdgroup_threads;
    const uint32_t m = num_tokens;
    const uint32_t n = args.n;
    const uint32_t k = args.k;
    const uint32_t Bm = MOE_DENSE_MATMUL_Bm;
    const uint32_t Bn = MOE_DENSE_MATMUL_Bn;
    const uint32_t Bk = MOE_DENSE_MATMUL_Bk;
    const uint32_t Sg_Bm = MOE_DENSE_MATMUL_Sg_Bm;
    const uint32_t Sg_Bn = MOE_DENSE_MATMUL_Sg_Bn;
    if (Bm % Sg_Bm != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul kernel launch: Bm (%" PRIu32 ") is not divisible by Sg_Bm (%" PRIu32 ")",
            Bm, Sg_Bm);
        return gptoss_status_invalid_argument;
    }
    if (Bn % Sg_Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul kernel launch: Bn (%" PRIu32 ") is not divisible by Sg_Bn (%" PRIu32 ")",
            Bn, Sg_Bn);
        return gptoss_status_invalid_argument;
    }

    const size_t threadgroup_size_x = (Bm / Sg_Bm) * (Bn / Sg_Bn) * threads_per_simdgroup;
    const size_t threadgroup_size_y = 1;
    const size_t threadgroup_size_z = 1;
    const size_t total_threadgroup_size = threadgroup_size_x * threadgroup_size_y * threadgroup_size_z;
    if (total_threadgroup_size > f32_mf4w_moe_dense_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul kernel launch: total threadgroup size (%zu) exceeds supported maximum (%zu)",
            total_threadgroup_size, f32_mf4w_moe_dense_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }
    if (n % Bn != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul kernel launch: n (%" PRIu32 ") is not divisible by Bn (%" PRIu32 ")",
            n, Bn);
        return gptoss_status_invalid_argument;
    }
    if (k % Bk != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_dense_matmul kernel launch: k (%" PRIu32 ") is not divisible by Bk (%" PRIu32 ")",
            k, Bk);
        return gptoss_status_invalid_argument;
    }

    const size_t grid_y = math_ceil_div(m, Bm);
    const size_t grid_x = n / Bn;
    const size_t grid_z = num_experts;

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_mf4w_moe_dense_matmul_fn,
        threadgroup_size_x, threadgroup_size_y, threadgroup_size_z,
        grid_x, grid_y, grid_z,
        sizeof(args), &args,
        6,
        (const struct gptoss_metal_buffer *[]) {expert_offsets_buffer, input_buffer, weight_block_buffer, weight_scale_buffer, bias_buffer, output_buffer},
        (const size_t[]) {expert_offsets_offset, input_offset, weight_block_offset, weight_scale_offset, bias_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_accumulate(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_accumulate_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_buffer,
    size_t expert_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_channels,
    uint32_t num_tokens,
    uint32_t num_experts)
{
    if (command_buffer->object == NULL || f32_accumulate_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels% 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_accumulate_fn->max_threadgroup_threads;
    } else if (threadgroup_size > f32_accumulate_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_channels / 4;
    const size_t num_vecs_per_expert = num_vecs * num_tokens;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_accumulate_args args = {
        .num_vecs_per_expert = num_vecs_per_expert,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .num_vecs = num_vecs,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_accumulate_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, num_tokens, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_buffer, output_buffer, control_buffer},
        (const size_t[]) {input_offset, expert_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_topk(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_topk_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_tokens,
    uint32_t num_experts,
    uint32_t num_active_experts)
{
    if (command_buffer->object == NULL || f32_topk_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_experts != 32  && num_experts != 128) {
        return gptoss_status_invalid_argument;
    }

    if (num_active_experts != 4) {
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_topk_args args = { 0 };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_topk_fn,
        /*threadgroup_size=*/32, 1, 1,
        num_tokens, 1, 1,
        sizeof(args), &args,
        3,
        (const struct gptoss_metal_buffer *[]) {input_buffer, output_buffer, control_buffer},
        (const size_t[]) {input_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_sdpa(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_sdpa_fn,
    const struct gptoss_metal_buffer* q_buffer,
    size_t q_offset,
    const struct gptoss_metal_buffer* kv_buffer,
    size_t kv_offset,
    const struct gptoss_metal_buffer* s_buffer,
    size_t s_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t window,
    uint32_t kv_stride,
    uint32_t num_q_tokens,
    uint32_t num_kv_tokens,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim)
{
    if (command_buffer->object == NULL || f32_sdpa_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_q_heads != num_kv_heads * 8) {
        GPTOSS_LOG_ERROR("number of Q heads (%" PRIu32 ") must be 8 times the number of KV heads (%" PRIu32 ")",
            num_q_heads, num_kv_heads);
        return gptoss_status_invalid_argument;
    }

    if (head_dim != 64) {
        GPTOSS_LOG_ERROR("attention head dimension (%" PRIu32 ") must be 64", head_dim);
        return gptoss_status_invalid_argument;
    }

    const size_t max_context_tokens = math_min(num_q_tokens + num_kv_tokens + 1, window);
    const size_t threadgroup_size = math_min(f32_sdpa_fn->max_threadgroup_threads,
        max_context_tokens * f32_sdpa_fn->simdgroup_threads);
    const size_t half_threadgroup_size = math_round_down_po2(threadgroup_size / 2, f32_sdpa_fn->simdgroup_threads);

    const struct gptoss_sdpa_args args = {
        .qkv_dim = head_dim * (num_q_heads + 2 * num_kv_heads),
        .num_kv_tokens = num_kv_tokens,
        .kv_stride = kv_stride,
        .window = window,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_sdpa_fn,
        threadgroup_size, 1, 1,
        num_q_tokens, num_kv_heads, 1,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {q_buffer, kv_buffer, s_buffer, output_buffer, control_buffer},
        (const size_t[]) {q_offset, kv_offset, s_offset, output_offset, control_offset},
        /*threadgroup_buffer_size=*/half_threadgroup_size * 8 * 4 * sizeof(float));
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_softmax(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_softmax_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* score_buffer,
    size_t score_offset,
    const struct gptoss_metal_buffer* argmax_buffer,
    size_t argmax_offset,
    const struct gptoss_metal_buffer* prob_buffer,
    size_t prob_offset,
    const struct gptoss_metal_buffer* sum_buffer,
    size_t sum_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint32_t num_channels,
    uint32_t num_tokens,
    float temperature,
    uint32_t* num_threadgroups_out,
    uint32_t* num_channels_per_threadgroup_out)
{
    *num_threadgroups_out = 0;
    *num_channels_per_threadgroup_out = 0;
    if (command_buffer->object == NULL || f32_softmax_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    const size_t num_vecs = num_channels;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_softmax_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .max_threadgroups = max_threadgroups,
        .temperature = temperature,
    };

    *num_threadgroups_out = num_threadgroups;
    *num_channels_per_threadgroup_out = num_vecs_per_threadgroup;
    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_softmax_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, num_tokens, 1,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {score_buffer, argmax_buffer, prob_buffer, sum_buffer, control_buffer},
        (const size_t[]) {score_offset, argmax_offset, prob_offset, sum_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_sample(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_sample_fn,
    size_t min_threadgroup_size,
    const struct gptoss_metal_buffer* prob_buffer,
    size_t prob_offset,
    const struct gptoss_metal_buffer* sum_buffer,
    size_t sum_offset,
    const struct gptoss_metal_buffer* token_buffer,
    size_t token_offset,
    const struct gptoss_metal_buffer* control_buffer,
    size_t control_offset,
    uint64_t rng_seed,
    uint32_t rng_offset,
    uint32_t num_blocks,
    uint32_t num_channels,
    uint32_t num_channels_per_block)
{
    if (command_buffer->object == NULL || f32_sample_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (min_threadgroup_size > f32_sample_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    if (min_threadgroup_size % f32_sample_fn->simdgroup_threads != 0) {
        return gptoss_status_invalid_argument;
    }

    if (num_blocks > f32_sample_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_sample_args args = {
        .rng_seed = rng_seed,
        .rng_offset = rng_offset,
        .num_blocks = num_blocks,
        .num_dims = num_channels,
        .num_dims_per_block = num_channels_per_block,
    };

    const size_t threadgroup_size = math_max(min_threadgroup_size,
        math_round_up_po2(num_blocks, f32_sample_fn->simdgroup_threads));
    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_sample_fn,
        threadgroup_size, 1, 1,
        1, 1, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {prob_buffer, sum_buffer, token_buffer, control_buffer},
        (const size_t[]) {prob_offset, sum_offset, token_offset, control_offset},
        /*threadgroup_buffer_size=*/0);
}
