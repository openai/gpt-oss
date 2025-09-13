#include <assert.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>  // errno, EISDIR, ENOENT, ENOTDIR
#include <fcntl.h>  // open
#include <mach/vm_page_size.h>  // vm_page_size
#include <sys/mman.h>  // mmap, PROT_READ, MAP_PRIVATE
#include <sys/stat.h>  // fstat, stat
#include <sys/types.h>  // off_t, ssize_t
#include <unistd.h>  // close

#include <gpt-oss.h>

#include "internal/datatype.h"
#include "internal/kernel-args.h"  // gptoss_expert_prediction
#include "internal/log.h"
#include "internal/uuid.h"
#include "internal/storage.h"
#include "internal/math.h"
#include "internal/model.h"


static size_t round_up_to_page_size(size_t bytes) {
    const size_t page_size_mask = (size_t) vm_page_size - 1;
    if ((bytes & page_size_mask) != 0) {
        bytes |= page_size_mask;
        bytes += 1;
    }
    return bytes;
}

static size_t round_down_to_page_size(size_t bytes) {
    const size_t page_size_mask = (size_t) vm_page_size - 1;
    return bytes & ~page_size_mask;
}

static enum gptoss_status read_fd(int fd, void* data, size_t size, const char* path) {
    assert(fd != -1);
    assert(data != NULL);
    assert(size != 0);

    size_t bytes_to_read = size;
    char* current_byte = (char*) data;
    do {
        const ssize_t read_result = read(fd, current_byte, bytes_to_read);
        if (read_result < 0) {
            GPTOSS_LOG_ERROR("reading %zu bytes from file %s failed with error %d",
                size, path, errno);
            return gptoss_status_io_error;
        }
        current_byte += (size_t) read_result;
        bytes_to_read -= (size_t) read_result;
    } while (bytes_to_read != 0);
    return gptoss_status_success;
}

static void prefetch_fd(int fd, size_t offset, size_t size, const char* path) {
    // radvisory.ra_count is int, so we can't prefetch 2GB+ at once
    const size_t prefetch_max = round_down_to_page_size((size_t) INT_MAX);
    do {
        const size_t prefetch_size = math_min(size, prefetch_max);
        const struct radvisory ra = {
            .ra_offset = offset,
            .ra_count = (int) prefetch_size,
        };
        if (fcntl(fd, F_RDADVISE, &ra) == -1) {
            GPTOSS_LOG_WARNING("fcntl(%s, F_RDADVISE, .ra_offset=%zu, .ra_count=%d) failed with error %d\n",
                path, (size_t) ra.ra_offset, ra.ra_count, errno);
            return;
        }
        offset += prefetch_size;
        size -= prefetch_size;
    } while (size != 0);
}

enum gptoss_status GPTOSS_ABI gptoss_model_create_from_file(
    const char* path,
    gptoss_model_t* model_out,
    size_t max_batch_tokens)
{
    *model_out = NULL;

    enum gptoss_status status = gptoss_status_success;
    struct gptoss_model* model = NULL;
    struct gptoss_tokenizer* tokenizer = NULL;
    int fd = -1;
    size_t file_offset = 0;

    fd = open(path, O_RDONLY);
    if (fd == -1) {
        GPTOSS_LOG_ERROR("open(%s) failed with error %d", path, errno);
        switch (errno) {
            case EISDIR:
            case ENOENT:
            case ENOTDIR:
                status = gptoss_status_invalid_argument;
                break;
            default:
                status = gptoss_status_io_error;
                break;
        }
        goto cleanup;
    }

    struct gptoss_file_header file_header;
    status = read_fd(fd, &file_header, sizeof(file_header), path);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    file_offset += sizeof(file_header);

    if (file_header.magic[0] != 'G' ||
        file_header.magic[1] != 'P' ||
        file_header.magic[2] != 'T' ||
        file_header.magic[3] != '-' ||
        file_header.magic[4] != 'O' ||
        file_header.magic[5] != 'S' ||
        file_header.magic[6] != 'S' ||
        file_header.magic[7] != ' ' ||
        file_header.magic[8] != 'v' ||
        file_header.magic[9] != '1' ||
        file_header.magic[10] != '.' ||
        file_header.magic[11] != '0' ||
        file_header.zero != 0)
    {
        GPTOSS_LOG_ERROR("invalid magic in file %s", path);
        status = gptoss_status_invalid_argument;
        goto cleanup;
    }

    struct gptoss_uuid model_uuid;
    status = read_fd(fd, &model_uuid, sizeof(model_uuid), path);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    file_offset += sizeof(model_uuid);

    if (!gptoss_is_gptoss_model_uuid(&model_uuid)) {
        GPTOSS_LOG_ERROR("unsupported model UUID " UUID_FORMAT, UUID_ARGS(model_uuid));
        status = gptoss_status_invalid_argument;
        goto cleanup;
    }

    struct gptoss_gptoss_model_header model_header;
    status = read_fd(fd, &model_header, sizeof(model_header), path);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    file_offset += sizeof(model_header);

    struct gptoss_uuid layout_uuid;
    status = read_fd(fd, &layout_uuid, sizeof(layout_uuid), path);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    file_offset += sizeof(layout_uuid);

    if (!gptoss_is_applegpu_layout_uuid(&layout_uuid)) {
        GPTOSS_LOG_ERROR("unsupported layout UUID " UUID_FORMAT, UUID_ARGS(layout_uuid));
        status = gptoss_status_invalid_argument;
        goto cleanup;
    }

    const size_t model_size = sizeof(struct gptoss_model) + model_header.num_blocks * sizeof(struct gptoss_metal_buffer);
    model = malloc(model_size);
    if (model == NULL) {
        GPTOSS_LOG_ERROR("failed to allocate %zu bytes for model descriptor", model_size);
        status = gptoss_status_insufficient_memory;
        goto cleanup;
    }
    memset(model, 0, model_size);

    atomic_store_explicit(&model->ref_count, 1, memory_order_relaxed);
    model->context_length = model_header.context_length;
    model->num_blocks = model_header.num_blocks;
    model->num_experts = model_header.num_experts;
    model->num_active_experts = model_header.num_active_experts;
    model->embedding_dim = model_header.embedding_dim;
    model->mlp_dim = model_header.mlp_dim;
    model->swiglu_limit = model_header.swiglu_limit;
    model->head_dim = model_header.head_dim;
    model->num_heads = model_header.num_heads;
    model->num_kv_heads = model_header.num_kv_heads;
    model->attention_window = model_header.attention_window;
    model->rope_theta = model_header.rope_theta;
    model->interpolation_scale = model_header.interpolation_scale;
    model->yarn_offset = model_header.yarn_offset;
    model->yarn_scale = model_header.yarn_scale;
    model->yarn_multiplier = model_header.yarn_multiplier;
    model->rmsnorm_epsilon = model_header.rmsnorm_epsilon;

    model->max_batch_tokens = max_batch_tokens == 0 ? GPTOSS_DEFAULT_BATCH_SIZE : max_batch_tokens;

    struct gptoss_uuid tokenizer_uuid;
    status = read_fd(fd, &tokenizer_uuid, sizeof(tokenizer_uuid), path);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    file_offset += sizeof(tokenizer_uuid);

    if (!gptoss_is_tiktoken_tokenizer_uuid(&tokenizer_uuid)) {
        GPTOSS_LOG_ERROR("unsupported tokenizer UUID " UUID_FORMAT, UUID_ARGS(tokenizer_uuid));
        status = gptoss_status_invalid_argument;
        goto cleanup;
    }

    struct gptoss_tiktoken_tokenizer_header tokenizer_header;
    status = read_fd(fd, &tokenizer_header, sizeof(tokenizer_header), path);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    file_offset += sizeof(tokenizer_header);

    tokenizer = malloc(sizeof(struct gptoss_tokenizer));
    if (tokenizer == NULL) {
        GPTOSS_LOG_ERROR("failed to allocate %zu bytes for tokenizer descriptor", sizeof(struct gptoss_tokenizer));
        status = gptoss_status_insufficient_memory;
        goto cleanup;
    }
    memset(tokenizer, 0, sizeof(struct gptoss_tokenizer));
    // Initialize all special token IDs to UINT32_MAX (0xFF in all bytes)
    memset(tokenizer->special_token_id, 0xFF, sizeof(tokenizer->special_token_id));

    atomic_store_explicit(&tokenizer->ref_count, 1, memory_order_relaxed);
    tokenizer->num_special_tokens = tokenizer_header.num_special_tokens;
    tokenizer->num_text_tokens = tokenizer_header.num_text_tokens;
    model->vocabulary_size = tokenizer_header.num_special_tokens + tokenizer_header.num_text_tokens;
    for (uint32_t t = 0; t < tokenizer_header.num_special_tokens; t++) {
        struct gptoss_uuid token_uuid;
        status = read_fd(fd, &token_uuid, sizeof(token_uuid), path);
        if (status != gptoss_status_success) {
            goto cleanup;
        }
        file_offset += sizeof(token_uuid);

        const enum gptoss_special_token token = gptoss_special_token_decode_uuid(&token_uuid);
        if (token != gptoss_special_token_invalid) {
            tokenizer->special_token_id[token - 1] = tokenizer_header.num_text_tokens + t;
        }
    }

    const size_t tokenizer_start_offset = file_offset;
    const size_t tokenizer_end_offset = tokenizer_start_offset + tokenizer_header.regex_size + tokenizer_header.tokens_size;
    const size_t tokenizer_mapping_start = round_down_to_page_size(tokenizer_start_offset);
    const size_t tokenizer_mapping_size = round_up_to_page_size(tokenizer_end_offset) - tokenizer_mapping_start;
    void* tokenizer_mapping_ptr = mmap(NULL, tokenizer_mapping_size, PROT_READ, MAP_PRIVATE, fd, tokenizer_mapping_start);
    if (tokenizer_mapping_ptr == (void*) -1) {
        GPTOSS_LOG_ERROR("failed to mmap(%s) tokenizer at offset %zu size %zu",
            path, tokenizer_mapping_start, tokenizer_mapping_size);
        status = gptoss_status_io_error;
        goto cleanup;
    }
    tokenizer->mapping_ptr = tokenizer_mapping_ptr;
    tokenizer->mapping_size = tokenizer_mapping_size;
    tokenizer->regex_ptr = (const char*) tokenizer_mapping_ptr + (tokenizer_start_offset - tokenizer_mapping_start);
    tokenizer->tokens_ptr = tokenizer->regex_ptr + tokenizer_header.regex_size;

    if (madvise(tokenizer_mapping_ptr, tokenizer_mapping_size, MADV_RANDOM | MADV_WILLNEED) != 0) {
        GPTOSS_LOG_WARNING("madvise(%s, size=%zu) failed with error %d", path, tokenizer_mapping_size, errno);
    }

    prefetch_fd(fd, tokenizer_mapping_start, tokenizer_mapping_size, path);

    struct stat model_stat = {0};
    int stat_result = fstat(fd, &model_stat);
    if (stat_result != 0) {
        GPTOSS_LOG_ERROR("stat(%s) failed with error %d", path, errno);
        status = gptoss_status_io_error;
        goto cleanup;
    }

    const size_t model_mapping_start = round_up_to_page_size(tokenizer_end_offset);
    const size_t model_mapping_size = round_up_to_page_size((size_t) model_stat.st_size) - model_mapping_start;
    void* model_mapping_ptr = mmap(NULL, model_mapping_size, PROT_READ, MAP_PRIVATE, fd, model_mapping_start);
    if (model_mapping_ptr == (void*) -1) {
        GPTOSS_LOG_ERROR("failed to mmap(%s) model weights at offset %zu size %zu",
            path, model_mapping_start, model_mapping_size);
        status = gptoss_status_io_error;
        goto cleanup;
    }
    model->mapping_ptr = model_mapping_ptr;
    model->mapping_size = model_mapping_size;

    if (madvise(model_mapping_ptr, model_mapping_size, MADV_SEQUENTIAL | MADV_WILLNEED) != 0) {
        GPTOSS_LOG_WARNING("madvise(%s, size=%zu) failed with error %d", path, model_mapping_size, errno);
    }

    prefetch_fd(fd, model_mapping_start, model_mapping_size, path);

    if (mlock(model_mapping_ptr, model_mapping_size) != 0) {
        GPTOSS_LOG_WARNING("mlock(%s, size=%zu) failed with error %d", path, model_mapping_size, errno);
    } else {
        model->lock_memory = true;
    }

    // Initialize Metal
    status = gptoss_metal_device_create_system_default(&model->device);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    model->max_threadgroups = model->device.num_cores * 3;
    status = gptoss_metal_command_queue_create(&model->device, &model->command_queue);
    if (status != gptoss_status_success) {
        goto cleanup;
    }

    // Metal kernels
    status = gptoss_metal_library_create_default(&model->device, &model->library);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_bf16_f32_embeddings", &model->bf16_f32_embeddings_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_bf16w_rmsnorm", &model->f32_bf16w_rmsnorm_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_bf16w_matmul", &model->f32_bf16w_matmul_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_bf16w_matmul_qkv", &model->f32_bf16w_matmul_qkv_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_bf16w_dense_matmul_qkv", &model->f32_bf16w_dense_matmul_qkv_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_bf16w_dense_matmul_attn_output", &model->f32_bf16w_dense_matmul_attn_output_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_bf16w_dense_matmul_mlp_gate", &model->f32_bf16w_dense_matmul_mlp_gate_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_bf16w_unembedding", &model->f32_bf16w_unembedding_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_rope", &model->f32_rope_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_mf4w_moe_matmul_swiglu", &model->f32_mf4w_moe_matmul_swiglu_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_mf4w_moe_matmul", &model->f32_mf4w_moe_matmul_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_accumulate_e4", &model->f32_accumulate_e4_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_topk_softmax_e32_k4", &model->f32_topk_softmax_e32_k4_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_topk_softmax_e128_k4", &model->f32_topk_softmax_e128_k4_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_softmax", &model->f32_softmax_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_sample", &model->f32_sample_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }
    status = gptoss_metal_function_create(&model->library, "gptoss_f32_sdpa_q8_d64", &model->f32_sdpa_q8_d64_fn);
    if (status != gptoss_status_success) {
        goto cleanup;
    }

    // Kernel launch parameters
    model->embeddings_threadgroup_size = 512;
    model->attn_qkv_threadgroup_size = 1024;
    model->attn_out_threadgroup_size = 768;
    model->mlp_gate_threadgroup_size = 256;
    model->mlp_swiglu_threadgroup_size = 192;
    model->mlp_out_threadgroup_size = 192;
    model->mlp_acc_threadgroup_size = 768;
    model->unembedding_threadgroup_size = 416;

    // Weight buffers
    const char* current_ptr = (const char*) model->mapping_ptr;

    const size_t embedding_weight_size = math_round_up_po2(model->vocabulary_size * model->embedding_dim * sizeof(gptoss_bfloat16), 16);
    model->attn_rmsnorm_gain_offset = embedding_weight_size;
    const size_t rmsnorm_weight_size = math_round_up_po2(model->embedding_dim * sizeof(gptoss_bfloat16), 16);
    model->attn_qkv_weight_offset = model->attn_rmsnorm_gain_offset + rmsnorm_weight_size;
    const size_t attn_qkv_dim = model->head_dim * (model->num_heads + 2 * model->num_kv_heads);
    const size_t attn_qkv_weight_size = math_round_up_po2(attn_qkv_dim * model->embedding_dim * sizeof(gptoss_bfloat16), 16);
    model->attn_qkv_bias_offset = model->attn_qkv_weight_offset + attn_qkv_weight_size;
    const size_t attn_qkv_bias_size = math_round_up_po2(attn_qkv_dim * sizeof(gptoss_bfloat16), 16);
    model->attn_sdpa_sink_offset = model->attn_qkv_bias_offset + attn_qkv_bias_size;
    const size_t attn_sink_weight_size = math_round_up_po2(model->num_heads * sizeof(gptoss_bfloat16), 16);
    model->attn_out_weight_offset = model->attn_sdpa_sink_offset + attn_sink_weight_size;
    const size_t attn_out_weight_size = math_round_up_po2(model->embedding_dim * model->num_heads * model->head_dim * sizeof(gptoss_bfloat16), 16);
    model->attn_out_bias_offset = model->attn_out_weight_offset + attn_out_weight_size;
    const size_t attn_out_bias_size = math_round_up_po2(model->embedding_dim * sizeof(gptoss_bfloat16), 16);
    model->mlp_rmsnorm_gain_offset = model->attn_out_bias_offset + attn_out_bias_size;
    model->mlp_gate_weight_offset = model->mlp_rmsnorm_gain_offset + rmsnorm_weight_size;
    const size_t mlp_gate_weight_size = math_round_up_po2(model->num_experts * model->embedding_dim * sizeof(gptoss_bfloat16), 16);
    model->mlp_gate_bias_offset = model->mlp_gate_weight_offset + mlp_gate_weight_size;
    const size_t mlp_gate_bias_size = math_round_up_po2(model->num_experts * sizeof(gptoss_bfloat16), 16);
    const size_t per_block_shared_weights_size =
        rmsnorm_weight_size + attn_qkv_weight_size + attn_qkv_bias_size + attn_sink_weight_size + attn_out_weight_size + attn_out_bias_size +
        rmsnorm_weight_size + mlp_gate_weight_size + mlp_gate_bias_size;
    model->rmsnorm_weight_offset = embedding_weight_size + model->num_blocks * per_block_shared_weights_size;
    model->unembedding_weight_offset = model->rmsnorm_weight_offset + rmsnorm_weight_size;
    const size_t unembedding_weight_size = math_round_up_po2(model->vocabulary_size * model->embedding_dim * sizeof(gptoss_bfloat16), 16);

    model->per_block_shared_weights_size = per_block_shared_weights_size;
    const size_t shared_weights_size =
        round_up_to_page_size(embedding_weight_size + rmsnorm_weight_size + unembedding_weight_size + model->num_blocks * per_block_shared_weights_size);

    status = gptoss_metal_buffer_wrap(&model->device, shared_weights_size, current_ptr, &model->shared_weight_buffer);
    if (status != gptoss_status_success) {
        GPTOSS_LOG_ERROR("failed to map expert-shared weight of size %zu onto a Metal buffer", shared_weights_size);
        goto cleanup;
    }
    current_ptr += shared_weights_size;
    model->weights_size += shared_weights_size;

    const size_t mlp_swiglu_weight_block_size = math_round_up_po2(2 * model->mlp_dim * model->embedding_dim / 2, 16);
    model->mlp_swiglu_scale_offset = mlp_swiglu_weight_block_size;
    const size_t mlp_swiglu_weight_scale_size = math_round_up_po2(2 * model->mlp_dim * model->embedding_dim / 32, 16);
    model->mlp_swiglu_bias_offset = model->mlp_swiglu_scale_offset + mlp_swiglu_weight_scale_size;
    const size_t mlp_swiglu_bias_size = math_round_up_po2(2 * model->mlp_dim * sizeof(gptoss_bfloat16), 16);
    model->mlp_out_block_offset = model->mlp_swiglu_bias_offset + mlp_swiglu_bias_size;
    const size_t mlp_out_weight_block_size = math_round_up_po2(model->embedding_dim * model->mlp_dim / 2, 16);
    model->mlp_out_scale_offset = model->mlp_out_block_offset + mlp_out_weight_block_size;
    const size_t mlp_out_weight_scale_size = math_round_up_po2(model->embedding_dim * model->mlp_dim / 32, 16);
    model->mlp_out_bias_offset = model->mlp_out_scale_offset + mlp_out_weight_scale_size;
    const size_t mlp_out_bias_size = math_round_up_po2(model->embedding_dim * sizeof(gptoss_bfloat16), 16);
    model->per_expert_block_weight_size =
        mlp_swiglu_weight_block_size + mlp_swiglu_weight_scale_size + mlp_swiglu_bias_size + mlp_out_weight_block_size + mlp_out_weight_scale_size + mlp_out_bias_size;
    const size_t moe_block_weight_size = round_up_to_page_size(model->num_experts * model->per_expert_block_weight_size);
    for (uint32_t n = 0; n < model->num_blocks; n++) {
        status = gptoss_metal_buffer_wrap(&model->device, moe_block_weight_size, current_ptr, &model->block_weight_buffers[n]);
        if (status != gptoss_status_success) {
            GPTOSS_LOG_ERROR("failed to map block #%" PRIu32 " MoE weight of size %zu onto a Metal buffer",
                n, moe_block_weight_size);
            goto cleanup;
        }
        current_ptr += moe_block_weight_size;
        model->weights_size += moe_block_weight_size;
    }

    // Commit tokenizer
    model->tokenizer = tokenizer;
    tokenizer = NULL;

    // Commit model
    *model_out = model;
    model = NULL;

cleanup:
    if (fd != -1) {
        close(fd);
        fd = -1;
    }
    gptoss_model_release(model);  // does nothing if model is NULL
    gptoss_tokenizer_release(tokenizer);  // does nothing if tokenizer is NULL
    return status;
}

enum gptoss_status GPTOSS_ABI gptoss_model_get_tokenizer(
    gptoss_model_t model,
    gptoss_tokenizer_t* tokenizer_out)
{
    gptoss_tokenizer_t tokenizer = model->tokenizer;
    atomic_fetch_add_explicit(&tokenizer->ref_count, 1, memory_order_relaxed);
    *tokenizer_out = tokenizer;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_model_get_max_context_length(
    gptoss_model_t model,
    size_t* max_context_length_out)
{
    *max_context_length_out = model->context_length;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_model_retain(
    gptoss_model_t model)
{
    atomic_fetch_add_explicit(&model->ref_count, 1, memory_order_relaxed);
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_model_release(
    gptoss_model_t model)
{
    if (model != NULL) {
        if (atomic_fetch_sub_explicit(&model->ref_count, 1, memory_order_acq_rel) == 1) {
            gptoss_tokenizer_release(model->tokenizer);

            // Weight buffers
            gptoss_metal_buffer_release(&model->shared_weight_buffer);
            for (uint32_t n = 0; n < model->num_blocks; n++) {
                gptoss_metal_buffer_release(&model->block_weight_buffers[n]);
            }

            // Metal kernels
            gptoss_metal_function_release(&model->bf16_f32_embeddings_fn);
            gptoss_metal_function_release(&model->f32_bf16w_rmsnorm_fn);
            gptoss_metal_function_release(&model->f32_bf16w_matmul_fn);
            gptoss_metal_function_release(&model->f32_bf16w_matmul_qkv_fn);
            gptoss_metal_function_release(&model->f32_bf16w_dense_matmul_qkv_fn);
            gptoss_metal_function_release(&model->f32_bf16w_dense_matmul_attn_output_fn);
            gptoss_metal_function_release(&model->f32_bf16w_dense_matmul_mlp_gate_fn);
            gptoss_metal_function_release(&model->f32_bf16w_unembedding_fn);
            gptoss_metal_function_release(&model->f32_rope_fn);
            gptoss_metal_function_release(&model->f32_mf4w_moe_matmul_swiglu_fn);
            gptoss_metal_function_release(&model->f32_mf4w_moe_matmul_fn);
            gptoss_metal_function_release(&model->f32_accumulate_e4_fn);
            gptoss_metal_function_release(&model->f32_topk_softmax_e32_k4_fn);
            gptoss_metal_function_release(&model->f32_topk_softmax_e128_k4_fn);
            gptoss_metal_function_release(&model->f32_softmax_fn);
            gptoss_metal_function_release(&model->f32_sample_fn);
            gptoss_metal_function_release(&model->f32_sdpa_q8_d64_fn);
            gptoss_metal_library_release(&model->library);

            gptoss_metal_command_queue_release(&model->command_queue);
            gptoss_metal_device_release(&model->device);
            // Weight buffers

            if (model->mapping_ptr != NULL && model->mapping_size != 0) {
                if (model->lock_memory) {
                    if (munlock(model->mapping_ptr, model->mapping_size) != 0) {
                        GPTOSS_LOG_WARNING("munlock for model weight mapping failed with error %d", errno);
                    }
                }

                if (munmap(model->mapping_ptr, model->mapping_size) != 0) {
                    GPTOSS_LOG_WARNING("munmap for model weight mapping failed with error %d", errno);
                }
            }

            const size_t model_size = sizeof(struct gptoss_model) + model->num_blocks * sizeof(struct gptoss_metal_buffer);
            memset(model, 0, model_size);
            free(model);
        }
    }
    return gptoss_status_success;
}
