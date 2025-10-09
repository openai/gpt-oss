#include <assert.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <sys/mman.h>

#include <gpt-oss.h>

#include "internal/log.h"
#include "internal/model.h"


enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_special_token_id(
    gptoss_tokenizer_t tokenizer,
    enum gptoss_special_token token_type,
    uint32_t* token_id_out)
{
    uint32_t token_id = UINT32_MAX;
    if (token_type != gptoss_special_token_invalid && token_type < gptoss_special_token_max)
    {
        token_id = tokenizer->special_token_id[(uint32_t) token_type - 1];
    }
    if (token_id == UINT32_MAX) {
        return gptoss_status_invalid_argument;
    }

    *token_id_out = token_id;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_num_text_tokens(
    gptoss_tokenizer_t tokenizer,
    uint32_t* num_text_tokens_out)
{
    *num_text_tokens_out = tokenizer->num_text_tokens;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_num_special_tokens(
    gptoss_tokenizer_t tokenizer,
    uint32_t* num_special_tokens_out)
{
    *num_special_tokens_out = tokenizer->num_special_tokens;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_num_tokens(
    gptoss_tokenizer_t tokenizer,
    uint32_t* num_tokens_out)
{
    *num_tokens_out = tokenizer->num_text_tokens + tokenizer->num_special_tokens;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_tokenizer_decode(
    gptoss_tokenizer_t tokenizer,
    uint32_t token_id,
    const void** token_ptr_out,
    size_t* token_size_out)
{
    if (token_id >= tokenizer->num_text_tokens) {
        return gptoss_status_invalid_argument;
    }

    const char* token_ptr = (const char*) tokenizer->tokens_ptr;
    for (uint32_t t = 0; t < token_id; t++) {
        // Reading unaligned uint16_t
        uint16_t token_length;
        memcpy(&token_length, token_ptr, sizeof(token_length));

        token_ptr += (size_t) token_length + sizeof(uint16_t);
    }

    *token_ptr_out = (const void*) (token_ptr + sizeof(uint16_t));
    *token_size_out = (size_t) *token_ptr;
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_tokenizer_retain(
    gptoss_tokenizer_t tokenizer)
{
    atomic_fetch_add_explicit(&tokenizer->ref_count, 1, memory_order_relaxed);
    return gptoss_status_success;
}

enum gptoss_status GPTOSS_ABI gptoss_tokenizer_release(
    gptoss_tokenizer_t tokenizer)
{
    if (tokenizer != NULL) {
        if (atomic_fetch_sub_explicit(&tokenizer->ref_count, 1, memory_order_acquire) == 1) {
            if (tokenizer->mapping_ptr != NULL && tokenizer->mapping_size != 0) {
                if (munmap(tokenizer->mapping_ptr, tokenizer->mapping_size) != 0) {
                    GPTOSS_LOG_WARNING("munmap for tokenizer mapping failed with error %d", errno);
                }
            }

            memset(tokenizer, 0, sizeof(struct gptoss_tokenizer));
            free(tokenizer);
        }
    }
    return gptoss_status_success;
}
