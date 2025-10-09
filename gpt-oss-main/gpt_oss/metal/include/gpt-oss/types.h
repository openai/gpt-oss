#pragma once

/*
 * Status codes returned by GPT-OSS API functions.
 */
enum gptoss_status {
    gptoss_status_success = 0,
    gptoss_status_invalid_argument = 1,
    gptoss_status_unsupported_argument = 2,
    gptoss_status_invalid_state = 3,
    gptoss_status_io_error = 4,
    gptoss_status_insufficient_memory = 5,
    gptoss_status_insufficient_resources = 6,
    gptoss_status_unsupported_system = 7,
    gptoss_status_context_overflow = 8,
};

enum gptoss_special_token {
    gptoss_special_token_invalid = 0,
    gptoss_special_token_return = 1,
    gptoss_special_token_start = 2,
    gptoss_special_token_message = 3,
    gptoss_special_token_end = 4,
    gptoss_special_token_refusal = 5,
    gptoss_special_token_constrain = 6,
    gptoss_special_token_channel = 7,
    gptoss_special_token_call = 8,
    gptoss_special_token_untrusted = 9,
    gptoss_special_token_end_untrusted = 10,
    gptoss_special_token_max,
};

/*
 * Model object is an opaque container comprised of:
 * - Weights
 * - Temporary buffers required to run the model
 * - Any other resources requires to run the model
 */
typedef struct gptoss_model* gptoss_model_t;

typedef struct gptoss_tokenizer* gptoss_tokenizer_t;

/*
 * Context is an opaque container comprised of:
 * - Input tokens
 * - Distribution over the output tokens
 * - KV cache
 * 
 * Multiple contexts can be created and used with the same model.
 */
typedef struct gptoss_context* gptoss_context_t;

/*
 * Sampler is an opaque container for sampling parameters:
 * - Temperature
 * - Top-p (nucleus sampling)
 * - Frequency penalty
 * - Presence penalty
 *
 * Multiple samplers can be created and used with the same context.
 */
typedef struct gptoss_sampler* gptoss_sampler_t;
