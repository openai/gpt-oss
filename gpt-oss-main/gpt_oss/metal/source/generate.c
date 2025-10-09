#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mach/mach_time.h>

#include <gpt-oss.h>

#include "internal/model.h"

struct {
    atomic_uint_least64_t inference_bytes;
    atomic_size_t num_prefill_tokens;
    atomic_uint_least64_t prefill_microseconds;
    atomic_size_t num_generated_tokens;
    atomic_uint_least64_t generation_microseconds;
} globals = {
    .inference_bytes = 0,
    .num_prefill_tokens = 0,
    .prefill_microseconds = 0,
    .num_generated_tokens = 0,
    .generation_microseconds = 0,
};

struct options {
    const char* model;
    const char* prompt;
    size_t context_length;
    size_t max_tokens;
    float temperature;
    bool verbose;
};

static inline double mach_timestamp_diff_to_seconds(uint64_t start_timestamp, uint64_t end_timestamp) {
    static mach_timebase_info_data_t timebase_info = {0};
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
    const uint64_t elapsed_mach_time = end_timestamp - start_timestamp;
    return ((double) elapsed_mach_time * (double) timebase_info.numer) / ((double) timebase_info.denom * 1.0e+9);
}

static inline uint64_t mach_timestamp_diff_to_microseconds(uint64_t start_timestamp, uint64_t end_timestamp) {
    static mach_timebase_info_data_t timebase_info = {0};
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
    const uint64_t elapsed_mach_time = end_timestamp - start_timestamp;
    const uint64_t denominator = timebase_info.denom * UINT64_C(1000);
    return (elapsed_mach_time * timebase_info.numer + denominator / 2) / denominator;
}

static void print_usage(const char* program_name) {
    printf("Usage: %s <model-path> [-p <prompt>] [-n <tokens>]\n", program_name);
}

struct options parse_options(int argc, char** argv) {
    struct options options = (struct options) {
        .model = NULL,
        .prompt = NULL,
        .context_length = 0,
        .max_tokens = 0,
        .temperature = 0.0f,
        .verbose = false,
    };
    if (argc < 2) {
        fprintf(stderr, "Error: missing required command-line argument\n");
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", argv[i]);
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            options.prompt = argv[++i];
        } else if (strcmp(argv[i], "--context-length") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for --context-length\n");
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            char* context_length_start = argv[++i];
            char* context_length_end = context_length_start;
            options.context_length = strtoul(context_length_start, &context_length_end, 10);
            if (context_length_end == context_length_start || *context_length_end != 0) {
                fprintf(stderr, "Error: failed to parse context length value \"%s\"\n", context_length_start);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", argv[i]);
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            char* max_tokens_start = argv[++i];
            char* max_tokens_end = max_tokens_start;
            options.max_tokens = strtoul(max_tokens_start, &max_tokens_end, 10);
            if (max_tokens_end == max_tokens_start || *max_tokens_end != 0) {
                fprintf(stderr, "Error: failed to max tokens value \"%s\"\n", max_tokens_start);
                exit(EXIT_FAILURE);
            }
            if (options.max_tokens == 0) {
                fprintf(stderr, "Error: invalid max tokens value %zu\n", options.max_tokens);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", argv[i]);
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            char* temperature_start = argv[++i];
            char* temperature_end = temperature_start;
            options.temperature = strtof(temperature_start, &temperature_end);
            if (temperature_end == temperature_start || *temperature_end != 0) {
                fprintf(stderr, "Error: failed to parse temperature value \"%s\"\n", temperature_start);
                exit(EXIT_FAILURE);
            }
            if (signbit(options.temperature) != 0 || !(options.temperature <= 2.0f)) {
                fprintf(stderr, "Error: invalid temperature value %f\n", options.temperature);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            options.verbose = true;
        } else {
            if (options.model == NULL) {
                options.model = argv[i];
            } else {
                fprintf(stderr, "Error: unexpected command-line argument %s\n", argv[i]);
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        }
    }
    if (options.model == NULL) {
        fprintf(stderr, "Error: missing required model argument\n");
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (options.prompt == NULL) {
        fprintf(stderr, "Error: missing required prompt argument\n");
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return options;
}


static void print_profile() {
    const size_t num_prefill_tokens = atomic_load(&globals.num_prefill_tokens);
    const uint64_t prefill_microseconds = atomic_load(&globals.prefill_microseconds);
    const size_t num_generated_tokens = atomic_load(&globals.num_generated_tokens);
    const uint64_t generation_microseconds = atomic_load(&globals.generation_microseconds);
    const uint64_t inference_bytes = atomic_load(&globals.inference_bytes);
    if (num_prefill_tokens != 0 || num_generated_tokens != 0) {
        printf("\n");
    }
    if (num_prefill_tokens != 0) {
        printf("Prefill speed (%zu tokens): %.1f tokens/second\n",
            num_prefill_tokens,
            (double) num_prefill_tokens / (double) prefill_microseconds * 1.0e+6);
    }
    if (num_generated_tokens != 0) {
        printf("Generation speed (%zu tokens): %.1f tokens/second\n",
            num_generated_tokens,
            (double) num_generated_tokens / (double) generation_microseconds * 1.0e+6);
    }
}

static void ctrl_c_handler(int signum) {
    print_profile();
    exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]) {
    enum gptoss_status status;
    gptoss_model_t model = NULL;
    gptoss_tokenizer_t tokenizer = NULL;
    gptoss_context_t context = NULL;

    struct sigaction act;
    act.sa_handler = ctrl_c_handler;
    sigaction(SIGINT, &act, NULL);

    setvbuf(stdout, NULL, _IONBF, 0);

    struct options options = parse_options(argc, argv);

    const uint64_t load_start_time = mach_continuous_time();
    status = gptoss_model_create_from_file(options.model, &model);
    if (status != gptoss_status_success) {
        fprintf(stderr, "Error: failed to load model from file %s\n", options.model);
        goto error;
    }
    size_t max_model_context_length = 0;
    status = gptoss_model_get_max_context_length(model, &max_model_context_length);
    if (status != gptoss_status_success) {
        fprintf(stderr, "Error: failed to query maximum context length\n");
        goto error;
    }
    assert(max_model_context_length != 0);
    if (options.context_length == 0) {
        options.context_length = max_model_context_length;
    } else if (options.context_length > max_model_context_length) {
        fprintf(stderr, "Error: context length %zu exceeds maximum context length %zu supported by the model\n", options.context_length, max_model_context_length);
        goto error;
    }

    status = gptoss_model_get_tokenizer(model, &tokenizer);
    if (status != gptoss_status_success) {
        fprintf(stderr, "Error: failed to retrieve Tokenizer\n");
        goto error;
    }

    uint32_t return_token_id = UINT32_MAX;
    status = gptoss_tokenizer_get_special_token_id(tokenizer, gptoss_special_token_return, &return_token_id);
    if (status != gptoss_status_success) {
        fprintf(stderr, "Error: failed to query end-of-text token ID\n");
        goto error;
    }

    status = gptoss_context_create(model, options.context_length, /*max_batch_tokens=*/0, &context);
    if (status != gptoss_status_success) {
        fprintf(stderr, "Error: failed to create Context object\n");
        goto error;
    }
    if (options.verbose) {
        printf("Model weights size: %.2lf MB\n", (double) model->weights_size * 0x1.0p-20);
        printf("Model allocation size: %.2lf MB\n", (double) model->allocation_size * 0x1.0p-20);
        printf("Context allocation size: %.2lf MB\n", (double) context->allocation_size * 0x1.0p-20);
        printf("  Including KV cache: %.2lf MB\n", (double) context->kvcache_size * 0x1.0p-20);
    }

    const uint64_t load_end_time = mach_continuous_time();
    const double load_elapsed_seconds = mach_timestamp_diff_to_seconds(load_start_time, load_end_time);
    if (options.verbose) {
        printf("Loaded model in %.3f seconds\n", load_elapsed_seconds);
    }

    const uint64_t prefill_start_time = mach_continuous_time();
    size_t num_prefill_tokens = 0;
    status = gptoss_context_append_chars(context, options.prompt, strlen(options.prompt), &num_prefill_tokens);
    if (status != gptoss_status_success) {
        fprintf(stderr, "Error: failed to tokenize prompt \"%s\"\n", options.prompt);
        goto error;
    }
    atomic_store(&globals.num_prefill_tokens, num_prefill_tokens);
    status = gptoss_context_process(context);
    if (status != gptoss_status_success) {
        fprintf(stderr, "Error: failed to process Context object\n");
        goto error;
    }
    const uint64_t prefill_end_time = mach_continuous_time();

    while (options.max_tokens == 0 || atomic_load(&globals.num_generated_tokens) < options.max_tokens) {

        uint32_t predicted_token = UINT32_MAX;
        size_t num_predicted_tokens = 0;
        const uint64_t inference_start_timestamp = mach_continuous_time();
        status = gptoss_context_sample(context, options.temperature, /*rng_state=*/0, /*num_tokens=*/1, &predicted_token, &num_predicted_tokens);
        if (status != gptoss_status_success) {
            fprintf(stderr, "Error: failed to sample from the Context object\n");
            goto error;
        }
        const uint64_t inference_end_timestamp = mach_continuous_time();

        if (predicted_token == return_token_id) {
            // Yield token -> stop generation
            break;
        }

        // Unembedding: detokenize
        size_t token_size = 0;
        const void* token_ptr = NULL;
        status = gptoss_tokenizer_decode(tokenizer, predicted_token, &token_ptr, &token_size);
        if (status != gptoss_status_success) {
            fprintf(stderr, "Error: failed to detokenize predicted token %" PRIu32 "\n", predicted_token);
            goto error;
        }
        const size_t previous_num_generated_tokens = atomic_fetch_add(&globals.num_generated_tokens, 1);
        if (previous_num_generated_tokens == 0) {
            atomic_fetch_add(&globals.prefill_microseconds, mach_timestamp_diff_to_microseconds(prefill_start_time, prefill_end_time));
        } else {
            atomic_fetch_add(&globals.generation_microseconds, mach_timestamp_diff_to_microseconds(inference_start_timestamp, inference_end_timestamp));
        }
        printf("%.*s", (int) token_size, (const char*) token_ptr);

        status = gptoss_context_append_tokens(context, 1, &predicted_token);
        if (status != gptoss_status_success) {
            fprintf(stderr, "Error: failed to append predicted token %" PRIu32 " to context\n", predicted_token);
            goto error;
        }
    }

    print_profile();

    return EXIT_SUCCESS;

error:
    gptoss_context_release(context);
    gptoss_tokenizer_release(tokenizer);
    gptoss_model_release(model);
    return EXIT_FAILURE;
}
