#include <gpt-oss.h>
#include <internal/model.h>

#include <array>
#include <cstdint>
#include <cstddef>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

#include <benchmark/benchmark.h>


constexpr std::uint32_t kNumGeneratedTokens = 100;


static void attn_qkv_tgsize(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);
    model->attn_qkv_threadgroup_size = static_cast<std::size_t>(state.range(0));

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_length=*/0, /*max_batch_tokens=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

static void AttnQKVThreadgroupSizeArguments(benchmark::internal::Benchmark* b) {
    b->ArgNames({"tgsize"});
    for (auto attn_qkv_threadgroup_size = 32; attn_qkv_threadgroup_size <= 1024; attn_qkv_threadgroup_size += 32) {
        const auto num_simdgroups = attn_qkv_threadgroup_size / 32;
        if (5120 % num_simdgroups != 0) {
            // Skip incompatible threadgroup sizes
            continue;
        }
        b->Args({attn_qkv_threadgroup_size});
    }
}

BENCHMARK_CAPTURE(attn_qkv_tgsize, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(AttnQKVThreadgroupSizeArguments);
BENCHMARK_CAPTURE(attn_qkv_tgsize, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(AttnQKVThreadgroupSizeArguments);

static void attn_out_tgsize(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);
    model->attn_out_threadgroup_size = static_cast<std::size_t>(state.range(0));

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_length=*/0, /*max_batch_tokens=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

static void AttnOutThreadgroupSizeArguments(benchmark::internal::Benchmark* b) {
    b->ArgNames({"tgsize"});
    for (auto attn_out_threadgroup_size = 32; attn_out_threadgroup_size <= 1024; attn_out_threadgroup_size += 32) {
        const auto num_simdgroups = attn_out_threadgroup_size / 32;
        if (2880 % num_simdgroups != 0) {
            // Skip incompatible threadgroup sizes
            continue;
        }
        b->Args({attn_out_threadgroup_size});
    }
}

BENCHMARK_CAPTURE(attn_out_tgsize, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(AttnOutThreadgroupSizeArguments);
BENCHMARK_CAPTURE(attn_out_tgsize, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(AttnOutThreadgroupSizeArguments);

static void mlp_gate_tgsize(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);
    model->mlp_gate_threadgroup_size = static_cast<std::size_t>(state.range(0));

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_length=*/0, /*max_batch_tokens=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

static void MlpGateThreadgroupSizeArguments(benchmark::internal::Benchmark* b) {
    b->ArgNames({"tgsize"});
    for (auto mlp_gate_threadgroup_size = 32; mlp_gate_threadgroup_size <= 1024; mlp_gate_threadgroup_size += 32) {
        const auto num_simdgroups = mlp_gate_threadgroup_size / 32;
        if (128 % num_simdgroups != 0) {
            // Skip incompatible threadgroup sizes
            continue;
        }
        b->Args({mlp_gate_threadgroup_size});
    }
}

BENCHMARK_CAPTURE(mlp_gate_tgsize, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpGateThreadgroupSizeArguments);
BENCHMARK_CAPTURE(mlp_gate_tgsize, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpGateThreadgroupSizeArguments);

static void mlp_swiglu_tgsize(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);
    model->mlp_swiglu_threadgroup_size = static_cast<std::size_t>(state.range(0));

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_length=*/0, /*max_batch_tokens=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

static void MlpSwigluThreadgroupSizeArguments(benchmark::internal::Benchmark* b) {
    b->ArgNames({"tgsize"});
    for (auto threadgroup_size = 64; threadgroup_size <= 1024; threadgroup_size += 64) {
        const auto num_simdgroups = threadgroup_size / 32;
        if (5760 % num_simdgroups != 0) {
            // Skip incompatible threadgroup sizes
            continue;
        }
        b->Args({threadgroup_size});
    }
}

BENCHMARK_CAPTURE(mlp_swiglu_tgsize, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpSwigluThreadgroupSizeArguments);
BENCHMARK_CAPTURE(mlp_swiglu_tgsize, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpSwigluThreadgroupSizeArguments);

static void mlp_out_tgsize(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);
    model->mlp_out_threadgroup_size = static_cast<std::size_t>(state.range(0));

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_length=*/0, /*max_batch_tokens=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

static void MlpOutThreadgroupSizeArguments(benchmark::internal::Benchmark* b) {
    b->ArgNames({"tgsize"});
    for (auto threadgroup_size = 64; threadgroup_size <= 1024; threadgroup_size += 64) {
        const auto num_simdgroups = threadgroup_size / 32;
        if (5760 % num_simdgroups != 0) {
            // Skip incompatible threadgroup sizes
            continue;
        }
        b->Args({threadgroup_size});
    }
}

BENCHMARK_CAPTURE(mlp_out_tgsize, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpOutThreadgroupSizeArguments);
BENCHMARK_CAPTURE(mlp_out_tgsize, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpOutThreadgroupSizeArguments);

static void mlp_acc_tgsize(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);
    model->mlp_acc_threadgroup_size = static_cast<std::size_t>(state.range(0));

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_length=*/0, /*max_batch_tokens=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

static void MlpAccThreadgroupSizeArguments(benchmark::internal::Benchmark* b) {
    b->ArgNames({"tgsize"});
    for (auto threadgroup_size = 32; threadgroup_size <= 1024; threadgroup_size += 32) {
        b->Args({threadgroup_size});
    }
}

BENCHMARK_CAPTURE(mlp_acc_tgsize, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpAccThreadgroupSizeArguments);
BENCHMARK_CAPTURE(mlp_acc_tgsize, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(MlpAccThreadgroupSizeArguments);

static void unembedding_tgsize(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);
    model->unembedding_threadgroup_size = static_cast<std::size_t>(state.range(0));

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_length=*/0, /*max_batch_tokens=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

static void UnembeddingThreadgroupSizeArguments(benchmark::internal::Benchmark* b) {
    b->ArgNames({"tgsize"});
    for (auto threadgroup_size = 32; threadgroup_size <= 1024; threadgroup_size += 32) {
        b->Args({threadgroup_size});
    }
}

BENCHMARK_CAPTURE(unembedding_tgsize, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(UnembeddingThreadgroupSizeArguments);
BENCHMARK_CAPTURE(unembedding_tgsize, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond)->Apply(UnembeddingThreadgroupSizeArguments);

BENCHMARK_MAIN();
