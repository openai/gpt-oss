#pragma once

#include <stddef.h>
#include <stdint.h>

#include <gpt-oss/macros.h>
#include <gpt-oss/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Creates a Model object from a file in the filesystem.
 *
 * @param path Path to the file containing the model in GPT-OSS format.
 * @param model_out Pointer to the Model object that will be created. Must be released with gptoss_release_model.
 *
 * On success, returns gptoss_status_success and saves a pointer to the created Model in the model_out argument.
 * On failure, returns an error code and stores null pointer in the model_out argument.
 */
enum gptoss_status GPTOSS_ABI gptoss_model_create_from_file(
    const char* path,
    gptoss_model_t* model_out);

/*
 * Query the Tokenizer object associated with the Model.
 *
 * @param model Pointer to the Model object created by gptoss_model_create_from_file.
 * @param tokenizer_out Pointer to the variable where the Tokenizer reference will be stored.
 *
 * On success, returns gptoss_status_success and stores reference to the Tokenizer object in the tokenizer_out argument.
 * On failure, returns an error code and stores NULL in the tokenizer_out argument.
 */
enum gptoss_status GPTOSS_ABI gptoss_model_get_tokenizer(
    gptoss_model_t model,
    gptoss_tokenizer_t* tokenizer_out);

/*
 * Query the maximum context length supported by the Model.
 *
 * @param model Pointer to the Model object created by gptoss_model_create_from_file.
 * @param max_context_length_out Pointer to the variable where the maximum context length will be stored.
 *
 * On success, returns gptoss_status_success and stores maximum context length in the max_context_length_out argument.
 * On failure, returns an error code and leaves the value specified by max_context_length_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_model_get_max_context_length(
    gptoss_model_t model,
    size_t* max_context_length_out);

/*
 * Increments a Model object's reference count.
 *
 * @param model Pointer to the Model object created by gptoss_model_create_from_file.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_model_retain(
    gptoss_model_t model);

/*
 * Decrements a Model object's reference count and possibly release associated resources.
 *
 * @param model Pointer to the Model object created by gptoss_model_create_from_file.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_model_release(
    gptoss_model_t model);

/*
 * Query the token ID for a special token in the Tokenizer vocabulary.
 *
 * @param tokenizer Pointer to the Tokenizer object created by gptoss_model_get_tokenizer.
 * @param token_type Type of the special token to query an ID for.
 * @param token_id_out Pointer to the variable where the token ID will be stored.
 *
 * On success, returns gptoss_status_success and stores the token ID in the token_id_out argument.
 * On failure, returns an error code and leaves the value specified by token_id_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_special_token_id(
    gptoss_tokenizer_t tokenizer,
    enum gptoss_special_token token_type,
    uint32_t* token_id_out);

/*
 * Query the number of text tokens in the Tokenizer vocabulary.
 *
 * @param tokenizer Pointer to the Tokenizer object created by gptoss_model_get_tokenizer.
 * @param num_text_tokens_out Pointer to the variable where the number of text tokens will be stored.
 *
 * On success, returns gptoss_status_success and stores the number of text tokens in the num_text_tokens_out argument.
 * On failure, returns an error code and leaves the value specified by num_text_tokens_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_num_text_tokens(
    gptoss_tokenizer_t tokenizer,
    uint32_t* num_text_tokens_out);

/*
 * Query the number of special tokens in the Tokenizer vocabulary.
 *
 * @param tokenizer Pointer to the Tokenizer object created by gptoss_model_get_tokenizer.
 * @param num_special_tokens_out Pointer to the variable where the number of special tokens will be stored.
 *
 * On success, returns gptoss_status_success and stores the number of text tokens in the num_special_tokens_out argument.
 * On failure, returns an error code and leaves the value specified by num_special_tokens_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_num_special_tokens(
    gptoss_tokenizer_t tokenizer,
    uint32_t* num_special_tokens_out);

/*
 * Query the total number of tokens in the Tokenizer vocabulary.
 *
 * @param tokenizer Pointer to the Tokenizer object created by gptoss_model_get_tokenizer.
 * @param num_tokens_out Pointer to the variable where the total number of tokens will be stored.
 *
 * On success, returns gptoss_status_success and stores the total number of tokens in the num_special_tokens_out argument.
 * On failure, returns an error code and leaves the value specified by num_special_tokens_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_tokenizer_get_num_tokens(
    gptoss_tokenizer_t tokenizer,
    uint32_t* num_tokens_out);

/*
 * Convert a text token ID to byte representation.
 *
 * @param tokenizer Pointer to the Tokenizer object returned by gptoss_model_get_tokenizer. The lifetime of the returned
 *                  byte representation would match the lifetime of this Tokenizer object.
 * @param token_ptr_out Pointer to the variable where the pointer to the byte representation of the token will be
 *                      stored.
 * @param token_size_out Pointer to the variable where the size of the byte representation of the token will be stored.
 *
 * On success, returns gptoss_status_success and stores pointer and size of the byte representation of the token in the
 *                     token_ptr_out and token_size_out arguments.
 * On failure, returns an error code and leaves the values specified in token_ptr_out and token_size_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_tokenizer_decode(
    gptoss_tokenizer_t tokenizer,
    uint32_t token_id,
    const void** token_ptr_out,
    size_t* token_size_out);

/*
 * Increments a Tokenizer object's reference count.
 *
 * @param tokenizer Pointer to the Tokenizer object returned by gptoss_model_get_tokenizer.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_tokenizer_retain(
    gptoss_tokenizer_t tokenizer);

/*
 * Decrements a Tokenizer object's reference count and possibly release associated resources.
 *
 * @param tokenizer Pointer to the Tokenizer object returned by gptoss_model_get_tokenizer.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_tokenizer_release(
    gptoss_tokenizer_t tokenizer);

/*
 * Creates a Context object for use with the particular Model object.
 *
 * @param model Model object to create a context for.
 * @param context_length Maximum number of tokens in the context.
 *                       Specify 0 to use the maximum context length supported by the model.
 * @param batch_size Maximum number of tokens that can be processed in a single batch.
 *                   Larger values may improve performance, but require more memory.
 * @param context_out Pointer to the Context object that will be created.
 *                    Must be released with gptoss_release_context.
 *
 * On success, returns gptoss_status_success and saves a pointer to the created Context in the context_out argument.
 * On failure, returns an error code and stores null pointer in the context_out argument.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_create(
    gptoss_model_t model,
    size_t context_length,
    gptoss_context_t* context_out);

/*
 * Query the current number of tokens cached in the Context.
 *
 * @param context Pointer to the Context object created by gptoss_context_create.
 * @param num_tokens_out Pointer to the variable where the current number of cached tokens will be stored.
 *
 * On success, returns gptoss_status_success and stores current number of cached tokens in the num_tokens_out argument.
 * On failure, returns an error code and leaves the value specified by num_tokens_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_get_num_tokens(
    gptoss_context_t context,
    size_t* num_tokens_out);

/*
 * Query the maximum number of tokens cached in the Context.
 *
 * @param context Pointer to the Context object created by gptoss_context_create.
 * @param max_tokens_out Pointer to the variable where the maximum number of cached tokens will be stored.
 *
 * On success, returns gptoss_status_success and stores maximum number of cached tokens in the max_tokens_out argument.
 * On failure, returns an error code and leaves the value specified by max_tokens_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_get_max_tokens(
    gptoss_context_t context,
    size_t* max_tokens_out);

/*
 * Query the list of token IDs cached in the Context.
 *
 * @param context Pointer to the Context object created by gptoss_context_create.
 * @param tokens_out Pointer to the array where up to max_tokens_out of cached tokens will be stored.
 * @param max_tokens Maximum capacity of the buffer specified by tokens_out.
 * @param num_tokens_out Pointer to the variable where the actual number of cached tokens will be stored.
 *                       This value can exceed max_tokens if the buffer capacity is insufficient.
 *
 * On success, returns gptoss_status_success and stores cached token IDs in the tokens_out argument and the number of
 * cached tokens in the num_tokens_out argument.
 * On failure, returns an error code and leaves the values specified by tokend_out and num_tokens_out unchanged.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_get_tokens(
    gptoss_context_t context,
    uint32_t* tokens_out,
    size_t max_tokens,
    size_t* num_tokens_out);

/*
 * Tokenize and appends a character string to the Context object.
 *
 * @param context Context object created by gptoss_context_create.
 * @param text Pointer to the character string to tokenizer and append.
 * @param text_length Length of the string, in chars.
 * @param num_tokens_out Optional pointer to the variable where the number of appended tokens will be stored. Ignored if a null pointer is provided.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_append_chars(
    gptoss_context_t context,
    const char* text,
    size_t text_length,
    size_t* num_tokens_out);

/*
 * Appends a list of tokens to the context.
 *
 * @param context Context object created by gptoss_context_create.
 * @param num_tokens Number of tokens to be appended.
 * @param tokens Pointer to the array of tokens to be appended.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_append_tokens(
    gptoss_context_t context,
    size_t num_tokens,
    const uint32_t* tokens);

/*
 * Resets the context, clearing its state.
 *
 * @param context Context object created by gptoss_context_create.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_reset(
    gptoss_context_t context);

/*
 * Pre-process the tokens in the Context and generate probability distribution over the next token.
 *
 * @param context Context object created by gptoss_context_create.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_process(
    gptoss_context_t context);

/*
 * Generate a token probability distribution over the next token conditioned on the Context.
 *
 * @param context Context object created by gptoss_context_create.
 * @param temperature Sampling temperature. Must be non-negative.
 * @param seed Random number generator seed to use for sampling.
 * @param token_out Pointer to the variable where the token ID will be stored.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_sample(
    gptoss_context_t context,
    float temperature,
    uint64_t seed,
    uint32_t* token_out);

/*
 * Increments a Context object's reference count.
 *
 * @param context Pointer to the Context object created by gptoss_create_context.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_retain(
    gptoss_context_t context);

/*
 * Decrements a Context object's reference count and possibly release associated resources.
 *
 * @param context Pointer to the Context object created by gptoss_create_context.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_context_release(
    gptoss_context_t context);

/*
 * Creates a Sampler object.
 *
 * @param sampler_out Pointer to the Sampler object that will be created.
 *                    Must be released with gptoss_sampler_release.
 *
 * On success, returns gptoss_status_success and saves a pointer to the created Sampler in the sampler_out argument.
 * On failure, returns an error code and stores a null pointer in the sampler_out argument.
 */
enum gptoss_status GPTOSS_ABI gptoss_sampler_create(
    gptoss_sampler_t* sampler_out);

/*
 * Sets the sampling temperature for the Sampler.
 *
 * @param sampler Sampler object created by gptoss_sampler_create.
 * @param temperature Temperature value to be set. Must be in the [0.0, 1.0] range.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_sampler_set_temperature(
    gptoss_sampler_t sampler,
    float temperature);

/*
 * Sets the Top-P nucleus sampling parameter for the Sampler.
 *
 * @param sampler Sampler object created by gptoss_sampler_create.
 * @param top_p Top-P value to be set. Must be in the (0.0, 1.0] range.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_sampler_set_top_p(
    gptoss_sampler_t sampler,
    float top_p);

/*
 * Sets the presence penalty for the Sampler.
 *
 * @param sampler Sampler object created by gptoss_sampler_create.
 * @param presence_penalty Presence penalty value to be set. Must be in the [-2.0, 2.0] range.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_sampler_set_presence_penalty(
    gptoss_sampler_t sampler,
    float presence_penalty);

/*
 * Sets the frequency penalty for the Sampler.
 *
 * @param sampler Sampler object created by gptoss_sampler_create.
 * @param frequency_penalty Frequency penalty value to be set. Must be in the [-2.0, 2.0] range.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_sampler_set_frequency_penalty(
    gptoss_sampler_t sampler,
    float frequency_penalty);

/*
 * Increments a Sampler object's reference count.
 *
 * @param sampler Pointer to the Sampler object created by gptoss_sampler_create.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_sampler_retain(
    gptoss_sampler_t sampler);

/*
 * Decrements a Sampler object's reference count and possibly releases associated resources.
 *
 * @param sampler Pointer to the Sampler object created by gptoss_sampler_create.
 *
 * On success, returns gptoss_status_success, otherwise returns an error code.
 */
enum gptoss_status GPTOSS_ABI gptoss_sampler_release(
    gptoss_sampler_t sampler);

#ifdef __cplusplus
}  // extern "C"
#endif
