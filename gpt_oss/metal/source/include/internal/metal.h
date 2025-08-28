#pragma once

#include <stddef.h>

#include <gpt-oss/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct gptoss_metal_device {
    void* object; // id<MTLDevice>
    size_t num_cores;
    size_t max_buffer_size;
    size_t max_threadgroup_memory;
    size_t max_threadgroup_threads_x;
    size_t max_threadgroup_threads_y;
    size_t max_threadgroup_threads_z;
};

enum gptoss_status gptoss_metal_device_create_system_default(
    struct gptoss_metal_device* device_out);

enum gptoss_status gptoss_metal_device_release(
    struct gptoss_metal_device* device);


struct gptoss_metal_library {
    void* object; // id<MTLLibrary>
};

enum gptoss_status gptoss_metal_library_create_default(
    const struct gptoss_metal_device* device,
    struct gptoss_metal_library* library_out);

enum gptoss_status gptoss_metal_library_release(
    struct gptoss_metal_library* library);

struct gptoss_metal_function {
    void* function_object; // id<MTLFunction>
    void* pipeline_state_object; // id<MTLComputePipelineState>
    size_t max_threadgroup_threads;
    size_t simdgroup_threads;
    size_t static_threadgroup_memory;
};

enum gptoss_status gptoss_metal_function_create(
    const struct gptoss_metal_library* library,
    const char* name,
    struct gptoss_metal_function* function_out);

enum gptoss_status gptoss_metal_function_release(
    struct gptoss_metal_function* function);

struct gptoss_metal_buffer {
    void* object; // id<MTLBuffer>
    size_t size;
    void* ptr;
};

enum gptoss_status gptoss_metal_buffer_create(
    const struct gptoss_metal_device* device,
    size_t size,
    const void* data,
    struct gptoss_metal_buffer* buffer_out);

enum gptoss_status gptoss_metal_buffer_wrap(
    const struct gptoss_metal_device* device,
    size_t size,
    const void* data,
    struct gptoss_metal_buffer* buffer_out);

enum gptoss_status gptoss_metal_buffer_release(
    struct gptoss_metal_buffer* buffer);

struct gptoss_metal_command_queue {
    void* object; // id<MTLCommandQueue>
};

enum gptoss_status gptoss_metal_command_queue_create(
    const struct gptoss_metal_device* device,
    struct gptoss_metal_command_queue* command_queue_out);

enum gptoss_status gptoss_metal_command_queue_release(
    struct gptoss_metal_command_queue* command_queue);

struct gptoss_metal_command_buffer {
    void* object; // id<MTLCommandBuffer>
};

enum gptoss_status gptoss_metal_command_buffer_create(
    const struct gptoss_metal_command_queue* command_queue,
    struct gptoss_metal_command_buffer* command_buffer_out);

enum gptoss_status gptoss_metal_command_buffer_encode_fill_buffer(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_buffer* buffer,
    size_t offset,
    size_t size,
    uint8_t fill_value);

enum gptoss_status gptoss_metal_command_buffer_encode_copy_buffer(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    size_t size);

enum gptoss_status gptoss_metal_command_buffer_encode_launch_kernel(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* function,
    size_t threadgroup_size_x,
    size_t threadgroup_size_y,
    size_t threadgroup_size_z,
    size_t num_threadgroups_x,
    size_t num_threadgroups_y,
    size_t num_threadgroups_z,
    size_t params_size,
    const void* params,
    size_t num_device_buffers,
    const struct gptoss_metal_buffer** device_buffers,
    const size_t* device_buffer_offsets,
    size_t threadgroup_buffer_size);

enum gptoss_status gptoss_metal_command_buffer_commit(
    const struct gptoss_metal_command_buffer* command_buffer);

enum gptoss_status gptoss_metal_command_buffer_wait_completion(
    const struct gptoss_metal_command_buffer* command_buffer,
    double* elapsed_seconds);

enum gptoss_status gptoss_metal_command_buffer_release(
    struct gptoss_metal_command_buffer* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
