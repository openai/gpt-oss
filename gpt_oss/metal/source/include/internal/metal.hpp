#pragma once

#include <array>
#include <initializer_list>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <gpt-oss/types.h>
#include <internal/metal.h>
#include <internal/metal-kernels.h>


namespace gptoss {

inline void Check(gptoss_status s, const char* what) {
    if (s != gptoss_status_success) {
        throw std::runtime_error(what);
    }
}

inline std::size_t round_up(std::size_t p, std::size_t q) {
    const std::size_t r = p % q;
    if (r == 0) {
        return p;
    } else {
        return p - r + q;
    }
}

namespace metal {

class Device {
public:
    inline Device() {
        Check(gptoss_metal_device_create_system_default(&device_), "create Device");
    }

    inline ~Device() {
        gptoss_metal_device_release(&device_);
    }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    inline Device(Device&& other) noexcept {
        device_ = other.device_;
        std::memset(&other.device_, 0, sizeof(other.device_));
    }

    inline Device& operator=(Device&& other) noexcept {
        if (this != &other) {
            gptoss_metal_device_release(&device_);
            device_ = other.device_;
            std::memset(&other.device_, 0, sizeof(other.device_));
        }
        return *this;
    }

    inline const gptoss_metal_device* handle() const noexcept { return &device_; }

    inline size_t max_buffer_size() const noexcept { return device_.max_buffer_size; }
    inline size_t max_threadgroup_memory() const noexcept { return device_.max_threadgroup_memory; }
    inline size_t max_threadgroup_threads_x() const noexcept { return device_.max_threadgroup_threads_x; }
    inline size_t max_threadgroup_threads_y() const noexcept { return device_.max_threadgroup_threads_y; }
    inline size_t max_threadgroup_threads_z() const noexcept { return device_.max_threadgroup_threads_z; }

private:
    gptoss_metal_device device_{};
};

class Library {
public:
    inline explicit Library(const Device& dev) {
        Check(gptoss_metal_library_create_default(dev.handle(), &library_),
            "gptoss_metal_library_create_default");
    }

    inline ~Library() {
        gptoss_metal_library_release(&library_);
    }

    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    inline Library(Library&& other) noexcept {
        library_ = other.library_;
        std::memset(&other.library_, 0, sizeof(other.library_));
    }

    inline Library& operator=(Library&& other) noexcept {
        if (this != &other) {
            gptoss_metal_library_release(&library_);
            library_ = other.library_;
            std::memset(&other.library_, 0, sizeof(other.library_));
        }
        return *this;
    }

    inline const gptoss_metal_library* handle() const noexcept {
        return &library_;
    }

private:
    gptoss_metal_library library_{};
};

class Function {
public:
    inline Function(const Library& library, const char* name) {
        Check(gptoss_metal_function_create(library.handle(), name, &function_),
            "gptoss_metal_function_create");
    }

    inline ~Function() {
        gptoss_metal_function_release(&function_);
    }

    Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;

    inline Function(Function&& other) noexcept {
        function_ = other.function_;
        std::memset(&other.function_, 0, sizeof(other.function_));
    }

    inline Function& operator=(Function&& other) noexcept {
        if (this != &other) {
            gptoss_metal_function_release(&function_);
            function_ = other.function_;
            std::memset(&other.function_, 0, sizeof(other.function_));
        }
        return *this;
    }

    inline const gptoss_metal_function* handle() const noexcept { return &function_; }

    inline size_t max_threadgroup_threads() const noexcept { return function_.max_threadgroup_threads; }
    inline size_t simdgroup_threads() const noexcept { return function_.simdgroup_threads; }
    inline size_t static_threadgroup_memory() const noexcept { return function_.static_threadgroup_memory; }

private:
    gptoss_metal_function function_{};
};

class Buffer {
public:
    inline Buffer(const Device& dev, size_t size, const void* data = nullptr) {
        Check(gptoss_metal_buffer_create(dev.handle(), size, data, &buffer_), "create buffer");
    }

    inline ~Buffer() {
        gptoss_metal_buffer_release(&buffer_);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    inline Buffer(Buffer&& other) noexcept {
        buffer_ = other.buffer_;
        std::memset(&other.buffer_, 0, sizeof(other.buffer_));
    }

    inline Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            gptoss_metal_buffer_release(&buffer_);
            buffer_ = other.buffer_;
            std::memset(&other.buffer_, 0, sizeof(other.buffer_));
        }
        return *this;
    }

    inline size_t size() const noexcept { return buffer_.size; }
    inline void* ptr() const noexcept { return buffer_.ptr; }

    inline const gptoss_metal_buffer* handle() const noexcept { return &buffer_; }

private:
    gptoss_metal_buffer buffer_{};
};

class CommandQueue {
public:
    inline explicit CommandQueue(const Device& dev) {
        Check(gptoss_metal_command_queue_create(dev.handle(), &command_queue_),
            "gptoss_metal_command_queue_create");
    }

    inline ~CommandQueue() {
        gptoss_metal_command_queue_release(&command_queue_);
    }

    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;

    inline CommandQueue(CommandQueue&& other) noexcept {
        command_queue_ = other.command_queue_;
        std::memset(&other.command_queue_, 0, sizeof(other.command_queue_));
    }

    inline CommandQueue& operator=(CommandQueue&& other) noexcept {
        if (this != &other) {
            gptoss_metal_command_queue_release(&command_queue_);
            command_queue_ = other.command_queue_;
            std::memset(&other.command_queue_, 0, sizeof(other.command_queue_));
        }
        return *this;
    }

    inline const gptoss_metal_command_queue* handle() const noexcept {
        return &command_queue_;
    }

private:
    gptoss_metal_command_queue command_queue_{};
};

class CommandBuffer {
public:
    inline explicit CommandBuffer(const CommandQueue& command_queue) {
        Check(gptoss_metal_command_buffer_create(command_queue.handle(), &command_buffer_),
            "gptoss_metal_command_buffer_create");
    }
    inline ~CommandBuffer() {
        gptoss_metal_command_buffer_release(&command_buffer_);
    }

    CommandBuffer(const CommandBuffer&)            = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    inline CommandBuffer(CommandBuffer&& other) noexcept  {
        command_buffer_ = other.command_buffer_;
        std::memset(&other.command_buffer_, 0, sizeof(other.command_buffer_));
    }

    inline CommandBuffer& operator=(CommandBuffer&& other) noexcept {
        if (this != &other) {
            gptoss_metal_command_buffer_release(&command_buffer_);
            command_buffer_ = other.command_buffer_;
            std::memset(&other.command_buffer_, 0, sizeof(other.command_buffer_));
        }
        return *this;
    }

    inline void encode_launch_kernel(const Function& function,
                                     const std::array<size_t, 3>& threadgroup_size,
                                     const std::array<size_t, 3>& num_threadgroups,
                                     size_t params_size, const void* params,
                                     std::initializer_list<const Buffer*> device_buffers = {},
                                     size_t threadgroup_buffer_size = 0)
    {
        std::vector<const gptoss_metal_buffer*> buffer_handles(device_buffers.size());
        std::transform(device_buffers.begin(), device_buffers.end(), buffer_handles.begin(),
            [](const Buffer* buffer) -> const gptoss_metal_buffer* { return buffer->handle(); });
        Check(gptoss_metal_command_buffer_encode_launch_kernel(
                &command_buffer_, function.handle(),
                threadgroup_size[0], threadgroup_size[1], threadgroup_size[2],
                num_threadgroups[0], num_threadgroups[1], num_threadgroups[2],
                params_size, params,
                buffer_handles.size(),
                buffer_handles.data(),
                /*buffer_offsets=*/nullptr,
                threadgroup_buffer_size),
            "gptoss_metal_command_buffer_encode_launch_kernel");
    }

    inline void encode_launch_f32_fill_random(const Function& f32_fill_random_fn,
                                              size_t threadgroup_size,
                                              size_t num_threadgroups,
                                              const Buffer& output_buffer,
                                              size_t output_offset,
                                              size_t num_channels,
                                              uint64_t rng_seed,
                                              uint64_t rng_offset,
                                              float rng_min,
                                              float rng_max)
    {
        Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
                &command_buffer_, f32_fill_random_fn.handle(),
                threadgroup_size, num_threadgroups,
                output_buffer.handle(), output_offset,
                num_channels,
                rng_seed, rng_offset, rng_min, rng_max),
            "gptoss_metal_command_buffer_encode_launch_f32_fill_random");
    }

    inline void encode_launch_bf16_fill_random(const Function& bf16_fill_random_fn,
                                               size_t threadgroup_size,
                                               size_t num_threadgroups,
                                               const Buffer& output_buffer,
                                               size_t output_offset,
                                               size_t num_channels,
                                               uint64_t rng_seed,
                                               uint64_t rng_offset,
                                               float rng_min,
                                               float rng_max)
    {
        Check(gptoss_metal_command_buffer_encode_launch_bf16_fill_random(
                &command_buffer_, bf16_fill_random_fn.handle(),
                threadgroup_size, num_threadgroups,
                output_buffer.handle(), output_offset,
                num_channels,
                rng_seed, rng_offset, rng_min, rng_max),
            "gptoss_metal_command_buffer_encode_launch_bf16_fill_random");
    }

    inline void encode_launch_u32_fill_random(const Function& u32_fill_random_fn,
                                              size_t threadgroup_size,
                                              size_t num_threadgroups,
                                              const Buffer& output_buffer,
                                              size_t output_offset,
                                              size_t num_channels,
                                              uint64_t rng_seed,
                                              uint64_t rng_offset)
    {
        Check(gptoss_metal_command_buffer_encode_launch_u32_fill_random(
                &command_buffer_, u32_fill_random_fn.handle(),
                threadgroup_size, num_threadgroups,
                output_buffer.handle(), output_offset,
                num_channels,
                rng_seed, rng_offset),
            "gptoss_metal_command_buffer_encode_launch_u32_fill_random");
    }

    inline void commit() {
        Check(gptoss_metal_command_buffer_commit(&command_buffer_), "commit");
    }

    inline double wait_completion() {
        double secs = 0.0;
        Check(gptoss_metal_command_buffer_wait_completion(&command_buffer_, &secs), "wait completion");
        return secs;
    }

    inline const gptoss_metal_command_buffer* handle() const noexcept { return &command_buffer_; }

private:
    gptoss_metal_command_buffer command_buffer_{};
};

} // namespace metal
} // namespace gptoss
