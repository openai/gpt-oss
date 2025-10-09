#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <mach-o/getsect.h>

#include <gpt-oss/types.h>

#include <internal/log.h>
#include <internal/metal.h>


static size_t gptoss_metal_device_get_core_count(id<MTLDevice> device) {
    if (!device) {
        return 0;
    }

    const uint64_t target_registry_id = [device registryID];

    io_iterator_t it = IO_OBJECT_NULL;
    const kern_return_t kr = IOServiceGetMatchingServices(
        kIOMainPortDefault,
        IOServiceMatching("IOAccelerator"),
        &it
    );
    if (kr != KERN_SUCCESS) {
        GPTOSS_LOG_ERROR("failed to find IOAccelerator objects: error %d", kr);
        return 0;
    }

    size_t result = 0;
    for (io_object_t obj = IOIteratorNext(it); obj != IO_OBJECT_NULL; obj = IOIteratorNext(it)) {
        uint64_t registry_id = 0;
        if (IORegistryEntryGetRegistryEntryID(obj, &registry_id) == KERN_SUCCESS &&
            registry_id == target_registry_id)
        {
            // Read "gpu-core-count" from this accelerator node
            const CFTypeRef value = IORegistryEntryCreateCFProperty(
                obj, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
            if (value != NULL) {
                if (CFGetTypeID(value) == CFNumberGetTypeID()) {
                    int32_t n = -1;
                    if (CFNumberGetValue((CFNumberRef) value, kCFNumberSInt32Type, &n) && n > 0) {
                        result = (size_t) n;
                    }
                }
                CFRelease(value);
            }
            IOObjectRelease(obj);
            break;
        }
        IOObjectRelease(obj);
    }

    IOObjectRelease(it);
    return result;
}

enum gptoss_status gptoss_metal_device_create_system_default(
    struct gptoss_metal_device* device_out)
{
    id<MTLDevice> device_obj = MTLCreateSystemDefaultDevice();
    if (device_obj == nil) {
        GPTOSS_LOG_ERROR("failed to create Metal device");
        return gptoss_status_unsupported_system;
    }

    device_out->object = (void*) device_obj;
    device_out->num_cores = gptoss_metal_device_get_core_count(device_obj);
    device_out->max_buffer_size = (size_t) [device_obj maxBufferLength];
    device_out->max_threadgroup_memory = (size_t) [device_obj maxThreadgroupMemoryLength];
    const MTLSize max_threadgroup_threads = [device_obj maxThreadsPerThreadgroup];
    device_out->max_threadgroup_threads_x = (size_t) max_threadgroup_threads.width;
    device_out->max_threadgroup_threads_y = (size_t) max_threadgroup_threads.height;
    device_out->max_threadgroup_threads_z = (size_t) max_threadgroup_threads.depth;
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_device_release(
    struct gptoss_metal_device* device)
{
    if (device->object != NULL) {
        id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
        [device_obj release];
    }
    memset(device, 0, sizeof(struct gptoss_metal_device));
    return gptoss_status_success;
}

extern const struct mach_header_64 __dso_handle;

enum gptoss_status gptoss_metal_library_create_default(
    const struct gptoss_metal_device* device,
    struct gptoss_metal_library* library_out)
{
    enum gptoss_status status = gptoss_status_success;
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLLibrary> library_obj = nil;
    NSAutoreleasePool* autorelease_pool = nil;
    dispatch_data_t library_blob = NULL;

    unsigned long library_size = 0;
    uint8_t* library_data = getsectiondata(&__dso_handle, "__METAL", "__shaders", &library_size);
    if (library_data != NULL) {
        library_blob = dispatch_data_create(library_data, library_size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

        autorelease_pool = [[NSAutoreleasePool alloc] init];
        NSError* error_obj = nil;
        library_obj = [device_obj newLibraryWithData:library_blob error:&error_obj];
        if (library_obj == nil) {
            GPTOSS_LOG_ERROR("failed to create Metal library: %s", [[error_obj localizedDescription] UTF8String]);
            status = gptoss_status_unsupported_system;
            goto cleanup;
        }
    } else {
        // Fall-back to loading from the bundle
        library_obj = [device_obj newDefaultLibrary];
        if (library_obj == nil) {
            GPTOSS_LOG_ERROR("failed to create Metal default library");
            status = gptoss_status_unsupported_system;
            goto cleanup;
        }
    }

    *library_out = (struct gptoss_metal_library) {
        .object = (void*) library_obj,
    };

cleanup:
    if (library_blob != NULL) {
        dispatch_release(library_blob);
    }
    if (autorelease_pool != nil) {
        [autorelease_pool drain];
    }
    return status;
}

enum gptoss_status gptoss_metal_library_release(
    struct gptoss_metal_library* library)
{
    if (library->object != NULL) {
        id<MTLLibrary> library_obj = (id<MTLLibrary>) library->object;
        [library_obj release];
    }
    memset(library, 0, sizeof(struct gptoss_metal_library));
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_function_create(
    const struct gptoss_metal_library* library,
    const char* name,
    struct gptoss_metal_function* function_out)
{
    __block NSString* error_string_obj = nil;
    id<MTLFunction> function_obj = nil;
    MTLComputePipelineDescriptor* pipeline_descriptor_obj = nil;
    __block id<MTLComputePipelineState> pipeline_state_obj = nil;
    dispatch_semaphore_t pipeline_build_semaphore = NULL;
    enum gptoss_status status = gptoss_status_success;

    NSAutoreleasePool* autorelease_pool = [[NSAutoreleasePool alloc] init];
    id<MTLLibrary> library_obj = (id<MTLLibrary>) library->object;
    NSString* name_obj = [NSString stringWithUTF8String:name];
    function_obj = [library_obj newFunctionWithName:name_obj];
    if (function_obj == nil) {
        GPTOSS_LOG_ERROR("failed to create Metal function %s", name);
        status = gptoss_status_unsupported_system;
        goto cleanup;
    }
    id<MTLDevice> device_obj = [library_obj device];
    pipeline_descriptor_obj = [[MTLComputePipelineDescriptor alloc] init];
    [pipeline_descriptor_obj setComputeFunction:function_obj];
    [pipeline_descriptor_obj setThreadGroupSizeIsMultipleOfThreadExecutionWidth:YES];

    pipeline_build_semaphore = dispatch_semaphore_create(/*value=*/0);
    [device_obj newComputePipelineStateWithDescriptor:pipeline_descriptor_obj
                                              options:MTLPipelineOptionNone
                                    completionHandler:^(id<MTLComputePipelineState> _Nullable new_state,
                                                        MTLComputePipelineReflection* _Nullable reflection,
                                                        NSError* _Nullable error_obj) {
        if (new_state != nil) {
            pipeline_state_obj = [new_state retain];
        }
        if (error_obj != nil) {
            error_string_obj = [[error_obj localizedDescription] copy];
        }
        dispatch_semaphore_signal(pipeline_build_semaphore);
    }];
    dispatch_semaphore_wait(pipeline_build_semaphore, DISPATCH_TIME_FOREVER);

    if (pipeline_state_obj == nil) {
        const char* error_string = "unknown error";
        if (error_string_obj != nil) {
            error_string = [error_string_obj UTF8String];
        }
        GPTOSS_LOG_ERROR("failed to create Metal compute pipeline state for function %s: %s",
            name, error_string);
        status = gptoss_status_unsupported_system;
        goto cleanup;
    }

    // Commit
    function_out->function_object = function_obj;
    function_out->pipeline_state_object = pipeline_state_obj;
    function_out->max_threadgroup_threads = (size_t) [pipeline_state_obj maxTotalThreadsPerThreadgroup];
    function_out->simdgroup_threads = (size_t) [pipeline_state_obj threadExecutionWidth];
    function_out->static_threadgroup_memory = (size_t) [pipeline_state_obj staticThreadgroupMemoryLength];

    function_obj = nil;
    pipeline_state_obj = nil;

cleanup:
    if (function_obj != nil) {
        [function_obj release];
    }
    if (pipeline_descriptor_obj != nil) {
        [pipeline_descriptor_obj release];
    }
    if (error_string_obj != nil) {
        [error_string_obj release];
    }
    if (pipeline_build_semaphore != NULL) {
        dispatch_release(pipeline_build_semaphore);
    }
    if (autorelease_pool != nil) {
        [autorelease_pool drain];
    }
    return status;
}

enum gptoss_status gptoss_metal_function_release(
    struct gptoss_metal_function* function)
{
    if (function->pipeline_state_object != NULL) {
        id<MTLComputePipelineState> pipeline_state_obj = (id<MTLComputePipelineState>) function->pipeline_state_object;
        [pipeline_state_obj release];
    }
    if (function->function_object != NULL) {
        id<MTLFunction> function_obj = (id<MTLFunction>) function->function_object;
        [function_obj release];
    }
    memset(function, 0, sizeof(struct gptoss_metal_function));
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_buffer_create(
    const struct gptoss_metal_device* device,
    size_t size,
    const void* data,
    struct gptoss_metal_buffer* buffer_out)
{
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLBuffer> buffer_obj = nil;
    if (data != NULL) {
        buffer_obj = [device_obj newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    } else {
        buffer_obj = [device_obj newBufferWithLength:size options:MTLResourceStorageModeShared];
    }
    if (buffer_obj == nil) {
        GPTOSS_LOG_ERROR("failed to create Metal buffer of size %zu", size);
        return gptoss_status_unsupported_system;
    }
    buffer_out->object = (void*) buffer_obj;
    buffer_out->size = size;
    buffer_out->ptr = [buffer_obj contents];
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_buffer_wrap(
    const struct gptoss_metal_device* device,
    size_t size,
    const void* data,
    struct gptoss_metal_buffer* buffer_out)
{
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLBuffer> buffer_obj = [device_obj newBufferWithBytesNoCopy:(void*) data length:size options:MTLResourceStorageModeShared deallocator:nil];
    if (buffer_obj == nil) {
        GPTOSS_LOG_ERROR("failed to wrap Metal buffer of size %zu", size);
        return gptoss_status_unsupported_system;
    }
    buffer_out->object = (void*) buffer_obj;
    buffer_out->size = size;
    buffer_out->ptr = (void*) data;
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_buffer_release(
    struct gptoss_metal_buffer* buffer)
{
    if (buffer->object != NULL) {
        id<MTLBuffer> buffer_obj = (id<MTLBuffer>) buffer->object;
        [buffer_obj release];
    }
    memset(buffer, 0, sizeof(struct gptoss_metal_buffer));
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_queue_create(
    const struct gptoss_metal_device* device,
    struct gptoss_metal_command_queue* command_queue_out)
{
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLCommandQueue> command_queue_obj = [device_obj newCommandQueue];
    if (command_queue_obj == nil) {
        GPTOSS_LOG_ERROR("failed to create Metal command queue");
        return gptoss_status_unsupported_system;
    }
    command_queue_out->object = (void*) command_queue_obj;
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_queue_release(
    struct gptoss_metal_command_queue* command_queue)
{
    if (command_queue->object != NULL) {
        id<MTLCommandQueue> command_queue_obj = (id<MTLCommandQueue>) command_queue->object;
        [command_queue_obj release];
    }
    memset(command_queue, 0, sizeof(struct gptoss_metal_command_queue));
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_create(
    const struct gptoss_metal_command_queue* command_queue,
    struct gptoss_metal_command_buffer* command_buffer_out)
{
    id<MTLCommandQueue> command_queue_obj = (id<MTLCommandQueue>) command_queue->object;
    id<MTLCommandBuffer> command_buffer_obj = [command_queue_obj commandBuffer];
    if (command_buffer_obj == nil) {
        GPTOSS_LOG_ERROR("failed to create Metal command buffer");
        return gptoss_status_unsupported_system;
    }
    [command_buffer_obj retain];
    command_buffer_out->object = (void*) command_buffer_obj;
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_encode_fill_buffer(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_buffer* buffer,
    size_t offset,
    size_t size,
    uint8_t fill_value)
{
    if (command_buffer->object == NULL) {
        return gptoss_status_invalid_state;
    }
    if (buffer->object == NULL) {
        return gptoss_status_invalid_argument;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    id<MTLBuffer> buffer_obj = (id<MTLBuffer>) buffer->object;

    id<MTLBlitCommandEncoder> command_encoder_obj = [command_buffer_obj blitCommandEncoder];

    const NSRange range = NSMakeRange((NSUInteger) offset, (NSUInteger) size);
    [command_encoder_obj fillBuffer:buffer_obj range:range value:fill_value];
    [command_encoder_obj endEncoding];

    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_encode_copy_buffer(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    size_t size)
{
    if (command_buffer->object == NULL) {
        return gptoss_status_invalid_state;
    }
    if (input_buffer->object == NULL) {
        return gptoss_status_invalid_argument;
    }
    if (output_buffer->object == NULL) {
        return gptoss_status_invalid_argument;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    id<MTLBuffer> input_buffer_obj = (id<MTLBuffer>) input_buffer->object;
    id<MTLBuffer> output_buffer_obj = (id<MTLBuffer>) output_buffer->object;

    id<MTLBlitCommandEncoder> command_encoder_obj = [command_buffer_obj blitCommandEncoder];

    [command_encoder_obj copyFromBuffer:input_buffer_obj sourceOffset:(NSUInteger) input_offset
                         toBuffer:output_buffer_obj destinationOffset:(NSUInteger) output_offset
                         size:(NSUInteger) size];
    [command_encoder_obj endEncoding];

    return gptoss_status_success;
}

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
    size_t threadgroup_buffer_size)
{
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    id<MTLComputePipelineState> pipeline_state_obj = (id<MTLComputePipelineState>) function->pipeline_state_object;

    id<MTLComputeCommandEncoder> command_encoder_obj = [command_buffer_obj computeCommandEncoder];

    // Set kernel arguments
    [command_encoder_obj setComputePipelineState:pipeline_state_obj];
    [command_encoder_obj setBytes:params length:params_size atIndex:0];
    for (size_t i = 0; i < num_device_buffers; ++i) {
        id<MTLBuffer> buffer_obj = (id<MTLBuffer>) device_buffers[i]->object;
        const NSUInteger offset = device_buffer_offsets == NULL ? 0 : (NSUInteger) device_buffer_offsets[i];
        [command_encoder_obj setBuffer:buffer_obj offset:offset atIndex:i + 1];
    }
    if (threadgroup_buffer_size != 0) {
        [command_encoder_obj setThreadgroupMemoryLength:threadgroup_buffer_size atIndex:0];
    }

    // Dispatch kernel
    const MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_x, threadgroup_size_y, threadgroup_size_z);
    const MTLSize num_threadgroups = MTLSizeMake(num_threadgroups_x, num_threadgroups_y, num_threadgroups_z);
    [command_encoder_obj dispatchThreadgroups:num_threadgroups threadsPerThreadgroup:threadgroup_size];
    [command_encoder_obj endEncoding];

    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_commit(
    const struct gptoss_metal_command_buffer* command_buffer)
{
    if (command_buffer->object == NULL) {
        return gptoss_status_invalid_state;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    [command_buffer_obj commit];
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_wait_completion(
    const struct gptoss_metal_command_buffer* command_buffer,
    double* elapsed_seconds)
{
    if (command_buffer->object == NULL) {
        return gptoss_status_invalid_state;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    [command_buffer_obj waitUntilCompleted];
    if (elapsed_seconds != NULL) {
        const CFTimeInterval start_time = [command_buffer_obj GPUStartTime];
        const CFTimeInterval end_time = [command_buffer_obj GPUEndTime];
        *elapsed_seconds = (double) end_time - (double) start_time;
    }
    return gptoss_status_success;
}

enum gptoss_status gptoss_metal_command_buffer_release(
    struct gptoss_metal_command_buffer* command_buffer)
{
    if (command_buffer->object != NULL) {
        id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
        [command_buffer_obj release];
    }
    memset(command_buffer, 0, sizeof(struct gptoss_metal_command_buffer));
    return gptoss_status_success;
}
