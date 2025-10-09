#include <assert.h>  // assert
#include <stdarg.h>  // va_list, va_copy, va_end
#include <stdio.h>  // vsnprintf
#include <stdlib.h>  // malloc, free

#include <unistd.h>  // STDERR_FILENO



#define GPTOSS_ON_STACK_FORMAT_BUFFER_SIZE 16384

void gptoss_format_log(const char* format, va_list args) {
    char stack_buffer[GPTOSS_ON_STACK_FORMAT_BUFFER_SIZE];
    char* heap_buffer = NULL;

    va_list args_copy;
    va_copy(args_copy, args);

    const int vsnprintf_result = vsnprintf(stack_buffer, GPTOSS_ON_STACK_FORMAT_BUFFER_SIZE, format, args);
    assert(vsnprintf_result >= 0);

    // At least a partially formatted buffer is ready.
    char* message_buffer = &stack_buffer[0];
    size_t message_size = (size_t) vsnprintf_result;
    if (message_size > GPTOSS_ON_STACK_FORMAT_BUFFER_SIZE) {
        heap_buffer = malloc(message_size);
        if (heap_buffer == NULL) {
            // Fall back to the truncated message in the on-stack buffer.
            message_size = GPTOSS_ON_STACK_FORMAT_BUFFER_SIZE;
        } else {
            // Use the full message in the in-heap buffer.
            vsnprintf(heap_buffer, message_size, format, args_copy);
            message_buffer = heap_buffer;
        }
    }

    ssize_t bytes_written;
    do {
        bytes_written = write(STDERR_FILENO, message_buffer, message_size);
        if (bytes_written > 0) {
            assert((size_t) bytes_written <= message_size);
            message_buffer += bytes_written;
            message_size -= bytes_written;
        }
    } while (bytes_written >= 0 && message_size != 0);

cleanup:
    free(heap_buffer);
    va_end(args_copy);
}
