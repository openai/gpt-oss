#pragma once

#include <stdarg.h>


void gptoss_format_log(const char* format, va_list args);

__attribute__((__format__(__printf__, 1, 2)))
inline static void gptoss_log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    gptoss_format_log(format, args);
    va_end(args);
}

#define GPTOSS_LOG_ERROR(message, ...) \
    gptoss_log("Error: " message "\n", ##__VA_ARGS__)

#define GPTOSS_LOG_WARNING(message, ...) \
    gptoss_log("Warning: " message "\n", ##__VA_ARGS__)
