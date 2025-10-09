#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "internal/macros.h"


struct GPTOSS_DENSELY_PACKED_STRUCTURE gptoss_uuid {
    uint8_t bytes[16];
};
static_assert(sizeof(struct gptoss_uuid) == 16, "UUID size is not 16 bytes");


#define UUID_FORMAT "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X"
#define UUID_ARGS(uuid) (uuid).bytes[0], (uuid).bytes[1], (uuid).bytes[2], (uuid).bytes[3], \
    (uuid).bytes[4], (uuid).bytes[5], (uuid).bytes[6], (uuid).bytes[7], (uuid).bytes[8], (uuid).bytes[9], \
    (uuid).bytes[10], (uuid).bytes[11], (uuid).bytes[12], (uuid).bytes[13], (uuid).bytes[14], (uuid).bytes[15]

static inline bool gptoss_is_gptoss_model_uuid(const struct gptoss_uuid* uuid) {
    return memcmp(
        &(struct gptoss_uuid) {0xDF, 0x52, 0xDC, 0x86, 0x17, 0x89, 0x4E, 0xD0, 0xA2, 0x95, 0x66, 0xF1, 0x05, 0x08, 0x14, 0x5B},
        uuid,
        sizeof(struct gptoss_uuid)) == 0;
}

static inline bool gptoss_is_applegpu_layout_uuid(const struct gptoss_uuid* uuid) {
    return memcmp(
        &(struct gptoss_uuid) {0x22, 0x91, 0x77, 0xA8, 0x57, 0x75, 0x42, 0x68, 0xBF, 0xD8, 0xD5, 0x88, 0xB3, 0x51, 0xC5, 0x6D},
        uuid,
        sizeof(struct gptoss_uuid)) == 0;
}

static inline bool gptoss_is_tiktoken_tokenizer_uuid(const struct gptoss_uuid* uuid) {
    return memcmp(
        &(struct gptoss_uuid) {0x74, 0x01, 0xAD, 0xED, 0x2A, 0x95, 0x40, 0xCB, 0xB7, 0x82, 0x9C, 0xCE, 0xBA, 0xAF, 0xE7, 0x2B},
        uuid,
        sizeof(struct gptoss_uuid)) == 0;
}

static inline enum gptoss_special_token gptoss_special_token_decode_uuid(const struct gptoss_uuid* uuid) {
    if (memcmp(
        &(struct gptoss_uuid) {0x55, 0xA7, 0x7C, 0x2F, 0x8A, 0x01, 0x4C, 0x54, 0x8A, 0xC2, 0x31, 0x3B, 0xFC, 0x7E, 0x20, 0x8D},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_start;
    } else if (memcmp(
        &(struct gptoss_uuid) {0x16, 0xE4, 0x04, 0x31, 0xF4, 0x7F, 0x4B, 0x22, 0xB5, 0x9B, 0x8B, 0x27, 0x8F, 0xC3, 0x0A, 0x54},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_message;
    } else if (memcmp(
        &(struct gptoss_uuid) {0xFC, 0xAC, 0x2F, 0x6D, 0x47, 0x05, 0x4F, 0x6B, 0xB2, 0x28, 0x64, 0x2A, 0xCC, 0xAC, 0x72, 0x38},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_end;
    } else if (memcmp(
        &(struct gptoss_uuid) {0xF7, 0x99, 0xFF, 0x69, 0x19, 0x92, 0x43, 0xC4, 0xA3, 0xD8, 0xD8, 0x31, 0xF4, 0x75, 0xDC, 0x75},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_return;
    } else if (memcmp(
        &(struct gptoss_uuid) {0xE1, 0x5B, 0xA7, 0x02, 0x28, 0xC4, 0x42, 0x92, 0xAB, 0x8F, 0xFF, 0xA4, 0x34, 0x70, 0x91, 0x28},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_refusal;
    } else if (memcmp(
        &(struct gptoss_uuid) {0xC0, 0xBB, 0x14, 0xC7, 0x60, 0x22, 0x49, 0xDA, 0xAD, 0x08, 0x79, 0x2D, 0x67, 0xE8, 0xB4, 0x70},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_constrain;
    } else if (memcmp(
        &(struct gptoss_uuid) {0xFD, 0x3D, 0xDA, 0x11, 0xC8, 0xAB, 0x40, 0x33, 0x87, 0x6E, 0xD9, 0x3D, 0xEB, 0x17, 0x2C, 0x93},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_channel;
    } else if (memcmp(
        &(struct gptoss_uuid) {0x12, 0x20, 0xF7, 0x96, 0xE3, 0x88, 0x4D, 0xE5, 0xB4, 0x87, 0xFE, 0x2E, 0xB5, 0xFE, 0x03, 0xC0},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_call;
    } else if (memcmp(
        &(struct gptoss_uuid) {0x07, 0xD7, 0xDA, 0x55, 0xB3, 0x46, 0x4C, 0xFF, 0x8B, 0x37, 0x7C, 0xEF, 0xAC, 0xF8, 0xA3, 0xE8},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_untrusted;
    } else if (memcmp(
        &(struct gptoss_uuid) {0xF2, 0x65, 0xBD, 0x9C, 0xC7, 0x17, 0x46, 0x9E, 0xA4, 0x47, 0x92, 0x06, 0x87, 0xD6, 0x5D, 0x90},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        return gptoss_special_token_end_untrusted;
    } else if (memcmp(
        &(struct gptoss_uuid) {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
        uuid,
        sizeof(struct gptoss_uuid)) == 0)
    {
        // Suppress warning
        return gptoss_special_token_invalid;
    } else {
        GPTOSS_LOG_WARNING("unsupported special token " UUID_FORMAT, UUID_ARGS(*uuid));
        return gptoss_special_token_invalid;
    }
}
