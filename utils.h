#pragma once

extern void* safe_malloc(size_t sz, const char *tag);

enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_ERROR
};

extern void log_set_level(int level);

extern void log_info(const char *fmt, ...);

extern void log_error(const char *fmt, ...);

extern void log_debug(const char *fmt, ...);