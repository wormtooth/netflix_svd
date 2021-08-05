#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "utils.h"

const char *LOG_TAGS[] = {
    "DEBUG",
    "INFO",
    "ERROR"
};

static int log_level = LOG_LEVEL_DEBUG;

static void log_fmt(int level, const char *fmt, va_list args) {
    if (level < log_level) return;
    time_t now;
    time(&now);
    char *date = ctime(&now);
    date[strlen(date) - 1] = '\0';
    printf("%s - %s: ", date, LOG_TAGS[level]);
    vprintf(fmt, args);
    printf("\n");
}

extern void* safe_malloc(size_t sz, const char *tag) {
    void *ptr = malloc(sz);
    if (!ptr) {
        log_error("Fail to allocate memory for `%s`", tag);
        exit(1);
    }
    return ptr;
}

extern void log_set_level(int level) {
    log_level = level;
}

extern void log_info(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_fmt(LOG_LEVEL_INFO, fmt, args);
    va_end(args);
}

extern void log_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_fmt(LOG_LEVEL_ERROR, fmt, args);
    va_end(args);
}

extern void log_debug(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_fmt(LOG_LEVEL_DEBUG, fmt, args);
    va_end(args);
}