/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#ifdef ENABLE_INSTRUMENT_FUNCTIONS

#include <cstdio>
#include <cstdarg>
#include "nccl_ofi_log.h"
#include "nccl_ofi_api.h"

// Simple logger function for testing
__attribute__((no_instrument_function))
static void test_logger(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, 
                       int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("LOG: ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

// Test functions that will be instrumented
static void test_function_a() {
    printf("Inside test_function_a\n");
}

static void test_function_b() {
    printf("Inside test_function_b, calling test_function_a\n");
    test_function_a();
}

static void test_function_c() {
    printf("Inside test_function_c, calling both functions\n");
    test_function_a();
    test_function_b();
}

int main() {
    printf("=== Testing function instrumentation ===\n");
    
    // Initialize the NCCL OFI logger so NCCL_OFI_INFO works
    ofi_log_function = test_logger;
    printf("Logger initialized\n");
    
    printf("You should see 'entering' and 'exiting' messages with function addresses\n\n");
    
    printf("Calling test_function_a():\n");
    test_function_a();
    
    printf("\nCalling test_function_b():\n");
    test_function_b();
    
    printf("\nCalling test_function_c():\n");
    test_function_c();
    
    printf("\n=== Function instrumentation test completed ===\n");
    return 0;
}

#else

int main() {
    printf("Function instrumentation is disabled, skipping test\n");
    return 0;
}

#endif /* ENABLE_INSTRUMENT_FUNCTIONS */
