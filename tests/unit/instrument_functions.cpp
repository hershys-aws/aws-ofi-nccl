/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#ifdef ENABLE_INSTRUMENT_FUNCTIONS

#include <cstdio>

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
