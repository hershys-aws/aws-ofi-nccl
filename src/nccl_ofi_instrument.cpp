/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#ifdef ENABLE_INSTRUMENT_FUNCTIONS

#include <cstdio>

extern "C" {

void __cyg_profile_func_enter(void *this_fn, void *call_site)
    __attribute__((no_instrument_function));
void __cyg_profile_func_exit(void *this_fn, void *call_site)
    __attribute__((no_instrument_function));

// Function called on every function entry
void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    (void)call_site;  // Suppress unused parameter warning
    printf("entering %p\n", this_fn);
}

// Function called on every function exit
void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    (void)call_site;  // Suppress unused parameter warning
    printf("exiting %p\n", this_fn);
}

}

#endif /* ENABLE_INSTRUMENT_FUNCTIONS */
