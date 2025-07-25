/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#ifdef ENABLE_INSTRUMENT_FUNCTIONS

#include <cstdio>
#include "nccl_ofi_log.h"

extern "C" {

void __cyg_profile_func_enter(void *this_fn, void *call_site)
    __attribute__((no_instrument_function));
void __cyg_profile_func_exit(void *this_fn, void *call_site)
    __attribute__((no_instrument_function));

// Function called on every function entry
void __cyg_profile_func_enter(void *this_fn, void *call_site) {
	if (!ofi_log_function) return;
    (void)call_site;  // Suppress unused parameter warning
    NCCL_OFI_INFO(NCCL_ALL, "entering %p", this_fn);
}

// Function called on every function exit
void __cyg_profile_func_exit(void *this_fn, void *call_site) {
	if (!ofi_log_function) return;
    (void)call_site;  // Suppress unused parameter warning
    NCCL_OFI_INFO(NCCL_ALL, "exiting %p", this_fn);
}

}

#endif /* ENABLE_INSTRUMENT_FUNCTIONS */
