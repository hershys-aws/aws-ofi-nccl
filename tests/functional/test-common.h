/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include "config.h"

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <set>
#include <vector>

#include <cuda_runtime.h>
#include <nccl/net.h>
#include <mpi.h>

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_param.h"

#define STR2(v)		#v
#define STR(v)		STR2(v)

#define NUM_REQUESTS	(NCCL_NET_MAX_REQUESTS)
#define SEND_SIZE	(5000)
#define RECV_SIZE	(5200)

#define OFINCCLCHECK(call)                                                \
	do {                                                              \
		ncclResult_t macro_res = call;                            \
		if (macro_res != ncclSuccess) {                           \
			NCCL_OFI_WARN("OFI NCCL failure: %d", macro_res); \
			return macro_res;                                 \
		}                                                         \
	} while (false);

#define OFINCCLCHECKGOTO(call, res, label) do {			\
	res = call;						\
	if (res != ncclSuccess) {				\
		NCCL_OFI_WARN("OFI NCCL failure: %d", res);	\
		goto label;					\
	}							\
} while (false);

#define CUDACHECK(call) do {						\
        cudaError_t e = call;						\
        if (e != cudaSuccess) {						\
	    const char *error_str = cudaGetErrorString(e);		\
	    NCCL_OFI_WARN("Cuda failure '%s'", error_str);		\
	    return ncclUnhandledCudaError;				\
        }								\
} while(false);

// Can be changed when porting new versions to the plugin
#define NCCL_PLUGIN_SYMBOL ncclNetPlugin_v9

typedef ncclNet_v9_t test_nccl_net_t;
typedef ncclNetProperties_v9_t test_nccl_properties_t;
typedef ncclNetDeviceHandle_v9_t test_nccl_net_device_handle_t;

// Include utility classes from tests/utils
#include "utils/thread_context.h"
#include "utils/thread_utils.h"
#include "utils/resource_tracker.h"
#include "utils/test_fixture.h"
#include "utils/env_parser.h"
#include "utils/buffer_utils.h"
#include "utils/data_transfer.h"
#include "utils/device_utils.h"
#include "utils/test_params.h"


static void logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
		   int line, const char *fmt, ...)
{
	va_list vargs;

	switch (level) {
		case NCCL_LOG_WARN:
			printf("WARN: Function: %s Line: %d: ", filefunc, line);
			break;
		case NCCL_LOG_INFO:
			printf("INFO: Function: %s Line: %d: ", filefunc, line);
			break;
		case NCCL_LOG_TRACE:
#if OFI_NCCL_TRACE
			printf("TRACE: Function: %s Line: %d: ", filefunc, line);
			break;
#else
			return;
#endif
		case NCCL_LOG_NONE:
		case NCCL_LOG_VERSION:
		case NCCL_LOG_ABORT:
		default:
			break;
	};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat=2"
	va_start(vargs, fmt);
	vprintf(fmt, vargs);
	printf("\n");
	va_end(vargs);
#pragma GCC diagnostic pop
}

#endif // End TEST_COMMON_H_
