/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef BUFFER_UTILS_H_
#define BUFFER_UTILS_H_

#include <cstring>
#include <nccl/net.h>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"

/**
 * @file buffer_utils.h
 * @brief Buffer management utilities for NCCL OFI plugin tests
 * 
 * This header provides utilities for allocating, initializing, validating,
 * and managing memory buffers (both host and CUDA device memory) used in tests.
 */

// Forward declaration for get_extNet (defined in test-common.h)
typedef ncclNet_v9_t test_nccl_net_t;
extern test_nccl_net_t *get_extNet(void);

// Macros from test-common.h
#define OFINCCLCHECK(call)                                                \
	do {                                                              \
		ncclResult_t macro_res = call;                            \
		if (macro_res != ncclSuccess) {                           \
			NCCL_OFI_WARN("OFI NCCL failure: %d", macro_res); \
			return macro_res;                                 \
		}                                                         \
	} while (false);

#define CUDACHECK(call) do {						\
        cudaError_t e = call;						\
        if (e != cudaSuccess) {						\
	    const char *error_str = cudaGetErrorString(e);		\
	    NCCL_OFI_WARN("Cuda failure '%s'", error_str);		\
	    return ncclUnhandledCudaError;				\
        }								\
} while(false);

#define VALIDATE_NOT_NULL(ptr, name) \
	do { \
		if ((ptr) == nullptr) { \
			NCCL_OFI_WARN("Invalid NULL %s pointer", name); \
			return ncclInvalidArgument; \
		} \
	} while (false)

#define VALIDATE_POSITIVE(value, name) \
	do { \
		if ((value) <= 0) { \
			NCCL_OFI_WARN("Invalid %s value: %d", name, (int)(value)); \
			return ncclInvalidArgument; \
		} \
	} while (false)

#define VALIDATE_BUFFER_TYPE(type) \
	do { \
		if ((type) != NCCL_PTR_HOST && (type) != NCCL_PTR_CUDA) { \
			NCCL_OFI_WARN("Invalid buffer type: %d", type); \
			return ncclInvalidArgument; \
		} \
	} while (false)

/**
 * Allocate buffer (host or CUDA device memory)
 * 
 * @param buf Output pointer to allocated buffer
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating CUDA buffer");
		CUDACHECK(cudaMalloc(buf, size));
		break;
	case NCCL_PTR_HOST:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating host buffer");
		CUDACHECK(cudaHostAlloc(buf, size, cudaHostAllocMapped));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Initialize buffer with a pattern
 * 
 * @param buf Buffer to initialize
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaMemset(buf, '1', size));
		break;
	case NCCL_PTR_HOST:
		memset(buf, '1', size);
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Deallocate buffer
 * 
 * @param buf Buffer to deallocate
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t deallocate_buffer(void *buf, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaFree(buf));
		break;
	case NCCL_PTR_HOST:
		CUDACHECK(cudaFreeHost((void *)buf));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Validate received data against expected data
 * 
 * @param recv_buf Received buffer to validate
 * @param expected_buf Expected buffer to compare against
 * @param size Size of buffers in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess if data matches, error code otherwise
 */
static inline ncclResult_t validate_data(char *recv_buf, char *expected_buf, size_t size, int buffer_type)
{
	int ret = 0;
	char *recv_buf_host = NULL;

	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		OFINCCLCHECK(allocate_buff((void **)&recv_buf_host, size, NCCL_PTR_HOST));
		CUDACHECK(cudaMemcpy(recv_buf_host, recv_buf, size, cudaMemcpyDeviceToHost));

		ret = memcmp(recv_buf_host, expected_buf, size);
		if (ret != 0) {
			NCCL_OFI_WARN("Data validation check failed. RC: %d, Buffer Type: %d",
				      ret, buffer_type);
			return ncclSystemError;
		}
		break;
	case NCCL_PTR_HOST:
		ret = memcmp(recv_buf, expected_buf, size);
		if (ret != 0) {
			NCCL_OFI_WARN("Data validation check failed. RC: %d, Buffer Type: %d",
				      ret, buffer_type);
			return ncclSystemError;
		}
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Register memory with communicator
 * 
 * Registers a memory region with a communicator for RDMA operations.
 * Calls the plugin's regMr function to obtain a memory handle.
 * 
 * @param comm Communicator to register memory with
 * @param buffer Pointer to buffer to register
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @param mhandle Output pointer to memory handle
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t register_memory(void* comm, void* buffer, size_t size, int buffer_type, void** mhandle)
{
	// Validate inputs using macros
	VALIDATE_NOT_NULL(comm, "comm");
	VALIDATE_NOT_NULL(buffer, "buffer");
	VALIDATE_NOT_NULL(mhandle, "mhandle");
	VALIDATE_POSITIVE(size, "size");
	VALIDATE_BUFFER_TYPE(buffer_type);
	
	*mhandle = nullptr;
	
	NCCL_OFI_TRACE(NCCL_NET, "Registering memory: buffer=%p, size=%zu, type=%d",
	               buffer, size, buffer_type);
	
	// Get plugin interface
	test_nccl_net_t* ext_net = get_extNet();
	if (ext_net == nullptr) {
		NCCL_OFI_WARN("Failed to get plugin interface");
		return ncclInternalError;
	}
	
	// Call plugin's regMr function
	ncclResult_t res = ext_net->regMr(comm, buffer, size, buffer_type, mhandle);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to register memory: %d", res);
		return res;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Memory registered successfully: mhandle=%p", *mhandle);
	return ncclSuccess;
}

/**
 * Deregister memory from communicator
 * 
 * Deregisters a previously registered memory region. Calls the plugin's
 * deregMr function to release the memory handle.
 * 
 * @param comm Communicator to deregister memory from
 * @param mhandle Memory handle to deregister
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t deregister_memory(void* comm, void* mhandle)
{
	VALIDATE_NOT_NULL(comm, "comm");
	
	// Deregistering NULL handle is a no-op, not an error
	if (mhandle == nullptr) {
		NCCL_OFI_TRACE(NCCL_NET, "Skipping deregistration of NULL memory handle");
		return ncclSuccess;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Deregistering memory: mhandle=%p", mhandle);
	
	// Get plugin interface
	test_nccl_net_t* ext_net = get_extNet();
	if (ext_net == nullptr) {
		NCCL_OFI_WARN("Failed to get plugin interface");
		return ncclInternalError;
	}
	
	// Call plugin's deregMr function
	ncclResult_t res = ext_net->deregMr(comm, mhandle);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to deregister memory: %d", res);
		return res;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Memory deregistered successfully");
	return ncclSuccess;
}

#endif // BUFFER_UTILS_H_
