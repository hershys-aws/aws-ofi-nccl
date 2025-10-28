/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef THREAD_UTILS_H_
#define THREAD_UTILS_H_

#include <pthread.h>
#include <vector>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "thread_context.h"

/**
 * @file thread_utils.h
 * @brief Threading utilities for multi-threaded NCCL OFI plugin tests
 * 
 * This header provides utilities for creating and managing worker threads,
 * synchronizing thread execution with barriers, and collecting thread results.
 */

// Validation macros
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

// Forward declaration for poll_with_timeout
template<typename Condition>
ncclResult_t poll_with_timeout(Condition condition, int timeout_ms, const char* operation_name);

/**
 * Initialize thread barriers for multi-threaded tests
 * 
 * Creates three barriers for coordinating thread execution phases:
 * setup, test, and cleanup. All barriers are initialized for the
 * specified number of threads.
 * 
 * @param num_threads Number of threads that will use the barriers
 * @param setup Pointer to setup barrier to initialize
 * @param test Pointer to test barrier to initialize
 * @param cleanup Pointer to cleanup barrier to initialize
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t init_thread_barriers(size_t num_threads,
                                                pthread_barrier_t* setup, 
                                                pthread_barrier_t* test, 
                                                pthread_barrier_t* cleanup)
{
	VALIDATE_NOT_NULL(setup, "setup");
	VALIDATE_NOT_NULL(test, "test");
	VALIDATE_NOT_NULL(cleanup, "cleanup");
	VALIDATE_POSITIVE(num_threads, "num_threads");
	
	NCCL_OFI_TRACE(NCCL_NET, "Initializing thread barriers for %zu threads", num_threads);
	
	// Initialize setup barrier
	int ret = pthread_barrier_init(setup, nullptr, num_threads);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize setup barrier: %d", ret);
		return ncclSystemError;
	}
	
	// Initialize test barrier
	ret = pthread_barrier_init(test, nullptr, num_threads);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize test barrier: %d", ret);
		pthread_barrier_destroy(setup);
		return ncclSystemError;
	}
	
	// Initialize cleanup barrier
	ret = pthread_barrier_init(cleanup, nullptr, num_threads);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize cleanup barrier: %d", ret);
		pthread_barrier_destroy(setup);
		pthread_barrier_destroy(test);
		return ncclSystemError;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Thread barriers initialized successfully");
	return ncclSuccess;
}

/**
 * Destroy thread barriers
 * 
 * Destroys all three barriers created by init_thread_barriers().
 * 
 * @param setup Pointer to setup barrier to destroy
 * @param test Pointer to test barrier to destroy
 * @param cleanup Pointer to cleanup barrier to destroy
 */
static inline void destroy_thread_barriers(pthread_barrier_t* setup, 
                                           pthread_barrier_t* test, 
                                           pthread_barrier_t* cleanup)
{
	// Validate inputs - but don't fail, just skip NULL pointers
	if (setup != nullptr) {
		int ret = pthread_barrier_destroy(setup);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to destroy setup barrier: %d", ret);
		}
	}
	
	if (test != nullptr) {
		int ret = pthread_barrier_destroy(test);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to destroy test barrier: %d", ret);
		}
	}
	
	if (cleanup != nullptr) {
		int ret = pthread_barrier_destroy(cleanup);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to destroy cleanup barrier: %d", ret);
		}
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Thread barriers destroyed");
}

/**
 * Create worker threads for multi-threaded tests
 * 
 * Creates the specified number of threads, passing a thread_context
 * to each thread function. Thread handles are stored in the threads array.
 * 
 * @param num_threads Number of threads to create
 * @param thread_func Thread function to execute
 * @param contexts Array of thread_context structures (one per thread)
 * @param threads Array to store thread handles
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t create_worker_threads(size_t num_threads, 
                                                 void* (*thread_func)(void*), 
                                                 thread_context* contexts,
                                                 pthread_t* threads)
{
	VALIDATE_NOT_NULL(thread_func, "thread_func");
	VALIDATE_NOT_NULL(contexts, "contexts");
	VALIDATE_NOT_NULL(threads, "threads");
	VALIDATE_POSITIVE(num_threads, "num_threads");
	
	NCCL_OFI_TRACE(NCCL_NET, "Creating %zu worker threads", num_threads);
	
	// Create threads
	for (size_t i = 0; i < num_threads; i++) {
		int ret = pthread_create(&threads[i], nullptr, thread_func, &contexts[i]);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to create thread %zu: %d", i, ret);
			
			// Clean up already created threads
			for (size_t j = 0; j < i; j++) {
				pthread_cancel(threads[j]);
				pthread_join(threads[j], nullptr);
			}
			
			return ncclSystemError;
		}
		
		NCCL_OFI_TRACE(NCCL_NET, "Created thread %zu", i);
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "All %zu worker threads created successfully", num_threads);
	return ncclSuccess;
}

/**
 * Wait for threads to complete with timeout
 * 
 * Joins all threads, waiting for them to complete. If timeout is exceeded,
 * returns an error. A timeout of 0 means wait indefinitely.
 * 
 * @param num_threads Number of threads to wait for
 * @param threads Array of thread handles
 * @param timeout_seconds Timeout in seconds (0 for no timeout)
 * @return ncclSuccess if all threads complete, ncclInternalError on timeout
 */
static inline ncclResult_t wait_for_threads(size_t num_threads, 
                                            pthread_t* threads, 
                                            int timeout_seconds)
{
	VALIDATE_NOT_NULL(threads, "threads");
	
	if (num_threads == 0) {
		return ncclSuccess;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Waiting for %zu threads (timeout=%d seconds)",
	               num_threads, timeout_seconds);
	
	std::vector<bool> joined(num_threads, false);
	size_t num_joined = 0;
	
	// Use polling helper with timeout (convert seconds to milliseconds)
	auto check_threads = [&]() {
		for (size_t i = 0; i < num_threads; i++) {
			if (joined[i]) continue;
			
#ifdef __GLIBC__
			int ret = pthread_tryjoin_np(threads[i], nullptr);
			if (ret == 0) {
				joined[i] = true;
				num_joined++;
				NCCL_OFI_TRACE(NCCL_NET, "Thread %zu joined (%zu/%zu)",
				               i, num_joined, num_threads);
			} else if (ret != EBUSY) {
				NCCL_OFI_WARN("Failed to join thread %zu: %d", i, ret);
				return false;
			}
#else
			int ret = pthread_join(threads[i], nullptr);
			if (ret == 0) {
				joined[i] = true;
				num_joined++;
				NCCL_OFI_TRACE(NCCL_NET, "Thread %zu joined (%zu/%zu)",
				               i, num_joined, num_threads);
			} else {
				NCCL_OFI_WARN("Failed to join thread %zu: %d", i, ret);
				return false;
			}
#endif
		}
		return num_joined >= num_threads;
	};
	
	ncclResult_t res = poll_with_timeout(check_threads, timeout_seconds * 1000, "threads");
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Timeout: %zu/%zu threads joined", num_joined, num_threads);
		return res;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "All %zu threads joined successfully", num_threads);
	return ncclSuccess;
}

/**
 * Collect results from multiple threads
 * 
 * Iterates through thread_context array and collects the result from each
 * thread. Returns the first error encountered, or ncclSuccess if all
 * threads succeeded.
 * 
 * @param num_threads Number of threads to collect results from
 * @param contexts Array of thread_context structures
 * @return Aggregated error code (first error or ncclSuccess)
 */
static inline ncclResult_t collect_thread_results(size_t num_threads, 
                                                  thread_context* contexts)
{
	VALIDATE_NOT_NULL(contexts, "contexts");
	
	if (num_threads == 0) {
		return ncclSuccess;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Collecting results from %zu threads", num_threads);
	
	// Iterate through all thread contexts and collect results
	ncclResult_t aggregated_result = ncclSuccess;
	size_t num_errors = 0;
	
	for (size_t i = 0; i < num_threads; i++) {
		if (contexts[i].get_result() != ncclSuccess) {
			NCCL_OFI_WARN("Thread %zu (thread_id=%d) failed with error: %d",
			              i, contexts[i].get_thread_id(), contexts[i].get_result());
			
			// Store the first error encountered
			if (aggregated_result == ncclSuccess) {
				aggregated_result = contexts[i].get_result();
			}
			
			num_errors++;
		}
	}
	
	if (num_errors > 0) {
		NCCL_OFI_WARN("Thread result collection: %zu/%zu threads failed",
		              num_errors, num_threads);
	} else {
		NCCL_OFI_TRACE(NCCL_NET, "All %zu threads completed successfully", num_threads);
	}
	
	return aggregated_result;
}

#if HAVE_CUDA
/**
 * Initialize CUDA for a worker thread
 * 
 * Sets the CUDA device for the current thread and initializes the CUDA
 * context. This should be called at the beginning of each worker thread
 * that needs to use CUDA.
 * 
 * @param cuda_device CUDA device index to use
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t init_cuda_for_thread(int cuda_device)
{
	// Validate input
	if (cuda_device < 0) {
		NCCL_OFI_WARN("Invalid CUDA device %d passed to init_cuda_for_thread", cuda_device);
		return ncclInvalidArgument;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Initializing CUDA for thread with device %d", cuda_device);
	
	// Set CUDA device for this thread
	cudaError_t cuda_ret = cudaSetDevice(cuda_device);
	if (cuda_ret != cudaSuccess) {
		NCCL_OFI_WARN("Failed to set CUDA device %d for thread: %s",
		              cuda_device, cudaGetErrorString(cuda_ret));
		return ncclUnhandledCudaError;
	}
	
	// Perform a dummy operation to initialize CUDA context for this thread
	cuda_ret = cudaFree(0);
	if (cuda_ret != cudaSuccess) {
		NCCL_OFI_WARN("Failed to initialize CUDA context for thread: %s",
		              cudaGetErrorString(cuda_ret));
		return ncclUnhandledCudaError;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "CUDA initialized successfully for thread with device %d", cuda_device);
	return ncclSuccess;
}
#endif

#endif // THREAD_UTILS_H_
