/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * @file test_params.cpp
 * @brief Test parameter parsing implementation
 */

#include "test_params.h"
#include "env_parser.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <nccl/net.h>

#include "nccl_ofi_log.h"

// Validation macros
#define VALIDATE_NOT_NULL(ptr, name) \
	do { \
		if ((ptr) == nullptr) { \
			NCCL_OFI_WARN("Invalid NULL %s pointer", name); \
			return ncclInvalidArgument; \
		} \
	} while (false)

ncclResult_t parse_test_arguments(int argc, 
                                 char* argv[], 
                                 test_parameters* params)
{
	VALIDATE_NOT_NULL(params, "params");
	
	// Set default values
	params->num_communicators = 1;
	params->num_threads = 1;
	params->buffer_type = NCCL_PTR_HOST;
	params->buffer_size = 4096;
	params->num_iterations = 1;
	params->verbose = false;
	
	NCCL_OFI_TRACE(NCCL_NET, "Parsing command-line arguments (argc=%d)", argc);
	
	// Parse command-line arguments
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			// Help flag - caller should handle this
			return ncclInvalidArgument;
		}
		else if (strcmp(argv[i], "--num-communicators") == 0) {
			if (i + 1 >= argc) {
				NCCL_OFI_WARN("Missing value for --num-communicators");
				return ncclInvalidArgument;
			}
			params->num_communicators = atoi(argv[++i]);
			if (params->num_communicators <= 0) {
				NCCL_OFI_WARN("Invalid --num-communicators value: %d", 
				              params->num_communicators);
				return ncclInvalidArgument;
			}
		}
		else if (strcmp(argv[i], "--num-threads") == 0) {
			if (i + 1 >= argc) {
				NCCL_OFI_WARN("Missing value for --num-threads");
				return ncclInvalidArgument;
			}
			params->num_threads = atoi(argv[++i]);
			if (params->num_threads <= 0) {
				NCCL_OFI_WARN("Invalid --num-threads value: %d", 
				              params->num_threads);
				return ncclInvalidArgument;
			}
		}
		else if (strcmp(argv[i], "--buffer-type") == 0) {
			if (i + 1 >= argc) {
				NCCL_OFI_WARN("Missing value for --buffer-type");
				return ncclInvalidArgument;
			}
			i++;
			if (strcmp(argv[i], "host") == 0 || strcmp(argv[i], "HOST") == 0) {
				params->buffer_type = NCCL_PTR_HOST;
			}
			else if (strcmp(argv[i], "cuda") == 0 || strcmp(argv[i], "CUDA") == 0) {
				params->buffer_type = NCCL_PTR_CUDA;
			}
			else {
				NCCL_OFI_WARN("Invalid --buffer-type value: %s (expected 'host' or 'cuda')", 
				              argv[i]);
				return ncclInvalidArgument;
			}
		}
		else if (strcmp(argv[i], "--buffer-size") == 0) {
			if (i + 1 >= argc) {
				NCCL_OFI_WARN("Missing value for --buffer-size");
				return ncclInvalidArgument;
			}
			params->buffer_size = static_cast<size_t>(atoll(argv[++i]));
			if (params->buffer_size == 0) {
				NCCL_OFI_WARN("Invalid --buffer-size value: %zu", 
				              params->buffer_size);
				return ncclInvalidArgument;
			}
		}
		else if (strcmp(argv[i], "--num-iterations") == 0) {
			if (i + 1 >= argc) {
				NCCL_OFI_WARN("Missing value for --num-iterations");
				return ncclInvalidArgument;
			}
			params->num_iterations = atoi(argv[++i]);
			if (params->num_iterations <= 0) {
				NCCL_OFI_WARN("Invalid --num-iterations value: %d", 
				              params->num_iterations);
				return ncclInvalidArgument;
			}
		}
		else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
			params->verbose = true;
		}
		else {
			NCCL_OFI_WARN("Unknown argument: %s", argv[i]);
			return ncclInvalidArgument;
		}
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Parsed arguments: num_communicators=%d, num_threads=%d, "
	               "buffer_type=%d, buffer_size=%zu, num_iterations=%d, verbose=%d",
	               params->num_communicators, params->num_threads, params->buffer_type,
	               params->buffer_size, params->num_iterations, params->verbose);
	
	return ncclSuccess;
}

ncclResult_t parse_test_environment(test_parameters* params)
{
	VALIDATE_NOT_NULL(params, "params");
	
	NCCL_OFI_TRACE(NCCL_NET, "Parsing environment variables");
	
	// Use helper functions for cleaner parsing
	parse_env_int("TEST_NUM_COMMUNICATORS", params->num_communicators);
	parse_env_int("TEST_NUM_THREADS", params->num_threads);
	parse_env_buffer_type("TEST_BUFFER_TYPE", params->buffer_type);
	parse_env_size_t("TEST_BUFFER_SIZE", params->buffer_size);
	parse_env_int("TEST_NUM_ITERATIONS", params->num_iterations);
	parse_env_bool("TEST_VERBOSE", params->verbose);
	
	NCCL_OFI_TRACE(NCCL_NET, "Environment parsing complete");
	return ncclSuccess;
}

void print_test_help(const char* program_name)
{
	// Validate input
	if (program_name == nullptr) {
		program_name = "test";
	}
	
	printf("\n");
	printf("Usage: %s [OPTIONS]\n", program_name);
	printf("\n");
	printf("NCCL OFI Plugin Test Framework\n");
	printf("\n");
	printf("Options:\n");
	printf("  --help, -h                  Show this help message and exit\n");
	printf("  --num-communicators <N>     Number of communicators to create (default: 1)\n");
	printf("  --num-threads <N>           Number of worker threads (default: 1)\n");
	printf("  --buffer-type <TYPE>        Buffer type: 'host' or 'cuda' (default: host)\n");
	printf("  --buffer-size <SIZE>        Size of test buffers in bytes (default: 4096)\n");
	printf("  --num-iterations <N>        Number of test iterations (default: 1)\n");
	printf("  --verbose, -v               Enable verbose logging\n");
	printf("\n");
	printf("Environment Variables:\n");
	printf("  TEST_NUM_COMMUNICATORS      Override default number of communicators\n");
	printf("  TEST_NUM_THREADS            Override default number of threads\n");
	printf("  TEST_BUFFER_TYPE            Override default buffer type ('host' or 'cuda')\n");
	printf("  TEST_BUFFER_SIZE            Override default buffer size in bytes\n");
	printf("  TEST_NUM_ITERATIONS         Override default number of iterations\n");
	printf("  TEST_VERBOSE                Enable verbose logging ('1', 'true', 'yes')\n");
	printf("\n");
	printf("Examples:\n");
	printf("  %s --num-threads 4 --buffer-type cuda\n", program_name);
	printf("  %s --num-communicators 2 --buffer-size 8192 --verbose\n", program_name);
	printf("  TEST_NUM_THREADS=8 %s\n", program_name);
	printf("\n");
}
