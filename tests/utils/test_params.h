/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef TEST_PARAMS_H_
#define TEST_PARAMS_H_

#include <cstddef>
#include "nccl_ofi.h"

/**
 * @file test_params.h
 * @brief Test parameter parsing and configuration utilities
 * 
 * This header provides utilities for parsing command-line arguments and
 * environment variables to configure test parameters.
 */

/**
 * test_parameters - Centralized configuration for test parameters
 * 
 * This structure holds common test configuration parameters that can be
 * set via command-line arguments or environment variables.
 */
struct test_parameters {
	int num_communicators;            // Number of communicators to create
	int num_threads;                  // Number of worker threads
	int buffer_type;                  // NCCL_PTR_HOST or NCCL_PTR_CUDA
	size_t buffer_size;               // Size of test buffers in bytes
	int num_iterations;               // Number of test iterations
	bool verbose;                     // Enable verbose logging
};

/**
 * Parse command-line arguments into test_parameters structure
 * 
 * Parses common test parameters from command-line arguments including:
 * - --num-communicators: Number of communicators to create
 * - --num-threads: Number of worker threads
 * - --buffer-type: Buffer type ('host' or 'cuda')
 * - --buffer-size: Size of test buffers in bytes
 * - --num-iterations: Number of test iterations
 * - --verbose, -v: Enable verbose logging
 * - --help, -h: Show help message (returns ncclInvalidArgument)
 * 
 * Sets sensible defaults for all parameters. Validates argument values
 * and returns an error if invalid values are provided.
 * 
 * @param argc Command-line argument count
 * @param argv Command-line argument values
 * @param params Output pointer to test_parameters structure to populate
 * @return ncclSuccess on success, ncclInvalidArgument if help requested or invalid arguments
 */
ncclResult_t parse_test_arguments(int argc, 
                                   char* argv[], 
                                   test_parameters* params);

/**
 * Parse environment variables into test_parameters structure
 * 
 * Reads common test parameters from environment variables:
 * - TEST_NUM_COMMUNICATORS: Number of communicators to create
 * - TEST_NUM_THREADS: Number of worker threads
 * - TEST_BUFFER_TYPE: Buffer type ('host' or 'cuda')
 * - TEST_BUFFER_SIZE: Size of test buffers in bytes
 * - TEST_NUM_ITERATIONS: Number of test iterations
 * - TEST_VERBOSE: Enable verbose logging ('1', 'true', 'yes')
 * 
 * Environment variables override the default values in the params structure.
 * Invalid values are logged as warnings but do not cause failure.
 * 
 * @param params Pointer to test_parameters structure to update
 * @return ncclSuccess on success, ncclInvalidArgument if params is NULL
 */
ncclResult_t parse_test_environment(test_parameters* params);

/**
 * Print help message for test program
 * 
 * Prints a formatted help message describing all available command-line
 * arguments, environment variables, default values, and usage examples.
 * 
 * @param program_name Name of the program (from argv[0])
 */
void print_test_help(const char* program_name);

#endif // TEST_PARAMS_H_
