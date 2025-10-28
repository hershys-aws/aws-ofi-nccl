/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef ENV_PARSER_H_
#define ENV_PARSER_H_

#include <cstdlib>
#include <cstring>
#include <nccl/net.h>

#include "nccl_ofi_log.h"

/**
 * @file env_parser.h
 * @brief Environment variable parsing utilities for test configuration
 * 
 * This header provides inline functions for parsing environment variables
 * into various types with validation and logging.
 */

/**
 * Parse integer from environment variable
 * 
 * @param name Environment variable name
 * @param output Reference to store parsed value
 * @return true if variable exists and was parsed successfully, false otherwise
 */
inline bool parse_env_int(const char* name, int& output)
{
	const char* value = getenv(name);
	if (value == nullptr) {
		return false;
	}
	
	int parsed = atoi(value);
	if (parsed <= 0) {
		NCCL_OFI_WARN("Invalid %s value: %s", name, value);
		return false;
	}
	
	output = parsed;
	NCCL_OFI_TRACE(NCCL_NET, "Set %s=%d from environment", name, parsed);
	return true;
}

/**
 * Parse size_t from environment variable
 * 
 * @param name Environment variable name
 * @param output Reference to store parsed value
 * @return true if variable exists and was parsed successfully, false otherwise
 */
inline bool parse_env_size_t(const char* name, size_t& output)
{
	const char* value = getenv(name);
	if (value == nullptr) {
		return false;
	}
	
	size_t parsed = static_cast<size_t>(atoll(value));
	if (parsed == 0) {
		NCCL_OFI_WARN("Invalid %s value: %s", name, value);
		return false;
	}
	
	output = parsed;
	NCCL_OFI_TRACE(NCCL_NET, "Set %s=%zu from environment", name, parsed);
	return true;
}

/**
 * Parse boolean from environment variable
 * 
 * Accepts: "1", "true", "TRUE", "yes", "YES"
 * 
 * @param name Environment variable name
 * @param output Reference to store parsed value
 * @return true if variable exists and was parsed successfully, false otherwise
 */
inline bool parse_env_bool(const char* name, bool& output)
{
	const char* value = getenv(name);
	if (value == nullptr) {
		return false;
	}
	
	if (strcmp(value, "1") == 0 || 
	    strcmp(value, "true") == 0 || 
	    strcmp(value, "TRUE") == 0 ||
	    strcmp(value, "yes") == 0 ||
	    strcmp(value, "YES") == 0) {
		output = true;
		NCCL_OFI_TRACE(NCCL_NET, "Set %s=true from environment", name);
		return true;
	}
	
	return false;
}

/**
 * Parse buffer type from environment variable
 * 
 * Accepts: "host", "HOST", "cuda", "CUDA"
 * 
 * @param name Environment variable name
 * @param output Reference to store parsed value (NCCL_PTR_HOST or NCCL_PTR_CUDA)
 * @return true if variable exists and was parsed successfully, false otherwise
 */
inline bool parse_env_buffer_type(const char* name, int& output)
{
	const char* value = getenv(name);
	if (value == nullptr) {
		return false;
	}
	
	if (strcmp(value, "host") == 0 || strcmp(value, "HOST") == 0) {
		output = NCCL_PTR_HOST;
		NCCL_OFI_TRACE(NCCL_NET, "Set %s=HOST from environment", name);
		return true;
	}
	else if (strcmp(value, "cuda") == 0 || strcmp(value, "CUDA") == 0) {
		output = NCCL_PTR_CUDA;
		NCCL_OFI_TRACE(NCCL_NET, "Set %s=CUDA from environment", name);
		return true;
	}
	else {
		NCCL_OFI_WARN("Invalid %s value: %s (expected 'host' or 'cuda')", name, value);
		return false;
	}
}

#endif // ENV_PARSER_H_
