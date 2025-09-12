/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <memory>

#include "nccl_ofi_platform.h"
#include "test-logger.h"

// Test helper class to access protected constructor
class TestPlatformManager : public PlatformManager {
public:
	using PlatformManager::PlatformManager;
	using PlatformManager::register_platform;
};

// Dummy test platform for registration testing
class TestPlatform : public Default {
public:
	const char* get_name() const override { return "TestPlatform"; }
	int get_priority() override { return 200; }
};

class CopyTestPlatform : public TestPlatform {
public:
	const char* get_name() const override { return "CopyTestPlatform"; }
	int get_priority() override { return 300; }
};

static int test_default_platform_creation()
{
	int ret = 0;

	PlatformPtr platform = std::make_unique<Default>();

	if (strcmp(platform->get_name(), "Default") != 0) {
		NCCL_OFI_WARN("Expected Default platform name, got %s", platform->get_name());
		ret++;
	}

	if (platform->get_priority() != 0) {
		NCCL_OFI_WARN("Expected Default platform priority to be 0, got %d", platform->get_priority());
		ret++;
	}

	const char *provider_filter = nullptr;
	if (platform->init(&provider_filter) != 0) {
		NCCL_OFI_WARN("Default platform init failed");
		ret++;
	}

	return ret;
}

static int test_platform_manager_default_registration()
{
	int ret = 0;

	TestPlatformManager manager;

	Platform& platform = manager.get_platform();
#ifdef WANT_AWS_PLATFORM
	if (strcmp(platform.get_name(), "Default") != 0 && strcmp(platform.get_name(), "AWS") != 0) {
		NCCL_OFI_WARN("Expected Default or AWS platform, got %s", platform.get_name());
#else
	if (strcmp(platform.get_name(), "Default") != 0) {
		NCCL_OFI_WARN("Expected Default platform, got %s", platform.get_name());
#endif
		ret++;
	}

	return ret;
}

static int test_registration_functionality()
{
	int ret = 0;

	TestPlatformManager manager;

	manager.register_platform(std::make_unique<Default>());

	// Test positive case
	manager.register_platform(std::make_unique<TestPlatform>());

	manager.register_platform(std::make_unique<CopyTestPlatform>());

	// Default, TestPlatform, CopyTestPlatform
	auto& platform = manager.get_platform();
	if (strcmp(platform.get_name(), "CopyTestPlatform") != 0) {
		NCCL_OFI_WARN("Expected CopyTestPlatform platform to be registered by default, got %s",
		              platform.get_name());
		ret++;
	}

	return ret;
}

int main(int argc, char *argv[])
{
	int ret = 0;

	ofi_log_function = logger;

	ret += test_default_platform_creation();
	ret += test_platform_manager_default_registration();
	ret += test_registration_functionality();

	if (ret == 0) {
		printf("Test completed successfully\n");
	}

	return ret;
}
