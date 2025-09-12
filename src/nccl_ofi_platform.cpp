/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "nccl_ofi_platform.h"
#ifdef WANT_AWS_PLATFORM
#include "platform-aws.h"
#endif

PlatformManager::PlatformManager()
: override_(ofi_nccl_platform_override.get()) {
	register_platform(std::make_unique<Default>());
#ifdef WANT_AWS_PLATFORM
	register_platform(std::make_unique<PlatformAWS>());
#endif
}

PlatformManager& PlatformManager::get_global() {
	static PlatformManager manager;
	return manager;
}
