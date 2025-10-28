/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef DEVICE_UTILS_H_
#define DEVICE_UTILS_H_

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "test-common.h"

/**
 * @file device_utils.h
 * @brief Device property utilities for NCCL OFI plugin tests
 * 
 * This header provides utilities for working with device properties,
 * including printing device information and checking capabilities.
 */

/**
 * Print device properties
 * 
 * Logs detailed information about a device's properties including
 * PCIe path, pointer support, GUID, speed, port, and limits.
 * 
 * @param dev Device index
 * @param props Pointer to device properties structure
 */
static inline void print_dev_props(int dev, test_nccl_properties_t *props)
{
	NCCL_OFI_TRACE(NCCL_NET, "****************** Device %d Properties ******************", dev);
	NCCL_OFI_TRACE(NCCL_NET, "%s: PCIe Path: %s", props->name, props->pciPath);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Plugin Support: %d", props->name, props->ptrSupport);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Device GUID: %zu", props->name, props->guid);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Device Speed: %d", props->name, props->speed);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Device Port: %d", props->name, props->port);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Communicators: %d", props->name, props->maxComms);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Grouped Receives: %d", props->name, props->maxRecvs);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Global registration: %d", props->name, props->regIsGlobal);
}

/**
 * Check if device supports GPUDirect RDMA (GDR)
 * 
 * Determines whether a device supports CUDA pointer types,
 * which indicates GDR capability.
 * 
 * @param ptr_support Pointer support flags from device properties
 * @return 1 if GDR is supported, 0 otherwise
 */
static inline int is_gdr_supported_nic(uint64_t ptr_support)
{
	if (ptr_support & NCCL_PTR_CUDA)
		return 1;

	return 0;
}

#endif // DEVICE_UTILS_H_
