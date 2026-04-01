/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_BASE_H_
#define NCCL_OFI_GIN_BASE_H_

/**
 * Abstract GIN endpoint. Obtained from domain via get_gin_ep().
 * Serves as the base class for transport-specific GIN EP implementations.
 *
 * TODO: Add pure virtual methods for the GIN API surface (listen,
 * regMrSymDmaBuf, iputSignal, etc.) so the API layer can work through
 * this interface without downcasting to concrete types.
 */
class nccl_ofi_gin_ep_t {
public:
	virtual ~nccl_ofi_gin_ep_t() = default;
};

#endif /* NCCL_OFI_GIN_BASE_H_ */
