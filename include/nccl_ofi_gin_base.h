/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_BASE_H_
#define NCCL_OFI_GIN_BASE_H_

/**
 * Abstract GIN endpoint. Obtained from domain via get_gin_ep().
 * Serves as the base class for transport-specific GIN EP implementations.
 *
 * TODO: Add pure virtual methods for the GIN API surface so the API
 * layer can work through this interface without downcasting.
 */
class nccl_ofi_gin_ep_t {
public:
	virtual ~nccl_ofi_gin_ep_t() = default;
};

/**
 * Abstract GIN symmetric MR handle. Opaque to the API layer.
 */
class nccl_ofi_gin_symm_mr_handle_t {
public:
	virtual ~nccl_ofi_gin_symm_mr_handle_t() = default;
};

/**
 * Abstract GIN request. Returned by iputSignal, polled via test().
 */
class nccl_ofi_gin_req_t {
public:
	virtual ~nccl_ofi_gin_req_t() = default;
};

/**
 * Abstract GIN listen communicator. Created during connection setup,
 * produces a put_comm via connect().
 */
class nccl_ofi_gin_listen_comm_t {
public:
	virtual ~nccl_ofi_gin_listen_comm_t() = default;
};

/**
 * Abstract GIN put communicator. Provides data transfer and MR operations.
 */
class nccl_ofi_gin_put_comm_t {
public:
	virtual ~nccl_ofi_gin_put_comm_t() = default;
};

#endif /* NCCL_OFI_GIN_BASE_H_ */
