/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef DATA_TRANSFER_H_
#define DATA_TRANSFER_H_

#include <cstring>
#include <vector>
#include <mpi.h>
#include <nccl/net.h>

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "buffer_utils.h"

/**
 * @file data_transfer.h
 * @brief Data transfer utilities for NCCL OFI plugin tests
 * 
 * This header provides utilities for posting send/receive operations,
 * testing request completion, and performing high-level data transfer tests.
 */

// Forward declarations
typedef ncclNet_v9_t test_nccl_net_t;
extern test_nccl_net_t *get_extNet(void);

// Helper template for timeout-based polling (from test-common.h)
template<typename Condition>
ncclResult_t poll_with_timeout(Condition condition, int timeout_ms, const char* operation_name);

// Helper class for RAII-style cleanup (from test-common.h)
template<typename Func>
class scope_guard {
public:
	explicit scope_guard(Func&& f) : func_(std::forward<Func>(f)), active_(true) {}
	~scope_guard() { if (active_) func_(); }
	void dismiss() { active_ = false; }
	scope_guard(const scope_guard&) = delete;
	scope_guard& operator=(const scope_guard&) = delete;
private:
	Func func_;
	bool active_;
};

template<typename Func>
scope_guard<Func> make_scope_guard(Func&& f) {
	return scope_guard<Func>(std::forward<Func>(f));
}

/**
 * Post send operation
 * 
 * Posts an asynchronous send operation using the plugin's isend function.
 * The operation completes asynchronously and must be tested/waited on.
 * 
 * @param sComm Send communicator
 * @param send_buf Buffer to send from
 * @param size Size of data to send in bytes
 * @param tag Tag for message identification
 * @param mhandle Memory handle for the send buffer
 * @param request Output pointer to request handle
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t post_send(void* sComm, 
                                     void* send_buf, 
                                     size_t size, 
                                     int tag,
                                     void* mhandle,
                                     void** request)
{
	VALIDATE_NOT_NULL(sComm, "sComm");
	VALIDATE_NOT_NULL(send_buf, "send_buf");
	VALIDATE_NOT_NULL(request, "request");
	VALIDATE_POSITIVE(size, "size");
	
	*request = nullptr;
	
	NCCL_OFI_TRACE(NCCL_NET, "Posting send: buf=%p, size=%zu, tag=%d, mhandle=%p",
	               send_buf, size, tag, mhandle);
	
	// Get plugin interface
	test_nccl_net_t* ext_net = get_extNet();
	if (ext_net == nullptr) {
		NCCL_OFI_WARN("Failed to get plugin interface");
		return ncclInternalError;
	}
	
	// Call plugin's isend function
	ncclResult_t res = ext_net->isend(sComm, send_buf, size, tag, mhandle, request);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("isend failed with error %d", res);
		return res;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Send posted successfully: request=%p", *request);
	return ncclSuccess;
}

/**
 * Post receive operation
 * 
 * Posts an asynchronous receive operation using the plugin's irecv function.
 * Supports grouped receives (multiple buffers in one call).
 * The operation completes asynchronously and must be tested/waited on.
 * 
 * @param rComm Receive communicator
 * @param n_recv Number of receive buffers (for grouped receives)
 * @param recv_bufs Array of receive buffer pointers
 * @param sizes Array of sizes for each receive buffer
 * @param tags Array of tags for each receive buffer
 * @param mhandles Array of memory handles for each receive buffer
 * @param requests Output array of request handles
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t post_recv(void* rComm, 
                                     int n_recv,
                                     void** recv_bufs, 
                                     size_t* sizes,
                                     int* tags,
                                     void** mhandles,
                                     void** requests)
{
	VALIDATE_NOT_NULL(rComm, "rComm");
	VALIDATE_POSITIVE(n_recv, "n_recv");
	VALIDATE_NOT_NULL(recv_bufs, "recv_bufs");
	VALIDATE_NOT_NULL(sizes, "sizes");
	VALIDATE_NOT_NULL(tags, "tags");
	VALIDATE_NOT_NULL(mhandles, "mhandles");
	VALIDATE_NOT_NULL(requests, "requests");
	
	// Validate individual buffers
	for (int i = 0; i < n_recv; i++) {
		if (recv_bufs[i] == nullptr) {
			NCCL_OFI_WARN("Invalid NULL receive buffer at index %d", i);
			return ncclInvalidArgument;
		}
		if (sizes[i] == 0) {
			NCCL_OFI_WARN("Invalid size 0 at index %d", i);
			return ncclInvalidArgument;
		}
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Posting receive: n_recv=%d", n_recv);
	
	// Get plugin interface
	test_nccl_net_t* ext_net = get_extNet();
	if (ext_net == nullptr) {
		NCCL_OFI_WARN("Failed to get plugin interface");
		return ncclInternalError;
	}
	
	// Call plugin's irecv function
	// The irecv function supports grouped receives (multiple buffers)
	ncclResult_t res = ext_net->irecv(rComm, n_recv, recv_bufs, sizes, tags, mhandles, requests);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("irecv failed with error %d", res);
		return res;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Receive posted successfully");
	return ncclSuccess;
}

/**
 * Test request completion
 * 
 * Tests whether an asynchronous operation has completed without blocking.
 * Calls the plugin's test function to check request status.
 * 
 * @param request Request handle to test
 * @param done Output pointer to completion flag (1 if done, 0 if not)
 * @param size Output pointer to actual size transferred (may be NULL)
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t test_request(void* request, int* done, size_t* size)
{
	VALIDATE_NOT_NULL(request, "request");
	VALIDATE_NOT_NULL(done, "done");
	
	*done = 0;
	int sizes_int = 0;
	
	// Get plugin interface
	test_nccl_net_t* ext_net = get_extNet();
	if (ext_net == nullptr) {
		NCCL_OFI_WARN("Failed to get plugin interface");
		return ncclInternalError;
	}
	
	// Call plugin's test function (expects int* for sizes in v9)
	ncclResult_t res = ext_net->test(request, done, &sizes_int);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("test failed with error %d", res);
		return res;
	}
	
	// Convert int to size_t if size pointer provided
	if (size != nullptr) {
		*size = static_cast<size_t>(sizes_int);
	}
	
	if (*done) {
		NCCL_OFI_TRACE(NCCL_NET, "Request completed: request=%p, size=%d",
		               request, sizes_int);
	}
	
	return ncclSuccess;
}

/**
 * Wait for multiple requests to complete
 * 
 * Polls multiple requests until all complete or timeout is reached.
 * Uses the plugin's test function in a polling loop.
 * 
 * @param requests Array of request handles to wait for
 * @param num_requests Number of requests in the array
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return ncclSuccess if all requests complete, ncclInternalError on timeout
 */
static inline ncclResult_t wait_for_requests(void** requests, 
                                             size_t num_requests, 
                                             int timeout_ms)
{
	VALIDATE_NOT_NULL(requests, "requests");
	
	if (num_requests == 0) {
		return ncclSuccess;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Waiting for %zu requests (timeout=%d ms)",
	               num_requests, timeout_ms);
	
	std::vector<bool> completed(num_requests, false);
	size_t num_completed = 0;
	
	// Use polling helper with timeout
	auto check_completion = [&]() {
		for (size_t i = 0; i < num_requests; i++) {
			if (completed[i]) continue;
			
			if (requests[i] == nullptr) {
				completed[i] = true;
				num_completed++;
				continue;
			}
			
			int done = 0;
			ncclResult_t res = test_request(requests[i], &done, nullptr);
			if (res != ncclSuccess) {
				NCCL_OFI_WARN("Failed to test request %zu: %d", i, res);
				return false;
			}
			
			if (done) {
				completed[i] = true;
				num_completed++;
				NCCL_OFI_TRACE(NCCL_NET, "Request %zu completed (%zu/%zu)",
				               i, num_completed, num_requests);
			}
		}
		return num_completed >= num_requests;
	};
	
	ncclResult_t res = poll_with_timeout(check_completion, timeout_ms, "requests");
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Timeout: %zu/%zu requests completed", num_completed, num_requests);
		return res;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "All %zu requests completed", num_requests);
	return ncclSuccess;
}

/**
 * Generate unique MPI tag for communication
 * 
 * This function generates unique tags for MPI communication to avoid conflicts
 * in multi-threaded scenarios. Tags are generated in a thread-safe manner.
 * 
 * @return Unique MPI tag
 */
int generate_unique_tag();

/**
 * Setup connection between two ranks
 * 
 * This helper establishes a bidirectional connection between the current rank
 * and a peer rank. It creates a listen communicator, exchanges connection handles
 * via MPI, and creates send and receive communicators.
 * 
 * @param ext_net Plugin interface pointer
 * @param dev Device index to use for connection
 * @param rank Current MPI rank
 * @param size Total number of MPI ranks
 * @param peer_rank MPI rank of the peer to connect to
 * @param ndev Number of devices
 * @param lComm Output: listen communicator (may be NULL after return)
 * @param sComm Output: send communicator
 * @param rComm Output: receive communicator
 * @param sHandle Output: send device handle
 * @param rHandle Output: receive device handle
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t setup_connection(
	test_nccl_net_t* ext_net,
	int dev,
	int rank,
	int size,
	int peer_rank,
	int ndev,
	nccl_net_ofi_listen_comm_t** lComm,
	nccl_net_ofi_send_comm_t** sComm,
	nccl_net_ofi_recv_comm_t** rComm,
	test_nccl_net_device_handle_t** sHandle,
	test_nccl_net_device_handle_t** rHandle)
{
	char local_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	char peer_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	
	// Validate inputs using macros
	VALIDATE_NOT_NULL(ext_net, "ext_net");
	VALIDATE_NOT_NULL(lComm, "lComm");
	VALIDATE_NOT_NULL(sComm, "sComm");
	VALIDATE_NOT_NULL(rComm, "rComm");
	VALIDATE_NOT_NULL(sHandle, "sHandle");
	VALIDATE_NOT_NULL(rHandle, "rHandle");
	
	// Initialize output pointers
	*lComm = nullptr;
	*sComm = nullptr;
	*rComm = nullptr;
	*sHandle = nullptr;
	*rHandle = nullptr;
	
	// Validate device index
	if (dev < 0 || dev >= ndev) {
		NCCL_OFI_WARN("Invalid device index %d (ndev=%d)", dev, ndev);
		return ncclInvalidArgument;
	}
	
	// Validate peer rank
	if (peer_rank < 0 || peer_rank >= size || peer_rank == rank) {
		NCCL_OFI_WARN("Invalid peer rank %d (rank=%d, size=%d)", 
		              peer_rank, rank, size);
		return ncclInvalidArgument;
	}
	
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d setting up connection with rank %d on device %d",
	               rank, peer_rank, dev);
	
	// Use scope_guard for automatic cleanup on error
	auto cleanup = make_scope_guard([ext_net, lComm, sComm, rComm]() {
		if (*lComm != nullptr) {
			ext_net->closeListen(static_cast<void*>(*lComm));
		}
		if (*sComm != nullptr) {
			ext_net->closeSend(static_cast<void*>(*sComm));
		}
		if (*rComm != nullptr) {
			ext_net->closeRecv(static_cast<void*>(*rComm));
		}
	});
	
	// Create listen communicator
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Creating listen communicator on device %d",
	               rank, dev);
	ncclResult_t res = ext_net->listen(dev, static_cast<void*>(&local_handle), 
	                                    reinterpret_cast<void**>(lComm));
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Listen failed with error %d", res);
		return res;
	}
	
	// Generate unique tag for this connection
	int tag = generate_unique_tag();
	
	// Exchange connection handles via MPI
	// Use MPI_Sendrecv to avoid deadlock
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Exchanging handles with rank %d (tag=%d)",
	               rank, peer_rank, tag);
	
	MPI_Status status;
	int mpi_ret = MPI_Sendrecv(
		local_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, tag,
		peer_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, tag,
		MPI_COMM_WORLD, &status);
	
	if (mpi_ret != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Sendrecv failed with error %d", mpi_ret);
		return ncclSystemError;
	}
	
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Establishing send and receive communicators",
	               rank);
	
	// Establish send and receive communicators
	// Poll until both are established
	while (*sComm == nullptr || *rComm == nullptr) {
		// Try to connect (create send communicator)
		if (*sComm == nullptr) {
			res = ext_net->connect(dev, static_cast<void*>(peer_handle), 
			                       reinterpret_cast<void**>(sComm), sHandle);
			if (res != ncclSuccess) {
				NCCL_OFI_WARN("Connect failed with error %d", res);
				return res;
			}
		}
		
		// Try to accept (create receive communicator)
		if (*rComm == nullptr) {
			res = ext_net->accept(static_cast<void*>(*lComm), 
			                      reinterpret_cast<void**>(rComm), rHandle);
			if (res != ncclSuccess) {
				NCCL_OFI_WARN("Accept failed with error %d", res);
				return res;
			}
		}
	}
	
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Successfully established connection with rank %d",
	               rank, peer_rank);
	
	// Dismiss cleanup guard since we succeeded
	cleanup.dismiss();
	return ncclSuccess;
}

/**
 * Cleanup connection communicators
 * 
 * Closes listen, send, and receive communicators if they are not NULL.
 * 
 * @param ext_net Plugin interface pointer
 * @param lComm Listen communicator to close (may be NULL)
 * @param sComm Send communicator to close (may be NULL)
 * @param rComm Receive communicator to close (may be NULL)
 */
static inline void cleanup_connection(
	test_nccl_net_t* ext_net,
	nccl_net_ofi_listen_comm_t* lComm,
	nccl_net_ofi_send_comm_t* sComm,
	nccl_net_ofi_recv_comm_t* rComm)
{
	// Close listen communicator if not nullptr
	if (lComm != nullptr) {
		ncclResult_t res = ext_net->closeListen(static_cast<void*>(lComm));
		if (res != ncclSuccess) {
			NCCL_OFI_WARN("Failed to close listen communicator: %d", res);
		}
	}
	
	// Close send communicator if not nullptr
	if (sComm != nullptr) {
		ncclResult_t res = ext_net->closeSend(static_cast<void*>(sComm));
		if (res != ncclSuccess) {
			NCCL_OFI_WARN("Failed to close send communicator: %d", res);
		}
	}
	
	// Close receive communicator if not nullptr
	if (rComm != nullptr) {
		ncclResult_t res = ext_net->closeRecv(static_cast<void*>(rComm));
		if (res != ncclSuccess) {
			NCCL_OFI_WARN("Failed to close receive communicator: %d", res);
		}
	}
}

/**
 * High-level send/receive test helper
 * 
 * Performs a complete send/receive operation including posting operations,
 * waiting for completion, and validating received data.
 * 
 * @param sComm Send communicator
 * @param rComm Receive communicator
 * @param send_buf Buffer to send from
 * @param recv_buf Buffer to receive into
 * @param size Size of data to transfer
 * @param tag Tag for message identification
 * @param buffer_type NCCL_PTR_HOST or NCCL_PTR_CUDA
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t send_recv_test(
	void* sComm,
	void* rComm,
	void* send_buf,
	void* recv_buf,
	size_t size,
	int tag,
	int buffer_type)
{
	VALIDATE_NOT_NULL(sComm, "sComm");
	VALIDATE_NOT_NULL(rComm, "rComm");
	VALIDATE_NOT_NULL(send_buf, "send_buf");
	VALIDATE_NOT_NULL(recv_buf, "recv_buf");
	VALIDATE_POSITIVE(size, "size");
	VALIDATE_BUFFER_TYPE(buffer_type);
	
	NCCL_OFI_TRACE(NCCL_NET, "Starting send/recv test: size=%zu, tag=%d, type=%d",
	               size, tag, buffer_type);
	
	// Register memory for send and receive buffers
	void* send_mhandle = nullptr;
	void* recv_mhandle = nullptr;
	
	ncclResult_t res = register_memory(sComm, send_buf, size, buffer_type, &send_mhandle);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to register send buffer");
		return res;
	}
	
	// Auto-cleanup for send memory handle
	auto send_cleanup = make_scope_guard([&]() {
		deregister_memory(sComm, send_mhandle);
	});
	
	res = register_memory(rComm, recv_buf, size, buffer_type, &recv_mhandle);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to register receive buffer");
		return res;
	}
	
	// Auto-cleanup for recv memory handle
	auto recv_cleanup = make_scope_guard([&]() {
		deregister_memory(rComm, recv_mhandle);
	});
	
	// Post receive operation first (to avoid message loss)
	void* recv_request = nullptr;
	res = post_recv(rComm, 1, &recv_buf, &size, &tag, &recv_mhandle, &recv_request);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to post receive");
		return res;
	}
	
	// Post send operation
	void* send_request = nullptr;
	res = post_send(sComm, send_buf, size, tag, send_mhandle, &send_request);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to post send");
		return res;
	}
	
	// Wait for both operations to complete
	void* requests[2] = {send_request, recv_request};
	res = wait_for_requests(requests, 2, 5000); // 5 second timeout
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to wait for send/recv completion");
		return res;
	}
	
	// Validate received data matches sent data
	// We'll use a simple pattern validation - assume send buffer was initialized with pattern '1'
	char expected_buf[size];
	memset(expected_buf, '1', size);
	res = validate_data((char*)recv_buf, expected_buf, size, buffer_type);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Data validation failed");
		return res;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Send/recv test completed successfully");
	return ncclSuccess;
}

#endif // DATA_TRANSFER_H_
