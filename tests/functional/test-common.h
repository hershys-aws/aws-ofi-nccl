/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include "config.h"

#include <array>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl/net.h>
#include <mpi.h>
#include <thread>

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_param.h"

#define STR2(v)		#v
#define STR(v)		STR2(v)

#define NUM_REQUESTS	(NCCL_NET_MAX_REQUESTS)
#define SEND_SIZE	(5000)
#define RECV_SIZE	(5200)

#define OFINCCLCHECK(call)                                                \
	do {                                                              \
		ncclResult_t macro_res = call;                            \
		if (macro_res != ncclSuccess) {                           \
			NCCL_OFI_WARN("OFI NCCL failure: %d", macro_res); \
			return macro_res;                                 \
		}                                                         \
	} while (false);

#define OFINCCLCHECKGOTO(call, res, label) do {			\
	res = call;						\
	if (res != ncclSuccess) {				\
		NCCL_OFI_WARN("OFI NCCL failure: %d", res);	\
		goto label;					\
	}							\
} while (false);

#define CUDACHECK(call) do {						\
        cudaError_t e = call;						\
        if (e != cudaSuccess) {						\
	    const char *error_str = cudaGetErrorString(e);		\
	    NCCL_OFI_WARN("Cuda failure '%s'", error_str);		\
	    return ncclUnhandledCudaError;				\
        }								\
} while(false);

#define PROC_NAME_IDX(i) ((i) * MPI_MAX_PROCESSOR_NAME)

// Can be changed when porting new versions to the plugin
#define NCCL_PLUGIN_SYMBOL ncclNetPlugin_v9

typedef ncclNet_v9_t test_nccl_net_t;
typedef ncclNetProperties_v9_t test_nccl_properties_t;
typedef ncclNetDeviceHandle_v9_t test_nccl_net_device_handle_t;

// Convenience aliases for testing
using ListenComms = std::vector<nccl_net_ofi_listen_comm_t*>;
using SendComms = std::vector<nccl_net_ofi_send_comm_t*>;
using RecvComms = std::vector<nccl_net_ofi_recv_comm_t*>;

static void logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
		   int line, const char *fmt, ...)
{
	va_list vargs;

	switch (level) {
		case NCCL_LOG_WARN:
			printf("WARN: Function: %s Line: %d: ", filefunc, line);
			break;
		case NCCL_LOG_INFO:
			printf("INFO: Function: %s Line: %d: ", filefunc, line);
			break;
		case NCCL_LOG_TRACE:
#if OFI_NCCL_TRACE
			printf("TRACE: Function: %s Line: %d: ", filefunc, line);
			break;
#else
			return;
#endif
		case NCCL_LOG_NONE:
		case NCCL_LOG_VERSION:
		case NCCL_LOG_ABORT:
		default:
			break;
	};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat=2"
	va_start(vargs, fmt);
	vprintf(fmt, vargs);
	printf("\n");
	va_end(vargs);
#pragma GCC diagnostic pop
}

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

static inline int is_gdr_supported_nic(uint64_t ptr_support)
{
	if (ptr_support & NCCL_PTR_CUDA)
		return 1;

	return 0;
}

/**
 * Allocate buffer (host or CUDA device memory)
 *
 * @param buf Output pointer to allocated buffer
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA: {
		// HACK: Round up to nearest 4KB to prevent running into mr_cache
		// bug on small unaligned memory allocations
		auto aligned_size = ((size + system_page_size - 1) / system_page_size) * system_page_size;
		CUDACHECK(cudaMalloc(buf, aligned_size));
		break;
	}
	case NCCL_PTR_HOST:
		CUDACHECK(cudaHostAlloc(buf, size, cudaHostAllocMapped));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Initialize buffer with a pattern
 *
 * @param buf Buffer to initialize
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaMemset(buf, '1', size));
		break;
	case NCCL_PTR_HOST:
		memset(buf, '1', size);
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Deallocate buffer
 *
 * @param buf Buffer to deallocate
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t deallocate_buffer(void *buf, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaFree(buf));
		break;
	case NCCL_PTR_HOST:
		CUDACHECK(cudaFreeHost((void *)buf));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Validate received data against expected data
 *
 * @param recv_buf Received buffer to validate
 * @param expected_buf Expected buffer to compare against
 * @param size Size of buffers in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess if data matches, error code otherwise
 */
static inline ncclResult_t validate_data(char *recv_buf, char *expected_buf, size_t size, int buffer_type)
{
	int ret = 0;
	char *recv_buf_host = NULL;

	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		OFINCCLCHECK(allocate_buff((void **)&recv_buf_host, size, NCCL_PTR_HOST));
		CUDACHECK(cudaMemcpy(recv_buf_host, recv_buf, size, cudaMemcpyDeviceToHost));

		ret = memcmp(recv_buf_host, expected_buf, size);
		if (ret != 0) {
			// Find first mismatch
			for (size_t i = 0; i < size; i++) {
				if (recv_buf_host[i] != expected_buf[i]) {
					NCCL_OFI_WARN("Data validation failed at byte %zu: recv=0x%02x expected=0x%02x (size=%zu)",
						      i, (unsigned char)recv_buf_host[i], (unsigned char)expected_buf[i], size);
					break;
				}
			}
			NCCL_OFI_WARN("Data validation check failed. RC: %d, Buffer Type: %d",
				      ret, buffer_type);
			return ncclSystemError;
		}
		break;
	case NCCL_PTR_HOST:
		ret = memcmp(recv_buf, expected_buf, size);
		if (ret != 0) {
			NCCL_OFI_WARN("Data validation check failed. RC: %d, Buffer Type: %d",
				      ret, buffer_type);
			return ncclSystemError;
		}
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Post send operation
 *
 * Posts an asynchronous send operation using the plugin's isend function.
 * The operation completes asynchronously and must be tested/waited on.
 *
 * @param ext_next Pointer to external plugin
 * @param scomm Send communicator
 * @param send_buf Buffer to send from
 * @param size Size of data to send in bytes
 * @param tag Tag for message identification
 * @param mhandle Memory handle for the send buffer
 * @param request Output pointer to request handle
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t post_send(test_nccl_net_t* ext_net,
				     void* scomm,
				     void* send_buf,
				     size_t size,
				     int tag,
				     void* mhandle,
				     void** request)
{
	*request = nullptr;

	NCCL_OFI_TRACE(NCCL_NET, "Posting send: buf=%p, size=%zu, tag=%d, mhandle=%p",
		send_buf, size, tag, mhandle);
	// Retry until we get a valid request
	while (*request == nullptr) {
		OFINCCLCHECK(ext_net->isend(scomm, send_buf, size, tag, mhandle, request));
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
 * @param ext_net Pointer to external plugin interface
 * @param rcomm Receive communicator
 * @param n_recv Number of receive buffers (for grouped receives)
 * @param recv_bufs Array of receive buffer pointers
 * @param sizes Array of sizes for each receive buffer
 * @param tags Array of tags for each receive buffer
 * @param mhandles Array of memory handles for each receive buffer
 * @param requests Output array of request handles
 * @return ncclSuccess on success, error code otherwise
 */
inline ncclResult_t post_recv(test_nccl_net_t* ext_net,
			      void* rcomm,
			      int n_recv,
			      void** recv_bufs,
			      size_t* sizes,
			      int* tags,
			      void** mhandles,
			      void** requests)
{
	// Retry until we get a valid request
	NCCL_OFI_TRACE(NCCL_NET, "Posting receive: n_recv=%d", n_recv);
	while (*requests == nullptr) {
		OFINCCLCHECK(ext_net->irecv(rcomm, n_recv, recv_bufs, sizes, tags, mhandles, requests));
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
 * @param ext_net Pointer to external plugin
 * @param request Request handle to test
 * @param done Output pointer to completion flag (1 if done, 0 if not)
 * @param size Output pointer to actual size transferred (may be NULL)
 * @return ncclSuccess on success, error code otherwise
 */
inline ncclResult_t test_request(test_nccl_net_t* ext_net,
				 void* request,
				 int* done,
				 size_t* size)
{
	*done = 0;
	int sizes_int = 0;

	while (!*done) {
		// Call plugin's test function (expects int* for sizes in v9)
		OFINCCLCHECK(ext_net->test(request, done, &sizes_int));
	}

	// Convert int to size_t if size pointer provided
	if (size != nullptr) {
		*size = static_cast<size_t>(sizes_int);
	}

	NCCL_OFI_TRACE(NCCL_NET, "Request completed: request=%p, size=%d",
		request, sizes_int);
	return ncclSuccess;
}

/**
 * Setup connection between two ranks
 *
 * This helper establishes a bidirectional connection between the current rank
 * and a peer rank. It creates a listen communicator, exchanges connection handles
 * via MPI, and creates send and receive communicators.
 *
 * @param ext_net Pointer to external plugin
 * @param dev Device index to use for connection
 * @param rank Current MPI rank
 * @param size Total number of MPI ranks
 * @param peer_rank MPI rank of the peer to connect to
 * @param ndev Number of devices
 * @param lcomm Output: listen communicator (may be NULL after return)
 * @param scomm Output: send communicator
 * @param rcomm Output: receive communicator
 * @param sHandle Output: send device handle
 * @param rHandle Output: receive device handle
 * @return ncclSuccess on success, error code otherwise
 */
inline ncclResult_t setup_connection(
	test_nccl_net_t* ext_net,
	int dev,
	int rank,
	int size,
	int peer_rank,
	int ndev,
	MPI_Comm comm,
	nccl_net_ofi_listen_comm_t** lcomm,
	nccl_net_ofi_send_comm_t** scomm,
	nccl_net_ofi_recv_comm_t** rcomm,
	test_nccl_net_device_handle_t** shandle,
	test_nccl_net_device_handle_t** rhandle)
{
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

	// Initialize output pointers
	*lcomm = nullptr;
	*scomm = nullptr;
	*rcomm = nullptr;
	*shandle = nullptr;
	*rhandle = nullptr;
	char local_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	char peer_handle[NCCL_NET_HANDLE_MAXSIZE] = {};



	// Create listen communicator
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Creating listen communicator on device %d",
		rank, dev);
	OFINCCLCHECK(ext_net->listen(dev, static_cast<void*>(&local_handle),
			      reinterpret_cast<void**>(lcomm)));

	// Exchange connection handles via MPI
	// Use MPI_Sendrecv to avoid deadlock
	MPI_Status status;
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Exchanging handles with rank %d",
		rank, peer_rank);
	auto mpi_ret = MPI_Sendrecv(
		local_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0,
		peer_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0,
		comm, &status);
	if (mpi_ret != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Sendrecv failed with error %d", mpi_ret);
		return ncclSystemError;
	}
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Establishing send and receive communicators",
		rank);

	// Establish send and receive communicators
	// Poll until both are established
	while (*scomm == nullptr || *rcomm == nullptr) {
		// Try to connect (create send communicator)
		if (*scomm == nullptr) {
			OFINCCLCHECK(ext_net->connect(dev, static_cast<void*>(peer_handle),
				 reinterpret_cast<void**>(scomm), shandle));
		}

		// Try to accept (create receive communicator)
		if (*rcomm == nullptr) {
			OFINCCLCHECK(ext_net->accept(static_cast<void*>(*lcomm),
				reinterpret_cast<void**>(rcomm), rhandle));
		}
	}

	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Successfully established connection with rank %d",
		rank, peer_rank);

	return ncclSuccess;
}

/**
 * Cleanup connection communicators
 *
 * Closes listen, send, and receive communicators if they are not NULL.
 *
 * @param ext_net Pointer to external plugin interface
 * @param lcomm Listen communicator to close (may be NULL)
 * @param scomm Send communicator to close (may be NULL)
 * @param rcomm Receive communicator to close (may be NULL)
 */
inline ncclResult_t cleanup_connection(
	test_nccl_net_t* ext_net,
	nccl_net_ofi_listen_comm_t* lcomm,
	nccl_net_ofi_send_comm_t* scomm,
	nccl_net_ofi_recv_comm_t* rcomm)
{
	// Close listen communicator if not nullptr
	if (lcomm != nullptr) {
		OFINCCLCHECK(ext_net->closeListen(static_cast<void*>(lcomm)));
	}

	// Close send communicator if not nullptr
	if (scomm != nullptr) {
		OFINCCLCHECK(ext_net->closeSend(static_cast<void*>(scomm)));
	}

	// Close receive communicator if not nullptr
	if (rcomm != nullptr) {
		OFINCCLCHECK(ext_net->closeRecv(static_cast<void*>(rcomm)));
	}

	return ncclSuccess;
}

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
	NCCL_OFI_TRACE(NCCL_NET, "Initializing CUDA for thread with device %d", cuda_device);

	// Set CUDA device for this thread and perform a dummy operation to
	// initialize CUDA context for this thread
	CUDACHECK(cudaSetDevice(cuda_device));
	CUDACHECK(cudaFree(nullptr));

	NCCL_OFI_TRACE(NCCL_NET, "CUDA initialized successfully for thread with device %d", cuda_device);
	return ncclSuccess;
}

/**
 * Initialize MPI and get rank information
 *
 * @param rank       Output: current rank
 * @param size       Output: total ranks
 * @param local_rank Output: local rank on node
 * @return ncclSuccess on success
 */
static inline ncclResult_t mpi_init_ranks(int* rank, int* size, int* local_rank)
{
	int provided;
	MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE) {
		NCCL_OFI_WARN("MPI does not support MPI_THREAD_MULTIPLE (provided=%d)", provided);
		return ncclSystemError;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, size);

	/* Get processor names to calculate local rank */
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	int proc_name_len;
	MPI_Get_processor_name(proc_name, &proc_name_len);

	std::shared_ptr<char[]> all_proc_names(
		new char[(*size) * MPI_MAX_PROCESSOR_NAME]
	);
	if (!all_proc_names) {
		NCCL_OFI_WARN("Failed to allocate memory for processor names");
		return ncclInternalError;
	}

	memcpy(&(all_proc_names.get()[PROC_NAME_IDX(*rank)]), proc_name, MPI_MAX_PROCESSOR_NAME);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_names.get(),
	       MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Calculate local rank */
	*local_rank = 0;
	for (int i = 0; i < *size; i++) {
		if (!strcmp(&(all_proc_names.get()[PROC_NAME_IDX(*rank)]),
	      &(all_proc_names.get()[PROC_NAME_IDX(i)]))) {
			if (i < *rank) {
				(*local_rank)++;
			}
		}
	}

	NCCL_OFI_INFO(NCCL_INIT, "MPI initialized: rank %d/%d, local_rank %d, host %s",
	              *rank, *size, *local_rank, proc_name);

	return ncclSuccess;
}

/**
 * @brief Determines GDR (GPUDirect RDMA) support for all available network devices
 *
 * This function queries all available network devices and checks if they support
 * GPUDirect RDMA operations. For each device, it retrieves properties and determines
 * GDR support based on the device's pointer support capabilities.
 *
 * @param ext_net Pointer to the NCCL network plugin interface
 *
 * @return std::shared_ptr<int[]> Array of GDR support flags for each device, where:
 *         - Each element is 1 if the corresponding device supports GDR, 0 if not
 *         - Array index corresponds to device index
 *         - Returns nullptr if device query fails or properties cannot be retrieved
 *         - Array size equals the number of available devices
 */
static inline std::shared_ptr<int[]> get_support_gdr(test_nccl_net_t* ext_net) {
	int ndev;
	if (ext_net->devices(&ndev)) return nullptr;
	auto gdr_support = std::shared_ptr<int[]>(new int[ndev]);

	for (int dev = 0; dev < ndev; dev++) {
		test_nccl_properties_t props = {};
		if (ext_net->getProperties(dev, &props) != ncclSuccess) return nullptr;

		print_dev_props(dev, &props);
		gdr_support[dev] = is_gdr_supported_nic(props.ptrSupport);
	}
	return gdr_support;
}

inline test_nccl_net_t *get_extNet(void)
{
	void *netPluginLib = NULL;
	test_nccl_net_t *extNet = NULL;

	// Get system page size for memory allocation calculations
	auto system_page_size_sysconf = sysconf(_SC_PAGESIZE);
	if (OFI_UNLIKELY(system_page_size_sysconf <= 0)) {
		throw std::runtime_error("Failed to get system page size");
	}
	system_page_size = static_cast<size_t>(system_page_size_sysconf);

	netPluginLib = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (netPluginLib == NULL) {
		NCCL_OFI_WARN("Unable to load libnccl-net.so: %s", dlerror());
		return NULL;
	}

	extNet = (test_nccl_net_t *)dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
	if (extNet == NULL) {
		NCCL_OFI_WARN("NetPlugin, could not find %s symbol",
			      STR(NCCL_PLUGIN_SYMBOL));
	}

	return extNet;
}

/**
 * RAII wrapper for buffer allocation and registration
 */
class BufferHandle {
public:
	char* buffer = nullptr;
	void* mhandle = nullptr;
	size_t size = 0;
	int type = 0;
	void* comm = nullptr;
	test_nccl_net_t* ext_net = nullptr;

	BufferHandle() = default;

	BufferHandle(size_t sz, int tp, void* cm, test_nccl_net_t* net)
		: size(sz), type(tp), comm(cm), ext_net(net) {
		if (allocate_buff((void**)&buffer, size, type) != ncclSuccess) {
			throw std::runtime_error("BufferHandle: Failed to allocate buffer of size " + std::to_string(size));
		}
		if (ext_net->regMr(comm, buffer, size, type, &mhandle) != ncclSuccess) {
			deallocate_buffer(buffer, type);
			throw std::runtime_error("BufferHandle: Failed to register memory of size " + std::to_string(size));
		}
	}

	~BufferHandle() {
		if (mhandle && comm && ext_net) {
			ext_net->deregMr(comm, mhandle);
		}
		if (buffer) {
			deallocate_buffer(buffer, type);
		}
	}

	// Move semantics
	BufferHandle(BufferHandle&& other) noexcept
		: buffer(other.buffer), mhandle(other.mhandle), size(other.size),
		  type(other.type), comm(other.comm), ext_net(other.ext_net) {
		other.buffer = nullptr;
		other.mhandle = nullptr;
		other.comm = nullptr;
		other.ext_net = nullptr;
	}

	BufferHandle& operator=(BufferHandle&& other) noexcept {
		if (this != &other) {
			// Clean up existing resources
			if (mhandle && comm && ext_net) {
				ext_net->deregMr(comm, mhandle);
			}
			if (buffer) {
				deallocate_buffer(buffer, type);
			}

			// Move from other
			buffer = other.buffer;
			mhandle = other.mhandle;
			size = other.size;
			type = other.type;
			comm = other.comm;
			ext_net = other.ext_net;

			other.buffer = nullptr;
			other.mhandle = nullptr;
			other.comm = nullptr;
			other.ext_net = nullptr;
		}
		return *this;
	}

	// Delete copy
	BufferHandle(const BufferHandle&) = delete;
	BufferHandle& operator=(const BufferHandle&) = delete;
};

/**
 * Buffer set for a single test size
 */
struct TestBuffer {
	size_t send_size;
	size_t recv_size;
	int buffer_type;
	std::vector<BufferHandle> send_bufs;  // [req_idx]
	std::vector<BufferHandle> recv_bufs;  // [req_idx]
	bool validated;

	TestBuffer() : send_size(0), recv_size(0), buffer_type(0), validated(false) {}

	/**
	 * Allocate buffers for this test size
	 */
	ncclResult_t allocate(int rank, int num_requests, int buf_type, 
	                     void* send_comm, void* recv_comm, test_nccl_net_t* net) {
		buffer_type = buf_type;
		auto& bufs = (rank == 0) ? send_bufs : recv_bufs;
		void* comm = (rank == 0) ? send_comm : recv_comm;
		size_t size = (rank == 0) ? send_size : recv_size;

		bufs.reserve(num_requests);
		for (int i = 0; i < num_requests; i++) {
			bufs.emplace_back(size, buf_type, comm, net);
			// BufferHandle constructor throws on failure
		}
		return ncclSuccess;
	}
};

/**
 * Thread context structure for test scenarios
 */
struct ThreadContext {
	size_t thread_id;
	test_nccl_net_t* ext_net;
	ListenComms lcomms;
	SendComms scomms;
	RecvComms rcomms;
	ncclResult_t result;
	void* scenario;  // Pointer to TestScenario, using void* to avoid circular dependency
	MPI_Comm thread_comm;
	int rank;
	int peer_rank;
	int ndev;
	std::vector<test_nccl_net_device_handle_t*> shandles;
	std::vector<test_nccl_net_device_handle_t*> rhandles;

	// Device mapping: dev_idx → physical_dev
	std::vector<int> device_map;

	// Per-device, per-test-size buffer management
	std::vector<std::vector<TestBuffer>> test_buffers;  // [dev_idx][size_idx]

	ThreadContext(size_t tid, test_nccl_net_t* net, void* scen, MPI_Comm comm)
		: thread_id(tid), ext_net(net), result(ncclSuccess), scenario(scen), 
		  thread_comm(comm), rank(-1), peer_rank(-1), ndev(0) {}

	/**
	 * Initialize buffer storage after ndev is known
	 */
	void initialize_buffer_storage(const std::vector<std::pair<size_t, size_t>>& test_sizes) {
		test_buffers.resize(ndev);
		for (int dev = 0; dev < ndev; dev++) {
			test_buffers[dev].resize(test_sizes.size());
			for (size_t i = 0; i < test_sizes.size(); i++) {
				test_buffers[dev][i].send_size = test_sizes[i].first;
				test_buffers[dev][i].recv_size = test_sizes[i].second;
				// BufferHandles will be allocated later in allocate_test_buffers()
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "Rank %d: Initialized buffer storage for %d devices, %zu test sizes", rank, ndev, test_sizes.size());
	}

	/**
	 * Setup connection for a specific device using context's own fields
	 */
	ncclResult_t setup_connection(int dev_idx, int size)
	{
		// Get physical device from mapping
		int physical_dev = device_map[dev_idx];
		
		// Validate device index
		if (physical_dev < 0 || physical_dev >= ndev) {
			NCCL_OFI_WARN("Invalid physical device %d (ndev=%d)", physical_dev, ndev);
			return ncclInvalidArgument;
		}

		// Validate peer rank
		if (peer_rank < 0 || peer_rank >= size || peer_rank == rank) {
			NCCL_OFI_WARN("Invalid peer rank %d (rank=%d, size=%d)",
			peer_rank, rank, size);
			return ncclInvalidArgument;
		}

		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d setting up connection with rank %d on physical_dev=%d (dev_idx=%d)",
			rank, peer_rank, physical_dev, dev_idx);

		char local_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
		char peer_handle[NCCL_NET_HANDLE_MAXSIZE] = {};

		nccl_net_ofi_listen_comm_t* lcomm = nullptr;
		nccl_net_ofi_send_comm_t* scomm = nullptr;
		nccl_net_ofi_recv_comm_t* rcomm = nullptr;
		test_nccl_net_device_handle_t* shandle = nullptr;
		test_nccl_net_device_handle_t* rhandle = nullptr;

		// Create listen communicator
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Creating listen communicator on physical device %d",
			rank, physical_dev);
		OFINCCLCHECK(ext_net->listen(physical_dev, static_cast<void*>(&local_handle),
				      reinterpret_cast<void**>(&lcomm)));

		// Exchange connection handles via MPI
		MPI_Status status;
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Exchanging handles with rank %d",
			rank, peer_rank);
		auto mpi_ret = MPI_Sendrecv(
			local_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0,
			peer_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0,
			thread_comm, &status);
		if (mpi_ret != MPI_SUCCESS) {
			NCCL_OFI_WARN("MPI_Sendrecv failed with error %d", mpi_ret);
			return ncclSystemError;
		}
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Establishing send and receive communicators",
			rank);

		// Establish send and receive communicators
		while (scomm == nullptr || rcomm == nullptr) {
			if (scomm == nullptr) {
				OFINCCLCHECK(ext_net->connect(physical_dev, static_cast<void*>(peer_handle),
					 reinterpret_cast<void**>(&scomm), &shandle));
			}

			if (rcomm == nullptr) {
				OFINCCLCHECK(ext_net->accept(static_cast<void*>(lcomm),
					reinterpret_cast<void**>(&rcomm), &rhandle));
			}
		}

		// Store at dev_idx for consistent access across ranks
		lcomms[dev_idx] = lcomm;
		scomms[dev_idx] = scomm;
		rcomms[dev_idx] = rcomm;
		shandles[dev_idx] = shandle;
		rhandles[dev_idx] = rhandle;

		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Successfully established connection with rank %d on physical_dev=%d (stored at dev_idx=%d)",
			rank, peer_rank, physical_dev, dev_idx);

		return ncclSuccess;
	}

	/**
	 * Cleanup connection at specific index using context's own vectors
	 */
	ncclResult_t cleanup_connection(size_t idx)
	{
		if (idx < lcomms.size() && lcomms[idx] != nullptr) {
			OFINCCLCHECK(ext_net->closeListen(static_cast<void*>(lcomms[idx])));
			lcomms[idx] = nullptr;
		}

		if (idx < scomms.size() && scomms[idx] != nullptr) {
			OFINCCLCHECK(ext_net->closeSend(static_cast<void*>(scomms[idx])));
			scomms[idx] = nullptr;
		}

		if (idx < rcomms.size() && rcomms[idx] != nullptr) {
			OFINCCLCHECK(ext_net->closeRecv(static_cast<void*>(rcomms[idx])));
			rcomms[idx] = nullptr;
		}

		return ncclSuccess;
	}

	/**
	 * Allocate and register test buffers for all test sizes on a specific device
	 */
	ncclResult_t allocate_test_buffers(int dev_idx, int buffer_type)
	{
		NCCL_OFI_INFO(NCCL_NET, "Rank %d: Allocating test buffers for dev %d, %zu test sizes, type=%d", 
		              rank, dev_idx, test_buffers[dev_idx].size(), buffer_type);

		nccl_net_ofi_send_comm_t* sComm = scomms[dev_idx];
		nccl_net_ofi_recv_comm_t* rComm = rcomms[dev_idx];

		for (auto& tb : test_buffers[dev_idx]) {
			OFINCCLCHECK(tb.allocate(rank, NUM_REQUESTS, buffer_type, sComm, rComm, ext_net));
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank %d: Allocated buffers for %zu test sizes on dev %d", rank, test_buffers[dev_idx].size(), dev_idx);
		return ncclSuccess;
	}

	/**
	 * Deallocate and deregister test buffers for all test sizes
	 * BufferHandle destructors handle cleanup automatically
	 */
	ncclResult_t deallocate_test_buffers(int dev_idx)
	{
		NCCL_OFI_INFO(NCCL_NET, "Rank %d: Deallocating test buffers for dev %d", rank, dev_idx);

		// Clearing vectors calls BufferHandle destructors which handle deregMr and deallocate_buffer
		for (auto& tb : test_buffers[dev_idx]) {
			tb.send_bufs.clear();
			tb.recv_bufs.clear();
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank %d: Deallocated test buffers for dev %d", rank, dev_idx);
		return ncclSuccess;
	}

	/**
	 * Poll until all requests complete and validate data
	 */
	ncclResult_t poll_and_validate(std::array<void*, NUM_REQUESTS>& requests, char** recv_buf, void** mhandle, 
	                                 size_t send_size, size_t recv_size, int buffer_type, void* rComm)
	{
		NCCL_OFI_INFO(NCCL_NET, "Rank %d: Polling for completion", rank);
		
		bool all_done = false;
		int poll_count = 0;
		while (!all_done) {
			all_done = true;
			for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
				if (requests[req_idx] != nullptr) {
					int done = 0;
					OFINCCLCHECK(ext_net->test(requests[req_idx], &done, nullptr));
					if (done) {
						requests[req_idx] = nullptr;
						
						// Rank 1: flush immediately after completion
						if (rank == 1 && buffer_type == NCCL_PTR_CUDA) {
							void* iflush_req = nullptr;
							int sizes_int[1] = {(int)recv_size};
							OFINCCLCHECK(ext_net->iflush(rComm, 1, (void**)&recv_buf[req_idx], sizes_int, &mhandle[req_idx], &iflush_req));
							if (iflush_req) {
								int flush_done = 0;
								while (!flush_done) {
									OFINCCLCHECK(ext_net->test(iflush_req, &flush_done, nullptr));
								}
							}
						}
					} else {
						all_done = false;
					}
				}
			}
			poll_count++;
		}
		
		// Validate after all requests complete (rank 1 only)
		if (rank == 1 && !(buffer_type == NCCL_PTR_CUDA && ofi_nccl_gdr_flush_disable())) {
			size_t validate_size = std::min(send_size, recv_size);
			char* expected_buf = nullptr;
			OFINCCLCHECK(allocate_buff((void**)&expected_buf, validate_size, NCCL_PTR_HOST));
			OFINCCLCHECK(initialize_buff(expected_buf, validate_size, NCCL_PTR_HOST));
			for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
				OFINCCLCHECK(validate_data(recv_buf[req_idx], expected_buf, validate_size, buffer_type));
			}
			OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
		}
		
		NCCL_OFI_INFO(NCCL_NET, "Rank %d: All requests completed after %d polls", rank, poll_count);
		return ncclSuccess;
	}

	/**
	 * Test send/receive with fresh buffer allocation per call (like master)
	 */
	ncclResult_t send_receive_test(int dev_idx, size_t size_idx, size_t send_size, size_t recv_size)
	{
		static constexpr int TAG = 1;
		static constexpr int NRECV = NCCL_OFI_MAX_RECVS;

		nccl_net_ofi_send_comm_t* sComm = scomms[dev_idx];
		nccl_net_ofi_recv_comm_t* rComm = rcomms[dev_idx];
		
		// Determine buffer type based on GDR support
		auto gdr_support = get_support_gdr(ext_net);
		int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		// Local buffer arrays
		char* send_buf[NUM_REQUESTS] = {nullptr};
		char* recv_buf[NUM_REQUESTS] = {nullptr};
		void* mhandle[NUM_REQUESTS] = {nullptr};
		std::array<void*, NUM_REQUESTS> requests{};

		if (rank == 0) {
			// Allocate and post sends
			for (int idx = 0; idx < NUM_REQUESTS; idx++) {
				OFINCCLCHECK(allocate_buff((void**)&send_buf[idx], send_size, buffer_type));
				OFINCCLCHECK(initialize_buff(send_buf[idx], send_size, buffer_type));
				OFINCCLCHECK(ext_net->regMr(sComm, send_buf[idx], send_size, buffer_type, &mhandle[idx]));
				OFINCCLCHECK(post_send(ext_net, sComm, send_buf[idx], send_size, TAG, mhandle[idx], &requests[idx]));
			}
		} else {
			// Allocate and post receives
			std::array<size_t, NRECV> sizes;
			std::array<int, NRECV> tags;
			std::fill(sizes.begin(), sizes.end(), recv_size);
			std::fill(tags.begin(), tags.end(), TAG);

			for (int idx = 0; idx < NUM_REQUESTS; idx++) {
				OFINCCLCHECK(allocate_buff((void**)&recv_buf[idx], recv_size, buffer_type));
				OFINCCLCHECK(ext_net->regMr(rComm, recv_buf[idx], recv_size, buffer_type, &mhandle[idx]));
				OFINCCLCHECK(post_recv(ext_net, rComm, NRECV, (void**)&recv_buf[idx], sizes.data(), tags.data(), &mhandle[idx], &requests[idx]));
			}
		}

		// Poll for completion with validation
		OFINCCLCHECK(poll_and_validate(requests, recv_buf, mhandle, send_size, recv_size, buffer_type, rComm));

		// Cleanup
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			if (mhandle[idx]) {
				if (rank == 0) ext_net->deregMr(sComm, mhandle[idx]);
				else ext_net->deregMr(rComm, mhandle[idx]);
			}
			if (send_buf[idx]) deallocate_buffer(send_buf[idx], buffer_type);
			if (recv_buf[idx]) deallocate_buffer(recv_buf[idx], buffer_type);
		}

		return ncclSuccess;
	}

	/**
	 * Flush all receive buffers for a device to ensure data visibility
	 */
	ncclResult_t flush_all_buffers(int dev_idx)
	{
		if (rank != 1) {
			return ncclSuccess;  // Only receiver needs to flush
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank 1: Flushing all buffers for dev %d", dev_idx);
		nccl_net_ofi_recv_comm_t* rComm = rcomms[dev_idx];

		for (size_t size_idx = 0; size_idx < test_buffers[dev_idx].size(); size_idx++) {
			TestBuffer& tb = test_buffers[dev_idx][size_idx];
			
			if (tb.buffer_type == NCCL_PTR_CUDA) {
				for (size_t req_idx = 0; req_idx < tb.recv_bufs.size(); req_idx++) {
					void* iflush_req = nullptr;
					int sizes_int[1] = {(int)tb.recv_size};
					OFINCCLCHECK(ext_net->iflush(rComm, 1, (void**)&tb.recv_bufs[req_idx].buffer, sizes_int, &tb.recv_bufs[req_idx].mhandle, &iflush_req));
					if (iflush_req) {
						int done = 0;
						while (!done) {
							OFINCCLCHECK(ext_net->test(iflush_req, &done, nullptr));
						}
					}
				}
			}
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank 1: Flushed all buffers for dev %d", dev_idx);
		return ncclSuccess;
	}

	/**
	 * Validate all test buffers after all tests complete
	 */
	ncclResult_t validate_all_tests(int dev_idx)
	{
		if (rank != 1) {
			return ncclSuccess;  // Only receiver validates
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank 1: Validating all test buffers for dev %d", dev_idx);

		for (size_t size_idx = 0; size_idx < test_buffers[dev_idx].size(); size_idx++) {
			TestBuffer& tb = test_buffers[dev_idx][size_idx];
			
			if (tb.validated) {
				continue;  // Already validated
			}

			// Validate data
			if (!(tb.buffer_type == NCCL_PTR_CUDA && ofi_nccl_gdr_flush_disable())) {
				// Validate the actual data sent (min of send_size and recv_size)
				size_t validate_size = std::min(tb.send_size, tb.recv_size);
				char* expected_buf = nullptr;
				OFINCCLCHECK(allocate_buff((void**)&expected_buf, validate_size, NCCL_PTR_HOST));
				OFINCCLCHECK(initialize_buff(expected_buf, validate_size, NCCL_PTR_HOST));
				for (size_t req_idx = 0; req_idx < tb.recv_bufs.size(); req_idx++) {
					NCCL_OFI_INFO(NCCL_NET, "Rank 1: Validating buffer %zu for dev %d size_idx %zu (send_size=%zu, recv_size=%zu, validate_size=%zu)", 
					              req_idx, dev_idx, size_idx, tb.send_size, tb.recv_size, validate_size);
					OFINCCLCHECK(validate_data(tb.recv_bufs[req_idx].buffer, expected_buf, validate_size, tb.buffer_type));
				}
				OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
			}

			tb.validated = true;
			NCCL_OFI_INFO(NCCL_NET, "Rank 1: Validation passed for dev %d size_idx %zu", dev_idx, size_idx);
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank 1: All validations passed for dev %d", dev_idx);
		return ncclSuccess;
	}
};

/**
 * Simple test scenario class
 */
class TestScenario {
public:
	TestScenario(std::string&& scenario_name, size_t num_threads_per_process = 0, size_t num_iterations = 1)
	: name(std::move(scenario_name)), iterations(num_iterations) {
		threads.resize(num_threads_per_process);
		// thread_contexts will be populated in execute() via emplace_back
	}

	virtual ~TestScenario() = default;

	void set_ext_net(test_nccl_net_t* net) { ext_net = net; }

	virtual ncclResult_t setup(ThreadContext& ctx) {
		// Get rank from thread communicator
		MPI_Comm_rank(ctx.thread_comm, &ctx.rank);
		
		// Calculate peer rank (assuming 2-rank test)
		ctx.peer_rank = (ctx.rank == 0) ? 1 : 0;
		
		// Get number of devices
		OFINCCLCHECK(ext_net->devices(&ctx.ndev));
		
		// Initialize device mapping
		// Rank 1 uses devices in reverse order to avoid contention
		ctx.device_map.resize(ctx.ndev);
		for (int dev_idx = 0; dev_idx < ctx.ndev; dev_idx++) {
			ctx.device_map[dev_idx] = (ctx.rank == 1) ? ctx.ndev - dev_idx - 1 : dev_idx;
		}
		
		// Resize connection vectors
		ctx.lcomms.resize(ctx.ndev, nullptr);
		ctx.scomms.resize(ctx.ndev, nullptr);
		ctx.rcomms.resize(ctx.ndev, nullptr);
		ctx.shandles.resize(ctx.ndev, nullptr);
		ctx.rhandles.resize(ctx.ndev, nullptr);
		
		// Setup connections for all devices
		for (int dev_idx = 0; dev_idx < ctx.ndev; dev_idx++) {
			OFINCCLCHECK(ctx.setup_connection(dev_idx, 2));
		}
		
		return ncclSuccess;
	}

	virtual ncclResult_t run(ThreadContext& ctx) = 0;

	virtual ncclResult_t teardown(ThreadContext& ctx) {
		// Cleanup all connections
		for (size_t i = 0; i < ctx.lcomms.size(); i++) {
			OFINCCLCHECK(ctx.cleanup_connection(i));
		}
		return ncclSuccess;
	}

	ncclResult_t execute() {
		NCCL_OFI_INFO(NCCL_NET, "Running: %s", this->name.c_str());

		// Clear any stale contexts from previous test runs
		thread_contexts.clear();

		// Single-threaded: create context and execute on main thread
		if (threads.size() == 0) {
			thread_contexts.emplace_back(0, ext_net, this, MPI_COMM_WORLD);

			for (size_t iter = 0; iter < iterations; iter++) {
				OFINCCLCHECK(setup(thread_contexts[0]));
				OFINCCLCHECK(run(thread_contexts[0]));
				OFINCCLCHECK(teardown(thread_contexts[0]));
				MPI_Barrier(MPI_COMM_WORLD);
			}
			return thread_contexts[0].result;
		}

		// Execute on pthreads
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		for (size_t i = 0; i < threads.size(); i++) {
			MPI_Comm comm;
			MPI_Comm_split(MPI_COMM_WORLD, i, rank, &comm);
			thread_contexts.emplace_back(i, ext_net, this, comm);
		}

		// Create threads
		for (size_t i = 0; i < threads.size(); i++) {
			int ret = pthread_create(&threads[i], nullptr,
			    thread_function, &thread_contexts[i]);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to create thread %zu: %s", i, strerror(ret));

				// Clean up threads that were created
				for (size_t j = 0; j < i; j++) {
					pthread_join(threads[j], nullptr);
				}
				return ncclSystemError;
			}
		}

		// Wait for all threads to complete
		ncclResult_t final_result = ncclSuccess;
		for (size_t i = 0; i < threads.size(); i++) {
			int ret = pthread_join(threads[i], nullptr);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to join thread %zu: %s", i, strerror(ret));
				final_result = ncclSystemError;
			}

			// Check thread execution result
			if (thread_contexts[i].result != ncclSuccess) {
				NCCL_OFI_WARN("Thread %zu failed with result %d", i, thread_contexts[i].result);
				final_result = thread_contexts[i].result;
			}
		}

		if (final_result != ncclSuccess) {
			return final_result;
		}

		return ncclSuccess;
	}

protected:
	static void* thread_function(void* arg) {
		ThreadContext* ctx = static_cast<ThreadContext*>(arg);
		TestScenario* scenario = static_cast<TestScenario*>(ctx->scenario);

		ncclResult_t result = ncclSuccess;

		// Execute iterations
		for (size_t iter = 0; iter < scenario->iterations; iter++) {
			// Execute the sequence - continue to teardown even if setup/run fails
			if ((result = scenario->setup(*ctx)) != ncclSuccess) {
				NCCL_OFI_WARN("Thread %zu setup failed on iteration %zu", ctx->thread_id, iter);
				ctx->result = result;
			}

			if (result == ncclSuccess && (result = scenario->run(*ctx)) != ncclSuccess) {
				NCCL_OFI_WARN("Thread %zu run failed on iteration %zu", ctx->thread_id, iter);
				ctx->result = result;
			}

			// Always call teardown to ensure cleanup happens
			ncclResult_t teardown_result = scenario->teardown(*ctx);
			if (teardown_result != ncclSuccess) {
				NCCL_OFI_WARN("Thread %zu teardown failed on iteration %zu", ctx->thread_id, iter);
				if (result == ncclSuccess) {
					ctx->result = teardown_result;
				}
			}

			// Barrier between iterations
			MPI_Barrier(ctx->thread_comm);

			// If any step failed, break out of iteration loop
			if (result != ncclSuccess) {
				break;
			}
		}

		// Free thread communicator after all iterations complete
		if (ctx->thread_comm != MPI_COMM_WORLD) {
			MPI_Comm_free(&ctx->thread_comm);
		}

		// Set final result if not already set
		if (ctx->result == ncclSuccess) {
			ctx->result = result;
		}

		return nullptr;
	}

	test_nccl_net_t* ext_net = nullptr;
	std::string name;
	std::vector<ThreadContext> thread_contexts;
	std::vector<pthread_t> threads;
	size_t iterations;

};

/**
 * Test registry for collecting test scenarios
 */
class TestSuite {
public:
	TestSuite() {
		// Get system page size for memory allocation calculations
		auto system_page_size_sysconf = sysconf(_SC_PAGESIZE);
		if (OFI_UNLIKELY(system_page_size_sysconf <= 0)) {
			throw std::runtime_error("Failed to get system page size");
		}
		system_page_size = static_cast<size_t>(system_page_size_sysconf);

		net_plugin_handle = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
		if (net_plugin_handle == nullptr) {
			throw std::runtime_error(std::string("Unable to load libnccl-net.so: ") + dlerror());
		}

		ext_net = static_cast<test_nccl_net_t*>(
			dlsym(net_plugin_handle, STR(NCCL_PLUGIN_SYMBOL)));
		if (ext_net == nullptr) {
			throw std::runtime_error("Could not find symbol");
		}
	}

	~TestSuite() {
		if (net_plugin_handle != nullptr) {
			dlclose(net_plugin_handle);
		}
	}

	void add(TestScenario* scenario) {
		tests.push_back(scenario);
	}

	ncclResult_t setup() {
		int rank, local_rank, size;
		OFINCCLCHECK(mpi_init_ranks(&rank, &size, &local_rank));
		if (size != expected_size) {
			throw std::runtime_error("Invalid rank count");
		}
		return ncclSuccess;
	}

	ncclResult_t teardown() {
		int ret = MPI_Finalize();
		if (ret != MPI_SUCCESS) {
			NCCL_OFI_WARN("MPI_Finalize failed: %d", ret);
			return ncclSystemError;
		}
		return ncclSuccess;
	}

	ncclResult_t run_all() {
		OFINCCLCHECK(setup());
		OFINCCLCHECK(ext_net->init(&logger));

		int passed = 0;
		for (const auto& test : tests) {
			test->set_ext_net(ext_net);
			if (test->execute() == ncclSuccess) {
				passed++;
			}
			// Ensure all ranks complete test before starting next one
			MPI_Barrier(MPI_COMM_WORLD);
		}

		OFINCCLCHECK(teardown());

		NCCL_OFI_INFO(NCCL_NET, "Results: %d/%zu passed", passed, tests.size());
		return (passed == static_cast<int>(tests.size())) ? ncclSuccess : ncclSystemError;
	}

private:
	static constexpr size_t expected_size = 2;
	void* net_plugin_handle = nullptr;
	test_nccl_net_t* ext_net = nullptr;
	std::vector<TestScenario*> tests;
};

#endif // End TEST_COMMON_H_
