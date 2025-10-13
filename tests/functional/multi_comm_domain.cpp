/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * Multi-communicator domain test with threading support.
 * Tests concurrent communicator creation and operations within shared domains.
 */

#include "config.h"
#include "test-common.h"
#include <pthread.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <dlfcn.h>

#if HAVE_CUDA
#include <cuda_runtime.h>

// Global variables for CUDA initialization (calculated once, used by all threads)
static int g_local_rank = -1;
static int g_cuda_device = -1;

// Calculate local rank once (thread-safe, called before threading starts)
static bool calculate_local_rank_and_cuda_device() {
	int rank, num_ranks, local_rank = 0;
	char name[MPI_MAX_PROCESSOR_NAME];
	int proc_name_len;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	MPI_Get_processor_name(name, &proc_name_len);

	// Get all processor names to determine local rank
	char *all_proc_name = (char *)malloc(sizeof(char) * num_ranks * MPI_MAX_PROCESSOR_NAME);
	if (!all_proc_name) {
		fprintf(stderr, "Failed to allocate memory for processor names\n");
		return false;
	}

	MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_proc_name,
		MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

	// Determine local rank (same logic as nccl_message_transfer.cpp)
	for (int i = 0; i < num_ranks; i++) {
		if (!strcmp(name, &all_proc_name[i * MPI_MAX_PROCESSOR_NAME])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}
	free(all_proc_name);

	// Query CUDA device count
	int device_count = 0;
	cudaError_t result = cudaGetDeviceCount(&device_count);
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA runtime initialization failed: %s\n", cudaGetErrorString(result));
		return false;
	}

	if (device_count == 0) {
		fprintf(stderr, "No CUDA devices found\n");
		return false;
	}

	// Store global values
	g_local_rank = local_rank;
	g_cuda_device = local_rank % device_count;

	NCCL_OFI_INFO(NCCL_INIT, "Calculated CUDA device %d for memory allocation (local_rank=%d)", g_cuda_device, g_local_rank);
	return true;
}

// Initialize CUDA runtime for worker thread (thread-safe, per-thread context)
static bool initialize_cuda_for_thread() {
	// Each thread needs its own CUDA context, so don't use global flag
	// Set device based on pre-calculated values
	cudaError_t result = cudaSetDevice(g_cuda_device);
	if (result != cudaSuccess) {
		fprintf(stderr, "Failed to set CUDA device %d: %s\n", g_cuda_device, cudaGetErrorString(result));
		return false;
	}

	// Force context creation by doing a simple CUDA operation
	void* dummy_ptr = nullptr;
	result = cudaMalloc(&dummy_ptr, 1);
	if (result == cudaSuccess && dummy_ptr) {
		cudaFree(dummy_ptr);
	} else {
		fprintf(stderr, "Failed to create CUDA context on device %d: %s\n", g_cuda_device, cudaGetErrorString(result));
		return false;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Using CUDA device %d for memory allocation (local_rank=%d)", g_cuda_device, g_local_rank);
	return true;
}
#endif

// Default parameters - can be overridden via command line or environment
static size_t MAX_TEST_COMMS = 4;
static size_t MAX_TEST_THREADS = 3;
static bool FORCE_HOST_BUFFERS = false;

class MultiCommContext {
private:
	std::vector<nccl_net_ofi_send_comm_t*> send_comms;
	std::vector<nccl_net_ofi_recv_comm_t*> recv_comms;
	std::vector<nccl_net_ofi_listen_comm_t*> listen_comms;
	std::vector<test_nccl_net_device_handle_t*> send_handles;
	std::vector<test_nccl_net_device_handle_t*> recv_handles;
	std::vector<std::array<char, NCCL_NET_HANDLE_MAXSIZE>> src_handles;
	std::vector<std::array<char, NCCL_NET_HANDLE_MAXSIZE>> handles;
	size_t active_comm_count{0};
	int device_id{0};

public:
	MultiCommContext()
		: send_comms(MAX_TEST_COMMS, nullptr)
		, recv_comms(MAX_TEST_COMMS, nullptr)
		, listen_comms(MAX_TEST_COMMS, nullptr)
		, send_handles(MAX_TEST_COMMS, nullptr)
		, recv_handles(MAX_TEST_COMMS, nullptr)
		, src_handles(MAX_TEST_COMMS)
		, handles(MAX_TEST_COMMS)
	{}

	ncclResult_t create_communicator_pair(test_nccl_net_t* ext_net, size_t comm_idx,
		int rank, int dev, int thread_id) {
		// Fixed: Ensure unique tags across all threads and communicators
		int base_tag = (thread_id * static_cast<int>(MAX_TEST_COMMS) * 2) + (static_cast<int>(comm_idx) * 2);

		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: create_communicator_pair comm %zu, dev %d, base_tag %d",
			rank, thread_id, comm_idx, dev, base_tag);

		// Listen
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Calling ext_net->listen for comm %zu", rank, thread_id, comm_idx);
		ncclResult_t listen_result = ext_net->listen(dev, static_cast<void*>(&handles[comm_idx]),
			reinterpret_cast<void**>(&listen_comms[comm_idx]));
		if (listen_result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d THREAD %d: Failed to create listen communicator for comm %zu on device %d: %d",
				rank, thread_id, comm_idx, dev, listen_result);
			return ncclInternalError;
		}
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Successfully created listen communicator for comm %zu", rank, thread_id, comm_idx);

		// Exchange handles
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Starting handle exchange for comm %zu", rank, thread_id, comm_idx);
		if (rank == 0) {
			NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Sending handle for comm %zu", rank, thread_id, comm_idx);
			MPI_Send(&handles[comm_idx], NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 1, base_tag, MPI_COMM_WORLD);
			NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Receiving handle for comm %zu", rank, thread_id, comm_idx);
			MPI_Recv(src_handles[comm_idx].data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 1, base_tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Receiving handle for comm %zu", rank, thread_id, comm_idx);
			MPI_Recv(src_handles[comm_idx].data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 0, base_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Sending handle for comm %zu", rank, thread_id, comm_idx);
			MPI_Send(handles[comm_idx].data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 0, base_tag + 1, MPI_COMM_WORLD);
		}
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Handle exchange completed for comm %zu", rank, thread_id, comm_idx);

		// Connect and Accept
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Starting connect/accept for comm %zu", rank, thread_id, comm_idx);
		while (!send_comms[comm_idx] || !recv_comms[comm_idx]) {
			if (!send_comms[comm_idx]) {
				ext_net->connect(dev, static_cast<void*>(src_handles[comm_idx].data()),
					reinterpret_cast<void**>(&send_comms[comm_idx]), &send_handles[comm_idx]);
			}
			if (!recv_comms[comm_idx]) {
				ext_net->accept(static_cast<void*>(listen_comms[comm_idx]),
					reinterpret_cast<void**>(&recv_comms[comm_idx]), &recv_handles[comm_idx]);
			}
		}
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Connect/accept completed for comm %zu", rank, thread_id, comm_idx);

		active_comm_count++;
		return ncclSuccess;
	}

	ncclResult_t validate_domain_sharing() const {
		// For single communicator, just validate it exists and is properly configured
		if (active_comm_count == 0) {
			NCCL_OFI_WARN("No active communicators to validate");
			return ncclInternalError;
		}

		for (size_t i = 0; i < active_comm_count; i++) {
			if (!send_comms[i] || !recv_comms[i]) {
				NCCL_OFI_WARN("Communicator %zu is null (send: %p, recv: %p)", i,
					static_cast<void*>(send_comms[i]), static_cast<void*>(recv_comms[i]));
				return ncclInternalError;
			}
			if (send_comms[i]->base.dev_id != device_id ||
			    recv_comms[i]->base.dev_id != device_id) {
				NCCL_OFI_WARN("Domain validation failed for comm %zu (expected dev %d, got send dev %d, recv dev %d)",
					i, device_id, send_comms[i]->base.dev_id, recv_comms[i]->base.dev_id);
				return ncclInternalError;
			}
		}

		// For multiple communicators, validate they share the same domain
		if (active_comm_count >= 2) {
			NCCL_OFI_INFO(NCCL_NET, "Validating domain sharing across %zu communicators", active_comm_count);
		}

		return ncclSuccess;
	}

	ncclResult_t test_concurrent_operations(test_nccl_net_t* ext_net, int buffer_type, int rank) const {
		if (!ext_net || active_comm_count == 0) return ncclSuccess;

		// Add additional safety checks
		for (size_t i = 0; i < active_comm_count; i++) {
			if (!send_comms[i] || !recv_comms[i]) {
				NCCL_OFI_WARN("RANK %d: Communicator %zu is null before operations (send: %p, recv: %p)",
					rank, i, static_cast<void*>(send_comms[i]), static_cast<void*>(recv_comms[i]));
				return ncclInternalError;
			}
		}

		constexpr size_t test_size = SEND_SIZE;
		std::vector<void*> send_buffers(MAX_TEST_COMMS, nullptr);
		std::vector<void*> recv_buffers(MAX_TEST_COMMS, nullptr);
		std::vector<void*> send_mhandles(MAX_TEST_COMMS, nullptr);
		std::vector<void*> recv_mhandles(MAX_TEST_COMMS, nullptr);
		std::vector<nccl_net_ofi_req_t*> send_reqs(MAX_TEST_COMMS, nullptr);
		std::vector<nccl_net_ofi_req_t*> recv_reqs(MAX_TEST_COMMS, nullptr);

		// Allocate and register buffers (following nccl_message_transfer.cpp pattern)
		for (size_t i = 0; i < active_comm_count; i++) {
			if (allocate_buff(&send_buffers[i], test_size, buffer_type) != ncclSuccess ||
			    initialize_buff(send_buffers[i], test_size, buffer_type) != ncclSuccess ||
			    allocate_buff(&recv_buffers[i], test_size, buffer_type) != ncclSuccess) {
				for (size_t j = 0; j <= i; j++) {
					if (send_buffers[j]) deallocate_buffer(send_buffers[j], buffer_type);
					if (recv_buffers[j]) deallocate_buffer(recv_buffers[j], buffer_type);
				}
				return ncclInternalError;
			}

			// Register memory handles like in nccl_message_transfer.cpp
			if (ext_net->regMr(static_cast<void*>(send_comms[i]), send_buffers[i], test_size, buffer_type, &send_mhandles[i]) != ncclSuccess ||
			    ext_net->regMr(static_cast<void*>(recv_comms[i]), recv_buffers[i], test_size, buffer_type, &recv_mhandles[i]) != ncclSuccess) {
				NCCL_OFI_WARN("RANK %d: Failed to register memory for comm %zu", rank, i);
				// Cleanup allocated buffers and registered handles
				for (size_t j = 0; j <= i; j++) {
					if (send_mhandles[j]) ext_net->deregMr(static_cast<void*>(send_comms[j]), send_mhandles[j]);
					if (recv_mhandles[j]) ext_net->deregMr(static_cast<void*>(recv_comms[j]), recv_mhandles[j]);
					if (send_buffers[j]) deallocate_buffer(send_buffers[j], buffer_type);
					if (recv_buffers[j]) deallocate_buffer(recv_buffers[j], buffer_type);
				}
				return ncclInternalError;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// RAII-style cleanup using lambda
		auto cleanup_buffers = [&]() {
			for (size_t i = 0; i < active_comm_count; i++) {
				if (send_mhandles[i]) ext_net->deregMr(static_cast<void*>(send_comms[i]), send_mhandles[i]);
				if (recv_mhandles[i]) ext_net->deregMr(static_cast<void*>(recv_comms[i]), recv_mhandles[i]);
				if (send_buffers[i]) deallocate_buffer(send_buffers[i], buffer_type);
				if (recv_buffers[i]) deallocate_buffer(recv_buffers[i], buffer_type);
			}
		};

		bool operations_successful = true;

		// Post operations with proper MR handles
		for (size_t i = 0; i < active_comm_count && operations_successful; i++) {
			// Double-check communicators are still valid
			if (!recv_comms[i] || !send_comms[i]) {
				fprintf(stderr, "RANK %d: Communicator %zu became null during operations (recv: %p, send: %p)\n",
					rank, i, static_cast<void*>(recv_comms[i]), static_cast<void*>(send_comms[i]));
				operations_successful = false;
				break;
			}

			// Check function pointers are valid before calling
			if (!recv_comms[i]->recv || !send_comms[i]->send) {
				NCCL_OFI_WARN("RANK %d: Function pointers null for comm %zu (recv: %s, send: %s)",
					rank, i, recv_comms[i]->recv ? "valid" : "null", send_comms[i]->send ? "valid" : "null");
				operations_successful = false;
				break;
			}

			// Use proper recv API like in nccl_message_transfer.cpp
			int nrecv = 1;
			void* recv_data[1] = {recv_buffers[i]};
			size_t recv_sizes[1] = {test_size};
			int recv_tags[1] = {static_cast<int>(i)};

			// Call recv operation with proper MR handle
			int recv_result = ext_net->irecv(static_cast<void*>(recv_comms[i]), nrecv, recv_data, recv_sizes,
				recv_tags, &recv_mhandles[i], reinterpret_cast<void**>(&recv_reqs[i]));
			if (recv_result != 0) {
				NCCL_OFI_WARN("RANK %d: irecv operation failed for comm %zu: %d", rank, i, recv_result);
				operations_successful = false;
				break;
			}

			// Call send operation with proper MR handle
			int send_result = ext_net->isend(static_cast<void*>(send_comms[i]), send_buffers[i], test_size, static_cast<int>(i),
				send_mhandles[i], reinterpret_cast<void**>(&send_reqs[i]));
			if (send_result != 0) {
				NCCL_OFI_WARN("RANK %d: isend operation failed for comm %zu: %d", rank, i, send_result);
				operations_successful = false;
				break;
			}

			// Verify requests were created
			if (!send_reqs[i] || !recv_reqs[i]) {
				NCCL_OFI_WARN("RANK %d: Request creation failed for comm %zu (send_req: %p, recv_req: %p)",
					rank, i, static_cast<void*>(send_reqs[i]), static_cast<void*>(recv_reqs[i]));
				operations_successful = false;
				break;
			}
		}

		bool all_complete = false;

		// Wait for completion only if operations were successful
		if (operations_successful) {
			for (int iteration = 0; !all_complete && iteration < 1000; iteration++) {
				all_complete = true;

				for (size_t i = 0; i < active_comm_count; i++) {
					// Enhanced safety checks for request testing
					if (send_reqs[i]) {
						if (!send_reqs[i]->test) {
							NCCL_OFI_WARN("RANK %d: send_reqs[%zu]->test is null", rank, i);
							operations_successful = false;
							all_complete = false;
							break;
						}
						int done = 0, size = 0;
						if (ext_net->test(static_cast<void*>(send_reqs[i]), &done, &size) == 0 && done) {
							send_reqs[i] = nullptr;
						} else {
							all_complete = false;
						}
					}

					if (recv_reqs[i]) {
						if (!recv_reqs[i]->test) {
							NCCL_OFI_WARN("RANK %d: recv_reqs[%zu]->test is null", rank, i);
							operations_successful = false;
							all_complete = false;
							break;
						}
						int done = 0, size = 0;
						if (ext_net->test(static_cast<void*>(recv_reqs[i]), &done, &size) == 0 && done) {
							recv_reqs[i] = nullptr;
						} else {
							all_complete = false;
						}
					}
				}

				if (!all_complete && operations_successful) {
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
				}
			}
		}

		cleanup_buffers();
		return (operations_successful && all_complete) ? ncclSuccess : ncclInternalError;
	}

	void cleanup(test_nccl_net_t* ext_net) {
		if (!ext_net) return;

		for (size_t i = 0; i < active_comm_count; i++) {
			if (listen_comms[i]) ext_net->closeListen(static_cast<void*>(listen_comms[i]));
			if (send_comms[i]) ext_net->closeSend(static_cast<void*>(send_comms[i]));
			if (recv_comms[i]) ext_net->closeRecv(static_cast<void*>(recv_comms[i]));
		}
		active_comm_count = 0;
	}

	void set_device_id(int dev_id) { device_id = dev_id; }
	size_t get_active_comm_count() const { return active_comm_count; }
};

class ThreadContext {
private:
	int thread_id{0}, rank{0}, device_id{0}, buffer_type{0};
	test_nccl_net_t* ext_net{nullptr};
	MultiCommContext comm_ctx{};
	std::atomic<bool> ready{false}, start_test{false}, completed{false};
	pthread_barrier_t* setup_barrier{nullptr}, *test_barrier{nullptr}, *cleanup_barrier{nullptr};
	ncclResult_t result{ncclSuccess};

public:
	ThreadContext() = default;

	// Initialization
	void initialize(int tid, int r, int dev_id, int buf_type, test_nccl_net_t* net,
		pthread_barrier_t* setup_bar, pthread_barrier_t* test_bar, pthread_barrier_t* cleanup_bar) {
		thread_id = tid;
		rank = r;
		device_id = dev_id;
		buffer_type = buf_type;
		ext_net = net;
		setup_barrier = setup_bar;
		test_barrier = test_bar;
		cleanup_barrier = cleanup_bar;
		result = ncclSuccess;
	}

	// Accessors
	int get_thread_id() const { return thread_id; }
	int get_rank() const { return rank; }
	int get_device_id() const { return device_id; }
	int get_buffer_type() const { return buffer_type; }
	test_nccl_net_t* get_ext_net() const { return ext_net; }
	MultiCommContext& get_comm_ctx() { return comm_ctx; }
	ncclResult_t get_result() const { return result; }

	// State management
	void set_ready(bool state) { ready.store(state); }
	bool is_ready() const { return ready.load(); }
	void set_start_test(bool state) { start_test.store(state); }
	bool should_start_test() const { return start_test.load(); }
	void set_completed(bool state) { completed.store(state); }
	bool is_completed() const { return completed.load(); }
	void set_result(ncclResult_t res) { result = res; }

	// Barrier operations
	void wait_setup_barrier() { if (setup_barrier) pthread_barrier_wait(setup_barrier); }
	void wait_test_barrier() { if (test_barrier) pthread_barrier_wait(test_barrier); }
	void wait_cleanup_barrier() { if (cleanup_barrier) pthread_barrier_wait(cleanup_barrier); }
};



static void* thread_worker(void* arg)
{
	auto* thread_ctx = static_cast<ThreadContext*>(arg);
	int rank = thread_ctx->get_rank();
	int thread_id = thread_ctx->get_thread_id();

	NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Starting thread_worker", rank, thread_id);

#if HAVE_CUDA
	// Initialize CUDA runtime for this worker thread
	bool cuda_initialized = initialize_cuda_for_thread();
	if (cuda_initialized) {
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: CUDA runtime initialized successfully", rank, thread_id);
	} else {
		NCCL_OFI_WARN("RANK %d THREAD %d: Failed to initialize CUDA runtime", rank, thread_id);
	}
#endif

	thread_ctx->get_comm_ctx().set_device_id(thread_ctx->get_device_id());

	// Create communicators
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Creating %zu communicators", rank, thread_id, MAX_TEST_COMMS);
	for (size_t comm_idx = 0; comm_idx < MAX_TEST_COMMS; comm_idx++) {
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Creating communicator %zu", rank, thread_id, comm_idx);

		if (thread_ctx->get_comm_ctx().create_communicator_pair(thread_ctx->get_ext_net(), comm_idx,
			thread_ctx->get_rank(), thread_ctx->get_device_id(), thread_ctx->get_thread_id()) != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d THREAD %d: Failed to create communicator %zu", rank, thread_id, comm_idx);
			thread_ctx->set_result(ncclInternalError);
			thread_ctx->get_comm_ctx().cleanup(thread_ctx->get_ext_net());
			thread_ctx->set_completed(true);
			return nullptr;
		}

		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Successfully created communicator %zu, waiting on barrier", rank, thread_id, comm_idx);
		thread_ctx->wait_setup_barrier();
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Passed barrier for communicator %zu", rank, thread_id, comm_idx);
	}

	// Signal ready and wait for start
	thread_ctx->set_ready(true);
	while (!thread_ctx->should_start_test()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	// Run tests with additional safety checks
	ncclResult_t result = ncclSuccess;

	NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Starting validation and testing phase", rank, thread_id);

	// Add a small delay to ensure all threads are synchronized
	std::this_thread::sleep_for(std::chrono::milliseconds(10));

	result = thread_ctx->get_comm_ctx().validate_domain_sharing();
	if (result != ncclSuccess) {
		NCCL_OFI_WARN("RANK %d THREAD %d: Domain validation failed: %d", rank, thread_id, result);
		thread_ctx->set_result(result);
	} else {
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Domain validation passed", rank, thread_id);

		result = thread_ctx->get_comm_ctx().test_concurrent_operations(thread_ctx->get_ext_net(),
			thread_ctx->get_buffer_type(), thread_ctx->get_rank());
		if (result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d THREAD %d: Concurrent operations test failed: %d", rank, thread_id, result);
			thread_ctx->set_result(result);
		} else {
			NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Concurrent operations test passed", rank, thread_id);
		}
	}

	thread_ctx->wait_test_barrier();

	// Cleanup
	thread_ctx->get_comm_ctx().cleanup(thread_ctx->get_ext_net());

	thread_ctx->wait_cleanup_barrier();

#if HAVE_CUDA
	// CUDA runtime cleanup is automatic when thread exits
	if (cuda_initialized) {
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: CUDA runtime cleanup handled automatically", rank, thread_id);
	}
#endif

	thread_ctx->set_completed(true);
	return nullptr;
}

static ncclResult_t test_multithreaded_communicators(test_nccl_net_t* ext_net, int device_id,
	int buffer_type, int rank)
{
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Starting test_multithreaded_communicators with device %d, buffer_type %d, threads %zu, comms %zu",
		rank, device_id, buffer_type, MAX_TEST_THREADS, MAX_TEST_COMMS);

	std::vector<pthread_t> threads(MAX_TEST_THREADS);
	std::vector<ThreadContext> thread_contexts(MAX_TEST_THREADS);
	pthread_barrier_t setup_barrier, test_barrier, cleanup_barrier;

	// Initialize barriers - but only if we have multiple threads
	// pthread_barrier with count=1 will deadlock!
	bool use_barriers = (MAX_TEST_THREADS > 1);
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: use_barriers = %s", rank, use_barriers ? "true" : "false");

	if (use_barriers) {
		int total_threads = MAX_TEST_THREADS;
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Initializing barriers for %d threads", rank, total_threads);
		if (pthread_barrier_init(&setup_barrier, nullptr, total_threads) != 0 ||
		    pthread_barrier_init(&test_barrier, nullptr, total_threads) != 0 ||
		    pthread_barrier_init(&cleanup_barrier, nullptr, total_threads) != 0) {
			NCCL_OFI_WARN("RANK %d: Failed to initialize barriers", rank);
			return ncclInternalError;
		}
	}

	// Create threads
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Creating %zu threads", rank, MAX_TEST_THREADS);
	for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
		auto& ctx = thread_contexts[i];
		ctx.initialize(static_cast<int>(i), rank, device_id, buffer_type, ext_net,
			use_barriers ? &setup_barrier : nullptr,
			use_barriers ? &test_barrier : nullptr,
			use_barriers ? &cleanup_barrier : nullptr);

		NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Creating thread %zu", rank, i);
		if (pthread_create(&threads[i], nullptr, thread_worker, &ctx) != 0) {
			NCCL_OFI_WARN("RANK %d: Failed to create thread %zu", rank, i);
			return ncclInternalError;
		}
	}

	// Wait for threads to be ready
	for (int wait_ms = 0; wait_ms < 30000; wait_ms += 10) {
		bool all_ready = true;
		for (size_t i = 0; i < MAX_TEST_THREADS && all_ready; i++) {
			all_ready = thread_contexts[i].is_ready();
		}
		if (all_ready) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	// Start testing
	for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
		thread_contexts[i].set_start_test(true);
	}

	// Wait for completion
	for (int wait_ms = 0; wait_ms < 30000; wait_ms += 10) {
		bool all_completed = true;
		for (size_t i = 0; i < MAX_TEST_THREADS && all_completed; i++) {
			all_completed = thread_contexts[i].is_completed();
		}
		if (all_completed) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	// Join threads and check results
	ncclResult_t final_result = ncclSuccess;
	for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
		pthread_join(threads[i], nullptr);
		if (thread_contexts[i].get_result() != ncclSuccess) final_result = thread_contexts[i].get_result();
	}

	// Cleanup barriers (only if we created them)
	if (use_barriers) {
		pthread_barrier_destroy(&cleanup_barrier);
		pthread_barrier_destroy(&test_barrier);
		pthread_barrier_destroy(&setup_barrier);
	}

	return final_result;
}

static void parse_arguments(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--comms" && i + 1 < argc) MAX_TEST_COMMS = std::max(1, std::stoi(argv[++i]));
		else if (arg == "--threads" && i + 1 < argc) MAX_TEST_THREADS = std::max(1, std::stoi(argv[++i]));
		else if (arg == "--host") FORCE_HOST_BUFFERS = true;
		else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: " << argv[0] << " [--comms N] [--threads N] [--host]\n"
			          << "  --comms N    Number of communicators per thread (default: 4)\n"
			          << "  --threads N  Number of threads per rank (default: 3)\n"
			          << "  --host       Force host buffers (default: auto-detect like nccl_connection)\n"
			          << "  --help       Show this help message\n"
			          << "\nDegenerate case: --comms 1 --threads 1 (equivalent to nccl_connection test)\n";
			std::exit(0);
		}
	}

	if (const char* env = std::getenv("MAX_TEST_COMMS")) MAX_TEST_COMMS = std::max(1, std::stoi(env));
	if (const char* env = std::getenv("MAX_TEST_THREADS")) MAX_TEST_THREADS = std::max(1, std::stoi(env));
	if (const char* env = std::getenv("FORCE_HOST_BUFFERS")) FORCE_HOST_BUFFERS = (std::string(env) == "1" || std::string(env) == "true");
}

int main(int argc, char* argv[])
{
	int rank, size, ndev;
	char name[MPI_MAX_PROCESSOR_NAME];
	int proc_name_len;

	parse_arguments(argc, argv);
	ofi_log_function = logger;

	// Try to disable some problematic features that might cause issues in test environment
	setenv("FI_EFA_USE_DEVICE_RDMA", "0", 0);  // Try to disable RDMA features that might need GPU buffers

	// If using host buffers, disable GDR flush to avoid GPU memory allocation issues
	if (FORCE_HOST_BUFFERS) {
		setenv("NCCL_OFI_GDR_FLUSH_DISABLE", "1", 0);
		NCCL_OFI_INFO(NCCL_INIT, "DEBUG: Disabled GDR flush due to --host flag");
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &proc_name_len);

	if (size != 2) {
		MPI_Finalize();
		return ncclInvalidArgument;
	}

	test_nccl_net_t* ext_net = get_extNet();
	if (!ext_net || ext_net->init(logger) != ncclSuccess || ext_net->devices(&ndev) != ncclSuccess) {
		MPI_Finalize();
		return ncclInternalError;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Process rank %d started. NCCLNet device used on %s is %s.",
		rank, name, ext_net->name);
	NCCL_OFI_INFO(NCCL_INIT, "Test configuration: %zu communicators per thread, %zu threads per rank, %s buffers",
		MAX_TEST_COMMS, MAX_TEST_THREADS, FORCE_HOST_BUFFERS ? "host" : "auto");

#if HAVE_CUDA
	// Calculate CUDA device mapping before creating threads (avoids MPI thread safety issues)
	if (!calculate_local_rank_and_cuda_device()) {
		NCCL_OFI_WARN("Failed to calculate CUDA device mapping");
		MPI_Finalize();
		return ncclInternalError;
	}
#endif

	// Test all devices
	for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {
		int dev = (rank == 1) ? (ndev - dev_idx - 1) : dev_idx;

		test_nccl_properties_t props = {};
		if (ext_net->getProperties(dev, &props) != ncclSuccess) continue;
		print_dev_props(dev, &props);

		int gdr_support = is_gdr_supported_nic(props.ptrSupport);
		if (gdr_support == 1) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Network supports CUDA buffers. Dev: %d", dev);
		}

		// Replicate nccl_connection behavior: use CUDA buffers if supported, unless --host is specified
		int buffer_type = (FORCE_HOST_BUFFERS || gdr_support != 1) ? NCCL_PTR_HOST : NCCL_PTR_CUDA;

		if (FORCE_HOST_BUFFERS) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using host buffers (forced via --host flag). Dev: %d", dev);
		} else if (gdr_support == 1) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using CUDA buffers (auto-detected like nccl_connection). Dev: %d", dev);
		} else {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using host buffers (GPU not supported). Dev: %d", dev);
		}
		if (test_multithreaded_communicators(ext_net, dev, buffer_type, rank) != ncclSuccess) {
			NCCL_OFI_WARN("Multithreaded test failed on device %d", dev);
			MPI_Finalize();
			return ncclInternalError;
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Multi-communicator domain test completed successfully for rank %d", rank);

	return ncclSuccess;
}
