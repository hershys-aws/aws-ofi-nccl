/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * Tests closing a communicator while there are still inflight requests
 * with support for multiple communicators and multiple threads
 */

#include "config.h"

#include <algorithm>
#include <vector>
#include <pthread.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <array>
#include <string>
#include <iostream>

#include "test-common.h"

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

#define PROC_NAME_IDX(i) (i * MPI_MAX_PROCESSOR_NAME)

#define RESTART_ITERS 10

// Default parameters - can be overridden via command line or environment
static size_t MAX_TEST_COMMS = 2;
static size_t MAX_TEST_THREADS = 2;

class MultiCommAbortContext {
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
	MultiCommAbortContext()
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
		// Unique tags across all threads and communicators
		int base_tag = (thread_id * static_cast<int>(MAX_TEST_COMMS) * 2) + (static_cast<int>(comm_idx) * 2);

		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: create_communicator_pair comm %zu, dev %d, base_tag %d",
			rank, thread_id, comm_idx, dev, base_tag);

		// Listen
		ncclResult_t listen_result = ext_net->listen(dev, static_cast<void*>(&handles[comm_idx]),
			reinterpret_cast<void**>(&listen_comms[comm_idx]));
		if (listen_result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d THREAD %d: Failed to create listen communicator for comm %zu on device %d: %d",
				rank, thread_id, comm_idx, dev, listen_result);
			return ncclInternalError;
		}

		// Exchange handles
		if (rank == 0) {
			MPI_Send(&handles[comm_idx], NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 1, base_tag, MPI_COMM_WORLD);
			MPI_Recv(src_handles[comm_idx].data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 1, base_tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(src_handles[comm_idx].data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 0, base_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(handles[comm_idx].data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 0, base_tag + 1, MPI_COMM_WORLD);
		}

		// Connect and Accept
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

		active_comm_count++;
		return ncclSuccess;
	}

	ncclResult_t test_inflight_abort(test_nccl_net_t* ext_net, int buffer_type, int rank) {
		if (!ext_net || active_comm_count == 0) return ncclSuccess;

		const size_t send_size = 1024 * 1024;
		const size_t recv_size = 1024 * 1024;
		const int nrecv = NCCL_OFI_MAX_RECVS;

		std::vector<std::vector<nccl_net_ofi_req_t*>> send_reqs(active_comm_count, std::vector<nccl_net_ofi_req_t*>(NUM_REQUESTS, nullptr));
		std::vector<std::vector<nccl_net_ofi_req_t*>> recv_reqs(active_comm_count, std::vector<nccl_net_ofi_req_t*>(NUM_REQUESTS, nullptr));
		std::vector<std::vector<void*>> send_mhandles(active_comm_count, std::vector<void*>(NUM_REQUESTS, nullptr));
		std::vector<std::vector<void*>> recv_mhandles(active_comm_count, std::vector<void*>(NUM_REQUESTS, nullptr));
		std::vector<std::vector<char*>> send_buffers(active_comm_count, std::vector<char*>(NUM_REQUESTS, nullptr));
		std::vector<std::vector<char*>> recv_buffers(active_comm_count, std::vector<char*>(NUM_REQUESTS, nullptr));

		// Allocate and register buffers for all communicators
		for (size_t comm_idx = 0; comm_idx < active_comm_count; comm_idx++) {
			for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
				if (rank == 0) {
					if (allocate_buff(reinterpret_cast<void**>(&send_buffers[comm_idx][req_idx]), send_size, buffer_type) != ncclSuccess ||
					    initialize_buff(send_buffers[comm_idx][req_idx], send_size, buffer_type) != ncclSuccess ||
					    ext_net->regMr(static_cast<void*>(send_comms[comm_idx]), send_buffers[comm_idx][req_idx],
					                   send_size, buffer_type, &send_mhandles[comm_idx][req_idx]) != ncclSuccess) {
						return ncclInternalError;
					}
				} else {
					if (allocate_buff(reinterpret_cast<void**>(&recv_buffers[comm_idx][req_idx]), recv_size, buffer_type) != ncclSuccess ||
					    ext_net->regMr(static_cast<void*>(recv_comms[comm_idx]), recv_buffers[comm_idx][req_idx],
					                   recv_size, buffer_type, &recv_mhandles[comm_idx][req_idx]) != ncclSuccess) {
						return ncclInternalError;
					}
				}
			}
		}

		// Post operations that will be inflight when we abort
		for (size_t comm_idx = 0; comm_idx < active_comm_count; comm_idx++) {
			int tag = static_cast<int>(comm_idx) + 1;
			
			if (rank == 0) {
				// Send operations
				for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
					while (send_reqs[comm_idx][req_idx] == nullptr) {
						ext_net->isend(static_cast<void*>(send_comms[comm_idx]), 
						              send_buffers[comm_idx][req_idx], send_size, tag,
						              send_mhandles[comm_idx][req_idx],
						              reinterpret_cast<void**>(&send_reqs[comm_idx][req_idx]));
					}
				}
			} else {
				// Receive operations
				for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
					size_t sizes[nrecv];
					int tags[nrecv];
					for (int recv_n = 0; recv_n < nrecv; recv_n++) {
						sizes[recv_n] = recv_size;
						tags[recv_n] = tag;
					}
					
					while (recv_reqs[comm_idx][req_idx] == nullptr) {
						ext_net->irecv(static_cast<void*>(recv_comms[comm_idx]), nrecv,
						              reinterpret_cast<void**>(&recv_buffers[comm_idx][req_idx]), 
						              sizes, tags, &recv_mhandles[comm_idx][req_idx],
						              reinterpret_cast<void**>(&recv_reqs[comm_idx][req_idx]));
					}
				}
			}
		}

		NCCL_OFI_INFO(NCCL_NET, "RANK %d: Posted %d inflight requests across %zu communicators", 
		              rank, NUM_REQUESTS, active_comm_count);

		// Cleanup with inflight requests (this is the abort scenario test)
		for (size_t comm_idx = 0; comm_idx < active_comm_count; comm_idx++) {
			for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
				if (rank == 0 && send_mhandles[comm_idx][req_idx]) {
					ext_net->deregMr(static_cast<void*>(send_comms[comm_idx]), send_mhandles[comm_idx][req_idx]);
				}
				if (rank == 1 && recv_mhandles[comm_idx][req_idx]) {
					ext_net->deregMr(static_cast<void*>(recv_comms[comm_idx]), recv_mhandles[comm_idx][req_idx]);
				}
			}
		}

		return ncclSuccess;
	}

	void cleanup(test_nccl_net_t* ext_net) {
		if (!ext_net) return;

		for (size_t i = 0; i < active_comm_count; i++) {
			if (send_comms[i]) ext_net->closeSend(static_cast<void*>(send_comms[i]));
			if (recv_comms[i]) ext_net->closeRecv(static_cast<void*>(recv_comms[i]));
			if (listen_comms[i]) ext_net->closeListen(static_cast<void*>(listen_comms[i]));
		}
		active_comm_count = 0;
	}

	void set_device_id(int dev_id) { device_id = dev_id; }
	size_t get_active_comm_count() const { return active_comm_count; }
};

class ThreadAbortContext {
private:
	int thread_id{0}, rank{0}, device_id{0}, buffer_type{0};
	test_nccl_net_t* ext_net{nullptr};
	MultiCommAbortContext comm_ctx{};
	std::atomic<bool> ready{false}, start_test{false}, completed{false};
	pthread_barrier_t* setup_barrier{nullptr}, *test_barrier{nullptr}, *cleanup_barrier{nullptr};
	ncclResult_t result{ncclSuccess};

public:
	ThreadAbortContext() = default;

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
	MultiCommAbortContext& get_comm_ctx() { return comm_ctx; }
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
	auto* thread_ctx = static_cast<ThreadAbortContext*>(arg);
	int rank = thread_ctx->get_rank();
	int thread_id = thread_ctx->get_thread_id();

	NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Starting abort scenario thread", rank, thread_id);

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
	for (size_t comm_idx = 0; comm_idx < MAX_TEST_COMMS; comm_idx++) {
		if (thread_ctx->get_comm_ctx().create_communicator_pair(thread_ctx->get_ext_net(), comm_idx,
			thread_ctx->get_rank(), thread_ctx->get_device_id(), thread_ctx->get_thread_id()) != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d THREAD %d: Failed to create communicator %zu", rank, thread_id, comm_idx);
			thread_ctx->set_result(ncclInternalError);
			thread_ctx->get_comm_ctx().cleanup(thread_ctx->get_ext_net());
			thread_ctx->set_completed(true);
			return nullptr;
		}
		thread_ctx->wait_setup_barrier();
	}

	// Signal ready and wait for start
	thread_ctx->set_ready(true);
	while (!thread_ctx->should_start_test()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	// Run abort scenario test
	ncclResult_t result = thread_ctx->get_comm_ctx().test_inflight_abort(thread_ctx->get_ext_net(),
		thread_ctx->get_buffer_type(), thread_ctx->get_rank());
	
	if (result != ncclSuccess) {
		NCCL_OFI_WARN("RANK %d THREAD %d: Inflight abort test failed: %d", rank, thread_id, result);
		thread_ctx->set_result(result);
	} else {
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Inflight abort test passed", rank, thread_id);
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

static ncclResult_t test_multithreaded_abort_scenarios(test_nccl_net_t* ext_net, int device_id,
	int buffer_type, int rank)
{
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Starting multithreaded abort scenarios with device %d, threads %zu, comms %zu",
		rank, device_id, MAX_TEST_THREADS, MAX_TEST_COMMS);

	std::vector<pthread_t> threads(MAX_TEST_THREADS);
	std::vector<ThreadAbortContext> thread_contexts(MAX_TEST_THREADS);
	pthread_barrier_t setup_barrier, test_barrier, cleanup_barrier;

	bool use_barriers = (MAX_TEST_THREADS > 1);
	if (use_barriers) {
		int total_threads = MAX_TEST_THREADS;
		if (pthread_barrier_init(&setup_barrier, nullptr, total_threads) != 0 ||
		    pthread_barrier_init(&test_barrier, nullptr, total_threads) != 0 ||
		    pthread_barrier_init(&cleanup_barrier, nullptr, total_threads) != 0) {
			NCCL_OFI_WARN("RANK %d: Failed to initialize barriers", rank);
			return ncclInternalError;
		}
	}

	// Create threads
	for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
		auto& ctx = thread_contexts[i];
		ctx.initialize(static_cast<int>(i), rank, device_id, buffer_type, ext_net,
			use_barriers ? &setup_barrier : nullptr,
			use_barriers ? &test_barrier : nullptr,
			use_barriers ? &cleanup_barrier : nullptr);

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

	// Cleanup barriers
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
		if (arg == "--comms" && i + 1 < argc) {
			MAX_TEST_COMMS = std::max(1, std::stoi(argv[++i]));
		} else if (arg == "--threads" && i + 1 < argc) {
			MAX_TEST_THREADS = std::max(1, std::stoi(argv[++i]));
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: " << argv[0] << " [--comms N] [--threads N]\n"
			          << "  --comms N    Number of communicators per thread (default: 2)\n"
			          << "  --threads N  Number of threads per rank (default: 2)\n"
			          << "  --help       Show this help message\n"
			          << "\nTests closing communicators with inflight requests across multiple threads and communicators\n";
			std::exit(0);
		}
	}

	if (const char* env = std::getenv("MAX_TEST_COMMS")) {
		MAX_TEST_COMMS = std::max(1, std::stoi(env));
	}
	if (const char* env = std::getenv("MAX_TEST_THREADS")) {
		MAX_TEST_THREADS = std::max(1, std::stoi(env));
	}
}

int main(int argc, char* argv[])
{
	int rank = -1, proc_name_len = -1, num_ranks = 0, local_rank = 0;
	
	test_nccl_properties_t props = {};

	/* Plugin defines */
	int ndev = -1;
	test_nccl_net_t *extNet = NULL;

	parse_arguments(argc, argv);
	ofi_log_function = logger;

	/* Indicates if NICs support GPUDirect */
	std::vector<int> test_support_gdr;

	/* All processors IDs, used to find out the local rank */
	std::vector<char> all_proc_name;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	if (num_ranks != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The inflight_close functional test should be run with exactly two ranks.",
			num_ranks);
		return ncclInvalidArgument;
	}

	all_proc_name.resize(num_ranks * MPI_MAX_PROCESSOR_NAME);

	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
			MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (int i = 0; i < num_ranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
				&all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", local_rank);

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL) {
		return ncclInternalError;
	}

	/* Init API */
	OFINCCLCHECK(extNet->init(&logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.", rank,
			&all_proc_name[PROC_NAME_IDX(rank)], extNet->name);
	NCCL_OFI_INFO(NCCL_INIT, "Test configuration: %zu communicators per thread, %zu threads per rank",
		MAX_TEST_COMMS, MAX_TEST_THREADS);

#if HAVE_CUDA
	// Calculate CUDA device mapping before creating threads (avoids MPI thread safety issues)
	if (!calculate_local_rank_and_cuda_device()) {
		NCCL_OFI_WARN("Failed to calculate CUDA device mapping");
		MPI_Finalize();
		return ncclInternalError;
	}
#endif

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	test_support_gdr.resize(ndev);

	/* Get Properties for the device */
	for (int dev = 0; dev < ndev; dev++) {
		OFINCCLCHECK(extNet->getProperties(dev, &props));
		print_dev_props(dev, &props);

		/* Set CUDA support */
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	/* Test all devices */
	for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {

		for (int i = 0; i < RESTART_ITERS; ++i) {

			int dev = dev_idx;
			if (rank == 1) {
				/* In rank 1 scan devices in the opposite direction */
				dev = ndev - dev_idx - 1;
			}

			int buffer_type = NCCL_PTR_HOST;
			if (test_support_gdr[dev] == 1) {
				NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
					"Network supports communication using CUDA buffers. Dev: %d", dev);
				buffer_type = NCCL_PTR_CUDA;
			}

			OFINCCLCHECK(test_multithreaded_abort_scenarios(extNet, dev, buffer_type, rank));

			MPI_Barrier(MPI_COMM_WORLD);

		}
	}

	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Multi-threaded communicator abort scenarios test completed successfully for rank %d", rank);

	return 0;
}
