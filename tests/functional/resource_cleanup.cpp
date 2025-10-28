/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * Resource cleanup validation test with threading support.
 * Tests that all resources (domains, memory registrations, endpoints, communicators, buffers)
 * are properly cleaned up after communicator lifecycle operations.
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
#include <unordered_set>

// Resource tracking for cleanup validation
struct ResourceSnapshot {
	size_t memory_registrations{0};
	size_t listen_comms{0};
	size_t send_comms{0};
	size_t recv_comms{0};
	size_t allocated_buffers{0};
	size_t cuda_allocations{0};
	
	void print(const std::string& label) const {
		NCCL_OFI_INFO(NCCL_NET, "%s - MRs: %zu, Listen: %zu, Send: %zu, Recv: %zu, Buffers: %zu, CUDA: %zu",
			label.c_str(), memory_registrations, listen_comms, send_comms, recv_comms, allocated_buffers, cuda_allocations);
	}
	
	bool operator==(const ResourceSnapshot& other) const {
		return memory_registrations == other.memory_registrations &&
		       listen_comms == other.listen_comms &&
		       send_comms == other.send_comms &&
		       recv_comms == other.recv_comms &&
		       allocated_buffers == other.allocated_buffers &&
		       cuda_allocations == other.cuda_allocations;
	}
};

class ResourceTracker {
private:
	std::unordered_set<void*> tracked_memory_registrations;
	std::unordered_set<void*> tracked_listen_comms;
	std::unordered_set<void*> tracked_send_comms;
	std::unordered_set<void*> tracked_recv_comms;
	std::unordered_set<void*> tracked_buffers;
	std::unordered_set<void*> tracked_cuda_allocations;
	mutable pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

	template<typename T>
	void track_resource(std::unordered_set<T>& set, T resource) {
		pthread_mutex_lock(&mutex);
		set.insert(resource);
		pthread_mutex_unlock(&mutex);
	}

	template<typename T>
	void untrack_resource(std::unordered_set<T>& set, T resource) {
		pthread_mutex_lock(&mutex);
		set.erase(resource);
		pthread_mutex_unlock(&mutex);
	}

public:
	~ResourceTracker() { pthread_mutex_destroy(&mutex); }

	void track_memory_registration(void* mr) { track_resource(tracked_memory_registrations, mr); }
	void untrack_memory_registration(void* mr) { untrack_resource(tracked_memory_registrations, mr); }
	void track_listen_comm(void* comm) { track_resource(tracked_listen_comms, comm); }
	void untrack_listen_comm(void* comm) { untrack_resource(tracked_listen_comms, comm); }
	void track_send_comm(void* comm) { track_resource(tracked_send_comms, comm); }
	void untrack_send_comm(void* comm) { untrack_resource(tracked_send_comms, comm); }
	void track_recv_comm(void* comm) { track_resource(tracked_recv_comms, comm); }
	void untrack_recv_comm(void* comm) { untrack_resource(tracked_recv_comms, comm); }
	void track_buffer(void* buffer) { track_resource(tracked_buffers, buffer); }
	void untrack_buffer(void* buffer) { untrack_resource(tracked_buffers, buffer); }
	void track_cuda_allocation(void* ptr) { track_resource(tracked_cuda_allocations, ptr); }
	void untrack_cuda_allocation(void* ptr) { untrack_resource(tracked_cuda_allocations, ptr); }

	ResourceSnapshot take_snapshot() const {
		pthread_mutex_lock(&mutex);
		ResourceSnapshot snapshot;
		snapshot.memory_registrations = tracked_memory_registrations.size();
		snapshot.listen_comms = tracked_listen_comms.size();
		snapshot.send_comms = tracked_send_comms.size();
		snapshot.recv_comms = tracked_recv_comms.size();
		snapshot.allocated_buffers = tracked_buffers.size();
		snapshot.cuda_allocations = tracked_cuda_allocations.size();
		pthread_mutex_unlock(&mutex);
		return snapshot;
	}

	bool validate_cleanup(const ResourceSnapshot& before, const ResourceSnapshot& after) const {
		bool clean = (before == after);
		if (!clean) {
			NCCL_OFI_WARN("Resource leak detected!");
			before.print("Before");
			after.print("After");
		}
		return clean;
	}
};

static ResourceTracker g_resource_tracker;

#if HAVE_CUDA
#include <cuda_runtime.h>

static int g_local_rank = -1;
static int g_cuda_device = -1;

static bool calculate_local_rank_and_cuda_device() {
	int rank, num_ranks, local_rank = 0;
	char name[MPI_MAX_PROCESSOR_NAME];
	int proc_name_len;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	MPI_Get_processor_name(name, &proc_name_len);

	char *all_proc_name = (char *)malloc(sizeof(char) * num_ranks * MPI_MAX_PROCESSOR_NAME);
	if (!all_proc_name) return false;

	MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_proc_name,
		MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

	for (int i = 0; i < num_ranks; i++) {
		if (!strcmp(name, &all_proc_name[i * MPI_MAX_PROCESSOR_NAME]) && i < rank) {
			++local_rank;
		}
	}
	free(all_proc_name);

	int device_count = 0;
	if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return false;

	g_local_rank = local_rank;
	g_cuda_device = local_rank % device_count;
	return true;
}

static bool initialize_cuda_for_thread() {
	if (cudaSetDevice(g_cuda_device) != cudaSuccess) return false;
	void* dummy_ptr = nullptr;
	cudaError_t result = cudaMalloc(&dummy_ptr, 1);
	if (result == cudaSuccess && dummy_ptr) cudaFree(dummy_ptr);
	return result == cudaSuccess;
}
#endif

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
		: send_comms(MAX_TEST_COMMS, nullptr), recv_comms(MAX_TEST_COMMS, nullptr)
		, listen_comms(MAX_TEST_COMMS, nullptr), send_handles(MAX_TEST_COMMS, nullptr)
		, recv_handles(MAX_TEST_COMMS, nullptr), src_handles(MAX_TEST_COMMS), handles(MAX_TEST_COMMS) {}

	ncclResult_t create_communicator_pair(test_nccl_net_t* ext_net, size_t comm_idx, int rank, int dev, int thread_id) {
		int base_tag = (thread_id * static_cast<int>(MAX_TEST_COMMS) * 2) + (static_cast<int>(comm_idx) * 2);

		// Listen
		ncclResult_t listen_result = ext_net->listen(dev, static_cast<void*>(&handles[comm_idx]),
			reinterpret_cast<void**>(&listen_comms[comm_idx]));
		if (listen_result != ncclSuccess) return ncclInternalError;
		g_resource_tracker.track_listen_comm(listen_comms[comm_idx]);

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
				if (send_comms[comm_idx]) g_resource_tracker.track_send_comm(send_comms[comm_idx]);
			}
			if (!recv_comms[comm_idx]) {
				ext_net->accept(static_cast<void*>(listen_comms[comm_idx]),
					reinterpret_cast<void**>(&recv_comms[comm_idx]), &recv_handles[comm_idx]);
				if (recv_comms[comm_idx]) g_resource_tracker.track_recv_comm(recv_comms[comm_idx]);
			}
		}

		active_comm_count++;
		return ncclSuccess;
	}

	ncclResult_t validate_domain_sharing() const {
		if (active_comm_count == 0) return ncclInternalError;
		for (size_t i = 0; i < active_comm_count; i++) {
			if (!send_comms[i] || !recv_comms[i]) return ncclInternalError;
			if (send_comms[i]->base.dev_id != device_id || recv_comms[i]->base.dev_id != device_id) {
				return ncclInternalError;
			}
		}
		return ncclSuccess;
	}

	ncclResult_t test_concurrent_operations(test_nccl_net_t* ext_net, int buffer_type, int rank) const {
		if (!ext_net || active_comm_count == 0) return ncclSuccess;

		constexpr size_t test_size = SEND_SIZE;
		std::vector<void*> send_buffers(MAX_TEST_COMMS, nullptr);
		std::vector<void*> recv_buffers(MAX_TEST_COMMS, nullptr);
		std::vector<void*> send_mhandles(MAX_TEST_COMMS, nullptr);
		std::vector<void*> recv_mhandles(MAX_TEST_COMMS, nullptr);
		std::vector<nccl_net_ofi_req_t*> send_reqs(MAX_TEST_COMMS, nullptr);
		std::vector<nccl_net_ofi_req_t*> recv_reqs(MAX_TEST_COMMS, nullptr);

		// Allocate and register buffers with tracking
		for (size_t i = 0; i < active_comm_count; i++) {
			if (allocate_buff(&send_buffers[i], test_size, buffer_type) != ncclSuccess ||
			    initialize_buff(send_buffers[i], test_size, buffer_type) != ncclSuccess ||
			    allocate_buff(&recv_buffers[i], test_size, buffer_type) != ncclSuccess) {
				// Cleanup on failure
				for (size_t j = 0; j <= i; j++) {
					if (send_buffers[j]) {
						g_resource_tracker.untrack_buffer(send_buffers[j]);
						if (buffer_type == NCCL_PTR_CUDA) g_resource_tracker.untrack_cuda_allocation(send_buffers[j]);
						deallocate_buffer(send_buffers[j], buffer_type);
					}
					if (recv_buffers[j]) {
						g_resource_tracker.untrack_buffer(recv_buffers[j]);
						if (buffer_type == NCCL_PTR_CUDA) g_resource_tracker.untrack_cuda_allocation(recv_buffers[j]);
						deallocate_buffer(recv_buffers[j], buffer_type);
					}
				}
				return ncclInternalError;
			}

			// Track allocated buffers
			g_resource_tracker.track_buffer(send_buffers[i]);
			g_resource_tracker.track_buffer(recv_buffers[i]);
			if (buffer_type == NCCL_PTR_CUDA) {
				g_resource_tracker.track_cuda_allocation(send_buffers[i]);
				g_resource_tracker.track_cuda_allocation(recv_buffers[i]);
			}

			// Register memory handles
			if (ext_net->regMr(static_cast<void*>(send_comms[i]), send_buffers[i], test_size, buffer_type, &send_mhandles[i]) != ncclSuccess ||
			    ext_net->regMr(static_cast<void*>(recv_comms[i]), recv_buffers[i], test_size, buffer_type, &recv_mhandles[i]) != ncclSuccess) {
				// Cleanup on failure
				for (size_t j = 0; j <= i; j++) {
					if (send_mhandles[j]) {
						g_resource_tracker.untrack_memory_registration(send_mhandles[j]);
						ext_net->deregMr(static_cast<void*>(send_comms[j]), send_mhandles[j]);
					}
					if (recv_mhandles[j]) {
						g_resource_tracker.untrack_memory_registration(recv_mhandles[j]);
						ext_net->deregMr(static_cast<void*>(recv_comms[j]), recv_mhandles[j]);
					}
				}
				return ncclInternalError;
			}

			// Track memory registrations
			g_resource_tracker.track_memory_registration(send_mhandles[i]);
			g_resource_tracker.track_memory_registration(recv_mhandles[i]);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// RAII cleanup
		auto cleanup_buffers = [&]() {
			for (size_t i = 0; i < active_comm_count; i++) {
				if (send_mhandles[i]) {
					g_resource_tracker.untrack_memory_registration(send_mhandles[i]);
					ext_net->deregMr(static_cast<void*>(send_comms[i]), send_mhandles[i]);
				}
				if (recv_mhandles[i]) {
					g_resource_tracker.untrack_memory_registration(recv_mhandles[i]);
					ext_net->deregMr(static_cast<void*>(recv_comms[i]), recv_mhandles[i]);
				}
				if (send_buffers[i]) {
					g_resource_tracker.untrack_buffer(send_buffers[i]);
					if (buffer_type == NCCL_PTR_CUDA) g_resource_tracker.untrack_cuda_allocation(send_buffers[i]);
					deallocate_buffer(send_buffers[i], buffer_type);
				}
				if (recv_buffers[i]) {
					g_resource_tracker.untrack_buffer(recv_buffers[i]);
					if (buffer_type == NCCL_PTR_CUDA) g_resource_tracker.untrack_cuda_allocation(recv_buffers[i]);
					deallocate_buffer(recv_buffers[i], buffer_type);
				}
			}
		};

		bool operations_successful = true;

		// Post operations
		for (size_t i = 0; i < active_comm_count && operations_successful; i++) {
			if (!recv_comms[i] || !send_comms[i] || !recv_comms[i]->recv || !send_comms[i]->send) {
				operations_successful = false;
				break;
			}

			int nrecv = 1;
			void* recv_data[1] = {recv_buffers[i]};
			size_t recv_sizes[1] = {test_size};
			int recv_tags[1] = {static_cast<int>(i)};

			if (ext_net->irecv(static_cast<void*>(recv_comms[i]), nrecv, recv_data, recv_sizes,
				recv_tags, &recv_mhandles[i], reinterpret_cast<void**>(&recv_reqs[i])) != 0 ||
			    ext_net->isend(static_cast<void*>(send_comms[i]), send_buffers[i], test_size, static_cast<int>(i),
				send_mhandles[i], reinterpret_cast<void**>(&send_reqs[i])) != 0 ||
			    !send_reqs[i] || !recv_reqs[i]) {
				operations_successful = false;
				break;
			}
		}

		// Wait for completion
		bool all_complete = false;
		if (operations_successful) {
			for (int iteration = 0; !all_complete && iteration < 1000; iteration++) {
				all_complete = true;
				for (size_t i = 0; i < active_comm_count; i++) {
					if (send_reqs[i] && send_reqs[i]->test) {
						int done = 0, size = 0;
						if (ext_net->test(static_cast<void*>(send_reqs[i]), &done, &size) == 0 && done) {
							send_reqs[i] = nullptr;
						} else {
							all_complete = false;
						}
					}
					if (recv_reqs[i] && recv_reqs[i]->test) {
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
			if (listen_comms[i]) {
				g_resource_tracker.untrack_listen_comm(listen_comms[i]);
				ext_net->closeListen(static_cast<void*>(listen_comms[i]));
				listen_comms[i] = nullptr;
			}
			if (send_comms[i]) {
				g_resource_tracker.untrack_send_comm(send_comms[i]);
				ext_net->closeSend(static_cast<void*>(send_comms[i]));
				send_comms[i] = nullptr;
			}
			if (recv_comms[i]) {
				g_resource_tracker.untrack_recv_comm(recv_comms[i]);
				ext_net->closeRecv(static_cast<void*>(recv_comms[i]));
				recv_comms[i] = nullptr;
			}
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
	void initialize(int tid, int r, int dev_id, int buf_type, test_nccl_net_t* net,
		pthread_barrier_t* setup_bar, pthread_barrier_t* test_bar, pthread_barrier_t* cleanup_bar) {
		thread_id = tid; rank = r; device_id = dev_id; buffer_type = buf_type; ext_net = net;
		setup_barrier = setup_bar; test_barrier = test_bar; cleanup_barrier = cleanup_bar;
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

static void* thread_worker(void* arg) {
	auto* thread_ctx = static_cast<ThreadContext*>(arg);
	int rank = thread_ctx->get_rank();
	int thread_id = thread_ctx->get_thread_id();

#if HAVE_CUDA
	bool cuda_initialized = initialize_cuda_for_thread();
#endif

	thread_ctx->get_comm_ctx().set_device_id(thread_ctx->get_device_id());

	// Create communicators
	for (size_t comm_idx = 0; comm_idx < MAX_TEST_COMMS; comm_idx++) {
		if (thread_ctx->get_comm_ctx().create_communicator_pair(thread_ctx->get_ext_net(), comm_idx,
			thread_ctx->get_rank(), thread_ctx->get_device_id(), thread_ctx->get_thread_id()) != ncclSuccess) {
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

	// Run tests
	ncclResult_t result = thread_ctx->get_comm_ctx().validate_domain_sharing();
	if (result == ncclSuccess) {
		result = thread_ctx->get_comm_ctx().test_concurrent_operations(thread_ctx->get_ext_net(),
			thread_ctx->get_buffer_type(), thread_ctx->get_rank());
	}
	thread_ctx->set_result(result);

	thread_ctx->wait_test_barrier();
	thread_ctx->get_comm_ctx().cleanup(thread_ctx->get_ext_net());
	thread_ctx->wait_cleanup_barrier();
	thread_ctx->set_completed(true);
	return nullptr;
}

static ncclResult_t test_multithreaded_communicators(test_nccl_net_t* ext_net, int device_id, int buffer_type, int rank) {
	std::vector<pthread_t> threads(MAX_TEST_THREADS);
	std::vector<ThreadContext> thread_contexts(MAX_TEST_THREADS);
	pthread_barrier_t setup_barrier, test_barrier, cleanup_barrier;

	bool use_barriers = (MAX_TEST_THREADS > 1);
	if (use_barriers) {
		if (pthread_barrier_init(&setup_barrier, nullptr, MAX_TEST_THREADS) != 0 ||
		    pthread_barrier_init(&test_barrier, nullptr, MAX_TEST_THREADS) != 0 ||
		    pthread_barrier_init(&cleanup_barrier, nullptr, MAX_TEST_THREADS) != 0) {
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

// Resource cleanup validation wrapper
static ncclResult_t test_multithreaded_communicators_with_cleanup_validation(test_nccl_net_t* ext_net, int device_id, int buffer_type, int rank) {
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Starting resource cleanup validation test", rank);
	
	ResourceSnapshot initial_snapshot = g_resource_tracker.take_snapshot();
	initial_snapshot.print("Initial state");
	
	ncclResult_t test_result = test_multithreaded_communicators(ext_net, device_id, buffer_type, rank);
	
	ResourceSnapshot final_snapshot = g_resource_tracker.take_snapshot();
	final_snapshot.print("Final state");
	
	bool cleanup_successful = g_resource_tracker.validate_cleanup(initial_snapshot, final_snapshot);
	
	if (!cleanup_successful) {
		NCCL_OFI_WARN("RANK %d: Resource cleanup validation FAILED", rank);
		return ncclInternalError;
	}
	
	if (test_result != ncclSuccess) {
		NCCL_OFI_WARN("RANK %d: Multithreaded communicator test FAILED: %d", rank, test_result);
		return test_result;
	}
	
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Resource cleanup validation PASSED", rank);
	return ncclSuccess;
}

static void parse_arguments(int argc, char* argv[]) {
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--comms" && i + 1 < argc) MAX_TEST_COMMS = std::max(1, std::stoi(argv[++i]));
		else if (arg == "--threads" && i + 1 < argc) MAX_TEST_THREADS = std::max(1, std::stoi(argv[++i]));
		else if (arg == "--host") FORCE_HOST_BUFFERS = true;
		else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: " << argv[0] << " [--comms N] [--threads N] [--host]\n"
			          << "  --comms N    Number of communicators per thread (default: 4)\n"
			          << "  --threads N  Number of threads per rank (default: 3)\n"
			          << "  --host       Force host buffers (default: auto-detect)\n"
			          << "  --help       Show this help message\n"
			          << "\nResource cleanup validation test - validates no resource leaks\n";
			std::exit(0);
		}
	}

	if (const char* env = std::getenv("MAX_TEST_COMMS")) MAX_TEST_COMMS = std::max(1, std::stoi(env));
	if (const char* env = std::getenv("MAX_TEST_THREADS")) MAX_TEST_THREADS = std::max(1, std::stoi(env));
	if (const char* env = std::getenv("FORCE_HOST_BUFFERS")) FORCE_HOST_BUFFERS = (std::string(env) == "1" || std::string(env) == "true");
}

int main(int argc, char* argv[]) {
	int rank, size, ndev;
	char name[MPI_MAX_PROCESSOR_NAME];
	int proc_name_len;

	parse_arguments(argc, argv);
	ofi_log_function = logger;

	setenv("FI_EFA_USE_DEVICE_RDMA", "0", 0);
	if (FORCE_HOST_BUFFERS) {
		setenv("NCCL_OFI_GDR_FLUSH_DISABLE", "1", 0);
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

#if HAVE_CUDA
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
		int buffer_type = (FORCE_HOST_BUFFERS || gdr_support != 1) ? NCCL_PTR_HOST : NCCL_PTR_CUDA;

		if (test_multithreaded_communicators_with_cleanup_validation(ext_net, dev, buffer_type, rank) != ncclSuccess) {
			NCCL_OFI_WARN("Resource cleanup validation test failed on device %d", dev);
			MPI_Finalize();
			return ncclInternalError;
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Resource cleanup validation test completed successfully for rank %d", rank);

	return ncclSuccess;
}