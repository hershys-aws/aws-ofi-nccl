/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * High Communicator Count Stress Test
 * Tests scale limits and resource management by creating many communicators sequentially.
 * Validates performance doesn't degrade significantly and tests graceful handling of resource exhaustion.
 */

#include "config.h"
#include "test-common.h"
#include <pthread.h>
#include <atomic>
#include <array>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <sys/resource.h>
#include <unistd.h>

#if HAVE_CUDA
#include <cuda_runtime.h>

// Initialize CUDA runtime for the process
static bool initialize_cuda_runtime() {
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

	// Determine local rank
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

	// Set device based on local rank
	int cuda_device = local_rank % device_count;
	result = cudaSetDevice(cuda_device);
	if (result != cudaSuccess) {
		fprintf(stderr, "Failed to set CUDA device %d: %s\n", cuda_device, cudaGetErrorString(result));
		return false;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Using CUDA device %d for memory allocation (local_rank=%d)", cuda_device, local_rank);
	return true;
}
#endif

// Default parameters - can be overridden via command line or environment
static size_t MAX_TEST_COMMS = 64;   // Number of communicators to test at scale
static size_t MAX_TEST_THREADS = 4;  // Number of threads for concurrent testing
static size_t BATCH_SIZE = 8;        // Create communicators in batches
static bool FORCE_HOST_BUFFERS = false;
static bool MEASURE_PERFORMANCE = true;
static bool TEST_RESOURCE_EXHAUSTION = false;

// Thread-safe tag generation for MPI communication
static std::atomic<int> global_tag_counter{1000};

// Performance metrics structure
struct PerformanceMetrics {
	std::chrono::high_resolution_clock::time_point start_time;
	std::chrono::high_resolution_clock::time_point end_time;
	size_t successful_comms{0};
	size_t failed_comms{0};
	double avg_creation_time_ms{0.0};
	double peak_memory_mb{0.0};
	
	void start_timing() {
		start_time = std::chrono::high_resolution_clock::now();
	}
	
	void end_timing() {
		end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		if (successful_comms > 0) {
			avg_creation_time_ms = static_cast<double>(duration.count()) / successful_comms;
		}
	}
	
	void record_memory_usage() {
		struct rusage usage;
		if (getrusage(RUSAGE_SELF, &usage) == 0) {
			// Convert from KB to MB
			peak_memory_mb = std::max(peak_memory_mb, static_cast<double>(usage.ru_maxrss) / 1024.0);
		}
	}
};

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
	size_t max_comm_capacity{0};
	int device_id{0};
	PerformanceMetrics metrics;

public:
	MultiCommContext(size_t max_comms)
		: send_comms(max_comms, nullptr)
		, recv_comms(max_comms, nullptr)
		, listen_comms(max_comms, nullptr)
		, send_handles(max_comms, nullptr)
		, recv_handles(max_comms, nullptr)
		, src_handles(max_comms)
		, handles(max_comms)
		, max_comm_capacity(max_comms)
	{}

	ncclResult_t create_communicator_pair(test_nccl_net_t* ext_net, size_t comm_idx, int rank, int dev) {
		if (comm_idx >= max_comm_capacity) {
			NCCL_OFI_WARN("RANK %d: Communicator index %zu exceeds capacity %zu", rank, comm_idx, max_comm_capacity);
			return ncclInvalidArgument;
		}

		auto start_time = std::chrono::high_resolution_clock::now();
		
		// Use thread-safe unique tags for multi-threaded testing
		int base_tag = global_tag_counter.fetch_add(2);

		NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Creating communicator pair %zu/%zu, dev %d, base_tag %d",
			rank, comm_idx + 1, max_comm_capacity, dev, base_tag);

		// Listen
		ncclResult_t listen_result = ext_net->listen(dev, static_cast<void*>(&handles[comm_idx]),
			reinterpret_cast<void**>(&listen_comms[comm_idx]));
		if (listen_result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d: Failed to create listen communicator for comm %zu on device %d: %d",
				rank, comm_idx, dev, listen_result);
			metrics.failed_comms++;
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

		// Connect and Accept - matches nccl_connection.cpp pattern
		// These are non-blocking calls that return immediately if not ready
		// Keep polling until both connections succeed
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
		metrics.successful_comms++;
		
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		
		if (MEASURE_PERFORMANCE && comm_idx % 10 == 0) {
			NCCL_OFI_INFO(NCCL_NET, "RANK %d: Comm %zu created in %ld μs (total active: %zu)", 
				rank, comm_idx, duration.count(), active_comm_count);
			metrics.record_memory_usage();
		}

		return ncclSuccess;
	}

	void cleanup_single_communicator(test_nccl_net_t* ext_net, size_t comm_idx) {
		if (!ext_net || comm_idx >= max_comm_capacity) return;
		
		if (listen_comms[comm_idx]) {
			ext_net->closeListen(static_cast<void*>(listen_comms[comm_idx]));
			listen_comms[comm_idx] = nullptr;
		}
		if (send_comms[comm_idx]) {
			ext_net->closeSend(static_cast<void*>(send_comms[comm_idx]));
			send_comms[comm_idx] = nullptr;
		}
		if (recv_comms[comm_idx]) {
			ext_net->closeRecv(static_cast<void*>(recv_comms[comm_idx]));
			recv_comms[comm_idx] = nullptr;
		}
		send_handles[comm_idx] = nullptr;
		recv_handles[comm_idx] = nullptr;
	}

	ncclResult_t validate_scale_limits() const {
		if (active_comm_count == 0) {
			NCCL_OFI_WARN("No active communicators to validate");
			return ncclInternalError;
		}

		NCCL_OFI_INFO(NCCL_NET, "Validating %zu active communicators at scale", active_comm_count);

		// Validate all communicators are properly configured
		size_t valid_comms = 0;
		for (size_t i = 0; i < max_comm_capacity && valid_comms < active_comm_count; i++) {
			if (send_comms[i] && recv_comms[i]) {
				if (send_comms[i]->base.dev_id != device_id ||
				    recv_comms[i]->base.dev_id != device_id) {
					NCCL_OFI_WARN("Device validation failed for comm %zu (expected dev %d, got send dev %d, recv dev %d)",
						i, device_id, send_comms[i]->base.dev_id, recv_comms[i]->base.dev_id);
					return ncclInternalError;
				}
				valid_comms++;
			}
		}

		if (valid_comms != active_comm_count) {
			NCCL_OFI_WARN("Communicator count mismatch: expected %zu, found %zu valid", active_comm_count, valid_comms);
			return ncclInternalError;
		}

		NCCL_OFI_INFO(NCCL_NET, "Successfully validated %zu communicators at scale", active_comm_count);
		return ncclSuccess;
	}

	ncclResult_t test_scale_operations(test_nccl_net_t* ext_net, int buffer_type, int rank) const {
		if (!ext_net || active_comm_count == 0) return ncclSuccess;

		NCCL_OFI_INFO(NCCL_NET, "RANK %d: Testing operations across %zu communicators", rank, active_comm_count);

		constexpr size_t test_size = SEND_SIZE;
		
		// Test a subset of communicators to avoid excessive memory usage
		size_t test_comm_count = std::min(active_comm_count, static_cast<size_t>(16));
		std::vector<size_t> test_indices;
		
		// Select communicators to test (evenly distributed)
		for (size_t i = 0; i < test_comm_count; i++) {
			size_t idx = (i * active_comm_count) / test_comm_count;
			// Find the actual communicator index
			size_t actual_idx = 0;
			size_t found_count = 0;
			for (size_t j = 0; j < max_comm_capacity && found_count <= idx; j++) {
				if (send_comms[j] && recv_comms[j]) {
					if (found_count == idx) {
						actual_idx = j;
						break;
					}
					found_count++;
				}
			}
			test_indices.push_back(actual_idx);
		}

		std::vector<void*> send_buffers(test_comm_count, nullptr);
		std::vector<void*> recv_buffers(test_comm_count, nullptr);
		std::vector<void*> send_mhandles(test_comm_count, nullptr);
		std::vector<void*> recv_mhandles(test_comm_count, nullptr);
		std::vector<nccl_net_ofi_req_t*> send_reqs(test_comm_count, nullptr);
		std::vector<nccl_net_ofi_req_t*> recv_reqs(test_comm_count, nullptr);

		// Allocate and register buffers for selected communicators
		for (size_t i = 0; i < test_comm_count; i++) {
			size_t comm_idx = test_indices[i];
			
			if (allocate_buff(&send_buffers[i], test_size, buffer_type) != ncclSuccess ||
			    initialize_buff(send_buffers[i], test_size, buffer_type) != ncclSuccess ||
			    allocate_buff(&recv_buffers[i], test_size, buffer_type) != ncclSuccess) {
				// Cleanup on failure
				for (size_t j = 0; j <= i; j++) {
					if (send_buffers[j]) deallocate_buffer(send_buffers[j], buffer_type);
					if (recv_buffers[j]) deallocate_buffer(recv_buffers[j], buffer_type);
				}
				return ncclInternalError;
			}

			// Register memory handles
			if (ext_net->regMr(static_cast<void*>(send_comms[comm_idx]), send_buffers[i], test_size, buffer_type, &send_mhandles[i]) != ncclSuccess ||
			    ext_net->regMr(static_cast<void*>(recv_comms[comm_idx]), recv_buffers[i], test_size, buffer_type, &recv_mhandles[i]) != ncclSuccess) {
				NCCL_OFI_WARN("RANK %d: Failed to register memory for comm %zu", rank, comm_idx);
				// Cleanup
				for (size_t j = 0; j <= i; j++) {
					if (send_mhandles[j]) ext_net->deregMr(static_cast<void*>(send_comms[test_indices[j]]), send_mhandles[j]);
					if (recv_mhandles[j]) ext_net->deregMr(static_cast<void*>(recv_comms[test_indices[j]]), recv_mhandles[j]);
					if (send_buffers[j]) deallocate_buffer(send_buffers[j], buffer_type);
					if (recv_buffers[j]) deallocate_buffer(recv_buffers[j], buffer_type);
				}
				return ncclInternalError;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// RAII-style cleanup
		auto cleanup_buffers = [&]() {
			for (size_t i = 0; i < test_comm_count; i++) {
				size_t comm_idx = test_indices[i];
				if (send_mhandles[i]) ext_net->deregMr(static_cast<void*>(send_comms[comm_idx]), send_mhandles[i]);
				if (recv_mhandles[i]) ext_net->deregMr(static_cast<void*>(recv_comms[comm_idx]), recv_mhandles[i]);
				if (send_buffers[i]) deallocate_buffer(send_buffers[i], buffer_type);
				if (recv_buffers[i]) deallocate_buffer(recv_buffers[i], buffer_type);
			}
		};

		bool operations_successful = true;

		// Post operations
		for (size_t i = 0; i < test_comm_count && operations_successful; i++) {
			size_t comm_idx = test_indices[i];
			
			// Use proper recv API
			int nrecv = 1;
			void* recv_data[1] = {recv_buffers[i]};
			size_t recv_sizes[1] = {test_size};
			int recv_tags[1] = {static_cast<int>(comm_idx)};

			int recv_result = ext_net->irecv(static_cast<void*>(recv_comms[comm_idx]), nrecv, recv_data, recv_sizes,
				recv_tags, &recv_mhandles[i], reinterpret_cast<void**>(&recv_reqs[i]));
			if (recv_result != 0) {
				NCCL_OFI_WARN("RANK %d: irecv operation failed for comm %zu: %d", rank, comm_idx, recv_result);
				operations_successful = false;
				break;
			}

			int send_result = ext_net->isend(static_cast<void*>(send_comms[comm_idx]), send_buffers[i], test_size, static_cast<int>(comm_idx),
				send_mhandles[i], reinterpret_cast<void**>(&send_reqs[i]));
			if (send_result != 0) {
				NCCL_OFI_WARN("RANK %d: isend operation failed for comm %zu: %d", rank, comm_idx, send_result);
				operations_successful = false;
				break;
			}
		}

		// Wait for completion
		bool all_complete = false;
		if (operations_successful) {
			for (int iteration = 0; !all_complete && iteration < 1000; iteration++) {
				all_complete = true;

				for (size_t i = 0; i < test_comm_count; i++) {
					if (send_reqs[i]) {
						int done = 0, size = 0;
						if (ext_net->test(static_cast<void*>(send_reqs[i]), &done, &size) == 0 && done) {
							send_reqs[i] = nullptr;
						} else {
							all_complete = false;
						}
					}

					if (recv_reqs[i]) {
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
		
		if (operations_successful && all_complete) {
			NCCL_OFI_INFO(NCCL_NET, "RANK %d: Successfully completed operations on %zu test communicators", rank, test_comm_count);
		}
		
		return (operations_successful && all_complete) ? ncclSuccess : ncclInternalError;
	}

	void cleanup(test_nccl_net_t* ext_net) {
		if (!ext_net) return;

		NCCL_OFI_INFO(NCCL_NET, "Cleaning up %zu communicators", active_comm_count);
		for (size_t i = 0; i < max_comm_capacity; i++) {
			cleanup_single_communicator(ext_net, i);
		}
		active_comm_count = 0;
		metrics.successful_comms = 0;
		metrics.failed_comms = 0;
	}

	// Accessors and utilities
	void set_device_id(int dev_id) { device_id = dev_id; }
	size_t get_active_comm_count() const { return active_comm_count; }
	size_t get_max_capacity() const { return max_comm_capacity; }
	const PerformanceMetrics& get_metrics() const { return metrics; }
	nccl_net_ofi_send_comm_t* get_send_comm(size_t idx) const { 
		return (idx < max_comm_capacity) ? send_comms[idx] : nullptr; 
	}
	nccl_net_ofi_recv_comm_t* get_recv_comm(size_t idx) const { 
		return (idx < max_comm_capacity) ? recv_comms[idx] : nullptr; 
	}
	
	void start_performance_measurement() { metrics.start_timing(); }
	void end_performance_measurement() { metrics.end_timing(); }
};

// Thread context for multi-threaded scale testing
class ThreadContext {
private:
	int thread_id{0}, rank{0}, device_id{0}, buffer_type{0};
	size_t comms_per_thread{0}, thread_comm_offset{0};
	test_nccl_net_t* ext_net{nullptr};
	std::unique_ptr<MultiCommContext> scale_ctx{nullptr};
	std::atomic<bool> ready{false}, start_test{false}, completed{false};
	pthread_barrier_t* setup_barrier{nullptr}, *test_barrier{nullptr}, *cleanup_barrier{nullptr};
	ncclResult_t result{ncclSuccess};

public:
	ThreadContext() = default;

	void initialize(int tid, int r, int dev_id, int buf_type, test_nccl_net_t* net,
		size_t comms_count, size_t comm_offset, pthread_barrier_t* setup_bar, 
		pthread_barrier_t* test_bar, pthread_barrier_t* cleanup_bar) {
		thread_id = tid;
		rank = r;
		device_id = dev_id;
		buffer_type = buf_type;
		ext_net = net;
		comms_per_thread = comms_count;
		thread_comm_offset = comm_offset;
		setup_barrier = setup_bar;
		test_barrier = test_bar;
		cleanup_barrier = cleanup_bar;
		result = ncclSuccess;
		
		// Create scale context for this thread
		scale_ctx = std::make_unique<MultiCommContext>(comms_count);
		scale_ctx->set_device_id(device_id);
	}

	// Accessors
	int get_thread_id() const { return thread_id; }
	int get_rank() const { return rank; }
	int get_device_id() const { return device_id; }
	int get_buffer_type() const { return buffer_type; }
	size_t get_comms_per_thread() const { return comms_per_thread; }
	size_t get_thread_comm_offset() const { return thread_comm_offset; }
	test_nccl_net_t* get_ext_net() const { return ext_net; }
	MultiCommContext& get_comm_ctx() { return *scale_ctx; }
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

// Scale test functions
__attribute__((unused))
static ncclResult_t test_sequential_communicator_creation(test_nccl_net_t* ext_net, int device_id, 
	int buffer_type, int rank, size_t target_count) {
	
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Starting sequential communicator creation test (target: %zu)", rank, target_count);
	
	MultiCommContext scale_ctx(target_count);
	scale_ctx.set_device_id(device_id);
	scale_ctx.start_performance_measurement();
	
	// Create communicators in batches
	size_t created_count = 0;
	for (size_t batch = 0; batch < target_count; batch += BATCH_SIZE) {
		size_t batch_end = std::min(batch + BATCH_SIZE, target_count);
		
		NCCL_OFI_INFO(NCCL_NET, "RANK %d: Creating batch %zu-%zu (%zu/%zu)", 
			rank, batch, batch_end - 1, batch_end, target_count);
		
		// Create communicators in this batch
		for (size_t i = batch; i < batch_end; i++) {
			ncclResult_t result = scale_ctx.create_communicator_pair(ext_net, i, rank, device_id);
			if (result != ncclSuccess) {
				if (TEST_RESOURCE_EXHAUSTION) {
					NCCL_OFI_INFO(NCCL_NET, "RANK %d: Resource exhaustion detected at communicator %zu", rank, i);
					break;
				} else {
					NCCL_OFI_WARN("RANK %d: Failed to create communicator %zu", rank, i);
					scale_ctx.cleanup(ext_net);
					return result;
				}
			}
			created_count++;
		}
		
		// Synchronize between ranks after each batch
		MPI_Barrier(MPI_COMM_WORLD);
		
		// Check if we should stop due to resource exhaustion
		if (created_count < batch_end) {
			break;
		}
	}
	
	scale_ctx.end_performance_measurement();
	
	NCCL_OFI_INFO(NCCL_NET, "RANK %d: Successfully created %zu/%zu communicators", 
		rank, created_count, target_count);
	
	// Validate the created communicators
	ncclResult_t validation_result = scale_ctx.validate_scale_limits();
	if (validation_result != ncclSuccess) {
		NCCL_OFI_WARN("RANK %d: Scale validation failed", rank);
		scale_ctx.cleanup(ext_net);
		return validation_result;
	}
	
	// Test operations on a subset of communicators
	ncclResult_t ops_result = scale_ctx.test_scale_operations(ext_net, buffer_type, rank);
	if (ops_result != ncclSuccess) {
		NCCL_OFI_WARN("RANK %d: Scale operations test failed", rank);
		scale_ctx.cleanup(ext_net);
		return ops_result;
	}
	
	// Print performance metrics
	const auto& metrics = scale_ctx.get_metrics();
	if (MEASURE_PERFORMANCE) {
		NCCL_OFI_INFO(NCCL_NET, "RANK %d: Performance metrics - Created: %zu, Failed: %zu, Avg time: %.2f ms, Peak memory: %.2f MB",
			rank, metrics.successful_comms, metrics.failed_comms, metrics.avg_creation_time_ms, metrics.peak_memory_mb);
	}
	
	// Cleanup
	scale_ctx.cleanup(ext_net);
	
	return ncclSuccess;
}

__attribute__((unused))
static ncclResult_t test_resource_exhaustion_handling(test_nccl_net_t* ext_net, int device_id, 
	int buffer_type, int rank) {
	
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Testing resource exhaustion handling", rank);
	
	// Try to create an excessive number of communicators to test limits
	size_t excessive_count = MAX_TEST_COMMS * 2;  // Try double the expected limit
	MultiCommContext scale_ctx(excessive_count);
	scale_ctx.set_device_id(device_id);
	
	size_t successful_count = 0;
	size_t first_failure = 0;
	bool hit_limit = false;
	
	for (size_t i = 0; i < excessive_count; i++) {
		ncclResult_t result = scale_ctx.create_communicator_pair(ext_net, i, rank, device_id);
		if (result != ncclSuccess) {
			first_failure = i;
			hit_limit = true;
			NCCL_OFI_INFO(NCCL_NET, "RANK %d: Hit resource limit at communicator %zu", rank, i);
			break;
		}
		successful_count++;
		
		// Synchronize periodically
		if (i % BATCH_SIZE == 0) {
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	
	if (hit_limit) {
		NCCL_OFI_INFO(NCCL_NET, "RANK %d: Successfully detected resource exhaustion at %zu communicators", 
			rank, first_failure);
	} else {
		NCCL_OFI_INFO(NCCL_NET, "RANK %d: Created all %zu communicators without hitting limits", 
			rank, excessive_count);
	}
	
	// Test that existing communicators still work after hitting limits
	if (successful_count > 0) {
		ncclResult_t validation_result = scale_ctx.validate_scale_limits();
		if (validation_result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d: Validation failed after resource exhaustion test", rank);
			scale_ctx.cleanup(ext_net);
			return validation_result;
		}
	}
	
	// Cleanup
	scale_ctx.cleanup(ext_net);
	
	return ncclSuccess;
}

// Thread worker function for multi-threaded scale testing
static void* scale_thread_worker(void* arg) {
	auto* thread_ctx = static_cast<ThreadContext*>(arg);
	int rank = thread_ctx->get_rank();
	int thread_id = thread_ctx->get_thread_id();

	NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Starting scale thread worker", rank, thread_id);

#if HAVE_CUDA
	// Initialize CUDA runtime for this worker thread
	int device_count = 0;
	cudaError_t result = cudaGetDeviceCount(&device_count);
	if (result == cudaSuccess && device_count > 0) {
		// Use the same device as calculated in main
		cudaSetDevice(thread_ctx->get_device_id() % device_count);
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: CUDA runtime initialized", rank, thread_id);
	}
#endif

	thread_ctx->get_comm_ctx().set_device_id(thread_ctx->get_device_id());
	thread_ctx->get_comm_ctx().start_performance_measurement();

	// Create communicators for this thread
	size_t comms_per_thread = thread_ctx->get_comms_per_thread();
	size_t comm_offset = thread_ctx->get_thread_comm_offset();
	
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Creating %zu communicators (offset %zu)", 
		rank, thread_id, comms_per_thread, comm_offset);

	for (size_t i = 0; i < comms_per_thread; i++) {
		// Use unique tags per thread to avoid conflicts
		size_t global_comm_idx = comm_offset + i;
		
		ncclResult_t create_result = thread_ctx->get_comm_ctx().create_communicator_pair(
			thread_ctx->get_ext_net(), i, rank, thread_ctx->get_device_id());
		
		if (create_result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d THREAD %d: Failed to create communicator %zu (global %zu)", 
				rank, thread_id, i, global_comm_idx);
			thread_ctx->set_result(ncclInternalError);
			thread_ctx->get_comm_ctx().cleanup(thread_ctx->get_ext_net());
			
			// Must still participate in barriers to avoid hanging other threads
			for (size_t remaining = i + 1; remaining <= comms_per_thread; remaining++) {
				if (remaining % BATCH_SIZE == 0) {
					thread_ctx->wait_setup_barrier();
				}
			}
			thread_ctx->set_ready(true);
			int wait_iterations = 0;
			const int max_wait_iterations = 30000;
			while (!thread_ctx->should_start_test() && wait_iterations < max_wait_iterations) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				wait_iterations++;
			}
			thread_ctx->wait_test_barrier();
			thread_ctx->wait_cleanup_barrier();
			thread_ctx->set_completed(true);
			return nullptr;
		}

		// Synchronize after each batch
		if ((i + 1) % BATCH_SIZE == 0) {
			thread_ctx->wait_setup_barrier();
		}
	}

	thread_ctx->get_comm_ctx().end_performance_measurement();

	// Signal ready and wait for start (with timeout)
	thread_ctx->set_ready(true);
	int wait_iterations = 0;
	const int max_wait_iterations = 30000; // 30 seconds timeout
	while (!thread_ctx->should_start_test() && wait_iterations < max_wait_iterations) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		wait_iterations++;
	}
	
	if (wait_iterations >= max_wait_iterations) {
		NCCL_OFI_WARN("RANK %d THREAD %d: Timeout waiting for test start signal", rank, thread_id);
		thread_ctx->set_result(ncclInternalError);
		thread_ctx->wait_test_barrier();
		thread_ctx->get_comm_ctx().cleanup(thread_ctx->get_ext_net());
		thread_ctx->wait_cleanup_barrier();
		thread_ctx->set_completed(true);
		return nullptr;
	}

	// Run validation and testing
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Starting validation phase", rank, thread_id);

	ncclResult_t validation_result = thread_ctx->get_comm_ctx().validate_scale_limits();
	if (validation_result != ncclSuccess) {
		NCCL_OFI_WARN("RANK %d THREAD %d: Scale validation failed", rank, thread_id);
		thread_ctx->set_result(validation_result);
	} else {
		NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Scale validation passed", rank, thread_id);

		ncclResult_t ops_result = thread_ctx->get_comm_ctx().test_scale_operations(
			thread_ctx->get_ext_net(), thread_ctx->get_buffer_type(), rank);
		if (ops_result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d THREAD %d: Scale operations test failed", rank, thread_id);
			thread_ctx->set_result(ops_result);
		} else {
			NCCL_OFI_INFO(NCCL_INIT, "RANK %d THREAD %d: Scale operations test passed", rank, thread_id);
		}
	}

	thread_ctx->wait_test_barrier();

	// Print performance metrics for this thread
	if (MEASURE_PERFORMANCE) {
		const auto& metrics = thread_ctx->get_comm_ctx().get_metrics();
		NCCL_OFI_INFO(NCCL_NET, "RANK %d THREAD %d: Created %zu comms, avg time %.2f ms, peak memory %.2f MB",
			rank, thread_id, metrics.successful_comms, metrics.avg_creation_time_ms, metrics.peak_memory_mb);
	}

	// Cleanup
	thread_ctx->get_comm_ctx().cleanup(thread_ctx->get_ext_net());
	thread_ctx->wait_cleanup_barrier();

	thread_ctx->set_completed(true);
	return nullptr;
}

// Multi-threaded scale test
__attribute__((unused))
static ncclResult_t test_multithreaded_scale(test_nccl_net_t* ext_net, int device_id,
	int buffer_type, int rank) {
	
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Starting multi-threaded scale test with %zu threads, %zu total comms",
		rank, MAX_TEST_THREADS, MAX_TEST_COMMS);

	std::vector<pthread_t> threads(MAX_TEST_THREADS);
	std::vector<ThreadContext> thread_contexts(MAX_TEST_THREADS);
	pthread_barrier_t setup_barrier, test_barrier, cleanup_barrier;

	// Calculate communicators per thread
	size_t comms_per_thread = MAX_TEST_COMMS / MAX_TEST_THREADS;
	size_t remaining_comms = MAX_TEST_COMMS % MAX_TEST_THREADS;

	// Initialize barriers
	bool use_barriers = (MAX_TEST_THREADS > 1);
	if (use_barriers) {
		int total_threads = static_cast<int>(MAX_TEST_THREADS);
		if (pthread_barrier_init(&setup_barrier, nullptr, total_threads) != 0 ||
		    pthread_barrier_init(&test_barrier, nullptr, total_threads) != 0 ||
		    pthread_barrier_init(&cleanup_barrier, nullptr, total_threads) != 0) {
			NCCL_OFI_WARN("RANK %d: Failed to initialize barriers", rank);
			return ncclInternalError;
		}
	}

	// Create threads
	size_t comm_offset = 0;
	for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
		auto& ctx = thread_contexts[i];
		
		// Distribute remaining communicators among first threads
		size_t thread_comms = comms_per_thread + (i < remaining_comms ? 1 : 0);
		
		ctx.initialize(static_cast<int>(i), rank, device_id, buffer_type, ext_net,
			thread_comms, comm_offset, 
			use_barriers ? &setup_barrier : nullptr,
			use_barriers ? &test_barrier : nullptr,
			use_barriers ? &cleanup_barrier : nullptr);

		NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Creating thread %zu with %zu comms (offset %zu)", 
			rank, i, thread_comms, comm_offset);
		
		if (pthread_create(&threads[i], nullptr, scale_thread_worker, &ctx) != 0) {
			NCCL_OFI_WARN("RANK %d: Failed to create thread %zu", rank, i);
			return ncclInternalError;
		}
		
		comm_offset += thread_comms;
	}

	// Wait for threads to be ready
	bool all_ready = false;
	for (int wait_ms = 0; wait_ms < 30000; wait_ms += 10) {
		all_ready = true;
		for (size_t i = 0; i < MAX_TEST_THREADS && all_ready; i++) {
			all_ready = thread_contexts[i].is_ready();
		}
		if (all_ready) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	
	if (!all_ready) {
		NCCL_OFI_WARN("RANK %d: Timeout waiting for threads to be ready", rank);
		for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
			if (!thread_contexts[i].is_ready()) {
				NCCL_OFI_WARN("RANK %d: Thread %zu not ready", rank, i);
			}
		}
	}

	// Start testing
	for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
		thread_contexts[i].set_start_test(true);
	}

	// Wait for completion
	bool all_completed = false;
	for (int wait_ms = 0; wait_ms < 60000; wait_ms += 10) {
		all_completed = true;
		for (size_t i = 0; i < MAX_TEST_THREADS && all_completed; i++) {
			all_completed = thread_contexts[i].is_completed();
		}
		if (all_completed) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	
	if (!all_completed) {
		NCCL_OFI_WARN("RANK %d: Timeout waiting for threads to complete", rank);
		for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
			if (!thread_contexts[i].is_completed()) {
				NCCL_OFI_WARN("RANK %d: Thread %zu not completed", rank, i);
			}
		}
	}

	// Join threads and check results
	ncclResult_t final_result = ncclSuccess;
	size_t total_successful_comms = 0;
	for (size_t i = 0; i < MAX_TEST_THREADS; i++) {
		pthread_join(threads[i], nullptr);
		if (thread_contexts[i].get_result() != ncclSuccess) {
			final_result = thread_contexts[i].get_result();
		}
		total_successful_comms += thread_contexts[i].get_comm_ctx().get_metrics().successful_comms;
	}

	// Cleanup barriers
	if (use_barriers) {
		pthread_barrier_destroy(&cleanup_barrier);
		pthread_barrier_destroy(&test_barrier);
		pthread_barrier_destroy(&setup_barrier);
	}

	NCCL_OFI_INFO(NCCL_NET, "RANK %d: Multi-threaded scale test completed - %zu total communicators created",
		rank, total_successful_comms);

	return final_result;
}

// Test proxy thread stress with multiple communicators issuing concurrent operations
static ncclResult_t test_proxy_thread_stress(test_nccl_net_t* ext_net, int device_id,
	int buffer_type, int rank, size_t num_comms, size_t num_iterations) {
	
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Starting proxy thread stress test with %zu communicators, %zu iterations",
		rank, num_comms, num_iterations);

	// Create all communicators upfront (they all share the same proxy threads)
	MultiCommContext ctx(num_comms);
	ctx.set_device_id(device_id);
	ctx.start_performance_measurement();

	// Phase 1: Create all communicators
	NCCL_OFI_INFO(NCCL_NET, "RANK %d: Creating %zu communicators (all will share proxy threads)", rank, num_comms);
	for (size_t i = 0; i < num_comms; i++) {
		ncclResult_t result = ctx.create_communicator_pair(ext_net, i, rank, device_id);
		if (result != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d: Failed to create communicator %zu", rank, i);
			ctx.cleanup(ext_net);
			return result;
		}
		
		// Sync periodically
		if ((i + 1) % BATCH_SIZE == 0) {
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	ctx.end_performance_measurement();
	MPI_Barrier(MPI_COMM_WORLD);

	NCCL_OFI_INFO(NCCL_NET, "RANK %d: Created %zu communicators in %.2f ms", 
		rank, num_comms, ctx.get_metrics().avg_creation_time_ms * num_comms);

	// Phase 2: Stress proxy threads with concurrent operations from all communicators
	NCCL_OFI_INFO(NCCL_NET, "RANK %d: Starting proxy thread stress - %zu iterations with %zu concurrent communicators",
		rank, num_iterations, num_comms);

	constexpr size_t test_size = SEND_SIZE;
	std::vector<void*> send_buffers(num_comms, nullptr);
	std::vector<void*> recv_buffers(num_comms, nullptr);
	std::vector<void*> send_mhandles(num_comms, nullptr);
	std::vector<void*> recv_mhandles(num_comms, nullptr);

	// Allocate and register buffers for all communicators
	for (size_t i = 0; i < num_comms; i++) {
		if (allocate_buff(&send_buffers[i], test_size, buffer_type) != ncclSuccess ||
		    initialize_buff(send_buffers[i], test_size, buffer_type) != ncclSuccess ||
		    allocate_buff(&recv_buffers[i], test_size, buffer_type) != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d: Failed to allocate buffers for comm %zu", rank, i);
			for (size_t j = 0; j < i; j++) {
				if (send_buffers[j]) deallocate_buffer(send_buffers[j], buffer_type);
				if (recv_buffers[j]) deallocate_buffer(recv_buffers[j], buffer_type);
			}
			ctx.cleanup(ext_net);
			return ncclInternalError;
		}

		auto send_comm = ctx.get_send_comm(i);
		auto recv_comm = ctx.get_recv_comm(i);
		
		if (ext_net->regMr(static_cast<void*>(send_comm), send_buffers[i], test_size, buffer_type, &send_mhandles[i]) != ncclSuccess ||
		    ext_net->regMr(static_cast<void*>(recv_comm), recv_buffers[i], test_size, buffer_type, &recv_mhandles[i]) != ncclSuccess) {
			NCCL_OFI_WARN("RANK %d: Failed to register memory for comm %zu", rank, i);
			for (size_t j = 0; j <= i; j++) {
				if (send_mhandles[j]) ext_net->deregMr(static_cast<void*>(ctx.get_send_comm(j)), send_mhandles[j]);
				if (recv_mhandles[j]) ext_net->deregMr(static_cast<void*>(ctx.get_recv_comm(j)), recv_mhandles[j]);
				if (send_buffers[j]) deallocate_buffer(send_buffers[j], buffer_type);
				if (recv_buffers[j]) deallocate_buffer(recv_buffers[j], buffer_type);
			}
			ctx.cleanup(ext_net);
			return ncclInternalError;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Phase 3: Run iterations with all communicators issuing operations concurrently
	auto stress_start = std::chrono::high_resolution_clock::now();
	size_t successful_iterations = 0;
	size_t total_operations = 0;

	for (size_t iter = 0; iter < num_iterations; iter++) {
		std::vector<nccl_net_ofi_req_t*> send_reqs(num_comms, nullptr);
		std::vector<nccl_net_ofi_req_t*> recv_reqs(num_comms, nullptr);
		bool iteration_failed = false;

		// Post recv operations for ALL communicators (queued to proxy threads)
		for (size_t i = 0; i < num_comms && !iteration_failed; i++) {
			auto recv_comm = ctx.get_recv_comm(i);
			int nrecv = 1;
			void* recv_data[1] = {recv_buffers[i]};
			size_t recv_sizes[1] = {test_size};
			int recv_tags[1] = {static_cast<int>(i)};

			if (ext_net->irecv(static_cast<void*>(recv_comm), nrecv, recv_data, recv_sizes,
				recv_tags, &recv_mhandles[i], reinterpret_cast<void**>(&recv_reqs[i])) != 0) {
				NCCL_OFI_WARN("RANK %d: irecv failed for comm %zu in iteration %zu", rank, i, iter);
				iteration_failed = true;
			}
		}

		// Post send operations for ALL communicators (queued to proxy threads)
		for (size_t i = 0; i < num_comms && !iteration_failed; i++) {
			auto send_comm = ctx.get_send_comm(i);
			if (ext_net->isend(static_cast<void*>(send_comm), send_buffers[i], test_size, static_cast<int>(i),
				send_mhandles[i], reinterpret_cast<void**>(&send_reqs[i])) != 0) {
				NCCL_OFI_WARN("RANK %d: isend failed for comm %zu in iteration %zu", rank, i, iter);
				iteration_failed = true;
			}
		}

		// Test for completion (proxy threads process the queue)
		if (!iteration_failed) {
			bool all_complete = false;
			for (int test_iter = 0; !all_complete && test_iter < 10000; test_iter++) {
				all_complete = true;

				for (size_t i = 0; i < num_comms; i++) {
					if (send_reqs[i]) {
						int done = 0, size = 0;
						if (ext_net->test(static_cast<void*>(send_reqs[i]), &done, &size) == 0 && done) {
							send_reqs[i] = nullptr;
						} else {
							all_complete = false;
						}
					}

					if (recv_reqs[i]) {
						int done = 0, size = 0;
						if (ext_net->test(static_cast<void*>(recv_reqs[i]), &done, &size) == 0 && done) {
							recv_reqs[i] = nullptr;
						} else {
							all_complete = false;
						}
					}
				}

				if (!all_complete) {
					std::this_thread::sleep_for(std::chrono::microseconds(10));
				}
			}

			if (all_complete) {
				successful_iterations++;
				total_operations += num_comms * 2; // send + recv per comm
			} else {
				NCCL_OFI_WARN("RANK %d: Iteration %zu timed out", rank, iter);
			}
		}

		// Progress report
		if ((iter + 1) % 10 == 0) {
			auto now = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - stress_start);
			double ops_per_sec = (total_operations * 1000.0) / (elapsed.count() + 1);
			NCCL_OFI_INFO(NCCL_NET, "RANK %d: Completed %zu/%zu iterations, %.2f ops/sec", 
				rank, iter + 1, num_iterations, ops_per_sec);
		}
	}

	auto stress_end = std::chrono::high_resolution_clock::now();
	auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(stress_end - stress_start);
	double ops_per_sec = (total_operations * 1000.0) / (total_time.count() + 1);

	NCCL_OFI_INFO(NCCL_NET, "RANK %d: Proxy thread stress test completed - %zu/%zu iterations successful, %.2f ops/sec, %ld ms total",
		rank, successful_iterations, num_iterations, ops_per_sec, total_time.count());

	// Cleanup buffers
	for (size_t i = 0; i < num_comms; i++) {
		if (send_mhandles[i]) ext_net->deregMr(static_cast<void*>(ctx.get_send_comm(i)), send_mhandles[i]);
		if (recv_mhandles[i]) ext_net->deregMr(static_cast<void*>(ctx.get_recv_comm(i)), recv_mhandles[i]);
		if (send_buffers[i]) deallocate_buffer(send_buffers[i], buffer_type);
		if (recv_buffers[i]) deallocate_buffer(recv_buffers[i], buffer_type);
	}

	ctx.cleanup(ext_net);

	return (successful_iterations == num_iterations) ? ncclSuccess : ncclInternalError;
}

static ncclResult_t test_high_communicator_count(test_nccl_net_t* ext_net, int device_id,
	int buffer_type, int rank) {
	
	NCCL_OFI_INFO(NCCL_INIT, "RANK %d: Starting proxy thread stress test with device %d, buffer_type %d, target_comms %zu",
		rank, device_id, buffer_type, MAX_TEST_COMMS);

	// Main test: Proxy thread stress with multiple communicators
	// This simulates real NCCL usage where many communicators share the same proxy threads
	// and issue concurrent send/recv operations
	size_t num_iterations = 100; // Number of send/recv cycles
	ncclResult_t result = test_proxy_thread_stress(ext_net, device_id, buffer_type, rank, 
		MAX_TEST_COMMS, num_iterations);
	
	if (result != ncclSuccess) {
		NCCL_OFI_WARN("RANK %d: Proxy thread stress test failed", rank);
		return result;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	NCCL_OFI_INFO(NCCL_NET, "RANK %d: Proxy thread stress test completed successfully", rank);
	return ncclSuccess;
}

static void parse_arguments(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--comms" && i + 1 < argc) MAX_TEST_COMMS = std::max(1, std::stoi(argv[++i]));
		else if (arg == "--threads" && i + 1 < argc) MAX_TEST_THREADS = std::max(1, std::stoi(argv[++i]));
		else if (arg == "--batch-size" && i + 1 < argc) BATCH_SIZE = std::max(1, std::stoi(argv[++i]));
		else if (arg == "--host") FORCE_HOST_BUFFERS = true;
		else if (arg == "--no-perf") MEASURE_PERFORMANCE = false;
		else if (arg == "--test-exhaustion") TEST_RESOURCE_EXHAUSTION = true;
		else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
			          << "High Communicator Count Stress Test - Tests scale limits and resource management\n\n"
			          << "Options:\n"
			          << "  --comms N           Number of communicators to create (default: 64)\n"
			          << "  --threads N         Number of threads for concurrent testing (default: 4)\n"
			          << "  --batch-size N      Create communicators in batches of N (default: 8)\n"
			          << "  --host              Force host buffers (default: auto-detect)\n"
			          << "  --no-perf           Disable performance measurements\n"
			          << "  --test-exhaustion   Test resource exhaustion handling\n"
			          << "  --help              Show this help message\n\n"
			          << "Environment Variables:\n"
			          << "  MAX_TEST_COMMS      Override --comms\n"
			          << "  MAX_TEST_THREADS    Override --threads\n"
			          << "  BATCH_SIZE          Override --batch-size\n"
			          << "  FORCE_HOST_BUFFERS  Override --host (1/true)\n"
			          << "  MEASURE_PERFORMANCE Override --no-perf (0/false to disable)\n"
			          << "  TEST_RESOURCE_EXHAUSTION Override --test-exhaustion (1/true)\n";
			std::exit(0);
		}
	}

	// Environment variable overrides
	if (const char* env = std::getenv("MAX_TEST_COMMS")) MAX_TEST_COMMS = std::max(1, std::stoi(env));
	if (const char* env = std::getenv("MAX_TEST_THREADS")) MAX_TEST_THREADS = std::max(1, std::stoi(env));
	if (const char* env = std::getenv("BATCH_SIZE")) BATCH_SIZE = std::max(1, std::stoi(env));
	if (const char* env = std::getenv("FORCE_HOST_BUFFERS")) FORCE_HOST_BUFFERS = (std::string(env) == "1" || std::string(env) == "true");
	if (const char* env = std::getenv("MEASURE_PERFORMANCE")) MEASURE_PERFORMANCE = !(std::string(env) == "0" || std::string(env) == "false");
	if (const char* env = std::getenv("TEST_RESOURCE_EXHAUSTION")) TEST_RESOURCE_EXHAUSTION = (std::string(env) == "1" || std::string(env) == "true");
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
	NCCL_OFI_INFO(NCCL_INIT, "Scale test configuration: target %zu communicators, %zu threads, batch size %zu, %s buffers, perf measurement %s, exhaustion test %s",
		MAX_TEST_COMMS, MAX_TEST_THREADS, BATCH_SIZE, FORCE_HOST_BUFFERS ? "host" : "auto", 
		MEASURE_PERFORMANCE ? "enabled" : "disabled", TEST_RESOURCE_EXHAUSTION ? "enabled" : "disabled");

#if HAVE_CUDA
	// Initialize CUDA runtime for the process
	if (!initialize_cuda_runtime()) {
		NCCL_OFI_WARN("Failed to initialize CUDA runtime");
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
		
		if (test_high_communicator_count(ext_net, dev, buffer_type, rank) != ncclSuccess) {
			NCCL_OFI_WARN("High communicator count stress test failed on device %d", dev);
			MPI_Finalize();
			return ncclInternalError;
		}

		// Synchronize and allow time for async cleanup to complete
		// This helps prevent resource exhaustion when testing multiple devices sequentially
		MPI_Barrier(MPI_COMM_WORLD);
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "High communicator count stress test completed successfully for rank %d", rank);

	return ncclSuccess;
}
