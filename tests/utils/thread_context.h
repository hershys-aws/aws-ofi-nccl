/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef THREAD_CONTEXT_H_
#define THREAD_CONTEXT_H_

#include <atomic>
#include <pthread.h>

#include <nccl/net.h>
#include "nccl_ofi.h"

// Forward declarations
typedef ncclNet_v9_t test_nccl_net_t;

/**
 * thread_context - Thread-specific state for multi-threaded tests
 * 
 * This class encapsulates all state needed by a worker thread in multi-threaded
 * tests, including thread identification, network interface, synchronization
 * barriers, and result tracking.
 */
class thread_context {
private:
	int thread_id{0}, rank{0}, device_id{0}, buffer_type{0};
	test_nccl_net_t* ext_net{nullptr};
	void* user_data{nullptr};
	std::atomic<bool> ready{false}, start_test{false}, completed{false};
	pthread_barrier_t* setup_barrier{nullptr}, *test_barrier{nullptr}, *cleanup_barrier{nullptr};
	ncclResult_t result{ncclSuccess};

public:
	thread_context() = default;

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
	void* get_user_data() const { return user_data; }
	void set_user_data(void* data) { user_data = data; }
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

#endif // THREAD_CONTEXT_H_
