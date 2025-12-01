/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL's connection establishment and
 * data transfer APIs
 */

#include "config.h"
#include <array>
#include "test-common.h"
#include <mutex>

class MessageTransferTest : public TestScenario {

public:
	explicit MessageTransferTest(size_t num_threads = 0) : TestScenario("NCCL Message Transfer Test", num_threads) {
		// Test all sizes
	}

	ncclResult_t setup(ThreadContext& ctx) override {
		// Initialize CUDA context BEFORE establishing connections
		OFINCCLCHECK(init_cuda_for_thread(0));

		// Base class establishes all connections and populates ctx
		OFINCCLCHECK(TestScenario::setup(ctx));

		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: rank %d completed connection setup", ctx.thread_id, ctx.rank);
		
		// Initialize buffer storage with test sizes
		ctx.initialize_buffer_storage(SEND_RECV_SIZES);
		
		// Allocate buffers for each device
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			auto dev = (ctx.rank == 1) ? ctx.ndev - dev_idx - 1 : dev_idx;
			auto gdr_support = get_support_gdr(ext_net);
			int buffer_type = gdr_support[dev] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
			OFINCCLCHECK(ctx.allocate_test_buffers(dev_idx, buffer_type));
		}
		
		return ncclSuccess;
	}

	ncclResult_t run(ThreadContext& ctx) override {
		// Get device properties
		test_nccl_properties_t props = {};
		OFINCCLCHECK(ext_net->getProperties(0, &props));

		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: rank %d running with %zu devices", ctx.thread_id, ctx.rank, ctx.lcomms.size());

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			NCCL_OFI_INFO(NCCL_NET, "Rank %d starting test on device index %lu", ctx.rank, dev_idx);

			// Run test over each of the sizes
			for (size_t size_idx = 0; size_idx < SEND_RECV_SIZES.size(); size_idx++) {
				const auto& [send_size, recv_size] = SEND_RECV_SIZES[size_idx];
				NCCL_OFI_INFO(NCCL_NET, "Rank %d testing size %lu->%lu on dev %lu", ctx.rank, send_size, recv_size, dev_idx);
				
				// Skip if send size > recv size and regIsGlobal == 0
				if (props.regIsGlobal == 0 && send_size > recv_size) {
					if (ctx.rank == 0) {
						NCCL_OFI_TRACE(NCCL_NET, "Skipping test for send size %zu > recv size %zu", send_size, recv_size);
					}
					continue;
				}

				// Run test with pre-allocated buffers (poll_until_complete ensures no data in flight)
				OFINCCLCHECK(ctx.send_receive_test(dev_idx, size_idx));
				
				// Workaround: Ensure both sender and receiver complete before validation
				MPI_Barrier(ctx.thread_comm);
				
				NCCL_OFI_INFO(NCCL_NET, "Rank %d completed size %lu->%lu on dev %lu", ctx.rank, send_size, recv_size, dev_idx);
			}
			
			NCCL_OFI_INFO(NCCL_NET, "Rank %d completed all sizes on device index %lu", ctx.rank, dev_idx);
		}
		return ncclSuccess;
	}

	ncclResult_t teardown(ThreadContext& ctx) override {
		// Deallocate buffers for each device
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			OFINCCLCHECK(ctx.deallocate_test_buffers(dev_idx));
		}
		
		// Base class cleans up connections
		return TestScenario::teardown(ctx);
	}

private:
	/* Data sizes for testing various thresholds */
	std::vector<std::pair<size_t, size_t>> SEND_RECV_SIZES {
		{512, 512},
		{4 * 1024, 4 * 1024},
		{16 * 1024, 16 * 1024},
		{1024 * 1024, 1024 * 1024},
		{5 * 1024, 4 * 1024},
		{17 * 1024, 16 * 1024},
		{2 * 1024 * 1024, 1024 * 1024},
		{4 * 1024, 5 * 1024},
		{16 * 1024, 17 * 1024},
		{1024 * 1024, 2 * 1024 * 1024}
	};
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	MessageTransferTest test;  // Single-threaded test only
	// MessageTransferTest mt_test(4);  // Disabled for now
	suite.add(&test);
	// suite.add(&mt_test);  // Disabled for now
	return suite.run_all();
}

