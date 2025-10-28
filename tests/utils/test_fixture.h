/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef TEST_FIXTURE_H_
#define TEST_FIXTURE_H_

#include <vector>
#include <mpi.h>
#include <nccl/net.h>

#include "nccl_ofi.h"

// Forward declarations
typedef ncclNet_v9_t test_nccl_net_t;
typedef ncclNetDeviceHandle_v9_t test_nccl_net_device_handle_t;

/**
 * TestFixture - Base class for NCCL OFI plugin functional tests
 * 
 * This class provides common setup and teardown functionality for tests,
 * including MPI initialization, plugin initialization, device enumeration,
 * and CUDA device setup. Tests inherit from this class and override the
 * run_test() method to implement their specific test logic.
 * 
 * Example usage:
 * 
 *   class MyTest : public test_fixture {
 *   protected:
 *       ncclResult_t run_test() override {
 *           // Test implementation here
 *           return ncclSuccess;
 *       }
 *   };
 *   
 *   int main(int argc, char* argv[]) {
 *       MyTest test;
 *       return test.run(argc, argv);
 *   }
 */
class test_fixture {
public:
	/**
	 * Constructor - Initializes member variables
	 */
	test_fixture();
	
	/**
	 * Destructor - Performs cleanup including MPI finalization
	 */
	virtual ~test_fixture();
	
	/**
	 * Main test entry point - call this from main()
	 * 
	 * This method handles all initialization (MPI, plugin, devices, CUDA),
	 * calls the virtual run_test() method, and performs cleanup.
	 * 
	 * @param argc Command-line argument count
	 * @param argv Command-line argument values
	 * @return Exit code (0 for success, non-zero for failure)
	 */
	int run(int argc, char* argv[]);
	
protected:
	/**
	 * Test implementation - override this in your test
	 * 
	 * This is the main test logic that subclasses must implement.
	 * The fixture handles all setup and teardown, so this method
	 * can focus on the specific test scenario.
	 * 
	 * @return ncclResult_t indicating test success or failure
	 */
	virtual ncclResult_t run_test() = 0;
	
	/**
	 * Optional: override for custom device iteration
	 * 
	 * By default, tests run on all devices. Override this method
	 * to skip certain devices based on custom criteria.
	 * 
	 * @param dev Device index
	 * @return true if the device should be tested, false to skip
	 */
	virtual bool should_test_device(int dev);
	
	// Accessor methods for test state
	
	/**
	 * Get MPI rank of current process
	 */
	int rank() const { return rank_; }
	
	/**
	 * Get total number of MPI ranks
	 */
	int size() const { return size_; }
	
	/**
	 * Get number of network devices
	 */
	int ndev() const { return ndev_; }
	
	/**
	 * Get pointer to NCCL plugin interface
	 */
	test_nccl_net_t* plugin() const { return ext_net_; }
	
	/**
	 * Check if a device supports GPUDirect RDMA
	 * 
	 * @param dev Device index
	 * @return true if device supports GDR, false otherwise
	 */
	bool is_gdr_supported(int dev) const;
	
	/**
	 * Get CUDA device assigned to this process
	 * 
	 * @return CUDA device index, or -1 if CUDA is not initialized
	 */
	int cuda_device() const { return cuda_device_; }
	
	/**
	 * Get processor name for this process
	 * 
	 * @return Processor name string
	 */
	const char* processor_name() const { return processor_name_; }
	
	// Note: Connection management utilities (setup_connection, cleanup_connection)
	// are now standalone functions in ../utils/data_transfer.h
	// The fixture provides convenience wrappers that pass fixture state
	
private:
	// MPI state
	int rank_;                                    // MPI rank
	int size_;                                    // MPI size
	char processor_name_[MPI_MAX_PROCESSOR_NAME]; // Processor name
	bool mpi_initialized_;                        // Track if MPI was initialized
	
	// Plugin state
	void* net_plugin_lib_;                        // Plugin library handle (for dlclose)
	test_nccl_net_t* ext_net_;                    // Plugin interface pointer
	int ndev_;                                    // Number of devices
	std::vector<bool> gdr_support_;               // GDR support per device
	
	// CUDA state
	int cuda_device_;                             // CUDA device index (-1 if not initialized)
	
	// Initialization helpers
	ncclResult_t initialize_mpi(int argc, char* argv[]);
	ncclResult_t initialize_plugin();
	ncclResult_t enumerate_devices();
	ncclResult_t initialize_cuda();
	
	// CUDA helpers
	int calculate_local_rank();
	int select_cuda_device(int local_rank);
};

#endif // TEST_FIXTURE_H_
