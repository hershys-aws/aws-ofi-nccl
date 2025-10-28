/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * @file test_fixture.cpp
 * @brief Test fixture implementations for NCCL OFI plugin functional tests
 *
 * This file contains the implementation of the test_fixture class,
 * providing reusable test infrastructure for NCCL OFI plugin tests.
 */

#include "test_fixture.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <dlfcn.h>

#include <mpi.h>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"

// Macro for stringification
#define STR2(v)		#v
#define STR(v)		STR2(v)

// Plugin symbol name
#define NCCL_PLUGIN_SYMBOL ncclNetPlugin_v9

/**
 * @brief Namespace for test framework implementation details
 */
namespace {

/**
 * Global tag counter for unique MPI tag generation
 * Uses atomic to ensure thread-safe tag allocation
 */
std::atomic<int> g_tag_counter(0);

/**
 * Calculate local rank based on processor names
 * 
 * Ranks on the same physical node will have the same processor name.
 * This function counts how many ranks with the same processor name
 * come before the current rank to determine the local rank.
 * 
 * @param processor_name Current processor name
 * @param rank Current MPI rank
 * @param size Total number of MPI ranks
 * @return Local rank (0-based index among ranks on same node)
 */
int calculate_local_rank_helper(const char* processor_name, int rank, int size)
{
	int local_rank = 0;
	
	// Use C++ vector for memory management
	std::vector<char> all_names_buffer(size * MPI_MAX_PROCESSOR_NAME);
	char (*all_names)[MPI_MAX_PROCESSOR_NAME] = 
		reinterpret_cast<char (*)[MPI_MAX_PROCESSOR_NAME]>(all_names_buffer.data());
	
	// Gather all processor names
	MPI_Allgather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
	              all_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
	              MPI_COMM_WORLD);
	
	// Count ranks with same processor name before current rank
	for (int i = 0; i < rank; i++) {
		if (strcmp(all_names[i], processor_name) == 0) {
			local_rank++;
		}
	}
	
	return local_rank;
}

} // anonymous namespace

// test_fixture Implementation

test_fixture::test_fixture()
	: rank_(0)
	, size_(0)
	, mpi_initialized_(false)
	, net_plugin_lib_(nullptr)
	, ext_net_(nullptr)
	, ndev_(0)
	, cuda_device_(-1)
{
	memset(processor_name_, 0, MPI_MAX_PROCESSOR_NAME);
}

test_fixture::~test_fixture()
{
	// Cleanup is performed here to ensure proper resource deallocation
	// even if the test exits early
	
	// Close plugin library handle to avoid resource leak
	if (net_plugin_lib_ != nullptr) {
		dlclose(net_plugin_lib_);
		net_plugin_lib_ = nullptr;
		ext_net_ = nullptr; // Pointer is now invalid
	}
	
	if (mpi_initialized_) {
		MPI_Finalize();
		mpi_initialized_ = false;
	}
}

int test_fixture::run(int argc, char* argv[])
{
	ncclResult_t res = ncclSuccess;
	
	// Set up logging
	ofi_log_function = logger;
	
	// Initialize MPI
	res = initialize_mpi(argc, argv);
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to initialize MPI");
		return static_cast<int>(res);
	}
	
	NCCL_OFI_INFO(NCCL_INIT, "Process rank %d started on %s", rank_, processor_name_);
	
	// Initialize plugin
	res = initialize_plugin();
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to initialize plugin");
		return static_cast<int>(res);
	}
	
	NCCL_OFI_INFO(NCCL_INIT, "NCCLNet device used on %s is %s",
	              processor_name_, ext_net_->name);
	
	// Enumerate devices
	res = enumerate_devices();
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to enumerate devices");
		return static_cast<int>(res);
	}
	
	NCCL_OFI_INFO(NCCL_INIT, "Received %d network devices", ndev_);
	
	// Initialize CUDA
	res = initialize_cuda();
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Failed to initialize CUDA");
		return static_cast<int>(res);
	}
	
	if (cuda_device_ >= 0) {
		NCCL_OFI_INFO(NCCL_INIT, "Rank %d using CUDA device %d", rank_, cuda_device_);
	}
	
	// Run the test
	res = run_test();
	if (res != ncclSuccess) {
		NCCL_OFI_WARN("Test failed with error: %d", res);
		return static_cast<int>(res);
	}
	
	// Final barrier to ensure all ranks complete
	MPI_Barrier(MPI_COMM_WORLD);
	
	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank_);
	
	return 0;
}

bool test_fixture::should_test_device(int dev)
{
	// By default, test all devices
	return true;
}

bool test_fixture::is_gdr_supported(int dev) const
{
	if (dev < 0 || dev >= static_cast<int>(gdr_support_.size())) {
		return false;
	}
	return gdr_support_[dev];
}

ncclResult_t test_fixture::initialize_mpi(int argc, char* argv[])
{
	// Initialize MPI
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Init failed");
		return ncclSystemError;
	}
	
	mpi_initialized_ = true;
	
	// Get rank and size
	if (MPI_Comm_rank(MPI_COMM_WORLD, &rank_) != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Comm_rank failed");
		return ncclSystemError;
	}
	
	if (MPI_Comm_size(MPI_COMM_WORLD, &size_) != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Comm_size failed");
		return ncclSystemError;
	}
	
	// Get processor name
	int proc_name_len;
	if (MPI_Get_processor_name(processor_name_, &proc_name_len) != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Get_processor_name failed");
		return ncclSystemError;
	}
	
	return ncclSuccess;
}

ncclResult_t test_fixture::initialize_plugin()
{
	// Load plugin library
	net_plugin_lib_ = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (net_plugin_lib_ == nullptr) {
		NCCL_OFI_WARN("Unable to load libnccl-net.so: %s", dlerror());
		return ncclInternalError;
	}
	
	// Get plugin interface symbol
	ext_net_ = (test_nccl_net_t *)dlsym(net_plugin_lib_, STR(NCCL_PLUGIN_SYMBOL));
	if (ext_net_ == nullptr) {
		NCCL_OFI_WARN("NetPlugin, could not find %s symbol", STR(NCCL_PLUGIN_SYMBOL));
		dlclose(net_plugin_lib_);
		net_plugin_lib_ = nullptr;
		return ncclInternalError;
	}
	
	// Initialize plugin
	OFINCCLCHECK(ext_net_->init(logger));
	
	return ncclSuccess;
}

ncclResult_t test_fixture::enumerate_devices()
{
	// Query number of devices
	OFINCCLCHECK(ext_net_->devices(&ndev_));
	
	if (ndev_ <= 0) {
		NCCL_OFI_WARN("No network devices found");
		return ncclSystemError;
	}
	
	// Allocate GDR support array
	gdr_support_.resize(ndev_, false);
	
	// Get properties for each device
	for (int dev = 0; dev < ndev_; dev++) {
		test_nccl_properties_t props = {};
		OFINCCLCHECK(ext_net_->getProperties(dev, &props));
		
		// Print device properties
		print_dev_props(dev, &props);
		
		// Store GDR support flag
		gdr_support_[dev] = is_gdr_supported_nic(props.ptrSupport);
		
		if (gdr_support_[dev]) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			              "Network supports communication using CUDA buffers. Dev: %d", dev);
		}
	}
	
	return ncclSuccess;
}

ncclResult_t test_fixture::initialize_cuda()
{
#if HAVE_CUDA
	// Calculate local rank
	int local_rank = calculate_local_rank();
	
	// Select CUDA device
	int device = select_cuda_device(local_rank);
	
	if (device < 0) {
		NCCL_OFI_WARN("Invalid CUDA device %d selected", device);
		return ncclInvalidArgument;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Initializing CUDA with device %d", device);
	
	// Set CUDA device
	cudaError_t cuda_ret = cudaSetDevice(device);
	if (cuda_ret != cudaSuccess) {
		NCCL_OFI_WARN("Failed to set CUDA device %d: %s",
		              device, cudaGetErrorString(cuda_ret));
		return ncclUnhandledCudaError;
	}
	
	// Perform a dummy operation to initialize CUDA runtime
	// cudaFree(0) is a common idiom for forcing CUDA initialization
	cuda_ret = cudaFree(0);
	if (cuda_ret != cudaSuccess) {
		NCCL_OFI_WARN("Failed to initialize CUDA runtime: %s",
		              cudaGetErrorString(cuda_ret));
		return ncclUnhandledCudaError;
	}
	
	cuda_device_ = device;
	
	NCCL_OFI_TRACE(NCCL_NET, "CUDA initialized successfully with device %d", device);
	return ncclSuccess;
#else
	// CUDA not available, skip initialization
	cuda_device_ = -1;
	return ncclSuccess;
#endif
}

int test_fixture::calculate_local_rank()
{
	NCCL_OFI_TRACE(NCCL_NET, "Calculating local rank for rank %d/%d on %s",
	               rank_, size_, processor_name_);
	
	int local_rank = calculate_local_rank_helper(processor_name_, rank_, size_);
	
	NCCL_OFI_TRACE(NCCL_NET, "Local rank calculated: %d", local_rank);
	return local_rank;
}

int test_fixture::select_cuda_device(int local_rank)
{
#if HAVE_CUDA
	if (local_rank < 0) {
		NCCL_OFI_WARN("Invalid local_rank %d passed to select_cuda_device", local_rank);
		return 0;
	}
	
	NCCL_OFI_TRACE(NCCL_NET, "Selecting CUDA device for local rank %d", local_rank);
	
	int num_devices = 0;
	cudaError_t cuda_ret = cudaGetDeviceCount(&num_devices);
	
	if (cuda_ret != cudaSuccess) {
		NCCL_OFI_WARN("Failed to get CUDA device count: %s",
		              cudaGetErrorString(cuda_ret));
		return 0;
	}
	
	if (num_devices <= 0) {
		NCCL_OFI_WARN("No CUDA devices found");
		return 0;
	}
	
	// Map local rank to device index using modulo arithmetic
	// This ensures round-robin distribution of ranks to devices
	int device_index = local_rank % num_devices;
	
	NCCL_OFI_TRACE(NCCL_NET, "Selected CUDA device %d (local_rank=%d, num_devices=%d)",
	               device_index, local_rank, num_devices);
	
	return device_index;
#else
	return 0;
#endif
}

int generate_unique_tag()
{
	// Atomically increment and return the tag counter
	// This ensures thread-safe tag generation
	return g_tag_counter.fetch_add(1);
}

// Connection management utilities moved to data_transfer.h

// Resource Tracking Utilities Implementation
// Moved to resource_tracker.cpp
