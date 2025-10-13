/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates high communicator count stress scenarios for NCCL OFI plugin
 * Tests creation of many communicators sequentially and validates resource management
 */

#include "config.h"

#include "test-common.h"
#include <chrono>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#include <dirent.h>

#define MAX_STRESS_COMMS 20
#define MIN_STRESS_COMMS 10
#define PERFORMANCE_THRESHOLD_MS 100.0
#define MEMORY_CHECK_INTERVAL 5

struct stress_test_context {
	std::vector<nccl_net_ofi_send_comm_t*> sComms;
	std::vector<nccl_net_ofi_recv_comm_t*> rComms;
	std::vector<nccl_net_ofi_listen_comm_t*> lComms;
	std::vector<test_nccl_net_device_handle_t*> s_handles;
	std::vector<test_nccl_net_device_handle_t*> r_handles;
	int active_comm_count;
	int device_id;
};

struct resource_usage {
	long memory_kb;
	int open_files;
	double cpu_time_ms;
};

struct performance_metrics {
	double total_creation_time_ms;
	double total_cleanup_time_ms;
	double avg_creation_time_ms;
	double avg_cleanup_time_ms;
	int successful_comms;
	int failed_comms;
	resource_usage initial_resources;
	resource_usage peak_resources;
	resource_usage final_resources;
	std::vector<double> creation_times_ms;
	std::vector<resource_usage> resource_snapshots;
};

/* Forward declarations */
static void init_stress_context(stress_test_context *ctx, int device_id);
static void cleanup_stress_context(stress_test_context *ctx, test_nccl_net_t *extNet);
static resource_usage get_current_resource_usage();
static void print_resource_usage(const resource_usage &usage, const char *label, int rank);
static void analyze_performance_degradation(const performance_metrics &metrics, int rank);
static void validate_resource_cleanup(const performance_metrics &metrics, int rank);
static void print_performance_metrics(const performance_metrics &metrics, int rank);

static void init_stress_context(stress_test_context *ctx, int device_id) {
	ctx->sComms.clear();
	ctx->rComms.clear();
	ctx->lComms.clear();
	ctx->s_handles.clear();
	ctx->r_handles.clear();
	ctx->active_comm_count = 0;
	ctx->device_id = device_id;
}

static void cleanup_stress_context(stress_test_context *ctx, test_nccl_net_t *extNet) {
	// Clean up in reverse order
	for (int i = ctx->active_comm_count - 1; i >= 0; i--) {
		if (ctx->lComms[i]) {
			extNet->closeListen((void *)ctx->lComms[i]);
			ctx->lComms[i] = NULL;
		}
		if (ctx->sComms[i]) {
			extNet->closeSend((void *)ctx->sComms[i]);
			ctx->sComms[i] = NULL;
		}
		if (ctx->rComms[i]) {
			extNet->closeRecv((void *)ctx->rComms[i]);
			ctx->rComms[i] = NULL;
		}
	}
	ctx->active_comm_count = 0;
}

static resource_usage get_current_resource_usage() {
	resource_usage usage = {};
	
	// Get memory usage
	struct rusage rusage_info;
	if (getrusage(RUSAGE_SELF, &rusage_info) == 0) {
		usage.memory_kb = rusage_info.ru_maxrss;
		usage.cpu_time_ms = (rusage_info.ru_utime.tv_sec * 1000.0) + (rusage_info.ru_utime.tv_usec / 1000.0);
	}
	
	// Get open file count (approximate)
	char proc_path[256];
	snprintf(proc_path, sizeof(proc_path), "/proc/%d/fd", getpid());
	
	// Count files in /proc/pid/fd directory
	DIR *fd_dir = opendir(proc_path);
	if (fd_dir) {
		struct dirent *entry;
		int count = 0;
		while ((entry = readdir(fd_dir)) != NULL) {
			if (entry->d_name[0] != '.') {
				count++;
			}
		}
		closedir(fd_dir);
		usage.open_files = count;
	}
	
	return usage;
}

static void print_resource_usage(const resource_usage &usage, const char *label, int rank) {
	NCCL_OFI_INFO(NCCL_NET, "Rank %d %s Resources:", rank, label);
	NCCL_OFI_INFO(NCCL_NET, "  Memory: %ld KB", usage.memory_kb);
	NCCL_OFI_INFO(NCCL_NET, "  Open files: %d", usage.open_files);
	NCCL_OFI_INFO(NCCL_NET, "  CPU time: %.2f ms", usage.cpu_time_ms);
}

static void analyze_performance_degradation(const performance_metrics &metrics, int rank) {
	if (metrics.creation_times_ms.size() < 2) return;
	
	// Calculate performance degradation
	double first_half_avg = 0.0;
	double second_half_avg = 0.0;
	int half_point = metrics.creation_times_ms.size() / 2;
	
	for (int i = 0; i < half_point; i++) {
		first_half_avg += metrics.creation_times_ms[i];
	}
	first_half_avg /= half_point;
	
	for (size_t i = half_point; i < metrics.creation_times_ms.size(); i++) {
		second_half_avg += metrics.creation_times_ms[i];
	}
	second_half_avg /= (metrics.creation_times_ms.size() - half_point);
	
	double degradation_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100.0;
	
	NCCL_OFI_INFO(NCCL_NET, "Rank %d Performance Analysis:", rank);
	NCCL_OFI_INFO(NCCL_NET, "  First half avg creation time: %.2f ms", first_half_avg);
	NCCL_OFI_INFO(NCCL_NET, "  Second half avg creation time: %.2f ms", second_half_avg);
	NCCL_OFI_INFO(NCCL_NET, "  Performance degradation: %.1f%%", degradation_percent);
	
	if (degradation_percent > 50.0) {
		NCCL_OFI_WARN("Significant performance degradation detected (%.1f%%) - may indicate resource pressure", 
			degradation_percent);
	}
}

static void validate_resource_cleanup(const performance_metrics &metrics, int rank) {
	long memory_growth = metrics.final_resources.memory_kb - metrics.initial_resources.memory_kb;
	int fd_growth = metrics.final_resources.open_files - metrics.initial_resources.open_files;
	
	NCCL_OFI_INFO(NCCL_NET, "Rank %d Resource Cleanup Validation:", rank);
	NCCL_OFI_INFO(NCCL_NET, "  Memory growth: %ld KB", memory_growth);
	NCCL_OFI_INFO(NCCL_NET, "  File descriptor growth: %d", fd_growth);
	
	// Allow some tolerance for memory growth due to allocator behavior
	if (memory_growth > 1024) { // More than 1MB growth
		NCCL_OFI_WARN("Potential memory leak detected: %ld KB growth after cleanup", memory_growth);
	}
	
	if (fd_growth > 2) { // More than 2 FDs left open
		NCCL_OFI_WARN("Potential file descriptor leak detected: %d FDs not cleaned up", fd_growth);
	}
	
	if (memory_growth <= 1024 && fd_growth <= 2) {
		NCCL_OFI_INFO(NCCL_NET, "Resource cleanup validation passed");
	}
}

static void print_performance_metrics(const performance_metrics &metrics, int rank) {
	NCCL_OFI_INFO(NCCL_NET, "Rank %d Performance Metrics:", rank);
	NCCL_OFI_INFO(NCCL_NET, "  Total creation time: %.2f ms", metrics.total_creation_time_ms);
	NCCL_OFI_INFO(NCCL_NET, "  Total cleanup time: %.2f ms", metrics.total_cleanup_time_ms);
	NCCL_OFI_INFO(NCCL_NET, "  Average creation time per comm: %.2f ms", metrics.avg_creation_time_ms);
	NCCL_OFI_INFO(NCCL_NET, "  Average cleanup time per comm: %.2f ms", metrics.avg_cleanup_time_ms);
	NCCL_OFI_INFO(NCCL_NET, "  Successful communicators: %d", metrics.successful_comms);
	NCCL_OFI_INFO(NCCL_NET, "  Failed communicators: %d", metrics.failed_comms);
	
	print_resource_usage(metrics.initial_resources, "Initial", rank);
	print_resource_usage(metrics.peak_resources, "Peak", rank);
	print_resource_usage(metrics.final_resources, "Final", rank);
	
	analyze_performance_degradation(metrics, rank);
	validate_resource_cleanup(metrics, rank);
}

int main(int argc, char* argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, size, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev;
	stress_test_context stress_ctx;
	test_nccl_net_t *extNet = NULL;
	performance_metrics perf_metrics = {};

	/* Declare variables to avoid goto crossing initialization */
	int dev = 0;
	auto start_time = std::chrono::high_resolution_clock::now();
	auto end_time = std::chrono::high_resolution_clock::now();
	auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	auto cleanup_start_time = std::chrono::high_resolution_clock::now();
	auto cleanup_end_time = std::chrono::high_resolution_clock::now();
	auto cleanup_duration = std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end_time - cleanup_start_time);

	ofi_log_function = logger;

	/* Indicates if NICs support GPUDirect */
	int *test_support_gdr = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The stress_scale functional test should be run with exactly two ranks.",
			size);
		res = ncclInvalidArgument;
		goto exit;
	}
	if (!(0 <= rank && rank <= 1)) {
		NCCL_OFI_WARN("World size was %d, but was local rank is %d. "
			      "MPI is behaving strangely, cannot continue.",
			      size, rank);
		res = ncclInvalidArgument;
		goto exit;
	}

	MPI_Get_processor_name(name, &proc_name);

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL) {
		res = ncclInternalError;
		goto exit;
	}

	/* Init API */
	OFINCCLCHECKGOTO(extNet->init(logger), res, exit);
	NCCL_OFI_INFO(NCCL_INIT, "Process rank %d started. NCCLNet device used on %s is %s.",
		      rank, name, extNet->name);

	/* Devices API */
	OFINCCLCHECKGOTO(extNet->devices(&ndev), res, exit);
	NCCL_OFI_INFO(NCCL_INIT, "Received %d network devices", ndev);

	test_support_gdr = (int *)malloc(sizeof(int) * ndev);
	if (test_support_gdr == NULL) {
		NCCL_OFI_WARN("Failed to allocate memory");
		res = ncclInternalError;
		goto exit;
	}

	/* Get Properties for the device */
	for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {
		test_nccl_properties_t props = {};
		OFINCCLCHECKGOTO(extNet->getProperties(dev_idx, &props), res, exit);
		print_dev_props(dev_idx, &props);

		/* Set CUDA support */
		test_support_gdr[dev_idx] = is_gdr_supported_nic(props.ptrSupport);
	}

	/* Test stress scenario on first device */
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses device %d for stress testing", rank, dev);

	if (test_support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Network supports communication using CUDA buffers. Dev: %d", dev);
	}

	init_stress_context(&stress_ctx, dev);

	// Resize vectors to accommodate maximum communicators
	stress_ctx.sComms.resize(MAX_STRESS_COMMS, nullptr);
	stress_ctx.rComms.resize(MAX_STRESS_COMMS, nullptr);
	stress_ctx.lComms.resize(MAX_STRESS_COMMS, nullptr);
	stress_ctx.s_handles.resize(MAX_STRESS_COMMS, nullptr);
	stress_ctx.r_handles.resize(MAX_STRESS_COMMS, nullptr);

	NCCL_OFI_INFO(NCCL_NET, "Starting stress test with up to %d communicators", MAX_STRESS_COMMS);

	// Take initial resource snapshot
	perf_metrics.initial_resources = get_current_resource_usage();
	perf_metrics.peak_resources = perf_metrics.initial_resources;
	print_resource_usage(perf_metrics.initial_resources, "Initial", rank);

	start_time = std::chrono::high_resolution_clock::now();

	/* Create multiple communicators sequentially */
	for (int comm_idx = 0; comm_idx < MAX_STRESS_COMMS; comm_idx++) {
		char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
		char handle[NCCL_NET_HANDLE_MAXSIZE] = {};

		auto comm_start_time = std::chrono::high_resolution_clock::now();

		NCCL_OFI_TRACE(NCCL_NET, "Rank %d creating communicator %d", rank, comm_idx);

		/* Listen API */
		ncclResult_t listen_res = extNet->listen(dev, (void *)&handle, (void **)&stress_ctx.lComms[comm_idx]);
		if (listen_res != ncclSuccess) {
			NCCL_OFI_WARN("Failed to create listen communicator %d: %d", comm_idx, listen_res);
			perf_metrics.failed_comms++;
			break;
		}

		if (rank == 0) {
			int peer_rank = 1;

			/* MPI send */
			MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, comm_idx, MPI_COMM_WORLD);

			/* MPI recv */
			MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			/* Connect and Accept */
			while (stress_ctx.sComms[comm_idx] == NULL || stress_ctx.rComms[comm_idx] == NULL) {
				/* Connect API */
				if (stress_ctx.sComms[comm_idx] == NULL) {
					ncclResult_t connect_res = extNet->connect(dev, (void *)src_handle, 
						(void **)&stress_ctx.sComms[comm_idx], &stress_ctx.s_handles[comm_idx]);
					if (connect_res != ncclSuccess) {
						NCCL_OFI_WARN("Failed to connect communicator %d: %d", comm_idx, connect_res);
						perf_metrics.failed_comms++;
						goto cleanup_partial;
					}
				}

				/* Accept API */
				if (stress_ctx.rComms[comm_idx] == NULL) {
					ncclResult_t accept_res = extNet->accept((void *)stress_ctx.lComms[comm_idx], 
						(void **)&stress_ctx.rComms[comm_idx], &stress_ctx.r_handles[comm_idx]);
					if (accept_res != ncclSuccess) {
						NCCL_OFI_WARN("Failed to accept communicator %d: %d", comm_idx, accept_res);
						perf_metrics.failed_comms++;
						goto cleanup_partial;
					}
				}
			}
		} else {
			int peer_rank = 0;

			/* MPI recv */
			MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			/* MPI send */
			MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, comm_idx, MPI_COMM_WORLD);

			/* Connect and Accept */
			while (stress_ctx.sComms[comm_idx] == NULL || stress_ctx.rComms[comm_idx] == NULL) {
				/* Connect API */
				if (stress_ctx.sComms[comm_idx] == NULL) {
					ncclResult_t connect_res = extNet->connect(dev, (void *)src_handle, 
						(void **)&stress_ctx.sComms[comm_idx], &stress_ctx.s_handles[comm_idx]);
					if (connect_res != ncclSuccess) {
						NCCL_OFI_WARN("Failed to connect communicator %d: %d", comm_idx, connect_res);
						perf_metrics.failed_comms++;
						goto cleanup_partial;
					}
				}

				/* Accept API */
				if (stress_ctx.rComms[comm_idx] == NULL) {
					ncclResult_t accept_res = extNet->accept((void *)stress_ctx.lComms[comm_idx], 
						(void **)&stress_ctx.rComms[comm_idx], &stress_ctx.r_handles[comm_idx]);
					if (accept_res != ncclSuccess) {
						NCCL_OFI_WARN("Failed to accept communicator %d: %d", comm_idx, accept_res);
						perf_metrics.failed_comms++;
						goto cleanup_partial;
					}
				}
			}
		}

		auto comm_end_time = std::chrono::high_resolution_clock::now();
		auto comm_duration = std::chrono::duration_cast<std::chrono::microseconds>(comm_end_time - comm_start_time);
		double comm_time_ms = comm_duration.count() / 1000.0;

		perf_metrics.total_creation_time_ms += comm_time_ms;
		perf_metrics.successful_comms++;
		perf_metrics.creation_times_ms.push_back(comm_time_ms);
		stress_ctx.active_comm_count++;

		// Take resource snapshot periodically
		if (comm_idx % MEMORY_CHECK_INTERVAL == 0) {
			resource_usage current_usage = get_current_resource_usage();
			perf_metrics.resource_snapshots.push_back(current_usage);
			
			// Update peak resources
			if (current_usage.memory_kb > perf_metrics.peak_resources.memory_kb) {
				perf_metrics.peak_resources.memory_kb = current_usage.memory_kb;
			}
			if (current_usage.open_files > perf_metrics.peak_resources.open_files) {
				perf_metrics.peak_resources.open_files = current_usage.open_files;
			}
			
			NCCL_OFI_TRACE(NCCL_NET, "Rank %d resource snapshot at comm %d: %ld KB, %d FDs", 
				rank, comm_idx, current_usage.memory_kb, current_usage.open_files);
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank %d successfully created communicator %d in %.2f ms", 
			rank, comm_idx, comm_time_ms);

		// Synchronize between ranks after each communicator creation
		MPI_Barrier(MPI_COMM_WORLD);

		// Check if we've reached minimum successful communicators and should continue
		if (comm_idx >= MIN_STRESS_COMMS - 1) {
			// Test if we're approaching resource limits by checking creation time
			if (comm_time_ms > PERFORMANCE_THRESHOLD_MS) {
				NCCL_OFI_INFO(NCCL_NET, "Rank %d stopping at %d communicators due to performance degradation (%.2f ms > %.2f ms)", 
					rank, comm_idx + 1, comm_time_ms, PERFORMANCE_THRESHOLD_MS);
				break;
			}
			
			// Check for excessive resource usage
			resource_usage current_usage = get_current_resource_usage();
			long memory_growth = current_usage.memory_kb - perf_metrics.initial_resources.memory_kb;
			if (memory_growth > 100000) { // More than 100MB growth
				NCCL_OFI_INFO(NCCL_NET, "Rank %d stopping at %d communicators due to excessive memory usage (%ld KB)", 
					rank, comm_idx + 1, memory_growth);
				break;
			}
		}
	}

	end_time = std::chrono::high_resolution_clock::now();
	total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	perf_metrics.total_creation_time_ms = total_duration.count() / 1000.0;

	if (perf_metrics.successful_comms > 0) {
		perf_metrics.avg_creation_time_ms = perf_metrics.total_creation_time_ms / perf_metrics.successful_comms;
	}

	NCCL_OFI_INFO(NCCL_NET, "Rank %d created %d communicators successfully, %d failed", 
		rank, perf_metrics.successful_comms, perf_metrics.failed_comms);

cleanup_partial:
	/* Cleanup all created communicators */
	NCCL_OFI_INFO(NCCL_NET, "Rank %d starting cleanup of %d communicators", rank, stress_ctx.active_comm_count);
	cleanup_start_time = std::chrono::high_resolution_clock::now();
	
	cleanup_stress_context(&stress_ctx, extNet);
	
	cleanup_end_time = std::chrono::high_resolution_clock::now();
	cleanup_duration = std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end_time - cleanup_start_time);
	perf_metrics.total_cleanup_time_ms = cleanup_duration.count() / 1000.0;
	
	if (perf_metrics.successful_comms > 0) {
		perf_metrics.avg_cleanup_time_ms = perf_metrics.total_cleanup_time_ms / perf_metrics.successful_comms;
	}

	// Take final resource snapshot
	perf_metrics.final_resources = get_current_resource_usage();

	print_performance_metrics(perf_metrics, rank);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	
	if (perf_metrics.successful_comms >= MIN_STRESS_COMMS) {
		NCCL_OFI_INFO(NCCL_NET, "Stress test completed successfully for rank %d", rank);
	} else {
		NCCL_OFI_WARN("Stress test failed to create minimum required communicators (%d) for rank %d", 
			MIN_STRESS_COMMS, rank);
		res = ncclInternalError;
	}

exit:
	if (test_support_gdr) {
		free(test_support_gdr);
		test_support_gdr = NULL;
	}

	return res;
}