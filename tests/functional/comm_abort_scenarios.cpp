/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL communicator abort scenarios
 * Tests graceful handling of communicator termination at various lifecycle stages
 * and validates that other communicators remain functional after abort scenarios
 */

#include "config.h"

#include "test-common.h"

#define MAX_TEST_COMMS 4

struct multi_comm_context {
	nccl_net_ofi_send_comm_t *sComms[MAX_TEST_COMMS];
	nccl_net_ofi_recv_comm_t *rComms[MAX_TEST_COMMS];
	nccl_net_ofi_listen_comm_t *lComms[MAX_TEST_COMMS];
	test_nccl_net_device_handle_t *s_handles[MAX_TEST_COMMS];
	test_nccl_net_device_handle_t *r_handles[MAX_TEST_COMMS];
	int active_comm_count;
	int device_id;
};

enum abort_scenario {
	ABORT_DURING_CONNECT,
	ABORT_DURING_SEND,
	ABORT_DURING_RECV,
	ABORT_DURING_OPERATIONS,
	ABORT_AFTER_OPERATIONS
};

static ncclResult_t init_multi_comm_context(multi_comm_context *ctx, int device_id)
{
	if (!ctx) return ncclInvalidArgument;
	
	memset(ctx, 0, sizeof(*ctx));
	ctx->device_id = device_id;
	ctx->active_comm_count = 0;
	
	return ncclSuccess;
}

static ncclResult_t create_communicator_pair(multi_comm_context *ctx, int comm_idx, 
					     test_nccl_net_t *extNet, int rank, int size)
{
	ncclResult_t res = ncclSuccess;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	
	if (!ctx || comm_idx >= MAX_TEST_COMMS || !extNet) {
		return ncclInvalidArgument;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Creating communicator pair %d on device %d", 
		      comm_idx, ctx->device_id);

	/* Listen API */
	OFINCCLCHECKGOTO(extNet->listen(ctx->device_id, (void *)&handle, 
					(void **)&ctx->lComms[comm_idx]), res, exit);

	if (rank == 0) {
		int peer_rank = 1;

		/* MPI send handle */
		MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 
			 comm_idx, MPI_COMM_WORLD);

		/* MPI recv handle */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
			 peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		NCCL_OFI_INFO(NCCL_INIT, "Comm %d: Establishing connections", comm_idx);

		while (ctx->sComms[comm_idx] == NULL || ctx->rComms[comm_idx] == NULL) {
			/* Connect API */
			if (ctx->sComms[comm_idx] == NULL) {
				OFINCCLCHECKGOTO(extNet->connect(ctx->device_id, (void *)src_handle, 
								(void **)&ctx->sComms[comm_idx], 
								&ctx->s_handles[comm_idx]), res, exit);
			}

			/* Accept API */
			if (ctx->rComms[comm_idx] == NULL) {
				OFINCCLCHECKGOTO(extNet->accept((void *)ctx->lComms[comm_idx], 
							       (void **)&ctx->rComms[comm_idx], 
							       &ctx->r_handles[comm_idx]), res, exit);
			}
		}
	} else {
		int peer_rank = 0;

		/* MPI recv handle */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
			 peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* MPI send handle */
		MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
			 peer_rank, comm_idx, MPI_COMM_WORLD);

		NCCL_OFI_INFO(NCCL_INIT, "Comm %d: Establishing connections", comm_idx);

		while (ctx->sComms[comm_idx] == NULL || ctx->rComms[comm_idx] == NULL) {
			/* Connect API */
			if (ctx->sComms[comm_idx] == NULL) {
				OFINCCLCHECKGOTO(extNet->connect(ctx->device_id, (void *)src_handle, 
								(void **)&ctx->sComms[comm_idx], 
								&ctx->s_handles[comm_idx]), res, exit);
			}

			/* Accept API */
			if (ctx->rComms[comm_idx] == NULL) {
				OFINCCLCHECKGOTO(extNet->accept((void *)ctx->lComms[comm_idx], 
							       (void **)&ctx->rComms[comm_idx], 
							       &ctx->r_handles[comm_idx]), res, exit);
			}
		}
	}

	ctx->active_comm_count++;
	NCCL_OFI_INFO(NCCL_INIT, "Successfully created communicator pair %d", comm_idx);

exit:
	return res;
}

static ncclResult_t test_abort_during_connect(multi_comm_context *ctx, int comm_idx, 
					     test_nccl_net_t *extNet, int rank, int size)
{
	ncclResult_t res = ncclSuccess;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	
	if (!ctx || comm_idx >= MAX_TEST_COMMS || !extNet) {
		return ncclInvalidArgument;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Testing abort during connection establishment for communicator %d", comm_idx);

	/* Listen API */
	OFINCCLCHECKGOTO(extNet->listen(ctx->device_id, (void *)&handle, 
					(void **)&ctx->lComms[comm_idx]), res, exit);

	if (rank == 0) {
		int peer_rank = 1;

		/* MPI send handle */
		MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 
			 comm_idx, MPI_COMM_WORLD);

		/* MPI recv handle */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
			 peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		NCCL_OFI_INFO(NCCL_INIT, "Comm %d: Starting connection, will abort during process", comm_idx);

		/* Start connection but abort before completion */
		OFINCCLCHECKGOTO(extNet->connect(ctx->device_id, (void *)src_handle, 
						(void **)&ctx->sComms[comm_idx], 
						&ctx->s_handles[comm_idx]), res, exit);

		/* Simulate abort during connection establishment */
		NCCL_OFI_INFO(NCCL_INIT, "Simulating abort during connection establishment");
		goto exit; /* Simulate abrupt termination */
	} else {
		int peer_rank = 0;

		/* MPI recv handle */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
			 peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* MPI send handle */
		MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
			 peer_rank, comm_idx, MPI_COMM_WORLD);

		NCCL_OFI_INFO(NCCL_INIT, "Comm %d: Starting connection, will abort during process", comm_idx);

		/* Start connection but abort before completion */
		OFINCCLCHECKGOTO(extNet->connect(ctx->device_id, (void *)src_handle, 
						(void **)&ctx->sComms[comm_idx], 
						&ctx->s_handles[comm_idx]), res, exit);

		/* Simulate abort during connection establishment */
		NCCL_OFI_INFO(NCCL_INIT, "Simulating abort during connection establishment");
		goto exit; /* Simulate abrupt termination */
	}

exit:
	return res;
}

static ncclResult_t test_send_recv_operations(multi_comm_context *ctx, int comm_idx, 
					      test_nccl_net_t *extNet, int rank, 
					      abort_scenario abort_timing)
{
	ncclResult_t res = ncclSuccess;
	void *send_buf = NULL, *recv_buf = NULL;
	void *send_mhandle = NULL, *recv_mhandle = NULL;
	void *send_req = NULL, *recv_req = NULL;
	int done = 0;
	size_t buffer_size = SEND_SIZE;
	int buffer_type = NCCL_PTR_HOST;
	
	if (!ctx || comm_idx >= MAX_TEST_COMMS || !extNet) {
		return ncclInvalidArgument;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Testing send/recv operations on communicator %d", comm_idx);

	/* Allocate buffers */
	OFINCCLCHECKGOTO(allocate_buff(&send_buf, buffer_size, buffer_type), res, exit);
	OFINCCLCHECKGOTO(allocate_buff(&recv_buf, buffer_size, buffer_type), res, exit);
	
	/* Initialize send buffer */
	OFINCCLCHECKGOTO(initialize_buff(send_buf, buffer_size, buffer_type), res, exit);

	/* Register memory */
	OFINCCLCHECKGOTO(extNet->regMr((void *)ctx->sComms[comm_idx], send_buf, buffer_size, 
				       buffer_type, &send_mhandle), res, exit);
	OFINCCLCHECKGOTO(extNet->regMr((void *)ctx->rComms[comm_idx], recv_buf, buffer_size, 
				       buffer_type, &recv_mhandle), res, exit);

	if (rank == 0) {
		/* Post receive first */
		OFINCCLCHECKGOTO(extNet->irecv((void *)ctx->rComms[comm_idx], 1, &recv_buf, 
					       &buffer_size, &buffer_type, &recv_mhandle, &recv_req), res, exit);

		/* Test abort during recv if specified */
		if (abort_timing == ABORT_DURING_RECV) {
			NCCL_OFI_INFO(NCCL_INIT, "Simulating abort during recv operation");
			goto exit; /* Simulate abrupt termination */
		}

		/* Send data */
		OFINCCLCHECKGOTO(extNet->isend((void *)ctx->sComms[comm_idx], send_buf, buffer_size, 
					       buffer_type, send_mhandle, &send_req), res, exit);

		/* Test abort during send if specified */
		if (abort_timing == ABORT_DURING_SEND) {
			NCCL_OFI_INFO(NCCL_INIT, "Simulating abort during send operation");
			goto exit; /* Simulate abrupt termination */
		}

		/* Wait for operations to complete */
		while (!done) {
			OFINCCLCHECKGOTO(extNet->test(send_req, &done, NULL), res, exit);
		}
		done = 0;
		while (!done) {
			OFINCCLCHECKGOTO(extNet->test(recv_req, &done, NULL), res, exit);
		}
	} else {
		/* Post send first */
		OFINCCLCHECKGOTO(extNet->isend((void *)ctx->sComms[comm_idx], send_buf, buffer_size, 
					       buffer_type, send_mhandle, &send_req), res, exit);

		/* Test abort during send if specified */
		if (abort_timing == ABORT_DURING_SEND) {
			NCCL_OFI_INFO(NCCL_INIT, "Simulating abort during send operation");
			goto exit; /* Simulate abrupt termination */
		}

		/* Receive data */
		OFINCCLCHECKGOTO(extNet->irecv((void *)ctx->rComms[comm_idx], 1, &recv_buf, 
					       &buffer_size, &buffer_type, &recv_mhandle, &recv_req), res, exit);

		/* Test abort during recv if specified */
		if (abort_timing == ABORT_DURING_RECV) {
			NCCL_OFI_INFO(NCCL_INIT, "Simulating abort during recv operation");
			goto exit; /* Simulate abrupt termination */
		}

		/* Wait for operations to complete */
		while (!done) {
			OFINCCLCHECKGOTO(extNet->test(send_req, &done, NULL), res, exit);
		}
		done = 0;
		while (!done) {
			OFINCCLCHECKGOTO(extNet->test(recv_req, &done, NULL), res, exit);
		}
	}

	NCCL_OFI_INFO(NCCL_INIT, "Send/recv operations completed successfully on communicator %d", comm_idx);

exit:
	/* Cleanup memory registrations */
	if (send_mhandle) {
		extNet->deregMr((void *)ctx->sComms[comm_idx], send_mhandle);
	}
	if (recv_mhandle) {
		extNet->deregMr((void *)ctx->rComms[comm_idx], recv_mhandle);
	}

	/* Cleanup buffers */
	if (send_buf) {
		deallocate_buffer(send_buf, buffer_type);
	}
	if (recv_buf) {
		deallocate_buffer(recv_buf, buffer_type);
	}

	return res;
}

static ncclResult_t abort_communicator(multi_comm_context *ctx, int comm_idx, 
				       test_nccl_net_t *extNet, abort_scenario scenario)
{
	ncclResult_t res = ncclSuccess;
	
	if (!ctx || comm_idx >= MAX_TEST_COMMS || !extNet) {
		return ncclInvalidArgument;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Aborting communicator %d with scenario %d", 
		      comm_idx, scenario);

	/* Force close communicator components abruptly */
	if (ctx->lComms[comm_idx]) {
		OFINCCLCHECK(extNet->closeListen((void *)ctx->lComms[comm_idx]));
		ctx->lComms[comm_idx] = NULL;
	}
	
	if (ctx->sComms[comm_idx]) {
		OFINCCLCHECK(extNet->closeSend((void *)ctx->sComms[comm_idx]));
		ctx->sComms[comm_idx] = NULL;
	}
	
	if (ctx->rComms[comm_idx]) {
		OFINCCLCHECK(extNet->closeRecv((void *)ctx->rComms[comm_idx]));
		ctx->rComms[comm_idx] = NULL;
	}

	ctx->active_comm_count--;
	NCCL_OFI_INFO(NCCL_INIT, "Successfully aborted communicator %d", comm_idx);

	return res;
}

static ncclResult_t validate_surviving_communicators(multi_comm_context *ctx, 
						    int aborted_comm_idx, 
						    test_nccl_net_t *extNet, int rank)
{
	ncclResult_t res = ncclSuccess;
	
	NCCL_OFI_INFO(NCCL_INIT, "Validating surviving communicators after abort of comm %d", 
		      aborted_comm_idx);

	/* Check that other communicators are still functional by performing operations */
	for (int i = 0; i < MAX_TEST_COMMS; i++) {
		if (i == aborted_comm_idx) continue;
		
		if (ctx->sComms[i] && ctx->rComms[i]) {
			NCCL_OFI_INFO(NCCL_INIT, "Testing functionality of surviving communicator %d", i);
			
			/* Perform a simple send/recv operation to validate functionality */
			OFINCCLCHECKGOTO(test_send_recv_operations(ctx, i, extNet, rank, ABORT_AFTER_OPERATIONS), res, exit);
			
			NCCL_OFI_INFO(NCCL_INIT, "Communicator %d is still active and functional", i);
		}
	}

exit:
	return res;
}

static ncclResult_t test_system_recovery_after_abort(multi_comm_context *ctx, 
						     test_nccl_net_t *extNet, 
						     int rank, int size, int dev)
{
	ncclResult_t res = ncclSuccess;
	
	NCCL_OFI_INFO(NCCL_INIT, "Testing system recovery: creating new communicators after abort");

	/* Test that new communicators can be created after abort scenarios */
	int recovery_comm_idx = 2; /* Use a different index for recovery test */
	
	OFINCCLCHECKGOTO(create_communicator_pair(ctx, recovery_comm_idx, extNet, rank, size), res, exit);
	
	NCCL_OFI_INFO(NCCL_INIT, "Successfully created new communicator %d after abort scenario", recovery_comm_idx);

	/* Test that the new communicator is functional */
	OFINCCLCHECKGOTO(test_send_recv_operations(ctx, recovery_comm_idx, extNet, rank, ABORT_AFTER_OPERATIONS), res, exit);
	
	NCCL_OFI_INFO(NCCL_INIT, "New communicator %d is fully functional after system recovery", recovery_comm_idx);

exit:
	return res;
}

static ncclResult_t validate_error_handling(multi_comm_context *ctx, int aborted_comm_idx, 
					    test_nccl_net_t *extNet)
{
	ncclResult_t res = ncclSuccess;
	
	NCCL_OFI_INFO(NCCL_INIT, "Validating proper error handling for aborted communicator %d", aborted_comm_idx);

	/* Verify that the aborted communicator is properly marked as inactive */
	if (ctx->sComms[aborted_comm_idx] == NULL && 
	    ctx->rComms[aborted_comm_idx] == NULL && 
	    ctx->lComms[aborted_comm_idx] == NULL) {
		NCCL_OFI_INFO(NCCL_INIT, "Aborted communicator %d properly cleaned up", aborted_comm_idx);
	} else {
		NCCL_OFI_WARN("Aborted communicator %d not properly cleaned up", aborted_comm_idx);
		res = ncclSystemError;
	}

	/* Verify active communicator count is correct */
	int expected_active = 0;
	for (int i = 0; i < MAX_TEST_COMMS; i++) {
		if (ctx->sComms[i] && ctx->rComms[i]) {
			expected_active++;
		}
	}

	if (ctx->active_comm_count == expected_active) {
		NCCL_OFI_INFO(NCCL_INIT, "Active communicator count is correct: %d", ctx->active_comm_count);
	} else {
		NCCL_OFI_WARN("Active communicator count mismatch: expected %d, got %d", 
			      expected_active, ctx->active_comm_count);
		res = ncclSystemError;
	}

	return res;
}

static ncclResult_t cleanup_multi_comm_context(multi_comm_context *ctx, test_nccl_net_t *extNet)
{
	ncclResult_t res = ncclSuccess;
	
	if (!ctx || !extNet) return ncclInvalidArgument;

	NCCL_OFI_INFO(NCCL_INIT, "Cleaning up remaining communicators");

	for (int i = 0; i < MAX_TEST_COMMS; i++) {
		if (ctx->lComms[i]) {
			OFINCCLCHECK(extNet->closeListen((void *)ctx->lComms[i]));
			ctx->lComms[i] = NULL;
		}
		
		if (ctx->sComms[i]) {
			OFINCCLCHECK(extNet->closeSend((void *)ctx->sComms[i]));
			ctx->sComms[i] = NULL;
		}
		
		if (ctx->rComms[i]) {
			OFINCCLCHECK(extNet->closeRecv((void *)ctx->rComms[i]));
			ctx->rComms[i] = NULL;
		}
	}

	ctx->active_comm_count = 0;
	return res;
}

int main(int argc, char* argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, size, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev;
	test_nccl_net_t *extNet = NULL;
	multi_comm_context comm_ctx = {};

	/* Declare variables to avoid goto crossing initialization */
	int dev = 0;
	abort_scenario scenarios[] = {ABORT_DURING_CONNECT, ABORT_DURING_SEND, ABORT_DURING_RECV, ABORT_DURING_OPERATIONS};
	int num_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

	ofi_log_function = logger;

	/* Indicates if NICs support GPUDirect */
	int *test_support_gdr = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The comm_abort_scenarios functional test should be run with exactly two ranks.",
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

	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d testing abort scenarios on device %d", rank, dev);

	if (test_support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Network supports communication using CUDA buffers. Dev: %d", dev);
	}

	/* Initialize multi-communicator context */
	res = init_multi_comm_context(&comm_ctx, dev);
	if (res != ncclSuccess) goto exit;

	for (int scenario_idx = 0; scenario_idx < num_scenarios; scenario_idx++) {
		abort_scenario current_scenario = scenarios[scenario_idx];
		
		NCCL_OFI_INFO(NCCL_INIT, "Testing abort scenario %d: %s", scenario_idx,
			      (current_scenario == ABORT_DURING_CONNECT) ? "ABORT_DURING_CONNECT" :
			      (current_scenario == ABORT_DURING_SEND) ? "ABORT_DURING_SEND" :
			      (current_scenario == ABORT_DURING_RECV) ? "ABORT_DURING_RECV" :
			      "ABORT_DURING_OPERATIONS");

		/* Create multiple communicator pairs for this scenario */
		NCCL_OFI_INFO(NCCL_INIT, "Creating %d communicator pairs for scenario %d", MAX_TEST_COMMS, scenario_idx);
		
		for (int i = 0; i < MAX_TEST_COMMS; i++) {
			if (current_scenario == ABORT_DURING_CONNECT && i == 1) {
				/* Test abort during connection establishment for communicator 1 */
				OFINCCLCHECKGOTO(test_abort_during_connect(&comm_ctx, i, extNet, rank, size), res, exit);
				/* Force cleanup of partially created communicator */
				abort_communicator(&comm_ctx, i, extNet, current_scenario);
			} else {
				OFINCCLCHECKGOTO(create_communicator_pair(&comm_ctx, i, extNet, rank, size), res, exit);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}

		NCCL_OFI_INFO(NCCL_INIT, "Created communicator pairs for scenario %d", scenario_idx);

		/* Test operations and abort scenarios */
		if (current_scenario == ABORT_DURING_SEND || current_scenario == ABORT_DURING_RECV) {
			/* Test send/recv operations with abort timing */
			for (int i = 0; i < MAX_TEST_COMMS; i++) {
				if (comm_ctx.sComms[i] && comm_ctx.rComms[i]) {
					if (i == 1) {
						/* This will simulate abort during send/recv */
						test_send_recv_operations(&comm_ctx, i, extNet, rank, current_scenario);
						/* Force cleanup after simulated abort */
						abort_communicator(&comm_ctx, i, extNet, current_scenario);
					} else {
						/* Normal operations for other communicators */
						test_send_recv_operations(&comm_ctx, i, extNet, rank, ABORT_AFTER_OPERATIONS);
					}
				}
			}
		} else if (current_scenario == ABORT_DURING_OPERATIONS) {
			/* Test abort during normal operations */
			int abort_comm_idx = 1;
			NCCL_OFI_INFO(NCCL_INIT, "Testing abort during operations: aborting communicator %d", abort_comm_idx);
			
			OFINCCLCHECKGOTO(abort_communicator(&comm_ctx, abort_comm_idx, extNet, current_scenario), res, exit);
		}
		
		/* Validate that other communicators remain functional */
		OFINCCLCHECKGOTO(validate_surviving_communicators(&comm_ctx, 1, extNet, rank), res, exit);
		
		/* Validate proper error handling and system state */
		OFINCCLCHECKGOTO(validate_error_handling(&comm_ctx, 1, extNet), res, exit);
		
		/* Test system recovery by creating new communicators */
		OFINCCLCHECKGOTO(test_system_recovery_after_abort(&comm_ctx, extNet, rank, size, dev), res, exit);

		/* Cleanup for next scenario */
		OFINCCLCHECKGOTO(cleanup_multi_comm_context(&comm_ctx, extNet), res, exit);
		OFINCCLCHECKGOTO(init_multi_comm_context(&comm_ctx, dev), res, exit);

		MPI_Barrier(MPI_COMM_WORLD);
		NCCL_OFI_INFO(NCCL_INIT, "Completed abort scenario %d", scenario_idx);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Cleanup remaining communicators */
	OFINCCLCHECKGOTO(cleanup_multi_comm_context(&comm_ctx, extNet), res, exit);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Abort scenarios test completed successfully for rank %d", rank);

exit:
	if (test_support_gdr) {
		free(test_support_gdr);
		test_support_gdr = NULL;
	}

	/* Emergency cleanup if needed */
	cleanup_multi_comm_context(&comm_ctx, extNet);

	return res;
}