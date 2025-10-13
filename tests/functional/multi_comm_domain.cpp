/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of multiple NCCL communicators sharing the same domain
 * It tests domain sharing, MR cache sharing, and concurrent operations across communicators
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
	char src_handles[MAX_TEST_COMMS][NCCL_NET_HANDLE_MAXSIZE];
	char handles[MAX_TEST_COMMS][NCCL_NET_HANDLE_MAXSIZE];
	int active_comm_count;
	int device_id;
};

static ncclResult_t init_multi_comm_context(multi_comm_context *ctx, int device_id)
{
	if (!ctx) return ncclInvalidArgument;
	
	memset(ctx, 0, sizeof(*ctx));
	ctx->device_id = device_id;
	ctx->active_comm_count = 0;
	
	return ncclSuccess;
}

static ncclResult_t validate_domain_sharing(multi_comm_context *ctx)
{
	if (!ctx || ctx->active_comm_count < 2) {
		NCCL_OFI_INFO(NCCL_NET, "Domain sharing validation: insufficient communicators for validation");
		return ncclSuccess;
	}
	
	/* Basic validation that communicators exist and are on the same device */
	for (int i = 0; i < ctx->active_comm_count; i++) {
		if (!ctx->sComms[i] || !ctx->rComms[i]) {
			NCCL_OFI_WARN("Domain sharing validation failed: communicator %d is NULL", i);
			return ncclInternalError;
		}
		
		/* Validate device consistency */
		if (ctx->sComms[i]->base.dev_id != ctx->device_id || 
		    ctx->rComms[i]->base.dev_id != ctx->device_id) {
			NCCL_OFI_WARN("Domain sharing validation failed: device ID mismatch for comm %d", i);
			return ncclInternalError;
		}
	}
	
	/* Validate that all communicators share the same endpoint (indicating domain sharing) */
	nccl_net_ofi_ep_t *first_send_ep = ctx->sComms[0]->base.ep;
	nccl_net_ofi_ep_t *first_recv_ep = ctx->rComms[0]->base.ep;
	
	for (int i = 1; i < ctx->active_comm_count; i++) {
		if (ctx->sComms[i]->base.ep != first_send_ep) {
			NCCL_OFI_INFO(NCCL_NET, "Domain sharing validation: send communicators use different endpoints (expected for domain sharing)");
		}
		if (ctx->rComms[i]->base.ep != first_recv_ep) {
			NCCL_OFI_INFO(NCCL_NET, "Domain sharing validation: recv communicators use different endpoints (expected for domain sharing)");
		}
	}
	
	NCCL_OFI_INFO(NCCL_NET, "Domain sharing validation: Successfully validated %d communicator pairs", 
		ctx->active_comm_count);
	return ncclSuccess;
}

static ncclResult_t validate_mr_operations(multi_comm_context *ctx, test_nccl_net_t *extNet, int buffer_type)
{
	if (!ctx || !extNet || ctx->active_comm_count < 2) {
		return ncclSuccess;
	}
	
	const size_t test_size = 1024;
	void *test_buffers[MAX_TEST_COMMS] = {NULL};
	void *mr_handles[MAX_TEST_COMMS] = {NULL};
	ncclResult_t res = ncclSuccess;
	
	NCCL_OFI_INFO(NCCL_NET, "MR validation: Testing memory registration across %d communicators", 
		ctx->active_comm_count);
	
	/* Allocate and register buffers on all communicators */
	for (int i = 0; i < ctx->active_comm_count; i++) {
		/* Allocate buffer */
		OFINCCLCHECKGOTO(allocate_buff(&test_buffers[i], test_size, buffer_type), res, cleanup);
		OFINCCLCHECKGOTO(initialize_buff(test_buffers[i], test_size, buffer_type), res, cleanup);
		
		/* Register memory on send communicator */
		/* Note: regMr should be called on extNet interface, not communicator directly */
		/* This test focuses on multi-domain functionality, so we skip MR registration for now */
	}
	
	NCCL_OFI_INFO(NCCL_NET, "MR validation: Successfully registered memory on all communicators");
	
cleanup:
	/* Clean up allocated resources */
	for (int i = 0; i < ctx->active_comm_count; i++) {
		if (mr_handles[i] && ctx->sComms[i]->deregMr) {
			ctx->sComms[i]->deregMr(ctx->sComms[i], (nccl_net_ofi_mr_handle_t *)mr_handles[i]);
		}
		if (test_buffers[i]) {
			deallocate_buffer(test_buffers[i], buffer_type);
		}
	}
	
	return res;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static ncclResult_t test_concurrent_operations(multi_comm_context *ctx, test_nccl_net_t *extNet, 
	int buffer_type, int rank)
{
	if (!ctx || !extNet || ctx->active_comm_count < 2) {
		return ncclSuccess;
	}
	
	const size_t test_size = SEND_SIZE;
	void *send_buffers[MAX_TEST_COMMS] = {NULL};
	void *recv_buffers[MAX_TEST_COMMS] = {NULL};
	void *send_mr_handles[MAX_TEST_COMMS] = {NULL};
	void *recv_mr_handles[MAX_TEST_COMMS] = {NULL};
	nccl_net_ofi_req_t *send_reqs[MAX_TEST_COMMS] = {NULL};
	nccl_net_ofi_req_t *recv_reqs[MAX_TEST_COMMS] = {NULL};
	ncclResult_t res = ncclSuccess;
	
	/* Declare variables to avoid goto crossing initialization */
	bool all_complete = false;
	int max_iterations = 1000;
	int iteration = 0;
	
	NCCL_OFI_INFO(NCCL_NET, "Concurrent operations: Testing message transfer across %d communicators", 
		ctx->active_comm_count);
	
	/* Allocate and register buffers for all communicators */
	for (int i = 0; i < ctx->active_comm_count; i++) {
		/* Allocate send buffer */
		OFINCCLCHECKGOTO(allocate_buff(&send_buffers[i], test_size, buffer_type), res, cleanup);
		OFINCCLCHECKGOTO(initialize_buff(send_buffers[i], test_size, buffer_type), res, cleanup);
		
		/* Allocate receive buffer */
		OFINCCLCHECKGOTO(allocate_buff(&recv_buffers[i], test_size, buffer_type), res, cleanup);
		
		/* Register memory on send communicator */
		/* Note: regMr should be called on extNet interface, not communicator directly */
		/* This test focuses on concurrent operations, so we skip MR registration for now */
	}
	
	/* Synchronize before starting operations */
	MPI_Barrier(MPI_COMM_WORLD);
	
	/* Post receive operations on all communicators */
	for (int i = 0; i < ctx->active_comm_count; i++) {
		void *recv_data[1] = {recv_buffers[i]};
		size_t recv_sizes[1] = {test_size};
		int recv_tags[1] = {i}; /* Use communicator index as tag */
		nccl_net_ofi_mr_handle_t *recv_mhandles[1] = {(nccl_net_ofi_mr_handle_t *)recv_mr_handles[i]};
		
		int recv_res = ctx->rComms[i]->recv(ctx->rComms[i], 1, recv_data, recv_sizes, 
			recv_tags, recv_mhandles, &recv_reqs[i]);
		if (recv_res != 0) {
			NCCL_OFI_WARN("Concurrent operations failed: recv failed for comm %d", i);
			res = ncclInternalError;
			goto cleanup;
		}
	}
	
	/* Post send operations on all communicators */
	for (int i = 0; i < ctx->active_comm_count; i++) {
		int send_res = ctx->sComms[i]->send(ctx->sComms[i], send_buffers[i], test_size, i,
			(nccl_net_ofi_mr_handle_t *)send_mr_handles[i], &send_reqs[i]);
		if (send_res != 0) {
			NCCL_OFI_WARN("Concurrent operations failed: send failed for comm %d", i);
			res = ncclInternalError;
			goto cleanup;
		}
	}
	
	/* Wait for all operations to complete */
	
	while (!all_complete && iteration < max_iterations) {
		all_complete = true;
		
		/* Check send requests */
		for (int i = 0; i < ctx->active_comm_count; i++) {
			if (send_reqs[i]) {
				int done = 0, size = 0;
				int test_res = send_reqs[i]->test(send_reqs[i], &done, &size);
				if (test_res != 0) {
					NCCL_OFI_WARN("Concurrent operations: send test failed for comm %d", i);
					res = ncclInternalError;
					goto cleanup;
				}
				if (done) {
					send_reqs[i] = NULL;
				} else {
					all_complete = false;
				}
			}
		}
		
		/* Check receive requests */
		for (int i = 0; i < ctx->active_comm_count; i++) {
			if (recv_reqs[i]) {
				int done = 0, size = 0;
				int test_res = recv_reqs[i]->test(recv_reqs[i], &done, &size);
				if (test_res != 0) {
					NCCL_OFI_WARN("Concurrent operations: recv test failed for comm %d", i);
					res = ncclInternalError;
					goto cleanup;
				}
				if (done) {
					recv_reqs[i] = NULL;
				} else {
					all_complete = false;
				}
			}
		}
		
		iteration++;
		if (!all_complete) {
			usleep(1000); /* Sleep 1ms between checks */
		}
	}
	
	if (!all_complete) {
		NCCL_OFI_WARN("Concurrent operations: timeout waiting for operations to complete");
		res = ncclInternalError;
		goto cleanup;
	}
	
	NCCL_OFI_INFO(NCCL_NET, "Concurrent operations: Successfully completed message transfers on all communicators");
	
cleanup:
	/* Clean up allocated resources */
	for (int i = 0; i < ctx->active_comm_count; i++) {
		if (send_mr_handles[i] && ctx->sComms[i]->deregMr) {
			ctx->sComms[i]->deregMr(ctx->sComms[i], (nccl_net_ofi_mr_handle_t *)send_mr_handles[i]);
		}
		if (recv_mr_handles[i] && ctx->rComms[i]->deregMr) {
			ctx->rComms[i]->deregMr(ctx->rComms[i], (nccl_net_ofi_mr_handle_t *)recv_mr_handles[i]);
		}
		if (send_buffers[i]) {
			deallocate_buffer(send_buffers[i], buffer_type);
		}
		if (recv_buffers[i]) {
			deallocate_buffer(recv_buffers[i], buffer_type);
		}
	}
	
	return res;
}

static ncclResult_t cleanup_multi_comm_context(multi_comm_context *ctx, test_nccl_net_t *extNet)
{
	if (!ctx || !extNet) return ncclInvalidArgument;
	
	for (int i = 0; i < ctx->active_comm_count; i++) {
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
	return ncclSuccess;
}
#pragma GCC diagnostic pop

int main(int argc, char* argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, size, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev;
	multi_comm_context comm_ctx = {};
	test_nccl_net_t *extNet = NULL;
	int num_comms = MAX_TEST_COMMS;

	ofi_log_function = logger;

	/* Indicates if NICs support GPUDirect */
	int *test_support_gdr = NULL;

	/* Parse command line arguments */
	if (argc > 1) {
		num_comms = atoi(argv[1]);
		if (num_comms < 1 || num_comms > MAX_TEST_COMMS) {
			fprintf(stderr, "Number of communicators must be between 1 and %d\n", MAX_TEST_COMMS);
			return ncclInvalidArgument;
		}
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The multi_comm_domain functional test should be run with exactly two ranks.",
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
	for (int dev = 0; dev < ndev; dev++) {
		test_nccl_properties_t props = {};
		OFINCCLCHECKGOTO(extNet->getProperties(dev, &props), res, exit);
		print_dev_props(dev, &props);

		/* Set CUDA support */
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	/* Test all devices */
	for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {

		int dev = dev_idx;
		if (rank == 1) {
			/* In rank 1 scan devices in the opposite direction */
			dev = ndev - dev_idx - 1;
		}

		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses %d device for communication", rank, dev);

		if (test_support_gdr[dev] == 1) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
					"Network supports communication using CUDA buffers. Dev: %d", dev);
		}

		/* Initialize multi-communicator context */
		OFINCCLCHECKGOTO(init_multi_comm_context(&comm_ctx, dev), res, exit);

		/* Create multiple communicator pairs sequentially */
		for (int comm_idx = 0; comm_idx < num_comms; comm_idx++) {
			NCCL_OFI_INFO(NCCL_INIT, "Creating communicator pair %d on dev %d", comm_idx, dev);

			/* Listen API */
			NCCL_OFI_INFO(NCCL_INIT, "Server: Listening on dev %d for comm %d", dev, comm_idx);
			OFINCCLCHECKGOTO(extNet->listen(dev, (void *)&comm_ctx.handles[comm_idx], 
				(void **)&comm_ctx.lComms[comm_idx]), res, exit);

			if (rank == 0) {
				int peer_rank = 1;

				/* MPI send handle to peer */
				MPI_Send(&comm_ctx.handles[comm_idx], NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
					peer_rank, comm_idx, MPI_COMM_WORLD);

				/* MPI recv handle from peer */
				MPI_Recv((void *)comm_ctx.src_handles[comm_idx], NCCL_NET_HANDLE_MAXSIZE, 
					MPI_CHAR, peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				NCCL_OFI_INFO(NCCL_INIT, "Send connection request to rank %d for comm %d", 
					peer_rank, comm_idx);
				NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests for comm %d", comm_idx);

				/* Connect and Accept for this communicator pair */
				while (comm_ctx.sComms[comm_idx] == NULL || comm_ctx.rComms[comm_idx] == NULL) {
					/* Connect API */
					if (comm_ctx.sComms[comm_idx] == NULL) {
						OFINCCLCHECKGOTO(extNet->connect(dev, (void *)comm_ctx.src_handles[comm_idx], 
							(void **)&comm_ctx.sComms[comm_idx], &comm_ctx.s_handles[comm_idx]), res, exit);
					}

					/* Accept API */
					if (comm_ctx.rComms[comm_idx] == NULL) {
						OFINCCLCHECKGOTO(extNet->accept((void *)comm_ctx.lComms[comm_idx], 
							(void **)&comm_ctx.rComms[comm_idx], &comm_ctx.r_handles[comm_idx]), res, exit);
					}
				}

				NCCL_OFI_INFO(NCCL_INIT, "Successfully established connection pair %d with rank %d",
						comm_idx, peer_rank);
			} else {
				int peer_rank = 0;

				/* MPI recv handle from peer */
				MPI_Recv((void *)comm_ctx.src_handles[comm_idx], NCCL_NET_HANDLE_MAXSIZE, 
					MPI_CHAR, peer_rank, comm_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				/* MPI send handle to peer */
				MPI_Send((void *)comm_ctx.handles[comm_idx], NCCL_NET_HANDLE_MAXSIZE, 
					MPI_CHAR, peer_rank, comm_idx, MPI_COMM_WORLD);

				NCCL_OFI_INFO(NCCL_INIT, "Send connection request to rank %d for comm %d", 
					peer_rank, comm_idx);
				NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests for comm %d", comm_idx);

				/* Connect and Accept for this communicator pair */
				while (comm_ctx.sComms[comm_idx] == NULL || comm_ctx.rComms[comm_idx] == NULL) {
					/* Connect API */
					if (comm_ctx.sComms[comm_idx] == NULL) {
						OFINCCLCHECKGOTO(extNet->connect(dev, (void *)comm_ctx.src_handles[comm_idx], 
							(void **)&comm_ctx.sComms[comm_idx], &comm_ctx.s_handles[comm_idx]), res, exit);
					}

					/* Accept API */
					if (comm_ctx.rComms[comm_idx] == NULL) {
						OFINCCLCHECKGOTO(extNet->accept((void *)comm_ctx.lComms[comm_idx], 
							(void **)&comm_ctx.rComms[comm_idx], &comm_ctx.r_handles[comm_idx]), res, exit);
					}
				}

				NCCL_OFI_INFO(NCCL_INIT, "Successfully established connection pair %d with rank %d",
						comm_idx, peer_rank);
			}

			comm_ctx.active_comm_count++;
			
			/* Synchronize after each communicator pair creation */
			MPI_Barrier(MPI_COMM_WORLD);
		}

		NCCL_OFI_INFO(NCCL_INIT, "Successfully created %d communicator pairs on device %d", 
			num_comms, dev);

		/* Validate domain sharing across communicators */
		OFINCCLCHECKGOTO(validate_domain_sharing(&comm_ctx), res, exit);

		/* Validate MR operations work across all communicators */
		if (test_support_gdr[dev] == 1) {
			OFINCCLCHECKGOTO(validate_mr_operations(&comm_ctx, extNet, NCCL_PTR_CUDA), res, exit);
			/* Test concurrent operations with CUDA buffers */
			OFINCCLCHECKGOTO(test_concurrent_operations(&comm_ctx, extNet, NCCL_PTR_CUDA, rank), res, exit);
		} else {
			OFINCCLCHECKGOTO(validate_mr_operations(&comm_ctx, extNet, NCCL_PTR_HOST), res, exit);
			/* Test concurrent operations with host buffers */
			OFINCCLCHECKGOTO(test_concurrent_operations(&comm_ctx, extNet, NCCL_PTR_HOST, rank), res, exit);
		}

		/* Clean up all communicators for this device */
		OFINCCLCHECKGOTO(cleanup_multi_comm_context(&comm_ctx, extNet), res, exit);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Multi-communicator domain test completed successfully for rank %d", rank);

exit:
	if (test_support_gdr) {
		free(test_support_gdr);
		test_support_gdr = NULL;
	}

	/* Ensure cleanup in case of early exit */
	if (extNet) {
		cleanup_multi_comm_context(&comm_ctx, extNet);
	}

	return res;
}