#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += $(MPI_CPPFLAGS) $(CUDA_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
LDADD = $(top_builddir)/src/libinternal_plugin.la $(MPI_LIBS) $(CUDA_LIBS)

# this is a little jenky, but we've always assumed we had wrapper compilers
# available for MPI.  We don't want to just override CXX to get mpicxx used,
# because we lose the C++ standard setting that happens in AX_CXX_COMPILE_STDCXX
# and Amazon Linux 2 in particular supports C++17 but defaults to C++11.
# Instead of overriding CXX, override CXXCOMPILE and CXXLINK to be the
# underlying rule, but with $(MPICXX) as the compiler and set the environment
# variable CXX so that mpicxx will pick up the -std= value.
CXXCOMPILE = OMPI_CXX="$(CXX)" MPICH_CXX="$(CXX)" \
	$(MPICXX) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)

CXXLINK = OMPI_CXX="$(CXX)" MPICH_CXX="$(CXX)" \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=link $(MPICXX) $(AM_CXXFLAGS) \
	$(CXXFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@


if ENABLE_FUNC_TESTS
noinst_HEADERS = test-common.h

bin_PROGRAMS = nccl_connection nccl_message_transfer ring inflight_close reuse_listen_comm \
	multi_comm_domain comm_abort_scenarios resource_cleanup stress_scale

nccl_connection_SOURCES = nccl_connection.cpp
nccl_message_transfer_SOURCES = nccl_message_transfer.cpp
ring_SOURCES = ring.cpp
inflight_close_SOURCES = inflight_close.cpp
reuse_listen_comm_SOURCES = reuse_listen_comm.cpp
multi_comm_domain_SOURCES = multi_comm_domain.cpp
comm_abort_scenarios_SOURCES = comm_abort_scenarios.cpp
resource_cleanup_SOURCES = resource_cleanup.cpp
stress_scale_SOURCES = stress_scale.cpp

# SLURM functional test targets
check-functional: $(bin_PROGRAMS)
	@echo "Running functional tests on SLURM..."
	@if [ -n "$(TEST)" ]; then \
		echo "Running specific test: $(TEST)"; \
		$(srcdir)/run-slurm-tests.sh "$(TEST)"; \
	else \
		echo "Running all functional tests"; \
		$(srcdir)/run-slurm-tests.sh; \
	fi

# Alternative target name for consistency
check-functional-tests: check-functional

# Force interactive mode
check-functional-interactive: $(bin_PROGRAMS)
	@echo "Running functional tests on SLURM (interactive mode)..."
	@if [ -n "$(TEST)" ]; then \
		echo "Running specific test: $(TEST)"; \
		$(srcdir)/run-slurm-tests.sh --interactive "$(TEST)"; \
	else \
		echo "Running all functional tests"; \
		$(srcdir)/run-slurm-tests.sh --interactive; \
	fi

# Force batch mode
check-functional-batch: $(bin_PROGRAMS)
	@echo "Running functional tests on SLURM (batch mode)..."
	@if [ -n "$(TEST)" ]; then \
		echo "Running specific test: $(TEST)"; \
		$(srcdir)/run-slurm-tests.sh --batch "$(TEST)"; \
	else \
		echo "Running all functional tests"; \
		$(srcdir)/run-slurm-tests.sh --batch; \
	fi

# Clean targets for SLURM-generated files
clean-functional:
	@echo "Cleaning SLURM functional test files..."
	@rm -f slurm-*.out slurm-*.err slurm_job_*.sh
	@echo "SLURM functional test files cleaned."

# Clean all generated files (including SLURM files)
clean-local: clean-functional

# Alternative clean target name
clean-slurm: clean-functional

.PHONY: check-functional check-functional-tests check-functional-interactive check-functional-batch clean-functional clean-slurm

endif
