# AWS OFI NCCL Release notes

# Supported Distributions

- Amazon Linux 2 and Amazon Linux 2023
- Ubuntu 22.04 LTS and 24.04 LTS

# v0.0.2 (2025-06)

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn3](https://github.com/aws/libfabric/commits/2.1.0amzn3.0).

### Bug Fixes and Improvements:

- On AWS platforms the following environment variables `NCCL_BUFFSIZE`, `NCCL_P2P_NET_CHUNKSIZE`, `NCCL_NVLSTREE_MAX_CHUNKSIZE`, `NCCL_NVLS_CHUNKSIZE`, `NCCL_NET_FORCE_FLUSH` may be set by the plugin
- Added `libnccl-tuner-ofi.so` symlink for easier configuration with `NCCL_TUNER_PLUGIN=ofi`


# v0.0.1 (2025-06)

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn3](https://github.com/aws/libfabric/commits/2.1.0amzn3.0).

### Bug Fixes and Improvements:

- On AWS platforms the following environment variables `NCCL_BUFFSIZE`, `NCCL_P2P_NET_CHUNKSIZE`, `NCCL_NVLSTREE_MAX_CHUNKSIZE`, `NCCL_NVLS_CHUNKSIZE`, `NCCL_NET_FORCE_FLUSH` may be set by the plugin
- Added `libnccl-tuner-ofi.so` symlink for easier configuration with `NCCL_TUNER_PLUGIN=ofi`

# v0.0.0 (2025-06)

The 1.16.0 release series supports [NCCL v2.27.5-1](https://github.com/NVIDIA/nccl/releases/tag/v2.27.5-1) while maintaining backward compatibility with older NCCL versions (([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn3](https://github.com/aws/libfabric/commits/2.1.0amzn3.0).

### Bug Fixes and Improvements:

- On AWS platforms the following environment variables `NCCL_BUFFSIZE`, `NCCL_P2P_NET_CHUNKSIZE`, `NCCL_NVLSTREE_MAX_CHUNKSIZE`, `NCCL_NVLS_CHUNKSIZE`, `NCCL_NET_FORCE_FLUSH` may be set by the plugin
- Fix bug that prevented communicators from aborting gracefully, as part of supporting NCCL fault tolerance features
- On AWS platforms, enable collective algorithm tuner by default
- Improve P6-B200 tuner configuration to improve performance for 4 -- 32 MiB messages across node counts and large message AllReduce on 8 nodes
- Added `libnccl-tuner-ofi.so` symlink for easier configuration with `NCCL_TUNER_PLUGIN=ofi`
