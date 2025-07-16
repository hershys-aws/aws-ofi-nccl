# AWS OFI NCCL vNIC Implementation

This document describes the virtual Network Interface Card (vNIC) implementation in the AWS OFI NCCL plugin.

## Overview

The vNIC implementation enables the aggregation of multiple physical NICs into a single virtual device, providing improved bandwidth and resource utilization. This implementation focuses on RDMA-only support with PCIe bridge level merging.

## Core Implementation

### Key Files

- `/src/nccl_ofi_vnic.cpp`: Core vNIC implementation
- `/include/nccl_ofi_rdma.h`: vNIC support structures and declarations
- `/tests/unit/vnic.cpp`: Unit tests

### Data Structures

```c
struct nccl_net_ofi_rdma_device {
    /* Existing fields */
    int *physical_device_refs;  /* Physical device array */
    int num_physical_devs;      /* Number of physical devices */
    int merge_level;           /* PATH_PIX for PCIe bridge */
}
```

### Primary Functions

- `nccl_net_ofi_makeVDevice_v9`: NCCL V9 API implementation for virtual device creation
- `nccl_net_ofi_rdma_get_virtual_properties`: Property aggregation for virtual devices
- `nccl_net_ofi_rdma_get_virtual_domain`: Domain management for virtual devices

## Technical Requirements

- RDMA protocol only implementation
- C-style arrays (no C++ vectors)
- PATH_PIX merge level for PCIe bridge merging
- Thread safety requirements
- Proper resource cleanup handling

## Testing

### Unit Tests Location
Tests are integrated into the existing unit test framework in `/tests/unit/vnic.cpp`

### Test Coverage

#### Core Functionality Tests
- Device creation validation
- Property aggregation verification
- Memory registration testing

#### Error Handling Tests
- Invalid parameter validation
- Resource cleanup verification
- Error condition handling

### Test Implementation Notes
- Uses existing test infrastructure
- No additional test dependencies
- Assert-based validation
- Integrated with project's build system via Makefile.am

## Building and Testing

```bash
# Build the plugin with vNIC support
./autogen.sh
./configure
make

# Run tests
make check
```

## Implementation Notes

- Uses C-style arrays for compatibility
- Focuses on RDMA-only implementation
- Implements proper cleanup and resource management
- Maintains thread safety
- Follows existing AWS OFI NCCL plugin architecture

## Validation

The implementation is validated through:
- Unit tests in existing test framework
- Resource leak verification
- Error handling validation

## Future Work

- Performance optimization
- Additional merge level support
- Enhanced fault tolerance
- Extended monitoring capabilities
