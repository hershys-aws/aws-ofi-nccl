#include "nccl_ofi.h"
#include "nccl_ofi_rdma.h"
#include <stdlib.h>

/* Helper function to check if device is virtual */
static inline bool is_virtual_device(nccl_net_ofi_rdma_device_t *device) {
    return (device->physical_device_refs != NULL && device->num_physical_devs > 0);
}

/* Get physical device reference */
static nccl_net_ofi_rdma_device_t* get_physical_device(int dev_idx) {
    return (nccl_net_ofi_rdma_device_t*)plugin->base.p_devs[dev_idx];
}

/* Create virtual device from physical devices */
ncclResult_t nccl_net_ofi_rdma_make_virtual_dev(int* d, int* physical_devs, int num_physical_devs) {
    nccl_net_ofi_rdma_device_t *device;
    NCCLCHECK(ncclCalloc(&device, 1));

    /* Store physical device references */
    device->physical_device_refs = physical_devs;
    device->num_physical_devs = num_physical_devs;
    device->merge_level = PATH_PIX;  /* PCIe bridge level merging */

    /* Copy base properties from first physical device */
    nccl_net_ofi_rdma_device_t *first_dev = get_physical_device(physical_devs[0]);
    device->base = first_dev->base;

    /* Override virtual device functions */
    device->base.get_properties = nccl_net_ofi_rdma_get_virtual_properties;
    device->base.get_domain = nccl_net_ofi_rdma_get_virtual_domain;
    device->base.get_mr_key = nccl_net_ofi_rdma_get_virtual_mr_key;

    /* Register with plugin */
    *d = plugin->base.p_num_devs++;
    plugin->base.p_devs[*d] = &device->base;

    return ncclSuccess;
}

/* Aggregate properties from physical devices */
ncclResult_t nccl_net_ofi_rdma_get_virtual_properties(nccl_net_ofi_device_t *base_dev, 
    nccl_ofi_properties_t *props) {
    nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t*)base_dev;
    
    /* Get first physical device properties */
    nccl_net_ofi_rdma_device_t *first_dev = get_physical_device(device->physical_device_refs[0]);
    NCCLCHECK(first_dev->base.get_properties(&first_dev->base, props));
    
    /* Aggregate properties from other devices */
    for (int i = 1; i < device->num_physical_devs; i++) {
        nccl_ofi_properties_t phys_props;
        nccl_net_ofi_rdma_device_t *phys_dev = get_physical_device(device->physical_device_refs[i]);
        NCCLCHECK(phys_dev->base.get_properties(&phys_dev->base, &phys_props));
        
        /* Aggregate bandwidth */
        props->port_speed += phys_props.port_speed;
        
        /* Use minimum latency */
        props->latency = MIN(props->latency, phys_props.latency);
        
        /* Max transfer sizes */
        props->max_p2p_bytes = MAX(props->max_p2p_bytes, phys_props.max_p2p_bytes);
        props->max_coll_bytes = MAX(props->max_coll_bytes, phys_props.max_coll_bytes);
    }
    
    return ncclSuccess;
}

/* Get domain from first physical device */
nccl_net_ofi_domain_t* nccl_net_ofi_rdma_get_virtual_domain(nccl_net_ofi_device_t *dev) {
    nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t*)dev;
    nccl_net_ofi_rdma_device_t *first_dev = get_physical_device(device->physical_device_refs[0]);
    return first_dev->base.get_domain(&first_dev->base);
}

/* Get MR key from first physical device */
ncclResult_t nccl_net_ofi_rdma_get_virtual_mr_key(nccl_net_ofi_device_t *base_dev, 
    void* mhandle, uint64_t* mr_key) {
    nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t*)base_dev;
    nccl_net_ofi_rdma_device_t *first_dev = get_physical_device(device->physical_device_refs[0]);
    return first_dev->base.get_mr_key(&first_dev->base, mhandle, mr_key);
}

/* NCCL V9 API implementation */
ncclResult_t nccl_net_ofi_makeVDevice_v9(int* d, ncclNetVDeviceProps_t* props) {
    /* Validate input */
    if (props->ndevs > NCCL_NET_MAX_DEVS_PER_NIC_V9 || props->ndevs <= 0) {
        NCCL_OFI_WARN("Invalid number of devices: %d", props->ndevs);
        return ncclInvalidArgument;
    }
    
    /* Create physical device array */
    int *physical_devs = (int*)calloc(props->ndevs, sizeof(int));
    if (!physical_devs) return ncclSystemError;
    
    /* Copy device indices */
    memcpy(physical_devs, props->devs, props->ndevs * sizeof(int));
    
    /* Create virtual device */
    return nccl_net_ofi_rdma_make_virtual_dev(d, physical_devs, props->ndevs);
}
