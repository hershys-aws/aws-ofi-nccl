/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "nccl_ofi.h"
#include "nccl_ofi_rdma.h"
#include <assert.h>

static void test_vnic_create()
{
    nccl_net_ofi_plugin_t* plugin = NULL;
    bool found_multi_rail = false;
    assert(nccl_net_ofi_rdma_init(NULL, &plugin, &found_multi_rail) == ncclSuccess);
    assert(found_multi_rail);

    // Test device creation
    ncclNetVDeviceProps_t props;
    int devIndex;
    props.ndevs = 2;
    props.devs[0] = 0;
    props.devs[1] = 1;
    assert(nccl_net_ofi_makeVDevice_v9(&devIndex, &props) == ncclSuccess);

    // Verify device properties
    nccl_ofi_properties_t vProps;
    assert(plugin->get_properties(devIndex, &vProps) == ncclSuccess);

    plugin->finalize();
}

static void test_vnic_invalid_params()
{
    nccl_net_ofi_plugin_t* plugin = NULL;
    bool found_multi_rail = false;
    assert(nccl_net_ofi_rdma_init(NULL, &plugin, &found_multi_rail) == ncclSuccess);

    // Test zero devices
    ncclNetVDeviceProps_t props;
    int devIndex;
    props.ndevs = 0;
    assert(nccl_net_ofi_makeVDevice_v9(&devIndex, &props) == ncclInvalidArgument);

    // Test too many devices
    props.ndevs = NCCL_NET_MAX_DEVS_PER_NIC_V9 + 1;
    assert(nccl_net_ofi_makeVDevice_v9(&devIndex, &props) == ncclInvalidArgument);

    plugin->finalize();
}

static void test_vnic_property_aggregation()
{
    nccl_net_ofi_plugin_t* plugin = NULL;
    bool found_multi_rail = false;
    assert(nccl_net_ofi_rdma_init(NULL, &plugin, &found_multi_rail) == ncclSuccess);

    // Get physical device properties
    nccl_ofi_properties_t phys_props[2];
    assert(plugin->get_properties(0, &phys_props[0]) == ncclSuccess);
    assert(plugin->get_properties(1, &phys_props[1]) == ncclSuccess);

    // Create virtual device
    ncclNetVDeviceProps_t props;
    int devIndex;
    props.ndevs = 2;
    props.devs[0] = 0;
    props.devs[1] = 1;
    assert(nccl_net_ofi_makeVDevice_v9(&devIndex, &props) == ncclSuccess);

    // Verify property aggregation
    nccl_ofi_properties_t vProps;
    assert(plugin->get_properties(devIndex, &vProps) == ncclSuccess);

    // Verify bandwidth aggregation
    assert(vProps.port_speed == phys_props[0].port_speed + phys_props[1].port_speed);

    // Verify latency is minimum
    float expected_latency = phys_props[0].latency < phys_props[1].latency ? 
                           phys_props[0].latency : phys_props[1].latency;
    assert(vProps.latency == expected_latency);

    plugin->finalize();
}

int main(int argc, char* argv[])
{
    test_vnic_create();
    test_vnic_invalid_params();
    test_vnic_property_aggregation();
    return 0;
}
