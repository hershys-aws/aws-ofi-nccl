/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef RESOURCE_TRACKER_H_
#define RESOURCE_TRACKER_H_

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <set>
#include <vector>

#include "nccl_ofi_log.h"

/**
 * resource_snapshot - Captures the state of allocated resources at a point in time
 * 
 * This structure is used for resource leak detection by taking snapshots before
 * and after test operations and comparing them to identify leaked resources.
 */
struct resource_snapshot {
	std::vector<void*> communicators;    // Active communicators
	std::vector<void*> memory_handles;   // Registered memory handles
	std::vector<void*> buffers;          // Allocated buffers
	size_t timestamp;                    // Snapshot timestamp
};

/**
 * ResourceTracker - Tracks allocated resources for leak detection
 * 
 * This class provides snapshot-based resource tracking to detect leaks.
 * Resources are tracked in thread-safe sets, and snapshots can be taken
 * before and after operations to identify leaked resources.
 * 
 * Example usage:
 * 
 *   resource_tracker tracker;
 *   
 *   // Track resources as they are allocated
 *   tracker.track_communicator(comm);
 *   tracker.track_buffer(buffer);
 *   
 *   // Take snapshot before operation
 *   resource_snapshot before;
 *   tracker.take_snapshot(&before);
 *   
 *   // Perform operation that might leak resources
 *   // ...
 *   
 *   // Take snapshot after operation
 *   resource_snapshot after;
 *   tracker.take_snapshot(&after);
 *   
 *   // Compare snapshots to detect leaks
 *   resource_snapshot leaked;
 *   if (tracker.compare_snapshots(before, after, &leaked)) {
 *       tracker.print_leaks(leaked);
 *   }
 */
class resource_tracker {
public:
	/**
	 * Constructor - Initializes empty resource sets and mutex
	 */
	resource_tracker()
		: communicators_()
		, memory_handles_()
		, buffers_()
		, mutex_()
	{
		NCCL_OFI_TRACE(NCCL_NET, "ResourceTracker initialized");
	}
	
	/**
	 * Destructor
	 */
	~resource_tracker()
	{
		NCCL_OFI_TRACE(NCCL_NET, "ResourceTracker destroyed");
	}
	
	/**
	 * Take snapshot of current resource state
	 * 
	 * Locks the mutex, copies current resource sets to the snapshot,
	 * sets the timestamp, and unlocks the mutex.
	 * 
	 * @param snapshot Output pointer to resource_snapshot to populate
	 */
	void take_snapshot(resource_snapshot* snapshot)
	{
		if (snapshot == nullptr) {
			NCCL_OFI_WARN("Invalid NULL snapshot pointer");
			return;
		}
		
		// Lock mutex for thread-safe access
		std::lock_guard<std::mutex> lock(mutex_);
		
		NCCL_OFI_TRACE(NCCL_NET, "Taking resource snapshot: %zu communicators, "
		               "%zu memory handles, %zu buffers",
		               communicators_.size(), memory_handles_.size(), buffers_.size());
		
		// Clear snapshot vectors
		snapshot->communicators.clear();
		snapshot->memory_handles.clear();
		snapshot->buffers.clear();
		
		// Copy communicators
		snapshot->communicators.reserve(communicators_.size());
		for (void* comm : communicators_) {
			snapshot->communicators.push_back(comm);
		}
		
		// Copy memory handles
		snapshot->memory_handles.reserve(memory_handles_.size());
		for (void* mhandle : memory_handles_) {
			snapshot->memory_handles.push_back(mhandle);
		}
		
		// Copy buffers
		snapshot->buffers.reserve(buffers_.size());
		for (void* buffer : buffers_) {
			snapshot->buffers.push_back(buffer);
		}
		
		// Set timestamp using C++ chrono
		auto now = std::chrono::system_clock::now();
		auto duration = now.time_since_epoch();
		snapshot->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
		
		NCCL_OFI_TRACE(NCCL_NET, "Snapshot taken at timestamp %zu", snapshot->timestamp);
	}
	
	/**
	 * Track a communicator
	 * 
	 * Adds a communicator to the tracked set in a thread-safe manner.
	 * 
	 * @param comm Communicator pointer to track
	 */
	void track_communicator(void* comm)
	{
		if (comm == nullptr) return;
		std::lock_guard<std::mutex> lock(mutex_);
		communicators_.insert(comm);
		NCCL_OFI_TRACE(NCCL_NET, "Tracking communicator: %p (total: %zu)",
		               comm, communicators_.size());
	}
	
	/**
	 * Untrack a communicator
	 * 
	 * Removes a communicator from the tracked set in a thread-safe manner.
	 * 
	 * @param comm Communicator pointer to untrack
	 */
	void untrack_communicator(void* comm)
	{
		if (comm == nullptr) return;
		std::lock_guard<std::mutex> lock(mutex_);
		communicators_.erase(comm);
		NCCL_OFI_TRACE(NCCL_NET, "Untracking communicator: %p (total: %zu)",
		               comm, communicators_.size());
	}
	
	/**
	 * Track a memory handle
	 * 
	 * Adds a memory handle to the tracked set in a thread-safe manner.
	 * 
	 * @param mhandle Memory handle pointer to track
	 */
	void track_memory_handle(void* mhandle)
	{
		if (mhandle == nullptr) return;
		std::lock_guard<std::mutex> lock(mutex_);
		memory_handles_.insert(mhandle);
		NCCL_OFI_TRACE(NCCL_NET, "Tracking memory handle: %p (total: %zu)",
		               mhandle, memory_handles_.size());
	}
	
	/**
	 * Untrack a memory handle
	 * 
	 * Removes a memory handle from the tracked set in a thread-safe manner.
	 * 
	 * @param mhandle Memory handle pointer to untrack
	 */
	void untrack_memory_handle(void* mhandle)
	{
		if (mhandle == nullptr) return;
		std::lock_guard<std::mutex> lock(mutex_);
		memory_handles_.erase(mhandle);
		NCCL_OFI_TRACE(NCCL_NET, "Untracking memory handle: %p (total: %zu)",
		               mhandle, memory_handles_.size());
	}
	
	/**
	 * Track a buffer
	 * 
	 * Adds a buffer to the tracked set in a thread-safe manner.
	 * 
	 * @param buffer Buffer pointer to track
	 */
	void track_buffer(void* buffer)
	{
		if (buffer == nullptr) return;
		std::lock_guard<std::mutex> lock(mutex_);
		buffers_.insert(buffer);
		NCCL_OFI_TRACE(NCCL_NET, "Tracking buffer: %p (total: %zu)",
		               buffer, buffers_.size());
	}
	
	/**
	 * Untrack a buffer
	 * 
	 * Removes a buffer from the tracked set in a thread-safe manner.
	 * 
	 * @param buffer Buffer pointer to untrack
	 */
	void untrack_buffer(void* buffer)
	{
		if (buffer == nullptr) return;
		std::lock_guard<std::mutex> lock(mutex_);
		buffers_.erase(buffer);
		NCCL_OFI_TRACE(NCCL_NET, "Untracking buffer: %p (total: %zu)",
		               buffer, buffers_.size());
	}
	
	/**
	 * Compare two snapshots to detect leaks
	 * 
	 * Compares before and after snapshots to identify resources that exist
	 * in the after snapshot but not in the before snapshot. These are
	 * considered leaked resources.
	 * 
	 * @param before Snapshot taken before the operation
	 * @param after Snapshot taken after the operation
	 * @param leaked Output snapshot containing leaked resources
	 * @return true if leaks were detected, false otherwise
	 */
	bool compare_snapshots(const resource_snapshot& before,
	                       const resource_snapshot& after,
	                       resource_snapshot* leaked)
	{
		if (leaked == nullptr) {
			NCCL_OFI_WARN("Invalid NULL leaked pointer");
			return false;
		}
		
		NCCL_OFI_TRACE(NCCL_NET, "Comparing snapshots: before (comm=%zu, mh=%zu, buf=%zu) "
		               "vs after (comm=%zu, mh=%zu, buf=%zu)",
		               before.communicators.size(), before.memory_handles.size(), 
		               before.buffers.size(),
		               after.communicators.size(), after.memory_handles.size(), 
		               after.buffers.size());
		
		// Clear leaked snapshot
		leaked->communicators.clear();
		leaked->memory_handles.clear();
		leaked->buffers.clear();
		leaked->timestamp = after.timestamp;
		
		// Convert before vectors to sets for efficient lookup
		std::set<void*> before_comms(before.communicators.begin(), 
		                              before.communicators.end());
		std::set<void*> before_mhandles(before.memory_handles.begin(), 
		                                 before.memory_handles.end());
		std::set<void*> before_buffers(before.buffers.begin(), 
		                                before.buffers.end());
		
		// Find leaked communicators (in after but not in before)
		for (void* comm : after.communicators) {
			if (before_comms.find(comm) == before_comms.end()) {
				leaked->communicators.push_back(comm);
			}
		}
		
		// Find leaked memory handles (in after but not in before)
		for (void* mhandle : after.memory_handles) {
			if (before_mhandles.find(mhandle) == before_mhandles.end()) {
				leaked->memory_handles.push_back(mhandle);
			}
		}
		
		// Find leaked buffers (in after but not in before)
		for (void* buffer : after.buffers) {
			if (before_buffers.find(buffer) == before_buffers.end()) {
				leaked->buffers.push_back(buffer);
			}
		}
		
		// Check if any leaks were detected
		bool has_leaks = !leaked->communicators.empty() || 
		                 !leaked->memory_handles.empty() || 
		                 !leaked->buffers.empty();
		
		if (has_leaks) {
			NCCL_OFI_WARN("Resource leaks detected: %zu communicators, "
			              "%zu memory handles, %zu buffers",
			              leaked->communicators.size(), 
			              leaked->memory_handles.size(), 
			              leaked->buffers.size());
		} else {
			NCCL_OFI_TRACE(NCCL_NET, "No resource leaks detected");
		}
		
		return has_leaks;
	}
	
	/**
	 * Print leaked resources
	 * 
	 * Prints detailed information about leaked resources in a readable format.
	 * 
	 * @param leaked Snapshot containing leaked resources
	 */
	void print_leaks(const resource_snapshot& leaked)
	{
		printf("\n");
		printf("=== Resource Leak Report ===\n");
		printf("Timestamp: %zu\n", leaked.timestamp);
		printf("\n");
		
		// Print leaked communicators
		if (!leaked.communicators.empty()) {
			printf("Leaked Communicators (%zu):\n", leaked.communicators.size());
			for (size_t i = 0; i < leaked.communicators.size(); i++) {
				printf("  [%zu] %p\n", i, leaked.communicators[i]);
			}
			printf("\n");
		}
		
		// Print leaked memory handles
		if (!leaked.memory_handles.empty()) {
			printf("Leaked Memory Handles (%zu):\n", leaked.memory_handles.size());
			for (size_t i = 0; i < leaked.memory_handles.size(); i++) {
				printf("  [%zu] %p\n", i, leaked.memory_handles[i]);
			}
			printf("\n");
		}
		
		// Print leaked buffers
		if (!leaked.buffers.empty()) {
			printf("Leaked Buffers (%zu):\n", leaked.buffers.size());
			for (size_t i = 0; i < leaked.buffers.size(); i++) {
				printf("  [%zu] %p\n", i, leaked.buffers[i]);
			}
			printf("\n");
		}
		
		// Print summary
		size_t total_leaks = leaked.communicators.size() + 
		                     leaked.memory_handles.size() + 
		                     leaked.buffers.size();
		
		if (total_leaks == 0) {
			printf("No resource leaks detected.\n");
		} else {
			printf("Total leaked resources: %zu\n", total_leaks);
		}
		
		printf("============================\n");
		printf("\n");
	}
	
private:
	std::set<void*> communicators_;
	std::set<void*> memory_handles_;
	std::set<void*> buffers_;
	std::mutex mutex_;
};

#endif // RESOURCE_TRACKER_H_
