#if !defined(_CUPTI_ACTIVITY_H_)
#define _CUPTI_ACTIVITY_H_


#include "cupti_result.h"


#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__LP64__)
#define CUPTILP64 1
#elif defined(_WIN64)
#define CUPTILP64 1
#else
#undef CUPTILP64
#endif

#define ACTIVITY_RECORD_ALIGNMENT 8
#if defined(_WIN32) // Windows 32- and 64-bit
#define START_PACKED_ALIGNMENT __pragma(pack(push,1)) // exact fit - no padding
#define PACKED_ALIGNMENT __declspec(align(ACTIVITY_RECORD_ALIGNMENT))
#define END_PACKED_ALIGNMENT __pragma(pack(pop))
#elif defined(__GNUC__) // GCC
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT __attribute__ ((__packed__)) __attribute__ ((aligned (ACTIVITY_RECORD_ALIGNMENT)))
#define END_PACKED_ALIGNMENT
#else // all other compilers
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT
#define END_PACKED_ALIGNMENT
#endif

#define CUPTI_UNIFIED_MEMORY_CPU_DEVICE_ID ((uint32_t) 0xFFFFFFFFU)
#define CUPTI_INVALID_CONTEXT_ID ((uint32_t) 0xFFFFFFFFU)
#define CUPTI_INVALID_STREAM_ID ((uint32_t) 0xFFFFFFFFU)
#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif



typedef enum {
  /**
   * The activity record is invalid.
   */
  CUPTI_ACTIVITY_KIND_INVALID  = 0,
  /**
   * A host<->host, host<->device, or device<->device memory copy. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityMemcpy4.
   */
  CUPTI_ACTIVITY_KIND_MEMCPY   = 1,
  /**
   * A memory set executing on the GPU. The corresponding activity
   * record structure is \ref CUpti_ActivityMemset3.
   */
  CUPTI_ACTIVITY_KIND_MEMSET   = 2,
  /**
   * A kernel executing on the GPU. This activity kind may significantly change
   * the overall performance characteristics of the application because all
   * kernel executions are serialized on the GPU. Other activity kind for kernel
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL doesn't break kernel concurrency.
   * The corresponding activity record structure is \ref CUpti_ActivityKernel6.
   */
  CUPTI_ACTIVITY_KIND_KERNEL   = 3,
  /**
   * A CUDA driver API function execution. The corresponding activity
   * record structure is \ref CUpti_ActivityAPI.
   */
  CUPTI_ACTIVITY_KIND_DRIVER   = 4,
  /**
   * A CUDA runtime API function execution. The corresponding activity
   * record structure is \ref CUpti_ActivityAPI.
   */
  CUPTI_ACTIVITY_KIND_RUNTIME  = 5,
  /**
   * An event value. The corresponding activity record structure is
   * \ref CUpti_ActivityEvent.
   */
  CUPTI_ACTIVITY_KIND_EVENT    = 6,
  /**
   * A metric value. The corresponding activity record structure is
   * \ref CUpti_ActivityMetric.
   */
  CUPTI_ACTIVITY_KIND_METRIC   = 7,
  /**
   * Information about a device. The corresponding activity record
   * structure is \ref CUpti_ActivityDevice3.
   */
  CUPTI_ACTIVITY_KIND_DEVICE   = 8,
  /**
   * Information about a context. The corresponding activity record
   * structure is \ref CUpti_ActivityContext.
   */
  CUPTI_ACTIVITY_KIND_CONTEXT  = 9,
  /**
   * A kernel executing on the GPU. This activity kind doesn't break
   * kernel concurrency. The corresponding activity record structure
   * is \ref CUpti_ActivityKernel6.
   */
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,
  /**
   * Thread, device, context, etc. name. The corresponding activity
   * record structure is \ref CUpti_ActivityName.
   */
  CUPTI_ACTIVITY_KIND_NAME     = 11,
  /**
   * Instantaneous, start, or end marker. The corresponding activity
   * record structure is \ref CUpti_ActivityMarker2.
   */
  CUPTI_ACTIVITY_KIND_MARKER = 12,
  /**
   * Extended, optional, data about a marker. The corresponding
   * activity record structure is \ref CUpti_ActivityMarkerData.
   */
  CUPTI_ACTIVITY_KIND_MARKER_DATA = 13,
  /**
   * Source information about source level result. The corresponding
   * activity record structure is \ref CUpti_ActivitySourceLocator.
   */
  CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR = 14,
  /**
   * Results for source-level global acccess. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityGlobalAccess3.
   */
  CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS = 15,
  /**
   * Results for source-level branch. The corresponding
   * activity record structure is \ref CUpti_ActivityBranch2.
   */
  CUPTI_ACTIVITY_KIND_BRANCH = 16,
  /**
   * Overhead activity records. The
   * corresponding activity record structure is
   * \ref CUpti_ActivityOverhead.
   */
  CUPTI_ACTIVITY_KIND_OVERHEAD = 17,
  /**
   * A CDP (CUDA Dynamic Parallel) kernel executing on the GPU. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityCdpKernel.  This activity can not be directly
   * enabled or disabled. It is enabled and disabled through
   * concurrent kernel activity i.e. _CONCURRENT_KERNEL.
   */
  CUPTI_ACTIVITY_KIND_CDP_KERNEL = 18,
  /**
   * Preemption activity record indicating a preemption of a CDP (CUDA
   * Dynamic Parallel) kernel executing on the GPU. The corresponding
   * activity record structure is \ref CUpti_ActivityPreemption.
   */
  CUPTI_ACTIVITY_KIND_PREEMPTION = 19,
  /**
   * Environment activity records indicating power, clock, thermal,
   * etc. levels of the GPU. The corresponding activity record
   * structure is \ref CUpti_ActivityEnvironment.
   */
  CUPTI_ACTIVITY_KIND_ENVIRONMENT = 20,
  /**
   * An event value associated with a specific event domain
   * instance. The corresponding activity record structure is \ref
   * CUpti_ActivityEventInstance.
   */
  CUPTI_ACTIVITY_KIND_EVENT_INSTANCE = 21,
  /**
   * A peer to peer memory copy. The corresponding activity record
   * structure is \ref CUpti_ActivityMemcpyPtoP3.
   */
  CUPTI_ACTIVITY_KIND_MEMCPY2 = 22,
  /**
   * A metric value associated with a specific metric domain
   * instance. The corresponding activity record structure is \ref
   * CUpti_ActivityMetricInstance.
   */
  CUPTI_ACTIVITY_KIND_METRIC_INSTANCE = 23,
  /**
   * Results for source-level instruction execution.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstructionExecution.
   */
  CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION = 24,
  /**
   * Unified Memory counter record. The corresponding activity
   * record structure is \ref CUpti_ActivityUnifiedMemoryCounter2.
   */
  CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER = 25,
  /**
   * Device global/function record. The corresponding activity
   * record structure is \ref CUpti_ActivityFunction.
   */
  CUPTI_ACTIVITY_KIND_FUNCTION = 26,
  /**
   * CUDA Module record. The corresponding activity
   * record structure is \ref CUpti_ActivityModule.
   */
  CUPTI_ACTIVITY_KIND_MODULE = 27,
  /**
   * A device attribute value. The corresponding activity record
   * structure is \ref CUpti_ActivityDeviceAttribute.
   */
  CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE   = 28,
  /**
   * Results for source-level shared acccess. The
   * corresponding activity record structure is \ref
   * CUpti_ActivitySharedAccess.
   */
  CUPTI_ACTIVITY_KIND_SHARED_ACCESS = 29,
  /**
   * Enable PC sampling for kernels. This will serialize
   * kernels. The corresponding activity record structure
   * is \ref CUpti_ActivityPCSampling3.
   */
  CUPTI_ACTIVITY_KIND_PC_SAMPLING = 30,
  /**
   * Summary information about PC sampling records. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityPCSamplingRecordInfo.
   */
  CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO = 31,
  /**
   * SASS/Source line-by-line correlation record.
   * This will generate sass/source correlation for functions that have source
   * level analysis or pc sampling results. The records will be generated only
   * when either of source level analysis or pc sampling activity is enabled.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstructionCorrelation.
   */
  CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION = 32,
  /**
   * OpenACC data events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenAccData.
   */
  CUPTI_ACTIVITY_KIND_OPENACC_DATA = 33,
  /**
   * OpenACC launch events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenAccLaunch.
   */
  CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH = 34,
  /**
   * OpenACC other events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenAccOther.
   */
  CUPTI_ACTIVITY_KIND_OPENACC_OTHER = 35,
  /**
   * Information about a CUDA event. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityCudaEvent.
   */
  CUPTI_ACTIVITY_KIND_CUDA_EVENT = 36,
  /**
   * Information about a CUDA stream. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityStream.
   */
  CUPTI_ACTIVITY_KIND_STREAM = 37,
  /**
   * Records for synchronization management. The
   * corresponding activity record structure is \ref
   * CUpti_ActivitySynchronization.
   */
  CUPTI_ACTIVITY_KIND_SYNCHRONIZATION = 38,
  /**
   * Records for correlation of different programming APIs. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityExternalCorrelation.
   */
  CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39,
  /**
   * NVLink information.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityNvLink4.
   */
  CUPTI_ACTIVITY_KIND_NVLINK = 40,
  /**
   * Instantaneous Event information.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousEvent.
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT = 41,
  /**
   * Instantaneous Event information for a specific event
   * domain instance.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousEventInstance
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE = 42,
  /**
   * Instantaneous Metric information
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousMetric.
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC = 43,
  /**
   * Instantaneous Metric information for a specific metric
   * domain instance.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousMetricInstance.
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE = 44,
  /**
   * Memory activity tracking allocation and freeing of the memory
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemory.
   */
  CUPTI_ACTIVITY_KIND_MEMORY = 45,
  /**
   * PCI devices information used for PCI topology.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityPcie.
   */
  CUPTI_ACTIVITY_KIND_PCIE = 46,
  /**
   * OpenMP parallel events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenMp.
   */
  CUPTI_ACTIVITY_KIND_OPENMP = 47,
  /**
   * A CUDA driver kernel launch occurring outside of any
   * public API function execution.  Tools can handle these
   * like records for driver API launch functions, although
   * the cbid field is not used here.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityAPI.
   */
  CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API = 48,
  /**
   * Memory activity tracking allocation and freeing of the memory
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemory2.
   */
  CUPTI_ACTIVITY_KIND_MEMORY2 = 49,

  /**
   * Memory pool activity tracking creation, destruction and
   * triming of the memory pool.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemoryPool.
   */
  CUPTI_ACTIVITY_KIND_MEMORY_POOL = 50,

  CUPTI_ACTIVITY_KIND_COUNT = 51,

  CUPTI_ACTIVITY_KIND_FORCE_INT     = 0x7fffffff
} CUpti_ActivityKind;

/**
 * \brief Partitioned global caching option
 */
typedef enum {
  /**
   * Partitioned global cache config unknown.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN = 0,
  /**
   * Partitioned global cache not supported.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED = 1,
  /**
   * Partitioned global cache config off.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF = 2,
  /**
   * Partitioned global cache config on.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON = 3,
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT  = 0x7fffffff
} CUpti_ActivityPartitionedGlobalCacheConfig;

/**
 * \brief The shared memory limit per block config for a kernel
 * This should be used to set 'cudaOccFuncShmemConfig' field in occupancy calculator API
 */
typedef enum  {
    /* The shared memory limit config is default */
    CUPTI_FUNC_SHMEM_LIMIT_DEFAULT              = 0x00,
    /* User has opted for a higher dynamic shared memory limit using function attribute
       'cudaFuncAttributeMaxDynamicSharedMemorySize' for runtime API or
       CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES for driver API */
    CUPTI_FUNC_SHMEM_LIMIT_OPTIN                = 0x01,
    CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT            = 0x7fffffff
} CUpti_FuncShmemLimitConfig;

typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;
      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchrnous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;
} CUpti_ActivityKernel6;


/**
 * \brief Activity attributes.
 *
 * These attributes are used to control the behavior of the activity
 * API.
 */
typedef enum {
    /**
     * The device memory size (in bytes) reserved for storing profiling data for concurrent
     * kernels (activity kind \ref CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), memcopies and memsets
     * for each buffer on a context. The value is a size_t.
     *
     * There is a limit on how many device buffers can be allocated per context. User
     * can query and set this limit using the attribute
     * \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT.
     * CUPTI doesn't pre-allocate all the buffers, it pre-allocates only those many
     * buffers as set by the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE.
     * When all of the data in a buffer is consumed, it is added in the reuse pool, and
     * CUPTI picks a buffer from this pool when a new buffer is needed. Thus memory
     * footprint does not scale with the kernel count. Applications with the high density
     * of kernels, memcopies and memsets might result in having CUPTI to allocate more device buffers.
     * CUPTI allocates another buffer only when it runs out of the buffers in the
     * reuse pool.
     *
     * Since buffer allocation happens in the main application thread, this might result
     * in stalls in the critical path. CUPTI pre-allocates 3 buffers of the same size to
     * mitigate this issue. User can query and set the pre-allocation limit using the
     * attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE.
     *
     * Having larger buffer size leaves less device memory for the application.
     * Having smaller buffer size increases the risk of dropping timestamps for
     * records if too many kernels or memcopies or memsets are launched at one time.
     *
     * This value only applies to new buffer allocations. Set this value before initializing
     * CUDA or before creating a context to ensure it is considered for the following allocations.
     *
     * The default value is 3200000 (~3MB) which can accommodate profiling data
     * up to 100,000 kernels, memcopies and memsets combined.
     *
     * Note: Starting with the CUDA 11.2 release, CUPTI allocates profiling buffer in the
     * pinned host memory by default as this might help in improving the performance of the
     * tracing run. Refer to the description of the attribute
     * \ref CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED for more details.
     * Size of the memory and maximum number of pools are still controlled by the attributes
     * \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE and \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT.
     *
     * Note: The actual amount of device memory per buffer reserved by CUPTI might be larger.
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE                      = 0,
    /**
     * The device memory size (in bytes) reserved for storing profiling
     * data for CDP operations for each buffer on a context. The
     * value is a size_t.
     *
     * Having larger buffer size means less flush operations but
     * consumes more device memory. This value only applies to new
     * allocations.
     *
     * Set this value before initializing CUDA or before creating a
     * context to ensure it is considered for the following allocations.
     *
     * The default value is 8388608 (8MB).
     *
     * Note: The actual amount of device memory per context reserved by
     * CUPTI might be larger.
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP          = 1,
    /**
     * The maximum number of device memory buffers per context. The value is a size_t.
     *
     * For an application with high rate of kernel launches, memcopies and memsets having a bigger pool
     * limit helps in timestamp collection for all these activties at the expense of a larger memory footprint.
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE
     * for more details.
     *
     * Setting this value will not modify the number of memory buffers
     * currently stored.
     *
     * Set this value before initializing CUDA to ensure the limit is
     * not exceeded.
     *
     * The default value is 250.
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT                = 2,

    /**
     * The profiling semaphore pool size reserved for storing profiling data for
     * serialized kernels tracing (activity kind \ref CUPTI_ACTIVITY_KIND_KERNEL)
     * for each context. The value is a size_t.
     *
     * There is a limit on how many semaphore pools can be allocated per context. User
     * can query and set this limit using the attribute
     * \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT.
     * CUPTI doesn't pre-allocate all the semaphore pools, it pre-allocates only those many
     * semaphore pools as set by the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE.
     * When all of the data in a semaphore pool is consumed, it is added in the reuse pool, and
     * CUPTI picks a semaphore pool from the reuse pool when a new semaphore pool is needed. Thus memory
     * footprint does not scale with the kernel count. Applications with the high density
     * of kernels might result in having CUPTI to allocate more semaphore pools.
     * CUPTI allocates another semaphore pool only when it runs out of the semaphore pools in the
     * reuse pool.
     *
     * Since semaphore pool allocation happens in the main application thread, this might result
     * in stalls in the critical path. CUPTI pre-allocates 3 semaphore pools of the same size to
     * mitigate this issue. User can query and set the pre-allocation limit using the
     * attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE.
     *
     * Having larger semaphore pool size leaves less device memory for the application.
     * Having smaller semaphore pool size increases the risk of dropping timestamps for
     * kernel records if too many kernels are issued/launched at one time.
     *
     * This value only applies to new semaphore pool allocations. Set this value before initializing
     * CUDA or before creating a context to ensure it is considered for the following allocations.
     *
     * The default value is 25000 which can accommodate profiling data for upto 25,000 kernels.
     *
     */
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE           = 3,
    /**
     * The maximum number of profiling semaphore pools per context. The value is a size_t.
     *
     * For an application with high rate of kernel launches, having a bigger
     * pool limit helps in timestamp collection for all the kernels, at the
     * expense of a larger device memory footprint.
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE
     * for more details.
     *
     * Set this value before initializing CUDA to ensure the limit is not exceeded.
     *
     * The default value is 250.
     */
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT          = 4,

    /**
     * The flag to indicate whether user should provide activity buffer of zero value.
     * The value is a uint8_t.
     *
     * If the value of this attribute is non-zero, user should provide
     * a zero value buffer in the \ref CUpti_BuffersCallbackRequestFunc.
     * If the user does not provide a zero value buffer after setting this to non-zero,
     * the activity buffer may contain some uninitialized values when CUPTI returns it in
     * \ref CUpti_BuffersCallbackCompleteFunc
     *
     * If the value of this attribute is zero, CUPTI will initialize the user buffer
     * received in the \ref CUpti_BuffersCallbackRequestFunc to zero before filling it.
     * If the user sets this to zero, a few stalls may appear in critical path because CUPTI
     * will zero out the buffer in the main thread.
     * Set this value before returning from \ref CUpti_BuffersCallbackRequestFunc to
     * ensure it is considered for all the subsequent user buffers.
     *
     * The default value is 0.
     */
    CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER              = 5,

    /**
     * Number of device buffers to pre-allocate for a context during the initialization phase.
     * The value is a size_t.
     *
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE
     * for details.
     *
     * This value must be less than the maximum number of device buffers set using
     * the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT
     *
     * Set this value before initializing CUDA or before creating a context to ensure it
     * is considered by the CUPTI.
     *
     * The default value is set to 3 to ping pong between these buffers (if possible).
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE        = 6,

    /**
     * Number of profiling semaphore pools to pre-allocate for a context during the
     * initialization phase. The value is a size_t.
     *
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE
     * for details.
     *
     * This value must be less than the maximum number of profiling semaphore pools set
     * using the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT
     *
     * Set this value before initializing CUDA or before creating a context to ensure it
     * is considered by the CUPTI.
     *
     * The default value is set to 3 to ping pong between these pools (if possible).
     */
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE  = 7,

    /**
     * Allocate page-locked (pinned) host memory for storing profiling data for concurrent
     * kernels, memcopies and memsets for each buffer on a context. The value is a uint8_t.
     *
     * Starting with the CUDA 11.2 release, CUPTI allocates profiling buffer in the pinned host
     * memory by default as this might help in improving the performance of the tracing run.
     * Allocating excessive amounts of pinned memory may degrade system performance, since it
     * reduces the amount of memory available to the system for paging. For this reason user
     * might want to change the location from pinned host memory to device memory by setting
     * value of this attribute to 0.
     *
     * The default value is 1.
     */
    CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED         = 8,

    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT                 = 0x7fffffff
} CUpti_ActivityAttribute;


/**
 * \brief The base activity record.
 *
 * The activity API uses a CUpti_Activity as a generic representation
 * for any activity. The 'kind' field is used to determine the
 * specific activity kind, and from that the CUpti_Activity object can
 * be cast to the specific activity record type appropriate for that kind.
 *
 * Note that all activity record types are padded and aligned to
 * ensure that each member of the record is naturally aligned.
 *
 * \see CUpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  CUpti_ActivityKind kind;
} CUpti_Activity;




/**
 * \brief Enable collection of a specific kind of activity record.
 *
 * Enable collection of a specific kind of activity record. Multiple
 * kinds can be enabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to collect
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityEnable(CUpti_ActivityKind kind);

/**
 * \brief Disable collection of a specific kind of activity record.
 *
 * Disable collection of a specific kind of activity record. Multiple
 * kinds can be disabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to stop collecting
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityDisable(CUpti_ActivityKind kind);

/**
 * \brief Enable collection of a specific kind of activity record for
 * a context.
 *
 * Enable collection of a specific kind of activity record for a
 * context.  This setting done by this API will supersede the global
 * settings for activity records enabled by \ref cuptiActivityEnable.
 * Multiple kinds can be enabled by calling this function multiple
 * times.
 *
 * \param context The context for which activity is to be enabled
 * \param kind The kind of activity record to collect
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityEnableContext(CUcontext context, CUpti_ActivityKind kind);

/**
 * \brief Disable collection of a specific kind of activity record for
 * a context.
 *
 * Disable collection of a specific kind of activity record for a context.
 * This setting done by this API will supersede the global settings
 * for activity records.
 * Multiple kinds can be enabled by calling this function multiple times.
 *
 * \param context The context for which activity is to be disabled
 * \param kind The kind of activity record to stop collecting
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityDisableContext(CUcontext context, CUpti_ActivityKind kind);

/**
 * \brief Get the number of activity records that were dropped of
 * insufficient buffer space.
 *
 * Get the number of records that were dropped because of insufficient
 * buffer space.  The dropped count includes records that could not be
 * recorded because CUPTI did not have activity buffer space available
 * for the record (because the CUpti_BuffersCallbackRequestFunc
 * callback did not return an empty buffer of sufficient size) and
 * also CDP records that could not be record because the device-size
 * buffer was full (size is controlled by the
 * CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP attribute). The dropped
 * count maintained for the queue is reset to zero when this function
 * is called.
 *
 * \param context The context, or NULL to get dropped count from global queue
 * \param streamId The stream ID
 * \param dropped The number of records that were dropped since the last call
 * to this function.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p dropped is NULL
 */
CUptiResult CUPTIAPI cuptiActivityGetNumDroppedRecords(CUcontext context, uint32_t streamId,
                                                       size_t *dropped);

/**
 * \brief Iterate over the activity records in a buffer.
 *
 * This is a helper function to iterate over the activity records in a
 * buffer. A buffer of activity records is typically obtained by
 * receiving a CUpti_BuffersCallbackCompleteFunc callback.
 *
 * An example of typical usage:
 * \code
 * CUpti_Activity *record = NULL;
 * CUptiResult status = CUPTI_SUCCESS;
 *   do {
 *      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
 *      if(status == CUPTI_SUCCESS) {
 *           // Use record here...
 *      }
 *      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
 *          break;
 *      else {
 *          goto Error;
 *      }
 *    } while (1);
 * \endcode
 *
 * \param buffer The buffer containing activity records
 * \param record Inputs the previous record returned by
 * cuptiActivityGetNextRecord and returns the next activity record
 * from the buffer. If input value is NULL, returns the first activity
 * record in the buffer. Records of kind CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
 * may contain invalid (0) timestamps, indicating that no timing information could
 * be collected for lack of device memory.
 * \param validBufferSizeBytes The number of valid bytes in the buffer.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_MAX_LIMIT_REACHED if no more records in the buffer
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p buffer is NULL.
 */
CUptiResult CUPTIAPI cuptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes,
                                                CUpti_Activity **record);

/**
 * \brief Function type for callback used by CUPTI to request an empty
 * buffer for storing activity records.
 *
 * This callback function signals the CUPTI client that an activity
 * buffer is needed by CUPTI. The activity buffer is used by CUPTI to
 * store activity records. The callback function can decline the
 * request by setting \p *buffer to NULL. In this case CUPTI may drop
 * activity records.
 *
 * \param buffer Returns the new buffer. If set to NULL then no buffer
 * is returned.
 * \param size Returns the size of the returned buffer.
 * \param maxNumRecords Returns the maximum number of records that
 * should be placed in the buffer. If 0 then the buffer is filled with
 * as many records as possible. If > 0 the buffer is filled with at
 * most that many records before it is returned.
 */
typedef void (CUPTIAPI *CUpti_BuffersCallbackRequestFunc)(
    uint8_t **buffer,
    size_t *size,
    size_t *maxNumRecords);

/**
 * \brief Function type for callback used by CUPTI to return a buffer
 * of activity records.
 *
 * This callback function returns to the CUPTI client a buffer
 * containing activity records.  The buffer contains \p validSize
 * bytes of activity records which should be read using
 * cuptiActivityGetNextRecord. The number of dropped records can be
 * read using cuptiActivityGetNumDroppedRecords. After this call CUPTI
 * relinquished ownership of the buffer and will not use it
 * anymore. The client may return the buffer to CUPTI using the
 * CUpti_BuffersCallbackRequestFunc callback.
 * Note: CUDA 6.0 onwards, all buffers returned by this callback are
 * global buffers i.e. there is no context/stream specific buffer.
 * User needs to parse the global buffer to extract the context/stream
 * specific activity records.
 *
 * \param context The context this buffer is associated with. If NULL, the
 * buffer is associated with the global activities. This field is deprecated
 * as of CUDA 6.0 and will always be NULL.
 * \param streamId The stream id this buffer is associated with.
 * This field is deprecated as of CUDA 6.0 and will always be NULL.
 * \param buffer The activity record buffer.
 * \param size The total size of the buffer in bytes as set in
 * CUpti_BuffersCallbackRequestFunc.
 * \param validSize The number of valid bytes in the buffer.
 */
typedef void (CUPTIAPI *CUpti_BuffersCallbackCompleteFunc)(
    CUcontext context,
    uint32_t streamId,
    uint8_t *buffer,
    size_t size,
    size_t validSize);

/**
 * \brief Registers callback functions with CUPTI for activity buffer
 * handling.
 *
 * This function registers two callback functions to be used in asynchronous
 * buffer handling. If registered, activity record buffers are handled using
 * asynchronous requested/completed callbacks from CUPTI.
 *
 * Registering these callbacks prevents the client from using CUPTI's
 * blocking enqueue/dequeue functions.
 *
 * \param funcBufferRequested callback which is invoked when an empty
 * buffer is requested by CUPTI
 * \param funcBufferCompleted callback which is invoked when a buffer
 * containing activity records is available from CUPTI
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if either \p
 * funcBufferRequested or \p funcBufferCompleted is NULL
 */
CUptiResult CUPTIAPI cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc funcBufferRequested,
        CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

/**
 * \brief Wait for all activity records to be delivered via the
 * completion callback.
 *
 * This function does not return until all activity records associated
 * with the specified context/stream are returned to the CUPTI client
 * using the callback registered in cuptiActivityRegisterCallbacks. To
 * ensure that all activity records are complete, the requested
 * stream(s), if any, are synchronized.
 *
 * If \p context is NULL, the global activity records (i.e. those not
 * associated with a particular stream) are flushed (in this case no
 * streams are synchonized).  If \p context is a valid CUcontext and
 * \p streamId is 0, the buffers of all streams of this context are
 * flushed.  Otherwise, the buffers of the specified stream in this
 * context is flushed.
 *
 * Before calling this function, the buffer handling callback api
 * must be activated by calling cuptiActivityRegisterCallbacks.
 *
 * \param context A valid CUcontext or NULL.
 * \param streamId The stream ID.
 * \param flag The flag can be set to indicate a forced flush. See CUpti_ActivityFlag
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_CUPTI_ERROR_INVALID_OPERATION if not preceeded
 * by a successful call to cuptiActivityRegisterCallbacks
 * \retval CUPTI_ERROR_UNKNOWN an internal error occurred
 *
 * **DEPRECATED** This method is deprecated
 * CONTEXT and STREAMID will be ignored. Use cuptiActivityFlushAll
 * to flush all data.
 */
CUptiResult CUPTIAPI cuptiActivityFlush(CUcontext context, uint32_t streamId, uint32_t flag);

/**
 * \brief Request to deliver activity records via the buffer completion callback.
 *
 * This function returns the activity records associated with all contexts/streams
 * (and the global buffers not associated with any stream) to the CUPTI client
 * using the callback registered in cuptiActivityRegisterCallbacks.
 *
 * This is a blocking call but it doesn't issue any CUDA synchronization calls
 * implicitly thus it's not guaranteed that all activities are completed on the
 * underlying devices. Activity record is considered as completed if it has all
 * the information filled up including the timestamps if any. It is the client's
 * responsibility to issue necessary CUDA synchronization calls before calling
 * this function if all activity records with complete information are expected
 * to be delivered.
 *
 * Behavior of the function based on the input flag:
 * - ::For default flush i.e. when flag is set as 0, it returns all the
 * activity buffers which have all the activity records completed, buffers need not
 * to be full though. It doesn't return buffers which have one or more incomplete
 * records. Default flush can be done at a regular interval in a separate thread.
 * - ::For forced flush i.e. when flag CUPTI_ACTIVITY_FLAG_FLUSH_FORCED is passed
 * to the function, it returns all the activity buffers including the ones which have
 * one or more incomplete activity records. It's suggested for clients to do the
 * force flush before the termination of the profiling session to allow remaining
 * buffers to be delivered. In general, it can be done in the at-exit handler.
 *
 * Before calling this function, the buffer handling callback api must be activated
 * by calling cuptiActivityRegisterCallbacks.
 *
 * \param flag The flag can be set to indicate a forced flush. See CUpti_ActivityFlag
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if not preceeded by a
 * successful call to cuptiActivityRegisterCallbacks
 * \retval CUPTI_ERROR_UNKNOWN an internal error occurred
 *
 * \see cuptiActivityFlushPeriod
 */
CUptiResult CUPTIAPI cuptiActivityFlushAll(uint32_t flag);


/**
 * \brief The kind of external APIs supported for correlation.
 *
 * Custom correlation kinds are reserved for usage in external tools.
 *
 * \see CUpti_ActivityExternalCorrelation
 */
typedef enum {
    CUPTI_EXTERNAL_CORRELATION_KIND_INVALID              = 0,

    /**
     * The external API is unknown to CUPTI
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN              = 1,

    /**
     * The external API is OpenACC
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC              = 2,

    /**
     * The external API is custom0
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0              = 3,

    /**
     * The external API is custom1
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1              = 4,

    /**
     * The external API is custom2
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2              = 5,

    /**
     * Add new kinds before this line
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_SIZE,

    CUPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT            = 0x7fffffff
} CUpti_ExternalCorrelationKind;

/**
 * \brief The activity record for correlation with external records
 *
 * This activity record correlates native CUDA records (e.g. CUDA Driver API,
 * kernels, memcpys, ...) with records from external APIs such as OpenACC.
 * (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION).
 *
 * \see CUpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of external API this record correlated to.
   */
  CUpti_ExternalCorrelationKind externalKind;

  /**
   * The correlation ID of the associated non-CUDA API record.
   * The exact field in the associated external record depends
   * on that record's activity kind (\see externalKind).
   */
  uint64_t externalId;

  /**
   * The correlation ID of the associated CUDA driver or runtime API record.
   */
  uint32_t correlationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t reserved;
} CUpti_ActivityExternalCorrelation;

/**
 * \brief Push an external correlation id for the calling thread
 *
 * This function notifies CUPTI that the calling thread is entering an external API region.
 * When a CUPTI activity API record is created while within an external API region and
 * CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION is enabled, the activity API record will
 * be preceeded by a CUpti_ActivityExternalCorrelation record for each \ref CUpti_ExternalCorrelationKind.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param id External correlation id.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER The external API kind is invalid
 */
CUptiResult CUPTIAPI cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind, uint64_t id);

/**
 * \brief Pop an external correlation id for the calling thread
 *
 * This function notifies CUPTI that the calling thread is leaving an external API region.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param lastId If the function returns successful, contains the last external correlation id for this \p kind, can be NULL.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER The external API kind is invalid.
 * \retval CUPTI_ERROR_QUEUE_EMPTY No external id is currently associated with \p kind.
 */
CUptiResult CUPTIAPI cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind, uint64_t *lastId);



#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_ACTIVITY_H_*/
