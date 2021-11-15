// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <thread>
#include <iostream>

#include "iree/hal/cuda/tracing.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#include "third_party/tracy/Tracy.hpp"
#include "third_party/tracy/client/TracyProfiler.hpp"
#include "third_party/tracy/common/TracyAlloc.hpp"
#include "iree/base/status.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/status_util.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/call_once.h"


// === Global State  ===

typedef struct {
  iree_slim_mutex_t mutex;

  iree_hal_cuda_tracing_context_t* tracing_context;


} iree_hal_cuda_tracing_context_global_t;

static iree_hal_cuda_tracing_context_global_t iree_hal_cuda_tracing_context_global_;
static iree_once_flag iree_hal_cuda_tracing_initialize_flag_ =
    IREE_ONCE_FLAG_INIT;

static void iree_hal_cuda_tracing_global_initialize(void) {
  memset(&iree_hal_cuda_tracing_context_global_, 0,
         sizeof(iree_hal_cuda_tracing_context_global_));
  iree_slim_mutex_initialize(&iree_hal_cuda_tracing_context_global_.mutex);
}

void iree_hal_cuda_tracing_globals_initialize(void) {
  iree_call_once(&iree_hal_cuda_tracing_initialize_flag_,
                 iree_hal_cuda_tracing_global_initialize);
}

// === /Global State  ===


#define ID_THREAD(thread_name) \
  std::thread::id this_id = std::this_thread::get_id(); \
  std::cout << "The " << thread_name << " thread: " << this_id << std::endl;

// Default size for each host-allocated CPUTI activity buffer for receiving results from the CUPTI activity api.
#define IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_SIZE (32 * 1024)

// Number of hal-allocated CUPTI activity buffers for tracing data collection.
#define IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_POOL_CAPACITY (32)

// Total number of queries the per-queue query pool will contain. This
// translates to the maximum number of outstanding queries before collection is
// required.
#define IREE_HAL_CUDA_TRACING_DEFAULT_QUERY_CAPACITY (64 * 1024)



// Keeps track of of allocated and assigned activity buffers requested by CUPTI.
typedef struct iree_hal_cuda_tracing_activity_buffer_t {
  uint8_t *buffer;
  size_t size;
  bool returned; // Set to true when this buffer is returned from CUPTI

  size_t valid_size;
} iree_hal_cuda_tracing_activity_buffer_t;

// Keeps track of command buffer begin/end timing for CUPTI events.
typedef struct iree_hal_cuda_tracing_command_buffer_t {
  bool active;
  uint32_t external_correlation_id;
  uint32_t query_start_id;
  uint32_t query_end_id;
  uint64_t first_timestamp;
  uint64_t last_timestamp;
  uint32_t kernel_count;
} iree_hal_cuda_tracing_command_buffer_t;

struct iree_hal_cuda_tracing_context_t {
  iree_hal_cuda_context_wrapper_t context_wrapper;
  iree_allocator_t host_allocator;
  iree_hal_cuda_dynamic_symbols_t* syms;

  // TODO GOES HERE:  queue/stream hook?

  // A unique GPU zone ID allocated from Tracy.
  // There is a global limit of 255 GPU zones (ID 255 is special).
  uint8_t id;

  // Indices into |query_pool| defining a ringbuffer.
  uint32_t query_head;
  uint32_t query_tail;
  uint32_t query_capacity;
  uint32_t outstanding_query_count;

  // Activity buffer pool
  iree_hal_cuda_tracing_activity_buffer_t buffer_pool[IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_POOL_CAPACITY];
  uint32_t buffer_pool_head;
  uint32_t buffer_pool_tail;
  uint32_t buffer_pool_capacity;
  uint32_t outstanding_buffers; // Number of buffers acquired or returned but not yet processed
  // TODO(kooljblack) buffer pool locks

  // Dispatch command buffer tracking
  iree_hal_cuda_tracing_command_buffer_t command_buffers[IREE_HAL_CUDA_TRACING_DEFAULT_QUERY_CAPACITY];
  uint32_t command_buffer_capacity;
  uint32_t command_buffer_head;

  uint64_t last_external_correlation_id;
  uint64_t last_correlation_id;
};

static iree_hal_cuda_tracing_command_buffer_t* iree_hal_cuda_tracing_find_command_buffer(
    iree_hal_cuda_tracing_context_t* tracing_context, uint64_t external_correlation_id) {

  uint32_t buffer_index = tracing_context->command_buffer_head;
  iree_hal_cuda_tracing_command_buffer_t* command_buffer;

  do {
    // printf("loop\n");
    command_buffer = &tracing_context->command_buffers[buffer_index];
    assert(command_buffer->active);
    if (command_buffer->external_correlation_id == external_correlation_id) {
      break;
    }
    if (buffer_index == 0) {
      buffer_index = tracing_context->command_buffer_capacity-1;
    } else {
      buffer_index--;
    }
  } while(true);
  return command_buffer;
}

static void iree_hal_cuda_process_activity_record(iree_hal_cuda_tracing_context_t *tracing_context, CUpti_Activity *record) {
  // printf("I received an activity record! -- kind:%d\n", record->kind);
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_RUNTIME: {
      // Ignore driver events since they provide redundant information.
      static int count = 0;
      count = count + 1;
      printf("The runtime count is %d\n", count);
      break;
    }
    case CUPTI_ACTIVITY_KIND_DRIVER: {
      // Ignore driver events since they provide redundant information.
      static int count = 0;
      count = count + 1;
      // printf("The driver count is %d\n", count);
      break;
    }

    // External correlation records update the last received ID pair.
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
      static int count = 0;
      count = count + 1;
      // printf("The correlation count is %d\n", count);

      CUpti_ActivityExternalCorrelation *correlation = (CUpti_ActivityExternalCorrelation *)record;

      // printf("\nEXTERNAL_CORRELATION: correlation %u, external %llu\n",
      //         correlation->correlationId,
      //         (unsigned long long) correlation->externalId);

      tracing_context->last_correlation_id = correlation->correlationId;
      tracing_context->last_external_correlation_id = correlation->externalId;
      break;
    }

    // Kernel records provid kernel timesing as well as possibly the first/last timestamps for command buffers
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      static int count = 0;
      count = count + 1;
      // printf("The kernel count is %d\n", count);

      CUpti_ActivityKernel6 *kernel = (CUpti_ActivityKernel6 *) record;
      // printf("Name: %s start: %lu end: %lu\n", kernel->name, kernel->start, kernel->end);

      // TODO(KoolJBlack): why do we downcast the query ids?

      // Find the relevant command buffer
      assert(tracing_context->last_correlation_id == kernel->correlationId); // Correlation IDs should always match the next record
      uint64_t external_correlation_id = tracing_context->last_external_correlation_id;
      iree_hal_cuda_tracing_command_buffer_t* command_buffer = iree_hal_cuda_tracing_find_command_buffer(tracing_context, external_correlation_id);
      assert(command_buffer); // We shouldn't have kernels that don't fall within a buffer.

      // printf("The kernel correlation id: %d last correlation id: %lu\n", kernel->correlationId, tracing_context->last_correlation_id);

      // Handle the first timestamp in a command buffer.
      if (tracing_context->query_tail == command_buffer->query_start_id) { // Are we at the beginning of the command buffer?
        command_buffer->first_timestamp = kernel->start;

        printf("Command buffer start! Start query_id: %u  timestamp: %lu\n",command_buffer->query_start_id, command_buffer->first_timestamp );

        auto* item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
        tracy::MemWrite(&item->gpuTime.gpuTime, command_buffer->first_timestamp);
        tracy::MemWrite(&item->gpuTime.queryId, (uint16_t)(command_buffer->query_start_id));
        tracy::MemWrite(&item->gpuTime.context, tracing_context->id);
        tracy::Profiler::QueueSerialFinish();

        // Advance the queue and update outsanding
        tracing_context->query_tail++;
        if (tracing_context->query_tail >= tracing_context->query_capacity) {
          tracing_context->query_tail = 0;
        }
        tracing_context->outstanding_query_count --;
      }

      // Record the start timestamp
      uint64_t start_timestamp = kernel->start;
      uint32_t start_query = tracing_context->query_tail;

      auto* item = tracy::Profiler::QueueSerial();
      tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
      tracy::MemWrite(&item->gpuTime.gpuTime, start_timestamp);
      tracy::MemWrite(&item->gpuTime.queryId, (uint16_t)(start_query));
      tracy::MemWrite(&item->gpuTime.context, tracing_context->id);
      tracy::Profiler::QueueSerialFinish();

      // Record the end timestamp
      uint64_t end_timestamp = kernel->end;
      uint32_t end_query = tracing_context->query_tail + 1;

      item = tracy::Profiler::QueueSerial();
      tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
      tracy::MemWrite(&item->gpuTime.gpuTime, end_timestamp);
      tracy::MemWrite(&item->gpuTime.queryId, (uint16_t)(end_query));
      tracy::MemWrite(&item->gpuTime.context, tracing_context->id);
      tracy::Profiler::QueueSerialFinish();

      // Advance the queue and update outsanding
      tracing_context->query_tail += 2; // read start/stop
      if (tracing_context->query_tail >= tracing_context->query_capacity) {
        tracing_context->query_tail = 0;
      }
      tracing_context->outstanding_query_count -= 2;
      // printf("After the end, the query tail: %u\n", tracing_context->query_tail);


      // Handle the final timestamp in a command buffer.
      if (tracing_context->query_tail == command_buffer->query_end_id) { // Are we at the end of the command buffer?
        command_buffer->last_timestamp = kernel->end;

        printf("Command buffer end! End query_id: %u  timestamp: %lu\n",command_buffer->query_end_id, command_buffer->last_timestamp );

        auto* item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
        tracy::MemWrite(&item->gpuTime.gpuTime, command_buffer->last_timestamp);
        tracy::MemWrite(&item->gpuTime.queryId, (uint16_t)(command_buffer->query_end_id));
        tracy::MemWrite(&item->gpuTime.context, tracing_context->id);
        tracy::Profiler::QueueSerialFinish();

        // Advance the queue and update outsanding
        tracing_context->query_tail++;
        if (tracing_context->query_tail >= tracing_context->query_capacity) {
          tracing_context->query_tail = 0;
        }
        tracing_context->outstanding_query_count --;

        // Cleanup the command buffer
        command_buffer->active = false;
      }

      break;
    }
    default: {
      printf("I got a default case");
      break;
    }
  }
}

void iree_hal_cuda_tracing_acquire_activity_buffer(
  iree_hal_cuda_tracing_context_t *tracing_context,
  iree_hal_cuda_tracing_activity_buffer_t **out_buffer) {
    iree_hal_cuda_tracing_activity_buffer_t* activity_buffer = &tracing_context->buffer_pool[tracing_context->buffer_pool_head];
    tracing_context->buffer_pool_head = (tracing_context->buffer_pool_head + 1) % tracing_context->buffer_pool_capacity;
    tracing_context->outstanding_buffers++;
    assert(tracing_context->buffer_pool_head != tracing_context->buffer_pool_tail);

    *out_buffer = activity_buffer;

    printf("iree_hal_cuda_tracing_acquire_activity_buffer(): outstanding_buffers: %d\n", tracing_context->outstanding_buffers);
}

void iree_hal_cuda_tracing_return_activity_buffer(
  iree_hal_cuda_tracing_context_t *tracing_context,
  uint8_t *buffer, size_t valid_size) {
  // Find the activity_buffer
  uint32_t buffer_index = -1;
  for (uint32_t i = 0; i < tracing_context->buffer_pool_capacity; ++i) {
    if (tracing_context->buffer_pool[i].buffer == buffer) {
      buffer_index = i;
      break;
    }
  }
  assert(buffer_index != -1);

  printf("Returning activity buffer at index: %d\n", buffer_index);
  iree_hal_cuda_tracing_activity_buffer_t* activity_buffer = &tracing_context->buffer_pool[buffer_index];
  activity_buffer->returned = true;
  activity_buffer->valid_size = valid_size;
}

void iree_hal_cuda_tracing_process_activity_buffer(
  iree_hal_cuda_tracing_context_t *tracing_context,
  iree_hal_cuda_tracing_activity_buffer_t* activity_buffer) {

  uint8_t *buffer = activity_buffer->buffer;
  size_t valid_size = activity_buffer->valid_size;

  if (valid_size > 0) {
    printf("Starting valid size\n");
    CUptiResult status;
    CUpti_Activity *record = NULL;
    int count = 0;
    do {
      status = tracing_context->syms->cuptiActivityGetNextRecord(buffer, valid_size, &record);
      if (status == CUPTI_SUCCESS) {
        iree_hal_cuda_process_activity_record(tracing_context, record);
        count++;
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        printf("Breaking at record %d \n", count);
        break;
      } else {
        // TODO (koolblack) this should never happen, error and fail?
        // CUPTI_IGNORE_ERROR(status);
        break;
      }
    } while (1);


  } else {
    printf("I failed valid size \n");
  }
}

bool iree_hal_cuda_tracing_query_returned_activity_buffers(
  iree_hal_cuda_tracing_context_t* tracing_context,
  uint32_t* out_buffer_base,
  uint32_t* out_buffer_head) {
  uint32_t returned_buffer_base = tracing_context->buffer_pool_tail;
  uint32_t returned_buffer_head = returned_buffer_base;
  bool pool_rollover = false;

  do {
    iree_hal_cuda_tracing_activity_buffer_t* activity_buffer = &tracing_context->buffer_pool[returned_buffer_head];
    if (!activity_buffer->returned) {
      break;
    }

    returned_buffer_head++;

    if(returned_buffer_head == tracing_context->buffer_pool_capacity) {
      returned_buffer_head = 0;
      pool_rollover = true;
      break;
    }

  } while (true);

  *out_buffer_base = returned_buffer_base;
  *out_buffer_head = returned_buffer_head;
  return pool_rollover;
}



void iree_hal_cuda_tracing_free_activity_buffer(
  iree_hal_cuda_tracing_context_t *tracing_context,
  iree_hal_cuda_tracing_activity_buffer_t* activity_buffer) {
  // TODO(kooljblack): do I need to zero the buffer for reuse?

  // Clear buffer contents
  activity_buffer->returned = false;
  activity_buffer->valid_size = -1;

  // Update pool
  tracing_context->buffer_pool_tail = (tracing_context->buffer_pool_tail + 1) % tracing_context->buffer_pool_capacity;
  tracing_context->outstanding_buffers--;
  // assert(tracing_context->buffer_pool_head != tracing_context->buffer_pool_tail);
}

static void iree_hal_cuda_tracing_buffer_requested(
uint8_t **buffer, size_t *size, size_t *max_num_records) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_tracing_context_t *tracing_context = iree_hal_cuda_tracing_context_global_.tracing_context;

  // iree_allocator_malloc(tracing_context->host_allocator, IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_SIZE, (void **)buffer);
  // *size = IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_SIZE;
  // *max_num_records = 0;
  iree_hal_cuda_tracing_activity_buffer_t* activity_buffer;
  iree_hal_cuda_tracing_acquire_activity_buffer(tracing_context, &activity_buffer);
  *buffer = activity_buffer->buffer;
  *size = activity_buffer->size;
  *max_num_records = 0;

  printf("tracing_buffer_requested(): size: %zu buffer: %p max_num_records: %zu\n", *size, buffer, *max_num_records);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_cuda_tracing_buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t *buffer, size_t size, size_t valid_size) {
  IREE_TRACE_ZONE_BEGIN(z0);

  printf("tracing_buffer_completed(): buffer: %p size: %zu valid_size: %zu\n", buffer, size, valid_size);

  ID_THREAD("buffer_completed");
  //TODO(kooljblack): These tracing callbacks aren't on main. Need to lock or handoff.

  iree_hal_cuda_tracing_context_t *tracing_context = iree_hal_cuda_tracing_context_global_.tracing_context;

  iree_hal_cuda_tracing_return_activity_buffer(tracing_context, buffer, valid_size);


  if (valid_size) {
    // Report any records dropped from the queue
    size_t dropped;
    CUPTI_IGNORE_ERROR(tracing_context->syms, cuptiActivityGetNumDroppedRecords(
                                                  ctx, stream_id, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int)dropped);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

// Prepares the Tracy-related GPU context that events are fed into. Each context
// will appear as a unique plot in the tracy UI with the given |queue_name|.
static void iree_hal_cuda_tracing_prepare_gpu_context(
  iree_hal_cuda_tracing_context_t *tracing_context,
    iree_string_view_t queue_name) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the process-unique GPU context ID. There's a max of 255 available;
  // if we are recreating devices a lot we may exceed that. Don't do that, or
  // wrap around and get weird (but probably still usable) numbers.
  tracing_context->id =
      tracy::GetGpuCtxCounter().fetch_add(1, std::memory_order_relaxed);
  if (tracing_context->id >= 255) {
    tracing_context->id %= 255;
  }

  // Since CUDA doesn't expose automated host/device timestamp callibration.
  uint8_t context_flags = 0;  // No Callibration

  // Provide timestamp basis by querying both the CPU and GPU.
  uint64_t tracy_timestamp = tracy::Profiler::GetTime();
  uint64_t cupti_timestamp;
  CUPTI_IGNORE_ERROR(tracing_context->syms, cuptiGetTimestamp(&cupti_timestamp));

  // Tracy lacks a CUDA context, Vulkan is used for now.
  tracy::GpuContextType context_type = tracy::GpuContextType::Vulkan;

  float timestamp_period = 1.0f;

  printf("iree_hal_cuda_tracing_callibrate_timestamps(): \n -- tracy: %lu\n -- cupti: %lu \n", tracy_timestamp, cupti_timestamp);

  {
    auto* item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuNewContext);
    tracy::MemWrite(&item->gpuNewContext.cpuTime, tracy_timestamp);
    tracy::MemWrite(&item->gpuNewContext.gpuTime, cupti_timestamp);
    memset(&item->gpuNewContext.thread, 0, sizeof(item->gpuNewContext.thread));
    tracy::MemWrite(&item->gpuNewContext.period, timestamp_period);
    tracy::MemWrite(&item->gpuNewContext.context, tracing_context->id);
    tracy::MemWrite(&item->gpuNewContext.flags, context_flags);
    tracy::MemWrite(&item->gpuNewContext.type, context_type);
    tracy::Profiler::QueueSerialFinish();
  }

  // Send the name of the context along.
  // NOTE: Tracy will unconditionally free the name so we must clone it here.
  // Since internally Tracy will use its own rpmalloc implementation we must
  // make sure we allocate from the same source.
  char* cloned_name = (char*)tracy::tracy_malloc(queue_name.size);
  memcpy(cloned_name, queue_name.data, queue_name.size);
  {
    auto* item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuContextName);
    tracy::MemWrite(&item->gpuContextNameFat.context, tracing_context->id);
    tracy::MemWrite(&item->gpuContextNameFat.ptr, (uint64_t)cloned_name);
    tracy::MemWrite(&item->gpuContextNameFat.size, queue_name.size);
    tracy::Profiler::QueueSerialFinish();
  }
  IREE_TRACE_ZONE_END(z0);
}

// Prepares the tracy query pool backing storage for our query ringbuffer.
static void iree_hal_cuda_tracing_prepare_query_pool(
    iree_hal_cuda_tracing_context_t* tracing_context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  tracing_context->query_capacity = IREE_HAL_CUDA_TRACING_DEFAULT_QUERY_CAPACITY;

  printf("iree_hal_cuda_tracing_prepare_query_pool()): query_capacity: %d \n", tracing_context->query_capacity);
  IREE_TRACE_ZONE_END(z0);

}

// Prepares the cupti activity buffer pool for collecting trace timestamps.
static iree_status_t iree_hal_cuda_tracing_prepare_buffer_pool(
    iree_hal_cuda_tracing_context_t* tracing_context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  tracing_context->buffer_pool_capacity =
      IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_POOL_CAPACITY;

  // Allocate buffers
  for (uint32_t i = 0; i < tracing_context->buffer_pool_capacity; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(tracing_context->host_allocator,
                              IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_SIZE,
                              (void**)&tracing_context->buffer_pool[i]));
    tracing_context->buffer_pool[i].size = IREE_HAL_CUDA_TRACING_CUPTI_ACTIVITY_BUFFER_SIZE;
    tracing_context->buffer_pool[i].returned = false;
  }

  printf(
      "iree_hal_cuda_tracing_prepare_buffer_pool()): buffer_pool_capacity: %d "
      "\n",
      tracing_context->buffer_pool_capacity);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda_tracing_prepare_command_buffers(
    iree_hal_cuda_tracing_context_t* tracing_context) {
    IREE_TRACE_ZONE_BEGIN(z0);

  tracing_context->command_buffer_capacity = IREE_ARRAYSIZE(tracing_context->command_buffers);
  for (uint32_t i = 0; i < tracing_context->command_buffer_capacity; ++i) {

    tracing_context->command_buffers[i].active=false;
    tracing_context->command_buffers[i].external_correlation_id=i; // Using list position ensures unique ids per buffer.

  }
  IREE_TRACE_ZONE_END(z0);

}

iree_status_t iree_hal_cuda_tracing_context_allocate(
    iree_hal_cuda_context_wrapper_t context_wrapper,
    iree_allocator_t host_allocator,
    iree_hal_cuda_dynamic_symbols_t *syms,
    iree_string_view_t queue_name,
    iree_hal_cuda_tracing_context_t **out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);


  // Allocate the context
  iree_hal_cuda_tracing_context_t *tracing_context = NULL;
  iree_status_t status = iree_allocator_malloc(host_allocator, sizeof(*tracing_context),
                                               (void **)&tracing_context);

  if (iree_status_is_ok(status)) {
    // TODO: Set context members
    tracing_context->context_wrapper = context_wrapper;
    tracing_context->host_allocator = host_allocator;
    tracing_context->syms = syms;

    // Set global pointer
    iree_hal_cuda_tracing_context_global_.tracing_context = tracing_context;

    // Enable device activity records for kernal invocation
    printf("iree_hal_cuda_tracing_context_allocate(): registering callbacks\n");
    CUPTI_RETURN_IF_ERROR(
        syms,
        cuptiActivityRegisterCallbacks(iree_hal_cuda_tracing_buffer_requested,
                                       iree_hal_cuda_tracing_buffer_completed),
        "registering callbacks");
    CUPTI_RETURN_IF_ERROR(
        syms, cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER),
        "enabling driver records");
    // CUPTI_RETURN_IF_ERROR(
    //     syms, cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),
    //     "enabling runtime records");
    CUPTI_RETURN_IF_ERROR(
        syms, cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION),
        "enabling external correlation records");
    CUPTI_RETURN_IF_ERROR(
        syms, cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
        "enabling kernel records");
    iree_hal_cuda_tracing_prepare_gpu_context(tracing_context, queue_name);

    iree_hal_cuda_tracing_prepare_query_pool(tracing_context);

    iree_hal_cuda_tracing_prepare_command_buffers(tracing_context);

    IREE_RETURN_IF_ERROR(iree_hal_cuda_tracing_prepare_buffer_pool(tracing_context));
  }

  if (iree_status_is_ok(status)) {
    *out_context = tracing_context;
  } else {
    iree_hal_cuda_tracing_context_free(tracing_context);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda_tracing_context_free(iree_hal_cuda_tracing_context_t* tracing_context) {
  if (!tracing_context) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  printf("iree_hal_cuda_tracing_context_free():\n");

  iree_hal_cuda_tracing_sync(tracing_context);

  ID_THREAD("context_free");

  iree_allocator_t host_allocator = tracing_context->host_allocator;
  iree_allocator_free(host_allocator, tracing_context);

  IREE_TRACE_ZONE_END(z0);
}

uint32_t iree_hal_cuda_tracing_context_acquire_query_id(
        iree_hal_cuda_tracing_context_t* tracing_context) {
  uint32_t id = tracing_context->query_head;
  tracing_context->query_head = (tracing_context->query_head + 1) % tracing_context->query_capacity;
  assert(tracing_context->query_head != tracing_context->query_tail); // We've run out of query ids.
  tracing_context->outstanding_query_count++;
  // printf("iree_hal_cuda_tracing_context_acquire_query_id(): query_head: %d, query_tail: %d\n", tracing_context->query_head, tracing_context->query_tail);
  return id;
}

void iree_hal_cuda_tracing_sync(
    iree_hal_cuda_tracing_context_t* tracing_context) {
  if (!tracing_context) {return;}
  if (tracing_context->query_tail == tracing_context->query_head) {
    // No outstanding tracing queries.
    printf("iree_hal_cuda_tracing_sync(): no outstanding queries \n");
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  printf("iree_hal_cuda_tracing_sync(): start \n");

  // Flush all outsanding buffers back from CUPTI
  CUPTI_IGNORE_ERROR(tracing_context->syms, cuptiActivityFlushAll(0));

  // Process returned activity records

  // Determine the returned range first (this can later be delegated to a synced method)
  uint32_t returned_buffer_base = 0;
  uint32_t returned_buffer_head = 0;
  bool pool_rollover = iree_hal_cuda_tracing_query_returned_activity_buffers(
    tracing_context, &returned_buffer_base, &returned_buffer_head);


  printf("Processing returned buffers from %d - %d  rollover: %d\n", returned_buffer_base, returned_buffer_head, pool_rollover);


  // Process all the returned contiguous records
  for (int i = returned_buffer_base; i < returned_buffer_head; ++i) {
    iree_hal_cuda_tracing_activity_buffer_t* activity_buffer = &tracing_context->buffer_pool[i];
    iree_hal_cuda_tracing_process_activity_buffer(tracing_context, activity_buffer);
  }

  // Free/release the process records (can later be delegated to a synced method. )
  for (int i = returned_buffer_base; i < returned_buffer_head; ++i) {
    iree_hal_cuda_tracing_activity_buffer_t* activity_buffer = &tracing_context->buffer_pool[i];
    iree_hal_cuda_tracing_free_activity_buffer(tracing_context, activity_buffer);
  }

  printf("iree_hal_cuda_tracing_sync(): end \n");

  IREE_TRACE_ZONE_END(z0);
}

// Begins a normal zone derived on the calling |src_loc|.
// Must be perfectly nested and paired with a corresponding zone end.
void iree_hal_cuda_tracing_kernel_zone_begin_impl(
    iree_hal_cuda_tracing_context_t* tracing_context,
    const iree_tracing_location_t* src_loc) {
  if (!tracing_context) {return;}
  // printf("iree_hal_cuda_tracing_kernel_zone_begin_impl():\n");


  uint32_t query_id = iree_hal_cuda_tracing_context_acquire_query_id(tracing_context);

  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneBegin.queryId, (uint16_t)query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, tracing_context->id);
  tracy::Profiler::QueueSerialFinish();

  }

// Begins an external zone using the given source information.
// The provided strings will be copied into the tracy buffer.
void iree_hal_cuda_tracing_kernel_zone_begin_external_impl(
    iree_hal_cuda_tracing_context_t* tracing_context,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  // printf("iree_hal_cuda_tracing_kernel_zone_begin_external_impl(): \n");


  const auto src_loc = tracy::Profiler::AllocSourceLocation(
      line, file_name, file_name_length, function_name, function_name_length,
      name, name_length);

  uint32_t query_id = iree_hal_cuda_tracing_context_acquire_query_id(tracing_context);

  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginAllocSrcLocSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneBegin.queryId, (uint16_t)query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, tracing_context->id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_hal_cuda_tracing_kernel_zone_end_impl(
    iree_hal_cuda_tracing_context_t* tracing_context) {
  if (!tracing_context) {return;}
  // printf("iree_hal_cuda_tracing_kernel_zone_end_impl():\n");

  uint32_t query_id = iree_hal_cuda_tracing_context_acquire_query_id(tracing_context);

  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
  tracy::MemWrite(&item->gpuZoneEnd.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneEnd.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneEnd.queryId, (uint16_t)query_id);
  tracy::MemWrite(&item->gpuZoneEnd.context, tracing_context->id);
  tracy::Profiler::QueueSerialFinish();

  // Advance the kernel count in the current command buffer
  tracing_context->command_buffers[tracing_context->command_buffer_head].kernel_count++;
}

void iree_hal_cuda_tracing_command_buffer_zone_begin_impl(
    iree_hal_cuda_tracing_context_t* tracing_context,
    const iree_tracing_location_t* src_loc) {

  uint32_t query_id = iree_hal_cuda_tracing_context_acquire_query_id(tracing_context);

  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneBegin.queryId, (uint16_t)query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, tracing_context->id);
  tracy::Profiler::QueueSerialFinish();

  // TODO(kooljblack): assert the last command buffer was completed before this new begin

  // Assign the next command buffer
  iree_hal_cuda_tracing_command_buffer_t* command_buffer = &tracing_context->command_buffers[tracing_context->command_buffer_head];
  if (command_buffer->active) { // This buffer is active, advance to the next one
    tracing_context->command_buffer_head++;
    tracing_context->command_buffer_head%=tracing_context->command_buffer_capacity;
    command_buffer = &tracing_context->command_buffers[tracing_context->command_buffer_head];
  }

  command_buffer->active = true;
  command_buffer->query_start_id = query_id;
  command_buffer->kernel_count = 0;
  CUPTI_IGNORE_ERROR(tracing_context->syms, cuptiGetTimestamp(&command_buffer->first_timestamp));

  printf(
      "iree_hal_cuda_tracing_command_buffer_zone_begin_impl(): external: %u "
      "start_query: %d\n",
      command_buffer->external_correlation_id, command_buffer->query_start_id);

  // Push id
  CUPTI_IGNORE_ERROR(tracing_context->syms, cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(command_buffer->external_correlation_id)));
}

void iree_hal_cuda_tracing_command_buffer_zone_end_impl(
    iree_hal_cuda_tracing_context_t* tracing_context) {

  uint32_t query_id = iree_hal_cuda_tracing_context_acquire_query_id(tracing_context);

  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
  tracy::MemWrite(&item->gpuZoneEnd.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneEnd.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneEnd.queryId, (uint16_t)query_id);
  tracy::MemWrite(&item->gpuZoneEnd.context, tracing_context->id);
  tracy::Profiler::QueueSerialFinish();

  // TODO(kooljblack): assert the last command buffer hasn't ended yet
  iree_hal_cuda_tracing_command_buffer_t* command_buffer = &tracing_context->command_buffers[tracing_context->command_buffer_head];
  command_buffer->query_end_id = query_id;
  CUPTI_IGNORE_ERROR(tracing_context->syms, cuptiGetTimestamp(&command_buffer->last_timestamp));

  // printf("iree_hal_cuda_tracing_command_buffer_zone_end_impl(): DONE \n");

  printf(
    "iree_hal_cuda_tracing_command_buffer_zone_end_impl(): external: %u "
    "end_query: %d\n",
    command_buffer->external_correlation_id, command_buffer->query_end_id);


  // Pop Id
  uint64_t id = 0;
  CUPTI_IGNORE_ERROR(tracing_context->syms, cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));

  // For zero kernel command buffers, skip their GPU zone since we won't receive any callbacks.
  if(command_buffer->kernel_count == 0) {
    // Record the start timestamp
    uint64_t start_timestamp = command_buffer->first_timestamp;
    uint32_t start_query = command_buffer->query_start_id;

    auto* item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
    tracy::MemWrite(&item->gpuTime.gpuTime, start_timestamp);
    tracy::MemWrite(&item->gpuTime.queryId, (uint16_t)(start_query));
    tracy::MemWrite(&item->gpuTime.context, tracing_context->id);
    tracy::Profiler::QueueSerialFinish();

    // Record the end timestamp
    uint64_t end_timestamp = command_buffer->last_timestamp;
    uint32_t end_query = command_buffer->query_end_id;

    item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
    tracy::MemWrite(&item->gpuTime.gpuTime, end_timestamp);
    tracy::MemWrite(&item->gpuTime.queryId, (uint16_t)(end_query));
    tracy::MemWrite(&item->gpuTime.context, tracing_context->id);
    tracy::Profiler::QueueSerialFinish();

    tracing_context->query_tail = (command_buffer->query_end_id + 1) % tracing_context->query_capacity;
    tracing_context->outstanding_query_count-=2;
  }
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
