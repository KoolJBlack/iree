// Copyright 2022 Google LLC
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

#ifndef IREE_HAL_CUDA_TRACING_H_
#define IREE_HAL_CUDA_TRACING_H_

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/api.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/context_wrapper.h"

// DO-NOT-SUBMIT: remove this defines
#define IREE_TRACING_FEATURES IREE_TRACING_FEATURE_INSTRUMENTATION

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Per-streaming Cuda tracing context.
// No-op if IREE tracing is not enabled.
//
// Thread-compatible: external synchronization is required if using from
// multiple threads (same as with VkQueue itself).
typedef struct iree_hal_cuda_tracing_context_t iree_hal_cuda_tracing_context_t;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Setup global variables for CUPTI callbacks..
//
// Must be invoked before any tracing functions and paired with the
// globals_destroy method for cleanup.
//
void iree_hal_cuda_tracing_globals_initialize(void);

// Allocates a tracing context for the given Cuda queue.
// Each context must only be used with the queue it was created with.
//
// |maintenance_dispatch_queue| may be used to perform query pool maintenance
// tasks and must support graphics or compute commands.
iree_status_t iree_hal_cuda_tracing_context_allocate(
  iree_hal_cuda_context_wrapper_t context_wrapper,
    iree_allocator_t host_allocator,
    iree_hal_cuda_dynamic_symbols_t* syms,
    iree_string_view_t queue_name,
    iree_hal_cuda_tracing_context_t** out_context);

// Finishes tracing capture and cleans up any internal state.
void iree_hal_cuda_tracing_context_free(
    iree_hal_cuda_tracing_context_t* context);

// Collects in-flight timestamp queries from the queue and feeds them to tracy.
// Must be called frequently (every submission, etc) to drain the backlog;
// tracing may start failing if the internal ringbuffer is exceeded.
//
// The provided |command_buffer| may receive additional bookkeeping commands
// that should have no impact on correctness or behavior. If VK_NULL_HANDLE is
// provided then collection will occur synchronously.
void iree_hal_cuda_tracing_sync(
    iree_hal_cuda_tracing_context_t* context);

void iree_hal_cuda_tracing_command_buffer_zone_begin_impl(
    iree_hal_cuda_tracing_context_t* context,
    const iree_tracing_location_t* src_loc);

void iree_hal_cuda_tracing_command_buffer_zone_end_impl(
    iree_hal_cuda_tracing_context_t* context);

// Begins an external zone using the given source information.
// The provided strings will be copied into the tracy buffer.
void iree_hal_cuda_tracing_kernel_zone_begin_external_impl(
    iree_hal_cuda_tracing_context_t* context,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length);

void iree_hal_cuda_tracing_kernel_zone_end_impl(
    iree_hal_cuda_tracing_context_t* context);

// Begins a new zone with the parent function name.
#define IREE_CUDA_TRACE_COMMAND_BUFFER_ZONE_BEGIN(context)                \
  static const iree_tracing_location_t TracyConcat(                       \
      __tracy_source_location, __LINE__) = {NULL, __FUNCTION__, __FILE__, \
                                            (uint32_t)__LINE__, 0};       \
  iree_hal_cuda_tracing_command_buffer_zone_begin_impl(                   \
      context, &TracyConcat(__tracy_source_location, __LINE__));

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_CUDA_TRACE_COMMAND_BUFFER_ZONE_END(context) \
  iree_hal_cuda_tracing_command_buffer_zone_end_impl(context)

// Begins an externally defined zone with a dynamic source location.
// The |file_name|, |function_name|, and optional |name| strings will be copied
// into the trace buffer and do not need to persist.
#define IREE_CUDA_TRACE_KERNEL_ZONE_BEGIN_EXTERNAL(              \
    context, file_name, file_name_length, line, function_name,   \
    function_name_length, name, name_length)                     \
  iree_hal_cuda_tracing_kernel_zone_begin_external_impl(         \
      context, file_name, file_name_length, line, function_name, \
      function_name_length, name, name_length)

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_CUDA_TRACE_KERNEL_ZONE_END(context) \
  iree_hal_cuda_tracing_kernel_zone_end_impl(context)

#else

#define IREE_CUDA_TRACE_COMMAND_BUFFER_ZONE_BEGIN(context)
#define IREE_CUDA_TRACE_COMMAND_BUFFER_ZONE_END(context)
#define IREE_CUDA_TRACE_KERNEL_ZONE_BEGIN_EXTERNAL(            \
    context, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)
#define IREE_CUDA_TRACE_KERNEL_ZONE_END(context)

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_TRACING_H_
