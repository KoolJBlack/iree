// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>  // std::chrono::seconds
#include <iostream>
#include <thread>  // std::this_thread::sleep_for

#include "iree/base/api.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cuda {
namespace {

#define CUDE_CHECK_ERRORS(expr)      \
  {                                  \
    CUresult status = expr;          \
    ASSERT_EQ(CUDA_SUCCESS, status); \
  }

#define CUPTI_CHECK_ERRORS(expr)      \
  {                                   \
    CUptiResult status = expr;        \
    ASSERT_EQ(CUPTI_SUCCESS, status); \
  }

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))

iree_hal_cuda_dynamic_symbols_t symbols;

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
  uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    // All records are parsed correctly.
    do {
      status = symbols.cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;  // No more records
      } else {
        ASSERT_EQ(status, CUPTI_SUCCESS);
      }
    } while (1);

    // No records dropped.
    size_t dropped;
    CUPTI_CHECK_ERRORS(
        symbols.cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    ASSERT_EQ(dropped, 0);
  }

  free(buffer);
}

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_status_t status = iree_hal_cuda_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  ASSERT_TRUE(iree_status_is_ok(status));

  // CUPTI timestamp querying
  uint64_t timestamp1 = 0;
  uint64_t timestamp2 = 0;
  std::cout << "The timestamp before: " << timestamp1 << std::endl;

  CUPTI_CHECK_ERRORS(symbols.cuptiGetTimestamp(&timestamp1));
  std::cout << "The timestamp after: " << timestamp1 << std::endl;

  std::this_thread::sleep_for(std::chrono::seconds(1));

  CUPTI_CHECK_ERRORS(symbols.cuptiGetTimestamp(&timestamp2));
  std::cout << "The timestamp after sleep 1 second: " << timestamp2
            << std::endl;

  auto timestamp_detla = timestamp2 - timestamp1;
  std::cout << "The timestamp delta: " << timestamp_detla << std::endl;

  ASSERT_GT(timestamp2, timestamp1);

  // Activity API record collection
  CUPTI_CHECK_ERRORS(
      symbols.cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

  // Activity API callback registration
  CUPTI_CHECK_ERRORS(
      symbols.cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // Initialize Cuda with device context
  int device_count = 0;
  CUDE_CHECK_ERRORS(symbols.cuInit(0));
  CUDE_CHECK_ERRORS(symbols.cuDeviceGetCount(&device_count));
  if (device_count > 0) {
    CUdevice device;
    CUDE_CHECK_ERRORS(symbols.cuDeviceGet(&device, /*ordinal=*/0));
    CUcontext context;
    CUDE_CHECK_ERRORS(symbols.cuCtxCreate(&context, 0, device));

    // Device timestamp querying
    uint64_t deviceTimerstamp = 0;
    CUPTI_CHECK_ERRORS(
        symbols.cuptiDeviceGetTimestamp(context, &deviceTimerstamp));
    ASSERT_GT(deviceTimerstamp, 0);
  }

  // Teardown
  iree_hal_cuda_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace cuda
}  // namespace hal
}  // namespace iree
