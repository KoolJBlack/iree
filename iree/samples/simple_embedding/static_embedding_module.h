#ifndef IREE_SAMPLES_SIMPLE_EMBEDDING_SIMPLE_EMBEDDING_TEST_H_
#define IREE_SAMPLES_SIMPLE_EMBEDDING_SIMPLE_EMBEDDING_TEST_H_

#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

const iree_hal_executable_library_header_t** iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version, void* reserved);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus


#endif  // IREE_SAMPLES_SIMPLE_EMBEDDING_SIMPLE_EMBEDDING_TEST_H_
