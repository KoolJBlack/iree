#if !defined(__CUPTI_CALLBACKS_H__)
#define __CUPTI_CALLBACKS_H__

/**
 * \defgroup CUPTI_CALLBACK_API CUPTI Callback API
 * Functions, types, and enums that implement the CUPTI Callback API.
 * @{
 */

/**
 * \brief Specifies the point in an API call that a callback is issued.
 *
 * Specifies the point in an API call that a callback is issued. This
 * value is communicated to the callback function via \ref
 * CUpti_CallbackData::callbackSite.
 */
typedef enum {
  /**
   * The callback is at the entry of the API call.
   */
  CUPTI_API_ENTER                 = 0,
  /**
   * The callback is at the exit of the API call.
   */
  CUPTI_API_EXIT                  = 1,
  CUPTI_API_CBSITE_FORCE_INT     = 0x7fffffff
} CUpti_ApiCallbackSite;

/**
 * \brief Callback domains.
 *
 * Callback domains. Each domain represents callback points for a
 * group of related API functions or CUDA driver activity.
 */
typedef enum {
  /**
   * Invalid domain.
   */
  CUPTI_CB_DOMAIN_INVALID           = 0,
  /**
   * Domain containing callback points for all driver API functions.
   */
  CUPTI_CB_DOMAIN_DRIVER_API        = 1,
  /**
   * Domain containing callback points for all runtime API
   * functions.
   */
  CUPTI_CB_DOMAIN_RUNTIME_API       = 2,
  /**
   * Domain containing callback points for CUDA resource tracking.
   */
  CUPTI_CB_DOMAIN_RESOURCE          = 3,
  /**
   * Domain containing callback points for CUDA synchronization.
   */
  CUPTI_CB_DOMAIN_SYNCHRONIZE       = 4,
  /**
   * Domain containing callback points for NVTX API functions.
   */
  CUPTI_CB_DOMAIN_NVTX              = 5,
  CUPTI_CB_DOMAIN_SIZE              = 6,
  CUPTI_CB_DOMAIN_FORCE_INT         = 0x7fffffff
} CUpti_CallbackDomain;



/**
 * \brief An ID for a driver API, runtime API, resource or
 * synchronization callback.
 *
 * An ID for a driver API, runtime API, resource or synchronization
 * callback. Within a driver API callback this should be interpreted
 * as a CUpti_driver_api_trace_cbid value (these values are defined in
 * cupti_driver_cbid.h). Within a runtime API callback this should be
 * interpreted as a CUpti_runtime_api_trace_cbid value (these values
 * are defined in cupti_runtime_cbid.h). Within a resource API
 * callback this should be interpreted as a \ref
 * CUpti_CallbackIdResource value. Within a synchronize API callback
 * this should be interpreted as a \ref CUpti_CallbackIdSync value.
 */
typedef uint32_t CUpti_CallbackId;

/**
 * \brief Data passed into a runtime or driver API callback function.
 *
 * Data passed into a runtime or driver API callback function as the
 * \p cbdata argument to \ref CUpti_CallbackFunc. The \p cbdata will
 * be this type for \p domain equal to CUPTI_CB_DOMAIN_DRIVER_API or
 * CUPTI_CB_DOMAIN_RUNTIME_API. The callback data is valid only within
 * the invocation of the callback function that is passed the data. If
 * you need to retain some data for use outside of the callback, you
 * must make a copy of that data. For example, if you make a shallow
 * copy of CUpti_CallbackData within a callback, you cannot
 * dereference \p functionParams outside of that callback to access
 * the function parameters. \p functionName is an exception: the
 * string pointed to by \p functionName is a global constant and so
 * may be accessed outside of the callback.
 */
typedef struct {
  /**
   * Point in the runtime or driver function from where the callback
   * was issued.
   */
  CUpti_ApiCallbackSite callbackSite;

  /**
   * Name of the runtime or driver API function which issued the
   * callback. This string is a global constant and so may be
   * accessed outside of the callback.
   */
  const char *functionName;

  /**
   * Pointer to the arguments passed to the runtime or driver API
   * call. See generated_cuda_runtime_api_meta.h and
   * generated_cuda_meta.h for structure definitions for the
   * parameters for each runtime and driver API function.
   */
  const void *functionParams;

  /**
   * Pointer to the return value of the runtime or driver API
   * call. This field is only valid within the exit::CUPTI_API_EXIT
   * callback. For a runtime API \p functionReturnValue points to a
   * \p cudaError_t. For a driver API \p functionReturnValue points
   * to a \p CUresult.
   */
  void *functionReturnValue;

  /**
   * Name of the symbol operated on by the runtime or driver API
   * function which issued the callback. This entry is valid only for
   * driver and runtime launch callbacks, where it returns the name of
   * the kernel.
   */
  const char *symbolName;

  /**
   * Driver context current to the thread, or null if no context is
   * current. This value can change from the entry to exit callback
   * of a runtime API function if the runtime initializes a context.
   */
  CUcontext context;

  /**
   * Unique ID for the CUDA context associated with the thread. The
   * UIDs are assigned sequentially as contexts are created and are
   * unique within a process.
   */
  uint32_t contextUid;

  /**
   * Pointer to data shared between the entry and exit callbacks of
   * a given runtime or drive API function invocation. This field
   * can be used to pass 64-bit values from the entry callback to
   * the corresponding exit callback.
   */
  uint64_t *correlationData;

  /**
   * The activity record correlation ID for this callback. For a
   * driver domain callback (i.e. \p domain
   * CUPTI_CB_DOMAIN_DRIVER_API) this ID will equal the correlation ID
   * in the CUpti_ActivityAPI record corresponding to the CUDA driver
   * function call. For a runtime domain callback (i.e. \p domain
   * CUPTI_CB_DOMAIN_RUNTIME_API) this ID will equal the correlation
   * ID in the CUpti_ActivityAPI record corresponding to the CUDA
   * runtime function call. Within the callback, this ID can be
   * recorded to correlate user data with the activity record. This
   * field is new in 4.1.
   */
  uint32_t correlationId;

} CUpti_CallbackData;

/**
 * \brief Function type for a callback.
 *
 * Function type for a callback. The type of the data passed to the
 * callback in \p cbdata depends on the \p domain. If \p domain is
 * CUPTI_CB_DOMAIN_DRIVER_API or CUPTI_CB_DOMAIN_RUNTIME_API the type
 * of \p cbdata will be CUpti_CallbackData. If \p domain is
 * CUPTI_CB_DOMAIN_RESOURCE the type of \p cbdata will be
 * CUpti_ResourceData. If \p domain is CUPTI_CB_DOMAIN_SYNCHRONIZE the
 * type of \p cbdata will be CUpti_SynchronizeData. If \p domain is
 * CUPTI_CB_DOMAIN_NVTX the type of \p cbdata will be CUpti_NvtxData.
 *
 * \param userdata User data supplied at subscription of the callback
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param cbdata Data passed to the callback.
 */
typedef void (CUPTIAPI *CUpti_CallbackFunc)(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void *cbdata);

/**
 * \brief A callback subscriber.
 */
typedef struct CUpti_Subscriber_st *CUpti_SubscriberHandle;

/**
 * \brief Pointer to an array of callback domains.
 */
typedef CUpti_CallbackDomain *CUpti_DomainTable;

#endif  // file guard
