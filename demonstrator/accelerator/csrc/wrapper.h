/**
 * wrapper.h
 *
 * Pure-C interface for the AIE inference pipeline.
 *
 * The opaque handle (accelerator_t) is simply a uintptr_t that holds the
 * address of a heap-allocated Context struct.  Python (via ctypes) sees
 * only this integer, keeping C++ internals completely hidden.
 *
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t accelerator_t;

accelerator_t accelerator_create(const char* xclbin_path,
                        int         num_events,
                        int         verbose,
                        int16_t     beta_threshold,
                        int16_t     distance_threshold);

int accelerator_setup(accelerator_t handle);

int accelerator_load(accelerator_t  handle,
                     const int16_t* num_data,
                     const int16_t* cell_data,
                     int            num_events,
                     int16_t        new_beta_threshold,
                     int16_t        new_distance_threshold);

int accelerator_run(accelerator_t handle,
                    int16_t*      out_features,
                    int16_t*      out_cps);

void accelerator_cleanup(accelerator_t handle);

#ifdef __cplusplus
}
#endif
