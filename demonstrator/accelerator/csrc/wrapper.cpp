/**
 * wrapper.cpp
 *
 * Thin C shim that translates the opaque handle (uintptr_t) to a typed
 * Context pointer and delegates to the lifecycle functions in host.cpp.
 */

#include "wrapper.h"
#include "host.h"

#include <stdexcept>
#include <iostream>

static inline Context* to_ctx(accelerator_t h) {
    return reinterpret_cast<Context*>(h);
}

accelerator_t accelerator_create(const char* xclbin_path,
                                 int         num_events,
                                 int         verbose,
                                 int16_t     beta_threshold,
                                 int16_t     distance_threshold) {
    try {
        int clamped = (num_events < MAX_EVENTS) ? num_events : MAX_EVENTS;
        Context* ctx = new Context(
            xclbin_path ? xclbin_path : "",
            clamped,
            verbose,
            beta_threshold,
            distance_threshold);
        return reinterpret_cast<accelerator_t>(ctx);
    } catch (const std::exception& ex) {
        std::cerr << "[USR] ERROR: accelerator_create failed: " << ex.what() << "\n";
        return 0;
    }
}

int accelerator_setup(accelerator_t handle) {
    Context* ctx = to_ctx(handle);
    if (!ctx) return -1;
    try {
        return setup(ctx);
    } catch (const std::exception& ex) {
        std::cerr << "[USR] ERROR: accelerator_setup failed: " << ex.what() << "\n";
        return -1;
    }
}

int accelerator_load(accelerator_t  handle,
                     const int16_t* num_data,
                     const int16_t* cell_data,
                     int            num_events,
                     int16_t        new_beta_threshold,
                     int16_t        new_distance_threshold) {
    Context* ctx = to_ctx(handle);
    if (!ctx) return -1;
    ctx->beta_threshold      = new_beta_threshold;
    ctx->distance_threshold  = new_distance_threshold;
    try {
        return load(ctx, num_data, cell_data, num_events);
    } catch (const std::exception& ex) {
        std::cerr << "[USR] ERROR: accelerator_load failed: " << ex.what() << "\n";
        return -1;
    }
}

int accelerator_run(accelerator_t handle,
                    int16_t*      out_features,
                    int16_t*      out_cps) {
    Context* ctx = to_ctx(handle);
    if (!ctx) return -1;
    try {
        return run(ctx, out_features, out_cps);
    } catch (const std::exception& ex) {
        std::cerr << "[USR] ERROR: accelerator_run failed: " << ex.what() << "\n";
        return -1;
    }
}

void accelerator_cleanup(accelerator_t handle) {
    Context* ctx = to_ctx(handle);
    if (!ctx) return;
    try {
        cleanup(ctx);
    } catch (const std::exception& ex) {
        std::cerr << "[USR] ERROR: accelerator_cleanup failed: " << ex.what() << "\n";
    }
}
