#pragma once

#include <optional> 

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "experimental/xrt_aie.h"
#include "experimental/xrt_graph.h"
#include "experimental/xrt_xclbin.h"
  
/* ── Harness global parameters ──────────────────────────────────────────── */
static const int MAX_EVENTS = 8192;
 
/* Derived from underlying model */
static const int N   = 128;
static const int PAR = 2;
 
static const int MODEL_INPUT_WIDTH  = 5;
static const int MODEL_OUTPUT_WIDTH = 5;
 
typedef int16_t model_input_t;
typedef int16_t model_output_t;
 
typedef std::array<model_input_t,  16> in_harness_t;
typedef std::array<model_output_t, 16> out_harness_t;
 
const int II = N / PAR;
 
/* ── Context ─────────────────────────────────────────────────────────────── */
struct Context {
    std::string xclbin_path;
    int         num_events;
    int         verbose;
    int16_t     beta_threshold;
    int16_t     distance_threshold;
 
    xrt::aie::device device;
    xrt::uuid        uuid;
 
    xrt::kernel loadkrnl;
    xrt::kernel akrnl;
    xrt::kernel ckrnl;
    xrt::kernel ekrnl;
    xrt::kernel gkrnl;
    xrt::kernel storekrnl;
 
    xrt::bo inputNumList;
    xrt::bo inputFeatureList;
    xrt::bo outputNumList;
    xrt::bo outputFeatureList;
 
    int*           inputNumListMap;
    in_harness_t*  inputFeatureListMap;
    int*           outputNumListMap;
    out_harness_t* outputFeatureListMap;
 
    xrt::run loadrun;
    xrt::run arun;
    xrt::run crun;
    xrt::run erun;
    xrt::run grun;
    xrt::run storerun;
 
    std::optional<xrt::graph> aie_graph;
 
    Context(const std::string& xclbin,
            int                nevents,
            int                verb,
            int16_t            beta_thr,
            int16_t            dist_thr)
        : xclbin_path(xclbin)
        , num_events(nevents)
        , verbose(verb)
        , beta_threshold(beta_thr)
        , distance_threshold(dist_thr)
        , device(0)
    {}
};
 
/* ── Lifecycle functions ─────────────────────────────────────────────────── */
int  setup(Context* ctx);
int  load(Context* ctx, const int16_t* num_data, const int16_t* cell_data, int num_events);
int  run(Context* ctx, int16_t* out_features, int16_t* out_cps);
void cleanup(Context* ctx);
