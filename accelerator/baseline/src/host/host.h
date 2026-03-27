#ifndef HOST_H__
#define HOST_H__

#include <cstring>
#include <iostream>
#include "getopt.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_graph.h"

#include <highfive/highfive.hpp>

/* Harness global parameters */
static const int MAX_EVENTS = 8192;
/* Derived from underlying model */
static const int N = 128;
static const int PAR = 1;

static const int MODEL_INPUT_WIDTH = 5;
static const int MODEL_OUTPUT_WIDTH = 5;

typedef int16_t model_input_t;
typedef int16_t model_output_t;

typedef std::array<model_input_t,16> in_harness_t;
typedef std::array<model_output_t,16> out_harness_t;

const int II = N/PAR;

#endif