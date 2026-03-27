#ifndef LOAD_H
#define LOAD_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <array>

#include "global.h"

void load(int& numEvents,
          model_input_burst_t* inFeatureList,
          int* inputNumList,
          hls::stream<model_input_t> outputFeatureStream[PAR],
          hls::stream<int> num_stream[2]);

#endif