#ifndef STORE_H
#define STORE_H

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <array>

#include "global.h"

void store(int& numEvents,
           model_output_burst_t* featureList,
           int* numList,
           hls::stream<ap_uint<N>> &layer_24_stream,
           hls::stream<model_output_t> layer_25_stream[PAR]);
#endif