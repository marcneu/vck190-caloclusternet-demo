#ifndef G_H
#define G_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <array>

using std::array;

#include "global.h"
#include "weights.h"

#include "dense.h"
#include "dense_relu.h"
#include "condensation_point_selection.h"
#include "concat.h"
#include "multiply.h"
#include "multicast.h"
#include "unpack.h"
#include "retile.h"
#include "load_from_buffer.h"
#include "rate_transition.h"

void g(int& numEvents,
       layer_24_beta_t &layer_24_beta_threshold,
       layer_24_distance_t &layer_24_distance_threshold,
       hls::stream<std::array<layer_0_input_t, 1>> energy_stream[PAR],
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_17_plio[2*PAR],
       hls::stream<ap_uint<N>> &layer_24_stream,
       hls::stream<std::array<layer_25_output_t, LAYER_25_OUTPUT_WIDTH>> layer_25_stream[PAR]);

#endif