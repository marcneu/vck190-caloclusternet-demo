#ifndef CALOCLUSTERNET_H
#define CALOCLUSTERNET_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <array>

#include "global.h"
#include "weights.h"

#include "dense.h"
#include "dense_relu.h"
#include "concat.h"
#include "multiply.h"
#include "multicast.h"
#include "gravnetconv.h"
#include "condensation_point_selection.h"

template<typename T, int F, int IN_PAR, int OUT_PAR, int II,int OFFSET, int DEPTH>
void multicast_and_forward(hls::stream<std::array<T,F>> in[IN_PAR], 
                           hls::stream<std::array<T,F>,DEPTH> out[OUT_PAR][IN_PAR], 
                           hls::stream<std::array<T,1>,DEPTH> forward[IN_PAR]) {
    for(int ii = 0; ii < II; ii++){
        #pragma HLS pipeline II=1 style=flp
        for(int i = 0; i < IN_PAR;i++) {
            std::array<T,F> data;
            in[i] >> data;

            for(int p = 0; p < OUT_PAR; p++) {
                out[p][i] << data;
            }

            std::array<T,1> element;
            element[0] = data[OFFSET];
            forward[i] << element;
        }
    }
}

void caloclusternet(int& numEvents,
                    layer_24_beta_t &layer_24_beta_threshold,
                    layer_24_distance_t &layer_24_distance_threshold,
                    hls::stream<std::array<layer_0_input_t, LAYER_0_INPUT_WIDTH>> input_stream[PAR],
                    hls::stream<int> num_stream[2],
                    hls::stream<ap_uint<N>> &layer_24_stream,
                    hls::stream<model_output_t> layer_25_stream[PAR]);

#endif