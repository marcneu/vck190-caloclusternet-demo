#ifndef A_H
#define A_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <array>

#include "global.h"
#include "weights.h"

#include "dense_relu.h"
#include "store_to_buffer.h"
#include "retile.h"
#include "pack.h"
#include "rate_transition.h"

template<typename T, int F, int IN_PAR, int OUT_PAR, int II,int OFFSET>
void multicast_and_forward(hls::stream<std::array<T,F>> in[IN_PAR], 
                           hls::stream<std::array<T,F>> out[OUT_PAR][IN_PAR], 
                           hls::stream<std::array<T,1>> forward[IN_PAR]) {
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

void a(int& numEvents,
       hls::stream<std::array<layer_0_input_t, LAYER_0_INPUT_WIDTH>> input_stream[PAR],
       hls::stream<std::array<layer_0_input_t, 1>> energy_stream[PAR],
       hls::stream<ap_axiu<64, 0, 0, 0>> layer_0_plio[2*PAR],
       hls::stream<ap_axiu<64, 0, 0, 0>> layer_1_plio[2*PAR]);

#endif