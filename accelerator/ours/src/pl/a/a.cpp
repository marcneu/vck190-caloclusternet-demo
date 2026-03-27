#include "a.h"

void a(int& numEvents,
       hls::stream<std::array<layer_0_input_t, LAYER_0_INPUT_WIDTH>> input_stream[PAR],
       hls::stream<std::array<layer_0_input_t, 1>> energy_stream[PAR],
       hls::stream<ap_axiu<64, 0, 0, 0>> layer_0_plio[2*PAR],
       hls::stream<ap_axiu<64, 0, 0, 0>> layer_1_plio[2*PAR]) {
    #pragma HLS INTERFACE mode=s_axilite port=numEvents
    #pragma HLS STABLE variable=numEvents
    #pragma HLS INTERFACE mode=axis port=input_stream
    #pragma HLS INTERFACE mode=axis port=energy_stream
    #pragma HLS INTERFACE mode=axis port=layer_0_plio
    #pragma HLS INTERFACE mode=axis port=layer_1_plio

    for(int e = 0; e < numEvents; e++){
        #pragma HLS DATAFLOW

        hls::stream<std::array<layer_0_input_t, LAYER_0_INPUT_WIDTH>> multicast_stream[2][PAR];
        multicast_and_forward<layer_0_input_t, LAYER_0_INPUT_WIDTH, PAR, 2, II,0>(input_stream, multicast_stream,energy_stream);

        hls::stream<std::array<layer_0_output_t,LAYER_0_OUTPUT_WIDTH>> layer_0_stream[PAR];
        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            dense_relu<layer_0_input_t,
                       layer_0_output_t,
                       layer_0_weights_t,
                       layer_0_biases_t,
                       layer_0_accum_t,
                       LAYER_0_INPUT_WIDTH,
                       LAYER_0_OUTPUT_WIDTH,
                       II>(multicast_stream[0][p], 
                           layer_0_stream[p],
                           layer_0_weights,
                           layer_0_biases);
        }

        hls::stream<std::array<layer_0_output_t,LAYER_0_OUTPUT_WIDTH>> transition_0_stream[2*PAR];
        rate_transition<std::array<layer_0_output_t,LAYER_0_OUTPUT_WIDTH>,PAR,2*PAR,II,0>(layer_0_stream,transition_0_stream);

        pack<layer_0_output_t, 64, 8,LAYER_0_OUTPUT_WIDTH,PAR*2,II/2>(transition_0_stream, layer_0_plio);

        hls::stream<std::array<layer_1_output_t,LAYER_1_OUTPUT_WIDTH>> layer_1_stream[PAR];
        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            dense_relu<layer_1_input_t,
                       layer_1_output_t,
                       layer_1_weights_t,
                       layer_1_biases_t,
                       layer_1_accum_t,
                       LAYER_1_INPUT_WIDTH,
                       LAYER_1_OUTPUT_WIDTH,
                       II>(multicast_stream[1][p], 
                           layer_1_stream[p],
                           layer_1_weights,
                           layer_1_biases);
        }

        hls::stream<std::array<layer_1_output_t,LAYER_1_OUTPUT_WIDTH>> transition_1_stream[2*PAR];
        rate_transition<std::array<layer_1_output_t,LAYER_1_OUTPUT_WIDTH>,PAR,2*PAR,II,0>(layer_1_stream,transition_1_stream);

        pack<layer_1_output_t,64,8,LAYER_1_OUTPUT_WIDTH,2*PAR,II/2>(transition_1_stream, layer_1_plio);
    }
}