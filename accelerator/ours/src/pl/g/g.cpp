#include "g.h"

void g(int& numEvents,
       layer_24_beta_t &layer_24_beta_threshold,
       layer_24_distance_t &layer_24_distance_threshold,
       hls::stream<std::array<layer_0_input_t, 1>> energy_stream[PAR],
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_17_plio[2*PAR],
       hls::stream<ap_uint<N>> &layer_24_stream,
       hls::stream<std::array<layer_25_output_t, LAYER_25_OUTPUT_WIDTH>> layer_25_stream[PAR]) {
    #pragma HLS INTERFACE mode=s_axilite port=numEvents
    #pragma HLS STABLE variable=numEvents
    #pragma HLS INTERFACE mode=s_axilite port=layer_24_beta_threshold
    #pragma HLS STABLE variable=layer_24_beta_threshold
    #pragma HLS INTERFACE mode=s_axilite port=layer_24_distance_threshold
    #pragma HLS STABLE variable=layer_24_distance_threshold
    #pragma HLS INTERFACE mode=axis port=layer_17_plio
    #pragma HLS INTERFACE mode=axis port=layer_24_stream
    #pragma HLS INTERFACE mode=axis port=layer_25_stream

    for(int e = 0; e < numEvents; e++){
        #pragma HLS DATAFLOW

        printf("Event %d/%d\n", e+1, numEvents);

        hls::stream<std::array<layer_18_input_t,LAYER_18_INPUT_WIDTH>> tile_17_stream[2*PAR];
        unpack<layer_18_input_t, 128, 8,LAYER_18_INPUT_WIDTH,PAR*2,II/2>(layer_17_plio, tile_17_stream);

        std::array<layer_18_input_t, LAYER_18_INPUT_WIDTH> layer_17_buffer[PAR*2][N/PAR/2];
        #pragma HLS ARRAY_PARTITION variable=layer_17_buffer complete dim=1
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            detile_TRxTC_to_NxM<layer_18_input_t,
                                N/PAR/2,
                                LAYER_18_INPUT_WIDTH,
                                4,
                                8,
                                16,
                                false>(tile_17_stream[p], layer_17_buffer[p]);
        }

        hls::stream<std::array<layer_18_input_t,LAYER_18_INPUT_WIDTH>> retile_17_stream[2*PAR];
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            load_from_buffer<layer_18_input_t,N/PAR/2,LAYER_18_INPUT_WIDTH,II/2>(layer_17_buffer[p], retile_17_stream[p]);
        }

        hls::stream<std::array<layer_18_input_t,LAYER_18_INPUT_WIDTH>> transition_17_stream[PAR];
        rate_transition<std::array<layer_18_input_t,LAYER_18_INPUT_WIDTH>,2*PAR,PAR,II,1>(retile_17_stream,transition_17_stream);
      
        hls::stream<array<layer_18_input_t,LAYER_18_INPUT_WIDTH>,2*II> multicast_stream[5][PAR];
        multicast<array<layer_18_input_t,LAYER_18_INPUT_WIDTH>,PAR,5,II>(transition_17_stream,multicast_stream);

        hls::stream<array<layer_18_output_t,LAYER_18_OUTPUT_WIDTH>> layer_18_stream[PAR];
        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            dense_relu<layer_18_input_t,
                       layer_18_output_t,
                       layer_18_weights_t,
                       layer_18_biases_t,
                       layer_18_accum_t,
                       LAYER_18_INPUT_WIDTH,
                       LAYER_18_OUTPUT_WIDTH,
                       II>(multicast_stream[0][p],
                           layer_18_stream[p],
                           layer_18_weights,
                           layer_18_biases);
        }

        hls::stream<array<layer_25_output_t,1>,12> layer_23_stream[PAR];
        multiply_and_scale<
            layer_0_input_t,
            layer_18_output_t,
            layer_25_output_t,
            8, //Energy Scale
            PAR,
            II,
            1>(energy_stream,
               layer_18_stream,
               layer_23_stream);
        
        hls::stream<array<layer_20_output_t,LAYER_20_OUTPUT_WIDTH>> layer_20_stream[PAR]; //tpos
        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            dense<layer_20_input_t,
                  layer_20_output_t,
                  layer_20_weights_t,
                  layer_20_biases_t,
                  layer_20_accum_t,
                  LAYER_20_INPUT_WIDTH,
                  LAYER_20_OUTPUT_WIDTH,
                  II>(multicast_stream[1][p],
                      layer_20_stream[p],
                      layer_20_weights,
                      layer_20_biases);
        }

        hls::stream<array<layer_21_output_t,LAYER_21_OUTPUT_WIDTH>,12> layer_21_stream[PAR]; //ccords
        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            dense<layer_21_input_t,
                  layer_21_output_t,
                  layer_21_weights_t,
                  layer_21_biases_t,
                  layer_21_accum_t,
                  LAYER_21_INPUT_WIDTH,
                  LAYER_21_OUTPUT_WIDTH,
                  II>(multicast_stream[2][p],
                      layer_21_stream[p],
                      layer_21_weights,
                      layer_21_biases);
        }

        hls::stream<array<layer_22_output_t,LAYER_22_OUTPUT_WIDTH>> layer_22_stream[PAR]; //beta
        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            dense<layer_22_input_t,
                  layer_22_output_t,
                  layer_22_weights_t,
                  layer_22_biases_t,
                  layer_22_accum_t,
                  LAYER_22_INPUT_WIDTH,
                  LAYER_22_OUTPUT_WIDTH,
                  II>(multicast_stream[3][p],
                      layer_22_stream[p],
                      layer_22_weights,
                      layer_22_biases);
        }

        hls::stream<array<layer_19_output_t,LAYER_19_OUTPUT_WIDTH>> layer_19_stream[PAR];
        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            dense<layer_19_input_t,
                  layer_19_output_t,
                  layer_19_weights_t,
                  layer_19_biases_t,
                  layer_19_accum_t,
                  LAYER_19_INPUT_WIDTH,
                  LAYER_19_OUTPUT_WIDTH,
                  II>(multicast_stream[4][p],
                      layer_19_stream[p],
                      layer_19_weights,
                      layer_19_biases);
        }

        condensation_point_selection<layer_24_input_t,
                                     layer_24_distance_t,
                                     layer_24_beta_t,
                                     ap_uint<ceillog2(N)>,
                                     1,
                                     LAYER_24_INPUT_WIDTH,
                                     N,
                                     PAR,
                                     II>(layer_21_stream, //ccords
                                         layer_22_stream, //beta
                                         layer_24_stream, //cps
                                         layer_24_beta_threshold,
                                         layer_24_distance_threshold);

        for (int p = 0; p < PAR; p++) {
            #pragma HLS unroll
            concat<layer_25_input_0_t,
                   layer_25_input_1_t,
                   layer_25_input_2_t,
                   layer_25_output_t,
                   LAYER_25_INPUT_0_WIDTH,
                   LAYER_25_INPUT_1_WIDTH,
                   LAYER_25_INPUT_2_WIDTH,
                   II>(layer_23_stream[p], //energy
                       layer_20_stream[p], //tpos
                       layer_19_stream[p], //signal
                       layer_25_stream[p]);
        }
    }

}