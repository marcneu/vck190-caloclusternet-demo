#include "caloclusternet.h"

void caloclusternet(int& numEvents,
                    layer_24_beta_t &layer_24_beta_threshold,
                    layer_24_distance_t &layer_24_distance_threshold,
                    hls::stream<std::array<layer_0_input_t, LAYER_0_INPUT_WIDTH>> input_stream[PAR],
                    hls::stream<int> num_stream[2],
                    hls::stream<ap_uint<N>> &layer_24_stream,
                    hls::stream<model_output_t> layer_25_stream[PAR]) {
    #pragma HLS INTERFACE mode=s_axilite port=numEvents
    #pragma HLS STABLE variable=numEvents
    #pragma HLS INTERFACE mode=s_axilite port=layer_24_beta_threshold
    #pragma HLS STABLE variable=layer_24_beta_threshold
    #pragma HLS INTERFACE mode=s_axilite port=layer_24_distance_threshold
    #pragma HLS STABLE variable=layer_24_distance_threshold
    #pragma HLS INTERFACE mode=axis port=input_stream
    #pragma HLS INTERFACE mode=axis port=layer_24_stream
    #pragma HLS INTERFACE mode=axis port=layer_25_stream

    #pragma HLS dataflow

    hls::stream<std::array<layer_0_input_t, LAYER_0_INPUT_WIDTH>,16*II> multicast_0_stream[2][PAR];
    hls::stream<std::array<layer_0_input_t,1>,16*II> energy_stream[PAR];
    multicast_and_forward<layer_0_input_t, LAYER_0_INPUT_WIDTH, PAR, 2, II,0,16*II>(input_stream, multicast_0_stream,energy_stream);

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
                    II>(multicast_0_stream[0][p], 
                        layer_0_stream[p],
                        layer_0_weights,
                        layer_0_biases);
    }

    hls::stream<std::array<layer_1_output_t,LAYER_1_OUTPUT_WIDTH>,32*II> layer_1_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense_relu<layer_1_input_t,
                    layer_1_output_t,
                    layer_1_weights_t,
                    layer_1_biases_t,
                    layer_1_accum_t,
                    LAYER_1_INPUT_WIDTH,
                    LAYER_1_OUTPUT_WIDTH,
                    II>(multicast_0_stream[1][p], 
                        layer_1_stream[p],
                        layer_1_weights,
                        layer_1_biases);
    }

    hls::stream<std::array<layer_2_output_t,LAYER_2_OUTPUT_WIDTH>> layer_2_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense_relu<layer_2_input_t,
                    layer_2_output_t,
                    layer_2_weights_t,
                    layer_2_biases_t,
                    layer_2_accum_t,
                    LAYER_2_INPUT_WIDTH,
                    LAYER_2_OUTPUT_WIDTH,
                    II>(layer_0_stream[p], 
                        layer_2_stream[p],
                        layer_2_weights,
                        layer_2_biases);
    }

    hls::stream<array<layer_2_output_t,LAYER_2_OUTPUT_WIDTH>,II*16> multicast_1_stream[3][PAR];
    multicast<array<layer_2_output_t, LAYER_2_OUTPUT_WIDTH>,PAR,3,II>(layer_2_stream,multicast_1_stream);
    
    hls::stream<std::array<layer_3_output_t,LAYER_3_OUTPUT_WIDTH>> layer_3_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_3_input_t,
                layer_3_output_t,
                layer_3_weights_t,
                layer_3_biases_t,
                layer_3_accum_t,
                LAYER_3_INPUT_WIDTH,
                LAYER_3_OUTPUT_WIDTH,
                II>(multicast_1_stream[0][p], 
                    layer_3_stream[p],
                    layer_3_weights,
                    layer_3_biases);
    }

    hls::stream<std::array<layer_4_output_t,LAYER_4_OUTPUT_WIDTH>> layer_4_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_4_input_t,
                layer_4_output_t,
                layer_4_weights_t,
                layer_4_biases_t,
                layer_4_accum_t,
                LAYER_4_INPUT_WIDTH,
                LAYER_4_OUTPUT_WIDTH,
                II>(multicast_1_stream[1][p], 
                    layer_4_stream[p],
                    layer_4_weights,
                    layer_4_biases);
    }

    hls::stream<std::array<layer_5_output_t,LAYER_5_OUTPUT_WIDTH>> layer_5_stream[PAR];

    gravnetconv<layer_5_coordinate_t,
                layer_5_feature_t,
                layer_5_distance_t,
                layer_5_exponential_t,
                layer_5_accum_t,
                layer_5_output_t,
                LAYER_5_COORDINATE_WIDTH,
                LAYER_5_FEATURE_WIDTH,
                LAYER_5_K,
                N,
                PAR,
                II>(layer_3_stream,
                    layer_4_stream,
                    layer_5_stream,
                    num_stream[0]);

    hls::stream<array<layer_6_output_t,LAYER_6_INPUT_0_WIDTH+LAYER_6_INPUT_1_WIDTH>> layer_6_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        concat<layer_6_input_0_t,
                layer_6_input_1_t,
                layer_6_output_t,
                LAYER_6_INPUT_0_WIDTH,
                LAYER_6_INPUT_1_WIDTH,
                II>(multicast_1_stream[2][p],
                    layer_5_stream[p],
                    layer_6_stream[p]);
    }

    hls::stream<std::array<layer_7_output_t,LAYER_7_OUTPUT_WIDTH>> layer_7_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense_relu<layer_7_input_t,
                    layer_7_output_t,
                    layer_7_weights_t,
                    layer_7_biases_t,
                    layer_7_accum_t,
                    LAYER_7_INPUT_WIDTH,
                    LAYER_7_OUTPUT_WIDTH,
                    II>(layer_6_stream[p], 
                        layer_7_stream[p],
                        layer_7_weights,
                        layer_7_biases);
    }

    hls::stream<std::array<layer_8_output_t,LAYER_8_OUTPUT_WIDTH>> layer_8_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense_relu<layer_8_input_t,
                    layer_8_output_t,
                    layer_8_weights_t,
                    layer_8_biases_t,
                    layer_8_accum_t,
                    LAYER_8_INPUT_WIDTH,
                    LAYER_8_OUTPUT_WIDTH,
                    II>(layer_7_stream[p], 
                        layer_8_stream[p],
                        layer_8_weights,
                        layer_8_biases);
    }

    /*Multicast */
    hls::stream<array<layer_8_output_t,LAYER_8_OUTPUT_WIDTH>,II*16> multicast_2_stream[2][PAR];
    multicast<array<layer_8_output_t, LAYER_8_OUTPUT_WIDTH>,PAR,2,II>(layer_8_stream,multicast_2_stream);
    
    hls::stream<std::array<layer_9_output_t,LAYER_9_OUTPUT_WIDTH>> layer_9_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_9_input_t,
                layer_9_output_t,
                layer_9_weights_t,
                layer_9_biases_t,
                layer_9_accum_t,
                LAYER_9_INPUT_WIDTH,
                LAYER_9_OUTPUT_WIDTH,
                II>(multicast_2_stream[0][p], 
                    layer_9_stream[p],
                    layer_9_weights,
                    layer_9_biases);
    }

    /*Multicast */
    hls::stream<array<layer_9_output_t,LAYER_9_OUTPUT_WIDTH>,II*16> multicast_3_stream[3][PAR];
    multicast<array<layer_9_output_t, LAYER_9_OUTPUT_WIDTH>,PAR,3,II>(layer_9_stream,multicast_3_stream);
    
    hls::stream<std::array<layer_10_output_t,LAYER_10_OUTPUT_WIDTH>> layer_10_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_10_input_t,
                layer_10_output_t,
                layer_10_weights_t,
                layer_10_biases_t,
                layer_10_accum_t,
                LAYER_10_INPUT_WIDTH,
                LAYER_10_OUTPUT_WIDTH,
                II>(multicast_3_stream[0][p], 
                    layer_10_stream[p],
                    layer_10_weights,
                    layer_10_biases);
    }

    hls::stream<std::array<layer_11_output_t,LAYER_11_OUTPUT_WIDTH>> layer_11_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_11_input_t,
                layer_11_output_t,
                layer_11_weights_t,
                layer_11_biases_t,
                layer_11_accum_t,
                LAYER_11_INPUT_WIDTH,
                LAYER_11_OUTPUT_WIDTH,
                II>(multicast_3_stream[1][p], 
                    layer_11_stream[p],
                    layer_11_weights,
                    layer_11_biases);
    }

    hls::stream<std::array<layer_12_output_t,LAYER_12_OUTPUT_WIDTH>> layer_12_stream[PAR];

    gravnetconv<layer_12_coordinate_t,
                layer_12_feature_t,
                layer_12_distance_t,
                layer_12_exponential_t,
                layer_12_accum_t,
                layer_12_output_t,
                LAYER_12_COORDINATE_WIDTH,
                LAYER_12_FEATURE_WIDTH,
                LAYER_12_K,
                N,
                PAR,
                II>(layer_10_stream,
                    layer_11_stream,
                    layer_12_stream,
                    num_stream[1]);

    hls::stream<array<layer_13_output_t,LAYER_13_INPUT_0_WIDTH+LAYER_13_INPUT_1_WIDTH>> layer_13_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        concat<layer_13_input_0_t,
                layer_13_input_1_t,
                layer_13_output_t,
                LAYER_13_INPUT_0_WIDTH,
                LAYER_13_INPUT_1_WIDTH,
                II>(multicast_3_stream[2][p],
                    layer_12_stream[p],
                    layer_13_stream[p]);
    }

    hls::stream<std::array<layer_14_output_t,LAYER_14_OUTPUT_WIDTH>> layer_14_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_14_input_t,
                layer_14_output_t,
                layer_14_weights_t,
                layer_14_biases_t,
                layer_14_accum_t,
                LAYER_14_INPUT_WIDTH,
                LAYER_14_OUTPUT_WIDTH,
                II>(layer_13_stream[p], 
                    layer_14_stream[p],
                    layer_14_weights,
                    layer_14_biases);
    }

    hls::stream<std::array<layer_15_output_t,LAYER_15_OUTPUT_WIDTH>> layer_15_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_15_input_t,
                layer_15_output_t,
                layer_15_weights_t,
                layer_15_biases_t,
                layer_15_accum_t,
                LAYER_15_INPUT_WIDTH,
                LAYER_15_OUTPUT_WIDTH,
                II>(layer_14_stream[p], 
                    layer_15_stream[p],
                    layer_15_weights,
                    layer_15_biases);
    }

    hls::stream<array<layer_16_input_0_t,LAYER_16_INPUT_0_WIDTH+LAYER_16_INPUT_1_WIDTH+LAYER_16_INPUT_2_WIDTH>> layer_16_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        concat<layer_16_input_0_t,
                layer_16_input_1_t,
                layer_16_input_2_t,
                layer_16_output_t,
                LAYER_16_INPUT_0_WIDTH,
                LAYER_16_INPUT_1_WIDTH,
                LAYER_16_INPUT_2_WIDTH,
                II>(layer_1_stream[p],
                    multicast_2_stream[1][p],
                    layer_15_stream[p],
                    layer_16_stream[p]);
    }

    hls::stream<std::array<layer_17_output_t,LAYER_17_OUTPUT_WIDTH>> layer_17_stream[PAR];
    for (int p = 0; p < PAR; p++) {
        #pragma HLS unroll
        dense<layer_17_input_t,
                layer_17_output_t,
                layer_17_weights_t,
                layer_17_biases_t,
                layer_17_accum_t,
                LAYER_17_INPUT_WIDTH,
                LAYER_17_OUTPUT_WIDTH,
                II>(layer_16_stream[p], 
                    layer_17_stream[p],
                    layer_17_weights,
                    layer_17_biases);
    }

    hls::stream<array<layer_18_input_t,LAYER_18_INPUT_WIDTH>,2*II> multicast_4_stream[5][PAR];
    multicast<array<layer_18_input_t,LAYER_18_INPUT_WIDTH>,PAR,5,II>(layer_17_stream,multicast_4_stream);

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
                    II>(multicast_4_stream[0][p],
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
                II>(multicast_4_stream[1][p],
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
                II>(multicast_4_stream[2][p],
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
                II>(multicast_4_stream[3][p],
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
                II>(multicast_4_stream[4][p],
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