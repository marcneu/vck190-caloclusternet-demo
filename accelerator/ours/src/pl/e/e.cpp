#include "e.h"

void e(int& numEvents,
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_1011_plio[2*PAR],
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_12_plio[2*PAR],
       hls::stream<int> &num_stream) {
    #pragma HLS INTERFACE mode=s_axilite port=numEvents
    #pragma HLS STABLE variable=numEvents
    #pragma HLS INTERFACE mode=axis port=layer_1011_plio
    #pragma HLS INTERFACE mode=axis port=layer_12_plio
    #pragma HLS INTERFACE mode=axis port=num_stream

    for(int e = 0; e < numEvents; e++){
        #pragma HLS DATAFLOW

        printf("Event %d/%d\n", e+1, numEvents);

        hls::stream<std::array<layer_12_coordinate_t,16>> layer_1011_stream[2*PAR];
        unpack<layer_12_coordinate_t, 128, 8,16,PAR*2,II/2>(layer_1011_plio, layer_1011_stream);

        std::array<layer_12_coordinate_t, 16> layer_1011_buffer[PAR*2][N/PAR/2];
        #pragma HLS ARRAY_PARTITION variable=layer_1011_buffer complete dim=1
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            detile_TRxTC_to_NxM<layer_12_coordinate_t,
                                N/PAR/2,
                                16,
                                4,
                                8,
                                16,
                                false>(layer_1011_stream[p], layer_1011_buffer[p]);
        }

        hls::stream<std::array<layer_12_coordinate_t,16>> retile_1011_stream[2*PAR];
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            load_from_buffer<layer_12_coordinate_t,N/PAR/2,16,II/2>(layer_1011_buffer[p], retile_1011_stream[p]);
        }

        hls::stream<std::array<layer_12_coordinate_t,16>> transition_1011_stream[PAR];
        rate_transition<std::array<layer_12_coordinate_t,16>,2*PAR,PAR,II,1>(retile_1011_stream,transition_1011_stream);
  
        hls::stream<std::array<layer_12_coordinate_t,LAYER_12_COORDINATE_WIDTH>> layer_10_stream[PAR];
        hls::stream<std::array<layer_12_feature_t,LAYER_12_FEATURE_WIDTH>> layer_11_stream[PAR];
        
        split_strip<layer_12_coordinate_t,
                    layer_12_coordinate_t,
                    layer_12_feature_t,
                    16,
                    LAYER_12_COORDINATE_WIDTH,
                    LAYER_12_FEATURE_WIDTH,
                    PAR,
                    II>(transition_1011_stream,layer_10_stream,layer_11_stream);


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
                        num_stream);

        hls::stream<std::array<layer_12_output_t,LAYER_12_OUTPUT_WIDTH>> transition_12_stream[2*PAR];
        rate_transition<std::array<layer_12_output_t,LAYER_12_OUTPUT_WIDTH>,PAR,2*PAR,II,0>(layer_12_stream,transition_12_stream);

        std::array<layer_12_output_t, LAYER_12_OUTPUT_WIDTH> layer_12_buffer[PAR*2][N/PAR/2];
        #pragma HLS ARRAY_PARTITION variable=layer_12_buffer complete dim=1
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            store_to_buffer<layer_12_output_t,
                            N/PAR/2,
                            LAYER_12_OUTPUT_WIDTH,
                            II/2>(transition_12_stream[p], layer_12_buffer[p]);
        }

        hls::stream<std::array<layer_12_output_t,16>> tile_12_stream[2*PAR];
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            tile_NxM_to_TRxTC<layer_12_output_t,
                              N/PAR/2,
                              LAYER_12_OUTPUT_WIDTH,
                              4,
                              8,
                              16,
                              false>(layer_12_buffer[p], tile_12_stream[p]);
        }

        pack<layer_12_output_t, 128, 8, 16, PAR*2, II/2>(tile_12_stream, layer_12_plio);
    }
}