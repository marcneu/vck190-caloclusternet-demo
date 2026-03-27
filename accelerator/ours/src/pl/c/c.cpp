#include "c.h"

void c(int& numEvents,
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_34_plio[2*PAR],
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_5_plio[2*PAR],
       hls::stream<int> &num_stream) {
    #pragma HLS INTERFACE mode=s_axilite port=numEvents
    #pragma HLS STABLE variable=numEvents
    #pragma HLS INTERFACE mode=axis port=layer_34_plio
    #pragma HLS INTERFACE mode=axis port=layer_5_plio
    #pragma HLS INTERFACE mode=axis port=num_stream

    for(int e = 0; e < numEvents; e++){
        #pragma HLS DATAFLOW

        printf("Event %d/%d\n", e+1, numEvents);

        hls::stream<std::array<layer_5_coordinate_t,16>> layer_34_stream[2*PAR];
        unpack<layer_5_coordinate_t, 128, 8,16,PAR*2,II/2>(layer_34_plio, layer_34_stream);

        std::array<layer_5_coordinate_t, 16> layer_34_buffer[PAR*2][N/PAR/2];
        #pragma HLS ARRAY_PARTITION variable=layer_34_buffer complete dim=1
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            detile_TRxTC_to_NxM<layer_5_coordinate_t,
                                N/PAR/2,
                                16,
                                4,
                                8,
                                16,
                                false>(layer_34_stream[p], layer_34_buffer[p]);
        }

        hls::stream<std::array<layer_5_coordinate_t,16>> retile_34_stream[2*PAR];
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            load_from_buffer<layer_5_coordinate_t,N/PAR/2,16,II/2>(layer_34_buffer[p], retile_34_stream[p]);
        }

        hls::stream<std::array<layer_5_coordinate_t,16>> transition_34_stream[PAR];
        rate_transition<std::array<layer_5_coordinate_t,16>,2*PAR,PAR,II,1>(retile_34_stream,transition_34_stream);
  
        hls::stream<std::array<layer_5_coordinate_t,LAYER_5_COORDINATE_WIDTH>> layer_3_stream[PAR];
        hls::stream<std::array<layer_5_feature_t,LAYER_5_FEATURE_WIDTH>> layer_4_stream[PAR];
        
        split_strip<layer_5_coordinate_t,
                    layer_5_coordinate_t,
                    layer_5_feature_t,
                    16,
                    LAYER_5_COORDINATE_WIDTH,
                    LAYER_5_FEATURE_WIDTH,
                    PAR,
                    II>(transition_34_stream,layer_3_stream,layer_4_stream);

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
                        num_stream);

        hls::stream<std::array<layer_5_output_t,LAYER_5_OUTPUT_WIDTH>> transition_5_stream[2*PAR];
        rate_transition<std::array<layer_5_output_t,LAYER_5_OUTPUT_WIDTH>,PAR,2*PAR,II,0>(layer_5_stream,transition_5_stream);

        std::array<layer_5_output_t, LAYER_5_OUTPUT_WIDTH> layer_5_buffer[PAR*2][N/PAR/2];
        #pragma HLS ARRAY_PARTITION variable=layer_5_buffer complete dim=1
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            store_to_buffer<layer_5_output_t,
                            N/PAR/2,
                            LAYER_5_OUTPUT_WIDTH,
                            II/2>(transition_5_stream[p], layer_5_buffer[p]);
        }

        hls::stream<std::array<layer_5_output_t,16>> tile_5_stream[2*PAR];
        for(int p = 0; p < PAR*2; p++) {
            #pragma HLS unroll
            tile_NxM_to_TRxTC<layer_5_output_t,
                              N/PAR/2,
                              LAYER_5_OUTPUT_WIDTH,
                              4,
                              8,
                              16,
                              false>(layer_5_buffer[p], tile_5_stream[p]);
        }

        pack<layer_5_output_t, 128, 8, 16, PAR*2, II/2>(tile_5_stream, layer_5_plio);

    }

}