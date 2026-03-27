#include "store.h"

void store(int& numEvents,
           model_output_burst_t* featureList,
           int* numList,
           hls::stream<ap_uint<N>> &layer_24_stream,
           hls::stream<model_output_t> layer_25_stream[PAR]) {
    #pragma HLS interface mode=axis port=layer_24_stream    
    #pragma HLS interface mode=axis port=layer_25_stream    
    #pragma HLS INTERFACE mode=m_axi port=featureList offset=slave bundle=gmem1 depth=64*II latency=64
    #pragma HLS INTERFACE mode=m_axi port=numList offset=slave bundle=gmem1 depth=MAX_EVENTS latency=64

    #pragma HLS stable variable=numEvents
    #pragma HLS interface mode=s_axilite port=numEvents

    assert(numEvents > 0);

    static int nums[MAX_EVENTS];
    static ap_uint<N> cps;

    for(int e = 0; e < MAX_EVENTS;e++) {
        #pragma HLS pipeline II=1
            int n = numList[e];
            nums[e] = numList[e];       
    }


    for(int i = 0; i < numEvents*II;i++) {
        #pragma HLS pipeline II=1
        int n = nums[i/II];
        assert(n > 0);
        model_output_burst_t burst;

        if(i%II == 0) {
            layer_24_stream >> cps;
        }

        for(int p = 0; p < PAR; p++) {
            model_output_t layer_25_beat;
            layer_25_stream[p] >> layer_25_beat;

            for(int j = 0; j < MODEL_OUTPUT_WIDTH; j++) {
                const int W = model_output_feature_t::width;
                burst[8*p+j](W-1,0)   = layer_25_beat[j](W-1,0);
            }
            burst[8*p + 7](1,0)  = cps[(i%II)*PAR + p];
        }

        featureList[i] = burst;
    }
}
