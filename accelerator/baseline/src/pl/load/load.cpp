#include"load.h"

void load(int& numEvents,
          model_input_burst_t* inFeatureList,
          int* inputNumList,
          hls::stream<model_input_t> outputFeatureStream[PAR],
          hls::stream<int> num_stream[2])
{
    #pragma HLS stable variable=numEvents
    #pragma HLS interface mode=s_axilite port=numEvents
    #pragma HLS INTERFACE mode=m_axi port=inputNumList offset=slave bundle=gmem0 depth=MAX_EVENTS latency=64
    #pragma HLS INTERFACE mode=m_axi port=inFeatureList offset=slave bundle=gmem0 depth=II*64 latency=64
    #pragma HLS INTERFACE mode=axis port=outputFeatureStream
    #pragma HLS INTERFACE mode=axis port=num_stream

    int nums[MAX_EVENTS];
    assert(numEvents > 0);
    for(int e = 0; e < MAX_EVENTS;e++) {
        #pragma HLS pipeline II=1
        int n = inputNumList[e];
        nums[e] = n;
    }

    for(int i = 0; i<numEvents*II; i++) {
        #pragma HLS pipeline II=1
        int n = nums[i/II]; 
        assert(n > 0);
        
        for(int j = 0; j<2; j++)     
            if(i % II == 0)
                num_stream[j] << n;
   
        model_input_burst_t burst = inFeatureList[i];
        for(int p = 0; p < PAR; p++) {
            model_input_t features;
            for(int f = 0; f < MODEL_INPUT_WIDTH; f++) {
                if((i%II)*PAR+p < n) {
                    features[f] = burst[p*MODEL_INPUT_WIDTH+f];
                } else {
                    features[f] = 0;
                }
            }
            outputFeatureStream[p] << features;  
        }
    }
}