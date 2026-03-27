#ifndef GLOBAL_H
#define GLOBAL_H

#include <ap_fixed.h>

//Hardware Parameters
const int N = 128;
const int PAR = 2;
const int MAX_EVENTS = 8192;
const int NUM_STREAM_MULTICAST = 4;

//Model Parameters
const int MODEL_INPUT_WIDTH = 5;
const int MODEL_OUTPUT_WIDTH = 5;

typedef ap_fixed<16,4> model_input_feature_t;
typedef ap_fixed<16,5> model_output_feature_t;

typedef std::array<model_input_feature_t,MODEL_INPUT_WIDTH>   model_input_t;
typedef std::array<model_input_feature_t,16>                  model_input_burst_t;
typedef std::array<model_output_feature_t,MODEL_OUTPUT_WIDTH> model_output_t; //energy,x,y,z,signal
typedef std::array<model_output_feature_t,16>                model_output_burst_t;

//Constants automatically derived from above parameters
const int II = N/PAR;

#endif
