/**
Description: Top-level test file for standalone caloclusternet test suite.
Author: Marc Neu
Date: 24.01.2025
*/
#include <iostream>
#include "getopt.h"
#include "global.h"
#include <highfive/highfive.hpp>

#include "pl/load/load.h"
#include "pl/caloclusternet/caloclusternet.h"
#include "pl/store/store.h"

using std::array;
using std::vector;

char* input_path = (char *) "build/input.h5";
char* output_path = (char *) "build/output.h5";
int num_events = 100;

void parse_command_line(int argc, char *argv[]) {

    int c;

    while (1)
    {
      static struct option long_options[] =
        {
          /* These options set a flag. */
          {"input_path", required_argument, 0, 'i'},
		  {"output_path", required_argument, 0, 'o'},
          {"num_events", required_argument, 0, 'n'},
          {0, 0, 0, 0}
        };
      /* getopt_long stores the option index here. */
      int option_index = 0;

      c = getopt_long (argc, argv, "i:c:f:o:n:", long_options, &option_index);

      /* Detect the end of the options. */
      if (c == -1)
        break;

      switch (c)
        {
            case 'i':
                input_path = (char*) optarg;
                break;

            case 'o':
                output_path = (char*) optarg;
                break;

            case 'n':
                num_events = (atoi(optarg) < MAX_EVENTS) ? atoi(optarg) : MAX_EVENTS;
                break;

            default:
                abort();
        }
    }
}

int main(int argc, char *argv[])
{

    parse_command_line(argc, argv);

    HighFive::File file(input_path, HighFive::File::ReadOnly);

    printf("Open Input Data\n");

    auto num_nodes_dataset = file.getDataSet("num");
    auto num_nodes_data = num_nodes_dataset.read<std::vector<int16_t>>();

    int available_events = num_nodes_data.size();
    if(num_events > available_events) {
        std::cout << "You requested to test " << num_events << " but there are only " << available_events << " available inside the provided dataset. Truncating." << std::endl;
        num_events = available_events;
    }

    int* input_num_list = (int*) calloc(num_events, sizeof(int));
    for(int e = 0; e < num_events; e++) {
        input_num_list[e] = num_nodes_data[e];
    }

    int* output_num_list = (int*) calloc(num_events, sizeof(int));
    for(int e = 0; e < num_events; e++) {
        output_num_list[e] = num_nodes_data[e];
    }

    auto input_dataset = file.getDataSet("cell");
    auto input_data = input_dataset.read<std::vector<std::vector<std::vector<int16_t>>>>();

    model_input_burst_t* input_feature_list = (model_input_burst_t*) calloc(num_events*II, sizeof(model_input_burst_t));
    for(int e = 0; e < num_events*II; e++) {
        model_input_burst_t burst;
        for(int p = 0; p < PAR; p++) {
            for(int f = 0; f < MODEL_INPUT_WIDTH; f++) {
                burst[p*8+f] = input_data[e / II][(e % II)*PAR + p][f];
            }
        }
        input_feature_list[e] = burst;
    }

    layer_24_beta_t beta_threshold;
    layer_24_distance_t distance_threshold;

    beta_threshold(layer_24_beta_t::width-1,0) = 0xEE66;
    distance_threshold(layer_24_distance_t::width-1,0) = 0x100;

    printf("┌───────────┬───────────────┐\n");
    printf("│ Threshold │ Value         │\n");
    printf("├───────────┼───────────────┤\n");
    printf("│ Beta      │ %-13s │\n", beta_threshold.to_string(10).c_str());
    printf("│ Isolation │ %-13s │\n", distance_threshold.to_string(10).c_str());
    printf("└───────────┴───────────────┘\n");

    printf("Start Execution\n");


    hls::stream<std::array<layer_0_input_t, LAYER_0_INPUT_WIDTH>> input_stream[PAR];
    hls::stream<int> num_stream[2];

    load(num_events,
         input_feature_list,
         input_num_list,
         input_stream,
         num_stream);

    hls::stream<ap_uint<N>> layer_24_stream;
    hls::stream<std::array<layer_25_output_t, LAYER_25_OUTPUT_WIDTH>> layer_25_stream[PAR];

    caloclusternet(num_events,
                   beta_threshold,
                   distance_threshold,
                   input_stream,
                   num_stream,
                   layer_24_stream,
                   layer_25_stream);

    model_output_burst_t* output_feature_list = (model_output_burst_t*) calloc(MAX_EVENTS*II, sizeof(model_output_burst_t));

    store(num_events,
          output_feature_list,
          output_num_list,
          layer_24_stream,
          layer_25_stream);

    vector<vector<vector<int16_t>>> cluster(num_events, vector<vector<int16_t>>(N, vector<int16_t>(MODEL_OUTPUT_WIDTH, 0)));
    vector<vector<vector<int16_t>>> cps(num_events, vector<vector<int16_t>>(N, vector<int16_t>(1, 0)));

    for(int e = 0; e < num_events*II; e++) {
        int n = num_nodes_data[e / II];
        model_output_burst_t burst = output_feature_list[e];
        for(int p = 0; p < PAR; p++) {
            for(int f = 0; f < MODEL_OUTPUT_WIDTH; f++) {
                if((e % II)*PAR + p < n) {
                    cluster[e / II][(e % II)*PAR + p][f] = burst[p*8+f];
                }
            }
            cps[e / II][(e % II)*PAR + p][0] = burst[p*8+7];
        }
    }

    HighFive::File output_file(output_path, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    vector<size_t> output_num_dims{static_cast<unsigned long>(num_events)};
    HighFive::DataSet output_num_dataset = output_file.createDataSet<int16_t>("num", HighFive::DataSpace(output_num_dims));
    output_num_dataset.write(num_nodes_data);

    vector<size_t> feature_dims{static_cast<unsigned long>(num_events), N, MODEL_OUTPUT_WIDTH};
    HighFive::DataSet output_dataset = output_file.createDataSet<int16_t>("features", HighFive::DataSpace(feature_dims));
    output_dataset.write(cluster);

    vector<size_t> cps_dims{static_cast<unsigned long>(num_events), N, 1};
    HighFive::DataSet cps_dataset = output_file.createDataSet<int16_t>("cps", HighFive::DataSpace(cps_dims));
    cps_dataset.write(cps);

    return 0;
}