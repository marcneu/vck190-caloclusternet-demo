#include "host.h"

/* Hardware Parameters */

std::string xclbin_path ("kernel.xclbin");
std::string input_path  ("input.h5");
std::string output_path ("output.h5");
static int num_events = 1;
static int verbose    = 0;
static int benchmark  = 0;
static int16_t beta_threshold = 0xEE66;
static int16_t distance_threshold = 0x100;
void parse_command_line(int argc, char *argv[]) {

    int c;

    while (1)
    {
      static struct option long_options[] =
        {
          {"verbose", no_argument, &verbose, 1},
          {"benchmark", no_argument, &benchmark, 1},
          {"xclbin_path", required_argument, 0, 'f'},
          {"num_events", required_argument, 0, 'n'},
          {"input_path", required_argument, 0, 'i'},
          {"output_path", required_argument, 0, 'o'},
          {"beta_threshold", required_argument, 0, 'b'},
          {"distance_threshold", required_argument, 0, 'd'},
          {0, 0, 0, 0}
        };
      /* getopt_long stores the option index here. */
      int option_index = 0;

      c = getopt_long (argc, argv, "f:n:i:o:", long_options, &option_index);

      /* Detect the end of the options. */
      if (c == -1)
        break;

      switch (c)
        {
            case 0:
                if (long_options[option_index].flag != 0)
                break;
  
            case 'f':
                xclbin_path = (char*) optarg;
                break;

            case 'i':
                input_path = (char*) optarg;
                break;

            case 'o':
                output_path = (char*) optarg;
                break;

            case 'n':
                num_events = (atoi(optarg) < MAX_EVENTS) ? atoi(optarg) : MAX_EVENTS;
                break;

            case 'b':
                beta_threshold = atoi(optarg);
                break;

            case 'd':
                distance_threshold = atoi(optarg);
                break;

            default:
                abort();
        }
    }
}

int run() {
    HighFive::File file(input_path, HighFive::File::ReadOnly);
    auto dataset = file.getDataSet("num");
    auto num_data = dataset.read<std::vector<int16_t>>();

    int available_events = num_data.size();
    if(benchmark) {
        std::cout << "You requested to benchmark. Input data is randomized." << std::endl;
    } else if(num_events > available_events) {
        std::cout << "You requested to test " << num_events << " but there are only " << available_events << " available inside the provided dataset. Truncating." << std::endl;
        num_events = available_events;
    }

    auto device = xrt::device(0);

    auto device_name = device.get_info<xrt::info::device::name>();

    std::cout << "[USR] " << "INFO: " << "found device " << device_name << std::endl;

    auto uuid = device.load_xclbin(xclbin_path);
    std::cout << "[USR] " << "INFO: " << "succesfully loaded xclbin " << xclbin_path << std::endl;

    auto loadkrnl  = xrt::kernel(device, uuid, "load");
    std::cout << "[USR] " << "INFO: " << "found kernel " << "load" << " on " << xclbin_path << std::endl;
    auto ccnkrnl   = xrt::kernel(device, uuid, "caloclusternet");
    std::cout << "[USR] " << "INFO: " << "found kernel " << "caloclusternet" << " on " << xclbin_path << std::endl;
    auto storekrnl = xrt::kernel(device, uuid, "store");
    std::cout << "[USR] " << "INFO: " << "found kernel " << "store" << " on " << xclbin_path << std::endl;    

    std::cout << "[USR] " << "INFO: " << "allocate Buffers in Global Memory\n";
    auto inputNumList = xrt::bo(device, sizeof(int) * MAX_EVENTS, loadkrnl.group_id(2));
    auto inputFeatureList = xrt::bo(device, sizeof(in_harness_t) * II * num_events, loadkrnl.group_id(1));
    auto outputNumList = xrt::bo(device, sizeof(int) * MAX_EVENTS, storekrnl.group_id(2));
    auto outputFeatureList = xrt::bo(device, sizeof(out_harness_t) * II * num_events, storekrnl.group_id(1));

    std::cout << "[USR] " << "INFO: " << "load number of events\n";
    auto inputNumListMap = inputNumList.map<int*>();
    if(benchmark) {
        for(int e = 0; e < num_events; e++) {        
            inputNumListMap[e] = N;
        }
    } else {
        for(int e = 0; e < num_events; e++) {        
            if(verbose) printf("Event %d consists of %d points.\n",e,num_data[e]);   
            inputNumListMap[e] = num_data[e];
        }
    }

    auto outputNumListMap = outputNumList.map<int*>();
    if(benchmark) {
        for(int e = 0; e < num_events; e++) {     
            outputNumListMap[e] = N;
        }
    } else {
        for(int e = 0; e < num_events; e++) {     
            if(verbose) printf("Event %d consists of %d points.\n",e,num_data[e]);   
            outputNumListMap[e] = num_data[e];
        }
    }

    std::cout << "[USR] " << "INFO: " << "load input features\n";
    auto inputFeatureListMap = inputFeatureList.map<in_harness_t*>();
    if(!benchmark) {
        HighFive::File file(input_path, HighFive::File::ReadOnly);
        auto dataset = file.getDataSet("cell");
        auto data = dataset.read<std::vector<std::vector<std::vector<int16_t>>>>();
        for(int e = 0; e < num_events*II; e++) {
            if(e % II == 0) if(verbose) printf("Event %d \n",e / II);   
            in_harness_t burst;
            for(int p = 0; p < PAR; p++) {
                for(int f = 0; f < MODEL_INPUT_WIDTH; f++) {
                    if(verbose) printf("%d, ",data[e / II][(e % II)*PAR + p][f]);   
                    burst[p*MODEL_INPUT_WIDTH+f] = data[e / II][(e % II)*PAR + p][f];
                }
            if(verbose) printf("\n");
            }
            inputFeatureListMap[e] = burst;
        }
    }

    std::cout << "[USR] " << "INFO: " << "input transfer from hard drive to system memory complete" << std::endl;;

    auto outputFeatureListMap = outputFeatureList.map<out_harness_t*>();
    for(int i = 0; i < num_events*II; i++) {
        out_harness_t new_node = {0};
        outputFeatureListMap[i] = new_node;
    }

    // Synchronize buffer content with device side
    std::cout << "[USR] " << "INFO: " << "synchronize input buffer data to device global memory\n";
    inputNumList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    inputFeatureList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    outputNumList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    outputFeatureList.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "[USR] " << "INFO: " << "set load arguments\n";
    auto loadrun = xrt::run(loadkrnl);
    loadrun.set_arg(0,num_events);
    loadrun.set_arg(1,inputFeatureList);
    loadrun.set_arg(2,inputNumList);

    std::cout << "[USR] " << "INFO: " << "set caloclusternet arguments\n";
    auto ccnrun = xrt::run(ccnkrnl);
    ccnrun.set_arg(0,num_events);
    ccnrun.set_arg(1,beta_threshold);
    ccnrun.set_arg(2,distance_threshold);

    std::cout << "[USR] " << "INFO: " << "set store arguments\n";
    auto storerun = xrt::run(storekrnl);
    storerun.set_arg(0,num_events);
    storerun.set_arg(1,outputFeatureList);
    storerun.set_arg(2,outputNumList);

    std::cout << "[USR] " << "INFO: " <<"start all kernels" <<  "\n";
    loadrun.start();
    storerun.start();
    ccnrun.start(xrt::autostart{static_cast<unsigned int>(num_events)});

    std::cout << "[USR] " << "INFO: " << "wait on completion for load...\n";
    loadrun.wait();
    std::cout << "[USR] " << "INFO: " << "wait on completion for store...\n";
    storerun.wait();
    //std::cout << "[USR] " << "INFO: " << "wait on completion for caloclusternet...\n";
    //ccnrun.stop();
    //ccnrun.wait();


    std::cout << "[USR] " << "INFO: " << "transfer from device memory to system memory" << std::endl;

    outputFeatureList.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<std::vector<std::vector<int16_t>>> output_features(num_events, std::vector<std::vector<int16_t>>(N, std::vector<int16_t>(MODEL_OUTPUT_WIDTH, 0)));
    std::vector<std::vector<std::vector<int16_t>>> output_cps(num_events, std::vector<std::vector<int16_t>>(N, std::vector<int16_t>(1, 0)));

    for(int e = 0; e < num_events*II; e++) {
        out_harness_t burst = outputFeatureListMap[e];
        for(int p = 0; p < PAR; p++) {
            for(int f = 0; f < MODEL_OUTPUT_WIDTH; f++) {
                output_features[e / II][(e % II)*PAR + p][f] = burst[p*8+f];
            }
            output_cps[e / II][(e % II)*PAR + p][0] = burst[p*8+7];
        }
    }

    HighFive::File output_file(output_path, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
    
    std::vector<size_t> feature_dims{static_cast<unsigned long>(num_events), N, MODEL_OUTPUT_WIDTH};
    HighFive::DataSet feature_dataset = output_file.createDataSet<int16_t>("int_features", HighFive::DataSpace(feature_dims));
    feature_dataset.write(output_features);;
    
    std::vector<size_t> cps_dims{static_cast<unsigned long>(num_events), N, 1};
    HighFive::DataSet cps_dataset = output_file.createDataSet<int16_t>("cps", HighFive::DataSpace(cps_dims));
    cps_dataset.write(output_cps);;

    std::cout << "[USR] " << "INFO: " << "SUCCESS\n";
    return 0;
}

int main(int argc, char* argv[]) {

    parse_command_line(argc,argv);

    int result = 0;
    result = run();

    return result;
}
