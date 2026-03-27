/**
 * host.cpp
 *
 *
 * All persistent XRT objects live inside a heap-allocated Context struct.
 * The C wrapper API exposes an opaque handle (uintptr_t) so that Python
 * (or any other language) can manage the object lifetime explicitly.
 */

#include "host.h"

int setup(Context* ctx) {

    auto device_name = ctx->device.get_info<xrt::info::device::name>();
    std::cout << "[USR] INFO: found device " << device_name << "\n";

    ctx->uuid = ctx->device.load_xclbin(ctx->xclbin_path);
    std::cout << "[USR] INFO: loaded xclbin " << ctx->xclbin_path << "\n";

    ctx->loadkrnl  = xrt::kernel(ctx->device, ctx->uuid, "load");
    ctx->akrnl     = xrt::kernel(ctx->device, ctx->uuid, "a");
    ctx->ckrnl     = xrt::kernel(ctx->device, ctx->uuid, "c");
    ctx->ekrnl     = xrt::kernel(ctx->device, ctx->uuid, "e");
    ctx->gkrnl     = xrt::kernel(ctx->device, ctx->uuid, "g");
    ctx->storekrnl = xrt::kernel(ctx->device, ctx->uuid, "store");
    std::cout << "[USR] INFO: all kernels found.\n";

    std::cout << "[USR] INFO: allocating buffers in global memory.\n";
    ctx->inputNumList     = xrt::bo(ctx->device, sizeof(int) * MAX_EVENTS, ctx->loadkrnl.group_id(2));
    ctx->inputFeatureList = xrt::bo(ctx->device, sizeof(in_harness_t) * II * MAX_EVENTS, ctx->loadkrnl.group_id(1));
    ctx->outputNumList    = xrt::bo(ctx->device, sizeof(int) * MAX_EVENTS, ctx->storekrnl.group_id(2));
    ctx->outputFeatureList= xrt::bo(ctx->device, sizeof(out_harness_t) * II * MAX_EVENTS, ctx->storekrnl.group_id(1));

    ctx->inputNumListMap      = ctx->inputNumList.map<int*>();
    ctx->inputFeatureListMap  = ctx->inputFeatureList.map<in_harness_t*>();
    ctx->outputNumListMap     = ctx->outputNumList.map<int*>();
    ctx->outputFeatureListMap = ctx->outputFeatureList.map<out_harness_t*>();

    std::cout << "[USR] INFO: populating input buffers.\n";

    for(int i = 0; i < MAX_EVENTS * II; i++) {
        in_harness_t zero = {0};
        ctx->inputFeatureListMap[i] = zero;
    }

    for (int i = 0; i < MAX_EVENTS * II; i++) {
        out_harness_t zero = {0};
        ctx->outputFeatureListMap[i] = zero;
    }

    std::cout << "[USR] INFO: syncing input buffers to device.\n";
    ctx->inputNumList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx->inputFeatureList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx->outputNumList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx->outputFeatureList.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "[USR] INFO: creating run objects.\n";
    ctx->loadrun  = xrt::run(ctx->loadkrnl);
    ctx->loadrun.set_arg(1, ctx->inputFeatureList);
    ctx->loadrun.set_arg(2, ctx->inputNumList);

    ctx->arun = xrt::run(ctx->akrnl);
    ctx->crun = xrt::run(ctx->ckrnl);
    ctx->erun = xrt::run(ctx->ekrnl);
    ctx->grun = xrt::run(ctx->gkrnl);

    ctx->storerun = xrt::run(ctx->storekrnl);
    ctx->storerun.set_arg(1, ctx->outputFeatureList);
    ctx->storerun.set_arg(2, ctx->outputNumList);

    std::cout << "[USR] INFO: setup() complete.\n";
    return 0;
}

int run(Context* ctx) {

    std::cout << "[USR] INFO: resetting AIE graph.\n";
    ctx->aie_graph = xrt::graph(ctx->device, ctx->uuid, "g");
    ctx->aie_graph->reset();

    std::cout << "[USR] INFO: arming AIE graph for " << ctx->num_events << " events.\n";
    ctx->aie_graph->run(ctx->num_events);

    std::cout << "[USR] INFO: starting all kernels.\n";
    ctx->storerun.start();
    ctx->grun.start();
    ctx->erun.start();
    ctx->crun.start();
    ctx->arun.start();
    ctx->loadrun.start();

    std::cout << "[USR] INFO: waiting for load...";  
    ctx->loadrun.wait();  
    std::cout << " done.\n";
    std::cout << "[USR] INFO: waiting for a...";     
    ctx->arun.wait();     
    std::cout << " done.\n";
    std::cout << "[USR] INFO: waiting for c...";     
    ctx->crun.wait();     
    std::cout << " done.\n";
    std::cout << "[USR] INFO: waiting for e...";     
    ctx->erun.wait();     
    std::cout << " done.\n";
    std::cout << "[USR] INFO: waiting for g...";     
    ctx->grun.wait();     
    std::cout << " done.\n";
    std::cout << "[USR] INFO: waiting for store..."; 
    ctx->storerun.wait(); 
    std::cout << " done.\n";

    std::cout << "[USR] INFO: syncing output buffer from device.\n";
    ctx->outputFeatureList.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::vector<std::vector<std::vector<int16_t>>> output_features(
        ctx->num_events,
        std::vector<std::vector<int16_t>>(N, std::vector<int16_t>(MODEL_OUTPUT_WIDTH, 0)));
    std::vector<std::vector<std::vector<int16_t>>> output_cps(
        ctx->num_events,
        std::vector<std::vector<int16_t>>(N, std::vector<int16_t>(1, 0)));

    for (int e = 0; e < ctx->num_events * II; e++) {
        out_harness_t burst = ctx->outputFeatureListMap[e];
        for (int p = 0; p < PAR; p++) {
            for (int f = 0; f < MODEL_OUTPUT_WIDTH; f++) {
                output_features[e / II][(e % II) * PAR + p][f] = burst[p * 8 + f];
            }
            output_cps[e / II][(e % II) * PAR + p][0] = burst[p * 8 + 7];
        }
    }

    HighFive::File output_file(ctx->output_path,HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    std::vector<size_t> feature_dims{static_cast<size_t>(ctx->num_events),static_cast<size_t>(N),static_cast<size_t>(MODEL_OUTPUT_WIDTH)};
    output_file.createDataSet<int16_t>("int_features",HighFive::DataSpace(feature_dims)).write(output_features);

    std::vector<size_t> cps_dims{static_cast<size_t>(ctx->num_events),static_cast<size_t>(N), 1UL};
    output_file.createDataSet<int16_t>("cps",HighFive::DataSpace(cps_dims)).write(output_cps);

    std::cout << "[USR] INFO: run() complete.\n";
    return 0;
}

int load(Context* ctx) {

    {
        HighFive::File file(ctx->input_path, HighFive::File::ReadOnly);
        auto num_data = file.getDataSet("num").read<std::vector<int16_t>>();
        int available = static_cast<int>(num_data.size());
        ctx->num_events = std::min(ctx->num_events, available);

        for (int e = 0; e < ctx->num_events; e++) {
            ctx->inputNumListMap[e]  = num_data[e];
            ctx->outputNumListMap[e] = num_data[e];
        }
    }

    {
        HighFive::File file(ctx->input_path, HighFive::File::ReadOnly);
        auto data = file.getDataSet("cell").read<std::vector<std::vector<std::vector<int16_t>>>>();

        for (int e = 0; e < ctx->num_events * II; e++) {
            in_harness_t burst;
            for (int p = 0; p < PAR; p++)
                for (int f = 0; f < MODEL_INPUT_WIDTH; f++)
                    burst[p * MODEL_INPUT_WIDTH + f] =
                        data[e / II][(e % II) * PAR + p][f];
            ctx->inputFeatureListMap[e] = burst;
        }
    }

    ctx->loadrun.set_arg(0, ctx->num_events);
    ctx->arun.set_arg(0, ctx->num_events);
    ctx->crun.set_arg(0, ctx->num_events);
    ctx->erun.set_arg(0, ctx->num_events);
    ctx->grun.set_arg(0, ctx->num_events);
    ctx->storerun.set_arg(0, ctx->num_events);

    std::cout << "[USR] INFO: input data reloaded from " << ctx->input_path << "\n";

    ctx->inputNumList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx->inputFeatureList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx->outputNumList.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::cout << "[USR] INFO: buffers synced to device.\n";

    ctx->grun.set_arg(1, ctx->beta_threshold);
    std::cout << "[USR] INFO: beta_threshold updated to " << ctx->beta_threshold << "\n";

    ctx->grun.set_arg(2, ctx->distance_threshold);
    std::cout << "[USR] INFO: distance_threshold updated to "  << ctx->distance_threshold << "\n";

    std::cout << "[USR] INFO: update() complete.\n";
    return 0;
}

void cleanup(Context* ctx) {
    if (!ctx) return;
    std::cout << "[USR] INFO: releasing resources.\n";
    delete ctx;
    std::cout << "[USR] INFO: cleanup() complete.\n";
}
