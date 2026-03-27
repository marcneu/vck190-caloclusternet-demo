/**
 * main.cpp
 *
 * Minimal smoke-test entry point for the refactored host.cpp.
 * Mirrors the original CLI interface so existing test scripts keep working.
 *
 * Usage:
 *   ./aie_test [--xclbin_path <f>] [--input_path <f>] [--output_path <f>]
 *              [--num_events <n>] [--beta_threshold <b>] [--distance_threshold <d>]
 *              [--verbose]
 */
 
#include "host.h"
#include <cstdlib>
#include <getopt.h>
#include <iostream>
 
/* ---- defaults (same as original) --------------------------------------- */
static std::string xclbin_path        = "kernel.xclbin";
static std::string input_path         = "input.h5";
static std::string output_path        = "output.h5";
static int         num_events         = 1;
static int         verbose            = 0;
static int16_t     beta_threshold     = 0xEE66;
static int16_t     distance_threshold = 0x0100;
 
static void parse_args(int argc, char* argv[]) {
    static struct option opts[] = {
        {"verbose",            no_argument,       &verbose, 1},
        {"xclbin_path",        required_argument, 0, 'f'},
        {"num_events",         required_argument, 0, 'n'},
        {"input_path",         required_argument, 0, 'i'},
        {"output_path",        required_argument, 0, 'o'},
        {"beta_threshold",     required_argument, 0, 'b'},
        {"distance_threshold", required_argument, 0, 'd'},
        {0, 0, 0, 0}
    };
    int c, idx = 0;
    while ((c = getopt_long(argc, argv, "f:n:i:o:b:d:", opts, &idx)) != -1) {
        switch (c) {
            case 0:   break;
            case 'f': xclbin_path        = optarg;        break;
            case 'i': input_path         = optarg;        break;
            case 'o': output_path        = optarg;        break;
            case 'n': num_events         = std::min(atoi(optarg), MAX_EVENTS); break;
            case 'b': beta_threshold     = (int16_t)atoi(optarg); break;
            case 'd': distance_threshold = (int16_t)atoi(optarg); break;
            default:  std::cerr << "Unknown option.\n"; std::exit(1);
        }
    }
}
 
int main(int argc, char* argv[]) {
 
    parse_args(argc, argv);
 
    /* 1. Allocate context ------------------------------------------------ */
    Context* ctx = new Context(
        xclbin_path, input_path, output_path,
        num_events, verbose,
        beta_threshold, distance_threshold);
 
    /* 2. Hardware setup -------------------------------------------------- */
    if (setup(ctx) != 0) {
        std::cerr << "[USR] ERROR: setup() failed.\n";
        cleanup(ctx);
        return 1;
    }
 
    /* 3. Load input data into buffers ------------------------------------ */
    if (load(ctx) != 0) {
        std::cerr << "[USR] ERROR: load() failed.\n";
        cleanup(ctx);
        return 1;
    }
 
    /* 4. Run inference --------------------------------------------------- */
    if (run(ctx) != 0) {
        std::cerr << "[USR] ERROR: run() failed.\n";
        cleanup(ctx);
        return 1;
    }
 
    /* 5. Release all resources ------------------------------------------- */
    cleanup(ctx);
 
    std::cout << "[USR] SUCCESS\n";
    return 0;
}