"""
infer.py — standalone CLI for running one inference pass on the AIE accelerator.

Input HDF5 datasets (read):
    "num"   int16  (num_events,)
    "cell"  int16  (num_events, N, MODEL_INPUT_WIDTH)

Output HDF5 datasets (written):
    "int_features"  int16  (num_events, N, MODEL_OUTPUT_WIDTH)
    "cps"           int16  (num_events, N)

Usage:
    python standalone.py design.xclbin input.h5 output.h5 [options]
"""

import argparse
import h5py
import numpy as np

from accelerator import Accelerator


def parse_args():
    p = argparse.ArgumentParser(description="Run CaloClusterNet inference on the AMD Versal VCK190")
    p.add_argument("xclbin",            help="Path to the .xclbin bitstream",default="dut.xclbin")
    p.add_argument("input",             help="Input HDF5 file (datasets: num, cell)", default="input.h5")
    p.add_argument("output",            help="Output HDF5 file (datasets: int_features, cps)", default="output.h5")
    p.add_argument("--num-events",      type=int,   default=None, help="Number of events to process (default: all events in file)")
    p.add_argument("--beta-threshold",  type=int,   default=0)
    p.add_argument("--dist-threshold",  type=int,   default=0)
    p.add_argument("--verbose",         action="store_true")
    return p.parse_args()


def load_input(path: str, num_events: int | None):
    with h5py.File(path, "r") as f:
        num_data  = f["num"][:]                  # (events,)
        cell_data = f["cell"][:]                 # (events, N, MODEL_INPUT_WIDTH)

    if num_events is not None:
        num_data  = num_data[:num_events]
        cell_data = cell_data[:num_events]

    return (
        np.asarray(num_data,  dtype=np.int16),
        np.asarray(cell_data, dtype=np.int16),
    )

def main():
    args = parse_args()

    num_data, cell_data = load_input(args.input, args.num_events)
    num_events = len(num_data)
    print(f" Loaded {num_events} events from {args.input}")

    with Accelerator(
        args.xclbin,
        num_events=num_events,
        beta_threshold=args.beta_threshold,
        distance_threshold=args.dist_threshold,
        verbose=args.verbose,
    ) as acc:
        acc.setup()
        acc.load(num_data,
                 cell_data,
                 args.beta_threshold,
                 args.dist_threshold)
        features, cps = acc.run()

    with h5py.File(args.output, "w") as f:
        f.create_dataset("int_features", data=features)   # (events, N, MODEL_OUTPUT_WIDTH)
        f.create_dataset("cps",          data=cps)        # (events, N)
    print(f" Output written to {args.output}")



if __name__ == "__main__":
    main()
