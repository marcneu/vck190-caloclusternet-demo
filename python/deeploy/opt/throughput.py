from importlib.resources import files
from pathlib import Path
import pandas as pd
import math

from deeploy.util.types import QFormat,Quantizer
from deeploy.util.logging import logger

_REPO_ROOT = Path(__file__).parent.parent  # adjust depth to your module location
_DATA_FILE = _REPO_ROOT / "data" / "sweep.csv"

_AIE_DENSE_SWEEP : pd.DataFrame = pd.read_csv(
        _DATA_FILE,
        dtype={
            "pe_type": str,
            "dtype": str,
            "status": str,
            "optlvl": int,
            "k": int,
            "n": int,
            "m": int,
        },
        na_values=["NaN"],
    )

def get_aie_cycles(
    k: int,
    n: int,
    m: int,
    pe_type: str = "dense",
    dtype: str = "int8",
    optlvl: int = 0,
) -> float:
    mask = (
        (_AIE_DENSE_SWEEP["k"] == k)
        & (_AIE_DENSE_SWEEP["n"] == n)
        & (_AIE_DENSE_SWEEP["m"] == m)
        & (_AIE_DENSE_SWEEP["pe_type"] == pe_type)
        & (_AIE_DENSE_SWEEP["dtype"] == dtype)
        & (_AIE_DENSE_SWEEP["optlvl"] == optlvl)
    )
    rows = _AIE_DENSE_SWEEP.loc[mask, "cycles"]
    if len(rows) == 0:
        raise KeyError(f"No entry found for {k=}, {n=}, {m=}, {pe_type=}, {dtype=}, {optlvl=}")
    if len(rows) > 1:
        raise ValueError(f"Multiple entries found for {k=}, {n=}, {m=}, {pe_type=}, {dtype=}, {optlvl=}")
    return rows.iloc[0]

def get_best_optlvl(
    k: int,
    n: int,
    m: int,
    pe_type: str,
    dtype: str,
) -> int:
    mask = (
        (_AIE_DENSE_SWEEP["k"] == k)
        & (_AIE_DENSE_SWEEP["n"] == n)
        & (_AIE_DENSE_SWEEP["m"] == m)
        & (_AIE_DENSE_SWEEP["pe_type"] == pe_type)
        & (_AIE_DENSE_SWEEP["dtype"] == dtype)
        & (_AIE_DENSE_SWEEP["cycles"].notna())
    )
    valid = _AIE_DENSE_SWEEP.loc[mask, "optlvl"]
    if valid.empty:
        print(f"No entry found for {k=}, {n=}, {m=}, {pe_type=}, {dtype=}. Defaulting to optlvl 0.")
        return 0
    else:
        return int(valid.max())


"""
requriements:
    - "eps": Requested Throughput in Events per Second (EPS)
    - "plfreq": Frequency of the Programmable Logic (PL) in Hz
    - "aiefreq": Frequency of the AI Engine (AIE) in Hz
"""

def opt_dense_pe(pe: dict, reqs:dict) -> dict:
    cycles_per_stage_pl = reqs["plfreq"] / reqs["eps"]
    cycles_per_stage_aie = reqs["aiefreq"] / reqs["eps"]
    
    if(pe["partition"] == "PL"):
        max_cycles = reqs["plfreq"] / reqs["eps"]
    elif(pe["partition"] == "AIE"):

        k=int(pe["k"])
        m=int(pe["m"])
        n=int(pe["n"])
        par=int(pe["par"])
        dtype = "int8" if (pe["output_type_cstr"] == "int8" or pe["output_type_cstr"] == "uint8") else "int16"

        while True:

            optlvl = get_best_optlvl(k,n,m, "dense",dtype)
            cycles = get_aie_cycles(k,n,m,"dense",dtype,optlvl)

            max_cycles = math.floor(reqs["aiefreq"] / reqs["eps"])
            logger.debug(f"Actual cycles: {cycles}, Max cycles: {max_cycles}, Optlvl: {optlvl}")

            if cycles < max_cycles:
                pe["m"] = int(m)
                pe["par"] = int(par)
                pe["optlvl"] = int(optlvl)
                pe["pipeline_efficiency"] = cycles / max_cycles
                logger.debug("Found configuration that meets the throughput requirements.")
                break
            else:
                m = m / 2
                par = par * 2
    return pe


def opt_throughput(cfg: dict, reqs: dict) -> dict:
    new_cfg = {}
    
    for pe_idx, pe in cfg.items():
        if (pe.get("pe_type") == "dense" or pe.get("pe_type") == "linear"):
            new_cfg[pe_idx] = opt_dense_pe(pe, reqs)
            logger.info(f"Layer {pe_idx} updated par from {pe['par']} to {new_cfg[pe_idx]['par']} with optlvl {new_cfg[pe_idx]['optlvl']}")
        else:
            new_cfg[pe_idx] = pe
            logger.info(f"Layer {pe_idx} updated par from {pe['par']} to {new_cfg[pe_idx]['par']}")

    return new_cfg
    