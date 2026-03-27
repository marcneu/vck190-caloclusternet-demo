#ifndef E_H
#define E_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <array>

#include "global.h"
#include "weights.h"

#include "gravnetconv.h"
#include "store_to_buffer.h"
#include "load_from_buffer.h"
#include "retile.h"
#include "pack.h"
#include "unpack.h"
#include "split.h"
#include "rate_transition.h"

void e(int& numEvents,
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_1011_plio[2*PAR],
       hls::stream<ap_axiu<128, 0, 0, 0>> layer_12_plio[2*PAR],
       hls::stream<int> &num_stream);

#endif