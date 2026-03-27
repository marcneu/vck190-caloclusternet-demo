#include <adf.h>
#include "adf/new_frontend/adf.h"
#include "params.h"
#include "dense.h"
#include "retile.h"
#include "concat.h"
#include "weights.h"

using namespace adf;

const int PAR = 4;

/* Footprint: (COLSxROWS): (6x1)
*/
template<int COL, int ROW>
class BGraph: public adf::graph {
private:
  kernel layer_2_kernel;
  kernel retile_2_kernel;
  kernel layer_34_kernel;
  kernel retile_34_kernel;

  static constexpr int BANKCOL = (ROW % 2 != 0) ? COL + 1 : COL;
  static constexpr int BANKROW = ROW;
public:
  port<input>  graph_layer_0_input;
  port<output> graph_layer_2_output;
  port<output> graph_layer_34_output;

  BGraph() {
    // ── layer_2 @ tile(COL+1, ROW) ────────────────────────────────────────
    layer_2_kernel = kernel::create_object<dense<uint8,uint8,int8,LAYER_2_M,LAYER_2_K,LAYER_2_N,LAYER_2_SO,LAYER_2_SB,LAYER_2_OPTLVL>>(layer_2_weights, layer_2_biases);
    source(layer_2_kernel) = "dense.cpp";
    location<kernel>   (layer_2_kernel)          = tile(COL+1,ROW);
    location<stack>    (layer_2_kernel)          = bank(BANKCOL,BANKROW,3);
    location<buffer>   (layer_2_kernel.in[0])    = { bank(BANKCOL,BANKROW,0), bank(BANKCOL,BANKROW,1) };
    location<parameter>(layer_2_kernel.param[0]) = bank(BANKCOL,BANKROW,2);
    location<parameter>(layer_2_kernel.param[1]) = bank(BANKCOL,BANKROW,2);
    runtime<ratio>(layer_2_kernel) = 1.0;

    connect(graph_layer_0_input, layer_2_kernel.in[0]);
    dimensions(layer_2_kernel.in[0])  = {LAYER_2_INPUT_LEN};
    dimensions(layer_2_kernel.out[0]) = {LAYER_2_OUTPUT_LEN};

    // ── retile_2 @ tile(COL+2, ROW) ────────────────────────────────────
    retile_2_kernel = kernel::create(retile_4x4_to_4x8_multicast2<uint8, LAYER_2_N*LAYER_2_M/32>);
    source(retile_2_kernel) = "retile.cpp";
    location<kernel>(retile_2_kernel)         = tile(COL+2, ROW);
    location<stack> (retile_2_kernel)          = bank(BANKCOL+1, BANKROW, 3);
    location<buffer>(retile_2_kernel.in[0])    = { bank(BANKCOL+1,BANKROW,0), bank(BANKCOL+1,BANKROW,1) };
    location<buffer>(retile_2_kernel.out[0])   = { bank(BANKCOL+2,BANKROW,2), bank(BANKCOL+2,BANKROW,3) };
    location<buffer>(retile_2_kernel.out[1])   = { bank(BANKCOL+2,BANKROW,0), bank(BANKCOL+2,BANKROW,1) };
    runtime<ratio>(retile_2_kernel) = 1.0;

    connect(layer_2_kernel.out[0], retile_2_kernel.in[0]);
    dimensions(retile_2_kernel.in[0])  = {LAYER_2_OUTPUT_LEN};
    dimensions(retile_2_kernel.out[0]) = {LAYER_2_OUTPUT_LEN};
    dimensions(retile_2_kernel.out[1]) = {LAYER_2_OUTPUT_LEN};

    connect s1 (retile_2_kernel.out[0], graph_layer_2_output);
    fifo_depth(s1) = 2048;
    location<fifo>(s1) = dma_fifo(aie_tile, BANKCOL+3, BANKROW, 0x0000, 2048);

    // ── layer_34 @ tile(COL+3, ROW) ────────────────────────────────────────
    layer_34_kernel = kernel::create_object<dense<uint8,int8,int8,LAYER_34_M,LAYER_34_K,LAYER_34_N,LAYER_34_SO,LAYER_34_SB,LAYER_34_OPTLVL>>(layer_34_weights, layer_34_biases);
    source(layer_34_kernel) = "dense.cpp";
    location<kernel>   (layer_34_kernel)           = tile(COL+4,ROW);
    location<stack>    (layer_34_kernel)           = bank(BANKCOL+4,BANKROW, 3);
    location<buffer>   (layer_34_kernel.in[0])     = { bank(BANKCOL+3,BANKROW,2), bank(BANKCOL+3,BANKROW,3) };
    location<parameter>(layer_34_kernel.param[0])  = bank(BANKCOL+4,BANKROW, 2);
    location<parameter>(layer_34_kernel.param[1])  = bank(BANKCOL+4,BANKROW, 2);
    runtime<ratio>(layer_34_kernel) = 1.0;

    connect(retile_2_kernel.out[1], layer_34_kernel.in[0]);
    dimensions(layer_34_kernel.in[0])  = {LAYER_34_INPUT_LEN};
    dimensions(layer_34_kernel.out[0]) = {LAYER_34_OUTPUT_LEN};

    // ── retile_34 @ tile(COL+4, ROW) ───────────────────────────────────────
    retile_34_kernel = kernel::create(retile_4x4_to_4x8<int8, LAYER_34_N*LAYER_34_M/32>);
    source(retile_34_kernel) = "retile.cpp";
    location<kernel>(retile_34_kernel)          = tile(COL+5,ROW);
    location<stack> (retile_34_kernel)          = bank(BANKCOL+5,BANKROW, 3);
    location<buffer>(retile_34_kernel.out[0])   = { bank(BANKCOL+5,BANKROW,0), bank(BANKCOL+5,BANKROW,1) };
    runtime<ratio>(retile_34_kernel) = 1.0;

    connect(layer_34_kernel.out[0], retile_34_kernel.in[0]);
    dimensions(retile_34_kernel.in[0])  = {LAYER_34_OUTPUT_LEN};
    dimensions(retile_34_kernel.out[0]) = {LAYER_34_OUTPUT_LEN};
    connect(retile_34_kernel.out[0], graph_layer_34_output);
  }
};

/*
Footprint 12x1
*/
template<int COL, int ROW>
class DGraph: public adf::graph {
private:
  kernel layer_6_kernel;
  kernel layer_7_kernel;
  kernel retile_7_kernel;
  kernel layer_8_kernel;
  kernel retile_8_kernel;
  kernel layer_9_kernel;
  kernel retile_9_kernel;
  kernel layer_1011_kernel;
  kernel retile_1011_kernel;

  static constexpr int BANKCOL = (ROW % 2 != 0) ? COL + 1 : COL;
  static constexpr int BANKROW = ROW;
public:
  port<input>  graph_layer_2_input;
  port<input>  graph_layer_5_input;
  port<output> graph_layer_8_output;
  port<output> graph_layer_9_output;
  port<output> graph_layer_1011_output;

  DGraph() {

    // ── layer_6 @ tile(COL+1, ROW) ──────────────────────────────────────────
    layer_6_kernel = kernel::create(concat_inner_srs<uint8,int8,int8,LAYER_6_INPUT_0_LEN,LAYER_6_INPUT_1_LEN,LAYER_6_INPUT_0_D,LAYER_6_INPUT_1_D,LAYER_6_INPUT_0_P,LAYER_6_INPUT_1_P,LAYER_6_INPUT_0_S,LAYER_6_INPUT_1_S,LAYER_6_OUTPUT_S>);
    source(layer_6_kernel) = "concat.cpp";
    location<kernel>(layer_6_kernel)         =   tile(COL+1,ROW);
    location<stack> (layer_6_kernel)         =   bank(BANKCOL+1,BANKROW,3);
    location<buffer>(layer_6_kernel.in[0])   = { bank(BANKCOL+0,BANKROW,0), bank(BANKCOL+0,BANKROW,1) };
    location<buffer>(layer_6_kernel.in[1])   = { bank(BANKCOL+0,BANKROW,2), bank(BANKCOL+0,BANKROW,3) };
    location<buffer>(layer_6_kernel.out[0])  = { bank(BANKCOL+1,BANKROW,0), bank(BANKCOL+1,BANKROW,1) };
    runtime<ratio>(layer_6_kernel) = 1.0;
    dimensions(layer_6_kernel.in[0])  = {LAYER_6_INPUT_0_LEN};
    dimensions(layer_6_kernel.in[1])  = {LAYER_6_INPUT_1_LEN};
    dimensions(layer_6_kernel.out[0]) = {LAYER_6_OUTPUT_LEN};

    connect s0 (graph_layer_2_input, layer_6_kernel.in[0]);
    connect    (graph_layer_5_input, layer_6_kernel.in[1]);

    // ── layer_7 @ tile(COL+2, ROW) ────────────────────────────────────────
    layer_7_kernel = kernel::create_object<dense<int8,uint8,int8,LAYER_7_M,LAYER_7_K,LAYER_7_N,LAYER_7_SO,LAYER_7_SB,LAYER_7_OPTLVL>>(layer_7_weights, layer_7_biases);
    source(layer_7_kernel) = "dense.cpp";
    location<kernel>   (layer_7_kernel)          =   tile(COL+2,ROW);
    location<stack>    (layer_7_kernel)          =   bank(BANKCOL+2,BANKROW,3);
    location<buffer>   (layer_7_kernel.in[0])    = { bank(BANKCOL+1,BANKROW,0), bank(BANKCOL+1,BANKROW,1) };
    location<parameter>(layer_7_kernel.param[0]) =   bank(BANKCOL+2,BANKROW,2);
    location<parameter>(layer_7_kernel.param[1]) =   bank(BANKCOL+2,BANKROW,2);
    runtime<ratio>(layer_7_kernel) = 1.0;
    dimensions(layer_7_kernel.in[0])  = {LAYER_7_INPUT_LEN};
    dimensions(layer_7_kernel.out[0]) = {LAYER_7_OUTPUT_LEN};

    connect(layer_6_kernel.out[0], layer_7_kernel.in[0]);

    // ── retile_7 @ tile(COL+3, ROW) ─────────────────────────────────────
    retile_7_kernel = kernel::create(retile_4x4_to_4x8<uint8, LAYER_7_N*LAYER_7_M/32>);
    source(retile_7_kernel) = "retile.cpp";
    location<kernel>(retile_7_kernel)         =   tile(COL+3,ROW);
    location<stack> (retile_7_kernel)         =   bank(BANKCOL+3,BANKROW,3);
    location<buffer>(retile_7_kernel.in[0])   = { bank(BANKCOL+2,BANKROW,0), bank(BANKCOL+2,BANKROW,1) };
    location<buffer>(retile_7_kernel.out[0])  = { bank(BANKCOL+3,BANKROW,0), bank(BANKCOL+3,BANKROW,1) };
    runtime<ratio>(retile_7_kernel) = 1.0;
    dimensions(retile_7_kernel.in[0])  = {LAYER_7_OUTPUT_LEN};
    dimensions(retile_7_kernel.out[0]) = {LAYER_7_OUTPUT_LEN};

    connect(layer_7_kernel.out[0], retile_7_kernel.in[0]);

    // ── layer_8 @ tile(COL+4, ROW) ──────────────────────────────────────
    layer_8_kernel = kernel::create_object<dense<uint8,uint8,int8,LAYER_8_M,LAYER_8_K,LAYER_8_N,LAYER_8_SO,LAYER_8_SB,LAYER_8_OPTLVL>>(layer_8_weights, layer_8_biases);
    source(layer_8_kernel) = "dense.cpp";
    location<kernel>   (layer_8_kernel)           =   tile(COL+4, ROW);
    location<stack>    (layer_8_kernel)           =   bank(BANKCOL+4,BANKROW,3);
    location<buffer>   (layer_8_kernel.in[0])     = { bank(BANKCOL+3,BANKROW,0), bank(BANKCOL+3,BANKROW,1) };
    location<parameter>(layer_8_kernel.param[0])  =   bank(BANKCOL+3,BANKROW,2);
    location<parameter>(layer_8_kernel.param[1])  =   bank(BANKCOL+3,BANKROW,2);
    runtime<ratio>(layer_8_kernel) = 1.0;
    dimensions(layer_8_kernel.in[0])  = {LAYER_8_INPUT_LEN};
    dimensions(layer_8_kernel.out[0]) = {LAYER_8_OUTPUT_LEN};

    connect(retile_7_kernel.out[0], layer_8_kernel.in[0]);

    // ── retile_8 @ tile(COL+5, ROW) ─────────────────────────────────────
    retile_8_kernel = kernel::create(retile_4x4_to_4x8_multicast2<uint8, LAYER_8_N*LAYER_8_M/32>);
    source(retile_8_kernel) = "retile.cpp";
    location<kernel>(retile_8_kernel)          =   tile(COL+5,ROW);
    location<stack> (retile_8_kernel)          =   bank(BANKCOL+4,BANKROW,2);
    location<buffer>(retile_8_kernel.in[0])    = { bank(BANKCOL+4,BANKROW,0), bank(BANKCOL+4,BANKROW,1) };
    location<buffer>(retile_8_kernel.out[0])   = { bank(BANKCOL+5,BANKROW,0), bank(BANKCOL+5,BANKROW,1) };
    location<buffer>(retile_8_kernel.out[1])   = { bank(BANKCOL+5,BANKROW,2), bank(BANKCOL+5,BANKROW,3) };
    runtime<ratio>(retile_8_kernel) = 1.0;
    dimensions(retile_8_kernel.in[0])  = {LAYER_8_OUTPUT_LEN};
    dimensions(retile_8_kernel.out[0]) = {LAYER_8_OUTPUT_LEN};
    dimensions(retile_8_kernel.out[1]) = {LAYER_8_OUTPUT_LEN};

    connect(layer_8_kernel.out[0], retile_8_kernel.in[0]);

    connect s2 (retile_8_kernel.out[1], graph_layer_8_output);
    fifo_depth(s2) = 8188;
    location<fifo>(s2) = dma_fifo(aie_tile, BANKCOL+6, BANKROW, 0x0000, 8188);

    // ── layer_9 @ tile(COL+8,ROW) ──────────────────────────────────────
    layer_9_kernel = kernel::create_object<dense<uint8,uint8,int8,LAYER_9_M,LAYER_9_K,LAYER_9_N,LAYER_9_SO,LAYER_9_SB,LAYER_9_OPTLVL>>(layer_9_weights, layer_9_biases);
    source(layer_9_kernel) = "dense.cpp";
    location<kernel>   (layer_9_kernel)           =   tile(COL+8, ROW);
    location<stack>    (layer_9_kernel)           =   bank(BANKCOL+7,BANKROW,3);
    location<buffer>   (layer_9_kernel.in[0])     = { bank(BANKCOL+7,BANKROW,0), bank(BANKCOL+7,BANKROW,1) };
    location<parameter>(layer_9_kernel.param[0])  =   bank(BANKCOL+8,BANKROW,2);
    location<parameter>(layer_9_kernel.param[1])  =   bank(BANKCOL+8,BANKROW,2);
    runtime<ratio>(layer_9_kernel) = 1.0;
    dimensions(layer_9_kernel.in[0])  = {LAYER_9_INPUT_LEN};
    dimensions(layer_9_kernel.out[0]) = {LAYER_9_OUTPUT_LEN};

    connect(retile_8_kernel.out[0], layer_9_kernel.in[0]);

    // ── retile_9 @ tile(COL+9, ROW) ─────────────────────────────────────
    retile_9_kernel = kernel::create(retile_4x4_to_4x8_multicast2<uint8, LAYER_9_N*LAYER_9_M/32>);
    source(retile_9_kernel) = "retile.cpp";
    location<kernel>(retile_9_kernel)          =   tile(COL+9,ROW);
    location<stack> (retile_9_kernel)          =   bank(BANKCOL+8,BANKROW,3);
    location<buffer>(retile_9_kernel.in[0])    = { bank(BANKCOL+8,BANKROW,0), bank(BANKCOL+8,BANKROW,1) };
    location<buffer>(retile_9_kernel.out[0])   = { bank(BANKCOL+9,BANKROW,0), bank(BANKCOL+9,BANKROW,1) };
    location<buffer>(retile_9_kernel.out[1])   = { bank(BANKCOL+9,BANKROW,2), bank(BANKCOL+9,BANKROW,3) };
    runtime<ratio>(retile_9_kernel) = 1.0;
    dimensions(retile_9_kernel.in[0])  = {LAYER_9_OUTPUT_LEN};
    dimensions(retile_9_kernel.out[0]) = {LAYER_9_OUTPUT_LEN};
    dimensions(retile_9_kernel.out[1]) = {LAYER_9_OUTPUT_LEN};

    connect(layer_9_kernel.out[0], retile_9_kernel.in[0]);
    connect s3 (retile_9_kernel.out[1], graph_layer_9_output);
    fifo_depth(s3) = 8188;
    location<fifo>(s3) = dma_fifo(aie_tile, BANKCOL+10, BANKROW, 0x0000, 8188);

    // ── layer_1011 @ tile(COL+12, ROW) ───────────────────────────────────────
    layer_1011_kernel = kernel::create_object<dense<uint8,int8,int8,LAYER_1011_M,LAYER_1011_K,LAYER_1011_N,LAYER_1011_SO,LAYER_1011_SB,LAYER_1011_OPTLVL>>(layer_1011_weights, layer_1011_biases);
    source(layer_1011_kernel) = "dense.cpp";
    location<kernel>   (layer_1011_kernel)           =   tile(COL+12,ROW);
    location<stack>    (layer_1011_kernel)           =   bank(BANKCOL+12,BANKROW,3);
    location<buffer>   (layer_1011_kernel.in[0])     = { bank(BANKCOL+11,BANKROW,0), bank(BANKCOL+11,BANKROW,1) };
    location<parameter>(layer_1011_kernel.param[0])  =   bank(BANKCOL+12,BANKROW,2);
    location<parameter>(layer_1011_kernel.param[1])  =   bank(BANKCOL+12,BANKROW,2);
    runtime<ratio>(layer_1011_kernel) = 1.0;
    dimensions(layer_1011_kernel.in[0])  = {LAYER_1011_INPUT_LEN};
    dimensions(layer_1011_kernel.out[0]) = {LAYER_1011_OUTPUT_LEN};

    connect(retile_9_kernel.out[0], layer_1011_kernel.in[0]);

    // ── retile_1011 @ tile(COL+13, ROW) ────────────────────────────────────
    retile_1011_kernel = kernel::create(retile_4x4_to_4x8<int8, LAYER_1011_N*LAYER_1011_M/32>);
    source(retile_1011_kernel) = "retile.cpp";
    location<kernel>(retile_1011_kernel)          =   tile(COL+13,ROW);
    location<stack> (retile_1011_kernel)          =   bank(BANKCOL+13,BANKROW, 3);
    location<buffer>(retile_1011_kernel.in[0])    = { bank(BANKCOL+12,BANKROW,0), bank(BANKCOL+12,BANKROW,1) };
    location<buffer>(retile_1011_kernel.out[0])   = { bank(BANKCOL+13,BANKROW,0), bank(BANKCOL+13,BANKROW,1) };
    runtime<ratio>(retile_1011_kernel) = 1.0;
    dimensions(retile_1011_kernel.in[0])  = {LAYER_1011_OUTPUT_LEN};
    dimensions(retile_1011_kernel.out[0]) = {LAYER_1011_OUTPUT_LEN};

    connect(layer_1011_kernel.out[0], retile_1011_kernel.in[0]);
    connect(retile_1011_kernel.out[0], graph_layer_1011_output);
  }
};

/*
  Footprint (COLxROW) : (9x1)
*/
template<int COL, int ROW>
class FGraph: public adf::graph {
private:
  kernel layer_13_kernel;
  kernel layer_14_kernel;
  kernel retile_14_kernel;
  kernel layer_15_kernel;
  kernel retile_15_kernel;
  kernel layer_16_kernel;
  kernel layer_17_kernel;
  kernel retile_17_kernel;

  static constexpr int BANKCOL = (ROW % 2 != 0) ? COL + 1 : COL;
  static constexpr int BANKROW = ROW;
public: 
  port<input>  graph_layer_1_input;
  port<input>  graph_layer_8_input;
  port<input>  graph_layer_9_input;
  port<input>  graph_layer_12_input;
  port<output> graph_layer_17_output;

  FGraph(){ 

    
    // ── layer_13_kernel @ tile(COL+1, ROW) ────────────────────────────────────
    layer_13_kernel = kernel::create(concat_inner_srs<uint8,int8,int8, LAYER_13_INPUT_0_LEN, LAYER_13_INPUT_1_LEN,LAYER_13_INPUT_0_D,LAYER_13_INPUT_1_D,LAYER_13_INPUT_0_P,LAYER_13_INPUT_1_P,LAYER_13_INPUT_0_S,LAYER_13_INPUT_1_S,LAYER_13_OUTPUT_S>);
    source(layer_13_kernel) = "concat.cpp";
    location<kernel>(layer_13_kernel)         =   tile(COL+1,ROW);
    location<stack> (layer_13_kernel)         =   bank(BANKCOL+1,BANKROW,3);
    location<buffer>(layer_13_kernel.in[0])   = { bank(BANKCOL+0,BANKROW,0), bank(BANKCOL+0,BANKROW,1) };
    location<buffer>(layer_13_kernel.in[1])   = { bank(BANKCOL+0,BANKROW,2), bank(BANKCOL+0,BANKROW,3) };
    location<buffer>(layer_13_kernel.out[0])  = { bank(BANKCOL+1,BANKROW,0), bank(BANKCOL+1,BANKROW,1) };
    runtime<ratio>(layer_13_kernel) = 1.0;
    dimensions(layer_13_kernel.in[0]) = {LAYER_13_INPUT_0_LEN};
    dimensions(layer_13_kernel.in[1]) = {LAYER_13_INPUT_1_LEN};
    dimensions(layer_13_kernel.out[0]) = {LAYER_13_OUTPUT_LEN};

    connect(graph_layer_9_input, layer_13_kernel.in[0]);
    connect(graph_layer_12_input, layer_13_kernel.in[1]);

    // ── layer_14_kernel @ tile(COL+2, ROW) ────────────────────────────────────
    layer_14_kernel = kernel::create_object<dense<int8,uint8,int8, LAYER_14_M, LAYER_14_K, LAYER_14_N, LAYER_14_SO,LAYER_14_SB,LAYER_14_OPTLVL>>(layer_14_weights,layer_14_biases);
    source(layer_14_kernel) = "dense.cpp";
    location<kernel>   (layer_14_kernel)           =   tile(COL+2,ROW);
    location<stack>    (layer_14_kernel)           =   bank(BANKCOL+2,BANKROW,3);
    location<buffer>   (layer_14_kernel.in[0])     = { bank(BANKCOL+1,BANKROW,0), bank(BANKCOL+1,BANKROW,1) };
    location<parameter>(layer_14_kernel.param[0])  =   bank(BANKCOL+2,BANKROW,2);
    location<parameter>(layer_14_kernel.param[1])  =   bank(BANKCOL+2,BANKROW,2);
    runtime<ratio>(layer_14_kernel) = 1.0;
    dimensions(layer_14_kernel.in[0]) = {LAYER_14_INPUT_LEN};
    dimensions(layer_14_kernel.out[0]) = {LAYER_14_OUTPUT_LEN};

    connect(layer_13_kernel.out[0], layer_14_kernel.in[0]);

    // ── retile_14_kernel @ tile(COL+3, ROW) ────────────────────────────────────
    retile_14_kernel = kernel::create(retile_4x4_to_4x8<uint8,LAYER_14_N*LAYER_14_M/32>);
    source(retile_14_kernel) = "retile.cpp";
    location<kernel>(retile_14_kernel)          =   tile(COL+3,ROW);
    location<stack> (retile_14_kernel)          =   bank(BANKCOL+3,BANKROW,3);
    location<buffer>(retile_14_kernel.in[0])    = { bank(BANKCOL+2,BANKROW,0), bank(BANKCOL+2,BANKROW,1) };
    location<buffer>(retile_14_kernel.out[0])   = { bank(BANKCOL+3,BANKROW,0), bank(BANKCOL+3,BANKROW,1) };
    runtime<ratio>(retile_14_kernel) = 1.0;
    dimensions(retile_14_kernel.in[0]) = {LAYER_14_OUTPUT_LEN};
    dimensions(retile_14_kernel.out[0]) = {LAYER_14_OUTPUT_LEN};

    connect(layer_14_kernel.out[0], retile_14_kernel.in[0]);

    // ── layer_15_kernel @ tile(COL+4, ROW) ────────────────────────────────────
    layer_15_kernel = kernel::create_object<dense<uint8,uint8,int8, LAYER_15_M, LAYER_15_K, LAYER_15_N, LAYER_15_SO,LAYER_15_SB,LAYER_15_OPTLVL>>(layer_15_weights,layer_15_biases);
    source(layer_15_kernel) = "dense.cpp";
    location<kernel>(layer_15_kernel)              =   tile(COL+4,ROW);
    location<stack>    (layer_15_kernel)           =   bank(BANKCOL+4,BANKROW,3);
    location<buffer>   (layer_15_kernel.in[0])     = { bank(BANKCOL+3,BANKROW,0), bank(BANKCOL+3,BANKROW,1) };
    location<parameter>(layer_15_kernel.param[0])  =   bank(BANKCOL+3,BANKROW,2);
    location<parameter>(layer_15_kernel.param[1])  =   bank(BANKCOL+3,BANKROW,2);
    runtime<ratio>(layer_15_kernel) = 1.0;
    dimensions(layer_15_kernel.in[0]) = {LAYER_15_INPUT_LEN};
    dimensions(layer_15_kernel.out[0]) = {LAYER_15_OUTPUT_LEN};

    connect(retile_14_kernel.out[0], layer_15_kernel.in[0]);

    // ── retile_15_kernel @ tile(COL+5, ROW) ────────────────────────────────────
    retile_15_kernel = kernel::create(retile_4x4_to_4x8<uint8,LAYER_15_N*LAYER_15_M/32>);
    source(retile_15_kernel) = "retile.cpp";
    location<kernel>(retile_15_kernel)          =   tile(COL+5,ROW);
    location<stack> (retile_15_kernel)          =   bank(BANKCOL+4,BANKROW,2);
    location<buffer>(retile_15_kernel.in[0])    = { bank(BANKCOL+4,BANKROW,0), bank(BANKCOL+4,BANKROW,1) };
    location<buffer>(retile_15_kernel.out[0])   = { bank(BANKCOL+5,BANKROW,0), bank(BANKCOL+5,BANKROW,1) };
    runtime<ratio>(retile_15_kernel) = 1.0;
    dimensions(retile_15_kernel.in[0]) = {LAYER_15_OUTPUT_LEN};
    dimensions(retile_15_kernel.out[0]) = {LAYER_15_OUTPUT_LEN};

    connect(layer_15_kernel.out[0], retile_15_kernel.in[0]);

    // ── layer_16_kernel @ tile(COL+6, ROW) ────────────────────────────────────
    layer_16_kernel = kernel::create(concat_inner_srs<uint8,uint8,uint8,uint8, LAYER_16_INPUT_0_LEN, LAYER_16_INPUT_1_LEN,LAYER_16_INPUT_2_LEN,LAYER_16_INPUT_0_D,LAYER_16_INPUT_1_D,LAYER_16_INPUT_2_D,LAYER_16_INPUT_0_P,LAYER_16_INPUT_1_P,LAYER_16_INPUT_2_P,LAYER_16_INPUT_0_S,LAYER_16_INPUT_1_S,LAYER_16_INPUT_2_S,LAYER_16_OUTPUT_S>);
    source(layer_16_kernel) = "concat.cpp";
    location<kernel>(layer_16_kernel) = tile(COL+6,ROW);
    location<stack> (layer_16_kernel)          =   bank(BANKCOL+6,BANKROW,3);
    location<buffer>(layer_16_kernel.in[0])    = { bank(BANKCOL+5,BANKROW,2), bank(BANKCOL+5,BANKROW,3) };
    location<buffer>(layer_16_kernel.in[1])    = { bank(BANKCOL+5,BANKROW,0), bank(BANKCOL+5,BANKROW,1) }; //Lets try this out, might deadlock
    location<buffer>(layer_16_kernel.in[2])    = { bank(BANKCOL+5,BANKROW,2), bank(BANKCOL+5,BANKROW,3) };
    location<buffer>(layer_16_kernel.out[0])   = { bank(BANKCOL+6,BANKROW,0), bank(BANKCOL+6,BANKROW,1) };
    runtime<ratio>(layer_16_kernel) = 1.0;
    dimensions(layer_16_kernel.in[0]) = {LAYER_16_INPUT_0_LEN};
    dimensions(layer_16_kernel.in[1]) = {LAYER_16_INPUT_1_LEN};
    dimensions(layer_16_kernel.in[2]) = {LAYER_16_INPUT_2_LEN};
    dimensions(layer_16_kernel.out[0]) = {LAYER_16_OUTPUT_LEN};

    connect(graph_layer_1_input, layer_16_kernel.in[0]);
    connect(graph_layer_8_input, layer_16_kernel.in[1]);
    connect(retile_15_kernel.out[0], layer_16_kernel.in[2]);

    // ── layer_16_kernel @ tile(COL+7, ROW) ────────────────────────────────────
    layer_17_kernel = kernel::create_object<dense<uint8,uint8,int8, LAYER_17_M, LAYER_17_K, LAYER_17_N, LAYER_17_SO,LAYER_17_SB,LAYER_17_OPTLVL>>(layer_17_weights,layer_17_biases);
    source(layer_17_kernel) = "dense.cpp";
    location<kernel>(layer_17_kernel)              =   tile(COL+7,ROW);
    location<stack>    (layer_17_kernel)           =   bank(BANKCOL+7,BANKROW,3);
    location<buffer>   (layer_17_kernel.in[0])     = { bank(BANKCOL+6,BANKROW,0), bank(BANKCOL+6,BANKROW,1) };
    location<parameter>(layer_17_kernel.param[0])  =   bank(BANKCOL+6,BANKROW,2);
    location<parameter>(layer_17_kernel.param[1])  =   bank(BANKCOL+6,BANKROW,2);
    runtime<ratio>(layer_17_kernel) = 1.0;
    dimensions(layer_17_kernel.in[0]) = {LAYER_17_INPUT_LEN};
    dimensions(layer_17_kernel.out[0]) = {LAYER_17_OUTPUT_LEN};

    connect(layer_16_kernel.out[0], layer_17_kernel.in[0]);

    // ── retile_17_kernel @ tile(COL+8, ROW) ────────────────────────────────────
    retile_17_kernel = kernel::create(retile_4x4_to_4x8<uint8,LAYER_17_N*LAYER_17_M/32>);
    source(retile_17_kernel) = "retile.cpp";
    location<kernel>(retile_17_kernel)          =   tile(COL+8,ROW);
    location<stack> (retile_17_kernel)          =   bank(BANKCOL+8,BANKROW,3);
    location<buffer>(retile_17_kernel.in[0])    = { bank(BANKCOL+7,BANKROW,0), bank(BANKCOL+7,BANKROW,1) };
    location<buffer>(retile_17_kernel.out[0])   = { bank(BANKCOL+8,BANKROW,0), bank(BANKCOL+8,BANKROW,1) };
    runtime<ratio>(retile_17_kernel) = 1.0;
    dimensions(retile_17_kernel.in[0]) = {LAYER_17_OUTPUT_LEN};
    dimensions(retile_17_kernel.out[0]) = {LAYER_17_OUTPUT_LEN};

    connect(layer_17_kernel.out[0], retile_17_kernel.in[0]);
    connect(retile_17_kernel.out[0], graph_layer_17_output);
  }
};

class TopGraph : public adf::graph {
private:

public:
  input_plio  graph_layer_0_input[PAR];
  input_plio  graph_layer_1_input[PAR];
  input_plio  graph_layer_5_input[PAR];
  input_plio  graph_layer_12_input[PAR];
  output_plio graph_layer_34_output[PAR];
  output_plio graph_layer_1011_output[PAR];
  output_plio graph_layer_17_output[PAR];

  BGraph<6,0> bgraph_0;
  BGraph<5,1> bgraph_1;
  BGraph<6,2> bgraph_2;
  BGraph<5,3> bgraph_3;
  DGraph<12,0> dgraph_0;
  DGraph<11,1> dgraph_1;
  DGraph<12,2> dgraph_2;
  DGraph<11,3> dgraph_3;
  FGraph<26,0>  fgraph_0;
  FGraph<25,1>  fgraph_1;
  FGraph<26,2>  fgraph_2;
  FGraph<25,3>  fgraph_3;

  TopGraph() {

    // ── Instance 0 ───────────────────────────────────────────────────────
    graph_layer_0_input[0]  = input_plio::create("layer_0_input_0",  plio_64_bits,  "layer_0_input_0.txt",  250.0);
    graph_layer_1_input[0]  = input_plio::create("layer_1_input_0",  plio_64_bits,  "layer_1_input_0.txt",  250.0);
    graph_layer_34_output[0] = output_plio::create("layer_34_output_0", plio_128_bits, "layer_34_output_0.txt", 250.0);
    graph_layer_5_input[0]  = input_plio::create("layer_5_input_0",  plio_128_bits, "layer_5_input_0.txt",  250.0);
    graph_layer_12_input[0] = input_plio::create("layer_12_input_0", plio_128_bits, "layer_12_input_0.txt", 250.0);
    graph_layer_1011_output[0] = output_plio::create("layer_1011_output_0", plio_128_bits,  "layer_1011_output_0.txt", 250.0);
    graph_layer_17_output[0] = output_plio::create("layer_17_output_0", plio_128_bits, "layer_17_output_0.txt", 250.0);

    connect(graph_layer_0_input[0].out[0],  bgraph_0.graph_layer_0_input);
    connect(bgraph_0.graph_layer_34_output,  graph_layer_34_output[0].in[0]);
    connect(bgraph_0.graph_layer_2_output,  dgraph_0.graph_layer_2_input);
    connect(graph_layer_5_input[0].out[0],  dgraph_0.graph_layer_5_input);
    connect(dgraph_0.graph_layer_1011_output, graph_layer_1011_output[0].in[0]);
    connect(graph_layer_1_input[0].out[0],  fgraph_0.graph_layer_1_input);
    connect(graph_layer_12_input[0].out[0], fgraph_0.graph_layer_12_input);
    connect(dgraph_0.graph_layer_8_output,  fgraph_0.graph_layer_8_input);
    connect(dgraph_0.graph_layer_9_output,  fgraph_0.graph_layer_9_input);
    connect(fgraph_0.graph_layer_17_output, graph_layer_17_output[0].in[0]);

    // ── Instance 1 ───────────────────────────────────────────────────────
    graph_layer_0_input[1]  = input_plio::create("layer_0_input_1",  plio_64_bits,  "layer_0_input_1.txt",  250.0);
    graph_layer_1_input[1]  = input_plio::create("layer_1_input_1",  plio_64_bits,  "layer_1_input_1.txt",  250.0);
    graph_layer_5_input[1]  = input_plio::create("layer_5_input_1",  plio_128_bits, "layer_5_input_1.txt",  250.0);
    graph_layer_12_input[1] = input_plio::create("layer_12_input_1", plio_128_bits, "layer_12_input_1.txt", 250.0);
    graph_layer_34_output[1] = output_plio::create("layer_34_output_1", plio_128_bits, "layer_34_output_1.txt", 250.0);
    graph_layer_1011_output[1] = output_plio::create("layer_1011_output_1", plio_64_bits,  "layer_1011_output_1.txt", 250.0);
    graph_layer_17_output[1] = output_plio::create("layer_17_output_1", plio_128_bits, "layer_17_output_1.txt", 250.0);
    
    connect(graph_layer_0_input[1].out[0],  bgraph_1.graph_layer_0_input);
    connect(bgraph_1.graph_layer_34_output,  graph_layer_34_output[1].in[0]);
    connect(bgraph_1.graph_layer_2_output,  dgraph_1.graph_layer_2_input);
    connect(graph_layer_5_input[1].out[0],  dgraph_1.graph_layer_5_input);
    connect(dgraph_1.graph_layer_1011_output, graph_layer_1011_output[1].in[0]);
    connect(graph_layer_1_input[1].out[0],  fgraph_1.graph_layer_1_input);
    connect(graph_layer_12_input[1].out[0], fgraph_1.graph_layer_12_input);
    connect(dgraph_1.graph_layer_8_output,  fgraph_1.graph_layer_8_input);
    connect(dgraph_1.graph_layer_9_output,  fgraph_1.graph_layer_9_input);
    connect(fgraph_1.graph_layer_17_output, graph_layer_17_output[1].in[0]);

    // ── Instance 2 ───────────────────────────────────────────────────────
    graph_layer_0_input[2]  = input_plio::create("layer_0_input_2",  plio_64_bits,  "layer_0_input_2.txt",  250.0);
    graph_layer_1_input[2]  = input_plio::create("layer_1_input_2",  plio_64_bits,  "layer_1_input_2.txt",  250.0);
    graph_layer_5_input[2]  = input_plio::create("layer_5_input_2",  plio_128_bits, "layer_5_input_2.txt",  250.0);
    graph_layer_12_input[2] = input_plio::create("layer_12_input_2", plio_128_bits, "layer_12_input_2.txt", 250.0);
    graph_layer_34_output[2] = output_plio::create("layer_34_output_2", plio_128_bits, "layer_34_output_2.txt", 250.0);
    graph_layer_1011_output[2] = output_plio::create("layer_1011_output_2", plio_64_bits,  "layer_1011_output_2.txt", 250.0);
    graph_layer_17_output[2] = output_plio::create("layer_17_output_2", plio_128_bits, "layer_17_output_2.txt", 250.0);
    
    connect(graph_layer_0_input[2].out[0],  bgraph_2.graph_layer_0_input);
    connect(bgraph_2.graph_layer_34_output,  graph_layer_34_output[2].in[0]);
    connect(bgraph_2.graph_layer_2_output,  dgraph_2.graph_layer_2_input);
    connect(graph_layer_5_input[2].out[0],  dgraph_2.graph_layer_5_input);
    connect(dgraph_2.graph_layer_1011_output, graph_layer_1011_output[2].in[0]);
    connect(graph_layer_1_input[2].out[0],  fgraph_2.graph_layer_1_input);
    connect(graph_layer_12_input[2].out[0], fgraph_2.graph_layer_12_input);
    connect(dgraph_2.graph_layer_8_output,  fgraph_2.graph_layer_8_input);
    connect(dgraph_2.graph_layer_9_output,  fgraph_2.graph_layer_9_input);
    connect(fgraph_2.graph_layer_17_output, graph_layer_17_output[2].in[0]);

    // ── Instance 3 ───────────────────────────────────────────────────────
    graph_layer_0_input[3]  = input_plio::create("layer_0_input_3",  plio_64_bits,  "layer_0_input_3.txt",  250.0);
    graph_layer_1_input[3]  = input_plio::create("layer_1_input_3",  plio_64_bits,  "layer_1_input_3.txt",  250.0);
    graph_layer_5_input[3]  = input_plio::create("layer_5_input_3",  plio_128_bits, "layer_5_input_3.txt",  250.0);
    graph_layer_12_input[3] = input_plio::create("layer_12_input_3", plio_128_bits, "layer_12_input_3.txt", 250.0);
    graph_layer_34_output[3] = output_plio::create("layer_34_output_3", plio_128_bits, "layer_34_output_3.txt", 250.0);
    graph_layer_1011_output[3] = output_plio::create("layer_1011_output_3", plio_64_bits,  "layer_1011_output_3.txt", 250.0);
    graph_layer_17_output[3] = output_plio::create("layer_17_output_3", plio_128_bits, "layer_17_output_3.txt", 250.0);
    
    connect(graph_layer_0_input[3].out[0],  bgraph_3.graph_layer_0_input);
    connect(bgraph_3.graph_layer_34_output,  graph_layer_34_output[3].in[0]);
    connect(bgraph_3.graph_layer_2_output,  dgraph_3.graph_layer_2_input);
    connect(graph_layer_5_input[3].out[0],  dgraph_3.graph_layer_5_input);
    connect(dgraph_3.graph_layer_1011_output, graph_layer_1011_output[3].in[0]);
    connect(graph_layer_1_input[3].out[0],  fgraph_3.graph_layer_1_input);
    connect(graph_layer_12_input[3].out[0], fgraph_3.graph_layer_12_input);
    connect(dgraph_3.graph_layer_8_output,  fgraph_3.graph_layer_8_input);
    connect(dgraph_3.graph_layer_9_output,  fgraph_3.graph_layer_9_input);
    connect(fgraph_3.graph_layer_17_output, graph_layer_17_output[3].in[0]);

  }
};