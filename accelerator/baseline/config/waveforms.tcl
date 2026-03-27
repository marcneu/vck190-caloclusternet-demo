## Remove all waveforms before adding new ones
#remove_wave -of [get_wave_config] [get_waves -of [get_wave_config] -regexp ".*"]

## Set the appropriate paths based upon the platform being used
set scope_path "/tb/DUT/vitis_design_wrapper_i/vitis_design_i"

## Create a wave group called CIPS and add all signals for the CIPS_0 to it
set CIPS [add_wave_group CIPS]
set cips_intf [get_objects -r $scope_path/CIPS_0/* -filter {type==proto_inst}]
add_wave -into $CIPS $cips_intf

## Create a wave group called CIPS_NOC and all signals of the CIPS NoC to it
set CIPS_NOC [add_wave_group CIPS_NOC]
set cips_intf [get_objects -r $scope_path/cips_noc/* -filter {type==proto_inst}]
add_wave -into $CIPS_NOC $cips_intf

## Create a wave group called AIENGINE and all signals of the AI Engine block to it
set AIENGINE [add_wave_group AIENGINE]
set aie_intf [get_objects -r $scope_path/ai_engine_0/* -filter {type==proto_inst}]
add_wave -into $AIENGINE $aie_intf

run all 