#!/bin/bash

LOG_ARGS="--log.all_level=debug"

PORT_ARGS="tx_port=tcp://*:2000,rx_port=tcp://localhost:2001"

ZMQ_ARGS="--rf.device_name=zmq --rf.device_args=\"${PORT_ARGS},id=enb,base_srate=23.04e6\""

OTHER_ARGS="--enb_files.rr_config=rr1.conf"

echo "Arguments: enb.conf ${LOG_ARGS} ${ZMQ_ARGS} ${OTHER_ARGS} $@"

read -n 1

sudo gdb ../build/srsenb/src/srsenb 
