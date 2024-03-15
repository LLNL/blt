#!/bin/bash

BLT_DIR=`pwd`
RUNNER_SCRIPT=`pwd`/tests/projects/run_tests.py
HOST_CONFIG_PATH=`pwd`/host-configs/llnl/$SYS_TYPE/$HOST_CONFIG

cd `pwd`/tests/projects/
python3 $RUNNER_SCRIPT --run-test $ENABLED_BLT_TESTS --host-config $HOST_CONFIG_PATH --verbose --clean --blt-source-dir $BLT_DIR
