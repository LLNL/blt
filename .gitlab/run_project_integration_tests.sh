#!/bin/bash

BLT_DIR=`pwd`
RUNNER_SCRIPT=`pwd`/tests/projects/run_tests.py
cd `pwd`/tests/projects/

if [[ -z $HOST_CONFIG  ]]; then
    HOST_CONFIG=`pwd`/host-configs/llnl/$SYS_TYPE/$HOST_CONFIG
    python3 $RUNNER_SCRIPT --run-test $ENABLED_BLT_TESTS --verbose --clean --blt-source-dir $BLT_DIR
else
    python3 $RUNNER_SCRIPT --run-test $ENABLED_BLT_TESTS --host-config $HOST_CONFIG --verbose --clean --blt-source-dir $BLT_DIR
fi
