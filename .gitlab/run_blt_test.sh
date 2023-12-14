#!/bin/bash

BLT_DIR=`pwd`
RUNNER_SCRIPT=`pwd`/blt_test/run_tests.py
cd `pwd`/blt_test/

if [[ -z $HOST_CONFIG  ]]; then
    HOST_CONFIG=`pwd`/blt_test/$HOST_CONFIG
    python3 $RUNNER_SCRIPT --run-test $ENABLED_BLT_TESTS --verbose --clean --blt-source-dir $BLT_DIR
else
    python3 $RUNNER_SCRIPT --run-test $ENABLED_BLT_TESTS --host-config $HOST_CONFIG --verbose --clean --blt-source-dir $BLT_DIR
fi
