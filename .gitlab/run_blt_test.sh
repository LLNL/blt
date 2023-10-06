#!/bin/bash

BLT_DIR=`pwd`
HOST_CONFIG=`pwd`/blt_test/$HOST_CONFIG

python3 run_tests.py --run-test $ENABLED_BLT_TESTS --host-config $HOST_CONFIG --verbose --clean --blt-source-dir $BLT_DIR