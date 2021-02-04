#!/bin/bash

set -e

rm -rf build
mkdir build
cd build

cmake -C $HOST_CONFIG ../tests/internal
make -j8
ctest -DCTEST_OUTPUT_ON_FAILURE=1 --no-compress-output -T Test -VV -j8
xsltproc -o junit.xml ../tests/ctest-to-junit.xsl Testing/*/Test.xml"
