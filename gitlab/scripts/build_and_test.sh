#!/bin/bash
set -x
rm -rf build_*
rm -rf install_*

BBD=`pwd`
suffix=".cmake"
module=""
host=""
alloc=""
display="true"

# Iterate through host-configs: ./host-configs/llnl/SYS_TYPE/<compiler>.cmake
for path in ./host-configs/llnl/*; do
 
  [ -d "${path}" ] || continue # if not a directory, skip
  sys="$(basename "${path}")"
  # echo -e "--\nSys: $sys"

  if [[ "$sys" == "toss_3_x86_64_ib" ]]; then
    alloc="salloc -N1 --exclusive -p pdebug --mpibind=off"
    if [[ "$LC_ZONE" == "CZ" ]]; then
      host=quartz
    else
      echo "LC Zone not recognized"
      exit 5
    fi
  elif [[ "$sys" == "blueos_3_ppc64le_ib" ]]; then
    alloc="lalloc 1 -W 60 -G guests"
    display="setenv DISPLAY ':0.0'"
    if [[ "$LC_ZONE" == "CZ" ]]; then
      host=ray
    else
      echo "LC Zone not recognized"
      exit 5
    fi
  elif [[ "$sys" == "blueos_3_ppc64le_ib_p9" ]]; then
    alloc="lalloc 1 -W 60 -G guests"
    display="setenv DISPLAY ':0.0'"
    if [[ "$LC_ZONE" == "CZ" ]]; then
      host=lassen
    else
      echo "LC Zone not recognized"
      exit 5
    fi
  elif [[ "$sys" == "windows" ]]; then
    echo "Skipping windows"
    continue
  else
    echo "OS not supported for this CI"
    exit 6
  fi
  
  echo "[Testing host-configs for $sys]"

  for file in ./host-configs/llnl/$sys/*; do
    hc_path="$BBD/$file"
    # echo "  File: $hc_path"
    [ -f "${hc_path}" ] || continue # if not a file, skip
    
    hc="$(basename "${file}")"
    compiler=${hc%$suffix}
    module="module load cmake/3.9.2"
    # override module for CUDA C++17 support
    if [[ $hc =~ "nvcc_c++17" ]];
    then
        module="module load cmake/3.18.0 cuda/11.0.2"
    fi

    echo "[Testing $compiler on $sys with $host]"
    echo "[host-config =  $hc]"
    echo "[Module = $module]"
    echo "[Path = $path]"
    echo "[alloc = $alloc]"

    mkdir build_$compiler
    mkdir install_$compiler    
    cd build_$compiler
    CUR_DIR=`pwd`

    ssh $host "cd $CUR_DIR && $module && $display && \
      $alloc cmake -C $hc_path -D CMAKE_INSTALL_PREFIX=../install_$compiler ../tests/internal && \
      $alloc make && \
      $alloc ctest -DCTEST_OUTPUT_ON_FAILURE=1 --no-compress-output -T Test -VV -j8 && \
      xsltproc -o junit.xml ../tests/ctest-to-junit.xsl Testing/*/Test.xml"
    
    cd $BBD
    echo "--"
  done
done
