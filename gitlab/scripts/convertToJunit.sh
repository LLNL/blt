#!/bin/bash
set -x

BBD=`pwd`
resultsfilename="CTestCostData.txt"
resultslogfilename="LastTest.log"
resultspath="Testing/Temporary/"

#Cleanup from previous run
rm -rf Artifacts
mkdir Artifacts
rm -f *.xml

xmlfilename="testresults.xml"
echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" >> $xmlfilename
echo "<testsuites>" >> $xmlfilename

for dir in *; do
    echo $dir
    if [[ -d "${dir}" ]] && [[ "$dir" == *"build_"* ]]; then
        testsuitename=$(echo $dir | tr '@.' '_')
#        echo $testsuitename

        echo "  <testsuite name=\"$testsuitename\">" >> $xmlfilename
        if [ ! -f $dir/$resultspath$resultsfilename ]; then
            echo "File $dir/$resultspath$resultsfilename not found!"
            echo "    <testcase time=\"\" classname=\"$testsuitename\" name=\"Results File Missing\">" >> $xmlfilename
            echo "      <failure type=\"\"> No Test Results Found </failure>" >> $xmlfilename
            echo "    </testcase>" >> $xmlfilename
        else
            cp $dir/$resultspath$resultsfilename $BBD/Artifacts/$testsuitename-$resultsfilename
            cp $dir/$resultspath$resultslogfilename $BBD/Artifacts/$testsuitename-$resultslogfilename

            while read -r line; do
                result="$line"

                IFS=' ' read -r -a array <<< "$result"
                echo "Line read from file - $result"

                if [ ! -z "${array[0]}" ] && [ ! -z "${array[1]}" ]; then
                    if [ "${array[1]}" = '1' ]; then
                        echo "Test Passed"
                        echo "    <testcase status=\"run\" time=\"${array[2]}\" classname=\"$testsuitename\" name=\"${array[0]}\"/>" >> $xmlfilename
                    else
                        echo "Test Failed"
                        echo "    <testcase time=\"${array[2]}\" classname=\"$testsuitename\" name=\"${array[0]}\">" >> $xmlfilename
                        echo "      <failure type=\"$testsuitename\"> Not Passed </failure>" >> $xmlfilename
                        echo "    </testcase>" >> $xmlfilename
                    fi
                fi
            done < "$dir/$resultspath$resultsfilename"
        fi
        echo "  </testsuite>" >> $xmlfilename
    fi
done
echo "</testsuites>" >> $xmlfilename
