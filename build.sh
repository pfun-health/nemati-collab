#!/bin/sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "===Running build script===\n\n"
for folder in $(ls pycardio | grep c_)
do
    echo "Building ${folder}\n"
    cd "./pycardio/${folder}"
    make
    echo "${folder} done.\n"
    echo "==================\n"
    cd $DIR
done
