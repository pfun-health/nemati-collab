#!/usr/bin/env bash

# build.sh

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR="${REPO_ROOT}"

echo "===Running build script===\n\n"
for folder in $REPO_ROOT/pycardio/c_*
do
    echo "Building shared objects in: ${folder}...\n"
    cd "${folder}"
    make
    echo "done.\n"
    echo "==================\n"
    cd $DIR
done

cd 
