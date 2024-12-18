#!/bin/bash

# for iir.cpp -> ./test.sh iir.cpp
# for the others -> ./test.sh <filename>.cpp -lsndfile -fopenmp -lfftw3f

if [ -z "$1" ]; then
    echo "You must specify the name of the C++ file."
    echo "Usage: $0 <src_file.cpp> [additional_compile_options]"
    exit 1
fi

# vars
SOURCE_FILE="$1"
EXECUTABLE="${SOURCE_FILE%.cpp}"
shift 
COMPILE_OPTIONS="$@" # compile options
cd ..
LOG_FILE="./test/${EXECUTABLE}_runs.txt"
NUM_RUNS=20

# cleans log file
> $LOG_FILE

# loops for each opt level
for OPT_LEVEL in 0 1 2 3; do
    echo "Compiling in -O$OPT_LEVEL..." | tee -a $LOG_FILE

    # compiling
    g++ -o $EXECUTABLE $SOURCE_FILE -O$OPT_LEVEL -march=native -msse2 $COMPILE_OPTIONS

    # execution
    for ((i = 1; i <= NUM_RUNS; i++)); do
        ./$EXECUTABLE >> $LOG_FILE
    done

    echo -e "\n----------------------------------------\n" | tee -a $LOG_FILE
done

echo "[OK] Results of runs saved in $LOG_FILE"

cd ./test

FILE_NAME="${LOG_FILE#./test/}"

python3 speedup_analysis.py "$FILE_NAME" >> $FILE_NAME
