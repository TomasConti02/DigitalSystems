#!/bin/bash

# Variables
SOURCE_FILE="lpf.cpp"
EXECUTABLE="lpf"
ARGS="1000 ./samples/fullSong1.wav"
LOG_FILE="log.txt"
NUM_RUNS=10

# cleans log file
> $LOG_FILE

# loop on opt. level
for OPT_LEVEL in 0 1 2 3; do
    echo "Compiling in -O$OPT_LEVEL..." | tee -a $LOG_FILE

    # compiling
    g++ -o $EXECUTABLE $SOURCE_FILE -O$OPT_LEVEL -lsndfile -lfftw3 -lm

    total=0.0

    # executes NUM_RUNS times
    for ((i = 1; i <= NUM_RUNS; i++)); do
        echo -e "\nExecution #$i of $EXECUTABLE with -O$OPT_LEVEL:" >> $LOG_FILE
        output=$(./$EXECUTABLE $ARGS)  # captures output

        # Log 
        echo "Output: $output" >> $LOG_FILE

        # sum
        total=$(echo "$total + $output" | bc)
    done

    # compute average
    avg=$(echo "scale=3; $total / $NUM_RUNS" | bc)

    # Log avg
    echo -e "Average for -O$OPT_LEVEL: $avg" | tee -a $LOG_FILE
    echo -e "\n----------------------------------------\n" | tee -a $LOG_FILE
done

echo "[OK] Results saved in $LOG_FILE"
