# SIMD-Accelerated Array Sum

## Overview
This C++ code performs the summation of a large array of 8-bit unsigned integers, and compares the performance of a scalar implementation against a SIMD-accelerated implementation. 

The problem being solved is the efficient computation of the sum of a large number of small integer values, which is a common operation in many data processing and scientific computing tasks.

## SIMD Optimization
The SIMD (Single Instruction, Multiple Data) implementation leverages the CPU's vector processing capabilities to perform multiple 8-bit additions simultaneously, thereby improving the overall computation speed.

The key steps of the SIMD implementation are:

1. Load 16 bytes of data from the input array into a 128-bit SIMD register.
2. Unpack the 8-bit values into two 16-bit vectors (low and high bytes). 
3. Accumulate the 16-bit values in two separate 32-bit accumulators.
4. Combine the two 32-bit accumulators into a single 64-bit result.

This approach allows the computation to take advantage of the CPU's SIMD instructions, which can process multiple data elements in parallel, leading to a significant performance improvement over a scalar implementation.

## Test Script
A test script is provided that runs the scalar and SIMD-accelerated versions of the array sum function with different compiler optimization levels (O0, O1, O2, and O3). The script reports the execution time and the speedup achieved by the SIMD implementation for each optimization level.

To run the test script, you'll need to have a C++ compiler (e.g., GCC or Clang) installed on your system. You can then execute the script using the following command:
```./test_script.sh```

The output of the script will show the execution times and speedup factors for the different optimization levels, allowing you to evaluate the effectiveness of the SIMD optimization under various conditions.

## Conclusion
This code demonstrates how SIMD-based optimizations can significantly improve the performance of array summation operations, which are commonly found in many computational tasks. The provided test script allows for easy evaluation of the performance improvements across different compiler optimization levels.