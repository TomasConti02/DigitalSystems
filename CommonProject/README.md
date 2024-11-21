# Audio equalizer with SIMD & CUDA
This repo contains the project of Conti Tomas and Chergui Jacopo: an audio equalizer that takes profit from the concepts and tools of parallel computing in SIMD and CUDA.

## Content of this folder ##
### Source code: ###
- **scalarEQ.cpp**: a program that takes a .wav file and applies a pre-defined equalization, the result is stored in a new file.
- **parallelEQ.cpp** : an equalizer in SIMD with double precision numbers.
- **parallelEQFloat.cpp** : an equalizer in SIMD with float precision numbers.
- **iir.cpp**: this is an equalizer implemented with an IIR filter. This file does not need the Fourier Transform and is much faster.
- **filters/**: low-pass, high-pass and band-pass filters in c++.
### Other: ###
- **samples/**: this folder contains sounds used in tests. The format is with a sample rate of 44100 Hz and a bit depth of 16, which means that every second we have 44100 little pieces of audio, and every piece is represented by 16 bits. All of these sounds are produced by Chergui Jacopo and free to use.
- **matlab/**: matlab folder with code used for graphs.
- **test/**: this folder contains some scripts to test the code in different optimization levels.

## How to run? ##
At the moment, only `iir.cpp` does not need the following libraries.
There are two libraries required:
- libfftw3 for the Fast Fourier Transform: we have to treat frequencies, so for every little frame we need to know its spectrum, to obtain frequencies we tranform every piece of audio in a complex number. ***Warning***: to deal with float numbers instead of double, use `-lfftw3f` when compiling.
- libsndfile1-dev for treating audio files.

To install these libraires:
``` 
  sudo apt install libsndfile1-dev
```

``` 
  sudo apt install libfftw3
```

To compile, you can choose different levels of optimization, in order: {`-O0`, `-O1`, `-O2`, `-O3`}. We also need to specify with `-msse2` to compile with optimization for SIMD. We also specify the two libraires installed previously (example with the bpf):
``` 
  g++ -o bpf bpf.cpp -lfftw3 -lsndfile -msse2 -O3
```

To run, we just specify the cutoff frequency as parameter (only for the bpf we specify two numbers, the lower freq. and the upper one):
``` 
./bpf 300 3000
```
