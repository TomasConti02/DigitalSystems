# Audio equalizer with SIMD & CUDA
This repo contains the project of Conti Tomas and Chergui Jacopo: an audio equalizer that takes profit from the concepts and tools of parallel computing in SIMD and CUDA.

## Content of this folder ##
### Source code: ###
- **scalarLPF.cpp**: a program that takes a .wav file and applies a Low-Pass Filter, the result is stored in a new file.
- **scalarEQ.cpp**: a program that takes a .wav file and applies a pre-defined equalization, the result is stored in a new file.
- **lpf.cpp**: a Low-Pass Filter implemented in scalar mode (like scalarLPF.cpp) and also in parallel with SSE2 intrinsics. It creates 3 log files, two with the clock cycles for the scalar and parallel computation and one for the speedup.
- **hpf.cpp**: High-Pass Filter implemented like lpf.cpp.
- **bpf.cpp**: Band-Pass Filter implemented like lpf.cpp and hpf.cpp.
### Other: ###
- **samples/**: this folder contains sounds used in tests. The format is with a sample rate of 44100 Hz and a bit depth of 16, which means that every second we have 44100 little pieces of audio, and every piece is represented by 16 bits. All of these sounds are produced by Chergui Jacopo.
- **matlab/**: matlab folder with code used for graphs.
- **test/**: this folder contains some scripts to test the code.

## How to run? ##
There are two libraries required:
- libfftw3 for the Fast Fourier Transform: we have to treat frequencies, so for every little frame we need to know its spectrum, to obtain frequencies we tranform every piece of audio in a complex number.
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
