# Audio equalizer with SIMD & CUDA
This repo contains the project of Conti Tomas and Chergui Jacopo: an audio equalizer that takes profit from the concepts and tools of parallel computing in SIMD and CUDA.

## Content of this folder ##
- **scalarLPF.cpp**: a program that takes a .wav file and applies a Low-Pass Filter, the result is stored in a new file.
- **lpf.cpp**: a Low-Pass Filter implemented in scalar mode (like scalarLPF.cpp) and also in parallel with SSE2 intrinsics. It creates 3 log files, two with the clock cycles for the scalar and parallel computation and one for the speedup.
- **hpf.cpp**: High-Pass Filter implemented like lpf.cpp.
- **bpf.cpp**: Band-Pass Filter implemented like lpf.cpp and hpf.cpp. 
- **samples/**: this folder contains sounds used in tests. The format is with a sample rate of 44100 Hz and a bit depth of 16, which means that every second we have 44100 little pieces of audio, and every piece is represented by 16 bits. All of these sounds are produced by Chergui Jacopo.
- matlab: matlab code used for graphs.
