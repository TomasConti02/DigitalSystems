# Audio equalizer with SIMD & CUDA
This repo contains the project of Conti Tomas and Chergui Jacopo: an audio equalizer that takes profit from the concepts and tools of parallel computing in SIMD and CUDA.

## Content of this folder ##
### Source code: ###
- **simd/**: this folder contains c++ source files with various versions of the equalizer implemented with Intel SSE Intrinsics.
- **cuda/**: same for `simd/` but implemented in CUDA.
- **filters/**: some simpler examples to get started with audio processing (low-pass filter, band-pass etc.).
### Other: ###
- **samples/**: this folder contains sounds used in tests. The format is with a sample rate of 44100 Hz and a bit depth of 16, which means that every second we have 44100 little pieces of audio, and every piece is represented by 16 bits. All of these sounds are produced by Chergui Jacopo and free to use.
- **matlab/**: matlab folder with code used for graphs.
- **test/**: this folder contains some scripts to test the code in different optimization levels.
## What kind of memory on and off chip we are using in our kernel ? ##
