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
**DRAM/**
slowest memory offchip, shared by every thread of GPU execution. We can call it "Local Memory". It have the same position of global memory. It have low bandwidth and high latency, it is private for thread and not shared. Normally used for variables that can't fit in registers due to space limitations.
It usefull for large structure and local array data.
**REGISTERS/**
fastest memory on-chip, it is private for every thread and they use it for temporal variables. Maximum bandwidth and minimum latency, we have 255 registr by 32bit. Normally variable without type qualifiers and provate for every thread.
we can have registrer splitting if exeeding hardware limite, they can be automatically move from registry memory to local memory.
**SHARED/**
fast memory, shared by every thread of a bock.
Every SM have a limitation onchip memory of 48-228 KB, it shared between Shared memory and Cache L1. It is organise inside memory bancks and can be a problem is each thread need the same back memory, in this case the access is serialize(very bad).
After the initialization of shared memmory we need a __syncthreads command con sync every thread of the block befour they try to access the data (bad for seed)
**CONSTANT/**
Read only memory off chipp but a limited portion is onchip for every SM, useful for high read frequency data. Data can't change during the execution.
Total size limited to 64 KB and memory space accessible by all threads in a kernel.
**Global Memory/**
