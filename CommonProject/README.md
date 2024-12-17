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
**DRAM**  
Slowest memory off-chip, shared by every thread of GPU execution. We can call it "Local Memory". It has the same position as global memory. It has low bandwidth and high latency, it is private for each thread and not shared. Normally used for variables that can't fit in registers due to space limitations.  
It is useful for large structures and local array data.

**REGISTERS**  
Fastest memory on-chip, private for each thread, and used for temporal variables. Maximum bandwidth and minimum latency, with 255 registers per 32-bit. Normally, variables without type qualifiers are private for each thread.  
Register splitting occurs if exceeding hardware limits, where they can be automatically moved from register memory to local memory.

**SHARED**  
Fast memory, shared by every thread of a block.  
Each SM has an on-chip memory limitation of 48-228 KB, shared between Shared memory and L1 Cache. It is organized inside memory banks and can be problematic if each thread needs the same bank memory, as in this case the access is serialized (very bad).  
After the initialization of shared memory, we need a `__syncthreads` command to sync every thread of the block before they try to access the data (bad for seed).

**CONSTANT**  
Read-only memory off-chip, but a limited portion is on-chip for every SM, useful for high-read frequency data. Data can't change during execution.  
Total size is limited to 64 KB, and the memory space is accessible by all threads in a kernel.

**Global Memory**
