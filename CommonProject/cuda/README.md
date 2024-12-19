## GPU Datasheet ##
we use Tesla T4 for testing our kernels.

## What kind of memory on and off chip we are using in our kernel ? ##
## GPU Memory Types
### **DRAM**  
Slowest memory off-chip, shared by every thread of GPU execution. We can call it "Local Memory". It has the same position as global memory. It has low bandwidth and high latency, it is private for each thread and not shared. Normally used for variables that can't fit in registers due to space limitations.  
It is useful for large structures and local array data.

### **REGISTERS**  
Fastest memory on-chip, private for each thread, and used for temporal variables. Maximum bandwidth and minimum latency, with 255 registers per 32-bit. Normally, variables without type qualifiers are private for each thread.  
Register splitting occurs if exceeding hardware limits, where they can be automatically moved from register memory to local memory.

### **SHARED**  
Fast memory, shared by every thread of a block.  
Each SM has an on-chip memory limitation of 48-228 KB, shared between Shared memory and L1 Cache. It is organized inside memory banks and can be problematic if each thread needs the same bank memory, as in this case the access is serialized (very bad).  
After the initialization of shared memory, we need a `__syncthreads` command to sync every thread of the block before they try to access the data (bad for seed).
We use the __shared__ qualifier.

### **CONSTANT**  
Read-only memory off-chip, but a limited portion is on-chip for every SM, useful for high-read frequency data. Data can't change during execution.  
Total size is limited to 64 KB, and the memory space is accessible by all threads in a kernel.

### **Global Memory**  
The largest and highest-latency GPU memory. It hasa global scope, accessible by all threads across all SMs within GPU.
Declared with the `__device__` qualifier, it can be allocated on the host using `cudaMalloc` and freed with `cudaFree`.
Remains active for the entire execution of the application on the GPU.  
It is very roomy but also slow, for this reason we have to initialize it by only transfer, exploiting the but PCle

### **GPU Cache: Structure and Operation**  
GPU caches are on-chip, non-programmable memory structures designed for fast data access.  

#### **L1 Cache**  
- The fastest cache, with one instance per SM.  
- Ensures fast data access by storing data from both local and global memory.  
- Includes data that doesn't fit in registers (register spills).  

#### **L2 Cache**  
- A single cache shared among all SMs.  
- Acts as a bridge between the faster L1 caches and the slower global memory.  
- Stores data from both local and global memory, including register spills.  
- Non-programmable.  

#### **Constant Cache (Read-Only, Per SM)**  
- Present in each SM.  
- Optimized for quick access to immutable data, such as lookup tables or constant parameters.  
- Stores data that doesn't change during kernel execution.  
---
---  
