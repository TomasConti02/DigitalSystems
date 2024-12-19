## GPU Datasheet ##
we use Tesla T4 for testing our kernels.
DRAM bandwidth -> 320GB/s
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
### Host-Device Memory Management
#### Data transfers**  
Data transfers between the host and the device can represent a bottleneck because the memory bandwidth between GPU memory and GPU cores is significantly 
higher compared to the PCIe bandwidth(bredge between CPU and GPU).
To address this, it is important to minimize data transfers between the host and the device. Ideally, all data should be transferred to GPU memory only once.
#### Pinned Memory in CUDA**  
When managing large amounts of data to send to the device, it is recommended to consider using pinned memory.
By default, host-allocated memory is pageable (subject to page faults). Before transferring the data, it needs to be moved into pinned memory (not subject to page faults), and only then can the transfer take place.
In our case, we use pinned memory when we neeed trasfert large amounts of data, the data resulting from the FFT transformation.
Pinned memory is more expensive to allocate/deallocate compared to pageable memory, but accelerates large
data transfers, especially when repeatedly using the same buffer, amortizing the initial cost.
If we use too much the pinnel memory there is the risk of inefficiency use of RAM by host.
#### Zero-Copy Memory**  
It's a technique that allows the device to directly access host memory without explicitly
copying data between the two memories, an exception to the rule that the host cannot directly access device variable
and for the device the same.
Both the host and the device can access zero-copy memory, with device accesses occurring directly via PCIe.
We decided not to use it because our data transfer is relatively simple, and using zero-copy memory would require synchronization of access between the host and the device.
For small shared data, zero-copy memory simplifies programming and offers reasonable performance but, for large datasets on discrete GPUs via PCIe, zero-copy causes significant performance degradation
#### Unified Virtual Addressing (UVA)**
Unified Virtual Addressing (UVA) is a technique that allows the CPU and GPU to share the same virtual address
space (while physical memory remains distinct). There is no difference between host and device pointer, the CUDA runtime system automatically manages the mapping of virtual addresses to physical addresses.

#### Unified Memory (UM)**
In this case, we can use a unified 49-bit virtual memory space that allows all system processors to access the same data using a single pointer (single-pointer-to-data). This type of memory simplifies application code and memory management because it is automatically handled by the underlying system and is interoperable with device-specific allocations.

Why did we decide to use normal memory addressing instead?
Because we manage very simple data transfers and do not have a large or complex system with multiple CPUs and/or GPUs. Unified Memory (UM) adds latency, and we have no control over it.

---
### Global Memory Management
Optimizing global memory use is crucial for kernel performance, without this optimization the kernel performance can be very bad.
Instructions and memory operation are issued and executed per warps(32 thread). A single request is serviced by one(optimal case) or more memory trasferts.

We shoud have an Aligned memory accesses and Coalesced memory accesses.
For this reason we do a great use of index for every thread memory data access. 
