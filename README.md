We store here our CUDA codes.
tomas conti.
jacopo chergui.
## SIMD
We use the SSE (Streaming SIMD Extensions) instruction set architecture for x86 processors to enable SIMD (Single Instruction, Multiple Data) operations. 
SSE provides 128-bit registers, which allow for parallel processing of multiple data elements, such as four 32-bit floating-point numbers.
When creating our data structures, we need to ensure that their memory addresses are aligned to 16 bytes, matching the register size. 
This alignment allows us to read an entire register's worth of data in a single CPU cycle.
by chatgpt
```
Memory Layout (Each block is 16 bytes):
|--------------|--------------|--------------|--------------|
| Address 0    | Address 16   | Address 32   | Address 48   |
|--------------|--------------|--------------|--------------|
| Data1: 16B   | Data2: 16B   | Data3: 16B   | Data4: 16B   |
|--------------|--------------|--------------|--------------|
                   ^
                   |
    Data aligned at Address 16
128-bit Register (16 bytes):
|---------------------------------------------------------|
|                  Data2: 16B                             |
|---------------------------------------------------------|
In a single CPU cycle, the 128-bit register reads 16 bytes starting from Address 16.
```
