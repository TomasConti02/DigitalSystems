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
|   16B Data   | Int1: 4B     |   16B Data   |   16B Data   |
|--------------| Int2: 4B     |--------------|--------------|
               | Int3: 4B     |                         
               | Int4: 4B     |                           
               |--------------|
                   ^
                   |
    Data aligned at Address 16
128-bit Register (16 bytes):
|---------------------------------------------------------|
|    Int1: 4B   |   Int2: 4B   |   Int3: 4B   |   Int4: 4B   |
|---------------------------------------------------------|
In a single CPU cycle, the 128-bit register reads the 4 integers from Address 16.
```
