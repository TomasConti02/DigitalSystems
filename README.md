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
## CUDA
CUDA A framework for developing SIMD (Single Instruction, Multiple Data) code on NVIDIA GPUs.
We can have 1D, 2D and 3D block grid anche block.
```
Host (CPU):
|----------------------------------|
|            Host Code             |
|----------------------------------|
          ^
          | Data transfert by BUS. We need a less number of this operations
          v
Device (GPU):
|----------------------------------------------------------|
|                       GPU (Device)                       |
|----------------------------------------------------------|
          |                Grid of Blocks                  |
          |------------------------------------------------|
          |   Block 0   |   Block 1   |   Block 2   | ...   |
          |------------------------------------------------|
              |             |             |
              v             v             v
          |--------|   |--------|   |--------|
          | Thread |   | Thread |   | Thread |
          | 0      |   | 0      |   | 0      |
          | Thread |   | Thread |   | Thread |
          | 1      |   | 1      |   | 1      |
          | ...    |   | ...    |   | ...    |
          | Thread |   | Thread |   | Thread |
          | n      |   | n      |   | n      |
          |--------|   |--------|   |--------|
Each block contains multiple threads, and the GPU executes threads in parallel.
```
##CNNs Tumor Detections
```
Model: "sequential_5"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_25 (Conv2D)                   │ (None, 128, 128, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_15 (MaxPooling2D)      │ (None, 64, 64, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_26 (Conv2D)                   │ (None, 64, 64, 128)         │          36,992 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_27 (Conv2D)                   │ (None, 64, 64, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_16 (MaxPooling2D)      │ (None, 32, 32, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_28 (Conv2D)                   │ (None, 32, 32, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_29 (Conv2D)                   │ (None, 32, 32, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_17 (MaxPooling2D)      │ (None, 16, 16, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_5 (Flatten)                  │ (None, 32768)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_12 (Dense)                     │ (None, 128)                 │       4,194,432 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_13 (Dense)                     │ (None, 512)                 │          66,048 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_14 (Dense)                     │ (None, 2)                   │           1,026 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

 Total params: 4,742,146 (18.09 MB)

 Trainable params: 4,742,146 (18.09 MB)

 Non-trainable params: 0 (0.00 B)
```
