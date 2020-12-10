# SharedOA

## Software Prerequisite
* Ubuntu Linux
* CUDA 10.1 (Not sure whether works with other versions)

## Description
Shared Object Allocator(SharedOA) is a type-based memory
allocator that allows objects to make use of inheritance
and virtual functions to be shared between the CPU and GPU.

## Allocator Interface

## Usage Example
we first alloacte a memory chuck that is shared between CPU and GPU 
```
  mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);


```
