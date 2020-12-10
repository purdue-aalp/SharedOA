# SharedOA

## Software Prerequisite
* Ubuntu Linux
* CUDA 10.1 (Not sure whether works with other versions)

## Description
Shared Object Allocator(SharedOA) is a type-based memory
allocator that allows objects to make use of inheritance
and virtual functions to be shared between the CPU and GPU.

## Allocator Interface

```
my_obj_alloc.my_new<Type1>()
return a pointer to an object of type Type1 , that can be shared between the CPU and GPU
```
```
my_obj_alloc.toDevice()
patches the vptr of all objects allocated by SharedOA to be used on Device
```

```
my_obj_alloc.toHost()
patches the vptr of all objects allocated by SharedOA to be used on Host
```
## Usage Example
we first alloacte a memory chuck that is shared between CPU and GPU 
```
  mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);


```
