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
First, alloacte a memory chuck that is shared between CPU and GPU 
```
  mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
```
Second, we instaitae an SharedOA object and we pass the the shared memory obj and we also pass the intial chunk size that is used by SharedOA
```
    obj_alloc my_obj_alloc(&shared_mem, 1024 * 1024ULL);
```

Third, we use the sharedOA to allocate an array of pointers using calloc .. note that we use calloc since we are allocating pointers not object 
```
  S1 **ptr = (S1 **)my_obj_alloc.calloc<S1 *>(NUM_OBJ);
```
Then, we start instantiating objects using SharedOA ( my_new<Type> )
```
    for (int i = 0; i < NUM_OBJ; i++)
    ptr[i] = (S1 *)my_obj_alloc.my_new<S2>();
```
At this point , the virtual functions can only be called from the host side
To make the virtual function accessable from the device we call toDevice()
  ```
    my_obj_alloc.toDevice();
  ```
To regain the CPU accessabilty to the virtual functions , we call toHost()
    ```
    my_obj_alloc.toHost();
  ```
