# SharedOA

## Software prerequisite
* Ubuntu Linux
* CUDA 10.1 (Not sure whether works with other versions)

## Description
Shared Object Allocator (SharedOA) is a type-based memory
allocator that allows objects to make use of inheritance
and virtual functions to be shared between the CPU and GPU.

We include an example to show how to use SharedOA to create objects.
{describe the example a little here}

## Allocator interface

Return a pointer to an object of type Type1 shared between the CPU and GPU:
```cpp
my_obj_alloc.my_new<Type1>()
```

Patches the virtual function pointers of all objects allocated by SharedOA to GPU device:
```cpp
my_obj_alloc.toDevice()
```

Patches the virtual function pointers of all objects allocated by SharedOA to the host CPU:
```cpp
my_obj_alloc.toHost()
```

## Compile and run example:

Make the example:
```bash
cd example
make
```

Run the example:
```bash
./main
```

Output would be like below:
```
vtbale [2S2][0]:0x8
vtbale [2S2][1]:0x10
total_count 1048576 , total 1048576 , type_size 16
Objects Creation Done
Host Call Done
0x7f7864002188 0x7f7864002160 1048576
Device Call Done
0x7f7864002188 0x7f7864002138 1048576
Host Call Done
ptr[0].var = -2
ptr[1].var = -2
...
ptr[1023].var = -2 
```

## Explain the use of SharedOA in the example:

The example implement the following routine:
1. Allocate the memory space for the SharedOA that is shared between CPU and GPU:
```cpp
  mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
```

2. Initialize the SharedOA with the memory space and the intial chunk size:
```cpp
    obj_alloc my_obj_alloc(&shared_mem, 1024 * 1024ULL);
```

3. Use the sharedOA to allocate an array of pointers using calloc.
Note that we use calloc since we allocate pointers instead of objects:
```cpp
S1 **ptr = (S1 **)my_obj_alloc.calloc<S1 *>(NUM_OBJ);
```

4. Start instantiating objects using SharedOA ( my_new<Type> )
```cpp
for (int i = 0; i < NUM_OBJ; i++) {
  ptr[i] = (S1 *)my_obj_alloc.my_new<S2>();
}
```

5. The virtual functions can only be called from the host side.
To make the virtual function accessable from the device we call toDevice():
```cpp
my_obj_alloc.toDevice();
```

6. To reclaim the CPU accessabilty to the virtual functions, we call toHost():
```cpp
my_obj_alloc.toHost();
```
