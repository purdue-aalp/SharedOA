# SharedOA

## Software prerequisite
* Ubuntu Linux
* CUDA 10.1 (Not sure whether works with other versions)

## Description
Shared Object Allocator (SharedOA) is a type-based memory
allocator that allows objects to make use of inheritance
and virtual functions to be shared between the CPU and GPU.

We include an example to show how to use SharedOA to create objects.
The example contains two classes (S1 and S2), which they both have virtual functions. we create an array of objects of type (S2), and then call the virtual functions on both the CPU and GPU, we call the inc function twice (on the device ) to increamte the var by 2. while on the host we call dec twice to decrement the var by 2. At the end of the execution all the objects should have 0 in the var.

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
Objects Creation Done
Host Call Done
Device Call Done
Host Call Done
Device Call Done
ptr[0].var = 0
ptr[1].var = 0
...
ptr[511].var = 0 
```
Here SharedOA shows the indces of each of the type S2 virtual functions in the Vtable  
```
vtbale [2S2][0]:0x8
vtbale [2S2][1]:0x10
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
