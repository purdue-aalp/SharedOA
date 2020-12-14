# SharedOA, COAL and TypePointer

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
vtable [2S2][0]:0x8
vtable [2S2][1]:0x10
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

## Compile and run example for COAL and TypePointer
to use with coal 
export TRANSFORM_SCRIPT= point them to https://github.com/brad-mengchi/asplos_2021_ae/tree/master/util/ptx_transform_coal
for tp 
export TRANSFORM_SCRIPT= point them tohttps://github.com/brad-mengchi/asplos_2021_ae/tree/master/util/ptx_transform_tp


```
# Generate dryrun file
get the commands used by nvcc to compile the code , we focus only on commands that cpmpile the ptx beacause we want to hack it
	nvcc --dryrun --keep $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC)  $(INC) -o $(EXECUTABLE) $(LIBS) 2> dryrun.sh 
	# Remove all lines before/including cicc
	sed -i '1,/cicc/d' dryrun.sh
	sed -i '/cicc/d' dryrun.sh
	# Remove rm line
	sed -i '/rm/d' dryrun.sh
	# Remove leading comment
	cut -c 3- dryrun.sh > dryrun1.sh
	mv dryrun1.sh dryrun.sh
  
  // now we generte the ptx using nvcc with --keep 
	nvcc  --keep   $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC)  $(INC) -o $(EXECUTABLE) $(LIBS)
  // we use our ptx tool to hack the vfun calls in the ptx
	$(PTX_GEN)/generator.py main.ptx
	cp main.ptx_coal main.ptx
  we use dryrun to recompile the script after hacking
	sh dryrun.sh
	rm -f *cpp* *fatbin* *cudafe*  *cubin* *.o *.module_id *dlink*
  
  ```
## Explain how to apply COAL with SharedOA
to apply COAL, we need to define these manged variables 
```
__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size;
__managed__ void *temp_coal;
```
after we done with obj creation 
we ask the sharedoa to create the vtable tree and provide pointers to the vtable tree and the tree size
```
  my_obj_alloc.create_tree();
  // we get a pointer to the tree
  range_tree = my_obj_alloc.get_range_tree();
  // we get the size of the tree
  tree_size = my_obj_alloc.get_tree_size();
  ```
now, for every vfun that we need to insert this code before the call
```
vtable = get_vfunc(ptr, range_tree, tree_size);  temp_coal = vtable[0]; // 0 here means the first vfun
```
to make our life easier and cleaner ,  we suggest that you define macros for each vfun 
similer to this one 
```
#define COAL_S1_inc(ptr){   vtable = get_vfunc(ptr, range_tree, tree_size);  temp_coal = vtable[0]; }
```
the indices of the vfun follows the same order that they are defined in the base class. we could also check the ptx to verify 
```
.global .align 8 .u64 _ZTV2S2[4] = {0, 0, _ZN2S23incEv, _ZN2S23decEv};

```
we ignore the fisrt two zeros , we start counting after that 
so 
_ZN2S23incEv will have index 0 , and _ZN2S23decEv will have index 1
now , inside each kerenl , we need to define this variable 
```
  void **vtable;
 ``` 
  example 
  ```cpp
__global__ void kernel(S1 **ptr) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // this variable must be defined in every kerenl that uses COAL
  void **vtable;
  if (tid < NUM_OBJ) {
    COAL_S1_inc(ptr[tid]); before the call to inc() , we need to insert the code or just use the macro
    ptr[tid]->inc(); 
  }
}
```

## Explain how to apply TypePointer with SharedOA
