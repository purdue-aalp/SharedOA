
#include "mem_alloc.h"
#include "COAL.h"
#define NUM_OBJ 512
class S1 {
public:
  int var;
  __host__ __device__ S1() {}
  virtual __host__ __device__ void inc() = 0;
  virtual __host__ __device__ void dec() = 0;
};

class S2 : public S1 {
public:
  __host__ __device__ S2() {}
  __host__ __device__ void inc() { this->var += 2; }

  __host__ __device__ void dec() { this->var -= 2; }
};

__global__ void kernel(S1 **ptr) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // this variable must be defined in every kerenl that uses COAL
  void **vtable;
  if (tid < NUM_OBJ) {
    COAL_S1_inc(ptr[tid]);
    ptr[tid]->inc();
  }
}

int main() {

  mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
  obj_alloc my_obj_alloc(&shared_mem, 1024 * 1024ULL);
  S1 **ptr = (S1 **)my_obj_alloc.calloc<S1 *>(NUM_OBJ);
  for (int i = 0; i < NUM_OBJ; i++)
    ptr[i] = (S1 *)my_obj_alloc.my_new<S2>();

  printf("Objects Creation Done\n");

  // after we get done with creating the objects
  // we ask the SharedOA to create the vfun Table as a binary tree
  my_obj_alloc.create_tree();
  // we get a pointer to the tree
  range_tree = my_obj_alloc.get_range_tree();
  // we get the size of the tree
  tree_size = my_obj_alloc.get_tree_size();

  // to access the virtual function from device we call toDevice
  my_obj_alloc.toDevice();
  int blockSize = 256;
  int numBlocks = (NUM_OBJ + blockSize - 1) / blockSize;
  kernel<<<numBlocks, blockSize>>>(ptr);
  cudaDeviceSynchronize();
  printf("Device Call Done\n");



  for (int i = 0; i < NUM_OBJ; i++)
    printf("ptr[%d].var = %d \n", i, ptr[i]->var);

  return 0;
}