#include "mem_alloc.h"
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
  if (tid < NUM_OBJ)
    ptr[tid]->inc();
}

int main() {

  mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
  obj_alloc my_obj_alloc(&shared_mem, 1024 * 1024ULL);
  S1 **ptr = (S1 **)my_obj_alloc.calloc<S1 *>(NUM_OBJ);
  for (int i = 0; i < NUM_OBJ; i++)
    ptr[i] = (S1 *)my_obj_alloc.my_new<S2>();

  printf("Objects Creation Done\n");
  // virtual function is now only accessable from the host
  for (int i = 0; i < NUM_OBJ; i++)
    ptr[i]->dec();
  printf("Host Call Done\n");

  // to access the virtual function from device we call toDevice
  my_obj_alloc.toDevice();
  int blockSize = 256;
  int numBlocks = (NUM_OBJ + blockSize - 1) / blockSize;
  kernel<<<numBlocks, blockSize>>>(ptr);
  cudaDeviceSynchronize();
  printf("Device Call Done\n");

  // virtual function is now only accessable from the device
  // to access the virtual function from host we call toHost
  my_obj_alloc.toHost();
  for (int i = 0; i < NUM_OBJ; i++)
    ptr[i]->dec();
  printf("Host Call Done\n");



  my_obj_alloc.toDevice();

  kernel<<<numBlocks, blockSize>>>(ptr);
  cudaDeviceSynchronize();
  printf("Device Call Done\n");

  for (int i = 0; i < NUM_OBJ; i++)
    printf("ptr[%d].var = %d \n", i, ptr[i]->var);

  return 0;
}