/* Refactor 01-hello-gpu.cu so that Hello from the GPU prints twice, once before Hello from the CPU, and once after. */

#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

__global__ void helloGPU()
{
  printf("Hello from the GPU!\n");
}

int main()
{
  // First GPU kernel launch
  helloGPU<<<1,1>>>();
  
  // Wait for all GPU tasks to complete before continuing.
  cudaDeviceSynchronize();

  // Now call the CPU function
  helloCPU();

  // Second GPU kernel launch
  helloGPU<<<1,1>>>();
  
  // Wait for all GPU tasks to complete before exiting.
  cudaDeviceSynchronize();

  return 0;
}
