/*Exercise: Array Manipulation on both the Host and Device
The 01-double-elements.cu program allocates an array, initializes it with integer values on the host, 
attempts to double each of these values in parallel on the GPU, 
and then confirms whether or not the doubling operations were successful, on the host. 

Currently the program will not work: 
it is attempting to interact on both the host and the device with an array at pointer a, 
but has only allocated the array (using malloc) to be accessible on the host. 
Refactor the application to meet the following conditions

1. a should be available to both host and device code.
2. The memory at a should be correctly freed.

*/

#include <stdio.h>

/*
 * Initialize array values on the host.
 */

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

/*
 * Double elements in parallel on the GPU.
 */

__global__
void doubleElements(int *a, int N)
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    a[i] *= 2;
  }
}

/*
 * Check all elements have been doubled on the host.
 */

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  int N = 100;
  int *a;

  size_t size = N * sizeof(int);

  /*
   * Refactor this memory allocation to provide a pointer
   * `a` that can be used on both the host and the device.
   */

  a = (int *)malloc(size);

  init(a, N);

  size_t threads_per_block = 10;
  size_t number_of_blocks = 10;

  /*
   * This launch will not work until the pointer `a` is also
   * available to the device.
   */

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  /*
   * Refactor to free memory that has been allocated to be
   * accessed by both the host and the device.
   */

  free(a);
}
