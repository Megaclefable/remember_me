/*Exercise: Accelerating a For Loop with a Mismatched Execution Configuration
The program in 02-mismatched-config-loop.cu allocates memory, 
using cudaMallocManaged for a 1000 element array of integers, 
and then seeks to initialize all the values of the array in parallel using a CUDA kernel. 
This program assumes that both N and the number of threads_per_block are known. 
Your task is to complete the following two objectives, refer to the solution if you get stuck:

Assign a value to number_of_blocks that will make sure there are at least as many threads as there are elements in a to work on.
Update the initializeElementsTo kernel to make sure that it does not attempt to work on data elements that are out of range.
*/

#include <stdio.h>

/*
 * Currently, `initializeElementsTo`, if executed in a thread whose
 * `i` is calculated to be greater than `N`, will try to access a value
 * outside the range of `a`.
 *
 * Refactor the kernel definition to prevent out of range accesses.
 */

__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  a[i] = initialValue;
}

int main()
{
  /*
   * Do not modify `N`.
   */

  int N = 1000;

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);

  /*
   * Assume we have reason to want the number of threads
   * fixed at `256`: do not modify `threads_per_block`.
   */

  size_t threads_per_block = 256;

  /*
   * Assign a value to `number_of_blocks` that will
   * allow for a working execution configuration given
   * the fixed values for `N` and `threads_per_block`.
   */

  size_t number_of_blocks = 0;

  int initialValue = 6;

  initializeElementsTo<<<number_of_blocks, threads_per_block>>>(initialValue, a, N);
  cudaDeviceSynchronize();

  /*
   * Check to make sure all values in `a`, were initialized.
   */

  for (int i = 0; i < N; ++i)
  {
    if(a[i] != initialValue)
    {
      printf("FAILURE: target value: %d\t a[%d]: %d\n", initialValue, i, a[i]);
      cudaFree(a);
      exit(1);
    }
  }
  printf("SUCCESS!\n");

  cudaFree(a);
}
