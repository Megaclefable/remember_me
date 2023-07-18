// API status -> decrease for :  !nsys profile --stats=true ./multi-thread-vector-add

#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */
/* Detail :
 * This device function addVectorsInto adds corresponding elements of arrays 'a' and 'b' and stores the result in 'result'.
 * The CUDA kernel uses parallel processing where each thread performs an addition operation.
 * 'index' is calculated using both thread and block indices to uniquely identify each thread.
 * 'stride' is used to handle cases where the array size is larger than the total number of launched threads.
 */


__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */
/* Details :
 * The host function checkElementsAre checks if all elements of 'vector' are equal to the 'target' value.
 * If an element is not equal to 'target', it prints an error message and the program is terminated.
 * If all elements are equal to 'target', it prints a success message.
 */
void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

/*
 * In the main function, arrays 'a', 'b', and 'c' are allocated memory using cudaMallocManaged, which enables Unified Memory access.
 * 'a' is initialized with 3, 'b' with 4, and 'c' with 0.
 * CUDA kernel addVectorsInto is then launched with one thread and one block to add 'a' and 'b' into 'c'.
 * The function then checks for any CUDA errors from the kernel execution or during device synchronization.
 * It also checks if all elements in 'c' are equal to 7 (3 + 4).
 * At the end, the memory allocated to 'a', 'b', and 'c' is freed using cudaFree.
 */
int main()
{
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  /*
   * nsys should register performance changes when execution configuration
   * is updated.
   */

  // Define execution configuration
  threadsPerBlock = 1;
  numberOfBlocks = 1;

  // Define CUDA error variables
  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  // Verify the result
  checkElementsAre(7, c, N);

  // Free the allocated memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}


/*
Execution Configuration: The performance of a CUDA application can be significantly affected by the configuration of the grid and blocks 
(i.e., the number of blocks and the number of threads per block). 
This can control how well the application utilizes the GPU's multiprocessors.

Memory Management: The use of different memory types (global, shared, constant, texture) can also affect performance. 
By strategically managing memory usage and optimizing memory access patterns, developers can potentially reduce memory latency and increase the overall speed of the program.

Asynchronous Operations: CUDA supports overlapping of computation and data transfer, as well as overlapping of computations from different streams. 
If your program uses these techniques, this could lead to significant performance improvements.

Optimization of Mathematical Operations: This might include using intrinsic functions that can map directly to specific GPU instructions, thus providing a performance benefit.

Here, given the code, we don't see an explicit optimization like using shared memory, 
overlapping data transfer and computation, or changing the execution configuration. 
If there's an optimization applied outside the given code, for example, during the compilation process with NVCC compiler flags, 
we can't see it in this context. To evaluate and verify the optimization effects, 
profiling tools like NVIDIA's Nsight Systems (nsys) can be used to analyze the performance before and after the changes.
*/
