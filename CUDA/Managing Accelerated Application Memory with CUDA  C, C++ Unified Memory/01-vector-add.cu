// Iterative Optimizations with the NVIDIA Command Line Profiler

/*
 The only way to be assured that attempts at optimizing accelerated code bases are actually successful is to profile the application for quantitative information about the application's performance. 
 nsys is the Nsight Systems command line tool. It ships with the CUDA toolkit, and is a powerful tool for profiling accelerated applications.
 nsys is easy to use. Its most basic usage is to simply pass it the path to an executable compiled with nvcc. 
 nsys will proceed to execute the application, 
 after which it will print a summary output of the application's GPU activities, CUDA API calls, 
 as well as information about Unified Memory activity, a topic which will be covered extensively later in this lab.

 When accelerating applications, or optimizing already-accelerated applications, 
 take a scientific and iterative approach. Profile your application after making changes, take note, 
 and record the implications of any refactoring on performance. Make these observations early and often: frequently, 
 enough performance boost can be gained with little effort such that you can ship your accelerated application. 
 Additionally, frequent profiling will teach you how specific changes to your CUDA code bases impact its actual performance: knowledge 
 that is hard to acquire when only profiling after many kinds of changes in your code bases.
*/

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
 * This is a CUDA kernel function that adds the elements of two arrays `a` and `b` and stores the result in the `result` array.
 * It uses the CUDA thread index and block index to calculate a unique index for each thread and 
 * the `stride` is calculated for handling cases where `N` is larger than the total threads launched.
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
/* Detail : 
 * This is a host function to confirm if all elements in the `vector` array are equal to the provided `target` value.
 * If they are not equal, it prints a fail message and the program exits with an error status.
 * If all values are equal, a success message is printed.
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

int main()
{
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  // Allocating Unified Memory so these arrays are accessible both from the host (CPU) and the device (GPU).
  cudaMallocManaged(&a, size);
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
  /* Detail :
   * The threads per block and the number of blocks are both set to 1.
   * This means only one thread will execute the kernel which could result in under-utilization of the GPU resources.
   */
  threadsPerBlock = 1;
  numberOfBlocks = 1;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  // The kernel is launched with `numberOfBlocks` blocks each with `threadsPerBlock` threads.
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}


// !nvcc -o single-thread-vector-add 01-vector-add/01-vector-add.cu -run
// !nsys profile --stats=true ./single-thread-vector-add
