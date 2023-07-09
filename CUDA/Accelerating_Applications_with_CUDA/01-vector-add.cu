/* 
Final Exercise: Accelerate Vector Addition Application
The following challenge will give you an opportunity to use everything that you have learned thus far in the lab. 
It involves accelerating a CPU-only vector addition program, which, while not the most sophisticated program, 
will give you an opportunity to focus on what you have learned about GPU-accelerating an application with CUDA. 
After completing this exercise, if you have time and interest, continue on to the Advanced Content section for some challenges that involve more complex code bases.

01-vector-add.cu contains a functioning CPU-only vector addition application. 
Accelerate its addVectorsInto function to run as a CUDA kernel on the GPU and to do its work in parallel. 
Consider the following that need to occur, and refer to the solution if you get stuck.

Augment the addVectorsInto definition so that it is a CUDA kernel.
Choose and utilize a working execution configuration so that addVectorsInto launches as a CUDA kernel.
Update memory allocations, and memory freeing to reflect that the 3 vectors a, b, and result need to be accessed by host and device code.

Refactor the body of addVectorsInto: 
it will be launched inside of a single thread, and only needs to do one thread's worth of work on the input vectors. 
Be certain the thread will never try to access elements outside the range of the input vectors,
and take care to note whether or not the thread needs to do work on more than one element of the input vectors.
Add error handling in locations where CUDA code might otherwise silently fail.
*/

#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

void addVectorsInto(float *result, float *a, float *b, int N)
{
  for(int i = 0; i < N; ++i)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  addVectorsInto(c, a, b, N);

  checkElementsAre(7, c, N);

  free(a);
  free(b);
  free(c);
}
