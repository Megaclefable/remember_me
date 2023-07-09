#include <stdio.h>

/*5 threads , 5 blocks =>  25 times*/

__global__ void firstParallel()
{
  printf("This is running in parallel.\n");
}

int main()
{
  firstParallel<<<5, 5>>>();
  cudaDeviceSynchronize();
}
