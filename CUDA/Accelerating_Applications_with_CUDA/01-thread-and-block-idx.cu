/*Currently the 01-thread-and-block-idx.cu file contains a working kernel that is printing a failure message. 
Open the file to learn how to update the execution configuration so that the success message will print.
After refactoring, compile and run the code with the code execution cell below to confirm your work.
 */

#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  } else {
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main()
{
  /*
   * Update the execution configuration so that the kernel
   * will print `"Success!"`.
   */

  printSuccessForCorrectExecutionConfiguration<<<1, 1>>>();
}
