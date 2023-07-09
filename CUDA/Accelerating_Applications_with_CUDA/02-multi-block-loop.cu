/* Currently, the loop function inside 02-multi-block-loop.cu runs a for loop that will serially print the numbers 0 through 9. 
* Refactor the loop function to be a CUDA kernel which will launch to execute N iterations in parallel. 
* After successfully refactoring, the numbers 0 through 9 should still be printed. 
For this exercise, as an additional constraint, use an execution configuration that launches at least 2 blocks of threads. */


#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

void loop(int N)
{
  for (int i = 0; i < N; ++i)
  {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, be sure to use more than 1 block in
   * the execution configuration.
   */

  int N = 10;
  loop(N);
}
