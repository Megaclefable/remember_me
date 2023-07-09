### Grids and Blocks of 2 and 3 Dimensions

Grids and blocks can be defined to have up to 3 dimensions. 
Defining them with multiple dimensions does not impact their performance in any way, 
but can be very helpful when dealing with data that has multiple dimensions, for example, 2d matrices.

To define either grids or blocks with two or 3 dimensions, use CUDA's dim3 type as such:

```cuda
dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks(16, 16, 1);
someKernel<<<number_of_blocks, threads_per_block>>>();
```

Given the example just above, the variables gridDim.x, gridDim.y, blockDim.x, and blockDim.y inside of someKernel, would all be equal to 16.

------------------------------------------------

Credit for the original Heat Conduction CPU source code in this task is given to the article An OpenACC Example Code for a C-based heat conduction code from the University of Houston.



http://docplayer.net/30411068-An-openacc-example-code-for-a-c-based-heat-conduction-code.html
