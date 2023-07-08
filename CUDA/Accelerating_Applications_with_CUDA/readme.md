#### Writing Application Code for the GPU
CUDA provides extensions for many common programming languages, in the case of this lab, C/C++. <br>
These language extensions easily allow developers to run functions in their source code on a GPU.

Below is a .cu file (.cu is the file extension for CUDA-accelerated programs). <br>
It contains two functions, the first which will run on the CPU, the second which will run on the GPU. 
Spend a little time identifying the differences between the functions, both in terms of how they are defined, and how they are invoked.

```cuda
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}

```
