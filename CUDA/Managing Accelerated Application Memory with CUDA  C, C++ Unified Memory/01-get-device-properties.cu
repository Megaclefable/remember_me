// Streaming Multiprocessors and Querying the Device

// Streaming Multiprocessors and Warps
/*
The GPUs that CUDA applications run on have processing units called streaming multiprocessors, or SMs. 
During kernel execution, blocks of threads are given to SMs to execute. In order to support the GPU's ability to perform as many parallel operations as possible,
performance gains can often be had by choosing a grid size that has a number of blocks that is a multiple of the number of SMs on a given GPU.

Additionally, SMs create, manage, schedule, and execute groupings of 32 threads from within a block called warps. 
A more in depth coverage of SMs and warps is beyond the scope of this course, however, 
it is important to know that performance gains can also be had by choosing a block size that has a number of threads that is a multiple of 32.
*/

// Programmatically Querying GPU Device Properties
/*In order to support portability, since the number of SMs on a GPU can differ depending on the specific GPU being used, 
the number of SMs should not be hard-coded into a code bases. 
Rather, this information should be acquired programatically.

The following shows how, in CUDA C/C++, to obtain a C struct which contains many properties about the currently active GPU device, including its number of SMs:
*/

/* 
int deviceId;
cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.

cudaDeviceProp props;
cudaGetDeviceProperties(&props, deviceId); // `props` now has many useful properties about
                                           // the active GPU device.
*/

// Exercise: Query the Device
/*Currently, 01-get-device-properties.cu contains many unassigned variables, and will print gibberish information intended to describe details about the currently active GPU.

Build out 01-get-device-properties.cu to print the actual values for the desired device properties indicated in the source code. 
In order to support your work, and as an introduction to them, use the CUDA Runtime Docs to help identify the relevant properties in the device props struct. 
Refer to the solution if you get stuck.
*/

#include <stdio.h>

int main()
{
  /*
   * Assign values to these variables so that the output string below prints the
   * requested properties of the currently active GPU.
   */

  int deviceId;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int warpSize;

  /*
   * There should be no need to modify the output string below.
   */

  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}


