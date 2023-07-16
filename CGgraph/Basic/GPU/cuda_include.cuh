#pragma once

#define CUDA_INCLUDE_TEMP  
                           

#ifdef CUDA_INCLUDE_TEMP

    #include </usr/local/cuda-11.7/include/cuda_runtime.h>
    #include </usr/local/cuda-11.7/include/cuda.h>
    #include </usr/local/cuda-11.7/include/device_launch_parameters.h>
#else
    #include <cuda_runtime.h>
    #include <cuda.h>
    #include <device_launch_parameters.h>
#endif