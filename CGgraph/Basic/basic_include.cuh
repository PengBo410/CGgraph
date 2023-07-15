#pragma once

#include "Platform/platform_def.hpp"

#include "Console/console.hpp"

#include "GPU/cuda_include.cuh"
#include "GPU/cuda_check.cuh"
#include "GPU/gpu_util.cuh"
#include "GPU/gpu_info.cuh"

#ifdef PLATFORM_LINUX
    #include "Thread/cpu_info.hpp"
#endif

#include "Thread/omp_def.hpp"
#include "Thread/atomic_basic.hpp"
#include "Thread/atomic.hpp"

#include "Time/time.hpp"
//#include "Time/time_cpu.hpp"
#include "Time/time_gpu.cuh"

#include "Bitmap/dense_bitset.hpp"
#include "Bitmap/fixed_bitset.cuh"

#include "Log/logger2.hpp"

#include "Other/create_folder.hpp"
#include "Other/IO.hpp"
#include "Other/doubleBuffer.cuh"
#include "Other/finder.cuh"
#include "Other/generate_mtx.hpp"

//Graph
#include "Graph/basic_def.hpp"
#include "Graph/basic_struct.hpp"
#include "Graph/graphFileList.hpp"
#include "Graph/graphBinReader.hpp"
#include "Graph/util.cuh"