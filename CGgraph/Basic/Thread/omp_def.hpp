#pragma once

#include <omp.h>
#include <iostream>
#include <thread>

#define omp_parallel _Pragma("omp parallel")
#define omp_parallel_for _Pragma("omp parallel for") for
#define omp_parallel_for_1 _Pragma("omp parallel for schedule (static,1)") for
#define omp_parallel_for_256 _Pragma("omp parallel for schedule (static,256)") for

// 无openMP时c++使用
//#define parallel_for for
//#define parallel_for_1 for
//#define parallel_for_256 for


/***********************************************************************
 *                              【CPU INFO】
 ***********************************************************************/
uint64_t ThreadNum = omp_get_max_threads();
uint64_t SocketNum = numa_num_configured_nodes();
uint64_t ThreadPerSocket = ThreadNum / SocketNum;

inline uint64_t getThreadSocketId(uint64_t threadId) {
	return threadId / ThreadPerSocket;
}

inline uint64_t getThreadSocketOffset(uint64_t threadId) {
	return threadId % ThreadPerSocket;
}

void reSetThreadNum(uint64_t threadNum)
{
	omp_set_num_threads(threadNum);
	ThreadNum = threadNum;
	ThreadPerSocket = ThreadNum / SocketNum;
}




bool threadBindToCore(int coreId)
{
	int nThreads = std::thread::hardware_concurrency();
	cpu_set_t set;
	CPU_ZERO(&set);

	if (coreId < 0 || coreId > nThreads)
	{
		printf("[Fail]: Input coreId invalid, coreId between [0, %u]\n", nThreads);
		return false;
	}

	CPU_SET(coreId, &set);
	if (sched_setaffinity(0, sizeof(cpu_set_t), &set) < 0)
	{
		printf("[Fail]: Unable to Set Affinity\n");
		return false;
	}

	return true;
}


void threadId_2_coreId()
{
	
#pragma omp parallel //num_threads(4)
	{

		int thread_id = omp_get_thread_num();
		int core_id = sched_getcpu();/

		printf("Thread[%2d] is running on CPU{%2d}\n", thread_id, core_id);
	}

	
}

