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



/* ******************************************************************************************************
 *                              【线程绑定】
 * 1.函数threadBindToCore(coreId) 用于将std::thread申请的线程绑定到固定的core上
 * 2.OMP的线程我们采用环境变量的方式绑定：
 *   ./bashrc中：
 *    export GOMP_CPU_AFFINITY="2 3 4 5 6 7 14 15 16 17 18 19"
 *    注意：这里的个数要与OMP_NUM_THREADS=12设定的个数对应上
 *         OMP_PROC_BIND的优先级比GOMP_CPU_AFFINITY高，因此设置GOMP_CPU_AFFINITY要取消OMP_PROC_BIND
 * ******************************************************************************************************/
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


//可以测试出omp中线程id与硬件id的对应关系,先关闭hybirdThread在测试
//注意环境变量 GOMP_CPU_AFFINITY 与 OMP_PROC_BIND  (https://blog.csdn.net/stormbjm/article/details/17136409/)
//当前仅仅在环境变量中设置：OMP_PROC_BIND=TRUE
void threadId_2_coreId()
{
	//所有线程都参与
	//发现线程号(thread_id)与物理id(core_id)一一对应 (见结果一)
#pragma omp parallel //num_threads(4)
	{

		int thread_id = omp_get_thread_num();//获取线程的id
		int core_id = sched_getcpu();//获取物理的id

		printf("Thread[%2d] is running on CPU{%2d}\n", thread_id, core_id);
	}

	// 部分线程参与
	// 发现线程号(thread_id)与物理id(core_id)虽然不是一一对应关系，但是尽量拼分到不同的socket上，且相邻的线程位于同一socket (见结果二)
	// 线程[0,1]运行在socket0的CPU{0,6}上，线程[2,3]运行在socket1的CPU{12,18}上
//#pragma omp parallel num_threads(4)
//	{
//
//		int thread_id = omp_get_thread_num();//获取线程的id
//		int core_id = sched_getcpu();//获取物理的id
//
//		printf("Thread[%2d] is running on CPU{%2d}\n", thread_id, core_id);
//	}
}


/*
【结果一】
Thread[ 0] is running on CPU{ 0}
Thread[20] is running on CPU{20}
Thread[19] is running on CPU{19}
Thread[11] is running on CPU{11}
Thread[ 6] is running on CPU{ 6}
Thread[21] is running on CPU{21}
Thread[15] is running on CPU{15}
Thread[17] is running on CPU{17}
Thread[18] is running on CPU{18}
Thread[14] is running on CPU{14}
Thread[ 3] is running on CPU{ 3}
Thread[12] is running on CPU{12}
Thread[22] is running on CPU{22}
Thread[10] is running on CPU{10}
Thread[23] is running on CPU{23}
Thread[ 5] is running on CPU{ 5}
Thread[ 2] is running on CPU{ 2}
Thread[ 4] is running on CPU{ 4}
Thread[ 8] is running on CPU{ 8}
Thread[ 1] is running on CPU{ 1}
Thread[ 9] is running on CPU{ 9}
Thread[16] is running on CPU{16}
Thread[ 7] is running on CPU{ 7}
Thread[13] is running on CPU{13}


【结果二】：
Thread[ 0] is running on CPU{ 0}
Thread[ 2] is running on CPU{12}
Thread[ 3] is running on CPU{18}
Thread[ 1] is running on CPU{ 6}
*/
