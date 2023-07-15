#pragma once

#include "Basic/Type/data_type.hpp"
#include "Basic/Graph/basic_def.hpp"

namespace CTA
{
	__device__ __forceinline__ unsigned int LaneId()
	{
		unsigned int ret;
		asm("mov.u32 %0, %%laneid;" : "=r"(ret));
		return ret;
	}

	//调用压缩是否会好点，32线程压一个，因为32线程他们的数据有关系
	struct Thread_data_type {
		offset_type nbrSize;
		offset_type offset_start;
		vertex_data_type msg; // vertexValue
	};

	struct Shared_data_type {
		count_type owner; // 超过block的最大size，表示刚开始谁都不具有该锁
		offset_type nbrSize;
		offset_type offset_start;
		vertex_data_type msg; // vertexValue
	};

	template <typename Work_type>
	__device__ __forceinline__ static void schedule(Thread_data_type& thread_data, Work_type work)
	{
		const count_type TB_SIZE = BLOCKSIZE;
        const count_type WARP_SIZE = WARPSIZE;

		__shared__ Shared_data_type shared;

		// 初始化锁 (这一步不加的话，win编译器会出错)
		if (threadIdx.x == 0)
		{
			shared.owner = 1025;
		}

		__syncthreads();

		// 大于256: block
		do {

			if (thread_data.nbrSize >= TB_SIZE) shared.owner = threadIdx.x;
			__syncthreads();

			if (shared.owner >= 1025) { __syncthreads(); break; }

			if (shared.owner == threadIdx.x)
			{
				shared.owner = 1025;// 归还锁
				shared.nbrSize = thread_data.nbrSize;
				shared.offset_start = thread_data.offset_start;
				shared.msg = thread_data.msg;

				thread_data.nbrSize = 0;
			}
			__syncthreads();

			offset_type offset_start = shared.offset_start;
			offset_type nbrSize = shared.nbrSize;
			vertex_data_type msg = shared.msg;

			//同一block中的所有线程共同完成单个顶点的工作 (多个worker)
			for (int tid = threadIdx.x; tid < nbrSize; tid += TB_SIZE)
			{
				work(offset_start + tid, msg);
			}
			__syncthreads();

		} while (true);


		//介于32-256之间 : warp
		const int lane_id = LaneId();  //返回调用线程的land ID (0-31)
		int mask = __ballot_sync(0xffffffff, thread_data.nbrSize >= WARP_SIZE);//哪些线程满足np_local.size >= WP_SIZE
		while (mask != 0)
		{
			// __ffs计算一个int的least significant bit, 也就是最右边的第一个1的位置，如10(1010), __ffs(10)= 2， 9(1001), __ffs(9)= 1
			int leader = __ffs(mask) - 1;

			//int clear_mask = ~(int(1) << (leader));
			mask &= ~(int(1) << (leader));

			//将lead的信息通知同一warp中的其余线程,所有线程都要通知，因为都要参与
			offset_type nbrSize = __shfl_sync(0xffffffff, thread_data.nbrSize, leader);//TODO 考虑将nbrSize和offset_start合并为一个
			offset_type offset_start = __shfl_sync(0xffffffff, thread_data.offset_start, leader);
			vertex_data_type msg = __shfl_sync(0xffffffff, thread_data.msg, leader);

			if (leader == lane_id) thread_data.nbrSize = 0;

			//同一warp中的所有线程共同完成单个顶点的工作
			for (int lid = lane_id; lid < nbrSize; lid += WARP_SIZE)
			{
				work(offset_start + lid, msg);
			}
		}

		// warp的第二个版本
		//const int lane_id = LaneId();  //返回调用线程的land ID (0-31)
		//while (__any_sync(0xffffffff, thread_data.nbrSize >= WARP_SIZE))
		//{
		//	int mask = __ballot_sync(0xffffffff, thread_data.nbrSize >= WARP_SIZE);//哪些线程满足np_local.size >= WP_SIZE
		//	int leader = __ffs(mask) - 1;// __ffs计算一个int或long型数它的二进制第一个最高位为1的位置

		//	//将lead的信息通知同一warp中的其余线程
		//	offset_type nbrSize = __shfl_sync(0xffffffff, thread_data.nbrSize, leader);
		//	offset_type offset_start = __shfl_sync(0xffffffff, thread_data.offset_start, leader);
		//	vertex_data_type msg = __shfl_sync(0xffffffff, thread_data.msg, leader);

		//	if (leader == lane_id) thread_data.nbrSize = 0;

		//	//同一warp中的所有线程共同完成单个顶点的工作
		//	for (int lid = lane_id; lid < nbrSize; lid += WARP_SIZE)
		//	{
		//		work(offset_start + lid, msg);
		//	}
		//}



		//小于32：thread
		for (offset_type tid = 0; tid < thread_data.nbrSize; tid++)
		{
			work(thread_data.offset_start + tid, thread_data.msg);
		}

	}// end of func [schedule]

}// end of namespace [CTA]