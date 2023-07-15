#pragma once

//#include "Basic/basic_include.cuh"
#include "Basic/Type/data_type.hpp"
#include "Basic/Graph/basic_def.hpp"


namespace WarpDonate
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
		countl_type offset_start;
		vertex_data_type msg; // vertexValue
	};

	//shared_memory
	struct Donate_data_type
	{
		offset_type nbrSize;
		countl_type offset_start;
		vertex_data_type msg; // vertexValue
	};

	//shared_memory
	struct Block_data_type {
		count_type owner; // 超过block的最大size，表示刚开始谁都不具有该锁
		offset_type nbrSize;
		countl_type offset_start;
		vertex_data_type msg; // vertexValue
	};


	template <typename Work_type>
	__device__ __forceinline__ static void schedule(Thread_data_type& thread_data, Work_type work)
	{
		//donate
		__shared__ Donate_data_type donatePool_share[DONATE_POOL_SIZE];//DONATE_POOL_SIZE 32 //每个block拥有的donate pool 的大小, 以顶点个数为单位
		__shared__ uint32_t donateTemp_share[WARP_NUM_PER_BLOCK];
		__shared__ uint32_t donateAtomic_share;
		__shared__ uint32_t helpAtomic_share;
		//__shared__ uint32_t donateBitmap_share;

		// block work
		__shared__ Block_data_type blockWork_share;



		//通用
		const unsigned int laneId = LaneId();  //返回调用线程的land ID (0-31)
		const unsigned int warpId = (threadIdx.x) >> 5;  //返回调用线程的land ID (0-31)



		/******************************************************************************
		 *                                【初始化】
		 ******************************************************************************/
		if (threadIdx.x == (BLOCKSIZE - 1))
		{
			blockWork_share.owner = 1025;
			donateAtomic_share = 0; //donate pool的已使用空间
			helpAtomic_share = 0; //所有线程均未结束donate
			//donateBitmap_share = 0;
		}

		if (threadIdx.x < DONATE_POOL_SIZE) donatePool_share[threadIdx.x].nbrSize = 0;

		__syncthreads();



		/******************************************************************************
		 *                                【Block - work】
		 * >= 256
		 ******************************************************************************/
		do {

			if (thread_data.nbrSize >= BLOCKSIZE) blockWork_share.owner = threadIdx.x;
			__syncthreads();

			if (blockWork_share.owner >= 1025) { __syncthreads(); break; }

			if (blockWork_share.owner == threadIdx.x)
			{
				blockWork_share.owner = 1025;// 归还锁
				blockWork_share.nbrSize = thread_data.nbrSize;
				blockWork_share.offset_start = thread_data.offset_start;
				blockWork_share.msg = thread_data.msg;

				thread_data.nbrSize = 0;
			}
			__syncthreads();

			countl_type offset_start = blockWork_share.offset_start;
			offset_type nbrSize = blockWork_share.nbrSize;
			vertex_data_type msg = blockWork_share.msg;

			//同一block中的所有线程共同完成单个顶点的工作 (多个worker)
			for (int tid = threadIdx.x; tid < nbrSize; tid += BLOCKSIZE)
			{
				work(offset_start + tid, msg);
			}
			__syncthreads();

		} while (true);




		/******************************************************************************
		 *                                【Warp - donate】
		 * [32 - 256)
		 ******************************************************************************/
		int mask = __ballot_sync(0xffffffff, thread_data.nbrSize >= WARPSIZE);//哪些线程满足np_local.size >= WP_SIZE
		int maskNum = __popc(mask);
		if (laneId == 0) donateTemp_share[warpId] = maskNum;
		__syncthreads();

		int sum = 0;
		for (int i = 0; i < WARP_NUM_PER_BLOCK; i++) sum += donateTemp_share[i];

		int avg = sum / WARP_NUM_PER_BLOCK; //TODO: float ?
		if (laneId == 0)
		{
			if (donateTemp_share[warpId] < (avg + 2)) donateTemp_share[warpId] = 0; //2 为阈值,为了让有需要的warp可以优先共享到
		}
		__syncthreads();

		// V-0: 需要donate的warp开始捐献work, 不需要捐献的开始执行warp - work
		//if (donateTemp_share[warpId] != 0)
		//{
		//	sum = 0; //归零
		//	for (int i = 0; i < WARP_NUM_PER_BLOCK; i++) sum += donateTemp_share[i];
		//	//int donateNum = donateTemp_share[warpId] * DONATE_POOL_SIZE / sum;

		//	//捐献任务
		//	if ((mask & ((int)1 << laneId)))
		//	{
		//		if (donateAtomic_share < DONATE_POOL_SIZE)
		//		{
		//			uint32_t shareIndex = atomicAdd(&donateAtomic_share, 1);
		//			//uint32_t shareIndex = atomicCanAdd(&donateAtomic_share, DONATE_POOL_SIZE);// 有问题
		//			if (shareIndex < DONATE_POOL_SIZE)
		//			{
		//				donatePool_share[shareIndex].nbrSize = thread_data.nbrSize;
		//				donatePool_share[shareIndex].offset_start = thread_data.offset_start;
		//				donatePool_share[shareIndex].msg = thread_data.msg;

		//				thread_data.nbrSize = 0;
		//			}
		//		}

		//	}
		//}

		// V-1: 需要donate的warp开始捐献work, 不需要捐献的开始执行warp - work
		if (donateTemp_share[warpId] != 0)
		{
			sum = 0; //归零
			for (int i = 0; i < WARP_NUM_PER_BLOCK; i++) sum += donateTemp_share[i];
			int donateNum = cpj_min(donateTemp_share[warpId] * DONATE_POOL_SIZE / sum, maskNum);
			//assert(donateNum <= maskNum);

			//一个warp中只有满足的线程进入
			//if ((mask & ((uint32_t)1 << laneId)))
			if (thread_data.nbrSize >= WARPSIZE)
			{
				int leader = __ffs(mask) - 1;    // select a leader
				uint32_t shareIndex = 0;
				if (laneId == leader) shareIndex = atomicAdd(&donateAtomic_share, donateNum);
				shareIndex = __shfl_sync(mask, shareIndex, leader);

				int offset = __popc(mask & ((1 << laneId) - 1));
				if (offset < donateNum)
				{
					donatePool_share[shareIndex + offset].nbrSize = thread_data.nbrSize;
					donatePool_share[shareIndex + offset].offset_start = thread_data.offset_start;
					donatePool_share[shareIndex + offset].msg = thread_data.msg;

					thread_data.nbrSize = 0;
				}

			}
		}
		__syncthreads(); //只能是等待玩所有warp执行完donate


		// 共享完成后做标记
		//if (laneId == 0)
		//{
		//	//atomicOr(&donateBitmap_share,((uint32_t)1 << warpId));//事实证明自己动手写的比原装的速度快
		//	uint32_t old = donateBitmap_share;
		//	uint32_t assumed;
		//	uint32_t msg_decode;
		//	do
		//	{
		//		assert((old & ((uint32_t)1 << warpId)) == 0);
		//		msg_decode = (old | ((uint32_t)1 << warpId));
		//		assumed = old;
		//		old = atomicCAS(&donateBitmap_share, assumed, msg_decode);
		//	} while (assumed != old);
		//	//atomicAdd(&donateBitmap_share, 1);
		//}

		//assert(__activemask() == 0xFFFFFFFF);

		/******************************************************************************
		 *                                【Warp - work】
		 * [16 - 256)
		 ******************************************************************************/
		mask = __ballot_sync(0xffffffff, thread_data.nbrSize >= (WARPSIZE >> 1));//哪些线程满足np_local.size >= WP_SIZE/2
		while (mask != 0)
		{
			// __ffs计算一个int的least significant bit, 也就是最右边的第一个1的位置，如10(1010), __ffs(10)= 2， 9(1001), __ffs(9)= 1
			int leader = __ffs(mask) - 1;

			//int clear_mask = ~(int(1) << (leader));
			mask &= ~(int(1) << (leader));

			//将lead的信息通知同一warp中的其余线程,所有线程都要通知，因为都要参与
			offset_type nbrSize = __shfl_sync(0xffffffff, thread_data.nbrSize, leader);//TODO 考虑将nbrSize和offset_start合并为一个
			countl_type offset_start = __shfl_sync(0xffffffff, thread_data.offset_start, leader);
			vertex_data_type msg = __shfl_sync(0xffffffff, thread_data.msg, leader);

			if (leader == laneId) thread_data.nbrSize = 0;

			//同一warp中的所有线程共同完成单个顶点的工作
			for (int lid = laneId; lid < nbrSize; lid += WARPSIZE)
			{
				work(offset_start + lid, msg);
			}
		}



		/******************************************************************************
		 *                                【Thread - work】
		 * [1-16)
		 ******************************************************************************/
		for (int tid = 0; tid < thread_data.nbrSize; tid++)
		{
			work(thread_data.offset_start + tid, thread_data.msg);  // stall-wait
		}


		//__syncthreads();

		/******************************************************************************
		 *                                【Warp - help】
		 ******************************************************************************/
		if (/*(donateBitmap_share == 0xFF) &&*/ (donateAtomic_share <= helpAtomic_share))
		{
			//assert(__activemask() == 0xFFFFFFFF);
			return; //所有warp都结束donate,且无donate的work
		}
		else
		{
			//do{
			uint32_t end = donateAtomic_share; //这里避免直接操作shared memory
			uint32_t cur;
			while (true)
			{
				if (laneId == 0)
				{
					//cur = atomicCanAdd(&helpAtomic_share, end);
					cur = atomicAdd(&helpAtomic_share, 1);
				}
				cur = __shfl_sync(0xffffffff, cur, 0);
				if (cur >= end) break;

				//执行 Warp - work
				offset_type nbrSize = donatePool_share[cur].nbrSize;
				countl_type offset_start = donatePool_share[cur].offset_start;
				vertex_data_type msg = donatePool_share[cur].msg;
				//assert(nbrSize != 0);

				for (int lid = laneId; lid < nbrSize; lid += WARPSIZE)
				{
					work(offset_start + lid, msg);
				}
			}
			//} while ((donateBitmap_share != 0xFF) || (helpAtomic_share < donateAtomic_share));
		}





		//if ((donateBitmap_share == 0xFF) && (helpAtomic_share == donateAtomic_share))  return; //所有warp都结束donate,且无donate的work
		//else 
		//{
		//	//do {
		   //	assert(donateAtomic_share <= DONATE_POOL_SIZE);
		//		for (int index_help = helpAtomic_share; index_help < donateAtomic_share; index_help ++)
		//		{
		//			if (donatePool_share[index_help].nbrSize != 0)//应该同一warp中的所有线程读取到的是同一值
		//			{
		//				offset_type nbrSize = 0;
		//				// 线程-0去尝试获取数据
		//				if (laneId == 0)
		//				{
		//					nbrSize = donatePool_share[index_help].nbrSize;
		//					offset_type assumed = 0xffffffff;
		//					do
		//					{
		//						if (nbrSize == 0) break; // 别人抢到了
		//						assumed = nbrSize;
		//						nbrSize = atomicCAS(&(donatePool_share[index_help].nbrSize), assumed, 0);
		//					} while (assumed != nbrSize);

		//					if (assumed == nbrSize) atomicAdd(&helpAtomic_share, 1);
		//				}

		//				nbrSize = __shfl_sync(0xffffffff, nbrSize, 0);

		//				if (nbrSize != 0)
		//				{
		//					offset_type offset_start = donatePool_share[index_help].offset_start;
		//					vertex_data_type msg = donatePool_share[index_help].msg;

		//					for (int lid = laneId; lid < nbrSize; lid += WARP_SIZE)
		//					{
		//						work(offset_start + lid, msg);
		//					}
		//				}
		//			}
		//		}

		//	//} while ((donateBitmap_share != 0xFF));
		//}



		//if ((donateBitmap_share == 0xFF) && (donateAtomic_share == 0)) {} //所有warp都结束donate,且无donate的work
		//else
		//{
		//	//do {
		//		for (int index_offset = 1; index_offset < DONATE_POOL_SIZE; index_offset++)
		//		{
		//			int index_help = ((warpId + 1) * 4 + index_offset) & 31;
		//			if (donatePool_share[index_help].nbrSize != 0)//应该同一warp中的所有线程读取到的是同一值
		//			{
		//				offset_type nbrSize = 0;
		//				// 线程-0去尝试获取数据
		//				if (laneId == 0)
		//				{
		//					nbrSize = donatePool_share[index_help].nbrSize;
		//					offset_type assumed;
		//					do
		//					{
		//						if (nbrSize == 0) break; // 别人抢到了
		//						assumed = nbrSize;
		//						nbrSize = atomicCAS(&(donatePool_share[index_help].nbrSize), assumed, 0);
		//					} while (assumed != nbrSize);
		//				}

		//				nbrSize = __shfl_sync(0xffffffff, nbrSize, 0);

		//				if (nbrSize != 0)
		//				{
		//					offset_type offset_start = donatePool_share[index_help].offset_start;
		//					vertex_data_type msg = donatePool_share[index_help].msg;

		//					for (int lid = laneId; lid < nbrSize; lid += WARP_SIZE)
		//					{
		//						work(offset_start + lid, msg);
		//					}
		//				}
		//			}
		//		}

		//	//} while ((donateBitmap_share != 0xFF));
		//}







	   //if (donateBitmap_share != 0xFF) printf("donateBitmap_share = [%u]\n");

	}// end of func [schedule(...)]



}// end of namespace [WarpDonate]