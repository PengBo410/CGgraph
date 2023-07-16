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
		count_type owner; // 
		offset_type nbrSize;
		countl_type offset_start;
		vertex_data_type msg; // vertexValue
	};


	template <typename Work_type>
	__device__ __forceinline__ static void schedule(Thread_data_type& thread_data, Work_type work)
	{
		//donate
		__shared__ Donate_data_type donatePool_share[DONATE_POOL_SIZE];//DONATE_POOL_SIZE 32 
		__shared__ uint32_t donateTemp_share[WARP_NUM_PER_BLOCK];
		__shared__ uint32_t donateAtomic_share;
		__shared__ uint32_t helpAtomic_share;
		//__shared__ uint32_t donateBitmap_share;

		// block work
		__shared__ Block_data_type blockWork_share;



		//通用
		const unsigned int laneId = LaneId(); 
		const unsigned int warpId = (threadIdx.x) >> 5;  



		/******************************************************************************
		 *                                【init】
		 ******************************************************************************/
		if (threadIdx.x == (BLOCKSIZE - 1))
		{
			blockWork_share.owner = 1025;
			donateAtomic_share = 0; 
			helpAtomic_share = 0; 
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
				blockWork_share.owner = 1025;
				blockWork_share.nbrSize = thread_data.nbrSize;
				blockWork_share.offset_start = thread_data.offset_start;
				blockWork_share.msg = thread_data.msg;

				thread_data.nbrSize = 0;
			}
			__syncthreads();

			countl_type offset_start = blockWork_share.offset_start;
			offset_type nbrSize = blockWork_share.nbrSize;
			vertex_data_type msg = blockWork_share.msg;

		
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
		int mask = __ballot_sync(0xffffffff, thread_data.nbrSize >= WARPSIZE);
		int maskNum = __popc(mask);
		if (laneId == 0) donateTemp_share[warpId] = maskNum;
		__syncthreads();

		int sum = 0;
		for (int i = 0; i < WARP_NUM_PER_BLOCK; i++) sum += donateTemp_share[i];

		int avg = sum / WARP_NUM_PER_BLOCK; //TODO: float ?
		if (laneId == 0)
		{
			if (donateTemp_share[warpId] < (avg + 2)) donateTemp_share[warpId] = 0; //2
		}
		__syncthreads();

		// 
		//if (donateTemp_share[warpId] != 0)
		//{
		//	sum = 0; //to 0
		//	for (int i = 0; i < WARP_NUM_PER_BLOCK; i++) sum += donateTemp_share[i];
		//	//int donateNum = donateTemp_share[warpId] * DONATE_POOL_SIZE / sum;

		//	
		//	if ((mask & ((int)1 << laneId)))
		//	{
		//		if (donateAtomic_share < DONATE_POOL_SIZE)
		//		{
		//			uint32_t shareIndex = atomicAdd(&donateAtomic_share, 1);
		//			//uint32_t shareIndex = atomicCanAdd(&donateAtomic_share, DONATE_POOL_SIZE);
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

		
		if (donateTemp_share[warpId] != 0)
		{
			sum = 0; 
			for (int i = 0; i < WARP_NUM_PER_BLOCK; i++) sum += donateTemp_share[i];
			int donateNum = cpj_min(donateTemp_share[warpId] * DONATE_POOL_SIZE / sum, maskNum);
			//assert(donateNum <= maskNum);

			//
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
		__syncthreads(); 


	
		//if (laneId == 0)
		//{
		//	//atomicOr(&donateBitmap_share,((uint32_t)1 << warpId));
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
		mask = __ballot_sync(0xffffffff, thread_data.nbrSize >= (WARPSIZE >> 1));
		while (mask != 0)
		{
			
			int leader = __ffs(mask) - 1;

			//int clear_mask = ~(int(1) << (leader));
			mask &= ~(int(1) << (leader));

			
			offset_type nbrSize = __shfl_sync(0xffffffff, thread_data.nbrSize, leader);
			countl_type offset_start = __shfl_sync(0xffffffff, thread_data.offset_start, leader);
			vertex_data_type msg = __shfl_sync(0xffffffff, thread_data.msg, leader);

			if (leader == laneId) thread_data.nbrSize = 0;

			
			for (int lid = laneId; lid < nbrSize; lid += WARPSIZE)
			{
				work(offset_start + lid, msg);
			}
		}



		/******************************************************************************
		 *                                【Thread - work】
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
			return; //
		}
		else
		{
			//do{
			uint32_t end = donateAtomic_share; //
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

				// Warp - work
				offset_type nbrSize = donatePool_share[cur].nbrSize;
				countl_type offset_start = donatePool_share[cur].offset_start;
				vertex_data_type msg = donatePool_share[cur].msg;

				for (int lid = laneId; lid < nbrSize; lid += WARPSIZE)
				{
					work(offset_start + lid, msg);
				}
			}
		}





		//if ((donateBitmap_share == 0xFF) && (helpAtomic_share == donateAtomic_share))  return; //
		//else 
		//{
		//	//do {
		   //	assert(donateAtomic_share <= DONATE_POOL_SIZE);
		//		for (int index_help = helpAtomic_share; index_help < donateAtomic_share; index_help ++)
		//		{
		//			if (donatePool_share[index_help].nbrSize != 0)//
		//			{
		//				offset_type nbrSize = 0;
		//				// 
		//				if (laneId == 0)
		//				{
		//					nbrSize = donatePool_share[index_help].nbrSize;
		//					offset_type assumed = 0xffffffff;
		//					do
		//					{
		//						if (nbrSize == 0) break; 
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



		//if ((donateBitmap_share == 0xFF) && (donateAtomic_share == 0)) {} //
		//else
		//{
		//	//do {
		//		for (int index_offset = 1; index_offset < DONATE_POOL_SIZE; index_offset++)
		//		{
		//			int index_help = ((warpId + 1) * 4 + index_offset) & 31;
		//			if (donatePool_share[index_help].nbrSize != 0)//
		//			{
		//				offset_type nbrSize = 0;
		//				// 
		//				if (laneId == 0)
		//				{
		//					nbrSize = donatePool_share[index_help].nbrSize;
		//					offset_type assumed;
		//					do
		//					{
		//						if (nbrSize == 0) break; // 
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