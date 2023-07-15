#pragma once

#include "Basic/basic_include.cuh"
#include "device_warpDonate.cuh"
#include "device_CTA.cuh"


namespace BFS_SPACE
{

    __global__ void bfs_kernel(
		common_type* common, uint32_t* trasfer_segment, uint64_t* bitmap,
		countl_type* csr_offset, vertex_id_type* csr_dest, vertex_data_type* vertexValue)
	{
		count_type vertexNum_up = gridDim.x * blockDim.x;

		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < vertexNum_up; threadId += vertexNum_up)
		{
            count_type vertexNum_segment = common[5]; // The number of vertices of each Segment 
            count_type wordNum_segment =  vertexNum_segment / WORDSIZE; // The number of words of each Segment 
            count_type offset_index = threadId / vertexNum_segment; // 当前线程的偏移量所在的位置
            count_type offset = trasfer_segment[offset_index];// 单前线程的偏移量;

            WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };
			//CTA::Thread_data_type thread_data = { 0, 0, 0 };

            //每个线层先确定自己要访问的active的位置
			count_type wordIndex = WORD_OFFSET(threadId) + (common[7] * wordNum_segment);
			count_type wordMod = WORD_MOD(threadId);
			uint64_t blockData_64 = bitmap[wordIndex];

			if ((blockData_64 >> wordMod) & 1)
			{
				// 需要注意的是： offload_offset的基本单位word, 所以一定记得乘以64
				count_type chunkMod = threadId % vertexNum_segment;
				vertex_id_type vertexId = offset * vertexNum_segment + chunkMod;

				thread_data.offset_start = csr_offset[vertexId];
				thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
                //assert(thread_data.nbrSize != 0);
				thread_data.msg = vertexValue[vertexId];

			}// 千万注意： 此处的 "}" 不能包住下面的代码, 否则就是：error 4 unspecified launch failure 以及 一晚上的辛苦

			WarpDonate::schedule(thread_data,
			//CTA::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					vertex_id_type dest = csr_dest[edge_cur];

					if (msg + 1 < vertexValue[dest])
					{
						atomicMin((vertexValue + dest), msg + 1);
					}
				}
			);
		}
	}// end of func [sssp_hybird_device_pcie_opt]


    template<typename GraphEnginer>
	void bfs_device(GraphEnginer& graphEnginer, const count_type nBlock, int deviceId)
	{
		bfs_kernel << <nBlock, BLOCKSIZE >> > (
			graphEnginer.common_device[deviceId], graphEnginer.trasfer_segment_device_vec[deviceId], graphEnginer.bitmap_device_vec[deviceId],
			graphEnginer.partitionResult.csrOffset_device_vec[deviceId], 
            graphEnginer.partitionResult.csrDest_device_vec[deviceId], 
            graphEnginer.vertexValue_temp_device[deviceId]);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}









	/****************************************************************************************************
	 *                                    【WORKLIST_MODEL】
	 * common[0]: worklist_count; common[1]: worklist_size; common[2]: vertexNum; common[3]: edgeNum
	 ****************************************************************************************************/
	__global__ void bfs_worklist_model_device(
		offset_type* common, vertex_id_type* worklist_in, vertex_id_type* worklist_out,
		countl_type* csr_offset, vertex_id_type* csr_dest, vertex_data_type* vertexValue)
	{
		count_type workNum_up = gridDim.x * blockDim.x;
		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < workNum_up; threadId += workNum_up)
		{
			//CTA::Thread_data_type thread_data = { 0, 0, 0 };
			WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };

			if (threadId < common[1])//worklist_size
			{
				vertex_id_type vertexId = worklist_in[threadId];
				thread_data.offset_start = csr_offset[vertexId];
				thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
				thread_data.msg = vertexValue[vertexId];
			}

			//CTA::schedule(thread_data,
			WarpDonate::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					vertex_id_type dest = csr_dest[edge_cur];
					if (msg + 1 < vertexValue[dest])
					{
						if (msg + 1 < atomicMin((vertexValue + dest), msg + 1))
						{
							uint32_t index = atomicAdd(common, 1); //worklist_count
							worklist_out[index] = dest;
						}
					}
				}
			);
		}
	}





	template<typename GraphDeviceWorklist>
	void bfs_worklist_model(GraphDeviceWorklist& graphDeviceWorklist, const count_type nBlock)
	{
		bfs_worklist_model_device << <nBlock, BLOCKSIZE >> > (
			graphDeviceWorklist.common_device, graphDeviceWorklist.worklist_device->in(), graphDeviceWorklist.worklist_device->out(),
			graphDeviceWorklist.csr_offset_device, graphDeviceWorklist.csr_dest_device, graphDeviceWorklist.vertexValue_device);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}






	/***********************************************************
	 * Func: GraphHybird_opt 对应的BFS算法的 Device 实现
	 *
	 * [common]
	 * [offload_offset]
	 * [offload_data]
	 * [csr_offset]
	 * [csr_dest]
	 * [vertexValue]
	 ***********************************************************/
	__global__ void bfs_hybird_device_pcie_opt(
		offset_type* common, count_type* offload_offset, uint64_t* offload_data,
		countl_type* csr_offset, vertex_id_type* csr_dest, vertex_data_type* vertexValue)
	{
		count_type vertexNum_up = gridDim.x * blockDim.x;

		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < vertexNum_up; threadId += vertexNum_up)
		{
			offset_type wordNum_chunk = common[5];// 每个chunk拥有的word数
			count_type vertexNum_chunk = wordNum_chunk * 64; // 每个chunk拥有的顶点数
			count_type offset_index = threadId / vertexNum_chunk;//当前线程的偏移量所在的位置
			count_type offset = offload_offset[offset_index];// 单前线程的偏移量;

			WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };

			//每个线层先确定自己要访问的active的位置
			count_type wordIndex = WORD_OFFSET(threadId);
			count_type wordMod = WORD_MOD(threadId);
			uint64_t blockData_64 = offload_data[wordIndex];

			if ((blockData_64 >> wordMod) & 1)
			{
				// 需要注意的是： offload_offset的基本单位word, 所以一定记得乘以64
				count_type chunkMod = threadId % vertexNum_chunk;
				vertex_id_type vertexId = offset * 64 + chunkMod;

				thread_data.offset_start = csr_offset[vertexId];
				thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
				thread_data.msg = vertexValue[vertexId];

			}// 千万注意： 此处的 "}" 不能包住下面的代码, 否则就是：error 4 unspecified launch failure 以及 一晚上的辛苦

			WarpDonate::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					vertex_id_type dest = csr_dest[edge_cur];

					if (msg + 1 < vertexValue[dest])
					{
						atomicMin((vertexValue + dest), msg + 1);
					}
				}
			);
		}
	}// end of func [sssp_hybird_device_pcie_opt]



	/***********************************************************
	 * Func: GraphHybird_opt 对应的BFS算法的 Device
	 *
	 * [graphHybird_opt.common_device]
	 * [graphHybird_opt.offload_offset_device]
	 * [graphHybird_opt.offload_data_device]
	 * [graphHybird_opt.csr_offset_device]
	 * [graphHybird_opt.csr_dest_device]
	 * [graphHybird_opt.vertexValue_device]
	 ***********************************************************/
	template<typename GraphHybird_opt>
	void bfs_hybird_pcie_opt(GraphHybird_opt& graphHybird_opt, const count_type nBlock)
	{
		bfs_hybird_device_pcie_opt << <nBlock, BLOCKSIZE >> > (
			graphHybird_opt.common_device, graphHybird_opt.offload_offset_device, graphHybird_opt.offload_data_device,
			graphHybird_opt.csr_offset_device, graphHybird_opt.csr_dest_device, graphHybird_opt.vertexValue_device);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}// end of namespace [BFS_SPACE]






namespace SSSP_SPACE
{

    __global__ void sssp_kernel(
		common_type* common, uint32_t* trasfer_segment, uint64_t* bitmap,
		countl_type* csr_offset, vertex_id_type* csr_dest, edge_data_type* csr_weight, vertex_data_type* vertexValue)
	{
		count_type vertexNum_up = gridDim.x * blockDim.x;

		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < vertexNum_up; threadId += vertexNum_up)
		{
            count_type vertexNum_segment = common[5]; // The number of vertices of each Segment  (1024)
            count_type wordNum_segment =  vertexNum_segment / WORDSIZE; // The number of words of each Segment (16)
            count_type offset_index = threadId / vertexNum_segment; // 当前线程的偏移量所在的位置
            count_type segmentId = trasfer_segment[offset_index];// 单前线程的偏移量;

			

            WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };

            //每个线层先确定自己要访问的active的位置
			count_type wordIndex = WORD_OFFSET(threadId) + (common[7] * wordNum_segment);
			count_type wordMod = WORD_MOD(threadId);
			uint64_t blockData_64 = bitmap[wordIndex];

			
			//if(threadId == 0) printf("segmentId = %u, wordIndex = %u\n", segmentId, wordIndex);

			if ((blockData_64 >> wordMod) & 1)
			{
				// 需要注意的是： offload_offset的基本单位word, 所以一定记得乘以64
				count_type chunkMod = threadId % vertexNum_segment;
				vertex_id_type vertexId = segmentId * vertexNum_segment + chunkMod;

				thread_data.offset_start = csr_offset[vertexId];
				thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
                //assert(thread_data.nbrSize != 0);
				thread_data.msg = vertexValue[vertexId];

			}// 千万注意： 此处的 "}" 不能包住下面的代码, 否则就是：error 4 unspecified launch failure 以及 一晚上的辛苦

			WarpDonate::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					vertex_id_type dest = csr_dest[edge_cur];
                    edge_data_type weight = csr_weight[edge_cur];

					if (msg + weight < vertexValue[dest])
					{
						//if(threadIdx.x + blockIdx.x * blockDim.x == 0) printf("dest = %u, value = %u\n", dest, msg + weight);
						atomicMin((vertexValue + dest), msg + weight);
					}
				}
			);
		}
	}// end of func [sssp_kernel]


    template<typename GraphEnginer>
	void sssp_device(GraphEnginer& graphEnginer, const count_type nBlock, int deviceId)
	{
		CUDA_CHECK(cudaSetDevice(deviceId));

		sssp_kernel << <nBlock, BLOCKSIZE >> > (
			graphEnginer.common_device[deviceId], graphEnginer.trasfer_segment_device_vec[deviceId], graphEnginer.bitmap_device_vec[deviceId],
			graphEnginer.partitionResult.csrOffset_device_vec[deviceId], 
            graphEnginer.partitionResult.csrDest_device_vec[deviceId], 
            graphEnginer.partitionResult.csrWeight_device_vec[deviceId],
            graphEnginer.vertexValue_temp_device[deviceId]);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}


	








	/****************************************************************************************************
	 *                                    【WORKLIST_MODEL】
	 * common[0]: worklist_count; common[1]: worklist_size; common[2]: vertexNum; common[3]: edgeNum
	 ****************************************************************************************************/
	__global__ void sssp_worklist_model_device(
		offset_type* common, vertex_id_type* worklist_in, vertex_id_type* worklist_out,
		countl_type* csr_offset, vertex_id_type* csr_dest, edge_data_type* csr_weight, vertex_data_type* vertexValue)
	{
		count_type workNum_up = gridDim.x * blockDim.x;
		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < workNum_up; threadId += workNum_up)
		{
			CTA::Thread_data_type thread_data = { 0, 0, 0 };
			//WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };

			if (threadId < common[1])//worklist_size
			{
				vertex_id_type vertexId = worklist_in[threadId];
				thread_data.offset_start = csr_offset[vertexId];
				thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
				thread_data.msg = vertexValue[vertexId];
			}

			CTA::schedule(thread_data,
			//WarpDonate::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					vertex_id_type dest = csr_dest[edge_cur];
					edge_data_type weight = csr_weight[edge_cur];
					if (msg + weight < vertexValue[dest])
					{
						if (msg + weight < atomicMin((vertexValue + dest), msg + weight))
						{
							uint32_t index = atomicAdd(common, 1); //worklist_count
							worklist_out[index] = dest;
		
						}
					}
				}
			);
		}
	}


	template<typename GraphDeviceWorklist>
	void sssp_worklist_model(GraphDeviceWorklist& graphDeviceWorklist, const count_type nBlock)
	{
		sssp_worklist_model_device << <nBlock, BLOCKSIZE >> > (
			graphDeviceWorklist.common_device, graphDeviceWorklist.worklist_device->in(), graphDeviceWorklist.worklist_device->out(),
			graphDeviceWorklist.csr_offset_device, graphDeviceWorklist.csr_dest_device, graphDeviceWorklist.csr_weight_device, graphDeviceWorklist.vertexValue_device);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}




	/***********************************************************
	 * Func: GraphHybird_opt 对应的SSSP算法的 Device 实现
	 *
	 * [common]
	 * [offload_offset]
	 * [offload_data]
	 * [csr_offset]
	 * [csr_dest]
	 * [csr_weight]
	 * [vertexValue]
	 ***********************************************************/
	__global__ void sssp_hybird_device_pcie_opt(
		offset_type* common, count_type* offload_offset, uint64_t* offload_data,
		countl_type* csr_offset, vertex_id_type* csr_dest, edge_data_type* csr_weight, vertex_data_type* vertexValue)
	{
		count_type vertexNum_up = gridDim.x * blockDim.x;

		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < vertexNum_up; threadId += vertexNum_up)
		{
			offset_type wordNum_chunk = common[5];// 每个chunk拥有的word数
			count_type vertexNum_chunk = wordNum_chunk * 64; // 每个chunk拥有的顶点数
			count_type offset_index = threadId / vertexNum_chunk;//当前线程的偏移量所在的位置
			count_type offset = offload_offset[offset_index];// 单前线程的偏移量;

			WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };
			//CTA::Thread_data_type thread_data = { 0, 0, 0 };

			//每个线层先确定自己要访问的active的位置
			count_type wordIndex = WORD_OFFSET(threadId);
			count_type wordMod = WORD_MOD(threadId);
			uint64_t blockData_64 = offload_data[wordIndex];

			if ((blockData_64 >> wordMod) & 1)
			{
				// 需要注意的是： offload_offset的基本单位word, 所以一定记得乘以64
				count_type chunkMod = threadId % vertexNum_chunk;
				vertex_id_type vertexId = offset * 64 + chunkMod; //  offset * 64 + chunkMod

				thread_data.offset_start = csr_offset[vertexId];
				thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
				thread_data.msg = vertexValue[vertexId];	

			}// 千万注意： 此处的 "}" 不能包住下面的代码, 否则就是：error 4 unspecified launch failure 以及 一晚上的辛苦

			WarpDonate::schedule(thread_data,
			//CTA::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					vertex_id_type dest = csr_dest[edge_cur];
					edge_data_type weight = csr_weight[edge_cur];

					if (msg + weight < vertexValue[dest])
					{
						atomicMin((vertexValue + dest), msg + weight);
					}
				}
			);	
		}
	}// end of func [sssp_hybird_device_pcie_opt]



	/***********************************************************
	 * Func: GraphHybird_opt 对应的SSSP算法的 Device
	 *
	 * [graphHybird_opt.common_device]
	 * [graphHybird_opt.offload_offset_device]
	 * [graphHybird_opt.offload_data_device]
	 * [graphHybird_opt.csr_offset_device]
	 * [graphHybird_opt.csr_dest_device]
	 * [graphHybird_opt.csr_weight_device]
	 * [graphHybird_opt.vertexValue_device]
	 ***********************************************************/
	template<typename GraphHybird_opt>
	void sssp_hybird_pcie_opt(GraphHybird_opt& graphHybird_opt, const count_type nBlock)
	{
		sssp_hybird_device_pcie_opt << <nBlock, BLOCKSIZE >> > (
			graphHybird_opt.common_device, graphHybird_opt.offload_offset_device, graphHybird_opt.offload_data_device,
			graphHybird_opt.csr_offset_device, graphHybird_opt.csr_dest_device, graphHybird_opt.csr_weight_device, graphHybird_opt.vertexValue_device);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}// end of namespace [SSSP_SPACE]















namespace WCC_SPACE
{


	/****************************************************************************************************
	 *                                    【WORKLIST_MODEL】
	 * common[0]: worklist_count; common[1]: worklist_size; common[2]: vertexNum; common[3]: edgeNum
	 ****************************************************************************************************/
	__global__ void wcc_worklist_model_device(
		offset_type* common, vertex_id_type* worklist_in, vertex_id_type* worklist_out,
		countl_type* csr_offset, vertex_id_type* csr_dest, vertex_data_type* vertexValue)
	{
		count_type workNum_up = gridDim.x * blockDim.x;
		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < workNum_up; threadId += workNum_up)
		{
			//CTA::Thread_data_type thread_data = { 0, 0, 0 };
			//AllBusy::Thread_data_type thread_data = { 0, 0, 0 };
			WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };

			if (threadId < common[1])//worklist_size
			{
				vertex_id_type vertexId = worklist_in[threadId];
				thread_data.offset_start = csr_offset[vertexId];
				thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
				thread_data.msg = vertexValue[vertexId];
			}

			//CTA::schedule(thread_data,
				//AllBusy::schedule(thread_data,
				WarpDonate::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					vertex_id_type dest = csr_dest[edge_cur];
					if (msg < vertexValue[dest])
					{
						if (msg  < atomicMin((vertexValue + dest), msg))
						{
							uint32_t index = atomicAdd(common, 1); //worklist_count
							worklist_out[index] = dest;		
						}
					}
				}
			);
		}
	}



	template<typename GraphDeviceWorklist>
	void wcc_worklist_model(GraphDeviceWorklist& graphDeviceWorklist, const count_type nBlock)
	{
		wcc_worklist_model_device << <nBlock, BLOCKSIZE >> > (
			graphDeviceWorklist.common_device, graphDeviceWorklist.worklist_device->in(), graphDeviceWorklist.worklist_device->out(),
			graphDeviceWorklist.csr_offset_device, graphDeviceWorklist.csr_dest_device, graphDeviceWorklist.vertexValue_device);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}// end of namespace [WCC_SPACE]




















namespace PR_SPACE
{

	/****************************************************************************************************
	 *                                    【WORKLIST_MODEL】
	 * common[0]: worklist_count; common[1]: worklist_size; common[2]: vertexNum; common[3]: edgeNum
	 ****************************************************************************************************/
	__global__ void pr_worklist_model_device(
		offset_type* common,
		countl_type* csr_offset, vertex_id_type* csr_dest, 
		vertex_data_type* vertexValue, vertex_data_type* vertexValue_pr)
	{
		count_type workNum_up = gridDim.x * blockDim.x;
		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < common[2]; threadId += workNum_up)
		{
			//CTA::Thread_data_type thread_data = { 0, 0, 0 };
			//AllBusy::Thread_data_type thread_data = { 0, 0, 0 };
			WarpDonate::Thread_data_type thread_data = { 0, 0, 0 };
			vertex_id_type vertexId = threadId;

			thread_data.offset_start = csr_offset[vertexId];
			thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
			thread_data.msg = vertexValue[vertexId];

			//CTA::schedule(thread_data,
				//AllBusy::schedule(thread_data,
			WarpDonate::schedule(thread_data,
				[&](offset_type edge_cur, vertex_data_type msg)
				{
					if (thread_data.nbrSize != 0)
					{
						atomicAdd((vertexValue_pr + vertexId), vertexValue[vertexId] / thread_data.nbrSize);
					}				
					
				}
			);
		}
	}


	__global__ void pr_vertex_device(offset_type* common, vertex_data_type* vertexValue_device, vertex_data_type* vertexValue_device_pr)
	{
		count_type workNum_up = gridDim.x * blockDim.x;
		for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < common[2]; threadId += workNum_up)
		{
			vertexValue_device_pr[threadId] = 0.15 / common[2] + (0.85 * vertexValue_device_pr[threadId]);
		}
	}

	

	template<typename GraphDeviceWorklist>
	void pr_worklist_model(GraphDeviceWorklist& graphDeviceWorklist, const count_type nBlock)
	{
		pr_worklist_model_device << <nBlock, BLOCKSIZE >> > (
			graphDeviceWorklist.common_device,
			graphDeviceWorklist.csr_offset_device, graphDeviceWorklist.csr_dest_device,
			graphDeviceWorklist.vertexValue_device, graphDeviceWorklist.vertexValue_device_pr);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());


		pr_vertex_device << <nBlock, BLOCKSIZE >> > (
			graphDeviceWorklist.common_device,
			graphDeviceWorklist.vertexValue_device, graphDeviceWorklist.vertexValue_device_pr
			);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}// end of namespace [PR_SPACE]