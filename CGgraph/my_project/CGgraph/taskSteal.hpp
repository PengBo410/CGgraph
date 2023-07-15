#pragma once

#include "Basic/basic_include.cuh"
#include "threadState.hpp"


class TaskSteal {

private:
	ThreadState* threadState;
	dense_bitset thread_working;//表示正在工作的线程

public:
	ThreadState::ThreadState_type** thread_state;

	TaskSteal()
	{
		omp_set_dynamic(0);
		threadState = new ThreadState();
		thread_state = threadState->thread_state;

		thread_working.resize(ThreadNum);
	}

	//给定了分配好的线程工作量
	TaskSteal(ThreadState* threadState_)
	{
		omp_set_dynamic(0);
		threadState = threadState_;
		thread_state = threadState_->thread_state;
		thread_working.resize(ThreadNum);
	}


    /* ======================================================================================*
	 *                              [twoStage_taskSteal]
	 * Func: 只是满足taskSteal,用户有更多的自定义空间
	 * ======================================================================================*/
	template<typename result_type = size_t, typename common_type = size_t>
	result_type twoStage_taskSteal(common_type work, std::function<void(size_t&, result_type&)> enCoderTask, size_t alignSize = 64)
	{
		allocateTaskForThread<common_type>(work, alignSize);

		result_type totalWorkloads = 0;
#pragma omp parallel reduction(+:totalWorkloads)
		{
			size_t thread_id = omp_get_thread_num();
			result_type totalTask_local = 0;

			/*************************************
			 *   2.1.【VERTEX_WORKING】
			 *************************************/
			while (true) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, 64);
				if (vertexId_current >= thread_state[thread_id]->end) break;

				enCoderTask(vertexId_current, totalTask_local);//[vertexId_current, vertexId_current + 64)都需要自定义处理

			}// end of [2.1.Vertex Working]


			/*************************************
			 *   2.2.【VERTEX_STEALING】
			 *************************************/
			thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
			for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {//线程窃取的偏移量
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程
				while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
					size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, 64);
					if (vertexId_current >= thread_state[threadId_help]->end) break;

					enCoderTask(vertexId_current, totalTask_local);//[vertexId_current, vertexId_current + 64)都需要自定义处理
				}
			}// end of [2.2.VERTEX_STEALING]


			totalWorkloads += totalTask_local;
		}

		return totalWorkloads;
	}




	
#ifdef NUMA_AWARE

	/* ======================================================================================*
	 *                           [ThreeStage_taskSteal - NUMA-AWARE]
	 * Func: 处理edge,但并不是所有的edge，根据vertex的bitmap来决定要处理的边，对应taskSteal_numa_v4
	 *       NUMA_AWARE; 三阶段; 无锁
	 *       必须为64 chunk
	 * [GraphEnginer&]
	 *       调用的Class
	 * [std::function<count_type(vertex_id_type, offset_type, offset_type, count_type, bool)]
	 *       处理逻辑：(vertexId_current, edge_current, edge_end, socketId, sameSocket)
	 *                返回值为：activeVertex
	 * ======================================================================================*/
	template<typename GraphEnginer>
	count_type threeStage_taskSteal(GraphEnginer& graphEnginer, /*count_type workloadSize*/
		std::function <count_type(vertex_id_type, countl_type, countl_type, count_type, bool)> processEdge
	){
		//> Partition The Workload Among The Threads
		for (count_type threadId = 0; threadId < ThreadNum; threadId++)
		{
			count_type socketId = getThreadSocketId(threadId);
			count_type socketOffset = getThreadSocketOffset(threadId);
			count_type taskSize = graphEnginer.vertexNum_numa[socketId];//workloadSize; // graphEnginer.vertexNum_numa[socketId];
			thread_state[threadId]->cur = (taskSize / ThreadPerSocket) / SUB_VERTEXSET * SUB_VERTEXSET * socketOffset;
			thread_state[threadId]->end = (taskSize / ThreadPerSocket) / SUB_VERTEXSET * SUB_VERTEXSET * (socketOffset + 1);
			thread_state[threadId]->edgeDonate.vertex = 0;
			thread_state[threadId]->edgeDonate.edge_cur = 0;
			thread_state[threadId]->edgeDonate.edge_end = 0;
			thread_state[threadId]->edgeDonate.edge_socket = 0;

			if (socketOffset == (ThreadPerSocket - 1)) thread_state[threadId]->end = taskSize;

			thread_state[threadId]->status = ThreadState::VERTEX_WORKING;				
		}
		thread_working.fill();


		//> TaskSteal
		count_type activeVertices = 0; // Total Active Vertices
	#pragma omp parallel reduction(+: activeVertices)
		{
			//LIKWID_MARKER_START("taskSteal");

			count_type thread_id = omp_get_thread_num();
			count_type socketId = getThreadSocketId(thread_id);
			count_type socketId_temp = socketId;
			count_type local_activeVertices = 0;


			//> 1.1 [ VERTEX_WORKING ]
			while (true) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
				if (vertexId_current >= thread_state[thread_id]->end) break;
				size_t word = graphEnginer.active.in().array[WORD_OFFSET(vertexId_current + graphEnginer.zeroOffset_numa[socketId])];
				while (word != 0) {
					if (word & 1) {
						//依据当前顶点的degree来判断是否开启EDGE_DONATE
						countl_type nbr_start = graphEnginer.csr_offset_numa[socketId][vertexId_current];
						countl_type nbr_end = graphEnginer.csr_offset_numa[socketId][vertexId_current + 1];
						if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
						{
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
							thread_state[thread_id]->edgeDonate.edge_socket = socketId;

							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
								if (edge_current >= nbr_end) break;
								countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
								//处理逻辑
								local_activeVertices += processEdge(vertexId_current, edge_current, edge_end, socketId, true);//自己负责一部分边
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; //切换回VERTEX_WORKING,此时不允许在窃取其edge
						}
						else if ((nbr_end - nbr_start) > 0)//为0的不调用
						{
							//处理逻辑
							local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end, socketId, true); //自己负责所有的边
						}
					}
					vertexId_current++;
					word = word >> 1;
				}// end of while word

			}// end of Stage 1.1 [ VERTEX_WORKING ]

			//printf("One Stage: [%u]\n", thread_id);

			//> 1.2 [ VERTEX_STEALING ]
			thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
			for (count_type s = 0; s < SocketNum; s++)
			{
				count_type socketId_steal = socketId_temp % SocketNum;//先窃取自己所在的socket				
				for (count_type steal_offset = 1; steal_offset <= ThreadPerSocket; steal_offset++) {//线程窃取的偏移量
					count_type threadId_help = ((thread_id + steal_offset) % ThreadPerSocket) + socketId_steal * ThreadPerSocket;//需要帮助的线程
					if (threadId_help == thread_id) continue;//跳过自己
					count_type socketId_help = getThreadSocketId(threadId_help);//getThreadSocketId获取当前帮助线程的socketId
					bool sameSocket = (socketId == socketId_help);
					//窃取任务
					while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
						size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);//这是窃取的vertexId
						if (vertexId_current >= thread_state[threadId_help]->end) break;
						size_t word = graphEnginer.active.in().array[WORD_OFFSET(vertexId_current + graphEnginer.zeroOffset_numa[socketId_help])];
						while (word != 0) {
							if (word & 1) {
								//依据当前顶点的degree来判断是否开启EDGE_DONATE
								countl_type nbr_start = graphEnginer.csr_offset_numa[socketId_help][vertexId_current];
								countl_type nbr_end = graphEnginer.csr_offset_numa[socketId_help][vertexId_current + 1];
								//vertex_data_type source_level = graphEnginer.vertexValue_numa[socketId_help][vertexId_current];//新增-------------
								if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
								{
									//我们将窃取的vertex的边数据存储到了自己的edge_cur、edge_end、edge_socket中
									thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
									thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
									thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
									thread_state[thread_id]->edgeDonate.edge_socket = socketId_help;
									thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

									while (true)
									{
										size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
										if (edge_current >= nbr_end) break;
										countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
										//处理逻辑
										local_activeVertices += processEdge(vertexId_current, edge_current, edge_end, socketId_help, sameSocket);//自己负责一部分边
									}
									thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; //切换回VERTEX_STEALING,此时不允许在窃取其edge
								}
								else if ((nbr_end - nbr_start) > 0)//为0的不调用
								{
									//处理逻辑
									local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end, socketId_help, sameSocket); //自己负责所有的边
								}
							}
							vertexId_current++;
							word = word >> 1;
						}// end of while word
					}
				}
				socketId_temp++;
			}// end of stage [1.2 [ VERTEX_STEALING ]]

			//printf("Two Stage: [%u]\n", thread_id);

			//> 1.3 [ EDGE_WORKING ]
			size_t edge_working_count = 0;//用于记录循环次数数(越大说明无用功越多)
			thread_working.clear_bit(thread_id);//线程走到这一步，说明前面两步以全部完成，只剩下edge_stealing的工作
			vertex_id_type vertexId_current;
			countl_type edge_end;
			count_type edge_socket;
			size_t edge_current;
			while (!thread_working.empty())
			{
				edge_working_count++;
				socketId_temp = socketId;
				for (count_type s = 0; s < SocketNum; ++s)
				{
					count_type socketId_steal = socketId_temp % SocketNum;//先窃取自己所在的socket				
					for (count_type steal_offset = 1; steal_offset <= ThreadPerSocket; ++steal_offset) {//线程窃取的偏移量
						count_type threadId_help = ((thread_id + steal_offset) % ThreadPerSocket) + socketId_steal * ThreadPerSocket;//需要帮助的线程
						if (threadId_help == thread_id) continue;//跳过自己				
						//EDGE_STEALING (donate 捐献)
						while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
						{
							do
							{
								vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
								edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
								edge_socket = thread_state[threadId_help]->edgeDonate.edge_socket;
								edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
							} while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current, edge_current + EDGESTEAL_CHUNK));


							if (edge_current >= edge_end) break;
							edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
							bool sameSocket = (socketId == edge_socket);

							//处理逻辑
							local_activeVertices += processEdge(vertexId_current, edge_current, edge_end, edge_socket, sameSocket);//自己负责一部分边
						}
					}
					socketId_temp++;
				}// 线程的workSteal
			}// end of Stage [1.3 [EDGE_WORKING]]

			//LIKWID_MARKER_STOP("taskSteal");

			activeVertices += local_activeVertices;//总的活跃顶点数

		}// end of [parallel for]

		return activeVertices;

	}// end of function [threeStage_taskSteal(...) NUMA-aware]



#else

/* ======================================================================================*
 *                              [ThreeStage_taskSteal - Non-NUMA-AWARE]
 * Func: 处理edge,但并不是所有的edge，根据vertex的bitmap来决定要处理的边，对应taskSteal_numa_v4
 *       NO NUMA_AWARE; 三阶段; 无锁
 *       必须为64 chunk
 * [Graph_Numa&]
 *       调用的Class
 * [std::function<count_type(vertex_id_type, offset_type, offset_type)]
 *       处理逻辑：(vertexId_current, edge_current, edge_end)
 *                返回值为：activeVertex
 * ======================================================================================*/
template<typename GraphEnginer>
count_type threeStage_taskSteal(GraphEnginer& graphEnginer, /*count_type workloadSize,*/
	std::function <count_type(vertex_id_type, offset_type, offset_type, count_type, bool)> processEdge //The Last [count_type, bool] Not Used
){
	//> Partition The Workload Among The Threads
	count_type taskSize = graphEnginer.vertexNum;//workloadSize; //graphEnginer.vertexNum;
	for (count_type threadId = 0; threadId < ThreadNum; threadId++)
	{
		thread_state[threadId]->start = (taskSize / ThreadNum) / SUB_VERTEXSET * SUB_VERTEXSET * threadId;
		thread_state[threadId]->cur = thread_state[threadId]->start;
		thread_state[threadId]->end = (taskSize / ThreadNum) / SUB_VERTEXSET * SUB_VERTEXSET * (threadId + 1);
		thread_state[threadId]->edgeDonate.vertex = 0;
		thread_state[threadId]->edgeDonate.edge_cur = 0;
		thread_state[threadId]->edgeDonate.edge_end = 0;
		if (threadId == (ThreadNum - 1)) thread_state[threadId]->end = taskSize;
		thread_state[threadId]->status = ThreadState::VERTEX_WORKING;
	}
	thread_working.fill();

	//> Task-Steal
	count_type activeVertices = 0;//总的活跃顶点数
#pragma omp parallel reduction(+:activeVertices)
	{
		LIKWID_MARKER_START("taskSteal");

		count_type thread_id = omp_get_thread_num();
		count_type local_activeVertices = 0;

		//> 1.1 VERTEX_WORKING
		while (true) {
			size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);

			if (vertexId_current >= thread_state[thread_id]->end) break;
			size_t word = graphEnginer.active.in().array[WORD_OFFSET(vertexId_current)];
			while (word != 0) {
				if (word & 1) {
					//依据当前顶点的degree来判断是否开启EDGE_DONATE
					offset_type nbr_start = graphEnginer.csr_offset[vertexId_current];
					offset_type nbr_end = graphEnginer.csr_offset[vertexId_current + 1];
					if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
					{
						thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
						thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
						thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

						thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

						while (true)
						{
							size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
							if (edge_current >= nbr_end) break;
							offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
							//处理逻辑
							local_activeVertices += processEdge(vertexId_current, edge_current, edge_end);//自己负责一部分边
						}
						thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; //切换回VERTEX_WORKING,此时不允许在窃取其edge
					}
					else if ((nbr_end - nbr_start) > 0)//为0的不调用
					{
						//处理逻辑
						local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end); //自己负责所有的边
					}
				}
				vertexId_current++;
				word = word >> 1;
			}// end of while word

		}// end of Stage [1.1 VERTEX_WORKING]


		//> 1.2  VERTEX_STEALING
		thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;	
		for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {//线程窃取的偏移量
			count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程
			//窃取任务
			while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);//这是窃取的vertexId
				if (vertexId_current >= thread_state[threadId_help]->end) break;
				size_t word = graphEnginer.active.in().array[WORD_OFFSET(vertexId_current)];
				while (word != 0) {
					if (word & 1) {

						//依据当前顶点的degree来判断是否开启EDGE_DONATE
						offset_type nbr_start = graphEnginer.csr_offset[vertexId_current];
						offset_type nbr_end = graphEnginer.csr_offset[vertexId_current + 1];
						if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
						{
							//我们将窃取的vertex的边数据存储到了自己的edge_cur、edge_end、edge_socket中
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
								if (edge_current >= nbr_end) break;
								offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
								//处理逻辑
								local_activeVertices += processEdge(vertexId_current, edge_current, edge_end);//自己负责一部分边
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; //切换回VERTEX_STEALING,此时不允许在窃取其edge
						}
						else if ((nbr_end - nbr_start) > 0)//为0的不调用
						{
							//处理逻辑
							local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end); //自己负责所有的边	
						}
					}
					vertexId_current++;
					word = word >> 1;
				}// end of while word
			}
		}// end of Stage [1.2 VERTEX_STEALING]



		//> 1.3 EDGE_WORKING
		size_t edge_working_count = 0;//用于记录循环次数数(越大说明无用功越多)
		thread_working.clear_bit(thread_id);//线程走到这一步，说明前面两步以全部完成，只剩下edge_stealing的工作
		vertex_data_type vertexId_current;
		offset_type edge_end;
		size_t edge_current;
		while (!thread_working.empty())
		{
			edge_working_count++;
			for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset) {//线程窃取的偏移量
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程//需要帮助的线程			
				//EDGE_STEALING (donate 捐献)
				while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
				{
					do
					{
						vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
						edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
						edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
					} while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current, edge_current + EDGESTEAL_CHUNK));

					if (edge_current >= edge_end) break;
					edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
					//处理逻辑
					local_activeVertices += processEdge(vertexId_current, edge_current, edge_end);//自己负责一部分边
				}
			}
		}// end of Stage [1.3 EDGE_WORKING]

		LIKWID_MARKER_STOP("taskSteal");
		activeVertices += local_activeVertices;//总的活跃顶点数

	}// end of [parallel for]

	return activeVertices;

}// end of function [threeStage_taskSteal(...) Non-NUMA-aware]


#endif





/* **************************************************************************************************************
 * 仅仅包括vertex-working和vertex-steal
 * 1. std::function<count_type(vertex_id_type, countl_type, countl_type, count_type, bool)
 *     中的 countl_type, offset_type本函数用不到,只是为了统一参数个数
 * **************************************************************************************************************/
template<typename Graph_Numa>
count_type taskSteal_numa(Graph_Numa& graphNuma, std::function<count_type(vertex_id_type, countl_type, countl_type, count_type, bool)> push)
{
#ifdef REORDER
	threadState->init_threadState();//REORDER已经分配好了每个线程的task

	//log_threadTask();
#else
	for (count_type threadId = 0; threadId < ThreadNum; threadId++)
	{
		count_type socketId = getThreadSocketId(threadId);
		count_type socketOffset = getThreadSocketOffset(threadId);
		count_type taskSize = graphNuma.vertexNum_numa[socketId];
		thread_state[threadId]->cur = (taskSize / graphNuma.threadPerSocket) / SUB_VERTEXSET * SUB_VERTEXSET * socketOffset;
		thread_state[threadId]->end = (taskSize / graphNuma.threadPerSocket) / SUB_VERTEXSET * SUB_VERTEXSET * (socketOffset + 1);

		if (socketOffset == (graphNuma.threadPerSocket - 1)) thread_state[threadId]->end = taskSize;

		thread_state[threadId]->status = ThreadState::VERTEX_WORKING;
	}
#endif
	count_type activeVertices = 0;//总的活跃顶点数

#pragma omp parallel reduction(+:activeVertices)
	{
#ifdef LOCAL_THREAD_DEBUG
		timer threadTime;
		timer threadTime_single;
#endif
		int thread_id = omp_get_thread_num();
		int socketId = getThreadSocketId(thread_id);
		count_type local_activeVertices = 0;
		/*thread_state[thread_id]->cur = thread_state[thread_id]->start;
		thread_state[thread_id]->status = WORKING;*/

		//【------------------------WORKING------------------------】
		while (true) {
			size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
			if (vertexId_current >= thread_state[thread_id]->end) break;
			//size_t word = graphNuma.active_numa[socketId].in().array[WORD_OFFSET_(vertexId_current)];//被替换
			size_t word = graphNuma.active.in().array[WORD_OFFSET_(vertexId_current + graphNuma.zeroOffset_numa[socketId])];//添加的
			//word = word << BIT_OFFSET_(vertexId_current);
			while (word != 0) {
				if (word & 1) {

					//依据当前顶点的degree来判断是否开启EDGE_DONATE
					countl_type nbr_start = graphNuma.csr_offset_numa[socketId][vertexId_current];//添加的
					countl_type nbr_end = graphNuma.csr_offset_numa[socketId][vertexId_current + 1];//添加的

#ifdef LOCAL_THREAD_DEBUG
					THREAD_LOCAL_socketVertices_single += 1;
					THREAD_LOCAL_socketVertices += 1;
#endif // LOCAL_THREAD_DEBUG
					//处理逻辑
					local_activeVertices += push(vertexId_current, nbr_start, nbr_end, socketId, true);//此处我们先尽量将更多的暴露到外面编写
				}
				vertexId_current++;
				word = word >> 1;
			}// end of while word
		}// 线程自己负责的部分结束

#ifdef LOCAL_THREAD_DEBUG
		THREAD_LOCAL_vertexWorkingTime_single = threadTime_single.current_time_millis();
		threadTime_single.start();
#endif

		//【------------------------STEALING------------------------】
		thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
		for (int t_offset = 1; t_offset < ThreadNum; t_offset++) {//从左往右，适合先小任务，在大任务
			int threadId_help = (thread_id + t_offset) % ThreadNum;//需要帮助的线程
			while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
				if (vertexId_current >= thread_state[threadId_help]->end) break;//应该是break吧
				count_type socketId_help = getThreadSocketId(threadId_help);//getSocketId获取当前帮助线程的socketId
				bool sameSocket = (socketId == socketId_help);
				size_t word = graphNuma.active.in().array[WORD_OFFSET(vertexId_current + graphNuma.zeroOffset_numa[socketId_help])];//添加的
				//size_t word = graphNuma.active_numa[socketId_help].in().array[WORD_OFFSET_(vertexId_current)];//被修改的
				//word = word << BIT_OFFSET_(vertexId_current);
				while (word != 0) {
					if (word & 1) {
#ifdef LOCAL_THREAD_DEBUG
						if (sameSocket) {
							THREAD_LOCAL_socketVertices_single += 1;
							THREAD_LOCAL_socketVertices += 1;
						}
						else {
							THREAD_LOCAL_crossSocketVertices_single += 1;
							THREAD_LOCAL_crossSocketVertices += 1;
						}
#endif // LOCAL_THREAD_DEBUG

						//依据当前顶点的degree来判断是否开启EDGE_DONATE
						countl_type nbr_start = graphNuma.csr_offset_numa[socketId_help][vertexId_current];
						countl_type nbr_end = graphNuma.csr_offset_numa[socketId_help][vertexId_current + 1];

						//处理逻辑
						local_activeVertices += push(vertexId_current, nbr_start, nbr_end, socketId_help, sameSocket);//此处我们先尽量将更多的暴露到外面编写
					}
					vertexId_current++;
					word = word >> 1;
				}// end of while word
			}
		}// 线程的workSteal

		activeVertices += local_activeVertices;//总的活跃顶点数

#ifdef LOCAL_THREAD_DEBUG
		THREAD_LOCAL_stealingTime_single = threadTime_single.current_time_millis();
		THREAD_LOCAL_processTime += threadTime.current_time_millis();
#endif

	}// end of [parallel for]

	return activeVertices;
}// end of func taskSteal_numa(...)



/*======================================================================================*
	 *                              【processEveryEdge】
	 * Func: 用于根据old2new (Rank) 来生成新的Graph Bin File
	 *       之前的generateNewGraph修改得到,更均衡的三状态taskSteal
	 *       NO NUMA_AWARE; 三阶段
	 * [Reorder&] 
	 *       调用的Class
	 * [std::function<void(vertex_id_type, offset_type, offset_type)>]
	 *       处理逻辑 (vertexId_current, edge_current, edge_end)
	 *======================================================================================*/
	template<typename Reorder>
	void processEveryEdge(Reorder& reorder, std::function<void(vertex_id_type, offset_type, offset_type)> reOrderTask)
	{
		/***************************
		 *   1.【分配任务】
		 ***************************/
		allocateTaskForThread<count_type>(reorder.vertexNum, 64);
		thread_working.fill();//启用了edgeDonate


		omp_parallel
		{
			count_type thread_id = omp_get_thread_num();
			
		    /*************************************
	         *   2.1.【VERTEX_WORKING】
	         *************************************/
			while (true) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, 64);
				if (vertexId_current >= thread_state[thread_id]->end) break;
				count_type vertexId_end = (vertexId_current + 64 >= thread_state[thread_id]->end) ?
					thread_state[thread_id]->end : (vertexId_current + 64);
				for (; vertexId_current < vertexId_end; vertexId_current++)
				{
					//依据当前顶点的degree来判断是否开启EDGE_DONATE
					offset_type nbr_start = reorder.csr_offset[vertexId_current];
					offset_type nbr_end = reorder.csr_offset[vertexId_current + 1];
					if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
					{
						thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
						thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
						thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
						thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

						while (true)
						{
							size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
							if (edge_current >= nbr_end) break;
							offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);

							//处理逻辑
							reOrderTask(vertexId_current, edge_current, edge_end);//自己负责一部分边
						}

						thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; //切换回VERTEX_WORKING,此时不允许在窃取其edge
					}
					else
					{
						//处理逻辑
						reOrderTask(vertexId_current, nbr_start, nbr_end); //自己负责所有的边
					}
				}//end of 每64个顶点
			}// end of [2.1.Vertex Working]



			/*************************************
			 *   2.2.【VERTEX_STEALING】
			 *************************************/
			thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
			for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {//线程窃取的偏移量
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程
				while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
					size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
					if (vertexId_current >= thread_state[threadId_help]->end) break;
					count_type vertexId_end = (vertexId_current + VERTEXSTEAL_CHUNK >= thread_state[threadId_help]->end) ?
						thread_state[threadId_help]->end : (vertexId_current + VERTEXSTEAL_CHUNK);
					for (; vertexId_current < vertexId_end; vertexId_current++)
					{
						//依据当前顶点的degree来判断是否开启EDGE_DONATE
						offset_type nbr_start = reorder.csr_offset[vertexId_current];
						offset_type nbr_end = reorder.csr_offset[vertexId_current + 1];
						if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
						{
							//我们将窃取的vertex的边数据存储到了自己的vertex、edge_cur、edge_end中
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
								if (edge_current >= nbr_end) break;
								offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);

								//处理逻辑
								reOrderTask(vertexId_current, edge_current, edge_end);//自己负责一部分边
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; //切换回VERTEX_WORKING,此时不允许在窃取其edge
						}
						else
						{
							//处理逻辑
							reOrderTask(vertexId_current, nbr_start, nbr_end); //自己负责所有的边
						}
					}//end of 每64个顶点
				}
			}// end of [2.2.VERTEX_STEALING]



			/*************************************
			 *   2.3.【EDGE WORKING】
			 *************************************/
			thread_working.clear_bit(thread_id);//线程走到这一步，说明前面两步以全部完成，只剩下edge_stealing的工作
			vertex_id_type vertexId_current;
			offset_type edge_end;
			size_t edge_current;
			while (!thread_working.empty())
			{
				for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {//线程窃取的偏移量
					count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程			
					//EDGE_STEALING (donate 捐献)
					while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
					{
						do
						{
							vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
							edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
							edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
						} while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current, edge_current + EDGESTEAL_CHUNK));


						if (edge_current >= edge_end) break;
						edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);

						//处理逻辑
						reOrderTask(vertexId_current, edge_current, edge_end);//自己负责一部分边
					}
				}
			}// end of [2.3.EDGE WORKING]

		}// end of [omp_parallel]
		
	}// end of func processEveryEdge(...)




	/*======================================================================================*
	 *                              【processEdge_v4】
	 * Func: 处理edge,但并不是所有的edge，根据vertex的bitmap来决定要处理的边，对应taskSteal_numa_v4
	 *       NO NUMA_AWARE; 三阶段; 无锁
	 *       必须为64 chunk
	 * [Graph_Numa&]
	 *       调用的Class
	 * [std::function<count_type(vertex_id_type, offset_type, offset_type)]
	 *       处理逻辑：(vertexId_current, edge_current, edge_end)
	 *                返回值为：activeVertex
	 *======================================================================================*/
	template<typename Graph>
	count_type threeSatge_noNUMA(Graph& graph, std::function<count_type(vertex_id_type, offset_type, offset_type)> push)
	{
		/***************************
		 *   1.【分配任务】
		 ***************************/
		count_type taskSize = graph.vertexNum;
		for (count_type threadId = 0; threadId < ThreadNum; threadId++)
		{
			thread_state[threadId]->start = (taskSize / ThreadNum) / VERTEXSTEAL_CHUNK * VERTEXSTEAL_CHUNK * threadId;
			thread_state[threadId]->cur = thread_state[threadId]->start;
			thread_state[threadId]->end = (taskSize / ThreadNum) / VERTEXSTEAL_CHUNK * VERTEXSTEAL_CHUNK * (threadId + 1);
			thread_state[threadId]->edgeDonate.vertex = 0;
			thread_state[threadId]->edgeDonate.edge_cur = 0;
			thread_state[threadId]->edgeDonate.edge_end = 0;
			if (threadId == (ThreadNum - 1)) thread_state[threadId]->end = taskSize;
			thread_state[threadId]->status = ThreadState::VERTEX_WORKING;
		}
		thread_working.fill();


		count_type activeVertices = 0;//总的活跃顶点数
#pragma omp parallel reduction(+:activeVertices)
		{

			count_type thread_id = omp_get_thread_num();
			count_type local_activeVertices = 0;

			/*************************************
			 *   2.1.【VERTEX_WORKING】
			 *************************************/
			while (true) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
				if (vertexId_current >= thread_state[thread_id]->end) break;
				size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
				while (word != 0) {
					if (word & 1) {
						//依据当前顶点的degree来判断是否开启EDGE_DONATE
						offset_type nbr_start = graph.csr_offset[vertexId_current];
						offset_type nbr_end = graph.csr_offset[vertexId_current + 1];
						if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
						{
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
								if (edge_current >= nbr_end) break;
								offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
								//处理逻辑
								local_activeVertices += push(vertexId_current, edge_current, edge_end);//自己负责一部分边
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; //切换回VERTEX_WORKING,此时不允许在窃取其edge
						}
						else if ((nbr_end - nbr_start) > 0)//为0的不调用
						{
							//处理逻辑
							local_activeVertices += push(vertexId_current, nbr_start, nbr_end); //自己负责所有的边
						}
						//处理逻辑
						//local_activeVertices += push(vertexId_current, nbr_start, nbr_end); //自己负责所有的边
					}
					vertexId_current++;
					word = word >> 1;
				}// end of while word

			}// 线程自己负责的部分结束


			/*************************************
			 *   2.2.【VERTEX_STEALING】
			 *************************************/
			thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;	
			for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {//线程窃取的偏移量
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程
				//窃取任务
				while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
					size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);//这是窃取的vertexId
					if (vertexId_current >= thread_state[threadId_help]->end) break;
					size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
					while (word != 0) {
						if (word & 1) {

							//依据当前顶点的degree来判断是否开启EDGE_DONATE
							offset_type nbr_start = graph.csr_offset[vertexId_current];
							offset_type nbr_end = graph.csr_offset[vertexId_current + 1];
							//vertex_data_type source_level = graphNuma.vertexValue_numa[socketId_help][vertexId_current];//新增-------------
							if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
							{
								//我们将窃取的vertex的边数据存储到了自己的edge_cur、edge_end、edge_socket中
								thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
								thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
								thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
								thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//当前线程允许其余线程窃取其edge

								while (true)
								{
									size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//一次窃取的量
									if (edge_current >= nbr_end) break;
									offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
									//处理逻辑
									local_activeVertices += push(vertexId_current, edge_current, edge_end);//自己负责一部分边
								}

								thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; //切换回VERTEX_STEALING,此时不允许在窃取其edge
							}
							else if ((nbr_end - nbr_start) > 0)//为0的不调用
							{
								//处理逻辑
								local_activeVertices += push(vertexId_current, nbr_start, nbr_end); //自己负责所有的边	
							}
						}
						vertexId_current++;
						word = word >> 1;
					}// end of while word
				}
			}

			/*************************************
			 *   2.3.【EDGE WORKING】
			 *************************************/
			size_t edge_working_count = 0;//用于记录循环次数数(越大说明无用功越多)
			thread_working.clear_bit(thread_id);//线程走到这一步，说明前面两步以全部完成，只剩下edge_stealing的工作
			vertex_data_type vertexId_current;
			offset_type edge_end;
			size_t edge_current;
			while (!thread_working.empty())
			{
				edge_working_count++;
				for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset) {//线程窃取的偏移量
					count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程//需要帮助的线程			
					//EDGE_STEALING (donate 捐献)
					while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
					{
						do
						{
							vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
							edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
							edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
						} while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current, edge_current + EDGESTEAL_CHUNK));


						if (edge_current >= edge_end) break;
						edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
						//处理逻辑
						local_activeVertices += push(vertexId_current, edge_current, edge_end);//自己负责一部分边
					}
				}
			}// 线程的workSteal
			activeVertices += local_activeVertices;//总的活跃顶点数

		}// end of omp_parallel

		return activeVertices;

	}// end of func threeSatge_noNUMA(...)




















    template<typename T>
	void allocateTaskForThread(T& workSize, size_t alignSize = 1, bool fillWord = false)
	{
		size_t bitNum = 8 * sizeof(size_t);
		if (fillWord) alignSize = bitNum;// fillWord下alignSize必须为64
		T taskSize = workSize;
		for (count_type threadId = 0; threadId < ThreadNum; threadId++)
		{			
			if (fillWord && WORD_MOD(taskSize) != 0) taskSize = (taskSize / bitNum + 1) * bitNum; //保证按bitNum对齐
			thread_state[threadId]->start = (taskSize / ThreadNum) / alignSize * alignSize * threadId;
			thread_state[threadId]->cur = thread_state[threadId]->start;
 			thread_state[threadId]->end = (taskSize / ThreadNum) / alignSize * alignSize * (threadId + 1);
			thread_state[threadId]->edgeDonate.vertex = 0;
			thread_state[threadId]->edgeDonate.edge_cur = 0;
			thread_state[threadId]->edgeDonate.edge_end = 0;
			thread_state[threadId]->edgeDonate.edge_socket = 0;
			if (threadId == (ThreadNum - 1)) thread_state[threadId]->end = taskSize;
			thread_state[threadId]->status = ThreadState::VERTEX_WORKING;
		}
		//threadState->log_threadTask();
#ifdef TASK_STEAL_DEBUG
		log_threadTask();
#endif
	}

};