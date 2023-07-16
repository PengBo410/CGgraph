#pragma once

#include "Basic/basic_include.cuh"
#include "threadState.hpp"


class TaskSteal {

private:
	ThreadState* threadState;
	dense_bitset thread_working;

public:
	ThreadState::ThreadState_type** thread_state;

	TaskSteal()
	{
		omp_set_dynamic(0);
		threadState = new ThreadState();
		thread_state = threadState->thread_state;

		thread_working.resize(ThreadNum);
	}

	
	TaskSteal(ThreadState* threadState_)
	{
		omp_set_dynamic(0);
		threadState = threadState_;
		thread_state = threadState_->thread_state;
		thread_working.resize(ThreadNum);
	}


    /* ======================================================================================*
	 *                              [twoStage_taskSteal]
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

				enCoderTask(vertexId_current, totalTask_local);//[vertexId_current, vertexId_current + 64)

			}// end of [2.1.Vertex Working]


			/*************************************
			 *   2.2.【VERTEX_STEALING】
			 *************************************/
			thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
			for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
				while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
					size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, 64);
					if (vertexId_current >= thread_state[threadId_help]->end) break;

					enCoderTask(vertexId_current, totalTask_local);//[vertexId_current, vertexId_current + 64)
				}
			}// end of [2.2.VERTEX_STEALING]


			totalWorkloads += totalTask_local;
		}

		return totalWorkloads;
	}




	
#ifdef NUMA_AWARE

	
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
						//EDGE_DONATE
						countl_type nbr_start = graphEnginer.csr_offset_numa[socketId][vertexId_current];
						countl_type nbr_end = graphEnginer.csr_offset_numa[socketId][vertexId_current + 1];
						if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
						{
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
							thread_state[thread_id]->edgeDonate.edge_socket = socketId;

							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//
								if (edge_current >= nbr_end) break;
								countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
								
								local_activeVertices += processEdge(vertexId_current, edge_current, edge_end, socketId, true);
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; 
						}
						else if ((nbr_end - nbr_start) > 0)
						{
							
							local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end, socketId, true); 
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
				count_type socketId_steal = socketId_temp % SocketNum;			
				for (count_type steal_offset = 1; steal_offset <= ThreadPerSocket; steal_offset++) {
					count_type threadId_help = ((thread_id + steal_offset) % ThreadPerSocket) + socketId_steal * ThreadPerSocket;
					if (threadId_help == thread_id) continue;
					count_type socketId_help = getThreadSocketId(threadId_help);
					bool sameSocket = (socketId == socketId_help);
					
					while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
						size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
						if (vertexId_current >= thread_state[threadId_help]->end) break;
						size_t word = graphEnginer.active.in().array[WORD_OFFSET(vertexId_current + graphEnginer.zeroOffset_numa[socketId_help])];
						while (word != 0) {
							if (word & 1) {
								
								countl_type nbr_start = graphEnginer.csr_offset_numa[socketId_help][vertexId_current];
								countl_type nbr_end = graphEnginer.csr_offset_numa[socketId_help][vertexId_current + 1];
								
								if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
								{
									
									thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
									thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
									thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
									thread_state[thread_id]->edgeDonate.edge_socket = socketId_help;
									thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

									while (true)
									{
										size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
										if (edge_current >= nbr_end) break;
										countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
										
										local_activeVertices += processEdge(vertexId_current, edge_current, edge_end, socketId_help, sameSocket);
									}
									thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; 
								}
								else if ((nbr_end - nbr_start) > 0)
								{
									
									local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end, socketId_help, sameSocket); 
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
			size_t edge_working_count = 0;
			thread_working.clear_bit(thread_id);
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
					count_type socketId_steal = socketId_temp % SocketNum;				
					for (count_type steal_offset = 1; steal_offset <= ThreadPerSocket; ++steal_offset) {
						count_type threadId_help = ((thread_id + steal_offset) % ThreadPerSocket) + socketId_steal * ThreadPerSocket;
						if (threadId_help == thread_id) continue;			
						//EDGE_STEALING
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

							
							local_activeVertices += processEdge(vertexId_current, edge_current, edge_end, edge_socket, sameSocket);
						}
					}
					socketId_temp++;
				}// workSteal
			}// end of Stage [1.3 [EDGE_WORKING]]

			//LIKWID_MARKER_STOP("taskSteal");

			activeVertices += local_activeVertices;

		}// end of [parallel for]

		return activeVertices;

	}// end of function [threeStage_taskSteal(...) NUMA-aware]



#else


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
	count_type activeVertices = 0;
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
					//
					offset_type nbr_start = graphEnginer.csr_offset[vertexId_current];
					offset_type nbr_end = graphEnginer.csr_offset[vertexId_current + 1];
					if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
					{
						thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
						thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
						thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

						thread_state[thread_id]->status = ThreadState::EDGE_DONATE;//

						while (true)
						{
							size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);//
							if (edge_current >= nbr_end) break;
							offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
							
							local_activeVertices += processEdge(vertexId_current, edge_current, edge_end);//
						}
						thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; 
					}
					else if ((nbr_end - nbr_start) > 0)
					{
						
						local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end); 
					}
				}
				vertexId_current++;
				word = word >> 1;
			}// end of while word

		}// end of Stage [1.1 VERTEX_WORKING]


		//> 1.2  VERTEX_STEALING
		thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;	
		for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {
			count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
			
			while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
				if (vertexId_current >= thread_state[threadId_help]->end) break;
				size_t word = graphEnginer.active.in().array[WORD_OFFSET(vertexId_current)];
				while (word != 0) {
					if (word & 1) {

						
						offset_type nbr_start = graphEnginer.csr_offset[vertexId_current];
						offset_type nbr_end = graphEnginer.csr_offset[vertexId_current + 1];
						if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
						{
							
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
								if (edge_current >= nbr_end) break;
								offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
								
								local_activeVertices += processEdge(vertexId_current, edge_current, edge_end);
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; //切换回VERTEX_STEALING,此时不允许在窃取其edge
						}
						else if ((nbr_end - nbr_start) > 0)
						{
							
							local_activeVertices += processEdge(vertexId_current, nbr_start, nbr_end); 
						}
					}
					vertexId_current++;
					word = word >> 1;
				}// end of while word
			}
		}// end of Stage [1.2 VERTEX_STEALING]



		//> 1.3 EDGE_WORKING
		size_t edge_working_count = 0;
		thread_working.clear_bit(thread_id);
		vertex_data_type vertexId_current;
		offset_type edge_end;
		size_t edge_current;
		while (!thread_working.empty())
		{
			edge_working_count++;
			for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset) {
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;		
				
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
					
					local_activeVertices += processEdge(vertexId_current, edge_current, edge_end);
				}
			}
		}// end of Stage [1.3 EDGE_WORKING]

		LIKWID_MARKER_STOP("taskSteal");
		activeVertices += local_activeVertices;

	}// end of [parallel for]

	return activeVertices;

}// end of function [threeStage_taskSteal(...) Non-NUMA-aware]


#endif





template<typename Graph_Numa>
count_type taskSteal_numa(Graph_Numa& graphNuma, std::function<count_type(vertex_id_type, countl_type, countl_type, count_type, bool)> push)
{
#ifdef REORDER
	threadState->init_threadState();
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
	count_type activeVertices = 0;

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
			//size_t word = graphNuma.active_numa[socketId].in().array[WORD_OFFSET_(vertexId_current)];
			size_t word = graphNuma.active.in().array[WORD_OFFSET_(vertexId_current + graphNuma.zeroOffset_numa[socketId])];
			//word = word << BIT_OFFSET_(vertexId_current);
			while (word != 0) {
				if (word & 1) {

					
					countl_type nbr_start = graphNuma.csr_offset_numa[socketId][vertexId_current];
					countl_type nbr_end = graphNuma.csr_offset_numa[socketId][vertexId_current + 1];

#ifdef LOCAL_THREAD_DEBUG
					THREAD_LOCAL_socketVertices_single += 1;
					THREAD_LOCAL_socketVertices += 1;
#endif // LOCAL_THREAD_DEBUG
					
					local_activeVertices += push(vertexId_current, nbr_start, nbr_end, socketId, true);
				}
				vertexId_current++;
				word = word >> 1;
			}// end of while word
		}

#ifdef LOCAL_THREAD_DEBUG
		THREAD_LOCAL_vertexWorkingTime_single = threadTime_single.current_time_millis();
		threadTime_single.start();
#endif

		//【------------------------STEALING------------------------】
		thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
		for (int t_offset = 1; t_offset < ThreadNum; t_offset++) {
			int threadId_help = (thread_id + t_offset) % ThreadNum;
			while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
				if (vertexId_current >= thread_state[threadId_help]->end) break;
				count_type socketId_help = getThreadSocketId(threadId_help);
				bool sameSocket = (socketId == socketId_help);
				size_t word = graphNuma.active.in().array[WORD_OFFSET(vertexId_current + graphNuma.zeroOffset_numa[socketId_help])];
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

						countl_type nbr_start = graphNuma.csr_offset_numa[socketId_help][vertexId_current];
						countl_type nbr_end = graphNuma.csr_offset_numa[socketId_help][vertexId_current + 1];

						
						local_activeVertices += push(vertexId_current, nbr_start, nbr_end, socketId_help, sameSocket);
					}
					vertexId_current++;
					word = word >> 1;
				}// end of while word
			}
		}// 

		activeVertices += local_activeVertices;//

#ifdef LOCAL_THREAD_DEBUG
		THREAD_LOCAL_stealingTime_single = threadTime_single.current_time_millis();
		THREAD_LOCAL_processTime += threadTime.current_time_millis();
#endif

	}// end of [parallel for]

	return activeVertices;
}// end of func taskSteal_numa(...)




	template<typename Reorder>
	void processEveryEdge(Reorder& reorder, std::function<void(vertex_id_type, offset_type, offset_type)> reOrderTask)
	{
		
		allocateTaskForThread<count_type>(reorder.vertexNum, 64);
		thread_working.fill();


		omp_parallel
		{
			count_type thread_id = omp_get_thread_num();
			
		   
			while (true) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, 64);
				if (vertexId_current >= thread_state[thread_id]->end) break;
				count_type vertexId_end = (vertexId_current + 64 >= thread_state[thread_id]->end) ?
					thread_state[thread_id]->end : (vertexId_current + 64);
				for (; vertexId_current < vertexId_end; vertexId_current++)
				{
				
					offset_type nbr_start = reorder.csr_offset[vertexId_current];
					offset_type nbr_end = reorder.csr_offset[vertexId_current + 1];
					if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
					{
						thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
						thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
						thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
						thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

						while (true)
						{
							size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
							if (edge_current >= nbr_end) break;
							offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);

							reOrderTask(vertexId_current, edge_current, edge_end);
						}

						thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; 
					}
					else
					{
						
						reOrderTask(vertexId_current, nbr_start, nbr_end); 
					}
				}//end of 64 vertices
			}// end of [2.1.Vertex Working]



			/*************************************
			 *   2.2.【VERTEX_STEALING】
			 *************************************/
			thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
			for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
				while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
					size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
					if (vertexId_current >= thread_state[threadId_help]->end) break;
					count_type vertexId_end = (vertexId_current + VERTEXSTEAL_CHUNK >= thread_state[threadId_help]->end) ?
						thread_state[threadId_help]->end : (vertexId_current + VERTEXSTEAL_CHUNK);
					for (; vertexId_current < vertexId_end; vertexId_current++)
					{
						
						offset_type nbr_start = reorder.csr_offset[vertexId_current];
						offset_type nbr_end = reorder.csr_offset[vertexId_current + 1];
						if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
						{
							
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
								if (edge_current >= nbr_end) break;
								offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);

								
								reOrderTask(vertexId_current, edge_current, edge_end);
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; 
						}
						else
						{						
							reOrderTask(vertexId_current, nbr_start, nbr_end); 
						}
					}//end of 64 vertices
				}
			}// end of [2.2.VERTEX_STEALING]



			/*************************************
			 *   2.3.【EDGE WORKING】
			 *************************************/
			thread_working.clear_bit(thread_id);
			vertex_id_type vertexId_current;
			offset_type edge_end;
			size_t edge_current;
			while (!thread_working.empty())
			{
				for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {
					count_type threadId_help = (thread_id + steal_offset) % ThreadNum;	
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

						reOrderTask(vertexId_current, edge_current, edge_end);
					}
				}
			}// end of [2.3.EDGE WORKING]

		}// end of [omp_parallel]
		
	}// end of func processEveryEdge(...)



	template<typename Graph>
	count_type threeSatge_noNUMA(Graph& graph, std::function<count_type(vertex_id_type, offset_type, offset_type)> push)
	{
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


		count_type activeVertices = 0;
#pragma omp parallel reduction(+:activeVertices)
		{

			count_type thread_id = omp_get_thread_num();
			count_type local_activeVertices = 0;

			
			while (true) {
				size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
				if (vertexId_current >= thread_state[thread_id]->end) break;
				size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
				while (word != 0) {
					if (word & 1) {
						
						offset_type nbr_start = graph.csr_offset[vertexId_current];
						offset_type nbr_end = graph.csr_offset[vertexId_current + 1];
						if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
						{
							thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
							thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
							thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

							thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

							while (true)
							{
								size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
								if (edge_current >= nbr_end) break;
								offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);							
								local_activeVertices += push(vertexId_current, edge_current, edge_end);
							}

							thread_state[thread_id]->status = ThreadState::VERTEX_WORKING; 
						}
						else if ((nbr_end - nbr_start) > 0)
						{
							local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
						}
					}
					vertexId_current++;
					word = word >> 1;
				}// end of while word

			}


			/*************************************
			 *   2.2.【VERTEX_STEALING】
			 *************************************/
			thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;	
			for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
				while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
					size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
					if (vertexId_current >= thread_state[threadId_help]->end) break;
					size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
					while (word != 0) {
						if (word & 1) {
							offset_type nbr_start = graph.csr_offset[vertexId_current];
							offset_type nbr_end = graph.csr_offset[vertexId_current + 1];
							//vertex_data_type source_level = graphNuma.vertexValue_numa[socketId_help][vertexId_current];//another
							if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
							{								
								thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
								thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
								thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
								thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

								while (true)
								{
									size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
									if (edge_current >= nbr_end) break;
									offset_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
									local_activeVertices += push(vertexId_current, edge_current, edge_end);
								}

								thread_state[thread_id]->status = ThreadState::VERTEX_STEALING; 
							}
							else if ((nbr_end - nbr_start) > 0)
							{
								local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
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
			size_t edge_working_count = 0;
			thread_working.clear_bit(thread_id);
			vertex_data_type vertexId_current;
			offset_type edge_end;
			size_t edge_current;
			while (!thread_working.empty())
			{
				edge_working_count++;
				for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset) {
					count_type threadId_help = (thread_id + steal_offset) % ThreadNum;			
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
						local_activeVertices += push(vertexId_current, edge_current, edge_end);
					}
				}
			}
			activeVertices += local_activeVertices;

		}// end of omp_parallel

		return activeVertices;

	}// end of func threeSatge_noNUMA(...)




















    template<typename T>
	void allocateTaskForThread(T& workSize, size_t alignSize = 1, bool fillWord = false)
	{
		size_t bitNum = 8 * sizeof(size_t);
		if (fillWord) alignSize = bitNum;
		T taskSize = workSize;
		for (count_type threadId = 0; threadId < ThreadNum; threadId++)
		{			
			if (fillWord && WORD_MOD(taskSize) != 0) taskSize = (taskSize / bitNum + 1) * bitNum;
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