#pragma once

#include <thread>
#include "Basic/basic_include.cuh"
#include "taskSteal.hpp"
#include <math.h>
#include <sys/mman.h>

class Graph_Numa
{
public:
	count_type vertexNum;
	countl_type edgeNum;
	count_type noZeroOutDegreeNum;

	countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;

	//NUMA
	count_type socketNum;
	count_type threadPerSocket;
	count_type threadNum;
	count_type vertexPerSocket; 

	//NUMA
	vertex_id_type* numa_offset;
	countl_type** csr_offset_numa;
	vertex_id_type** csr_dest_numa;
	edge_data_type** csr_weight_numa;
	count_type* vertexNum_numa;
	countl_type* edgeNum_numa;
	count_type* zeroOffset_numa;

	//TaskSteal
	TaskSteal* taskSteal;
	TaskSteal* taskSteal_align64;
#ifdef LOCK
	std::vector<simple_spinlock> push_vlocks;
#endif // LOCK


#ifdef VV_NUMA
	vertex_data_type** vertexValue_numa;//[socketId][vertexValue]
	//PUSH
	Bitamp_NUMA* active_in_numa;//[socketId]
	Bitamp_NUMA* active_out_numa;//[socketId]
#else
	vertex_data_type* vertexValue;
	vertex_data_type* vertexValue_pr;
	//PUSH
	dense_bitset active_in;
	dense_bitset active_out;
#endif
	

	//Page
	count_type PAGENUM = 1;

	//COLD_PACKAGE
#ifdef COLD_PACKAGE
	count_type*** cold_package;//[threadId][socketId][vertexId-perSocket]
	
#endif
	bool is_outDegreeThread = false;
	countl_type* zeroDegree_numa;
	offset_type* outDegree;

public:
	inline count_type getVertexSocketId(vertex_id_type vertexId)
	{
		count_type socketId = 0;
		for (; socketId < socketNum; socketId++)
		{
			if (vertexId < numa_offset[socketId + 1])
			{
				break;
			}
		}
		return socketId;
	}


public:
	/* ======================================================================== *
	 *                                    【con】
	 * ======================================================================== */
	Graph_Numa(const CSR_Result_type& csrResult, ThreadState* threadState_reorder_, countl_type* zeroDegree_numa_) :
		vertexNum(0),
		edgeNum(0)
	{
		
		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;

#ifdef ALIGN_PAGESIZE
		PAGENUM = PAGESIZE / sizeof(vertex_id_type);
#endif
		
		is_outDegreeThread = true;
		taskSteal = new TaskSteal(threadState_reorder_);
		zeroDegree_numa = zeroDegree_numa_;
		
		
		initGraphNuma();
		Msg_info("initGraphNuma-Thread finish");

		//outDegree
		outDegree = new offset_type[vertexNum];
		for (size_t vertexId = 0; vertexId < vertexNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}

		
		partitionGraphByNuma();
		Msg_info("partitionGraphByNuma finsih");


		allocate_vertexValueAndActive();
		Msg_info("allocate_vertexValueAndActive finish");
	}

	Graph_Numa(const CSR_Result_type& csrResult) :
		vertexNum(0),
		edgeNum(0)
	{

		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;

#ifdef ALIGN_PAGESIZE
		PAGENUM = PAGESIZE / sizeof(vertex_id_type);
#endif

		taskSteal = new TaskSteal();

		initGraphNuma();
		Msg_info("initGraphNuma finish");

		//outDegree
		outDegree = new offset_type[vertexNum];
		for (size_t vertexId = 0; vertexId < vertexNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}

		//NUMA
		partitionGraphByNuma();
		Msg_info("partitionGraphByNuma finsih");	

		// vertexValue和active
		allocate_vertexValueAndActive();
		Msg_info("allocate_vertexValueAndActive finsih");
	}




	double graphProcess(Algorithm_type algorithm, vertex_id_type root = 0)
	{
		//Algorithm
		init_algorithm(algorithm, root);

		count_type ite = 0;
		double processTime = 0.0;
		count_type activeNum = 0;
		timer iteTime;

		/*for (size_t i = 0; i < 8; i++)
		{
			printf("begin：vertexValue[%d] = %f, vertexValue[%d] = %f \n", i, vertexValue[i], i, vertexValue_pr[i]);
		}*/


		do
		{
			ite++;

#ifdef LOCAL_THREAD_DEBUG
			init_THREAD_LOCAL_socket_single();
#endif
			if (Algorithm_type::PR != algorithm)
			{
				clear_active_out(algorithm);
			}
			else
			{
				omp_parallel_for(size_t vertexId = 0; vertexId < vertexNum; vertexId++)
				{
					vertexValue_pr[vertexId] = (edge_data_type)0.0;
				}
			}


			timer single_time;
			activeNum = taskSteal->threeStage_taskSteal<Graph_Numa>(*this,
			//activeNum = taskSteal->taskSteal_numa<Graph_Numa>(*this,
				[&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end, count_type socketId, bool sameSocket)
				{
					if (Algorithm_type::BFS == algorithm)
					{
						return BFS_SPACE::bfs_numa<Graph_Numa>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
					}
					else if (Algorithm_type::SSSP == algorithm)
					{
						return SSSP_SPACE::sssp_numa<Graph_Numa>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
					}
					else if (Algorithm_type::CC == algorithm)
					{
						return CC_SPACE::cc_numa<Graph_Numa>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
					}
					else if (Algorithm_type::PR == algorithm)
					{

						return PR_SPACE::pr_numa<Graph_Numa>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);							
					}
					else
					{
						assert_msg(false, "taskSteal_numa error");
                        return static_cast<count_type>(0);
					}		
				}
            );


			if (Algorithm_type::PR == algorithm)
			{
				count_type active_pr = 1;
#pragma omp parallel for 
				for (size_t vertexId = 0; vertexId < vertexNum; vertexId++)
				{

					vertexValue_pr[vertexId] = beta / vertexNum + (alpha * vertexValue_pr[vertexId]);

				}
				activeNum = active_pr;
			}

			

			std::cout << "\t[Single]：(" << ite << "), time = (" << single_time.current_time_millis()
				<< " ms), active = (" << activeNum << ")" << std::endl;
#ifdef LOCAL_THREAD_DEBUG
			print_THREAD_socket_single(threadNum, ite);

			logstream(LOG_INFO) << "\t[Single]：(" << ite << "), time = (" << single_time.current_time_millis()
				<< " ms), active = (" << activeNum << ")" << std::endl;
#endif 
		
			if (algorithm == Algorithm_type::PR)
			{
				if ((activeNum == 0) || ite >= iteN)
				{
					processTime = iteTime.current_time_millis();
					std::cout << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
						<< ", time: " << std::setw(6) << processTime << " (ms)" << std::endl;
#ifdef LOCAL_THREAD_DEBUG
					logstream(LOG_INFO) << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
						<< ", time: " << std::setw(6) << processTime << " (ms)" << std::endl;
#endif
					break;
				}
			}
			else
			{
				if (activeNum == 0)
				{
					processTime = iteTime.current_time_millis();
					std::cout << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
						<< ", time: " << std::setw(6) << processTime << " (ms)" << std::endl;
#ifdef LOCAL_THREAD_DEBUG
					logstream(LOG_INFO) << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
						<< ", time: " << std::setw(6) << processTime << " (ms)" << std::endl;
#endif
					break;
				}
			}


			

			//swap
#ifdef VV_NUMA
			for (size_t socketId = 0; socketId < socketNum; socketId++)
			{
				active_numa[socketId].swap();
			}
#else
			if (algorithm == Algorithm_type::PR) std::swap(vertexValue, vertexValue_pr);
			else   active.swap();
#endif

			

		} while (true);
		
		

		

#ifdef LOCAL_THREAD_DEBUG
		print_THREAD_socket(threadNum);
		print_THREAD_processTime(threadNum);
#endif

		return processTime;
	}


private:

	/* ======================================================================== *
	 *                               【initGraphNuma】
	 * ======================================================================== */
	void initGraphNuma()
	{
		assert_msg(numa_available() != -1, "NUMA can not used");
		assert_msg(sizeof(size_t) == 8, "64 machine");
		if (vertexNum >= UINT32_MAX) printf("[WARNING]: vertexNum >=UINT32_MAX, count_type uint64_t\n");
		if (edgeNum >= UINT32_MAX) printf("[WARNING]: edgeNum >=UINT32_MAX, countl_type uint64_t\n");

		threadNum = ThreadNum;
		socketNum = SocketNum;
		threadPerSocket = ThreadNum / SocketNum;

		omp_parallel_for(int threadId = 0; threadId < threadNum; threadId++)
		{
#ifdef GRAPHNUMA_STAGE_THREE_DEBUG
			int thread_id = omp_get_thread_num();
			int core_id = sched_getcpu();
			int socket_id = getThreadSocketId(threadId);
			logstream(LOG_INFO) << "[" << std::setw(2) << threadId
				<< "]: Thread(" << std::setw(2) << thread_id
				<< ") is running on CPU{" << std::setw(2) << core_id
				<< "}, socketId = (" << std::setw(2) << socket_id
				<< ")" << std::endl;
#endif // STAGE_THREE_DEBUG		
			assert_msg(numa_run_on_node(getThreadSocketId(threadId)) == 0, "numa_run_on_node error");
		}

		//CSR_NUMA
		csr_offset_numa = new countl_type * [socketNum];
		csr_dest_numa = new vertex_id_type * [socketNum];
		csr_weight_numa = new vertex_id_type * [socketNum];
#ifdef VV_NUMA
		vertexValue_numa = new vertex_data_type * [socketNum];
		//PUSH
		active_in_numa = new Bitamp_NUMA[socketNum];
		active_out_numa = new Bitamp_NUMA[socketNum];
		active_numa = new DoubleBuffer<Bitamp_NUMA>[socketNum];
#endif
		vertexNum_numa = new count_type[socketNum];
		edgeNum_numa = new countl_type[socketNum];
		zeroOffset_numa = new count_type[socketNum];

		//taskSteal
		taskSteal_align64 = new TaskSteal();
#ifdef LOCK
		push_vlocks.resize(vertexNum);
#endif // LOCK		
	}



	/* ======================================================================== *
	 *                          【partitionGraphByNuma】
	 * ======================================================================== */
	void partitionGraphByNuma()
	{
		
		numa_offset = new vertex_id_type[socketNum + 1];
		numa_offset[0] = 0;
#ifdef VV_NUMA
		vertexPerSocket = vertexNum / socketNum / PAGENUM * PAGENUM; 
#else
		vertexPerSocket = vertexNum / socketNum / 64 * 64; 
#endif	
		for (count_type socketId = 1; socketId < socketNum; socketId++)
		{
			numa_offset[socketId] = numa_offset[socketId - 1] + vertexPerSocket;
		}
		numa_offset[socketNum] = vertexNum;

		
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			count_type vertexs_numa = numa_offset[socketId + 1] - numa_offset[socketId];
			countl_type edges_numa = csr_offset[numa_offset[socketId + 1]] - csr_offset[numa_offset[socketId]];

			csr_offset_numa[socketId] = (countl_type*)numa_alloc_onnode((vertexs_numa + 1) * sizeof(countl_type), socketId);
			countl_type offset = csr_offset[numa_offset[socketId]];
			for (count_type i = 0; i < (vertexs_numa + 1); i++)
			{
				csr_offset_numa[socketId][i] = csr_offset[numa_offset[socketId] + i] - offset;
			}

			csr_dest_numa[socketId] = (vertex_id_type*)numa_alloc_onnode((edges_numa) * sizeof(vertex_id_type), socketId);
			memcpy(csr_dest_numa[socketId], csr_dest + csr_offset[numa_offset[socketId]], edges_numa * sizeof(vertex_id_type));

			csr_weight_numa[socketId] = (edge_data_type*)numa_alloc_onnode((edges_numa) * sizeof(edge_data_type), socketId);
			memcpy(csr_weight_numa[socketId], csr_weight + csr_offset[numa_offset[socketId]], edges_numa * sizeof(edge_data_type));

			
			vertexNum_numa[socketId] = vertexs_numa;
			edgeNum_numa[socketId] = edges_numa;
			zeroOffset_numa[socketId] = numa_offset[socketId];
		}

		
		delete[] csr_offset;
		delete[] csr_dest;
		delete[] csr_weight;

#ifdef GRAPHNUMA_STAGE_THREE_DEBUG
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			Msg_info("Socket[%u] vertices：(%9u), edges：(%10u)", socketId,
				(numa_offset[socketId + 1] - numa_offset[socketId]), edgeNum_numa[socketId]);
		}
#endif // GRAPHNUMA_STAGE_THREE_DEBUG

#ifdef GRAPHNUMA_STAGE_THREE_DEBUG
		//debugCsrArray();
		debugSocketArray();
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			check_array_numaNode(csr_offset_numa[socketId], vertexNum_numa[socketId], socketId);
			check_array_numaNode(csr_dest_numa[socketId], edgeNum_numa[socketId], socketId);
			check_array_numaNode(csr_weight_numa[socketId], edgeNum_numa[socketId], socketId);
		}
		Msg_check("(csr_offset_numa, csr_dest_numa, csr_weight_numa) check_array_numaNode 通过");
#endif
	}

	/* ======================================================================== *
	 *                   【allocate_vertexValueAndActive】
	 * ======================================================================== */
	void allocate_vertexValueAndActive()
	{	
#ifdef VV_NUMA
		std::vector<count_type> taskSize_vec(socketNum);
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			taskSize_vec[socketId] = vertexNum_numa[socketId];
		}
		taskSteal_align64->allocateTaskForThread_numa<count_type>(taskSize_vec, 64, true);
		clear_vector<count_type>(taskSize_vec);
#else
		taskSteal_align64->allocateTaskForThread<count_type>(vertexNum,64,true);
#endif // VV_NUMA

		

#ifdef VV_NUMA
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			count_type vertexNum_numa_ = vertexNum_numa[socketId];
#ifdef MMAP
			vertexValue_numa[socketId] = mmap_alloc(vertexNum_numa_, socketId);
#else
			vertexValue_numa[socketId] = (vertex_data_type*)numa_alloc_onnode((vertexNum_numa_) * sizeof(vertex_data_type), socketId);
#endif
			active_in_numa[socketId].setSize(vertexNum_numa_, socketId);
			active_out_numa[socketId].setSize(vertexNum_numa_, socketId);
			active_numa[socketId].setDoubleBuffer(active_in_numa[socketId], active_out_numa[socketId]);
		}
#else
#ifdef PIN_MEM
		CUDA_CHECK(cudaMallocHost((void**)&(vertexValue), (vertexNum) * sizeof(vertex_data_type)));
		CUDA_CHECK(cudaMallocHost((void**)&(vertexValue_pr), (vertexNum) * sizeof(vertex_data_type)));
#else
		vertexValue = new vertex_data_type[vertexNum];
#endif
		active_in.resize(vertexNum);
		active_out.resize(vertexNum);
		active.setDoubleBuffer(active_in, active_out);
#endif

	}

	vertex_data_type* mmap_alloc(count_type& size, count_type& socketId)
	{
		char* array = (char*)mmap(NULL, sizeof(vertex_data_type) * size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		assert_msg(array != NULL, "mmap error");
		numa_tonode_memory(array, sizeof(vertex_data_type) * size, socketId);
		return (vertex_data_type*)array;
	}

	/* ======================================================================== *
	 *                   【init_vertexValueAndActive】
	 * ======================================================================== */
	void init_vertexValueAndActive()
	{
#ifdef VV_NUMA
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			count_type vertexNum_numa_ = vertexNum_numa[socketId];
			for (count_type i = 0; i < vertexNum_numa_; i++) vertexValue_numa[socketId][i] = VertexValue_MAX;
			active_numa[socketId].in().clear_memset();
			active_numa[socketId].out().clear_memset();
		}
#else	

		for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = VertexValue_MAX;
		active.in().clear_memset();
		active.out().clear_memset();
#endif

	}

	/* ======================================================================== *
	 *                         【init_algorithm】
	 * ======================================================================== */
	void init_algorithm(Algorithm_type algorithm, vertex_id_type root)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			init_vertexValueAndActive();

				
#ifdef VV_NUMA
			count_type rootSocket = getVertexSocketId(root);
			vertex_id_type rootOffset = zeroOffset_numa[rootSocket];
			vertexValue_numa[rootSocket][root - rootOffset] = 0;
			active_numa[rootSocket].in().set_bit(root - rootOffset);
			//printf("root = %d, rootSocket = %u, root - rootOffset = %u\n", root, rootSocket, root - rootOffset);
#else
			vertexValue[root] = 0;
			active.in().set_bit(root);
#endif					
		}
		else if ((Algorithm_type::CC == algorithm))
		{
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = i;
			active.in().fill();
			active.out().clear_memset();
		}
		else if ((Algorithm_type::PR == algorithm))
		{
			for (count_type i = 0; i < vertexNum; i++)
			{
				vertexValue[i] = (vertex_data_type)(1.0 / vertexNum);
				vertexValue_pr[i] = (vertex_data_type)0.0;
			}
			active.in().fill();
			active.out().clear_memset();
		}
		else
		{
			assert_msg(false, "init_algorithm error");
		}

#ifdef LOCAL_THREAD_DEBUG
		init_THREAD_LOCAL_socket();
#endif
	}


	/* ======================================================================== *
	 *                         【clear active_out_numa】
	 * ======================================================================== */
	void clear_active_out(Algorithm_type algorithm)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm) || (Algorithm_type::CC == algorithm))
		{
			omp_parallel
			{
#ifdef LOCAL_THREAD_DEBUG
				timer active_time;
#endif
			count_type threadId = omp_get_thread_num();
			size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
			size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);
#ifdef VV_NUMA			
			count_type socketId = getThreadSocketId(threadId);
			memset(active_numa[socketId].out().array + cur, 0, sizeof(size_t) * (end - cur));
#else
			memset(active.out().array + cur, 0, sizeof(size_t) * (end - cur));
#endif
#ifdef LOCAL_THREAD_DEBUG
				THREAD_LOCAL_activeClear += active_time.current_time_millis();
#endif
			}
		}
		else if (Algorithm_type::PR == algorithm)
		{
			omp_parallel
			{
#ifdef LOCAL_THREAD_DEBUG
				timer active_time;
#endif
			count_type threadId = omp_get_thread_num();
			size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
			size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);
#ifdef VV_NUMA			
			count_type socketId = getThreadSocketId(threadId);
			memset(active_numa[socketId].out().array + cur, 0, sizeof(size_t) * (end - cur));
#else
			memset(active.out().array + cur, 0, sizeof(size_t) * (end - cur));
#endif
#ifdef LOCAL_THREAD_DEBUG
				THREAD_LOCAL_activeClear += active_time.current_time_millis();
#endif
			}
		}
		else
		{
			assert_msg(false, "clear_active_out error");
		}
	}



	/* ======================================================================== *
	 *                      【check_array_numaNode】
	 * ======================================================================== */
    template<typename T>
	void check_array_numaNode(T* adr, size_t size, size_t node)
	{
		//move_pages Page align
		size_t pageNum_temp = PAGESIZE / sizeof(vertex_id_type);
		size_t len = (size + pageNum_temp - 1) / pageNum_temp;
		for (size_t i = 0; i < len; i++)
		{
			int checkNumaNode = getAdrNumaNode((adr + (i * pageNum_temp)));
			assert_msg(checkNumaNode == node, "getAdrNumaNode error");
		}
	}



	/* ======================================================================== *
	 *                                【DEBUG】
	 * ======================================================================== */
	void debugCsrArray()
	{
		std::stringstream ss;
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			count_type vertexs_numa = vertexNum_numa[socketId];
			countl_type edges_numa = edgeNum_numa[socketId];
			ss << "\nsocket[" << socketId << "]: vertexs_numa = (" << vertexs_numa << "), edges_numa = (" << edges_numa << ")\n";

			ss << "csr_offset_numa :\n";
			for (count_type i = 0; i < (vertexs_numa + 1); i++)
			{
				ss << std::setw(3) << csr_offset_numa[socketId][i] << " ";//TODO
			}

			ss << "\ncsr_dest_numa :\n";
			for (countl_type i = 0; i < (edges_numa); i++)
			{
				ss << std::setw(3) << csr_dest_numa[socketId][i] << " ";
			}

			ss << "\ncsr_weight_numa :\n";
			for (countl_type i = 0; i < (edges_numa); i++)
			{
				ss << std::setw(3) << csr_weight_numa[socketId][i] << " ";
			}
			ss << "\n\n";
		}

		logstream(LOG_INFO) << ss.str() << std::endl << std::endl;
	}

	void debugSocketArray()
	{
		std::stringstream ss;
		ss << "zeroOffset_numa:\n";
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			ss << std::setw(9) << zeroOffset_numa[socketId] << " ";
		}

		ss << "\nvertexNum_numa:\n";
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			ss << std::setw(9) << vertexNum_numa[socketId] << " ";
		}

		ss << "\nedgeNum_numa:\n";
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			ss << std::setw(9) << edgeNum_numa[socketId] << " ";
		}

		logstream(LOG_INFO) << ss.str() << std::endl << std::endl;
	}


};//end of class [Graph_Numa]