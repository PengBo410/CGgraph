#pragma once



#include "Basic/basic_include.cuh"
#include "Block_partition.cuh"

#include <thread>

#define PIN_MEM

class Block_engine
{
public:

    //> Partition Result
    typedef Block_partition::Block_result_type Block_result_type;
    Block_result_type blockResult;

    //> Graph Info
    count_type vertexNum;
	countl_type edgeNum;
	count_type zeroOutDegreeNum;
	count_type noZeroOutDegreeNum;

    countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;
	offset_type* outDegree;

    //> VertexValue
	vertex_data_type* vertexValue;

     //> Algorithm
	Algorithm_type algorithm;
	count_type ite = 0; // 迭代次数

    //> TaskSteal
	TaskSteal* taskSteal;          // graph process used by host
	TaskSteal* taskSteal_align64;  // clear active

    //> Active
	dense_bitset active_in;  // Fixed_Bitset | dense_bitset
	dense_bitset active_out; // Fixed_Bitset | dense_bitset
	dense_bitset active_steal;
	DoubleBuffer<dense_bitset> active; //active_in与active_out最终封装到DoubleBuffer
    count_type activeNum_device = 0;  // Device 端激活的顶点数
	count_type activeNum_host = 0;    // Host   端激活的顶点数
	count_type activeNum = 0;         // Device + Host 端总共激活的顶点数
	countl_type activeEdgeNum = 0;

    dense_bitset activeBlock_in;
    dense_bitset activeBlock_out;
    DoubleBuffer<dense_bitset> activeBlock; //active_in与active_out最终封装到DoubleBuffer

    // CPU Queue
    std::vector<std::vector<uint64_t>> workloadBlockQueue_2d; 



    /* *********************************************************************************
     * Func: [Constructor]
     * *********************************************************************************/
    Block_engine(
        const CSR_Result_type& csrResult, const Block_result_type& blockResult_,
        Algorithm_type algorithm_
    ):
        vertexNum(0),
        edgeNum(0),
        noZeroOutDegreeNum(0),
        blockResult(blockResult_)
    {
        //Get Graph CSR
		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
        noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;
		zeroOutDegreeNum = vertexNum - noZeroOutDegreeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;
        Msg_info("zeroOutDegreeNum = %u, noZeroOutDegreeNum = %u", (vertexNum - noZeroOutDegreeNum), noZeroOutDegreeNum);

        // algorithm
        algorithm = algorithm_;
        Msg_info("The Algorithm Name: (%s)",getAlgName(algorithm).c_str());
        Msg_info("Has Data BlockNum : %zu", static_cast<uint64_t>(blockResult.bitmap_allBlock.popcount()));

        //> Host, Init Graph Numa
        timer constructTime;
		initGraphNuma();
		Msg_info("Init Graph Numa, Used time: %.2lf (ms)", constructTime.get_time_ms());

        //> Host - Allocate vertexValue and active
		constructTime.start();
		allocate_vertexValueAndActive();
		Msg_info("Allocate-Active: Used time: %.2lf (ms)", constructTime.get_time_ms());

        //> Host - Build outDegree
		constructTime.start();
		get_outDegree();
		Msg_info("Build-outDegree: Used time: %.2lf (ms)", constructTime.get_time_ms());
    }


    double graphProcess(vertex_id_type root = 0)
	{
        if (root >= noZeroOutDegreeNum) { 
			Msg_info("Current root = (%zu), outDegree is 0, exit directly !", static_cast<uint64_t>(root)); 
			return 0.0;
	    }

		// Init Variable and Algorithm
		resetVariable();
		initAlgorithm(root);



        // CPU/GPU Co-execution
		double duration = 0.0;
		timer iteTime_total;
		timer iteTime_single;
		timer tempTime;
		timer heavyTemp_time; //测量Heavy内部的时间

		// Iterative fashion
		iteTime_total.start();

        do{
			iteTime_single.start();

			ite++;

            #ifdef CGGRAPHENGINE_DETAIL_DEBUG
				tempTime.start();
			#endif
			clearActiveOut();
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				logstream(LOG_INFO) << "\t\t1. [Clear Active Out], Used time :" 
					<< std::setw(7) << std::setprecision(2) << std::fixed
					<< tempTime.get_time_ms() << " (ms)" << std::endl;
			#endif
        
        }while(true);
            

        return duration;

    }// end of func [graphProcess(...)]















private:
    /* **********************************************************
	 * Func: Host Func, Init Graph and Numa
	 * **********************************************************/
	void initGraphNuma()
	{
		//threadNum = ThreadNum;

		//Init outDegree
		outDegree = new offset_type[vertexNum];
        memset(outDegree, 0, sizeof(offset_type) * vertexNum);

        //Init taskSteal
		taskSteal = new TaskSteal();
		taskSteal_align64 = new TaskSteal();

	}// end of function [initGraphNuma()]


    /* **************************************************************************
	 * Func: Host Function, Allocate Memory For vertexValue and Active
	 * **************************************************************************/
	void allocate_vertexValueAndActive()
	{

#ifdef PIN_MEM
		CUDA_CHECK(cudaMallocHost((void**)&(vertexValue), (vertexNum) * sizeof(vertex_data_type)));
#else
		vertexValue = new vertex_data_type[vertexNum];
		Msg_info("**********************************************************************");
		Msg_info("*   [Used PIN Memory Tranfer VertexValue Between Host And Device]   *");
		Msg_info("**********************************************************************");
#endif

		active_in.resize(noZeroOutDegreeNum);
		active_out.resize(noZeroOutDegreeNum);
		active_steal.resize(noZeroOutDegreeNum);
		active.setDoubleBuffer(active_in, active_out); // Encapsulated into DoubleBuffer
	}

    /* **********************************************************
	 * Func: Host Function, Build outDegree
	 * **********************************************************/
	void get_outDegree()
	{
		omp_parallel_for(vertex_id_type vertexId = 0; vertexId < noZeroOutDegreeNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}
	}


    /* **********************************************************
	 * Func: Reset Required Variables
	 * **********************************************************/
	void resetVariable()
	{

        taskSteal_align64->allocateTaskForThread<count_type>(noZeroOutDegreeNum, 64, true);
		
		//for(int deviceId = 0; deviceId < useDeviceNum; deviceId++) wakeupDevice[deviceId] = 0; 
		//hybirdComplete = 0; nBlock = 0; usedDevice = false;        // Device
		activeNum_device = 0; activeNum_host = 0; activeNum = 0; activeEdgeNum = 0;  // Active
		ite = 0;                                                                     // Ite                                                          // Workload
	}// end of func [resetVariable()]



    /* **********************************************************
	 * Func: Init The Algorithm
	 *
	 * @param [vertex_id_type root] Some Alg NO Need This param
	 * **********************************************************/
	void initAlgorithm(vertex_id_type root)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			// Host - Active (The noZeroOutDegreeVertices is Relatively Small, We don not use parallel)	

			active.in().clear_memset(); 
			active.out().clear_memset();

			active.in().set_bit(root);
			activeNum = 1;

			//Host - vertexValue
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = VertexValue_MAX;
			vertexValue[root] = 0;

			// for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++)
			// {
			// 	CUDA_CHECK(cudaSetDevice(deviceId));
			// 	memcpy(vertexValue_temp_host[deviceId], vertexValue, vertexNum * sizeof(vertex_data_type));
			// 	CUDA_CHECK(H2D(vertexValue_temp_device[deviceId], vertexValue, vertexNum));
			// }
		}
		else if(Algorithm_type::CC == algorithm)
		{
			// Host - Active

			active.in().clear_memset();
			active.out().clear_memset();

			active.in().fill(); //can used parallel_fill
			activeNum = noZeroOutDegreeNum;

			//Host - vertexValue
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = i;

			// for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++)
			// {
			// 	CUDA_CHECK(cudaSetDevice(deviceId));
			// 	memcpy(vertexValue_temp_host[deviceId], vertexValue, vertexNum * sizeof(vertex_data_type));
			// 	CUDA_CHECK(H2D(vertexValue_temp_device[deviceId], vertexValue, vertexNum));
			// }

		}
		else if(Algorithm_type::PR == algorithm)
		{
			assert_msg(false, "TODO: PageRank !");
		}
		else
		{
			assert_msg(false, "initAlgorithm Meet Unknown Algorithm");
		}
	}


    /* *******************************************************************
	 * Func: Host Function, Clear The ActiveOut After Each Iteration
	 * *******************************************************************/
	void clearActiveOut()
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm)|| 
			(Algorithm_type::CC == algorithm))
		{
			omp_parallel
			{
				count_type threadId = omp_get_thread_num();
				size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
				size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);
				memset(active.out().array + cur, 0, sizeof(size_t) * (end - cur));
			}
		}
		else if (Algorithm_type::PR == algorithm)
		{
			assert_msg(false, "TODO: PageRank");
		}
		else
		{
			assert_msg(false, "clearActiveOut Meet Unknown Algorithm");
		}
	}

};// end of class [Block_engine]
