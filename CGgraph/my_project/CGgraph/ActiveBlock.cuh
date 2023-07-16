#pragma once


#include <math.h>
#include <sys/mman.h>
#include <thread>


#define ACTIVE_BLOCK

class ActiveBlock
{
public:
    count_type vertexNum;
	countl_type edgeNum;
	count_type noZeroOutDegreeNum;

	countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;

    //TaskSteal
	TaskSteal* taskSteal;
	TaskSteal* taskSteal_align64;

    vertex_data_type* vertexValue;
	vertex_data_type* vertexValue_pr;

	//PUSH
	dense_bitset active_in;
	dense_bitset active_out;
	DoubleBuffer<dense_bitset> active; 

    offset_type* outDegree;

	count_type ite = 0;

	// Block
	count_type segmentSize = 6200000;
	count_type partition = 0;
	std::vector<std::vector<countl_type>> blockDis;


    ActiveBlock(const CSR_Result_type& csrResult) :
		vertexNum(0),
		edgeNum(0)
	{
	
		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;


		taskSteal = new TaskSteal();
        taskSteal_align64 = new TaskSteal();

		
		outDegree = new offset_type[vertexNum];
		for (size_t vertexId = 0; vertexId < vertexNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}

		
		allocate_vertexValueAndActive();
		Msg_info("allocate_vertexValueAndActive 完成");

#ifdef ACTIVE_BLOCK
		
		partition =(vertexNum + segmentSize - 1 ) / segmentSize;
        count_type block_num = partition * partition;
        Msg_finish("segmentSize: %u, 1-D: %u, 2-D: %u", segmentSize, partition, block_num);

		blockDis.resize(partition);
		for (size_t rowId = 0; rowId < partition; rowId++)
		{
			blockDis[rowId].resize(partition, 0);
		}
#endif
		
	}



	double graphProcess(Algorithm_type algorithm, vertex_id_type root = 0)
	{
		
		init_algorithm(algorithm, root);

		ite = 0;
		double processTime = 0.0;
		count_type activeNum = 0;
		timer iteTime;

		do
		{
			ite++;

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
			activeNum = taskSteal->threeSatge_noNUMA<ActiveBlock>(*this,
				[&](vertex_id_type vertex, offset_type nbr_start, offset_type nbr_end)
				{
					if (Algorithm_type::BFS == algorithm)
						return BFS_SPACE::bfs_noNuma<ActiveBlock>(*this, vertex, nbr_start, nbr_end);
					else if(Algorithm_type::SSSP == algorithm)
						return SSSP_SPACE::sssp_noNuma<ActiveBlock>(*this, vertex, nbr_start, nbr_end);
					else
					{
						assert_msg(false, "threeSatge_noNUMA");
						 return static_cast<count_type>(0);
					}
						
				}
			);

			
#ifdef ACTIVE_BLOCK

			countl_type totalEdgesIte = 0;
			for (count_type blockRow = 0; blockRow < partition; blockRow++)
			{
				for (count_type blockColumn = 0; blockColumn < partition; blockColumn++)
				{
					totalEdgesIte += blockDis[blockRow][blockColumn];
				}
			}



			std::string outFile = "---" + std::to_string(ite) + ".csv";
			std::ofstream out_file;
			out_file.open(outFile.c_str(),
				std::ios_base::out | std::ios_base::binary);//Opens as a binary read and writes to disk
			if (!out_file.good()) assert_msg(false, "Error opening out-file: %s", outFile.c_str());


			for (count_type blockRow = 0; blockRow < partition; blockRow++)
			{
				std::stringstream ss_log;
				ss_log.clear();
				for (count_type blockColumn = 0; blockColumn < partition; blockColumn++)
				{
					out_file << ((double)blockDis[blockRow][blockColumn] / totalEdgesIte) * 100 << ",";

					blockDis[blockRow][blockColumn] = 0;

				}
				out_file << std::endl;
			}
			out_file.close();
			
#endif


			if (activeNum == 0)
			{
				processTime = iteTime.current_time_millis();
				std::cout << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
					<< ", time: " << std::setw(6) << processTime << " (ms)" << std::endl;
				break;
			}

			active.swap();

		} while (true);

		return processTime;
		
	}




private:
    void allocate_vertexValueAndActive()
    {
        taskSteal_align64->allocateTaskForThread<count_type>(vertexNum,64,true);

        CUDA_CHECK(cudaMallocHost((void**)&(vertexValue), (vertexNum) * sizeof(vertex_data_type)));

		active_in.resize(vertexNum);
		active_out.resize(vertexNum);
		active.setDoubleBuffer(active_in, active_out);

    }

	/* ======================================================================== *
	 *                         【init_algorithm】
	 * ======================================================================== */
	void init_algorithm(Algorithm_type algorithm, vertex_id_type root)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = VertexValue_MAX;
			active.in().clear_memset();
			active.out().clear_memset();

					
			vertexValue[root] = 0;
			active.in().set_bit(root);					
		}
		else if ((Algorithm_type::CC == algorithm))
		{
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = i;
			active.in().fill();
			active.out().clear_memset();
		}
		else if ((Algorithm_type::PR == algorithm))
		{
			CUDA_CHECK(cudaMallocHost((void**)&(vertexValue_pr), (vertexNum) * sizeof(vertex_data_type)));

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
				count_type threadId = omp_get_thread_num();
				size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
				size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);
				memset(active.out().array + cur, 0, sizeof(size_t) * (end - cur));
			}
		}
		else if (Algorithm_type::PR == algorithm)
		{
			omp_parallel
			{
				count_type threadId = omp_get_thread_num();
				size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
				size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);

				memset(active.out().array + cur, 0, sizeof(size_t) * (end - cur));
			}
		}
		else
		{
			assert_msg(false, "clear_active_out error");
		}
	}







};// end of class [ActiveBlock]
