#pragma once

#include "Basic/basic_include.cuh"
#include "device_algorithm.cuh"
#include <type_traits>



class GraphDeviceWorklist {

public:
	count_type vertexNum;
	countl_type edgeNum;

	count_type zeroOutDegreeNum = 0;
	count_type noZeroOutDegreeNum = 0;

	countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;
	vertex_data_type* vertexValue;
	vertex_data_type* vertexValue_pr;


	countl_type* csr_offset_device;
	vertex_id_type* csr_dest_device;
	edge_data_type* csr_weight_device;
	vertex_data_type* vertexValue_device;
	vertex_data_type* vertexValue_device_pr;

	vertex_id_type* worklist_in_device;
	vertex_id_type* worklist_out_device;
	countl_type worklist_allocateSize;
	DoubleBuffer_array<vertex_id_type>* worklist_device;

	
	count_type common_size = 5;
	offset_type* common;
	offset_type* common_device;

	Algorithm_type algorithm;

public:

	//===============================================================================
	//                                    【con】
	//===============================================================================
	GraphDeviceWorklist(CSR_Result_type& csr_result, Algorithm_type algorithm_) :
		vertexNum(0),
		edgeNum(0)
	{
		vertexNum = csr_result.vertexNum;
		edgeNum = csr_result.edgeNum;

		csr_offset = csr_result.csr_offset;
		csr_dest = csr_result.csr_dest;
		csr_weight = csr_result.csr_weight;

		algorithm = algorithm_;		

		worklist_allocateSize = 0.2 * edgeNum;
		Msg_info("Worklist_model_device 中 worklist_allocateSize = %zu", static_cast<uint64_t>(worklist_allocateSize));

		timer t;
		allocate();
		Msg_info("Host + Device time：%f (ms)", t.get_time_ms());

		t.start();
		csrToDevice();
		Msg_info("CSR Host to Device time：%f (ms)", t.get_time_ms());
	}

public:
	~GraphDeviceWorklist()
	{
		
		CUDA_CHECK(cudaFree(csr_offset_device));
		CUDA_CHECK(cudaFree(csr_dest_device));
		CUDA_CHECK(cudaFree(common_device));
		CUDA_CHECK(cudaFree(vertexValue_device));
		CUDA_CHECK(cudaFree(vertexValue_device_pr));
		CUDA_CHECK(cudaFree(worklist_in_device));
		CUDA_CHECK(cudaFree(worklist_out_device));

		CUDA_CHECK(cudaFreeHost(vertexValue));
		CUDA_CHECK(cudaFreeHost(vertexValue_pr));
		CUDA_CHECK(cudaFreeHost(common));

	}

public:
	double graphProcess_device(Algorithm_type algorithm, vertex_id_type root = 0)
	{
	
		init_algorithm(algorithm, root);

		
		count_type ite = 0;
		double processTime = 0.0;
		count_type nBlock = 0;
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			common[0] = 1;
		}
		else if (Algorithm_type::CC == algorithm || Algorithm_type::PR == algorithm)
		{
			common[0] = vertexNum;
		}
		else
		{
			assert_msg(false, "init_algorithm error");
		}

		timer iteTime;
		timer singTime;
		do
		{
			ite++;
			singTime.start();
			common[1] = common[0]; 
			common[0] = 0;			 
			CUDA_CHECK(cudaMemcpy(common_device, common, 2 * sizeof(offset_type), cudaMemcpyHostToDevice));//H2D

			nBlock = (common[1] + BLOCKSIZE - 1) / BLOCKSIZE;
			if (algorithm == Algorithm_type::BFS)
			{
				BFS_SPACE::bfs_worklist_model<GraphDeviceWorklist>(*this, nBlock);
			}
			else if (algorithm == Algorithm_type::SSSP)
			{
				SSSP_SPACE::sssp_worklist_model<GraphDeviceWorklist>(*this, nBlock);
			}
			else if (algorithm == Algorithm_type::CC)
			{
				WCC_SPACE::wcc_worklist_model<GraphDeviceWorklist>(*this, nBlock);
			}
			else if (algorithm == Algorithm_type::PR)
			{
				for (count_type i = 0; i < vertexNum; i++) vertexValue_pr[i] = 0.0;
				CUDA_CHECK(cudaMemcpy(vertexValue_device_pr, vertexValue_pr, (vertexNum) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));

				nBlock = (vertexNum + BLOCKSIZE - 1) / BLOCKSIZE;
				PR_SPACE::pr_worklist_model<GraphDeviceWorklist>(*this, nBlock);
				std::swap(vertexValue_device, vertexValue_device_pr);
				
			}
			else
			{
				assert_msg(false, "graphProcess_device 时, error");
			}
			

			CUDA_CHECK(cudaMemcpy(common, common_device, sizeof(offset_type), cudaMemcpyDeviceToHost));//D2H 
			Msg_info("[%2u], worklistCount = (%8u), time：%.2f (ms)", ite, common[0], singTime.get_time_ms());

			worklist_device->swap();


			if (algorithm == Algorithm_type::PR)
			{
				if (ite >= root) break;
			}
			else
			{
				if (common[0] == 0) break;
			}

			
		} while (true);

		processTime = iteTime.get_time_ms();
		std::cout << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
			<< ", time: " << std::setw(6) << processTime << " (ms)" << std::endl;

		return processTime;
	}


private:

	void allocate()
	{
		CUDA_CHECK(cudaMallocHost((void**)&(vertexValue), (vertexNum) * sizeof(vertex_data_type)));
		if (algorithm == Algorithm_type::PR)
		{
			CUDA_CHECK(cudaMallocHost((void**)&(vertexValue_pr), (vertexNum) * sizeof(vertex_data_type)));
		}
		CUDA_CHECK(cudaMallocHost((void**)&(common), (common_size) * sizeof(offset_type)));

		CUDA_CHECK(cudaMalloc((void**)&(csr_offset_device), (vertexNum + 1) * sizeof(countl_type)));
		CUDA_CHECK(cudaMalloc((void**)&(csr_dest_device), (edgeNum) * sizeof(vertex_id_type)));
		if (algorithm == Algorithm_type::SSSP)
		{
			CUDA_CHECK(cudaMalloc((void**)&(csr_weight_device), (edgeNum) * sizeof(edge_data_type)));
		}
		CUDA_CHECK(cudaMalloc((void**)&(common_device), (common_size) * sizeof(offset_type)));
		CUDA_CHECK(cudaMalloc((void**)&(vertexValue_device), (vertexNum) * sizeof(vertex_data_type)));
		if (algorithm == Algorithm_type::PR)
		{
			CUDA_CHECK(cudaMalloc((void**)&(vertexValue_device_pr), (vertexNum) * sizeof(vertex_data_type)));
		}
		CUDA_CHECK(cudaMalloc((void**)&(worklist_in_device), (worklist_allocateSize) * sizeof(vertex_id_type)));
		CUDA_CHECK(cudaMalloc((void**)&(worklist_out_device), (worklist_allocateSize) * sizeof(vertex_id_type)));
		worklist_device = new DoubleBuffer_array<vertex_id_type>(worklist_in_device, worklist_out_device);
	}


	void csrToDevice()
	{
		CUDA_CHECK(cudaMemcpy(csr_offset_device, csr_offset, (vertexNum + 1) * sizeof(countl_type), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(csr_dest_device, csr_dest, (edgeNum) * sizeof(vertex_id_type), cudaMemcpyHostToDevice));
		if (algorithm == Algorithm_type::SSSP)
		{
			CUDA_CHECK(cudaMemcpy(csr_weight_device, csr_weight, (edgeNum) * sizeof(edge_data_type), cudaMemcpyHostToDevice));
		}

		count_type vertexNum_ = (count_type)vertexNum; 
		CUDA_CHECK(cudaMemcpy(common_device + 2, &vertexNum_, sizeof(count_type), cudaMemcpyHostToDevice));
		countl_type edgeNum_ = (countl_type)edgeNum; 
		CUDA_CHECK(cudaMemcpy(common_device + 3, &edgeNum_, sizeof(countl_type), cudaMemcpyHostToDevice));		
	}


	void init_algorithm(Algorithm_type algorithm, vertex_id_type root)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			//vertexValue
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = VertexValue_MAX;
			vertexValue[root] = 0;
			CUDA_CHECK(cudaMemcpy(vertexValue_device, vertexValue, (vertexNum) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));

			//worklist
			CUDA_CHECK(cudaMemcpy(worklist_device->in(), &root, sizeof(vertex_id_type), cudaMemcpyHostToDevice));
		}
		else if (Algorithm_type::CC == algorithm)
		{
			//vertexValue
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = i;
			CUDA_CHECK(cudaMemcpy(vertexValue_device, vertexValue, (vertexNum) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));

			/
			CUDA_CHECK(cudaMemcpy(worklist_device->in(), vertexValue, sizeof(vertex_data_type)* vertexNum, cudaMemcpyHostToDevice));
		}
		else if (Algorithm_type::PR == algorithm)
		{
		
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = 1/vertexNum;
			CUDA_CHECK(cudaMemcpy(vertexValue_device, vertexValue, (vertexNum) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));
		}
		else
		{
			assert_msg(false, "init_algorithm error");
		}
	}

public:

	void result_D2H()
	{
		CUDA_CHECK(cudaMemcpy(vertexValue, vertexValue_device, sizeof(vertex_data_type) * vertexNum, cudaMemcpyDeviceToHost));//D2H
	}






};// end of class [GraphDevice]