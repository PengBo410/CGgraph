#pragma once

#include "sortNbr.hpp"
#include "Two-level_partition.cuh"
#include "CGgraph_engine.cuh"

#include "GraphNuma.cuh"
#include "GraphDevice.cuh"
#include "GraphCooperation.cuh"
#include "Block_partition.cuh"
#include "Compute_indegree.cuh"
#include "Block_engine.cuh"
#include "Reorder.hpp"
#include "ActiveBlock.cuh"
#include "./LG.cuh"



void main_partition()
{
    std::string graphName = "";
	OrderMethod orderMethod = OrderMethod::NATIVE; 
	int64_t root = 0; 
	count_type runs = 1;
	bool has_csrFile = 1;
	bool logResult = true;
	bool rootTranfer = true;
	bool RCM_LAST_ZERO_OUTDEGREE_check = true;
	Algorithm_type algorithm = Algorithm_type::SSSP;
	Engine_type  engine_type = Engine_type::COORDINATION;
	int useDeviceNum = 1;
	bool toyExample = false;

	// 
	if(graphName.length() >= 3 && graphName.substr(0,3).compare("toy") == 0) toyExample = true;
	else  toyExample = false;	

	setLogNFS("log.txt", true);
	assert_msg(sizeof(size_t) == 8, "CGgraph Need 64-bits Platform");
	CPUInfo cpuInfo;
	cpuInfo.print();
	Msg_info("Used Thread: [%zu]", static_cast<uint64_t>(ThreadNum));

	
	CSR_Result_type csrResult;
	count_type zeroOutDegreeNum = 0;
	count_type vertices =0;
	countl_type edges = 0;
	vertex_id_type* old2new;
	if(!toyExample)
	{
		GraphFile_type graphFile = getGraphFile(graphName, orderMethod);
		std::string file_old2new = graphFile.old2newFile;
		std::string file_addition = graphFile.addtitionFile;
		vertices = graphFile.vertices;
		edges = graphFile.edges;
		//root = graphFile.common_root;
		Msg_info("GraphName:[%s], |V| = %zu, |E| = %zu", graphName.c_str(), static_cast<uint64_t>(vertices), static_cast<uint64_t>(edges));

		
		if (has_csrFile)
		{
			std::string csrOffsetFile = graphFile.csrOffsetFile;
			std::string csrDestFile = graphFile.csrDestFile;
			std::string csrWeightFile = graphFile.csrWeightFile;	
			
			countl_type* csr_offset = load_binFile<countl_type>(csrOffsetFile, static_cast<uint64_t>(vertices + 1));
			vertex_id_type* csr_dest = load_binFile<vertex_id_type>(csrDestFile, static_cast<uint64_t>(edges));
			edge_data_type* csr_weight = load_binFile<edge_data_type>(csrWeightFile, static_cast<uint64_t>(edges));

			
			csrResult.csr_offset = csr_offset;
			csrResult.csr_dest = csr_dest;
			csrResult.csr_weight = csr_weight;
			csrResult.vertexNum = vertices;
			csrResult.edgeNum = edges;
			Msg_finish("Using CSRFile Construct csrResult complete");

			if ((orderMethod == OrderMethod::RCM_LAST_ZERO_OUTDEGREE) ||
				(orderMethod == OrderMethod::RCM_INDEGREE) ||
				(orderMethod == OrderMethod::RCM_OUTDEGREE) ||
				(orderMethod == OrderMethod::NORMAL_OUTDEGREE_DEC)
				)
			{
				count_type* temp = load_binFile<count_type>(file_addition, 1);
				zeroOutDegreeNum = temp[0];
				Msg_info("zeroOutDegreeNum = [%zu] (%.2lf%%)", static_cast<uint64_t>(zeroOutDegreeNum), ((double)zeroOutDegreeNum / csrResult.vertexNum) * 100);

				if (RCM_LAST_ZERO_OUTDEGREE_check)
				{
					assert_msg((csrResult.csr_offset[csrResult.vertexNum - zeroOutDegreeNum] == csrResult.edgeNum),
						"(csrResult.csr_offset[csrResult.vertexNum - zeroOutDegreeNum] != csrResult.edgeNum) - (%zu != %zu)",
						static_cast<uint64_t>(csrResult.csr_offset[csrResult.vertexNum - zeroOutDegreeNum]),
						static_cast<uint64_t>(csrResult.edgeNum));

					omp_parallel_for(size_t i = (csrResult.vertexNum - zeroOutDegreeNum); i < csrResult.vertexNum; i++)
					{
						countl_type degreeSize = csrResult.csr_offset[i + 1] - csrResult.csr_offset[i];
						assert_msg(degreeSize == 0, "degreeSize != 0, vertexId = %zu, degreeSize = %zu", static_cast<uint64_t>(i), static_cast<uint64_t>(degreeSize));
					}
					Msg_finish("RCM_LAST_ZERO_OUTDEGREE Check Finished !");
				}
			}
			csrResult.noZeroOutDegreeNum = csrResult.vertexNum - zeroOutDegreeNum;
		}
		else
		{
			std::string filePath = graphFile.graphFile;
			SharedMemory::GraphBinReader graphBinReader(GraphRepresentation::CSR);
			Msg_info("GraphFile: %s", filePath.c_str());
			graphBinReader.group_load_directedGraph(filePath, vertices);
			graphBinReader.sort_nbr();
			//CSR
			graphBinReader.getStruct_csrResult(csrResult);
			Msg_info("Using GraphFile Construct csrResult complete");
			//graphBinReader.printAdjList();
			graphBinReader.clearAdjList();

			if ((orderMethod == OrderMethod::RCM_LAST_ZERO_OUTDEGREE) ||
				(orderMethod == OrderMethod::RCM_INDEGREE) ||
				(orderMethod == OrderMethod::RCM_OUTDEGREE) ||
				(orderMethod == OrderMethod::NORMAL_OUTDEGREE_DEC)
				)
			{
				count_type* temp = load_binFile<count_type>(file_addition, 1);
				zeroOutDegreeNum = temp[0];
				Msg_info("zeroOutDegreeNum = [%u] (%.2f)", zeroOutDegreeNum, ((double)zeroOutDegreeNum / csrResult.vertexNum) * 100);

				if (RCM_LAST_ZERO_OUTDEGREE_check)
				{
					omp_parallel_for(size_t i = (csrResult.vertexNum - zeroOutDegreeNum); i < csrResult.vertexNum; i++)
					{
						countl_type degreeSize = csrResult.csr_offset[i + 1] - csrResult.csr_offset[i];
						assert_msg(degreeSize == 0, "degreeSize != 0, vertexId = %zu, degreeSize = %zu", static_cast<uint64_t>(i), static_cast<uint64_t>(degreeSize));
					}
					Msg_check("RCM_LAST_ZERO_OUTDEGREE checked!");
				}
			}
		}

		//check		
		if ((orderMethod != OrderMethod::NATIVE) && rootTranfer)
		{
			old2new = load_binFile<vertex_id_type>(file_old2new, vertices);
		}

		nbrSort_taskSteal(csrResult);

	}// end of Real-Word

	else
	{

		csrResult = getToyGraph(graphName, orderMethod);
	}// end of Toy-Example

    
    



	/* ****************************************************【Enginer】****************************************************************/
	if(engine_type == Engine_type::COORDINATION)
	{

		TwoLevelPartition* twoLevelPartition;
		twoLevelPartition = new TwoLevelPartition(csrResult, zeroOutDegreeNum, useDeviceNum, algorithm);
		TwoLevelPartition::Partition_type partitionResult = twoLevelPartition->getPartitionResult();

		CGgraphEngine* CGgraphEngine_;
		CGgraphEngine_ = new CGgraphEngine(csrResult, partitionResult, algorithm, useDeviceNum);

		std::vector<double> usingTime_vec;
		double usingTime = 0.0;
		usingTime_vec.resize(runs);

		do
		{
			if (root < 0 || root >= vertices) break;
			vertex_id_type switch_root = root;
			if (orderMethod != OrderMethod::NATIVE && rootTranfer)
			{
				Msg_info("old-root: %ld, new-root: %u", root, old2new[root]);
				switch_root = old2new[root];
			}
			else Msg_info("root no switch: %ld", root);

			
			for (count_type runId = 0; runId < runs; runId++)
			{
				logstream(LOG_INFO) << "==========================【" << (runId + 1) << "】 Run==========================" << std::endl;

				usingTime_vec[runId] = CGgraphEngine_->CG_co_execution(switch_root);

				
		#ifdef CHECK_RESULT_EACH_RUN			
				if (logResult)
				{
					CheckInfo_type checkInfo_type;
					checkInfo_type.graphName = graphName;
					checkInfo_type.algorithm = algorithm;
					checkInfo_type.root = root;

					checkBinResult<vertex_data_type>(getResultFilePath(checkInfo_type), graphCooperation->vertexValue, vertices, orderMethod, old2new);
				}
		#endif

			}

			sort(usingTime_vec.begin(), usingTime_vec.end());
			if (runs > 2)
			{
				for (count_type runId = 1; runId < runs - 1; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs - 2));
				usingTime = 0;
			}
			else
			{
				for (count_type runId = 0; runId < runs; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs));
				usingTime = 0;
			}

		} while (0);

		#ifndef CHECK_RESULT_EACH_RUN
		if (logResult)
		{
			CheckInfo_type checkInfo_type;
			checkInfo_type.graphName = graphName;
			checkInfo_type.algorithm = algorithm;
			checkInfo_type.root = root;

			checkBinResult<vertex_data_type>(getResultFilePath(checkInfo_type), CGgraphEngine_->vertexValue, vertices, orderMethod, old2new);
		}
		#endif 
	
	}




	else if(engine_type == Engine_type::MULTI_CORE)
	{
		Graph_Numa* graph_Numa;
		if (orderMethod != OrderMethod::OUTDEGREE_THREAD)
		{
			graph_Numa = new Graph_Numa(csrResult);
		}
		else assert_msg(false, "Current We Delete OrderMethod::OUTDEGREE_THREAD");


		std::vector<double> usingTime_vec;
		double usingTime = 0.0;
		usingTime_vec.resize(runs);

		do
		{
			std::cout << "---> input Root [0, " << vertices << "):\n";
			if (root < 0 || root >= vertices) break;
			vertex_id_type switch_root = root;
			if (orderMethod != OrderMethod::NATIVE && rootTranfer)
			{
				Msg_info("old-root: %ld, new-root: %u", root, old2new[root]);
				switch_root = old2new[root];
			}
			else Msg_info("root no switch: %ld", root);

		
			for (count_type runId = 0; runId < runs; runId++)
			{
				logstream(LOG_INFO) << "========================== 【" << (runId + 1) << "】 Run==========================" << std::endl;
				usingTime_vec[runId] = graph_Numa->graphProcess(algorithm, switch_root);		
			}

	
			sort(usingTime_vec.begin(), usingTime_vec.end());
			if (runs > 2)
			{
				for (count_type runId = 1; runId < runs - 1; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs - 2));
				usingTime = 0;
			}
			else
			{
				for (count_type runId = 0; runId < runs; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs));
				usingTime = 0;
			}
		} while (0);

		// result
		#ifndef CHECK_RESULT_EACH_RUN
		if (logResult)
		{
			CheckInfo_type checkInfo_type;
			checkInfo_type.graphName = graphName;
			checkInfo_type.algorithm = algorithm;
			checkInfo_type.root = root;

			checkBinResult<vertex_data_type>(getResultFilePath(checkInfo_type), graph_Numa->vertexValue, vertices, orderMethod, old2new);
		}
		#endif 


	}

	else if(engine_type == Engine_type::SINGLE_GPU)
	{
		GraphDeviceWorklist* graphDeviceWorklist;
		graphDeviceWorklist = new GraphDeviceWorklist(csrResult, algorithm);

		std::vector<double> usingTime_vec;
		double usingTime = 0.0;
		usingTime_vec.resize(runs);

		do
		{
			if (root < 0 || root >= vertices) break;
			vertex_id_type switch_root = root;
			if (orderMethod != OrderMethod::NATIVE && rootTranfer && algorithm != Algorithm_type::PR)
			{
				Msg_info("old-root: %ld, new-root: %u", root, old2new[root]);
				switch_root = old2new[root];
			}
			else Msg_info("root no switch: %ld", root);

			
			for (count_type runId = 0; runId < runs; runId++)
			{
				logstream(LOG_INFO) << "========================== 【" << (runId + 1) << "】 Run==========================" << std::endl;

				usingTime_vec[runId] = graphDeviceWorklist->graphProcess_device(algorithm, switch_root);	
			}

			sort(usingTime_vec.begin(), usingTime_vec.end());
			if (runs > 2)
			{
				for (count_type runId = 1; runId < runs - 1; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs - 2));
				usingTime = 0;
			}
			else
			{
				for (count_type runId = 0; runId < runs; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs));
				usingTime = 0;
			}

		} while (0);

		// result
		#ifndef CHECK_RESULT_EACH_RUN
		if (logResult)
		{
			CheckInfo_type checkInfo_type;
			checkInfo_type.graphName = graphName;
			checkInfo_type.algorithm = algorithm;
			checkInfo_type.root = root;

			graphDeviceWorklist->result_D2H();
			checkBinResult<vertex_data_type>(getResultFilePath(checkInfo_type), graphDeviceWorklist->vertexValue, vertices, orderMethod, old2new);
		}
		#endif 

	}

	else if(engine_type == Engine_type::COOPERATION)
	{
		GraphCooperation* graphCooperation;
		graphCooperation = new GraphCooperation(csrResult, zeroOutDegreeNum, algorithm, graphName, 0);

		std::vector<double> usingTime_vec;
		double usingTime = 0.0;
		usingTime_vec.resize(runs);

		do
		{
			if (root < 0 || root >= vertices) break;
			vertex_id_type switch_root = root;
			if (orderMethod != OrderMethod::NATIVE && rootTranfer)
			{
				Msg_info("old-root: %ld, new-root: %u", root, old2new[root]);
				switch_root = old2new[root];
			}
			else Msg_info("root no switch: %ld", root);

			for (count_type runId = 0; runId < runs; runId++)
			{
				logstream(LOG_INFO) << "========================== 【" << (runId + 1) << "】 Run==========================" << std::endl;

				usingTime_vec[runId] = graphCooperation->graphCooperationExecution(switch_root);
			}
			sort(usingTime_vec.begin(), usingTime_vec.end());
			if (runs > 2)
			{
				for (count_type runId = 1; runId < runs - 1; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs - 2));
				usingTime = 0;
			}
			else
			{
				for (count_type runId = 0; runId < runs; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs));
				usingTime = 0;
			}

		} while (0);

		
		#ifndef CHECK_RESULT_EACH_RUN
		if (logResult)
		{
			CheckInfo_type checkInfo_type;
			checkInfo_type.graphName = graphName;
			checkInfo_type.algorithm = algorithm;
			checkInfo_type.root = root;

			checkBinResult<vertex_data_type>(getResultFilePath(checkInfo_type), graphCooperation->vertexValue, vertices, orderMethod, old2new);
		}
		#endif 
	}


	else if(engine_type == Engine_type::BLOCK_2D)
	{

		Block_partition* blockPartition = new Block_partition(csrResult, 0, algorithm);//(csrResult, zeroOutDegreeNum, algorithm);
		Block_engine* blockEngine = new Block_engine(csrResult, blockPartition->get_blockResult(), algorithm);
	}




	//> 
	else if(engine_type == Engine_type::COMPUTE_INDEGREE)
	{
		Compute_indegree* compute_indegree = new Compute_indegree(csrResult, zeroOutDegreeNum);

		compute_indegree->sortIndegree();
	}

	//> REORDER
	else if(engine_type == Engine_type::REORDER)
	{
		Compute_indegree* compute_indegree = new Compute_indegree(csrResult, zeroOutDegreeNum);
		offset_type* inDegree = compute_indegree->getIndegree();
		offset_type* outDegree = compute_indegree->getOutdegree();

		if(toyExample)
		{
			printf("OutDegree:");
			for (count_type i = 0; i < csrResult.vertexNum; i++) printf("%2zu ", static_cast<uint64_t>(outDegree[i]));
			printf("\n");
			
			printf("InDegree: ");
			for (size_t i = 0; i < csrResult.vertexNum; i++) printf("%2zu ", static_cast<uint64_t>(inDegree[i]));
			printf("\n");
		}

		
		Reorder* reorder = new Reorder(csrResult, outDegree, inDegree);
		reorder->generateReorderGraphFile(OrderMethod::IN_DEGREE_DEC, graphName, toyExample);
	}


	//> Active Block
	else if(engine_type == Engine_type::ACTIVEBLOCK)
	{
		ActiveBlock* activeBlock = new ActiveBlock(csrResult);

		std::vector<double> usingTime_vec;
		double usingTime = 0.0;
		usingTime_vec.resize(runs);

		do
		{
			std::cout << "--->  input Root [0, " << vertices << "):\n";
			if (root < 0 || root >= vertices) break;
			vertex_id_type switch_root = root;
			if (orderMethod != OrderMethod::NATIVE && rootTranfer)
			{
				Msg_info("old-root: %ld, new-root: %u", root, old2new[root]);
				switch_root = old2new[root];
			}
			else Msg_info("root no switch : %ld", root);

			
			for (count_type runId = 0; runId < runs; runId++)
			{
				logstream(LOG_INFO) << "========================== 【" << (runId + 1) << "】 Run==========================" << std::endl;
				usingTime_vec[runId] = activeBlock->graphProcess(algorithm, switch_root);		
			}

			
			sort(usingTime_vec.begin(), usingTime_vec.end());
			if (runs > 2)
			{
				for (count_type runId = 1; runId < runs - 1; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs - 2));
				usingTime = 0;
			}
			else
			{
				for (count_type runId = 0; runId < runs; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs));
				usingTime = 0;
			}
		} while (0);

		// result
		#ifndef CHECK_RESULT_EACH_RUN
		if (logResult)
		{
			CheckInfo_type checkInfo_type;
			checkInfo_type.graphName = graphName;
			checkInfo_type.algorithm = algorithm;
			checkInfo_type.root = root;

			checkBinResult<vertex_data_type>(getResultFilePath(checkInfo_type), activeBlock->vertexValue, vertices, orderMethod, old2new);
		}
		#endif 
	}
	else if(engine_type == Engine_type::LG)
	{

		LG* lg;
		lg = new LG(csrResult, algorithm);

		std::vector<double> usingTime_vec;
		double usingTime = 0.0;
		usingTime_vec.resize(runs);

		do
		{
			if (root < 0 || root >= vertices) break;
			vertex_id_type switch_root = root;
			if (orderMethod != OrderMethod::NATIVE && rootTranfer && algorithm != Algorithm_type::PR)
			{
				Msg_info("old-root: %ld, new-root: %u", root, old2new[root]);
				switch_root = old2new[root];
			}
			else Msg_info("root no switch: %ld", root);

			
			for (count_type runId = 0; runId < runs; runId++)
			{
				logstream(LOG_INFO) << "========================== 【" << (runId + 1) << "】 Run==========================" << std::endl;

				usingTime_vec[runId] = lg->graphProcess(algorithm, switch_root);	
			}

			sort(usingTime_vec.begin(), usingTime_vec.end());
			if (runs > 2)
			{
				for (count_type runId = 1; runId < runs - 1; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs - 2));
				usingTime = 0;
			}
			else
			{
				for (count_type runId = 0; runId < runs; runId++)
				{
					usingTime += usingTime_vec[runId];
				}
				printf("[Total Run Time]: %f (ms)\n", usingTime / (runs));
				usingTime = 0;
			}

		} while (0);

		// result
		#ifndef CHECK_RESULT_EACH_RUN
		if (logResult)
		{
			CheckInfo_type checkInfo_type;
			checkInfo_type.graphName = graphName;
			checkInfo_type.algorithm = algorithm;
			checkInfo_type.root = root;

			lg->result_D2H();
			checkBinResult<vertex_data_type>(getResultFilePath(checkInfo_type), lg->vertexValue, vertices, orderMethod, old2new);
		}
		#endif 


	}

	else
	{
		assert_msg(false, "Unknown engine_type");
	}



}