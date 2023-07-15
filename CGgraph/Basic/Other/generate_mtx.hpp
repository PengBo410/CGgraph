#pragma once

#include "../Type/data_type.hpp"
#include "../Graph/basic_def.hpp"
#include "../Graph/graphFileList.hpp"
#include "../Graph/graphBinReader.hpp"

void main_generate_MTX()
{
    std::string graphName = "friendster";//gsh2015tpd	
	OrderMethod orderMethod = OrderMethod::NATIVE; //如果是OUTDEGREE_THREAD需要开启REORDER宏定义，tasksteal要用，TODO
	bool has_csrFile = 1;
    bool RCM_LAST_ZERO_OUTDEGREE_check = true;
    //std::string outFile = "/home/pengjie/graph_data/" + graphName + "/" + graphName + "_RCM_LAST_ZERO_OUTDEGREE_.mtx";
    std::string outFile = "/home/pengjie/graph_data/" + graphName + "/" + graphName + "_NATIVE.mtx";


    GraphFile_type graphFile = getGraphFile(graphName, orderMethod);
	std::string file_old2new = graphFile.old2newFile;
	std::string file_addition = graphFile.addtitionFile;
	count_type vertices = graphFile.vertices;
	countl_type edges = graphFile.edges;
	//root = graphFile.common_root;
	Msg_info("GraphName:[%s], |V| = %zu, |E| = %zu", graphName.c_str(), static_cast<uint64_t>(vertices), static_cast<uint64_t>(edges));


    /*****************************************************【正常读取图文件】****************************************************************/
	CSR_Result_type csrResult;
	count_type zeroOutDegreeNum = 0;
	if (has_csrFile)
	{
		std::string csrOffsetFile = graphFile.csrOffsetFile;
		std::string csrDestFile = graphFile.csrDestFile;
		std::string csrWeightFile = graphFile.csrWeightFile;


		//root = graphFile.common_root;

		//从CSR文件中读取csr
		countl_type* csr_offset = load_binFile<countl_type>(csrOffsetFile, (vertices + 1));
		vertex_id_type* csr_dest = load_binFile<vertex_id_type>(csrDestFile, edges);
		edge_data_type* csr_weight = load_binFile<edge_data_type>(csrWeightFile, edges);

		//构建CSR
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
		graphBinReader.group_load_directedGraph(filePath, vertices);//测试完以后加入到graphFileList中，
		graphBinReader.sort_nbr();
		//获取CSR
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
				Msg_check("RCM_LAST_ZERO_OUTDEGREE 通过检查！");
			}
		}
	}


    //> 生成 .mtx 文件
    bool mtxFile_exist = (access(outFile.c_str(), F_OK) >= 0);
    if(!mtxFile_exist)
    {
        std::ofstream out_file;
		out_file.open(outFile.c_str(),
			std::ios_base::out | std::ios_base::binary);//以二进制读的方式打开,并写入磁盘

        if (!out_file.good()) {
            assert_msg(false, "Error opening out-file: %s", outFile.c_str());
        }

        out_file << "%%MatrixMarket matrix coordinate real general" << std::endl;
        out_file << csrResult.vertexNum << " " << csrResult.vertexNum << " " << csrResult.edgeNum << std::endl;

        timer t;
        uint64_t writeEdges = 0;
        for(count_type vertexId = 0; vertexId < csrResult.vertexNum; vertexId++)
        {
            countl_type nbrStart = csrResult.csr_offset[vertexId];
            countl_type nbrEnd = csrResult.csr_offset[vertexId + 1];

            for(countl_type nbrId = nbrStart; nbrId < nbrEnd; nbrId ++)
            {
                // mtx 文件是从1开始计数,所有我们要在之前的顶点上+1
                out_file << (vertexId + 1) << " " << (csrResult.csr_dest[nbrId] + 1) << " " << csrResult.csr_weight[nbrId] << std::endl;

                writeEdges ++;
                if ((writeEdges != 0) && (writeEdges % RATE_UTIL == 0)) 
                    Msg_rate("Write To MTX File Has Finished %.2lf %%", 
			            static_cast<double>(writeEdges) / static_cast<double>(csrResult.edgeNum) * 100 );
            }
        }
        Msg_rate("Write To MTX File Has Finished 100%%, Used time: %.2lf (ms)", t.get_time());

        assert_msg((writeEdges == csrResult.edgeNum), "Write MTX Error, writeEdges = %zu, edgeNum = %zu",
            static_cast<uint64_t>(writeEdges), static_cast<uint64_t>(csrResult.edgeNum)
        );

        out_file.close();
    }
    else
    {
        Msg_info("MTX File: %s 已经存在", outFile.c_str());
    }

		

}