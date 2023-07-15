#pragma once

#include "../Other/IO.hpp"
#include "../Console/console.hpp"
#include "graphFileList.hpp"
#include "graphBinReader.hpp"


void saveCSR()
{
	std::string graphName = "uk-union";//gsh2015tpd	
	OrderMethod orderMethod = OrderMethod::NATIVE; //如果是OUTDEGREE_THREAD需要开启REORDER宏定义，tasksteal要用，TODO

	GraphFile_type graphFile = getGraphFile(graphName, orderMethod);
	std::string filePath = graphFile.graphFile;
	size_t vertices_ = graphFile.vertices;
	size_t edges_ = graphFile.edges;

	//检查:数据类型容量检测
	if ((std::numeric_limits<count_type>::max() <= vertices_) || (std::numeric_limits<countl_type>::max() <= edges_))
	{
		assert_msg(false, "vertices 或 edges 超过了定义的 count_type 或 countl_type");
	}

	count_type vertices = vertices_;
	//countl_type edges = edges_;


	SharedMemory::GraphBinReader graphBinReader(GraphRepresentation::CSR);
	Msg_info("GraphFile: %s", filePath.c_str());
	graphBinReader.group_load_directedGraph(filePath, vertices);//测试完以后加入到graphFileList中
	graphBinReader.saveCSRtoFile(
		BASE_GRAPHFILE_PATH + graphName + "/native_csrOffset_u32.bin",
		BASE_GRAPHFILE_PATH + graphName + "/native_csrDest_u32.bin",
		BASE_GRAPHFILE_PATH + graphName + "/native_csrWeight_u32.bin"
	);
}