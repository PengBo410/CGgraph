#pragma once

#include "../Other/IO.hpp"
#include "../Console/console.hpp"
#include "graphFileList.hpp"
#include "graphBinReader.hpp"


void saveCSR()
{
	std::string graphName = "";/
	OrderMethod orderMethod = OrderMethod::NATIVE;

	GraphFile_type graphFile = getGraphFile(graphName, orderMethod);
	std::string filePath = graphFile.graphFile;
	size_t vertices_ = graphFile.vertices;
	size_t edges_ = graphFile.edges;

	
	if ((std::numeric_limits<count_type>::max() <= vertices_) || (std::numeric_limits<countl_type>::max() <= edges_))
	{
		assert_msg(false, "vertices or edges > count_type or countl_type");
	}

	count_type vertices = vertices_;


	SharedMemory::GraphBinReader graphBinReader(GraphRepresentation::CSR);
	Msg_info("GraphFile: %s", filePath.c_str());
	graphBinReader.group_load_directedGraph(filePath, vertices);
	graphBinReader.saveCSRtoFile(
		BASE_GRAPHFILE_PATH + graphName + "...",
		BASE_GRAPHFILE_PATH + graphName + "...",
		BASE_GRAPHFILE_PATH + graphName + "."
	);
}