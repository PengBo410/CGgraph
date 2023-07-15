#pragma once

#include <string>
#include <assert.h>
#include "basic_struct.hpp"
#include "../Console/console.hpp"

std::string BASE_GRAPHFILE_PATH = "/home/pengjie/graph_data/";


GraphFile_type getGraphFile(std::string graphName, OrderMethod orderMethod = OrderMethod::NATIVE)
{
    GraphFile_type graphFile;

    /* ***************************************************************************
	 *                               [cusha]
	 * ***************************************************************************/
    if (graphName == "cusha")
	{
        graphFile.vertices = 8;
		graphFile.edges = 14;

		if (orderMethod == OrderMethod::NATIVE)
		{
			graphFile.graphFile = "/home/omnisky/inputFile/cushaSsspInput_u32_u32.bin";			
			graphFile.common_root = 0;

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrOffset_u32.bin";
			graphFile.csrDestFile =   BASE_GRAPHFILE_PATH + graphName + "/native_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrWeight_u32.bin";

			return graphFile;
		}
        else if (orderMethod == OrderMethod::RCM_LAST_ZERO_OUTDEGREE)
		{
			graphFile.graphFile =     BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile =   BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_addition.txt";

			graphFile.common_root = 3;// 0 -> 3

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile =   BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::RCM_INDEGREE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCMIndegree_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCMIndegree_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/RCMIndegree_LAST_ZERO_OUTDEGREE_addition.txt";

			graphFile.common_root = 0;//?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCMIndegree_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCMIndegree_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCMIndegree_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::RCM_OUTDEGREE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCMOutdegree_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCMOutdegree_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/RCMOutdegree_LAST_ZERO_OUTDEGREE_addition.txt";

			graphFile.common_root = 0;// ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCMOutdegree_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCMOutdegree_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCMOutdegree_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::NORMAL_OUTDEGREE_DEC)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_addition.txt";

			graphFile.common_root = 0; // ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::MY_BFS_OUT)
		{
			//graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_addition.txt";

			graphFile.common_root = 0; // ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_csrWeight_u32.bin";

			return graphFile;
		}
		else
		{
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod", graphName.c_str());
		}
    }// end of [cusha]




    /* ***************************************************************************
	 *                               [enwiki]
	 * ***************************************************************************/
    else if (graphName == "enwiki")
	{
        graphFile.vertices = 4206785;
		graphFile.edges = 101311614;

		if (orderMethod == OrderMethod::NATIVE)
		{
			graphFile.graphFile = "/data/webgraph/bin/enwiki-2013-uint32t.bin";		
			graphFile.common_root = 0;

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrOffset_u32.bin";
			graphFile.csrDestFile =   BASE_GRAPHFILE_PATH + graphName + "/native_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrWeight_u32.bin";

			return graphFile;
		}
        else if (orderMethod == OrderMethod::RCM_LAST_ZERO_OUTDEGREE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_addition.txt";

			graphFile.common_root = 441042;// 0 -> 441042 

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
        else if (orderMethod == OrderMethod::RCM)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_old2new.txt";

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrWeight_u32.bin";

			graphFile.common_root = 379625;

			return graphFile;
		}
        else if (orderMethod == OrderMethod::OUTDEGREE_NODE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_node_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_node_old2new.txt";
			graphFile.common_root = 7891;

			return graphFile;
		}
        else if (orderMethod == OrderMethod::OUTDEGREE_NODE_dec)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_nodeDec_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_nodeDec_old2new.txt";

			graphFile.common_root = 0;
			Msg_info("common_root 需要更新");

			return graphFile;
		}
        else
		{
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod", graphName.c_str());
		}


    }// end of [enwiki]







    /* ***************************************************************************
	 *                               [twitter2010]
	 * ***************************************************************************/
    else if (graphName == "twitter2010")
	{
        graphFile.vertices = 61578415;
		graphFile.edges = 1468364884;

		if (orderMethod == OrderMethod::NATIVE)
		{
			graphFile.graphFile = "/data/webgraph/2021-6-2/twitter2010_hasEdge_uint32t.bin";
			graphFile.common_root = 892741;

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrWeight_u32.bin";

			return graphFile;
		}
        else if (orderMethod == OrderMethod::RCM_LAST_ZERO_OUTDEGREE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_addition.txt";// 21475137 (34.87%)

			graphFile.common_root = 34367363;// old-root: 892741, new-root: 34367363

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
        else if (orderMethod == OrderMethod::RCM)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_old2new.txt";

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrWeight_u32.bin";

			graphFile.common_root = 40582822;

			return graphFile;
		}
		else if (orderMethod == OrderMethod::OUTDEGREE_NODE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_node_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_node_old2new.txt";
			graphFile.common_root = 60959618;

			return graphFile;
		}
        else if (orderMethod == OrderMethod::OUTDEGREE_NODE_dec)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_nodeDec_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_nodeDec_old2new.txt";

			graphFile.common_root = 0;
			graphFile.common_root = 616395;

			return graphFile;
		}
		else if (orderMethod == OrderMethod::MY_BFS_OUT)
		{
			//graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_addition.txt";

			graphFile.common_root = 0; // ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::MY_BFS_IN)
		{
			//graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_IN_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_IN_addition.txt";

			graphFile.common_root = 0; // ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_IN_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_IN_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_IN_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::MY_BFS_OUT_IN)
		{
			//graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_IN_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_IN_addition.txt";

			graphFile.common_root = 0; // ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_IN_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_IN_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/MY_BFS_OUT_IN_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::MY_IMAX_BFS_IN)
		{
			//graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/MY_IMAX_BFS_IN_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/MY_IMAX_BFS_IN_addition.txt";

			graphFile.common_root = 0; // ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/MY_IMAX_BFS_IN_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/MY_IMAX_BFS_IN_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/MY_IMAX_BFS_IN_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::IN_DEGREE_DEC)
		{
			//graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/NORMAL_OUTDEGREE_DEC_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/IN_DEGREE_DEC_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/IN_DEGREE_DEC_addition.txt";

			graphFile.common_root = 0; // ?

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/IN_DEGREE_DEC_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/IN_DEGREE_DEC_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/IN_DEGREE_DEC_csrWeight_u32.bin";

			return graphFile;
		}
		else
		{
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod", graphName.c_str());
		}
        
    }// end of [twitter2010]


	/* ***************************************************************************
	 *                               [friendster]
	 * ***************************************************************************/
    else if (graphName == "friendster")
	{
		graphFile.vertices = 124836180;
		graphFile.edges = 1806067135;

		if (orderMethod == OrderMethod::NATIVE)
		{
			graphFile.graphFile = "/data/webgraph/bin/friendster/native_bin.txt";
			graphFile.common_root = 25689;

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::RCM_LAST_ZERO_OUTDEGREE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_addition.txt";

			graphFile.common_root = 0;// old-root: 892741, new-root: 34367363

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
		else if (orderMethod == OrderMethod::OUTDEGREE_NODE)
		{
			/*graphFile.graphFile = "/data/webgraph/bin/" + graphName + "/outDegree_node_bin.txt";
			graphFile.old2newFile = "/data/webgraph/bin/" + graphName + "/outDegree_node_old2new.txt";
			graphFile.common_root = 60959618;

			return graphFile;*/
		}
		else
		{
			Msg_info("图文件[%s]不支持当前的OrderMethod", graphName.c_str());
			exit(1);
		}
	}



	/* ***************************************************************************
	 *                               [uk-union]
	 * ***************************************************************************/
    else if (graphName == "uk-union")
	{
        graphFile.vertices = 133633040;
		graphFile.edges = 5475109924;

		if (orderMethod == OrderMethod::NATIVE)
		{
			graphFile.graphFile = "/home/pengjie/graph_data/uk-union/uk-union_u32_reGen.bin";
			graphFile.common_root = 0; //todo

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/native_csrWeight_u32.bin";

			return graphFile;
		}
        else if (orderMethod == OrderMethod::RCM_LAST_ZERO_OUTDEGREE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_old2new.txt";
			graphFile.addtitionFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_addition.txt";// 21475137 (34.87%)

			graphFile.common_root = 34367363;// old-root: 892741, new-root: 34367363

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_LAST_ZERO_OUTDEGREE_csrWeight_u32.bin";

			return graphFile;
		}
        else if (orderMethod == OrderMethod::RCM)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_old2new.txt";

			graphFile.csrOffsetFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrOffset_u32.bin";
			graphFile.csrDestFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrDest_u32.bin";
			graphFile.csrWeightFile = BASE_GRAPHFILE_PATH + graphName + "/RCM_csrWeight_u32.bin";

			graphFile.common_root = 40582822;

			return graphFile;
		}
		else if (orderMethod == OrderMethod::OUTDEGREE_NODE)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_node_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_node_old2new.txt";
			graphFile.common_root = 60959618;

			return graphFile;
		}
        else if (orderMethod == OrderMethod::OUTDEGREE_NODE_dec)
		{
			graphFile.graphFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_nodeDec_bin.txt";
			graphFile.old2newFile = BASE_GRAPHFILE_PATH + graphName + "/outDegree_nodeDec_old2new.txt";

			graphFile.common_root = 0;
			graphFile.common_root = 616395;

			return graphFile;
		}
        
    }// end of [twitter2010]

	else
	{
		assert_msg(false, "没有找到图名为[%s]的相应文件", graphName.c_str());
	}

    return graphFile;
}





CSR_Result_type getToyGraph(std::string graphName, OrderMethod orderMethod = OrderMethod::NATIVE)
{
	CSR_Result_type csrResult;

	if(graphName == "toy_0")
	{
		countl_type* offset = new countl_type[13];
		offset[0] = 0; offset[1] = 3; offset[2] = 6; offset[3] = 6; offset[4] = 8; offset[5] = 8;
		offset[6] = 11; offset[7] = 15; offset[8] = 16; offset[9] = 21; offset[10] = 22; offset[11] = 24;
		offset[12] = 26;

		vertex_id_type* dest = new vertex_id_type[26];
		dest[0] = 1; dest[1] = 3; dest[2] = 7; dest[3] = 3; dest[4] = 5; dest[5] = 9;
		dest[6] = 5; dest[7] = 6; dest[8] = 7; dest[9] = 9; dest[10] = 11; dest[11] = 0;
		dest[12] = 5; dest[13] = 8; dest[14] = 9; dest[15] = 5; dest[16] = 0; dest[17] = 2;
		dest[18] = 4; dest[19] = 9; dest[20] = 11; dest[21] = 5; dest[22] = 2; dest[23] = 8;
		dest[24] = 2; dest[25] = 5;

		edge_data_type* weight = new edge_data_type[26];
		weight[0] = 1; weight[1] = 3; weight[2] = 7; weight[3] = 3; weight[4] = 5; weight[5] = 9;
		weight[6] = 5; weight[7] = 6; weight[8] = 7; weight[9] = 9; weight[10] = 11; weight[11] = 0;
		weight[12] = 5; weight[13] = 8; weight[14] = 9; weight[15] = 5; weight[16] = 0; weight[17] = 2;
		weight[18] = 4; weight[19] = 9; weight[20] = 11; weight[21] = 5; weight[22] = 2; weight[23] = 8;
		weight[24] = 2; weight[25] = 5;

		csrResult.vertexNum = 12;
		csrResult.edgeNum = 26;
		csrResult.csr_offset = offset;
		csrResult.csr_dest = dest;
		csrResult.csr_weight = weight;
	}
	else if(graphName == "toy_1")
	{
		countl_type* offset = new countl_type[13];
		offset[0] = 0; offset[1] = 3; offset[2] = 6; offset[3] = 6; offset[4] = 8; offset[5] = 8;
		offset[6] = 11; offset[7] = 16; offset[8] = 17; offset[9] = 21; offset[10] = 24; offset[11] = 26;
		offset[12] = 28;

		vertex_id_type* dest = new vertex_id_type[28];
		dest[0] = 1; dest[1] = 3; dest[2] = 7; dest[3] = 3; dest[4] = 5; dest[5] = 9;
		dest[6] = 5; dest[7] = 6; dest[8] = 7; dest[9] = 9; dest[10] = 11; dest[11] = 0;
		dest[12] = 5; dest[13] = 8; dest[14] = 9; dest[15] = 10; dest[16] = 5; dest[17] = 0;
		dest[18] = 4; dest[19] = 9; dest[20] = 11; dest[21] = 5; dest[22] = 6; dest[23] = 8;
		dest[24] = 2; dest[25] = 8;dest[26] = 2; dest[27] = 5;

		edge_data_type* weight = new edge_data_type[28];
		weight[0] = 1; weight[1] = 3; weight[2] = 7; weight[3] = 3; weight[4] = 5; weight[5] = 9;
		weight[6] = 5; weight[7] = 6; weight[8] = 7; weight[9] = 9; weight[10] = 11; weight[11] = 0;
		weight[12] = 5; weight[13] = 8; weight[14] = 9; weight[15] = 5; weight[16] = 0; weight[17] = 2;
		weight[18] = 4; weight[19] = 9; weight[20] = 11; weight[21] = 5; weight[22] = 2; weight[23] = 8;
		weight[24] = 2; weight[25] = 5; weight[26] = 2; weight[27] = 5;

		csrResult.vertexNum = 12;
		csrResult.edgeNum = 28;
		csrResult.csr_offset = offset;
		csrResult.csr_dest = dest;
		csrResult.csr_weight = weight;
	}
	else
	{
		assert_msg(false, "未找到对应的Toy Graph");
	}

	return csrResult;
}