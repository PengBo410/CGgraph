#pragma once

#include "../Type/data_type.hpp"
#include <string>
#include <vector>

struct CSR_Result_type
{
	count_type vertexNum;
	countl_type edgeNum;
	countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;

	count_type noZeroOutDegreeNum;
};

struct CSC_Result_type
{
	count_type vertexNum;
	countl_type edgeNum;
	countl_type* csc_offset;
	vertex_id_type* csc_src;
	edge_data_type* csc_weight;
};

struct Degree_type
{
	count_type vertexNum;
	countl_type edgeNum;
	offset_type* outDegree;
	offset_type* inDegree;	
};

typedef std::vector<std::pair<vertex_id_type, vertex_id_type>> BoundPointer_type;


/****************************************************************************
 *                             [Re - OrderMethod]
 ****************************************************************************/
enum OrderMethod {
	NATIVE,
	OUTDEGREE_NODE,
	OUTDEGREE_SOCKET,
	OUTDEGREE_THREAD,
	GORDER,
	RCM,
	RCM_LAST_ZERO_OUTDEGREE,
	RCM_LAST_ONE_ZERO_DEGREE, 

	
	OUTDEGREE_NODE_dec,
	DEGREE_NODE,

	
	RCM_INDEGREE,
	RCM_OUTDEGREE,
	NORMAL_OUTDEGREE_DEC,

	UnDirected_NATIVE,
	UnDirected_RCM,
	UnDirected_RCM_LAST_ZERO_OUTDEGREE,

	//??????outDegree??????
	MY_BFS_OUT,
	MY_BFS_IN,
	MY_BFS_OUT_IN,

	//??????inDegree??????
	MY_IMAX_BFS_OUT,
	MY_IMAX_BFS_IN,

	//degree
	IN_DEGREE_DEC
};


/****************************************************************************
 *                              [Degree ≈≈–Ú]
 ****************************************************************************/
enum SortDegree {
	OUTDEGREE,
	INDEGREES,
	DEGREE
};

/****************************************************************************
 *                            [Graph Representation]
 ****************************************************************************/
enum GraphRepresentation {
	CSR,
	CSC
};


/****************************************************************************
 *                              [GraphFile Struct]
 ****************************************************************************/
struct GraphFile_type {
	std::string graphFile = "";
	size_t vertices = 0;

	size_t common_root = 0;
	size_t edges = 0;
	std::string old2newFile = "";
	std::string addtitionFile = "";

	std::string csrOffsetFile = "";
	std::string csrDestFile = "";
	std::string csrWeightFile = "";
};



/****************************************************************************
 *                              [Adaptive]
 ****************************************************************************/
struct Adaptive_type {
	double rate_cpu = 0.0; 
	double rate_gpu = 0.0; 
	double rate_pcie = 0.0;
	std::string adaptiveFile = "";
};

enum RATE_Type {
	CPU,
	GPU,
	PCIe,
	Reduce
};

struct Adaptive_info_type
{
	RATE_Type rate_type;
	//std::string graphName = "";
	//std::string algorithm = "";
	size_t ite = 0; 
	size_t vertexNum_workload = 0; 
	size_t edgeNum_workload = 0; 
	double score = 0.0; 
	double time = 0.0; 
	double rate = 0.0; 
};


enum Algorithm_type
{
	BFS,
	SSSP,
	CC,
	PR,
	UNDEFINE
};

/****************************************************************************
 *                            [Workload]
 ****************************************************************************/
enum Workload_heterogeneous  {
	LIGHT,
	HEAVY
};


/****************************************************************************
 *                            [Workload]
 ****************************************************************************/
enum Engine_type  {
	COORDINATION,
	COOPERATION,
	SINGLE_CORE,
	MULTI_CORE,
	SINGLE_GPU,
	BLOCK_2D,

	//Other Compute
	COMPUTE_INDEGREE, // Use CSR Compute Indgree
	REORDER,
	ACTIVEBLOCK
	
};


struct CheckInfo_type {
	std::string graphName = "";
	Algorithm_type algorithm = Algorithm_type::UNDEFINE;
	size_t root = 0;
};




typedef std::vector<std::pair<vertex_id_type, vertex_id_type>> EdgeList_noWeight_type;
