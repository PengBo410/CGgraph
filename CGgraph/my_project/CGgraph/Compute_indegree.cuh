#pragma once

#include "Basic/basic_include.cuh"

class Compute_indegree
{

public:
    count_type vertexNum;
	countl_type edgeNum;
	count_type zeroOutDegreeNum;
	count_type noZeroOutDegreeNum;	

	countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;
	offset_type* outDegree;

    offset_type* inDegree;
    bool hasIndegree = false;
    bool hasOutdegree = false;


public:
    Compute_indegree(const CSR_Result_type& csrResult, const count_type zeroOutDegreeNum_):
        vertexNum(0),
		edgeNum(0),
		zeroOutDegreeNum(0),
		noZeroOutDegreeNum(0)
    {
        // Get Graph CSR
		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;

		zeroOutDegreeNum = zeroOutDegreeNum_;
		noZeroOutDegreeNum = vertexNum - zeroOutDegreeNum;
		Msg_info("zeroOutDegreeNum = %zu, noZeroOutDegreeNum = %zu", 
			static_cast<uint64_t>(zeroOutDegreeNum), static_cast<uint64_t>(noZeroOutDegreeNum));

        //> 初始化Graph
        timer constructTime;
		initGraph();
		Msg_info("Init-Graph: Used time: %.2lf (ms)", constructTime.get_time_ms());

        //> 构建inDegree
        constructTime.start();
        compute_inDegree();
        Msg_info("Compute-inDegree: Used time: %.2lf (ms)", constructTime.get_time_ms());
    }

    offset_type* getIndegree()
    {
        if(hasIndegree) return inDegree;
        else{
            assert_msg(false, "inDegree has not finish");
            return nullptr;
        }
    }

    offset_type* getOutdegree()
    {
        if(hasOutdegree) return outDegree;
        else{
            assert_msg(false, "outDegree has not finish");
            return nullptr;
        }
    }

private:

    /* **********************************************************
	 * Func: Host Function , Init Graph
	 * **********************************************************/
	void initGraph()
	{
		if (vertexNum >= std::numeric_limits<count_type>::max()){assert_msg(false, "vertexNum >= count_type:max()");}		
		if (edgeNum >= std::numeric_limits<countl_type>::max()){assert_msg(false, "vertexNum >= countl_type:max()");}
			
		// Init outDegree
		inDegree = new offset_type[vertexNum];
        memset(inDegree, 0, sizeof(offset_type) * vertexNum);

        outDegree = new offset_type[vertexNum];
        memset(outDegree, 0, sizeof(offset_type) * vertexNum);

        // Get outDegree
        hasOutdegree = true;
        omp_parallel_for(vertex_id_type vertexId = 0; vertexId < vertexNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}
        //check
        bool check = true;
        if(check)
        {
            countl_type edgeNum_total = 0;
            #pragma omp parallel for reduction(+: edgeNum_total)
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                edgeNum_total += outDegree[vertexId];
            }
            
            assert_msg((edgeNum_total == edgeNum), "edgeNum_total = %zu", static_cast<uint64_t>(edgeNum_total));
            Msg_finish("inDegree finish check");
        }

	} // end of func [initGraph()]

    /* **********************************************************
	 * Func: Host Function , Compute-inDegree
	 * **********************************************************/
    void compute_inDegree()
    {
        TaskSteal* taskSteal = new TaskSteal();
        size_t totalWorkloads = 0;
        totalWorkloads = taskSteal->twoStage_taskSteal<size_t, size_t>(static_cast<size_t>(noZeroOutDegreeNum),
            [&](size_t& current, size_t& local_workloads)
            {
                size_t end = current + VERTEXWORK_CHUNK;
                size_t length = VERTEXWORK_CHUNK;
                if (end >= noZeroOutDegreeNum) length = noZeroOutDegreeNum - current;

                for (size_t in = 0; in < length; in++)
                {
                    countl_type nbr_first = csr_offset[current + in];
                    countl_type nbr_end = csr_offset[current + in + 1];

                    for (countl_type nbrId = nbr_first; nbrId < nbr_end; nbrId++)
                    {
                        vertex_id_type dest = csr_dest[nbrId];
                        __sync_fetch_and_add(&inDegree[dest], 1);
                    }
                }
            },
            VERTEXWORK_CHUNK
        );

        hasIndegree = true;

        //check
        bool check = true;
        if(check)
        {
            countl_type edgeNum_total = 0;
            #pragma omp parallel for reduction(+: edgeNum_total)
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                edgeNum_total += inDegree[vertexId];
            }
            
            assert_msg((edgeNum_total == edgeNum), "edgeNum_total = %zu", static_cast<uint64_t>(edgeNum_total));
            Msg_finish("inDegree finish check");
        }
    }

public:
    /* **********************************************************
	 * Func: Host Function , 获取最大的前n个inDegree的vertexId和值
	 * **********************************************************/
    void sortIndegree(uint64_t printfNum = 100)
    {
        struct In_type{
            vertex_id_type vertexId = 0;
            offset_type indegreeValue = 0;
        };


        std::vector<In_type> inDegree_vec;
        inDegree_vec.resize(vertexNum);
        omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId ++)
        {
            In_type in;
            in.vertexId = vertexId;
            in.indegreeValue = inDegree[vertexId];
            //in.indegreeValue = outDegree[vertexId];
            //in.indegreeValue = inDegree[vertexId] + outDegree[vertexId];

            inDegree_vec[vertexId] = in;
        }

        std::sort(inDegree_vec.begin(), inDegree_vec.end(),
            [&](In_type& a, In_type& b)-> bool
            {
                if(a.indegreeValue > b.indegreeValue) return true; // 如果定义为a.indegreeValue >= b.indegreeValue,会出现:Segmentation fault (core dumped)错误
                else return false;
            }
        );


        printfNum = (printfNum < vertexNum) ? printfNum : vertexNum;
        for(uint64_t i =0; i < printfNum; i++)
        {
            printf("(%2zu): id (%u) - (in)Degree (%u) -> {%u}\n", i,
                 inDegree_vec[i].vertexId, inDegree_vec[i].indegreeValue, inDegree_vec[i].vertexId/1966050);
        }

    }
    
};// end of class [Compute_indegree]

