#pragma once


#define BLOCK_PARTITION_CHECK

class Block_partition
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

    Algorithm_type algorithm;

    count_type segment = std::numeric_limits<uint16_t>::max() * 10 * 3; 
    //count_type segment = 4;
    count_type partition = 0;
    


    struct Edge_type{
            vertex_id_type src;
            vertex_id_type dest;
            edge_data_type weight;
        };

    struct Block_type{
        //count_type vertexNum_block = 0;
        countl_type edgeNum_block = 0;
        vertex_id_type min_src = std::numeric_limits<vertex_id_type>::max();
        vertex_id_type max_src = 0;
        vertex_id_type min_dest = std::numeric_limits<vertex_id_type>::max();
        vertex_id_type max_dest = 0;

        uint64_t capacity = 0;

        count_type block_x = 0;
        count_type block_y = 0;
        tbb::concurrent_vector<Edge_type> block_edge_data;
    };

    tbb::concurrent_vector<tbb::concurrent_vector<Block_type>> block_vec_2d;

    dense_bitset bitmap_allBlock; 
    count_type** vertexNum_allBlock;
    countl_type** edgeNum_allBlock;

    countl_type*** csr_offset_allBlock;
    vertex_id_type*** csr_dest_allBlock;
    edge_data_type*** csr_weight_allBlock;

    struct Block_result_type{

        dense_bitset bitmap_allBlock; 

        count_type** vertexNum_allBlock;
        countl_type** edgeNum_allBlock;

        countl_type*** csr_offset_allBlock;
        vertex_id_type*** csr_dest_allBlock;
        edge_data_type*** csr_weight_allBlock;

        count_type segemnt = 0;
        count_type partition = 0;

    };

public:

   /* **************************************************************************
    *                                [Construct]
    * @param [const CSR_Result_type& csrResult]    CSR
    * @param [const count_typezeroOutDegreeNum_]   Non-zero vertices
    * @param [const size_t deviceNum_]             GPU Num
    * **************************************************************************/
    Block_partition(const CSR_Result_type& csrResult, const count_type zeroOutDegreeNum_, Algorithm_type algorithm_):
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

		algorithm = algorithm_;

        //> 
        timer constructTime;
		initGraph();
		Msg_info("Init-Graph: Used time: %.2lf (ms)", constructTime.get_time_ms());

        constructTime.start();
        buildBlock();
        Msg_info("Build-Block: Used time: %.2lf (ms)", constructTime.get_time_ms());
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
		outDegree = new offset_type[noZeroOutDegreeNum];
        // Get outDegree
        omp_parallel_for(vertex_id_type vertexId = 0; vertexId < noZeroOutDegreeNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}


        partition =(noZeroOutDegreeNum + segment - 1 ) / segment;
        count_type block_num = partition * partition;
        Msg_finish("Segment: %u, 1-D: %u, 2-D: %u", segment, partition, block_num);

        bitmap_allBlock.resize(block_num);
        bitmap_allBlock.fill();


	} // end of func [initGraph()]

    void buildBlock()
    {
        block_vec_2d.resize(partition);       
        for (size_t i = 0; i < partition; i++)
        {
            block_vec_2d[i].resize(partition);
        }

        omp_parallel_for (size_t blockx = 0; blockx < partition; blockx++)
        {
            for(size_t blocky = 0; blocky < partition; blocky++)
            {
                Block_type block;
                block.block_x = blockx;
                block.block_y = blocky;

                block_vec_2d[blockx][blocky] = block;
            }
        }


        TaskSteal* taskSteal = new TaskSteal();
        size_t totalWorkloads = 0;

        timer buildBlock_time;	
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

                    vertex_id_type src = current + in; 
                    vertex_id_type block_x = src / segment; 
                    vertex_id_type block_y = 0;
                    Edge_type edge_type;

                    vertex_id_type dest;
                    for (countl_type i = nbr_first; i < nbr_end; i++)
                    {
                        dest = csr_dest[i];
                        block_y = dest / segment;

                        edge_type.src = src % segment;
                        edge_type.dest = dest % segment;
                        edge_type.weight = csr_weight[i];

                        assert_msg((edge_type.src < segment) && (edge_type.dest < segment),
                            "edge_type.src = %u, edge_type.dest = %u", edge_type.src, edge_type.dest
                        );
                    
                        assert_msg((block_x < partition) && (block_y < partition),
                            "block_x = %u, block_y = %u", block_x, block_y
                        );


                        Block_type& block = block_vec_2d[block_x][block_y];

                        __sync_fetch_and_add(&block.edgeNum_block, 1);
                        if(block.min_src > edge_type.src) Gemini_atomic::write_min(&block.min_src, edge_type.src);
                        if(block.max_src < edge_type.src) Gemini_atomic::write_max(&block.max_src, edge_type.src);

                        if(block.min_dest > edge_type.dest) Gemini_atomic::write_min(&block.min_dest, edge_type.dest);
                        if(block.max_dest < edge_type.dest) Gemini_atomic::write_max(&block.max_dest, edge_type.dest);

                       
                        
                        block.block_edge_data.emplace_back(edge_type);

                        
                    }
                }
            },
            VERTEXWORK_CHUNK
        );
        Msg_info("Build Block ：%.2f(ms)", buildBlock_time.get_time_ms());

        //Check
        #ifdef BLOCK_PARTITION_CHECK
        countl_type totalEdge = 0;
        #pragma omp parallel for reduction(+: totalEdge)
        for(count_type blockRow = 0; blockRow < partition; blockRow ++)
        {
            countl_type local_edgeNum = 0;
            for(count_type blockColumn = 0; blockColumn < partition; blockColumn ++)
            {
                Block_type block = block_vec_2d[blockRow][blockColumn];           
                local_edgeNum += block.edgeNum_block;
            }
            totalEdge += local_edgeNum;
        }
        assert_msg((edgeNum == totalEdge), "totalEdge = %zu", static_cast<uint64_t>(totalEdge));
        Msg_finish("Build Block Check Finish !");
        #endif


        //> 
        vertexNum_allBlock = new count_type* [partition];
        edgeNum_allBlock = new countl_type* [partition];
        csr_offset_allBlock = new countl_type** [partition];
        csr_dest_allBlock = new vertex_id_type** [partition];
        csr_weight_allBlock = new edge_data_type** [partition];

        for (size_t i = 0; i < partition; i++)
        {
            vertexNum_allBlock[i] = new count_type [partition];
            edgeNum_allBlock[i] = new countl_type [partition];
            csr_offset_allBlock[i] = new countl_type* [partition];
            csr_dest_allBlock[i] = new vertex_id_type* [partition];
            csr_weight_allBlock[i] = new edge_data_type* [partition];
        }
        



        //
        typedef std::pair<vertex_id_type, edge_data_type> nbr_pair_type;//first is dest，second is weight
        tbb::concurrent_vector<tbb::concurrent_vector<nbr_pair_type>> adjlist;//block
        adjlist.resize(segment);
        int count = 0;
        for(count_type blockRow = 0; blockRow < partition; blockRow ++)
        {           
            for(count_type blockColumn = 0; blockColumn < partition; blockColumn ++)
            {
                Block_type& block = block_vec_2d[blockRow][blockColumn];

                if(block.edgeNum_block == 0)
                {
                    bitmap_allBlock.clear_bit(blockRow * partition + blockColumn);
                    continue;
                } 

                             
                omp_parallel_for(countl_type edgeId = 0; edgeId < block.edgeNum_block; edgeId++)
                {
                    Edge_type edge = block.block_edge_data[edgeId];
                    adjlist[edge.src].emplace_back(std::make_pair(edge.dest, edge.weight));
                }

                //check
                #ifdef BLOCK_PARTITION_CHECK
                countl_type totalEdgeCount = 0;
                #pragma omp parallel for reduction(+: totalEdgeCount)
                for(vertex_id_type vertexId = 0; vertexId < segment; vertexId ++)
                {
                    totalEdgeCount += adjlist[vertexId].size();
                }
                assert_msg(totalEdgeCount == block.edgeNum_block, "totalEdgeCount = %zu", static_cast<uint64_t>(totalEdgeCount));
                #endif


                //> build CSR
                count_type vertexNum_block = segment;
                if(blockColumn == (partition - 1)) vertexNum_block = noZeroOutDegreeNum - (blockColumn * segment);

                //printf("v = %u, e = %u\n",vertexNum_block,  block.edgeNum_block);

                vertexNum_allBlock[blockRow][blockColumn] =  vertexNum_block;
                edgeNum_allBlock[blockRow][blockColumn] =  block.edgeNum_block;

                csr_offset_allBlock[blockRow][blockColumn] = new countl_type [vertexNum_block + 1];
                csr_offset_allBlock[blockRow][blockColumn][0] = 0;
                for (count_type i = 1; i <= vertexNum_block; i++)
                {
                    csr_offset_allBlock[blockRow][blockColumn][i] =  csr_offset_allBlock[blockRow][blockColumn][i - 1] +
                        static_cast<countl_type>(adjlist[i-1].size());
                }
                assert_msg(csr_offset_allBlock[blockRow][blockColumn][vertexNum_block] == block.edgeNum_block, 
                    "csr_offset_allBlock[blockRow][blockColumn][vertexNum_block] != edgeNum, csr_offset_allBlock = %u, block.edgeNum_block = %u,",
                    csr_offset_allBlock[blockRow][blockColumn][vertexNum_block], block.edgeNum_block);

                csr_dest_allBlock[blockRow][blockColumn] = new vertex_id_type [block.edgeNum_block];
                csr_weight_allBlock[blockRow][blockColumn] = new edge_data_type [block.edgeNum_block];

                omp_parallel_for(count_type i = 0; i < vertexNum_block; i++)
                {
                    tbb::concurrent_vector<nbr_pair_type> nbr = adjlist[i];
                    countl_type offset = csr_offset_allBlock[blockRow][blockColumn][i];
                    for (countl_type j = 0; j < nbr.size(); j++)
                    {
                        csr_dest_allBlock[blockRow][blockColumn][offset + j] = nbr[j].first;
                        csr_weight_allBlock[blockRow][blockColumn][offset + j] = nbr[j].second;
                    }
                }
          
                for(vertex_id_type vertexId = 0; vertexId < segment; vertexId ++)
                {
                    adjlist[vertexId].clear();
                }

            }
            
        }


        //> check
        // #ifdef BLOCK_PARTITION_CHECK
        // for(count_type blockRow = 0; blockRow < partition; blockRow ++)
        // {           
        //     for(count_type blockColumn = 0; blockColumn < partition; blockColumn ++)
        //     {
        //         count_type vertexNum_ = vertexNum_allBlock[blockRow][blockColumn];
        //         count_type edgeNum_ = edgeNum_allBlock[blockRow][blockColumn];

        //         printf("Block[%u,%u]:\n", blockRow, blockColumn);
        //         printf("\tcsr_offset:");
        //         for (size_t i = 0; i <= vertexNum_; i++)
        //         {
        //             printf("%u, ", csr_offset_allBlock[blockRow][blockColumn][i]);
        //         }
        //         printf("\n");
        //         printf("\tcsr_dest:");
        //         for (size_t i = 0; i < edgeNum_; i++)
        //         {
        //             printf("%u, ", csr_dest_allBlock[blockRow][blockColumn][i]);
        //         }
        //         printf("\n");
        //         printf("\tcsr_weight:");
        //         for (size_t i = 0; i < edgeNum_; i++)
        //         {
        //             printf("%u, ", csr_weight_allBlock[blockRow][blockColumn][i]);
        //         }
        //         printf("\n");
                
        //     }
        // }
        // #endif

    }

public:
    Block_result_type get_blockResult()
    {
        tbb::concurrent_vector<tbb::concurrent_vector<Block_type>>().swap(block_vec_2d);
		Msg_finish("block_vec_2d Free Finished");


        Block_result_type block_result_type;

        block_result_type.bitmap_allBlock = bitmap_allBlock;
        block_result_type.vertexNum_allBlock = vertexNum_allBlock;
        block_result_type.edgeNum_allBlock = edgeNum_allBlock;
        block_result_type.csr_offset_allBlock = csr_offset_allBlock;
        block_result_type.csr_dest_allBlock = csr_dest_allBlock;
        block_result_type.csr_weight_allBlock = csr_weight_allBlock;

        block_result_type.segemnt = segment;
        block_result_type.partition = partition;

        return block_result_type;
    }
   




};// end of class [Block_partition]

