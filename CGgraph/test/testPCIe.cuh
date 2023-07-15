#pragma once

#include "../src/Basic/basic_include.cuh"

//PCIe: 12.04 GB/s,  Byte: 12040000, time: 1.01 (ms)  (3010000个uint32_t)
void test_PCIe()
{
    uint32_t num = 10000;
    int ite = 150;
    uint32_t increment = 500000;

    for (int i = 0; i < ite; i++)
    {
       uint32_t size = num + i * increment;
       //size = 42000000;
       uint32_t* array;
       CUDA_CHECK(MALLOC_HOST(&array, (size)));

       uint32_t* array_device;
       CUDA_CHECK(MALLOC_DEVICE(&array_device, (size)));

       timer t;
       H2D(array_device, array, size);

       Msg_info("[%2d] Length: %u, Byte: %u, time: %.2lf (ms)", i, (size), size * 4, t.get_time_ms());

       CUDA_CHECK(FREE_HOST(array));
       CUDA_CHECK(FREE_DEVICE(array_device));
    }
}
    
    
    void testTwoLevel()
    {
        std::string graphName = "twitter2010";//gsh2015tpd	
        OrderMethod orderMethod = OrderMethod::MY_IMAX_BFS_IN; //如果是OUTDEGREE_THREAD需要开启REORDER宏定义，tasksteal要用，TODO
        int64_t root = 892741;  //23841917(gsh2015tpd)   | 892741(twitter2010)  |  25689(friendster),746480(RCM-LAST)
        count_type runs = 1;
        bool has_csrFile = 1;
        bool logResult = true;
        bool rootTranfer = true;
        bool RCM_LAST_ZERO_OUTDEGREE_check = true;
        Algorithm_type algorithm = Algorithm_type::SSSP;
        Engine_type  engine_type = Engine_type::MULTI_CORE;
        int useDeviceNum = 2;

        //日志
        setLogNFS("2023-6-7.csv", false);//日志

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
            countl_type* csr_offset = load_binFile<countl_type>(csrOffsetFile, static_cast<uint64_t>(vertices + 1));
            vertex_id_type* csr_dest = load_binFile<vertex_id_type>(csrDestFile, static_cast<uint64_t>(edges));
            edge_data_type* csr_weight = load_binFile<edge_data_type>(csrWeightFile, static_cast<uint64_t>(edges));

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

        //检查
        vertex_id_type* old2new;
        if ((orderMethod != OrderMethod::NATIVE) && rootTranfer)
        {
            old2new = load_binFile<vertex_id_type>(file_old2new, vertices);
        }


        // 检查CSR nbr的排序情况
        nbrSort_taskSteal(csrResult);



        /* **********************[Begin]************************* */

        count_type per = std::numeric_limits<uint16_t>::max() * 10 * 3 * 2;
        per = 6200000;
        count_type partition =(csrResult.vertexNum + per - 1 ) / per;
        count_type block_num = partition * partition;
        Msg_info("Per: %u, 1-D: %u, 2-D: %u", per, partition, block_num);

        struct Edge_type{
            vertex_id_type src;
            vertex_id_type dest;
        };

        struct Block_type{
            //count_type vertexNum_block = 0;
            countl_type edgeNum_block = 0;
            vertex_id_type min_src = std::numeric_limits<vertex_id_type>::max();
            vertex_id_type max_src = 0;
            vertex_id_type min_dest = std::numeric_limits<vertex_id_type>::max();
            vertex_id_type max_dest = 0;

            uint64_t capacity = 0;//占用的容量

            count_type block_x = 0;
            count_type block_y = 0;
            tbb::concurrent_vector<Edge_type> block_edge_data;
        };

        tbb::concurrent_vector<tbb::concurrent_vector<Block_type>> block_vec_2d;
        


        //> 给所有的block赋值
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
        totalWorkloads = taskSteal->twoStage_taskSteal<size_t, size_t>(static_cast<size_t>(csrResult.vertexNum),
            [&](size_t& current, size_t& local_workloads)
            {
                size_t end = current + VERTEXWORK_CHUNK;
                size_t length = VERTEXWORK_CHUNK;
                if (end >= csrResult.vertexNum) length = csrResult.vertexNum - current;

                for (size_t in = 0; in < length; in++)
                {
                    countl_type nbr_first = csrResult.csr_offset[current + in];
                    countl_type nbr_end = csrResult.csr_offset[current + in + 1];

                    vertex_id_type src = current + in; 
                    vertex_id_type block_x = src / per; 
                    vertex_id_type block_y = 0;
                    Edge_type edge_type;

                    vertex_id_type dest;
                    for (countl_type i = nbr_first; i < nbr_end; i++)
                    {
                        dest = csrResult.csr_dest[i];
                        block_y = dest / per;

                        edge_type.src = src % per;
                        edge_type.dest = dest % per;

                        assert_msg((edge_type.src < per) && (edge_type.dest < per),
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

                        //__sync_lock_test_and_set(&block.block_x, block_x);
                        //__sync_lock_test_and_set(&block.block_y, block_y);
                        
                        block.block_edge_data.emplace_back(edge_type);

                        // printf("GlobalId(%u-%u) -> LocalId%u-%u), BlockId(%u, %u) %u\n", 
                        //     src, dest, edge_type.src, edge_type.dest, block_x, block_y, block.edgeNum_block);
                    }
                }
            },
            VERTEXWORK_CHUNK
        );
        Msg_info("Build Block 用时：%.2f(ms)", buildBlock_time.get_time_ms());
        

        bool check = true;
        if(true)
        {
            countl_type totalEdge = 0;
            //printf("partition = %u\n", partition);
            #pragma omp parallel for reduction(+: totalEdge)
            for(count_type blockRow = 0; blockRow < partition; blockRow ++)
            {
                countl_type local_edgeNum = 0;
                for(count_type blockColumn = 0; blockColumn < partition; blockColumn ++)
                {
                    Block_type block = block_vec_2d[blockRow][blockColumn];
                    //printf("Block(%u,%u). edgeNum_block = %u\n", block.block_x, block.block_y, block.edgeNum_block);
                    local_edgeNum += block.edgeNum_block;
                }
                totalEdge += local_edgeNum;
            }
            assert_msg((csrResult.edgeNum == totalEdge), "totalEdge = %zu", static_cast<uint64_t>(totalEdge));
            Msg_finish("Bild Block Check Finish !");
        }

        bool log = true;
        if(log)
        {
            std::string outFile = "/home/pengjie/vs_log/2023-6-12-twitter2010-MY_IMAX_BFS_IN(10-10).csv";
            std::ofstream out_file;
            out_file.open(outFile.c_str(),
                std::ios_base::out | std::ios_base::binary);//Opens as a binary read and writes to disk
            if (!out_file.good()) assert_msg(false, "Error opening out-file: %s", outFile.c_str());


            for(count_type blockRow = 0; blockRow < partition; blockRow ++)
            {
                std::stringstream ss_log;
                ss_log.clear();
                for(count_type blockColumn = 0; blockColumn < partition; blockColumn ++)
                {
                    Block_type block = block_vec_2d[blockRow][blockColumn];

                    //只计算边数
                    out_file << block.edgeNum_block << ",";

                    // logstream(LOG_INFO) << "Block[" << block.block_x <<", " << block.block_y << "] :"
                    //     << " edgeNum_block = " << block.edgeNum_block
                    //     << ", src (" << block.min_src << ", " << block.max_src
                    //     << "), dest (" << block.min_dest << ", " << block.max_dest << ")"
                    //     << std::endl;

                    // logstream(LOG_INFO) << "Block[" << block.block_x <<"-" << block.block_y << "],"
                    //     << block.edgeNum_block << "," << block.min_src << "," << block.max_src << ","
                    //     << block.min_dest << "," << block.max_dest << std::endl;
                }
                //ss_log << std::endl;
                out_file << std::endl;
            }
            out_file.close();

            
        }



        //检查Block的顶点的平衡性
        // tbb::concurrent_vector<tbb::concurrent_vector<vertex_id_type>> adjlist;
        // adjlist.resize(per);
        // int count = 0;
        // for(count_type blockRow = 0; blockRow < partition; blockRow ++)
        // {
        //     for(count_type blockColumn = 0; blockColumn < partition; blockColumn ++)
        //     {
        //         Block_type& block = block_vec_2d[blockRow][blockColumn];

        //         if(block.edgeNum_block == 0) continue;

        //         Msg_info("(%d) Block[%u,%u]: block_edgeNum = %u, src:(%u, %u), dest:(%u, %u)", count, blockRow, blockColumn,
        //             block.edgeNum_block,block.min_src, block.max_src, block.min_dest, block.max_dest);

        //         //生成csr
                
        //         omp_parallel_for(countl_type edgeId = 0; edgeId < block.edgeNum_block; edgeId++)
        //         {
        //             Edge_type edge = block.block_edge_data[edgeId];
        //             adjlist[edge.src].emplace_back(edge.dest);
        //         }

        //         //check
        //         countl_type totalEdgeCount = 0;
        //         #pragma omp parallel for reduction(+: totalEdgeCount)
        //         for(vertex_id_type vertexId = 0; vertexId < per; vertexId ++)
        //         {
        //             totalEdgeCount += adjlist[vertexId].size();
        //         }
        //         assert_msg(totalEdgeCount == block.edgeNum_block, "totalEdgeCount = %zu", totalEdgeCount);

        //         //不同的bucket
        //         count_type* bucket = new count_type[5];
        //         memset(bucket, 0, sizeof(count_type) * 5);
        //         for(vertex_id_type vertexId = 0; vertexId < per; vertexId ++)
        //         {
        //             if(adjlist[vertexId].size() <= 16) bucket[0] ++;
        //             else if(17 <= adjlist[vertexId].size() && adjlist[vertexId].size() <= 32) bucket[1] ++;
        //             else if(33 <= adjlist[vertexId].size() && adjlist[vertexId].size() <= 256) bucket[2] ++;
        //             else if(257 <= adjlist[vertexId].size() && adjlist[vertexId].size() <= 512) bucket[3] ++;
        //             else if(513 <= adjlist[vertexId].size()) bucket[4] ++;
        //             else assert_msg(false, "");
        //         }

        //         //check
        //         totalEdgeCount = 0;
        //         for(int bucketId = 0; bucketId < 5; bucketId++)
        //         {
        //             totalEdgeCount += bucket[bucketId];
        //         }
        //         assert_msg(totalEdgeCount == per, "totalEdgeCount = %zu", totalEdgeCount);

        //         for(int bucketId = 0; bucketId < 5; bucketId++)
        //         {
        //             printf("\tBucketId[%u] = %u (%.2lf %%)\n", 
        //                 bucketId, bucket[bucketId], (double)bucket[bucketId] / per * 100);
        //         }

        //         // 释放
        //         delete[] bucket;              
        //         for(vertex_id_type vertexId = 0; vertexId < per; vertexId ++)
        //         {
        //             adjlist[vertexId].clear();
        //         }

        //         count++;
        //     }
            
        // }


       

        // 对edge_vec中的边就行排序
        // std::sort(block.block_edge_data.begin(), block.block_edge_data.end(),
        //     [&](Edge_type& a, Edge_type& b) -> bool
        //     {
        //         if(a.src < b.src) return true;
        //         else if(a.src == b.src)
        //         {
        //             if(a.dest <= b.dest) return true;
        //             else return false;
        //         }
        //         else return false;
        //     }
        // );

        

        

        

    }// testTwoLevel
    


