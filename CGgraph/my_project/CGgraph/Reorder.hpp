#pragma once

#include "Basic/basic_include.cuh"
#include <queue>

class Reorder
{
public:
    count_type vertexNum;
	countl_type edgeNum;

    vertex_id_type* old2new;
	vertex_id_type* new2old;

    countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;
    offset_type* outDegree;
	offset_type* inDegree;

    CSR_Result_type csrResult_reorder;

private:
	struct Vid_degree_type
	{
		vertex_id_type vertex_id;
		offset_type degree;
	};
	std::vector<Vid_degree_type> sort_vec;

	TaskSteal* taskSteal;

     bool isToyGraph = false;

public:
	
	typedef std::pair<vertex_id_type, edge_data_type> nbr_pair_type;
	typedef std::vector<nbr_pair_type> neighbor_type;
	std::vector<neighbor_type> adjlist_sort;



    Reorder(const CSR_Result_type& csrResult, offset_type* outDegree_, offset_type* inDegree_):
        vertexNum(0),
        edgeNum(0)
    {
        vertexNum = csrResult.vertexNum;
        edgeNum = csrResult.edgeNum;
        csr_offset = csrResult.csr_offset;
        csr_dest = csrResult.csr_dest;
        csr_weight = csrResult.csr_weight;

        outDegree = outDegree_;
        inDegree = inDegree_;
    }


    void generateReorderGraphFile(const OrderMethod& orderMethod, std::string graph, bool isToyGraph_ = false)
	{
        isToyGraph = isToyGraph_;
        if(!isToyGraph) createFloder((BASE_GRAPHFILE_PATH + graph + "/").c_str());
        

        //> MY_BFS_OUT
        if(orderMethod == OrderMethod::MY_BFS_OUT)
        {
            sort_vec.resize(vertexNum);
            sortDegree(SortDegree::OUTDEGREE);
            std::vector<vertex_id_type> order;
			std::vector<count_type> additionInfo;

            timer t;
            MY_BFS_OUT(order, additionInfo);
            Msg_info("MY_BFS_OUT, time：%f (ms)", t.get_time_ms());
			Msg_info("MY_BFS_OUT zeroDegreenNum = %u (%.2f%%)", additionInfo[0], ((double)additionInfo[0] / vertexNum) * 100);

             if(isToyGraph)
            {
                printf("new: ");
                for (count_type i = 0; i < vertexNum; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
                printf("\nold: ");
                for (count_type i = 0; i < vertexNum; i++) printf("%4zu ", static_cast<uint64_t>(order[i]));
                printf("\n");
            }

            saveCSR_reorderFile(order, additionInfo,
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "..."
            );
            
            std::vector<vertex_id_type>().swap(order);
			Msg_info("MY_BFS_OUT order clear()");
        }

        //> MY_BFS_IN
        else if(orderMethod == OrderMethod::MY_BFS_IN)
        {
            sort_vec.resize(vertexNum);
            sortDegree(SortDegree::OUTDEGREE);
            std::vector<vertex_id_type> order;
			std::vector<count_type> additionInfo;

            timer t;
            MY_BFS_IN(order, additionInfo);
            Msg_info("MY_BFS_IN, time：%f (ms)", t.get_time_ms());
			Msg_info("MY_BFS_IN zeroDegreenNum = %u (%.2f%%)", additionInfo[0], ((double)additionInfo[0] / vertexNum) * 100);

            if(isToyGraph)
            {
                printf("new: ");
                for (count_type i = 0; i < vertexNum; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
                printf("\nold: ");
                for (count_type i = 0; i < vertexNum; i++) printf("%4zu ", static_cast<uint64_t>(order[i]));
                printf("\n");
            }

            saveCSR_reorderFile(order, additionInfo,
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "..."
            );
            
            std::vector<vertex_id_type>().swap(order);
			Msg_info("MY_BFS_IN order clear");
        }

        //> MY_BFS_OUT_IN
        else if(orderMethod == OrderMethod::MY_BFS_OUT_IN)
        {
            sort_vec.resize(vertexNum);
            sortDegree(SortDegree::OUTDEGREE);
            std::vector<vertex_id_type> order;
			std::vector<count_type> additionInfo;

            timer t;
            MY_BFS_OUT_IN(order, additionInfo);
            Msg_info("MY_BFS_OUT_IN time：%f (ms)", t.get_time_ms());
			Msg_info("MY_BFS_OUT_IN zeroDegreenNum = %u (%.2f%%)", additionInfo[0], ((double)additionInfo[0] / vertexNum) * 100);

            if(isToyGraph)
            {
               printf("new: ");
                for (count_type i = 0; i < vertexNum; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
                printf("\nold: ");
                for (count_type i = 0; i < vertexNum; i++) printf("%4zu ", static_cast<uint64_t>(order[i]));
                printf("\n");
            }

            saveCSR_reorderFile(order, additionInfo,
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "..."
            );
            
            std::vector<vertex_id_type>().swap(order);
			Msg_info("MY_BFS_OUT_IN order finish");
        }

        //> MY_IMAX_BFS_IN
        else if(orderMethod == OrderMethod::MY_IMAX_BFS_IN)
        {
            sort_vec.resize(vertexNum);
            sortDegree(SortDegree::INDEGREES);
            std::vector<vertex_id_type> order;
			std::vector<count_type> additionInfo;

            timer t;
            MY_BFS_IN(order, additionInfo); 
            Msg_info("MY_IMAX_BFS_IN time：%f (ms)", t.get_time_ms());
			Msg_info("MY_IMAX_BFS_IN zeroDegreenNum = %u (%.2f%%)", additionInfo[0], ((double)additionInfo[0] / vertexNum) * 100);

            if(isToyGraph)
            {
                printf("new: ");
                for (count_type i = 0; i < vertexNum; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
                printf("\nold: ");
                for (count_type i = 0; i < vertexNum; i++) printf("%4zu ", static_cast<uint64_t>(order[i]));
                printf("\n");
            }

            saveCSR_reorderFile(order, additionInfo,
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "..."
            );
            
            std::vector<vertex_id_type>().swap(order);
			Msg_info("MY_IMAX_BFS_IN order clear");
        }

        else if(orderMethod == OrderMethod::IN_DEGREE_DEC)
        {
            sort_vec.resize(vertexNum);
            sortDegree(SortDegree::INDEGREES);
            std::vector<vertex_id_type> order;
            std::vector<count_type> additionInfo;

            timer t;
            order.resize(vertexNum);
            for (vertex_id_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                order[vertexId] = sort_vec[vertexId].vertex_id;
            }
            Msg_info("IN_DEGREE_DEC time：%f (ms)", t.get_time_ms());

            if(isToyGraph)
            {
                printf("new: ");
                for (count_type i = 0; i < vertexNum; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
                printf("\nold: ");
                for (count_type i = 0; i < vertexNum; i++) printf("%4zu ", static_cast<uint64_t>(order[i]));
                printf("\n");
            }

            saveCSR_reorderFile(order, additionInfo,
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "...",
                BASE_GRAPHFILE_PATH + graph + "..."
            );

            std::vector<vertex_id_type>().swap(order);
			Msg_info("IN_DEGREE_DEC  order clear");
            
        }

        else
		{
			assert_msg(0, "Undefined Vertex Order Method");
		}
    }

    CSR_Result_type get_CSR_reorder()
    {
        return csrResult_reorder;
    }


private:
    
	void sortDegree(SortDegree sortDegree)
	{       

        omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++)
        {
            Vid_degree_type vid_degree;
            vid_degree.vertex_id = vertexId;
            if (sortDegree == SortDegree::OUTDEGREE)
            {
                vid_degree.degree = outDegree[vertexId];
            }
            else if (sortDegree == SortDegree::INDEGREES)
            {
                vid_degree.degree = inDegree[vertexId];
            }
            else if(sortDegree == SortDegree::DEGREE)
            {
                vid_degree.degree = outDegree[vertexId] + inDegree[vertexId];

            }
            else
            {
                assert_msg(false, "");
            }	            
            sort_vec[vertexId] = vid_degree;

        }

		sort(sort_vec.begin(), sort_vec.end(), 
            [&](Vid_degree_type& a, Vid_degree_type& b) -> bool
            {
                if(a.degree > b.degree) return true;
                else if(a.degree == b.degree){
                    if(a.vertex_id < b.vertex_id) return true;
                    else return false;
                }else{
                    return false;
                }
            }
        );
	}


    void MY_BFS_OUT(std::vector<vertex_id_type>& retorder, std::vector<count_type>& additionInfo)
    {
        std::queue<vertex_id_type> que;
		bool* BFSflag = new bool[vertexNum];
		memset(BFSflag, 0, sizeof(bool) * vertexNum);

        std::vector<vertex_id_type> tmp;
		vertex_id_type now;
		std::vector<vertex_id_type> order; 
		std::vector<vertex_id_type> zeroDegree;
        for (count_type k=0; k<vertexNum; k++)
		{
			vertex_id_type temp = sort_vec[k].vertex_id;
			if (BFSflag[temp] == false)
			{
				if (outDegree[temp] == 0)
				{
					zeroDegree.push_back(temp);
					BFSflag[temp] = true;
				}
				else
				{
					que.push(temp);
					BFSflag[temp] = true;
					order.push_back(temp);
				}

				while (que.empty() == false)
				{
					now = que.front();
					que.pop();
					tmp.clear();
					for (vertex_id_type it = csr_offset[now], limit = csr_offset[now+1]; it < limit; it++) {
						vertex_id_type dest = csr_dest[it];
						
						if((outDegree[dest] == 0) && (BFSflag[dest] == false))
						{
							zeroDegree.push_back(dest);
							BFSflag[dest] = true;
						}
						else
						{
							tmp.push_back(dest);
						}		
					}
					sort(tmp.begin(), tmp.end(), 
                        [&](const vertex_id_type& a, const vertex_id_type& b)->bool 
                        {
                            if (outDegree[a] > outDegree[b])
                                return true;
                            else
                                return false;
					    }
                    );

					for (vertex_id_type nbrId = 0; nbrId < tmp.size(); nbrId++)
					{
						if (BFSflag[tmp[nbrId]] == false)
						{
							que.push(tmp[nbrId]);
							BFSflag[tmp[nbrId]] = true;
							order.push_back(tmp[nbrId]);
						}
					}
				}
			}
		}

        delete[] BFSflag;
		assert_msg(((order.size() + zeroDegree.size()) == vertexNum),
			"((order.size() + zeroDegree.size()) != vertexNum), order.size() = %u, zeroDegree.size() = %u", order.size(), zeroDegree.size());
        
        additionInfo.resize(1);
		additionInfo[0] = zeroDegree.size();

        retorder.resize(vertexNum);
        retorder.insert(retorder.begin(), order.begin(), order.end());
        retorder.insert(retorder.begin() + order.size(), zeroDegree.begin(), zeroDegree.end());
    }


    void MY_BFS_IN(std::vector<vertex_id_type>& retorder, std::vector<count_type>& additionInfo)
    {
        std::queue<vertex_id_type> que;
		bool* BFSflag = new bool[vertexNum];
		memset(BFSflag, 0, sizeof(bool) * vertexNum);

        std::vector<vertex_id_type> tmp;
		vertex_id_type now;
		std::vector<vertex_id_type> order; 
		std::vector<vertex_id_type> zeroDegree;
        for (count_type k=0; k<vertexNum; k++)
		{
			vertex_id_type temp = sort_vec[k].vertex_id;
			if (BFSflag[temp] == false)
			{
				if (outDegree[temp] == 0)
				{
					zeroDegree.push_back(temp);
					BFSflag[temp] = true;
				}
				else
				{
					que.push(temp);
					BFSflag[temp] = true;
					order.push_back(temp);
				}

				while (que.empty() == false)
				{
					now = que.front();
					que.pop();
					tmp.clear();
					for (vertex_id_type it = csr_offset[now], limit = csr_offset[now+1]; it < limit; it++) {
						vertex_id_type dest = csr_dest[it];
						
						if((outDegree[dest] == 0) && (BFSflag[dest] == false))
						{
							zeroDegree.push_back(dest);
							BFSflag[dest] = true;
						}
						else
						{
							tmp.push_back(dest);
						}		
					}
					sort(tmp.begin(), tmp.end(), 
                        [&](const vertex_id_type& a, const vertex_id_type& b)->bool 
                        {
                            if (inDegree[a] > inDegree[b])
                                return true;
                            else
                                return false;
					    }
                    );

					for (vertex_id_type nbrId = 0; nbrId < tmp.size(); nbrId++)
					{
						if (BFSflag[tmp[nbrId]] == false)
						{
							que.push(tmp[nbrId]);
							BFSflag[tmp[nbrId]] = true;
							order.push_back(tmp[nbrId]);
						}
					}
				}
			}
		}

        delete[] BFSflag;
		assert_msg(((order.size() + zeroDegree.size()) == vertexNum),
			"((order.size() + zeroDegree.size()) != vertexNum), order.size() = %u, zeroDegree.size() = %u", order.size(), zeroDegree.size());
        
        additionInfo.resize(1);
		additionInfo[0] = zeroDegree.size();

        retorder.resize(vertexNum);
        retorder.insert(retorder.begin(), order.begin(), order.end());
        retorder.insert(retorder.begin() + order.size(), zeroDegree.begin(), zeroDegree.end());
    }



    void MY_BFS_OUT_IN(std::vector<vertex_id_type>& retorder, std::vector<count_type>& additionInfo)
    {
        std::queue<vertex_id_type> que;
		bool* BFSflag = new bool[vertexNum];
		memset(BFSflag, 0, sizeof(bool) * vertexNum);

        std::vector<vertex_id_type> tmp;
		vertex_id_type now;
		std::vector<vertex_id_type> order; 
		std::vector<vertex_id_type> zeroDegree;
        for (count_type k=0; k<vertexNum; k++)
		{
			vertex_id_type temp = sort_vec[k].vertex_id;
			if (BFSflag[temp] == false)
			{
				if (outDegree[temp] == 0)
				{
					zeroDegree.push_back(temp);
					BFSflag[temp] = true;
				}
				else
				{
					que.push(temp);
					BFSflag[temp] = true;
					order.push_back(temp);
				}

				while (que.empty() == false)
				{
					now = que.front();
					que.pop();
					tmp.clear();
					for (vertex_id_type it = csr_offset[now], limit = csr_offset[now+1]; it < limit; it++) {
						vertex_id_type dest = csr_dest[it];
						
						if((outDegree[dest] == 0) && (BFSflag[dest] == false))
						{
							zeroDegree.push_back(dest);
							BFSflag[dest] = true;
						}
						else
						{
							tmp.push_back(dest);
						}		
					}
					sort(tmp.begin(), tmp.end(), 
                        [&](const vertex_id_type& a, const vertex_id_type& b)->bool 
                        {
                            if (inDegree[a] + outDegree[a] > inDegree[b] + outDegree[b])
                                return true;
                            else
                                return false;
					    }
                    );

					for (vertex_id_type nbrId = 0; nbrId < tmp.size(); nbrId++)
					{
						if (BFSflag[tmp[nbrId]] == false)
						{
							que.push(tmp[nbrId]);
							BFSflag[tmp[nbrId]] = true;
							order.push_back(tmp[nbrId]);
						}
					}
				}
			}
		}

        delete[] BFSflag;
		assert_msg(((order.size() + zeroDegree.size()) == vertexNum),
			"((order.size() + zeroDegree.size()) != vertexNum), order.size() = %u, zeroDegree.size() = %u", order.size(), zeroDegree.size());
        
        additionInfo.resize(1);
		additionInfo[0] = zeroDegree.size();

        retorder.resize(vertexNum);
        retorder.insert(retorder.begin(), order.begin(), order.end());
        retorder.insert(retorder.begin() + order.size(), zeroDegree.begin(), zeroDegree.end());
    }


    void saveCSR_reorderFile(const std::vector<vertex_id_type>& order, const std::vector<count_type>& addition_vec,
        std::string csrOffsetFile, std::string csrDestFile, std::string csrWeightFile, std::string old2newFile, std::string addition)
    {
        if(isToyGraph)
        {
            bool exist_csrOffsetFile = (access(csrOffsetFile.c_str(), F_OK) >= 0);
            bool exist_csrDestFile = (access(csrOffsetFile.c_str(), F_OK) >= 0);
            bool exist_csrWeightFile = (access(csrOffsetFile.c_str(), F_OK) >= 0);
            bool exist_old2newFile = (access(old2newFile.c_str(), F_OK) >= 0);
            bool exist_addition = (access(addition.c_str(), F_OK) >= 0);

            if (exist_csrOffsetFile && exist_csrDestFile && exist_csrWeightFile && exist_old2newFile && exist_addition)
            {
                Msg_info("vertexOrder error");
                return;
            }
        }       

        vertex_id_type* old2new = new vertex_id_type[vertexNum];
        omp_parallel_for(vertex_id_type vertexId = 0; vertexId < vertexNum; vertexId++)
        {
            old2new[order[vertexId]] = vertexId;
        }



       
        timer t;
        adjlist_sort.resize(vertexNum);
       
        for (size_t vertexId = 0; vertexId < vertexNum; vertexId++)
        {
          
            vertex_id_type oldId = order[vertexId];
            count_type degree_old = csr_offset[oldId + 1] - csr_offset[oldId];
            adjlist_sort[vertexId].resize(degree_old);
            
        }
        
        // printf("\n");
        // for (size_t i = 0; i < vertexNum; i++)
        // {
        //      printf("adjlist_sort[%u].size() = %u\n", i, adjlist_sort[i].size());    
        // }
        

        
        TaskSteal* taskSteal = new TaskSteal();// compute task
	    size_t totalWorkloads = 0;

	    timer sort_time;
        totalWorkloads = taskSteal->twoStage_taskSteal<size_t, size_t>(static_cast<size_t>(vertexNum),
            [&](size_t& current, size_t& local_workloads)
            {
                size_t end = current + VERTEXWORK_CHUNK;
                size_t length = VERTEXWORK_CHUNK;
                if (end >= vertexNum) length = vertexNum - current;

                
                for (size_t in = 0; in < length; in++)
                {
                    vertex_id_type newId = current + in;
                    vertex_id_type oldId = order[newId];

                    countl_type nbr_first = csr_offset[oldId];
                    countl_type nbr_end = csr_offset[oldId + 1];

                    for (countl_type i = nbr_first; i < nbr_end; i++)
                    {
                        vertex_id_type dest = csr_dest[i];
                        edge_data_type weight = csr_weight[i];
                        
                        adjlist_sort[newId][i - nbr_first] = std::make_pair(old2new[dest], weight);//
                    }
                }
            }, 64
        );
	    Msg_info("nbrSort_taskSteal 用时：%.2lf(ms)", sort_time.get_time_ms());	


        
        t.start();
        //Build csrOffset
        countl_type* csr_offset_ = new countl_type[vertexNum + 1];
        csr_offset_[0] = 0;
        for (count_type i = 1; i <= vertexNum; i++)
        {
            csr_offset_[i] = csr_offset_[i - 1] + static_cast<countl_type>(adjlist_sort[i-1].size());
        }
        assert_msg(csr_offset_[vertexNum] == edgeNum, "csr_offset_[vertexNum] != edgeNum, csr_offset_[vertexNum] = %u", csr_offset_[vertexNum]);

        //Build csrDest and csr_weight
        vertex_id_type* csr_dest_ = new vertex_id_type[edgeNum];
        edge_data_type* csr_weight_ = new edge_data_type[edgeNum];
        omp_parallel_for(count_type i = 0; i < vertexNum; i++)
        {
            neighbor_type nbr = adjlist_sort[i];
            countl_type offset = csr_offset_[i];
            for (countl_type j = 0; j < nbr.size(); j++)
            {
                csr_dest_[offset + j] = nbr[j].first;
                csr_weight_[offset + j] = nbr[j].second;
            }
        }

        
        if(isToyGraph)
        {
            printf("=> csr_offset:\n");
            for (count_type i = 0; i < vertexNum + 1; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
            printf("\n");
            for (count_type i = 0; i < vertexNum + 1; i++) printf("%4zu ", static_cast<uint64_t>(csr_offset_[i]));
            printf("\n");

            printf("=> csr_dest:\n");
            for (countl_type i = 0; i < edgeNum; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
            printf("\n");
            for (countl_type i = 0; i < edgeNum; i++) printf("%4zu ", static_cast<uint64_t>(csr_dest_[i]));
            printf("\n");

            printf("=> csr_weight:\n");
            for (countl_type i = 0; i < edgeNum; i++) printf("[%2zu] ", static_cast<uint64_t>(i));
            printf("\n");
            for (countl_type i = 0; i < edgeNum; i++) printf("%4zu ", static_cast<uint64_t>(csr_weight_[i]));
            printf("\n");
        }

        
        else
        {
            save_binArray<countl_type>(csrOffsetFile, csr_offset_, vertexNum + 1);
		    save_binArray<vertex_id_type>(csrDestFile, csr_dest_, edgeNum);
		    save_binArray<edge_data_type>(csrWeightFile, csr_weight_, edgeNum);

            //old2new
            std::ofstream out_file_old2new(old2newFile.c_str(),
                std::ios_base::out | std::ios_base::binary);
            if (!out_file_old2new.good()) {
                std::cout << "[old2new]Error opening out-file: " << old2newFile << std::endl;
                assert(false);
            }
            t.start();
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                out_file_old2new.write((char*)(&old2new[vertexId]), sizeof(vertex_id_type));

                if ((vertexId != 0) && (vertexId % 100000000 == 0)) Msg_rate("[PROGRESS]: Reorder old2new(order/rank) %.2f%%", (double)vertexId / vertexNum * 100);
            }
            Msg_rate("old2new(order/rank) 100%, : %f (ms)", t.get_time_ms());
            out_file_old2new.close();

            //addition
            std::ofstream out_file_addition(addition.c_str(),
                std::ios_base::out | std::ios_base::binary);
            if (!out_file_addition.good()) {
                std::cout << "[old2new]Error opening out-file: " << addition << std::endl;
                assert(false);
            }
            t.start();
            for (count_type i = 0; i < addition_vec.size(); i++)
            {
                out_file_addition.write((char*)(&addition_vec[i]), sizeof(count_type));
            }
            Msg_rate("addition 100%, : %f (ms)", t.get_time_ms());
            out_file_addition.close();
        }

		Msg_finish("SaveCSRtoFile Finsihed ! ： %.2lf (ms)", t.get_time_ms());

        csrResult_reorder.vertexNum = vertexNum;
        csrResult_reorder.edgeNum = edgeNum;
        csrResult_reorder.csr_offset = csr_offset_;
        csrResult_reorder.csr_dest = csr_dest_;
        csrResult_reorder.csr_weight = csr_weight_;

        // delete[] csr_offset_;
        // delete[] csr_dest_;
        // delete[] csr_weight_;
        
        std::vector<neighbor_type>().swap(adjlist_sort);
    }
};

