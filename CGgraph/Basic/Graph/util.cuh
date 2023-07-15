#pragma once

#include "basic_struct.hpp"
#include "../Other/IO.hpp"

#include <numaif.h>// move_page

std::string getAlgName(Algorithm_type alg)
{
	std::string algorithm = "";

	if (alg == Algorithm_type::BFS) return algorithm = "BFS";
	else if (alg == Algorithm_type::SSSP) return algorithm = "SSSP";
	else if (alg == Algorithm_type::CC) return algorithm = "CC";
	else if (alg == Algorithm_type::PR) return algorithm = "PageRank";
	else
	{
		assert_msg(false, "getAlgName ERROR");
		return "";
	}
}


std::string getResultFilePath(CheckInfo_type& checkInfo)
{
	std::string rightFile = "";
	if (checkInfo.algorithm == Algorithm_type::BFS)
	{
		rightFile = "/home/pengjie/graph_data/checkResult/BFS/" + checkInfo.graphName + "_" + getAlgName(checkInfo.algorithm) + "_" + std::to_string(checkInfo.root) + ".bin";

		bool isExist = (access(rightFile.c_str(), F_OK) >= 0);
		if (!isExist)
		{
			assert_msg(false, "getResultFilePath时, 未发现对应的[%s]文件", rightFile.c_str());
		}

		return rightFile;
	}
	else if (checkInfo.algorithm == Algorithm_type::SSSP)
	{
		rightFile = "/home/pengjie/graph_data/checkResult/SSSP/" + checkInfo.graphName + "_" + getAlgName(checkInfo.algorithm) + "_" + std::to_string(checkInfo.root) + ".bin";

		bool isExist = (access(rightFile.c_str(), F_OK) >= 0);
		if (!isExist)
		{
			assert_msg(false, "getResultFilePath时, 未发现对应的[%s]文件", rightFile.c_str());
		}

		return rightFile;
	}
	else
	{
		assert_msg(false, "getResultFilePath时, 未知算法");
		return "";
	}
}


template<typename T>
void checkBinResult(std::string resultFilePath, T* current_array, size_t length, OrderMethod orderMethod, T* old2new)
{
	T* resultArray = load_binFile<T>(resultFilePath, length);

	if (orderMethod == OrderMethod::NATIVE)
	{
		omp_parallel_for(count_type i = 0; i < length; i++)
		{
			if constexpr (std::is_same_v<T, uint32_t>)
			{
				assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %u, resultArray[%u] = %u",
					i, current_array[i], i, resultArray[i]);
			}
			else if constexpr (std::is_same_v<T, float>)
			{
				assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %f, resultArray[%u] = %f",
					i, current_array[i], i, resultArray[i]);
			}
			else if constexpr (std::is_same_v<T, double>)
			{
				assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %lf, resultArray[%u] = %lf",
					i, current_array[i], i, resultArray[i]);
			}
			else
			{
				assert_msg(false, "Function [checkBinResult] Meet Unknown Data_type");
			}
		}
	}
	else
	{
		omp_parallel_for(count_type i = 0; i < length; i++)
		{
			if constexpr (std::is_same_v<T, uint32_t>)
			{
				//if(old2new[i] < 4198894)
				assert_msg((current_array[old2new[i]] == resultArray[i]), "检查结果出错:current_array[%u] = %u, resultArray[%u] = %u",
					old2new[i], current_array[old2new[i]], i, resultArray[i]);
			}
			else if constexpr (std::is_same_v<T, float>)
			{
				assert_msg((current_array[old2new[i]] == resultArray[i]), "检查结果出错:current_array[%u] = %f, resultArray[%u] = %f",
					old2new[i], current_array[old2new[i]], i, resultArray[i]);
			}
			else if constexpr (std::is_same_v<T, double>)
			{
				assert_msg((current_array[old2new[i]] == resultArray[i]), "检查结果出错:current_array[%u] = %lf, resultArray[%u] = %lf",
					old2new[i], current_array[old2new[i]], i, resultArray[i]);
			}
			else
			{
				assert_msg(false, "Function [checkBinResult] Meet Unknown Data_type");
			}
			
		}
	}	

	Msg_finish("All The Algorithm Result Finished The Check !");
}



int getAdrNumaNode(void* ptr)
{
	int status;
	if ((move_pages(0, 1, &ptr, NULL, &status, 0)) == -1) {
		perror("move_pages");
		return -1;
	}
	return status;
}