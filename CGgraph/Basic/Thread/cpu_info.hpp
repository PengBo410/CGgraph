#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <assert.h>
#include <numa.h>




class CPUInfo {

private:

	int socketNum = 0;
	int coreNum = 0;
	int threadNum = 0;
	
	std::vector<std::pair<int, int>> coreId_threadId;
	std::vector <int> socketId_vec;
	std::vector <int> threadId_vec;
	std::vector <int> coreId_vec;

	int max_socketId = -1;
	int max_coreId  = -1;
	int max_threadId = -1;

	struct Util{
		int socketId_;
		int threadId_;
		int coreId_;
	};
	std::vector < Util > util_vec;

	//std::vector < std::vector < std::vector <int> > > socketId_coreId_threadId;
public:
	CPUInfo()
	{
		readCpuInfo();
	}

private:
	void readCpuInfo(std::string inputFile = "/proc/cpuinfo")
	{
		std::vector<std::vector<int>> coreId_threadId; 

		std::vector<std::vector<std::string>> file2vector;




		std::ifstream in_file(inputFile.c_str(), std::ios_base::in);
		if (!in_file.good()) {
			std::cout << "[readCpuInfo]Error opening in-file: " << inputFile << std::endl;
			assert(false);
		}

		
		std::string lineString;
		std::vector<std::string> temp;
		while (in_file.good() && !in_file.eof()) {

			std::getline(in_file, lineString);

			
			if (!lineString.empty()) 
			{
				temp.push_back(lineString);
			}
			else
			{
				if (temp.size() != 0) 
				{
					file2vector.push_back(temp);
					temp.clear();
				}

			}
		}

		in_file.close();

		
		for (size_t blockId = 0; blockId < file2vector.size(); blockId++)
		{
			std::vector<std::string> block = file2vector[blockId];
			for (size_t lineId = 0; lineId < block.size(); lineId++)
			{
				std::string line = block[lineId];

				int socketId = parseSocketId(line);
				if(socketId != -1) socketId_vec.push_back(socketId);
				if (socketId > max_socketId) max_socketId = socketId;

				int coreId = parsePhysicalCore(line);
				if (coreId != -1) coreId_vec.push_back(coreId);
				if (coreId > max_coreId) max_coreId = coreId;

				int threadId = parseLogicalCore(line);
				if (threadId != -1) threadId_vec.push_back(threadId);
				if (threadId > max_threadId) max_threadId = threadId;
			}
		}

		
		
		assert(socketId_vec.size() == coreId_vec.size());
		assert(socketId_vec.size() == threadId_vec.size());

		for (size_t i = 0; i < socketId_vec.size(); i++)
		{
			Util util;
			util.socketId_ = socketId_vec[i];
			util.coreId_ = coreId_vec[i];
			util.threadId_ = threadId_vec[i];

			util_vec.push_back(util);
		}

		
		sort(util_vec.begin(), util_vec.end(),
			[](const Util& a, const Util& b)
			{
				if (a.socketId_ < b.socketId_)
				{
					return true;
				}
				else if (a.socketId_ == b.socketId_)
				{
					if (a.coreId_ < b.coreId_)
					{
						return true;
					}
					else if (a.coreId_ == b.coreId_)
					{
						if (a.threadId_ <= b.threadId_)
						{
							return true;
						}
						else
						{
							return false;
						}
					}
					else
					{
						return false;
					}
				}
				else
				{
					return false;
				}
			});


		
		std::vector <int> socketId_vec_temp;
		std::vector <int> coreId_vec_temp;
		std::vector <int> threadId_vec_temp;
		
		for (size_t i = 0; i < util_vec.size(); i++)
		{
			Util util = util_vec[i];
			socketId_vec_temp.push_back(util.socketId_);
			coreId_vec_temp.push_back(util.coreId_);
			threadId_vec_temp.push_back(util.threadId_);
		}

		socketId_vec_temp.erase(unique(socketId_vec_temp.begin(), socketId_vec_temp.end()), socketId_vec_temp.end());
		coreId_vec_temp.erase(unique(coreId_vec_temp.begin(), coreId_vec_temp.end()), coreId_vec_temp.end());
		threadId_vec_temp.erase(unique(threadId_vec_temp.begin(), threadId_vec_temp.end()), threadId_vec_temp.end());

		socketNum = socketId_vec_temp.size();
		coreNum = coreId_vec_temp.size();
		threadNum = threadId_vec_temp.size();
		//printf("threadNum = %d\n", threadNum);



		//socketId_coreId_threadId vector 的coreId长度
		//socketId_coreId_threadId.resize(max_socketId + 1);
		//int p = util_vec[0].socketId_;
		//int maxCoreId_ofEachSocket = 1;
		//for (size_t i = 1; i < util_vec.size(); i++)
		//{
		//	Util util = util_vec[i];
		//	if (util.socketId_ == p)
		//	{
		//		maxCoreId_ofEachSocket++;
		//	}
		//	else
		//	{
		//		socketId_coreId_threadId[p].resize(maxCoreId_ofEachSocket);
		//		p = util.socketId_;
		//		maxCoreId_ofEachSocket = 1;
		//	}
		//}
		//socketId_coreId_threadId[p].resize(maxCoreId_ofEachSocket);

		////构建socketId_coreId_threadId
		//for (size_t i = 0; i < util_vec.size(); i++)
		//{
		//	Util util = util_vec[i];
		//	socketId_coreId_threadId[util.socketId_][util.coreId_].emplace_back(util.threadId_);
		//}

		////打印
		//for (size_t socketId = 0; socketId < socketId_coreId_threadId.size(); socketId++)
		//{
		//	printf("socket{%d}\n", socketId);
		//	auto core_vec = socketId_coreId_threadId[socketId];
		//	for (size_t coreId = 0; coreId < core_vec.size(); coreId++)
		//	{
		//		printf("\tcore[%d]\n", coreId);
		//		auto thread_vec = socketId_coreId_threadId[socketId][coreId];
		//		for (size_t threadId = 0; threadId < thread_vec.size(); threadId++)
		//		{
		//			printf("\t\tthread(%d)\n", thread_vec[threadId]);
		//		}

		//	}
		//}

	}




	int parseLogicalCore(std::string line)
	{
		int pro = -1;
		//寻找logical core，也就是threadNum
		if (line.find("processor", 0) != std::string::npos)
		{
			//寻找 ": "
			int index = line.find_last_of(": ");
			if (index != -1)
			{
				std::string value = line.substr(index);   
				pro = atoi(value.c_str());
			}
		}
		return pro;
	}

	int parsePhysicalCore(std::string line)
	{
		int core = -1;
		
		if (line.find("core id", 0) != std::string::npos)
		{
			//寻找 ": "
			int index = line.find_last_of(": ");
			if (index != -1)
			{
				std::string value = line.substr(index);  
				core = atoi(value.c_str());
			}
		}
		return core;
	}

	int parseSocketId(std::string line)
	{
		int socketId = -1;
		
		if (line.find("physical id", 0) != std::string::npos)
		{
			
			int index = line.find_last_of(": ");
			if (index != -1)
			{
				std::string value = line.substr(index);   
				socketId = atoi(value.c_str());
			}
		}
		return socketId;
	}

public:

	int getSocketNum()
	{
		return socketNum;
	}

	int getCoreNum()
	{
		return coreNum;
	}

	int getThreadNum()
	{
		return threadNum;
	}

	bool isOpenHybirdThread()
	{
		return (coreNum == threadNum) ? false : true;
	}

	void print()
	{
		printf("Socket num = {%d}, Core num = [%d], Thread num = (%d), isOpenHybirdThread = *%d*\n",
			socketNum, coreNum, threadNum, isOpenHybirdThread()
			);
		if (isOpenHybirdThread()) 
		{
			printf("***************************************************************\n");
			printf("*                        ---------------                      *\n");
			printf("*                        |             |                      *\n");
			printf("*                        |Hybird-Thread|                      *\n");
			printf("*                        |             |                      *\n");
			printf("*                        ---------------                      *\n");
			printf("***************************************************************\n");
		}
	}

	void print_socket_core_thread()
	{
		for (size_t i = 0; i < util_vec.size(); i++)
		{
			Util util = util_vec[i];
			printf("%zu: socket{%d} - core[%d] - threadId(%d)\n", i, util.socketId_, util.coreId_, util.threadId_);
		}
	

};
