#pragma once

#include "../Type/data_type.hpp"
#include "../Time/time.hpp"
#include <type_traits> // std::is_same

#include "stdlib.h"
#include <assert.h>
#include <fstream>
#include <string>
#include <sstream> // std::stringstream

/*****************************************************************************
 * 解析的格式为：
 * 101     102
 * 101     104
 * update on 2022-10-30 添加了数据类型容量检测, 适配大图如clueweb12
 * 修改：将添加边的范围从[1,20)变成[1,100))
 * TODO:考虑并行版本，读取不同的行，写到多个中间文件中(使用的线程数个)，然后合并
 *      添加重复边检测
 *****************************************************************************/
template <typename vertex_id_type, typename edge_data_type>
size_t convert2binaryAddEdge(
	std::string inputFile,
	std::string outputFile
) {
	std::ifstream in_file(inputFile.c_str(),
		std::ios_base::in | std::ios_base::binary);//以二进制读的方式打开,并写入内存
	if (!in_file.good()) {
		std::cout << "[convert2binary]Error opening in-file: " << inputFile << std::endl;
		assert(false);
		return false;
	}

	std::ofstream out_file(outputFile.c_str(),
		std::ios_base::out | std::ios_base::binary);//以二进制读的方式打开,并写入磁盘
	if (!out_file.good()) {
		std::cout << "[convert2binary]Error opening out-file: " << outputFile << std::endl;
		assert(false);
		return false;
	}

	size_t maxVid = 0;
	size_t linecount = 0;
	size_t effectiveLine = 0;
	timer ti; ti.start();

	while (in_file.good() && !in_file.eof()) {
		std::string line;
		std::getline(in_file, line);
		if (line.empty()) continue;
		if (in_file.fail()) break;

		//行解析
		std::stringstream strm(line);

		uint64_t sourceId_;
		strm >> sourceId_;
		if (std::numeric_limits<vertex_id_type>::max() <= sourceId_)
		{
			printf("定义的vertex_id_type不足以表示sourceId = %lu (%lu 行)\n", sourceId_, linecount);
			assert(0);
		}
		vertex_id_type sourceId = static_cast<vertex_id_type>(sourceId_);

		while (1) {

			uint64_t destId_;
			strm >> destId_;
			if (std::numeric_limits<vertex_id_type>::max() <= destId_)
			{
				printf("定义的vertex_id_type不足以表示destId_ = %lu (%lu 行)\n", destId_, linecount);
				assert(0);
			}
			vertex_id_type destId = static_cast<vertex_id_type>(destId_);


			if (sourceId == destId)
			{
				break;
			}

			if (strm.fail())
				break;


			if (sourceId < 0 || destId < 0)
			{
				break;
			}
			size_t max_temp = ((sourceId > destId) ? sourceId : destId);
			maxVid = ((maxVid > max_temp) ? maxVid : max_temp);

			out_file.write((char*)(&sourceId), sizeof(vertex_id_type));
			out_file.write((char*)(&destId), sizeof(vertex_id_type));

			//添加边(刚开始我们定义的范围是[1,20), 现在变成了[1,100))
			edge_data_type edge_data = (rand() % 99) + 1;//[1,100) //要取得[a,b)的随机整数，使用(rand() % (b-a))+ a;
			out_file.write((char*)(&edge_data), sizeof(edge_data_type));

			effectiveLine++;
		}

		linecount++;
		if (linecount % 100000000 == 0) printf("已处理完(%lu)行\n", linecount);//每1亿行打印一次
	}

	in_file.close();
	out_file.close();

	std::cout << "共有【" << linecount << "】行从文件 <" << inputFile << "> 以二进制形式写入到 <" << outputFile << "> 文件中！\n"
		<< "其中有效的行数(总边数)为：【" << effectiveLine << "】行," << "总顶点数为：[" << (maxVid + 1) << "]\n"
		<< "用时：" << ti.current_time() << "(秒)。" << std::endl;

	return (maxVid + 1);
}