#pragma once

#include "../Type/data_type.hpp"
#include "../Time/time.hpp"
#include <type_traits> // std::is_same

#include "stdlib.h"
#include <assert.h>
#include <fstream>
#include <string>
#include <sstream> // std::stringstream


template <typename vertex_id_type, typename edge_data_type>
size_t convert2binaryAddEdge(
	std::string inputFile,
	std::string outputFile
) {
	std::ifstream in_file(inputFile.c_str(),
		std::ios_base::in | std::ios_base::binary);
	if (!in_file.good()) {
		std::cout << "[convert2binary]Error opening in-file: " << inputFile << std::endl;
		assert(false);
		return false;
	}

	std::ofstream out_file(outputFile.c_str(),
		std::ios_base::out | std::ios_base::binary);
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
			assert(0);
		}
		vertex_id_type sourceId = static_cast<vertex_id_type>(sourceId_);

		while (1) {

			uint64_t destId_;
			strm >> destId_;
			if (std::numeric_limits<vertex_id_type>::max() <= destId_)
			{
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

			edge_data_type edge_data = (rand() % 99) + 1;
			out_file.write((char*)(&edge_data), sizeof(edge_data_type));

			effectiveLine++;
		}

		linecount++;
		if (linecount % 100000000 == 0) printf("-> (%lu) lines\n", linecount);
	}

	in_file.close();
	out_file.close();


	return (maxVid + 1);
}