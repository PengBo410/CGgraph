#pragma once

#include <stdint.h>


typedef uint32_t count_type;
typedef uint32_t countl_type;     //主要针对edgeNum
typedef uint32_t degree_type;     //主要针对outDegree与inDegree
typedef uint32_t offset_type;     //此处是一个错误,下一个版本用degree_type,因为offset一定与边数为一个类型，除非是被划分了
typedef uint32_t vertex_id_type;
typedef uint32_t edge_data_type;
typedef uint32_t common_type; //For Common Device


typedef uint32_t vertex_data_type;
//typedef float vertex_data_type;

typedef vertex_id_type u1;
typedef int order_type;

typedef uint8_t compress_type; 
typedef uint32_t compress_offset_type;