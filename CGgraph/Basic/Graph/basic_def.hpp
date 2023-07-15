#pragma once

#define WORDSIZE 64
#define RATE_UTIL 100000000     //Displays the percentage to print
#define PAGESIZE 4096           //PageSize 4096


#define BLOCKSIZE 256
#define WARPSIZE 32
#define HALFWARP 16
const constexpr uint32_t WARP_NUM_PER_BLOCK = BLOCKSIZE / WARPSIZE;  //编译时计算
#define DONATE_POOL_SIZE 32 //每个block拥有的donate pool 的大小, 以顶点个数为单位

#define SUB_VERTEXSET 64            //线程的划分chunk
#define VERTEXSTEAL_CHUNK 64        //窃取线程调度顶点的chunk
#define EDGESTEAL_THRESHOLD 12800 //启用edgeSteal的阈值 12800
#define EDGESTEAL_CHUNK  6400  //线程调度边的chunk 6400
#define VertexValue_MAX 999999999   // 这里不能用MAX命名, 会导致CUB出错

#define GB(size) size /(1024*1024*1024)  // bytes to GB

/* **********************************************************************
 *                              【ALIGN】
 * **********************************************************************/
#define CACHE_ALIGNED __attribute__((aligned(64)))


/* **********************************************************************
 *                              【WORD】
 * **********************************************************************/
#define WORD_OFFSET(word) ((word) >> 6)    //word / 64
#define WORD_MOD(word)    ((word) & 0x3f)  //word % 64

//不检查类型
#define cpj_max(a,b) ((a>b)?a:b) 
#define cpj_min(a,b) ((a>b)?b:a) 

#define NUMA_AWARE