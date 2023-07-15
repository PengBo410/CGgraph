#pragma once

#include <thread>

#include "Basic/basic_include.cuh"
#include "taskSteal.hpp"
#include "Two-level_partition.cuh"
#include "host_algorithm.hpp"
#include "device_algorithm.cuh"
#include <mutex>
#include <execution>


#define CGGRAPHENGINE_DEBUG
//#define CGGRAPHENGINE_CHECK
#define CGGRAPHENGINE_DETAIL_DEBUG

#define PIN_MEM // vertexValue

#define GPUCORE_TIME 4 //Hidden The Ability

#define HEAVY_ACTIVE_VERTICES 16384
#define HEAVY_ACTIVE_EDGES 25000000

#define CPU_STEAL_GRANULARITY 41000  // 单位是JOB，默认是 20，当前设置下为400个segment
#define GPU_WORK_GRANULARITY 128


class CGgraphEngine{

public:
	//> Partition Result
    typedef TwoLevelPartition::Partition_type Partition_type;
	typedef TwoLevelPartition::Segment_type Segment_type;
	Partition_type partitionResult;

	//> Graph Info
    count_type vertexNum;
	countl_type edgeNum;
	count_type zeroOutDegreeNum;
	count_type noZeroOutDegreeNum;

    countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;
	offset_type* outDegree;

    //> NUMA Info and NUMA Array
	count_type socketNum;
	count_type threadPerSocket;
	count_type threadNum;
	count_type vertexPerSocket; // 按PAGESIZE对齐

	count_type* numa_offset;// 长度为：socketNum + 1  -> 将本机的顶点进行了划分
	countl_type** csr_offset_numa;// [socketId][offset]
	vertex_id_type** csr_dest_numa;// [socketId][dest]
	edge_data_type** csr_weight_numa;// [socketId][weight]	
	count_type* vertexNum_numa;//每个numa的顶点数, 长度为：socketNum, 应该按照pageSize对齐
	countl_type* edgeNum_numa;//每个numa的边数, 长度为：socketNum
	count_type* zeroOffset_numa;//归零的偏移量, 长度为：socketNum
	countl_type* zeroDegree_numa;

    //> TaskSteal
	TaskSteal* taskSteal;          // graph process used by host
	TaskSteal* taskSteal_align64;  // clear active
	TaskSteal* taskSteal_hostHelp; // set active used by host
	TaskSteal* taskSteal_totalTask;// compute task

    //> VertexValue
	vertex_data_type* vertexValue;
    std::vector<vertex_data_type*> vertexValue_temp_host;
	std::vector<vertex_data_type*> vertexValue_temp_device;
	//vertex_data_type* vertexValue_temp;// Offload的中间介质

    //> Active
	dense_bitset active_in;  // Fixed_Bitset | dense_bitset
	dense_bitset active_out; // Fixed_Bitset | dense_bitset
	dense_bitset active_steal;
	DoubleBuffer<dense_bitset> active; //active_in与active_out最终封装到DoubleBuffer
	count_type activeNum_device = 0;  // Device 端激活的顶点数
	count_type activeNum_host = 0;    // Host   端激活的顶点数
	count_type activeNum = 0;         // Device + Host 端总共激活的顶点数
	countl_type activeEdgeNum = 0;
	size_t firstWordIndex = 0;
	size_t lastWordIndex = 0;
	uint64_t* eachGPU_activeNum;

    //> Algorithm
	Algorithm_type algorithm;
	count_type ite = 0; // 迭代次数
	bool is_outDegreeThread = false;

    //> Device
    int useDeviceNum;
    int useDeviceId = 0;
	uint64_t nBlock;     //要使用的Device开启的blockNum

	std::mutex wakeupDevice_mutex; // 用于唤醒Device
	std::mutex queue_mutex;
	std::vector<size_t> wakeupDevice;
	bool hybirdComplete = false;     // Hybird迭代是否完成
	bool usedDevice = false;         // 标志是否使用过Device
	bool usedDevice_ite = false;     // 标记本次迭代是否使用过Device
	dense_bitset stealMask;        // size is useDeviceNum, 0 mark cannot steal, 1 mark can steal 
	size_t noWorkloadSeg = 0;        // The number of no workload segment
    
	count_type common_size = 9; // [0]: worklist_count; [1]: worklist_size; [2]: vertexNum; [3]: edgeNum;  [4]: noZeroOutDegreeNum; 
                                // [5]: SEGMENT_SIZE:   [6]: JOB_SIZE       [7] segment_start; [8] GPU_MODEL_START
	//std::vector<common_type*> common_host;
	std::vector<common_type*> common_device;

	std::vector<cudaStream_t > stream_device_vec;
    

    //> Job
    std::vector<uint64_t> jobSize_vec; //Util is vertices

	//> Workload
	struct Workload_type{
		uint32_t seg_id;
		count_type seg_activeVertexNum;
	};

	uint32_t workloadQueueHead_host = 0;
	uint32_t workloadQueueTail_host = 0;
	bool hasOwnWorkload_CPU = false; // 代替workloadQueueTail_host, 表示CPU在本次迭代中是否具有workload
	std::vector<uint32_t> workloadQueue_host_vec;

	// 在设计从大到小减的运算中, 类型要是用无符号的, 切不要是用有符号和无符号的数据进行比较大小
	std::vector<int64_t> workloadQueueHead_device_vec; 
	std::vector<int64_t> workloadQueueTail_device_vec;  //Segment Num, 要减去，所以要为有符号
	std::vector<uint32_t> workloadQueueConstTail_device_vec;  //Segment Num
	std::vector<std::vector<Workload_type>> workloadQueue_device_vec2;  // 粒度是segment

	std::vector<uint64_t*> bitmap_host_vec; // Bitmap For GPU To Prevent Its Work-Efficient
	std::vector<uint64_t*> bitmap_device_vec; // Bitmap For GPU To Prevent Its Work-Efficient

	std::vector<uint32_t*> trasfer_segment_host_vec;
	std::vector<uint32_t*> trasfer_segment_device_vec;

	std::vector<uint64_t> processSegmentCount_vec;//统计每次迭代中所有处理器处理完成的segment数(GPU-0,GPU-1,CPU),用于验证
	std::vector<uint64_t> CPUSegmentCount_vec;
	std::vector<timeval> processorEndTime_vec;
	timer processorEnd_time;
	
	
	


    
public:

    /* *********************************************************************************
     * Func: [Constructor]
     * *********************************************************************************/
    CGgraphEngine(const CSR_Result_type& csrResult, const Partition_type& partitionResult_,
        Algorithm_type algorithm_, int useDeviceNum_
    ):
        vertexNum(0),
        edgeNum(0),
        noZeroOutDegreeNum(0),
        partitionResult(partitionResult_),
        useDeviceNum(0)
    {
        assert_msg((ThreadNum >= useDeviceNum_ + 2), "The machine requires at least (useDeviceNum_ + 2) threads, current threadNum = %zu", static_cast<uint64_t>(ThreadNum));

        //Get Graph CSR
		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
        noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;
		zeroOutDegreeNum = vertexNum - noZeroOutDegreeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;
        Msg_info("zeroOutDegreeNum = %u, noZeroOutDegreeNum = %u", (vertexNum - noZeroOutDegreeNum), noZeroOutDegreeNum);

        // algorithm
        algorithm = algorithm_;
        Msg_info("The Algorithm Name: (%s)",getAlgName(algorithm).c_str());

        //Device
        useDeviceNum = useDeviceNum_;
        GPUInfo* gpuInfo = new GPUInfo(); 
		int _maxDeviceNum = gpuInfo->getDeviceNum();
		assert_msg((useDeviceNum <= _maxDeviceNum),
		 "(useDeviceNum > _maxDeviceNum), useDeviceNum = %d, _maxDeviceNum = %d", useDeviceNum, _maxDeviceNum);
		stealMask.resize(useDeviceNum);

         //Job Size
         jobSize_vec.resize(useDeviceNum);
         for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
         {
            int GPU_cores = gpuInfo->getCoreNum()[deviceId];
            int min_workload_vertices = GPU_cores * GPUCORE_TIME;
            uint64_t jobSize = min_workload_vertices - (min_workload_vertices % SEGMENT_SIZE);
            jobSize_vec[deviceId] = jobSize;
            Msg_info("The [%d] GPU Has Cores: (%zu), Set The JobSize: (%zu) Vertices, Include (%zu) Segments", deviceId, 
                static_cast<uint64_t>(GPU_cores), static_cast<uint64_t>(jobSize), static_cast<uint64_t>(jobSize / SEGMENT_SIZE));
         }

		 //Workload
		 workloadQueue_host_vec.resize(partitionResult.segmentNum);
		 workloadQueueHead_device_vec.resize(useDeviceNum);
		 workloadQueueTail_device_vec.resize(useDeviceNum);
		 workloadQueueConstTail_device_vec.resize(useDeviceNum);
		 workloadQueue_device_vec2.resize(useDeviceNum);	
		 trasfer_segment_device_vec.resize(useDeviceNum);
		 trasfer_segment_host_vec.resize(useDeviceNum);	
		 wakeupDevice.resize(useDeviceNum);
		 processSegmentCount_vec.resize((useDeviceNum + 1), 0);
		 CPUSegmentCount_vec.resize(ThreadNum);
		 processorEndTime_vec.resize((useDeviceNum + 1));
		 for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		 {
			workloadQueueHead_device_vec[deviceId] = 0;
			workloadQueueTail_device_vec[deviceId] = 0;
			workloadQueueConstTail_device_vec[deviceId] = 0;
			workloadQueue_device_vec2[deviceId].resize(partitionResult.segmentNum);	

			wakeupDevice[deviceId] = 0;
		 }
         
        //> Host, Init Graph Numa
        timer constructTime;
		initGraphNuma();
		Msg_info("Init Graph Numa, Used time: %.2lf (ms)", constructTime.get_time_ms());

        //> Host, Partition The Host Graph According To The NUMA
        constructTime.start();
        partitionGraphByNuma();
		Msg_info("NUMA-Aware Host Graph: Used time: %.2lf (ms)", constructTime.get_time_ms());

        //> Host - Allocate vertexValue and active
		constructTime.start();
		allocate_vertexValueAndActive();
		Msg_info("Allocate-Active: Used time: %.2lf (ms)", constructTime.get_time_ms());

        //> Host - Build outDegree
		constructTime.start();
		get_outDegree();
		Msg_info("Build-outDegree: Used time: %.2lf (ms)", constructTime.get_time_ms());

        //> Device - Malloc The Memory For Device
		constructTime.start();
		malloc_device();
		Msg_info("Malloc The Memory For Device: Used time: %.2lf (ms)", constructTime.get_time_ms());

		//> Device - Worklaod_bitmap
		constructTime.start();
		build_bitmapForDevice();
		Msg_info("Build The Bitmap (Active) For Device: Used time: %.2lf (ms)", constructTime.get_time_ms());

		//CG_co_execution();

    }// end of function [Constructor: CGgraphEngine]





public:
	/* ==============================================================================================================================*
	 *                                                                                                                                      *
	 *                                       [CPU/GPU Cooperation Execution Model]                                                   *
	 *                                                                                                                                      *
	 * ==============================================================================================================================*/
	double CG_co_execution(vertex_id_type root = 0)
	{
		if (root >= noZeroOutDegreeNum) { 
			Msg_info("Current root = (%zu), outDegree is 0, exit directly !", static_cast<uint64_t>(root)); 
			return 0.0;
	    }

		// Init Variable and Algorithm
		resetVariable();
		initAlgorithm(root);
		// Register The Agent
		std::vector<std::thread> agent_vec;
		agent_vec.resize(useDeviceNum);
		for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++){
			CUDA_CHECK(cudaSetDevice(deviceId));
			agent_vec[deviceId] = std::thread(&CGgraphEngine::GPU_Execution_Model, this, deviceId);//开启控制Device的线程
		}

		// CPU/GPU Co-execution
		double duration = 0.0;
		timer iteTime_total;
		timer iteTime_single;
		timer tempTime;
		timer heavyTemp_time; //测量Heavy内部的时间

		// Iterative fashion
		iteTime_total.start();
		do{
			iteTime_single.start();

			ite++;

			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				tempTime.start();
			#endif
			clearActiveOut();
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				logstream(LOG_INFO) << "\t\t1. [Clear Active Out], Used time :" 
					<< std::setw(7) << std::setprecision(2) << std::fixed
					<< tempTime.get_time_ms() << " (ms)" << std::endl;
			#endif


			//> Heavy or Light
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			tempTime.start();
			#endif
			Workload_heterogeneous workloadHet = Workload_heterogeneous::LIGHT;
			// When there are only one vertex, it considered as [Light Workload]
			if(activeNum == 1)//1
			{
				workloadHet = Workload_heterogeneous::LIGHT;
			}else if((activeNum == noZeroOutDegreeNum) && (edgeNum >= HEAVY_ACTIVE_EDGES))
			{
				workloadHet = Workload_heterogeneous::HEAVY;
				activeEdgeNum = edgeNum;
			}else if(activeNum >= HEAVY_ACTIVE_VERTICES){ //尽量使用activeNum多做点事

				//1. Calcuate The workloadHet
				firstWordIndex = 0; lastWordIndex = 0; activeEdgeNum = 0;
				hasOwnWorkload_CPU = false;
				for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
				{
					processSegmentCount_vec[deviceId] = 0;
					workloadQueueHead_device_vec[deviceId] = 0;
					workloadQueueTail_device_vec[deviceId] = 0;
				}
				processSegmentCount_vec[useDeviceNum] = 0;
			
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				workloadQueueTail_host = 0;
				for(int tid=0; tid<ThreadNum; tid++) CPUSegmentCount_vec[tid] = 0;
				#endif
				

				firstWordIndex = findFirstActiveWord();
				lastWordIndex = findLastActiveWord();

				activeEdgeNum = getActiveEdgeNum(firstWordIndex, lastWordIndex + 1);

				if(activeEdgeNum >= HEAVY_ACTIVE_EDGES){
					workloadHet = Workload_heterogeneous::HEAVY;
				}
			}
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				logstream(LOG_INFO) << "\t\t2. [Heavy  or  Light], Used time :" 
					<< std::setw(7) << std::setprecision(2) << std::fixed
					<< tempTime.get_time_ms() << " (ms)" 
					<< ", workloadHet (0-Light, 1-Heavy): " << workloadHet
					<< ", firstWordIndex = " << firstWordIndex 
					<< ", activeEdgeNum = " << activeEdgeNum
					<< std::endl;
			#endif
		

			//? Simulate The Heavy
			//workloadHet = Workload_heterogeneous::LIGHT;

			//> workloadHet : [Heavy]
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			tempTime.start();
			#endif
			if(workloadHet == Workload_heterogeneous::HEAVY)
			{
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				Msg_major("-------------- The [Heavy] Workload -------------- ");
				#endif
				usedDevice_ite = true;

				//> [Heavy Workload]: 1. Partition The Workload
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif
				omp_parallel_for (uint32_t segmentId = 0; segmentId < partitionResult.segmentNum; segmentId++)
				{
					int owner = partitionResult.seg_vec[segmentId].seg_owner;
					vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
					vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE;
					if (endVertexId >= noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;
					assert_msg((WORD_MOD(startVertexId) == 0), "startVertexId %% 64 is not 0");

					uint64_t startWordIndex = WORD_OFFSET(startVertexId);
					uint64_t endWordIndex = (endVertexId + WORDSIZE  - 1) / WORDSIZE; // WORD_OFFSET(endVertexId);
					
					if(owner != -1)
					{
						count_type seg_activeVertexNum = 0;
						for (uint64_t wordIndex = startWordIndex; wordIndex < endWordIndex; wordIndex++)
						{
							if(active.in().array[wordIndex] != 0)
							{
								seg_activeVertexNum += __builtin_popcountl(active.in().array[wordIndex]);
							}
						}
						
						if(seg_activeVertexNum != 0){
							Workload_type workload;
							workload.seg_id = segmentId;
							workload.seg_activeVertexNum = seg_activeVertexNum;

							int owner = partitionResult.seg_vec[segmentId].seg_owner;
							assert_msg((owner < useDeviceNum), "owner >= useDeviceNum");

							uint64_t queueIndex = __sync_fetch_and_add(&workloadQueueTail_device_vec[owner], 1);
							workloadQueue_device_vec2[owner][queueIndex] = workload;
						}
						#ifdef CGGRAPHENGINE_CHECK
						else
						{
							__sync_fetch_and_add(&noWorkloadSeg, 1);
						}
						#endif
					}
					else
					{	
						hasOwnWorkload_CPU = true;
						#ifdef CGGRAPHENGINE_DETAIL_DEBUG
						__sync_fetch_and_add(&workloadQueueTail_host, 1);
						#endif
						//uint64_t queueIndex = __sync_fetch_and_add(&workloadQueueTail_host, 1);
						//workloadQueue_host_vec[queueIndex] = segmentId;
						// Clear The active.in()
						//memset(active.in().array + startWordIndex, 0, sizeof(size_t) * (endWordIndex - startWordIndex));					
					}

				}// end of [Partition The Workload]

				for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
				{
					workloadQueueConstTail_device_vec[deviceId] = workloadQueueTail_device_vec[deviceId];					
				}
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.1 [H: Par. Workload], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
				#endif
				

				// Check
				#ifdef CGGRAPHENGINE_CHECK
				//count_type partitionWorkload_vertexNum_host = workloadQueueTail_host;//parallel_popcount(active.in().array);
				Msg_info("Partition The Workload, Total Active VertexNum: (%zu)", static_cast<uint64_t>(activeNum));
				Msg_info("Partition The Workload, CPU Has Active Vertex: (%d)", hasOwnWorkload_CPU);
				Msg_info("Partition The Workload, No Active Segments: (%zu)", static_cast<uint64_t>(noWorkloadSeg));
				uint64_t partitionWorkload_vertexNum_device = 0;
				eachGPU_activeNum = new uint64_t[useDeviceNum+1];
				memset(eachGPU_activeNum, 0, useDeviceNum * sizeof(uint64_t));
				for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
				{
					assert_msg(workloadQueueTail_device_vec[deviceId] <= partitionResult.segmentNum,
						"The [%d] GPU Has  Active Segment (%zu), Which Large Than Total Segment (%zu)",
						deviceId, static_cast<uint64_t>(workloadQueueTail_device_vec[deviceId]),
						static_cast<uint64_t>(partitionResult.segmentNum)
					);

					#pragma omp parallel for reduction(+: partitionWorkload_vertexNum_device)
					for (uint64_t queueIndex = 0; queueIndex < workloadQueueTail_device_vec[deviceId]; queueIndex++)
					{
						Workload_type workload = workloadQueue_device_vec2[deviceId][queueIndex];
						partitionWorkload_vertexNum_device += workload.seg_activeVertexNum;
					}
					eachGPU_activeNum[deviceId + 1] = partitionWorkload_vertexNum_device;
					Msg_info("Partition The Workload, GPU [%d] Active VertexNum: (%zu), Active Segment: (%zu)", deviceId,
						(eachGPU_activeNum[deviceId + 1] - eachGPU_activeNum[deviceId]), 
						static_cast<uint64_t>(workloadQueueTail_device_vec[deviceId]));			
				}
				assert_msg((partitionWorkload_vertexNum_device + workloadQueueTail_host) == activeNum,
					"CPU Workload + GPUs Workload != Total Workload");
				Msg_finish("Partition The Workload Finished The Check !");
				#endif
				
				
				//> [Heavy Workload]: 2. Sort Each GPU Workload and Transfer The vertexValue To Device
				//TODO 我们采用异步引擎，然后对比sort能否带来优化
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif

				//opt_h2d();
				#pragma omp parallel for num_threads(useDeviceNum)
				for(int deviceId = 0; deviceId < useDeviceNum; deviceId++ )
				{
					CUDA_CHECK(cudaSetDevice(deviceId));	
					CUDA_CHECK(cudaMemcpyAsync(vertexValue_temp_device[deviceId], vertexValue, 
						noZeroOutDegreeNum * sizeof(vertex_data_type), cudaMemcpyHostToDevice, stream_device_vec[deviceId]));
				}
				
				#pragma omp parallel for num_threads(useDeviceNum)
				for(int deviceId = 0; deviceId < useDeviceNum; deviceId++ )
				{
					//sort(std::execution::par, workloadQueue_device_vec2[deviceId].begin(),
					sort(workloadQueue_device_vec2[deviceId].begin(), 
						workloadQueue_device_vec2[deviceId].begin() + workloadQueueTail_device_vec[deviceId],
						[&](Workload_type& a, Workload_type& b) -> bool
						{
						if(a.seg_activeVertexNum > b.seg_activeVertexNum) return true;
						else if(a.seg_activeVertexNum == b.seg_activeVertexNum)
						{
							if(a.seg_id < b.seg_id) return true;
							else                    return false;
						}else{
							return false;
						}
						}
					);	
				}

				#pragma omp parallel for num_threads(useDeviceNum)
				for(int deviceId = 0; deviceId < useDeviceNum; deviceId++ )
				{
					CUDA_CHECK(cudaStreamSynchronize(stream_device_vec[deviceId])); // Wait for the CUDA stream to finish
				}				
				

				// #pragma omp parallel num_threads(useDeviceNum * 2)
				// {
				// 	int deviceId = omp_get_thread_num();

				// 	if(deviceId < useDeviceNum)
				// 	{
				// 		sort(workloadQueue_device_vec2[deviceId].begin(), 
				// 	     workloadQueue_device_vec2[deviceId].begin() + workloadQueueTail_device_vec[deviceId],
				// 		 [&](Workload_type& a, Workload_type& b) -> bool
				// 		 {
				// 			if(a.seg_activeVertexNum > b.seg_activeVertexNum) return true;
				// 			else if(a.seg_activeVertexNum == b.seg_activeVertexNum)
				// 			{
				// 				if(a.seg_id < b.seg_id) return true;
				// 				else                    return false;
				// 			}else{
				// 				return false;
				// 			}
				// 		 }
				// 		);	
				// 	}
				// 	else
				// 	{
				// 		deviceId = deviceId % useDeviceNum;
				// 		CUDA_CHECK(cudaSetDevice(deviceId));
				// 		CUDA_CHECK(H2D(vertexValue_temp_device[deviceId], vertexValue, noZeroOutDegreeNum));
				// 	}								
				// }
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.2 [H: Sort and Tra.], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed 
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
				#endif			

				//Check The Sort
				#ifdef CGGRAPHENGINE_CHECK	
				for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
				{					
					if(workloadQueueTail_device_vec[deviceId] >= 1)
					{
						for (uint32_t workloadId = 1; workloadId < workloadQueueTail_device_vec[deviceId]; workloadId++)
						{
							assert_msg((workloadQueue_device_vec2[deviceId][workloadId - 1].seg_activeVertexNum) >=
								(workloadQueue_device_vec2[deviceId][workloadId].seg_activeVertexNum), 
								"The [%d] GPU Check The Sort Error, The Former activeVertexNum = (%zu), The Latter activeVertexNum = (%zu)",
								deviceId,
								static_cast<uint64_t>((workloadQueue_device_vec2[deviceId][workloadId - 1].seg_activeVertexNum)),
								static_cast<uint64_t>(workloadQueue_device_vec2[deviceId][workloadId].seg_activeVertexNum)
							);
						}
					}
					Msg_finish("The [%d] GPU Finished The WorkLoad Segment Sort !", deviceId);							
				}	
				#endif

				////打印
				// std::ofstream out_file;
				// out_file.open("/home/pengjie/vs_log/temp.txt",
				// 	std::ios_base::out | std::ios_base::binary);//Opens as a binary read and writes to disk
				// if (!out_file.good()) assert_msg(false, "Error opening out-file");
				// for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
				// {
				// 	for (uint32_t workloadId = 0; workloadId < workloadQueueTail_device_vec[deviceId]; workloadId++)
				// 	{
				// 		out_file << "[" << deviceId << "], (" << workloadId << "). seg_id = " <<
				// 		(workloadQueue_device_vec2[deviceId][workloadId].seg_id) << ", activeVertices = " <<
				// 		(workloadQueue_device_vec2[deviceId][workloadId].seg_activeVertexNum) << std::endl;
				// 	}
				// 	out_file << "------------------------------" << std::endl;
				// }
				// out_file.close();

				//>[Heavy Workload]: 3. Copy active dense_bitset
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif
				if(!partitionResult.canHoldAllSegment)
				{
					omp_parallel_for(uint64_t wordId=0; wordId<active.in().arrlen; wordId++)
					{
						active_steal.array[wordId] = active.in().array[wordId];
					}
				}
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.3 [H:  Copy  active], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed 
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
				#endif		


				//> [Heavy Workload]: 3. Build The Bitmap To Prevent GPU Work-Efficient	
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif		
				for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++)
				{
					omp_parallel_for(uint32_t workloadId = 0; workloadId < workloadQueueTail_device_vec[deviceId]; workloadId++)
					{
						Workload_type workload = workloadQueue_device_vec2[deviceId][workloadId];
						uint32_t segmentId = workload.seg_id;

						vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
						vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE;
						if (endVertexId >= noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;
						assert_msg((WORD_MOD(startVertexId) == 0), "startVertexId %% 64 is not 0");

						// active.in()
						uint64_t startWordIndex = WORD_OFFSET(startVertexId);
						uint64_t endWordIndex = (endVertexId + WORDSIZE  - 1) / WORDSIZE; // WORD_OFFSET(endVertexId);

						// Align
						if((endWordIndex - startWordIndex) == (SEGMENT_SIZE / WORDSIZE))
						{
							//assert_msg(segmentId != (partitionResult.segmentNum - 1), "Only The Last Segment May Be Not Align");
							std::memcpy(bitmap_host_vec[deviceId] + (SEGMENT_SIZE / WORDSIZE) * workloadId, 
								   active.in().array + startWordIndex, sizeof(uint64_t) * (SEGMENT_SIZE / WORDSIZE)
							);
							// 我们只有在GPU无法容纳时才启用清除
							if(!partitionResult.canHoldAllSegment){
								memset(active.in().array + startWordIndex, 0, sizeof(uint64_t) * (SEGMENT_SIZE / WORDSIZE));
							}
						}
						// Not Align (//TODO: 让不对齐的总是留给CPU)
						else
						{
							#ifdef CGGRAPHENGINE_CHECK
							Msg_rate("The [%d] GPU. This Msg Can Only Appear One !", deviceId);
							assert_msg(segmentId = (partitionResult.segmentNum - 1), "Only The Last Segment May Be Not Align");
							#endif

							uint64_t* temp_append = new uint64_t[SEGMENT_SIZE / WORDSIZE];
							memset(temp_append, 0, sizeof(uint64_t) * (SEGMENT_SIZE / WORDSIZE));//Init To Zero

							std::memcpy(temp_append, active.in().array + startWordIndex, 
								sizeof(uint64_t) * (endWordIndex - startWordIndex)
							);

							std::memcpy(bitmap_host_vec[deviceId] + (SEGMENT_SIZE / WORDSIZE) * workloadId, 
								   temp_append, sizeof(uint64_t) * (SEGMENT_SIZE / WORDSIZE)
							);
							// 我们只有在GPU无法容纳时才启用清除
							if(!partitionResult.canHoldAllSegment){
								memset(active.in().array + startWordIndex, 0, sizeof(uint64_t) * (endWordIndex - startWordIndex));
							}
							
						}
					}
				}
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.3 [H: Bitmap to GPU], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
				#endif

				//Check Each GPU's Bitmap
				
				#ifdef CGGRAPHENGINE_CHECK
				Msg_info("After Partition, In Active.in(), The Result Bits Are (%zu)", active.in().popcount());
				for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++)
				{
					uint64_t active_vertexNum = 0;
					#pragma omp paralle for reduction(+:active_vertexNum)
					for(uint64_t wordId = 0; wordId < partitionResult.segmentNum_device_vec[deviceId] * (SEGMENT_SIZE / WORDSIZE); wordId ++)
					{
						uint64_t active_vertexNum_local = __builtin_popcountl(bitmap_host_vec[deviceId][wordId]);
						assert_msg((active_vertexNum_local <= 64), 
							"active_vertexNum_local Error, active_vertexNum_local = %zu, current_word = %zu",
							active_vertexNum_local, bitmap_host_vec[deviceId][wordId]);
						active_vertexNum += active_vertexNum_local;
					}
					assert_msg(((eachGPU_activeNum[deviceId + 1] - eachGPU_activeNum[deviceId]) == active_vertexNum),
						"The [%d] GPU (Bitmap_device != eachGPU_activeNum), eachGPU_activeNum = (%zu), active_vertexNum = (%zu)", 
						deviceId, static_cast<uint64_t>(eachGPU_activeNum[deviceId + 1] - eachGPU_activeNum[deviceId]),
						static_cast<uint64_t>(active_vertexNum));
					Msg_finish("The [%d] GPU Finished The Bitmap_device Check !", deviceId);
				}			
				#endif


				//> [Heavy Workload]: 4. Transfer The Bitmap (Active) To Device
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif
				#pragma omp parallel num_threads(useDeviceNum)
				{
					int deviceId = omp_get_thread_num();
					CUDA_CHECK(cudaSetDevice(deviceId));
					CUDA_CHECK(H2D(bitmap_device_vec[deviceId], bitmap_host_vec[deviceId], partitionResult.segmentNum_device_vec[deviceId] * (SEGMENT_SIZE / WORDSIZE)));
				}	
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.4 [H: Trans. Bitmap], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
				#endif


				//> [Heavy Workload]: 5. Notice The Agent Driven The GPU
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif
				for (size_t deviceId = 0; deviceId < useDeviceNum; deviceId++)
				{
					assert_msg((wakeupDevice[deviceId] == 0), "(wakeupDevice != 0), wakeupDevice = %zu", wakeupDevice[deviceId]);
					wakeupDevice_mutex.lock();
					wakeupDevice[deviceId] = 1;
					wakeupDevice_mutex.unlock();
					usedDevice = true;
				}
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.5 [H: Notice  Agent], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
				#endif
				
				


				//> [Heavy Workload]: 6. CPU Processes Its Own Workload First 
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif
				if(hasOwnWorkload_CPU)
				{
					CPU_Execution_Model();
				}
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.6 [H: CPU Proc. Own], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
				#endif

				//> [Heavy Workload]: 7. CPU Steal The WorkLoads From The GPUs, Steal Granularity Is A Job
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif				

				stealMask.fill();
				int stealNum = 0;
				do{
					int currentStealTarget = (stealNum++) % useDeviceNum;
					if(stealMask.get(currentStealTarget))
					{
						if(workloadQueueHead_device_vec[currentStealTarget] < workloadQueueTail_device_vec[currentStealTarget])
						{
							//Msg_major("开始窃取GPU [%d], 此时的tail = %ld", currentStealTarget, workloadQueueTail_device_vec[currentStealTarget]);
							//TODO 我们现在采用任务抢夺的方式来窃取任务，但需要注意我们要让每个GPU的workloadsize按照jobSize对齐
							int64_t stop = workloadQueueTail_device_vec[currentStealTarget];
							int64_t stopconst = stop;
							uint32_t steal_segmentNum = workloadQueueTail_device_vec[currentStealTarget] % (jobSize_vec[currentStealTarget] / SEGMENT_SIZE);
							if(steal_segmentNum == 0) steal_segmentNum = jobSize_vec[currentStealTarget] / SEGMENT_SIZE;
							steal_segmentNum += CPU_STEAL_GRANULARITY * (jobSize_vec[currentStealTarget] / SEGMENT_SIZE);//我们希望多窃取
							stop -= steal_segmentNum;
							if(stop < 0) stop = 0;

							//! 从头开始测试
							// #pragma omp parallel
							// {
							// 	while(true)
							// 	{
							// 		uint64_t current_Head = __sync_fetch_and_add(&workloadQueueHead_device_vec[currentStealTarget], 1);
							// 		if(current_Head >= workloadQueueTail_device_vec[currentStealTarget]) break;

							// 		Workload_type workload = workloadQueue_device_vec2[currentStealTarget][current_Head];
							// 		uint32_t segmentId = workload.seg_id;
									
							// 		vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
							// 		vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE;
							// 		if (endVertexId >= noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;

							// 		uint64_t startWordIndex = WORD_OFFSET(startVertexId);
							// 		uint64_t endWordIndex = (endVertexId + WORDSIZE  - 1) / WORDSIZE;

							// 		vertex_id_type vertexId_current = 0;
							// 		for (uint64_t wordId = startWordIndex; wordId < endWordIndex; wordId++)
							// 		{
							// 			vertexId_current = wordId * WORDSIZE;
							// 			size_t word = active.in().array[wordId];

							// 			while (word != 0) {
							// 				if (word & 1) {
							// 					if(vertexId_current >= noZeroOutDegreeNum) break;
							// 					count_type vertexSocketId = getVertexSocketId(vertexId_current);
							// 					vertex_id_type vertexId_numa = vertexId_current - zeroOffset_numa[vertexSocketId];
							// 					countl_type nbr_start = csr_offset_numa[vertexSocketId][vertexId_numa];
							// 					countl_type nbr_end = csr_offset_numa[vertexSocketId][vertexId_numa + 1];

							// 					//逻辑
							// 					vertex_data_type srcVertexValue = vertexValue[vertexId_current];
							// 					for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
							// 					{
							// 						//获取dest的Offset
							// 						vertex_id_type dest = csr_dest_numa[vertexSocketId][nbr_cur];
							// 						edge_data_type weight = csr_weight_numa[vertexSocketId][nbr_cur];
							// 						vertex_data_type distance = srcVertexValue + weight;

							// 						if (distance < vertexValue[dest]) {
							// 							if (Gemini_atomic::write_min(&vertexValue[dest], distance)) {
							// 								active.out().set_bit(dest);					
							// 							}
							// 						}
							// 					}
							// 				}
							// 				vertexId_current++;
							// 				word = word >> 1;
							// 			}
							// 		}
							// 	}
							// }// end of [parallel for]

							
							//! 从尾部开始
							#pragma omp parallel
							{
								while(true)
								{
									int64_t current_Tail = __sync_sub_and_fetch(&workloadQueueTail_device_vec[currentStealTarget], 1);//每个线程一次窃取一个segment
									if(current_Tail < stop) break; // 本次要处理到的点
									if(Gemini_atomic::atomic_large(&workloadQueueHead_device_vec[currentStealTarget], current_Tail)) break;
									//if(current_Tail <  static_cast<int64_t>()) break;//用原子操作

									//assert_msg(current_Tail >=0, "current_Tail = %ld", current_Tail);
									Workload_type workload = workloadQueue_device_vec2[currentStealTarget][current_Tail];
							 		uint32_t segmentId = workload.seg_id;

									#ifdef CGGRAPHENGINE_DETAIL_DEBUG
									CPUSegmentCount_vec[omp_get_thread_num()]++;
									#endif

									vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
							 		vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE;
							 		if (endVertexId >= noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;

							 		uint64_t startWordIndex = WORD_OFFSET(startVertexId);
							 		uint64_t endWordIndex = (endVertexId + WORDSIZE  - 1) / WORDSIZE;

							 		vertex_id_type vertexId_current = 0;
									for (uint64_t wordId = startWordIndex; wordId < endWordIndex; wordId++)
							 		{
										vertexId_current = wordId * WORDSIZE;
										size_t word = 0;
										if(partitionResult.canHoldAllSegment)
										{
											word =  active.in().array[wordId];
										}
										else
										{
											word = active_steal.array[wordId];
										}

										while (word != 0) {
											if (word & 1) {
												if(vertexId_current >= noZeroOutDegreeNum) break;
												count_type vertexSocketId = getVertexSocketId(vertexId_current);
												vertex_id_type vertexId_numa = vertexId_current - zeroOffset_numa[vertexSocketId];
												countl_type nbr_start = csr_offset_numa[vertexSocketId][vertexId_numa];
												countl_type nbr_end = csr_offset_numa[vertexSocketId][vertexId_numa + 1];

												//逻辑
												if (Algorithm_type::BFS == algorithm)
												{
													BFS_SPACE::bfs_numa_steal<CGgraphEngine>(*this, vertexId_current, nbr_start, nbr_end, vertexSocketId, true);
												}
												else if (Algorithm_type::SSSP == algorithm)
												{
													SSSP_SPACE::sssp_numa_steal<CGgraphEngine>(*this, vertexId_current, nbr_start, nbr_end, vertexSocketId, true);
												}
												else if (Algorithm_type::CC == algorithm)
												{
													CC_SPACE::wcc_numa_steal<CGgraphEngine>(*this, vertexId_current, nbr_start, nbr_end, vertexSocketId, true);
												}
												else
												{
													assert_msg(false, "CPU Steal Model Meet Unknown Algorithm");
												}
											}
											vertexId_current++;
							 				word = word >> 1;
										}
									}
									
								}
							}// end of [parallel for]

							// 我们要将多减去的归位
							//assert_msg(workloadQueueTail_device_vec[currentStealTarget] <= stop, "error");
							__sync_lock_test_and_set(&workloadQueueTail_device_vec[currentStealTarget], stop);
							
							//Msg_major("窃取GPU [%d]结束, 此时的tail = %ld", currentStealTarget,workloadQueueTail_device_vec[currentStealTarget]);
						}
						else
						{
							stealMask.clear_bit(currentStealTarget);
							//Msg_info("The [%d] GPU Has No Workload To Steal", currentStealTarget);
							if(stealMask.empty()) break;
							
						}										
					}// Current GPU Still Has Workload

				}while(true);
				//printf("退出\n");
				processorEndTime_vec[useDeviceNum] = processorEnd_time.curTimeval();
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.7 [H: CPU Fi. Steal], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< ", stealNum = " << stealNum
						<< std::endl;
				#endif

				//> 8. Wait To All Processors Complete
				{
					#ifdef CGGRAPHENGINE_DETAIL_DEBUG
						heavyTemp_time.start();
					#endif
					for (size_t deviceId = 0; deviceId < useDeviceNum; deviceId++)
					{
						while (true)
						{
							wakeupDevice_mutex.lock();
							bool deviceComplete = (wakeupDevice[deviceId] == 0);
							wakeupDevice_mutex.unlock();
							if (deviceComplete) break;
							__asm volatile ("pause" ::: "memory");
						}
					}
									
					#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.8 [H: CPU  Waiting], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;
					#endif
				}
				

				//> 9. Reduce
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					heavyTemp_time.start();
				#endif
				if(usedDevice_ite) reduce();
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
					logstream(LOG_INFO) << "\t\t\t2.9 [H: CPU  Reducing], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< heavyTemp_time.get_time_ms() << " (ms)" 
						<< std::endl;

					//分析时间
					for(int deviceId=0; deviceId < useDeviceNum; deviceId++)
					{
						logstream(LOG_INFO) << "\t\t\t2.10[H: CPU vs Mu-GPU], Time: The GPU[" << deviceId
							<<"] - CPU :" <<  std::setw(7) << std::setprecision(2) << std::fixed
							<< ((double)(processorEndTime_vec[deviceId].tv_sec - processorEndTime_vec[useDeviceNum].tv_sec) +
								((double)(processorEndTime_vec[deviceId].tv_usec - processorEndTime_vec[useDeviceNum].tv_usec)) / 1.0E6) * 1000
							<< " (ms)"
							<< std::endl;
					}
					//分析segment量
					uint64_t total_CPU_segments = 0;
					for (int tid = 0; tid < ThreadNum; tid++)
					{
						total_CPU_segments += CPUSegmentCount_vec[tid];
					}
					uint64_t total_GPU_segments = 0;
					uint64_t total_segments_ready = 0;
					for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
					{
						total_GPU_segments += processSegmentCount_vec[deviceId];
						total_segments_ready += workloadQueueConstTail_device_vec[deviceId];
					}	
					total_segments_ready += workloadQueueTail_host;				
					logstream(LOG_INFO) << "\t\t\t2.11[H: CPU vs Mu-GPU], Segment: The GPU Process Segments (" << total_GPU_segments
						<< "), The CPU Process Segments (" << total_CPU_segments << "), Total_segments (" << total_segments_ready << ")"
						<< std::endl;
				#endif
			}// end of [Heavy Workload]

			//> workloadHet : [Light]
			else
			{
				usedDevice_ite = false;
				CPU_Execution_Model();

			}// end of [Light Workload]

			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			logstream(LOG_INFO) << "\t\t2. [Light or Heavy Computation], Used time :" 
						<< std::setw(7) << std::setprecision(2) << std::fixed
						<< tempTime.get_time_ms() << " (ms)" 
						<< std::endl;
			#endif


			//activeNum = parallel_popcount(active.out().array);
			//activeNum = active.out().parallel_popcount();

			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				tempTime.start();
			#endif
			if(workloadHet == Workload_heterogeneous::HEAVY) {
				activeNum = parallel_popcount(active.out().array);
				activeNum_host = activeNum - activeNum_device;
			}else{		
				activeNum = parallel_popcount(active.out().array);		
				activeNum_host = activeNum; //! 注意这种计数并不准确，但是可以用于确定activeNum是否为0,需要精确则需调用parallel_popcount()
				activeNum_device = 0;
			}
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				logstream(LOG_INFO) << "\t\t3. [Get Active Num], Used time :" 
					<< std::setw(7) << std::setprecision(2) << std::fixed
					<< tempTime.get_time_ms() << " (ms)" 
					<< std::endl;
			#endif
           
			

			std::cout << "\t[Single-Ite]：第(" << std::setw(3) << ite << ")次迭代完成, WorkloadHet:{"
					<< workloadHet 
					<< "}, time = (" << std::setw(7) << std::setprecision(2) << std::fixed << iteTime_single.current_time_millis()
					<< " ms), activeNum_host = " << activeNum_host
					<< ", activeNum_device = " << activeNum_device
					<< ", activeNum  = (" << activeNum << ")" 
					<< std::endl;

			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				logstream(LOG_INFO) << "\t[Single-Ite]：第(" << std::setw(3) << ite << ")次迭代完成, WorkloadHet:{"
					<< workloadHet 
					<< "}, time = (" << std::setw(7) << std::setprecision(2) << std::fixed << iteTime_single.current_time_millis()
					<< " ms), activeNum_host = " << activeNum_host
					<< ", activeNum_device = " << activeNum_device
					<< ", activeNum  = (" << activeNum << ")" << std::endl;
			#endif

			if (activeNum == 0)
			{
				duration = iteTime_total.current_time_millis();
				std::cout << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
					<< ", time: " << std::setw(6) << std::setprecision(2) << std::fixed << duration << " (ms)" << std::endl;

				hybirdComplete = 1;
				for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
				{
					agent_vec[deviceId].join();
				}

				break;
			}

			// #ifdef CGGRAPHENGINE_DETAIL_DEBUG
			// 	tempTime.start();
			// #endif
			active.swap();
			// #ifdef CGGRAPHENGINE_DETAIL_DEBUG
			// 	logstream(LOG_INFO) << "\t\t[4. Swap], Used time :" 
			// 		<< std::setw(7) << std::setprecision(2) << std::fixed <<
			// 		<< tempTime.get_time_ms() << " (ms)" 
			// 		<< std::endl;
			// #endif

		}while(1);

		//[CHOOSE]: 如果启用过Device, 最后将outDegree为0的顶点从Device端传回到Host端更新
		if (usedDevice)
		{
			reduce_zeroOutDegree();
		}	

		return duration;
		
	}







private:

	/* **********************************************************
	 * Func: Host Func, Init Graph and Numa
	 * **********************************************************/
	void initGraphNuma()
	{
		//检查NUMA是否可用
		assert_msg(numa_available() != -1, "NUMA Not Available");

		threadNum = ThreadNum;
		socketNum = SocketNum;
		threadPerSocket = ThreadNum / SocketNum;

		omp_parallel_for(int threadId = 0; threadId < threadNum; threadId++)
		{
#ifdef CGGRAPHENGINE_DEBUG
			int thread_id = omp_get_thread_num();//获取线程的id
			int core_id = sched_getcpu();//获取物理的id
			int socket_id = getThreadSocketId(threadId);
			logstream(LOG_INFO) << "[" << std::setw(2) << threadId
				<< "]: Thread(" << std::setw(2) << thread_id
				<< ") is running on CPU{" << std::setw(2) << core_id
				<< "}, socketId = (" << std::setw(2) << socket_id
				<< ")" << std::endl;
#endif	
			assert_msg(numa_run_on_node(getThreadSocketId(threadId)) == 0, "numa_run_on_node 出错");
		}

		//Init CSR_NUMA Pointer
		csr_offset_numa = new countl_type * [socketNum];
		csr_dest_numa = new vertex_id_type * [socketNum];
		csr_weight_numa = new vertex_id_type * [socketNum];

		vertexNum_numa = new count_type[socketNum];
		edgeNum_numa = new countl_type[socketNum];
		zeroOffset_numa = new count_type[socketNum];

		//Init taskSteal
		taskSteal = new TaskSteal();
		taskSteal_align64 = new TaskSteal();
		taskSteal_hostHelp = new TaskSteal();
		taskSteal_totalTask = new TaskSteal();

		//Init outDegree
		outDegree = new offset_type[vertexNum];
	}// end of function [initGraphNuma()]



    /* ************************************************************************
	 * Func: Host Function, Partition The Host Graph According To The NUMA
	 * ************************************************************************/
	void partitionGraphByNuma()
	{
		//First, calculate the partition points, and we use a simple average partition
		numa_offset = new count_type [socketNum + 1];
		numa_offset[0] = 0;

		vertexPerSocket = noZeroOutDegreeNum / socketNum / WORDSIZE * WORDSIZE; //必选按word对齐,否则active会在窃取时乱套

		for (count_type socketId = 1; socketId < socketNum; socketId++)
		{
			numa_offset[socketId] = numa_offset[socketId - 1] + vertexPerSocket;
		}
		numa_offset[socketNum] = noZeroOutDegreeNum;

		// Partition [CSR_Result_type] To Different NUMA Node
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			count_type vertex_numa = numa_offset[socketId + 1] - numa_offset[socketId];
			countl_type edges_numa = csr_offset[numa_offset[socketId + 1]] - csr_offset[numa_offset[socketId]];

			csr_offset_numa[socketId] = (countl_type*)numa_alloc_onnode((vertex_numa + 1) * sizeof(countl_type), socketId);
			offset_type offset = csr_offset[numa_offset[socketId]];
			for (count_type i = 0; i < (vertex_numa + 1); i++)
			{
				csr_offset_numa[socketId][i] = csr_offset[numa_offset[socketId] + i] - offset; //让CSR的offset从零开始
			}

			csr_dest_numa[socketId] = (vertex_id_type*)numa_alloc_onnode((edges_numa) * sizeof(vertex_id_type), socketId);
			memcpy(csr_dest_numa[socketId], csr_dest + csr_offset[numa_offset[socketId]], edges_numa * sizeof(vertex_id_type));

			csr_weight_numa[socketId] = (edge_data_type*)numa_alloc_onnode((edges_numa) * sizeof(edge_data_type), socketId);
			memcpy(csr_weight_numa[socketId], csr_weight + csr_offset[numa_offset[socketId]], edges_numa * sizeof(edge_data_type));

			//存储常用到的变量
			vertexNum_numa[socketId] = vertex_numa;
			edgeNum_numa[socketId] = edges_numa;
			zeroOffset_numa[socketId] = numa_offset[socketId];
		}


#ifdef CGGRAPHENGINE_DEBUG
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			Msg_info("Socket[%zu] Get Vertices:(%9zu), Edges: (%10zu)", socketId,
				static_cast<uint64_t>(numa_offset[socketId + 1] - numa_offset[socketId]), static_cast<uint64_t>(edgeNum_numa[socketId]));
		}
#endif

	}// end of function [partitionGraphByNuma()]

	/* **********************************************************
	 * Func: Host Function, Get socketId of vertexId
	 * **********************************************************/
	inline count_type getVertexSocketId(vertex_id_type vertexId)
	{
		count_type socketId = 0;
		for (; socketId < socketNum; socketId++)
		{
			if(vertexId < numa_offset[socketId + 1]) return socketId;
		}

		assert_msg(false, "getVertexSocketId Error, vertexId = %u, numa_offset[%u] = %u", vertexId, socketId , numa_offset[socketId]);
		return static_cast<count_type>(0);
	}


    /* **************************************************************************
	 * Func: Host Function, Allocate Memory For vertexValue and Active
	 * **************************************************************************/
	void allocate_vertexValueAndActive()
	{
		taskSteal_align64->allocateTaskForThread<count_type>(noZeroOutDegreeNum, 64, true);
		taskSteal_hostHelp->allocateTaskForThread<count_type>(noZeroOutDegreeNum);

#ifdef PIN_MEM
		CUDA_CHECK(cudaMallocHost((void**)&(vertexValue), (vertexNum) * sizeof(vertex_data_type)));
#else
		vertexValue = new vertex_data_type[vertexNum];
		Msg_info("**********************************************************************");
		Msg_info("*   [Used PIN Memory Tranfer VertexValue Between Host And Device]   *");
		Msg_info("**********************************************************************");
#endif

		active_in.resize(noZeroOutDegreeNum);
		active_out.resize(noZeroOutDegreeNum);
		active_steal.resize(noZeroOutDegreeNum);
		active.setDoubleBuffer(active_in, active_out); // Encapsulated into DoubleBuffer
	}

    /* **********************************************************
	 * Func: Host Function, Build outDegree
	 * **********************************************************/
	void get_outDegree()
	{
		omp_parallel_for(vertex_id_type vertexId = 0; vertexId < noZeroOutDegreeNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}
	}


    /* **********************************************************
	 * Func: Host Function, Malloc The Memory For Device
	 * **********************************************************/
	void malloc_device()
	{ 
        //common_host.resize(useDeviceNum);
        common_device.resize(useDeviceNum);
        vertexValue_temp_host.resize(useDeviceNum);
        vertexValue_temp_device.resize(useDeviceNum);

        for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++)
        {
			CUDA_CHECK(cudaSetDevice(deviceId));

            // common
           //common_type* common_temp_host;
            common_type* common_temp_device;
            //CUDA_CHECK(MALLOC_HOST(&common_temp_host, common_size));
            CUDA_CHECK(MALLOC_DEVICE(&common_temp_device, common_size));

            if((std::numeric_limits<common_type>::max() <= vertexNum)||
            (std::numeric_limits<common_type>::max() <= edgeNum)){ 
                assert_msg(false, "<common_type> overflow !");}

            common_type vertexNum_ = static_cast<common_type>(vertexNum); 
            CUDA_CHECK(H2D(common_temp_device + 2, &vertexNum_, 1));
            common_type edgeNum_ = static_cast<common_type>(edgeNum); 
            CUDA_CHECK(H2D(common_temp_device + 3, &edgeNum_, 1));
            common_type noZeroOutDegreeNum_ = static_cast<common_type>(noZeroOutDegreeNum); 
            CUDA_CHECK(H2D(common_temp_device + 4, &noZeroOutDegreeNum_, 1));
            common_type SEGMENT_SIZE_ = static_cast<common_type>(SEGMENT_SIZE); 
            CUDA_CHECK(H2D(common_temp_device + 5, &SEGMENT_SIZE_, 1));
            common_type JOB_SIZE_ = static_cast<common_type>(jobSize_vec[deviceId]); 
            CUDA_CHECK(H2D(common_temp_device + 6, &JOB_SIZE_, 1));
			common_type segment_start_ = static_cast<common_type>(0); 
            CUDA_CHECK(H2D(common_temp_device + 7, &segment_start_, 1));

            //common_host[deviceId] = common_temp_host;
            common_device[deviceId] = common_temp_device;


            // vertexValue
            vertex_data_type* temp_host;
            vertex_data_type* temp_device;
            CUDA_CHECK(MALLOC_HOST(&temp_host, vertexNum));
            CUDA_CHECK(MALLOC_DEVICE(&temp_device, vertexNum));
            vertexValue_temp_host[deviceId] = temp_host;
            vertexValue_temp_device[deviceId] = temp_device;          
        }     
	}// end of function [malloc_device]


	/* ***********************************************************
	 * Func: Malloc The Bitmap (Active) For Device
	 * ***********************************************************/
	void build_bitmapForDevice()
	{
		 bitmap_host_vec.resize(useDeviceNum);
		 bitmap_device_vec.resize(useDeviceNum);
		 stream_device_vec.resize(useDeviceNum);

		 for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		 {
			CUDA_CHECK(cudaSetDevice(deviceId));

			uint64_t bitmap_arrayLength = partitionResult.segmentNum_device_vec[deviceId] * (SEGMENT_SIZE / WORDSIZE);// Max
			Msg_info("The [%d] GPU Allocate (%zu) Bits, (%zu) Words For Bitmap (Active)", deviceId,
				static_cast<uint64_t>(bitmap_arrayLength * WORDSIZE),static_cast<uint64_t>(bitmap_arrayLength));

			uint64_t* bitmap_temp_host;
			uint64_t* bitmap_temp_device;
			CUDA_CHECK(MALLOC_HOST(&bitmap_temp_host, bitmap_arrayLength));
			memset(bitmap_temp_host, 0, sizeof(uint64_t) * bitmap_arrayLength);		
			bitmap_host_vec[deviceId] = bitmap_temp_host;

			CUDA_CHECK(MALLOC_DEVICE(&bitmap_temp_device, bitmap_arrayLength));
			CUDA_CHECK(MEMSET_DEVICE(bitmap_temp_device, bitmap_arrayLength));
			bitmap_device_vec[deviceId] = bitmap_temp_device;

			uint32_t* trasfer_segment_temp_host;
			uint32_t* trasfer_segment_temp_device;
			CUDA_CHECK(MALLOC_HOST(&trasfer_segment_temp_host, GPU_WORK_GRANULARITY * (jobSize_vec[deviceId]/SEGMENT_SIZE)));
			CUDA_CHECK(MALLOC_DEVICE(&trasfer_segment_temp_device, GPU_WORK_GRANULARITY * (jobSize_vec[deviceId]/SEGMENT_SIZE)));
			trasfer_segment_host_vec[deviceId] = trasfer_segment_temp_host;
			trasfer_segment_device_vec[deviceId] = trasfer_segment_temp_device;

			cudaStream_t stream;
			cudaStreamCreate(&stream);
			stream_device_vec[deviceId] = stream;
		 }
	}// end of function [build_bitmapForDevice()]


	/* **********************************************************
	 * Func: Reset Required Variables
	 * **********************************************************/
	void resetVariable()
	{
		
		for(int deviceId = 0; deviceId < useDeviceNum; deviceId++) wakeupDevice[deviceId] = 0; 
		hybirdComplete = 0; nBlock = 0; usedDevice = false;        // Device
		activeNum_device = 0; activeNum_host = 0; activeNum = 0; activeEdgeNum = 0;  // Active
		ite = 0;                                                                     // Ite
		noWorkloadSeg = 0;                                                           // Workload

		// Workload
		workloadQueueHead_host = 0;
		workloadQueueTail_host = 0;
		for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			workloadQueueHead_device_vec[deviceId] = 0;
			workloadQueueTail_device_vec[deviceId] = 0;
		}
		

	}// end of func [resetVariable()]


	/* **********************************************************
	 * Func: Init The Algorithm
	 *
	 * @param [vertex_id_type root] Some Alg NO Need This param
	 * **********************************************************/
	void initAlgorithm(vertex_id_type root)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			// Host - Active (The noZeroOutDegreeVertices is Relatively Small, We don not use parallel)	

			active.in().clear_memset(); 
			active.out().clear_memset();

			active.in().set_bit(root);
			activeNum = 1;

			//Host - vertexValue
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = VertexValue_MAX;
			vertexValue[root] = 0;

			for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++)
			{
				CUDA_CHECK(cudaSetDevice(deviceId));
				memcpy(vertexValue_temp_host[deviceId], vertexValue, vertexNum * sizeof(vertex_data_type));
				CUDA_CHECK(H2D(vertexValue_temp_device[deviceId], vertexValue, vertexNum));
			}
		}
		else if(Algorithm_type::CC == algorithm)
		{
			// Host - Active

			active.in().clear_memset();
			active.out().clear_memset();

			active.in().fill(); //can used parallel_fill
			activeNum = noZeroOutDegreeNum;

			//Host - vertexValue
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = i;

			for(int deviceId = 0; deviceId < useDeviceNum; deviceId ++)
			{
				CUDA_CHECK(cudaSetDevice(deviceId));
				memcpy(vertexValue_temp_host[deviceId], vertexValue, vertexNum * sizeof(vertex_data_type));
				CUDA_CHECK(H2D(vertexValue_temp_device[deviceId], vertexValue, vertexNum));
			}

		}
		else if(Algorithm_type::PR == algorithm)
		{
			assert_msg(false, "TODO: PageRank !");
		}
		else
		{
			assert_msg(false, "initAlgorithm Meet Unknown Algorithm");
		}
	}


	/* *******************************************************************
	 * Func: Host Function, Clear The ActiveOut After Each Iteration
	 * *******************************************************************/
	void clearActiveOut()
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm)|| 
			(Algorithm_type::CC == algorithm))
		{
			omp_parallel
			{
				count_type threadId = omp_get_thread_num();
				size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
				size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);
				memset(active.out().array + cur, 0, sizeof(size_t) * (end - cur));
			}
		}
		else if (Algorithm_type::PR == algorithm)
		{
			assert_msg(false, "TODO: PageRank");
		}
		else
		{
			assert_msg(false, "clearActiveOut Meet Unknown Algorithm");
		}
	}


	/* ***************************************************************************
	 * Func: Find the Word index where the first 1 in the active is located
	 * ***************************************************************************/
	count_type findFirstActiveWord()
	{
		count_type firstWord = 0;
		for (count_type i = 0; i < active.in().arrlen; i++)
		{
			if (active.in().array[i] != 0)
			{
				firstWord = i;
				break;
			}
		}

		return firstWord;
	}


	/* ***************************************************************************
	 * Func: Find the Word index where the last 1 in the active is located
	 * ***************************************************************************/
	count_type findLastActiveWord()
	{
		count_type lastWord = active.in().arrlen - 1;
		for (int64_t i = (active.in().arrlen - 1); i >= 0; i--)
		{
			if (active.in().array[i] != 0)
			{
				lastWord = i;
				break;
			}
		}

		return lastWord;
	}

	/* ***************************************************************************
	 * Func: Get The Total Active Edges In The Current Workload
	 * ***************************************************************************/
	countl_type getActiveEdgeNum(size_t first, size_t last)
	{
		size_t _first = first * WORDSIZE; //回归到顶点,第一个顶点
		size_t _last = last * WORDSIZE;   //回归到顶点,最后一个顶点
		size_t work = _last - _first;
		size_t totalWorkloads = 0;

		totalWorkloads = taskSteal_totalTask->twoStage_taskSteal<size_t, size_t>(work,
			[&](size_t& current, size_t& local_workloads)
			{
				current += _first;

				size_t word = active.in().array[WORD_OFFSET(current)];
				while (word != 0)
				{
					if (word & 1)
					{
						local_workloads += outDegree[current];
					}
					current++;
					word = word >> 1;
				}
			},
			VERTEXWORK_CHUNK
		);
		return static_cast<countl_type>(totalWorkloads);
	}


	/* **********************************************************
	 * Func: Host Function, Parallel Get popcount
	 * **********************************************************/
	size_t parallel_popcount(size_t* array_process)
	{
		size_t totalSet = 0;
#pragma omp parallel reduction(+:totalSet)
		{
			count_type threadId = omp_get_thread_num();
			size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
			size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);
			count_type totalSet_local = 0;
			for (size_t i = cur; i < end; i++)
			{
				totalSet_local += __builtin_popcountl(array_process[i]);
			}
			totalSet += totalSet_local;
		}
		return totalSet;
	}




private:

	/* **********************************************************
	 * Func: CPU Execution Model
	 * **********************************************************/
	inline void CPU_Execution_Model()
	{
		activeNum_host = taskSteal->threeStage_taskSteal<CGgraphEngine>(*this, 
			[&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end, count_type socketId, bool sameSocket)
			{
				if (Algorithm_type::BFS == algorithm)
				{
					return BFS_SPACE::bfs_numa<CGgraphEngine>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
					//return BFS_SPACE::bfs_numa_lastzero<CGgraphEngine>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
				}
				else if (Algorithm_type::SSSP == algorithm)
				{
					return SSSP_SPACE::sssp_numa<CGgraphEngine>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
					//return SSSP_SPACE::sssp_numa_lastzero<CGgraphEngine>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
				}
				else if (Algorithm_type::CC == algorithm)
				{
					return CC_SPACE::cc_numa<CGgraphEngine>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
					//return CC_SPACE::cc_numa_lastzero<CGgraphEngine>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
				}
				else
				{
					assert_msg(false, "CPU Execution Model Meet Unknown Algorithm");
					return static_cast<count_type>(0);
				}
			}
		);
	}

	void GPU_Execution_Model_1(int deviceId)
	{
		while(!hybirdComplete)
		{
			while (true)
			{
				wakeupDevice_mutex.lock();
				bool deviceProcess = ((wakeupDevice[deviceId] == 1) || (hybirdComplete));
				wakeupDevice_mutex.unlock();
				if (deviceProcess) break;
				__asm volatile ("pause" ::: "memory");
			}
			if (hybirdComplete) break;

			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			processorEndTime_vec[deviceId] = processorEnd_time.curTimeval();
			#endif

			wakeupDevice_mutex.lock();
			wakeupDevice[deviceId] = 0;
			wakeupDevice_mutex.unlock();
		}
		
	}

	/* **********************************************************
	 * Func: GPU Execution Model
	 * **********************************************************/
	void GPU_Execution_Model(int deviceId)
	{
		//Bind Agents To Given Cores
		int coreId = 0;
		
		// The Current Server Of SUS, Its Two GPUs All In NUMA-0
		if(deviceId == 0) coreId = 0;
		else if (deviceId == 1) coreId = 1;
		else assert_msg(useDeviceNum <= 2, "When The UseDeviceNum > 2, YOU Can Add The CoreId That You Want To Bind.");

		if (threadBindToCore(coreId))
		{
			assert_msg((sched_getcpu() == coreId), "(sched_getcpu() != coreId) -> (%u != %u)", sched_getcpu(), coreId);
			Msg_info("std::thread success bind to core [%u]", coreId);
		}
		else Msg_info("[Failed]: std::thread bind to core [%u] failed", coreId);


		CUDA_CHECK(cudaSetDevice(deviceId));

		timer deviceTimeTotal;
		timer deviceTime;

		while(!hybirdComplete)
		{
			while (true)
			{
				wakeupDevice_mutex.lock();
				bool deviceProcess = ((wakeupDevice[deviceId] == 1) || (hybirdComplete));
				wakeupDevice_mutex.unlock();
				if (deviceProcess) break;
				__asm volatile ("pause" ::: "memory");
			}
			if (hybirdComplete) break;

			#ifdef CGGRAPHENGINE_DEBUG
			deviceTimeTotal.start();
			deviceTime.start();
			#endif

			//> 1. Async Transfer The Bitmap_device To Device
			CUDA_CHECK(cudaMemcpyAsync(bitmap_device_vec[deviceId], bitmap_host_vec[deviceId], 
				workloadQueueConstTail_device_vec[deviceId] * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_device_vec[deviceId]));

			//> 2. Get The SegmentId and GPU Execution
			int itee = 0;
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			deviceTime.start();		
			#endif	
			while(true){				
				//Get The SegmentId
				// printf("[%d], %d行, workloadQueueHead_device_vec[deviceId] = %zu, len = %zu\n", deviceId, __LINE__, 
				// workloadQueueHead_device_vec[deviceId], (jobSize_vec[deviceId] / SEGMENT_SIZE));
				// TODO 把segmentId也全部传过去
				size_t changeVar = (GPU_WORK_GRANULARITY / (pow(2, itee))) > 20 ?  (GPU_WORK_GRANULARITY / (pow(2, itee))) : 20;
				size_t changeSize =  changeVar * (jobSize_vec[deviceId] / SEGMENT_SIZE) ;
				size_t segmentId_current = __sync_fetch_and_add(&workloadQueueHead_device_vec[deviceId], changeSize);
				if(Gemini_atomic::atomic_smallEqu(&workloadQueueTail_device_vec[deviceId], static_cast<int64_t>(segmentId_current))) break;
				//if(segmentId_current >= workloadQueueTail_device_vec[deviceId])  break;

				int64_t segment_length = static_cast<int64_t>(changeSize) ;
				//queue_mutex.lock();
				int64_t atomic_len = Gemini_atomic::atomic_length(&workloadQueueTail_device_vec[deviceId], 
					static_cast<int64_t>(segmentId_current));
				if(atomic_len <= 0) break;
				if(atomic_len <  segment_length)  segment_length = atomic_len;
				
				#ifdef CGGRAPHENGINE_DETAIL_DEBUG
				processSegmentCount_vec[deviceId] += segment_length;
				#endif
				
				// if(segmentId_current + segment_length >= workloadQueueTail_device_vec[deviceId]){
				// 	segment_length =  workloadQueueTail_device_vec[deviceId] - segmentId_current;
				// }				
				// Msg_info("--->The GPU [%d] 第 (%d) 次迭代的segmentId_current = %zu, segment_length = %zu", deviceId, itee,
				// 	 static_cast<uint64_t>(segmentId_current),
				// 	 static_cast<uint64_t>(segment_length)
				// );
				//queue_mutex.unlock();
				for(int64_t segmentId = 0; segmentId < segment_length; segmentId ++)
				{
					trasfer_segment_host_vec[deviceId][segmentId] = 
						workloadQueue_device_vec2[deviceId][segmentId_current + segmentId].seg_id;
					// if(deviceId == 0 && itee < 2)
					// {
					// 	// Msg_info("--->The GPU [%d] 第 (%d) 次迭代的segment = %zu, (%zu),: index = %zu, offset = %zu", deviceId, itee, 
					// 	// 	static_cast<uint64_t>(trasfer_segment_host_vec[deviceId][segmentId]),
					// 	// 	static_cast<uint64_t>(workloadQueue_device_vec2[deviceId][segmentId_current + segmentId].seg_id),
					// 	// 	segmentId,
					// 	// 	static_cast<uint64_t>(segmentId_current + segmentId)
					// 	// );
					// }
					
				}
				//Msg_rate("【%d】trasfer_segment_device_vec-H2D前", deviceId);
				CUDA_CHECK(H2D(trasfer_segment_device_vec[deviceId], trasfer_segment_host_vec[deviceId], segment_length));
				//Msg_rate("【%d】trasfer_segment_device_vec-H2D完成", deviceId);

				common_type segment_start_ = static_cast<common_type>(segmentId_current); 
				CUDA_CHECK(H2D(common_device[deviceId] + 7, &segment_start_, 1));
				//Msg_rate("segment_start_-H2D完成");

				CUDA_CHECK(cudaStreamSynchronize(stream_device_vec[deviceId])); // Wait for the CUDA stream to finish
				//Msg_rate("【%d】cudaStreamSynchronize完成", deviceId);

				// GPU execution
				nBlock = (segment_length * SEGMENT_SIZE + BLOCKSIZE - 1) / BLOCKSIZE;
				if (algorithm == Algorithm_type::BFS)
				{
					BFS_SPACE::bfs_device<CGgraphEngine>(*this, nBlock, deviceId);
				}
				else if (algorithm == Algorithm_type::SSSP)
				{
					SSSP_SPACE::sssp_device<CGgraphEngine>(*this, nBlock, deviceId);
				}
				else if (algorithm == Algorithm_type::CC)
				{
					assert_msg(false, "Wait For CC In GPU_EXECUTION_MODEL");
				}
				else if (algorithm == Algorithm_type::PR)
				{
					assert_msg(false, "Wait For PagerRank In GPU_EXECUTION_MODEL");
				}
				else
				{
					assert_msg(false, "graphProcessDevice 时, 发现未知算法");
				}
				itee ++;
				//Msg_info("The [%d] GPU Finish The (%d) Kernel", deviceId, itee++);			
			}
			

			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			processorEndTime_vec[deviceId] = processorEnd_time.curTimeval();
			Msg_info("The GPU[%d] Workload-Kernel Finish, Used time: %.2lf (ms)", deviceId, deviceTime.get_time_ms());
			#endif

			//> 3. Trasfer The Result To The Host
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			deviceTime.start();		
			#endif	
			CUDA_CHECK(D2H(vertexValue_temp_host[deviceId], vertexValue_temp_device[deviceId], noZeroOutDegreeNum));
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			Msg_info("The GPU[%d] Result Transfer, Used time: %.2lf (ms)", deviceId, deviceTime.get_time_ms());
			#endif	

			//> 4. Wait To WakeUp
			wakeupDevice_mutex.lock();
			wakeupDevice[deviceId] = 0;
			wakeupDevice_mutex.unlock();
			#ifdef CGGRAPHENGINE_DETAIL_DEBUG
			Msg_finish("The [%d] GPU Finish All (%zu) Kernel, Use time: %.2lf (ms)", deviceId, static_cast<uint64_t>(itee), deviceTimeTotal.get_time_ms());
			#endif	
		}

	}// end of function [GPU_Execution_Model]



	/* **********************************************************
	 * Func: Reduce   //TODO
	 * **********************************************************/
	void reduce()
	{
		count_type activeNum_device_total = 0;

		if(useDeviceNum == 2)
		{
			// First,deviceReduce
			omp_parallel_for(int vertexId =0; vertexId < noZeroOutDegreeNum; vertexId++)
			{
				vertex_data_type msg = vertexValue_temp_host[1][vertexId];
				if (msg < vertexValue_temp_host[0][vertexId])
				{
					vertexValue_temp_host[0][vertexId] = msg;
					//printf("----\n");
				}
			} 	

			// Second, Reduce
			#pragma omp parallel reduction(+:activeNum_device_total)
			{
				count_type threadId = omp_get_thread_num();
				count_type activeNum_device_local = 0;

				for (size_t i = taskSteal_hostHelp->thread_state[threadId]->cur;
					i < taskSteal_hostHelp->thread_state[threadId]->end;
					i++)
				{
					vertex_data_type msg = vertexValue_temp_host[0][i];
					if (msg < vertexValue[i])
					{
						vertexValue[i] = msg;
						active.out().set_bit(i);
						activeNum_device_local += 1;
					}
				}

				activeNum_device_total += activeNum_device_local;
			}

		}
		else if (useDeviceNum == 1)
		{
#pragma omp parallel reduction(+:activeNum_device_total)
			{
				count_type threadId = omp_get_thread_num();
				count_type activeNum_device_local = 0;

				for (size_t i = taskSteal_hostHelp->thread_state[threadId]->cur;
					i < taskSteal_hostHelp->thread_state[threadId]->end;
					i++)
				{
					vertex_data_type msg = vertexValue_temp_host[0][i];
					if (msg < vertexValue[i])
					{
						vertexValue[i] = msg;
						active.out().set_bit(i);
						activeNum_device_local += 1;
					}
				}

				activeNum_device_total += activeNum_device_local;
			}
		}
		else
		{	
			assert_msg(false, "Reduce Error");
		}

		activeNum_device = activeNum_device_total;

	}

	/* **********************************************************
	 * Func: Reduce Zero Out Degree  //TODO
	 * **********************************************************/
	void reduce_zeroOutDegree()
	{
		if(useDeviceNum == 2)
		{
			#pragma omp parallel num_threads(useDeviceNum)
			{
				int threadId = omp_get_thread_num();
				CUDA_CHECK(
					D2H(vertexValue_temp_host[threadId] + noZeroOutDegreeNum, 
					vertexValue_temp_device[threadId] + noZeroOutDegreeNum, (zeroOutDegreeNum))
				);
			}

			// First,deviceReduce
			omp_parallel_for(vertex_id_type vertexId = noZeroOutDegreeNum; vertexId < vertexNum; vertexId++)
			{
				vertex_data_type msg = vertexValue_temp_host[1][vertexId];
				if (msg < vertexValue_temp_host[0][vertexId])
				{
					vertexValue_temp_host[0][vertexId] = msg;
				}
			} 	

			// Second, Reduce
			omp_parallel_for(int vertexId = noZeroOutDegreeNum; vertexId < vertexNum; vertexId++)
			{
				vertex_data_type msg = vertexValue_temp_host[0][vertexId];
				if (msg < vertexValue[vertexId])
				{
					vertexValue[vertexId] = msg;
				}
			}
		}
		else if (useDeviceNum == 1)
		{

			CUDA_CHECK(
				D2H(vertexValue_temp_host[0] + noZeroOutDegreeNum, vertexValue_temp_device[0] + noZeroOutDegreeNum, (zeroOutDegreeNum))
			);

			omp_parallel_for(int vertexId = noZeroOutDegreeNum; vertexId < vertexNum; vertexId++)
			{
				vertex_data_type msg = vertexValue_temp_host[0][vertexId];
				if (msg < vertexValue[vertexId])
				{
					vertexValue[vertexId] = msg;
				}
			}
		}

		else
		{	
			assert_msg(false, "Reduce Error");
		}
	}


	void opt_h2d()
	{
		#pragma omp parallel for num_threads(useDeviceNum)
		for(int deviceId = 0; deviceId < useDeviceNum; deviceId++ )
		{
			if(deviceId == 0)
			{
				CUDA_CHECK(cudaSetDevice(deviceId));
				CUDA_CHECK(cudaMemcpy(vertexValue_temp_device[deviceId], vertexValue, 
				(noZeroOutDegreeNum /2) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));
			}
			else
			{
				CUDA_CHECK(cudaSetDevice(deviceId));
				CUDA_CHECK(cudaMemcpy(vertexValue_temp_device[deviceId] + (noZeroOutDegreeNum /2), vertexValue + (noZeroOutDegreeNum /2), 
				(noZeroOutDegreeNum - noZeroOutDegreeNum/2) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));
			}				
		}

		#pragma omp parallel for num_threads(useDeviceNum)
		for(int deviceId = 0; deviceId < useDeviceNum; deviceId++ )
		{
			if(deviceId == 1)
			{
				CUDA_CHECK(cudaSetDevice(deviceId));
				CUDA_CHECK(cudaMemcpy(vertexValue_temp_device[deviceId], vertexValue, 
				(noZeroOutDegreeNum /2) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));
			}
			else
			{
				CUDA_CHECK(cudaSetDevice(deviceId));
				CUDA_CHECK(cudaMemcpy(vertexValue_temp_device[deviceId] + (noZeroOutDegreeNum /2), vertexValue + (noZeroOutDegreeNum /2), 
				(noZeroOutDegreeNum - noZeroOutDegreeNum/2) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));
			}		
		}	

		

	}


public:

	// TODO
	~CGgraphEngine()
	{
		for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			cudaStreamDestroy(stream_device_vec[deviceId]);
		}
		
	}

};// end of Class [CGgraphEngine]