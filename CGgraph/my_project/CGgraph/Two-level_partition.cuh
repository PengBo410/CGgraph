#pragma once


#define TWO_LEVEL_DEBUG
#define TWO_LEVEL_CHECK

#define SEGMENT_SIZE 1024 // 16 * 64
#define SEGMENT_WORDNUM (SEGMENT_SIZE/WORDSIZE) 

#define BUCKET_NUM 4

#define GPU_RESERVED_MEMORY 1024 * 1024 * 1024 

#define MORE_BALANCE_PARTITION 

#include "taskSteal.hpp"


class TwoLevelPartition
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

	//dense_bitset
	dense_bitset bitmap_host;

	Algorithm_type algorithm;

	struct Segment_type{
		uint32_t seg_id = 0;
		count_type seg_vertices = 0;
		countl_type seg_edges = 0;
		double seg_degree = 0.0;
		vertex_id_type seg_nbrMin = std::numeric_limits<vertex_id_type>::max();
		vertex_id_type seg_nbrMax = 0; 
		int seg_owner = -1; // -1 为Host

		//sub_segemnt; //TODO
	};

	// Segment Vector
	std::vector<Segment_type> seg_vec;
	uint64_t segmentNum;

	//Bucket Vector
	std::vector<std::vector<uint32_t>> bucket_vec; // Hub Bucket + Normal Buckets 
	uint64_t* bucketNum_array;




	// Device
	//device
	GPUInfo* gpuInfo;
	int maxDeviceNum;
	int useDeviceNum;

	std::vector<uint32_t> segmentNum_device_vec; // The number of segment in each device
	std::vector<dense_bitset> segmentId_device_vec; //  The id of segement in each device

	std::vector<count_type> vertexNum_device_vec; //device vertices
	std::vector<countl_type> edgeNum_device_vec; //device edges
	std::vector<countl_type*> csrOffset_device_vec; //device csr_offset
	std::vector<vertex_id_type*> csrDest_device_vec; //device csr_dest
	std::vector<edge_data_type*> csrWeight_device_vec; //device csr_weight


	struct Partition_type{

		uint64_t segmentNum;
		bool canHoldAllSegment = true;
		std::vector<Segment_type> seg_vec;

		std::vector<uint32_t> segmentNum_device_vec; // The number of segment in each device
		std::vector<dense_bitset> segmentId_device_vec; //  The id of segement in each device

		std::vector<count_type> vertexNum_device_vec; 
		std::vector<countl_type> edgeNum_device_vec; 

		std::vector<countl_type*> csrOffset_device_vec; 
		std::vector<vertex_id_type*> csrDest_device_vec; 
		std::vector<edge_data_type*> csrWeight_device_vec; 
	};

	bool canHoldAllSegment = true;



public:

	/* **************************************************************************
	 *                                [Construct]
	 * @param [const CSR_Result_type& csrResult]    CSR
	 * @param [const count_typezeroOutDegreeNum_]   Non-zero vertices
	 * @param [const size_t deviceNum_]             GPU Num
	 * **************************************************************************/
	TwoLevelPartition(const CSR_Result_type& csrResult, const count_type zeroOutDegreeNum_, const size_t deviceNum_, Algorithm_type algorithm_):
		vertexNum(0),
		edgeNum(0),
		zeroOutDegreeNum(0),
		noZeroOutDegreeNum(0),
		maxDeviceNum(0),
		useDeviceNum(0)
	{
		// Get Graph CSR
		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;

		useDeviceNum = deviceNum_;
		zeroOutDegreeNum = zeroOutDegreeNum_;
		noZeroOutDegreeNum = vertexNum - zeroOutDegreeNum;
		Msg_info("zeroOutDegreeNum = %zu, noZeroOutDegreeNum = %zu, useDeviceNum = %d", 
			static_cast<uint64_t>(zeroOutDegreeNum), static_cast<uint64_t>(noZeroOutDegreeNum), useDeviceNum);

		algorithm = algorithm_;

		//> Host - Init the basic info
		timer constructTime;
		initGraph();
		Msg_info("Init-Host: Used time: %.2lf (ms)", constructTime.get_time_ms());

		//> Host -Generate outDegree
		constructTime.start();
		get_outDegree();
		Msg_info("Build-outDegree: Used time: %.2lf (ms)", constructTime.get_time_ms());

		//> Host - Segment Info
		constructTime.start();
		get_SegmentInfo();
		Msg_info("Get-SegmentInfo: Used time: %.2lf (ms)", constructTime.get_time_ms());

		//> Host - Buckets
		constructTime.start();
		partition_buckets();
		Msg_info("Partition-Bucket: Used time: %.2lf (ms)", constructTime.get_time_ms());

		//> Host - Partition Segments To GPUs
		constructTime.start();
		partition_GPU();
		Msg_info("Partition-GPU: Used time: %.2lf (ms)", constructTime.get_time_ms());

		//> Host - Build Device CSR
		constructTime.start();
		build_deviceCSR();
		Msg_info("Build-DeviceCSR: Used time: %.2lf (ms)", constructTime.get_time_ms());


	}




private:

	/* **********************************************************
	 * Func: Host Function , Init Graph
	 * **********************************************************/
	void initGraph()
	{
		if (vertexNum >= std::numeric_limits<count_type>::max()){assert_msg(false, "vertexNum >= count_type:max()");}		
		if (edgeNum >= std::numeric_limits<countl_type>::max()){assert_msg(false, "vertexNum >= countl_type:max()");}
			
		assert_msg((WORD_MOD(SEGMENT_SIZE) == 0), "SEGMENT_SIZE must be an integer multiple of WORD");


		//Init outDegree
		outDegree = new offset_type[noZeroOutDegreeNum];

		bitmap_host.resize(noZeroOutDegreeNum);

		//device
		gpuInfo = new GPUInfo();
		int _maxDeviceNum = gpuInfo->getDeviceNum();
		assert_msg((useDeviceNum <= _maxDeviceNum),
		 "(useDeviceNum > _maxDeviceNum), useDeviceNum = %d, _maxDeviceNum = %d", useDeviceNum, _maxDeviceNum);
		maxDeviceNum = _maxDeviceNum;

		vertexNum_device_vec.resize(useDeviceNum);
		edgeNum_device_vec.resize(useDeviceNum);
		csrOffset_device_vec.resize(useDeviceNum);
		csrDest_device_vec.resize(useDeviceNum);
		csrWeight_device_vec.resize(useDeviceNum);
	} // end of func [initGraph()]


	/* **********************************************************
	 * Func: Host Function, Get outDegree
	 * **********************************************************/
	void get_outDegree()
	{
		omp_parallel_for(vertex_id_type vertexId = 0; vertexId < noZeroOutDegreeNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}
	}


	/* **********************************************************
	 * Func: Host Func, Get Segment Info (Level-1)
	 * **********************************************************/
	void get_SegmentInfo()
	{
		uint64_t segmentNum_ = (noZeroOutDegreeNum + SEGMENT_SIZE - 1) / SEGMENT_SIZE;	
		assert_msg((std::numeric_limits<uint32_t>::max() >= segmentNum_), "segmentNum out of the <uint32_t>");
		Msg_info("Segment Num = (%zu)", static_cast<uint64_t>(segmentNum_));

		segmentNum = segmentNum_;
		seg_vec.resize(segmentNum);

		//> Each Segment Info
		timer getSegmentTime;
		omp_parallel_for(uint64_t segmentId = 0; segmentId < segmentNum; segmentId ++)
		{
			vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
			vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE;
			if(endVertexId > noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;

			// Compute seg_edges, seg_degree, seg_nbrMin, seg_nbrMax of Segment
			uint64_t segmentEdges = 0;
			vertex_id_type segmentNbrMin = std::numeric_limits<vertex_id_type>::max();
			vertex_id_type segmentNbrMax = 0;

			for (uint64_t vertexId = startVertexId; vertexId < endVertexId; vertexId++)
			{
				size_t startNbrId = csr_offset[vertexId];
				size_t endNbrId = csr_offset[vertexId + 1];

				vertex_id_type vertexNbrMin = csr_dest[startNbrId];
				vertex_id_type vertexNbrMax = csr_dest[endNbrId - 1];

				if (vertexNbrMin < segmentNbrMin) segmentNbrMin = vertexNbrMin;
				if (vertexNbrMax > segmentNbrMax) vertexNbrMax = segmentNbrMax;

				segmentEdges += outDegree[vertexId];
			}

			Segment_type seg;
			seg.seg_id = static_cast<uint32_t>(segmentId);
			seg.seg_vertices = static_cast<count_type>(endVertexId - startVertexId);
			seg.seg_edges = static_cast<countl_type>(segmentEdges);
			seg.seg_degree = static_cast<double>(seg.seg_edges / seg.seg_vertices);
			seg.seg_nbrMin = segmentNbrMin;
			seg.seg_nbrMax = segmentNbrMax;

			seg_vec[segmentId] = seg;

		}// end of [omp_parallel_for]

	}// end of func [get_SegmentInfo()]


	/* **********************************************************
	 * Func: Host Func, Partition Bucket (Level-2)
	 * **********************************************************/
	void partition_buckets()
	{
		// First, Sort The Segment According To Their Degree
		sort(seg_vec.begin(), seg_vec.end(),
			[&](const Segment_type& a, Segment_type& b) -> bool
			{
				if(a.seg_degree < b.seg_degree) return true;
				else if(a.seg_degree == b.seg_degree)
				{
					if (a.seg_id < b.seg_id) return true;
					else                     return false;
				}
				else                         return false;
			}		
		);

		//We check the seg_vec
#ifdef TWO_LEVEL_CHECK
		uint64_t checkTotalEdges = 0;
#pragma omp parallel for reduction(+: checkTotalEdges)
		for (uint64_t segmentId = 0; segmentId < seg_vec.size(); segmentId++)
		{
			checkTotalEdges += seg_vec[segmentId].seg_edges;
		}
		assert_msg((checkTotalEdges == edgeNum), "checkTotalEdges = %zu, edgeNum = %zu", checkTotalEdges, static_cast<uint64_t>(edgeNum));
		Msg_finish("Segement_vec Finished EdgeNum Check");
#endif		

		// Second, Partition The Segmnet To The Different Buckets		
		uint64_t bucketOne_index = lower_bound<uint64_t, double>(segmentNum, static_cast<double>(HALFWARP + 1),
			[&](uint64_t binSearch_mid)
			{
				return seg_vec[binSearch_mid].seg_degree;
			}
		);

		uint64_t bucketTwo_index = lower_bound<uint64_t, double>(segmentNum, static_cast<double>(WARPSIZE + 1),
			[&](uint64_t binSearch_mid)
			{
				return seg_vec[binSearch_mid].seg_degree;
			}
		);

		uint64_t bucketThree_index = lower_bound<uint64_t, double>(segmentNum, static_cast<double>(BLOCKSIZE + 1),
			[&](uint64_t binSearch_mid)
			{
				return seg_vec[binSearch_mid].seg_degree;
			}
		);

		bucketNum_array = new uint64_t[BUCKET_NUM] + 1;
		bucketNum_array[0] = 0;
		bucketNum_array[1] = bucketOne_index;
		bucketNum_array[2] = bucketTwo_index;
		bucketNum_array[3] = bucketThree_index;
		bucketNum_array[4] = segmentNum;
		Msg_info("Bucket-1   has (%6zu) Segments \n" 
		"         Bucket-2   has (%6zu) Segments \n"
		"         Bucket-3   has (%6zu) Segments \n" 
		"         Bucket-Hub has (%6zu) Segments",
			bucketNum_array[1], 
			bucketNum_array[2] - bucketNum_array[1],
			bucketNum_array[3] - bucketNum_array[2],
			bucketNum_array[4] - bucketNum_array[3]
		);

		bucket_vec.resize(BUCKET_NUM);
		for (uint64_t i = 0; i < BUCKET_NUM; i++)
		{
			bucket_vec[i].resize(bucketNum_array[i+1] - bucketNum_array[i]);
			for (uint64_t segmentId = bucketNum_array[i]; segmentId < bucketNum_array[i+1]; segmentId++)
			{
				bucket_vec[i][segmentId - bucketNum_array[i]] = seg_vec[segmentId].seg_id;
			}
		}
	}



	/* **********************************************************
	 * Func: Host Func, Partition GPU 
	 *       Partition Segment For Each GPU
	 * **********************************************************/
	void partition_GPU()
	{
		
		std::vector<int64_t> memoryCapacity_vec(maxDeviceNum, 0);
		for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			memoryCapacity_vec[deviceId] = static_cast<int64_t>(gpuInfo->getGlobalMem_byte()[deviceId]);
		}
		

		for (int deviceId = 0; deviceId < maxDeviceNum; deviceId++)
		{
			Msg_info("The [%d] GPU's Global Memory Capacity: %.2lf (GB)", deviceId, (double)GB(memoryCapacity_vec[deviceId]));
		}

		
		std::vector<uint64_t> memoryUsed_vec(maxDeviceNum, 0); // Used Memory Size
		uint64_t memoryCapacity_total = 0;// In current Node, all GPUs avaliable memory
		for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			memoryUsed_vec[deviceId] += (vertexNum * sizeof(vertex_data_type));//vertexValue
			memoryUsed_vec[deviceId] += ((noZeroOutDegreeNum + 1) * sizeof(countl_type));//csr_offset
			memoryUsed_vec[deviceId] += GPU_RESERVED_MEMORY; // Reserved
			memoryCapacity_vec[deviceId] -= memoryUsed_vec[deviceId];
			Msg_info("The [%d] GPU's Result Memory After Remove [vertexValue], [csr_offset] and [GPU_RESERVED_MEMORY]: %.2lf (GB)",
				 deviceId, (double)GB(memoryCapacity_vec[deviceId]));
			memoryCapacity_total += memoryCapacity_vec[deviceId];
		}
		Msg_info("In Current Node, All GPUs Avaliable Memory: %.2lf (GB)", (double)GB(memoryCapacity_total));


		
		segmentNum_device_vec.resize(useDeviceNum, 0);
		segmentId_device_vec.resize(useDeviceNum);
		for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			segmentId_device_vec[deviceId].resize(segmentNum);
			segmentId_device_vec[deviceId].clear();
		}
		
		timer  partition_time;
		uint64_t reachedBucketId = 0; 	
		for (uint64_t bucketId = 0; bucketId < BUCKET_NUM; bucketId++)
		{
			reachedBucketId = bucketId;
			uint32_t segmentId = 0;
//#pragma omp parallel for   //[break] can not used in the omp parallel for
			for (; segmentId < bucket_vec[bucketId].size(); segmentId++)
			{
				uint32_t segment_group = 0;
#ifdef MORE_BALANCE_PARTITION
				// Build the more Balance vertexNum_device and edgeNum_device among the GPUs
				segment_group = segmentId / useDeviceNum;
#endif
				int targetDevice = (segmentId + segment_group) % useDeviceNum;
				

				uint32_t segmentIndex = bucket_vec[bucketId][segmentId];
				Segment_type& seg = seg_vec[bucketNum_array[bucketId] + segmentId];
				assert_msg((segmentIndex == seg.seg_id), "segmentIndex = %u, seg.seg_id = %u", segmentIndex, seg.seg_id);
				uint64_t occupancyBytes = 0;
				if(algorithm == Algorithm_type::SSSP){
					occupancyBytes = seg.seg_edges * (sizeof(vertex_id_type) + sizeof(edge_data_type));
				} else {
					occupancyBytes = seg.seg_edges * sizeof(vertex_id_type);
				}

				memoryCapacity_vec[targetDevice] -= occupancyBytes;

				
				if(memoryCapacity_vec[targetDevice] <= 0)
				{
					Msg_info("The GPUs Can Store Part Graph, ReachedBucketId = [%zu], Current Segment Index: (%u)",
						bucketId, segmentId);
					memoryCapacity_vec[targetDevice] += occupancyBytes;
					canHoldAllSegment = false;
					
					break;
				}
				
				seg.seg_owner = targetDevice;
				segmentId_device_vec[targetDevice].set_bit(segmentIndex);
				segmentNum_device_vec[targetDevice] ++;
			}	
			if(segmentId != bucket_vec[bucketId].size())
			{
				for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
				{
					Msg_major("Current Node's GPUs Can not Hold The Entire Graph, ReachedBucketId = [%zu], ReachedSegmentId = [%u], Current Bucket Total Segment Num: (%zu)\n"
					"         The [%d] GPU Result Memory Capacity: (%.2lf) (GB)",
					reachedBucketId, segmentId, bucket_vec[bucketId].size(), deviceId, (double)GB(memoryCapacity_vec[deviceId]));
				}
				canHoldAllSegment = false;
				break;
			}
		}

		if(canHoldAllSegment)
		{
			for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
			{
				Msg_info("Current Node's GPUs Can Hold All Segment.\n"
				"         The [%d] GPU Result Memory Capacity: (%.2lf) (GB)",
					deviceId, (double)GB(memoryCapacity_vec[deviceId]));
			}		
		}
		Msg_finish("Partition The Segments To GPUs Finished, Used time: %.2lf (ms)", partition_time.get_time_ms());


		// Check
#ifdef TWO_LEVEL_CHECK
		uint64_t check_segments = 0;
		for (uint64_t deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			assert_msg(segmentId_device_vec[deviceId].popcount() == segmentNum_device_vec[deviceId], "dense_bitset != segmentNum_device_vec");
			check_segments += segmentNum_device_vec[deviceId];
		}
		if(canHoldAllSegment)
		{
			assert_msg((check_segments == segmentNum), "(check_segments != segmentNum), check_segments = %zu, segmentNum = %zu",
				static_cast<uint64_t>(check_segments), static_cast<uint64_t>(segmentNum));
		}
		

		Msg_finish("Partition The Segments To GPUs Checked Finished");
#endif
	}




	/* **********************************************************
	 * Func: Host Func, Build The CSR For CSR
	 * **********************************************************/
	void build_deviceCSR()
	{
		// To easy access, sort the seg_vec by the seg_id
		sort(seg_vec.begin(), seg_vec.end(),
			[&](const Segment_type& a, Segment_type& b) -> bool
			{
				if(a.seg_id < b.seg_id) return true;
				else                    return false;
			}		
		);

#ifdef TWO_LEVEL_CHECK
			dense_bitset vertex_bitmap;
			vertex_bitmap.resize(noZeroOutDegreeNum);
			vertex_bitmap.clear();
#endif


		timer buildDeviceCSR_time;

		// Get each GPU's CSR
		for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			CUDA_CHECK(cudaSetDevice(deviceId));

			//> First, Get The vertexNum_device and edgeNum_device of each GPU
			for (uint64_t segmentId = 0; segmentId < segmentId_device_vec[deviceId].size(); segmentId++)
			{
				if( segmentId_device_vec[deviceId].get(segmentId))
				{
					vertexNum_device_vec[deviceId] += seg_vec[segmentId].seg_vertices;
					edgeNum_device_vec[deviceId] += seg_vec[segmentId].seg_edges;
				}
			}
			Msg_info("The [%d] GPUs  segmentNum_device: (%zu), vertexNum_device: (%zu), edgeNum_device: (%zu)", deviceId,
				static_cast<uint64_t>(segmentNum_device_vec[deviceId]),
				static_cast<uint64_t>(vertexNum_device_vec[deviceId]), static_cast<uint64_t>(edgeNum_device_vec[deviceId]));

			//> Second, Malloc The Temp Memory
			countl_type* csr_offset_host_ = nullptr;
			vertex_id_type* csr_dest_host_= nullptr;
			edge_data_type* csr_weight_host_ = nullptr;

			countl_type* csr_offset_device_ = nullptr;
			vertex_id_type* csr_dest_device_ = nullptr;
			edge_data_type* csr_weight_device_ = nullptr;

			// csr_offset
			CUDA_CHECK(MALLOC_HOST(&csr_offset_host_, (noZeroOutDegreeNum + 1)));
			CUDA_CHECK(MALLOC_DEVICE(&csr_offset_device_,  (noZeroOutDegreeNum + 1)));
			MEMSET_HOST(csr_offset_host_, (noZeroOutDegreeNum + 1));
			CUDA_CHECK(MEMSET_DEVICE(csr_offset_device_, (noZeroOutDegreeNum + 1)));

			//csr_dest
			CUDA_CHECK(MALLOC_HOST(&csr_dest_host_, edgeNum_device_vec[deviceId]));
			CUDA_CHECK(MALLOC_DEVICE(&csr_dest_device_,  edgeNum_device_vec[deviceId]));
			MEMSET_HOST(csr_dest_host_, edgeNum_device_vec[deviceId]);
			CUDA_CHECK(MEMSET_DEVICE(csr_dest_device_, edgeNum_device_vec[deviceId]));

			//csr_weight
			if (algorithm == Algorithm_type::SSSP){
				CUDA_CHECK(MALLOC_HOST(&csr_weight_host_, edgeNum_device_vec[deviceId]));
				CUDA_CHECK(MALLOC_DEVICE(&csr_weight_device_,  edgeNum_device_vec[deviceId]));
				MEMSET_HOST(csr_weight_host_, edgeNum_device_vec[deviceId]);
				CUDA_CHECK(MEMSET_DEVICE(csr_weight_device_, edgeNum_device_vec[deviceId]));
			}

			Msg_finish("The [%d] GPU Malloc Finished, Used time: %.2lf (ms)", deviceId, buildDeviceCSR_time.get_time_ms());

			//> Third, Fill The Array
			buildDeviceCSR_time.start();

			TaskSteal* taskSteal = new TaskSteal();// compute task
			uint64_t WORKLOAD_CHUNK = 64;
			taskSteal->twoStage_taskSteal<uint64_t, uint64_t>(
				static_cast<uint64_t>(segmentId_device_vec[deviceId].size()), // total_workload
				[&](uint64_t& current, uint64_t& local_workloads)
				{
					uint64_t end = current + WORKLOAD_CHUNK;
					uint64_t length = WORKLOAD_CHUNK;
					if (end >= segmentId_device_vec[deviceId].size()) length = segmentId_device_vec[deviceId].size() - current;

					for (uint64_t segmentId = current; segmentId < (current + length); segmentId++)
					{
						if( segmentId_device_vec[deviceId].get(segmentId))
						{
							vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
							vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE ;
							if (endVertexId >= noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;
#ifdef TWO_LEVEL_CHECK
							Segment_type seg = seg_vec[segmentId];
							assert_msg((endVertexId - startVertexId) == seg.seg_vertices, "Build [csr_offset_host_] Error");
#endif

							for (vertex_id_type vertexId = startVertexId; vertexId < endVertexId; vertexId++)
							{
								csr_offset_host_[vertexId + 1] = outDegree[vertexId];
							}
						}
					}
				},
				WORKLOAD_CHUNK
			);

			// csr_offset
			for (uint64_t vertexId = 1; vertexId <= noZeroOutDegreeNum; vertexId++)
			{
				csr_offset_host_[vertexId] += csr_offset_host_[vertexId - 1];
			}

			assert_msg((csr_offset_host_[noZeroOutDegreeNum] == edgeNum_device_vec[deviceId]), 
				"(csr_offset_host_[noZeroOutDegreeNum] != edgeNum_device_vec[deviceId]), csr_offset_host_[noZeroOutDegreeNum] = %zu, edgeNum_device_vec[deviceId] = %zu",
				static_cast<uint64_t>(csr_offset_host_[noZeroOutDegreeNum]), static_cast<uint64_t>(edgeNum_device_vec[deviceId]));

			//csr_dest and csr_weight
			taskSteal->twoStage_taskSteal<uint64_t, uint64_t>(
				static_cast<uint64_t>(segmentId_device_vec[deviceId].size()), // total_workload
				[&](uint64_t& current, uint64_t& local_workloads)
				{
					uint64_t end = current + WORKLOAD_CHUNK;
					uint64_t length = WORKLOAD_CHUNK;
					if (end >= segmentId_device_vec[deviceId].size()) length = segmentId_device_vec[deviceId].size() - current;

					for (uint64_t segmentId = current; segmentId < (current + length); segmentId++)
					{
						if( segmentId_device_vec[deviceId].get(segmentId))
						{
							Segment_type seg = seg_vec[segmentId];

							vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
							vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE ;
							if (endVertexId >= noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;

							for (vertex_id_type vertexId = startVertexId; vertexId < endVertexId; vertexId++)
							{
								uint64_t offset = csr_offset_host_[vertexId];
								uint64_t nbrLength = csr_offset_host_[vertexId + 1] - csr_offset_host_[vertexId];

								uint64_t offset_old = csr_offset[vertexId];
								uint64_t nbrLength_old = csr_offset[vertexId + 1] - csr_offset[vertexId];
								assert_msg((nbrLength == nbrLength_old),
									"(nbrLength != nbrLength_old), nbrLength = %zu, nbrLength_old = %zu", nbrLength, nbrLength_old);

								memcpy(csr_dest_host_ + offset, csr_dest + offset_old, nbrLength * sizeof(vertex_id_type));
								if (algorithm == Algorithm_type::SSSP){
									memcpy(csr_weight_host_ + offset, csr_weight + offset_old, nbrLength * sizeof(edge_data_type));
								}								
							}
						}
					}
				},
				WORKLOAD_CHUNK
			);

			Msg_finish("The [%d] GPU CSR Build Finished, Used time: %.2lf (ms)", deviceId, buildDeviceCSR_time.get_time_ms());

			// Check The GPU's CSR
#ifdef TWO_LEVEL_CHECK
			omp_parallel_for(vertex_id_type segmentId = 0; segmentId < segmentNum; segmentId ++)
			{			
				if(segmentId_device_vec[deviceId].get(segmentId)){

					vertex_id_type startVertexId = segmentId * SEGMENT_SIZE;
					vertex_id_type endVertexId = (segmentId + 1) * SEGMENT_SIZE ;
					if (endVertexId >= noZeroOutDegreeNum) endVertexId = noZeroOutDegreeNum;

					for (vertex_id_type vertexId = startVertexId; vertexId < endVertexId; vertexId++)
					{
						countl_type nbr_first = csr_offset_host_[vertexId];
						countl_type nbr_end = csr_offset_host_[vertexId + 1];

						countl_type nbr_first_old = csr_offset[vertexId];
						countl_type nbr_end_old = csr_offset[vertexId + 1];

						assert_msg((nbr_end - nbr_first) == (nbr_end_old - nbr_first_old), 
							"VertexId [%zu] Nbr Size Error: error_nbrSize = %zu, right_nbrSize = %zu, outDegree = %zu",
							static_cast<uint64_t>(vertexId), static_cast<uint64_t>(nbr_end - nbr_first), 
							static_cast<uint64_t>(nbr_end_old - nbr_first_old), static_cast<uint64_t>(outDegree[vertexId]));

						for (countl_type vertex_id = nbr_first, vertex_id_old = nbr_first_old; 
							vertex_id < nbr_end && vertex_id_old < nbr_end_old; 
							vertex_id++, vertex_id_old++)
						{
							assert_msg((csr_dest_host_[vertex_id] == csr_dest[vertex_id_old]), 
								"VertexId [%zu] Nbr Error: csr_dest_host_ = %zu, csr_dest = %zu",
								static_cast<uint64_t>(vertexId),
								static_cast<uint64_t>(csr_dest_host_[vertex_id]),
								static_cast<uint64_t>(csr_dest[vertex_id_old]));
						}
						bitmap_host.set_bit(vertexId);
					}

											
				}				
			}
			Msg_finish("The [%d] GPU Finished GPU's CSR Check !", deviceId);
#endif

			// H2D
			buildDeviceCSR_time.start();
			H2D(csr_offset_device_, csr_offset_host_, (noZeroOutDegreeNum + 1));
			H2D(csr_dest_device_, csr_dest_host_, edgeNum_device_vec[deviceId]);
			if (algorithm == Algorithm_type::SSSP){
				H2D(csr_weight_device_, csr_weight_host_, edgeNum_device_vec[deviceId]);
			}
			Msg_finish("The [%d] GPU CSR Has Transfer To Device, Used time: %.2lf (ms)", deviceId, buildDeviceCSR_time.get_time_ms());

			// Add the pointer to vector
			csrOffset_device_vec[deviceId] = csr_offset_device_;
			csrDest_device_vec[deviceId] = csr_dest_device_;
			if (algorithm == Algorithm_type::SSSP){
				csrWeight_device_vec[deviceId] = csr_weight_device_;
			}

			// Free the host
			FREE_HOST(csr_offset_host_);
			FREE_HOST(csr_dest_host_);
			if (algorithm == Algorithm_type::SSSP){
				FREE_HOST(csr_weight_host_);
			}
		}// end of for each device

#ifdef TWO_LEVEL_CHECK
		count_type total_vertice_device = 0;
		for (int deviceId = 0; deviceId < useDeviceNum; deviceId++)
		{
			total_vertice_device += vertexNum_device_vec[deviceId];
		}
		assert_msg((total_vertice_device == bitmap_host.popcount()), 
			"A Total Of GPU Vertices Error, total_vertice_device = %zu, bitmap_host.popcount() = %zu",
			static_cast<uint64_t>(total_vertice_device), static_cast<uint64_t>(bitmap_host.popcount()));
		//Msg_finish("A Total Of (%zu) GPU Vertices Passed The Check", static_cast<uint64_t>(bitmap_host.popcount()));
#endif

	}// end of function [build_deviceCSR()]



    
public:

	Partition_type getPartitionResult()
	{
		Partition_type partitionResult;

		partitionResult.segmentNum = segmentNum;
		partitionResult.seg_vec = seg_vec;
		partitionResult.segmentNum_device_vec = segmentNum_device_vec;
		partitionResult.segmentId_device_vec = segmentId_device_vec;

		partitionResult.vertexNum_device_vec = vertexNum_device_vec;
		partitionResult.edgeNum_device_vec = edgeNum_device_vec;

		partitionResult.csrOffset_device_vec = csrOffset_device_vec;
		partitionResult.csrDest_device_vec = csrDest_device_vec;
		partitionResult.csrWeight_device_vec = csrWeight_device_vec;

		partitionResult.canHoldAllSegment = canHoldAllSegment;

		return partitionResult;
	}



};// end of class [TwoLevelPartition]






















// 		// Third, Begin To Partition
// 		timer pTime;

// 		int reachedBucketId = 0;// The bucketId when reaching all GPUs memory capacity
// 		uint64_t reachedDataSize = 0; // The data size when reaching all GPUs memory capacity
// 		for(int bucketId = 0; bucketId < BUCKET_NUM; bucketId++)
// 		{
// 			uint64_t current_bucket_bytes = 0;
// 			uint64_t current_bucket_edges = 0;

// #pragma omp parallel for reduction(+: current_bucket_edges)
// 			for (uint64_t segmentId = 0; segmentId < bucket_vec[bucketId].size(); segmentId++)
// 			{
// 				uint32_t segmentIndex = bucket_vec[bucketId][segmentId];
// 				Segment_type seg = seg_vec[segmentIndex];
// 				current_bucket_edges += seg.seg_edges;
// 			}

// 			if(algorithm == Algorithm_type::SSSP)
// 			{
// 				current_bucket_bytes = (current_bucket_edges * (sizeof(vertex_id_type) + sizeof(edge_data_type))); // edge + weight
// 			}
// 			else
// 			{
// 				current_bucket_bytes = (current_bucket_edges * sizeof(vertex_id_type)); // edge
// 			}
			
// 			Msg_info("The Bucket-%d, Edge: %zu, Usage: %zu (Bytes) - %.2lf (GB)", 
// 				bucketId, current_bucket_edges, current_bucket_bytes, (double) GB(current_bucket_bytes));

// 			if(current_bucket_bytes >= memoryCapacity_total) reachedBucketId = bucketId;
// 			else
// 			{
// 				memoryCapacity_total -= current_bucket_bytes;
// 				reachedBucketId = bucketId;
// 			}  
// 		}

// 		//> [Entire Graph]
// 		if(reachedBucketId == (BUCKET_NUM-1)) 
// 		{
// 			Msg_info("The GPUs Can Store Entire Graph, ReachedBucketId = [%d], Result Memory Capacity: (%.2lf) (GB)",
// 				reachedBucketId, (double)GB(memoryCapacity_total));

// 			//> partition the segment in the Buckets in a round-robin fashion
// 			segmentNum_device_vec.resize(useDeviceNum, 0);
// 			for (uint64_t bucketId = 0; bucketId < BUCKET_NUM; bucketId++)
// 			{
// 				for (uint32_t segmentId = 0; segmentId < bucket_vec[bucketId].size(); segmentId++)
// 				{
// 					int targetDevice = segmentId % useDeviceNum;
					
// 				}
				
// 			}
			

// 		}
// 		//> [Part Graph]
// 		else
// 		{
// 			Msg_info("The GPUs Can Store Part Graph, ReachedBucketId = [%d], Result Memory Capacity: (%.2lf) (GB)", 
// 				reachedBucketId, (double)GB(memoryCapacity_total));
// 			// Partition the bucket to match the result GPU memory capacity
// 			uint64_t canProcessEdges = 0; // The Result Memory Can Process The Edges
// 			if(algorithm == Algorithm_type::SSSP)
// 			{
// 				canProcessEdges = memoryCapacity_total / (sizeof(vertex_id_type) + sizeof(edge_data_type));// edge + weight
// 			}
// 			else
// 			{
// 				canProcessEdges = memoryCapacity_total / (sizeof(vertex_id_type));// edge
// 			}
// 			Msg_info("The Result GPU Memory Can Process The Edge Num: %zu", canProcessEdges);

// 			// In Current Bucket, Can Process Segment Num
// 			uint64_t fillSegmentNum = 0; // Number of segments that can still be filled in
// 			uint64_t totalEdges_fillSegment = 0; // Number of edges of filled Segment
// 			for (uint64_t segmentId = 0; segmentId < bucket_vec[reachedBucketId + 1].size(); segmentId++)
// 			{
// 				uint32_t segmentIndex = bucket_vec[reachedBucketId + 1][segmentId];
// 				Segment_type seg = seg_vec[segmentIndex];
				
// 				if(totalEdges_fillSegment > canProcessEdges)
// 				{
// 					fillSegmentNum = segmentId;
// 					Msg_info("Bucket-%d Provides (%zu) Segments (Including (%zu) Edges) For Result GPU Memory Capacity",
// 						(reachedBucketId + 1 + 1), fillSegmentNum, totalEdges_fillSegment);
// 					break;
// 				} 
// 				else
// 				{
// 					totalEdges_fillSegment += seg.seg_edges;
// 				}
// 			}
			
// 		}