#pragma once

#include "Basic/basic_include.cuh"
#include "device_algorithm.cuh"
#include <type_traits>



//[DEBUG]
#define GRAPHHYBIRD_DEBUG
#define TEMP_DEBUG
//#define COLLECT_ADAPTIVE_INFO
//#define GPU_MODEL_CHUNK


//[TASK]
#define TASK_MIX_CENTER

//[TASK OFFLOAD]
#define INDEX_TO_LEFT
#define MIDDLE_TO_LEFT


//[OFFLOAD]
#define RESERVE_CHUNK_VEC 10000                 // 这里定义的是chunkTask_vec的预申请大小
#define WORD_NUM_CHUNK (128)             // 一个chunk中包含的word数(默认: 2560个word, 即：163840个顶点一组)
#define USED_DEVICE_RATE 0.1                   //(worload的顶点数/总顶点数) >= USED_DEVICE_RATE 时启用Device辅助计算
#define OFFLOAD_TASK_RATE 0.30                  // 分配给Device的总任务数的比率
#define OUR_DOUBLE_MAX 100000000.0 
#define DEVICE_PERFER 15                       // device喜欢的degree





class GraphCooperation {

	// [Host used]
public:
	count_type vertexNum;
	countl_type edgeNum;
	count_type zeroOutDegreeNum;
	count_type noZeroOutDegreeNum;

	countl_type* csr_offset;
	vertex_id_type* csr_dest;
	edge_data_type* csr_weight;
	offset_type* outDegree;

	//NUMA的信息
	count_type socketNum;
	count_type threadPerSocket;
	count_type threadNum;
	count_type vertexPerSocket; // 按PAGESIZE对齐

	//NUMA数组
	vertex_id_type* numa_offset;// 长度为：socketNum + 1  -> 将本机的顶点进行了划分
	countl_type** csr_offset_numa;// [socketId][offset]
	vertex_id_type** csr_dest_numa;// [socketId][dest]
	edge_data_type** csr_weight_numa;// [socketId][weight]	
	count_type* vertexNum_numa;//每个numa的顶点数, 长度为：socketNum, 应该按照pageSize对齐
	countl_type* edgeNum_numa;//每个numa的边数, 长度为：socketNum
	count_type* zeroOffset_numa;//归零的偏移量, 长度为：socketNum
	countl_type* zeroDegree_numa;

	//TaskSteal
	TaskSteal* taskSteal;          // graph process used by host
	TaskSteal* taskSteal_align64;  // clear active
	TaskSteal* taskSteal_hostHelp; // set active used by host
	TaskSteal* taskSteal_totalTask;// compute task

	//VertexValue
	vertex_data_type* vertexValue;
	vertex_data_type* vertexValue_temp;// Offload的中间介质

	//Active
	dense_bitset active_in;
	dense_bitset active_out;
	DoubleBuffer<dense_bitset> active; //active_in与active_out最终封装到DoubleBuffer
	count_type activeNum_device = 0;  // Device 端激活的顶点数
	count_type activeNum_host = 0;    // Host   端激活的顶点数
	count_type activeNum = 0;         // Device + Host 端总共激活的顶点数

	// algorithm
	Algorithm_type algorithm;
	count_type ite = 0; // 迭代次数
	bool is_outDegreeThread = false;




	//[Device used]
public:
	countl_type* csr_offset_device;
	vertex_id_type* csr_dest_device;
	edge_data_type* csr_weight_device;
	vertex_data_type* vertexValue_device;

	// [0]: worklist_count; [1]: worklist_size; [2]: vertexNum; [3]: edgeNum;  [4]: noZeroOutDegreeNum; [5]: WORD_NUM_CHUNK: [6] GPU_MODEL_START
	count_type common_size = 7;
	offset_type* common;
	offset_type* common_device;

	int deviceId;  //要使用的Device的Id
	size_t nBlock;     //要使用的Device开启的blockNum





	//[Offload]:(基本单位是chunk, 而chunk都以word为基本单位)
public:
 
	struct ChunkTask_type {
		size_t chunkSize = 0;//记录当前的chunk包含的word的数量,主要为了对不齐的
		size_t vertices = 0; //chunk拥有的顶点数
		size_t edges = 0;    //chunk拥有的边数
		double score = OUR_DOUBLE_MAX;  //chunk的最终分数
	};
	std::vector<std::pair<size_t, ChunkTask_type>> chunkTask_vec; // [first： chunkId, second: chunkSetNum], 用于排序每个chunk的工作量
	//std::vector<size_t> chunkTotalTask_vec;                       // 记录chunk的总工作量
	size_t zeroChunk = 0;                                         // 无任何工作的chunk个数

	count_type* offload_offset;         //Host:   每个chunk的偏移量
	uint64_t* offload_data;             //Host:   每个chunk的active数据
	count_type* offload_offset_device;  //Device: 每个chunk的偏移量
	uint64_t* offload_data_device;      //Device: 每个chunk的active数据

	// 配合协助
	std::mutex wakeupDevice_mutex; // 用于唤醒Device
	size_t wakeupDevice = 0;
	std::mutex setActive_hostHelp_mutex; // Device To Host 的 vertexValue 是否需要Host帮忙(当Host-task先完成后,希望利用Host的多线程加快处理性能)
	int setActive_hostHelp = 0;          // 0 初始值; 1 表示需要host帮忙完成; 2 表示device正在完成，host无需接管;
	bool hybirdComplete = false;     // Hybird迭代是否完成
	bool usedDevice = false;         // 标志是否使用过Device

	count_type offload_wordNum = MAX_OFFFLOAD_CHUNK * WORD_NUM_CHUNK; //固定chunk
	count_type offload_offsetNum = MAX_OFFFLOAD_CHUNK; // 一个chunk至少需要一个offset
	count_type MAX_OFFFLOAD_CHUNK = RESERVE_CHUNK_VEC;  // 最大offload的chunk数 (我们用它进行预申请内存,因此不能超过) 
	size_t offload_chunkNum = 0; // 实际的offload的chunk数

	double alpha = static_cast<double>(OFFLOAD_TASK_RATE);
	
	//用于手动优化offloading
	size_t hand_offload_max = 400;
	std::vector<double> hand_offload_vec;


	// [Adaptive Info]
public:
	bool has_adaptive = false; // 是否启用了Adaptive

#ifdef COLLECT_ADAPTIVE_INFO
	double rate_cpu = 0.0;
	double rate_gpu = 0.0;
	double rate_pcie = 0.0;

	size_t adaptiveNum = 0;      // 统计的次数
	size_t MAX_ADAPTIVE = 1000;  // 最大的统计次数
	std::vector<Adaptive_info_type> cpu_adaptive_vec;
	std::vector<Adaptive_info_type> gpu_adaptive_vec;
	std::vector<Adaptive_info_type> pcie_adaptive_vec;
	std::vector<Adaptive_info_type> reduce_adaptive_vec;

	std::vector<double> chunkTotalScore_vec;	
#endif
	double totalScore = 0.0;
	

	std::string graphName = "";
	int begin = 0;
	int end = 0;






public:
	/* ======================================================================== *
	 *                                    【构造函数】
	 * ======================================================================== */
	GraphCooperation(const CSR_Result_type& csrResult, const count_type zeroOutDegreeNum_, Algorithm_type algorithm_,std::string graphName_, int deviceId_ = 0) :
		vertexNum(0),
		edgeNum(0),
		zeroOutDegreeNum(0),
		noZeroOutDegreeNum(0)
	{
		assert_msg((ThreadNum >= 2), "The machine requires at least 2 threads, current threadNum = %zu", ThreadNum);


		//获得原图数据的CSR
		vertexNum = csrResult.vertexNum;
		edgeNum = csrResult.edgeNum;
		csr_offset = csrResult.csr_offset;
		csr_dest = csrResult.csr_dest;
		csr_weight = csrResult.csr_weight;

		zeroOutDegreeNum = zeroOutDegreeNum_;
		noZeroOutDegreeNum = vertexNum - zeroOutDegreeNum;
		Msg_info("zeroOutDegreeNum = %u, noZeroOutDegreeNum = %u", zeroOutDegreeNum, noZeroOutDegreeNum);

		algorithm = algorithm_;
		graphName = graphName_;


		//Device
		deviceId = deviceId_;
		Msg_info("Device[%u]要处理的algorithm: (%s)", deviceId, getAlgName(algorithm).c_str());

		timer constructTime;
		//Host - 初始化基本信息
		initGraphNuma();
		Msg_info("Init-Host: 用时：%.2f (ms)", constructTime.get_time_ms());

		//Host - 根据NUMA平分顶点
		constructTime.start();
		partitionGraphByNuma();
		Msg_info("Partition-HostNuma: 用时：%.2f (ms)", constructTime.get_time_ms());

		//Host - 分配vertexValue和active
		constructTime.start();
		allocate_vertexValueAndActive();
		Msg_info("Allocate-Active: 用时：%.2f (ms)", constructTime.get_time_ms());

		//Host - 生成outDegree
		constructTime.start();
		get_outDegree();
		Msg_info("Build-outDegree: 用时：%.2f (ms)", constructTime.get_time_ms());

		//Device - cudaMalloc
		constructTime.start();
		allocate_device();
		Msg_info("Allocate-Device: 用时：%.2f (ms)", constructTime.get_time_ms());

		//Device - csrToDevice
		constructTime.start();
		graphToDevice();
		Msg_info("Host-To-Device: 用时：%.2f (ms)", constructTime.get_time_ms());


		//Host + Device: offload
		constructTime.start();
		if (offload_wordNum > active.in().arrlen)
		{
			offload_wordNum = active.in().arrlen;
			MAX_OFFFLOAD_CHUNK = (offload_wordNum + WORD_NUM_CHUNK - 1) / WORD_NUM_CHUNK;
			offload_wordNum = MAX_OFFFLOAD_CHUNK * WORD_NUM_CHUNK;
			offload_offsetNum = MAX_OFFFLOAD_CHUNK;
			Msg_info("MAX_OFFFLOAD_CHUNK 设置太大会造成内存浪费, 根据图数据更新为: MAX_OFFFLOAD_CHUNK = %u, offload_offsetNum = %u, offload_wordNum = %u",
				MAX_OFFFLOAD_CHUNK, offload_offsetNum, offload_wordNum);
		}
		allocate_offload();
		Msg_info("Allocate-Offload: 用时：%.2f (ms)", constructTime.get_time_ms());

		// hand opt
		hand_offload_vec.resize(hand_offload_max, static_cast<double>(OFFLOAD_TASK_RATE));
		if(graphName == "twitter2010")
		{
			begin = 3;
			end = 6;
			hand_offload_vec[3] = 0.68; hand_offload_vec[4] = 0.61; hand_offload_vec[5] = 0.7; hand_offload_vec[6] = 0.7;// 优化 twitter2010(24-thread)
		}
		else if(graphName == "friendster")
		{
			begin = 5;
			end = 12;
			hand_offload_vec[5] = 0.8; hand_offload_vec[6] = 0.82; hand_offload_vec[7] = 0.75; hand_offload_vec[8] = 0.73; hand_offload_vec[9] = 0.72;
			hand_offload_vec[10] = 0.72;  hand_offload_vec[11] = 0.72; hand_offload_vec[12] = 0.72;
		}
		
	}


	/* ======================================================================== *
	 *                            【设置Aadaptive的基本参考】
	 * ======================================================================== */
	void set_adaptive(Adaptive_type& adaptive)
	{
        #ifdef COLLECT_ADAPTIVE_INFO
		rate_cpu = adaptive.rate_cpu;
		rate_gpu = adaptive.rate_gpu;
		rate_pcie = adaptive.rate_pcie;

		if ((rate_cpu != 0.0) && (rate_gpu != 0.0) && (rate_pcie != 0.0) )
		{
			Msg_info("读取到Adaptive:rate_cpu = %.3f, rate_gpu = %.3f, rate_pcie = %.3f", rate_cpu, rate_gpu, rate_pcie);
			has_adaptive = true;
		}
        #endif
	}

	
	void set_graphName(std::string name)
	{
		graphName = name;
	}



public:
	/*======================================================================================================================================*
	 *                                                                                                                                      *
	 *                                                      [Graph Cooperation Execution]                                                   *
	 *                                                                                                                                      *
	 *======================================================================================================================================*/


	/***********************************************************
	 * Func: Host + Device 共同执行 Graph
	 *
	 * [root] 迭代的起始顶点
	 ***********************************************************/
	double graphCooperationExecution(vertex_id_type root = 0)
	{
		if (root >= noZeroOutDegreeNum) { Msg_info("当前root 的 outDegree为0, 无需计算, 直接退出！"); return 0.0; }

		resetVariable();// 重置所需的变量
		bool ite_useGPU = false;

		std::thread thread_device(&GraphCooperation::GPU_Execution_Model, this);//开启控制Device的线程

		init_algorithm(algorithm, root);// 初始化算法

		// 计时器
		double processTime = 0.0;
		timer iteTime;
		timer singleHost_time; // 记录Host单次迭代所需的时间
		timer single_time;     // 记录单次迭代所需的时间


		do 
		{
			ite++;
			clear_activeOut_host(algorithm);

			single_time.start();

			//[ONE]: Get Current Total Workloads 
			double workload_total = 0.0; //边计数
			size_t firstWordIndex = 0;
			size_t lastWordIndex = 0;
			{
				singleHost_time.start();
				if (activeNum != 1) // 第一次迭代求workload无意义
				{
#pragma omp parallel num_threads(2)
					{
						size_t threadId_host = omp_get_thread_num();
						if (threadId_host == 0) firstWordIndex = findFirstActiveWord();
						else                    lastWordIndex = findLastActiveWord();
					}

					//workload_total = getTotalTask(firstWordIndex, lastWordIndex + 1); // totalTask
					workload_total = getTotalTask_general(firstWordIndex, lastWordIndex + 1); // totalTask
				}
#ifdef GRAPHHYBIRD_DEBUG
				logstream(LOG_INFO) << "\t\t第[" << ite << "]次迭代, firstWordIndex = " << firstWordIndex
					<< ", lastWordIndex = " << lastWordIndex << ", size = " << (lastWordIndex - firstWordIndex) 
					<< ", workload_total = " << workload_total << std::endl;
				logstream(LOG_INFO) << "\t\t第[" << ite << "]次迭代, [ONE: Get Current Total Workloads] 总用时: " << singleHost_time.get_time_ms() << "(ms)" << std::endl;
#endif
			}


			//[TWO]: Cooperation Execution Condition
			{
				singleHost_time.start();
#ifdef TEMP_DEBUG
				if (ite >= begin && ite <= end) // 控制调用哪种执行模型
#else
				if(activeNum >= (USED_DEVICE_RATE * noZeroOutDegreeNum))
#endif
				{
					ite_useGPU = true;
					partition_workloads(firstWordIndex, lastWordIndex, workload_total);
				}

#ifdef GRAPHHYBIRD_DEBUG
				logstream(LOG_INFO) << "\t\t第[" << ite << "]次迭代, [TWO: Cooperation Execution Condition] 总用时: " << singleHost_time.get_time_ms() << "(ms)" << std::endl;
#endif
			}



			//[Three]: Host-compute
			{
#ifdef GRAPHHYBIRD_DEBUG
				//size_t vertexTask_host_front = parallel_popcount(active.in().array);
				//assert_msg((vertexTask_host_front == active.in().popcount()), "parallel_popcount(...)函数存在问题");
				//logstream(LOG_INFO) << "\t\t->第[" << ite << "]次迭代, [Host需要迭代的顶点数] = " << vertexTask_host_front << std::endl;
#endif
				singleHost_time.start();
				activeNum_host = taskSteal->threeStage_taskSteal<GraphCooperation>(*this,
					[&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end, count_type socketId, bool sameSocket)
					{
						if (Algorithm_type::BFS == algorithm)
                        {
                            return BFS_SPACE::bfs_numa_lastzero<GraphCooperation>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
                        }
                        else if (Algorithm_type::SSSP == algorithm)
                        {
                            return SSSP_SPACE::sssp_numa_lastzero<GraphCooperation>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
                        }
                        else if (Algorithm_type::CC == algorithm)
                        {
                            return CC_SPACE::cc_numa_lastzero<GraphCooperation>(*this, vertex, nbr_start, nbr_end, socketId, sameSocket);
                        }
                        else
                        {
                            assert_msg(false, "CPU Execution Model Meet Unknown Algorithm");
                            return static_cast<count_type>(0);
                        }
					}
				);
				double cpu_execution_time = singleHost_time.get_time_ms();
#ifdef GRAPHHYBIRD_DEBUG
				//size_t vertexTask_host_end = parallel_popcount(active.out().array);
				//logstream(LOG_INFO) << "\t\t->第[" << ite << "]次迭代, [Host实际迭代出的顶点数] = " << vertexTask_host_end << std::endl;
				logstream(LOG_INFO) << "\t\t第[" << ite << "]次迭代, [THREE: Host-Compute] 总用时: <" << cpu_execution_time << "(ms)>" << std::endl;
#endif
#ifdef COLLECT_ADAPTIVE_INFO
				if (ite_useGPU)
				{
					cpu_adaptive_vec[adaptiveNum - 1].time = cpu_execution_time; // 从1开始
					cpu_adaptive_vec[adaptiveNum - 1].rate = cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload / (cpu_execution_time / 1000);
				}				
#endif
			}




//			//[CHOOSE]: Host Help
//			bool needHelp = 0;
//			setActive_hostHelp_mutex.lock();
//			needHelp = (wakeupDevice && (setActive_hostHelp == 1)); // 确保Device不在执行setActive
//			//if (needHelp) setActive_hostHelp = 1; // 通知Device无需执行setActive
//			setActive_hostHelp_mutex.unlock();
//#ifdef TEMP_DEBUG
//			///*if (ite == 3) */needHelp = 1; // 强制第三次迭代由CPU完成reduce
//#endif
			


			//[FOUR]: Sync_Host-Device
			{
				singleHost_time.start();
				while (true)
				{
					wakeupDevice_mutex.lock();
					bool deviceComplete = (wakeupDevice == 0);
					wakeupDevice_mutex.unlock();
					if (deviceComplete) break;
					__asm volatile ("pause" ::: "memory");
				}
#ifdef GRAPHHYBIRD_DEBUG
				logstream(LOG_INFO) << "\t\t第[" << ite << "]次迭代, [FOUR: Host-wait] 总用时: " << singleHost_time.get_time_ms() << "(ms)" << std::endl;
#endif
			}


#ifdef COLLECT_ADAPTIVE_INFO
			if (ite_useGPU)
			{
				std::cout << "       \t(adaptiveNum - 1) = " << (adaptiveNum - 1) <<
					": CPU_R =" << cpu_adaptive_vec[adaptiveNum - 1].rate / (1000000) <<
					"(ME/s), GPU_R = " << gpu_adaptive_vec[adaptiveNum - 1].rate / (1000000) << 
					"(ME/s), C/G = " << (cpu_adaptive_vec[adaptiveNum - 1].rate / gpu_adaptive_vec[adaptiveNum - 1].rate) << std::endl;
			}
#endif
		

			//[CHOOSE]: Host Help
			bool needHelp = false;
			if (setActive_hostHelp == 1)
			{
				needHelp = true;
			}



			//[CHOOSE]: Host Help
			{
				singleHost_time.start();
				if (needHelp) setActive_hostHelp_func();
#ifdef GRAPHHYBIRD_DEBUG
				logstream(LOG_INFO) << "\t\t第[" << ite << "]次迭代, [Host-help] 总用时: " << singleHost_time.get_time_ms() << "(ms)" << std::endl;
#endif

				setActive_hostHelp_mutex.lock();
				setActive_hostHelp = 0; // 恢复
				setActive_hostHelp_mutex.unlock();

				activeNum = parallel_popcount(active.out().array);
#ifdef GRAPHHYBIRD_DEBUG
				activeNum_host = activeNum - activeNum_device;
				logstream(LOG_INFO) << "\t[Single-Ite]：第(" << ite << ")次迭代完成, time = " << single_time.current_time_millis()
					<< " (ms), active_total = (" << activeNum << ")"
					<< ", active_host = " << activeNum_host << ", active_device = " << activeNum_device
					<< std::endl << std::endl;
				std::cout << "\t[Single-Ite]：第(" << std::setw(2) << ite << ")次迭代完成, time = (" << single_time.current_time_millis()
					<< " ms), active = (" << activeNum << ")" << std::endl;
#endif
			}



			// [FIVE]: Break - condition
			if (activeNum == 0)
			{
				processTime = iteTime.current_time_millis();
				std::cout << "[Complete]: " << getAlgName(algorithm) << "-> iteration: " << std::setw(3) << ite
					<< ", time: " << std::setw(6) << processTime << " (ms)" << std::endl;

				hybirdComplete = 1;
				thread_device.join();

				break;
			}

			//[SIX]: SWAP
			active.swap();
			activeNum_device = 0;
			offload_chunkNum = 0;
			ite_useGPU = false;

		} while (true);

		//[CHOOSE]: 如果启用过Device, 最后将outDegree为0的顶点从Device端传回到Host端更新
		if (usedDevice)
		{
			CUDA_CHECK(cudaMemcpy(vertexValue_temp + noZeroOutDegreeNum, vertexValue_device + noZeroOutDegreeNum, (zeroOutDegreeNum) * sizeof(vertex_data_type), cudaMemcpyDeviceToHost));
			omp_parallel_for(vertex_id_type vertexId = noZeroOutDegreeNum; vertexId < vertexNum; vertexId++)
			{
				vertex_data_type msg = vertexValue_temp[vertexId];
				if (msg < vertexValue[vertexId])
				{
					vertexValue[vertexId] = msg;
				}
			}
		}


		return processTime;
	
	}// end of func [graphCooperationExecution(...)]






	/***********************************************************
	 * Func: 重置所需的变量
	 ***********************************************************/
	void resetVariable()
	{
		wakeupDevice = 0; hybirdComplete = 0; nBlock = 0; usedDevice = false; // Device
		activeNum_device = 0; activeNum_host = 0; activeNum = 0;              // Active
		ite = 0;
		zeroChunk = 0;
		for (size_t threadId = 0; threadId < omp_get_max_threads(); threadId++)
		{
#ifdef COLLECT_ADAPTIVE_INFO
			chunkTotalScore_vec[threadId] = 0;
			totalScore = 0.0;
#endif
		}
	}// end of func [resetVariable()]










	/*======================================================================================================================================*
	 *                                                                                                                                      *
	 *                                                          [Cooperation]                                                               *
	 *                                                                                                                                      *
	 *======================================================================================================================================*/

private:

	/***********************************************************
	 * Func: Device 执行 Graph
	 ***********************************************************/
	void GPU_Execution_Model()
	{
		int coreId = 1;
		if (threadBindToCore(coreId))
		{
			assert_msg((sched_getcpu() == coreId), "(sched_getcpu() != coreId) -> (%u != %u)", sched_getcpu(), coreId);
			Msg_info("std::thread success bind to core [%u]", coreId);
		}
		else Msg_info("【Failed】: std::thread bind to core [%u] failed", coreId);

		timer deviceTimeTotal;
		timer deviceTime;

		while (!hybirdComplete)
		{
			while (true)
			{
				wakeupDevice_mutex.lock();
				bool deviceProcess = ((wakeupDevice == 1) || (hybirdComplete));
				wakeupDevice_mutex.unlock();
				if (deviceProcess) break;
				__asm volatile ("pause" ::: "memory");
			}
			if (hybirdComplete) break;


			// offload Host to Device	
#ifdef GRAPHHYBIRD_DEBUG
			deviceTimeTotal.start();
			deviceTime.start();
#endif
			//[ONE]: Device 计算所需的数据
			CUDA_CHECK(cudaMemcpy(offload_offset_device, offload_offset, (offload_chunkNum) * sizeof(count_type), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(offload_data_device, offload_data, (offload_chunkNum * WORD_NUM_CHUNK) * sizeof(uint64_t), cudaMemcpyHostToDevice));
#ifdef GRAPHHYBIRD_DEBUG			
			logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, ==> [1]. Device: Host To Device 完成, 用时: " << deviceTime.get_time_ms() << "(ms)" << std::endl;
#endif




			//[TWO]: Device-process
			deviceTime.start();
#ifdef GPU_MODEL_CHUNK
			offset_type gpu_model_num = 2;
			offset_type gpu_model_size = offload_chunkNum / gpu_model_num;
			offset_type gpu_model_start = 0;
			offset_type gpu_model_end = 0;

			for (offset_type gpu_model_id = 0; gpu_model_id < gpu_model_num; gpu_model_id++)
			{
				gpu_model_start = gpu_model_id * gpu_model_size;
				if (gpu_model_id != (gpu_model_num - 1))
				{
					gpu_model_end = (gpu_model_id + 1) * gpu_model_size;
				}
				else
				{
					gpu_model_end = offload_chunkNum;
				}

				nBlock = ((gpu_model_end - gpu_model_start) * WORD_NUM_CHUNK * 64 + BLOCKSIZE - 1) / BLOCKSIZE;
				printf("nBlock = %lu\n", nBlock);
				CUDA_CHECK(cudaMemcpy(common_device + 6, &gpu_model_start, sizeof(offset_type), cudaMemcpyHostToDevice));

				if (algorithm == Algorithm_type::BFS)
				{
					BFS_SPACE::bfs_hybird_pcie_cooperation<GraphCooperation>(*this, nBlock);
				}
				else if (algorithm == Algorithm_type::SSSP)
				{
					SSSP_SPACE::sssp_hybird_pcie_cooperation<GraphCooperation>(*this, nBlock);
				}
				else
				{
					assert_msg(false, "graphProcessDevice 时, 发现未知算法");
				}				
			}

#else
			nBlock = (offload_chunkNum * WORD_NUM_CHUNK * 64 + BLOCKSIZE - 1) / BLOCKSIZE;
			if (algorithm == Algorithm_type::BFS)
			{
				BFS_SPACE::bfs_hybird_pcie_opt<GraphCooperation>(*this, nBlock);
			}
			else if (algorithm == Algorithm_type::SSSP)
			{
				SSSP_SPACE::sssp_hybird_pcie_opt<GraphCooperation>(*this, nBlock);
			}
			else
			{
				assert_msg(false, "graphProcessDevice 时, 发现未知算法");
			}
#endif		
			double gpu_execution_time = deviceTime.get_time_ms();
#ifdef GRAPHHYBIRD_DEBUG		
			logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, ==> [2]. Device: Compute 完成, 用时: <" << deviceTime.get_time_ms() << "(ms)>, nBlock = " << nBlock << std::endl;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
			gpu_adaptive_vec[adaptiveNum - 1].time = gpu_execution_time; // 从1开始
			gpu_adaptive_vec[adaptiveNum - 1].rate = gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload / (gpu_execution_time / 1000);
#endif


			//[THREE]: vertexValue Device to Host
			deviceTime.start();
			CUDA_CHECK(cudaMemcpy(vertexValue_temp, vertexValue_device, (noZeroOutDegreeNum) * sizeof(vertex_data_type), cudaMemcpyDeviceToHost));//D2H - 传回上次迭代的worklist_count
			double pcie_time = deviceTime.get_time_ms();

#ifdef GRAPHHYBIRD_DEBUG		
			logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, ==> [3]. Device: Device To Host 完成, 用时: " << pcie_time << "(ms)" << std::endl;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
			pcie_adaptive_vec[adaptiveNum - 1].ite = ite;
			pcie_adaptive_vec[adaptiveNum - 1].score = noZeroOutDegreeNum;
			pcie_adaptive_vec[adaptiveNum - 1].time = pcie_time; // 从1开始
			pcie_adaptive_vec[adaptiveNum - 1].rate = pcie_adaptive_vec[adaptiveNum - 1].score / (pcie_time / 1000);
#endif

			//check numa node
			//check_array_numaNode(vertexValue_temp,vertexNum, 1);
			//check_array_numaNode(vertexValue, vertexNum, 1);

			//[FOUR]: Update Active Out
			bool hostHelp = false;
			setActive_hostHelp_mutex.lock();
			hostHelp = setActive_hostHelp;	
			setActive_hostHelp = 1; //新修改：总是让host帮忙
			setActive_hostHelp_mutex.unlock();

			hostHelp = 1; //新修改：总是让host帮忙

			deviceTime.start();
#ifdef TEMP_DEBUG
			//if (ite != 3) // 强制第三次迭代不使用Agent
#endif // TEMP_DEBUG	
			{
				if (!hostHelp)
				{
					setActive_hostHelp_mutex.lock();
					setActive_hostHelp = 2; // 防止Host干扰
					setActive_hostHelp_mutex.unlock();

					for (size_t i = 0; i < noZeroOutDegreeNum; i++)
					{
						vertex_data_type msg = vertexValue_temp[i];
						if (msg < vertexValue[i])
						{
							if (Gemini_atomic::write_min(&vertexValue[i], msg))
							{
								active.out().set_bit(i);
								activeNum_device += 1;
							}
						}
					}


#ifdef GRAPHHYBIRD_DEBUG
					logstream(LOG_INFO) << "\t\t第[" << ite << "]次迭代, ==> [Device]: activeNum_device =  " << activeNum_device << std::endl;
#endif

				}// end of weather hostHelp
			}

#ifdef GRAPHHYBIRD_DEBUG
			logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, ==> [4]. Device: Set Active Out 完成, 用时: " << deviceTime.get_time_ms() << "(ms)" << std::endl;
			logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, ==> [Device] 总用时: " << deviceTimeTotal.get_time_ms() << "(ms)" << std::endl;
#endif


			//wating weakup
			wakeupDevice_mutex.lock();
			wakeupDevice = 0;
			wakeupDevice_mutex.unlock();
		}


	}// end of func [GPU_Execution_Model]


	/***********************************************************
	 * Func: 迭代前, 初始化算法
	 *
	 * [algorithm] 要执行的算法
	 * [root] 迭代的起始顶点
	 ***********************************************************/
	void init_algorithm(Algorithm_type algorithm, vertex_id_type root)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			// Host - Active			
			active.in().clear_memset();
			active.out().clear_memset();
			active.in().set_bit(root);

			//Host - vertexValue
			for (count_type i = 0; i < vertexNum; i++) vertexValue[i] = VertexValue_MAX;
			vertexValue[root] = 0;
			activeNum = 1;

			//Device - vertexValue
			CUDA_CHECK(cudaMemcpy(vertexValue_device, vertexValue, (vertexNum) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));
		}
		else
		{
			assert_msg(false, "init_algorithm 时获取到未知的算法");
		}
	}


	/***********************************************************
	 * Func: 衡量当前迭代的总Task量
	 *
	 * [first] first task in active.in() (包含)
	 * [last]  last  task in active.in() (不包含)
	 ***********************************************************/
	size_t getTotalTask(size_t first, size_t last)
	{
		size_t _first = first * 64; //回归到顶点,第一个顶点
		size_t _last = last * 64;   //回归到顶点,最后一个顶点
		size_t work = _last - _first;
		taskSteal_totalTask->allocateTaskForThread<size_t>(work, SUB_VERTEXSET);

		size_t totalWorkloads = 0;
#pragma omp parallel reduction(+:totalWorkloads)
		{
			size_t thread_id = omp_get_thread_num();
			count_type totalTask_local = 0;

			/*************************************
			 *   2.1.【VERTEX_WORKING】
			 *************************************/
			while (true) {
				size_t vertexId_current = __sync_fetch_and_add(&taskSteal_totalTask->thread_state[thread_id]->cur, VERTEXWORK_CHUNK);
				if (vertexId_current >= taskSteal_totalTask->thread_state[thread_id]->end) break;

				vertexId_current += _first;

				size_t word = active.in().array[WORD_OFFSET(vertexId_current)];
				while (word != 0)
				{
					if (word & 1)
					{
						totalTask_local += outDegree[vertexId_current];
					}
					vertexId_current++;
					word = word >> 1;
				}
			}// end of [2.1.Vertex Working]


			/*************************************
			 *   2.2.【VERTEX_STEALING】
			 *************************************/
			taskSteal_totalTask->thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
			for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++) {//线程窃取的偏移量
				count_type threadId_help = (thread_id + steal_offset) % ThreadNum;//需要帮助的线程
				while (taskSteal_totalTask->thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING) {
					size_t vertexId_current = __sync_fetch_and_add(&taskSteal_totalTask->thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
					if (vertexId_current >= taskSteal_totalTask->thread_state[threadId_help]->end) break;

					vertexId_current += _first;

					size_t word = active.in().array[WORD_OFFSET(vertexId_current)];
					while (word != 0)
					{
						if (word & 1)
						{
							totalTask_local += outDegree[vertexId_current];
						}
						vertexId_current++;
						word = word >> 1;
					}
				}
			}// end of [2.2.VERTEX_STEALING]


			totalWorkloads += totalTask_local;
		}

		return totalWorkloads;
	}


	double getTotalTask_general(size_t first, size_t last)
	{
		size_t _first = first * 64; //回归到顶点,第一个顶点
		size_t _last = last * 64;   //回归到顶点,最后一个顶点
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
			SUB_VERTEXSET
		);

		//用单线程求总任务
		/*double tt = 0;
		for (size_t i=0; i<noZeroOutDegreeNum; i++)
		{
			if(active.in().get(i)) tt += static_cast<double>(outDegree[i]);
		}

		printf("tt = %lf, totalWorkloads = %lu\n", tt, totalWorkloads);
		assert_msg((tt == totalWorkloads), "tt = %lf, totalWorkloads = %lu", tt, totalWorkloads);*/

		return static_cast<double>(totalWorkloads);
	}





	/***********************************************************
	 * Func: Host 执行 partition_workloads_adaptive
	 ***********************************************************/
	void partition_workloads(size_t firstWordIndex, size_t lastWordIndex, double totalTask)
	{
#ifdef COLLECT_ADAPTIVE_INFO
		adaptiveNum++; //需要统计adaptive信息
		if (adaptiveNum >= MAX_ADAPTIVE)
		{
			MAX_ADAPTIVE = MAX_ADAPTIVE * 1.5;
			cpu_adaptive_vec.resize(MAX_ADAPTIVE);
			gpu_adaptive_vec.resize(MAX_ADAPTIVE);
			pcie_adaptive_vec.resize(MAX_ADAPTIVE);
			reduce_adaptive_vec.resize(MAX_ADAPTIVE);
			Msg_info("超过了adaptive_vec 预定义的最大容量,修改为: %lu", MAX_ADAPTIVE);
		}
#endif

		zeroChunk = 0;
		size_t active_chunkNum = 0;

		//对齐
		bool isAlign = 0;
		if (isAlign)
		{
			size_t align = 0;
			if ((lastWordIndex - firstWordIndex + 1) >= WORD_NUM_CHUNK)
			{
				align = (lastWordIndex - firstWordIndex + 1) % WORD_NUM_CHUNK;
			}
			lastWordIndex = lastWordIndex - align;
		}

#ifdef COLLECT_ADAPTIVE_INFO
		std::vector<size_t> chunk_totalVertex(omp_get_max_threads());
		size_t totalVertices = 0;
		for (size_t threadId = 0; threadId < omp_get_max_threads(); threadId++)
		{
			chunkTotalScore_vec[threadId] = 0;
			chunk_totalVertex[threadId] = 0;
		}
		totalScore = 0.0;
#endif


		active_chunkNum = ((lastWordIndex - firstWordIndex + 1) + WORD_NUM_CHUNK - 1) / WORD_NUM_CHUNK;// 按照 WORD_NUM_CHUNK 来切割Active

		// 计算每个CHUNK的任务量
		if (active_chunkNum > RESERVE_CHUNK_VEC)
		{
			chunkTask_vec.resize(active_chunkNum);
			Msg_info("当前的 chunkNum > RESERVE_CHUNK_VEC, chunkTask_vec更新为: %zu", active_chunkNum);
		}
		omp_parallel_for(size_t chunkId = 0; chunkId < active_chunkNum; chunkId++)
		{
			size_t word_start = chunkId * WORD_NUM_CHUNK + firstWordIndex;
			size_t word_end = (chunkId + 1) * WORD_NUM_CHUNK + firstWordIndex;
			if (word_end > lastWordIndex) word_end = lastWordIndex + 1;

			ChunkTask_type chunkTask;
			getChunkTask(word_start, word_end, chunkTask);

			chunkTask_vec[chunkId] = std::make_pair(chunkId, chunkTask);

#ifdef COLLECT_ADAPTIVE_INFO
			chunkTotalScore_vec[omp_get_thread_num()] += chunkTask.score;
			chunk_totalVertex[omp_get_thread_num()] += chunkTask.vertices;
#endif
		}

		//并排序
		sort(chunkTask_vec.begin(), chunkTask_vec.begin() + active_chunkNum,
			[&](const std::pair<size_t, ChunkTask_type>& aa, const std::pair<size_t, ChunkTask_type>& bb)->bool
			{
				if (aa.second.score < bb.second.score)  return true;
				else                                    return false;
			}
		);

#ifdef COLLECT_ADAPTIVE_INFO
		for (size_t threadId = 0; threadId < omp_get_max_threads(); threadId++)
		{
			totalScore += chunkTotalScore_vec[threadId];
			totalVertices += chunk_totalVertex[threadId];
		}
		totalScore = totalScore - zeroChunk * OUR_DOUBLE_MAX;

		assert_msg((totalVertices == activeNum),"(totalVertices != activeNum), totalVertices = %lu, activeNum = %lu", totalVertices, activeNum);

		adaptiveAlpha(); //动态计算alpha

#ifdef GRAPHHYBIRD_DEBUG
		logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, totalVertices = " << totalVertices << ", activeNum = " << activeNum << std::endl;
		logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, active_chunkNum = " << active_chunkNum << ", zeroChunk = " << zeroChunk << ", totalScore = " << totalScore << std::endl;
#endif
#endif

		//打印
		//logWorkloadsChunk(active_chunkNum);
#pragma omp parallel num_threads(2)
		{
			size_t threadId_host = omp_get_thread_num();
			if (threadId_host != 0)
			{
				//开始 partition workload
				selsetOffloadChunk(firstWordIndex, (active_chunkNum), totalTask, totalScore);//(active_chunkNum - 1): 舍弃最后一个chunk, 保证GPU端的对齐
			}
			else
			{
				CUDA_CHECK(cudaMemcpy(vertexValue_device, vertexValue, (noZeroOutDegreeNum) * sizeof(vertex_data_type), cudaMemcpyHostToDevice));
			}
		}

		//通知 Device-Thread
		assert_msg((wakeupDevice == 0), "(wakeupDevice != 0), wakeupDevice = %zu", wakeupDevice);
		wakeupDevice_mutex.lock();
		wakeupDevice = 1;
		wakeupDevice_mutex.unlock();
		usedDevice = true;

	}// end of func [partition_workloads_adaptive]


	void logWorkloadsChunk(size_t active_chunkNum)
	{
		logstream(LOG_INFO) << "开始打印排序: " << std::endl;
		for (size_t i = 0; i < active_chunkNum; i++)
		{
			// 详细log
			logstream(LOG_INFO) << "【 i= " << i << ", vertices = " << chunkTask_vec[i].second.vertices << ", edges = " << chunkTask_vec[i].second.edges
				<< ", score = " << chunkTask_vec[i].second.score << "】" << std::endl;
			// 只log score
			//logstream(LOG_INFO) << chunkTask_vec[i].second.score << std::endl;
		}
		logstream(LOG_INFO) << "排序完成:" << std::endl;
	}


	/***********************************************************
	 * Func: adaptiveAlpha
	 ***********************************************************/
    #ifdef COLLECT_ADAPTIVE_INFO
	void adaptiveAlpha()
	{
		if (adaptiveNum == 1)
		{
			//alpha = (double)cudaCores / (omp_get_max_threads() * 256);
			//std::cout << "default_alpha = " << alpha << std::endl;
		}
		//else
		{
			size_t sum_cpuEdges = 0;
			size_t sum_gpuEdges = 0;
			double sum_cpuRate = 0.0;
			double sum_gpuRate = 0.0;
			for (size_t i = 0; i < adaptiveNum - 1; i++)
			{
				sum_cpuEdges += cpu_adaptive_vec[i].edgeNum_workload;
				sum_gpuEdges += gpu_adaptive_vec[i].edgeNum_workload;
			}

			for (size_t i = 0; i < adaptiveNum - 1; i++)
			{
				sum_cpuRate += cpu_adaptive_vec[i].rate * ((double)(cpu_adaptive_vec[i].edgeNum_workload) / sum_cpuEdges);
				sum_gpuRate += gpu_adaptive_vec[i].rate * ((double)(gpu_adaptive_vec[i].edgeNum_workload) / sum_gpuEdges);
			}

			rate_cpu = sum_cpuRate;
			rate_gpu = sum_gpuRate;

			

			//std::cout << "rate_cpu =" << rate_cpu /(1000000) << "(ME/s), rate_gpu = " << rate_gpu / (1000000) << "(ME/s)" << std::endl;

			compute_alpha(cpu_adaptive_vec[adaptiveNum - 2].edgeNum_workload + gpu_adaptive_vec[adaptiveNum - 2].edgeNum_workload);

		}
	}
   


	double compute_alpha(double workload_total_current_)
	{
		double pcie_time = pcie_adaptive_vec[adaptiveNum - 2].time;
		double workload_total = (workload_total_current_);
		double up = (workload_total)-((pcie_time / 1000) * rate_cpu);
		double down = workload_total * (1 + (rate_cpu / rate_gpu));

		alpha = up / down;
		//std::cout << "\t\talpha = [" << alpha << "], rate_cpu = " << rate_cpu / (1000000) << "(ME/s), rate_gpu = " << rate_gpu / (1000000) << "(ME/s)" << std::endl;

		return alpha;
	}
 #endif
	/***********************************************************
	 * Func: 从activeChunk中选出准备Offload的chunk
	 *
	 * [firstWordIndex]    first task in active.in() (包含)
	 * [active_chunkNum_]  activeChunk 数
	 * [totalTask]         总的task数
	 * [totalScore]        总的score数
	 ***********************************************************/
	void selsetOffloadChunk(size_t firstWordIndex, size_t active_chunkNum_, double totalTask, double totalScore)
	{

#if defined(INDEX_TO_LEFT)

		
		size_t index = lower_bound(active_chunkNum_, static_cast<double>(DEVICE_PERFER));// 我们首先利用二分查找想要的位置
		if (index < (active_chunkNum_ - zeroChunk))
		{		
#ifdef GRAPHHYBIRD_DEBUG
			logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, total_noZeroActiveChunk = " << (active_chunkNum_ - zeroChunk) << ", GPU_workload_start = " << index << std::endl;
#endif
			middleToLeft(firstWordIndex, active_chunkNum_, index, totalTask, totalScore);//从[index]开始往左走
		}
		else
		{
			Msg_info("offload_start未找到");		
			middleToLeft(firstWordIndex, active_chunkNum_, ((active_chunkNum_ - zeroChunk) / 2), totalTask, totalScore);//从[active_chunkNum_ / 2]开始往左走
		}

#elif defined(MIDDLE_TO_LEFT)

		middleToLeft(firstWordIndex, active_chunkNum_, ((active_chunkNum_ - zeroChunk) / 2), totalTask, totalScore);

#else

#error "未知的 Offload Chunk 选取方案"

#endif

#ifdef GRAPHHYBIRD_DEBUG
		logstream(LOG_INFO) << "\t\t\t\t第[" << ite << "]次迭代, Have [" << offload_chunkNum << "] Chunk Offload To Device" << std::endl;
#endif
	}


	size_t lower_bound(size_t active_chunkNum_, double target)
	{
		//我们首先利用二分查找想要的位置	
		size_t binSearch_start = 0;
		size_t binSearch_end = active_chunkNum_ - zeroChunk;
		size_t binSearch_mid = 0;
		while ((binSearch_end - binSearch_start) > 0)
		{
			size_t _count2 = (binSearch_end - binSearch_start) >> 1;
			binSearch_mid = binSearch_start + _count2;

			//printf("[%lu, %lu], mid = %lu\n", binSearch_start, binSearch_end, binSearch_mid);
			if (chunkTask_vec[binSearch_mid].second.score >= target)
			{
				binSearch_end = binSearch_mid;
			}
			else
			{
				binSearch_start = binSearch_mid + 1;
			}
		}

		return binSearch_start;
	}

	void middleToLeft(size_t firstWordIndex, size_t active_chunkNum_, size_t startIndex, double totalTask, double totalScore)
	{
		offload_chunkNum = 0;
		size_t offload_task_current = 0;
#ifdef COLLECT_ADAPTIVE_INFO
		size_t offloadVertexNum_task_current = 0;
		double offloadScore_task_current = 0.0;
#endif
		

		double temp_rate = OFFLOAD_TASK_RATE;// alpha
#ifdef TEMP_DEBUG
		//if (ite == 3) temp_rate = 0.98;

		//如果temp_rate太大时，我们模拟全部给GPU
		if (temp_rate > 0.95)
		{
			size_t chunkId;
			size_t chunkNum_offload_current = (active_chunkNum_ - zeroChunk) - 20;
			for (chunkId = 0; chunkId < chunkNum_offload_current; chunkId++)
			{
				size_t wordOffset = firstWordIndex + chunkTask_vec[chunkId].first * WORD_NUM_CHUNK;
				offload_offset[chunkId] = wordOffset;
				/*assert_msg((offload_offset[chunkId] % WORD_NUM_CHUNK) == 0,
					"offload_offset[offload_chunkNum]不能被WORD_NUM_CHUNK整除, chunkId = %lu, offload_offset[offload_chunkNum] = %lu",
					chunkId, offload_offset[chunkId]);*/


				if (chunkTask_vec[chunkId].second.chunkSize == WORD_NUM_CHUNK)
				{
					memcpy(offload_data + WORD_NUM_CHUNK * chunkId, active.in().array + wordOffset, sizeof(size_t) * (WORD_NUM_CHUNK));
					memset(active.in().array + wordOffset, 0, sizeof(size_t) * (WORD_NUM_CHUNK));
				}
				else
				{
					printf("Offloading对不齐的数据, chunkSize = %lu\n", chunkTask_vec[chunkId].second.chunkSize);
					size_t* temp_append = new size_t[WORD_NUM_CHUNK];
					memset(temp_append, 0, sizeof(size_t) * (WORD_NUM_CHUNK));//全部初始化为0
				
					memcpy(temp_append, active.in().array + wordOffset, sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

					memcpy(offload_data + WORD_NUM_CHUNK * chunkId, temp_append, sizeof(size_t) * (WORD_NUM_CHUNK));
					memset(active.in().array + wordOffset, 0, sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

					delete[] temp_append;
				}


				offload_chunkNum++;
				offload_task_current += chunkTask_vec[chunkId].second.edges;

#ifdef COLLECT_ADAPTIVE_INFO
				offloadVertexNum_task_current += chunkTask_vec[chunkId].second.vertices;
				offloadScore_task_current += chunkTask_vec[chunkId].second.score;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
				gpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
				gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = offloadVertexNum_task_current;
				gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = offload_task_current;
				gpu_adaptive_vec[adaptiveNum - 1].score = offloadScore_task_current;

				cpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
				cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = activeNum - offloadVertexNum_task_current;
				cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = totalTask - offload_task_current;
				cpu_adaptive_vec[adaptiveNum - 1].score = totalScore - offloadScore_task_current;
#endif
			
			}
		}
		else
		{
#endif
			int64_t chunkId = static_cast<int64_t>(startIndex);
			bool complete = false;

			// 向左
			for (; chunkId >= 0; chunkId--)
			{
				//过滤
				if (chunkTask_vec[chunkId].second.vertices < 64) continue;

				size_t wordOffset = firstWordIndex + chunkTask_vec[chunkId].first * WORD_NUM_CHUNK;

				offload_offset[offload_chunkNum] = wordOffset;
				/*assert_msg((offload_offset[offload_chunkNum] % WORD_NUM_CHUNK) == 0,
					"offload_offset[offload_chunkNum]不能被WORD_NUM_CHUNK整除, chunkId = %lu, offload_offset[offload_chunkNum] = %lu", 
					chunkId, offload_offset[offload_chunkNum]);*/

				/*memcpy(offload_data + WORD_NUM_CHUNK * offload_chunkNum, active.in().array + offload_offset[offload_chunkNum], sizeof(size_t) * (WORD_NUM_CHUNK));
				memset(active.in().array + offload_offset[offload_chunkNum], 0, sizeof(size_t) * (WORD_NUM_CHUNK));*/

				if (chunkTask_vec[chunkId].second.chunkSize == WORD_NUM_CHUNK)
				{
					memcpy(offload_data + WORD_NUM_CHUNK * offload_chunkNum, active.in().array + wordOffset, sizeof(size_t) * (WORD_NUM_CHUNK));
					memset(active.in().array + wordOffset, 0, sizeof(size_t) * (WORD_NUM_CHUNK));
				}
				else
				{
					printf("Offloading对不齐的数据, chunkSize = %lu\n", chunkTask_vec[chunkId].second.chunkSize);
					size_t* temp_append = new size_t[WORD_NUM_CHUNK];
					memset(temp_append, 0, sizeof(size_t) * (WORD_NUM_CHUNK));//全部初始化为0

					memcpy(temp_append, active.in().array + wordOffset, sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

					memcpy(offload_data + WORD_NUM_CHUNK * offload_chunkNum, temp_append, sizeof(size_t) * (WORD_NUM_CHUNK));
					memset(active.in().array + wordOffset, 0, sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

					delete[] temp_append;
				}



				offload_chunkNum++;

#ifdef TASK_MIX_CENTER
				offload_task_current += chunkTask_vec[chunkId].second.edges; // TASK_MIX_CENTER下用(edges/vertices)来衡量任务比例不合适
#else
				offload_task_current += chunkTask_vec[chunkId].second.score;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
				offloadVertexNum_task_current += chunkTask_vec[chunkId].second.vertices;
				offloadScore_task_current += chunkTask_vec[chunkId].second.score;
#endif


				if (offload_task_current >= (totalTask * hand_offload_vec[ite])) //adaptive_offload_rate  simulation_workload_rate[ite - 4] alpha
				{
					complete = true;
#ifdef COLLECT_ADAPTIVE_INFO
					gpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
					gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = offloadVertexNum_task_current;
					gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = offload_task_current;
					gpu_adaptive_vec[adaptiveNum - 1].score = offloadScore_task_current;

					assert_msg(static_cast<int64_t>(activeNum - offloadVertexNum_task_current) >= 0, "activeNum <= offloadVertexNum_task_current");
					assert_msg(static_cast<int64_t>(totalTask - offload_task_current) >= 0, "totalTask <= offload_task_current");
					assert_msg(static_cast<int64_t>(totalScore - offloadScore_task_current) >= 0, "totalScore <= offloadScore_task_current");

					cpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始				
					cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = activeNum - offloadVertexNum_task_current;
					cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = totalTask - offload_task_current;
					cpu_adaptive_vec[adaptiveNum - 1].score = totalScore - offloadScore_task_current;
#endif
					break;
				}
			}

			// 向左不足时, 从中间转向右边
			if (!complete)
			{
				chunkId = static_cast<int64_t>(startIndex + 1);
				for (; chunkId < (active_chunkNum_ - zeroChunk); chunkId++)// 舍弃最后一个chunk, 保证GPU端的对齐
				{
					//过滤
					if (chunkTask_vec[chunkId].second.vertices < 64) continue;

					size_t wordOffset = firstWordIndex + chunkTask_vec[chunkId].first * WORD_NUM_CHUNK;
					offload_offset[offload_chunkNum] = wordOffset;
					/*assert_msg((offload_offset[offload_chunkNum] % WORD_NUM_CHUNK) == 0,
						"offload_offset[offload_chunkNum]不能被WORD_NUM_CHUNK整除, chunkId = %lu, offload_offset[offload_chunkNum] = %lu",
						chunkId, offload_offset[offload_chunkNum]);*/

					if (chunkTask_vec[chunkId].second.chunkSize == WORD_NUM_CHUNK)
					{
						memcpy(offload_data + WORD_NUM_CHUNK * offload_chunkNum, active.in().array + wordOffset, sizeof(size_t) * (WORD_NUM_CHUNK));
						memset(active.in().array + wordOffset, 0, sizeof(size_t) * (WORD_NUM_CHUNK));
					}
					else
					{
						printf("Offloading对不齐的数据, chunkSize = %lu\n", chunkTask_vec[chunkId].second.chunkSize);
						size_t* temp_append = new size_t[WORD_NUM_CHUNK];
						memset(temp_append, 0, sizeof(size_t) * (WORD_NUM_CHUNK));//全部初始化为0

						memcpy(temp_append, active.in().array + wordOffset, sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

						memcpy(offload_data + WORD_NUM_CHUNK * offload_chunkNum, temp_append, sizeof(size_t) * (WORD_NUM_CHUNK));
						memset(active.in().array + wordOffset, 0, sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

						delete[] temp_append;
					}

					offload_chunkNum++;

#ifdef TASK_MIX_CENTER
					offload_task_current += chunkTask_vec[chunkId].second.edges; // TASK_MIX_CENTER下用(edges/vertices)来衡量任务比例不合适
#else
					offload_task_current += chunkTask_vec[chunkId].second.score;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
					offloadVertexNum_task_current += chunkTask_vec[chunkId].second.vertices;
					offloadScore_task_current += chunkTask_vec[chunkId].second.score;
#endif

					if (offload_task_current >= (totalTask * hand_offload_vec[ite])) {  //  simulation_workload_rate[ite - 4]  alpha
#ifdef COLLECT_ADAPTIVE_INFO
						gpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
						gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = offloadVertexNum_task_current;
						gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = offload_task_current;
						gpu_adaptive_vec[adaptiveNum - 1].score = offloadScore_task_current;

						assert_msg(static_cast<int64_t>(activeNum - offloadVertexNum_task_current) >= 0, "activeNum <= offloadVertexNum_task_current");
						assert_msg(static_cast<int64_t>(totalTask - offload_task_current) >= 0, "totalTask <= offload_task_current");
						assert_msg(static_cast<int64_t>(totalScore - offloadScore_task_current) >= 0, "totalScore <= offloadScore_task_current");

						cpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始						
						cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = activeNum - offloadVertexNum_task_current;
						cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = totalTask - offload_task_current;
						cpu_adaptive_vec[adaptiveNum - 1].score = totalScore - offloadScore_task_current;;
#endif
						break;
					}
				}
			}
#ifdef TEMP_DEBUG
		}
#endif // TEMP_DEBUG

#ifdef COLLECT_ADAPTIVE_INFO
#ifdef GRAPHHYBIRD_DEBUG
		logstream(LOG_INFO) << "\t\t\t\t\t第[" << ite << "]次迭代, CPU分到的edgeNum_workload = " << cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload
			<< ", vertexNum_workload = " << cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload
			<< ", GPU分到的edgeNum_workload = " << gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload
			<< ", vertexNum_workload = " << gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload
			<< std::endl;
#endif
#endif

		if (offload_chunkNum >= MAX_OFFFLOAD_CHUNK) assert_msg(false, "Offload Chunk Num Large Than MAX_OFFFLOAD_CHUNK");
	}



	void middleToLeft_old(size_t firstWordIndex, size_t active_chunkNum_, size_t startIndex, double totalTask, double totalScore)
	{
		offload_chunkNum = 0;
		size_t offload_task_current = 0;
#ifdef COLLECT_ADAPTIVE_INFO
		size_t offloadVertexNum_task_current = 0;
		double offloadScore_task_current = 0.0;
#endif


		double temp_rate = OFFLOAD_TASK_RATE;// alpha
#ifdef TEMP_DEBUG
		//if (ite == 3) temp_rate = 0.98;

		//如果temp_rate太大时，我们模拟全部给GPU
		if (temp_rate > 0.95)
		{
			size_t chunkId;
			size_t chunkNum_offload_current = (active_chunkNum_ - zeroChunk) - 20;
			for (chunkId = 0; chunkId < chunkNum_offload_current; chunkId++)
			{
				offload_offset[chunkId] = firstWordIndex + chunkTask_vec[chunkId].first * WORD_NUM_CHUNK;

				if (chunkTask_vec[chunkId].second.chunkSize == WORD_NUM_CHUNK)
				{
					memcpy(offload_data + WORD_NUM_CHUNK * chunkId, active.in().array + offload_offset[chunkId], sizeof(size_t) * (WORD_NUM_CHUNK));
					memset(active.in().array + offload_offset[chunkId], 0, sizeof(size_t) * (WORD_NUM_CHUNK));
				}
				else
				{
					printf("Offloading对不齐的数据, chunkSize = %lu\n", chunkTask_vec[chunkId].second.chunkSize);
					size_t* temp_append = new size_t[WORD_NUM_CHUNK];
					memset(temp_append, 0, sizeof(size_t) * (WORD_NUM_CHUNK));//全部初始化为0

					memcpy(temp_append, active.in().array + offload_offset[chunkId], sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

					memcpy(offload_data + WORD_NUM_CHUNK * chunkId, temp_append, sizeof(size_t) * (WORD_NUM_CHUNK));
					memset(active.in().array + offload_offset[chunkId], 0, sizeof(size_t) * (chunkTask_vec[chunkId].second.chunkSize));

					delete[] temp_append;
				}


				offload_chunkNum++;
				offload_task_current += chunkTask_vec[chunkId].second.edges;

#ifdef COLLECT_ADAPTIVE_INFO
				offloadVertexNum_task_current += chunkTask_vec[chunkId].second.vertices;
				offloadScore_task_current += chunkTask_vec[chunkId].second.score;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
				gpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
				gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = offloadVertexNum_task_current;
				gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = offload_task_current;
				gpu_adaptive_vec[adaptiveNum - 1].score = offloadScore_task_current;

				cpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
				cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = activeNum - offloadVertexNum_task_current;
				cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = totalTask - offload_task_current;
				cpu_adaptive_vec[adaptiveNum - 1].score = totalScore - offloadScore_task_current;
#endif

			}
		}
		else
		{
#endif
			int64_t chunkId = static_cast<int64_t>(startIndex);
			bool complete = false;

			// 向左
			for (; chunkId >= 0; chunkId--)
			{
				//过滤
				if (chunkTask_vec[chunkId].second.vertices <= 64) continue;

				offload_offset[offload_chunkNum] = firstWordIndex + chunkTask_vec[chunkId].first * WORD_NUM_CHUNK;
				assert_msg((offload_offset[offload_chunkNum] % WORD_NUM_CHUNK) == 0,
					"offload_offset[offload_chunkNum]不能被WORD_NUM_CHUNK整除, chunkId = %ld, offload_offset[offload_chunkNum] = %zu",
					chunkId, static_cast<uint64_t>(offload_offset[offload_chunkNum]));

				memcpy(offload_data + WORD_NUM_CHUNK * offload_chunkNum, active.in().array + offload_offset[offload_chunkNum], sizeof(size_t) * (WORD_NUM_CHUNK));
				memset(active.in().array + offload_offset[offload_chunkNum], 0, sizeof(size_t) * (WORD_NUM_CHUNK));

				offload_chunkNum++;

#ifdef TASK_MIX_CENTER
				offload_task_current += chunkTask_vec[chunkId].second.edges; // TASK_MIX_CENTER下用(edges/vertices)来衡量任务比例不合适
#else
				offload_task_current += chunkTask_vec[chunkId].second.score;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
				offloadVertexNum_task_current += chunkTask_vec[chunkId].second.vertices;
				offloadScore_task_current += chunkTask_vec[chunkId].second.score;
#endif




				if (offload_task_current >= (totalTask * hand_offload_vec[ite])) //adaptive_offload_rate  simulation_workload_rate[ite - 4] alpha
				{
					complete = true;
#ifdef COLLECT_ADAPTIVE_INFO
					gpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
					gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = offloadVertexNum_task_current;
					gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = offload_task_current;
					gpu_adaptive_vec[adaptiveNum - 1].score = offloadScore_task_current;

					assert_msg(static_cast<int64_t>(activeNum - offloadVertexNum_task_current) >= 0, "activeNum <= offloadVertexNum_task_current");
					assert_msg(static_cast<int64_t>(totalTask - offload_task_current) >= 0, "totalTask <= offload_task_current");
					assert_msg(static_cast<int64_t>(totalScore - offloadScore_task_current) >= 0, "totalScore <= offloadScore_task_current");

					cpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始				
					cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = activeNum - offloadVertexNum_task_current;
					cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = totalTask - offload_task_current;
					cpu_adaptive_vec[adaptiveNum - 1].score = totalScore - offloadScore_task_current;
#endif
					break;
				}
			}

			// 向左不足时, 从中间转向右边
			if (!complete)
			{
				chunkId = static_cast<int64_t>(startIndex);
				for (; chunkId < (active_chunkNum_ - zeroChunk); chunkId++)// 舍弃最后一个chunk, 保证GPU端的对齐
				{
					//过滤
					if (chunkTask_vec[chunkId].second.vertices <= 64) continue;

					offload_offset[offload_chunkNum] = firstWordIndex + chunkTask_vec[chunkId].first * WORD_NUM_CHUNK;

					memcpy(offload_data + WORD_NUM_CHUNK * offload_chunkNum, active.in().array + offload_offset[offload_chunkNum], sizeof(size_t) * (WORD_NUM_CHUNK));
					memset(active.in().array + offload_offset[offload_chunkNum], 0, sizeof(size_t) * (WORD_NUM_CHUNK));
					offload_chunkNum++;

#ifdef TASK_MIX_CENTER
					offload_task_current += chunkTask_vec[chunkId].second.edges; // TASK_MIX_CENTER下用(edges/vertices)来衡量任务比例不合适
#else
					offload_task_current += chunkTask_vec[chunkId].second.score;
#endif

#ifdef COLLECT_ADAPTIVE_INFO
					offloadVertexNum_task_current += chunkTask_vec[chunkId].second.vertices;
					offloadScore_task_current += chunkTask_vec[chunkId].second.score;
#endif

					if (offload_task_current >= (totalTask * hand_offload_vec[ite])) {  //  simulation_workload_rate[ite - 4]  alpha
#ifdef COLLECT_ADAPTIVE_INFO
						gpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始			
						gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = offloadVertexNum_task_current;
						gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = offload_task_current;
						gpu_adaptive_vec[adaptiveNum - 1].score = offloadScore_task_current;

						assert_msg(static_cast<int64_t>(activeNum - offloadVertexNum_task_current) >= 0, "activeNum <= offloadVertexNum_task_current");
						assert_msg(static_cast<int64_t>(totalTask - offload_task_current) >= 0, "totalTask <= offload_task_current");
						assert_msg(static_cast<int64_t>(totalScore - offloadScore_task_current) >= 0, "totalScore <= offloadScore_task_current");

						cpu_adaptive_vec[adaptiveNum - 1].ite = ite; // 从1开始						
						cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload = activeNum - offloadVertexNum_task_current;
						cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload = totalTask - offload_task_current;
						cpu_adaptive_vec[adaptiveNum - 1].score = totalScore - offloadScore_task_current;;
#endif
						break;
					}
				}
			}
#ifdef TEMP_DEBUG
		}
#endif // TEMP_DEBUG

#ifdef COLLECT_ADAPTIVE_INFO
#ifdef GRAPHHYBIRD_DEBUG
		logstream(LOG_INFO) << "\t\t\t\t\t第[" << ite << "]次迭代, CPU分到的edgeNum_workload = " << cpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload
			<< ", vertexNum_workload = " << cpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload
			<< ", GPU分到的edgeNum_workload = " << gpu_adaptive_vec[adaptiveNum - 1].edgeNum_workload
			<< ", vertexNum_workload = " << gpu_adaptive_vec[adaptiveNum - 1].vertexNum_workload
			<< std::endl;
#endif
#endif


		if (offload_chunkNum >= MAX_OFFFLOAD_CHUNK) assert_msg(false, "Offload Chunk Num Large Than MAX_OFFFLOAD_CHUNK");
	}

	/***********************************************************
	 * Func: Host帮助Device完成setActive
	 ***********************************************************/
	void setActive_hostHelp_func()
	{
		count_type activeNum_device_total = 0;
#pragma omp parallel reduction(+:activeNum_device_total)
		{
			count_type threadId = omp_get_thread_num();
			count_type activeNum_device_local = 0;

			for (size_t i = taskSteal_hostHelp->thread_state[threadId]->cur;
				i < taskSteal_hostHelp->thread_state[threadId]->end;
				i++)
			{
				vertex_data_type msg = vertexValue_temp[i];
				if (msg < vertexValue[i])
				{
					vertexValue[i] = msg;
					active.out().set_bit(i);
					activeNum_device_local += 1;
				}
			}

			activeNum_device_total += activeNum_device_local;
		}
		activeNum_device = activeNum_device_total;

#ifdef GRAPHHYBIRD_DEBUG
		//Msg_info("activeNum_device = 【%u】", activeNum_device);
		logstream(LOG_INFO) << "\t\t\t第[" << ite << "]次迭代, ==> [HostHelp]: activeNum_device =  " << activeNum_device << std::endl;
#endif

	}

	/***********************************************************
	 * Func: 获取当前Chunk的task量
	 *
	 * [word_start] first task in active.in() (包含)
	 * [word_end]   last  task in active.in() (不包含)
	 * 不要在inline中用return
	 ***********************************************************/
	inline void getChunkTask(size_t word_start, size_t word_end, ChunkTask_type& chunkTask)
	{
		//ChunkTask_type chunkTask;
#if defined(TASK_MIX_CENTER)

		size_t vertices = 0;
		size_t edges = 0;
		for (size_t wordId = word_start; wordId < word_end; wordId++)
		{
			vertices += __builtin_popcountl(active.in().array[wordId]);
		}

		if (vertices != 0)
		{
			for (size_t wordId = word_start; wordId < word_end; wordId++)
			{
				size_t vertex_offset = wordId * 64;
				size_t temp = active.in().array[wordId];
				size_t k = 0;
				while (temp != 0)
				{
					if (temp & 1) edges += outDegree[vertex_offset + k];
					k++;
					temp = temp >> 1;
				}
			}

			chunkTask.chunkSize = word_end - word_start;
			chunkTask.vertices = vertices;
			chunkTask.edges = edges;
			chunkTask.score = static_cast<double>(edges) / static_cast<double>(vertices);
		}
		else
		{
			__sync_fetch_and_add_8(&zeroChunk, 1);

			chunkTask.chunkSize = word_end - word_start;
			chunkTask.vertices = 0;
			chunkTask.edges = 0;
			chunkTask.score = OUR_DOUBLE_MAX;
		}
#else

#error "未知的 Chunk Task 衡量方案"

#endif


	}

	/*======================================================================================================================================*
	 *                                                                                                                                      *
	 *                                                             [Host]                                                                   *
	 *                                                                                                                                      *
	 *======================================================================================================================================*/

public:

	/***********************************************************
	 * Func: Host函数, 获取当前顶点所属的Socket
	 *
	 * [vertexId] 顶点Id
	 ***********************************************************/
	inline count_type getVertexSocketId(vertex_id_type vertexId)
	{
		count_type socketId = 0;
		for (; socketId < socketNum; socketId++)
		{
			if (vertexId < numa_offset[socketId + 1])
			{
				break;
			}
		}
		return socketId;
	}


private:

	/***********************************************************
	 * Func: Host函数, 初始化Graph 和 Numa
	 ***********************************************************/
	void initGraphNuma()
	{
		//检查NUMA是否可用
		assert_msg(numa_available() != -1, "NUMA不可用");
		assert_msg(sizeof(size_t) == 8, "需要使用64位机器");
		if (vertexNum >= UINT32_MAX) printf("[WARNING]: vertexNum >=UINT32_MAX, 注意设置count_type为 uint64_t\n");
		if (edgeNum >= UINT32_MAX) printf("[WARNING]: edgeNum >=UINT32_MAX, 注意设置countl_type为 uint64_t\n");

		threadNum = ThreadNum;
		socketNum = SocketNum;
		threadPerSocket = ThreadNum / SocketNum;

		omp_parallel_for(int threadId = 0; threadId < threadNum; threadId++)
		{
#ifdef GRAPHHYBIRD_DEBUG
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

		//初始化基本的CSR_NUMA指针
		csr_offset_numa = new countl_type * [socketNum];
		csr_dest_numa = new vertex_id_type * [socketNum];
		csr_weight_numa = new vertex_id_type * [socketNum];

		vertexNum_numa = new count_type[socketNum];
		edgeNum_numa = new countl_type[socketNum];
		zeroOffset_numa = new count_type[socketNum];

		//初始化taskSteal
		taskSteal = new TaskSteal();
		taskSteal_align64 = new TaskSteal();
		taskSteal_hostHelp = new TaskSteal();
		taskSteal_totalTask = new TaskSteal();

		//初始化outDegree
		outDegree = new offset_type[vertexNum];
	}



	/***********************************************************
	 * Func: Host函数, 将Graph分配到不同的NUMA节点上
	 ***********************************************************/
	void partitionGraphByNuma()
	{
		//先计算划分点，我们采用简单的平均划分点
		numa_offset = new vertex_id_type[socketNum + 1];
		numa_offset[0] = 0;

		vertexPerSocket = noZeroOutDegreeNum / socketNum / 64 * 64; //必选按word对齐,否则active会在窃取时乱套

		for (count_type socketId = 1; socketId < socketNum; socketId++)
		{
			numa_offset[socketId] = numa_offset[socketId - 1] + vertexPerSocket;
		}
		numa_offset[socketNum] = noZeroOutDegreeNum;

		// 将CSR_Result_type划分到不同的NUMA上
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			count_type vertexs_numa = numa_offset[socketId + 1] - numa_offset[socketId];
			countl_type edges_numa = csr_offset[numa_offset[socketId + 1]] - csr_offset[numa_offset[socketId]];

			csr_offset_numa[socketId] = (countl_type*)numa_alloc_onnode((vertexs_numa + 1) * sizeof(countl_type), socketId);
			countl_type offset = csr_offset[numa_offset[socketId]];
			for (count_type i = 0; i < (vertexs_numa + 1); i++)
			{
				csr_offset_numa[socketId][i] = csr_offset[numa_offset[socketId] + i] - offset; //让CSR的offset从零开始
			}

			csr_dest_numa[socketId] = (vertex_id_type*)numa_alloc_onnode((edges_numa) * sizeof(vertex_id_type), socketId);
			memcpy(csr_dest_numa[socketId], csr_dest + csr_offset[numa_offset[socketId]], edges_numa * sizeof(vertex_id_type));

			csr_weight_numa[socketId] = (edge_data_type*)numa_alloc_onnode((edges_numa) * sizeof(edge_data_type), socketId);
			memcpy(csr_weight_numa[socketId], csr_weight + csr_offset[numa_offset[socketId]], edges_numa * sizeof(edge_data_type));

			//存储常用到的变量
			vertexNum_numa[socketId] = vertexs_numa;
			edgeNum_numa[socketId] = edges_numa;
			zeroOffset_numa[socketId] = numa_offset[socketId];
		}


#ifdef GRAPHHYBIRD_DEBUG
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			Msg_info("Socket[%zu] 分到的顶点数：(%9zu), 分到的边数：(%10zu)", socketId,
				static_cast<uint64_t>(numa_offset[socketId + 1] - numa_offset[socketId]), static_cast<uint64_t>(edgeNum_numa[socketId]));
		}
#endif

#ifdef GRAPHHYBIRD_DEBUG
		//debugCsrArray();
		debugSocketArray();
		for (count_type socketId = 0; socketId < socketNum; socketId++)
		{
			check_array_numaNode(csr_offset_numa[socketId], vertexNum_numa[socketId], socketId);
			check_array_numaNode(csr_dest_numa[socketId], edgeNum_numa[socketId], socketId);
			check_array_numaNode(csr_weight_numa[socketId], edgeNum_numa[socketId], socketId);
		}
		Msg_check("(csr_offset_numa, csr_dest_numa, csr_weight_numa) check_array_numaNode 通过");
#endif
	}



	/***********************************************************
	 * Func: Host函数, debug每个NUMA节点的Graph分配情况
	 ***********************************************************/
	void debugSocketArray()
	{
		std::stringstream ss;
		ss << "zeroOffset_numa:\n";
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			ss << std::setw(9) << zeroOffset_numa[socketId] << " ";
		}

		ss << "\nvertexNum_numa:\n";
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			ss << std::setw(9) << vertexNum_numa[socketId] << " ";
		}

		ss << "\nedgeNum_numa:\n";
		for (size_t socketId = 0; socketId < socketNum; socketId++)
		{
			ss << std::setw(9) << edgeNum_numa[socketId] << " ";
		}

		logstream(LOG_INFO) << ss.str() << std::endl << std::endl;
	}



	/***********************************************************
	 * Func: Host函数, 检查数组是否被分配到指定的NUMA节点上
	 *
	 * [adr]   要检查的数组首地址
	 * [size]  要检查的数组的长度
	 * [node]  要检查的数组所希望的NUMA顶点
	 ***********************************************************/
    template<typename T>
	void check_array_numaNode(T* adr, size_t size, size_t node)
	{
		//move_pages必须是Page对齐的，因此此处需要重新计算
		size_t pageNum_temp = PAGESIZE / sizeof(vertex_id_type);
		size_t len = (size + pageNum_temp - 1) / pageNum_temp;
		for (size_t i = 0; i < len; i++)
		{
			int checkNumaNode = getAdrNumaNode((adr + (i * pageNum_temp)));
			assert_msg(checkNumaNode == node, "getAdrNumaNode 出错");
		}
	}



	/***********************************************************
	 * Func: Host函数, 为vertexValue和Active分配所需内存
	 ***********************************************************/
	void allocate_vertexValueAndActive()
	{
		taskSteal_align64->allocateTaskForThread<count_type>(noZeroOutDegreeNum, 64, true);
		taskSteal_hostHelp->allocateTaskForThread<count_type>(noZeroOutDegreeNum);

#ifdef PIN_MEM
		CUDA_CHECK(cudaMallocHost((void**)&(vertexValue), (vertexNum) * sizeof(vertex_data_type)));
#else
		vertexValue = new vertex_data_type[vertexNum];
		Msg_info("**********************************************************************");
		Msg_info("*   【Used PIN Memory Tranfer VertexValue Between Host And Device】   *");
		Msg_info("**********************************************************************");
#endif

		active_in.resize(noZeroOutDegreeNum);
		active_out.resize(noZeroOutDegreeNum);
		active.setDoubleBuffer(active_in, active_out);//封装到DoubleBuffer中

	}



	/***********************************************************
	 * Func: Host函数, 根据指定的算法,清理每次迭代后的active
	 *
	 * [algorithm]  要执行的算法
	 ***********************************************************/
	void clear_activeOut_host(Algorithm_type algorithm)
	{
		if ((Algorithm_type::BFS == algorithm) || (Algorithm_type::SSSP == algorithm))
		{
			//本Socket的线程共同完成本socket的active clear
			omp_parallel
			{
				count_type threadId = omp_get_thread_num();
				size_t cur = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->cur);
				size_t end = WORD_OFFSET(taskSteal_align64->thread_state[threadId]->end);
				memset(active.out().array + cur, 0, sizeof(size_t) * (end - cur));
			}
		}
		else
		{
			assert_msg(false, "clear_active_out 时获取到未知算法");
		}
	}



	/***********************************************************
	 * Func: Host函数, 获取Graph的outDegree
	 ***********************************************************/
	void get_outDegree()
	{
		omp_parallel_for(vertex_id_type vertexId = 0; vertexId < noZeroOutDegreeNum; vertexId++)
		{
			outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
		}
	}



	/***********************************************************
	 * Func: Host函数, 并行计算popcount
	 ***********************************************************/
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



	/***********************************************************
	 * Func: 寻找active中第一个1所在的word索引
	 ***********************************************************/
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


	/***********************************************************
	 * Func: 寻找active中最后一个1所在的word索引
	 ***********************************************************/
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



	/*======================================================================================================================================*
	 *                                                                                                                                      *
	 *                                                             [Device]                                                                 *
	 *                                                                                                                                      *
	 *======================================================================================================================================*/

private:

	/***********************************************************
	 * Func: Host函数, 在Device上分配所需的内存
	 ***********************************************************/
	void allocate_device()
	{
		CUDA_CHECK(cudaMallocHost((void**)&(common), (common_size) * sizeof(offset_type)));

		CUDA_CHECK(cudaMalloc((void**)&(csr_offset_device), (vertexNum + 1) * sizeof(countl_type)));
		CUDA_CHECK(cudaMalloc((void**)&(csr_dest_device), (edgeNum) * sizeof(vertex_id_type)));
		if (algorithm == Algorithm_type::SSSP)
		{
			CUDA_CHECK(cudaMalloc((void**)&(csr_weight_device), (edgeNum) * sizeof(edge_data_type)));
		}
		CUDA_CHECK(cudaMalloc((void**)&(common_device), (common_size) * sizeof(offset_type)));
		CUDA_CHECK(cudaMalloc((void**)&(vertexValue_device), (vertexNum) * sizeof(vertex_data_type)));
	}


	/***********************************************************
	 * Func: Host函数, 将所需的数据由Host传到Device端
	 ***********************************************************/
	void graphToDevice()
	{
		CUDA_CHECK(cudaMemcpy(csr_offset_device, csr_offset, (vertexNum + 1) * sizeof(countl_type), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(csr_dest_device, csr_dest, (edgeNum) * sizeof(vertex_id_type), cudaMemcpyHostToDevice));
		if (algorithm == Algorithm_type::SSSP)
		{
			CUDA_CHECK(cudaMemcpy(csr_weight_device, csr_weight, (edgeNum) * sizeof(edge_data_type), cudaMemcpyHostToDevice));
		}

		offset_type vertexNum_ = static_cast<offset_type>(vertexNum); // 防止 count_type 与 offset_type不一致
		CUDA_CHECK(cudaMemcpy(common_device + 2, &vertexNum_, sizeof(offset_type), cudaMemcpyHostToDevice));
		if(std::numeric_limits<offset_type>::max() < edgeNum) assert_msg(false, "edge need large type");
		offset_type edgeNum_ = static_cast<offset_type>(edgeNum); // 防止 count_type 与 offset_type不一致
		CUDA_CHECK(cudaMemcpy(common_device + 3, &edgeNum_, sizeof(offset_type), cudaMemcpyHostToDevice));
		offset_type noZeroOutDegreeNum_temp = static_cast<offset_type>(noZeroOutDegreeNum); // 防止数据类型不一致
		CUDA_CHECK(cudaMemcpy(common_device + 4, &noZeroOutDegreeNum_temp, sizeof(offset_type), cudaMemcpyHostToDevice));
		offset_type wordNumChunk_temp = static_cast<offset_type>(WORD_NUM_CHUNK); // 防止数据类型不一致
		CUDA_CHECK(cudaMemcpy(common_device + 5, &wordNumChunk_temp, sizeof(offset_type), cudaMemcpyHostToDevice));
	}


	/***********************************************************
	 * Func: Host函数, 在Device上分配offload所需的内存
	 ***********************************************************/
	void allocate_offload()
	{
		CUDA_CHECK(cudaMallocHost((void**)&(offload_offset), (offload_offsetNum) * sizeof(count_type)));
		CUDA_CHECK(cudaMalloc((void**)&(offload_offset_device), (offload_offsetNum) * sizeof(count_type)));
		CUDA_CHECK(cudaMallocHost((void**)&(offload_data), (offload_wordNum) * sizeof(uint64_t)));
		CUDA_CHECK(cudaMalloc((void**)&(offload_data_device), (offload_wordNum) * sizeof(uint64_t)));
		CUDA_CHECK(cudaMallocHost((void**)&(vertexValue_temp), (vertexNum) * sizeof(vertex_data_type)));

		chunkTask_vec.resize(RESERVE_CHUNK_VEC);
		//chunkTotalTask_vec.resize(omp_get_max_threads(), 0);
        #ifdef COLLECT_ADAPTIVE_INFO
		chunkTotalScore_vec.resize(omp_get_max_threads(), 0.0);
        #endif

#ifdef COLLECT_ADAPTIVE_INFO
		cpu_adaptive_vec.resize(MAX_ADAPTIVE);
		gpu_adaptive_vec.resize(MAX_ADAPTIVE);
		pcie_adaptive_vec.resize(MAX_ADAPTIVE);
		reduce_adaptive_vec.resize(MAX_ADAPTIVE);

		for (size_t i = 0; i < MAX_ADAPTIVE; i++)
		{
			cpu_adaptive_vec[i].rate_type = RATE_Type::CPU;
			gpu_adaptive_vec[i].rate_type = RATE_Type::GPU;
			pcie_adaptive_vec[i].rate_type = RATE_Type::PCIe;
			reduce_adaptive_vec[i].rate_type = RATE_Type::Reduce;
		}
#endif
	}



public:
	~GraphCooperation()
	{
		CUDA_CHECK(cudaFree(csr_offset_device));
		CUDA_CHECK(cudaFree(csr_dest_device));
		CUDA_CHECK(cudaFree(common_device));
		CUDA_CHECK(cudaFree(vertexValue_device));
		CUDA_CHECK(cudaFree(offload_offset_device));
		CUDA_CHECK(cudaFree(offload_data_device));

		CUDA_CHECK(cudaFreeHost(vertexValue));
		CUDA_CHECK(cudaFreeHost(common));
		CUDA_CHECK(cudaFreeHost(offload_offset));
		CUDA_CHECK(cudaFreeHost(offload_data));
		CUDA_CHECK(cudaFreeHost(vertexValue_temp));

		/*delete[] csr_offset;
		delete[] csr_dest;
		delete[] csr_weight;
		delete[] outDegree;*/
	}

};// end of class [GraphCooperatiion]