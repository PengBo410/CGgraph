#ifndef GPUInfo_CUH
#define GPUInfo_CUH

#include "cuda_include.cuh"
#include "cuda_check.cuh"

#include <string>
#include <vector>

#include <sstream>
#include <assert.h>

/**
 * Func: 获取GPU的基本信息
 *
 * 用法：  GPUInfo GPUInfo;
 *	      GPUInfo.getAllGPUInfo(true);
 */
//namespace GY {

    class GPUInfo {

    private:

        //cudaDeviceProp prop; //主体

        /**
         * =============================================【基本信息】
         */      
        int device_count;
        int currentDeviceId;
        std::vector<char*>         deviceName;
        //std::string deviceName;

        



        /**
         * =============================================【计算能力与时钟频率】
         */
        //核频率
        std::vector<int>          clockRate; //GPU时钟频率
        std::vector<int>          multiProcessorCount; //SM数量
        std::vector<int>          coreNum;//总核数

        //计算能力
        std::vector<int>          major;                      /**< Major compute capability */
        std::vector<int>          minor;                      /**< Minor compute capability */

        //计算模型
        std::vector<int>          computeMode;                /**< Compute mode (See ::cudaComputeMode) */
        std::vector<int>          asyncEngineCount; //GPU是否支持设备重叠(Device Overlap)功能,支持设备重叠功能的GPU能够,在执行一个CUDA C核函数的同时，还能在设备与主机之间执行复制等操作,





        /**
         * =============================================【内存与时钟频率】
         */
        //内存时钟频率
        std::vector<int>          memoryClockRate;                        /**< Peak memory clock frequency in kilohertz */
        std::vector<int>          memoryBusWidth;//显存位宽是显存在一个时钟周期内所能传送数据的位数，位数越大则瞬间所能传输的数据量越大，这是显存的重要参数之一     /**< Global memory bus width in bits */

        //内存
        std::vector < size_t >      totalGlobalMem;
        std::vector < size_t >      sharedMemPerBlock;
        std::vector<int>         regsPerBlock;
        std::vector<int>          l2CacheSize;
        std::vector<int>          canMapHostMemory;//是否支持CPU和GPU之间的内存映射
        std::vector<int>          unifiedAddressing;//是否支持设备与主机共享一个统一的地址空间
        std::vector<int>          managedMemory;              /**< Device supports allocating managed memory on this system */


        
        /**
         * =============================================【通用】
         */     
        //线程限制
        std::vector<int>          maxThreadsPerMultiProcessor;
        std::vector < size_t >       sharedMemPerMultiprocessor;
        std::vector<int>          regsPerMultiprocessor;
        std::vector<int>          isMultiGpuBoard;            /**< Device is on a multi-GPU board */

        bool isReady = false;


    public:

        GPUInfo(bool isPrint = false)
        {
            getAllGPUInfo(isPrint);
        }

        // 获取当前机器总的GPU数
        int getDeviceNum()
        {
            cudaGetDeviceCount(&device_count);
            return device_count;
        }

        // 获取当前机器拥有GPU的Core数
        std::vector<int> getCoreNum()
        {
            assert(isReady);
            return coreNum;
        }

        // 获取当前机器拥有GPU的SM数
        std::vector<int> getSMNum()
        {
            assert(isReady);
            return multiProcessorCount;
        }

		// 获取最大容量(单位是byte)
		std::vector<size_t> getGlobalMem_byte()
		{
			assert(isReady);
			return totalGlobalMem;
		}


        // 获取GPU的核数
        int getSPcores(int& mp, int& major, int& minor)
        {
            int cores = 0;
            switch (major) {
            case 2: // Fermi
                if (minor == 1) cores = mp * 48;
                else cores = mp * 32;
                break;
            case 3: // Kepler
                cores = mp * 192;
                break;
            case 5: // Maxwell
                cores = mp * 128;
                break;
            case 6: // Pascal
                if ((minor == 1) || (minor == 2)) cores = mp * 128;
                else if (minor == 0) cores = mp * 64;
                else printf("Unknown device type\n");
                break;
            case 7: // Volta and Turing
                if ((minor == 0) || (minor == 5)) cores = mp * 64;
                else printf("Unknown device type\n");
                break;
            default:
                printf("Unknown device type\n");
                break;
            }
            return cores;
        }

        // 打印GPU的全部信息
        void getAllGPUInfo(bool isPrint = false)
        {
            cudaGetDeviceCount(&device_count);
            //logstream(LOG_INFO) << "当前服务器共拥有【" << device_count << "】个GPU" << std::endl;

            deviceName.resize(device_count);
            clockRate.resize(device_count);
            multiProcessorCount.resize(device_count);
            major.resize(device_count);
            minor.resize(device_count);
            computeMode.resize(device_count);
            asyncEngineCount.resize(device_count);
            memoryClockRate.resize(device_count);
            memoryBusWidth.resize(device_count);
            totalGlobalMem.resize(device_count);
            sharedMemPerBlock.resize(device_count);
            regsPerBlock.resize(device_count);
            l2CacheSize.resize(device_count);
            canMapHostMemory.resize(device_count);
            unifiedAddressing.resize(device_count);
            managedMemory.resize(device_count);
            maxThreadsPerMultiProcessor.resize(device_count);
            sharedMemPerMultiprocessor.resize(device_count);
            regsPerMultiprocessor.resize(device_count);
            isMultiGpuBoard.resize(device_count);
            coreNum.resize(device_count);

            std::stringstream sstream;
            for (currentDeviceId = 0; currentDeviceId < device_count; ++currentDeviceId)
            {
                cudaSetDevice(currentDeviceId);
                cudaDeviceProp device_prop;
                cudaGetDeviceProperties(&device_prop, currentDeviceId);

                

                deviceName[currentDeviceId] = device_prop.name;
                clockRate[currentDeviceId] = device_prop.clockRate;
                multiProcessorCount[currentDeviceId] = device_prop.multiProcessorCount;
                major[currentDeviceId] = device_prop.major;
                minor[currentDeviceId] = device_prop.minor;
                computeMode[currentDeviceId] = device_prop.computeMode;
                asyncEngineCount[currentDeviceId] = device_prop.asyncEngineCount;
                memoryClockRate[currentDeviceId] = device_prop.memoryClockRate;
                memoryBusWidth[currentDeviceId] = device_prop.memoryBusWidth;
                totalGlobalMem[currentDeviceId] = device_prop.totalGlobalMem;
                sharedMemPerBlock[currentDeviceId] = device_prop.sharedMemPerBlock;
                regsPerBlock[currentDeviceId] = device_prop.regsPerBlock;
                l2CacheSize[currentDeviceId] = device_prop.l2CacheSize;
                canMapHostMemory[currentDeviceId] = device_prop.canMapHostMemory;
                unifiedAddressing[currentDeviceId] = device_prop.unifiedAddressing;
                managedMemory[currentDeviceId] = device_prop.managedMemory;
                maxThreadsPerMultiProcessor[currentDeviceId] = device_prop.maxThreadsPerMultiProcessor;
                sharedMemPerMultiprocessor[currentDeviceId] = device_prop.sharedMemPerMultiprocessor;
                regsPerMultiprocessor[currentDeviceId] = device_prop.regsPerMultiprocessor;
                isMultiGpuBoard[currentDeviceId] = device_prop.isMultiGpuBoard;

                coreNum[currentDeviceId] = getSPcores(
                    multiProcessorCount[currentDeviceId], major[currentDeviceId], minor[currentDeviceId]);

                sstream << "\nCurrent server has [" << device_count << "] GPU, Id = [" << currentDeviceId << "] 的GPU:\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| GPU Name:                                |  " << deviceName[currentDeviceId] << "\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| SM Number:                               |  " << multiProcessorCount[currentDeviceId] << "\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| Total Core Number:                       |  " << coreNum[currentDeviceId] << "\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| Global Memory Capacity:                  |  " << (double) totalGlobalMem[currentDeviceId]/(1024*1024*1024) << "(GB)\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| Shared Memory / Block:                   |  " << (double)sharedMemPerBlock[currentDeviceId]/(1024) << "(KB)\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| Memory Bandwidth:                        |  " << memoryBusWidth[currentDeviceId] << "(GB/S)\n" 
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| L2-Cache Size:                           |  " << (double)l2CacheSize[currentDeviceId] / (1024) << "(KB)\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| Core Frequency:                          |  " << clockRate[currentDeviceId] << "\n"                          
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| Major.Minor Compute:                     |  " << major[currentDeviceId] << "." << minor[currentDeviceId] <<"\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << "\t| Compute Model:                           |  " << computeMode[currentDeviceId] << "\n"
                    << "\t| GPU Is Support Overlap:                  |  " << asyncEngineCount[currentDeviceId] << "\n"
                    << "\t| Memory Frequency:                        |  " << memoryClockRate[currentDeviceId] << "\n"                                
                    << "\t| Registers / Block:                       |  " << regsPerBlock[currentDeviceId] << "\n"                 
                    << "\t| Can Mapping Between CPU and GPU:         |  " << canMapHostMemory[currentDeviceId] << "\n"
                    << "\t| Can CPU And GPU Share A Unified Address: |  " << unifiedAddressing[currentDeviceId] << "\n"
                    << "\t| Is Support ManagedMemory Memory:         |  " << managedMemory[currentDeviceId] << "\n"
                    << "\t| Max Threads / SM:                        |  " << maxThreadsPerMultiProcessor[currentDeviceId] << "\n"
                    << "\t| Shared Memory / SM:                      |  " << (double)sharedMemPerMultiprocessor[currentDeviceId] / (1024) << "(KB)\n"
                    << "\t| Max Registers / Block:                   |  " << regsPerMultiprocessor[currentDeviceId] << "\n"
                    << "\t| Is Support Multi-GPU Motherboard:        |  " << isMultiGpuBoard[currentDeviceId] << "\n"
                    << "\t----------------------------------------------------------------------------\n"
                    << std::endl;
            }

            //logstream(LOG_INFO) << sstream.str();

            if (isPrint) std::cout << sstream.str();

            isReady = true;          
        }


    };
    
//}

//参考资料：https://www.sohu.com/a/275199558_100007018
//        https://www.cnpython.com/qa/62311
//        https://blog.csdn.net/fengbingchun/article/details/76902556

#endif // !GPUInfo_CUH
