#ifndef GPUInfo_CUH
#define GPUInfo_CUH

#include "cuda_include.cuh"
#include "cuda_check.cuh"

#include <string>
#include <vector>

#include <sstream>
#include <assert.h>




    class GPUInfo {

    private:

        //cudaDeviceProp prop; 

        /**
         * =============================================
         */      
        int device_count;
        int currentDeviceId;
        std::vector<char*>         deviceName;
        //std::string deviceName;

        



        /**
         * =============================================
         */
        //
        std::vector<int>          clockRate; //GPU clock
        std::vector<int>          multiProcessorCount; //SMs
        std::vector<int>          coreNum;//cores

        //
        std::vector<int>          major;                      /**< Major compute capability */
        std::vector<int>          minor;                      /**< Minor compute capability */

        //
        std::vector<int>          computeMode;                /**< Compute mode (See ::cudaComputeMode) */
        std::vector<int>          asyncEngineCount; 




        /**
         * =============================================
         */
        //
        std::vector<int>          memoryClockRate;                        /**< Peak memory clock frequency in kilohertz */
        std::vector<int>          memoryBusWidth;
        //
        std::vector < size_t >      totalGlobalMem;
        std::vector < size_t >      sharedMemPerBlock;
        std::vector<int>         regsPerBlock;
        std::vector<int>          l2CacheSize;
        std::vector<int>          canMapHostMemory;
        std::vector<int>          unifiedAddressing;
        std::vector<int>          managedMemory;              /**< Device supports allocating managed memory on this system */


        
        /**
         * =============================================
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

        
        int getDeviceNum()
        {
            cudaGetDeviceCount(&device_count);
            return device_count;
        }

        
        std::vector<int> getCoreNum()
        {
            assert(isReady);
            return coreNum;
        }

        
        std::vector<int> getSMNum()
        {
            assert(isReady);
            return multiProcessorCount;
        }

		
		std::vector<size_t> getGlobalMem_byte()
		{
			assert(isReady);
			return totalGlobalMem;
		}


        
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

        
        void getAllGPUInfo(bool isPrint = false)
        {
            cudaGetDeviceCount(&device_count);
            

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
    


#endif // !GPUInfo_CUH
