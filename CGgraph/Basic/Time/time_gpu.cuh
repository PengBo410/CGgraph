#pragma once

#include "../GPU/cuda_include.cuh"

/*
    util::timer_t timer;
    timer.begin();
    //----
    timer.end();
*/

struct Timer_GPU {
    float time;

    inline Timer_GPU() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }

    ~Timer_GPU() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Alias of each other, start the timer.
    void begin() { cudaEventRecord(start_); }
    void start() { this->begin(); }

    // Alias of each other, stop the timer.
    float end() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&time, start_, stop_);

        return milliseconds();
    }
    float stop() { return this->end(); }

    float seconds() { return time * 1e-3; }
    float milliseconds() { return time; }

    inline double get_time_ms() 
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&time, start_, stop_);

        return time;
   }

    inline double get_time() 
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&time, start_, stop_);

        return time * 1e-3;
    }

 private:
    cudaEvent_t start_, stop_;
};