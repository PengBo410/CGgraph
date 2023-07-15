#pragma once

#include <chrono>

class Timer {
public:

    inline void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    inline void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedMilliseconds() {
        std::chrono::duration<double, std::milli> elapsed_time = end_time_ - start_time_;
        return elapsed_time.count();
    }

    inline Timer() { start(); }

    inline double get_time_ms() 
    {
        stop();
        std::chrono::duration<double> elapsed_time = end_time_ - start_time_;
        return elapsed_time.count() * 1000;
    }

    inline double get_time() 
    {
        stop();
        std::chrono::duration<double> elapsed_time = end_time_ - start_time_;
        return elapsed_time.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
};