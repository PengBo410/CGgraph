#pragma once

#include <omp.h>
#include <iostream>
#include "../Platform/platform_def.hpp"
#include <libgen.h>

#define TOTAL_DEBUG       //no color
#define TOTAL_DEBUG_COLOR //color



#define ESC_START     "\033["
#define ESC_END       "\033[0m"
#define COLOR_FATAL   "31;40;5m"   //5:
#define COLOR_ALERT   "31;40;1m"
#define COLOR_CRIT    "31;40;1m" 
#define COLOR_ERROR   "31;48;1m"   //31:
#define COLOR_WARN    "33;40;1m"   //33:
#define COLOR_NOTICE  "34;40;1m"   //34:
#define COLOR_CHECK   "32;48;5m"   //32: 
#define COLOR_WRITE   "36;48;1m"   //36:
#define COLOR_FREES   "32;48;1m"   //32:
#define COLOR_FINSH   "32;48;1m"   //32:
#define COLOR_INFOS   "37;48;1m"   //37:
#define COLOR_RATES   "33;48;1m"   //33:


//************************************************[WINDOW]***********************************************
#if defined (PLATFORM_WIN)
//************************************************[WINDOW]***********************************************


    std::string basename(std::string fullPath)
    {
        std::size_t found = fullPath.find_last_of("/\\");
        return fullPath.substr(found + 1);
    }


    #if defined (TOTAL_DEBUG_COLOR)

        #define assert_msg(condition, format, ...) \
        if(true){                                      \
            if(!condition){                          \
                printf(ESC_START COLOR_ERROR "<ERROR>:" format " -> T[%u] {%s: %u}\n" ESC_END, __VA_ARGS__, omp_get_thread_num(), basename(__FILE__).c_str(),__LINE__); \
                exit(1);                                   \
            }                                            \
        }

        #define Msg_info(format, ...)  (printf(ESC_START COLOR_INFOS "[INFOS]: " format " -> {%s: %u}" ESC_END "\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_check(format, ...) (printf(ESC_START COLOR_CHECK "[CHECK]: " format " -> {%s: %u}" ESC_END "\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_write(format, ...) (printf(ESC_START COLOR_WRITE "[WRITE]: " format " -> {%s: %u}" ESC_END "\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_free(format, ...)  (printf(ESC_START COLOR_FREES "[FREES]: " format " -> {%s: %u}" ESC_END "\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_rate(format, ...)  (printf(ESC_START COLOR_RATES "[RATES]: " format " -> {%s: %u}" ESC_END "\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))

    #elif defined (TOTAL_DEBUG)

        #define assert_msg(condition, format, ...) \
        if(true){                                      \
            if(!condition){                          \
                printf("<ERROR>:" format " -> T[%u] {%s: %u}\n", __VA_ARGS__, omp_get_thread_num(), basename(__FILE__).c_str(),__LINE__); \
                exit(1);                                   \
            }                                            \
        }

        #define Msg_info(format, ...)  (printf("[INFOS]: " format " -> {%s: %u}\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_check(format, ...) (printf("[CHECK]: " format " -> {%s: %u}\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_write(format, ...) (printf("[WRITE]: " format " -> {%s: %u}\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_free(format, ...)  (printf("[FREES]: " format " -> {%s: %u}\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))
        #define Msg_rate(format, ...)  (printf("[RATES]: " format " -> {%s: %u}\n", __VA_ARGS__, basename(__FILE__).c_str(),__LINE__))

    #else

        #define assert_msg(condition, format, ...) \
        if(true){                                      \
            if(!condition){                          \
                printf("<ERROR>:" format " -> T[%u] {%s: %u}\n", __VA_ARGS__, omp_get_thread_num(), basename(__FILE__).c_str(),__LINE__); \
                exit(1);                                   \
            }                                            \
        }

        #define Msg_info(format, ...)
        #define Msg_check(format, ...)
        #define Msg_write(format, ...)
        #define Msg_free(format, ...)
        #define Msg_rate(format, ...)


    #endif // defined TOTAL_DEBUG_COLOR


//************************************************[LINUX]***********************************************
#elif defined(PLATFORM_LINUX)
//************************************************[LINUX]***********************************************

#include <libgen.h>

    #if defined (TOTAL_DEBUG_COLOR)

        #define assert_msg(condition, format, args...) \
        if(true){                                      \
            if(__builtin_expect(!(condition), 0)){                          \
                printf(ESC_START COLOR_ERROR "<ERROR>:" format " -> T[%u] {%s: %u行}\n" ESC_END, ##args, omp_get_thread_num(), basename((char *)(__FILE__)),__LINE__); \
                exit(1);                                   \
            }                                              \
        }

        #define Msg_info(format, args...)  (printf(ESC_START COLOR_INFOS "[INFOS]: " format " -> {%s: %u行}" ESC_END "\n"  , ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_finish(format, args...)(printf(ESC_START COLOR_FINSH "[FINSH]: " format " -> {%s: %u行}" ESC_END "\n"  , ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_check(format, args...) (printf(ESC_START COLOR_CHECK "[CHECK]: " format " -> {%s: %u行}" ESC_END "\n"  , ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_write(format, args...) (printf(ESC_START COLOR_WRITE "[WRITE]: " format " -> {%s: %u行}" ESC_END "\n"  , ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_free(format, args...)  (printf(ESC_START COLOR_FREES "[FREES]: " format " -> {%s: %u行}" ESC_END "\n"  , ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_rate(format, args...)  (printf(ESC_START COLOR_RATES "[RATES]: " format " -> {%s: %u行}" ESC_END "\n"  , ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_major(format, args...) (printf(ESC_START COLOR_RATES "[MAJOR]: " format " -> {%s: %u行}" ESC_END "\n"  , ##args, basename((char *)__FILE__),__LINE__))

    #elif defined (TOTAL_DEBUG)

        #define assert_msg(condition, format, args...) \
        if(true){                                      \
            if(__builtin_expect(!(condition), 0)){                          \
                printf("【ERROR】:" format " -> T[%u] {%s: %u行}\n", ##args, omp_get_thread_num(), basename(__FILE__),__LINE__); \
                exit(1);                                   \
            }                                            \
        }  

        #define Msg_info(format, args...)  (printf("[INFOS]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_finsh(format, args...) (printf("[FINSH]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_check(format, args...) (printf("[CHECK]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_write(format, args...) (printf("[WRITE]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_free(format, args...)  (printf("[FREES]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__),__LINE__))
        #define Msg_rate(format, args...)  (printf("[RATES]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__),__LINE__))     
        #define Msg_major(format, args...) (printf("[MAJOR]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__),__LINE__))     
            
    #else

       #define assert_msg(condition, format, args...) \
        if(true){                                      \
            if(__builtin_expect(!(condition), 0)){                          \
                printf("【ERROR】:" format " -> T[%u] {%s: %u行}\n", ##args, omp_get_thread_num(), basename(__FILE__),__LINE__); \
                exit(1);                                   \
            }                                            \
        } 

        #define Msg_info(format, args...)  
        #define Msg_finsh(format, args...)
        #define Msg_check(format, args...) 
        #define Msg_write(format, args...) 
        #define Msg_free(format, args...)  
        #define Msg_rate(format, args...) 
        #define Msg_major(format, args...)
    #endif

 

#endif // end of PLATFORM