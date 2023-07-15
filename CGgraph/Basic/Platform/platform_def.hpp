#pragma once

#if defined(_WIN32) || defined(WIN32) || defined(__WIN32__)
    #include <windows.h>
    #define PLATFORM_NAME "WIN"
    #define PLATFORM_WIN
#elif   defined(__linux__)
    #include <stdlib.h>
    #define PLATFORM_NAME "LINUX"
    #define PLATFORM_LINUX
#else
    #define PLATFORM "UNKNOWN"
    #error  "We need [Linux] or [Win] platform, current platform is [unknown]"
#endif
