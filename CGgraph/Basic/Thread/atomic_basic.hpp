#pragma once




#include <type_traits>
#include <cstdint>
#include "../Platform/platform_def.hpp"



//-------------------------------------------【WINDOW】---------------------------------------------
#if defined (PLATFORM_WIN)
//-------------------------------------------【WINDOW】---------------------------------------------

template <typename T>
inline T atomic_fetch_and_add(T* ptr, T value)
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->fetch_add(value);// std::memory_order_seq_cst
}

template <typename T>
inline T atomic_add_and_fetch(T* ptr, T value) 
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->fetch_add(value) + value;// std::memory_order_seq_cst
}




template <typename T>
inline T atomic_fetch_and_sub(T* ptr, T value)
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->fetch_sub(value);// std::memory_order_seq_cst
}

template <typename T>
inline T atomic_sub_and_fetch(T* ptr, T value)
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->fetch_sub(value) - value;// std::memory_order_seq_cst
}


template <typename T>
inline T atomic_fetch_and_or(T* ptr, T value)
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->fetch_or(value);// std::memory_order_seq_cst
}

template <typename T>
inline T atomic_fetch_and_and(T* ptr, T value)
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->fetch_and(value);// std::memory_order_seq_cst
}

template <typename T>
inline T atomic_fetch_and_xor(T* ptr, T value)
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->fetch_xor(value);// std::memory_order_seq_cst
}

template <typename T>
inline T atomic_cas(T* ptr, T old_val, T new_val) 
{
    std::atomic<T>* a_ptr = (std::atomic<T> *) ptr;
    return a_ptr->compare_exchange_strong(old_val, new_val);// std::memory_order_seq_cst
}


//-------------------------------------------【LINUX】---------------------------------------------
#elif defined (PLATFORM_LINUX)
//-------------------------------------------【LINUX】---------------------------------------------

    template <typename T>
    inline T atomic_fetch_and_add(T* address, T value)
    {
       return  __sync_fetch_and_add(address, value);
    }

    template <typename T>
    inline T atomic_add_and_fetch(T* address, T value)
    {
        return __sync_add_and_fetch(address, value);
    }


    template <typename T>
    inline T atomic_fetch_and_sub(T* address, T value)
    {
        return __sync_fetch_and_sub(address, value);
    }

    template <typename T>
    inline T atomic_sub_and_fetch(T* address, T value)
    {
       return  __sync_sub_and_fetch(address, value);
    }

    template <typename T>
    inline T atomic_cas(T* address, T old_value, T new_value)
    {
        return __sync_bool_compare_and_swap(&address, old_value, new_value);
    }


    template <typename T>
    inline T atomic_fetch_and_or(T* ptr, T value)
    {
        return __sync_fetch_and_or(ptr, value);
    }

    template <typename T>
    inline T atomic_fetch_and_and(T* ptr, T value)
    {
        return __sync_fetch_and_and(ptr, value);
    }

    template <typename T>
    inline T atomic_fetch_and_xor(T* ptr, T value)
    {
        return __sync_fetch_and_xor(ptr, value);
    }


#else
    #error "Platform error in {atomic_basic.hpp}"
#endif



#ifdef PLATFORM_LINUX

/*
 * With GCC there are other caveats:
 * http://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Atomic-Builtins.html
 */


#endif