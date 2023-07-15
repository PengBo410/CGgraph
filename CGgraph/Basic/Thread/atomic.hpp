#pragma once

#include "atomic_basic.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


/* ***************************************************
 * 为了支持double 和 float，重写cas
 * **************************************************/
template <class T>
inline bool cas(T* ptr, T old_val, T new_val)
 {
    if constexpr (sizeof(T) == 4)
    {
        uint32_t* a_ptr = reinterpret_cast<uint32_t*>(&ptr);
        const uint32_t* oldval_ptr = reinterpret_cast<const uint32_t*>(&old_val);
        const uint32_t* newval_ptr = reinterpret_cast<const uint32_t*>(&new_val);
        return atomic_cas(a_ptr, *oldval_ptr, *newval_ptr);
    }
    else if constexpr (sizeof(T) == 8)
    {
        uint64_t* a_ptr = reinterpret_cast<uint64_t*>(&ptr);
        const uint64_t* oldval_ptr = reinterpret_cast<const uint64_t*>(&old_val);
        const uint64_t* newval_ptr = reinterpret_cast<const uint64_t*>(&new_val);
        return atomic_cas(a_ptr, *oldval_ptr, *newval_ptr);
    }
    else
    {
        assert(false);
    }
}



template <class T>
inline bool atomic_min(T* ptr, T val) {
    volatile T curr_val; bool done = false;//volatile 有用吗
    do {		
        curr_val = *ptr;			
    } while (curr_val > val && !(done = cas(ptr, curr_val, val)));
    return done;
}

template <class T>
inline void atomic_add(T* ptr, T val) {
    volatile T new_val, old_val;
    do {
        old_val = *ptr;
        new_val = old_val + val;
    } while (!cas(ptr, old_val, new_val));
}