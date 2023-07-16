#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <iterator>
#include <numa.h>
#include <assert.h>
#include <omp.h>
#include "../GPU/cuda_check.cuh"
#include "../Console/console.hpp"
#include "../Thread/omp_def.hpp"
#include "../Thread/simple_spinlock.hpp"

typedef uint64_t bit_type;

#define BITSIZE 64
#define BIT_OFFSET(i) ((i) >> 6)
#define BIT_MOD(i) ((i) & 0x3f) //0x3f = 63


class Fixed_Bitset{

public:
    bit_type* array;
    bit_type len;
    bit_type arrlen;


    

    //Fixed_Bitset(): array(NULL), len(0), arrlen(0) {}

    Fixed_Bitset() : array(NULL), len(0), arrlen(0) {}

    Fixed_Bitset(bit_type n) : array(NULL), len(0), arrlen(0) {

        setSize(n);
    }

    Fixed_Bitset(const Fixed_Bitset& db) {
        array = NULL;
        len = 0;
        arrlen = 0;
        *this = db;
    }

    ~Fixed_Bitset() {CUDA_CHECK(cudaFreeHost(array));}


    /* ***************************************************************************************************************************
     *                                              Common Function
     * ***************************************************************************************************************************/
    void fix_trailing_bits() {
        bit_type lastbits = BIT_MOD(len);
        if (lastbits == 0) return;
        array[arrlen - 1] &= ((bit_type(1) << lastbits) - 1);     
    }



    void setSize(bit_type n)
    {
        if constexpr (sizeof(bit_type) != 8) assert_msg(false, "<bit_type> Only Support With 64 Bits");

        if(len != 0) assert_msg(false, "Fixed_Bitset Not Allow Set Size More Time");

        len = n;
        arrlen = BIT_OFFSET(n) + (BIT_MOD(n) > 0);
        CUDA_CHECK(cudaMallocHost((void**)&array, arrlen * sizeof(bit_type)));
        fix_trailing_bits();
        parallel_clear();
    }


    inline bit_type size() const {
        return len;
    }

 
    inline bool empty() const {
        for (bit_type i = 0; i < arrlen; ++i) if (array[i]) return false;
        return true;
    }

    inline bool parallel_empty() const {
        volatile bool flag = true;
        #pragma omp parallel for shared(flag)
        for (bit_type i = 0; i < arrlen; ++i)
        {
            if(!flag) continue;
            if(array[i] == 0) flag = false;
        }
        return flag;
    }

    /* ***************************************************************************************************************************
     *                                                  fill and clear
     * Note: We provide single and parallel two versions
     * Performace:   parallel_clear() is better
     *               parallel_fill()  is better
     * ***************************************************************************************************************************/
    inline void clear() {
        for (bit_type i = 0; i < arrlen; ++i) 
            array[i] = 0;
    }

    inline void clear_memset_()
    {
        memset((void*)array, 0, sizeof(bit_type) * arrlen);
    }

    inline void parallel_clear() {
        omp_parallel_for (bit_type i = 0; i < arrlen; ++i){
            array[i] = 0;
        }            
    }

    inline void fill() {
        for (bit_type i = 0; i < arrlen; ++i) array[i] = (bit_type)-1;
        fix_trailing_bits();
    }

    inline void parallel_fill() {
        omp_parallel_for (bit_type i = 0; i < arrlen; ++i) array[i] = (bit_type)-1;
        fix_trailing_bits();
    }


    /* ***************************************************************************************************************************
     *                                              Normal Function
     * ***************************************************************************************************************************/
    inline bool get(bit_type b) const {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        return array[arrpos] & (bit_type(1) << bit_type(bitpos));
    }

    inline bool set_bit(bit_type b) {
        // use CAS to set the bit
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type mask(bit_type(1) << bit_type(bitpos));
        return __sync_fetch_and_or(array + arrpos, mask) & mask;//GCC
    }

     inline bool set_bit_unsync(bit_type b) {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type mask(bit_type(1) << bit_type(bitpos));
        bool ret = array[arrpos] & mask;/
        array[arrpos] |= mask;
        return ret;
    }

    inline bool clear_bit(bit_type b) {
        // use CAS to set the bit
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type test_mask(bit_type(1) << bit_type(bitpos));
        const bit_type clear_mask(~test_mask);
        return __sync_fetch_and_and(array + arrpos, clear_mask) & test_mask;
    }

    inline bool clear_bit_unsync(bit_type b) {
        // use CAS to set the bit
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type test_mask(bit_type(1) << bit_type(bitpos));
        const bit_type clear_mask(~test_mask);
        bool ret = array[arrpos] & test_mask;
        array[arrpos] &= clear_mask;
        return ret;
    }

    inline bool set(bit_type b, bool value) {
        if (value) return set_bit(b);
        else return clear_bit(b);
    }

    inline bool set_unsync(bit_type b, bool value) {
        if (value) return set_bit_unsync(b);
        else return clear_bit_unsync(b);
    }

    bit_type popcount() const {
        bit_type ret = 0;
        for (bit_type i = 0; i < arrlen; ++i) {
            //：https://blog.csdn.net/gaochao1900/article/details/5646211 
            ret += __builtin_popcountl(array[i]);
        }
        return ret;
    }

    bit_type parallel_popcount() const {
        bit_type ret = 0;
        #pragma omp parallel for reduction(+: ret)
        for (bit_type i = 0; i < arrlen; ++i) {
            /：https://blog.csdn.net/gaochao1900/article/details/5646211 
            ret += __builtin_popcountl(array[i]);
        }
        return ret;
    }

    inline size_t containing_word(size_t b) {
        size_t arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        return array[arrpos];
    }


    /* ***************************************************************************************************************************
     *                                                  operator
     * Note: The Other operator Waiting...
     * ***************************************************************************************************************************/
     inline Fixed_Bitset& operator=(const Fixed_Bitset& db) {
        len = db.len;
        arrlen = db.arrlen;
        CUDA_CHECK(cudaMallocHost((void**)&array, arrlen * sizeof(bit_type)));
        memcpy(array, db.array, sizeof(bit_type) * arrlen);
        return *this;
    }

    //操作符() - 这里我们采用operate()进行引用传递, Used In DoubleBuffer
    inline Fixed_Bitset& operator()(const Fixed_Bitset& db) {
        len = db.len;
        arrlen = db.arrlen;
        array = db.array;
        return *this;
    }


     /* ***************************************************************************************************************************
     *                                                 dense_bitset
     * ***************************************************************************************************************************/
    void resize(bit_type n)
    {
        setSize(n);
    }

    void clear_memset()
    {
        parallel_clear();
    }

private:

    inline static void bit_to_pos(bit_type b, bit_type& arrpos, bit_type& bitpos) {
        // the compiler better optimize this...
        arrpos = BIT_OFFSET(b);
        bitpos = BIT_MOD(b);
    }


};// end of class [Fixed_Bitset]