#pragma once

#ifndef DENSEBITSET
#define DENSEBITSET

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <iterator>
#include <numa.h>
#include <assert.h>
//#include <omp.h>

#define WORD_OFFSET_(i) ((i) >> 6)
#define BIT_OFFSET_(i) ((i) & 0x3f) //0x3f = 63

/********************************************************************************************************************************************************
 *                                                       Class Functions                                                                                *
 *                                                     ----------------------                                                                           *
 * 【功能】：实现原子dense bitset
 ********************************************************************************************************************************************************/
//namespace GY {

    class dense_bitset
    {
    public:

        size_t* array;
        size_t len;
        size_t arrlen;

        template <size_t len>
        friend class fixed_dense_bitset;

        /****************************************************************************************************************************                                                     *
         *                                           ------------------------                                                       *
         ****************************************************************************************************************************/
         /// Constructs a bitset of 0 length
        dense_bitset() : array(NULL), len(0), arrlen(0) {
        }

        /// Constructs a bitset with 'size' bits. All bits will be cleared.
        explicit dense_bitset(size_t size) : array(NULL), len(0), arrlen(0) {
            resize(size);
            clear();
        }

        /// 
        dense_bitset(const dense_bitset& db) {
            array = NULL;
            len = 0;
            arrlen = 0;
            *this = db;
        }

        //
        ~dense_bitset() {free(array); }

        //=
        inline dense_bitset& operator=(const dense_bitset& db) {
            resize(db.size());
            len = db.len;
            arrlen = db.arrlen;
            memcpy(array, db.array, sizeof(size_t) * arrlen);
            return *this;
        }

		//()
		inline dense_bitset& operator()(const dense_bitset& db) {
			len = db.len;
			arrlen = db.arrlen;
			array = db.array;
			return *this;
		}

        dense_bitset operator&(const dense_bitset& other) const {
            assert((size()== other.size()));
            dense_bitset ret(size());
            for (size_t i = 0; i < arrlen; ++i) {
                ret.array[i] = array[i] & other.array[i];
            }
            return ret;
        }


        dense_bitset operator|(const dense_bitset& other) const {
			assert((size() == other.size()));
            dense_bitset ret(size());
            for (size_t i = 0; i < arrlen; ++i) {
                ret.array[i] = array[i] | other.array[i];
            }
            return ret;
        }

        dense_bitset operator-(const dense_bitset& other) const {
			assert((size() == other.size()));
            dense_bitset ret(size());
            for (size_t i = 0; i < arrlen; ++i) {
                ret.array[i] = array[i] - (array[i] & other.array[i]);
            }
            return ret;
        }


        dense_bitset& operator&=(const dense_bitset& other) {
			assert((size() == other.size()));
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] &= other.array[i];
            }
            return *this;
        }


        dense_bitset& operator|=(const dense_bitset& other) {
			assert((size() == other.size()));
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] |= other.array[i];
            }
            return *this;
        }

        dense_bitset& operator-=(const dense_bitset& other) {
			assert((size() == other.size()));
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] = array[i] - (array[i] & other.array[i]);
            }
            return *this;
        }



        inline void resize(size_t n) {
            len = n;
            //need len bits
            size_t prev_arrlen = arrlen;

            arrlen = (n / (sizeof(size_t) * 8)) + (n % (sizeof(size_t) * 8) > 0);
           /* GY_CUDA_CHECK(
                cudaMallocHost((void**)&array, arrlen * sizeof(size_t)));*/
            array = (size_t*)realloc(array, sizeof(size_t) * arrlen);
            fix_trailing_bits();
            // if we grew, we need to zero all new blocks
            if (arrlen > prev_arrlen) {
                for (size_t i = prev_arrlen; i < arrlen; ++i) {
                    array[i] = 0;
                }
            }
        }


        void fix_trailing_bits() {
            // how many bits are in the last block
            size_t lastbits = len % (8 * sizeof(size_t));
            if (lastbits == 0) return;
            array[arrlen - 1] &= ((size_t(1) << lastbits) - 1);
        }

        // len
        inline size_t size() const {
            return len;
        }

        // To 0
        inline void clear() {
            for (size_t i = 0; i < arrlen; ++i) 
                array[i] = 0;
        }

		
		inline void clear_memset()
		{
			memset((void*)array, 0, sizeof(size_t) * arrlen);
		}


        
        inline bool empty() const {
            for (size_t i = 0; i < arrlen; ++i) if (array[i]) return false;
            return true;
        }

        
        inline void fill() {
            for (size_t i = 0; i < arrlen; ++i) array[i] = (size_t)-1;
            fix_trailing_bits();
        }

        /// Prefetches the word containing the bit b
        inline void prefetch(size_t b) const {
            __builtin_prefetch(&(array[b / (8 * sizeof(size_t))]));
        }

        /// Returns the value of the bit b
        inline bool get(size_t b) const {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            return array[arrpos] & (size_t(1) << size_t(bitpos));
        }


        inline bool set_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));       
            return __sync_fetch_and_or(array + arrpos, mask) & mask;//GCC
        }

        //! Atomically xors a bit with 1
        inline bool xor_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));
            return __sync_fetch_and_xor(array + arrpos, mask) & mask;
        }

        inline size_t containing_word(size_t b) {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            return array[arrpos];
        }

       
        inline void transfer_approximate_unsafe(
            dense_bitset& other,
            size_t& start,
            size_t& b
        ) {
            // must be identical in length
            assert(other.len == len);//位于logger/assertions.hpp
            assert(other.arrlen == arrlen);
            size_t arrpos, bitpos;
            bit_to_pos(start, arrpos, bitpos);
            size_t initial_arrpos = arrpos;
            if (arrpos >= arrlen) arrpos = 0;
            // ok. we will only look at arrpos
            size_t transferred = 0;
            while (transferred < b) {
                if (other.array[arrpos] > 0) {
                    transferred += __builtin_popcountl(other.array[arrpos]);
                    array[arrpos] |= other.array[arrpos];
                    other.array[arrpos] = 0;
                }
                ++arrpos;
                if (arrpos >= other.arrlen) arrpos = 0;
                else if (arrpos == initial_arrpos) break;
            }
            start = 8 * sizeof(size_t) * arrpos;
            b = transferred;
        }

        inline bool set_bit_unsync(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));
            // _sync_fetch_and_or(array + arrpos, mask) & mask;
            bool ret = array[arrpos] & mask;
            array[arrpos] |= mask;
            return ret;
        }

        inline bool set(size_t b, bool value) {
            if (value) return set_bit(b);
            else return clear_bit(b);
        }

        
        inline bool set_unsync(size_t b, bool value) {
            if (value) return set_bit_unsync(b);
            else return clear_bit_unsync(b);
        }

        
        inline bool clear_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t test_mask(size_t(1) << size_t(bitpos));
            const size_t clear_mask(~test_mask);
            return __sync_fetch_and_and(array + arrpos, clear_mask) & test_mask;
        }

        
        inline bool clear_bit_unsync(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t test_mask(size_t(1) << size_t(bitpos));
            const size_t clear_mask(~test_mask);
            bool ret = array[arrpos] & test_mask;
            array[arrpos] &= clear_mask;
            return ret;
        }

      
        

        // array 1
        size_t popcount() const {
            size_t ret = 0;
            for (size_t i = 0; i < arrlen; ++i) {
                //：https://blog.csdn.net/gaochao1900/article/details/5646211 
                ret += __builtin_popcountl(array[i]);
            }
            return ret;
        }



        void invert() {
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] = ~array[i];
            }
            fix_trailing_bits();
        }
    private:

        inline static void bit_to_pos(size_t b, size_t& arrpos, size_t& bitpos) {
            // the compiler better optimize this...
            arrpos = b / (8 * sizeof(size_t));
            bitpos = b & (8 * sizeof(size_t) - 1);
        }

       
        inline size_t next_bit_in_block(const size_t& b, const size_t& block) const {
            size_t belowselectedbit = size_t(-1) - (((size_t(1) << b) - 1) | (size_t(1) << b));
            size_t x = block & belowselectedbit;
            if (x == 0) return 0;
            else return (size_t)__builtin_ctzl(x);
        }

        inline size_t first_bit_in_block(const size_t& block) const {
            if (block == 0) return 0;
            else return (size_t)__builtin_ctzl(block);
        }

    };// end of class dense_bitset

   
    template <size_t len>
    class fixed_dense_bitset {
    public:
        /// Constructs a bitset of 0 length
        fixed_dense_bitset() {
            clear();
        }

        /// Make a copy of the bitset db
        fixed_dense_bitset(const fixed_dense_bitset<len>& db) {
            *this = db;
        }

        /** Initialize this fixed dense bitset by copying
            ceil(len/(wordlen)) words from mem
        */
        void initialize_from_mem(void* mem, size_t memlen) {
            memcpy(array, mem, memlen);//void *memcpy(void *dest, const void *src, size_t n);
        }

        /// destructor
        ~fixed_dense_bitset() {}

        /// Make a copy of the bitset db
        inline fixed_dense_bitset<len>& operator=(const fixed_dense_bitset<len>& db) {
            memcpy(array, db.array, sizeof(size_t) * arrlen);
            return *this;
        }

        /// Sets all bits to 0
        inline void clear() {
            memset((void*)array, 0, sizeof(size_t) * arrlen);
        }

        /// Sets all bits to 1
        inline void fill() {
            for (size_t i = 0; i < arrlen; ++i) array[i] = -1;
            fix_trailing_bits();
        }

        inline bool empty() const {
            for (size_t i = 0; i < arrlen; ++i) if (array[i]) return false;
            return true;
        }

        /// Prefetches the word containing the bit b
        inline void prefetch(size_t b) const {
            __builtin_prefetch(&(array[b / (8 * sizeof(size_t))]));
        }

        /// Returns the value of the bit b
        inline bool get(size_t b) const {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            return array[arrpos] & (size_t(1) << size_t(bitpos));
        }

        //! Atomically sets the bit at b to true returning the old value
        inline bool set_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));
            return __sync_fetch_and_or(array + arrpos, mask) & mask;
        }


        //! Returns the value of the word containing the bit b 
        inline size_t containing_word(size_t b) {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            return array[arrpos];
        }


        /** Set the bit at position b to true returning the old value.
            Unlike set_bit(), this uses a non-atomic set which is faster,
            but is unsafe if accessed by multiple threads.
        */
        inline bool set_bit_unsync(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));
            bool ret = array[arrpos] & mask;
            array[arrpos] |= mask;
            return ret;
        }

        /** Set the state of the bit returning the old value.
          This version uses a non-atomic set which is faster, but
          is unsafe if accessed by multiple threads.
        */
        inline bool set(size_t b, bool value) {
            if (value) return set_bit(b);
            else return clear_bit(b);
        }

        /** Set the state of the bit returning the old value.
          This version uses a non-atomic set which is faster, but
          is unsafe if accessed by multiple threads.
        */
        inline bool set_unsync(size_t b, bool value) {
            if (value) return set_bit_unsync(b);
            else return clear_bit_unsync(b);
        }


        //! Atomically set the bit at b to false returning the old value
        inline bool clear_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t test_mask(size_t(1) << size_t(bitpos));
            const size_t clear_mask(~test_mask);
            return __sync_fetch_and_and(array + arrpos, clear_mask) & test_mask;
        }

        /** Clears the state of the bit returning the old value.
          This version uses a non-atomic set which is faster, but
          is unsafe if accessed by multiple threads.
        */
        inline bool clear_bit_unsync(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t test_mask(size_t(1) << size_t(bitpos));
            const size_t clear_mask(~test_mask);
            bool ret = array[arrpos] & test_mask;
            array[arrpos] &= clear_mask;
            return ret;
        }


        struct bit_pos_iterator {
            typedef std::input_iterator_tag iterator_category;
            typedef size_t value_type;
            typedef size_t difference_type;
            typedef const size_t reference;
            typedef const size_t* pointer;
            size_t pos;
            const fixed_dense_bitset* db;
            bit_pos_iterator() :pos(0xFFFFFFFFFFFFFFFF), db(NULL) {} //pos(-1)
            bit_pos_iterator(const fixed_dense_bitset* const db, size_t pos) :pos(pos), db(db) {}

            size_t operator*() const {
                return pos;
            }
            size_t operator++() {
                if (db->next_bit(pos) == false) pos = (size_t)(-1);
                return pos;
            }
            size_t operator++(int) {
                size_t prevpos = pos;
                if (db->next_bit(pos) == false) pos = (size_t)(-1);
                return prevpos;
            }
            bool operator==(const bit_pos_iterator& other) const {
                bool same = (db == other.db);
                assert(same);
                return other.pos == pos;
            }
            bool operator!=(const bit_pos_iterator& other) const {
                assert(db == other.db);
                return other.pos != pos;
            }
        };

        typedef bit_pos_iterator iterator;
        typedef bit_pos_iterator const_iterator;


        bit_pos_iterator begin() const {
            size_t pos;
            if (first_bit(pos) == false) pos = size_t(-1);
            return bit_pos_iterator(this, pos);
        }

        bit_pos_iterator end() const {
            return bit_pos_iterator(this, (size_t)(-1));
        }

        /** Returns true with b containing the position of the
            first bit set to true.
            If such a bit does not exist, this function returns false.
        */
        inline bool first_bit(size_t& b) const {
            for (size_t i = 0; i < arrlen; ++i) {
                if (array[i]) {
                    b = (size_t)(i * (sizeof(size_t) * 8)) + first_bit_in_block(array[i]);
                    return true;
                }
            }
            return false;
        }

        /** Returns true with b containing the position of the
            first bit set to false.
            If such a bit does not exist, this function returns false.
        */
        inline bool first_zero_bit(size_t& b) const {
            for (size_t i = 0; i < arrlen; ++i) {
                if (~array[i]) {
                    b = (size_t)(i * (sizeof(size_t) * 8)) + first_bit_in_block(~array[i]);
                    return true;
                }
            }
            return false;
        }



        /** Where b is a bit index, this function will return in b,
            the position of the next bit set to true, and return true.
            If all bits after b are false, this function returns false.
        */
        inline bool next_bit(size_t& b) const {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            //try to find the next bit in this block
            bitpos = next_bit_in_block(bitpos, array[arrpos]);
            if (bitpos != 0) {
                b = (size_t)(arrpos * (sizeof(size_t) * 8)) + bitpos;
                return true;
            }
            else {
                // we have to loop through the rest of the array
                for (size_t i = arrpos + 1; i < arrlen; ++i) {
                    if (array[i]) {
                        b = (size_t)(i * (sizeof(size_t) * 8)) + first_bit_in_block(array[i]);
                        return true;
                    }
                }
            }
            return false;
        }

        ///  Returns the number of bits in this bitset
        inline size_t size() const {
            return len;
        }

        /// Serializes this bitset to an archive
        //inline void save(oarchive& oarc) const {
        //    //oarc <<len << arrlen;
        //    //if (arrlen > 0)
        //    serialize(oarc, array, arrlen * sizeof(size_t));
        //}

        ///// Deserializes this bitset from an archive
        //inline void load(iarchive& iarc) {
        //    /*size_t l;
        //    size_t arl;
        //    iarc >> l >> arl;
        //    ASSERT_EQ(l, len);
        //    ASSERT_EQ(arl, arrlen);*/
        //    //if (arrlen > 0) {
        //    deserialize(iarc, array, arrlen * sizeof(size_t));
        //    //}
        //}

        size_t popcount() const {
            size_t ret = 0;
            for (size_t i = 0; i < arrlen; ++i) {
                ret += __builtin_popcountl(array[i]);
            }
            return ret;
        }

        fixed_dense_bitset operator&(const fixed_dense_bitset& other) const {
            assert(size() == other.size());
            fixed_dense_bitset ret(size());
            for (size_t i = 0; i < arrlen; ++i) {
                ret.array[i] = array[i] & other.array[i];
            }
            return ret;
        }


        fixed_dense_bitset operator|(const fixed_dense_bitset& other) const {
            assert(size() == other.size());
            fixed_dense_bitset ret(size());
            for (size_t i = 0; i < arrlen; ++i) {
                ret.array[i] = array[i] | other.array[i];
            }
            return ret;
        }

        fixed_dense_bitset operator-(const fixed_dense_bitset& other) const {
            assert(size() == other.size());
            fixed_dense_bitset ret(size());
            for (size_t i = 0; i < arrlen; ++i) {
                ret.array[i] = array[i] - (array[i] & other.array[i]);
            }
            return ret;
        }


        fixed_dense_bitset& operator&=(const fixed_dense_bitset& other) {
            assert(size() == other.size());
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] &= other.array[i];
            }
            return *this;
        }


        fixed_dense_bitset& operator|=(const fixed_dense_bitset& other) {
            assert(size() == other.size());
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] |= other.array[i];
            }
            return *this;
        }

        fixed_dense_bitset& operator-=(const fixed_dense_bitset& other) {
            assert(size() == other.size());
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] = array[i] - (array[i] & other.array[i]);
            }
            return *this;
        }

        bool operator==(const fixed_dense_bitset& other) const {
            assert(size() == other.size());
            assert(arrlen == other.arrlen);
            bool ret = true;
            for (size_t i = 0; i < arrlen; ++i) {
                ret &= (array[i] == other.array[i]);
            }
            return ret;
        }


    private:
        inline static void bit_to_pos(size_t b, size_t& arrpos, size_t& bitpos) {
            // the compiler better optimize this...
            arrpos = b / (8 * sizeof(size_t));
            bitpos = b & (8 * sizeof(size_t) - 1);
        }


        // returns 0 on failure
        inline size_t next_bit_in_block(const size_t& b, const size_t& block) const {
            size_t belowselectedbit = size_t(-1) - (((size_t(1) << b) - 1) | (size_t(1) << b));
            size_t x = block & belowselectedbit;
            if (x == 0) return 0;
            else return (size_t)__builtin_ctzl(x);
        }

        // returns 0 on failure
        inline size_t first_bit_in_block(const size_t& block) const {
            // use CAS to set the bit
            if (block == 0) return 0;
            else return (size_t)__builtin_ctzl(block);
        }

        // clears the trailing bits in the last block which are not part
        // of the actual length of the bitset
        void fix_trailing_bits() {
            // how many bits are in the last block
            size_t lastbits = len % (8 * sizeof(size_t));
            if (lastbits == 0) return;
            array[arrlen - 1] &= ((size_t(1) << lastbits) - 1);
        }


        static const size_t arrlen;
        size_t array[len / (sizeof(size_t) * 8) + (len % (sizeof(size_t) * 8) > 0)];
    };

    template<size_t len>
    const size_t fixed_dense_bitset<len>::arrlen = len / (sizeof(size_t) * 8) + (len % (sizeof(size_t) * 8) > 0);



    

#endif // !DENSEBITSET

