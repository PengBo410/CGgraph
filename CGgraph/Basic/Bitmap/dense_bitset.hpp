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

        size_t* array;//数组
        size_t len;// 想要申请的len
        size_t arrlen;// 数组的长度

        template <size_t len>
        friend class fixed_dense_bitset;

        /****************************************************************************************************************************
         *                                            构造函数、析构函数与操作符                                                        *
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

        /// 复制操作
        dense_bitset(const dense_bitset& db) {
            array = NULL;
            len = 0;
            arrlen = 0;
            *this = db;
        }

        //析构器
        ~dense_bitset() {free(array); }

        //操作符=
        inline dense_bitset& operator=(const dense_bitset& db) {
            resize(db.size());//db.size()为db的len
            len = db.len;
            arrlen = db.arrlen;
            memcpy(array, db.array, sizeof(size_t) * arrlen);
            return *this;
        }

		//操作符() - 这里我们采用operate()进行引用传递
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


        // 将当前bitset的大小调整为保留n位,不会更改现有位的值
        // 如果array的长度增加，则新的bit值不会被定义
        // When shirnking, the current implementation may still leave the "deleted" bits in place which will mess up the popcount
        // 当隐藏时，当前实现可能仍然保留“已删除”位，这将扰乱popcount
        inline void resize(size_t n) {
            len = n;// 想要申请的长度
            //need len bits
            size_t prev_arrlen = arrlen;// 当前数组的长度
            // sizeof的单位是字节，而1字节=8bit，(n % (sizeof(size_t) * 8) > 0)返回true为1，false为0
            arrlen = (n / (sizeof(size_t) * 8)) + (n % (sizeof(size_t) * 8) > 0);//你想申请的len的对应的array的len
           /* GY_CUDA_CHECK(
                cudaMallocHost((void**)&array, arrlen * sizeof(size_t)));*/
            array = (size_t*)realloc(array, sizeof(size_t) * arrlen);//改变array所指内存区域的大小为（sizeof(size_t) * arrlen）长度
            // this zeros the remainder of the block after the last bit
            fix_trailing_bits();// 这会将块最后一位之后的剩余部分归零(但是到底有什么用)
            // if we grew, we need to zero all new blocks
            if (arrlen > prev_arrlen) {
                for (size_t i = prev_arrlen; i < arrlen; ++i) {
                    array[i] = 0;
                }
            }
        }

        //最后一位之后的剩余部分归零
        // 若len = 65，则lastbits = 65 % 64 = 1; arrlen = 2; array[0] = array[0] & ((1 << 1) - 1)
        // 左移x位，相当于原始乘以2的x次，所以有： array[1] = array[1] & 1;
        // ((size_t(1) << lastbits) - 1)确保了所需的lastbits位一直为1，如：lastbits=1 =>00000001(1);lastbits=2 =>00000011(3);lastbits=3 =>00000111(7);...
        // &=后，剩余位一定为0
        void fix_trailing_bits() {
            // how many bits are in the last block
            size_t lastbits = len % (8 * sizeof(size_t));
            if (lastbits == 0) return;
            array[arrlen - 1] &= ((size_t(1) << lastbits) - 1);
        }

        // 返回bitset的bit位数，即：len
        inline size_t size() const {
            return len;
        }

        // 将所有的bit位设置为0
        inline void clear() {
            for (size_t i = 0; i < arrlen; ++i) 
                array[i] = 0;
        }

		//性能比for高
		inline void clear_memset()
		{
			memset((void*)array, 0, sizeof(size_t) * arrlen);
		}


        // 判断是否所有的bit位都为0
        inline bool empty() const {
            for (size_t i = 0; i < arrlen; ++i) if (array[i]) return false;
            return true;
        }

        //将所有的bit设置为1，但不包含未使用到的位数
        inline void fill() {
            for (size_t i = 0; i < arrlen; ++i) array[i] = (size_t)-1;//将-1强转为size_t，也就是64个1，对应的十进制为：18446744073709551615
            fix_trailing_bits();//多余的位数归零
        }

        /// Prefetches the word containing the bit b
        inline void prefetch(size_t b) const {
            //__builtin_prefetch() 是 gcc 的一个内置函数。它通过对数据手工预取的方法，减少了读取延迟，从而提高了性能，但该函数也需要 CPU 的支持
            // 在 linux 内核中，经常会使用到这种预抓取技术。参考资料：https://www.cnblogs.com/dongzhiquan/p/3694858.html
            __builtin_prefetch(&(array[b / (8 * sizeof(size_t))]));
        }

        /// Returns the value of the bit b
        // 返回第b位的值(true 或 false)
        inline bool get(size_t b) const {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            //(size_t(1) << size_t(bitpos))保证了第b位一定为1，所以原数据array[arrpos]中，若第b位为1则&的结果为1,返回true，否则为0，返回false；
            return array[arrpos] & (size_t(1) << size_t(bitpos));
        }

        //! Atomically sets the bit at position b to true returning the old value
        // 原子地将位置b处的位设置为true，返回旧值
        inline bool set_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));
            //OR是或运算，A OR B的结果：当A、duB中只要有一个或者两个都为1时，结果果为1，否则为0；
            return __sync_fetch_and_or(array + arrpos, mask) & mask;//GCC里面的函数
        }

        //! Atomically xors a bit with 1
        inline bool xor_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));
            // XOR是异或运算，A XOR B的结果：当A、B两个不同时结果为1，否则为0。
            return __sync_fetch_and_xor(array + arrpos, mask) & mask;
        }

        // 返回包含第b位的word
        // word定义为特定下标的数组值，如：array[0],array[1]等
        inline size_t containing_word(size_t b) {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            return array[arrpos];
        }

        //! Returns the value of the word containing the bit b 
        // 返回原先位置处的旧值，并赋值新值0
        //inline size_t get_containing_word_and_zero(size_t b) {
        //    size_t arrpos, bitpos;
        //    bit_to_pos(b, arrpos, bitpos);
        //    // 本质为：type __sync_lock_test_and_set (type *ptr, type value, ...)//将*ptr设为value并返回*ptr操作之前的值。
        //    return fetch_and_store(array[arrpos], size_t(0));//将array[arrpos]原子的设置为0(位于parallel/atomic_ops.hpp中)
        //}

        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 【Func】：
        //       将大约b位bit从[other]bitset传输到此bitset，从给定[start]开始，将至少[b]位从另一个bitset“移动”到此bitset
        //       从语义上讲，这会实现如下功能：
        // 【举例】：
        //       idx = start;
        //       if (other.get_bit(idx) == false) {
        //          idx = next true bit after idx in other(with loop around)
        //       }
        //       for (transferred = 0; transferred < b; transferred++) {
        //          other.clear_bit(idx);
        //          this->set_bit(idx);
        //          idx = next true bit after idx in other.
        //          if no more bits, return
        //       }
        //       注意：然而，这里的实现可以传送超过b比特，( up to b + 2 * wordsize_in_bits )
        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

        // 将setbit的b位设置为true，并返回旧值。不同于set_bit()的是，此方法不使用原子操作，因此更快，但不够安全（可能存在多线程修改）
        inline bool set_bit_unsync(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos));
            // _sync_fetch_and_or(array + arrpos, mask) & mask;
            bool ret = array[arrpos] & mask;//TODO：为什么相对于set_bit()是先&，在|=
            array[arrpos] |= mask;
            return ret;
        }

        // 将setbit的b位设置为新值[value]，并返回旧值。是线程安全的
        inline bool set(size_t b, bool value) {
            if (value) return set_bit(b);
            else return clear_bit(b);
        }

        // 将setbit的b位设置为新值[value]，并返回旧值。不同于set()的是，此方法不使用原子操作，因此更快，但不够安全（可能存在多线程修改）
        inline bool set_unsync(size_t b, bool value) {
            if (value) return set_bit_unsync(b);
            else return clear_bit_unsync(b);
        }

        // 原子的将setbit的b位设置为false，并返回旧值
        inline bool clear_bit(size_t b) {
            // use CAS to set the bit
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t test_mask(size_t(1) << size_t(bitpos));
            const size_t clear_mask(~test_mask);
            return __sync_fetch_and_and(array + arrpos, clear_mask) & test_mask;
        }

        // 将setbit的b位设置为false，并返回旧值。不同于clear_bit()的是，此方法不使用原子操作，因此更快，但不够安全（可能存在多线程修改）
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

        // 定义bitset迭代器
        struct bit_pos_iterator {
            //在STL体系下定义iterator，要满足规定的一些规范：iterator_category, 有5中分类，决定胃具体的操作，如：++，--，！ = ， == 等，归纳为：
            //①. Input_iterator : 只读，不写
            //②. output_iterator : 只写，不读
            //③. forward_iterator : 具备向前读写
            //④. bidirectional_iterator : 具备向前，向后读写
            //⑤. random_access_iterator : 提供算数能力  ，如 += ， -=
            typedef std::input_iterator_tag iterator_category;//只读不写
            typedef size_t value_type;// 模版类型
            typedef size_t difference_type;// 一般为ptrdiff_t（系统定义好的）
            typedef const size_t reference;// 模版类型的引用类型
            typedef const size_t* pointer;// 模版类型的指针类型
            // 参考资料：https://www.cnblogs.com/ypdxcn/p/9685951.html

            size_t pos;//迭代的位置
            const dense_bitset* db;//要迭代的bitset
            bit_pos_iterator() :pos(0xFFFFFFFFFFFFFFFF), db(NULL) {} //pos(-1)
            bit_pos_iterator(const dense_bitset* const db, size_t pos) :pos(pos), db(db) {}

            //操作符*，返回当前的迭代位置
            size_t operator*() const {
                return pos;
            }
            //操作符*，从当前pos寻找下一个为1的pos，并返回。若没有则pos变为-1
            size_t operator++() {
                if (db->next_bit(pos) == false) pos = (size_t)(-1);//如果b之后，没有为1的位，则返回false，且pos设置为-1
                return pos;//否则返回下一个为true的位（pos已经更新）
            }
            size_t operator++(int) {
                size_t prevpos = pos;
                if (db->next_bit(pos) == false) pos = (size_t)(-1);
                return prevpos;
            }
            // 操作符==,判断两个bitset的pos是否相同，相同返回true，不同返回false，但前提是两个bitset要相等
            bool operator==(const bit_pos_iterator& other) const {
                //写成ASSERT_TRUE(db == other.db);会报错，不知道为什么
                bool is = ((other.db) == (db));
                assert(is);
                return other.pos == pos;
            }
            // 操作符！=,判断两个bitset的pos是否不等，不等返回true，相等返回false，但前提是两个bitset要相等
            bool operator!=(const bit_pos_iterator& other) const {
                //写成ASSERT_TRUE(db == other.db);会报错，不知道为什么
                assert((bool)(db == other.db));
                return other.pos != pos;
            }
        };// end of bit_pos_iterator

        typedef bit_pos_iterator iterator;
        typedef bit_pos_iterator const_iterator;

        // 返回array中，第一个为1的bit位置，作为迭代的begin(),并返回true；如果整个array都为0，则返回false
        bit_pos_iterator begin() const {
            size_t pos;
            if (first_bit(pos) == false) pos = size_t(-1);
            return bit_pos_iterator(this, pos);
        }

        // 迭代的end()
        bit_pos_iterator end() const {
            return bit_pos_iterator(this, (size_t)(-1));
        }

        // 返回array中，第一个为1的bit位置，作为迭代的begin(),并返回true；如果整个array都为0，则返回false
        inline bool first_bit(size_t& b) const {
            for (size_t i = 0; i < arrlen; ++i) {
                if (array[i]) {
                    b = (size_t)(i * (sizeof(size_t) * 8)) + first_bit_in_block(array[i]);
                    return true;
                }
            }
            return false;
        }

        // 返回array中，第一个为0的bit位置，并返回true；如果整个array都为1，则返回false
        inline bool first_zero_bit(size_t& b) const {
            for (size_t i = 0; i < arrlen; ++i) {
                if (~array[i]) {
                    b = (size_t)(i * (sizeof(size_t) * 8)) + first_bit_in_block(~array[i]);
                    return true;
                }
            }
            return false;
        }

        // 找出b位后，为1的下一个位置的索引号，如果b之后都为0，则函数返回false
        inline bool next_bit(size_t& b) const {
            size_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            //try to find the next bit in this block
            bitpos = next_bit_in_block(bitpos, array[arrpos]);//在当前block中尝试寻找为true的位(block是指array[i])
            if (bitpos != 0) {
                b = (size_t)(arrpos * (sizeof(size_t) * 8)) + bitpos;//返回下一个为1的bit位置，注：要返回的是整个array中的位置，bitpos只是相对于当前block的位置
                return true;
            }
            // 当前block中bitpos后没有为1的位置，则查询下一个block
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

        //未导入序列化的文件，此处先注释掉
        /// Serializes this bitset to an archive
        /*inline void save(oarchive& oarc) const {
            oarc << len << arrlen;
            if (arrlen > 0) serialize(oarc, array, arrlen * sizeof(size_t));
        }*/

        /// Deserializes this bitset from an archive
        /*inline void load(iarchive& iarc) {
            if (array != NULL) free(array);
            array = NULL;
            iarc >> len >> arrlen;
            if (arrlen > 0) {
                array = (size_t*)malloc(arrlen * sizeof(size_t));
                deserialize(iarc, array, arrlen * sizeof(size_t));
            }
        }*/

        // array中所有为1的总数
        size_t popcount() const {
            size_t ret = 0;
            for (size_t i = 0; i < arrlen; ++i) {
                //参考资料：https://blog.csdn.net/gaochao1900/article/details/5646211 (VC上也可以实现__builtin_popcountl)
                ret += __builtin_popcountl(array[i]);//计算一个 64 位无符号整数有多少个位为1（_builtin_popcount()是计算32位的）
            }
            return ret;
        }




        //反转，bitset中，1变成0,0变成1
        void invert() {
            for (size_t i = 0; i < arrlen; ++i) {
                array[i] = ~array[i];
            }
            fix_trailing_bits();
        }
    private:

        inline static void bit_to_pos(size_t b, size_t& arrpos, size_t& bitpos) {
            // the compiler better optimize this...
            arrpos = b / (8 * sizeof(size_t));//当前bit位在数组中的位置，如：65位在数组的array[1]中
            bitpos = b & (8 * sizeof(size_t) - 1);//相当于bitpos = b % (sizeof(size_t)),也就是在对应的数组中的位数
        }

        // 返回block中从b位开始下一个为1的索引号，函数返回0说明：block中b位后没有为1的位置
        // 【举例】：
        //        block = 0000000000000000000000000000000000000000000000000000000001100100, b = 10 时； 函数的返回值为0
        //        block = 0000000000000000000000000000000000000000000000000000000001100100, b = 3 时； 函数的返回值为5
        inline size_t next_bit_in_block(const size_t& b, const size_t& block) const {
            size_t belowselectedbit = size_t(-1) - (((size_t(1) << b) - 1) | (size_t(1) << b));
            size_t x = block & belowselectedbit;
            if (x == 0) return 0;
            //处理二进制的内置函数：__builtin_ctzl(),返回后面的0的个数（从左往右），也就是第一个为1的位置（二进制是从右往左）
            //如： block = 0000000000000000000000000000000000000000000000000000000001100100, (size_t)__builtin_ctzl(block)=2
            else return (size_t)__builtin_ctzl(x);
        }

        // 返回当前block中第一个为1的位置索引
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



    

//}//end of namespace GY
#endif // !DENSEBITSET

