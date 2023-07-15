#pragma once

#include <functional>

/* ********************************************************************************
 * Func: For Sort Container，return the target index
 *       return the first index satisfy >= target 
 * Example: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14]
 *           2  3  4  5  6  7  8  9  10 11 11  11 11   12  13
 *         1.  target = 12, return 13
 *         2.  target = 11, return 9
 *         3.  target = 6,  return 4
 * ********************************************************************************/
template <typename Container_size_type, typename Element_type>
Container_size_type lower_bound(Container_size_type container_size, Element_type target, std::function<Container_size_type(Container_size_type&)> container_element)
{
    Container_size_type binSearch_start = 0;
    Container_size_type binSearch_end = container_size;
    Container_size_type binSearch_mid = 0;
    while ((binSearch_end - binSearch_start) > 0)
    {
        Container_size_type _count2 = (binSearch_end - binSearch_start) >> 1;
        binSearch_mid = binSearch_start + _count2;

        //if (chunkInfo_vec[binSearch_mid].score >= target)
        if(container_element(binSearch_mid) >= target)
        {
            binSearch_end = binSearch_mid;
        }
        else
        {
            binSearch_start = binSearch_mid + 1;
        }
    }

    return binSearch_start;
}


/**
   int size = 15;
   std::vector<int> vec;
   vec.resize(size);
   for (int i = 0; i < 10; i++)
   {
        vec[i] = i + 2;
   }
   vec[10] = 11;vec[11] = 11;vec[12] = 11;vec[13] = 13; vec[14] = 13;
   
   int target = 6;
   printf("index = %d\n", lower_bound<int, int>(size, target,
        [&](int binSearch_mid)
        { 
            return vec[binSearch_mid];
        })
    );
*/