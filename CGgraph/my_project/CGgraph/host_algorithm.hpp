#pragma once

#include "Basic/basic_include.cuh"

#define ACTIVE_BLOCK

namespace Gemini_atomic{

	template <class T>
	inline bool cas(T* ptr, T old_val, T new_val) {
		if (sizeof(T) == 8) {
			return __sync_bool_compare_and_swap((long*)ptr, *((long*)&old_val), *((long*)&new_val));
		}
		else if (sizeof(T) == 4) {
			return __sync_bool_compare_and_swap((int*)ptr, *((int*)&old_val), *((int*)&new_val));
		}
		else {
			assert(false);
			return static_cast<bool>(0);
		}
	}

	template <class T>
	inline bool write_min(T* ptr, T val) {
		volatile T curr_val; bool done = false;
#ifdef LOCAL_THREAD_DEBUG
		size_t count = 0;
#endif
		do {
			curr_val = *ptr;
#ifdef LOCAL_THREAD_DEBUG
			count += 1;
#endif			
		} while (curr_val > val && !(done = cas(ptr, curr_val, val)));
#ifdef LOCAL_THREAD_DEBUG
		THREAD_LOCAL_pushConflictCount_single_max = cpj_max(THREAD_LOCAL_pushConflictCount_single_max, count);
#endif	
		return done;
	}

	template <class ET>
	inline bool write_max(ET* a, ET b) {
		ET c;
		bool r = 0;
		do c = *a;
		while (c < b && !(r = cas(a, c, b)));
		return r;
	}


	template <class T>
	inline void write_add(T* ptr, T val) {
		volatile T new_val, old_val;
		do {
			old_val = *ptr;
			new_val = old_val + val;
		} while (!cas(ptr, old_val, new_val));
	}


	
	template <class T>
	inline bool atomic_large(T* ptr, T val)
	{
		volatile T curr_val; bool done = false;

		do
		{
			curr_val = *ptr;
			done = curr_val > val;
		} while (!cas(ptr, curr_val, curr_val));

		return done;		
	}


	
	template <class T>
	inline bool atomic_largeEqu(T* ptr, T val)
	{
		volatile T curr_val; bool done = false;

		do
		{
			curr_val = *ptr;
			done = curr_val >= val;
		} while (!cas(ptr, curr_val, curr_val));

		return done;		
	}


	
	template <class T>
	inline bool atomic_smallEqu(T* ptr, T val)
	{
		volatile T curr_val; bool done = false;

		do
		{
			curr_val = *ptr;
			done = curr_val <= val;
		} while (!cas(ptr, curr_val, curr_val));

		return done;		
	}

	
	template <class T>
	inline int64_t atomic_length(T* ptr, T val)
	{
		volatile T curr_val; int64_t length = 0;

		do
		{
			curr_val = *ptr;
			length = static_cast<int64_t>(curr_val) - static_cast<int64_t>(val);
		} while (!cas(ptr, curr_val, curr_val));

		return length;		
	}



}







/* ************************************************************************************************
 *                                       [BFS]
 * ************************************************************************************************/
namespace BFS_SPACE
{
    
    template<typename GraphEnginer>
	count_type bfs_numa_lastzero(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		count_type local_activeVertices = 0;
		vertex_data_type level = graphEnginer.vertexValue[vertexId + graphEnginer.zeroOffset_numa[socketId]] + 1;

		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];

			if (level < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], level)) {
                    if (dest < graphEnginer.noZeroOutDegreeNum)
                    {
                        graphEnginer.active.out().set_bit(dest);
                        local_activeVertices += 1;
                    }											
				}
			}
		}
		return local_activeVertices;
	}

    template<typename GraphEnginer>
	count_type bfs_numa(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		count_type local_activeVertices = 0;
		vertex_data_type level = graphEnginer.vertexValue[vertexId + graphEnginer.zeroOffset_numa[socketId]] + 1;

		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];

			if (level < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], level)) {
                    graphEnginer.active.out().set_bit(dest);
                    local_activeVertices += 1;											
				}
			}
		}
		return local_activeVertices;
	}



	// @param [vertexId] 无NUMA偏移
	// @param [socketId] vertexId所在的SocketId
	template<typename GraphEnginer>
	void bfs_numa_steal(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		vertex_data_type srcVertexValue = graphEnginer.vertexValue[vertexId];
		vertex_data_type level = srcVertexValue + 1;

		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];
			
			if (level < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], level)) {
					graphEnginer.active.out().set_bit(dest);					
				}
			}
		}
	}




	// NO NUMA
	template<typename Graph>
	count_type bfs_noNuma(Graph& graph, vertex_id_type vertexId, offset_type nbr_start, offset_type nbr_end)
	{
		count_type local_activeVertices = 0;
		vertex_data_type level = graph.vertexValue[vertexId] + 1;

		for (offset_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graph.csr_dest[nbr_cur];

			if (level < graph.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graph.vertexValue[dest], level)) {
					graph.active.out().set_bit(dest);		
					local_activeVertices += 1;
				}
			}
		}

		return local_activeVertices;
	}


} // end of namespace [BFS_SPACE]






/* ************************************************************************************************
 *                                       [SSSP]
 * ************************************************************************************************/
namespace SSSP_SPACE
{
    template<typename GraphEnginer>
	count_type sssp_numa_lastzero(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		count_type local_activeVertices = 0;
		vertex_data_type srcVertexValue = graphEnginer.vertexValue[vertexId + graphEnginer.zeroOffset_numa[socketId]];

		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];
			edge_data_type weight = graphEnginer.csr_weight_numa[socketId][nbr_cur];

			vertex_data_type distance = srcVertexValue + weight;

			if (distance < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], distance)) {
					if (dest < graphEnginer.noZeroOutDegreeNum) //加一句简单的话就会让性能下降，所以最好用sssp_numa
					{
						graphEnginer.active.out().set_bit(dest);
						local_activeVertices += 1;
					}						
				}
			}
		}
		return local_activeVertices;
	}

    template<typename GraphEnginer>
	count_type sssp_numa(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		count_type local_activeVertices = 0;
		vertex_data_type srcVertexValue = graphEnginer.vertexValue[vertexId + graphEnginer.zeroOffset_numa[socketId]];

		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];
			edge_data_type weight = graphEnginer.csr_weight_numa[socketId][nbr_cur];

			vertex_data_type distance = srcVertexValue + weight;

			if (distance < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], distance)) {
					graphEnginer.active.out().set_bit(dest);
					local_activeVertices += 1;						
				}
			}
		}
		return local_activeVertices;
	}


	// @param [vertexId] 无NUMA偏移
	// @param [socketId] vertexId所在的SocketId
	template<typename GraphEnginer>
	void sssp_numa_steal(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		vertex_data_type srcVertexValue = graphEnginer.vertexValue[vertexId];
		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];
			edge_data_type weight = graphEnginer.csr_weight_numa[socketId][nbr_cur];
			vertex_data_type distance = srcVertexValue + weight;

			if (distance < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], distance)) {
					graphEnginer.active.out().set_bit(dest);					
				}
			}
		}
	}



	// NO NUMA
	template<typename Graph>
	count_type sssp_noNuma(Graph& graph, vertex_id_type vertexId, offset_type nbr_start, offset_type nbr_end)
	{
		count_type local_activeVertices = 0;
		vertex_data_type srcVertexValue = graph.vertexValue[vertexId];

		#ifdef ACTIVE_BLOCK
		count_type block_x = 0;
		if(graph.ite >=3 && graph.ite<=5) block_x = vertexId / graph.segmentSize;	
		#endif

		for (offset_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graph.csr_dest[nbr_cur];
			edge_data_type weight = graph.csr_weight[nbr_cur];
			vertex_data_type distance = srcVertexValue + weight;

			#ifdef ACTIVE_BLOCK
			if(graph.ite >=3 && graph.ite<=5)
			{
				count_type block_y = dest / graph.segmentSize;
				__sync_fetch_and_add(&graph.blockDis[block_x][block_y], 1);
			}
			
			#endif

			if (distance < graph.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graph.vertexValue[dest], distance)) {
					graph.active.out().set_bit(dest);		
					local_activeVertices += 1;
				}
			}
		}

		return local_activeVertices;
	}





}// end of namespace [SSSP_SPACE]





/* ************************************************************************************************
 *                                       [CC]
 * ************************************************************************************************/
namespace CC_SPACE
{

    template<typename GraphEnginer>
	count_type cc_numa_lastzero(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		count_type local_activeVertices = 0;
		vertex_data_type level = graphEnginer.vertexValue[vertexId + graphEnginer.zeroOffset_numa[socketId]];

		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];

			if (level < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], level)) {
                    if (dest < graphEnginer.noZeroOutDegreeNum)
                    {
                        graphEnginer.active.out().set_bit(dest);		
					    local_activeVertices += 1;
                    }			
				}
			}
		}

		return local_activeVertices;
	}





    template<typename GraphEnginer>
	count_type cc_numa(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		count_type local_activeVertices = 0;
		vertex_data_type level = graphEnginer.vertexValue[vertexId + graphEnginer.zeroOffset_numa[socketId]];

		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];

			if (level < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], level)) {
					graphEnginer.active.out().set_bit(dest);		
					local_activeVertices += 1;
				}
			}
		}

		return local_activeVertices;
	}


	// @param [vertexId] 无NUMA偏移
	// @param [socketId] vertexId所在的SocketId
	template<typename GraphEnginer>
	void wcc_numa_steal(GraphEnginer& graphEnginer, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end,
		count_type socketId, bool sameSocket)
	{
		vertex_data_type srcVertexValue = graphEnginer.vertexValue[vertexId];
		
		for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
		{
			//获取dest的Offset
			vertex_id_type dest = graphEnginer.csr_dest_numa[socketId][nbr_cur];
			
			if (srcVertexValue < graphEnginer.vertexValue[dest]) {
				if (Gemini_atomic::write_min(&graphEnginer.vertexValue[dest], srcVertexValue)) {
					graphEnginer.active.out().set_bit(dest);					
				}
			}
		}
	}

}// end of namespace [CC_SPACE]