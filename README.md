# CGgraph

CGgraph: An Ultra-fast Graph Processing System on Modern Commodity CPU-GPU Co-processor

# Quick Start
Before building CGgraph make sure you have CUDA,TBB,CUB and C++17 installed on your system. 

Implementations of four graph analytics applications (PageRank, Weakly Connected Components, Single-Source Shortest Paths, Breadth-First Search).

In this version, You need to specify the name of the dataset you want to run (the grapg file path needs to be set in the ./CGgraph/CGgraph/Basic/Graph/graphFileList.hpp), algorithm name, and additional parameters in the ./CGgraph/CGgraph/my_project/CGgraph/main_partition.hpp file.

To build:

make
