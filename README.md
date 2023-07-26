# CGgraph

CGgraph: An Ultra-fast Graph Processing System on Modern Commodity CPU-GPU Co-processor

Implementations of four graph analytics applications (PageRank, Weakly Connected Components, Single-Source Shortest Paths, Breadth-First Search).

# Quick Start
Before building CGgraph make sure you have CUDA,TBB,CUB and C++17 installed on your system. 

In this version, You need to specify the name of the dataset you want to run (the grapg file path needs to be set in the ./CGgraph/CGgraph/Basic/Graph/graphFileList.hpp), algorithm name, and additional parameters in the ./CGgraph/CGgraph/my_project/CGgraph/main_partition.hpp file.

To build:
```cpp
make
```

For example:
```cpp
std::string graphName = "twitter2010";  //graphName
OrderMethod orderMethod = OrderMethod::NATIVE;  //orderMethod
int64_t root = 0;                               // root
count_type runs = 10;
bool has_csrFile = 1;
bool logResult = true;
bool rootTranfer = true;
bool RCM_LAST_ZERO_OUTDEGREE_check = true;
Algorithm_type algorithm = Algorithm_type::SSSP;
Engine_type  engine_type = Engine_type::COORDINATION;
int useDeviceNum = 1;
```
