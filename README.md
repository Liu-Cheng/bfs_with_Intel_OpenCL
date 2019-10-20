## Brief introduction to OBFS
To obtain both the near handcrafted design performance and better software-like features such as 
portability and maintenance, we propose OBFS, an OpenCL based BFS accelerator 
on software programmable FPGAs, and explore a series of high-level 
optimizations to the OpenCL design. With the observation that OpenCL based 
FPGA design is rather inefficient on irregular memory accesses, 
we propose approaches including data alignment, graph reordering and batching
to ensure coalesced memory accesses. In addition, we take advantage
of the on-chip buffer to reduce inefficient DRAM memory accesses. 
Finally, we shift the random level update in BFS out from the main processing pipeline and 
have it overlapped with the following BFS processing task. 
According to the experiments, OBFS achieves 9.5X and 5.5X performance speedup on average compared to a 
vertex-centric implementation and an edge-centric implementation respectively 
on Intel Harp-v2. When compared to prior handcrafted designs, 
it achieves comparable or even better performance. 

## Prerequisites
* gcc4.8 or above
* Altera SDK for OpenCL 16.0.2
* Target FPGA boards Intel Xeon-FPGA (Harp-v2)

## Quick usage
* Preprare the graph data stored in edges or CSR.
* Compile *.cl code in ./src to generate FPGA bitstream of the kernel
* Compile *.cpp code in ./src
* move the script to target host machine
* run the script 

## Cite this work
This work will appear in FPT'19.
```
@inproceedings{cheng2019obfs,
title={{OBFS}:OpenCL Based BFS Optimizations on Software Programmable {FPGAs}},
author={Liu, Cheng and Chen, Xinyu and He, Bingsheng and Liao, Xiaofei and Wang, Ying and Zhang, Lei},
booktitle={Field-Programmable Technology (FPT), 2019 International Conference on},
year={2019},
organization={IEEE}
}
```

## Acknowlegement 
* We acknowledge Intel for the access to Intel Xeon+FPGA system through the Hardware Accelerator Research Program(HARP-v2).
* This work is originally done in National University of Singapore. 
