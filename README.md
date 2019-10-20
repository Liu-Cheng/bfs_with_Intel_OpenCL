Breadth First Search (BFS) is a key building block of graph processing 
and there have been considerable efforts devoted to accelerating BFS on FPGAs
for the sake of both performance and energy efficiency. Prior work 
typically built the BFS accelerator through handcrafted circuit design using 
hardware description language (HDL). Despite the relatively good performance, 
the HDL based design leads to extremely low design productivity, and incurs 
high portability and maintenance cost. While high level synthesis (HLS) 
tools make it convenient to create a functionally correct BFS accelerator, 
the performance can be much lower the handcrafted design with HDL. 

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
