Breadth First Search (BFS) is a key building block of graph processing
and there have been considerable efforts devoted to accelerating BFS on FPGAs
for the sake of both performance and energy efficiency. While prior work
typically built the BFS accelerator through handcrafted circuit using
hardware description language (HDL). Despite the relatively good performance,
the HDL based design leads to extremely low design productivity, and incurs
high portability and maintenance cost. While the evolving high level synthesis (HLS)
tools make it convenient to create a functional correct BFS accelerator,
the performance of the baseline design remains much lower.

To obtain both the near hand-crafted design performance and the software-like features,
we explored the BFS accelerator design and optimization using Intel OpenCL on software
programmable FPGAs. On the one hand, we improved the memoy access efficiency by reducing
and offloading the irregular memory accesses that dramatically affect the BFS performance.
On the other hand, we reorganized
and simplified the data path parallelization with a large distributed bitmap
and edge batching, which are preferable to the OpenCL compilation tools to generate
efficient hardware. According to the experiments on a set of representative graphs,
the proposed OpenCL based BFS accelerator achieves up to 9X performance speedup
compared to the reference design in Spector benchmark on Intel Harp v2.
When compared to prior handcrafted design on similar FPGA cards, it achieves
comparable performance or even better on some R-MAT graphs.
