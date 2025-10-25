# Matrix-Multiplication-with-CUDA
Using CUDA to optimize matrix multiplication (on google colab)

# Introduction
The intent I had with this project was to get familiar with programming using the cuda framework. I also ended up comparing the benefits of multi-threading by comparing the execution times of the CUDA kernel and a straightofward triple-loop approach on the CPU for matrix multiplication. I didn't have direct access to an Nvidia GPU when creating this, so I used Google Colab. Coding C++ on colab can be annoying, but it works, and I'll walk you through making sure it works on your end as well! 

# How does parallel processing work (how the data moves between the CPU and GPU)
In CUDA, threads are grouped into blocks, and blocks form a grid (here a 2D grid) that covers the $N \times N$ output. Each thread in a block computes one cell $c_{ij}$, using its 2D indices inside the block and the block’s 2D position in the grid. This hierarchy matters because threads within the same block can synchronize and share fast shared memory on the Streaming Multiprocessor (SM), while all blocks see the slower global memory (device DRAM). Each thread also has registers—its fastest private storage—for loop counters and intermediate sums.

By default (CPU) threads read operands directly from global memory each time they need them, which is bandwidth-heavy. So, a common upgrade is tiling: threads in a block cooperatively load small tiles of $A$ and $B$ from global memory into shared memory (each load goes through thread registers on the way), synchronize, and then perform many multiply–adds using those tiles, with values moved from shared memory $\rightarrow$ registers for the arithmetic. Because each loaded element is reused by many threads (e.g., up to $B_x$ or $B_y$ times in a $B_x \times B_y$ block), the expensive global-memory traffic is amortized over a lot of math. That reuse—plus thousands of threads running at once—is the real win of GPU multithreading: far fewer slow global reads per floating-point operation, so time scales with useful work instead of memory latency.

This is similar but different from other parallel processing frameworks, like Metal (from Apple). With Metal’s shared physical RAM (e.g., MTLStorageModeShared), the CPU and GPU truly access the same bytes, so there’s no page migration or hidden PCIe copies - you get lower latency for small/streaming workloads, smaller memory footprints, and simpler sync via command-buffer completion/fences. In contrast, cudaMallocManaged unifies virtual addresses but still migrates pages between separate CPU RAM and GPU VRAM, which can cause page-fault stalls unless you prefetch/advise carefully.


# Approach (Cell by Cell)
1. Get CUDA working in colab environment (Cells 1-2) -> downloaded nvcc4jupyiter module and detected the GPU
2. Build and run a CUDA kernel in the notebook (Cell 3)
   Familiarizing myself with the CUDA framework and its method by running matrix multiplication one time. I also compared it to the standard CPU approach to verify results.
3. Benchmark both across sizes N = 2^0 ... 2^11 (Cell 4)
    Redefined the kernel. Also created 2 separate functions - one to carry out the process on the GPU and return the time, and one to carry out the process on the CPU and return the        time. Called these functions, and printed out the times w/ results.
4. Plot results (MatPlotLib)

# How the Kernel Works
1. Each CUDA thread computes one output element $c_{ij}$, with indices mapped by $i=\text{blockIdx}_y\cdot B_y+\text{threadIdx}_y$ and $j=\text{blockIdx}_x\cdot B_x+\text{threadIdx}_x$
2. For valid $(i,j)$, the thread performs the dot product $c_{ij}=\sum_{k=0}^{N-1} a_{ik}\,b_{kj}$.
3. A bounds check $(i<N \land j<N)$ prevents out-of-range threads from writing, ensuring correctness at matrix edges.
4. The launch uses a 2D grid so $\lceil N/B_x \rceil \times \lceil N/B_y \rceil$ blocks with $B_x \times B_y$ threads cover all $(i,j)$ (e.g., $16\times16$).
5. Work per thread is $O(N)$, giving $O(N^3)$ overall; memory is row-major for $A$ and column-wise for $B$ (improvable with shared-memory tiling).

# So, Benefits of Multi Threading 
Everyone knows this but I think these graphs are cool lol
<img width="1774" height="1173" alt="image" src="https://github.com/user-attachments/assets/f58eb26d-4c84-4af3-a4cd-562c9c489e6e" />
<img width="1774" height="974" alt="image" src="https://github.com/user-attachments/assets/cbb51e09-59f7-4a11-97d3-c61907b58e0e" />

## What the graphs show

These figures compare end-to-end execution time of a naïve CPU matrix multiply against a CUDA GPU kernel across sizes $N=2^0 \ldots 2^{11}$ (top), and the resulting speedup $\text{CPU time} / \text{GPU time}$ (bottom). The GPU overtakes the CPU as $N$ grows, with speedup accelerating once kernel launch overheads and memory latency are amortized.

## Why GPU graph looks ~linear

On the GPU, we launch one thread per output cell of C (that’s about N^2 threads), and the hardware runs huge batches of those threads at the same time. Because so many threads run in parallel, the N^2 part is mostly “flattened out,” and the time you feel is mainly each thread’s inner loop of length N — so the curve looks roughly linear in N. The exceptions for this are when for tiny N, fixed overheads (kernel launch, setup) dominate and hide this trend; for very large N, the line can bend upward again when you run out of parallel capacity or memory bandwidth, so not all threads can progress at full speed.

# To run the notebook:
1. Change runtime type to NVIDIA GPU
2. Click Run all
