# Scaling the Cube: A Self-Implemented Maunam Framework for Cache-Optimized 2.5D Matrix Multiplication

This repository contains a C++ simulation framework designed to analyze the computational performance and scaling characteristics of **2.5D Communication-Avoiding Matrix Multiplication**. Based on the logic of the **Maunam compiler**, these simulations isolate the local computational workload from network overhead to evaluate how multi-dimensional partitioning strategies interact with modern hardware cache hierarchies.

---

## Section 1: Codebase Overview

The codebase is organized into four primary simulation files, each targeting a specific variable in high-performance matrix multiplication scaling:

### 1. `matmul.cpp` (Strong Scaling Analysis)
This is the baseline simulation used to evaluate **Strong Scaling**. It fixes the global matrix size ($N=2048$) and the replication factor ($c=2$) while increasing the processor count ($p$).
* **Logic**: It calculates local tile sizes as $N / \sqrt{p/c}$.
* **Purpose**: To demonstrate how increasing parallelism reduces tile size, eventually allowing the working set to fit within CPU caches (L1/L2/L3), often resulting in significant super-linear speedup.

### 2. `matmul_c.cpp` (Replication Factor Analysis)
This file investigates the multi-dimensional trade-off between memory replication and local computational efficiency.
* **Logic**: It varies the replication factor ($c \in \{1, 2, 4, 8\}$) across multiple processor counts.
* **Purpose**: To quantify the "computational cost" of communication avoidance. While higher $c$ reduces network bandwidth requirements in real clusters, this simulation reveals how it increases local tile sizes and can degrade performance if tiles exceed cache boundaries.

### 3. `matmul_nonsquare.cpp` (Aspect Ratio Analysis)
Real-world high-performance computing (HPC) and Deep Learning workloads often involve non-square matrices. This file tests three distinct geometries: **Square**, **Tall-Skinny**, and **Short-Wide**.
* **Logic**: Implements a rectangular `local_matmul` kernel where dimensions ($M, K, N$) are partitioned based on Maunam grid logic.
* **Purpose**: To validate how the automated partitioning logic adapts to diverse tensor shapes to maximize hierarchical cache utilization.

### 4. `matmul_weakscaling.cpp` (Weak Scaling Analysis)
This file evaluates the framework’s robustness as problem sizes and resources grow proportionally.
* **Logic**: It maintains a **constant workload per processor** (fixed $512 \times 512$ local tiles) while increasing both $p$ and the global $N$.
* **Purpose**: To measure the overhead introduced by the increasing number of systolic steps required as the virtual 3D processor grid expands.

---

## Section 2: Usage Instructions

### Prerequisites
* A C++ compiler supporting the C++11 standard or higher (e.g., `g++`, `clang++`).
* Standard math libraries (usually included with the compiler).

### Compilation
For accurate performance analysis, it is critical to compile with **high-level optimization flags** (`-O3`). This ensures the compiler enables vectorization and optimizations that simulate production-grade HPC performance.

```bash
# Compile the strong scaling baseline
g++ -O3 matmul.cpp -o strong_scaling

# Compile the replication factor study
g++ -O3 matmul_c.cpp -o c_analysis

# Compile the non-square geometry study
g++ -O3 matmul_nonsquare.cpp -o geometry_analysis

# Compile the weak scaling study
g++ -O3 matmul_weakscaling.cpp -o weak_scaling
