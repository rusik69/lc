package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          203,
			Title:       "GPU Architecture and Parallel Computing",
			Description: "Learn about Graphics Processing Units, parallel computing models, CUDA programming, and how GPUs differ from CPUs in architecture and execution.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "GPU vs CPU Architecture",
					Content: `Graphics Processing Units (GPUs) are specialized processors designed for parallel computation, fundamentally different from CPUs in their architecture and execution model.

**Key Differences:**

**CPU Architecture:**
- Few powerful cores (4-32 cores typical)
- Optimized for sequential execution
- Large cache hierarchy (L1, L2, L3)
- Complex control logic (branch prediction, out-of-order execution)
- Low latency, high single-thread performance
- General-purpose computing

**GPU Architecture:**
- Many simple cores (thousands of cores)
- Optimized for parallel execution
- Small cache, large shared memory
- Simple control logic (in-order execution)
- High throughput, lower single-thread performance
- Specialized for parallel workloads

**GPU Design Philosophy:**
- **Throughput over Latency**: Better to execute many threads slowly than few threads quickly
- **SIMD/SIMT**: Single Instruction, Multiple Data/Threads
- **Massive Parallelism**: Thousands of threads executing simultaneously
- **Memory Bandwidth**: High bandwidth to feed many cores

**When to Use GPU:**
- Highly parallel workloads (matrix operations, image processing)
- Data-parallel algorithms
- Large datasets with independent operations
- Compute-intensive tasks (machine learning, scientific computing)

**When to Use CPU:**
- Sequential algorithms
- Complex control flow
- Small datasets
- Latency-sensitive applications`,
					CodeExamples: `// CPU: Sequential processing
void cpu_vector_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // One addition per iteration
    }
}
// Time: O(n) sequential operations

// GPU: Parallel processing (CUDA)
__global__ void gpu_vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];  // All threads execute simultaneously
    }
}
// Time: O(1) parallel operations (theoretical)

// CPU execution:
// Thread 1: c[0] = a[0] + b[0]
// Thread 1: c[1] = a[1] + b[1]
// Thread 1: c[2] = a[2] + b[2]
// ... (sequential)

// GPU execution:
// Thread 0: c[0] = a[0] + b[0]
// Thread 1: c[1] = a[1] + b[1]
// Thread 2: c[2] = a[2] + b[2]
// ... (all simultaneously)`,
				},
				{
					Title: "CUDA Programming Model",
					Content: `CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that enables developers to use GPUs for general-purpose computing.

**CUDA Execution Model:**

**Thread Hierarchy:**
- **Thread**: Smallest execution unit
- **Block**: Group of threads (up to 1024 threads)
- **Grid**: Collection of blocks
- **Warp**: 32 threads executed together (NVIDIA)

**Memory Hierarchy:**
- **Global Memory**: Large, slow, accessible by all threads
- **Shared Memory**: Fast, on-chip, shared within a block
- **Registers**: Fastest, private to each thread
- **Constant Memory**: Read-only, cached
- **Texture Memory**: Optimized for 2D/3D access patterns

**Kernel Execution:**
- **Kernel**: Function executed on GPU
- Launched with grid and block dimensions
- Each thread executes same code on different data
- Threads identified by threadIdx, blockIdx

**Synchronization:**
- **__syncthreads()**: Synchronize threads within a block
- **Atomic Operations**: Synchronize memory access
- **Memory Fences**: Ensure memory consistency`,
					CodeExamples: `// CUDA kernel example
__global__ void matrix_multiply(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host code to launch kernel
int main() {
    int N = 1024;
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    
    matrix_multiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}

// Shared memory optimization
__global__ void matrix_multiply_shared(float *A, float *B, float *C, int N) {
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + 15) / 16; tile++) {
        // Load tile into shared memory
        if (row < N && tile * 16 + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tile * 16 + threadIdx.x];
        }
        if (tile * 16 + threadIdx.y < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        }
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < 16; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}`,
				},
				{
					Title: "Warp Execution and SIMT",
					Content: `NVIDIA GPUs execute threads in groups called warps (32 threads). Understanding warp execution is crucial for GPU performance optimization.

**Warp Execution Model:**

**SIMT (Single Instruction, Multiple Threads):**
- All threads in a warp execute the same instruction
- Each thread operates on different data
- Threads can diverge (different execution paths)
- Divergence causes serialization (performance penalty)

**Warp Characteristics:**
- **Size**: 32 threads (NVIDIA), 64 threads (AMD)
- **Scheduling**: Warps scheduled independently
- **Latency Hiding**: While one warp waits, others execute
- **Occupancy**: Number of active warps per SM (Streaming Multiprocessor)

**Branch Divergence:**
- When threads in a warp take different paths
- All paths executed sequentially
- Performance penalty: up to 32× slowdown
- Minimize divergence for better performance

**Memory Coalescing:**
- Consecutive threads should access consecutive memory
- Coalesced access: Single memory transaction
- Non-coalesced access: Multiple transactions (slow)

**Occupancy Optimization:**
- Balance between threads and resources
- More threads = better latency hiding
- But limited by registers and shared memory
- Optimal occupancy: 50-75% typically`,
					CodeExamples: `// Example: Branch divergence (BAD)
__global__ void divergent_kernel(int *data, int *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] % 2 == 0) {  // Divergence!
            result[idx] = data[idx] * 2;  // Some threads take this path
        } else {
            result[idx] = data[idx] + 1;  // Others take this path
        }
    }
}
// Warp execution: Execute even path, then odd path (2× slower)

// Example: Avoiding divergence (GOOD)
__global__ void non_divergent_kernel(int *data, int *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int is_even = (data[idx] % 2 == 0);
        result[idx] = is_even * (data[idx] * 2) + 
                     (1 - is_even) * (data[idx] + 1);
    }
}
// All threads execute same path (no divergence)

// Example: Memory coalescing (GOOD)
__global__ void coalesced_access(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // Consecutive access
    }
}
// Thread 0: input[0], Thread 1: input[1], ... (coalesced)

// Example: Non-coalesced access (BAD)
__global__ void non_coalesced_access(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * 2] * 2.0f;  // Strided access
    }
}
// Thread 0: input[0], Thread 1: input[2], Thread 2: input[4] (not coalesced)

// Example: Occupancy calculation
// SM resources:
// - 65536 registers
// - 48 KB shared memory
// - 2048 threads max
// 
// Kernel requirements:
// - 32 registers per thread
// - 8 KB shared memory per block
// - 256 threads per block
//
// Register limit: 65536 / 32 = 2048 threads
// Shared memory limit: 48 KB / 8 KB = 6 blocks = 1536 threads
// Thread limit: 2048 threads
// 
// Occupancy: min(2048, 1536, 2048) = 1536 threads = 75%`,
				},
				{
					Title: "GPU Memory Hierarchy",
					Content: `GPUs have a complex memory hierarchy optimized for throughput rather than latency. Understanding this hierarchy is essential for performance.

**Memory Types:**

**1. Global Memory:**
- Largest capacity (GBs)
- Highest latency (400-800 cycles)
- High bandwidth (hundreds of GB/s)
- Cached in L2 cache
- Accessible by all threads

**2. Shared Memory:**
- On-chip, very fast (1-2 cycles)
- Limited capacity (48-164 KB per SM)
- Shared within a block
- Used for data reuse and communication
- Key to performance optimization

**3. Registers:**
- Fastest (1 cycle)
- Private to each thread
- Limited (65536 per SM)
- Spilling to local memory hurts performance

**4. Constant Memory:**
- Read-only, cached
- Broadcast to all threads
- Good for read-only parameters
- 64 KB total

**5. Texture Memory:**
- Optimized for 2D/3D access
- Cached, supports filtering
- Good for image processing

**Memory Access Patterns:**

**Coalesced Access:**
- Threads access consecutive memory locations
- Single memory transaction
- Optimal bandwidth utilization

**Bank Conflicts (Shared Memory):**
- Shared memory divided into banks
- Multiple threads accessing same bank causes serialization
- Avoid by ensuring different banks accessed`,
					CodeExamples: `// Example: Global memory access
__global__ void global_memory_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Global memory access
    }
}

// Example: Shared memory optimization
__global__ void shared_memory_kernel(float *input, float *output, int n) {
    __shared__ float tile[256];  // Shared memory
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory (coalesced)
    if (idx < n) {
        tile[tid] = input[idx];
    }
    __syncthreads();
    
    // Process from shared memory (fast)
    if (idx < n) {
        output[idx] = tile[tid] * 2.0f;
    }
}

// Example: Bank conflicts (BAD)
__shared__ float data[32];
int tid = threadIdx.x;
data[tid * 2] = value;  // All threads access even banks (conflict!)

// Example: Avoiding bank conflicts (GOOD)
__shared__ float data[33];  // Add padding
int tid = threadIdx.x;
data[tid * 2] = value;  // No conflicts (33 banks, stride 2)

// Example: Constant memory
__constant__ float PI = 3.14159f;
__constant__ float coefficients[16];

__global__ void use_constants(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = PI * coefficients[idx % 16];  // Broadcast to all threads
    }
}

// Example: Register spilling
__global__ void register_pressure(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Too many local variables cause register spilling
        float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
        // ... (spills to local memory, slow)
    }
}`,
				},
				{
					Title: "Parallel Algorithms on GPU",
					Content: `Many algorithms can be parallelized for GPU execution. Understanding parallel algorithm patterns is key to effective GPU programming.

**Common Parallel Patterns:**

**1. Map:**
- Apply function to each element independently
- Perfect for GPU (embarrassingly parallel)
- Example: Vector addition, element-wise operations

**2. Reduce:**
- Combine elements into single value
- Requires synchronization
- Tree-based reduction common
- Example: Sum, maximum, minimum

**3. Scan (Prefix Sum):**
- Compute cumulative sum
- More complex than reduce
- Used in many algorithms
- Example: Stream compaction, sorting

**4. Stencil:**
- Each element depends on neighbors
- Requires shared memory for efficiency
- Example: Image filters, finite differences

**5. Sort:**
- Bitonic sort, radix sort common on GPU
- Parallel merge sort
- Optimized for GPU architecture

**Performance Considerations:**
- Minimize global memory access
- Use shared memory for data reuse
- Avoid divergence
- Maximize occupancy
- Coalesce memory accesses`,
					CodeExamples: `// Example: Map (Vector Addition)
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // Independent operations
    }
}

// Example: Reduce (Sum)
__global__ void reduce_sum(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree-based reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Example: Stencil (2D Convolution)
__global__ void convolution_2d(float *input, float *output, 
                                float *kernel, int width, int height) {
    __shared__ float tile[18][18];  // 16x16 + 2 pixel border
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Load tile with halo
    if (x < width && y < height) {
        tile[ty + 1][tx + 1] = input[y * width + x];
    }
    // Load halo pixels
    if (tx == 0 && x > 0) {
        tile[ty + 1][0] = input[y * width + x - 1];
    }
    // ... (load other halo pixels)
    __syncthreads();
    
    // Compute convolution
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                sum += tile[ty + ky][tx + kx] * kernel[ky * 3 + kx];
            }
        }
        output[y * width + x] = sum;
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
