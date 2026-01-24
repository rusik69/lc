package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          202,
			Title:       "Advanced Processor Design",
			Description: "Explore superscalar execution, out-of-order processing, branch prediction, and multi-core architectures.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Superscalar & Out-of-Order Execution",
					Content: `Superscalar processors attempt to execute multiple instructions in a single clock cycle by having redundant hardware units (ALUs).

**1. The Out-of-Order (OoO) Flow:**
Modern CPUs don't just follow your code line-by-line; they look ahead to find instructions that are ready to run.

` + "```" + `
[ Instruction Fetch ] ──▶ [ Register Renaming ] ──┐
                                                 │
      ┌──────────────────────────────────────────┘
      ▼
[ Issue Queue ] ──▶ [ Execution Units (ALU/FPU) ] ──┐
      ▲                                           │
      └──────────(Data Ready?)────────────────────┘
                                                 ▼
[ Reorder Buffer ] ◀─────────────────────────────┘
      │
      └─▶ [ Commit in Program Order ]
` + "```" + `

**2. Key Technologies:**
*   **Instruction-Level Parallelism (ILP):** Finding independent instructions to run at once.
*   **Register Renaming:** Using extra physical registers to resolve "false" dependencies (WAR/WAW).
*   **Speculation:** Guessing the result of a branch and executing ahead. If wrong, the CPU "flushes" and restarts.

**3. The Reorder Buffer (ROB):**
Instructions finish whenever they can (Out-of-Order), but the ROB ensures they are **committed** (results written) in the exact order you wrote them.`,
					CodeExamples: `// Register Renaming Example:
// Original:
add r1, r2, r3  // Writes r1
sub r1, r4, r5  // Overwrites r1 (WAW Hazard)

// Renamed (Physical Registers p1, p2...):
add p10, r2, r3 // Maps r1 -> p10
sub p11, r4, r5 // Maps r1 -> p11 (No conflict!)`,
				},
				{
					Title: "Branch Prediction",
					Content: `Branch prediction is crucial for maintaining pipeline efficiency, as branches occur frequently and can stall the pipeline.

**Branch Prediction Strategies:**

**1. Static Prediction:**
- **Always Not Taken**: Predict branch never taken
- **Always Taken**: Predict branch always taken
- **Backward Taken, Forward Not Taken**: Heuristic based on loop behavior
- Simple but limited accuracy (~60-70%)

**2. Dynamic Prediction:**

**1-Bit Predictor:**
- Single bit: taken/not taken
- Problem: Mispredicts on alternating patterns
- Accuracy: ~70-80%

**2-Bit Saturating Counter:**
- States: 00 (strongly not taken), 01 (weakly not taken), 10 (weakly taken), 11 (strongly taken)
- More stable, better accuracy
- Accuracy: ~85-95%

**3. Correlating Predictors:**

**GShare:**
- Uses global branch history
- XOR with branch address to index predictor
- Better for correlated branches

**GSelect:**
- Concatenates branch history with branch address
- More bits for indexing

**Tournament Predictor:**
- Combines multiple predictors
- Selects best predictor per branch
- Highest accuracy (~95%+)

**Branch Target Prediction:**

**Branch Target Buffer (BTB):**
- Caches target addresses of branches
- Fast lookup for predicted taken branches

**Return Address Stack (RAS):**
- Predicts return addresses for function calls
- Stack-based structure
- Very high accuracy (~99%)`,
					CodeExamples: `// Example: 2-bit saturating counter
// States: 00, 01, 10, 11
// Transition: Taken → increment, Not Taken → decrement

// Branch behavior: T, T, T, N, T, T, T, N (pattern)
// State transitions:
// 00 → 01 (T) → 10 (T) → 11 (T) → 10 (N) → 11 (T) → 11 (T) → 11 (T) → 10 (N)

// Example: GShare predictor
// Global history: 1011 (recent branches: T, N, T, T)
// Branch address: 0x1000
// Index = history XOR address = 1011 XOR 1000 = 0011
// Use index 3 to access predictor table

// Example: Branch misprediction penalty
// Pipeline depth: 5 stages
// Misprediction detected in EX stage (stage 3)
// Penalty: 3 cycles (flush pipeline, fetch correct path)

// Example: Return address prediction
void function_a() {
    function_b();  // Call
    // ...
}

void function_b() {
    // ...
    return;  // Return address predicted from RAS
}

// RAS operation:
// Call: Push return address onto RAS
// Return: Pop address from RAS (prediction)

// Example: Branch prediction accuracy impact
// Without prediction: 20% branches, 3-cycle penalty
// CPI increase: 0.2 × 3 = 0.6 cycles per instruction

// With 90% accuracy:
// CPI increase: 0.2 × 0.1 × 3 = 0.06 cycles per instruction
// 10× improvement`,
				},
				{
					Title: "Multi-Core & Cache Coherence",
					Content: `Modern CPUs have multiple cores, each with its own local cache (L1/L2). Keeping these caches consistent is the job of **Cache Coherence Protocols**.

**1. The MESI Protocol (State Machine):**
Each cache line is in one of these four states:

` + "```" + `
[ M ] Modified:  I have the latest info; RAM is old.
[ E ] Exclusive: I'm the only one with this; it matches RAM.
[ S ] Shared:    Others have this too; it matches RAM.
[ I ] Invalid:   My copy is old/garbage; don't use it.
` + "```" + `

**2. State Transitions:**
*   If Core 1 writes to a **Shared** line, it must send an "Invalidate" signal to all other cores.
*   The line in Core 1 moves to **Modified**, and all others move to **Invalid**.

**3. Parallelism Strategies:**
*   **Task Parallelism:** Different cores run different functions (e.g., UI vs. Network).
*   **Data Parallelism (SIMD):** All cores/units run the SAME math on different data (e.g., Image Processing).
*   **False Sharing:** A performance bug where two cores fight over different variables that happen to be on the *same cache line*.`,
					CodeExamples: `// SIMD (Single Instruction Multiple Data) Example:
// Instead of adding 4 numbers one-by-one:
float a[4], b[4], c[4];
for(int i=0; i<4; i++) c[i] = a[i] + b[i];

// SIMD does it in ONE instruction:
__m128 va = _mm_load_ps(a);
__m128 vb = _mm_load_ps(b);
__m128 vc = _mm_add_ps(va, vb); // Parallel hardware add`,
				},
				{
					Title: "Performance Optimization and Measurement",
					Content: `Understanding performance metrics and optimization techniques is essential for designing efficient systems.

**Performance Metrics:**

**1. Execution Time:**
- **Wall Clock Time**: Total elapsed time
- **CPU Time**: Time CPU is executing (user + system)
- **User Time**: Time in user mode
- **System Time**: Time in kernel mode

**2. Throughput:**
- Instructions per second (IPS)
- Operations per second (OPS)
- Transactions per second (TPS)

**3. Latency:**
- Time to complete single operation
- Critical for interactive applications

**4. CPI (Cycles Per Instruction):**
- Average cycles per instruction
- Lower is better
- CPI = CPU Cycles / Instruction Count

**5. IPC (Instructions Per Cycle):**
- Inverse of CPI
- Higher is better
- IPC = 1 / CPI

**Performance Equation:**
Execution Time = Instruction Count × CPI × Clock Cycle Time

**Optimization Techniques:**

**1. Reduce Instruction Count:**
- Better algorithms
- Compiler optimizations
- Loop unrolling

**2. Reduce CPI:**
- Pipelining
- Superscalar execution
- Better branch prediction
- Cache optimization

**3. Reduce Clock Cycle Time:**
- Higher clock frequency
- Shorter pipeline stages
- Faster circuits

**Benchmarking:**

**Synthetic Benchmarks:**
- Designed to test specific aspects
- Examples: Dhrystone, Whetstone, LINPACK

**Application Benchmarks:**
- Real-world applications
- Examples: SPEC CPU, TPC-C

**Profiling:**
- Identify bottlenecks
- CPU profiling: Where time is spent
- Memory profiling: Memory usage patterns
- Cache profiling: Cache hit/miss rates`,
					CodeExamples: `// Example: Performance calculation
// Program: 1 billion instructions
// CPU: 2 GHz (2×10⁹ cycles/second)
// CPI: 1.5 cycles/instruction

// Execution time:
// CPU cycles = 1×10⁹ × 1.5 = 1.5×10⁹ cycles
// Time = 1.5×10⁹ / 2×10⁹ = 0.75 seconds

// Example: CPI improvement
// Original: CPI = 2.0
// Optimized: CPI = 1.5
// Speedup = 2.0 / 1.5 = 1.33× faster

// Example: Amdahl's Law
// If 50% of program can be parallelized:
// Speedup = 1 / (0.5 + 0.5/4) = 1 / 0.625 = 1.6×
// (with 4 cores)

// Example: Cache optimization
// Bad: Random access pattern
for (int i = 0; i < N; i++) {
    sum += array[random_index[i]];
}

// Good: Sequential access
for (int i = 0; i < N; i++) {
    sum += array[i];
}

// Example: Loop unrolling
// Original:
for (int i = 0; i < N; i++) {
    sum += array[i];
}

// Unrolled (4×):
for (int i = 0; i < N; i += 4) {
    sum += array[i] + array[i+1] + array[i+2] + array[i+3];
}
// Reduces loop overhead, better instruction-level parallelism

// Example: Branch optimization
// Bad: Unpredictable branches
if (data[i] < threshold) {  // Random pattern
    count++;
}

// Good: Predictable branches
if (i < N/2) {  // Predictable pattern
    count++;
}

// Example: Memory alignment
// Unaligned access (slow):
struct {
    char a;
    int b;  // May be unaligned
} data;

// Aligned access (fast):
struct {
    int b;  // Aligned to 4-byte boundary
    char a;
} data;`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          207,
			Title:       "Advanced Memory Systems",
			Description: "Explore advanced memory technologies: DDR architecture, NVMe SSDs, persistent memory, memory controllers, and error correction codes.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "DDR Memory Architecture",
					Content: `DDR (Double Data Rate) SDRAM is the standard for main memory in modern computers. Understanding DDR architecture is essential for system design.

**DDR Evolution:**

**DDR Generations:**
- **DDR**: 200-400 MT/s, 2.5V
- **DDR2**: 400-1066 MT/s, 1.8V
- **DDR3**: 800-2133 MT/s, 1.5V
- **DDR4**: 1600-3200 MT/s, 1.2V
- **DDR5**: 3200-6400 MT/s, 1.1V

**Key Improvements:**
- Higher data rates
- Lower voltage (power efficiency)
- Higher capacity per chip
- Better signal integrity

**DDR Architecture:**

**Double Data Rate:**
- Data transferred on both rising and falling clock edges
- Effective data rate = 2 × clock frequency
- Example: 1600 MHz clock = 3200 MT/s

**Memory Organization:**
- **Rank**: Set of chips sharing command/address bus
- **Bank**: Independent memory array within chip
- **Row**: Horizontal line in memory array
- **Column**: Vertical line in memory array

**Memory Controller:**
- Manages memory access
- Handles timing constraints
- Optimizes access patterns
- Error correction

**Timing Parameters:**

**CAS Latency (CL):**
- Column Address Strobe latency
- Time from column address to data available
- Lower is better (faster)

**RAS to CAS Delay (tRCD):**
- Row Address Strobe to CAS delay
- Time to activate row before column access

**Row Precharge Time (tRP):**
- Time to close row before opening new row

**Activate to Precharge (tRAS):**
- Minimum time row must be open`,
					CodeExamples: `// Example: DDR memory access timing
// DDR4-3200: 1600 MHz clock, CL=22, tRCD=22, tRP=22

// Sequential access (same row):
// Cycle 0: Activate row (tRCD = 22 cycles)
// Cycle 22: Read column 0 (CL = 22 cycles)
// Cycle 44: Data available
// Cycle 44: Read column 1 (CL = 22 cycles, but pipelined)
// Cycle 66: Data available
// Efficient: Row already open

// Random access (different rows):
// Cycle 0: Activate row A (tRCD = 22)
// Cycle 22: Read column (CL = 22)
// Cycle 44: Data available
// Cycle 44: Precharge row A (tRP = 22)
// Cycle 66: Activate row B (tRCD = 22)
// Cycle 88: Read column (CL = 22)
// Cycle 110: Data available
// Inefficient: Row switches required

// Example: Memory interleaving
// Distribute data across multiple channels
// Channel 0: Addresses 0x0000, 0x0008, 0x0010, ...
// Channel 1: Addresses 0x0004, 0x000C, 0x0014, ...
// Improves bandwidth by parallel access

// Example: Bank interleaving
// Distribute across banks within same rank
// Bank 0: Row 0, Row 4, Row 8, ...
// Bank 1: Row 1, Row 5, Row 9, ...
// Allows pipelining of row operations

// Example: Memory controller optimization
void memory_controller_optimize(access_t *accesses, int n) {
    // Group accesses by row
    sort_by_row(accesses, n);
    
    // Schedule accesses to minimize row switches
    for (int i = 0; i < n; i++) {
        if (same_row(accesses[i], current_row)) {
            // Same row: fast access
            schedule_access(accesses[i]);
        } else {
            // Different row: need precharge + activate
            precharge_row(current_row);
            activate_row(accesses[i].row);
            schedule_access(accesses[i]);
        }
    }
}`,
				},
				{
					Title: "NVMe and PCIe SSDs",
					Content: `NVMe (Non-Volatile Memory Express) is the interface protocol for modern SSDs, designed to leverage PCIe's high bandwidth.

**NVMe Architecture:**

**PCIe Interface:**
- Direct connection to CPU via PCIe
- No SATA bottleneck
- Multiple lanes (x1, x2, x4)
- Gen 3: 8 GT/s per lane, Gen 4: 16 GT/s, Gen 5: 32 GT/s

**NVMe Protocol:**
- Command queue-based
- Up to 65,535 queues
- Each queue up to 64K commands
- Low latency (microseconds)

**NVMe vs SATA:**

**SATA:**
- Legacy interface
- Single queue (32 commands)
- Higher latency
- Limited bandwidth (600 MB/s)

**NVMe:**
- Modern interface
- Multiple queues
- Lower latency
- High bandwidth (GB/s)

**SSD Architecture:**

**NAND Flash:**
- **SLC**: 1 bit per cell, fastest, most durable
- **MLC**: 2 bits per cell, balanced
- **TLC**: 3 bits per cell, higher capacity, lower endurance
- **QLC**: 4 bits per cell, highest capacity, lowest endurance

**Flash Controller:**
- Wear leveling
- Bad block management
- Garbage collection
- Error correction

**Performance Characteristics:**

**Sequential Read/Write:**
- High bandwidth (GB/s)
- Good for large files

**Random Read/Write:**
- High IOPS (Input/Output Operations Per Second)
- Good for databases

**Latency:**
- Read: 50-100 microseconds
- Write: 10-50 microseconds`,
					CodeExamples: `// Example: NVMe command submission (simplified)
struct nvme_command {
    uint8_t opcode;
    uint8_t flags;
    uint16_t command_id;
    uint32_t nsid;
    uint64_t metadata;
    uint64_t prp1;  // Physical Region Page
    uint64_t prp2;
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
};

// Read command
void nvme_read(uint64_t lba, uint32_t count, void *buffer) {
    struct nvme_command cmd = {0};
    cmd.opcode = 0x02;  // Read
    cmd.nsid = 1;
    cmd.cdw10 = lba & 0xFFFFFFFF;
    cmd.cdw11 = (lba >> 32) & 0xFFFFFFFF;
    cmd.cdw12 = (count - 1) & 0xFFFF;  // Number of logical blocks
    cmd.prp1 = (uint64_t)buffer;
    
    // Submit to submission queue
    submit_command(&cmd);
    
    // Wait for completion
    wait_for_completion();
}

// Example: Queue management
void nvme_init_queues(void) {
    // Create submission queue
    submission_queue = allocate_queue(1024);
    
    // Create completion queue
    completion_queue = allocate_queue(1024);
    
    // Configure queues
    nvme_admin_create_sq(submission_queue_id, submission_queue, 1024);
    nvme_admin_create_cq(completion_queue_id, completion_queue, 1024);
}

// Example: Parallel I/O
void nvme_parallel_read(uint64_t *lbas, int count, void **buffers) {
    // Submit multiple commands to different queues
    for (int i = 0; i < count; i++) {
        struct nvme_command cmd = {0};
        cmd.opcode = 0x02;
        cmd.cdw10 = lbas[i] & 0xFFFFFFFF;
        cmd.prp1 = (uint64_t)buffers[i];
        
        // Submit to queue i % num_queues
        submit_to_queue(cmd, i % num_queues);
    }
    
    // Wait for all completions
    wait_for_all_completions();
}

// Example: SSD wear leveling (concept)
void wear_leveling_write(uint64_t lba, void *data) {
    // Find physical block with least wear
    physical_block = find_least_worn_block();
    
    // Write to physical block
    write_to_block(physical_block, data);
    
    // Update mapping: LBA -> physical block
    lba_to_physical[lba] = physical_block;
    
    // Update wear count
    wear_count[physical_block]++;
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          203,
			Title:       "GPU Architecture and Parallel Computing",
			Description: "Learn about Graphics Processing Units, parallel computing models, CUDA programming, and how GPUs differ from CPUs in architecture and execution.",
			Order:       10,
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
			},
			ProblemIDs: []int{},
		},
		{
			ID:          206,
			Title:       "Hardware Security",
			Description: "Learn about hardware security threats: side-channel attacks, secure enclaves, hardware security modules, and processor vulnerabilities like Meltdown and Spectre.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Side-Channel Attacks",
					Content: `Side-channel attacks exploit physical characteristics of hardware implementations rather than algorithmic weaknesses.

**Attack Vectors:**

**Timing Attacks:**
- Measure execution time
- Infer secret information
- Example: Password comparison timing

**Power Analysis:**
- Measure power consumption
- Correlate with operations
- Simple Power Analysis (SPA): Direct observation
- Differential Power Analysis (DPA): Statistical analysis

**Electromagnetic (EM) Attacks:**
- Measure EM emissions
- Correlate with operations
- Non-invasive
- Can be done from distance

**Cache Attacks:**
- Exploit cache timing
- Infer memory access patterns
- Example: Prime+Probe, Flush+Reload

**Acoustic Attacks:**
- Measure acoustic emissions
- Correlate with operations
- Less common but possible

**Countermeasures:**

**Constant-Time Implementation:**
- Execution time independent of secret data
- Prevents timing attacks

**Power Analysis Resistance:**
- Masking: Randomize intermediate values
- Hiding: Reduce signal-to-noise ratio
- Balanced logic: Equal power consumption

**Cache Attack Mitigation:**
- Cache partitioning
- Constant-time memory access
- Cache flushing`,
					CodeExamples: `// Example: Timing attack vulnerability (VULNERABLE)
int compare_password(char *input, char *correct) {
    for (int i = 0; i < strlen(correct); i++) {
        if (input[i] != correct[i]) {
            return 0;  // Early return leaks timing information
        }
    }
    return 1;
}
// Attacker can measure time to determine password length
// and guess characters one by one

// Example: Constant-time comparison (SECURE)
int constant_time_compare(char *a, char *b, int len) {
    int result = 0;
    for (int i = 0; i < len; i++) {
        result |= a[i] ^ b[i];  // Always executes same operations
    }
    return result == 0;
}
// Execution time independent of data

// Example: Power analysis vulnerability
void aes_encrypt(uint8_t *plaintext, uint8_t *key, uint8_t *ciphertext) {
    // S-box lookup leaks power consumption
    uint8_t sbox_output = sbox[plaintext[0] ^ key[0]];
    // Power consumption correlates with sbox_output value
    // Attacker can recover key by analyzing power traces
}

// Example: Masking countermeasure
void aes_encrypt_masked(uint8_t *plaintext, uint8_t *key, uint8_t *ciphertext) {
    uint8_t mask = random_byte();
    uint8_t masked_plaintext = plaintext[0] ^ mask;
    uint8_t masked_key = key[0] ^ mask;
    
    uint8_t sbox_output = sbox[masked_plaintext ^ masked_key];
    ciphertext[0] = sbox_output ^ mask;
    // Power consumption randomized, harder to analyze
}

// Example: Cache timing attack (simplified)
void victim_function(uint8_t secret) {
    array[secret * 4096]++;  // Access depends on secret
}

void attacker_function() {
    // Flush cache
    flush_cache();
    
    // Trigger victim
    victim_function(secret);
    
    // Measure access time
    for (int i = 0; i < 256; i++) {
        uint64_t start = rdtsc();
        access(array[i * 4096]);
        uint64_t end = rdtsc();
        
        if (end - start < threshold) {
            // Cache hit - this was accessed by victim
            printf("Secret byte: %d\n", i);
        }
    }
}`,
				},
				{
					Title: "Secure Enclaves",
					Content: `Secure enclaves provide isolated execution environments protected from the rest of the system, including the operating system.

**Intel SGX (Software Guard Extensions):**

**Architecture:**
- Enclave: Protected memory region
- Enclave Page Cache (EPC): Encrypted memory
- CPU enforces access control
- Even OS cannot access enclave memory

**Features:**
- Remote attestation
- Sealed storage
- Enclave creation and destruction
- Memory encryption

**Use Cases:**
- DRM (Digital Rights Management)
- Secure computation
- Privacy-preserving analytics

**ARM TrustZone:**

**Architecture:**
- Two worlds: Secure and Normal
- CPU switches between worlds
- Secure world has access to both
- Normal world cannot access secure world

**Components:**
- TrustZone-aware peripherals
- Secure monitor
- Trusted Execution Environment (TEE)

**Use Cases:**
- Mobile payment systems
- Biometric authentication
- Secure boot

**AMD Memory Guard:**

**Architecture:**
- Secure Memory Encryption (SME)
- Secure Encrypted Virtualization (SEV)
- Memory encryption at hardware level

**Comparison:**

**SGX:**
- Fine-grained protection
- Application-level enclaves
- Requires code changes

**TrustZone:**
- Coarse-grained protection
- System-level separation
- Transparent to applications`,
					CodeExamples: `// Example: Intel SGX enclave (simplified)
// Enclave definition file (.edl)
enclave {
    trusted {
        public void enclave_function([in, size=len] uint8_t* data, size_t len);
    };
    untrusted {
        void ocall_print([in, string] const char* str);
    };
};

// Enclave code
void enclave_function(uint8_t* data, size_t len) {
    // This code runs in secure enclave
    // Memory is encrypted and protected
    // Even OS cannot access this memory
    
    // Process sensitive data
    for (size_t i = 0; i < len; i++) {
        data[i] = encrypt(data[i]);
    }
}

// Host code
int main() {
    // Create enclave
    sgx_enclave_id_t eid;
    sgx_create_enclave("enclave.signed.so", &eid);
    
    // Call enclave function
    uint8_t data[100];
    enclave_function(eid, data, sizeof(data));
    
    // Destroy enclave
    sgx_destroy_enclave(eid);
}

// Example: ARM TrustZone (simplified)
// Secure world code
void secure_world_function(void) {
    // This code runs in secure world
    // Has access to secure resources
    // Cannot be accessed from normal world
    
    // Access secure peripherals
    secure_gpio_set(1);
    
    // Return to normal world
    smc_return();
}

// Normal world code
void normal_world_function(void) {
    // This code runs in normal world
    // Cannot access secure world resources
    
    // Call secure world via SMC (Secure Monitor Call)
    smc_call(SECURE_FUNCTION_ID);
}

// Example: Remote attestation (SGX)
sgx_quote_t* get_quote(sgx_enclave_id_t eid) {
    sgx_report_t report;
    sgx_create_report(&report);
    
    // Quote proves enclave identity and integrity
    sgx_quote_t* quote = sgx_get_quote(&report);
    return quote;
}

// Verifier can check quote to verify:
// - Enclave is genuine Intel SGX enclave
// - Enclave code matches expected hash
// - Enclave is running on genuine Intel CPU`,
				},
				{
					Title: "Meltdown and Spectre Vulnerabilities",
					Content: `Meltdown and Spectre are critical processor vulnerabilities that exploit speculative execution to leak sensitive data.

**Speculative Execution:**

**What is Speculative Execution?**
- CPU executes instructions before knowing if they're needed
- Improves performance
- Results discarded if speculation wrong
- But side effects may remain

**Meltdown:**

**Vulnerability:**
- Exploits out-of-order execution
- Accesses kernel memory speculatively
- Data loaded into cache
- Cache timing reveals data

**Affected Systems:**
- Intel x86 processors
- Some ARM processors
- Not AMD (different architecture)

**Impact:**
- Read arbitrary kernel memory
- Break process isolation
- Steal passwords, keys

**Mitigation:**
- Kernel Page Table Isolation (KPTI)
- Separate page tables for kernel/user
- Performance overhead: 5-30%

**Spectre:**

**Variant 1: Bounds Check Bypass**
- Exploits conditional branch prediction
- Trains predictor to mispredict
- Accesses out-of-bounds data speculatively
- Cache timing reveals data

**Variant 2: Branch Target Injection**
- Exploits indirect branch prediction
- Poisons branch target buffer
- Redirects execution speculatively
- Accesses unauthorized data

**Affected Systems:**
- All modern processors
- Intel, AMD, ARM, IBM

**Impact:**
- Read memory from other processes
- Break sandboxing
- Steal secrets

**Mitigations:**
- Retpoline (return trampoline)
- Indirect Branch Restricted Speculation (IBRS)
- Software fixes (compiler barriers)
- Performance overhead: varies`,
					CodeExamples: `// Example: Meltdown attack (simplified concept)
uint8_t secret = kernel_memory[secret_address];  // This would normally fault
uint8_t value = array[secret * 4096];  // But speculatively executed
// Cache state reveals secret value

// Attack code (simplified):
uint64_t start, end;
uint8_t value;

// Flush cache
flush_cache();

// Access kernel memory (will fault, but speculatively executed)
try {
    value = kernel_memory[secret_address];
    array[value * 4096]++;  // Load into cache
} catch (fault) {
    // Fault handled, but cache already modified
}

// Measure access time
for (int i = 0; i < 256; i++) {
    start = rdtsc();
    access(array[i * 4096]);
    end = rdtsc();
    
    if (end - start < threshold) {
        // Cache hit - this is the secret value
        printf("Secret byte: %d\n", i);
    }
}

// Example: Spectre Variant 1
// Vulnerable code:
if (index < array_size) {
    value = array[index];  // Bounds check
    result = array2[value * 4096];  // Access
}

// Attack:
// 1. Train branch predictor: valid indices
for (int i = 0; i < 100; i++) {
    if (valid_index < array_size) {
        value = array[valid_index];
        result = array2[value * 4096];
    }
}

// 2. Mispredict with out-of-bounds index
if (malicious_index < array_size) {  // Predictor thinks true
    // Speculatively executes:
    value = array[malicious_index];  // Out of bounds!
    result = array2[value * 4096];  // Leaks data via cache
}
// Branch misprediction detected, but cache already modified

// Example: Mitigation - LFENCE (Load Fence)
if (index < array_size) {
    lfence();  // Serialize loads
    value = array[index];
    result = array2[value * 4096];
}
// Prevents speculative execution past lfence

// Example: Retpoline mitigation (Spectre Variant 2)
// Instead of indirect call:
call *rax  // Vulnerable to branch target injection

// Use retpoline:
call retpoline_setup
retpoline_target:
    pause
    jmp retpoline_target
retpoline_setup:
    mov %rax, (%rsp)
    ret  // Return to target, but speculative execution trapped in loop`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          208,
			Title:       "System Interconnects",
			Description: "Learn about system interconnect technologies: PCIe architecture, USB standards, Thunderbolt, Network-on-Chip, and bus arbitration protocols.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "PCIe Architecture and Protocol",
					Content: `PCIe (Peripheral Component Interconnect Express) is the standard high-speed interconnect for connecting peripherals to the CPU.

**PCIe Architecture:**

**Layered Architecture:**
- **Transaction Layer**: Handles transactions, TLP (Transaction Layer Packet)
- **Data Link Layer**: Error detection/correction, flow control
- **Physical Layer**: Electrical signaling, encoding

**Lane Configuration:**
- **x1**: 1 lane (250 MB/s per direction, Gen 1)
- **x2**: 2 lanes
- **x4**: 4 lanes
- **x8**: 8 lanes
- **x16**: 16 lanes (common for GPUs)

**PCIe Generations:**

**Gen 1**: 2.5 GT/s per lane (250 MB/s)
**Gen 2**: 5 GT/s per lane (500 MB/s)
**Gen 3**: 8 GT/s per lane (~985 MB/s)
**Gen 4**: 16 GT/s per lane (~1.97 GB/s)
**Gen 5**: 32 GT/s per lane (~3.94 GB/s)
**Gen 6**: 64 GT/s per lane (planned)

**Topology:**
- **Root Complex**: Connects CPU to PCIe
- **Switches**: Route traffic between devices
- **Endpoints**: PCIe devices (GPU, NIC, SSD, etc.)

**Transaction Types:**
- **Memory Read/Write**: Access device memory
- **I/O Read/Write**: Legacy I/O space
- **Configuration**: Device configuration
- **Message**: Interrupts, power management`,
					CodeExamples: `// Example: PCIe configuration space access
// Each device has 256-byte configuration space
// First 64 bytes standardized

struct pci_config_header {
    uint16_t vendor_id;
    uint16_t device_id;
    uint16_t command;
    uint16_t status;
    uint8_t revision_id;
    uint8_t class_code[3];
    uint8_t cache_line_size;
    uint8_t latency_timer;
    uint8_t header_type;
    uint8_t bist;
    uint32_t bar[6];  // Base Address Registers
    uint32_t cardbus_cis;
    uint16_t subsystem_vendor_id;
    uint16_t subsystem_id;
    uint32_t expansion_rom_base;
    uint8_t capabilities_ptr;
    uint8_t reserved[7];
    uint8_t interrupt_line;
    uint8_t interrupt_pin;
    uint8_t min_gnt;
    uint8_t max_lat;
};

// Read configuration space
uint32_t pci_read_config(uint8_t bus, uint8_t device, 
                          uint8_t function, uint8_t offset) {
    uint32_t address = (1 << 31) |           // Enable bit
                       (bus << 16) |         // Bus number
                       (device << 11) |      // Device number
                       (function << 8) |     // Function number
                       (offset & 0xFC);      // Register offset
    
    outl(0xCF8, address);  // Configuration address port
    return inl(0xCFC);      // Configuration data port
}

// Example: PCIe memory mapping
void map_pcie_device(uint8_t bus, uint8_t device) {
    // Read BAR (Base Address Register)
    uint32_t bar = pci_read_config(bus, device, 0, 0x10);
    
    // Determine BAR type
    if (bar & 1) {
        // I/O space (legacy)
        uint16_t io_base = bar & 0xFFFC;
        // Map I/O port
    } else {
        // Memory space
        uint64_t mem_base;
        if ((bar >> 1) & 1) {
            // 64-bit BAR
            uint32_t bar_high = pci_read_config(bus, device, 0, 0x14);
            mem_base = ((uint64_t)bar_high << 32) | (bar & 0xFFFFFFF0);
        } else {
            // 32-bit BAR
            mem_base = bar & 0xFFFFFFF0;
        }
        // Map to virtual address space
        void *vaddr = ioremap(mem_base, device_size);
    }
}

// Example: PCIe transaction (simplified)
struct pcie_tlp {
    uint32_t header[4];
    uint8_t data[];
};

// Memory read request
void pcie_memory_read(uint64_t address, uint32_t length) {
    struct pcie_tlp tlp = {0};
    
    // TLP header
    tlp.header[0] = (3 << 0) |        // Format: Memory Read
                    (0 << 4) |        // Type: Memory Read
                    (0 << 8) |        // T9-T10: Reserved
                    (length << 20) |  // Length in DW
                    (0 << 29) |       // Attr
                    (0 << 32);        // Requester ID
    
    tlp.header[1] = address & 0xFFFFFFFF;  // Address low
    tlp.header[2] = (address >> 32) & 0xFFFFFFFF;  // Address high
    
    // Send TLP
    send_tlp(&tlp);
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          205,
			Title:       "Quantum Computing Architecture",
			Description: "Explore quantum computing fundamentals: qubits, quantum gates, error correction, quantum processors, and hybrid quantum-classical systems.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Quantum Bits (Qubits) vs Classical Bits",
					Content: `Quantum computing uses quantum mechanical phenomena to perform computation. Understanding qubits is fundamental to quantum architecture.

**Classical Bits:**
- Two states: 0 or 1
- Deterministic: Always in one state
- Can be copied perfectly
- Measurement doesn't change state
- Independent: State of one bit doesn't affect others

**Quantum Bits (Qubits):**
- Superposition: Can be in state |0⟩, |1⟩, or both simultaneously
- Probability amplitudes: α|0⟩ + β|1⟩ where |α|² + |β|² = 1
- Measurement collapses to |0⟩ or |1⟩ probabilistically
- Cannot be copied (No-Cloning Theorem)
- Entanglement: Qubits can be correlated

**Qubit States:**

**Computational Basis:**
- |0⟩ = [1, 0]ᵀ (classical 0)
- |1⟩ = [0, 1]ᵀ (classical 1)

**Superposition States:**
- |+⟩ = (|0⟩ + |1⟩)/√2 (equal superposition)
- |-⟩ = (|0⟩ - |1⟩)/√2 (equal superposition, different phase)

**Bloch Sphere Representation:**
- Qubit state represented as point on sphere
- |0⟩ at north pole, |1⟩ at south pole
- Superposition states on equator
- Phase represented by angle

**Key Quantum Properties:**

**Superposition:**
- Qubit exists in multiple states simultaneously
- Enables parallel computation
- Lost on measurement

**Entanglement:**
- Qubits become correlated
- Measurement of one affects the other
- Enables quantum algorithms

**Interference:**
- Probability amplitudes can cancel or reinforce
- Used in quantum algorithms`,
					CodeExamples: `# Example: Qubit representation (using Qiskit)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Single qubit in superposition
qc = QuantumCircuit(1, 1)
qc.h(0)  # Apply Hadamard gate: |0⟩ → (|0⟩ + |1⟩)/√2
qc.measure(0, 0)

# Two qubits
qr = QuantumRegister(2)
cr = ClassicalRegister(2)
qc = QuantumCircuit(qr, cr)

# Create Bell state (entangled)
qc.h(0)        # |0⟩ → (|0⟩ + |1⟩)/√2
qc.cx(0, 1)    # CNOT: entangle qubits
# Result: (|00⟩ + |11⟩)/√2`,
				},
				{
					Title: "Quantum Gates and Circuits",
					Content: `Quantum gates manipulate qubit states. Understanding quantum gates is essential for building quantum algorithms.

**Single-Qubit Gates:**

**Pauli Gates:**
- **X (NOT)**: |0⟩ ↔ |1⟩
- **Y**: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
- **Z (Phase)**: |0⟩ → |0⟩, |1⟩ → -|1⟩

**Hadamard Gate (H):**
- Creates superposition: |0⟩ → (|0⟩ + |1⟩)/√2
- Essential for quantum algorithms
- Self-inverse: H² = I

**Rotation Gates:**
- **Rx(θ)**: Rotation around X-axis
- **Ry(θ)**: Rotation around Y-axis
- **Rz(θ)**: Rotation around Z-axis (phase rotation)

**Multi-Qubit Gates:**

**CNOT (Controlled-NOT):**
- Two-qubit gate
- Flips target if control is |1⟩
- Creates entanglement
- |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩

**Toffoli (CCNOT):**
- Three-qubit gate
- Universal for classical computation
- Flips target if both controls are |1⟩

**SWAP:**
- Exchanges two qubits
- |01⟩ ↔ |10⟩

**Quantum Circuits:**
- Sequence of gates applied to qubits
- Read left to right
- Measurement at end
- Reversible (except measurement)`,
					CodeExamples: `# Example: Quantum gates (Qiskit)
from qiskit import QuantumCircuit
import numpy as np

# Single-qubit gates
qc = QuantumCircuit(1)
qc.x(0)    # Pauli-X (NOT)
qc.y(0)    # Pauli-Y
qc.z(0)    # Pauli-Z (phase flip)
qc.h(0)    # Hadamard (superposition)
qc.ry(np.pi/4, 0)  # Rotation around Y-axis

# Two-qubit gates
qc = QuantumCircuit(2)
qc.cx(0, 1)  # CNOT: control=0, target=1
qc.swap(0, 1)  # SWAP qubits

# Three-qubit gates
qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)  # Toffoli: controls=0,1, target=2`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          204,
			Title:       "Embedded Systems and Microcontrollers",
			Description: "Learn about microcontroller architecture, real-time systems, interrupt handling, power management, and peripheral interfaces.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Microcontroller Architecture",
					Content: `Microcontrollers are complete computer systems on a single chip, designed for embedded applications with constraints on power, cost, and size.

**Microcontroller Components:**

**1. CPU Core:**
- Typically 8-bit, 16-bit, or 32-bit
- RISC architecture common (ARM Cortex-M, AVR, RISC-V)
- Lower clock speeds (MHz range)
- Optimized for low power

**2. Memory:**
- **Flash Memory**: Program storage (non-volatile)
- **SRAM**: Data storage (volatile)
- **EEPROM**: Small non-volatile data storage
- Limited capacity (KB to MB range)

**3. Peripherals:**
- **GPIO**: General Purpose Input/Output pins
- **Timers/Counters**: For timing and PWM
- **ADC/DAC**: Analog-to-Digital and Digital-to-Analog converters
- **Communication**: UART, SPI, I2C, CAN, USB
- **Interrupt Controller**: Manage interrupts

**4. Power Management:**
- Sleep modes for power saving
- Clock gating
- Voltage scaling

**Common Microcontroller Families:**

**ARM Cortex-M:**
- 32-bit RISC architecture
- Wide range (M0 to M7)
- Used in many embedded applications
- Examples: STM32, NXP LPC, Nordic nRF`,
					CodeExamples: `// Example: GPIO configuration (ARM Cortex-M style)
// Configure pin as output
GPIOA->MODER |= (1 << (PIN * 2));  // Set mode to output
GPIOA->OTYPER &= ~(1 << PIN);       // Push-pull output
GPIOA->OSPEEDR |= (3 << (PIN * 2)); // High speed
GPIOA->PUPDR &= ~(3 << (PIN * 2));  // No pull-up/pull-down

// Set pin high
GPIOA->BSRR = (1 << PIN);

// Set pin low
GPIOA->BSRR = (1 << (PIN + 16));`,
				},
				{
					Title: "Real-Time Systems",
					Content: `Real-time systems must respond to events within guaranteed time constraints. Understanding real-time requirements is crucial for embedded systems.

**Real-Time System Types:**

**Hard Real-Time:**
- Missing deadline causes system failure
- Safety-critical applications
- Examples: Medical devices, automotive brakes, flight control

**Soft Real-Time:**
- Missing deadline degrades performance
- Acceptable occasional delays
- Examples: Multimedia streaming, user interfaces

**Firm Real-Time:**
- Occasional deadline misses acceptable
- But must be rare
- Examples: Network protocols, some control systems`,
					CodeExamples: `// Example: Real-time task structure
typedef struct {
    uint32_t period;        // Task period (ms)
    uint32_t deadline;      // Task deadline (ms)
    uint32_t wcet;          // Worst-case execution time (ms)
    uint32_t priority;      // Task priority
    void (*task_func)(void); // Task function
} rt_task_t;`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
