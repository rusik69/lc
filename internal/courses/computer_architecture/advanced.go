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
					Title: "Superscalar Processors",
					Content: `Superscalar processors can execute multiple instructions per cycle by having multiple execution units.

**Superscalar Concepts:**

**Instruction-Level Parallelism (ILP):**
- Multiple instructions can execute simultaneously if independent
- Limited by data dependencies and control dependencies

**Execution Units:**
- **Integer ALU**: Arithmetic and logic operations
- **Floating-Point Unit (FPU)**: Floating-point operations
- **Load/Store Unit**: Memory operations
- **Branch Unit**: Control flow operations

**Superscalar Pipeline:**
- Fetch multiple instructions per cycle
- Decode and dispatch to appropriate execution units
- Execute in parallel (if no dependencies)
- Commit results in order (or out-of-order)

**Dependency Detection:**

**Data Dependencies:**
- **RAW (Read After Write)**: True dependency, must wait
- **WAR (Write After Read)**: Anti-dependency, can be handled with renaming
- **WAW (Write After Write)**: Output dependency, can be handled with renaming

**Control Dependencies:**
- Instructions after branch depend on branch outcome
- Handled with branch prediction and speculation

**Register Renaming:**
- Map architectural registers to physical registers
- Eliminates WAR and WAW dependencies
- Allows more parallelism

**Issue Policies:**

**In-Order Issue:**
- Instructions issued in program order
- Simpler but less efficient

**Out-of-Order Issue:**
- Instructions issued as soon as operands ready
- More complex but better performance
- Requires instruction window and reorder buffer`,
					CodeExamples: `// Example: Superscalar execution
// Processor: 2 integer ALUs, 1 FPU, 1 load/store unit
// Instructions:
// I1: ADD R1, R2, R3    (Integer ALU 1)
// I2: SUB R4, R5, R6    (Integer ALU 2) - parallel with I1
// I3: FADD F1, F2, F3   (FPU) - parallel with I1, I2
// I4: LOAD R7, [R8]    (Load/Store) - parallel with others
// I5: ADD R1, R1, R4    (Integer ALU) - depends on I1, I2

// Cycle 1: Issue I1, I2, I3, I4 (all independent)
// Cycle 2: Execute I1, I2, I3, I4 (in parallel)
// Cycle 3: Issue I5 (operands ready), Execute I5

// Example: Register renaming
// Original code:
ADD R1, R2, R3    // R1 = R2 + R3
SUB R4, R1, R5    // R4 = R1 - R5 (depends on R1)
ADD R1, R6, R7    // R1 = R6 + R7 (WAR hazard)

// After renaming:
ADD P1, P2, P3    // P1 = P2 + P3 (R1 → P1)
SUB P4, P1, P5    // P4 = P1 - P5 (R4 → P4, uses P1)
ADD P8, P6, P7    // P8 = P6 + P7 (R1 → P8, no conflict)

// Example: Out-of-order execution
// Instructions:
// I1: LOAD R1, [A]      // Memory access (slow)
// I2: ADD R2, R3, R4    // Independent, fast
// I3: SUB R5, R1, R6    // Depends on I1

// In-order: I1 → I2 → I3 (wait for I1)
// Out-of-order: I1 starts, I2 executes immediately, I3 waits for I1

// Example: Reorder buffer
// Maintains instruction order for commit
// Instructions execute out-of-order but commit in-order
// Ensures precise exceptions`,
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
					Title: "Multi-Core and Parallel Processing",
					Content: `Modern processors use multiple cores to achieve higher performance through parallel execution.

**Multi-Core Architecture:**

**Symmetric Multiprocessing (SMP):**
- Multiple identical cores
- Shared memory (UMA - Uniform Memory Access)
- All cores equal, any core can run any task

**Non-Uniform Memory Access (NUMA):**
- Cores have local memory
- Accessing remote memory slower
- Better scalability

**Cache Coherence:**

**Problem:**
- Multiple cores have separate caches
- Same memory location cached in multiple caches
- Need to maintain consistency

**Coherence Protocols:**

**1. MESI Protocol:**
- **Modified**: Cache has exclusive write access, data modified
- **Exclusive**: Cache has exclusive access, data unmodified
- **Shared**: Multiple caches have read-only copy
- **Invalid**: Cache line invalid or not present

**2. MOESI Protocol:**
- Adds **Owned** state
- Owned cache responsible for supplying data to other caches

**Synchronization:**

**Locks:**
- Mutual exclusion for critical sections
- Hardware support: test-and-set, compare-and-swap
- Software: mutexes, semaphores

**Memory Barriers:**
- Ensure memory operations complete in order
- Prevents reordering optimizations
- Critical for multi-threaded correctness

**Parallel Programming Models:**

**1. Shared Memory:**
- Threads share address space
- Communication via shared variables
- Synchronization needed

**2. Message Passing:**
- Processes have separate memory
- Communication via messages
- No shared state

**3. Data Parallelism:**
- Same operation on different data
- SIMD (Single Instruction, Multiple Data)
- Vector processors, GPUs`,
					CodeExamples: `// Example: Cache coherence scenario
// Core 1: Read X (loads into cache, state: Shared)
// Core 2: Read X (loads into cache, state: Shared)
// Core 1: Write X (invalidates Core 2's cache, state: Modified)
// Core 2: Read X (must get from Core 1, state: Shared)

// Example: MESI state transitions
// Core 1: Read X
//   → Cache miss, request from memory
//   → State: Exclusive (only Core 1 has it)

// Core 2: Read X
//   → Cache miss, request from memory
//   → Memory controller: Core 1 has it, change to Shared
//   → Core 1: State → Shared, send data to Core 2
//   → Core 2: State → Shared

// Core 1: Write X
//   → Invalidate other caches (Core 2)
//   → Core 1: State → Modified
//   → Core 2: State → Invalid

// Example: False sharing
// Core 1: Frequently writes to cache line containing X
// Core 2: Frequently writes to cache line containing Y
// Problem: X and Y in same cache line
// Solution: Padding to separate into different cache lines

struct {
    int x;           // Core 1 writes
    char padding[60]; // Padding to next cache line
    int y;           // Core 2 writes
} data;

// Example: Memory barrier
// Without barrier:
// Thread 1:        Thread 2:
// x = 1;           while (flag == 0);
// flag = 1;        print(x);  // May print 0!

// With barrier:
// Thread 1:        Thread 2:
// x = 1;           while (flag == 0);
// memory_barrier(); memory_barrier();
// flag = 1;        print(x);  // Always prints 1

// Example: SIMD (Single Instruction, Multiple Data)
// Scalar:
for (int i = 0; i < 4; i++) {
    c[i] = a[i] + b[i];
}

// SIMD (4 elements at once):
__m128 va = _mm_load_ps(a);
__m128 vb = _mm_load_ps(b);
__m128 vc = _mm_add_ps(va, vb);
_mm_store_ps(c, vc);`,
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
	})
}
