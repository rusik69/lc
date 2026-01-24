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
	})
}
