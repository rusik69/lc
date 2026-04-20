package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2835,
			Title:       "GPU Architecture and Compute",
			Description: "Deep dive into GPU microarchitecture, SIMT execution, memory hierarchy, warp scheduling, and general-purpose GPU computing (GPGPU).",
			Order:       35,
			Lessons: []problems.Lesson{
				{
					Title: "GPU Microarchitecture",
					Content: `GPUs evolved from fixed-function graphics pipelines to massively parallel programmable processors. Understanding GPU architecture is essential for AI/ML, scientific computing, and graphics.

**GPU vs CPU Design Philosophy:**
` + "```" + `
CPU Design: Optimize for SINGLE-THREAD LATENCY
  ┌────────────────────────────────────────────┐
  │  ┌──────────────────────────┐ ┌─────────┐  │
  │  │    Large Control Logic    │ │ Branch  │  │
  │  │  Out-of-order engine     │ │Predictor│  │
  │  │  Speculative execution   │ │ (huge)  │  │
  │  └──────────────────────────┘ └─────────┘  │
  │  ┌──────────────────────────────────────┐  │
  │  │         Large Cache Hierarchy         │  │
  │  │   L1: 32KB  L2: 1MB  L3: 32MB       │  │
  │  └──────────────────────────────────────┘  │
  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐              │
  │  │ALU │ │ALU │ │FPU │ │FPU │ (few, fast)  │
  │  └────┘ └────┘ └────┘ └────┘              │
  └────────────────────────────────────────────┘
  ~5-8 wide superscalar, deep OoO, big caches
  Optimized for: branch-heavy, pointer-chasing, serial code

GPU Design: Optimize for TOTAL THROUGHPUT
  ┌────────────────────────────────────────────┐
  │ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐ × many  │
  │ │FP││FP││FP││FP││FP││FP││FP││FP│ SM/CU   │
  │ └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘         │
  │ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐         │
  │ │FP││FP││FP││FP││FP││FP││FP││FP│         │
  │ └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘         │
  │  Small control │ Small cache │ Scheduler   │
  │  In-order exec │ 128KB shared│ Warp-based  │
  └────────────────────────────────────────────┘
  Thousands of simple cores, minimal control logic
  Rely on MASSIVE parallelism to hide latency
  Optimized for: data-parallel, regular, arithmetic-heavy code

NVIDIA H100 SXM:
  132 Streaming Multiprocessors (SMs)
  18,432 FP32 CUDA cores
  ~60 TFLOPS FP32
  ~1,979 TFLOPS FP8 (Tensor Cores)
  80 GB HBM3, 3.35 TB/s bandwidth
  700W TDP
` + "```" + `

**SIMT Execution Model:**
` + "```" + `
SIMT: Single Instruction, Multiple Threads
  Similar to SIMD but more flexible
  
NVIDIA warp = 32 threads executing in lockstep
AMD wavefront = 32 or 64 threads (RDNA = 32, CDNA = 64)

Streaming Multiprocessor (SM) - NVIDIA H100:
  ┌─────────────────────────────────────────┐
  │ SM (Streaming Multiprocessor)            │
  │                                          │
  │ Warp Scheduler 0    Warp Scheduler 1     │
  │ Dispatch Unit ×2    Dispatch Unit ×2     │
  │                                          │
  │ ┌──Processing Block 0──┐                 │
  │ │ 16 FP32 + 16 INT32   │ × 4 blocks     │
  │ │ 1 Tensor Core (4th gen)│                │
  │ │ 8 LD/ST units         │                │
  │ │ 4 SFU (sin/cos/sqrt) │                │
  │ └───────────────────────┘                │
  │                                          │
  │ 256 KB Register File (65536 × 32-bit)   │
  │ 256 KB L1/Shared Memory (configurable)  │
  │ 4 Warp Schedulers (up to 64 warps/SM)   │
  └─────────────────────────────────────────┘
  
Total per SM: 128 FP32, 64 INT32, 4 Tensor Cores

Warp execution:
  32 threads issue same instruction simultaneously
  Each thread operates on different data (SIMD within warp)
  
  Instruction: add r0, r1, r2
  Thread 0: r0[0] = r1[0] + r2[0]
  Thread 1: r0[1] = r1[1] + r2[1]
  ...
  Thread 31: r0[31] = r1[31] + r2[31]
  → 32 additions in 1 instruction issue

Warp Divergence:
  What happens with branches?
  
  if (threadIdx % 2 == 0)
      A();  // Even threads
  else
      B();  // Odd threads
  
  Execution:
  Cycle 1: Even threads execute A(), odd threads are MASKED
  Cycle 2: Odd threads execute B(), even threads are MASKED
  → Both paths execute serially! 50% efficiency loss

  With re-convergence optimization (since Volta):
  Independent Thread Scheduling
  Threads can diverge and re-converge more flexibly
  But divergence still hurts performance
` + "```" + `

**GPU Memory Hierarchy:**
` + "```" + `
GPU Memory Hierarchy (NVIDIA):

Register File (per SM):
  256 KB, ~1 cycle latency
  Divided among active threads
  More registers per thread → fewer threads per SM
  
Shared Memory / L1 (per SM):
  256 KB configurable split (H100)
  Shared among thread block
  ~30 cycles latency
  Used for: inter-thread communication, data reuse
  Bank conflicts: 32 banks, concurrent access if different banks

L2 Cache (chip-wide):
  50 MB (H100)
  ~200 cycles latency
  Shared by all SMs

Global Memory (HBM):
  80 GB HBM3 (H100)
  ~400-600 cycles latency
  3.35 TB/s bandwidth
  
  Memory access pattern matters enormously:
  
  Coalesced access (FAST):
    Thread 0 reads addr 0
    Thread 1 reads addr 4
    Thread 2 reads addr 8
    ...
    → Single 128-byte memory transaction for 32 threads
  
  Strided access (SLOW):
    Thread 0 reads addr 0
    Thread 1 reads addr 1024
    Thread 2 reads addr 2048
    ...
    → 32 separate memory transactions!
    → 32× more memory traffic

Texture Memory:
  2D spatial locality caching
  Good for image processing, interpolation
  Cached through separate texture units

Constant Memory:
  64 KB, cached with dedicated constant cache
  Broadcast: all threads reading same address = 1 transaction
  Good for: lookup tables, kernel parameters
` + "```" + `

**Occupancy and Performance:**
` + "```" + `
GPU Occupancy = Active Warps / Maximum Warps per SM

H100: Maximum 64 warps per SM (2048 threads)

Resource limits that affect occupancy:
  1. Registers per thread:
     256KB registers / 2048 threads = 128 registers max
     If kernel uses 128 regs → 100% occupancy
     If kernel uses 256 regs → 50% occupancy (only 1024 threads)
     
  2. Shared memory per block:
     If block uses 128KB shared → max 2 blocks per SM
     
  3. Thread block size:
     Must be multiple of warp size (32)
     Too small: waste scheduler slots
     Too large: limits number of blocks

  Why occupancy matters:
    GPU hides memory latency by switching warps
    Warp A: memory request (400 cycles to complete)
    Warp B: execute while A waits
    Warp C: execute while A,B wait
    ...
    Need enough warps to fill 400 cycles of latency!
    
    Rule of thumb: need ~20+ warps per SM for good latency hiding
    
  But higher occupancy isn't always better:
    More threads → more register pressure → spill to local memory
    Sometimes lower occupancy with more registers = faster
    Profile to find the sweet spot!
` + "```" + ``,
					CodeExamples: `// GPU architecture concepts simulation
package main

import (
    "fmt"
    "math"
)

// GPU configuration
type GPUConfig struct {
    name          string
    smCount       int
    coresPerSM    int
    clockMHz      int
    registerKB    int // Per SM
    sharedMemKB   int // Per SM
    l2CacheMB     int
    vramGB        int
    vramBWGBs     float64
    maxWarpsPerSM int
    warpSize      int
    tdpWatts      int
}

func (g GPUConfig) TotalCores() int {
    return g.smCount * g.coresPerSM
}

func (g GPUConfig) PeakFP32TFLOPS() float64 {
    return float64(g.TotalCores()) * float64(g.clockMHz) * 2 / 1e6 // FMA = 2 ops
}

func (g GPUConfig) MaxThreads() int {
    return g.smCount * g.maxWarpsPerSM * g.warpSize
}

// Occupancy calculator
type KernelConfig struct {
    name           string
    regsPerThread  int
    sharedMemBytes int
    blockSize      int
}

func calcOccupancy(gpu GPUConfig, kernel KernelConfig) (float64, int, string) {
    warpSize := gpu.warpSize
    maxWarps := gpu.maxWarpsPerSM
    
    // Warps per block
    warpsPerBlock := (kernel.blockSize + warpSize - 1) / warpSize
    
    // Register limit
    totalRegs := gpu.registerKB * 1024 / 4 // 32-bit registers
    maxThreadsByRegs := totalRegs / kernel.regsPerThread
    maxWarpsByRegs := maxThreadsByRegs / warpSize
    if maxWarpsByRegs > maxWarps { maxWarpsByRegs = maxWarps }
    
    // Shared memory limit
    maxBlocksByShmem := 1
    if kernel.sharedMemBytes > 0 {
        maxBlocksByShmem = (gpu.sharedMemKB * 1024) / kernel.sharedMemBytes
    } else {
        maxBlocksByShmem = 32 // Unlimited
    }
    maxWarpsByShmem := maxBlocksByShmem * warpsPerBlock
    if maxWarpsByShmem > maxWarps { maxWarpsByShmem = maxWarps }
    
    // Block size limit
    maxBlocks := maxWarps / warpsPerBlock
    maxWarpsByBlocks := maxBlocks * warpsPerBlock
    
    // Take minimum
    activeWarps := maxWarpsByRegs
    limiter := "registers"
    if maxWarpsByShmem < activeWarps {
        activeWarps = maxWarpsByShmem
        limiter = "shared memory"
    }
    if maxWarpsByBlocks < activeWarps {
        activeWarps = maxWarpsByBlocks
        limiter = "block size"
    }
    
    occupancy := float64(activeWarps) / float64(maxWarps) * 100
    return occupancy, activeWarps, limiter
}

// Memory coalescing simulation
type MemoryAccess struct {
    pattern      string
    transactions int
    bytesTotal   int
    efficiency   float64
}

func analyzeCoalescing(warpSize int, accessPattern func(tid int) int) MemoryAccess {
    cacheLineSize := 128 // bytes
    lines := make(map[int]bool)
    bytesUseful := 0
    
    for tid := 0; tid < warpSize; tid++ {
        addr := accessPattern(tid)
        line := addr / cacheLineSize
        lines[line] = true
        bytesUseful += 4 // 4 bytes per thread (float32)
    }
    
    transactions := len(lines)
    bytesTotal := transactions * cacheLineSize
    efficiency := float64(bytesUseful) / float64(bytesTotal) * 100
    
    return MemoryAccess{
        transactions: transactions,
        bytesTotal:   bytesTotal,
        efficiency:   efficiency,
    }
}

// Warp divergence simulation
func simulateDivergence(warpSize int, branchCondition func(tid int) bool) (int, float64) {
    takenCount := 0
    for tid := 0; tid < warpSize; tid++ {
        if branchCondition(tid) {
            takenCount++
        }
    }
    notTakenCount := warpSize - takenCount
    
    totalPasses := 0
    if takenCount > 0 { totalPasses++ }
    if notTakenCount > 0 { totalPasses++ }
    
    efficiency := 1.0 / float64(totalPasses) * 100
    if takenCount == 0 || notTakenCount == 0 {
        efficiency = 100
    } else {
        // Weighted efficiency
        maxActive := takenCount
        if notTakenCount > maxActive { maxActive = notTakenCount }
        efficiency = float64(warpSize) / float64(totalPasses*warpSize) * 100
    }
    
    return totalPasses, efficiency
}

func main() {
    gpus := []GPUConfig{
        {"NVIDIA RTX 4090", 128, 128, 2520, 256, 128, 72, 24, 1008, 48, 32, 450},
        {"NVIDIA H100 SXM", 132, 128, 1830, 256, 228, 50, 80, 3350, 64, 32, 700},
        {"NVIDIA A100", 108, 64, 1410, 256, 164, 40, 80, 2039, 64, 32, 400},
        {"AMD MI300X", 304, 64, 2100, 256, 64, 256, 192, 5300, 32, 64, 750},
    }
    
    fmt.Println("=== GPU Architecture Comparison ===")
    fmt.Printf("%-18s │ SMs │ Cores │ FP32 TFLOPS │ VRAM    │ BW (TB/s)│ TDP\n", "GPU")
    fmt.Println("───────────────────┼─────┼───────┼─────────────┼─────────┼──────────┼────")
    for _, g := range gpus {
        fmt.Printf("%-18s │ %3d │ %5d │ %8.1f    │ %3d GB  │ %5.1f    │%3dW\n",
            g.name, g.smCount, g.TotalCores(), g.PeakFP32TFLOPS(),
            g.vramGB, g.vramBWGBs/1000, g.tdpWatts)
    }
    
    // Occupancy analysis
    fmt.Println("\n=== Occupancy Analysis (H100) ===")
    h100 := gpus[1]
    
    kernels := []KernelConfig{
        {"Light kernel (32 regs, 0 shmem, 256 threads)", 32, 0, 256},
        {"Medium kernel (64 regs, 16KB shmem, 256 threads)", 64, 16384, 256},
        {"Heavy kernel (128 regs, 48KB shmem, 128 threads)", 128, 49152, 128},
        {"Register-hungry (255 regs, 0 shmem, 128 threads)", 255, 0, 128},
        {"Shmem-hungry (32 regs, 128KB shmem, 512 threads)", 32, 131072, 512},
    }
    
    for _, k := range kernels {
        occ, warps, limiter := calcOccupancy(h100, k)
        fmt.Printf("\n  %s:\n", k.name)
        fmt.Printf("    Occupancy: %.0f%% (%d warps), limited by: %s\n",
            occ, warps, limiter)
    }
    
    // Memory coalescing analysis
    fmt.Println("\n\n=== Memory Coalescing Analysis ===")
    
    patterns := []struct {
        name    string
        pattern func(int) int
    }{
        {"Coalesced (stride-1)", func(tid int) int { return tid * 4 }},
        {"Stride-2", func(tid int) int { return tid * 8 }},
        {"Stride-32", func(tid int) int { return tid * 128 }},
        {"Random (worst case)", func(tid int) int { return (tid*7919 + 3) % 65536 * 4 }},
        {"Broadcast (all same)", func(tid int) int { return 0 }},
        {"Aligned block (good)", func(tid int) int { return 4096 + tid*4 }},
    }
    
    fmt.Printf("%-25s │ Transactions │ Bytes │ Efficiency\n", "Access Pattern")
    fmt.Println("──────────────────────────┼──────────────┼───────┼───────────")
    for _, p := range patterns {
        result := analyzeCoalescing(32, p.pattern)
        fmt.Printf("%-25s │ %8d     │ %5d │ %6.1f%%\n",
            p.name, result.transactions, result.bytesTotal, result.efficiency)
    }
    
    // Warp divergence
    fmt.Println("\n=== Warp Divergence Impact ===")
    
    divergenceTests := []struct {
        name string
        cond func(int) bool
    }{
        {"No divergence (all true)", func(tid int) bool { return true }},
        {"50/50 (even/odd)", func(tid int) bool { return tid%2 == 0 }},
        {"75/25", func(tid int) bool { return tid%4 != 0 }},
        {"1/31 (only thread 0)", func(tid int) bool { return tid == 0 }},
    }
    
    fmt.Printf("%-30s │ Passes │ Efficiency\n", "Branch Pattern")
    fmt.Println("───────────────────────────────┼────────┼───────────")
    for _, t := range divergenceTests {
        passes, eff := simulateDivergence(32, t.cond)
        fmt.Printf("%-30s │ %4d   │ %5.0f%%\n", t.name, passes, eff)
    }
    
    // Performance model: compute vs memory bound
    fmt.Println("\n=== GPU Roofline (H100) ===")
    peakTFLOPS := gpus[1].PeakFP32TFLOPS()
    memBW := gpus[1].vramBWGBs
    ridgeOI := peakTFLOPS * 1000 / memBW // FLOPS per byte
    
    fmt.Printf("Peak: %.0f TFLOPS, Memory BW: %.0f GB/s, Ridge point: %.1f FLOP/B\n",
        peakTFLOPS, memBW, ridgeOI)
    
    workloads := []struct {
        name string
        oi   float64 // operational intensity (FLOP/byte)
    }{
        {"Vector add", 0.25},
        {"DGEMM (small)", 5},
        {"DGEMM (large)", 50},
        {"Stencil", 2},
        {"SpMV", 0.5},
        {"FFT", 3},
        {"Convolution (CNN)", 20},
        {"Transformer attention", 10},
    }
    
    fmt.Printf("\n%-25s │ OI    │ Achieved │ %% Peak │ Bound\n", "Workload")
    fmt.Println("──────────────────────────┼───────┼──────────┼────────┼─────────")
    for _, w := range workloads {
        achieved := math.Min(peakTFLOPS, memBW*w.oi/1000)
        pctPeak := achieved / peakTFLOPS * 100
        bound := "memory"
        if w.oi >= ridgeOI { bound = "compute" }
        fmt.Printf("%-25s │ %5.1f │ %6.1f TF│ %5.1f%% │ %s\n",
            w.name, w.oi, achieved, pctPeak, bound)
    }
}`,
				},
				{
					Title: "Tensor Cores and AI Acceleration",
					Content: `Tensor Cores are specialized matrix multiply-accumulate units designed for deep learning workloads. They represent the most significant architectural addition to modern GPUs.

**Tensor Core Operation:**
` + "```" + `
Tensor Core: D = A × B + C (matrix FMA)

4th Generation Tensor Cores (H100):
  Operation: 16×16×16 matrix multiply
  
  A (16×16) × B (16×16) + C (16×16) = D (16×16)
  
  Supported precisions:
  Format  │ Size  │ TFLOPS (H100) │ Use Case
  FP64    │ 64-bit│    67         │ Scientific computing
  TF32    │ 19-bit│   989         │ FP32 training (automatic)
  BF16    │ 16-bit│  1,979        │ Mixed-precision training
  FP16    │ 16-bit│  1,979        │ Mixed-precision training
  FP8     │ 8-bit │  3,958        │ Inference
  INT8    │ 8-bit │  3,958        │ Quantized inference
  
  Comparison to CUDA cores:
    FP32 CUDA cores: ~60 TFLOPS
    TF32 Tensor Cores: ~989 TFLOPS → 16× faster!
    FP8 Tensor Cores: ~3,958 TFLOPS → 66× faster!

Why matrix multiply matters for AI:
  Linear layer: Y = W × X + b  (matrix multiply)
  Convolution: Im2col + GEMM  (matrix multiply)
  Attention: Q × K^T, then × V  (matrix multiply)
  
  ~90% of training time is spent in matrix multiplications
  → Specialized hardware gives enormous speedup

Mixed Precision Training:
  Forward pass: FP16 (fast Tensor Core operations)
  Loss scaling: Multiply loss by large factor (prevent underflow)
  Weight update: FP32 (maintain accuracy)
  
  ┌────────┐  FP16   ┌────────┐  FP16   ┌────────┐
  │Forward │────────→│ Loss   │────────→│Backward│
  │ Pass   │         │Compute │         │ Pass   │
  └────────┘         └────────┘         └───┬────┘
                                            │ FP16 gradients
  ┌────────┐  FP32   ┌────────┐  FP32      │
  │Master  │←────────│Weight  │←────────────┘
  │Weights │         │Update  │  Cast grad to FP32
  └────────┘         └────────┘
  
  Result: ~2× faster training with negligible accuracy loss
` + "```" + `

**Transformer Architecture on GPU:**
` + "```" + `
Self-Attention (the core of Transformers):

Input: Sequence of N tokens, each D-dimensional
  Q = X × W_Q  (N×D × D×D = N×D)  ← Tensor Core GEMM
  K = X × W_K  (N×D × D×D = N×D)  ← Tensor Core GEMM
  V = X × W_V  (N×D × D×D = N×D)  ← Tensor Core GEMM
  
  Attention = softmax(Q × K^T / sqrt(D)) × V
              ↑ N×N matrix (quadratic!)

Memory bottleneck for long sequences:
  N=2048: Attention matrix = 2048² × 4B = 16 MB (fits in SRAM)
  N=8192: Attention matrix = 8192² × 4B = 256 MB (too large!)
  N=131K: Attention matrix = 131K² × 4B = 64 GB (larger than A100!)

Flash Attention (Dao et al. 2022):
  Key insight: Keep attention matrix in SRAM, never write to HBM
  
  Traditional:                 Flash Attention:
  Q,K,V in HBM                Q,K,V in HBM
  Compute S = QK^T → HBM      Block Q,K,V in SRAM tiles
  Compute P = softmax(S) →HBM  Compute attention in SRAM
  Compute O = PV → HBM         Write only final O → HBM
  
  3× fewer HBM reads/writes → 2-4× faster
  O(N) memory instead of O(N²)
  
  This is a perfect example of:
    Architecture knowledge → algorithm design → massive speedup
    Understanding GPU memory hierarchy drives the algorithm

Multi-Query Attention (MQA) / Grouped Query Attention (GQA):
  Standard: separate K,V per attention head
  MQA: share K,V across all heads (8× less KV-cache memory)
  GQA: share K,V within groups (compromise)
  
  Reduces memory bandwidth for inference
  LLM inference is memory-bandwidth bound → MQA helps significantly
` + "```" + `

**GPU Compute Workload Patterns:**
` + "```" + `
Common GPU Compute Patterns:

Map (embarrassingly parallel):
  output[i] = f(input[i])
  Perfect for GPU: 1 thread per element
  Examples: activation functions, element-wise operations
  
Reduce:
  sum = Σ input[i]
  Requires coordination (shared memory, atomics)
  Tree reduction in shared memory: O(log N) steps
  
  Step 1: thread i += thread i+16  (half active)
  Step 2: thread i += thread i+8   (quarter active)
  ...
  Step 5: thread 0 has final sum
  
Scan (prefix sum):
  output[i] = Σ(j=0 to i) input[j]
  Blelloch algorithm: O(N) work, O(log N) steps
  Used for: sort, unique, compact (stream compaction)

Stencil:
  output[i][j] = f(input[i-1][j], input[i+1][j], 
                    input[i][j-1], input[i][j+1])
  Halo exchange needed for tile boundaries
  Use shared memory for data reuse

Scatter/Gather:
  output[index[i]] = input[i]  (scatter)
  output[i] = input[index[i]]  (gather)
  Irregular access → poor coalescing → use sort-and-segment

SpMV (Sparse Matrix-Vector):
  Irregular memory access patterns
  Multiple formats: CSR, CSC, COO, BSR, ELL
  Memory-bandwidth bound typically
  Roofline OI ≈ 0.25-0.5 FLOP/byte (very low)
` + "```" + ``,
					CodeExamples: `// GPU computing concepts simulation
package main

import (
    "fmt"
    "math"
)

// Matrix multiply performance model
type MatMulPerf struct {
    M, N, K     int     // Matrix dimensions: (M×K) × (K×N) = (M×N)
    precision   string  // FP32, FP16, INT8
    bytesPerElem int
}

func (m MatMulPerf) FLOPs() float64 {
    return 2.0 * float64(m.M) * float64(m.N) * float64(m.K) // multiply + add
}

func (m MatMulPerf) MemoryBytes() float64 {
    return float64(m.bytesPerElem) * float64(m.M*m.K + m.K*m.N + m.M*m.N)
}

func (m MatMulPerf) ArithmeticIntensity() float64 {
    return m.FLOPs() / m.MemoryBytes()
}

// Transformer performance model
type TransformerConfig struct {
    seqLen     int
    hiddenDim  int
    numHeads   int
    numLayers  int
    batchSize  int
    precision  int // bytes per element
}

func (t TransformerConfig) AttentionFLOPs() float64 {
    // QKV projection + attention + output projection
    qkvProj := 3.0 * 2.0 * float64(t.batchSize*t.seqLen*t.hiddenDim*t.hiddenDim)
    attention := 2.0 * float64(t.batchSize*t.numHeads*t.seqLen*t.seqLen*(t.hiddenDim/t.numHeads))
    attnV := 2.0 * float64(t.batchSize*t.numHeads*t.seqLen*t.seqLen*(t.hiddenDim/t.numHeads))
    outProj := 2.0 * float64(t.batchSize*t.seqLen*t.hiddenDim*t.hiddenDim)
    return qkvProj + attention + attnV + outProj
}

func (t TransformerConfig) FFNFLOPs() float64 {
    // Two linear layers with 4× expansion
    return 2.0 * 2.0 * float64(t.batchSize*t.seqLen*t.hiddenDim*4*t.hiddenDim)
}

func (t TransformerConfig) TotalFLOPs() float64 {
    return float64(t.numLayers) * (t.AttentionFLOPs() + t.FFNFLOPs())
}

func (t TransformerConfig) AttentionMemory() float64 {
    // QKV + attention matrix + output
    qkv := 3 * t.batchSize * t.seqLen * t.hiddenDim * t.precision
    attnMatrix := t.batchSize * t.numHeads * t.seqLen * t.seqLen * t.precision
    return float64(qkv + attnMatrix)
}

func (t TransformerConfig) KVCacheBytes() float64 {
    // Per layer: K and V, shape [batch, heads, seq, head_dim]
    headDim := t.hiddenDim / t.numHeads
    perLayer := 2 * t.batchSize * t.numHeads * t.seqLen * headDim * t.precision
    return float64(perLayer * t.numLayers)
}

// Parallel reduction simulation
func parallelReduce(data []float64) float64 {
    n := len(data)
    buf := make([]float64, n)
    copy(buf, data)
    
    steps := 0
    for stride := n / 2; stride > 0; stride /= 2 {
        steps++
        for i := 0; i < stride; i++ {
            buf[i] += buf[i+stride]
        }
    }
    
    fmt.Printf("    Reduced %d elements in %d steps (log2 = %.0f)\n",
        n, steps, math.Log2(float64(n)))
    return buf[0]
}

func main() {
    fmt.Println("=== Matrix Multiply Performance Model ===")
    
    matmuls := []MatMulPerf{
        {1024, 1024, 1024, "FP32", 4},
        {4096, 4096, 4096, "FP32", 4},
        {4096, 4096, 4096, "FP16", 2},
        {4096, 4096, 4096, "INT8", 1},
        {16384, 16384, 16384, "FP16", 2},
    }
    
    fmt.Printf("%-12s │ M×N×K         │ TFLOP │ Memory │ AI (FLOP/B)\n", "Precision")
    fmt.Println("─────────────┼───────────────┼───────┼────────┼───────────")
    
    for _, m := range matmuls {
        flops := m.FLOPs() / 1e12
        mem := m.MemoryBytes() / 1e9
        ai := m.ArithmeticIntensity()
        fmt.Printf("%-12s │ %4d×%4d×%4d │ %5.2f │ %5.1fGB│ %8.1f\n",
            m.precision, m.M, m.N, m.K, flops, mem, ai)
    }
    
    // GPU execution time estimates
    fmt.Println("\n=== Estimated Execution Time (H100) ===")
    h100TFLOPS := map[string]float64{
        "FP32":  60,
        "TF32":  989,
        "FP16":  1979,
        "INT8":  3958,
    }
    h100BW := 3350.0 // GB/s
    
    for _, m := range matmuls {
        flops := m.FLOPs()
        mem := m.MemoryBytes()
        ai := m.ArithmeticIntensity()
        
        peakName := m.precision
        if peakName == "FP32" { peakName = "TF32" } // Tensor core auto
        peak := h100TFLOPS[peakName]
        
        computeTime := flops / (peak * 1e12) * 1000 // ms
        memTime := mem / (h100BW * 1e9) * 1000       // ms
        actualTime := math.Max(computeTime, memTime)
        bound := "compute"
        if memTime > computeTime { bound = "memory" }
        
        fmt.Printf("  %s %dx%dx%d: compute=%.2fms mem=%.2fms → %.2fms (%s, AI=%.0f)\n",
            m.precision, m.M, m.N, m.K, computeTime, memTime, actualTime, bound, ai)
    }
    
    // Transformer analysis
    fmt.Println("\n=== Transformer Model Analysis ===")
    
    models := []struct {
        name   string
        config TransformerConfig
    }{
        {"GPT-2 (124M)", TransformerConfig{1024, 768, 12, 12, 1, 2}},
        {"LLaMA-7B", TransformerConfig{2048, 4096, 32, 32, 1, 2}},
        {"LLaMA-70B", TransformerConfig{4096, 8192, 64, 80, 1, 2}},
        {"GPT-4 (est.)", TransformerConfig{8192, 12288, 96, 120, 1, 2}},
    }
    
    for _, m := range models {
        totalTFLOP := m.config.TotalFLOPs() / 1e12
        attnMem := m.config.AttentionMemory() / 1e9
        kvCache := m.config.KVCacheBytes() / 1e9
        
        // Inference time on H100
        inferTimeMS := m.config.TotalFLOPs() / (1979e12) * 1000
        
        fmt.Printf("\n  %s:\n", m.name)
        fmt.Printf("    Seq=%d, Hidden=%d, Heads=%d, Layers=%d\n",
            m.config.seqLen, m.config.hiddenDim, m.config.numHeads, m.config.numLayers)
        fmt.Printf("    Total FLOPs: %.2f TFLOP per forward pass\n", totalTFLOP)
        fmt.Printf("    Attention memory (per layer): %.2f GB\n", attnMem)
        fmt.Printf("    KV-Cache (all layers): %.2f GB\n", kvCache)
        fmt.Printf("    Est. H100 inference: %.2f ms (compute-bound est.)\n", inferTimeMS)
    }
    
    // Flash Attention memory savings
    fmt.Println("\n=== Flash Attention Memory Savings ===")
    fmt.Printf("%-12s │ Standard Attn │ Flash Attn │ Savings\n", "Seq Length")
    fmt.Println("─────────────┼───────────────┼────────────┼────────")
    
    for _, seqLen := range []int{512, 2048, 8192, 32768, 131072} {
        // Standard: O(N²) attention matrix
        standardMB := float64(seqLen*seqLen*4) / 1e6
        // Flash: O(N) working memory (tile size)
        tileSize := 256 // SRAM tile
        flashMB := float64(seqLen*tileSize*4*2) / 1e6
        savings := (1 - flashMB/standardMB) * 100
        
        stdStr := fmt.Sprintf("%.1f MB", standardMB)
        if standardMB > 1024 { stdStr = fmt.Sprintf("%.1f GB", standardMB/1024) }
        
        fmt.Printf("  %7d    │ %13s │ %7.1f MB │ %.0f%%\n",
            seqLen, stdStr, flashMB, savings)
    }
    
    // Parallel reduction
    fmt.Println("\n=== GPU Parallel Reduction ===")
    data := make([]float64, 32)
    for i := range data { data[i] = float64(i + 1) }
    
    fmt.Printf("  Input: [1, 2, 3, ..., 32]\n")
    sum := parallelReduce(data)
    fmt.Printf("  Result: %.0f (expected: %.0f)\n", sum, 32*33/2.0)
}`,
				},
			},
		},
	})
}
