package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2832,
			Title:       "Performance Analysis and Optimization",
			Description: "Master CPU performance metrics, profiling tools, bottleneck analysis, and systematic optimization techniques for modern processors.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "CPU Performance Metrics and Models",
					Content: `Understanding computer performance requires precise metrics. Intuitive notions of "fast" can be misleading without formal measurement methodologies.

**The CPU Performance Equation:**
` + "```" + `
Execution Time = Instructions × CPI × Clock Period

Where:
  Instructions = IC (Instruction Count) - determined by ISA + compiler
  CPI = Cycles Per Instruction (average)
  Clock Period = 1/Frequency

Equivalently:
  CPU Time = IC × CPI / Clock Rate

Example:
  Program: 10 billion instructions
  CPI: 1.5 cycles/instruction  
  Clock: 3 GHz (3 × 10⁹ Hz)
  
  CPU Time = 10×10⁹ × 1.5 / (3×10⁹) = 5 seconds

Factors affecting each:
  IC depends on:    ISA, compiler, algorithm choice
  CPI depends on:   Microarchitecture, cache behavior, branch prediction
  Clock depends on:  Technology, circuit design, power/thermal limits

Iron Law dilemma:
  Higher clock → more pipeline stages → higher CPI (more hazards)
  CISC ISA → fewer instructions → but each takes more cycles
  Simpler core → lower CPI → but need more instructions
` + "```" + `

**IPC and Throughput:**
` + "```" + `
IPC (Instructions Per Cycle) = 1/CPI

Modern cores strive for IPC > 1 via superscalar execution:

Processor Era          │ Typical IPC │ How
8086 (1978)            │   0.1       │ Many multi-cycle instructions
i486 (1989)            │   0.5-1.0   │ Pipelining
Pentium (1993)         │   1.0-2.0   │ Dual-issue superscalar
P6/Athlon (1997-99)    │   1.5-3.0   │ Out-of-order, 3-wide
Core 2 (2006)          │   2.0-4.0   │ 4-wide, better branch pred
Skylake (2015)         │   3.0-5.0   │ 4-wide, µop cache, better OoO
Golden Cove (2021)     │   4.0-6.0   │ 6-wide decode, deeper OoO
Zen 4 (2022)           │   4.0-6.0   │ 6-wide, large caches
Apple M3 (2023)        │   5.0-8.0   │ 8-wide decode, massive ROB

Throughput metrics:
  FLOPS = Floating-point Operations Per Second
  MIPS  = Million Instructions Per Second (misleading!)
  
  Why MIPS is misleading:
  Program A: 100M instructions, CPI=1.0 → 100 MIPS on 100 MHz
  Program B: 50M instructions, CPI=2.0 → 50 MIPS on 100 MHz
  Program B finishes in same time as A, but has lower MIPS!
  
  Better: Measure actual wall-clock execution time
` + "```" + `

**Amdahl's Law and Scaling:**
` + "```" + `
Amdahl's Law:
  Speedup = 1 / ((1 - P) + P/S)
  
  P = fraction that can be improved
  S = speedup of improved portion

Example: Program is 80% parallelizable, run on 4 cores
  Speedup = 1 / (0.2 + 0.8/4) = 1/0.4 = 2.5×
  NOT 4× even though we used 4 cores!

With infinite cores:
  Speedup_max = 1 / (1 - P)
  If P = 0.95 (95% parallel): max speedup = 20×
  If P = 0.99 (99% parallel): max speedup = 100×
  If P = 0.50 (50% parallel): max speedup = 2×

  Cores │ P=50% │ P=90% │ P=95% │ P=99%
     1  │  1.0× │  1.0× │  1.0× │  1.0×
     2  │  1.3× │  1.8× │  1.9× │  2.0×
     4  │  1.6× │  3.1× │  3.5× │  3.9×
     8  │  1.8× │  4.7× │  5.9× │  7.5×
    16  │  1.9× │  6.4× │  9.1× │ 13.9×
    64  │  2.0× │  8.8× │ 15.4× │ 39.3×
   256  │  2.0× │  9.7× │ 18.6× │ 72.1×
     ∞  │  2.0× │ 10.0× │ 20.0× │100.0×

Gustafson's Law (alternative view):
  Scaled Speedup = N - (1 - P)(N - 1)
  
  Argument: As we get more cores, we solve BIGGER problems
  The serial fraction stays ~constant in absolute time
  So the parallel fraction grows with problem size
  → More optimistic than Amdahl for real workloads
` + "```" + `

**Cache Performance Impact:**
` + "```" + `
Memory Access Time = Hit Rate × Hit Time + Miss Rate × Miss Penalty

For cache hierarchy:
  AMAT = T_L1 + MR_L1 × (T_L2 + MR_L2 × (T_L3 + MR_L3 × T_mem))

Example:
  L1: 1 cycle, 5% miss rate
  L2: 10 cycles, 20% miss rate (of L1 misses)
  L3: 30 cycles, 30% miss rate (of L2 misses)
  Memory: 200 cycles

  AMAT = 1 + 0.05 × (10 + 0.20 × (30 + 0.30 × 200))
       = 1 + 0.05 × (10 + 0.20 × (30 + 60))
       = 1 + 0.05 × (10 + 18)
       = 1 + 1.4
       = 2.4 cycles

Impact on CPI:
  Base CPI (no misses) = 1.0
  Memory stall cycles per instruction =
    Mem accesses/instruction × miss rate × miss penalty
    
  Example: 0.3 mem accesses/instr, 5% L1 miss, 200-cycle penalty
    Stall cycles = 0.3 × 0.05 × 200 = 3.0 cycles/instr
    Effective CPI = 1.0 + 3.0 = 4.0 (cache misses 3× worse than compute!)

  With L2 cache (95% of L1 misses caught in L2, 10-cycle penalty):
    Stall = 0.3 × 0.05 × (0.95 × 10 + 0.05 × 200)
          = 0.3 × 0.05 × (9.5 + 10)
          = 0.3 × 0.05 × 19.5
          = 0.293 cycles/instr
    Effective CPI = 1.293 (much better!)
` + "```" + ``,
					CodeExamples: `// Performance analysis tools simulation
package main

import (
    "fmt"
    "math"
)

// CPU Performance Calculator
type CPUPerformance struct {
    clockGHz      float64
    ic            float64 // Instruction count (billions)
    baseCPI       float64 // Base CPI without stalls
}

func (p CPUPerformance) ExecutionTime(effectiveCPI float64) float64 {
    return p.ic * 1e9 * effectiveCPI / (p.clockGHz * 1e9)
}

func (p CPUPerformance) MIPS(effectiveCPI float64) float64 {
    return p.clockGHz * 1000 / effectiveCPI
}

// Cache hierarchy model
type CacheLevel struct {
    name      string
    sizeMB    float64
    latency   float64 // cycles
    missRate  float64 // local miss rate
}

func computeAMAT(hierarchy []CacheLevel, memLatency float64) float64 {
    if len(hierarchy) == 0 {
        return memLatency
    }
    
    first := hierarchy[0]
    restAMAT := computeAMAT(hierarchy[1:], memLatency)
    return first.latency + first.missRate*restAMAT
}

func computeStallCPI(memAccessPerInstr float64, hierarchy []CacheLevel,
    memLatency float64) float64 {
    // Calculate average penalty for a memory access
    avgPenalty := computeAMAT(hierarchy, memLatency)
    return memAccessPerInstr * (avgPenalty - hierarchy[0].latency) // subtract L1 hit (in base CPI)
}

// Amdahl's Law
func amdahlSpeedup(parallelFraction float64, cores int) float64 {
    serial := 1 - parallelFraction
    return 1.0 / (serial + parallelFraction/float64(cores))
}

func gustafsonSpeedup(parallelFraction float64, cores int) float64 {
    serial := 1 - parallelFraction
    return float64(cores) - serial*float64(cores-1)
}

// Roofline model
type RooflineModel struct {
    peakGFLOPS float64
    memBWGBs   float64
}

func (r RooflineModel) AchievableGFLOPS(oi float64) float64 {
    return math.Min(r.peakGFLOPS, r.memBWGBs*oi)
}

func (r RooflineModel) RidgePoint() float64 {
    return r.peakGFLOPS / r.memBWGBs
}

func (r RooflineModel) Bottleneck(oi float64) string {
    if oi < r.RidgePoint() { return "MEMORY-BOUND" }
    return "COMPUTE-BOUND"
}

func main() {
    // CPU Performance Equation
    fmt.Println("=== CPU Performance Equation ===")
    
    scenarios := []struct {
        name    string
        cpu     CPUPerformance
        effCPI  float64
    }{
        {"Simple in-order core", CPUPerformance{2.0, 5, 1.0}, 1.5},
        {"Wide OoO core", CPUPerformance{4.0, 3, 0.5}, 0.8},
        {"With cache misses", CPUPerformance{4.0, 3, 0.5}, 3.2},
    }

    for _, s := range scenarios {
        time := s.cpu.ExecutionTime(s.effCPI)
        mips := s.cpu.MIPS(s.effCPI)
        ipc := 1.0/s.effCPI
        fmt.Printf("\n%s:\n", s.name)
        fmt.Printf("  %.1f GHz, %.0fB instr, CPI=%.1f\n",
            s.cpu.clockGHz, s.cpu.ic, s.effCPI)
        fmt.Printf("  Time=%.3fs, IPC=%.2f, MIPS=%.0f\n", time, ipc, mips)
    }

    // Cache hierarchy analysis
    fmt.Println("\n\n=== Cache Hierarchy Analysis ===")
    
    configs := []struct {
        name      string
        hierarchy []CacheLevel
    }{
        {"L1 only", []CacheLevel{
            {"L1", 0.032, 1, 0.05},
        }},
        {"L1 + L2", []CacheLevel{
            {"L1", 0.032, 1, 0.05},
            {"L2", 0.256, 10, 0.20},
        }},
        {"L1 + L2 + L3", []CacheLevel{
            {"L1", 0.032, 1, 0.05},
            {"L2", 0.256, 10, 0.20},
            {"L3", 16, 30, 0.30},
        }},
        {"Large L3 (64MB, 3D V-Cache)", []CacheLevel{
            {"L1", 0.032, 1, 0.04},
            {"L2", 1.0, 8, 0.15},
            {"L3", 64, 25, 0.10},
        }},
    }

    memLatency := 200.0
    memAccessRate := 0.3

    for _, cfg := range configs {
        amat := computeAMAT(cfg.hierarchy, memLatency)
        stallCPI := computeStallCPI(memAccessRate, cfg.hierarchy, memLatency)
        effectiveCPI := 1.0 + stallCPI
        
        fmt.Printf("\n%s:\n", cfg.name)
        for _, l := range cfg.hierarchy {
            fmt.Printf("  %s: %.0f KB, %2.0f cycles, %.0f%% miss\n",
                l.name, l.sizeMB*1024, l.latency, l.missRate*100)
        }
        fmt.Printf("  AMAT: %.2f cycles\n", amat)
        fmt.Printf("  Stall CPI: %.3f, Effective CPI: %.3f\n", stallCPI, effectiveCPI)
        fmt.Printf("  Relative performance: %.1f%%\n", 100.0/effectiveCPI)
    }

    // Amdahl's Law
    fmt.Println("\n\n=== Amdahl's Law vs Gustafson's Law ===")
    fmt.Println("\nAmdahl's Law (fixed problem size):")
    fmt.Printf("  Cores │")
    for _, p := range []float64{0.5, 0.9, 0.95, 0.99} {
        fmt.Printf(" P=%.0f%% │", p*100)
    }
    fmt.Println()
    fmt.Println("  ──────┼───────┼───────┼───────┼───────")
    for _, n := range []int{1, 2, 4, 8, 16, 64, 256} {
        fmt.Printf("  %5d │", n)
        for _, p := range []float64{0.5, 0.9, 0.95, 0.99} {
            s := amdahlSpeedup(p, n)
            fmt.Printf(" %5.1f× │", s)
        }
        fmt.Println()
    }

    fmt.Println("\nGustafson's Law (scaled problem size):")
    fmt.Printf("  Cores │")
    for _, p := range []float64{0.5, 0.9, 0.95, 0.99} {
        fmt.Printf(" P=%.0f%% │", p*100)
    }
    fmt.Println()
    fmt.Println("  ──────┼───────┼───────┼───────┼───────")
    for _, n := range []int{1, 2, 4, 8, 16, 64, 256} {
        fmt.Printf("  %5d │", n)
        for _, p := range []float64{0.5, 0.9, 0.95, 0.99} {
            s := gustafsonSpeedup(p, n)
            fmt.Printf(" %5.1f× │", s)
        }
        fmt.Println()
    }

    // Roofline model
    fmt.Println("\n\n=== Roofline Model ===")
    processors := []struct {
        name  string
        model RooflineModel
    }{
        {"Zen 4 (1 core)", RooflineModel{76.8, 40}},
        {"Apple M3 Pro", RooflineModel{1800, 150}},
        {"NVIDIA A100", RooflineModel{19500, 2039}},
    }

    for _, proc := range processors {
        fmt.Printf("\n%s (Peak: %.0f GFLOPS, BW: %.0f GB/s, Ridge: %.1f FLOP/B):\n",
            proc.name, proc.model.peakGFLOPS, proc.model.memBWGBs,
            proc.model.RidgePoint())
        
        for _, oi := range []float64{0.1, 0.5, 1, 2, 5, 10, 50} {
            achieved := proc.model.AchievableGFLOPS(oi)
            bound := proc.model.Bottleneck(oi)
            pct := achieved / proc.model.peakGFLOPS * 100
            fmt.Printf("  OI=%4.1f: %8.1f GFLOPS (%5.1f%%) %s\n",
                oi, achieved, pct, bound)
        }
    }
}`,
				},
				{
					Title: "Profiling and Bottleneck Identification",
					Content: `Performance profiling is the empirical side of computer architecture. Modern CPUs have hardware performance counters that provide deep insight into execution behavior.

**Hardware Performance Counters:**
` + "```" + `
Performance Monitoring Unit (PMU):
  Modern CPUs have hundreds of hardware event counters
  Can count microarchitectural events with near-zero overhead
  
Key counters (Intel):

Instruction Flow:
  INST_RETIRED.ANY          - Instructions completed
  CPU_CLK_UNHALTED.THREAD   - Cycles the core was active
  BR_MISP_RETIRED.ALL_BRANCHES - Branch mispredictions
  MACHINE_CLEARS.ANY        - Pipeline flushes (speculation gone wrong)
  
Cache Events:
  L1D_CACHE.MISS            - L1 data cache misses
  L2_RQSTS.MISS             - L2 cache misses
  LLC_MISSES                - Last-Level Cache misses
  DTLB_LOAD_MISSES.WALK_COMPLETED - TLB miss requiring page walk
  
Memory:
  MEM_LOAD_RETIRED.L1_HIT   - Loads served by L1
  MEM_LOAD_RETIRED.L2_HIT   - Loads served by L2
  MEM_LOAD_RETIRED.L3_HIT   - Loads served by L3
  MEM_LOAD_RETIRED.L3_MISS  - Loads going to DRAM
  
Execution:
  UOPS_DISPATCHED.THREAD    - Micro-ops dispatched
  UOPS_RETIRED.RETIRE_SLOTS - Micro-ops completed
  IDQ_UOPS_NOT_DELIVERED.CORE - Front-end starvation
  CYCLE_ACTIVITY.STALLS_MEM_ANY - Memory stall cycles
  RESOURCE_STALLS.ANY        - Back-end resource stalls

Derived metrics:
  IPC = INST_RETIRED / CPU_CLK_UNHALTED
  Branch miss rate = BR_MISP / BR_INST
  L1 data miss rate = L1D.MISS / L1D.ACCESS
  MPKI (Misses Per Kilo Instructions) = misses × 1000 / instructions
  Cycles stalled = stall_cycles / total_cycles (%)
` + "```" + `

**Top-Down Microarchitecture Analysis (TMA):**
` + "```" + `
Intel's TMA methodology classifies pipeline slots into 4 categories:

Level 1 (Pipeline slot usage):
┌──────────────────────────────────────────────┐
│              All Pipeline Slots               │
├──────────────┬───────────────────────────────┤
│  Retiring    │         Not Retiring           │
│  (useful)    ├───────────┬─────────┬─────────┤
│              │Bad Specul.│Front-End│Back-End  │
│              │           │  Bound  │  Bound   │
└──────────────┴───────────┴─────────┴─────────┘

Retiring: Slots used by instructions that completed (good!)
Bad Speculation: Slots wasted on mispredicted branch paths
Front-End Bound: Slots empty because front-end couldn't deliver µops
Back-End Bound: Slots stalled because back-end couldn't accept µops

Level 2 breakdown:
  Front-End Bound:
    → Fetch Latency (I-cache miss, ITLB miss, branch resteers)
    → Fetch Bandwidth (decoder limits, µop cache misses)
    
  Back-End Bound:
    → Memory Bound:
        → L1 Bound (store forwarding stalls, lock contention)
        → L2 Bound (L2 hit but L1 miss)
        → L3 Bound (L3 hit but L2 miss)
        → DRAM Bound (memory latency, bandwidth)
    → Core Bound:
        → Divider (long division operations)
        → Port Utilization (execution unit contention)

Example TMA results:
  Program A (well-optimized):
    Retiring: 45% ← Good!
    Bad Speculation: 5%
    Front-end: 10%
    Back-end: 40% (Memory: 35%, Core: 5%)
    → Memory-bound, focus on cache optimization
    
  Program B (branch-heavy):
    Retiring: 20%
    Bad Speculation: 40% ← Many mispredictions!
    Front-end: 25%
    Back-end: 15%
    → Need better branch prediction or branchless code
    
  Program C (compute-heavy):
    Retiring: 60% ← Excellent!
    Bad Speculation: 5%
    Front-end: 5%
    Back-end: 30% (Core: 25%, Memory: 5%)
    → Well-optimized, could try wider SIMD
` + "```" + `

**Linux perf Tool:**
` + "```" + `
perf stat (summary counters):
  $ perf stat ./my_program
  
  Performance counter stats for './my_program':
    3,421,897,124  cycles           #  3.421 GHz
    5,634,218,943  instructions     #  1.65  insn per cycle
      856,234,567  branches         # 856.235 M/sec
       12,345,678  branch-misses    #  1.44% of all branches
      234,567,890  cache-references                    
       23,456,789  cache-misses     # 10.00% of cache refs
      
      1.002345678 seconds time elapsed
      0.998234567 seconds user
      0.003123456 seconds sys

perf record + report (sampling profiler):
  $ perf record -g ./my_program   # Record with call graphs
  $ perf report                    # Analyze results
  
  43.21%  my_program  [.] hot_function
  22.15%  my_program  [.] sort_array
  12.34%  libc.so     [.] memcpy
   8.56%  my_program  [.] parse_input
   5.67%  libc.so     [.] malloc

perf top (live monitoring):
  $ perf top -g    # Like 'top' but for functions by CPU usage

perf annotate (instruction-level):
  $ perf annotate hot_function
  Shows which assembly instructions are hottest

perf c2c (cache-to-cache / false sharing):
  $ perf c2c record ./my_program
  $ perf c2c report
  Shows cacheline contention between cores
` + "```" + `

**Common Bottleneck Patterns:**
` + "```" + `
Pattern 1: Cache Thrashing
  Symptom: High LLC miss rate, DRAM-bound in TMA
  Cause: Working set exceeds cache size
  Fix: Blocking/tiling, reduce data structure size, 
       improve spatial locality (SoA vs AoS)

Pattern 2: False Sharing
  Symptom: High cache-to-cache transfers, poor scaling
  Cause: Different cores write to same cache line
  Fix: Pad structures to cache line boundaries (64B)
  
  struct Counter {
    value uint64       // Core 0 writes this
    // padding [56]byte // ADD THIS
    other uint64       // Core 1 writes this  
  }
  // Both fields on same 64-byte cache line = false sharing!

Pattern 3: Branch Misprediction
  Symptom: High bad speculation % in TMA
  Cause: Unpredictable branches (random data)
  Fix: Branchless code (CMOV), sorting data, lookup tables

Pattern 4: Front-End Starvation
  Symptom: High front-end bound % in TMA
  Cause: Large code footprint (I-cache misses)
  Fix: Hot/cold code splitting, PGO, reduce code bloat

Pattern 5: Memory Bandwidth Saturation
  Symptom: Stalls even with good hit rates
  Cause: Too many outstanding memory accesses
  Fix: Reduce memory traffic, use SIMD, prefetch
  
Pattern 6: TLB Misses
  Symptom: DTLB/ITLB page walk events high
  Cause: Scattered memory access across many pages
  Fix: Huge pages (2MB/1GB), improve spatial locality
` + "```" + ``,
					CodeExamples: `// Performance analysis simulation in Go
package main

import (
    "fmt"
    "math"
    "math/rand"
    "sort"
    "time"
)

// Simulated performance counters
type PerfCounters struct {
    cycles          uint64
    instructions    uint64
    branchInstr     uint64
    branchMisses    uint64
    l1Accesses      uint64
    l1Misses        uint64
    l2Accesses      uint64
    l2Misses        uint64
    l3Accesses      uint64
    l3Misses        uint64
    stallCyclesMem  uint64
    stallCyclesExec uint64
}

func (p PerfCounters) IPC() float64 {
    if p.cycles == 0 { return 0 }
    return float64(p.instructions) / float64(p.cycles)
}

func (p PerfCounters) BranchMissRate() float64 {
    if p.branchInstr == 0 { return 0 }
    return float64(p.branchMisses) / float64(p.branchInstr) * 100
}

func (p PerfCounters) L1MissRate() float64 {
    if p.l1Accesses == 0 { return 0 }
    return float64(p.l1Misses) / float64(p.l1Accesses) * 100
}

func (p PerfCounters) MPKI(level string) float64 {
    if p.instructions == 0 { return 0 }
    var misses uint64
    switch level {
    case "L1": misses = p.l1Misses
    case "L2": misses = p.l2Misses
    case "L3": misses = p.l3Misses
    }
    return float64(misses) * 1000 / float64(p.instructions)
}

func (p PerfCounters) Report(name string) {
    fmt.Printf("\n=== perf stat: %s ===\n", name)
    fmt.Printf("  %15d cycles\n", p.cycles)
    fmt.Printf("  %15d instructions     # %.2f IPC\n", p.instructions, p.IPC())
    fmt.Printf("  %15d branches\n", p.branchInstr)
    fmt.Printf("  %15d branch-misses    # %.2f%%\n", p.branchMisses, p.BranchMissRate())
    fmt.Printf("  %15d L1-dcache-accesses\n", p.l1Accesses)
    fmt.Printf("  %15d L1-dcache-misses  # %.2f%% (%.1f MPKI)\n",
        p.l1Misses, p.L1MissRate(), p.MPKI("L1"))
    fmt.Printf("  %15d LLC-misses       # %.1f MPKI\n", p.l3Misses, p.MPKI("L3"))
    
    totalStall := p.stallCyclesMem + p.stallCyclesExec
    memPct := float64(p.stallCyclesMem) / float64(p.cycles) * 100
    execPct := float64(p.stallCyclesExec) / float64(p.cycles) * 100
    retPct := 100 - memPct - execPct
    
    fmt.Printf("\n  TMA Level 1:\n")
    fmt.Printf("    Retiring:       %5.1f%%\n", retPct)
    fmt.Printf("    Backend-Bound:  %5.1f%% (Memory: %.1f%%, Core: %.1f%%)\n",
        float64(totalStall)/float64(p.cycles)*100, memPct, execPct)
}

// Demonstrate cache-friendly vs cache-unfriendly access
func benchmarkCacheAccess() {
    fmt.Println("\n=== Cache Access Pattern Benchmark ===")
    
    const N = 1 << 20 // 1M elements
    data := make([]int, N)
    for i := range data { data[i] = i }
    
    // Sequential access (cache-friendly)
    start := time.Now()
    sum := 0
    for i := 0; i < N; i++ {
        sum += data[i]
    }
    seqTime := time.Since(start)
    
    // Random access (cache-unfriendly)
    indices := make([]int, N)
    for i := range indices { indices[i] = rand.Intn(N) }
    
    start = time.Now()
    sum = 0
    for i := 0; i < N; i++ {
        sum += data[indices[i]]
    }
    randTime := time.Since(start)
    
    fmt.Printf("  Sequential access: %v\n", seqTime)
    fmt.Printf("  Random access:     %v\n", randTime)
    if seqTime > 0 {
        fmt.Printf("  Slowdown: %.1fx\n", float64(randTime)/float64(seqTime))
    }
}

// Demonstrate branch prediction impact
func benchmarkBranchPrediction() {
    fmt.Println("\n=== Branch Prediction Benchmark ===")
    
    const N = 1 << 18
    data := make([]int, N)
    for i := range data { data[i] = rand.Intn(256) }
    
    // Sorted data (predictable branches)
    sorted := make([]int, N)
    copy(sorted, data)
    sort.Ints(sorted)
    
    countAbove := func(arr []int, threshold int) int {
        count := 0
        for _, v := range arr {
            if v >= threshold { // This branch is the key
                count++
            }
        }
        return count
    }
    
    start := time.Now()
    _ = countAbove(sorted, 128) // Sorted: branch is predictable
    sortedTime := time.Since(start)
    
    start = time.Now()
    _ = countAbove(data, 128) // Unsorted: random branch pattern
    unsortedTime := time.Since(start)
    
    fmt.Printf("  Sorted data (predictable):   %v\n", sortedTime)
    fmt.Printf("  Unsorted data (random):      %v\n", unsortedTime)
    if sortedTime > 0 {
        fmt.Printf("  Slowdown: %.1fx\n", float64(unsortedTime)/float64(sortedTime))
    }
    
    // Branchless version
    countAboveBranchless := func(arr []int) int {
        count := 0
        for _, v := range arr {
            // Branchless: convert comparison to 0/1
            mask := (v - 128) >> 31 // 0 if v >= 128, -1 if v < 128
            count += 1 + mask        // 1 if v >= 128, 0 if v < 128
        }
        return count
    }
    
    start = time.Now()
    _ = countAboveBranchless(data) // Branchless on unsorted
    branchlessTime := time.Since(start)
    
    fmt.Printf("  Branchless (unsorted data):  %v\n", branchlessTime)
}

// False sharing demonstration concept
func demonstrateFalseSharing() {
    fmt.Println("\n=== False Sharing Concept ===")
    
    type BadLayout struct {
        counter1 uint64 // Core 0 increments this
        counter2 uint64 // Core 1 increments this
        // Both on same 64-byte cache line!
    }
    
    type GoodLayout struct {
        counter1 uint64
        _pad1    [56]byte // Pad to 64-byte cache line
        counter2 uint64
        _pad2    [56]byte
    }
    
    fmt.Printf("  BadLayout size:  %d bytes (both counters in same cache line)\n",
        16) // simplified
    fmt.Printf("  GoodLayout size: %d bytes (each counter in own cache line)\n",
        128) // simplified
    fmt.Println("  False sharing causes cache lines to bounce between cores")
    fmt.Println("  Padding separates them to different cache lines")
    fmt.Println()
    
    // Show cache line math
    cacheLineSize := 64
    fmt.Printf("  Cache line size: %d bytes\n", cacheLineSize)
    type ExampleStruct struct {
        a int64  // 8 bytes, offset 0
        b int64  // 8 bytes, offset 8
        c int64  // 8 bytes, offset 16
    }
    fmt.Printf("  ExampleStruct: 3 × int64 = 24 bytes → fits in 1 cache line\n")
    fmt.Printf("  If Core0 writes .a and Core1 writes .c → false sharing!\n")
    fmt.Printf("  Fix: add %d bytes padding between .a and .c\n",
        cacheLineSize-8)
}

// Working set size vs cache performance
func cacheWorkingSetAnalysis() {
    fmt.Println("\n=== Working Set Size vs Performance ===")
    
    cacheSizes := []struct {
        name string
        kb   int
    }{
        {"L1 (32KB)", 32},
        {"L2 (256KB)", 256},
        {"L3 (16MB)", 16384},
        {"L3 (64MB V-Cache)", 65536},
    }
    
    workingSets := []int{16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144}
    
    fmt.Printf("%8s │", "WS(KB)")
    for _, cs := range cacheSizes {
        fmt.Printf(" %12s │", cs.name)
    }
    fmt.Println()
    fmt.Println("─────────┼──────────────┼──────────────┼──────────────┼──────────────")
    
    for _, ws := range workingSets {
        fmt.Printf("%8d │", ws)
        for _, cs := range cacheSizes {
            var latency float64
            if ws <= cs.kb {
                // Fits in cache
                latency = math.Log2(float64(cs.kb)) + 1 // Simplified
            } else {
                // Exceeds cache - higher latency
                ratio := float64(ws) / float64(cs.kb)
                latency = math.Log2(float64(cs.kb)) + 1 + math.Log2(ratio)*10
            }
            indicator := " "
            if ws <= cs.kb { indicator = "✓" }
            fmt.Printf(" %6.1f ns %s  │", latency, indicator)
        }
        fmt.Println()
    }
}

func main() {
    // Simulated perf counter reports
    counters := []struct {
        name string
        perf PerfCounters
    }{
        {"Matrix multiply (naive)", PerfCounters{
            cycles: 5000000000, instructions: 2000000000,
            branchInstr: 100000000, branchMisses: 500000,
            l1Accesses: 1500000000, l1Misses: 150000000,
            l2Accesses: 150000000, l2Misses: 30000000,
            l3Accesses: 30000000, l3Misses: 10000000,
            stallCyclesMem: 3000000000, stallCyclesExec: 500000000,
        }},
        {"Matrix multiply (blocked)", PerfCounters{
            cycles: 1200000000, instructions: 2200000000,
            branchInstr: 110000000, branchMisses: 600000,
            l1Accesses: 1800000000, l1Misses: 20000000,
            l2Accesses: 20000000, l2Misses: 2000000,
            l3Accesses: 2000000, l3Misses: 100000,
            stallCyclesMem: 300000000, stallCyclesExec: 200000000,
        }},
        {"Sort (random input)", PerfCounters{
            cycles: 800000000, instructions: 600000000,
            branchInstr: 200000000, branchMisses: 40000000,
            l1Accesses: 400000000, l1Misses: 40000000,
            l2Accesses: 40000000, l2Misses: 5000000,
            l3Accesses: 5000000, l3Misses: 500000,
            stallCyclesMem: 200000000, stallCyclesExec: 300000000,
        }},
    }

    for _, c := range counters {
        c.perf.Report(c.name)
    }

    benchmarkCacheAccess()
    benchmarkBranchPrediction()
    demonstrateFalseSharing()
    cacheWorkingSetAnalysis()
}`,
				},
			},
		},
	})
}
