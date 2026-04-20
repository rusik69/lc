package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1625,
			Title:       "Go Runtime Internals",
			Description: "Understand the Go runtime: garbage collector, memory allocator, stack management, and performance tuning.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Garbage Collector Deep Dive",
					Content: `Go uses a concurrent, tri-color mark-and-sweep garbage collector. Understanding its behavior is critical for writing low-latency applications.

**GC Algorithm:**
` + "```" + `
Go's GC is a non-generational, concurrent, tri-color, mark-sweep collector.

Phases:
  1. Mark Setup (STW ~10-30μs)
     - Stop the world briefly
     - Enable write barrier
     - Start marking goroutines
     
  2. Marking (concurrent)
     - GC goroutines trace reachable objects
     - Runs concurrently with application
     - Uses 25% of GOMAXPROCS by default
     - Write barrier tracks pointer modifications
     
  3. Mark Termination (STW ~10-30μs)
     - Stop the world briefly
     - Disable write barrier
     - Clean up
     
  4. Sweeping (concurrent)
     - Reclaim unmarked objects
     - Runs between collections
     - No stop-the-world needed

Total STW pause: ~20-60μs (typically)
Go 1.5+: sub-millisecond pauses
Go 1.19+: soft memory limit (GOMEMLIMIT)

Tri-color marking:
  White: Not yet visited (potentially garbage)
  Grey:  Visited but children not fully scanned
  Black: Visited and all children scanned
  
  Start: All objects white, roots grey
  Process: Pick grey object → scan children (make grey) → make black
  End: All reachable = black, unreachable = white (freed)
  
  Write barrier ensures correctness during concurrent marking:
  When application modifies a pointer during GC marking,
  the write barrier records the change so GC doesn't miss objects.
` + "```" + `

**GC Tuning:**
` + "```" + `
GOGC (default: 100):
  Controls GC frequency
  GC triggers when heap grows by GOGC% since last collection
  
  GOGC=100 → GC when heap doubles (default)
  GOGC=50  → GC when heap grows 50% (more frequent, less memory)
  GOGC=200 → GC when heap triples (less frequent, more memory)
  GOGC=off → Disable GC entirely
  
  Lower GOGC → more CPU for GC, less memory used
  Higher GOGC → less CPU for GC, more memory used

GOMEMLIMIT (Go 1.19+):
  Sets a soft memory limit for the Go runtime
  
  GOMEMLIMIT=1GiB  → Target ~1 GiB total memory
  GOMEMLIMIT=512MiB → Target ~512 MiB total memory
  
  When near the limit:
    - GC runs more aggressively
    - GOGC effectively decreases
    - Prevents OOM in containerized environments
    
  Common pattern for containers:
    GOGC=off + GOMEMLIMIT=<container limit * 0.8>
    GC only runs when approaching memory limit
    Eliminates unnecessary GC cycles
    
  Example: Container with 2 GiB limit
    GOGC=off GOMEMLIMIT=1600MiB ./myapp

runtime.SetGCPercent(percent int) → set GOGC programmatically
debug.SetMemoryLimit(limit int64) → set GOMEMLIMIT programmatically
debug.SetGCPercent(-1) → disable GC (equivalent to GOGC=off)

GC diagnostics:
  GODEBUG=gctrace=1 → print GC stats to stderr
  
  Output: gc 1 @0.012s 2%: 0.010+0.33+0.020 ms clock, ...
    gc N         : Nth collection
    @0.012s      : Time since program start
    2%           : % of CPU spent in GC
    0.010+0.33+0.020: STW1 + concurrent + STW2 durations
` + "```" + `

**Reducing GC Pressure:**
` + "```" + `
1. Object pooling (sync.Pool):
   var bufPool = sync.Pool{
       New: func() any { return new(bytes.Buffer) },
   }
   
   buf := bufPool.Get().(*bytes.Buffer)
   buf.Reset()
   defer bufPool.Put(buf)
   // Use buf

2. Pre-allocate slices:
   ✗ var items []Item
     for _, raw := range data {
         items = append(items, parse(raw)) // Grows, allocates
     }
   
   ✓ items := make([]Item, 0, len(data)) // Pre-allocate
     for _, raw := range data {
         items = append(items, parse(raw)) // No reallocation
     }

3. Avoid string concatenation in loops:
   ✗ s := ""
     for i := 0; i < 1000; i++ { s += "x" } // O(n²) allocations
   
   ✓ var b strings.Builder
     for i := 0; i < 1000; i++ { b.WriteString("x") } // O(n)
     s := b.String()

4. Use value types (stack allocation):
   ✗ type Point struct { X, Y float64 }
     p := &Point{1, 2}  // May escape to heap
   
   ✓ p := Point{1, 2}   // Stays on stack (no GC)

5. Reuse structures:
   ✗ for range data { result := make(map[string]int) }
   ✓ result := make(map[string]int)
     for range data { clear(result) /* Go 1.21 */ }

6. Batch allocations:
   Allocate a large slice, sub-slice from it
   One allocation instead of many small ones
` + "```" + ``,
					CodeExamples: `// GC analysis and optimization patterns
package main

import (
    "bytes"
    "fmt"
    "runtime"
    "runtime/debug"
    "strings"
    "sync"
    "time"
)

// Memory stats helper
func printMemStats(label string) {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("  [%s] Alloc=%d KB, TotalAlloc=%d KB, Sys=%d KB, NumGC=%d\n",
        label,
        m.Alloc/1024,
        m.TotalAlloc/1024,
        m.Sys/1024,
        m.NumGC,
    )
}

// Demo 1: Effect of GOGC
func demoGOGC() {
    fmt.Println("=== GOGC Effect ===")
    
    // Default GOGC=100
    printMemStats("before")
    
    // Create garbage
    for i := 0; i < 100000; i++ {
        _ = make([]byte, 1024)
    }
    printMemStats("after 100K allocs (GOGC=100)")
    
    var numGCBefore uint32
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    numGCBefore = m.NumGC
    
    // Increase GOGC → fewer collections
    old := debug.SetGCPercent(400)
    for i := 0; i < 100000; i++ {
        _ = make([]byte, 1024)
    }
    runtime.ReadMemStats(&m)
    fmt.Printf("  GOGC=400: %d GC cycles for 100K allocs\n", m.NumGC-numGCBefore)
    
    // Restore
    debug.SetGCPercent(old)
}

// Demo 2: sync.Pool reduces allocations
func demoSyncPool() {
    fmt.Println("\n=== sync.Pool Optimization ===")
    
    // Without pool
    start := time.Now()
    var totalAlloc1 uint64
    {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        before := m.TotalAlloc
        
        for i := 0; i < 100000; i++ {
            buf := new(bytes.Buffer)
            buf.WriteString("hello world")
            _ = buf.String()
        }
        
        runtime.ReadMemStats(&m)
        totalAlloc1 = m.TotalAlloc - before
    }
    d1 := time.Since(start)
    
    // With pool
    pool := sync.Pool{
        New: func() any { return new(bytes.Buffer) },
    }
    
    start = time.Now()
    var totalAlloc2 uint64
    {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        before := m.TotalAlloc
        
        for i := 0; i < 100000; i++ {
            buf := pool.Get().(*bytes.Buffer)
            buf.Reset()
            buf.WriteString("hello world")
            _ = buf.String()
            pool.Put(buf)
        }
        
        runtime.ReadMemStats(&m)
        totalAlloc2 = m.TotalAlloc - before
    }
    d2 := time.Since(start)
    
    fmt.Printf("  Without pool: %d KB allocated in %v\n", totalAlloc1/1024, d1)
    fmt.Printf("  With pool:    %d KB allocated in %v\n", totalAlloc2/1024, d2)
    fmt.Printf("  Reduction:    %.1f%%\n", (1-float64(totalAlloc2)/float64(totalAlloc1))*100)
}

// Demo 3: Pre-allocation vs append growth
func demoPreAllocation() {
    fmt.Println("\n=== Pre-allocation vs Dynamic Growth ===")
    n := 1000000
    
    // Dynamic growth
    var m1Before, m1After runtime.MemStats
    runtime.ReadMemStats(&m1Before)
    start := time.Now()
    
    var s1 []int
    for i := 0; i < n; i++ {
        s1 = append(s1, i)
    }
    d1 := time.Since(start)
    runtime.ReadMemStats(&m1After)
    
    // Pre-allocated
    var m2Before, m2After runtime.MemStats
    runtime.ReadMemStats(&m2Before)
    start = time.Now()
    
    s2 := make([]int, 0, n)
    for i := 0; i < n; i++ {
        s2 = append(s2, i)
    }
    d2 := time.Since(start)
    runtime.ReadMemStats(&m2After)
    
    fmt.Printf("  Dynamic:       %v, allocs=%d KB\n", d1, (m1After.TotalAlloc-m1Before.TotalAlloc)/1024)
    fmt.Printf("  Pre-allocated: %v, allocs=%d KB\n", d2, (m2After.TotalAlloc-m2Before.TotalAlloc)/1024)
    _ = s1
    _ = s2
}

// Demo 4: String building
func demoStringBuilding() {
    fmt.Println("\n=== String Building ===")
    n := 50000
    
    // String concatenation (O(n²))
    start := time.Now()
    s := ""
    for i := 0; i < n; i++ {
        s += "x"
    }
    d1 := time.Since(start)
    
    // strings.Builder (O(n))
    start = time.Now()
    var b strings.Builder
    for i := 0; i < n; i++ {
        b.WriteString("x")
    }
    _ = b.String()
    d2 := time.Since(start)
    
    // strings.Builder with pre-allocation
    start = time.Now()
    var b2 strings.Builder
    b2.Grow(n)
    for i := 0; i < n; i++ {
        b2.WriteString("x")
    }
    _ = b2.String()
    d3 := time.Since(start)
    
    fmt.Printf("  Concatenation: %v\n", d1)
    fmt.Printf("  Builder:       %v\n", d2)
    fmt.Printf("  Builder+Grow:  %v\n", d3)
    _ = s
}

// Demo 5: Value types vs pointer types
func demoValueVsPointer() {
    fmt.Println("\n=== Value vs Pointer (Stack vs Heap) ===")
    
    type Point struct{ X, Y float64 }
    
    n := 1000000
    
    // Pointers (may escape to heap)
    var mBefore, mAfter runtime.MemStats
    runtime.ReadMemStats(&mBefore)
    
    points := make([]*Point, n)
    for i := 0; i < n; i++ {
        points[i] = &Point{X: float64(i), Y: float64(i)}
    }
    runtime.ReadMemStats(&mAfter)
    allocPointers := (mAfter.TotalAlloc - mBefore.TotalAlloc) / 1024
    
    // Values (stays on stack or contiguous memory)
    runtime.ReadMemStats(&mBefore)
    
    values := make([]Point, n)
    for i := 0; i < n; i++ {
        values[i] = Point{X: float64(i), Y: float64(i)}
    }
    runtime.ReadMemStats(&mAfter)
    allocValues := (mAfter.TotalAlloc - mBefore.TotalAlloc) / 1024
    
    fmt.Printf("  Pointers: %d KB allocated (%d objects + slice)\n", allocPointers, n)
    fmt.Printf("  Values:   %d KB allocated (contiguous slice)\n", allocValues)
    _ = points
    _ = values
}

func main() {
    demoGOGC()
    demoSyncPool()
    demoPreAllocation()
    demoStringBuilding()
    demoValueVsPointer()
    
    fmt.Println("\n=== Final Stats ===")
    printMemStats("end")
}`,
				},
				{
					Title: "Memory Allocator and Stack Management",
					Content: `Go's memory allocator is designed for high-concurrency workloads. Understanding how Go manages memory helps you write efficient programs and debug memory issues.

**Memory Allocator Architecture:**
` + "```" + `
Go's allocator is based on TCMalloc (Thread-Caching Malloc):

  Per-P mcache  ←→  Central mcentral  ←→  Global mheap  ←→  OS
  (per goroutine)   (per size class)      (all memory)
  
Allocation path for small objects (≤32 KB):
  1. Try P's mcache (no lock needed!)
  2. If mcache empty → get span from mcentral (lock per size class)
  3. If mcentral empty → get pages from mheap (global lock)
  4. If mheap empty → get from OS (mmap/sbrk)

Size classes:
  Go has ~67 size classes from 8 bytes to 32 KB
  Each size class has its own free list
  Reduces fragmentation: 8B object → 8B slot, not 4KB page
  
  Examples:
  8, 16, 24, 32, 48, 64, 80, 96, 112, 128, ...
  Requested 20 bytes → allocated from 24-byte class (12.5% waste)
  Max internal fragmentation: ~23%

Large objects (>32 KB):
  Allocated directly from mheap with page-level granularity
  Each large allocation is its own span
  No size class rounding

Memory terminology:
  mspan:     A contiguous block of pages for one size class
  mcache:    Per-P cache of mspans (no lock!)
  mcentral:  Shared cache of mspans (per size class)
  mheap:     The entire heap, manages page-level allocation
  
Why per-P cache:
  Most allocations don't need any locking
  Lock-free fast path → excellent scalability
  Similar to how goroutine scheduling is per-P
` + "```" + `

**Stack Management:**
` + "```" + `
Goroutine stacks in Go:

Initial size: 2 KB (since Go 1.4, was 8 KB before)
  Compare: OS thread stack = 1-8 MB default
  This is why millions of goroutines are possible!

Stack growth (contiguous stacks, since Go 1.4):
  1. Function prologue checks: is stack big enough?
  2. If not → runtime.morestack called
  3. Allocate new stack, 2x current size
  4. Copy entire old stack to new stack
  5. Update all pointers to the old stack
  6. Free old stack
  
  Stack sizes: 2KB → 4KB → 8KB → 16KB → ... (doubling)
  Max stack: 1 GB (configurable via runtime.SetMaxStack)

Stack shrinking:
  During GC, stacks that used < 1/4 capacity are shrunk by 50%
  Prevents goroutines from holding large stacks forever

Escape analysis:
  Go compiler decides: stack or heap?
  Variables that "escape" the function go to heap
  
  func foo() *int {
      x := 42
      return &x    // x escapes! → heap allocated
  }
  
  func bar() int {
      x := 42
      return x     // x doesn't escape → stack allocated
  }
  
  See escape analysis:
    go build -gcflags="-m" ./...
    
  Output:
    ./main.go:5:2: moved to heap: x
    ./main.go:10:2: x does not escape

Why escape matters:
  Stack allocation: FREE (just move stack pointer)
  Heap allocation: Costs ~50-100ns + future GC work
  
  Hot loop with heap escapes → GC pressure → latency spikes
  
Common escapes:
  - Returning pointer to local variable
  - Storing in interface (sometimes)
  - Captured by closure that outlives function
  - Sent to channel
  - Stored in slice that grows (reallocation)
  - Large objects (compiler decides based on size)
` + "```" + `

**Profiling Memory:**
` + "```" + `
pprof for memory profiling:

  import _ "net/http/pprof"
  go http.ListenAndServe(":6060", nil)
  
  // In another terminal:
  go tool pprof http://localhost:6060/debug/pprof/heap
  
  pprof commands:
    top         → Top memory consumers
    list func   → Show annotated source
    web         → Open SVG in browser
    
  Profile types:
    /debug/pprof/heap       → Current heap usage
    /debug/pprof/allocs     → All allocations since start
    /debug/pprof/goroutine  → All goroutines
    /debug/pprof/threadcreate → OS threads

Programmatic profiling:
  f, _ := os.Create("mem.prof")
  runtime.GC() // Clean up before profiling
  pprof.WriteHeapProfile(f)
  f.Close()
  
  // Analyze:
  go tool pprof mem.prof

Memory metrics to watch:
  runtime.MemStats fields:
    Alloc        → Currently allocated bytes (live objects)
    TotalAlloc   → Cumulative bytes allocated (ever)
    Sys          → Total bytes obtained from OS
    HeapAlloc    → Heap bytes allocated
    HeapInuse    → Heap bytes in use
    HeapIdle     → Heap bytes waiting to be used
    HeapReleased → Heap bytes returned to OS
    StackInuse   → Stack bytes in use
    NumGC        → Number of GC cycles
    PauseTotalNs → Total GC pause time

Common memory issues:
  1. Memory leak: goroutine leak (goroutine holds reference)
  2. OOM: unbounded caching/buffering
  3. High GC%: too many small allocations
  4. RSS not decreasing: Go returns memory slowly (MADV_FREE)
     Use debug.FreeOSMemory() to force return
` + "```" + ``,
					CodeExamples: `// Memory profiling and escape analysis demo
package main

import (
    "fmt"
    "runtime"
    "time"
    "unsafe"
)

// Escape analysis demonstration

// This does NOT escape (stays on stack)
//
//go:noinline
func stackAlloc() int {
    x := 42
    return x
}

// This DOES escape (pointer returned)
//
//go:noinline
func heapAlloc() *int {
    x := 42
    return &x
}

// Interface may cause escape
//
//go:noinline
func interfaceEscape(n int) any {
    return n // n may escape due to interface boxing
}

// Closure escape
//
//go:noinline
func closureEscape() func() int {
    x := 42
    return func() int { return x } // x escapes (captured by closure)
}

// Memory layout inspection
func inspectLayout() {
    fmt.Println("=== Memory Layout ===")
    
    // Size and alignment of types
    type Example struct {
        a bool    // 1 byte
        b int64   // 8 bytes
        c bool    // 1 byte
        d int32   // 4 bytes
    }
    
    type OptimizedExample struct {
        b int64   // 8 bytes
        d int32   // 4 bytes
        a bool    // 1 byte
        c bool    // 1 byte
        // 2 bytes padding
    }
    
    fmt.Printf("  Example:          size=%d, align=%d\n", 
        unsafe.Sizeof(Example{}), unsafe.Alignof(Example{}))
    fmt.Printf("  OptimizedExample: size=%d, align=%d\n",
        unsafe.Sizeof(OptimizedExample{}), unsafe.Alignof(OptimizedExample{}))
    
    // Field offsets
    var e Example
    fmt.Printf("\n  Example field offsets:\n")
    fmt.Printf("    a (bool):  offset=%d\n", unsafe.Offsetof(e.a))
    fmt.Printf("    b (int64): offset=%d\n", unsafe.Offsetof(e.b))
    fmt.Printf("    c (bool):  offset=%d\n", unsafe.Offsetof(e.c))
    fmt.Printf("    d (int32): offset=%d\n", unsafe.Offsetof(e.d))
    
    var o OptimizedExample
    fmt.Printf("\n  OptimizedExample field offsets:\n")
    fmt.Printf("    b (int64): offset=%d\n", unsafe.Offsetof(o.b))
    fmt.Printf("    d (int32): offset=%d\n", unsafe.Offsetof(o.d))
    fmt.Printf("    a (bool):  offset=%d\n", unsafe.Offsetof(o.a))
    fmt.Printf("    c (bool):  offset=%d\n", unsafe.Offsetof(o.c))
    
    fmt.Printf("\n  Savings: %d bytes per struct\n",
        unsafe.Sizeof(Example{}) - unsafe.Sizeof(OptimizedExample{}))
}

// Stack growth demonstration
func deepRecursion(depth int) int {
    if depth <= 0 {
        // Print current goroutine stack size
        var buf [64]byte
        runtime.Stack(buf[:], false)
        return 0
    }
    // Allocate some stack space
    var arr [256]byte
    arr[0] = byte(depth)
    return deepRecursion(depth-1) + int(arr[0])
}

// Memory stats comparison
func benchmarkAllocations() {
    fmt.Println("\n=== Allocation Patterns Benchmark ===")
    
    n := 1000000
    
    // Pattern 1: Many small heap allocations
    var m1 runtime.MemStats
    runtime.ReadMemStats(&m1)
    start := time.Now()
    
    for i := 0; i < n; i++ {
        p := new(int)
        *p = i
        _ = p
    }
    
    var m2 runtime.MemStats
    runtime.ReadMemStats(&m2)
    
    fmt.Printf("  Small heap allocs (%d):\n", n)
    fmt.Printf("    Time: %v\n", time.Since(start))
    fmt.Printf("    Bytes: %d KB\n", (m2.TotalAlloc-m1.TotalAlloc)/1024)
    fmt.Printf("    Mallocs: %d\n", m2.Mallocs-m1.Mallocs)
    
    // Pattern 2: Batch allocation (slice)
    runtime.ReadMemStats(&m1)
    start = time.Now()
    
    batch := make([]int, n)
    for i := 0; i < n; i++ {
        batch[i] = i
    }
    
    runtime.ReadMemStats(&m2)
    
    fmt.Printf("\n  Batch allocation (slice of %d):\n", n)
    fmt.Printf("    Time: %v\n", time.Since(start))
    fmt.Printf("    Bytes: %d KB\n", (m2.TotalAlloc-m1.TotalAlloc)/1024)
    fmt.Printf("    Mallocs: %d\n", m2.Mallocs-m1.Mallocs)
    _ = batch
    
    // Pattern 3: Stack allocation (no escape)
    runtime.ReadMemStats(&m1)
    start = time.Now()
    
    sum := 0
    for i := 0; i < n; i++ {
        x := i * 2
        sum += x
    }
    
    runtime.ReadMemStats(&m2)
    
    fmt.Printf("\n  Stack allocation (no escape):\n")
    fmt.Printf("    Time: %v\n", time.Since(start))
    fmt.Printf("    Bytes: %d KB\n", (m2.TotalAlloc-m1.TotalAlloc)/1024)
    fmt.Printf("    Mallocs: %d\n", m2.Mallocs-m1.Mallocs)
    _ = sum
}

func demoGCPacing() {
    fmt.Println("\n=== GC Pacing ===")
    
    var stats debug.GCStats
    
    // Create some allocations and watch GC behavior
    var m runtime.MemStats
    for i := 0; i < 10; i++ {
        // Allocate ~1MB
        data := make([]byte, 1<<20)
        data[0] = byte(i)
        _ = data
        
        runtime.ReadMemStats(&m)
        debug.ReadGCStats(&stats)
        
        if len(stats.Pause) > 0 {
            fmt.Printf("  Iter %d: HeapAlloc=%d KB, NumGC=%d, LastPause=%v\n",
                i, m.HeapAlloc/1024, m.NumGC, stats.Pause[0])
        }
    }
}

func main() {
    inspectLayout()
    
    fmt.Println("\n=== Escape Analysis Results ===")
    fmt.Println("  (Run: go build -gcflags='-m' to see)")
    
    v := stackAlloc()
    fmt.Printf("  stackAlloc: %d (stack)\n", v)
    
    p := heapAlloc()
    fmt.Printf("  heapAlloc: %d (heap)\n", *p)
    
    i := interfaceEscape(42)
    fmt.Printf("  interfaceEscape: %v (may escape)\n", i)
    
    f := closureEscape()
    fmt.Printf("  closureEscape: %d (captures variable)\n", f())
    
    // Stack growth demo
    fmt.Println("\n=== Stack Growth ===")
    fmt.Printf("  Deep recursion result: %d\n", deepRecursion(100))
    
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("  Stack in use: %d KB\n", m.StackInuse/1024)
    
    benchmarkAllocations()
    demoGCPacing()
}`,
				},
				{
					Title: "Performance Profiling and Optimization",
					Content: `Go provides world-class profiling tools. Mastering them is essential for identifying and fixing performance bottlenecks.

**pprof Profiling:**
` + "```" + `
CPU profiling:
  import "runtime/pprof"
  
  f, _ := os.Create("cpu.prof")
  pprof.StartCPUProfile(f)
  defer pprof.StopCPUProfile()
  // ... run workload
  
  // Analyze:
  go tool pprof cpu.prof
  
  Interactive commands:
    top 20          → Top 20 functions by CPU
    top -cum        → By cumulative time (including callees)
    list funcName   → Annotated source
    web             → Graphical call graph
    peek funcName   → Callers and callees
    disasm funcName → Assembly view

HTTP server profiling:
  import _ "net/http/pprof"
  
  // In separate goroutine:
  go http.ListenAndServe("localhost:6060", nil)
  
  // Capture 30-second CPU profile:
  go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
  
  // Available profiles:
  /debug/pprof/profile     → CPU
  /debug/pprof/heap        → Memory (live objects)
  /debug/pprof/allocs      → All allocations
  /debug/pprof/goroutine   → Goroutine stacks
  /debug/pprof/block       → Blocking (mutex/channel waits)
  /debug/pprof/mutex       → Mutex contention

Flamegraph (best visualization):
  go tool pprof -http=:8080 cpu.prof
  Opens web UI with flamegraph, graph, source views
` + "```" + `

**Benchmarking:**
` + "```" + `
Go has built-in benchmarking in testing package:

func BenchmarkFoo(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Foo()
    }
}

Run benchmarks:
  go test -bench=. -benchmem ./...
  
  Output:
  BenchmarkFoo-8    5000000    234 ns/op    48 B/op    2 allocs/op
  
  Meaning:
    -8:       GOMAXPROCS
    5000000:  Number of iterations
    234 ns/op: Time per operation
    48 B/op:   Bytes allocated per operation
    2 allocs/op: Heap allocations per operation

Benchmark techniques:
  
  // Reset timer (exclude setup):
  func BenchmarkComplex(b *testing.B) {
      data := expensiveSetup()
      b.ResetTimer()
      for i := 0; i < b.N; i++ {
          Process(data)
      }
  }
  
  // Sub-benchmarks:
  func BenchmarkSort(b *testing.B) {
      for _, size := range []int{10, 100, 1000, 10000} {
          b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
              data := generateData(size)
              b.ResetTimer()
              for i := 0; i < b.N; i++ {
                  sort.Ints(data)
              }
          })
      }
  }
  
  // Report custom metrics:
  func BenchmarkThroughput(b *testing.B) {
      for i := 0; i < b.N; i++ {
          processBytes(data)
      }
      b.SetBytes(int64(len(data))) // Reports MB/s
  }
  
  // Compare benchmarks (benchstat):
  go test -bench=. -count=10 > old.txt
  // Make changes
  go test -bench=. -count=10 > new.txt
  benchstat old.txt new.txt
` + "```" + `

**Tracing:**
` + "```" + `
Go execution tracer provides timeline view:

  import "runtime/trace"
  
  f, _ := os.Create("trace.out")
  trace.Start(f)
  defer trace.Stop()
  
  // Or via HTTP:
  curl -o trace.out http://localhost:6060/debug/pprof/trace?seconds=5
  
  // Analyze:
  go tool trace trace.out
  Opens web UI showing:
    - Goroutine execution timeline
    - Network blocking
    - Syscalls
    - GC events
    - Scheduler events
    
  Very useful for:
    - Understanding concurrency behavior
    - Finding goroutine starvation
    - Visualizing GC impact
    - Diagnosing latency issues

Optimization checklist:
  1. Profile first (don't guess!)
  2. Focus on hot paths (top 5 functions)
  3. Check allocations (benchmem)
  4. Look at escape analysis
  5. Consider algorithmic improvements first
  6. Then micro-optimizations if needed
  7. Verify with benchstat (statistical significance)
` + "```" + ``,
					CodeExamples: `// Profiling and optimization patterns
package main

import (
    "fmt"
    "math"
    "sort"
    "strings"
    "time"
)

// Example: Optimizing a function step by step

// Version 1: Naive (lots of allocations)
func findDuplicatesNaive(items []string) []string {
    var result []string
    for i, a := range items {
        for j, b := range items {
            if i != j && a == b {
                // Check if already in result
                found := false
                for _, r := range result {
                    if r == a {
                        found = true
                        break
                    }
                }
                if !found {
                    result = append(result, a)
                }
            }
        }
    }
    return result
}

// Version 2: Use map (O(n) instead of O(n³))
func findDuplicatesMap(items []string) []string {
    counts := make(map[string]int, len(items))
    for _, item := range items {
        counts[item]++
    }
    
    var result []string
    for item, count := range counts {
        if count > 1 {
            result = append(result, item)
        }
    }
    return result
}

// Version 3: Pre-allocate and single pass
func findDuplicatesOptimized(items []string) []string {
    seen := make(map[string]struct{}, len(items))
    dups := make(map[string]struct{})
    
    for _, item := range items {
        if _, ok := seen[item]; ok {
            dups[item] = struct{}{}
        }
        seen[item] = struct{}{}
    }
    
    result := make([]string, 0, len(dups))
    for item := range dups {
        result = append(result, item)
    }
    return result
}

// Benchmark comparison helper
func benchFunc(name string, fn func(), iterations int) time.Duration {
    start := time.Now()
    for i := 0; i < iterations; i++ {
        fn()
    }
    d := time.Since(start)
    fmt.Printf("  %-25s %v/op (%d iterations)\n", name, d/time.Duration(iterations), iterations)
    return d
}

// String building optimization comparison
func stringBuildComparison() {
    fmt.Println("\n=== String Building Optimization ===")
    n := 10000
    words := make([]string, n)
    for i := range words {
        words[i] = fmt.Sprintf("word%d", i)
    }
    
    // Naive: string concatenation
    benchFunc("concatenation", func() {
        s := ""
        for _, w := range words {
            s += w + " "
        }
        _ = s
    }, 10)
    
    // strings.Join
    benchFunc("strings.Join", func() {
        _ = strings.Join(words, " ")
    }, 100)
    
    // strings.Builder
    benchFunc("strings.Builder", func() {
        var b strings.Builder
        for _, w := range words {
            b.WriteString(w)
            b.WriteByte(' ')
        }
        _ = b.String()
    }, 100)
    
    // strings.Builder with Grow
    benchFunc("Builder+Grow", func() {
        totalLen := 0
        for _, w := range words {
            totalLen += len(w) + 1
        }
        var b strings.Builder
        b.Grow(totalLen)
        for _, w := range words {
            b.WriteString(w)
            b.WriteByte(' ')
        }
        _ = b.String()
    }, 100)
}

// Sort optimization comparison
func sortComparison() {
    fmt.Println("\n=== Sort Optimization ===")
    
    for _, size := range []int{100, 1000, 10000} {
        fmt.Printf("\n  Size: %d\n", size)
        data := make([]int, size)
        
        // sort.Ints (interface-based)
        benchFunc(fmt.Sprintf("sort.Ints(n=%d)", size), func() {
            for i := range data {
                data[i] = size - i
            }
            sort.Ints(data)
        }, 1000)
        
        // sort.Slice (closure-based)
        benchFunc(fmt.Sprintf("sort.Slice(n=%d)", size), func() {
            for i := range data {
                data[i] = size - i
            }
            sort.Slice(data, func(i, j int) bool {
                return data[i] < data[j]
            })
        }, 1000)
    }
}

// Map vs slice for lookups
func lookupComparison() {
    fmt.Println("\n=== Lookup: Map vs Sorted Slice ===")
    
    sizes := []int{10, 100, 1000}
    
    for _, size := range sizes {
        // Build map
        m := make(map[int]bool, size)
        for i := 0; i < size; i++ {
            m[i] = true
        }
        
        // Build sorted slice
        s := make([]int, size)
        for i := 0; i < size; i++ {
            s[i] = i
        }
        
        target := size / 2 // Search for middle element
        
        fmt.Printf("\n  Size: %d\n", size)
        
        // Map lookup O(1)
        benchFunc(fmt.Sprintf("map[n=%d]", size), func() {
            _ = m[target]
        }, 100000)
        
        // Binary search O(log n)
        benchFunc(fmt.Sprintf("binary_search[n=%d]", size), func() {
            sort.SearchInts(s, target)
        }, 100000)
    }
}

// Interface vs concrete type
type Shape interface {
    Area() float64
}

type Circle struct { Radius float64 }
func (c Circle) Area() float64 { return math.Pi * c.Radius * c.Radius }

type Rectangle struct { Width, Height float64 }
func (r Rectangle) Area() float64 { return r.Width * r.Height }

func interfaceOverhead() {
    fmt.Println("\n=== Interface Overhead ===")
    n := 10000000
    
    // Direct (no interface)
    c := Circle{Radius: 5}
    benchFunc("direct call", func() {
        for i := 0; i < n; i++ {
            _ = c.Area()
        }
    }, 1)
    
    // Through interface
    var s Shape = Circle{Radius: 5}
    benchFunc("interface call", func() {
        for i := 0; i < n; i++ {
            _ = s.Area()
        }
    }, 1)
}

func main() {
    // Duplicate finding comparison
    fmt.Println("=== Duplicate Finding Optimization ===")
    items := make([]string, 1000)
    for i := range items {
        items[i] = fmt.Sprintf("item%d", i%500) // 500 unique, 500 duplicates
    }
    
    benchFunc("naive O(n³)", func() { findDuplicatesNaive(items[:100]) }, 10) // Only 100 items
    benchFunc("map O(n)", func() { findDuplicatesMap(items) }, 1000)
    benchFunc("optimized O(n)", func() { findDuplicatesOptimized(items) }, 1000)
    
    stringBuildComparison()
    sortComparison()
    lookupComparison()
    interfaceOverhead()
}`,
				},
			},
		},
	})
}
