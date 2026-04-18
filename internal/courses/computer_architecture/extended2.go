package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2820,
			Title:       "Cache Effects on Software Performance",
			Description: "Understand how CPU caches affect real software performance: cache lines, false sharing, prefetching, and writing cache-friendly code.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Cache Lines and Memory Access Patterns",
					Content: `Every programmer should understand CPU caches because they can cause 100x performance differences in real code.

**Cache Line Basics:**
- CPU caches don't fetch individual bytes — they fetch **cache lines** (typically 64 bytes)
- When you access one byte, the CPU loads the entire 64-byte cache line containing it
- Accessing the next byte in the same cache line is essentially free (already cached)

**Why This Matters for Arrays vs Linked Lists:**
` + "```" + `
Array (contiguous memory — cache-friendly):
┌────────────────────────────────────────────────────────────────┐
│ elem0 │ elem1 │ elem2 │ elem3 │ elem4 │ elem5 │ elem6 │ elem7│  ← One cache line
└────────────────────────────────────────────────────────────────┘
   ↑ Access elem0 → ALL elements loaded into cache. 
     elem1-elem7 are "free" to access.

Linked List (scattered memory — cache-hostile):
┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
│ Node1 │────→│ Node2 │────→│ Node3 │────→│ Node4 │
└───────┘     └───────┘     └───────┘     └───────┘
   addr: 0x100  addr: 0x8400  addr: 0x200   addr: 0x9100
   ↑ Each node may be on a DIFFERENT cache line → cache miss on every access
` + "```" + `

**Benchmark Results (typical):**
- Sequential array access: ~1 ns per element
- Random linked list traversal: ~50-100 ns per element
- That's a 50-100x difference from cache effects alone!

**Row-Major vs Column-Major Traversal:**
` + "```" + `
Matrix[4][4] stored in row-major order (C, Go, Python):
┌──────────────────────────┐
│ [0][0] [0][1] [0][2] [0][3] │  ← Row 0 (contiguous in memory)
│ [1][0] [1][1] [1][2] [1][3] │  ← Row 1 (contiguous in memory)
│ [2][0] [2][1] [2][2] [2][3] │  ← Row 2
│ [3][0] [3][1] [3][2] [3][3] │  ← Row 3
└──────────────────────────┘

FAST: for i in rows: for j in cols: access M[i][j]  (sequential)
SLOW: for j in cols: for i in rows: access M[i][j]  (jumping between rows)
` + "```" + `

This is why matrix multiplication order matters so much — a naive implementation can be 10x slower than one that respects cache lines.`,
					CodeExamples: `// Go example: Cache-friendly vs cache-hostile array traversal
package main

import (
    "fmt"
    "time"
)

const N = 10000

func main() {
    // Matrix allocated as 2D array (contiguous in memory)
    var matrix [N][N]int

    // FAST: Row-major traversal (cache-friendly)
    start := time.Now()
    sum := 0
    for i := 0; i < N; i++ {
        for j := 0; j < N; j++ {
            sum += matrix[i][j]  // Sequential memory access
        }
    }
    fmt.Printf("Row-major:    %v\n", time.Since(start))

    // SLOW: Column-major traversal (cache-hostile)
    start = time.Now()
    sum = 0
    for j := 0; j < N; j++ {
        for i := 0; i < N; i++ {
            sum += matrix[i][j]  // Jumping N*8 bytes each access
        }
    }
    fmt.Printf("Column-major: %v\n", time.Since(start))
    // Column-major is typically 3-10x slower!
}

# Python: Same effect with NumPy
import numpy as np
import time

arr = np.zeros((10000, 10000))

# Fast: row-major (C order, default)
start = time.time()
total = arr.sum(axis=1).sum()  # Sum along rows
print(f"Row-major: {time.time() - start:.3f}s")

# Slow: column-major
start = time.time()
total = arr.sum(axis=0).sum()  # Sum along columns
print(f"Col-major: {time.time() - start:.3f}s")`,
				},
				{
					Title: "False Sharing in Multi-Core Systems",
					Content: `False sharing is one of the most insidious performance bugs in concurrent programming. It happens when two CPU cores modify different variables that happen to sit on the **same cache line**.

**How False Sharing Works:**
` + "```" + `
struct Counters {
    counterA int64  // Core 0 writes this
    counterB int64  // Core 1 writes this
}

Memory layout:
┌─────────────────────────────────────────────────────────────┐
│ counterA (8 bytes) │ counterB (8 bytes) │ ... padding ...   │ ← ONE cache line (64 bytes)
└─────────────────────────────────────────────────────────────┘
       ↑                    ↑
    Core 0 writes       Core 1 writes

Problem: When Core 0 writes counterA, the ENTIRE cache line is invalidated
on Core 1. Core 1 must reload the cache line before it can write counterB.
And vice versa. They keep invalidating each other's cache!
` + "```" + `

**The Fix: Cache Line Padding**
` + "```" + `
struct Counters {
    counterA int64
    _pad     [56]byte  // Pad to fill the rest of the 64-byte cache line
    counterB int64
}

Now counterA and counterB are on DIFFERENT cache lines.
No more cross-core invalidation!
` + "```" + `

**Impact:**
False sharing can cause 10-50x slowdown in concurrent code. It's invisible in profilers — you'll just see "slow" atomic/mutex operations.

**Detection:**
- Linux: ` + "`" + `perf c2c` + "`" + ` detects cache-to-cache transfers
- Intel: VTune shows false sharing hotspots
- Pattern: If adding padding between atomically-updated fields dramatically improves performance, you had false sharing

**Real-World Examples:**
- Go's runtime pads certain internal structures for this exact reason
- Java uses ` + "`" + `@Contended` + "`" + ` annotation to prevent false sharing
- Database engines pad lock arrays to avoid false sharing`,
					CodeExamples: `// Go: False sharing demonstration
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

// BAD: counters on same cache line
type SharedCounters struct {
    a int64
    b int64
}

// GOOD: counters on separate cache lines
type PaddedCounters struct {
    a   int64
    _   [56]byte // Cache line padding (64 - 8 = 56)
    b   int64
}

func bench(name string, incA, incB func()) {
    const iterations = 100_000_000
    var wg sync.WaitGroup
    wg.Add(2)

    start := time.Now()
    go func() {
        for i := 0; i < iterations; i++ { incA() }
        wg.Done()
    }()
    go func() {
        for i := 0; i < iterations; i++ { incB() }
        wg.Done()
    }()
    wg.Wait()
    fmt.Printf("%s: %v\n", name, time.Since(start))
}

func main() {
    // With false sharing
    shared := &SharedCounters{}
    bench("False sharing",
        func() { atomic.AddInt64(&shared.a, 1) },
        func() { atomic.AddInt64(&shared.b, 1) },
    )

    // Without false sharing  
    padded := &PaddedCounters{}
    bench("Padded (no false sharing)",
        func() { atomic.AddInt64(&padded.a, 1) },
        func() { atomic.AddInt64(&padded.b, 1) },
    )
    // Padded version is typically 2-5x faster!
}`,
				},
				{
					Title: "Memory Alignment and Struct Layout",
					Content: `CPUs access memory most efficiently when data is naturally aligned — when a variable's address is a multiple of its size.

**Alignment Rules (on 64-bit systems):**
` + "```" + `
Type         Size    Alignment
bool         1 byte  1 byte
int8/byte    1 byte  1 byte
int16        2 bytes 2 bytes
int32/float32 4 bytes 4 bytes
int64/float64 8 bytes 8 bytes
pointer      8 bytes 8 bytes
string       16 bytes 8 bytes (pointer + length)
slice        24 bytes 8 bytes (pointer + length + capacity)
` + "```" + `

**Struct Padding:**
The compiler inserts invisible "padding" bytes to maintain alignment:

` + "```" + `
// BAD layout: 24 bytes (with 7 bytes of padding!)
type Bad struct {
    a bool    // 1 byte
    // 7 bytes padding (to align b to 8 bytes)
    b int64   // 8 bytes
    c bool    // 1 byte
    // 7 bytes padding (to align struct to 8 bytes)
}
// Total: 1 + 7 + 8 + 1 + 7 = 24 bytes

// GOOD layout: 16 bytes (only 6 bytes of padding)
type Good struct {
    b int64   // 8 bytes
    a bool    // 1 byte
    c bool    // 1 byte
    // 6 bytes padding
}
// Total: 8 + 1 + 1 + 6 = 16 bytes
` + "```" + `

**The Rule: Order struct fields from largest to smallest.**

**Why This Matters:**
- For one struct: 8 bytes wasted is nothing
- For 10 million structs in a slice: 80 MB wasted
- More data per cache line = faster iteration
- The Go compiler does NOT reorder struct fields (unlike C compilers with some flags)

**Tools:**
- Go: ` + "`" + `go vet -fieldalignment ./...` + "`" + ` detects inefficient struct layouts
- ` + "`" + `unsafe.Sizeof()` + "`" + `, ` + "`" + `unsafe.Alignof()` + "`" + `, ` + "`" + `unsafe.Offsetof()` + "`" + ` show actual sizes and offsets`,
					CodeExamples: `package main

import (
    "fmt"
    "unsafe"
)

// Inefficient layout: 32 bytes
type Inefficient struct {
    a bool    // 1 + 7 padding
    b float64 // 8
    c bool    // 1 + 3 padding
    d int32   // 4
    e bool    // 1 + 7 padding
}

// Efficient layout: 24 bytes (same fields!)
type Efficient struct {
    b float64 // 8
    d int32   // 4
    a bool    // 1
    c bool    // 1
    e bool    // 1 + 1 padding
}

func main() {
    fmt.Printf("Inefficient: %d bytes\n", unsafe.Sizeof(Inefficient{}))  // 32
    fmt.Printf("Efficient:   %d bytes\n", unsafe.Sizeof(Efficient{}))    // 24
    // Saved 8 bytes per struct = 25% reduction

    // For 10 million items:
    // Inefficient: 320 MB
    // Efficient:   240 MB (80 MB saved!)

    // Check alignment
    fmt.Printf("int64 alignment: %d\n", unsafe.Alignof(int64(0)))  // 8
    fmt.Printf("bool alignment:  %d\n", unsafe.Alignof(true))      // 1

    // Check field offsets
    var e Efficient
    fmt.Printf("b offset: %d\n", unsafe.Offsetof(e.b)) // 0
    fmt.Printf("d offset: %d\n", unsafe.Offsetof(e.d)) // 8
    fmt.Printf("a offset: %d\n", unsafe.Offsetof(e.a)) // 12
}

# Run go vet fieldalignment check:
# go vet -vettool=$(which fieldalignment) ./...`,
				},
			},
		},
		{
			ID:          2821,
			Title:       "How Programs Actually Execute",
			Description: "Understand the journey from source code to running program: compilation, linking, loading, system calls, and the role of the OS.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "From Source Code to Execution",
					Content: `Understanding how code becomes a running program helps with debugging, performance tuning, and making architectural decisions.

**The Journey:**
` + "```" + `
Source Code (.go, .c, .py)
    │
    ▼
Compiler/Interpreter
    │
    ├─── Compiled Languages (Go, C, Rust):
    │    │
    │    ▼  Lexer → Parser → AST → Optimizer → Code Generator
    │    │
    │    ▼  Object Files (.o)
    │    │
    │    ▼  Linker → Executable Binary (ELF, Mach-O, PE)
    │    │
    │    ▼  OS Loader → Memory (Text, Data, BSS, Heap, Stack)
    │    │
    │    ▼  CPU Execution (Fetch → Decode → Execute → Writeback)
    │
    └─── Interpreted Languages (Python, JavaScript):
         │
         ▼  Bytecode Compiler → Bytecodes (.pyc)
         │
         ▼  Virtual Machine (CPython VM, V8 Engine)
         │
         ▼  JIT Compiler (optional: V8, PyPy) → Native Code
         │
         ▼  CPU Execution
` + "```" + `

**Process Memory Layout (Linux/macOS):**
` + "```" + `
High Address ┌──────────────────────┐
             │   Kernel Space       │ (not accessible to user code)
             ├──────────────────────┤
             │   Stack              │ ← Grows downward
             │   (local variables,  │   Each thread/goroutine gets its own
             │    return addresses)  │
             ├──────────────────────┤
             │   ↓ grows down       │
             │                      │
             │   ↑ grows up         │
             ├──────────────────────┤
             │   Heap               │ ← Dynamic allocation (malloc, new)
             │   (objects, buffers) │   Managed by GC in Go/Java/Python
             ├──────────────────────┤
             │   BSS                │ ← Uninitialized global variables (zeroed)
             ├──────────────────────┤
             │   Data               │ ← Initialized global variables
             ├──────────────────────┤
             │   Text (Code)        │ ← Machine instructions (read-only)
Low Address  └──────────────────────┘
` + "```" + `

**Key Concepts:**
1. **Static linking** (Go default): All code in one binary. Larger file, no dependencies.
2. **Dynamic linking** (C default): Shared libraries (.so/.dylib/.dll). Smaller binary, needs libraries at runtime.
3. **Position-Independent Code (PIC)**: Required for shared libraries and ASLR (Address Space Layout Randomization).`,
					CodeExamples: `# Examine a Go binary's sections
$ go build -o myapp main.go
$ size myapp                    # Text, Data, BSS sizes
$ file myapp                    # File type and architecture
$ objdump -h myapp | head -20  # Section headers

# On macOS:
$ otool -l myapp | grep -A 3 __TEXT

# Examine a C program:
$ gcc -o hello hello.c
$ readelf -l hello              # Program headers (Linux)
$ nm hello | head -20           # Symbol table

# Examine Go binary symbols:
$ go tool nm myapp | grep main  # Find main package functions

# Check if binary is statically linked:
$ ldd myapp   # Linux: "not a dynamic executable" = static
$ otool -L myapp  # macOS: shows dynamic libraries

# Go cross-compilation is trivial because of static linking:
$ GOOS=linux GOARCH=amd64 go build -o myapp-linux
$ GOOS=darwin GOARCH=arm64 go build -o myapp-mac
$ GOOS=windows GOARCH=amd64 go build -o myapp.exe`,
				},
				{
					Title: "System Calls: The OS Interface",
					Content: `Every time your program reads a file, sends network data, or allocates memory, it makes a **system call** (syscall) — a request to the operating system kernel.

**Why System Calls Matter:**
- They're the ONLY way user programs interact with hardware
- Each syscall has overhead (context switch from user to kernel mode)
- Minimizing syscalls is a key optimization strategy

**Common System Calls:**
` + "```" + `
Category        Linux Syscall    What It Does
─────────────────────────────────────────────────
File I/O        open, read,      Open/read/write/close files
                write, close
Memory          mmap, brk        Map memory, grow heap
Network         socket, connect, Create connections,
                send, recv       send/receive data
Process         fork, exec,      Create/replace/wait for
                wait, exit       processes
Threads         clone, futex     Create threads, synchronize
Signals         sigaction        Register signal handlers
` + "```" + `

**The Syscall Overhead:**
` + "```" + `
User Space                    Kernel Space
─────────────────────────────────────────────
Your code                     
    │                         
    ▼ syscall instruction     
    ├─── Save registers ──────┐
    │                         ▼
    │                    Kernel handler
    │                    (check permissions,
    │                     do the work)
    │                         │
    ├─── Restore registers ◄──┘
    ▼                         
Your code continues           

Each transition costs ~100-1000 ns
Compare to a function call: ~1-5 ns
` + "```" + `

**Why Buffered I/O Exists:**
` + "```" + `
Without buffering: write("H") write("e") write("l") write("l") write("o")
  → 5 syscalls for 5 bytes!

With buffering:    buffer.write("Hello") → flush → write("Hello")
  → 1 syscall for 5 bytes
` + "```" + `

This is why ` + "`" + `bufio.Writer` + "`" + ` in Go and ` + "`" + `BufferedWriter` + "`" + ` in Java exist — they batch many small writes into fewer syscalls.

**Tracing System Calls:**
` + "```" + `
# Linux
strace -c ./myapp            # Count syscalls by type
strace -e trace=open ./myapp # Trace specific syscalls

# macOS
dtruss ./myapp 2>&1 | head -50
` + "```" + ``,
					CodeExamples: `// Go: Demonstrate syscall overhead
package main

import (
    "os"
    "time"
)

func main() {
    // Write 1MB: one byte at a time vs one large write
    data := make([]byte, 1024*1024)

    // SLOW: 1M syscalls
    f1, _ := os.Create("/tmp/slow.bin")
    start := time.Now()
    for i := range data {
        f1.Write(data[i : i+1])  // One syscall per byte!
    }
    f1.Close()
    slow := time.Since(start)

    // FAST: 1 syscall
    f2, _ := os.Create("/tmp/fast.bin")
    start = time.Now()
    f2.Write(data)  // One syscall for entire buffer
    f2.Close()
    fast := time.Since(start)

    fmt.Printf("1M writes of 1 byte:  %v\n", slow)
    fmt.Printf("1 write of 1MB:       %v\n", fast)
    fmt.Printf("Ratio: %.0fx faster\n", float64(slow)/float64(fast))
    // Typically 100-1000x faster!
}

# Trace syscalls on Linux:
# strace -c go run main.go
# 
# Output shows:
# % time     seconds  usecs/call     calls    errors syscall
# ------ ----------- ----------- --------- --------- ----------------
#  95.00    2.500000           2   1048576           write
#   5.00    0.130000           1    123456           read`,
				},
			},
		},
	})
}
