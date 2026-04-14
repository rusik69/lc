package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          1301,
			Title:       "CPU Pipelining and Performance",
			Description: "Learn about instruction pipelining, hazards, performance optimization, cache hierarchy, virtual memory, and I/O systems.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Instruction Pipelining",
					Content: `Pipelining is a technique where the CPU works on multiple instructions at once, like an assembly line. This increases throughput (instructions per second) without necessarily speeding up a single instruction.

**1. The 5-Stage Classic Pipeline:**
*   **IF (Instruction Fetch):** Read the next instruction from the instruction cache using the PC.
*   **ID (Instruction Decode / Register Read):** Decode the opcode, read source registers.
*   **EX (Execute / ALU):** Perform arithmetic/logic or compute memory address.
*   **MEM (Memory Access):** Read from or write to data cache (loads/stores only).
*   **WB (Write Back):** Write the result into the destination register.

Without pipelining, each instruction takes 5 cycles. With pipelining, one instruction completes every cycle (after the pipeline fills), giving ~5x throughput improvement.

**2. Pipeline Performance:**
*   **Throughput:** Instructions per cycle (IPC). Ideal = 1.0 for a simple pipeline.
*   **Latency per instruction:** Still 5 cycles. Pipelining improves throughput, not latency.
*   **Speedup = Pipeline Depth** (ideal). A 5-stage pipeline is ideally 5x faster than non-pipelined.
*   **Real IPC** is less than 1.0 due to hazards and stalls.

**3. Pipeline Hazards (Roadblocks):**

**Structural Hazards:**
*   Two instructions need the same hardware at the same time.
*   Example: Single memory port used by both IF and MEM stages.
*   Solution: Separate instruction and data caches (Harvard-style L1).

**Data Hazards:**
*   Read After Write (RAW): True dependency. Instruction needs result of previous instruction.
*   Write After Read (WAR): Anti-dependency. Can reorder to fix (register renaming).
*   Write After Write (WAW): Output dependency. Also fixed by register renaming.
*   RAW is the most common and hardest to handle.

**Solutions for Data Hazards:**
*   **Stalling (Bubbling):** Insert NOPs until data is ready. Simple but wastes cycles.
*   **Forwarding (Bypassing):** Wire the ALU output directly back to the ALU input, skipping the register file. Eliminates most RAW stalls.
*   **Load-Use Hazard:** Even with forwarding, a load followed immediately by a use requires a 1-cycle stall (data comes from MEM, not EX).

**Control Hazards:**
*   Branches change the PC, but the pipeline has already fetched the next sequential instruction.
*   **Branch Penalty:** Number of wasted cycles when a branch is taken.
*   Solutions: Branch prediction, delayed branching, speculative execution.

**4. Deeper Pipelines:**
*   More stages = higher clock frequency (less work per stage).
*   Intel Pentium 4 (Prescott) had 31 pipeline stages at 3.8 GHz.
*   Downside: Higher branch misprediction penalty, more forwarding paths needed.
*   Modern CPUs typically use 14-19 stages as a balancing point.`,
					CodeExamples: `// Data Hazard & Forwarding Example:
// Without forwarding (3-cycle stall):
add r1, r2, r3   // r1 written in WB (cycle 5)
sub r4, r1, r5   // r1 needed in ID (cycle 3) -- STALL 2 cycles!

// With forwarding (0 stalls):
add r1, r2, r3   // r1 computed in EX (cycle 3)
sub r4, r1, r5   // r1 forwarded from EX output to EX input

// Load-Use Hazard (1 unavoidable stall even with forwarding):
lw  r1, 0(r2)    // r1 available after MEM (cycle 4)
add r3, r1, r4   // r1 needed in EX (cycle 3) -- must stall 1 cycle!

// Instruction scheduling (compiler reorders to hide stall):
// Before:
lw  r1, 0(r2)    // load
add r3, r1, r4   // uses r1 immediately -- STALL

// After (compiler inserts independent instruction):
lw  r1, 0(r2)    // load
add r7, r8, r9   // independent work (fills the stall slot)
add r3, r1, r4   // r1 is now ready -- no stall!

// Pipeline throughput calculation:
// Non-pipelined: 5 cycles/instruction, 100 instructions = 500 cycles
// 5-stage pipeline: 5 + (100-1) = 104 cycles
// Speedup: 500/104 = 4.8x (approaches 5x for large N)`,
				},
				{
					Title: "Branch Prediction Basics",
					Content: `Branch instructions appear every 5-8 instructions in typical code. Without prediction, the pipeline stalls until the branch outcome is known. Branch prediction guesses the outcome and keeps the pipeline full.

**1. Static Prediction (Compile-Time Guesses):**
*   **Always Not Taken:** Predict all branches as not taken. Continue fetching sequentially.
*   **Always Taken:** Predict all branches as taken. ~60% accuracy (loops are usually taken).
*   **BTFNT (Backward Taken, Forward Not Taken):** Backward branches (loops) predicted taken, forward branches (if-else) predicted not taken. ~65% accuracy.
*   **Compiler Hints:** GCC __builtin_expect() lets the programmer hint likely/unlikely paths.

**2. Dynamic Prediction (Runtime Learning):**

**1-Bit Predictor:**
*   Remembers last outcome for each branch (taken or not taken).
*   Problem: Alternating patterns cause 100% misprediction.
*   Example: A loop that runs 10 times mispredicts on entry and exit = 2 mispredictions per loop invocation.

**2-Bit Saturating Counter:**
*   Four states: Strongly Not Taken (00), Weakly Not Taken (01), Weakly Taken (10), Strongly Taken (11).
*   Must mispredict twice to change direction.
*   Now a loop that runs 10 times only mispredicts on exit = 1 misprediction per invocation.
*   Accuracy: ~85-93% on typical workloads.

**Branch History Table (BHT):**
*   Array of 2-bit counters indexed by low bits of branch PC.
*   Simple, fast lookup.
*   Problem: Different branches may alias to the same entry.

**3. Correlated Predictors:**

**Two-Level Adaptive (GShare):**
*   Uses a Global History Register (GHR) that records the last N branch outcomes as a bit vector.
*   XOR the GHR with the branch PC to index the predictor table.
*   Captures correlations between branches: if (x > 0) ... if (x > 5) -- the second branch is correlated with the first.
*   Accuracy: ~92-97%.

**Tournament Predictor (Alpha 21264, Intel Core):**
*   Runs multiple predictors in parallel (e.g., local + global).
*   A meta-predictor (chooser) selects which one to trust for each branch.
*   Intel uses TAGE (TAgged GEometric) predictors: multiple tables with different history lengths.
*   Accuracy: ~95-98%.

**4. Branch Target Prediction:**
*   **BTB (Branch Target Buffer):** Caches the target address of recently taken branches. Without it, even a correct direction prediction is useless because we don't know WHERE to fetch.
*   **RAS (Return Address Stack):** Specialized predictor for function returns. Push on CALL, pop on RET. ~99% accuracy.
*   **Indirect Branch Predictor:** For virtual function calls and switch statements where the target varies.

**5. Misprediction Cost:**
*   Pipeline flush: all speculative work discarded.
*   Cost = pipeline depth in cycles (14-19 cycles on modern CPUs).
*   A branch that mispredicts 5% of the time on a 15-stage pipeline:
    *   0.05 * 15 = 0.75 extra cycles per branch on average.
    *   Branches every ~6 instructions: 0.75/6 = ~12% throughput loss.`,
					CodeExamples: `// 2-bit saturating counter state diagram:
// State transitions (predict taken when state >= 10):
//
//   Strongly NT (00) --[taken]--> Weakly NT (01)
//   Weakly NT  (01) --[taken]--> Weakly T  (10) -- prediction flips!
//   Weakly T   (10) --[taken]--> Strongly T (11)
//   Strongly T (11) --[not taken]--> Weakly T (10)
//   Weakly T   (10) --[not taken]--> Weakly NT (01) -- prediction flips!
//   Weakly NT  (01) --[not taken]--> Strongly NT (00)

// GShare predictor:
// GHR (last 12 branches): 1010 0110 0111
// Branch PC (low bits):   0011 1000 1100
// Index = GHR XOR PC:     1001 1110 1011  --> look up 2-bit counter

// Compiler hint for branch prediction (GCC):
if (__builtin_expect(error_code != 0, 0)) {
    // unlikely path: error handling
    handle_error(error_code);
}
// likely path: continues here without branch penalty

// Profile-Guided Optimization (PGO):
// 1. Compile with instrumentation: gcc -fprofile-generate
// 2. Run with representative workload
// 3. Recompile with profile data: gcc -fprofile-use
// Result: compiler knows exact branch frequencies`,
				},
				{
					Title: "Cache Memory Architecture",
					Content: `The cache is the most performance-critical component in modern CPUs. A single L1 cache miss costs ~12x more than a hit; a DRAM access costs ~200x more. Writing cache-aware code can yield 10-100x speedups.

**1. Cache Organization:**
A cache line (block) is the unit of transfer: typically 64 bytes. When you access one byte, the entire 64-byte line is loaded.

*   **Tag:** High bits of the address. Identifies which memory block is cached.
*   **Index:** Middle bits. Selects the cache set.
*   **Offset:** Low bits. Selects the byte within the cache line.

Address decomposition: [Tag | Index | Offset]

**2. Mapping Strategies:**

**Direct Mapped:**
*   Each memory block maps to exactly one cache line (index = block mod N).
*   Fast lookup (one comparison).
*   Problem: Two frequently used blocks with the same index continuously evict each other (thrashing).

**Set-Associative:**
*   Each memory block maps to a set of N lines (N-way associative).
*   Must check N tags in parallel.
*   2-way, 4-way, 8-way, 16-way are common.
*   Sweet spot between speed and conflict misses.
*   Most L1 caches: 4-8 way. L2/L3: 8-16 way.

**Fully Associative:**
*   A block can go anywhere in the cache.
*   No conflict misses, but must compare all tags (impractical for large caches).
*   Used for TLBs and small special-purpose caches.

**3. The 3 C's of Cache Misses:**
*   **Compulsory (Cold):** First access to a block. Unavoidable. Reduced by prefetching.
*   **Capacity:** Working set exceeds cache size. Solution: larger cache or algorithmic optimization (tiling/blocking).
*   **Conflict:** Multiple blocks map to the same set. Solution: increase associativity.
*   **Coherency (4th C):** In multi-core systems, invalidations from other cores cause misses.

**4. Cache Replacement Policies:**
*   **LRU (Least Recently Used):** Evict the line used longest ago. Good but expensive to track for high associativity.
*   **Pseudo-LRU:** Approximate LRU with a binary tree. Much cheaper hardware.
*   **Random:** Surprisingly competitive with LRU for large caches.
*   **RRIP (Re-Reference Interval Prediction):** Intel uses this. Predicts how soon a line will be reused.

**5. Write Policies:**

**Write Hit:**
*   **Write-Through:** Write to both cache and memory. Simple, consistent, but slow (every write goes to memory).
*   **Write-Back:** Write only to cache. Mark line as "dirty." Write to memory only when evicted. Much faster, but more complex.

**Write Miss:**
*   **Write-Allocate:** Load the block into cache, then write. Pairs well with write-back.
*   **No-Write-Allocate:** Write directly to memory without loading into cache. Pairs with write-through.

**6. Prefetching:**
The CPU or compiler predicts future accesses and loads data before it is needed.
*   **Hardware Prefetcher:** Detects sequential/strided access patterns automatically.
*   **Software Prefetch:** Explicit instructions (e.g., _mm_prefetch() on x86). Useful for irregular patterns.
*   **Prefetch too early:** Data evicted before use. Too late: data not ready. Finding the sweet spot is key.`,
					CodeExamples: `// Cache line size demonstration:
// Accessing arr[0] loads bytes arr[0..63] into cache (64-byte line)
// arr[1] through arr[15] are already cached (for int32 array)!

// Row-major vs Column-major:
// Given: int matrix[1024][1024];
// Row stride = 4 bytes, Column stride = 4096 bytes

// Cache-friendly (row-major traversal):
for (int i = 0; i < 1024; i++)      // ~2 ms
    for (int j = 0; j < 1024; j++)
        sum += matrix[i][j];          // stride = 4 bytes (sequential)

// Cache-unfriendly (column-major traversal):
for (int j = 0; j < 1024; j++)      // ~20 ms (10x slower!)
    for (int i = 0; i < 1024; i++)
        sum += matrix[i][j];          // stride = 4096 bytes (cache miss every access)

// Loop tiling (blocking) for cache optimization:
// Process BxB sub-blocks that fit in cache
#define B 32
for (int ii = 0; ii < N; ii += B)
    for (int jj = 0; jj < N; jj += B)
        for (int i = ii; i < ii+B; i++)
            for (int j = jj; j < jj+B; j++)
                C[i][j] += A[i][j] * B_mat[i][j];

// Software prefetch (x86 intrinsic):
for (int i = 0; i < N; i++) {
    _mm_prefetch(&arr[i + 16], _MM_HINT_T0);  // prefetch 16 ahead
    sum += arr[i];
}

// Cache-line alignment:
struct alignas(64) CacheLine {
    int data[16];  // Exactly fills one 64-byte cache line
};`,
				},
				{
					Title: "Cache Coherency in Multi-Core Systems",
					Content: `When multiple CPU cores each have their own L1/L2 caches, a memory location can exist in multiple caches simultaneously. If one core writes, the other copies become stale. Cache coherency protocols keep all caches consistent.

**1. The Problem:**
*   Core 0 reads variable X (X = 5 in both Core 0's cache and memory).
*   Core 1 writes X = 10 to its own cache.
*   Core 0 still sees X = 5 -- stale data!
*   Without coherency, multi-threaded programs would produce random incorrect results.

**2. MESI Protocol (Most Common):**
Each cache line has a state:
*   **M (Modified):** This core has written the line. Only copy is here. Memory is STALE. Must write back before another core reads.
*   **E (Exclusive):** This core is the only one with this line, and it matches memory. Can transition to M on write without bus traffic.
*   **S (Shared):** Multiple cores have this line. All copies match memory. Read-only. Must invalidate others before writing.
*   **I (Invalid):** Line is not valid. Must fetch from memory or another cache.

**State Transitions:**
*   Read miss (no other cache has it): I -> E
*   Read miss (another cache has it): I -> S (and the other goes E -> S or M -> S with writeback)
*   Write hit in E: E -> M (silent, no bus traffic)
*   Write hit in S: S -> M (must invalidate all other copies first)
*   Another core reads our M line: M -> S (writeback to memory)

**3. False Sharing (Silent Performance Killer):**
Two cores write to different variables that happen to be on the same cache line. The coherency protocol ping-pongs the line between caches, causing massive slowdowns.

Example: Two threads incrementing adjacent counters:
*   counter[0] and counter[1] are in the same 64-byte cache line.
*   Every increment by Thread 0 invalidates Thread 1's copy, and vice versa.
*   Solution: Pad each counter to its own cache line (64-byte alignment).

**4. MOESI and MESIF (Extensions):**
*   **MOESI (AMD):** Adds O (Owned) state. A modified line can be shared without writing back to memory -- one core keeps "ownership."
*   **MESIF (Intel):** Adds F (Forward) state. One designated core responds to requests for shared lines, reducing bus traffic.

**5. Directory-Based Coherency (For Many-Core Systems):**
*   Snooping (MESI) broadcasts on a shared bus -- doesn't scale beyond ~8 cores.
*   Directory protocols maintain a centralized (or distributed) directory tracking which cores have each line.
*   Scales to hundreds of cores but adds latency for lookups.
*   Used in server-class CPUs (Intel Xeon) and GPUs.`,
					CodeExamples: `// False sharing example:
struct BadCounters {
    int counter0;  // offset 0
    int counter1;  // offset 4 -- SAME cache line as counter0!
};

// Fix: pad to separate cache lines
struct GoodCounters {
    alignas(64) int counter0;  // own cache line
    alignas(64) int counter1;  // own cache line
};

// MESI state transitions example:
// Initial: X in Memory = 5, all caches Invalid
//
// Core 0 reads X:  Core0 = E(5), Memory = 5
// Core 1 reads X:  Core0 = S(5), Core1 = S(5), Memory = 5
// Core 0 writes X=10:
//   1. Core 0 sends invalidation to Core 1
//   2. Core 1: S -> I
//   3. Core 0: S -> M(10), Memory still = 5 (stale)
// Core 1 reads X:
//   1. Core 0 writes back: Memory = 10, Core 0: M -> S
//   2. Core 1: I -> S(10)

// Detecting false sharing with perf (Linux):
// perf c2c record -- ./my_program
// perf c2c report
// Shows contested cache lines and contributing code locations

// Go example of false sharing fix:
type PaddedCounter struct {
    value int64
    _pad  [56]byte  // pad to 64 bytes total
}
var counters [NumThreads]PaddedCounter`,
				},
				{
					Title: "Virtual Memory and Address Translation",
					Content: `Virtual memory gives each process the illusion of a large, private, contiguous address space. The hardware (MMU) transparently translates virtual addresses to physical addresses.

**1. Why Virtual Memory?**
*   **Isolation:** Each process has its own address space. Process A cannot access Process B's memory.
*   **Abstraction:** Programs don't need to know physical memory layout.
*   **Overcommit:** Total virtual memory can exceed physical RAM (pages swapped to disk).
*   **Shared Libraries:** Multiple processes can share a single physical copy of libc.
*   **Memory-Mapped Files:** Files can be accessed as if they were in memory.

**2. Page-Based Virtual Memory:**
*   Virtual and physical memory are divided into fixed-size pages (typically 4 KB).
*   The page table maps virtual page numbers (VPN) to physical page frame numbers (PFN).
*   Page Table Entry (PTE) contains: PFN, Valid bit, Dirty bit, Access permissions (R/W/X), Referenced bit.

**3. Multi-Level Page Tables:**
*   A flat page table for a 48-bit address space would need 512 GB of entries!
*   Solution: Hierarchical page tables (4 levels on x86-64):
    *   PML4 -> PDPT -> PD -> PT -> Physical Page.
    *   Only allocate table pages that are actually used.
    *   A process using just 1 GB of memory needs only a few KB of page table entries.

**4. TLB (Translation Lookaside Buffer):**
*   A small, fast cache inside the CPU that stores recent VPN -> PFN translations.
*   L1 TLB: ~64 entries, ~1 cycle access. L2 TLB: ~1024 entries, ~7-10 cycles.
*   TLB hit: translation in 1 cycle. TLB miss: full page table walk (~100-1000 cycles).
*   TLB reach = TLB entries * page size. With 1024 entries and 4 KB pages = 4 MB coverage.

**5. Huge Pages (Large Pages):**
*   Standard: 4 KB pages. Huge: 2 MB or 1 GB pages.
*   Advantage: Fewer TLB entries needed -> better TLB reach -> fewer TLB misses.
*   Linux: Transparent Huge Pages (THP) or explicit hugetlbfs.
*   Used by databases, VMs, and large-memory applications.

**6. Page Faults:**
*   **Minor Fault:** Page table entry exists but not in TLB. Just load into TLB.
*   **Major Fault:** Page is not in physical memory (swapped to disk). OS must load from disk. ~5-10 ms (millions of cycles). This is why swapping kills performance.
*   **Invalid Fault:** Access to unmapped memory -> segmentation fault (SIGSEGV).

**7. Memory Protection:**
*   Each PTE has permission bits: Read, Write, Execute.
*   Write-protecting code pages prevents code injection attacks.
*   NX (No-Execute) bit prevents executing data as code (DEP/W^X).
*   ASLR (Address Space Layout Randomization): Randomizes the base addresses of stack, heap, and libraries to make exploits harder.`,
					CodeExamples: `// x86-64 virtual address breakdown (48-bit):
// Bits [47:39] - PML4 index  (9 bits = 512 entries)
// Bits [38:30] - PDPT index  (9 bits = 512 entries)
// Bits [29:21] - PD index    (9 bits = 512 entries)
// Bits [20:12] - PT index    (9 bits = 512 entries)
// Bits [11:0]  - Page offset (12 bits = 4096 bytes)

// TLB access pattern:
// CPU accesses virtual address 0x00007FFE1234ABCD
// 1. Extract VPN = 0x00007FFE1234A
// 2. Check TLB for VPN
// 3. HIT: Get PFN, form physical address in 1 cycle
// 4. MISS: Walk 4-level page table (4 memory accesses)

// Page fault types:
// Program accesses virtual address 0xDEADBEEF:
// Case 1 - Minor fault: Page mapped but not in TLB
//   Cost: ~100 cycles (page table walk)
// Case 2 - Major fault: Page swapped to disk
//   Cost: ~10,000,000 cycles (disk I/O)
// Case 3 - Invalid: Address not mapped
//   Result: SIGSEGV (crash)

// Huge pages in Linux:
// echo 1024 > /proc/sys/vm/nr_hugepages  // Reserve 1024 x 2MB pages
// mmap(NULL, size, PROT_READ|PROT_WRITE,
//      MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);

// ASLR demonstration (addresses change each run):
// Run 1: stack=0x7FFE1234, heap=0x556789AB, libc=0x7F12ABCD
// Run 2: stack=0x7FFC5678, heap=0x55A1B2C3, libc=0x7F34CDEF`,
				},
				{
					Title: "I/O Systems, Interrupts, and DMA",
					Content: `I/O systems allow the CPU to communicate with external devices. The three main I/O methods represent a trade-off between simplicity and performance.

**1. Programmed I/O (Polling):**
*   The CPU actively checks device status in a loop.
*   Simple to implement.
*   Wastes CPU cycles while waiting.
*   Acceptable for very fast devices or real-time systems where latency matters more than throughput.

**2. Interrupt-Driven I/O:**
*   The device signals the CPU via an interrupt when data is ready.
*   The CPU handles the event, then returns to normal work.
*   Much more efficient than polling for slow devices.
*   Overhead per interrupt: ~1-10 microseconds (save context, jump to handler, restore context).
*   Problem: Very fast devices (10 Gbps NIC) can generate millions of interrupts per second, overwhelming the CPU.

**3. Direct Memory Access (DMA):**
*   A DMA controller transfers data directly between device and memory, bypassing the CPU.
*   The CPU sets up the transfer (source, destination, length) and continues other work.
*   The DMA controller interrupts the CPU only when the entire transfer completes.
*   Used for: disk I/O, network packets, GPU memory transfers.

**4. Interrupt Architecture:**

**Hardware Interrupts:**
*   Generated by external devices (keyboard, timer, NIC, disk controller).
*   Asynchronous -- can occur at any point during execution.
*   Managed by an interrupt controller (APIC on x86).

**Software Interrupts (Traps / Exceptions):**
*   Generated by the CPU itself.
*   Synchronous -- triggered by a specific instruction.
*   Examples: system calls (INT 0x80 / SYSCALL), divide by zero, page faults, breakpoints.

**Interrupt Priority:**
*   Multiple interrupts can occur simultaneously.
*   Priority levels determine which is handled first.
*   Higher priority interrupts can preempt lower priority handlers (nested interrupts).
*   Critical interrupts (NMI -- non-maskable interrupt) cannot be disabled.

**5. Interrupt Handling Steps:**
1.  Device asserts interrupt line.
2.  CPU finishes current instruction.
3.  CPU saves context (registers, flags, PC) to the stack.
4.  CPU reads interrupt vector number from the interrupt controller.
5.  CPU jumps to the Interrupt Service Routine (ISR) via the Interrupt Descriptor Table (IDT).
6.  ISR handles the event (reads device data, updates state).
7.  ISR sends End of Interrupt (EOI) to the interrupt controller.
8.  CPU restores context from the stack and resumes the interrupted program (IRET instruction).

**6. Modern I/O Optimization:**

**Interrupt Coalescing:**
*   Batch multiple events into a single interrupt.
*   Reduces interrupt overhead for high-speed devices (NICs).

**NAPI (Linux New API for Networking):**
*   Hybrid approach: interrupt triggers polling mode.
*   First packet: interrupt. Subsequent packets: poll until queue is empty.
*   Prevents interrupt storms at high packet rates.

**Memory-Mapped I/O (MMIO) vs Port I/O:**
*   MMIO: Device registers mapped into the memory address space. Access via normal load/store. Modern standard.
*   Port I/O: Separate I/O address space accessed via IN/OUT instructions. Legacy x86.

**7. I/O Performance Metrics:**
*   **IOPS (I/O Operations Per Second):** Measures random access performance. NVMe SSDs: 500K-1M IOPS.
*   **Throughput (MB/s or GB/s):** Sequential read/write bandwidth. NVMe: 3-7 GB/s.
*   **Latency:** Time for a single I/O operation. NVMe: ~10-25 us. HDD: ~5-10 ms.`,
					CodeExamples: `// Polling I/O:
while (!(device->status & DATA_READY)) {
    // Busy wait -- wasting CPU cycles
}
data = device->data_register;

// Interrupt-driven I/O:
void keyboard_isr() {
    char key = inb(KEYBOARD_DATA_PORT);
    buffer_push(key);
    send_eoi();
}
// Register ISR in IDT
idt[KEYBOARD_IRQ] = &keyboard_isr;

// DMA setup for disk read:
dma_controller->source = DISK_BUFFER;
dma_controller->destination = RAM_ADDRESS;
dma_controller->count = 4096;  // 4 KB
dma_controller->command = DMA_READ | DMA_START;
// CPU is now free to do other work
// DMA completion triggers an interrupt

// Linux interrupt handler (kernel module):
static irqreturn_t my_handler(int irq, void *dev_id) {
    uint32_t status = readl(dev->base + STATUS_REG);
    if (!(status & MY_DEVICE_IRQ))
        return IRQ_NONE;  // Not our interrupt
    writel(status, dev->base + STATUS_REG);  // ACK
    tasklet_schedule(&my_tasklet);  // Defer heavy work
    return IRQ_HANDLED;
}

// MMIO vs Port I/O:
// MMIO (modern):
volatile uint32_t *reg = (uint32_t *)0xFEDC0000;
*reg = 0x1;            // Write to device register
uint32_t val = *reg;   // Read from device register

// Port I/O (legacy x86):
outb(0x1, 0x3F8);      // Write 0x1 to port 0x3F8 (COM1)
uint8_t val = inb(0x3F8); // Read from port 0x3F8`,
				},
				{
					Title: "Performance Metrics and Benchmarking",
					Content: `Understanding performance metrics is essential for evaluating and comparing computer architectures. These metrics formalize our intuition about "fast" and "slow."

**1. CPU Performance Equation:**
*   **Execution Time = IC * CPI * Clock Period**
    *   IC = Instruction Count (how many instructions the program executes).
    *   CPI = Cycles Per Instruction (average cycles per instruction).
    *   Clock Period = 1 / Clock Frequency.
*   To improve performance, reduce any of the three factors.

**2. IPC (Instructions Per Cycle) = 1 / CPI:**
*   Modern CPUs: IPC = 3-6 (superscalar, out-of-order).
*   Higher IPC means more work per cycle.
*   Apple M-series: IPC ~6-8 (wide decode, large ROB).

**3. MIPS and FLOPS:**
*   **MIPS:** Millions of Instructions Per Second. Simple but misleading (different ISAs have different instruction granularity).
*   **FLOPS:** Floating-Point Operations Per Second. Standard for scientific computing.
*   Modern GPU: ~80 TFLOPS (FP16). Modern CPU: ~1 TFLOPS (FP64 with AVX-512).

**4. Amdahl's Law:**
*   Speedup from improving a fraction f of the workload by factor S:
    *   Speedup = 1 / ((1 - f) + f/S)
*   If you speed up 90% of the program by 10x, total speedup = 1 / (0.1 + 0.09) = 5.26x (NOT 10x).
*   The serial portion limits the overall speedup. This is why parallelism has diminishing returns.

**5. Gustafson's Law (Alternative View):**
*   As problem size grows, the parallel portion grows with it.
*   Scaled Speedup = s + p * N (where s = serial fraction, p = parallel fraction, N = processors).
*   More optimistic than Amdahl for real workloads that scale with compute.

**6. Benchmarking:**
*   **SPEC CPU:** Industry standard for CPU performance (integer and floating-point).
*   **Geekbench:** Cross-platform, single-core and multi-core.
*   **LINPACK:** Dense linear algebra (used for TOP500 supercomputer ranking).
*   **MLPerf:** Machine learning inference and training benchmark.
*   **Microbenchmarks:** Measure specific operations (memory latency, branch misprediction rate).

**7. Power and Energy:**
*   **Dynamic Power = C * V^2 * f** (capacitance * voltage squared * frequency).
*   Reducing voltage is the most effective way to reduce power (quadratic effect).
*   **Power Wall:** Cannot increase frequency further without exceeding thermal limits.
*   **Performance per Watt:** The key metric for mobile and data center chips.
*   Apple M-series excels here: ARM cores at moderate frequency with high IPC = great perf/watt.`,
					CodeExamples: `// CPU Performance Equation example:
// Program: 1 billion instructions
// CPI: 1.5 cycles per instruction
// Clock: 3 GHz (3 * 10^9 cycles/second)
//
// Execution Time = (10^9 * 1.5) / (3 * 10^9) = 0.5 seconds
// If we improve CPI to 1.0: Time = 10^9 / (3 * 10^9) = 0.33 seconds
// Speedup = 0.5 / 0.33 = 1.5x

// Amdahl's Law example:
// 80% of code is parallelizable, using 4 cores:
// Speedup = 1 / (0.2 + 0.8/4) = 1 / (0.2 + 0.2) = 2.5x
// Using 8 cores:
// Speedup = 1 / (0.2 + 0.8/8) = 1 / (0.2 + 0.1) = 3.33x
// Using infinite cores:
// Speedup = 1 / 0.2 = 5x maximum! (limited by serial 20%)

// Power calculation:
// P = C * V^2 * f
// Original: V=1.0V, f=3GHz  -> P = C * 1.0 * 3 = 3C
// Reduced:  V=0.8V, f=2.4GHz -> P = C * 0.64 * 2.4 = 1.536C
// Power reduction: 49%! Performance reduction: ~20%
// Performance per watt improved significantly

// Linux perf counters (measuring real hardware):
// perf stat -e instructions,cycles,cache-misses ./my_program
// Output:
//   1,234,567,890  instructions  # 2.5 IPC
//     493,827,156  cycles
//       1,234,567  cache-misses  # 0.1% miss rate`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
