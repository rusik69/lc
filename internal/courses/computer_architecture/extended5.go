package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2826,
			Title:       "Memory Hierarchy Deep Dive",
			Description: "Explore advanced memory system topics: SRAM vs DRAM internals, DDR protocols, cache coherence, memory controllers, and NUMA architectures.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "SRAM and DRAM Internals",
					Content: `Understanding how memory physically stores bits explains why we need a memory hierarchy and why different levels have such different speeds and costs.

**SRAM (Static RAM):**
` + "```" + `
6-Transistor (6T) SRAM Cell:
         VDD            VDD
          │              │
       ┌──┴──┐        ┌──┴──┐
       │ P1  │        │ P2  │
       └──┬──┘        └──┬──┘
          ├──────┐ ┌─────┤
       ┌──┴──┐   │ │  ┌──┴──┐
       │ N1  │   │ │  │ N2  │      Cross-coupled inverters
       └──┬──┘   │ │  └──┬──┘      store one bit
          │      │ │     │
         GND    Q  Q'   GND
          │      │ │     │
       ┌──┴──┐   │ │  ┌──┴──┐
       │ N3  │───┘ └──│ N4  │      Access transistors
       └──┬──┘        └──┬──┘
          │              │
       BitLine         BitLine'
          │              │
    WordLine ─────────────────

Properties:
  - 6 transistors per bit
  - NO refresh needed (static — holds data as long as power on)
  - Access time: 0.5-5 ns
  - Used for: CPU caches (L1, L2, L3)
  - Cost: ~$500-5000/GB
  - Typical sizes: 32KB (L1) to 64MB (L3)
` + "```" + `

**DRAM (Dynamic RAM):**
` + "```" + `
1-Transistor 1-Capacitor (1T1C) DRAM Cell:
       WordLine
          │
       ┌──┴──┐
       │  N  │── Access transistor
       └──┬──┘
          │
       ┌──┴──┐
       │  C  │── Storage capacitor (~30 fF)
       └──┬──┘
          │
       BitLine

Properties:
  - 1 transistor + 1 capacitor per bit (much denser than SRAM)
  - Capacitor leaks charge → must REFRESH every 64 ms
  - Reading is destructive: charge shared with bitline → must write back
  - Access time: 50-100 ns
  - Used for: Main memory
  - Cost: ~$3-10/GB
  - Typical sizes: 8-128 GB

DRAM Read Operation:
  1. Precharge bitlines to VDD/2
  2. Assert WordLine → charge sharing between capacitor and bitline
  3. Sense amplifier detects tiny voltage difference (±200 mV)
  4. Sense amp amplifies to full VDD or 0
  5. Write back amplified value to capacitor (destructive read)
  6. Read data from sense amp output

Timing: tRAS (Row Access Strobe) + tCAS (Column Access Strobe)
  - tRAS ≈ 30-35 ns (open a row)
  - tCAS ≈ 12-15 ns (read a column from open row)
  - Burst: Once row is open, consecutive columns are fast
` + "```" + `

**DRAM Organization:**
` + "```" + `
DRAM Chip Structure:
┌─────────────────────────────────────┐
│           Row Decoder               │
│  ┌─────────────────────────────┐    │
│  │                             │    │
│  │     Memory Cell Array       │    │
│  │     (bank of rows×cols)     │    │
│  │                             │    │
│  └─────────────────────────────┘    │
│           Sense Amplifiers          │
│           Column Decoder            │
│           I/O Gating                │
└─────────────────────────────────────┘

Modern DRAM: Multiple banks for parallelism
  DDR4: 16-32 banks (4 bank groups × 4-8 banks)
  DDR5: 32 banks (8 bank groups × 4 banks), 2 channels per DIMM

Address mapping:
  Physical address → {Channel, DIMM, Rank, Bank Group, Bank, Row, Column}

Row Buffer (Page): Once a row is "opened" (activated),
  the entire row (~8KB) is in the sense amplifiers.
  Subsequent accesses to the same row are fast (row hit).
  Access to different row = row miss (must precharge + activate).
  
Row Buffer Hit:  ~15 ns (CAS latency only)
Row Buffer Miss: ~50 ns (RAS + CAS)
Row Buffer Conflict: ~60 ns (precharge + RAS + CAS)
` + "```" + `

**DDR Evolution:**
` + "```" + `
Generation │ Data Rate │ Bandwidth │ Voltage │ Prefetch │ Year
DDR        │ 200-400   │ 1.6-3.2   │ 2.5V    │ 2n       │ 2000
DDR2       │ 400-1066  │ 3.2-8.5   │ 1.8V    │ 4n       │ 2003
DDR3       │ 800-2133  │ 6.4-17    │ 1.5V    │ 8n       │ 2007
DDR4       │ 1600-3200 │ 12.8-25.6 │ 1.2V    │ 8n       │ 2014
DDR5       │ 3200-8400 │ 25.6-67.2 │ 1.1V    │ 16n      │ 2020
           │ MT/s      │ GB/s      │         │ burst    │

"DDR" = Double Data Rate: transfers on both clock edges
Prefetch: reads 2n/4n/8n/16n bits per internal clock
  → External interface runs at 2×/4×/8×/16× internal clock
  → Bandwidth increases without speeding up core memory array

DDR5 improvements over DDR4:
  - Two 32-bit channels (vs one 64-bit) → better utilization
  - On-die ECC → better reliability
  - Higher bank count → more parallelism
  - Decision feedback equalization → cleaner signal at high speeds
` + "```" + ``,
					CodeExamples: `// Memory system simulation in Go
package main

import "fmt"

// DRAM timing model
type DRAMTiming struct {
    tRCD    int // RAS-to-CAS delay (row activation)
    tCAS    int // CAS latency (column access)
    tRP     int // Row Precharge time
    tRAS    int // Row Active time
    tBurst  int // Burst transfer time
}

var DDR4_3200 = DRAMTiming{
    tRCD:   22, // ~13.75 ns at 1600 MHz
    tCAS:   22, // CL22
    tRP:    22, // ~13.75 ns
    tRAS:   52, // ~32 ns
    tBurst: 4,  // 8 beats at DDR = 4 cycles
}

// DRAM Bank state
type DRAMBank struct {
    timing    DRAMTiming
    openRow   int  // Currently open row (-1 if none)
    busy      int  // Cycles until available
}

func NewDRAMBank(timing DRAMTiming) *DRAMBank {
    return &DRAMBank{timing: timing, openRow: -1}
}

type AccessResult struct {
    latency int
    hitType string
}

func (b *DRAMBank) Access(row, col int) AccessResult {
    if b.openRow == row {
        // Row buffer hit — just CAS
        latency := b.timing.tCAS + b.timing.tBurst
        return AccessResult{latency, "Row Hit"}
    }

    if b.openRow == -1 {
        // Empty — activate + CAS
        b.openRow = row
        latency := b.timing.tRCD + b.timing.tCAS + b.timing.tBurst
        return AccessResult{latency, "Row Miss (empty)"}
    }

    // Row conflict — precharge + activate + CAS
    latency := b.timing.tRP + b.timing.tRCD + b.timing.tCAS + b.timing.tBurst
    b.openRow = row
    return AccessResult{latency, "Row Conflict"}
}

// Memory controller with address mapping
type MemoryController struct {
    banks     []*DRAMBank
    numBanks  int
    rowBits   int
    colBits   int
    bankBits  int
    // Statistics
    hits      int
    misses    int
    conflicts int
}

func NewMemoryController(numBanks int, timing DRAMTiming) *MemoryController {
    mc := &MemoryController{
        numBanks: numBanks,
        rowBits:  14,
        colBits:  10,
        bankBits: 4,
    }
    mc.banks = make([]*DRAMBank, numBanks)
    for i := range mc.banks {
        mc.banks[i] = NewDRAMBank(timing)
    }
    return mc
}

func (mc *MemoryController) Access(addr uint64) AccessResult {
    // Address mapping: ...| Row | Bank | Column | Byte offset |
    col := int((addr >> 6) & ((1 << mc.colBits) - 1))  // 64B cache line
    bank := int((addr >> (6 + mc.colBits)) & ((1 << mc.bankBits) - 1))
    row := int((addr >> (6 + mc.colBits + mc.bankBits)) & ((1 << mc.rowBits) - 1))

    bank = bank % mc.numBanks
    result := mc.banks[bank].Access(row, col)

    switch result.hitType {
    case "Row Hit": mc.hits++
    case "Row Miss (empty)": mc.misses++
    case "Row Conflict": mc.conflicts++
    }
    return result
}

func main() {
    mc := NewMemoryController(16, DDR4_3200)

    fmt.Println("DRAM Access Patterns:")

    // Sequential access (cache-line granularity, same row = hits)
    fmt.Println("\n1. Sequential access (8KB within same row):")
    totalLatency := 0
    for i := 0; i < 128; i++ { // 128 × 64B = 8KB
        addr := uint64(0x10000 + i*64)
        result := mc.Access(addr)
        totalLatency += result.latency
        if i < 4 || i == 127 {
            fmt.Printf("   Addr 0x%X: %s (%d cycles)\n",
                addr, result.hitType, result.latency)
        }
    }
    fmt.Printf("   Total: %d cycles, Avg: %.1f cycles/access\n",
        totalLatency, float64(totalLatency)/128)

    // Reset
    mc2 := NewMemoryController(16, DDR4_3200)

    // Strided access (different rows = conflicts)
    fmt.Println("\n2. Large-stride access (crossing rows):")
    totalLatency = 0
    stride := 16 * 1024 // 16KB stride (crosses rows)
    for i := 0; i < 16; i++ {
        addr := uint64(0x10000 + i*stride)
        result := mc2.Access(addr)
        totalLatency += result.latency
        fmt.Printf("   Addr 0x%X: %s (%d cycles)\n",
            addr, result.hitType, result.latency)
    }
    fmt.Printf("   Total: %d cycles, Avg: %.1f cycles/access\n",
        totalLatency, float64(totalLatency)/16)

    // Bandwidth calculation
    fmt.Println("\n3. DDR4-3200 Theoretical Bandwidth:")
    dataRate := 3200 // MT/s
    busWidth := 8    // bytes (64 bits)
    bw := dataRate * busWidth
    fmt.Printf("   Single channel: %d MT/s × %d bytes = %d MB/s (%.1f GB/s)\n",
        dataRate, busWidth, bw, float64(bw)/1000)
    fmt.Printf("   Dual channel: %.1f GB/s\n", float64(bw)*2/1000)
}`,
				},
				{
					Title: "Cache Coherence Protocols",
					Content: `When multiple CPU cores each have private caches, they can hold copies of the same memory location. Cache coherence protocols ensure all cores see a consistent view of memory.

**The Coherence Problem:**
` + "```" + `
Core 0 Cache     Core 1 Cache     Memory
┌──────────┐    ┌──────────┐    ┌──────────┐
│ X = 42   │    │ X = 42   │    │ X = 42   │
└──────────┘    └──────────┘    └──────────┘

Core 0 writes X = 100:
┌──────────┐    ┌──────────┐    ┌──────────┐
│ X = 100  │    │ X = 42   │    │ X = 42   │  ← INCONSISTENT!
└──────────┘    └──────────┘    └──────────┘

Core 1 reads X → gets stale value 42!

Coherence invariants:
1. A read must return the value of the most recent write
2. Writes to the same location are serialized (all cores see them in same order)
3. A write is eventually visible to all cores
` + "```" + `

**MESI Protocol:**
` + "```" + `
Each cache line is in one of 4 states:

M (Modified):  Line is dirty, only copy in this cache
               Core can read/write without bus transaction
               Must write back to memory before others can read

E (Exclusive): Line is clean, only copy in this cache
               Core can read freely, can write (transitions to M)
               No bus transaction needed for write

S (Shared):    Line is clean, may exist in other caches too
               Core can read freely
               Write requires invalidation of other copies

I (Invalid):   Line is not valid in this cache
               Any access is a cache miss

State Transitions:
                    ┌───────────────────────────┐
             ┌──────│        Modified (M)        │──────┐
             │      └─────────────┬─────────────┘      │
         BusRd/                   │                  BusRdX/
         Flush                Read Hit               Flush
             │                    │                     │
             ↓           Write    ↓                     ↓
    ┌────────────┐    ┌───────────────────┐    ┌────────────┐
    │ Shared (S) │←───│  Exclusive (E)    │    │ Invalid(I) │
    └────────────┘    └───────────────────┘    └────────────┘
          │                    ↑                      ↑
          │              Read Miss                    │
          │            (no sharers)                   │
          │                    │                      │
          └────Write───→ Invalidate all ──────────────┘
                         other copies

Key: 
  BusRd = Another core wants to read this line
  BusRdX = Another core wants to read for ownership (will write)
  Flush = Supply data and transition
` + "```" + `

**MOESI Protocol (AMD):**
` + "```" + `
Adds Owner (O) state:
  O (Owner): Line is dirty AND shared
    - This core is responsible for supplying data on requests
    - Memory is stale — owner has latest copy
    - Other cores may have Shared copies
    - Avoids writing back to memory on sharing

Benefit: When M→S transition occurs, instead of writing back:
  MESI: Core writes to memory, both get S state
  MOESI: Core becomes O (dirty shared), other gets S
         → Saves a memory write!

Cache-to-cache transfer:
  MESI:  Modified cache → Memory → Requesting cache (slow)
  MOESI: Modified cache → Requesting cache directly (fast)
         Owner retains O state, requestor gets S
` + "```" + `

**Directory-Based Coherence:**
` + "```" + `
Snooping doesn't scale beyond ~8 cores (bus bandwidth bottleneck).
Directory-based protocols scale to hundreds of cores.

Directory Entry for each cache line in memory:
┌──────────┬──────────────────────────┐
│  State   │  Sharers bit-vector      │
│ (U/S/M)  │  (one bit per core)      │
└──────────┴──────────────────────────┘

U = Uncached:  No core has this line
S = Shared:    Multiple cores may have clean copies
M = Modified:  Exactly one core has dirty copy

Read Miss (line in M state at Core X):
  1. Requesting Core → Home node: "I want to read line A"
  2. Home node looks up directory: line A is in M state at Core X
  3. Home → Core X: "Send line A to Core Y, transition to S"
  4. Core X → Core Y: sends data, transitions M → S
  5. Home updates directory: S state, sharers = {X, Y}

Write Miss (line in S state at Cores X, Y):
  1. Core Z → Home: "I want to write line A"
  2. Home sees S state, sharers = {X, Y}
  3. Home → Core X: "Invalidate line A"
  4. Home → Core Y: "Invalidate line A"
  5. X and Y acknowledge invalidation
  6. Home → Core Z: "You have exclusive access"
  7. Home updates: M state, owner = Z
` + "```" + `

**False Sharing:**
` + "```" + `
Two cores write to DIFFERENT variables that happen to be 
on the SAME cache line:

struct { int x; int y; } data;  // x and y on same 64-byte line

Core 0: repeatedly writes data.x
Core 1: repeatedly writes data.y

Even though x and y are independent:
  Core 0 writes x → invalidates Core 1's line
  Core 1 writes y → invalidates Core 0's line
  → Cache line ping-pongs between cores
  → Performance drops 10-100x!

Solution: Pad structures to cache line boundaries:
struct {
    int x;
    char pad[60];  // Ensure y is on different cache line
    int y;
};
` + "```" + ``,
					CodeExamples: `// MESI cache coherence protocol simulator
package main

import "fmt"

type CacheState int
const (
    Invalid CacheState = iota
    Shared
    Exclusive
    Modified
)

func (s CacheState) String() string {
    return [...]string{"Invalid", "Shared", "Exclusive", "Modified"}[s]
}

type CacheLine struct {
    state CacheState
    tag   uint64
    data  int
}

type CoreCache struct {
    id    int
    lines map[uint64]*CacheLine
}

func NewCoreCache(id int) *CoreCache {
    return &CoreCache{id: id, lines: make(map[uint64]*CacheLine)}
}

type CoherenceController struct {
    caches  []*CoreCache
    memory  map[uint64]int
    bus     []string
}

func NewCoherenceController(numCores int) *CoherenceController {
    cc := &CoherenceController{
        memory: make(map[uint64]int),
    }
    for i := 0; i < numCores; i++ {
        cc.caches = append(cc.caches, NewCoreCache(i))
    }
    return cc
}

func (cc *CoherenceController) log(msg string) {
    cc.bus = append(cc.bus, msg)
    fmt.Printf("  BUS: %s\n", msg)
}

func (cc *CoherenceController) Read(coreID int, addr uint64) int {
    cache := cc.caches[coreID]
    line, exists := cache.lines[addr]

    if exists && line.state != Invalid {
        fmt.Printf("Core %d READ addr 0x%X: HIT (%s) value=%d\n",
            coreID, addr, line.state, line.data)
        return line.data
    }

    // Cache miss — need to get data
    cc.log(fmt.Sprintf("Core %d BusRd addr 0x%X", coreID, addr))

    // Check other caches
    suppliedBy := -1
    value := cc.memory[addr]
    otherShared := false

    for i, other := range cc.caches {
        if i == coreID { continue }
        otherLine, ok := other.lines[addr]
        if !ok || otherLine.state == Invalid { continue }

        switch otherLine.state {
        case Modified:
            // Flush to memory and supply
            cc.memory[addr] = otherLine.data
            value = otherLine.data
            otherLine.state = Shared
            suppliedBy = i
            otherShared = true
            cc.log(fmt.Sprintf("Core %d supplies dirty data, M→S", i))
        case Exclusive:
            otherLine.state = Shared
            value = otherLine.data
            suppliedBy = i
            otherShared = true
            cc.log(fmt.Sprintf("Core %d E→S", i))
        case Shared:
            value = otherLine.data
            otherShared = true
        }
    }

    // Allocate new line
    newState := Exclusive
    if otherShared {
        newState = Shared
    }
    cache.lines[addr] = &CacheLine{state: newState, tag: addr, data: value}

    src := "memory"
    if suppliedBy >= 0 { src = fmt.Sprintf("Core %d", suppliedBy) }
    fmt.Printf("Core %d READ addr 0x%X: MISS → %s (from %s) value=%d\n",
        coreID, addr, newState, src, value)
    return value
}

func (cc *CoherenceController) Write(coreID int, addr uint64, value int) {
    cache := cc.caches[coreID]
    line, exists := cache.lines[addr]

    if exists {
        switch line.state {
        case Modified:
            line.data = value
            fmt.Printf("Core %d WRITE addr 0x%X: HIT (Modified) value=%d\n",
                coreID, addr, value)
            return
        case Exclusive:
            line.data = value
            line.state = Modified
            fmt.Printf("Core %d WRITE addr 0x%X: HIT (E→M) value=%d\n",
                coreID, addr, value)
            return
        case Shared:
            // Need to invalidate others
            cc.log(fmt.Sprintf("Core %d BusUpgr addr 0x%X (invalidate others)", coreID, addr))
        }
    } else {
        // Write miss
        cc.log(fmt.Sprintf("Core %d BusRdX addr 0x%X", coreID, addr))
    }

    // Invalidate all other copies
    for i, other := range cc.caches {
        if i == coreID { continue }
        otherLine, ok := other.lines[addr]
        if !ok { continue }
        if otherLine.state != Invalid {
            if otherLine.state == Modified {
                cc.memory[addr] = otherLine.data
                cc.log(fmt.Sprintf("Core %d flushes dirty data before invalidation", i))
            }
            otherLine.state = Invalid
            cc.log(fmt.Sprintf("Core %d invalidated", i))
        }
    }

    cache.lines[addr] = &CacheLine{state: Modified, tag: addr, data: value}
    fmt.Printf("Core %d WRITE addr 0x%X: value=%d (Modified)\n",
        coreID, addr, value)
}

func (cc *CoherenceController) PrintState(addr uint64) {
    fmt.Printf("\n--- State for addr 0x%X ---\n", addr)
    for i, cache := range cc.caches {
        line, exists := cache.lines[addr]
        if exists && line.state != Invalid {
            fmt.Printf("  Core %d: %s, value=%d\n", i, line.state, line.data)
        } else {
            fmt.Printf("  Core %d: Invalid\n", i)
        }
    }
    fmt.Printf("  Memory: %d\n", cc.memory[addr])
    fmt.Println()
}

func main() {
    cc := NewCoherenceController(4)
    cc.memory[0x1000] = 42

    fmt.Println("=== MESI Coherence Protocol Demo ===\n")

    // Core 0 reads (gets Exclusive)
    cc.Read(0, 0x1000)
    cc.PrintState(0x1000)

    // Core 1 reads (both become Shared)
    cc.Read(1, 0x1000)
    cc.PrintState(0x1000)

    // Core 0 writes (invalidates Core 1)
    cc.Write(0, 0x1000, 100)
    cc.PrintState(0x1000)

    // Core 2 reads (gets data from Core 0)
    cc.Read(2, 0x1000)
    cc.PrintState(0x1000)

    // Core 1 writes (invalidates others)
    cc.Write(1, 0x1000, 200)
    cc.PrintState(0x1000)

    fmt.Printf("Bus transactions: %d\n", len(cc.bus))
}`,
				},
				{
					Title: "Memory Controllers and NUMA Architectures",
					Content: `The memory controller is the bridge between the CPU and DRAM. Its design profoundly impacts system performance, especially in multi-socket and multi-core systems.

**Memory Controller Functions:**
` + "```" + `
Request Queue → Scheduler → DRAM Command Generator → DRAM Interface

1. Request Buffering:
   - Receives read/write requests from last-level cache
   - Buffers requests in read/write queues
   - Typical: 32-64 entry read queue, 32-64 entry write queue

2. Address Translation:
   Physical addr → {Channel, Rank, Bank Group, Bank, Row, Column}
   
   Different mappings optimize for different workloads:
   - Row interleaving: consecutive addresses → different rows (streaming)
   - Bank interleaving: consecutive addresses → different banks (random)
   - Channel interleaving: consecutive addresses → different channels

3. Request Scheduling:
   Goal: maximize throughput while maintaining fairness
   
   FR-FCFS (First-Ready, First-Come-First-Served):
   Priority 1: Row buffer hits (already open, fast access)
   Priority 2: Oldest request (FCFS among row misses)
   
   This prioritizes row hits → higher throughput
   But can starve requests to closed rows!

4. Refresh Management:
   Every 64 ms: all rows must be refreshed
   - Distributed: refresh one row every 7.8 µs
   - Postponed: delay refresh when busy, catch up later
   - Same-bank refresh (DDR5): only one bank unavailable
` + "```" + `

**Write Policies:**
` + "```" + `
Write Buffer:
  Writes don't block the CPU — they're buffered:
  
  CPU → Write Buffer → [Drain to DRAM when convenient]
  
  Write buffer draining strategies:
  - Eager: Drain immediately
  - Lazy: Drain when buffer nearly full
  - Opportunistic: Drain during read pauses
  
  Read-Write Turnaround:
  Switching between reads and writes costs ~7.5 ns on DDR4
  → Memory scheduler batches reads together, writes together
  → "Write drain" mode: flush write buffer when it's nearly full
  
  Typical split: 70% reads, 30% writes
  Write high watermark (start draining): 80% full
  Write low watermark (stop draining): 40% full
` + "```" + `

**NUMA (Non-Uniform Memory Access):**
` + "```" + `
In multi-socket systems, each CPU socket has its own memory controller:

┌─────────────────────────────────────────────────────┐
│  Socket 0                    Socket 1                │
│  ┌──────────┐               ┌──────────┐            │
│  │  Cores   │               │  Cores   │            │
│  │  0-15    │←── QPI/UPI ──→│  16-31   │            │
│  │          │  (inter-CPU)  │          │            │
│  └────┬─────┘               └────┬─────┘            │
│       │                          │                   │
│  ┌────┴─────┐               ┌────┴─────┐            │
│  │ Memory   │               │ Memory   │            │
│  │ Ctrl 0   │               │ Memory   │            │
│  └────┬─────┘               │ Ctrl 1   │            │
│       │                     └────┬─────┘            │
│  ┌────┴─────┐               ┌────┴─────┐            │
│  │ DRAM 0   │               │ DRAM 1   │            │
│  │ (local)  │               │ (local)  │            │
│  └──────────┘               └──────────┘            │
└─────────────────────────────────────────────────────┘

Local memory access (same socket):  ~80 ns
Remote memory access (other socket): ~140 ns (1.75x slower!)

NUMA Ratio = Remote latency / Local latency
  Typical: 1.5x - 2x for 2-socket
  Gets worse with more sockets

Implications for software:
  - Allocate memory on the socket where the thread runs
  - Pin threads to cores near their data
  - Linux: numactl --membind=0 ./program
  - Linux: first-touch policy (memory allocated on first access node)
` + "```" + `

**ECC (Error-Correcting Code) Memory:**
` + "```" + `
Soft errors: Cosmic rays and alpha particles can flip bits in DRAM
  Rate: ~1 bit flip per GB per month (at sea level)
  
SECDED (Single Error Correct, Double Error Detect):
  - Uses Hamming code with extra parity bit
  - For 64-bit data: needs 8 check bits (72 bits total)
  - Can correct any single-bit error
  - Can detect (but not correct) any double-bit error

ECC operation:
  Write: compute syndrome bits from data, store data + ECC bits
  Read:  compute syndrome from stored data + ECC bits
         syndrome = 0 → no error
         syndrome ≠ 0 (weight 1) → single-bit error, correct it
         syndrome ≠ 0 (weight 2) → double-bit error, flag UE

Chipkill (AMD) / SDDC (Intel):
  - Can correct failure of an entire DRAM chip
  - Stripes ECC across multiple chips
  - Critical for server reliability

DDR5 On-Die ECC:
  - ECC computed WITHIN each DRAM chip
  - Corrects errors before data leaves the chip
  - Transparent to the memory controller
  - Reduces error rate seen by system ECC
` + "```" + ``,
					CodeExamples: `// Memory controller and NUMA simulation
package main

import (
    "fmt"
    "math/rand"
)

// Memory controller request scheduler
type MemRequest struct {
    addr      uint64
    isWrite   bool
    data      int
    row       int
    bank      int
    col       int
    timestamp int
}

type MemScheduler struct {
    readQueue  []MemRequest
    writeQueue []MemRequest
    bankState  []int // Open row per bank (-1 = closed)
    numBanks   int
    cycle      int
    
    // Stats
    rowHits    int
    rowMisses  int
    rowConflicts int
}

func NewMemScheduler(numBanks int) *MemScheduler {
    state := make([]int, numBanks)
    for i := range state { state[i] = -1 }
    return &MemScheduler{
        bankState: state,
        numBanks:  numBanks,
    }
}

func (ms *MemScheduler) Enqueue(req MemRequest) {
    req.timestamp = ms.cycle
    req.bank = int((req.addr >> 6) % uint64(ms.numBanks))
    req.row = int((req.addr >> 16) & 0x3FFF)
    req.col = int((req.addr >> 6) & 0x3FF)

    if req.isWrite {
        ms.writeQueue = append(ms.writeQueue, req)
    } else {
        ms.readQueue = append(ms.readQueue, req)
    }
}

// FR-FCFS scheduling
func (ms *MemScheduler) Schedule() *MemRequest {
    // Priority to reads over writes (unless write queue full)
    queue := &ms.readQueue
    if len(ms.readQueue) == 0 || len(ms.writeQueue) > 48 {
        queue = &ms.writeQueue
    }
    if len(*queue) == 0 { return nil }

    // Find best request: prefer row hits
    bestIdx := 0
    bestIsHit := false
    bestAge := 0

    for i, req := range *queue {
        isHit := ms.bankState[req.bank] == req.row
        age := ms.cycle - req.timestamp

        // FR-FCFS: hits first, then oldest
        if isHit && !bestIsHit {
            bestIdx = i
            bestIsHit = true
            bestAge = age
        } else if isHit == bestIsHit && age > bestAge {
            bestIdx = i
            bestAge = age
        }
    }

    req := (*queue)[bestIdx]
    *queue = append((*queue)[:bestIdx], (*queue)[bestIdx+1:]...)

    // Update stats
    if ms.bankState[req.bank] == req.row {
        ms.rowHits++
    } else if ms.bankState[req.bank] == -1 {
        ms.rowMisses++
    } else {
        ms.rowConflicts++
    }
    ms.bankState[req.bank] = req.row

    ms.cycle++
    return &req
}

// NUMA topology simulation
type NUMANode struct {
    id        int
    cores     []int
    localMem  map[uint64]int
    localLat  int
    remoteLat int
}

type NUMASystem struct {
    nodes       []*NUMANode
    accessCount [2][2]int // [from_node][to_node] access counts
}

func NewNUMASystem() *NUMASystem {
    return &NUMASystem{
        nodes: []*NUMANode{
            {id: 0, cores: []int{0, 1, 2, 3}, localMem: make(map[uint64]int),
                localLat: 80, remoteLat: 140},
            {id: 1, cores: []int{4, 5, 6, 7}, localMem: make(map[uint64]int),
                localLat: 80, remoteLat: 140},
        },
    }
}

func (ns *NUMASystem) addrToNode(addr uint64) int {
    // Simple interleaving: even pages → node 0, odd pages → node 1
    return int((addr >> 12) % 2)
}

func (ns *NUMASystem) Access(coreID int, addr uint64) int {
    fromNode := 0
    if coreID >= 4 { fromNode = 1 }
    toNode := ns.addrToNode(addr)
    ns.accessCount[fromNode][toNode]++

    if fromNode == toNode {
        return ns.nodes[toNode].localLat
    }
    return ns.nodes[toNode].remoteLat
}

// ECC simulation
type ECCWord struct {
    data     uint64
    ecc      uint8  // 8-bit ECC for 64-bit data
}

func computeECC(data uint64) uint8 {
    var ecc uint8
    for i := 0; i < 64; i++ {
        if data&(1<<i) != 0 {
            ecc ^= uint8(i + 1)
        }
    }
    return ecc
}

func checkECC(word ECCWord) (corrected uint64, errType string) {
    syndrome := computeECC(word.data) ^ word.ecc
    if syndrome == 0 {
        return word.data, "no error"
    }
    // Single-bit error: syndrome gives bit position
    if syndrome <= 64 {
        corrected = word.data ^ (1 << (syndrome - 1))
        return corrected, "corrected single-bit"
    }
    return word.data, "uncorrectable"
}

func main() {
    // Memory scheduler demo
    ms := NewMemScheduler(8)
    
    // Simulate mixed access pattern
    fmt.Println("=== Memory Scheduler (FR-FCFS) ===")
    
    // Sequential reads (should get row hits)
    for i := 0; i < 10; i++ {
        ms.Enqueue(MemRequest{addr: uint64(0x10000 + i*64)})
    }
    // Random reads (likely row conflicts)
    for i := 0; i < 10; i++ {
        ms.Enqueue(MemRequest{addr: uint64(rand.Intn(1 << 24))})
    }
    
    for {
        req := ms.Schedule()
        if req == nil { break }
    }
    
    total := ms.rowHits + ms.rowMisses + ms.rowConflicts
    fmt.Printf("Row hits: %d (%.0f%%), Misses: %d, Conflicts: %d\n",
        ms.rowHits, float64(ms.rowHits)/float64(total)*100,
        ms.rowMisses, ms.rowConflicts)

    // NUMA demo
    fmt.Println("\n=== NUMA Access Patterns ===")
    numa := NewNUMASystem()
    
    totalLocal, totalRemote := 0, 0
    // Core 0 (node 0) accessing node 0 memory (local)
    for i := 0; i < 100; i++ {
        lat := numa.Access(0, uint64(i*4096*2)) // Even pages → node 0
        totalLocal += lat
    }
    // Core 0 (node 0) accessing node 1 memory (remote)
    for i := 0; i < 100; i++ {
        lat := numa.Access(0, uint64(i*4096*2+4096)) // Odd pages → node 1
        totalRemote += lat
    }
    
    fmt.Printf("Local accesses:  avg %d ns\n", totalLocal/100)
    fmt.Printf("Remote accesses: avg %d ns\n", totalRemote/100)
    fmt.Printf("NUMA ratio: %.2fx\n", float64(totalRemote)/float64(totalLocal))

    // ECC demo
    fmt.Println("\n=== ECC Memory Demo ===")
    data := uint64(0xDEADBEEFCAFEBABE)
    ecc := computeECC(data)
    word := ECCWord{data: data, ecc: ecc}
    fmt.Printf("Original: 0x%016X, ECC: 0x%02X\n", data, ecc)
    
    _, status := checkECC(word)
    fmt.Printf("Check clean data: %s\n", status)
    
    // Inject single-bit error
    word.data ^= (1 << 42) // Flip bit 42
    fmt.Printf("Corrupted: 0x%016X\n", word.data)
    corrected, status := checkECC(word)
    fmt.Printf("Check: %s → 0x%016X\n", status, corrected)
    fmt.Printf("Matches original: %v\n", corrected == data)
}`,
				},
				{
					Title: "Advanced Cache Replacement and Prefetching",
					Content: `Cache replacement policies decide which line to evict when the cache is full, and prefetching proactively loads data before it's needed. Together, they significantly impact hit rates.

**Cache Replacement Policies:**
` + "```" + `
LRU (Least Recently Used):
  - Evict the line that hasn't been used for the longest time
  - Optimal for temporal locality workloads
  - Hardware cost: need to maintain recency order
  
  For N-way set-associative: need log₂(N!) bits per set
  Example: 16-way → 44 bits per set (expensive!)
  
  In practice: use Pseudo-LRU (PLRU) — tree-based approximation
  Binary tree points: 15 bits for 16-way (vs 44 for true LRU)

  Tree-PLRU for 4-way cache:
           [0]
          /   \
        [1]   [2]
        / \   / \
       W0 W1 W2 W3
  
  Each pointer indicates the "less recently used" subtree.
  Evict: follow pointers to leaf.
  Access: flip pointers from root to accessed way.

RRIP (Re-Reference Interval Prediction):
  - Used in Intel processors since Ivy Bridge
  - Each line has a 2-3 bit counter (RRIP value)
  - Higher value = predicted longer until next re-reference
  - On access: set RRIP to 0 (near-immediate re-reference)
  - On eviction: find line with max RRIP, increment others if none at max
  
  SRRIP (Static): New lines get RRIP = 2 (long interval)
  BRRIP (Bimodal): Most new lines get RRIP = 3, rarely 2
  DRRIP (Dynamic): Monitor both, use whichever is better (set dueling)

BIP (Bimodal Insertion Policy):
  - Most lines inserted at LRU position (easy to evict)
  - Occasionally inserted at MRU position
  - Helps with scanning/thrashing workloads where LRU fails

Bélády's Algorithm (OPT):
  - Evict the line that will be used farthest in the future
  - Optimal but requires future knowledge (impossible in hardware)
  - Used as upper bound for evaluating other policies
` + "```" + `

**Hardware Prefetching:**
` + "```" + `
Next-Line Prefetcher:
  On access to line N, prefetch line N+1
  Simple but effective for sequential access
  Accuracy: ~50% (many useless prefetches for non-sequential)

Stride Prefetcher:
  Track access patterns to detect strides:
  
  Access history: 0x1000, 0x1040, 0x1080, 0x10C0
  Detected stride: 64 bytes
  Prefetch: 0x1100, 0x1140, ...
  
  Implementation: Reference Prediction Table
  ┌──────┬──────────┬──────┬───────┐
  │  PC  │ Last Addr│Stride│ State │
  ├──────┼──────────┼──────┼───────┤
  │ 0x4A │ 0x10C0   │  64  │ Steady│
  │ 0x8C │ 0x2480   │ 128  │ Init  │
  └──────┴──────────┴──────┴───────┘
  
  State machine: Init → Transient → Steady
  Only prefetch in Steady state (stride confirmed)

Spatial Pattern Prefetcher:
  When a cache line is first brought in, record which OTHER lines
  in the same spatial region are accessed.
  
  On future access to a new region, prefetch the recorded pattern.
  
  Region: 1KB (16 cache lines)
  Pattern: bitmap of which lines accessed
  Example: pattern = 0b1010110100000001
  → When new region accessed, prefetch lines matching pattern

Stream Prefetcher:
  Detect ascending or descending address streams
  Prefetch ahead in the stream direction
  Multiple stream trackers for out-of-order accesses
  
  Intel typically has 32 stream trackers per core
  Can prefetch 4-8 lines ahead of demand
` + "```" + `

**Prefetch Timeliness and Accuracy:**
` + "```" + `
Perfect prefetch: arrives EXACTLY when needed (hides all latency)
Too early: may get evicted before use → wastes cache capacity
Too late: partially hides latency → some stall remains
Useless prefetch: never used → wastes bandwidth and cache space

Metrics:
  Accuracy = useful prefetches / total prefetches
  Coverage = L2 misses eliminated / total L2 misses without prefetch
  Timeliness = fraction of useful prefetches that arrived on time

Prefetch throttling:
  If accuracy drops below threshold:
  → Reduce prefetch distance (less aggressive)
  → Reduce number of prefetch streams
  → In extreme cases, disable prefetching

Prefetch-aware replacement:
  Don't give prefetched lines high priority until they're actually used
  Avoids "demand pollution" — prefetches evicting useful data
` + "```" + ``,
					CodeExamples: `// Cache replacement and prefetching simulation
package main

import (
    "fmt"
    "math/rand"
)

// LRU Cache
type LRUCache struct {
    ways    int
    sets    int
    cache   [][]int64  // [set][way] = tag (-1 = invalid)
    order   [][]int    // [set][position] = way (0=MRU, n-1=LRU)
    hits    int
    misses  int
}

func NewLRUCache(sets, ways int) *LRUCache {
    c := &LRUCache{ways: ways, sets: sets}
    c.cache = make([][]int64, sets)
    c.order = make([][]int, sets)
    for i := range c.cache {
        c.cache[i] = make([]int64, ways)
        c.order[i] = make([]int, ways)
        for j := range c.cache[i] {
            c.cache[i][j] = -1
            c.order[i][j] = j
        }
    }
    return c
}

func (c *LRUCache) Access(addr int64) bool {
    set := int(addr / 64) % c.sets
    tag := addr / int64(64*c.sets)

    // Check for hit
    for w := 0; w < c.ways; w++ {
        if c.cache[set][w] == tag {
            c.hits++
            c.promote(set, w)
            return true
        }
    }

    // Miss — evict LRU
    c.misses++
    lruWay := c.order[set][c.ways-1]
    c.cache[set][lruWay] = tag
    c.promote(set, lruWay)
    return false
}

func (c *LRUCache) promote(set, way int) {
    pos := -1
    for i, w := range c.order[set] {
        if w == way { pos = i; break }
    }
    if pos < 0 { return }
    copy(c.order[set][1:pos+1], c.order[set][0:pos])
    c.order[set][0] = way
}

func (c *LRUCache) HitRate() float64 {
    total := c.hits + c.misses
    if total == 0 { return 0 }
    return float64(c.hits) / float64(total) * 100
}

// Stride Prefetcher
type StridePrefetcher struct {
    entries     map[uint64]*strideEntry
    prefetched  map[int64]bool
    useful      int
    useless     int
    total       int
}

type strideEntry struct {
    lastAddr int64
    stride   int64
    state    int // 0=init, 1=transient, 2=steady
}

func NewStridePrefetcher() *StridePrefetcher {
    return &StridePrefetcher{
        entries:    make(map[uint64]*strideEntry),
        prefetched: make(map[int64]bool),
    }
}

func (sp *StridePrefetcher) Observe(pc uint64, addr int64) []int64 {
    var prefetches []int64
    
    entry, exists := sp.entries[pc]
    if !exists {
        sp.entries[pc] = &strideEntry{lastAddr: addr}
        return nil
    }

    newStride := addr - entry.lastAddr
    entry.lastAddr = addr

    if entry.state == 0 {
        entry.stride = newStride
        entry.state = 1
    } else if entry.state == 1 {
        if newStride == entry.stride {
            entry.state = 2 // Confirmed
        } else {
            entry.stride = newStride
        }
    }

    // Only prefetch in steady state
    if entry.state == 2 && newStride == entry.stride {
        for i := 1; i <= 4; i++ {
            prefAddr := addr + entry.stride*int64(i)
            prefetches = append(prefetches, prefAddr)
            sp.prefetched[prefAddr] = true
            sp.total++
        }
    }

    return prefetches
}

func (sp *StridePrefetcher) WasUseful(addr int64) {
    if sp.prefetched[addr] {
        sp.useful++
        delete(sp.prefetched, addr)
    }
}

func (sp *StridePrefetcher) Accuracy() float64 {
    if sp.total == 0 { return 0 }
    return float64(sp.useful) / float64(sp.total) * 100
}

func main() {
    // Compare LRU with different associativities
    fmt.Println("=== LRU Cache: Associativity Impact ===")
    
    // Generate workload: 80% spatial locality, 20% random
    workload := make([]int64, 10000)
    for i := range workload {
        if rand.Float64() < 0.8 {
            workload[i] = int64(i%256) * 64 // 256 lines working set
        } else {
            workload[i] = int64(rand.Intn(4096)) * 64
        }
    }

    for _, ways := range []int{1, 2, 4, 8, 16} {
        cache := NewLRUCache(64, ways)
        for _, addr := range workload {
            cache.Access(addr)
        }
        fmt.Printf("  %2d-way: Hit rate = %.1f%% (hits=%d, misses=%d)\n",
            ways, cache.HitRate(), cache.hits, cache.misses)
    }

    // Stride prefetcher demo
    fmt.Println("\n=== Stride Prefetcher ===")
    prefetcher := NewStridePrefetcher()
    cacheWithPF := NewLRUCache(64, 8)
    cacheNoPF := NewLRUCache(64, 8)

    // Strided access pattern (simulating array traversal with stride 128)
    pc := uint64(0x4000) // Instruction PC
    for i := 0; i < 1000; i++ {
        addr := int64(i) * 128

        // Without prefetcher
        cacheNoPF.Access(addr)

        // With prefetcher
        hit := cacheWithPF.Access(addr)
        if hit {
            prefetcher.WasUseful(addr)
        }
        prefetches := prefetcher.Observe(pc, addr)
        for _, pa := range prefetches {
            cacheWithPF.Access(pa)
        }
    }

    fmt.Printf("  Without prefetch: Hit rate = %.1f%%\n", cacheNoPF.HitRate())
    fmt.Printf("  With prefetch:    Hit rate = %.1f%%\n", cacheWithPF.HitRate())
    fmt.Printf("  Prefetch accuracy: %.1f%%\n", prefetcher.Accuracy())
}`,
				},
			},
		},
	})
}
