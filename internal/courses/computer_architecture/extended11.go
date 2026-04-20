package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2834,
			Title:       "Interconnects and Network-on-Chip",
			Description: "Study on-chip and off-chip interconnection networks, bus protocols, coherence fabrics, and the communication backbone of modern multicore processors.",
			Order:       34,
			Lessons: []problems.Lesson{
				{
					Title: "On-Chip Interconnects",
					Content: `As processors evolved from single-core to many-core designs, the interconnection network became a critical component. The network-on-chip (NoC) often determines system performance more than individual core speed.

**Bus vs Crossbar vs Mesh:**
` + "```" + `
Shared Bus (simplest, oldest):
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │Core 0│ │Core 1│ │Core 2│ │Core 3│
  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
  ═══╪════════╪════════╪════════╪════ Shared Bus
  └──┬───┘ └──┬───┘
  │ L3   │ │ Mem  │
  │Cache │ │ Ctrl │
  └──────┘ └──────┘
  
  Pros: Simple, low area
  Cons: ONE transaction at a time → doesn't scale
  Limit: ~4-8 agents before bottleneck
  Used in: Simple MCUs, older systems

Crossbar Switch:
  Every source connected to every destination
  
       Dest0  Dest1  Dest2  Dest3
  Src0  ──X────X────X────X──
  Src1  ──X────X────X────X──
  Src2  ──X────X────X────X──
  Src3  ──X────X────X────X──
  
  Pros: Full bandwidth, any-to-any simultaneous
  Cons: Area grows O(N²) with ports → prohibitive for many cores
  Limit: ~8-16 ports practical
  Used in: AMD Zen CCX (8-core crossbar within CCD)

Ring (Intel since Sandy Bridge):
  ┌─Core0─Core1─Core2─Core3─┐
  │                          │
  └─LLC───LLC───LLC───MemC──┘  (bidirectional)
  
  Pros: Simple routing, good bandwidth for moderate core counts
  Cons: Latency grows with ring size O(N), bandwidth O(1)
  Limit: ~12-20 cores per ring (multiple rings for more)
  Used in: Intel Core/Xeon (up to ~28 cores with 2 rings)

Mesh Network:
  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │C0+LLC├───┤C1+LLC├───┤C2+LLC├───┤C3+LLC│
  └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
     │          │          │          │
  ┌──┴───┐   ┌──┴───┐   ┌──┴───┐   ┌──┴───┐
  │C4+LLC├───┤C5+LLC├───┤C6+LLC├───┤C7+LLC│
  └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
     │          │          │          │
  ┌──┴───┐   ┌──┴───┐   ┌──┴───┐   ┌──┴───┐
  │C8+LLC├───┤C9+LLC├───┤MemC0 ├───┤MemC1 │
  └──────┘   └──────┘   └──────┘   └──────┘
  
  Pros: Scalable bandwidth O(N), lower latency than ring for large N
  Cons: Complex routing, more area and power
  Limit: Scales to 100+ nodes
  Used in: Intel Xeon Scalable (28+ cores), GPU interconnects
` + "```" + `

**Routing and Flow Control:**
` + "```" + `
Routing Algorithms:

Dimension-Ordered Routing (XY routing):
  In a 2D mesh, first route in X direction, then Y
  Simple, deadlock-free (no circular dependencies)
  
  Source: (0,0) → Destination: (3,2)
  Path: (0,0)→(1,0)→(2,0)→(3,0)→(3,1)→(3,2)
  First all X hops, then all Y hops

Adaptive Routing:
  Can choose between multiple paths to avoid congestion
  More complex, needs deadlock avoidance mechanism
  Used when traffic patterns are non-uniform

Flow Control:

Store-and-Forward:
  Each router buffers entire packet before forwarding
  Latency = hops × (header + packet_size/bandwidth)
  High latency, simple hardware

Wormhole Routing:
  Packet split into flits (flow control units)
  Head flit reserves path, body flits follow pipeline-style
  Only need to buffer a few flits, not entire packet
  
  Head flit → Router0 → Router1 → Router2
                Body1  → Router0 → Router1
                          Body2  → Router0
  
  Latency ≈ hops × router_delay + packet_size/bandwidth
  Much lower latency than store-and-forward
  Used in most modern NoCs

Virtual Channels:
  Multiple virtual queues sharing one physical link
  Prevents head-of-line blocking
  Enables deadlock-free adaptive routing
  
  Physical link
  ┌──────────────────┐
  │ VC0: packet A    │
  │ VC1: packet B    │  ← packets share bandwidth
  │ VC2: (empty)     │
  │ VC3: packet C    │
  └──────────────────┘
` + "```" + `

**Coherence Interconnects:**
` + "```" + `
Snoopy Protocols (bus/ring-based):
  Every cache controller "snoops" all transactions
  When core 0 reads X: broadcast request
  All caches check if they have X and respond
  
  Works well for small core counts (broadcast scales poorly)
  Used in: Most desktop processors (2-16 cores)

Directory Protocols (mesh/large systems):
  Central directory tracks which caches hold each line
  
  Directory entry for cache line at address A:
  ┌──────────┬──────────┬─────────────┐
  │ State    │ Owner    │ Sharers     │
  │ (M/S/I)  │ (core#)  │ (bitmask)   │
  └──────────┴──────────┴─────────────┘
  
  Read request:
  1. Core sends request to home node (determined by address hash)
  2. Home node checks directory
  3. If Shared: send data to requester, add to sharers
  4. If Modified: forward request to owner, update directory
  5. Owner sends data to requester (intervention)
  
  Advantage: Point-to-point messages, no broadcast
  Scalable to 100s of cores
  Used in: Server processors, AMD EPYC, Intel Xeon Scalable

AMD Infinity Fabric:
  Scalable coherent interconnect for chiplet architecture
  
  Within CCD: Crossbar (8 cores)
  CCD ↔ IOD: Infinity Fabric links (32B/cycle, 36 GB/s)
  Socket ↔ Socket: xGMI links (up to 4 sockets)
  
  Features:
  - Coherent (cache coherence across chiplets/sockets)
  - Low latency within CCD (~10ns), higher across CCDs (~40ns)
  - Bandwidth scales with link count
  
Intel UPI (Ultra Path Interconnect):
  Successor to QPI (Quick Path Interconnect)
  Up to 16 GT/s per direction
  3 links per socket (up to 8-socket systems)
  Full cache coherence across sockets
  
CXL (Compute Express Link):
  Built on PCIe 5.0/6.0 physical layer
  Three protocols:
    CXL.io:    PCIe-compatible I/O
    CXL.cache: Device can cache host memory
    CXL.mem:   Host can access device memory (memory expansion)
  
  Use cases:
  - Memory expansion (CXL-attached DRAM)
  - Memory pooling (shared memory across servers)
  - Accelerator coherence (GPU/FPGA coherent with CPU)
` + "```" + ``,
					CodeExamples: `// Network-on-Chip simulation
package main

import (
    "fmt"
    "math"
)

// Mesh network topology
type MeshNetwork struct {
    rows, cols int
    routers    [][]Router
}

type Router struct {
    x, y        int
    bufferDepth int
    packetsIn   int
    packetsOut  int
}

func NewMesh(rows, cols int) *MeshNetwork {
    m := &MeshNetwork{rows: rows, cols: cols}
    m.routers = make([][]Router, rows)
    for i := range m.routers {
        m.routers[i] = make([]Router, cols)
        for j := range m.routers[i] {
            m.routers[i][j] = Router{x: j, y: i, bufferDepth: 4}
        }
    }
    return m
}

// XY routing: route in X first, then Y
func (m *MeshNetwork) XYRoute(srcX, srcY, dstX, dstY int) []Router {
    path := []Router{m.routers[srcY][srcX]}
    
    // Route in X direction first
    x, y := srcX, srcY
    for x != dstX {
        if x < dstX { x++ } else { x-- }
        path = append(path, m.routers[y][x])
    }
    // Then route in Y direction
    for y != dstY {
        if y < dstY { y++ } else { y-- }
        path = append(path, m.routers[y][x])
    }
    return path
}

func (m *MeshNetwork) Hops(srcX, srcY, dstX, dstY int) int {
    return abs(dstX-srcX) + abs(dstY-srcY)
}

func abs(x int) int { if x < 0 { return -x }; return x }

// Latency models for different topologies
type Topology struct {
    name     string
    nodes    int
    // Returns latency in cycles for a message between two nodes
    latency  func(src, dst, nodes int) float64
    // Total bisection bandwidth (relative)
    bisectBW func(nodes int) float64
    // Area (relative)
    area     func(nodes int) float64
}

func busLatency(src, dst, nodes int) float64 {
    return 3 + 1 // arbitration + transfer (constant but serialized)
}

func ringLatency(src, dst, nodes int) float64 {
    forward := (dst - src + nodes) % nodes
    backward := (src - dst + nodes) % nodes
    hops := forward
    if backward < forward { hops = backward }
    return float64(1 + hops) // 1 cycle per hop
}

func meshLatency(src, dst, nodes int) float64 {
    side := int(math.Sqrt(float64(nodes)))
    srcX, srcY := src%side, src/side
    dstX, dstY := dst%side, dst/side
    hops := abs(dstX-srcX) + abs(dstY-srcY)
    return float64(1 + hops) // 1 cycle per hop
}

func crossbarLatency(src, dst, nodes int) float64 {
    return 1 // Always 1 cycle (direct connection)
}

func main() {
    fmt.Println("=== Mesh Network XY Routing ===")
    mesh := NewMesh(4, 4)
    
    routes := []struct{ sx, sy, dx, dy int }{
        {0, 0, 3, 3},
        {0, 0, 3, 0},
        {1, 1, 2, 3},
        {3, 0, 0, 3},
    }
    
    for _, r := range routes {
        path := mesh.XYRoute(r.sx, r.sy, r.dx, r.dy)
        hops := mesh.Hops(r.sx, r.sy, r.dx, r.dy)
        fmt.Printf("(%d,%d) → (%d,%d): %d hops, path: ", r.sx, r.sy, r.dx, r.dy, hops)
        for i, p := range path {
            if i > 0 { fmt.Print("→") }
            fmt.Printf("(%d,%d)", p.x, p.y)
        }
        fmt.Println()
    }
    
    // Average hop count for mesh
    fmt.Println("\n=== Average Hop Count Analysis ===")
    for _, size := range []int{4, 9, 16, 36, 64} {
        side := int(math.Sqrt(float64(size)))
        totalHops := 0
        count := 0
        for sy := 0; sy < side; sy++ {
            for sx := 0; sx < side; sx++ {
                for dy := 0; dy < side; dy++ {
                    for dx := 0; dx < side; dx++ {
                        if sx == dx && sy == dy { continue }
                        totalHops += abs(dx-sx) + abs(dy-sy)
                        count++
                    }
                }
            }
        }
        avg := float64(totalHops) / float64(count)
        fmt.Printf("  %dx%d mesh (%2d nodes): avg hops = %.2f, max = %d\n",
            side, side, size, avg, 2*(side-1))
    }
    
    // Topology comparison
    fmt.Println("\n=== Topology Comparison ===")
    topologies := []Topology{
        {"Bus", 0,
            busLatency,
            func(n int) float64 { return 1 },
            func(n int) float64 { return float64(n) }},
        {"Ring", 0,
            ringLatency,
            func(n int) float64 { return 2 },
            func(n int) float64 { return float64(n) }},
        {"Crossbar", 0,
            crossbarLatency,
            func(n int) float64 { return float64(n * n / 4) },
            func(n int) float64 { return float64(n * n) }},
        {"2D Mesh", 0,
            meshLatency,
            func(n int) float64 { return 2 * math.Sqrt(float64(n)) },
            func(n int) float64 { return 4 * float64(n) }},
    }
    
    for _, nodeCount := range []int{4, 8, 16, 32, 64} {
        fmt.Printf("\n%d nodes:\n", nodeCount)
        fmt.Printf("  %-10s │ Avg Lat │ Max Lat │ BisectBW │ Area\n", "Topology")
        fmt.Println("  ───────────┼─────────┼─────────┼──────────┼──────")
        
        for _, t := range topologies {
            // Compute average and max latency
            totalLat := 0.0
            maxLat := 0.0
            count := 0
            for src := 0; src < nodeCount; src++ {
                for dst := 0; dst < nodeCount; dst++ {
                    if src == dst { continue }
                    lat := t.latency(src, dst, nodeCount)
                    totalLat += lat
                    if lat > maxLat { maxLat = lat }
                    count++
                }
            }
            avgLat := totalLat / float64(count)
            bw := t.bisectBW(nodeCount)
            area := t.area(nodeCount)
            
            fmt.Printf("  %-10s │ %5.1f   │ %5.1f   │ %6.0f   │ %5.0f\n",
                t.name, avgLat, maxLat, bw, area)
        }
    }
    
    // Wormhole vs Store-and-Forward latency
    fmt.Println("\n\n=== Wormhole vs Store-and-Forward ===")
    type FlowControl struct {
        name     string
        latencyFunc func(hops int, packetFlits int, routerDelay int) float64
    }
    
    storeForward := FlowControl{"Store-and-Forward",
        func(hops, packetFlits, routerDelay int) float64 {
            return float64(hops) * (float64(routerDelay) + float64(packetFlits))
        },
    }
    wormhole := FlowControl{"Wormhole",
        func(hops, packetFlits, routerDelay int) float64 {
            return float64(hops)*float64(routerDelay) + float64(packetFlits)
        },
    }
    
    fmt.Printf("%-20s │", "Hops/PacketSize")
    for _, pkt := range []int{1, 4, 8, 16} {
        fmt.Printf(" %d flits │", pkt)
    }
    fmt.Println()
    fmt.Println("─────────────────────┼────────┼────────┼────────┼────────")
    
    routerDelay := 2
    for _, hops := range []int{1, 4, 8, 16} {
        fmt.Printf("S&F   %2d hops        │", hops)
        for _, pkt := range []int{1, 4, 8, 16} {
            lat := storeForward.latencyFunc(hops, pkt, routerDelay)
            fmt.Printf(" %5.0f  │", lat)
        }
        fmt.Println()
        fmt.Printf("WH    %2d hops        │", hops)
        for _, pkt := range []int{1, 4, 8, 16} {
            lat := wormhole.latencyFunc(hops, pkt, routerDelay)
            fmt.Printf(" %5.0f  │", lat)
        }
        fmt.Println()
    }
}`,
				},
				{
					Title: "Off-Chip Interconnects and Memory Interfaces",
					Content: `Off-chip interconnects connect the processor to memory, I/O devices, and other processors. These links are the primary bandwidth and latency bottleneck in modern systems.

**DDR Memory Interface:**
` + "```" + `
DDR SDRAM Evolution:

Standard │ Data Rate │ Transfer Rate│ Voltage│ Prefetch
DDR1     │ 200-400   │ 1.6-3.2 GB/s│ 2.5V   │ 2n
DDR2     │ 400-1066  │ 3.2-8.5 GB/s│ 1.8V   │ 4n
DDR3     │ 800-2133  │ 6.4-17 GB/s │ 1.5V   │ 8n
DDR4     │ 1600-3200 │ 12.8-25.6   │ 1.2V   │ 8n
DDR5     │ 3200-8400 │ 25.6-67.2   │ 1.1V   │ 16n
LPDDR5   │ 3200-8533 │ 25.6-68.2   │ 1.05V  │ 16n

"Double Data Rate": Transfer on both clock edges
"Prefetch": Internal bus wider than external, burst transfers

DDR5 key improvements over DDR4:
  - Two independent 32-bit channels (was one 64-bit)
  - Higher bank count: 32 banks (4 bank groups × 8 banks)
  - On-die ECC: Built-in error correction
  - PMIC on DIMM: Better power regulation
  - Same-bank refresh: Other banks remain accessible

Bandwidth calculation:
  DDR5-5600 (single channel):
    Data rate: 5600 MT/s (megatransfers per second)
    Bus width: 64 bits = 8 bytes
    Bandwidth: 5600 × 8 = 44,800 MB/s = 44.8 GB/s
    
  Dual-channel DDR5-5600:
    89.6 GB/s total bandwidth
    
  Quad-channel (server):
    179.2 GB/s total bandwidth
` + "```" + `

**High Bandwidth Memory (HBM):**
` + "```" + `
HBM Architecture:
  3D-stacked DRAM dies on silicon interposer
  Connected to processor via thousands of micro-bumps
  
  ┌──────────┐
  │ DRAM Die 3│ ← 4-12 stacked DRAM dies
  ├──────────┤
  │ DRAM Die 2│
  ├──────────┤
  │ DRAM Die 1│
  ├──────────┤
  │ Logic Die │ ← Base die with TSVs and control
  └────┬─────┘
       │ micro-bumps (thousands per stack)
  ┌────┴──────────────────────────────────┐
  │        Silicon Interposer              │ ← Connects HBM to GPU/CPU
  └────┬──────────────────────────────────┘
       │
  ┌────┴─────┐
  │  GPU/CPU  │
  └──────────┘

HBM Evolution:
Standard │ Layers │ BW/stack │ Capacity │ Bus Width
HBM1     │ 4      │ 128 GB/s │ 4 GB     │ 1024 bit
HBM2     │ 8      │ 256 GB/s │ 8 GB     │ 1024 bit
HBM2E    │ 8      │ 460 GB/s │ 16 GB    │ 1024 bit
HBM3     │ 12     │ 665 GB/s │ 24 GB    │ 1024 bit
HBM3E    │ 12     │ 1.2 TB/s │ 36 GB    │ 1024 bit

Example: NVIDIA H100 GPU
  6 × HBM3 stacks = 80 GB, ~3.35 TB/s bandwidth
  Compare: DDR5 quad-channel ≈ 180 GB/s
  HBM is ~18× more bandwidth than DDR5!

Why HBM matters:
  AI/ML workloads are memory-bandwidth bound
  Matrix multiply: O(n³) compute, O(n²) data movement
  More bandwidth → higher achieved FLOPS for real workloads
` + "```" + `

**PCIe Architecture:**
` + "```" + `
PCIe Evolution:
Gen │ Rate/Lane │ Encoding  │ x16 BW (each dir) │ Year
1.0 │ 2.5 GT/s  │ 8b/10b   │ 4 GB/s             │ 2003
2.0 │ 5.0 GT/s  │ 8b/10b   │ 8 GB/s             │ 2007
3.0 │ 8.0 GT/s  │ 128b/130b│ 15.75 GB/s         │ 2010
4.0 │ 16 GT/s   │ 128b/130b│ 31.5 GB/s          │ 2017
5.0 │ 32 GT/s   │ 128b/130b│ 63 GB/s            │ 2019
6.0 │ 64 GT/s   │ PAM-4    │ 126 GB/s           │ 2022
7.0 │ 128 GT/s  │ PAM-4    │ 242 GB/s           │ 2025(est)

Encoding overhead:
  8b/10b: 20% overhead (8 data bits encoded as 10)
  128b/130b: ~1.5% overhead
  PAM-4: 4-level signaling (2 bits per symbol)

PCIe is packet-based (not bus-based):
  Transaction Layer Packet (TLP):
    - Memory Read/Write
    - I/O Read/Write
    - Configuration Read/Write
    - Message (interrupts, errors, power management)
    
  Data Link Layer Packet (DLLP):
    - Acknowledgment (ACK/NAK)
    - Flow control credits
    - Power management
    
  Physical Layer:
    - Serialization, encoding, electrical signaling
    - Lane bonding (x1, x2, x4, x8, x16)
    - Training and equalization
` + "```" + `

**Emerging Memory Technologies:**
` + "```" + `
CXL Memory Expansion:
  Use PCIe 5.0/6.0 physical layer for memory
  CXL.mem protocol: CPU treats CXL device as extra DRAM
  
  ┌──────────────┐     ┌─────────────────────┐
  │   CPU         │     │ CXL Memory Expander │
  │               │PCIe/│ 256 GB-2 TB DRAM    │
  │  Local DRAM   │CXL  │ Lower cost/GB       │
  │  (fast, small)│─────│ Higher latency       │
  └──────────────┘     └─────────────────────┘
  
  Latencies:
    Local DDR5: ~80 ns
    CXL memory: ~150-300 ns (added PCIe latency)
  
  Use case: Double memory capacity without buying bigger server
  NUMA-like: OS can place cold pages on CXL memory

Processing-in-Memory:
  Samsung HBM-PIM: SIMD engines inside HBM layers
  UPMEM PIM-DIMM: 128 DPU cores per DIMM
  
  Advantage: Process data where it lives
    Traditional: Data → (slow bus) → CPU → (slow bus) → Memory
    PIM:         Data → PIM engine → Result (stays in memory)
  
  Good for: Database scans, graph analytics, simple reductions

Persistent Memory (Intel Optane):
  3D XPoint technology (discontinued but influential)
  Byte-addressable like DRAM, persistent like SSD
  Latency: ~300 ns (between DRAM ~80ns and SSD ~20µs)
  
  Legacy and influence:
  - CXL memory devices inheriting the concept
  - New persistent memory protocols in CXL 3.0
  - Software ecosystem (pmemkv, PMDK) lives on
` + "```" + ``,
					CodeExamples: `// Memory and interconnect bandwidth modeling
package main

import (
    "fmt"
    "math"
)

// Memory technology specification
type MemorySpec struct {
    name       string
    dataRate   int     // MT/s
    busWidth   int     // bits
    channels   int
    encoding   float64 // efficiency (1.0 = no overhead)
    latencyNS  float64 // access latency in nanoseconds
}

func (m MemorySpec) BandwidthGBs() float64 {
    return float64(m.dataRate) * float64(m.busWidth) / 8 * float64(m.channels) * m.encoding / 1000
}

// PCIe specification
type PCIeSpec struct {
    gen      string
    rateGTs  float64 // Gigatransfers per second per lane
    encoding float64 // efficiency
    lanes    int
}

func (p PCIeSpec) BandwidthGBs() float64 {
    return p.rateGTs * float64(p.lanes) * p.encoding / 8 // bidirectional per direction
}

// Memory hierarchy with bandwidth and latency
type MemoryNode struct {
    name       string
    capacityGB float64
    bwGBs      float64
    latencyNS  float64
}

func main() {
    fmt.Println("=== DDR Memory Bandwidth Comparison ===")
    
    ddrSpecs := []MemorySpec{
        {"DDR4-3200 (dual)", 3200, 64, 2, 1.0, 85},
        {"DDR5-5600 (dual)", 5600, 64, 2, 1.0, 80},
        {"DDR5-5600 (quad server)", 5600, 64, 4, 1.0, 80},
        {"DDR5-8400 (dual)", 8400, 64, 2, 1.0, 78},
        {"LPDDR5X-8533 (Apple M3)", 8533, 64, 2, 1.0, 75},
    }
    
    fmt.Printf("%-30s │ BW (GB/s) │ Latency\n", "Memory Configuration")
    fmt.Println("───────────────────────────────┼───────────┼────────")
    for _, spec := range ddrSpecs {
        fmt.Printf("%-30s │ %7.1f   │ %4.0f ns\n", spec.name, spec.BandwidthGBs(), spec.latencyNS)
    }
    
    fmt.Println("\n=== HBM Bandwidth Comparison ===")
    hbmSpecs := []MemorySpec{
        {"HBM2 (4 stacks)", 2000, 1024, 4, 1.0, 100},
        {"HBM2E (4 stacks)", 3600, 1024, 4, 1.0, 95},
        {"HBM3 (6 stacks, H100)", 5200, 1024, 6, 1.0, 90},
        {"HBM3E (8 stacks, B200)", 9200, 1024, 8, 1.0, 85},
    }
    
    fmt.Printf("%-30s │ BW (TB/s) │ Latency\n", "HBM Configuration")
    fmt.Println("───────────────────────────────┼───────────┼────────")
    for _, spec := range hbmSpecs {
        bw := spec.BandwidthGBs()
        fmt.Printf("%-30s │ %7.2f   │ %4.0f ns\n", spec.name, bw/1000, spec.latencyNS)
    }
    
    fmt.Println("\n=== PCIe Bandwidth Evolution ===")
    pcieSpecs := []PCIeSpec{
        {"PCIe 3.0", 8.0, 128.0 / 130.0, 16},
        {"PCIe 4.0", 16.0, 128.0 / 130.0, 16},
        {"PCIe 5.0", 32.0, 128.0 / 130.0, 16},
        {"PCIe 6.0", 64.0, 128.0 / 130.0, 16},
        {"PCIe 7.0", 128.0, 128.0 / 130.0, 16},
    }
    
    fmt.Printf("%-12s │ Rate/Lane │ x16 BW (each dir)\n", "Generation")
    fmt.Println("─────────────┼───────────┼──────────────────")
    for _, spec := range pcieSpecs {
        fmt.Printf("%-12s │ %5.0f GT/s│ %6.1f GB/s\n", spec.gen, spec.rateGTs, spec.BandwidthGBs())
    }
    
    // Bandwidth gap analysis
    fmt.Println("\n=== CPU-Memory Bandwidth Gap ===")
    type YearData struct {
        year      int
        cpuGFLOPS float64
        memGBs    float64
    }
    
    years := []YearData{
        {2000, 2, 3.2},
        {2005, 20, 12.8},
        {2010, 100, 25.6},
        {2015, 500, 68},
        {2020, 2000, 100},
        {2024, 5000, 180},
    }
    
    fmt.Printf("Year │ CPU GFLOPS │ Mem BW (GB/s) │ Bytes/FLOP │ Bound at OI\n")
    fmt.Println("─────┼────────────┼───────────────┼────────────┼────────────")
    for _, y := range years {
        bytePerFlop := y.memGBs / y.cpuGFLOPS
        ridgeOI := y.cpuGFLOPS / y.memGBs
        fmt.Printf("%d │ %8.0f   │ %10.0f    │ %8.2f    │ OI > %.1f\n",
            y.year, y.cpuGFLOPS, y.memGBs, bytePerFlop, ridgeOI)
    }
    fmt.Println("\nTrend: Compute grows faster than memory bandwidth")
    fmt.Println("→ Higher operational intensity needed to be compute-bound")
    fmt.Println("→ Drives demand for HBM and CXL memory expansion")
    
    // CXL memory tiering simulation
    fmt.Println("\n=== CXL Memory Tiering ===")
    hierarchy := []MemoryNode{
        {"L1 Cache", 0.000032, 1000, 1},
        {"L2 Cache", 0.001, 500, 4},
        {"L3 Cache", 0.032, 300, 10},
        {"Local DDR5", 64, 180, 80},
        {"CXL Tier 1", 256, 60, 200},
        {"CXL Tier 2 (pooled)", 2048, 30, 400},
        {"NVMe SSD", 8192, 7, 20000},
    }
    
    fmt.Printf("%-22s │ Capacity │ BW (GB/s)│ Latency │ $/GB (est)\n", "Tier")
    fmt.Println("───────────────────────┼──────────┼──────────┼─────────┼──────────")
    for _, node := range hierarchy {
        capStr := ""
        if node.capacityGB < 1 {
            capStr = fmt.Sprintf("%.0f KB", node.capacityGB*1024*1024)
        } else if node.capacityGB < 100 {
            capStr = fmt.Sprintf("%.0f MB", node.capacityGB*1024)
        } else {
            capStr = fmt.Sprintf("%.0f GB", node.capacityGB)
        }
        
        costPerGB := 1000 / math.Max(node.capacityGB, 0.001) // Very rough estimate
        if costPerGB > 10000 { costPerGB = 0 } // Skip for caches
        
        latStr := fmt.Sprintf("%6.0f ns", node.latencyNS)
        if node.latencyNS >= 1000 {
            latStr = fmt.Sprintf("%5.0f µs", node.latencyNS/1000)
        }
        
        fmt.Printf("%-22s │ %8s │ %6.0f   │ %s │",
            node.name, capStr, node.bwGBs, latStr)
        if costPerGB > 0 && costPerGB < 10000 {
            fmt.Printf(" $%.0f\n", costPerGB)
        } else {
            fmt.Println(" -")
        }
    }
}`,
				},
			},
		},
	})
}
