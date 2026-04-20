package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2828,
			Title:       "I/O Systems and Bus Architectures",
			Description: "Deep exploration of I/O systems: interrupt handling, DMA, bus protocols (PCIe, USB, NVMe), display interfaces, and storage controllers.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "Interrupt Handling and DMA",
					Content: `I/O devices need to communicate with the CPU. The two primary mechanisms are interrupts (event-driven notification) and DMA (direct memory access for bulk transfers).

**Polling vs Interrupts:**
` + "```" + `
Polling (Programmed I/O):
    while (device.status != READY) {
        // CPU spins, wasting cycles
    }
    data = device.read()

    Pros: Simple, predictable timing
    Cons: CPU wastes time polling, especially if device is slow
          A disk taking 10ms to read wastes ~30 million CPU cycles!

Interrupts:
    1. CPU initiates I/O operation
    2. CPU continues executing other instructions
    3. When device is ready, it asserts interrupt signal
    4. CPU:
       a. Saves current state (PC, registers) to stack
       b. Looks up handler in Interrupt Vector Table (IVT)
       c. Jumps to interrupt handler
       d. Handler processes I/O data
       e. Restores state, returns to interrupted program

    Pros: CPU does useful work while waiting
    Cons: Interrupt overhead (~1-10 µs per interrupt)
          High interrupt rates can overwhelm CPU
` + "```" + `

**Interrupt Controller Hardware:**
` + "```" + `
Advanced Programmable Interrupt Controller (APIC):

   Device IRQs:
   [Timer]──IRQ0──┐
   [Keyboard]────IRQ1──┐
   [COM2]────────IRQ3──┤
   [COM1]────────IRQ4──┤     ┌──────────┐
   [LPT]────────IRQ5──┼────→│  Local   │
   [Floppy]──────IRQ6──┤     │  APIC    │──→ CPU Core
   [IDE1]────────IRQ14─┤     │ (per core)│
   [IDE2]────────IRQ15─┘     └──────────┘
                                   ↑
   Network ──→ [I/O APIC] ────────┘
   USB    ──→ (routes to appropriate Local APIC)
   NVMe   ──→

Modern x86 Interrupt Delivery:
   MSI (Message Signaled Interrupts):
   - Device writes to a special memory address
   - Message contains interrupt vector number
   - No shared IRQ lines → no IRQ conflicts
   
   MSI-X:
   - Up to 2048 interrupt vectors per device
   - Each vector can target a different CPU core
   - Critical for NVMe, high-speed networking

Interrupt Priorities:
   ┌─────────────────────────────────────────┐
   │ NMI (Non-Maskable): Hardware failure     │ ← Highest
   │ Machine Check: CPU error                 │
   │ Device Interrupts: I/O devices           │
   │ Timer Interrupt: Scheduling tick         │
   │ Software Interrupts: System calls        │ ← Lowest
   └─────────────────────────────────────────┘

   Nested interrupts: Higher priority can preempt lower
   Interrupt masking: CPU can temporarily disable interrupts
     cli / sti instructions (x86)
     Critical sections must be short to avoid latency
` + "```" + `

**DMA (Direct Memory Access):**
` + "```" + `
Without DMA (Programmed I/O):
    CPU reads byte from device register → writes to memory
    Repeat for every byte
    CPU is 100% busy during transfer!

    CPU → [read device] → [write memory] → repeat × 1,000,000
    Transferring 1MB at 1 byte/cycle = millions of wasted CPU cycles

With DMA:
    1. CPU programs DMA controller:
       - Source address (device or memory)
       - Destination address (memory or device)
       - Transfer count
       - Direction (device→memory or memory→device)
    2. DMA controller takes over the bus
    3. DMA transfers data directly between device and memory
    4. CPU is FREE to do other work
    5. When done, DMA controller interrupts CPU

    CPU: [program DMA] → [do other work...] → [handle completion interrupt]
    DMA:                [transfer data to/from memory autonomously]

DMA Modes:
    Cycle stealing: DMA uses bus for one transfer, then gives back
      → Minimal CPU interference, slower transfer
    
    Burst mode: DMA holds bus for entire transfer
      → Faster transfer, but CPU blocked from memory
    
    Block mode: DMA transfers a block, then releases
      → Compromise between cycle stealing and burst

Scatter-Gather DMA:
    Traditional DMA: contiguous memory regions only
    Scatter-gather: DMA controller reads a linked list of
      (address, length) pairs
    → Can transfer to/from non-contiguous memory pages
    → Essential for virtual memory systems
    → Used by all modern devices (NVMe, network cards, GPUs)
` + "```" + `

**IOMMU (I/O Memory Management Unit):**
` + "```" + `
Problem: DMA devices access PHYSICAL memory addresses
         But OS uses VIRTUAL addresses
         DMA can access ANY physical memory → security risk!

IOMMU solution:
    Device → IOMMU → Physical Memory
    
    IOMMU translates device (I/O) virtual addresses to physical
    Provides:
    1. Address translation (virtual → physical for DMA)
    2. Memory protection (device can only access permitted pages)
    3. Interrupt remapping (prevent malicious interrupt injection)
    
    Intel: VT-d (Virtualization Technology for Directed I/O)
    AMD:   AMD-Vi

Critical for:
    - Device passthrough to VMs
    - Protection from buggy/malicious devices
    - PCIe peer-to-peer without CPU involvement
` + "```" + ``,
					CodeExamples: `// Interrupt controller and DMA simulation
package main

import "fmt"

// Interrupt priorities
const (
    PRI_NMI     = 0  // Highest
    PRI_TIMER   = 1
    PRI_DISK    = 2
    PRI_NETWORK = 3
    PRI_KBD     = 4
    PRI_SOFTWARE = 5 // Lowest
)

type Interrupt struct {
    vector   int
    priority int
    source   string
    handler  func()
}

type InterruptController struct {
    pending    []Interrupt
    ivt        map[int]func()  // Interrupt Vector Table
    maskLevel  int             // Interrupts below this priority are masked
    handling   bool
    currentPri int
    stats      map[string]int
}

func NewInterruptController() *InterruptController {
    return &InterruptController{
        ivt:       make(map[int]func()),
        maskLevel: 6,
        stats:     make(map[string]int),
    }
}

func (ic *InterruptController) RegisterHandler(vector int, handler func()) {
    ic.ivt[vector] = handler
}

func (ic *InterruptController) RaiseInterrupt(irq Interrupt) {
    if irq.priority >= ic.maskLevel {
        return // Masked
    }
    ic.pending = append(ic.pending, irq)
    ic.stats[irq.source]++
}

func (ic *InterruptController) ProcessInterrupts() {
    if len(ic.pending) == 0 { return }

    // Find highest priority pending interrupt
    bestIdx := 0
    for i, irq := range ic.pending {
        if irq.priority < ic.pending[bestIdx].priority {
            bestIdx = i
        }
    }

    irq := ic.pending[bestIdx]
    ic.pending = append(ic.pending[:bestIdx], ic.pending[bestIdx+1:]...)

    // Can preempt current handler if higher priority
    if ic.handling && irq.priority >= ic.currentPri {
        ic.pending = append(ic.pending, irq) // Re-queue
        return
    }

    // Handle interrupt
    oldPri := ic.currentPri
    wasHandling := ic.handling
    ic.handling = true
    ic.currentPri = irq.priority

    fmt.Printf("  → Handling IRQ %d (%s) priority=%d\n",
        irq.vector, irq.source, irq.priority)
    
    if handler, ok := ic.ivt[irq.vector]; ok {
        handler()
    }

    ic.handling = wasHandling
    ic.currentPri = oldPri
}

// DMA Controller
type DMAChannel struct {
    srcAddr    uint64
    dstAddr    uint64
    count      int
    direction  string // "read" or "write"
    active     bool
    transferred int
    mode       string // "burst" or "cycle-stealing"
}

type DMAController struct {
    channels [4]DMAChannel
    memory   []byte
    ic       *InterruptController
}

func NewDMAController(memSize int, ic *InterruptController) *DMAController {
    return &DMAController{
        memory: make([]byte, memSize),
        ic:     ic,
    }
}

func (dma *DMAController) ProgramChannel(ch int, src, dst uint64, count int, mode string) {
    dma.channels[ch] = DMAChannel{
        srcAddr:   src,
        dstAddr:   dst,
        count:     count,
        active:    true,
        mode:      mode,
        direction: "read",
    }
    fmt.Printf("DMA ch%d: programmed %d bytes from 0x%X to 0x%X (%s mode)\n",
        ch, count, src, dst, mode)
}

func (dma *DMAController) Tick() bool {
    anyActive := false
    for i := range dma.channels {
        ch := &dma.channels[i]
        if !ch.active { continue }
        anyActive = true

        // Transfer data
        bytesPerTick := 64 // Cache line size
        if ch.mode == "cycle-stealing" {
            bytesPerTick = 8 // One word at a time
        }

        remaining := ch.count - ch.transferred
        if bytesPerTick > remaining {
            bytesPerTick = remaining
        }

        ch.transferred += bytesPerTick

        if ch.transferred >= ch.count {
            ch.active = false
            fmt.Printf("DMA ch%d: transfer complete (%d bytes)\n",
                i, ch.count)
            // Raise completion interrupt
            dma.ic.RaiseInterrupt(Interrupt{
                vector:   32 + i,
                priority: PRI_DISK,
                source:   fmt.Sprintf("DMA-ch%d", i),
            })
        }
    }
    return anyActive
}

// Scatter-Gather DMA descriptor
type SGEntry struct {
    addr   uint64
    length int
}

type ScatterGatherDMA struct {
    descriptors []SGEntry
    totalBytes  int
}

func NewSGDMA(entries []SGEntry) *ScatterGatherDMA {
    total := 0
    for _, e := range entries { total += e.length }
    return &ScatterGatherDMA{descriptors: entries, totalBytes: total}
}

func main() {
    ic := NewInterruptController()
    
    // Register interrupt handlers
    ic.RegisterHandler(0, func() { fmt.Println("    Timer tick") })
    ic.RegisterHandler(1, func() { fmt.Println("    Keyboard: key pressed") })
    ic.RegisterHandler(14, func() { fmt.Println("    Disk: I/O complete") })
    ic.RegisterHandler(32, func() { fmt.Println("    DMA ch0: transfer done") })

    fmt.Println("=== Interrupt Controller Demo ===")
    
    // Raise multiple interrupts at different priorities
    ic.RaiseInterrupt(Interrupt{1, PRI_KBD, "Keyboard", nil})
    ic.RaiseInterrupt(Interrupt{14, PRI_DISK, "Disk", nil})
    ic.RaiseInterrupt(Interrupt{0, PRI_TIMER, "Timer", nil})

    // Process in priority order
    for len(ic.pending) > 0 {
        ic.ProcessInterrupts()
    }

    fmt.Printf("\nInterrupt stats: %v\n", ic.stats)

    // DMA demo
    fmt.Println("\n=== DMA Controller Demo ===")
    dma := NewDMAController(65536, ic)
    
    // Program a burst DMA transfer (e.g., disk read)
    dma.ProgramChannel(0, 0x1000, 0x8000, 4096, "burst")
    
    cycle := 0
    for dma.Tick() {
        cycle++
    }
    fmt.Printf("Burst transfer completed in %d bus cycles\n", cycle)
    
    // Process DMA completion interrupt
    ic.ProcessInterrupts()

    // Scatter-gather demo
    fmt.Println("\n=== Scatter-Gather DMA ===")
    sg := NewSGDMA([]SGEntry{
        {0x1000, 4096},
        {0x5000, 2048},
        {0xA000, 8192},
        {0xF000, 512},
    })
    fmt.Printf("SG DMA: %d descriptors, %d total bytes\n",
        len(sg.descriptors), sg.totalBytes)
    for i, e := range sg.descriptors {
        fmt.Printf("  Descriptor %d: addr=0x%X, len=%d\n", i, e.addr, e.length)
    }
}`,
				},
				{
					Title: "PCIe Architecture",
					Content: `PCI Express is the dominant interconnect for high-performance I/O devices. Understanding its architecture is essential for modern computer systems.

**PCIe vs PCI:**
` + "```" + `
PCI (legacy):
  - Shared parallel bus (32/64 bit)
  - Half-duplex: data flows one direction at a time
  - All devices share bandwidth (133-533 MB/s total)
  - Bus arbitration needed

PCIe:
  - Point-to-point serial links
  - Full-duplex: simultaneous send and receive
  - Each device gets dedicated bandwidth
  - Packet-switched (like a network)

PCIe Link:
    ┌──────┐  Lane 0 TX →  ┌──────┐
    │      │  Lane 0 RX ←  │      │
    │ Root │  Lane 1 TX →  │Device│
    │  or  │  Lane 1 RX ←  │      │
    │Switch│      ...       │      │
    │      │  Lane N TX →  │      │
    │      │  Lane N RX ←  │      │
    └──────┘               └──────┘

Each lane: 1 differential pair per direction (TX and RX)
Link width: x1, x2, x4, x8, x16 lanes
Higher width = more bandwidth
` + "```" + `

**PCIe Generations:**
` + "```" + `
Gen │ Per Lane   │ x1       │ x4       │ x8       │ x16      │ Encoding │ Year
1.0 │ 2.5 GT/s   │ 250 MB/s │ 1 GB/s   │ 2 GB/s   │ 4 GB/s   │ 8b/10b   │ 2003
2.0 │ 5 GT/s     │ 500 MB/s │ 2 GB/s   │ 4 GB/s   │ 8 GB/s   │ 8b/10b   │ 2007
3.0 │ 8 GT/s     │ 985 MB/s │ 3.9 GB/s │ 7.9 GB/s │ 15.8 GB/s│ 128b/130b│ 2010
4.0 │ 16 GT/s    │ 1.97 GB/s│ 7.9 GB/s │ 15.8 GB/s│ 31.5 GB/s│ 128b/130b│ 2017
5.0 │ 32 GT/s    │ 3.94 GB/s│ 15.8 GB/s│ 31.5 GB/s│ 63 GB/s  │ 128b/130b│ 2019
6.0 │ 64 GT/s    │ 7.56 GB/s│ 30.2 GB/s│ 60.5 GB/s│ 121 GB/s │ PAM4+FEC │ 2022

GT/s = Giga-transfers per second
8b/10b: 20% encoding overhead (8 data bits per 10 transmitted)
128b/130b: 1.5% overhead
PAM4: 4-level signaling (2 bits per symbol) — allows higher data rate

Doubling bandwidth each generation:
  Gen 1→2: Double clock    (2.5→5 GT/s)
  Gen 2→3: Double clock    (5→8 GT/s, with better encoding)
  Gen 3→4: Double clock    (8→16 GT/s)
  Gen 4→5: Double clock    (16→32 GT/s)
  Gen 5→6: PAM4 signaling (same symbol rate, 2x bits/symbol)
` + "```" + `

**PCIe Topology:**
` + "```" + `
                    ┌──────────┐
                    │   CPU    │
                    │ Root     │
                    │ Complex  │
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │  Switch  │
                    └─┬──┬──┬──┘
                      │  │  │
                 ┌────┘  │  └────┐
                 │       │       │
            ┌────┴──┐┌───┴──┐┌───┴──┐
            │  GPU  ││ NVMe ││ NIC  │
            │(x16)  ││ (x4) ││ (x4) │
            └───────┘└──────┘└──────┘

Root Complex: CPU's PCIe interface
Switch: Routes packets between ports (like network switch)
Endpoint: Device (GPU, NVMe, network card)

PCIe uses packet-based communication:
    Transaction Layer Packet (TLP):
    ┌────────┬─────────┬──────────┬──────┐
    │ Header │ Address  │ Data     │ ECRC │
    │ (3-4DW)│ (opt)    │ (0-1024B)│      │
    └────────┴─────────┴──────────┴──────┘

TLP Types:
    Memory Read/Write: DMA transfers
    I/O Read/Write: Legacy PIO
    Configuration Read/Write: Device setup
    Message: Interrupts, errors, power management
    Completion: Response to read requests
` + "```" + `

**NVMe Over PCIe:**
` + "```" + `
NVMe (Non-Volatile Memory Express):
    Protocol designed specifically for SSDs over PCIe
    Replaces AHCI (designed for spinning disks)

Key advantages over AHCI:
    AHCI: 1 command queue, 32 entries deep
    NVMe: 65,535 queues, 65,536 entries each!
    
    AHCI: ~6 µs per I/O (many register reads)
    NVMe: ~2 µs per I/O (memory-mapped doorbell)

NVMe Command Flow:
    1. Host writes command to Submission Queue (in host memory)
    2. Host writes new tail to SQ Doorbell (MMIO register)
    3. Controller fetches command via DMA
    4. Controller processes command (read/write flash)
    5. Controller writes completion entry to Completion Queue
    6. Controller raises MSI-X interrupt
    7. Host processes completion, writes CQ Doorbell

NVMe Queue Pairs:
    Each CPU core can have its own SQ/CQ pair
    → No locking needed between cores
    → Scales linearly with core count
    
    Typical modern NVMe SSD:
    - Sequential read: 7 GB/s (PCIe 4.0 x4)
    - Random 4K read: 1,000,000 IOPS
    - Latency: 10-20 µs
` + "```" + ``,
					CodeExamples: `// PCIe and NVMe concepts simulation
package main

import "fmt"

// PCIe bandwidth calculator
type PCIeConfig struct {
    gen      int
    lanes    int
    encoding string
}

func (c PCIeConfig) RawRate() float64 {
    rates := map[int]float64{
        1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0,
    }
    return rates[c.gen]
}

func (c PCIeConfig) EfficiencyPercent() float64 {
    switch c.encoding {
    case "8b/10b":   return 80.0
    case "128b/130b": return 98.46
    case "PAM4+FEC": return 94.2  // ~242/256 with FEC overhead
    }
    return 100.0
}

func (c PCIeConfig) BandwidthGBs() float64 {
    rawGbps := c.RawRate() * float64(c.lanes)
    effectiveGbps := rawGbps * c.EfficiencyPercent() / 100.0
    return effectiveGbps / 8.0 // Convert Gbps to GB/s
}

// NVMe Queue simulation
type NVMeCommand struct {
    opcode  uint8
    nsid    uint32  // Namespace ID
    lba     uint64  // Logical Block Address
    nlb     uint16  // Number of Logical Blocks
    prp1    uint64  // Physical Region Page 1
}

type NVMeCompletion struct {
    sqID    uint16
    sqHead  uint16
    cmdID   uint16
    status  uint16
}

type NVMeQueuePair struct {
    sqID     int
    sqDepth  int
    cqDepth  int
    sq       []NVMeCommand
    cq       []NVMeCompletion
    sqTail   int
    sqHead   int
    cqHead   int
    cqTail   int
    submitted int
    completed int
}

func NewNVMeQueuePair(sqID, depth int) *NVMeQueuePair {
    return &NVMeQueuePair{
        sqID:    sqID,
        sqDepth: depth,
        cqDepth: depth,
        sq:      make([]NVMeCommand, depth),
        cq:      make([]NVMeCompletion, depth),
    }
}

func (qp *NVMeQueuePair) Submit(cmd NVMeCommand) bool {
    nextTail := (qp.sqTail + 1) % qp.sqDepth
    if nextTail == qp.sqHead {
        return false // Queue full
    }
    qp.sq[qp.sqTail] = cmd
    qp.sqTail = nextTail
    qp.submitted++
    return true
}

func (qp *NVMeQueuePair) Complete() *NVMeCompletion {
    if qp.cqHead == qp.cqTail {
        return nil // No completions
    }
    c := qp.cq[qp.cqHead]
    qp.cqHead = (qp.cqHead + 1) % qp.cqDepth
    qp.completed++
    return &c
}

// Simulate controller processing
func (qp *NVMeQueuePair) ProcessOne() {
    if qp.sqHead == qp.sqTail {
        return // Nothing to process
    }
    // "Process" the command
    qp.sqHead = (qp.sqHead + 1) % qp.sqDepth
    // Post completion
    nextTail := (qp.cqTail + 1) % qp.cqDepth
    qp.cq[qp.cqTail] = NVMeCompletion{
        sqID:   uint16(qp.sqID),
        sqHead: uint16(qp.sqHead),
        status: 0, // Success
    }
    qp.cqTail = nextTail
}

// DMA engine with IOMMU
type IOVAMapping struct {
    iova    uint64 // I/O Virtual Address
    pa      uint64 // Physical Address
    size    uint64
    perm    string // "r", "w", "rw"
}

type IOMMU struct {
    mappings []IOVAMapping
}

func (iommu *IOMMU) AddMapping(iova, pa, size uint64, perm string) {
    iommu.mappings = append(iommu.mappings, IOVAMapping{iova, pa, size, perm})
}

func (iommu *IOMMU) Translate(iova uint64, write bool) (uint64, error) {
    for _, m := range iommu.mappings {
        if iova >= m.iova && iova < m.iova+m.size {
            if write && m.perm == "r" {
                return 0, fmt.Errorf("IOMMU: write permission denied for IOVA 0x%X", iova)
            }
            offset := iova - m.iova
            return m.pa + offset, nil
        }
    }
    return 0, fmt.Errorf("IOMMU: no mapping for IOVA 0x%X", iova)
}

func main() {
    // PCIe bandwidth comparison
    fmt.Println("=== PCIe Bandwidth Comparison ===")
    configs := []PCIeConfig{
        {3, 4, "128b/130b"},   // Typical NVMe SSD (Gen 3)
        {4, 4, "128b/130b"},   // Modern NVMe SSD
        {4, 16, "128b/130b"},  // GPU
        {5, 16, "128b/130b"},  // High-end GPU
        {5, 4, "128b/130b"},   // Gen 5 NVMe
    }
    for _, c := range configs {
        fmt.Printf("  PCIe Gen %d x%d: %.1f GB/s (%.1f%% efficiency, %s)\n",
            c.gen, c.lanes, c.BandwidthGBs(), c.EfficiencyPercent(), c.encoding)
    }

    // NVMe queue simulation
    fmt.Println("\n=== NVMe Queue Pair Simulation ===")
    qp := NewNVMeQueuePair(1, 1024)

    // Submit a batch of I/O commands
    for i := 0; i < 32; i++ {
        cmd := NVMeCommand{
            opcode: 0x02, // Read
            nsid:   1,
            lba:    uint64(i * 8),
            nlb:    7, // 8 blocks (4KB)
        }
        if qp.Submit(cmd) {
            if i < 3 {
                fmt.Printf("  Submitted: Read LBA %d-%d\n", cmd.lba, cmd.lba+uint64(cmd.nlb))
            }
        }
    }
    fmt.Printf("  ... submitted %d commands total\n", qp.submitted)

    // Simulate controller processing
    for i := 0; i < 32; i++ {
        qp.ProcessOne()
    }

    // Process completions
    completions := 0
    for {
        c := qp.Complete()
        if c == nil { break }
        completions++
    }
    fmt.Printf("  Completed: %d commands\n", completions)

    // IOMMU demo
    fmt.Println("\n=== IOMMU Translation ===")
    iommu := &IOMMU{}
    iommu.AddMapping(0x00000000, 0x100000000, 0x10000, "rw") // 64KB DMA buffer
    iommu.AddMapping(0x00010000, 0x200000000, 0x1000, "r")   // Read-only region

    tests := []struct{
        iova  uint64
        write bool
    }{
        {0x00000100, false},  // Valid read
        {0x00000200, true},   // Valid write
        {0x00010800, false},  // Valid read from read-only
        {0x00010800, true},   // Write to read-only → denied!
        {0x00090000, false},  // Unmapped → error!
    }

    for _, t := range tests {
        op := "READ"
        if t.write { op = "WRITE" }
        pa, err := iommu.Translate(t.iova, t.write)
        if err != nil {
            fmt.Printf("  %s IOVA 0x%X: %s\n", op, t.iova, err)
        } else {
            fmt.Printf("  %s IOVA 0x%X → PA 0x%X\n", op, t.iova, pa)
        }
    }
}`,
				},
				{
					Title: "USB and Display Interfaces",
					Content: `USB is the most ubiquitous I/O standard for peripheral devices, while display interfaces (DisplayPort, HDMI) handle the demanding task of transmitting high-resolution video.

**USB Architecture:**
` + "```" + `
USB Generations:
Gen        │ Speed       │ Encoding  │ Connector        │ Year
USB 1.1    │ 12 Mbps     │ NRZI      │ Type-A/B         │ 1998
USB 2.0    │ 480 Mbps    │ NRZI      │ Type-A/B/Mini/Micro│ 2000
USB 3.0    │ 5 Gbps      │ 8b/10b    │ Type-A (blue)/C  │ 2008
USB 3.1    │ 10 Gbps     │ 128b/132b │ Type-A/C         │ 2013
USB 3.2    │ 20 Gbps     │ 128b/132b │ Type-C (2 lanes) │ 2017
USB4       │ 40 Gbps     │ 128b/132b │ Type-C            │ 2019
USB4v2     │ 80 Gbps     │ PAM3      │ Type-C            │ 2022

USB protocol stack:
    Application Layer
    ──────────────────
    Class Driver (HID, Mass Storage, Audio, Video...)
    USB Driver (URB management)
    Host Controller Driver (xHCI)
    ──────────────────
    Host Controller Hardware
    ──────────────────
    Physical Layer (cables, connectors)

USB transfer types:
    Control:     Device setup/configuration, guaranteed delivery
    Bulk:        Large data (mass storage), best-effort bandwidth
    Interrupt:   Small, periodic (keyboards, mice), guaranteed latency
    Isochronous: Streaming (audio, video), guaranteed bandwidth, no retransmit
` + "```" + `

**USB Type-C and Power Delivery:**
` + "```" + `
Type-C Connector:
    24 pins, reversible (no wrong way to plug in)
    Carries: USB data, power, video (DisplayPort alt-mode), Thunderbolt

Pin layout (simplified):
    ┌──────────────────────────────────────┐
    │ GND TX1+ TX1- VBus CC1 D+ D- SBU1   │ ← Top row
    │ GND RX2+ RX2- VBus CC2 D+ D- SBU2   │ ← Bottom row (flipped)
    │ GND TX2+ TX2- VBus │ GND RX1+ RX1-  │
    └──────────────────────────────────────┘

USB Power Delivery:
    Standard USB: 5V @ 0.9A = 4.5W
    PD profiles:
      5V  @ 3A  = 15W
      9V  @ 3A  = 27W
      15V @ 3A  = 45W
      20V @ 3A  = 60W
      20V @ 5A  = 100W (needs e-marked cable)
      48V @ 5A  = 240W (Extended Power Range, USB PD 3.1)

Alternate Modes:
    DisplayPort: Up to 4K@120Hz or 8K@60Hz
    Thunderbolt: 40 Gbps (TB3) or 80 Gbps (TB4/USB4)
    HDMI: HDMI 2.1 through Type-C
` + "```" + `

**Display Interfaces:**
` + "```" + `
DisplayPort (DP):
    Main Link: 1, 2, or 4 lanes
    DP 1.4:  8.1 Gbps/lane × 4 lanes = 32.4 Gbps → 4K@120Hz, 8K@30Hz
    DP 2.0: 13.5 Gbps/lane × 4 lanes = 54 Gbps → 8K@60Hz, 4K@240Hz
    DP 2.1: 13.5 Gbps/lane, improved cable specs

    Display Stream Compression (DSC): visually lossless 3:1 compression
    → Enables 8K@60Hz over DP 1.4 with DSC
    
    Adaptive-Sync (FreeSync/G-Sync Compatible):
    Variable refresh rate — display syncs to GPU frame rate
    Eliminates tearing and judder

HDMI:
    HDMI 2.0: 18 Gbps → 4K@60Hz
    HDMI 2.1: 48 Gbps → 4K@120Hz, 8K@60Hz
    
    HDMI uses TMDS encoding (up to 2.0) or FRL (2.1+)
    Carries audio (up to 32 channels) + video + CEC control

Bandwidth Requirements:
    Resolution    │ Refresh │ Color  │ Raw Bandwidth
    1920×1080     │ 60 Hz   │ 24-bit │ 3.0 Gbps
    2560×1440     │ 144 Hz  │ 24-bit │ 10.6 Gbps
    3840×2160     │ 60 Hz   │ 24-bit │ 11.9 Gbps
    3840×2160     │ 120 Hz  │ 30-bit │ 29.7 Gbps
    7680×4320     │ 60 Hz   │ 24-bit │ 47.8 Gbps

Formula: Width × Height × RefreshRate × BitsPerPixel × (1 + blanking ~6%)
` + "```" + ``,
					CodeExamples: `// USB and display bandwidth calculations
package main

import "fmt"

// USB specification
type USBSpec struct {
    name     string
    speedMbps float64
    encoding  string
    maxPowerW float64
}

var usbSpecs = []USBSpec{
    {"USB 1.1", 12, "NRZI", 2.5},
    {"USB 2.0", 480, "NRZI", 2.5},
    {"USB 3.0", 5000, "8b/10b", 4.5},
    {"USB 3.1", 10000, "128b/132b", 100},
    {"USB 3.2×2", 20000, "128b/132b", 100},
    {"USB4", 40000, "128b/132b", 100},
    {"USB4v2", 80000, "PAM3", 240},
}

func (s USBSpec) EffectiveMBps() float64 {
    efficiency := 1.0
    switch s.encoding {
    case "NRZI":       efficiency = 1.0   // Bit-stuffing overhead ~1%
    case "8b/10b":     efficiency = 0.8
    case "128b/132b":  efficiency = 0.9697
    case "PAM3":       efficiency = 0.95  // Approximate with FEC
    }
    return s.speedMbps * efficiency / 8.0 // Convert Mbps to MB/s
}

// Display resolution and bandwidth
type DisplayConfig struct {
    name       string
    width      int
    height     int
    refresh    int
    bitsPerPx  int
}

func (d DisplayConfig) BandwidthGbps() float64 {
    pixels := float64(d.width) * float64(d.height)
    bitsPerFrame := pixels * float64(d.bitsPerPx)
    bitsPerSecond := bitsPerFrame * float64(d.refresh)
    blanking := 1.06 // ~6% blanking overhead
    return bitsPerSecond * blanking / 1e9
}

func (d DisplayConfig) RequiredInterface() string {
    bw := d.BandwidthGbps()
    switch {
    case bw <= 14.4: return "HDMI 2.0 (18 Gbps) or DP 1.2 (17.28 Gbps)"
    case bw <= 25.92: return "DP 1.4 (25.92 Gbps)"
    case bw <= 32.4: return "DP 1.4 (32.4 Gbps with HBR3)"
    case bw <= 48: return "HDMI 2.1 (48 Gbps) or DP 2.0"
    default: return "DP 2.0 (77.4 Gbps) with DSC recommended"
    }
}

// USB transfer time calculator
func transferTime(sizeMB float64, spec USBSpec) float64 {
    mbps := spec.EffectiveMBps()
    return sizeMB / mbps // seconds
}

func main() {
    // USB bandwidth comparison
    fmt.Println("=== USB Specifications ===")
    fmt.Printf("%-12s │ %10s │ %10s │ %8s\n",
        "Version", "Speed", "Effective", "Max Power")
    fmt.Println("─────────────┼────────────┼────────────┼──────────")
    for _, s := range usbSpecs {
        speedStr := fmt.Sprintf("%.0f Mbps", s.speedMbps)
        if s.speedMbps >= 1000 {
            speedStr = fmt.Sprintf("%.0f Gbps", s.speedMbps/1000)
        }
        fmt.Printf("%-12s │ %10s │ %7.0f MB/s│ %5.0fW\n",
            s.name, speedStr, s.EffectiveMBps(), s.maxPowerW)
    }

    // File transfer times
    fmt.Println("\n=== Transfer Time: 10 GB File ===")
    for _, s := range usbSpecs {
        t := transferTime(10*1024, s)
        if t > 60 {
            fmt.Printf("  %-12s: %.0f min %.0f sec\n", s.name, t/60, float64(int(t)%60))
        } else {
            fmt.Printf("  %-12s: %.1f sec\n", s.name, t)
        }
    }

    // Display bandwidth requirements
    fmt.Println("\n=== Display Bandwidth Requirements ===")
    displays := []DisplayConfig{
        {"1080p@60", 1920, 1080, 60, 24},
        {"1080p@144", 1920, 1080, 144, 24},
        {"1440p@144", 2560, 1440, 144, 24},
        {"4K@60", 3840, 2160, 60, 24},
        {"4K@120", 3840, 2160, 120, 30},
        {"4K@240", 3840, 2160, 240, 24},
        {"8K@60", 7680, 4320, 60, 24},
    }

    for _, d := range displays {
        bw := d.BandwidthGbps()
        fmt.Printf("  %-10s: %5.1f Gbps → %s\n",
            d.name, bw, d.RequiredInterface())
    }

    // Thunderbolt bandwidth
    fmt.Println("\n=== Thunderbolt Comparison ===")
    fmt.Println("  TB3:  40 Gbps (PCIe 3.0 x4 + DP 1.2)")
    fmt.Println("  TB4:  40 Gbps (guaranteed, dual 4K or single 8K)")
    fmt.Println("  TB5:  80 Gbps (PCIe 4.0 + DP 2.1)")
    fmt.Printf("\n  TB3 with 4K@60 display (~12 Gbps):\n")
    fmt.Printf("    Remaining for data: ~28 Gbps = %.1f GB/s\n", 28.0/8)
}`,
				},
			},
		},
		{
			ID:          2829,
			Title:       "Parallel Computing Architectures",
			Description: "Explore SIMD, GPU computing, CUDA architecture, parallel programming models, and the fundamental laws governing parallel performance.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "Flynn's Taxonomy and SIMD Processing",
					Content: `Parallel computing architectures are classified by how they handle instruction and data streams. Understanding these categories reveals the tradeoffs in modern processor design.

**Flynn's Taxonomy:**
` + "```" + `
                     Single Data          Multiple Data
                  ┌─────────────────┬─────────────────┐
Single            │                 │                 │
Instruction       │     SISD        │     SIMD        │
                  │ (Traditional    │ (Vector/GPU     │
                  │  sequential)    │  processing)    │
                  ├─────────────────┼─────────────────┤
Multiple          │                 │                 │
Instruction       │     MISD        │     MIMD        │
                  │ (Rare: fault    │ (Multi-core     │
                  │  tolerance)     │  processors)    │
                  └─────────────────┴─────────────────┘

SISD: One instruction, one data element at a time
  → Classic von Neumann machine
  → Example: Single-core ARM without NEON

SIMD: One instruction operates on MULTIPLE data elements
  → Same operation applied to all elements simultaneously
  → Example: Intel SSE/AVX, ARM NEON, GPU warps

MISD: Multiple instructions on same data (rare)
  → Used in fault-tolerant systems (space shuttle, triple modular redundancy)

MIMD: Multiple instructions on multiple data
  → Most general: each core runs different code
  → Example: Multi-core CPU, distributed systems
` + "```" + `

**x86 SIMD Extensions:**
` + "```" + `
Evolution of x86 SIMD:

MMX (1997):    64-bit registers (mm0-mm7), integer only
SSE (1999):    128-bit registers (xmm0-xmm7), single-precision float
SSE2 (2001):   + double-precision, integer in XMM
SSE3 (2004):   + horizontal operations
SSE4 (2007):   + dot product, string operations
AVX (2011):    256-bit registers (ymm0-ymm15), 3-operand instructions
AVX2 (2013):   + integer in 256-bit
AVX-512 (2017): 512-bit registers (zmm0-zmm31), masking, scatter/gather

Register widths and throughput:
  SSE:     128 bits = 4× float or 2× double per instruction
  AVX:     256 bits = 8× float or 4× double per instruction
  AVX-512: 512 bits = 16× float or 8× double per instruction

Example: Adding 16 floats
  Scalar:    16 ADD instructions
  SSE:        4 ADDPS instructions (128-bit, 4 floats each)
  AVX:        2 VADDPS instructions (256-bit, 8 floats each)
  AVX-512:    1 VADDPS instruction (512-bit, 16 floats at once!)

SIMD operations:
  Arithmetic: add, sub, mul, div, sqrt, fma (fused multiply-add)
  Comparison: equal, greater-than → produce mask
  Shuffle:    reorder elements within register
  Blend:      select elements from two registers based on mask
  Gather:     load from non-contiguous memory (AVX2+)
  Scatter:    store to non-contiguous memory (AVX-512)
` + "```" + `

**SIMD Auto-Vectorization:**
` + "```" + `
Modern compilers try to automatically vectorize loops:

// Original scalar code:
for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
}

// Compiler auto-vectorizes to (AVX2):
for (int i = 0; i < N; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);
    __m256 vb = _mm256_load_ps(&b[i]);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_store_ps(&c[i], vc);
}

Conditions for auto-vectorization:
  ✓ Simple loop with known iteration count
  ✓ No loop-carried dependencies
  ✓ Contiguous memory access (ideally aligned)
  ✗ Cannot vectorize if iterations depend on each other
  ✗ Non-contiguous access patterns need gather/scatter
  ✗ Conditionals need masking (AVX-512 makes this easier)

Vectorization hints:
  #pragma omp simd           // OpenMP SIMD directive
  __attribute__((aligned(32)))  // Ensure 32-byte alignment for AVX
  -march=native -O3          // Compiler: use native SIMD instructions
` + "```" + ``,
					CodeExamples: `// SIMD concepts in Go (using standard library)
package main

import (
    "fmt"
    "time"
)

// Simulate SIMD by processing multiple elements
// (Go doesn't expose SIMD intrinsics directly, but the compiler
//  auto-vectorizes many operations)

// Scalar addition
func addScalar(a, b, c []float32) {
    for i := range a {
        c[i] = a[i] + b[i]
    }
}

// "SIMD-style" addition (processes 8 at a time, compiler may vectorize)
func addSIMD8(a, b, c []float32) {
    n := len(a)
    i := 0
    // Process 8 elements at a time (mimics AVX)
    for ; i+8 <= n; i += 8 {
        c[i+0] = a[i+0] + b[i+0]
        c[i+1] = a[i+1] + b[i+1]
        c[i+2] = a[i+2] + b[i+2]
        c[i+3] = a[i+3] + b[i+3]
        c[i+4] = a[i+4] + b[i+4]
        c[i+5] = a[i+5] + b[i+5]
        c[i+6] = a[i+6] + b[i+6]
        c[i+7] = a[i+7] + b[i+7]
    }
    // Remainder
    for ; i < n; i++ {
        c[i] = a[i] + b[i]
    }
}

// Dot product (benefits greatly from SIMD + FMA)
func dotProduct(a, b []float32) float32 {
    var sum float32
    for i := range a {
        sum += a[i] * b[i]
    }
    return sum
}

// Matrix multiply (SIMD + cache-friendly)
func matMul(a, b, c []float32, n int) {
    // Cache-friendly order: i, k, j (reuses a[i][k])
    for i := 0; i < n; i++ {
        for k := 0; k < n; k++ {
            aik := a[i*n+k]
            for j := 0; j < n; j++ {
                c[i*n+j] += aik * b[k*n+j]
            }
        }
    }
}

func benchmark(name string, fn func()) time.Duration {
    start := time.Now()
    fn()
    return time.Since(start)
}

func main() {
    const N = 1 << 20 // 1M elements
    a := make([]float32, N)
    b := make([]float32, N)
    c := make([]float32, N)

    for i := range a {
        a[i] = float32(i) * 0.1
        b[i] = float32(i) * 0.2
    }

    fmt.Println("=== SIMD Performance Demo ===")
    fmt.Printf("Array size: %d elements (%.1f MB each)\n\n", N, float64(N*4)/1e6)

    t1 := benchmark("Scalar Add", func() {
        for iter := 0; iter < 100; iter++ {
            addScalar(a, b, c)
        }
    })

    t2 := benchmark("SIMD-style Add", func() {
        for iter := 0; iter < 100; iter++ {
            addSIMD8(a, b, c)
        }
    })

    fmt.Printf("Scalar add (100 iters): %v\n", t1)
    fmt.Printf("SIMD-style add (100 iters): %v\n", t2)
    fmt.Printf("Speedup: %.2fx\n\n", float64(t1)/float64(t2))

    // Matrix multiply comparison
    const M = 256
    ma := make([]float32, M*M)
    mb := make([]float32, M*M)
    mc := make([]float32, M*M)
    for i := range ma {
        ma[i] = float32(i%100) * 0.01
        mb[i] = float32(i%100) * 0.01
    }

    t3 := benchmark("MatMul", func() {
        matMul(ma, mb, mc, M)
    })
    fmt.Printf("Matrix multiply (%dx%d): %v\n", M, M, t3)
    flops := float64(2*M*M*M) / t3.Seconds()
    fmt.Printf("Performance: %.1f GFLOPS\n", flops/1e9)

    fmt.Println("\n=== SIMD Width Simulation ===")
    elements := 1024
    for _, width := range []int{1, 4, 8, 16} {
        ops := elements / width
        fmt.Printf("  Width %2d: %d elements needs %d vector operations\n",
            width, elements, ops)
    }
}`,
				},
				{
					Title: "GPU Architecture and GPGPU Computing",
					Content: `GPUs evolved from fixed-function graphics accelerators to massively parallel general-purpose processors. Their architecture is fundamentally different from CPUs, optimized for throughput over latency.

**GPU vs CPU Design Philosophy:**
` + "```" + `
CPU: Latency-optimized
  ┌────────────────────────────────────────────┐
  │ [Large Cache    ] [Branch Predictor      ] │
  │ [Out-of-Order   ] [Speculative Execution ] │
  │ [Few wide cores ] [Complex control logic  ] │
  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
  │ │Core 0│ │Core 1│ │Core 2│ │Core 3│       │   4-16 large cores
  │ └──────┘ └──────┘ └──────┘ └──────┘       │
  │ L3 Cache: 16-64 MB                        │
  └────────────────────────────────────────────┘
  Goal: Make individual threads as fast as possible

GPU: Throughput-optimized
  ┌──────────────────────────────────────────────┐
  │ [Tiny cache per SM] [Simple in-order cores ] │
  │ [Massive parallelism] [Hide latency with   ] │
  │ [Many narrow cores  ] [thread switching     ] │
  │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐  │
  │ │SM│SM│SM│SM│SM│SM│SM│SM│SM│SM│SM│SM│SM│  │   64-144 SMs
  │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘  │   × 128 cores/SM
  │      L2 Cache: 4-96 MB                      │   = 8,000-18,000 cores
  └──────────────────────────────────────────────┘
  Goal: Maximize total throughput across all threads

Key difference: CPUs spend transistors on making ONE thread fast;
GPUs spend transistors on running MANY threads simultaneously.
` + "```" + `

**NVIDIA GPU Architecture (Streaming Multiprocessor):**
` + "```" + `
Streaming Multiprocessor (SM) — Ada Lovelace Architecture:
┌──────────────────────────────────────────────┐
│ Warp Scheduler 0    Warp Scheduler 1         │
│ ┌────────────────┐ ┌────────────────┐        │
│ │ Dispatch Unit  │ │ Dispatch Unit  │        │
│ └────────────────┘ └────────────────┘        │
│                                              │
│ ┌───────────────────────────────────┐        │
│ │ FP32 Units:  128 CUDA Cores      │        │
│ │ INT32 Units:  64 Cores           │        │
│ │ FP64 Units:    2 Cores           │        │
│ │ Tensor Cores:  4 (4th gen)       │        │
│ │ RT Cores:      1 (3rd gen)       │        │
│ │ Load/Store:   32 Units           │        │
│ │ SFU:          16 (sin,cos,sqrt)  │        │
│ └───────────────────────────────────┘        │
│                                              │
│ Register File: 256 KB (65,536 × 32-bit)     │
│ Shared Memory / L1 Cache: 128 KB            │
│ Active Warps: Up to 48 (1,536 threads)      │
└──────────────────────────────────────────────┘

Warp: group of 32 threads executing in lockstep (SIMT)
  All 32 threads execute the SAME instruction simultaneously
  If threads diverge (branch), both paths are executed sequentially
  → Branch divergence is expensive on GPUs!

Thread Hierarchy:
  Thread → Warp (32 threads) → Block → Grid
  Block: group of warps sharing shared memory
  Grid: all blocks for a kernel launch
` + "```" + `

**GPU Memory Hierarchy:**
` + "```" + `
Memory Type     │ Scope       │ Speed    │ Size     │ Cached
─────────────────┼─────────────┼──────────┼──────────┼───────
Registers        │ Per thread  │ ~0 cycles│ 255 regs │ N/A
Shared Memory    │ Per block   │ ~5 cycles│ Up to 100KB│ N/A
L1 Cache         │ Per SM      │ ~28 cyc  │ 128 KB   │ Auto
L2 Cache         │ Global      │ ~200 cyc │ 4-96 MB  │ Auto
Global Memory    │ Global      │ ~500 cyc │ 4-80 GB  │ Yes
(HBM/GDDR)

Memory Bandwidth:
  NVIDIA A100: 2 TB/s (HBM2e)
  NVIDIA H100: 3.35 TB/s (HBM3)
  CPU DDR5:    ~50 GB/s (dual channel)
  
  GPU memory bandwidth is 40-60x higher than CPU!
  This is what makes GPUs so fast for data-parallel workloads.

Coalesced Memory Access:
  When 32 threads in a warp access consecutive addresses,
  the memory controller combines them into a single transaction.
  
  GOOD: thread i accesses address base + i × 4 (coalesced → 1 transaction)
  BAD:  thread i accesses address base + i × 1024 (strided → 32 transactions!)
  
  Coalesced access can be 32x faster than uncoalesced!
` + "```" + `

**Tensor Cores:**
` + "```" + `
Tensor Cores perform matrix multiply-accumulate on small matrices:
    D = A × B + C
    
    4th Gen Tensor Core (Ada Lovelace):
    - FP16: 4×4 matrix multiply per clock
    - BF16: 4×4 matrix multiply per clock  
    - INT8:  4×4 per clock (for inference)
    - FP8:   4×4 per clock (Hopper/Ada)
    - TF32:  Automatic for FP32 operations

    Performance comparison (H100):
    FP32 (CUDA cores):  67 TFLOPS
    TF32 (Tensor):     989 TFLOPS
    FP16 (Tensor):    1979 TFLOPS
    INT8 (Tensor):    3958 TOPS

    Why so much faster?
    - Dedicated hardware for matrix math
    - Lower precision = more operations per transistor
    - Matrix operations are highly regular and parallelizable
    
    Used in: AI training (FP16/BF16), inference (INT8/FP8), 
             scientific computing (TF32), ray tracing (RT cores)
` + "```" + ``,
					CodeExamples: `// GPU concepts simulation in Go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Simulated GPU thread/warp/block hierarchy
const (
    WARP_SIZE    = 32
    BLOCK_SIZE   = 256 // threads per block
    WARPS_PER_BK = BLOCK_SIZE / WARP_SIZE
)

// Simulated Streaming Multiprocessor
type SM struct {
    id           int
    sharedMem    [16384]float32 // 64KB shared memory
    activeWarps  int
    computeCores int
}

// GPU Kernel represents a function to be executed on GPU
type GPUKernel struct {
    name        string
    gridSize    int // number of blocks
    blockSize   int // threads per block
    sharedBytes int
}

func (k *GPUKernel) TotalThreads() int {
    return k.gridSize * k.blockSize
}

func (k *GPUKernel) TotalWarps() int {
    warpsPerBlock := (k.blockSize + WARP_SIZE - 1) / WARP_SIZE
    return k.gridSize * warpsPerBlock
}

// Simulate vector addition kernel
func vectorAddGPU(a, b, c []float32, numSMs int) time.Duration {
    n := len(a)
    threadsPerBlock := 256
    numBlocks := (n + threadsPerBlock - 1) / threadsPerBlock

    start := time.Now()
    
    // Simulate GPU execution with goroutines as blocks
    var wg sync.WaitGroup
    sem := make(chan struct{}, numSMs) // Limit concurrency to num SMs
    
    for block := 0; block < numBlocks; block++ {
        wg.Add(1)
        sem <- struct{}{} // Acquire SM slot
        go func(blockIdx int) {
            defer wg.Done()
            defer func() { <-sem }()
            
            startIdx := blockIdx * threadsPerBlock
            for t := 0; t < threadsPerBlock; t++ {
                idx := startIdx + t
                if idx < n {
                    c[idx] = a[idx] + b[idx]
                }
            }
        }(block)
    }
    wg.Wait()
    return time.Since(start)
}

// Simulate matrix multiply with "shared memory" tiling
func matMulTiled(a, b, c []float32, n, tileSize int) {
    for bi := 0; bi < n; bi += tileSize {
        for bj := 0; bj < n; bj += tileSize {
            for bk := 0; bk < n; bk += tileSize {
                // Tile multiply — simulates loading into shared memory
                for i := bi; i < bi+tileSize && i < n; i++ {
                    for k := bk; k < bk+tileSize && k < n; k++ {
                        aik := a[i*n+k]
                        for j := bj; j < bj+tileSize && j < n; j++ {
                            c[i*n+j] += aik * b[k*n+j]
                        }
                    }
                }
            }
        }
    }
}

// Occupancy calculator
type OccupancyCalc struct {
    maxThreadsPerSM    int
    maxBlocksPerSM     int
    maxRegistersPerSM  int
    maxSharedPerSM     int // bytes
}

func (oc *OccupancyCalc) Calculate(threadsPerBlock, regsPerThread, sharedPerBlock int) float64 {
    // Threads limit
    blocksFromThreads := oc.maxThreadsPerSM / threadsPerBlock
    if blocksFromThreads > oc.maxBlocksPerSM {
        blocksFromThreads = oc.maxBlocksPerSM
    }
    
    // Registers limit
    regsPerBlock := threadsPerBlock * regsPerThread
    blocksFromRegs := oc.maxRegistersPerSM / regsPerBlock
    
    // Shared memory limit
    blocksFromShared := oc.maxSharedPerSM / sharedPerBlock
    if sharedPerBlock == 0 {
        blocksFromShared = oc.maxBlocksPerSM
    }
    
    // Minimum of all limits
    activeBlocks := blocksFromThreads
    if blocksFromRegs < activeBlocks { activeBlocks = blocksFromRegs }
    if blocksFromShared < activeBlocks { activeBlocks = blocksFromShared }
    
    activeThreads := activeBlocks * threadsPerBlock
    return float64(activeThreads) / float64(oc.maxThreadsPerSM)
}

func main() {
    // Vector addition comparison
    N := 1 << 20 // 1M elements
    a := make([]float32, N)
    b := make([]float32, N)
    c1 := make([]float32, N)
    c2 := make([]float32, N)
    
    for i := range a {
        a[i] = float32(i) * 0.1
        b[i] = float32(i) * 0.2
    }
    
    fmt.Println("=== GPU vs CPU Vector Addition ===")
    
    // "CPU" (sequential)
    start := time.Now()
    for i := range a { c1[i] = a[i] + b[i] }
    cpuTime := time.Since(start)
    
    // "GPU" (parallel with limited SMs)
    gpuTime := vectorAddGPU(a, b, c2, 16)
    
    fmt.Printf("CPU (sequential): %v\n", cpuTime)
    fmt.Printf("GPU (16 SMs):     %v\n", gpuTime)
    
    // Kernel configuration demo
    fmt.Println("\n=== Kernel Configuration ===")
    kernel := &GPUKernel{
        name:      "vectorAdd",
        gridSize:  (N + 255) / 256,
        blockSize: 256,
    }
    fmt.Printf("Kernel: %s\n", kernel.name)
    fmt.Printf("Grid: %d blocks × %d threads = %d total threads\n",
        kernel.gridSize, kernel.blockSize, kernel.TotalThreads())
    fmt.Printf("Total warps: %d\n", kernel.TotalWarps())
    
    // Occupancy calculation
    fmt.Println("\n=== Occupancy Calculator ===")
    oc := &OccupancyCalc{
        maxThreadsPerSM:   2048,
        maxBlocksPerSM:    32,
        maxRegistersPerSM: 65536,
        maxSharedPerSM:    102400, // 100KB
    }
    
    configs := []struct{ threads, regs, shared int }{
        {128, 32, 0},
        {256, 32, 16384},
        {512, 64, 32768},
        {1024, 128, 49152},
    }
    
    for _, cfg := range configs {
        occ := oc.Calculate(cfg.threads, cfg.regs, cfg.shared)
        fmt.Printf("  Threads=%d Regs=%d Shared=%dKB → Occupancy: %.0f%%\n",
            cfg.threads, cfg.regs, cfg.shared/1024, occ*100)
    }
    
    // GPU specs comparison
    fmt.Println("\n=== GPU Specifications ===")
    type GPUSpec struct {
        name    string
        cores   int
        tflops  float64
        memGB   int
        memBW   float64 // GB/s
    }
    gpus := []GPUSpec{
        {"RTX 4090", 16384, 82.6, 24, 1008},
        {"A100 80GB", 6912, 19.5, 80, 2039},
        {"H100 SXM", 16896, 67, 80, 3350},
    }
    for _, g := range gpus {
        fmt.Printf("  %-12s: %5d cores, %5.1f TFLOPS, %dGB @ %.0f GB/s\n",
            g.name, g.cores, g.tflops, g.memGB, g.memBW)
    }
}`,
				},
				{
					Title: "Parallel Performance Laws",
					Content: `Three fundamental laws govern the performance of parallel systems: Amdahl's Law, Gustafson's Law, and the Roofline Model. Understanding these is essential for predicting and optimizing parallel performance.

**Amdahl's Law:**
` + "```" + `
What portion of a program can be parallelized determines the maximum speedup.

    Speedup(N) = 1 / (s + (1-s)/N)

Where:
    s = serial fraction (portion that cannot be parallelized)
    N = number of processors
    1-s = parallelizable fraction

Key insight: Maximum speedup is LIMITED by serial fraction:
    As N → ∞:  Speedup_max = 1/s

Examples:
    s = 5% (95% parallel):   max speedup = 20x
    s = 10% (90% parallel):  max speedup = 10x
    s = 25% (75% parallel):  max speedup = 4x
    s = 50% (50% parallel):  max speedup = 2x

Even with infinite processors:
┌──────────────────────────────────────────────┐
│ Speedup vs. Processors for various s:         │
│                                               │
│  20x ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  s=5%         │
│                                               │
│  10x ─ ─ ─ ─ ─ ─ ─ ─  s=10%                │
│                                               │
│   4x ─ ─ ─ ─  s=25%                         │
│                                               │
│   2x ─ s=50%                                 │
│   1x─┴────┴────┴────┴────┴────┴─→            │
│     1   16   64   256  1024  ∞  Processors   │
└──────────────────────────────────────────────┘

Implications:
  - Adding more cores has diminishing returns
  - Optimizing the serial portion is CRITICAL
  - Going from 4 to 8 cores gives much less speedup than 1 to 4
` + "```" + `

**Gustafson's Law (Scaled Speedup):**
` + "```" + `
Amdahl's Law assumes FIXED problem size.
Gustafson's Law assumes we SCALE the problem with more processors.

    Speedup(N) = N - s × (N - 1) = s + N × (1 - s)

Where:
    s = serial fraction of the PARALLEL execution time
    N = number of processors

Key insight: With more processors, we solve BIGGER problems,
not the same problem faster.

Example: Weather simulation
  With 1 core:   1km resolution,  10 hours
  With 1000 cores: 10m resolution, 10 hours (same time, bigger problem!)
  
  Scaled speedup = 1000 - 0.01 × 999 = 990x
  (if serial fraction is 1% of parallel runtime)

Gustafson's Law is more optimistic and more realistic for many
scientific and data-processing applications where the problem
grows with available resources.

When to use which:
  Amdahl's: Fixed problem (real-time constraints, interactive apps)
  Gustafson's: Scalable problem (simulations, big data, rendering)
` + "```" + `

**Roofline Model:**
` + "```" + `
The Roofline Model shows whether a kernel is compute-bound 
or memory-bound, and how far it is from peak performance.

Performance (FLOPS/s)
    ↑
    │   ┌─────────────────── Peak Compute (flat)
    │   │ /
    │   │/  ← Peak Memory BW slope
    │  /│
    │ / │        * Kernel A (compute-bound)
    │/  │
    │   │  * Kernel B (memory-bound)
    │   │
    └───┴──────────────────────→
    Operational Intensity (FLOPS/Byte)

Operational Intensity (OI):
    OI = FLOPs performed / Bytes moved from memory
    
    Higher OI = more compute per byte = more likely compute-bound

Ridge Point:
    Where memory bandwidth line meets compute ceiling
    OI_ridge = Peak FLOPS / Peak Bandwidth
    
    H100: 67 TFLOPS / 3350 GB/s = 20 FLOPS/byte
    RTX 4090: 82.6 TFLOPS / 1008 GB/s = 82 FLOPS/byte

Example Analysis:
    Dense matrix multiply: OI ≈ N/16 (grows with matrix size)
      256×256: OI = 16 → compute-bound on H100 ✓
      
    Sparse matrix-vector: OI ≈ 0.25-2 → memory-bound
      Need to optimize memory access patterns
      
    Elementwise add: OI = 0.083 (1 FLOP per 12 bytes)
      Always memory-bound → optimize for bandwidth
` + "```" + `

**Strong vs Weak Scaling:**
` + "```" + `
Strong Scaling: Fixed problem size, increase processors
    Ideal: T(N) = T(1) / N
    Measured: Efficiency = T(1) / (N × T(N))
    
    Example (4K×4K matrix multiply):
    Cores │ Time  │ Speedup │ Efficiency
      1   │ 100s  │  1.0x   │   100%
      4   │  26s  │  3.85x  │    96%
     16   │  7.5s │ 13.3x   │    83%
     64   │  2.5s │ 40.0x   │    63%
    256   │  1.2s │ 83.3x   │    33%
    → Diminishing returns (Amdahl's Law)

Weak Scaling: Problem size grows with processors
    Ideal: T(N) = T(1) = constant
    Measured: Efficiency = T(1) / T(N)
    
    Example (matrix per core: 1K×1K):
    Cores │ Total Size │ Time  │ Efficiency
      1   │   1K×1K    │  10s  │   100%
      4   │   2K×2K    │  11s  │    91%
     16   │   4K×4K    │  13s  │    77%
     64   │   8K×8K    │  16s  │    63%
    → Communication overhead grows with more cores

Communication overhead:
    Computation: O(N³/P) per processor
    Communication: O(N²/√P) for 2D decomposition
    Comm/Comp ratio grows as √P → eventually dominates
` + "```" + ``,
					CodeExamples: `// Parallel performance analysis tools
package main

import (
    "fmt"
    "math"
)

// Amdahl's Law
func amdahlSpeedup(serialFraction float64, processors int) float64 {
    return 1.0 / (serialFraction + (1-serialFraction)/float64(processors))
}

func amdahlMaxSpeedup(serialFraction float64) float64 {
    if serialFraction == 0 { return math.Inf(1) }
    return 1.0 / serialFraction
}

// Gustafson's Law
func gustafsonSpeedup(serialFraction float64, processors int) float64 {
    n := float64(processors)
    return serialFraction + n*(1-serialFraction)
}

// Roofline Model
type RooflineModel struct {
    peakFLOPS     float64 // GFLOPS
    peakBandwidth float64 // GB/s
}

func (rm *RooflineModel) RidgePoint() float64 {
    return rm.peakFLOPS / rm.peakBandwidth
}

func (rm *RooflineModel) AttainableFLOPS(operationalIntensity float64) float64 {
    memBound := rm.peakBandwidth * operationalIntensity
    if memBound < rm.peakFLOPS {
        return memBound // Memory-bound
    }
    return rm.peakFLOPS // Compute-bound
}

func (rm *RooflineModel) IsComputeBound(oi float64) bool {
    return oi >= rm.RidgePoint()
}

// Strong scaling analysis
func strongScaling(baseTime float64, serialFrac float64, cores []int) {
    fmt.Println("\n=== Strong Scaling Analysis ===")
    fmt.Printf("Base time: %.1f s, Serial fraction: %.1f%%\n",
        baseTime, serialFrac*100)
    fmt.Printf("%-8s │ %-10s │ %-10s │ %-10s\n",
        "Cores", "Time", "Speedup", "Efficiency")
    fmt.Println("─────────┼────────────┼────────────┼────────────")
    
    for _, n := range cores {
        speedup := amdahlSpeedup(serialFrac, n)
        time := baseTime / speedup
        efficiency := speedup / float64(n) * 100
        fmt.Printf("%-8d │ %8.2f s │ %8.2fx │ %8.1f%%\n",
            n, time, speedup, efficiency)
    }
}

// Weak scaling analysis
func weakScaling(baseTime, commOverhead float64, cores []int) {
    fmt.Println("\n=== Weak Scaling Analysis ===")
    fmt.Printf("%-8s │ %-10s │ %-10s │ %-12s\n",
        "Cores", "Time", "Efficiency", "Comm Overhead")
    fmt.Println("─────────┼────────────┼────────────┼─────────────")
    
    for _, n := range cores {
        comm := commOverhead * math.Sqrt(float64(n))
        totalTime := baseTime + comm
        efficiency := baseTime / totalTime * 100
        fmt.Printf("%-8d │ %8.2f s │ %8.1f%% │ %8.2f s\n",
            n, totalTime, efficiency, comm)
    }
}

func main() {
    // Amdahl's Law
    fmt.Println("=== Amdahl's Law ===")
    serialFractions := []float64{0.01, 0.05, 0.10, 0.25, 0.50}
    processors := []int{1, 2, 4, 8, 16, 64, 256, 1024}
    
    for _, s := range serialFractions {
        fmt.Printf("\nSerial fraction = %.0f%% (max speedup = %.1fx):\n",
            s*100, amdahlMaxSpeedup(s))
        for _, p := range processors {
            speedup := amdahlSpeedup(s, p)
            efficiency := speedup / float64(p) * 100
            fmt.Printf("  %4d cores: speedup = %6.1fx, efficiency = %5.1f%%\n",
                p, speedup, efficiency)
        }
    }

    // Gustafson comparison
    fmt.Println("\n=== Amdahl vs Gustafson (s=5%%, 64 cores) ===")
    s := 0.05
    n := 64
    fmt.Printf("Amdahl's speedup:    %.1fx\n", amdahlSpeedup(s, n))
    fmt.Printf("Gustafson's speedup: %.1fx\n", gustafsonSpeedup(s, n))

    // Roofline Model
    fmt.Println("\n=== Roofline Model ===")
    gpus := []struct{
        name string
        model RooflineModel
    }{
        {"NVIDIA H100", RooflineModel{67000, 3350}},   // 67 TFLOPS, 3350 GB/s
        {"RTX 4090", RooflineModel{82600, 1008}},       // 82.6 TFLOPS, 1008 GB/s
        {"AMD MI300X", RooflineModel{81700, 5300}},     // 81.7 TFLOPS, 5300 GB/s
    }

    kernels := []struct{
        name string
        oi   float64
        flops float64 // GFLOPS actually achieved
    }{
        {"SpMV (sparse)", 0.5, 0},
        {"BLAS DAXPY", 0.083, 0},
        {"Dense MatMul 256", 16, 0},
        {"Dense MatMul 4096", 256, 0},
        {"Convolution", 8, 0},
    }

    for _, gpu := range gpus {
        fmt.Printf("\n%s (%.0f GFLOPS, %.0f GB/s, ridge=%.1f FLOPS/byte):\n",
            gpu.name, gpu.model.peakFLOPS, gpu.model.peakBandwidth,
            gpu.model.RidgePoint())
        for _, k := range kernels {
            attainable := gpu.model.AttainableFLOPS(k.oi)
            bound := "memory-bound"
            if gpu.model.IsComputeBound(k.oi) {
                bound = "COMPUTE-BOUND"
            }
            fmt.Printf("  %-20s OI=%6.2f → %8.0f GFLOPS (%s)\n",
                k.name, k.oi, attainable, bound)
        }
    }

    // Strong and weak scaling
    strongScaling(100.0, 0.05, []int{1, 2, 4, 8, 16, 32, 64, 128, 256})
    weakScaling(10.0, 0.5, []int{1, 4, 16, 64, 256, 1024})
}`,
				},
			},
		},
	})
}
