package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2831,
			Title:       "Modern Processor Design Trends",
			Description: "Explore chiplet architectures, 3D stacking, heterogeneous computing, RISC-V open ISA, and emerging computing paradigms.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "Chiplet and Advanced Packaging",
					Content: `Modern processors have moved beyond monolithic die designs toward chiplet-based architectures. This approach divides a processor into smaller dies (chiplets) connected via advanced packaging technologies.

**Monolithic vs Chiplet Design:**
` + "```" + `
Monolithic Die (Traditional):
┌─────────────────────────────────────────────┐
│                                             │
│  CPU Cores   Cache   Memory Controller      │
│                                             │
│  I/O         PCIe    Fabric                 │
│                                             │
│  ALL on one large die                       │
│  Single manufacturing process               │
│  Yield decreases as die gets larger         │
│  One defect = entire chip scrapped          │
└─────────────────────────────────────────────┘

Chiplet-Based (AMD Zen 2+):
┌──────────┐ ┌──────────┐ ┌──────────┐
│  CCD #1  │ │  CCD #2  │ │  CCD #3  │
│ 8 cores  │ │ 8 cores  │ │ 8 cores  │
│  7nm     │ │  7nm     │ │  7nm     │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │             │             │
┌────┴─────────────┴─────────────┴────┐
│            I/O Die (cIOD)           │
│  Memory controllers, PCIe, USB     │
│  12nm (cheaper, mature process)    │
└─────────────────────────────────────┘

Advantages:
  ✓ Higher yield (smaller dies = fewer defects per die)
  ✓ Mix-and-match process nodes (7nm for cores, 12nm for I/O)
  ✓ Scalable: same chiplet design for 8, 16, 32, 64, 128 cores
  ✓ Cost effective: one chiplet design, many products
  ✓ Faster time-to-market for new SKUs

Challenges:
  ✗ Inter-chiplet latency (higher than on-die)
  ✗ Packaging complexity and cost
  ✗ Power delivery challenges
  ✗ Testing and validation complexity
` + "```" + `

**Advanced Packaging Technologies:**
` + "```" + `
2D (Traditional):
  Die on substrate, wire bonds or flip-chip bumps
  Bump pitch: ~100-150 µm
  Bandwidth: Limited by bump count

2.5D (Silicon Interposer):
  Multiple dies on a silicon interposer with TSVs
  Examples: AMD MI250X (GPU), Xilinx Versal
  ┌──────┐ ┌──────┐
  │ Die1 │ │ Die2 │
  └──┬───┘ └──┬───┘
  ┌──┴────────┴──┐
  │  Interposer   │  ← Silicon with micro-bumps + TSVs
  └──────┬────────┘
  ┌──────┴────────┐
  │   Substrate    │
  └───────────────┘
  Bump pitch: ~36-55 µm (micro-bumps)
  Bandwidth: 10-100x more than 2D

EMIB (Embedded Multi-die Interconnect Bridge):
  Intel's alternative to full interposer
  Small silicon bridge embedded in substrate
  ┌──────┐  ┌──────┐
  │ Die1 ├──┤ Die2 │  ← Connected by small bridge
  └──┬───┘  └──┬───┘
  ┌──┴──bridge──┴──┐
  │    Substrate    │
  └────────────────┘
  Lower cost than full interposer
  Used in Ponte Vecchio, Sapphire Rapids

3D Stacking:
  Dies stacked vertically with TSVs
  Examples: HBM (DRAM stacks), AMD 3D V-Cache
  ┌──────────┐
  │  SRAM    │ ← 3D V-Cache (64MB L3)
  │   Die    │
  ├──────────┤ ← Hybrid bonding (< 10 µm pitch)
  │  CPU     │
  │  Die     │
  └──────────┘
  Bandwidth: Massive (thousands of connections per mm²)
  Latency: Very low (vertical distance is short)

Hybrid Bonding:
  Direct copper-to-copper connection between dies
  Bump pitch: < 10 µm (vs 36+ µm for micro-bumps)
  Density: 100x more connections than micro-bumps
  Used in: AMD 3D V-Cache, TSMC SoIC
` + "```" + `

**Intel Disaggregated Architecture:**
` + "```" + `
Intel's Tile Architecture (Meteor Lake and beyond):

┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Compute   │ │ SoC      │ │Graphics  │ │ I/O      │
│Tile      │ │ Tile     │ │Tile      │ │Extender  │
│Intel 4   │ │ TSMC N6  │ │ TSMC N5  │ │ TSMC N6  │
│P+E cores │ │NPU, media│ │Xe-LPG   │ │Thunderbolt│
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     └─────────────┴─────────────┴─────────────┘
              Foveros 3D Base Tile

Key: Different tiles can use different foundries!
  Compute: Intel 4 (in-house)
  Graphics: TSMC N5 (outsourced)
  SoC: TSMC N6 (outsourced)

Foveros:
  Intel's 3D stacking technology
  Face-to-face die bonding
  Base tile provides power and I/O routing
  Top tiles provide compute
  36 µm bump pitch → 25 µm → eventually < 10 µm

UCIe (Universal Chiplet Interconnect Express):
  Open standard for chiplet interconnect
  Backed by Intel, AMD, ARM, TSMC, Samsung, etc.
  Goal: Mix chiplets from different vendors
  
  UCIe Specifications:
    Standard Package: 100+ GB/s per mm
    Advanced Package: 1+ TB/s per mm
    Latency: 2-5 ns die-to-die
    Supported: CXL, PCIe protocols
` + "```" + ``,
					CodeExamples: `// Chiplet architecture modeling
package main

import (
    "fmt"
    "math"
)

// Die yield modeling
type ManufacturingProcess struct {
    name           string
    nodeNm         int
    defectDensity  float64 // defects per cm²
    costPerMM2     float64 // $ per mm²
    waferCostK     float64 // $ thousands per 300mm wafer
}

var processes = map[string]ManufacturingProcess{
    "7nm":  {"TSMC N7", 7, 0.10, 0.12, 12.0},
    "5nm":  {"TSMC N5", 5, 0.09, 0.17, 17.0},
    "3nm":  {"TSMC N3", 3, 0.11, 0.30, 30.0},
    "12nm": {"TSMC 12FFC", 12, 0.06, 0.05, 5.0},
}

func poissonYield(area float64, defectDensity float64) float64 {
    return math.Exp(-area * defectDensity / 100) // area in mm², density in /cm²
}

func diePerWafer(dieMM2 float64) int {
    waferAreaMM2 := math.Pi * 150.0 * 150.0 // 300mm wafer
    return int(waferAreaMM2 / dieMM2 * 0.85) // 85% edge loss factor
}

// Chiplet configuration
type ChipletConfig struct {
    name     string
    areamm2  float64
    process  string
    count    int
}

type ProcessorDesign struct {
    name     string
    chiplets []ChipletConfig
    ioDie    ChipletConfig
    pkgCost  float64 // advanced packaging cost
}

func analyzeCost(design ProcessorDesign) {
    fmt.Printf("\n=== %s Cost Analysis ===\n", design.name)
    totalCost := design.pkgCost
    
    // I/O die
    proc := processes[design.ioDie.process]
    yield := poissonYield(design.ioDie.areamm2, proc.defectDensity)
    dpw := diePerWafer(design.ioDie.areamm2)
    costPer := proc.waferCostK * 1000 / float64(dpw) / yield
    totalCost += costPer * float64(design.ioDie.count)
    
    fmt.Printf("  I/O Die (%s, %.0fmm², %s):\n", design.ioDie.name,
        design.ioDie.areamm2, proc.name)
    fmt.Printf("    Yield: %.1f%%, Dies/wafer: %d, Cost: $%.2f × %d\n",
        yield*100, dpw, costPer, design.ioDie.count)

    for _, c := range design.chiplets {
        proc := processes[c.process]
        yield := poissonYield(c.areamm2, proc.defectDensity)
        dpw := diePerWafer(c.areamm2)
        costPer := proc.waferCostK * 1000 / float64(dpw) / yield
        totalCost += costPer * float64(c.count)
        
        fmt.Printf("  %s (%s, %.0fmm²):\n", c.name, proc.name, c.areamm2)
        fmt.Printf("    Yield: %.1f%%, Dies/wafer: %d, Cost: $%.2f × %d\n",
            yield*100, dpw, costPer, c.count)
    }
    
    fmt.Printf("  Packaging cost: $%.2f\n", design.pkgCost)
    fmt.Printf("  → Total die + packaging cost: $%.2f\n", totalCost)
}

func main() {
    fmt.Println("=== Manufacturing Yield Comparison ===")
    
    areas := []float64{50, 100, 200, 400, 600}
    for _, proc := range []string{"12nm", "7nm", "5nm", "3nm"} {
        p := processes[proc]
        fmt.Printf("\n%s (defect density: %.2f/cm²):\n", p.name, p.defectDensity)
        for _, area := range areas {
            yield := poissonYield(area, p.defectDensity) * 100
            dpw := diePerWafer(area)
            cost := p.waferCostK * 1000 / float64(dpw) / (yield / 100)
            fmt.Printf("  %4.0fmm²: yield=%.1f%% dies/wafer=%4d cost=$%6.2f\n",
                area, yield, dpw, cost)
        }
    }

    // Compare monolithic vs chiplet designs
    monolithic := ProcessorDesign{
        name: "Monolithic 64-core (like hypothetical big die)",
        chiplets: []ChipletConfig{
            {"Compute Die", 600, "7nm", 1},
        },
        ioDie:  ChipletConfig{"(integrated)", 0, "7nm", 0},
        pkgCost: 10,
    }
    // Adjusted: add IO die cost to chiplets for monolithic
    monolithic.chiplets[0].areamm2 = 700 // All in one

    chiplet := ProcessorDesign{
        name: "AMD EPYC-style (8 CCDs + 1 IOD)",
        chiplets: []ChipletConfig{
            {"CCD (8-core)", 80, "7nm", 8},
        },
        ioDie:  ChipletConfig{"IOD", 416, "12nm", 1},
        pkgCost: 50, // Advanced packaging overhead
    }

    analyzeCost(monolithic)
    analyzeCost(chiplet)

    // Interconnect bandwidth comparison
    fmt.Println("\n\n=== Interconnect Technology Comparison ===")
    type Interconnect struct {
        name       string
        pitchUM    float64
        bwPerMM    float64 // GB/s per mm of edge
        latencyNS  float64
    }
    
    interconnects := []Interconnect{
        {"Organic substrate (flip-chip)", 130, 1, 5},
        {"Silicon bridge (EMIB)", 55, 20, 3},
        {"Silicon interposer (2.5D)", 36, 50, 2},
        {"Hybrid bonding (3D)", 9, 500, 1},
        {"UCIe Standard", 100, 28, 4},
        {"UCIe Advanced", 25, 165, 2},
    }
    
    fmt.Printf("%-30s │ Pitch │ BW/mm  │ Latency\n", "Technology")
    fmt.Println("───────────────────────────────┼───────┼────────┼────────")
    for _, ic := range interconnects {
        fmt.Printf("%-30s │ %3.0fµm │ %5.0f GB/s│ %3.0f ns\n",
            ic.name, ic.pitchUM, ic.bwPerMM, ic.latencyNS)
    }
}`,
				},
				{
					Title: "RISC-V Open Architecture",
					Content: `RISC-V is an open-source instruction set architecture (ISA) that has disrupted the processor industry. Unlike proprietary ISAs (x86, ARM), RISC-V is free to implement without licensing fees.

**RISC-V Core Concepts:**
` + "```" + `
RISC-V ISA Organization:

Base Integer ISA (one required):
  RV32I  - 32-bit base integer (47 instructions)
  RV64I  - 64-bit base integer
  RV128I - 128-bit base integer (future)
  RV32E  - 32-bit embedded (16 registers instead of 32)

Standard Extensions:
  M - Integer Multiplication/Division
  A - Atomic Instructions (for multi-core)
  F - Single-Precision Floating Point
  D - Double-Precision Floating Point
  C - Compressed Instructions (16-bit, like Thumb)
  V - Vector Extension (SIMD)
  B - Bit Manipulation
  H - Hypervisor (virtualization)
  
Common combinations:
  RV32IMC   - Small embedded (microcontrollers)
  RV64GC    - General purpose (G = IMAFD)
  RV64GCV   - High performance with vectors

Register File:
  x0  (zero) - Hardwired to 0
  x1  (ra)   - Return address
  x2  (sp)   - Stack pointer
  x3  (gp)   - Global pointer
  x4  (tp)   - Thread pointer
  x5-x7      - Temporaries
  x8  (s0/fp)- Saved register / Frame pointer
  x9  (s1)   - Saved register
  x10-x11    - Function args / return values
  x12-x17    - Function arguments
  x18-x27    - Saved registers (s2-s11)
  x28-x31    - Temporaries

Key design principles:
  - No condition codes (flags) - use explicit comparisons
  - No branch delay slots
  - No predicated instructions (ARM has them)
  - Clean, orthogonal encoding
  - Easy to decode (fixed instruction formats)
` + "```" + `

**RISC-V Instruction Formats:**
` + "```" + `
RISC-V has 6 instruction formats (all 32-bit for base ISA):

R-type (Register-Register):
  ┌────────┬─────┬─────┬──────┬─────┬────────┐
  │ funct7 │ rs2 │ rs1 │funct3│ rd  │opcode  │
  │  7b    │ 5b  │ 5b  │  3b  │ 5b  │  7b    │
  └────────┴─────┴─────┴──────┴─────┴────────┘
  Example: add x5, x6, x7  → x5 = x6 + x7

I-type (Immediate):
  ┌──────────────┬─────┬──────┬─────┬────────┐
  │  imm[11:0]   │ rs1 │funct3│ rd  │opcode  │
  │    12b       │ 5b  │  3b  │ 5b  │  7b    │
  └──────────────┴─────┴──────┴─────┴────────┘
  Example: addi x5, x6, 10  → x5 = x6 + 10
  Example: lw x5, 0(x6)     → x5 = Mem[x6+0]

S-type (Store):
  ┌────────┬─────┬─────┬──────┬────────┬───────┐
  │imm[11:5]│ rs2│ rs1 │funct3│imm[4:0]│opcode │
  │  7b    │ 5b  │ 5b  │  3b  │  5b    │  7b   │
  └────────┴─────┴─────┴──────┴────────┴───────┘
  Example: sw x5, 0(x6)     → Mem[x6+0] = x5

B-type (Branch):
  ┌───────┬─────┬─────┬──────┬──────┬────────┐
  │imm    │ rs2 │ rs1 │funct3│ imm  │opcode  │
  │[12|10:5]│5b │ 5b  │  3b  │[4:1|11]│ 7b  │
  └───────┴─────┴─────┴──────┴──────┴────────┘
  Example: beq x5, x6, label → if x5==x6, jump

U-type (Upper Immediate):
  ┌────────────────────────┬─────┬────────┐
  │     imm[31:12]         │ rd  │opcode  │
  │        20b             │ 5b  │  7b    │
  └────────────────────────┴─────┴────────┘
  Example: lui x5, 0x12345 → x5 = 0x12345000

J-type (Jump):
  ┌────────────────────────┬─────┬────────┐
  │ imm[20|10:1|11|19:12]  │ rd  │opcode  │
  │        20b             │ 5b  │  7b    │
  └────────────────────────┴─────┴────────┘
  Example: jal x1, label   → x1 = PC+4; jump to label
` + "```" + `

**RISC-V Ecosystem:**
` + "```" + `
Major RISC-V Implementations:

Commercial Cores:
  SiFive U74     - Linux-capable (used in StarFive VisionFive 2)
  SiFive P670    - High-performance application processor
  Alibaba C910   - Server-class (Xuantie series)
  Ventana Veyron - Data center (192 cores, 2.4 GHz)
  Tenstorrent    - AI accelerator with RISC-V control plane

Microcontrollers:
  Espressif ESP32-C3  - Wi-Fi/BLE MCU (RV32IMC)
  GigaDevice GD32VF103 - Arduino-compatible
  WCH CH32V003  - $0.10 MCU (cheapest RISC-V chip)
  Bouffalo BL602 - Wi-Fi/BLE IoT chip

Open Source Cores:
  BOOM          - Berkeley Out-of-Order Machine
  Rocket        - In-order, 5-stage pipeline
  CVA6 (Ariane) - Linux-capable, academic
  PicoRV32      - Tiny, size-optimized for FPGA
  VexRiscv      - SpinalHDL-based, configurable

Software Ecosystem:
  GCC:      Full support (gcc-riscv64-unknown-elf)
  LLVM:     Full support
  Linux:    Mainline kernel support
  Debian:   RISC-V port available
  Android:  Being ported by Google
  Zephyr:   RTOS support
  FreeRTOS: RTOS support
  Rust:     Tier 2 support

Why RISC-V matters:
  1. No licensing fees → democratizes chip design
  2. Custom extensions → domain-specific accelerators
  3. Security transparency → auditable, no hidden backdoors
  4. Academic access → anyone can study/modify the ISA
  5. Geopolitical → independence from US/UK IP (ARM, x86)
` + "```" + ``,
					CodeExamples: `// RISC-V instruction encoding/decoding simulator
package main

import "fmt"

// RISC-V opcodes
const (
    OP_RTYPE  uint32 = 0b0110011 // R-type (add, sub, etc.)
    OP_ITYPE  uint32 = 0b0010011 // I-type (addi, etc.)
    OP_LOAD   uint32 = 0b0000011 // Loads
    OP_STORE  uint32 = 0b0100011 // Stores
    OP_BRANCH uint32 = 0b1100011 // Branches
    OP_LUI    uint32 = 0b0110111 // LUI
    OP_JAL    uint32 = 0b1101111 // JAL
    OP_JALR   uint32 = 0b1100111 // JALR
)

// Register names
var regNames = [32]string{
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
}

// Encode R-type instruction
func encodeR(funct7, rs2, rs1, funct3, rd, opcode uint32) uint32 {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode
}

// Encode I-type instruction
func encodeI(imm, rs1, funct3, rd, opcode uint32) uint32 {
    return (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

// Decode instruction
func decode(instr uint32) {
    opcode := instr & 0x7F
    rd := (instr >> 7) & 0x1F
    funct3 := (instr >> 12) & 0x7
    rs1 := (instr >> 15) & 0x1F
    rs2 := (instr >> 20) & 0x1F
    funct7 := (instr >> 25) & 0x7F
    immI := int32(instr) >> 20 // Sign-extended I-immediate

    fmt.Printf("  Instruction: 0x%08X (binary: %032b)\n", instr, instr)
    fmt.Printf("  opcode=%07b rd=%s rs1=%s rs2=%s funct3=%03b funct7=%07b\n",
        opcode, regNames[rd], regNames[rs1], regNames[rs2], funct3, funct7)

    switch opcode {
    case OP_RTYPE:
        op := "unknown"
        switch {
        case funct3 == 0 && funct7 == 0:  op = "add"
        case funct3 == 0 && funct7 == 32: op = "sub"
        case funct3 == 1 && funct7 == 0:  op = "sll"
        case funct3 == 4 && funct7 == 0:  op = "xor"
        case funct3 == 6 && funct7 == 0:  op = "or"
        case funct3 == 7 && funct7 == 0:  op = "and"
        }
        fmt.Printf("  → %s %s, %s, %s\n", op, regNames[rd], regNames[rs1], regNames[rs2])
    case OP_ITYPE:
        op := "unknown"
        switch funct3 {
        case 0: op = "addi"
        case 4: op = "xori"
        case 6: op = "ori"
        case 7: op = "andi"
        }
        fmt.Printf("  → %s %s, %s, %d\n", op, regNames[rd], regNames[rs1], immI)
    case OP_LOAD:
        op := "unknown"
        switch funct3 { case 0: op = "lb"; case 1: op = "lh"; case 2: op = "lw" }
        fmt.Printf("  → %s %s, %d(%s)\n", op, regNames[rd], immI, regNames[rs1])
    case OP_LUI:
        immU := instr & 0xFFFFF000
        fmt.Printf("  → lui %s, 0x%X\n", regNames[rd], immU>>12)
    default:
        fmt.Println("  → (other instruction type)")
    }
}

// Simple RISC-V simulator
type RV32Core struct {
    regs [32]int32
    pc   uint32
    mem  [4096]byte
}

func (c *RV32Core) Reset() {
    for i := range c.regs { c.regs[i] = 0 }
    c.pc = 0
}

func (c *RV32Core) ReadReg(r uint32) int32 {
    if r == 0 { return 0 }
    return c.regs[r]
}

func (c *RV32Core) WriteReg(r uint32, val int32) {
    if r != 0 { c.regs[r] = val }
}

func (c *RV32Core) Execute(instr uint32) {
    opcode := instr & 0x7F
    rd := (instr >> 7) & 0x1F
    funct3 := (instr >> 12) & 0x7
    rs1 := (instr >> 15) & 0x1F
    rs2 := (instr >> 20) & 0x1F
    funct7 := (instr >> 25) & 0x7F
    immI := int32(instr) >> 20

    switch opcode {
    case OP_RTYPE:
        v1 := c.ReadReg(rs1)
        v2 := c.ReadReg(rs2)
        var result int32
        switch {
        case funct3 == 0 && funct7 == 0:  result = v1 + v2
        case funct3 == 0 && funct7 == 32: result = v1 - v2
        case funct3 == 7 && funct7 == 0:  result = v1 & v2
        case funct3 == 6 && funct7 == 0:  result = v1 | v2
        case funct3 == 4 && funct7 == 0:  result = v1 ^ v2
        }
        c.WriteReg(rd, result)
    case OP_ITYPE:
        v1 := c.ReadReg(rs1)
        var result int32
        switch funct3 {
        case 0: result = v1 + immI
        case 7: result = v1 & immI
        case 6: result = v1 | immI
        case 4: result = v1 ^ immI
        }
        c.WriteReg(rd, result)
    case OP_LUI:
        c.WriteReg(rd, int32(instr&0xFFFFF000))
    }
    c.pc += 4
}

func (c *RV32Core) DumpRegs() {
    for i := 0; i < 32; i += 4 {
        for j := 0; j < 4; j++ {
            r := i + j
            if c.regs[r] != 0 {
                fmt.Printf("  x%-2d (%-4s) = %d", r, regNames[r], c.regs[r])
            }
        }
    }
}

func main() {
    fmt.Println("=== RISC-V Instruction Encoding ===")
    
    // Encode and decode some instructions
    instructions := []struct{
        name string
        instr uint32
    }{
        {"add t0, t1, t2", encodeR(0, 7, 6, 0, 5, OP_RTYPE)},
        {"sub t0, t1, t2", encodeR(32, 7, 6, 0, 5, OP_RTYPE)},
        {"addi a0, a1, 42", encodeI(42, 11, 0, 10, OP_ITYPE)},
        {"ori a0, zero, 0xFF", encodeI(0xFF, 0, 6, 10, OP_ITYPE)},
        {"lw a0, 0(sp)", encodeI(0, 2, 2, 10, OP_LOAD)},
    }

    for _, inst := range instructions {
        fmt.Printf("\n%s:\n", inst.name)
        decode(inst.instr)
    }

    // Run a small program
    fmt.Println("\n\n=== Simple RISC-V Program Execution ===")
    cpu := &RV32Core{}
    cpu.Reset()

    program := []struct{
        desc  string
        instr uint32
    }{
        {"addi a0, zero, 10", encodeI(10, 0, 0, 10, OP_ITYPE)},
        {"addi a1, zero, 20", encodeI(20, 0, 0, 11, OP_ITYPE)},
        {"add a2, a0, a1",    encodeR(0, 11, 10, 0, 12, OP_RTYPE)},
        {"addi a3, a2, -5",   encodeI(uint32(int32(-5)), 12, 0, 13, OP_ITYPE)},
        {"sub a4, a2, a0",    encodeR(32, 10, 12, 0, 14, OP_RTYPE)},
    }

    for _, step := range program {
        fmt.Printf("PC=0x%04X: %s\n", cpu.pc, step.desc)
        cpu.Execute(step.instr)
    }

    fmt.Println("\nRegister state after execution:")
    for i := 10; i <= 14; i++ {
        fmt.Printf("  x%d (%s) = %d\n", i, regNames[i], cpu.regs[i])
    }
}`,
				},
				{
					Title: "Heterogeneous and Emerging Computing",
					Content: `Modern computing increasingly relies on heterogeneous architectures where different types of processors work together. New computing paradigms are emerging to address the end of Moore's Law.

**Apple Silicon Architecture:**
` + "```" + `
Apple M-series (Heterogeneous SoC):

Apple M3 Pro Layout:
┌─────────────────────────────────────────────┐
│ ┌───────────────────┐ ┌──────────────────┐  │
│ │ P-cores (6×)      │ │ E-cores (6×)     │  │
│ │ 192KB L1I+128KB L1D│ │ 128KB L1I+64KB L1D│ │
│ │ Wide decode (8)   │ │ Narrow decode (4)│  │
│ │ OoO, speculation  │ │ In-order, simple │  │
│ └───────────────────┘ └──────────────────┘  │
│ ┌──────────────────────────────────────────┐│
│ │          36MB Shared L2 Cache            ││
│ └──────────────────────────────────────────┘│
│ ┌───────────┐ ┌───────┐ ┌────────────────┐ │
│ │ GPU       │ │Neural │ │ Media Engine   │ │
│ │ 18 cores  │ │Engine │ │ ProRes, H.265  │ │
│ │ (Apple GPU│ │ 16-   │ │ AV1 decode     │ │
│ │  ISA)     │ │ core  │ │ HW encoder     │ │
│ └───────────┘ └───────┘ └────────────────┘ │
│ ┌───────────────────┐ ┌──────────────────┐  │
│ │ Memory Controller │ │ I/O: Thunderbolt │  │
│ │ Unified LPDDR5    │ │ PCIe, USB, NVMe  │  │
│ │ 150 GB/s          │ │                  │  │
│ └───────────────────┘ └──────────────────┘  │
└─────────────────────────────────────────────┘

Key Innovation: Unified Memory Architecture (UMA)
  - CPU, GPU, Neural Engine share same physical memory
  - No copying data between CPU and GPU memory
  - Lower latency, lower power than discrete GPU with VRAM
  - Trade-off: Memory bandwidth shared among all units

big.LITTLE Scheduling:
  P-cores (Performance): Complex tasks, single-threaded perf
  E-cores (Efficiency): Background tasks, multi-threaded efficiency
  
  macOS scheduler considers:
  - QoS class of the thread
  - Current thermal state
  - Battery vs plugged-in
  - Thread priority and deadline
` + "```" + `

**Domain-Specific Accelerators:**
` + "```" + `
Types of Accelerators in Modern SoCs:

Neural Processing Unit (NPU):
  - Matrix multiply engines optimized for inference
  - INT8/INT4 quantized computation
  - Apple Neural Engine: 18 TOPS
  - Qualcomm Hexagon: 45 TOPS
  - Intel NPU: 10+ TOPS
  Use: On-device ML (face recognition, voice, image processing)

Digital Signal Processor (DSP):
  - VLIW (Very Long Instruction Word) architecture
  - Hardware MAC (Multiply-Accumulate) units
  - Circular buffers, bit-reverse addressing
  Use: Audio processing, sensor fusion, 5G modem

Image Signal Processor (ISP):
  - Dedicated pipeline for camera data
  - Demosaic, noise reduction, HDR, autofocus
  - Processes billions of pixels per second
  Use: Real-time camera processing

Video Codec Engine:
  - Fixed-function H.264/H.265/AV1/VP9 encode/decode
  - 100x more efficient than software codec
  Use: Video playback, recording, streaming

Cryptographic Accelerator:
  - AES-256 encryption/decryption
  - SHA-256/SHA-3 hashing
  - RSA/ECC operations
  Use: Disk encryption, secure boot, TLS

Tensor Processing Unit (Google TPU):
  - Systolic array for matrix multiplication
  - BFloat16 / INT8 compute
  - TPU v4: 275 TFLOPS BF16
  - Connected via custom interconnect (ICI)
  Use: Large-scale ML training and inference
` + "```" + `

**Emerging Computing Paradigms:**
` + "```" + `
Neuromorphic Computing:
  Inspired by biological neural networks
  - Spiking Neural Networks (SNNs)
  - Event-driven (only active neurons consume power)
  - Intel Loihi 2: up to 1 million neurons
  - IBM TrueNorth: 1 million neurons, 256 million synapses
  
  Advantages:
  - Ultra-low power (mW range for complex inference)
  - Inherently parallel
  - Good for temporal/spike-coded data
  - Online learning capability
  
  Limitations:
  - Hard to program (no established frameworks)
  - Not good for traditional workloads
  - Training algorithms still immature

Quantum Computing (Brief Overview):
  - Uses qubits (superposition of 0 and 1)
  - Quantum gates instead of logic gates
  - Good for: factoring, optimization, simulation
  - IBM: 1000+ qubit processors
  - Google: "quantum advantage" with Sycamore
  - Still NISQ era (Noisy Intermediate-Scale Quantum)
  - Not replacing classical computers; complementary

Processing-In-Memory (PIM):
  - Move computation to where data lives
  - Reduces data movement (biggest energy cost)
  - Samsung HBM-PIM: compute in DRAM layers
  - UPMEM: 2048 PIM processors in DIMM
  
  Target: Memory-bound workloads (database, graph, AI)

Photonic Computing:
  - Use light instead of electrons for computation
  - Ultra-fast matrix multiplication at speed of light
  - Near-zero energy for interconnect
  - Lightmatter, Luminous, Intel photonics
  
  Target: AI inference, optical interconnects
  
Approximate Computing:
  - Accept occasional errors for huge efficiency gains
  - Skip computations that don't significantly affect output
  - Example: Image processing (human can't see 1% error)
  - Example: Neural network inference (inherently noise-tolerant)
` + "```" + ``,
					CodeExamples: `// Heterogeneous computing simulation
package main

import (
    "fmt"
    "math"
)

// Processor core types
type CoreType int
const (
    PerformanceCore CoreType = iota
    EfficiencyCore
    GPU
    NPU
    DSP
)

func (ct CoreType) String() string {
    return [...]string{"P-Core", "E-Core", "GPU", "NPU", "DSP"}[ct]
}

// Heterogeneous processor model
type HetCore struct {
    coreType    CoreType
    count       int
    freqGHz     float64
    powerW      float64 // per core
    peakGFLOPS  float64 // per core
    peakTOPS    float64 // per core (for NPU)
}

type HetProcessor struct {
    name  string
    cores []HetCore
    memBW float64 // GB/s
}

// Workload characteristics
type Workload struct {
    name          string
    cpuIntensive  float64 // fraction needing CPU
    gpuIntensive  float64 // fraction needing GPU
    aiIntensive   float64 // fraction needing NPU
    totalGFLOPS   float64 // total compute needed
}

func (p HetProcessor) RunWorkload(w Workload) {
    fmt.Printf("\n--- %s on %s ---\n", w.name, p.name)
    
    totalTime := 0.0
    totalPower := 0.0
    
    for _, core := range p.cores {
        var fraction float64
        var perf float64
        
        switch core.coreType {
        case PerformanceCore:
            fraction = w.cpuIntensive * 0.7 // P-cores handle 70% of CPU work
            perf = core.peakGFLOPS * float64(core.count)
        case EfficiencyCore:
            fraction = w.cpuIntensive * 0.3 // E-cores handle 30%
            perf = core.peakGFLOPS * float64(core.count)
        case GPU:
            fraction = w.gpuIntensive
            perf = core.peakGFLOPS * float64(core.count)
        case NPU:
            fraction = w.aiIntensive
            perf = core.peakTOPS * 1000 * float64(core.count) // Convert TOPS to GOPS
        default:
            continue
        }
        
        if fraction > 0 && perf > 0 {
            work := w.totalGFLOPS * fraction
            time := work / perf
            power := core.powerW * float64(core.count) * (fraction)
            totalTime += time
            totalPower += power
            
            fmt.Printf("  %s ×%d: %.1f GFLOPS work, %.2f ms, %.1f W\n",
                core.coreType, core.count, work, time*1000, power)
        }
    }
    
    energy := totalPower * totalTime // Watt-seconds
    fmt.Printf("  Total: %.2f ms, %.1f W avg, %.3f mJ energy\n",
        totalTime*1000, totalPower, energy*1000)
}

// Roofline model for heterogeneous systems
func rooflineAnalysis(peakGFLOPS, memBW, opIntensity float64) float64 {
    memBound := memBW * opIntensity
    return math.Min(peakGFLOPS, memBound)
}

func main() {
    // Define processors
    m3pro := HetProcessor{
        name: "Apple M3 Pro",
        cores: []HetCore{
            {PerformanceCore, 6, 4.0, 5.0, 50, 0},
            {EfficiencyCore, 6, 2.0, 1.0, 15, 0},
            {GPU, 18, 1.4, 1.5, 100, 0}, // 18 GPU cores
            {NPU, 1, 0, 1.0, 0, 18},     // 18 TOPS
        },
        memBW: 150,
    }

    intel14900 := HetProcessor{
        name: "Intel i9-14900K",
        cores: []HetCore{
            {PerformanceCore, 8, 6.0, 15.0, 80, 0},
            {EfficiencyCore, 16, 4.3, 3.0, 25, 0},
            {GPU, 32, 1.6, 2.0, 50, 0}, // Integrated
        },
        memBW: 90,
    }

    // Define workloads
    workloads := []Workload{
        {"Video Editing (4K)", 0.3, 0.5, 0.2, 500},
        {"Machine Learning Inference", 0.1, 0.3, 0.6, 300},
        {"Compiling Large Project", 0.9, 0.05, 0.05, 200},
        {"Gaming (1080p)", 0.2, 0.7, 0.1, 1000},
    }

    for _, proc := range []HetProcessor{m3pro, intel14900} {
        fmt.Printf("\n=== %s ===\n", proc.name)
        for _, w := range workloads {
            proc.RunWorkload(w)
        }
    }

    // Roofline analysis
    fmt.Println("\n\n=== Roofline Model Analysis ===")
    fmt.Printf("%-25s │ Peak GFLOPS │ Mem BW │ Op Intensity │ Achieved\n", "Unit")
    fmt.Println("──────────────────────────┼─────────────┼────────┼──────────────┼─────────")
    
    units := []struct {
        name string
        peak float64
        bw   float64
        oi   float64
    }{
        {"M3 Pro P-core", 50, 150, 2.0},
        {"M3 Pro GPU", 1800, 150, 20.0},
        {"M3 Pro GPU (mem-bound)", 1800, 150, 0.5},
        {"i9-14900K all cores", 800, 90, 4.0},
        {"i9-14900K (mem-bound)", 800, 90, 0.25},
    }

    for _, u := range units {
        achieved := rooflineAnalysis(u.peak, u.bw, u.oi)
        bound := "compute"
        if u.bw*u.oi < u.peak { bound = "memory" }
        fmt.Printf("%-25s │ %8.0f    │ %4.0fGB/s│ %8.1f     │ %6.0f GFLOPS (%s)\n",
            u.name, u.peak, u.bw, u.oi, achieved, bound)
    }
}`,
				},
			},
		},
	})
}
