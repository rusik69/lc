package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2837,
			Title:       "FPGA and Reconfigurable Computing",
			Description: "Understand FPGA architecture, hardware description languages, reconfigurable computing paradigms, and how FPGAs compare to CPUs, GPUs, and ASICs.",
			Order:       37,
			Lessons: []problems.Lesson{
				{
					Title: "FPGA Architecture",
					Content: `Field-Programmable Gate Arrays (FPGAs) are semiconductor devices that can be configured after manufacturing to implement custom digital circuits. They sit between general-purpose processors and fixed-function ASICs in the flexibility-efficiency spectrum.

**FPGA vs CPU vs GPU vs ASIC:**
` + "```" + `
Comparison across key metrics:

            │ Flexibility │ Performance │ Power Eff. │ Unit Cost │ NRE Cost
CPU         │ ★★★★★      │ ★★          │ ★★         │ $$$       │ $
GPU         │ ★★★★       │ ★★★★        │ ★★★        │ $$$$      │ $
FPGA        │ ★★★★       │ ★★★         │ ★★★★       │ $$$       │ $$
ASIC        │ ★           │ ★★★★★      │ ★★★★★      │ $         │ $$$$$

NRE = Non-Recurring Engineering (design cost)
  CPU/GPU: Write software → $0-$1M
  FPGA: Write HDL, synthesize → $100K-$5M
  ASIC: Design, tape-out, fab → $10M-$500M+

When to use FPGA:
  ✓ Low latency requirements (sub-microsecond)
  ✓ Custom data paths (bit manipulation, protocol handling)
  ✓ Low-to-medium volume production
  ✓ Field-updatable hardware  
  ✓ Mixed-signal interfaces
  ✗ Not for general-purpose computing
  ✗ Not cost-effective at very high volumes (ASIC wins)
  ✗ Lower clock speeds than custom silicon

FPGA use cases:
  - Network packet processing (firewalls, switches)
  - 5G/LTE base stations (digital signal processing)
  - Financial trading (sub-microsecond order execution)
  - Video/image processing (real-time encoding)
  - AI inference acceleration
  - Prototyping ASICs before tape-out
  - Embedded systems (military, space, automotive)
  - Data center accelerators (Microsoft Brainwave, AWS F1)
` + "```" + `

**FPGA Internal Architecture:**
` + "```" + `
FPGA consists of:

1. Configurable Logic Blocks (CLBs):
   ┌─────────────────────────────┐
   │ CLB                          │
   │ ┌─────────┐ ┌─────────┐    │
   │ │  LUT-6  │ │  LUT-6  │    │  LUT = Look-Up Table
   │ │ (64-bit │ │ (64-bit │    │  Can implement ANY
   │ │  SRAM)  │ │  SRAM)  │    │  6-input boolean function
   │ └────┬────┘ └────┬────┘    │
   │      │           │          │
   │ ┌────┴────┐ ┌────┴────┐    │
   │ │ Flip-   │ │ Flip-   │    │  Register stores state
   │ │ Flop    │ │ Flop    │    │  (sequential logic)
   │ └────┬────┘ └────┬────┘    │
   │      │           │          │
   │ ┌────┴───────────┴────┐    │
   │ │   Carry Chain Logic  │    │  Fast arithmetic
   │ └─────────────────────┘    │
   └─────────────────────────────┘
   
   LUT-6: 6-input lookup table
     → 2⁶ = 64 SRAM bits store the truth table
     → Can implement ANY function of 6 variables
     → Two LUT-6s can combine for 7+ input functions

2. Routing Network:
   ┌────┐    ┌────┐    ┌────┐
   │CLB ├────┤ SB ├────┤CLB │
   └──┬─┘    └──┬─┘    └──┬─┘
      │         │         │       SB = Switch Box
   ┌──┴─┐    ┌──┴─┐    ┌──┴─┐   CB = Connection Box
   │ CB │    │ SB │    │ CB │
   └──┬─┘    └──┬─┘    └──┬─┘   Routing uses 60-80%
      │         │         │      of total FPGA area!
   ┌──┴─┐    ┌──┴─┐    ┌──┴─┐
   │CLB ├────┤ SB ├────┤CLB │
   └────┘    └────┘    └────┘
   
   Island-style architecture (most common)
   Hierarchical routing: local, intermediate, long lines

3. Hard IP Blocks:
   - Block RAM (BRAM): 18Kb/36Kb dual-port memory blocks
   - DSP Slices: 27×18 multiplier + 48-bit accumulator
   - Clock Management: PLL, MMCM for clock generation
   - I/O: SerDes (up to 112 Gbps per lane)
   - PCIe: Hard PCIe Gen4/Gen5 controller
   - Memory Controller: DDR4/DDR5 hard controller
   - ARM Cores: Embedded processor (Xilinx Zynq, Intel Agilex)
` + "```" + `

**FPGA Design Flow:**
` + "```" + `
FPGA Design Process:

1. Specification → Define what the circuit should do

2. RTL Design (Register Transfer Level):
   Write in HDL (Verilog, VHDL, SystemVerilog)
   Or use HLS (High-Level Synthesis) from C/C++
   
3. Simulation:
   Verify logic correctness before synthesis
   Testbench stimulates inputs, checks outputs
   
4. Synthesis:
   HDL → Gate-level netlist (technology mapping)
   LUTs, flip-flops, memories chosen from FPGA primitives
   
5. Place and Route:
   Assign netlist elements to physical FPGA resources
   Route wires between them
   Most time-consuming step (NP-hard problem)
   
6. Timing Analysis:
   Verify all paths meet timing constraints
   Setup time: data must arrive before clock edge
   Hold time: data must remain stable after clock edge
   
7. Bitstream Generation:
   Create binary configuration file
   Programs SRAM cells to configure LUTs and routing
   
8. Programming:
   Load bitstream into FPGA via JTAG or flash
   FPGA implements your custom circuit!
   
   Total design time: hours to months
   ASIC: months to years (but faster result)

Timing Closure Challenge:
  Target: e.g., 250 MHz clock → 4 ns period
  Critical path: longest combinational delay between registers
  If critical path > 4 ns → timing failure
  Options: pipeline more, restructure logic, optimize placement
` + "```" + `

**Modern FPGA Platforms:**
` + "```" + `
AMD/Xilinx:
  Versal: AI Engine + FPGA + ARM cores
    - AI Engines: 400+ vector processors (SIMD/VLIW)
    - Programmable Logic: 1.9M LUTs
    - DSP: 1,968 DSP engines
    - ARM: Dual Cortex-A72 + Dual Cortex-R5F
    - Network-on-Chip (NoC) interconnect
    
  Alveo (datacenter accelerators):
    - Alveo U55C: 1.3M LUTs, 16GB HBM2
    - Used in: financial trading, genomics, compression

Intel/Altera:
  Agilex: FPGA + HBM + PCIe Gen5
    - Up to 10M logic elements (equiv.)
    - Integrated HBM2E (32 GB, 820 GB/s)
    - CXL connectivity
    
  Stratix: High-end FPGA
    - Used in: 5G infrastructure, military radar

Lattice:
  CrossLink-NX: Small, low-power FPGAs
    - milliwatt-range power
    - Used in: cameras, edge AI, IoT
    
Microchip (formerly Actel/Microsemi):
  PolarFire: Radiation-tolerant, non-volatile
    - Flash-based (instant-on, no configuration time)
    - Space-qualified versions available
    - Used in: aerospace, defense, medical

FPGA Resources (Xilinx Versal Premium VP1902):
  LUTs:           1,954,560
  Flip-Flops:     3,909,120
  Block RAM:      2,160 (36Kb each) = 77 Mb
  UltraRAM:       960 (288Kb each) = 270 Mb
  DSP Slices:     1,968
  AI Engines:     400
  Transmitters:   80 × 112Gbps = 8.96 Tbps
  PCIe:           4 × Gen5
  DDR:            3 × DDR5 controllers
  Price:          $50,000+ (high-end FPGAs are expensive!)
` + "```" + ``,
					CodeExamples: `// FPGA concepts simulation
package main

import (
    "fmt"
    "math"
    "strings"
)

// LUT (Look-Up Table) simulation
type LUT6 struct {
    truthTable [64]bool // 2^6 = 64 entries
    inputs     [6]string
    name       string
}

func NewLUT6(name string, f func(a, b, c, d, e, g bool) bool) *LUT6 {
    lut := &LUT6{name: name}
    for i := 0; i < 64; i++ {
        a := (i>>0)&1 == 1
        b := (i>>1)&1 == 1
        c := (i>>2)&1 == 1
        d := (i>>3)&1 == 1
        e := (i>>4)&1 == 1
        g := (i>>5)&1 == 1
        lut.truthTable[i] = f(a, b, c, d, e, g)
    }
    return lut
}

func (l *LUT6) Evaluate(inputs [6]bool) bool {
    idx := 0
    for i, v := range inputs {
        if v { idx |= 1 << i }
    }
    return l.truthTable[idx]
}

func (l *LUT6) Utilization() float64 {
    ones := 0
    for _, v := range l.truthTable {
        if v { ones++ }
    }
    return float64(ones) / 64.0 * 100
}

// FPGA resource utilization
type FPGADesign struct {
    name       string
    lutsUsed   int
    ffsUsed    int
    bramsUsed  int
    dspsUsed   int
    fMaxMHz    float64
}

type FPGADevice struct {
    name       string
    totalLUTs  int
    totalFFs   int
    totalBRAMs int
    totalDSPs  int
    maxMHz     float64
}

func (d FPGADesign) Utilization(dev FPGADevice) map[string]float64 {
    return map[string]float64{
        "LUT":  float64(d.lutsUsed) / float64(dev.totalLUTs) * 100,
        "FF":   float64(d.ffsUsed) / float64(dev.totalFFs) * 100,
        "BRAM": float64(d.bramsUsed) / float64(dev.totalBRAMs) * 100,
        "DSP":  float64(d.dspsUsed) / float64(dev.totalDSPs) * 100,
    }
}

// Compare FPGA vs CPU vs GPU for specific workloads
type AcceleratorComparison struct {
    workload     string
    cpuLatencyUS float64
    gpuLatencyUS float64
    fpgaLatencyUS float64
    cpuThroughput float64 // Gops
    gpuThroughput float64
    fpgaThroughput float64
    cpuPowerW     float64
    gpuPowerW     float64
    fpgaPowerW    float64
}

func (a AcceleratorComparison) Print() {
    fmt.Printf("\n  %s:\n", a.workload)
    fmt.Printf("    %-6s │ Latency │ Throughput │ Power │ Perf/W\n", "Device")
    fmt.Println("    ───────┼─────────┼────────────┼───────┼───────")
    
    devices := []struct {
        name string
        lat  float64
        thr  float64
        pow  float64
    }{
        {"CPU", a.cpuLatencyUS, a.cpuThroughput, a.cpuPowerW},
        {"GPU", a.gpuLatencyUS, a.gpuThroughput, a.gpuPowerW},
        {"FPGA", a.fpgaLatencyUS, a.fpgaThroughput, a.fpgaPowerW},
    }
    
    for _, d := range devices {
        perfPerW := d.thr / d.pow
        latStr := fmt.Sprintf("%.1f µs", d.lat)
        if d.lat >= 1000 {
            latStr = fmt.Sprintf("%.1f ms", d.lat/1000)
        }
        fmt.Printf("    %-6s │ %7s │ %6.1f Gops│ %4.0f W│ %.2f\n",
            d.name, latStr, d.thr, d.pow, perfPerW)
    }
}

// FPGA pipeline performance model
type PipelineDesign struct {
    name        string
    stages      int
    clockMHz    float64
    dataWidthBits int
}

func (p PipelineDesign) Latency() float64 {
    return float64(p.stages) / p.clockMHz * 1000 // nanoseconds
}

func (p PipelineDesign) ThroughputGbps() float64 {
    return float64(p.dataWidthBits) * p.clockMHz / 1000
}

func main() {
    fmt.Println("=== LUT-6 Demonstration ===")
    
    // Implement various functions in LUTs
    andGate := NewLUT6("6-input AND", func(a, b, c, d, e, f bool) bool {
        return a && b && c && d && e && f
    })
    
    mux4 := NewLUT6("4:1 MUX", func(a, b, c, d, sel0, sel1 bool) bool {
        switch {
        case !sel1 && !sel0: return a
        case !sel1 && sel0:  return b
        case sel1 && !sel0:  return c
        default:             return d
        }
    })
    
    fullAdder := NewLUT6("Full Adder Sum", func(a, b, cin, _, _, _ bool) bool {
        return a != b != cin // XOR
    })
    
    majority := NewLUT6("5-input Majority", func(a, b, c, d, e, _ bool) bool {
        count := 0
        for _, v := range []bool{a, b, c, d, e} {
            if v { count++ }
        }
        return count >= 3
    })
    
    luts := []*LUT6{andGate, mux4, fullAdder, majority}
    
    for _, lut := range luts {
        fmt.Printf("  %s: %.1f%% truth table utilization\n",
            lut.name, lut.Utilization())
    }
    
    // Test the MUX
    fmt.Println("\n  4:1 MUX test (sel1=0, sel0=1 → selects input B):")
    result := mux4.Evaluate([6]bool{false, true, false, false, true, false})
    fmt.Printf("    inputs=[0,1,0,0] sel=[1,0] → output=%v (selected B=1)\n", result)
    
    // FPGA resource utilization
    fmt.Println("\n=== FPGA Design Resource Utilization ===")
    
    device := FPGADevice{
        name: "Xilinx VU9P", totalLUTs: 1182240,
        totalFFs: 2364480, totalBRAMs: 2160, totalDSPs: 6840,
        maxMHz: 500,
    }
    
    designs := []FPGADesign{
        {"100GbE NIC", 120000, 180000, 400, 100, 300},
        {"Neural Network Inference", 450000, 600000, 1800, 5000, 250},
        {"Video Encoder (H.265)", 280000, 350000, 800, 2400, 350},
        {"Crypto Mining (SHA-256)", 800000, 1000000, 200, 6000, 400},
        {"Network Firewall", 95000, 120000, 300, 50, 350},
    }
    
    fmt.Printf("\n%-25s │ LUT%%  │ FF%%   │ BRAM%% │ DSP%%  │ Fmax\n", "Design")
    fmt.Println("──────────────────────────┼───────┼───────┼───────┼───────┼──────")
    
    for _, d := range designs {
        u := d.Utilization(device)
        fmt.Printf("%-25s │ %4.1f%%│ %4.1f%%│ %4.1f%%│ %4.1f%%│ %3.0f MHz\n",
            d.name, u["LUT"], u["FF"], u["BRAM"], u["DSP"], d.fMaxMHz)
    }
    
    // FPGA vs CPU vs GPU comparison
    fmt.Println("\n=== FPGA vs CPU vs GPU Comparison ===")
    
    comparisons := []AcceleratorComparison{
        {"Network packet processing (100GbE line rate)",
            50, 100, 0.5,   // Latency µs
            10, 20, 148.8,  // Throughput Gops
            250, 300, 40},  // Power W
        {"Matrix multiply (4096×4096 FP16)",
            5000, 200, 1000,
            2, 100, 10,
            250, 300, 75},
        {"AES-256 encryption",
            10, 5, 0.1,
            50, 200, 400,
            250, 300, 25},
        {"Database query (string matching)",
            100, 500, 5,
            5, 2, 50,
            250, 300, 35},
    }
    
    for _, c := range comparisons {
        c.Print()
    }
    
    // Pipeline design comparison
    fmt.Println("\n\n=== FPGA Pipeline Designs ===")
    
    pipelines := []PipelineDesign{
        {"AES-256 pipeline", 14, 300, 128},
        {"SHA-256 pipeline", 66, 350, 512},
        {"FIR filter (32-tap)", 32, 400, 32},
        {"Packet parser", 8, 500, 512},
        {"JPEG encoder", 45, 250, 64},
    }
    
    fmt.Printf("%-20s │ Stages │ Clock  │ Latency │ Throughput\n", "Design")
    fmt.Println("─────────────────────┼────────┼────────┼─────────┼──────────")
    
    for _, p := range pipelines {
        fmt.Printf("%-20s │ %4d   │ %3.0f MHz│ %5.1f ns│ %6.1f Gbps\n",
            p.name, p.stages, p.clockMHz, p.Latency(), p.ThroughputGbps())
    }
    
    // FPGA cost analysis
    fmt.Println("\n=== FPGA vs ASIC Cost Crossover ===")
    
    fpgaUnitCost := 500.0     // $ per unit
    fpgaDesignCost := 500000.0 // $ NRE
    asicUnitCost := 5.0        // $ per unit at volume
    asicDesignCost := 50000000.0 // $ NRE (tape-out + masks)
    
    // Find crossover volume
    // FPGA total = fpgaDesignCost + fpgaUnitCost * N
    // ASIC total = asicDesignCost + asicUnitCost * N
    // Crossover: fpgaDesignCost + fpgaUnitCost * N = asicDesignCost + asicUnitCost * N
    crossover := (asicDesignCost - fpgaDesignCost) / (fpgaUnitCost - asicUnitCost)
    
    fmt.Printf("FPGA: $%.0fK NRE + $%.0f/unit\n", fpgaDesignCost/1000, fpgaUnitCost)
    fmt.Printf("ASIC: $%.0fM NRE + $%.0f/unit\n", asicDesignCost/1e6, asicUnitCost)
    fmt.Printf("Crossover at: %.0f units\n\n", crossover)
    
    volumes := []int{100, 1000, 10000, 50000, 100000, 500000, 1000000}
    fmt.Printf("%-12s │ FPGA Total │ ASIC Total │ Winner │ Savings\n", "Volume")
    fmt.Println("─────────────┼────────────┼────────────┼────────┼────────")
    
    for _, n := range volumes {
        fpgaTotal := fpgaDesignCost + fpgaUnitCost*float64(n)
        asicTotal := asicDesignCost + asicUnitCost*float64(n)
        winner := "FPGA"
        savings := asicTotal - fpgaTotal
        if asicTotal < fpgaTotal {
            winner = "ASIC"
            savings = fpgaTotal - asicTotal
        }
        
        fmtCost := func(c float64) string {
            if c >= 1e6 { return fmt.Sprintf("$%.1fM", c/1e6) }
            return fmt.Sprintf("$%.0fK", c/1000)
        }
        
        fmt.Printf("%10d   │ %10s │ %10s │ %-6s │ %s\n",
            n, fmtCost(fpgaTotal), fmtCost(asicTotal), winner, fmtCost(savings))
    }
    
    // FPGA design complexity visualization
    fmt.Println("\n=== Design Complexity vs Approach ===")
    approaches := []struct {
        name       string
        abstraction string
        designTime string
        perfPct    float64
        skill      string
    }{
        {"Hand-coded RTL (Verilog)", "Low", "Months", 100, "Hardware engineer"},
        {"IP-based (Vivado IPI)", "Medium", "Weeks", 85, "FPGA developer"},
        {"HLS (C/C++ → RTL)", "High", "Days-Weeks", 60, "Software + HW"},
        {"OpenCL for FPGA", "High", "Days", 40, "Software engineer"},
        {"Overlay (soft CPU)", "Very High", "Hours", 10, "Any programmer"},
    }
    
    fmt.Printf("%-25s │ Design Time │ Perf%% │ Skill Level\n", "Approach")
    fmt.Println("──────────────────────────┼─────────────┼───────┼────────────────")
    for _, a := range approaches {
        bar := strings.Repeat("█", int(a.perfPct/5))
        fmt.Printf("%-25s │ %-11s │ %s %2.0f%%│ %s\n",
            a.name, a.designTime, bar, a.perfPct, a.skill)
    }
    _ = math.Pi // use math package
}`,
				},
			},
		},
	})
}
