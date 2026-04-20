package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2836,
			Title:       "Reliability, Power, and Thermal Management",
			Description: "Understand power consumption models, dynamic voltage/frequency scaling, thermal throttling, error detection/correction, and reliability engineering in processors.",
			Order:       36,
			Lessons: []problems.Lesson{
				{
					Title: "Power Consumption in CMOS",
					Content: `Power consumption is the primary constraint in modern processor design. Understanding the sources of power consumption is essential for designing efficient systems from embedded to datacenter scale.

**CMOS Power Equation:**
` + "```" + `
Total Power = Dynamic Power + Static Power + Short-Circuit Power

Dynamic Power (switching activity):
  P_dynamic = α × C × V² × f
  
  α = activity factor (fraction of gates switching per cycle)
  C = capacitance (proportional to transistor count & wire length)
  V = supply voltage
  f = clock frequency
  
  Key insight: Power scales with V² !
    Reduce voltage by 20% → reduce dynamic power by 36%
    This is why voltage scaling is the most effective technique

Static Power (leakage):
  P_static = V × I_leak
  
  I_leak = leakage current (flows even when transistors are "off")
  Increases exponentially with:
    - Temperature (hot → more leakage → more heat → thermal runaway!)
    - Smaller transistors (thinner gate oxide → more tunneling)
    - Lower threshold voltage (faster transistors leak more)
  
  At modern process nodes (5nm, 3nm):
    Leakage can be 30-50% of total power!
    
  Techniques to control leakage:
    Power gating: Cut power to unused blocks entirely
    Multi-Vt: Use high-Vt transistors in non-critical paths
    Body biasing: Adjust transistor threshold voltage dynamically

Short-Circuit Power:
  During switching, both PMOS and NMOS briefly conduct
  Usually ~10% of dynamic power
  Minimized by fast signal transitions
` + "```" + `

**Voltage-Frequency Scaling:**
` + "```" + `
DVFS (Dynamic Voltage and Frequency Scaling):

Frequency is proportional to voltage:
  f_max ∝ (V - Vth)^α / V
  (α ≈ 1-2, Vth = threshold voltage)

Power scales cubically with frequency (since V must increase too):
  P ∝ f³ (approximate, since V ∝ f)

Example (Intel Core i9-14900K):
  Efficiency cores:
    Base: 2.2 GHz @ 0.75V → ~8W
    Boost: 4.4 GHz @ 1.15V → ~45W
    2× frequency → ~5.6× power!
    
  Performance cores:
    Base: 3.2 GHz @ 0.85V → ~20W
    Boost: 6.0 GHz @ 1.35V → ~120W
    1.9× frequency → ~6× power!

Energy-Delay Product (EDP):
  EDP = Energy × Delay = P × T × T = P × T²
  
  Lower voltage/frequency:
    Takes longer (T↑) but uses much less energy (P↓↓↓)
    Sweet spot: highest perf/watt, not highest performance
    
  ┌──────────────────────────────────┐
  │ Performance                       │
  │    ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲            │
  │   │                   diminishing │
  │  ╱                    returns     │
  │ │                                 │
  │─┼────────────────────── Voltage   │
  │                                   │
  │ Power                             │
  │                    ╱              │
  │               ╱╱╱                 │
  │          ╱╱                       │
  │      ╱                            │
  │─────────────────────── Voltage    │
  └──────────────────────────────────┘
  
  Operating near maximum voltage:
    5% more performance → 15% more power
    → Very bad energy efficiency

Race to Idle:
  Strategy: Run at max speed, finish quickly, sleep
  Sometimes more efficient than running at lower speed
  Because: leakage power consumed during entire execution
  Works when: leakage dominates, deep sleep available
  
  Energy at max speed: P_max × T_min + P_sleep × (T_total - T_min)
  Energy at half speed: P_half × T_half + P_sleep × (T_total - T_half)
  If P_sleep is very low → race to idle wins
` + "```" + `

**Thermal Management:**
` + "```" + `
Power Density and Thermal Limits:

Heat generation:
  Q = P / Area  (W/cm² or W/mm²)
  
  i486 (1989): 1 W/cm² → passive cooling sufficient
  Pentium 4 (2004): 100 W/cm² → aggressive cooling needed
  Modern hotspots: 500+ W/cm² locally → thermal throttling
  
  "Power Wall": Cannot dissipate heat fast enough
  Stopped clock frequency scaling at ~4 GHz (2004)
  Led to: multi-core revolution

Thermal equation:
  T_junction = T_ambient + θ_JA × P
  
  θ_JA = junction-to-ambient thermal resistance (°C/W)
  
  Example air-cooled desktop:
    θ_JA = 0.3 °C/W
    T_ambient = 25°C
    P = 125W
    T_junction = 25 + 0.3 × 125 = 62.5°C (fine, <100°C)
    
  Example laptop:
    θ_JA = 0.8 °C/W (worse cooling)
    P = 125W
    T_junction = 25 + 0.8 × 125 = 125°C (too hot! Throttle!)
    Must reduce to ~60W → T = 25 + 0.8 × 60 = 73°C

Thermal Throttling:
  When T_junction approaches T_max (~100°C for most CPUs):
    1. Reduce clock frequency (DVFS)
    2. Reduce voltage (saves power quadratically)
    3. Duty cycling (skip clock cycles - "clock modulation")
    4. Power capping (limit total package power)
    5. Emergency shutdown (prevent permanent damage)

Power Delivery:
  Voltage Regulator Module (VRM):
    Converts 12V ATX to ~0.7-1.4V CPU core voltage
    Must deliver 300+ amps at low voltage
    Efficiency: 90-95% (5-10% loss = significant heat!)
    
  IR Drop (Voltage Droop):
    Resistance in power delivery network causes voltage drop
    Sudden load change → voltage droops → may cause errors
    Must design with margins → reduces max frequency
    
  Decoupling capacitors:
    Store charge near transistors
    Supply current during fast load transients
    Multiple levels: on-die, on-package, on-board
` + "```" + ``,
					CodeExamples: `// Power and thermal modeling
package main

import (
    "fmt"
    "math"
)

// CMOS power model
type PowerModel struct {
    activityFactor float64 // α
    capacitancePF  float64 // Total switching capacitance (picofarads)
    voltage        float64 // Supply voltage
    frequencyGHz   float64 // Clock frequency
    leakageW       float64 // Static/leakage power
}

func (p PowerModel) DynamicPower() float64 {
    return p.activityFactor * p.capacitancePF * 1e-12 *
        p.voltage * p.voltage * p.frequencyGHz * 1e9
}

func (p PowerModel) TotalPower() float64 {
    return p.DynamicPower() + p.leakageW
}

func (p PowerModel) EnergyPerOp() float64 {
    return p.TotalPower() / (p.frequencyGHz * 1e9) // Joules per cycle
}

// DVFS analysis
type DVFSPoint struct {
    freqGHz float64
    voltage float64
}

func dvfsPower(base PowerModel, point DVFSPoint) float64 {
    vRatio := point.voltage / base.voltage
    fRatio := point.freqGHz / base.frequencyGHz
    dynamic := base.DynamicPower() * vRatio * vRatio * fRatio
    // Leakage increases with voltage (simplified)
    leakage := base.leakageW * vRatio * 1.5
    return dynamic + leakage
}

// Thermal model
type ThermalModel struct {
    thetaJA     float64 // °C/W junction-to-ambient
    tAmbient    float64 // °C
    tMax        float64 // °C maximum junction temp
    thermalCap  float64 // J/°C (thermal capacitance)
}

func (t ThermalModel) SteadyStateTemp(power float64) float64 {
    return t.tAmbient + t.thetaJA*power
}

func (t ThermalModel) MaxPower() float64 {
    return (t.tMax - t.tAmbient) / t.thetaJA
}

func (t ThermalModel) ThrottledPower(requestedPower float64) float64 {
    maxP := t.MaxPower()
    if requestedPower <= maxP {
        return requestedPower
    }
    return maxP
}

// Battery life estimation
type MobileDevice struct {
    batteryWh   float64
    screenW     float64
    baselineW   float64 // Idle power (screen, wifi, etc.)
}

func (d MobileDevice) BatteryLifeHours(cpuPower float64) float64 {
    totalPower := d.baselineW + cpuPower
    return d.batteryWh / totalPower
}

func main() {
    fmt.Println("=== CMOS Power Analysis ===")
    
    baseModel := PowerModel{
        activityFactor: 0.3,
        capacitancePF:  50000, // 50nF total switching capacitance
        voltage:        1.0,
        frequencyGHz:   3.0,
        leakageW:       20,
    }
    
    fmt.Printf("Base configuration: %.1f GHz @ %.2f V\n", baseModel.frequencyGHz, baseModel.voltage)
    fmt.Printf("  Dynamic power:  %.1f W\n", baseModel.DynamicPower())
    fmt.Printf("  Leakage power:  %.1f W\n", baseModel.leakageW)
    fmt.Printf("  Total power:    %.1f W\n", baseModel.TotalPower())
    
    // DVFS analysis
    fmt.Println("\n=== DVFS Power/Performance Trade-off ===")
    dvfsPoints := []DVFSPoint{
        {1.0, 0.65},
        {1.5, 0.72},
        {2.0, 0.80},
        {2.5, 0.88},
        {3.0, 1.00},
        {3.5, 1.10},
        {4.0, 1.20},
        {4.5, 1.30},
        {5.0, 1.40},
    }
    
    fmt.Printf("%-8s │ %-7s │ Power  │ Perf/W │ EDP    │ vs Base\n", "Freq", "Voltage")
    fmt.Println("─────────┼─────────┼────────┼────────┼────────┼────────")
    
    basePower := dvfsPower(baseModel, DVFSPoint{3.0, 1.0})
    
    for _, pt := range dvfsPoints {
        power := dvfsPower(baseModel, pt)
        perfPerWatt := pt.freqGHz / power
        delay := 1.0 / pt.freqGHz
        edp := power * delay * delay
        
        fmt.Printf("%5.1f GHz│ %.2f V  │ %5.1f W│ %6.3f │ %6.4f │ %.1f%% pwr\n",
            pt.freqGHz, pt.voltage, power, perfPerWatt, edp,
            (power/basePower-1)*100)
    }
    
    // Find optimal perf/watt point
    bestPPW := 0.0
    bestPoint := dvfsPoints[0]
    for _, pt := range dvfsPoints {
        power := dvfsPower(baseModel, pt)
        ppw := pt.freqGHz / power
        if ppw > bestPPW {
            bestPPW = ppw
            bestPoint = pt
        }
    }
    fmt.Printf("\nBest perf/watt: %.1f GHz @ %.2fV (%.3f perf/W)\n",
        bestPoint.freqGHz, bestPoint.voltage, bestPPW)
    
    // Thermal analysis
    fmt.Println("\n=== Thermal Analysis ===")
    
    configs := []struct {
        name    string
        thermal ThermalModel
        power   float64
    }{
        {"Gaming desktop (tower cooler)", ThermalModel{0.25, 25, 100, 200}, 150},
        {"Gaming desktop (AIO water)", ThermalModel{0.15, 25, 100, 300}, 250},
        {"Thin laptop (15W)", ThermalModel{1.5, 30, 100, 50}, 15},
        {"Thin laptop (trying 45W)", ThermalModel{1.5, 30, 100, 50}, 45},
        {"Server (datacenter, 25°C)", ThermalModel{0.20, 25, 100, 400}, 350},
        {"Server (hot aisle, 40°C)", ThermalModel{0.20, 40, 100, 400}, 350},
    }
    
    fmt.Printf("%-35s │ Request│ Actual │ T_j   │ Status\n", "Configuration")
    fmt.Println("────────────────────────────────────┼────────┼────────┼───────┼────────")
    
    for _, c := range configs {
        actual := c.thermal.ThrottledPower(c.power)
        temp := c.thermal.SteadyStateTemp(actual)
        maxP := c.thermal.MaxPower()
        status := "OK"
        if c.power > maxP {
            status = fmt.Sprintf("THROTTLED to %.0fW", actual)
        }
        fmt.Printf("%-35s │ %4.0f W │ %4.0f W │ %4.0f°C│ %s\n",
            c.name, c.power, actual, temp, status)
    }

    // Voltage scaling impact
    fmt.Println("\n=== Voltage Scaling Impact ===")
    fmt.Printf("Voltage │ Dynamic Power │ Relative │ Freq (approx)\n")
    fmt.Println("────────┼───────────────┼──────────┼──────────────")
    
    baseV := 1.0
    baseDynamic := baseModel.DynamicPower()
    for v := 0.5; v <= 1.5; v += 0.1 {
        ratio := (v / baseV) * (v / baseV)
        power := baseDynamic * ratio
        freqRatio := v / baseV // Simplified linear
        fmt.Printf("%.2f V  │ %8.1f W    │ %6.1f%% │ %.1f GHz\n",
            v, power, ratio*100, 3.0*freqRatio)
    }

    // Mobile battery life
    fmt.Println("\n=== Mobile Battery Life ===")
    phone := MobileDevice{batteryWh: 18, screenW: 2.0, baselineW: 1.0}
    laptop := MobileDevice{batteryWh: 75, screenW: 5.0, baselineW: 3.0}
    
    fmt.Println("\nSmartphone (18 Wh battery):")
    for _, cpuW := range []float64{0.5, 1.0, 2.0, 5.0, 10.0} {
        life := phone.BatteryLifeHours(cpuW)
        fmt.Printf("  CPU at %4.1f W: %.1f hours\n", cpuW, life)
    }
    
    fmt.Println("\nLaptop (75 Wh battery):")
    for _, cpuW := range []float64{5, 15, 28, 45, 65} {
        life := laptop.BatteryLifeHours(cpuW)
        fmt.Printf("  CPU at %2.0f W: %.1f hours\n", cpuW, life)
    }
    
    // Datacenter power
    fmt.Println("\n=== Datacenter Power Analysis ===")
    type Server struct {
        name    string
        cpuW    float64
        count   int
        gpuW    float64
        gpuCnt  int
        otherW  float64
    }
    
    servers := []Server{
        {"Standard 2S server", 250, 2, 0, 0, 200},
        {"GPU training server", 350, 2, 700, 8, 500},
        {"GPU inference server", 250, 2, 300, 4, 400},
        {"Edge node", 125, 1, 0, 0, 75},
    }
    
    pue := 1.3 // Power Usage Effectiveness (datacenter overhead)
    
    fmt.Printf("%-22s │ IT Power│ With PUE │ $/year @$0.10/kWh\n", "Server Type")
    fmt.Println("───────────────────────┼─────────┼──────────┼──────────────────")
    for _, s := range servers {
        itPower := float64(s.count)*s.cpuW + float64(s.gpuCnt)*s.gpuW + s.otherW
        totalPower := itPower * pue
        annualCost := totalPower * 8760 / 1000 * 0.10 // kWh × $/kWh
        fmt.Printf("%-22s │ %5.0f W │ %6.0f W │ $%,.0f\n",
            s.name, itPower, totalPower, annualCost)
    }
    fmt.Printf("\nPUE = %.1f (%.0f%% overhead for cooling, power distribution)\n",
        pue, (pue-1)*100)
}`,
				},
				{
					Title: "Error Detection and Correction",
					Content: `As transistors shrink and voltages drop, hardware errors become more frequent. Modern systems use extensive error detection and correction mechanisms to maintain reliability.

**Sources of Hardware Errors:**
` + "```" + `
Soft Errors (transient):
  Cosmic rays: High-energy particles strike silicon
    → Flip a bit in memory or register
    → Rate: ~1000 FIT per Mbit (Failure In Time = failures per 10⁹ hours)
    → At sea level: ~1 soft error per day per 128 GB RAM
    → At altitude: 10× higher (less atmosphere shielding)
    → Airplane/satellite: 100-1000× higher
  
  Alpha particles: From radioactive traces in packaging
    → Mostly eliminated with modern packaging
  
  Voltage noise: Temporary voltage droop causes wrong logic value
    → More common at lower voltages (less noise margin)
    → Triggered by sudden workload changes

Hard Errors (permanent):
  Electromigration: Metal atoms pushed by current
    → Wires thin out and break over time
    → Worse at higher temperatures and current density
  
  Hot Carrier Injection: Energetic electrons damage gate oxide
    → Transistor threshold voltage shifts
    → Slower over time
    
  NBTI (Negative Bias Temperature Instability):
    → PMOS transistors degrade under negative bias
    → Threshold voltage increases → slower circuit
    
  Time-Dependent Dielectric Breakdown (TDDB):
    → Gate oxide gradually breaks down
    → Eventual short circuit
` + "```" + `

**Error Correction Codes (ECC):**
` + "```" + `
Parity (simplest):
  Add 1 bit: even parity = XOR of all data bits
  Detects: 1-bit errors
  Corrects: NOTHING (can only detect, not fix)
  Used in: PCIe, CPU caches (some)

SECDED (Single Error Correct, Double Error Detect):
  Hamming code with extra overall parity bit
  For 64-bit data: need 8 check bits (72-bit ECC DIMM)
  Detects: 2-bit errors
  Corrects: 1-bit errors
  Used in: Server DRAM (ECC DIMMs), L2/L3 caches

  How it works:
  Data: 64 bits → Compute H matrix → 8 syndrome bits
  On read: Recompute syndrome
  Syndrome = 0: No error
  Syndrome ≠ 0 with odd weight: 1-bit error, syndrome indicates position
  Syndrome ≠ 0 with even weight: 2-bit error detected (uncorrectable)

Chipkill / SDDC:
  Symbol-based ECC (4-bit or 8-bit symbols)
  Can correct all bits in one DRAM chip failing
  Used in: Server memory (AMD, Intel Xeon)
  
  Motivation: Entire DRAM chip can fail, affecting 4+ bits
  Regular SECDED can only correct 1 bit → insufficient
  Chipkill treats each chip as a symbol → correct entire chip failure

DDR5 On-Die ECC:
  ECC computed inside DRAM chip itself
  128-bit data + 8-bit ECC per internal transfer
  Corrects single-bit errors before data leaves the chip
  Transparent to memory controller
  Catches errors that occur in DRAM array (most common)

RAID for Memory (Memory Mirroring/Sparing):
  Mirror mode: Two DIMMs with identical data
    → Survives complete DIMM failure
    → 50% capacity loss
  
  Spare mode: Reserve one rank as spare
    → Copy data to spare when errors accumulate
    → Lose one rank of capacity

Lockstep Execution:
  Two cores execute same instructions simultaneously
  Compare outputs every cycle
  Any mismatch → error detected
  Used in: Safety-critical systems (automotive, aerospace)
  Cost: 2× the hardware for same throughput
` + "```" + `

**Reliability Metrics:**
` + "```" + `
Failure Rate and MTBF:

MTBF (Mean Time Between Failures):
  MTBF = Total operating time / Number of failures
  
  Consumer SSD: MTBF ~ 1.5 million hours
  Enterprise SSD: MTBF ~ 2 million hours
  HDD: MTBF ~ 1.2 million hours
  Server: MTBF ~ 100,000 hours (many components)
  
  Misleading at system level!
  100 disks × 1.2M hour MTBF → expect 1 failure every 12,000 hours
  = ~every 1.4 years → need redundancy!

FIT (Failures In Time):
  Failures per 10⁹ device-hours
  FIT = 10⁹ / MTBF
  
  Example: DRAM module with 1000 FIT
  → 1 failure per million hours per module
  → With 1000 modules: 1 failure per 1000 hours (~42 days)

Availability:
  A = MTBF / (MTBF + MTTR)
  MTTR = Mean Time To Repair
  
  "Nines" of availability:
    99% (two nines): 3.65 days downtime/year
    99.9% (three nines): 8.76 hours downtime/year
    99.99% (four nines): 52.6 minutes downtime/year
    99.999% (five nines): 5.26 minutes downtime/year
    99.9999% (six nines): 31.5 seconds downtime/year

Bathtub Curve:
  Failure Rate
  │\
  │ \    Early      Useful life      Wear-out
  │  \   failures   (constant rate)  /
  │   \____________________________ /
  │                                /
  └──────────────────────────────── Time
  
  Infant mortality: defective units fail early (burn-in testing)
  Useful life: random failures at constant rate
  Wear-out: accumulated degradation causes increasing failures
` + "```" + ``,
					CodeExamples: `// Reliability and ECC simulation
package main

import (
    "fmt"
    "math"
    "math/rand"
)

// Hamming code (SECDED) simulation
type HammingCode struct {
    dataBits  int
    parityBits int
}

func NewHamming(dataBits int) *HammingCode {
    // Calculate parity bits needed: 2^p >= d + p + 1
    p := 0
    for (1 << p) < dataBits+p+1 {
        p++
    }
    return &HammingCode{dataBits: dataBits, parityBits: p + 1} // +1 for overall parity
}

func (h *HammingCode) Encode(data uint64) uint64 {
    n := h.dataBits + h.parityBits
    encoded := uint64(0)
    
    // Place data bits (skip power-of-2 positions)
    dataIdx := 0
    for i := 1; i <= n; i++ {
        if i&(i-1) != 0 { // Not a power of 2
            if data&(1<<dataIdx) != 0 {
                encoded |= 1 << i
            }
            dataIdx++
            if dataIdx >= h.dataBits { break }
        }
    }
    
    // Calculate parity bits
    for p := 0; (1 << p) <= n; p++ {
        pos := 1 << p
        parity := uint64(0)
        for i := 1; i <= n; i++ {
            if i&pos != 0 {
                parity ^= (encoded >> i) & 1
            }
        }
        if parity != 0 {
            encoded |= 1 << pos
        }
    }
    
    // Overall parity (position 0)
    overallParity := uint64(0)
    for i := 1; i <= n; i++ {
        overallParity ^= (encoded >> i) & 1
    }
    encoded |= overallParity
    
    return encoded
}

func (h *HammingCode) Decode(received uint64) (uint64, int, string) {
    n := h.dataBits + h.parityBits
    
    // Calculate syndrome
    syndrome := 0
    for p := 0; (1 << p) <= n; p++ {
        pos := 1 << p
        parity := uint64(0)
        for i := 1; i <= n; i++ {
            if i&pos != 0 {
                parity ^= (received >> i) & 1
            }
        }
        if parity != 0 {
            syndrome |= pos
        }
    }
    
    // Check overall parity
    overallParity := uint64(0)
    for i := 0; i <= n; i++ {
        overallParity ^= (received >> i) & 1
    }
    
    status := "no error"
    corrected := received
    
    if syndrome == 0 && overallParity == 0 {
        status = "no error"
    } else if syndrome != 0 && overallParity != 0 {
        // Single-bit error, correctable
        corrected ^= 1 << syndrome
        status = fmt.Sprintf("1-bit error at position %d (CORRECTED)", syndrome)
    } else if syndrome != 0 && overallParity == 0 {
        status = "2-bit error DETECTED (uncorrectable)"
    }
    
    // Extract data bits from corrected
    data := uint64(0)
    dataIdx := 0
    for i := 1; i <= n; i++ {
        if i&(i-1) != 0 { // Not power of 2
            if corrected&(1<<i) != 0 {
                data |= 1 << dataIdx
            }
            dataIdx++
            if dataIdx >= h.dataBits { break }
        }
    }
    
    return data, syndrome, status
}

// Reliability calculations
func mtbfToFIT(mtbfHours float64) float64 {
    return 1e9 / mtbfHours
}

func fitToMTBF(fit float64) float64 {
    return 1e9 / fit
}

func systemMTBF(componentMTBFs []float64) float64 {
    totalFailRate := 0.0
    for _, mtbf := range componentMTBFs {
        totalFailRate += 1.0 / mtbf
    }
    return 1.0 / totalFailRate
}

func availability(mtbf, mttr float64) float64 {
    return mtbf / (mtbf + mttr)
}

func nines(avail float64) float64 {
    return -math.Log10(1 - avail)
}

// Soft error rate estimation
func softErrorRate(capacityGB int, fitPerMbit float64) float64 {
    totalMbits := float64(capacityGB) * 8 * 1024 // GB → Mbit
    totalFIT := totalMbits * fitPerMbit
    // Expected errors per year
    hoursPerYear := 8760.0
    return totalFIT * hoursPerYear / 1e9
}

func main() {
    fmt.Println("=== Hamming SECDED Demonstration ===")
    hamming := NewHamming(8)
    
    testData := []uint64{0xA5, 0xFF, 0x42, 0x00}
    
    for _, data := range testData {
        encoded := hamming.Encode(data)
        decoded, _, status := hamming.Decode(encoded)
        fmt.Printf("  Data: 0x%02X → Encoded: 0x%04X → Decoded: 0x%02X (%s)\n",
            data, encoded, decoded, status)
    }
    
    // Inject single-bit error
    fmt.Println("\nSingle-bit error injection:")
    data := uint64(0xA5)
    encoded := hamming.Encode(data)
    for bit := 1; bit <= 12; bit++ {
        corrupted := encoded ^ (1 << bit)
        decoded, syndrome, status := hamming.Decode(corrupted)
        correct := "✓"
        if decoded != data { correct = "✗" }
        fmt.Printf("  Flip bit %2d: syndrome=%d, decoded=0x%02X %s (%s)\n",
            bit, syndrome, decoded, correct, status)
    }
    
    // Inject double-bit error
    fmt.Println("\nDouble-bit error detection:")
    corrupted := encoded ^ (1 << 3) ^ (1 << 7) // Flip 2 bits
    _, syndrome, status := hamming.Decode(corrupted)
    fmt.Printf("  Flip bits 3,7: syndrome=%d (%s)\n", syndrome, status)
    
    // Soft error rates
    fmt.Println("\n=== Soft Error Rates ===")
    fmt.Printf("%-20s │ Capacity │ ECC │ Errors/Year\n", "Configuration")
    fmt.Println("─────────────────────┼──────────┼─────┼────────────")
    
    fitPerMbit := 1000.0 // Typical DRAM FIT rate
    
    configs := []struct{
        name       string
        capacityGB int
        hasECC     bool
    }{
        {"Laptop (no ECC)", 16, false},
        {"Desktop (no ECC)", 32, false},
        {"Server (ECC)", 256, true},
        {"HPC cluster node", 1024, true},
        {"Large cloud server", 2048, true},
    }
    
    for _, c := range configs {
        errPerYear := softErrorRate(c.capacityGB, fitPerMbit)
        eccNote := "exposed"
        if c.hasECC {
            eccNote = "corrected silently"
            // With ECC, only multi-bit errors cause issues
            errPerYear *= 0.001 // ~0.1% become uncorrectable
        }
        fmt.Printf("%-20s │ %4d GB  │ %-3v │ %.2f (%s)\n",
            c.name, c.capacityGB, c.hasECC, errPerYear, eccNote)
    }
    
    // System reliability
    fmt.Println("\n=== System Reliability ===")
    
    type Component struct {
        name  string
        mtbf  float64
        count int
    }
    
    serverComponents := []Component{
        {"CPU", 2000000, 2},
        {"DRAM DIMM", 1500000, 16},
        {"SSD", 2000000, 8},
        {"PSU", 500000, 2},
        {"Fan", 200000, 6},
        {"NIC", 3000000, 2},
        {"Motherboard", 1000000, 1},
    }
    
    mtbfs := []float64{}
    fmt.Printf("%-15s │ Count │ MTBF (hrs) │ FIT    │ Fail/Year\n", "Component")
    fmt.Println("────────────────┼───────┼────────────┼────────┼──────────")
    
    for _, c := range serverComponents {
        for i := 0; i < c.count; i++ {
            mtbfs = append(mtbfs, c.mtbf)
        }
        fit := mtbfToFIT(c.mtbf) * float64(c.count)
        failPerYear := float64(c.count) * 8760 / c.mtbf
        fmt.Printf("%-15s │ %3d   │ %10.0f │ %6.0f │ %.4f\n",
            c.name, c.count, c.mtbf, fit, failPerYear)
    }
    
    sysMTBF := systemMTBF(mtbfs)
    sysFailPerYear := 8760 / sysMTBF
    
    fmt.Printf("\nSystem MTBF: %.0f hours (%.1f years)\n", sysMTBF, sysMTBF/8760)
    fmt.Printf("Expected failures per year: %.2f\n", sysFailPerYear)
    
    // Availability analysis
    fmt.Println("\n=== Availability Analysis ===")
    mttrHours := []float64{0.5, 1, 4, 24}
    
    fmt.Printf("MTTR    │ Availability  │ Nines │ Downtime/Year\n")
    fmt.Println("────────┼───────────────┼───────┼──────────────")
    for _, mttr := range mttrHours {
        avail := availability(sysMTBF, mttr)
        n := nines(avail)
        downtime := (1 - avail) * 8760 * 60 // minutes per year
        fmt.Printf("%5.1f hr│ %.6f%%  │ %.1f   │ %.1f minutes\n",
            mttr, avail*100, n, downtime)
    }

    // Monte Carlo failure simulation
    fmt.Println("\n=== Monte Carlo Failure Simulation (1000 servers, 1 year) ===")
    serverCount := 1000
    failures := 0
    
    for i := 0; i < serverCount; i++ {
        // Each server: simulate if any component fails in 8760 hours
        serverFailed := false
        for _, c := range serverComponents {
            for j := 0; j < c.count; j++ {
                // Exponential distribution
                ttf := -c.mtbf * math.Log(1-rand.Float64())
                if ttf < 8760 {
                    serverFailed = true
                    break
                }
            }
            if serverFailed { break }
        }
        if serverFailed { failures++ }
    }
    
    fmt.Printf("  Servers: %d, Failures in 1 year: %d (%.1f%%)\n",
        serverCount, failures, float64(failures)/float64(serverCount)*100)
    fmt.Printf("  Expected from calculation: %.0f (%.1f%%)\n",
        float64(serverCount)*sysFailPerYear, sysFailPerYear/1*100)
}`,
				},
			},
		},
	})
}
