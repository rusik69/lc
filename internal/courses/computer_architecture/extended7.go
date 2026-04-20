package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2830,
			Title:       "Embedded Systems and Real-Time Computing",
			Description: "Explore microcontroller architectures, real-time operating systems, interrupt priorities, bare-metal programming, and IoT hardware.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "Microcontroller Architecture",
					Content: `Microcontrollers (MCUs) are complete computers on a single chip, designed for embedded applications. Unlike desktop CPUs optimized for maximum performance, MCUs optimize for low power, small size, and real-time capability.

**MCU vs CPU vs SoC:**
` + "```" + `
Desktop CPU (Intel Core i9):
  - Separate chips: CPU, RAM, SSD, GPU, network
  - 5+ GHz clock, 8-24 cores
  - 100-250W TDP
  - Gigabytes of RAM, terabytes storage
  - General-purpose computing

Microcontroller (STM32F4):
  - ALL on one chip: CPU core, Flash, SRAM, peripherals
  - 168 MHz clock, 1 core
  - ~100 mW active, ~2 µW standby
  - 1 MB Flash, 192 KB SRAM
  - GPIO, ADC, DAC, UART, SPI, I2C, timers, USB, CAN

System-on-Chip (Raspberry Pi RP2040):
  - Multiple CPU cores + peripherals + PIO
  - Higher capability than MCU, lower than desktop CPU
  - 133 MHz, dual-core Cortex-M0+
  - 264 KB SRAM, external flash
  - Unique Programmable I/O (PIO) state machines
` + "```" + `

**ARM Cortex-M Architecture:**
` + "```" + `
ARM Cortex-M Family (most popular MCU cores):

Core      │ Pipeline │ Features              │ Use Case
Cortex-M0 │ 3-stage  │ Thumb-2 subset, tiny  │ Sensors, IoT
Cortex-M0+│ 2-stage  │ + MPU, lower power    │ Ultra-low power
Cortex-M3 │ 3-stage  │ Full Thumb-2, MPU     │ Industrial
Cortex-M4 │ 3-stage  │ + DSP, opt FPU        │ Audio, motor ctrl
Cortex-M7 │ 6-stage  │ + Cache, dual-issue   │ High performance
Cortex-M33│ 3-stage  │ + TrustZone security  │ Secure IoT
Cortex-M55│ 4-stage  │ + MVE (Helium SIMD)   │ ML at the edge
Cortex-M85│ 6-stage  │ + MVE + TrustZone     │ Advanced ML/security

Cortex-M Memory Map (standardized across all vendors):
0x00000000 - 0x1FFFFFFF: Code (Flash)
0x20000000 - 0x3FFFFFFF: SRAM
0x40000000 - 0x5FFFFFFF: Peripherals (APB/AHB)
0x60000000 - 0x9FFFFFFF: External RAM (optional)
0xA0000000 - 0xDFFFFFFF: External Device
0xE0000000 - 0xE00FFFFF: Private Peripheral Bus (NVIC, SysTick, SCB)

Key advantage: Consistent memory map across ALL Cortex-M vendors
  → Code portability between STM32, NXP, Nordic, etc.
` + "```" + `

**MCU Peripheral System:**
` + "```" + `
Common Peripherals (memory-mapped registers):

GPIO (General Purpose I/O):
  - Each pin configurable: input, output, alternate function, analog
  - Input modes: floating, pull-up, pull-down
  - Output modes: push-pull, open-drain
  - Speed: up to 100 MHz toggle rate
  - Typical: 32 pins per GPIO port

ADC (Analog-to-Digital Converter):
  - 12-bit resolution (0-4095 for 0-3.3V)
  - Sample rate: 1-5 MSPS
  - Multiple channels (8-20+ analog inputs)
  - DMA support for continuous sampling

Timers:
  - PWM output (motor control, LED dimming)
  - Input capture (frequency measurement, encoders)
  - Output compare (precise timing events)
  - Watchdog timer (reset if software hangs)

Communication:
  UART: Serial (115200 baud typical, up to 10+ Mbaud)
  SPI:  4-wire, up to 50+ MHz (displays, flash, sensors)
  I2C:  2-wire, 100/400 kHz/1 MHz (sensors, EEPROMs)
  CAN:  Automotive bus, up to 8 Mbps (CAN FD)
  USB:  Device or host, up to 480 Mbps (USB 2.0 HS)

DMA (Direct Memory Access):
  - Transfer data between peripherals and memory WITHOUT CPU
  - Circular buffer mode for continuous ADC sampling
  - Memory-to-memory for fast array copy
  - Frees CPU for computation while data moves
` + "```" + `

**Power Management:**
` + "```" + `
MCU Power Modes (STM32L4 example):
    
    Mode           │ Current  │ Wake Time │ What's Running
    Run (80 MHz)   │ ~26 mA   │ N/A       │ Everything  
    Low-power Run  │ ~6 mA    │ N/A       │ CPU at 2 MHz
    Sleep          │ ~1 mA    │ ~4 µs     │ Peripherals only
    Low-power Sleep│ ~100 µA  │ ~30 µs    │ Limited peripherals
    Stop 0         │ ~6 µA    │ ~12 µs    │ RTC, SRAM retained
    Stop 1         │ ~3 µA    │ ~30 µs    │ RTC, partial SRAM
    Stop 2         │ ~1 µA    │ ~60 µs    │ RTC, minimal SRAM
    Standby        │ ~0.3 µA  │ ~60 µs    │ RTC only
    Shutdown       │ ~0.03 µA │ ~200 µs   │ Nothing (cold boot)

Key strategies:
    - Sleep between events (interrupt-driven design)
    - Clock gating: disable unused peripheral clocks
    - Voltage scaling: lower Vcore when high speed not needed
    - Peripheral-on-demand: only power peripherals when needed
    - Burst processing: run fast, finish quickly, sleep longer
` + "```" + ``,
					CodeExamples: `// Embedded systems concepts simulation in Go
package main

import "fmt"

// GPIO simulation
type GPIOPin struct {
    port     string
    pin      int
    mode     string // "input", "output", "alt", "analog"
    pullMode string // "none", "pullup", "pulldown"
    state    bool
}

type GPIOPort struct {
    name  string
    pins  [16]GPIOPin
    odr   uint16 // Output Data Register
    idr   uint16 // Input Data Register
}

func NewGPIOPort(name string) *GPIOPort {
    port := &GPIOPort{name: name}
    for i := range port.pins {
        port.pins[i] = GPIOPin{port: name, pin: i, mode: "input"}
    }
    return port
}

func (p *GPIOPort) SetMode(pin int, mode string) {
    p.pins[pin].mode = mode
}

func (p *GPIOPort) Write(pin int, high bool) {
    if p.pins[pin].mode != "output" {
        return
    }
    p.pins[pin].state = high
    if high {
        p.odr |= 1 << pin
    } else {
        p.odr &^= 1 << pin
    }
}

func (p *GPIOPort) Read(pin int) bool {
    return p.idr&(1<<pin) != 0
}

// ADC simulation
type ADC struct {
    resolution int
    channels   []float64 // Voltage on each channel (0-3.3V)
    vref       float64
}

func NewADC(channels int) *ADC {
    return &ADC{
        resolution: 12,
        channels:   make([]float64, channels),
        vref:       3.3,
    }
}

func (a *ADC) Read(channel int) uint16 {
    if channel >= len(a.channels) { return 0 }
    maxVal := (1 << a.resolution) - 1
    ratio := a.channels[channel] / a.vref
    if ratio > 1.0 { ratio = 1.0 }
    if ratio < 0.0 { ratio = 0.0 }
    return uint16(ratio * float64(maxVal))
}

func (a *ADC) ToVoltage(raw uint16) float64 {
    maxVal := float64((1 << a.resolution) - 1)
    return float64(raw) / maxVal * a.vref
}

// Timer/PWM simulation
type Timer struct {
    prescaler  uint16
    period     uint16
    counter    uint16
    pwmDuty    [4]uint16 // 4 PWM channels
    frequency  float64   // Hz
    clockMHz   float64
}

func NewTimer(clockMHz float64) *Timer {
    return &Timer{clockMHz: clockMHz}
}

func (t *Timer) ConfigurePWM(prescaler, period uint16) {
    t.prescaler = prescaler
    t.period = period
    t.frequency = t.clockMHz * 1e6 / float64(prescaler+1) / float64(period+1)
}

func (t *Timer) SetDuty(channel int, duty uint16) {
    if channel < 4 {
        t.pwmDuty[channel] = duty
    }
}

func (t *Timer) DutyPercent(channel int) float64 {
    if t.period == 0 { return 0 }
    return float64(t.pwmDuty[channel]) / float64(t.period+1) * 100
}

// Power consumption estimator
type PowerMode struct {
    name       string
    currentUA  float64
    wakeTimeUS float64
    features   string
}

func estimateBatteryLife(batteryMAh float64, mode PowerMode, dutyCyclePercent float64,
    activeCurrentMA float64) float64 {
    avgCurrentMA := activeCurrentMA*dutyCyclePercent/100 +
        (mode.currentUA/1000)*(100-dutyCyclePercent)/100
    return batteryMAh / avgCurrentMA // Hours
}

func main() {
    // GPIO demo
    fmt.Println("=== GPIO Simulation ===")
    gpioa := NewGPIOPort("GPIOA")
    gpioa.SetMode(5, "output") // LED on PA5
    gpioa.SetMode(0, "input")  // Button on PA0

    gpioa.Write(5, true)
    fmt.Printf("GPIO %s Pin %d: mode=%s state=%v\n",
        gpioa.name, 5, gpioa.pins[5].mode, gpioa.pins[5].state)
    fmt.Printf("ODR register: 0x%04X\n", gpioa.odr)

    // ADC demo  
    fmt.Println("\n=== ADC Simulation ===")
    adc := NewADC(8)
    adc.channels[0] = 1.65  // Half of Vref (e.g., potentiometer at 50%)
    adc.channels[1] = 0.75  // Temperature sensor
    adc.channels[4] = 3.1   // Battery voltage divider

    for _, ch := range []int{0, 1, 4} {
        raw := adc.Read(ch)
        voltage := adc.ToVoltage(raw)
        fmt.Printf("ADC Ch%d: raw=%4d (0x%03X) → %.3fV (actual: %.3fV)\n",
            ch, raw, raw, voltage, adc.channels[ch])
    }

    // Timer/PWM demo
    fmt.Println("\n=== Timer/PWM Simulation ===")
    tim := NewTimer(168) // 168 MHz clock (STM32F4)
    
    // 50 Hz servo control (20ms period)
    tim.ConfigurePWM(167, 19999) // 168MHz/168/20000 = 50 Hz
    tim.SetDuty(0, 1000)  // 1ms pulse = 0 degrees
    tim.SetDuty(1, 1500)  // 1.5ms pulse = 90 degrees
    tim.SetDuty(2, 2000)  // 2ms pulse = 180 degrees
    
    fmt.Printf("PWM Frequency: %.1f Hz\n", tim.frequency)
    for ch := 0; ch < 3; ch++ {
        fmt.Printf("Channel %d: duty=%d (%.1f%%) → pulse=%.1fms\n",
            ch, tim.pwmDuty[ch], tim.DutyPercent(ch),
            float64(tim.pwmDuty[ch])/tim.frequency/10)
    }

    // Power estimation
    fmt.Println("\n=== Battery Life Estimation ===")
    modes := []PowerMode{
        {"Run (80MHz)", 26000, 0, "Everything"},
        {"Sleep", 1000, 4, "Peripherals"},
        {"Stop 2", 1, 60, "RTC only"},
        {"Shutdown", 0.03, 200, "Nothing"},
    }

    batteryMAh := 250.0 // CR2032 coin cell
    fmt.Printf("Battery: %.0f mAh CR2032\n\n", batteryMAh)

    for _, mode := range modes {
        // Scenario: wake every 1 second, process for 10ms
        life := estimateBatteryLife(batteryMAh, mode, 1.0, 26.0)
        fmt.Printf("%-15s (%.1f µA sleep): Battery life = ", mode.name, mode.currentUA)
        if life > 8760 {
            fmt.Printf("%.1f years\n", life/8760)
        } else if life > 24 {
            fmt.Printf("%.0f days\n", life/24)
        } else {
            fmt.Printf("%.1f hours\n", life)
        }
    }
}`,
				},
				{
					Title: "Real-Time Operating Systems (RTOS)",
					Content: `Real-time systems must respond to events within guaranteed time bounds. An RTOS provides task scheduling, synchronization, and resource management with deterministic timing.

**Hard vs Soft Real-Time:**
` + "```" + `
Hard Real-Time:
  - Missing a deadline = system failure
  - Example: airbag deployment (must fire within 15ms of impact)
  - Example: anti-lock braking (must respond within 5ms)
  - Example: cardiac pacemaker (pulse timing critical)
  Guarantee: WORST-CASE response time is bounded

Soft Real-Time:
  - Missing a deadline = degraded performance (not failure)
  - Example: video playback (dropped frames, but system continues)
  - Example: network packet processing (retransmit on timeout)
  Guarantee: AVERAGE response time is good; occasional misses OK

Firm Real-Time:
  - Missing deadline = result is useless, but no catastrophe
  - Example: radar tracking (stale data is worthless)
  Guarantee: Late results are discarded, not used

Important: "Real-time" does not mean "fast"!
  It means "predictable" and "deterministic."
  A 1-second response that's GUARANTEED is real-time.
  A 1-nanosecond response that MIGHT take 10 seconds is NOT.
` + "```" + `

**RTOS Task Scheduling:**
` + "```" + `
Preemptive Priority-Based Scheduling:
  Each task has a priority (higher = more important)
  Running task is preempted if higher-priority task becomes ready

  Task Priorities:
    Emergency stop:    Priority 7 (highest)
    Sensor reading:    Priority 5
    Motor control:     Priority 4
    Communication:     Priority 3
    Display update:    Priority 2
    Logging:           Priority 1
    Idle:              Priority 0 (lowest)

  Time →  0    1    2    3    4    5    6    7    8
  Task7:  ████                     ██
  Task5:       ████          ████       ████
  Task3:            ████████                ████████
  Task1:                                              (starved!)
  
  Higher-priority tasks always run first.
  Lower-priority tasks only run when no higher task is ready.

Rate Monotonic Scheduling (RMS):
  - Static priorities based on task period
  - Shorter period → higher priority
  - Optimal among fixed-priority algorithms
  
  Schedulability test (Liu & Layland):
    N tasks schedulable if:
    Σ(Ci/Ti) ≤ N × (2^(1/N) - 1)
    
    Where Ci = worst-case execution time, Ti = period
    For 1 task: utilization ≤ 100%
    For 2 tasks: ≤ 82.8%
    For 3 tasks: ≤ 78.0%
    For ∞ tasks: ≤ 69.3% (≈ ln(2))

Earliest Deadline First (EDF):
  - Dynamic priorities based on closest deadline
  - Optimal: can schedule any feasible taskset
  - Schedulable if Σ(Ci/Ti) ≤ 1.0 (100% utilization!)
  - More overhead than RMS (priority changes at runtime)
` + "```" + `

**Priority Inversion and Solutions:**
` + "```" + `
Priority Inversion: A high-priority task is blocked by a low-priority task.

Scenario:
  Task H (high priority) needs mutex M
  Task L (low priority) holds mutex M
  Task M (medium priority) preempts Task L
  
  Result: H waits for L, but L can't run because M is running!
  H effectively runs at L's priority → INVERSION!
  
  Time → 0  1  2  3  4  5  6  7  8  9
  Task H:         ██ block──────────── RUN
  Task M:            ████████████
  Task L: ██████ ██                ██ release █RUN

  Mars Pathfinder (1997): Priority inversion caused system resets!
  Fix: Priority inheritance

Priority Inheritance Protocol:
  When high-priority task H blocks on mutex held by L:
  L temporarily inherits H's priority
  → L cannot be preempted by M
  → L finishes quickly, releases mutex
  → H runs immediately

  Time → 0  1  2  3  4  5  6  7  8
  Task H:         ██ block───── RUN████
  Task M:                           ████
  Task L: ██████ ██(elevated)██ rel

Priority Ceiling Protocol:
  Each mutex has a "ceiling" = highest priority of any task that uses it
  When a task locks mutex, its priority is raised to the ceiling
  → Prevents deadlock AND priority inversion
  → No need to detect inversion at runtime
` + "```" + `

**Common RTOS Features:**
` + "```" + `
Popular RTOS options:
  FreeRTOS:     Open source, most popular for MCUs
  Zephyr:       Linux Foundation, modern, security-focused
  RTEMS:        NASA-grade, space-qualified
  VxWorks:      Commercial, aerospace/defense (Mars rovers)
  ThreadX:      Azure RTOS, certified for safety (IEC 61508)
  QNX:          Microkernel, automotive (BlackBerry)
  
FreeRTOS task states:
  Running → Blocked (waiting for event)
          → Ready (preempted by higher priority)
          → Suspended (explicitly suspended)
  
Synchronization primitives:
  Binary Semaphore:  Signal between tasks (1 or 0)
  Counting Semaphore: Track multiple resources
  Mutex:             Mutual exclusion with priority inheritance
  Event Groups:      Wait for combination of events
  Message Queue:     Pass data between tasks (mailbox)
  Stream Buffer:     Byte-oriented data passing
  
Timing services:
  vTaskDelay():      Sleep for N ticks
  vTaskDelayUntil(): Sleep until absolute time (periodic tasks)
  Software Timers:   One-shot or auto-reload callbacks
  
Memory management:
  heap_1: Allocate only (never free) — simplest, deterministic
  heap_2: Allocate and free (no coalescing)
  heap_3: Wraps standard malloc/free with thread safety
  heap_4: Coalescing free blocks — most flexible
  heap_5: Multiple non-contiguous memory regions
` + "```" + ``,
					CodeExamples: `// RTOS concepts simulation in Go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Task representation
type RTOSTask struct {
    name     string
    priority int
    period   time.Duration
    wcet     time.Duration // Worst-case execution time
    fn       func()
    state    string
}

// Priority-based scheduler
type RTOSScheduler struct {
    tasks     []*RTOSTask
    mu        sync.Mutex
    running   *RTOSTask
    tickCount int
}

func NewScheduler() *RTOSScheduler {
    return &RTOSScheduler{}
}

func (s *RTOSScheduler) AddTask(t *RTOSTask) {
    t.state = "ready"
    s.tasks = append(s.tasks, t)
}

func (s *RTOSScheduler) Schedule() *RTOSTask {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    var best *RTOSTask
    for _, t := range s.tasks {
        if t.state != "ready" { continue }
        if best == nil || t.priority > best.priority {
            best = t
        }
    }
    return best
}

// Rate Monotonic Schedulability test
func rmsSchedulable(tasks []struct{ wcet, period float64 }) bool {
    n := len(tasks)
    utilization := 0.0
    for _, t := range tasks {
        utilization += t.wcet / t.period
    }
    
    // Liu & Layland bound
    import_math_pow := func(base, exp float64) float64 {
        result := 1.0
        for i := 0; i < int(exp); i++ { result *= base }
        return result // Simplified
    }
    _ = import_math_pow
    
    // Use simpler check: n * (2^(1/n) - 1)
    bound := 0.0
    switch n {
    case 1: bound = 1.0
    case 2: bound = 0.828
    case 3: bound = 0.780
    case 4: bound = 0.757
    case 5: bound = 0.743
    default: bound = 0.693 // ln(2)
    }
    
    return utilization <= bound
}

// Priority Inheritance Mutex
type PIMutex struct {
    mu        sync.Mutex
    locked    bool
    owner     *RTOSTask
    origPri   int
    waiters   []*RTOSTask
}

func (m *PIMutex) Lock(task *RTOSTask) {
    m.mu.Lock()
    if !m.locked {
        m.locked = true
        m.owner = task
        m.origPri = task.priority
        m.mu.Unlock()
        return
    }
    
    // Priority inheritance: boost owner if waiter has higher priority
    if task.priority > m.owner.priority {
        fmt.Printf("  Priority inheritance: %s (%d) boosts %s (%d → %d)\n",
            task.name, task.priority, m.owner.name, m.owner.priority, task.priority)
        m.owner.priority = task.priority
    }
    m.waiters = append(m.waiters, task)
    m.mu.Unlock()
    
    task.state = "blocked"
    fmt.Printf("  %s BLOCKED on mutex (waiting for %s)\n", task.name, m.owner.name)
}

func (m *PIMutex) Unlock(task *RTOSTask) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if m.owner != task { return }
    
    // Restore original priority
    if task.priority != m.origPri {
        fmt.Printf("  %s priority restored (%d → %d)\n",
            task.name, task.priority, m.origPri)
        task.priority = m.origPri
    }
    
    m.locked = false
    m.owner = nil
    
    // Wake highest-priority waiter
    if len(m.waiters) > 0 {
        bestIdx := 0
        for i, w := range m.waiters {
            if w.priority > m.waiters[bestIdx].priority { bestIdx = i }
        }
        waiter := m.waiters[bestIdx]
        m.waiters = append(m.waiters[:bestIdx], m.waiters[bestIdx+1:]...)
        waiter.state = "ready"
        m.locked = true
        m.owner = waiter
        m.origPri = waiter.priority
        fmt.Printf("  Mutex granted to %s\n", waiter.name)
    }
}

func main() {
    fmt.Println("=== Rate Monotonic Schedulability ===")
    
    taskSets := []struct{
        name  string
        tasks []struct{ wcet, period float64 }
    }{
        {"Feasible set", []struct{ wcet, period float64 }{
            {1, 5}, {2, 10}, {3, 20},
        }},
        {"Borderline set", []struct{ wcet, period float64 }{
            {2, 5}, {3, 10}, {4, 20},
        }},
        {"Infeasible set", []struct{ wcet, period float64 }{
            {3, 5}, {3, 10}, {3, 15},
        }},
    }

    for _, ts := range taskSets {
        util := 0.0
        for _, t := range ts.tasks {
            util += t.wcet / t.period
        }
        feasible := rmsSchedulable(ts.tasks)
        fmt.Printf("\n%s:\n", ts.name)
        for i, t := range ts.tasks {
            fmt.Printf("  Task %d: WCET=%.0f, Period=%.0f, Util=%.1f%%\n",
                i+1, t.wcet, t.period, t.wcet/t.period*100)
        }
        fmt.Printf("  Total utilization: %.1f%%, RMS feasible: %v\n",
            util*100, feasible)
    }

    // Priority Inversion demo
    fmt.Println("\n=== Priority Inversion Demo ===")
    
    low := &RTOSTask{name: "TaskL", priority: 1, state: "running"}
    med := &RTOSTask{name: "TaskM", priority: 3, state: "ready"}
    high := &RTOSTask{name: "TaskH", priority: 5, state: "ready"}

    mutex := &PIMutex{}
    
    fmt.Println("1. TaskL acquires mutex")
    mutex.Lock(low)
    
    fmt.Println("2. TaskH tries to acquire mutex")
    mutex.Lock(high)  // Triggers priority inheritance
    
    fmt.Println("3. TaskL completes and releases mutex")
    mutex.Unlock(low)
    
    fmt.Printf("\nFinal: TaskH priority=%d, TaskL priority=%d, TaskM priority=%d\n",
        high.priority, low.priority, med.priority)

    // Task timing demo
    fmt.Println("\n=== Periodic Task Timing ===")
    type PeriodicTask struct {
        name    string
        period  int // ms
        wcet    int // ms
    }
    periodicTasks := []PeriodicTask{
        {"Sensor Read", 10, 2},
        {"Control Loop", 20, 5},
        {"Display Update", 100, 15},
        {"Logging", 1000, 50},
    }
    
    fmt.Printf("%-16s │ Period │ WCET │ CPU%%  │ Priority (RMS)\n", "Task")
    fmt.Println("─────────────────┼────────┼──────┼───────┼──────────────")
    totalUtil := 0.0
    for i, t := range periodicTasks {
        util := float64(t.wcet) / float64(t.period) * 100
        totalUtil += util
        fmt.Printf("%-16s │ %4dms │ %3dms│ %5.1f%%│ %d (shorter period = higher)\n",
            t.name, t.period, t.wcet, util, len(periodicTasks)-i)
    }
    fmt.Printf("─────────────────┼────────┼──────┼───────┤\n")
    fmt.Printf("Total CPU Usage  │        │      │ %5.1f%%│\n", totalUtil)
}`,
				},
				{
					Title: "Bare-Metal and IoT Hardware",
					Content: `Bare-metal programming runs directly on hardware without an OS. Understanding this level gives insight into what RTOS and OS kernels actually do, and is essential for IoT device development.

**Boot Process:**
` + "```" + `
Cortex-M Boot Sequence:
1. Power-on reset
2. CPU reads initial SP from address 0x00000000
3. CPU reads Reset Vector from address 0x00000004
4. CPU jumps to Reset Handler
5. Reset Handler:
   a. Copy .data section from Flash to SRAM (initialized globals)
   b. Zero .bss section in SRAM (uninitialized globals)
   c. Initialize clock system (PLL, bus dividers)
   d. Call SystemInit() (vendor-specific setup)
   e. Call main()

Vector Table (first 16 entries are ARM-defined):
┌──────────┬───────────────────────────┐
│ Offset   │ Handler                   │
├──────────┼───────────────────────────┤
│ 0x00     │ Initial Stack Pointer     │
│ 0x04     │ Reset Handler             │
│ 0x08     │ NMI Handler               │
│ 0x0C     │ HardFault Handler         │
│ 0x10     │ MemManage Handler         │
│ 0x14     │ BusFault Handler          │
│ 0x18     │ UsageFault Handler        │
│ 0x2C     │ SVCall Handler            │
│ 0x38     │ PendSV Handler            │
│ 0x3C     │ SysTick Handler           │
│ 0x40+    │ Device-specific IRQs      │
└──────────┴───────────────────────────┘

NVIC (Nested Vectored Interrupt Controller):
  - Up to 240 external interrupts
  - 8-256 priority levels (configurable)
  - Tail-chaining: back-to-back interrupts without full context save
  - Late arrival: higher priority IRQ can preempt during context save
` + "```" + `

**Linker Script:**
` + "```" + `
The linker script defines memory layout for bare-metal programs:

MEMORY {
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 192K
}

SECTIONS {
    .text : {           /* Code section */
        KEEP(*(.isr_vector))  /* Vector table at start of flash */
        *(.text*)             /* All code */
        *(.rodata*)           /* Read-only data (const) */
    } > FLASH

    .data : {           /* Initialized data */
        _sdata = .;
        *(.data*)
        _edata = .;
    } > SRAM AT > FLASH /* Lives in SRAM, loaded from Flash */
    _sidata = LOADADDR(.data);

    .bss : {            /* Uninitialized data (zeroed) */
        _sbss = .;
        *(.bss*)
        _ebss = .;
    } > SRAM

    /* Stack at end of SRAM */
    _estack = ORIGIN(SRAM) + LENGTH(SRAM);
}
` + "```" + `

**IoT Communication Protocols:**
` + "```" + `
Short-Range Wireless:
  BLE (Bluetooth Low Energy):
    Range: 10-100m, Data rate: 1-2 Mbps
    Power: ~10 mA TX, ~1 µA sleep
    Use: Wearables, beacons, sensors
    
  Wi-Fi (ESP32):
    Range: 30-100m, Data rate: 54-600 Mbps
    Power: ~120 mA TX, ~10 µA sleep
    Use: Smart home, camera, rich IoT
    
  Zigbee/Thread (802.15.4):
    Range: 10-100m, Data rate: 250 kbps
    Power: ~20 mA TX, ~1 µA sleep
    Use: Mesh networks, home automation, lighting

Long-Range LPWAN:
  LoRa/LoRaWAN:
    Range: 2-15 km (urban), 30+ km (rural)
    Data rate: 0.3-50 kbps
    Power: ~40 mA TX, ~1 µA sleep
    Use: Agriculture, city monitoring, asset tracking
    
  NB-IoT (cellular):
    Range: 10+ km (cell towers)
    Data rate: ~100 kbps
    Power: ~200 mA TX, ~3 µA sleep
    Use: Smart meters, industrial monitoring

Protocol Stack for IoT:
  Application:  MQTT, CoAP, HTTP
  Transport:    TCP, UDP, DTLS
  Network:      IPv6, 6LoWPAN
  Data Link:    BLE, 802.15.4, LoRa
  Physical:     Radio
` + "```" + `

**Hardware Security:**
` + "```" + `
ARM TrustZone (Cortex-M33+):
  Divides system into Secure and Non-Secure worlds
  
  Secure World:           │  Non-Secure World:
  - Crypto keys           │  - Application code
  - Secure boot           │  - User interface
  - Attestation           │  - Network stack
  - Trusted execution     │  - Non-critical tasks
                          │
  Transition via: NSC (Non-Secure Callable) functions
  Access controlled by: SAU (Security Attribution Unit)

Secure Boot Chain:
  ROM Bootloader → verifies 1st stage bootloader signature
  1st Stage BL   → verifies 2nd stage/firmware signature
  Firmware       → verifies application integrity
  
  Each stage MUST verify the next before executing
  Broken chain = boot failure (not malicious code execution)

Hardware Crypto Accelerators:
  AES engine:     10-100x faster than software AES
  SHA engine:     Hardware hash acceleration
  TRNG:           True Random Number Generator (entropy from hardware noise)
  PKA:            Public Key Accelerator (RSA, ECC)
  Key storage:    OTP (One-Time Programmable) fuses for unique keys
` + "```" + ``,
					CodeExamples: `// Bare-metal concepts simulation
package main

import "fmt"

// Simulated vector table
type VectorTable struct {
    initialSP     uint32
    resetHandler  func()
    nmiHandler    func()
    hardFault     func()
    handlers      [240]func()
}

// Memory regions (simulated)
type MemoryMap struct {
    flash   [1024 * 1024]byte  // 1MB Flash
    sram    [192 * 1024]byte   // 192KB SRAM
    periph  map[uint32]uint32  // Peripheral registers
}

// Simulated register-level GPIO
type GPIO_TypeDef struct {
    MODER   uint32  // Mode register
    OTYPER  uint32  // Output type register
    OSPEEDR uint32  // Output speed register
    PUPDR   uint32  // Pull-up/pull-down register
    IDR     uint32  // Input data register
    ODR     uint32  // Output data register
    BSRR    uint32  // Bit set/reset register
}

func (g *GPIO_TypeDef) SetPinOutput(pin int) {
    g.MODER &= ^(uint32(3) << (pin * 2))  // Clear mode bits
    g.MODER |= uint32(1) << (pin * 2)      // Set output mode (01)
}

func (g *GPIO_TypeDef) SetPin(pin int) {
    g.BSRR = 1 << pin  // Set bit (lower 16 bits)
    g.ODR |= 1 << pin
}

func (g *GPIO_TypeDef) ResetPin(pin int) {
    g.BSRR = 1 << (pin + 16)  // Reset bit (upper 16 bits)
    g.ODR &= ^(uint32(1) << pin)
}

func (g *GPIO_TypeDef) ReadPin(pin int) bool {
    return g.IDR&(uint32(1)<<pin) != 0
}

// Startup code simulation
type StartupCode struct {
    flashDataStart uint32
    sramDataStart  uint32
    sramDataEnd    uint32
    sramBssStart   uint32
    sramBssEnd     uint32
    stackTop       uint32
}

func (s *StartupCode) CopyData() {
    size := s.sramDataEnd - s.sramDataStart
    fmt.Printf("  Copying .data: 0x%08X → 0x%08X (%d bytes)\n",
        s.flashDataStart, s.sramDataStart, size)
}

func (s *StartupCode) ZeroBss() {
    size := s.sramBssEnd - s.sramBssStart
    fmt.Printf("  Zeroing .bss: 0x%08X - 0x%08X (%d bytes)\n",
        s.sramBssStart, s.sramBssEnd, size)
}

func (s *StartupCode) InitClocks() {
    fmt.Println("  Configuring PLL: HSE 8MHz × 21 = 168 MHz")
    fmt.Println("  AHB: 168 MHz, APB1: 42 MHz, APB2: 84 MHz")
}

// IoT device power budget calculator
type IoTDevice struct {
    name          string
    txCurrentMA   float64
    rxCurrentMA   float64
    sleepCurrentUA float64
    txTimeMS      float64
    rxTimeMS      float64
    intervalSec   float64
    batteryMAh    float64
}

func (d *IoTDevice) AverageCurrentMA() float64 {
    cycleMS := d.intervalSec * 1000
    activeMA := (d.txCurrentMA*d.txTimeMS + d.rxCurrentMA*d.rxTimeMS) / cycleMS
    sleepMA := d.sleepCurrentUA / 1000 * (cycleMS - d.txTimeMS - d.rxTimeMS) / cycleMS
    return activeMA + sleepMA
}

func (d *IoTDevice) BatteryLifeYears() float64 {
    hours := d.batteryMAh / d.AverageCurrentMA()
    return hours / 8760
}

func main() {
    // Boot sequence simulation
    fmt.Println("=== Cortex-M Boot Sequence ===")
    startup := &StartupCode{
        flashDataStart: 0x08010000,
        sramDataStart:  0x20000000,
        sramDataEnd:    0x20001000,
        sramBssStart:   0x20001000,
        sramBssEnd:     0x20003000,
        stackTop:       0x20030000,
    }

    fmt.Printf("1. Read initial SP: 0x%08X\n", startup.stackTop)
    fmt.Println("2. Jump to Reset Handler")
    startup.CopyData()
    startup.ZeroBss()
    startup.InitClocks()
    fmt.Println("  Calling main()...")

    // GPIO register-level access
    fmt.Println("\n=== Register-Level GPIO ===")
    gpioa := &GPIO_TypeDef{}
    
    // Configure PA5 as output (LED)
    gpioa.SetPinOutput(5)
    fmt.Printf("GPIOA->MODER = 0x%08X (PA5 = output)\n", gpioa.MODER)
    
    // Toggle LED
    gpioa.SetPin(5)
    fmt.Printf("GPIOA->ODR = 0x%08X (PA5 HIGH)\n", gpioa.ODR)
    gpioa.ResetPin(5)
    fmt.Printf("GPIOA->ODR = 0x%08X (PA5 LOW)\n", gpioa.ODR)

    // IoT power budget
    fmt.Println("\n=== IoT Device Battery Life ===")
    devices := []IoTDevice{
        {
            name: "BLE Sensor (1 min interval)",
            txCurrentMA: 10, rxCurrentMA: 12, sleepCurrentUA: 1.5,
            txTimeMS: 5, rxTimeMS: 3, intervalSec: 60,
            batteryMAh: 230, // CR2032
        },
        {
            name: "LoRa Tracker (15 min interval)",
            txCurrentMA: 40, rxCurrentMA: 12, sleepCurrentUA: 1,
            txTimeMS: 50, rxTimeMS: 100, intervalSec: 900,
            batteryMAh: 2600, // 18650
        },
        {
            name: "Wi-Fi Camera (always on)",
            txCurrentMA: 120, rxCurrentMA: 80, sleepCurrentUA: 10,
            txTimeMS: 500, rxTimeMS: 300, intervalSec: 1,
            batteryMAh: 5000, // Large LiPo
        },
    }

    for _, d := range devices {
        avgMA := d.AverageCurrentMA()
        life := d.BatteryLifeYears()
        fmt.Printf("\n%s:\n", d.name)
        fmt.Printf("  Average current: %.3f mA\n", avgMA)
        fmt.Printf("  Battery: %.0f mAh\n", d.batteryMAh)
        if life >= 1 {
            fmt.Printf("  Battery life: %.1f years\n", life)
        } else {
            fmt.Printf("  Battery life: %.0f days\n", life*365)
        }
    }
}`,
				},
			},
		},
	})
}
