package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2822,
			Title:       "Digital Logic Fundamentals",
			Description: "Deep dive into combinational and sequential logic circuits, Boolean algebra, logic gates, and the building blocks of all digital systems.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Boolean Algebra and Logic Gates",
					Content: `All digital computers are built from simple logic gates that implement Boolean algebra. Understanding these fundamentals reveals how transistors become computers.

**Boolean Algebra Axioms:**
` + "```" + `
Identity:       A + 0 = A        A · 1 = A
Null:           A + 1 = 1        A · 0 = 0
Idempotent:     A + A = A        A · A = A
Inverse:        A + A' = 1       A · A' = 0
Commutative:    A + B = B + A    A · B = B · A
Associative:    (A+B)+C = A+(B+C)    (A·B)·C = A·(B·C)
Distributive:   A·(B+C) = A·B+A·C   A+(B·C) = (A+B)·(A+C)
DeMorgan's:     (A·B)' = A'+B'      (A+B)' = A'·B'
` + "```" + `

**The Seven Basic Logic Gates:**
` + "```" + `
AND Gate:           OR Gate:            NOT Gate (Inverter):
A B │ Out          A B │ Out           A │ Out
0 0 │  0           0 0 │  0            0 │  1
0 1 │  0           0 1 │  1            1 │  0
1 0 │  0           1 0 │  1
1 1 │  1           1 1 │  1

NAND Gate:          NOR Gate:           XOR Gate:           XNOR Gate:
A B │ Out          A B │ Out           A B │ Out           A B │ Out
0 0 │  1           0 0 │  1            0 0 │  0            0 0 │  1
0 1 │  1           0 1 │  0            0 1 │  1            0 1 │  0
1 0 │  1           1 0 │  0            1 0 │  1            1 0 │  0
1 1 │  0           1 1 │  0            1 1 │  0            1 1 │  1
` + "```" + `

**NAND as Universal Gate:**
Any logic function can be built using only NAND gates. This is why NAND gates are the foundation of modern VLSI design:
` + "```" + `
NOT from NAND:          AND from NAND:           OR from NAND:
A ─┬─[NAND]─ A'        A ─[NAND]─┬─[NAND]─ A·B  A'─[NAND]─ A+B
   └─┘                  B ─┘      └─┘              B'─┘
                                                    (Apply DeMorgan's)
` + "```" + `

**Transistor Implementation:**
In CMOS technology, each logic gate is built from complementary MOSFET transistors:
- **NMOS** transistor: conducts when gate is HIGH (pulls output LOW)
- **PMOS** transistor: conducts when gate is LOW (pulls output HIGH)

` + "```" + `
CMOS Inverter:                  CMOS NAND Gate:
    VDD                             VDD
     │                               │
  ┌──┴──┐                      ┌──┴──┐ ┌──┴──┐
  │ PMOS │                      │ P1  │ │ P2  │    (Parallel PMOS)
  └──┬──┘                      └──┬──┘ └──┬──┘
     ├──── Output                  ├───┬───┘
  ┌──┴──┐                         ├──── Output
  │ NMOS │                      ┌──┴──┐
  └──┬──┘                      │ N1  │            (Series NMOS)
     │                          └──┬──┘
    GND                         ┌──┴──┐
                                │ N2  │
     Input ──→ Both gates       └──┬──┘
                                   │
                                  GND
     A ──→ P1, N1 gates
     B ──→ P2, N2 gates
` + "```" + `

**Key Insight:** NAND gates require fewer transistors than AND gates (NAND = 4 transistors, AND = NAND + NOT = 6 transistors), which is why real hardware uses NAND-NAND logic rather than AND-OR logic.

**Power Consumption in CMOS:**
- **Static power:** Leakage current when transistors are "off" (grows with smaller process nodes)
- **Dynamic power:** P = α · C · V² · f (switching activity × capacitance × voltage² × frequency)
- Reducing voltage has quadratic effect on power — this is why voltage scaling is so important
- Modern chips use multiple voltage domains and dynamic voltage/frequency scaling (DVFS)`,
					CodeExamples: `// Simulating logic gates in Go
package main

import "fmt"

// Basic gates
func AND(a, b bool) bool  { return a && b }
func OR(a, b bool) bool   { return a || b }
func NOT(a bool) bool     { return !a }
func NAND(a, b bool) bool { return !(a && b) }
func NOR(a, b bool) bool  { return !(a || b) }
func XOR(a, b bool) bool  { return a != b }
func XNOR(a, b bool) bool { return a == b }

// Building AND from NAND only (universal gate proof)
func AND_from_NAND(a, b bool) bool {
    nandResult := NAND(a, b)
    return NAND(nandResult, nandResult) // NOT via NAND
}

// Building OR from NAND only
func OR_from_NAND(a, b bool) bool {
    notA := NAND(a, a)
    notB := NAND(b, b)
    return NAND(notA, notB)
}

// Truth table printer
func printTruthTable(name string, gate func(bool, bool) bool) {
    fmt.Printf("\n%s Truth Table:\n", name)
    fmt.Println("A | B | Out")
    fmt.Println("--+---+----")
    for _, a := range []bool{false, true} {
        for _, b := range []bool{false, true} {
            result := gate(a, b)
            fmt.Printf("%d | %d |  %d\n", boolToInt(a), boolToInt(b), boolToInt(result))
        }
    }
}

func boolToInt(b bool) int {
    if b { return 1 }
    return 0
}

// Boolean expression: F = A'B + AB' (XOR)
func expression(a, b bool) bool {
    return OR(AND(NOT(a), b), AND(a, NOT(b)))
}

func main() {
    printTruthTable("AND", AND)
    printTruthTable("OR", OR)
    printTruthTable("NAND", NAND)
    printTruthTable("XOR", XOR)
    printTruthTable("AND from NAND", AND_from_NAND)
    printTruthTable("OR from NAND", OR_from_NAND)

    // Verify DeMorgan's Theorem
    fmt.Println("\nDeMorgan's Theorem Verification:")
    for _, a := range []bool{false, true} {
        for _, b := range []bool{false, true} {
            lhs := NOT(AND(a, b))    // (A·B)'
            rhs := OR(NOT(a), NOT(b)) // A' + B'
            fmt.Printf("A=%d B=%d: (A·B)'=%d, A'+B'=%d, Equal=%v\n",
                boolToInt(a), boolToInt(b), boolToInt(lhs), boolToInt(rhs), lhs == rhs)
        }
    }
}`,
				},
				{
					Title: "Combinational Logic Circuits",
					Content: `Combinational circuits produce outputs that depend only on current inputs — they have no memory. These are the building blocks of arithmetic units, multiplexers, and decoders.

**Multiplexer (MUX):**
A multiplexer selects one of several input signals based on select lines:
` + "```" + `
2-to-1 MUX:                    4-to-1 MUX:
              ┌─────┐                       ┌─────┐
    I0 ──────│     │                 I0 ───│     │
    I1 ──────│ MUX │── Output       I1 ───│     │
              │     │                 I2 ───│ MUX │── Output
    S  ──────│     │                 I3 ───│     │
              └─────┘                       │     │
                                     S1 S0─│     │
    Output = S'·I0 + S·I1                  └─────┘

    Output = S1'·S0'·I0 + S1'·S0·I1 + S1·S0'·I2 + S1·S0·I3
` + "```" + `

**Any Boolean function can be implemented with a MUX!** A 2^n-to-1 MUX can implement any function of n variables by connecting the truth table outputs to the MUX inputs.

**Demultiplexer (DEMUX) / Decoder:**
` + "```" + `
2-to-4 Decoder:
              ┌──────────┐
    A ───────│          │── D0 = A'·B'
    B ───────│ Decoder  │── D1 = A'·B
              │          │── D2 = A·B'
    EN ──────│          │── D3 = A·B
              └──────────┘

When EN=1, exactly ONE output is HIGH based on A,B inputs.
Used in: memory address decoding, instruction decoding.
` + "```" + `

**Arithmetic Circuits — Half Adder:**
` + "```" + `
Half Adder: Adds two 1-bit numbers
    A  B │ Sum  Carry
    0  0 │  0    0
    0  1 │  1    0
    1  0 │  1    0
    1  1 │  0    1

    Sum   = A XOR B
    Carry = A AND B

         A ──┬──[XOR]── Sum
         B ──┤
             └──[AND]── Carry
` + "```" + `

**Full Adder:**
` + "```" + `
Full Adder: Adds two bits plus carry-in
    A  B  Cin │ Sum  Cout
    0  0   0  │  0    0
    0  0   1  │  1    0
    0  1   0  │  1    0
    0  1   1  │  0    1
    1  0   0  │  1    0
    1  0   1  │  0    1
    1  1   0  │  0    1
    1  1   1  │  1    1

    Sum  = A XOR B XOR Cin
    Cout = (A AND B) OR (Cin AND (A XOR B))

Built from two half adders:
    HA1: A,B → P=A⊕B, G=A·B
    HA2: P,Cin → Sum=P⊕Cin, C2=P·Cin
    Cout = G OR C2
` + "```" + `

**Ripple Carry Adder:**
` + "```" + `
4-bit Ripple Carry Adder:
    A3 B3      A2 B2      A1 B1      A0 B0
     │  │       │  │       │  │       │  │
  ┌──┴──┴──┐ ┌──┴──┴──┐ ┌──┴──┴──┐ ┌──┴──┴──┐
  │  FA3   │ │  FA2   │ │  FA1   │ │  FA0   │
  └──┬──┬──┘ └──┬──┬──┘ └──┬──┬──┘ └──┬──┬──┘
     │  └───Cin─┘  └───Cin─┘  └───Cin─┘  │
    Cout  S3        S2        S1        S0  Cin=0

Problem: Carry "ripples" through all adders.
Delay = N × (full adder delay) for N-bit adder
For a 64-bit adder, this is too slow!
` + "```" + `

**Carry Lookahead Adder (CLA):**
Instead of waiting for carry to ripple, compute all carries in parallel:
` + "```" + `
Generate: Gi = Ai · Bi    (bit i generates a carry)
Propagate: Pi = Ai ⊕ Bi   (bit i propagates incoming carry)

C1 = G0 + P0·C0
C2 = G1 + P1·G0 + P1·P0·C0
C3 = G2 + P2·G1 + P2·P1·G0 + P2·P1·P0·C0
C4 = G3 + P3·G2 + P3·P2·G1 + P3·P2·P1·G0 + P3·P2·P1·P0·C0

Delay is O(1) for 4-bit groups, O(log N) for N bits
64-bit CLA: ~4 gate delays vs ~128 for ripple carry
` + "```" + `

**Comparator:**
` + "```" + `
N-bit Equality Comparator: A == B?
    Equal = (A0 XNOR B0) AND (A1 XNOR B1) AND ... AND (AN-1 XNOR BN-1)

N-bit Magnitude Comparator: A > B?
    Compare bit by bit from MSB:
    If Ai > Bi → A > B (done)
    If Ai < Bi → A < B (done)
    If Ai = Bi → compare next bit
` + "```" + `

**Priority Encoder:**
` + "```" + `
8-to-3 Priority Encoder:
    Input:  D7 D6 D5 D4 D3 D2 D1 D0
    Output: Y2 Y1 Y0 (binary code of highest active input)
    Valid:  V (at least one input active)

    If D7=1 → Y=111 (7)
    If D6=1 → Y=110 (6)  (assuming D7=0)
    ... etc.

    Used in: interrupt controllers (which interrupt has highest priority?)
` + "```" + ``,
					CodeExamples: `// Simulating combinational circuits in Go
package main

import "fmt"

// Half Adder
func halfAdder(a, b bool) (sum, carry bool) {
    sum = a != b    // XOR
    carry = a && b  // AND
    return
}

// Full Adder
func fullAdder(a, b, cin bool) (sum, cout bool) {
    s1, c1 := halfAdder(a, b)
    sum, c2 := halfAdder(s1, cin)
    cout = c1 || c2
    return
}

// N-bit Ripple Carry Adder
func rippleCarryAdder(a, b []bool, cin bool) ([]bool, bool) {
    n := len(a)
    sum := make([]bool, n)
    carry := cin
    for i := 0; i < n; i++ {
        sum[i], carry = fullAdder(a[i], b[i], carry)
    }
    return sum, carry
}

// 4-to-1 Multiplexer
func mux4to1(inputs [4]bool, sel [2]bool) bool {
    idx := 0
    if sel[0] { idx += 1 }
    if sel[1] { idx += 2 }
    return inputs[idx]
}

// 2-to-4 Decoder
func decoder2to4(a, b, en bool) [4]bool {
    var out [4]bool
    if en {
        out[0] = !a && !b
        out[1] = !a && b
        out[2] = a && !b
        out[3] = a && b
    }
    return out
}

// Priority Encoder
func priorityEncoder(inputs [8]bool) (code [3]bool, valid bool) {
    for i := 7; i >= 0; i-- {
        if inputs[i] {
            code[0] = (i & 1) != 0
            code[1] = (i & 2) != 0
            code[2] = (i & 4) != 0
            valid = true
            return
        }
    }
    return code, false
}

// N-bit Comparator
func compare(a, b []bool) (equal, greater bool) {
    n := len(a)
    equal = true
    for i := n - 1; i >= 0; i-- { // MSB first
        if a[i] && !b[i] {
            return false, true
        }
        if !a[i] && b[i] {
            return false, false
        }
    }
    return true, false
}

func boolSliceToInt(bits []bool) int {
    result := 0
    for i, b := range bits {
        if b { result |= 1 << i }
    }
    return result
}

func intToBoolSlice(val, bits int) []bool {
    result := make([]bool, bits)
    for i := range result {
        result[i] = (val & (1 << i)) != 0
    }
    return result
}

func main() {
    // Test ripple carry adder
    a := intToBoolSlice(13, 4)  // 1101
    b := intToBoolSlice(7, 4)   // 0111
    sum, cout := rippleCarryAdder(a, b, false)
    fmt.Printf("13 + 7 = %d (carry=%v)\n", boolSliceToInt(sum), cout) // 20

    // Test decoder
    outputs := decoder2to4(true, false, true)
    fmt.Printf("Decoder(A=1, B=0): %v\n", outputs) // [false false true false]

    // Test priority encoder
    inputs := [8]bool{false, false, true, false, false, true, false, false}
    code, valid := priorityEncoder(inputs)
    fmt.Printf("Priority encoder: code=%d%d%d valid=%v\n",
        boolToInt(code[2]), boolToInt(code[1]), boolToInt(code[0]), valid)
}

func boolToInt(b bool) int {
    if b { return 1 }
    return 0
}`,
				},
				{
					Title: "Sequential Logic: Latches and Flip-Flops",
					Content: `Sequential circuits have memory — their outputs depend on both current inputs and past state. They form the basis of registers, counters, and all stateful elements in processors.

**SR Latch (Set-Reset):**
` + "```" + `
SR Latch from NOR gates:
    S ──[NOR]──┬── Q
         ┌─────┘
         └─────┐
    R ──[NOR]──┴── Q'

Truth Table:
    S  R │ Q    Q'   │ Action
    0  0 │ Q    Q'   │ Hold (no change)
    0  1 │ 0    1    │ Reset
    1  0 │ 1    0    │ Set
    1  1 │ 0    0    │ Invalid! (both outputs 0, unstable)

The invalid state is why we need gated/clocked versions.
` + "```" + `

**Gated D Latch:**
` + "```" + `
D Latch (Level-Sensitive):
    D ────┬──[AND]──S──[SR Latch]── Q
          │                          Q'
    EN ──┤
          │
          └──[NOT]──[AND]──R──┘

When EN=1: Q follows D (transparent)
When EN=0: Q holds previous value (opaque)

Problem: While EN=1, output changes with input
→ Cannot reliably sample data in a pipeline
→ Need edge-triggered flip-flops
` + "```" + `

**D Flip-Flop (Edge-Triggered):**
` + "```" + `
Positive-Edge-Triggered D Flip-Flop:
Built from two D latches (master-slave):

    D ──[D Latch (Master)]──[D Latch (Slave)]── Q
         EN = CLK'              EN = CLK

         CLK ─┘

On rising edge of CLK:
  1. Master was transparent (CLK'=1), captured D
  2. Master becomes opaque (CLK'=0), holds value
  3. Slave becomes transparent (CLK=1), outputs master's value
  4. Slave becomes opaque on next falling edge

Result: Q changes ONLY on the rising edge of CLK
This is the fundamental storage element in synchronous circuits!
` + "```" + `

**Timing Parameters:**
` + "```" + `
         ┌────┐    ┌────┐    ┌────┐
CLK ─────┘    └────┘    └────┘    └────
         ↑              ↑
         │←─ Tsetup ─→│←─ Thold ─→│
              D must be       D must remain
              stable BEFORE   stable AFTER
              clock edge      clock edge

    Tsetup: How early D must be stable before CLK edge
    Thold:  How long D must remain stable after CLK edge
    Tclk-to-q: Delay from CLK edge to Q changing
    
If Tsetup or Thold violated → METASTABILITY
    Q may oscillate or settle to random value!

Maximum Clock Frequency:
    Tclock ≥ Tclk-to-q + Tcombinational + Tsetup
    fmax = 1 / (Tclk-to-q + Tcomb + Tsetup)
` + "```" + `

**Common Flip-Flop Variations:**
` + "```" + `
JK Flip-Flop:             T Flip-Flop (Toggle):
J K │ Q(next)             T │ Q(next)
0 0 │ Q (hold)            0 │ Q (hold)
0 1 │ 0 (reset)           1 │ Q' (toggle)
1 0 │ 1 (set)
1 1 │ Q' (toggle)         T flip-flop = JK with J=K=T

D Flip-Flop with Enable and Reset:
    if RESET:   Q = 0       (asynchronous)
    elif EN:    Q = D       (on clock edge)
    else:       Q = Q       (hold)
` + "```" + `

**Metastability:**
When setup/hold times are violated (e.g., asynchronous input crossing clock domains):
` + "```" + `
Normal:          Metastable:           Resolution:
    ┌──           ───?───              ┌── (eventually settles
    │                │                 │    to 0 or 1, but
────┘            ────┘                 │    may take many
                                   ────┘    clock cycles)

Solution: Synchronizer (2+ flip-flops in series)
    async_input → [FF1] → [FF2] → synchronized_output
                   CLK     CLK
    
    Probability of metastability decreases exponentially with
    each additional flip-flop stage.
    Mean Time Between Failures (MTBF) doubles with each stage.
` + "```" + ``,
					CodeExamples: `// Simulating sequential circuits in Go
package main

import "fmt"

// D Flip-Flop
type DFlipFlop struct {
    q       bool
    master  bool
    prevClk bool
}

func (ff *DFlipFlop) Update(d, clk bool) bool {
    // Detect rising edge
    if clk && !ff.prevClk {
        ff.q = ff.master
    }
    // Master is transparent when clock is low
    if !clk {
        ff.master = d
    }
    ff.prevClk = clk
    return ff.q
}

// JK Flip-Flop
type JKFlipFlop struct {
    q       bool
    prevClk bool
}

func (ff *JKFlipFlop) Update(j, k, clk bool) bool {
    if clk && !ff.prevClk { // Rising edge
        switch {
        case !j && !k: // Hold
        case !j && k:  ff.q = false  // Reset
        case j && !k:  ff.q = true   // Set
        case j && k:   ff.q = !ff.q  // Toggle
        }
    }
    ff.prevClk = clk
    return ff.q
}

// N-bit Register (array of D flip-flops)
type Register struct {
    bits    []DFlipFlop
    n       int
}

func NewRegister(n int) *Register {
    return &Register{bits: make([]DFlipFlop, n), n: n}
}

func (r *Register) Load(data []bool, clk bool) []bool {
    result := make([]bool, r.n)
    for i := 0; i < r.n; i++ {
        result[i] = r.bits[i].Update(data[i], clk)
    }
    return result
}

// 4-bit Counter using T flip-flops
type Counter4 struct {
    bits [4]bool
    prevClk bool
}

func (c *Counter4) Tick(clk bool) [4]bool {
    if clk && !c.prevClk { // Rising edge
        carry := true
        for i := 0; i < 4; i++ {
            if carry {
                c.bits[i] = !c.bits[i]
                carry = !c.bits[i] // Carry if bit went from 1→0
            }
        }
    }
    c.prevClk = clk
    return c.bits
}

func main() {
    // Simulate D flip-flop
    ff := &DFlipFlop{}
    fmt.Println("D Flip-Flop simulation:")
    testData := []struct{ d, clk bool }{
        {true, false}, {true, true}, {false, false},
        {false, true}, {true, false}, {true, true},
    }
    for _, t := range testData {
        q := ff.Update(t.d, t.clk)
        edge := ""
        if t.clk && !ff.prevClk { edge = " ↑" }
        fmt.Printf("D=%v CLK=%v → Q=%v%s\n", t.d, t.clk, q, edge)
    }

    // Simulate 4-bit counter
    fmt.Println("\n4-bit Counter:")
    counter := &Counter4{}
    for i := 0; i < 20; i++ {
        counter.Tick(false) // Low
        bits := counter.Tick(true)  // Rising edge
        val := 0
        for j := 3; j >= 0; j-- {
            val <<= 1
            if bits[j] { val |= 1 }
        }
        fmt.Printf("Count: %04b (%d)\n", val, val)
    }
}`,
				},
				{
					Title: "Finite State Machines",
					Content: `Finite State Machines (FSMs) are the fundamental model for sequential logic design. Every digital controller, protocol handler, and CPU control unit is an FSM.

**Two Types of FSMs:**
` + "```" + `
Moore Machine:                     Mealy Machine:
  Output depends ONLY on state      Output depends on state AND input
  
  ┌───────────────┐                 ┌───────────────┐
  │               │                 │               │
  │   Next State  │                 │   Next State  │
  │     Logic     │←── Input        │     Logic     │←── Input
  │               │                 │               │        │
  └───────┬───────┘                 └───────┬───────┘        │
          │                                 │                │
    ┌─────┴─────┐                     ┌─────┴─────┐         │
    │  State    │                     │  State    │         │
    │ Register  │←── CLK              │ Register  │←── CLK  │
    └─────┬─────┘                     └─────┬─────┘         │
          │                                 │                │
    ┌─────┴─────┐                     ┌─────┴─────┐         │
    │  Output   │                     │  Output   │←────────┘
    │   Logic   │                     │   Logic   │
    └─────┬─────┘                     └─────┬─────┘
          │                                 │
       Output                            Output

Moore: Outputs change synchronously (one cycle after input)
Mealy: Outputs can change asynchronously (same cycle as input)
       → Mealy machines can react faster but may have glitches
` + "```" + `

**Example: Traffic Light Controller (Moore FSM)**
` + "```" + `
States:
  S0: Green for Main Road    (Main=Green, Side=Red)
  S1: Yellow for Main Road   (Main=Yellow, Side=Red)
  S2: Green for Side Road    (Main=Red, Side=Green)
  S3: Yellow for Side Road   (Main=Red, Side=Yellow)

Inputs: timer_expired, sensor (car waiting on side road)

State Diagram:
  S0 ──timer && sensor──→ S1
  S0 ──otherwise─────────→ S0
  S1 ──timer─────────────→ S2
  S2 ──timer─────────────→ S3
  S3 ──timer─────────────→ S0

State Transition Table:
  Current │ Input           │ Next │ Output
  S0      │ timer∧sensor    │ S1   │ Main=G, Side=R
  S0      │ otherwise       │ S0   │ Main=G, Side=R
  S1      │ timer           │ S2   │ Main=Y, Side=R
  S1      │ otherwise       │ S1   │ Main=Y, Side=R
  S2      │ timer           │ S3   │ Main=R, Side=G
  S3      │ timer           │ S0   │ Main=R, Side=Y

State Encoding (2 bits):
  S0 = 00, S1 = 01, S2 = 10, S3 = 11
` + "```" + `

**Example: Sequence Detector (Mealy FSM)**
Detect the sequence "1011" in a serial bit stream:
` + "```" + `
States: IDLE, GOT_1, GOT_10, GOT_101

        Input=0          Input=1
IDLE    → IDLE           → GOT_1
GOT_1   → GOT_10         → GOT_1
GOT_10  → IDLE           → GOT_101
GOT_101 → GOT_10         → GOT_1 (output=1! detected "1011")

                    0              1
          ┌────────────────┐ ┌──────────┐
          │                ↓ │          ↓
        IDLE ──1──→ GOT_1 ──0──→ GOT_10 ──1──→ GOT_101
          ↑                                      │  0│
          │              1/output=1               │  │
          └──────────────────────────────────────┘  │
                                                     ↓
                                                  GOT_10
` + "```" + `

**State Encoding Strategies:**
` + "```" + `
4 states can be encoded as:

Binary:     S0=00, S1=01, S2=10, S3=11
  + Minimum flip-flops (2)
  - Complex next-state logic
  
One-Hot:    S0=0001, S1=0010, S2=0100, S3=1000
  + Simple next-state logic (one gate per transition)
  + Fast (less combinational delay)
  - More flip-flops (N for N states)
  - Common in FPGAs (flip-flops are plentiful)

Gray Code:  S0=00, S1=01, S2=11, S3=10
  + Only one bit changes per transition
  + Reduces glitches and power consumption
  - Only works for sequential state transitions
` + "```" + `

**FSM Optimization:**
` + "```" + `
State Minimization:
  Two states are equivalent if:
    1. Same output (Moore) or same output for all inputs (Mealy)
    2. For every input, they transition to equivalent states

  Implication Table Method:
    1. Mark pairs with different outputs as non-equivalent
    2. For remaining pairs, list implied equivalences
    3. Iterate until no more pairs can be marked
    4. Merge equivalent states

State Assignment:
  - Adjacent states that share transitions → adjacent codes (Gray code)
  - Reduces number of gates in next-state logic
  - Heuristic: minimize the sum of distances between transition pairs
` + "```" + ``,
					CodeExamples: `// FSM implementations in Go
package main

import "fmt"

// Moore FSM: Traffic Light Controller
type TrafficState int
const (
    MainGreen TrafficState = iota
    MainYellow
    SideGreen
    SideYellow
)

type TrafficLight struct {
    state TrafficState
}

type TrafficOutput struct {
    mainLight string
    sideLight string
}

func (tl *TrafficLight) Output() TrafficOutput {
    switch tl.state {
    case MainGreen:  return TrafficOutput{"GREEN", "RED"}
    case MainYellow: return TrafficOutput{"YELLOW", "RED"}
    case SideGreen:  return TrafficOutput{"RED", "GREEN"}
    case SideYellow: return TrafficOutput{"RED", "YELLOW"}
    }
    return TrafficOutput{"RED", "RED"}
}

func (tl *TrafficLight) Tick(timerExpired, carWaiting bool) {
    switch tl.state {
    case MainGreen:
        if timerExpired && carWaiting { tl.state = MainYellow }
    case MainYellow:
        if timerExpired { tl.state = SideGreen }
    case SideGreen:
        if timerExpired { tl.state = SideYellow }
    case SideYellow:
        if timerExpired { tl.state = MainGreen }
    }
}

// Mealy FSM: Sequence detector for "1011"
type SeqState int
const (
    Idle SeqState = iota
    Got1
    Got10
    Got101
)

type SequenceDetector struct {
    state SeqState
}

func (sd *SequenceDetector) Input(bit int) bool {
    detected := false
    switch sd.state {
    case Idle:
        if bit == 1 { sd.state = Got1 } else { sd.state = Idle }
    case Got1:
        if bit == 0 { sd.state = Got10 } else { sd.state = Got1 }
    case Got10:
        if bit == 1 { sd.state = Got101 } else { sd.state = Idle }
    case Got101:
        if bit == 1 {
            detected = true
            sd.state = Got1
        } else {
            sd.state = Got10
        }
    }
    return detected
}

// Generic FSM framework
type FSM[S comparable, I any, O any] struct {
    state      S
    transition func(S, I) S
    output     func(S) O  // Moore output
}

func (fsm *FSM[S, I, O]) Step(input I) O {
    fsm.state = fsm.transition(fsm.state, input)
    return fsm.output(fsm.state)
}

func main() {
    // Traffic light simulation
    tl := &TrafficLight{state: MainGreen}
    events := []struct{ timer, car bool }{
        {false, false}, {false, true}, {true, true},
        {true, false}, {true, false}, {true, false},
    }
    fmt.Println("Traffic Light FSM:")
    for i, e := range events {
        out := tl.Output()
        fmt.Printf("Cycle %d: Main=%s Side=%s\n", i, out.mainLight, out.sideLight)
        tl.Tick(e.timer, e.car)
    }

    // Sequence detector
    sd := &SequenceDetector{state: Idle}
    stream := []int{1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1}
    fmt.Println("\nSequence Detector (looking for 1011):")
    for i, bit := range stream {
        if sd.Input(bit) {
            fmt.Printf("Bit %d: %d → DETECTED at position %d!\n", i, bit, i)
        } else {
            fmt.Printf("Bit %d: %d\n", i, bit)
        }
    }
}`,
				},
				{
					Title: "ALU Design and Arithmetic Units",
					Content: `The Arithmetic Logic Unit (ALU) is the computational heart of every processor. It combines arithmetic operations (add, subtract, multiply) with logic operations (AND, OR, XOR, shift) into a single circuit.

**Simple 1-bit ALU Slice:**
` + "```" + `
1-bit ALU:
                    ┌──────────────────────┐
    A ──────────────│                      │
    B ──────────────│   Operation Select   │
    CarryIn ────────│      (2 bits)        │
                    │                      │
    Op[1:0] ────────│  00: AND             │────── Result
                    │  01: OR              │
                    │  10: ADD             │────── CarryOut
                    │  11: SLT (set less)  │
                    └──────────────────────┘

Internal structure:
    AND unit:  A AND B
    OR unit:   A OR B
    Adder:     Full Adder(A, B, CarryIn) → sum, carry
    
    4-to-1 MUX selects which result to output based on Op
` + "```" + `

**32-bit ALU from 1-bit slices:**
` + "```" + `
    A[31:0]  B[31:0]  Op[1:0]
       │        │        │
    ┌──┴──┐  ┌──┴──┐     │
    │     │  │     │     │
   A31 B31  A30 B30     │      ...    A1 B1    A0 B0
    │   │    │   │      │              │  │     │  │
  ┌─┴───┴─┐┌─┴───┴─┐   │          ┌──┴──┴─┐┌──┴──┴─┐
  │ALU 31 ││ALU 30 │   Op         │ALU 1  ││ALU 0  │
  └─┬──┬──┘└─┬──┬──┘              └─┬──┬──┘└─┬──┬──┘
    │  Cout←Cin  Cout←Cin  ...       │  Cout←Cin│
   R31       R30                    R1       R0  Cin=0 (add)
                                                  Cin=1 (sub)

For SUBTRACT: invert B inputs and set Cin=1 (two's complement)
    A - B = A + NOT(B) + 1
` + "```" + `

**Overflow Detection:**
` + "```" + `
Signed overflow occurs when:
  - Adding two positive numbers gives negative result
  - Adding two negative numbers gives positive result

Detection: Overflow = CarryIn[MSB] XOR CarryOut[MSB]

Examples (4-bit signed, range -8 to +7):
  0111 (+7) + 0001 (+1) = 1000 (-8) → OVERFLOW!
  1000 (-8) + 1111 (-1) = 0111 (+7) → OVERFLOW!
  0011 (+3) + 0010 (+2) = 0101 (+5) → OK
` + "```" + `

**Barrel Shifter:**
` + "```" + `
A barrel shifter can shift by any amount in one cycle using a cascade of MUXes:

8-bit left shift by amount[2:0]:
    Layer 0 (shift by 4 if amount[2]=1):
        out[i] = amount[2] ? in[i-4] : in[i]
        
    Layer 1 (shift by 2 if amount[1]=1):
        out[i] = amount[1] ? prev[i-2] : prev[i]
        
    Layer 2 (shift by 1 if amount[0]=1):
        out[i] = amount[0] ? prev[i-1] : prev[i]

For 32-bit: 5 layers of MUXes (shift by 16, 8, 4, 2, 1)
For 64-bit: 6 layers
Each layer has N 2-to-1 MUXes
Total delay: O(log N) MUX delays
` + "```" + `

**Hardware Multiplication:**
` + "```" + `
Binary multiplication is shift-and-add:
    1101 (13)
  × 1011 (11)
  ──────
    1101      (1101 × 1, shifted 0)
   1101       (1101 × 1, shifted 1)
  0000        (1101 × 0, shifted 2)
 1101         (1101 × 1, shifted 3)
──────────
10001111      (143)

Array Multiplier: All partial products computed simultaneously
    N×N AND gates generate partial products
    Array of full adders sums them
    Delay: O(N), Area: O(N²)

Wallace Tree Multiplier:
    Use carry-save adders to reduce partial products in parallel
    3:2 compressors reduce three numbers to two
    Delay: O(log N), but more complex routing
    
Booth's Algorithm:
    Reduce number of partial products for signed multiplication
    Recode multiplier: groups of 1s → subtract at start, add at end
    Example: 0111 → 1001 (i.e., 8-1=7)
    Radix-4 Booth: processes 2 bits at a time, halving partial products
` + "```" + `

**Division Circuit:**
` + "```" + `
Restoring Division (like long division in binary):
    1. Shift remainder left, bring down next dividend bit
    2. Subtract divisor from remainder
    3. If result >= 0: quotient bit = 1 (keep result)
    4. If result < 0: quotient bit = 0 (restore old remainder)

Non-Restoring Division:
    Instead of restoring, just add divisor next time
    Saves one addition per step
    
SRT Division (used in modern CPUs):
    Lookup table predicts quotient bits
    Can produce multiple bits per cycle
    Intel Pentium FDIV bug (1994): error in lookup table!
` + "```" + ``,
					CodeExamples: `// ALU simulation in Go
package main

import "fmt"

// ALU operations
const (
    ALU_AND = iota
    ALU_OR
    ALU_ADD
    ALU_SUB
    ALU_SLT  // Set Less Than
    ALU_XOR
    ALU_SLL  // Shift Left Logical
    ALU_SRL  // Shift Right Logical
    ALU_SRA  // Shift Right Arithmetic
    ALU_MUL
)

type ALUResult struct {
    Result   int32
    Zero     bool
    Negative bool
    Overflow bool
    CarryOut bool
}

func ALU(a, b int32, op int) ALUResult {
    var result int32
    var overflow, carry bool

    switch op {
    case ALU_AND:
        result = a & b
    case ALU_OR:
        result = a | b
    case ALU_XOR:
        result = a ^ b
    case ALU_ADD:
        result = a + b
        // Overflow: same sign inputs, different sign output
        overflow = (a > 0 && b > 0 && result < 0) ||
                   (a < 0 && b < 0 && result > 0)
        carry = uint32(a)+uint32(b) < uint32(a)
    case ALU_SUB:
        result = a - b
        overflow = (a > 0 && b < 0 && result < 0) ||
                   (a < 0 && b > 0 && result > 0)
    case ALU_SLT:
        if a < b { result = 1 } else { result = 0 }
    case ALU_SLL:
        result = a << (b & 0x1F)
    case ALU_SRL:
        result = int32(uint32(a) >> (b & 0x1F))
    case ALU_SRA:
        result = a >> (b & 0x1F) // Arithmetic shift preserves sign
    case ALU_MUL:
        result = a * b
    }

    return ALUResult{
        Result:   result,
        Zero:     result == 0,
        Negative: result < 0,
        Overflow: overflow,
        CarryOut: carry,
    }
}

// Barrel shifter simulation
func barrelShiftLeft(value uint32, amount uint32) uint32 {
    amount &= 31
    // Layer by layer, like hardware
    if amount&16 != 0 { value <<= 16 }
    if amount&8 != 0  { value <<= 8 }
    if amount&4 != 0  { value <<= 4 }
    if amount&2 != 0  { value <<= 2 }
    if amount&1 != 0  { value <<= 1 }
    return value
}

// Array multiplier simulation
func arrayMultiply(a, b uint16) uint32 {
    var result uint32
    for i := 0; i < 16; i++ {
        if b&(1<<i) != 0 {
            result += uint32(a) << i
        }
    }
    return result
}

// Booth's multiplication (radix-2)
func boothMultiply(multiplicand, multiplier int16) int32 {
    m := int32(multiplicand)
    product := int32(0)
    prev := 0 // Previous bit

    for i := 0; i < 16; i++ {
        curr := int(multiplier>>i) & 1
        switch {
        case curr == 1 && prev == 0: product -= m << i  // Start of run of 1s
        case curr == 0 && prev == 1: product += m << i  // End of run of 1s
        }
        prev = curr
    }
    return product
}

func main() {
    // Test ALU operations
    tests := []struct {
        a, b int32
        op   int
        name string
    }{
        {42, 15, ALU_AND, "AND"},
        {42, 15, ALU_OR, "OR"},
        {42, 15, ALU_ADD, "ADD"},
        {42, 15, ALU_SUB, "SUB"},
        {42, 15, ALU_SLT, "SLT"},
        {42, 15, ALU_XOR, "XOR"},
        {42, 3, ALU_SLL, "SLL"},
        {2147483647, 1, ALU_ADD, "ADD (overflow)"},
    }

    fmt.Println("ALU Operations:")
    for _, t := range tests {
        r := ALU(t.a, t.b, t.op)
        fmt.Printf("%s(%d, %d) = %d (Z=%v N=%v V=%v)\n",
            t.name, t.a, t.b, r.Result, r.Zero, r.Negative, r.Overflow)
    }

    // Test barrel shifter
    fmt.Printf("\nBarrel shift: 0x1 << 5 = 0x%X\n", barrelShiftLeft(1, 5))
    fmt.Printf("Barrel shift: 0xFF << 8 = 0x%X\n", barrelShiftLeft(0xFF, 8))

    // Test multipliers
    fmt.Printf("\nArray multiply: 13 × 11 = %d\n", arrayMultiply(13, 11))
    fmt.Printf("Booth multiply: 13 × (-11) = %d\n", boothMultiply(13, -11))
    fmt.Printf("Booth multiply: (-7) × (-6) = %d\n", boothMultiply(-7, -6))
}`,
				},
			},
		},
		{
			ID:          2823,
			Title:       "Advanced Digital Design Techniques",
			Description: "Explore Karnaugh maps, hazards, timing analysis, and programmable logic devices used in modern digital design.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Karnaugh Maps and Logic Minimization",
					Content: `Karnaugh maps (K-maps) are a graphical method for minimizing Boolean expressions — essential for designing circuits with fewer gates, less delay, and lower power consumption.

**2-Variable K-Map:**
` + "```" + `
Function: F(A,B) = Σm(1,2,3)

    B=0  B=1
A=0 │ 0 │ 1 │
A=1 │ 1 │ 1 │

Group the three 1s:
  - Row A=1 (both columns): gives A
  - Column B=1 (both rows): gives B
  
F = A + B  (minimized from A'B + AB' + AB)
` + "```" + `

**3-Variable K-Map:**
` + "```" + `
Function: F(A,B,C) = Σm(0,2,4,5,6)

          BC
         00  01  11  10
    A=0 │ 1 │ 0 │ 0 │ 1 │
    A=1 │ 1 │ 1 │ 0 │ 1 │

Groups:
  - 4-cell group {0,2,4,6} (corners via wrap-around): C'
  - 2-cell group {4,5}: AB'
  
F = C' + AB'
(Reduced from A'B'C' + A'BC' + AB'C' + AB'C + ABC')
` + "```" + `

**4-Variable K-Map:**
` + "```" + `
Function: F(A,B,C,D) = Σm(0,1,2,5,8,9,10)

              CD
           00  01  11  10
    AB=00 │ 1 │ 1 │ 0 │ 1 │
    AB=01 │ 0 │ 1 │ 0 │ 0 │
    AB=11 │ 0 │ 0 │ 0 │ 0 │
    AB=10 │ 1 │ 1 │ 0 │ 1 │

Groups:
  - 4-cell: {0,1,8,9} → B'D' ... wait, let me redo:
  - {0,2,8,10}: B'D' 
  - {0,1,8,9}: B'C'
  - {1,5}: A'D ... no: {1,5} = A'·C'·D

Correct grouping:
  - 4-cell {0,1,8,9}: B'C'
  - 4-cell {0,2,8,10}: B'D'
  - 2-cell {1,5}: A'C'D (but 1 is already covered)
  - We need {5}: covered by {1,5} → A'C'D

F = B'C' + B'D' + A'C'D
` + "```" + `

**K-Map Rules:**
` + "```" + `
1. Groups must be rectangular and contain 2^n cells (1, 2, 4, 8, 16)
2. Groups can wrap around edges (top↔bottom, left↔right)
3. Every 1 must be covered by at least one group
4. Larger groups → simpler terms (fewer literals)
5. Overlapping groups are OK
6. Don't-care conditions (X) can be 0 or 1 — use them to make larger groups

Prime Implicants: Groups that cannot be made larger
Essential Prime Implicants: Prime implicants that cover at least one minterm not covered by any other prime implicant
` + "```" + `

**Don't-Care Conditions:**
` + "```" + `
BCD to 7-segment display: inputs 1010-1111 never occur
These are don't-cares (X) — we can treat them as 0 or 1
to create larger groups and simpler logic.

              CD
           00  01  11  10
    AB=00 │ 1 │ 0 │ 0 │ 1 │
    AB=01 │ 0 │ 1 │ 1 │ 0 │
    AB=11 │ X │ X │ X │ X │    ← Don't-care (invalid BCD)
    AB=10 │ 1 │ 0 │ X │ X │

By treating appropriate X's as 1, we get larger groups
and simpler combinational logic.
` + "```" + `

**Quine-McCluskey Algorithm:**
For more than 4-5 variables, K-maps become impractical. The Quine-McCluskey algorithm is a tabular method:
` + "```" + `
Step 1: List all minterms grouped by number of 1s
Step 2: Compare adjacent groups, find terms differing by 1 bit
Step 3: Replace differing bit with '-' (don't care)
Step 4: Repeat until no more combinations possible
Step 5: Create prime implicant chart
Step 6: Select minimum set of prime implicants covering all minterms

Advantages over K-maps:
- Works for any number of variables
- Algorithmic (can be automated)
- Guaranteed optimal solution
` + "```" + ``,
					CodeExamples: `// Quine-McCluskey algorithm implementation in Go
package main

import (
    "fmt"
    "strings"
)

type Implicant struct {
    minterms []int
    binary   string // Uses '-' for don't-care bits
    used     bool
}

func countOnes(n int) int {
    count := 0
    for n > 0 {
        count += n & 1
        n >>= 1
    }
    return count
}

func intToBinary(n, bits int) string {
    s := make([]byte, bits)
    for i := bits - 1; i >= 0; i-- {
        if n&1 == 1 {
            s[i] = '1'
        } else {
            s[i] = '0'
        }
        n >>= 1
    }
    return string(s)
}

func canCombine(a, b string) (string, bool) {
    diffs := 0
    result := make([]byte, len(a))
    for i := range a {
        if a[i] != b[i] {
            diffs++
            result[i] = '-'
        } else {
            result[i] = a[i]
        }
    }
    return string(result), diffs == 1
}

func quineMcCluskey(minterms []int, numVars int) []Implicant {
    // Step 1: Create initial implicants
    implicants := make([]Implicant, len(minterms))
    for i, m := range minterms {
        implicants[i] = Implicant{
            minterms: []int{m},
            binary:   intToBinary(m, numVars),
        }
    }

    var primeImplicants []Implicant

    for {
        var newImplicants []Implicant
        combined := make(map[string]bool)

        for i := 0; i < len(implicants); i++ {
            for j := i + 1; j < len(implicants); j++ {
                if result, ok := canCombine(implicants[i].binary, implicants[j].binary); ok {
                    if !combined[result] {
                        merged := append([]int{}, implicants[i].minterms...)
                        merged = append(merged, implicants[j].minterms...)
                        newImplicants = append(newImplicants, Implicant{
                            minterms: merged,
                            binary:   result,
                        })
                        combined[result] = true
                    }
                    implicants[i].used = true
                    implicants[j].used = true
                }
            }
        }

        // Collect unused implicants as prime implicants
        for _, imp := range implicants {
            if !imp.used {
                primeImplicants = append(primeImplicants, imp)
            }
        }

        if len(newImplicants) == 0 {
            break
        }
        implicants = newImplicants
    }

    return primeImplicants
}

func implicantToExpression(binary string, vars []string) string {
    var terms []string
    for i, c := range binary {
        switch c {
        case '1':
            terms = append(terms, vars[i])
        case '0':
            terms = append(terms, vars[i]+"'")
        }
    }
    if len(terms) == 0 {
        return "1"
    }
    return strings.Join(terms, "")
}

func main() {
    // F(A,B,C,D) = Σm(0,1,2,5,8,9,10)
    minterms := []int{0, 1, 2, 5, 8, 9, 10}
    vars := []string{"A", "B", "C", "D"}

    primes := quineMcCluskey(minterms, 4)

    fmt.Println("Prime Implicants:")
    for _, p := range primes {
        expr := implicantToExpression(p.binary, vars)
        fmt.Printf("  %s → %s (covers minterms %v)\n", p.binary, expr, p.minterms)
    }
}`,
				},
				{
					Title: "Hazards and Timing in Digital Circuits",
					Content: `Real digital circuits don't switch instantaneously — signal propagation takes time, and this creates hazards: unwanted glitches in the output during transitions.

**Types of Hazards:**
` + "```" + `
Static-1 Hazard:
    Output should stay 1 but briefly glitches to 0
    
    F = AB + A'C, with A changing, B=C=1:
    
    A: ──────┐          ┌──────
             └──────────┘
    AB:──────┐          
             └──────────────── (falls when A falls)
    A'C:─────────────┌──────  (rises after gate delay)
                     │
    F:  ─────┐  ↓   ┌──────  (glitch!)
             └──┘───┘
             
    The glitch occurs because AB falls before A'C rises
    (due to inverter delay on A → A')

Static-0 Hazard:
    Output should stay 0 but briefly glitches to 1
    Dual of static-1 hazard (in product-of-sums)

Dynamic Hazard:
    Output should change once but changes multiple times
    Occurs in multi-level circuits with reconvergent fanout
` + "```" + `

**Fixing Hazards with Consensus Terms:**
` + "```" + `
F = AB + A'C  has a hazard when A changes, B=C=1

K-map:
          AC
       00  01  11  10
  B=0 │ 0 │ 1 │ 1 │ 0 │
  B=1 │ 0 │ 1 │ 1 │ 1 │

The two groups (AB and A'C) are adjacent but don't overlap.
Add the consensus term BC:

F = AB + A'C + BC  (hazard-free!)

The added BC term "bridges" the transition:
  When A changes with B=C=1, BC=1 keeps output at 1
  during the glitch window.
` + "```" + `

**Setup and Hold Time Analysis:**
` + "```" + `
Critical Path Analysis:
    
    The longest combinational path between any two flip-flops
    determines the maximum clock frequency.

    FF1 ──[Logic Cloud]──→ FF2
     ↑                      ↑
     CLK                   CLK
     
    Timing constraint:
    Tclk ≥ Tclk-to-q(FF1) + Tlogic(worst case) + Tsetup(FF2)
    
    Slack = Tclk - (Tclk-to-q + Tlogic + Tsetup)
    If slack < 0 → timing violation → circuit fails!

    Example:
    Tclk-to-q = 0.5 ns
    Tlogic = 3.2 ns (critical path through 8 gate levels)
    Tsetup = 0.3 ns
    Minimum clock period = 0.5 + 3.2 + 0.3 = 4.0 ns
    Maximum frequency = 1/4.0 ns = 250 MHz
` + "```" + `

**Clock Skew:**
` + "```" + `
Clock skew: arrival time difference of clock at different flip-flops

    CLK ──────┬──[wire delay]──→ FF1 (CLK arrives at t=0.1ns)
              └──[wire delay]──→ FF2 (CLK arrives at t=0.3ns)
              
    Skew = 0.3 - 0.1 = 0.2 ns

    Setup time constraint (with skew):
    Tclk + Tskew ≥ Tclk-to-q + Tlogic + Tsetup
    (Positive skew at destination helps setup, hurts hold)
    
    Hold time constraint (with skew):
    Tclk-to-q + Tlogic ≥ Thold + Tskew
    (Must ensure data doesn't arrive TOO EARLY at destination)

    Clock Distribution:
    - H-tree: balanced binary tree, equal path lengths
    - Clock mesh: grid overlaid on chip, low skew
    - PLL/DLL: Phase-locked loops to synchronize clocks
` + "```" + `

**Power-Delay Product (PDP):**
` + "```" + `
PDP = Power × Delay  (energy per operation)
    - Lower PDP = more energy-efficient circuit
    - Fundamental tradeoff: faster circuits use more power

Energy-Delay Product (EDP):
    EDP = Energy × Delay = Power × Delay²
    - Better metric for comparing at different voltages
    - Optimal Vdd minimizes EDP

Voltage Scaling:
    Power ∝ V²f    (dynamic power)
    Delay ∝ V/(V-Vth)²  (approximately)
    
    Reducing V from 1.0V to 0.7V:
    → Power drops to ~49% (0.7²/1.0²)
    → Delay increases by ~1.5x
    → PDP drops to ~33%
    → Near-threshold computing: V ≈ Vth, very low power but slow
` + "```" + ``,
					CodeExamples: `// Timing analysis simulation
package main

import "fmt"

// Gate with propagation delay
type Gate struct {
    name  string
    delay float64 // nanoseconds
    fn    func([]bool) bool
}

// Timing path analysis
type TimingPath struct {
    source string
    dest   string
    gates  []Gate
}

func (tp TimingPath) TotalDelay() float64 {
    total := 0.0
    for _, g := range tp.gates {
        total += g.delay
    }
    return total
}

// Static Timing Analysis (simplified)
type STA struct {
    paths     []TimingPath
    clkPeriod float64
    tSetup    float64
    tHold     float64
    tClkToQ   float64
    clkSkew   float64
}

func (sta *STA) Analyze() {
    fmt.Printf("Clock Period: %.2f ns (%.0f MHz)\n",
        sta.clkPeriod, 1000.0/sta.clkPeriod)
    fmt.Printf("Clock Skew: %.2f ns\n", sta.clkSkew)
    fmt.Println()

    worstSlack := 999.0
    for _, path := range sta.paths {
        pathDelay := path.TotalDelay()
        required := sta.clkPeriod - sta.tClkToQ - sta.tSetup + sta.clkSkew
        slack := required - pathDelay

        status := "PASS"
        if slack < 0 {
            status = "FAIL"
        }

        fmt.Printf("Path %s → %s:\n", path.source, path.dest)
        fmt.Printf("  Delay: %.2f ns (%d gates)\n", pathDelay, len(path.gates))
        fmt.Printf("  Required: %.2f ns\n", required)
        fmt.Printf("  Slack: %.2f ns [%s]\n\n", slack, status)

        if slack < worstSlack {
            worstSlack = slack
        }
    }

    if worstSlack < 0 {
        maxFreq := 1000.0 / (sta.tClkToQ + worstSlack + sta.clkPeriod - worstSlack + sta.tSetup)
        fmt.Printf("TIMING VIOLATION! Worst slack: %.2f ns\n", worstSlack)
        fmt.Printf("Max achievable frequency: %.0f MHz\n", maxFreq)
    } else {
        fmt.Printf("All paths pass. Worst slack: %.2f ns\n", worstSlack)
    }

    // Hold time analysis
    fmt.Println("\nHold Time Analysis:")
    for _, path := range sta.paths {
        pathDelay := path.TotalDelay()
        holdSlack := sta.tClkToQ + pathDelay - sta.tHold - sta.clkSkew
        status := "PASS"
        if holdSlack < 0 {
            status = "FAIL"
        }
        fmt.Printf("Path %s → %s: hold slack = %.2f ns [%s]\n",
            path.source, path.dest, holdSlack, status)
    }
}

func main() {
    andGate := Gate{"AND", 0.3, nil}
    orGate := Gate{"OR", 0.35, nil}
    notGate := Gate{"NOT", 0.15, nil}
    muxGate := Gate{"MUX", 0.4, nil}
    adder := Gate{"ADDER", 1.2, nil}

    sta := &STA{
        clkPeriod: 4.0,
        tSetup:    0.3,
        tHold:     0.1,
        tClkToQ:   0.5,
        clkSkew:   0.1,
        paths: []TimingPath{
            {
                source: "RegA",
                dest:   "RegB",
                gates:  []Gate{muxGate, adder, andGate, orGate},
            },
            {
                source: "RegB",
                dest:   "RegC",
                gates:  []Gate{notGate, andGate, orGate, muxGate, andGate, orGate, notGate, andGate},
            },
            {
                source: "RegA",
                dest:   "RegC",
                gates:  []Gate{adder, adder, muxGate}, // Long arithmetic path
            },
        },
    }

    sta.Analyze()
}`,
				},
				{
					Title: "Programmable Logic Devices",
					Content: `Programmable Logic Devices (PLDs) allow hardware designs to be implemented in reconfigurable chips rather than custom ASICs, dramatically reducing development time and cost.

**PLD Evolution:**
` + "```" + `
ROM (Read-Only Memory):
  - Fixed AND array (full decoder), programmable OR array
  - Can implement ANY function of N inputs (stores truth table)
  - Wasteful for functions with many don't-cares

PLA (Programmable Logic Array):
  - Programmable AND array + Programmable OR array
  - Can share product terms across outputs
  - More efficient than ROM for sparse functions

PAL (Programmable Array Logic):
  - Programmable AND array + Fixed OR array
  - Simpler to manufacture, faster than PLA
  - Limited product terms per output
  
CPLD (Complex PLD):
  - Multiple PAL-like blocks connected by programmable interconnect
  - Predictable timing (fixed interconnect delays)
  - Good for glue logic, state machines

FPGA (Field-Programmable Gate Array):
  - Lookup tables (LUTs) + Flip-flops + Programmable routing
  - Most flexible, highest capacity
  - Dominant PLD technology today
` + "```" + `

**FPGA Architecture:**
` + "```" + `
FPGA Internal Structure:
┌──────────────────────────────────────────┐
│  I/O │  I/O │  I/O │  I/O │  I/O │  I/O │
├──────┼──────┼──────┼──────┼──────┼──────┤
│  I/O │  CLB │  CLB │  CLB │  CLB │  I/O │
│      │      │      │      │      │      │
├──────┼──────┼──────┼──────┼──────┼──────┤
│  I/O │  CLB │ BRAM │  CLB │  DSP │  I/O │
│      │      │      │      │      │      │
├──────┼──────┼──────┼──────┼──────┼──────┤
│  I/O │  CLB │  CLB │  CLB │  CLB │  I/O │
│      │      │      │      │      │      │
├──────┼──────┼──────┼──────┼──────┼──────┤
│  I/O │  I/O │  I/O │  I/O │  I/O │  I/O │
└──────┴──────┴──────┴──────┴──────┴──────┘

CLB  = Configurable Logic Block (LUTs + FFs)
BRAM = Block RAM (embedded memory)
DSP  = Digital Signal Processing block (multiplier + accumulator)
I/O  = Input/Output block (configurable pin drivers)

Everything connected by programmable routing (switch matrices)
` + "```" + `

**Configurable Logic Block (CLB):**
` + "```" + `
Modern CLB (Xilinx Slice):
┌─────────────────────────────────┐
│  ┌──────┐   ┌────┐   ┌────┐   │
│  │ 6-LUT│──→│ MUX│──→│ FF │──→│ Q
│  │      │   │    │   │    │   │
│  └──────┘   │    │   └────┘   │
│             │    │──────────→│ Comb Out
│  ┌──────┐   └────┘            │
│  │ 6-LUT│                     │
│  └──┬───┘                     │
│     └── Carry Chain ──────────│
│                               │
│  ┌──────────────────┐         │
│  │ Carry Logic      │         │
│  │ (fast arithmetic)│         │
│  └──────────────────┘         │
└─────────────────────────────────┘

6-input LUT: Can implement ANY Boolean function of 6 variables
  - Internally: 64-bit SRAM storing the truth table
  - Configuration: write the truth table to SRAM
  - Also usable as: 64-bit shift register, small RAM (distributed RAM)
` + "```" + `

**FPGA vs ASIC Comparison:**
` + "```" + `
                FPGA              ASIC
Speed:          ~300-800 MHz      ~1-5 GHz
Power:          10-100W           Optimized per design
Area:           10-100x overhead  Minimal
Cost (unit):    $10-$10,000       $0.10-$10 (high volume)
Cost (NRE):     ~$0               $1M-$100M (masks, verification)
Time to market: Days-weeks        6-18 months
Flexibility:    Reprogrammable    Fixed after fabrication
Best for:       Prototyping,      High-volume products,
                low volume,       maximum performance
                ASIC replacement
` + "```" + `

**Hardware Description Languages (HDLs):**
` + "```" + `
Verilog example — 4-bit counter:

module counter4(
    input  wire clk,
    input  wire reset,
    output reg [3:0] count
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= 4'b0000;
        else
            count <= count + 1;
    end
endmodule

VHDL example — same counter:

entity counter4 is
    port(
        clk   : in  std_logic;
        reset : in  std_logic;
        count : out std_logic_vector(3 downto 0)
    );
end counter4;

architecture behavioral of counter4 is
    signal cnt : unsigned(3 downto 0);
begin
    process(clk, reset)
    begin
        if reset = '1' then
            cnt <= (others => '0');
        elsif rising_edge(clk) then
            cnt <= cnt + 1;
        end if;
    end process;
    count <= std_logic_vector(cnt);
end behavioral;

Modern alternatives: Chisel (Scala-based), SpinalHDL, Amaranth (Python)
` + "```" + ``,
					CodeExamples: `// Simulating FPGA concepts in Go
package main

import "fmt"

// Lookup Table (LUT) - the fundamental FPGA building block
type LUT struct {
    numInputs int
    table     []bool // Truth table stored in SRAM
}

func NewLUT(numInputs int, truthTable []bool) *LUT {
    size := 1 << numInputs
    if len(truthTable) != size {
        panic("truth table size mismatch")
    }
    return &LUT{numInputs: numInputs, table: truthTable}
}

func (lut *LUT) Evaluate(inputs []bool) bool {
    idx := 0
    for i, b := range inputs {
        if b {
            idx |= 1 << i
        }
    }
    return lut.table[idx]
}

// Configure LUT to implement a specific function
func ConfigureAsAND(n int) *LUT {
    size := 1 << n
    table := make([]bool, size)
    table[size-1] = true // Only all-1s input gives 1
    return NewLUT(n, table)
}

func ConfigureAsXOR2() *LUT {
    return NewLUT(2, []bool{false, true, true, false})
}

func ConfigureAsFullAdder() (*LUT, *LUT) {
    // 3-input LUT for Sum = A XOR B XOR Cin
    sumTable := make([]bool, 8)
    carryTable := make([]bool, 8)
    for i := 0; i < 8; i++ {
        a := (i >> 0) & 1
        b := (i >> 1) & 1
        c := (i >> 2) & 1
        sum := a ^ b ^ c
        carry := (a&b) | (b&c) | (a&c)
        sumTable[i] = sum == 1
        carryTable[i] = carry == 1
    }
    return NewLUT(3, sumTable), NewLUT(3, carryTable)
}

// Simple FPGA Slice: 2 LUTs + 2 FFs + carry chain
type FPGASlice struct {
    lut0, lut1 *LUT
    ff0, ff1   bool
    carryIn    bool
}

func (s *FPGASlice) Clock(lut0Inputs, lut1Inputs []bool) (q0, q1 bool) {
    s.ff0 = s.lut0.Evaluate(lut0Inputs)
    s.ff1 = s.lut1.Evaluate(lut1Inputs)
    return s.ff0, s.ff1
}

// Programmable interconnect matrix
type SwitchMatrix struct {
    connections map[string]string // source → destination
}

func NewSwitchMatrix() *SwitchMatrix {
    return &SwitchMatrix{connections: make(map[string]string)}
}

func (sm *SwitchMatrix) Connect(src, dst string) {
    sm.connections[src] = dst
}

func (sm *SwitchMatrix) Route(src string) string {
    if dst, ok := sm.connections[src]; ok {
        return dst
    }
    return ""
}

func main() {
    // Create LUTs for different functions
    andLUT := ConfigureAsAND(3)
    xorLUT := ConfigureAsXOR2()
    sumLUT, carryLUT := ConfigureAsFullAdder()

    // Test AND LUT
    fmt.Println("3-input AND LUT:")
    for i := 0; i < 8; i++ {
        inputs := []bool{i&1 != 0, i&2 != 0, i&4 != 0}
        fmt.Printf("  %v → %v\n", inputs, andLUT.Evaluate(inputs))
    }

    // Test XOR LUT
    fmt.Println("\n2-input XOR LUT:")
    for i := 0; i < 4; i++ {
        inputs := []bool{i&1 != 0, i&2 != 0}
        fmt.Printf("  %v → %v\n", inputs, xorLUT.Evaluate(inputs))
    }

    // Test Full Adder from LUTs
    fmt.Println("\nFull Adder from LUTs:")
    fmt.Println("A B Cin │ Sum Cout")
    for i := 0; i < 8; i++ {
        inputs := []bool{i&1 != 0, i&2 != 0, i&4 != 0}
        sum := sumLUT.Evaluate(inputs)
        carry := carryLUT.Evaluate(inputs)
        fmt.Printf("%d %d  %d  │  %d   %d\n",
            boolToInt(inputs[0]), boolToInt(inputs[1]), boolToInt(inputs[2]),
            boolToInt(sum), boolToInt(carry))
    }

    // Demonstrate LUT configurability
    fmt.Println("\nLUT Reconfiguration Demo:")
    lut := NewLUT(2, []bool{false, true, true, false}) // XOR
    fmt.Printf("Configured as XOR: f(0,1) = %v\n", lut.Evaluate([]bool{false, true}))
    
    lut = NewLUT(2, []bool{false, false, false, true}) // AND
    fmt.Printf("Reconfigured as AND: f(0,1) = %v\n", lut.Evaluate([]bool{false, true}))
    fmt.Printf("Reconfigured as AND: f(1,1) = %v\n", lut.Evaluate([]bool{true, true}))
}

func boolToInt(b bool) int {
    if b { return 1 }
    return 0
}`,
				},
			},
		},
	})
}
