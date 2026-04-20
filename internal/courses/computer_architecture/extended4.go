package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2824,
			Title:       "CPU Datapath Design",
			Description: "Understand how processors execute instructions: the datapath, control unit, single-cycle and multi-cycle designs, and the fetch-decode-execute pipeline.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Instruction Set Architecture Fundamentals",
					Content: `The Instruction Set Architecture (ISA) defines the interface between software and hardware. It specifies what instructions the processor understands, how operands are accessed, and how data is encoded.

**ISA Design Philosophies:**
` + "```" + `
CISC (Complex Instruction Set Computing):
  - Many instructions, variable length (1-15 bytes)
  - Single instructions can do complex operations (e.g., string copy)
  - Memory-to-memory operations allowed
  - Example: x86/x64
  - Advantage: Dense code, fewer instructions per program
  - Disadvantage: Complex decode logic, hard to pipeline

RISC (Reduced Instruction Set Computing):
  - Fewer instructions, fixed length (4 bytes typical)
  - Load/store architecture: only loads/stores access memory
  - All arithmetic operates on registers
  - Example: ARM, RISC-V, MIPS
  - Advantage: Simple decode, easy to pipeline, regular timing
  - Disadvantage: More instructions per program

Modern Reality:
  x86 is CISC on the outside, RISC on the inside:
  - Frontend decodes complex x86 instructions into micro-ops (µops)
  - Backend executes µops like a RISC machine
  - Best of both worlds: backward compatibility + modern performance
` + "```" + `

**RISC-V ISA Example (RV32I — 32-bit Integer):**
` + "```" + `
Register File: 32 registers × 32 bits (x0 always = 0)

Instruction Formats (all 32 bits):
┌─────────────────────────────────────────────────────────┐
│ R-type: funct7  │ rs2 │ rs1 │funct3│  rd  │  opcode    │
│  bits:    7       5     5     3       5       7         │
│  Used: ALU operations (add, sub, and, or, slt)          │
├─────────────────────────────────────────────────────────┤
│ I-type:   imm[11:0]   │ rs1 │funct3│  rd  │  opcode    │
│  bits:      12           5     3       5       7        │
│  Used: Loads, ADDI, JALR                                │
├─────────────────────────────────────────────────────────┤
│ S-type: imm[11:5]│ rs2 │ rs1 │funct3│imm[4:0]│ opcode  │
│  bits:     7        5     5     3       5        7      │
│  Used: Stores (sw, sh, sb)                              │
├─────────────────────────────────────────────────────────┤
│ B-type: imm[12│10:5]│rs2│rs1│f3│imm[4:1│11]│  opcode   │
│  Used: Branches (beq, bne, blt, bge)                    │
├─────────────────────────────────────────────────────────┤
│ U-type:      imm[31:12]           │  rd  │  opcode      │
│  Used: LUI (load upper immediate), AUIPC                │
├─────────────────────────────────────────────────────────┤
│ J-type: imm[20│10:1│11│19:12]     │  rd  │  opcode      │
│  Used: JAL (jump and link)                               │
└─────────────────────────────────────────────────────────┘

Key Instructions:
  add  rd, rs1, rs2    # rd = rs1 + rs2
  addi rd, rs1, imm    # rd = rs1 + sign_extend(imm)
  lw   rd, offset(rs1) # rd = Mem[rs1 + offset]
  sw   rs2, offset(rs1) # Mem[rs1 + offset] = rs2
  beq  rs1, rs2, label # if rs1==rs2, PC += offset
  jal  rd, label       # rd = PC+4; PC += offset
` + "```" + `

**Addressing Modes:**
` + "```" + `
Mode            Example              Effective Address
Immediate       ADDI x1, x0, 42     N/A (operand in instruction)
Register        ADD x1, x2, x3      N/A (operands in registers)
Base+Offset     LW x1, 8(x2)        EA = x2 + 8
PC-Relative     BEQ x1, x2, label   EA = PC + offset
Upper Immediate LUI x1, 0x12345     x1 = 0x12345000

RISC-V uses only these simple modes.
x86 also has: scaled index (base + index*scale + disp),
  segment-based, and more complex modes.
` + "```" + ``,
					CodeExamples: `// RISC-V instruction decoder in Go
package main

import "fmt"

// RISC-V instruction types
const (
    OP_LUI    = 0b0110111
    OP_AUIPC  = 0b0010111
    OP_JAL    = 0b1101111
    OP_JALR   = 0b1100111
    OP_BRANCH = 0b1100011
    OP_LOAD   = 0b0000011
    OP_STORE  = 0b0100011
    OP_OPIMM  = 0b0010011
    OP_OP     = 0b0110011
)

type DecodedInst struct {
    Opcode uint32
    Rd     uint32
    Rs1    uint32
    Rs2    uint32
    Funct3 uint32
    Funct7 uint32
    ImmI   int32
    ImmS   int32
    ImmB   int32
    ImmU   int32
    ImmJ   int32
}

func decode(inst uint32) DecodedInst {
    d := DecodedInst{
        Opcode: inst & 0x7F,
        Rd:     (inst >> 7) & 0x1F,
        Funct3: (inst >> 12) & 0x7,
        Rs1:    (inst >> 15) & 0x1F,
        Rs2:    (inst >> 20) & 0x1F,
        Funct7: (inst >> 25) & 0x7F,
    }

    // I-type immediate
    d.ImmI = int32(inst) >> 20

    // S-type immediate
    d.ImmS = (int32(inst)>>20)&^0x1F | int32((inst>>7)&0x1F)

    // B-type immediate
    d.ImmB = int32(0)
    if inst&(1<<31) != 0 { d.ImmB |= -4096 }
    d.ImmB |= int32((inst >> 7) & 1) << 11
    d.ImmB |= int32((inst >> 25) & 0x3F) << 5
    d.ImmB |= int32((inst >> 8) & 0xF) << 1

    // U-type immediate
    d.ImmU = int32(inst) & ^int32(0xFFF)

    // J-type immediate
    d.ImmJ = int32(0)
    if inst&(1<<31) != 0 { d.ImmJ |= -(1 << 20) }
    d.ImmJ |= int32((inst >> 12) & 0xFF) << 12
    d.ImmJ |= int32((inst >> 20) & 1) << 11
    d.ImmJ |= int32((inst >> 21) & 0x3FF) << 1

    return d
}

func disassemble(inst uint32) string {
    d := decode(inst)
    switch d.Opcode {
    case OP_OP:
        op := "unknown"
        switch d.Funct3 {
        case 0:
            if d.Funct7 == 0 { op = "add" } else { op = "sub" }
        case 1: op = "sll"
        case 2: op = "slt"
        case 4: op = "xor"
        case 6: op = "or"
        case 7: op = "and"
        }
        return fmt.Sprintf("%s x%d, x%d, x%d", op, d.Rd, d.Rs1, d.Rs2)
    case OP_OPIMM:
        op := "unknown"
        switch d.Funct3 {
        case 0: op = "addi"
        case 2: op = "slti"
        case 4: op = "xori"
        case 6: op = "ori"
        case 7: op = "andi"
        }
        return fmt.Sprintf("%s x%d, x%d, %d", op, d.Rd, d.Rs1, d.ImmI)
    case OP_LOAD:
        return fmt.Sprintf("lw x%d, %d(x%d)", d.Rd, d.ImmI, d.Rs1)
    case OP_STORE:
        return fmt.Sprintf("sw x%d, %d(x%d)", d.Rs2, d.ImmS, d.Rs1)
    case OP_BRANCH:
        op := "b??"
        switch d.Funct3 {
        case 0: op = "beq"
        case 1: op = "bne"
        case 4: op = "blt"
        case 5: op = "bge"
        }
        return fmt.Sprintf("%s x%d, x%d, %d", op, d.Rs1, d.Rs2, d.ImmB)
    case OP_LUI:
        return fmt.Sprintf("lui x%d, 0x%X", d.Rd, uint32(d.ImmU)>>12)
    case OP_JAL:
        return fmt.Sprintf("jal x%d, %d", d.Rd, d.ImmJ)
    }
    return fmt.Sprintf("unknown (0x%08X)", inst)
}

func main() {
    // Some RISC-V instructions to decode
    instructions := []uint32{
        0x002081B3, // add x3, x1, x2
        0x00508093, // addi x1, x1, 5
        0x0020A023, // sw x2, 0(x1)
        0x00002183, // lw x3, 0(x0)
        0x00208463, // beq x1, x2, 8
        0x123450B7, // lui x1, 0x12345
    }

    fmt.Println("RISC-V Disassembly:")
    for _, inst := range instructions {
        fmt.Printf("  0x%08X → %s\n", inst, disassemble(inst))
    }
}`,
				},
				{
					Title: "Single-Cycle Datapath",
					Content: `A single-cycle processor completes every instruction in one clock cycle. While simple, it illustrates all the key datapath components.

**Single-Cycle RISC-V Datapath:**
` + "```" + `
                      ┌─────────────────────────────────────────┐
                      │                                         │
    ┌──────┐     ┌────┴───┐     ┌──────────┐     ┌──────┐     │
    │  PC  │────→│ Instr  │────→│ Register │────→│      │     │
    │      │     │ Memory │     │   File   │    →│  ALU │────→│
    └──┬───┘     └────────┘     └──┬───┬───┘     │      │     │
       │                           │   │         └──┬───┘     │
       │         ┌──────────┐      │   │            │         │
       └────────→│  PC + 4  │      │   │      ┌────┴─────┐   │
                 └──────────┘      │   │      │  Data    │   │
                      │            │   │      │  Memory  │   │
                      ▼            │   │      └──────────┘   │
                    [MUX]          │   │            │         │
                      │            │   └────────[MUX]─────→   │
                      └────────────┘            Write Back    │
                                                              │
    Components:                                               │
    1. PC (Program Counter): holds current instruction address │
    2. Instruction Memory: ROM storing the program             │
    3. Register File: 32 registers, 2 read ports, 1 write port│
    4. ALU: performs arithmetic/logic operations                │
    5. Data Memory: RAM for load/store instructions            │
    6. MUXes: select between alternative data sources          │
    7. Sign Extender: extends immediates to 32 bits            │
    └─────────────────────────────────────────────────────────┘
` + "```" + `

**Instruction Execution Flow:**
` + "```" + `
ADD x3, x1, x2:
  1. Fetch: Read instruction from Instr Memory at PC
  2. Decode: Extract rd=x3, rs1=x1, rs2=x2, opcode=OP
  3. Read registers: A = RegFile[x1], B = RegFile[x2]
  4. Execute: ALU computes A + B
  5. Write back: RegFile[x3] = ALU result
  6. PC update: PC = PC + 4

LW x3, 8(x1):   (Load Word)
  1. Fetch: Read instruction
  2. Decode: rd=x3, rs1=x1, imm=8, opcode=LOAD
  3. Read register: A = RegFile[x1]
  4. Execute: ALU computes A + 8 (address calculation)
  5. Memory: Read DataMem[A + 8]
  6. Write back: RegFile[x3] = memory data
  7. PC = PC + 4

SW x2, 12(x1):  (Store Word)
  1. Fetch, Decode
  2. Read registers: A = RegFile[x1], B = RegFile[x2]
  3. Execute: ALU computes A + 12 (address)
  4. Memory: Write DataMem[A + 12] = B
  5. No write back to registers
  6. PC = PC + 4

BEQ x1, x2, offset:  (Branch if Equal)
  1. Fetch, Decode
  2. Read registers: A = RegFile[x1], B = RegFile[x2]
  3. Execute: ALU computes A - B, check if zero
  4. PC = (Zero flag) ? PC + offset : PC + 4
` + "```" + `

**Control Unit:**
` + "```" + `
The control unit generates signals based on the opcode:

Opcode  │ ALUSrc │ MemToReg │ RegWrite │ MemRead │ MemWrite │ Branch │ ALUOp
────────┼────────┼──────────┼──────────┼─────────┼──────────┼────────┼──────
R-type  │   0    │    0     │    1     │    0    │    0     │   0    │  10
LW      │   1    │    1     │    1     │    1    │    0     │   0    │  00
SW      │   1    │    X     │    0     │    0    │    1     │   0    │  00
BEQ     │   0    │    X     │    0     │    0    │    0     │   1    │  01

ALUSrc:   0 = register B, 1 = immediate
MemToReg: 0 = ALU result, 1 = memory data
RegWrite: 1 = write to register file
Branch:   1 = conditional branch

ALU Control (from ALUOp + funct3 + funct7):
  ALUOp=00: ADD (for loads/stores — address calculation)
  ALUOp=01: SUB (for branches — comparison)
  ALUOp=10: depends on funct3/funct7 (R-type operations)
` + "```" + `

**Single-Cycle Limitations:**
` + "```" + `
Clock period must accommodate the SLOWEST instruction:
  - Load is slowest: Fetch + Decode + ALU + MemRead + WriteBack
  - Even fast instructions (ADD) must wait for the full period

Example timing:
  Instruction Memory:  200 ps
  Register Read:       100 ps
  ALU:                 200 ps
  Data Memory:         200 ps
  Register Write:      100 ps

  ADD: 200 + 100 + 200 + 0 + 100 = 600 ps
  LW:  200 + 100 + 200 + 200 + 100 = 800 ps  ← critical path
  
  Clock period must be ≥ 800 ps → max 1.25 GHz
  But ADD only needs 600 ps — 200 ps wasted every ADD!
  
  Solution: Multi-cycle or pipelined design
` + "```" + ``,
					CodeExamples: `// Single-cycle RISC-V processor simulator in Go
package main

import "fmt"

const NUM_REGS = 32

type SingleCycleCPU struct {
    PC      uint32
    Regs    [NUM_REGS]int32
    IMem    []uint32  // Instruction memory
    DMem    [4096]int32 // Data memory (4K words)
    Halted  bool
}

type ControlSignals struct {
    ALUSrc   bool
    MemToReg bool
    RegWrite bool
    MemRead  bool
    MemWrite bool
    Branch   bool
    Jump     bool
    ALUOp    int
}

func (cpu *SingleCycleCPU) controlUnit(opcode uint32) ControlSignals {
    switch opcode {
    case 0b0110011: // R-type
        return ControlSignals{ALUOp: 2, RegWrite: true}
    case 0b0010011: // I-type ALU
        return ControlSignals{ALUSrc: true, ALUOp: 2, RegWrite: true}
    case 0b0000011: // Load
        return ControlSignals{ALUSrc: true, MemToReg: true, RegWrite: true, MemRead: true}
    case 0b0100011: // Store
        return ControlSignals{ALUSrc: true, MemWrite: true}
    case 0b1100011: // Branch
        return ControlSignals{Branch: true, ALUOp: 1}
    case 0b1101111: // JAL
        return ControlSignals{Jump: true, RegWrite: true}
    case 0b0110111: // LUI
        return ControlSignals{RegWrite: true, ALUOp: 3}
    }
    return ControlSignals{}
}

func (cpu *SingleCycleCPU) aluControl(aluOp int, funct3, funct7 uint32) int {
    switch aluOp {
    case 0: return 0 // ADD for loads/stores
    case 1: return 1 // SUB for branches
    case 2: // R-type or I-type
        switch funct3 {
        case 0:
            if funct7 == 0x20 { return 1 } // SUB
            return 0 // ADD
        case 1: return 5 // SLL
        case 2: return 7 // SLT
        case 4: return 3 // XOR
        case 6: return 2 // OR
        case 7: return 4 // AND
        }
    case 3: return 8 // LUI pass-through
    }
    return 0
}

func (cpu *SingleCycleCPU) executeALU(op int, a, b int32) (int32, bool) {
    var result int32
    switch op {
    case 0: result = a + b
    case 1: result = a - b
    case 2: result = a | b
    case 3: result = a ^ b
    case 4: result = a & b
    case 5: result = a << (b & 0x1F)
    case 7:
        if a < b { result = 1 } else { result = 0 }
    case 8: result = b // Pass through (LUI)
    }
    return result, result == 0
}

func (cpu *SingleCycleCPU) Step() {
    if cpu.Halted || int(cpu.PC/4) >= len(cpu.IMem) {
        cpu.Halted = true
        return
    }

    // Fetch
    inst := cpu.IMem[cpu.PC/4]
    
    // Decode
    opcode := inst & 0x7F
    rd := (inst >> 7) & 0x1F
    funct3 := (inst >> 12) & 0x7
    rs1 := (inst >> 15) & 0x1F
    rs2 := (inst >> 20) & 0x1F
    funct7 := (inst >> 25) & 0x7F
    immI := int32(inst) >> 20
    immS := (int32(inst)>>20)&^0x1F | int32((inst>>7)&0x1F)
    immB := int32(0)
    if inst&(1<<31) != 0 { immB = -4096 }
    immB |= int32((inst>>7)&1)<<11 | int32((inst>>25)&0x3F)<<5 | int32((inst>>8)&0xF)<<1
    immU := int32(inst) & ^int32(0xFFF)
    immJ := int32(0)
    if inst&(1<<31) != 0 { immJ = -(1 << 20) }
    immJ |= int32((inst>>12)&0xFF)<<12 | int32((inst>>20)&1)<<11 | int32((inst>>21)&0x3FF)<<1

    ctrl := cpu.controlUnit(opcode)

    // Register read
    regA := cpu.Regs[rs1]
    regB := cpu.Regs[rs2]

    // ALU input MUX
    aluB := regB
    if ctrl.ALUSrc {
        switch opcode {
        case 0b0100011: aluB = immS
        case 0b0110111: aluB = immU
        default: aluB = immI
        }
    }

    aluOp := cpu.aluControl(ctrl.ALUOp, funct3, funct7)
    aluResult, zero := cpu.executeALU(aluOp, regA, aluB)

    // Memory
    var memData int32
    if ctrl.MemRead {
        addr := uint32(aluResult) / 4
        if addr < uint32(len(cpu.DMem)) {
            memData = cpu.DMem[addr]
        }
    }
    if ctrl.MemWrite {
        addr := uint32(aluResult) / 4
        if addr < uint32(len(cpu.DMem)) {
            cpu.DMem[addr] = regB
        }
    }

    // Write back
    if ctrl.RegWrite && rd != 0 {
        if ctrl.MemToReg {
            cpu.Regs[rd] = memData
        } else if ctrl.Jump {
            cpu.Regs[rd] = int32(cpu.PC + 4)
        } else {
            cpu.Regs[rd] = aluResult
        }
    }

    // PC update
    if ctrl.Branch && zero {
        cpu.PC = uint32(int32(cpu.PC) + immB)
    } else if ctrl.Jump {
        cpu.PC = uint32(int32(cpu.PC) + immJ)
    } else {
        cpu.PC += 4
    }
}

func main() {
    // Program: compute sum of 1..10
    program := []uint32{
        0x00000093, // addi x1, x0, 0    (sum = 0)
        0x00100113, // addi x2, x0, 1    (i = 1)
        0x00B00193, // addi x3, x0, 11   (limit = 11)
        0x00208093, // add  x1, x1, x2   (sum += i)  <- loop
        0x00110113, // addi x2, x2, 1    (i++)
        0xFE311CE3, // bne  x2, x3, -8   (if i != 11, goto loop)
    }

    cpu := &SingleCycleCPU{IMem: program}
    cycle := 0
    for !cpu.Halted && cycle < 100 {
        cpu.Step()
        cycle++
    }
    fmt.Printf("Sum of 1..10 = %d (x1)\n", cpu.Regs[1])
    fmt.Printf("Cycles: %d\n", cycle)
}`,
				},
				{
					Title: "Pipelining Fundamentals",
					Content: `Pipelining overlaps instruction execution to dramatically increase throughput. Instead of completing one instruction before starting the next, we process multiple instructions simultaneously in different stages.

**The Laundry Analogy:**
` + "```" + `
Non-pipelined (sequential):
    Load1: [Wash 30m][Dry 30m][Fold 30m]
    Load2:                              [Wash 30m][Dry 30m][Fold 30m]
    Load3:                                                          [Wash 30m][Dry 30m][Fold 30m]
    Total: 270 minutes for 3 loads

Pipelined:
    Load1: [Wash 30m][Dry 30m][Fold 30m]
    Load2:          [Wash 30m][Dry 30m][Fold 30m]
    Load3:                   [Wash 30m][Dry 30m][Fold 30m]
    Total: 150 minutes for 3 loads (1.8x speedup)
    
    With N loads: sequential = N × 90 min
                  pipelined  = 90 + (N-1) × 30 min
    As N → ∞: speedup → 3x (number of stages)
` + "```" + `

**Classic 5-Stage RISC Pipeline:**
` + "```" + `
Stage 1: IF  (Instruction Fetch)     — Read instruction from memory
Stage 2: ID  (Instruction Decode)    — Decode instruction, read registers
Stage 3: EX  (Execute)              — ALU operation or address calculation
Stage 4: MEM (Memory Access)         — Read/write data memory
Stage 5: WB  (Write Back)           — Write result to register file

Pipeline Registers between stages store intermediate results:
    IF/ID → ID/EX → EX/MEM → MEM/WB

Time →  1   2   3   4   5   6   7   8   9
Inst1: [IF][ID][EX][MEM][WB]
Inst2:     [IF][ID][EX][MEM][WB]
Inst3:         [IF][ID][EX][MEM][WB]
Inst4:             [IF][ID][EX][MEM][WB]
Inst5:                 [IF][ID][EX][MEM][WB]

After pipeline fills (cycle 5+): one instruction completes per cycle!
Throughput: 5x improvement over single-cycle
Latency per instruction: unchanged (5 cycles)
` + "```" + `

**Pipeline Performance:**
` + "```" + `
Ideal speedup = Number of pipeline stages = k
Real speedup < k due to:
  1. Pipeline hazards (stalls)
  2. Unbalanced stages (clock = slowest stage)
  3. Pipeline register overhead

CPI (Cycles Per Instruction):
  Ideal pipeline: CPI = 1
  Real pipeline:  CPI = 1 + stall_cycles_per_instruction

Throughput = Instructions / Time = 1 / (CPI × Tclock)

Example:
  5-stage pipeline, Tclock = 200 ps (per stage)
  Average stalls = 0.3 cycles per instruction
  CPI = 1.3
  Throughput = 1 / (1.3 × 200ps) = 3.85 billion instructions/sec

  Compare to single-cycle at 800 ps:
  Speedup = 800 / (1.3 × 200) = 3.08x
  (Less than ideal 5x due to stalls and overhead)
` + "```" + `

**Deeper Pipelines:**
` + "```" + `
More stages = higher clock frequency but:
  - More hazards (longer data dependencies)
  - Higher branch penalty
  - More pipeline register power consumption
  - Diminishing returns

Historical trend:
  MIPS R2000 (1985):     5 stages,  15 MHz
  Intel Pentium (1993):  5 stages, 66 MHz  
  Pentium III (1999):   10 stages, 500 MHz
  Pentium 4 (2000):     20 stages, 1.5 GHz
  Pentium 4 (2004):     31 stages, 3.8 GHz ← NetBurst (too deep!)
  Core (2006):          14 stages, 2.67 GHz ← fewer stages, more efficient
  Modern (2024):        ~15-20 stages, 5-6 GHz

The Pentium 4 "NetBurst" architecture showed that deeper isn't always
better — the branch misprediction penalty was too high.
` + "```" + ``,
					CodeExamples: `// Pipelined processor simulator in Go
package main

import "fmt"

// Pipeline register contents
type IFIDReg struct {
    Instruction uint32
    PC          uint32
    Valid       bool
}

type IDEXReg struct {
    PC       uint32
    RegA     int32
    RegB     int32
    Imm      int32
    Rd       uint32
    Rs1      uint32
    Rs2      uint32
    Funct3   uint32
    Funct7   uint32
    Opcode   uint32
    ALUSrc   bool
    MemRead  bool
    MemWrite bool
    RegWrite bool
    MemToReg bool
    Branch   bool
    Valid    bool
}

type EXMEMReg struct {
    ALUResult int32
    RegB      int32
    Rd        uint32
    Zero      bool
    MemRead   bool
    MemWrite  bool
    RegWrite  bool
    MemToReg  bool
    Branch    bool
    BranchPC  uint32
    Valid     bool
}

type MEMWBReg struct {
    ALUResult int32
    MemData   int32
    Rd        uint32
    RegWrite  bool
    MemToReg  bool
    Valid     bool
}

type PipelinedCPU struct {
    PC   uint32
    Regs [32]int32
    IMem []uint32
    DMem [1024]int32

    // Pipeline registers (double-buffered)
    ifid  IFIDReg
    idex  IDEXReg
    exmem EXMEMReg
    memwb MEMWBReg

    // Statistics
    cycles     int
    completed  int
    stalls     int
}

func (cpu *PipelinedCPU) Run(maxCycles int) {
    for cpu.cycles = 0; cpu.cycles < maxCycles; cpu.cycles++ {
        // Execute stages in reverse order to avoid conflicts
        cpu.writeBack()
        cpu.memoryAccess()
        cpu.execute()
        cpu.decode()
        cpu.fetch()
    }
}

func (cpu *PipelinedCPU) fetch() {
    if int(cpu.PC/4) >= len(cpu.IMem) {
        cpu.ifid = IFIDReg{Valid: false}
        return
    }
    cpu.ifid = IFIDReg{
        Instruction: cpu.IMem[cpu.PC/4],
        PC:          cpu.PC,
        Valid:       true,
    }
    cpu.PC += 4
}

func (cpu *PipelinedCPU) decode() {
    if !cpu.ifid.Valid {
        cpu.idex = IDEXReg{Valid: false}
        return
    }
    inst := cpu.ifid.Instruction
    opcode := inst & 0x7F
    rs1 := (inst >> 15) & 0x1F
    rs2 := (inst >> 20) & 0x1F

    cpu.idex = IDEXReg{
        PC:     cpu.ifid.PC,
        RegA:   cpu.Regs[rs1],
        RegB:   cpu.Regs[rs2],
        Rd:     (inst >> 7) & 0x1F,
        Rs1:    rs1,
        Rs2:    rs2,
        Funct3: (inst >> 12) & 0x7,
        Funct7: (inst >> 25) & 0x7F,
        Opcode: opcode,
        Imm:    int32(inst) >> 20,
        Valid:  true,
    }

    // Set control signals based on opcode
    switch opcode {
    case 0b0110011: cpu.idex.RegWrite = true
    case 0b0010011: cpu.idex.ALUSrc = true; cpu.idex.RegWrite = true
    case 0b0000011: cpu.idex.ALUSrc = true; cpu.idex.MemRead = true
                    cpu.idex.MemToReg = true; cpu.idex.RegWrite = true
    case 0b0100011: cpu.idex.ALUSrc = true; cpu.idex.MemWrite = true
    case 0b1100011: cpu.idex.Branch = true
    }
}

func (cpu *PipelinedCPU) execute() {
    if !cpu.idex.Valid {
        cpu.exmem = EXMEMReg{Valid: false}
        return
    }
    a := cpu.idex.RegA
    b := cpu.idex.RegB
    if cpu.idex.ALUSrc {
        b = cpu.idex.Imm
    }
    result := a + b // Simplified: always ADD
    if cpu.idex.Opcode == 0b0110011 && cpu.idex.Funct7 == 0x20 {
        result = a - b
    }
    cpu.exmem = EXMEMReg{
        ALUResult: result,
        RegB:      cpu.idex.RegB,
        Rd:        cpu.idex.Rd,
        Zero:      result == 0,
        MemRead:   cpu.idex.MemRead,
        MemWrite:  cpu.idex.MemWrite,
        RegWrite:  cpu.idex.RegWrite,
        MemToReg:  cpu.idex.MemToReg,
        Branch:    cpu.idex.Branch,
        Valid:     true,
    }
}

func (cpu *PipelinedCPU) memoryAccess() {
    if !cpu.exmem.Valid {
        cpu.memwb = MEMWBReg{Valid: false}
        return
    }
    var memData int32
    addr := uint32(cpu.exmem.ALUResult) / 4
    if cpu.exmem.MemRead && addr < uint32(len(cpu.DMem)) {
        memData = cpu.DMem[addr]
    }
    if cpu.exmem.MemWrite && addr < uint32(len(cpu.DMem)) {
        cpu.DMem[addr] = cpu.exmem.RegB
    }
    cpu.memwb = MEMWBReg{
        ALUResult: cpu.exmem.ALUResult,
        MemData:   memData,
        Rd:        cpu.exmem.Rd,
        RegWrite:  cpu.exmem.RegWrite,
        MemToReg:  cpu.exmem.MemToReg,
        Valid:     true,
    }
}

func (cpu *PipelinedCPU) writeBack() {
    if !cpu.memwb.Valid || !cpu.memwb.RegWrite || cpu.memwb.Rd == 0 {
        return
    }
    if cpu.memwb.MemToReg {
        cpu.Regs[cpu.memwb.Rd] = cpu.memwb.MemData
    } else {
        cpu.Regs[cpu.memwb.Rd] = cpu.memwb.ALUResult
    }
    cpu.completed++
}

func main() {
    program := []uint32{
        0x00500093, // addi x1, x0, 5
        0x00300113, // addi x2, x0, 3
        0x00000013, // nop
        0x00000013, // nop
        0x002081B3, // add x3, x1, x2
    }
    cpu := &PipelinedCPU{IMem: program}
    cpu.Run(15)
    fmt.Printf("x1=%d x2=%d x3=%d\n", cpu.Regs[1], cpu.Regs[2], cpu.Regs[3])
    fmt.Printf("Completed: %d instructions in %d cycles\n", cpu.completed, cpu.cycles)
}`,
				},
				{
					Title: "Pipeline Hazards and Solutions",
					Content: `Pipeline hazards prevent the next instruction from executing in the next clock cycle. There are three types, and handling them efficiently is the key to high-performance processor design.

**1. Structural Hazards:**
` + "```" + `
Two instructions need the same hardware resource at the same time.

Example: Single-port memory used for both instruction fetch and data access

Time →   1    2    3    4    5
Load:   [IF] [ID] [EX] [MEM][WB]
Inst4:              [IF] [ID] [EX] ...
      ↑ conflict! Both need memory in cycle 4

Solutions:
  a) Separate instruction and data memories (Harvard architecture)
  b) Duplicate the resource
  c) Stall the pipeline (insert bubble)
  
Modern CPUs: Use separate I-cache and D-cache → no structural hazard
` + "```" + `

**2. Data Hazards:**
` + "```" + `
An instruction depends on data from a previous instruction still in the pipeline.

Types:
  RAW (Read After Write) — TRUE dependency:
    add x1, x2, x3    # writes x1
    sub x4, x1, x5    # reads x1 — needs value from ADD!

  WAR (Write After Read) — ANTI dependency:
    add x1, x2, x3    # reads x2
    sub x2, x4, x5    # writes x2
    (Not a problem in simple 5-stage pipeline — reads in ID, writes in WB)

  WAW (Write After Write) — OUTPUT dependency:
    add x1, x2, x3    # writes x1
    sub x1, x4, x5    # writes x1
    (Not a problem in in-order pipeline — writes happen in order)

RAW hazard timeline:
    Time →   1    2    3    4    5    6
    ADD x1: [IF] [ID] [EX] [MEM][WB]
                                  ↑ x1 written here
    SUB x4: [IF] [ID] ...
                   ↑ x1 read here — but not yet written!

Without forwarding: must stall 2 cycles
With forwarding: result available after EX, forward to next EX
` + "```" + `

**Data Forwarding (Bypassing):**
` + "```" + `
Forward ALU result from EX/MEM register to ALU input:

    ADD x1: [IF] [ID] [EX] [MEM][WB]
                        │
                    Result available!
                        │─── forward ───┐
                                        ↓
    SUB:         [IF] [ID] [EX] [MEM][WB]
                            ↑
                      Uses forwarded value — no stall!

Forward from MEM/WB to ALU input:
    ADD x1: [IF] [ID] [EX] [MEM][WB]
                              │── forward ──┐
    NOP:         [IF] [ID] [EX] [MEM][WB]  │
                                             ↓
    SUB:              [IF] [ID] [EX] [MEM][WB]

Forwarding paths needed:
  1. EX/MEM → EX input (1 cycle back)
  2. MEM/WB → EX input (2 cycles back)
  3. MEM/WB → MEM input (for store after load)
` + "```" + `

**Load-Use Hazard (unavoidable stall):**
` + "```" + `
    LW x1, 0(x2):  [IF] [ID] [EX] [MEM][WB]
                                     ↑ data available after MEM
    ADD x3, x1, x5: [IF] [ID] [EX]...
                               ↑ needs x1 in EX — but LW data not ready!

Even with forwarding, need 1 stall cycle:
    LW x1:  [IF] [ID] [EX] [MEM][WB]
                              │── forward ──┐
    bubble:      [IF] [ID] [stall]          │
                                             ↓
    ADD:              [IF]  [ID] [EX] [MEM][WB]

The compiler can avoid this by reordering instructions:
  Before:                 After:
    lw x1, 0(x2)           lw x1, 0(x2)
    add x3, x1, x5         ori x7, x8, 0x42  ← moved here
    ori x7, x8, 0x42       add x3, x1, x5    ← no stall!
` + "```" + `

**3. Control Hazards (Branch Hazards):**
` + "```" + `
Branch outcome not known until EX stage:
    BEQ x1,x2,L: [IF] [ID] [EX] ← branch decided here
    Inst (PC+4):      [IF] [ID]  ← already fetched!
    Inst (PC+8):           [IF]  ← already fetched!

If branch taken: 2 instructions fetched incorrectly → must flush

Solutions:
  a) Always stall: Insert 2 bubbles after every branch (simple but slow)
  
  b) Branch prediction: Guess branch outcome, flush if wrong
     Static: Predict not-taken, or backward-taken/forward-not-taken
     Dynamic: Use history to predict (covered in next module)
  
  c) Delayed branch: The instruction after branch ALWAYS executes
     (MIPS used this; RISC-V does not)
  
  d) Move branch decision to ID stage: Reduces penalty to 1 cycle
     Add comparator in ID stage; only 1 bubble if mispredicted

Branch penalty:
  5-stage pipeline: 1-2 cycles per misprediction
  20-stage pipeline: 15-20 cycles per misprediction!
  → Deep pipelines NEED good branch prediction
` + "```" + ``,
					CodeExamples: `// Data hazard detection and forwarding unit
package main

import "fmt"

type HazardUnit struct {
    // Pipeline register info for hazard detection
    idexRs1    uint32
    idexRs2    uint32
    exmemRd    uint32
    exmemRegWr bool
    memwbRd    uint32
    memwbRegWr bool
    exmemMemRd bool  // Load instruction in EX/MEM
}

type ForwardSel int
const (
    FWD_NONE ForwardSel = iota // Use register file value
    FWD_EX                     // Forward from EX/MEM
    FWD_MEM                    // Forward from MEM/WB
)

func (hu *HazardUnit) DetectForwarding() (fwdA, fwdB ForwardSel) {
    fwdA = FWD_NONE
    fwdB = FWD_NONE

    // EX hazard (forward from EX/MEM stage)
    if hu.exmemRegWr && hu.exmemRd != 0 {
        if hu.exmemRd == hu.idexRs1 { fwdA = FWD_EX }
        if hu.exmemRd == hu.idexRs2 { fwdB = FWD_EX }
    }

    // MEM hazard (forward from MEM/WB stage)
    // Only if EX hazard doesn't already handle it
    if hu.memwbRegWr && hu.memwbRd != 0 {
        if hu.memwbRd == hu.idexRs1 && fwdA == FWD_NONE { fwdA = FWD_MEM }
        if hu.memwbRd == hu.idexRs2 && fwdB == FWD_NONE { fwdB = FWD_MEM }
    }

    return
}

func (hu *HazardUnit) DetectLoadUseHazard() bool {
    // Load instruction in EX/MEM, and next instruction reads the loaded register
    if hu.exmemMemRd && hu.exmemRd != 0 {
        if hu.exmemRd == hu.idexRs1 || hu.exmemRd == hu.idexRs2 {
            return true // Must stall!
        }
    }
    return false
}

// Branch prediction statistics
type BranchPredictor struct {
    predictions int
    correct     int
    method      string
}

// Static: always predict not-taken
func (bp *BranchPredictor) PredictStatic(taken bool) {
    bp.predictions++
    bp.method = "static-not-taken"
    if !taken {
        bp.correct++
    }
}

func (bp *BranchPredictor) Accuracy() float64 {
    if bp.predictions == 0 { return 0 }
    return float64(bp.correct) / float64(bp.predictions) * 100
}

func main() {
    // Demonstrate hazard detection
    hu := &HazardUnit{
        idexRs1:    1,  // Next instruction reads x1
        idexRs2:    5,  // Next instruction reads x5
        exmemRd:    1,  // Previous instruction writes x1
        exmemRegWr: true,
        memwbRd:    5,
        memwbRegWr: true,
    }

    fwdA, fwdB := hu.DetectForwarding()
    fmt.Println("Hazard Detection:")
    fmt.Printf("  Forward A (rs1=x%d): %v\n", hu.idexRs1, fwdA)
    fmt.Printf("  Forward B (rs2=x%d): %v\n", hu.idexRs2, fwdB)

    // Load-use hazard
    hu2 := &HazardUnit{
        idexRs1:    3,
        exmemRd:    3,
        exmemRegWr: true,
        exmemMemRd: true, // Load instruction!
    }
    stall := hu2.DetectLoadUseHazard()
    fmt.Printf("\nLoad-use hazard (lw x3 followed by use of x3): stall=%v\n", stall)

    // Branch prediction simulation
    bp := &BranchPredictor{}
    // Simulate loop that iterates 10 times
    // Branch is taken 9 times (loop back), not taken once (exit)
    for i := 0; i < 10; i++ {
        taken := i < 9
        bp.PredictStatic(taken)
    }
    fmt.Printf("\nBranch prediction (static not-taken) for 10-iteration loop:\n")
    fmt.Printf("  Accuracy: %.1f%% (%d/%d correct)\n",
        bp.Accuracy(), bp.correct, bp.predictions)
    fmt.Println("  Note: Static not-taken is terrible for loops!")
    fmt.Println("  A 'backward-taken' heuristic would get 90% accuracy here.")
}`,
				},
			},
		},
		{
			ID:          2825,
			Title:       "Branch Prediction and Speculative Execution",
			Description: "Deep dive into branch prediction algorithms from simple 1-bit predictors to modern TAGE predictors, and how speculative execution enables high performance.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Dynamic Branch Prediction",
					Content: `Branch prediction is one of the most critical components in modern processors. A misprediction penalty of 15-20 cycles means even 5% misprediction rate can reduce performance by 40%.

**1-Bit Predictor:**
` + "```" + `
Branch History Table (BHT): array indexed by low bits of branch PC

    PC[9:2] → BHT[256 entries]
    Each entry: 1 bit (0 = predict not-taken, 1 = predict taken)

    On branch resolution:
        If correct: no change
        If wrong:   flip the bit

    Problem: Inner loop branch in nested loops
        for i: (always predict taken after first iteration)
            for j: [taken, taken, ..., NOT TAKEN, taken, taken, ...]
            
    The inner loop exit misses twice:
        1. When exiting (was predicting taken)
        2. On re-entry (now predicting not-taken, but it's taken!)
` + "```" + `

**2-Bit Saturating Counter Predictor:**
` + "```" + `
Each BHT entry is a 2-bit counter:
    
    State Machine:
    
    Strongly      Weakly       Weakly       Strongly
    Not-Taken    Not-Taken     Taken         Taken
    [00] ──taken──→ [01] ──taken──→ [10] ──taken──→ [11]
      ↑                                              │
      └──not-taken── [01] ←not-taken── [10] ←not-taken─┘

    Predict taken if counter ≥ 2 (bit[1] = 1)
    Need TWO consecutive mispredictions to change prediction

    Solves the nested loop problem:
        Exit: 11 → 10 (still predict taken! ✓)
        Re-entry: 10 → 11 (correct)
        Only 1 misprediction instead of 2

    Typical accuracy: ~85-90% for 2-bit, 4K-entry BHT
` + "```" + `

**Correlating Predictors (Two-Level):**
` + "```" + `
Key insight: Branch behavior often depends on OTHER branches.

Example:
    if (a == 0) ...    // Branch 1
    if (b == 0) ...    // Branch 2  
    if (a == b) ...    // Branch 3 depends on outcomes of 1 and 2!
    
    If both Branch 1 and 2 are taken → a==0 and b==0 → Branch 3 taken

(m,n) Predictor:
    - m-bit Global History Register (GHR): last m branch outcomes
    - For each GHR value: separate n-bit counter predictor
    - Total entries: 2^m × 2^n possibilities per branch
    
    GHR     Pattern   Prediction
    00  →   [counter] → predict based on when last 2 branches were NT,NT
    01  →   [counter] → predict based on NT,T pattern
    10  →   [counter] → predict based on T,NT pattern  
    11  →   [counter] → predict based on T,T pattern

(2,2) Predictor example:
    2-bit GHR + 2-bit counters
    Index = {GHR, PC[low bits]}
    Much more accurate than simple 2-bit predictor!
` + "```" + `

**Tournament Predictor:**
` + "```" + `
Use a meta-predictor to choose between a local and global predictor:

    ┌─────────────┐
    │   Global    │──→ prediction_g
    │  Predictor  │
    └─────────────┘
                        ┌──────────┐
                   ──→  │  Choice  │──→ Final Prediction
                        │ Predictor│
    ┌─────────────┐     └──────────┘
    │   Local     │──→ prediction_l
    │  Predictor  │
    └─────────────┘

    Choice Predictor: 2-bit counter per branch
        00,01: Use global predictor
        10,11: Use local predictor
    
    Update: If predictors disagree and one was right,
            move counter toward the correct one.

    Alpha 21264 (1998): Tournament predictor
        Local: 1024 10-bit history → 1024 3-bit counters
        Global: 12-bit GHR → 4096 2-bit counters
        Choice: 4096 2-bit counters
        Accuracy: ~95%+
` + "```" + `

**TAGE Predictor (TAgged GEometric):**
` + "```" + `
The state-of-the-art predictor used in modern high-performance CPUs.

Key idea: Use multiple tables with GEOMETRICALLY increasing history lengths.

    History length:    0    5    15    44    130    385
                       │    │     │     │      │      │
                       ▼    ▼     ▼     ▼      ▼      ▼
                     [T0] [T1]  [T2]  [T3]   [T4]   [T5]
                       │    │     │     │      │      │
                       └────┴─────┴─────┴──────┴──────┘
                                    │
                            Longest match wins
                            (with tag verification)

    Each table entry: {tag, 3-bit counter, useful bit}
    Lookup: hash(PC, history[0:L]) for each table
    Prediction: Use entry from table with longest matching history

    Benefits:
    - Adapts to branches with different history depths
    - Short-history branches use T0/T1
    - Loop branches use medium tables
    - Deep correlations use long-history tables
    
    Accuracy: 97-99%+ on typical workloads
    Used in: Intel Skylake/Ice Lake, AMD Zen, ARM Cortex
` + "```" + ``,
					CodeExamples: `// Branch predictors implementation in Go
package main

import "fmt"

// 2-bit Saturating Counter Predictor
type TwoBitPredictor struct {
    table     []uint8 // 2-bit counters (0-3)
    size      int
    correct   int
    total     int
}

func NewTwoBitPredictor(size int) *TwoBitPredictor {
    table := make([]uint8, size)
    for i := range table {
        table[i] = 1 // Weakly not-taken
    }
    return &TwoBitPredictor{table: table, size: size}
}

func (p *TwoBitPredictor) Predict(pc uint32) bool {
    idx := pc % uint32(p.size)
    return p.table[idx] >= 2
}

func (p *TwoBitPredictor) Update(pc uint32, taken bool) {
    idx := pc % uint32(p.size)
    predicted := p.table[idx] >= 2
    p.total++
    if predicted == taken { p.correct++ }
    if taken {
        if p.table[idx] < 3 { p.table[idx]++ }
    } else {
        if p.table[idx] > 0 { p.table[idx]-- }
    }
}

// Correlating (GShare) Predictor
type GSharePredictor struct {
    table   []uint8
    size    int
    ghr     uint32 // Global History Register
    ghrBits int
    correct int
    total   int
}

func NewGSharePredictor(tableBits, ghrBits int) *GSharePredictor {
    size := 1 << tableBits
    table := make([]uint8, size)
    for i := range table { table[i] = 1 }
    return &GSharePredictor{
        table: table, size: size,
        ghrBits: ghrBits,
    }
}

func (p *GSharePredictor) index(pc uint32) int {
    return int((pc ^ p.ghr) % uint32(p.size))
}

func (p *GSharePredictor) Predict(pc uint32) bool {
    return p.table[p.index(pc)] >= 2
}

func (p *GSharePredictor) Update(pc uint32, taken bool) {
    idx := p.index(pc)
    predicted := p.table[idx] >= 2
    p.total++
    if predicted == taken { p.correct++ }
    if taken {
        if p.table[idx] < 3 { p.table[idx]++ }
    } else {
        if p.table[idx] > 0 { p.table[idx]-- }
    }
    // Update GHR
    p.ghr = (p.ghr << 1) | btou(taken)
    p.ghr &= (1 << p.ghrBits) - 1
}

// Tournament Predictor
type TournamentPredictor struct {
    local   *TwoBitPredictor
    global  *GSharePredictor
    chooser []uint8 // 2-bit: 0-1=global, 2-3=local
    size    int
    correct int
    total   int
}

func NewTournamentPredictor(size, ghrBits int) *TournamentPredictor {
    chooser := make([]uint8, size)
    for i := range chooser { chooser[i] = 1 } // Start with global
    return &TournamentPredictor{
        local:   NewTwoBitPredictor(size),
        global:  NewGSharePredictor(12, ghrBits),
        chooser: chooser,
        size:    size,
    }
}

func (p *TournamentPredictor) Predict(pc uint32) bool {
    idx := pc % uint32(p.size)
    if p.chooser[idx] >= 2 {
        return p.local.Predict(pc)
    }
    return p.global.Predict(pc)
}

func (p *TournamentPredictor) Update(pc uint32, taken bool) {
    idx := pc % uint32(p.size)
    localPred := p.local.Predict(pc)
    globalPred := p.global.Predict(pc)

    predicted := globalPred
    if p.chooser[idx] >= 2 { predicted = localPred }
    p.total++
    if predicted == taken { p.correct++ }

    // Update chooser when predictors disagree
    if localPred != globalPred {
        if localPred == taken && p.chooser[idx] < 3 {
            p.chooser[idx]++
        }
        if globalPred == taken && p.chooser[idx] > 0 {
            p.chooser[idx]--
        }
    }

    p.local.Update(pc, taken)
    p.global.Update(pc, taken)
}

func btou(b bool) uint32 { if b { return 1 }; return 0 }

func main() {
    predictors := []struct{
        name string
        predict func(uint32) bool
        update  func(uint32, bool)
        accuracy func() float64
    }{
        {"2-Bit",
            NewTwoBitPredictor(1024).Predict,
            NewTwoBitPredictor(1024).Update,
            nil},
    }
    _ = predictors

    // Simple benchmark: nested loop pattern
    bp2 := NewTwoBitPredictor(1024)
    gshare := NewGSharePredictor(12, 10)
    tournament := NewTournamentPredictor(1024, 10)

    // Simulate: outer loop 10 iter, inner loop 20 iter
    outerPC := uint32(0x1000)
    innerPC := uint32(0x1010)
    ifPC := uint32(0x1020)

    for i := 0; i < 10; i++ {
        for j := 0; j < 20; j++ {
            innerTaken := j < 19
            bp2.Update(innerPC, innerTaken)
            gshare.Update(innerPC, innerTaken)
            tournament.Update(innerPC, innerTaken)

            condTaken := (i+j)%3 == 0
            bp2.Update(ifPC, condTaken)
            gshare.Update(ifPC, condTaken)
            tournament.Update(ifPC, condTaken)
        }
        outerTaken := i < 9
        bp2.Update(outerPC, outerTaken)
        gshare.Update(outerPC, outerTaken)
        tournament.Update(outerPC, outerTaken)
    }

    fmt.Println("Branch Prediction Results:")
    fmt.Printf("  2-Bit BHT:   %d/%d correct (%.1f%%)\n",
        bp2.correct, bp2.total, float64(bp2.correct)/float64(bp2.total)*100)
    fmt.Printf("  GShare:      %d/%d correct (%.1f%%)\n",
        gshare.correct, gshare.total, float64(gshare.correct)/float64(gshare.total)*100)
    fmt.Printf("  Tournament:  %d/%d correct (%.1f%%)\n",
        tournament.correct, tournament.total, float64(tournament.correct)/float64(tournament.total)*100)
}`,
				},
				{
					Title: "Speculative Execution and Out-of-Order Processing",
					Content: `Modern processors don't just predict branches — they speculatively execute instructions past predicted branches and even reorder instructions for maximum throughput. This is the foundation of superscalar execution.

**Speculative Execution:**
` + "```" + `
Without speculation:
    BEQ x1, x2, target
    [STALL][STALL]              ← Wait for branch resolution
    next_instruction            ← Only then continue

With speculation:
    BEQ x1, x2, target
    [predicted path instructions executed speculatively]
    ... (branch resolves)
    If prediction CORRECT: speculative results become real → no penalty!
    If prediction WRONG:   flush speculative results → penalty = pipeline depth

Key requirement: Speculative changes must be reversible
  - Register writes: write to physical registers, only commit when confirmed
  - Memory writes: buffer in store queue, only write to cache when committed
  - Exceptions: suppress until instruction confirmed non-speculative
` + "```" + `

**Out-of-Order Execution:**
` + "```" + `
In-order pipeline wastes time waiting for long-latency operations:
    DIV  x1, x2, x3    # Takes 20 cycles!
    ADD  x4, x1, x5    # Depends on DIV — must wait
    MUL  x6, x7, x8    # Independent! But stalled behind ADD
    OR   x9, x10, x11  # Independent! Also stalled

Out-of-order allows MUL and OR to execute while DIV is running:
    Cycle 1:  DIV starts executing (20 cycles)
    Cycle 2:  MUL can execute (independent of DIV)
    Cycle 3:  OR can execute (independent of DIV)
    Cycle 20: DIV completes
    Cycle 21: ADD can execute (now has DIV result)

Speedup: Instructions that would have waited can execute immediately
` + "```" + `

**Tomasulo's Algorithm:**
` + "```" + `
The classic algorithm for out-of-order execution (IBM 360/91, 1967):

Key structures:
  1. Reservation Stations (RS): Hold instructions waiting to execute
  2. Common Data Bus (CDB): Broadcasts results to all waiting RS
  3. Register Alias Table (RAT): Maps architectural → physical registers

Instruction lifecycle:
  1. ISSUE: Fetch instruction, allocate RS entry
     - If operands ready: copy values into RS
     - If operands not ready: record which RS will produce them (tag)
  
  2. EXECUTE: When all operands available, execute
     - Multiple instructions can execute simultaneously
     - No need to wait for instructions issued earlier
  
  3. WRITE RESULT: Broadcast result on CDB
     - All RS entries watching for this tag receive the value
     - Update register file

Example:
    Instruction    Issue  Execute  Write
    MUL  F0,F2,F4    1      2-11     12
    ADD  F6,F0,F8    2    (wait)     14    ← Waits for MUL result
    DIV  F8,F0,F6    3    (wait)     ?     ← Waits for both
    ADD  F2,F8,F4    4    (wait)     ?     ← Waits for DIV
    SUB  F10,F2,F6   5      6       7      ← Can execute immediately!
` + "```" + `

**Register Renaming:**
` + "```" + `
Eliminates WAW and WAR hazards by giving each instruction its own
physical register for its destination:

Architectural registers: x0-x31 (programmer sees these)
Physical registers: p0-p127 (hardware has many more)

Before renaming:
    ADD x1, x2, x3    # WAR: reads x2
    SUB x2, x4, x5    # WAR: writes x2 (would destroy x2 before ADD reads it)
    MUL x1, x6, x7    # WAW: writes x1 (which ADD also writes)

After renaming:
    ADD p32, p2, p3    # x1 → p32
    SUB p33, p4, p5    # x2 → p33 (new physical reg for x2)
    MUL p34, p6, p7    # x1 → p34 (new physical reg for x1)

Now there are NO dependencies between SUB and ADD (different physical regs)
and MUL can execute before ADD completes (different physical regs)!

Free List: Pool of available physical registers
RAT (Register Alias Table): x_reg → current physical register mapping
ROB (Reorder Buffer): Ensures instructions commit in program order
` + "```" + `

**Reorder Buffer (ROB):**
` + "```" + `
Even though instructions execute out of order, they must COMMIT in order
(for precise exceptions and correct program state):

    ROB entries (circular buffer):
    ┌─────┬──────────┬────────┬───────────┬──────────┐
    │ ROB#│ Instr    │ Dest   │ Value     │ Complete │
    ├─────┼──────────┼────────┼───────────┼──────────┤
    │  1  │ MUL      │ p32    │ 42        │ ✓        │
    │  2  │ ADD      │ p33    │ pending   │          │
    │  3  │ DIV      │ p34    │ 7         │ ✓        │  ← Done but can't commit
    │  4  │ SUB      │ p35    │ 15        │ ✓        │  ← Done but can't commit
    └─────┴──────────┴────────┴───────────┴──────────┘
    Head ↑                                           ↑ Tail
    
    Commit pointer at head:
    - ROB #1 complete → COMMIT (update architectural state, free old phys reg)
    - ROB #2 not complete → STALL commit (but execution continues!)
    - ROB #3 and #4 done but can't commit until #2 commits
    
    If exception at ROB #2:
    - Flush ROB #2, #3, #4 (even though #3/#4 completed)
    - Restore architectural state to ROB #1
    → Precise exceptions maintained!
` + "```" + `

**Spectre and Meltdown:**
` + "```" + `
Speculative execution can leak information through side channels:

Spectre (2018):
    1. Attacker trains branch predictor to predict a specific direction
    2. Victim speculatively accesses secret data past a bounds check
    3. Speculative access loads data into cache based on secret value  
    4. Even after flush, cache timing reveals which line was loaded
    5. Attacker measures cache timing → deduces secret data

    // Victim code:
    if (x < array1_size) {           // Bounds check
        y = array2[array1[x] * 256]; // Speculative access with secret as index
    }
    // Speculatively: array1[x] reads secret byte S
    //   array2[S * 256] gets cached
    //   Attacker probes array2 timing to find S

Mitigations:
  - Retpoline: Replace indirect branches with return-based sequences
  - IBRS/STIBP: Hardware indirect branch prediction barriers
  - Speculation barriers (lfence, csdb)
  - Site isolation in browsers
  - Kernel page table isolation (KPTI/KAISER) for Meltdown
` + "```" + ``,
					CodeExamples: `// Out-of-order execution simulator (Tomasulo-style)
package main

import "fmt"

type RSEntry struct {
    busy    bool
    op      string
    vj, vk  int     // Operand values
    qj, qk int     // RS that will produce operand (-1 = ready)
    dest    int     // Physical register destination
    cycles  int     // Execution cycles remaining
    result  int
}

type OoOProcessor struct {
    rs       [8]RSEntry   // Reservation stations
    physRegs [64]int      // Physical registers
    rat      [32]int      // Register Alias Table (arch → phys)
    busy     [64]bool     // Physical register busy flags
    nextPhys int
    cycle    int
    committed int
}

func NewOoOProcessor() *OoOProcessor {
    p := &OoOProcessor{nextPhys: 32}
    for i := range p.rat {
        p.rat[i] = i // Initially arch reg i = phys reg i
    }
    return p
}

func (p *OoOProcessor) allocPhysReg() int {
    reg := p.nextPhys
    p.nextPhys++
    return reg
}

func (p *OoOProcessor) findFreeRS() int {
    for i := range p.rs {
        if !p.rs[i].busy { return i }
    }
    return -1
}

func (p *OoOProcessor) Issue(op string, rd, rs1, rs2, latency int) bool {
    rsIdx := p.findFreeRS()
    if rsIdx < 0 { return false } // Structural hazard

    entry := &p.rs[rsIdx]
    entry.busy = true
    entry.op = op
    entry.cycles = latency

    // Rename destination
    newPhys := p.allocPhysReg()
    entry.dest = newPhys

    // Read source operands (with renaming)
    phys1 := p.rat[rs1]
    if p.busy[phys1] {
        entry.qj = phys1 // Will get value when it broadcasts
    } else {
        entry.vj = p.physRegs[phys1]
        entry.qj = -1
    }

    phys2 := p.rat[rs2]
    if p.busy[phys2] {
        entry.qk = phys2
    } else {
        entry.vk = p.physRegs[phys2]
        entry.qk = -1
    }

    // Update RAT for destination
    p.rat[rd] = newPhys
    p.busy[newPhys] = true

    fmt.Printf("  Cycle %d: ISSUE %s → RS[%d] (p%d = p%d %s p%d)\n",
        p.cycle, op, rsIdx, newPhys, phys1, op, phys2)
    return true
}

func (p *OoOProcessor) Execute() {
    for i := range p.rs {
        if !p.rs[i].busy { continue }
        if p.rs[i].qj != -1 || p.rs[i].qk != -1 { continue } // Operands not ready
        
        if p.rs[i].cycles > 0 {
            p.rs[i].cycles--
            if p.rs[i].cycles == 0 {
                // Compute result
                switch p.rs[i].op {
                case "ADD": p.rs[i].result = p.rs[i].vj + p.rs[i].vk
                case "MUL": p.rs[i].result = p.rs[i].vj * p.rs[i].vk
                case "SUB": p.rs[i].result = p.rs[i].vj - p.rs[i].vk
                }
            }
        }
    }
}

func (p *OoOProcessor) WriteBack() {
    for i := range p.rs {
        if !p.rs[i].busy || p.rs[i].cycles > 0 || p.rs[i].qj != -1 || p.rs[i].qk != -1 {
            continue
        }
        
        // Broadcast result
        dest := p.rs[i].dest
        p.physRegs[dest] = p.rs[i].result
        p.busy[dest] = false

        fmt.Printf("  Cycle %d: WRITEBACK RS[%d] → p%d = %d\n",
            p.cycle, i, dest, p.rs[i].result)

        // Wake up waiting RS entries
        for j := range p.rs {
            if p.rs[j].qj == dest {
                p.rs[j].vj = p.rs[i].result
                p.rs[j].qj = -1
            }
            if p.rs[j].qk == dest {
                p.rs[j].vk = p.rs[i].result
                p.rs[j].qk = -1
            }
        }

        p.rs[i].busy = false
        p.committed++
    }
}

func main() {
    proc := NewOoOProcessor()
    
    // Set initial register values
    proc.physRegs[1] = 10  // x1 = 10
    proc.physRegs[2] = 20  // x2 = 20
    proc.physRegs[3] = 5   // x3 = 5
    proc.physRegs[4] = 3   // x4 = 3
    proc.physRegs[5] = 7   // x5 = 7

    fmt.Println("Out-of-Order Execution Simulation:")
    fmt.Println("Instructions: MUL x6,x1,x2 | ADD x7,x6,x3 | SUB x8,x4,x5")
    fmt.Println()

    // Issue instructions
    proc.cycle = 1
    proc.Issue("MUL", 6, 1, 2, 3)  // MUL x6, x1, x2 (3 cycles)
    proc.Issue("ADD", 7, 6, 3, 1)  // ADD x7, x6, x3 (depends on MUL)
    proc.Issue("SUB", 8, 4, 5, 1)  // SUB x8, x4, x5 (independent!)

    // Run pipeline
    for cycle := 2; cycle <= 8; cycle++ {
        proc.cycle = cycle
        proc.Execute()
        proc.WriteBack()
    }

    fmt.Printf("\nFinal: x6(MUL)=%d, x7(ADD)=%d, x8(SUB)=%d\n",
        proc.physRegs[proc.rat[6]],
        proc.physRegs[proc.rat[7]],
        proc.physRegs[proc.rat[8]])
    fmt.Println("Note: SUB completed before ADD (out-of-order!)")
}`,
				},
			},
		},
	})
}
