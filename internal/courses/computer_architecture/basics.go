package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          1300,
			Title:       "Introduction to Computer Architecture",
			Description: "Learn the fundamentals of computer architecture: CPU design, instruction sets, memory hierarchy, logic gates, number systems, and basic hardware components.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Computer Architecture?",
					Content: `Computer Architecture is the blueprint for how hardware and software work together. It is the contract that ensures your code actually runs on physical transistors.

**Why does it matter?**
*   **Performance:** Write code that leverages the hardware.
*   **Efficiency:** Understand why certain algorithms are slow (e.g., cache misses).
*   **Specialization:** Why do we need GPUs for AI but CPUs for general tasks?
*   **Debugging:** Hardware-related bugs like race conditions and memory-ordering issues are invisible in source code.

**Historical Evolution:**
1.  **1945:** Von Neumann Architecture (The stored-program concept).
2.  **1947:** Invention of the transistor at Bell Labs.
3.  **1958:** First integrated circuits (Jack Kilby, Texas Instruments).
4.  **1971:** Intel 4004 -- first commercial single-chip microprocessor.
5.  **1978:** x86 lineage begins with Intel 8086.
6.  **1990s:** Pipelining and Superscalar CPUs (doing multiple things at once).
7.  **2005:** Intel shifts to multi-core (Pentium D) after hitting the power wall.
8.  **2010s-2020s:** Domain-specific accelerators (GPUs, TPUs, NPUs, FPGAs).

**Levels of Abstraction:**
*   **Application Software** -- what the user sees.
*   **System Software** -- OS, compilers, drivers.
*   **ISA (Instruction Set Architecture)** -- the hardware/software boundary.
*   **Microarchitecture** -- how the ISA is implemented.
*   **Logic and Circuits** -- gates, flip-flops, adders.
*   **Transistors and Physics** -- CMOS, voltage, current.

Each level hides the complexity below it, but performance-conscious engineers need to understand at least two levels above and below their usual layer.`,
					CodeExamples: `// Example: Understanding instruction execution
// High-level code:
int sum = a + b;

// Assembly (x86-64):
mov eax, [a]      // Load value of a into register EAX
add eax, [b]      // Add value of b to EAX
mov [sum], eax    // Store result in sum

// Machine code (binary):
// 8B 05 [address]  // mov instruction
// 03 05 [address]  // add instruction
// 89 05 [address]  // mov instruction

// CPU execution steps:
// 1. Fetch instruction from memory
// 2. Decode instruction
// 3. Fetch operands
// 4. Execute operation
// 5. Store result

// Levels at work:
// C code:         sum = a + b
// Assembly:       add eax, ebx
// Microarch:      Pipeline stage EX fires the ALU
// Logic:          Ripple-carry adder toggles gates
// Transistors:    CMOS switches flip in ~0.3 ns`,
				},
				{
					Title: "CPU Architecture and The von Neumann Model",
					Content: `The CPU is the brain of the computer. Most modern computers follow the von Neumann Architecture, where both data and instructions are stored in the same memory.

**1. The von Neumann Model:**
The key components are the Control Unit, ALU, registers, memory, and I/O. Instructions and data share the same memory bus, which creates the von Neumann bottleneck -- the memory bus is the single pipeline for both code and data.

**2. Key CPU Components:**
*   **Control Unit (CU):** The conductor. It fetches instructions, decodes them, and orchestrates the other units.
*   **ALU (Arithmetic Logic Unit):** The calculator. It performs math (ADD, SUB, MUL, DIV) and logic (AND, OR, XOR, NOT, shift).
*   **Registers:** Tiny, ultra-fast storage cells inside the CPU. Examples: RAX (accumulator), RSP (stack pointer), RIP (instruction pointer).
*   **Program Counter (PC / RIP):** A special register that holds the address of the next instruction to execute.
*   **Instruction Register (IR):** Holds the instruction currently being decoded.
*   **Memory Address Register (MAR):** Holds the address of the memory location being accessed.
*   **Memory Data Register (MDR):** Holds the data read from or written to memory.

**3. The Fetch-Decode-Execute Cycle:**
1.  **Fetch:** Load the next instruction from memory using the PC.
2.  **Decode:** Translate binary opcode into control signals for the ALU/CU.
3.  **Execute:** Perform the arithmetic, logic, or memory operation.
4.  **Writeback:** Save the result to a register or memory, increment PC.

This cycle repeats billions of times per second on a modern CPU (measured in GHz -- a 4 GHz CPU does 4 billion cycles per second).

**4. Harvard Architecture (alternative):**
*   Separate memory buses for instructions and data.
*   Eliminates the von Neumann bottleneck.
*   Used in many embedded systems and DSPs.
*   Modern CPUs use a modified Harvard: separate L1 instruction and data caches feed a unified L2/L3.`,
					CodeExamples: `// x86-64 Register Example:
mov rax, 10      // Store 10 in register RAX (64-bit)
add rax, 20      // Add 20 to RAX (RAX is now 30)

// Controlling flow:
cmp rax, 30      // Compare RAX to 30 (sets flags)
je  is_thirty    // Jump to 'is_thirty' if equal (checks ZF flag)

// Key x86-64 General-Purpose Registers:
// RAX - Accumulator (return values)
// RBX - Base register
// RCX - Counter (loop counts)
// RDX - Data register
// RSI - Source index (string ops)
// RDI - Destination index (string ops)
// RSP - Stack pointer (top of stack)
// RBP - Base pointer (stack frame)
// R8-R15 - Additional GP registers (x86-64 only)
// RIP - Instruction pointer (current instruction)
// RFLAGS - Status flags (ZF, CF, SF, OF)`,
				},
				{
					Title: "Logic Gates and Boolean Algebra",
					Content: `All computation reduces to combinations of logic gates, which implement Boolean algebra on binary signals (0 and 1).

**1. Basic Gates:**
*   **NOT (Inverter):** Output is opposite of input. NOT(0) = 1, NOT(1) = 0.
*   **AND:** Output is 1 only if BOTH inputs are 1. 1 AND 1 = 1; all others = 0.
*   **OR:** Output is 1 if EITHER input is 1. 0 OR 0 = 0; all others = 1.
*   **XOR (Exclusive OR):** Output is 1 if inputs DIFFER. 0 XOR 1 = 1; same inputs = 0.
*   **NAND:** NOT(AND). Universal gate -- can build any other gate from NANDs alone.
*   **NOR:** NOT(OR). Also universal.

**2. Boolean Algebra Laws:**
*   **Identity:** A AND 1 = A, A OR 0 = A.
*   **Null:** A AND 0 = 0, A OR 1 = 1.
*   **Complement:** A AND NOT(A) = 0, A OR NOT(A) = 1.
*   **De Morgan's:** NOT(A AND B) = NOT(A) OR NOT(B), NOT(A OR B) = NOT(A) AND NOT(B).
*   **Distributive:** A AND (B OR C) = (A AND B) OR (A AND C).

**3. Building Blocks from Gates:**
*   **Half Adder:** XOR for sum, AND for carry. Adds two 1-bit numbers.
*   **Full Adder:** Adds three bits (two inputs plus carry-in). Chains to build N-bit adders.
*   **Multiplexer (MUX):** Selects one of many inputs based on select lines. A 2:1 MUX uses AND, OR, NOT.
*   **Decoder:** Converts N select lines into 2^N output lines (exactly one active).
*   **Flip-Flop:** Stores one bit of state. The basic building block of registers and SRAM.

**4. Combinational vs Sequential Logic:**
*   **Combinational:** Output depends only on current inputs (gates, adders, MUXes).
*   **Sequential:** Output depends on current inputs AND stored state (flip-flops, counters, registers).

Sequential circuits need a clock signal to synchronize state changes. The clock frequency determines how fast the circuit can operate.`,
					CodeExamples: `// Truth Tables:
//
// AND:          OR:           XOR:          NAND:
// A B | Out     A B | Out     A B | Out     A B | Out
// 0 0 |  0     0 0 |  0     0 0 |  0     0 0 |  1
// 0 1 |  0     0 1 |  1     0 1 |  1     0 1 |  1
// 1 0 |  0     1 0 |  1     1 0 |  1     1 0 |  1
// 1 1 |  1     1 1 |  1     1 1 |  0     1 1 |  0

// Half Adder (2 inputs: A, B):
// Sum   = A XOR B
// Carry = A AND B

// Full Adder (3 inputs: A, B, Cin):
// Sum   = A XOR B XOR Cin
// Cout  = (A AND B) OR (Cin AND (A XOR B))

// 4-bit Ripple-Carry Adder:
// Chain 4 full adders, carry-out feeds next carry-in.
// Latency = 4 * gate_delay (slow for large widths)
// Solution: Carry-Lookahead Adder (CLA) computes carries in parallel`,
				},
				{
					Title: "Number Systems and Binary Arithmetic",
					Content: `Computers use binary (base 2). Understanding number systems is essential for assembly, debugging, and hardware design.

**1. Number Systems:**
*   **Binary (base 2):** Digits 0-1. Each position is a power of 2. Example: 1011 = 8 + 0 + 2 + 1 = 11.
*   **Octal (base 8):** Digits 0-7. Used in Unix permissions (chmod 755). Each octal digit = 3 binary bits.
*   **Hexadecimal (base 16):** Digits 0-9, A-F. Compact binary representation. Each hex digit = 4 binary bits.
*   **Converting:** Group binary digits in sets of 4 for hex, 3 for octal. 0b11010110 = 0xD6 = 0o326.

**2. Unsigned Integers:**
*   N bits represent values 0 to 2^N - 1.
*   8 bits: 0 to 255. 16 bits: 0 to 65,535. 32 bits: 0 to ~4.3 billion. 64 bits: 0 to ~1.8 x 10^19.

**3. Signed Integers -- Two's Complement:**
Two's complement is the universal standard for signed integers.
*   **Positive numbers:** Same as unsigned. The MSB (most significant bit) is 0.
*   **Negative numbers:** MSB is 1. To negate: invert all bits and add 1.
*   **Range for N bits:** -2^(N-1) to 2^(N-1) - 1.
*   **8-bit example:** +5 = 00000101. Invert: 11111010. Add 1: 11111011 = -5.
*   **Advantage:** Addition and subtraction use the same hardware circuit. No special case for negative numbers.

**4. Overflow and Underflow:**
*   Overflow: result is too large for the bit width (e.g., 127 + 1 = -128 in 8-bit signed).
*   Underflow: result is too small (or too close to zero in floating point).
*   CPUs set the overflow flag (OF) and carry flag (CF) to detect these.

**5. Floating Point -- IEEE 754:**
Used for real numbers. Works like scientific notation: 1.mantissa * 2^exponent.
*   **Single precision (float):** 32 bits = 1 sign + 8 exponent + 23 mantissa.
*   **Double precision (double):** 64 bits = 1 sign + 11 exponent + 52 mantissa.
*   **Special values:** +/-Infinity, NaN (Not a Number), denormalized (subnormal) numbers near zero.
*   **Precision issues:** 0.1 + 0.2 != 0.3 because 0.1 has no exact binary representation. Always compare floats with an epsilon tolerance.

**6. Character Encoding:**
*   **ASCII:** 7-bit (128 characters). Basic English letters, digits, punctuation.
*   **Unicode:** Assigns a code point to every character in every language plus emoji.
*   **UTF-8:** Variable-length encoding (1-4 bytes). ASCII-compatible. Dominant on the web.
*   **UTF-16:** 2 or 4 bytes per character. Used internally by Windows and Java.`,
					CodeExamples: `// Number system conversions:
// Decimal 42 in different bases:
// Binary:       0b00101010
// Octal:        0o52
// Hexadecimal:  0x2A

// Two's complement (8-bit):
// +42 = 0010 1010
// -42: invert = 1101 0101, add 1 = 1101 0110

// IEEE 754 single precision (32-bit):
// Sign | Exponent | Mantissa
//  1   |  8 bits  | 23 bits
//
// Example: -6.5
// Sign: 1 (negative)
// 6.5 = 110.1 in binary = 1.101 * 2^2
// Exponent: 2 + 127 (bias) = 129 = 10000001
// Mantissa: 10100000000000000000000
// Result: 1 10000001 10100000000000000000000

// Bitwise operations:
int a = 0b1010; // 10
int b = 0b1100; // 12
int c = a & b;  // AND: 0b1000 = 8
int d = a | b;  // OR:  0b1110 = 14
int e = a ^ b;  // XOR: 0b0110 = 6
int f = ~a;     // NOT: inverts all bits
int g = a << 2; // Left shift: 0b101000 = 40 (multiply by 4)
int h = a >> 1; // Right shift: 0b0101 = 5 (divide by 2)`,
				},
				{
					Title: "Memory Hierarchy and Organization",
					Content: `Memory is a trade-off between Speed and Cost. Modern systems use a hierarchy to give the illusion of a massive, super-fast memory.

**1. The Memory Pyramid (Fastest to Slowest):**
*   **CPU Registers:** ~0.3 ns, ~1 KB, cost: highest.
*   **L1 Cache:** ~1 ns, 32-64 KB per core (split I-cache + D-cache).
*   **L2 Cache:** ~4-10 ns, 256 KB-1 MB per core.
*   **L3 Cache:** ~10-40 ns, 4-64 MB shared across cores.
*   **Main Memory (DRAM):** ~50-100 ns, 8-256 GB.
*   **SSD:** ~25-100 us (microseconds), 256 GB-8 TB.
*   **HDD:** ~3-10 ms (milliseconds), 1-20 TB.

Each level is roughly 10x slower and 10x larger than the one above it.

**2. Locality of Reference (Why Hierarchies Work):**
*   **Temporal Locality:** If you access data once, you will likely access it again soon. Keeping recently used data in cache exploits this.
*   **Spatial Locality:** If you access address X, you will likely access X+1, X+2, etc. Cache lines (typically 64 bytes) exploit this by loading a block of adjacent memory.

**3. Volatile vs Non-Volatile Memory:**
*   **Volatile:** Loses data when powered off (registers, SRAM caches, DRAM).
*   **Non-Volatile:** Retains data (Flash/SSD, HDD, ROM, NOR/NAND flash).

**4. Memory Technologies:**
*   **SRAM (Static RAM):** 6 transistors per bit. Fast, no refresh needed. Used for caches.
*   **DRAM (Dynamic RAM):** 1 transistor + 1 capacitor per bit. Cheap, dense, but needs periodic refresh (every ~64 ms). Used for main memory.
*   **Flash (NAND):** Non-volatile, block-erasable. Used for SSDs and USB drives.

**5. Memory Layout of a Program:**
*   **Text (Code):** Compiled instructions (read-only).
*   **Data (.data / .bss):** Global and static variables.
*   **Heap:** Dynamically allocated memory (malloc/new). Grows upward.
*   **Stack:** Function call frames (local variables, return addresses). Grows downward.
*   Stack and heap grow toward each other; collisions cause stack overflow.

**6. Endianness:**
*   **Little-Endian:** Least significant byte at lowest address. Used by x86, ARM (default).
*   **Big-Endian:** Most significant byte at lowest address. Used by network protocols (TCP/IP).
*   Value 0x12345678 stored at address 0x100:
    *   Little-endian: 78 56 34 12
    *   Big-endian: 12 34 56 78`,
					CodeExamples: `// Memory hierarchy latency comparison:
// Register access   : ~0.3 ns   (1 cycle)
// L1 cache hit      : ~1 ns     (3-4 cycles)
// L2 cache hit      : ~4 ns     (12 cycles)
// L3 cache hit      : ~12 ns    (40 cycles)
// DRAM access       : ~60 ns    (200 cycles)
// SSD random read   : ~25 us    (25,000 ns)
// HDD random read   : ~5 ms     (5,000,000 ns)

// C-style memory placement:
int global_var = 42;       // .data segment
int uninitialized;         // .bss segment (zero-initialized)
const char* str = "hello"; // .rodata segment

void function() {
    int local = 10;            // Stack
    int* ptr = malloc(100);    // Heap
    free(ptr);                 // Return to heap
}

// Spatial locality example:
int arr[1000];
int sum = 0;
for (int i = 0; i < 1000; i++)
    sum += arr[i]; // Sequential access = cache-friendly

// Temporal locality example:
for (int iter = 0; iter < 100; iter++)
    sum += arr[0]; // Same location accessed repeatedly`,
				},
				{
					Title: "Instruction Set Architecture (ISA)",
					Content: `The ISA is the language the CPU speaks. It defines the available instructions, registers, addressing modes, and data types. The ISA is the boundary between hardware and software.

**1. CISC vs. RISC:**
*   **CISC (Complex Instruction Set Computer):**
    *   Goal: Do more in a single instruction (minimize code size).
    *   Variable-length instructions (1-15 bytes for x86).
    *   Examples: x86/x86-64 (Intel/AMD).
    *   Pros: Rich instruction set, dense code, backward compatible.
    *   Cons: Complex decode logic, harder to pipeline.
*   **RISC (Reduced Instruction Set Computer):**
    *   Goal: Simple, fast instructions. One instruction per cycle.
    *   Fixed-length instructions (4 bytes for ARM, RISC-V).
    *   Examples: ARM (Apple M-series, phones), RISC-V (open standard), MIPS.
    *   Pros: Energy efficient, easier to pipeline, simpler hardware.
    *   Cons: More instructions needed for the same task, larger code size.

**Modern Reality:** Intel/AMD CPUs are CISC on the outside but decode x86 instructions into internal RISC-like micro-operations (uops) for execution. So the CISC/RISC distinction is blurred.

**2. Addressing Modes (How to Find Data):**
*   **Immediate:** Data is embedded in the instruction: MOV R1, #10.
*   **Register:** Data is in a register: ADD R1, R2.
*   **Direct (Absolute):** Address in instruction: LOAD R1, [0x1000].
*   **Register Indirect:** Register holds the address: LOAD R1, [R2].
*   **Base + Offset:** Register + constant: LOAD R1, [R2 + 8] (array/struct access).
*   **Indexed:** Base + scaled index: LOAD R1, [R2 + R3*4] (array with stride).
*   **PC-Relative:** Offset from current instruction. Used for branches and position-independent code (PIC).

**3. Instruction Types:**
*   **Data Transfer:** MOV, LOAD, STORE (move data between registers and memory).
*   **Arithmetic:** ADD, SUB, MUL, DIV, INC, DEC.
*   **Logic:** AND, OR, XOR, NOT, SHL (shift left), SHR (shift right).
*   **Control Flow:** JMP (unconditional), JE/JNE/JG/JL (conditional), CALL, RET.
*   **System:** INT (interrupt), SYSCALL, NOP, HLT.

**4. RISC-V (The Open ISA):**
*   Open-source ISA -- no licensing fees.
*   Modular: base integer set (RV32I/RV64I) plus optional extensions (M=multiply, A=atomic, F/D=float, C=compressed).
*   Growing ecosystem: used in academia, IoT, and increasingly in data centers.`,
					CodeExamples: `// CISC Example (x86-64 - Variable length instructions):
add eax, [ebx + ecx*4 + 12]  // Complex! Scale, add, load, add in one instruction
rep movsb                      // Copy ECX bytes from [RSI] to [RDI] -- one instruction

// RISC Example (ARM - Fixed 4-byte instructions):
ldr r0, [r1, r2, lsl #2]     // Load from r1 + r2*4
add r0, r0, r3                 // Add r3 to r0

// RISC-V Example (Fixed 4-byte instructions):
lw   x5, 0(x10)               // Load word from address in x10
add  x6, x5, x7               // x6 = x5 + x7
sw   x6, 4(x10)               // Store word to address x10+4
beq  x5, x6, label            // Branch if x5 == x6

// Addressing Modes in x86-64:
mov rax, 42                   // Immediate: value 42
mov rax, rbx                  // Register: value of RBX
mov rax, [0x1000]             // Direct: value at address 0x1000
mov rax, [rbx]                // Register Indirect: value at address in RBX
mov rax, [rbx + 8]            // Base + Offset: for struct fields
mov rax, [rbx + rcx*8]        // Indexed: for array access
mov rax, [rip + offset]       // PC-Relative: position-independent`,
				},
				{
					Title: "Data Representation",
					Content: `Computers only understand Binary (0 and 1). Every piece of data -- text, numbers, videos -- is just a sequence of bits. Understanding representation is essential for debugging precision issues, overflow bugs, and cross-platform portability.

**1. Integers and Two's Complement (Review and Depth):**
*   Unsigned: N bits represent 0 to 2^N - 1.
*   Signed (Two's Complement): N bits represent -2^(N-1) to 2^(N-1) - 1.
*   Overflow wraps around: 127 + 1 = -128 in signed 8-bit.
*   Sign extension: When widening, copy the sign bit. -5 in 8-bit (11111011) becomes -5 in 16-bit (1111111111111011).

**2. Fixed-Point Arithmetic:**
*   A fixed number of bits for the integer and fractional parts.
*   Example: 8.8 fixed point = 8 integer bits + 8 fractional bits.
*   Range is limited but computation is fast (integer hardware).
*   Used in DSP, audio processing, and embedded systems.

**3. Floating-Point Performance Implications:**
*   FP add/multiply: 3-5 cycles on modern CPUs.
*   FP divide: 10-20 cycles (much slower).
*   Denormalized numbers can be 10-100x slower (trap to microcode on some CPUs).
*   FP operations are not associative: (a + b) + c may differ from a + (b + c).

**4. BCD (Binary-Coded Decimal):**
*   Each decimal digit stored in 4 bits.
*   0-9 maps to 0000-1001. Values 1010-1111 are invalid.
*   Used in financial systems and calculators where exact decimal representation matters.

**5. Bitwise Operations in Hardware:**
*   AND, OR, XOR, NOT are single-cycle operations.
*   Shift left by N = multiply by 2^N (free in hardware via wiring).
*   Shift right by N = divide by 2^N (with rounding toward negative infinity for arithmetic shift).
*   Rotate: bits shifted out one end come in the other.
*   Bit manipulation tricks: x AND (x-1) clears the lowest set bit, x AND (-x) isolates the lowest set bit.`,
					CodeExamples: `// Two's complement edge cases:
int8_t x = 127;    // 0111 1111
x = x + 1;         // Overflow! Now -128 (1000 0000)

// Sign extension:
int8_t small = -5;       // 1111 1011
int16_t big = small;     // 1111 1111 1111 1011 (still -5)

// Floating-point surprise:
float a = 0.1f + 0.2f;   // NOT exactly 0.3!
// a = 0.300000011920929... due to binary representation

// Bit manipulation tricks:
int n = 52;              // 0011 0100
int lowest = n & (-n);   // 0000 0100 = 4 (isolate lowest set bit)
int cleared = n & (n-1); // 0011 0000 = 48 (clear lowest set bit)
int popcount = __builtin_popcount(n); // 3 (number of set bits)

// Fixed-point example (8.8 format):
uint16_t fp_pi = 0x0324;  // 3.140625 (3 * 256 + 36 = 804)
uint16_t fp_two = 0x0200; // 2.0
uint16_t fp_result = (fp_pi * fp_two) >> 8; // Multiply and shift`,
				},
				{
					Title: "Bus Architecture and Interconnects",
					Content: `A bus is a shared communication pathway that connects CPU, memory, and I/O devices. Understanding bus architecture explains why bandwidth and latency differ across components.

**1. Bus Components:**
*   **Address Bus:** Carries the memory address (determines addressable memory space). 32-bit address bus = 4 GB addressable.
*   **Data Bus:** Carries actual data. Width determines throughput per transfer (64-bit data bus = 8 bytes/transfer).
*   **Control Bus:** Carries control signals (read/write, interrupt, bus request/grant, clock).

**2. Bus Types:**
*   **System Bus (Front-Side Bus / QPI / UPI):** Connects CPU to memory controller. Historical: FSB. Modern Intel: QPI/UPI. Modern AMD: Infinity Fabric.
*   **Memory Bus:** Connects memory controller to DRAM modules. DDR4: 64-bit wide, ~3200 MT/s.
*   **I/O Bus:** Connects peripherals. PCIe (modern), USB (external), SATA (storage).
*   **Internal Bus (On-chip):** Connects components within the CPU. Ring bus (Intel), mesh interconnect, crossbar switch.

**3. Bus Arbitration:**
When multiple devices want the bus simultaneously, an arbiter decides who goes first.
*   **Daisy Chain:** Device closest to arbiter gets priority. Simple, unfair.
*   **Centralized Parallel:** Each device has a request line. Arbiter selects one. Fair but complex wiring.
*   **Distributed:** Devices negotiate among themselves (e.g., CAN bus in cars).

**4. Modern Interconnects (Beyond Traditional Buses):**
*   **PCIe (PCI Express):** Point-to-point serial links. Each lane carries ~2 GB/s (PCIe 4.0). GPUs use x16 lanes = ~32 GB/s.
*   **NVLink:** NVIDIA GPU-to-GPU interconnect. 600 GB/s (NVLink 4.0).
*   **CXL (Compute Express Link):** CPU-to-device coherent interconnect for accelerators and memory expansion. Built on PCIe PHY.
*   **USB (Universal Serial Bus):** USB 2.0: 480 Mbps. USB 3.2: 20 Gbps. USB4: 40 Gbps.

**5. Bandwidth vs Latency:**
*   **Bandwidth:** Amount of data per second (GB/s). Like pipe diameter.
*   **Latency:** Time for first bit to arrive (ns or us). Like pipe length.
*   High bandwidth does not mean low latency. A cargo ship has high bandwidth but high latency; a text message is low bandwidth but low latency.`,
					CodeExamples: `// Bus bandwidth calculations:
// DDR4-3200 DRAM:
//   64-bit bus, 3200 MT/s (megatransfers per second)
//   Bandwidth = 64 bits * 3200 * 10^6 / 8 = 25.6 GB/s
//   Dual-channel: 51.2 GB/s

// PCIe 4.0 x16 (GPU slot):
//   16 lanes * 2 GB/s per lane = ~32 GB/s (each direction)
//   Total bidirectional: ~64 GB/s

// USB comparison:
//   USB 2.0:  480 Mbps = ~60 MB/s
//   USB 3.0:  5 Gbps   = ~625 MB/s
//   USB 3.2:  20 Gbps  = ~2.5 GB/s
//   USB4:     40 Gbps  = ~5 GB/s

// Latency comparison:
//   L1 cache hit:   ~1 ns
//   DRAM access:    ~60 ns     (60x L1)
//   PCIe round-trip: ~1-2 us   (1000x L1)
//   NVMe SSD read:  ~10-25 us
//   HDD seek:       ~5-10 ms   (5,000,000x L1)
//   Network (LAN):  ~0.5 ms
//   Network (WAN):  ~50-200 ms`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
