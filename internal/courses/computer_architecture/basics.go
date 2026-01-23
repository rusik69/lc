package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          200,
			Title:       "Introduction to Computer Architecture",
			Description: "Learn the fundamentals of computer architecture: CPU design, instruction sets, memory organization, and basic hardware components.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Computer Architecture?",
					Content: `Computer architecture is the design and organization of computer systems, including the structure and behavior of hardware components and how they interact.

**Key Concepts:**

**Architecture vs. Organization:**
- **Architecture**: The attributes visible to the programmer (instruction set, data types, addressing modes)
- **Organization**: The operational units and their interconnections (hardware implementation details)

**Levels of Abstraction:**
- **Application Level**: High-level programs
- **System Software Level**: Operating systems, compilers
- **Instruction Set Architecture (ISA)**: Machine instructions
- **Microarchitecture**: CPU implementation details
- **Logic Level**: Gates and circuits
- **Device Level**: Transistors and physical components

**Why Study Computer Architecture?**
- Understand how computers work at a fundamental level
- Write more efficient code
- Design better systems
- Debug hardware-related issues
- Optimize performance

**Historical Context:**
- **1940s**: First electronic computers (ENIAC)
- **1950s**: Stored-program concept (von Neumann architecture)
- **1960s**: Integrated circuits, minicomputers
- **1970s**: Microprocessors (Intel 4004, 8080)
- **1980s**: RISC architecture, personal computers
- **1990s**: Superscalar processors, pipelining
- **2000s**: Multi-core processors
- **2010s**: GPUs, specialized accelerators
- **2020s**: AI accelerators, quantum computing`,
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
// 5. Store result`,
				},
				{
					Title: "CPU Architecture Basics",
					Content: `The Central Processing Unit (CPU) is the brain of the computer, executing instructions and performing calculations.

**CPU Components:**

**1. Control Unit (CU):**
- Fetches instructions from memory
- Decodes instructions
- Coordinates execution
- Manages data flow

**2. Arithmetic Logic Unit (ALU):**
- Performs arithmetic operations (add, subtract, multiply, divide)
- Performs logical operations (AND, OR, XOR, NOT)
- Compares values
- Shifts and rotates bits

**3. Registers:**
- **General Purpose Registers**: Store data and addresses
- **Program Counter (PC)**: Points to next instruction
- **Instruction Register (IR)**: Holds current instruction
- **Status Register**: Contains flags (zero, carry, overflow, sign)
- **Stack Pointer (SP)**: Points to top of stack
- **Base Pointer (BP)**: Points to base of current frame

**4. Cache:**
- Fast memory close to CPU
- Stores frequently accessed data
- Reduces memory access time

**CPU Operation Cycle (Fetch-Decode-Execute):**

1. **Fetch**: Get instruction from memory at address in PC
2. **Decode**: Determine what operation to perform
3. **Execute**: Perform the operation
4. **Store**: Save results to registers or memory
5. **Update PC**: Move to next instruction

**Instruction Types:**
- **Data Transfer**: MOV, LOAD, STORE
- **Arithmetic**: ADD, SUB, MUL, DIV
- **Logical**: AND, OR, XOR, NOT
- **Control Flow**: JMP, CALL, RET, BRANCH
- **Comparison**: CMP, TEST`,
					CodeExamples: `// Example: CPU register usage
// x86-64 registers:
// General purpose: RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP
// Instruction pointer: RIP
// Flags: RFLAGS

// Example: Adding two numbers
mov rax, 10      // Load 10 into RAX
mov rbx, 20      // Load 20 into RBX
add rax, rbx     // Add RBX to RAX (result: 30)

// Example: Conditional jump based on flags
cmp rax, 0       // Compare RAX with 0
je zero_label    // Jump if equal (zero flag set)
jne non_zero_label // Jump if not equal

// Example: Function call
call my_function // Push return address, jump to function
ret              // Pop return address, jump back

// Example: Stack operations
push rax         // Push RAX onto stack (decrement RSP)
pop rbx          // Pop from stack into RBX (increment RSP)`,
				},
				{
					Title: "Memory Organization",
					Content: `Memory is organized hierarchically to balance speed, capacity, and cost.

**Memory Hierarchy (Fastest to Slowest):**

1. **CPU Registers**: Fastest, smallest (kilobytes), most expensive
2. **L1 Cache**: Very fast, small (tens of KB), expensive
3. **L2 Cache**: Fast, medium (hundreds of KB to MB), moderate cost
4. **L3 Cache**: Moderate speed, larger (MB), moderate cost
5. **Main Memory (RAM)**: Slower, large (GB), cheaper
6. **Secondary Storage**: Slowest, largest (TB), cheapest

**Memory Characteristics:**

**Volatile vs. Non-Volatile:**
- **Volatile**: Loses data when power is off (RAM, cache)
- **Non-Volatile**: Retains data when power is off (ROM, disk, flash)

**Random Access vs. Sequential:**
- **Random Access**: Any location accessible in same time (RAM)
- **Sequential**: Must access in order (tape)

**Memory Addressing:**

**Byte Addressing:**
- Each byte has unique address
- Addresses are sequential integers
- Common in modern systems

**Word Addressing:**
- Each word (multiple bytes) has unique address
- Used in some older systems
- More efficient for word-sized operations

**Memory Layout:**
- **Code Segment**: Program instructions
- **Data Segment**: Global/static variables
- **Stack**: Function calls, local variables
- **Heap**: Dynamically allocated memory

**Endianness:**
- **Big-Endian**: Most significant byte at lowest address
- **Little-Endian**: Least significant byte at lowest address
- x86 uses little-endian
- Network byte order is big-endian`,
					CodeExamples: `// Example: Memory layout
// Stack grows downward (high to low addresses)
// Heap grows upward (low to high addresses)

// Stack frame layout (x86-64):
// [High Address]
//   Previous frame pointer
//   Return address
//   Local variables
//   Function arguments
// [Low Address]

// Example: Memory access patterns
int array[100];
for (int i = 0; i < 100; i++) {
    array[i] = i;  // Sequential access (cache-friendly)
}

// Example: Endianness
uint32_t value = 0x12345678;
// Little-endian memory layout:
// Address: 0x1000: 0x78
// Address: 0x1001: 0x56
// Address: 0x1002: 0x34
// Address: 0x1003: 0x12

// Big-endian memory layout:
// Address: 0x1000: 0x12
// Address: 0x1001: 0x34
// Address: 0x1002: 0x56
// Address: 0x1003: 0x78`,
				},
				{
					Title: "Instruction Set Architecture (ISA)",
					Content: `The Instruction Set Architecture defines the interface between software and hardware, specifying what instructions the processor can execute.

**ISA Types:**

**1. CISC (Complex Instruction Set Computer):**
- Many complex instructions
- Variable-length instructions
- Instructions can perform multiple operations
- Examples: x86, x86-64
- Advantages: Compact code, powerful instructions
- Disadvantages: Complex hardware, harder to optimize

**2. RISC (Reduced Instruction Set Computer):**
- Fewer, simpler instructions
- Fixed-length instructions
- Load-store architecture (only LOAD/STORE access memory)
- Examples: ARM, MIPS, RISC-V
- Advantages: Simple hardware, easier to pipeline, faster execution
- Disadvantages: More instructions needed, larger code size

**3. VLIW (Very Long Instruction Word):**
- Multiple operations per instruction
- Compiler schedules operations
- Examples: Itanium, some DSPs
- Advantages: Explicit parallelism
- Disadvantages: Complex compiler, less flexible

**Instruction Format:**

**Typical RISC Instruction:**
- **Opcode**: Operation to perform (6 bits)
- **Register Source 1**: First source register (5 bits)
- **Register Source 2**: Second source register (5 bits)
- **Register Destination**: Destination register (5 bits)
- **Function Code**: Additional operation info (11 bits)
- Total: 32 bits

**Addressing Modes:**

1. **Immediate**: Operand is in instruction
   - Example: ADD R1, R2, #5 (add 5 to R2)

2. **Register**: Operand is in register
   - Example: ADD R1, R2, R3 (add R2 and R3)

3. **Direct**: Address is in instruction
   - Example: LOAD R1, [0x1000] (load from address 0x1000)

4. **Indirect**: Register contains address
   - Example: LOAD R1, [R2] (load from address in R2)

5. **Indexed**: Address = base + offset
   - Example: LOAD R1, [R2 + #4] (load from R2 + 4)

6. **Relative**: Address = PC + offset
   - Example: JMP #100 (jump to PC + 100)`,
					CodeExamples: `// Example: RISC instruction format
// ADD R1, R2, R3
// Opcode: ADD (000000)
// R1: destination (00001)
// R2: source 1 (00010)
// R3: source 2 (00011)
// Function: ADD (100000)
// Binary: 000000 00001 00010 00011 00000 100000

// Example: CISC instruction (x86)
// ADD EAX, [EBX + ECX*4 + 10]
// Single instruction does:
// 1. Calculate address: EBX + ECX*4 + 10
// 2. Load value from memory
// 3. Add to EAX
// 4. Store result in EAX

// Example: Addressing modes
// Immediate
MOV R1, #42        // R1 = 42

// Register
ADD R1, R2, R3     // R1 = R2 + R3

// Direct
LOAD R1, [0x1000]  // R1 = memory[0x1000]

// Indirect
LOAD R1, [R2]      // R1 = memory[R2]

// Indexed
LOAD R1, [R2 + #4] // R1 = memory[R2 + 4]

// Relative
JMP #100           // PC = PC + 100`,
				},
				{
					Title: "Data Representation",
					Content: `Computers represent all data as binary digits (bits). Understanding how different data types are represented is crucial.

**Number Systems:**

**Binary (Base 2):**
- Uses digits 0 and 1
- Each position represents power of 2
- Example: 1011₂ = 1×2³ + 0×2² + 1×2¹ + 1×2⁰ = 11₁₀

**Hexadecimal (Base 16):**
- Uses digits 0-9 and A-F
- Each position represents power of 16
- Convenient for representing binary (4 bits per hex digit)
- Example: 0x2F = 2×16¹ + 15×16⁰ = 47₁₀

**Integer Representation:**

**Unsigned Integers:**
- All bits represent magnitude
- Range: 0 to 2ⁿ - 1 for n bits
- Example: 8-bit unsigned: 0 to 255

**Signed Integers (Two's Complement):**
- Most significant bit is sign bit
- Negative numbers: invert bits and add 1
- Range: -2ⁿ⁻¹ to 2ⁿ⁻¹ - 1 for n bits
- Example: 8-bit signed: -128 to 127
- Advantage: Single representation for zero, easy arithmetic

**Floating Point (IEEE 754):**
- **Single Precision (32-bit)**: 1 sign bit, 8 exponent bits, 23 mantissa bits
- **Double Precision (64-bit)**: 1 sign bit, 11 exponent bits, 52 mantissa bits
- Format: (-1)ˢ × (1 + mantissa) × 2^(exponent - bias)
- Special values: ±0, ±∞, NaN (Not a Number)

**Character Representation:**

**ASCII:**
- 7-bit encoding (128 characters)
- Extended ASCII: 8-bit (256 characters)
- Example: 'A' = 65₁₀ = 0x41

**Unicode:**
- Universal character encoding
- UTF-8: Variable-length (1-4 bytes)
- UTF-16: Variable-length (2 or 4 bytes)
- UTF-32: Fixed-length (4 bytes)`,
					CodeExamples: `// Example: Binary to decimal
// 1011₂ = 1×8 + 0×4 + 1×2 + 1×1 = 11₁₀

// Example: Hexadecimal
// 0x2F = 2×16 + 15 = 47₁₀
// 0xFF = 255₁₀

// Example: Two's complement
// Positive: 5₁₀ = 00000101₂
// Negative: -5₁₀
//   1. Invert: 11111010₂
//   2. Add 1:  11111011₂

// Example: Floating point (simplified)
// 3.14159 ≈ 0x40490FDB (single precision)
// Sign: 0 (positive)
// Exponent: 10000000₂ = 128₁₀, bias = 127, so 2¹
// Mantissa: 1.10010010000111111011011₂
// Result: 1.5708... × 2¹ ≈ 3.14159

// Example: Character encoding
// ASCII: 'A' = 65₁₀ = 0x41
// ASCII: 'a' = 97₁₀ = 0x61
// UTF-8: '€' = 0xE2 0x82 0xAC (3 bytes)

// Example: Bitwise operations
uint8_t a = 0b10101010;  // 170
uint8_t b = 0b11110000;  // 240

a & b;  // AND:  0b10100000 (160)
a | b;  // OR:   0b11111010 (250)
a ^ b;  // XOR:  0b01011010 (90)
~a;     // NOT:  0b01010101 (85)
a << 2; // Left shift:  0b1010101000 (680, but truncated to 8 bits)
a >> 2; // Right shift: 0b00101010 (42)`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
