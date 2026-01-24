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
					Content: `Computer Architecture is the blueprint for how hardware and software work together. It’s the "contract" that ensures your code actually runs on the physical transistors.

**Why does it matter?**
*   **Performance:** Write code that leverages the hardware.
*   **Efficiency:** Understand why certain algorithms are slow (e.g., cache misses).
*   **Specialization:** Why do we need GPUs for AI but CPUs for general tasks?

**Historical Evolution:**
1.  **1945:** Von Neumann Architecture (The stored-program concept).
2.  **1970s:** First single-chip microprocessors (Intel 4004).
3.  **1990s:** Pipelining and Superscalar CPUs (Doing multiple things at once).
4.  **2010s:** The "End of Moore's Law" shift to Multi-core and Accelerators (GPUs, TPUs).`,
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
					Title: "CPU Architecture & The von Neumann Model",
					Content: `The CPU is the "brain" of the computer. Most modern computers follow the **von Neumann Architecture**, where both data and instructions are stored in the same memory.

**1. The von Neumann Diagram:**
` + "```" + `
   [ Control Unit ] <───> [ Arithmetic Logic Unit ]
          │                          │
          └───────────┬──────────────┘
                      ▼
               [ Registers / Cache ]
                      ▲
          ┌───────────┴───────────┐
          ▼                       ▼
    [ Main Memory ] <───> [ Input / Output ]
` + "```" + `

**2. Key CPU Components:**
*   **Control Unit (CU):** The conductor. It fetches and decodes instructions.
*   **ALU:** The calculator. It performs math (ADD, SUB) and logic (AND, XOR).
*   **Registers:** Tiny, ultra-fast storage inside the CPU (e.g., RAX, RIP).
*   **Program Counter (PC):** A special register that holds the address of the *next* instruction.

**3. The Fetch-Decode-Execute Cycle:**
1.  **Fetch:** Load the next instruction from memory (using the PC).
2.  **Decode:** Translate binary code into signals for the ALU/CU.
3.  **Execute:** Perform the math or move data.
4.  **Writeback:** Save the result back to memory or a register.`,
					CodeExamples: `// x86-64 Register Example:
mov rax, 10      // Store 10 in register RAX
add rax, 20      // Add 20 to RAX (RAX is now 30)

// Controlling flow:
cmp rax, 30      // Compare RAX to 30
je  is_thirty    // Jump to 'is_thirty' if equal`,
				},
				{
					Title: "Memory Hierarchy & Organization",
					Content: `Memory is a trade-off between **Speed** and **Cost**. Modern systems use a hierarchy to give you the illusion of a massive, super-fast memory.

**1. The Memory Pyramid (Fastest to Slowest):**
` + "```" + `
      /  CPU Registers  \      <--  Fastest ($$$)
     /      L1 Cache      \
    /       L2 Cache       \
   /        L3 Cache        \
  /       Main Memory (RAM)  \
 /   Secondary Storage (SSD/HDD) \  <--  Slowest ($)
` + "```" + `

**2. Key Characteristics:**
*   **Locality of Reference:** Why hierarchies work.
    *   *Temporal Locality:* If you use data once, you'll likely use it again soon.
    *   *Spatial Locality:* If you use data, you'll likely use the data next to it (e.g., in an array).
*   **Volatile Memory:** Loses data without power (Registers, Cache, RAM).
*   **Non-Volatile Memory:** Persistent storage (SSD, HDD).

**3. Memory Layout of a Program:**
*   **Stack:** Function parameters and local variables.
*   **Heap:** Dynamic memory (managed by 'malloc' or the Go runtime).
*   **Code/Data:** The compiled binary and static variables.`,
					CodeExamples: `// C-style memory example:
int* ptr = malloc(sizeof(int)); // Memory from Heap
int value = 10;                 // Memory from Stack

// Why sequential access is faster:
int arr[1000];
for (int i=0; i<1000; i++) sum += arr[i]; // Spatial Locality!`,
				},
				{
					Title: "Instruction Set Architecture (ISA)",
					Content: `The ISA is the "language" the CPU speaks. It defines the available instructions, registers, and memory addressing modes.

**1. CISC vs. RISC:**
*   **CISC (Complex Instruction Set Computer):**
    *   *Goal:* Do more in a single instruction (minimize code size).
    *   *Example:* x86 (Intel/AMD).
    *   *Pros:* Versatile, legacy support.
*   **RISC (Reduced Instruction Set Computer):**
    *   *Goal:* Simple, fast instructions that take exactly one cycle.
    *   *Example:* ARM (Apple M1, Smartphones), RISC-V.
    *   *Pros:* Energy efficient, easier to pipeline.

**2. Addressing Modes (How to find data):**
*   **Immediate:** Data is in the instruction ('MOV R1, #10').
*   **Direct:** Address is in the instruction ('LOAD R1, [0x1000]').
*   **Indirect:** Register points to the address ('LOAD R1, [R2]').
*   **Register:** Data is in a register ('ADD R1, R2').

**Modern Trend:** Most modern CISC processors (like Intel Core) actually translate complex instructions into "u-ops" (micro-operations) that look like RISC under the hood!`,
					CodeExamples: `// CISC Example (x86 - Variable length)
add eax, [ebx + ecx*4 + 12] // Complex! Multiplies, adds, loads, and adds.

// RISC Example (ARM - Fixed length)
lsr r2, r1, #2              // Logical shift right
ldr r0, [r1, r2]            // Load from memory
add r0, r0, r3              // Add registers`,
				},
				{
					Title: "Data Representation",
					Content: `Computers only understand **Binary** (0 and 1). Every piece of data—text, numbers, videos—is just a sequence of bits.

**1. Integers & Two's Complement:**
To represent negative numbers, we use Two's Complement.
*   **Positive (5):** '00000101'
*   **Step 1 (Invert):** '11111010'
*   **Step 2 (Add 1):** '11111011' (This is -5).
*   *Benefit:* Hardware can use the same circuit for both addition and subtraction.

**2. Floating Point (IEEE 754):**
Used for decimals. It works like scientific notation: '1.23 x 10^4'.
*   **Sign:** 1 bit (0 for +, 1 for -).
*   **Exponent:** Where the decimal point is.
*   **Mantissa:** The actual digits.

**3. Character Encoding:**
*   **ASCII:** 7-bit (128 characters). Basic English.
*   **Unicode (UTF-8):** Variable length. Supports every language and Emoji! 🚀`,
					CodeExamples: `// Bitwise Operations:
byte a = 0b1010; // 10
byte b = 0b1100; // 12

byte res = a & b; // AND: 0b1000 (8)
byte res2 = a | b; // OR: 0b1110 (14)

// Character vs Code Point:
// 'A' is 65 in decimal, 0x41 in Hex.`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
