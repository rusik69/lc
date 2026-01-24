package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          201,
			Title:       "CPU Pipelining and Performance",
			Description: "Learn about instruction pipelining, hazards, performance optimization, and cache hierarchy.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Instruction Pipelining",
					Content: `Pipelining is a technique where the CPU works on multiple instructions at once, like an assembly line. This increases **throughput** (instructions per second) without necessarily speeding up a single instruction.

**1. The 5-Stage Classic Pipeline:**
` + "```" + `
[ IF ] --> [ ID ] --> [ EX ] --> [ MEM ] --> [ WB ]
  │          │          │           │            └─ Write Back
  │          │          │           └─ Memory Access
  │          │          └─ Execute / ALU
  │          └─ Instruction Decode / Register Read
  └─ Instruction Fetch
` + "```" + `

**2. Pipeline Hazards (Roadblocks):**
*   **Structural:** Two instructions need the same hardware at the same time (e.g., both need memory).
*   **Data:** An instruction needs the result of a previous one that isn't finished yet.
    *   *Solution:* **Forwarding** (sending data directly from EX to ID).
*   **Control:** A branch (if/jump) makes it unclear which instruction is next.
    *   *Solution:* **Branch Prediction** (guessing where the code goes).`,
					CodeExamples: `// Data Hazard & Forwarding Example:
add r1, r2, r3  // r1 calculated in EX stage
sub r4, r1, r5  // r4 needs r1 immediately!

// Without forwarding: 'sub' must wait (stall) for 2-3 cycles.
// With forwarding: hardware wires send r1 from 'add' to 'sub' directly.`,
				},
				{
					Title: "Cache Memory & Mapping Strategies",
					Content: `The Cache is a high-speed buffer that hides memory latency. The way memory addresses are mapped to cache lines determines overall performance.

**1. Mapping Strategies:**
` + "```" + `
| Strategy          | Mapping Rule                      | Hardware |
|-------------------|-----------------------------------|----------|
| Direct Mapped     | One-to-One (fixed position)       | Simple   |
| Set Associative   | N-to-One (choice of N slots)      | Balanced |
| Fully Associative | Any-to-Any (any line)             | Complex  |
` + "```" + `

**2. The 3 C's of Cache Misses:**
*   **Compulsory:** Cold start. Data has never been in the cache (Unavoidable).
*   **Capacity:** The cache is too small for the working data set.
*   **Conflict:** Multiple data blocks compete for the same cache slot (Direct Mapped only).

**3. Cache-Friendly Code (Writing for speed):**
*   **Spatial Locality:** Access data sequentially (like an array) rather than jumping around.
*   **Temporal Locality:** Reuse the same variables multiple times in a tight loop.`,
					CodeExamples: `// ❌ Cache-Unfriendly (Random Access)
for(int i=0; i<N; i++) sum += arr[random_index[i]];

// ✅ Cache-Friendly (Sequential)
for(int i=0; i<N; i++) sum += arr[i]; // CPU prefetches next data`,
				},
				{
					Title: "Virtual Memory & Translation (TLB)",
					Content: `Virtual Memory gives programs the illusion of a massive, private memory space, while the hardware maps it to smaller, physical RAM.

**1. Address Translation Path:**
` + "```" + `
[ CPU Virtual Addr ] ──▶ [ TLB (Cache) ] ──┐
                               │           │ (HIT: Fast)
      ┌────────────────────────┘           ▼
      │ (MISS: Slow)               [ Physical Addr ]
      ▼                                    │
[ Page Table (RAM) ] ──────────────────────┘
` + "```" + `

**2. Key Terms:**
*   **TLB (Translation Lookaside Buffer):** A high-speed cache inside the CPU that stores recent virtual-to-physical mappings.
*   **Page Table:** The "Map" stored in RAM. If the TLB misses, the CPU must look here.
*   **Page Fault:** The requested data isn't in RAM at all; the OS must fetch it from Disk (SSD/HDD). This is a major performance killer.

**3. Memory Protection:**
Virtual memory prevents "Process A" from reading or writing to the memory of "Process B".`,
					CodeExamples: `// Virtual Memory logic (simplified):
if (mapping_in_tlb(vaddr)) {
    return get_phys_addr(vaddr); // 1-3 cycles
} else {
    walk_page_table(vaddr);      // 100+ cycles (Memory Wall)
}`,
				},
				{
					Title: "I/O Systems and Interrupts",
					Content: `Input/Output systems allow the CPU to communicate with external devices like keyboards, disks, and network interfaces.

**I/O Methods:**

**1. Programmed I/O (Polling):**
- CPU continuously checks device status
- Simple but wastes CPU cycles
- Suitable for fast devices

**2. Interrupt-Driven I/O:**
- Device signals CPU when ready
- CPU handles interrupt, then continues
- More efficient than polling
- Common for most devices

**3. Direct Memory Access (DMA):**
- Device transfers data directly to/from memory
- CPU only involved at start and end
- Very efficient for large transfers
- Used for disk, network, graphics

**Interrupts:**

**Interrupt Types:**

**1. Hardware Interrupts:**
- External devices (keyboard, timer, network)
- Asynchronous (can occur at any time)
- Handled by interrupt controller

**2. Software Interrupts (Traps):**
- Generated by instructions (system calls, exceptions)
- Synchronous (occurs at specific instruction)
- Examples: INT instruction, page fault, division by zero

**Interrupt Handling:**

1. **Save Context**: Push registers, flags onto stack
2. **Identify Interrupt**: Read interrupt vector
3. **Jump to Handler**: Execute interrupt service routine (ISR)
4. **Process Interrupt**: Handle the event
5. **Restore Context**: Pop registers, flags
6. **Return**: Resume interrupted program

**Interrupt Priority:**
- Higher priority interrupts can preempt lower priority ones
- Critical interrupts (e.g., power failure) have highest priority
- Interrupts can be masked/disabled

**DMA Operation:**

**DMA Transfer Steps:**
1. CPU sets up DMA controller (source, destination, size)
2. DMA controller requests bus
3. DMA transfers data directly
4. DMA signals completion via interrupt
5. CPU processes completion`,
					CodeExamples: `// Example: Polling I/O
// Check if keyboard has data
while (!(keyboard_status & DATA_READY)) {
    // Wait...
}
char key = keyboard_data;

// Example: Interrupt-driven I/O
// Set up interrupt handler
void keyboard_interrupt_handler() {
    char key = keyboard_data;
    process_key(key);
}

// Enable keyboard interrupt
keyboard_enable_interrupt();

// CPU continues other work
// When key pressed, interrupt fires, handler called

// Example: DMA setup
// Configure DMA for disk read
dma_source = disk_buffer_address;
dma_destination = memory_address;
dma_size = 4096;  // 4 KB
dma_start();

// CPU continues execution
// DMA transfers data in background
// Interrupt when complete

// Example: Interrupt vector table
// x86 interrupt vectors:
// 0x00: Divide by zero
// 0x01: Debug
// 0x02: Non-maskable interrupt
// 0x08: Timer interrupt
// 0x0E: Page fault
// 0x21: Keyboard interrupt
// 0x80: System call (Linux)

// Example: Interrupt handler (simplified)
void timer_interrupt_handler() {
    // Save context
    push_all_registers();
    
    // Handle interrupt
    update_system_time();
    schedule_next_timer();
    
    // Send EOI (End of Interrupt)
    send_eoi();
    
    // Restore context
    pop_all_registers();
    iret();  // Return from interrupt`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
