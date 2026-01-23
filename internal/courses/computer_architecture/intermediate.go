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
					Content: `Pipelining is a technique that allows multiple instructions to be processed simultaneously, improving CPU throughput.

**Basic Pipeline Stages:**

**5-Stage Pipeline (Classic RISC):**
1. **IF (Instruction Fetch)**: Get instruction from memory
2. **ID (Instruction Decode)**: Decode instruction, read registers
3. **EX (Execute)**: Perform ALU operation
4. **MEM (Memory Access)**: Access data memory if needed
5. **WB (Write Back)**: Write result to register

**Pipeline Benefits:**
- **Throughput**: More instructions completed per cycle
- **Efficiency**: Better hardware utilization
- **Speedup**: Up to N× for N-stage pipeline (in ideal case)

**Pipeline Example:**
Without pipelining: 5 cycles per instruction
- Instruction 1: IF → ID → EX → MEM → WB (5 cycles)
- Instruction 2: IF → ID → EX → MEM → WB (5 cycles)
- Total: 10 cycles for 2 instructions

With pipelining: 1 cycle per instruction (after initial latency)
- Cycle 1: I1-IF
- Cycle 2: I1-ID, I2-IF
- Cycle 3: I1-EX, I2-ID, I3-IF
- Cycle 4: I1-MEM, I2-EX, I3-ID, I4-IF
- Cycle 5: I1-WB, I2-MEM, I3-EX, I4-ID, I5-IF
- After 5 cycles: 1 instruction per cycle

**Pipeline Hazards:**

**1. Structural Hazards:**
- Resource conflict (e.g., same memory port)
- Solution: Duplicate resources or stall

**2. Data Hazards:**
- **RAW (Read After Write)**: Instruction needs data not yet written
- **WAR (Write After Read)**: Write before previous read completes
- **WAW (Write After Write)**: Multiple writes to same register
- Solutions: Forwarding, stalling, register renaming

**3. Control Hazards:**
- Branch instructions change PC
- Solution: Branch prediction, delayed branches, predication`,
					CodeExamples: `// Example: Pipeline execution
// Instructions:
// I1: ADD R1, R2, R3
// I2: SUB R4, R1, R5  (depends on R1 from I1)
// I3: MUL R6, R4, R7  (depends on R4 from I2)

// Without forwarding (stalls):
// Cycle 1: I1-IF
// Cycle 2: I1-ID
// Cycle 3: I1-EX
// Cycle 4: I1-MEM
// Cycle 5: I1-WB, I2-IF (stall)
// Cycle 6: I2-ID (stall - waiting for R1)
// Cycle 7: I2-EX (R1 now available)
// ...

// With forwarding:
// Cycle 1: I1-IF
// Cycle 2: I1-ID
// Cycle 3: I1-EX (R1 computed)
// Cycle 4: I1-MEM, I2-EX (forward R1 to I2)
// Cycle 5: I1-WB, I2-MEM, I3-EX (forward R4 to I3)
// ...

// Example: Data hazard detection
// RAW hazard:
ADD R1, R2, R3    // Writes R1 in WB stage
SUB R4, R1, R5    // Reads R1 in ID stage
// Solution: Forward R1 from EX/MEM to ID stage

// Example: Branch prediction
if (condition) {
    // Branch taken
} else {
    // Branch not taken
}
// Predictor: 2-bit saturating counter
// 00: Strongly not taken
// 01: Weakly not taken
// 10: Weakly taken
// 11: Strongly taken`,
				},
				{
					Title: "Cache Memory",
					Content: `Cache memory is a small, fast memory that stores frequently accessed data to reduce average memory access time.

**Cache Organization:**

**Cache Levels:**
- **L1 Cache**: Smallest, fastest, closest to CPU (separate instruction and data caches)
- **L2 Cache**: Larger, slower, unified (instructions and data)
- **L3 Cache**: Largest on-chip cache, shared among cores

**Cache Structure:**

**Cache Block/Line:**
- Unit of data transfer between cache and memory
- Typically 32-128 bytes
- Contains: valid bit, tag, data

**Cache Mapping:**

**1. Direct Mapped:**
- Each memory block maps to exactly one cache line
- Simple, fast, but can cause conflicts
- Address: [Tag | Index | Offset]

**2. Fully Associative:**
- Any memory block can go to any cache line
- No conflicts, but expensive (needs content-addressable memory)
- Address: [Tag | Offset]

**3. Set Associative:**
- Compromise: cache divided into sets, each set has N ways
- N-way set associative: N blocks per set
- Address: [Tag | Set Index | Offset]

**Cache Performance:**

**Hit Rate**: Percentage of accesses found in cache
**Miss Rate**: 1 - Hit Rate
**Average Access Time**: Hit Time + Miss Rate × Miss Penalty

**Cache Miss Types:**

**1. Compulsory (Cold Start):**
- First access to a block
- Unavoidable

**2. Capacity:**
- Cache too small to hold all needed data
- Solution: Larger cache

**3. Conflict:**
- Multiple blocks map to same cache line
- Solution: Higher associativity

**4. Coherence (Multi-core):**
- Cache invalidation due to other cores
- Solution: Cache coherence protocols`,
					CodeExamples: `// Example: Direct-mapped cache
// Cache size: 4 KB, block size: 64 bytes
// Number of blocks: 4096 / 64 = 64 blocks
// Index bits: log₂(64) = 6 bits
// Offset bits: log₂(64) = 6 bits
// Tag bits: 32 - 6 - 6 = 20 bits

// Address: 0x12345678
// Tag:     0x12345
// Index:   0x67
// Offset:  0x38

// Example: Cache hit/miss
// Access pattern: A, B, C, A, B, C
// Direct-mapped cache (2 blocks):
// Access A: Miss, load A into block 0
// Access B: Miss, load B into block 1
// Access C: Miss, load C into block 0 (evicts A)
// Access A: Miss, load A into block 0 (evicts C)
// Access B: Hit (still in block 1)
// Access C: Miss, load C into block 0 (evicts A)

// Example: 2-way set associative
// Cache: 8 KB, block size: 64 bytes, 2-way
// Sets: (8192 / 64) / 2 = 64 sets
// Index bits: log₂(64) = 6 bits
// Each set can hold 2 blocks

// Example: Cache-friendly code
// Good: Sequential access
for (int i = 0; i < N; i++) {
    sum += array[i];  // Sequential, cache-friendly
}

// Bad: Random access
for (int i = 0; i < N; i++) {
    sum += array[random[i]];  // Random, cache-unfriendly
}

// Example: Block size impact
// Small blocks (16 bytes): More blocks, more tags, less waste
// Large blocks (128 bytes): Fewer blocks, fewer tags, more waste on small data`,
				},
				{
					Title: "Memory Hierarchy and Virtual Memory",
					Content: `The memory hierarchy provides different levels of storage with varying speed, capacity, and cost characteristics.

**Memory Hierarchy Levels:**

1. **Registers**: ~1 cycle, ~1 KB, $$$
2. **L1 Cache**: ~1-3 cycles, ~32-64 KB, $$$
3. **L2 Cache**: ~10 cycles, ~256 KB - 1 MB, $$
4. **L3 Cache**: ~30-40 cycles, ~8-32 MB, $$
5. **Main Memory (RAM)**: ~100-300 cycles, ~8-64 GB, $
6. **SSD**: ~100,000 cycles, ~500 GB - 4 TB, $
7. **HDD**: ~10,000,000 cycles, ~1-20 TB, ¢

**Locality Principles:**

**Temporal Locality:**
- Recently accessed items likely to be accessed again
- Exploited by: Caching

**Spatial Locality:**
- Items near recently accessed items likely to be accessed
- Exploited by: Larger cache blocks, prefetching

**Virtual Memory:**

**Purpose:**
- Provide illusion of larger memory than physically available
- Memory protection and isolation
- Simplified memory management

**Paging:**
- Memory divided into fixed-size pages (typically 4 KB)
- Virtual address: [Virtual Page Number | Offset]
- Physical address: [Physical Page Number | Offset]
- Page table maps virtual pages to physical pages

**Page Table Structure:**
- **Page Table Entry (PTE)**: Contains physical page number, valid bit, protection bits
- **Translation Lookaside Buffer (TLB)**: Cache for page table entries
- **Multi-level Page Tables**: Hierarchical structure to reduce memory

**Page Fault:**
- Access to page not in physical memory
- OS loads page from disk
- May evict another page (page replacement algorithm)`,
					CodeExamples: `// Example: Memory access time calculation
// L1 hit: 1 cycle (90% of accesses)
// L2 hit: 10 cycles (8% of accesses)
// L3 hit: 40 cycles (1.5% of accesses)
// Memory: 200 cycles (0.5% of accesses)

// Average access time:
// 0.9 × 1 + 0.08 × 10 + 0.015 × 40 + 0.005 × 200
// = 0.9 + 0.8 + 0.6 + 1.0
// = 3.3 cycles

// Example: Virtual address translation
// Virtual address: 0x12345678
// Page size: 4 KB = 0x1000
// VPN: 0x12345
// Offset: 0x678

// Page table lookup:
// PTE[0x12345] = {physical_page: 0xABCD, valid: 1}
// Physical address: 0xABCD678

// Example: TLB (Translation Lookaside Buffer)
// TLB hit: 1 cycle (most common)
// TLB miss: Need page table walk (100+ cycles)
// TLB hit rate: ~99%

// Example: Page replacement algorithms
// LRU (Least Recently Used):
// Evict page not used for longest time
// Requires tracking access times

// FIFO (First In First Out):
// Evict oldest page
// Simple but may evict frequently used pages

// Example: Memory-mapped I/O
// I/O devices mapped to memory addresses
// Reading/writing memory addresses communicates with devices
uint32_t *gpio = (uint32_t*)0x20000000;
gpio[0] = 0x01;  // Write to GPIO register`,
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
