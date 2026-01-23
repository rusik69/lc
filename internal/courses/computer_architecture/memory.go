package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          207,
			Title:       "Advanced Memory Systems",
			Description: "Explore advanced memory technologies: DDR architecture, NVMe SSDs, persistent memory, memory controllers, and error correction codes.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "DDR Memory Architecture",
					Content: `DDR (Double Data Rate) SDRAM is the standard for main memory in modern computers. Understanding DDR architecture is essential for system design.

**DDR Evolution:**

**DDR Generations:**
- **DDR**: 200-400 MT/s, 2.5V
- **DDR2**: 400-1066 MT/s, 1.8V
- **DDR3**: 800-2133 MT/s, 1.5V
- **DDR4**: 1600-3200 MT/s, 1.2V
- **DDR5**: 3200-6400 MT/s, 1.1V

**Key Improvements:**
- Higher data rates
- Lower voltage (power efficiency)
- Higher capacity per chip
- Better signal integrity

**DDR Architecture:**

**Double Data Rate:**
- Data transferred on both rising and falling clock edges
- Effective data rate = 2 × clock frequency
- Example: 1600 MHz clock = 3200 MT/s

**Memory Organization:**
- **Rank**: Set of chips sharing command/address bus
- **Bank**: Independent memory array within chip
- **Row**: Horizontal line in memory array
- **Column**: Vertical line in memory array

**Memory Controller:**
- Manages memory access
- Handles timing constraints
- Optimizes access patterns
- Error correction

**Timing Parameters:**

**CAS Latency (CL):**
- Column Address Strobe latency
- Time from column address to data available
- Lower is better (faster)

**RAS to CAS Delay (tRCD):**
- Row Address Strobe to CAS delay
- Time to activate row before column access

**Row Precharge Time (tRP):**
- Time to close row before opening new row

**Activate to Precharge (tRAS):**
- Minimum time row must be open`,
					CodeExamples: `// Example: DDR memory access timing
// DDR4-3200: 1600 MHz clock, CL=22, tRCD=22, tRP=22

// Sequential access (same row):
// Cycle 0: Activate row (tRCD = 22 cycles)
// Cycle 22: Read column 0 (CL = 22 cycles)
// Cycle 44: Data available
// Cycle 44: Read column 1 (CL = 22 cycles, but pipelined)
// Cycle 66: Data available
// Efficient: Row already open

// Random access (different rows):
// Cycle 0: Activate row A (tRCD = 22)
// Cycle 22: Read column (CL = 22)
// Cycle 44: Data available
// Cycle 44: Precharge row A (tRP = 22)
// Cycle 66: Activate row B (tRCD = 22)
// Cycle 88: Read column (CL = 22)
// Cycle 110: Data available
// Inefficient: Row switches required

// Example: Memory interleaving
// Distribute data across multiple channels
// Channel 0: Addresses 0x0000, 0x0008, 0x0010, ...
// Channel 1: Addresses 0x0004, 0x000C, 0x0014, ...
// Improves bandwidth by parallel access

// Example: Bank interleaving
// Distribute across banks within same rank
// Bank 0: Row 0, Row 4, Row 8, ...
// Bank 1: Row 1, Row 5, Row 9, ...
// Allows pipelining of row operations

// Example: Memory controller optimization
void memory_controller_optimize(access_t *accesses, int n) {
    // Group accesses by row
    sort_by_row(accesses, n);
    
    // Schedule accesses to minimize row switches
    for (int i = 0; i < n; i++) {
        if (same_row(accesses[i], current_row)) {
            // Same row: fast access
            schedule_access(accesses[i]);
        } else {
            // Different row: need precharge + activate
            precharge_row(current_row);
            activate_row(accesses[i].row);
            schedule_access(accesses[i]);
        }
    }
}`,
				},
				{
					Title: "NVMe and PCIe SSDs",
					Content: `NVMe (Non-Volatile Memory Express) is the interface protocol for modern SSDs, designed to leverage PCIe's high bandwidth.

**NVMe Architecture:**

**PCIe Interface:**
- Direct connection to CPU via PCIe
- No SATA bottleneck
- Multiple lanes (x1, x2, x4)
- Gen 3: 8 GT/s per lane, Gen 4: 16 GT/s, Gen 5: 32 GT/s

**NVMe Protocol:**
- Command queue-based
- Up to 65,535 queues
- Each queue up to 64K commands
- Low latency (microseconds)

**NVMe vs SATA:**

**SATA:**
- Legacy interface
- Single queue (32 commands)
- Higher latency
- Limited bandwidth (600 MB/s)

**NVMe:**
- Modern interface
- Multiple queues
- Lower latency
- High bandwidth (GB/s)

**SSD Architecture:**

**NAND Flash:**
- **SLC**: 1 bit per cell, fastest, most durable
- **MLC**: 2 bits per cell, balanced
- **TLC**: 3 bits per cell, higher capacity, lower endurance
- **QLC**: 4 bits per cell, highest capacity, lowest endurance

**Flash Controller:**
- Wear leveling
- Bad block management
- Garbage collection
- Error correction

**Performance Characteristics:**

**Sequential Read/Write:**
- High bandwidth (GB/s)
- Good for large files

**Random Read/Write:**
- High IOPS (Input/Output Operations Per Second)
- Good for databases

**Latency:**
- Read: 50-100 microseconds
- Write: 10-50 microseconds`,
					CodeExamples: `// Example: NVMe command submission (simplified)
struct nvme_command {
    uint8_t opcode;
    uint8_t flags;
    uint16_t command_id;
    uint32_t nsid;
    uint64_t metadata;
    uint64_t prp1;  // Physical Region Page
    uint64_t prp2;
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
};

// Read command
void nvme_read(uint64_t lba, uint32_t count, void *buffer) {
    struct nvme_command cmd = {0};
    cmd.opcode = 0x02;  // Read
    cmd.nsid = 1;
    cmd.cdw10 = lba & 0xFFFFFFFF;
    cmd.cdw11 = (lba >> 32) & 0xFFFFFFFF;
    cmd.cdw12 = (count - 1) & 0xFFFF;  // Number of logical blocks
    cmd.prp1 = (uint64_t)buffer;
    
    // Submit to submission queue
    submit_command(&cmd);
    
    // Wait for completion
    wait_for_completion();
}

// Example: Queue management
void nvme_init_queues(void) {
    // Create submission queue
    submission_queue = allocate_queue(1024);
    
    // Create completion queue
    completion_queue = allocate_queue(1024);
    
    // Configure queues
    nvme_admin_create_sq(submission_queue_id, submission_queue, 1024);
    nvme_admin_create_cq(completion_queue_id, completion_queue, 1024);
}

// Example: Parallel I/O
void nvme_parallel_read(uint64_t *lbas, int count, void **buffers) {
    // Submit multiple commands to different queues
    for (int i = 0; i < count; i++) {
        struct nvme_command cmd = {0};
        cmd.opcode = 0x02;
        cmd.cdw10 = lbas[i] & 0xFFFFFFFF;
        cmd.prp1 = (uint64_t)buffers[i];
        
        // Submit to queue i % num_queues
        submit_to_queue(cmd, i % num_queues);
    }
    
    // Wait for all completions
    wait_for_all_completions();
}

// Example: SSD wear leveling (concept)
void wear_leveling_write(uint64_t lba, void *data) {
    // Find physical block with least wear
    physical_block = find_least_worn_block();
    
    // Write to physical block
    write_to_block(physical_block, data);
    
    // Update mapping: LBA -> physical block
    lba_to_physical[lba] = physical_block;
    
    // Update wear count
    wear_count[physical_block]++;
}`,
				},
				{
					Title: "Persistent Memory",
					Content: `Persistent memory (PMEM) combines the speed of DRAM with the persistence of storage, enabling new computing paradigms.

**Persistent Memory Technologies:**

**Intel Optane DC Persistent Memory:**
- 3D XPoint technology
- Byte-addressable
- Non-volatile
- High capacity (128 GB - 512 GB per DIMM)
- Lower latency than SSDs

**NVDIMM (Non-Volatile DIMM):**
- DRAM with backup power
- Battery or capacitor backup
- Flushes to NAND on power loss
- Appears as regular DRAM

**Memory Mapped I/O:**
- Direct access via load/store instructions
- No file system overhead
- Atomic operations possible
- Crash consistency challenges

**Use Cases:**

**In-Memory Databases:**
- Fast recovery after crash
- No need to reload from disk
- Faster checkpointing

**High-Performance Computing:**
- Large datasets
- Fast checkpoint/restart
- Reduced I/O overhead

**Analytics:**
- Real-time processing
- Fast data access
- Reduced latency

**Challenges:**

**Crash Consistency:**
- Data may be inconsistent after crash
- Need transaction support
- Atomic operations required

**Durability:**
- Write ordering matters
- Cache flushing required
- Power-fail protected writes`,
					CodeExamples: `// Example: Persistent memory mapping
#include <libpmem.h>

// Map persistent memory
void *pmem_addr = pmem_map_file("/mnt/pmem/file", 
                                 file_size,
                                 PMEM_FILE_CREATE,
                                 0666, &mapped_len, &is_pmem);

if (pmem_addr == NULL) {
    perror("pmem_map_file");
    return;
}

// Use as regular memory
int *data = (int *)pmem_addr;
data[0] = 42;
data[1] = 100;

// Flush to ensure persistence
if (is_pmem) {
    pmem_persist(data, sizeof(int) * 2);
} else {
    pmem_msync(data, sizeof(int) * 2);
}

// Example: Atomic operations
#include <stdatomic.h>

// Atomic increment
atomic_int *counter = (atomic_int *)pmem_addr;
atomic_fetch_add(counter, 1);

// Flush after atomic operation
pmem_persist(counter, sizeof(atomic_int));

// Example: Transaction support (libpmemobj)
#include <libpmemobj.h>

PMEMobjpool *pop = pmemobj_open("/mnt/pmem/pool", NULL);
if (pop == NULL) {
    pop = pmemobj_create("/mnt/pmem/pool", "layout", 
                         PMEMOBJ_MIN_POOL, 0666);
}

// Begin transaction
TX_BEGIN(pop) {
    // Allocate in transaction
    TOID(struct my_struct) obj;
    obj = TX_NEW(struct my_struct);
    
    // Modify
    D_RW(obj)->value = 42;
    
    // Transaction automatically committed
} TX_END

// Example: Crash-consistent data structure
struct persistent_list {
    struct node *head;
    uint64_t checksum;
};

void add_node(struct persistent_list *list, int value) {
    // Allocate new node
    struct node *new_node = pmemobj_tx_alloc(sizeof(struct node),
                                             TOID_TYPE_NODE);
    
    // Modify in transaction
    TX_BEGIN(pop) {
        new_node->value = value;
        new_node->next = list->head;
        list->head = new_node;
        list->checksum = compute_checksum(list);
    } TX_END
}

// After crash, verify checksum
int verify_list(struct persistent_list *list) {
    uint64_t computed = compute_checksum(list);
    return computed == list->checksum;
}`,
				},
				{
					Title: "Memory Controllers and Interleaving",
					Content: `Memory controllers manage access to memory, optimizing performance through techniques like interleaving and scheduling.

**Memory Controller Functions:**

**Address Translation:**
- Convert logical addresses to physical
- Handle memory interleaving
- Manage multiple channels/ranks

**Command Scheduling:**
- Reorder commands for efficiency
- Minimize row switches
- Balance channels

**Timing Management:**
- Enforce timing constraints
- Handle refresh operations
- Manage power states

**Error Handling:**
- ECC (Error Correction Code)
- Retry on errors
- Bad block management

**Memory Interleaving:**

**Channel Interleaving:**
- Distribute data across memory channels
- Round-robin or address-based
- Improves bandwidth
- Example: 2 channels = 2× bandwidth potential

**Rank Interleaving:**
- Distribute across ranks
- Within same channel
- Reduces conflicts

**Bank Interleaving:**
- Distribute across banks
- Within same rank
- Allows pipelining

**Interleaving Granularity:**
- Cache line (64 bytes) common
- Larger granularity: Better for sequential access
- Smaller granularity: Better for random access`,
					CodeExamples: `// Example: Channel interleaving
// 2-channel system, 64-byte cache line
// Channel 0: Addresses 0x0000-0x003F, 0x0080-0x00BF, ...
// Channel 1: Addresses 0x0040-0x007F, 0x00C0-0x00FF, ...

uint64_t get_channel(uint64_t address) {
    // Interleave at cache line granularity
    uint64_t cache_line = address >> 6;  // 64 bytes = 2^6
    return cache_line % num_channels;
}

// Example: Memory controller scheduling
struct memory_request {
    uint64_t address;
    enum {READ, WRITE} type;
    uint64_t timestamp;
};

void schedule_requests(struct memory_request *requests, int n) {
    // Group by channel
    struct memory_request *channels[num_channels][MAX_REQUESTS];
    int channel_counts[num_channels] = {0};
    
    for (int i = 0; i < n; i++) {
        int ch = get_channel(requests[i].address);
        channels[ch][channel_counts[ch]++] = &requests[i];
    }
    
    // Schedule each channel independently
    for (int ch = 0; ch < num_channels; ch++) {
        schedule_channel(channels[ch], channel_counts[ch]);
    }
}

// Example: Row buffer management
void schedule_channel(struct memory_request **requests, int n) {
    int current_row = -1;
    
    for (int i = 0; i < n; i++) {
        int row = get_row(requests[i]->address);
        
        if (row != current_row) {
            // Need to switch rows
            if (current_row != -1) {
                precharge_row(current_row);
            }
            activate_row(row);
            current_row = row;
        }
        
        // Access column
        access_column(requests[i]->address);
    }
}

// Example: ECC (Error Correction Code)
// Single Error Correct, Double Error Detect (SECDED)
// 64-bit data + 8-bit ECC = 72 bits total

uint8_t compute_ecc(uint64_t data) {
    uint8_t ecc = 0;
    // Compute parity bits
    for (int i = 0; i < 64; i++) {
        if (data & (1ULL << i)) {
            ecc ^= ecc_matrix[i];
        }
    }
    return ecc;
}

int correct_error(uint64_t *data, uint8_t *ecc) {
    uint8_t computed_ecc = compute_ecc(*data);
    uint8_t syndrome = computed_ecc ^ *ecc;
    
    if (syndrome == 0) {
        return 0;  // No error
    }
    
    // Check for single-bit error
    int error_bit = find_error_bit(syndrome);
    if (error_bit >= 0 && error_bit < 64) {
        *data ^= (1ULL << error_bit);  // Correct error
        *ecc = compute_ecc(*data);
        return 1;  // Corrected
    }
    
    return -1;  // Uncorrectable error
}`,
				},
				{
					Title: "Error Correction Codes (ECC)",
					Content: `Error Correction Codes detect and correct errors in memory systems, essential for reliability in modern systems.

**Error Types:**

**Soft Errors:**
- Transient, caused by radiation
- Single-bit upsets (SBU)
- Can be corrected

**Hard Errors:**
- Permanent, caused by defects
- Require replacement
- Can be detected

**ECC Types:**

**Parity:**
- Single bit for error detection
- Cannot correct errors
- Simple, low overhead

**Hamming Code:**
- Single Error Correct (SEC)
- Multiple parity bits
- Overhead: log₂(n) + 1 bits for n data bits

**SECDED:**
- Single Error Correct, Double Error Detect
- Common in memory systems
- 8 bits ECC for 64 bits data

**Reed-Solomon:**
- Multiple error correction
- Used in storage systems
- Higher overhead

**ECC Implementation:**

**On-the-Fly Correction:**
- Errors corrected automatically
- Transparent to software
- Performance impact minimal

**Scrubbing:**
- Periodic read and correction
- Prevents error accumulation
- Background operation`,
					CodeExamples: `// Example: Hamming code (7,4)
// 4 data bits + 3 parity bits = 7 bits total
// Can correct single-bit errors

uint8_t hamming_encode(uint4_t data) {
    uint8_t encoded = 0;
    
    // Data bits: positions 3, 5, 6, 7
    encoded |= (data & 1) << 3;      // Bit 0 -> position 3
    encoded |= ((data >> 1) & 1) << 5; // Bit 1 -> position 5
    encoded |= ((data >> 2) & 1) << 6; // Bit 2 -> position 6
    encoded |= ((data >> 3) & 1) << 7; // Bit 3 -> position 7
    
    // Parity bits: positions 1, 2, 4
    uint8_t p1 = ((encoded >> 3) ^ (encoded >> 5) ^ (encoded >> 7)) & 1;
    uint8_t p2 = ((encoded >> 3) ^ (encoded >> 6) ^ (encoded >> 7)) & 1;
    uint8_t p4 = ((encoded >> 5) ^ (encoded >> 6) ^ (encoded >> 7)) & 1;
    
    encoded |= p1 << 1;
    encoded |= p2 << 2;
    encoded |= p4 << 4;
    
    return encoded;
}

uint4_t hamming_decode(uint8_t encoded) {
    // Compute syndrome
    uint8_t s1 = ((encoded >> 1) ^ (encoded >> 3) ^ 
                  (encoded >> 5) ^ (encoded >> 7)) & 1;
    uint8_t s2 = ((encoded >> 2) ^ (encoded >> 3) ^ 
                  (encoded >> 6) ^ (encoded >> 7)) & 1;
    uint8_t s4 = ((encoded >> 4) ^ (encoded >> 5) ^ 
                  (encoded >> 6) ^ (encoded >> 7)) & 1;
    
    uint8_t syndrome = (s4 << 2) | (s2 << 1) | s1;
    
    // Correct error if any
    if (syndrome != 0) {
        encoded ^= (1 << syndrome);  // Flip error bit
    }
    
    // Extract data
    uint4_t data = 0;
    data |= (encoded >> 3) & 1;
    data |= ((encoded >> 5) & 1) << 1;
    data |= ((encoded >> 6) & 1) << 2;
    data |= ((encoded >> 7) & 1) << 3;
    
    return data;
}

// Example: SECDED for 64-bit data
// 8-bit ECC can correct 1 error, detect 2 errors

uint8_t compute_secded_ecc(uint64_t data) {
    uint8_t ecc = 0;
    
    // Compute parity for each ECC bit
    // ECC bit i covers data bits where bit i of address is set
    for (int i = 0; i < 64; i++) {
        if (data & (1ULL << i)) {
            ecc ^= (i + 1);  // XOR with address
        }
    }
    
    // Overall parity
    uint8_t parity = __builtin_parityll(data);
    ecc |= parity << 7;
    
    return ecc;
}

int correct_secded_error(uint64_t *data, uint8_t *ecc) {
    uint8_t computed_ecc = compute_secded_ecc(*data);
    uint8_t syndrome = computed_ecc ^ *ecc;
    
    if (syndrome == 0) {
        return 0;  // No error
    }
    
    // Check overall parity
    uint8_t overall_parity = (syndrome >> 7) & 1;
    uint8_t error_position = syndrome & 0x7F;
    
    if (overall_parity == 1 && error_position > 0 && error_position <= 64) {
        // Single-bit error
        *data ^= (1ULL << (error_position - 1));
        *ecc = compute_secded_ecc(*data);
        return 1;  // Corrected
    } else if (overall_parity == 0 && error_position != 0) {
        // Double-bit error (detected but not corrected)
        return -1;  // Uncorrectable
    }
    
    return -1;  // Uncorrectable
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
