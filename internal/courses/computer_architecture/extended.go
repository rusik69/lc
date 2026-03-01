package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          209,
			Title:       "SIMD, Vector Processing, and Modern ISA Extensions",
			Description: "Learn about SIMD instructions, vector processing units, x86 SSE/AVX, ARM NEON/SVE, and how modern CPUs accelerate data-parallel workloads.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "SIMD and Vector Processing Fundamentals",
					Content: `SIMD (Single Instruction, Multiple Data) allows a single instruction to operate on multiple data elements simultaneously. This is the CPU's built-in parallelism for data-intensive workloads.

**1. Flynn's Taxonomy (Processor Classification):**
*   **SISD:** Single Instruction, Single Data -- classic sequential CPU.
*   **SIMD:** Single Instruction, Multiple Data -- one instruction processes a vector of data.
*   **MISD:** Multiple Instruction, Single Data -- rare; fault-tolerant systems.
*   **MIMD:** Multiple Instruction, Multiple Data -- multi-core CPUs, clusters.

**2. How SIMD Works:**
*   The CPU has wide registers (128, 256, or 512 bits) that hold multiple data elements.
*   A 256-bit register can hold: 8 x 32-bit floats, 4 x 64-bit doubles, 32 x 8-bit integers, etc.
*   A single ADD instruction adds ALL elements in parallel.
*   Throughput: Up to 16x for 8-bit data with 128-bit registers (32x with 256-bit, 64x with 512-bit).

**3. x86 SIMD Evolution:**
*   **MMX (1997):** 64-bit registers, integer-only. 8 registers (MM0-MM7). Shared with FPU -- could not use both simultaneously.
*   **SSE (1999):** 128-bit registers (XMM0-XMM7). Added floating-point SIMD. 4 x float32 per instruction.
*   **SSE2 (2001):** Added double-precision (2 x float64). Integer ops on XMM registers.
*   **SSE3/SSSE3/SSE4:** Horizontal ops, string processing, rounding, dot product.
*   **AVX (2011):** 256-bit registers (YMM0-YMM15). 8 x float32 per instruction. Non-destructive 3-operand encoding.
*   **AVX2 (2013):** Extended integer ops to 256-bit. Gather instructions for non-contiguous memory access.
*   **AVX-512 (2016):** 512-bit registers (ZMM0-ZMM31). 16 x float32. 32 registers (vs 16 for AVX2). Mask registers for predication. Only on Intel Xeon/i9, not on consumer AMD.

**4. ARM SIMD:**
*   **NEON:** 128-bit SIMD (32 x 128-bit registers). Available on all ARM Cortex-A cores. Fixed-width vectors.
*   **SVE (Scalable Vector Extension):** Variable-length vectors (128-2048 bits). Code is vector-length agnostic -- runs on any SVE implementation without recompilation. Used in Arm Neoverse (data center) and Fujitsu A64FX (Fugaku supercomputer).
*   **SVE2:** Extends SVE with more operations for general-purpose workloads.

**5. When SIMD Helps (and When It Doesn't):**
*   **Best for:** Array operations, image/audio processing, matrix math, parsing, memcpy, compression, crypto.
*   **Bad for:** Irregular data access patterns, heavy branching, pointer chasing, scalar-dependent computations.
*   The data must be contiguous in memory (or use gather/scatter, which are slower).
*   Alignment matters: Aligned loads (movaps) are faster than unaligned (movups), though modern CPUs have narrowed the gap.`,
					CodeExamples: `// Scalar addition (1 element at a time):
for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];  // 1 add per cycle

// SSE SIMD (4 floats at a time):
#include <xmmintrin.h>
for (int i = 0; i < N; i += 4) {
    __m128 va = _mm_load_ps(&a[i]);   // Load 4 floats
    __m128 vb = _mm_load_ps(&b[i]);   // Load 4 floats
    __m128 vc = _mm_add_ps(va, vb);   // Add 4 floats in parallel
    _mm_store_ps(&c[i], vc);          // Store 4 floats
}

// AVX2 SIMD (8 floats at a time):
#include <immintrin.h>
for (int i = 0; i < N; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);   // Load 8 floats
    __m256 vb = _mm256_load_ps(&b[i]);   // Load 8 floats
    __m256 vc = _mm256_add_ps(va, vb);   // Add 8 floats in parallel
    _mm256_store_ps(&c[i], vc);          // Store 8 floats
}

// AVX-512 SIMD (16 floats at a time):
for (int i = 0; i < N; i += 16) {
    __m512 va = _mm512_load_ps(&a[i]);
    __m512 vb = _mm512_load_ps(&b[i]);
    __m512 vc = _mm512_add_ps(va, vb);   // 16 floats in ONE instruction
    _mm512_store_ps(&c[i], vc);
}

// ARM NEON:
#include <arm_neon.h>
for (int i = 0; i < N; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    float32x4_t vc = vaddq_f32(va, vb);
    vst1q_f32(&c[i], vc);
}

// Auto-vectorization hint (compiler does it for you):
// gcc -O2 -march=native -ftree-vectorize
// The compiler may vectorize simple loops automatically.
// Check: gcc -O2 -march=native -ftree-vectorize -fopt-info-vec-all`,
				},
				{
					Title: "Matrix Operations and Specialized Instructions",
					Content: `Modern CPUs include specialized instructions for matrix operations, string processing, cryptography, and AI inference -- all built on SIMD foundations.

**1. Matrix Multiplication Acceleration:**
*   **FMA (Fused Multiply-Add):** Computes a * b + c in a single instruction with one rounding (more accurate and faster than separate multiply and add). Available in AVX2 (FMA3) and ARM NEON.
*   **AMX (Advanced Matrix Extensions, Intel):** Dedicated matrix tiles (up to 16 x 64 bytes). A single TDPBF16PS instruction multiplies two tile matrices. Massive throughput for AI workloads.
*   **SME (Scalable Matrix Extension, ARM):** Similar to AMX for ARM. Outer-product based matrix operations on SVE-length vectors.

**2. AI/ML Specific Instructions:**
*   **VNNI (Vector Neural Network Instructions, Intel):** Packed multiply-accumulate for int8/int16. Designed for inference quantized models.
*   **BF16 (Brain Float 16):** 16-bit float with same exponent range as float32 but reduced mantissa. Good for training where precision in mantissa matters less.
*   **INT8/INT4:** Quantized inference uses 8-bit or 4-bit integers with scaling factors. CPUs and GPUs support these natively.

**3. Cryptographic Acceleration:**
*   **AES-NI (Intel):** Hardware AES encryption/decryption. A single AESENC instruction performs one AES round. ~100x faster than software AES.
*   **SHA Extensions:** Hardware SHA-1 and SHA-256 computation.
*   **CRC32:** Hardware checksum computation for data integrity.
*   **CLMUL (Carry-Less Multiply):** Used for GCM (Galois/Counter Mode) authenticated encryption and CRC computation.

**4. String and Text Processing:**
*   **SSE4.2 PCMPESTRI/PCMPISTRM:** Compare 16 bytes of strings simultaneously. Used in strlen(), memchr(), and JSON parsers (e.g., simdjson).
*   **VPERMB (AVX-512):** Arbitrary byte permutation. Used for fast UTF-8 validation and base64 encoding.

**5. Bit Manipulation:**
*   **BMI1/BMI2 (Bit Manipulation Instructions):** PDEP (parallel bit deposit), PEXT (parallel bit extract), TZCNT (trailing zero count), LZCNT (leading zero count).
*   Used in chess engines, compression algorithms, and hash tables.

**6. Gather/Scatter:**
*   **Gather:** Load non-contiguous elements into a vector register using an index vector. Example: VGATHERDPS loads floats from base + indices.
*   **Scatter:** Store vector elements to non-contiguous memory locations.
*   Useful for sparse matrix operations and indirect array access.
*   Slower than contiguous loads but still faster than scalar loops.`,
					CodeExamples: `// FMA: Fused Multiply-Add (a*b + c in one instruction):
__m256 result = _mm256_fmadd_ps(a, b, c);  // 8 FMAs in parallel

// Matrix multiplication using FMA (4x4, simplified):
for (int i = 0; i < 4; i++) {
    __m256 row = _mm256_setzero_ps();
    for (int k = 0; k < 4; k++) {
        __m256 a_val = _mm256_broadcast_ss(&A[i][k]);
        __m256 b_row = _mm256_load_ps(&B[k][0]);
        row = _mm256_fmadd_ps(a_val, b_row, row);
    }
    _mm256_store_ps(&C[i][0], row);
}

// AES-NI encryption (one block):
#include <wmmintrin.h>
__m128i block = _mm_loadu_si128(plaintext);
block = _mm_xor_si128(block, roundkeys[0]);  // Initial XOR
for (int i = 1; i < 10; i++)
    block = _mm_aesenc_si128(block, roundkeys[i]);  // AES round
block = _mm_aesenclast_si128(block, roundkeys[10]); // Last round

// CRC32 hardware instruction:
uint32_t crc = 0xFFFFFFFF;
for (int i = 0; i < len; i += 8)
    crc = _mm_crc32_u64(crc, *(uint64_t*)(data + i));

// Gather (load non-contiguous data):
__m256i indices = _mm256_set_epi32(7, 3, 1, 0, 15, 11, 5, 2);
__m256 result = _mm256_i32gather_ps(array, indices, 4);
// Loads array[7], array[3], array[1], array[0], ...

// SIMD JSON parsing concept (simdjson approach):
// Process 64 bytes at a time to find structural characters
__m256i chunk1 = _mm256_loadu_si256(input);
__m256i chunk2 = _mm256_loadu_si256(input + 32);
__m256i quotes = _mm256_cmpeq_epi8(chunk1, _mm256_set1_epi8('"'));
__m256i braces = _mm256_cmpeq_epi8(chunk1, _mm256_set1_epi8('{'));
// Build bitmask of structural characters for fast parsing`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          210,
			Title:       "Memory Ordering and Consistency Models",
			Description: "Understand memory consistency models, memory barriers, atomic operations, and how they affect multi-threaded programming on different architectures.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Memory Consistency Models",
					Content: `When multiple threads access shared memory, the order in which they see each other's writes is determined by the memory consistency model. Getting this wrong causes subtle, hard-to-reproduce bugs.

**1. Why Memory Ordering Matters:**
*   Compilers reorder instructions for optimization.
*   CPUs reorder memory accesses (store buffers, out-of-order execution).
*   Caches introduce delays in propagating writes.
*   Result: Thread A writes X then Y, but Thread B might see Y before X.

**2. Sequential Consistency (SC) -- The Intuitive Model:**
*   All threads see all memory operations in a single global order.
*   Each thread's operations appear in program order.
*   Simple to reason about, but expensive to implement (no reordering allowed).
*   No real hardware implements pure SC anymore (too slow).

**3. Total Store Order (TSO) -- x86 Model:**
*   Writes may be delayed in a store buffer (store-load reordering allowed).
*   All other orderings are preserved (load-load, load-store, store-store).
*   Reads always see the latest write from the same core (store buffer forwarding).
*   This is what x86/x86-64 provides. Relatively strong -- most code works without explicit barriers.
*   Only problematic case: Thread A writes X, then reads Y; Thread B writes Y, then reads X. Both might see stale values.

**4. Weak Ordering -- ARM and RISC-V Model:**
*   Almost no ordering guarantees by default.
*   Any reordering is possible: load-load, load-store, store-load, store-store.
*   The programmer (or compiler) must insert explicit memory barriers (fences) to enforce ordering.
*   More flexibility for hardware optimization -> better performance and power efficiency.
*   This is why code ported from x86 to ARM may have subtle concurrency bugs.

**5. Release-Acquire Semantics:**
*   **Acquire:** No reads or writes after the acquire can be reordered before it. Used when taking a lock or reading a flag.
*   **Release:** No reads or writes before the release can be reordered after it. Used when releasing a lock or setting a flag.
*   Acquire-Release pairs create a happens-before relationship between threads.
*   More efficient than full barriers because they only restrict ordering in one direction.

**6. Memory Barriers (Fences):**
*   **Full Barrier (MFENCE on x86, DMB on ARM):** No reordering of any memory operation across the barrier.
*   **Store Barrier (SFENCE):** All stores before the barrier complete before stores after.
*   **Load Barrier (LFENCE):** All loads before the barrier complete before loads after.
*   Barriers are expensive (tens of cycles) -- use them only where needed.`,
					CodeExamples: `// Problem: Store buffer can cause stale reads
// Thread 1:             Thread 2:
// X = 1;               Y = 1;
// r1 = Y;              r2 = X;
// Possible result on x86 (TSO): r1 = 0 AND r2 = 0!
// Both writes are in store buffers, both reads see old values.

// Fix with memory barriers:
// Thread 1:             Thread 2:
// X = 1;               Y = 1;
// MFENCE;              MFENCE;
// r1 = Y;              r2 = X;
// Now at least one of r1 or r2 must be 1.

// C11/C++11 atomic memory orderings:
#include <stdatomic.h>

// Sequential consistency (strongest, default):
atomic_store(&flag, 1);  // memory_order_seq_cst
int val = atomic_load(&flag);

// Release-Acquire (sufficient for most lock-free code):
atomic_store_explicit(&data_ready, 1, memory_order_release);
while (!atomic_load_explicit(&data_ready, memory_order_acquire));

// Relaxed (no ordering, only atomicity):
atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);

// Go sync/atomic:
// Go's sync/atomic provides sequential consistency.
// atomic.StoreInt64(&x, 1)  // release semantics
// val := atomic.LoadInt64(&x)  // acquire semantics

// ARM requires explicit barriers (compiled from C11 atomics):
// store-release on ARM:
// STR X0, [X1]     // store data
// DMB ISH           // data memory barrier (inner shareable)
// STR X2, [X3]     // store flag (now visible in correct order)

// x86 store-release:
// MOV [X], 1       // store data (no barrier needed -- TSO!)
// MOV [flag], 1    // store flag`,
				},
				{
					Title: "Atomic Operations and Lock-Free Programming",
					Content: `Atomic operations are indivisible memory operations that enable lock-free synchronization. They are implemented directly in hardware.

**1. Why Atomics?**
*   A simple increment (x++) is NOT atomic: it's a load, add, and store. Another thread can intervene between any of these steps.
*   Atomics guarantee that the entire operation completes without interruption.
*   Locks work but have problems: priority inversion, deadlock, convoying, and overhead.
*   Lock-free programming uses atomics to avoid locks entirely.

**2. Hardware Atomic Primitives:**

**Compare-And-Swap (CAS):**
*   Atomically: if memory[addr] == expected, set memory[addr] = new_value, return true. Else return false.
*   x86: CMPXCHG instruction (with LOCK prefix for multi-core).
*   ARM: LDXR/STXR (load-exclusive/store-exclusive) loop. The store succeeds only if no other core has written to that cache line since the load.
*   CAS is the foundation of most lock-free data structures.

**Fetch-And-Add:**
*   Atomically: old = memory[addr]; memory[addr] = old + value; return old.
*   x86: LOCK XADD.
*   Common use: Concurrent counters, sequence numbers.

**Test-And-Set:**
*   Atomically: old = memory[addr]; memory[addr] = 1; return old.
*   Used for simple spinlocks.

**Load-Linked / Store-Conditional (LL/SC) -- ARM, RISC-V, MIPS:**
*   LL loads a value and sets a reservation.
*   SC stores a value only if the reservation is still valid (no other write to that cache line).
*   More general than CAS: can implement any atomic read-modify-write operation.
*   Spurious failures are possible (e.g., context switch clears the reservation), so always used in a retry loop.

**3. The ABA Problem:**
*   CAS checks if value == expected, but the value might have changed from A to B and back to A.
*   CAS succeeds even though the value was modified (it just returned to the original).
*   Solution: Double-width CAS with a counter (DCAS), or hazard pointers, or epoch-based reclamation.

**4. Lock-Free vs Wait-Free:**
*   **Lock-Free:** Guaranteed that at least one thread makes progress. May have starvation for individual threads.
*   **Wait-Free:** Every thread makes progress in bounded steps. Harder to implement, stronger guarantee.
*   **Obstruction-Free:** Progress guaranteed if a thread runs in isolation (weakest guarantee).

**5. Common Lock-Free Data Structures:**
*   **Lock-Free Stack (Treiber Stack):** CAS on the head pointer to push/pop.
*   **Lock-Free Queue (Michael-Scott Queue):** CAS on head and tail pointers.
*   **Lock-Free Hash Map:** Per-bucket CAS or split-ordered lists.
*   **Read-Copy-Update (RCU):** Readers never block. Writers create a new version and atomically swap the pointer. Used extensively in the Linux kernel.`,
					CodeExamples: `// Compare-And-Swap (CAS) in C11:
#include <stdatomic.h>
_Atomic int counter = 0;

void atomic_increment() {
    int old_val, new_val;
    do {
        old_val = atomic_load(&counter);
        new_val = old_val + 1;
    } while (!atomic_compare_exchange_weak(&counter, &old_val, new_val));
}

// Lock-free stack (Treiber Stack):
struct Node { int data; struct Node* next; };
_Atomic(struct Node*) top = NULL;

void push(int value) {
    struct Node* new_node = malloc(sizeof(struct Node));
    new_node->data = value;
    struct Node* old_top;
    do {
        old_top = atomic_load(&top);
        new_node->next = old_top;
    } while (!atomic_compare_exchange_weak(&top, &old_top, new_node));
}

// Go atomic CAS:
import "sync/atomic"
var counter int64

func increment() {
    for {
        old := atomic.LoadInt64(&counter)
        if atomic.CompareAndSwapInt64(&counter, old, old+1) {
            break
        }
    }
}

// x86 assembly for LOCK CMPXCHG:
// lock cmpxchg [rdi], rsi
// If [rdi] == rax: [rdi] = rsi, ZF=1
// Else:            rax = [rdi], ZF=0

// ARM load-exclusive / store-exclusive:
// retry:
//   ldxr  w0, [x1]        // Load-exclusive
//   add   w0, w0, #1      // Modify
//   stxr  w2, w0, [x1]    // Store-exclusive
//   cbnz  w2, retry       // Retry if store failed`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          211,
			Title:       "Hardware Virtualization",
			Description: "Learn how CPUs support virtual machines: VT-x, EPT/NPT, SR-IOV, and how hypervisors leverage hardware assistance for efficient virtualization.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "CPU Virtualization (VT-x / AMD-V)",
					Content: `Hardware virtualization allows multiple operating systems to run simultaneously on a single physical CPU by providing hardware support for trapping and emulating privileged operations.

**1. The Virtualization Problem:**
*   An OS expects to run in Ring 0 (kernel mode) with full hardware control.
*   With multiple VMs, only the hypervisor can run in Ring 0.
*   Guest OSes think they are in Ring 0 but aren't -- their privileged instructions must be intercepted.
*   x86 was historically hard to virtualize because some privileged instructions silently fail in Ring 3 instead of trapping.

**2. Software Virtualization (Before Hardware Support):**
*   **Binary Translation (VMware):** Scan guest code, replace problematic instructions with safe equivalents at runtime. Works but slow and complex.
*   **Paravirtualization (Xen):** Modify the guest OS to call the hypervisor directly (hypercalls). Fast but requires OS modification.

**3. Hardware-Assisted Virtualization:**
*   **Intel VT-x (2005) / AMD-V (SVM):** Add a new CPU mode specifically for hypervisors.
*   Two new modes: VMX Root (hypervisor) and VMX Non-Root (guest).
*   Guest runs in its own Ring 0, but certain operations cause a VM Exit to the hypervisor.
*   The hypervisor handles the exit and resumes the guest with VM Enter.

**VMCS (Virtual Machine Control Structure):**
*   A per-VM data structure that defines what causes VM Exits.
*   Contains: guest state (registers, CR3, IDTR), host state (hypervisor registers), execution controls (which events trigger exits).
*   VMLAUNCH starts a VM. VMRESUME re-enters after an exit.

**4. Common VM Exit Causes:**
*   I/O instructions (IN/OUT) -- unless passed through.
*   CR3 writes (page table changes) -- unless using EPT/NPT.
*   CPUID -- hypervisor spoofs CPU features.
*   External interrupts.
*   MSR (Model-Specific Register) access.
*   HLT instruction.

**5. Nested Virtualization:**
*   Running a hypervisor inside a VM (VM inside a VM).
*   Requires hardware support for nested VMCS.
*   Intel added VMCS shadowing for efficient nested exits.
*   Use case: Testing hypervisors, cloud-in-cloud scenarios.

**6. Performance Impact:**
*   VM Exits are expensive: ~1000-3000 cycles each.
*   Goal: Minimize exits. Modern hypervisors use EPT, interrupt virtualization (APICv), and device passthrough to reduce exits.
*   A well-tuned VM runs at 95-99% of bare-metal performance for compute workloads.`,
					CodeExamples: `// Simplified hypervisor VM lifecycle:
// 1. Allocate VMCS
vmcs = vmcs_alloc();
vmcs_clear(vmcs);
vmcs_load(vmcs);

// 2. Configure guest state
vmcs_write(GUEST_CS, guest_cs_selector);
vmcs_write(GUEST_RIP, guest_entry_point);
vmcs_write(GUEST_RSP, guest_stack_pointer);
vmcs_write(GUEST_CR3, guest_page_table);
vmcs_write(GUEST_RFLAGS, 0x2); // Minimum flags

// 3. Configure exit controls
vmcs_write(CPU_EXEC_CONTROL, EXIT_ON_HLT | EXIT_ON_IO);

// 4. Enter VM
vmlaunch();

// 5. Handle VM Exit
while (1) {
    int reason = vmcs_read(VM_EXIT_REASON);
    switch (reason) {
        case EXIT_REASON_IO:
            emulate_io(vmcs_read(EXIT_QUAL));
            break;
        case EXIT_REASON_CPUID:
            emulate_cpuid(); // Spoof CPU features
            break;
        case EXIT_REASON_HLT:
            schedule_other_vm();
            break;
    }
    vmresume();
}

// KVM (Linux) ioctl interface:
int vm_fd = ioctl(kvm_fd, KVM_CREATE_VM, 0);
int vcpu_fd = ioctl(vm_fd, KVM_CREATE_VCPU, 0);
struct kvm_run *run = mmap(NULL, size, PROT_RW, MAP_SHARED, vcpu_fd, 0);

while (1) {
    ioctl(vcpu_fd, KVM_RUN, 0);
    switch (run->exit_reason) {
        case KVM_EXIT_IO:
            handle_io(run->io);
            break;
        case KVM_EXIT_MMIO:
            handle_mmio(run->mmio);
            break;
    }
}`,
				},
				{
					Title: "Memory Virtualization (EPT/NPT) and I/O Virtualization",
					Content: `Memory and I/O virtualization are the two biggest performance challenges in virtual machines. Hardware solutions eliminate most of the overhead.

**1. The Memory Virtualization Problem:**
*   Guest uses Guest Virtual Addresses (GVA), translated to Guest Physical Addresses (GPA) by the guest page table.
*   But GPAs are not real physical addresses. The hypervisor must translate GPA -> HPA (Host Physical Address).
*   Without hardware help: Shadow Page Tables. The hypervisor maintains a GVA -> HPA mapping, intercepting every page table modification. Extremely expensive.

**2. EPT (Extended Page Tables, Intel) / NPT (Nested Page Tables, AMD):**
*   Hardware performs a two-dimensional page walk: GVA -> GPA (guest page table), then GPA -> HPA (EPT/NPT table).
*   No shadow page tables needed. No VM exits on CR3 writes or page faults in guest.
*   EPT violation (GPA not mapped in EPT): causes a VM exit for the hypervisor to handle.
*   Downside: TLB miss is more expensive (up to 24 memory accesses for a 4-level walk in both dimensions). Hardware caches (EPT TLB, page walk cache) mitigate this.
*   With EPT, memory virtualization overhead drops from ~30% to ~2-5%.

**3. I/O Virtualization:**

**Emulated I/O:**
*   Hypervisor intercepts guest I/O and emulates the device in software.
*   Example: QEMU emulates a virtual NIC or disk controller.
*   Flexible (any device) but slow (every I/O = VM exit).

**Virtio (Paravirtual I/O):**
*   Standardized interface between guest and hypervisor.
*   Guest uses a virtio driver that communicates via shared memory rings (no emulation).
*   Much faster than full emulation. The standard for KVM/QEMU.
*   virtio-net, virtio-blk, virtio-scsi, virtio-gpu.

**SR-IOV (Single Root I/O Virtualization):**
*   Hardware-level I/O virtualization.
*   A single physical NIC or SSD presents itself as multiple Virtual Functions (VFs).
*   Each VF is assigned directly to a VM -- the guest talks to real hardware with no hypervisor in the data path.
*   Near bare-metal I/O performance.
*   Requires: SR-IOV capable device + IOMMU (Intel VT-d / AMD-Vi).

**4. IOMMU (Intel VT-d / AMD-Vi):**
*   Translates device DMA addresses to host physical addresses.
*   Prevents a device (or VM with a passthrough device) from accessing memory it shouldn't.
*   Essential for security when using device passthrough.
*   Also provides interrupt remapping (directing device interrupts to the correct VM).

**5. Live Migration:**
*   Moving a running VM from one physical host to another with minimal downtime.
*   Steps: Pre-copy (iteratively copy dirty pages), stop VM, transfer final dirty pages and CPU state, start on destination.
*   EPT/NPT make dirty page tracking efficient (the "dirty" bit in EPT entries).
*   Typical downtime: 10-100 ms for well-tuned systems.`,
					CodeExamples: `// EPT page table structure (Intel):
// EPT PML4 Entry -> EPT PDPT Entry -> EPT PD Entry -> EPT PT Entry -> HPA
//
// Each entry:
// Bits [51:12]: Physical address of next level / page frame
// Bit 0: Read access
// Bit 1: Write access  
// Bit 2: Execute access
// Bit 6: Dirty bit (for live migration tracking)

// Two-dimensional page walk (worst case):
// Guest CR3 -> GPA (need EPT walk: 4 accesses)
// PML4[GVA] -> GPA (need EPT walk: 4 accesses)
// PDPT[GVA] -> GPA (need EPT walk: 4 accesses)
// PD[GVA]   -> GPA (need EPT walk: 4 accesses)
// PT[GVA]   -> GPA (need EPT walk: 4 accesses)
// Final page -> GPA (need EPT walk: 4 accesses)
// Total: up to 24 memory accesses for one TLB miss!

// Virtio ring buffer (simplified):
struct virtq {
    struct virtq_desc  desc[QUEUE_SIZE];   // Descriptors
    struct virtq_avail avail;              // Guest -> Host
    struct virtq_used  used;               // Host -> Guest
};

// Guest sends packet:
// 1. Write data to desc[i].addr
// 2. Add index i to avail ring
// 3. Kick host (write to MMIO register)
// Host processes:
// 4. Read desc from avail ring
// 5. Process data (e.g., send network packet)
// 6. Add index to used ring
// 7. Inject interrupt to guest

// SR-IOV configuration (Linux):
// Enable VFs on a NIC:
// echo 4 > /sys/class/net/eth0/device/sriov_numvfs
// Assign VF to VM via VFIO:
// vfio-bind 0000:03:10.0
// In QEMU: -device vfio-pci,host=03:10.0`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          212,
			Title:       "Power Management and Thermal Design",
			Description: "Understand CPU power states, dynamic voltage and frequency scaling, thermal management, and energy-efficient processor design for mobile and data center.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "CPU Power Management",
					Content: `Power consumption is the dominant constraint in modern processor design. The "power wall" ended the era of frequency scaling and drove the shift to multi-core and heterogeneous architectures.

**1. Sources of Power Consumption:**

**Dynamic Power:**
*   P_dynamic = alpha * C * V^2 * f
*   alpha = activity factor (fraction of transistors switching).
*   C = capacitance (proportional to transistor count and wire length).
*   V = supply voltage (MOST IMPORTANT -- quadratic effect).
*   f = clock frequency.
*   Reducing voltage from 1.0V to 0.8V cuts dynamic power by 36%.

**Static Power (Leakage):**
*   P_static = V * I_leak
*   Current leaks through transistors even when not switching.
*   Leakage increases exponentially with temperature.
*   At smaller process nodes (7nm, 5nm), leakage becomes a larger fraction of total power.
*   In modern CPUs, leakage can be 30-50% of total power.

**2. DVFS (Dynamic Voltage and Frequency Scaling):**
*   The CPU adjusts voltage and frequency together based on workload.
*   Higher demand -> increase frequency and voltage -> more power, more performance.
*   Idle or low demand -> decrease both -> huge power savings.
*   Intel: Speed Shift (hardware-controlled, ~1 ms transitions). AMD: Precision Boost.
*   OS role: The governor (e.g., Linux schedutil) tells hardware the desired performance level.

**3. CPU Power States (ACPI):**

**P-States (Performance States):**
*   P0: Maximum performance (highest frequency and voltage).
*   P1, P2, ...: Progressively lower frequency/voltage.
*   Transitions take ~10-100 microseconds.

**C-States (Idle States):**
*   C0: Active -- CPU is executing instructions.
*   C1 (Halt): Clock stopped. Wake-up: ~1 microsecond.
*   C3 (Sleep): L1/L2 caches flushed. Wake-up: ~100 microseconds.
*   C6 (Deep Sleep): Core voltage reduced to near zero. Wake-up: ~200 microseconds.
*   C7+: Package-level power gating. All cores in C6, shared cache powered down.
*   Deeper C-states save more power but have higher wake-up latency.

**4. Thermal Design:**

**TDP (Thermal Design Power):**
*   The maximum sustained power the cooling solution must handle.
*   NOT the maximum instantaneous power (which can be 2x TDP during turbo boost).
*   Example: Intel i9-13900K TDP: 125W, max turbo power: 253W.

**Thermal Throttling:**
*   When the CPU exceeds its thermal limit (~100C), it reduces frequency to cool down.
*   This is automatic and unavoidable -- inadequate cooling directly reduces performance.

**5. Heterogeneous Architectures (big.LITTLE / Hybrid):**
*   **Big cores:** High performance, high power. For burst workloads.
*   **Little cores (efficiency cores):** Lower performance, much lower power. For background tasks.
*   ARM big.LITTLE: Cortex-A78 (big) + Cortex-A55 (little). Apple M-series: P-cores + E-cores. Intel Alder Lake: P-cores + E-cores.
*   The OS scheduler (thread director) assigns tasks to the appropriate core type.
*   Background downloads -> E-core. Video encoding -> P-core. Idle -> E-core in deep sleep.

**6. Data Center Power Optimization:**
*   PUE (Power Usage Effectiveness) = Total facility power / IT equipment power. Industry avg: ~1.3-1.6. Google: ~1.1.
*   Server power management: race to idle (finish fast, sleep deep), DVFS, load-based frequency.
*   Liquid cooling, immersion cooling for high-density racks.`,
					CodeExamples: `// Power calculation example:
// Original: V=1.0V, f=4GHz, C=1nF, alpha=0.3
// P = 0.3 * 1e-9 * (1.0)^2 * 4e9 = 1.2W per gate group

// Reduced: V=0.7V, f=2GHz
// P = 0.3 * 1e-9 * (0.7)^2 * 2e9 = 0.294W
// Power savings: 75.5%! Performance loss: 50%
// Energy per operation: actually BETTER at lower voltage

// Linux CPU frequency governor:
// cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
// Options: performance, powersave, schedutil (recommended)
// echo "schedutil" > scaling_governor

// Check current frequency:
// cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
// 4200000 (4.2 GHz)

// Check available C-states:
// cat /sys/devices/system/cpu/cpu0/cpuidle/state*/name
// POLL, C1, C1E, C3, C6, C7s, C8, C9, C10

// C-state residency:
// cat /sys/devices/system/cpu/cpu0/cpuidle/state3/time
// 123456789 (microseconds spent in C3)

// Intel RAPL (Running Average Power Limit):
// Read power consumption via MSR:
// perf stat -e power/energy-pkg/ ./my_program
// Output: 42.56 Joules power/energy-pkg/
//         (for 10 seconds = average 4.256W package power)

// Thermal monitoring:
// cat /sys/class/thermal/thermal_zone0/temp
// 52000 (52.0 degrees Celsius)

// Heterogeneous scheduling concept:
// Linux kernel's Energy-Aware Scheduling (EAS):
// 1. Task wakes up
// 2. Estimate task utilization
// 3. If utilization < threshold: place on little core
// 4. If utilization > threshold: place on big core
// 5. Migrate if utilization changes significantly`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          213,
			Title:       "Modern ARM and RISC-V Architectures",
			Description: "Deep dive into ARM and RISC-V: Apple Silicon, ARM Cortex series, RISC-V extensions, and how these architectures are reshaping computing.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "ARM Architecture Deep Dive",
					Content: `ARM (Advanced RISC Machines) dominates mobile computing and is rapidly expanding into servers and desktops. Understanding ARM is essential as it becomes the primary alternative to x86.

**1. ARM Business Model:**
*   ARM Holdings designs the ISA and reference cores but does not fabricate chips.
*   Licenses designs to companies (Apple, Qualcomm, Samsung, MediaTek, Ampere).
*   Two license types: architectural license (design own cores using ARM ISA; Apple, Qualcomm) and core license (use ARM's pre-designed cores; most others).

**2. ARM Cortex Series:**

**Cortex-A (Application):**
*   High-performance cores for phones, tablets, laptops, servers.
*   Cortex-A78, A710, A715, X3, X4 (big), A510, A520 (little).
*   Out-of-order execution, branch prediction, multi-level caches.

**Cortex-R (Real-Time):**
*   Deterministic, low-latency cores for safety-critical systems.
*   Automotive (ABS, airbags), industrial controllers, 5G modems.
*   Typically in-order, tightly coupled memories.

**Cortex-M (Microcontroller):**
*   Ultra-low power cores for IoT, sensors, embedded.
*   M0/M0+ (simplest), M3/M4 (with DSP), M7 (highest performance), M33/M55 (with security).
*   In-order, 2-3 stage pipeline, microamps of power.

**3. Apple Silicon (M-series):**
*   Custom ARM cores designed by Apple. Not Cortex cores but ARM ISA compatible.
*   Apple M1/M2/M3/M4: Performance (Firestorm/Avalanche/etc.) + Efficiency cores.
*   Key innovations:
    *   Extremely wide decode (8-wide for P-cores vs 4-6 for typical ARM).
    *   Huge Reorder Buffer (600+ entries vs ~200 for Intel).
    *   Unified Memory Architecture (UMA): CPU, GPU, Neural Engine share the same memory pool (no data copying).
    *   Integrated GPU, Neural Engine, media engines, and Secure Enclave on one SoC.
*   Performance per watt significantly exceeds x86 competitors.

**4. ARM in the Data Center:**
*   AWS Graviton3/4 (custom ARM Neoverse cores): Lower cost, lower power per vCPU than x86.
*   Ampere Altra: 128 Cortex-N1 cores. Predictable per-core performance (no turbo boost -- consistent for cloud).
*   NVIDIA Grace: ARM Neoverse V2 cores paired with NVIDIA GPU via NVLink.
*   Microsoft Azure Cobalt: Custom ARM for Azure VMs.

**5. ARMv9 Architecture (Latest):**
*   SVE2 (Scalable Vector Extension 2) replaces NEON as the primary SIMD.
*   Confidential Compute Architecture (CCA): Hardware-isolated realms for security.
*   Transactional Memory Extension (TME): Hardware transactional memory.
*   Memory Tagging Extension (MTE): Tags memory to detect use-after-free and buffer overflows.`,
					CodeExamples: `// ARM vs x86 instruction comparison:
// Task: Add two values from memory

// ARM (RISC - load/store architecture):
LDR  X0, [X1]        // Load from memory into register
LDR  X2, [X3]        // Load from memory into register
ADD  X4, X0, X2      // Add registers
STR  X4, [X5]        // Store result to memory
// 4 instructions, each 4 bytes = 16 bytes

// x86 (CISC - can operate directly on memory):
MOV  EAX, [RBX]      // Load from memory
ADD  EAX, [RCX]      // Add from memory (load + add combined)
MOV  [RDX], EAX      // Store result
// 3 instructions, but variable length (3-7 bytes each)

// Apple M-series Unified Memory:
// No separate CPU RAM and GPU VRAM
// CPU writes data -> GPU reads it immediately (zero-copy)
// Traditional x86: CPU writes to RAM -> copy over PCIe to GPU VRAM
// UMA eliminates the PCIe bottleneck for CPU-GPU data sharing

// ARM Memory Tagging Extension (MTE):
// Each 16-byte granule of memory gets a 4-bit tag
// Pointers also carry a tag in the top bits
// On access: if pointer tag != memory tag -> fault
// Catches use-after-free: free() changes the memory tag
// Catches buffer overflow: adjacent granules have different tags

// ARM SVE (vector-length agnostic code):
// This same binary runs on 128-bit, 256-bit, or 512-bit SVE hardware
//   whilelt p0.s, xzr, x0      // Create predicate for loop
// loop:
//   ld1w   z0.s, p0/z, [x1]   // Load vector (length determined by hardware)
//   ld1w   z1.s, p0/z, [x2]   // Load vector
//   add    z2.s, z0.s, z1.s   // Add vectors
//   st1w   z2.s, p0, [x3]     // Store vector
//   incw   x1                    // Increment by vector length
//   whilelt p0.s, x1, x0       // Update predicate
//   b.first loop                // Loop if elements remain`,
				},
				{
					Title: "RISC-V: The Open ISA Revolution",
					Content: `RISC-V is an open-source ISA that is free to implement without licensing fees. It is growing rapidly from academic research to commercial products.

**1. Why RISC-V Matters:**
*   **No licensing fees:** Anyone can design a RISC-V chip without paying ARM or Intel.
*   **Open standard:** The ISA specification is public and governed by RISC-V International.
*   **Modular:** Start with a tiny base ISA and add only the extensions you need.
*   **Clean design:** No legacy baggage (unlike x86 with 40 years of backward compatibility).
*   **Growing ecosystem:** Compilers (GCC, LLVM), Linux support, RTOS support, commercial IP vendors.

**2. RISC-V Base ISAs:**
*   **RV32I:** 32-bit base integer. 32 registers (x0-x31). ~47 instructions. Minimum viable ISA.
*   **RV64I:** 64-bit base integer. For servers, desktops, high-performance.
*   **RV32E:** Embedded variant with 16 registers (for tiny microcontrollers).
*   **RV128I:** 128-bit (future, for very large address spaces).

**3. Standard Extensions (Modular):**
*   **M:** Integer Multiply and Divide.
*   **A:** Atomic instructions (LR/SC, AMO).
*   **F:** Single-precision floating point (32-bit).
*   **D:** Double-precision floating point (64-bit).
*   **C:** Compressed instructions (16-bit encodings for common instructions). Reduces code size by ~25-30%.
*   **V:** Vector extension (scalable SIMD, similar philosophy to ARM SVE).
*   **B:** Bit manipulation.
*   **Zicsr:** Control and Status Registers.
*   **G = IMAFD:** "General-purpose" combination, the typical set for application processors.

**4. RISC-V Design Philosophy:**
*   Fixed 32-bit instruction encoding (except C extension).
*   Load-store architecture: Only LW/SW access memory; all computation on registers.
*   No condition codes / flags register. Branches compare two registers directly (BEQ, BLT, BGE).
*   x0 is hardwired to zero (useful as a discard register and for generating constants).
*   Simple encoding: opcodes are regular, making decoders simpler.

**5. RISC-V in Practice:**

**Commercial Products:**
*   **SiFive:** Performance P670 (application processor), Intelligence X280 (AI).
*   **Alibaba T-Head:** Xuantie C910 (server-class core).
*   **Espressif ESP32-C3:** WiFi+BLE microcontroller (replacing ARM Cortex-M).
*   **StarFive JH7110:** Quad-core RV64GC SoC running Linux.
*   **Qualcomm and Google:** Investing in RISC-V for wearables and embedded.

**Custom Extensions:**
*   Companies can add proprietary instructions for their specific domain.
*   Example: AI accelerator with custom matrix-multiply instructions.
*   Must not conflict with standard extensions (reserved opcode space for custom use).

**6. RISC-V vs ARM vs x86 Comparison:**
*   **Code density:** x86 > RISC-V+C > ARM A64 > RISC-V (no C).
*   **Decode complexity:** RISC-V << ARM < x86 (variable-length x86 is hardest).
*   **Ecosystem maturity:** x86 >> ARM >> RISC-V (but RISC-V is catching up fast).
*   **License cost:** RISC-V = $0, ARM = per-chip royalty, x86 = not licensable.`,
					CodeExamples: `// RISC-V assembly example (RV64I):
// Function: int add(int a, int b) { return a + b; }
add_func:
    add  a0, a0, a1     // a0 = a0 + a1 (result in a0)
    ret                  // Return (jalr x0, ra, 0)

// RISC-V register convention (calling convention):
// x0  (zero): Hardwired to 0
// x1  (ra):   Return address
// x2  (sp):   Stack pointer
// x3  (gp):   Global pointer
// x4  (tp):   Thread pointer
// x5-x7 (t0-t2):   Temporaries (caller-saved)
// x8  (s0/fp): Saved register / frame pointer
// x9  (s1):   Saved register
// x10-x11 (a0-a1): Function arguments / return values
// x12-x17 (a2-a7): Function arguments
// x18-x27 (s2-s11): Saved registers (callee-saved)
// x28-x31 (t3-t6): Temporaries (caller-saved)

// No flags register! Branch by comparing registers:
// x86:   CMP rax, rbx ; JE label    (2 instructions + flags)
// RISC-V: beq a0, a1, label          (1 instruction, no flags)

// RISC-V atomic (A extension):
// Compare-and-swap using LR/SC:
retry:
    lr.w  t0, (a0)       // Load-reserved word from [a0]
    bne   t0, a1, fail   // If *a0 != expected, fail
    sc.w  t1, a2, (a0)   // Store-conditional a2 to [a0]
    bnez  t1, retry       // If SC failed (interrupted), retry
    li    a0, 1           // Success
    ret
fail:
    li    a0, 0           // Failure
    ret

// RISC-V Compressed instructions (C extension):
// Standard:    add x10, x10, x11    // 32 bits
// Compressed:  c.add x10, x11       // 16 bits (same operation!)

// RISC-V Vector extension (V):
// Vector add: a[i] = b[i] + c[i]
vsetvli t0, a0, e32, m1  // Set vector length (e32=32-bit elements)
vle32.v v1, (a1)          // Load vector from b
vle32.v v2, (a2)          // Load vector from c
vadd.vv v3, v1, v2        // Vector add
vse32.v v3, (a3)          // Store result to a`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          214,
			Title:       "Compiler Optimization and Hardware Interaction",
			Description: "Learn how compilers optimize code for hardware: instruction scheduling, register allocation, loop transformations, and profile-guided optimization.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Compiler Optimizations for Modern CPUs",
					Content: `The compiler is the bridge between your source code and the hardware. Understanding compiler optimizations explains why the same algorithm can run 2-10x faster with the right compilation flags.

**1. Optimization Levels:**
*   **-O0:** No optimization. Fastest compile, slowest execution. Good for debugging.
*   **-O1:** Basic optimizations (dead code elimination, constant folding). Good balance.
*   **-O2:** Most optimizations without increasing code size excessively. The "standard" for releases.
*   **-O3:** Aggressive optimizations (loop unrolling, vectorization, function inlining). May increase code size.
*   **-Os:** Optimize for size (important for embedded, instruction cache pressure).
*   **-Ofast:** O3 plus unsafe math optimizations (reassociation, no NaN/Inf checks). NOT IEEE 754 compliant.

**2. Key Optimizations:**

**Constant Folding and Propagation:**
*   Compute constant expressions at compile time.
*   x = 3 + 5 becomes x = 8. No runtime addition needed.
*   Propagate: if a = 10 and b = a * 2, then b = 20 at compile time.

**Dead Code Elimination:**
*   Remove code that has no effect on the output.
*   Unreachable code after a return statement, unused variable computations.

**Common Subexpression Elimination (CSE):**
*   If the same expression is computed twice, compute it once and reuse.
*   a = b * c + d; e = b * c + f; becomes tmp = b * c; a = tmp + d; e = tmp + f;

**Strength Reduction:**
*   Replace expensive operations with cheaper ones.
*   x * 2 becomes x << 1 (shift is 1 cycle, multiply is 3-5 cycles).
*   x / 8 becomes x >> 3.
*   x % 4 becomes x & 3.

**Function Inlining:**
*   Replace a function call with the function body.
*   Eliminates call/return overhead and enables further optimizations across the inlined code.
*   Downside: increases code size (may cause instruction cache pressure).

**3. Loop Optimizations:**

**Loop Unrolling:**
*   Replicate the loop body multiple times per iteration.
*   Reduces branch overhead and enables better instruction scheduling.
*   4x unroll: 75% fewer loop branches.

**Loop Vectorization (Auto-vectorization):**
*   The compiler converts scalar loops to SIMD instructions.
*   for (i=0; i<N; i++) c[i] = a[i] + b[i]; becomes SIMD vector adds.
*   Requires: no loop-carried dependencies, contiguous memory access, no complex control flow.

**Loop Tiling (Blocking):**
*   Process data in cache-sized blocks to improve temporal locality.
*   Essential for matrix multiplication (O(N^3) -> fits in L1 cache with tiling).

**Loop Interchange:**
*   Swap inner and outer loops to improve spatial locality.
*   Column-major traversal swapped to row-major for C arrays.

**4. Profile-Guided Optimization (PGO):**
*   Step 1: Compile with instrumentation (-fprofile-generate).
*   Step 2: Run with representative workload (collects branch frequencies, hot paths).
*   Step 3: Recompile with profile data (-fprofile-use).
*   Result: Compiler knows exact branch probabilities, hot/cold functions, loop trip counts.
*   Typical speedup: 10-30% beyond O2.

**5. Link-Time Optimization (LTO):**
*   Optimize across compilation units (source files) at link time.
*   Enables inlining and CSE across files.
*   Downside: Slower link step. Can be mitigated with ThinLTO (parallel, incremental).`,
					CodeExamples: `// Strength reduction:
// Before:
for (int i = 0; i < N; i++)
    a[i] = i * 7;

// After (compiler transforms):
int tmp = 0;
for (int i = 0; i < N; i++) {
    a[i] = tmp;
    tmp += 7;  // Addition instead of multiplication
}

// Loop unrolling (4x):
// Before:
for (int i = 0; i < N; i++) sum += a[i];

// After:
for (int i = 0; i < N; i += 4) {
    sum0 += a[i];
    sum1 += a[i+1];
    sum2 += a[i+2];
    sum3 += a[i+3];
}
sum = sum0 + sum1 + sum2 + sum3;
// 4 independent accumulators break the dependency chain

// Auto-vectorization check (GCC):
// gcc -O2 -march=native -ftree-vectorize -fopt-info-vec-all main.c
// Output: "loop vectorized using 32 byte vectors"

// PGO workflow:
// Step 1: gcc -O2 -fprofile-generate -o prog prog.c
// Step 2: ./prog < representative_input.txt
// Step 3: gcc -O2 -fprofile-use -o prog_optimized prog.c
// Step 4: ./prog_optimized  # 10-30% faster

// Compiler barriers (prevent reordering):
// C11:
atomic_signal_fence(memory_order_seq_cst); // Compiler barrier only
atomic_thread_fence(memory_order_seq_cst); // Compiler + hardware barrier

// GCC:
asm volatile("" ::: "memory");  // Compiler barrier (no hardware fence)

// Go:
// The Go compiler performs fewer aggressive optimizations than GCC/LLVM
// but benefits from escape analysis, inlining, and bounds check elimination.
// //go:noinline    -- prevent inlining
// //go:nosplit     -- prevent stack split check
// //go:noescape   -- mark that pointer does not escape`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
