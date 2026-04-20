package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          2833,
			Title:       "Virtualization and Security Architecture",
			Description: "Understand hardware virtualization support, trusted execution environments, side-channel attacks, and security features in modern processors.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "Hardware Virtualization",
					Content: `Hardware virtualization allows multiple operating systems to run simultaneously on the same physical hardware. Modern CPUs include dedicated features to make virtualization efficient.

**Virtualization Fundamentals:**
` + "```" + `
The Popek-Goldberg Requirements (1974):
  A VM monitor (hypervisor) must:
  1. Equivalence: Software runs identically in VM as on bare metal
  2. Resource Control: Hypervisor controls all resources
  3. Efficiency: Most instructions execute directly on hardware

Problem with x86 (before VT-x):
  Some "sensitive" instructions don't trap when executed in user mode
  Example: SGDT (Store GDT Register) - returns different value in VM
  These instructions must be manually trapped via binary translation
  
  Binary Translation (VMware pre-2005):
    Scan guest code → replace sensitive instructions with traps
    Overhead: 10-30% for some workloads
    Complex to implement correctly

Hardware Virtual Machine Extensions:

Intel VT-x / AMD-V (2005-2006):
  New processor mode: VMX root (hypervisor) + VMX non-root (guest)
  VMCS (Virtual Machine Control Structure):
    - Stores guest/host state
    - Controls which events cause VM exits
    - Hardware automatically saves/restores context
    
  Key operations:
    VMLAUNCH: Start a new VM
    VMRESUME: Resume a suspended VM
    VM Exit:  Control returns to hypervisor (trap)
    VM Entry: Control passes to guest VM
    
  VM Exit causes (configurable):
    - Privileged instructions (HLT, CPUID, IN/OUT)
    - Interrupt/exception delivery
    - Access to certain control registers
    - Page fault (if configured)
    - External interrupts
` + "```" + `

**Memory Virtualization:**
` + "```" + `
Two-Level Address Translation:

Without hardware support (Shadow Page Tables):
  Guest Virtual → Guest Physical → Host Physical
  Hypervisor maintains "shadow" page tables
  Maps Guest Virtual directly to Host Physical
  Expensive: every guest page table change requires shadow update
  
  Guest PT:  GVA → GPA
  Shadow PT: GVA → HPA  (maintained by hypervisor)
  
  Problem: Guest writes CR3 → VM exit → hypervisor rebuilds shadow PT

With hardware support (Intel EPT / AMD NPT):
  Hardware walks TWO page tables:
  1. Guest page table: GVA → GPA (guest controlled)
  2. EPT (Extended Page Table): GPA → HPA (hypervisor controlled)
  
  ┌─────┐    Guest PT     ┌─────┐    EPT        ┌─────┐
  │ GVA │───────────────→ │ GPA │──────────────→│ HPA │
  └─────┘                 └─────┘               └─────┘
  
  Page walk becomes nested:
    Each level of guest PT lookup requires EPT translation
    4-level guest PT × 4-level EPT = up to 24 memory accesses!
    (Mitigated by: large TLB, VPID tags, page walk caching)

  VPID (Virtual Processor ID):
    Tags TLB entries with VM identifier
    No need to flush TLB on VM switch
    Dramatic performance improvement

SR-IOV (Single Root I/O Virtualization):
  Hardware-level I/O virtualization for network/storage
  NIC presents multiple "Virtual Functions" (VFs)
  Each VM gets direct access to its own VF
  Bypasses hypervisor for data plane → near-native I/O performance
  
  Physical Function (PF): Full NIC, managed by hypervisor
  Virtual Function (VF):  Lightweight, assigned to VM
  
  Without SR-IOV: VM → virtio → hypervisor → driver → NIC (slow)
  With SR-IOV:    VM → VF driver → NIC directly (fast!)
` + "```" + `

**I/O Virtualization and IOMMU:**
` + "```" + `
IOMMU (Intel VT-d / AMD-Vi):
  Like an MMU but for device DMA access
  
  Problem: Devices do DMA using physical addresses
  In VM, device might corrupt host memory!
  
  IOMMU provides:
  1. DMA Remapping: Device DMA addresses → actual physical addresses
     Device thinks it's accessing GPA, IOMMU translates to HPA
  2. Interrupt Remapping: Prevents devices from injecting arbitrary interrupts
  3. Device Isolation: Each device can only access its assigned memory

  Without IOMMU:
    Device DMA → Bus → Physical Memory (any address!)
    
  With IOMMU:
    Device DMA → Bus → IOMMU → checks page table → Physical Memory
                                ↓ (if unauthorized)
                              FAULT (DMA blocked)

  Use cases:
  - Safe device passthrough to VMs
  - Device isolation (security)
  - Userspace drivers (DPDK, SPDK)
  - Preventing DMA attacks from malicious devices
` + "```" + ``,
					CodeExamples: `// Hardware virtualization concepts simulation
package main

import "fmt"

// Page table entry
type PageTableEntry struct {
    present  bool
    writable bool
    user     bool
    pfn      uint64 // Page Frame Number
    accessed bool
    dirty    bool
}

// Two-level address translation simulation
type GuestPageTable struct {
    entries map[uint64]PageTableEntry // GVA page → GPA page
}

type ExtendedPageTable struct {
    entries map[uint64]PageTableEntry // GPA page → HPA page
}

type VirtualMachine struct {
    id       int
    name     string
    guestPT  *GuestPageTable
    vcpuRegs [16]uint64
    vmExits  int
}

type Hypervisor struct {
    ept   *ExtendedPageTable
    vms   []*VirtualMachine
    vpids map[int]int // VM ID → VPID
}

func NewHypervisor() *Hypervisor {
    return &Hypervisor{
        ept:   &ExtendedPageTable{entries: make(map[uint64]PageTableEntry)},
        vpids: make(map[int]int),
    }
}

func (h *Hypervisor) CreateVM(name string) *VirtualMachine {
    vm := &VirtualMachine{
        id:   len(h.vms),
        name: name,
        guestPT: &GuestPageTable{
            entries: make(map[uint64]PageTableEntry),
        },
    }
    h.vms = append(h.vms, vm)
    h.vpids[vm.id] = vm.id + 1
    return vm
}

func (h *Hypervisor) MapGuestMemory(vm *VirtualMachine, gva, gpa, hpa uint64) {
    // Guest page table: GVA → GPA
    vm.guestPT.entries[gva] = PageTableEntry{
        present: true, writable: true, user: true, pfn: gpa,
    }
    // EPT: GPA → HPA (with VM isolation)
    eptKey := uint64(vm.id)<<48 | gpa
    h.ept.entries[eptKey] = PageTableEntry{
        present: true, writable: true, pfn: hpa,
    }
}

func (h *Hypervisor) TranslateAddress(vm *VirtualMachine, gva uint64) (uint64, bool, int) {
    walks := 0
    
    // Step 1: Guest PT walk (GVA → GPA)
    walks++
    gpte, ok := vm.guestPT.entries[gva]
    if !ok || !gpte.present {
        return 0, false, walks
    }
    gpa := gpte.pfn
    
    // Step 2: EPT walk (GPA → HPA)
    walks++
    eptKey := uint64(vm.id)<<48 | gpa
    epte, ok := h.ept.entries[eptKey]
    if !ok || !epte.present {
        return 0, false, walks
    }
    
    return epte.pfn, true, walks
}

// IOMMU simulation
type IOMMUEntry struct {
    deviceID  int
    gpa       uint64
    hpa       uint64
    readable  bool
    writable  bool
}

type IOMMU struct {
    entries []IOMMUEntry
    faults  int
}

func (iommu *IOMMU) AddMapping(deviceID int, gpa, hpa uint64, r, w bool) {
    iommu.entries = append(iommu.entries, IOMMUEntry{
        deviceID: deviceID, gpa: gpa, hpa: hpa,
        readable: r, writable: w,
    })
}

func (iommu *IOMMU) TranslateDMA(deviceID int, gpa uint64, write bool) (uint64, bool) {
    for _, e := range iommu.entries {
        if e.deviceID == deviceID && e.gpa == gpa {
            if write && !e.writable {
                iommu.faults++
                return 0, false
            }
            if !write && !e.readable {
                iommu.faults++
                return 0, false
            }
            return e.hpa, true
        }
    }
    iommu.faults++
    return 0, false
}

// VM Exit simulation
type VMExitReason int
const (
    ExitCPUID VMExitReason = iota
    ExitHLT
    ExitIO
    ExitCR
    ExitEPTViolation
    ExitExternalInt
)

func (r VMExitReason) String() string {
    return [...]string{"CPUID", "HLT", "I/O", "CR Access",
        "EPT Violation", "External Interrupt"}[r]
}

func simulateVMExit(vm *VirtualMachine, reason VMExitReason) {
    vm.vmExits++
    fmt.Printf("    VM Exit #%d: %s → hypervisor handles → VM Resume\n",
        vm.vmExits, reason)
}

func main() {
    fmt.Println("=== Hardware Virtualization Simulation ===")
    
    hv := NewHypervisor()
    
    // Create two VMs
    vm1 := hv.CreateVM("WebServer-VM")
    vm2 := hv.CreateVM("Database-VM")
    
    // Map memory for VM1
    // GVA 0x1000 → GPA 0x1000 → HPA 0x100000
    hv.MapGuestMemory(vm1, 0x1000, 0x1000, 0x100000)
    hv.MapGuestMemory(vm1, 0x2000, 0x2000, 0x200000)
    hv.MapGuestMemory(vm1, 0x3000, 0x3000, 0x300000)
    
    // Map memory for VM2 (different HPA, even with same GVA/GPA)
    hv.MapGuestMemory(vm2, 0x1000, 0x1000, 0x400000)
    hv.MapGuestMemory(vm2, 0x2000, 0x2000, 0x500000)
    
    fmt.Println("\n--- Two-Level Address Translation ---")
    testAddrs := []uint64{0x1000, 0x2000, 0x3000, 0x9999}
    
    for _, vm := range []*VirtualMachine{vm1, vm2} {
        fmt.Printf("\n%s (VPID=%d):\n", vm.name, hv.vpids[vm.id])
        for _, gva := range testAddrs {
            hpa, ok, walks := hv.TranslateAddress(vm, gva)
            if ok {
                fmt.Printf("  GVA 0x%04X → HPA 0x%06X (%d page walks)\n", gva, hpa, walks)
            } else {
                fmt.Printf("  GVA 0x%04X → PAGE FAULT (%d walks before fault)\n", gva, walks)
            }
        }
    }
    
    // VM isolation: same GVA maps to different HPA
    fmt.Println("\n--- Memory Isolation ---")
    hpa1, _, _ := hv.TranslateAddress(vm1, 0x1000)
    hpa2, _, _ := hv.TranslateAddress(vm2, 0x1000)
    fmt.Printf("  VM1 GVA 0x1000 → HPA 0x%06X\n", hpa1)
    fmt.Printf("  VM2 GVA 0x1000 → HPA 0x%06X\n", hpa2)
    fmt.Printf("  Same GVA, different HPA → VMs are isolated ✓\n")
    
    // VM Exits
    fmt.Println("\n--- VM Exit Simulation ---")
    fmt.Printf("  %s executing:\n", vm1.name)
    simulateVMExit(vm1, ExitCPUID)
    simulateVMExit(vm1, ExitIO)
    simulateVMExit(vm1, ExitExternalInt)
    fmt.Printf("  Total VM exits for %s: %d\n", vm1.name, vm1.vmExits)

    // IOMMU
    fmt.Println("\n--- IOMMU Simulation ---")
    iommu := &IOMMU{}
    
    // NIC (device 0) can access VM1's memory
    iommu.AddMapping(0, 0x1000, 0x100000, true, true)
    iommu.AddMapping(0, 0x2000, 0x200000, true, false) // read-only
    
    // Disk (device 1) can access VM2's memory
    iommu.AddMapping(1, 0x1000, 0x400000, true, true)
    
    dmaTests := []struct {
        dev   int
        devName string
        addr  uint64
        write bool
    }{
        {0, "NIC", 0x1000, false},  // NIC reads VM1 mem → OK
        {0, "NIC", 0x2000, true},   // NIC writes VM1 read-only → FAULT
        {0, "NIC", 0x9000, false},  // NIC reads unmapped → FAULT
        {1, "Disk", 0x1000, true},  // Disk writes VM2 mem → OK
        {1, "Disk", 0x2000, false}, // Disk reads unmapped → FAULT
    }
    
    for _, t := range dmaTests {
        hpa, ok := iommu.TranslateDMA(t.dev, t.addr, t.write)
        op := "READ"
        if t.write { op = "WRITE" }
        if ok {
            fmt.Printf("  %s %s GPA 0x%04X → HPA 0x%06X ✓\n", t.devName, op, t.addr, hpa)
        } else {
            fmt.Printf("  %s %s GPA 0x%04X → DMA FAULT (blocked!) ✗\n", t.devName, op, t.addr)
        }
    }
    fmt.Printf("  Total IOMMU faults: %d\n", iommu.faults)
}`,
				},
				{
					Title: "Side-Channel Attacks and Mitigations",
					Content: `Side-channel attacks exploit microarchitectural implementation details rather than software bugs. They represent a fundamental tension between performance optimization and security.

**Spectre and Meltdown (2018):**
` + "```" + `
Spectre Variant 1 (Bounds Check Bypass):
  Exploits speculative execution past array bounds check
  
  Vulnerable pseudo-code:
    if (x < array1_size) {         // bounds check
        y = array2[array1[x] * 256]; // speculative access
    }
  
  Attack:
  1. Train branch predictor: call with valid x many times
  2. Call with out-of-bounds x (e.g., x = secret_offset)
  3. Branch predictor predicts "taken" (based on training)
  4. CPU speculatively executes: reads array1[secret_offset]
  5. Uses secret value to index array2 → brings cache line
  6. After branch resolves: speculative results discarded
  7. BUT cache state remains! Attacker probes array2 timing
  8. Cache line that loads fast → reveals the secret byte

Spectre Variant 2 (Branch Target Injection):
  Exploits indirect branch predictor
  Attacker can influence which address CPU speculatively jumps to
  → Can execute arbitrary "gadgets" speculatively

Meltdown (Variant 3):
  Exploits out-of-order execution with delayed privilege check
  
  Attack:
  1. Load kernel memory address (forbidden for user code)
  2. CPU loads it anyway (out-of-order, before privilege check)
  3. Use loaded value as cache index (side channel)
  4. Privilege check eventually fires → exception
  5. But cache state reveals kernel data
  
  Fix: KPTI (Kernel Page Table Isolation)
    - Separate page tables for user and kernel mode
    - Kernel pages not mapped at all in user page table
    - Performance cost: ~5% (TLB flushes on syscalls)
    - Modern CPUs: Fixed in hardware (no Meltdown on newer chips)
` + "```" + `

**Cache Side Channels:**
` + "```" + `
Prime+Probe:
  1. PRIME: Fill cache set with attacker's lines
  2. Wait: Let victim execute
  3. PROBE: Measure time to access attacker's lines
  If slow → victim evicted attacker's line → victim accessed that set
  Works across VMs (shared LLC)

Flush+Reload:
  1. FLUSH: clflush on shared memory line
  2. Wait: Let victim execute
  3. RELOAD: Measure access time
  If fast → victim accessed the line (it's cached)
  Requires shared pages (e.g., shared libraries)

Evict+Time:
  1. Measure victim's execution time
  2. Evict specific cache lines
  3. Measure victim's execution time again
  Slower → victim needed those cache lines

Microarchitectural Data Sampling (MDS):
  Zombie Load, RIDL, Fallout
  Exploit speculative reads from internal CPU buffers
  (Line Fill Buffer, Store Buffer, Load Port)
  Can leak data across hyperthreads, security domains
  
  Mitigation: Flush buffers on context switch (VERW instruction)
  Performance cost: 3-8% depending on workload
` + "```" + `

**Hardware Security Features:**
` + "```" + `
Intel SGX (Software Guard Extensions):
  Creates encrypted enclaves in memory
  Even OS/hypervisor cannot read enclave data
  CPU encrypts/decrypts at memory controller
  
  ┌────────────────────────────────────┐
  │ Application                        │
  │  ┌────────────────────┐            │
  │  │ SGX Enclave         │ ← Encrypted in DRAM  │
  │  │ (secret data,       │   Decrypted only    │
  │  │  private keys,      │   inside CPU        │
  │  │  sensitive compute) │                     │
  │  └────────────────────┘            │
  └────────────────────────────────────┘
  
  Limitations:
  - Small enclave size (128-256MB)
  - Side-channel vulnerable (cache, branch timing)
  - Attestation infrastructure complexity

AMD SEV (Secure Encrypted Virtualization):
  Encrypts entire VM memory with per-VM keys
  Hypervisor cannot read VM memory
  SEV-ES: Also encrypts register state
  SEV-SNP: Adds integrity protection + attestation
  
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ VM1      │  │ VM2      │  │Hypervisor│
  │ Key: K1  │  │ Key: K2  │  │ Key: KH  │
  └──────────┘  └──────────┘  └──────────┘
  Each VM's memory encrypted with different key
  AMD Secure Processor manages keys
  Hardware memory controller handles encrypt/decrypt

ARM TrustZone:
  Divides processor into Secure and Normal worlds
  Hardware-enforced isolation at bus level
  Used for: Secure boot, key storage, DRM, payment
  
ARM CCA (Confidential Compute Architecture):
  Adds "Realm" world for confidential VMs
  Normal │ Secure │ Realm │ Root
  VMs can be isolated from hypervisor (like AMD SEV)
  Dynamic Realm creation without reboot

Intel TDX (Trust Domain Extensions):
  Similar to AMD SEV-SNP for Intel
  Hardware-isolated VMs (Trust Domains)
  Memory encryption + integrity + attestation
  Designed for confidential cloud computing
` + "```" + `

**Speculative Execution Mitigations:**
` + "```" + `
Software Mitigations:
  Retpoline (Return Trampoline):
    Replaces indirect jumps with return-based sequence
    Prevents Spectre v2 by defeating indirect branch predictor
    Performance cost: 1-5%
    
  LFENCE (Load Fence):
    Serializes instruction execution
    Prevents speculative loads past this point
    Used to mitigate Spectre v1 after bounds checks
    Performance cost: varies (only at sensitive points)

  Array index masking:
    index &= mask  // Ensure index stays in bounds
    Even speculative execution uses masked value
    
Hardware Mitigations (newer CPUs):
  IBRS (Indirect Branch Restricted Speculation)
  STIBP (Single Thread Indirect Branch Predictors)
  IBPB (Indirect Branch Predictor Barrier)
  Enhanced IBRS: Always-on, minimal overhead
  BHI mitigation: Clear branch history on transitions
  
  Modern CPUs (2019+):
  - Meltdown fixed in hardware (no KPTI needed)
  - Enhanced IBRS (Spectre v2 mitigation in silicon)
  - MDS mitigations improved
  - Still vulnerable to new variants (ongoing research)
  
Cost summary:
  Mitigation        │ Performance Impact │ Attack
  KPTI              │ 1-5%               │ Meltdown
  Retpoline         │ 1-5%               │ Spectre v2
  LFENCE barriers   │ 0-2%               │ Spectre v1
  MDS buffer flush  │ 3-8%               │ MDS/ZombieLoad
  SMT disable       │ up to 30%          │ Cross-HT attacks
  Total (worst case)│ 10-30%             │ All
` + "```" + ``,
					CodeExamples: `// Security architecture concepts simulation 
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "fmt"
    "io"
    "time"
)

// Simulated cache for side-channel demonstration
type SimCache struct {
    lines     map[int]bool // cached addresses
    accessLog []struct {
        addr   int
        hit    bool
        timeNS int
    }
}

func NewSimCache() *SimCache {
    return &SimCache{lines: make(map[int]bool)}
}

func (c *SimCache) Access(addr int) int {
    hit := c.lines[addr]
    latency := 200 // Cache miss: ~200 cycles
    if hit {
        latency = 4 // Cache hit: ~4 cycles
    }
    c.lines[addr] = true
    c.accessLog = append(c.accessLog, struct {
        addr   int
        hit    bool
        timeNS int
    }{addr, hit, latency})
    return latency
}

func (c *SimCache) Flush(addr int) {
    delete(c.lines, addr)
}

func (c *SimCache) FlushAll() {
    c.lines = make(map[int]bool)
}

// Demonstrate Flush+Reload concept
func flushReloadDemo() {
    fmt.Println("=== Flush+Reload Side Channel (Conceptual) ===")
    
    cache := NewSimCache()
    
    // Shared array (256 cache-line-sized entries)
    probeArray := make([]int, 256)
    for i := range probeArray { probeArray[i] = i * 64 } // 64-byte cache lines
    
    // Secret value we want to leak
    secret := byte(42)
    
    fmt.Println("\nStep 1: FLUSH - evict all probe array lines from cache")
    for i := 0; i < 256; i++ {
        cache.Flush(probeArray[i])
    }
    
    fmt.Println("Step 2: VICTIM - accesses probe_array[secret * 64]")
    // Victim speculatively accesses based on secret
    victimAccess := probeArray[int(secret)]
    cache.Access(victimAccess)
    
    fmt.Println("Step 3: RELOAD - measure access time for each entry")
    fmt.Println("  Addr    │ Time    │ Result")
    fmt.Println("  ────────┼─────────┼───────")
    
    leaked := -1
    for i := 0; i < 256; i++ {
        latency := cache.Access(probeArray[i])
        if latency < 10 { // Cache hit = fast = this was the secret
            leaked = i
            fmt.Printf("  [%3d]×64│ %3d cyc │ *** HIT (leaked value: %d)\n",
                i, latency, i)
        }
    }
    // Only show a few misses for brevity
    missSample := 0
    for i := 0; i < 256 && missSample < 3; i++ {
        if i != leaked {
            fmt.Printf("  [%3d]×64│ 200 cyc │ miss\n", i)
            missSample++
        }
    }
    fmt.Println("  ...      │         │ (252 more misses)")
    
    if leaked == int(secret) {
        fmt.Printf("\n  Secret value %d successfully leaked via cache timing!\n", secret)
    }
}

// Spectre v1 concept
func spectreV1Demo() {
    fmt.Println("\n=== Spectre Variant 1 (Bounds Check Bypass) ===")
    
    array1 := []byte{10, 20, 30, 40, 50} // Public array
    secretByte := byte(0xAB)              // Secret in memory after array1
    _ = secretByte
    
    fmt.Println("\nVulnerable pattern:")
    fmt.Println("  if (x < array1_size) {          // bounds check")
    fmt.Println("      y = array2[array1[x] * 256]; // data-dependent access")  
    fmt.Println("  }")
    fmt.Println()
    fmt.Println("Attack steps:")
    fmt.Println("  1. Train branch predictor: call with x=0,1,2,3 (valid)")
    fmt.Println("  2. Flush array1_size from cache (slow bounds check)")
    fmt.Println("  3. Call with x=out_of_bounds (points to secret)")
    fmt.Println("  4. Branch predictor says 'taken' → speculative load")
    fmt.Println("  5. Speculative: reads secret, uses as cache index")
    fmt.Println("  6. Branch resolves: mis-predicted → results discarded")
    fmt.Println("  7. Cache state persists → probe to find loaded line")
    
    fmt.Println("\nMitigation: Insert LFENCE after bounds check")
    fmt.Println("  if (x < array1_size) {")
    fmt.Println("      __lfence();  // Serializes execution")
    fmt.Println("      y = array2[array1[x] * 256];")
    fmt.Println("  }")
    fmt.Println("  Or: Use index masking")
    fmt.Println("  x &= mask;  // Keeps x in bounds even speculatively")
}

// Memory encryption simulation (like AMD SEV)
func memoryEncryptionDemo() {
    fmt.Println("\n=== Memory Encryption (AMD SEV concept) ===")
    
    // Generate per-VM keys
    type VMKey struct {
        vmName string
        key    []byte
    }
    
    vms := []VMKey{
        {"WebServer-VM", make([]byte, 32)},
        {"Database-VM", make([]byte, 32)},
    }
    
    for i := range vms {
        if _, err := io.ReadFull(rand.Reader, vms[i].key); err != nil {
            fmt.Printf("Error generating key: %v\n", err)
            return
        }
    }
    
    // Simulate encrypting VM memory
    plaintext := []byte("SECRET-DATA:credit_card=4111-1111-1111-1111")
    fmt.Printf("\nPlaintext data: %q\n", string(plaintext))
    
    for _, vm := range vms {
        // AES-GCM encryption (similar to what SEV uses)
        block, err := aes.NewCipher(vm.key)
        if err != nil { fmt.Println(err); continue }
        
        gcm, err := cipher.NewGCM(block)
        if err != nil { fmt.Println(err); continue }
        
        nonce := make([]byte, gcm.NonceSize())
        if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
            fmt.Println(err); continue
        }
        
        ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
        
        fmt.Printf("\n%s:\n", vm.vmName)
        fmt.Printf("  Key (first 8 bytes): %x...\n", vm.key[:8])
        fmt.Printf("  Encrypted in DRAM:   %x...\n", ciphertext[:24])
        fmt.Printf("  Length: %d bytes (plaintext: %d + overhead: %d)\n",
            len(ciphertext), len(plaintext), len(ciphertext)-len(plaintext))
        
        // Verify decryption works with correct key
        nonceSize := gcm.NonceSize()
        decrypted, err := gcm.Open(nil, ciphertext[:nonceSize], ciphertext[nonceSize:], nil)
        if err != nil {
            fmt.Printf("  Decryption: FAILED (%v)\n", err)
        } else {
            fmt.Printf("  Decryption with correct key: %q ✓\n", string(decrypted))
        }
    }
    
    // Show that hypervisor can't read VM memory
    fmt.Println("\nHypervisor sees only encrypted data in DRAM")
    fmt.Println("Each VM has unique key managed by hardware security processor")
    fmt.Println("Even memory dump reveals nothing without the key")
}

// Timing attack demonstration (constant-time comparison)
func timingAttackDemo() {
    fmt.Println("\n=== Timing Attack: Non-constant vs Constant-time ===")
    
    secret := []byte("SuperSecretPassword123")
    
    // VULNERABLE: early-exit comparison
    insecureCompare := func(a, b []byte) bool {
        if len(a) != len(b) { return false }
        for i := range a {
            if a[i] != b[i] {
                return false // Early exit reveals position of first mismatch!
            }
        }
        return true
    }
    
    // SECURE: constant-time comparison
    constantTimeCompare := func(a, b []byte) bool {
        if len(a) != len(b) { return false }
        result := byte(0)
        for i := range a {
            result |= a[i] ^ b[i] // No early exit
        }
        return result == 0
    }
    
    // Demonstrate timing difference
    tests := [][]byte{
        []byte("X"),                       // Wrong from byte 0
        []byte("SuperX"),                  // Wrong from byte 5
        []byte("SuperSecretPX"),           // Wrong from byte 12
        []byte("SuperSecretPassword12X"),  // Wrong from byte 21
        []byte("SuperSecretPassword123"),  // Correct
    }
    
    fmt.Printf("\n%-30s │ Insecure │ Constant-time\n", "Guess")
    fmt.Println("───────────────────────────────┼──────────┼──────────────")
    
    for _, guess := range tests {
        start := time.Now()
        for i := 0; i < 100000; i++ { insecureCompare(secret, guess) }
        insecTime := time.Since(start)
        
        start = time.Now()
        for i := 0; i < 100000; i++ { constantTimeCompare(secret, guess) }
        constTime := time.Since(start)
        
        display := string(guess)
        if len(display) > 25 { display = display[:25] + "..." }
        fmt.Printf("%-30s │ %8v │ %8v\n", display, insecTime, constTime)
    }
    
    fmt.Println("\nInsecure: time correlates with matching prefix length (leaks info)")
    fmt.Println("Constant-time: same duration regardless of input (safe)")
}

func main() {
    flushReloadDemo()
    spectreV1Demo()
    memoryEncryptionDemo()
    timingAttackDemo()
}`,
				},
			},
		},
	})
}
