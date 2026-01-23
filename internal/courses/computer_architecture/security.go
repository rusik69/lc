package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          206,
			Title:       "Hardware Security",
			Description: "Learn about hardware security threats: side-channel attacks, secure enclaves, hardware security modules, and processor vulnerabilities like Meltdown and Spectre.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Side-Channel Attacks",
					Content: `Side-channel attacks exploit physical characteristics of hardware implementations rather than algorithmic weaknesses.

**Attack Vectors:**

**Timing Attacks:**
- Measure execution time
- Infer secret information
- Example: Password comparison timing

**Power Analysis:**
- Measure power consumption
- Correlate with operations
- Simple Power Analysis (SPA): Direct observation
- Differential Power Analysis (DPA): Statistical analysis

**Electromagnetic (EM) Attacks:**
- Measure EM emissions
- Correlate with operations
- Non-invasive
- Can be done from distance

**Cache Attacks:**
- Exploit cache timing
- Infer memory access patterns
- Example: Prime+Probe, Flush+Reload

**Acoustic Attacks:**
- Measure acoustic emissions
- Correlate with operations
- Less common but possible

**Countermeasures:**

**Constant-Time Implementation:**
- Execution time independent of secret data
- Prevents timing attacks

**Power Analysis Resistance:**
- Masking: Randomize intermediate values
- Hiding: Reduce signal-to-noise ratio
- Balanced logic: Equal power consumption

**Cache Attack Mitigation:**
- Cache partitioning
- Constant-time memory access
- Cache flushing`,
					CodeExamples: `// Example: Timing attack vulnerability (VULNERABLE)
int compare_password(char *input, char *correct) {
    for (int i = 0; i < strlen(correct); i++) {
        if (input[i] != correct[i]) {
            return 0;  // Early return leaks timing information
        }
    }
    return 1;
}
// Attacker can measure time to determine password length
// and guess characters one by one

// Example: Constant-time comparison (SECURE)
int constant_time_compare(char *a, char *b, int len) {
    int result = 0;
    for (int i = 0; i < len; i++) {
        result |= a[i] ^ b[i];  // Always executes same operations
    }
    return result == 0;
}
// Execution time independent of data

// Example: Power analysis vulnerability
void aes_encrypt(uint8_t *plaintext, uint8_t *key, uint8_t *ciphertext) {
    // S-box lookup leaks power consumption
    uint8_t sbox_output = sbox[plaintext[0] ^ key[0]];
    // Power consumption correlates with sbox_output value
    // Attacker can recover key by analyzing power traces
}

// Example: Masking countermeasure
void aes_encrypt_masked(uint8_t *plaintext, uint8_t *key, uint8_t *ciphertext) {
    uint8_t mask = random_byte();
    uint8_t masked_plaintext = plaintext[0] ^ mask;
    uint8_t masked_key = key[0] ^ mask;
    
    uint8_t sbox_output = sbox[masked_plaintext ^ masked_key];
    ciphertext[0] = sbox_output ^ mask;
    // Power consumption randomized, harder to analyze
}

// Example: Cache timing attack (simplified)
void victim_function(uint8_t secret) {
    array[secret * 4096]++;  // Access depends on secret
}

void attacker_function() {
    // Flush cache
    flush_cache();
    
    // Trigger victim
    victim_function(secret);
    
    // Measure access time
    for (int i = 0; i < 256; i++) {
        uint64_t start = rdtsc();
        access(array[i * 4096]);
        uint64_t end = rdtsc();
        
        if (end - start < threshold) {
            // Cache hit - this was accessed by victim
            printf("Secret byte: %d\n", i);
        }
    }
}`,
				},
				{
					Title: "Secure Enclaves",
					Content: `Secure enclaves provide isolated execution environments protected from the rest of the system, including the operating system.

**Intel SGX (Software Guard Extensions):**

**Architecture:**
- Enclave: Protected memory region
- Enclave Page Cache (EPC): Encrypted memory
- CPU enforces access control
- Even OS cannot access enclave memory

**Features:**
- Remote attestation
- Sealed storage
- Enclave creation and destruction
- Memory encryption

**Use Cases:**
- DRM (Digital Rights Management)
- Secure computation
- Privacy-preserving analytics

**ARM TrustZone:**

**Architecture:**
- Two worlds: Secure and Normal
- CPU switches between worlds
- Secure world has access to both
- Normal world cannot access secure world

**Components:**
- TrustZone-aware peripherals
- Secure monitor
- Trusted Execution Environment (TEE)

**Use Cases:**
- Mobile payment systems
- Biometric authentication
- Secure boot

**AMD Memory Guard:**

**Architecture:**
- Secure Memory Encryption (SME)
- Secure Encrypted Virtualization (SEV)
- Memory encryption at hardware level

**Comparison:**

**SGX:**
- Fine-grained protection
- Application-level enclaves
- Requires code changes

**TrustZone:**
- Coarse-grained protection
- System-level separation
- Transparent to applications`,
					CodeExamples: `// Example: Intel SGX enclave (simplified)
// Enclave definition file (.edl)
enclave {
    trusted {
        public void enclave_function([in, size=len] uint8_t* data, size_t len);
    };
    untrusted {
        void ocall_print([in, string] const char* str);
    };
};

// Enclave code
void enclave_function(uint8_t* data, size_t len) {
    // This code runs in secure enclave
    // Memory is encrypted and protected
    // Even OS cannot access this memory
    
    // Process sensitive data
    for (size_t i = 0; i < len; i++) {
        data[i] = encrypt(data[i]);
    }
}

// Host code
int main() {
    // Create enclave
    sgx_enclave_id_t eid;
    sgx_create_enclave("enclave.signed.so", &eid);
    
    // Call enclave function
    uint8_t data[100];
    enclave_function(eid, data, sizeof(data));
    
    // Destroy enclave
    sgx_destroy_enclave(eid);
}

// Example: ARM TrustZone (simplified)
// Secure world code
void secure_world_function(void) {
    // This code runs in secure world
    // Has access to secure resources
    // Cannot be accessed from normal world
    
    // Access secure peripherals
    secure_gpio_set(1);
    
    // Return to normal world
    smc_return();
}

// Normal world code
void normal_world_function(void) {
    // This code runs in normal world
    // Cannot access secure world resources
    
    // Call secure world via SMC (Secure Monitor Call)
    smc_call(SECURE_FUNCTION_ID);
}

// Example: Remote attestation (SGX)
sgx_quote_t* get_quote(sgx_enclave_id_t eid) {
    sgx_report_t report;
    sgx_create_report(&report);
    
    // Quote proves enclave identity and integrity
    sgx_quote_t* quote = sgx_get_quote(&report);
    return quote;
}

// Verifier can check quote to verify:
// - Enclave is genuine Intel SGX enclave
// - Enclave code matches expected hash
// - Enclave is running on genuine Intel CPU`,
				},
				{
					Title: "Hardware Security Modules (HSM)",
					Content: `Hardware Security Modules are dedicated hardware devices designed to securely manage cryptographic keys and perform cryptographic operations.

**HSM Characteristics:**

**Physical Security:**
- Tamper-resistant/tamper-evident
- Physical attacks trigger key erasure
- FIPS 140-2 certified (various levels)

**Key Management:**
- Secure key generation
- Key storage (never leaves HSM)
- Key backup and recovery
- Key lifecycle management

**Cryptographic Operations:**
- Encryption/decryption
- Digital signatures
- Key derivation
- Random number generation

**HSM Types:**

**Network HSM:**
- Standalone device on network
- Multiple clients can use
- Examples: SafeNet Luna, Thales nShield

**PCIe HSM:**
- Card in server
- Direct connection
- Lower latency
- Examples: SafeNet PCIe, Thales PCIe

**USB HSM:**
- Portable device
- USB connection
- Lower performance
- Examples: YubiHSM, Nitrokey HSM

**Cloud HSM:**
- Managed service
- Cloud provider manages hardware
- Examples: AWS CloudHSM, Azure Dedicated HSM

**Use Cases:**
- Certificate Authority (CA) operations
- Database encryption
- Application signing
- Payment processing
- Blockchain key management`,
					CodeExamples: `// Example: HSM key generation (PKCS#11 API)
#include <pkcs11.h>

CK_SESSION_HANDLE session;
CK_OBJECT_HANDLE public_key, private_key;

// Initialize PKCS#11
CK_C_Initialize(NULL);

// Open session
CK_SLOT_ID slot_id = 0;
CK_SESSION_HANDLE session;
C_OpenSession(slot_id, CKF_SERIAL_SESSION, NULL, NULL, &session);

// Login
C_Login(session, CKU_USER, (CK_UTF8CHAR_PTR)"password", 8);

// Generate RSA key pair
CK_MECHANISM mechanism = {CKM_RSA_PKCS_KEY_PAIR_GEN, NULL, 0};
CK_ULONG modulus_bits = 2048;
CK_BYTE public_exponent[] = {0x01, 0x00, 0x01};  // 65537

CK_ATTRIBUTE public_key_template[] = {
    {CKA_MODULUS_BITS, &modulus_bits, sizeof(modulus_bits)},
    {CKA_PUBLIC_EXPONENT, public_exponent, sizeof(public_exponent)},
    {CKA_ENCRYPT, &true, sizeof(true)},
    {CKA_VERIFY, &true, sizeof(true)},
};

CK_ATTRIBUTE private_key_template[] = {
    {CKA_SIGN, &true, sizeof(true)},
    {CKA_DECRYPT, &true, sizeof(true)},
    {CKA_SENSITIVE, &true, sizeof(true)},  // Key never leaves HSM
    {CKA_EXTRACTABLE, &false, sizeof(false)},
};

C_GenerateKeyPair(session, &mechanism,
                  public_key_template, 4,
                  private_key_template, 4,
                  &public_key, &private_key);

// Example: Signing with HSM
CK_BYTE data[] = "Hello, World!";
CK_BYTE signature[256];

CK_MECHANISM sign_mechanism = {CKM_RSA_PKCS, NULL, 0};
C_SignInit(session, &sign_mechanism, private_key);
C_Sign(session, data, sizeof(data), signature, &signature_len);

// Private key never leaves HSM
// Signature computed inside HSM

// Example: Encryption with HSM
CK_BYTE plaintext[] = "Secret data";
CK_BYTE ciphertext[256];
CK_ULONG ciphertext_len = sizeof(ciphertext);

CK_MECHANISM encrypt_mechanism = {CKM_RSA_PKCS, NULL, 0};
C_EncryptInit(session, &encrypt_mechanism, public_key);
C_Encrypt(session, plaintext, sizeof(plaintext),
          ciphertext, &ciphertext_len);

// Example: HSM key backup (wrapped)
CK_BYTE wrapped_key[512];
CK_ULONG wrapped_key_len;

// Wrap key with master key (stored in HSM)
CK_MECHANISM wrap_mechanism = {CKM_AES_KEY_WRAP, NULL, 0};
C_WrapKey(session, &wrap_mechanism, master_key, private_key,
          wrapped_key, &wrapped_key_len);

// Wrapped key can be stored securely
// Can be unwrapped later with same master key`,
				},
				{
					Title: "Meltdown and Spectre Vulnerabilities",
					Content: `Meltdown and Spectre are critical processor vulnerabilities that exploit speculative execution to leak sensitive data.

**Speculative Execution:**

**What is Speculative Execution?**
- CPU executes instructions before knowing if they're needed
- Improves performance
- Results discarded if speculation wrong
- But side effects may remain

**Meltdown:**

**Vulnerability:**
- Exploits out-of-order execution
- Accesses kernel memory speculatively
- Data loaded into cache
- Cache timing reveals data

**Affected Systems:**
- Intel x86 processors
- Some ARM processors
- Not AMD (different architecture)

**Impact:**
- Read arbitrary kernel memory
- Break process isolation
- Steal passwords, keys

**Mitigation:**
- Kernel Page Table Isolation (KPTI)
- Separate page tables for kernel/user
- Performance overhead: 5-30%

**Spectre:**

**Variant 1: Bounds Check Bypass**
- Exploits conditional branch prediction
- Trains predictor to mispredict
- Accesses out-of-bounds data speculatively
- Cache timing reveals data

**Variant 2: Branch Target Injection**
- Exploits indirect branch prediction
- Poisons branch target buffer
- Redirects execution speculatively
- Accesses unauthorized data

**Affected Systems:**
- All modern processors
- Intel, AMD, ARM, IBM

**Impact:**
- Read memory from other processes
- Break sandboxing
- Steal secrets

**Mitigations:**
- Retpoline (return trampoline)
- Indirect Branch Restricted Speculation (IBRS)
- Software fixes (compiler barriers)
- Performance overhead: varies`,
					CodeExamples: `// Example: Meltdown attack (simplified concept)
uint8_t secret = kernel_memory[secret_address];  // This would normally fault
uint8_t value = array[secret * 4096];  // But speculatively executed
// Cache state reveals secret value

// Attack code (simplified):
uint64_t start, end;
uint8_t value;

// Flush cache
flush_cache();

// Access kernel memory (will fault, but speculatively executed)
try {
    value = kernel_memory[secret_address];
    array[value * 4096]++;  // Load into cache
} catch (fault) {
    // Fault handled, but cache already modified
}

// Measure access time
for (int i = 0; i < 256; i++) {
    start = rdtsc();
    access(array[i * 4096]);
    end = rdtsc();
    
    if (end - start < threshold) {
        // Cache hit - this is the secret value
        printf("Secret byte: %d\n", i);
    }
}

// Example: Spectre Variant 1
// Vulnerable code:
if (index < array_size) {
    value = array[index];  // Bounds check
    result = array2[value * 4096];  // Access
}

// Attack:
// 1. Train branch predictor: valid indices
for (int i = 0; i < 100; i++) {
    if (valid_index < array_size) {
        value = array[valid_index];
        result = array2[value * 4096];
    }
}

// 2. Mispredict with out-of-bounds index
if (malicious_index < array_size) {  // Predictor thinks true
    // Speculatively executes:
    value = array[malicious_index];  // Out of bounds!
    result = array2[value * 4096];  // Leaks data via cache
}
// Branch misprediction detected, but cache already modified

// Example: Mitigation - LFENCE (Load Fence)
if (index < array_size) {
    lfence();  // Serialize loads
    value = array[index];
    result = array2[value * 4096];
}
// Prevents speculative execution past lfence

// Example: Retpoline mitigation (Spectre Variant 2)
// Instead of indirect call:
call *rax  // Vulnerable to branch target injection

// Use retpoline:
call retpoline_setup
retpoline_target:
    pause
    jmp retpoline_target
retpoline_setup:
    mov %rax, (%rsp)
    ret  // Return to target, but speculative execution trapped in loop`,
				},
				{
					Title: "Trusted Execution Environments",
					Content: `Trusted Execution Environments (TEEs) provide secure, isolated execution environments for sensitive operations.

**TEE Characteristics:**

**Isolation:**
- Separate from normal execution environment
- Protected memory
- Secure storage
- Isolated execution

**Attestation:**
- Prove identity and integrity
- Remote attestation possible
- Cryptographic proofs

**Secure Storage:**
- Encrypted storage
- Key management
- Data protection

**TEE Implementations:**

**Intel SGX:**
- Application-level enclaves
- Fine-grained protection
- Remote attestation
- Sealed storage

**ARM TrustZone:**
- System-level separation
- Secure and normal worlds
- TrustZone-aware peripherals
- TEE implementations (OP-TEE, Trustonic)

**AMD Memory Guard:**
- Memory encryption
- Secure Encrypted Virtualization (SEV)
- VM-level protection

**RISC-V Keystone:**
- Open-source TEE
- Customizable
- Research and development

**TEE Use Cases:**

**Mobile:**
- Payment systems
- Biometric authentication
- DRM
- Secure boot

**Cloud:**
- Confidential computing
- Secure multi-party computation
- Privacy-preserving analytics

**IoT:**
- Device authentication
- Secure updates
- Key management`,
					CodeExamples: `// Example: TEE application structure
// Normal World (Rich OS)
void normal_world_app() {
    // Initialize TEE
    TEEC_Context context;
    TEEC_InitializeContext(NULL, &context);
    
    // Open session with TEE
    TEEC_Session session;
    TEEC_UUID uuid = TA_UUID;
    TEEC_OpenSession(&context, &session, &uuid,
                     TEEC_LOGIN_PUBLIC, NULL, NULL, NULL);
    
    // Call TEE function
    TEEC_Operation op;
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_VALUE_INPUT,
        TEEC_VALUE_OUTPUT,
        TEEC_NONE,
        TEEC_NONE);
    op.params[0].value.a = 10;
    op.params[0].value.b = 20;
    
    TEEC_InvokeCommand(&session, CMD_ADD, &op, NULL);
    
    int result = op.params[1].value.a;
    
    // Close session
    TEEC_CloseSession(&session);
    TEEC_FinalizeContext(&context);
}

// Trusted Application (TEE)
void TA_Add(uint32_t param_types, TEEC_Parameter params[4]) {
    uint32_t a = params[0].value.a;
    uint32_t b = params[0].value.b;
    
    // This code runs in secure world
    // Memory is protected
    // Cannot be accessed from normal world
    
    uint32_t result = a + b;
    params[1].value.a = result;
}

// Example: Secure storage
void store_secret_in_tee(uint8_t *secret, size_t len) {
    // Store in TEE secure storage
    // Encrypted and protected
    // Only accessible from TEE
    TEEC_SharedMemory mem;
    mem.size = len;
    mem.flags = TEEC_MEM_INPUT;
    TEEC_AllocateSharedMemory(&context, &mem);
    memcpy(mem.buffer, secret, len);
    
    TEEC_InvokeCommand(&session, CMD_STORE_SECRET, &op, NULL);
}

// Example: Remote attestation
void attest_tee() {
    // Generate attestation report
    TEEC_Operation op;
    TEEC_InvokeCommand(&session, CMD_GENERATE_ATTESTATION, &op, NULL);
    
    // Report contains:
    // - TEE identity
    // - TEE measurements (hash of code)
    // - Nonce (prevent replay)
    // - Signature (proves authenticity)
    
    // Send to verifier
    send_attestation_report(op.params[0].memref.buffer,
                           op.params[0].memref.size);
    
    // Verifier checks:
    // - Signature valid
    // - TEE is genuine
    // - Code matches expected hash
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
