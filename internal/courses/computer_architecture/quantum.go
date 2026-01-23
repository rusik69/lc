package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          205,
			Title:       "Quantum Computing Architecture",
			Description: "Explore quantum computing fundamentals: qubits, quantum gates, error correction, quantum processors, and hybrid quantum-classical systems.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Quantum Bits (Qubits) vs Classical Bits",
					Content: `Quantum computing uses quantum mechanical phenomena to perform computation. Understanding qubits is fundamental to quantum architecture.

**Classical Bits:**
- Two states: 0 or 1
- Deterministic: Always in one state
- Can be copied perfectly
- Measurement doesn't change state
- Independent: State of one bit doesn't affect others

**Quantum Bits (Qubits):**
- Superposition: Can be in state |0⟩, |1⟩, or both simultaneously
- Probability amplitudes: α|0⟩ + β|1⟩ where |α|² + |β|² = 1
- Measurement collapses to |0⟩ or |1⟩ probabilistically
- Cannot be copied (No-Cloning Theorem)
- Entanglement: Qubits can be correlated

**Qubit States:**

**Computational Basis:**
- |0⟩ = [1, 0]ᵀ (classical 0)
- |1⟩ = [0, 1]ᵀ (classical 1)

**Superposition States:**
- |+⟩ = (|0⟩ + |1⟩)/√2 (equal superposition)
- |-⟩ = (|0⟩ - |1⟩)/√2 (equal superposition, different phase)

**Bloch Sphere Representation:**
- Qubit state represented as point on sphere
- |0⟩ at north pole, |1⟩ at south pole
- Superposition states on equator
- Phase represented by angle

**Key Quantum Properties:**

**Superposition:**
- Qubit exists in multiple states simultaneously
- Enables parallel computation
- Lost on measurement

**Entanglement:**
- Qubits become correlated
- Measurement of one affects the other
- Enables quantum algorithms

**Interference:**
- Probability amplitudes can cancel or reinforce
- Used in quantum algorithms`,
					CodeExamples: `# Example: Qubit representation (using Qiskit)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Single qubit in superposition
qc = QuantumCircuit(1, 1)
qc.h(0)  # Apply Hadamard gate: |0⟩ → (|0⟩ + |1⟩)/√2
qc.measure(0, 0)

# Two qubits
qr = QuantumRegister(2)
cr = ClassicalRegister(2)
qc = QuantumCircuit(qr, cr)

# Create Bell state (entangled)
qc.h(0)        # |0⟩ → (|0⟩ + |1⟩)/√2
qc.cx(0, 1)    # CNOT: entangle qubits
# Result: (|00⟩ + |11⟩)/√2

# Example: Qubit states
# |0⟩ = [1, 0]ᵀ
# |1⟩ = [0, 1]ᵀ
# |+⟩ = [1/√2, 1/√2]ᵀ
# |-⟩ = [1/√2, -1/√2]ᵀ

# Example: Measurement
# Qubit in state: α|0⟩ + β|1⟩
# Probability of measuring |0⟩: |α|²
# Probability of measuring |1⟩: |β|²
# After measurement, qubit collapses to measured state

# Example: No-Cloning Theorem
# Cannot create: |ψ⟩ ⊗ |ψ⟩ from |ψ⟩ ⊗ |0⟩
# Quantum information cannot be copied perfectly
# Important for quantum error correction`,
				},
				{
					Title: "Quantum Gates and Circuits",
					Content: `Quantum gates manipulate qubit states. Understanding quantum gates is essential for building quantum algorithms.

**Single-Qubit Gates:**

**Pauli Gates:**
- **X (NOT)**: |0⟩ ↔ |1⟩
- **Y**: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
- **Z (Phase)**: |0⟩ → |0⟩, |1⟩ → -|1⟩

**Hadamard Gate (H):**
- Creates superposition: |0⟩ → (|0⟩ + |1⟩)/√2
- Essential for quantum algorithms
- Self-inverse: H² = I

**Rotation Gates:**
- **Rx(θ)**: Rotation around X-axis
- **Ry(θ)**: Rotation around Y-axis
- **Rz(θ)**: Rotation around Z-axis (phase rotation)

**Multi-Qubit Gates:**

**CNOT (Controlled-NOT):**
- Two-qubit gate
- Flips target if control is |1⟩
- Creates entanglement
- |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩

**Toffoli (CCNOT):**
- Three-qubit gate
- Universal for classical computation
- Flips target if both controls are |1⟩

**SWAP:**
- Exchanges two qubits
- |01⟩ ↔ |10⟩

**Quantum Circuits:**
- Sequence of gates applied to qubits
- Read left to right
- Measurement at end
- Reversible (except measurement)`,
					CodeExamples: `# Example: Quantum gates (Qiskit)
from qiskit import QuantumCircuit
import numpy as np

# Single-qubit gates
qc = QuantumCircuit(1)
qc.x(0)    # Pauli-X (NOT)
qc.y(0)    # Pauli-Y
qc.z(0)    # Pauli-Z (phase flip)
qc.h(0)    # Hadamard (superposition)
qc.ry(np.pi/4, 0)  # Rotation around Y-axis

# Two-qubit gates
qc = QuantumCircuit(2)
qc.cx(0, 1)  # CNOT: control=0, target=1
qc.swap(0, 1)  # SWAP qubits

# Three-qubit gates
qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)  # Toffoli: controls=0,1, target=2

# Example: Bell state creation
qc = QuantumCircuit(2, 2)
qc.h(0)        # |0⟩ → (|0⟩ + |1⟩)/√2
qc.cx(0, 1)    # Entangle: (|00⟩ + |11⟩)/√2
qc.measure_all()

# Example: Quantum teleportation circuit
qc = QuantumCircuit(3, 2)
# Alice's qubit (to teleport)
qc.h(0)
qc.ry(np.pi/4, 0)
# Bell pair (shared between Alice and Bob)
qc.h(1)
qc.cx(1, 2)
# Alice's operations
qc.cx(0, 1)
qc.h(0)
qc.measure(0, 0)
qc.measure(1, 1)
# Bob's operations (classical communication needed)
qc.cx(1, 2)
qc.cz(0, 2)
# Qubit 2 now in state of qubit 0

# Example: Gate matrices
# X gate: [[0, 1], [1, 0]]
# Y gate: [[0, -i], [i, 0]]
# Z gate: [[1, 0], [0, -1]]
# H gate: (1/√2)[[1, 1], [1, -1]]
# CNOT: [[1, 0, 0, 0],
#        [0, 1, 0, 0],
#        [0, 0, 0, 1],
#        [0, 0, 1, 0]]`,
				},
				{
					Title: "Quantum Error Correction",
					Content: `Quantum systems are extremely sensitive to errors. Quantum error correction is essential for building reliable quantum computers.

**Sources of Quantum Errors:**

**Decoherence:**
- Interaction with environment
- Loss of quantum state
- Main source of errors
- Timescale: microseconds to milliseconds

**Gate Errors:**
- Imperfect gate operations
- Systematic or random
- Typically 0.1-1% error rate

**Measurement Errors:**
- Imperfect readout
- Can be corrected with repetition

**Error Types:**

**Bit Flip:**
- |0⟩ ↔ |1⟩
- Analogous to classical bit errors
- Corrected with repetition codes

**Phase Flip:**
- |+⟩ ↔ |-⟩
- Quantum-specific error
- Requires quantum error correction

**General Errors:**
- Combination of bit and phase flips
- Most realistic scenario

**Quantum Error Correction Codes:**

**3-Qubit Repetition Code:**
- Encode: |0⟩ → |000⟩, |1⟩ → |111⟩
- Detects and corrects single bit flips
- Cannot correct phase flips

**Shor Code (9 qubits):**
- Corrects both bit and phase flips
- Encodes 1 logical qubit in 9 physical qubits
- First practical quantum error correction code

**Surface Code:**
- 2D lattice of qubits
- Scalable error correction
- Used in many quantum computers
- High threshold (~1% error rate)

**Error Correction Process:**
1. Encode logical qubit in multiple physical qubits
2. Perform syndrome measurement (detect errors)
3. Classically decode syndrome
4. Apply correction operations`,
					CodeExamples: `# Example: 3-qubit repetition code
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Encoding
qr = QuantumRegister(3)
cr = ClassicalRegister(3)
qc = QuantumCircuit(qr, cr)

# Encode |0⟩ as |000⟩
qc.cx(0, 1)  # Copy to qubit 1
qc.cx(0, 2)  # Copy to qubit 2
# |0⟩ → |000⟩

# Error detection (syndrome measurement)
# Add ancilla qubits for syndrome
ancilla = QuantumRegister(2)
qc.add_register(ancilla)

# Measure parity: qubits 0,1 and qubits 1,2
qc.cx(0, 3)  # Ancilla 0: parity of 0,1
qc.cx(1, 3)
qc.cx(1, 4)  # Ancilla 1: parity of 1,2
qc.cx(2, 4)

# Measure ancillas
syndrome = ClassicalRegister(2)
qc.add_register(syndrome)
qc.measure(3, 0)  # Syndrome bit 0
qc.measure(4, 1)  # Syndrome bit 1

# Syndrome interpretation:
# 00: No error
# 01: Error on qubit 2
# 10: Error on qubit 0
# 11: Error on qubit 1

# Correction (classical feedback needed)
# Apply X gate to erroneous qubit based on syndrome

# Example: Shor code (simplified)
# Encodes 1 logical qubit in 9 physical qubits
# Corrects arbitrary single-qubit errors

# Encoding:
# |0⟩ → (|000⟩ + |111⟩) ⊗ (|000⟩ + |111⟩) ⊗ (|000⟩ + |111⟩) / 2√2
# |1⟩ → (|000⟩ - |111⟩) ⊗ (|000⟩ - |111⟩) ⊗ (|000⟩ - |111⟩) / 2√2

# Error detection:
# Measure stabilizers (operators that don't change code space)
# Identify error from syndrome

# Example: Surface code
# 2D lattice of data and ancilla qubits
# Measure stabilizers on plaquettes
# Errors create chains of defects
# Minimum-weight matching for correction

# Error threshold:
# If physical error rate < threshold, logical error rate decreases
# Surface code threshold: ~1%`,
				},
				{
					Title: "Quantum Processor Architectures",
					Content: `Different physical implementations of quantum processors have different architectures and characteristics.

**Superconducting Qubits:**

**Transmon Qubits:**
- Most common superconducting qubit
- Josephson junction + capacitor
- Frequency: 4-6 GHz
- Coherence time: 10-100 microseconds
- Used by: IBM, Google, Rigetti

**Architecture:**
- Qubits on 2D chip
- Coupled via resonators
- Microwave control
- Cryogenic operation (10-20 mK)

**Trapped Ion Qubits:**

**Ion Traps:**
- Ions trapped in electromagnetic fields
- Qubit: internal energy levels
- Coherence time: seconds to minutes
- High fidelity gates
- Used by: IonQ, Honeywell

**Architecture:**
- Linear or 2D ion chains
- Laser control
- All-to-all connectivity possible
- Room temperature or cryogenic

**Photonic Quantum Computing:**

**Photon Qubits:**
- Qubit: photon polarization or path
- Very long coherence (speed of light)
- Room temperature operation
- Used by: Xanadu, PsiQuantum

**Architecture:**
- Optical circuits
- Beam splitters, phase shifters
- Single-photon sources and detectors
- Linear optical quantum computing

**Topological Qubits:**

**Anyons:**
- Non-Abelian anyons for qubits
- Topological protection
- Very long coherence (theoretical)
- Used by: Microsoft (research)

**Architecture:**
- 2D electron systems
- Majorana fermions
- Topological protection from errors

**Comparison:**

**Superconducting:**
- Pros: Mature, scalable, fast gates
- Cons: Short coherence, cryogenic

**Trapped Ions:**
- Pros: Long coherence, high fidelity
- Cons: Slow gates, scalability challenges

**Photonic:**
- Pros: Room temperature, long coherence
- Cons: Probabilistic gates, scalability`,
					CodeExamples: `# Example: Superconducting qubit control
# Qubit frequency: 5 GHz
# Control: Microwave pulses
# Readout: Dispersive readout

# Gate operations:
# X gate: π-pulse at qubit frequency
# Y gate: π-pulse with 90° phase shift
# Z gate: Virtual Z (phase accumulation)
# Two-qubit: Cross-resonance or parametric gates

# Example: Trapped ion control
# Qubit: Hyperfine levels (ground state)
# Control: Laser pulses
# Gates: Single-qubit (Raman transitions)
#        Two-qubit (Mølmer-Sørensen gate)

# Ion chain:
# Ions: [Ca⁺, Ca⁺, Ca⁺, ...]
# Qubits: Internal states of each ion
# Coupling: Collective motion (phonons)

# Example: Photonic quantum computing
# Qubit: Photon path or polarization
# Gates: Beam splitters, phase shifters
# Measurement: Single-photon detectors

# Linear optical quantum computing:
# - Probabilistic gates (success probability < 1)
# - Requires feed-forward
# - Cluster states for deterministic computation

# Example: Quantum processor layout
# Superconducting (IBM):
# - 2D grid of qubits
# - Nearest-neighbor coupling
# - Readout resonators
# - Control lines

# Trapped ion (IonQ):
# - Linear chain of ions
# - All-to-all connectivity
# - Laser addressing
# - Fluorescence readout

# Comparison table:
# Architecture    | Coherence | Gate Time | Temp      | Scalability
# Superconducting| 10-100 μs | 10-100 ns | 10-20 mK  | High
# Trapped Ion    | 1-100 s   | 1-100 μs  | Room/Cryo | Medium
# Photonic       | ∞         | ps        | Room      | Medium
# Topological    | Very long | TBD       | Cryo      | Research`,
				},
				{
					Title: "Quantum-Classical Hybrid Systems",
					Content: `Most practical quantum computing applications use hybrid quantum-classical systems, combining quantum and classical processors.

**Hybrid Architecture:**

**Classical Computer:**
- Control quantum processor
- Pre-process data
- Post-process results
- Optimize parameters

**Quantum Processor:**
- Execute quantum algorithms
- Generate quantum states
- Perform measurements

**Communication:**
- Classical control signals
- Quantum measurement results
- Feedback loops

**Hybrid Algorithms:**

**Variational Quantum Eigensolver (VQE):**
- Find ground state energy
- Quantum: Prepare ansatz state
- Classical: Optimize parameters
- Iterative optimization

**Quantum Approximate Optimization Algorithm (QAOA):**
- Solve optimization problems
- Quantum: Prepare parameterized state
- Classical: Optimize parameters
- Approximate solutions

**Quantum Machine Learning:**
- Quantum neural networks
- Quantum feature maps
- Hybrid training
- Classical optimization

**Quantum Error Mitigation:**
- Post-process measurement results
- Correct for errors classically
- No error correction overhead
- Limited error reduction

**Use Cases:**

**Chemistry:**
- Molecular simulation
- Drug discovery
- Material design

**Optimization:**
- Logistics
- Finance
- Machine learning

**Cryptography:**
- Quantum key distribution
- Post-quantum cryptography`,
					CodeExamples: `# Example: VQE (Variational Quantum Eigensolver)
from qiskit import QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.quantum_info import SparsePauliOp

# Define Hamiltonian (molecular Hamiltonian)
hamiltonian = SparsePauliOp.from_list([
    ("II", 0.5),
    ("IZ", 0.3),
    ("ZI", 0.2),
    ("ZZ", 0.1),
])

# Ansatz circuit (parameterized)
def ansatz_circuit(params):
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    return qc

# VQE algorithm
optimizer = SPSA(maxiter=100)
vqe = VQE(ansatz_circuit, optimizer=optimizer)

# Execute
result = vqe.compute_minimum_eigenvalue(hamiltonian)
ground_state_energy = result.eigenvalue

# Example: QAOA (Quantum Approximate Optimization Algorithm)
def qaoa_circuit(params, problem_graph):
    qc = QuantumCircuit(len(problem_graph))
    
    # Initial state: superposition
    for i in range(len(problem_graph)):
        qc.h(i)
    
    # Problem and mixer layers
    for p in range(len(params) // 2):
        # Problem layer
        for edge in problem_graph:
            qc.rz(params[2*p], edge[0])
            qc.rz(params[2*p], edge[1])
            qc.cx(edge[0], edge[1])
            qc.rz(params[2*p], edge[1])
            qc.cx(edge[0], edge[1])
        
        # Mixer layer
        for i in range(len(problem_graph)):
            qc.rx(params[2*p+1], i)
    
    return qc

# Example: Quantum-classical feedback loop
def hybrid_optimization():
    params = initialize_params()
    
    for iteration in range(max_iterations):
        # Quantum: Prepare state with current params
        qc = prepare_quantum_state(params)
        results = execute_quantum(qc, shots=1000)
        
        # Classical: Process results
        cost = compute_cost(results)
        
        # Classical: Update parameters
        params = optimizer.update(params, cost)
        
        if converged(cost):
            break
    
    return params

# Example: Error mitigation
def error_mitigation(results):
    # Zero-noise extrapolation
    # Run at different noise levels
    results_low = run_at_noise_level(0.5)
    results_high = run_at_noise_level(1.0)
    
    # Extrapolate to zero noise
    results_zero = extrapolate(results_low, results_high)
    
    return results_zero`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
