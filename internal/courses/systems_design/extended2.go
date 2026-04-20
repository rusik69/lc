package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          2418,
			Title:       "Distributed Systems Fundamentals",
			Description: "Master CAP theorem, consensus algorithms, distributed transactions, vector clocks, CRDTs, and fault tolerance patterns.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "CAP Theorem Consensus and Distributed Transactions",
					Content: `Distributed systems present unique challenges around consistency, availability, and partition tolerance that don't exist in single-machine systems.

**CAP Theorem:**

The CAP theorem states that a distributed system can only guarantee two of three properties simultaneously:

Consistency (C): Every read receives the most recent write
Availability (A): Every request receives a response (not error)
Partition Tolerance (P): System continues despite network partitions

Since network partitions are inevitable, the real choice is between CP and AP:

CP Systems: Choose consistency over availability during partitions
  Examples: ZooKeeper, etcd, HBase, MongoDB (default), Redis Cluster
  Use when: Financial systems, inventory, leader election
  
AP Systems: Choose availability over consistency during partitions  
  Examples: Cassandra, DynamoDB, CouchDB, Riak
  Use when: Social media feeds, analytics, caching

PACELC Extension:
  if Partition → choose Availability or Consistency
  else (normal operation) → choose Latency or Consistency
  
  PA/EL: Available during partition, low latency normally (Cassandra)
  PA/EC: Available during partition, consistent normally (Cosmos DB)
  PC/EL: Consistent during partition, low latency normally (rare)
  PC/EC: Consistent always (traditional RDBMS, ZooKeeper)

**Consistency Models:**

Strong Consistency:
  All reads see the latest write
  Linearizability — operations appear atomic and ordered
  Implementation: Raft, Paxos, two-phase commit
  Cost: Higher latency, lower availability
  
Sequential Consistency:
  Operations from each process appear in program order
  Global order exists but may not reflect real-time ordering
  
Causal Consistency:
  Causally related operations appear in order
  Concurrent operations may appear in different orders
  Implementation: Vector clocks, version vectors
  
Eventual Consistency:
  All replicas converge to same value eventually
  Window of inconsistency during updates
  Implementation: Anti-entropy, read repair, hinted handoff
  Conflict resolution: Last-write-wins, vector clocks, CRDTs
  
Read-Your-Writes:
  A process always sees its own writes
  Implementation: Sticky sessions, read from leader after write

**Consensus Algorithms:**

Raft:
  Leader election → log replication → safety
  
  Leader Election:
    Nodes start as followers
    Timeout → become candidate → request votes
    Majority votes → become leader
    Term numbers prevent split brain
    
  Log Replication:
    Leader receives client requests
    Appends to local log
    Sends AppendEntries RPCs to followers
    Commits after majority acknowledgment
    Followers apply committed entries
    
  Safety:
    Election restriction: candidate must have all committed entries
    Leader completeness: committed entry exists in all future leaders
    
  Failure handling:
    Leader failure: new election
    Follower failure: leader retries
    Network partition: majority side continues

Paxos:
  Proposer → Acceptor → Learner
  
  Phase 1 (Prepare):
    Proposer sends prepare(n) to acceptors
    Acceptors respond with highest accepted proposal
    
  Phase 2 (Accept):
    Proposer sends accept(n, value) to acceptors
    Acceptors accept if no higher prepare seen
    
  Multi-Paxos: Optimization for repeated consensus

**Distributed Transactions:**

Two-Phase Commit (2PC):
  Phase 1 (Prepare/Vote):
    Coordinator asks all participants to prepare
    Participants acquire locks, write to WAL, vote yes/no
    
  Phase 2 (Commit/Abort):
    If all vote yes: coordinator sends commit
    If any votes no: coordinator sends abort
    Participants apply and release locks
    
  Problems:
    Blocking: Coordinator failure blocks all
    Single point of failure: Coordinator
    Network partition can cause inconsistency

Three-Phase Commit (3PC):
  Adds pre-commit phase between prepare and commit
  Non-blocking under certain failure models
  Still vulnerable to network partitions

Saga Pattern:
  Sequence of local transactions with compensating transactions
  
  Choreography:
    Services publish events
    Other services react
    Compensation events for rollback
    No central coordinator
    
  Orchestration:
    Central saga orchestrator
    Directs each step
    Handles compensation on failure
    Easier to understand and debug

  Compensation:
    Each step has a compensating action
    Must be idempotent
    Semantic undo (not physical)
    
  Example — Order Processing:
    1. Create Order → Cancel Order
    2. Reserve Inventory → Release Inventory
    3. Process Payment → Refund Payment
    4. Ship Order → Return Shipment

**Vector Clocks and Causality:**

Lamport Timestamps:
  Simple logical clocks
  Each process maintains counter
  On send: increment and attach
  On receive: max(local, received) + 1
  Establishes partial order
  Cannot detect concurrent events

Vector Clocks:
  Array of counters, one per process
  VC[i] = number of events at process i
  
  Rules:
    Local event: increment own counter
    Send: increment own counter, attach VC
    Receive: merge (element-wise max), increment own
    
  Comparison:
    VC1 < VC2 if all elements ≤ and at least one <
    VC1 || VC2 if neither < nor > (concurrent)
    
  Use: Detect causal ordering and conflicts in replicated data

Version Vectors:
  Like vector clocks but for replicated objects
  Track which replica has seen which updates
  Used in Dynamo-style systems

**CRDTs (Conflict-free Replicated Data Types):**

Operation-based (CmRDTs):
  Replicate operations
  Require reliable, ordered delivery
  
State-based (CvRDTs):
  Replicate full state
  Merge function must be commutative, associative, idempotent
  
Common CRDTs:
  G-Counter: Grow-only counter (array of per-node counts)
  PN-Counter: Positive-negative counter (two G-Counters)
  G-Set: Grow-only set (union merge)
  2P-Set: Two-phase set (add set + remove set)
  OR-Set: Observed-remove set (unique tags per add)
  LWW-Register: Last-writer-wins register
  LWW-Element-Set: LWW per element
  Sequence CRDT: Ordered text editing (RGA, LSEQ, Logoot)

**Failure Detection:**

Heartbeat:
  Periodic pings between nodes
  Timeout → suspect failure
  Simple but slow detection

Gossip-based (Phi Accrual):
  Track heartbeat intervals
  Calculate suspicion level (phi)
  Adaptive threshold
  Used in Cassandra, Akka

SWIM Protocol:
  Ping random node
  If no response → indirect ping through k others
  If still no response → suspect
  Disseminate membership via gossip`,
					CodeExamples: `# Distributed Systems Implementation Examples

import time
import hashlib
import threading
import random
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# ============================================================
# Vector Clock Implementation
# ============================================================

class VectorClock:
    """Vector clock for causality tracking."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.clock: Dict[str, int] = {node_id: 0}
    
    def increment(self):
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
    
    def update(self, other: 'VectorClock'):
        for node, count in other.clock.items():
            self.clock[node] = max(self.clock.get(node, 0), count)
        self.increment()
    
    def __le__(self, other: 'VectorClock') -> bool:
        for node in self.clock:
            if self.clock[node] > other.clock.get(node, 0):
                return False
        return True
    
    def __lt__(self, other: 'VectorClock') -> bool:
        return self <= other and self.clock != other.clock
    
    def is_concurrent(self, other: 'VectorClock') -> bool:
        return not (self <= other) and not (other <= self)
    
    def merge(self, other: 'VectorClock') -> 'VectorClock':
        merged = VectorClock(self.node_id)
        all_nodes = set(self.clock.keys()) | set(other.clock.keys())
        for node in all_nodes:
            merged.clock[node] = max(
                self.clock.get(node, 0),
                other.clock.get(node, 0))
        return merged
    
    def copy(self) -> 'VectorClock':
        vc = VectorClock(self.node_id)
        vc.clock = dict(self.clock)
        return vc
    
    def __repr__(self):
        return f"VC({dict(sorted(self.clock.items()))})"


# ============================================================
# CRDT Implementations
# ============================================================

class GCounter:
    """Grow-only counter CRDT."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: Dict[str, int] = {}
    
    def increment(self, amount: int = 1):
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount
    
    @property
    def value(self) -> int:
        return sum(self.counts.values())
    
    def merge(self, other: 'GCounter') -> 'GCounter':
        result = GCounter(self.node_id)
        all_nodes = set(self.counts.keys()) | set(other.counts.keys())
        for node in all_nodes:
            result.counts[node] = max(
                self.counts.get(node, 0),
                other.counts.get(node, 0))
        return result


class PNCounter:
    """Positive-negative counter CRDT."""
    
    def __init__(self, node_id: str):
        self.positive = GCounter(node_id)
        self.negative = GCounter(node_id)
    
    def increment(self, amount: int = 1):
        self.positive.increment(amount)
    
    def decrement(self, amount: int = 1):
        self.negative.increment(amount)
    
    @property
    def value(self) -> int:
        return self.positive.value - self.negative.value
    
    def merge(self, other: 'PNCounter') -> 'PNCounter':
        result = PNCounter(self.positive.node_id)
        result.positive = self.positive.merge(other.positive)
        result.negative = self.negative.merge(other.negative)
        return result


class GSet:
    """Grow-only set CRDT."""
    
    def __init__(self):
        self.elements: Set[Any] = set()
    
    def add(self, element: Any):
        self.elements.add(element)
    
    def contains(self, element: Any) -> bool:
        return element in self.elements
    
    def merge(self, other: 'GSet') -> 'GSet':
        result = GSet()
        result.elements = self.elements | other.elements
        return result
    
    @property
    def value(self) -> Set[Any]:
        return frozenset(self.elements)


class ORSet:
    """Observed-Remove Set CRDT."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self._counter = 0
        self.elements: Dict[Any, Set[Tuple[str, int]]] = {}
        self.tombstones: Dict[Any, Set[Tuple[str, int]]] = {}
    
    def _unique_tag(self) -> Tuple[str, int]:
        self._counter += 1
        return (self.node_id, self._counter)
    
    def add(self, element: Any):
        tag = self._unique_tag()
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)
    
    def remove(self, element: Any):
        if element in self.elements:
            if element not in self.tombstones:
                self.tombstones[element] = set()
            self.tombstones[element].update(self.elements[element])
            del self.elements[element]
    
    def contains(self, element: Any) -> bool:
        if element not in self.elements:
            return False
        alive = self.elements[element] - self.tombstones.get(element, set())
        return len(alive) > 0
    
    @property
    def value(self) -> Set[Any]:
        return {e for e in self.elements if self.contains(e)}
    
    def merge(self, other: 'ORSet') -> 'ORSet':
        result = ORSet(self.node_id)
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        for elem in all_elements:
            tags = set()
            tags.update(self.elements.get(elem, set()))
            tags.update(other.elements.get(elem, set()))
            if tags:
                result.elements[elem] = tags
        all_tombs = set(self.tombstones.keys()) | set(other.tombstones.keys())
        for elem in all_tombs:
            tags = set()
            tags.update(self.tombstones.get(elem, set()))
            tags.update(other.tombstones.get(elem, set()))
            if tags:
                result.tombstones[elem] = tags
        return result


class LWWRegister:
    """Last-Writer-Wins Register CRDT."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self._value: Any = None
        self._timestamp: float = 0
    
    def set(self, value: Any, timestamp: float = None):
        ts = timestamp or time.time()
        if ts >= self._timestamp:
            self._value = value
            self._timestamp = ts
    
    @property
    def value(self) -> Any:
        return self._value
    
    def merge(self, other: 'LWWRegister') -> 'LWWRegister':
        result = LWWRegister(self.node_id)
        if self._timestamp >= other._timestamp:
            result._value = self._value
            result._timestamp = self._timestamp
        else:
            result._value = other._value
            result._timestamp = other._timestamp
        return result


# ============================================================
# Raft Consensus (Simplified)
# ============================================================

class RaftState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    term: int
    command: str
    index: int


class RaftNode:
    """Simplified Raft consensus node."""
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.state = RaftState.FOLLOWER
        self.commit_index = -1
        self.last_applied = -1
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Election
        self.votes_received: Set[str] = set()
        self.leader_id: Optional[str] = None
    
    @property
    def last_log_index(self) -> int:
        return len(self.log) - 1
    
    @property
    def last_log_term(self) -> int:
        if self.log:
            return self.log[-1].term
        return 0
    
    def start_election(self):
        self.state = RaftState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.leader_id = None
    
    def receive_vote(self, voter_id: str, term: int, granted: bool):
        if term > self.current_term:
            self._step_down(term)
            return
        
        if (self.state == RaftState.CANDIDATE and
                term == self.current_term and granted):
            self.votes_received.add(voter_id)
            
            if len(self.votes_received) > (len(self.peers) + 1) // 2:
                self._become_leader()
    
    def _become_leader(self):
        self.state = RaftState.LEADER
        self.leader_id = self.node_id
        
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = -1
    
    def _step_down(self, term: int):
        self.current_term = term
        self.state = RaftState.FOLLOWER
        self.voted_for = None
        self.votes_received.clear()
    
    def handle_vote_request(self, candidate_id: str, term: int,
                           last_log_index: int, last_log_term: int) -> Tuple[int, bool]:
        if term < self.current_term:
            return self.current_term, False
        
        if term > self.current_term:
            self._step_down(term)
        
        # Check if we can vote
        can_vote = (self.voted_for is None or
                   self.voted_for == candidate_id)
        
        # Check if candidate's log is up-to-date
        log_ok = (last_log_term > self.last_log_term or
                 (last_log_term == self.last_log_term and
                  last_log_index >= self.last_log_index))
        
        if can_vote and log_ok:
            self.voted_for = candidate_id
            return self.current_term, True
        
        return self.current_term, False
    
    def append_entry(self, command: str) -> Optional[LogEntry]:
        if self.state != RaftState.LEADER:
            return None
        
        entry = LogEntry(
            term=self.current_term,
            command=command,
            index=len(self.log))
        self.log.append(entry)
        return entry
    
    def handle_append_entries(self, leader_id: str, term: int,
                             prev_log_index: int, prev_log_term: int,
                             entries: List[LogEntry],
                             leader_commit: int) -> Tuple[int, bool]:
        if term < self.current_term:
            return self.current_term, False
        
        if term >= self.current_term:
            self._step_down(term)
            self.leader_id = leader_id
        
        # Check previous log entry
        if prev_log_index >= 0:
            if prev_log_index >= len(self.log):
                return self.current_term, False
            if self.log[prev_log_index].term != prev_log_term:
                return self.current_term, False
        
        # Append new entries
        for entry in entries:
            if entry.index < len(self.log):
                if self.log[entry.index].term != entry.term:
                    self.log = self.log[:entry.index]
                    self.log.append(entry)
            else:
                self.log.append(entry)
        
        # Update commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)
        
        return self.current_term, True
    
    def update_commit_index(self):
        if self.state != RaftState.LEADER:
            return
        
        for n in range(self.commit_index + 1, len(self.log)):
            if self.log[n].term != self.current_term:
                continue
            
            replicated = 1  # Count self
            for peer in self.peers:
                if self.match_index.get(peer, -1) >= n:
                    replicated += 1
            
            if replicated > (len(self.peers) + 1) // 2:
                self.commit_index = n


# ============================================================
# Saga Pattern Implementation
# ============================================================

class SagaStepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    name: str
    action: Any  # Callable
    compensation: Any  # Callable
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Any = None
    error: Optional[str] = None


class SagaOrchestrator:
    """Orchestrate distributed transactions using saga pattern."""
    
    def __init__(self, name: str):
        self.name = name
        self.steps: List[SagaStep] = []
        self.context: Dict[str, Any] = {}
    
    def add_step(self, name: str, action, compensation):
        self.steps.append(SagaStep(
            name=name,
            action=action,
            compensation=compensation))
    
    def execute(self) -> bool:
        completed_steps = []
        
        for step in self.steps:
            step.status = SagaStepStatus.RUNNING
            try:
                result = step.action(self.context)
                step.result = result
                step.status = SagaStepStatus.COMPLETED
                completed_steps.append(step)
                
                if isinstance(result, dict):
                    self.context.update(result)
                    
            except Exception as e:
                step.status = SagaStepStatus.FAILED
                step.error = str(e)
                
                self._compensate(completed_steps)
                return False
        
        return True
    
    def _compensate(self, completed_steps: List[SagaStep]):
        for step in reversed(completed_steps):
            step.status = SagaStepStatus.COMPENSATING
            try:
                step.compensation(self.context)
                step.status = SagaStepStatus.COMPENSATED
            except Exception as e:
                step.error = f"Compensation failed: {e}"


# ============================================================
# Failure Detector (Phi Accrual)
# ============================================================

import math

class PhiAccrualDetector:
    """Phi accrual failure detector."""
    
    def __init__(self, threshold: float = 8.0, window_size: int = 100):
        self.threshold = threshold
        self.window_size = window_size
        self._intervals: List[float] = []
        self._last_heartbeat: Optional[float] = None
    
    def heartbeat(self, timestamp: float = None):
        now = timestamp or time.time()
        if self._last_heartbeat is not None:
            interval = now - self._last_heartbeat
            self._intervals.append(interval)
            if len(self._intervals) > self.window_size:
                self._intervals.pop(0)
        self._last_heartbeat = now
    
    @property
    def phi(self) -> float:
        if not self._intervals or self._last_heartbeat is None:
            return 0.0
        
        now = time.time()
        elapsed = now - self._last_heartbeat
        
        mean = sum(self._intervals) / len(self._intervals)
        variance = sum((x - mean) ** 2 for x in self._intervals) / len(self._intervals)
        std = max(math.sqrt(variance), 0.001)
        
        # P(t) = 1 - F(t) where F is normal CDF
        y = (elapsed - mean) / std
        prob = 1.0 / (1.0 + math.exp(-1.5976 * y * (1 + 0.04 * y * y)))
        
        if prob >= 1.0:
            return float('inf')
        if prob <= 0.0:
            return 0.0
        
        return -math.log10(1 - prob)
    
    @property
    def is_alive(self) -> bool:
        return self.phi < self.threshold


# ============================================================
# Two-Phase Commit
# ============================================================

class TwoPhaseState(Enum):
    INIT = "init"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"


class Participant:
    """Transaction participant in 2PC."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = TwoPhaseState.INIT
        self._data: Dict[str, Any] = {}
        self._prepared_data: Dict[str, Any] = {}
    
    def prepare(self, changes: Dict[str, Any]) -> bool:
        try:
            self._prepared_data = changes
            self.state = TwoPhaseState.PREPARED
            return True
        except Exception:
            self.state = TwoPhaseState.ABORTED
            return False
    
    def commit(self):
        self._data.update(self._prepared_data)
        self._prepared_data.clear()
        self.state = TwoPhaseState.COMMITTED
    
    def abort(self):
        self._prepared_data.clear()
        self.state = TwoPhaseState.ABORTED


class TwoPhaseCoordinator:
    """Two-phase commit coordinator."""
    
    def __init__(self):
        self.participants: List[Participant] = []
        self.state = TwoPhaseState.INIT
    
    def add_participant(self, participant: Participant):
        self.participants.append(participant)
    
    def execute(self, changes_per_participant: Dict[str, Dict[str, Any]]) -> bool:
        # Phase 1: Prepare
        self.state = TwoPhaseState.PREPARING
        votes = {}
        
        for participant in self.participants:
            changes = changes_per_participant.get(participant.name, {})
            vote = participant.prepare(changes)
            votes[participant.name] = vote
        
        # Phase 2: Commit or Abort
        if all(votes.values()):
            self.state = TwoPhaseState.COMMITTING
            for participant in self.participants:
                participant.commit()
            self.state = TwoPhaseState.COMMITTED
            return True
        else:
            self.state = TwoPhaseState.ABORTING
            for participant in self.participants:
                participant.abort()
            self.state = TwoPhaseState.ABORTED
            return False`,
				},
			},
		},
		{
			ID:          2419,
			Title:       "Scalability Patterns and Load Balancing",
			Description: "Learn horizontal and vertical scaling, load balancing algorithms, auto-scaling strategies, connection pooling, and capacity planning.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Scaling Strategies Load Balancers and Capacity Planning",
					Content: `Scalability is the ability of a system to handle increasing load by adding resources. Understanding scaling patterns is essential for designing systems that grow gracefully.

**Scaling Dimensions:**

Vertical Scaling (Scale Up):
  Add more CPU, RAM, storage to existing machine
  Pros: Simple, no code changes, strong consistency
  Cons: Hardware limits, expensive, single point of failure
  Limits: ~128 cores, ~12TB RAM, ~100TB SSD
  Use when: Database primary, real-time systems, simplicity preferred

Horizontal Scaling (Scale Out):
  Add more machines to handle load
  Pros: Theoretically unlimited, cost-effective, fault tolerant
  Cons: Complexity, consistency challenges, network overhead
  Requires: Stateless services, distributed data, load balancing
  Use when: Web servers, microservices, read-heavy workloads

Diagonal Scaling:
  Combine vertical and horizontal
  Scale up until cost-effective limit, then scale out
  Example: Scale each node to 16 cores, then add more nodes

**Load Balancing:**

Layer 4 (Transport):
  Operates at TCP/UDP level
  Fast — only looks at IP and port
  No content inspection
  Examples: AWS NLB, HAProxy (TCP mode)
  
  Algorithms:
    Round Robin: Rotate through servers sequentially
    Weighted Round Robin: Assign weights based on capacity
    Least Connections: Route to server with fewest connections
    Weighted Least Connections: Combine weights with connection count
    Random: Simple random selection
    IP Hash: Hash source IP for sticky sessions

Layer 7 (Application):
  Operates at HTTP level
  Content-based routing
  SSL termination, compression, caching
  Examples: AWS ALB, Nginx, HAProxy (HTTP mode)
  
  Features:
    Path-based routing: /api → backend, /static → CDN
    Header-based routing: API version, content type
    Cookie-based sticky sessions
    A/B testing with weighted routing
    Rate limiting per endpoint
    Request/response transformation

Global Load Balancing:
  DNS-based (Route 53, Cloudflare)
  GeoDNS: Route based on user location
  Latency-based: Route to lowest latency region
  Failover: Active-passive with health checks
  
  Anycast:
    Same IP advertised from multiple locations
    BGP routing to nearest instance
    Used by CDNs and DNS servers

Health Checking:
  Active: Load balancer polls backends
    HTTP health endpoint: GET /health
    TCP connection check
    Custom scripts
    
  Passive: Monitor real traffic
    Error rate tracking
    Response time monitoring
    Circuit breaker pattern
    
  Configuration:
    Interval: 10-30 seconds
    Timeout: 5-10 seconds
    Healthy threshold: 2-3 consecutive successes
    Unhealthy threshold: 2-3 consecutive failures

**Connection Pooling:**

Benefits:
  Reduce connection establishment overhead
  Limit connections to backend services
  Better resource utilization

Patterns:
  Fixed pool: Pre-allocated connections (e.g., PgBouncer)
  Dynamic pool: Grow/shrink based on demand
  Per-service pools: Isolated pools per downstream service

Configuration:
  Min pool size: Baseline connections
  Max pool size: Upper limit
  Idle timeout: Close unused connections
  Max lifetime: Recycle connections
  Wait timeout: Queue wait time

**Auto-scaling:**

Reactive Scaling:
  Based on current metrics
  CPU utilization > 70% → scale up
  Request rate > threshold → add instances
  Queue depth > limit → add workers
  
  Metrics:
    CPU, memory utilization
    Request per second (RPS)
    Response latency (p95, p99)
    Queue depth
    Active connections
    Custom business metrics

Predictive Scaling:
  Based on historical patterns
  ML models predict future load
  Pre-warm instances before peaks
  AWS Predictive Scaling, time-based rules

Scheduled Scaling:
  Known traffic patterns
  Business hours: scale up at 9am, down at 6pm
  Seasonal: Black Friday, year-end
  Events: marketing campaigns

Cool-down Periods:
  Prevent rapid scale oscillation
  Scale-out cool-down: 3-5 minutes
  Scale-in cool-down: 10-15 minutes
  Allows metrics to stabilize

**Capacity Planning:**

Little's Law:
  L = λ × W
  L = number in system
  λ = arrival rate
  W = time in system
  
  Example: If 100 req/sec arrive and each takes 50ms:
  L = 100 × 0.05 = 5 concurrent requests

Universal Scalability Law (USL):
  C(N) = N / (1 + α(N-1) + β·N·(N-1))
  
  α = contention penalty (serialization)
  β = coherence penalty (crosstalk)
  
  With only contention (β=0): Amdahl's Law
  With coherence: Throughput degrades after peak

Back-of-envelope calculations:
  QPS per server: 1K-10K (web), 50-500 (API with DB)
  Read: 10x faster than write typically
  Storage: estimate data × growth × replication
  Network: payload × QPS × 8 bits/byte
  
  Standard numbers:
    Sequential read SSD: 500 MB/s
    Random read SSD: ~100K IOPS
    Network within DC: 10 Gbps
    Network across regions: varies
    Memory read: 100 ns
    L1 cache: 0.5 ns
    Disk seek HDD: 10 ms
    TCP round trip within DC: 0.5 ms
    TCP round trip cross-country: 30-50 ms

**Database Scaling:**

Read Replicas:
  Write to primary, read from replicas
  Replication lag (eventual consistency)
  Read-your-writes via reading from primary after write

Sharding:
  Range-based: Shard by ID ranges (hotspot risk)
  Hash-based: Hash function to assign shard (even distribution)
  Directory-based: Lookup table for shard mapping (flexible)
  Geographic: Shard by region (data locality)
  
  Challenges:
    Cross-shard queries
    Rebalancing when adding shards
    Referential integrity
    Global ordering

Write Scaling:
  Write-behind caching (async write to DB)
  Event sourcing (append-only log)
  CQRS (separate read/write models)
  Multi-leader replication (conflict resolution needed)`,
					CodeExamples: `# Scalability and Load Balancing Implementation Examples

import time
import random
import hashlib
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod

# ============================================================
# Load Balancer Implementations
# ============================================================

@dataclass
class Server:
    address: str
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    total_latency: float = 0.0
    
    @property
    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests


class LoadBalancer(ABC):
    def __init__(self):
        self.servers: List[Server] = []
    
    def add_server(self, server: Server):
        self.servers.append(server)
    
    def remove_server(self, address: str):
        self.servers = [s for s in self.servers if s.address != address]
    
    @property
    def healthy_servers(self) -> List[Server]:
        return [s for s in self.servers if s.healthy]
    
    @abstractmethod
    def select(self) -> Optional[Server]:
        pass


class RoundRobinLB(LoadBalancer):
    def __init__(self):
        super().__init__()
        self._index = 0
    
    def select(self) -> Optional[Server]:
        servers = self.healthy_servers
        if not servers:
            return None
        server = servers[self._index % len(servers)]
        self._index += 1
        return server


class WeightedRoundRobinLB(LoadBalancer):
    def __init__(self):
        super().__init__()
        self._current_weight = 0
        self._index = -1
    
    def select(self) -> Optional[Server]:
        servers = self.healthy_servers
        if not servers:
            return None
        
        max_weight = max(s.weight for s in servers)
        gcd_weight = servers[0].weight
        for s in servers[1:]:
            gcd_weight = math.gcd(gcd_weight, s.weight)
        
        while True:
            self._index = (self._index + 1) % len(servers)
            if self._index == 0:
                self._current_weight -= gcd_weight
                if self._current_weight <= 0:
                    self._current_weight = max_weight
            
            if servers[self._index].weight >= self._current_weight:
                return servers[self._index]


class LeastConnectionsLB(LoadBalancer):
    def select(self) -> Optional[Server]:
        servers = self.healthy_servers
        if not servers:
            return None
        return min(servers, key=lambda s: s.active_connections)


class IPHashLB(LoadBalancer):
    def select_for_ip(self, client_ip: str) -> Optional[Server]:
        servers = self.healthy_servers
        if not servers:
            return None
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return servers[hash_val % len(servers)]
    
    def select(self) -> Optional[Server]:
        return self.select_for_ip("0.0.0.0")


class LeastLatencyLB(LoadBalancer):
    def select(self) -> Optional[Server]:
        servers = self.healthy_servers
        if not servers:
            return None
        
        # Power of two choices
        if len(servers) < 2:
            return servers[0]
        
        s1, s2 = random.sample(servers, 2)
        if s1.avg_latency <= s2.avg_latency:
            return s1
        return s2


# ============================================================
# Consistent Hash Ring
# ============================================================

class ConsistentHashRing:
    """Consistent hashing with virtual nodes."""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []
        self._nodes: set = set()
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str):
        self._nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self._ring[hash_val] = node
        self._sorted_keys = sorted(self._ring.keys())
    
    def remove_node(self, node: str):
        self._nodes.discard(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self._ring.pop(hash_val, None)
        self._sorted_keys = sorted(self._ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        if not self._sorted_keys:
            return None
        
        hash_val = self._hash(key)
        
        # Binary search for first key >= hash_val
        lo, hi = 0, len(self._sorted_keys) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted_keys[mid] < hash_val:
                lo = mid + 1
            else:
                hi = mid
        
        if self._sorted_keys[lo] < hash_val:
            lo = 0  # Wrap around
        
        return self._ring[self._sorted_keys[lo]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        if not self._sorted_keys or count <= 0:
            return []
        
        result = []
        seen = set()
        hash_val = self._hash(key)
        
        lo = 0
        for i, k in enumerate(self._sorted_keys):
            if k >= hash_val:
                lo = i
                break
        
        for i in range(len(self._sorted_keys)):
            idx = (lo + i) % len(self._sorted_keys)
            node = self._ring[self._sorted_keys[idx]]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) >= count:
                    break
        
        return result


# ============================================================
# Auto-Scaler
# ============================================================

@dataclass
class ScalingPolicy:
    metric: str
    threshold_up: float
    threshold_down: float
    scale_up_count: int = 1
    scale_down_count: int = 1
    cooldown_up: float = 300.0  # 5 minutes
    cooldown_down: float = 600.0  # 10 minutes
    evaluation_periods: int = 3


class AutoScaler:
    """Reactive auto-scaler with cooldown."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 100):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.policies: List[ScalingPolicy] = []
        self._last_scale_up: float = 0
        self._last_scale_down: float = 0
        self._metric_history: Dict[str, deque] = {}
    
    def add_policy(self, policy: ScalingPolicy):
        self.policies.append(policy)
        self._metric_history[policy.metric] = deque(maxlen=policy.evaluation_periods)
    
    def record_metric(self, name: str, value: float):
        if name in self._metric_history:
            self._metric_history[name].append(value)
    
    def evaluate(self) -> int:
        now = time.time()
        desired = self.current_instances
        
        for policy in self.policies:
            history = self._metric_history.get(policy.metric, deque())
            if len(history) < policy.evaluation_periods:
                continue
            
            avg = sum(history) / len(history)
            
            # Check scale up
            if avg > policy.threshold_up:
                if now - self._last_scale_up >= policy.cooldown_up:
                    desired = max(desired,
                                 self.current_instances + policy.scale_up_count)
            
            # Check scale down
            elif avg < policy.threshold_down:
                if now - self._last_scale_down >= policy.cooldown_down:
                    desired = min(desired,
                                 self.current_instances - policy.scale_down_count)
        
        # Apply bounds
        desired = max(self.min_instances, min(self.max_instances, desired))
        
        if desired > self.current_instances:
            self._last_scale_up = now
        elif desired < self.current_instances:
            self._last_scale_down = now
        
        self.current_instances = desired
        return desired


# ============================================================
# Rate Limiter Implementations
# ============================================================

class TokenBucketLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    def allow(self, tokens: int = 1) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate)
        self._last_refill = now


class SlidingWindowLimiter:
    """Sliding window log rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()
    
    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            
            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False


class LeakyBucketLimiter:
    """Leaky bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self._queue: deque = deque()
        self._last_leak = time.time()
        self._lock = threading.Lock()
    
    def allow(self) -> bool:
        with self._lock:
            self._leak()
            if len(self._queue) < self.capacity:
                self._queue.append(time.time())
                return True
            return False
    
    def _leak(self):
        now = time.time()
        elapsed = now - self._last_leak
        leak_count = int(elapsed * self.rate)
        for _ in range(min(leak_count, len(self._queue))):
            self._queue.popleft()
        self._last_leak = now


# ============================================================
# Circuit Breaker
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 half_open_max: int = 1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    return True
                return False
            
            # HALF_OPEN
            return self._half_open_calls < self.half_open_max
    
    def record_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_max:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = 0
    
    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.failure_count = 0
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        if not self.can_execute():
            raise RuntimeError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise`,
				},
			},
		},
	})
}
