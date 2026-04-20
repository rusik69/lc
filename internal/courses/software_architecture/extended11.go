package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2329,
			Title:       "Infrastructure as Code and GitOps",
			Description: "Design infrastructure management with IaC principles, Terraform patterns, GitOps workflows, and infrastructure testing strategies.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "Infrastructure as Code Architecture",
					Content: `Infrastructure as Code (IaC) manages infrastructure through machine-readable definition files rather than manual processes.

**IaC Principles:**

Declarative over Imperative:
  Declarative: Define desired state, tool figures out how
    "I want 3 instances behind a load balancer"
    Tools: Terraform, CloudFormation, Pulumi
    
  Imperative: Define steps to reach desired state  
    "Create instance 1, create instance 2, create LB, attach instances"
    Tools: Ansible, Chef, scripts
    
  Prefer declarative for infrastructure provisioning
  Use imperative for configuration management

Idempotency:
  Running the same code multiple times produces the same result
  First run: Create resources
  Second run: No changes (already in desired state)
  Prevents duplicate resources or errors on re-run

Version Control:
  All infrastructure definitions in Git
  Pull requests for infrastructure changes
  Code review for infrastructure
  Audit trail of who changed what and when
  Rollback by reverting commits

Immutability:
  Replace, don't modify
  Change infrastructure by deploying new version
  Packer for golden images
  Container images for applications

**Terraform Architecture Patterns:**

Module Structure:
  modules/
    vpc/
      main.tf
      variables.tf
      outputs.tf
    eks-cluster/
      main.tf
      variables.tf
      outputs.tf
    rds/
      main.tf
      variables.tf
      outputs.tf
  environments/
    dev/
      main.tf (uses modules)
      terraform.tfvars
    staging/
      main.tf
      terraform.tfvars
    production/
      main.tf
      terraform.tfvars

State Management:
  Remote state in S3/GCS/Azure Blob
  State locking with DynamoDB/GCS
  One state per environment
  State isolation prevents blast radius

  Backend configuration:
    terraform {
      backend "s3" {
        bucket         = "company-terraform-state"
        key            = "prod/infrastructure.tfstate"
        region         = "us-east-1"
        dynamodb_table = "terraform-locks"
        encrypt        = true
      }
    }

Workspace Patterns:
  Workspace per environment (simple but limited)
  Directory per environment (more flexible, recommended)
  Terragrunt for DRY configurations

Dependency Management:
  Use data sources for cross-stack references
  Output values for inter-module communication
  Avoid circular dependencies between modules

**GitOps:**

Principles:
  1. Declarative: System described declaratively
  2. Versioned: Desired state stored in Git
  3. Automated: Approved changes auto-applied
  4. Self-healing: Agents ensure actual matches desired

GitOps Workflow:
  Developer -> Pull Request -> Review -> Merge
    -> GitOps Agent detects change
    -> Agent applies change to cluster
    -> Agent monitors and reconciles drift

Pull-based (recommended):
  Agent in cluster pulls changes from Git
  No external access to cluster needed
  Self-healing: drift detection and correction
  Tools: ArgoCD, Flux

Push-based:
  CI/CD pipeline pushes changes to cluster
  Requires cluster credentials in CI
  No automatic drift detection
  Tools: Jenkins, GitHub Actions

ArgoCD Architecture:
  Application CRD: Defines source repo, target cluster, sync policy
  Application Controller: Monitors apps, detects drift
  Repo Server: Fetches Git repos, generates manifests
  API Server: UI and CLI interface
  
  Sync Policies:
    Manual: Requires human approval
    Automated: Auto-sync on Git change
    Self-heal: Revert manual cluster changes
    Prune: Delete resources removed from Git

**Infrastructure Testing:**

Unit Tests (fast, isolated):
  Validate Terraform configuration syntax
  Check module input/output contracts
  Policy-as-code (OPA, Sentinel)
  Tools: terraform validate, tflint, conftest

Integration Tests (medium speed):
  Deploy infrastructure to test environment
  Verify resources are created correctly
  Test connectivity and configuration
  Tools: Terratest, Kitchen-Terraform

End-to-End Tests (slow, expensive):
  Full environment deployment
  Application deployment on infrastructure
  Load testing and chaos testing
  Cleanup after tests

Policy-as-Code:
  Enforce organizational rules automatically
  Examples:
    All S3 buckets must be encrypted
    No public security group rules
    All resources must have tags
    Instance types must be from approved list
  
  Tools:
    OPA/Rego: General-purpose policy engine
    Sentinel (HashiCorp): Terraform Enterprise
    Checkov: Static analysis for IaC
    tfsec: Security scanner for Terraform

**Drift Detection:**

  Infrastructure drift: Actual state differs from desired state
  
  Causes:
    Manual changes via console/CLI
    External automation modifying resources
    State file corruption or inconsistency
  
  Detection:
    terraform plan: Shows difference between state and reality
    GitOps agents: Continuous reconciliation
    Scheduled drift detection runs
    
  Prevention:
    Lock down manual access (read-only console)
    All changes through Git/IaC pipeline
    Automated drift detection and alerting
    Self-healing GitOps agents

**Environment Promotion:**

  dev -> staging -> production
  
  Promote by:
    Updating version reference in environment config
    Same infrastructure modules, different parameters
    Automated testing gates between environments
    
  Strategies:
    Blue/Green: Two identical environments, switch traffic
    Canary: Gradual traffic shift to new version
    Rolling: Update instances one at a time

**Secrets in IaC:**

  Never store secrets in Terraform files or state
  Use external secrets managers
  Reference secrets dynamically
  
  Approaches:
    AWS Secrets Manager / Parameter Store
    HashiCorp Vault
    SOPS (encrypted files in Git)
    External Secrets Operator (Kubernetes)
    Sealed Secrets (Kubernetes)`,
					CodeExamples: `// Infrastructure as Code patterns and testing

// Terraform module structure (Go representation for testing)
type TerraformModule struct {
    Name      string
    Path      string
    Variables map[string]Variable
    Outputs   map[string]Output
    Resources []Resource
}

type Variable struct {
    Name        string
    Type        string
    Default     interface{}
    Description string
    Required    bool
    Validation  *ValidationRule
}

type Output struct {
    Name        string
    Value       string
    Description string
    Sensitive   bool
}

type Resource struct {
    Type   string
    Name   string
    Config map[string]interface{}
}

// Infrastructure test framework
type InfraTest struct {
    terraformDir string
    vars         map[string]interface{}
    output       map[string]string
    cleanup      func()
}

func NewInfraTest(dir string, vars map[string]interface{}) *InfraTest {
    return &InfraTest{
        terraformDir: dir,
        vars:         vars,
        output:       make(map[string]string),
    }
}

func (t *InfraTest) Deploy(ctx context.Context) error {
    // Init
    if err := runTerraform(ctx, t.terraformDir, "init"); err != nil {
        return fmt.Errorf("terraform init failed: %w", err)
    }
    
    // Plan
    args := []string{"plan", "-out=tfplan"}
    for k, v := range t.vars {
        args = append(args, fmt.Sprintf("-var=%s=%v", k, v))
    }
    if err := runTerraform(ctx, t.terraformDir, args...); err != nil {
        return fmt.Errorf("terraform plan failed: %w", err)
    }
    
    // Apply
    if err := runTerraform(ctx, t.terraformDir, "apply", "-auto-approve", "tfplan"); err != nil {
        return fmt.Errorf("terraform apply failed: %w", err)
    }
    
    // Capture outputs
    outputs, err := getTerraformOutputs(ctx, t.terraformDir)
    if err != nil {
        return fmt.Errorf("failed to get outputs: %w", err)
    }
    t.output = outputs
    
    t.cleanup = func() {
        destroyCtx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
        defer cancel()
        runTerraform(destroyCtx, t.terraformDir, "destroy", "-auto-approve")
    }
    
    return nil
}

func (t *InfraTest) GetOutput(key string) string {
    return t.output[key]
}

func (t *InfraTest) Destroy() {
    if t.cleanup != nil {
        t.cleanup()
    }
}

// Policy checker (OPA-inspired)
type PolicyEngine struct {
    policies []Policy
}

type Policy struct {
    Name        string
    Description string
    Severity    string // error, warning, info
    Check       func(Resource) *PolicyViolation
}

type PolicyViolation struct {
    Policy   string
    Resource string
    Message  string
    Severity string
}

func NewPolicyEngine() *PolicyEngine {
    return &PolicyEngine{}
}

func (e *PolicyEngine) AddPolicy(p Policy) {
    e.policies = append(e.policies, p)
}

func (e *PolicyEngine) Evaluate(resources []Resource) []PolicyViolation {
    var violations []PolicyViolation
    
    for _, resource := range resources {
        for _, policy := range e.policies {
            if v := policy.Check(resource); v != nil {
                violations = append(violations, *v)
            }
        }
    }
    
    return violations
}

// Standard policies
func S3EncryptionPolicy() Policy {
    return Policy{
        Name:        "s3-encryption-required",
        Description: "All S3 buckets must have server-side encryption enabled",
        Severity:    "error",
        Check: func(r Resource) *PolicyViolation {
            if r.Type != "aws_s3_bucket" {
                return nil
            }
            encryption, ok := r.Config["server_side_encryption_configuration"]
            if !ok || encryption == nil {
                return &PolicyViolation{
                    Policy:   "s3-encryption-required",
                    Resource: r.Name,
                    Message:  "S3 bucket must have server-side encryption enabled",
                    Severity: "error",
                }
            }
            return nil
        },
    }
}

func RequiredTagsPolicy(requiredTags []string) Policy {
    return Policy{
        Name:        "required-tags",
        Description: "All resources must have required tags",
        Severity:    "error",
        Check: func(r Resource) *PolicyViolation {
            tags, ok := r.Config["tags"].(map[string]string)
            if !ok {
                return &PolicyViolation{
                    Policy:   "required-tags",
                    Resource: r.Name,
                    Message:  "Resource is missing tags",
                    Severity: "error",
                }
            }
            for _, required := range requiredTags {
                if _, exists := tags[required]; !exists {
                    return &PolicyViolation{
                        Policy:   "required-tags",
                        Resource: r.Name,
                        Message:  fmt.Sprintf("Missing required tag: %s", required),
                        Severity: "error",
                    }
                }
            }
            return nil
        },
    }
}

func NoPublicAccessPolicy() Policy {
    return Policy{
        Name:        "no-public-access",
        Description: "Security groups must not allow unrestricted inbound access",
        Severity:    "error",
        Check: func(r Resource) *PolicyViolation {
            if r.Type != "aws_security_group_rule" {
                return nil
            }
            ruleType, _ := r.Config["type"].(string)
            cidr, _ := r.Config["cidr_blocks"].([]string)
            
            if ruleType == "ingress" {
                for _, block := range cidr {
                    if block == "0.0.0.0/0" {
                        port, _ := r.Config["from_port"].(int)
                        if port != 443 && port != 80 {
                            return &PolicyViolation{
                                Policy:   "no-public-access",
                                Resource: r.Name,
                                Message:  fmt.Sprintf("Port %d open to 0.0.0.0/0", port),
                                Severity: "error",
                            }
                        }
                    }
                }
            }
            return nil
        },
    }
}

// GitOps reconciliation loop
type GitOpsReconciler struct {
    gitRepo     GitRepository
    cluster     ClusterClient
    interval    time.Duration
    logger      *Logger
}

type DesiredState struct {
    Namespace  string
    Resources  []KubeResource
    CommitHash string
}

type KubeResource struct {
    APIVersion string
    Kind       string
    Name       string
    Namespace  string
    Spec       map[string]interface{}
}

func (r *GitOpsReconciler) Start(ctx context.Context) {
    ticker := time.NewTicker(r.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            if err := r.reconcile(ctx); err != nil {
                r.logger.Error("reconciliation failed", "error", err)
            }
        }
    }
}

func (r *GitOpsReconciler) reconcile(ctx context.Context) error {
    // Fetch desired state from Git
    desired, err := r.gitRepo.GetDesiredState(ctx)
    if err != nil {
        return fmt.Errorf("failed to get desired state: %w", err)
    }
    
    // Get actual state from cluster
    actual, err := r.cluster.GetResources(ctx, desired.Namespace)
    if err != nil {
        return fmt.Errorf("failed to get actual state: %w", err)
    }
    
    // Compute diff
    diff := computeDiff(desired.Resources, actual)
    
    if len(diff.ToCreate) == 0 && len(diff.ToUpdate) == 0 && len(diff.ToDelete) == 0 {
        r.logger.Info("no drift detected", "commit", desired.CommitHash)
        return nil
    }
    
    r.logger.Info("drift detected",
        "create", len(diff.ToCreate),
        "update", len(diff.ToUpdate),
        "delete", len(diff.ToDelete),
        "commit", desired.CommitHash,
    )
    
    // Apply changes
    for _, resource := range diff.ToCreate {
        if err := r.cluster.Apply(ctx, resource); err != nil {
            return fmt.Errorf("failed to create %s/%s: %w", resource.Kind, resource.Name, err)
        }
    }
    
    for _, resource := range diff.ToUpdate {
        if err := r.cluster.Apply(ctx, resource); err != nil {
            return fmt.Errorf("failed to update %s/%s: %w", resource.Kind, resource.Name, err)
        }
    }
    
    for _, resource := range diff.ToDelete {
        if err := r.cluster.Delete(ctx, resource); err != nil {
            return fmt.Errorf("failed to delete %s/%s: %w", resource.Kind, resource.Name, err)
        }
    }
    
    r.logger.Info("reconciliation complete", "commit", desired.CommitHash)
    return nil
}

type ResourceDiff struct {
    ToCreate []KubeResource
    ToUpdate []KubeResource
    ToDelete []KubeResource
}`,
				},
			},
		},
		{
			ID:          2330,
			Title:       "Distributed Systems Consensus and Coordination",
			Description: "Understand distributed consensus with CAP theorem, Raft algorithm, distributed locks, leader election, and consistency models in distributed systems.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "Consensus and Coordination in Distributed Systems",
					Content: `Distributed systems face fundamental challenges around agreement, ordering, and consistency that require careful architectural decisions.

**CAP Theorem:**

  In a distributed system, you can only guarantee two of three:
  
  Consistency (C):
    Every read receives the most recent write
    All nodes see the same data at the same time
    Like a single-node system from client's perspective
    
  Availability (A):
    Every request receives a response
    No timeouts or errors (even if data is stale)
    System always responds to read and write requests
    
  Partition Tolerance (P):
    System continues despite network partitions
    Nodes can't communicate but system still functions
    Required in any distributed system
    
  In practice (since partitions happen):
    CP: Consistent but may be unavailable during partitions
      Examples: ZooKeeper, etcd, HBase, MongoDB (default)
      Use when: Financial transactions, leader election
      
    AP: Available but may serve stale data during partitions
      Examples: Cassandra, DynamoDB, CouchDB
      Use when: Product catalogs, social media feeds

  PACELC Theorem (extends CAP):
    During Partition: Choose Availability or Consistency
    Else (no partition): Choose Latency or Consistency
    
    PA/EL: Maximize availability and latency (DynamoDB, Cassandra)
    PC/EC: Maximize consistency always (traditional databases)
    PA/EC: Available during partition, consistent otherwise

**Consistency Models:**

Strong Consistency (Linearizability):
  Operations appear instantaneous
  Once a write completes, all reads see it
  Expensive: requires coordination for every operation
  Example: Single-leader database

Sequential Consistency:
  Operations appear in program order per process
  Different processes may see different orderings
  No real-time guarantee

Causal Consistency:
  Causally related operations appear in order
  Concurrent operations may appear in any order
  Cheaper than strong consistency
  Vector clocks track causality

Eventual Consistency:
  Given no new updates, replicas eventually converge
  No guarantee when convergence happens
  Cheapest, most available
  Suitable for many read-heavy workloads
  
  Variants:
    Read-your-writes: You see your own recent writes
    Monotonic reads: Once you see a value, you won't see older
    Session consistency: Consistency within a client session

**Distributed Consensus Algorithms:**

Raft Algorithm:
  Designed for understandability (vs Paxos)
  Three roles: Leader, Follower, Candidate
  
  Leader Election:
    All nodes start as followers
    If no heartbeat from leader, follower becomes candidate
    Candidate requests votes from other nodes
    Majority votes -> becomes leader
    Leader sends heartbeats to maintain authority
  
  Log Replication:
    Client sends request to leader
    Leader appends to its log
    Leader replicates to followers
    Once majority acknowledges -> committed
    Leader responds to client
  
  Safety:
    At most one leader per term
    Leader's log is always the most up-to-date
    Committed entries are durable (in majority of nodes)

Paxos:
  Foundational consensus algorithm
  Three roles: Proposer, Acceptor, Learner
  Multi-Paxos for continuous consensus
  More complex than Raft, equivalent guarantees

**Distributed Locks:**

Requirements:
  Mutual exclusion: Only one client holds the lock
  Deadlock-free: Lock is eventually released
  Fault-tolerant: Lock works despite failures

Single-Node Lock (Redis SETNX):
  SET key value NX EX 30
  Simple, but single point of failure
  
RedLock (Redis distributed lock):
  Lock acquired on majority of N independent Redis instances
  Lock timeout prevents deadlocks
  Client must complete work before lock expires
  Steps:
    1. Get current time
    2. Try to acquire lock on all N instances
    3. Lock acquired if majority + time elapsed < TTL
    4. Work time = TTL - elapsed time
    5. Release lock on all instances

Consensus-Based Locks (ZooKeeper, etcd):
  Stronger guarantees than Redis
  Use consensus protocol for distributed coordination
  ZooKeeper: Ephemeral nodes with sequential ordering
  etcd: Lease-based locks with TTL

**Leader Election:**

When needed:
  One node should perform certain operations
  Avoid split-brain in active-passive setups
  Coordinate distributed tasks

Approaches:
  ZooKeeper/etcd: Create ephemeral node, lowest sequence wins
  Bully Algorithm: Highest ID wins, simpler but less robust
  Raft: Built-in leader election
  Kubernetes: Leader election via Lease objects

**Vector Clocks and Causality:**

  Track causal ordering of events in distributed system
  Each node maintains a vector of logical timestamps
  
  Rules:
    Before sending: Increment own counter
    On receive: Take max of each position + increment own
    
  Example:
    Node A: [1,0,0] -> sends to B
    Node B: receives -> [1,1,0]
    Node B: [1,2,0] -> sends to C
    Node C: receives -> [1,2,1]
    
  Comparison:
    A < B if all positions in A <= B and at least one <
    A || B (concurrent) if neither A < B nor B < A

**Crdt (Conflict-Free Replicated Data Types):**

  Data structures that can be replicated across nodes
  Always converge without coordination
  No conflicts to resolve
  
  Types:
    G-Counter: Grow-only counter (each node increments own slot)
    PN-Counter: Positive-negative counter (two G-Counters)
    G-Set: Grow-only set (add-only, union for merge)
    OR-Set: Observed-Remove set (add and remove)
    LWW-Register: Last-Writer-Wins register
    
  Use cases:
    Shopping cart (OR-Set)
    Like counters (PN-Counter)
    Collaborative editing
    Eventually consistent databases`,
					CodeExamples: `// Distributed systems patterns

// Distributed lock with fencing token
type DistributedLock struct {
    client  LockClient
    key     string
    value   string
    ttl     time.Duration
    token   int64
}

type LockClient interface {
    Acquire(ctx context.Context, key, value string, ttl time.Duration) (int64, error)
    Release(ctx context.Context, key, value string) error
    Extend(ctx context.Context, key, value string, ttl time.Duration) error
}

func NewDistributedLock(client LockClient, key string, ttl time.Duration) *DistributedLock {
    return &DistributedLock{
        client: client,
        key:    key,
        value:  generateUUID(),
        ttl:    ttl,
    }
}

func (l *DistributedLock) Lock(ctx context.Context) error {
    token, err := l.client.Acquire(ctx, l.key, l.value, l.ttl)
    if err != nil {
        return fmt.Errorf("failed to acquire lock: %w", err)
    }
    l.token = token
    return nil
}

func (l *DistributedLock) Unlock(ctx context.Context) error {
    return l.client.Release(ctx, l.key, l.value)
}

func (l *DistributedLock) FencingToken() int64 {
    return l.token
}

// Leader election
type LeaderElector struct {
    mu          sync.Mutex
    nodeID      string
    isLeader    bool
    leaderID    string
    client      CoordinationClient
    leaseTTL    time.Duration
    onElected   func(ctx context.Context)
    onDeposed   func()
    logger      *Logger
}

type CoordinationClient interface {
    TryAcquireLease(ctx context.Context, name, holder string, ttl time.Duration) (bool, error)
    RenewLease(ctx context.Context, name, holder string, ttl time.Duration) error
    GetLeaseHolder(ctx context.Context, name string) (string, error)
}

func NewLeaderElector(nodeID string, client CoordinationClient, ttl time.Duration) *LeaderElector {
    return &LeaderElector{
        nodeID:   nodeID,
        client:   client,
        leaseTTL: ttl,
    }
}

func (e *LeaderElector) Start(ctx context.Context) {
    ticker := time.NewTicker(e.leaseTTL / 3) // Renew at 1/3 of TTL
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            e.resign()
            return
        case <-ticker.C:
            e.tryElection(ctx)
        }
    }
}

func (e *LeaderElector) tryElection(ctx context.Context) {
    e.mu.Lock()
    defer e.mu.Unlock()
    
    if e.isLeader {
        // Try to renew lease
        err := e.client.RenewLease(ctx, "leader", e.nodeID, e.leaseTTL)
        if err != nil {
            e.logger.Warn("lost leadership", "node", e.nodeID, "error", err)
            e.isLeader = false
            if e.onDeposed != nil {
                e.onDeposed()
            }
        }
        return
    }
    
    // Try to become leader
    acquired, err := e.client.TryAcquireLease(ctx, "leader", e.nodeID, e.leaseTTL)
    if err != nil {
        e.logger.Error("election error", "error", err)
        return
    }
    
    if acquired {
        e.logger.Info("elected as leader", "node", e.nodeID)
        e.isLeader = true
        e.leaderID = e.nodeID
        if e.onElected != nil {
            go e.onElected(ctx)
        }
    }
}

func (e *LeaderElector) resign() {
    e.mu.Lock()
    defer e.mu.Unlock()
    
    if e.isLeader {
        e.isLeader = false
        if e.onDeposed != nil {
            e.onDeposed()
        }
    }
}

func (e *LeaderElector) IsLeader() bool {
    e.mu.Lock()
    defer e.mu.Unlock()
    return e.isLeader
}

// Vector clock
type VectorClock struct {
    mu     sync.RWMutex
    clocks map[string]uint64
}

func NewVectorClock(nodeID string) *VectorClock {
    return &VectorClock{
        clocks: map[string]uint64{nodeID: 0},
    }
}

func (vc *VectorClock) Increment(nodeID string) {
    vc.mu.Lock()
    defer vc.mu.Unlock()
    vc.clocks[nodeID]++
}

func (vc *VectorClock) Merge(other *VectorClock) {
    vc.mu.Lock()
    defer vc.mu.Unlock()
    other.mu.RLock()
    defer other.mu.RUnlock()
    
    for node, clock := range other.clocks {
        if clock > vc.clocks[node] {
            vc.clocks[node] = clock
        }
    }
}

func (vc *VectorClock) HappensBefore(other *VectorClock) bool {
    vc.mu.RLock()
    defer vc.mu.RUnlock()
    other.mu.RLock()
    defer other.mu.RUnlock()
    
    atLeastOneLess := false
    for node, clock := range vc.clocks {
        otherClock := other.clocks[node]
        if clock > otherClock {
            return false
        }
        if clock < otherClock {
            atLeastOneLess = true
        }
    }
    
    // Check for nodes in other but not in vc
    for node := range other.clocks {
        if _, exists := vc.clocks[node]; !exists {
            atLeastOneLess = true
        }
    }
    
    return atLeastOneLess
}

func (vc *VectorClock) Concurrent(other *VectorClock) bool {
    return !vc.HappensBefore(other) && !other.HappensBefore(vc)
}

// G-Counter CRDT
type GCounter struct {
    mu       sync.RWMutex
    nodeID   string
    counters map[string]uint64
}

func NewGCounter(nodeID string) *GCounter {
    return &GCounter{
        nodeID:   nodeID,
        counters: map[string]uint64{nodeID: 0},
    }
}

func (c *GCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.counters[c.nodeID]++
}

func (c *GCounter) Value() uint64 {
    c.mu.RLock()
    defer c.mu.RUnlock()
    var total uint64
    for _, count := range c.counters {
        total += count
    }
    return total
}

func (c *GCounter) Merge(other *GCounter) {
    c.mu.Lock()
    defer c.mu.Unlock()
    other.mu.RLock()
    defer other.mu.RUnlock()
    
    for node, count := range other.counters {
        if count > c.counters[node] {
            c.counters[node] = count
        }
    }
}

// PN-Counter CRDT (supports increment and decrement)
type PNCounter struct {
    positive *GCounter
    negative *GCounter
}

func NewPNCounter(nodeID string) *PNCounter {
    return &PNCounter{
        positive: NewGCounter(nodeID),
        negative: NewGCounter(nodeID),
    }
}

func (c *PNCounter) Increment() {
    c.positive.Increment()
}

func (c *PNCounter) Decrement() {
    c.negative.Increment()
}

func (c *PNCounter) Value() int64 {
    return int64(c.positive.Value()) - int64(c.negative.Value())
}

func (c *PNCounter) Merge(other *PNCounter) {
    c.positive.Merge(other.positive)
    c.negative.Merge(other.negative)
}`,
				},
			},
		},
	})
}
