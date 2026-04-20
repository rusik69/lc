package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2123,
			Title:       "AWS Compute and Auto Scaling",
			Description: "Master EC2 instance types, placement groups, Auto Scaling, Launch Templates, Spot instances, and Elastic Beanstalk.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "EC2 Auto Scaling Spot and Compute Services",
					Content: `AWS compute services span from virtual machines (EC2) to managed platforms (Elastic Beanstalk), with sophisticated scaling and purchasing options.

**EC2 Instance Types:**

General Purpose (M, T):
  M7i, M7g (Graviton): Balanced compute/memory/networking
  T3, T4g: Burstable, CPU credits
  M7i-flex: Smaller sizes for variable workloads
  
  Use: Web servers, app servers, small databases

Compute Optimized (C):
  C7i, C7g, C7gn: High-performance processors
  Best compute-to-memory ratio
  
  Use: Batch processing, ML inference, gaming, HPC

Memory Optimized (R, X, z):
  R7i, R7g: High memory
  X2idn: Up to 4 TB RAM
  z1d: High frequency + high memory
  
  Use: In-memory databases, real-time analytics

Storage Optimized (I, D, H):
  I4i: High random I/O (NVMe)
  D3en: Dense HDD storage (336 TB)
  H1: HDD throughput
  
  Use: Data warehouses, distributed file systems

Accelerated Computing (P, G, Inf, Trn, DL):
  P5: NVIDIA H100 GPUs (ML training)
  G5: NVIDIA A10G (graphic/inference)
  Inf2: AWS Inferentia2 (ML inference)
  Trn1: AWS Trainium (ML training)
  DL2q: Qualcomm AI (inference)
  
  Use: ML training/inference, graphics, video encoding

HPC:
  Hpc7a: AMD EPYC, EFA networking
  Hpc7g: Graviton3E, EFA networking
  
  Use: Computational fluid dynamics, weather modeling

**Graviton Processors:**
  ARM-based, AWS-designed
  Up to 40% better price/performance vs x86
  Available in M, C, R, T, G, Hpc families
  Graviton3: 200+ instance types
  Graviton4: Latest generation

**EC2 Placement Groups:**

Cluster:
  Instances in same rack/AZ
  Lowest latency (10 Gbps between instances)
  Risk: All fail if rack fails
  Use: HPC, tightly coupled workloads

Spread:
  Each instance on different hardware
  Max 7 instances per AZ per group
  Reduces correlated failures
  Use: Critical individual instances

Partition:
  Instances in logical partitions
  Each partition on separate rack
  Up to 7 partitions per AZ
  Use: Hadoop, Cassandra, Kafka

**EC2 Purchasing Options:**

On-Demand:
  Pay per second (Linux) or per hour (Windows)
  No commitment
  Full price
  
Reserved Instances:
  1 or 3 year commitment
  Up to 72% discount
  Standard RI: Fixed instance type
  Convertible RI: Change instance type
  Payment: All upfront, partial upfront, no upfront
  
Savings Plans:
  Compute Savings Plans: Any instance family/region
  EC2 Instance Savings Plans: Specific family + region
  1 or 3 year commitment
  Up to 72% discount
  
Spot Instances:
  Up to 90% discount
  Can be interrupted (2-minute warning)
  Spot Fleet: Multiple instance types
  
  Strategies:
    Capacity-optimized: Least likely to be interrupted
    Lowest-price: Cheapest instances
    Diversified: Spread across pools
    Price-capacity-optimized: Balance of both
    
  Best practices:
    Use multiple instance types
    Use multiple AZs
    Mix with On-Demand
    Checkpointing for stateful work
    Handle interruption gracefully

Dedicated Hosts:
  Physical server dedicated to you
  Socket/core visibility
  Bring your own license (BYOL)
  Compliance requirements

Dedicated Instances:
  Dedicated hardware (no other accounts)
  May share with your other instances
  Per-instance pricing

Capacity Reservations:
  Reserve capacity in specific AZ
  No commitment discount
  Combines with Reserved/Savings Plans

**Auto Scaling:**

Launch Template:
  AMI ID, instance type, key pair
  Security groups, subnet
  User data, IAM role
  Block devices, network interfaces
  Multiple versions

Auto Scaling Group (ASG):
  Min, Max, Desired capacity
  
  Scaling Policies:
    Target Tracking:
      Maintain metric at target (e.g., CPU at 50%)
      Simplest, most common
      Predefined: CPU, Network, ALB request count
      Custom: CloudWatch metric
      
    Step Scaling:
      Scale based on CloudWatch alarm
      Step adjustments by threshold
      Example: +1 at 60% CPU, +3 at 80% CPU
      
    Simple Scaling:
      Single adjustment
      Cooldown period
      Legacy, use step/target instead
      
    Scheduled Scaling:
      Scale at specific times
      Predictable demand patterns
      Cron expressions
      
    Predictive Scaling:
      ML-based forecast
      Pre-provision capacity
      Combines with dynamic scaling

  Health Checks:
    EC2: Instance status checks
    ELB: Load balancer health checks
    Custom: Via API
    Grace period: Wait after launch
    
  Instance Refresh:
    Rolling update of instances
    Minimum healthy percentage
    Warm-up time
    Checkpoint (pause during)
    
  Lifecycle Hooks:
    Pending:Wait -> Custom actions -> InService
    Terminating:Wait -> Custom actions -> Terminated
    Use cases: Install software, drain connections, backup
    
  Warm Pools:
    Pre-initialized stopped instances
    Faster scaling (skip boot/init)
    Hibernate or Stop state
    
  Mixed Instance Policy:
    Multiple instance types
    On-Demand base capacity
    Spot percentage
    Allocation strategies

**Elastic Beanstalk:**

  Managed platform for web apps
  
  Supported Platforms:
    Java, .NET, PHP, Node.js, Python, Ruby, Go
    Docker (single/multi-container)
    
  Deployment Strategies:
    All at once: Fast, downtime
    Rolling: Update batch at a time
    Rolling with additional batch: Maintain capacity
    Immutable: New ASG, swap when healthy
    Blue/Green: Separate environment, swap CNAME
    Traffic splitting: Canary deployment
    
  Environment Tiers:
    Web Server: ALB + EC2/ECS
    Worker: SQS + EC2 (background processing)
    
  Configuration:
    .ebextensions: YAML/JSON config files
    Saved Configurations: Reusable templates
    Environment variables
    Custom AMI
    
**AWS Batch:**

  Managed batch computing
  
  Components:
    Job Definition: Container, vCPU, memory, command
    Job Queue: Priority-based
    Compute Environment: Managed or unmanaged EC2/Fargate
    
  Features:
    Array Jobs: Run many copies
    Multi-node Jobs: MPI workloads
    Spot integration
    Fair share scheduling
    Job dependencies (sequential/fan-out)`,
					CodeExamples: `// AWS compute and auto scaling implementations

package main

import (
    "fmt"
    "math"
    "math/rand"
    "sort"
    "strings"
    "sync"
    "time"
)

// EC2 instance manager
type EC2Manager struct {
    instances     map[string]*EC2Instance
    reservations  map[string]*EC2Reservation
    spotRequests  map[string]*SpotRequest
    mu            sync.RWMutex
}

type EC2Instance struct {
    InstanceID    string
    Type          string
    State         string // pending, running, stopping, stopped, terminated
    AZ            string
    SubnetID      string
    VpcID         string
    PrivateIP     string
    PublicIP      string
    SecurityGroups []string
    KeyPair       string
    LaunchTime    time.Time
    Monitoring    bool
    IAMRole       string
    Tags          map[string]string
    Placement     *PlacementConfig
    PurchaseType  string // on-demand, reserved, spot
    SpotPrice     float64
}

type PlacementConfig struct {
    GroupName string
    GroupType string // cluster, spread, partition
    Partition int
}

type EC2Reservation struct {
    ID           string
    InstanceType string
    Count        int
    Duration     int // months
    Offering     string // AllUpfront, PartialUpfront, NoUpfront
    State        string
    StartDate    time.Time
    EndDate      time.Time
    Used         int
}

type SpotRequest struct {
    ID            string
    InstanceType  string
    MaxPrice      float64
    CurrentPrice  float64
    State         string // open, active, closed, cancelled
    InstanceID    string
    LaunchTime    time.Time
    InterruptedAt time.Time
}

func NewEC2Manager() *EC2Manager {
    return &EC2Manager{
        instances:    make(map[string]*EC2Instance),
        reservations: make(map[string]*EC2Reservation),
        spotRequests: make(map[string]*SpotRequest),
    }
}

func (m *EC2Manager) LaunchInstance(instType, az, subnetID string, purchase string) (*EC2Instance, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    instance := &EC2Instance{
        InstanceID:   fmt.Sprintf("i-%s", generateComputeID()),
        Type:         instType,
        State:        "running",
        AZ:           az,
        SubnetID:     subnetID,
        PrivateIP:    fmt.Sprintf("10.0.%d.%d", rand.Intn(255), 2+rand.Intn(253)),
        LaunchTime:   time.Now(),
        PurchaseType: purchase,
        Tags:         make(map[string]string),
    }
    
    m.instances[instance.InstanceID] = instance
    return instance, nil
}

func (m *EC2Manager) TerminateInstance(instanceID string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    inst, exists := m.instances[instanceID]
    if !exists {
        return fmt.Errorf("instance %s not found", instanceID)
    }
    
    inst.State = "terminated"
    return nil
}

func (m *EC2Manager) GetRunningInstances() []*EC2Instance {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    var running []*EC2Instance
    for _, inst := range m.instances {
        if inst.State == "running" {
            running = append(running, inst)
        }
    }
    return running
}

// Auto Scaling Group
type AutoScalingGroup struct {
    Name             string
    LaunchTemplate   *LaunchTemplate
    MinSize          int
    MaxSize          int
    DesiredCapacity  int
    Instances        []*ASGInstance
    TargetGroups     []string
    HealthCheckType  string // EC2, ELB
    HealthCheckGrace time.Duration
    Policies         []ScalingPolicyConfig
    ScheduledActions []ScheduledAction
    WarmPool         *WarmPoolConfig
    MixedPolicy      *MixedInstancePolicy
    Tags             map[string]string
    mu               sync.RWMutex
}

type LaunchTemplate struct {
    ID              string
    Version         int
    AMI             string
    InstanceType    string
    KeyPair         string
    SecurityGroups  []string
    UserData        string
    IAMProfile      string
    BlockDevices    []BlockDevice
}

type BlockDevice struct {
    DeviceName string
    VolumeType string
    VolumeSize int
    IOPS       int
    Encrypted  bool
}

type ASGInstance struct {
    InstanceID     string
    AZ             string
    LifecycleState string // Pending, InService, Terminating, Terminated
    HealthStatus   string // Healthy, Unhealthy
    LaunchTime     time.Time
    WarmPool       bool
}

type ScalingPolicyConfig struct {
    Name        string
    Type        string // TargetTracking, StepScaling, SimpleScaling
    Metric      string
    TargetValue float64
    Steps       []StepAdjustment
    Cooldown    time.Duration
}

type StepAdjustment struct {
    LowerBound  float64
    UpperBound  float64
    Adjustment  int
}

type ScheduledAction struct {
    Name         string
    Schedule     string // cron
    MinSize      int
    MaxSize      int
    Desired      int
}

type WarmPoolConfig struct {
    MinSize      int
    State        string // Stopped, Running, Hibernated
    ReuseOnScale bool
}

type MixedInstancePolicy struct {
    OnDemandBase    int
    OnDemandPercent int
    SpotPercent     int
    InstanceTypes   []string
    SpotAllocation  string // capacity-optimized, lowest-price, price-capacity-optimized
}

func NewAutoScalingGroup(name string, min, max, desired int) *AutoScalingGroup {
    return &AutoScalingGroup{
        Name:            name,
        MinSize:         min,
        MaxSize:         max,
        DesiredCapacity: desired,
        HealthCheckType: "EC2",
        HealthCheckGrace: 300 * time.Second,
        Tags:            make(map[string]string),
    }
}

func (asg *AutoScalingGroup) EvaluateScaling(currentMetric float64) int {
    asg.mu.RLock()
    defer asg.mu.RUnlock()
    
    for _, policy := range asg.Policies {
        switch policy.Type {
        case "TargetTracking":
            return asg.targetTrackingScale(currentMetric, policy)
        case "StepScaling":
            return asg.stepScale(currentMetric, policy)
        }
    }
    return 0
}

func (asg *AutoScalingGroup) targetTrackingScale(current float64, policy ScalingPolicyConfig) int {
    diff := current - policy.TargetValue
    currentCount := len(asg.Instances)
    
    if diff > 5 { // Scale out if > 5% above target
        newCount := int(math.Ceil(float64(currentCount) * current / policy.TargetValue))
        adjustment := newCount - currentCount
        if currentCount+adjustment > asg.MaxSize {
            adjustment = asg.MaxSize - currentCount
        }
        return adjustment
    }
    
    if diff < -10 { // Scale in if > 10% below target
        newCount := int(math.Ceil(float64(currentCount) * current / policy.TargetValue))
        adjustment := newCount - currentCount
        if currentCount+adjustment < asg.MinSize {
            adjustment = asg.MinSize - currentCount
        }
        return adjustment
    }
    
    return 0
}

func (asg *AutoScalingGroup) stepScale(current float64, policy ScalingPolicyConfig) int {
    for _, step := range policy.Steps {
        if current >= step.LowerBound && (step.UpperBound == 0 || current < step.UpperBound) {
            newCount := len(asg.Instances) + step.Adjustment
            if newCount > asg.MaxSize {
                return asg.MaxSize - len(asg.Instances)
            }
            if newCount < asg.MinSize {
                return asg.MinSize - len(asg.Instances)
            }
            return step.Adjustment
        }
    }
    return 0
}

func (asg *AutoScalingGroup) AddInstance(instanceID, az string) {
    asg.mu.Lock()
    defer asg.mu.Unlock()
    
    asg.Instances = append(asg.Instances, &ASGInstance{
        InstanceID:     instanceID,
        AZ:             az,
        LifecycleState: "InService",
        HealthStatus:   "Healthy",
        LaunchTime:     time.Now(),
    })
}

func (asg *AutoScalingGroup) RemoveUnhealthy() []string {
    asg.mu.Lock()
    defer asg.mu.Unlock()
    
    var removed []string
    healthy := make([]*ASGInstance, 0)
    
    for _, inst := range asg.Instances {
        if inst.HealthStatus == "Unhealthy" {
            removed = append(removed, inst.InstanceID)
        } else {
            healthy = append(healthy, inst)
        }
    }
    
    asg.Instances = healthy
    return removed
}

func (asg *AutoScalingGroup) GetAZDistribution() map[string]int {
    asg.mu.RLock()
    defer asg.mu.RUnlock()
    
    dist := make(map[string]int)
    for _, inst := range asg.Instances {
        if inst.LifecycleState == "InService" {
            dist[inst.AZ]++
        }
    }
    return dist
}

// Spot instance price tracker
type SpotPriceTracker struct {
    history map[string][]SpotPricePoint
    mu      sync.RWMutex
}

type SpotPricePoint struct {
    InstanceType string
    AZ           string
    Price        float64
    Timestamp    time.Time
}

func NewSpotPriceTracker() *SpotPriceTracker {
    return &SpotPriceTracker{
        history: make(map[string][]SpotPricePoint),
    }
}

func (t *SpotPriceTracker) RecordPrice(instanceType, az string, price float64) {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    key := instanceType + ":" + az
    t.history[key] = append(t.history[key], SpotPricePoint{
        InstanceType: instanceType,
        AZ:           az,
        Price:        price,
        Timestamp:    time.Now(),
    })
}

func (t *SpotPriceTracker) GetCheapestAZ(instanceType string) (string, float64) {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    cheapestAZ := ""
    cheapestPrice := math.MaxFloat64
    
    for key, points := range t.history {
        if !strings.HasPrefix(key, instanceType+":") {
            continue
        }
        if len(points) > 0 {
            latest := points[len(points)-1]
            if latest.Price < cheapestPrice {
                cheapestPrice = latest.Price
                cheapestAZ = latest.AZ
            }
        }
    }
    
    return cheapestAZ, cheapestPrice
}

func (t *SpotPriceTracker) GetInterruptionRisk(instanceType, az string) string {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    key := instanceType + ":" + az
    points := t.history[key]
    
    if len(points) < 10 {
        return "unknown"
    }
    
    // Analyze price volatility
    recent := points[len(points)-10:]
    var sum, sumSq float64
    for _, p := range recent {
        sum += p.Price
        sumSq += p.Price * p.Price
    }
    mean := sum / 10
    variance := sumSq/10 - mean*mean
    cv := math.Sqrt(variance) / mean
    
    if cv < 0.1 {
        return "low"
    } else if cv < 0.3 {
        return "medium"
    }
    return "high"
}

// Cost calculator
type EC2CostCalculator struct {
    pricing map[string]float64 // instance type -> hourly on-demand price
}

func NewEC2CostCalculator() *EC2CostCalculator {
    return &EC2CostCalculator{
        pricing: map[string]float64{
            "t3.micro":    0.0104,
            "t3.small":    0.0208,
            "t3.medium":   0.0416,
            "m5.large":    0.096,
            "m5.xlarge":   0.192,
            "m5.2xlarge":  0.384,
            "c5.large":    0.085,
            "c5.xlarge":   0.17,
            "r5.large":    0.126,
            "r5.xlarge":   0.252,
            "m6g.large":   0.077,
            "c6g.large":   0.068,
        },
    }
}

func (c *EC2CostCalculator) MonthlyOnDemand(instanceType string, count int) float64 {
    price, ok := c.pricing[instanceType]
    if !ok {
        return 0
    }
    return price * 730 * float64(count) // 730 hours/month
}

func (c *EC2CostCalculator) MonthlyReserved(instanceType string, count, years int, upfront string) float64 {
    onDemand := c.MonthlyOnDemand(instanceType, count)
    
    var discount float64
    switch {
    case years == 1 && upfront == "AllUpfront":
        discount = 0.40
    case years == 1 && upfront == "PartialUpfront":
        discount = 0.35
    case years == 1 && upfront == "NoUpfront":
        discount = 0.30
    case years == 3 && upfront == "AllUpfront":
        discount = 0.62
    case years == 3 && upfront == "PartialUpfront":
        discount = 0.57
    case years == 3 && upfront == "NoUpfront":
        discount = 0.50
    }
    
    return onDemand * (1 - discount)
}

func (c *EC2CostCalculator) MonthlySavingsPlan(instanceType string, count int) float64 {
    return c.MonthlyOnDemand(instanceType, count) * 0.35 // ~65% of on-demand
}

func (c *EC2CostCalculator) RecommendPurchase(instances map[string]int, steadyPercent float64) map[string]PurchaseRecommendation {
    recommendations := make(map[string]PurchaseRecommendation)
    
    for instType, count := range instances {
        steady := int(math.Floor(float64(count) * steadyPercent))
        burst := count - steady
        
        rec := PurchaseRecommendation{
            InstanceType: instType,
            Reserved:     steady,
            OnDemand:     burst / 2,
            Spot:         burst - burst/2,
            MonthlySavings: c.MonthlyOnDemand(instType, count) -
                c.MonthlyReserved(instType, steady, 1, "PartialUpfront") -
                c.MonthlyOnDemand(instType, burst/2) -
                c.MonthlyOnDemand(instType, burst-burst/2)*0.3,
        }
        
        recommendations[instType] = rec
    }
    
    return recommendations
}

type PurchaseRecommendation struct {
    InstanceType   string
    Reserved       int
    OnDemand       int
    Spot           int
    MonthlySavings float64
}

func generateComputeID() string {
    return fmt.Sprintf("%017x", time.Now().UnixNano())
}`,
				},
			},
		},
	})
}
