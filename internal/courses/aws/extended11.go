package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2125,
			Title:       "AWS Migration, DR, Cost Optimization and Well-Architected",
			Description: "Master AWS migration strategies, disaster recovery patterns, cost optimization techniques, and the Well-Architected Framework.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Migration Disaster Recovery Cost Optimization and Well-Architected",
					Content: `AWS provides comprehensive tools and frameworks for migrating workloads, ensuring resilience, optimizing costs, and building well-architected solutions.

**AWS Migration Strategies (7 Rs):**

Retire:
  Decommission applications no longer needed
  Reduce attack surface and costs
  Often 10-20% of portfolio

Retain:
  Keep applications as-is (not ready to migrate)
  Complex dependencies, compliance needs
  Revisit later

Rehost (Lift and Shift):
  Move to cloud without changes
  AWS Application Migration Service (MGN)
  Fastest migration path
  Automation with CloudEndure
  
  Process:
    Install agent on source servers
    Continuous replication to AWS
    Test instances and cutover
    Minimal downtime

Relocate:
  Move to cloud without purchasing new hardware
  VMware Cloud on AWS
  Same operations, different infrastructure

Replatform (Lift, Tinker, and Shift):
  Minor optimizations during migration
  Examples:
    MySQL → RDS MySQL
    Self-managed Redis → ElastiCache
    On-prem app → Elastic Beanstalk
    
Refactor/Re-architect:
  Redesign using cloud-native features
  Highest short-term cost, highest long-term benefit
  
  Examples:
    Monolith → microservices (ECS/EKS)
    On-prem DB → DynamoDB/Aurora
    Batch → serverless (Lambda/Step Functions)
    File share → S3 + CloudFront

Repurchase:
  Move to SaaS
  CRM → Salesforce
  Email → Amazon WorkMail
  HR → Workday

**AWS Migration Services:**

Application Migration Service (MGN):
  Automated lift-and-shift
  Continuous block-level replication
  Non-disruptive testing
  Cutover with minimal downtime
  Supports physical, virtual, cloud
  
Database Migration Service (DMS):
  Migrate databases with minimal downtime
  Continuous replication
  Schema Conversion Tool (SCT)
  
  Supported:
    Oracle → Aurora PostgreSQL
    SQL Server → Aurora MySQL
    MongoDB → DocumentDB
    Cassandra → Keyspaces

Migration Hub:
  Central tracking for migrations
  Discovery tools
  Strategy recommendations
  Refactor Spaces: Incremental refactoring
  
Transfer Family:
  SFTP, FTPS, FTP to S3/EFS
  
DataSync:
  Online data transfer (10x faster than open source)
  On-premises ↔ AWS
  AWS ↔ AWS (cross-region)
  
Snow Family:
  Snowcone: 8-14 TB, rugged, portable
  Snowball Edge:
    Storage Optimized: 80 TB
    Compute Optimized: 42 TB + GPU
  Snowmobile: 100 PB (truck)
  
  Use cases:
    Large data transfers
    Edge computing
    Disconnected environments

**Disaster Recovery Strategies:**

Backup and Restore:
  Lowest cost, highest RTO/RPO
  S3 cross-region replication
  EBS snapshots
  RDS automated backups
  AMI copies
  
  RPO: Hours
  RTO: Hours
  Cost: $

Pilot Light:
  Core services always running (minimal)
  Database replication active
  Scale up on disaster
  
  Components always on:
    RDS read replica (cross-region)
    Route 53 health checks
    
  Scaled up on failover:
    EC2 instances
    Auto Scaling groups
    
  RPO: Minutes
  RTO: 10s of minutes
  Cost: $$

Warm Standby:
  Scaled-down version in DR region
  All services running (min capacity)
  Scale up on disaster
  
  Always running:
    EC2/ECS (min capacity)
    RDS Multi-AZ (DR region)
    ALB + minimal ASG
    
  RPO: Seconds to minutes
  RTO: Minutes
  Cost: $$$

Multi-Site Active/Active:
  Full production in multiple regions
  Traffic split across regions
  Lowest RTO/RPO
  
  Components:
    Route 53 latency/failover routing
    Global Accelerator
    DynamoDB Global Tables
    Aurora Global Database
    S3 Cross-Region Replication
    CloudFront multi-origin
    
  RPO: Near zero
  RTO: Near zero (automated)
  Cost: $$$$

**DR Testing:**
  
  Game Days: Simulate failures
  Chaos Engineering: AWS Fault Injection Simulator (FIS)
  Runbooks: Documented procedures
  Regular testing: Quarterly minimum

**AWS Cost Optimization:**

Right Sizing:
  AWS Compute Optimizer
  Trusted Advisor recommendations
  CloudWatch utilization metrics
  Tools: Cost Explorer, Compute Optimizer
  
  Process:
    1. Identify underutilized resources
    2. Test smaller instance types
    3. Monitor performance after resize
    4. Repeat periodically

Purchasing Options:
  Reserved Instances / Savings Plans
  Spot Instances for fault-tolerant workloads
  Graviton instances (up to 40% savings)
  
  Strategy:
    Baseline: Reserved/Savings Plans (60-70%)
    Variable: On-Demand (10-20%)
    Fault-tolerant: Spot (10-20%)

Storage Optimization:
  S3 Intelligent-Tiering
  S3 Lifecycle policies (Standard → IA → Glacier)
  EBS volume right-sizing
  Delete unattached EBS volumes
  gp3 vs gp2 (20% cheaper, better performance)
  
Data Transfer:
  VPC endpoints (avoid NAT Gateway)
  CloudFront for content delivery
  S3 Transfer Acceleration
  Direct Connect for high volume
  
  Cost traps:
    NAT Gateway: $0.045/GB processed
    Cross-AZ: $0.01/GB each way
    Cross-region: $0.02/GB
    Internet egress: $0.09/GB (first 10TB)

Serverless:
  Lambda: Pay per invocation
  Fargate: Pay per vCPU/memory
  Aurora Serverless: Pay per ACU
  DynamoDB On-Demand: Pay per request
  
Monitoring and Governance:
  AWS Cost Explorer: Visualize costs
  AWS Budgets: Alerts on thresholds
  Cost Anomaly Detection: ML-based alerts
  AWS Organizations: Consolidated billing
  Service Control Policies: Restrict services
  Tag policies: Enforce cost allocation tags

**AWS Well-Architected Framework:**

Operational Excellence:
  Perform operations as code
  Make frequent, small, reversible changes
  Refine operations procedures frequently
  Anticipate failure
  Learn from all operational events
  
  Services: CloudFormation, Config, CloudTrail, CloudWatch, X-Ray, Systems Manager

Security:
  Implement a strong identity foundation
  Enable traceability
  Apply security at all layers
  Automate security best practices
  Protect data in transit and at rest
  Keep people away from data
  Prepare for security events
  
  Services: IAM, Organizations, CloudTrail, GuardDuty, Security Hub, KMS, WAF, Shield

Reliability:
  Automatically recover from failure
  Test recovery procedures
  Scale horizontally
  Stop guessing capacity
  Manage change in automation
  
  Services: CloudWatch, Auto Scaling, CloudFormation, S3 (11 9s durability), Route 53 (health checks)

Performance Efficiency:
  Democratize advanced technologies
  Go global in minutes
  Use serverless architectures
  Experiment more often
  Consider mechanical sympathy
  
  Services: Auto Scaling, Lambda, ECS/EKS, ElastiCache, CloudFront, Global Accelerator

Cost Optimization:
  Implement cloud financial management
  Adopt a consumption model
  Measure overall efficiency
  Stop spending on undifferentiated heavy lifting
  Analyze and attribute expenditure
  
  Services: Cost Explorer, Budgets, Trusted Advisor, Compute Optimizer, Savings Plans

Sustainability:
  Understand your impact
  Establish sustainability goals
  Maximize utilization
  Anticipate and adopt new offerings
  Use managed services
  Reduce downstream impact
  
  Services: Compute Optimizer, Graviton, Serverless, Auto Scaling

**Well-Architected Tool:**
  Assessment questionnaire
  Improvement plan
  Milestone tracking
  Custom lenses
  Integration with Trusted Advisor

**AWS Control Tower:**
  Multi-account governance
  Landing Zone: Best-practice multi-account setup
  Guardrails: Preventive (SCP) + Detective (Config rules)
  Account Factory: Automated account provisioning
  Customizations: CloudFormation hooks

**AWS Organizations:**
  Consolidated billing
  Service Control Policies (SCPs)
  Tag policies
  Backup policies
  AI services opt-out policies
  OU hierarchy: Root → OUs → Accounts
  
  Best practices:
    Separate accounts per environment
    Security OU for centralized security
    Sandbox OUs for development
    Shared services account

**Elastic Load Balancing Advanced:**

Application Load Balancer (ALB):
  Layer 7 (HTTP/HTTPS)
  Path-based routing
  Host-based routing
  HTTP header/method routing
  Query string routing
  Source IP routing
  Weighted target groups
  Lambda targets
  gRPC support
  WebSocket support
  Sticky sessions
  Authentication (OIDC, Cognito)
  
  Algorithms:
    Round robin
    Least outstanding requests
    
Network Load Balancer (NLB):
  Layer 4 (TCP/UDP/TLS)
  Ultra-low latency (~100μs)
  Millions of requests/second
  Static IP per AZ
  Elastic IP support
  Preserve source IP
  Long-lived TCP connections
  TLS termination
  UDP support
  PrivateLink integration
  
Gateway Load Balancer (GWLB):
  Layer 3 (IP)
  Transparent network gateway
  Third-party virtual appliances
  GENEVE protocol
  Cross-AZ load balancing
  
  Use cases:
    Firewalls
    IDS/IPS
    Deep packet inspection
    Network monitoring

Classic Load Balancer (CLB):
  Legacy, Layer 4/7
  Not recommended for new deployments

**Global Accelerator:**
  AWS global network
  Static anycast IPs
  Intelligent routing
  Health checking
  Endpoint weights
  Client affinity
  DDoS protection
  
  vs CloudFront:
    GA: TCP/UDP, non-HTTP
    CloudFront: HTTP/S, caching
    GA: Gaming, IoT, VoIP
    CloudFront: Static content, APIs`,
					CodeExamples: `// AWS migration, DR, cost optimization implementations

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

// Migration tracker
type MigrationTracker struct {
    applications map[string]*MigrationApp
    waves        []*MigrationWave
    mu           sync.RWMutex
}

type MigrationApp struct {
    Name          string
    Strategy      string // Retire, Retain, Rehost, Relocate, Replatform, Refactor, Repurchase
    Source        string
    Target        string
    Status        string // Assessed, InProgress, Completed, Failed
    Dependencies  []string
    Complexity    string // Low, Medium, High
    BusinessValue string // Low, Medium, High, Critical
    WaveID        string
    StartDate     time.Time
    EndDate       time.Time
    Notes         string
}

type MigrationWave struct {
    ID           string
    Name         string
    Applications []string
    StartDate    time.Time
    EndDate      time.Time
    Status       string
}

func NewMigrationTracker() *MigrationTracker {
    return &MigrationTracker{
        applications: make(map[string]*MigrationApp),
    }
}

func (m *MigrationTracker) AddApplication(app *MigrationApp) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.applications[app.Name] = app
}

func (m *MigrationTracker) AssessStrategy(appName string) (string, string) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    app, exists := m.applications[appName]
    if !exists {
        return "", "application not found"
    }
    
    // Decision tree for migration strategy
    switch {
    case app.BusinessValue == "Low" && app.Complexity == "High":
        return "Retire", "Low business value with high complexity - consider decommissioning"
    case app.BusinessValue == "Low":
        return "Retain", "Low business value - retain and reassess later"
    case app.Complexity == "Low":
        return "Rehost", "Low complexity - lift and shift for quick migration"
    case app.Complexity == "Medium" && app.BusinessValue == "Medium":
        return "Replatform", "Medium complexity - optimize during migration"
    case app.Complexity == "High" && app.BusinessValue == "Critical":
        return "Refactor", "Critical app with high complexity - redesign for cloud-native"
    default:
        return "Replatform", "Default recommendation - replatform with minor optimizations"
    }
}

func (m *MigrationTracker) PlanWaves() []*MigrationWave {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    // Group apps by complexity and dependencies
    var retired, simple, medium, complex []*MigrationApp
    
    for _, app := range m.applications {
        switch app.Strategy {
        case "Retire", "Retain":
            retired = append(retired, app)
        case "Rehost":
            simple = append(simple, app)
        case "Replatform":
            medium = append(medium, app)
        case "Refactor":
            complex = append(complex, app)
        }
    }
    
    waves := make([]*MigrationWave, 0)
    waveNum := 0
    
    // Wave 0: Retire/Retain
    if len(retired) > 0 {
        wave := &MigrationWave{
            ID:   fmt.Sprintf("wave-%d", waveNum),
            Name: "Retire and Retain",
        }
        for _, app := range retired {
            wave.Applications = append(wave.Applications, app.Name)
            app.WaveID = wave.ID
        }
        waves = append(waves, wave)
        waveNum++
    }
    
    // Subsequent waves: batch by type
    batchSize := 5
    for i := 0; i < len(simple); i += batchSize {
        end := i + batchSize
        if end > len(simple) {
            end = len(simple)
        }
        wave := &MigrationWave{
            ID:   fmt.Sprintf("wave-%d", waveNum),
            Name: fmt.Sprintf("Rehost Batch %d", waveNum),
        }
        for _, app := range simple[i:end] {
            wave.Applications = append(wave.Applications, app.Name)
            app.WaveID = wave.ID
        }
        waves = append(waves, wave)
        waveNum++
    }
    
    for _, app := range medium {
        wave := &MigrationWave{
            ID:           fmt.Sprintf("wave-%d", waveNum),
            Name:         fmt.Sprintf("Replatform: %s", app.Name),
            Applications: []string{app.Name},
        }
        app.WaveID = wave.ID
        waves = append(waves, wave)
        waveNum++
    }
    
    for _, app := range complex {
        wave := &MigrationWave{
            ID:           fmt.Sprintf("wave-%d", waveNum),
            Name:         fmt.Sprintf("Refactor: %s", app.Name),
            Applications: []string{app.Name},
        }
        app.WaveID = wave.ID
        waves = append(waves, wave)
        waveNum++
    }
    
    m.waves = waves
    return waves
}

func (m *MigrationTracker) GetProgress() map[string]int {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    progress := map[string]int{
        "total":      0,
        "assessed":   0,
        "inProgress": 0,
        "completed":  0,
        "failed":     0,
    }
    
    for _, app := range m.applications {
        progress["total"]++
        switch app.Status {
        case "Assessed":
            progress["assessed"]++
        case "InProgress":
            progress["inProgress"]++
        case "Completed":
            progress["completed"]++
        case "Failed":
            progress["failed"]++
        }
    }
    
    return progress
}

// Disaster Recovery manager
type DRManager struct {
    strategy     string
    primaryRegion string
    drRegion      string
    resources     map[string]*DRResource
    runbooks      []*DRRunbook
    lastTest      time.Time
    mu            sync.RWMutex
}

type DRResource struct {
    Name         string
    Type         string // EC2, RDS, S3, DynamoDB, Route53
    PrimaryARN   string
    DRStatus     string // Replicated, Standby, Active, NotConfigured
    RPO          time.Duration
    RTO          time.Duration
    ReplicationType string // Sync, Async, Snapshot
    LastSync     time.Time
}

type DRRunbook struct {
    Name   string
    Steps  []DRStep
    Tested bool
    LastRun time.Time
}

type DRStep struct {
    Order       int
    Description string
    Action      string
    Automated   bool
    EstimatedMin int
}

func NewDRManager(strategy, primary, dr string) *DRManager {
    return &DRManager{
        strategy:      strategy,
        primaryRegion: primary,
        drRegion:      dr,
        resources:     make(map[string]*DRResource),
    }
}

func (d *DRManager) AddResource(resource *DRResource) {
    d.mu.Lock()
    defer d.mu.Unlock()
    d.resources[resource.Name] = resource
}

func (d *DRManager) EstimateRTORPO() (time.Duration, time.Duration) {
    d.mu.RLock()
    defer d.mu.RUnlock()
    
    var maxRTO, maxRPO time.Duration
    
    for _, r := range d.resources {
        if r.RTO > maxRTO {
            maxRTO = r.RTO
        }
        if r.RPO > maxRPO {
            maxRPO = r.RPO
        }
    }
    
    // Add overhead based on strategy
    switch d.strategy {
    case "backup-restore":
        maxRTO += 2 * time.Hour
        maxRPO += 1 * time.Hour
    case "pilot-light":
        maxRTO += 30 * time.Minute
        maxRPO += 10 * time.Minute
    case "warm-standby":
        maxRTO += 10 * time.Minute
        maxRPO += 5 * time.Minute
    case "multi-site":
        maxRTO += 1 * time.Minute
        maxRPO += 30 * time.Second
    }
    
    return maxRTO, maxRPO
}

func (d *DRManager) EstimateMonthlyCost(primaryCost float64) float64 {
    switch d.strategy {
    case "backup-restore":
        return primaryCost * 0.05 // ~5% of primary (just backups)
    case "pilot-light":
        return primaryCost * 0.15 // ~15% (core services)
    case "warm-standby":
        return primaryCost * 0.40 // ~40% (scaled-down)
    case "multi-site":
        return primaryCost * 1.0 // ~100% (full duplicate)
    }
    return 0
}

func (d *DRManager) SimulateFailover() *FailoverReport {
    d.mu.Lock()
    defer d.mu.Unlock()
    
    report := &FailoverReport{
        Timestamp:  time.Now(),
        Strategy:   d.strategy,
        FromRegion: d.primaryRegion,
        ToRegion:   d.drRegion,
        Steps:      make([]FailoverStep, 0),
    }
    
    var totalTime time.Duration
    
    // DNS failover
    dnsTime := 60 * time.Second
    report.Steps = append(report.Steps, FailoverStep{
        Name:     "DNS Failover (Route 53)",
        Duration: dnsTime,
        Status:   "Success",
    })
    totalTime += dnsTime
    
    // Resource activation
    for name, r := range d.resources {
        var activationTime time.Duration
        
        switch d.strategy {
        case "backup-restore":
            activationTime = r.RTO
        case "pilot-light":
            if r.DRStatus == "Replicated" {
                activationTime = 5 * time.Minute
            } else {
                activationTime = 15 * time.Minute
            }
        case "warm-standby":
            activationTime = 2 * time.Minute
        case "multi-site":
            activationTime = 0
        }
        
        report.Steps = append(report.Steps, FailoverStep{
            Name:     fmt.Sprintf("Activate %s (%s)", name, r.Type),
            Duration: activationTime,
            Status:   "Success",
        })
        
        if activationTime > totalTime {
            totalTime = activationTime
        }
    }
    
    report.TotalTime = totalTime
    report.Success = true
    d.lastTest = time.Now()
    
    return report
}

type FailoverReport struct {
    Timestamp  time.Time
    Strategy   string
    FromRegion string
    ToRegion   string
    Steps      []FailoverStep
    TotalTime  time.Duration
    Success    bool
}

type FailoverStep struct {
    Name     string
    Duration time.Duration
    Status   string
}

// Cost optimization engine
type CostOptimizer struct {
    resources     []*CloudResource
    recommendations []*CostRecommendation
    mu            sync.RWMutex
}

type CloudResource struct {
    ID            string
    Type          string // EC2, RDS, EBS, S3, NAT, etc.
    Name          string
    Region        string
    MonthlyCost   float64
    Utilization   float64 // 0.0 - 1.0
    Tags          map[string]string
    CreatedAt     time.Time
    LastAccessed  time.Time
}

type CostRecommendation struct {
    ResourceID    string
    ResourceType  string
    Category      string // RightSizing, PurchaseOption, Unused, StorageTier
    Description   string
    CurrentCost   float64
    ProjectedCost float64
    Savings       float64
    SavingsPercent float64
    Risk          string // Low, Medium, High
    Effort        string // Low, Medium, High
}

func NewCostOptimizer() *CostOptimizer {
    return &CostOptimizer{}
}

func (c *CostOptimizer) AddResource(resource *CloudResource) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.resources = append(c.resources, resource)
}

func (c *CostOptimizer) Analyze() []*CostRecommendation {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    c.recommendations = nil
    
    for _, r := range c.resources {
        // Check for underutilized EC2
        if r.Type == "EC2" && r.Utilization < 0.3 {
            savings := r.MonthlyCost * 0.5
            c.recommendations = append(c.recommendations, &CostRecommendation{
                ResourceID:    r.ID,
                ResourceType:  r.Type,
                Category:      "RightSizing",
                Description:   fmt.Sprintf("Instance %s is %.0f%% utilized - consider downsizing", r.Name, r.Utilization*100),
                CurrentCost:   r.MonthlyCost,
                ProjectedCost: r.MonthlyCost - savings,
                Savings:       savings,
                SavingsPercent: 50,
                Risk:          "Medium",
                Effort:        "Low",
            })
        }
        
        // Check for unused EBS volumes
        if r.Type == "EBS" && r.Utilization == 0 {
            c.recommendations = append(c.recommendations, &CostRecommendation{
                ResourceID:    r.ID,
                ResourceType:  r.Type,
                Category:      "Unused",
                Description:   fmt.Sprintf("EBS volume %s appears unused - consider deleting", r.Name),
                CurrentCost:   r.MonthlyCost,
                ProjectedCost: 0,
                Savings:       r.MonthlyCost,
                SavingsPercent: 100,
                Risk:          "Low",
                Effort:        "Low",
            })
        }
        
        // Check for old S3 data
        if r.Type == "S3" && time.Since(r.LastAccessed) > 90*24*time.Hour {
            savings := r.MonthlyCost * 0.7
            c.recommendations = append(c.recommendations, &CostRecommendation{
                ResourceID:    r.ID,
                ResourceType:  r.Type,
                Category:      "StorageTier",
                Description:   fmt.Sprintf("S3 bucket %s not accessed in 90+ days - enable Intelligent-Tiering or move to Glacier", r.Name),
                CurrentCost:   r.MonthlyCost,
                ProjectedCost: r.MonthlyCost - savings,
                Savings:       savings,
                SavingsPercent: 70,
                Risk:          "Low",
                Effort:        "Low",
            })
        }
        
        // Check for NAT Gateway optimization
        if r.Type == "NAT" && r.MonthlyCost > 100 {
            savings := r.MonthlyCost * 0.6
            c.recommendations = append(c.recommendations, &CostRecommendation{
                ResourceID:    r.ID,
                ResourceType:  r.Type,
                Category:      "Architecture",
                Description:   fmt.Sprintf("NAT Gateway %s has high data processing costs - consider VPC endpoints for S3/DynamoDB", r.Name),
                CurrentCost:   r.MonthlyCost,
                ProjectedCost: r.MonthlyCost - savings,
                Savings:       savings,
                SavingsPercent: 60,
                Risk:          "Low",
                Effort:        "Medium",
            })
        }
        
        // Graviton migration
        if r.Type == "EC2" && !strings.Contains(r.Name, "g.") {
            savings := r.MonthlyCost * 0.2
            c.recommendations = append(c.recommendations, &CostRecommendation{
                ResourceID:    r.ID,
                ResourceType:  r.Type,
                Category:      "Graviton",
                Description:   fmt.Sprintf("Instance %s could save ~20%% by migrating to Graviton", r.Name),
                CurrentCost:   r.MonthlyCost,
                ProjectedCost: r.MonthlyCost - savings,
                Savings:       savings,
                SavingsPercent: 20,
                Risk:          "Medium",
                Effort:        "Medium",
            })
        }
    }
    
    // Sort by savings descending
    sort.Slice(c.recommendations, func(i, j int) bool {
        return c.recommendations[i].Savings > c.recommendations[j].Savings
    })
    
    return c.recommendations
}

func (c *CostOptimizer) GetTotalSavings() (float64, float64) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    var totalCurrent, totalSavings float64
    for _, r := range c.recommendations {
        totalCurrent += r.CurrentCost
        totalSavings += r.Savings
    }
    
    return totalCurrent, totalSavings
}

func (c *CostOptimizer) GetSavingsByCategory() map[string]float64 {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    categories := make(map[string]float64)
    for _, r := range c.recommendations {
        categories[r.Category] += r.Savings
    }
    return categories
}

// Well-Architected review
type WellArchitectedReview struct {
    WorkloadName string
    Pillars      map[string]*PillarReview
    OverallRisk  string
    CreatedAt    time.Time
}

type PillarReview struct {
    Name           string
    Questions      []*WAQuestion
    HighRiskCount  int
    MediumRiskCount int
    NoRiskCount    int
}

type WAQuestion struct {
    ID           string
    Question     string
    Pillar       string
    BestPractices []string
    Applied      []string
    Risk         string // HIGH_RISK, MEDIUM_RISK, NO_RISK, NOT_APPLICABLE
    Notes        string
    ImprovementPlan string
}

func NewWellArchitectedReview(workload string) *WellArchitectedReview {
    review := &WellArchitectedReview{
        WorkloadName: workload,
        Pillars:      make(map[string]*PillarReview),
        CreatedAt:    time.Now(),
    }
    
    // Initialize pillars
    pillars := []string{
        "OperationalExcellence", "Security", "Reliability",
        "PerformanceEfficiency", "CostOptimization", "Sustainability",
    }
    
    for _, p := range pillars {
        review.Pillars[p] = &PillarReview{Name: p}
    }
    
    return review
}

func (r *WellArchitectedReview) AddQuestion(q *WAQuestion) {
    pillar, exists := r.Pillars[q.Pillar]
    if !exists {
        return
    }
    
    pillar.Questions = append(pillar.Questions, q)
    
    switch q.Risk {
    case "HIGH_RISK":
        pillar.HighRiskCount++
    case "MEDIUM_RISK":
        pillar.MediumRiskCount++
    case "NO_RISK":
        pillar.NoRiskCount++
    }
}

func (r *WellArchitectedReview) CalculateOverallRisk() string {
    totalHigh := 0
    totalMedium := 0
    
    for _, pillar := range r.Pillars {
        totalHigh += pillar.HighRiskCount
        totalMedium += pillar.MediumRiskCount
    }
    
    switch {
    case totalHigh > 5:
        r.OverallRisk = "HIGH"
    case totalHigh > 0 || totalMedium > 5:
        r.OverallRisk = "MEDIUM"
    default:
        r.OverallRisk = "LOW"
    }
    
    return r.OverallRisk
}

func (r *WellArchitectedReview) GetImprovementPlan() []string {
    var plan []string
    
    // High risk items first
    for _, pillar := range r.Pillars {
        for _, q := range pillar.Questions {
            if q.Risk == "HIGH_RISK" {
                plan = append(plan, fmt.Sprintf("[HIGH] %s: %s - %s",
                    pillar.Name, q.Question, q.ImprovementPlan))
            }
        }
    }
    
    // Then medium risk
    for _, pillar := range r.Pillars {
        for _, q := range pillar.Questions {
            if q.Risk == "MEDIUM_RISK" {
                plan = append(plan, fmt.Sprintf("[MEDIUM] %s: %s - %s",
                    pillar.Name, q.Question, q.ImprovementPlan))
            }
        }
    }
    
    return plan
}

// Load balancer configuration
type ALBConfig struct {
    Name          string
    Scheme        string // internet-facing, internal
    Type          string // application, network, gateway
    Listeners     []*LBListener
    TargetGroups  []*LBTargetGroup
    SecurityGroups []string
    Subnets       []string
    AccessLogs    bool
    WAFEnabled    bool
}

type LBListener struct {
    Port     int
    Protocol string
    Rules    []*LBRule
    DefaultAction string
    CertARN  string
}

type LBRule struct {
    Priority   int
    Conditions []LBCondition
    Actions    []LBAction
}

type LBCondition struct {
    Field  string // path-pattern, host-header, http-header, source-ip, query-string
    Values []string
}

type LBAction struct {
    Type           string // forward, redirect, fixed-response, authenticate-oidc
    TargetGroupARN string
    RedirectConfig *RedirectConfig
    FixedResponse  *FixedResponse
}

type RedirectConfig struct {
    Protocol   string
    Port       string
    Host       string
    Path       string
    StatusCode string // HTTP_301, HTTP_302
}

type FixedResponse struct {
    StatusCode  string
    ContentType string
    Body        string
}

type LBTargetGroup struct {
    Name         string
    Protocol     string
    Port         int
    TargetType   string // instance, ip, lambda
    HealthCheck  *LBHealthCheck
    Targets      []*LBTarget
    Algorithm    string // round_robin, least_outstanding_requests
    Stickiness   bool
    StickyDuration int // seconds
    Weight       int
}

type LBHealthCheck struct {
    Protocol            string
    Port                string
    Path                string
    Interval            int
    Timeout             int
    HealthyThreshold    int
    UnhealthyThreshold  int
    Matcher             string
}

type LBTarget struct {
    ID     string
    Port   int
    AZ     string
    Health string
    Weight int
}

func NewALBConfig(name string) *ALBConfig {
    return &ALBConfig{
        Name:   name,
        Scheme: "internet-facing",
        Type:   "application",
    }
}

func (alb *ALBConfig) AddHTTPSListener(certARN string) {
    // HTTP → HTTPS redirect
    alb.Listeners = append(alb.Listeners, &LBListener{
        Port:     80,
        Protocol: "HTTP",
        DefaultAction: "redirect",
    })
    
    // HTTPS listener
    alb.Listeners = append(alb.Listeners, &LBListener{
        Port:     443,
        Protocol: "HTTPS",
        CertARN:  certARN,
        DefaultAction: "forward",
    })
}

func (alb *ALBConfig) AddPathRule(listenerPort int, priority int, pathPattern, targetGroupARN string) {
    for _, l := range alb.Listeners {
        if l.Port == listenerPort {
            l.Rules = append(l.Rules, &LBRule{
                Priority: priority,
                Conditions: []LBCondition{
                    {Field: "path-pattern", Values: []string{pathPattern}},
                },
                Actions: []LBAction{
                    {Type: "forward", TargetGroupARN: targetGroupARN},
                },
            })
            return
        }
    }
}

func (alb *ALBConfig) AddWeightedTargetGroups(listenerPort, priority int, targets map[string]int) {
    for _, l := range alb.Listeners {
        if l.Port == listenerPort {
            var actions []LBAction
            for tgARN, weight := range targets {
                actions = append(actions, LBAction{
                    Type:           "forward",
                    TargetGroupARN: tgARN,
                })
                _ = weight // Weight applied at target group level
            }
            
            l.Rules = append(l.Rules, &LBRule{
                Priority: priority,
                Actions:  actions,
            })
            return
        }
    }
}

// Budget and cost alerting
type AWSBudget struct {
    Name         string
    BudgetType   string // COST, USAGE, RI_UTILIZATION, SAVINGS_PLANS_UTILIZATION
    Amount       float64
    TimeUnit     string // MONTHLY, QUARTERLY, ANNUALLY
    Alerts       []*BudgetAlert
    ActualSpend  float64
    ForecastSpend float64
}

type BudgetAlert struct {
    Threshold    float64 // percentage
    Type         string  // ACTUAL, FORECASTED
    Notification string  // EMAIL, SNS, CHATBOT
    Recipients   []string
}

func NewAWSBudget(name string, amount float64) *AWSBudget {
    return &AWSBudget{
        Name:       name,
        BudgetType: "COST",
        Amount:     amount,
        TimeUnit:   "MONTHLY",
    }
}

func (b *AWSBudget) AddAlert(threshold float64, alertType string, recipients []string) {
    b.Alerts = append(b.Alerts, &BudgetAlert{
        Threshold:    threshold,
        Type:         alertType,
        Notification: "EMAIL",
        Recipients:   recipients,
    })
}

func (b *AWSBudget) CheckAlerts() []string {
    var triggered []string
    
    for _, alert := range b.Alerts {
        var currentPercent float64
        
        switch alert.Type {
        case "ACTUAL":
            currentPercent = (b.ActualSpend / b.Amount) * 100
        case "FORECASTED":
            currentPercent = (b.ForecastSpend / b.Amount) * 100
        }
        
        if currentPercent >= alert.Threshold {
            triggered = append(triggered, fmt.Sprintf(
                "%s alert: %.1f%% threshold exceeded (current: %.1f%%)",
                alert.Type, alert.Threshold, currentPercent))
        }
    }
    
    return triggered
}

// Organizations SCP evaluator
type SCPEvaluator struct {
    policies map[string]*ServiceControlPolicy
    ous      map[string]*OrgUnit
    mu       sync.RWMutex
}

type ServiceControlPolicy struct {
    Name      string
    Effect    string // Allow, Deny
    Actions   []string
    Resources []string
    Conditions map[string]string
}

type OrgUnit struct {
    Name     string
    Parent   string
    Accounts []string
    Policies []string
}

func NewSCPEvaluator() *SCPEvaluator {
    return &SCPEvaluator{
        policies: make(map[string]*ServiceControlPolicy),
        ous:      make(map[string]*OrgUnit),
    }
}

func (e *SCPEvaluator) AddPolicy(name string, policy *ServiceControlPolicy) {
    e.mu.Lock()
    defer e.mu.Unlock()
    e.policies[name] = policy
}

func (e *SCPEvaluator) AttachPolicy(ouName, policyName string) {
    e.mu.Lock()
    defer e.mu.Unlock()
    
    ou, exists := e.ous[ouName]
    if !exists {
        ou = &OrgUnit{Name: ouName}
        e.ous[ouName] = ou
    }
    ou.Policies = append(ou.Policies, policyName)
}

func (e *SCPEvaluator) IsAllowed(ouName, action, resource string) bool {
    e.mu.RLock()
    defer e.mu.RUnlock()
    
    ou, exists := e.ous[ouName]
    if !exists {
        return true // No OU = no restriction
    }
    
    for _, policyName := range ou.Policies {
        policy, exists := e.policies[policyName]
        if !exists {
            continue
        }
        
        if policy.Effect == "Deny" {
            for _, a := range policy.Actions {
                if matchesWildcard(a, action) {
                    for _, r := range policy.Resources {
                        if r == "*" || matchesWildcard(r, resource) {
                            return false
                        }
                    }
                }
            }
        }
    }
    
    return true
}

func matchesWildcard(pattern, value string) bool {
    if pattern == "*" {
        return true
    }
    
    if strings.HasSuffix(pattern, "*") {
        prefix := strings.TrimSuffix(pattern, "*")
        return strings.HasPrefix(value, prefix)
    }
    
    return pattern == value
}

// Global Accelerator simulator
type GlobalAccelerator struct {
    Name           string
    IPAddresses    []string
    Listeners      []*GAListener
    EndpointGroups []*GAEndpointGroup
}

type GAListener struct {
    Protocol string
    Ports    []int
}

type GAEndpointGroup struct {
    Region          string
    Endpoints       []*GAEndpoint
    TrafficDial     float64 // 0.0 to 1.0
    HealthCheckPath string
    HealthCheckPort int
}

type GAEndpoint struct {
    ID     string
    Type   string // ALB, NLB, EC2, EIP
    Weight int
    Health bool
}

func NewGlobalAccelerator(name string) *GlobalAccelerator {
    return &GlobalAccelerator{
        Name: name,
        IPAddresses: []string{
            fmt.Sprintf("75.2.%d.%d", rand.Intn(255), rand.Intn(255)),
            fmt.Sprintf("99.83.%d.%d", rand.Intn(255), rand.Intn(255)),
        },
    }
}

func (ga *GlobalAccelerator) RouteRequest(clientRegion string) *GAEndpoint {
    // Find closest healthy endpoint group
    var bestGroup *GAEndpointGroup
    bestLatency := math.MaxFloat64
    
    regionLatencies := map[string]map[string]float64{
        "us-east-1": {"us-east-1": 5, "us-west-2": 60, "eu-west-1": 80, "ap-southeast-1": 200},
        "eu-west-1": {"us-east-1": 80, "us-west-2": 140, "eu-west-1": 5, "ap-southeast-1": 160},
        "ap-southeast-1": {"us-east-1": 200, "us-west-2": 160, "eu-west-1": 160, "ap-southeast-1": 5},
    }
    
    latencies := regionLatencies[clientRegion]
    
    for _, group := range ga.EndpointGroups {
        if group.TrafficDial == 0 {
            continue
        }
        
        hasHealthy := false
        for _, ep := range group.Endpoints {
            if ep.Health {
                hasHealthy = true
                break
            }
        }
        
        if !hasHealthy {
            continue
        }
        
        latency := latencies[group.Region]
        if latency == 0 {
            latency = 100
        }
        
        if latency < bestLatency {
            bestLatency = latency
            bestGroup = group
        }
    }
    
    if bestGroup == nil {
        return nil
    }
    
    // Weighted selection among healthy endpoints
    var healthyEndpoints []*GAEndpoint
    totalWeight := 0
    for _, ep := range bestGroup.Endpoints {
        if ep.Health {
            healthyEndpoints = append(healthyEndpoints, ep)
            totalWeight += ep.Weight
        }
    }
    
    if len(healthyEndpoints) == 0 {
        return nil
    }
    
    r := rand.Intn(totalWeight)
    for _, ep := range healthyEndpoints {
        r -= ep.Weight
        if r < 0 {
            return ep
        }
    }
    
    return healthyEndpoints[0]
}`,
				},
			},
		},
	})
}
