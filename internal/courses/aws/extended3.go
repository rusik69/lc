package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2117,
			Title:       "AWS Container Services and EKS",
			Description: "Master Amazon ECS, EKS, Fargate, ECR, App Runner, and container orchestration patterns on AWS.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "ECS EKS Fargate and Container Orchestration",
					Content: `AWS provides multiple container orchestration services for deploying and managing containerized applications.

**Amazon ECS (Elastic Container Service):**

Architecture:
  Cluster: Logical grouping of services/tasks
  Task Definition: Blueprint for containers
  Task: Running instance of task definition
  Service: Maintains desired task count
  Container Instance: EC2 instance in cluster (EC2 launch type)

Launch Types:
  EC2: You manage EC2 instances
    Control over instance type
    GPU, custom AMI support
    Capacity reservations
    Spot instances for cost savings
    
  Fargate: Serverless containers
    No infrastructure management
    Per-vCPU and memory pricing
    Auto-scaling built-in
    Fargate Spot for fault-tolerant workloads

Task Definition:
  Family: Task definition name
  Container Definitions:
    Image: ECR URI or Docker Hub
    CPU/Memory: Hard/soft limits
    Port Mappings: Host to container
    Environment Variables
    Secrets (from Secrets Manager/SSM)
    Log Configuration (awslogs, splunk, fluentd)
    Health Check
    
  Task Role: IAM role for containers
  Execution Role: IAM role for ECS agent
  Network Mode: awsvpc, bridge, host, none
  Volumes: EFS, EBS, bind mount

Service Configuration:
  Desired Count: Number of tasks
  Deployment Strategy:
    Rolling Update: Replace tasks gradually
    Blue/Green (CodeDeploy): Shift traffic
  Load Balancer: ALB or NLB integration
  Auto Scaling:
    Target Tracking: CPU, memory, custom metric
    Step Scaling: CloudWatch alarm-based
    Scheduled Scaling: Time-based
  Service Discovery: Cloud Map integration

**Amazon EKS (Elastic Kubernetes Service):**

Architecture:
  Control Plane: AWS-managed (multi-AZ)
  Data Plane: Your worker nodes
  
  Node Types:
    Managed Node Groups:
      AWS manages EC2 lifecycle
      Automatic AMI updates
      Spot instance support
    
    Self-Managed Nodes:
      Full EC2 control
      Custom AMIs
      GPU instances
    
    Fargate Profiles:
      Serverless pods
      Per-pod isolation
      No node management
      
EKS Add-ons:
  VPC CNI: Pod networking with VPC IPs
  CoreDNS: Cluster DNS
  kube-proxy: Service proxy
  EBS CSI Driver: Persistent volumes
  EFS CSI Driver: Shared file storage
  
EKS Networking:
  VPC CNI Plugin:
    Each pod gets VPC IP address
    Native VPC networking
    Security groups for pods
    Prefix delegation for high pod density
    
  Service Types:
    ClusterIP: Internal only
    NodePort: External via node port
    LoadBalancer: Creates NLB/CLB
    
  AWS Load Balancer Controller:
    Ingress -> ALB
    Service type LoadBalancer -> NLB
    TargetGroupBinding
    WAF integration

EKS Security:
  IAM Roles for Service Accounts (IRSA):
    Map Kubernetes SA to IAM role
    Pod-level AWS API access
    OIDC provider integration
    
  Pod Security:
    Security contexts
    Pod Security Standards
    OPA Gatekeeper / Kyverno
    
  Network Policies:
    Calico CNI
    VPC CNI network policies
    Cilium

**Amazon ECR (Elastic Container Registry):**

  Private container registry
  Repository policies for access control
  Image scanning: Basic and Enhanced (Inspector)
  Cross-region/cross-account replication
  Lifecycle policies: Expire old images
  Immutable tags: Prevent overwrites
  OCI artifact support (Helm charts)
  Pull-through cache (Docker Hub, ECR Public)

**AWS App Runner:**

  Fully managed container service
  Source: ECR image or source code
  Auto scaling (concurrent requests)
  Built-in load balancing
  VPC connectivity
  Custom domains with TLS
  
  Ideal for: Web apps, APIs, microservices
  Not for: Batch jobs, long-running tasks

**AWS Copilot CLI:**

  Opinionated tool for ECS/Fargate
  Scaffolds infrastructure
  Service types:
    Load Balanced Web Service
    Backend Service
    Worker Service
    Request-Driven Web Service (App Runner)
    Scheduled Job
  
  Pipelines: GitHub/CodeCommit CI/CD
  Environments: Dev, staging, prod

**Container Design Patterns on AWS:**

Sidecar:
  Main container + helper containers
  Log aggregation (Fluent Bit sidecar)
  Service mesh proxy (Envoy)
  Secrets rotation

Ambassador:
  Proxy outbound connections
  Database connection pooling
  Rate limiting

Init Container:
  Run before main container
  Database migration
  Configuration setup
  Dependency check`,
					CodeExamples: `// AWS container service implementations

package main

import (
    "fmt"
    "math"
    "sort"
    "strings"
    "sync"
    "time"
)

// ECS cluster manager
type ECSCluster struct {
    Name          string
    Services      map[string]*ECSService
    TaskDefs      map[string]*ECSTaskDefinition
    Instances     []*ContainerInstance
    mu            sync.RWMutex
}

type ECSService struct {
    Name            string
    TaskDefinition  string
    DesiredCount    int
    RunningCount    int
    PendingCount    int
    LaunchType      string // EC2, FARGATE
    Tasks           []*ECSTask
    LoadBalancer    *ECSLoadBalancer
    DeployConfig    *DeploymentConfig
    AutoScaling     *ECSAutoScaling
    CreatedAt       time.Time
}

type ECSTaskDefinition struct {
    Family     string
    Revision   int
    Containers []ECSContainerDef
    CPU        string // 256, 512, 1024, 2048, 4096
    Memory     string // 512, 1024, ...
    NetworkMode string
    TaskRoleArn string
    ExecRoleArn string
    Volumes    []ECSVolume
}

type ECSContainerDef struct {
    Name       string
    Image      string
    CPU        int
    Memory     int
    MemoryRes  int
    Essential  bool
    PortMappings []PortMapping
    Environment  []EnvVar
    Secrets      []SecretRef
    LogConfig    *LogConfig
    HealthCheck  *HealthCheckDef
    DependsOn    []ContainerDep
}

type PortMapping struct {
    ContainerPort int
    HostPort      int
    Protocol      string
}

type EnvVar struct {
    Name  string
    Value string
}

type SecretRef struct {
    Name      string
    ValueFrom string
}

type LogConfig struct {
    Driver  string
    Options map[string]string
}

type HealthCheckDef struct {
    Command     []string
    Interval    time.Duration
    Timeout     time.Duration
    Retries     int
    StartPeriod time.Duration
}

type ContainerDep struct {
    ContainerName string
    Condition     string // START, COMPLETE, SUCCESS, HEALTHY
}

type ECSVolume struct {
    Name     string
    Type     string // efs, ebs, bind
    Config   map[string]string
}

type ECSTask struct {
    TaskArn    string
    TaskDefArn string
    Status     string // RUNNING, PENDING, STOPPED
    StartedAt  time.Time
    StoppedAt  time.Time
    Containers []TaskContainer
    LaunchType string
}

type TaskContainer struct {
    Name    string
    Image   string
    Status  string
    Health  string
    ExitCode int
}

type ECSLoadBalancer struct {
    TargetGroupArn string
    ContainerName  string
    ContainerPort  int
}

type DeploymentConfig struct {
    MaxPercent        int
    MinHealthyPercent int
    Type              string // ROLLING, BLUE_GREEN
}

type ECSAutoScaling struct {
    MinCapacity int
    MaxCapacity int
    Policies    []ScalingPolicy
}

type ScalingPolicy struct {
    Name       string
    Type       string // TargetTracking, StepScaling
    MetricName string
    TargetValue float64
    ScaleInCooldown  int
    ScaleOutCooldown int
}

type ContainerInstance struct {
    InstanceID    string
    EC2InstanceID string
    Status        string
    CPU           int // Available CPU units
    Memory        int // Available memory
    RunningTasks  int
    RegisteredAt  time.Time
}

func NewECSCluster(name string) *ECSCluster {
    return &ECSCluster{
        Name:     name,
        Services: make(map[string]*ECSService),
        TaskDefs: make(map[string]*ECSTaskDefinition),
    }
}

func (c *ECSCluster) RegisterTaskDefinition(td *ECSTaskDefinition) string {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    td.Revision++
    arn := fmt.Sprintf("arn:aws:ecs:us-east-1:123456789:task-definition/%s:%d",
        td.Family, td.Revision)
    c.TaskDefs[arn] = td
    return arn
}

func (c *ECSCluster) CreateService(name, taskDefArn, launchType string, desired int) (*ECSService, error) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if _, exists := c.Services[name]; exists {
        return nil, fmt.Errorf("service %s already exists", name)
    }
    
    svc := &ECSService{
        Name:           name,
        TaskDefinition: taskDefArn,
        DesiredCount:   desired,
        LaunchType:     launchType,
        CreatedAt:      time.Now(),
        DeployConfig: &DeploymentConfig{
            MaxPercent:        200,
            MinHealthyPercent: 100,
            Type:              "ROLLING",
        },
    }
    
    c.Services[name] = svc
    return svc, nil
}

func (c *ECSCluster) ScaleService(name string, desired int) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    svc, exists := c.Services[name]
    if !exists {
        return fmt.Errorf("service %s not found", name)
    }
    
    if svc.AutoScaling != nil {
        if desired < svc.AutoScaling.MinCapacity || desired > svc.AutoScaling.MaxCapacity {
            return fmt.Errorf("desired count %d outside autoscaling bounds [%d, %d]",
                desired, svc.AutoScaling.MinCapacity, svc.AutoScaling.MaxCapacity)
        }
    }
    
    svc.DesiredCount = desired
    return nil
}

func (c *ECSCluster) GetServiceStatus(name string) map[string]interface{} {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    svc, exists := c.Services[name]
    if !exists {
        return nil
    }
    
    return map[string]interface{}{
        "name":         svc.Name,
        "desired":      svc.DesiredCount,
        "running":      svc.RunningCount,
        "pending":      svc.PendingCount,
        "launchType":   svc.LaunchType,
        "taskDef":      svc.TaskDefinition,
    }
}

// EKS node group manager
type EKSCluster struct {
    Name       string
    Version    string
    NodeGroups map[string]*EKSNodeGroup
    FargateProfiles map[string]*FargateProfile
    AddOns     []EKSAddOn
    mu         sync.RWMutex
}

type EKSNodeGroup struct {
    Name         string
    InstanceTypes []string
    DesiredSize  int
    MinSize      int
    MaxSize      int
    AMIType      string
    DiskSize     int
    Labels       map[string]string
    Taints       []NodeTaint
    CapacityType string // ON_DEMAND, SPOT
    Status       string
    Nodes        []EKSNode
}

type EKSNode struct {
    InstanceID   string
    InstanceType string
    PrivateIP    string
    Status       string
    Pods         int
    CPU          float64
    Memory       float64
}

type NodeTaint struct {
    Key    string
    Value  string
    Effect string // NoSchedule, PreferNoSchedule, NoExecute
}

type FargateProfile struct {
    Name       string
    Selectors  []FargateSelector
    SubnetIDs  []string
    PodExecRole string
}

type FargateSelector struct {
    Namespace string
    Labels    map[string]string
}

type EKSAddOn struct {
    Name    string
    Version string
    Status  string
}

func NewEKSCluster(name, version string) *EKSCluster {
    return &EKSCluster{
        Name:            name,
        Version:         version,
        NodeGroups:      make(map[string]*EKSNodeGroup),
        FargateProfiles: make(map[string]*FargateProfile),
    }
}

func (c *EKSCluster) CreateNodeGroup(ng *EKSNodeGroup) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if _, exists := c.NodeGroups[ng.Name]; exists {
        return fmt.Errorf("node group %s already exists", ng.Name)
    }
    
    ng.Status = "CREATING"
    c.NodeGroups[ng.Name] = ng
    return nil
}

func (c *EKSCluster) ScaleNodeGroup(name string, desired int) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    ng, exists := c.NodeGroups[name]
    if !exists {
        return fmt.Errorf("node group %s not found", name)
    }
    
    if desired < ng.MinSize || desired > ng.MaxSize {
        return fmt.Errorf("desired %d outside bounds [%d, %d]",
            desired, ng.MinSize, ng.MaxSize)
    }
    
    ng.DesiredSize = desired
    return nil
}

func (c *EKSCluster) GetClusterCapacity() (totalCPU, totalMemory float64) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    for _, ng := range c.NodeGroups {
        for _, node := range ng.Nodes {
            totalCPU += node.CPU
            totalMemory += node.Memory
        }
    }
    return
}

// ECR repository manager
type ECRRegistry struct {
    repos map[string]*ECRRepository
    mu    sync.RWMutex
}

type ECRRepository struct {
    Name          string
    URI           string
    Images        []*ECRImage
    ScanOnPush    bool
    ImmutableTags bool
    LifecyclePolicy string
    Policy         string
    EncryptionType string
}

type ECRImage struct {
    Tag       string
    Digest    string
    SizeBytes int64
    PushedAt  time.Time
    ScanStatus string
    Vulnerabilities map[string]int // severity -> count
}

func NewECRRegistry(accountID, region string) *ECRRegistry {
    return &ECRRegistry{
        repos: make(map[string]*ECRRepository),
    }
}

func (r *ECRRegistry) CreateRepository(name string, scanOnPush, immutable bool) *ECRRepository {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    repo := &ECRRepository{
        Name:          name,
        URI:           fmt.Sprintf("123456789.dkr.ecr.us-east-1.amazonaws.com/%s", name),
        ScanOnPush:    scanOnPush,
        ImmutableTags: immutable,
        EncryptionType: "AES256",
    }
    
    r.repos[name] = repo
    return repo
}

func (r *ECRRegistry) PushImage(repoName, tag, digest string, size int64) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    repo, exists := r.repos[repoName]
    if !exists {
        return fmt.Errorf("repository %s not found", repoName)
    }
    
    if repo.ImmutableTags {
        for _, img := range repo.Images {
            if img.Tag == tag {
                return fmt.Errorf("tag %s is immutable and already exists", tag)
            }
        }
    }
    
    image := &ECRImage{
        Tag:       tag,
        Digest:    digest,
        SizeBytes: size,
        PushedAt:  time.Now(),
    }
    
    repo.Images = append(repo.Images, image)
    return nil
}

func (r *ECRRegistry) ApplyLifecyclePolicy(repoName string, maxImages int) int {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    repo, exists := r.repos[repoName]
    if !exists {
        return 0
    }
    
    if len(repo.Images) <= maxImages {
        return 0
    }
    
    // Sort by push time, keep newest
    sort.Slice(repo.Images, func(i, j int) bool {
        return repo.Images[i].PushedAt.After(repo.Images[j].PushedAt)
    })
    
    removed := len(repo.Images) - maxImages
    repo.Images = repo.Images[:maxImages]
    return removed
}`,
				},
			},
		},
	})
}
