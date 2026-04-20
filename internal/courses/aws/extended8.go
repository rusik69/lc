package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2122,
			Title:       "AWS Observability and CI/CD",
			Description: "Master CloudWatch, X-Ray, CloudFormation, CDK, CodePipeline, CodeBuild, CodeDeploy, and AWS observability patterns.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "CloudWatch X-Ray CloudFormation and DevOps on AWS",
					Content: `AWS provides comprehensive observability and CI/CD services for building, deploying, and monitoring applications.

**Amazon CloudWatch:**

Metrics:
  Namespace: AWS/EC2, AWS/RDS, Custom
  Dimensions: InstanceId, AutoScalingGroupName
  Period: 1 second to 1 day
  Statistics: Average, Sum, Min, Max, p99, p95, p50
  Custom Metrics: PutMetricData API
  
  High-Resolution Metrics:
    1-second granularity
    Higher cost
    Use for latency-sensitive apps
  
  Metric Math:
    Aggregate across dimensions
    Formulas: METRICS("m1")/METRICS("m2")
    SEARCH expressions

CloudWatch Alarms:
  States: OK, ALARM, INSUFFICIENT_DATA
  
  Types:
    Metric Alarm: Based on single metric
    Composite Alarm: Combine multiple alarms
    Anomaly Detection: ML-based baselines
    
  Actions:
    SNS notification
    Auto Scaling policy
    EC2 action (stop, terminate, reboot)
    Systems Manager OpsItem
    Lambda function

CloudWatch Logs:
  Log Groups: Container for log streams
  Log Streams: Sequence of log events
  
  Features:
    Log Insights: SQL-like query language
    Metric Filters: Extract metrics from logs
    Subscription Filters: Real-time to Lambda/Kinesis/Firehose
    Cross-Account: Centralize logs
    Logs Live Tail: Real-time streaming
    
  Log Insights Queries:
    fields @timestamp, @message
    | filter @message like /ERROR/
    | stats count() by bin(5m)
    | sort @timestamp desc
    | limit 20
    
  Retention: 1 day to 10 years (or never expire)
  Export: S3 (batch), Firehose (streaming)

CloudWatch Container Insights:
  ECS, EKS, Kubernetes metrics
  Task/pod-level metrics
  Prometheus metrics support
  Performance log events

CloudWatch Application Insights:
  Auto-detect application components
  Auto-configure monitoring
  Problem detection dashboard
  .NET and SQL Server focus

CloudWatch Synthetics (Canaries):
  Scripted checks of endpoints/APIs
  Node.js or Python scripts
  Screenshots and HAR files
  Cron schedules

CloudWatch RUM (Real User Monitoring):
  JavaScript client library
  Page load times, errors, sessions
  Browser, device, geography breakdown

**AWS X-Ray:**

  Distributed tracing
  Request flow visualization
  Performance bottleneck identification
  
  Concepts:
    Segment: Service processing unit
    Subsegment: Downstream call detail
    Trace: End-to-end request path
    Service Map: Visual dependency graph
    Annotations: Indexed key-value pairs (filterable)
    Metadata: Non-indexed additional data
    
  Sampling:
    Reservoir: Fixed number per second
    Rate: Percentage of additional requests
    Default: 1 req/s + 5% additional
    Custom rules: By service, URL pattern
    
  Integration:
    SDK instrumentation (auto/manual)
    AWS services: API Gateway, Lambda, ECS, EKS
    X-Ray daemon: UDP listener -> X-Ray API
    OpenTelemetry collector alternative
    
  X-Ray Insights:
    Anomaly detection
    Root cause analysis
    Fault/error rate trends

**AWS CloudFormation:**

  Infrastructure as Code (IaC)
  JSON or YAML templates
  
  Template Structure:
    AWSTemplateFormatVersion
    Description
    Parameters: User inputs
    Mappings: Static key-value lookups
    Conditions: Conditional resource creation
    Resources: AWS resources (required)
    Outputs: Stack outputs
    
  Key Features:
    Stack: Collection of resources
    StackSets: Deploy across accounts/regions
    Nested Stacks: Reusable templates
    Change Sets: Preview changes before applying
    Drift Detection: Detect manual changes
    
  Intrinsic Functions:
    Ref: Reference parameter or resource
    Fn::GetAtt: Get resource attribute
    Fn::Join: Concatenate strings
    Fn::Sub: String substitution
    Fn::Select: Select from list
    Fn::Split: Split string
    Fn::If: Conditional value
    Fn::ImportValue: Cross-stack reference
    
  Resource Dependencies:
    DependsOn: Explicit dependency
    Ref/GetAtt: Implicit dependency
    
  Update Behaviors:
    No Interruption: Live update
    Some Interruption: Brief disruption
    Replacement: Delete and recreate
    
  Rollback:
    On failure: Automatic rollback
    Rollback triggers: CloudWatch alarms
    Disable rollback for debugging

**AWS CDK (Cloud Development Kit):**

  Define infrastructure in programming languages
  TypeScript, Python, Java, C#, Go
  
  Constructs:
    L1: CFN resources (1:1 mapping)
    L2: Curated with defaults and helpers
    L3: Patterns (complete solutions)
    
  CDK Workflow:
    cdk init: Create project
    cdk synth: Generate CloudFormation
    cdk diff: Preview changes
    cdk deploy: Deploy stack
    cdk destroy: Delete stack
    
  Features:
    Assets: Bundle Lambda code, Docker images
    Context: Environment-specific values
    Aspects: Apply cross-cutting concerns
    Custom Resources: Lambda-backed
    cdk-nag: Security/compliance rules

**AWS CodePipeline:**

  CI/CD orchestration
  
  Stages:
    Source: CodeCommit, GitHub, S3, ECR
    Build: CodeBuild
    Test: CodeBuild, third-party
    Deploy: CodeDeploy, ECS, CloudFormation, S3
    Approval: Manual approval gates
    
  Actions:
    Parallel actions within stage
    Cross-region actions
    Cross-account actions

**AWS CodeBuild:**

  Managed build service
  Build specifications (buildspec.yml)
  
  Phases:
    install: Dependencies
    pre_build: Pre-build commands
    build: Build commands
    post_build: Post-build commands
    
  Features:
    Docker support
    Custom build environments
    VPC access
    Build caching (S3, local)
    Batch builds
    Reports (test, coverage)

**AWS CodeDeploy:**

  Automated deployments
  
  Compute Targets:
    EC2/On-premises
    ECS (Blue/Green)
    Lambda (canary, linear, all-at-once)
    
  Deployment Types:
    In-place: Update instances (EC2)
    Blue/Green: New environment, traffic shift
    
  Deployment Configurations:
    CodeDeployDefault.AllAtOnce
    CodeDeployDefault.HalfAtATime
    CodeDeployDefault.OneAtATime
    Custom: MinimumHealthyHosts
    
  AppSpec File:
    Hooks: BeforeInstall, AfterInstall, ApplicationStart, ValidateService
    ECS: Task definition, container, port
    Lambda: Function name, version, alias

**AWS Systems Manager:**

  Operational management
  
  Key Features:
    Parameter Store: Configuration/secrets
      Standard: Free, 10K parameters
      Advanced: Paid, 100K, policies, higher throughput
      SecureString: KMS encrypted
      
    Session Manager: Shell access without SSH
      No inbound ports, no bastion hosts
      Audit via CloudTrail
      Logging to CloudWatch/S3
      
    Run Command: Execute commands remotely
    Patch Manager: Automated patching
    State Manager: Desired state configuration
    Automation: Runbooks for common tasks
    OpsCenter: Operational issue management
    Inventory: Software/config inventory`,
					CodeExamples: `// AWS observability and CI/CD implementations

package main

import (
    "encoding/json"
    "fmt"
    "math"
    "sort"
    "strings"
    "sync"
    "time"
)

// CloudWatch metrics manager
type CloudWatchManager struct {
    metrics    map[string]*MetricData
    alarms     map[string]*CWAlarm
    logGroups  map[string]*CWLogGroup
    mu         sync.RWMutex
}

type MetricData struct {
    Namespace  string
    MetricName string
    Dimensions map[string]string
    DataPoints []MetricDataPoint
}

type MetricDataPoint struct {
    Timestamp time.Time
    Value     float64
    Unit      string
}

type CWAlarm struct {
    Name              string
    Namespace         string
    MetricName        string
    Statistic         string
    Period            int
    EvaluationPeriods int
    Threshold         float64
    ComparisonOp      string
    State             string // OK, ALARM, INSUFFICIENT_DATA
    Actions           []string
    LastEvaluated     time.Time
}

type CWLogGroup struct {
    Name       string
    Retention  int // days
    Streams    map[string]*CWLogStream
    MetricFilters []MetricFilter
}

type CWLogStream struct {
    Name   string
    Events []LogEvent
}

type LogEvent struct {
    Timestamp time.Time
    Message   string
}

type MetricFilter struct {
    Name         string
    Pattern      string
    MetricName   string
    MetricNamespace string
    MetricValue  float64
}

func NewCloudWatchManager() *CloudWatchManager {
    return &CloudWatchManager{
        metrics:   make(map[string]*MetricData),
        alarms:    make(map[string]*CWAlarm),
        logGroups: make(map[string]*CWLogGroup),
    }
}

func (cw *CloudWatchManager) PutMetricData(namespace, metricName string, dimensions map[string]string, value float64, unit string) {
    cw.mu.Lock()
    defer cw.mu.Unlock()
    
    key := fmt.Sprintf("%s/%s/%v", namespace, metricName, dimensions)
    
    metric, exists := cw.metrics[key]
    if !exists {
        metric = &MetricData{
            Namespace:  namespace,
            MetricName: metricName,
            Dimensions: dimensions,
        }
        cw.metrics[key] = metric
    }
    
    metric.DataPoints = append(metric.DataPoints, MetricDataPoint{
        Timestamp: time.Now(),
        Value:     value,
        Unit:      unit,
    })
    
    // Keep last 24h of data
    cutoff := time.Now().Add(-24 * time.Hour)
    filtered := make([]MetricDataPoint, 0)
    for _, dp := range metric.DataPoints {
        if dp.Timestamp.After(cutoff) {
            filtered = append(filtered, dp)
        }
    }
    metric.DataPoints = filtered
}

func (cw *CloudWatchManager) GetMetricStatistics(namespace, metricName, statistic string, period time.Duration, start, end time.Time) []MetricDataPoint {
    cw.mu.RLock()
    defer cw.mu.RUnlock()
    
    var allPoints []MetricDataPoint
    
    for _, metric := range cw.metrics {
        if metric.Namespace != namespace || metric.MetricName != metricName {
            continue
        }
        for _, dp := range metric.DataPoints {
            if dp.Timestamp.After(start) && dp.Timestamp.Before(end) {
                allPoints = append(allPoints, dp)
            }
        }
    }
    
    if len(allPoints) == 0 {
        return nil
    }
    
    // Aggregate by period
    sort.Slice(allPoints, func(i, j int) bool {
        return allPoints[i].Timestamp.Before(allPoints[j].Timestamp)
    })
    
    var results []MetricDataPoint
    periodStart := start
    
    for periodStart.Before(end) {
        periodEnd := periodStart.Add(period)
        var periodValues []float64
        
        for _, dp := range allPoints {
            if dp.Timestamp.After(periodStart) && dp.Timestamp.Before(periodEnd) {
                periodValues = append(periodValues, dp.Value)
            }
        }
        
        if len(periodValues) > 0 {
            var value float64
            switch statistic {
            case "Average":
                sum := 0.0
                for _, v := range periodValues {
                    sum += v
                }
                value = sum / float64(len(periodValues))
            case "Sum":
                for _, v := range periodValues {
                    value += v
                }
            case "Maximum":
                value = periodValues[0]
                for _, v := range periodValues[1:] {
                    if v > value {
                        value = v
                    }
                }
            case "Minimum":
                value = periodValues[0]
                for _, v := range periodValues[1:] {
                    if v < value {
                        value = v
                    }
                }
            case "p99":
                sort.Float64s(periodValues)
                idx := int(float64(len(periodValues)-1) * 0.99)
                value = periodValues[idx]
            }
            
            results = append(results, MetricDataPoint{
                Timestamp: periodStart,
                Value:     value,
            })
        }
        
        periodStart = periodEnd
    }
    
    return results
}

func (cw *CloudWatchManager) PutAlarm(alarm *CWAlarm) {
    cw.mu.Lock()
    defer cw.mu.Unlock()
    alarm.State = "INSUFFICIENT_DATA"
    cw.alarms[alarm.Name] = alarm
}

func (cw *CloudWatchManager) EvaluateAlarms() []string {
    cw.mu.Lock()
    defer cw.mu.Unlock()
    
    var triggered []string
    
    for _, alarm := range cw.alarms {
        // Get metric statistics for evaluation period
        period := time.Duration(alarm.Period) * time.Second
        evalWindow := period * time.Duration(alarm.EvaluationPeriods)
        end := time.Now()
        start := end.Add(-evalWindow)
        
        var values []float64
        for _, metric := range cw.metrics {
            if metric.Namespace == alarm.Namespace && metric.MetricName == alarm.MetricName {
                for _, dp := range metric.DataPoints {
                    if dp.Timestamp.After(start) && dp.Timestamp.Before(end) {
                        values = append(values, dp.Value)
                    }
                }
            }
        }
        
        if len(values) == 0 {
            alarm.State = "INSUFFICIENT_DATA"
            continue
        }
        
        // Calculate statistic
        var statValue float64
        switch alarm.Statistic {
        case "Average":
            sum := 0.0
            for _, v := range values {
                sum += v
            }
            statValue = sum / float64(len(values))
        case "Maximum":
            statValue = values[0]
            for _, v := range values[1:] {
                if v > statValue {
                    statValue = v
                }
            }
        }
        
        // Compare
        inAlarm := false
        switch alarm.ComparisonOp {
        case "GreaterThanThreshold":
            inAlarm = statValue > alarm.Threshold
        case "LessThanThreshold":
            inAlarm = statValue < alarm.Threshold
        case "GreaterThanOrEqualToThreshold":
            inAlarm = statValue >= alarm.Threshold
        }
        
        oldState := alarm.State
        if inAlarm {
            alarm.State = "ALARM"
        } else {
            alarm.State = "OK"
        }
        
        if oldState != alarm.State && alarm.State == "ALARM" {
            triggered = append(triggered, alarm.Name)
        }
        
        alarm.LastEvaluated = time.Now()
    }
    
    return triggered
}

// CloudWatch Logs Insights query engine
type LogInsightsEngine struct {
    logGroups map[string]*CWLogGroup
}

func (e *LogInsightsEngine) Query(logGroupName, query string, start, end time.Time) []map[string]string {
    group, exists := e.logGroups[logGroupName]
    if !exists {
        return nil
    }
    
    // Simple query parser
    var results []map[string]string
    
    filterTerm := ""
    if strings.Contains(query, "filter") {
        parts := strings.SplitN(query, "filter @message like /", 2)
        if len(parts) > 1 {
            filterTerm = strings.TrimSuffix(parts[1], "/")
            filterTerm = strings.Split(filterTerm, "/")[0]
        }
    }
    
    for _, stream := range group.Streams {
        for _, event := range stream.Events {
            if event.Timestamp.Before(start) || event.Timestamp.After(end) {
                continue
            }
            
            if filterTerm != "" && !strings.Contains(event.Message, filterTerm) {
                continue
            }
            
            results = append(results, map[string]string{
                "@timestamp": event.Timestamp.Format(time.RFC3339),
                "@message":   event.Message,
                "@logStream": stream.Name,
            })
        }
    }
    
    // Sort by timestamp desc
    sort.Slice(results, func(i, j int) bool {
        return results[i]["@timestamp"] > results[j]["@timestamp"]
    })
    
    return results
}

// X-Ray trace manager
type XRayTracer struct {
    traces  map[string]*XRayTrace
    mu      sync.RWMutex
}

type XRayTrace struct {
    TraceID   string
    Segments  []*XRaySegment
    Duration  time.Duration
    Status    int
    HasFault  bool
    HasError  bool
}

type XRaySegment struct {
    ID          string
    Name        string
    Service     string
    StartTime   time.Time
    EndTime     time.Time
    Duration    time.Duration
    StatusCode  int
    Fault       bool
    Error       bool
    Throttle    bool
    Subsegments []*XRaySubsegment
    Annotations map[string]string
    Metadata    map[string]interface{}
}

type XRaySubsegment struct {
    ID        string
    Name      string
    Namespace string // aws, remote, local
    StartTime time.Time
    EndTime   time.Time
    Duration  time.Duration
    Fault     bool
    Error     bool
    SQL       *SQLData
}

type SQLData struct {
    URL               string
    DatabaseType      string
    SanitizedQuery    string
}

func NewXRayTracer() *XRayTracer {
    return &XRayTracer{
        traces: make(map[string]*XRayTrace),
    }
}

func (t *XRayTracer) CreateTrace() *XRayTrace {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    traceID := fmt.Sprintf("1-%08x-%024x", time.Now().Unix(), time.Now().UnixNano())
    trace := &XRayTrace{
        TraceID: traceID,
    }
    
    t.traces[traceID] = trace
    return trace
}

func (t *XRayTracer) AddSegment(traceID string, segment *XRaySegment) {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    trace, exists := t.traces[traceID]
    if !exists {
        return
    }
    
    segment.Duration = segment.EndTime.Sub(segment.StartTime)
    trace.Segments = append(trace.Segments, segment)
    
    if segment.Fault {
        trace.HasFault = true
    }
    if segment.Error {
        trace.HasError = true
    }
}

func (t *XRayTracer) GetServiceMap() map[string]ServiceMapNode {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    nodes := make(map[string]ServiceMapNode)
    
    for _, trace := range t.traces {
        for _, seg := range trace.Segments {
            node, exists := nodes[seg.Service]
            if !exists {
                node = ServiceMapNode{
                    Name:  seg.Service,
                    Edges: make(map[string]int),
                }
            }
            
            node.Requests++
            node.AvgLatency = (node.AvgLatency*float64(node.Requests-1) + seg.Duration.Seconds()) / float64(node.Requests)
            
            if seg.Fault {
                node.Faults++
            }
            if seg.Error {
                node.Errors++
            }
            
            for _, sub := range seg.Subsegments {
                if sub.Namespace == "aws" || sub.Namespace == "remote" {
                    node.Edges[sub.Name]++
                }
            }
            
            nodes[seg.Service] = node
        }
    }
    
    return nodes
}

type ServiceMapNode struct {
    Name       string
    Requests   int
    Faults     int
    Errors     int
    AvgLatency float64
    Edges      map[string]int
}

// CloudFormation stack manager
type CFNStackManager struct {
    stacks map[string]*CFNStack
    mu     sync.RWMutex
}

type CFNStack struct {
    Name        string
    Status      string
    Template    json.RawMessage
    Parameters  map[string]string
    Outputs     map[string]string
    Resources   map[string]*CFNResource
    Events      []CFNEvent
    CreatedAt   time.Time
    UpdatedAt   time.Time
    Tags        map[string]string
}

type CFNResource struct {
    LogicalID  string
    Type       string
    PhysicalID string
    Status     string
    Properties map[string]interface{}
}

type CFNEvent struct {
    Timestamp     time.Time
    ResourceStatus string
    ResourceType   string
    LogicalID      string
    Reason         string
}

func NewCFNStackManager() *CFNStackManager {
    return &CFNStackManager{
        stacks: make(map[string]*CFNStack),
    }
}

func (m *CFNStackManager) CreateStack(name string, template json.RawMessage, params map[string]string) (*CFNStack, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if _, exists := m.stacks[name]; exists {
        return nil, fmt.Errorf("stack %s already exists", name)
    }
    
    stack := &CFNStack{
        Name:       name,
        Status:     "CREATE_IN_PROGRESS",
        Template:   template,
        Parameters: params,
        Outputs:    make(map[string]string),
        Resources:  make(map[string]*CFNResource),
        Tags:       make(map[string]string),
        CreatedAt:  time.Now(),
    }
    
    stack.Events = append(stack.Events, CFNEvent{
        Timestamp:      time.Now(),
        ResourceStatus: "CREATE_IN_PROGRESS",
        ResourceType:   "AWS::CloudFormation::Stack",
        LogicalID:      name,
    })
    
    m.stacks[name] = stack
    
    // Simulate completion
    stack.Status = "CREATE_COMPLETE"
    stack.Events = append(stack.Events, CFNEvent{
        Timestamp:      time.Now(),
        ResourceStatus: "CREATE_COMPLETE",
        ResourceType:   "AWS::CloudFormation::Stack",
        LogicalID:      name,
    })
    
    return stack, nil
}

func (m *CFNStackManager) DeleteStack(name string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    stack, exists := m.stacks[name]
    if !exists {
        return fmt.Errorf("stack %s not found", name)
    }
    
    stack.Status = "DELETE_COMPLETE"
    delete(m.stacks, name)
    return nil
}

func (m *CFNStackManager) DetectDrift(name string) []DriftResult {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    stack, exists := m.stacks[name]
    if !exists {
        return nil
    }
    
    var results []DriftResult
    for _, res := range stack.Resources {
        results = append(results, DriftResult{
            LogicalID:   res.LogicalID,
            ResourceType: res.Type,
            DriftStatus: "IN_SYNC",
        })
    }
    return results
}

type DriftResult struct {
    LogicalID    string
    ResourceType string
    DriftStatus  string // IN_SYNC, MODIFIED, DELETED
    Differences  []PropertyDiff
}

type PropertyDiff struct {
    PropertyPath string
    Expected     string
    Actual       string
    DiffType     string // ADD, REMOVE, NOT_EQUAL
}`,
				},
			},
		},
	})
}
