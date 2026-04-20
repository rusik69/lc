package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2120,
			Title:       "AWS Serverless and Event-Driven Architecture",
			Description: "Master Lambda, API Gateway, Step Functions, EventBridge, SQS, SNS, and serverless application patterns on AWS.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Lambda API Gateway Step Functions and Event-Driven Patterns",
					Content: `AWS serverless services enable building applications without managing infrastructure, scaling automatically from zero to massive throughput.

**AWS Lambda:**

Execution Model:
  Event-driven, stateless compute
  Max execution time: 15 minutes
  Memory: 128 MB to 10,240 MB (10 GB)
  vCPU: Proportional to memory
  Ephemeral storage: 512 MB to 10 GB (/tmp)
  Deployment package: 50 MB (zip), 10 GB (container image)
  Layers: Up to 5 layers, 250 MB total unzipped

Invocation Types:
  Synchronous: RequestResponse (API Gateway, SDK)
  Asynchronous: Event (S3, SNS, EventBridge)
    Built-in retry: 2 retries
    Dead-letter queue: SQS or SNS
    Destinations: On success/failure
  Poll-based: Stream (SQS, Kinesis, DynamoDB Streams)
    Batch size: 1-10000
    Concurrent batches per shard
    Failure handling: bisect batch, max retries

Cold Start:
  First invocation or after idle
  Init phase: Runtime init + handler init
  Optimization:
    Provisioned Concurrency: Pre-warm instances
    SnapStart (Java): Snapshot/restore
    Smaller packages
    Native compilation (GraalVM, Go)
    Init outside handler

Lambda Extensions:
  Internal: Run within execution environment
  External: Run as separate process
  Use cases: Monitoring, security, config
  AWS Parameters and Secrets Lambda Extension

Lambda@Edge / CloudFront Functions:
  Lambda@Edge: Run at CloudFront edge (Node.js, Python)
    Viewer request/response, Origin request/response
    Up to 30 seconds, 10 GB memory
  CloudFront Functions: Lightweight edge compute
    Viewer request/response only
    Up to 2 MB, < 1 ms execution
    HTTP header manipulation, URL rewrites

**Amazon API Gateway:**

REST API:
  Full API lifecycle management
  Resource-based: /users/{id}/orders
  Methods: GET, POST, PUT, DELETE, PATCH
  
  Integration Types:
    Lambda Proxy: Pass entire request to Lambda
    Lambda Custom: Transform request/response
    HTTP Proxy: Forward to HTTP endpoint
    AWS Service: Direct AWS API call
    Mock: Return response without backend
    
  Features:
    Request validation
    Request/response transformation
    API keys and usage plans
    Throttling (per-stage, per-method)
    Caching (300 MB, 0.5-237 GB)
    Custom domain names
    WAF integration
    Mutual TLS

HTTP API:
  Simplified, lower cost (70% cheaper)
  Faster performance (lower latency)
  Lambda and HTTP integrations
  JWT authorizer
  CORS support
  No usage plans or API keys

WebSocket API:
  Full-duplex communication
  Connection management
  Route selection ($connect, $disconnect, $default, custom)
  Use cases: Chat, real-time gaming, notifications

Authorizers:
  IAM: AWS Signature V4
  Cognito: User pool tokens
  Lambda Authorizer: Custom auth logic
    Token-based: Authorization header
    Request-based: Headers, query params, context
    Caching: Policy cached per token

**AWS Step Functions:**

  Visual workflow orchestration
  State machines with JSON definition
  
  State Types:
    Task: Execute work (Lambda, ECS, Glue, SageMaker)
    Choice: Branching logic (if/else)
    Parallel: Execute branches concurrently
    Map: Iterate over collection
    Wait: Delay execution
    Pass: Pass input to output
    Succeed/Fail: Terminal states
    
  Workflow Types:
    Standard: Up to 1 year, exactly-once
    Express: Up to 5 minutes, at-least-once
      Synchronous: Wait for result
      Asynchronous: Fire and forget
      
  Patterns:
    Sequential: Task -> Task -> Task
    Parallel: Fork -> [Tasks] -> Join
    Choice: Decision point -> Branch A / Branch B
    Map: For each item -> Process
    Retry: Automatic retry with backoff
    Catch: Error handling to fallback
    
  SDK Integration:
    200+ AWS services direct integration
    No Lambda needed for AWS API calls
    Optimized Integrations: Response handling
    
  Features:
    Input/Output processing
    ResultPath, ResultSelector
    Execution history (25,000 events)
    Activity tasks (external workers)
    Callback patterns (waitForTaskToken)

**Amazon EventBridge:**

  Serverless event bus
  
  Event Sources:
    AWS services (100+)
    Custom applications
    SaaS partners (Zendesk, Datadog, etc.)
    
  Event Rules:
    Pattern matching on event content
    Schedule (cron or rate expressions)
    
  Targets (18+):
    Lambda, Step Functions, SQS, SNS
    API Gateway, Kinesis, Firehose
    ECS task, CodePipeline, CodeBuild
    EventBridge API Destination (HTTP)
    
  Advanced Features:
    Event Archive: Store and replay events
    Schema Registry: Auto-discover schemas
    Pipes: Point-to-point integration
    Global Endpoints: Multi-region failover
    Dead-letter queues

**Amazon SQS (Simple Queue Service):**

  Standard Queue:
    Unlimited throughput
    At-least-once delivery
    Best-effort ordering
    
  FIFO Queue:
    300 msg/s (3000 with batching)
    Exactly-once processing
    Strict ordering
    Message group ID for parallel processing
    
  Features:
    Visibility Timeout: 0s to 12 hours
    Message Retention: 1 minute to 14 days
    Long Polling: Reduce empty responses (up to 20s)
    Dead-Letter Queue: After N delivery attempts
    Delay Queue: Delay message delivery (0-15 min)
    SSE: KMS or SQS-managed encryption
    Max Message Size: 256 KB
    Extended Client Library: Up to 2 GB (S3)

**Amazon SNS (Simple Notification Service):**

  Pub/sub messaging
  
  Topic Types:
    Standard: High throughput, at-least-once
    FIFO: Ordered, exactly-once, deduplification
    
  Subscriptions:
    Lambda, SQS, HTTP/HTTPS, Email
    SMS, Kinesis Firehose
    Mobile push (APNS, FCM, ADM)
    
  Features:
    Message filtering: Subscribe with filter policy
    Message archiving (Firehose)
    Dead-letter queues
    Cross-account/cross-region

**Serverless Patterns:**

Fan-Out:
  SNS -> [SQS-1, SQS-2, Lambda-1]
  Decouple producers from consumers

Saga (Choreography):
  EventBridge events between services
  Each service emits success/failure event
  Compensating actions on failure

Saga (Orchestration):
  Step Functions coordinates all steps
  Centralized error handling
  Retry and compensation logic

Event Sourcing:
  DynamoDB Streams -> Lambda -> EventBridge
  Immutable event log
  Rebuild state from events

CQRS:
  Write: API -> Lambda -> DynamoDB
  Read: DynamoDB Streams -> Lambda -> ElastiCache
  Separate read and write models`,
					CodeExamples: `// AWS serverless implementations

package main

import (
    "encoding/json"
    "fmt"
    "math/rand"
    "strings"
    "sync"
    "time"
)

// Lambda function manager
type LambdaManager struct {
    functions map[string]*LambdaFunction
    layers    map[string]*LambdaLayer
    mu        sync.RWMutex
}

type LambdaFunction struct {
    Name           string
    ARN            string
    Runtime        string
    Handler        string
    Role           string
    MemorySize     int
    Timeout        int
    Environment    map[string]string
    Layers         []string
    CodeSize       int64
    Version        string
    Aliases        map[string]*LambdaAlias
    Concurrency    *ConcurrencyConfig
    State          string
    LastModified   time.Time
    Invocations    int64
    Duration       time.Duration
    Errors         int64
}

type LambdaAlias struct {
    Name           string
    Version        string
    RoutingConfig  *AliasRoutingConfig
}

type AliasRoutingConfig struct {
    AdditionalVersion string
    Weight           float64 // 0.0 to 1.0 for canary
}

type ConcurrencyConfig struct {
    Reserved      int
    Provisioned   int
}

type LambdaLayer struct {
    Name    string
    ARN     string
    Version int
    Size    int64
}

type LambdaInvocation struct {
    FunctionName string
    Payload      json.RawMessage
    InvocationType string // RequestResponse, Event, DryRun
    Qualifier    string // version, alias, $LATEST
}

type LambdaResponse struct {
    StatusCode    int
    Payload       json.RawMessage
    FunctionError string
    LogResult     string
    ExecutedVersion string
    Duration      time.Duration
}

func NewLambdaManager() *LambdaManager {
    return &LambdaManager{
        functions: make(map[string]*LambdaFunction),
        layers:    make(map[string]*LambdaLayer),
    }
}

func (m *LambdaManager) CreateFunction(name, runtime, handler, role string, memory, timeout int) (*LambdaFunction, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if _, exists := m.functions[name]; exists {
        return nil, fmt.Errorf("function %s already exists", name)
    }
    
    fn := &LambdaFunction{
        Name:         name,
        ARN:          fmt.Sprintf("arn:aws:lambda:us-east-1:123456789:function:%s", name),
        Runtime:      runtime,
        Handler:      handler,
        Role:         role,
        MemorySize:   memory,
        Timeout:      timeout,
        Environment:  make(map[string]string),
        Aliases:      make(map[string]*LambdaAlias),
        Version:      "$LATEST",
        State:        "Active",
        LastModified: time.Now(),
    }
    
    m.functions[name] = fn
    return fn, nil
}

func (m *LambdaManager) Invoke(inv LambdaInvocation) (*LambdaResponse, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    fn, exists := m.functions[inv.FunctionName]
    if !exists {
        return nil, fmt.Errorf("function %s not found", inv.FunctionName)
    }
    
    if fn.State != "Active" {
        return nil, fmt.Errorf("function %s is not active", inv.FunctionName)
    }
    
    fn.Invocations++
    start := time.Now()
    
    // Simulate execution
    duration := time.Duration(50+rand.Intn(200)) * time.Millisecond
    fn.Duration += duration
    
    response := &LambdaResponse{
        StatusCode:      200,
        Payload:         []byte(` + "`" + `{"statusCode": 200, "body": "OK"}` + "`" + `),
        ExecutedVersion: fn.Version,
        Duration:        duration,
    }
    
    // Check alias routing for canary
    if inv.Qualifier != "" {
        if alias, ok := fn.Aliases[inv.Qualifier]; ok {
            if alias.RoutingConfig != nil && rand.Float64() < alias.RoutingConfig.Weight {
                response.ExecutedVersion = alias.RoutingConfig.AdditionalVersion
            }
        }
    }
    
    _ = start
    return response, nil
}

func (m *LambdaManager) CreateAlias(fnName, aliasName, version string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    fn, exists := m.functions[fnName]
    if !exists {
        return fmt.Errorf("function %s not found", fnName)
    }
    
    fn.Aliases[aliasName] = &LambdaAlias{
        Name:    aliasName,
        Version: version,
    }
    return nil
}

func (m *LambdaManager) SetProvisionedConcurrency(fnName string, reserved, provisioned int) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    fn, exists := m.functions[fnName]
    if !exists {
        return fmt.Errorf("function %s not found", fnName)
    }
    
    fn.Concurrency = &ConcurrencyConfig{
        Reserved:    reserved,
        Provisioned: provisioned,
    }
    return nil
}

// SQS queue simulator
type SQSQueue struct {
    Name             string
    URL              string
    Type             string // standard, fifo
    VisibilityTimeout time.Duration
    RetentionPeriod  time.Duration
    DelaySeconds     int
    MaxReceiveCount  int
    DLQArn           string
    Messages         []*SQSMessage
    InFlight         map[string]*SQSMessage
    mu               sync.Mutex
}

type SQSMessage struct {
    MessageID      string
    ReceiptHandle  string
    Body           string
    Attributes     map[string]string
    GroupID        string // FIFO only
    DeduplicationID string // FIFO only
    SentTimestamp  time.Time
    ReceiveCount   int
    FirstReceive   time.Time
    VisibleAt      time.Time
}

func NewSQSQueue(name, queueType string) *SQSQueue {
    return &SQSQueue{
        Name:              name,
        URL:               fmt.Sprintf("https://sqs.us-east-1.amazonaws.com/123456789/%s", name),
        Type:              queueType,
        VisibilityTimeout: 30 * time.Second,
        RetentionPeriod:   4 * 24 * time.Hour,
        MaxReceiveCount:   3,
        InFlight:          make(map[string]*SQSMessage),
    }
}

func (q *SQSQueue) SendMessage(body string, groupID, dedupID string, delay int) (string, error) {
    q.mu.Lock()
    defer q.mu.Unlock()
    
    if q.Type == "fifo" && groupID == "" {
        return "", fmt.Errorf("MessageGroupId required for FIFO queue")
    }
    
    // FIFO deduplication
    if q.Type == "fifo" && dedupID != "" {
        for _, msg := range q.Messages {
            if msg.DeduplicationID == dedupID {
                return msg.MessageID, nil // Duplicate, return existing
            }
        }
    }
    
    msgID := fmt.Sprintf("msg-%d-%d", time.Now().UnixNano(), rand.Intn(10000))
    
    visibleAt := time.Now()
    if delay > 0 {
        visibleAt = visibleAt.Add(time.Duration(delay) * time.Second)
    }
    
    msg := &SQSMessage{
        MessageID:       msgID,
        Body:            body,
        GroupID:         groupID,
        DeduplicationID: dedupID,
        SentTimestamp:   time.Now(),
        VisibleAt:       visibleAt,
        Attributes:      make(map[string]string),
    }
    
    q.Messages = append(q.Messages, msg)
    return msgID, nil
}

func (q *SQSQueue) ReceiveMessages(maxMessages int, waitTime time.Duration) []*SQSMessage {
    q.mu.Lock()
    defer q.mu.Unlock()
    
    now := time.Now()
    var received []*SQSMessage
    
    for i := 0; i < len(q.Messages) && len(received) < maxMessages; i++ {
        msg := q.Messages[i]
        if msg.VisibleAt.After(now) {
            continue
        }
        
        // Generate receipt handle
        msg.ReceiptHandle = fmt.Sprintf("receipt-%d", time.Now().UnixNano())
        msg.ReceiveCount++
        msg.VisibleAt = now.Add(q.VisibilityTimeout)
        
        if msg.ReceiveCount == 1 {
            msg.FirstReceive = now
        }
        
        q.InFlight[msg.ReceiptHandle] = msg
        received = append(received, msg)
    }
    
    return received
}

func (q *SQSQueue) DeleteMessage(receiptHandle string) error {
    q.mu.Lock()
    defer q.mu.Unlock()
    
    msg, exists := q.InFlight[receiptHandle]
    if !exists {
        return fmt.Errorf("invalid receipt handle")
    }
    
    // Remove from messages
    for i, m := range q.Messages {
        if m.MessageID == msg.MessageID {
            q.Messages = append(q.Messages[:i], q.Messages[i+1:]...)
            break
        }
    }
    
    delete(q.InFlight, receiptHandle)
    return nil
}

func (q *SQSQueue) MoveToDLQ(dlq *SQSQueue) int {
    q.mu.Lock()
    defer q.mu.Unlock()
    
    moved := 0
    remaining := make([]*SQSMessage, 0)
    
    for _, msg := range q.Messages {
        if msg.ReceiveCount >= q.MaxReceiveCount {
            dlq.mu.Lock()
            dlq.Messages = append(dlq.Messages, msg)
            dlq.mu.Unlock()
            moved++
        } else {
            remaining = append(remaining, msg)
        }
    }
    
    q.Messages = remaining
    return moved
}

// EventBridge event bus
type EventBridge struct {
    buses map[string]*EventBus
    mu    sync.RWMutex
}

type EventBus struct {
    Name   string
    Rules  []*EventRule
    Events []EventBridgeEvent
}

type EventRule struct {
    Name     string
    Pattern  EventPattern
    Targets  []EventTarget
    State    string // ENABLED, DISABLED
    Schedule string // rate(5 minutes), cron(...)
}

type EventPattern struct {
    Source     []string
    DetailType []string
    Detail    map[string]interface{}
}

type EventTarget struct {
    ID    string
    ARN   string
    Input string // Optional input transformation
}

type EventBridgeEvent struct {
    Source     string
    DetailType string
    Detail    json.RawMessage
    Time      time.Time
    Region    string
    Resources []string
}

func NewEventBridge() *EventBridge {
    return &EventBridge{
        buses: map[string]*EventBus{
            "default": {Name: "default"},
        },
    }
}

func (eb *EventBridge) PutEvents(busName string, events []EventBridgeEvent) error {
    eb.mu.Lock()
    defer eb.mu.Unlock()
    
    bus, exists := eb.buses[busName]
    if !exists {
        return fmt.Errorf("bus %s not found", busName)
    }
    
    for _, event := range events {
        event.Time = time.Now()
        bus.Events = append(bus.Events, event)
    }
    
    return nil
}

func (eb *EventBridge) AddRule(busName string, rule *EventRule) error {
    eb.mu.Lock()
    defer eb.mu.Unlock()
    
    bus, exists := eb.buses[busName]
    if !exists {
        return fmt.Errorf("bus %s not found", busName)
    }
    
    rule.State = "ENABLED"
    bus.Rules = append(bus.Rules, rule)
    return nil
}

func (eb *EventBridge) MatchRules(busName string, event EventBridgeEvent) []*EventRule {
    eb.mu.RLock()
    defer eb.mu.RUnlock()
    
    bus, exists := eb.buses[busName]
    if !exists {
        return nil
    }
    
    var matched []*EventRule
    for _, rule := range bus.Rules {
        if rule.State != "ENABLED" {
            continue
        }
        
        if matchEventPattern(rule.Pattern, event) {
            matched = append(matched, rule)
        }
    }
    
    return matched
}

func matchEventPattern(pattern EventPattern, event EventBridgeEvent) bool {
    if len(pattern.Source) > 0 {
        found := false
        for _, s := range pattern.Source {
            if s == event.Source {
                found = true
                break
            }
        }
        if !found {
            return false
        }
    }
    
    if len(pattern.DetailType) > 0 {
        found := false
        for _, dt := range pattern.DetailType {
            if dt == event.DetailType {
                found = true
                break
            }
        }
        if !found {
            return false
        }
    }
    
    return true
}

// Step Functions state machine
type StepFunctionStateMachine struct {
    Name       string
    ARN        string
    Definition StateMachineDefinition
    Executions map[string]*SFExecution
    mu         sync.RWMutex
}

type StateMachineDefinition struct {
    StartAt string
    States  map[string]SFState
}

type SFState struct {
    Type       string // Task, Choice, Parallel, Map, Wait, Pass, Succeed, Fail
    Resource   string
    Next       string
    End        bool
    Retry      []RetryConfig
    Catch      []CatchConfig
    Choices    []ChoiceRule
    Default    string
    Seconds    int
    Branches   []StateMachineDefinition
}

type RetryConfig struct {
    ErrorEquals     []string
    IntervalSeconds int
    MaxAttempts     int
    BackoffRate     float64
}

type CatchConfig struct {
    ErrorEquals []string
    Next        string
}

type ChoiceRule struct {
    Variable     string
    StringEquals string
    NumericGreaterThan float64
    Next         string
}

type SFExecution struct {
    ID         string
    Status     string // RUNNING, SUCCEEDED, FAILED, TIMED_OUT, ABORTED
    Input      json.RawMessage
    Output     json.RawMessage
    StartTime  time.Time
    StopTime   time.Time
    CurrentState string
    History    []SFHistoryEvent
}

type SFHistoryEvent struct {
    Type      string
    State     string
    Timestamp time.Time
    Input     json.RawMessage
    Output    json.RawMessage
    Error     string
}

func NewStepFunctionStateMachine(name string, def StateMachineDefinition) *StepFunctionStateMachine {
    return &StepFunctionStateMachine{
        Name:       name,
        ARN:        fmt.Sprintf("arn:aws:states:us-east-1:123456789:stateMachine:%s", name),
        Definition: def,
        Executions: make(map[string]*SFExecution),
    }
}

func (sm *StepFunctionStateMachine) StartExecution(input json.RawMessage) string {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    
    execID := fmt.Sprintf("exec-%d", time.Now().UnixNano())
    
    exec := &SFExecution{
        ID:           execID,
        Status:       "RUNNING",
        Input:        input,
        StartTime:    time.Now(),
        CurrentState: sm.Definition.StartAt,
    }
    
    exec.History = append(exec.History, SFHistoryEvent{
        Type:      "ExecutionStarted",
        Timestamp: time.Now(),
        Input:     input,
    })
    
    sm.Executions[execID] = exec
    return execID
}

func (sm *StepFunctionStateMachine) GetExecutionStatus(execID string) *SFExecution {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    return sm.Executions[execID]
}`,
				},
			},
		},
	})
}
