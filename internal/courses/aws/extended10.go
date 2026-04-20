package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2124,
			Title:       "AWS Data Analytics and Machine Learning",
			Description: "Master Kinesis, Athena, Glue, EMR, Lake Formation, SageMaker, QuickSight, and data pipeline architectures.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Data Analytics Services and Architectures",
					Content: `AWS provides a comprehensive suite of data analytics and machine learning services for building end-to-end data platforms.

**Amazon Kinesis:**

Kinesis Data Streams:
  Real-time data streaming
  Shards: Unit of capacity
    Write: 1 MB/s or 1000 records/s per shard
    Read: 2 MB/s per shard
  Retention: 24 hours (default) to 365 days
  
  Producers:
    AWS SDK (PutRecord, PutRecords)
    Kinesis Producer Library (KPL): Batching, compression
    Kinesis Agent: Log file monitoring
    
  Consumers:
    Kinesis Client Library (KCL)
    AWS Lambda
    Kinesis Data Analytics
    
  Enhanced Fan-Out:
    Dedicated 2 MB/s per consumer per shard
    Push model (HTTP/2)
    Lower latency (~70ms vs ~200ms)
    
  Capacity Modes:
    Provisioned: Manage shards manually
    On-demand: Auto-scales (up to 200 MB/s write, 400 MB/s read)
    
  Operations:
    Split shard: Increase capacity
    Merge shards: Decrease capacity
    Resharding: Rebalance data

Kinesis Data Firehose:
  Managed delivery service
  No real-time (buffer: 60s-900s or 1MB-128MB)
  
  Destinations:
    S3, Redshift, OpenSearch, Splunk
    HTTP endpoint (custom)
    3rd party: Datadog, New Relic, MongoDB
    
  Transformations:
    Lambda (custom transformation)
    Format conversion (JSON to Parquet/ORC)
    Dynamic partitioning
    
  Features:
    Compression (GZIP, Snappy, ZIP)
    Encryption (SSE)
    Failed data to S3 backup

Kinesis Data Analytics:
  Apache Flink managed service
  SQL or Flink application
  
  Use cases:
    Streaming ETL
    Continuous metrics
    Real-time dashboards
    Anomaly detection

Kinesis Video Streams:
  Stream video from devices
  ML/analytics processing
  Playback, storage
  WebRTC signaling

**Amazon Athena:**

  Serverless query service for S3
  
  SQL engine: Presto/Trino
  
  Features:
    Query S3 data directly
    No infrastructure to manage
    Pay per query (per TB scanned)
    
  Data Formats:
    CSV, JSON, ORC, Avro, Parquet
    Parquet/ORC recommended (columnar, compressed)
    
  Performance:
    Partition data (e.g., year/month/day)
    Use columnar formats (Parquet/ORC)
    Compress data (Snappy, GZIP)
    Use larger files (>128 MB)
    CTAS: Create Table As Select
    
  Federated Query:
    Query data in RDS, DynamoDB, Redshift, etc.
    Lambda-based connectors
    
  ACID Transactions:
    Apache Iceberg table format
    INSERT, UPDATE, DELETE, MERGE
    Time travel queries
    
  Views:
    Standard views
    Materialized views (preview)

**AWS Glue:**

  Managed ETL service
  
  Components:
    Data Catalog:
      Hive-compatible metastore
      Databases and tables
      Schema discovery
      Integration: Athena, Redshift, EMR
      
    Crawlers:
      Automatically discover schemas
      Populate Data Catalog
      Classifiers: CSV, JSON, Parquet, etc.
      Schedule or on-demand
      
    ETL Jobs:
      Spark-based or Ray-based
      Python or Scala
      Visual ETL editor
      Auto-generates code
      
    Job Bookmarks:
      Track processed data
      Prevent reprocessing
      
    Workflow:
      Orchestrate multiple jobs
      Triggers and conditions
      
  Glue DataBrew:
    Visual data preparation
    250+ transformations
    Data profiling
    Recipe-based
    
  Glue Elastic Views:
    Materialized views across data stores
    
  Glue Schema Registry:
    Schema versioning
    Compatibility checking
    Avro/JSON Schema support
    
  DynamicFrame:
    Extension of Spark DataFrame
    Handles messy data
    Schema on read
    Relationalize: Flatten nested data

**Amazon EMR:**

  Managed Hadoop/Spark clusters
  
  Frameworks:
    Hadoop, Spark, Hive, HBase, Presto, Flink, Pig
    
  Deployment Modes:
    EMR on EC2: Traditional clusters
    EMR on EKS: Run on existing Kubernetes
    EMR Serverless: No infrastructure management
    
  Node Types:
    Primary: Coordinates cluster
    Core: Run tasks + store data (HDFS)
    Task: Run tasks only (no data)
    
  Instance Fleets:
    Multiple instance types per fleet
    On-Demand and Spot mix
    Target capacity-based
    
  Storage:
    HDFS: Local storage (ephemeral)
    EMRFS: S3 as persistent storage
    EBS: Additional local storage
    
  Security:
    Kerberos authentication
    Encryption at rest (LUKS, SSE-S3)
    Encryption in transit (TLS)
    Ranger/Lake Formation integration
    
  Managed Scaling:
    Auto-scale based on workload
    Core+Task node scaling
    Scale-down behavior configurable

**Amazon Redshift:**

  Cloud data warehouse (petabyte-scale)
  
  Architecture:
    Leader Node: Query planning, aggregation
    Compute Nodes: Execute queries, store data
    
  Node Types:
    RA3: Managed storage (separate compute/storage)
    DC2: Dense compute (SSD)
    DS2: Dense storage (HDD, legacy)
    
  Redshift Serverless:
    No cluster management
    Auto-scales
    Pay per RPU (Redshift Processing Unit)
    
  Features:
    Columnar storage
    Massive parallel processing (MPP)
    Result caching
    Concurrency scaling (burst capacity)
    Materialized views
    Federated query (RDS, Aurora)
    
  Distribution Styles:
    AUTO: Redshift chooses
    EVEN: Round-robin
    KEY: Hash on column (joins)
    ALL: Copy to all nodes (small tables)
    
  Sort Keys:
    Compound: Multiple columns in order
    Interleaved: Equal weight to each column
    
  Data Loading:
    COPY command (S3, DynamoDB, EMR)
    Kinesis Data Firehose
    AWS DMS
    
  Redshift Spectrum:
    Query S3 data from Redshift
    Extends Redshift to data lake
    Uses Athena infrastructure
    
  Workload Management (WLM):
    Queue queries by type
    Priority-based
    Auto WLM: ML-powered

**AWS Lake Formation:**

  Data lake management service
  
  Features:
    Centralized governance
    Fine-grained access control (column/row/cell)
    Cross-account sharing
    Tag-based access control (LF-Tags)
    Data filters
    
  Data Sources:
    S3, RDS, Aurora, on-premises databases
    
  Blueprint:
    Pre-built ETL templates
    Database, log file ingestion
    
  Integration:
    Athena, Redshift, EMR, Glue
    Governed tables (ACID on S3)

**Amazon QuickSight:**

  Serverless BI/visualization service
  
  Features:
    SPICE: In-memory acceleration
    ML Insights: Anomaly detection, forecasting
    Natural language queries (Q)
    Embedded analytics
    Row-level/column-level security
    
  Data Sources:
    Athena, Redshift, RDS, Aurora
    S3, OpenSearch, Timestream
    JDBC/ODBC connections
    
  Sharing:
    Dashboards, analyses, datasets
    Namespace isolation

**Amazon SageMaker:**

  Managed ML platform
  
  Components:
    Ground Truth: Data labeling
    Studio: IDE for ML
    Notebooks: Jupyter notebooks
    Processing: Data processing jobs
    Training: Distributed training
    Tuning: Hyperparameter optimization
    Hosting: Model deployment
    
  Built-in Algorithms:
    XGBoost, Linear Learner, K-NN
    Image Classification, Object Detection
    BlazingText, Seq2Seq
    DeepAR (time series)
    Random Cut Forest (anomaly)
    
  Deployment:
    Real-time inference
    Batch transform
    Serverless inference
    Asynchronous inference
    Multi-model endpoints
    
  SageMaker Pipelines:
    ML workflow orchestration
    CI/CD for ML
    Model registry
    
  SageMaker Canvas:
    No-code ML for business users
    AutoML
    
  SageMaker Feature Store:
    Centralized feature repository
    Online/offline store
    Feature sharing
    
  SageMaker Clarify:
    Bias detection
    Model explainability

**Data Pipeline Architectures:**

Batch Analytics Pipeline:
  S3 (raw) → Glue Crawler → Data Catalog
  → Glue ETL → S3 (processed) → Athena/Redshift

Real-time Analytics Pipeline:
  Producers → Kinesis Data Streams
  → Lambda/Kinesis Analytics → DynamoDB/OpenSearch
  → API Gateway → Dashboard

Data Lake Architecture:
  Sources → Lake Formation → S3 (raw/processed/curated)
  → Glue Data Catalog → Athena/Redshift Spectrum/EMR

ML Pipeline:
  S3 → SageMaker Processing → Feature Store
  → SageMaker Training → Model Registry
  → SageMaker Endpoint → Application`,
					CodeExamples: `// AWS data analytics and ML service implementations

package main

import (
    "encoding/json"
    "fmt"
    "math"
    "math/rand"
    "sort"
    "strings"
    "sync"
    "time"
)

// Kinesis Data Streams simulator
type KinesisStream struct {
    Name           string
    Shards         []*Shard
    RetentionHours int
    Mode           string // PROVISIONED, ON_DEMAND
    mu             sync.RWMutex
}

type Shard struct {
    ShardID       string
    StartHash     string
    EndHash       string
    Records       []*KinesisRecord
    SequenceNum   int64
    Parent        string
    Children      []string
    Status        string // OPEN, CLOSED
    mu            sync.Mutex
}

type KinesisRecord struct {
    SequenceNumber string
    PartitionKey   string
    Data           []byte
    Timestamp      time.Time
}

type ShardIterator struct {
    ShardID   string
    Type      string // TRIM_HORIZON, LATEST, AT_SEQUENCE, AT_TIMESTAMP
    Position  int
    StreamName string
}

func NewKinesisStream(name string, shardCount int) *KinesisStream {
    stream := &KinesisStream{
        Name:           name,
        RetentionHours: 24,
        Mode:           "PROVISIONED",
    }
    
    for i := 0; i < shardCount; i++ {
        stream.Shards = append(stream.Shards, &Shard{
            ShardID:   fmt.Sprintf("shardId-%012d", i),
            StartHash: fmt.Sprintf("%032d", i*math.MaxInt64/shardCount),
            EndHash:   fmt.Sprintf("%032d", (i+1)*math.MaxInt64/shardCount),
            Status:    "OPEN",
        })
    }
    
    return stream
}

func (s *KinesisStream) PutRecord(partitionKey string, data []byte) (*KinesisRecord, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    // Hash partition key to find shard
    shard := s.findShard(partitionKey)
    if shard == nil {
        return nil, fmt.Errorf("no open shard available")
    }
    
    shard.mu.Lock()
    defer shard.mu.Unlock()
    
    shard.SequenceNum++
    record := &KinesisRecord{
        SequenceNumber: fmt.Sprintf("%020d", shard.SequenceNum),
        PartitionKey:   partitionKey,
        Data:           data,
        Timestamp:      time.Now(),
    }
    
    shard.Records = append(shard.Records, record)
    return record, nil
}

func (s *KinesisStream) PutRecords(records []struct{ PartitionKey string; Data []byte }) (int, int) {
    success, failed := 0, 0
    
    for _, r := range records {
        _, err := s.PutRecord(r.PartitionKey, r.Data)
        if err != nil {
            failed++
        } else {
            success++
        }
    }
    
    return success, failed
}

func (s *KinesisStream) GetRecords(iterator *ShardIterator, limit int) ([]*KinesisRecord, *ShardIterator) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    var shard *Shard
    for _, sh := range s.Shards {
        if sh.ShardID == iterator.ShardID {
            shard = sh
            break
        }
    }
    
    if shard == nil {
        return nil, nil
    }
    
    shard.mu.Lock()
    defer shard.mu.Unlock()
    
    start := iterator.Position
    if start >= len(shard.Records) {
        return nil, &ShardIterator{
            ShardID:    iterator.ShardID,
            Position:   start,
            StreamName: iterator.StreamName,
        }
    }
    
    end := start + limit
    if end > len(shard.Records) {
        end = len(shard.Records)
    }
    
    records := shard.Records[start:end]
    
    return records, &ShardIterator{
        ShardID:    iterator.ShardID,
        Position:   end,
        StreamName: iterator.StreamName,
    }
}

func (s *KinesisStream) findShard(partitionKey string) *Shard {
    // Simple hash-based shard selection
    hash := 0
    for _, c := range partitionKey {
        hash = (hash*31 + int(c)) % len(s.Shards)
    }
    
    if hash < 0 {
        hash = -hash
    }
    
    idx := hash % len(s.Shards)
    if s.Shards[idx].Status == "OPEN" {
        return s.Shards[idx]
    }
    return nil
}

func (s *KinesisStream) SplitShard(shardID string) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    var target *Shard
    for _, sh := range s.Shards {
        if sh.ShardID == shardID {
            target = sh
            break
        }
    }
    
    if target == nil {
        return fmt.Errorf("shard %s not found", shardID)
    }
    
    target.Status = "CLOSED"
    
    childID1 := fmt.Sprintf("shardId-%012d", len(s.Shards))
    childID2 := fmt.Sprintf("shardId-%012d", len(s.Shards)+1)
    
    s.Shards = append(s.Shards,
        &Shard{ShardID: childID1, Parent: shardID, Status: "OPEN"},
        &Shard{ShardID: childID2, Parent: shardID, Status: "OPEN"},
    )
    
    target.Children = []string{childID1, childID2}
    return nil
}

// Kinesis Consumer with checkpointing
type KinesisConsumer struct {
    StreamName  string
    AppName     string
    Workers     int
    checkpoints map[string]string // shardID -> sequence number
    mu          sync.Mutex
}

func NewKinesisConsumer(stream, app string, workers int) *KinesisConsumer {
    return &KinesisConsumer{
        StreamName:  stream,
        AppName:     app,
        Workers:     workers,
        checkpoints: make(map[string]string),
    }
}

func (c *KinesisConsumer) Checkpoint(shardID, sequenceNumber string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.checkpoints[shardID] = sequenceNumber
}

func (c *KinesisConsumer) GetCheckpoint(shardID string) string {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.checkpoints[shardID]
}

// Firehose delivery stream simulator
type FirehoseStream struct {
    Name          string
    Destination   string // S3, Redshift, OpenSearch
    BufferSize    int    // MB
    BufferInterval int   // seconds
    Compression   string
    Transform     func([]byte) ([]byte, error)
    buffer        [][]byte
    delivered     int
    failed        int
    mu            sync.Mutex
}

func NewFirehoseStream(name, dest string) *FirehoseStream {
    return &FirehoseStream{
        Name:           name,
        Destination:    dest,
        BufferSize:     5,    // 5 MB default
        BufferInterval: 300,  // 5 min default
        Compression:    "GZIP",
    }
}

func (f *FirehoseStream) PutRecord(data []byte) error {
    f.mu.Lock()
    defer f.mu.Unlock()
    
    if f.Transform != nil {
        transformed, err := f.Transform(data)
        if err != nil {
            f.failed++
            return err
        }
        data = transformed
    }
    
    f.buffer = append(f.buffer, data)
    
    // Check if buffer should be flushed
    totalSize := 0
    for _, b := range f.buffer {
        totalSize += len(b)
    }
    
    if totalSize >= f.BufferSize*1024*1024 {
        return f.flush()
    }
    
    return nil
}

func (f *FirehoseStream) flush() error {
    if len(f.buffer) == 0 {
        return nil
    }
    
    f.delivered += len(f.buffer)
    f.buffer = nil
    return nil
}

func (f *FirehoseStream) GetMetrics() map[string]int {
    f.mu.Lock()
    defer f.mu.Unlock()
    
    return map[string]int{
        "delivered":    f.delivered,
        "failed":       f.failed,
        "buffered":     len(f.buffer),
    }
}

// Athena query simulator
type AthenaService struct {
    databases  map[string]*GlueDatabase
    queries    map[string]*AthenaQuery
    mu         sync.RWMutex
}

type GlueDatabase struct {
    Name   string
    Tables map[string]*GlueTable
}

type GlueTable struct {
    Name         string
    Location     string // S3 path
    Format       string // CSV, JSON, Parquet, ORC
    Columns      []GlueColumn
    Partitions   []GluePartition
    RecordCount  int64
    SizeBytes    int64
    Compressed   bool
}

type GlueColumn struct {
    Name    string
    Type    string
    Comment string
}

type GluePartition struct {
    Values   []string
    Location string
}

type AthenaQuery struct {
    ID            string
    Database      string
    SQL           string
    State         string // QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED
    BytesScanned  int64
    ExecutionTime time.Duration
    ResultCount   int
    SubmitTime    time.Time
    CompletionTime time.Time
    Cost          float64
}

func NewAthenaService() *AthenaService {
    return &AthenaService{
        databases: make(map[string]*GlueDatabase),
        queries:   make(map[string]*AthenaQuery),
    }
}

func (a *AthenaService) CreateDatabase(name string) {
    a.mu.Lock()
    defer a.mu.Unlock()
    
    a.databases[name] = &GlueDatabase{
        Name:   name,
        Tables: make(map[string]*GlueTable),
    }
}

func (a *AthenaService) CreateTable(database string, table *GlueTable) error {
    a.mu.Lock()
    defer a.mu.Unlock()
    
    db, exists := a.databases[database]
    if !exists {
        return fmt.Errorf("database %s not found", database)
    }
    
    db.Tables[table.Name] = table
    return nil
}

func (a *AthenaService) StartQuery(database, sql string) (string, error) {
    a.mu.Lock()
    defer a.mu.Unlock()
    
    db, exists := a.databases[database]
    if !exists {
        return "", fmt.Errorf("database %s not found", database)
    }
    
    queryID := fmt.Sprintf("query-%d", time.Now().UnixNano())
    
    // Estimate bytes scanned based on table format
    var bytesScanned int64
    for _, table := range db.Tables {
        if strings.Contains(strings.ToUpper(sql), strings.ToUpper(table.Name)) {
            bytesScanned += table.SizeBytes
            if table.Format == "Parquet" || table.Format == "ORC" {
                bytesScanned = bytesScanned / 5 // Columnar is ~5x more efficient
            }
            if table.Compressed {
                bytesScanned = bytesScanned / 3
            }
        }
    }
    
    cost := float64(bytesScanned) / (1024 * 1024 * 1024 * 1024) * 5.0 // $5 per TB
    
    query := &AthenaQuery{
        ID:           queryID,
        Database:     database,
        SQL:          sql,
        State:        "SUCCEEDED",
        BytesScanned: bytesScanned,
        Cost:         cost,
        SubmitTime:   time.Now(),
        CompletionTime: time.Now(),
    }
    
    a.queries[queryID] = query
    return queryID, nil
}

func (a *AthenaService) GetQuery(queryID string) (*AthenaQuery, error) {
    a.mu.RLock()
    defer a.mu.RUnlock()
    
    q, exists := a.queries[queryID]
    if !exists {
        return nil, fmt.Errorf("query %s not found", queryID)
    }
    return q, nil
}

// Glue ETL job simulator
type GlueETLJob struct {
    Name          string
    Role          string
    Script        string
    Workers       int
    WorkerType    string // G.1X, G.2X, G.025X
    Timeout       int    // minutes
    MaxRetries    int
    Bookmark      *GlueBookmark
    Runs          []*GlueJobRun
    mu            sync.Mutex
}

type GlueBookmark struct {
    Key   string
    Value string
}

type GlueJobRun struct {
    ID         string
    State      string // STARTING, RUNNING, SUCCEEDED, FAILED, STOPPED
    StartTime  time.Time
    EndTime    time.Time
    DPUSeconds float64
    Error      string
}

func NewGlueETLJob(name string, workers int) *GlueETLJob {
    return &GlueETLJob{
        Name:       name,
        Workers:    workers,
        WorkerType: "G.1X",
        Timeout:    2880,
        MaxRetries: 0,
    }
}

func (j *GlueETLJob) StartRun() *GlueJobRun {
    j.mu.Lock()
    defer j.mu.Unlock()
    
    run := &GlueJobRun{
        ID:        fmt.Sprintf("jr_%d", time.Now().UnixNano()),
        State:     "RUNNING",
        StartTime: time.Now(),
    }
    
    j.Runs = append(j.Runs, run)
    return run
}

func (j *GlueETLJob) CompleteRun(runID string, success bool, dpuSeconds float64) error {
    j.mu.Lock()
    defer j.mu.Unlock()
    
    for _, run := range j.Runs {
        if run.ID == runID {
            run.EndTime = time.Now()
            run.DPUSeconds = dpuSeconds
            if success {
                run.State = "SUCCEEDED"
            } else {
                run.State = "FAILED"
            }
            return nil
        }
    }
    
    return fmt.Errorf("run %s not found", runID)
}

func (j *GlueETLJob) EstimateCost(dpuSeconds float64) float64 {
    // $0.44 per DPU-hour
    return dpuSeconds / 3600 * 0.44
}

// EMR Cluster simulator
type EMRCluster struct {
    ID              string
    Name            string
    State           string
    PrimaryNodes    int
    CoreNodes       int
    TaskNodes       int
    InstanceType    string
    Applications    []string
    AutoScaling     *EMRAutoScaling
    Steps           []*EMRStep
    LogURI          string
    CreatedAt       time.Time
}

type EMRAutoScaling struct {
    MinNodes    int
    MaxNodes    int
    ScaleOutCPU float64 // threshold
    ScaleInCPU  float64
    Cooldown    time.Duration
}

type EMRStep struct {
    Name   string
    Type   string // CUSTOM_JAR, SPARK, HIVE, PIG
    Args   []string
    State  string
}

func NewEMRCluster(name string, cores int, apps []string) *EMRCluster {
    return &EMRCluster{
        ID:           fmt.Sprintf("j-%s", generateAnalyticsID()),
        Name:         name,
        State:        "RUNNING",
        PrimaryNodes: 1,
        CoreNodes:    cores,
        Applications: apps,
        LogURI:       "s3://emr-logs/",
        CreatedAt:    time.Now(),
    }
}

func (c *EMRCluster) AddStep(name, stepType string, args []string) {
    c.Steps = append(c.Steps, &EMRStep{
        Name:  name,
        Type:  stepType,
        Args:  args,
        State: "PENDING",
    })
}

func (c *EMRCluster) EstimateHourlyCost() float64 {
    // Approximate pricing
    costs := map[string]float64{
        "m5.xlarge":  0.192,
        "m5.2xlarge": 0.384,
        "r5.xlarge":  0.252,
        "c5.xlarge":  0.17,
    }
    
    rate, ok := costs[c.InstanceType]
    if !ok {
        rate = 0.192
    }
    
    totalNodes := c.PrimaryNodes + c.CoreNodes + c.TaskNodes
    emrRate := rate * 0.27 // EMR surcharge ~27%
    return float64(totalNodes) * (rate + emrRate)
}

// Redshift data warehouse simulator
type RedshiftCluster struct {
    ID              string
    NodeType        string
    NodeCount       int
    Database        string
    Tables          map[string]*RedshiftTable
    DistStyle       string
    SortKey         string
    Queries         []*RedshiftQuery
    mu              sync.RWMutex
}

type RedshiftTable struct {
    Name          string
    Columns       []RedshiftColumn
    DistStyle     string // AUTO, EVEN, KEY, ALL
    DistKey       string
    SortKeys      []string
    SortStyle     string // COMPOUND, INTERLEAVED
    RowCount      int64
    SizeBytes     int64
    Compressed    bool
}

type RedshiftColumn struct {
    Name     string
    Type     string
    Encoding string // RAW, AZ64, BYTEDICT, DELTA, LZO, MOSTLY, RUNLENGTH, TEXT255, TEXT32K, ZSTD
    Nullable bool
}

type RedshiftQuery struct {
    ID         string
    SQL        string
    Queue      string
    State      string
    StartTime  time.Time
    EndTime    time.Time
    RowsReturn int64
    BytesRead  int64
}

func NewRedshiftCluster(nodeType string, count int) *RedshiftCluster {
    return &RedshiftCluster{
        ID:        fmt.Sprintf("redshift-%s", generateAnalyticsID()),
        NodeType:  nodeType,
        NodeCount: count,
        Tables:    make(map[string]*RedshiftTable),
    }
}

func (r *RedshiftCluster) CreateTable(table *RedshiftTable) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.Tables[table.Name] = table
}

func (r *RedshiftCluster) OptimizeTableDesign(tableName string, queryPatterns []string) []string {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    table, exists := r.Tables[tableName]
    if !exists {
        return nil
    }
    
    var recommendations []string
    
    // Distribution style recommendation
    if table.DistStyle == "AUTO" || table.DistStyle == "" {
        if table.RowCount < 1000000 {
            recommendations = append(recommendations,
                fmt.Sprintf("Table %s has %d rows - consider DISTSTYLE ALL for small tables used in joins",
                    tableName, table.RowCount))
        }
        
        // Check for join patterns
        for _, q := range queryPatterns {
            if strings.Contains(strings.ToUpper(q), "JOIN") && strings.Contains(strings.ToUpper(q), strings.ToUpper(tableName)) {
                recommendations = append(recommendations,
                    fmt.Sprintf("Table %s appears in JOINs - consider DISTKEY on the join column", tableName))
                break
            }
        }
    }
    
    // Sort key recommendations
    if len(table.SortKeys) == 0 {
        for _, q := range queryPatterns {
            upper := strings.ToUpper(q)
            if strings.Contains(upper, "WHERE") || strings.Contains(upper, "ORDER BY") {
                recommendations = append(recommendations,
                    fmt.Sprintf("Table %s has no sort key - add sort key on frequently filtered/ordered columns", tableName))
                break
            }
        }
    }
    
    // Compression recommendation
    if !table.Compressed {
        recommendations = append(recommendations,
            fmt.Sprintf("Table %s is not compressed - run ANALYZE COMPRESSION to find optimal encodings", tableName))
    }
    
    return recommendations
}

func (r *RedshiftCluster) EstimateMonthlyCost() float64 {
    costs := map[string]float64{
        "ra3.xlplus":  1.086,
        "ra3.4xlarge": 3.26,
        "ra3.16xlarge": 13.04,
        "dc2.large":   0.25,
        "dc2.8xlarge": 4.80,
    }
    
    rate, ok := costs[r.NodeType]
    if !ok {
        rate = 1.086
    }
    
    return rate * 730 * float64(r.NodeCount)
}

// Lake Formation access control
type LakeFormation struct {
    databases   map[string]*LFDatabase
    permissions map[string][]LFPermission
    lfTags      map[string][]string
    mu          sync.RWMutex
}

type LFDatabase struct {
    Name     string
    Location string
    Tables   map[string]*LFTable
}

type LFTable struct {
    Name    string
    Columns []string
    Tags    map[string]string
}

type LFPermission struct {
    Principal     string
    Resource      string
    ResourceType  string // DATABASE, TABLE, COLUMN
    Permissions   []string // SELECT, INSERT, DELETE, ALTER, DROP, DESCRIBE
    Grantable     bool
    TagFilters    map[string][]string
}

func NewLakeFormation() *LakeFormation {
    return &LakeFormation{
        databases:   make(map[string]*LFDatabase),
        permissions: make(map[string][]LFPermission),
        lfTags:      make(map[string][]string),
    }
}

func (lf *LakeFormation) GrantPermission(principal, resource, resourceType string, perms []string) {
    lf.mu.Lock()
    defer lf.mu.Unlock()
    
    lf.permissions[principal] = append(lf.permissions[principal], LFPermission{
        Principal:    principal,
        Resource:     resource,
        ResourceType: resourceType,
        Permissions:  perms,
    })
}

func (lf *LakeFormation) GrantWithLFTags(principal string, tags map[string][]string, perms []string) {
    lf.mu.Lock()
    defer lf.mu.Unlock()
    
    lf.permissions[principal] = append(lf.permissions[principal], LFPermission{
        Principal:   principal,
        ResourceType: "TAG",
        Permissions: perms,
        TagFilters:  tags,
    })
}

func (lf *LakeFormation) CheckAccess(principal, resource, permission string) bool {
    lf.mu.RLock()
    defer lf.mu.RUnlock()
    
    perms, exists := lf.permissions[principal]
    if !exists {
        return false
    }
    
    for _, p := range perms {
        if p.Resource == resource || p.ResourceType == "TAG" {
            for _, allowed := range p.Permissions {
                if allowed == permission || allowed == "ALL" {
                    return true
                }
            }
        }
    }
    
    return false
}

// Data pipeline orchestrator
type DataPipeline struct {
    Name     string
    Steps    []*PipelineStep
    Schedule string
    State    string
    Runs     []*PipelineRun
    mu       sync.Mutex
}

type PipelineStep struct {
    Name         string
    Type         string // GLUE_JOB, ATHENA_QUERY, EMR_STEP, LAMBDA
    Config       map[string]string
    DependsOn    []string
    RetryCount   int
    TimeoutMin   int
}

type PipelineRun struct {
    ID        string
    StartTime time.Time
    EndTime   time.Time
    Status    string
    StepStats map[string]*StepRunStatus
}

type StepRunStatus struct {
    Status    string
    StartTime time.Time
    EndTime   time.Time
    Error     string
    Retries   int
}

func NewDataPipeline(name, schedule string) *DataPipeline {
    return &DataPipeline{
        Name:     name,
        Schedule: schedule,
        State:    "ACTIVE",
    }
}

func (p *DataPipeline) AddStep(step *PipelineStep) {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.Steps = append(p.Steps, step)
}

func (p *DataPipeline) Execute() *PipelineRun {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    run := &PipelineRun{
        ID:        fmt.Sprintf("run-%d", len(p.Runs)+1),
        StartTime: time.Now(),
        Status:    "RUNNING",
        StepStats: make(map[string]*StepRunStatus),
    }
    
    // Topological sort of steps
    order := p.topologicalSort()
    
    for _, stepName := range order {
        var step *PipelineStep
        for _, s := range p.Steps {
            if s.Name == stepName {
                step = s
                break
            }
        }
        
        if step == nil {
            continue
        }
        
        stepRun := &StepRunStatus{
            Status:    "SUCCEEDED",
            StartTime: time.Now(),
            EndTime:   time.Now(),
        }
        
        run.StepStats[step.Name] = stepRun
    }
    
    run.EndTime = time.Now()
    run.Status = "SUCCEEDED"
    p.Runs = append(p.Runs, run)
    
    return run
}

func (p *DataPipeline) topologicalSort() []string {
    // Build adjacency and in-degree
    inDegree := make(map[string]int)
    adj := make(map[string][]string)
    
    for _, step := range p.Steps {
        if _, exists := inDegree[step.Name]; !exists {
            inDegree[step.Name] = 0
        }
        for _, dep := range step.DependsOn {
            adj[dep] = append(adj[dep], step.Name)
            inDegree[step.Name]++
        }
    }
    
    // BFS
    var queue []string
    for name, deg := range inDegree {
        if deg == 0 {
            queue = append(queue, name)
        }
    }
    
    var order []string
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        order = append(order, node)
        
        for _, next := range adj[node] {
            inDegree[next]--
            if inDegree[next] == 0 {
                queue = append(queue, next)
            }
        }
    }
    
    return order
}

// SageMaker model manager
type SageMakerManager struct {
    models      map[string]*MLModel
    endpoints   map[string]*MLEndpoint
    experiments map[string]*MLExperiment
    mu          sync.RWMutex
}

type MLModel struct {
    Name        string
    Version     int
    Algorithm   string
    Framework   string
    ArtifactURI string
    Metrics     map[string]float64
    Parameters  map[string]string
    CreatedAt   time.Time
    Status      string
}

type MLEndpoint struct {
    Name          string
    ModelName     string
    InstanceType  string
    InstanceCount int
    Status        string
    Variant       string
    InvocationCount int64
    AvgLatencyMs  float64
}

type MLExperiment struct {
    Name   string
    Trials []*MLTrial
}

type MLTrial struct {
    Name       string
    Parameters map[string]string
    Metrics    map[string]float64
    Status     string
    StartTime  time.Time
    EndTime    time.Time
}

func NewSageMakerManager() *SageMakerManager {
    return &SageMakerManager{
        models:      make(map[string]*MLModel),
        endpoints:   make(map[string]*MLEndpoint),
        experiments: make(map[string]*MLExperiment),
    }
}

func (sm *SageMakerManager) RegisterModel(model *MLModel) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    sm.models[model.Name] = model
}

func (sm *SageMakerManager) DeployEndpoint(name, modelName, instanceType string, count int) (*MLEndpoint, error) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    
    _, exists := sm.models[modelName]
    if !exists {
        return nil, fmt.Errorf("model %s not found", modelName)
    }
    
    endpoint := &MLEndpoint{
        Name:          name,
        ModelName:     modelName,
        InstanceType:  instanceType,
        InstanceCount: count,
        Status:        "InService",
        Variant:       "AllTraffic",
    }
    
    sm.endpoints[name] = endpoint
    return endpoint, nil
}

func (sm *SageMakerManager) Invoke(endpointName string, payload []byte) ([]byte, error) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    
    ep, exists := sm.endpoints[endpointName]
    if !exists {
        return nil, fmt.Errorf("endpoint %s not found", endpointName)
    }
    
    if ep.Status != "InService" {
        return nil, fmt.Errorf("endpoint %s not in service", endpointName)
    }
    
    ep.InvocationCount++
    
    // Simulate inference
    result := map[string]interface{}{
        "prediction": rand.Float64(),
        "confidence": 0.85 + rand.Float64()*0.15,
    }
    
    return json.Marshal(result)
}

func (sm *SageMakerManager) RunHyperparameterTuning(experiment string, baseParams map[string]string, paramRanges map[string][2]float64, maxTrials int) *MLExperiment {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    
    exp := &MLExperiment{Name: experiment}
    
    for i := 0; i < maxTrials; i++ {
        trial := &MLTrial{
            Name:       fmt.Sprintf("trial-%d", i),
            Parameters: make(map[string]string),
            Metrics:    make(map[string]float64),
            Status:     "Completed",
            StartTime:  time.Now(),
            EndTime:    time.Now(),
        }
        
        for k, v := range baseParams {
            trial.Parameters[k] = v
        }
        
        // Random search within ranges
        for param, bounds := range paramRanges {
            value := bounds[0] + rand.Float64()*(bounds[1]-bounds[0])
            trial.Parameters[param] = fmt.Sprintf("%.4f", value)
        }
        
        // Simulate metrics
        trial.Metrics["accuracy"] = 0.7 + rand.Float64()*0.25
        trial.Metrics["loss"] = 0.1 + rand.Float64()*0.5
        trial.Metrics["f1_score"] = 0.65 + rand.Float64()*0.3
        
        exp.Trials = append(exp.Trials, trial)
    }
    
    // Sort by accuracy
    sort.Slice(exp.Trials, func(i, j int) bool {
        return exp.Trials[i].Metrics["accuracy"] > exp.Trials[j].Metrics["accuracy"]
    })
    
    sm.experiments[experiment] = exp
    return exp
}

func (sm *SageMakerManager) GetBestTrial(experiment string) (*MLTrial, error) {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    
    exp, exists := sm.experiments[experiment]
    if !exists {
        return nil, fmt.Errorf("experiment %s not found", experiment)
    }
    
    if len(exp.Trials) == 0 {
        return nil, fmt.Errorf("no trials in experiment %s", experiment)
    }
    
    return exp.Trials[0], nil
}

func (sm *SageMakerManager) EstimateEndpointCost(endpointName string) float64 {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    
    ep, exists := sm.endpoints[endpointName]
    if !exists {
        return 0
    }
    
    costs := map[string]float64{
        "ml.t3.medium":   0.0464,
        "ml.m5.large":    0.115,
        "ml.m5.xlarge":   0.23,
        "ml.c5.xlarge":   0.204,
        "ml.g4dn.xlarge": 0.7364,
        "ml.p3.2xlarge":  3.825,
    }
    
    rate, ok := costs[ep.InstanceType]
    if !ok {
        rate = 0.23
    }
    
    return rate * 730 * float64(ep.InstanceCount) // Monthly
}

func generateAnalyticsID() string {
    return fmt.Sprintf("%013x", time.Now().UnixNano()%0x1000000000000)
}`,
				},
			},
		},
	})
}
