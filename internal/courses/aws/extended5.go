package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2119,
			Title:       "AWS Database Services",
			Description: "Master RDS, Aurora, DynamoDB, ElastiCache, Redshift, Neptune, DocumentDB, and database migration strategies on AWS.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "RDS Aurora DynamoDB and Database Services",
					Content: `AWS offers purpose-built database services for different data models, performance requirements, and cost targets.

**Amazon RDS (Relational Database Service):**

Supported Engines:
  MySQL, PostgreSQL, MariaDB
  Oracle, SQL Server
  Amazon Aurora (MySQL/PostgreSQL compatible)

Instance Classes:
  General Purpose (db.m5, db.m6g): Balanced
  Memory Optimized (db.r5, db.r6g): Large datasets
  Burstable (db.t3, db.t4g): Variable workloads
  Graviton (db.m6g, db.r6g): ARM-based, cost-effective

Storage Types:
  gp3: General purpose SSD (3000 IOPS baseline, up to 16K)
  io1/io2: Provisioned IOPS (up to 256K IOPS)
  magnetic: Legacy, not recommended

Multi-AZ Deployment:
  Synchronous replication to standby
  Automatic failover (1-2 minutes)
  No read access to standby
  Multi-AZ DB Cluster: 2 readable standbys (MySQL/PostgreSQL)

Read Replicas:
  Asynchronous replication
  Up to 15 replicas
  Cross-region supported
  Can be promoted to standalone
  Use for read-heavy workloads

Backup and Recovery:
  Automated Backups: Daily (35-day retention max)
  Manual Snapshots: No retention limit
  Point-in-Time Recovery: Transaction log-based
  Cross-Region: Snapshot copy and read replicas

RDS Proxy:
  Connection pooling
  IAM authentication
  Failover-aware
  Reduces database load
  Serverless-friendly (Lambda)

**Amazon Aurora:**

Architecture:
  Cluster Volume: 6 copies across 3 AZs
  Primary Instance: Read/write
  Aurora Replicas: Up to 15 (auto-failover)
  Storage: Auto-grows up to 128 TB
  
Aurora vs Standard RDS:
  5x throughput of MySQL
  3x throughput of PostgreSQL
  10ms replica lag (vs seconds for RDS)
  Self-healing storage
  Automatic failover < 30 seconds

Aurora Serverless v2:
  Auto-scales compute capacity
  Scales in 0.5 ACU increments
  Min/max ACU configuration
  Pay per ACU-second
  Mixed with provisioned instances

Aurora Global Database:
  Primary Region: Read/write
  Secondary Region(s): Read-only (< 1 second lag)
  RPO: < 1 second
  RTO: < 1 minute (managed failover)
  Up to 5 secondary regions

Aurora Features:
  Backtrack: Rewind to point in time (no restore)
  Parallel Query: Push down query processing to storage
  Machine Learning: SageMaker/Comprehend integration
  Activity Streams: Database activity monitoring
  Zero-ETL to Redshift: Near real-time analytics

**Amazon DynamoDB:**

Data Model:
  Tables -> Items -> Attributes
  Primary Key: Partition key, or partition + sort key
  Secondary Indexes: GSI and LSI
  
Capacity Modes:
  Provisioned: Specify RCU/WCU
    RCU: 1 strongly consistent read/s (up to 4KB)
    WCU: 1 write/s (up to 1KB)
    Auto Scaling: Target utilization
  On-Demand: Pay per request
    No capacity planning
    Instantly accommodates up to 2x previous peak

Consistency:
  Eventually Consistent: Default (half the RCU)
  Strongly Consistent: Read from leader (full RCU)
  Transactional: ACID transactions (2x RCU/WCU)

Secondary Indexes:
  Global Secondary Index (GSI):
    Different partition/sort key
    Eventually consistent only
    Separate provisioned capacity
    Up to 20 per table
  Local Secondary Index (LSI):
    Same partition key, different sort key
    Strong or eventual consistency
    Shares table capacity
    Must create at table creation
    Up to 5 per table

DynamoDB Streams:
  Time-ordered change log
  Item-level modifications
  24-hour retention
  Lambda triggers
  Use for: Event-driven, replication, aggregation

Global Tables:
  Multi-region, multi-active
  Active-active replication
  < 1 second replication
  Automatic conflict resolution (last writer wins)

DAX (DynamoDB Accelerator):
  In-memory cache for DynamoDB
  Microsecond read latency
  Write-through cache
  Compatible with DynamoDB API
  Cluster within your VPC

Single-Table Design:
  Store multiple entity types in one table
  Use composite keys: PK=USER#123, SK=ORDER#456
  GSI overloading for access patterns
  Reduces cost and complexity

**Amazon ElastiCache:**

Redis:
  Data structures: Strings, lists, sets, sorted sets, hashes
  Cluster Mode: Sharding across up to 500 nodes
  Replication: Up to 5 replicas per shard
  Global Datastore: Cross-region replication
  Features: Pub/sub, Lua scripting, geospatial
  
Memcached:
  Simple key-value
  Multi-threaded
  No replication/persistence
  Auto Discovery

Use Cases:
  Session management
  Database caching (cache-aside, write-through)
  Leaderboards (Redis sorted sets)
  Rate limiting
  Real-time analytics

**Amazon Redshift:**

  Columnar data warehouse
  Up to petabyte scale
  SQL-based (PostgreSQL compatible)
  
  Node Types:
    RA3: Managed storage (separate compute/storage)
    DC2: Dense compute (local SSD)
    
  Redshift Serverless: Auto-provisioned
  
  Features:
    Concurrency Scaling: Auto-add clusters for burst
    Spectrum: Query S3 data directly
    AQUA: Hardware-accelerated cache
    Data Sharing: Cross-cluster queries
    Materialized Views: Automatic refresh
    Zero-ETL from Aurora

**AWS Database Migration Service (DMS):**

  Migrate databases to AWS
  Heterogeneous: Oracle to Aurora, SQL Server to PostgreSQL
  Homogeneous: MySQL to RDS MySQL
  
  Components:
    Replication Instance: EC2 performing migration
    Source Endpoint: Source database connection
    Target Endpoint: Target database connection
    Replication Task: Defines what to migrate
    
  Migration Types:
    Full Load: One-time copy
    CDC (Change Data Capture): Ongoing replication
    Full Load + CDC: Initial copy then ongoing
    
  Schema Conversion Tool (SCT):
    Convert schema between engines
    Identify conversion issues
    Assessment report`,
					CodeExamples: `// AWS database service implementations

package main

import (
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "math"
    "sort"
    "strings"
    "sync"
    "time"
)

// DynamoDB table simulator
type DynamoDBTable struct {
    Name          string
    PartitionKey  string
    SortKey       string
    CapacityMode  string // PROVISIONED, ON_DEMAND
    ReadCapacity  int
    WriteCapacity int
    Items         map[string]map[string]DynamoDBItem
    GSIs          []*DynamoDBGSI
    LSIs          []*DynamoDBLSI
    StreamEnabled bool
    StreamEvents  []StreamEvent
    mu            sync.RWMutex
}

type DynamoDBItem map[string]DynamoDBValue

type DynamoDBValue struct {
    S    string
    N    string
    B    []byte
    BOOL bool
    L    []DynamoDBValue
    M    map[string]DynamoDBValue
    NULL bool
    Type string // S, N, B, BOOL, L, M, NULL
}

type DynamoDBGSI struct {
    IndexName    string
    PartitionKey string
    SortKey      string
    Projection   string // ALL, KEYS_ONLY, INCLUDE
    ProjectionAttrs []string
    ReadCapacity int
    WriteCapacity int
}

type DynamoDBLSI struct {
    IndexName    string
    SortKey      string
    Projection   string
}

type StreamEvent struct {
    EventName string // INSERT, MODIFY, REMOVE
    Keys      map[string]DynamoDBValue
    NewImage  DynamoDBItem
    OldImage  DynamoDBItem
    Timestamp time.Time
    SequenceNumber string
}

func NewDynamoDBTable(name, pk, sk, capacityMode string) *DynamoDBTable {
    return &DynamoDBTable{
        Name:         name,
        PartitionKey: pk,
        SortKey:      sk,
        CapacityMode: capacityMode,
        Items:        make(map[string]map[string]DynamoDBItem),
    }
}

func (t *DynamoDBTable) PutItem(item DynamoDBItem) error {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    pkVal, ok := item[t.PartitionKey]
    if !ok {
        return fmt.Errorf("missing partition key: %s", t.PartitionKey)
    }
    
    pkStr := getStringValue(pkVal)
    skStr := ""
    if t.SortKey != "" {
        skVal, ok := item[t.SortKey]
        if !ok {
            return fmt.Errorf("missing sort key: %s", t.SortKey)
        }
        skStr = getStringValue(skVal)
    }
    
    if _, exists := t.Items[pkStr]; !exists {
        t.Items[pkStr] = make(map[string]DynamoDBItem)
    }
    
    oldItem := t.Items[pkStr][skStr]
    t.Items[pkStr][skStr] = item
    
    if t.StreamEnabled {
        eventName := "INSERT"
        if oldItem != nil {
            eventName = "MODIFY"
        }
        t.StreamEvents = append(t.StreamEvents, StreamEvent{
            EventName: eventName,
            Keys:      map[string]DynamoDBValue{t.PartitionKey: pkVal},
            NewImage:  item,
            OldImage:  oldItem,
            Timestamp: time.Now(),
            SequenceNumber: fmt.Sprintf("%d", len(t.StreamEvents)+1),
        })
    }
    
    return nil
}

func (t *DynamoDBTable) GetItem(pk, sk string) (DynamoDBItem, error) {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    partition, exists := t.Items[pk]
    if !exists {
        return nil, nil
    }
    
    item, exists := partition[sk]
    if !exists {
        return nil, nil
    }
    
    return item, nil
}

func (t *DynamoDBTable) Query(pk string, skBegins string) ([]DynamoDBItem, error) {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    partition, exists := t.Items[pk]
    if !exists {
        return nil, nil
    }
    
    var results []DynamoDBItem
    for sk, item := range partition {
        if skBegins == "" || strings.HasPrefix(sk, skBegins) {
            results = append(results, item)
        }
    }
    
    // Sort by sort key
    sort.Slice(results, func(i, j int) bool {
        iSK := getStringValue(results[i][t.SortKey])
        jSK := getStringValue(results[j][t.SortKey])
        return iSK < jSK
    })
    
    return results, nil
}

func (t *DynamoDBTable) DeleteItem(pk, sk string) error {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    if partition, exists := t.Items[pk]; exists {
        if item, exists := partition[sk]; exists {
            delete(partition, sk)
            
            if t.StreamEnabled {
                t.StreamEvents = append(t.StreamEvents, StreamEvent{
                    EventName: "REMOVE",
                    OldImage:  item,
                    Timestamp: time.Now(),
                    SequenceNumber: fmt.Sprintf("%d", len(t.StreamEvents)+1),
                })
            }
        }
    }
    
    return nil
}

func (t *DynamoDBTable) Scan(filterAttr, filterValue string) []DynamoDBItem {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    var results []DynamoDBItem
    for _, partition := range t.Items {
        for _, item := range partition {
            if filterAttr == "" {
                results = append(results, item)
                continue
            }
            val, ok := item[filterAttr]
            if ok && getStringValue(val) == filterValue {
                results = append(results, item)
            }
        }
    }
    return results
}

func getStringValue(v DynamoDBValue) string {
    switch v.Type {
    case "S":
        return v.S
    case "N":
        return v.N
    default:
        return v.S
    }
}

// RDS instance manager
type RDSManager struct {
    instances map[string]*RDSInstance
    clusters  map[string]*AuroraCluster
    mu        sync.RWMutex
}

type RDSInstance struct {
    ID               string
    Engine           string
    EngineVersion    string
    InstanceClass    string
    MultiAZ          bool
    StorageType      string
    AllocatedStorage int
    IOPS             int
    Status           string
    Endpoint         string
    ReadReplicas     []string
    BackupRetention  int
    EncryptionConfig *DBEncryption
    Proxy            *RDSProxyConfig
}

type DBEncryption struct {
    Enabled bool
    KMSKeyID string
}

type RDSProxyConfig struct {
    Name       string
    Endpoint   string
    IAMAuth    bool
    PoolConfig ConnectionPoolConfig
}

type ConnectionPoolConfig struct {
    MaxConnections     int
    MaxIdleConnections int
    BorrowTimeout      time.Duration
    InitQuery          string
}

type AuroraCluster struct {
    ID          string
    Engine      string
    Writer      string
    Readers     []string
    Serverless  bool
    MinACU      float64
    MaxACU      float64
    Global      bool
    Endpoint    AuroraEndpoints
    BacktrackWindow time.Duration
}

type AuroraEndpoints struct {
    Writer string
    Reader string
    Custom map[string]string
}

func NewRDSManager() *RDSManager {
    return &RDSManager{
        instances: make(map[string]*RDSInstance),
        clusters:  make(map[string]*AuroraCluster),
    }
}

func (m *RDSManager) CreateInstance(id, engine, instanceClass string, multiAZ bool) (*RDSInstance, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if _, exists := m.instances[id]; exists {
        return nil, fmt.Errorf("instance %s already exists", id)
    }
    
    instance := &RDSInstance{
        ID:              id,
        Engine:          engine,
        InstanceClass:   instanceClass,
        MultiAZ:         multiAZ,
        StorageType:     "gp3",
        AllocatedStorage: 100,
        Status:          "creating",
        Endpoint:        fmt.Sprintf("%s.xxx.us-east-1.rds.amazonaws.com", id),
        BackupRetention: 7,
    }
    
    m.instances[id] = instance
    return instance, nil
}

func (m *RDSManager) CreateReadReplica(sourceID, replicaID string) (*RDSInstance, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    source, exists := m.instances[sourceID]
    if !exists {
        return nil, fmt.Errorf("source instance %s not found", sourceID)
    }
    
    replica := &RDSInstance{
        ID:            replicaID,
        Engine:        source.Engine,
        InstanceClass: source.InstanceClass,
        StorageType:   source.StorageType,
        Status:        "creating",
        Endpoint:      fmt.Sprintf("%s.xxx.us-east-1.rds.amazonaws.com", replicaID),
    }
    
    source.ReadReplicas = append(source.ReadReplicas, replicaID)
    m.instances[replicaID] = replica
    return replica, nil
}

// ElastiCache manager
type ElastiCacheManager struct {
    clusters map[string]*CacheCluster
    mu       sync.RWMutex
}

type CacheCluster struct {
    ID         string
    Engine     string // redis, memcached
    NodeType   string
    NumNodes   int
    Shards     int
    Replicas   int
    Status     string
    Endpoint   string
    Data       map[string]*CacheEntry
    mu         sync.RWMutex
}

type CacheEntry struct {
    Value     string
    TTL       time.Duration
    CreatedAt time.Time
}

func NewElastiCacheManager() *ElastiCacheManager {
    return &ElastiCacheManager{
        clusters: make(map[string]*CacheCluster),
    }
}

func (m *ElastiCacheManager) CreateCluster(id, engine, nodeType string, nodes int) *CacheCluster {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    cluster := &CacheCluster{
        ID:       id,
        Engine:   engine,
        NodeType: nodeType,
        NumNodes: nodes,
        Status:   "available",
        Endpoint: fmt.Sprintf("%s.xxx.cache.amazonaws.com", id),
        Data:     make(map[string]*CacheEntry),
    }
    
    m.clusters[id] = cluster
    return cluster
}

func (c *CacheCluster) Set(key, value string, ttl time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    c.Data[key] = &CacheEntry{
        Value:     value,
        TTL:       ttl,
        CreatedAt: time.Now(),
    }
}

func (c *CacheCluster) Get(key string) (string, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    entry, exists := c.Data[key]
    if !exists {
        return "", false
    }
    
    if entry.TTL > 0 && time.Since(entry.CreatedAt) > entry.TTL {
        return "", false
    }
    
    return entry.Value, true
}

func (c *CacheCluster) Delete(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    delete(c.Data, key)
}

// DynamoDB capacity calculator
type DynamoDBCapacityCalculator struct{}

func (c *DynamoDBCapacityCalculator) CalculateRCU(itemSizeKB float64, readsPerSec int, stronglyConsistent bool) int {
    // Each RCU = 4KB strongly consistent read/s or 8KB eventually consistent
    readUnitsPerItem := math.Ceil(itemSizeKB / 4.0)
    totalRCU := float64(readsPerSec) * readUnitsPerItem
    
    if !stronglyConsistent {
        totalRCU /= 2.0
    }
    
    return int(math.Ceil(totalRCU))
}

func (c *DynamoDBCapacityCalculator) CalculateWCU(itemSizeKB float64, writesPerSec int) int {
    // Each WCU = 1KB write/s
    writeUnitsPerItem := math.Ceil(itemSizeKB / 1.0)
    return int(math.Ceil(float64(writesPerSec) * writeUnitsPerItem))
}

func (c *DynamoDBCapacityCalculator) CalculateTransactionalRCU(itemSizeKB float64, readsPerSec int) int {
    // Transactional reads cost 2x
    return c.CalculateRCU(itemSizeKB, readsPerSec, true) * 2
}

func (c *DynamoDBCapacityCalculator) EstimateMonthlyCost(rcu, wcu int, region string) float64 {
    // Simplified pricing (us-east-1)
    rcuCost := float64(rcu) * 0.00065 * 730 // per hour * hours/month
    wcuCost := float64(wcu) * 0.00065 * 730
    return rcuCost + wcuCost
}`,
				},
			},
		},
	})
}
