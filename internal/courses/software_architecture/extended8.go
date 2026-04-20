package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2323,
			Title:       "Data Architecture and Data Management",
			Description: "Design data architectures with data mesh, data lakehouse, ETL/ELT pipelines, data governance, and polyglot persistence strategies.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Modern Data Architecture Patterns",
					Content: `Data architecture defines how data is collected, stored, transformed, distributed, and consumed across an organization.

**Data Architecture Evolution:**

Data Warehouse (Traditional):
  Centralized, schema-on-write
  ETL: Extract-Transform-Load
  Star/Snowflake schemas
  Optimized for analytics (OLAP)
  Tools: Teradata, Oracle, Snowflake, BigQuery
  
  Pros: Strong consistency, SQL-based, well-understood
  Cons: Rigid schemas, slow to adapt, expensive to scale

Data Lake:
  Centralized storage for raw data
  Schema-on-read
  Store everything, decide structure later
  Tools: HDFS, S3, Azure Data Lake, GCS
  
  Pros: Flexible, cheap storage, handles all data types
  Cons: Can become "data swamp", governance challenges

Data Lakehouse:
  Combines data lake + data warehouse
  Raw data in object storage with ACID transactions
  Schema enforcement + evolution
  Tools: Delta Lake, Apache Iceberg, Apache Hudi
  
  Key features:
    ACID transactions on data lake
    Schema enforcement and evolution
    Time travel (query historical data)
    Unified batch and streaming

Data Mesh:
  Domain-oriented decentralized data ownership
  Data as a product mindset
  Self-serve data infrastructure
  Federated computational governance
  
  Principles:
    Domain Ownership: Each domain owns its data pipelines
    Data as Product: Discoverable, addressable, trustworthy
    Self-Serve Platform: Infrastructure as a service
    Federated Governance: Global standards, local autonomy

  Domain Data Product:
    Input ports: How data enters the domain
    Output ports: How data is served to consumers
    SLOs: Quality, freshness, availability guarantees
    Documentation: Schema, semantics, lineage

**ETL vs ELT:**

ETL (Extract-Transform-Load):
  Transform data before loading into warehouse
  Data is cleaned and structured
  Lower storage costs
  Good for: Structured data, well-known schemas

ELT (Extract-Load-Transform):
  Load raw data, transform in destination
  Leverage destination compute power
  More flexible, faster ingestion
  Good for: Large volumes, evolving schemas
  Tools: dbt, Spark, Trino

**Polyglot Persistence:**

Use the right database for each use case:

  Relational (PostgreSQL, MySQL):
    Transactions, complex queries, joins
    User accounts, orders, inventory

  Document (MongoDB, DynamoDB):
    Flexible schemas, hierarchical data
    Product catalogs, content management

  Key-Value (Redis, Memcached):
    Caching, sessions, counters
    Sub-millisecond access

  Column-Family (Cassandra, HBase):
    High write throughput, time-series
    IoT data, activity feeds

  Graph (Neo4j, Neptune):
    Relationship traversal
    Social networks, recommendations

  Search (Elasticsearch, Solr):
    Full-text search, log analysis
    Product search, log aggregation

  Time-Series (InfluxDB, TimescaleDB):
    Metrics, IoT, monitoring data
    High write throughput, time-based queries

**Data Governance:**

Data Quality Dimensions:
  Accuracy: Data correctly represents reality
  Completeness: All required data is present
  Consistency: Same data across systems agrees
  Timeliness: Data is up-to-date
  Validity: Data conforms to defined rules
  Uniqueness: No duplicate records

Data Catalog:
  Discover and understand available data
  Technical metadata (schema, types, lineage)
  Business metadata (descriptions, owners, classifications)
  Tools: Apache Atlas, DataHub, Amundsen

Data Lineage:
  Track data from source to consumption
  Understand transformations applied
  Impact analysis for changes
  Compliance and auditing

Data Classification:
  Public: No restrictions
  Internal: Internal use only
  Confidential: Limited access
  Restricted: Highest protection (PII, financial)

**Change Data Capture (CDC):**

  Track changes in database and propagate them
  
  Log-based CDC:
    Read database transaction log (WAL, binlog)
    Most reliable, minimal impact on source
    Tools: Debezium, Maxwell
    
  Query-based CDC:
    Poll source tables for changes
    Uses timestamps or version columns
    Simpler but less reliable
    
  Trigger-based CDC:
    Database triggers capture changes
    Higher overhead on source database

CDC Pipeline:
  Source DB -> Debezium -> Kafka -> Consumer
  
  Use cases:
    Replicate data between databases
    Feed search indexes
    Update caches
    Event-driven architectures
    Maintain materialized views`,
					CodeExamples: `// Data architecture patterns

// Repository pattern with polyglot persistence
type UserRepository interface {
    FindByID(ctx context.Context, id string) (*User, error)
    FindByEmail(ctx context.Context, email string) (*User, error)
    Save(ctx context.Context, user *User) error
    Delete(ctx context.Context, id string) error
}

type ProductSearchRepository interface {
    Search(ctx context.Context, query SearchQuery) (*SearchResult, error)
    Index(ctx context.Context, product *Product) error
    DeleteFromIndex(ctx context.Context, id string) error
}

type SessionStore interface {
    Get(ctx context.Context, sessionID string) (*Session, error)
    Set(ctx context.Context, session *Session, ttl time.Duration) error
    Delete(ctx context.Context, sessionID string) error
}

// Data pipeline stages
type Pipeline struct {
    stages []Stage
    logger *Logger
}

type Stage interface {
    Name() string
    Process(ctx context.Context, data *DataBatch) (*DataBatch, error)
}

type DataBatch struct {
    Records   []Record
    Metadata  map[string]string
    CreatedAt time.Time
}

type Record struct {
    Key     string
    Value   map[string]interface{}
    Headers map[string]string
}

func NewPipeline(stages ...Stage) *Pipeline {
    return &Pipeline{stages: stages}
}

func (p *Pipeline) Execute(ctx context.Context, input *DataBatch) (*DataBatch, error) {
    current := input
    for _, stage := range p.stages {
        p.logger.Info("executing stage",
            "stage", stage.Name(),
            "records", len(current.Records),
        )
        
        result, err := stage.Process(ctx, current)
        if err != nil {
            return nil, fmt.Errorf("stage %s failed: %w", stage.Name(), err)
        }
        current = result
    }
    return current, nil
}

// ETL stages
type ExtractStage struct {
    source DataSource
}

func (s *ExtractStage) Name() string { return "extract" }

func (s *ExtractStage) Process(ctx context.Context, _ *DataBatch) (*DataBatch, error) {
    records, err := s.source.Read(ctx)
    if err != nil {
        return nil, fmt.Errorf("extraction failed: %w", err)
    }
    return &DataBatch{
        Records:   records,
        Metadata:  map[string]string{"source": s.source.Name()},
        CreatedAt: time.Now(),
    }, nil
}

type TransformStage struct {
    transformers []Transformer
}

type Transformer func(Record) (Record, error)

func (s *TransformStage) Name() string { return "transform" }

func (s *TransformStage) Process(ctx context.Context, batch *DataBatch) (*DataBatch, error) {
    result := &DataBatch{
        Records:   make([]Record, 0, len(batch.Records)),
        Metadata:  batch.Metadata,
        CreatedAt: batch.CreatedAt,
    }
    
    for _, record := range batch.Records {
        current := record
        for _, transform := range s.transformers {
            var err error
            current, err = transform(current)
            if err != nil {
                return nil, fmt.Errorf("transform failed for key %s: %w", record.Key, err)
            }
        }
        result.Records = append(result.Records, current)
    }
    
    return result, nil
}

type LoadStage struct {
    sink      DataSink
    batchSize int
}

func (s *LoadStage) Name() string { return "load" }

func (s *LoadStage) Process(ctx context.Context, batch *DataBatch) (*DataBatch, error) {
    for i := 0; i < len(batch.Records); i += s.batchSize {
        end := i + s.batchSize
        if end > len(batch.Records) {
            end = len(batch.Records)
        }
        
        chunk := batch.Records[i:end]
        if err := s.sink.Write(ctx, chunk); err != nil {
            return nil, fmt.Errorf("load failed at offset %d: %w", i, err)
        }
    }
    return batch, nil
}

// Data quality validation
type DataValidator struct {
    rules []ValidationRule
}

type ValidationRule struct {
    Name    string
    Check   func(Record) error
}

type ValidationReport struct {
    TotalRecords   int
    ValidRecords   int
    InvalidRecords int
    Errors         []ValidationIssue
}

type ValidationIssue struct {
    RecordKey string
    Rule      string
    Message   string
}

func (v *DataValidator) Validate(batch *DataBatch) *ValidationReport {
    report := &ValidationReport{TotalRecords: len(batch.Records)}
    
    for _, record := range batch.Records {
        valid := true
        for _, rule := range v.rules {
            if err := rule.Check(record); err != nil {
                valid = false
                report.Errors = append(report.Errors, ValidationIssue{
                    RecordKey: record.Key,
                    Rule:      rule.Name,
                    Message:   err.Error(),
                })
            }
        }
        if valid {
            report.ValidRecords++
        } else {
            report.InvalidRecords++
        }
    }
    
    return report
}

// CDC event consumer
type CDCEvent struct {
    Operation string                 "json:\"op\""       // c=create, u=update, d=delete
    Before    map[string]interface{} "json:\"before\""
    After     map[string]interface{} "json:\"after\""
    Source    CDCSource              "json:\"source\""
    Timestamp int64                  "json:\"ts_ms\""
}

type CDCSource struct {
    Database string "json:\"db\""
    Table    string "json:\"table\""
    LSN      int64  "json:\"lsn\""
}

type CDCHandler interface {
    HandleCreate(ctx context.Context, after map[string]interface{}) error
    HandleUpdate(ctx context.Context, before, after map[string]interface{}) error
    HandleDelete(ctx context.Context, before map[string]interface{}) error
}

type CDCConsumer struct {
    handlers map[string]CDCHandler  // table -> handler
    logger   *Logger
}

func (c *CDCConsumer) Process(ctx context.Context, event *CDCEvent) error {
    handler, ok := c.handlers[event.Source.Table]
    if !ok {
        c.logger.Warn("no handler for table", "table", event.Source.Table)
        return nil
    }
    
    switch event.Operation {
    case "c":
        return handler.HandleCreate(ctx, event.After)
    case "u":
        return handler.HandleUpdate(ctx, event.Before, event.After)
    case "d":
        return handler.HandleDelete(ctx, event.Before)
    default:
        return fmt.Errorf("unknown CDC operation: %s", event.Operation)
    }
}`,
				},
			},
		},
		{
			ID:          2324,
			Title:       "Architecture Modernization and Migration",
			Description: "Modernize legacy systems with strangler fig pattern, monolith decomposition, database migration strategies, and incremental modernization approaches.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Modernizing Legacy Systems",
					Content: `Legacy system modernization requires careful planning and incremental execution to minimize risk.

**Strangler Fig Pattern:**

  Named after the strangler fig tree that grows around a host tree
  Gradually replace parts of the legacy system with new implementations
  Both systems coexist during migration
  
  Steps:
    1. Identify: Choose a bounded context to migrate
    2. Transform: Build new implementation
    3. Coexist: Run both old and new in parallel
    4. Eliminate: Remove old implementation
    
  Implementation with reverse proxy:
    ┌──────────┐     ┌───────────────┐     ┌──────────┐
    │  Client   │ --> │ Reverse Proxy │ --> │  Legacy   │
    │           │     │  (Router)     │ --> │  System   │
    └──────────┘     └───────────────┘     └──────────┘
                            │
                            └──> ┌──────────────┐
                                 │ New Service   │
                                 │ (migrated)    │
                                 └──────────────┘
    
    Route by URL path:
      /api/users/* -> New User Service
      /api/orders/* -> Legacy System (not yet migrated)
      /api/products/* -> New Product Service

  Feature Flags for Migration:
    Enable new service for percentage of traffic
    Monitor metrics and error rates
    Gradually increase traffic to new service
    Instant rollback by disabling flag

**Monolith Decomposition Strategies:**

Identify Bounded Contexts:
  Use domain-driven design to find boundaries
  Look for modules with high internal cohesion
  Look for modules with low coupling to others
  Start with the most independent modules

Decomposition Approaches:

  By Business Capability:
    User Management -> User Service
    Order Processing -> Order Service
    Payment -> Payment Service
    Inventory -> Inventory Service

  By Subdomain:
    Core: Main business differentiator (extract carefully)
    Supporting: Important but not core (good candidates)
    Generic: Commodity functions (replace with SaaS)

  By Data Ownership:
    Identify which tables belong to which domain
    Extract tables along with the service
    Create APIs for cross-domain data access

Database Decomposition:

  Shared Database (starting point):
    All services share one database
    Simple but tightly coupled
    
  Database per Service (target):
    Each service owns its data
    Communication through APIs or events
    
  Migration steps:
    1. Add API layer over shared database
    2. Create new database for extracted service
    3. Dual-write to both databases
    4. Verify data consistency
    5. Switch reads to new database
    6. Remove writes to old database
    7. Remove old tables

  Data Synchronization during migration:
    Change Data Capture (CDC)
    Dual writes with consistency checks
    Event-driven synchronization
    Batch migration with incremental sync

**Anti-Corruption Layer:**
  Translates between old and new models
  Prevents legacy domain concepts from leaking into new code
  Adapter pattern between bounded contexts
  
  New Service -> Anti-Corruption Layer -> Legacy System
    Maps new domain model to legacy data model
    Translates API calls
    Handles data format differences

**Branch by Abstraction:**
  1. Create abstraction over the code to be replaced
  2. Existing code uses abstraction
  3. Build new implementation of abstraction
  4. Switch from old to new implementation
  5. Remove old implementation and abstraction
  
  Example:
    Step 1: Create NotificationSender interface
    Step 2: LegacyEmailNotifier implements NotificationSender
    Step 3: NewMultiChannelNotifier implements NotificationSender
    Step 4: Switch configuration to NewMultiChannelNotifier
    Step 5: Remove LegacyEmailNotifier

**Incremental Migration Best Practices:**

  Prioritize by business value:
    Which parts cause the most pain?
    Which parts need to change most frequently?
    Which parts have the highest business impact?

  Keep the migration reversible:
    Feature flags for instant rollback
    Run old and new in parallel
    Compare results before switching

  Measure success:
    Deployment frequency
    Lead time for changes
    Mean time to recovery
    Change failure rate

  Common pitfalls:
    Big bang rewrite (almost always fails)
    Underestimating data migration complexity
    Not maintaining the legacy system during migration
    Trying to migrate everything at once
    Not having a rollback plan

**API Gateway / Backend for Frontend (BFF):**

  API Gateway:
    Single entry point for all clients
    Request routing
    Authentication and authorization
    Rate limiting
    Response aggregation
    Protocol translation (REST to gRPC)

  BFF Pattern:
    Separate gateways for different client types
    Mobile BFF: Optimized for mobile (smaller payloads)
    Web BFF: Optimized for web (more features)
    Third-party BFF: Restricted API for partners
    
    Each BFF can aggregate data differently
    Reduces client-side complexity
    Allows independent evolution per client type`,
					CodeExamples: `// Migration and modernization patterns

// Anti-corruption layer
type LegacyOrderSystem interface {
    GetOrder(orderNumber string) (*LegacyOrder, error)
    CreateOrder(data map[string]interface{}) (string, error)
    UpdateOrderStatus(orderNumber string, status int) error
}

type LegacyOrder struct {
    OrderNum    string
    CustID      int
    TotalCents  int
    StatusCode  int
    CreatedDate string
}

// Anti-corruption layer translates between domains
type OrderAntiCorruptionLayer struct {
    legacy LegacyOrderSystem
}

func (acl *OrderAntiCorruptionLayer) GetOrder(ctx context.Context, id OrderID) (*Order, error) {
    legacyOrder, err := acl.legacy.GetOrder(string(id))
    if err != nil {
        return nil, fmt.Errorf("legacy order lookup failed: %w", err)
    }
    return acl.translateOrder(legacyOrder)
}

func (acl *OrderAntiCorruptionLayer) translateOrder(lo *LegacyOrder) (*Order, error) {
    createdAt, err := time.Parse("2006-01-02 15:04:05", lo.CreatedDate)
    if err != nil {
        return nil, fmt.Errorf("invalid date format: %w", err)
    }
    
    return &Order{
        ID:         OrderID(lo.OrderNum),
        CustomerID: CustomerID(fmt.Sprintf("cust-%d", lo.CustID)),
        Total:      Money{Amount: int64(lo.TotalCents), Currency: "USD"},
        Status:     acl.translateStatus(lo.StatusCode),
        CreatedAt:  createdAt,
    }, nil
}

func (acl *OrderAntiCorruptionLayer) translateStatus(code int) OrderStatus {
    switch code {
    case 1: return OrderStatusDraft
    case 2: return OrderStatusPlaced
    case 3: return OrderStatusPaid
    case 4: return OrderStatusShipped
    case 5: return OrderStatusDelivered
    case 9: return OrderStatusCancelled
    default: return OrderStatusDraft
    }
}

// Feature flag for gradual migration
type FeatureFlags struct {
    mu    sync.RWMutex
    flags map[string]*FeatureFlag
}

type FeatureFlag struct {
    Name       string
    Enabled    bool
    Percentage int // 0-100 for gradual rollout
}

func (ff *FeatureFlags) IsEnabled(name string, userID string) bool {
    ff.mu.RLock()
    defer ff.mu.RUnlock()
    
    flag, ok := ff.flags[name]
    if !ok || !flag.Enabled {
        return false
    }
    
    if flag.Percentage >= 100 {
        return true
    }
    
    // Consistent hashing for user-based rollout
    hash := fnv.New32a()
    hash.Write([]byte(name + ":" + userID))
    return int(hash.Sum32()%100) < flag.Percentage
}

// Router for strangler fig pattern
type MigrationRouter struct {
    legacyHandler http.Handler
    newHandlers   map[string]http.Handler
    flags         *FeatureFlags
    logger        *Logger
}

func NewMigrationRouter(legacy http.Handler, flags *FeatureFlags) *MigrationRouter {
    return &MigrationRouter{
        legacyHandler: legacy,
        newHandlers:   make(map[string]http.Handler),
        flags:         flags,
    }
}

func (r *MigrationRouter) RegisterNewHandler(pathPrefix string, handler http.Handler) {
    r.newHandlers[pathPrefix] = handler
}

func (r *MigrationRouter) ServeHTTP(w http.ResponseWriter, req *http.Request) {
    for prefix, handler := range r.newHandlers {
        if strings.HasPrefix(req.URL.Path, prefix) {
            flagName := "migration_" + strings.Trim(prefix, "/")
            userID := extractUserID(req)
            
            if r.flags.IsEnabled(flagName, userID) {
                r.logger.Info("routing to new service",
                    "path", req.URL.Path,
                    "user_id", userID,
                )
                handler.ServeHTTP(w, req)
                return
            }
        }
    }
    
    r.legacyHandler.ServeHTTP(w, req)
}

// Branch by Abstraction
type NotificationSender interface {
    Send(ctx context.Context, notification Notification) error
}

type Notification struct {
    Recipient string
    Channel   string
    Subject   string
    Body      string
    Metadata  map[string]string
}

// Old implementation
type LegacyEmailNotifier struct {
    smtpHost string
    smtpPort int
}

func (n *LegacyEmailNotifier) Send(ctx context.Context, notif Notification) error {
    // Legacy SMTP implementation
    return sendSMTP(n.smtpHost, n.smtpPort, notif.Recipient, notif.Subject, notif.Body)
}

// New implementation
type MultiChannelNotifier struct {
    channels map[string]ChannelSender
    logger   *Logger
}

type ChannelSender interface {
    Send(ctx context.Context, recipient, subject, body string) error
}

func (n *MultiChannelNotifier) Send(ctx context.Context, notif Notification) error {
    sender, ok := n.channels[notif.Channel]
    if !ok {
        return fmt.Errorf("unsupported channel: %s", notif.Channel)
    }
    
    err := sender.Send(ctx, notif.Recipient, notif.Subject, notif.Body)
    if err != nil {
        n.logger.Error("notification failed",
            "channel", notif.Channel,
            "recipient", notif.Recipient,
            "error", err,
        )
        return err
    }
    
    n.logger.Info("notification sent",
        "channel", notif.Channel,
        "recipient", notif.Recipient,
    )
    return nil
}

// Switchable implementation for migration
type SwitchableNotifier struct {
    legacy *LegacyEmailNotifier
    modern *MultiChannelNotifier
    flags  *FeatureFlags
}

func (n *SwitchableNotifier) Send(ctx context.Context, notif Notification) error {
    if n.flags.IsEnabled("use_modern_notifications", notif.Recipient) {
        return n.modern.Send(ctx, notif)
    }
    return n.legacy.Send(ctx, notif)
}

// Database migration helper
type DatabaseMigration struct {
    sourceDB *sql.DB
    targetDB *sql.DB
    logger   *Logger
}

func (m *DatabaseMigration) MigrateTable(ctx context.Context, table string, batchSize int) error {
    offset := 0
    for {
        query := fmt.Sprintf("SELECT * FROM %s ORDER BY id LIMIT %d OFFSET %d",
            table, batchSize, offset)
        
        rows, err := m.sourceDB.QueryContext(ctx, query)
        if err != nil {
            return fmt.Errorf("query source failed at offset %d: %w", offset, err)
        }
        
        count, err := m.insertBatch(ctx, table, rows)
        rows.Close()
        if err != nil {
            return fmt.Errorf("insert batch failed at offset %d: %w", offset, err)
        }
        
        if count == 0 {
            break
        }
        
        offset += batchSize
        m.logger.Info("migrated batch",
            "table", table,
            "offset", offset,
            "count", count,
        )
    }
    
    return nil
}

func (m *DatabaseMigration) insertBatch(ctx context.Context, table string, rows *sql.Rows) (int, error) {
    columns, err := rows.Columns()
    if err != nil {
        return 0, err
    }
    
    count := 0
    tx, err := m.targetDB.BeginTx(ctx, nil)
    if err != nil {
        return 0, err
    }
    defer tx.Rollback()
    
    placeholders := make([]string, len(columns))
    for i := range placeholders {
        placeholders[i] = fmt.Sprintf("$%d", i+1)
    }
    insertSQL := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s) ON CONFLICT DO NOTHING",
        table,
        strings.Join(columns, ", "),
        strings.Join(placeholders, ", "),
    )
    
    stmt, err := tx.PrepareContext(ctx, insertSQL)
    if err != nil {
        return 0, err
    }
    defer stmt.Close()
    
    for rows.Next() {
        values := make([]interface{}, len(columns))
        valuePtrs := make([]interface{}, len(columns))
        for i := range values {
            valuePtrs[i] = &values[i]
        }
        
        if err := rows.Scan(valuePtrs...); err != nil {
            return count, err
        }
        
        if _, err := stmt.ExecContext(ctx, values...); err != nil {
            return count, err
        }
        count++
    }
    
    return count, tx.Commit()
}`,
				},
			},
		},
	})
}
