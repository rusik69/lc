package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1249,
			Title:       "Azure Database Services",
			Description: "Master Azure database services including Azure SQL, Cosmos DB, PostgreSQL, MySQL, and Redis Cache for various workload patterns.",
			Order:       49,
			Lessons: []problems.Lesson{
				{
					Title: "Azure SQL and Relational Databases",
					Content: `Azure offers managed relational database services that handle patching, backups, and high availability automatically.

**Azure SQL Database:**
` + "```" + `
Deployment models:
  Single Database:    Individual database with dedicated resources
  Elastic Pool:       Share resources among multiple databases
  Managed Instance:   Near 100% SQL Server compatibility, VNet native

Purchasing models:
  DTU (Database Transaction Unit):
    - Bundled CPU, memory, I/O
    - Basic, Standard, Premium tiers
    - Simple, predictable pricing
  
  vCore:
    - Choose CPU, memory, storage independently
    - General Purpose, Business Critical, Hyperscale
    - Azure Hybrid Benefit (use existing SQL licenses)

Service tiers (vCore):
  General Purpose:
    - Remote storage (Azure Premium Storage)
    - 5-10ms I/O latency
    - Up to 128 vCores, 3TB data
    - Zone redundant available
  
  Business Critical:
    - Local SSD storage
    - <2ms I/O latency
    - Built-in read replica
    - In-memory OLTP
    - Up to 128 vCores, 4TB data
  
  Hyperscale:
    - Up to 100TB data
    - Near-instant backups
    - Fast scale up/down
    - Up to 4 named replicas
    - Up to 30 HA replicas

Create Azure SQL:
  # Create server
  az sql server create \
    --resource-group myRG \
    --name mysqlserver-unique \
    --admin-user sqladmin \
    --admin-password "SecurePass123!" \
    --location eastus
  
  # Create database
  az sql db create \
    --resource-group myRG \
    --server mysqlserver-unique \
    --name mydb \
    --edition GeneralPurpose \
    --compute-model Serverless \
    --auto-pause-delay 60 \
    --min-capacity 0.5 \
    --max-size 32GB
  
  # Firewall rules
  az sql server firewall-rule create \
    --resource-group myRG --server mysqlserver-unique \
    --name AllowMyIP --start-ip-address 203.0.113.1 --end-ip-address 203.0.113.1
  
  # Allow Azure services
  az sql server firewall-rule create \
    --resource-group myRG --server mysqlserver-unique \
    --name AllowAzure --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0

Elastic Pool:
  az sql elastic-pool create \
    --resource-group myRG --server mysqlserver-unique \
    --name mypool --edition GeneralPurpose \
    --capacity 4 --db-max-capacity 2 --db-min-capacity 0.25
  
  az sql db create \
    --resource-group myRG --server mysqlserver-unique \
    --name db1 --elastic-pool mypool
` + "```" + `

**Azure Database for PostgreSQL:**
` + "```" + `
Deployment modes:
  Flexible Server (recommended):
    - Zone-redundant HA
    - Point-in-time restore (35 days)
    - Burstable, General Purpose, Memory Optimized
    - PgBouncer built-in
    - PostgreSQL 12-16
  
  Cosmos DB for PostgreSQL (formerly Hyperscale/Citus):
    - Distributed PostgreSQL
    - Horizontal scaling (sharding)
    - Multi-node clusters

Create Flexible Server:
  az postgres flexible-server create \
    --resource-group myRG \
    --name mypostgres-unique \
    --location eastus \
    --admin-user pgadmin \
    --admin-password "SecurePass123!" \
    --sku-name Standard_D2s_v3 \
    --tier GeneralPurpose \
    --storage-size 128 \
    --version 16 \
    --high-availability ZoneRedundant \
    --zone 1 --standby-zone 2
  
  # Create database
  az postgres flexible-server db create \
    --resource-group myRG --server-name mypostgres-unique \
    --database-name myappdb
  
  # Configure server parameters
  az postgres flexible-server parameter set \
    --resource-group myRG --server-name mypostgres-unique \
    --name shared_buffers --value "256MB"
  
  # Configure firewall
  az postgres flexible-server firewall-rule create \
    --resource-group myRG --name mypostgres-unique \
    --rule-name AllowMyIP \
    --start-ip-address 203.0.113.1 --end-ip-address 203.0.113.1
  
  # VNet integration (private access)
  az postgres flexible-server create \
    --resource-group myRG --name mypostgres-private \
    --vnet myVNet --subnet db-subnet \
    --private-dns-zone mypostgres.private.postgres.database.azure.com

Backup and restore:
  # Point-in-time restore
  az postgres flexible-server restore \
    --resource-group myRG \
    --name mypostgres-restored \
    --source-server mypostgres-unique \
    --restore-time "2024-01-15T10:30:00Z"
  
  # Geo-restore
  az postgres flexible-server geo-restore \
    --resource-group myRG \
    --name mypostgres-geo \
    --source-server mypostgres-unique \
    --location westus
` + "```" + `

**Azure Cache for Redis:**
` + "```" + `
Managed Redis for caching, session store, and messaging.

Tiers:
  Basic:       Single node, no SLA, dev/test
  Standard:    Replicated (primary + replica), 99.9% SLA
  Premium:     Clustering, persistence, VNet, geo-replication
  Enterprise:  Redis Labs, RediSearch, RedisBloom, RedisTimeSeries

Create:
  az redis create \
    --resource-group myRG \
    --name myredis-unique \
    --location eastus \
    --sku Premium --vm-size P1 \
    --replicas-per-master 1 \
    --minimum-tls-version 1.2

  # Get connection info
  az redis show --resource-group myRG --name myredis-unique \
    --query "{host:hostName, port:sslPort}" -o json
  
  az redis list-keys --resource-group myRG --name myredis-unique

  # Enable clustering (Premium)
  az redis create \
    --resource-group myRG --name myredis-cluster \
    --sku Premium --vm-size P1 \
    --shard-count 3

  # Private endpoint
  az network private-endpoint create \
    --resource-group myRG --name redis-pe \
    --vnet-name myVNet --subnet cache-subnet \
    --private-connection-resource-id /subscriptions/.../redis/myredis-unique \
    --group-ids redisCache --connection-name redis-connection
` + "```" + ``,
					CodeExamples: `# Azure database management

# 1. Database health check
#!/bin/bash
echo "=== Azure Database Health Check ==="

# SQL Databases
echo "--- Azure SQL ---"
for server in $(az sql server list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az sql server show --name "$server" --query "resourceGroup" -o tsv)
    echo "Server: $server (RG: $RG)"
    
    az sql db list --server "$server" --resource-group "$RG" \
        --query "[?name!='master'].{
            name:name, status:status, tier:currentServiceObjectiveName,
            size:maxSizeBytes, location:location
        }" -o table 2>/dev/null
done

# PostgreSQL
echo ""
echo "--- PostgreSQL Flexible Servers ---"
for server in $(az postgres flexible-server list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az postgres flexible-server show --name "$server" --query "resourceGroup" -o tsv 2>/dev/null)
    
    az postgres flexible-server show --name "$server" --resource-group "$RG" \
        --query "{name:name, state:state, version:version, sku:sku.name, ha:highAvailability.mode}" \
        -o json 2>/dev/null | jq .
done

# Redis
echo ""
echo "--- Azure Cache for Redis ---"
az redis list --query "[].{
    name:name, rg:resourceGroup, sku:sku.name,
    version:redisVersion, ssl:minimumTlsVersion,
    state:provisioningState
}" -o table 2>/dev/null

# 2. SQL DTU usage monitor
#!/bin/bash
SERVER="${1:?Usage: $0 <server-name>}"
RG=$(az sql server show --name "$SERVER" --query "resourceGroup" -o tsv)

echo "=== SQL DTU Usage: $SERVER ==="

for db in $(az sql db list --server "$SERVER" -g "$RG" \
    --query "[?name!='master'].name" -o tsv 2>/dev/null); do
    
    echo ""
    echo "Database: $db"
    
    # Current DTU metrics
    az monitor metrics list \
        --resource "/subscriptions/$(az account show -q id -o tsv)/resourceGroups/$RG/providers/Microsoft.Sql/servers/$SERVER/databases/$db" \
        --metric "dtu_consumption_percent" \
        --interval PT1H \
        --query "value[0].timeseries[0].data[-5:].{time:timeStamp, avg:average}" \
        -o table 2>/dev/null
done

# 3. Database backup verification
#!/bin/bash
echo "=== Database Backup Status ==="

echo "--- SQL Database Backups ---"
for server in $(az sql server list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az sql server show --name "$server" --query "resourceGroup" -o tsv)
    
    for db in $(az sql db list --server "$server" -g "$RG" \
        --query "[?name!='master'].name" -o tsv 2>/dev/null); do
        
        RETENTION=$(az sql db show --server "$server" -g "$RG" -n "$db" \
            --query "earliestRestoreDate" -o tsv 2>/dev/null)
        
        printf "  %-20s %-20s  Earliest restore: %s\n" \
            "$server" "$db" "${RETENTION:-unknown}"
    done
done`,
				},
				{
					Title: "Azure Cosmos DB",
					Content: `Azure Cosmos DB is a globally distributed, multi-model NoSQL database service with guaranteed single-digit millisecond latency.

**Cosmos DB Fundamentals:**
` + "```" + `
Key features:
  - Global distribution (any Azure region)
  - 5 consistency levels
  - Single-digit millisecond SLA at 99th percentile
  - Automatic and instant scalability
  - Multiple APIs: NoSQL, MongoDB, Cassandra, Gremlin, Table

APIs:
  NoSQL (Core):     Document DB, SQL-like queries, native SDK
  MongoDB:          Wire protocol compatible, existing tools work
  Cassandra:        CQL compatible, wide-column store
  Gremlin:          Graph database queries
  Table:            Azure Table Storage compatible (better performance)

Consistency levels (strongest to weakest):
  Strong:           Linearizable reads, global ordering
                    Highest latency, lowest throughput
  Bounded Staleness: Reads lag behind writes by k versions or t time
                     Consistent within region
  Session:          Default. Read-your-own-writes guarantee
                    Best balance of consistency and performance
  Consistent Prefix: Reads never see out-of-order writes
                     No gaps in ordering
  Eventual:         No ordering guarantee, lowest latency
                    Highest throughput

Request Units (RU/s):
  - Normalized measure of throughput
  - 1 RU = 1 point read of 1KB document by ID
  - Write costs ~5.5x read
  - Queries cost depends on complexity
  - Provisioned or serverless mode
  
  Provisioning:
    Manual:     Set specific RU/s (minimum 400)
    Autoscale:  Set max RU/s, scales between 10%-100%
    Serverless: Pay per request (dev/test, sporadic workloads)

Create Cosmos DB:
  # Create account
  az cosmosdb create \
    --resource-group myRG \
    --name mycosmosdb-unique \
    --kind GlobalDocumentDB \
    --default-consistency-level Session \
    --locations regionName=eastus failoverPriority=0 isZoneRedundant=true \
    --locations regionName=westus failoverPriority=1 isZoneRedundant=false \
    --enable-automatic-failover true
  
  # Create database
  az cosmosdb sql database create \
    --resource-group myRG \
    --account-name mycosmosdb-unique \
    --name mydb \
    --throughput 400
  
  # Create container with autoscale
  az cosmosdb sql container create \
    --resource-group myRG \
    --account-name mycosmosdb-unique \
    --database-name mydb \
    --name users \
    --partition-key-path "/userId" \
    --max-throughput 4000
  
  # Create with unique key and TTL
  az cosmosdb sql container create \
    --resource-group myRG \
    --account-name mycosmosdb-unique \
    --database-name mydb \
    --name events \
    --partition-key-path "/eventType" \
    --default-ttl 2592000 \
    --unique-key-policy '{"uniqueKeys":[{"paths":["/eventId"]}]}'
` + "```" + `

**Partition Strategy:**
` + "```" + `
Choosing partition key is the most important Cosmos DB design decision.

Good partition key properties:
  1. High cardinality (many distinct values)
  2. Even distribution of data and requests
  3. Used in most queries as filter

Common patterns:
  Users:          /userId
  Orders:         /customerId (not /orderId for range queries)
  IoT telemetry:  /deviceId
  Multi-tenant:   /tenantId
  Events:         /eventType + hierarchical partitioning
  
  Bad choices:
    /status (low cardinality: active/inactive)
    /timestamp (sequential = hot partition)
    /country (uneven distribution)

Hierarchical partition keys (preview):
  Up to 3 levels for better distribution
  /tenantId → /userId → /sessionId

Cross-partition queries:
  - Queries without partition key = fan-out (expensive)
  - Always include partition key in WHERE clause
  - Use change feed for cross-partition aggregations

Physical vs Logical partitions:
  Logical:   Your partition key value
  Physical:  Azure managed, up to 50GB and 10,000 RU/s each
  Split:     Azure automatically splits when physical partition grows
` + "```" + `

**Querying and Indexing:**
` + "```" + `
SQL API queries:
  SELECT * FROM c WHERE c.userId = 'user123'
  SELECT c.name, c.email FROM c WHERE c.age > 25
  SELECT VALUE COUNT(1) FROM c WHERE c.status = 'active'
  SELECT c.name FROM c JOIN t IN c.tags WHERE t = 'vip'
  SELECT TOP 10 * FROM c ORDER BY c.createdAt DESC
  
  # Aggregate functions
  SELECT VALUE AVG(c.price) FROM c
  SELECT c.category, COUNT(1) as cnt FROM c GROUP BY c.category

Indexing policy:
  Default: All properties automatically indexed
  
  Custom policy:
  {
    "indexingMode": "consistent",
    "includedPaths": [
      { "path": "/name/?" },
      { "path": "/userId/?" },
      { "path": "/createdAt/?" }
    ],
    "excludedPaths": [
      { "path": "/description/?" },
      { "path": "/largePayload/*" }
    ],
    "compositeIndexes": [
      [
        { "path": "/userId", "order": "ascending" },
        { "path": "/createdAt", "order": "descending" }
      ]
    ],
    "spatialIndexes": [
      { "path": "/location/*", "types": ["Point"] }
    ]
  }

Change Feed:
  - Stream of changes to documents
  - Ordered within partition
  - Process with Azure Functions or SDK
  - Use for: event sourcing, materialized views,
    real-time analytics, cross-partition aggregation
` + "```" + ``,
					CodeExamples: `# Cosmos DB management

# 1. Cosmos DB cost optimizer
#!/bin/bash
echo "=== Cosmos DB Cost Analysis ==="

for account in $(az cosmosdb list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az cosmosdb show --name "$account" --query "resourceGroup" -o tsv)
    
    echo ""
    echo "Account: $account (RG: $RG)"
    
    # List databases and their throughput
    for db in $(az cosmosdb sql database list \
        --account-name "$account" -g "$RG" \
        --query "[].name" -o tsv 2>/dev/null); do
        
        echo "  Database: $db"
        
        # Check database-level throughput
        DB_TP=$(az cosmosdb sql database throughput show \
            --account-name "$account" -g "$RG" --name "$db" \
            --query "resource.throughput" -o tsv 2>/dev/null)
        
        if [ -n "$DB_TP" ] && [ "$DB_TP" != "None" ]; then
            echo "    Shared throughput: ${DB_TP} RU/s"
        fi
        
        # Check container-level throughput
        for container in $(az cosmosdb sql container list \
            --account-name "$account" -g "$RG" --database-name "$db" \
            --query "[].name" -o tsv 2>/dev/null); do
            
            TP=$(az cosmosdb sql container throughput show \
                --account-name "$account" -g "$RG" \
                --database-name "$db" --name "$container" \
                --query "resource.{throughput:throughput, autoscale:autoscaleSettings.maxThroughput}" \
                -o json 2>/dev/null)
            
            MANUAL=$(echo "$TP" | jq -r '.throughput // empty')
            AUTOSCALE=$(echo "$TP" | jq -r '.autoscale // empty')
            
            if [ -n "$AUTOSCALE" ]; then
                echo "    Container: $container — Autoscale max: ${AUTOSCALE} RU/s"
            elif [ -n "$MANUAL" ]; then
                echo "    Container: $container — Manual: ${MANUAL} RU/s"
            fi
        done
    done
done

# 2. Cosmos DB replication status
#!/bin/bash
echo "=== Cosmos DB Replication Status ==="

for account in $(az cosmosdb list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az cosmosdb show --name "$account" --query "resourceGroup" -o tsv)
    
    echo ""
    echo "Account: $account"
    
    # Show regions
    az cosmosdb show --name "$account" -g "$RG" \
        --query "locations[].{region:locationName, priority:failoverPriority, zone:isZoneRedundant}" \
        -o table 2>/dev/null
    
    # Consistency
    CONSISTENCY=$(az cosmosdb show --name "$account" -g "$RG" \
        --query "consistencyPolicy.defaultConsistencyLevel" -o tsv)
    echo "  Consistency: $CONSISTENCY"
    
    # Automatic failover
    FAILOVER=$(az cosmosdb show --name "$account" -g "$RG" \
        --query "enableAutomaticFailover" -o tsv)
    echo "  Auto-failover: $FAILOVER"
done`,
				},
			},
		},
	})
}
