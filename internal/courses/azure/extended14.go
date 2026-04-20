package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1261,
			Title:       "Azure Data Services and Analytics",
			Description: "Learn Azure data services including Azure SQL, Cosmos DB, Data Factory, Synapse Analytics, and data lake architectures.",
			Order:       61,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Database Services Deep Dive",
					Content: `Azure provides managed database services for relational, NoSQL, in-memory, and graph workloads.

**Azure SQL Database:**
` + "```" + `
Deployment models:
  Single Database:
    - Individual database with dedicated resources
    - DTU or vCore purchasing model
    - Serverless option (auto-pause, auto-scale)
  
  Elastic Pool:
    - Multiple databases sharing resources
    - Cost-effective for variable workloads
    - eDTU or vCore pricing
  
  Managed Instance:
    - Near 100% SQL Server compatibility
    - VNet integration (private)
    - Best for migration from on-premises
  
  SQL Server on Azure VM:
    - Full SQL Server instance
    - OS-level access
    - For workloads requiring OS customization

Create Azure SQL:
  # Server
  az sql server create \
    --name mysqlserver --resource-group myRG \
    --location eastus \
    --admin-user sqladmin \
    --admin-password 'P@ssw0rd123!'
  
  # Database (serverless)
  az sql db create \
    --server mysqlserver -g myRG \
    --name mydb --edition GeneralPurpose \
    --compute-model Serverless \
    --auto-pause-delay 60 \
    --min-capacity 0.5 --capacity 2 \
    --backup-storage-redundancy Local
  
  # Database (provisioned)
  az sql db create \
    --server mysqlserver -g myRG \
    --name proddb --edition BusinessCritical \
    --capacity 2 --max-size 100GB

Security:
  # Firewall rules
  az sql server firewall-rule create \
    --server mysqlserver -g myRG \
    --name AllowMyIP \
    --start-ip-address 203.0.113.1 \
    --end-ip-address 203.0.113.1
  
  # Private endpoint
  az network private-endpoint create \
    --name sql-pe -g myRG \
    --vnet-name myVNet --subnet data-subnet \
    --private-connection-resource-id $(az sql server show -n mysqlserver -g myRG --query id -o tsv) \
    --group-id sqlServer \
    --connection-name sql-connection
  
  # Azure AD authentication
  az sql server ad-admin create \
    --server mysqlserver -g myRG \
    --display-name "DBA Group" \
    --object-id <group-object-id>
  
  # TDE (Transparent Data Encryption)
  # Enabled by default with service-managed key
  # Can use customer-managed key via Key Vault
  
  # Auditing
  az sql server audit-policy update \
    --server mysqlserver -g myRG \
    --state Enabled \
    --storage-account mystorageacct

High availability:
  Basic/Standard/General Purpose:
    - Remote storage replication
    - 99.99% SLA
  
  Premium/Business Critical:
    - Local SSD, Always On AG
    - 99.99% SLA
    - Read replicas (1 free)
  
  Hyperscale:
    - Up to 100TB
    - Near-instant backups
    - Up to 4 read replicas
    - 99.99% SLA
  
  Geo-replication:
    az sql db replica create \
      --server mysqlserver -g myRG --name mydb \
      --partner-server mysqlserver-west \
      --partner-resource-group myRG-west
  
  Failover groups:
    az sql failover-group create \
      --server mysqlserver -g myRG \
      --partner-server mysqlserver-west \
      --partner-resource-group myRG-west \
      --name myfailover \
      --failover-policy Automatic \
      --grace-period 1
` + "```" + `

**Azure Cosmos DB:**
` + "```" + `
Multi-model, globally distributed database.

APIs:
  NoSQL (native): JSON documents, SQL-like queries
  MongoDB:        MongoDB wire protocol
  Cassandra:      CQL compatible
  Gremlin:        Graph database
  Table:          Azure Table Storage compatible
  PostgreSQL:     Distributed PostgreSQL (Citus)

Create Cosmos DB:
  # NoSQL API
  az cosmosdb create \
    --name mycosmosdb -g myRG \
    --default-consistency-level Session \
    --locations regionName=eastus failoverPriority=0 \
    --locations regionName=westus failoverPriority=1 \
    --enable-automatic-failover true
  
  # Create database
  az cosmosdb sql database create \
    --account-name mycosmosdb -g myRG \
    --name mydb
  
  # Create container
  az cosmosdb sql container create \
    --account-name mycosmosdb -g myRG \
    --database-name mydb --name orders \
    --partition-key-path "/customerId" \
    --throughput 400
  
  # Serverless mode
  az cosmosdb create \
    --name mycosmosdb-serverless -g myRG \
    --capabilities EnableServerless \
    --default-consistency-level Session

Consistency levels:
  Strong:            Linearizable reads, highest latency
  Bounded staleness: Reads lag by K versions or T time
  Session:           Read-your-own-writes (default, recommended)
  Consistent prefix: Reads never see out-of-order writes
  Eventual:          No ordering guarantee, lowest latency

Throughput:
  Provisioned:  Fixed RU/s (manual or autoscale)
    Manual:     400 - 1,000,000 RU/s
    Autoscale:  100 - max RU/s (auto adjusts)
  
  Serverless:   Pay per request, max 5000 RU/s per container
  
  RU (Request Unit):
    1 RU = 1 point read of 1KB document by ID
    Writes cost ~5-6x reads
    Queries cost depends on complexity

Partition strategy:
  Good partition keys:
    - High cardinality (many distinct values)
    - Even distribution
    - Used in WHERE clauses
  
  Examples:
    Orders:    /customerId or /orderId
    IoT:       /deviceId
    Users:     /userId
    Logs:      /tenantId (multi-tenant)
  
  Anti-patterns:
    - Low cardinality (status, country)
    - Monotonically increasing (timestamp)
    - Single value for all docs

Global distribution:
  - Multi-region writes
  - <10ms read/write at 99th percentile
  - Automatic conflict resolution
  - Five consistency models
  - Transparent failover
` + "```" + ``,
					CodeExamples: `# Azure database management scripts

# 1. Database monitoring dashboard
#!/bin/bash
echo "=== Azure Database Monitor ==="

# SQL databases
echo "--- Azure SQL Databases ---"
for server in $(az sql server list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az sql server list --query "[?name=='$server'].resourceGroup" -o tsv | head -1)
    echo "Server: $server ($RG)"
    
    az sql db list -s "$server" -g "$RG" \
        --query "[?name!='master'].{
            name:name, status:status,
            sku:currentSku.name, maxSize:maxSizeBytes,
            zoneRedundant:zoneRedundant
        }" -o table 2>/dev/null
    
    # DTU/CPU usage
    az sql db list-usages -s "$server" -g "$RG" -n "$(
        az sql db list -s "$server" -g "$RG" --query "[?name!='master'].name | [0]" -o tsv 2>/dev/null
    )" --query "[?name=='database_size'].{used:currentValue, limit:limit}" -o table 2>/dev/null
done

# Cosmos DB accounts
echo ""
echo "--- Cosmos DB Accounts ---"
for acct in $(az cosmosdb list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az cosmosdb list --query "[?name=='$acct'].resourceGroup" -o tsv | head -1)
    echo "Account: $acct ($RG)"
    
    # Regions
    az cosmosdb show -n "$acct" -g "$RG" \
        --query "locations[].{region:locationName, priority:failoverPriority}" \
        -o table 2>/dev/null
    
    # Databases
    az cosmosdb sql database list \
        --account-name "$acct" -g "$RG" \
        --query "[].{name:name}" -o table 2>/dev/null
done

# 2. Cosmos DB throughput manager
#!/bin/bash
echo "=== Cosmos DB Throughput ==="

ACCOUNT="${1:-mycosmosdb}"
RG="${2:-myRG}"

for db in $(az cosmosdb sql database list \
    --account-name "$ACCOUNT" -g "$RG" \
    --query "[].name" -o tsv 2>/dev/null); do
    
    echo "Database: $db"
    
    for container in $(az cosmosdb sql container list \
        --account-name "$ACCOUNT" -g "$RG" \
        --database-name "$db" \
        --query "[].name" -o tsv 2>/dev/null); do
        
        THROUGHPUT=$(az cosmosdb sql container throughput show \
            --account-name "$ACCOUNT" -g "$RG" \
            --database-name "$db" --name "$container" \
            --query "resource.throughput" -o tsv 2>/dev/null)
        
        AUTOSCALE=$(az cosmosdb sql container throughput show \
            --account-name "$ACCOUNT" -g "$RG" \
            --database-name "$db" --name "$container" \
            --query "resource.autoscaleSettings.maxThroughput" -o tsv 2>/dev/null)
        
        PK=$(az cosmosdb sql container show \
            --account-name "$ACCOUNT" -g "$RG" \
            --database-name "$db" --name "$container" \
            --query "resource.partitionKey.paths[0]" -o tsv 2>/dev/null)
        
        if [ -n "$AUTOSCALE" ]; then
            echo "  $container: Autoscale max=$AUTOSCALE RU/s, PK=$PK"
        else
            echo "  $container: Provisioned $THROUGHPUT RU/s, PK=$PK"
        fi
    done
done

# 3. SQL failover group status
#!/bin/bash
echo "=== SQL Failover Groups ==="

for server in $(az sql server list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az sql server list --query "[?name=='$server'].resourceGroup" -o tsv | head -1)
    
    GROUPS=$(az sql failover-group list -s "$server" -g "$RG" \
        --query "[].name" -o tsv 2>/dev/null)
    
    for fg in $GROUPS; do
        echo "Failover Group: $fg"
        az sql failover-group show -s "$server" -g "$RG" -n "$fg" \
            --query "{
                primary:partnerServers[0].replicationRole,
                replicationState:partnerServers[0].replicationState,
                databases:databases | length(@),
                readWriteEndpoint:readWriteEndpoint.failoverPolicy,
                gracePeriod:readWriteEndpoint.failoverWithDataLossGracePeriodMinutes
            }" -o json 2>/dev/null | jq .
    done
done`,
				},
				{
					Title: "Azure Analytics and Data Processing",
					Content: `Azure provides comprehensive analytics services for batch processing, real-time analytics, and data warehousing.

**Azure Synapse Analytics:**
` + "```" + `
Azure Synapse = Data warehouse + Big data analytics + Data integration

Components:
  Dedicated SQL Pool:
    - Formerly SQL Data Warehouse
    - Massively Parallel Processing (MPP)
    - Columnar storage
    - Petabyte-scale
    - DWU-based scaling
  
  Serverless SQL Pool:
    - Query data in place (Data Lake)
    - No infrastructure to manage
    - Pay per TB processed
    - T-SQL support
  
  Apache Spark Pool:
    - Big data processing
    - ML model training
    - Support for Python, Scala, .NET, R
    - Auto-scale and auto-pause
  
  Data Explorer Pool:
    - Log and telemetry analytics
    - KQL (Kusto Query Language)
    - Near real-time ingestion
  
  Pipelines:
    - Data integration (like Data Factory)
    - ETL/ELT workflows
    - 90+ connectors

Create Synapse workspace:
  az synapse workspace create \
    --name mysynapse -g myRG \
    --storage-account mystorageaccount \
    --file-system synapse-data \
    --sql-admin-login-user sqladmin \
    --sql-admin-login-password 'P@ssw0rd123!' \
    --location eastus

  # Dedicated SQL pool
  az synapse sql pool create \
    --workspace-name mysynapse -g myRG \
    --name mydw --performance-level DW100c
  
  # Spark pool
  az synapse spark pool create \
    --workspace-name mysynapse -g myRG \
    --name myspark \
    --spark-version 3.3 \
    --node-count 3 --node-size Medium \
    --enable-auto-pause true --auto-pause-delay 15 \
    --enable-auto-scale true \
    --min-node-count 3 --max-node-count 10

Data warehouse patterns:
  Star schema:
    Fact tables (measures) + Dimension tables (attributes)
    
    FactSales:
      SaleKey, DateKey, ProductKey, CustomerKey,
      Quantity, Revenue, Discount
    
    DimProduct:
      ProductKey, Name, Category, Price
    
    DimDate:
      DateKey, Date, Month, Quarter, Year
    
    DimCustomer:
      CustomerKey, Name, Region, Segment
  
  Distribution types:
    HASH:        Distribute by column (large fact tables)
    ROUND_ROBIN: Even distribution (default, staging)
    REPLICATE:   Full copy on each node (small dim tables)
  
  CREATE TABLE FactSales (
    SaleKey BIGINT,
    DateKey INT,
    ProductKey INT,
    Revenue DECIMAL(18,2)
  )
  WITH (
    DISTRIBUTION = HASH(ProductKey),
    CLUSTERED COLUMNSTORE INDEX
  );
` + "```" + `

**Azure Data Factory:**
` + "```" + `
Cloud ETL/ELT service for data integration.

Components:
  Pipeline:    Workflow of activities
  Activity:    A step (copy, transform, control)
  Dataset:     Data structure reference
  Linked Service: Connection string to data stores
  Trigger:     Schedule or event-based execution
  Integration Runtime: Compute for activities

Create Data Factory:
  az datafactory create \
    --name myaDF -g myRG --location eastus

Common activities:
  Copy Activity:
    - Move data between 90+ sources
    - No code, configuration-driven
    - Parallel copy, compression
  
  Data Flow:
    - Visual data transformation
    - Runs on Spark cluster
    - Join, aggregate, pivot, etc.
  
  Azure Function:
    - Call Azure Functions
  
  Stored Procedure:
    - Execute SQL stored procedures
  
  Web Activity:
    - Call REST APIs
  
  Control flow:
    - ForEach, If/Else, Switch
    - Wait, Until, Set Variable
    - Execute Pipeline (nested)

Pipeline patterns:
  Delta loading:
    1. Get last watermark (max date)
    2. Query source for new/changed rows
    3. Copy to destination
    4. Update watermark
  
  Full refresh:
    1. Truncate destination
    2. Copy all data from source
  
  Slowly Changing Dimensions:
    Type 1: Overwrite (no history)
    Type 2: New row with version (keep history)
    Type 3: Add columns (limited history)

Monitoring:
  # Pipeline runs
  az datafactory pipeline-run query-by-factory \
    --factory-name myaDF -g myRG \
    --last-updated-after "2024-01-01T00:00:00Z" \
    --last-updated-before "2024-01-31T23:59:59Z" \
    --filters '[{"operand": "Status", "operator": "Equals", "values": ["Failed"]}]'
` + "```" + `

**Azure Data Lake and Stream Analytics:**
` + "```" + `
Azure Data Lake Storage Gen2:
  - Hadoop-compatible file system
  - Built on Blob Storage
  - Hierarchical namespace
  - Fine-grained ACLs (POSIX-like)
  - Low cost for massive data
  
  Create:
  az storage account create -n mydatalake -g myRG \
    --sku Standard_LRS --kind StorageV2 \
    --hierarchical-namespace true
  
  # Create file system (container)
  az storage fs create -n raw --account-name mydatalake
  az storage fs create -n processed --account-name mydatalake
  az storage fs create -n curated --account-name mydatalake
  
  Data lake zones:
    raw/           Landing zone (as-is from source)
    processed/     Cleaned, validated, standardized
    curated/       Business-ready, aggregated
    sandbox/       Exploration, data science

Azure Stream Analytics:
  - Real-time event processing
  - SQL-like query language
  - Input: Event Hub, IoT Hub, Blob Storage
  - Output: SQL, Cosmos DB, Blob, Power BI, etc.
  
  az stream-analytics job create \
    --name mystreamjob -g myRG \
    --location eastus \
    --compatibility-level 1.2
  
  Example query:
    SELECT
      System.Timestamp() AS WindowEnd,
      DeviceId,
      AVG(Temperature) AS AvgTemp,
      MAX(Temperature) AS MaxTemp,
      COUNT(*) AS EventCount
    FROM IoTInput
    TIMESTAMP BY EventTime
    GROUP BY
      DeviceId,
      TumblingWindow(minute, 5)
    HAVING AVG(Temperature) > 100

Azure HDInsight:
  Managed clusters for:
  - Apache Spark
  - Apache Hadoop
  - Apache Kafka
  - Apache HBase
  - Apache Storm

Azure Databricks:
  - Apache Spark-based analytics
  - Collaborative notebooks
  - Unity Catalog (data governance)
  - Delta Lake (ACID on data lake)
  - MLflow integration
` + "```" + ``,
					CodeExamples: `# Azure analytics scripts

# 1. Data Lake health check
#!/bin/bash
echo "=== Data Lake Status ==="

ACCOUNT="${1:-mydatalake}"
RG="${2:-myRG}"

echo "Storage Account: $ACCOUNT"
az storage account show -n "$ACCOUNT" -g "$RG" \
    --query "{sku:sku.name, kind:kind, hns:isHnsEnabled, access:accessTier}" \
    -o json 2>/dev/null | jq .

# File systems
echo ""
echo "--- File Systems ---"
for fs in $(az storage fs list --account-name "$ACCOUNT" \
    --query "[].name" -o tsv --auth-mode login 2>/dev/null); do
    
    # Count files and total size
    echo "  $fs:"
    az storage fs file list -f "$fs" --account-name "$ACCOUNT" \
        --auth-mode login --recursive \
        --query "length(@)" -o tsv 2>/dev/null | xargs -I {} echo "    Files: {}"
done

# 2. Synapse Analytics dashboard
#!/bin/bash
echo "=== Synapse Analytics Status ==="

for ws in $(az synapse workspace list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az synapse workspace list --query "[?name=='$ws'].resourceGroup" -o tsv | head -1)
    echo "Workspace: $ws ($RG)"
    
    # SQL pools
    echo "  SQL Pools:"
    az synapse sql pool list --workspace-name "$ws" -g "$RG" \
        --query "[].{name:name, status:status, sku:sku.name}" \
        -o table 2>/dev/null
    
    # Spark pools
    echo "  Spark Pools:"
    az synapse spark pool list --workspace-name "$ws" -g "$RG" \
        --query "[].{name:name, nodes:nodeCount, size:nodeSize, spark:sparkVersion}" \
        -o table 2>/dev/null
done

# 3. Data Factory pipeline monitor
#!/bin/bash
echo "=== Data Factory Pipeline Monitor ==="

for factory in $(az datafactory list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az datafactory list --query "[?name=='$factory'].resourceGroup" -o tsv | head -1)
    echo "Factory: $factory ($RG)"
    
    # List pipelines
    az datafactory pipeline list \
        --factory-name "$factory" -g "$RG" \
        --query "[].{name:name}" -o table 2>/dev/null
    
    # Recent runs (last 24h)
    AFTER=$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v-24H +%Y-%m-%dT%H:%M:%SZ)
    BEFORE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    echo "  Recent runs (24h):"
    az datafactory pipeline-run query-by-factory \
        --factory-name "$factory" -g "$RG" \
        --last-updated-after "$AFTER" \
        --last-updated-before "$BEFORE" \
        --query "value[].{
            pipeline:pipelineName,
            status:status,
            start:runStart,
            duration:durationInMs
        }" -o table 2>/dev/null
done

# 4. Stream Analytics monitor
#!/bin/bash
echo "=== Stream Analytics Jobs ==="

for job in $(az stream-analytics job list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az stream-analytics job list --query "[?name=='$job'].resourceGroup" -o tsv | head -1)
    
    STATUS=$(az stream-analytics job show -n "$job" -g "$RG" \
        --query "jobState" -o tsv 2>/dev/null)
    
    echo "Job: $job ($RG) - Status: $STATUS"
    
    # Inputs
    az stream-analytics input list --job-name "$job" -g "$RG" \
        --query "[].{name:name, type:properties.type}" -o table 2>/dev/null
    
    # Outputs
    az stream-analytics output list --job-name "$job" -g "$RG" \
        --query "[].{name:name, datasource:properties.datasource.type}" -o table 2>/dev/null
done`,
				},
			},
		},
	})
}
