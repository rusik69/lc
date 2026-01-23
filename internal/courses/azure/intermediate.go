package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          410,
			Title:       "Azure Functions",
			Description: "Learn Azure Functions: serverless computing, triggers, bindings, and event-driven architecture.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Functions Fundamentals",
					Content: `Azure Functions is a serverless compute service for running event-driven code.

**Function Types:**
- **HTTP Trigger**: Respond to HTTP requests
- **Timer Trigger**: Scheduled execution
- **Blob Trigger**: React to blob storage events
- **Queue Trigger**: Process queue messages
- **Event Hub Trigger**: Process event streams
- **Cosmos DB Trigger**: React to database changes

**Hosting Plans:**
- **Consumption Plan**: Pay per execution, auto-scaling
- **Premium Plan**: Pre-warmed instances, VNet integration
- **Dedicated Plan**: App Service plan hosting

**Supported Languages:**
- C#, JavaScript/TypeScript, Python, Java, PowerShell
- Custom handlers for other languages

**Bindings:**
- **Input Bindings**: Read data from services
- **Output Bindings**: Write data to services
- **Trigger Bindings**: Invoke function execution

**Best Practices:**
- Keep functions small and focused
- Use async/await for I/O operations
- Implement retry logic for transient failures
- Use Application Insights for monitoring
- Optimize cold start times`,
					CodeExamples: `# Create function app
az functionapp create \\
    --resource-group myResourceGroup \\
    --consumption-plan-location eastus \\
    --runtime node \\
    --runtime-version 14 \\
    --functions-version 3 \\
    --name myFunctionApp \\
    --storage-account mystorageaccount

# Create HTTP trigger function
az functionapp function create \\
    --resource-group myResourceGroup \\
    --name myFunctionApp \\
    --function-name HttpTrigger1 \\
    --template "HTTP trigger"

# Deploy function code
func azure functionapp publish myFunctionApp

# Example function (JavaScript)
module.exports = async function (context, req) {
    context.log('HTTP trigger function processed a request.');
    const name = (req.query.name || (req.body && req.body.name));
    const responseMessage = name
        ? "Hello, " + name + ". This HTTP triggered function executed successfully."
        : "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.";
    context.res = {
        status: 200,
        body: responseMessage
    };
};`,
				},
				{
					Title: "Durable Functions",
					Content: `Durable Functions is an extension of Azure Functions for building stateful workflows.

**Durable Functions Patterns:**
- **Function Chaining**: Execute functions in sequence
- **Fan-out/Fan-in**: Parallel execution with aggregation
- **Async HTTP APIs**: Long-running operations with status endpoints
- **Monitoring**: Recurring patterns for monitoring
- **Human Interaction**: Wait for external events
- **Orchestrator Functions**: Coordinate other functions
- **Activity Functions**: Perform actual work
- **Entity Functions**: Stateful entities for small pieces of state

**Orchestration:**
- **Orchestrator Function**: Defines workflow logic
- **Activity Functions**: Called by orchestrator
- **Deterministic**: Must be deterministic (no random, DateTime.Now, etc.)
- **Replay**: Functions replayed for state reconstruction

**Durable Client:**
- **Start Orchestration**: Start new orchestration instance
- **Query Status**: Check orchestration status
- **Terminate**: Stop running orchestration
- **Raise Event**: Send event to orchestration

**Best Practices:**
- Keep orchestrator functions deterministic
- Use activity functions for I/O operations
- Handle errors and retries properly
- Use sub-orchestrations for complex workflows
- Monitor orchestration instances
- Set appropriate timeout values
- Use durable timers instead of Thread.Sleep`,
					CodeExamples: `# Install Durable Functions extension
# For Node.js: npm install durable-functions
# For C#: Already included in Azure Functions runtime

# Example orchestrator function (C#)
[FunctionName("OrchestratorFunction")]
public static async Task<string> RunOrchestrator(
    [OrchestrationTrigger] IDurableOrchestrationContext context)
{
    var input = context.GetInput<string>();
    
    // Call activity function
    var result1 = await context.CallActivityAsync<string>("ActivityFunction1", input);
    var result2 = await context.CallActivityAsync<string>("ActivityFunction2", result1);
    
    return result2;
}

# Example activity function (C#)
[FunctionName("ActivityFunction1")]
public static string RunActivity([ActivityTrigger] string input, ILogger log)
{
    log.LogInformation($"Processing {input}");
    return $"Processed: {input}";
}

# Example client function (C#)
[FunctionName("HttpStart")]
public static async Task<HttpResponseMessage> HttpStart(
    [HttpTrigger(AuthorizationLevel.Function, "post")] HttpRequestMessage req,
    [DurableClient] IDurableOrchestrationClient starter,
    ILogger log)
{
    string instanceId = await starter.StartNewAsync("OrchestratorFunction", null);
    
    log.LogInformation($"Started orchestration with ID = '{instanceId}'.");
    
    return starter.CreateCheckStatusResponse(req, instanceId);
}

# Query orchestration status
az rest \\
    --method GET \\
    --uri "https://myFunctionApp.azurewebsites.net/runtime/webhooks/durabletask/instances/{instanceId}"`,
				},
				{
					Title: "Function Monitoring",
					Content: `Monitoring Azure Functions is essential for understanding performance and troubleshooting issues.

**Application Insights Integration:**
- **Automatic Integration**: Built-in Application Insights support
- **Metrics**: Execution count, duration, success rate
- **Logs**: Function execution logs and traces
- **Dependencies**: Track calls to other services
- **Exceptions**: Exception tracking and stack traces

**Key Metrics:**
- **Execution Count**: Number of function executions
- **Success Rate**: Percentage of successful executions
- **Average Duration**: Average execution time
- **Error Rate**: Percentage of failed executions
- **Throttles**: Number of throttled executions

**Logging:**
- **ILogger**: Use ILogger for structured logging
- **Log Levels**: Trace, Debug, Information, Warning, Error, Critical
- **Custom Properties**: Add custom properties to logs
- **Correlation**: Automatic correlation IDs

**Alerts:**
- **Metric Alerts**: Alert on execution count, duration, errors
- **Log Alerts**: Alert on specific log patterns
- **Action Groups**: Email, SMS, webhook notifications
- **Smart Detection**: Automatic anomaly detection

**Performance Monitoring:**
- **Cold Start**: Time to initialize function
- **Warm Start**: Time for already initialized function
- **Dependency Calls**: Time spent calling external services
- **Memory Usage**: Function memory consumption

**Best Practices:**
- Enable Application Insights for all function apps
- Use structured logging with ILogger
- Set up alerts for errors and performance issues
- Monitor cold start times
- Track dependency performance
- Review logs regularly
- Use Application Insights dashboards
- Set up automated alerts`,
					CodeExamples: `# Enable Application Insights
az monitor app-insights component create \\
    --app myFunctionAppInsights \\
    --location eastus \\
    --resource-group myResourceGroup \\
    --application-type web

# Link function app to Application Insights
az functionapp config appsettings set \\
    --resource-group myResourceGroup \\
    --name myFunctionApp \\
    --settings APPINSIGHTS_INSTRUMENTATIONKEY=<instrumentation-key>

# View function metrics
az monitor metrics list \\
    --resource /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Web/sites/myFunctionApp \\
    --metric "FunctionExecutionCount" \\
    --start-time 2024-01-01T00:00:00Z \\
    --end-time 2024-01-31T23:59:59Z

# Example logging in function (C#)
public static async Task<IActionResult> Run(
    [HttpTrigger(AuthorizationLevel.Function, "get", "post")] HttpRequest req,
    ILogger log)
{
    log.LogInformation("Function started");
    log.LogWarning("This is a warning");
    log.LogError("This is an error");
    
    return new OkObjectResult("Function executed");
}

# Create alert rule
az monitor metrics alert create \\
    --name "HighErrorRate" \\
    --resource-group myResourceGroup \\
    --scopes /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Web/sites/myFunctionApp \\
    --condition "avg FunctionExecutionFailures > 10" \\
    --window-size 5m \\
    --evaluation-frequency 1m`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          411,
			Title:       "Azure Cosmos DB",
			Description: "Learn Cosmos DB: globally distributed NoSQL database, consistency levels, and multi-model support.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Cosmos DB Fundamentals",
					Content: `Azure Cosmos DB is a globally distributed, multi-model NoSQL database service.

**API Models:**
- **SQL API**: Document database (default)
- **MongoDB API**: MongoDB compatible
- **Cassandra API**: Apache Cassandra compatible
- **Gremlin API**: Graph database
- **Table API**: Azure Table Storage compatible

**Consistency Levels:**
- **Strong**: Highest consistency, lowest performance
- **Bounded Staleness**: Configurable staleness window
- **Session**: Per-session consistency
- **Consistent Prefix**: Ordering guarantees
- **Eventual**: Highest performance, eventual consistency

**Global Distribution:**
- Multi-region replication
- Automatic failover
- Low latency worldwide
- Multi-master writes

**Throughput:**
- **Request Units (RU)**: Measure of throughput
- **Autoscale**: Automatic scaling based on usage
- **Manual**: Fixed throughput allocation

**Best Practices:**
- Choose appropriate consistency level
- Use partition keys effectively
- Enable autoscale for variable workloads
- Use change feed for event processing
- Monitor RU consumption`,
					CodeExamples: `# Create Cosmos DB account
az cosmosdb create \\
    --name mycosmosdb \\
    --resource-group myResourceGroup \\
    --default-consistency-level Session \\
    --locations regionName=eastus failoverPriority=0 \\
    --locations regionName=westus failoverPriority=1

# Create SQL database
az cosmosdb sql database create \\
    --account-name mycosmosdb \\
    --resource-group myResourceGroup \\
    --name myDatabase

# Create container
az cosmosdb sql container create \\
    --account-name mycosmosdb \\
    --resource-group myResourceGroup \\
    --database-name myDatabase \\
    --name myContainer \\
    --partition-key-path "/id" \\
    --throughput 400

# Enable autoscale
az cosmosdb sql container throughput update \\
    --account-name mycosmosdb \\
    --resource-group myResourceGroup \\
    --database-name myDatabase \\
    --name myContainer \\
    --max-throughput 4000`,
				},
				{
					Title: "Partitioning Strategy",
					Content: `Effective partitioning is crucial for Cosmos DB performance and scalability.

**Partition Key Selection:**
- **High Cardinality**: Many distinct values
- **Even Distribution**: Balanced data distribution
- **Query Patterns**: Align with common queries
- **Avoid Hot Partitions**: Prevent single partition overload

**Partition Key Best Practices:**
- Choose properties with many distinct values
- Avoid frequently updated properties
- Consider query patterns
- Use composite keys if needed
- Test partition distribution

**Logical Partitions:**
- **Partition Key**: Defines logical partition
- **Physical Partitions**: Managed by Cosmos DB
- **Partition Size**: Up to 20GB per logical partition
- **Throughput**: Distributed across partitions

**Partitioning Considerations:**
- **Storage**: 20GB limit per logical partition
- **Throughput**: Minimum 400 RU/s per physical partition
- **Scaling**: Add partitions by increasing throughput
- **Distribution**: Cosmos DB distributes data automatically

**Best Practices:**
- Choose partition key with high cardinality
- Monitor partition key distribution
- Avoid hot partitions
- Use composite partition keys if needed
- Test with realistic data volumes
- Review partition metrics regularly`,
					CodeExamples: `# Create container with partition key
az cosmosdb sql container create \\
    --account-name mycosmosdb \\
    --resource-group myResourceGroup \\
    --database-name myDatabase \\
    --name myContainer \\
    --partition-key-path "/userId" \\
    --throughput 400

# Create container with composite partition key (requires API)
# Note: Composite partition keys require using SDK or Portal

# Query partition key distribution
# Use Cosmos DB Metrics in Azure Portal to view partition distribution

# Example good partition key: userId (high cardinality)
# Example bad partition key: status (low cardinality, e.g., active/inactive)

# Using SDK (Python)
from azure.cosmos import CosmosClient

client = CosmosClient(url, key)
database = client.get_database_client("myDatabase")
container = database.get_container_client("myContainer")

# Query with partition key for better performance
items = container.query_items(
    query="SELECT * FROM c WHERE c.userId = @userId",
    parameters=[{"name": "@userId", "value": "user123"}],
    partition_key="user123"
)`,
				},
				{
					Title: "Change Feed",
					Content: `Cosmos DB Change Feed provides a sorted list of documents that changed in order of modification.

**Change Feed Features:**
- **Real-time Processing**: Process changes as they occur
- **Incremental Processing**: Only process changed documents
- **Ordered**: Changes in order of modification time
- **Scalable**: Process changes in parallel

**Use Cases:**
- **Event Sourcing**: Track all changes to data
- **Real-time Analytics**: Process changes for analytics
- **Data Synchronization**: Sync data to other systems
- **Audit Logging**: Track all modifications
- **Trigger Workflows**: Trigger downstream processes

**Change Feed Processor:**
- **Lease Container**: Tracks processing progress
- **Multiple Processors**: Scale out processing
- **Automatic Load Balancing**: Distribute partitions
- **Checkpointing**: Resume from last processed item

**Change Feed Modes:**
- **Latest Version**: Only latest version of document
- **All Versions**: All versions and deletes
- **Time-based**: Changes since specific time

**Best Practices:**
- Use Change Feed Processor for production
- Implement proper error handling
- Monitor processing lag
- Use separate lease container
- Implement idempotent processing
- Handle deletes appropriately
- Scale processors as needed`,
					CodeExamples: `# Enable change feed (enabled by default)
# Change feed is automatically enabled for all containers

# Using Change Feed Processor (C#)
using Microsoft.Azure.Cosmos;

var leaseContainer = database.GetContainer("leases");
var monitoredContainer = database.GetContainer("myContainer");

var changeFeedProcessor = monitoredContainer
    .GetChangeFeedProcessorBuilder("myProcessor", HandleChangesAsync)
    .WithInstanceName("instance1")
    .WithLeaseContainer(leaseContainer)
    .Build();

await changeFeedProcessor.StartAsync();

async Task HandleChangesAsync(
    ChangeFeedProcessorContext context,
    IReadOnlyCollection<MyDocument> changes,
    CancellationToken cancellationToken)
{
    foreach (var document in changes)
    {
        // Process document change
        Console.WriteLine($"Processing change for document {document.Id}");
    }
}

# Using Azure Functions Change Feed Trigger
[FunctionName("CosmosDBChangeFeed")]
public static void Run(
    [CosmosDBTrigger(
        databaseName: "myDatabase",
        collectionName: "myContainer",
        ConnectionStringSetting = "CosmosDBConnection",
        LeaseCollectionName = "leases")] IReadOnlyList<Document> documents,
    ILogger log)
{
    foreach (var document in documents)
    {
        log.LogInformation($"Processing document {document.Id}");
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          412,
			Title:       "Azure Application Gateway",
			Description: "Learn Application Gateway: web traffic load balancer, WAF, SSL termination, and URL routing.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Application Gateway Fundamentals",
					Content: `Azure Application Gateway is a web traffic load balancer with advanced features.

**Features:**
- **Layer 7 Load Balancing**: HTTP/HTTPS traffic routing
- **Web Application Firewall (WAF)**: Protection against common attacks
- **SSL Termination**: Offload SSL processing
- **URL-based Routing**: Route based on URL path
- **Multi-site Hosting**: Host multiple sites on one gateway
- **Redirection**: HTTP to HTTPS redirection

**SKUs:**
- **Standard**: Basic load balancing
- **WAF**: Includes Web Application Firewall
- **Standard_v2**: Improved performance and features
- **WAF_v2**: WAF with v2 features

**Routing Rules:**
- **Basic**: Route all traffic to backend pool
- **Path-based**: Route based on URL path
- **Multi-site**: Route based on host header

**Health Probes:**
- HTTP/HTTPS health checks
- Custom probe paths
- Probe interval and timeout configuration

**Best Practices:**
- Use WAF for production workloads
- Enable HTTPS redirection
- Use path-based routing for microservices
- Configure appropriate health probes
- Monitor gateway metrics`,
					CodeExamples: `# Create public IP
az network public-ip create \\
    --resource-group myResourceGroup \\
    --name myAGPublicIPAddress \\
    --allocation-method Static \\
    --sku Standard

# Create application gateway
az network application-gateway create \\
    --resource-group myResourceGroup \\
    --name myAppGateway \\
    --location eastus \\
    --sku WAF_v2 \\
    --capacity 2 \\
    --public-ip-address myAGPublicIPAddress \\
    --vnet-name myVNet \\
    --subnet myAGSubnet \\
    --servers 10.0.1.4 10.0.1.5

# Create WAF policy
az network application-gateway waf-policy create \\
    --resource-group myResourceGroup \\
    --name myWAFPolicy \\
    --location eastus

# Enable WAF
az network application-gateway waf-policy set \\
    --resource-group myResourceGroup \\
    --policy-name myWAFPolicy \\
    --state Enabled \\
    --mode Prevention`,
				},
				{
					Title: "WAF Rules",
					Content: `Web Application Firewall (WAF) rules protect applications from common web vulnerabilities.

**WAF Rule Sets:**
- **OWASP 3.2**: Open Web Application Security Project rules
- **OWASP 3.1**: Previous version rules
- **Microsoft Bot Manager**: Bot protection rules
- **Custom Rules**: Organization-specific rules

**WAF Modes:**
- **Detection**: Log attacks but don't block
- **Prevention**: Block attacks and log

**Rule Actions:**
- **Allow**: Allow request
- **Block**: Block request
- **Log**: Log request
- **Redirect**: Redirect to URL

**Custom Rules:**
- **Match Conditions**: IP address, request headers, request body
- **Actions**: Allow, Block, Log, Redirect
- **Priority**: Rule evaluation order
- **Rate Limiting**: Limit requests per IP

**Best Practices:**
- Start with Detection mode
- Review WAF logs regularly
- Create custom rules for application-specific needs
- Use rate limiting for DDoS protection
- Test rules in Detection mode first
- Monitor false positives
- Keep rule sets updated`,
					CodeExamples: `# Create custom WAF rule
az network application-gateway waf-policy custom-rule create \\
    --resource-group myResourceGroup \\
    --policy-name myWAFPolicy \\
    --name BlockIPRange \\
    --priority 100 \\
    --rule-type MatchRule \\
    --action Block \\
    --match-conditions "[{matchVariables:[{variableName:'RemoteAddr'}],operator:'IPMatch',matchValues:['1.2.3.0/24']}]"

# Create rate limit rule
az network application-gateway waf-policy custom-rule create \\
    --resource-group myResourceGroup \\
    --policy-name myWAFPolicy \\
    --name RateLimit \\
    --priority 200 \\
    --rule-type RateLimitRule \\
    --rate-limit-threshold 100 \\
    --action Block`,
				},
				{
					Title: "SSL Termination",
					Content: `Application Gateway can terminate SSL/TLS connections, offloading encryption from backend servers.

**SSL Termination Benefits:**
- **Performance**: Offload CPU-intensive encryption
- **Certificate Management**: Centralized certificate management
- **Backend Simplification**: Backends don't need SSL certificates
- **Cost**: Reduce backend server resources

**SSL Certificates:**
- **Application Gateway Certificates**: Managed by Application Gateway
- **Key Vault Integration**: Store certificates in Key Vault
- **Auto-renewal**: Automatic certificate renewal
- **Multiple Certificates**: Support for multiple domains

**SSL Policies:**
- **Policy Type**: Predefined or custom
- **Min Protocol Version**: Minimum TLS version
- **Cipher Suites**: Allowed cipher suites
- **Client Certificate**: Require client certificates

**End-to-End SSL:**
- **Backend SSL**: Encrypt traffic to backend
- **Health Probes**: Use HTTPS for health checks
- **Certificate Validation**: Validate backend certificates

**Best Practices:**
- Use Key Vault for certificate storage
- Enable SSL termination for performance
- Use strong SSL policies
- Monitor certificate expiration
- Use end-to-end SSL for sensitive data
- Enable HSTS headers
- Use latest TLS versions`,
					CodeExamples: `# Upload SSL certificate
az network application-gateway ssl-cert create \\
    --resource-group myResourceGroup \\
    --gateway-name myAppGateway \\
    --name mySSLCert \\
    --cert-file certificate.pfx \\
    --cert-password "CertPassword"

# Configure SSL policy
az network application-gateway ssl-policy set \\
    --resource-group myResourceGroup \\
    --gateway-name myAppGateway \\
    --policy-type Predefined \\
    --policy-name AppGwSslPolicy20150501

# Use Key Vault certificate
az network application-gateway ssl-cert create \\
    --resource-group myResourceGroup \\
    --gateway-name myAppGateway \\
    --name myKeyVaultCert \\
    --key-vault-secret-id "https://mykeyvault.vault.azure.net/secrets/mysecret"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          413,
			Title:       "Azure Container Instances",
			Description: "Learn Container Instances: serverless containers, quick deployment, and container orchestration basics.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Container Instances Fundamentals",
					Content: `Azure Container Instances (ACI) provides the fastest way to run containers in Azure.

**Features:**
- **Serverless Containers**: No VM management
- **Quick Start**: Containers start in seconds
- **Per-second Billing**: Pay only for running time
- **Hypervisor-level Isolation**: Secure container isolation
- **Custom Sizes**: Specify CPU and memory

**Use Cases:**
- Quick deployments
- Batch processing
- CI/CD pipelines
- Development and testing
- Microservices

**Container Groups:**
- Group multiple containers together
- Shared networking and storage
- Sidecar pattern support
- Restart policies

**Networking:**
- Public IP addresses
- Private IP addresses
- Virtual network integration
- Custom DNS servers

**Best Practices:**
- Use for short-lived workloads
- Leverage container groups for related containers
- Use managed identities for Azure resource access
- Monitor container logs
- Optimize container images`,
					CodeExamples: `# Create container instance
az container create \\
    --resource-group myResourceGroup \\
    --name mycontainer \\
    --image mcr.microsoft.com/azuredocs/aci-helloworld \\
    --dns-name-label mycontainer \\
    --ports 80 \\
    --location eastus

# Create container with environment variables
az container create \\
    --resource-group myResourceGroup \\
    --name mycontainer \\
    --image nginx \\
    --environment-variables 'KEY1=VALUE1' 'KEY2=VALUE2' \\
    --ports 80

# Create container group
az container create \\
    --resource-group myResourceGroup \\
    --name mycontainergroup \\
    --image mcr.microsoft.com/azuredocs/aci-tutorial-sidecar \\
    --ip-address Public \\
    --ports 80

# View container logs
az container logs \\
    --resource-group myResourceGroup \\
    --name mycontainer`,
				},
				{
					Title: "Container Groups",
					Content: `Container groups allow multiple containers to share resources and networking.

**Container Group Features:**
- **Shared Networking**: Containers share IP and port namespace
- **Shared Storage**: Mount Azure File shares
- **Sidecar Pattern**: Deploy supporting containers
- **Restart Policy**: Control container restart behavior

**Restart Policies:**
- **Always**: Always restart container
- **Never**: Never restart container
- **OnFailure**: Restart only on failure

**Networking:**
- **Public IP**: Internet-accessible containers
- **Private IP**: Internal-only containers
- **DNS Name**: Custom DNS name label
- **Port Mapping**: Map container ports

**Best Practices:**
- Use container groups for related containers
- Implement proper restart policies
- Use shared storage for data sharing
- Monitor container group resources
- Use sidecar pattern for logging/monitoring`,
					CodeExamples: `# Create container group with multiple containers
az container create \\
    --resource-group myResourceGroup \\
    --name myContainerGroup \\
    --image nginx \\
    --cpu 1 \\
    --memory 1.5 \\
    --ip-address Public \\
    --ports 80 \\
    --restart-policy Always`,
				},
				{
					Title: "Networking",
					Content: `Container Instances support various networking options for connectivity.

**Networking Options:**
- **Public IP**: Internet-accessible
- **Private IP**: VNet integration
- **DNS**: Custom DNS names
- **Port Mapping**: Container port mapping

**VNet Integration:**
- **Delegated Subnet**: Deploy to VNet subnet
- **Private IP**: Private connectivity
- **Service Endpoints**: Access Azure services privately
- **Network Security Groups**: Apply NSG rules

**Best Practices:**
- Use VNet integration for secure connectivity
- Use private IPs for internal services
- Configure NSGs appropriately
- Use service endpoints for Azure services`,
					CodeExamples: `# Create container with VNet
az container create \\
    --resource-group myResourceGroup \\
    --name myContainer \\
    --image nginx \\
    --vnet myVNet \\
    --subnet mySubnet \\
    --ip-address Private`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          414,
			Title:       "Azure Database for PostgreSQL",
			Description: "Learn Azure Database for PostgreSQL: managed PostgreSQL, high availability, and scaling.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "PostgreSQL Database Fundamentals",
					Content: `Azure Database for PostgreSQL is a fully managed PostgreSQL database service.

**Deployment Options:**
- **Single Server**: Traditional deployment model
- **Flexible Server**: More control and flexibility
- **Hyperscale (Citus)**: Distributed PostgreSQL for scale

**Service Tiers:**
- **Basic**: Development and testing
- **General Purpose**: Balanced compute and storage
- **Memory Optimized**: High-performance workloads

**Features:**
- Automated backups (point-in-time restore)
- High availability options
- Read replicas for read scaling
- Advanced threat protection
- PostgreSQL extensions support

**Scaling:**
- Scale compute up or down
- Scale storage independently
- Read replicas for read scaling
- Auto-grow storage

**Best Practices:**
- Use Flexible Server for production
- Enable high availability for critical workloads
- Use read replicas for read-heavy applications
- Monitor performance metrics
- Regular backup testing`,
					CodeExamples: `# Create PostgreSQL server
az postgres server create \\
    --resource-group myResourceGroup \\
    --name mypostgresserver \\
    --location eastus \\
    --admin-user myadmin \\
    --admin-password SecurePassword123! \\
    --sku-name GP_Gen5_2 \\
    --version 11

# Create database
az postgres db create \\
    --resource-group myResourceGroup \\
    --server-name mypostgresserver \\
    --name mydatabase

# Create read replica
az postgres server replica create \\
    --resource-group myResourceGroup \\
    --name mypostgresserver-replica \\
    --source-server mypostgresserver \\
    --location westus

# Configure firewall rule
az postgres server firewall-rule create \\
    --resource-group myResourceGroup \\
    --server mypostgresserver \\
    --name AllowMyIP \\
    --start-ip-address 1.2.3.4 \\
    --end-ip-address 1.2.3.4`,
				},
				{
					Title: "High Availability",
					Content: `Azure Database for PostgreSQL provides high availability options for production workloads.

**High Availability Features:**
- **Zone Redundant**: Deploy across availability zones
- **Automatic Failover**: Automatic failover to standby
- **99.99% SLA**: High availability guarantee
- **Data Protection**: Zero data loss on failover

**Flexible Server HA:**
- **Standby Server**: Automatic standby in different zone
- **Synchronous Replication**: Data replicated synchronously
- **Fast Failover**: Sub-60 second failover time
- **Application Transparency**: No application changes needed

**Best Practices:**
- Enable HA for production databases
- Monitor replication lag
- Test failover procedures
- Use connection retry logic
- Monitor HA status`,
					CodeExamples: `# Enable high availability
az postgres flexible-server update \\
    --resource-group myResourceGroup \\
    --name myserver \\
    --high-availability Enabled`,
				},
				{
					Title: "Read Replicas",
					Content: `Read replicas provide read scaling and geographic distribution for PostgreSQL.

**Read Replica Benefits:**
- **Read Scaling**: Offload read queries
- **Geographic Distribution**: Replicate to different regions
- **Disaster Recovery**: Use as backup
- **Reporting**: Isolated reporting workloads

**Replication Types:**
- **Synchronous**: Zero data loss, higher latency
- **Asynchronous**: Lower latency, potential data loss

**Best Practices:**
- Use replicas for read-heavy workloads
- Monitor replication lag
- Use for reporting and analytics
- Test failover to replica`,
					CodeExamples: `# Create read replica
az postgres flexible-server replica create \\
    --resource-group myResourceGroup \\
    --replica-name myreplica \\
    --source-server myserver \\
    --location westus`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          415,
			Title:       "Azure Service Bus",
			Description: "Learn Service Bus: messaging service, queues, topics, and pub/sub patterns.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Service Bus Fundamentals",
					Content: `Azure Service Bus is a fully managed enterprise message broker.

**Messaging Patterns:**
- **Queues**: Point-to-point messaging
- **Topics**: Publish-subscribe messaging
- **Relays**: On-premises integration

**Queue Features:**
- **FIFO**: First-in-first-out ordering
- **Dead-letter Queue**: Failed message handling
- **Message Sessions**: Ordered message processing
- **Duplicate Detection**: Prevent duplicate messages

**Topic Features:**
- **Subscriptions**: Multiple consumers
- **Filters**: Message filtering per subscription
- **Actions**: Message transformation
- **Dead-letter Subscriptions**: Failed message handling

**Service Tiers:**
- **Basic**: Queues only, limited features
- **Standard**: Queues and topics, standard features
- **Premium**: Higher throughput, dedicated resources

**Best Practices:**
- Use topics for pub/sub scenarios
- Implement dead-letter queue handling
- Use message sessions for ordered processing
- Monitor queue depth and latency
- Use Premium tier for high throughput`,
					CodeExamples: `# Create namespace
az servicebus namespace create \\
    --resource-group myResourceGroup \\
    --name myServiceBusNamespace \\
    --location eastus \\
    --sku Standard

# Create queue
az servicebus queue create \\
    --resource-group myResourceGroup \\
    --namespace-name myServiceBusNamespace \\
    --name myQueue \\
    --max-size 1024 \\
    --default-message-time-to-live P1D

# Create topic
az servicebus topic create \\
    --resource-group myResourceGroup \\
    --namespace-name myServiceBusNamespace \\
    --name myTopic \\
    --max-size 1024

# Create subscription
az servicebus topic subscription create \\
    --resource-group myResourceGroup \\
    --namespace-name myServiceBusNamespace \\
    --topic-name myTopic \\
    --name mySubscription \\
    --max-delivery-count 10`,
				},
				{
					Title: "Message Sessions",
					Content: `Message sessions enable ordered message processing in Service Bus.

**Session Features:**
- **Ordered Processing**: Messages processed in order
- **Session ID**: Group related messages
- **FIFO Guarantee**: First-in-first-out within session
- **Concurrent Sessions**: Process multiple sessions in parallel

**Use Cases:**
- **Order Processing**: Process order items in sequence
- **User Sessions**: Maintain user context
- **Workflow Processing**: Sequential workflow steps

**Best Practices:**
- Use sessions for ordered processing
- Keep session IDs meaningful
- Handle session timeouts
- Process sessions concurrently`,
					CodeExamples: `# Create queue with sessions enabled
az servicebus queue create \\
    --resource-group myResourceGroup \\
    --namespace-name myServiceBusNamespace \\
    --name myQueue \\
    --requires-session true`,
				},
				{
					Title: "Dead Letter Queues",
					Content: `Dead letter queues store messages that cannot be processed successfully.

**DLQ Features:**
- **Automatic**: Created automatically
- **Failed Messages**: Messages that exceed max delivery count
- **Manual Dead Lettering**: Manually move messages
- **Inspection**: Review and fix issues

**DLQ Management:**
- **Review Messages**: Inspect failed messages
- **Fix and Resubmit**: Fix issues and resubmit
- **Purge**: Remove processed messages
- **Monitoring**: Monitor DLQ depth

**Best Practices:**
- Monitor DLQ regularly
- Set appropriate max delivery count
- Review and fix failed messages
- Implement alerting for DLQ depth
- Document resolution procedures`,
					CodeExamples: `# View dead letter messages
az servicebus queue show \\
    --resource-group myResourceGroup \\
    --namespace-name myServiceBusNamespace \\
    --name myQueue/deadletter`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          416,
			Title:       "Azure Event Hubs",
			Description: "Learn Event Hubs: big data streaming, event ingestion, and real-time analytics.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Event Hubs Fundamentals",
					Content: `Azure Event Hubs is a big data streaming platform and event ingestion service.

**Features:**
- **High Throughput**: Millions of events per second
- **Low Latency**: Sub-second latency
- **Scalability**: Auto-scaling based on demand
- **Retention**: Configurable event retention period
- **Capture**: Automatic event capture to storage

**Tiers:**
- **Basic**: Basic features, limited throughput
- **Standard**: Standard features, higher throughput
- **Dedicated**: Isolated clusters, highest throughput

**Consumer Groups:**
- Multiple independent event streams
- Each consumer group reads all events
- Enable parallel processing
- Scale consumers independently

**Partitions:**
- Events distributed across partitions
- Partition key for ordering
- Parallel processing per partition
- Configurable partition count

**Best Practices:**
- Use appropriate partition keys
- Create multiple consumer groups for different consumers
- Enable Capture for data archival
- Monitor throughput units
- Use Dedicated tier for high-scale scenarios`,
					CodeExamples: `# Create namespace
az eventhubs namespace create \\
    --resource-group myResourceGroup \\
    --name myEventHubNamespace \\
    --location eastus \\
    --sku Standard \\
    --capacity 1

# Create event hub
az eventhubs eventhub create \\
    --resource-group myResourceGroup \\
    --namespace-name myEventHubNamespace \\
    --name myEventHub \\
    --partition-count 4 \\
    --message-retention 1

# Create consumer group
az eventhubs eventhub consumer-group create \\
    --resource-group myResourceGroup \\
    --namespace-name myEventHubNamespace \\
    --eventhub-name myEventHub \\
    --name myConsumerGroup

# Enable Capture
az eventhubs eventhub update \\
    --resource-group myResourceGroup \\
    --namespace-name myEventHubNamespace \\
    --name myEventHub \\
    --enable-capture true \\
    --capture-interval 300 \\
    --capture-size-limit 314572800 \\
    --destination-name EventHubArchive.AzureBlockBlob \\
    --storage-account myStorageAccount`,
				},
				{
					Title: "Partitioning",
					Content: `Event Hubs partitions enable parallel processing and scalability.

**Partition Concepts:**
- **Partition Key**: Route events to specific partition
- **Partition Count**: Number of partitions (1-32)
- **Ordering**: Events with same key stay ordered
- **Throughput**: Each partition has throughput limit

**Partition Strategy:**
- **Round Robin**: Distribute evenly
- **Partition Key**: Route by key
- **Balanced**: Automatic balancing

**Best Practices:**
- Choose partition count based on throughput needs
- Use partition keys for ordering
- Monitor partition distribution
- Scale partitions as needed`,
					CodeExamples: `# Create event hub with partitions
az eventhubs eventhub create \\
    --resource-group myResourceGroup \\
    --namespace-name myEventHubNamespace \\
    --name myEventHub \\
    --partition-count 4`,
				},
				{
					Title: "Consumer Groups",
					Content: `Consumer groups enable multiple independent event stream readers.

**Consumer Group Features:**
- **Independent Streams**: Each group reads all events
- **Parallel Processing**: Multiple consumers per group
- **Offset Tracking**: Track reading position
- **Scaling**: Scale consumers independently

**Best Practices:**
- Use separate groups for different consumers
- Scale consumers based on throughput
- Monitor consumer lag
- Handle checkpointing properly`,
					CodeExamples: `# Create consumer group
az eventhubs eventhub consumer-group create \\
    --resource-group myResourceGroup \\
    --namespace-name myEventHubNamespace \\
    --eventhub-name myEventHub \\
    --name myConsumerGroup`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          417,
			Title:       "Azure Monitor",
			Description: "Learn Azure Monitor: metrics, logs, alerts, and application performance monitoring.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Monitor Fundamentals",
					Content: `Azure Monitor provides comprehensive monitoring for Azure resources and applications.

**Data Types:**
- **Metrics**: Numerical values over time
- **Logs**: Text-based data from various sources
- **Traces**: Distributed tracing data
- **Changes**: Resource configuration changes

**Components:**
- **Metrics Explorer**: Visualize metrics
- **Log Analytics**: Query and analyze logs
- **Application Insights**: Application performance monitoring
- **Alerts**: Notifications based on conditions
- **Dashboards**: Custom visualization

**Data Sources:**
- Azure platform metrics
- Application Insights telemetry
- Custom metrics
- Activity logs
- Resource logs
- Guest OS metrics

**Alert Rules:**
- **Metric Alerts**: Based on metric values
- **Log Alerts**: Based on log queries
- **Activity Log Alerts**: Based on resource events
- **Smart Detection**: AI-powered anomaly detection

**Best Practices:**
- Enable Application Insights for applications
- Create meaningful alert rules
- Use Log Analytics for deep analysis
- Set up action groups for notifications
- Monitor cost and data ingestion`,
					CodeExamples: `# Create Log Analytics workspace
az monitor log-analytics workspace create \\
    --resource-group myResourceGroup \\
    --workspace-name myWorkspace \\
    --location eastus

# Enable diagnostic settings
az monitor diagnostic-settings create \\
    --resource /subscriptions/<sub-id>/resourceGroups/myResourceGroup/providers/Microsoft.Storage/storageAccounts/mystorageaccount \\
    --name myDiagnosticSetting \\
    --workspace myWorkspace \\
    --logs '[{"category": "StorageRead", "enabled": true}]' \\
    --metrics '[{"category": "Transaction", "enabled": true}]'

# Create metric alert
az monitor metrics alert create \\
    --name "High CPU" \\
    --resource-group myResourceGroup \\
    --scopes /subscriptions/<sub-id>/resourceGroups/myResourceGroup/providers/Microsoft.Compute/virtualMachines/myVM \\
    --condition "avg Percentage CPU > 80" \\
    --window-size 5m \\
    --evaluation-frequency 1m`,
				},
				{
					Title: "Log Analytics Queries",
					Content: `Log Analytics provides powerful query language (KQL) for analyzing log data.

**KQL Basics:**
- **Tables**: Data sources (AppTraces, AppExceptions, etc.)
- **Operators**: Filter, project, summarize, join
- **Functions**: Aggregations, time functions, string functions
- **Visualizations**: Charts and graphs

**Common Queries:**
- **Error Analysis**: Find and analyze errors
- **Performance**: Identify slow operations
- **Usage Patterns**: Understand usage trends
- **Correlations**: Find related events

**Best Practices:**
- Use time ranges appropriately
- Optimize queries for performance
- Use functions for reusability
- Create saved queries
- Use workbooks for dashboards`,
					CodeExamples: `# Example KQL query
AppTraces
| where TimeGenerated > ago(1h)
| where SeverityLevel >= 3
| summarize count() by bin(TimeGenerated, 5m)
| render timechart

# Query with join
AppTraces
| join (AppExceptions) on OperationId
| summarize count() by ExceptionType`,
				},
				{
					Title: "Alert Rules",
					Content: `Alert rules notify you when specific conditions are met in your data.

**Alert Types:**
- **Metric Alerts**: Based on metric values
- **Log Alerts**: Based on log query results
- **Activity Log Alerts**: Based on resource events

**Alert Configuration:**
- **Condition**: What triggers the alert
- **Action Group**: Who to notify
- **Severity**: Alert severity level
- **Frequency**: How often to evaluate

**Best Practices:**
- Set appropriate thresholds
- Use action groups for notifications
- Test alert rules
- Review and tune alerts regularly
- Document alert purposes`,
					CodeExamples: `# Create metric alert
az monitor metrics alert create \\
    --name "HighErrorRate" \\
    --resource-group myResourceGroup \\
    --scopes /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Web/sites/mywebapp \\
    --condition "avg HttpServerErrors > 10" \\
    --window-size 5m \\
    --evaluation-frequency 1m`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          418,
			Title:       "Azure Resource Manager",
			Description: "Learn Resource Manager: ARM templates, infrastructure as code, and resource organization.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Resource Manager Fundamentals",
					Content: `Azure Resource Manager (ARM) is the deployment and management service for Azure.

**Key Concepts:**
- **Resource Groups**: Logical containers for resources
- **Resources**: Individual Azure services
- **Subscriptions**: Billing and access boundaries
- **Management Groups**: Organize subscriptions hierarchically

**ARM Templates:**
- **JSON Templates**: Declarative infrastructure as code
- **Template Specs**: Reusable template libraries
- **Bicep**: Simplified template language
- **Idempotent**: Safe to run multiple times

**Template Structure:**
- **Parameters**: Input values
- **Variables**: Reusable values
- **Resources**: Resources to deploy
- **Outputs**: Returned values

**Deployment Methods:**
- Azure Portal
- Azure CLI
- Azure PowerShell
- REST API
- GitHub Actions
- Azure DevOps

**Best Practices:**
- Use resource groups for logical grouping
- Tag resources for organization
- Use ARM templates for repeatable deployments
- Use Bicep for better developer experience
- Implement policy for governance`,
					CodeExamples: `# Deploy ARM template
az deployment group create \\
    --resource-group myResourceGroup \\
    --template-file template.json \\
    --parameters @parameters.json

# Example ARM template (template.json)
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "storageAccountName": {
      "type": "string"
    }
  },
  "resources": [
    {
      "type": "Microsoft.Storage/storageAccounts",
      "apiVersion": "2021-04-01",
      "name": "[parameters('storageAccountName')]",
      "location": "[resourceGroup().location]",
      "sku": {
        "name": "Standard_LRS"
      },
      "kind": "StorageV2"
    }
  ]
}

# Using Bicep
param storageAccountName string
resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: storageAccountName
  location: resourceGroup().location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
}`,
				},
				{
					Title: "Bicep Templates",
					Content: `Bicep provides a simpler syntax for creating ARM templates.

**Bicep Advantages:**
- **Readability**: More readable than JSON
- **Type Safety**: Compile-time validation
- **IntelliSense**: Better IDE support
- **Modules**: Reusable components

**Bicep Structure:**
- **Parameters**: Input values
- **Variables**: Reusable values
- **Resources**: Resource declarations
- **Outputs**: Returned values
- **Modules**: Reusable Bicep files

**Best Practices:**
- Use Bicep for new deployments
- Create modules for reusability
- Use strong typing
- Version control Bicep files`,
					CodeExamples: `# Compile Bicep
az bicep build --file main.bicep

# Deploy Bicep
az deployment group create \\
    --resource-group myResourceGroup \\
    --template-file main.bicep`,
				},
				{
					Title: "Template Deployment",
					Content: `ARM and Bicep templates can be deployed using various methods.

**Deployment Methods:**
- **Azure Portal**: Visual deployment
- **Azure CLI**: Command-line deployment
- **PowerShell**: PowerShell deployment
- **REST API**: Programmatic deployment
- **GitHub Actions**: CI/CD integration

**Deployment Modes:**
- **Incremental**: Add/update resources
- **Complete**: Replace resource group

**Best Practices:**
- Validate before deploying
- Use what-if for preview
- Test in non-production
- Use version control
- Document parameters`,
					CodeExamples: `# Validate template
az deployment group validate \\
    --resource-group myResourceGroup \\
    --template-file template.json

# What-if preview
az deployment group what-if \\
    --resource-group myResourceGroup \\
    --template-file template.json`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          419,
			Title:       "Azure Automation",
			Description: "Learn Azure Automation: runbooks, automation accounts, and infrastructure automation.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Automation Fundamentals",
					Content: `Azure Automation provides cloud-based automation and configuration services.

**Components:**
- **Runbooks**: Automated scripts (PowerShell, Python, PowerShell Workflow)
- **Automation Accounts**: Containers for automation resources
- **Modules**: PowerShell modules for runbooks
- **Variables**: Stored values for runbooks
- **Credentials**: Secure credential storage

**Runbook Types:**
- **PowerShell**: PowerShell scripts
- **PowerShell Workflow**: PowerShell workflows
- **Python 2/3**: Python scripts
- **Graphical**: Visual runbook designer

**Features:**
- **Hybrid Worker**: Run runbooks on-premises
- **Update Management**: Patch management for VMs
- **Change Tracking**: Track configuration changes
- **Inventory**: Software inventory collection
- **State Configuration**: Desired State Configuration (DSC)

**Scheduling:**
- **Schedules**: Time-based execution
- **Webhooks**: HTTP-triggered execution
- **Azure Monitor Alerts**: Alert-triggered execution

**Best Practices:**
- Use managed identities for Azure resource access
- Store secrets in Key Vault
- Use modules for reusable code
- Test runbooks in test automation accounts
- Monitor runbook execution`,
					CodeExamples: `# Create automation account
az automation account create \\
    --name myAutomationAccount \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --sku Basic

# Create runbook
az automation runbook create \\
    --automation-account-name myAutomationAccount \\
    --resource-group myResourceGroup \\
    --name MyRunbook \\
    --type PowerShell \\
    --location eastus

# Publish runbook
az automation runbook publish \\
    --automation-account-name myAutomationAccount \\
    --resource-group myResourceGroup \\
    --name MyRunbook

# Start runbook
az automation runbook start \\
    --automation-account-name myAutomationAccount \\
    --resource-group myResourceGroup \\
    --name MyRunbook

# Example PowerShell runbook
workflow MyRunbook
{
    Write-Output "Hello from Azure Automation"
    $resourceGroups = Get-AzResourceGroup
    foreach ($rg in $resourceGroups) {
        Write-Output $rg.ResourceGroupName
    }
}`,
				},
				{
					Title: "Runbook Development",
					Content: `Runbooks are automated scripts that run in Azure Automation.

**Runbook Types:**
- **PowerShell**: PowerShell scripts
- **PowerShell Workflow**: PowerShell workflows
- **Python**: Python scripts
- **Graphical**: Visual runbook designer

**Runbook Development:**
- **Test Pane**: Test runbooks before publishing
- **Versioning**: Track runbook versions
- **Parameters**: Accept input parameters
- **Output**: Return output values

**Best Practices:**
- Test runbooks thoroughly
- Use parameters for flexibility
- Handle errors properly
- Use managed identities
- Document runbook purposes`,
					CodeExamples: `# Create runbook
az automation runbook create \\
    --automation-account-name myAutomationAccount \\
    --resource-group myResourceGroup \\
    --name MyRunbook \\
    --type PowerShell \\
    --location eastus

# Publish runbook
az automation runbook publish \\
    --automation-account-name myAutomationAccount \\
    --resource-group myResourceGroup \\
    --name MyRunbook`,
				},
				{
					Title: "Hybrid Workers",
					Content: `Hybrid workers enable runbooks to run on on-premises or other cloud resources.

**Hybrid Worker Benefits:**
- **On-Premises Access**: Access on-premises resources
- **Network Isolation**: Run in isolated networks
- **Custom Software**: Access to custom software
- **Compliance**: Meet data residency requirements

**Hybrid Worker Groups:**
- **Windows Workers**: Windows-based workers
- **Linux Workers**: Linux-based workers
- **Load Balancing**: Distribute workload

**Best Practices:**
- Use hybrid workers for on-premises scenarios
- Monitor worker health
- Implement proper security
- Use for compliance requirements`,
					CodeExamples: `# Create hybrid worker group
az automation hybrid-worker-group create \\
    --automation-account-name myAutomationAccount \\
    --resource-group myResourceGroup \\
    --name myHybridGroup`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
