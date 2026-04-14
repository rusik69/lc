package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1210,
			Title:       "Azure Functions",
			Description: "Learn Azure Functions: serverless computing, triggers, bindings, and event-driven architecture.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Functions Fundamentals",
					Content: `Azure Functions is a serverless compute service that lets you run event-driven code without having to manage infrastructure. Think of it as the cloud equivalent of a motion-sensor light: the code only runs when something triggers it, and you only pay for the time it is actually executing. This model eliminates the need to provision, patch, or scale virtual machines, making it ideal for workloads that are intermittent, unpredictable, or bursty in nature.

**1. Function Types — Choosing the Right Trigger**

Every Azure Function starts with a trigger, which is the event that causes the function to execute. Selecting the correct trigger type is the first architectural decision you will make.

An **HTTP Trigger** turns your function into a lightweight REST endpoint that responds to HTTP requests — perfect for building APIs, webhooks, or form-processing backends. A **Timer Trigger** fires on a cron-like schedule, making it the go-to choice for periodic batch jobs such as nightly report generation or cache warming. **Blob Triggers** react automatically when a new file lands in Azure Blob Storage, which is incredibly useful for image processing pipelines or ETL workflows. **Queue Triggers** dequeue messages from Azure Storage Queues or Service Bus queues, enabling reliable asynchronous processing. **Event Hub Triggers** are designed for high-throughput streaming scenarios, such as ingesting telemetry from millions of IoT devices. Finally, **Cosmos DB Triggers** fire whenever a document is created or updated in a Cosmos DB container, enabling real-time data synchronization and change-driven workflows.

**2. Hosting Plans — Matching Cost to Workload**

Azure Functions offers three hosting plans, each with different trade-offs between cost, performance, and control. The **Consumption Plan** is the purest serverless option: Azure automatically allocates compute resources, scales out to meet demand, and bills you only for the number of executions and the resources consumed during each execution. The downside is cold starts — if a function has not been invoked recently, the first call may take a few extra seconds while the runtime initializes. The **Premium Plan** solves the cold-start problem by keeping pre-warmed instances ready. It also supports VNet integration, larger instance sizes, and longer execution durations, making it suitable for enterprise workloads that need both serverless elasticity and network isolation. The **Dedicated (App Service) Plan** runs functions on traditional App Service infrastructure, which is beneficial when you already have underutilized App Service capacity or need fine-grained control over the underlying VM size and scaling rules.

**3. Supported Languages**

Azure Functions supports a broad set of languages, including C#, JavaScript, TypeScript, Python, Java, and PowerShell. If your language of choice is not in the list, you can use Custom Handlers to delegate execution to any process that speaks HTTP, effectively opening the door to languages like Go, Rust, or Ruby.

**4. Bindings — Declarative Data Access**

One of the most powerful features of Azure Functions is its binding system. Bindings provide a declarative way to connect your function to external data sources and services without writing boilerplate integration code. **Input Bindings** automatically read data from a service (such as reading a document from Cosmos DB) and pass it to your function as a parameter. **Output Bindings** write data to a service (such as adding a message to a queue) when your function completes. **Trigger Bindings** are the special bindings that actually invoke the function. By combining triggers with input and output bindings, you can build sophisticated integrations with just a few lines of configuration.

**5. Best Practices**

Keep each function small and focused on a single responsibility — this simplifies testing, debugging, and scaling. Always use async/await patterns for I/O-bound operations so you do not block threads unnecessarily. Implement retry logic with exponential backoff for transient failures, especially when calling external services. Enable Application Insights from day one so you have full visibility into execution counts, durations, failure rates, and dependency calls. Finally, pay close attention to cold-start optimization: keep your deployment package small, minimize dependencies, and consider the Premium Plan if cold starts are unacceptable for your use case.`,
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
					Content: `Durable Functions is an extension of Azure Functions that brings stateful workflow orchestration to the serverless world. Standard Azure Functions are inherently stateless — each invocation starts with a blank slate. Durable Functions changes this by introducing an orchestration framework that automatically manages state, checkpoints, and restarts behind the scenes. Think of it as a choreographer that tells individual dancers (activity functions) when to perform, waits for them to finish, and remembers exactly where the performance left off if anything goes wrong.

**1. Durable Functions Patterns**

Durable Functions supports several well-known workflow patterns out of the box. **Function Chaining** is the simplest: functions execute in sequence, with the output of one becoming the input of the next — much like an assembly line. **Fan-out/Fan-in** enables parallel execution: the orchestrator spawns multiple activity functions simultaneously and waits for all of them to complete before aggregating their results, which is ideal for batch-processing scenarios such as resizing hundreds of images concurrently. The **Async HTTP API** pattern is perfect for long-running operations: a client kicks off the orchestration, receives a status-check URL immediately, and can poll for completion — eliminating the need for long-lived HTTP connections. The **Monitoring** pattern implements a recurring polling loop with configurable intervals, useful for watching an external system until a condition is met. The **Human Interaction** pattern pauses the workflow to wait for an external event (like a manager's approval email), resuming only when that event arrives or a timeout elapses.

The building blocks of these patterns are three function types. **Orchestrator Functions** define the workflow logic — they coordinate the sequence, parallelism, and error handling. **Activity Functions** are where the actual work happens: calling APIs, querying databases, or transforming data. **Entity Functions** maintain small pieces of durable state and expose operations to read or update that state, which is useful for counters, aggregators, or lightweight actor-like patterns.

**2. Orchestration Internals**

Under the hood, orchestrator functions use an event-sourcing replay mechanism. The Durable Task Framework records every scheduling decision the orchestrator makes. If the function is evicted from memory (for example, during scaling or after a checkpoint), the framework replays the history to reconstruct the orchestrator's state up to the last completed activity. This means orchestrator code must be **deterministic**: it must not use random numbers, current date/time, GUIDs, or any non-deterministic API. Instead, use the context object's methods (such as context.CurrentUtcDateTime) to get deterministic values.

**3. Durable Client — Controlling Orchestrations**

A Durable Client is the entry point for managing orchestration instances from outside the workflow. You use it to **start** a new orchestration, **query** the current status (running, completed, failed), **terminate** a running instance, or **raise an event** that an orchestrator is waiting for. This makes it easy to build HTTP APIs or timer-based functions that control long-running business processes.

**4. Best Practices**

Always keep orchestrator functions deterministic — move all I/O, randomness, and side effects into activity functions. Handle errors with try-catch blocks in the orchestrator and configure retry policies on activity calls for transient failures. Break complex workflows into sub-orchestrations so each piece remains manageable and testable. Monitor orchestration instances through the built-in status APIs or the Durable Functions Monitor extension. Set appropriate timeout values on activities and external events to prevent orchestrations from hanging indefinitely. Use durable timers (context.CreateTimer) instead of Thread.Sleep, because durable timers survive process restarts while Thread.Sleep does not.`,
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
					Content: `Monitoring Azure Functions is not just a nice-to-have — it is the primary way you gain visibility into a system where there are no servers to SSH into and no persistent processes to inspect. Because serverless functions are ephemeral by nature, comprehensive monitoring and logging become your eyes and ears in production.

**1. Application Insights Integration**

Azure Functions has first-class, built-in integration with Application Insights, Microsoft's application performance management (APM) service. When you link a Function App to an Application Insights resource, telemetry is collected automatically with zero code changes. This telemetry includes **metrics** such as execution count, average duration, and success rate; **logs** containing the output of every function invocation; **dependency tracking** that records calls your function makes to databases, HTTP endpoints, and other Azure services; and **exception tracking** that captures stack traces whenever an unhandled error occurs. The result is a holistic view of your function's behavior that lets you spot issues before users do.

**2. Key Metrics to Watch**

The most important metrics for Azure Functions fall into five categories. **Execution Count** tells you how many times your function ran — useful for capacity planning and detecting unexpected spikes. **Success Rate** shows the percentage of invocations that completed without error — a sudden drop is an early warning sign. **Average Duration** reveals performance trends and helps you identify slow functions that might benefit from optimization. **Error Rate** is the inverse of success rate and is the metric you should alert on most aggressively. **Throttle Count** indicates how many executions were throttled due to resource limits, which is a signal that you may need to move to a higher hosting plan.

**3. Structured Logging**

Effective logging goes beyond printing strings. In Azure Functions, you should use the **ILogger** interface (C#) or the equivalent logging module in your language to produce structured log entries. Structured logs include key-value pairs that can be queried, filtered, and aggregated in Log Analytics. Use the standard **Log Levels** — Trace for verbose debugging, Debug for developer-focused detail, Information for normal operations, Warning for recoverable problems, Error for failures that need attention, and Critical for catastrophic issues. Add **custom properties** to enrich your log entries with business context (such as a customer ID or order number). Azure Functions also automatically generates **correlation IDs** that link together all logs belonging to a single invocation chain, making it easy to trace a request across multiple functions and services.

**4. Alerts — Turning Data into Action**

Monitoring data is only useful if someone acts on it. Azure Monitor lets you create **Metric Alerts** that fire when a metric crosses a threshold (for example, average duration exceeds 5 seconds), **Log Alerts** that trigger based on the results of a KQL query against your logs (for example, more than 10 errors matching a specific pattern in the last 15 minutes), and **Activity Log Alerts** that watch for control-plane events like function app restarts. Each alert is connected to an **Action Group** that defines who gets notified and how — via email, SMS, voice call, webhook, or even triggering another Azure Function. **Smart Detection** uses machine learning to automatically identify anomalies in failure rates and response times without requiring you to configure explicit thresholds.

**5. Performance Monitoring**

For serverless functions, performance monitoring revolves around understanding latency and resource consumption. **Cold Start** time measures how long it takes to initialize a new function instance — this is the biggest performance concern in Consumption plan. **Warm Start** time measures latency when an instance is already running. **Dependency Calls** track the time your function spends waiting on external services — often the largest contributor to total duration. **Memory Usage** helps you right-size your function and avoid out-of-memory failures.

**6. Best Practices**

Enable Application Insights for every function app from day one — the cost is minimal compared to the operational visibility it provides. Use structured logging with ILogger rather than Console.WriteLine. Set up alerts for error rate spikes and performance degradation. Pay close attention to cold-start times and optimize startup paths. Track dependency performance to identify slow downstream services. Review logs and dashboards regularly as part of your operational routine, and create shared Application Insights dashboards so the entire team has visibility.`,
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
			ID:          1211,
			Title:       "Azure Cosmos DB",
			Description: "Learn Cosmos DB: globally distributed NoSQL database, consistency levels, and multi-model support.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Cosmos DB Fundamentals",
					Content: `Azure Cosmos DB is a globally distributed, multi-model NoSQL database service designed for mission-critical applications that demand low latency, high availability, and elastic scalability anywhere in the world. Imagine a database that can serve data in single-digit milliseconds to users on every continent, automatically replicates across regions, and lets you choose the exact consistency guarantees your application needs — that is Cosmos DB in a nutshell.

**1. API Models — One Database, Many Faces**

Cosmos DB is unique in that it exposes multiple API models over the same underlying storage engine. The **SQL (Core) API** is the default and most feature-rich, providing a document database that you query with a SQL-like syntax. If your team already uses MongoDB, you can adopt the **MongoDB API** and connect with existing MongoDB drivers and tools without rewriting your application. The **Cassandra API** provides wire-protocol compatibility with Apache Cassandra, making it attractive for teams migrating wide-column workloads to the cloud. The **Gremlin API** turns Cosmos DB into a graph database, ideal for social networks, recommendation engines, and fraud detection where relationships between entities matter as much as the entities themselves. Finally, the **Table API** is a drop-in replacement for Azure Table Storage with richer indexing and global distribution.

**2. Consistency Levels — The Spectrum of Trade-offs**

One of Cosmos DB's most innovative features is its five-level consistency spectrum, which lets you choose precisely where you want to land between strong consistency and eventual consistency. **Strong** consistency guarantees that reads always return the most recently committed write — perfect for financial transactions but at the cost of higher latency. **Bounded Staleness** allows reads to lag behind writes by a configurable time window or number of operations, offering near-strong consistency with better performance. **Session** consistency (the default and most popular choice) ensures that within a single client session, reads always reflect that session's own writes — an excellent balance for most web applications. **Consistent Prefix** guarantees that reads never see out-of-order writes, though they may lag behind. **Eventual** consistency offers the highest throughput and lowest latency, with the understanding that reads may temporarily return stale data.

**3. Global Distribution**

Cosmos DB was built from the ground up for global distribution. You can replicate your data to any number of Azure regions with a single click, and Cosmos DB handles synchronization automatically. If a region goes down, **automatic failover** redirects traffic to the next closest region with zero application changes. **Multi-master writes** allow you to write to any region simultaneously, which is essential for applications that need low write latency worldwide — such as a global e-commerce platform where customers in Tokyo and New York both need fast checkout experiences.

**4. Throughput — Understanding Request Units**

Cosmos DB measures throughput in **Request Units (RU)**, a blended metric that accounts for CPU, memory, and I/O consumed by a database operation. A simple point read of a 1 KB document costs roughly 1 RU. You provision throughput in RU/s, and Cosmos DB guarantees that capacity. **Autoscale** mode automatically adjusts your provisioned throughput between a minimum and a maximum based on real-time usage, making it ideal for workloads with unpredictable traffic patterns. **Manual (provisioned)** mode gives you a fixed throughput allocation at a predictable cost, suitable for steady-state workloads.

**5. Best Practices**

Choose the lowest consistency level that satisfies your application's requirements — Session consistency is right for the vast majority of use cases. Invest time in selecting an effective partition key (covered in the next lesson), as it directly determines performance and cost. Enable autoscale for workloads with variable or unpredictable traffic to avoid both over-provisioning and throttling. Leverage the Change Feed for building event-driven architectures that react to database mutations in real time. Monitor RU consumption closely using Azure Monitor and Cosmos DB metrics to identify expensive queries and right-size your throughput.`,
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
					Content: `Effective partitioning is the single most important design decision you will make when working with Cosmos DB. A well-chosen partition key delivers blazing performance and linear scalability; a poor choice leads to hot partitions, throttled requests, and expensive cross-partition queries. Think of partitioning like organizing books in a library: if every book is filed under the same category, that section becomes overcrowded and slow to search, while the rest of the library sits empty.

**1. Partition Key Selection — The Four Golden Rules**

When choosing a partition key, evaluate candidates against four criteria. First, **High Cardinality**: the property should have many distinct values (hundreds, thousands, or millions) so that data spreads across many logical partitions. A userId in a multi-tenant SaaS application is a classic example. Second, **Even Distribution**: the values should be roughly balanced in terms of storage and request volume. If 90% of your traffic targets a single value, that partition becomes a bottleneck regardless of how many other partitions exist. Third, **Query Pattern Alignment**: most of your queries should include the partition key in their WHERE clause, because queries scoped to a single partition are dramatically cheaper and faster than cross-partition fan-out queries. Fourth, **Avoid Hot Partitions**: steer clear of keys that cause a disproportionate share of reads or writes to land on one partition — for example, using a "status" field with only two values (active/inactive) would concentrate nearly all data into just two partitions.

**2. Partition Key Best Practices**

Prefer properties that are set once when a document is created and rarely (or never) change, because Cosmos DB does not allow you to change a document's partition key after creation. If no single property meets all four criteria, consider creating a **synthetic composite key** by concatenating two or more properties (for example, tenantId + "-" + region). Always test your partition key strategy with realistic data volumes and traffic patterns before going to production — what looks balanced with 1,000 documents may become wildly skewed with 10 million.

**3. Logical vs. Physical Partitions**

A **logical partition** is defined by a unique partition key value. All documents with the same partition key value belong to the same logical partition. Cosmos DB then maps logical partitions to **physical partitions** behind the scenes — this mapping is fully managed and transparent to you. Each logical partition can hold up to **20 GB** of data. If you need more, you must choose a partition key with higher cardinality so that data distributes across more logical (and therefore physical) partitions. Provisioned throughput (RU/s) is divided evenly across physical partitions, so having more physical partitions means each one gets a smaller share — another reason balanced distribution matters.

**4. Partitioning Considerations at Scale**

Keep the **20 GB storage limit** per logical partition in mind when designing for large datasets. Each physical partition supports a minimum of **400 RU/s**, and adding throughput can trigger Cosmos DB to split physical partitions for better parallelism. When planning scaling, remember that Cosmos DB distributes data automatically — you never manually manage partitions — but the key you choose determines how effective that distribution is.

**5. Best Practices Summary**

Choose a partition key with high cardinality and even distribution. Monitor partition-level metrics (storage, RU consumption, throttling) using Azure Portal's Cosmos DB Metrics blade. Watch for hot partitions by reviewing the "Partition Key Range Throughput" metric. Use composite partition keys when a single property does not provide enough granularity. Test with realistic data volumes and production-like query patterns. Review partition metrics regularly and adjust your data model if skew emerges.`,
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
					Content: `The Cosmos DB Change Feed is one of the service's most powerful yet underappreciated features. It provides a persistent, ordered log of every insert and update that occurs in a container, sorted by modification time within each logical partition. Think of it as a conveyor belt in a factory: every time a product (document) is created or modified, it appears on the belt, and downstream workers can pick it up and act on it in near-real-time.

**1. Change Feed Features**

The Change Feed enables **real-time processing** of data changes as they happen, with typical end-to-end latencies measured in seconds. It supports **incremental processing**, meaning consumers only see documents that have changed since their last checkpoint — there is no need to scan the entire container. Changes within a logical partition are guaranteed to be **ordered** by modification time, so consumers process events in the correct sequence. The feed is inherently **scalable**: because it is partitioned just like the underlying container, multiple consumers can process different partitions in parallel.

**2. Use Cases — Why Change Feed Matters**

The Change Feed unlocks a wide range of event-driven architectures. **Event Sourcing** captures every mutation as an immutable event, enabling you to reconstruct state at any point in time. **Real-time Analytics** pipelines can transform and aggregate changes as they flow in, feeding dashboards or alerting systems. **Data Synchronization** uses the feed to replicate changes to other data stores, search indexes (such as Azure Cognitive Search), or caches. **Audit Logging** records every modification for compliance, making it straightforward to answer questions like "who changed this record and when." **Trigger Workflows** lets database changes kick off downstream business processes — for example, sending a confirmation email when an order document is created.

**3. Change Feed Processor — Production-Ready Consumption**

While you can read the Change Feed directly via the SDK, the recommended approach for production workloads is the **Change Feed Processor** library. This library manages the complexities of distributed consumption for you. It uses a **Lease Container** — a separate Cosmos DB container — to track which changes each consumer instance has already processed. You can run **multiple processor instances** across different machines, and the library automatically distributes (load-balances) partitions among them. If a processor instance fails, its partitions are automatically reassigned to surviving instances. **Checkpointing** ensures that processing resumes from the last successfully processed item after a restart, so you never lose or double-process changes (assuming your handler is idempotent).

**4. Change Feed Modes**

The Change Feed supports different modes to match your needs. **Latest Version** mode (the default) delivers only the most recent version of each changed document — if a document was updated five times in rapid succession, you see only the final state. **All Versions and Deletes** mode captures every intermediate version and also includes delete events, which is essential for true event-sourcing scenarios. **Time-based** initialization lets you start reading changes from a specific point in time rather than from the beginning, which is useful when onboarding a new consumer.

**5. Best Practices**

Always use the Change Feed Processor library for production workloads rather than writing your own partition management logic. Implement **idempotent processing** in your change handlers — the processor guarantees at-least-once delivery, so your code must handle the possibility of seeing the same change twice. Use a **separate lease container** from your data container to avoid mixing operational data with processing metadata. Monitor **processing lag** (the difference between when a change occurred and when it was processed) to detect consumers falling behind. Handle deletes carefully: in Latest Version mode, deletes are not surfaced, so consider using soft deletes (a "deleted" flag) if you need delete notifications. Scale out processor instances as throughput increases — the library handles partition redistribution automatically.`,
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
			ID:          1212,
			Title:       "Azure Application Gateway",
			Description: "Learn Application Gateway: web traffic load balancer, WAF, SSL termination, and URL routing.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Application Gateway Fundamentals",
					Content: `Azure Application Gateway is a Layer 7 (application layer) load balancer purpose-built for web traffic. Unlike a traditional Layer 4 load balancer that routes packets based on IP addresses and TCP ports, Application Gateway understands HTTP and HTTPS, which means it can make intelligent routing decisions based on URL paths, host headers, cookies, and more. Think of it as a smart receptionist at a large hotel who reads your reservation details and directs you to the right building, floor, and room — rather than randomly assigning you to any available room.

**1. Core Features**

**Layer 7 Load Balancing** is the foundation: Application Gateway inspects every HTTP/HTTPS request and distributes it to the most appropriate backend server. The integrated **Web Application Firewall (WAF)** provides centralized protection against common web exploits such as SQL injection, cross-site scripting (XSS), and other OWASP Top 10 threats — without requiring changes to your application code. **SSL Termination** offloads the computationally expensive work of encrypting and decrypting HTTPS traffic from your backend servers, freeing their CPU for business logic. **URL-based Routing** lets you direct requests to different backend pools based on the URL path (for example, /api/* goes to your API servers while /images/* goes to a storage-backed pool). **Multi-site Hosting** enables a single Application Gateway to serve multiple websites, each identified by its host header and associated with its own backend pool and SSL certificate. **HTTP-to-HTTPS Redirection** ensures that users who accidentally visit the HTTP version of your site are automatically redirected to HTTPS.

**2. SKUs — Choosing the Right Tier**

Application Gateway comes in four SKUs. **Standard** provides basic Layer 7 load balancing without WAF capabilities. **WAF** adds the Web Application Firewall to the Standard tier. **Standard_v2** is the next-generation SKU with significant improvements: autoscaling (no need to pre-size the gateway), zone redundancy for high availability, static VIP addresses, faster provisioning, and header-rewrite support. **WAF_v2** combines all v2 improvements with full WAF functionality. For most production workloads, WAF_v2 is the recommended choice because it provides both advanced routing and security in a single resource.

**3. Routing Rules**

Routing rules define how incoming requests are matched and forwarded. A **Basic** rule sends all traffic from a listener to a single backend pool — simple and effective for a single application. **Path-based** rules inspect the URL path and route to different backend pools accordingly, which is perfect for microservice architectures where /orders, /products, and /users are served by different services. **Multi-site** rules match on the Host header, allowing a single gateway to route traffic for app1.contoso.com and app2.contoso.com to entirely different backends.

**4. Health Probes**

Application Gateway continuously monitors backend health using HTTP or HTTPS probes. You can configure custom probe paths (such as /health), probe intervals, timeout thresholds, and the number of consecutive failures before a backend is marked unhealthy. Unhealthy backends are automatically removed from the rotation and re-added once they start passing probes again, ensuring that users are never routed to a broken server.

**5. Best Practices**

Always use the WAF_v2 SKU for production workloads to get both autoscaling and security. Enable HTTPS redirection so that all traffic is encrypted in transit. Use path-based routing to decouple frontend URLs from backend service topology. Configure health probes that test real application functionality (not just TCP connectivity). Monitor gateway metrics — request count, failed requests, backend response time, and WAF blocked requests — to detect issues early.`,
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
					Content: `The Web Application Firewall (WAF) is your application's front-line defense against the relentless barrage of automated attacks, vulnerability scanners, and exploit attempts that target every internet-facing web application. Rather than hardening each individual application against every known attack vector, WAF provides a centralized security layer at the gateway that inspects every incoming request and blocks malicious traffic before it ever reaches your backend servers.

**1. WAF Rule Sets — Standing on the Shoulders of Security Research**

WAF rule sets are curated collections of detection rules maintained by security experts. The **OWASP 3.2** rule set (based on the Open Web Application Security Project Core Rule Set) is the latest and most comprehensive, covering SQL injection, cross-site scripting (XSS), remote code execution, local file inclusion, and dozens of other attack categories. **OWASP 3.1** is the previous version, still available for backward compatibility. The **Microsoft Bot Manager** rule set specifically targets malicious bots, scrapers, and credential-stuffing tools while allowing legitimate bots (such as search engine crawlers) to pass. **Custom Rules** let you write organization-specific detection logic — for example, blocking traffic from certain geographic regions or requiring specific headers that only your legitimate clients send.

**2. WAF Modes — Detection vs. Prevention**

WAF operates in one of two modes. **Detection** mode evaluates every request against the rule sets and logs matches, but does not block any traffic. This mode is invaluable during the initial rollout because it lets you see exactly what the WAF would block without risking disruption to legitimate users. Once you have reviewed the logs, tuned out false positives, and gained confidence, you switch to **Prevention** mode, where the WAF actively blocks requests that match rules and returns a 403 Forbidden response. The two-phase approach (detect first, then prevent) is a universally recommended best practice.

**3. Rule Actions — Fine-Grained Control**

Each rule (or custom rule) can be configured with one of four actions. **Allow** explicitly permits the request, which is useful for whitelisting known-good traffic patterns. **Block** rejects the request and returns an error response. **Log** records the match for analysis without taking any blocking action — similar to Detection mode but on a per-rule basis. **Redirect** sends the requester to a different URL, which can be used to serve a custom error page or honeypot.

**4. Custom Rules — Tailoring Security to Your Application**

Custom rules give you granular control over WAF behavior. You define **match conditions** based on properties such as the client IP address, request headers, query string parameters, or request body content. Each rule has a **priority** number that determines evaluation order (lower numbers are evaluated first). One especially useful feature is **rate limiting**: you can create rules that limit the number of requests per IP address within a time window, providing an effective defense against application-layer DDoS attacks and brute-force login attempts.

**5. Best Practices**

Always start with Detection mode and leave it running for at least a few days under production traffic before switching to Prevention. Review WAF logs regularly using Log Analytics or Application Insights to understand what is being flagged and whether any legitimate traffic is caught (false positives). Create custom rules for patterns specific to your application — for example, blocking requests to admin endpoints from outside your corporate IP range. Enable rate limiting to protect login pages and API endpoints from abuse. Test new rules and rule-set updates in Detection mode before promoting them to Prevention. Monitor false-positive rates continuously and create exclusions for known-safe patterns. Keep your managed rule sets updated to the latest versions to protect against newly discovered vulnerabilities.`,
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
					Content: `SSL/TLS termination at the Application Gateway is one of the most impactful performance optimizations you can make for a web application. Encrypting and decrypting HTTPS traffic is CPU-intensive work, and when you handle it at the gateway level, you free your backend servers to focus entirely on running application code. Think of it like a security checkpoint at an airport: passengers (requests) go through the security screening (SSL decryption) once at the entrance, and then move freely within the terminal (backend network) without being screened again at every gate.

**1. SSL Termination Benefits**

The performance benefit is the most obvious advantage: SSL/TLS handshakes and symmetric encryption consume significant CPU cycles, and offloading this work to a dedicated gateway appliance means your backend servers can handle more requests per second with the same hardware. **Centralized certificate management** is another major win — instead of deploying and rotating certificates on every backend server, you manage them in one place on the gateway. **Backend simplification** means your application servers do not need SSL certificates at all (for the internal leg of the connection), which simplifies deployment and eliminates an entire class of configuration errors. The net result is lower infrastructure **cost** because you need fewer or smaller backend servers to handle the same traffic volume.

**2. SSL Certificates — Flexible Options**

Application Gateway supports certificates uploaded directly to the gateway resource, but the preferred approach is **Key Vault Integration**: store your certificates in Azure Key Vault and reference them from the gateway. This provides centralized secret management, access auditing, and **auto-renewal** — when your certificate is renewed in Key Vault (either manually or via an integrated Certificate Authority), the gateway picks up the new certificate automatically. Application Gateway also supports **multiple certificates** for different listeners, so you can host multiple domains (each with its own certificate) on a single gateway using Server Name Indication (SNI).

**3. SSL Policies — Hardening Your Cryptographic Posture**

An SSL policy defines which TLS protocol versions and cipher suites your gateway accepts. You can choose a **predefined policy** (Microsoft maintains several that balance compatibility and security) or create a **custom policy** that specifies the exact minimum TLS version (TLS 1.2 is the recommended minimum) and the allowed cipher suites. For high-security environments, you can also require **client certificates** (mutual TLS or mTLS), where the gateway verifies that connecting clients present a valid certificate before forwarding the request — a powerful technique for service-to-service authentication and zero-trust architectures.

**4. End-to-End SSL**

While SSL termination decrypts traffic at the gateway and sends it to backends over plain HTTP, some compliance frameworks (such as PCI DSS) or security policies require encryption for the entire path. **End-to-end SSL** re-encrypts traffic between the gateway and the backend using a separate SSL connection. In this configuration, the gateway still terminates the client-facing SSL (allowing it to inspect and route based on HTTP content), then establishes a new encrypted connection to the backend. You can configure the gateway to **validate backend certificates** to ensure it is talking to a legitimate server, and health probes can use HTTPS to verify that backends are serving traffic correctly over their encrypted ports.

**5. Best Practices**

Store all certificates in Azure Key Vault and reference them from the gateway — never manage certificate files manually on individual resources. Enable SSL termination by default for the performance benefits. Enforce a minimum of TLS 1.2 using a custom or up-to-date predefined SSL policy, and disable weak cipher suites. Monitor certificate expiration dates with Azure Monitor alerts so you are never caught off guard by an expired certificate. Use end-to-end SSL for workloads handling sensitive data or subject to regulatory compliance. Add HSTS (HTTP Strict Transport Security) response headers to instruct browsers to always use HTTPS. Regularly audit your SSL configuration against evolving security best practices.`,
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
			ID:          1213,
			Title:       "Azure Container Instances",
			Description: "Learn Container Instances: serverless containers, quick deployment, and container orchestration basics.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Container Instances Fundamentals",
					Content: `Azure Container Instances (ACI) provides the fastest and simplest way to run containers in Azure, without provisioning or managing any virtual machines. If you have a container image and just want to run it in the cloud right now, ACI is often the answer. Think of it as the serverless equivalent of "docker run" — you specify your image, how much CPU and memory it needs, and Azure handles everything else.

**1. Core Features**

**Serverless Containers** means there is no infrastructure to manage — no OS patches, no cluster nodes, no orchestrator to configure. You simply point ACI at a container image and it runs. **Quick Start** is one of ACI's biggest selling points: containers typically start within seconds, making it dramatically faster than spinning up a virtual machine. **Per-second Billing** ensures you pay only for the exact compute time consumed — when the container stops, billing stops. Each container instance runs with **hypervisor-level isolation**, meaning your container gets the same security boundary as a virtual machine, not just the namespace isolation of a shared container runtime. You specify **custom CPU and memory sizes** to match your workload's requirements, with options ranging from a fraction of a vCPU up to multi-core configurations.

**2. Use Cases — When ACI Shines**

ACI is ideal for workloads that do not require persistent long-running infrastructure. **Quick deployments** and proof-of-concept demonstrations benefit from the zero-setup experience. **Batch processing** jobs — such as video transcoding, data transformation, or report generation — can spin up, do their work, and shut down. **CI/CD pipelines** use ACI as an ephemeral build or test environment that is created on demand and destroyed after the pipeline completes. **Development and testing** workflows benefit from the ability to spin up isolated environments in seconds. ACI can also host individual **microservices** when you do not need the full orchestration capabilities of Kubernetes.

**3. Container Groups — Multi-Container Coordination**

A container group is ACI's equivalent of a Kubernetes pod: a collection of containers that are scheduled on the same host and share networking and storage resources. This enables the **sidecar pattern**, where a main application container is paired with helper containers (for example, a logging agent, a reverse proxy, or a configuration updater). All containers within a group share the same IP address and port namespace, so they communicate over localhost. You can configure **restart policies** (Always, OnFailure, or Never) to control what happens when a container exits.

**4. Networking**

ACI supports both **public IP addresses** for internet-facing workloads and **private IP addresses** for internal services. With **virtual network integration**, you can deploy containers directly into an Azure VNet subnet, giving them private connectivity to other VNet resources (such as databases, application servers, or on-premises networks via VPN). You can also configure **custom DNS servers** so containers resolve names using your organization's DNS infrastructure.

**5. Best Practices**

Use ACI for short-lived, burst, or ephemeral workloads — if you need long-running services with auto-scaling and rolling updates, consider AKS instead. Leverage container groups when you need sidecar containers alongside your main application. Use **managed identities** to authenticate to Azure resources (like Key Vault or Storage) without embedding credentials in your container. Monitor container logs using the Azure CLI or Azure Monitor to troubleshoot issues. Optimize your container images by using minimal base images (like Alpine), reducing layer count, and avoiding unnecessary packages — smaller images mean faster pull times and quicker starts.`,
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
					Content: `Container groups are ACI's way of co-locating multiple containers on the same host so they can share resources and communicate efficiently — much like pods in Kubernetes. This is a fundamental building block for real-world container deployments where a single container image is rarely sufficient; you typically need helper containers for logging, monitoring, proxying, or secret injection running alongside your main application.

**1. Container Group Features**

**Shared Networking** is the defining characteristic: all containers in a group share the same IP address and port namespace, allowing them to communicate over localhost without any network hops. This makes the sidecar pattern natural — your main web server listens on port 80 while a sidecar logging agent listens on port 8080, and they talk to each other as if they were processes on the same machine. **Shared Storage** lets you mount Azure File shares into multiple containers within the group, enabling them to exchange data through a common filesystem. The **Sidecar Pattern** is one of the most powerful architectural patterns enabled by container groups: you deploy supporting containers (reverse proxies, log forwarders, certificate rotators) alongside your main container without modifying the main container's image. **Restart Policies** give you control over what happens when a container exits, which is critical for matching container lifecycle to workload semantics.

**2. Restart Policies — Matching Behavior to Workload**

The **Always** restart policy continuously restarts a container whenever it exits, making it suitable for long-running services like web servers or background workers. **Never** means the container runs once and stops — ideal for batch jobs, data migrations, or one-time tasks where you want the container group to terminate after completion. **OnFailure** restarts the container only when it exits with a non-zero exit code, which is useful for jobs that should complete successfully but should be retried if they crash.

**3. Networking Options**

Container groups can be assigned a **Public IP** address to make them accessible from the internet — useful for web applications, APIs, and demos. Alternatively, a **Private IP** keeps the container group accessible only within its Azure Virtual Network, which is the right choice for backend services that should not be directly exposed. You can attach a **DNS Name** label to create a human-readable FQDN (like myapp.eastus.azurecontainer.io) instead of memorizing IP addresses. **Port Mapping** allows you to expose specific container ports to the group's IP address and control which services are reachable from outside.

**4. Best Practices**

Use container groups whenever you have tightly coupled containers that need to share networking or storage — resist the temptation to cram unrelated services into the same group. Choose restart policies deliberately: Always for services, Never for batch jobs, OnFailure for fault-tolerant tasks. Use Azure File share mounts for shared configuration files, certificates, or data that multiple containers need to access. Monitor resource utilization (CPU and memory) at the container group level to ensure you have allocated enough resources for all containers combined. Embrace the sidecar pattern for cross-cutting concerns like logging, monitoring, and security — this keeps your main application container clean and focused on business logic.`,
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
					Content: `Networking is a critical consideration when deploying containers with Azure Container Instances. The right networking configuration determines whether your container is accessible from the internet, restricted to internal resources, or integrated into your organization's broader network architecture. Understanding these options is essential for both security and functionality.

**1. Networking Options**

ACI provides several networking modes to match different deployment scenarios. A **Public IP** address makes your container group directly accessible from the internet — suitable for public-facing APIs, web applications, and demo environments. A **Private IP** integrates the container group into an Azure Virtual Network, keeping it completely hidden from the internet while allowing communication with other resources on the same VNet. **DNS** name labels give you a human-readable hostname (like mycontainer.eastus.azurecontainer.io) so clients do not need to know the raw IP address. **Port Mapping** lets you control exactly which container ports are exposed on the group's IP address, so you can run multiple services internally while only exposing the ones that need external access.

**2. VNet Integration — Enterprise-Grade Network Isolation**

For production and enterprise workloads, VNet integration is the recommended networking approach. When you deploy a container group into a VNet, you specify a **delegated subnet** — a subnet that is reserved exclusively for ACI. The container group receives a **private IP** from that subnet and can communicate with any resource on the VNet (databases, application servers, on-premises networks connected via VPN or ExpressRoute) using standard private networking. You can attach **Service Endpoints** to the subnet so your containers can access Azure PaaS services (like Storage or SQL Database) over the Azure backbone network rather than the public internet, improving both performance and security. **Network Security Groups (NSGs)** can be applied to the delegated subnet to control inbound and outbound traffic with fine-grained rules — for example, allowing traffic only from specific IP ranges or only on specific ports.

**3. Best Practices**

Always use VNet integration for workloads that handle sensitive data or need to communicate with other internal resources — do not expose containers to the internet unnecessarily. Assign private IPs to backend services and reserve public IPs only for containers that genuinely need internet accessibility. Configure NSGs on the ACI subnet with least-privilege rules: allow only the specific ports and source addresses that are required, and deny everything else. Use Service Endpoints or Private Endpoints to access Azure PaaS services securely without routing traffic through the public internet. Plan your subnet sizing carefully — each container group consumes one IP address, so ensure the subnet has enough address space for your expected deployment scale.`,
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
			ID:          1214,
			Title:       "Azure Database for PostgreSQL",
			Description: "Learn Azure Database for PostgreSQL: managed PostgreSQL, high availability, and scaling.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "PostgreSQL Database Fundamentals",
					Content: `Azure Database for PostgreSQL is a fully managed PostgreSQL database service that handles the undifferentiated heavy lifting of running a production database — patching, backups, high availability, monitoring, and security — so you can focus on your application's data model and queries. If you have ever spent a weekend recovering from a failed manual PostgreSQL upgrade or debugging replication lag on self-managed servers, you will appreciate how much operational burden a managed service eliminates.

**1. Deployment Options — Three Models for Different Needs**

Azure offers three deployment models. **Single Server** is the original managed PostgreSQL offering, suitable for existing workloads, but it is heading toward deprecation in favor of Flexible Server. **Flexible Server** is the recommended choice for new deployments: it provides finer control over database engine parameters, maintenance windows, cost-optimization (with burstable compute tiers and the ability to stop/start the server), and zone-redundant high availability. **Hyperscale (Citus)** is designed for workloads that exceed the capacity of a single node by distributing data and queries across multiple worker nodes using the Citus extension. It is ideal for multi-tenant SaaS applications, real-time analytics dashboards, and time-series data at massive scale.

**2. Service Tiers — Right-Sizing Your Database**

Each deployment model offers multiple compute tiers. The **Basic** tier provides entry-level CPU and memory for development, testing, and light workloads at the lowest cost. **General Purpose** offers a balanced ratio of compute, memory, and I/O, making it the right choice for the majority of production applications — web backends, content management systems, and moderate transactional workloads. **Memory Optimized** tiers provide a higher memory-to-compute ratio for workloads that benefit from large buffer pools, such as complex analytical queries, large working sets, or high-concurrency transactional databases.

**3. Features That Make Managed PostgreSQL Worthwhile**

**Automated backups** are taken daily and transaction logs are archived continuously, enabling **point-in-time restore** to any second within the retention period (up to 35 days). **High availability** configurations replicate data synchronously to a standby in a different availability zone, with automatic failover if the primary fails. **Read replicas** create asynchronous copies of your database in the same or different regions, allowing you to offload read queries and improve global read latency. **Advanced threat protection** monitors database activity for suspicious patterns such as SQL injection attempts or access from unusual locations. Support for **PostgreSQL extensions** (like PostGIS, pg_stat_statements, and pgcrypto) means you can use the rich PostgreSQL ecosystem without leaving the managed service.

**4. Scaling — Grow Without Downtime**

You can **scale compute** up or down by changing the tier or VM size — the operation typically involves a brief reconnection. **Storage** scales independently of compute and can be increased without downtime. **Auto-grow storage** automatically expands disk capacity when free space drops below a threshold, preventing unexpected out-of-disk failures. For read-heavy workloads, adding **read replicas** provides horizontal scaling without changing your primary server.

**5. Best Practices**

Use Flexible Server for all new production deployments — it offers the best combination of features, flexibility, and cost control. Enable zone-redundant high availability for any database that supports a production application. Deploy read replicas to offload reporting, analytics, and read-intensive API queries from the primary. Monitor key performance metrics (CPU utilization, memory usage, IOPS, active connections, replication lag) using Azure Monitor. Test your backup and restore procedures regularly — a backup you have never tested is a backup you cannot trust.`,
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
					Content: `High availability (HA) for your database is not a luxury — it is a fundamental requirement for any application where downtime translates directly into lost revenue, damaged reputation, or broken service-level agreements. Azure Database for PostgreSQL provides built-in HA capabilities that would take weeks of expert engineering to implement on self-managed infrastructure.

**1. High Availability Features**

The HA architecture for Azure Database for PostgreSQL is built around **zone redundancy**: your primary database server and a hot standby are deployed in different Azure Availability Zones, which are physically separate data center buildings within the same region with independent power, cooling, and networking. If the primary zone experiences a failure, **automatic failover** promotes the standby to primary with zero manual intervention. The service provides a **99.99% SLA** when zone-redundant HA is enabled — that translates to less than 53 minutes of allowed downtime per year. Because replication is synchronous, **zero data loss** is guaranteed on failover: every transaction committed on the primary is already present on the standby before the commit is acknowledged to the application.

**2. Flexible Server HA — How It Works Under the Hood**

In the Flexible Server deployment model, enabling HA automatically provisions a **standby server** in a different availability zone. Data is replicated from primary to standby using **synchronous replication**, meaning each write must be confirmed by both servers before it is considered committed. This ensures durability but adds a small amount of write latency (typically single-digit milliseconds within the same region). When a failure is detected — whether hardware failure, OS crash, or network partition — the standby is promoted to primary with a **sub-60-second failover** time. Critically, this is **transparent to the application**: the DNS name resolves to the new primary automatically, so you do not need to update connection strings or reconfigure your application.

**3. Best Practices**

Enable HA for every production database — the cost of the standby replica is far less than the cost of extended downtime during an unplanned failure. Monitor replication lag between primary and standby to detect any synchronization issues early. **Test failover procedures** periodically by triggering a manual failover during a maintenance window and verifying that your application reconnects correctly. Implement **connection retry logic** with exponential backoff in your application code — even with fast failover, there will be a brief period (up to 60 seconds) during promotion where new connections may fail. Monitor HA status in the Azure portal and set up alerts for failover events so your operations team is immediately aware when a failover occurs.`,
					CodeExamples: `# Enable high availability
az postgres flexible-server update \\
    --resource-group myResourceGroup \\
    --name myserver \\
    --high-availability Enabled`,
				},
				{
					Title: "Read Replicas",
					Content: `Read replicas are one of the most effective strategies for scaling PostgreSQL workloads beyond the capacity of a single server. They create asynchronous copies of your primary database that can serve read queries independently, effectively multiplying your read throughput without adding load to the primary. Think of it like photocopying a popular library book: instead of everyone waiting in line for the single original, you distribute copies so multiple readers can access the content simultaneously.

**1. Read Replica Benefits**

**Read Scaling** is the primary use case: by directing SELECT queries to replicas, you offload work from the primary server, which can then dedicate its resources to writes and critical transactions. This is especially valuable for applications with a high read-to-write ratio (such as content platforms, dashboards, or e-commerce product catalogs). **Geographic Distribution** lets you place replicas in different Azure regions, so users in Europe can read from a European replica with low latency while the primary sits in North America. **Disaster Recovery** is another important benefit: if the primary region suffers a catastrophic failure, you can promote a replica in another region to become the new primary, significantly reducing your Recovery Time Objective (RTO). **Reporting and analytics** workloads often execute long-running, resource-intensive queries that can slow down the primary if run directly against it — replicas provide a completely isolated environment for these workloads.

**2. Replication Types — Understanding the Trade-offs**

Azure Database for PostgreSQL uses **asynchronous replication** for read replicas by default: changes committed on the primary are streamed to replicas in near-real-time but without waiting for replica acknowledgment. This means replicas may lag a few seconds behind the primary, but write performance on the primary is not affected by the replica's speed. In contrast, the high-availability standby (discussed in the previous lesson) uses **synchronous replication** with zero data loss but at the cost of slightly higher write latency. When designing your application, keep this distinction in mind: read replicas may serve slightly stale data (eventual consistency for reads), while HA standbys guarantee zero data loss but are not accessible for read queries in the standard HA configuration.

**3. Best Practices**

Use read replicas for any application where reads significantly outnumber writes — even a single replica can double your effective read capacity. Monitor **replication lag** closely using the pg_stat_replication view or Azure Monitor metrics; lag that grows steadily indicates the replica cannot keep up and may need a larger compute tier. Design your application to tolerate the small replication delay when reading from replicas — never read from a replica immediately after a write to the primary if you expect to see the latest data (use read-your-writes patterns instead). Replicas are excellent for **reporting and analytics** because they completely isolate heavy queries from production traffic. Test failover to a replica periodically so you know the process works when a real disaster strikes, and document the steps required to promote a replica to an independent server.`,
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
			ID:          1215,
			Title:       "Azure Service Bus",
			Description: "Learn Service Bus: messaging service, queues, topics, and pub/sub patterns.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Service Bus Fundamentals",
					Content: `Azure Service Bus is a fully managed enterprise message broker that enables reliable, asynchronous communication between decoupled application components. If Azure Storage Queues are like a simple mailbox, Service Bus is like a full-featured postal system with tracking, priority handling, guaranteed delivery, and the ability to route mail to multiple recipients simultaneously. It is the backbone messaging service for enterprise integration, microservice architectures, and any scenario where you need guaranteed message delivery with rich processing semantics.

**1. Messaging Patterns**

Service Bus supports three fundamental messaging patterns. **Queues** implement point-to-point messaging: a sender places a message in the queue, and exactly one receiver picks it up and processes it. This pattern is ideal for workload decoupling and load leveling — for example, a web API that enqueues order processing requests so a backend worker can handle them at its own pace. **Topics** implement publish-subscribe messaging: a publisher sends a message to a topic, and every subscription attached to that topic receives its own copy. This enables fan-out scenarios where a single event (like "new customer registered") needs to trigger multiple independent downstream actions (send welcome email, update CRM, notify analytics). **Relays** provide a bridge between on-premises services and the cloud by enabling bidirectional communication without opening inbound firewall ports.

**2. Queue Features — Enterprise-Grade Reliability**

Service Bus queues go far beyond basic enqueue/dequeue. **FIFO (First-In-First-Out)** ordering ensures messages are delivered in the exact order they were sent (when using sessions). The **Dead-Letter Queue (DLQ)** automatically captures messages that cannot be processed after a configurable number of delivery attempts, preventing poison messages from blocking the queue. **Message Sessions** group related messages by a session ID and guarantee that all messages within a session are processed in order by a single consumer — essential for workflows where sequence matters (like processing all line items for a single order). **Duplicate Detection** uses a message ID to automatically filter out duplicate messages within a configurable time window, protecting against at-least-once delivery semantics causing double processing.

**3. Topic Features — Flexible Publish-Subscribe**

Topics extend the queue model to multiple consumers. Each topic can have multiple **subscriptions**, and each subscription behaves like a virtual queue that receives a copy of every message published to the topic. **Filters** on subscriptions let each consumer receive only the messages it cares about — for example, a subscription might filter for messages where the "region" property equals "europe." **Actions** can modify message properties during delivery, enabling lightweight message transformation without changing the publisher. **Dead-letter subscriptions** work the same way as queue DLQs, capturing messages that a particular subscription fails to process.

**4. Service Tiers**

The **Basic** tier supports queues only and is limited to a maximum message size of 256 KB — suitable for simple scenarios and development. The **Standard** tier adds topics, subscriptions, and a larger feature set at moderate cost. The **Premium** tier provides dedicated resources (no noisy neighbors), larger message sizes (up to 100 MB), and significantly higher throughput — it is the right choice for mission-critical production workloads where performance predictability and SLA guarantees are essential.

**5. Best Practices**

Use topics and subscriptions for any scenario where an event should trigger multiple independent consumers — this decouples consumers from each other and allows you to add new subscribers without modifying the publisher. Always implement dead-letter queue monitoring and handling: unprocessed messages in the DLQ represent data loss or business logic failures that need human attention. Use message sessions when ordering matters. Monitor queue depth (the number of messages waiting) and processing latency to detect consumers falling behind. Choose the Premium tier for production workloads that demand predictable performance and high throughput.`,
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
					Content: `Message sessions are Service Bus's mechanism for guaranteeing strict FIFO (first-in-first-out) message processing while still allowing high concurrency across independent streams. In a standard queue, messages are delivered roughly in order but without hard guarantees — especially when multiple consumers are competing for messages. Sessions solve this by partitioning the message stream into independent ordered channels, each identified by a unique session ID. Think of it like lanes at a bank: each lane processes customers in strict order, but multiple lanes operate independently and concurrently.

**1. Session Features**

**Ordered Processing** is the core guarantee: within a single session, messages are always delivered in the exact order they were enqueued, regardless of how many consumers are connected to the queue. The **Session ID** is a string property set by the sender that groups related messages together — for example, all messages belonging to order #12345 would share the session ID "order-12345." The **FIFO Guarantee** is absolute within a session: message A sent before message B with the same session ID will always be delivered to the consumer before message B. Meanwhile, **Concurrent Sessions** allow different sessions to be processed in parallel by different consumers, so you get the best of both worlds: strict ordering where it matters and parallelism where it is safe.

**2. Use Cases — When Ordering is Non-Negotiable**

**Order Processing** is the textbook example: when an e-commerce system processes a multi-item order, the items must be handled in sequence (validate inventory, charge payment, ship items) rather than in random order. Using a session ID of the order number ensures all events for that order flow through a single processing pipeline. **User Sessions** use a user ID as the session key to maintain context across a series of messages — for example, processing a sequence of user actions in a chat application or maintaining state during a multi-step form submission. **Workflow Processing** relies on sessions to ensure that sequential workflow steps (approve, sign, archive) execute in the correct order even when multiple workflow instances run simultaneously.

**3. Best Practices**

Use sessions whenever message ordering is a business requirement — do not try to enforce ordering with a single consumer and hope for the best, as that approach does not scale and is fragile. Choose **meaningful session IDs** that align with your domain concepts (order IDs, user IDs, workflow instance IDs) rather than arbitrary values. Be aware of **session timeouts**: if a consumer holds a session lock but does not process messages quickly enough, the lock will expire and the session will be reassigned to another consumer, potentially causing reprocessing. Design your session handlers to be efficient and process messages promptly. Take advantage of concurrent session processing by running multiple consumers, each handling a different session — this gives you both ordering guarantees and horizontal scalability.`,
					CodeExamples: `# Create queue with sessions enabled
az servicebus queue create \\
    --resource-group myResourceGroup \\
    --namespace-name myServiceBusNamespace \\
    --name myQueue \\
    --requires-session true`,
				},
				{
					Title: "Dead Letter Queues",
					Content: `Dead-letter queues (DLQs) are the safety net of enterprise messaging. When a message cannot be processed successfully — because it is malformed, because the handler throws an unrecoverable error, or because it has been retried too many times — the message is moved to the dead-letter queue rather than being lost forever or blocking the main queue. Think of a DLQ as the "undeliverable mail" room at the post office: letters that could not be delivered are collected in one place so someone can investigate, fix the address, and redeliver them.

**1. DLQ Features**

Every Service Bus queue and subscription automatically has a dead-letter sub-queue — you do not need to create or configure it separately. Messages land in the DLQ in two primary scenarios: **automatic dead-lettering** occurs when a message exceeds its **max delivery count** (the number of times Service Bus attempts to deliver it to a consumer before giving up) or when the message's time-to-live (TTL) expires without being processed. **Manual dead-lettering** allows your application code to explicitly move a message to the DLQ when it determines that the message is unprocessable — for example, if the message payload fails schema validation. Once in the DLQ, messages sit until they are explicitly inspected, resubmitted, or purged.

**2. DLQ Management — Turning Failures into Fixes**

Managing the DLQ is an operational discipline, not a one-time setup. **Reviewing messages** involves reading from the DLQ (using the same SDK you use for the main queue but targeting the /$deadletterqueue path) and examining each message's body, properties, and dead-letter reason. **Fix and resubmit** is the most common resolution: determine why the message failed (often a bug in the consumer or bad data from the producer), fix the root cause, and send the message back to the main queue for reprocessing. **Purge** removes messages from the DLQ after they have been reviewed and resolved — or when they are no longer relevant. **Monitoring** the DLQ depth (number of messages) over time is critical: a growing DLQ means your consumers have a persistent problem that needs attention.

**3. Best Practices**

Monitor your DLQ depth using Azure Monitor and set up alerts that fire when the count exceeds a threshold (even a single message may warrant investigation). Set the **max delivery count** to a value that balances retry resilience with timeliness — too low and transient failures dead-letter messages unnecessarily, too high and poison messages block the consumer for an extended period. Common values range from 5 to 10. Review dead-lettered messages regularly as part of your operational routine — ignoring the DLQ is equivalent to ignoring error logs. Implement **alerting** that notifies your operations team when new messages appear in the DLQ so problems are addressed promptly. Document clear **resolution procedures** for common dead-letter scenarios so any team member can investigate and resolve issues without guesswork. Consider building an automated DLQ handler that retries messages with exponential backoff or routes them to a human review workflow.`,
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
			ID:          1216,
			Title:       "Azure Event Hubs",
			Description: "Learn Event Hubs: big data streaming, event ingestion, and real-time analytics.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Event Hubs Fundamentals",
					Content: `Azure Event Hubs is a big data streaming platform and event ingestion service designed for scenarios where you need to collect, buffer, and process massive volumes of events in real time. While Service Bus excels at transactional messaging with guaranteed delivery semantics, Event Hubs is optimized for raw throughput: it can ingest millions of events per second from thousands of simultaneous producers, making it the go-to service for telemetry, IoT data pipelines, clickstream analytics, and log aggregation. Think of Event Hubs as a massive funnel that collects a firehose of data and makes it available for downstream processing in an ordered, durable, and scalable way.

**1. Core Features**

**High Throughput** is Event Hubs' defining characteristic: a single namespace can handle millions of events per second, with each event up to 1 MB in size. **Low Latency** ensures events are available to consumers within milliseconds of being published, enabling near-real-time processing for dashboards, alerts, and streaming analytics. **Scalability** is achieved through throughput units (Standard tier) or processing units (Premium tier) that you can scale up or down based on demand — and auto-inflate can automatically increase capacity as load grows. Events are stored for a **configurable retention period** (1 to 90 days on Standard, up to 90 days on Premium), so consumers can replay events or catch up after downtime. **Capture** is a turnkey feature that automatically archives all events to Azure Blob Storage or Azure Data Lake Storage in Avro format, giving you a cost-effective, long-term data lake without writing any code.

**2. Tiers — Matching Capacity to Demand**

The **Basic** tier offers limited throughput (1 throughput unit, 1 consumer group) and is suitable for development or low-volume scenarios. The **Standard** tier supports up to 40 throughput units, 20 consumer groups, and features like Capture, making it appropriate for most production workloads. The **Dedicated** tier provides fully isolated clusters with guaranteed capacity — no shared infrastructure, no noisy neighbors — and is designed for enterprises with extreme throughput requirements or strict compliance needs. The **Premium** tier (newer than Dedicated) offers similar isolation with more flexible scaling and a per-processing-unit pricing model.

**3. Consumer Groups — Independent Views of the Same Stream**

A consumer group represents an independent view of the event stream. Each consumer group maintains its own read position (offset), so multiple applications can read the same events independently at their own pace. For example, one consumer group might feed a real-time alerting system while another feeds a batch analytics pipeline, and a third feeds a data warehouse loader. Each group reads all events — they are not competing for messages like queue consumers would. This design enables you to add new downstream systems without affecting existing ones.

**4. Partitions — The Unit of Parallelism**

Events in an Event Hub are distributed across **partitions**, which are the fundamental unit of parallelism. When a producer sends an event with a **partition key**, Event Hubs hashes the key to determine which partition receives the event, guaranteeing that all events with the same key land in the same partition and are ordered. Within each partition, consumers process events sequentially. The number of partitions you configure (between 1 and 32 on Standard, more on Dedicated) determines the maximum degree of consumer parallelism — you can have at most one active consumer per partition within a consumer group.

**5. Best Practices**

Choose partition keys that distribute events evenly across partitions — a skewed key (like a single device ID that dominates traffic) creates hot partitions that limit throughput. Create dedicated consumer groups for each downstream application so they do not interfere with each other's read positions. Enable Capture for any event hub whose data might have long-term analytical value — it is far cheaper than building a custom archival pipeline. Monitor throughput unit utilization and enable auto-inflate or scale up proactively before you hit capacity limits. Use the Dedicated or Premium tier for workloads with strict latency requirements or regulatory isolation needs.`,
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
					Content: `Partitions are the core mechanism that gives Event Hubs its massive scalability and parallelism. Understanding how partitions work — and choosing the right partitioning strategy — is essential for building event-driven architectures that perform well under load. Think of partitions as lanes on a highway: more lanes mean more vehicles (events) can flow simultaneously, but each lane maintains its own strict ordering of traffic.

**1. Partition Concepts**

A **Partition Key** is a string value assigned by the producer when publishing an event. Event Hubs hashes this key to determine which partition receives the event, ensuring that all events with the same partition key always land in the same partition. This is critical for use cases where ordering matters — for example, if you are tracking a user's clickstream, using the user ID as the partition key guarantees that all of that user's events are processed in sequence. The **Partition Count** is set when the Event Hub is created (1 to 32 on Standard tier, higher on Dedicated). This number is immutable after creation on some tiers, so plan carefully — it defines the upper bound on consumer parallelism. **Ordering** within a partition is guaranteed: events are appended in the order they arrive and consumed in the same order. However, there is no ordering guarantee across different partitions. Each partition has a **throughput limit** of 1 MB/s ingress and 2 MB/s egress (on Standard tier with 1 throughput unit allocated per partition), so the total Event Hub throughput scales linearly with the number of partitions.

**2. Partition Strategy — How Events Get Distributed**

When a producer sends events without specifying a partition key, Event Hubs uses a **Round Robin** strategy to distribute events evenly across all partitions. This maximizes throughput but sacrifices ordering — events from the same logical entity may end up in different partitions. When ordering is required, use a **Partition Key** strategy: specify a meaningful key (device ID, session ID, tenant ID) so that related events cluster together. The **Balanced** approach is implicit when using round-robin: Event Hubs automatically balances the load across available partitions without any manual intervention.

**3. Best Practices**

Choose your partition count based on your expected peak throughput needs, and err on the side of more partitions since you often cannot increase the count later. Use partition keys whenever you need ordering guarantees for related events — but choose keys with high cardinality to avoid hot partitions where one key generates disproportionate traffic. Monitor partition-level metrics (incoming events, outgoing events, backlog) to detect imbalances. If a single partition is consistently hotter than others, reconsider your partition key strategy. Remember that the maximum number of active consumers in a consumer group equals the partition count, so plan partitions with your downstream processing architecture in mind.`,
					CodeExamples: `# Create event hub with partitions
az eventhubs eventhub create \\
    --resource-group myResourceGroup \\
    --namespace-name myEventHubNamespace \\
    --name myEventHub \\
    --partition-count 4`,
				},
				{
					Title: "Consumer Groups",
					Content: `Consumer groups are the mechanism by which multiple applications can independently read from the same Event Hub without interfering with each other. Each consumer group maintains its own view of the event stream — its own set of read positions (offsets) across all partitions. This is fundamentally different from queue-based messaging, where consuming a message removes it from the queue. In Event Hubs, events persist for the configured retention period and can be read by any number of consumer groups, each at its own pace.

**1. Consumer Group Features**

**Independent Streams** means that each consumer group sees the complete event stream independently. If three consumer groups are reading from an Event Hub with 100 events, each group sees all 100 events — events are not divided or load-balanced between groups. This is exactly what you want when different applications need the same data for different purposes (real-time alerting, batch analytics, data archival). Within a single consumer group, **parallel processing** is achieved by assigning different partitions to different consumer instances. For example, if you have 8 partitions and 4 consumer instances in a group, each instance processes events from 2 partitions. **Offset Tracking** allows each consumer group to remember where it left off in each partition, so if a consumer crashes and restarts, it can resume from the last successfully processed event rather than reprocessing everything from the beginning. Consumers within different groups can be **scaled independently** — your real-time alerting pipeline might need 8 instances to keep up, while your batch analytics consumer might only need 2.

**2. Best Practices**

Create a **separate consumer group for each downstream application** that needs to read from the Event Hub. Never share a consumer group between unrelated applications — they will fight over partition assignments and offsets. Scale the number of consumer instances within a group to match your throughput needs, keeping in mind that you can have at most one active consumer per partition per group. Monitor **consumer lag** — the difference between the latest event in a partition and the last event processed by the consumer — to detect consumers falling behind. Implement **proper checkpointing**: periodically save the current offset (using Azure Blob Storage as the checkpoint store is a common pattern) so that recovery after a failure does not require reprocessing large volumes of events. Avoid checkpointing after every single event (too expensive) or too infrequently (too much reprocessing on failure); find a balance based on your processing semantics and throughput.`,
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
			ID:          1217,
			Title:       "Azure Monitor",
			Description: "Learn Azure Monitor: metrics, logs, alerts, and application performance monitoring.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Monitor Fundamentals",
					Content: `Azure Monitor is the unified monitoring platform for everything running in Azure — from virtual machines and databases to serverless functions and Kubernetes clusters. It collects, analyzes, and acts on telemetry from your cloud and on-premises environments, providing a single pane of glass for understanding the health, performance, and availability of your entire infrastructure and application stack. If you think of your Azure environment as a factory, Azure Monitor is the control room: it shows you real-time dashboards, sounds alarms when something goes wrong, and records everything for post-incident analysis.

**1. Data Types — The Four Pillars of Observability**

Azure Monitor organizes telemetry into four categories. **Metrics** are lightweight numerical measurements collected at regular intervals — CPU percentage, memory usage, request count, error rate. They are ideal for real-time monitoring and triggering alerts because they are fast to query and inexpensive to store. **Logs** are richly structured records (text, JSON, key-value pairs) that capture detailed information about events, errors, and operations. Logs are stored in Log Analytics workspaces and queried using the powerful Kusto Query Language (KQL). **Traces** follow a single request as it flows through multiple services in a distributed system, enabling you to pinpoint exactly where latency or failures occur. **Changes** track modifications to resource configurations over time, helping you answer "what changed?" when investigating an incident.

**2. Components — The Monitoring Toolkit**

Azure Monitor is not a single tool but a collection of integrated capabilities. **Metrics Explorer** provides an interactive charting experience for visualizing metrics over time, with support for filtering, splitting, and aggregating. **Log Analytics** is the query engine for log data — you write KQL queries to search, filter, join, and summarize logs from across your environment. **Application Insights** is the application performance monitoring (APM) component that instruments your application code to capture request/response metrics, dependency calls, exceptions, and custom telemetry. **Alerts** define conditions that trigger notifications or automated actions when something needs attention. **Dashboards** let you compose custom views that combine metrics charts, log query results, and other visualizations into a single screen for your operations team.

**3. Data Sources — Comprehensive Coverage**

Azure Monitor automatically collects **platform metrics** from every Azure resource (VM CPU, Storage transactions, SQL DTU usage) without any configuration. **Application Insights telemetry** requires instrumenting your application (usually by adding an SDK or auto-instrumentation agent) but provides deep application-level visibility. You can publish **custom metrics** from your own code for business-specific measurements (orders processed per minute, cache hit ratio). **Activity logs** record control-plane operations (who created, deleted, or modified a resource). **Resource logs** (formerly diagnostic logs) capture data-plane operations specific to each resource type. **Guest OS metrics** collect performance counters from within virtual machines using the Azure Monitor Agent.

**4. Alert Rules — Proactive Problem Detection**

Alert rules transform raw data into actionable notifications. **Metric Alerts** evaluate a metric against a threshold (static or dynamic) and fire when the condition is met — for example, "CPU > 80% for 5 minutes." **Log Alerts** run a KQL query on a schedule and fire when the query returns results — useful for detecting complex conditions like "more than 10 unique users experienced HTTP 500 errors in the last 15 minutes." **Activity Log Alerts** trigger on resource management events — for example, notifying you when someone deletes a production resource. **Smart Detection** (part of Application Insights) uses machine learning to automatically detect anomalies in failure rates, response times, and dependency durations without requiring you to define explicit rules.

**5. Best Practices**

Enable Application Insights for every application from day one — the overhead is minimal and the operational visibility is invaluable. Create alert rules that are meaningful and actionable: every alert should correspond to a condition that requires human investigation or automated remediation. Use Log Analytics workspaces as the central repository for all log data and invest time in learning KQL — it is one of the most powerful query languages in the monitoring space. Set up action groups that route alerts to the right people through the right channels (email for low severity, PagerDuty/SMS for critical). Monitor your monitoring costs: log data ingestion can become expensive at scale, so configure data collection rules to filter out noisy, low-value logs and set daily caps where appropriate.`,
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
					Content: `Log Analytics is the query engine at the heart of Azure Monitor, and its query language — Kusto Query Language (KQL) — is one of the most powerful tools in your monitoring toolkit. KQL allows you to search, filter, aggregate, join, and visualize massive volumes of log data in seconds. Learning KQL is an investment that pays dividends across your entire Azure practice, because the same language is used in Azure Monitor, Azure Sentinel, Azure Data Explorer, and Microsoft Defender.

**1. KQL Basics — Building Blocks of Powerful Queries**

KQL queries operate on **tables**, which represent different data sources ingested into your Log Analytics workspace. Common tables include AppTraces (application log entries), AppExceptions (exception telemetry), AppRequests (HTTP request telemetry), Heartbeat (agent health checks), and AzureActivity (control-plane operations). You chain **operators** using the pipe (|) symbol to progressively transform data: "where" filters rows, "project" selects or renames columns, "summarize" groups and aggregates, "join" combines data from multiple tables, "extend" adds computed columns, and "order by" sorts results. **Functions** provide built-in capabilities for aggregation (count, sum, avg, percentile), time manipulation (ago, bin, datetime_diff), and string processing (contains, startswith, extract with regex). Query results can be rendered as **visualizations** — timecharts, bar charts, pie charts, and scatter plots — directly in the Log Analytics query editor or embedded in Azure Dashboards and Workbooks.

**2. Common Query Patterns**

**Error Analysis** queries filter for high-severity log entries or exceptions and summarize them by type, message, or source to identify the most frequent or impactful failures. For example, querying AppExceptions and summarizing count by ExceptionType reveals which exception types are dominating your error landscape. **Performance** queries identify slow operations by analyzing request durations — using percentile calculations to find the 95th or 99th percentile response time, or filtering for requests that exceed an acceptable threshold. **Usage Patterns** queries aggregate request counts over time (using the bin operator to bucket by hour or day) to understand traffic trends, peak hours, and growth trajectories. **Correlation** queries join multiple tables on a shared field (like OperationId) to trace a single user request across application logs, dependency calls, and exceptions — the bread and butter of distributed tracing investigation.

**3. Best Practices**

Always scope your queries with a **time range** — querying unbounded data is slow and expensive. Start with a narrow window (last hour) for investigation and widen only if needed. **Optimize queries** by placing the most selective filters early in the pipeline (close to the table reference) so KQL can prune data before expensive operations like joins or summarizations. Create **saved queries** and **functions** for patterns you use repeatedly — this promotes consistency and saves time. Build **workbooks** (interactive, parameterized dashboards) for recurring analysis scenarios like weekly error reviews or monthly performance reports. Share workbooks with your team so everyone works from the same data and queries.`,
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
					Content: `Alert rules are the mechanism that transforms passive monitoring data into proactive incident detection. Without alerts, your beautiful dashboards and comprehensive logs are only useful when someone is actively watching them — and in the middle of the night, nobody is. Alert rules continuously evaluate conditions against your monitoring data and fire notifications when something needs attention, ensuring that problems are detected and addressed quickly regardless of when they occur.

**1. Alert Types — Three Flavors for Different Data**

**Metric Alerts** evaluate a metric value against a threshold at regular intervals. They are the fastest and most cost-effective alert type because metrics are lightweight and evaluated in near-real-time. Use metric alerts for straightforward conditions like "VM CPU > 90% for 5 minutes" or "HTTP error count > 50 in 1 minute." Azure also supports **dynamic thresholds** that use machine learning to establish a normal baseline and alert when the metric deviates from expected behavior — ideal when you do not know what the "right" static threshold should be. **Log Alerts** run a KQL query on a schedule and fire when the query returns a specified number of results or when an aggregated value crosses a threshold. They are more flexible than metric alerts (you can express arbitrarily complex conditions in KQL) but have higher latency because queries run on a schedule (as frequently as every 1 minute). Use log alerts for conditions like "more than 5 distinct users encountered a specific exception in the last 10 minutes." **Activity Log Alerts** fire when a specific control-plane event occurs — such as a resource being deleted, a role assignment changing, or a service health incident being reported. These are essential for governance and security monitoring.

**2. Alert Configuration — Anatomy of an Alert Rule**

Every alert rule has four key components. The **Condition** defines what triggers the alert — a metric threshold, a KQL query result, or an activity log event pattern. The **Action Group** specifies who to notify and how: email, SMS, voice call, push notification, webhook, Azure Function, Logic App, or ITSM integration. You can assign a **Severity** level (0 through 4, where 0 is Critical and 4 is Informational) to prioritize alerts in your operations workflow. The **Frequency** (evaluation frequency and time window) controls how often the condition is checked and how much historical data each evaluation considers.

**3. Best Practices**

Set thresholds that are **meaningful and actionable**: every alert should require a specific response from someone. If you find yourself ignoring certain alerts, either tune the threshold or disable the rule — alert fatigue is the enemy of operational reliability. Use **action groups** to route alerts to the right team through the right channel: page the on-call engineer for critical production issues, send an email digest for informational trends. **Test alert rules** by deliberately triggering the condition in a non-production environment to verify that notifications arrive as expected. Review and **tune alerts regularly** — as your application and infrastructure evolve, thresholds that were appropriate six months ago may generate too many false positives or miss real issues today. **Document the purpose and expected response** for each alert rule so that anyone on the on-call rotation knows exactly what the alert means and what action to take.`,
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
			ID:          1218,
			Title:       "Azure Resource Manager",
			Description: "Learn Resource Manager: ARM templates, infrastructure as code, and resource organization.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Resource Manager Fundamentals",
					Content: `Azure Resource Manager (ARM) is the deployment and management layer that underpins every interaction you have with Azure — whether you click a button in the Azure Portal, run a CLI command, call the REST API, or deploy an infrastructure-as-code template. Every request to create, update, or delete an Azure resource flows through ARM, which authenticates the caller, authorizes the action, and then routes the request to the appropriate resource provider. Think of ARM as the air traffic control tower for your entire Azure estate: no plane (resource operation) takes off or lands without ARM's coordination.

**1. Key Concepts — The Organizational Hierarchy**

Azure organizes resources into a four-level hierarchy that mirrors how enterprises structure their cloud environments. At the top are **Management Groups**, which let you organize multiple Azure subscriptions into a tree structure and apply governance policies (like Azure Policy or RBAC role assignments) that cascade down to all subscriptions beneath them — ideal for large organizations with multiple business units. **Subscriptions** are the billing and access boundary: each subscription has its own billing account, spending limits, and resource quotas. Within a subscription, **Resource Groups** act as logical containers that group related resources together for lifecycle management — when you delete a resource group, all resources inside it are deleted too, which makes cleanup straightforward. Finally, **Resources** are the individual Azure services themselves: a virtual machine, a storage account, a database, a virtual network.

**2. ARM Templates — Declarative Infrastructure as Code**

ARM templates allow you to define your entire infrastructure in a declarative JSON file that describes the desired end state rather than the step-by-step procedure to achieve it. This approach is called Infrastructure as Code (IaC), and it provides enormous benefits: repeatable deployments across environments (dev, staging, production), version-controlled infrastructure that can be code-reviewed and audited, and idempotent execution — meaning you can deploy the same template multiple times and ARM will only make the changes needed to reach the desired state without duplicating resources. **Template Specs** let you publish reusable templates to Azure as versioned artifacts that other teams can consume, promoting standardization across the organization. **Bicep** is Microsoft's answer to the verbosity of JSON templates — it is a domain-specific language that compiles down to ARM JSON but offers dramatically improved readability, IntelliSense, type safety, and modularity.

**3. Template Structure — The Four Sections**

Every ARM template has four main sections. **Parameters** define the inputs that make your template flexible — for example, the name of a storage account, the SKU, or the Azure region. Parameters can have default values, allowed values, and validation rules. **Variables** are computed values derived from parameters or expressions — they reduce repetition and make templates easier to maintain. **Resources** is the core section where you declare each Azure resource you want to deploy, including its type, API version, name, location, and properties. **Outputs** return values after deployment completes — for example, the connection string of a newly created database or the public IP address of a load balancer — making it easy to chain deployments together.

**4. Deployment Methods — Multiple On-Ramps**

ARM templates can be deployed through virtually any interface Azure supports. The **Azure Portal** offers a visual deployment experience for ad-hoc deployments. **Azure CLI** and **Azure PowerShell** are the go-to tools for scripted and automated deployments from developer workstations or CI/CD pipelines. The **REST API** enables programmatic deployments from custom applications. **GitHub Actions** and **Azure DevOps Pipelines** integrate template deployments into your CI/CD workflows, enabling fully automated infrastructure provisioning as part of your software delivery pipeline.

**5. Best Practices**

Organize resources into resource groups by lifecycle and ownership — resources that are deployed, updated, and deleted together should live in the same group. Apply **tags** consistently for cost allocation, environment identification, and ownership tracking. Use ARM templates or Bicep for every deployment, even one-off resources, because manual Portal deployments are not reproducible and cannot be code-reviewed. Prefer Bicep over raw JSON for new templates — it is easier to read, write, and maintain. Implement **Azure Policy** at the management group or subscription level to enforce organizational standards (like required tags, allowed regions, or mandatory encryption). Store your templates in version control and require pull request reviews before deploying to production.`,
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
					Content: `Bicep is Microsoft's domain-specific language for deploying Azure resources, designed to replace the verbose and error-prone experience of writing raw ARM JSON templates. If ARM JSON is like writing assembly language — technically powerful but tedious and hard to read — then Bicep is like writing in a modern high-level language: concise, readable, and equipped with tooling that catches mistakes before deployment. Bicep compiles transparently to ARM JSON, so there is zero loss of capability — anything you can do in ARM JSON, you can do in Bicep, but with far less boilerplate.

**1. Bicep Advantages — Why the Industry is Moving to Bicep**

**Readability** is the most immediately obvious advantage. A Bicep file that defines a storage account is roughly one-third the length of the equivalent ARM JSON, with no curly-brace nesting hell and no repetitive schema declarations. **Type safety** means the Bicep compiler validates your resource definitions at compile time — it catches typos in property names, invalid enum values, and missing required properties before you ever attempt a deployment, saving you the frustrating cycle of deploy-fail-fix-redeploy. **IntelliSense** in Visual Studio Code (via the official Bicep extension) provides auto-completion for every Azure resource type, property, and API version, making it dramatically faster to author templates without constantly referencing documentation. **Modules** allow you to break large templates into reusable, composable pieces — for example, a networking module, a database module, and a compute module that can be independently versioned and shared across teams.

**2. Bicep Structure — Clean and Intuitive**

A Bicep file is organized into clearly delineated sections. **Parameters** declare inputs using a clean syntax: "param storageAccountName string" is all it takes, compared to the multi-line JSON object required in ARM templates. Parameters support decorators like @minLength(), @maxLength(), @allowed(), and @description() for validation and documentation. **Variables** are declared with "var" and can reference parameters, other variables, or built-in functions. **Resources** are declared with the resource keyword followed by a symbolic name, the resource type and API version, and a property block — the syntax reads almost like natural language. **Outputs** return values from the deployment, which is essential for chaining deployments or retrieving connection strings. **Modules** reference other Bicep files (local or from a registry) and pass parameters to them, enabling clean separation of concerns — for example, your main.bicep might call modules/network.bicep and modules/database.bicep.

**3. Bicep Registry — Sharing and Reusing Modules**

Bicep supports publishing modules to Azure Container Registry (ACR), creating a private module registry for your organization. Teams can publish versioned, tested infrastructure modules that other teams consume by referencing the registry path in their Bicep files. This promotes standardization (everyone uses the same approved networking module) and reduces duplication (the module is authored once and maintained centrally).

**4. Best Practices**

Use Bicep for all new infrastructure-as-code projects — there is no reason to start with raw ARM JSON in new work. Migrate existing ARM JSON templates to Bicep incrementally using the "az bicep decompile" command. Create reusable modules for common patterns in your organization (like a standard VNet layout, a secure storage account, or a web app with diagnostics enabled) and publish them to a Bicep module registry. Leverage decorators (@description, @secure, @allowed) to make your parameters self-documenting. Version-control all Bicep files alongside your application code and require pull request reviews before deploying infrastructure changes to production. Use the "what-if" deployment mode to preview changes before applying them, especially for templates that manage production resources.`,
					CodeExamples: `# Compile Bicep
az bicep build --file main.bicep

# Deploy Bicep
az deployment group create \\
    --resource-group myResourceGroup \\
    --template-file main.bicep`,
				},
				{
					Title: "Template Deployment",
					Content: `Deploying ARM and Bicep templates is the moment where your infrastructure-as-code goes from a design document to running cloud resources. Understanding the deployment methods, modes, and safety mechanisms is critical because a misconfigured deployment can delete resources, cause downtime, or provision expensive infrastructure you did not intend. Think of deployment like launching a rocket: you want thorough pre-flight checks, a clear understanding of what will happen, and the ability to abort if something looks wrong.

**1. Deployment Methods — Choose Your Interface**

Azure provides multiple on-ramps for deploying templates, each suited to different workflows. The **Azure Portal** offers a visual deployment experience where you upload a template, fill in parameters through a form, and click Deploy — convenient for one-off deployments and learning but not suitable for production automation. **Azure CLI** ("az deployment group create") is the most popular choice for developers and DevOps engineers because it integrates naturally into shell scripts, Makefiles, and CI/CD pipelines. **Azure PowerShell** ("New-AzResourceGroupDeployment") is preferred by Windows-centric teams and integrates well with existing PowerShell automation. The **REST API** enables programmatic deployments from custom applications or platforms that do not use the Azure SDK. **GitHub Actions** and **Azure DevOps Pipelines** embed template deployments into your software delivery pipeline, enabling you to deploy infrastructure changes through the same pull-request-and-merge workflow you use for application code.

**2. Deployment Modes — Incremental vs. Complete**

ARM supports two deployment modes that fundamentally differ in how they handle existing resources. **Incremental** mode (the default and safest option) adds or updates the resources defined in the template while leaving any existing resources in the resource group untouched. If your template defines a storage account and a virtual machine, but the resource group also contains a database that is not in the template, the database remains unaffected. **Complete** mode is far more aggressive: it treats the template as the complete desired state and deletes any resources in the resource group that are not defined in the template. Complete mode is powerful for ensuring exact resource parity with your template, but it is also dangerous — accidentally omitting a resource from the template means it gets deleted. Use Complete mode only when you are confident that your template defines every resource in the group, and always preview with what-if first.

**3. Safety Mechanisms — Validate and What-If**

ARM provides two critical safety mechanisms that you should use before every production deployment. The **validate** command ("az deployment group validate") checks your template for syntax errors, invalid resource definitions, and parameter mismatches without actually deploying anything — it is the equivalent of a compiler check. The **what-if** command ("az deployment group what-if") goes further: it compares your template against the current state of the resource group and shows you exactly what will be created, modified, or deleted. The output is color-coded (green for creates, yellow for modifications, red for deletes), making it easy to spot unintended changes. Think of what-if as a dry run — it gives you confidence that the deployment will do exactly what you expect before you commit.

**4. Best Practices**

Always validate templates before deploying — catch errors at compile time rather than at deploy time. Run what-if before every production deployment and review the output carefully, especially when using Complete mode. Test template deployments in a non-production environment (dev or staging) before applying to production. Store templates in version control (Git) and require pull request reviews for changes that affect production infrastructure. Document every parameter with a description so that consumers of your template understand what each input controls. Use deployment scopes appropriately: deploy at the resource group level for most resources, at the subscription level for resource groups and policies, and at the management group level for cross-subscription governance.`,
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
			ID:          1219,
			Title:       "Azure Automation",
			Description: "Learn Azure Automation: runbooks, automation accounts, and infrastructure automation.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Automation Fundamentals",
					Content: `Azure Automation is a cloud-based service that provides process automation, configuration management, and update orchestration across your Azure and on-premises environments. If you have ever found yourself manually running the same PowerShell script every Monday morning to clean up old snapshots, restart a stuck service, or generate a compliance report, Azure Automation is designed to eliminate that repetitive toil. Think of it as a tireless robotic assistant that executes your operational scripts on schedule, on demand, or in response to alerts — consistently, reliably, and without forgetting a step.

**1. Components — The Building Blocks of Automation**

At the center of Azure Automation is the **Automation Account**, a container that holds all your automation resources in one place. Within an account, **Runbooks** are the automated scripts that perform your tasks — written in PowerShell, Python, or the PowerShell Workflow language. **Modules** extend the capabilities of your runbooks by importing PowerShell modules (like Az modules for Azure management, or custom modules for your internal tools). **Variables** store values that runbooks can read at runtime — things like resource group names, thresholds, or feature flags — without hardcoding them into the script. **Credentials** securely store username/password pairs that runbooks use to authenticate to external systems, and **Certificates** store X.509 certificates for scenarios like client authentication or encryption.

**2. Runbook Types — Choose the Right Tool**

**PowerShell** runbooks are the most popular choice because PowerShell is the lingua franca of Azure administration. They execute as standard PowerShell scripts with full access to the Az module library. **PowerShell Workflow** runbooks support long-running operations with checkpointing — if the runbook is interrupted, it can resume from the last checkpoint rather than starting over. **Python 2/3** runbooks are available for teams that prefer Python or need to leverage Python-specific libraries. **Graphical** runbooks provide a visual drag-and-drop designer for building automation workflows without writing code — useful for less technical team members who need to author simple automations.

**3. Features — Beyond Simple Script Execution**

Azure Automation extends well beyond running scripts. **Hybrid Worker** allows runbooks to execute on machines in your on-premises data center or in other cloud providers, enabling automation of resources that are not directly accessible from Azure. **Update Management** provides a centralized dashboard for assessing patch compliance and scheduling OS updates across both Azure VMs and on-premises servers — a critical capability for security and compliance. **Change Tracking** monitors configuration changes on your servers (files, registry keys, services, software) and records them for auditing and troubleshooting. **Inventory** collects detailed software and configuration inventories from managed machines. **State Configuration (DSC)** lets you define the desired state of your servers using PowerShell DSC configurations and automatically enforce that state, correcting any drift.

**4. Scheduling and Triggering — When and How Runbooks Execute**

**Schedules** trigger runbooks at specific times or on recurring intervals (hourly, daily, weekly, monthly), making them perfect for routine maintenance tasks like nightly backups or weekly compliance checks. **Webhooks** expose an HTTP endpoint that triggers a runbook when called, enabling integration with external systems — for example, a monitoring tool can call a webhook to trigger a remediation runbook when it detects an issue. **Azure Monitor Alert** integration lets you trigger runbooks automatically in response to monitoring alerts, creating closed-loop automation where Azure detects a problem and fixes it without human intervention.

**5. Best Practices**

Use **managed identities** (system-assigned or user-assigned) for authenticating runbooks to Azure resources — this eliminates the need to store credentials and simplifies secret rotation. Store any secrets your runbooks need in **Azure Key Vault** and access them at runtime using the managed identity. Organize common logic into reusable **PowerShell modules** that multiple runbooks can import. Maintain separate automation accounts for production and testing so you can safely develop and test runbooks without risking production systems. Monitor runbook execution through the Automation Account's job history and set up alerts for failed jobs so operational issues are caught promptly.`,
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
					Content: `Developing runbooks is where Azure Automation goes from a concept to a working system that saves your team hours of manual effort. Writing a good runbook is much like writing good application code — it requires thoughtful structure, error handling, parameterization, and testing. The difference is that runbooks typically automate operational tasks (provisioning, cleanup, patching, reporting) rather than serving end-user requests, so reliability and idempotency are paramount.

**1. Runbook Types — Matching Language to Task**

**PowerShell** runbooks are the workhorses of Azure Automation. They support the full PowerShell language and have access to all Azure PowerShell (Az) modules, making them the natural choice for Azure resource management tasks like creating VMs, rotating secrets, or generating compliance reports. **PowerShell Workflow** runbooks add checkpointing and parallel execution capabilities — if a long-running workflow is interrupted (by a platform update or transient failure), it resumes from the last checkpoint rather than restarting from scratch. This is especially valuable for runbooks that take hours to complete, such as patching hundreds of servers or processing large datasets. **Python** runbooks bring the Python ecosystem to Azure Automation, which is valuable when you need libraries that do not have PowerShell equivalents or when your team is more proficient in Python. **Graphical** runbooks provide a visual canvas where you drag and drop activities, connect them with links, and configure properties through a form-based UI — ideal for team members who are not comfortable writing code but need to build simple automations.

**2. Runbook Development Workflow — From Draft to Production**

The development workflow follows a structured path from authoring to production. You write your runbook code in the **Azure Portal editor**, in **Visual Studio Code** with the Azure Automation extension, or in any text editor and then upload it. The **Test Pane** in the Azure Portal is an invaluable tool: it lets you execute the runbook in a sandbox environment, pass test parameters, view real-time output, and debug issues — all without affecting production. Once testing is complete, you **publish** the runbook, which creates a versioned snapshot that is available for scheduling, webhook triggering, or manual execution. **Versioning** ensures you can roll back to a previous version if a new change introduces a bug. **Parameters** make your runbooks flexible and reusable — instead of hardcoding a resource group name, you accept it as a parameter so the same runbook can operate on any resource group. **Output** values let runbooks return results to callers or downstream automation, enabling runbook chaining where the output of one runbook feeds the input of the next.

**3. Error Handling — Building Resilient Automation**

Robust error handling separates production-quality runbooks from fragile scripts. Use try-catch blocks around every operation that could fail (Azure API calls, network operations, file system access). Log detailed error information including the error message, exception type, and any relevant context (like which resource or iteration failed). Implement **retry logic** with exponential backoff for transient failures — Azure API calls can occasionally fail due to throttling or temporary service issues. For critical runbooks, send notifications (email, Teams message, or webhook) when errors occur so the operations team can investigate promptly.

**4. Best Practices**

Test runbooks thoroughly in the Test Pane before publishing — a failed production runbook can cause outages or leave resources in an inconsistent state. Use parameters for all configurable values (resource names, thresholds, regions) to make runbooks reusable across environments. Handle errors gracefully with try-catch blocks and produce meaningful log output that helps with troubleshooting. Use **managed identities** for authentication to Azure resources — never embed credentials or API keys in runbook code. Document each runbook with a clear description of its purpose, parameters, prerequisites, and expected behavior so that any team member can understand and maintain it.`,
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
					Content: `Hybrid Runbook Workers extend the reach of Azure Automation beyond the Azure cloud, allowing your runbooks to execute directly on machines in your on-premises data centers, in other cloud providers (AWS, GCP), or on edge devices. This is a game-changer for organizations with hybrid infrastructure because it means you can use a single automation platform — Azure Automation — to manage resources everywhere, not just in Azure. Think of the hybrid worker as an agent that sits on your on-premises server, listens for instructions from Azure Automation, and executes runbooks locally with full access to the local network, file system, and installed software.

**1. Hybrid Worker Benefits — Why Run Locally?**

**On-premises access** is the primary driver: many organizations have resources that are not reachable from the Azure cloud — legacy databases behind corporate firewalls, file servers on internal networks, or applications that communicate only on private subnets. A hybrid worker sitting inside that network can access these resources directly, eliminating the need for complex VPN tunnels or firewall exceptions. **Network isolation** means the runbook executes within your controlled network environment, which is important for workloads that handle sensitive data that must not leave the premises. **Custom software** availability is another key benefit: if your runbook needs to invoke a proprietary command-line tool, a licensed application, or a driver that only exists on your on-premises servers, the hybrid worker runs it natively. **Compliance** requirements often dictate that certain data processing or management tasks must occur within specific geographic boundaries or on specific infrastructure — hybrid workers let you satisfy these constraints while still benefiting from Azure Automation's scheduling, monitoring, and management capabilities.

**2. Hybrid Worker Groups — Organizing and Scaling**

Hybrid workers are organized into **Hybrid Worker Groups**, which act as pools of execution targets. When a runbook is scheduled to run on a hybrid worker group, Azure Automation selects an available worker from the pool to execute it. **Windows Workers** run on Windows Server machines and execute PowerShell and PowerShell Workflow runbooks natively. **Linux Workers** run on supported Linux distributions and execute Python and PowerShell (via PowerShell Core) runbooks. **Load Balancing** across a worker group means that if you have multiple workers, Azure Automation distributes runbook jobs among them, providing both scalability (handle more concurrent jobs) and resilience (if one worker goes offline, others pick up the work). You can create multiple groups organized by purpose (one for database maintenance, another for file processing) or by network zone (one for the DMZ, another for the internal network).

**3. Architecture and Connectivity**

The hybrid worker communicates with Azure Automation through an outbound HTTPS connection — it polls Azure for pending jobs and pushes back status updates and output. This means you do not need to open any inbound firewall ports, which simplifies security configuration. The worker runs as a Windows service or Linux daemon and can be installed alongside other applications on the same machine, or on a dedicated automation server.

**4. Best Practices**

Deploy hybrid workers whenever your runbooks need to access resources that are not reachable from the Azure sandbox — do not try to work around network isolation with complex firewall rules when a local worker is the cleaner solution. Monitor worker health regularly using Azure Automation's built-in worker status reporting, and set up alerts for workers that go offline or become unresponsive. Implement proper security by running workers under service accounts with least-privilege permissions and keeping the worker agent updated to the latest version. Use worker groups for load balancing and resilience — never rely on a single worker for critical automation tasks. For compliance-sensitive workloads, document which worker groups operate in which network zones and ensure your runbook assignments align with your data residency requirements.`,
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
