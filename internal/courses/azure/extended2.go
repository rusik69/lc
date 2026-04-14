package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1235,
			Title:       "Azure Container Registry & Container Apps",
			Description: "Master Azure Container Registry for image management and Azure Container Apps for modern serverless container deployments.",
			Order:       35,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Container Registry (ACR)",
					Content: `Azure Container Registry is a managed Docker registry service for storing and managing container images and OCI artifacts. It is essential for any Azure AKS or container-based workflow.

**1. SKU Tiers:**
*   **Basic:** Development use. 10 GB storage, 2 webhooks.
*   **Standard:** Production workloads. 100 GB storage, 10 webhooks, expanded throughput.
*   **Premium:** Geo-replication, content trust, private link, customer-managed keys. 500 GB+.

**2. Key Features:**
*   **Geo-replication (Premium):** Replicate images across Azure regions for multi-region deployments. Push once, pull locally.
*   **ACR Tasks:** Build container images in Azure without a local Docker daemon. Supports multi-step tasks and automatic rebuilds on source code commit or base image update.
*   **Content Trust:** Image signing using Docker Content Trust/Notary. Ensures only signed images are deployed.
*   **Private Link:** Access ACR over a private endpoint in your VNet (no public internet exposure).
*   **Artifact streaming (Premium):** Start containers before the full image is downloaded. Reduces cold start time for large images.

**3. Authentication:**
*   **Azure AD:** Recommended. Use managed identities for AKS to pull images.
*   **Admin account:** Simple username/password. Use only for development/CI.
*   **Service principal:** For CI/CD pipelines.
*   **Token-based:** Scoped repository-level access.

**4. Image Lifecycle:**
*   Enable retention policies to automatically delete old untagged manifests.
*   Use ` + "`" + `az acr purge` + "`" + ` to clean up images based on age and tag filters.
*   Tag strategy: Use git-sha or build-id for traceability, plus ` + "`" + `latest` + "`" + ` for convenience.`,
					CodeExamples: `# Create a Premium ACR
az acr create \
  --resource-group myRG \
  --name myregistry \
  --sku Premium \
  --admin-enabled false

# Build an image in ACR (no local Docker needed)
az acr build \
  --registry myregistry \
  --image myapp:v1.0.0 \
  --file Dockerfile .

# Push a local image to ACR
az acr login --name myregistry
docker tag myapp:latest myregistry.azurecr.io/myapp:v1.0.0
docker push myregistry.azurecr.io/myapp:v1.0.0

# Attach ACR to AKS (allows AKS to pull images)
az aks update \
  --resource-group myRG \
  --name myAKS \
  --attach-acr myregistry

# Enable geo-replication
az acr replication create \
  --registry myregistry \
  --location westeurope

# List repositories and tags
az acr repository list --name myregistry --output table
az acr repository show-tags --name myregistry --repository myapp

# Purge old images (keep last 5 tags)
az acr run --registry myregistry --cmd "acr purge \
  --filter 'myapp:.*' \
  --ago 30d \
  --keep 5 \
  --untagged" /dev/null`,
				},
				{
					Title: "Azure Container Apps",
					Content: `Azure Container Apps is a serverless container platform built on Kubernetes (internally uses AKS + Envoy + KEDA). It abstracts away cluster management while providing powerful scaling and traffic features.

**1. When to Use Container Apps vs AKS:**
*   **Container Apps:** Microservices, APIs, event-driven processing. You don't want to manage Kubernetes.
*   **AKS:** You need full Kubernetes control, custom operators, specific CNI plugins, or GPU workloads.
*   **Container Instances:** Simple, short-lived containers with no scaling needs.

**2. Key Concepts:**
*   **Environment:** Shared boundary for container apps. Apps in the same environment share a virtual network, logging, and Dapr components.
*   **App:** A single deployable unit with one or more containers.
*   **Revision:** An immutable version of your app. Traffic can be split between revisions (blue/green, canary).
*   **Replica:** An instance of a revision. Auto-scaled by rules.

**3. Scaling:**
*   **HTTP:** Scale based on concurrent HTTP requests (0 to N).
*   **KEDA:** Scale on any event source (Azure Queue, Kafka, Cron, custom metrics).
*   **Scale to zero:** When no traffic or events, replicas drop to 0 (pay nothing).
*   **Min/max replicas:** Set bounds to control cost and availability.

**4. Networking:**
*   **External:** Publicly accessible with an auto-generated FQDN and TLS.
*   **Internal:** Only accessible within the VNet (for backend services).
*   Built-in Envoy-based ingress with traffic splitting.

**5. Dapr Integration:**
*   Built-in Dapr sidecar for service invocation, pub/sub, state management, and bindings.
*   Enables microservice patterns without client library dependencies.`,
					CodeExamples: `# Create a Container Apps environment
az containerapp env create \
  --name myenv \
  --resource-group myRG \
  --location eastus

# Deploy a container app
az containerapp create \
  --name myapi \
  --resource-group myRG \
  --environment myenv \
  --image myregistry.azurecr.io/myapi:v1.0 \
  --target-port 8080 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 10 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --registry-server myregistry.azurecr.io

# Update with traffic splitting (canary)
az containerapp update \
  --name myapi \
  --resource-group myRG \
  --image myregistry.azurecr.io/myapi:v2.0

az containerapp ingress traffic set \
  --name myapi \
  --resource-group myRG \
  --revision-weight \
    myapi--v1=90 \
    myapi--v2=10

# KEDA scaling rule (scale on Azure Queue length)
az containerapp update \
  --name myworker \
  --resource-group myRG \
  --min-replicas 0 \
  --max-replicas 30 \
  --scale-rule-name queue-rule \
  --scale-rule-type azure-queue \
  --scale-rule-metadata "queueName=tasks" "queueLength=5" \
  --scale-rule-auth "connection=queue-connection"

# View logs
az containerapp logs show --name myapi --resource-group myRG --follow`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1236,
			Title:       "Managed Identities & Zero Trust",
			Description: "Implement passwordless authentication with Azure Managed Identities and apply Zero Trust security principles across Azure resources.",
			Order:       36,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Managed Identities",
					Content: `Managed Identities eliminate the need for storing credentials in code, configuration files, or environment variables. Azure handles the entire credential lifecycle automatically.

**1. Types:**
*   **System-assigned:** Tied to a specific Azure resource (e.g., a VM, App Service, or AKS). Created and deleted with the resource. One identity per resource.
*   **User-assigned:** Standalone Azure resource. Can be assigned to multiple resources. Lifecycle is independent of the resources using it. Better for shared access patterns.

**2. How It Works:**
1. You enable a managed identity on an Azure resource.
2. Azure creates a service principal in Azure AD automatically.
3. Azure handles credential rotation (certificates are rotated every 46 days).
4. Your code requests tokens from the local IMDS (Instance Metadata Service) endpoint -- ` + "`" + `http://169.254.169.254/metadata/identity/oauth2/token` + "`" + `.
5. The Azure SDK handles this transparently with ` + "`" + `DefaultAzureCredential` + "`" + `.

**3. Common Use Cases:**
*   App Service accessing Azure SQL Database (no connection string with password!).
*   VM accessing Key Vault secrets.
*   AKS pods accessing Azure Storage or Cosmos DB.
*   Azure Functions accessing Service Bus.
*   CI/CD pipelines running in Azure accessing deployment targets.

**4. DefaultAzureCredential Chain:**
*   The Azure SDKs provide ` + "`" + `DefaultAzureCredential` + "`" + ` which tries multiple auth methods in order:
    1. Environment variables (service principal).
    2. Workload identity (Kubernetes).
    3. Managed identity (system or user-assigned).
    4. Azure CLI credentials (local development).
    5. Azure PowerShell.
*   Use this in code so it works in both local dev and production.`,
					CodeExamples: `# Enable system-assigned managed identity on a VM
az vm identity assign \
  --resource-group myRG \
  --name myVM

# Create a user-assigned managed identity
az identity create \
  --resource-group myRG \
  --name myAppIdentity

# Assign to an App Service
az webapp identity assign \
  --resource-group myRG \
  --name myWebApp \
  --identities /subscriptions/<sub>/resourceGroups/myRG/providers/Microsoft.ManagedIdentity/userAssignedIdentities/myAppIdentity

# Grant the identity access to Key Vault
az keyvault set-policy \
  --name myKeyVault \
  --object-id $(az identity show -g myRG -n myAppIdentity --query principalId -o tsv) \
  --secret-permissions get list

# Grant access to Storage (RBAC)
az role assignment create \
  --assignee $(az identity show -g myRG -n myAppIdentity --query principalId -o tsv) \
  --role "Storage Blob Data Reader" \
  --scope /subscriptions/<sub>/resourceGroups/myRG/providers/Microsoft.Storage/storageAccounts/mystorageaccount

# Azure SQL: No password needed with managed identity
# Connection string: "Server=myserver.database.windows.net;Database=mydb;Authentication=Active Directory Managed Identity"

# Python SDK using DefaultAzureCredential
# from azure.identity import DefaultAzureCredential
# from azure.keyvault.secrets import SecretClient
# credential = DefaultAzureCredential()
# client = SecretClient(vault_url="https://myvault.vault.azure.net", credential=credential)
# secret = client.get_secret("my-secret")`,
				},
				{
					Title: "Zero Trust on Azure",
					Content: `Zero Trust is a security model based on the principle "never trust, always verify." Azure provides comprehensive tools to implement Zero Trust across identity, devices, network, and data.

**1. Core Principles:**
*   **Verify explicitly:** Always authenticate and authorize based on all available data points (user identity, location, device health, service/workload, data classification).
*   **Use least privilege access:** Limit user access with Just-In-Time (JIT) and Just-Enough-Access (JEA). Use risk-based adaptive policies.
*   **Assume breach:** Minimize blast radius with segmentation. Encrypt all traffic. Use analytics for threat detection.

**2. Identity (Azure AD / Entra ID):**
*   **Conditional Access:** Enforce MFA based on risk level, location, device compliance, and application sensitivity.
*   **PIM (Privileged Identity Management):** Time-bound, approval-based activation of privileged roles. No standing admin access.
*   **Identity Protection:** ML-based detection of risky sign-ins and compromised accounts.
*   **Passwordless:** FIDO2 keys, Windows Hello, Microsoft Authenticator.

**3. Network:**
*   **Private endpoints:** All PaaS services accessed via private IP (no public internet).
*   **NSGs + ASGs:** Microsegmentation of network traffic.
*   **Azure Firewall:** Centralized network policy enforcement with TLS inspection.
*   **Azure Bastion:** Secure RDP/SSH access without public IPs on VMs.

**4. Data:**
*   **Encryption at rest:** All Azure services encrypt data by default (Microsoft-managed keys). Use customer-managed keys in Key Vault for sensitive data.
*   **Encryption in transit:** TLS 1.2+ enforced everywhere.
*   **Azure Information Protection:** Classify and label sensitive data. Policies follow the data.

**5. Devices:**
*   **Intune:** Device compliance policies (encryption, OS version, antivirus).
*   **Conditional Access device compliance:** Only compliant/hybrid-joined devices can access resources.`,
					CodeExamples: `# Conditional Access: Require MFA for Azure Management
# (Created via Azure Portal or Microsoft Graph API)
# Policy: Require MFA for Azure Management access from non-compliant devices
# Target: All users accessing Azure Resource Manager
# Conditions: Device not compliant
# Grant: Require MFA

# Enable PIM for a role
az rest --method POST \
  --uri "https://graph.microsoft.com/v1.0/roleManagement/directory/roleAssignmentScheduleRequests" \
  --body '{
    "action": "AdminAssign",
    "justification": "Temporary admin access for deployment",
    "roleDefinitionId": "<role-id>",
    "directoryScopeId": "/",
    "principalId": "<user-id>",
    "scheduleInfo": {
      "startDateTime": "2024-01-15T08:00:00Z",
      "expiration": {
        "type": "AfterDuration",
        "duration": "PT4H"
      }
    }
  }'

# Create private endpoint for Storage
az network private-endpoint create \
  --name myStoragePE \
  --resource-group myRG \
  --vnet-name myVNet \
  --subnet privateEndpoints \
  --private-connection-resource-id /subscriptions/<sub>/resourceGroups/myRG/providers/Microsoft.Storage/storageAccounts/mystorage \
  --group-id blob \
  --connection-name myConnection

# Disable public access on Storage
az storage account update \
  --name mystorage \
  --resource-group myRG \
  --public-network-access Disabled

# Azure Bastion (secure VM access without public IP)
az network bastion create \
  --name myBastion \
  --resource-group myRG \
  --vnet-name myVNet \
  --location eastus`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1237,
			Title:       "Azure Data & Analytics",
			Description: "Build data pipelines and analytics solutions with Azure Data Factory, Synapse Analytics, Databricks, and Stream Analytics.",
			Order:       37,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Data Factory",
					Content: `Azure Data Factory (ADF) is a cloud-based ETL/ELT service for orchestrating data movement and transformation at scale. It connects to 90+ data sources.

**1. Core Concepts:**
*   **Pipeline:** A logical grouping of activities that together perform a task (e.g., copy data, transform, load).
*   **Activity:** A single step in a pipeline. Types: Copy, Data Flow, Lookup, ForEach, If Condition, Web, Stored Procedure.
*   **Dataset:** A named view of data pointing to a data source (e.g., a specific table, blob, or file).
*   **Linked Service:** Connection string to a data store or compute (e.g., Azure SQL, Blob Storage, Databricks).
*   **Trigger:** What starts a pipeline. Types: Schedule, Tumbling Window, Event (blob created), Manual.

**2. Integration Runtime:**
*   **Azure IR:** Managed compute for data movement and transformation in Azure regions.
*   **Self-hosted IR:** Install on on-premises machines to access data behind firewalls.
*   **Azure-SSIS IR:** Run SSIS packages in the cloud.

**3. Data Flows:**
*   Visual, no-code data transformation built on Spark.
*   Operations: Source, Sink, Filter, Derived Column, Aggregate, Join, Pivot, Window, Sort.
*   Execute at scale with auto-provisioned Spark clusters.

**4. Monitoring:**
*   Built-in monitoring in Azure Portal shows pipeline runs, activity runs, and trigger runs.
*   Integration with Azure Monitor for alerts on failures.
*   Re-run failed pipelines or individual activities.`,
					CodeExamples: `# Create Data Factory
az datafactory create \
  --resource-group myRG \
  --factory-name myDataFactory \
  --location eastus

# Create a linked service (Azure SQL)
az datafactory linked-service create \
  --resource-group myRG \
  --factory-name myDataFactory \
  --linked-service-name AzureSqlLS \
  --properties '{
    "type": "AzureSqlDatabase",
    "typeProperties": {
      "connectionString": "Server=myserver.database.windows.net;Database=mydb;Authentication=Active Directory Managed Identity"
    }
  }'

# Create a pipeline (simplified JSON)
# {
#   "name": "CopyPipeline",
#   "properties": {
#     "activities": [{
#       "name": "CopyBlob2SQL",
#       "type": "Copy",
#       "inputs": [{"referenceName": "BlobDataset"}],
#       "outputs": [{"referenceName": "SqlDataset"}],
#       "typeProperties": {
#         "source": {"type": "BlobSource"},
#         "sink": {"type": "SqlSink", "writeBehavior": "upsert"}
#       }
#     }],
#     "trigger": {
#       "type": "ScheduleTrigger",
#       "recurrence": {
#         "frequency": "Hour",
#         "interval": 1
#       }
#     }
#   }
# }

# Trigger a pipeline run
az datafactory pipeline create-run \
  --resource-group myRG \
  --factory-name myDataFactory \
  --name CopyPipeline`,
				},
				{
					Title: "Azure Synapse Analytics",
					Content: `Azure Synapse Analytics is a unified analytics platform that combines data warehousing, big data processing, and data integration in a single workspace.

**1. Components:**
*   **Dedicated SQL pool (formerly SQL DW):** MPP (Massively Parallel Processing) data warehouse. Excellent for structured data analytics and BI.
*   **Serverless SQL pool:** Query data in-place on the data lake using T-SQL. No infrastructure to manage. Pay per query.
*   **Apache Spark pool:** Big data processing with PySpark, Scala, .NET for Spark. Auto-scaling and auto-pause.
*   **Data Integration:** Built-in ADF-compatible pipelines for ETL/ELT.
*   **Synapse Studio:** Unified web IDE for SQL, Spark, pipelines, and monitoring.

**2. Data Lake Integration:**
*   Synapse natively reads data from Azure Data Lake Storage Gen2.
*   **OPENROWSET():** Query CSV, Parquet, JSON files directly with serverless SQL.
*   **External tables:** Create SQL views over lake data for BI tools.
*   **Lake Database:** Define a database schema over files in the data lake (logical data warehouse).

**3. Architecture Patterns:**
*   **Modern Data Warehouse:** Raw data → Data Lake → Synapse (transform) → Power BI.
*   **Lakehouse:** Combine data lake flexibility with warehouse structure using Delta Lake format.
*   **Real-time analytics:** Event Hubs → Stream Analytics → Synapse → Dashboard.

**4. Performance:**
*   **Distribution:** Hash, Round-Robin, or Replicated table distribution for parallelism.
*   **Indexing:** Clustered columnstore (default, best for analytics), heap, clustered index.
*   **Result set caching:** Automatically caches query results for repeated queries.
*   **Materialized views:** Pre-computed aggregations that the query optimizer uses automatically.`,
					CodeExamples: `# Create Synapse workspace
az synapse workspace create \
  --resource-group myRG \
  --name mysynapse \
  --storage-account mystorageaccount \
  --file-system synapsefs \
  --sql-admin-login-user sqladmin \
  --sql-admin-login-password 'SecureP@ss123'

# Create a dedicated SQL pool (data warehouse)
az synapse sql pool create \
  --resource-group myRG \
  --workspace-name mysynapse \
  --name mydw \
  --performance-level DW100c

# Create a Spark pool
az synapse spark pool create \
  --resource-group myRG \
  --workspace-name mysynapse \
  --name mysparkpool \
  --node-count 3 \
  --node-size Medium \
  --spark-version 3.3

# Serverless SQL: Query Parquet files directly
# SELECT TOP 100 *
# FROM OPENROWSET(
#   BULK 'https://mystorageaccount.dfs.core.windows.net/data/sales/*.parquet',
#   FORMAT = 'PARQUET'
# ) AS sales
# WHERE year = 2024

# Create external table over lake data
# CREATE EXTERNAL TABLE sales_external (
#   id INT, product VARCHAR(100), amount DECIMAL(10,2), sale_date DATE
# )
# WITH (
#   LOCATION = 'sales/',
#   DATA_SOURCE = lake_data,
#   FILE_FORMAT = parquet_format
# )

# Pause/Resume dedicated pool (cost savings)
az synapse sql pool pause --resource-group myRG --workspace-name mysynapse --name mydw
az synapse sql pool resume --resource-group myRG --workspace-name mysynapse --name mydw`,
				},
				{
					Title: "Azure Databricks",
					Content: `Azure Databricks is a managed Apache Spark platform optimized for Azure. It provides collaborative notebooks, ML workflows, and Delta Lake for reliable data processing.

**1. Key Differentiators:**
*   **Delta Lake:** ACID transactions on data lakes. Time travel, schema enforcement, and efficient upserts on Parquet files.
*   **Photon engine:** C++ vectorized query engine. 2-8x faster than standard Spark for SQL workloads.
*   **Unity Catalog:** Centralized governance for data and AI across workspaces. Fine-grained access control.
*   **MLflow:** Built-in experiment tracking, model registry, and model serving.

**2. Workspace Concepts:**
*   **Workspace:** The Databricks environment. Contains notebooks, clusters, jobs, and data.
*   **Cluster:** A set of VMs running Spark. Types: All-Purpose (interactive, shared), Job (ephemeral, optimized for automated jobs).
*   **Notebook:** Interactive documents mixing code (Python, SQL, Scala, R), markdown, and visualizations.
*   **Job:** Scheduled or triggered execution of notebooks or JARs.

**3. Cluster Sizing:**
*   **Autoscaling:** Set min/max workers. Databricks adds/removes workers based on load.
*   **Spot instances:** Use Azure Spot VMs for workers to reduce cost by 60-80%.
*   **Auto-termination:** Clusters shut down after N minutes of inactivity (save cost).

**4. Common Patterns:**
*   **Medallion Architecture:** Bronze (raw) → Silver (cleaned) → Gold (business-level). Each layer in Delta format.
*   **Streaming:** Structured Streaming reads from Event Hubs/Kafka, processes incrementally, writes to Delta.
*   **Feature Store:** Store and manage ML features with point-in-time correctness.`,
					CodeExamples: `# Create Databricks workspace
az databricks workspace create \
  --resource-group myRG \
  --name myDatabricks \
  --location eastus \
  --sku premium

# Delta Lake operations (in a Databricks notebook)
# Bronze layer: Read raw data
# df = spark.read.format("json").load("/mnt/raw/events/")
# df.write.format("delta").mode("append").save("/mnt/bronze/events/")

# Silver layer: Clean and transform
# bronze_df = spark.read.format("delta").load("/mnt/bronze/events/")
# silver_df = bronze_df \
#   .filter("event_type IS NOT NULL") \
#   .withColumn("processed_at", current_timestamp()) \
#   .dropDuplicates(["event_id"])
# silver_df.write.format("delta").mode("merge").save("/mnt/silver/events/")

# Gold layer: Business aggregations
# SELECT product_category, DATE(event_date) as date,
#        COUNT(*) as total_events, SUM(revenue) as total_revenue
# FROM delta.` + "`" + `/mnt/silver/events/` + "`" + `
# GROUP BY product_category, DATE(event_date)

# Delta Lake time travel
# SELECT * FROM delta.` + "`" + `/mnt/silver/events/` + "`" + ` VERSION AS OF 5
# SELECT * FROM delta.` + "`" + `/mnt/silver/events/` + "`" + ` TIMESTAMP AS OF '2024-01-15'

# Optimize Delta table
# OPTIMIZE delta.` + "`" + `/mnt/silver/events/` + "`" + ` ZORDER BY (event_date, user_id)

# Databricks CLI: Create a job
# databricks jobs create --json '{
#   "name": "Daily ETL",
#   "tasks": [{
#     "task_key": "bronze_to_silver",
#     "notebook_task": {"notebook_path": "/Pipelines/bronze_silver"},
#     "new_cluster": {
#       "spark_version": "13.3.x-scala2.12",
#       "node_type_id": "Standard_DS3_v2",
#       "autoscale": {"min_workers": 2, "max_workers": 8}
#     }
#   }],
#   "schedule": {"quartz_cron_expression": "0 0 6 * * ?", "timezone_id": "UTC"}
# }'`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1238,
			Title:       "Infrastructure as Code on Azure",
			Description: "Deploy and manage Azure infrastructure using Bicep, Terraform, and GitHub Actions for automated, repeatable cloud deployments.",
			Order:       38,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Bicep Templates",
					Content: `Bicep is Azure's domain-specific language for deploying ARM resources. It compiles to ARM JSON but is dramatically more readable and maintainable.

**1. Bicep vs ARM JSON:**
*   Bicep is transpiled to ARM JSON. No runtime differences -- same API, same deployment engine.
*   ~60% less code than equivalent ARM JSON.
*   Type-safe with IntelliSense in VS Code.
*   Module system for reusable components.

**2. Modules:**
*   Break large deployments into reusable, parameterized modules.
*   ` + "`" + `module <name> './path/module.bicep' = { params: {...} }` + "`" + `.
*   Modules can be stored in Azure Container Registry (Bicep Module Registry) for sharing across teams.

**3. Conditionals and Loops:**
*   ` + "`" + `if` + "`" + ` for conditional deployment: ` + "`" + `resource ... = if (condition) { ... }` + "`" + `.
*   ` + "`" + `for` + "`" + ` for creating multiple resources: ` + "`" + `[for i in range(0, count): { ... }]` + "`" + `.
*   Combine for powerful patterns like deploying N instances with conditional features.

**4. What-If and Validation:**
*   ` + "`" + `az deployment group what-if` + "`" + ` -- Preview changes before deployment.
*   ` + "`" + `az bicep build` + "`" + ` -- Validate and compile to ARM JSON.
*   Linting: ` + "`" + `bicepconfig.json` + "`" + ` configures rules for best practices.

**5. Deployment Scopes:**
*   Resource group (default), subscription, management group, or tenant level.
*   Subscription-level: Create resource groups, policies, role assignments.
*   Management group: Apply policies across all subscriptions.`,
					CodeExamples: `// main.bicep -- deploy a complete web application stack
@description('Location for all resources')
param location string = resourceGroup().location

@allowed(['dev', 'staging', 'prod'])
param environment string

@description('Number of app instances')
param instanceCount int = environment == 'prod' ? 3 : 1

// Modules
module network './modules/network.bicep' = {
  name: 'networkDeploy'
  params: {
    location: location
    environment: environment
  }
}

module appService './modules/appservice.bicep' = {
  name: 'appServiceDeploy'
  params: {
    location: location
    subnetId: network.outputs.appSubnetId
    instanceCount: instanceCount
    sku: environment == 'prod' ? 'P2v3' : 'B1'
  }
}

// Conditional: only deploy Redis in prod
module redis './modules/redis.bicep' = if (environment == 'prod') {
  name: 'redisDeploy'
  params: {
    location: location
    subnetId: network.outputs.dataSubnetId
  }
}

// Loop: deploy multiple storage accounts
resource storageAccounts 'Microsoft.Storage/storageAccounts@2023-01-01' = [for i in range(0, 2): {
  name: 'storage${environment}${i}'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
}]

output appUrl string = appService.outputs.defaultHostname

// Deploy
// az deployment group create -g myRG --template-file main.bicep --parameters environment=prod
// az deployment group what-if -g myRG --template-file main.bicep --parameters environment=prod`,
				},
				{
					Title: "Terraform on Azure",
					Content: `Terraform is the industry-standard multi-cloud IaC tool. The AzureRM provider supports 1000+ Azure resource types and is actively maintained by HashiCorp and Microsoft.

**1. Azure Provider Setup:**
*   Authentication: Service principal, managed identity, Azure CLI, or OIDC (recommended for CI/CD).
*   State backend: Always use remote state in production. Azure Blob Storage with state locking (via lease).
*   ` + "`" + `features {}` + "`" + ` block is required in the provider configuration.

**2. Best Practices:**
*   **Modules:** Create reusable modules for common patterns (e.g., VNet+NSG, AKS cluster, App Service).
*   **Workspaces or directories:** Separate environments (dev/staging/prod) by workspace or directory.
*   **State locking:** Use Azure Blob Storage backend with locking to prevent concurrent modifications.
*   **Plan before apply:** Always run ` + "`" + `terraform plan` + "`" + ` and review changes before ` + "`" + `terraform apply` + "`" + `.
*   **Drift detection:** Schedule ` + "`" + `terraform plan` + "`" + ` in CI to detect infrastructure drift.

**3. AzureRM Provider Features:**
*   ` + "`" + `prevent_deletion_if_contains_resources` + "`" + ` -- Prevent accidental resource group deletion.
*   ` + "`" + `skip_provider_registration` + "`" + ` -- Useful in restricted environments.
*   Import existing resources: ` + "`" + `terraform import azurerm_resource_group.example /subscriptions/.../resourceGroups/myRG` + "`" + `.

**4. Terraform vs Bicep:**
*   **Terraform:** Multi-cloud, huge ecosystem, HCL language, state management complexity.
*   **Bicep:** Azure-only, no state file (ARM tracks state), simpler for pure Azure shops.
*   Choose Terraform for multi-cloud or if team already knows it. Choose Bicep for Azure-only simplicity.`,
					CodeExamples: `# backend.tf -- Remote state in Azure Blob Storage
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }
  backend "azurerm" {
    resource_group_name  = "terraform-state"
    storage_account_name = "tfstate12345"
    container_name       = "tfstate"
    key                  = "prod.terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
  }
}

# main.tf -- AKS cluster with managed identity
resource "azurerm_resource_group" "main" {
  name     = "rg-myapp-prod"
  location = "eastus"
}

resource "azurerm_kubernetes_cluster" "main" {
  name                = "aks-myapp-prod"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "myapp"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D4s_v3"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "calico"
  }
}

# Deploy
# terraform init
# terraform plan -out=tfplan
# terraform apply tfplan`,
				},
				{
					Title: "GitHub Actions for Azure",
					Content: `GitHub Actions provides powerful CI/CD workflows that integrate natively with Azure through official actions and OIDC authentication.

**1. Authentication:**
*   **OIDC (Federated Credentials):** Recommended. No secrets stored. GitHub exchanges a token with Azure AD.
*   **Service Principal:** Store client ID, secret, and tenant ID in GitHub Secrets.
*   ` + "`" + `azure/login@v1` + "`" + ` action handles authentication.

**2. Common Workflows:**
*   **Build and deploy to App Service:** Build app, push to ACR, deploy to App Service.
*   **Infrastructure deployment:** Run Terraform or Bicep to create/update infrastructure.
*   **AKS deployment:** Build image, push to ACR, deploy Kubernetes manifests with ` + "`" + `kubectl` + "`" + `.
*   **Database migration:** Run schema migrations as part of deployment.

**3. Best Practices:**
*   Use environments for approval gates (prod deployments require manual approval).
*   Pin action versions to full SHA for security.
*   Use reusable workflows (` + "`" + `workflow_call` + "`" + `) for shared CI/CD patterns.
*   Separate infrastructure and application pipelines.

**4. Key Azure Actions:**
*   ` + "`" + `azure/login` + "`" + ` -- Authenticate to Azure.
*   ` + "`" + `azure/docker-login` + "`" + ` -- Authenticate to ACR.
*   ` + "`" + `azure/webapps-deploy` + "`" + ` -- Deploy to App Service.
*   ` + "`" + `azure/aks-set-context` + "`" + ` -- Set Kubernetes context for AKS.
*   ` + "`" + `azure/arm-deploy` + "`" + ` -- Deploy Bicep/ARM templates.`,
					CodeExamples: `# .github/workflows/deploy.yml
name: Build and Deploy to AKS

on:
  push:
    branches: [main]

permissions:
  id-token: write  # Required for OIDC
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Azure Login (OIDC)
      uses: azure/login@v1
      with:
        client-id: ${{ secrets.AZURE_CLIENT_ID }}
        tenant-id: ${{ secrets.AZURE_TENANT_ID }}
        subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

    - name: Build and push to ACR
      run: |
        az acr login --name myregistry
        docker build -t myregistry.azurecr.io/myapp:${{ github.sha }} .
        docker push myregistry.azurecr.io/myapp:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production  # Requires approval
    steps:
    - uses: actions/checkout@v4

    - name: Azure Login
      uses: azure/login@v1
      with:
        client-id: ${{ secrets.AZURE_CLIENT_ID }}
        tenant-id: ${{ secrets.AZURE_TENANT_ID }}
        subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

    - name: Set AKS context
      uses: azure/aks-set-context@v3
      with:
        resource-group: myRG
        cluster-name: myAKS

    - name: Deploy to AKS
      run: |
        kubectl set image deployment/myapp \
          myapp=myregistry.azurecr.io/myapp:${{ github.sha }} \
          --namespace production
        kubectl rollout status deployment/myapp -n production`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
