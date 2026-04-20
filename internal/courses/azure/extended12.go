package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1257,
			Title:       "Azure Architecture Patterns",
			Description: "Learn Azure architecture design patterns including microservices, event-driven architectures, multi-region deployments, and cost optimization.",
			Order:       57,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Architecture Design Patterns",
					Content: `Azure architecture patterns provide proven approaches for building reliable, scalable, and secure cloud applications.

**Microservices on Azure:**
` + "```" + `
Microservices hosting options:

  Azure Kubernetes Service (AKS):
    Best for: Complex microservices, team expertise in K8s
    Features: Full orchestration, service mesh, auto-scaling
    Considerations: Operational overhead, cluster management
  
  Azure Container Apps:
    Best for: Simpler microservices, Dapr integration
    Features: Serverless containers, built-in scaling, Dapr
    Considerations: Less control than AKS, newer service
  
  Azure App Service:
    Best for: Web APIs, simple services
    Features: Managed platform, deployment slots, auto-scale
    Considerations: Less flexibility, cost per plan
  
  Azure Functions:
    Best for: Event-driven, small services
    Features: Serverless, pay-per-execution
    Considerations: Cold start, duration limits

Microservices communication:
  Synchronous:
    - HTTP/REST (most common)
    - gRPC (high performance)
    - Azure API Management (gateway)
  
  Asynchronous:
    - Azure Service Bus (enterprise messaging)
    - Azure Event Hub (streaming)
    - Azure Event Grid (event routing)
    - Azure Queue Storage (simple queues)

API Management:
  az apim create \
    --resource-group myRG --name myapim \
    --publisher-name "MyOrg" --publisher-email admin@example.com \
    --sku-name Developer
  
  Features:
  - API gateway (rate limiting, caching, auth)
  - Developer portal
  - API versioning
  - OAuth 2.0 / JWT validation
  - Request/response transformation
  - Analytics and monitoring

Service communication patterns:
  API Gateway:
    Client → API Management → Service A
                             → Service B
  
  Service Mesh (Istio on AKS):
    Service A → Envoy → Envoy → Service B
    (mTLS, retries, circuit breaking)
  
  Event-driven:
    Service A → Event Hub/Service Bus → Service B
                                      → Service C
` + "```" + `

**Event-Driven Architecture:**
` + "```" + `
Azure Event Grid:
  - Event routing service
  - React to Azure resource events
  - Custom topics for your events
  - At-least-once delivery
  - Filtering and fan-out
  
  Event sources:
    Azure services (Blob created, VM changed, etc.)
    Custom topics (your applications)
  
  Event handlers:
    Azure Functions, Logic Apps, Event Hub
    Storage Queue, Webhook, Service Bus

Azure Service Bus:
  - Enterprise message broker
  - Guaranteed delivery (at-least-once, at-most-once)
  - FIFO with sessions
  - Dead-letter queue
  - Transactions
  - Topics and subscriptions (pub/sub)
  
  az servicebus namespace create \
    --resource-group myRG --name mySBns \
    --sku Premium --location eastus
  
  az servicebus queue create \
    --resource-group myRG --namespace-name mySBns \
    --name orders --max-size 5120 \
    --default-message-time-to-live P14D \
    --enable-dead-lettering-on-message-expiration true

Azure Event Hub:
  - Big data streaming platform
  - Millions of events per second
  - Apache Kafka compatible
  - Capture to storage/data lake
  - Partition-based (ordered within partition)
  
  az eventhubs namespace create \
    --resource-group myRG --name myEHns \
    --sku Standard --location eastus
  
  az eventhubs eventhub create \
    --resource-group myRG --namespace-name myEHns \
    --name events --partition-count 4 \
    --message-retention 7

Choosing messaging service:
  Simple queue:           Azure Queue Storage
  Enterprise messaging:   Azure Service Bus
  Event routing:          Azure Event Grid
  Big data streaming:     Azure Event Hub
  IoT telemetry:          Azure IoT Hub
` + "```" + `

**Multi-Region Architecture:**
` + "```" + `
Active-Passive:
  Primary region:    All traffic, read/write
  Secondary region:  Standby, failover target
  
  Components:
  - Azure Traffic Manager (DNS failover)
  - Geo-replication for databases
  - GRS/GZRS storage
  - Pre-deployed (but scaled down) compute
  
  RTO: Minutes to hours
  RPO: Depends on replication lag

Active-Active:
  Both regions:  Handle traffic simultaneously
  
  Components:
  - Azure Front Door (global load balancer)
  - Cosmos DB multi-region writes
  - Azure SQL failover groups
  - Zone-redundant services
  
  RTO: Near zero
  RPO: Near zero (with sync replication)

Azure Front Door:
  - Global HTTP load balancer
  - SSL offloading
  - WAF (Web Application Firewall)
  - Caching at edge
  - URL-based routing
  - Session affinity
  
  az afd profile create \
    --resource-group myRG --profile-name myFrontDoor \
    --sku Premium_AzureFrontDoor
  
  az afd endpoint create \
    --resource-group myRG --profile-name myFrontDoor \
    --endpoint-name myendpoint

Azure Traffic Manager:
  - DNS-based load balancing
  - Multiple routing methods
  - Health monitoring
  
  Routing methods:
    Priority:     Active-passive failover
    Weighted:     Distribute by weight %
    Performance:  Closest region (lowest latency)
    Geographic:   Route by user location
    MultiValue:   Return multiple healthy endpoints
    Subnet:       Map IP ranges to endpoints

Disaster Recovery pattern:
  Normal operation:
    Users → Front Door → Region 1 (primary)
                        → Region 2 (secondary, read-only)
  
  Failover:
    Users → Front Door → Region 2 (promoted to primary)
    Region 1 down/recovering
  
  Data sync:
    Cosmos DB: multi-master auto-sync
    SQL: async geo-replication (failover groups)
    Storage: RA-GRS (async, read from secondary)
` + "```" + ``,
					CodeExamples: `# Azure architecture patterns

# 1. Multi-region health checker
#!/bin/bash
echo "=== Multi-Region Health Check ==="

REGIONS=("eastus" "westus" "northeurope")
ENDPOINTS=(
    "https://myapp-eastus.azurewebsites.net/health"
    "https://myapp-westus.azurewebsites.net/health"
    "https://myapp-northeurope.azurewebsites.net/health"
)

for i in "${!REGIONS[@]}"; do
    REGION="${REGIONS[$i]}"
    ENDPOINT="${ENDPOINTS[$i]}"
    
    START=$(date +%s%N)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$ENDPOINT" 2>/dev/null)
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))
    
    if [ "$HTTP_CODE" = "200" ]; then
        STATUS="HEALTHY"
    else
        STATUS="UNHEALTHY ($HTTP_CODE)"
    fi
    
    printf "  %-15s  %s  Latency: %dms\n" "$REGION" "$STATUS" "$LATENCY"
done

# Traffic Manager status
echo ""
echo "--- Traffic Manager ---"
for profile in $(az network traffic-manager profile list \
    --query "[].name" -o tsv 2>/dev/null); do
    
    RG=$(az network traffic-manager profile list \
        --query "[?name=='$profile'].resourceGroup" -o tsv | head -1)
    
    echo "Profile: $profile"
    az network traffic-manager endpoint list \
        --profile-name "$profile" -g "$RG" \
        --query "[].{name:name, status:endpointStatus, target:target}" \
        -o table 2>/dev/null
done

# Front Door status
echo ""
echo "--- Front Door ---"
for profile in $(az afd profile list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az afd profile list --query "[?name=='$profile'].resourceGroup" -o tsv | head -1)
    echo "Profile: $profile"
    
    az afd endpoint list --profile-name "$profile" -g "$RG" \
        --query "[].{name:name, status:enabledState, hostname:hostName}" \
        -o table 2>/dev/null
done

# 2. Event messaging status
#!/bin/bash
echo "=== Azure Messaging Status ==="

# Service Bus namespaces
echo "--- Service Bus ---"
for ns in $(az servicebus namespace list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az servicebus namespace list --query "[?name=='$ns'].resourceGroup" -o tsv | head -1)
    echo "Namespace: $ns (RG: $RG)"
    
    # Queues
    az servicebus queue list --namespace-name "$ns" -g "$RG" \
        --query "[].{name:name, active:countDetails.activeMessageCount, dead:countDetails.deadLetterMessageCount}" \
        -o table 2>/dev/null
    
    # Topics
    az servicebus topic list --namespace-name "$ns" -g "$RG" \
        --query "[].{name:name, subscriptions:subscriptionCount}" \
        -o table 2>/dev/null
done

# Event Hubs
echo ""
echo "--- Event Hubs ---"
for ns in $(az eventhubs namespace list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az eventhubs namespace list --query "[?name=='$ns'].resourceGroup" -o tsv | head -1)
    echo "Namespace: $ns (RG: $RG)"
    
    az eventhubs eventhub list --namespace-name "$ns" -g "$RG" \
        --query "[].{name:name, partitions:partitionCount, retention:messageRetentionInDays}" \
        -o table 2>/dev/null
done

# 3. Cost optimization report
#!/bin/bash
echo "=== Azure Cost Optimization ==="

# Unattached disks
echo "--- Unattached Managed Disks ---"
az disk list --query "[?managedBy==null].{
    name:name, rg:resourceGroup, size:diskSizeGb, sku:sku.name
}" -o table

# Stopped but not deallocated VMs
echo ""
echo "--- VMs Stopped (still billing) ---"
for rg in $(az group list --query "[].name" -o tsv); do
    az vm list -g "$rg" -d --query "[?powerState=='VM stopped'].{
        name:name, size:hardwareProfile.vmSize
    }" -o table 2>/dev/null
done

# Underutilized resources (from Advisor)
echo ""
echo "--- Azure Advisor Cost Recommendations ---"
az advisor recommendation list \
    --query "[?category=='Cost'].{
        impact:impact, problem:shortDescription.problem,
        solution:shortDescription.solution
    }" -o table 2>/dev/null | head -20`,
				},
				{
					Title: "Azure Cost Management and Governance",
					Content: `Cost management and governance ensure Azure spending stays within budget and resources comply with organizational policies.

**Azure Cost Management:**
` + "```" + `
Cost analysis:
  # View current cost
  az consumption usage list \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --query "[].{resource:instanceName, cost:pretaxCost, currency:currency}" \
    -o table

Budget creation:
  az consumption budget create \
    --budget-name monthly-budget \
    --amount 5000 \
    --category cost \
    --time-grain monthly \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --resource-group myRG

Notifications:
  az consumption budget create \
    --budget-name prod-budget \
    --amount 10000 \
    --time-grain monthly \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --notifications '{
      "80percent": {
        "enabled": true,
        "operator": "GreaterThan",
        "threshold": 80,
        "contactEmails": ["admin@example.com"]
      },
      "100percent": {
        "enabled": true,
        "operator": "GreaterThan",
        "threshold": 100,
        "contactEmails": ["admin@example.com", "manager@example.com"]
      }
    }'

Cost saving strategies:
  1. Reserved Instances (1/3 year):
     - Up to 72% savings for consistent workloads
     - VMs, SQL, Cosmos DB, App Service
     - Exchangeable (not refundable for 1-year)
  
  2. Spot VMs:
     - Up to 90% savings
     - For fault-tolerant, batch workloads
     - Can be evicted with 30s notice
  
  3. Azure Hybrid Benefit:
     - Use existing Windows/SQL licenses
     - Up to 85% savings on Windows VMs
  
  4. Auto-scaling:
     - Scale down during off-hours
     - Start/stop development VMs
  
  5. Right-sizing:
     - Use Azure Advisor recommendations
     - Monitor actual CPU/memory usage
     - Downgrade oversized VMs
  
  6. Storage optimization:
     - Lifecycle management (tier down old data)
     - Delete unattached disks
     - Use appropriate redundancy (LRS vs GRS)
` + "```" + `

**Azure Policy and Governance:**
` + "```" + `
Azure Policy enforces organizational standards.

Built-in policies:
  - Allowed locations
  - Allowed VM SKUs
  - Require tag on resources
  - Inherit tag from resource group
  - Enforce HTTPS on storage
  - Audit VMs without managed disks
  - Not allowed resource types

Assign policy:
  # Require Environment tag
  az policy assignment create \
    --name "require-env-tag" \
    --display-name "Require Environment Tag" \
    --policy "/providers/Microsoft.Authorization/policyDefinitions/871b6d14-10aa-478d-b466-98cf0fc0b90d" \
    --scope "/subscriptions/<sub-id>" \
    --params '{"tagName": {"value": "Environment"}}'
  
  # Restrict VM sizes
  az policy assignment create \
    --name "allowed-vm-sizes" \
    --policy "/providers/Microsoft.Authorization/policyDefinitions/cccc23c7-8427-4f53-ad12-b6a63eb452b3" \
    --scope "/subscriptions/<sub-id>/resourceGroups/myRG" \
    --params '{"listOfAllowedSKUs": {"value": ["Standard_D2s_v5", "Standard_D4s_v5"]}}'
  
  # Restrict locations
  az policy assignment create \
    --name "allowed-locations" \
    --policy "/providers/Microsoft.Authorization/policyDefinitions/e56962a6-4747-49cd-b67b-bf8b01975c4c" \
    --scope "/subscriptions/<sub-id>" \
    --params '{"listOfAllowedLocations": {"value": ["eastus", "westus", "northeurope"]}}'

Custom policy:
  {
    "mode": "All",
    "policyRule": {
      "if": {
        "allOf": [
          {
            "field": "type",
            "equals": "Microsoft.Storage/storageAccounts"
          },
          {
            "field": "Microsoft.Storage/storageAccounts/allowBlobPublicAccess",
            "notEquals": false
          }
        ]
      },
      "then": {
        "effect": "deny"
      }
    },
    "parameters": {}
  }

Policy initiatives (groups):
  - CIS Azure Benchmark
  - NIST SP 800-53
  - PCI-DSS
  - ISO 27001
  - Azure Security Benchmark

Management Groups:
  Root Management Group
  ├── Production
  │   ├── Subscription: Prod-App1
  │   └── Subscription: Prod-App2
  ├── Non-Production
  │   ├── Subscription: Dev
  │   └── Subscription: Staging
  └── Sandbox
      └── Subscription: Innovation
  
  # Policies applied at management group level
  # cascade down to all subscriptions and resources

Azure Blueprints (being replaced by Deployment Stacks):
  - Package of role assignments, policies, ARM templates
  - Versioned, assigned to subscriptions
  - Lockable (prevent modification)

Resource tagging strategy:
  Required tags:
    Environment:    dev/staging/prod
    CostCenter:     department code
    Owner:          team or person
    Application:    app name
    ManagedBy:      terraform/bicep/manual
  
  Optional:
    Expiry:         auto-cleanup date
    Confidentiality: public/internal/confidential
    Compliance:     pci/hipaa/none
` + "```" + ``,
					CodeExamples: `# Azure governance scripts

# 1. Resource tagging compliance
#!/bin/bash
echo "=== Tagging Compliance Report ==="
REQUIRED_TAGS=("Environment" "Owner" "CostCenter")

TOTAL=0
COMPLIANT=0
NON_COMPLIANT=0

for rg in $(az group list --query "[].name" -o tsv); do
    RESOURCES=$(az resource list -g "$rg" --query "[].{id:id, name:name, type:type, tags:tags}" -o json 2>/dev/null)
    
    echo "$RESOURCES" | jq -c '.[]' | while read -r resource; do
        name=$(echo "$resource" | jq -r '.name')
        type=$(echo "$resource" | jq -r '.type')
        ((TOTAL++))
        
        MISSING=""
        for tag in "${REQUIRED_TAGS[@]}"; do
            HAS_TAG=$(echo "$resource" | jq -r ".tags.\"$tag\" // empty")
            if [ -z "$HAS_TAG" ]; then
                MISSING="$MISSING $tag"
            fi
        done
        
        if [ -n "$MISSING" ]; then
            ((NON_COMPLIANT++))
            echo "  NON-COMPLIANT: $name ($type) - Missing:$MISSING"
        else
            ((COMPLIANT++))
        fi
    done
done

echo ""
echo "Total: $TOTAL, Compliant: $COMPLIANT, Non-compliant: $NON_COMPLIANT"

# 2. Policy compliance checker
#!/bin/bash
echo "=== Azure Policy Compliance ==="

# Overall compliance
az policy state summarize --query "{
    totalPolicies:results.policyAssignments | length(@),
    nonCompliant:results.nonCompliantResources,
    compliant:results.resourceDetails[?complianceState=='Compliant'] | length(@)
}" -o json 2>/dev/null | jq .

echo ""
echo "--- Non-Compliant Policy Assignments ---"
az policy state summarize \
    --query "policyAssignments[?results.nonCompliantResources>0].{
        name:policyAssignmentId | split('/') | [-1],
        nonCompliant:results.nonCompliantResources,
        total:results.totalResources
    }" -o table 2>/dev/null

# 3. Resource cleanup script
#!/bin/bash
echo "=== Resource Cleanup Candidates ==="

# Empty resource groups
echo "--- Empty Resource Groups ---"
for rg in $(az group list --query "[].name" -o tsv); do
    COUNT=$(az resource list -g "$rg" --query "length(@)" -o tsv 2>/dev/null)
    if [ "$COUNT" = "0" ]; then
        CREATED=$(az group show -n "$rg" --query "tags.CreatedDate // 'unknown'" -o tsv 2>/dev/null)
        echo "  $rg (created: $CREATED)"
    fi
done

# Unattached public IPs
echo ""
echo "--- Unattached Public IPs ---"
az network public-ip list \
    --query "[?ipConfiguration==null].{name:name, rg:resourceGroup, ip:ipAddress}" \
    -o table

# Unattached NICs
echo ""
echo "--- Unattached NICs ---"
az network nic list \
    --query "[?virtualMachine==null].{name:name, rg:resourceGroup}" \
    -o table

# Old snapshots (>90 days)
echo ""
echo "--- Old Snapshots (>90 days) ---"
CUTOFF=$(date -d '90 days ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-90d +%Y-%m-%dT%H:%M:%S)
az snapshot list --query "[?timeCreated<'$CUTOFF'].{
    name:name, rg:resourceGroup, size:diskSizeGb,
    created:timeCreated
}" -o table 2>/dev/null`,
				},
			},
		},
	})
}
