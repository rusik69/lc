package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1269,
			Title:       "Azure Container and Serverless Services",
			Description: "Master Azure container services including AKS advanced topics, Container Apps, Container Instances, and serverless computing with Azure Functions and Logic Apps.",
			Order:       69,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Kubernetes Service Advanced Topics",
					Content: `Azure Kubernetes Service provides advanced features for production-grade container orchestration.

**AKS Advanced Configuration:**
` + "```" + `
Node pools:
  System node pool:
    - Runs system pods (CoreDNS, metrics-server)
    - At least 1 required
    - CriticalAddonsOnly taint by default
  
  User node pool:
    - Runs application workloads
    - Multiple pools with different VM sizes
    - Can be added/removed dynamically
  
  # Create cluster with system pool
  az aks create -n myAKS -g myRG \
    --node-count 3 --node-vm-size Standard_D4s_v5 \
    --network-plugin azure --network-policy calico \
    --enable-managed-identity \
    --enable-workload-identity \
    --enable-oidc-issuer \
    --generate-ssh-keys
  
  # Add GPU node pool
  az aks nodepool add -n gpu --cluster-name myAKS -g myRG \
    --node-count 2 --node-vm-size Standard_NC6s_v3 \
    --node-taints "sku=gpu:NoSchedule" \
    --labels workload=gpu \
    --enable-cluster-autoscaler \
    --min-count 0 --max-count 5
  
  # Add spot node pool (cost savings)
  az aks nodepool add -n spot --cluster-name myAKS -g myRG \
    --priority Spot --eviction-policy Delete \
    --spot-max-price -1 \
    --node-count 3 --node-vm-size Standard_D4s_v5 \
    --enable-cluster-autoscaler \
    --min-count 0 --max-count 10

Networking:
  Azure CNI:
    - Pod gets VNet IP directly
    - NSG/UDR works at pod level
    - Better for large clusters
    - IP planning required
  
  Azure CNI Overlay:
    - Private pod CIDR (not from VNet)
    - Saves VNet IP addresses
    - Still integrates with VNet
  
  Kubenet:
    - UDR-based routing
    - Fewer VNet IPs consumed
    - Limited NSG support for pods

  az aks create -n myAKS -g myRG \
    --network-plugin azure \
    --vnet-subnet-id "/subscriptions/.../subnets/aks-subnet" \
    --service-cidr 10.0.0.0/16 \
    --dns-service-ip 10.0.0.10

Ingress:
  # Application Gateway Ingress Controller (AGIC)
  az aks enable-addons -a ingress-appgw \
    -n myAKS -g myRG \
    --appgw-name myAppGW --appgw-subnet-cidr "10.2.0.0/16"
  
  # NGINX Ingress (via Helm)
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
  helm install nginx-ingress ingress-nginx/ingress-nginx \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-health-probe-request-path"=/healthz

Monitoring:
  # Container Insights
  az aks enable-addons -a monitoring \
    -n myAKS -g myRG \
    --workspace-resource-id "/subscriptions/.../workspaces/myLA"
  
  # Prometheus + Grafana (managed)
  az aks update -n myAKS -g myRG \
    --enable-azure-monitor-metrics
  
  # Azure Managed Grafana
  az grafana create -n myGrafana -g myRG

Security:
  # Azure AD integration (RBAC)
  az aks update -n myAKS -g myRG \
    --enable-azure-rbac --enable-aad
  
  # Azure Policy for AKS
  az aks enable-addons -a azure-policy -n myAKS -g myRG
  
  # Defender for Containers
  az security pricing create -n Containers --tier Standard
  
  # Private cluster
  az aks create -n myPrivateAKS -g myRG \
    --enable-private-cluster \
    --private-dns-zone system
  
  # Workload Identity (pod-level Azure identity)
  az aks update -n myAKS -g myRG \
    --enable-workload-identity --enable-oidc-issuer

Upgrades:
  # Check available versions
  az aks get-versions -l eastus -o table
  
  # Upgrade cluster
  az aks upgrade -n myAKS -g myRG --kubernetes-version 1.29.0
  
  # Node image upgrade
  az aks nodepool upgrade \
    --cluster-name myAKS -g myRG -n nodepool1 \
    --node-image-only
` + "```" + ``,
					CodeExamples: `# AKS management scripts

# 1. AKS cluster health check
#!/bin/bash
echo "=== AKS Health Check ==="

for cluster in $(az aks list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az aks list --query "[?name=='$cluster'].resourceGroup" -o tsv | head -1)
    echo "Cluster: $cluster ($RG)"
    
    # Cluster info
    az aks show -n "$cluster" -g "$RG" \
        --query "{
            version:kubernetesVersion,
            state:provisioningState,
            powerState:powerState.code,
            fqdn:fqdn,
            networkPlugin:networkProfile.networkPlugin,
            networkPolicy:networkProfile.networkPolicy
        }" -o json 2>/dev/null | jq .
    
    # Node pools
    echo "  Node Pools:"
    az aks nodepool list --cluster-name "$cluster" -g "$RG" \
        --query "[].{
            name:name, count:count, vmSize:vmSize,
            mode:mode, version:currentOrchestratorVersion,
            state:provisioningState, autoscale:enableAutoScaling
        }" -o table 2>/dev/null
    
    # Available upgrades
    UPGRADES=$(az aks get-upgrades -n "$cluster" -g "$RG" \
        --query "controlPlaneProfile.upgrades[].kubernetesVersion" -o tsv 2>/dev/null)
    if [ -n "$UPGRADES" ]; then
        echo "  Available upgrades: $UPGRADES"
    fi
    echo ""
done

# 2. AKS addon status
#!/bin/bash
echo "=== AKS Addons Status ==="

CLUSTER="${1}"
RG="${2}"

if [ -z "$CLUSTER" ] || [ -z "$RG" ]; then
    echo "Usage: $0 <cluster-name> <resource-group>"
    exit 1
fi

az aks show -n "$CLUSTER" -g "$RG" \
    --query "addonProfiles" -o json 2>/dev/null | \
    jq 'to_entries[] | {addon: .key, enabled: .value.enabled}'

# RBAC status
echo ""
echo "--- RBAC Configuration ---"
az aks show -n "$CLUSTER" -g "$RG" \
    --query "{
        rbacEnabled:enableRbac,
        azureRbac:aadProfile.enableAzureRbac,
        aadEnabled:aadProfile.managed,
        localAccounts:disableLocalAccounts
    }" -o json 2>/dev/null | jq .

# Network profile
echo ""
echo "--- Network Profile ---"
az aks show -n "$CLUSTER" -g "$RG" \
    --query "networkProfile" -o json 2>/dev/null | jq '{
        networkPlugin,
        networkPolicy,
        serviceCidr,
        dnsServiceIP,
        loadBalancerSku
    }'

# 3. AKS cost analysis
#!/bin/bash
echo "=== AKS Cost Analysis ==="

for cluster in $(az aks list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az aks list --query "[?name=='$cluster'].resourceGroup" -o tsv | head -1)
    echo "Cluster: $cluster ($RG)"
    
    TOTAL_NODES=0
    TOTAL_VCPU=0
    
    while IFS=$'\t' read -r name count size mode; do
        echo "  Pool: $name ($mode) - $count x $size"
        TOTAL_NODES=$((TOTAL_NODES + count))
    done < <(az aks nodepool list --cluster-name "$cluster" -g "$RG" \
        --query "[].{name:name, count:count, size:vmSize, mode:mode}" \
        -o tsv 2>/dev/null)
    
    echo "  Total nodes: $TOTAL_NODES"
    echo ""
done`,
				},
				{
					Title: "Azure Serverless Computing",
					Content: `Azure serverless services enable event-driven computing without managing infrastructure.

**Azure Container Apps:**
` + "```" + `
Fully managed serverless container platform.

Features:
  - Serverless containers (scale to zero)
  - Dapr integration (built-in)
  - KEDA-based autoscaling
  - Revision management
  - Traffic splitting
  - Built-in ingress
  - VNet integration

Create Container App:
  # Environment
  az containerapp env create \
    --name myEnv -g myRG --location eastus \
    --logs-workspace-id "$LA_WORKSPACE_ID" \
    --logs-workspace-key "$LA_KEY"
  
  # App from container image
  az containerapp create \
    --name myapp --environment myEnv -g myRG \
    --image mcr.microsoft.com/azuredocs/containerapps-helloworld \
    --target-port 80 --ingress external \
    --min-replicas 0 --max-replicas 10 \
    --cpu 0.5 --memory 1.0Gi
  
  # App with secrets and env vars
  az containerapp create \
    --name api --environment myEnv -g myRG \
    --image myacr.azurecr.io/api:latest \
    --registry-server myacr.azurecr.io \
    --target-port 8080 --ingress external \
    --secrets "db-conn=Server=..." \
    --env-vars "DB_CONNECTION=secretref:db-conn" "ENVIRONMENT=production"

Scaling rules:
  az containerapp update --name myapp -g myRG \
    --scale-rule-name http-rule \
    --scale-rule-type http \
    --scale-rule-http-concurrency 50
  
  # Custom scaling (KEDA)
  az containerapp update --name myapp -g myRG \
    --scale-rule-name queue-rule \
    --scale-rule-type azure-queue \
    --scale-rule-metadata "queueName=orders" "queueLength=10" \
    --scale-rule-auth "connection=queue-connection-secret"

Revisions and traffic:
  # New revision
  az containerapp update --name myapp -g myRG \
    --image myacr.azurecr.io/api:v2
  
  # Traffic split
  az containerapp ingress traffic set \
    --name myapp -g myRG \
    --revision-weight myapp--v1=80 myapp--v2=20

Dapr:
  # Enable Dapr sidecar
  az containerapp update --name myapp -g myRG \
    --enable-dapr true \
    --dapr-app-id myapp \
    --dapr-app-port 8080 \
    --dapr-app-protocol http
  
  # Dapr components (state store, pub/sub, bindings)
  az containerapp env dapr-component set \
    --name myEnv -g myRG \
    --dapr-component-name statestore \
    --yaml statestore.yaml

Jobs:
  # One-time or scheduled container jobs
  az containerapp job create \
    --name myjob --environment myEnv -g myRG \
    --image myacr.azurecr.io/batch:latest \
    --trigger-type Schedule \
    --cron-expression "0 */6 * * *" \
    --cpu 1.0 --memory 2.0Gi \
    --replica-timeout 1800
` + "```" + `

**Azure Functions:**
` + "```" + `
Event-driven serverless compute.

Hosting plans:
  Consumption:
    - Pay per execution
    - Auto-scale (up to 200 instances)
    - 5-min timeout (configurable to 10)
    - Cold start
  
  Premium:
    - Pre-warmed instances (no cold start)
    - VNet integration
    - Unlimited duration
    - Larger instances
  
  Dedicated (App Service):
    - Run on App Service Plan
    - Predictable cost
    - Always warm

Create Function App:
  # Consumption plan
  az functionapp create \
    --name myfunc -g myRG \
    --storage-account mystorageacct \
    --consumption-plan-location eastus \
    --runtime dotnet-isolated --runtime-version 8 \
    --functions-version 4
  
  # Premium plan
  az functionapp plan create \
    --name myfuncplan -g myRG \
    --sku EP1 --min-instances 1 --max-burst 10
  
  az functionapp create \
    --name myfunc -g myRG \
    --storage-account mystorageacct \
    --plan myfuncplan \
    --runtime node --runtime-version 20

Triggers and bindings:
  Triggers (what starts the function):
    HTTP:           REST API endpoints
    Timer:          CRON schedule
    Blob Storage:   File created/updated
    Queue Storage:  Message in queue
    Service Bus:    Message in queue/topic
    Event Hub:      Event stream
    Event Grid:     Event subscriptions
    Cosmos DB:      Document changes (change feed)
    Durable:        Orchestration activities
  
  Bindings (input/output):
    Input:
      Blob Storage (read), Cosmos DB (read), Table Storage
    Output:
      Blob, Queue, Service Bus, Event Hub, Cosmos DB,
      HTTP response, SignalR, Twilio (SMS), SendGrid (email)

Durable Functions:
  Orchestration patterns:
    Function chaining:  F1 → F2 → F3
    Fan-out/fan-in:     F1 → [F2a, F2b, F2c] → F3
    Async HTTP API:     Start → Poll status → Get result
    Monitor:            Periodic polling with timeout
    Human interaction:  Wait for external event/approval

Azure Logic Apps:
  - Visual workflow designer
  - 400+ connectors
  - Enterprise integration
  - B2B workflows
  - Low-code/no-code
  
  Common connectors:
    Office 365, SharePoint, Dynamics 365
    Salesforce, SAP, Oracle
    HTTP, SQL, Service Bus
    Twitter, Slack, Teams

Azure Container Instances (ACI):
  - Run containers without orchestration
  - Per-second billing
  - Quick start (<30 seconds)
  - Best for: burst tasks, CI/CD, simple apps
  
  az container create \
    --name mycontainer -g myRG \
    --image myacr.azurecr.io/worker:latest \
    --cpu 2 --memory 4 \
    --restart-policy OnFailure \
    --environment-variables 'QUEUE_NAME=tasks'
` + "```" + ``,
					CodeExamples: `# Azure serverless management

# 1. Function Apps overview
#!/bin/bash
echo "=== Azure Functions Overview ==="

for func in $(az functionapp list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az functionapp list --query "[?name=='$func'].resourceGroup" -o tsv | head -1)
    
    RUNTIME=$(az functionapp show -n "$func" -g "$RG" \
        --query "siteConfig.linuxFxVersion // siteConfig.netFrameworkVersion" -o tsv 2>/dev/null)
    STATE=$(az functionapp show -n "$func" -g "$RG" --query "state" -o tsv 2>/dev/null)
    KIND=$(az functionapp show -n "$func" -g "$RG" --query "kind" -o tsv 2>/dev/null)
    
    echo "  $func ($STATE) - $RUNTIME"
    
    # List functions
    az functionapp function list -n "$func" -g "$RG" \
        --query "[].{name:name}" -o tsv 2>/dev/null | while read -r fn; do
        echo "    Function: $fn"
    done
done

# 2. Container Apps status
#!/bin/bash
echo "=== Container Apps Status ==="

for env in $(az containerapp env list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az containerapp env list --query "[?name=='$env'].resourceGroup" -o tsv | head -1)
    echo "Environment: $env ($RG)"
    
    for app in $(az containerapp list -g "$RG" \
        --query "[?properties.environmentId | contains(@, '$env')].name" -o tsv 2>/dev/null); do
        
        REPLICAS=$(az containerapp show -n "$app" -g "$RG" \
            --query "properties.template.scale.{min:minReplicas, max:maxReplicas}" -o json 2>/dev/null)
        INGRESS=$(az containerapp show -n "$app" -g "$RG" \
            --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)
        
        echo "  App: $app"
        echo "    URL: https://$INGRESS"
        echo "    Scale: $REPLICAS"
        
        # Revisions
        az containerapp revision list -n "$app" -g "$RG" \
            --query "[].{name:name, active:properties.active, replicas:properties.replicas, traffic:properties.trafficWeight}" \
            -o table 2>/dev/null
    done
done

# 3. Serverless cost estimator
#!/bin/bash
echo "=== Serverless Cost Summary ==="

echo "--- Function Apps ---"
for func in $(az functionapp list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az functionapp list --query "[?name=='$func'].resourceGroup" -o tsv | head -1)
    
    PLAN=$(az functionapp show -n "$func" -g "$RG" --query "appServicePlanId" -o tsv 2>/dev/null)
    PLAN_NAME=$(echo "$PLAN" | rev | cut -d'/' -f1 | rev)
    
    SKU=$(az appservice plan show --ids "$PLAN" --query "sku.tier" -o tsv 2>/dev/null)
    
    echo "  $func: Plan=$PLAN_NAME ($SKU)"
done

echo ""
echo "--- Container Apps ---"
for app in $(az containerapp list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az containerapp list --query "[?name=='$app'].resourceGroup" -o tsv | head -1)
    
    CPU=$(az containerapp show -n "$app" -g "$RG" \
        --query "properties.template.containers[0].resources.cpu" -o tsv 2>/dev/null)
    MEM=$(az containerapp show -n "$app" -g "$RG" \
        --query "properties.template.containers[0].resources.memory" -o tsv 2>/dev/null)
    MIN=$(az containerapp show -n "$app" -g "$RG" \
        --query "properties.template.scale.minReplicas" -o tsv 2>/dev/null)
    MAX=$(az containerapp show -n "$app" -g "$RG" \
        --query "properties.template.scale.maxReplicas" -o tsv 2>/dev/null)
    
    echo "  $app: CPU=$CPU, Mem=$MEM, Scale=$MIN-$MAX"
done

echo ""
echo "--- Container Instances ---"
az container list --query "[].{
    name:name, rg:resourceGroup,
    cpu:containers[0].resources.requests.cpu,
    memory:containers[0].resources.requests.memoryInGb,
    state:instanceView.state
}" -o table 2>/dev/null`,
				},
			},
		},
	})
}
