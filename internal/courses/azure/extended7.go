package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1247,
			Title:       "Azure App Services and Serverless",
			Description: "Master Azure App Service for web application hosting, Azure Functions for serverless compute, and Azure Container Instances for container workloads.",
			Order:       47,
			Lessons: []problems.Lesson{
				{
					Title: "Azure App Service",
					Content: `Azure App Service is a fully managed platform for building, deploying, and scaling web applications, REST APIs, and mobile backends.

**App Service Plans:**
` + "```" + `
App Service Plan defines compute resources:

Tiers:
  Free (F1):       Shared compute, 1GB, 60 min/day
  Shared (D1):     Shared compute, 1GB, custom domain
  Basic (B1-B3):   Dedicated, manual scale, dev/test
  Standard (S1-S3): Dedicated, auto-scale, staging slots
  Premium (P1-P3): Enhanced performance, more scale
  PremiumV3:       Latest hardware, better performance/price
  Isolated (I1-I3): Dedicated VNet environment (ASE)

Key features by tier:
  Free/Shared:  No SLA, no custom SSL, no auto-scale
  Basic:        Custom domain/SSL, no auto-scale, no slots
  Standard:     Auto-scale (10 instances), 5 deployment slots
  Premium:      Auto-scale (30 instances), 20 slots, VNet integration
  Isolated:     100+ instances, private, ASE

Create App Service:
  # Create App Service Plan
  az appservice plan create \
    --resource-group myRG \
    --name myPlan \
    --sku P1V3 \
    --is-linux
  
  # Create Web App (Node.js)
  az webapp create \
    --resource-group myRG \
    --plan myPlan \
    --name mywebapp-unique \
    --runtime "NODE:18-lts"
  
  # Create Web App (Python)
  az webapp create \
    --resource-group myRG \
    --plan myPlan \
    --name mypythonapp \
    --runtime "PYTHON:3.11"
  
  # Create Web App (Docker)
  az webapp create \
    --resource-group myRG \
    --plan myPlan \
    --name mydockerapp \
    --deployment-container-image-name nginx:latest

Supported runtimes:
  .NET 6/7/8, Java 8/11/17, Node 16/18/20
  Python 3.9-3.12, PHP 8.x, Ruby 2.7/3.x
  Go (via container), Rust (via container)
` + "```" + `

**Configuration and Deployment:**
` + "```" + `
App Settings (environment variables):
  az webapp config appsettings set \
    --resource-group myRG --name mywebapp \
    --settings \
      DATABASE_URL="postgresql://..." \
      REDIS_URL="redis://..." \
      NODE_ENV="production"
  
  # From Key Vault reference
  az webapp config appsettings set \
    --resource-group myRG --name mywebapp \
    --settings \
      DB_PASSWORD="@Microsoft.KeyVault(VaultName=myVault;SecretName=db-pass)"

Connection strings:
  az webapp config connection-string set \
    --resource-group myRG --name mywebapp \
    --connection-string-type SQLAzure \
    --settings \
      DefaultConnection="Server=..."

Deployment methods:
  1. Git deployment:
     az webapp deployment source config-local-git \
       --resource-group myRG --name mywebapp
     git remote add azure <deploy-url>
     git push azure main
  
  2. GitHub Actions:
     az webapp deployment github-actions add \
       --resource-group myRG --name mywebapp \
       --repo "user/repo" --branch main
  
  3. ZIP deploy:
     az webapp deploy \
       --resource-group myRG --name mywebapp \
       --src-path app.zip --type zip
  
  4. Docker:
     az webapp config container set \
       --resource-group myRG --name mywebapp \
       --docker-custom-image-name myregistry.azurecr.io/myapp:v1 \
       --docker-registry-server-url https://myregistry.azurecr.io

Deployment slots:
  # Create staging slot
  az webapp deployment slot create \
    --resource-group myRG --name mywebapp \
    --slot staging
  
  # Deploy to staging
  az webapp deploy \
    --resource-group myRG --name mywebapp \
    --slot staging --src-path app.zip --type zip
  
  # Swap staging to production
  az webapp deployment slot swap \
    --resource-group myRG --name mywebapp \
    --slot staging --target-slot production
  
  # Auto-swap on deployment
  az webapp deployment slot auto-swap \
    --resource-group myRG --name mywebapp \
    --slot staging

Scaling:
  # Manual scale
  az appservice plan update \
    --resource-group myRG --name myPlan \
    --number-of-workers 5
  
  # Auto-scale
  az monitor autoscale create \
    --resource-group myRG \
    --resource myPlan \
    --resource-type Microsoft.Web/serverfarms \
    --name web-autoscale \
    --min-count 2 --max-count 10 --count 3
  
  az monitor autoscale rule create \
    --resource-group myRG \
    --autoscale-name web-autoscale \
    --condition "CpuPercentage > 70 avg 5m" \
    --scale out 2
` + "```" + `

**VNet Integration and Security:**
` + "```" + `
VNet Integration:
  # Regional VNet integration (recommended)
  az webapp vnet-integration add \
    --resource-group myRG --name mywebapp \
    --vnet myVNet --subnet app-integration-subnet
  
  # Route all traffic through VNet
  az webapp config appsettings set \
    --resource-group myRG --name mywebapp \
    --settings WEBSITE_VNET_ROUTE_ALL=1

Access restrictions:
  # Allow only from specific IPs
  az webapp config access-restriction add \
    --resource-group myRG --name mywebapp \
    --rule-name office --priority 100 --action Allow \
    --ip-address 203.0.113.0/24
  
  # Allow from VNet subnet
  az webapp config access-restriction add \
    --resource-group myRG --name mywebapp \
    --rule-name from-agw --priority 110 --action Allow \
    --vnet-name myVNet --subnet agw-subnet
  
  # Default deny
  az webapp config access-restriction set \
    --resource-group myRG --name mywebapp \
    --default-action Deny

Custom domain and SSL:
  # Add custom domain
  az webapp config hostname add \
    --resource-group myRG --webapp-name mywebapp \
    --hostname www.example.com
  
  # Add managed SSL certificate
  az webapp config ssl create \
    --resource-group myRG --name mywebapp \
    --hostname www.example.com
  
  # Enforce HTTPS
  az webapp update \
    --resource-group myRG --name mywebapp \
    --https-only true
` + "```" + ``,
					CodeExamples: `# Azure App Service management

# 1. App Service deployment pipeline
#!/bin/bash
set -euo pipefail

RG="${1:?Usage: $0 <rg> <app-name> <image>}"
APP="${2:?Usage: $0 <rg> <app-name> <image>}"
IMAGE="${3:?Usage: $0 <rg> <app-name> <image>}"

echo "=== Deploying $APP ==="

# Pre-deployment health check
echo "Checking current health..."
CURRENT_URL="https://${APP}.azurewebsites.net/health"
if curl -sf --max-time 10 "$CURRENT_URL" > /dev/null 2>&1; then
    echo "  Current deployment healthy"
else
    echo "  WARNING: Current deployment unhealthy"
fi

# Deploy to staging slot
echo "Deploying to staging slot..."
az webapp config container set \
    -g "$RG" -n "$APP" --slot staging \
    --docker-custom-image-name "$IMAGE"

# Wait for staging to be ready
echo "Waiting for staging..."
STAGING_URL="https://${APP}-staging.azurewebsites.net/health"
for i in $(seq 1 60); do
    if curl -sf --max-time 5 "$STAGING_URL" > /dev/null 2>&1; then
        echo "  Staging is healthy!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  ERROR: Staging failed health check"
        exit 1
    fi
    sleep 5
done

# Swap to production
echo "Swapping staging → production..."
az webapp deployment slot swap \
    -g "$RG" -n "$APP" \
    --slot staging --target-slot production

# Verify production
echo "Verifying production..."
sleep 10
if curl -sf --max-time 10 "$CURRENT_URL" > /dev/null 2>&1; then
    echo "  Production is healthy!"
else
    echo "  WARNING: Production health check failed, rolling back..."
    az webapp deployment slot swap \
        -g "$RG" -n "$APP" \
        --slot staging --target-slot production
    echo "  Rolled back!"
    exit 1
fi

echo "=== Deployment complete ==="

# 2. App Service monitoring
#!/bin/bash
echo "=== App Service Status ==="

for rg in $(az group list --query "[].name" -o tsv); do
    APPS=$(az webapp list -g "$rg" --query "[].name" -o tsv 2>/dev/null)
    
    for app in $APPS; do
        STATE=$(az webapp show -g "$rg" -n "$app" --query "state" -o tsv 2>/dev/null)
        PLAN=$(az webapp show -g "$rg" -n "$app" --query "appServicePlanId" -o tsv 2>/dev/null | rev | cut -d/ -f1 | rev)
        URL="https://${app}.azurewebsites.net"
        
        # Quick health check
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$URL" 2>/dev/null)
        
        printf "  %-30s  State: %-8s  Plan: %-15s  HTTP: %s\n" \
            "$app" "$STATE" "$PLAN" "$HTTP_CODE"
    done
done`,
				},
				{
					Title: "Azure Functions and Container Instances",
					Content: `Azure Functions provides serverless compute for event-driven applications. Azure Container Instances (ACI) offers the fastest way to run containers in Azure.

**Azure Functions:**
` + "```" + `
Serverless compute triggered by events.

Hosting plans:
  Consumption:   Pay per execution, auto-scale 0-200 instances
                 Cold start latency, 5/10 min timeout
  Premium:       Pre-warmed instances, VNet, no cold start
                 Unlimited duration
  Dedicated:     Run on App Service plan, predictable cost

Triggers:
  HTTP:          REST API endpoints
  Timer:         CRON schedule
  Blob:          New/modified blob
  Queue:         New queue message
  Event Hub:     Streaming events
  Event Grid:    Event-driven
  Service Bus:   Message queue/topic
  Cosmos DB:     Document changes

Create Function App:
  # Create storage account (required)
  az storage account create \
    --resource-group myRG --name funcstorageacct \
    --sku Standard_LRS
  
  # Create Function App
  az functionapp create \
    --resource-group myRG \
    --name myfuncapp \
    --storage-account funcstorageacct \
    --consumption-plan-location eastus \
    --runtime python --runtime-version 3.11 \
    --functions-version 4 \
    --os-type Linux

  # Or with a dedicated plan
  az functionapp create \
    --resource-group myRG \
    --name myfuncapp \
    --storage-account funcstorageacct \
    --plan myPremiumPlan \
    --runtime node --runtime-version 18 \
    --functions-version 4

Local development:
  # Install Azure Functions Core Tools
  npm install -g azure-functions-core-tools@4
  
  # Create project
  func init myproject --worker-runtime python
  cd myproject
  
  # Create function
  func new --name HttpTrigger --template "HTTP trigger"
  
  # Run locally
  func start
  
  # Deploy
  func azure functionapp publish myfuncapp

Durable Functions (orchestration):
  - Function chaining: f1 → f2 → f3
  - Fan-out/fan-in: parallel execution
  - Monitor: polling pattern
  - Human interaction: approval workflows
  
  Patterns:
    Orchestrator: Defines workflow
    Activity:     Individual work units
    Entity:       Stateful singleton
` + "```" + `

**Azure Container Instances (ACI):**
` + "```" + `
Run containers without managing infrastructure.

Features:
  - Fast startup (seconds)
  - Per-second billing
  - Custom sizes (CPU/memory)
  - Linux and Windows containers
  - Container groups (similar to pods)
  - VNet integration
  - Persistent volumes (Azure Files)

Create container:
  # Simple container
  az container create \
    --resource-group myRG \
    --name mycontainer \
    --image nginx:latest \
    --dns-name-label myapp-unique \
    --ports 80 \
    --cpu 1 --memory 1.5
  
  # Container with environment variables
  az container create \
    --resource-group myRG \
    --name myapp \
    --image myregistry.azurecr.io/myapp:v1 \
    --registry-login-server myregistry.azurecr.io \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --environment-variables \
      DB_HOST=mydb.postgres.database.azure.com \
      APP_ENV=production \
    --secure-environment-variables \
      DB_PASSWORD=secret123 \
    --cpu 2 --memory 4 \
    --ports 8080

  # With Azure File volume
  az container create \
    --resource-group myRG \
    --name myapp \
    --image myapp:v1 \
    --azure-file-volume-account-name mystorageacct \
    --azure-file-volume-account-key "$STORAGE_KEY" \
    --azure-file-volume-share-name myshare \
    --azure-file-volume-mount-path /data

Container groups (YAML):
  apiVersion: 2021-10-01
  location: eastus
  name: my-container-group
  properties:
    containers:
    - name: webapp
      properties:
        image: myapp:v1
        ports:
        - port: 80
        resources:
          requests:
            cpu: 1
            memoryInGb: 1.5
    - name: sidecar
      properties:
        image: fluentd:latest
        resources:
          requests:
            cpu: 0.5
            memoryInGb: 0.5
    osType: Linux
    ipAddress:
      type: Public
      ports:
      - port: 80
  
  az container create --resource-group myRG \
    --file container-group.yaml

Management:
  az container list -g myRG --output table
  az container logs -g myRG --name mycontainer
  az container exec -g myRG --name mycontainer --exec-command /bin/bash
  az container show -g myRG --name mycontainer
  az container restart -g myRG --name mycontainer
  az container delete -g myRG --name mycontainer

ACI vs AKS vs App Service:
  ACI:
    + Simplest, fastest startup
    + Per-second billing
    + Burst workloads
    - No orchestration
    - Limited networking
  
  AKS:
    + Full Kubernetes
    + Complex orchestration
    + Advanced networking
    - Cluster management overhead
    - Minimum cost (control plane)
  
  App Service:
    + Fully managed platform
    + Built-in CI/CD, SSL, domains
    + Deployment slots
    - Less control
    - Higher minimum cost
` + "```" + ``,
					CodeExamples: `# Serverless and container management

# 1. Function App deployment
#!/bin/bash
set -euo pipefail

RG="${1:?Usage: $0 <rg> <func-app-name>}"
FUNC_APP="${2:?Usage: $0 <rg> <func-app-name>}"

echo "=== Function App Deployment: $FUNC_APP ==="

# Get current settings
echo "Current configuration:"
az functionapp config show -g "$RG" -n "$FUNC_APP" \
    --query "{runtime:linuxFxVersion, state:state, defaultHostName:defaultHostName}" \
    -o json | jq .

# Deploy
echo "Deploying..."
func azure functionapp publish "$FUNC_APP" --build remote 2>&1 | tail -10

# Verify
echo ""
echo "Function list:"
az functionapp function list -g "$RG" -n "$FUNC_APP" \
    --query "[].{name:name, language:language}" -o table

echo ""
echo "Deployment complete."

# 2. ACI monitoring script
#!/bin/bash
echo "=== Azure Container Instances Status ==="

for rg in $(az group list --query "[].name" -o tsv); do
    CONTAINERS=$(az container list -g "$rg" \
        --query "[].{name:name, state:instanceView.state, ip:ipAddress.ip, cpu:containers[0].resources.requests.cpu, mem:containers[0].resources.requests.memoryInGb}" \
        -o json 2>/dev/null)
    
    if [ "$(echo "$CONTAINERS" | jq length 2>/dev/null)" -gt 0 ]; then
        echo ""
        echo "Resource Group: $rg"
        echo "$CONTAINERS" | jq -r '.[] | 
            "  \(.name)\tState: \(.state)\tIP: \(.ip // "none")\tCPU: \(.cpu)\tMem: \(.mem)GB"'
    fi
done

# 3. Serverless cost tracker
#!/bin/bash
echo "=== Serverless Resource Inventory ==="

echo "--- Function Apps ---"
az functionapp list --query "[].{
    name:name, rg:resourceGroup, state:state,
    plan:appServicePlanId | split('/') | [-1]
}" -o table

echo ""
echo "--- Container Instances ---"
for rg in $(az group list --query "[].name" -o tsv); do
    az container list -g "$rg" --query "[].{
        name:name, os:osType,
        cpu:containers[].resources.requests.cpu | [0],
        memory:containers[].resources.requests.memoryInGb | [0],
        state:provisioningState
    }" -o table 2>/dev/null
done`,
				},
			},
		},
	})
}
