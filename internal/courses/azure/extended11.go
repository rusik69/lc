package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1255,
			Title:       "Azure Monitoring and Observability",
			Description: "Master Azure Monitor, Application Insights, Log Analytics, alerts, and dashboards for comprehensive observability.",
			Order:       55,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Monitor and Log Analytics",
					Content: `Azure Monitor provides comprehensive monitoring for Azure and hybrid resources. It collects metrics, logs, and traces into a unified platform.

**Azure Monitor Architecture:**
` + "```" + `
Data sources:
  Application:    Application Insights (APM)
  Guest OS:       VM extensions, diagnostics
  Azure Resource: Platform metrics, activity logs
  Subscription:   Service health, security center
  Tenant:         Azure AD audit logs

Data stores:
  Metrics:        Numeric time-series data (near real-time)
  Logs:           Structured/unstructured data in
                  Log Analytics workspace (KQL queries)
  Traces:         Distributed tracing data

Components:
  Azure Monitor Metrics:  Time-series database
  Log Analytics:          Log aggregation and query
  Application Insights:   APM for applications
  Alerts:                 Proactive notification
  Workbooks:              Interactive reports
  Dashboards:             Visual overview

Log Analytics Workspace:
  # Create workspace
  az monitor log-analytics workspace create \
    --resource-group myRG \
    --workspace-name myWorkspace \
    --location eastus \
    --sku PerGB2018 \
    --retention-time 90
  
  # Enable diagnostics for resource
  az monitor diagnostic-settings create \
    --name diag-storage \
    --resource /subscriptions/.../storageAccounts/mystorageacct \
    --workspace myWorkspace \
    --logs '[{"category":"StorageRead","enabled":true},
             {"category":"StorageWrite","enabled":true}]' \
    --metrics '[{"category":"Transaction","enabled":true}]'
` + "```" + `

**KQL (Kusto Query Language):**
` + "```" + `
KQL is the query language for Log Analytics.

Basic queries:
  // Get recent application errors
  AppExceptions
  | where TimeGenerated > ago(24h)
  | order by TimeGenerated desc
  | take 100
  
  // Count requests by status code
  AppRequests
  | where TimeGenerated > ago(1h)
  | summarize count() by resultCode
  | order by count_ desc
  
  // Average response time by operation
  AppRequests
  | where TimeGenerated > ago(1h)
  | summarize avg(duration) by name
  | order by avg_duration desc
  
  // Find slow requests
  AppRequests
  | where TimeGenerated > ago(1h)
  | where duration > 5000  // > 5 seconds
  | project TimeGenerated, name, duration, resultCode
  | order by duration desc

Advanced queries:
  // Error rate trend
  AppRequests
  | where TimeGenerated > ago(24h)
  | summarize total=count(), errors=countif(success==false) by bin(TimeGenerated, 1h)
  | extend error_rate = round(todouble(errors) / total * 100, 2)
  | project TimeGenerated, total, errors, error_rate
  
  // P95 latency by endpoint
  AppRequests
  | where TimeGenerated > ago(1h)
  | summarize percentile(duration, 95) by name
  | order by percentile_duration_95 desc
  
  // Join exceptions with requests
  AppRequests
  | where TimeGenerated > ago(1h)
  | where success == false
  | join kind=leftouter (
      AppExceptions
      | where TimeGenerated > ago(1h)
    ) on operation_Id
  | project TimeGenerated, name, resultCode, outerMessage
  | take 50
  
  // VM performance
  Perf
  | where TimeGenerated > ago(1h)
  | where ObjectName == "Processor" and CounterName == "% Processor Time"
  | where InstanceName == "_Total"
  | summarize avg(CounterValue) by Computer, bin(TimeGenerated, 5m)
  
  // Activity log analysis
  AzureActivity
  | where TimeGenerated > ago(7d)
  | where OperationNameValue contains "delete"
  | project TimeGenerated, Caller, OperationNameValue, 
            ResourceGroup, _ResourceId, ActivityStatusValue
  | order by TimeGenerated desc
` + "```" + `

**Alerts:**
` + "```" + `
Alert types:
  Metric alerts:      Based on metric thresholds
  Log alerts:         Based on log query results
  Activity log:       Based on Azure operations
  Smart detection:    AI-based anomaly detection

Create metric alert:
  az monitor metrics alert create \
    --resource-group myRG \
    --name "High CPU Alert" \
    --scopes /subscriptions/.../virtualMachines/myVM \
    --condition "avg Percentage CPU > 85" \
    --window-size 5m \
    --evaluation-frequency 1m \
    --severity 2 \
    --action /subscriptions/.../actionGroups/myActionGroup \
    --description "VM CPU is above 85%"

Create log alert:
  az monitor scheduled-query create \
    --resource-group myRG \
    --name "App Errors Alert" \
    --scopes /subscriptions/.../workspaces/myWorkspace \
    --condition "count > 10" \
    --condition-query "AppExceptions | where severityLevel >= 3" \
    --evaluation-frequency 5m \
    --window-size 15m \
    --severity 1 \
    --action /subscriptions/.../actionGroups/myActionGroup

Action Groups:
  az monitor action-group create \
    --resource-group myRG \
    --name myActionGroup \
    --short-name myAG \
    --action email admin admin@example.com \
    --action webhook ops-webhook https://hooks.example.com/alert \
    --action azureapppush mobile-push admin@example.com

Recommended alerts for production:
  VM:
    - CPU > 85% for 10 min
    - Memory > 90%
    - Disk > 90% full
    - VM unavailable
  
  App Service:
    - HTTP 5xx > 10 in 5 min
    - Response time P95 > 5s
    - CPU > 80%
    - Memory > 80%
  
  SQL Database:
    - DTU > 80%
    - Deadlocks > 5 in 5 min
    - Failed connections > 10
  
  AKS:
    - Node not ready
    - Pod crash loops
    - CPU/memory > 80%
` + "```" + ``,
					CodeExamples: `# Azure monitoring scripts

# 1. Monitoring setup script
#!/bin/bash
set -euo pipefail

RG="${1:?Usage: $0 <resource-group> <workspace-name>}"
WORKSPACE="${2:?Usage: $0 <resource-group> <workspace-name>}"
LOCATION="${3:-eastus}"

echo "=== Setting up Azure Monitoring ==="

# Create workspace
echo "Creating Log Analytics workspace..."
az monitor log-analytics workspace create \
    -g "$RG" -n "$WORKSPACE" --location "$LOCATION" \
    --sku PerGB2018 --retention-time 90

WORKSPACE_ID=$(az monitor log-analytics workspace show \
    -g "$RG" -n "$WORKSPACE" --query "id" -o tsv)

# Create action group
echo "Creating action group..."
az monitor action-group create \
    -g "$RG" -n "ops-alerts" --short-name "ops" \
    --action email admin admin@example.com

# Enable diagnostics for all VMs
echo "Enabling VM diagnostics..."
for vm in $(az vm list -g "$RG" --query "[].name" -o tsv 2>/dev/null); do
    echo "  Enabling for $vm..."
    az monitor diagnostic-settings create \
        --name "diag-$vm" \
        --resource "$(az vm show -g "$RG" -n "$vm" --query "id" -o tsv)" \
        --workspace "$WORKSPACE_ID" \
        --metrics '[{"category":"AllMetrics","enabled":true}]' 2>/dev/null || true
done

# Create key alerts
echo "Creating alerts..."
ACTION_GROUP_ID=$(az monitor action-group show -g "$RG" -n "ops-alerts" --query "id" -o tsv)

# VM CPU alert
for vm in $(az vm list -g "$RG" --query "[].id" -o tsv 2>/dev/null); do
    VM_NAME=$(basename "$vm")
    az monitor metrics alert create \
        -g "$RG" -n "cpu-$VM_NAME" \
        --scopes "$vm" \
        --condition "avg Percentage CPU > 85" \
        --window-size 10m --evaluation-frequency 5m \
        --severity 2 --action "$ACTION_GROUP_ID" 2>/dev/null || true
done

echo "=== Monitoring setup complete ==="

# 2. Log Analytics query runner
#!/bin/bash
WORKSPACE="${1:?Usage: $0 <workspace-name> <query>}"
QUERY="${2:?Usage: $0 <workspace-name> <query>}"

RG=$(az monitor log-analytics workspace list \
    --query "[?name=='$WORKSPACE'].resourceGroup" -o tsv | head -1)

az monitor log-analytics query \
    --workspace "$WORKSPACE" -g "$RG" \
    --analytics-query "$QUERY" \
    --timespan PT24H \
    -o table

# 3. Alert status dashboard
#!/bin/bash
echo "=== Azure Alert Status ==="

# Active alerts
echo "--- Active Alerts ---"
az monitor alert list --query "[?state=='Active' || state=='New'].{
    name:name, severity:severity, state:state,
    fired:firedDateTime, target:targetResourceName
}" -o table 2>/dev/null || \
az monitor metrics alert list --query "[].{
    name:name, severity:severity, enabled:enabled
}" -o table 2>/dev/null

# Alert rules
echo ""
echo "--- Metric Alert Rules ---"
for rg in $(az group list --query "[].name" -o tsv); do
    ALERTS=$(az monitor metrics alert list -g "$rg" \
        --query "[].{name:name, severity:severity, enabled:enabled}" \
        -o json 2>/dev/null)
    
    if [ "$(echo "$ALERTS" | jq length 2>/dev/null)" -gt 0 ]; then
        echo "Resource Group: $rg"
        echo "$ALERTS" | jq -r '.[] | "  \(.name)\tSeverity: \(.severity)\tEnabled: \(.enabled)"'
    fi
done`,
				},
				{
					Title: "Application Insights and Distributed Tracing",
					Content: `Application Insights provides deep application performance monitoring (APM) with automatic dependency tracking, distributed tracing, and smart alerting.

**Application Insights Setup:**
` + "```" + `
Create Application Insights:
  az monitor app-insights component create \
    --resource-group myRG \
    --app myAppInsights \
    --location eastus \
    --kind web \
    --application-type web \
    --workspace /subscriptions/.../workspaces/myWorkspace

Connection string:
  az monitor app-insights component show \
    --resource-group myRG --app myAppInsights \
    --query "connectionString" -o tsv

SDK Integration:

  Python:
    pip install opencensus-ext-azure
    
    from opencensus.ext.azure.trace_exporter import AzureExporter
    from opencensus.trace.tracer import Tracer
    
    tracer = Tracer(
        exporter=AzureExporter(connection_string='InstrumentationKey=...'),
        sampler=ProbabilitySampler(1.0),
    )

  Go:
    // Using Application Insights Go SDK
    client := appinsights.NewTelemetryClient(instrumentationKey)
    client.TrackEvent("UserLogin")
    client.TrackMetric("ResponseTime", 250)
    
    // Track request
    request := appinsights.NewRequestTelemetry(method, url, duration, responseCode)
    client.Track(request)

  Node.js:
    const appInsights = require("applicationinsights");
    appInsights.setup(connectionString)
        .setAutoCollectRequests(true)
        .setAutoCollectPerformance(true)
        .setAutoCollectExceptions(true)
        .setAutoCollectDependencies(true)
        .start();

  .NET:
    // In Program.cs
    builder.Services.AddApplicationInsightsTelemetry();

Auto-instrumentation (codeless):
  # App Service
  az webapp config appsettings set \
    -g myRG -n myWebApp \
    --settings \
      APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..." \
      ApplicationInsightsAgent_EXTENSION_VERSION="~3"
  
  # AKS (auto-instrumentation operator)
  # Deploy Application Insights auto-instrumentation
  kubectl apply -f https://github.com/microsoft/Application-Insights-K8s-Codeless-Attach/...
` + "```" + `

**Key Features:**
` + "```" + `
Application Map:
  - Visual dependency topology
  - Shows all components and dependencies
  - Highlights failing components
  - Click to drill into details

Live Metrics:
  - Real-time performance and failure stream
  - No query delay
  - Useful during deployments
  - Request rate, response time, failures

Smart Detection:
  - Automatic anomaly detection
  - Failure anomalies (spike in exception rate)
  - Performance anomalies (slow page load)
  - Memory leak detection
  - Degradation in server response time
  - Abnormal rise in exception volume

Availability tests:
  # URL ping test
  az monitor app-insights web-test create \
    --resource-group myRG \
    --app myAppInsights \
    --name "Homepage Test" \
    --web-test-name "homepage" \
    --defined-web-test-name "homepage" \
    --locations "us-ca-sjc-azr" "us-tx-sn1-azr" "emea-nl-ams-azr" \
    --frequency 300 \
    --timeout 120 \
    --kind ping \
    --geo-locations "us-ca-sjc-azr" \
    --request-url "https://myapp.azurewebsites.net/health"

Key KQL queries:
  // Application health overview
  AppRequests
  | where TimeGenerated > ago(1h)
  | summarize 
      requests=count(),
      failures=countif(success==false),
      avg_duration=avg(duration),
      p95_duration=percentile(duration, 95)
  
  // Dependency failures
  AppDependencies
  | where TimeGenerated > ago(1h)
  | where success == false
  | summarize count() by target, type, resultCode
  | order by count_ desc
  
  // User sessions and flows
  AppPageViews
  | where TimeGenerated > ago(24h)
  | summarize pageViews=count(), users=dcount(user_Id) by name
  | order by pageViews desc
  
  // End-to-end transaction
  AppRequests
  | where operation_Id == "specific-operation-id"
  | union (AppDependencies | where operation_Id == "specific-operation-id")
  | union (AppExceptions | where operation_Id == "specific-operation-id")
  | order by TimeGenerated asc
  
  // Custom events tracking
  AppEvents
  | where TimeGenerated > ago(24h)
  | summarize count() by name
  | order by count_ desc
` + "```" + ``,
					CodeExamples: `# Application Insights management

# 1. Application Insights health report
#!/bin/bash
APP_INSIGHTS="${1:?Usage: $0 <app-insights-name>}"
RG=$(az monitor app-insights component list \
    --query "[?name=='$APP_INSIGHTS'].resourceGroup" -o tsv | head -1)

echo "=== Application Insights Report: $APP_INSIGHTS ==="

# Application overview
echo ""
echo "--- Last 1 Hour Summary ---"
az monitor app-insights metrics show \
    --app "$APP_INSIGHTS" -g "$RG" \
    --metric "requests/count" \
    --interval PT1H 2>/dev/null | jq '.value' || echo "No data"

az monitor app-insights metrics show \
    --app "$APP_INSIGHTS" -g "$RG" \
    --metric "requests/failed" \
    --interval PT1H 2>/dev/null | jq '.value' || echo "No data"

az monitor app-insights metrics show \
    --app "$APP_INSIGHTS" -g "$RG" \
    --metric "requests/duration" \
    --interval PT1H --aggregation avg \
    2>/dev/null | jq '.value' || echo "No data"

# Recent exceptions
echo ""
echo "--- Recent Exceptions ---"
az monitor app-insights events show \
    --app "$APP_INSIGHTS" -g "$RG" \
    --type exceptions --limit 10 \
    --query "value[].{time:timestamp, type:exception.type, message:exception.outerMessage}" \
    -o table 2>/dev/null

# Availability
echo ""
echo "--- Availability Tests ---"
az monitor app-insights metrics show \
    --app "$APP_INSIGHTS" -g "$RG" \
    --metric "availabilityResults/availabilityPercentage" \
    --interval PT1H 2>/dev/null | jq '.value' || echo "No data"

# 2. Cost analysis for monitoring
#!/bin/bash
echo "=== Monitoring Cost Analysis ==="

# Log Analytics workspaces
echo "--- Log Analytics Workspaces ---"
az monitor log-analytics workspace list --query "[].{
    name:name, rg:resourceGroup, sku:sku.name,
    retention:retentionInDays
}" -o table

# Application Insights
echo ""
echo "--- Application Insights ---"
az monitor app-insights component list --query "[].{
    name:name, rg:resourceGroup, kind:kind,
    retention:retentionInDays
}" -o table

# Data volume estimate
echo ""
echo "--- Data Volume (Last 30 Days) ---"
for ws in $(az monitor log-analytics workspace list --query "[].name" -o tsv); do
    RG=$(az monitor log-analytics workspace list \
        --query "[?name=='$ws'].resourceGroup" -o tsv | head -1)
    
    echo "Workspace: $ws"
    az monitor log-analytics query \
        -w "$ws" -g "$RG" \
        --analytics-query "Usage | where TimeGenerated > ago(30d) | summarize DataGB=sum(Quantity)/1024 by DataType | order by DataGB desc | take 10" \
        -o table 2>/dev/null | head -15
done`,
				},
			},
		},
	})
}
