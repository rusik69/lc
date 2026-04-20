package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1265,
			Title:       "Azure Monitoring and Observability",
			Description: "Master Azure monitoring with Azure Monitor, Log Analytics, Application Insights, alerts, and diagnostic settings for comprehensive observability.",
			Order:       65,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Monitor and Log Analytics",
					Content: `Azure Monitor provides a comprehensive solution for collecting, analyzing, and acting on telemetry from cloud and on-premises environments.

**Azure Monitor Architecture:**
` + "```" + `
Data sources:
  Application:     Application Insights (APM)
  Guest OS:        Diagnostics extension, agents
  Azure Resources: Platform metrics, diagnostic logs
  Subscription:    Activity log
  Tenant:          Azure AD logs

Data platforms:
  Metrics:
    - Numeric time-series data
    - Lightweight, near real-time
    - Stored in time-series database
    - 93 days retention (free)
    - Metric namespaces per resource type
  
  Logs:
    - Log Analytics Workspace
    - Kusto Query Language (KQL)
    - Up to 2 years interactive retention
    - Archive up to 12 years
    - Cross-workspace queries

Azure Monitor Agent (AMA):
  Replaces legacy agents (MMA, OMS, Telegraf)
  
  # Install via extension
  az vm extension set \
    --resource-group myRG --vm-name myVM \
    --name AzureMonitorLinuxAgent \
    --publisher Microsoft.Azure.Monitor
  
  # Data Collection Rules (DCR)
  az monitor data-collection rule create \
    --name myDCR -g myRG --location eastus \
    --data-flows '[{
      "streams": ["Microsoft-Syslog", "Microsoft-Perf"],
      "destinations": ["myWorkspace"]
    }]' \
    --log-analytics '[{
      "name": "myWorkspace",
      "workspaceResourceId": "/subscriptions/.../workspaces/myLA"
    }]' \
    --syslog '[{
      "name": "syslogDataSource",
      "streams": ["Microsoft-Syslog"],
      "facilityNames": ["auth", "authpriv", "daemon"],
      "logLevels": ["Warning", "Error", "Critical"]
    }]'

Log Analytics Workspace:
  az monitor log-analytics workspace create \
    --workspace-name myLA -g myRG \
    --location eastus --sku PerGB2018

  Pricing tiers:
    Pay-As-You-Go:    Per GB ingested
    Commitment tiers:  100, 200, 300, 400, 500+ GB/day (discounted)
  
  Retention:
    Free:    31 days
    Paid:    31-730 days (interactive)
    Archive: Up to 12 years (query with restore)
  
  Best practices:
    - Centralize logs in fewer workspaces
    - Separate by data sovereignty requirements
    - Use tables for data organization
    - Configure data collection rules carefully
    - Set daily cap to prevent cost overruns
` + "```" + `

**KQL (Kusto Query Language):**
` + "```" + `
Basic queries:
  // Search across tables
  search "error" | take 100
  
  // Basic table query
  AzureActivity
  | where TimeGenerated > ago(24h)
  | where Level == "Error"
  | project TimeGenerated, OperationName, Caller, ResourceGroup
  | order by TimeGenerated desc
  | take 50

Filtering and projecting:
  // Complex filters
  Heartbeat
  | where TimeGenerated > ago(1h)
  | where Computer startswith "prod-"
  | where OSType == "Linux"
  | project Computer, ComputerIP, OSType, TimeGenerated
  | distinct Computer, ComputerIP

  // String operations
  SecurityEvent
  | where Account contains "admin"
  | where Account !endswith "$"
  | where Activity has "logon"
  
  // Regex
  ContainerLog
  | where LogEntry matches regex @"ERROR|FATAL|CRITICAL"

Aggregations:
  // Count by category
  AzureActivity
  | where TimeGenerated > ago(7d)
  | summarize count() by OperationNameValue
  | order by count_ desc
  | take 10
  
  // Time-based aggregation
  Perf
  | where ObjectName == "Processor" and CounterName == "% Processor Time"
  | where TimeGenerated > ago(24h)
  | summarize AvgCPU=avg(CounterValue), MaxCPU=max(CounterValue)
    by bin(TimeGenerated, 1h), Computer
  | order by TimeGenerated asc
  
  // Percentiles
  requests
  | where timestamp > ago(1h)
  | summarize percentiles(duration, 50, 90, 95, 99) by bin(timestamp, 5m)

Joins:
  // Join VM performance with heartbeat
  Perf
  | where TimeGenerated > ago(1h)
  | where ObjectName == "Memory" and CounterName == "% Used Memory"
  | summarize AvgMem=avg(CounterValue) by Computer
  | join kind=inner (
      Heartbeat
      | where TimeGenerated > ago(1h)
      | distinct Computer, ComputerIP, OSType
  ) on Computer
  | project Computer, ComputerIP, OSType, AvgMem

Advanced:
  // Dynamic parsing
  AzureDiagnostics
  | extend parsedLog = parse_json(properties_s)
  | extend statusCode = tostring(parsedLog.statusCode)
  | where statusCode != "200"
  
  // Time series analysis
  requests
  | where timestamp > ago(7d)
  | make-series reqCount=count() on timestamp step 1h
  | extend anomalies = series_decompose_anomalies(reqCount)
  
  // Render charts
  Perf
  | where ObjectName == "Processor"
  | summarize avg(CounterValue) by bin(TimeGenerated, 5m), Computer
  | render timechart
  
  // Materialized views for common queries
  // Functions for reusable query logic
  let threshold = 90;
  Perf
  | where ObjectName == "Processor"
  | where CounterValue > threshold
  | summarize count() by Computer
` + "```" + `

**Alerts and Action Groups:**
` + "```" + `
Alert types:
  Metric alerts:
    - Threshold on metric values
    - Static or dynamic thresholds
    - Multi-resource support
  
  Log alerts:
    - KQL query results
    - Number of results or metric measurement
    - Frequency: 1 min to 1 day
  
  Activity log alerts:
    - Azure resource events
    - Service health alerts
    - Resource health alerts
  
  Smart detection:
    - Application Insights anomaly detection
    - Automatic (no configuration)

Create metric alert:
  az monitor metrics alert create \
    --name high-cpu -g myRG \
    --scopes "/subscriptions/.../virtualMachines/myVM" \
    --condition "avg Percentage CPU > 90" \
    --window-size 5m \
    --evaluation-frequency 1m \
    --severity 2 \
    --action "/subscriptions/.../actionGroups/myAG"

Create log alert:
  az monitor scheduled-query create \
    --name error-spike -g myRG \
    --scopes "/subscriptions/.../workspaces/myLA" \
    --condition "count 'union AppExceptions, AppTraces | where SeverityLevel >= 3 | where TimeGenerated > ago(5m)' > 100" \
    --condition-query "union AppExceptions, AppTraces | where SeverityLevel >= 3" \
    --evaluation-frequency 5m \
    --window-size 5m \
    --severity 1 \
    --action "/subscriptions/.../actionGroups/myAG"

Action groups:
  az monitor action-group create \
    --name myAG -g myRG \
    --short-name AG \
    --email-receiver name=Admin email=admin@example.com \
    --sms-receiver name=OnCall country-code=1 phone-number=5551234567 \
    --webhook-receiver name=Slack uri="https://hooks.slack.com/services/..." \
    --azure-function-receiver name=AutoRemediate \
      function-app-resource-id="/subscriptions/.../sites/myFuncApp" \
      function-name="remediate" \
      http-trigger-url="https://myFuncApp.azurewebsites.net/api/remediate"

Alert processing rules:
  # Suppress alerts during maintenance
  az monitor alert-processing-rule create \
    --name maintenance-window -g myRG \
    --scopes "/subscriptions/<sub-id>/resourceGroups/myRG" \
    --rule-type RemoveAllActionGroups \
    --schedule-recurrence-type Weekly \
    --schedule-recurrence "Sunday" \
    --schedule-start-datetime "2024-01-01 02:00:00" \
    --schedule-end-datetime "2024-12-31 06:00:00"

Diagnostic settings:
  # Send resource logs to Log Analytics
  az monitor diagnostic-settings create \
    --name send-to-la \
    --resource "/subscriptions/.../appServices/myApp" \
    --workspace "/subscriptions/.../workspaces/myLA" \
    --logs '[{"category": "AppServiceHTTPLogs", "enabled": true},
             {"category": "AppServiceConsoleLogs", "enabled": true},
             {"category": "AppServiceAppLogs", "enabled": true}]' \
    --metrics '[{"category": "AllMetrics", "enabled": true}]'
` + "```" + ``,
					CodeExamples: `# Azure monitoring scripts

# 1. Comprehensive monitoring check
#!/bin/bash
echo "=== Azure Monitoring Status ==="

# Log Analytics workspaces
echo "--- Log Analytics Workspaces ---"
az monitor log-analytics workspace list \
    --query "[].{name:name, rg:resourceGroup, sku:sku.name, retention:retentionInDays}" \
    -o table 2>/dev/null

# Active alerts
echo ""
echo "--- Active Alerts ---"
az monitor alert list \
    --query "[?essentials.monitorCondition=='Fired'].{
        name:name, severity:essentials.severity,
        target:essentials.targetResource,
        fired:essentials.startDateTime
    }" -o table 2>/dev/null | head -20

# Action groups
echo ""
echo "--- Action Groups ---"
for ag in $(az monitor action-group list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az monitor action-group list --query "[?name=='$ag'].resourceGroup" -o tsv | head -1)
    echo "  $ag ($RG)"
    
    RECEIVERS=$(az monitor action-group show -n "$ag" -g "$RG" \
        --query "{
            email:emailReceivers | length(@),
            sms:smsReceivers | length(@),
            webhook:webhookReceivers | length(@),
            function:azureFunctionReceivers | length(@)
        }" -o json 2>/dev/null)
    echo "    Receivers: $RECEIVERS"
done

# Alert rules
echo ""
echo "--- Metric Alert Rules ---"
az monitor metrics alert list \
    --query "[].{name:name, rg:resourceGroup, enabled:enabled, severity:severity}" \
    -o table 2>/dev/null | head -15

echo ""
echo "--- Log Alert Rules ---"
az monitor scheduled-query list \
    --query "[].{name:name, rg:resourceGroup, enabled:enabled}" \
    -o table 2>/dev/null | head -15

# 2. Resource diagnostic settings audit
#!/bin/bash
echo "=== Diagnostic Settings Audit ==="

RESOURCE_TYPES=(
    "Microsoft.Web/sites"
    "Microsoft.Sql/servers/databases"
    "Microsoft.ContainerService/managedClusters"
    "Microsoft.KeyVault/vaults"
    "Microsoft.Network/applicationGateways"
)

for type in "${RESOURCE_TYPES[@]}"; do
    SHORT=${type##*/}
    echo "--- $SHORT ---"
    
    for id in $(az resource list --resource-type "$type" --query "[].id" -o tsv 2>/dev/null); do
        NAME=$(echo "$id" | rev | cut -d'/' -f1 | rev)
        
        DIAG=$(az monitor diagnostic-settings list --resource "$id" \
            --query "value | length(@)" -o tsv 2>/dev/null)
        
        if [ "$DIAG" = "0" ] || [ -z "$DIAG" ]; then
            echo "  [NO DIAG] $NAME"
        else
            echo "  [OK]      $NAME ($DIAG settings)"
        fi
    done
done

# 3. KQL query runner
#!/bin/bash
echo "=== Run KQL Query ==="

WORKSPACE="${1}"
QUERY="${2}"

if [ -z "$WORKSPACE" ] || [ -z "$QUERY" ]; then
    echo "Usage: $0 <workspace-name> <kql-query>"
    echo "Example: $0 myLA 'Heartbeat | summarize count() by Computer'"
    exit 1
fi

RG=$(az monitor log-analytics workspace list \
    --query "[?name=='$WORKSPACE'].resourceGroup" -o tsv | head -1)

WORKSPACE_ID=$(az monitor log-analytics workspace show \
    -n "$WORKSPACE" -g "$RG" --query "customerId" -o tsv 2>/dev/null)

echo "Workspace: $WORKSPACE (ID: $WORKSPACE_ID)"
echo "Query: $QUERY"
echo ""

az monitor log-analytics query \
    --workspace "$WORKSPACE_ID" \
    --analytics-query "$QUERY" \
    -o table 2>/dev/null`,
				},
				{
					Title: "Application Insights and APM",
					Content: `Application Insights provides application performance monitoring for web applications, microservices, and distributed systems.

**Application Insights Setup:**
` + "```" + `
Create Application Insights:
  # Workspace-based (recommended)
  az monitor app-insights component create \
    --app myappinsights -g myRG \
    --location eastus \
    --workspace "/subscriptions/.../workspaces/myLA" \
    --application-type web
  
  # Get connection string
  az monitor app-insights component show \
    --app myappinsights -g myRG \
    --query "connectionString" -o tsv
  
  # Get instrumentation key (legacy)
  az monitor app-insights component show \
    --app myappinsights -g myRG \
    --query "instrumentationKey" -o tsv

Data types:
  Requests:       Incoming HTTP requests
  Dependencies:   Calls to external services (SQL, HTTP, etc.)
  Exceptions:     Unhandled/caught exceptions
  Traces:         Log messages (structured logging)
  PageViews:      Browser telemetry
  CustomEvents:   Business events you define
  CustomMetrics:  Application-specific metrics
  Availability:   Uptime test results

Auto-instrumentation (codeless):
  Supported platforms:
    .NET (IIS, App Service):  Fully automatic
    Java:                     Java agent
    Node.js:                  Auto-attach
    Python:                   OpenCensus
  
  App Service:
    az webapp config appsettings set -n myapp -g myRG \
      --settings APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
    
    az monitor app-insights component connect-webapp \
      --app myappinsights -g myRG \
      --web-app myapp
  
  AKS:
    # Enable monitoring add-on
    az aks enable-addons -a monitoring \
      -n myAKS -g myRG \
      --workspace-resource-id "/subscriptions/.../workspaces/myLA"

Distributed tracing:
  - Automatic correlation across services
  - W3C Trace Context standard
  - End-to-end transaction view
  - Application Map (dependency graph)
  - Operation ID links requests across services

Sampling:
  Adaptive sampling:
    Automatically adjusts volume to stay within target
    Default for .NET (5 events/sec)
  
  Fixed-rate sampling:
    Sample percentage of all telemetry
    Consistent across services (same operation sampled everywhere)
  
  Ingestion sampling:
    Applied at service endpoint
    Reduces stored data (and cost)
    Client still sends everything

Live Metrics:
  Real-time dashboard showing:
  - Request rate and duration
  - Failure rate
  - Dependency calls
  - CPU/memory usage
  - Live log stream
  - No sampling (all events)

Availability tests:
  URL ping test:
    - Simple HTTP GET
    - Up to 5 locations
    - 5-minute intervals
    - SSL certificate check
  
  Standard test:
    - HTTP request with custom headers
    - POST with body
    - Custom success criteria
    - SSL validation
  
  Custom TrackAvailability:
    - Programmatic availability tests
    - Complex multi-step scenarios
  
  az monitor app-insights web-test create \
    --name "homepage-check" -g myRG \
    --app-insights myappinsights \
    --location "us-il-ch1-azr" \
    --web-test-kind "ping" \
    --defined-web-test-name "Homepage" \
    --request-url "https://myapp.azurewebsites.net/" \
    --frequency 300 --timeout 120 \
    --expected-status-code 200
` + "```" + `

**Application Insights Analytics:**
` + "```" + `
Common queries:

  // Slow requests
  requests
  | where timestamp > ago(1h)
  | where duration > 5000
  | project timestamp, name, url, duration, resultCode
  | order by duration desc
  | take 20
  
  // Failed requests
  requests
  | where timestamp > ago(24h)
  | where success == false
  | summarize count() by name, resultCode
  | order by count_ desc
  
  // Dependency failures
  dependencies
  | where timestamp > ago(24h)
  | where success == false
  | summarize count() by target, name, resultCode
  | order by count_ desc
  
  // Exception trends
  exceptions
  | where timestamp > ago(7d)
  | summarize count() by bin(timestamp, 1h), type
  | render timechart
  
  // End-to-end transaction
  union requests, dependencies, exceptions
  | where operation_Id == "abc123"
  | order by timestamp asc
  | project timestamp, itemType, name, duration, success, resultCode
  
  // User sessions
  pageViews
  | where timestamp > ago(24h)
  | summarize pageCount=count(), duration=avg(duration)
    by user_Id, session_Id
  | summarize sessions=dcount(session_Id), avgPages=avg(pageCount)
  
  // Performance buckets
  requests
  | where timestamp > ago(1h)
  | summarize count() by performanceBucket
  | order by performanceBucket asc
  
  // Dependency map data
  dependencies
  | where timestamp > ago(1h)
  | summarize calls=count(), avgDuration=avg(duration),
    failures=countif(success == false)
    by target, type
  | order by calls desc

Smart Detection alerts:
  Automatically detects:
  - Abnormal failure rates
  - Abnormal rise in exception volume
  - Memory leak potential
  - Slow page load time
  - Slow server response time
  - Degradation in dependency duration
  - Degradation in trace severity ratio

Workbooks:
  Interactive visual reports
  - Combine KQL queries with visualizations
  - Parameters and filters
  - Links and drill-downs
  - Share across team
  - Gallery of templates
` + "```" + ``,
					CodeExamples: `# Application Insights management

# 1. App Insights overview
#!/bin/bash
echo "=== Application Insights Overview ==="

for app in $(az monitor app-insights component list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az monitor app-insights component list \
        --query "[?name=='$app'].resourceGroup" -o tsv | head -1)
    
    echo "App: $app ($RG)"
    
    APP_ID=$(az monitor app-insights component show \
        --app "$app" -g "$RG" --query "appId" -o tsv 2>/dev/null)
    
    # Request metrics (last 24h)
    echo "  Requests (24h):"
    az monitor app-insights metrics show \
        --app "$app" -g "$RG" \
        --metric "requests/count" \
        --interval PT24H \
        --query "value.\"requests/count\".sum" -o tsv 2>/dev/null | xargs -I {} echo "    Total: {}"
    
    az monitor app-insights metrics show \
        --app "$app" -g "$RG" \
        --metric "requests/failed" \
        --interval PT24H \
        --query "value.\"requests/failed\".sum" -o tsv 2>/dev/null | xargs -I {} echo "    Failed: {}"
    
    # Availability
    echo "  Availability tests:"
    az monitor app-insights web-test list -g "$RG" \
        --query "[?contains(id, '$app')].{name:name, enabled:enabled}" \
        -o table 2>/dev/null
    
    echo ""
done

# 2. Performance report
#!/bin/bash
echo "=== Performance Report ==="

APP="${1:-myappinsights}"
RG="${2:-myRG}"

echo "App: $APP"

# Response time percentiles
echo "--- Response Time (last 1h) ---"
az monitor app-insights metrics show \
    --app "$APP" -g "$RG" \
    --metric "requests/duration" \
    --interval PT1H \
    --aggregation avg min max \
    -o json 2>/dev/null | jq '.value["requests/duration"]'

# Top slow endpoints
echo ""
echo "--- Slowest Endpoints ---"
az monitor app-insights query \
    --app "$APP" -g "$RG" \
    --analytics-query "
        requests
        | where timestamp > ago(1h)
        | summarize avg_duration=avg(duration), p95=percentile(duration, 95), count=count()
          by name
        | order by avg_duration desc
        | take 10
    " -o table 2>/dev/null

# Dependency performance
echo ""
echo "--- Dependency Performance ---"
az monitor app-insights query \
    --app "$APP" -g "$RG" \
    --analytics-query "
        dependencies
        | where timestamp > ago(1h)
        | summarize avg_duration=avg(duration), failures=countif(success==false), calls=count()
          by target, type
        | order by avg_duration desc
        | take 10
    " -o table 2>/dev/null

# 3. Exception analysis
#!/bin/bash
echo "=== Exception Analysis ==="

APP="${1:-myappinsights}"
RG="${2:-myRG}"

echo "--- Top Exceptions (24h) ---"
az monitor app-insights query \
    --app "$APP" -g "$RG" \
    --analytics-query "
        exceptions
        | where timestamp > ago(24h)
        | summarize count=count() by type, outerMessage
        | order by count desc
        | take 15
    " -o table 2>/dev/null

echo ""
echo "--- Exception Trend ---"
az monitor app-insights query \
    --app "$APP" -g "$RG" \
    --analytics-query "
        exceptions
        | where timestamp > ago(7d)
        | summarize count() by bin(timestamp, 1h)
        | order by timestamp desc
        | take 24
    " -o table 2>/dev/null`,
				},
			},
		},
	})
}
