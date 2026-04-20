package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1462,
			Title:       "Site Reliability Engineering Practices",
			Description: "Implement SRE practices including SLOs, error budgets, incident management, chaos engineering, and reliability patterns for production systems.",
			Order:       62,
			Lessons: []problems.Lesson{
				{
					Title: "SLOs, Error Budgets, and Reliability",
					Content: `Site Reliability Engineering (SRE) provides a framework for running reliable production systems through service level objectives and error budgets.

**Service Level Indicators and Objectives:**
` + "```" + `
Terminology:
  SLI (Service Level Indicator):
    Quantitative measure of service behavior.
    Examples: latency, error rate, throughput, availability.
  
  SLO (Service Level Objective):
    Target value for an SLI.
    Example: 99.9% of requests complete in <300ms.
  
  SLA (Service Level Agreement):
    Business agreement with consequences.
    Example: 99.9% uptime or credits issued.
  
  Error Budget:
    Amount of unreliability allowed.
    100% - SLO = Error budget
    99.9% SLO → 0.1% error budget → 43.8 min/month

Common SLIs:
  Availability:
    successful_requests / total_requests
    
    sum(rate(http_requests_total{status!~"5.."}[30d]))
    / sum(rate(http_requests_total[30d]))
  
  Latency:
    Proportion of requests faster than threshold.
    
    histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
  
  Throughput:
    Requests per second.
    
    sum(rate(http_requests_total[5m]))
  
  Correctness:
    Proportion of correct responses.
    (Requires application-specific validation.)
  
  Freshness:
    How recent is the data?
    (For data pipelines, caches.)

Error budget calculations:
  SLO: 99.9% availability
  
  Monthly error budget:
    43,200 minutes × 0.1% = 43.2 minutes downtime
  
  Quarterly error budget:
    129,600 minutes × 0.1% = 129.6 minutes downtime
  
  Annual error budget:
    525,600 minutes × 0.1% = 525.6 minutes downtime

  Error budget tracking:
    remaining_budget = error_budget - consumed_errors
    
    consumed_30d = 1 - (
      sum(rate(http_requests_total{status!~"5.."}[30d]))
      / sum(rate(http_requests_total[30d]))
    )
    
    budget_remaining = 0.001 - consumed_30d  # for 99.9% SLO

Error budget policy:
  Budget remaining > 50%:
    Normal development velocity
    Deploy new features freely
  
  Budget remaining 25-50%:
    Increased caution
    Extra testing for changes
    No risky deployments
  
  Budget remaining < 25%:
    Reduce deployment velocity
    Focus on reliability improvements
    Require extra review for changes
  
  Budget exhausted (0%):
    Feature freeze
    All engineering on reliability
    Only critical fixes deployed
    Post-mortem required

Multi-window error budget:
  Fast burn:  2% budget consumed in 1 hour
              → Page (immediate attention)
  
  Slow burn:  5% budget consumed in 6 hours
              → Page (during business hours)
  
  Slow burn:  10% budget consumed in 3 days
              → Ticket (follow-up needed)
` + "```" + `

**SLO-based Alerting:**
` + "```" + `
Burn rate alerting:
  Traditional alert:
    Alert when error rate > 0.1%
    Problem: Too sensitive, alert fatigue
  
  Burn rate alert:
    Alert when burning budget too fast
    Budget awareness built into alerting
  
  Multi-window, multi-burn-rate:
    # Fast burn (page)
    - alert: ErrorBudgetFastBurn
      expr: |
        (
          sum(rate(http_requests_total{status=~"5.."}[1h]))
          / sum(rate(http_requests_total[1h]))
        ) > (14.4 * 0.001)
        and
        (
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m]))
        ) > (14.4 * 0.001)
      for: 2m
      labels:
        severity: critical
    
    # Slow burn (ticket)
    - alert: ErrorBudgetSlowBurn
      expr: |
        (
          sum(rate(http_requests_total{status=~"5.."}[6h]))
          / sum(rate(http_requests_total[6h]))
        ) > (6 * 0.001)
        and
        (
          sum(rate(http_requests_total{status=~"5.."}[30m]))
          / sum(rate(http_requests_total[30m]))
        ) > (6 * 0.001)
      for: 5m
      labels:
        severity: warning

Latency SLO alerting:
  # P99 latency SLO: <500ms for 99.9% of requests
  - alert: LatencySLOViolation
    expr: |
      histogram_quantile(0.99,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
      ) > 0.5
    for: 10m
    labels:
      severity: warning

On-call best practices:
  Rotation:
    - 1-week rotations (min)
    - 2-person on-call (primary + secondary)
    - Follow-the-sun for global teams
    - Handoff meeting between rotations
  
  Alert quality:
    - Every alert should be actionable
    - Remove alerts with >50% false positive rate
    - Target <2 pages per on-call shift
    - Automate routine responses
  
  Runbooks:
    For each alert:
    - What it means
    - Impact and urgency
    - Investigation steps
    - Common fixes
    - Escalation path
    - Related dashboards
` + "```" + ``,
					CodeExamples: `# SRE practice scripts

# 1. SLO dashboard
#!/bin/bash
echo "=== SLO Status Dashboard ==="

PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"

query_prom() {
    local query="$1"
    curl -s "$PROM_URL/api/v1/query" --data-urlencode "query=$query" 2>/dev/null | \
        jq -r '.data.result[0].value[1] // "N/A"' 2>/dev/null
}

# Availability SLO
echo "--- Availability (30d) ---"
AVAIL=$(query_prom '
    sum(rate(http_requests_total{status!~"5.."}[30d]))
    / sum(rate(http_requests_total[30d]))
')
if [ "$AVAIL" != "N/A" ]; then
    AVAIL_PCT=$(echo "scale=4; $AVAIL * 100" | bc 2>/dev/null)
    echo "  Current: ${AVAIL_PCT}%"
    echo "  SLO Target: 99.9%"
    BUDGET=$(echo "scale=4; ($AVAIL - 0.999) * 43200" | bc 2>/dev/null)
    echo "  Budget Remaining: ${BUDGET} minutes"
fi

# Latency SLO
echo ""
echo "--- Latency P99 (30d) ---"
P99=$(query_prom '
    histogram_quantile(0.99,
        sum(rate(http_request_duration_seconds_bucket[30d])) by (le)
    )
')
if [ "$P99" != "N/A" ]; then
    P99_MS=$(echo "scale=1; $P99 * 1000" | bc 2>/dev/null)
    echo "  Current P99: ${P99_MS}ms"
    echo "  SLO Target: <500ms"
fi

# Error rate trend
echo ""
echo "--- Error Rate (hourly trend) ---"
for i in 0 1 2 3 4 5; do
    OFFSET="${i}h"
    RATE=$(query_prom "sum(rate(http_requests_total{status=~\"5..\"}[1h] offset ${OFFSET})) / sum(rate(http_requests_total[1h] offset ${OFFSET}))")
    if [ "$RATE" != "N/A" ]; then
        RATE_PCT=$(echo "scale=4; $RATE * 100" | bc 2>/dev/null)
        echo "  -${i}h: ${RATE_PCT}%"
    fi
done

# 2. Incident timer
#!/bin/bash
echo "=== Incident Timer ==="

INCIDENT_START="${1:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
SLO_BUDGET_MINUTES="${2:-43}"  # 99.9% monthly

NOW=$(date +%s)
START=$(date -d "$INCIDENT_START" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$INCIDENT_START" +%s 2>/dev/null)

if [ -n "$START" ]; then
    DURATION_SEC=$((NOW - START))
    DURATION_MIN=$((DURATION_SEC / 60))
    
    echo "  Incident start: $INCIDENT_START"
    echo "  Duration: ${DURATION_MIN} minutes"
    echo "  Monthly error budget: ${SLO_BUDGET_MINUTES} minutes"
    REMAINING=$((SLO_BUDGET_MINUTES - DURATION_MIN))
    echo "  Budget remaining: ${REMAINING} minutes"
    
    if [ "$REMAINING" -lt 0 ]; then
        echo "  STATUS: ERROR BUDGET EXHAUSTED"
    elif [ "$REMAINING" -lt $((SLO_BUDGET_MINUTES / 4)) ]; then
        echo "  STATUS: BUDGET CRITICAL (<25%)"
    else
        echo "  STATUS: BUDGET OK"
    fi
fi

# 3. On-call handoff report
#!/bin/bash
echo "=== On-Call Handoff Report ==="
echo "Date: $(date)"

# Active incidents
echo ""
echo "--- Active Incidents ---"
# (Would query incident management tool)
echo "  Check PagerDuty/OpsGenie for active incidents"

# Alert summary
echo ""
echo "--- Alert Summary (last 7 days) ---"
PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"
curl -s "$PROM_URL/api/v1/query" \
    --data-urlencode 'query=count(ALERTS{alertstate="firing"}) by (alertname)' \
    2>/dev/null | jq -r '.data.result[] | "  \(.metric.alertname): firing"' 2>/dev/null

# Recent deployments
echo ""
echo "--- Recent Deployments ---"
kubectl get deployments --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.metadata.annotations["deployment.kubernetes.io/revision"]) | 
    "\(.metadata.namespace)/\(.metadata.name): revision \(.metadata.annotations["deployment.kubernetes.io/revision"])"' 2>/dev/null | head -10

# Known issues
echo ""
echo "--- Known Issues ---"
echo "  (Document ongoing issues here)"`,
				},
				{
					Title: "Chaos Engineering and Incident Management",
					Content: `Chaos engineering proactively tests system resilience, while incident management provides structured response to production issues.

**Chaos Engineering:**
` + "```" + `
Principles:
  1. Hypothesize about steady state
  2. Vary real-world events
  3. Run experiments in production
  4. Automate experiments
  5. Minimize blast radius

Chaos experiments:
  Infrastructure:
    - Kill random pods/instances
    - Network latency injection
    - DNS failures
    - Disk fill
    - CPU/memory stress
    - AZ/Region failures
  
  Application:
    - Dependency failures
    - Database performance degradation
    - Cache failures
    - Queue saturation
    - Certificate expiry
  
  Human:
    - Game days
    - Disaster recovery drills
    - Runbook validation

Chaos Mesh (Kubernetes):
  # Install
  helm install chaos-mesh chaos-mesh/chaos-mesh \
    --namespace chaos-testing --create-namespace
  
  # Pod kill experiment
  apiVersion: chaos-mesh.org/v1alpha1
  kind: PodChaos
  metadata:
    name: pod-kill
  spec:
    action: pod-kill
    mode: one
    selector:
      namespaces: [production]
      labelSelectors:
        app: api-server
    scheduler:
      cron: "@every 2h"
  
  # Network delay
  apiVersion: chaos-mesh.org/v1alpha1
  kind: NetworkChaos
  metadata:
    name: network-delay
  spec:
    action: delay
    mode: all
    selector:
      namespaces: [production]
      labelSelectors:
        app: payment-service
    delay:
      latency: "200ms"
      correlation: "50"
      jitter: "50ms"
    duration: "5m"
  
  # IO chaos
  apiVersion: chaos-mesh.org/v1alpha1
  kind: IOChaos
  metadata:
    name: io-delay
  spec:
    action: latency
    mode: one
    selector:
      labelSelectors:
        app: database
    volumePath: /var/lib/postgresql
    path: ""
    delay: "100ms"
    percent: 50
    duration: "10m"

Litmus Chaos:
  # ChaosEngine
  apiVersion: litmuschaos.io/v1alpha1
  kind: ChaosEngine
  metadata:
    name: nginx-chaos
  spec:
    appinfo:
      appns: default
      applabel: "app=nginx"
      appkind: deployment
    chaosServiceAccount: litmus-admin
    experiments:
      - name: pod-delete
        spec:
          components:
            env:
              - name: TOTAL_CHAOS_DURATION
                value: "30"
              - name: CHAOS_INTERVAL
                value: "10"

Steady state hypothesis:
  Before experiment:
    - Define "normal" metrics
    - Error rate < 0.1%
    - P99 latency < 500ms
    - All health checks passing
  
  During experiment:
    - Monitor metrics continuously
    - Auto-abort if safety thresholds exceeded
  
  After experiment:
    - Verify system recovered
    - Check metrics returned to normal
    - Document findings
` + "```" + `

**Incident Management:**
` + "```" + `
Incident lifecycle:
  Detection:
    - Automated monitoring alerts
    - Customer reports
    - Internal discovery
  
  Triage:
    - Severity assessment (SEV1-SEV4)
    - Impact scope (users, revenue)
    - Initial response team
  
  Response:
    - Incident commander assigned
    - Communication channel opened
    - Investigation begins
    - Status page updated
  
  Mitigation:
    - Apply immediate fix (rollback, scale, redirect)
    - Communicate resolution
    - Verify system stability
  
  Resolution:
    - Root cause confirmed
    - Permanent fix deployed
    - Incident closed
  
  Post-incident:
    - Blameless post-mortem
    - Action items created
    - Lessons shared

Severity levels:
  SEV1 (Critical):
    Impact: Service down, major data loss
    Response: All hands, 15-min updates
    Example: Website completely down
  
  SEV2 (Major):
    Impact: Significant degradation, partial outage
    Response: On-call team, 30-min updates
    Example: Payment processing failing for 30% of users
  
  SEV3 (Minor):
    Impact: Minor degradation, workaround exists
    Response: Next business day
    Example: Image upload slow but functional
  
  SEV4 (Low):
    Impact: Cosmetic, no user impact
    Response: Backlog
    Example: Admin dashboard graph not loading

Incident commander responsibilities:
  - Overall coordination
  - Delegate tasks
  - Communicate status
  - Maintain timeline
  - Decide escalation
  - Call for additional resources
  - Determine resolution criteria

Post-mortem template:
  # Incident Post-Mortem
  
  ## Summary
  What happened, when, how long, impact
  
  ## Timeline
  HH:MM - Event/action (who)
  
  ## Root Cause
  The actual underlying cause
  
  ## Impact
  - Users affected: X
  - Revenue impact: $Y
  - Duration: Z minutes
  
  ## What Went Well
  - Detection time
  - Communication
  - Team response
  
  ## What Could Be Improved
  - Monitoring gaps
  - Process issues
  - Documentation
  
  ## Action Items
  | ID | Action | Owner | Priority | Due |
  |----|--------|-------|----------|-----|
  | 1  | Add alert for X | @oncall | P1 | 1 week |
  | 2  | Update runbook | @team | P2 | 2 weeks |
  
  Blameless culture:
  - Focus on systems, not individuals
  - "How did the system allow this?"
  - Improve defenses, not assign blame
  - Psychological safety encourages honesty
` + "```" + ``,
					CodeExamples: `# Incident management scripts

# 1. Incident response toolkit
#!/bin/bash
echo "=== Incident Response Toolkit ==="

SEVERITY="${1:-SEV2}"
INCIDENT_ID="INC-$(date +%Y%m%d%H%M%S)"

echo "Incident ID: $INCIDENT_ID"
echo "Severity: $SEVERITY"
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Quick system status
echo "--- System Status ---"
echo "Kubernetes:"
kubectl get nodes 2>/dev/null | head -5
echo ""
kubectl get pods --all-namespaces --field-selector='status.phase!=Running' 2>/dev/null | head -10

echo ""
echo "--- Recent Events ---"
kubectl get events --sort-by='.lastTimestamp' --all-namespaces 2>/dev/null | tail -10

echo ""
echo "--- Recent Deployments ---"
kubectl get deployments --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | "\(.metadata.namespace)/\(.metadata.name)"' 2>/dev/null | head -5

echo ""
echo "--- Resource Usage ---"
kubectl top nodes 2>/dev/null
echo ""
kubectl top pods --all-namespaces --sort-by=cpu 2>/dev/null | head -10

# 2. Chaos experiment runner
#!/bin/bash
echo "=== Chaos Experiment ==="

EXPERIMENT="${1:-pod-kill}"
NAMESPACE="${2:-default}"
LABEL="${3:-app=web}"
DURATION="${4:-60}"

echo "Experiment: $EXPERIMENT"
echo "Target: $LABEL in $NAMESPACE"
echo "Duration: ${DURATION}s"

# Pre-experiment check
echo ""
echo "--- Pre-experiment State ---"
kubectl get pods -n "$NAMESPACE" -l "$LABEL" 2>/dev/null
echo ""
echo "Healthy pods:"
kubectl get pods -n "$NAMESPACE" -l "$LABEL" --field-selector='status.phase=Running' --no-headers 2>/dev/null | wc -l | tr -d ' '

case "$EXPERIMENT" in
    pod-kill)
        POD=$(kubectl get pods -n "$NAMESPACE" -l "$LABEL" -o name 2>/dev/null | shuf | head -1)
        if [ -n "$POD" ]; then
            echo ""
            echo "Killing: $POD"
            kubectl delete "$POD" -n "$NAMESPACE" --grace-period=0 2>/dev/null
        fi
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        ;;
esac

# Wait and check recovery
echo ""
echo "Waiting ${DURATION}s for recovery..."
sleep "$DURATION"

echo ""
echo "--- Post-experiment State ---"
kubectl get pods -n "$NAMESPACE" -l "$LABEL" 2>/dev/null
echo ""
echo "Healthy pods:"
kubectl get pods -n "$NAMESPACE" -l "$LABEL" --field-selector='status.phase=Running' --no-headers 2>/dev/null | wc -l | tr -d ' '

# 3. Post-mortem generator
#!/bin/bash
echo "=== Post-Mortem Template ==="

INCIDENT_ID="${1:-INC-UNKNOWN}"
START="${2:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

cat << EOF
# Post-Mortem: $INCIDENT_ID

## Summary
[What happened, impact, duration]

## Timeline
- $START - Incident detected
- [HH:MM] - [Event]
- [HH:MM] - [Resolution]

## Root Cause
[Why this happened]

## Impact
- Users affected: [number]
- Duration: [minutes]
- Error budget consumed: [percentage]

## What Went Well
- [Item]

## What Could Be Improved  
- [Item]

## Action Items
| ID | Action | Owner | Priority | Due |
|----|--------|-------|----------|-----|
| 1  | [action] | @[owner] | P[1-3] | [date] |
EOF`,
				},
			},
		},
	})
}
