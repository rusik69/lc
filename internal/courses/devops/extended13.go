package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1474,
			Title:       "Site Reliability Engineering Advanced Practices",
			Description: "Advanced SRE practices including toil reduction, capacity planning, production readiness reviews, and on-call management.",
			Order:       74,
			Lessons: []problems.Lesson{
				{
					Title: "Toil Reduction and Automation Strategy",
					Content: `Toil is manual, repetitive, automatable work that scales linearly with service growth. SRE aims to keep toil below 50% of total work.

**Identifying and Measuring Toil:**
` + "```" + `
Toil characteristics:
  Manual: Requires human action
  Repetitive: Done over and over
  Automatable: Could be done by software
  Tactical: Interrupt-driven, reactive
  No enduring value: Doesn't improve service permanently
  Scales with growth: More load = more toil

Toil measurement:
  Track time spent on:
    Ticket handling
    Manual deployments
    Certificate renewals
    Account provisioning
    Capacity adjustments
    Incident response (non-novel)
    Configuration changes
    Backup verification
    Log rotation
    SSL certificate management
  
  Metrics:
    Toil hours per week per engineer
    Percentage of time spent on toil
    Number of manual interventions
    Ticket volume by category
    Time per ticket resolution
  
  Toil budget:
    Target: <50% of SRE time on toil
    Track: Weekly toil surveys
    Alert: When toil exceeds budget
    Action: Prioritize automation projects

Automation priority matrix:
  High frequency + High time per task = Automate first
  High frequency + Low time = Automate second
  Low frequency + High time = Document well, automate third
  Low frequency + Low time = Document, may not automate

Common automation targets:
  Self-healing:
    Pod restart on health check failure ✓ (Kubernetes native)
    Auto-remediation of known issues
    Circuit breakers for cascading failures
    Automatic failover for databases
  
  Provisioning:
    New service creation → scaffolding tool
    Database provisioning → Crossplane/Operators
    Environment creation → Terraform modules
    User access → RBAC automation
  
  Operational:
    Certificate rotation → cert-manager
    Secret rotation → External Secrets Operator
    Log rotation → Fluentd/logrotate
    Backup management → CronJobs
    Capacity scaling → HPA/VPA/KEDA

Runbook automation:
  Convert manual runbooks to executable scripts.
  
  Levels:
    L0: Fully manual with documentation
    L1: Semi-automated (human triggers, script executes)
    L2: Automated with human approval
    L3: Fully automated (detect → remediate → notify)
  
  Example progression:
    L0: "SSH to server, restart service, check logs"
    L1: Script that restarts and checks, triggered by human
    L2: Alert triggers script, human approves
    L3: Health check fails → auto-restart → alert if fails again
` + "```" + `

**Capacity Planning:**
` + "```" + `
Capacity planning process:
  1. Measure current utilization
  2. Model growth trends
  3. Forecast future demand
  4. Plan capacity additions
  5. Validate with load testing

Demand forecasting:
  Historical analysis:
    Look at 3-12 months of data
    Identify growth patterns
    Account for seasonality
    Include known upcoming events
  
  Metrics to track:
    CPU utilization trends
    Memory usage trends
    Storage growth rate
    Network bandwidth usage
    Request rate growth
    User growth rate
    Data volume growth
  
  Forecasting methods:
    Linear regression: Steady growth
    Exponential: Rapid growth
    Seasonal decomposition: Periodic patterns
    Capacity models: Service-specific calculations

Prometheus capacity queries:
  Current utilization:
    CPU: avg(rate(container_cpu_usage_seconds_total[5m])) by (namespace)
    Memory: sum(container_memory_working_set_bytes) by (namespace)
    Storage: kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes
  
  Growth rate (per day):
    deriv(sum(container_memory_working_set_bytes)[30d:1h])
  
  Days until full:
    (kubelet_volume_stats_capacity_bytes - kubelet_volume_stats_used_bytes) /
    deriv(kubelet_volume_stats_used_bytes[30d:1h]) / 86400
  
  Headroom:
    1 - (sum(kube_pod_container_resource_requests{resource="cpu"}) /
    sum(kube_node_status_allocatable{resource="cpu"}))

Production Readiness Review (PRR):
  Checklist before production launch:
  
  Architecture:
    [ ] Architecture documented
    [ ] Dependencies identified
    [ ] Single points of failure addressed
    [ ] Failure modes documented
  
  Reliability:
    [ ] SLOs defined
    [ ] Error budgets configured
    [ ] Graceful degradation implemented
    [ ] Load testing completed
    [ ] Chaos testing performed
  
  Operability:
    [ ] Monitoring dashboards created
    [ ] Alerts configured
    [ ] Runbooks written
    [ ] On-call rotation set up
    [ ] Incident response plan documented
  
  Security:
    [ ] Security review completed
    [ ] Vulnerabilities scanned
    [ ] Access controls configured
    [ ] Data encryption verified
    [ ] Compliance requirements met
  
  Scalability:
    [ ] Auto-scaling configured
    [ ] Capacity plan documented
    [ ] Performance benchmarks established
    [ ] Resource limits set
  
  Data:
    [ ] Backup strategy implemented
    [ ] Recovery tested
    [ ] Data retention policy defined
    [ ] Migration plan ready
` + "```" + ``,
					CodeExamples: `# SRE automation scripts

# 1. Toil tracker
#!/bin/bash
echo "=== Toil Tracking Report ==="

TOIL_LOG="${TOIL_LOG:-$HOME/.toil-log}"

case "${1:-report}" in
    log)
        CATEGORY="${2:?Category required (deploy|incident|provision|config|other)}"
        MINUTES="${3:?Minutes required}"
        DESCRIPTION="${4:-No description}"
        
        echo "$(date +%Y-%m-%d),$(date +%H:%M),$CATEGORY,$MINUTES,$DESCRIPTION" >> "$TOIL_LOG"
        echo "Logged: $CATEGORY, ${MINUTES}min - $DESCRIPTION"
        ;;
    
    report)
        PERIOD="${2:-7}"  # days
        CUTOFF=$(date -v-${PERIOD}d +%Y-%m-%d 2>/dev/null || date -d "-${PERIOD} days" +%Y-%m-%d)
        
        echo "Period: Last $PERIOD days (since $CUTOFF)"
        echo ""
        
        if [ ! -f "$TOIL_LOG" ]; then
            echo "No toil data. Log with: $0 log <category> <minutes> [description]"
            exit 0
        fi
        
        echo "--- By Category ---"
        awk -F',' -v cutoff="$CUTOFF" '$1 >= cutoff {
            cat[$3] += $4
            total += $4
        } END {
            for (c in cat) printf "  %-15s %4d min (%5.1f%%)\n", c, cat[c], cat[c]/total*100
            printf "  %-15s %4d min\n", "TOTAL", total
        }' "$TOIL_LOG"
        
        echo ""
        echo "--- Daily Trend ---"
        awk -F',' -v cutoff="$CUTOFF" '$1 >= cutoff {
            day[$1] += $4
        } END {
            for (d in day) printf "  %s: %d min\n", d, day[d]
        }' "$TOIL_LOG" | sort
        
        echo ""
        TOTAL_MIN=$(awk -F',' -v cutoff="$CUTOFF" '$1 >= cutoff {sum+=$4} END {print sum+0}' "$TOIL_LOG")
        WORK_HOURS=$((PERIOD * 8))
        TOIL_PCT=$(echo "scale=1; $TOTAL_MIN / ($WORK_HOURS * 60) * 100" | bc 2>/dev/null || echo "N/A")
        echo "Toil percentage: ${TOIL_PCT}% (target: <50%)"
        ;;
    
    *)
        echo "Usage: $0 <log|report> [options]"
        echo "  log <category> <minutes> [description]"
        echo "  report [days]"
        ;;
esac

# 2. Capacity planner
#!/bin/bash
echo "=== Capacity Planning Report ==="

echo "--- Current Cluster Utilization ---"

# Node capacity
echo ""
echo "Nodes:"
kubectl get nodes -o json 2>/dev/null | \
    jq -r '.items[] | "\(.metadata.name): CPU=\(.status.allocatable.cpu) Mem=\(.status.allocatable.memory)"' 2>/dev/null

# Namespace resource usage
echo ""
echo "Namespace Resource Usage:"
echo "Namespace | CPU Req | CPU Lim | Mem Req | Mem Lim"
echo "----------|---------|---------|---------|--------"

for ns in $(kubectl get ns -o name 2>/dev/null | cut -d'/' -f2 | grep -v "^kube-"); do
    STATS=$(kubectl get pods -n "$ns" -o json 2>/dev/null | \
        jq -r '[.items[].spec.containers[] | {
            cpu_req: (.resources.requests.cpu // "0"),
            cpu_lim: (.resources.limits.cpu // "0"),
            mem_req: (.resources.requests.memory // "0"),
            mem_lim: (.resources.limits.memory // "0")
        }] | {
            cpu_req: [.[].cpu_req] | join("+"),
            cpu_lim: [.[].cpu_lim] | join("+"),
            mem_req: [.[].mem_req] | join("+"),
            mem_lim: [.[].mem_lim] | join("+")
        } | "\(.cpu_req)|\(.cpu_lim)|\(.mem_req)|\(.mem_lim)"' 2>/dev/null)
    
    if [ -n "$STATS" ] && [ "$STATS" != "|||" ]; then
        echo "$ns | $STATS" | tr '|' ' | '
    fi
done

# Cluster-wide
echo ""
echo "--- Cluster Summary ---"
TOTAL_CPU=$(kubectl get nodes -o json 2>/dev/null | \
    jq '[.items[].status.allocatable.cpu | rtrimstr("m") | 
    if test("^[0-9]+$") then tonumber * 1000 else tonumber end] | add // 0')
REQUESTED_CPU=$(kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq '[.items[].spec.containers[].resources.requests.cpu // "0" |
    if endswith("m") then rtrimstr("m") | tonumber
    elif test("^[0-9.]+$") then tonumber * 1000
    else 0 end] | add // 0')

if [ -n "$TOTAL_CPU" ] && [ "$TOTAL_CPU" -gt 0 ] 2>/dev/null; then
    CPU_UTIL=$(echo "scale=1; $REQUESTED_CPU * 100 / $TOTAL_CPU" | bc 2>/dev/null || echo "N/A")
    echo "CPU: ${REQUESTED_CPU}m / ${TOTAL_CPU}m (${CPU_UTIL}% requested)"
fi

# PVC usage
echo ""
echo "--- Storage Usage ---"
kubectl get pvc --all-namespaces --no-headers 2>/dev/null | \
    awk '{printf "%s/%s: %s (%s)\n", $1, $2, $4, $6}'

# 3. On-call handoff generator
#!/bin/bash
echo "=== On-Call Handoff Report ==="
echo "Generated: $(date)"
echo "Period: Last 7 days"
echo ""

# Recent incidents
echo "--- Incidents ---"
echo "(Check your incident management tool)"

# Active alerts
echo ""
echo "--- Active Alerts ---"
if command -v amtool &>/dev/null; then
    amtool alert query --alertmanager.url=http://localhost:9093 2>/dev/null | head -10
else
    echo "Connect to Alertmanager for active alerts"
fi

# Recent deployments
echo ""
echo "--- Recent Deployments ---"
kubectl get events --all-namespaces --sort-by='.lastTimestamp' 2>/dev/null | \
    grep -i "pulled\|scaled\|created" | tail -10

# Known issues
echo ""
echo "--- Known Issues ---"
echo "  Check team wiki/runbook for ongoing issues"

# Upcoming changes
echo ""
echo "--- Upcoming Changes ---"
echo "  Check change management calendar"

# Action items
echo ""
echo "--- Action Items for Next On-Call ---"
echo "  1. Monitor [specific service] after recent deploy"
echo "  2. Follow up on [specific issue]"
echo "  3. Check backup verification results"`,
				},
				{
					Title: "Incident Management and Post-Incident Learning",
					Content: `Structured incident management minimizes impact and drives continuous improvement through blameless post-mortems.

**Incident Response Framework:**
` + "```" + `
Incident severity levels:
  SEV1 (Critical):
    Complete service outage or data loss
    All customers affected
    Revenue impact
    Response: Immediate, all-hands
    Communication: Every 15 minutes
  
  SEV2 (Major):
    Significant degradation
    Many customers affected
    Partial functionality lost
    Response: Within 15 minutes
    Communication: Every 30 minutes
  
  SEV3 (Minor):
    Limited impact
    Workaround available
    Few customers affected
    Response: Within 1 hour
    Communication: Every 2 hours
  
  SEV4 (Low):
    Minimal impact
    Cosmetic issues
    Single customer affected
    Response: Next business day

Incident roles:
  Incident Commander (IC):
    Coordinates response
    Makes decisions
    Manages communication
    Delegates tasks
  
  Operations Lead:
    Executes technical remediation
    Coordinates with engineering
    Implements fixes
  
  Communications Lead:
    Updates status page
    Notifies stakeholders
    Manages customer communication
  
  Scribe:
    Documents timeline
    Records actions taken
    Logs decisions made

Incident lifecycle:
  Detection:
    Monitoring alerts
    Customer reports
    Automated health checks
  
  Triage:
    Assess severity
    Assign IC
    Open incident channel
    Start timeline
  
  Mitigation:
    Identify root cause
    Implement fix or workaround
    Verify resolution
    Monitor for recurrence
  
  Resolution:
    Confirm service restored
    Update status page
    Notify stakeholders
    Schedule post-mortem
  
  Post-incident:
    Write post-mortem
    Identify action items
    Share learnings
    Track follow-ups

Incident communication templates:
  Initial:
    "We are investigating reports of [issue].
     Impact: [description]
     Next update in [X] minutes."
  
  Update:
    "Update on [issue]:
     Status: [investigating/identified/monitoring]
     Impact: [current impact]
     Actions: [what we're doing]
     Next update in [X] minutes."
  
  Resolution:
    "Resolved: [issue] has been resolved.
     Duration: [start] to [end]
     Impact: [summary]
     Root cause: [brief description]
     Post-mortem will be shared within [X] days."
` + "```" + `

**Blameless Post-Mortems:**
` + "```" + `
Post-mortem template:
  Title: [Service] Outage on [Date]
  Author: [Name]
  Date: [Post-mortem date]
  Severity: SEV[X]
  Duration: [Start time] to [End time] ([duration])
  
  Summary:
    Brief description of what happened.
  
  Impact:
    Users affected: [number/percentage]
    Revenue impact: [if applicable]
    SLA impact: [error budget consumed]
    Customer-facing effects: [description]
  
  Timeline:
    HH:MM - Alert fired for [condition]
    HH:MM - IC assigned, incident channel opened
    HH:MM - Root cause identified
    HH:MM - Fix deployed
    HH:MM - Service restored
    HH:MM - Monitoring confirmed stable
  
  Root Cause:
    Detailed technical explanation of what caused the incident.
    Include contributing factors.
  
  Detection:
    How was the incident detected?
    Could it have been detected earlier?
    Detection time: [X] minutes after onset
  
  Resolution:
    What actions resolved the incident?
    What was the fix?
  
  Lessons Learned:
    What went well:
      - [Positive aspects of response]
    What went poorly:
      - [Areas for improvement]
    Where we got lucky:
      - [Things that could have been worse]
  
  Action Items:
    | Action | Owner | Priority | Due Date | Status |
    |--------|-------|----------|----------|--------|
    | Add monitoring for X | @engineer | P1 | [date] | Open |
    | Update runbook X | @sre | P2 | [date] | Open |
    | Fix timeout handling | @team | P1 | [date] | Open |

Blameless culture principles:
  Focus on systems, not individuals.
  Assume people made the best decisions with available information.
  Ask "how did this happen?" not "who did this?"
  Share openly to prevent recurrence.
  Track action items to completion.
  Review past incidents in team meetings.

Post-mortem review process:
  1. IC drafts post-mortem within 48 hours
  2. Team reviews and adds context
  3. Action items assigned with owners and deadlines
  4. Post-mortem shared organization-wide
  5. Action items tracked in regular meetings
  6. Quarterly review of incident trends

Incident metrics:
  MTTD (Mean Time to Detect):
    Time from incident start to detection.
    Goal: Reduce through better monitoring.
  
  MTTA (Mean Time to Acknowledge):
    Time from alert to human response.
    Goal: Reduce through better on-call practices.
  
  MTTR (Mean Time to Resolve):
    Time from detection to resolution.
    Goal: Reduce through better runbooks and automation.
  
  MTBF (Mean Time Between Failures):
    Time between incidents.
    Goal: Increase through better engineering.
  
  Incident frequency by severity.
  Action item completion rate.
  Recurring incident percentage.
` + "```" + ``,
					CodeExamples: `# Incident management scripts

# 1. Incident opener
#!/bin/bash
set -e

SEVERITY="${1:?Usage: $0 <sev1|sev2|sev3|sev4> <description>}"
DESCRIPTION="${2:?Description required}"
IC="${IC:-$(whoami)}"
INCIDENT_ID="INC-$(date +%Y%m%d-%H%M%S)"

echo "=== Opening Incident ==="
echo "ID: $INCIDENT_ID"
echo "Severity: $SEVERITY"
echo "IC: $IC"
echo "Description: $DESCRIPTION"
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Create incident file
INCIDENT_DIR="${INCIDENT_DIR:-$HOME/incidents}"
mkdir -p "$INCIDENT_DIR"
INCIDENT_FILE="$INCIDENT_DIR/${INCIDENT_ID}.md"

cat > "$INCIDENT_FILE" << EOF
# $INCIDENT_ID

**Severity:** $SEVERITY
**IC:** $IC
**Status:** Investigating
**Started:** $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Description
$DESCRIPTION

## Timeline
- $(date +%H:%M) - Incident opened by $IC

## Actions Taken

## Root Cause

## Resolution

## Action Items
EOF

echo ""
echo "Incident file: $INCIDENT_FILE"

# Notify Slack
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"
if [ -n "$SLACK_WEBHOOK" ]; then
    EMOJI="⚠️"
    case "$SEVERITY" in
        sev1) EMOJI="🔴" ;;
        sev2) EMOJI="🟠" ;;
        sev3) EMOJI="🟡" ;;
        sev4) EMOJI="🔵" ;;
    esac
    
    curl -s -X POST "$SLACK_WEBHOOK" \
        -H 'Content-type: application/json' \
        -d "{\"text\":\"$EMOJI INCIDENT $INCIDENT_ID [$SEVERITY]\n$DESCRIPTION\nIC: $IC\"}" 2>/dev/null
    echo "Slack notification sent"
fi

echo ""
echo "Next steps:"
echo "  1. Assess impact and update severity if needed"
echo "  2. Begin investigation"
echo "  3. Update timeline: echo '- HH:MM - <action>' >> $INCIDENT_FILE"

# 2. Post-mortem generator
#!/bin/bash
INCIDENT_ID="${1:?Usage: $0 <incident-id>}"
INCIDENT_DIR="${INCIDENT_DIR:-$HOME/incidents}"
INCIDENT_FILE="$INCIDENT_DIR/${INCIDENT_ID}.md"

if [ ! -f "$INCIDENT_FILE" ]; then
    echo "Incident not found: $INCIDENT_FILE"
    exit 1
fi

POSTMORTEM_FILE="$INCIDENT_DIR/${INCIDENT_ID}-postmortem.md"

cat > "$POSTMORTEM_FILE" << EOF
# Post-Mortem: $INCIDENT_ID

**Date:** $(date +%Y-%m-%d)
**Author:** $(whoami)
**Reviewers:** [Add reviewers]

## Summary
[Brief description of what happened]

## Impact
- **Duration:** [start] to [end]
- **Users affected:** [number/percentage]
- **Error budget consumed:** [X%]

## Timeline
[Copy from incident file and expand]

## Root Cause
[Detailed technical explanation]

## Contributing Factors
- [Factor 1]
- [Factor 2]

## Detection
- How detected: [alert/customer report/manual]
- Time to detect: [X minutes]
- Could we detect earlier? [Yes/No - how]

## Resolution
[What fixed the issue]

## Lessons Learned

### What went well
- 

### What went poorly
- 

### Where we got lucky
- 

## Action Items

| # | Action | Owner | Priority | Due | Status |
|---|--------|-------|----------|-----|--------|
| 1 | | | P1 | | Open |
| 2 | | | P2 | | Open |
| 3 | | | P2 | | Open |

## References
- Incident file: $INCIDENT_FILE
- Dashboards: [links]
- Logs: [links]
EOF

echo "Post-mortem created: $POSTMORTEM_FILE"
echo "Please fill in the template within 48 hours"

# 3. Incident metrics reporter
#!/bin/bash
echo "=== Incident Metrics Report ==="

INCIDENT_DIR="${INCIDENT_DIR:-$HOME/incidents}"
PERIOD="${1:-30}"  # days

echo "Period: Last $PERIOD days"
echo ""

if [ ! -d "$INCIDENT_DIR" ]; then
    echo "No incident directory found"
    exit 0
fi

# Count incidents by severity
echo "--- Incidents by Severity ---"
for sev in sev1 sev2 sev3 sev4; do
    COUNT=$(find "$INCIDENT_DIR" -name "INC-*.md" -not -name "*postmortem*" \
        -newer "$INCIDENT_DIR" -exec grep -l "Severity.*$sev" {} \; 2>/dev/null | wc -l | tr -d ' ')
    echo "  $sev: $COUNT"
done

# Action items status
echo ""
echo "--- Action Items ---"
OPEN=$(grep -r "| Open |" "$INCIDENT_DIR"/*-postmortem.md 2>/dev/null | wc -l | tr -d ' ')
DONE=$(grep -r "| Done |" "$INCIDENT_DIR"/*-postmortem.md 2>/dev/null | wc -l | tr -d ' ')
echo "  Open: $OPEN"
echo "  Done: $DONE"
if [ $((OPEN + DONE)) -gt 0 ]; then
    COMPLETION=$(echo "scale=0; $DONE * 100 / ($OPEN + $DONE)" | bc 2>/dev/null || echo "N/A")
    echo "  Completion: ${COMPLETION}%"
fi

echo ""
echo "--- Recent Incidents ---"
ls -t "$INCIDENT_DIR"/INC-*.md 2>/dev/null | grep -v postmortem | head -5 | while read f; do
    NAME=$(basename "$f" .md)
    SEV=$(grep "Severity" "$f" 2>/dev/null | head -1 | grep -o "sev[0-9]" || echo "unknown")
    STATUS=$(grep "Status" "$f" 2>/dev/null | head -1 | sed 's/.*\*\*Status:\*\* //' || echo "unknown")
    echo "  $NAME [$SEV] - $STATUS"
done`,
				},
			},
		},
	})
}
