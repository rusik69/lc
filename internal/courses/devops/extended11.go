package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1470,
			Title:       "Disaster Recovery and Business Continuity",
			Description: "Design and implement disaster recovery strategies including RTO/RPO planning, failover mechanisms, backup strategies, and business continuity testing.",
			Order:       70,
			Lessons: []problems.Lesson{
				{
					Title: "DR Planning and Architecture",
					Content: `Disaster Recovery (DR) ensures systems can recover from catastrophic failures with minimal data loss and downtime.

**DR Fundamentals:**
` + "```" + `
Key metrics:
  RTO (Recovery Time Objective):
    Maximum acceptable downtime.
    Example: 4 hours means system must be restored within 4 hours.
  
  RPO (Recovery Point Objective):
    Maximum acceptable data loss.
    Example: 1 hour means at most 1 hour of data can be lost.
  
  MTTR (Mean Time to Recovery):
    Average time to restore service.
  
  MTBF (Mean Time Between Failures):
    Average time between incidents.

DR strategies (increasing cost and speed):
  1. Backup & Restore:
     RTO: Hours to days
     RPO: Hours (last backup)
     Cost: $
     - Regular backups to offsite storage
     - Manual or scripted restore process
     - Cheapest but slowest recovery
  
  2. Pilot Light:
     RTO: Minutes to hours
     RPO: Minutes
     Cost: $$
     - Core infrastructure always running
     - Databases replicated
     - Compute scaled up on failover
     - Small always-on footprint
  
  3. Warm Standby:
     RTO: Minutes
     RPO: Seconds to minutes
     Cost: $$$
     - Scaled-down copy of production
     - All components running
     - Scale up during failover
     - Regular data sync
  
  4. Multi-Site Active/Active:
     RTO: Seconds
     RPO: Near zero
     Cost: $$$$
     - Full production in multiple regions
     - Traffic split across regions
     - Real-time data replication
     - Automatic failover

AWS multi-region architecture:
  Route 53 (DNS failover):
    Health checks → automatic failover
    Routing policies: Failover, Latency, Weighted
  
  RDS Multi-Region:
    Primary region: Write/Read
    Secondary region: Read replica → Promote on failover
    
    Cross-region read replica:
      aws rds create-db-instance-read-replica \
        --db-instance-identifier replica-us-west-2 \
        --source-db-instance-identifier primary-us-east-1 \
        --region us-west-2
  
  S3 Cross-Region Replication:
    aws s3api put-bucket-replication \
      --bucket source-bucket \
      --replication-configuration '{
        "Role": "arn:aws:iam::role/replication",
        "Rules": [{
          "Status": "Enabled",
          "Destination": {
            "Bucket": "arn:aws:s3:::dest-bucket"
          }
        }]
      }'
  
  DynamoDB Global Tables:
    Automatic multi-region active-active.
    Sub-second replication.
    Conflict resolution: Last writer wins.

Kubernetes DR:
  Multi-cluster strategies:
    Cluster API: Manage cluster lifecycle
    Submariner: Cross-cluster networking
    Liqo: Virtual kubelet for multi-cluster
    
    Velero for cluster backup/restore:
      # Backup entire cluster
      velero backup create full-backup \
        --include-resources '*' \
        --include-cluster-resources=true
      
      # Schedule daily backups
      velero schedule create daily \
        --schedule="0 2 * * *" \
        --ttl 720h
      
      # Restore to different cluster
      velero restore create --from-backup full-backup \
        --restore-volumes --include-namespaces production
    
    etcd backup:
      ETCDCTL_API=3 etcdctl snapshot save backup.db \
        --endpoints=https://127.0.0.1:2379 \
        --cacert=/etc/kubernetes/pki/etcd/ca.crt \
        --cert=/etc/kubernetes/pki/etcd/server.crt \
        --key=/etc/kubernetes/pki/etcd/server.key
      
      ETCDCTL_API=3 etcdctl snapshot restore backup.db \
        --data-dir=/var/lib/etcd-restored
` + "```" + ``,
					CodeExamples: `# Disaster recovery scripts

# 1. DR readiness checker
#!/bin/bash
echo "=== DR Readiness Assessment ==="
DATE=$(date +%Y-%m-%d)

READY=0
NOT_READY=0
TOTAL=0

check_dr() {
    local name="$1" status="$2" details="$3"
    ((TOTAL++))
    case "$status" in
        READY) ((READY++)); echo "  [READY]     $name - $details" ;;
        NOT_READY) ((NOT_READY++)); echo "  [NOT READY] $name - $details" ;;
        PARTIAL) echo "  [PARTIAL]   $name - $details" ;;
    esac
}

# Check backups
echo "--- Backup Status ---"

# Database backups
LATEST_BACKUP=$(ls -t /backups/*.dump 2>/dev/null | head -1)
if [ -n "$LATEST_BACKUP" ]; then
    BACKUP_AGE=$(( ($(date +%s) - $(stat -f %m "$LATEST_BACKUP" 2>/dev/null || stat -c %Y "$LATEST_BACKUP" 2>/dev/null)) / 3600 ))
    if [ "$BACKUP_AGE" -lt 24 ]; then
        check_dr "Database backup" "READY" "Last: ${BACKUP_AGE}h ago"
    else
        check_dr "Database backup" "NOT_READY" "Last: ${BACKUP_AGE}h ago (>24h)"
    fi
else
    check_dr "Database backup" "NOT_READY" "No backups found"
fi

# Velero backups
VELERO_LAST=$(velero backup get --sort-by=.metadata.creationTimestamp 2>/dev/null | \
    grep -v "NAME" | tail -1 | awk '{print $4}')
if [ -n "$VELERO_LAST" ]; then
    check_dr "Cluster backup (Velero)" "READY" "Status: $VELERO_LAST"
else
    check_dr "Cluster backup (Velero)" "NOT_READY" "No Velero backups"
fi

# Check replication
echo ""
echo "--- Replication Status ---"

# Database replication lag
if command -v psql &>/dev/null; then
    LAG=$(psql -h localhost -U postgres -t -c \
        "SELECT EXTRACT(EPOCH FROM replay_lag) FROM pg_stat_replication LIMIT 1" 2>/dev/null | tr -d ' ')
    if [ -n "$LAG" ] && [ "$LAG" != "" ]; then
        if (( $(echo "$LAG < 60" | bc -l 2>/dev/null) )); then
            check_dr "DB replication" "READY" "Lag: ${LAG}s"
        else
            check_dr "DB replication" "NOT_READY" "Lag: ${LAG}s (>60s)"
        fi
    else
        check_dr "DB replication" "NOT_READY" "No replication configured"
    fi
fi

# Check DNS failover
echo ""
echo "--- Failover Configuration ---"

if command -v aws &>/dev/null; then
    HC_COUNT=$(aws route53 list-health-checks --query 'HealthChecks | length(@)' 2>/dev/null || echo "0")
    if [ "$HC_COUNT" -gt 0 ]; then
        check_dr "DNS health checks" "READY" "$HC_COUNT configured"
    else
        check_dr "DNS health checks" "NOT_READY" "No health checks"
    fi
fi

echo ""
echo "=== Summary ==="
echo "Ready: $READY / $TOTAL"
echo "Not Ready: $NOT_READY / $TOTAL"

# 2. Automated failover script
#!/bin/bash
set -e

echo "=== Initiating Failover ==="

# Validate
echo "Pre-flight checks..."
DR_REGION="${DR_REGION:?DR_REGION not set}"
PRIMARY_REGION="${PRIMARY_REGION:?PRIMARY_REGION not set}"

# Check DR site health
echo "Checking DR site health..."
DR_HEALTHY=true

# Verify DR database
DB_REPLICA="${DB_REPLICA_ID:?DB replica ID required}"
DB_STATUS=$(aws rds describe-db-instances \
    --db-instance-identifier "$DB_REPLICA" \
    --region "$DR_REGION" \
    --query 'DBInstances[0].DBInstanceStatus' \
    --output text 2>/dev/null)

if [ "$DB_STATUS" != "available" ]; then
    echo "ERROR: DR database not available (status: $DB_STATUS)"
    DR_HEALTHY=false
fi

if [ "$DR_HEALTHY" = false ]; then
    echo "DR site not healthy. Aborting."
    exit 1
fi

echo ""
echo "DR site healthy. Proceeding with failover..."

# Step 1: Promote database replica
echo "Step 1: Promoting database replica..."
aws rds promote-read-replica \
    --db-instance-identifier "$DB_REPLICA" \
    --region "$DR_REGION" 2>/dev/null

echo "Waiting for promotion..."
aws rds wait db-instance-available \
    --db-instance-identifier "$DB_REPLICA" \
    --region "$DR_REGION" 2>/dev/null

echo "Database promoted successfully"

# Step 2: Update DNS
echo "Step 2: Updating DNS..."
HOSTED_ZONE_ID="${HOSTED_ZONE_ID:?Hosted zone required}"
DOMAIN="${DOMAIN:?Domain required}"
DR_ENDPOINT="${DR_ENDPOINT:?DR endpoint required}"

aws route53 change-resource-record-sets \
    --hosted-zone-id "$HOSTED_ZONE_ID" \
    --change-batch '{
        "Changes": [{
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "'"$DOMAIN"'",
                "Type": "CNAME",
                "TTL": 60,
                "ResourceRecords": [{"Value": "'"$DR_ENDPOINT"'"}]
            }
        }]
    }' 2>/dev/null

echo "DNS updated to DR endpoint"

# Step 3: Verify
echo "Step 3: Verifying..."
sleep 10
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://$DOMAIN/health" 2>/dev/null || echo "000")
if [ "$HTTP_STATUS" = "200" ]; then
    echo "Failover SUCCESSFUL - Service responding"
else
    echo "WARNING: Service not yet responding (HTTP $HTTP_STATUS)"
    echo "DNS propagation may take additional time"
fi

echo ""
echo "Failover complete at $(date)"`,
				},
				{
					Title: "DR Testing and Runbooks",
					Content: `Regular DR testing validates recovery procedures and identifies gaps before actual disasters occur.

**DR Testing Types:**
` + "```" + `
Testing approaches:
  1. Tabletop Exercise:
     Walk through DR plan on paper.
     No actual systems affected.
     Identify gaps in documentation.
     Frequency: Quarterly.
  
  2. Walkthrough Test:
     Step through procedures manually.
     Verify access to systems and tools.
     Test communication channels.
     Frequency: Quarterly.
  
  3. Simulation Test:
     Simulate specific failure scenarios.
     Execute recovery procedures.
     Measure actual RTO/RPO.
     Frequency: Semi-annually.
  
  4. Parallel Test:
     Bring up DR environment alongside production.
     Verify functionality without switching traffic.
     Validate data consistency.
     Frequency: Semi-annually.
  
  5. Full Interruption Test:
     Actually fail over to DR.
     Most realistic test.
     Highest risk.
     Frequency: Annually.

Game Day planning:
  Preparation:
    Define scope and objectives
    Identify participants and roles
    Prepare rollback procedures
    Notify stakeholders
    Schedule during low-traffic period
  
  Execution:
    Inject failure
    Start timer for RTO measurement
    Execute recovery procedures
    Document each step and timing
    Monitor DR environment
  
  Post-mortem:
    Compare actual vs target RTO/RPO
    Document issues encountered
    Update runbooks
    Create action items
    Schedule follow-up

DR runbook template:
  Title: [Service] Disaster Recovery
  Last Updated: [Date]
  Owner: [Team]
  
  1. Detection
     Symptoms: [How to identify the disaster]
     Monitoring: [Alerts and dashboards]
     Escalation: [Who to notify]
  
  2. Assessment
     Impact Analysis:
       - Affected services
       - User impact
       - Data at risk
     Decision criteria for DR activation
  
  3. Communication
     Internal: [Slack channel, email list]
     External: [Status page update]
     Cadence: Updates every [X] minutes
  
  4. Recovery Steps
     Step 1: [Action] (Est: X minutes)
       Command: [exact command]
       Verification: [how to verify]
     Step 2: [Action] (Est: X minutes)
       ...
  
  5. Verification
     Health checks
     Data integrity validation
     Performance baseline comparison
  
  6. Failback
     Steps to return to primary
     Data synchronization
     DNS cutover
  
  7. Post-Incident
     Timeline documentation
     Root cause analysis
     Improvement action items
` + "```" + `

**Automated DR Testing:**
` + "```" + `
DR test automation:
  Scheduled DR tests:
    # CronJob for DR validation
    apiVersion: batch/v1
    kind: CronJob
    metadata:
      name: dr-validation
    spec:
      schedule: "0 3 * * 0"  # Weekly Sunday 3 AM
      jobTemplate:
        spec:
          template:
            spec:
              containers:
                - name: dr-test
                  image: myregistry/dr-validator:latest
                  env:
                    - name: DR_REGION
                      value: us-west-2
                    - name: TEST_TYPE
                      value: parallel
              restartPolicy: Never
  
  Backup restore validation:
    Test latest backup can be restored.
    Verify data integrity.
    Compare checksums.
    Run application health checks.
    
    Steps:
      1. Get latest backup from S3/storage
      2. Restore to isolated test environment
      3. Run integrity checks
      4. Compare critical table counts
      5. Run smoke tests against restored data
      6. Report results and cleanup

Infrastructure as Code for DR:
  Terraform multi-region:
    module "primary" {
      source = "./modules/region"
      region = "us-east-1"
      role   = "primary"
      
      providers = {
        aws = aws.us_east_1
      }
    }
    
    module "dr" {
      source = "./modules/region"
      region = "us-west-2"
      role   = "dr"
      
      providers = {
        aws = aws.us_west_2
      }
    }
    
    # Cross-region replication
    resource "aws_s3_bucket_replication_configuration" "dr" {
      role   = aws_iam_role.replication.arn
      bucket = module.primary.bucket_id
      
      rule {
        status = "Enabled"
        destination {
          bucket        = module.dr.bucket_arn
          storage_class = "STANDARD_IA"
        }
      }
    }

Communication during DR:
  Status page tools:
    Statuspage (Atlassian)
    Cachet (self-hosted)
    Instatus
    Better Uptime
  
  Automated status updates:
    #!/bin/bash
    # Update status page
    curl -X POST "https://api.statuspage.io/v1/pages/$PAGE_ID/incidents" \
      -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "incident": {
          "name": "Service Degradation - DR Activated",
          "status": "investigating",
          "impact_override": "major",
          "body": "We are experiencing issues and have activated our DR plan.",
          "component_ids": ["component_id"],
          "components": {"component_id": "major_outage"}
        }
      }'

Recovery validation checklist:
  Infrastructure:
    [ ] All services running in DR region
    [ ] Load balancers healthy
    [ ] DNS pointing to DR
    [ ] SSL certificates valid
    [ ] CDN configured
  
  Data:
    [ ] Database restored and accessible
    [ ] Data integrity verified
    [ ] Replication lag acceptable
    [ ] Cache warmed up
  
  Application:
    [ ] Health endpoints responding
    [ ] Authentication working
    [ ] Key user flows functional
    [ ] Background jobs running
    [ ] External integrations connected
  
  Monitoring:
    [ ] Alerts configured for DR
    [ ] Dashboards accessible
    [ ] Log collection active
    [ ] Metrics flowing
` + "```" + ``,
					CodeExamples: `# DR testing and automation scripts

# 1. DR test executor
#!/bin/bash
set -e

echo "=== Disaster Recovery Test ==="
TEST_TYPE="${1:-parallel}"
DR_REGION="${DR_REGION:-us-west-2}"
TEST_ID="dr-test-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/tmp/${TEST_ID}.log"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }

log "Test ID: $TEST_ID"
log "Test Type: $TEST_TYPE"
log "DR Region: $DR_REGION"

STEPS_PASSED=0
STEPS_FAILED=0

test_step() {
    local name="$1" cmd="$2"
    log "Testing: $name"
    START=$(date +%s)
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        END=$(date +%s)
        DURATION=$((END - START))
        log "  PASSED ($DURATION seconds)"
        ((STEPS_PASSED++))
    else
        END=$(date +%s)
        DURATION=$((END - START))
        log "  FAILED ($DURATION seconds)"
        ((STEPS_FAILED++))
    fi
}

# Test 1: DR infrastructure accessible
test_step "DR cluster connectivity" \
    "kubectl --context dr-cluster get nodes --no-headers 2>/dev/null | grep -q Ready"

# Test 2: Backup availability
test_step "Latest backup exists" \
    "aws s3 ls s3://backups/latest/ --region $DR_REGION 2>/dev/null | grep -q dump"

# Test 3: Backup restore
if [ "$TEST_TYPE" = "parallel" ] || [ "$TEST_TYPE" = "full" ]; then
    test_step "Backup restore" \
        "echo 'Simulating backup restore...'; sleep 2; true"
fi

# Test 4: DNS failover
test_step "DNS health check configured" \
    "aws route53 list-health-checks --query 'HealthChecks[0].Id' --output text 2>/dev/null | grep -q ."

# Test 5: DR application health
test_step "DR endpoint reachable" \
    "curl -sf --max-time 10 https://dr.example.com/health 2>/dev/null || true"

# Test 6: Monitoring in DR
test_step "DR monitoring active" \
    "kubectl --context dr-cluster get pods -n monitoring --no-headers 2>/dev/null | grep -q Running || true"

# Summary
log ""
log "=== Results ==="
log "Passed: $STEPS_PASSED"
log "Failed: $STEPS_FAILED"
log "Total: $((STEPS_PASSED + STEPS_FAILED))"
log "Log: $LOG_FILE"

if [ "$STEPS_FAILED" -gt 0 ]; then
    log "STATUS: ISSUES FOUND"
    exit 1
else
    log "STATUS: ALL TESTS PASSED"
fi

# 2. Backup restore validator
#!/bin/bash
set -e

echo "=== Backup Restore Validation ==="

DB_NAME="${DB_NAME:-mydb}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
VERIFY_DB="${DB_NAME}_verify_$(date +%s)"

# Find latest backup
LATEST=$(ls -t "$BACKUP_DIR"/${DB_NAME}_*.dump 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    echo "ERROR: No backup found"
    exit 1
fi

echo "Backup: $LATEST"
echo "Verify DB: $VERIFY_DB"

# Create verify database
createdb "$VERIFY_DB" 2>/dev/null || true

# Restore
echo ""
echo "--- Restoring ---"
START=$(date +%s)
pg_restore -d "$VERIFY_DB" --no-acl --no-owner "$LATEST" 2>/dev/null
END=$(date +%s)
RESTORE_TIME=$((END - START))
echo "Restore time: ${RESTORE_TIME}s"

# Verify
echo ""
echo "--- Verification ---"

ERRORS=0

# Table count
ORIG_TABLES=$(psql -d "$DB_NAME" -t -c \
    "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null | tr -d ' ')
REST_TABLES=$(psql -d "$VERIFY_DB" -t -c \
    "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null | tr -d ' ')

echo "Tables - Original: $ORIG_TABLES, Restored: $REST_TABLES"
if [ "$ORIG_TABLES" != "$REST_TABLES" ]; then
    echo "  WARNING: Table count mismatch"
    ((ERRORS++))
fi

# Row counts for key tables
for table in users orders products; do
    ORIG=$(psql -d "$DB_NAME" -t -c "SELECT count(*) FROM $table" 2>/dev/null | tr -d ' ')
    REST=$(psql -d "$VERIFY_DB" -t -c "SELECT count(*) FROM $table" 2>/dev/null | tr -d ' ')
    echo "$table - Original: $ORIG, Restored: $REST"
    if [ "$ORIG" != "$REST" ] 2>/dev/null; then
        echo "  WARNING: Row count mismatch"
        ((ERRORS++))
    fi
done 2>/dev/null || true

# Cleanup
echo ""
echo "--- Cleanup ---"
dropdb "$VERIFY_DB" 2>/dev/null

echo ""
echo "=== Summary ==="
echo "Restore time: ${RESTORE_TIME}s"
echo "Errors: $ERRORS"
if [ "$ERRORS" -eq 0 ]; then
    echo "Status: VALIDATED"
else
    echo "Status: ISSUES FOUND"
fi

# 3. DR communication helper
#!/bin/bash
echo "=== DR Communication ==="

ACTION="${1:?Usage: $0 <start|update|resolve>}"
MESSAGE="${2:-DR event in progress}"

SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"
PAGERDUTY_KEY="${PAGERDUTY_ROUTING_KEY}"

case "$ACTION" in
    start)
        echo "Initiating DR communication..."
        
        # Slack
        if [ -n "$SLACK_WEBHOOK" ]; then
            curl -s -X POST "$SLACK_WEBHOOK" \
                -H 'Content-type: application/json' \
                -d "{\"text\":\"🚨 DR ACTIVATED: $MESSAGE\nTime: $(date)\nIncident Commander: $(whoami)\"}" 2>/dev/null
            echo "Slack notification sent"
        fi
        
        echo "DR communication initiated"
        ;;
    
    update)
        if [ -n "$SLACK_WEBHOOK" ]; then
            curl -s -X POST "$SLACK_WEBHOOK" \
                -H 'Content-type: application/json' \
                -d "{\"text\":\"📋 DR UPDATE: $MESSAGE\nTime: $(date)\"}" 2>/dev/null
            echo "Update sent"
        fi
        ;;
    
    resolve)
        if [ -n "$SLACK_WEBHOOK" ]; then
            curl -s -X POST "$SLACK_WEBHOOK" \
                -H 'Content-type: application/json' \
                -d "{\"text\":\"✅ DR RESOLVED: $MESSAGE\nTime: $(date)\nDuration: Check incident log\"}" 2>/dev/null
            echo "Resolution sent"
        fi
        ;;
esac`,
				},
			},
		},
	})
}
