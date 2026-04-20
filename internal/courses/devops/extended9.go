package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1466,
			Title:       "DevSecOps Pipeline Integration",
			Description: "Integrate security testing throughout the CI/CD pipeline with SAST, DAST, dependency scanning, and compliance automation.",
			Order:       66,
			Lessons: []problems.Lesson{
				{
					Title: "Static and Dynamic Security Testing",
					Content: `DevSecOps integrates security practices at every stage of the software development lifecycle, shifting security left.

**SAST (Static Application Security Testing):**
` + "```" + `
SAST tools by language:
  Go:
    gosec: Go security checker
      gosec ./...
      gosec -fmt json -out results.json ./...
      gosec -exclude=G104 ./...  
    
    staticcheck: Advanced Go analysis
      staticcheck ./...
    
    govulncheck: Known vulnerability check
      govulncheck ./...
  
  Python:
    bandit: Python security linter
      bandit -r src/ -f json -o results.json
      bandit -r src/ -ll  # Only medium+ severity
    
    safety: Check dependencies
      safety check --json
      pip-audit
  
  JavaScript/TypeScript:
    eslint-plugin-security
    semgrep:
      semgrep scan --config auto
      semgrep scan --config p/javascript
    
    njsscan:
      njsscan --json -o results.json ./src
  
  General (multi-language):
    semgrep:
      semgrep scan --config auto
      semgrep scan --config p/owasp-top-ten
      semgrep scan --config p/secrets
    
    SonarQube:
      sonar-scanner \
        -Dsonar.projectKey=myproject \
        -Dsonar.sources=src \
        -Dsonar.host.url=http://sonar:9000
    
    CodeQL (GitHub):
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: go, javascript
      - name: Perform analysis
        uses: github/codeql-action/analyze@v3

CI/CD integration (GitHub Actions):
  name: Security Scan
  on: [push, pull_request]
  
  jobs:
    sast:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        
        - name: Semgrep
          uses: semgrep/semgrep-action@v1
          with:
            config: >-
              p/security-audit
              p/secrets
              p/owasp-top-ten
          env:
            SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
        
        - name: GoSec
          uses: securego/gosec@master
          with:
            args: ./...
        
        - name: Upload SARIF
          uses: github/codeql-action/upload-sarif@v3
          with:
            sarif_file: results.sarif
` + "```" + `

**DAST (Dynamic Application Security Testing):**
` + "```" + `
DAST tools:
  OWASP ZAP:
    # Full scan
    docker run -t ghcr.io/zaproxy/zaproxy:stable zap-full-scan.py \
      -t https://myapp.example.com
    
    # API scan
    docker run -t ghcr.io/zaproxy/zaproxy:stable zap-api-scan.py \
      -t https://myapp.example.com/api/openapi.json \
      -f openapi
    
    # Baseline scan (quick)
    docker run -t ghcr.io/zaproxy/zaproxy:stable zap-baseline.py \
      -t https://myapp.example.com
    
    # CI/CD integration
    - name: ZAP Scan
      uses: zaproxy/action-full-scan@v0.10.0
      with:
        target: 'https://staging.example.com'
        rules_file_name: '.zap/rules.tsv'
        allow_issue_writing: false
  
  Nuclei:
    nuclei -u https://myapp.example.com -t cves/ -severity critical,high
    nuclei -u https://myapp.example.com -t exposures/
    nuclei -l urls.txt -t technologies/ -o results.txt
  
  Nikto:
    nikto -h https://myapp.example.com -o results.html -Format html

Dependency scanning:
  Dependabot (GitHub):
    # .github/dependabot.yml
    version: 2
    updates:
      - package-ecosystem: gomod
        directory: /
        schedule:
          interval: weekly
        open-pull-requests-limit: 10
        reviewers:
          - security-team
      
      - package-ecosystem: docker
        directory: /
        schedule:
          interval: weekly
      
      - package-ecosystem: github-actions
        directory: /
        schedule:
          interval: monthly
  
  Renovate:
    # renovate.json
    {
      "$schema": "https://docs.renovatebot.com/renovate-schema.json",
      "extends": ["config:recommended", "security:openssf-scorecard"],
      "vulnerabilityAlerts": {
        "labels": ["security"],
        "automerge": true
      },
      "packageRules": [
        {
          "matchUpdateTypes": ["patch"],
          "automerge": true
        }
      ]
    }

Secret scanning:
  Gitleaks:
    gitleaks detect --source . --verbose
    gitleaks detect --source . --report-format json --report-path results.json
    
    # Pre-commit hook
    gitleaks protect --staged --verbose
    
    # CI/CD
    - name: Gitleaks
      uses: gitleaks/gitleaks-action@v2
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  
  TruffleHog:
    trufflehog git file://. --only-verified
    trufflehog github --org=myorg
  
  git-secrets:
    git secrets --install
    git secrets --register-aws
    git secrets --scan
` + "```" + ``,
					CodeExamples: `# DevSecOps pipeline configuration

# 1. Complete security scanning pipeline
# .github/workflows/security.yml
name: Security Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday scan

permissions:
  contents: read
  security-events: write

jobs:
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: $${{ secrets.GITHUB_TOKEN }}

  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Semgrep
        uses: semgrep/semgrep-action@v1
        with:
          config: p/owasp-top-ten p/secrets

  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - name: Govulncheck
        run: |
          go install golang.org/x/vuln/cmd/govulncheck@latest
          govulncheck ./...

  container-scan:
    runs-on: ubuntu-latest
    needs: [sast]
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t myapp:test .
      - name: Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:test
          format: sarif
          output: trivy.sarif
          severity: CRITICAL,HIGH
      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy.sarif

  dast:
    runs-on: ubuntu-latest
    needs: [container-scan]
    steps:
      - uses: actions/checkout@v4
      - name: Start app
        run: |
          docker compose -f docker-compose.test.yml up -d
          sleep 10
      - name: ZAP Baseline
        uses: zaproxy/action-baseline@v0.12.0
        with:
          target: http://localhost:8080
      - name: Stop app
        if: always()
        run: docker compose -f docker-compose.test.yml down

# 2. Security gate script
#!/bin/bash
set -e

echo "=== Security Gate Check ==="

CRITICAL=0
HIGH=0
MEDIUM=0

# Check SAST results
if [ -f sast-results.json ]; then
    SAST_CRITICAL=$(jq '[.results[] | select(.severity == "CRITICAL")] | length' sast-results.json)
    SAST_HIGH=$(jq '[.results[] | select(.severity == "HIGH")] | length' sast-results.json)
    CRITICAL=$((CRITICAL + SAST_CRITICAL))
    HIGH=$((HIGH + SAST_HIGH))
    echo "SAST: Critical=$SAST_CRITICAL High=$SAST_HIGH"
fi

# Check container scan results
if [ -f trivy-results.json ]; then
    TRIVY_CRITICAL=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' trivy-results.json)
    TRIVY_HIGH=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "HIGH")] | length' trivy-results.json)
    CRITICAL=$((CRITICAL + TRIVY_CRITICAL))
    HIGH=$((HIGH + TRIVY_HIGH))
    echo "Container: Critical=$TRIVY_CRITICAL High=$TRIVY_HIGH"
fi

# Check dependency scan results
if [ -f deps-results.json ]; then
    DEP_CRITICAL=$(jq '[.vulnerabilities[] | select(.severity == "CRITICAL")] | length' deps-results.json)
    CRITICAL=$((CRITICAL + DEP_CRITICAL))
    echo "Dependencies: Critical=$DEP_CRITICAL"
fi

echo ""
echo "Total: Critical=$CRITICAL High=$HIGH"

if [ "$CRITICAL" -gt 0 ]; then
    echo ""
    echo "GATE FAILED: $CRITICAL critical vulnerabilities found"
    exit 1
fi

if [ "$HIGH" -gt 5 ]; then
    echo ""
    echo "GATE FAILED: Too many high vulnerabilities ($HIGH)"
    exit 1
fi

echo ""
echo "GATE PASSED"`,
				},
				{
					Title: "Compliance as Code and Policy Automation",
					Content: `Compliance as Code automates security standards enforcement, audit evidence collection, and regulatory compliance.

**Compliance Frameworks:**
` + "```" + `
Common frameworks:
  SOC 2 Type II:
    Trust Service Criteria:
      Security: Access controls, encryption, vulnerability management
      Availability: Uptime, disaster recovery, monitoring
      Processing Integrity: Data accuracy, error handling
      Confidentiality: Data classification, access restrictions
      Privacy: PII handling, consent management
  
  PCI DSS:
    Requirements:
      1. Install firewall configuration
      2. No vendor-supplied defaults
      3. Protect stored cardholder data
      4. Encrypt transmission of cardholder data
      5. Protect against malware
      6. Develop secure systems
      7. Restrict access by need-to-know
      8. Identify and authenticate access
      9. Restrict physical access
      10. Track and monitor access
      11. Test security systems
      12. Maintain security policy
  
  HIPAA:
    Technical safeguards:
      Access control, audit controls, integrity controls,
      transmission security
  
  GDPR:
    Data protection requirements:
      Consent, data minimization, right to erasure,
      data portability, breach notification

Policy as Code tools:
  Open Policy Agent (OPA):
    # Rego policy
    package kubernetes.admission
    
    deny[msg] {
      input.request.kind.kind == "Pod"
      container := input.request.object.spec.containers[_]
      not container.resources.limits
      msg := sprintf("Container %v must have resource limits", [container.name])
    }
    
    deny[msg] {
      input.request.kind.kind == "Pod"
      container := input.request.object.spec.containers[_]
      container.securityContext.privileged
      msg := sprintf("Container %v cannot be privileged", [container.name])
    }
    
    deny[msg] {
      input.request.kind.kind == "Deployment"
      not input.request.object.spec.template.metadata.labels.app
      msg := "Deployment must have app label"
    }
  
  Kyverno:
    apiVersion: kyverno.io/v1
    kind: ClusterPolicy
    metadata:
      name: require-labels
      annotations:
        policies.kyverno.io/title: Require Labels
        policies.kyverno.io/category: Best Practices
        policies.kyverno.io/severity: medium
    spec:
      validationFailureAction: Enforce
      background: true
      rules:
        - name: check-labels
          match:
            any:
              - resources:
                  kinds: [Deployment, StatefulSet]
          validate:
            message: "Must have labels: app, team, env"
            pattern:
              metadata:
                labels:
                  app: "?*"
                  team: "?*"
                  env: "production|staging|development"
        
        - name: add-default-labels
          match:
            any:
              - resources:
                  kinds: [Pod]
          mutate:
            patchStrategicMerge:
              metadata:
                labels:
                  +(managed-by): "kyverno"

Audit logging:
  Kubernetes audit policy:
    apiVersion: audit.k8s.io/v1
    kind: Policy
    rules:
      - level: Metadata
        resources:
          - group: ""
            resources: ["secrets", "configmaps"]
      
      - level: RequestResponse
        resources:
          - group: ""
            resources: ["pods"]
        verbs: ["create", "update", "patch", "delete"]
      
      - level: Request
        resources:
          - group: "rbac.authorization.k8s.io"
        verbs: ["create", "update", "patch", "delete"]
      
      - level: None
        resources:
          - group: ""
            resources: ["endpoints", "events"]
  
  CloudTrail (AWS):
    aws cloudtrail create-trail \
      --name security-trail \
      --s3-bucket-name audit-logs \
      --is-multi-region-trail \
      --enable-log-file-validation
  
  Audit log shipping:
    Fluent Bit → S3/Elasticsearch
    Vector → Loki/S3
    Filebeat → Elasticsearch
` + "```" + `

**Infrastructure Compliance Scanning:**
` + "```" + `
Terraform compliance:
  tfsec:
    tfsec .
    tfsec . --format json --out results.json
    tfsec . --minimum-severity HIGH
  
  Checkov:
    checkov -d .
    checkov -f main.tf
    checkov --framework terraform -d . --output json
    checkov -d . --check CKV_AWS_18,CKV_AWS_21
  
  Terrascan:
    terrascan scan -t aws -d .
    terrascan scan -t k8s -f deployment.yaml

Cloud compliance:
  AWS Config Rules:
    Resources:
      ConfigRule:
        Type: AWS::Config::ConfigRule
        Properties:
          ConfigRuleName: s3-bucket-encryption
          Source:
            Owner: AWS
            SourceIdentifier: S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED
          Scope:
            ComplianceResourceTypes:
              - AWS::S3::Bucket
  
  Azure Policy:
    {
      "mode": "Indexed",
      "policyRule": {
        "if": {
          "field": "type",
          "equals": "Microsoft.Storage/storageAccounts"
        },
        "then": {
          "effect": "audit",
          "details": {
            "type": "Microsoft.Storage/storageAccounts",
            "existenceCondition": {
              "field": "Microsoft.Storage/storageAccounts/encryption.services.blob.enabled",
              "equals": "true"
            }
          }
        }
      }
    }

Compliance reporting:
  Generate evidence:
    - Access control lists
    - Vulnerability scan results
    - Encryption status reports
    - Change management logs
    - Incident response records
    - Backup verification logs
  
  Automated compliance dashboard:
    Collect metrics:
      - Percentage of compliant resources
      - Open vulnerabilities by severity
      - Mean time to remediate
      - Policy violations over time
      - Audit log coverage
` + "```" + ``,
					CodeExamples: `# Compliance automation scripts

# 1. Compliance evidence collector
#!/bin/bash
echo "=== Compliance Evidence Collection ==="
DATE=$(date +%Y-%m-%d)
REPORT_DIR="compliance-evidence/${DATE}"
mkdir -p "$REPORT_DIR"

# Access controls
echo "--- Collecting Access Controls ---"
kubectl get clusterrolebindings -o json 2>/dev/null > "$REPORT_DIR/rbac-bindings.json"
kubectl get rolebindings --all-namespaces -o json 2>/dev/null > "$REPORT_DIR/role-bindings.json"

# Network policies
echo "--- Collecting Network Policies ---"
kubectl get networkpolicies --all-namespaces -o json 2>/dev/null > "$REPORT_DIR/network-policies.json"

# Pod security
echo "--- Collecting Pod Security ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq '[.items[] | {
        namespace: .metadata.namespace,
        name: .metadata.name,
        serviceAccount: .spec.serviceAccountName,
        runAsNonRoot: .spec.securityContext.runAsNonRoot,
        readOnlyRootFs: (.spec.containers[].securityContext.readOnlyRootFilesystem // false),
        privileged: (.spec.containers[].securityContext.privileged // false)
    }]' 2>/dev/null > "$REPORT_DIR/pod-security.json"

# Encryption status
echo "--- Collecting Encryption Status ---"
kubectl get secrets --all-namespaces --no-headers 2>/dev/null | wc -l > "$REPORT_DIR/secret-count.txt"

# Image versions
echo "--- Collecting Image Inventory ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '[.items[] | .spec.containers[] | .image] | sort | unique[]' 2>/dev/null > "$REPORT_DIR/images.txt"

# Vulnerabilities
if command -v trivy &>/dev/null; then
    echo "--- Running Vulnerability Scans ---"
    while IFS= read -r image; do
        SAFE_NAME=$(echo "$image" | tr '/:' '_')
        trivy image --format json "$image" > "$REPORT_DIR/vuln-${SAFE_NAME}.json" 2>/dev/null || true
    done < "$REPORT_DIR/images.txt"
fi

echo ""
echo "Evidence collected in $REPORT_DIR"
ls -la "$REPORT_DIR"

# 2. Policy violation reporter
#!/bin/bash
echo "=== Policy Violation Report ==="

# Kyverno violations
if kubectl get crd clusterpolicies.kyverno.io &>/dev/null; then
    echo "--- Kyverno Policy Violations ---"
    kubectl get policyreport --all-namespaces -o json 2>/dev/null | \
        jq -r '.items[] | .results[]? | select(.result == "fail") |
        "\(.policy) | \(.resources[0].namespace)/\(.resources[0].name) | \(.message)"' 2>/dev/null | head -20
fi

# OPA/Gatekeeper violations
if kubectl get crd constrainttemplates.templates.gatekeeper.sh &>/dev/null; then
    echo ""
    echo "--- Gatekeeper Constraint Violations ---"
    for constraint in $(kubectl get constraints -o name 2>/dev/null); do
        VIOLATIONS=$(kubectl get "$constraint" -o json 2>/dev/null | \
            jq '.status.totalViolations // 0')
        if [ "$VIOLATIONS" -gt 0 ]; then
            echo "  $constraint: $VIOLATIONS violations"
            kubectl get "$constraint" -o json 2>/dev/null | \
                jq -r '.status.violations[]? | "    \(.enforcementAction): \(.message)"' 2>/dev/null | head -5
        fi
    done
fi

echo ""
echo "--- Summary ---"
TOTAL_VIOLATIONS=0
if kubectl get policyreport --all-namespaces -o json &>/dev/null; then
    KYERNO_VIOLATIONS=$(kubectl get policyreport --all-namespaces -o json 2>/dev/null | \
        jq '[.items[].results[]? | select(.result == "fail")] | length' 2>/dev/null)
    TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + KYERNO_VIOLATIONS))
fi
echo "Total violations: $TOTAL_VIOLATIONS"

# 3. Terraform compliance check
#!/bin/bash
echo "=== Infrastructure Compliance ==="

if ! command -v checkov &>/dev/null; then
    echo "checkov not installed"
    exit 1
fi

echo "--- Scanning Terraform ---"
checkov -d . --framework terraform --compact --quiet 2>/dev/null

echo ""
echo "--- High Severity Issues ---"
checkov -d . --framework terraform --output json 2>/dev/null | \
    jq -r '.results.failed_checks[] | select(.severity == "HIGH" or .severity == "CRITICAL") |
    "\(.severity): \(.check_id) - \(.name) [\(.file_path):\(.file_line_range[0])]"' 2>/dev/null | head -20`,
				},
			},
		},
		{
			ID:          1467,
			Title:       "Cloud Cost Optimization and FinOps",
			Description: "Implement cloud cost management strategies including resource right-sizing, spot instances, reserved capacity, and FinOps practices.",
			Order:       67,
			Lessons: []problems.Lesson{
				{
					Title: "Cost Monitoring and Resource Optimization",
					Content: `FinOps is the practice of bringing financial accountability to cloud spending, enabling teams to make cost-informed decisions.

**Cost Visibility:**
` + "```" + `
Cost monitoring tools:
  AWS:
    Cost Explorer:
      aws ce get-cost-and-usage \
        --time-period Start=2024-01-01,End=2024-01-31 \
        --granularity MONTHLY \
        --metrics "BlendedCost" "UnblendedCost" \
        --group-by Type=DIMENSION,Key=SERVICE
    
    Budgets:
      aws budgets create-budget \
        --account-id 123456789 \
        --budget '{
          "BudgetName": "monthly-limit",
          "BudgetLimit": {"Amount": "10000", "Unit": "USD"},
          "TimeUnit": "MONTHLY",
          "BudgetType": "COST"
        }' \
        --notifications-with-subscribers '[{
          "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80
          },
          "Subscribers": [{
            "SubscriptionType": "EMAIL",
            "Address": "team@example.com"
          }]
        }]'
  
  GCP:
    Billing Export → BigQuery:
      SELECT
        service.description,
        SUM(cost) as total_cost,
        SUM(usage.amount) as usage_amount
      FROM project.dataset.gcp_billing_export
      WHERE invoice.month = '202401'
      GROUP BY service.description
      ORDER BY total_cost DESC
  
  Azure:
    Cost Management:
      az costmanagement query \
        --type ActualCost \
        --timeframe MonthToDate \
        --dataset-aggregation '{"totalCost":{"name":"Cost","function":"Sum"}}'

Kubernetes cost allocation:
  Kubecost:
    helm install kubecost cost-analyzer \
      --repo https://kubecost.github.io/cost-analyzer/ \
      --namespace kubecost --create-namespace \
      --set kubecostToken="your-token"
    
    # Cost by namespace
    curl http://kubecost:9090/model/allocation \
      --data-urlencode 'window=7d' \
      --data-urlencode 'aggregate=namespace'
    
    # Cost by label
    curl http://kubecost:9090/model/allocation \
      --data-urlencode 'window=30d' \
      --data-urlencode 'aggregate=label:team'
  
  OpenCost:
    kubectl apply -f https://raw.githubusercontent.com/opencost/opencost/develop/kubernetes/opencost.yaml
  
  Resource tagging:
    Required tags:
      team: engineering
      environment: production
      cost-center: CC-1234
      project: myproject
      owner: team@example.com
    
    Tag enforcement (Kyverno):
      apiVersion: kyverno.io/v1
      kind: ClusterPolicy
      metadata:
        name: require-cost-tags
      spec:
        validationFailureAction: Enforce
        rules:
          - name: check-tags
            match:
              any:
                - resources:
                    kinds: [Deployment, StatefulSet]
            validate:
              message: "Must have cost tags: team, cost-center"
              pattern:
                metadata:
                  labels:
                    team: "?*"
                    cost-center: "?*"
` + "```" + `

**Resource Right-Sizing:**
` + "```" + `
Kubernetes resource optimization:
  VPA (Vertical Pod Autoscaler):
    apiVersion: autoscaling.k8s.io/v1
    kind: VerticalPodAutoscaler
    metadata:
      name: myapp-vpa
    spec:
      targetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: myapp
      updatePolicy:
        updateMode: "Off"  # Recommendation only
      resourcePolicy:
        containerPolicies:
          - containerName: '*'
            minAllowed:
              cpu: 50m
              memory: 64Mi
            maxAllowed:
              cpu: 2000m
              memory: 2Gi
    
    # Check recommendations
    kubectl get vpa myapp-vpa -o json | \
      jq '.status.recommendation.containerRecommendations'
  
  Goldilocks (VPA dashboard):
    kubectl label ns production goldilocks.fairwinds.com/enabled=true
    helm install goldilocks fairwinds-stable/goldilocks

  Request/limit analysis:
    # Over-provisioned pods (request >> actual usage)
    kubectl top pods -n production --containers 2>/dev/null | \
      sort -k3 -n -r | head -20
    
    # Prometheus query for right-sizing
    Actual CPU vs requested:
      container_cpu_usage_seconds_total / 
      kube_pod_container_resource_requests{resource="cpu"}
    
    Memory waste:
      1 - (container_memory_working_set_bytes / 
      kube_pod_container_resource_requests{resource="memory"})

AWS right-sizing:
  Compute Optimizer:
    aws compute-optimizer get-ec2-instance-recommendations \
      --filters name=Finding,values=OVER_PROVISIONED
  
  Trusted Advisor:
    aws support describe-trusted-advisor-checks \
      --language en | jq '.checks[] | select(.category == "cost_optimizing")'

Idle resource cleanup:
  Unused volumes:
    aws ec2 describe-volumes \
      --filters Name=status,Values=available \
      --query 'Volumes[*].{ID:VolumeId,Size:Size,Created:CreateTime}'
  
  Unused Elastic IPs:
    aws ec2 describe-addresses \
      --filters "Name=association-id,Values=" \
      --query 'Addresses[*].{IP:PublicIp,AllocId:AllocationId}'
  
  Old snapshots:
    aws ec2 describe-snapshots --owner-ids self \
      --query 'Snapshots[?StartTime<=` + "`" + `2023-06-01` + "`" + `]'
  
  Unattached load balancers:
    aws elbv2 describe-load-balancers | \
      jq '.LoadBalancers[] | select(.State.Code == "active")' | \
      while read lb; do
        TG_COUNT=$(aws elbv2 describe-target-groups \
          --load-balancer-arn "$lb" | jq '.TargetGroups | length')
        if [ "$TG_COUNT" = "0" ]; then echo "Unused: $lb"; fi
      done
` + "```" + ``,
					CodeExamples: `# Cost optimization scripts

# 1. Kubernetes cost analyzer
#!/bin/bash
echo "=== Kubernetes Cost Analysis ==="

# Resource usage vs allocation
echo "--- CPU: Allocated vs Used ---"
echo "Namespace | Requested | Used | Efficiency"
echo "----------|-----------|------|----------"

for ns in $(kubectl get ns -o name 2>/dev/null | cut -d'/' -f2); do
    REQUESTED=$(kubectl get pods -n "$ns" -o json 2>/dev/null | \
        jq '[.items[].spec.containers[].resources.requests.cpu // "0m" | 
        if endswith("m") then rtrimstr("m") | tonumber else tonumber * 1000 end] | add // 0')
    
    if [ "$REQUESTED" -gt 0 ] 2>/dev/null; then
        USED=$(kubectl top pods -n "$ns" --no-headers 2>/dev/null | \
            awk '{sum += $2} END {print sum+0}' | sed 's/m//')
        if [ -n "$USED" ] && [ "$USED" -gt 0 ] 2>/dev/null; then
            EFF=$(echo "scale=0; $USED * 100 / $REQUESTED" | bc 2>/dev/null || echo "N/A")
            echo "$ns | ${REQUESTED}m | ${USED}m | ${EFF}%"
        fi
    fi
done

# Over-provisioned pods
echo ""
echo "--- Over-Provisioned Pods (top 10) ---"
kubectl top pods --all-namespaces --no-headers 2>/dev/null | \
    sort -k3 -n | head -10

# Pods without resource requests
echo ""
echo "--- Pods Without Resource Requests ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.containers[].resources.requests == null) |
    "\(.metadata.namespace)/\(.metadata.name)"' 2>/dev/null | head -10

# 2. AWS cost report
#!/bin/bash
echo "=== AWS Cost Report ==="

# Current month costs by service
echo "--- Costs by Service (This Month) ---"
MONTH_START=$(date -u +"%Y-%m-01")
TODAY=$(date -u +"%Y-%m-%d")

aws ce get-cost-and-usage \
    --time-period "Start=$MONTH_START,End=$TODAY" \
    --granularity MONTHLY \
    --metrics "UnblendedCost" \
    --group-by Type=DIMENSION,Key=SERVICE \
    --query 'ResultsByTime[0].Groups[*].[Keys[0],Metrics.UnblendedCost.Amount]' \
    --output table 2>/dev/null | head -20

# Savings opportunities
echo ""
echo "--- Unused Resources ---"

# Unused EBS volumes
echo "Unattached EBS Volumes:"
aws ec2 describe-volumes \
    --filters Name=status,Values=available \
    --query 'Volumes[*].{ID:VolumeId,Size:Size,Type:VolumeType}' \
    --output table 2>/dev/null

# Unused Elastic IPs
echo ""
echo "Unused Elastic IPs:"
aws ec2 describe-addresses \
    --query 'Addresses[?AssociationId==null].{IP:PublicIp,AllocId:AllocationId}' \
    --output table 2>/dev/null

# Old snapshots (>90 days)
echo ""
echo "Old Snapshots (>90 days):"
CUTOFF=$(date -u -v-90d +"%Y-%m-%dT%H:%M:%S" 2>/dev/null || date -u -d "-90 days" +"%Y-%m-%dT%H:%M:%S")
aws ec2 describe-snapshots --owner-ids self \
    --query "Snapshots[?StartTime<='$CUTOFF'].{ID:SnapshotId,Size:VolumeSize,Created:StartTime}" \
    --output table 2>/dev/null | head -10

# 3. Cost anomaly detector
#!/bin/bash
echo "=== Cost Anomaly Detection ==="

# Compare current week to previous week
CUR_START=$(date -u -v-7d +"%Y-%m-%d" 2>/dev/null || date -u -d "-7 days" +"%Y-%m-%d")
CUR_END=$(date -u +"%Y-%m-%d")
PREV_START=$(date -u -v-14d +"%Y-%m-%d" 2>/dev/null || date -u -d "-14 days" +"%Y-%m-%d")
PREV_END="$CUR_START"

CURRENT=$(aws ce get-cost-and-usage \
    --time-period "Start=$CUR_START,End=$CUR_END" \
    --granularity DAILY \
    --metrics "UnblendedCost" \
    --query 'ResultsByTime[].Total.UnblendedCost.Amount' \
    --output text 2>/dev/null | awk '{sum+=$1} END {print sum}')

PREVIOUS=$(aws ce get-cost-and-usage \
    --time-period "Start=$PREV_START,End=$PREV_END" \
    --granularity DAILY \
    --metrics "UnblendedCost" \
    --query 'ResultsByTime[].Total.UnblendedCost.Amount' \
    --output text 2>/dev/null | awk '{sum+=$1} END {print sum}')

if [ -n "$CURRENT" ] && [ -n "$PREVIOUS" ]; then
    CHANGE=$(echo "scale=1; ($CURRENT - $PREVIOUS) / $PREVIOUS * 100" | bc 2>/dev/null)
    echo "Current week: \$$CURRENT"
    echo "Previous week: \$$PREVIOUS"
    echo "Change: ${CHANGE}%"
    
    THRESHOLD=20
    if (( $(echo "$CHANGE > $THRESHOLD" | bc -l 2>/dev/null) )); then
        echo ""
        echo "WARNING: Cost increase exceeds ${THRESHOLD}% threshold"
    fi
fi`,
				},
				{
					Title: "Spot Instances and Reserved Capacity",
					Content: `Optimizing compute costs through strategic use of spot instances, reserved instances, and savings plans.

**Spot Instance Strategies:**
` + "```" + `
AWS Spot Instances:
  Up to 90% discount vs on-demand.
  Can be interrupted with 2-minute warning.
  
  Best practices:
    Diversify instance types:
      Use multiple instance families and sizes
      Spread across availability zones
      Use capacity-optimized allocation strategy
    
    Handle interruptions:
      Monitor termination notices
      Use spot instance interruption handler
      Implement graceful shutdown
      Save state to persistent storage
  
  Auto Scaling with Spot:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      MixedInstancesPolicy:
        InstancesDistribution:
          OnDemandBaseCapacity: 2
          OnDemandPercentageAboveBaseCapacity: 20
          SpotAllocationStrategy: capacity-optimized-prioritized
          SpotMaxPrice: "" # Use on-demand price
        LaunchTemplate:
          LaunchTemplateSpecification:
            LaunchTemplateId: !Ref LaunchTemplate
            Version: !GetAtt LaunchTemplate.LatestVersionNumber
          Overrides:
            - InstanceType: m5.xlarge
            - InstanceType: m5a.xlarge
            - InstanceType: m5d.xlarge
            - InstanceType: m6i.xlarge
            - InstanceType: m6a.xlarge
  
  EKS with Karpenter (Spot):
    apiVersion: karpenter.sh/v1beta1
    kind: NodePool
    metadata:
      name: spot-pool
    spec:
      template:
        spec:
          requirements:
            - key: kubernetes.io/arch
              operator: In
              values: ["amd64"]
            - key: karpenter.sh/capacity-type
              operator: In
              values: ["spot"]
            - key: karpenter.k8s.aws/instance-category
              operator: In
              values: ["c", "m", "r"]
            - key: karpenter.k8s.aws/instance-generation
              operator: Gt
              values: ["4"]
          nodeClassRef:
            name: default
      limits:
        cpu: "100"
        memory: 400Gi
      disruption:
        consolidationPolicy: WhenUnderutilized
        expireAfter: 720h

GCP Preemptible/Spot VMs:
  Up to 91% discount.
  24-hour maximum lifetime (preemptible).
  Spot VMs: no max lifetime but can be reclaimed.
  
  GKE node pool:
    gcloud container node-pools create spot-pool \
      --cluster=my-cluster \
      --spot \
      --num-nodes=3 \
      --machine-type=n2-standard-4 \
      --enable-autoscaling \
      --min-nodes=0 \
      --max-nodes=10

Azure Spot VMs:
  Up to 90% discount.
  Eviction types: Capacity, Price.
  
  AKS node pool:
    az aks nodepool add \
      --resource-group myRG \
      --cluster-name myCluster \
      --name spotpool \
      --priority Spot \
      --eviction-policy Delete \
      --spot-max-price -1 \
      --node-count 3 \
      --enable-cluster-autoscaler \
      --min-count 0 \
      --max-count 10
` + "```" + `

**Reserved Capacity and Savings Plans:**
` + "```" + `
AWS Savings Plans:
  Compute Savings Plans:
    Flexible across EC2, Fargate, Lambda.
    1 or 3 year commitment.
    Up to 66% savings.
    
    aws savingsplans create-savings-plan \
      --savings-plan-offering-id offering-id \
      --commitment 100.00 \
      --savings-plan-type ComputeSavingsPlans
  
  EC2 Instance Savings Plans:
    Specific to instance family and region.
    Up to 72% savings.
  
  Reserved Instances:
    Standard: Up to 72% savings, limited modification.
    Convertible: Up to 66% savings, can change instance type.
    
    Payment options:
      All Upfront: Maximum discount
      Partial Upfront: Balance savings/cash flow
      No Upfront: Minimum commitment

GCP Committed Use Discounts:
  1 or 3 year commitment.
  Up to 57% discount (3 year).
  Specific to machine type family and region.
  
  gcloud compute commitments create my-commitment \
    --plan 36-month \
    --resources vcpu=100,memory=400GB \
    --region us-central1

Azure Reservations:
  1 or 3 year term.
  Up to 72% savings.
  Applies to VMs, SQL Database, Cosmos DB, etc.
  
  az reservations reservation-order purchase \
    --sku Standard_D2s_v3 \
    --location westus2 \
    --quantity 10 \
    --term P3Y

Cost optimization strategy:
  Layer 1: Steady-state workloads → Reserved/Savings Plans
  Layer 2: Scalable workloads → Spot/Preemptible
  Layer 3: Peak/burst demand → On-demand
  
  Recommended allocation:
    60-70% Reserved (base capacity)
    20-30% Spot (fault-tolerant workloads)
    10% On-demand (peak/critical)

Kubernetes scheduling for cost:
  Node affinity for spot:
    affinity:
      nodeAffinity:
        preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 80
            preference:
              matchExpressions:
                - key: karpenter.sh/capacity-type
                  operator: In
                  values: ["spot"]
  
  Topology spread for availability:
    topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: myapp
  
  Pod disruption budgets:
    apiVersion: policy/v1
    kind: PodDisruptionBudget
    metadata:
      name: myapp-pdb
    spec:
      minAvailable: 2  # or maxUnavailable: 1
      selector:
        matchLabels:
          app: myapp
` + "```" + ``,
					CodeExamples: `# Spot instance and capacity planning scripts

# 1. Spot instance price checker
#!/bin/bash
echo "=== Spot Instance Price History ==="

INSTANCE_TYPES=("m5.xlarge" "m5a.xlarge" "m6i.xlarge" "c5.xlarge" "r5.xlarge")
REGION="${AWS_REGION:-us-east-1}"

for INSTANCE_TYPE in "${INSTANCE_TYPES[@]}"; do
    echo ""
    echo "--- $INSTANCE_TYPE ---"
    
    # Current spot prices by AZ
    aws ec2 describe-spot-price-history \
        --instance-types "$INSTANCE_TYPE" \
        --product-descriptions "Linux/UNIX" \
        --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        --region "$REGION" \
        --query 'SpotPriceHistory[*].{AZ:AvailabilityZone,Price:SpotPrice}' \
        --output table 2>/dev/null
    
    # On-demand price for comparison
    OD_PRICE=$(aws pricing get-products \
        --service-code AmazonEC2 \
        --filters "Type=TERM_MATCH,Field=instanceType,Value=$INSTANCE_TYPE" \
                  "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
                  "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)" \
        --query 'PriceList[0]' \
        --output text 2>/dev/null | jq -r '.terms.OnDemand | to_entries[0].value.priceDimensions | to_entries[0].value.pricePerUnit.USD' 2>/dev/null)
    
    if [ -n "$OD_PRICE" ]; then
        echo "  On-demand: \$$OD_PRICE/hr"
    fi
done

# 2. Reserved instance coverage analyzer
#!/bin/bash
echo "=== RI Coverage Analysis ==="

# Current reservations
echo "--- Active Reservations ---"
aws ec2 describe-reserved-instances \
    --filters Name=state,Values=active \
    --query 'ReservedInstances[*].{Type:InstanceType,Count:InstanceCount,End:End,Offering:OfferingType}' \
    --output table 2>/dev/null

# Running instances
echo ""
echo "--- Running Instances ---"
aws ec2 describe-instances \
    --filters Name=instance-state-name,Values=running \
    --query 'Reservations[].Instances[].[InstanceType]' \
    --output text 2>/dev/null | sort | uniq -c | sort -rn

# Coverage gaps
echo ""
echo "--- RI Coverage ---"
aws ce get-reservation-coverage \
    --time-period "Start=$(date -u -v-30d +%Y-%m-%d 2>/dev/null || date -u -d '-30 days' +%Y-%m-%d),End=$(date -u +%Y-%m-%d)" \
    --group-by Type=DIMENSION,Key=INSTANCE_TYPE \
    --query 'CoveragesByTime[0].Groups[*].{Type:Attributes.instanceType,Coverage:Coverage.CoverageHours.CoverageHoursPercentage}' \
    --output table 2>/dev/null

# Savings Plan utilization
echo ""
echo "--- Savings Plan Utilization ---"
aws ce get-savings-plans-utilization \
    --time-period "Start=$(date -u -v-30d +%Y-%m-%d 2>/dev/null || date -u -d '-30 days' +%Y-%m-%d),End=$(date -u +%Y-%m-%d)" \
    --query 'Total.{Utilization:Utilization,Savings:AmortizedCommitment}' \
    --output table 2>/dev/null

# 3. Karpenter cost reporter
#!/bin/bash
echo "=== Karpenter Node Cost Report ==="

# Node types and capacity
kubectl get nodes -o json 2>/dev/null | \
    jq -r '.items[] | select(.metadata.labels["karpenter.sh/nodepool"] != null) |
    {
        name: .metadata.name,
        pool: .metadata.labels["karpenter.sh/nodepool"],
        type: .metadata.labels["node.kubernetes.io/instance-type"],
        capacity: .metadata.labels["karpenter.sh/capacity-type"],
        zone: .metadata.labels["topology.kubernetes.io/zone"],
        cpu: .status.capacity.cpu,
        memory: .status.capacity.memory
    } | "\(.pool) | \(.type) | \(.capacity) | \(.zone) | CPU:\(.cpu) Mem:\(.memory)"' 2>/dev/null

echo ""
echo "--- Node Utilization ---"
kubectl top nodes 2>/dev/null

echo ""
echo "--- Capacity Type Distribution ---"
echo "Spot nodes:"
kubectl get nodes -l "karpenter.sh/capacity-type=spot" --no-headers 2>/dev/null | wc -l | tr -d ' '
echo "On-demand nodes:"
kubectl get nodes -l "karpenter.sh/capacity-type=on-demand" --no-headers 2>/dev/null | wc -l | tr -d ' '`,
				},
			},
		},
	})
}
