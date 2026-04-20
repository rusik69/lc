package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1468,
			Title:       "Platform Engineering and Developer Experience",
			Description: "Build internal developer platforms with self-service capabilities, golden paths, and developer portals to improve productivity.",
			Order:       68,
			Lessons: []problems.Lesson{
				{
					Title: "Internal Developer Platforms",
					Content: `Platform Engineering creates self-service infrastructure abstractions that enable developers to focus on application logic while maintaining operational standards.

**Platform Architecture:**
` + "```" + `
Platform layers:
  Developer Interface:
    Service Catalog, CLI tools, Web Portal, APIs
  
  Platform Services:
    CI/CD pipelines, Observability, Security scanning
    Service mesh, Secrets management, DNS management
  
  Infrastructure Orchestration:
    Kubernetes operators, Terraform modules
    Cloud APIs, GitOps controllers
  
  Infrastructure:
    Cloud providers, Kubernetes clusters
    Networking, Storage, Compute

Backstage (Developer Portal):
  Spotify's open-source developer portal.
  
  Components:
    Software Catalog: Track all services, APIs, resources
    Software Templates: Create new services from templates
    TechDocs: Documentation-as-code
    Plugins: Extensible plugin architecture
  
  Catalog entity:
    apiVersion: backstage.io/v1alpha1
    kind: Component
    metadata:
      name: payment-service
      description: Payment processing service
      annotations:
        github.com/project-slug: myorg/payment-service
        backstage.io/techdocs-ref: dir:.
      tags:
        - go
        - grpc
        - payments
      links:
        - url: https://grafana.example.com/d/payment
          title: Grafana Dashboard
    spec:
      type: service
      lifecycle: production
      owner: payments-team
      system: checkout
      providesApis:
        - payment-api
      dependsOn:
        - component:user-service
        - resource:payment-db
  
  API entity:
    apiVersion: backstage.io/v1alpha1
    kind: API
    metadata:
      name: payment-api
      description: Payment processing API
    spec:
      type: grpc
      lifecycle: production
      owner: payments-team
      definition: |
        syntax = "proto3";
        service PaymentService {
          rpc ProcessPayment(PaymentRequest) returns (PaymentResponse);
          rpc GetPayment(GetPaymentRequest) returns (Payment);
        }
  
  Software template:
    apiVersion: scaffolder.backstage.io/v1beta3
    kind: Template
    metadata:
      name: go-service
      title: Go Microservice
      description: Create a new Go microservice
    spec:
      owner: platform-team
      type: service
      parameters:
        - title: Service Details
          required: [name, owner]
          properties:
            name:
              title: Service Name
              type: string
              pattern: '^[a-z][a-z0-9-]*$'
            owner:
              title: Owner
              type: string
              ui:field: OwnerPicker
            description:
              title: Description
              type: string
        
        - title: Infrastructure
          properties:
            database:
              title: Database
              type: string
              enum: [none, postgres, redis]
              default: none
            hasIngress:
              title: Public Endpoint
              type: boolean
              default: false
      
      steps:
        - id: fetch-template
          name: Fetch Template
          action: fetch:template
          input:
            url: ./skeleton
            values:
              name: ${{ parameters.name }}
              owner: ${{ parameters.owner }}
              database: ${{ parameters.database }}
        
        - id: publish
          name: Create Repository
          action: publish:github
          input:
            repoUrl: github.com?repo=${{ parameters.name }}&owner=myorg
            defaultBranch: main
        
        - id: register
          name: Register in Catalog
          action: catalog:register
          input:
            repoContentsUrl: ${{ steps.publish.output.repoContentsUrl }}
            catalogInfoPath: /catalog-info.yaml
      
      output:
        links:
          - title: Repository
            url: ${{ steps.publish.output.remoteUrl }}
          - title: Open in catalog
            icon: catalog
            entityRef: ${{ steps.register.output.entityRef }}
` + "```" + `

**Crossplane and Infrastructure Abstraction:**
` + "```" + `
Crossplane:
  Kubernetes-native infrastructure management.
  Compose cloud resources into higher-level abstractions.
  
  Managed resource (low-level):
    apiVersion: rds.aws.upbound.io/v1beta1
    kind: Instance
    metadata:
      name: payment-db
    spec:
      forProvider:
        region: us-east-1
        instanceClass: db.t3.medium
        engine: postgres
        engineVersion: "15"
        allocatedStorage: 20
        masterUsername: admin
      writeConnectionSecretToRef:
        name: payment-db-creds
        namespace: payments
  
  Composite Resource Definition (XRD):
    apiVersion: apiextensions.crossplane.io/v1
    kind: CompositeResourceDefinition
    metadata:
      name: xdatabases.platform.example.com
    spec:
      group: platform.example.com
      names:
        kind: XDatabase
        plural: xdatabases
      claimNames:
        kind: Database
        plural: databases
      versions:
        - name: v1alpha1
          served: true
          referenceable: true
          schema:
            openAPIV3Schema:
              type: object
              properties:
                spec:
                  type: object
                  properties:
                    engine:
                      type: string
                      enum: [postgres, mysql]
                    size:
                      type: string
                      enum: [small, medium, large]
                    region:
                      type: string
                      default: us-east-1
  
  Composition:
    apiVersion: apiextensions.crossplane.io/v1
    kind: Composition
    metadata:
      name: database-aws
    spec:
      compositeTypeRef:
        apiVersion: platform.example.com/v1alpha1
        kind: XDatabase
      resources:
        - name: rds-instance
          base:
            apiVersion: rds.aws.upbound.io/v1beta1
            kind: Instance
            spec:
              forProvider:
                engine: postgres
                engineVersion: "15"
          patches:
            - type: FromCompositeFieldPath
              fromFieldPath: spec.size
              toFieldPath: spec.forProvider.instanceClass
              transforms:
                - type: map
                  map:
                    small: db.t3.small
                    medium: db.t3.medium
                    large: db.r5.large
            - type: FromCompositeFieldPath
              fromFieldPath: spec.region
              toFieldPath: spec.forProvider.region
  
  Developer claim (self-service):
    apiVersion: platform.example.com/v1alpha1
    kind: Database
    metadata:
      name: my-database
      namespace: payments
    spec:
      engine: postgres
      size: medium
      region: us-east-1
` + "```" + ``,
					CodeExamples: `# Platform engineering scripts

# 1. Service scaffolding tool
#!/bin/bash
set -e

SERVICE_NAME="${1:?Usage: $0 <service-name> [template]}"
TEMPLATE="${2:-go-service}"
BASE_DIR="${3:-.}"

echo "=== Creating Service: $SERVICE_NAME ==="

# Validate name
if ! echo "$SERVICE_NAME" | grep -qE '^[a-z][a-z0-9-]*$'; then
    echo "Error: Service name must match ^[a-z][a-z0-9-]*$"
    exit 1
fi

SERVICE_DIR="$BASE_DIR/$SERVICE_NAME"
mkdir -p "$SERVICE_DIR"

case "$TEMPLATE" in
    go-service)
        echo "Using Go service template..."
        
        # Go module
        mkdir -p "$SERVICE_DIR"/{cmd/server,internal/{config,handlers,middleware},api,deployments}
        
        cat > "$SERVICE_DIR/go.mod" << EOF
module github.com/myorg/$SERVICE_NAME

go 1.22
EOF
        
        # Main
        cat > "$SERVICE_DIR/cmd/server/main.go" << 'GOEOF'
package main

import (
    "log"
    "net/http"
    "os"
)

func main() {
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }
    
    mux := http.NewServeMux()
    mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("ok"))
    })
    
    log.Printf("Starting server on :%s", port)
    if err := http.ListenAndServe(":"+port, mux); err != nil {
        log.Fatal(err)
    }
}
GOEOF
        
        # Dockerfile
        cat > "$SERVICE_DIR/Dockerfile" << DEOF
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /server ./cmd/server

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /server /server
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/server"]
DEOF
        
        # Kubernetes manifests
        cat > "$SERVICE_DIR/deployments/deployment.yaml" << KEOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $SERVICE_NAME
  labels:
    app: $SERVICE_NAME
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $SERVICE_NAME
  template:
    metadata:
      labels:
        app: $SERVICE_NAME
    spec:
      containers:
        - name: $SERVICE_NAME
          image: myregistry/$SERVICE_NAME:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
      securityContext:
        runAsNonRoot: true
KEOF
        
        # Makefile
        cat > "$SERVICE_DIR/Makefile" << 'MEOF'
.PHONY: build test run docker-build

build:
	go build -o bin/server ./cmd/server

test:
	go test ./... -v

run:
	go run ./cmd/server

docker-build:
	docker build -t $(SERVICE_NAME):latest .
MEOF
        ;;
esac

# Catalog info for Backstage
cat > "$SERVICE_DIR/catalog-info.yaml" << CEOF
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: $SERVICE_NAME
  description: $SERVICE_NAME service
spec:
  type: service
  lifecycle: experimental
  owner: team-unknown
CEOF

echo ""
echo "Service created at $SERVICE_DIR"
ls -la "$SERVICE_DIR"

# 2. Platform health checker
#!/bin/bash
echo "=== Platform Health Check ==="

CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_TOTAL=0

check_component() {
    local name="$1" ns="$2" label="$3"
    ((CHECKS_TOTAL++))
    
    READY=$(kubectl get pods -n "$ns" -l "$label" --no-headers 2>/dev/null | \
        awk '{split($2,a,"/"); if(a[1]==a[2]) print "ready"}' | wc -l | tr -d ' ')
    TOTAL=$(kubectl get pods -n "$ns" -l "$label" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    
    if [ "$READY" = "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        echo "  [OK]   $name ($READY/$TOTAL)"
        ((CHECKS_PASSED++))
    else
        echo "  [FAIL] $name ($READY/$TOTAL)"
        ((CHECKS_FAILED++))
    fi
}

echo "--- Core Platform ---"
check_component "ArgoCD" "argocd" "app.kubernetes.io/name=argocd-server"
check_component "Cert-Manager" "cert-manager" "app.kubernetes.io/name=cert-manager"
check_component "Ingress Controller" "ingress-nginx" "app.kubernetes.io/name=ingress-nginx"
check_component "External DNS" "external-dns" "app.kubernetes.io/name=external-dns"

echo ""
echo "--- Observability ---"
check_component "Prometheus" "monitoring" "app.kubernetes.io/name=prometheus"
check_component "Grafana" "monitoring" "app.kubernetes.io/name=grafana"
check_component "Loki" "monitoring" "app.kubernetes.io/name=loki"

echo ""
echo "--- Security ---"
check_component "Kyverno" "kyverno" "app.kubernetes.io/name=kyverno"
check_component "Falco" "falco" "app.kubernetes.io/name=falco"
check_component "Trivy Operator" "trivy-system" "app.kubernetes.io/name=trivy-operator"

echo ""
echo "--- Summary ---"
echo "Passed: $CHECKS_PASSED/$CHECKS_TOTAL"
if [ "$CHECKS_FAILED" -gt 0 ]; then
    echo "WARNING: $CHECKS_FAILED components unhealthy"
fi`,
				},
				{
					Title: "Golden Paths and Developer Self-Service",
					Content: `Golden paths provide opinionated, well-maintained workflows that guide developers toward best practices while maintaining flexibility.

**Golden Path Design:**
` + "```" + `
Golden path components:
  Repository Templates:
    Standardized project structure
    Pre-configured CI/CD pipelines
    Default security scanning
    Consistent observability setup
    Standard Kubernetes manifests
  
  Service Standards:
    Language/framework choices
    API design guidelines
    Testing requirements
    Documentation templates
    Deployment patterns

Example golden path (Go microservice):
  Repository structure:
    service-name/
    ├── cmd/
    │   └── server/
    │       └── main.go
    ├── internal/
    │   ├── config/
    │   ├── handlers/
    │   ├── middleware/
    │   └── service/
    ├── api/
    │   └── openapi.yaml
    ├── deployments/
    │   ├── base/
    │   │   ├── kustomization.yaml
    │   │   ├── deployment.yaml
    │   │   ├── service.yaml
    │   │   └── hpa.yaml
    │   ├── overlays/
    │   │   ├── staging/
    │   │   └── production/
    │   └── helm/
    │       └── Chart.yaml
    ├── .github/
    │   └── workflows/
    │       ├── ci.yaml
    │       ├── cd.yaml
    │       └── security.yaml
    ├── Dockerfile
    ├── Makefile
    ├── go.mod
    ├── catalog-info.yaml
    └── README.md

CI/CD golden path:
  name: CI Pipeline
  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]
  
  jobs:
    lint:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-go@v5
        - uses: golangci/golangci-lint-action@v4
    
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-go@v5
        - run: go test -race -coverprofile=coverage.out ./...
        - run: go tool cover -func=coverage.out
    
    security:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: |
            go install golang.org/x/vuln/cmd/govulncheck@latest
            govulncheck ./...
        - uses: securego/gosec@master
    
    build:
      needs: [lint, test, security]
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: docker/build-push-action@v5
          with:
            push: true
            tags: myregistry/${{ github.repository }}:${{ github.sha }}
    
    deploy-staging:
      needs: [build]
      if: github.ref == 'refs/heads/main'
      runs-on: ubuntu-latest
      environment: staging
      steps:
        - uses: actions/checkout@v4
        - run: |
            cd deployments/overlays/staging
            kustomize edit set image app=myregistry/${{ github.repository }}:${{ github.sha }}
        - uses: stefanzweifel/git-auto-commit-action@v5
          with:
            commit_message: "deploy: staging ${{ github.sha }}"
` + "```" + `

**Developer CLI and Self-Service:**
` + "```" + `
Platform CLI:
  Common commands:
    platform create service --name my-service --template go
    platform create database --engine postgres --size medium
    platform create cache --engine redis --size small
    platform deploy --env staging
    platform logs --service my-service --env production
    platform rollback --service my-service --revision 5
    platform status --service my-service
    platform cost --team my-team --period monthly

Self-service portal features:
  Service creation:
    Choose template → Configure → Generate → Deploy
    Automatic Git repo creation
    CI/CD pipeline setup
    Monitoring dashboards
    Alert rules
  
  Environment management:
    Create preview environments from PRs
    Promote between environments
    Environment TTL (auto-cleanup)
  
  Database provisioning:
    Choose engine and size
    Automatic backup configuration
    Connection string injection
    Credential rotation
  
  Observability:
    Auto-instrumented services
    Default dashboards
    Standard alert rules
    Log aggregation

Score/Humanitec style platform:
  Workload specification (Score):
    apiVersion: score.dev/v1b1
    metadata:
      name: my-service
    containers:
      main:
        image: .
        variables:
          PORT: "8080"
          DB_HOST: "${resources.db.host}"
          DB_PORT: "${resources.db.port}"
          DB_NAME: "${resources.db.name}"
    resources:
      db:
        type: postgres
      dns:
        type: dns
      cache:
        type: redis
    
    Benefits:
      Developer-centric workload spec
      Platform resolves resource bindings
      Same spec across environments
      No Kubernetes knowledge needed

Platform metrics:
  Developer productivity:
    Time to first deployment
    Deployment frequency
    Lead time for changes
    Mean time to recovery
  
  Platform adoption:
    Services using golden paths
    Template usage by team
    Self-service vs ticket requests
    Developer satisfaction (surveys)
  
  Platform reliability:
    Platform component uptime
    CI/CD pipeline success rate
    Mean time to provision
    Support ticket volume
` + "```" + ``,
					CodeExamples: `# Developer experience tools

# 1. Environment provisioner
#!/bin/bash
set -e

ACTION="${1:?Usage: $0 <create|delete|list> [options]}"
ENV_NAME="${2}"
NAMESPACE_PREFIX="preview"
TTL_HOURS="${TTL_HOURS:-24}"

case "$ACTION" in
    create)
        [ -z "$ENV_NAME" ] && { echo "Error: env name required"; exit 1; }
        NS="${NAMESPACE_PREFIX}-${ENV_NAME}"
        
        echo "Creating preview environment: $NS"
        
        # Create namespace with TTL annotation
        EXPIRY=$(date -u -v+"${TTL_HOURS}H" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || \
                 date -u -d "+${TTL_HOURS} hours" +%Y-%m-%dT%H:%M:%SZ)
        
        kubectl create namespace "$NS" --dry-run=client -o yaml | \
            kubectl apply -f - 2>/dev/null
        kubectl annotate namespace "$NS" \
            "platform.example.com/expires=$EXPIRY" \
            "platform.example.com/owner=$(git config user.email)" \
            --overwrite 2>/dev/null
        kubectl label namespace "$NS" \
            "environment=preview" \
            "managed-by=platform-cli" \
            --overwrite 2>/dev/null
        
        # Apply resource quota
        cat <<EOF | kubectl apply -f - 2>/dev/null
apiVersion: v1
kind: ResourceQuota
metadata:
  name: preview-quota
  namespace: $NS
spec:
  hard:
    requests.cpu: "2"
    requests.memory: 4Gi
    limits.cpu: "4"
    limits.memory: 8Gi
    pods: "20"
    services: "10"
EOF
        
        echo "Environment created: $NS"
        echo "Expires: $EXPIRY"
        echo ""
        echo "Deploy with:"
        echo "  kubectl apply -k deployments/overlays/preview -n $NS"
        ;;
    
    delete)
        [ -z "$ENV_NAME" ] && { echo "Error: env name required"; exit 1; }
        NS="${NAMESPACE_PREFIX}-${ENV_NAME}"
        
        echo "Deleting preview environment: $NS"
        kubectl delete namespace "$NS" --wait=false 2>/dev/null
        echo "Namespace deletion initiated"
        ;;
    
    list)
        echo "=== Preview Environments ==="
        echo "Namespace | Owner | Expires | Pods"
        echo "----------|-------|---------|-----"
        
        for ns in $(kubectl get ns -l environment=preview -o name 2>/dev/null | cut -d'/' -f2); do
            OWNER=$(kubectl get ns "$ns" -o jsonpath='{.metadata.annotations.platform\.example\.com/owner}' 2>/dev/null || echo "unknown")
            EXPIRES=$(kubectl get ns "$ns" -o jsonpath='{.metadata.annotations.platform\.example\.com/expires}' 2>/dev/null || echo "never")
            PODS=$(kubectl get pods -n "$ns" --no-headers 2>/dev/null | wc -l | tr -d ' ')
            echo "$ns | $OWNER | $EXPIRES | $PODS"
        done
        ;;
    
    cleanup)
        echo "=== Cleaning Up Expired Environments ==="
        NOW=$(date -u +%s)
        
        for ns in $(kubectl get ns -l environment=preview -o name 2>/dev/null | cut -d'/' -f2); do
            EXPIRES=$(kubectl get ns "$ns" -o jsonpath='{.metadata.annotations.platform\.example\.com/expires}' 2>/dev/null)
            if [ -n "$EXPIRES" ]; then
                EXPIRY_TS=$(date -jf "%Y-%m-%dT%H:%M:%SZ" "$EXPIRES" +%s 2>/dev/null || \
                           date -d "$EXPIRES" +%s 2>/dev/null)
                if [ -n "$EXPIRY_TS" ] && [ "$NOW" -gt "$EXPIRY_TS" ]; then
                    echo "Deleting expired: $ns (expired: $EXPIRES)"
                    kubectl delete namespace "$ns" --wait=false 2>/dev/null
                fi
            fi
        done
        ;;
    
    *)
        echo "Usage: $0 <create|delete|list|cleanup> [env-name]"
        exit 1
        ;;
esac

# 2. Service status dashboard
#!/bin/bash
echo "=== Service Status Dashboard ==="

NAMESPACE="${1:-production}"

echo "Namespace: $NAMESPACE"
echo ""

# Deployments
echo "--- Deployments ---"
echo "Name | Ready | Up-to-date | Available | Age"
echo "-----|-------|------------|-----------|----"
kubectl get deployments -n "$NAMESPACE" --no-headers 2>/dev/null | \
    awk '{printf "%s | %s | %s | %s | %s\n", $1, $2, $3, $4, $5}'

# Services
echo ""
echo "--- Services ---"
kubectl get services -n "$NAMESPACE" --no-headers 2>/dev/null | \
    awk '{printf "%s | %s | %s | %s\n", $1, $2, $4, $5}'

# Ingresses
echo ""
echo "--- Ingresses ---"
kubectl get ingress -n "$NAMESPACE" --no-headers 2>/dev/null | \
    awk '{printf "%s | %s | %s\n", $1, $3, $4}'

# Recent events
echo ""
echo "--- Recent Events ---"
kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' 2>/dev/null | \
    grep -E "Warning|Error" | tail -5

# HPA status
echo ""
echo "--- Autoscaler Status ---"
kubectl get hpa -n "$NAMESPACE" --no-headers 2>/dev/null | \
    awk '{printf "%s | Min:%s Max:%s Current:%s | %s\n", $1, $3, $4, $6, $2}'`,
				},
			},
		},
		{
			ID:          1469,
			Title:       "Database Operations and Reliability",
			Description: "Manage database operations including migrations, backups, replication, performance tuning, and disaster recovery strategies.",
			Order:       69,
			Lessons: []problems.Lesson{
				{
					Title: "Database Migration and Schema Management",
					Content: `Database operations (DBOps) ensures databases are reliable, performant, and safely managed throughout the application lifecycle.

**Schema Migration Tools:**
` + "```" + `
Migration tools by database:
  General:
    Flyway:
      flyway migrate
      flyway info
      flyway validate
      flyway repair
      
      Configuration (flyway.conf):
        flyway.url=jdbc:postgresql://localhost:5432/mydb
        flyway.user=admin
        flyway.locations=filesystem:sql/migrations
        flyway.baselineOnMigrate=true
    
    Liquibase:
      liquibase update
      liquibase rollbackCount 1
      liquibase status
      
      Changeset:
        databaseChangeLog:
          - changeSet:
              id: 1
              author: dev
              changes:
                - createTable:
                    tableName: users
                    columns:
                      - column:
                          name: id
                          type: bigint
                          autoIncrement: true
                          constraints:
                            primaryKey: true
                      - column:
                          name: email
                          type: varchar(255)
                          constraints:
                            nullable: false
                            unique: true
  
  Go:
    golang-migrate:
      migrate -path migrations -database "postgres://localhost/mydb?sslmode=disable" up
      migrate -path migrations -database "postgres://localhost/mydb?sslmode=disable" down 1
      migrate create -ext sql -dir migrations -seq add_users
    
    goose:
      goose postgres "postgres://localhost/mydb" up
      goose postgres "postgres://localhost/mydb" down
      goose create add_users sql
    
    Atlas (declarative):
      atlas schema apply \
        --url "postgres://localhost:5432/mydb?sslmode=disable" \
        --to "file://schema.hcl" \
        --dev-url "docker://postgres/15"
  
  Python:
    Alembic:
      alembic revision --autogenerate -m "add users"
      alembic upgrade head
      alembic downgrade -1
      alembic history

Migration best practices:
  Naming:
    V001__create_users_table.sql
    V002__add_email_index.sql
    V003__create_orders_table.sql
  
  Forward-only migrations:
    Always write reversible migrations where possible.
    Separate deploy from migration.
    Use expand-and-contract pattern.
  
  Expand and contract:
    Phase 1 (Expand):
      Add new column (nullable)
      Deploy code that writes to both old and new
      Backfill data
    
    Phase 2 (Migrate):
      Deploy code that reads from new column
      Verify correctness
    
    Phase 3 (Contract):
      Remove old column
      Deploy code without old column references
  
  Online schema changes (large tables):
    pt-online-schema-change (MySQL):
      pt-online-schema-change \
        --alter "ADD COLUMN status VARCHAR(20)" \
        D=mydb,t=users --execute
    
    gh-ost (MySQL):
      gh-ost \
        --host=db.example.com \
        --database=mydb \
        --table=users \
        --alter="ADD COLUMN status VARCHAR(20)" \
        --execute
    
    PostgreSQL (concurrent):
      CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
      ALTER TABLE users ADD COLUMN status VARCHAR(20);
      -- No lock for adding nullable column in PostgreSQL
` + "```" + `

**Backup and Recovery:**
` + "```" + `
PostgreSQL backups:
  Logical backup:
    pg_dump -h localhost -U admin mydb > backup.sql
    pg_dump -Fc -h localhost -U admin mydb > backup.dump
    pg_dumpall > all_databases.sql
    
    Restore:
      pg_restore -h localhost -U admin -d mydb backup.dump
      psql -h localhost -U admin mydb < backup.sql
  
  Physical backup (WAL archiving):
    postgresql.conf:
      wal_level = replica
      archive_mode = on
      archive_command = 'aws s3 cp %p s3://backups/wal/%f'
    
    pgBackRest:
      pgbackrest --stanza=mydb --type=full backup
      pgbackrest --stanza=mydb --type=diff backup
      pgbackrest --stanza=mydb --type=incr backup
      pgbackrest --stanza=mydb --target="2024-01-15 10:00:00" \
        --type=time restore
    
    Barman:
      barman backup mydb
      barman list-backup mydb
      barman recover mydb latest /var/lib/pgsql/data

MySQL backups:
  mysqldump:
    mysqldump --single-transaction --routines mydb > backup.sql
    mysqldump --all-databases > all.sql
  
  Percona XtraBackup:
    xtrabackup --backup --target-dir=/backups/full
    xtrabackup --backup --target-dir=/backups/incr --incremental-basedir=/backups/full
    xtrabackup --prepare --target-dir=/backups/full
    xtrabackup --prepare --target-dir=/backups/full --incremental-dir=/backups/incr

Kubernetes database backups:
  Velero:
    velero backup create db-backup \
      --include-namespaces databases \
      --include-resources pvc,pv
    
    velero schedule create daily-db \
      --schedule="0 2 * * *" \
      --include-namespaces databases
    
    velero restore create --from-backup db-backup
  
  Stash (by AppsCode):
    apiVersion: stash.appscode.com/v1beta1
    kind: BackupConfiguration
    metadata:
      name: postgres-backup
    spec:
      schedule: "0 */6 * * *"
      repository:
        name: s3-repo
      target:
        ref:
          apiVersion: apps/v1
          kind: StatefulSet
          name: postgres
      retentionPolicy:
        name: keep-last-5
        keepLast: 5
        prune: true

Backup verification:
  Test restores regularly:
    Restore to staging daily/weekly
    Verify data integrity
    Measure recovery time (RTO)
    Verify recovery point (RPO)
  
  Automated verification:
    Restore backup to test instance
    Run integrity checks
    Compare row counts
    Run application health checks
    Alert on failures
` + "```" + ``,
					CodeExamples: `# Database operations scripts

# 1. Database migration manager
#!/bin/bash
set -e

ACTION="${1:?Usage: $0 <migrate|rollback|status|create> [options]}"
DB_URL="${DATABASE_URL:?DATABASE_URL not set}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-./migrations}"

case "$ACTION" in
    migrate)
        echo "=== Running Migrations ==="
        if command -v migrate &>/dev/null; then
            migrate -path "$MIGRATIONS_DIR" -database "$DB_URL" up
        elif command -v goose &>/dev/null; then
            goose -dir "$MIGRATIONS_DIR" postgres "$DB_URL" up
        else
            echo "No migration tool found (install golang-migrate or goose)"
            exit 1
        fi
        echo "Migrations complete"
        ;;
    
    rollback)
        STEPS="${2:-1}"
        echo "=== Rolling Back $STEPS Migration(s) ==="
        if command -v migrate &>/dev/null; then
            migrate -path "$MIGRATIONS_DIR" -database "$DB_URL" down "$STEPS"
        elif command -v goose &>/dev/null; then
            for i in $(seq 1 "$STEPS"); do
                goose -dir "$MIGRATIONS_DIR" postgres "$DB_URL" down
            done
        fi
        echo "Rollback complete"
        ;;
    
    status)
        echo "=== Migration Status ==="
        if command -v migrate &>/dev/null; then
            migrate -path "$MIGRATIONS_DIR" -database "$DB_URL" version
        elif command -v goose &>/dev/null; then
            goose -dir "$MIGRATIONS_DIR" postgres "$DB_URL" status
        fi
        ;;
    
    create)
        NAME="${2:?Migration name required}"
        echo "=== Creating Migration: $NAME ==="
        if command -v migrate &>/dev/null; then
            migrate create -ext sql -dir "$MIGRATIONS_DIR" -seq "$NAME"
        elif command -v goose &>/dev/null; then
            goose -dir "$MIGRATIONS_DIR" create "$NAME" sql
        fi
        echo "Migration files created"
        ;;
    
    *)
        echo "Usage: $0 <migrate|rollback|status|create> [options]"
        exit 1
        ;;
esac

# 2. Backup manager
#!/bin/bash
set -e

ACTION="${1:?Usage: $0 <backup|restore|list|verify> [options]}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:?DB_NAME required}"
DB_USER="${DB_USER:-postgres}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

case "$ACTION" in
    backup)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.dump"
        
        echo "=== Creating Backup ==="
        echo "Database: $DB_NAME"
        echo "File: $BACKUP_FILE"
        
        mkdir -p "$BACKUP_DIR"
        
        pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            -Fc --no-acl --no-owner "$DB_NAME" > "$BACKUP_FILE"
        
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        echo "Backup complete: $SIZE"
        
        # Upload to S3 if configured
        if [ -n "$BACKUP_S3_BUCKET" ]; then
            echo "Uploading to S3..."
            aws s3 cp "$BACKUP_FILE" "s3://$BACKUP_S3_BUCKET/backups/$DB_NAME/" 2>/dev/null
            echo "Upload complete"
        fi
        
        # Cleanup old backups
        echo "Cleaning up backups older than $RETENTION_DAYS days..."
        find "$BACKUP_DIR" -name "${DB_NAME}_*.dump" -mtime +"$RETENTION_DAYS" -delete 2>/dev/null
        ;;
    
    restore)
        BACKUP_FILE="${2:?Backup file required}"
        TARGET_DB="${3:-${DB_NAME}_restored}"
        
        echo "=== Restoring Backup ==="
        echo "From: $BACKUP_FILE"
        echo "To: $TARGET_DB"
        
        # Create target database
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$TARGET_DB" 2>/dev/null || true
        
        pg_restore -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            -d "$TARGET_DB" --no-acl --no-owner "$BACKUP_FILE"
        
        echo "Restore complete"
        ;;
    
    list)
        echo "=== Available Backups ==="
        ls -lh "$BACKUP_DIR"/${DB_NAME}_*.dump 2>/dev/null || echo "No backups found"
        
        if [ -n "$BACKUP_S3_BUCKET" ]; then
            echo ""
            echo "--- S3 Backups ---"
            aws s3 ls "s3://$BACKUP_S3_BUCKET/backups/$DB_NAME/" 2>/dev/null || echo "No S3 backups"
        fi
        ;;
    
    verify)
        BACKUP_FILE="${2}"
        if [ -z "$BACKUP_FILE" ]; then
            BACKUP_FILE=$(ls -t "$BACKUP_DIR"/${DB_NAME}_*.dump 2>/dev/null | head -1)
        fi
        
        [ -z "$BACKUP_FILE" ] && { echo "No backup found"; exit 1; }
        
        echo "=== Verifying Backup ==="
        echo "File: $BACKUP_FILE"
        
        VERIFY_DB="${DB_NAME}_verify_$$"
        
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$VERIFY_DB" 2>/dev/null
        
        if pg_restore -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            -d "$VERIFY_DB" --no-acl --no-owner "$BACKUP_FILE" 2>/dev/null; then
            
            TABLES=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$VERIFY_DB" \
                -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null | tr -d ' ')
            echo "Tables: $TABLES"
            echo "Verification: PASSED"
        else
            echo "Verification: FAILED"
        fi
        
        dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$VERIFY_DB" 2>/dev/null
        ;;
esac`,
				},
				{
					Title: "Database Replication and Performance",
					Content: `Database replication ensures high availability and read scalability, while performance tuning maximizes throughput.

**Replication Strategies:**
` + "```" + `
PostgreSQL replication:
  Streaming replication (async):
    Primary (postgresql.conf):
      wal_level = replica
      max_wal_senders = 10
      wal_keep_size = 1GB
      hot_standby = on
    
    Primary (pg_hba.conf):
      host replication replicator 10.0.0.0/8 scram-sha-256
    
    Replica setup:
      pg_basebackup -h primary -U replicator -D /var/lib/postgresql/data -P -R
      # -R creates standby.signal and configures primary_conninfo
    
    Verify:
      SELECT * FROM pg_stat_replication;  -- On primary
      SELECT * FROM pg_stat_wal_receiver; -- On replica
  
  Synchronous replication:
    postgresql.conf (primary):
      synchronous_standby_names = 'FIRST 1 (replica1, replica2)'
      synchronous_commit = on
    
    Tradeoff: Data safety vs latency.
  
  Logical replication:
    -- On publisher
    CREATE PUBLICATION my_pub FOR TABLE users, orders;
    
    -- On subscriber
    CREATE SUBSCRIPTION my_sub
      CONNECTION 'host=publisher dbname=mydb'
      PUBLICATION my_pub;
    
    Benefits: Selective table replication, cross-version,
              different indexes on replica.

MySQL replication:
  Source (my.cnf):
    server-id = 1
    log_bin = mysql-bin
    binlog_format = ROW
    gtid_mode = ON
    enforce_gtid_consistency = ON
  
  Replica:
    CHANGE REPLICATION SOURCE TO
      SOURCE_HOST='primary',
      SOURCE_USER='replicator',
      SOURCE_AUTO_POSITION=1;
    START REPLICA;
    SHOW REPLICA STATUS\G
  
  Group Replication:
    Multi-primary or single-primary mode.
    Automatic failover.
    Conflict detection.

Kubernetes database operators:
  CloudNativePG (PostgreSQL):
    apiVersion: postgresql.cnpg.io/v1
    kind: Cluster
    metadata:
      name: mydb
    spec:
      instances: 3
      postgresql:
        parameters:
          max_connections: "200"
          shared_buffers: 256MB
      storage:
        size: 10Gi
        storageClass: gp3
      backup:
        barmanObjectStore:
          destinationPath: s3://backups/mydb
          s3Credentials:
            accessKeyID:
              name: aws-creds
              key: ACCESS_KEY_ID
            secretAccessKey:
              name: aws-creds
              key: SECRET_ACCESS_KEY
          wal:
            compression: gzip
        retentionPolicy: "30d"
      monitoring:
        enablePodMonitor: true
  
  Percona Operator (MySQL):
    apiVersion: pxc.percona.com/v1
    kind: PerconaXtraDBCluster
    metadata:
      name: mysql-cluster
    spec:
      pxc:
        size: 3
        resources:
          requests:
            memory: 1G
            cpu: 500m
        volumeSpec:
          persistentVolumeClaim:
            storageClassName: gp3
            resources:
              requests:
                storage: 20Gi
      haproxy:
        size: 2
      backup:
        schedule:
          - name: daily
            schedule: "0 2 * * *"
            keep: 7
            storageName: s3-backup
` + "```" + `

**Performance Tuning:**
` + "```" + `
PostgreSQL tuning:
  Memory:
    shared_buffers = 25% of RAM (e.g., 4GB for 16GB)
    effective_cache_size = 75% of RAM
    work_mem = RAM / (max_connections * 4)
    maintenance_work_mem = RAM / 16
  
  Write performance:
    wal_buffers = 64MB
    checkpoint_timeout = 15min
    checkpoint_completion_target = 0.9
    max_wal_size = 4GB
  
  Query tuning:
    random_page_cost = 1.1 (SSD) or 4.0 (HDD)
    effective_io_concurrency = 200 (SSD)
    default_statistics_target = 100 (or higher for complex queries)
  
  Connections:
    max_connections = 200
    Use PgBouncer for connection pooling:
      [databases]
      mydb = host=localhost port=5432 dbname=mydb
      
      [pgbouncer]
      pool_mode = transaction
      max_client_conn = 1000
      default_pool_size = 50
      min_pool_size = 10

Query analysis:
  EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT ...;
  
  Key metrics:
    Seq Scan: Full table scan (bad for large tables)
    Index Scan: Using index (good)
    Sort: In-memory vs disk
    Hash Join: Memory usage
    Buffers: Shared hit vs read
  
  Index strategies:
    B-tree: Default, equality and range queries
    Hash: Equality only
    GIN: Full-text search, arrays, JSONB
    GiST: Geometry, full-text search
    BRIN: Large tables with natural ordering
  
  Common optimizations:
    CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
    CREATE INDEX idx_orders_user_created ON orders(user_id, created_at DESC);
    CREATE INDEX idx_products_data ON products USING gin(metadata jsonb_path_ops);
    
    Partial indexes:
      CREATE INDEX idx_active_users ON users(email) WHERE active = true;
    
    Covering indexes:
      CREATE INDEX idx_orders_cover ON orders(user_id) INCLUDE (total, status);
  
  Monitoring queries:
    -- Slow queries
    SELECT query, calls, mean_exec_time, total_exec_time
    FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;
    
    -- Table bloat
    SELECT schemaname, tablename, 
           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
    FROM pg_tables WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
    
    -- Index usage
    SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
    FROM pg_stat_user_indexes ORDER BY idx_scan;
    
    -- Cache hit ratio
    SELECT sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
    FROM pg_statio_user_tables;
` + "```" + ``,
					CodeExamples: `# Database performance and monitoring scripts

# 1. PostgreSQL health check
#!/bin/bash
echo "=== PostgreSQL Health Check ==="

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
PGDATABASE="${PGDATABASE:-postgres}"

run_query() {
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
        -t -A -c "$1" 2>/dev/null
}

# Connection info
echo "--- Connections ---"
ACTIVE=$(run_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
IDLE=$(run_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'idle'")
MAX=$(run_query "SHOW max_connections")
echo "Active: $ACTIVE, Idle: $IDLE, Max: $MAX"

# Replication status
echo ""
echo "--- Replication ---"
REPLICAS=$(run_query "SELECT count(*) FROM pg_stat_replication")
echo "Connected replicas: $REPLICAS"
if [ "$REPLICAS" -gt 0 ]; then
    run_query "SELECT client_addr, state, sent_lsn, write_lsn, replay_lsn, 
        (sent_lsn - replay_lsn) as lag FROM pg_stat_replication"
fi

# Database sizes
echo ""
echo "--- Database Sizes ---"
run_query "SELECT datname, pg_size_pretty(pg_database_size(datname)) as size 
    FROM pg_database WHERE datname NOT IN ('template0','template1') 
    ORDER BY pg_database_size(datname) DESC"

# Cache hit ratio
echo ""
echo "--- Cache Hit Ratio ---"
RATIO=$(run_query "SELECT round(sum(heap_blks_hit)::numeric / 
    nullif(sum(heap_blks_hit) + sum(heap_blks_read), 0) * 100, 2) 
    FROM pg_statio_user_tables")
echo "Cache hit ratio: ${RATIO}%"
if [ -n "$RATIO" ]; then
    THRESHOLD=95
    if (( $(echo "$RATIO < $THRESHOLD" | bc -l 2>/dev/null) )); then
        echo "WARNING: Below ${THRESHOLD}% threshold"
    fi
fi

# Slow queries
echo ""
echo "--- Top Slow Queries ---"
run_query "SELECT substring(query, 1, 80) as query, 
    calls, round(mean_exec_time::numeric, 2) as avg_ms,
    round(total_exec_time::numeric, 2) as total_ms
    FROM pg_stat_statements 
    ORDER BY mean_exec_time DESC LIMIT 5" 2>/dev/null || echo "pg_stat_statements not available"

# Table bloat
echo ""
echo "--- Largest Tables ---"
run_query "SELECT schemaname || '.' || tablename as table_name,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as data_size,
    pg_size_pretty(pg_indexes_size(schemaname || '.' || tablename::regclass)) as index_size
    FROM pg_tables WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC LIMIT 10"

# Unused indexes
echo ""
echo "--- Unused Indexes ---"
run_query "SELECT schemaname || '.' || indexrelname as index,
    pg_size_pretty(pg_relation_size(indexrelid)) as size,
    idx_scan as scans
    FROM pg_stat_user_indexes 
    WHERE idx_scan = 0 AND indexrelname NOT LIKE '%pkey%'
    ORDER BY pg_relation_size(indexrelid) DESC LIMIT 10"

# 2. Connection pool monitor
#!/bin/bash
echo "=== Connection Pool Status ==="

# PgBouncer stats
if command -v psql &>/dev/null; then
    PGBOUNCER_PORT="${PGBOUNCER_PORT:-6432}"
    
    echo "--- Pool Stats ---"
    psql -h localhost -p "$PGBOUNCER_PORT" -U pgbouncer pgbouncer \
        -c "SHOW POOLS" 2>/dev/null || echo "PgBouncer not available"
    
    echo ""
    echo "--- Client Stats ---"
    psql -h localhost -p "$PGBOUNCER_PORT" -U pgbouncer pgbouncer \
        -c "SHOW STATS" 2>/dev/null || echo "PgBouncer not available"
fi`,
				},
			},
		},
	})
}
