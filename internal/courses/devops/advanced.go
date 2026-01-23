package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          270,
			Title:       "Advanced Kubernetes",
			Description: "Advanced Kubernetes: RBAC, NetworkPolicies, Custom Resources, Operators, and advanced patterns.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Kubernetes Security and RBAC",
					Content: `Secure Kubernetes clusters with RBAC and security policies.

**RBAC (Role-Based Access Control):**
- **Role**: Namespace-scoped permissions
- **ClusterRole**: Cluster-scoped permissions
- **RoleBinding**: Bind role to users/groups
- **ServiceAccount**: Pod identity

**Network Policies:**
- Control pod-to-pod communication
- Ingress and egress rules
- Namespace isolation
- Label selectors

**Pod Security:**
- Security contexts
- Pod security policies
- Non-root containers
- Read-only root filesystems

**Secrets Management:**
- Kubernetes secrets
- External secret operators
- Encryption at rest
- Secret rotation

**Best Practices:**
- Least privilege
- Network segmentation
- Regular security audits
- Use service accounts
- Encrypt secrets`,
					CodeExamples: `# ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa
  namespace: default

# Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]

# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
subjects:
- kind: ServiceAccount
  name: app-sa
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

# NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: app-network-policy
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432

# Pod with security context
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: app
    image: myapp:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL`,
				},
				{
					Title: "Kubernetes Operators and Custom Resources",
					Content: `Extend Kubernetes with custom resources and operators.

**Custom Resources:**
- Extend Kubernetes API
- Custom resource definitions (CRD)
- Custom controllers
- Operator pattern

**Operators:**
- Automate complex applications
- Lifecycle management
- Domain-specific knowledge
- Kubernetes-native applications

**Operator Framework:**
- Operator SDK
- Operator Lifecycle Manager
- Operator Registry

**Use Cases:**
- Database operators
- Monitoring operators
- CI/CD operators
- Application operators`,
					CodeExamples: `# Custom Resource Definition
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              databaseName:
                type: string
              replicas:
                type: integer
  scope: Namespaced
  names:
    plural: databases
    singular: database
    kind: Database

# Custom Resource
apiVersion: example.com/v1
kind: Database
metadata:
  name: my-database
spec:
  databaseName: myapp
  replicas: 3

# Operator example (simplified)
# Operator watches for Database resources
# and creates StatefulSet, Service, etc.`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          271,
			Title:       "Service Mesh (Istio, Linkerd)",
			Description: "Service mesh patterns: Istio, Linkerd, traffic management, security, and observability.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Service Mesh Fundamentals",
					Content: `Service mesh provides advanced traffic management and observability.

**What is Service Mesh?**
- Infrastructure layer for microservices
- Handles service-to-service communication
- Sidecar proxy pattern
- Transparent to applications

**Key Features:**
- **Traffic Management**: Load balancing, routing
- **Security**: mTLS, authentication
- **Observability**: Metrics, traces, logs
- **Policy Enforcement**: Rate limiting, access control

**Service Mesh Components:**
- **Control Plane**: Configuration and management
- **Data Plane**: Sidecar proxies
- **Service Discovery**: Automatic discovery
- **Load Balancing**: Intelligent routing

**Popular Solutions:**
- **Istio**: Feature-rich, complex
- **Linkerd**: Lightweight, simple
- **Consul Connect**: HashiCorp solution
- **AWS App Mesh**: AWS-native`,
					CodeExamples: `# Istio VirtualService
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
  - reviews
  http:
  - match:
    - headers:
        end-user:
          exact: jason
    route:
    - destination:
        host: reviews
        subset: v2
  - route:
    - destination:
        host: reviews
        subset: v1

# DestinationRule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN

# Gateway
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: myapp-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - myapp.example.com

# PeerAuthentication (mTLS)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
spec:
  mtls:
    mode: STRICT`,
				},
				{
					Title: "Service Mesh Patterns and Best Practices",
					Content: `Service mesh patterns and best practices for production deployments.

**Service Mesh Patterns:**
- **Traffic Splitting**: A/B testing and canary deployments
- **Circuit Breaker**: Fault tolerance
- **Retry Logic**: Automatic retries with backoff
- **Timeout Management**: Request timeouts
- **Load Balancing**: Intelligent traffic distribution
- **Security**: mTLS and authentication
- **Observability**: Metrics, logs, traces

**Best Practices:**
- Start with observability
- Implement mTLS gradually
- Use traffic policies carefully
- Monitor service mesh overhead
- Test failure scenarios
- Document policies
- Regular reviews

**Common Patterns:**
- **Canary Deployment**: Gradual traffic shift
- **Blue-Green**: Instant switch
- **Circuit Breaker**: Fail fast
- **Retry with Backoff**: Handle transient failures
- **Timeout**: Prevent hanging requests
- **Rate Limiting**: Protect services

**Performance Considerations:**
- Service mesh adds latency
- Monitor overhead
- Optimize configurations
- Use appropriate timeouts
- Balance security and performance

**Common Pitfalls:**
- Over-complicating policies
- Not monitoring overhead
- Ignoring security
- Poor timeout configuration
- Not testing failure scenarios`,
					CodeExamples: `# Traffic splitting (Istio)
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
  - reviews
  http:
  - match:
    - headers:
        end-user:
          exact: jason
    route:
    - destination:
        host: reviews
        subset: v2
      weight: 100
  - route:
    - destination:
        host: reviews
        subset: v1
      weight: 90
    - destination:
        host: reviews
        subset: v2
      weight: 10

# Circuit breaker
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 10
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50

# Retry policy
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: httpbin
spec:
  hosts:
  - httpbin
  http:
  - match:
    - uri:
        prefix: /status
    route:
    - destination:
        host: httpbin
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: 5xx,reset,connect-failure,refused-stream

# Timeout
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: httpbin
spec:
  hosts:
  - httpbin
  http:
  - timeout: 3s
    route:
    - destination:
        host: httpbin

# Rate limiting
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: httpbin
spec:
  host: httpbin
  trafficPolicy:
    loadBalancer:
      consistentHash:
        httpHeaderName: x-user-id
    connectionPool:
      http:
        http1MaxPendingRequests: 10
        maxRequestsPerConnection: 2`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          272,
			Title:       "GitOps (ArgoCD, Flux)",
			Description: "GitOps workflows: ArgoCD, Flux, declarative deployments, and Git-based operations.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "GitOps Principles",
					Content: `GitOps uses Git as the single source of truth for infrastructure and applications.

**GitOps Principles:**
- **Declarative**: Desired state in Git
- **Version Controlled**: All changes in Git
- **Automated**: Automatic synchronization
- **Observable**: Clear state and changes
- **Pull-based**: Agent pulls from Git

**Benefits:**
- Faster deployments
- Better audit trail
- Rollback capabilities
- Collaboration
- Consistency

**GitOps Tools:**
- **ArgoCD**: Kubernetes-native, UI
- **Flux**: CNCF project, GitOps toolkit
- **Jenkins X**: CI/CD + GitOps
- **Tekton**: Cloud-native CI/CD

**Workflow:**
1. Developer pushes code
2. CI builds and tests
3. CI pushes manifests to Git
4. GitOps tool syncs to cluster
5. Cluster matches desired state`,
					CodeExamples: `# ArgoCD Application
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/user/repo
    targetRevision: main
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true

# Flux GitRepository
apiVersion: source.toolkit.fluxcd.io/v1beta1
kind: GitRepository
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/user/repo
  ref:
    branch: main
  secretRef:
    name: git-credentials

# Flux Kustomization
apiVersion: kustomize.toolkit.fluxcd.io/v1beta1
kind: Kustomization
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 5m
  path: ./k8s/overlays/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: myapp
  validation: client

# Directory structure
repo/
  k8s/
    base/
      deployment.yaml
      service.yaml
    overlays/
      production/
        kustomization.yaml
      staging/
        kustomization.yaml`,
				},
				{
					Title: "Advanced GitOps Workflows",
					Content: `Advanced GitOps patterns and workflows for complex deployments.

**Advanced GitOps Patterns:**
- **Multi-Environment**: Dev, staging, production
- **Progressive Delivery**: Gradual rollout
- **Feature Flags**: Toggle features via Git
- **Multi-Cluster**: Deploy to multiple clusters
- **App of Apps**: Manage multiple applications
- **Helm Integration**: Use Helm charts
- **Kustomize**: Environment-specific overlays

**Workflow Patterns:**
- **Push-based**: CI pushes to Git, GitOps syncs
- **Pull-based**: GitOps polls Git (recommended)
- **Hybrid**: Combination of both
- **Branching Strategy**: GitFlow, GitHub Flow
- **Promotion**: Promote between environments

**Best Practices:**
- Use separate repos for apps and infrastructure
- Implement approval workflows
- Use Kustomize or Helm for overlays
- Monitor sync status
- Implement rollback procedures
- Use App of Apps pattern
- Document workflows

**Common Pitfalls:**
- Not using separate repos
- Manual cluster changes
- Not monitoring sync
- Poor branching strategy
- Not testing in dev first`,
					CodeExamples: `# ArgoCD App of Apps pattern
# Root application
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: root-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/user/apps-repo
    targetRevision: main
    path: apps
  destination:
    server: https://kubernetes.default.svc
  syncPolicy:
    automated:
      prune: true
      selfHeal: true

# Individual app (apps/myapp.yaml)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
spec:
  project: default
  source:
    repoURL: https://github.com/user/app-repo
    targetRevision: main
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production

# Multi-cluster deployment
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-us-east
spec:
  destination:
    server: https://us-east-cluster.example.com
    namespace: production

---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-eu-west
spec:
  destination:
    server: https://eu-west-cluster.example.com
    namespace: production

# Flux with Helm
apiVersion: source.toolkit.fluxcd.io/v1beta1
kind: HelmRepository
metadata:
  name: myrepo
  namespace: flux-system
spec:
  interval: 1h
  url: https://charts.example.com

apiVersion: helm.toolkit.fluxcd.io/v2beta1
kind: HelmRelease
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 5m
  chart:
    spec:
      chart: myapp
      sourceRef:
        kind: HelmRepository
        name: myrepo
      interval: 1h
  values:
    replicaCount: 3
    image:
      tag: v1.0.0

# Progressive delivery with Flagger
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: myapp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  service:
    port: 80
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
    - name: request-duration
      thresholdRange:
        max: 500
    webhooks:
    - name: load-test
      url: http://flagger-loadtester.test/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://myapp-canary.test/"

# GitOps promotion workflow
# 1. Deploy to dev (automatic)
# 2. Test in dev
# 3. Promote to staging (manual approval)
# 4. Test in staging
# 5. Promote to production (manual approval)

# Promotion script
#!/bin/bash
ENV=$1
VERSION=$2

# Update kustomize overlay
cd k8s/overlays/$ENV
kustomize edit set image myapp=myapp:$VERSION

# Commit and push
git add .
git commit -m "Promote myapp $VERSION to $ENV"
git push origin main

# ArgoCD will sync automatically`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          273,
			Title:       "Cloud-Native Security",
			Description: "Security in cloud-native environments: scanning, compliance, secrets management, and security best practices.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Container and Image Security",
					Content: `Secure containerized applications and images.

**Image Security:**
- Scan images for vulnerabilities
- Use minimal base images
- Keep images updated
- Sign images
- Use trusted registries

**Container Security:**
- Run as non-root
- Read-only filesystems
- Drop capabilities
- Network policies
- Resource limits

**Runtime Security:**
- Runtime scanning
- Behavioral analysis
- Anomaly detection
- Policy enforcement

**Tools:**
- **Trivy**: Vulnerability scanner
- **Falco**: Runtime security
- **OPA**: Policy engine
- **Aqua Security**: Container security platform`,
					CodeExamples: `# Scan image with Trivy
trivy image myapp:latest

# Scan in CI/CD
- name: Scan image
  run: |
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
      aquasec/trivy image myapp:latest

# OPA Gatekeeper policy
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
      validation:
        openAPIV3Schema:
          properties:
            labels:
              type: array
              items:
                type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8srequiredlabels
        violation[{"msg": msg}] {
          required := input.parameters.labels
          provided := input.review.object.metadata.labels
          missing := required[_]
          not provided[missing]
          msg := sprintf("Missing required label: %v", [missing])
        }

# Enforce policy
apiVersion: config.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: must-have-labels
spec:
  match:
    kinds:
      - apiGroups: ["apps"]
        kinds: ["Deployment"]
  parameters:
    labels: ["app", "version"]`,
				},
				{
					Title: "Secrets Management and Compliance",
					Content: `Manage secrets securely and ensure compliance.

**Secrets Management:**
- External secret operators
- Vault integration
- Secret rotation
- Encryption at rest and in transit

**Compliance:**
- CIS benchmarks
- Security policies
- Audit logging
- Compliance scanning

**Best Practices:**
- Never commit secrets
- Use secret management tools
- Rotate secrets regularly
- Encrypt secrets
- Audit secret access`,
					CodeExamples: `# External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets"

apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: db-secret
  data:
  - secretKey: password
    remoteRef:
      key: database/credentials
      property: password`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          274,
			Title:       "Disaster Recovery and Backup",
			Description: "Disaster recovery strategies: backups, replication, failover, and business continuity.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Backup Strategies",
					Content: `Comprehensive backup strategies for cloud-native applications.

**Backup Types:**
- **Full Backup**: Complete system backup
- **Incremental**: Changes since last backup
- **Differential**: Changes since full backup
- **Snapshot**: Point-in-time copy

**What to Backup:**
- Application data
- Configuration
- Secrets (encrypted)
- Database dumps
- Infrastructure state

**Backup Tools:**
- **Velero**: Kubernetes backup
- **Kasten K10**: Data management
- **Restic**: Backup tool
- **Cloud-native**: Cloud provider backups

**RTO and RPO:**
- **RTO**: Recovery Time Objective
- **RPO**: Recovery Point Objective

**Best Practices:**
- Regular automated backups
- Test restore procedures
- Off-site backups
- Encrypted backups
- Versioned backups`,
					CodeExamples: `# Velero backup
apiVersion: velero.io/v1
kind: Backup
metadata:
  name: app-backup
  namespace: velero
spec:
  includedNamespaces:
  - production
  includedResources:
  - '*'
  excludedResources:
  - events
  - events.events.k8s.io
  storageLocation: default
  ttl: 720h0m0s

# Velero schedule
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"
  template:
    includedNamespaces:
    - production
    storageLocation: default
    ttl: 720h0m0s

# Restore from backup
apiVersion: velero.io/v1
kind: Restore
metadata:
  name: app-restore
  namespace: velero
spec:
  backupName: app-backup
  includedNamespaces:
  - production

# Database backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="myapp"

# Backup database
pg_dump -h db-host -U postgres $DB_NAME | gzip > $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz s3://backups/database/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete`,
				},
				{
					Title: "Disaster Recovery Planning",
					Content: `Plan and implement disaster recovery procedures.

**DR Strategies:**
- **Backup and Restore**: Traditional approach
- **Pilot Light**: Minimal infrastructure ready
- **Warm Standby**: Scaled-down environment
- **Hot Standby**: Fully operational duplicate
- **Multi-Site**: Active-active across sites

**Failover Procedures:**
- Automated failover
- Manual failover
- Failback procedures
- Testing procedures

**Best Practices:**
- Document procedures
- Regular DR drills
- Monitor backup health
- Test restore procedures
- Define RTO/RPO`,
					CodeExamples: `# Multi-region deployment
# Primary region
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  region: "us-east-1"
  replica-region: "us-west-2"

# Failover script
#!/bin/bash
PRIMARY_REGION="us-east-1"
SECONDARY_REGION="us-west-2"

# Check primary health
if ! check_health $PRIMARY_REGION; then
    echo "Primary region unhealthy, failing over..."
    
    # Update DNS to point to secondary
    update_dns_route53 $SECONDARY_REGION
    
    # Scale up secondary
    kubectl --context=$SECONDARY_REGION scale deployment app --replicas=10
    
    # Notify team
    send_alert "Failover to $SECONDARY_REGION initiated"
fi`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          275,
			Title:       "Multi-Cloud Strategies",
			Description: "Multi-cloud architectures: vendor lock-in avoidance, hybrid cloud, and cloud-agnostic patterns.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Multi-Cloud Architecture",
					Content: `Design applications for multiple cloud providers.

**Multi-Cloud Benefits:**
- Vendor lock-in avoidance
- Best-of-breed services
- Disaster recovery
- Cost optimization
- Compliance requirements

**Challenges:**
- Complexity
- Cost management
- Skills requirements
- Data synchronization
- Network latency

**Architecture Patterns:**
- **Cloud-Agnostic**: Use common services
- **Cloud-Specific**: Leverage unique features
- **Hybrid**: On-prem + cloud
- **Federated**: Multiple clouds

**Abstraction Layers:**
- Kubernetes: Common orchestration
- Terraform: Multi-cloud IaC
- Service mesh: Unified networking
- Monitoring: Unified observability`,
					CodeExamples: `# Terraform multi-cloud
# AWS
provider "aws" {
    region = "us-east-1"
}

resource "aws_instance" "web" {
    # AWS-specific
}

# Azure
provider "azurerm" {
    features {}
}

resource "azurerm_linux_virtual_machine" "web" {
    # Azure-specific
}

# Cloud-agnostic Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        # Works on any Kubernetes cluster

# Multi-cloud monitoring
# Use Prometheus with federation
# or cloud-agnostic APM tools`,
				},
				{
					Title: "Multi-Cloud Implementation",
					Content: `Implementing multi-cloud architectures in practice.

**Implementation Strategies:**
- **Cloud-Agnostic Services**: Use Kubernetes, containers
- **Abstraction Layers**: Terraform, service mesh
- **Data Replication**: Sync data across clouds
- **Traffic Routing**: Route to appropriate cloud
- **Disaster Recovery**: Failover between clouds
- **Cost Management**: Unified cost tracking

**Implementation Steps:**
1. Choose abstraction layer (Kubernetes, Terraform)
2. Design cloud-agnostic architecture
3. Implement data synchronization
4. Set up monitoring across clouds
5. Implement failover procedures
6. Test disaster recovery

**Best Practices:**
- Use Kubernetes for portability
- Use Terraform for multi-cloud IaC
- Implement unified monitoring
- Test failover regularly
- Monitor costs across clouds
- Document cloud-specific differences

**Common Pitfalls:**
- Vendor lock-in despite multi-cloud
- Poor data synchronization
- Inconsistent monitoring
- Not testing failover
- Cost overruns`,
					CodeExamples: `# Multi-cloud Terraform module
# modules/app/main.tf
variable "cloud_provider" {
  type = string
}

resource "kubernetes_deployment" "app" {
  metadata {
    name = "app"
  }
  spec {
    replicas = var.replicas
    template {
      spec {
        container {
          image = var.image
        }
      }
    }
  }
}

# AWS implementation
module "app_aws" {
  source = "./modules/app"
  cloud_provider = "aws"
  replicas = 3
  image = "myapp:latest"
}

# Azure implementation
module "app_azure" {
  source = "./modules/app"
  cloud_provider = "azure"
  replicas = 3
  image = "myapp:latest"
}

# Multi-cloud monitoring (Prometheus federation)
# prometheus-aws.yml
global:
  external_labels:
    cluster: 'aws-us-east-1'

scrape_configs:
  - job_name: 'federate-azure'
    scrape_interval: 15s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job=~".+"}'
    static_configs:
      - targets:
        - 'prometheus-azure.example.com:9090'

# Multi-cloud failover script
#!/bin/bash
PRIMARY_CLOUD="aws"
SECONDARY_CLOUD="azure"

check_health() {
    kubectl --context=$1 get nodes
    return $?
}

if ! check_health $PRIMARY_CLOUD; then
    echo "Primary cloud unhealthy, failing over..."
    # Update DNS/load balancer to point to secondary
    # Scale up secondary
    kubectl --context=$SECONDARY_CLOUD scale deployment app --replicas=10
fi`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          276,
			Title:       "Infrastructure Monitoring",
			Description: "Infrastructure monitoring: Prometheus, Grafana, alerting, and infrastructure observability.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Infrastructure Monitoring Stack",
					Content: `Monitor infrastructure health and performance.

**Monitoring Stack:**
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Alertmanager**: Alert routing
- **Node Exporter**: Node metrics
- **cAdvisor**: Container metrics

**Key Metrics:**
- **CPU**: Usage, load average
- **Memory**: Usage, swap
- **Disk**: Usage, I/O
- **Network**: Bandwidth, errors
- **Application**: Custom metrics

**Alerting:**
- Alert rules
- Alert routing
- Notification channels
- Alert grouping
- Silence rules

**Best Practices:**
- Monitor everything
- Set meaningful thresholds
- Use dashboards
- Test alerts
- Document runbooks`,
					CodeExamples: `# Prometheus configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
  
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

# Alert rule
groups:
  - name: infrastructure
    rules:
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
      
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disk space low on {{ $labels.instance }}"

# Grafana dashboard (JSON snippet)
{
  "dashboard": {
    "title": "Infrastructure Overview",
    "panels": [
      {
        "title": "CPU Usage",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
          }
        ]
      }
    ]
  }
}`,
				},
				{
					Title: "Advanced Monitoring Patterns",
					Content: `Advanced monitoring patterns and observability practices.

**Advanced Patterns:**
- **Service Dependency Mapping**: Visualize service dependencies
- **Distributed Tracing**: Track requests across services
- **Synthetic Monitoring**: Proactive health checks
- **Real User Monitoring**: Track actual user experience
- **Anomaly Detection**: Identify unusual patterns
- **Predictive Monitoring**: Forecast issues

**Observability Pillars:**
- **Metrics**: Numerical measurements
- **Logs**: Event records
- **Traces**: Request journeys
- **Profiles**: Resource usage details

**Monitoring Best Practices:**
- Implement the three pillars
- Use correlation IDs
- Monitor business metrics
- Set up alert fatigue prevention
- Use SLOs for alerting
- Implement runbooks
- Regular reviews

**Common Pitfalls:**
- Too many alerts
- Not using SLOs
- Missing correlation
- Ignoring business metrics
- Poor runbooks`,
					CodeExamples: `# Service dependency mapping (Jaeger)
# Track request across services
from opentelemetry import trace
from opentelemetry.instrumentation.requests import RequestsInstrumentor

tracer = trace.get_tracer(__name__)
RequestsInstrumentor().instrument()

def handle_request():
    with tracer.start_as_current_span("handle_request") as span:
        span.set_attribute("user.id", user_id)
        # Call service A
        with tracer.start_as_current_span("call_service_a"):
            service_a_response = call_service_a()
        # Call service B
        with tracer.start_as_current_span("call_service_b"):
            service_b_response = call_service_b()

# Synthetic monitoring (Pingdom/Site24x7)
# Monitor critical endpoints
@app.route('/health')
def health():
    # Check dependencies
    db_ok = check_database()
    cache_ok = check_cache()
    return {'status': 'ok' if db_ok and cache_ok else 'degraded'}

# Anomaly detection (Prometheus)
# Alert on unusual patterns
groups:
  - name: anomalies
    rules:
      - alert: UnusualTrafficPattern
        expr: |
          abs(
            rate(http_requests_total[5m]) - 
            avg_over_time(rate(http_requests_total[1h])[1h:5m])
          ) > avg_over_time(rate(http_requests_total[1h])[1h:5m]) * 0.5
        for: 10m

# SLO-based alerting
# Alert when SLO is at risk
groups:
  - name: slo_alerts
    rules:
      - alert: SLORisk
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[5m])) /
            sum(rate(http_requests_total[5m]))
          ) < 0.99
        for: 5m
        annotations:
          summary: "Availability SLO at risk"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          277,
			Title:       "Cost Optimization",
			Description: "Optimize cloud costs: resource rightsizing, reserved instances, spot instances, and cost management.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Cloud Cost Optimization",
					Content: `Optimize cloud infrastructure costs effectively.

**Cost Optimization Strategies:**
- **Rightsizing**: Match resources to needs
- **Reserved Instances**: Commit for discounts
- **Spot Instances**: Use spare capacity
- **Auto-scaling**: Scale based on demand
- **Resource Scheduling**: Stop unused resources

**Cost Monitoring:**
- Cost allocation tags
- Cost dashboards
- Budget alerts
- Cost reports
- Cost anomaly detection

**Optimization Areas:**
- Compute: Right-size instances
- Storage: Use appropriate tiers
- Networking: Optimize data transfer
- Databases: Choose right size
- Reserved capacity: Commit for discounts

**Tools:**
- **AWS Cost Explorer**: Cost analysis
- **Azure Cost Management**: Cost tracking
- **GCP Billing**: Cost monitoring
- **Kubecost**: Kubernetes cost analysis`,
					CodeExamples: `# Kubernetes resource requests/limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    spec:
      containers:
      - name: app
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"

# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: app-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  updatePolicy:
    updateMode: "Auto"

# Spot instance configuration (AWS)
apiVersion: v1
kind: Pod
metadata:
  name: spot-pod
spec:
  nodeSelector:
    node.kubernetes.io/instance-type: spot
  containers:
  - name: app
    image: myapp:latest

# Cost allocation tags (AWS)
# Tag resources for cost tracking
tags:
  Environment: production
  Team: backend
  Project: myapp
  CostCenter: engineering`,
				},
				{
					Title: "Cost Optimization Tools and Techniques",
					Content: `Tools and techniques for effective cost optimization.

**Cost Optimization Tools:**
- **Kubecost**: Kubernetes cost analysis
- **CloudHealth**: Multi-cloud cost management
- **Spot.io**: Spot instance optimization
- **ParkMyCloud**: Resource scheduling
- **CloudCheckr**: Cost optimization insights
- **AWS Cost Explorer**: AWS cost analysis
- **Azure Cost Management**: Azure cost tracking

**Optimization Techniques:**
- **Right-sizing**: Match resources to actual needs
- **Reserved Instances**: Commit for discounts (30-70% savings)
- **Spot Instances**: Use spare capacity (up to 90% savings)
- **Auto-scaling**: Scale based on demand
- **Resource Scheduling**: Stop unused resources
- **Storage Optimization**: Use appropriate tiers
- **Data Transfer Optimization**: Minimize cross-region transfers

**Cost Monitoring:**
- Set up budget alerts
- Track costs by team/project
- Monitor cost trends
- Identify cost anomalies
- Regular cost reviews

**Best Practices:**
- Tag all resources
- Set up budget alerts
- Regular cost reviews
- Right-size continuously
- Use spot instances for non-critical workloads
- Schedule non-production resources
- Optimize storage tiers

**Common Pitfalls:**
- Not tagging resources
- Ignoring cost alerts
- Over-provisioning
- Not using reserved instances
- Ignoring storage costs`,
					CodeExamples: `# Kubecost cost allocation
# Label resources for cost tracking
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app: myapp
    team: backend
    cost-center: engineering
spec:
  template:
    metadata:
      labels:
        app: myapp
        team: backend
        cost-center: engineering

# AWS Cost Explorer query
# Get costs by tag
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=TAG,Key=Team

# Budget alert (AWS)
aws budgets create-budget \
  --account-id 123456789012 \
  --budget file://budget.json

# budget.json
{
  "BudgetName": "MonthlyBudget",
  "BudgetLimit": {
    "Amount": "1000",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}

# Resource scheduling script
#!/bin/bash
# Stop non-production resources during off-hours

ENV=$1
CURRENT_HOUR=$(date +%H)

if [ "$ENV" != "production" ] && [ $CURRENT_HOUR -ge 20 ] || [ $CURRENT_HOUR -lt 8 ]; then
    echo "Stopping $ENV resources"
    kubectl scale deployment --all --replicas=0 -n $ENV
else
    echo "Starting $ENV resources"
    kubectl scale deployment --all --replicas=1 -n $ENV
fi

# Cost optimization report script
#!/bin/bash
echo "Cost Optimization Report"
echo "========================"
echo ""
echo "Top 10 most expensive resources:"
kubectl top pods --all-namespaces --sort-by=memory | head -10
echo ""
echo "Underutilized resources:"
# Find pods with low CPU/memory usage
kubectl top pods --all-namespaces | awk '$3 < 10 || $4 < 100'`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          278,
			Title:       "DevOps Culture and Practices",
			Description: "DevOps culture: team practices, collaboration, blameless postmortems, and continuous improvement.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "DevOps Culture and Collaboration",
					Content: `Build a strong DevOps culture for team success.

**Cultural Principles:**
- **Shared Responsibility**: Everyone owns the system
- **Fail Fast**: Learn from failures quickly
- **Continuous Learning**: Always improving
- **Blameless Culture**: Focus on systems
- **Automation First**: Automate everything

**Team Practices:**
- **Daily Standups**: Quick sync
- **Retrospectives**: Regular improvement
- **Pair Programming**: Knowledge sharing
- **Code Reviews**: Quality and learning
- **Documentation**: Knowledge preservation

**Communication:**
- Transparent communication
- Shared knowledge
- Cross-functional teams
- Regular feedback
- Open discussions

**Metrics:**
- Deployment frequency
- Lead time
- Mean time to recovery
- Change failure rate`,
					CodeExamples: `# Postmortem template
# Incident: [Title]
# Date: [Date]
# Duration: [Duration]
# Impact: [Impact]

## Timeline
- [Time] - Event occurred
- [Time] - Detection
- [Time] - Response started
- [Time] - Resolution

## Root Cause
[Analysis of root cause]

## Impact
- Users affected: [Number]
- Services affected: [List]
- Data loss: [Details]

## Actions Taken
1. [Action]
2. [Action]

## Prevention
1. [Preventive measure]
2. [Preventive measure]

# Runbook template
## Service: [Service Name]
## Owner: [Team]

### Health Check
- Endpoint: /health
- Expected: 200 OK
- Check interval: 30s

### Common Issues
1. **High latency**
   - Check: CPU usage, database queries
   - Fix: Scale up, optimize queries

2. **Service down**
   - Check: Pod status, logs
   - Fix: Restart pods, check dependencies`,
				},
				{
					Title: "Implementing DevOps Culture",
					Content: `Practical steps to implement DevOps culture in organizations.

**Implementation Steps:**
1. **Assess Current State**: Understand existing culture
2. **Define Vision**: Set clear DevOps goals
3. **Start Small**: Begin with pilot projects
4. **Train Teams**: Provide DevOps training
5. **Implement Tools**: Introduce automation tools
6. **Measure Progress**: Track metrics
7. **Iterate**: Continuous improvement

**Change Management:**
- **Leadership Support**: Get executive buy-in
- **Communication**: Clear messaging
- **Training**: Invest in skills
- **Incentives**: Reward DevOps behaviors
- **Remove Barriers**: Eliminate obstacles

**Key Practices to Implement:**
- **Infrastructure as Code**: Version control everything
- **CI/CD**: Automate deployments
- **Monitoring**: Comprehensive observability
- **Incident Response**: Blameless postmortems
- **Documentation**: Knowledge sharing
- **Automation**: Reduce manual work

**Measuring Success:**
- Deployment frequency
- Lead time for changes
- Mean time to recovery (MTTR)
- Change failure rate
- Team satisfaction

**Common Challenges:**
- Resistance to change
- Siloed teams
- Lack of skills
- Tool complexity
- Legacy systems

**Best Practices:**
- Start with culture, not tools
- Get leadership support
- Celebrate small wins
- Share success stories
- Learn from failures
- Continuous improvement`,
					CodeExamples: `# DevOps maturity assessment
# Rate each area 1-5
Areas:
  - Version Control: 5
  - CI/CD: 4
  - Infrastructure as Code: 3
  - Monitoring: 4
  - Automation: 3
  - Collaboration: 4

# DevOps metrics dashboard
Metrics:
  - Deployment Frequency: Daily
  - Lead Time: 2 hours
  - MTTR: 15 minutes
  - Change Failure Rate: 2%

# Team retrospective template
## What went well?
- Automated deployments
- Improved monitoring

## What could be better?
- Faster feedback loops
- Better documentation

## Action items
1. Implement feature flags
2. Improve runbooks

# DevOps training plan
Week 1: Git and version control
Week 2: CI/CD basics
Week 3: Infrastructure as Code
Week 4: Monitoring and observability
Week 5: Containerization
Week 6: Kubernetes basics

# Culture change checklist
- [ ] Leadership buy-in
- [ ] Team training
- [ ] Tool selection
- [ ] Pilot project
- [ ] Metrics definition
- [ ] Communication plan
- [ ] Success criteria

# DevOps charter example
## Mission
Enable teams to deliver value faster and more reliably

## Principles
1. Automation first
2. Shared responsibility
3. Continuous improvement
4. Blameless culture

## Goals
- Reduce deployment time by 50%
- Increase deployment frequency to daily
- Reduce MTTR to under 30 minutes`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          279,
			Title:       "Site Reliability Engineering (SRE)",
			Description: "SRE principles: SLIs, SLOs, error budgets, toil reduction, and reliability engineering.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "SRE Fundamentals",
					Content: `Site Reliability Engineering balances reliability and feature velocity.

**SRE Principles:**
- **SLI**: Service Level Indicator (metric)
- **SLO**: Service Level Objective (target)
- **SLA**: Service Level Agreement (contract)
- **Error Budget**: Acceptable unreliability

**Key Concepts:**
- **Toil**: Manual, repetitive work
- **Automation**: Reduce toil
- **Monitoring**: Measure everything
- **Incident Response**: Fast recovery
- **Postmortems**: Learn from incidents

**Error Budget:**
- 100% - SLO = Error Budget
- Example: 99.9% SLO = 0.1% error budget
- Use budget for releases
- Exceed budget = stop releases

**Best Practices:**
- Define clear SLIs/SLOs
- Monitor error budgets
- Automate toil
- Blameless postmortems
- Continuous improvement`,
					CodeExamples: `# SLI Examples
# Availability SLI
availability = (successful_requests / total_requests) * 100

# Latency SLI
latency_p95 = 95th_percentile(request_duration)

# Error Rate SLI
error_rate = (error_requests / total_requests) * 100

# SLO Definition
# 99.9% availability over 30 days
# = 43.2 minutes of downtime allowed per month

# Error Budget Calculation
error_budget = 100% - SLO
# Example: 99.9% SLO = 0.1% error budget

# Prometheus SLO monitoring
# Availability SLO
availability_sli = (
    sum(rate(http_requests_total{status!~"5.."}[5m])) /
    sum(rate(http_requests_total[5m]))
) * 100

# Alert when below SLO
- alert: SLOViolation
  expr: availability_sli < 99.9
  for: 5m
  annotations:
    summary: "Availability below SLO"

# Error Budget Burn Rate
burn_rate = error_rate / error_budget
# Alert if burning too fast
- alert: ErrorBudgetBurn
  expr: burn_rate > 2
  for: 5m`,
				},
				{
					Title: "SRE Practices and Automation",
					Content: `Implement SRE practices and reduce toil through automation.

**Toil Reduction:**
- Identify repetitive tasks
- Automate manual work
- Self-service tools
- Documentation

**Incident Management:**
- On-call rotation
- Escalation procedures
- Incident response
- Postmortems

**Reliability Practices:**
- Chaos engineering
- Load testing
- Capacity planning
- Disaster recovery

**Automation:**
- Infrastructure automation
- Deployment automation
- Monitoring automation
- Incident response automation`,
					CodeExamples: `# On-call rotation (PagerDuty example)
# Schedule rotation weekly
# Escalate after 15 minutes

# Incident response runbook
1. Acknowledge alert
2. Assess severity
3. Check dashboards
4. Execute runbook steps
5. Escalate if needed
6. Document actions

# Chaos engineering (Chaos Mesh)
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: pod-failure
spec:
  action: pod-failure
  mode: one
  selector:
    namespaces:
      - production
    labelSelectors:
      app: myapp
  duration: "5m"

# Capacity planning
# Monitor resource usage
# Plan for growth
# Right-size resources
# Auto-scale appropriately`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
