package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          70,
			Title:       "RBAC & Security",
			Description: "Implement role-based access control, authentication, and security best practices.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Authentication and Authorization",
					Content: `**Authentication** verifies who you are. **Authorization** determines what you can do.

**Authentication Methods:**
- **Service Accounts**: For pods and services
- **User Accounts**: For humans (managed outside Kubernetes)
- **X.509 Certificates**: Client certificate authentication
- **Bearer Tokens**: Static tokens, bootstrap tokens
- **OpenID Connect**: OIDC integration

**Authorization Modes:**
- **RBAC**: Role-Based Access Control (recommended)
- **ABAC**: Attribute-Based Access Control
- **Node**: Authorizes kubelet requests
- **Webhook**: External authorization service

**RBAC Components:**
- **Role**: Namespace-scoped permissions
- **RoleBinding**: Binds Role to subjects
- **ClusterRole**: Cluster-scoped permissions
- **ClusterRoleBinding**: Binds ClusterRole to subjects

**Subjects:**
- **Users**: Human users
- **Groups**: User groups
- **ServiceAccounts**: Pod service accounts

**Verbs:**
- **get, list, watch**: Read operations
- **create, update, patch**: Write operations
- **delete**: Delete operations
- **use**: For PodSecurityPolicy (deprecated)`,
					CodeExamples: `# Service Account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa
  namespace: production

# Role (namespace-scoped)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]

# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
- kind: ServiceAccount
  name: app-sa
  namespace: production
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

# ClusterRole (cluster-scoped)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-admin
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]

# ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-binding
subjects:
- kind: User
  name: admin
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: cluster-admin
  apiGroup: rbac.authorization.k8s.io

# View RBAC resources
kubectl get roles,rolebindings
kubectl get clusterroles,clusterrolebindings

# Check permissions
kubectl auth can-i create pods
kubectl auth can-i delete pods --as=system:serviceaccount:default:app-sa`,
				},
				{
					Title: "Service Accounts",
					Content: `**Service Accounts** provide identity for pods. They enable RBAC and access control for applications.

**Default Service Account:**
- Every namespace has default service account
- Pods use default if not specified
- Limited permissions by default

**Service Account Features:**
- **Secrets**: Automatically mounted in pods
- **Image Pull Secrets**: For private registries
- **RBAC**: Can be bound to Roles/ClusterRoles
- **Token**: Automatically created and mounted

**Service Account Tokens:**
- Mounted at /var/run/secrets/kubernetes.io/serviceaccount
- Used for API authentication
- Automatically rotated
- Bound to service account

**Use Cases:**
- Pod authentication to API server
- Access to secrets
- RBAC for pod operations
- Private registry access`,
					CodeExamples: `# Create Service Account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa
  namespace: production

# Service Account with image pull secret
apiVersion: v1
kind: ServiceAccount
metadata:
  name: registry-sa
  namespace: production
imagePullSecrets:
- name: registry-secret

# Pod using Service Account
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
  namespace: production
spec:
  serviceAccountName: app-sa
  containers:
  - name: app
    image: myapp:1.0

# View Service Accounts
kubectl get serviceaccounts
kubectl describe serviceaccount app-sa

# View Service Account token
kubectl get secret -n production | grep app-sa`,
				},
				{
					Title: "Security Contexts",
					Content: `**Security Contexts** define privilege and access control settings for pods and containers.

**Pod Security Context:**
- Applies to all containers in pod
- User/group IDs
- SELinux options
- AppArmor profiles
- Seccomp profiles
- FSGroup

**Container Security Context:**
- Applies to specific container
- Overrides pod-level settings
- Run as user/group
- Capabilities
- Read-only root filesystem
- Allow privilege escalation

**Security Best Practices:**
- Run as non-root user
- Use read-only root filesystem
- Drop unnecessary capabilities
- Use Pod Security Standards
- Enable seccomp/AppArmor

**Pod Security Standards:**
- **Privileged**: Unrestricted (not recommended)
- **Baseline**: Minimally restrictive
- **Restricted**: Highly restrictive (recommended)`,
					CodeExamples: `# Pod with security context
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: nginx:1.21
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE
    volumeMounts:
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: tmp
    emptyDir: {}

# Pod Security Policy (deprecated, use Pod Security Standards)
# Pod Security Standard via namespace
apiVersion: v1
kind: Namespace
metadata:
  name: secure-ns
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          71,
			Title:       "Monitoring & Observability",
			Description: "Set up monitoring, logging, and observability for Kubernetes clusters.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Metrics and Prometheus",
					Content: `**Metrics** provide quantitative data about cluster and application performance.

**Metric Types:**
- **Counter**: Monotonically increasing value
- **Gauge**: Value that can go up or down
- **Histogram**: Distribution of values
- **Summary**: Similar to histogram

**Prometheus:**
- Popular metrics collection system
- Pull-based metrics scraping
- Time-series database
- PromQL query language
- Alertmanager for alerts

**Kubernetes Metrics:**
- **Node Metrics**: CPU, memory, disk, network
- **Pod Metrics**: CPU, memory per pod
- **Container Metrics**: Per-container metrics
- **Custom Metrics**: Application-specific metrics

**Metrics Sources:**
- **cAdvisor**: Container metrics
- **kubelet**: Node and pod metrics
- **kube-state-metrics**: Kubernetes object metrics
- **Application**: Custom application metrics

**Prometheus Operator:**
- Manages Prometheus instances
- ServiceMonitor for service discovery
- PrometheusRule for alerting rules
- Simplifies Prometheus deployment`,
					CodeExamples: `# Install Prometheus Operator
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Prometheus instance
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: prometheus
spec:
  serviceAccountName: prometheus
  serviceMonitorSelector:
    matchLabels:
      team: frontend
  resources:
    requests:
      memory: 2Gi

# ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: app-metrics
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  - port: metrics
    interval: 30s

# PrometheusRule for alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: app-alerts
spec:
  groups:
  - name: app.rules
    rules:
    - alert: HighMemoryUsage
      expr: container_memory_usage_bytes > 1000000000
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage detected"

# View metrics
kubectl top nodes
kubectl top pods`,
				},
				{
					Title: "Logging",
					Content: `**Logging** captures events and messages from applications and system components.

**Logging Architecture:**
- **Application Logs**: Written to stdout/stderr
- **Node Logs**: System and kubelet logs
- **Audit Logs**: API server audit trail
- **Event Logs**: Kubernetes events

**Log Collection:**
- **Fluentd**: Popular log aggregator
- **Fluent Bit**: Lightweight log processor
- **Filebeat**: Elastic log shipper
- **Logstash**: Log processing pipeline

**Log Storage:**
- **Elasticsearch**: Search and analytics
- **Loki**: Log aggregation (Prometheus-like)
- **Cloud Storage**: S3, GCS, Azure Blob
- **Local Storage**: For small clusters

**EFK Stack:**
- **Elasticsearch**: Log storage and search
- **Fluentd/Fluent Bit**: Log collection
- **Kibana**: Log visualization

**Loki Stack:**
- **Loki**: Log aggregation
- **Promtail**: Log collection agent
- **Grafana**: Visualization`,
					CodeExamples: `# Fluentd DaemonSet for log collection
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: elasticsearch.logging.svc.cluster.local
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers

# View pod logs
kubectl logs <pod-name>
kubectl logs <pod-name> -c <container-name>
kubectl logs <pod-name> --previous
kubectl logs -f <pod-name>  # Follow logs

# View logs from all pods with label
kubectl logs -l app=myapp

# View events
kubectl get events --sort-by='.lastTimestamp'`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          72,
			Title:       "Helm & Package Management",
			Description: "Use Helm to package, deploy, and manage Kubernetes applications.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Helm Charts",
					Content: `**Helm** is the package manager for Kubernetes. It simplifies application deployment and management.

**Key Concepts:**
- **Chart**: Package of Kubernetes resources
- **Release**: Instance of deployed chart
- **Repository**: Collection of charts
- **Values**: Configuration for charts

**Chart Structure:**
- **Chart.yaml**: Chart metadata
- **values.yaml**: Default configuration
- **templates/**: Kubernetes manifest templates
- **templates/_helpers.tpl**: Template helpers

**Helm Commands:**
- **install**: Install a chart
- **upgrade**: Upgrade a release
- **uninstall**: Remove a release
- **list**: List releases
- **status**: Show release status

**Chart Repositories:**
- **Artifact Hub**: Public chart repository
- **GitHub**: Charts in Git repositories
- **Local**: Charts in local directory
- **Private**: Private repositories`,
					CodeExamples: `# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Add repository
helm repo add stable https://charts.helm.sh/stable
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Search charts
helm search repo nginx
helm search repo mysql

# Install chart
helm install my-nginx bitnami/nginx

# Install with values
helm install my-nginx bitnami/nginx -f values.yaml

# Install with custom values
helm install my-nginx bitnami/nginx \
  --set service.type=LoadBalancer \
  --set replicaCount=3

# List releases
helm list
helm list --all-namespaces

# Upgrade release
helm upgrade my-nginx bitnami/nginx --set replicaCount=5

# Rollback release
helm rollback my-nginx

# Uninstall release
helm uninstall my-nginx

# Chart structure:
# mychart/
#   Chart.yaml
#   values.yaml
#   templates/
#     deployment.yaml
#     service.yaml
#     _helpers.tpl`,
				},
				{
					Title: "Chart Development",
					Content: `**Creating Charts** allows you to package your applications for easy deployment.

**Chart Template Syntax:**
- **{{ .Values.key }}**: Access values
- **{{ .Release.Name }}**: Release name
- **{{ .Chart.Name }}**: Chart name
- **{{ .Template.Name }}**: Template name
- **{{- }}**: Trim whitespace

**Template Functions:**
- **include**: Include template
- **required**: Required value
- **default**: Default value
- **range**: Iterate over values
- **if/else**: Conditional logic

**Values Files:**
- **values.yaml**: Default values
- **-f values.yaml**: Override with file
- **--set**: Override with command line
- **--set-file**: Override from file

**Chart Dependencies:**
- **Chart.yaml**: Define dependencies
- **charts/**: Dependency charts
- **helm dependency update**: Update dependencies

**Best Practices:**
- Use semantic versioning
- Document values
- Provide sensible defaults
- Test charts before publishing`,
					CodeExamples: `# Create new chart
helm create mychart

# Chart.yaml
apiVersion: v2
name: myapp
description: My application chart
version: 1.0.0
appVersion: "1.0"

# values.yaml
replicaCount: 3
image:
  repository: nginx
  tag: "1.21"
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
  port: 80

# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "mychart.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ include "mychart.name" . }}
  template:
    metadata:
      labels:
        app: {{ include "mychart.name" . }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - containerPort: 80

# Template helpers (_helpers.tpl)
{{- define "mychart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

# Package chart
helm package mychart

# Install local chart
helm install myapp ./mychart

# Test chart (dry-run)
helm install myapp ./mychart --dry-run --debug`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          73,
			Title:       "Advanced Topics",
			Description: "Explore CRDs, Operators, service mesh, autoscaling, and multi-cluster management.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Custom Resource Definitions",
					Content: `**Custom Resource Definitions (CRDs)** extend Kubernetes API with custom resources.

**CRD Use Cases:**
- Domain-specific resources
- Application-specific configurations
- Operator-managed resources
- Custom controllers

**CRD Structure:**
- **Group**: API group (e.g., example.com)
- **Version**: API version (e.g., v1)
- **Kind**: Resource type (e.g., Database)
- **Plural**: Plural name (e.g., databases)

**CRD Spec:**
- **Names**: Kind, plural, singular, shortNames
- **Scope**: Namespaced or Cluster-scoped
- **Versions**: Multiple API versions
- **Schema**: OpenAPI v3 schema validation
- **Subresources**: Status, scale

**CRD Controller:**
- Watches CRD instances
- Implements business logic
- Updates status
- Creates/manages resources

**Operator Pattern:**
- CRD + Controller = Operator
- Manages complex applications
- Encodes operational knowledge
- Automates tasks`,
					CodeExamples: `# CRD definition
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
          status:
            type: object
            properties:
              phase:
                type: string
  scope: Namespaced
  names:
    plural: databases
    singular: database
    kind: Database
    shortNames:
    - db

# Custom resource instance
apiVersion: example.com/v1
kind: Database
metadata:
  name: mysql-db
spec:
  databaseName: myapp
  replicas: 3

# View CRDs
kubectl get crds
kubectl get databases

# View custom resource
kubectl get database mysql-db -o yaml`,
				},
				{
					Title: "Operators",
					Content: `**Operators** are Kubernetes controllers that manage complex applications using CRDs.

**Operator Benefits:**
- Automates operational tasks
- Encodes domain knowledge
- Self-healing capabilities
- Lifecycle management

**Operator Framework:**
- **Operator SDK**: Build operators
- **Operator Lifecycle Manager**: Install/manage operators
- **Operator Registry**: Operator catalog

**Common Operators:**
- **Prometheus Operator**: Manages Prometheus
- **PostgreSQL Operator**: Manages PostgreSQL
- **Elasticsearch Operator**: Manages Elasticsearch
- **Cert-Manager**: Manages certificates

**Operator Components:**
- **CRD**: Custom resource definition
- **Controller**: Watches and reconciles
- **RBAC**: Permissions for operator
- **Deployment**: Operator pod

**Operator Patterns:**
- **Reconciliation Loop**: Watch, compare, act
- **Status Updates**: Report current state
- **Finalizers**: Cleanup before deletion`,
					CodeExamples: `# Install Operator Lifecycle Manager
kubectl create -f https://github.com/operator-framework/operator-lifecycle-manager/releases/download/v0.25.0/crds.yaml
kubectl create -f https://github.com/operator-framework/operator-lifecycle-manager/releases/download/v0.25.0/olm.yaml

# Install operator via OLM
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: prometheus-operator
spec:
  channel: stable
  name: prometheus
  source: operatorhubio-catalog
  sourceNamespace: olm

# Use operator-managed resource
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: prometheus
spec:
  replicas: 2
  serviceAccountName: prometheus`,
				},
				{
					Title: "Autoscaling",
					Content: `**Autoscaling** automatically adjusts resources based on demand.

**Autoscaling Types:**

**Horizontal Pod Autoscaler (HPA):**
- Scales pods based on metrics
- CPU, memory, custom metrics
- Min/max replica limits
- Target utilization percentage

**Vertical Pod Autoscaler (VPA):**
- Adjusts resource requests/limits
- Based on historical usage
- Can downscale unused resources
- Requires admission controller

**Cluster Autoscaler:**
- Adds/removes nodes
- Based on pod scheduling needs
- Cloud provider integration
- Node pool management

**Metrics for HPA:**
- **Resource**: CPU, memory
- **Custom**: Application metrics
- **External**: External metrics
- **Object**: Object metrics`,
					CodeExamples: `# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

# HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa-custom
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app-deployment
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"

# View HPA
kubectl get hpa
kubectl describe hpa app-hpa`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          74,
			Title:       "Production Best Practices",
			Description: "Learn production-ready practices for running Kubernetes in production.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Resource Management",
					Content: `**Resource Management** ensures applications have adequate resources and don't interfere with each other.

**Resource Requests:**
- Minimum resources guaranteed
- Used for scheduling decisions
- Pod won't be scheduled if unavailable
- Should be set for all containers

**Resource Limits:**
- Maximum resources allowed
- Prevents resource exhaustion
- Container killed if exceeded
- Protects node stability

**Quality of Service (QoS):**
- **Guaranteed**: Requests = Limits (all resources)
- **Burstable**: Requests < Limits (some resources)
- **BestEffort**: No requests/limits

**Resource Quotas:**
- Limit resources per namespace
- CPU, memory, storage limits
- Object count limits
- Prevents resource exhaustion

**Limit Ranges:**
- Default requests/limits
- Min/max constraints
- Per-container or per-pod
- Enforced at pod creation`,
					CodeExamples: `# Pod with resource requests and limits
apiVersion: v1
kind: Pod
metadata:
  name: resource-pod
spec:
  containers:
  - name: app
    image: nginx:1.21
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"

# Resource Quota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: production
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    persistentvolumeclaims: "10"
    pods: "10"

# Limit Range
apiVersion: v1
kind: LimitRange
metadata:
  name: mem-limit-range
  namespace: production
spec:
  limits:
  - default:
      memory: "512Mi"
      cpu: "500m"
    defaultRequest:
      memory: "256Mi"
      cpu: "250m"
    type: Container
  - max:
      memory: "1Gi"
      cpu: "1"
    min:
      memory: "128Mi"
      cpu: "100m"
    type: Container

# View resource usage
kubectl top nodes
kubectl top pods
kubectl describe quota compute-quota`,
				},
				{
					Title: "Pod Disruption Budgets",
					Content: `**Pod Disruption Budgets (PDBs)** ensure minimum availability during voluntary disruptions.

**Use Cases:**
- Node maintenance
- Cluster upgrades
- Voluntary pod evictions
- Draining nodes

**PDB Spec:**
- **minAvailable**: Minimum pods available
- **maxUnavailable**: Maximum pods unavailable
- **selector**: Pods to protect
- Only one of minAvailable/maxUnavailable

**Voluntary Disruptions:**
- User-initiated actions
- Node drain
- Pod deletion
- Deployment updates

**Involuntary Disruptions:**
- Node failure
- Hardware failure
- Network partition
- Not protected by PDB

**Best Practices:**
- Set PDB for critical workloads
- Use minAvailable for high availability
- Test PDBs before production
- Monitor PDB violations`,
					CodeExamples: `# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: app-pdb
  namespace: production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: myapp

# PDB with maxUnavailable
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: app-pdb-max
  namespace: production
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: myapp

# View PDBs
kubectl get poddisruptionbudgets
kubectl describe pdb app-pdb

# Check if pod can be evicted
kubectl drain <node-name> --ignore-daemonsets`,
				},
				{
					Title: "Troubleshooting",
					Content: `**Troubleshooting** Kubernetes issues requires systematic investigation.

**Common Issues:**

**Pod Not Starting:**
- Check pod status: kubectl describe pod
- Check events: kubectl get events
- Check image pull errors
- Check resource constraints
- Check node capacity

**Pod Crashing:**
- Check logs: kubectl logs
- Check previous logs: kubectl logs --previous
- Check container exit code
- Check health probes
- Check resource limits

**Service Not Working:**
- Check service endpoints: kubectl get endpoints
- Check pod labels match selector
- Check service type
- Check network policies
- Check DNS resolution

**Debugging Commands:**
- kubectl describe: Detailed resource info
- kubectl logs: Container logs
- kubectl exec: Execute commands in pod
- kubectl get events: Cluster events
- kubectl top: Resource usage`,
					CodeExamples: `# Debug pod issues
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl logs <pod-name> --previous
kubectl get events --field-selector involvedObject.name=<pod-name>

# Debug service issues
kubectl describe service <service-name>
kubectl get endpoints <service-name>
kubectl get pods -l app=<label>
kubectl exec -it <pod-name> -- nslookup <service-name>

# Debug node issues
kubectl describe node <node-name>
kubectl get pods -o wide --field-selector spec.nodeName=<node-name>
kubectl top node <node-name>

# Debug deployment issues
kubectl describe deployment <deployment-name>
kubectl get replicasets
kubectl rollout history deployment/<deployment-name>
kubectl rollout status deployment/<deployment-name>

# Common troubleshooting workflow:
# 1. Check resource status
kubectl get all
# 2. Describe problematic resource
kubectl describe <resource-type> <resource-name>
# 3. Check logs
kubectl logs <pod-name>
# 4. Check events
kubectl get events --sort-by='.lastTimestamp'
# 5. Check resource usage
kubectl top pods
kubectl top nodes`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
