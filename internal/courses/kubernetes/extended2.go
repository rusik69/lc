package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1720,
			Title:       "Kubernetes Security",
			Description: "Master Kubernetes security: RBAC, Pod Security Standards, network policies, secrets management, and supply chain security.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "RBAC Deep Dive",
					Content: `Role-Based Access Control (RBAC) is the primary authorization mechanism in Kubernetes. It controls who can do what on which resources.

**Core RBAC Concepts:**
` + "```" + `
Four resource types:
  Role              → namespace-scoped permissions
  ClusterRole       → cluster-wide permissions
  RoleBinding       → binds Role/ClusterRole to subjects in a namespace
  ClusterRoleBinding → binds ClusterRole to subjects cluster-wide

Subjects:
  User              → human user (managed externally via certs/OIDC)
  Group             → group of users
  ServiceAccount    → identity for pods

Verbs (permissions):
  get, list, watch         → read operations
  create, update, patch    → write operations
  delete, deletecollection → delete operations
  impersonate              → act as another user
  bind, escalate           → RBAC-specific (dangerous)
` + "```" + `

**Role and ClusterRole:**
` + "```" + `yaml
# Namespace-scoped Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: pod-reader
rules:
- apiGroups: [""]           # core API group
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]    # subresource
  verbs: ["get"]

---
# ClusterRole for cluster-wide access
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: node-reader
rules:
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["nodes"]
  verbs: ["get", "list"]

---
# Aggregated ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: monitoring-view
  labels:
    rbac.authorization.k8s.io/aggregate-to-view: "true"
rules:
- apiGroups: ["monitoring.coreos.com"]
  resources: ["prometheuses", "alertmanagers", "servicemonitors"]
  verbs: ["get", "list", "watch"]
` + "```" + `

**Bindings:**
` + "```" + `yaml
# Bind Role to ServiceAccount
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
- kind: ServiceAccount
  name: monitoring-sa
  namespace: production
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

---
# Bind ClusterRole to Group
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: read-nodes-devs
subjects:
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: node-reader
  apiGroup: rbac.authorization.k8s.io

---
# Bind ClusterRole to User in specific namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: admin-staging
  namespace: staging
subjects:
- kind: User
  name: alice@example.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: admin
  apiGroup: rbac.authorization.k8s.io
` + "```" + `

**ServiceAccount Best Practices:**
` + "```" + `yaml
# Dedicated ServiceAccount per workload
apiVersion: v1
kind: ServiceAccount
metadata:
  name: order-processor
  namespace: production
  annotations:
    # AWS IAM Roles for Service Accounts (IRSA)
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/order-processor
    # GCP Workload Identity
    iam.gke.io/gcp-service-account: order-proc@project.iam.gserviceaccount.com
automountServiceAccountToken: false  # Don't mount unless needed

---
# Pod using the ServiceAccount
apiVersion: v1
kind: Pod
metadata:
  name: order-processor
  namespace: production
spec:
  serviceAccountName: order-processor
  automountServiceAccountToken: true  # Explicitly opt in
  containers:
  - name: processor
    image: myapp/order-processor:v1
` + "```" + `

**RBAC Debugging:**
` + "```" + `
# Check if user can perform action
kubectl auth can-i create deployments --namespace production
kubectl auth can-i create deployments --namespace production --as alice@example.com
kubectl auth can-i '*' '*' --as system:serviceaccount:kube-system:default

# List all roles/bindings
kubectl get roles,rolebindings -n production
kubectl get clusterroles,clusterrolebindings

# Describe to see rules
kubectl describe clusterrole admin
kubectl describe rolebinding read-pods -n production

# Audit who has access to what (using kubectl-who-can plugin)
kubectl who-can create pods -n production
kubectl who-can delete nodes
` + "```" + ``,
					CodeExamples: `# RBAC Configuration Examples

# 1. Developer role - can manage deployments and services but not secrets
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: developer
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "persistentvolumeclaims"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods/log", "pods/exec", "pods/portforward"]
  verbs: ["get", "create"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
# Secrets: read-only  
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list", "watch"]
# No cluster-level access (nodes, namespaces, etc.)

---
# 2. CI/CD ServiceAccount - deploy only
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cicd-deployer
  namespace: cicd
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: deployer
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: ["apps"]
  resources: ["deployments/rollback"]
  verbs: ["create"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "list", "watch"]
---
# Bind deployer to specific namespaces
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: cicd-deploy-staging
  namespace: staging
subjects:
- kind: ServiceAccount
  name: cicd-deployer
  namespace: cicd
roleRef:
  kind: ClusterRole
  name: deployer
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: cicd-deploy-production
  namespace: production
subjects:
- kind: ServiceAccount
  name: cicd-deployer
  namespace: cicd
roleRef:
  kind: ClusterRole
  name: deployer
  apiGroup: rbac.authorization.k8s.io

---
# 3. Read-only auditor
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: auditor
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["get", "list", "watch"]
# Exclude secrets from listing
- apiGroups: [""]
  resources: ["secrets"]
  verbs: []  # Override: no secret access
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: security-auditor
subjects:
- kind: Group
  name: security-team
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: auditor
  apiGroup: rbac.authorization.k8s.io

---
# 4. Namespace admin (full access within namespace)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: team-alpha-admin
  namespace: team-alpha
subjects:
- kind: Group
  name: team-alpha-leads
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: admin  # Built-in admin role
  apiGroup: rbac.authorization.k8s.io

---
# 5. Least-privilege monitoring
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus-reader
rules:
- apiGroups: [""]
  resources: ["nodes", "nodes/metrics", "services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["extensions", "networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics", "/metrics/cadvisor"]
  verbs: ["get"]`,
				},
				{
					Title: "Pod Security Standards",
					Content: `Pod Security Standards (PSS) replace the deprecated PodSecurityPolicy (PSP). They define three security profiles enforced via Pod Security Admission (PSA).

**Three Security Profiles:**
` + "```" + `
Privileged:
  - Unrestricted policy
  - For system-level workloads (CNI, storage drivers)
  - Allows everything

Baseline:
  - Prevents known privilege escalations
  - Blocks: hostNetwork, hostPID, hostIPC, privileged containers
  - Allows: most standard workloads
  - Good default for most apps

Restricted:
  - Heavily restricted, follows hardening best practices
  - Requires: non-root, drop ALL capabilities, read-only root FS
  - Requires: seccomp profile, no privilege escalation
  - For security-sensitive workloads

Enforcement modes:
  enforce  → reject pods that violate the policy
  audit    → log violations but allow the pod
  warn     → display warnings to user but allow the pod
` + "```" + `

**Namespace Labels:**
` + "```" + `yaml
# Apply security standards to namespace
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # Enforce restricted for all pods
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    # Audit and warn at restricted level
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest

---
# Baseline for less critical namespace
apiVersion: v1
kind: Namespace
metadata:
  name: development
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/warn: restricted

---
# Privileged for system namespace
apiVersion: v1
kind: Namespace
metadata:
  name: kube-system
  labels:
    pod-security.kubernetes.io/enforce: privileged
` + "```" + `

**Restricted-Compliant Pod:**
` + "```" + `yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  namespace: production
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 65534
    runAsGroup: 65534
    fsGroup: 65534
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: myapp:v1
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
      runAsNonRoot: true
    resources:
      limits:
        cpu: "500m"
        memory: "256Mi"
      requests:
        cpu: "100m"
        memory: "128Mi"
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /var/cache
  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir:
      sizeLimit: 100Mi
` + "```" + `

**Security Context Fields:**
` + "```" + `
Pod-level securityContext:
  runAsUser         → UID to run containers as
  runAsGroup        → GID for all containers
  fsGroup           → GID for volume ownership
  runAsNonRoot      → must run as non-root (fail otherwise)
  seccompProfile    → syscall filter profile
  sysctls           → kernel parameters
  supplementalGroups → additional GIDs

Container-level securityContext:
  allowPrivilegeEscalation → can process gain more privileges (setuid)
  readOnlyRootFilesystem   → root FS is read-only
  capabilities.drop        → Linux capabilities to drop
  capabilities.add         → Linux capabilities to add
  privileged               → full host access (DANGEROUS)
  procMount                → /proc mount type
  seccompProfile           → per-container seccomp
  seLinuxOptions           → SELinux labels

Capability management:
  DROP ALL first, then add only what's needed:
  capabilities:
    drop: ["ALL"]
    add: ["NET_BIND_SERVICE"]  # Only if binding port < 1024
  
  Never add unless required:
    SYS_ADMIN    → almost like root
    NET_RAW      → raw sockets (packet crafting)
    SYS_PTRACE   → debug other processes
` + "```" + ``,
					CodeExamples: `# Pod Security Standards Examples

# 1. Hardened Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: app
        image: myregistry/web-app:v1.2.3
        ports:
        - containerPort: 8080
          protocol: TCP
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
          requests:
            cpu: "200m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: config
          mountPath: /etc/app
          readOnly: true
        env:
        - name: APP_PORT
          value: "8080"
      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 50Mi
      - name: config
        configMap:
          name: web-app-config

---
# 2. Init container with shared volume
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-init
  namespace: production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: app-with-init
  template:
    metadata:
      labels:
        app: app-with-init
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      initContainers:
      - name: config-fetcher
        image: myregistry/config-fetcher:v1
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: config
          mountPath: /config
        command: ["sh", "-c", "cp /defaults/* /config/"]
      containers:
      - name: app
        image: myregistry/app:v1
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: config
          mountPath: /etc/app
          readOnly: true
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: config
        emptyDir: {}
      - name: tmp
        emptyDir: {}

---
# 3. Kyverno policy to enforce security standards
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-non-root
spec:
  validationFailureAction: Enforce
  background: true
  rules:
  - name: check-runAsNonRoot
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Containers must run as non-root"
      pattern:
        spec:
          containers:
          - securityContext:
              runAsNonRoot: true
  - name: check-capabilities
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Containers must drop ALL capabilities"
      pattern:
        spec:
          containers:
          - securityContext:
              capabilities:
                drop: ["ALL"]`,
				},
				{
					Title: "Network Policies",
					Content: `NetworkPolicy is Kubernetes' built-in firewall. It controls pod-to-pod, pod-to-service, and external traffic at L3/L4 (IP/port level).

**Key Concepts:**
` + "```" + `
Default behavior:
  - Without any NetworkPolicy: ALL traffic is allowed
  - Once you apply a NetworkPolicy selecting a pod:
    - Selected traffic type (ingress/egress) defaults to DENY ALL
    - Only explicitly allowed traffic is permitted
  - Policies are additive (union of all matching policies)

Requirements:
  - CNI plugin must support NetworkPolicy
  - Supported: Calico, Cilium, Antrea, Weave Net
  - NOT supported: Flannel (by default)

Policy types:
  - Ingress: incoming traffic to selected pods
  - Egress: outgoing traffic from selected pods
  - Both can be specified in one policy
` + "```" + `

**Default Deny Policies:**
` + "```" + `yaml
# Default deny ALL ingress in namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}  # selects ALL pods in namespace
  policyTypes:
  - Ingress

---
# Default deny ALL egress in namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress

---
# Allow DNS egress (required for service discovery)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
` + "```" + `

**Application-Specific Policies:**
` + "```" + `yaml
# Backend receives from frontend only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

---
# Database: only from backend, specific port
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 5432
  egress: []  # No outbound needed

---
# Cross-namespace access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-monitoring
  namespace: production
spec:
  podSelector: {}  # All pods in production
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090

---
# CIDR-based egress (external service)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - ipBlock:
        cidr: 10.0.0.0/8
        except:
        - 10.0.1.0/24
    ports:
    - protocol: TCP
      port: 443
  - to:  # Allow DNS
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - protocol: UDP
      port: 53
` + "```" + `

**Cilium Network Policies (L7):**
` + "```" + `yaml
# Cilium L7 policy — HTTP path-based
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: api-l7-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: api
  ingress:
  - fromEndpoints:
    - matchLabels:
        app: frontend
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - method: GET
          path: "/api/v1/users"
        - method: GET
          path: "/api/v1/products"
        - method: POST
          path: "/api/v1/orders"
` + "```" + ``,
					CodeExamples: `# Complete Network Security Setup

# Step 1: Default deny everything
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# Step 2: Allow DNS for all pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53

---
# Step 3: Frontend - ingress from ingress controller, egress to backend
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: frontend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: ingress-nginx
    ports:
    - protocol: TCP
      port: 3000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 8080

---
# Step 4: Backend - from frontend, egress to database and cache
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
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
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:  # External API
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 10.0.0.0/8
        - 172.16.0.0/12
        - 192.168.0.0/16
    ports:
    - protocol: TCP
      port: 443

---
# Step 5: Database - ingress from backend only, no egress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: postgres-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 5432
  egress: []  # No outbound

---
# Step 6: Redis cache - ingress from backend only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: redis-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: redis
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 6379
  egress: []

---
# Step 7: Allow inter-pod communication for replicas (e.g., Redis Cluster)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: redis-cluster-internal
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: redis-cluster
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: redis-cluster
    ports:
    - protocol: TCP
      port: 6379
    - protocol: TCP
      port: 16379  # Cluster bus
  - from:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 6379
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis-cluster
    ports:
    - protocol: TCP
      port: 6379
    - protocol: TCP
      port: 16379`,
				},
			},
		},
	})
}
