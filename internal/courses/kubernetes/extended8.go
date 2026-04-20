package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1726,
			Title:       "Multi-Cluster and Advanced Networking",
			Description: "Master multi-cluster Kubernetes: federation, service mesh, Ingress controllers, DNS, and cross-cluster communication.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Ingress Controllers and Load Balancing",
					Content: `Ingress exposes HTTP/HTTPS routes to services. Ingress controllers implement routing rules with features like TLS termination, rate limiting, and authentication.

**Ingress Controller Comparison:**
` + "```" + `
nginx-ingress (Kubernetes community):
  ✓ Most widely used
  ✓ Annotation-based configuration
  ✓ Good docs, large community
  ✗ Limited L7 features without Lua plugins

Traefik:
  ✓ Auto-discovery, middleware chains
  ✓ IngressRoute CRD for advanced routing
  ✓ Built-in Let's Encrypt
  ✓ Dashboard included

Envoy (via Contour, Ambassador/Emissary):
  ✓ High performance, gRPC-native
  ✓ Rate limiting, circuit breaking built-in
  ✓ xDS API for dynamic configuration
  ✓ Used by Istio as data plane

HAProxy:
  ✓ Very high performance
  ✓ TCP/L4 support
  ✗ Smaller Kubernetes community

AWS ALB Ingress Controller:
  ✓ Native AWS ALB integration
  ✓ WAF, Cognito auth integration
  ✓ Target group binding
  ✗ AWS-only

Gateway API (the future):
  ✓ Standard Kubernetes API (graduating to GA)
  ✓ Role-based: infra team manages Gateways, devs manage Routes
  ✓ TCP/UDP/gRPC support
  ✓ Cross-namespace routing
  ✓ Replaces Ingress long-term
` + "```" + `

**Nginx Ingress Configuration:**
` + "```" + `yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp
  namespace: production
  annotations:
    # TLS
    cert-manager.io/cluster-issuer: letsencrypt-prod
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "5"
    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com"
    # Authentication
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    # Redirect HTTP to HTTPS
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    # WebSocket support
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    # Custom headers
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "X-XSS-Protection: 1; mode=block";
    # Body size
    nginx.ingress.kubernetes.io/proxy-body-size: 50m
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - app.example.com
    - api.example.com
    secretName: app-tls
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
  - host: api.example.com
    http:
      paths:
      - path: /api/v1
        pathType: Prefix
        backend:
          service:
            name: api-v1
            port:
              number: 8080
      - path: /api/v2
        pathType: Prefix
        backend:
          service:
            name: api-v2
            port:
              number: 8080
` + "```" + `

**Gateway API:**
` + "```" + `yaml
# GatewayClass — managed by infra team
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: nginx
spec:
  controllerName: gateway.nginx.org/nginx-gateway-controller

---
# Gateway — shared infrastructure (infra team manages)
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: production-gateway
  namespace: gateway-system
spec:
  gatewayClassName: nginx
  listeners:
  - name: http
    protocol: HTTP
    port: 80
    hostname: "*.example.com"
    allowedRoutes:
      namespaces:
        from: All
  - name: https
    protocol: HTTPS
    port: 443
    hostname: "*.example.com"
    tls:
      mode: Terminate
      certificateRefs:
      - name: wildcard-tls
    allowedRoutes:
      namespaces:
        from: All

---
# HTTPRoute — dev team manages
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: myapp
  namespace: production
spec:
  parentRefs:
  - name: production-gateway
    namespace: gateway-system
  hostnames:
  - "app.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /api
      headers:
      - name: version
        value: v2
    backendRefs:
    - name: api-v2
      port: 8080
      weight: 100
  - matches:
    - path:
        type: PathPrefix
        value: /api
    backendRefs:
    - name: api-v1
      port: 8080
      weight: 90
    - name: api-v2
      port: 8080
      weight: 10
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: frontend
      port: 3000

---
# GRPCRoute
apiVersion: gateway.networking.k8s.io/v1alpha2
kind: GRPCRoute
metadata:
  name: grpc-service
  namespace: production
spec:
  parentRefs:
  - name: production-gateway
    namespace: gateway-system
  hostnames:
  - grpc.example.com
  rules:
  - matches:
    - method:
        service: myapp.v1.UserService
    backendRefs:
    - name: user-service
      port: 9090
` + "```" + `

**cert-manager:**
` + "```" + `yaml
# ClusterIssuer for Let's Encrypt
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
    - http01:
        ingress:
          ingressClassName: nginx
    - dns01:
        cloudDNS:
          project: my-gcp-project
        selector:
          dnsZones:
          - "example.com"

---
# Certificate
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: wildcard-tls
  namespace: production
spec:
  secretName: wildcard-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - "example.com"
  - "*.example.com"
  duration: 2160h    # 90 days
  renewBefore: 360h  # 15 days before expiry
` + "```" + ``,
					CodeExamples: `# Ingress and Gateway Examples

# 1. Multi-service Ingress with path-based routing
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: platform-ingress
  namespace: production
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - platform.example.com
    secretName: platform-tls
  rules:
  - host: platform.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-frontend
            port:
              number: 80
      - path: /api/users
        pathType: Prefix
        backend:
          service:
            name: user-service
            port:
              number: 8080
      - path: /api/orders
        pathType: Prefix
        backend:
          service:
            name: order-service
            port:
              number: 8080
      - path: /api/products
        pathType: Prefix
        backend:
          service:
            name: product-service
            port:
              number: 8080
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: websocket-service
            port:
              number: 8080
      - path: /grafana
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000

---
# 2. External DNS for automatic DNS records
apiVersion: apps/v1
kind: Deployment
metadata:
  name: external-dns
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: external-dns
  template:
    metadata:
      labels:
        app: external-dns
    spec:
      serviceAccountName: external-dns
      containers:
      - name: external-dns
        image: registry.k8s.io/external-dns/external-dns:v0.14.0
        args:
        - --source=ingress
        - --source=service
        - --domain-filter=example.com
        - --provider=aws
        - --policy=sync
        - --aws-zone-type=public
        - --registry=txt
        - --txt-owner-id=production-cluster

---
# 3. Internal/External Ingress separation
apiVersion: networking.k8s.io/v1
kind: IngressClass
metadata:
  name: nginx-internal
spec:
  controller: k8s.io/ingress-nginx-internal
---
apiVersion: networking.k8s.io/v1
kind: IngressClass
metadata:
  name: nginx-external
spec:
  controller: k8s.io/ingress-nginx-external

---
# Public API
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: public-api
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/limit-rps: "50"
spec:
  ingressClassName: nginx-external
  tls:
  - hosts: [api.example.com]
    secretName: api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /api/v1/public
        pathType: Prefix
        backend:
          service:
            name: public-api
            port:
              number: 8080

---
# Internal admin (only within VPC/VPN)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: admin-internal
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/whitelist-source-range: "10.0.0.0/8"
spec:
  ingressClassName: nginx-internal
  rules:
  - host: admin.internal.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admin-dashboard
            port:
              number: 8080

---
# 4. Gateway API with traffic splitting
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: canary-route
  namespace: production
spec:
  parentRefs:
  - name: production-gateway
    namespace: gateway-system
  hostnames:
  - app.example.com
  rules:
  - matches:
    - headers:
      - name: x-canary
        value: "true"
    backendRefs:
    - name: myapp-canary
      port: 8080
  - backendRefs:
    - name: myapp-stable
      port: 8080
      weight: 95
    - name: myapp-canary
      port: 8080
      weight: 5
    filters:
    - type: ResponseHeaderModifier
      responseHeaderModifier:
        add:
        - name: X-Served-By
          value: stable

---
# 5. Rate limiting with Gateway API
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: rate-limited-api
  namespace: production
spec:
  parentRefs:
  - name: production-gateway
    namespace: gateway-system
  hostnames:
  - api.example.com
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /api
    backendRefs:
    - name: api-service
      port: 8080
    filters:
    - type: ExtensionRef
      extensionRef:
        group: gateway.envoyproxy.io
        kind: RateLimitFilter
        name: api-rate-limit`,
				},
				{
					Title: "Multi-Cluster Management",
					Content: `Multi-cluster Kubernetes manages workloads across multiple clusters for high availability, disaster recovery, and geographic distribution.

**Multi-Cluster Patterns:**
` + "```" + `
Patterns:
  1. Active-Active:  Traffic to all clusters, load balanced
  2. Active-Passive: Primary cluster, failover to secondary
  3. Regional:       Clusters per region, traffic routed by geo
  4. Hub-Spoke:      Management cluster + workload clusters

Use cases:
  - High availability / disaster recovery
  - Geographic compliance (data sovereignty)
  - Blast radius reduction
  - Different environments (dev/staging/prod)
  - Edge computing

Tools:
  Cluster API      → cluster lifecycle management
  Submariner       → cross-cluster networking
  Liqo             → multi-cluster resource sharing
  KubeFed          → Kubernetes federation (deprecated)
  Admiralty         → multi-cluster scheduling
  Skupper           → application-level connectivity
  k0rdent          → fleet management
` + "```" + `

**Cluster API:**
` + "```" + `yaml
# Cluster API manages cluster lifecycle declaratively

# Cluster definition
apiVersion: cluster.x-k8s.io/v1beta1
kind: Cluster
metadata:
  name: production-east
  namespace: clusters
spec:
  clusterNetwork:
    pods:
      cidrBlocks: ["192.168.0.0/16"]
    services:
      cidrBlocks: ["10.96.0.0/12"]
  controlPlaneRef:
    apiVersion: controlplane.cluster.x-k8s.io/v1beta1
    kind: KubeadmControlPlane
    name: production-east-cp
  infrastructureRef:
    apiVersion: infrastructure.cluster.x-k8s.io/v1beta2
    kind: AWSCluster
    name: production-east

---
# AWS infrastructure
apiVersion: infrastructure.cluster.x-k8s.io/v1beta2
kind: AWSCluster
metadata:
  name: production-east
  namespace: clusters
spec:
  region: us-east-1
  sshKeyName: k8s-key
  network:
    vpc:
      cidrBlock: 10.0.0.0/16
    subnets:
    - availabilityZone: us-east-1a
      cidrBlock: 10.0.1.0/24
      isPublic: false
    - availabilityZone: us-east-1b
      cidrBlock: 10.0.2.0/24
      isPublic: false

---
# Control plane
apiVersion: controlplane.cluster.x-k8s.io/v1beta1
kind: KubeadmControlPlane
metadata:
  name: production-east-cp
  namespace: clusters
spec:
  replicas: 3
  version: v1.29.0
  machineTemplate:
    infrastructureRef:
      apiVersion: infrastructure.cluster.x-k8s.io/v1beta2
      kind: AWSMachineTemplate
      name: production-east-cp
  kubeadmConfigSpec:
    initConfiguration:
      nodeRegistration:
        kubeletExtraArgs:
          cloud-provider: external
    clusterConfiguration:
      apiServer:
        extraArgs:
          cloud-provider: external

---
# Worker nodes
apiVersion: cluster.x-k8s.io/v1beta1
kind: MachineDeployment
metadata:
  name: production-east-workers
  namespace: clusters
spec:
  clusterName: production-east
  replicas: 5
  selector:
    matchLabels: {}
  template:
    spec:
      clusterName: production-east
      version: v1.29.0
      bootstrap:
        configRef:
          apiVersion: bootstrap.cluster.x-k8s.io/v1beta1
          kind: KubeadmConfigTemplate
          name: production-east-workers
      infrastructureRef:
        apiVersion: infrastructure.cluster.x-k8s.io/v1beta2
        kind: AWSMachineTemplate
        name: production-east-workers
` + "```" + `

**Cross-Cluster Service Discovery:**
` + "```" + `
Submariner — L3 connectivity between clusters:
  - Creates encrypted tunnels between clusters
  - ServiceImport/ServiceExport for cross-cluster services
  - GlobalNet for overlapping CIDRs
  
  # Export service from cluster A:
  apiVersion: multicluster.x-k8s.io/v1alpha1
  kind: ServiceExport
  metadata:
    name: database
    namespace: production
  
  # Service available in cluster B as:
  database.production.svc.clusterset.local

Istio multicluster:
  - Shared control plane or multi-network
  - mTLS between clusters
  - Transparent cross-cluster routing
` + "```" + `

**Multi-Cluster GitOps:**
` + "```" + `yaml
# ArgoCD managing multiple clusters
apiVersion: v1
kind: Secret
metadata:
  name: production-east
  namespace: argocd
  labels:
    argocd.argoproj.io/secret-type: cluster
type: Opaque
stringData:
  name: production-east
  server: https://k8s-east.example.com
  config: |
    {
      "bearerToken": "<token>",
      "tlsClientConfig": {
        "insecure": false,
        "caData": "<ca-data>"
      }
    }

---
# ApplicationSet for multi-cluster deployment
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: platform-apps
  namespace: argocd
spec:
  generators:
  - clusters:
      selector:
        matchLabels:
          env: production
  template:
    metadata:
      name: "platform-{{name}}"
    spec:
      project: platform
      source:
        repoURL: https://github.com/org/platform
        path: "clusters/{{metadata.labels.region}}"
        targetRevision: main
      destination:
        server: "{{server}}"
        namespace: platform
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
` + "```" + ``,
					CodeExamples: `# Multi-Cluster Configuration

# 1. DNS-based multi-cluster routing with ExternalDNS
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: production
  annotations:
    external-dns.alpha.kubernetes.io/hostname: app.example.com
    external-dns.alpha.kubernetes.io/ttl: "60"
spec:
  type: LoadBalancer
  selector:
    app: myapp
  ports:
  - port: 443
    targetPort: 8080

---
# 2. Global load balancing with Route53 health checks
# (Applied via Terraform/Crossplane in each cluster)
# Route53 weighted routing:
#   app.example.com → cluster-east (weight: 50, health check: /healthz)
#   app.example.com → cluster-west (weight: 50, health check: /healthz)
# Automatic failover when health check fails

---
# 3. Crossplane for multi-cluster infrastructure
apiVersion: aws.crossplane.io/v1alpha1
kind: DBInstance
metadata:
  name: orders-db-east
  namespace: crossplane-system
spec:
  forProvider:
    region: us-east-1
    dbInstanceClass: db.r6g.xlarge
    engine: postgres
    engineVersion: "16"
    masterUsername: admin
    allocatedStorage: 100
    multiAZ: true
    storageEncrypted: true
    vpcSecurityGroupIds:
    - sg-12345678
    dbSubnetGroupName: private-subnets
  writeConnectionSecretToRef:
    name: orders-db-credentials
    namespace: production

---
# 4. Velero backup for DR
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
    - staging
    storageLocation: aws-east
    volumeSnapshotLocations:
    - aws-east
    ttl: 720h  # 30 days
    snapshotMoveData: true  # Cross-region copy
  useOwnerReferencesInBackup: false
---
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: aws-east
  namespace: velero
spec:
  provider: aws
  objectStorage:
    bucket: velero-backups-east
    prefix: production
  config:
    region: us-east-1

---
# 5. Resource Quotas per namespace (multi-tenancy)
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-quota
  namespace: team-alpha
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "50"
    services.loadbalancers: "5"
    pods: "100"
    secrets: "100"
    configmaps: "100"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: team-alpha
spec:
  limits:
  - default:
      cpu: 500m
      memory: 512Mi
    defaultRequest:
      cpu: 100m
      memory: 128Mi
    type: Container
  - max:
      cpu: "8"
      memory: 16Gi
    type: Container

---
# 6. PodDisruptionBudget for HA workloads
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: myapp-pdb
  namespace: production
spec:
  minAvailable: "80%"
  selector:
    matchLabels:
      app: myapp

---
# 7. Priority Classes for workload prioritization
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: critical-production
value: 1000000
globalDefault: false
description: "Critical production workloads - highest priority"
preemptionPolicy: PreemptLowerPriority
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: standard-production
value: 500000
globalDefault: true
description: "Standard production workloads"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: batch-processing
value: 100000
globalDefault: false
description: "Batch jobs - can be preempted"
preemptionPolicy: Never`,
				},
			},
		},
	})
}
