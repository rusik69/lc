package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1731,
			Title:       "Kubernetes in Production",
			Description: "Master production Kubernetes: high availability, disaster recovery, cost optimization, upgrade strategies, and operational best practices.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "High Availability and Disaster Recovery",
					Content: `Production Kubernetes clusters must be highly available and have disaster recovery plans. A single component failure should not cause downtime.

**Control Plane HA:**
` + "```" + `
HA control plane requires:
  - 3+ API server instances (behind load balancer)
  - 3+ etcd members (Raft consensus)
  - Multiple scheduler and controller-manager (leader election)

API Server:
  - Stateless → can run multiple instances
  - Load balanced (cloud LB or HAProxy/keepalived)
  - Each instance talks to all etcd members

etcd:
  - Raft consensus → needs quorum (N/2 + 1)
  - 3 members: tolerates 1 failure
  - 5 members: tolerates 2 failures
  - More members = more reliable but slower writes
  - Recommended: 3 for small, 5 for large clusters

Scheduler & Controller Manager:
  - Only one active (leader election via Kubernetes lease)
  - Others are standby
  - Automatic failover on leader failure

Topology:
  Stacked etcd:    etcd runs on control plane nodes
    Pro: simpler, fewer nodes
    Con: losing a node loses both API server and etcd member
    
  External etcd:   etcd runs on dedicated nodes
    Pro: better isolation, independent scaling
    Con: more nodes to manage
    
  # For most production: stacked etcd with 3 control plane nodes
  # spread across availability zones
` + "```" + `

**Worker Node HA:**
` + "```" + `
Strategies:
  1. Multi-AZ deployment
     - Nodes spread across 3 availability zones
     - Pods scheduled across zones (topologySpreadConstraints)
     - Tolerates AZ failure
  
  2. PodDisruptionBudgets
     - Ensure minimum available pods during disruptions
     - Prevents draining too many pods at once
  
  3. Pod anti-affinity
     - Don't schedule replicas on same node
     - Spread across fault domains
  
  4. Resource headroom
     - Don't run at 100% capacity
     - Leave room for rescheduling during node failure
     - Rule: 30% headroom for N+1 style redundancy

Pod topology spread:
  spec:
    topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: DoNotSchedule
      labelSelector:
        matchLabels:
          app: myapp
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: ScheduleAnyway
      labelSelector:
        matchLabels:
          app: myapp
` + "```" + `

**Disaster Recovery:**
` + "```" + `
Backup strategy:
  1. etcd snapshots (most critical)
     - Contains ALL cluster state
     - Schedule regular snapshots
     - Store in off-site/cross-region storage
     - Test restoration regularly!
     
     # Backup:
     ETCDCTL_API=3 etcdctl snapshot save /backup/etcd-$(date +%Y%m%d).db \
       --endpoints=https://127.0.0.1:2379 \
       --cacert=/etc/kubernetes/pki/etcd/ca.crt \
       --cert=/etc/kubernetes/pki/etcd/server.crt \
       --key=/etc/kubernetes/pki/etcd/server.key
     
     # Verify:
     ETCDCTL_API=3 etcdctl snapshot status /backup/etcd-latest.db

  2. Application-level backups (Velero)
     - Backs up Kubernetes resources + persistent volumes
     - Scheduled or on-demand
     - Cross-cluster restore
     - Storage: S3, GCS, Azure Blob

  3. GitOps as backup
     - All manifests in Git = declarative backup
     - Restore: point GitOps tool to new cluster
     - Limitation: doesn't backup runtime state (PVs, PVCs data)

Recovery Time Objective (RTO):
  - etcd restore: minutes
  - Full cluster rebuild: hours
  - GitOps re-apply: minutes to hours
  - PV data restore: depends on storage backend

Recovery Point Objective (RPO):
  - etcd snapshots: typically last snapshot (hourly = 1h RPO)
  - Velero: depends on schedule
  - PV snapshots: depends on schedule
` + "```" + `

**Velero Backup Configuration:**
` + "```" + `yaml
# Velero scheduled backup
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: production-daily
  namespace: velero
spec:
  schedule: "0 1 * * *"
  template:
    includedNamespaces:
    - production
    - monitoring
    storageLocation: default
    volumeSnapshotLocations:
    - default
    ttl: 720h
    includedResources:
    - '*'
    excludedResources:
    - events
    - events.events.k8s.io
    labelSelector:
      matchExpressions:
      - key: velero.io/exclude
        operator: DoesNotExist
    snapshotMoveData: true
    defaultVolumesToFsBackup: false

---
# Restore
apiVersion: velero.io/v1
kind: Restore
metadata:
  name: production-restore
  namespace: velero
spec:
  backupName: production-daily-20240101010000
  includedNamespaces:
  - production
  restorePVs: true
  preserveNodePorts: true
  existingResourcePolicy: none

---
# Backup storage location
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: default
  namespace: velero
spec:
  provider: aws
  objectStorage:
    bucket: velero-backups-production
    prefix: cluster-east
  config:
    region: us-east-1
    s3ForcePathStyle: "true"

---
# Volume snapshot location
apiVersion: velero.io/v1
kind: VolumeSnapshotLocation
metadata:
  name: default
  namespace: velero
spec:
  provider: aws
  config:
    region: us-east-1
` + "```" + ``,
					CodeExamples: `# Production HA Configuration

# 1. HA Deployment with all best practices
apiVersion: apps/v1
kind: Deployment
metadata:
  name: critical-api
  namespace: production
  labels:
    app: critical-api
    tier: frontend
spec:
  replicas: 5
  revisionHistoryLimit: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: "25%"
      maxUnavailable: 0  # Zero downtime
  selector:
    matchLabels:
      app: critical-api
  template:
    metadata:
      labels:
        app: critical-api
        tier: frontend
    spec:
      serviceAccountName: critical-api
      automountServiceAccountToken: false
      terminationGracePeriodSeconds: 60
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      # Spread across zones and nodes
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: critical-api
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: critical-api
      # Anti-affinity
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values: [critical-api]
              topologyKey: kubernetes.io/hostname
      containers:
      - name: api
        image: myregistry/critical-api:v2.0.0
        ports:
        - containerPort: 8080
          name: http
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        resources:
          requests:
            cpu: "1"
            memory: 1Gi
          limits:
            cpu: "1"
            memory: 1Gi
        # Startup probe (slow-starting apps)
        startupProbe:
          httpGet:
            path: /healthz
            port: http
          failureThreshold: 30
          periodSeconds: 2
        # Liveness (is the app alive?)
        livenessProbe:
          httpGet:
            path: /healthz
            port: http
          initialDelaySeconds: 0
          periodSeconds: 10
          failureThreshold: 3
        # Readiness (is the app ready for traffic?)
        readinessProbe:
          httpGet:
            path: /readyz
            port: http
          initialDelaySeconds: 0
          periodSeconds: 5
          failureThreshold: 2
        # Graceful shutdown
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 100Mi

---
# 2. PDB for zero-downtime updates
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: critical-api-pdb
  namespace: production
spec:
  minAvailable: "80%"
  selector:
    matchLabels:
      app: critical-api

---
# 3. etcd backup CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          hostNetwork: true
          nodeSelector:
            node-role.kubernetes.io/control-plane: ""
          tolerations:
          - key: node-role.kubernetes.io/control-plane
            effect: NoSchedule
          containers:
          - name: backup
            image: bitnami/etcd:3.5
            command:
            - /bin/sh
            - -c
            - |
              TIMESTAMP=$(date +%Y%m%d-%H%M%S)
              etcdctl snapshot save /backup/etcd-${TIMESTAMP}.db \
                --endpoints=https://127.0.0.1:2379 \
                --cacert=/etc/kubernetes/pki/etcd/ca.crt \
                --cert=/etc/kubernetes/pki/etcd/server.crt \
                --key=/etc/kubernetes/pki/etcd/server.key
              etcdctl snapshot status /backup/etcd-${TIMESTAMP}.db
              # Upload to S3
              # aws s3 cp /backup/etcd-${TIMESTAMP}.db s3://backups/etcd/
              # Cleanup old backups
              find /backup -name "etcd-*.db" -mtime +7 -delete
            volumeMounts:
            - name: etcd-certs
              mountPath: /etc/kubernetes/pki/etcd
              readOnly: true
            - name: backup-dir
              mountPath: /backup
          volumes:
          - name: etcd-certs
            hostPath:
              path: /etc/kubernetes/pki/etcd
          - name: backup-dir
            hostPath:
              path: /var/backup/etcd
          restartPolicy: OnFailure

---
# 4. Health check endpoints implementation
# (Go HTTP handlers for /healthz and /readyz)
#
# /healthz — Liveness: Is the process running correctly?
#   - Check: basic health (not deadlocked, not corrupted)
#   - Should NOT check dependencies (DB, cache)
#   - Simple: return 200 if process is alive
#
# /readyz — Readiness: Can this pod serve traffic?
#   - Check: dependencies available
#   - Database connection pool healthy
#   - Cache reachable
#   - Initial data loaded
#   - Returns 503 during graceful shutdown

# /startupz — Startup: Has the app started?
#   - Check: initialization complete
#   - Migrations run
#   - Caches warmed
#   - Connections established`,
				},
				{
					Title: "Cost Optimization and Operational Best Practices",
					Content: `Running Kubernetes in production requires attention to costs, operational efficiency, and ongoing cluster maintenance.

**Cost Optimization:**
` + "```" + `
Top cost areas:
  1. Compute (60-70% of cluster cost)
  2. Storage (15-25%)
  3. Network / Data transfer (5-15%)
  4. Load balancers

Compute optimization:
  Right-sizing:
    - Use VPA in recommend mode to find actual usage
    - Most pods are over-provisioned (2-5x)
    - Set requests based on p95 usage + 20% buffer
    - Review monthly
  
  Spot/Preemptible instances:
    - 60-90% cost savings
    - Use for: batch jobs, dev/staging, stateless workers
    - Don't use for: databases, control plane
    - Handle interruptions: PDBs, graceful shutdown
    
    # Karpenter spot config
    requirements:
    - key: karpenter.sh/capacity-type
      operator: In
      values: ["spot", "on-demand"]
    # Uses spot first, falls back to on-demand

  Autoscaling:
    - HPA: scale pods to match load
    - Cluster autoscaler: scale nodes to match pods
    - Scale down aggressively during off-peak
    - KEDA for scale-to-zero

  Bin-packing:
    - Karpenter: selects optimal instance types
    - Consolidation: replaces underutilized nodes
    - node.kubernetes.io/instance-type diversity

Storage optimization:
  - Delete unused PVCs (orphan detection)
  - Use appropriate storage tiers (hot/warm/cold)
  - Compress and archive old data
  - Volume snapshots instead of full backups

Network optimization:
  - Keep traffic in same AZ (topology-aware routing)
  - internalTrafficPolicy: Local
  - Avoid cross-region data transfer
  - Use private endpoints for cloud services
` + "```" + `

**Cluster Upgrade Strategy:**
` + "```" + `
Pre-upgrade checklist:
  □ Review release notes and deprecations
  □ Check API version compatibility
  □ Test upgrade in staging cluster first
  □ Backup etcd
  □ Verify PDBs are in place
  □ Ensure sufficient node headroom
  □ Notify team and schedule maintenance window

Upgrade approaches:
  1. In-place rolling upgrade:
     - Upgrade control plane first
     - Then rolling upgrade workers (one at a time)
     - Minimal disruption
     - Rollback: harder
  
  2. Blue-green cluster:
     - Create new cluster with new version
     - Migrate workloads
     - Switch traffic
     - Delete old cluster
     - Safest but most resource-intensive
  
  3. Canary nodes:
     - Add nodes with new version
     - Gradually drain old nodes
     - Monitor for issues
     - Rollback: just remove new nodes

Node upgrade process:
  1. Cordon node (no new pods scheduled)
     kubectl cordon <node>
  2. Drain node (evict pods respecting PDBs)
     kubectl drain <node> --ignore-daemonsets --delete-emptydir-data
  3. Upgrade kubelet and components
  4. Reboot if needed
  5. Uncordon
     kubectl uncordon <node>
  6. Verify
     kubectl get node <node>

Version skew policy:
  kubelet:           ±1 minor version from API server
  kube-proxy:        same or older minor version as kubelet
  kubectl:           ±1 minor version from API server
  → Upgrade API server first, then kubelets
` + "```" + `

**Operational Runbooks:**
` + "```" + `
Common scenarios and responses:

Node unresponsive:
  1. Check node status: kubectl describe node <node>
  2. Check cloud provider console
  3. If SSH available: check kubelet, containerd, disk, memory
  4. If not recoverable: cordon + drain → delete node
  5. Cluster autoscaler or Karpenter replaces automatically

High API server latency:
  1. Check etcd: etcd_request_duration_seconds
  2. Check API server: apiserver_request_duration_seconds
  3. Check webhook latency (mutating/validating)
  4. Check resource count: kubectl get all -A | wc -l
  5. Consider: etcd defrag, increase API server resources

Certificate expiration:
  1. Check: kubeadm certs check-expiration
  2. Renew: kubeadm certs renew all
  3. Restart control plane components
  4. Distribute new admin kubeconfig

Persistent volume full:
  1. Identify: kubectl get pvc -A | grep -v Bound
  2. Check usage: exec into pod → df -h
  3. Expand PVC (if StorageClass allows)
  4. Clean up data if possible
  5. Alert: set PVC usage monitoring

OOM cascade:
  1. Identify OOMKilled pods: kubectl get events | grep OOMKill
  2. Check node memory pressure
  3. Increase limits or right-size pods
  4. Check for memory leaks in application
  5. Consider node-level memory reservation
` + "```" + `

**Governance and Compliance:**
` + "```" + `
Policy enforcement tools:
  Kyverno:
    - Kubernetes-native policies
    - Validate, mutate, generate resources
    - ClusterPolicy and Policy resources
    
  OPA Gatekeeper:
    - Open Policy Agent for Kubernetes
    - ConstraintTemplate + Constraint
    - Rego language for policy rules

Common policies:
  - Require labels (team, app, environment)
  - Require resource requests/limits
  - Restrict image registries (allow list)
  - Require non-root containers
  - Forbid latest tag
  - Require probes
  - Limit LoadBalancer services
  - Require NetworkPolicies per namespace
  
Audit logging:
  - Enable API server audit logging
  - Log who did what when
  - Ship to SIEM (Splunk, ELK, etc.)
  - Alert on sensitive operations (secret access, exec)
` + "```" + ``,
					CodeExamples: `# Production Operational Configuration

# 1. Kyverno policies for production cluster
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-labels
spec:
  validationFailureAction: Enforce
  background: true
  rules:
  - name: check-required-labels
    match:
      any:
      - resources:
          kinds:
          - Deployment
          - StatefulSet
          - DaemonSet
    validate:
      message: "Labels 'app', 'team', and 'env' are required"
      pattern:
        metadata:
          labels:
            app: "?*"
            team: "?*"
            env: "?*"

---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resources
spec:
  validationFailureAction: Enforce
  rules:
  - name: check-container-resources
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "All containers must have resource requests and limits"
      pattern:
        spec:
          containers:
          - resources:
              requests:
                cpu: "?*"
                memory: "?*"
              limits:
                memory: "?*"

---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: restrict-image-registries
spec:
  validationFailureAction: Enforce
  rules:
  - name: allowed-registries
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Images must be from approved registries"
      pattern:
        spec:
          containers:
          - image: "myregistry.azurecr.io/* | gcr.io/myproject/* | docker.io/library/*"
          initContainers:
          - image: "myregistry.azurecr.io/* | gcr.io/myproject/* | docker.io/library/*"

---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-latest-tag
spec:
  validationFailureAction: Enforce
  rules:
  - name: no-latest
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Using ':latest' tag is not allowed"
      pattern:
        spec:
          containers:
          - image: "!*:latest"

---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-probes
spec:
  validationFailureAction: Audit  # Warn first, enforce later
  rules:
  - name: check-probes
    match:
      any:
      - resources:
          kinds:
          - Deployment
    validate:
      message: "Liveness and readiness probes are required"
      pattern:
        spec:
          template:
            spec:
              containers:
              - livenessProbe:
                  httpGet:
                    path: "?*"
                readinessProbe:
                  httpGet:
                    path: "?*"

---
# 2. API Server audit policy
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
# Don't log read-only requests to certain resources
- level: None
  resources:
  - group: ""
    resources: ["events"]
# Log secret access at metadata level
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets"]
# Log all write operations at request level
- level: Request
  verbs: ["create", "update", "patch", "delete"]
# Log everything else at metadata
- level: Metadata

---
# 3. Cost optimization labels
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
  labels:
    app: myapp
    team: platform
    env: production
    cost-center: "CC-1234"
    # Cloud provider cost allocation tags
  annotations:
    cost.example.com/monthly-budget: "500"
    cost.example.com/owner: "platform-team@example.com"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        team: platform
        env: production
    spec:
      # Use spot-friendly configuration
      terminationGracePeriodSeconds: 30
      containers:
      - name: app
        image: myregistry/myapp:v2.0.0
        resources:
          requests:
            cpu: 250m    # Right-sized from VPA recommendation
            memory: 512Mi
          limits:
            memory: 512Mi  # No CPU limit for efficiency
      tolerations:
      - key: karpenter.sh/capacity-type
        operator: Equal
        value: spot
        effect: NoSchedule`,
				},
			},
		},
	})
}
