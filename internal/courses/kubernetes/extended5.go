package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1723,
			Title:       "Storage and StatefulSets",
			Description: "Master Kubernetes persistent storage: PV/PVC, StorageClasses, CSI drivers, StatefulSets, and data management patterns.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Persistent Volumes and Storage Classes",
					Content: `Kubernetes storage abstracts physical storage into PersistentVolumes (PV) and PersistentVolumeClaims (PVC) with StorageClasses for dynamic provisioning.

**Storage Concepts:**
` + "```" + `
Volume types:
  emptyDir           → ephemeral, shares data between containers in a pod
  hostPath           → node filesystem mount (testing only!)
  configMap, secret  → mount config/secrets as files
  persistentVolumeClaim → durable storage that survives pod restarts

Storage abstraction layers:
  PersistentVolume (PV)    → actual storage resource (disk, NFS, etc.)
  PersistentVolumeClaim (PVC) → request for storage by a pod
  StorageClass             → template for dynamic PV provisioning
  CSI Driver               → plugin that talks to storage backend

Provisioning:
  Static:  Admin creates PV manually, user creates PVC to bind
  Dynamic: User creates PVC with StorageClass, PV auto-created

Access Modes:
  ReadWriteOnce (RWO)  → single node read-write (most block storage)
  ReadOnlyMany (ROX)   → multiple nodes read-only
  ReadWriteMany (RWX)  → multiple nodes read-write (NFS, EFS, CephFS)
  ReadWriteOncePod     → single pod read-write (K8s 1.27+)

Reclaim Policies:
  Retain → keep PV after PVC deletion (manual cleanup)
  Delete → delete PV and underlying storage (default for dynamic)
  Recycle → deprecated
` + "```" + `

**StorageClass:**
` + "```" + `yaml
# AWS EBS StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
  kmsKeyId: arn:aws:kms:us-east-1:123456:key/abcd-1234
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer  # Bind to first pod's zone

---
# GCP PD StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ssd
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd
  replication-type: regional-pd  # Regional for HA
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
# NFS StorageClass (ReadWriteMany)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs
provisioner: nfs.csi.k8s.io
parameters:
  server: nfs.example.com
  share: /exports
reclaimPolicy: Delete
volumeBindingMode: Immediate
mountOptions:
  - nfsvers=4.1
  - hard
  - timeo=600
` + "```" + `

**PVC and Pod Usage:**
` + "```" + `yaml
# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-volume
  namespace: production
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: gp3
  resources:
    requests:
      storage: 50Gi

---
# Pod using PVC
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: app
    image: myapp:v1
    volumeMounts:
    - name: data
      mountPath: /data
    - name: cache
      mountPath: /cache
    - name: config
      mountPath: /etc/app
      readOnly: true
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: data-volume
  - name: cache
    emptyDir:
      sizeLimit: 1Gi
  - name: config
    configMap:
      name: app-config
` + "```" + `

**Volume Expansion:**
` + "```" + `
# StorageClass must have: allowVolumeExpansion: true

# Expand PVC:
kubectl patch pvc data-volume -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# For file system expansion (most CSI):
#   - Online: Automatically resizes without pod restart
#   - Offline: Requires pod restart (older drivers)

# Check expansion status:
kubectl get pvc data-volume -o jsonpath='{.status.conditions}'
# condition: FileSystemResizePending → waiting for pod restart
# condition: Resizing → in progress
` + "```" + `

**Volume Snapshots:**
` + "```" + `yaml
# VolumeSnapshotClass
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshotClass
metadata:
  name: ebs-snap
driver: ebs.csi.aws.com
deletionPolicy: Delete

---
# Create snapshot
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot-2024-01
spec:
  volumeSnapshotClassName: ebs-snap
  source:
    persistentVolumeClaimName: data-volume

---
# Restore from snapshot
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-restored
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: gp3
  resources:
    requests:
      storage: 50Gi
  dataSource:
    name: data-snapshot-2024-01
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
` + "```" + ``,
					CodeExamples: `# Storage Configuration Examples

# 1. Multi-tier storage setup
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
provisioner: ebs.csi.aws.com
parameters:
  type: io2
  iops: "10000"
  encrypted: "true"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: cold-storage
provisioner: ebs.csi.aws.com
parameters:
  type: sc1
  encrypted: "true"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
# 2. Static PV (pre-provisioned disk)
apiVersion: v1
kind: PersistentVolume
metadata:
  name: legacy-data-pv
spec:
  capacity:
    storage: 500Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""  # Empty = no dynamic provisioning
  csi:
    driver: ebs.csi.aws.com
    volumeHandle: vol-0abcdef1234567890
    fsType: ext4
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: topology.ebs.csi.aws.com/zone
          operator: In
          values:
          - us-east-1a

---
# PVC bound to static PV
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: legacy-data
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: ""
  resources:
    requests:
      storage: 500Gi
  volumeName: legacy-data-pv  # Explicit binding

---
# 3. Shared volume (RWX) for multi-pod access
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-uploads
  namespace: production
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: nfs
  resources:
    requests:
      storage: 100Gi

---
# Multiple pods sharing the volume
apiVersion: apps/v1
kind: Deployment
metadata:
  name: upload-processor
  namespace: production
spec:
  replicas: 5
  selector:
    matchLabels:
      app: upload-processor
  template:
    metadata:
      labels:
        app: upload-processor
    spec:
      containers:
      - name: processor
        image: myapp/processor:v1
        volumeMounts:
        - name: uploads
          mountPath: /data/uploads
      volumes:
      - name: uploads
        persistentVolumeClaim:
          claimName: shared-uploads

---
# 4. Ephemeral volume with size limit
apiVersion: v1
kind: Pod
metadata:
  name: data-processor
spec:
  containers:
  - name: main
    image: myapp/processor:v1
    volumeMounts:
    - name: scratch
      mountPath: /scratch
    - name: cache
      mountPath: /cache
    resources:
      limits:
        ephemeral-storage: 5Gi  # Total ephemeral limit
  volumes:
  - name: scratch
    emptyDir:
      sizeLimit: 2Gi
  - name: cache
    emptyDir:
      medium: Memory  # tmpfs (RAM-backed)
      sizeLimit: 512Mi`,
				},
				{
					Title: "StatefulSets and Stateful Workloads",
					Content: `StatefulSets manage stateful applications that need stable identity, ordered deployment, and persistent storage per pod.

**StatefulSet vs Deployment:**
` + "```" + `
Deployment:
  ✓ Pods are interchangeable (same identity)
  ✓ Random pod names (deploy-abc123)
  ✓ Parallel creation/deletion
  ✓ Shared storage (or no storage)
  ✓ Use for: stateless web apps, APIs, workers

StatefulSet:
  ✓ Stable, unique pod names (sts-0, sts-1, sts-2)
  ✓ Stable network identities (headless service)
  ✓ Ordered, graceful deployment and scaling
  ✓ Per-pod persistent storage (volumeClaimTemplates)
  ✓ Use for: databases, message queues, distributed systems

Guarantees:
  Ordered creation:  pod-0 → pod-1 → pod-2
  Ordered deletion:  pod-2 → pod-1 → pod-0
  Ordered updates:   pod-2 → pod-1 → pod-0 (reverse)
  Storage stability:  PVC "data-sts-0" always binds to pod "sts-0"
` + "```" + `

**StatefulSet Configuration:**
` + "```" + `yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: production
spec:
  serviceName: postgres-headless  # Required: headless service name
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 0          # Update pods >= partition index
      maxUnavailable: 1     # K8s 1.24+
  podManagementPolicy: OrderedReady  # or Parallel
  minReadySeconds: 30
  revisionHistoryLimit: 10
  template:
    metadata:
      labels:
        app: postgres
    spec:
      terminationGracePeriodSeconds: 120
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
      - name: postgres
        image: postgres:16-alpine
        ports:
        - containerPort: 5432
          name: postgresql
        env:
        - name: POSTGRES_DB
          value: myapp
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        resources:
          limits:
            cpu: "2"
            memory: 4Gi
          requests:
            cpu: "500m"
            memory: 2Gi
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        - name: config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        livenessProbe:
          exec:
            command: ["pg_isready", "-U", "postgres"]
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command: ["pg_isready", "-U", "postgres"]
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: postgres-config
  # volumeClaimTemplates create PVCs per pod
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi

---
# Headless Service (required for StatefulSet)
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
  namespace: production
spec:
  type: ClusterIP
  clusterIP: None  # Headless!
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: postgresql
` + "```" + `

**DNS and Networking:**
` + "```" + `
Headless Service creates DNS records:
  postgres-headless.production.svc.cluster.local → all pod IPs
  
Individual pod DNS:
  postgres-0.postgres-headless.production.svc.cluster.local
  postgres-1.postgres-headless.production.svc.cluster.local
  postgres-2.postgres-headless.production.svc.cluster.local

Use cases:
  Primary:  postgres-0.postgres-headless.production.svc.cluster.local
  Replicas: postgres-1, postgres-2 (for reads)
  
  Connection string example:
  postgresql://user:pass@postgres-0.postgres-headless:5432/myapp
` + "```" + `

**Scaling and Update Patterns:**
` + "```" + `
Scaling:
  kubectl scale statefulset postgres --replicas=5
  # Creates pods 3, 4 (ordered)
  # PVCs: data-postgres-3, data-postgres-4 created
  
  kubectl scale statefulset postgres --replicas=2
  # Deletes pods 4, 3, 2 (reverse order)
  # PVCs: data-postgres-2, data-postgres-3, data-postgres-4 RETAINED!
  # Must manually delete orphaned PVCs if storage no longer needed

Canary update (partition):
  spec:
    updateStrategy:
      type: RollingUpdate
      rollingUpdate:
        partition: 2  # Only pods >= 2 get updated
  
  # Useful for canary testing on higher ordinals:
  # partition: 2 → only pod-2 updated, pod-0 and pod-1 stay
  # partition: 1 → pods 1 and 2 updated
  # partition: 0 → all pods updated

Parallel pod management:
  spec:
    podManagementPolicy: Parallel
  # Creates/deletes all pods simultaneously
  # Use when order doesn't matter (e.g., Cassandra, CockroachDB)
` + "```" + ``,
					CodeExamples: `# StatefulSet Examples

# 1. Redis Cluster StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: production
spec:
  serviceName: redis-cluster-headless
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  podManagementPolicy: Parallel
  updateStrategy:
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      terminationGracePeriodSeconds: 60
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        command: ["redis-server"]
        args:
        - /etc/redis/redis.conf
        - --cluster-enabled yes
        - --cluster-config-file /data/nodes.conf
        - --cluster-node-timeout 5000
        resources:
          limits:
            cpu: "1"
            memory: 2Gi
          requests:
            cpu: 250m
            memory: 1Gi
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /etc/redis
        readinessProbe:
          exec:
            command: ["redis-cli", "ping"]
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          exec:
            command: ["redis-cli", "ping"]
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-cluster-headless
  namespace: production
spec:
  clusterIP: None
  selector:
    app: redis-cluster
  ports:
  - port: 6379
    targetPort: client
    name: client
  - port: 16379
    targetPort: gossip
    name: gossip

---
# 2. Elasticsearch StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
  namespace: logging
spec:
  serviceName: elasticsearch-headless
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  podManagementPolicy: Parallel
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      terminationGracePeriodSeconds: 120
      initContainers:
      - name: sysctl
        image: busybox:1.36
        securityContext:
          privileged: true
        command: ['sh', '-c', 'sysctl -w vm.max_map_count=262144']
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
        ports:
        - containerPort: 9200
          name: http
        - containerPort: 9300
          name: transport
        env:
        - name: node.name
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: cluster.name
          value: "logging"
        - name: discovery.seed_hosts
          value: "elasticsearch-headless"
        - name: cluster.initial_master_nodes
          value: "elasticsearch-0,elasticsearch-1,elasticsearch-2"
        - name: ES_JAVA_OPTS
          value: "-Xms2g -Xmx2g"
        - name: xpack.security.enabled
          value: "false"
        resources:
          limits:
            cpu: "2"
            memory: 4Gi
          requests:
            cpu: "1"
            memory: 3Gi
        volumeMounts:
        - name: data
          mountPath: /usr/share/elasticsearch/data
        readinessProbe:
          httpGet:
            path: /_cluster/health?local=true
            port: 9200
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /_cluster/health?local=true
            port: 9200
          initialDelaySeconds: 90
          periodSeconds: 30
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 200Gi
---
apiVersion: v1
kind: Service
metadata:
  name: elasticsearch-headless
  namespace: logging
spec:
  clusterIP: None
  selector:
    app: elasticsearch
  ports:
  - port: 9200
    name: http
  - port: 9300
    name: transport
---
apiVersion: v1
kind: Service
metadata:
  name: elasticsearch
  namespace: logging
spec:
  selector:
    app: elasticsearch
  ports:
  - port: 9200
    name: http

---
# 3. PodDisruptionBudget for StatefulSets
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: postgres-pdb
  namespace: production
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: postgres

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: redis-cluster-pdb
  namespace: production
spec:
  minAvailable: 4  # At least 4 of 6 nodes must be available
  selector:
    matchLabels:
      app: redis-cluster`,
				},
			},
		},
	})
}
