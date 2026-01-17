package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          65,
			Title:       "ConfigMaps & Secrets",
			Description: "Manage configuration data and sensitive information in Kubernetes.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "ConfigMaps",
					Content: `**ConfigMaps** store non-confidential configuration data in key-value pairs. They decouple configuration from container images.

**Use Cases:**
- Configuration files
- Environment variables
- Command-line arguments
- Volume data

**Creating ConfigMaps:**
- **From literal values**: kubectl create configmap
- **From files**: kubectl create configmap --from-file
- **From YAML**: Define in manifest file

**Using ConfigMaps:**
- **Environment variables**: Inject as env vars
- **Volume mounts**: Mount as files
- **Command-line arguments**: Use in command/args

**ConfigMap Data:**
- **data**: UTF-8 strings
- **binaryData**: Binary data (base64 encoded)

**Best Practices:**
- Keep ConfigMaps small (< 1MB)
- Use separate ConfigMaps for different concerns
- Version control ConfigMaps
- Don't store sensitive data (use Secrets)`,
					CodeExamples: `# Create ConfigMap from literal
kubectl create configmap app-config \
  --from-literal=database_url=postgresql://localhost/mydb \
  --from-literal=api_key=abc123

# Create ConfigMap from file
kubectl create configmap app-config --from-file=config.properties

# Create ConfigMap from YAML
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://localhost/mydb"
  log_level: "info"
  config.properties: |
    server.port=8080
    server.host=0.0.0.0

# Use ConfigMap as environment variables
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:1.0
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: database_url
    envFrom:
    - configMapRef:
        name: app-config

# Mount ConfigMap as volume
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:1.0
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
  volumes:
  - name: config-volume
    configMap:
      name: app-config

# View ConfigMaps
kubectl get configmaps
kubectl describe configmap app-config`,
				},
				{
					Title: "Secrets",
					Content: `**Secrets** store sensitive information like passwords, tokens, and keys. They're similar to ConfigMaps but designed for sensitive data.

**Secret Types:**
- **Opaque**: Arbitrary user-defined data (default)
- **kubernetes.io/dockerconfigjson**: Docker registry credentials
- **kubernetes.io/tls**: TLS certificate and key
- **kubernetes.io/basic-auth**: Basic authentication
- **kubernetes.io/ssh-auth**: SSH key

**Secret Storage:**
- Stored in etcd (encrypted at rest in newer versions)
- Base64 encoded (not encrypted!)
- Should be encrypted at rest and in transit
- Access controlled via RBAC

**Creating Secrets:**
- **From literal**: kubectl create secret
- **From files**: kubectl create secret --from-file
- **From YAML**: Base64 encode values first

**Using Secrets:**
- Similar to ConfigMaps
- Can be mounted as volumes
- Can be used as environment variables
- Automatically decoded when used

**Best Practices:**
- Use external secret management (Vault, AWS Secrets Manager)
- Rotate secrets regularly
- Use least privilege access
- Don't commit secrets to version control
- Use sealed-secrets for GitOps`,
					CodeExamples: `# Create secret from literal
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secret123

# Create secret from file
kubectl create secret generic db-secret \
  --from-file=username=./username.txt \
  --from-file=password=./password.txt

# Create TLS secret
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key

# Create Docker registry secret
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=username \
  --docker-password=password \
  --docker-email=email@example.com

# Secret YAML (base64 encoded)
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  username: YWRtaW4=  # base64 encoded
  password: c2VjcmV0MTIz  # base64 encoded

# Use secret as environment variable
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:1.0
    env:
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password

# Mount secret as volume
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:1.0
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true
  volumes:
  - name: secret-volume
    secret:
      secretName: db-secret

# View secrets
kubectl get secrets
kubectl describe secret db-secret

# Decode secret value
echo "YWRtaW4=" | base64 -d`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          66,
			Title:       "Volumes & Storage",
			Description: "Understand persistent storage, volume types, and dynamic provisioning.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Volume Types",
					Content: `**Volumes** provide persistent storage for pods. Unlike container filesystem, volumes persist across container restarts.

**Volume Types:**

**emptyDir:**
- Temporary storage
- Created when pod starts
- Deleted when pod terminates
- Shared between containers in pod
- Uses node's storage

**hostPath:**
- Mounts file/directory from host node
- Useful for system-level access
- Not portable across nodes
- Security risk (host access)
- Use with caution

**PersistentVolume (PV):**
- Cluster-wide storage resource
- Provisioned by administrator
- Independent of pod lifecycle
- Can be statically or dynamically provisioned

**PersistentVolumeClaim (PVC):**
- Request for storage
- User's view of storage
- Binds to PersistentVolume
- Pods use PVCs, not PVs directly

**Storage Classes:**
- Defines storage "classes"
- Enables dynamic provisioning
- Maps to storage backends
- Allows different storage types`,
					CodeExamples: `# Pod with emptyDir volume
apiVersion: v1
kind: Pod
metadata:
  name: volume-pod
spec:
  containers:
  - name: app
    image: nginx:1.21
    volumeMounts:
    - name: cache-volume
      mountPath: /cache
  volumes:
  - name: cache-volume
    emptyDir:
      sizeLimit: 500Mi

# Pod with hostPath volume
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-pod
spec:
  containers:
  - name: app
    image: nginx:1.21
    volumeMounts:
    - name: host-storage
      mountPath: /host-data
  volumes:
  - name: host-storage
    hostPath:
      path: /data
      type: DirectoryOrCreate

# PersistentVolume
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-example
spec:
  capacity:
    storage: 10Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: slow
  hostPath:
    path: /mnt/data

# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-example
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: slow

# Pod using PVC
apiVersion: v1
kind: Pod
metadata:
  name: pvc-pod
spec:
  containers:
  - name: app
    image: nginx:1.21
    volumeMounts:
    - name: pvc-storage
      mountPath: /data
  volumes:
  - name: pvc-storage
    persistentVolumeClaim:
      claimName: pvc-example`,
				},
				{
					Title: "PersistentVolumes and PersistentVolumeClaims",
					Content: `**PersistentVolumes (PVs)** are cluster resources provisioned by administrators. **PersistentVolumeClaims (PVCs)** are requests for storage by users.

**PV Lifecycle:**
1. **Provisioning**: PV created (static or dynamic)
2. **Binding**: PVC binds to PV
3. **Using**: Pod uses PVC
4. **Releasing**: Pod deleted, PVC released
5. **Reclaiming**: PV reclaimed based on policy

**Access Modes:**
- **ReadWriteOnce (RWO)**: Single node read-write
- **ReadOnlyMany (ROX)**: Multiple nodes read-only
- **ReadWriteMany (RWX)**: Multiple nodes read-write
- **ReadWriteOncePod (RWOP)**: Single pod read-write

**Reclaim Policies:**
- **Retain**: Manual reclamation (keep data)
- **Recycle**: Basic scrub (deprecated)
- **Delete**: Automatically delete PV and storage

**Binding:**
- PVC requests storage with specific requirements
- PV matches PVC requirements
- Binding is one-to-one (1 PVC → 1 PV)
- Best match algorithm selects PV

**Dynamic Provisioning:**
- StorageClass enables dynamic provisioning
- PV created automatically when PVC created
- No need to pre-create PVs
- Simplifies storage management`,
					CodeExamples: `# StorageClass for dynamic provisioning
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  fsType: ext4
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true

# PVC with StorageClass
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-dynamic
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 20Gi

# View PVs and PVCs
kubectl get pv
kubectl get pvc

# Describe PV/PVC
kubectl describe pv pv-example
kubectl describe pvc pvc-example

# Delete PVC (triggers reclaim)
kubectl delete pvc pvc-example`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          67,
			Title:       "StatefulSets & DaemonSets",
			Description: "Learn specialized workloads for stateful applications and node-level pods.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "StatefulSets",
					Content: `**StatefulSets** manage stateful applications. They provide stable network identities and ordered deployment/scaling.

**Key Features:**
- **Stable Network Identity**: Each pod gets stable hostname
- **Stable Storage**: Each pod gets persistent storage
- **Ordered Deployment**: Pods created in order
- **Ordered Scaling**: Pods scaled in order
- **Ordered Updates**: Pods updated in reverse order

**Use Cases:**
- Databases (MySQL, PostgreSQL, MongoDB)
- Message queues (RabbitMQ, Kafka)
- Distributed systems requiring identity
- Applications needing stable storage

**Pod Identity:**
- Pod name: <statefulset-name>-<ordinal>
- Stable hostname: <pod-name>.<service-name>.<namespace>.svc.cluster.local
- Ordinal: 0, 1, 2, ...

**Headless Service:**
- Required for StatefulSet
- clusterIP: None
- Returns pod IPs directly
- Enables stable DNS names

**Volume Claims:**
- Each pod gets its own PVC
- PVC name: <volumeClaimTemplate-name>-<statefulset-name>-<ordinal>
- Persistent across pod restarts`,
					CodeExamples: `# Headless service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  clusterIP: None
  selector:
    app: mysql
  ports:
  - port: 3306

# StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: password
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi

# Scale StatefulSet
kubectl scale statefulset mysql --replicas=5

# View StatefulSet pods
kubectl get pods -l app=mysql

# Pods will be: mysql-0, mysql-1, mysql-2, mysql-3, mysql-4`,
				},
				{
					Title: "DaemonSets",
					Content: `**DaemonSets** ensure a copy of a pod runs on all (or specific) nodes. Useful for node-level services.

**Key Features:**
- **One Pod Per Node**: Ensures pod on every node
- **Node Selection**: Can target specific nodes
- **Automatic Scheduling**: New nodes get pod automatically
- **Node Removal**: Pods removed when nodes removed

**Use Cases:**
- **Logging Agents**: Fluentd, Fluent Bit
- **Monitoring Agents**: Node exporter, Datadog agent
- **Network Plugins**: CNI plugins
- **Storage Daemons**: GlusterFS, Ceph
- **Security Agents**: Falco, security scanners

**Node Selection:**
- **nodeSelector**: Simple node selection
- **affinity**: Advanced node selection
- **tolerations**: Allow scheduling on tainted nodes

**Update Strategy:**
- **OnDelete**: Update when pod deleted
- **RollingUpdate**: Rolling update (default)

**DaemonSet vs Deployment:**
- Deployment: N replicas anywhere
- DaemonSet: One pod per node`,
					CodeExamples: `# DaemonSet for logging agent
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
        image: fluent/fluentd:latest
        resources:
          limits:
            memory: 200Mi
          requests:
            memory: 200Mi
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

# DaemonSet with nodeSelector
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-monitor
spec:
  selector:
    matchLabels:
      app: gpu-monitor
  template:
    metadata:
      labels:
        app: gpu-monitor
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-k80
      containers:
      - name: gpu-monitor
        image: nvidia/cuda:latest

# View DaemonSets
kubectl get daemonsets
kubectl describe daemonset fluentd

# View DaemonSet pods
kubectl get pods -l app=fluentd`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          68,
			Title:       "Jobs & CronJobs",
			Description: "Run batch jobs and scheduled tasks in Kubernetes.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Jobs",
					Content: `**Jobs** create one or more pods and ensure they complete successfully. Used for batch processing and one-time tasks.

**Job Characteristics:**
- Run to completion (not continuously)
- Retry on failure
- Parallel execution support
- Completion tracking

**Job Types:**
- **Non-parallel**: Single pod runs until completion
- **Parallel with fixed completions**: Multiple pods, fixed number succeed
- **Parallel with work queue**: Multiple pods, process queue until empty

**Job Spec:**
- **completions**: Number of successful completions needed
- **parallelism**: Number of pods running concurrently
- **backoffLimit**: Number of retries before marking failed
- **activeDeadlineSeconds**: Maximum time job can run

**Completion:**
- Job completes when specified number of pods succeed
- Completed pods not deleted (for logs)
- Job remains for inspection
- Can be manually deleted

**Use Cases:**
- Database migrations
- Data processing
- Batch jobs
- One-time tasks`,
					CodeExamples: `# Simple Job
apiVersion: batch/v1
kind: Job
metadata:
  name: pi-job
spec:
  template:
    spec:
      containers:
      - name: pi
        image: perl:5.34
        command: ["perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"]
      restartPolicy: Never
  backoffLimit: 4

# Parallel Job with fixed completions
apiVersion: batch/v1
kind: Job
metadata:
  name: parallel-job
spec:
  completions: 5
  parallelism: 2
  template:
    spec:
      containers:
      - name: worker
        image: busybox
        command: ["sh", "-c", "echo Processing item; sleep 5"]
      restartPolicy: Never

# Job with work queue
apiVersion: batch/v1
kind: Job
metadata:
  name: queue-job
spec:
  parallelism: 3
  completions: 6
  template:
    spec:
      containers:
      - name: worker
        image: busybox
        command: ["sh", "-c", "process-queue-item"]
      restartPolicy: OnFailure

# View Jobs
kubectl get jobs
kubectl describe job pi-job

# View Job pods
kubectl get pods -l job-name=pi-job

# Delete Job (pods remain)
kubectl delete job pi-job`,
				},
				{
					Title: "CronJobs",
					Content: `**CronJobs** create Jobs on a time-based schedule. They run recurring tasks automatically.

**Cron Schedule:**
- Uses standard cron format
- Format: minute hour day month weekday
- Examples:
  - 0 * * * *: Every hour
  - 0 0 * * *: Every day at midnight
  - */5 * * * *: Every 5 minutes

**CronJob Spec:**
- **schedule**: Cron expression
- **jobTemplate**: Template for creating Jobs
- **concurrencyPolicy**: What to do with overlapping jobs
- **startingDeadlineSeconds**: Deadline for starting job
- **successfulJobsHistoryLimit**: Keep N successful jobs
- **failedJobsHistoryLimit**: Keep N failed jobs

**Concurrency Policies:**
- **Allow**: Allow concurrent jobs (default)
- **Forbid**: Skip new job if previous running
- **Replace**: Replace running job with new one

**Time Zones:**
- Uses controller's time zone
- Can specify time zone in schedule
- Format: CRON_TZ=UTC 0 0 * * *

**Use Cases:**
- Scheduled backups
- Periodic data sync
- Cleanup tasks
- Health checks
- Reports generation`,
					CodeExamples: `# CronJob - run every hour
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hourly-backup
spec:
  schedule: "0 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:latest
            command: ["/bin/sh", "-c", "backup-database"]
          restartPolicy: OnFailure
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1

# CronJob - run daily at midnight
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-report
spec:
  schedule: "0 0 * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: report
            image: report-generator:latest
            command: ["generate-report"]
          restartPolicy: Never

# CronJob with timezone
apiVersion: batch/v1
kind: CronJob
metadata:
  name: tz-job
spec:
  schedule: "CRON_TZ=America/New_York 0 9 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: job
            image: busybox
            command: ["echo", "9 AM EST"]

# View CronJobs
kubectl get cronjobs
kubectl describe cronjob hourly-backup

# View Jobs created by CronJob
kubectl get jobs

# Suspend CronJob
kubectl patch cronjob hourly-backup -p '{"spec":{"suspend":true}}'

# Resume CronJob
kubectl patch cronjob hourly-backup -p '{"spec":{"suspend":false}}'`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          69,
			Title:       "Ingress & Load Balancing",
			Description: "Expose applications externally with Ingress controllers and advanced routing.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Ingress Controllers",
					Content: `**Ingress** provides HTTP/HTTPS routing to services. It acts as a reverse proxy and load balancer.

**Ingress vs Service:**
- **Service**: L4 (TCP/UDP) load balancing
- **Ingress**: L7 (HTTP/HTTPS) routing
- **Ingress Controller**: Required to implement Ingress

**Ingress Controllers:**
- **NGINX Ingress**: Most popular, feature-rich
- **Traefik**: Modern, automatic configuration
- **HAProxy**: High performance
- **Istio Gateway**: Part of service mesh
- **AWS ALB**: AWS Application Load Balancer
- **GCE Ingress**: Google Cloud Load Balancer

**Ingress Resource:**
- Defines routing rules
- Maps domain/path to services
- Handles TLS termination
- Requires Ingress Controller

**Ingress Controller:**
- Pod that watches Ingress resources
- Implements routing rules
- Manages load balancer
- Handles SSL/TLS certificates

**Installation:**
- Deploy Ingress Controller as DaemonSet or Deployment
- Usually in kube-system namespace
- May require NodePort or LoadBalancer service`,
					CodeExamples: `# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Basic Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: basic-ingress
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80

# Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
spec:
  tls:
  - hosts:
    - example.com
    secretName: tls-secret
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80

# View Ingress
kubectl get ingress
kubectl describe ingress basic-ingress`,
				},
				{
					Title: "Ingress Rules and Routing",
					Content: `**Ingress Rules** define how traffic is routed to backend services based on host and path.

**Path Types:**
- **Exact**: Exact path match
- **Prefix**: Path prefix match
- **ImplementationSpecific**: Depends on controller

**Host-Based Routing:**
- Route based on HTTP Host header
- Multiple hosts in one Ingress
- Wildcard hosts supported
- Default backend for no match

**Path-Based Routing:**
- Route based on URL path
- Multiple paths in one Ingress
- Path rewriting supported
- Regex patterns (some controllers)

**Annotations:**
- Controller-specific configuration
- NGINX: nginx.ingress.kubernetes.io/rewrite-target
- Traefik: traefik.ingress.kubernetes.io/rule-type
- Customize behavior per Ingress

**Default Backend:**
- Handles requests with no matching rule
- Can be specified in Ingress
- Useful for 404 pages`,
					CodeExamples: `# Ingress with multiple hosts
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multi-host-ingress
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
  - host: www.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80

# Ingress with path-based routing
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: path-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
      - path: /admin
        pathType: Prefix
        backend:
          service:
            name: admin-service
            port:
              number: 80

# Ingress with default backend
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: default-backend-ingress
spec:
  defaultBackend:
    service:
      name: default-service
      port:
        number: 80
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80`,
				},
				{
					Title: "TLS/SSL Termination",
					Content: `**TLS Termination** decrypts HTTPS traffic at Ingress and forwards HTTP to backend services.

**TLS Configuration:**
- **TLS Secret**: Contains certificate and key
- **Hosts**: List of hosts for certificate
- **Automatic TLS**: Using cert-manager

**Creating TLS Secrets:**
- From existing certificate/key files
- Using cert-manager (automatic)
- Using Let's Encrypt (free certificates)

**cert-manager:**
- Automatically manages certificates
- Integrates with Let's Encrypt
- Auto-renewal before expiration
- Supports multiple issuers

**Certificate Issuers:**
- **Let's Encrypt**: Free, automatic certificates
- **Self-signed**: For development
- **Custom CA**: Enterprise certificates

**Best Practices:**
- Use cert-manager for production
- Enable automatic renewal
- Use strong cipher suites
- Redirect HTTP to HTTPS`,
					CodeExamples: `# Create TLS secret manually
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key

# Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
spec:
  tls:
  - hosts:
    - example.com
    - www.example.com
    secretName: tls-secret
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

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
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx

# Certificate resource
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: example-com-cert
spec:
  secretName: tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - example.com
  - www.example.com

# Ingress with cert-manager annotation
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cert-manager-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - example.com
    secretName: tls-secret
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
