package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          60,
			Title:       "Introduction to Kubernetes",
			Description: "Learn what Kubernetes is, why it matters, and how to get started with container orchestration.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Kubernetes?",
					Content: `Kubernetes (often abbreviated as K8s) is an open-source container orchestration platform originally developed by Google. It automates the deployment, scaling, and management of containerized applications.

**History:**
- Originally developed at Google based on their internal Borg system
- Open-sourced in 2014
- Donated to Cloud Native Computing Foundation (CNCF) in 2015
- Now one of the most popular container orchestration platforms

**Key Capabilities:**
- **Service Discovery & Load Balancing**: Automatically expose containers using DNS or IP
- **Storage Orchestration**: Mount storage systems (local, cloud, network)
- **Automated Rollouts & Rollbacks**: Gradually update application instances
- **Automatic Bin Packing**: Place containers based on resource requirements
- **Self-Healing**: Restart failed containers, replace and reschedule containers
- **Secret & Configuration Management**: Store and manage sensitive information

**Why Use Kubernetes?**
- **Scalability**: Scale applications up or down automatically
- **High Availability**: Distribute workloads across multiple nodes
- **Portability**: Run anywhere (cloud, on-premises, hybrid)
- **Ecosystem**: Rich ecosystem of tools and extensions
- **Industry Standard**: Widely adopted by enterprises

**Kubernetes vs Alternatives:**
- **Docker Swarm**: Simpler but less feature-rich
- **Apache Mesos**: More complex, better for heterogeneous workloads
- **Nomad**: Simpler, good for multi-cloud deployments
- **Kubernetes**: Most features, largest ecosystem, industry standard

**Common Use Cases:**
- Microservices architectures
- CI/CD pipelines
- Multi-cloud deployments
- Machine learning workloads
- Edge computing`,
					CodeExamples: `# Check Kubernetes version
kubectl version --client

# Get cluster information
kubectl cluster-info

# View nodes in the cluster
kubectl get nodes

# Kubernetes architecture overview
# Control Plane (Master):
#   - API Server: Entry point for all operations
#   - etcd: Distributed key-value store
#   - Scheduler: Assigns pods to nodes
#   - Controller Manager: Runs controllers
#
# Worker Nodes:
#   - kubelet: Agent that communicates with control plane
#   - kube-proxy: Network proxy for services
#   - Container Runtime: Runs containers (Docker, containerd, CRI-O)

# Basic kubectl commands
kubectl get pods                    # List pods
kubectl get services               # List services
kubectl get deployments            # List deployments
kubectl get nodes                  # List nodes
kubectl describe pod <pod-name>    # Detailed pod information`,
				},
				{
					Title: "Container Orchestration Concepts",
					Content: `**Container Orchestration** is the automated management of containerized applications, including deployment, scaling, networking, and availability.

**Why Container Orchestration?**

**Challenges Without Orchestration:**
- Manual container management is error-prone
- Difficult to scale applications
- No automatic recovery from failures
- Complex networking between containers
- Resource management is manual
- Rolling updates are complex

**Benefits of Orchestration:**
- Automated deployment and scaling
- Self-healing (restarts failed containers)
- Load balancing and service discovery
- Rolling updates and rollbacks
- Resource management and limits
- Health monitoring and logging

**Key Concepts:**

**Pods:**
- Smallest deployable unit in Kubernetes
- Contains one or more containers
- Share network and storage namespaces
- Ephemeral (can be created/destroyed)
- Each pod gets unique IP address
- Containers in pod share localhost
- **Use cases**: Co-located containers (app + sidecar), tight coupling
- **Best practice**: One container per pod usually, unless containers are tightly coupled

**Services:**
- Stable network endpoint for pods
- Provides service discovery (DNS)
- Load balances traffic to pods
- Abstracts pod IP addresses (pods are ephemeral)
- Types: ClusterIP, NodePort, LoadBalancer, ExternalName
- **Use cases**: Expose pods to other pods or external traffic
- **Best practice**: Always use services, never access pods directly

**Deployments:**
- Manages ReplicaSets and Pods
- Declarative updates (describe desired state)
- Rolling updates and rollbacks
- Desired state management (reconciliation loop)
- **Use cases**: Stateless applications, web servers, APIs
- **Best practice**: Use Deployments for most stateless workloads

**ReplicaSets:**
- Ensures specified number of pod replicas
- Created and managed by Deployments
- Can be used directly (rare)
- **Use cases**: Ensuring pod availability
- **Best practice**: Let Deployments manage ReplicaSets

**Namespaces:**
- Virtual clusters within a physical cluster
- Resource isolation (not security isolation)
- Access control boundaries (RBAC)
- Organization and management
- Default namespaces: default, kube-system, kube-public, kube-node-lease
- **Use cases**: Multi-tenant clusters, environment separation (dev/staging/prod)
- **Best practice**: Use namespaces for organization, not security

**Labels & Selectors:**
- Key-value pairs attached to objects
- Used for identification and selection
- Enable flexible organization
- Core to Kubernetes operations (how objects find each other)
- **Examples**: app=frontend, env=production, version=v1.2.3
- **Best practice**: Use consistent labeling strategy

**Controllers:**
- Control loops that watch cluster state
- Make changes to move current state to desired state
- Examples: Deployment, ReplicaSet, StatefulSet, DaemonSet, Job, CronJob
- **How they work**: Watch API server, compare desired vs actual, make changes
- **Best practice**: Use appropriate controller for workload type

**Declarative vs Imperative:**

**Declarative (Preferred):**
- Describe desired state (YAML files)
- Kubernetes figures out how to achieve it
- Idempotent (can apply multiple times)
- Version controlled
- Example: kubectl apply -f deployment.yaml

**Imperative:**
- Give commands to change state (kubectl commands)
- Useful for quick testing/debugging
- Not idempotent
- Example: kubectl create deployment nginx --image=nginx

**Best Practice:** Use declarative approach for production, imperative for quick tests

**Common Pitfalls:**

**1. Not Using Services:**
- Accessing pods directly by IP
- Pods are ephemeral, IPs change
- Solution: Always use Services

**2. Wrong Controller Choice:**
- Using Deployment for stateful workloads
- Using StatefulSet for stateless workloads
- Solution: Understand workload requirements

**3. Ignoring Resource Limits:**
- Not setting requests/limits
- Can cause resource exhaustion
- Solution: Always set resource requests and limits

**4. Not Using Labels:**
- Hard to organize and select resources
- Difficult to manage at scale
- Solution: Use consistent labeling strategy

**5. Mixing Declarative and Imperative:**
- Applying YAML then using imperative commands
- Can cause conflicts
- Solution: Stick to one approach

**Best Practices:**

**1. Use Declarative Configuration:**
- Store YAML in version control
- Use kubectl apply, not create
- Review changes before applying

**2. Set Resource Limits:**
- Define requests and limits
- Prevent resource exhaustion
- Enable better scheduling

**3. Use Appropriate Controllers:**
- Deployment for stateless
- StatefulSet for stateful
- DaemonSet for node-level services
- Job/CronJob for batch work

**4. Implement Health Checks:**
- Liveness probes (restart unhealthy)
- Readiness probes (traffic routing)
- Startup probes (slow-starting apps)

**5. Use Namespaces:**
- Organize resources
- Separate environments
- Apply RBAC policies`,
					CodeExamples: `# Declarative approach (preferred)
# Create a deployment from YAML
kubectl apply -f deployment.yaml

# Imperative approach
# Create deployment directly
kubectl create deployment nginx --image=nginx

# View current state
kubectl get pods -o wide

# Describe desired state in YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80`,
				},
				{
					Title: "Installation and Setup",
					Content: `**Local Development Options:**

**minikube:**
- Single-node Kubernetes cluster
- Runs in a VM on your machine
- Great for learning and development
- Easy to install and use

**k3s:**
- Lightweight Kubernetes distribution
- Single binary, < 50MB
- Great for edge computing
- Production-ready

**kind (Kubernetes in Docker):**
- Runs Kubernetes in Docker containers
- Fast startup
- Good for CI/CD testing
- Multiple clusters support

**kubeadm:**
- Production-grade installation tool
- Creates clusters with best practices
- Used for multi-node clusters
- More complex setup

**Cloud Options:**
- **EKS** (Amazon Elastic Kubernetes Service)
- **GKE** (Google Kubernetes Engine)
- **AKS** (Azure Kubernetes Service)
- **DigitalOcean Kubernetes**
- **Linode Kubernetes Engine**

**Installation Steps (minikube):**
1. Install kubectl (Kubernetes command-line tool)
2. Install minikube
3. Start minikube cluster
4. Verify installation

**Prerequisites:**
- Virtualization support (VirtualBox, Hyper-V, or KVM)
- 2+ CPUs
- 2GB+ free memory
- 20GB+ free disk space`,
					CodeExamples: `# Install kubectl (macOS)
brew install kubectl

# Install kubectl (Linux)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install minikube (macOS)
brew install minikube

# Install minikube (Linux)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start minikube cluster
minikube start

# Check cluster status
minikube status

# View cluster information
kubectl cluster-info

# Stop minikube
minikube stop

# Delete minikube cluster
minikube delete

# Install k3s (single command)
curl -sfL https://get.k3s.io | sh -

# Check k3s status
sudo systemctl status k3s

# Get kubeconfig for k3s
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          61,
			Title:       "Kubernetes Architecture",
			Description: "Understand Kubernetes cluster architecture, control plane components, and node components.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Control Plane Components",
					Content: `The **Control Plane** (also called Master) is the brain of Kubernetes. It manages the cluster and makes decisions about the cluster.

**API Server (kube-apiserver):**
- Central management entity
- Exposes Kubernetes API
- Validates and processes requests
- Only component that talks to etcd
- Can be horizontally scaled

**etcd:**
- Distributed key-value store
- Stores cluster state and configuration
- Highly available and consistent
- Backup critical for disaster recovery
- Only API server can write to etcd

**Scheduler (kube-scheduler):**
- Assigns pods to nodes
- Considers resource requirements
- Considers constraints and policies
- Watches for newly created pods
- Makes placement decisions

**Controller Manager (kube-controller-manager):**
- Runs controller processes
- Node controller, Replication controller, etc.
- Watches cluster state
- Makes changes to move to desired state
- Multiple controllers in one process

**Cloud Controller Manager:**
- Links cluster to cloud provider
- Manages cloud-specific resources
- Load balancers, routes, volumes
- Optional component`,
					CodeExamples: `# View control plane components
kubectl get componentstatuses

# View API server endpoints
kubectl cluster-info

# Check etcd health (if accessible)
kubectl get --raw /healthz/etcd

# View scheduler logs
kubectl logs -n kube-system kube-scheduler-<node-name>

# View controller manager logs
kubectl logs -n kube-system kube-controller-manager-<node-name>

# Architecture diagram:
# 
# Control Plane
# ┌─────────────────────────────────────┐
# │  API Server (kube-apiserver)        │
# │  - Validates requests               │
# │  - Processes operations              │
# │  - Only etcd client                 │
# └──────────────┬──────────────────────┘
#                │
# ┌──────────────▼──────────────────────┐
# │  etcd                                │
# │  - Cluster state storage            │
# │  - Configuration data               │
# └─────────────────────────────────────┘
#                │
# ┌──────────────▼──────────────────────┐
# │  Scheduler (kube-scheduler)         │
# │  - Pod placement decisions          │
# └─────────────────────────────────────┘
#                │
# ┌──────────────▼──────────────────────┐
# │  Controller Manager                 │
# │  - Node Controller                  │
# │  - Replication Controller            │
# │  - Endpoints Controller              │
# └─────────────────────────────────────┘`,
				},
				{
					Title: "Node Components",
					Content: `**Worker Nodes** (also called Minions) run your containerized applications. Each node must have:

**kubelet:**
- Primary node agent
- Communicates with API server
- Manages pod lifecycle
- Reports node and pod status
- Ensures containers are running

**kube-proxy:**
- Network proxy on each node
- Maintains network rules
- Enables Service abstraction
- Uses iptables or IPVS
- Handles load balancing

**Container Runtime:**
- Software that runs containers
- Examples: Docker, containerd, CRI-O
- Implements Container Runtime Interface (CRI)
- Pulls images and runs containers
- Manages container lifecycle

**Node Communication:**
- Nodes communicate with control plane
- kubelet reports node status
- kubelet receives pod assignments
- kube-proxy updates network rules
- Container runtime pulls images

**Node Status:**
- **Ready**: Node is healthy and ready for pods
- **NotReady**: Node is not healthy
- **Unknown**: Node controller hasn't heard from node`,
					CodeExamples: `# View all nodes
kubectl get nodes

# Get detailed node information
kubectl describe node <node-name>

# View node resources
kubectl top node <node-name>

# View pods on a specific node
kubectl get pods -o wide --field-selector spec.nodeName=<node-name>

# Check kubelet status
systemctl status kubelet

# View kubelet logs
journalctl -u kubelet

# Node architecture:
#
# Worker Node
# ┌─────────────────────────────────────┐
# │  kubelet                            │
# │  - Pod lifecycle management         │
# │  - Reports to API server            │
# └──────────────┬──────────────────────┘
#                │
# ┌──────────────▼──────────────────────┐
# │  Container Runtime (Docker/CRI-O)   │
# │  - Runs containers                  │
# │  - Pulls images                     │
# └─────────────────────────────────────┘
#                │
# ┌──────────────▼──────────────────────┐
# │  kube-proxy                          │
# │  - Network rules                    │
# │  - Service proxy                    │
# └─────────────────────────────────────┘`,
				},
				{
					Title: "Cluster Architecture",
					Content: `**Kubernetes Cluster** consists of one or more control plane nodes and multiple worker nodes.

**High Availability Setup:**
- Multiple control plane nodes (3 or 5)
- Load balancer for API server
- etcd cluster (3 or 5 nodes)
- Distributed across availability zones

**Network Architecture:**
- **Pod Network**: Each pod gets unique IP
- **Service Network**: Virtual IPs for services
- **Node Network**: Physical network between nodes
- **CNI Plugins**: Manage pod networking

**Storage Architecture:**
- **Local Storage**: On nodes (ephemeral)
- **Network Storage**: NFS, Ceph, cloud volumes
- **Persistent Volumes**: Cluster-wide storage
- **Storage Classes**: Dynamic provisioning

**Communication Flow:**
1. User/Controller sends request to API server
2. API server validates and stores in etcd
3. Scheduler assigns pod to node
4. kubelet on node creates pod
5. Container runtime starts containers
6. kube-proxy sets up networking

**Cluster Components Interaction:**
- All components communicate via API server
- etcd is single source of truth
- Controllers watch API server for changes
- Nodes report status to API server`,
					CodeExamples: `# View cluster components
kubectl get all --all-namespaces

# View system pods (control plane)
kubectl get pods -n kube-system

# View cluster events
kubectl get events --sort-by='.lastTimestamp'

# Check cluster health
kubectl get --raw /healthz

# View API resources
kubectl api-resources

# Check API versions
kubectl api-versions

# Cluster architecture diagram:
#
#                    Load Balancer
#                         │
#         ┌───────────────┼───────────────┐
#         │               │               │
#    Control Plane    Control Plane    Control Plane
#    ┌──────────┐    ┌──────────┐    ┌──────────┐
#    │API Server│    │API Server│    │API Server│
#    │ etcd    │    │ etcd    │    │ etcd    │
#    │Scheduler│    │Scheduler│    │Scheduler│
#    └────┬─────┘    └────┬─────┘    └────┬─────┘
#         └───────────────┼───────────────┘
#                        │
#         ┌───────────────┼───────────────┐
#         │               │               │
#      Worker Node     Worker Node     Worker Node
#      ┌─────────┐    ┌─────────┐    ┌─────────┐
#      │ kubelet │    │ kubelet │    │ kubelet │
#      │kube-proxy│   │kube-proxy│   │kube-proxy│
#      │Runtime  │    │Runtime  │    │Runtime  │
#      └─────────┘    └─────────┘    └─────────┘`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          62,
			Title:       "Pods & Containers",
			Description: "Master the fundamental building blocks: pods, containers, and their lifecycle.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Pod Fundamentals",
					Content: `**Pods** are the smallest deployable units in Kubernetes. A pod represents a single instance of a running process in your cluster.

**Pod Characteristics:**
- Contains one or more containers
- Containers share network namespace
- Containers share storage volumes
- Pods are ephemeral (created/destroyed)
- Each pod gets unique IP address

**Pod Lifecycle:**
1. **Pending**: Pod accepted but containers not created
2. **Running**: Pod bound to node, all containers running
3. **Succeeded**: All containers terminated successfully
4. **Failed**: At least one container terminated with failure
5. **Unknown**: Pod state cannot be determined

**Pod States:**
- **ContainerCreating**: Pulling image, creating container
- **Running**: Container is running
- **Terminating**: Pod is being deleted
- **CrashLoopBackOff**: Container keeps crashing

**Pod Specifications:**
- **metadata**: Name, labels, annotations
- **spec**: Container definitions, volumes, restart policy
- **status**: Current state (read-only)

**Restart Policies:**
- **Always**: Always restart (default)
- **OnFailure**: Restart only on failure
- **Never**: Never restart`,
					CodeExamples: `# Create a simple pod
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    ports:
    - containerPort: 80

# Apply pod manifest
kubectl apply -f pod.yaml

# View pods
kubectl get pods

# Get pod details
kubectl describe pod nginx-pod

# View pod logs
kubectl logs nginx-pod

# Execute command in pod
kubectl exec -it nginx-pod -- /bin/bash

# Delete pod
kubectl delete pod nginx-pod

# Watch pods
kubectl get pods -w

# Get pod YAML
kubectl get pod nginx-pod -o yaml`,
				},
				{
					Title: "Container Specifications",
					Content: `**Container Spec** defines how containers run within a pod.

**Required Fields:**
- **name**: Container name (unique within pod)
- **image**: Container image to run

**Common Fields:**
- **ports**: Container ports to expose
- **env**: Environment variables
- **resources**: CPU and memory limits/requests
- **volumeMounts**: Mount volumes into container
- **command**: Override default entrypoint
- **args**: Arguments to command

**Resource Management:**
- **requests**: Minimum resources guaranteed
- **limits**: Maximum resources allowed
- CPU measured in cores (1000m = 1 core)
- Memory measured in bytes (Mi, Gi)

**Environment Variables:**
- Static values
- ConfigMap values
- Secret values
- Field references (pod metadata)

**Image Pull Policy:**
- **Always**: Always pull image
- **IfNotPresent**: Pull if not present (default)
- **Never**: Never pull, use local only`,
					CodeExamples: `# Pod with resource limits
apiVersion: v1
kind: Pod
metadata:
  name: resource-demo
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
    ports:
    - containerPort: 80
    env:
    - name: ENV_VAR
      value: "production"
    - name: CONFIG_VAR
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: config-key

# Pod with command override
apiVersion: v1
kind: Pod
metadata:
  name: command-demo
spec:
  containers:
  - name: app
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "while true; do echo hello; sleep 10; done"]

# Check resource usage
kubectl top pod <pod-name>

# View container logs
kubectl logs <pod-name> -c <container-name>`,
				},
				{
					Title: "Multi-Container Pods",
					Content: `**Multi-Container Pods** contain multiple containers that work together.

**Use Cases:**
- **Sidecar Pattern**: Helper container alongside main app
- **Adapter Pattern**: Transform output format
- **Ambassador Pattern**: Proxy network traffic
- **Logging/Monitoring**: Collect logs or metrics

**Shared Resources:**
- **Network**: Containers share IP and ports
- **Storage**: Containers can share volumes
- **IPC**: Containers can communicate via localhost

**Common Patterns:**

**Sidecar for Logging:**
- Main container writes logs to volume
- Sidecar container reads and ships logs

**Sidecar for Caching:**
- Main container uses cache
- Sidecar container manages cache

**Adapter for Metrics:**
- Main container exposes custom metrics
- Adapter transforms to standard format`,
					CodeExamples: `# Multi-container pod with sidecar
apiVersion: v1
kind: Pod
metadata:
  name: multi-container-pod
spec:
  containers:
  - name: app
    image: nginx:1.21
    volumeMounts:
    - name: shared-logs
      mountPath: /var/log/nginx
  - name: log-sidecar
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "tail -f /var/log/nginx/access.log"]
    volumeMounts:
    - name: shared-logs
      mountPath: /var/log/nginx
  volumes:
  - name: shared-logs
    emptyDir: {}

# Pod with init container
apiVersion: v1
kind: Pod
metadata:
  name: init-demo
spec:
  initContainers:
  - name: init-db
    image: busybox
    command: ["/bin/sh", "-c", "sleep 10"]
  containers:
  - name: app
    image: nginx:1.21

# View logs from specific container
kubectl logs multi-container-pod -c log-sidecar`,
				},
				{
					Title: "Init Containers",
					Content: `**Init Containers** run before application containers start. They can be used for setup tasks.

**Characteristics:**
- Run to completion before app containers start
- Run in sequence (one after another)
- If init container fails, pod restarts
- Can have different images than app containers

**Common Use Cases:**
- **Database Migration**: Run migrations before app starts
- **Wait for Dependencies**: Wait for external services
- **Setup Scripts**: Prepare filesystem or configuration
- **Security**: Fetch secrets or certificates

**Init Container Spec:**
- Defined in spec.initContainers
- Same container spec as regular containers
- Run in order (top to bottom)
- All must succeed for pod to start`,
					CodeExamples: `# Pod with init containers
apiVersion: v1
kind: Pod
metadata:
  name: init-demo
spec:
  initContainers:
  - name: wait-for-db
    image: busybox
    command: ['sh', '-c', 'until nc -z database 3306; do echo waiting for db; sleep 2; done']
  - name: init-config
    image: busybox
    command: ['sh', '-c', 'echo "config initialized" > /shared/config.txt']
    volumeMounts:
    - name: shared-data
      mountPath: /shared
  containers:
  - name: app
    image: nginx:1.21
    volumeMounts:
    - name: shared-data
      mountPath: /app/config
  volumes:
  - name: shared-data
    emptyDir: {}

# Check init container status
kubectl describe pod init-demo

# View init container logs
kubectl logs init-demo -c wait-for-db`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          63,
			Title:       "Services & Networking",
			Description: "Learn how Kubernetes handles networking, service discovery, and exposes applications.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Service Types",
					Content: `**Services** provide stable network endpoints for pods. Since pods are ephemeral, services abstract pod IP addresses.

**Service Types:**

**ClusterIP (Default):**
- Internal IP accessible only within cluster
- Default service type
- Used for internal communication
- Not accessible from outside cluster

**NodePort:**
- Exposes service on each node's IP at static port
- Accessible from outside cluster
- Port range: 30000-32767
- Creates ClusterIP service automatically

**LoadBalancer:**
- Exposes service externally using cloud provider's load balancer
- Creates NodePort and ClusterIP automatically
- Requires cloud provider support
- Each service gets external IP

**ExternalName:**
- Maps service to external DNS name
- Returns CNAME record
- No proxy, just DNS mapping
- Used for external services

**Service Selectors:**
- Match pods using labels
- Traffic routed to matching pods
- Load balanced across pods
- Health checks determine endpoints`,
					CodeExamples: `# ClusterIP service
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 9376

# NodePort service
apiVersion: v1
kind: Service
metadata:
  name: my-nodeport-service
spec:
  type: NodePort
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 9376
    nodePort: 30080

# LoadBalancer service
apiVersion: v1
kind: Service
metadata:
  name: my-lb-service
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 9376

# View services
kubectl get services

# Describe service
kubectl describe service my-service

# Get service endpoints
kubectl get endpoints my-service`,
				},
				{
					Title: "Service Discovery & DNS",
					Content: `**Service Discovery** allows pods to find and communicate with services without hardcoding IP addresses.

**DNS-Based Discovery:**
- Kubernetes provides DNS service
- Services get DNS names automatically
- Format: service-name.namespace.svc.cluster.local
- Short form: service-name.namespace or service-name (same namespace)

**CoreDNS:**
- Default DNS server in Kubernetes
- Resolves service names to ClusterIPs
- Provides DNS for pods and services
- Configurable via ConfigMap

**DNS Records:**
- **A Records**: Service name → ClusterIP
- **SRV Records**: Named ports
- **CNAME**: ExternalName services

**Namespaces:**
- DNS names are namespace-scoped
- Services in same namespace can use short name
- Cross-namespace requires FQDN

**Headless Services:**
- clusterIP: None
- Returns pod IPs directly
- Used for StatefulSets
- Enables direct pod-to-pod communication`,
					CodeExamples: `# Service with DNS
apiVersion: v1
kind: Service
metadata:
  name: database
  namespace: production
spec:
  selector:
    app: database
  ports:
  - port: 5432

# DNS resolution:
# database.production.svc.cluster.local → ClusterIP
# database.production → ClusterIP
# database → ClusterIP (if in same namespace)

# Headless service
apiVersion: v1
kind: Service
metadata:
  name: headless-service
spec:
  clusterIP: None
  selector:
    app: stateful-app
  ports:
  - port: 80

# Test DNS from pod
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup database

# View CoreDNS config
kubectl get configmap coredns -n kube-system -o yaml`,
				},
				{
					Title: "Network Policies",
					Content: `**Network Policies** control traffic flow between pods. They act as firewall rules for pods.

**Key Concepts:**
- **Pod Selectors**: Which pods the policy applies to
- **Ingress Rules**: Allowed incoming traffic
- **Egress Rules**: Allowed outgoing traffic
- **Default Deny**: No traffic allowed unless explicitly allowed

**Policy Types:**
- **Ingress**: Controls incoming traffic
- **Egress**: Controls outgoing traffic
- **Both**: Controls both directions

**Selectors:**
- **podSelector**: Selects pods in same namespace
- **namespaceSelector**: Selects pods in other namespaces
- **ipBlock**: Selects by IP CIDR

**Default Behavior:**
- If no policy applies: All traffic allowed
- If policy exists: Only explicitly allowed traffic
- Default deny all: Create policy with no rules

**Use Cases:**
- Isolate namespaces
- Restrict database access
- Multi-tenant clusters
- Security compliance`,
					CodeExamples: `# Network policy - deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
spec:
  podSelector: {}
  policyTypes:
  - Ingress

# Network policy - allow specific ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-app-to-db
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: webapp
    ports:
    - protocol: TCP
      port: 5432

# Network policy with egress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restrict-egress
spec:
  podSelector:
    matchLabels:
      app: restricted-app
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: allowed-namespace
    ports:
    - protocol: TCP
      port: 443

# View network policies
kubectl get networkpolicies`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          64,
			Title:       "Deployments & ReplicaSets",
			Description: "Master declarative application deployment, scaling, and updates.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Deployment Fundamentals",
					Content: `**Deployments** provide declarative updates for Pods and ReplicaSets. They manage the desired state of your application.

**Key Features:**
- **Declarative Updates**: Describe desired state
- **Rolling Updates**: Update pods gradually
- **Rollback**: Revert to previous version
- **Scaling**: Scale up or down
- **Self-Healing**: Replace failed pods

**Deployment Components:**
- **ReplicaSet**: Manages pod replicas
- **Pod Template**: Spec for creating pods
- **Replicas**: Desired number of pod instances
- **Strategy**: How to update pods

**Update Strategies:**
- **RollingUpdate** (default): Gradual replacement
- **Recreate**: Terminate all, then create new

**Rolling Update Parameters:**
- **maxUnavailable**: Max pods unavailable during update
- **maxSurge**: Max pods created above desired count

**Deployment Status:**
- **Progressing**: Deployment is progressing
- **Complete**: All replicas updated
- **Failed**: Deployment failed`,
					CodeExamples: `# Basic deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80

# Apply deployment
kubectl apply -f deployment.yaml

# View deployments
kubectl get deployments

# View deployment details
kubectl describe deployment nginx-deployment

# View rollout status
kubectl rollout status deployment/nginx-deployment

# View deployment history
kubectl rollout history deployment/nginx-deployment`,
				},
				{
					Title: "Rolling Updates and Rollbacks",
					Content: `**Rolling Updates** allow you to update application without downtime.

**Update Process:**
1. Create new ReplicaSet with new pod template
2. Gradually scale up new ReplicaSet
3. Gradually scale down old ReplicaSet
4. Old ReplicaSet kept for rollback

**Update Methods:**
- **kubectl set image**: Change container image
- **kubectl edit**: Edit deployment directly
- **kubectl apply**: Apply updated YAML
- **kubectl patch**: Patch deployment

**Rollback:**
- Revert to previous ReplicaSet
- Can rollback to any revision
- Preserves deployment history
- Fast and safe operation

**Rollout Commands:**
- **pause**: Pause rollout
- **resume**: Resume paused rollout
- **restart**: Restart deployment
- **undo**: Rollback to previous revision`,
					CodeExamples: `# Update deployment image
kubectl set image deployment/nginx-deployment nginx=nginx:1.22

# Update with specific strategy
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.22

# Pause rollout
kubectl rollout pause deployment/nginx-deployment

# Resume rollout
kubectl rollout resume deployment/nginx-deployment

# Rollback to previous version
kubectl rollout undo deployment/nginx-deployment

# Rollback to specific revision
kubectl rollout undo deployment/nginx-deployment --to-revision=2

# View rollout history
kubectl rollout history deployment/nginx-deployment

# View specific revision
kubectl rollout history deployment/nginx-deployment --revision=2`,
				},
				{
					Title: "ReplicaSets and Scaling",
					Content: `**ReplicaSets** ensure a specified number of pod replicas are running at any time.

**ReplicaSet Purpose:**
- Maintain desired number of pods
- Replace failed pods
- Scale pods up or down
- Managed by Deployments

**ReplicaSet vs ReplicationController:**
- ReplicaSet uses set-based selectors
- ReplicationController uses equality-based selectors
- ReplicaSet is newer and preferred

**Scaling Methods:**
- **Manual Scaling**: Change replica count
- **Horizontal Pod Autoscaler**: Auto-scale based on metrics
- **Vertical Pod Autoscaler**: Adjust resource requests

**Scaling Commands:**
- **kubectl scale**: Scale deployment/replicaset
- **kubectl autoscale**: Create HPA

**Pod Selection:**
- Uses label selectors
- Can use multiple labels
- Set-based matching (in, notin, exists)`,
					CodeExamples: `# Manual scaling
kubectl scale deployment/nginx-deployment --replicas=5

# Scale using YAML
kubectl scale --replicas=5 -f deployment.yaml

# View ReplicaSets
kubectl get replicasets

# View ReplicaSet details
kubectl describe replicaset <replicaset-name>

# ReplicaSet example
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: nginx-replicaset
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21

# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nginx-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nginx-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70`,
				},
				{
					Title: "Health Checks",
					Content: `**Health Checks** ensure pods are running correctly and ready to serve traffic.

**Probe Types:**

**Liveness Probe:**
- Determines if container is running
- If fails, container is restarted
- Use for detecting deadlocks
- Should be lightweight

**Readiness Probe:**
- Determines if container is ready for traffic
- If fails, pod removed from service endpoints
- Use for startup checks
- Prevents traffic to unready pods

**Startup Probe:**
- Determines if application has started
- Disables liveness/readiness until succeeds
- Useful for slow-starting apps
- Prevents premature restarts

**Probe Mechanisms:**
- **HTTP GET**: Check HTTP endpoint
- **TCP Socket**: Check if port is open
- **Exec**: Execute command

**Probe Parameters:**
- **initialDelaySeconds**: Wait before first probe
- **periodSeconds**: How often to probe
- **timeoutSeconds**: Probe timeout
- **successThreshold**: Consecutive successes needed
- **failureThreshold**: Consecutive failures before failure`,
					CodeExamples: `# Deployment with health checks
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:1.0
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /startup
            port: 8080
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 30

# TCP socket probe
livenessProbe:
  tcpSocket:
    port: 8080
  initialDelaySeconds: 15
  periodSeconds: 20

# Exec probe
livenessProbe:
  exec:
    command:
    - cat
    - /tmp/healthy
  initialDelaySeconds: 5
  periodSeconds: 5`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
