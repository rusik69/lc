package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          260,
			Title:       "Advanced Docker",
			Description: "Advanced Docker concepts: multi-stage builds, optimization, security, and best practices.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Docker Optimization and Best Practices",
					Content: `Optimize Docker images and containers for production.

**Image Optimization:**
- Use minimal base images (Alpine)
- Multi-stage builds
- Layer caching
- .dockerignore files
- Combine RUN commands
- Remove unnecessary files

**Security Best Practices:**
- Don't run as root
- Scan images for vulnerabilities
- Use specific image tags
- Keep images updated
- Minimal attack surface
- Secrets management

**Performance:**
- Reduce image size
- Optimize layer order
- Use build cache
- Parallel builds
- Health checks

**Common Patterns:**
- Builder pattern
- Multi-stage builds
- Volume mounts
- Network isolation`,
					CodeExamples: `# Optimized Dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && \
    npm cache clean --force
COPY . .
RUN npm run build

FROM node:18-alpine
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
WORKDIR /app
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --chown=nodejs:nodejs package*.json ./
USER nodejs
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD node healthcheck.js
CMD ["node", "dist/server.js"]

# .dockerignore
node_modules
.git
.env
*.log
dist
coverage
.DS_Store

# Security scan
docker scan myimage:latest

# Build with build args
docker build --build-arg NODE_ENV=production -t myapp .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t myapp .`,
				},
				{
					Title: "Docker Networking and Volumes",
					Content: `Advanced Docker networking and storage management.

**Networking:**
- Bridge network: Default network
- Host network: Use host network
- Overlay network: Multi-host networking
- Macvlan: MAC address per container
- Custom networks: Isolated networks

**Volumes:**
- Named volumes: Managed by Docker
- Bind mounts: Host filesystem
- tmpfs mounts: In-memory storage
- Volume drivers: External storage

**Best Practices:**
- Use named volumes for data
- Network isolation for services
- Volume backups
- Network policies`,
					CodeExamples: `# Create network
docker network create app-network

# Run container with network
docker run -d --network app-network --name web nginx

# Create volume
docker volume create postgres-data

# Use volume
docker run -d \
    -v postgres-data:/var/lib/postgresql/data \
    postgres:14

# Bind mount
docker run -d \
    -v /host/path:/container/path \
    nginx

# Docker Compose networking
version: '3.8'
services:
  app:
    networks:
      - frontend
      - backend
  db:
    networks:
      - backend

networks:
  frontend:
  backend:
    driver: bridge`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          261,
			Title:       "Kubernetes Fundamentals",
			Description: "Introduction to Kubernetes: architecture, pods, services, and core concepts.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Kubernetes Architecture",
					Content: `Understanding Kubernetes cluster architecture and components.

**Control Plane Components:**
- **API Server**: Entry point for all operations
- **etcd**: Distributed key-value store
- **Scheduler**: Assigns pods to nodes
- **Controller Manager**: Runs controllers

**Node Components:**
- **kubelet**: Agent on each node
- **kube-proxy**: Network proxy
- **Container Runtime**: Runs containers

**Key Concepts:**
- **Pod**: Smallest deployable unit
- **Service**: Stable network endpoint
- **Deployment**: Manages pod replicas
- **Namespace**: Virtual cluster
- **ConfigMap**: Configuration data
- **Secret**: Sensitive data

**Kubernetes Objects:**
- Declarative configuration
- YAML manifests
- Desired state management
- Self-healing`,
					CodeExamples: `# Pod definition
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    ports:
    - containerPort: 80

# Deployment
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

# Service
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer

# Apply resources
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get services
kubectl describe pod <pod-name>`,
				},
				{
					Title: "Kubernetes Objects and Resources",
					Content: `Understanding Kubernetes object model and resource management.

**Kubernetes Object Model:**
- **Objects**: Persistent entities representing cluster state
- **Spec**: Desired state (what you want)
- **Status**: Current state (what Kubernetes has achieved)
- **Labels**: Key-value pairs for identification
- **Annotations**: Metadata for tools and libraries
- **Namespaces**: Virtual clusters within a cluster

**Core Objects:**
- **Pod**: Smallest deployable unit (1+ containers)
- **Service**: Stable network endpoint for pods
- **Deployment**: Manages pod replicas and updates
- **ConfigMap**: Non-sensitive configuration data
- **Secret**: Sensitive data (passwords, tokens)
- **Namespace**: Logical grouping of resources

**Workload Objects:**
- **Deployment**: Stateless applications
- **StatefulSet**: Stateful applications with stable identity
- **DaemonSet**: One pod per node
- **Job**: One-time tasks
- **CronJob**: Scheduled tasks

**Configuration Objects:**
- **ConfigMap**: Configuration data
- **Secret**: Sensitive data
- **PersistentVolume**: Storage abstraction
- **PersistentVolumeClaim**: Storage request

**Service Objects:**
- **Service**: Network service
- **Ingress**: HTTP/HTTPS routing
- **NetworkPolicy**: Network traffic rules

**Best Practices:**
- Use labels consistently
- Organize with namespaces
- Use ConfigMaps for configuration
- Use Secrets for sensitive data
- Define resource requests and limits
- Use Deployments for stateless apps
- Use StatefulSets for stateful apps

**Common Pitfalls:**
- Not setting resource limits
- Using default namespace
- Not using labels properly
- Hardcoding configuration
- Not using ConfigMaps/Secrets`,
					CodeExamples: `# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: default
data:
  database_url: "postgresql://db:5432/mydb"
  log_level: "info"
  max_connections: "100"

# Secret
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
stringData:
  username: admin
  password: secret123

# Using ConfigMap and Secret in Pod
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:latest
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: database_url
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: log_level
    envFrom:
    - secretRef:
        name: app-secret
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
  volumes:
  - name: config-volume
    configMap:
      name: app-config

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
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi

# DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: log-collector
spec:
  selector:
    matchLabels:
      app: log-collector
  template:
    metadata:
      labels:
        app: log-collector
    spec:
      containers:
      - name: fluentd
        image: fluentd:latest
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

# Job
apiVersion: batch/v1
kind: Job
metadata:
  name: backup-job
spec:
  template:
    spec:
      containers:
      - name: backup
        image: backup-tool:latest
        command: ["/bin/sh", "-c", "backup.sh"]
      restartPolicy: Never
  backoffLimit: 4

# CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-job
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: cleanup-tool:latest
          restartPolicy: OnFailure

# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    environment: production

# Resource with labels and annotations
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
  namespace: production
  labels:
    app: myapp
    version: v1.0
    environment: production
  annotations:
    description: "Main application deployment"
    contact: "team@example.com"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        version: v1.0
    spec:
      containers:
      - name: app
        image: myapp:v1.0
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          262,
			Title:       "Kubernetes Deployment and Services",
			Description: "Deploy applications to Kubernetes: deployments, services, ingress, and scaling.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Deploying Applications",
					Content: `Deploy and manage applications in Kubernetes.

**Deployment Strategies:**
- **Rolling Update**: Gradual replacement
- **Recreate**: Stop all, start new
- **Blue-Green**: Two versions, switch
- **Canary**: Gradual traffic shift

**Scaling:**
- Manual scaling: kubectl scale
- Horizontal Pod Autoscaler (HPA)
- Vertical Pod Autoscaler (VPA)
- Cluster Autoscaler

**Service Types:**
- **ClusterIP**: Internal only
- **NodePort**: Expose on node port
- **LoadBalancer**: Cloud load balancer
- **ExternalName**: External service

**Ingress:**
- HTTP/HTTPS routing
- SSL termination
- Path-based routing
- Host-based routing`,
					CodeExamples: `# Deployment with rolling update
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
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
        image: myapp:v1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10

# Horizontal Pod Autoscaler
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

# Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80`,
				},
				{
					Title: "Advanced Deployment Patterns",
					Content: `Advanced Kubernetes deployment strategies and patterns.

**Deployment Strategies:**
- **Rolling Update**: Gradual replacement (default)
- **Recreate**: Stop all, start new
- **Blue-Green**: Two versions, switch traffic
- **Canary**: Gradual traffic shift
- **A/B Testing**: Split traffic between versions

**Advanced Patterns:**
- **Probes**: Health checks (liveness, readiness, startup)
- **Resource Management**: Requests and limits
- **Affinity/Anti-affinity**: Pod placement rules
- **Taints and Tolerations**: Node selection
- **Pod Disruption Budgets**: Maintain availability during disruptions
- **Horizontal Pod Autoscaler**: Automatic scaling
- **Vertical Pod Autoscaler**: Resource optimization

**Best Practices:**
- Use readiness probes
- Set resource requests and limits
- Use rolling updates for zero downtime
- Implement pod disruption budgets
- Use HPA for automatic scaling
- Monitor deployment status
- Test rollback procedures

**Common Pitfalls:**
- Not setting resource limits
- Missing health checks
- Not testing rollbacks
- Ignoring pod disruption budgets
- Not monitoring deployments`,
					CodeExamples: `# Deployment with probes and resource limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
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
        image: myapp:v1.0
        ports:
        - containerPort: 8080
        # Resource limits
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        # Health probes
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
      # Pod disruption budget
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - myapp
              topologyKey: kubernetes.io/hostname

# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: app-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: myapp

# Horizontal Pod Autoscaler
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max

# Blue-Green Deployment
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
      - name: app
        image: myapp:v1.0

# Green deployment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
      - name: app
        image: myapp:v2.0

# Service switches between blue and green
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: myapp
    version: blue  # Switch to green for deployment
  ports:
  - port: 80
    targetPort: 8080

# Canary Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
spec:
  replicas: 1  # Small percentage
  selector:
    matchLabels:
      app: myapp
      track: canary
  template:
    metadata:
      labels:
        app: myapp
        track: canary
    spec:
      containers:
      - name: app
        image: myapp:v2.0`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          263,
			Title:       "Cloud Platforms (AWS/Azure/GCP)",
			Description: "Cloud platform essentials: compute, storage, networking, and managed services.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "AWS Essentials",
					Content: `Amazon Web Services core services for DevOps.

**Compute:**
- **EC2**: Virtual servers
- **ECS**: Container service
- **EKS**: Kubernetes service
- **Lambda**: Serverless functions

**Storage:**
- **S3**: Object storage
- **EBS**: Block storage
- **EFS**: File storage
- **Glacier**: Archive storage

**Networking:**
- **VPC**: Virtual network
- **ELB**: Load balancer
- **Route 53**: DNS
- **CloudFront**: CDN

**Database:**
- **RDS**: Managed databases
- **DynamoDB**: NoSQL database
- **ElastiCache**: Caching

**DevOps Services:**
- **CodePipeline**: CI/CD
- **CodeBuild**: Build service
- **CodeDeploy**: Deployment
- **CloudFormation**: IaC`,
					CodeExamples: `# AWS CLI examples
aws s3 ls                          # List buckets
aws s3 cp file.txt s3://bucket/    # Upload file
aws ec2 describe-instances         # List instances
aws eks list-clusters              # List EKS clusters

# CloudFormation template
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  MyBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: my-bucket
      VersioningConfiguration:
        Status: Enabled

  MyInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: t2.micro
      Tags:
        - Key: Name
          Value: MyInstance

# Terraform AWS
resource "aws_instance" "web" {
    ami           = "ami-0c55b159cbfafe1f0"
    instance_type = "t2.micro"
    
    tags = {
        Name = "WebServer"
    }
}

resource "aws_s3_bucket" "data" {
    bucket = "my-data-bucket"
    
    versioning {
        enabled = true
    }
}`,
				},
				{
					Title: "Azure and GCP Essentials",
					Content: `Microsoft Azure and Google Cloud Platform essentials.

**Azure Services:**
- **Virtual Machines**: Compute
- **AKS**: Kubernetes service
- **Blob Storage**: Object storage
- **Azure DevOps**: CI/CD
- **ARM Templates**: Infrastructure as Code

**GCP Services:**
- **Compute Engine**: VMs
- **GKE**: Kubernetes service
- **Cloud Storage**: Object storage
- **Cloud Build**: CI/CD
- **Deployment Manager**: IaC

**Common Patterns:**
- Managed Kubernetes
- Object storage
- CI/CD pipelines
- Infrastructure as Code
- Monitoring and logging`,
					CodeExamples: `# Azure CLI
az login
az group create --name myResourceGroup --location eastus
az vm create --resource-group myResourceGroup --name myVM --image UbuntuLTS

# Azure ARM Template
{
    "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
    "resources": [
        {
            "type": "Microsoft.Storage/storageAccounts",
            "name": "mystorageaccount",
            "apiVersion": "2021-09-01",
            "location": "eastus",
            "sku": {
                "name": "Standard_LRS"
            }
        }
    ]
}

# GCP gcloud CLI
gcloud auth login
gcloud compute instances create my-instance --zone=us-central1-a
gcloud container clusters create my-cluster

# GCP Deployment Manager
resources:
- name: my-instance
  type: compute.v1.instance
  properties:
    zone: us-central1-a
    machineType: zones/us-central1-a/machineTypes/n1-standard-1
    disks:
    - deviceName: boot
      type: PERSISTENT
      boot: true
      autoDelete: true
      initializeParams:
        sourceImage: projects/debian-cloud/global/images/family/debian-11`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          264,
			Title:       "Infrastructure as Code (Terraform)",
			Description: "Master Terraform: infrastructure provisioning, state management, and best practices.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Terraform Fundamentals",
					Content: `Terraform enables infrastructure as code with declarative configuration.

**Terraform Concepts:**
- **Provider**: Cloud/platform plugin
- **Resource**: Infrastructure component
- **State**: Current infrastructure state
- **Plan**: Preview changes
- **Apply**: Execute changes
- **Module**: Reusable configuration

**Workflow:**
1. Write configuration (.tf files)
2. Initialize: terraform init
3. Plan: terraform plan
4. Apply: terraform apply
5. Destroy: terraform destroy

**State Management:**
- Local state: Default
- Remote state: Shared (S3, etc.)
- State locking: Prevent conflicts
- State backends: Storage location

**Best Practices:**
- Version control
- Remote state
- Modules for reusability
- Variables for flexibility
- Outputs for values`,
					CodeExamples: `# main.tf
terraform {
    required_version = ">= 1.0"
    required_providers {
        aws = {
            source  = "hashicorp/aws"
            version = "~> 4.0"
        }
    }
    backend "s3" {
        bucket = "my-terraform-state"
        key    = "terraform.tfstate"
        region = "us-east-1"
    }
}

provider "aws" {
    region = var.aws_region
}

resource "aws_instance" "web" {
    ami           = var.ami_id
    instance_type = var.instance_type
    
    tags = {
        Name = "WebServer"
        Environment = var.environment
    }
}

# variables.tf
variable "aws_region" {
    description = "AWS region"
    type        = string
    default     = "us-east-1"
}

variable "instance_type" {
    description = "EC2 instance type"
    type        = string
    default     = "t2.micro"
}

# outputs.tf
output "instance_id" {
    value = aws_instance.web.id
}

output "public_ip" {
    value = aws_instance.web.public_ip
}

# terraform.tfvars
aws_region     = "us-east-1"
instance_type  = "t2.micro"
environment    = "production"`,
				},
				{
					Title: "Advanced Terraform",
					Content: `Advanced Terraform patterns and best practices.

**Modules:**
- Reusable components
- Input/output variables
- Versioning
- Module registry

**Workspaces:**
- Environment isolation
- State separation
- Workspace-specific configs

**Data Sources:**
- Query existing resources
- Dynamic configuration
- Cross-resource references

**Provisioners:**
- Local-exec: Run local commands
- Remote-exec: Run on resource
- File: Copy files

**Best Practices:**
- Use modules
- Workspaces for environments
- Data sources for existing resources
- Avoid provisioners when possible
- Use terraform fmt and validate`,
					CodeExamples: `# Module structure
# modules/ec2/main.tf
variable "instance_type" {
    type = string
}

variable "ami_id" {
    type = string
}

resource "aws_instance" "this" {
    ami           = var.ami_id
    instance_type = var.instance_type
}

output "instance_id" {
    value = aws_instance.this.id
}

# Using module
module "web_server" {
    source = "./modules/ec2"
    
    instance_type = "t2.micro"
    ami_id        = "ami-0c55b159cbfafe1f0"
}

# Data source
data "aws_ami" "ubuntu" {
    most_recent = true
    owners      = ["099720109477"]
    
    filter {
        name   = "name"
        values = ["ubuntu/images/hubuntu-*-amd64-server-*"]
    }
}

resource "aws_instance" "web" {
    ami           = data.aws_ami.ubuntu.id
    instance_type = "t2.micro"
}

# Workspaces
terraform workspace new dev
terraform workspace select dev
terraform apply

# Conditional resources
resource "aws_instance" "web" {
    count = var.create_instance ? 1 : 0
    # ...
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          265,
			Title:       "Configuration Management (Ansible)",
			Description: "Master Ansible: playbooks, roles, inventory management, and automation.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Ansible Playbooks and Roles",
					Content: `Ansible automates configuration management and application deployment.

**Ansible Concepts:**
- **Playbook**: Automation script (YAML)
- **Task**: Single action
- **Module**: Task implementation
- **Role**: Reusable playbook
- **Inventory**: Host list
- **Vault**: Encrypted secrets

**Common Modules:**
- **apt/yum**: Package management
- **systemd**: Service management
- **copy/template**: File operations
- **user/group**: User management
- **docker_container**: Docker operations

**Best Practices:**
- Use roles for reusability
- Idempotent tasks
- Use variables
- Encrypt secrets with vault
- Use handlers for notifications`,
					CodeExamples: `# playbook.yml
- hosts: webservers
  become: yes
  vars:
    nginx_version: "1.21.0"
  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
    
    - name: Install nginx
      apt:
        name: nginx
        state: present
    
    - name: Copy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: restart nginx
    
    - name: Start nginx
      systemd:
        name: nginx
        state: started
        enabled: yes
  
  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted

# Role structure
roles/nginx/
  tasks/main.yml
  handlers/main.yml
  templates/nginx.conf.j2
  vars/main.yml
  defaults/main.yml

# Using role
- hosts: webservers
  roles:
    - nginx
    - { role: database, db_name: myapp }

# Inventory
[webservers]
web1 ansible_host=192.168.1.10
web2 ansible_host=192.168.1.11

[databases]
db1 ansible_host=192.168.1.20

[all:vars]
ansible_user=ubuntu
ansible_ssh_private_key_file=~/.ssh/id_rsa`,
				},
				{
					Title: "Advanced Ansible",
					Content: `Advanced Ansible features and patterns.

**Ansible Vault:**
- Encrypt sensitive data
- Encrypt files or variables
- Password or key file

**Dynamic Inventory:**
- AWS EC2
- Azure
- GCP
- Custom scripts

**Conditionals and Loops:**
- when: Conditional execution
- loop: Iterate over items
- with_items: Legacy loops

**Tags:**
- Organize tasks
- Run specific tasks
- Skip tasks

**Best Practices:**
- Use vault for secrets
- Dynamic inventory for cloud
- Tags for organization
- Test playbooks with --check`,
					CodeExamples: `# Encrypt file
ansible-vault create secrets.yml

# Encrypt variable
ansible-vault encrypt_string 'secret_password' --name db_password

# Use vault in playbook
- hosts: all
  vars_files:
    - secrets.yml
  tasks:
    - name: Use secret
      debug:
        var: db_password

# Conditional task
- name: Install package
  apt:
    name: nginx
  when: os_family == "Debian"

# Loop
- name: Install packages
  apt:
    name: "{{ item }}"
  loop:
    - nginx
    - mysql
    - redis

# Tags
- name: Install nginx
  apt:
    name: nginx
  tags:
    - packages
    - nginx

# Run with tags
ansible-playbook playbook.yml --tags packages

# Dynamic inventory (AWS)
plugin: aws_ec2
regions:
  - us-east-1
filters:
  - instance-state-name: running`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          266,
			Title:       "CI/CD Pipelines (Jenkins, GitHub Actions)",
			Description: "Build advanced CI/CD pipelines with Jenkins and GitHub Actions.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Jenkins CI/CD",
					Content: `Jenkins is a self-hosted automation server for CI/CD.

**Jenkins Concepts:**
- **Job**: Automation task
- **Pipeline**: Code-defined workflow
- **Node/Agent**: Execution environment
- **Plugin**: Extend functionality
- **Jenkinsfile**: Pipeline as code

**Pipeline Types:**
- **Declarative**: Structured syntax
- **Scripted**: Groovy-based

**Stages:**
- Build
- Test
- Deploy
- Post-actions

**Best Practices:**
- Use Jenkinsfile
- Version control pipelines
- Use shared libraries
- Secure credentials
- Parallel execution`,
					CodeExamples: `# Jenkinsfile (Declarative)
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
        
        stage('Test') {
            steps {
                sh 'npm test'
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_REGISTRY}/app:${IMAGE_TAG}")
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl set image deployment/app app=${DOCKER_REGISTRY}/app:${IMAGE_TAG}'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}`,
				},
				{
					Title: "GitHub Actions Advanced",
					Content: `Advanced GitHub Actions workflows and patterns.

**GitHub Actions Features:**
- Workflows: YAML-defined automation
- Actions: Reusable steps
- Events: Triggers (push, PR, etc.)
- Secrets: Encrypted variables
- Matrix: Multiple versions
- Caching: Speed up builds

**Workflow Patterns:**
- Multi-job workflows
- Dependent jobs
- Matrix builds
- Conditional execution
- Reusable workflows

**Best Practices:**
- Use actions from marketplace
- Cache dependencies
- Use matrix for testing
- Secure secrets
- Use environment protection`,
					CodeExamples: `# Advanced GitHub Actions workflow
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  NODE_VERSION: '18'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
        os: [ubuntu-latest, windows-latest]
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache
          cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache,mode=max

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/app \
            app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          kubectl rollout status deployment/app`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          267,
			Title:       "Container Orchestration",
			Description: "Advanced container orchestration: scaling, service discovery, and multi-container patterns.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Orchestration Patterns",
					Content: `Patterns for orchestrating containerized applications.

**Service Discovery:**
- DNS-based discovery
- Service registry
- Health checks
- Load balancing

**Scaling Strategies:**
- Horizontal scaling
- Vertical scaling
- Auto-scaling
- Manual scaling

**Deployment Patterns:**
- Rolling updates
- Blue-green deployment
- Canary deployment
- A/B testing

**Multi-Container Patterns:**
- Sidecar pattern
- Ambassador pattern
- Adapter pattern
- Init containers`,
					CodeExamples: `# Service discovery (Kubernetes)
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

# Auto-scaling
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

# Sidecar pattern
apiVersion: v1
kind: Pod
metadata:
  name: app-with-sidecar
spec:
  containers:
  - name: app
    image: myapp:latest
  - name: sidecar
    image: logging-sidecar:latest
    volumeMounts:
    - name: logs
      mountPath: /var/log/app`,
				},
				{
					Title: "Advanced Orchestration Patterns",
					Content: `Advanced patterns for orchestrating complex containerized applications.

**Advanced Patterns:**
- **Service Mesh Integration**: Traffic management and security
- **Multi-Region Deployment**: Geographic distribution
- **Hybrid Cloud**: On-premises and cloud integration
- **Edge Computing**: Distributed edge deployments
- **Event-Driven Architecture**: Event-based service communication
- **Circuit Breaker**: Fault tolerance patterns
- **Bulkhead**: Resource isolation

**Scaling Strategies:**
- **Horizontal Scaling**: Add more instances
- **Vertical Scaling**: Increase resources
- **Auto-scaling**: Automatic resource adjustment
- **Predictive Scaling**: Scale based on predictions
- **Scheduled Scaling**: Scale on schedule

**Service Discovery:**
- **DNS-based**: Traditional DNS
- **Service Registry**: Centralized registry (Consul, etcd)
- **Client-side Discovery**: Client queries registry
- **Server-side Discovery**: Load balancer queries registry

**Load Balancing:**
- **Round Robin**: Distribute evenly
- **Least Connections**: Fewest active connections
- **IP Hash**: Consistent hashing
- **Weighted**: Based on server capacity
- **Geographic**: Route by location

**Best Practices:**
- Implement health checks
- Use service discovery
- Implement circuit breakers
- Monitor service dependencies
- Use graceful shutdown
- Implement retry logic with backoff

**Common Pitfalls:**
- Not implementing health checks
- Ignoring service dependencies
- No circuit breakers
- Poor load balancing configuration
- Not monitoring service health`,
					CodeExamples: `# Service discovery with Consul
# consul-config.json
{
  "service": {
    "name": "myapp",
    "port": 8080,
    "check": {
      "http": "http://localhost:8080/health",
      "interval": "10s"
    }
  }
}

# Register service
consul agent -config-dir=/etc/consul.d

# Discover services
curl http://localhost:8500/v1/catalog/service/myapp

# Circuit breaker pattern (Python)
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_service():
    # Make external API call
    response = requests.get('https://api.example.com')
    return response.json()

# Load balancer configuration (Nginx)
upstream backend {
    least_conn;
    server app1:8080 weight=3;
    server app2:8080 weight=2;
    server app3:8080 backup;
    
    keepalive 32;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Health check endpoint
@app.route('/health')
def health():
    checks = {
        'database': check_database(),
        'cache': check_cache(),
        'external_api': check_external_api()
    }
    
    status = 'healthy' if all(checks.values()) else 'unhealthy'
    return jsonify({'status': status, 'checks': checks}), 200 if status == 'healthy' else 503

# Retry with exponential backoff
import time
import random

def retry_with_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)

# Graceful shutdown
import signal
import sys

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    # Stop accepting new requests
    # Finish processing current requests
    # Close connections
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          268,
			Title:       "Logging and Log Aggregation",
			Description: "Centralized logging: ELK Stack, Loki, Fluentd, and log aggregation strategies.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Centralized Logging",
					Content: `Centralized logging for distributed systems.

**Logging Architecture:**
- **Collection**: Agents collect logs
- **Aggregation**: Central log store
- **Storage**: Time-series or searchable
- **Visualization**: Dashboards and search
- **Alerting**: Notifications on errors

**ELK Stack:**
- **Elasticsearch**: Search and analytics
- **Logstash**: Log processing
- **Kibana**: Visualization

**Loki Stack:**
- **Loki**: Log aggregation
- **Promtail**: Log collector
- **Grafana**: Visualization

**Fluentd/Fluent Bit:**
- Log collection and forwarding
- Lightweight and efficient
- Plugin ecosystem

**Best Practices:**
- Structured logging (JSON)
- Log levels
- Include context
- Don't log sensitive data
- Centralized collection`,
					CodeExamples: `# Fluentd configuration
<source>
  @type tail
  path /var/log/app/*.log
  pos_file /var/log/fluentd-app.log.pos
  tag app.logs
  <parse>
    @type json
  </parse>
</source>

<match app.logs>
  @type elasticsearch
  host elasticsearch.logging.svc.cluster.local
  port 9200
  index_name app-logs
  type_name _doc
</match>

# Promtail configuration (Loki)
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: app
    static_configs:
      - targets:
          - localhost
        labels:
          job: app
          __path__: /var/log/app/*.log

# Logstash pipeline
input {
  file {
    path => "/var/log/app/*.log"
    codec => json
  }
}

filter {
  if [level] == "ERROR" {
    mutate {
      add_tag => [ "error" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "app-logs-%{+YYYY.MM.dd}"
  }
}

# Application logging (structured)
{
    "timestamp": "2024-01-17T10:00:00Z",
    "level": "INFO",
    "service": "user-service",
    "request_id": "abc123",
    "message": "User created",
    "user_id": 123,
    "duration_ms": 45
}`,
				},
				{
					Title: "Log Analysis and Troubleshooting",
					Content: `Effective log analysis and troubleshooting techniques.

**Log Analysis Techniques:**
- **Pattern Recognition**: Identify common patterns
- **Correlation**: Link related log entries
- **Aggregation**: Group similar events
- **Filtering**: Focus on relevant logs
- **Search**: Find specific events
- **Visualization**: Graphical representation

**Troubleshooting Workflow:**
1. **Identify**: What is the problem?
2. **Locate**: Where did it occur?
3. **Timeframe**: When did it happen?
4. **Correlate**: Link related events
5. **Analyze**: Understand root cause
6. **Resolve**: Fix the issue
7. **Verify**: Confirm resolution

**Common Issues:**
- **High Error Rates**: Application errors
- **Slow Response Times**: Performance issues
- **Memory Leaks**: Resource exhaustion
- **Connection Issues**: Network problems
- **Authentication Failures**: Security issues
- **Database Errors**: Data access problems

**Log Analysis Tools:**
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Loki**: Log aggregation system
- **Splunk**: Enterprise log analysis
- **Grafana**: Visualization and alerting
- **Grep/AWK**: Command-line tools

**Best Practices:**
- Use structured logging
- Include correlation IDs
- Log at appropriate levels
- Don't log sensitive data
- Use log rotation
- Monitor log volume
- Set up alerts on errors

**Common Pitfalls:**
- Logging too much or too little
- Not using structured logs
- Missing correlation IDs
- Logging sensitive information
- Not monitoring logs
- Poor log organization`,
					CodeExamples: `# Structured logging with correlation ID
import logging
import uuid

logger = logging.getLogger(__name__)

def handle_request(request):
    correlation_id = str(uuid.uuid4())
    logger.info("Request received", extra={
        "correlation_id": correlation_id,
        "method": request.method,
        "path": request.path,
        "ip": request.remote_addr
    })
    
    try:
        result = process_request(request)
        logger.info("Request completed", extra={
            "correlation_id": correlation_id,
            "status": "success",
            "duration_ms": result.duration
        })
        return result
    except Exception as e:
        logger.error("Request failed", extra={
            "correlation_id": correlation_id,
            "error": str(e),
            "error_type": type(e).__name__
        }, exc_info=True)
        raise

# Log analysis queries (ELK/Kibana)
# Find all errors in last hour
level:ERROR AND @timestamp:[now-1h TO now]

# Find slow requests (>1s)
duration_ms:>1000 AND @timestamp:[now-1h TO now]

# Group errors by type
level:ERROR | stats count by error_type

# Find correlation ID
correlation_id:"abc-123-def"

# Log analysis with grep
# Find all errors
grep -i error /var/log/app.log

# Find errors in last 10 lines
tail -n 100 /var/log/app.log | grep -i error

# Count errors by type
grep -i error /var/log/app.log | awk '{print $5}' | sort | uniq -c

# Find slow requests
grep "duration_ms" /var/log/app.log | awk '$NF > 1000'

# Log analysis script
#!/bin/bash
LOG_FILE="/var/log/app.log"
ERROR_THRESHOLD=10

# Count errors in last hour
ERROR_COUNT=$(grep -c "level:ERROR" $LOG_FILE)

if [ $ERROR_COUNT -gt $ERROR_THRESHOLD ]; then
    echo "ALERT: High error count: $ERROR_COUNT"
    # Send alert
    # Send email, Slack notification, etc.
fi

# Find top error types
echo "Top error types:"
grep "level:ERROR" $LOG_FILE | \
    awk -F'"' '{print $4}' | \
    sort | uniq -c | sort -rn | head -10

# Log rotation configuration (logrotate)
/var/log/app/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 app app
    sharedscripts
    postrotate
        systemctl reload app
    endscript
}

# Loki query examples
# Find errors
{app="myapp"} |= "ERROR"

# Find slow requests
{app="myapp"} | json | duration_ms > 1000

# Count errors by service
sum by (service) (count_over_time({level="ERROR"}[5m]))`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          269,
			Title:       "Application Performance Monitoring",
			Description: "APM tools and practices: New Relic, Datadog, Prometheus, and performance monitoring.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "APM Fundamentals",
					Content: `Application Performance Monitoring tracks application health and performance.

**APM Metrics:**
- **Response Time**: Request latency
- **Throughput**: Requests per second
- **Error Rate**: Failed requests
- **Apdex**: User satisfaction score
- **Resource Usage**: CPU, memory, disk

**Key Concepts:**
- **Traces**: Request journey
- **Spans**: Individual operations
- **Metrics**: Aggregated data
- **Logs**: Event records

**APM Tools:**
- **New Relic**: Full-stack APM
- **Datadog**: Infrastructure and APM
- **Prometheus**: Metrics collection
- **Jaeger**: Distributed tracing
- **Zipkin**: Distributed tracing

**Best Practices:**
- Monitor key transactions
- Set up alerts
- Track business metrics
- Use distributed tracing
- Correlate metrics and logs`,
					CodeExamples: `# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

requests_total = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'Request duration')
active_connections = Gauge('active_connections', 'Active connections')

@app.route('/api/users')
def get_users():
    start_time = time.time()
    requests_total.labels(method='GET', endpoint='/api/users').inc()
    active_connections.inc()
    
    try:
        # Process request
        result = fetch_users()
        return jsonify(result)
    finally:
        request_duration.observe(time.time() - start_time)
        active_connections.dec()

# Distributed tracing (OpenTelemetry)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

def process_order(order_id):
    with tracer.start_as_current_span("process_order") as span:
        span.set_attribute("order_id", order_id)
        # Process order
        with tracer.start_as_current_span("validate_payment"):
            # Validate payment
            pass
        with tracer.start_as_current_span("update_inventory"):
            # Update inventory
            pass

# Prometheus query examples
rate(http_requests_total[5m])
histogram_quantile(0.95, http_request_duration_seconds_bucket)
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))`,
				},
				{
					Title: "APM Implementation",
					Content: `Implementing APM in applications and infrastructure.

**APM Implementation Steps:**
1. **Instrumentation**: Add APM agents/libraries
2. **Configuration**: Set up APM tools
3. **Metrics Collection**: Define what to measure
4. **Dashboards**: Create visualizations
5. **Alerts**: Set up notifications
6. **Analysis**: Regular review and optimization

**Instrumentation Methods:**
- **Agent-based**: Install APM agents
- **Library-based**: Use APM libraries
- **Service Mesh**: Automatic instrumentation
- **eBPF**: Kernel-level tracing
- **OpenTelemetry**: Standard observability

**Key Metrics to Track:**
- **Application Metrics**: Response time, throughput, errors
- **Business Metrics**: User actions, conversions
- **Infrastructure Metrics**: CPU, memory, disk, network
- **Custom Metrics**: Domain-specific measurements

**APM Tools:**
- **New Relic**: Full-stack APM
- **Datadog**: Infrastructure and APM
- **AppDynamics**: Enterprise APM
- **Dynatrace**: AI-powered APM
- **Prometheus + Grafana**: Open-source stack
- **Jaeger/Zipkin**: Distributed tracing

**Best Practices:**
- Start with key transactions
- Monitor end-to-end flows
- Set up alerts early
- Track business metrics
- Use distributed tracing
- Correlate metrics and logs
- Regular performance reviews

**Common Pitfalls:**
- Over-instrumentation
- Not tracking business metrics
- Ignoring infrastructure metrics
- Not setting up alerts
- Poor dashboard design
- Not reviewing regularly`,
					CodeExamples: `# New Relic instrumentation (Node.js)
const newrelic = require('newrelic');

// Automatic instrumentation with environment variable
// NEW_RELIC_LICENSE_KEY=your-license-key
// NEW_RELIC_APP_NAME=MyApp

// Custom instrumentation
newrelic.startWebTransaction('/api/users', function() {
    // Your code here
    newrelic.endTransaction();
});

// Custom metrics
newrelic.recordMetric('Custom/UserSignups', 1);
newrelic.recordMetric('Custom/Revenue', 100.50);

# Datadog APM (Python)
from ddtrace import patch_all
patch_all()

from flask import Flask
app = Flask(__name__)

@app.route('/api/users')
def get_users():
    # Automatically traced
    return {'users': []}

# Custom span
from ddtrace import tracer

with tracer.trace('custom.operation') as span:
    span.set_tag('user.id', 123)
    # Your code here

# Prometheus instrumentation (Go)
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )
    
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds",
            Help: "HTTP request duration",
        },
        []string{"method", "endpoint"},
    )
)

func init() {
    prometheus.MustRegister(httpRequestsTotal)
    prometheus.MustRegister(httpRequestDuration)
}

func handler(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    
    // Your handler code
    
    duration := time.Since(start).Seconds()
    httpRequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration)
    httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, "200").Inc()
}

# OpenTelemetry instrumentation (Python)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)

@app.route('/api/users')
def get_users():
    with tracer.start_as_current_span("get_users") as span:
        span.set_attribute("user.count", len(users))
        return {'users': users}

# APM dashboard query (Prometheus)
# Average response time by endpoint
avg(rate(http_request_duration_seconds_sum[5m])) / avg(rate(http_request_duration_seconds_count[5m]))

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Alert rule
groups:
  - name: apm_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) / 
          sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        annotations:
          summary: "Error rate above 5%"
      
      - alert: SlowResponseTime
        expr: |
          histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        annotations:
          summary: "P95 response time above 1s"`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
