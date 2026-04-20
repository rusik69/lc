package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1251,
			Title:       "Azure Kubernetes Service (AKS)",
			Description: "Master Azure Kubernetes Service including cluster creation, networking, scaling, monitoring, and production best practices.",
			Order:       51,
			Lessons: []problems.Lesson{
				{
					Title: "AKS Cluster Management",
					Content: `Azure Kubernetes Service provides a managed Kubernetes control plane with integrated Azure services for networking, identity, monitoring, and scaling.

**AKS Architecture:**
` + "```" + `
AKS components:
  Control Plane (Azure managed, free):
    - API Server
    - etcd
    - Scheduler
    - Controller Manager
    - Cloud Controller Manager
  
  Node Pools:
    System pool:  Required, runs system pods (CoreDNS, metrics-server)
    User pools:   Optional, for application workloads
    
  Networking:
    kubenet:       Basic, Azure-managed, pod IPs from separate space
    Azure CNI:     Pods get VNet IPs, full VNet integration
    Azure CNI Overlay: Pods use overlay network, VNet for nodes
    Cilium:        eBPF-based networking (preview)

Create AKS cluster:
  # Basic cluster
  az aks create \
    --resource-group myRG \
    --name myAKS \
    --node-count 3 \
    --node-vm-size Standard_D4s_v5 \
    --network-plugin azure \
    --network-policy calico \
    --generate-ssh-keys \
    --zones 1 2 3 \
    --enable-managed-identity \
    --enable-cluster-autoscaler \
    --min-count 3 --max-count 10 \
    --kubernetes-version 1.29
  
  # Production cluster with advanced options
  az aks create \
    --resource-group myRG \
    --name myAKS-prod \
    --node-count 3 \
    --node-vm-size Standard_D4s_v5 \
    --nodepool-name system \
    --network-plugin azure \
    --network-policy calico \
    --vnet-subnet-id /subscriptions/.../subnets/aks-subnet \
    --service-cidr 172.16.0.0/16 \
    --dns-service-ip 172.16.0.10 \
    --load-balancer-sku standard \
    --zones 1 2 3 \
    --enable-managed-identity \
    --enable-cluster-autoscaler \
    --min-count 3 --max-count 10 \
    --enable-addons monitoring \
    --workspace-resource-id /subscriptions/.../workspaces/myWorkspace \
    --enable-defender \
    --enable-azure-rbac \
    --enable-aad \
    --aad-admin-group-object-ids "<group-id>" \
    --tier standard \
    --uptime-sla

  # Connect to cluster
  az aks get-credentials --resource-group myRG --name myAKS
  kubectl get nodes
` + "```" + `

**Node Pools:**
` + "```" + `
Node pools allow different VM types for different workloads.

Add node pool:
  # General workload pool
  az aks nodepool add \
    --resource-group myRG --cluster-name myAKS \
    --name apppool \
    --node-count 3 \
    --node-vm-size Standard_D4s_v5 \
    --zones 1 2 3 \
    --enable-cluster-autoscaler \
    --min-count 3 --max-count 20 \
    --mode User \
    --labels workload=app tier=standard \
    --node-taints ""
  
  # GPU pool
  az aks nodepool add \
    --resource-group myRG --cluster-name myAKS \
    --name gpupool \
    --node-count 0 \
    --node-vm-size Standard_NC6s_v3 \
    --enable-cluster-autoscaler \
    --min-count 0 --max-count 5 \
    --mode User \
    --labels workload=gpu \
    --node-taints "nvidia.com/gpu=present:NoSchedule"
  
  # Spot (preemptible) pool
  az aks nodepool add \
    --resource-group myRG --cluster-name myAKS \
    --name spotpool \
    --node-count 0 \
    --node-vm-size Standard_D4s_v5 \
    --priority Spot --spot-max-price -1 \
    --eviction-policy Delete \
    --enable-cluster-autoscaler \
    --min-count 0 --max-count 20 \
    --mode User \
    --labels workload=batch \
    --node-taints "kubernetes.azure.com/scalesetpriority=spot:NoSchedule"

Scale node pool:
  az aks nodepool scale \
    --resource-group myRG --cluster-name myAKS \
    --name apppool --node-count 5

Upgrade node pool:
  # Check available versions
  az aks get-upgrades --resource-group myRG --name myAKS --output table
  
  # Upgrade control plane first
  az aks upgrade --resource-group myRG --name myAKS --kubernetes-version 1.30
  
  # Then upgrade node pools
  az aks nodepool upgrade \
    --resource-group myRG --cluster-name myAKS \
    --name apppool --kubernetes-version 1.30
` + "```" + `

**AKS Networking:**
` + "```" + `
Ingress options:
  NGINX Ingress Controller:
    - Most popular, community supported
    - L7 load balancing
    
  Application Gateway Ingress Controller (AGIC):
    - Azure Application Gateway integration
    - WAF support
    - Azure-native

  Azure Service Mesh (Istio):
    - Built-in service mesh addon
    
Internal load balancer:
  apiVersion: v1
  kind: Service
  metadata:
    name: internal-app
    annotations:
      service.beta.kubernetes.io/azure-load-balancer-internal: "true"
      service.beta.kubernetes.io/azure-load-balancer-internal-subnet: "app-subnet"
  spec:
    type: LoadBalancer
    ports:
    - port: 80
    selector:
      app: myapp

Private cluster:
  az aks create \
    --resource-group myRG --name myAKS-private \
    --enable-private-cluster \
    --private-dns-zone system \
    --enable-managed-identity
  
  # API server is only accessible from VNet
  # Use az aks command invoke for management:
  az aks command invoke \
    --resource-group myRG --name myAKS-private \
    --command "kubectl get nodes"

Network Policy:
  # Deny all ingress by default
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata:
    name: deny-all
    namespace: production
  spec:
    podSelector: {}
    policyTypes:
    - Ingress
    ingress: []
  
  # Allow specific traffic
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata:
    name: allow-web-to-api
    namespace: production
  spec:
    podSelector:
      matchLabels:
        app: api
    ingress:
    - from:
      - podSelector:
          matchLabels:
            app: web
      ports:
      - port: 8080
` + "```" + ``,
					CodeExamples: `# AKS management scripts

# 1. AKS cluster health check
#!/bin/bash
RG="${1:?Usage: $0 <resource-group> <cluster-name>}"
CLUSTER="${2:?Usage: $0 <resource-group> <cluster-name>}"

echo "=== AKS Health Check: $CLUSTER ==="

# Cluster overview
az aks show -g "$RG" -n "$CLUSTER" --query "{
    version:kubernetesVersion, 
    state:provisioningState,
    tier:sku.tier,
    networkPlugin:networkProfile.networkPlugin,
    networkPolicy:networkProfile.networkPolicy
}" -o json | jq .

# Node pools
echo ""
echo "--- Node Pools ---"
az aks nodepool list -g "$RG" --cluster-name "$CLUSTER" --query "[].{
    name:name, count:count, vmSize:vmSize,
    mode:mode, version:orchestratorVersion,
    autoscale:enableAutoScaling, min:minCount, max:maxCount
}" -o table

# Get credentials and check nodes
az aks get-credentials -g "$RG" -n "$CLUSTER" --overwrite-existing 2>/dev/null

echo ""
echo "--- Node Status ---"
kubectl get nodes -o wide 2>/dev/null

echo ""
echo "--- Pod Status ---"
NOT_RUNNING=$(kubectl get pods -A --field-selector=status.phase!=Running,status.phase!=Succeeded 2>/dev/null | tail -n+2)
if [ -n "$NOT_RUNNING" ]; then
    echo "Unhealthy pods:"
    echo "$NOT_RUNNING"
else
    echo "All pods healthy"
fi

echo ""
echo "--- Resource Usage ---"
kubectl top nodes 2>/dev/null || echo "Metrics server not available"

# 2. AKS cost optimization
#!/bin/bash
echo "=== AKS Cost Optimization Report ==="

for rg in $(az group list --query "[].name" -o tsv); do
    CLUSTERS=$(az aks list -g "$rg" --query "[].name" -o tsv 2>/dev/null)
    
    for cluster in $CLUSTERS; do
        echo ""
        echo "--- Cluster: $cluster (RG: $rg) ---"
        
        # Check for overprovisioned pools
        az aks nodepool list -g "$rg" --cluster-name "$cluster" \
            --query "[].{
                name:name, vmSize:vmSize, count:count,
                autoscale:enableAutoScaling, minCount:minCount, maxCount:maxCount
            }" -o table 2>/dev/null
        
        # Check for spot nodes
        SPOT_POOLS=$(az aks nodepool list -g "$rg" --cluster-name "$cluster" \
            --query "[?scaleSetPriority=='Spot'].name" -o tsv 2>/dev/null)
        
        if [ -z "$SPOT_POOLS" ]; then
            echo "  TIP: No spot node pools configured. Consider spot for batch workloads."
        fi
        
        # Check tier
        TIER=$(az aks show -g "$rg" -n "$cluster" --query "sku.tier" -o tsv 2>/dev/null)
        if [ "$TIER" = "Free" ]; then
            echo "  NOTE: Free tier (no SLA). Consider Standard for production."
        fi
    done
done

# 3. AKS upgrade planner
#!/bin/bash
echo "=== AKS Upgrade Status ==="

for rg in $(az group list --query "[].name" -o tsv); do
    CLUSTERS=$(az aks list -g "$rg" --query "[].name" -o tsv 2>/dev/null)
    
    for cluster in $CLUSTERS; do
        CURRENT=$(az aks show -g "$rg" -n "$cluster" \
            --query "kubernetesVersion" -o tsv 2>/dev/null)
        
        echo ""
        echo "Cluster: $cluster (RG: $rg) - Current: $CURRENT"
        
        # Available upgrades
        UPGRADES=$(az aks get-upgrades -g "$rg" -n "$cluster" \
            --query "controlPlaneProfile.upgrades[].kubernetesVersion" -o tsv 2>/dev/null)
        
        if [ -n "$UPGRADES" ]; then
            echo "  Available upgrades: $UPGRADES"
        else
            echo "  Up to date (or at latest version)"
        fi
        
        # Node pool versions
        az aks nodepool list -g "$rg" --cluster-name "$cluster" \
            --query "[].{name:name, version:orchestratorVersion}" \
            -o table 2>/dev/null
    done
done`,
				},
				{
					Title: "AKS Security and Monitoring",
					Content: `Securing AKS clusters and implementing comprehensive monitoring are essential for production workloads.

**AKS Security:**
` + "```" + `
Azure AD integration:
  - AKS-managed Azure AD (recommended)
  - Azure RBAC for Kubernetes
  - Kubernetes RBAC with Azure AD groups
  
  # Cluster admin via Azure AD group
  az aks update -g myRG -n myAKS \
    --aad-admin-group-object-ids "<group-id>"
  
  # Azure RBAC roles for Kubernetes
  Azure Kubernetes Service RBAC Admin
  Azure Kubernetes Service RBAC Cluster Admin
  Azure Kubernetes Service RBAC Reader
  Azure Kubernetes Service RBAC Writer
  
  # Assign role
  az role assignment create \
    --assignee user@example.com \
    --role "Azure Kubernetes Service RBAC Writer" \
    --scope /subscriptions/.../managedClusters/myAKS/namespaces/myapp

Workload Identity (replacing pod identity):
  # Enable workload identity
  az aks update -g myRG -n myAKS \
    --enable-oidc-issuer \
    --enable-workload-identity
  
  # Create managed identity
  az identity create -g myRG -n myapp-identity
  
  # Create federated credential
  az identity federated-credential create \
    -g myRG --identity-name myapp-identity \
    --name myapp-fed-cred \
    --issuer $(az aks show -g myRG -n myAKS --query "oidcIssuerProfile.issuerUrl" -o tsv) \
    --subject "system:serviceaccount:myapp:myapp-sa" \
    --audience "api://AzureADTokenExchange"
  
  # Use in pod
  apiVersion: v1
  kind: ServiceAccount
  metadata:
    name: myapp-sa
    namespace: myapp
    annotations:
      azure.workload.identity/client-id: "<managed-identity-client-id>"
  ---
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: myapp
    namespace: myapp
  spec:
    template:
      metadata:
        labels:
          azure.workload.identity/use: "true"
      spec:
        serviceAccountName: myapp-sa

Azure Policy for AKS:
  # Enable Azure Policy addon
  az aks enable-addons -g myRG -n myAKS --addons azure-policy
  
  Built-in policies:
  - Do not allow privileged containers
  - Enforce HTTPS ingress
  - Ensure containers use allowed images only
  - Enforce resource limits on containers
  - Restrict host networking and ports

Microsoft Defender for Containers:
  az aks update -g myRG -n myAKS --enable-defender
  
  Features:
  - Vulnerability scanning of images
  - Runtime threat detection
  - Security recommendations
  - Behavioral analytics
` + "```" + `

**AKS Monitoring:**
` + "```" + `
Azure Monitor Container Insights:
  # Enable monitoring
  az aks enable-addons -g myRG -n myAKS \
    --addons monitoring \
    --workspace-resource-id /subscriptions/.../workspaces/myWorkspace
  
  # or update existing cluster
  az aks update -g myRG -n myAKS \
    --enable-addons monitoring \
    --workspace-resource-id /subscriptions/.../workspaces/myWorkspace

Key metrics:
  Cluster level:
    - Node CPU/memory utilization
    - Node count and conditions
    - Pod count per node
  
  Pod level:
    - Container CPU/memory usage vs limits
    - Container restart count
    - Pod phase (Pending, Running, Failed)
  
  Application level:
    - Request rate, latency, errors
    - Custom metrics via Application Insights

Log queries (KQL):
  // Pod logs for specific app
  ContainerLogV2
  | where PodName startswith "myapp"
  | where LogMessage contains "error"
  | project TimeGenerated, PodName, LogMessage
  | order by TimeGenerated desc
  | take 100
  
  // Node resource usage
  KubeNodeInventory
  | where ClusterName == "myAKS"
  | summarize avg(CPUCapacityNanoCores), avg(MemoryCapacityBytes) by Computer
  
  // OOM killed containers
  KubeEvents
  | where Reason == "OOMKilling"
  | project TimeGenerated, Name, Message

Prometheus + Grafana (Azure Managed):
  # Enable Azure Monitor managed Prometheus
  az aks update -g myRG -n myAKS \
    --enable-azure-monitor-metrics
  
  # Create Azure Managed Grafana
  az grafana create -g myRG -n myGrafana
  
  # Link to Prometheus workspace
  az grafana data-source create \
    -n myGrafana \
    --definition '{
      "name": "Azure Monitor",
      "type": "grafana-azure-monitor-datasource"
    }'

Alerting:
  # High CPU alert
  az monitor metrics alert create \
    --resource-group myRG \
    --name "AKS High CPU" \
    --scopes /subscriptions/.../managedClusters/myAKS \
    --condition "avg node_cpu_usage_percentage > 80" \
    --window-size 5m \
    --evaluation-frequency 1m
  
  # Pod restart alert
  az monitor scheduled-query create \
    --resource-group myRG \
    --name "Pod Restarts" \
    --scopes /subscriptions/.../workspaces/myWorkspace \
    --condition "count > 5" \
    --condition-query "KubePodInventory | where ClusterName == 'myAKS' | where PodRestartCount > 3" \
    --evaluation-frequency 5m --window-size 15m
` + "```" + ``,
					CodeExamples: `# AKS security and monitoring

# 1. AKS security audit
#!/bin/bash
RG="${1:?Usage: $0 <resource-group> <cluster-name>}"
CLUSTER="${2:?Usage: $0 <resource-group> <cluster-name>}"

echo "=== AKS Security Audit: $CLUSTER ==="

# Check Azure AD integration
AAD=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "aadProfile.managed" -o tsv 2>/dev/null)
echo "Azure AD: ${AAD:-not configured}"

# Check RBAC
RBAC=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "enableRbac" -o tsv 2>/dev/null)
echo "RBAC: ${RBAC:-unknown}"

# Check Azure RBAC
AZURE_RBAC=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "aadProfile.enableAzureRbac" -o tsv 2>/dev/null)
echo "Azure RBAC: ${AZURE_RBAC:-not configured}"

# Check network policy
NET_POLICY=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "networkProfile.networkPolicy" -o tsv 2>/dev/null)
echo "Network Policy: ${NET_POLICY:-none (WARNING)}"

# Check private cluster
PRIVATE=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "apiServerAccessProfile.enablePrivateCluster" -o tsv 2>/dev/null)
echo "Private Cluster: ${PRIVATE:-false}"

# Check Defender
DEFENDER=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "securityProfile.defender.securityMonitoring.enabled" -o tsv 2>/dev/null)
echo "Defender: ${DEFENDER:-not configured}"

# Check Azure Policy
POLICY=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "addonProfiles.azurepolicy.enabled" -o tsv 2>/dev/null)
echo "Azure Policy: ${POLICY:-not configured}"

# Check monitoring
MONITORING=$(az aks show -g "$RG" -n "$CLUSTER" \
    --query "addonProfiles.omsagent.enabled" -o tsv 2>/dev/null)
echo "Monitoring: ${MONITORING:-not configured}"

# Kubernetes-level checks
az aks get-credentials -g "$RG" -n "$CLUSTER" --overwrite-existing 2>/dev/null

echo ""
echo "--- Privileged Pods ---"
kubectl get pods -A -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.containers[].securityContext.privileged==true) | 
    "\(.metadata.namespace)/\(.metadata.name)"'

echo ""
echo "--- Pods Running as Root ---"
kubectl get pods -A -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.containers[].securityContext.runAsUser==0 or 
    (.spec.securityContext.runAsUser==0)) | 
    "\(.metadata.namespace)/\(.metadata.name)"'

# 2. AKS monitoring dashboard
#!/bin/bash
RG="${1:?Usage: $0 <resource-group> <cluster-name>}"
CLUSTER="${2:?Usage: $0 <resource-group> <cluster-name>}"

az aks get-credentials -g "$RG" -n "$CLUSTER" --overwrite-existing 2>/dev/null

echo "=== AKS Monitoring: $CLUSTER ==="

echo "--- Node Resources ---"
kubectl top nodes 2>/dev/null || echo "Metrics unavailable"

echo ""
echo "--- Top CPU Pods ---"
kubectl top pods -A --sort-by=cpu 2>/dev/null | head -15

echo ""
echo "--- Top Memory Pods ---"
kubectl top pods -A --sort-by=memory 2>/dev/null | head -15

echo ""
echo "--- Recent Events ---"
kubectl get events -A --sort-by='.lastTimestamp' 2>/dev/null | \
    grep -E "Warning|Error" | tail -20

echo ""
echo "--- Pending Pods ---"
kubectl get pods -A --field-selector=status.phase=Pending 2>/dev/null

echo ""
echo "--- CrashLoopBackOff Pods ---"
kubectl get pods -A 2>/dev/null | grep CrashLoopBackOff`,
				},
			},
		},
	})
}
