package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          420,
			Title:       "Azure Kubernetes Service (AKS)",
			Description: "Learn AKS: managed Kubernetes, clusters, pods, services, and container orchestration.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "AKS Fundamentals",
					Content: `Azure Kubernetes Service (AKS) is a managed Kubernetes service for deploying containerized applications.

**AKS Features:**
- Managed Kubernetes control plane
- Automatic upgrades and patching
- Integrated monitoring and logging
- Azure AD integration
- Virtual node support for serverless containers

**Cluster Components:**
- **Control Plane**: Managed by Azure
- **Node Pools**: Worker nodes running containers
- **Pods**: Smallest deployable units
- **Services**: Stable network endpoints
- **Deployments**: Manage pod replicas

**Node Pool Types:**
- **System Node Pool**: System pods
- **User Node Pool**: Application pods
- **Spot Node Pool**: Cost-optimized nodes
- **Windows Node Pool**: Windows containers

**Scaling:**
- **Manual Scaling**: Set node count
- **Cluster Autoscaler**: Automatic node scaling
- **Horizontal Pod Autoscaler**: Automatic pod scaling
- **Vertical Pod Autoscaler**: Automatic resource adjustment

**Networking:**
- **kubenet**: Basic networking
- **Azure CNI**: Advanced networking with VNet integration
- **Network Policies**: Pod-to-pod communication rules
- **Ingress Controllers**: External traffic routing

**Best Practices:**
- Use multiple node pools for different workloads
- Enable cluster autoscaler
- Use Azure AD for RBAC
- Implement network policies
- Monitor cluster health`,
					CodeExamples: `# Create AKS cluster
az aks create \\
    --resource-group myResourceGroup \\
    --name myAKSCluster \\
    --node-count 3 \\
    --enable-addons monitoring \\
    --generate-ssh-keys \\
    --network-plugin azure \\
    --network-policy azure

# Get cluster credentials
az aks get-credentials \\
    --resource-group myResourceGroup \\
    --name myAKSCluster

# Create node pool
az aks nodepool add \\
    --resource-group myResourceGroup \\
    --cluster-name myAKSCluster \\
    --name mynodepool \\
    --node-count 2 \\
    --node-vm-size Standard_DS2_v2

# Enable cluster autoscaler
az aks update \\
    --resource-group myResourceGroup \\
    --name myAKSCluster \\
    --enable-cluster-autoscaler \\
    --min-count 1 \\
    --max-count 10

# Scale cluster
az aks scale \\
    --resource-group myResourceGroup \\
    --name myAKSCluster \\
    --node-count 5`,
				},
				{
					Title: "Ingress Controllers",
					Content: `Ingress controllers provide HTTP/HTTPS routing and load balancing for AKS clusters.

**Ingress Concepts:**
- **Ingress Resource**: Kubernetes resource defining routing rules
- **Ingress Controller**: Controller implementing ingress rules
- **TLS Termination**: SSL/TLS termination at ingress
- **Path-based Routing**: Route based on URL path
- **Host-based Routing**: Route based on host header

**Ingress Controllers:**
- **NGINX Ingress**: Popular open-source controller
- **Application Gateway Ingress**: Azure Application Gateway integration
- **Traefik**: Modern ingress controller

**Application Gateway Ingress Controller (AGIC):**
- **Azure Integration**: Native Azure integration
- **WAF Integration**: Web Application Firewall support
- **SSL Termination**: Centralized SSL management
- **Health Probes**: Automatic health checking

**Best Practices:**
- Use Application Gateway for Azure-native features
- Configure TLS certificates properly
- Use path-based routing for microservices
- Monitor ingress metrics
- Implement proper health checks`,
					CodeExamples: `# Install NGINX ingress controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx

# Example ingress resource
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  tls:
  - hosts:
    - myapp.contoso.com
    secretName: my-tls-secret
  rules:
  - host: myapp.contoso.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: 80`,
				},
				{
					Title: "Pod Security Policies",
					Content: `Pod Security Policies control security-sensitive aspects of pod specification.

**PSP Features:**
- **Privileged Containers**: Control privileged container usage
- **Host Namespaces**: Access to host network, PID, IPC
- **Volumes**: Allowed volume types
- **Capabilities**: Linux capabilities
- **SELinux**: SELinux context
- **RunAsUser**: User and group IDs

**Note**: Pod Security Policies are deprecated in Kubernetes 1.21+. Use Pod Security Standards instead.

**Pod Security Standards:**
- **Privileged**: Unrestricted (default)
- **Baseline**: Minimally restrictive
- **Restricted**: Highly restrictive

**Best Practices:**
- Use Pod Security Standards (not PSP)
- Apply security standards at namespace level
- Start with baseline, move to restricted
- Test applications with security standards
- Document security requirements`,
					CodeExamples: `# Enable Pod Security Standards (namespace level)
apiVersion: v1
kind: Namespace
metadata:
  name: my-namespace
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          421,
			Title:       "Azure DevOps",
			Description: "Learn Azure DevOps: CI/CD pipelines, repositories, boards, and DevOps practices.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Azure DevOps Fundamentals",
					Content: `Azure DevOps provides a complete DevOps toolchain for software development and delivery.

**Services:**
- **Azure Repos**: Git repositories
- **Azure Pipelines**: CI/CD pipelines
- **Azure Boards**: Work item tracking
- **Azure Artifacts**: Package management
- **Azure Test Plans**: Test management

**Pipelines:**
- **YAML Pipelines**: Code-based pipeline definitions
- **Classic Pipelines**: Visual pipeline designer
- **Multi-stage Pipelines**: Complex deployment workflows
- **Release Pipelines**: Multi-environment deployments

**Agents:**
- **Microsoft-hosted Agents**: Pre-configured agents
- **Self-hosted Agents**: Custom agents on your infrastructure
- **Agent Pools**: Group agents for organization

**Artifacts:**
- **NuGet**: .NET packages
- **npm**: Node.js packages
- **Maven**: Java packages
- **PyPI**: Python packages
- **Universal Packages**: Any package type

**Best Practices:**
- Use YAML pipelines for version control
- Implement multi-stage pipelines
- Use variable groups for secrets
- Enable branch policies
- Implement code reviews`,
					CodeExamples: `# Create Azure DevOps project
az devops project create \\
    --name myProject \\
    --organization https://dev.azure.com/myorg

# Example YAML pipeline
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

stages:
- stage: Build
  jobs:
  - job: BuildJob
    steps:
    - task: NodeTool@0
      inputs:
        versionSpec: '14.x'
    - script: |
        npm install
        npm run build
      displayName: 'Build application'

- stage: Deploy
  jobs:
  - deployment: DeployJob
    environment: production
    strategy:
      runOnce:
        deploy:
          steps:
          - script: echo "Deploying application"
            displayName: 'Deploy'`,
				},
				{
					Title: "Pipeline Best Practices",
					Content: `Following best practices ensures reliable and maintainable CI/CD pipelines.

**Pipeline Design:**
- **Modularity**: Break into reusable templates
- **Idempotency**: Safe to run multiple times
- **Fast Feedback**: Quick build and test cycles
- **Parallel Execution**: Run tasks in parallel when possible

**Security:**
- **Secrets Management**: Use variable groups and Key Vault
- **Least Privilege**: Minimal permissions for service connections
- **Scanning**: Security scanning in pipelines
- **Approvals**: Require approvals for production

**Best Practices:**
- Use YAML pipelines for version control
- Implement proper error handling
- Use pipeline templates
- Cache dependencies
- Run tests in parallel
- Use feature flags
- Monitor pipeline metrics`,
					CodeExamples: `# Example pipeline with best practices
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
- group: my-variable-group
- name: buildConfiguration
  value: 'Release'

stages:
- stage: Build
  jobs:
  - job: BuildJob
    steps:
    - task: UseNode@1
      inputs:
        version: '14.x'
    - script: |
        npm ci
        npm run build
      displayName: 'Build'
    - task: PublishTestResults@2
      condition: succeededOrFailed()`,
				},
				{
					Title: "Artifact Management",
					Content: `Azure Artifacts provides package management for software development.

**Supported Package Types:**
- **NuGet**: .NET packages
- **npm**: Node.js packages
- **Maven**: Java packages
- **PyPI**: Python packages
- **Universal Packages**: Any package type

**Artifact Feeds:**
- **Organization Feeds**: Shared across organization
- **Project Feeds**: Scoped to project
- **Public Feeds**: Public packages from npm, NuGet, etc.

**Best Practices:**
- Use organization feeds for shared packages
- Publish packages as part of CI/CD
- Version packages semantically
- Use upstream sources for public packages
- Secure feeds appropriately`,
					CodeExamples: `# Publish npm package
npm publish --registry https://pkgs.dev.azure.com/myorg/myproject/_packaging/myfeed/npm/registry/

# Install from feed
npm install --registry https://pkgs.dev.azure.com/myorg/myproject/_packaging/myfeed/npm/registry/`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          422,
			Title:       "Advanced Azure Networking",
			Description: "Learn advanced networking: VNet peering, VPN Gateway, ExpressRoute, and Virtual WAN.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Networking Concepts",
					Content: `Advanced Azure networking enables complex hybrid and multi-cloud architectures.

**VNet Peering:**
- Connect VNets in same or different regions
- Private IP communication
- Non-transitive by default
- Global peering across regions

**VPN Gateway:**
- **Site-to-Site VPN**: Connect on-premises networks
- **Point-to-Site VPN**: Connect individual users
- **VNet-to-VNet VPN**: Connect Azure VNets
- **Active-Active**: High availability configuration

**ExpressRoute:**
- Private connection to Azure
- Bypass internet
- Consistent network performance
- Multiple connectivity providers
- **ExpressRoute Global Reach**: Connect ExpressRoute circuits

**Virtual WAN:**
- Centralized network management
- Hub-spoke architecture
- Integrated security
- Branch connectivity
- SD-WAN integration

**Best Practices:**
- Use Virtual WAN for complex networks
- Implement hub-spoke architecture
- Use ExpressRoute for production workloads
- Plan IP address spaces carefully
- Monitor network performance`,
					CodeExamples: `# Create VNet peering
az network vnet peering create \\
    --resource-group myResourceGroup \\
    --name myPeering \\
    --vnet-name myVNet1 \\
    --remote-vnet myVNet2 \\
    --allow-vnet-access

# Create VPN gateway
az network vnet-gateway create \\
    --resource-group myResourceGroup \\
    --name myVPNGateway \\
    --vnet myVNet \\
    --public-ip-address myGatewayIP \\
    --gateway-type Vpn \\
    --vpn-type RouteBased \\
    --sku VpnGw1

# Create ExpressRoute circuit
az network express-route create \\
    --resource-group myResourceGroup \\
    --name myExpressRoute \\
    --bandwidth 1000 \\
    --peering-location "Silicon Valley" \\
    --provider "Equinix" \\
    --peering-type AzurePrivatePeering

# Create Virtual WAN
az network vwan create \\
    --resource-group myResourceGroup \\
    --name myVirtualWAN \\
    --location eastus`,
				},
				{
					Title: "Virtual WAN",
					Content: `Virtual WAN provides centralized network management and connectivity.

**Virtual WAN Benefits:**
- **Centralized Management**: Single management interface
- **Hub-Spoke Architecture**: Scalable network topology
- **Branch Connectivity**: Connect branches easily
- **SD-WAN Integration**: Integrate with SD-WAN vendors

**Virtual WAN Hubs:**
- **Regional Hubs**: Deploy in Azure regions
- **Hub Types**: Standard or Basic
- **Services**: VPN, ExpressRoute, Point-to-site VPN

**Best Practices:**
- Use Virtual WAN for complex networks
- Deploy hubs in strategic regions
- Use hub-spoke architecture
- Integrate with SD-WAN for branches`,
					CodeExamples: `# Create Virtual WAN hub
az network vwan create \\
    --resource-group myResourceGroup \\
    --name myVirtualWAN \\
    --location eastus

# Create hub
az network vhub create \\
    --resource-group myResourceGroup \\
    --name myHub \\
    --address-prefix 10.0.0.0/24 \\
    --vwan myVirtualWAN \\
    --location eastus`,
				},
				{
					Title: "ExpressRoute",
					Content: `ExpressRoute provides private connectivity to Azure.

**ExpressRoute Benefits:**
- **Private Connection**: Bypass internet
- **Consistent Performance**: Predictable network performance
- **Higher Bandwidth**: Up to 100 Gbps
- **Dual Redundancy**: Redundant connections

**ExpressRoute Circuits:**
- **Provider**: Connectivity provider (Equinix, AT&T, etc.)
- **Peering Locations**: Physical locations
- **Bandwidth**: Circuit bandwidth
- **SKU**: Standard or Premium

**Peering Types:**
- **Azure Private Peering**: Connect to Azure VNets
- **Microsoft Peering**: Connect to Microsoft services
- **Public Peering**: Deprecated

**Best Practices:**
- Use ExpressRoute for production workloads
- Implement redundant circuits
- Monitor circuit health
- Use ExpressRoute Global Reach for multi-region`,
					CodeExamples: `# Create ExpressRoute circuit
az network express-route create \\
    --resource-group myResourceGroup \\
    --name myExpressRoute \\
    --bandwidth 1000 \\
    --peering-location "Silicon Valley" \\
    --provider "Equinix" \\
    --peering-type AzurePrivatePeering`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          423,
			Title:       "Azure Security Center",
			Description: "Learn Security Center: threat protection, security recommendations, and compliance management.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Security Center Fundamentals",
					Content: `Azure Security Center provides unified security management and threat protection.

**Features:**
- **Security Posture**: Continuous assessment
- **Threat Protection**: Advanced threat detection
- **Compliance**: Regulatory compliance monitoring
- **Security Recommendations**: Actionable security guidance
- **Just-in-Time Access**: Time-limited VM access

**Tiers:**
- **Free**: Basic security recommendations
- **Standard**: Advanced threat protection and compliance

**Threat Protection:**
- **Adaptive Application Controls**: Whitelist applications
- **Just-in-Time VM Access**: Reduce attack surface
- **Adaptive Network Hardening**: Network security recommendations
- **File Integrity Monitoring**: Detect file changes
- **Vulnerability Assessment**: Identify vulnerabilities

**Compliance:**
- **Regulatory Compliance**: Built-in compliance standards
- **Custom Compliance**: Define custom policies
- **Compliance Reports**: Track compliance status

**Best Practices:**
- Enable Standard tier for production
- Review and implement security recommendations
- Enable Just-in-Time access for VMs
- Use adaptive application controls
- Monitor security alerts`,
					CodeExamples: `# Enable Security Center
az security auto-provisioning-setting update \\
    --name "default" \\
    --auto-provision "On"

# Enable Just-in-Time access
az vm update \\
    --resource-group myResourceGroup \\
    --name myVM \\
    --set securityProfile.jitNetworkAccessPolicyEnabled=true

# Request JIT access
az security jit-policy create \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --name myVM \\
    --ports "22" \\
    --start-time "2023-01-01T00:00:00Z" \\
    --duration "PT3H" \\
    --source-addresses "*"

# Enable adaptive application controls
az security adaptive-application-controls show \\
    --resource-group myResourceGroup \\
    --name myGroup`,
				},
				{
					Title: "Threat Protection",
					Content: `Security Center provides advanced threat protection capabilities.

**Threat Protection Features:**
- **Adaptive Application Controls**: Whitelist applications
- **Just-in-Time VM Access**: Reduce attack surface
- **Adaptive Network Hardening**: Network security recommendations
- **File Integrity Monitoring**: Detect file changes
- **Vulnerability Assessment**: Identify vulnerabilities

**Threat Detection:**
- **Alerts**: Security alerts and recommendations
- **Threat Intelligence**: Known threat patterns
- **Behavioral Analysis**: Anomaly detection
- **Machine Learning**: AI-powered threat detection

**Best Practices:**
- Enable Standard tier for production
- Review security alerts regularly
- Implement adaptive application controls
- Use Just-in-Time access for VMs
- Enable file integrity monitoring`,
					CodeExamples: `# Enable Just-in-Time access
az vm update \\
    --resource-group myResourceGroup \\
    --name myVM \\
    --set securityProfile.jitNetworkAccessPolicyEnabled=true

# Request JIT access
az security jit-policy create \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --name myVM \\
    --ports "22" \\
    --start-time "2024-01-01T00:00:00Z" \\
    --duration "PT3H"`,
				},
				{
					Title: "Compliance",
					Content: `Security Center helps maintain regulatory compliance.

**Compliance Standards:**
- **PCI DSS**: Payment card industry compliance
- **ISO 27001**: Information security management
- **SOC 2**: Service organization control
- **Azure Security Benchmark**: Azure-specific security baseline

**Compliance Monitoring:**
- **Compliance Dashboard**: View compliance status
- **Regulatory Compliance**: Built-in compliance standards
- **Custom Compliance**: Define custom policies
- **Compliance Reports**: Export compliance reports

**Best Practices:**
- Enable regulatory compliance monitoring
- Review compliance regularly
- Address compliance gaps
- Export compliance reports
- Use Azure Policy for compliance`,
					CodeExamples: `# View compliance status
az security regulatory-compliance-standards list \\
    --resource-group myResourceGroup`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          424,
			Title:       "Azure Policy",
			Description: "Learn Azure Policy: governance, compliance, and resource management policies.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Policy Fundamentals",
					Content: `Azure Policy helps enforce organizational standards and assess compliance.

**Policy Types:**
- **Built-in Policies**: Pre-defined policies from Microsoft
- **Custom Policies**: Organization-specific policies
- **Policy Initiatives**: Groups of related policies

**Policy Effects:**
- **Audit**: Log non-compliant resources
- **Deny**: Prevent non-compliant resource creation
- **Modify**: Change resource properties
- **DeployIfNotExists**: Deploy resources if missing
- **Append**: Add properties to resources

**Policy Assignment:**
- **Management Groups**: Apply to multiple subscriptions
- **Subscriptions**: Apply to all resources in subscription
- **Resource Groups**: Apply to resources in group
- **Individual Resources**: Apply to specific resource

**Compliance:**
- **Compliance State**: Compliant or non-compliant
- **Compliance Reports**: Track compliance over time
- **Remediation**: Fix non-compliant resources

**Best Practices:**
- Start with audit effects
- Use initiatives for related policies
- Assign at management group level
- Enable remediation tasks
- Review compliance regularly`,
					CodeExamples: `# List built-in policies
az policy definition list --query "[?displayName=='Allowed locations']"

# Create custom policy
az policy definition create \\
    --name "require-tag" \\
    --display-name "Require tag" \\
    --description "Requires a specific tag" \\
    --rules policy.json \\
    --params policy-params.json

# Assign policy
az policy assignment create \\
    --name "require-tag-assignment" \\
    --display-name "Require tag assignment" \\
    --policy "require-tag" \\
    --params '{"tagName":{"value":"Environment"}}' \\
    --scope /subscriptions/<subscription-id>

# Check compliance
az policy state list \\
    --resource /subscriptions/<subscription-id> \\
    --filter "ComplianceState eq 'NonCompliant'"`,
				},
				{
					Title: "Policy Initiatives",
					Content: `Policy initiatives group multiple related policies together.

**Initiative Benefits:**
- **Grouping**: Related policies together
- **Simplified Management**: Manage as single unit
- **Compliance Standards**: Pre-built compliance initiatives
- **Custom Initiatives**: Create organization-specific initiatives

**Built-in Initiatives:**
- **Azure Security Benchmark**: Security best practices
- **CIS Microsoft Azure Foundations**: CIS benchmark
- **PCI DSS**: Payment card compliance
- **NIST**: NIST compliance standards

**Best Practices:**
- Use built-in initiatives when possible
- Create custom initiatives for organization needs
- Assign at management group level
- Review initiative compliance regularly`,
					CodeExamples: `# Assign built-in initiative
az policy assignment create \\
    --name "security-benchmark" \\
    --display-name "Azure Security Benchmark" \\
    --policy-set-definition "/providers/Microsoft.Authorization/policySetDefinitions/1f3afdf9-d0c9-4c3d-847f-89da613e70a8" \\
    --scope /subscriptions/<subscription-id>`,
				},
				{
					Title: "Remediation",
					Content: `Remediation tasks automatically fix non-compliant resources.

**Remediation Types:**
- **DeployIfNotExists**: Deploy missing resources
- **Modify**: Change resource properties
- **Append**: Add properties to resources

**Remediation Tasks:**
- **Manual Remediation**: Trigger remediation manually
- **Automatic Remediation**: Automatic remediation on creation
- **On-demand Remediation**: Remediate existing resources

**Best Practices:**
- Enable remediation for appropriate policies
- Test remediation in non-production
- Monitor remediation tasks
- Review remediation results`,
					CodeExamples: `# Create remediation task
az policy remediation create \\
    --name myRemediation \\
    --policy-assignment "require-tag-assignment" \\
    --resource-group myResourceGroup`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          425,
			Title:       "Azure Cost Management",
			Description: "Learn Cost Management: cost analysis, budgets, alerts, and optimization strategies.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Cost Management Fundamentals",
					Content: `Azure Cost Management helps monitor, analyze, and optimize cloud spending.

**Features:**
- **Cost Analysis**: Detailed cost breakdowns
- **Budgets**: Set spending limits and alerts
- **Cost Alerts**: Notifications for spending thresholds
- **Cost Recommendations**: Optimization suggestions
- **Reserved Instance Recommendations**: Savings opportunities

**Cost Views:**
- **By Resource**: Cost per resource
- **By Service**: Cost per Azure service
- **By Location**: Cost by region
- **By Tag**: Cost by resource tags
- **By Subscription**: Cost per subscription

**Budgets:**
- **Cost Budgets**: Based on cost amount
- **Usage Budgets**: Based on usage quantity
- **Multiple Budgets**: Different budgets for different scopes
- **Budget Alerts**: Email and action group notifications

**Optimization:**
- **Reserved Instances**: 1-3 year commitments for savings
- **Spot VMs**: Use unused capacity at lower cost
- **Right-sizing**: Match VM size to workload
- **Shutdown Unused Resources**: Identify and stop unused resources

**Best Practices:**
- Set up budgets and alerts
- Review cost analysis regularly
- Use tags for cost allocation
- Implement right-sizing recommendations
- Consider Reserved Instances for predictable workloads`,
					CodeExamples: `# View cost analysis
az consumption usage list \\
    --start-date 2023-01-01 \\
    --end-date 2023-01-31

# Create budget
az consumption budget create \\
    --budget-name myBudget \\
    --amount 1000 \\
    --time-grain Monthly \\
    --start-date 2023-01-01 \\
    --end-date 2023-12-31 \\
    --category Cost \\
    --resource-group myResourceGroup

# List reservations
az reservation list \\
    --query "[].{Name:name, State:properties.provisioningState}"

# Purchase reserved instance
az reservation order purchase \\
    --reserved-resource-type "VirtualMachines" \\
    --billing-scope-id /subscriptions/<subscription-id> \\
    --term "P1Y" \\
    --quantity 1 \\
    --display-name "MyReservation" \\
    --applied-scope-type "Single" \\
    --applied-scopes /subscriptions/<subscription-id> \\
    --sku Standard_B1s`,
				},
				{
					Title: "Reserved Instances",
					Content: `Reserved Instances provide significant cost savings for predictable workloads.

**Reserved Instance Benefits:**
- **Cost Savings**: Up to 72% savings vs pay-as-you-go
- **Commitment**: 1 or 3 year commitments
- **Flexibility**: Can exchange or cancel (with fees)
- **Scope**: Single subscription or shared

**Reserved Instance Types:**
- **Virtual Machines**: VM reserved instances
- **SQL Database**: Database reserved instances
- **Cosmos DB**: Cosmos DB reserved capacity
- **App Service**: App Service reserved instances

**Best Practices:**
- Use for predictable, steady-state workloads
- Analyze usage before purchasing
- Consider 3-year for maximum savings
- Use shared scope for flexibility
- Monitor utilization`,
					CodeExamples: `# Purchase reserved instance
az reservation order purchase \\
    --reserved-resource-type "VirtualMachines" \\
    --billing-scope-id /subscriptions/<subscription-id> \\
    --term "P1Y" \\
    --quantity 1 \\
    --display-name "MyReservation" \\
    --applied-scope-type "Single" \\
    --applied-scopes /subscriptions/<subscription-id> \\
    --sku Standard_B1s`,
				},
				{
					Title: "Cost Optimization",
					Content: `Cost optimization strategies help reduce Azure spending.

**Optimization Strategies:**
- **Right-sizing**: Match resources to actual usage
- **Reserved Instances**: Commit for savings
- **Spot VMs**: Use for fault-tolerant workloads
- **Auto-shutdown**: Shutdown dev/test resources
- **Resource Cleanup**: Remove unused resources

**Azure Advisor:**
- **Cost Recommendations**: Identify savings opportunities
- **Right-sizing**: VM size recommendations
- **Reserved Instance Recommendations**: RI purchase suggestions
- **Unused Resources**: Identify unused resources

**Best Practices:**
- Review Advisor recommendations regularly
- Right-size VMs based on metrics
- Use Spot VMs for appropriate workloads
- Enable auto-shutdown for dev/test
- Clean up unused resources
- Monitor costs continuously`,
					CodeExamples: `# View cost recommendations
az advisor recommendation list \\
    --category Cost \\
    --resource-group myResourceGroup

# Enable auto-shutdown for VM
az vm auto-shutdown \\
    --resource-group myResourceGroup \\
    --name myVM \\
    --time 18:00`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          426,
			Title:       "Azure Front Door",
			Description: "Learn Front Door: global load balancing, CDN, WAF, and application acceleration.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Front Door Fundamentals",
					Content: `Azure Front Door is a global application delivery network with built-in security.

**Features:**
- **Global Load Balancing**: Route traffic to closest backend
- **CDN**: Content delivery network capabilities
- **Web Application Firewall**: DDoS and application protection
- **SSL Termination**: Offload SSL processing
- **URL Rewrite**: Modify request URLs

**Routing Methods:**
- **Priority**: Route to highest priority backend
- **Weighted**: Distribute traffic by weight
- **Performance**: Route to lowest latency
- **Session Affinity**: Sticky sessions

**Backend Pools:**
- **Azure Services**: App Service, Storage, etc.
- **External Backends**: On-premises or other clouds
- **Health Probes**: Monitor backend health
- **Load Balancing**: Distribute across backends

**WAF Policies:**
- **Managed Rules**: Pre-configured rule sets
- **Custom Rules**: Organization-specific rules
- **Rate Limiting**: Prevent abuse
- **Geo-filtering**: Block/allow by location

**Best Practices:**
- Use WAF for production workloads
- Configure health probes appropriately
- Use custom domains with SSL
- Monitor Front Door metrics
- Implement rate limiting`,
					CodeExamples: `# Create Front Door profile
az afd profile create \\
    --resource-group myResourceGroup \\
    --profile-name myFrontDoor \\
    --sku Premium_AzureFrontDoor

# Create backend pool
az afd origin-group create \\
    --resource-group myResourceGroup \\
    --profile-name myFrontDoor \\
    --origin-group-name myBackendPool \\
    --probe-request-type GET \\
    --probe-protocol Http \\
    --probe-path /health

# Create WAF policy
az network front-door waf-policy create \\
    --resource-group myResourceGroup \\
    --name myWAFPolicy \\
    --sku Premium_AzureFrontDoor

# Create routing rule
az afd route create \\
    --resource-group myResourceGroup \\
    --profile-name myFrontDoor \\
    --endpoint-name myEndpoint \\
    --route-name myRoute \\
    --origin-group myBackendPool \\
    --patterns "/*" \\
    --supported-protocols Http Https`,
				},
				{
					Title: "WAF Integration",
					Content: `Front Door integrates with Web Application Firewall for security.

**WAF Integration:**
- **WAF Policy**: Attach WAF policy to Front Door
- **Managed Rules**: OWASP and Microsoft rule sets
- **Custom Rules**: Organization-specific rules
- **Rate Limiting**: DDoS protection

**Best Practices:**
- Enable WAF for production
- Use managed rule sets
- Create custom rules as needed
- Monitor WAF logs
- Tune rules to reduce false positives`,
					CodeExamples: `# Create WAF policy
az network front-door waf-policy create \\
    --resource-group myResourceGroup \\
    --name myWAFPolicy \\
    --sku Premium_AzureFrontDoor

# Attach to Front Door
az network front-door update \\
    --resource-group myResourceGroup \\
    --name myFrontDoor \\
    --set frontendEndpoints[0].webApplicationFirewallPolicyLink.id=/subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Network/frontDoorWebApplicationFirewallPolicies/myWAFPolicy`,
				},
				{
					Title: "Routing Rules",
					Content: `Front Door routing rules determine how traffic is routed to backends.

**Routing Methods:**
- **Priority**: Route to highest priority backend
- **Weighted**: Distribute by weight
- **Performance**: Route to lowest latency
- **Session Affinity**: Sticky sessions

**Rule Configuration:**
- **Patterns**: URL patterns to match
- **Backend Pool**: Target backend pool
- **Protocols**: HTTP, HTTPS, or both
- **Query String**: Forward or strip query strings

**Best Practices:**
- Use performance routing for global apps
- Use weighted routing for A/B testing
- Configure appropriate health probes
- Monitor routing metrics`,
					CodeExamples: `# Create routing rule
az afd route create \\
    --resource-group myResourceGroup \\
    --profile-name myFrontDoor \\
    --endpoint-name myEndpoint \\
    --route-name myRoute \\
    --origin-group myBackendPool \\
    --patterns "/*" \\
    --supported-protocols Http Https \\
    --forwarding-protocol MatchRequest`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          427,
			Title:       "Azure Logic Apps",
			Description: "Learn Logic Apps: workflow automation, integrations, and serverless workflows.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Logic Apps Fundamentals",
					Content: `Azure Logic Apps is a cloud platform for creating and running automated workflows.

**Workflow Types:**
- **Consumption**: Serverless, pay-per-execution
- **Standard**: Dedicated hosting, better performance

**Triggers:**
- **HTTP Trigger**: Webhook or request trigger
- **Schedule Trigger**: Time-based execution
- **Service-specific Triggers**: Azure services, SaaS apps
- **Event Grid Trigger**: Event-driven execution

**Actions:**
- **Built-in Actions**: Native Azure operations
- **Connector Actions**: Third-party service integrations
- **Control Actions**: Loops, conditions, switches
- **Custom Code**: Inline code or Azure Functions

**Connectors:**
- **Azure Connectors**: Azure service integrations
- **SaaS Connectors**: Office 365, Salesforce, etc.
- **On-premises Connectors**: SQL Server, File System
- **Enterprise Connectors**: SAP, Oracle, etc.

**Best Practices:**
- Use Consumption plan for variable workloads
- Implement error handling
- Use managed identities for authentication
- Monitor workflow execution
- Optimize for cost`,
					CodeExamples: `# Create logic app
az logicapp create \\
    --resource-group myResourceGroup \\
    --name myLogicApp \\
    --location eastus \\
    --sku Consumption

# Example workflow (JSON)
{
  "definition": {
    "$schema": "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
    "triggers": {
      "Recurrence": {
        "type": "Recurrence",
        "recurrence": {
          "frequency": "Hour",
          "interval": 1
        }
      }
    },
    "actions": {
      "Send_Email": {
        "type": "ApiConnection",
        "inputs": {
          "host": {
            "connection": {
              "name": "@parameters('$connections')['office365']['connectionId']"
            }
          },
          "method": "post",
          "path": "/v2/Mail",
          "body": {
            "To": "user@example.com",
            "Subject": "Hello from Logic Apps",
            "Body": "This is a test email"
          }
        }
      }
    }
  }
}`,
				},
				{
					Title: "Workflow Design",
					Content: `Effective workflow design ensures reliable and maintainable Logic Apps.

**Workflow Patterns:**
- **Sequential**: Execute steps in order
- **Parallel**: Execute steps simultaneously
- **Conditional**: Branch based on conditions
- **Loops**: Repeat steps
- **Error Handling**: Handle errors gracefully

**Best Practices:**
- Keep workflows focused and simple
- Use parameters for configurability
- Implement proper error handling
- Use managed identities
- Monitor workflow execution
- Document workflow logic`,
					CodeExamples: `# Example workflow with error handling
{
  "definition": {
    "actions": {
      "Try": {
        "type": "Scope",
        "actions": {
          "ProcessData": {
            "type": "Http",
            "inputs": {
              "method": "POST",
              "uri": "https://api.example.com/process"
            }
          }
        },
        "runAfter": {},
        "catch": [
          {
            "cases": [
              {
                "case": "Error",
                "actions": {
                  "HandleError": {
                    "type": "Http",
                    "inputs": {
                      "method": "POST",
                      "uri": "https://api.example.com/error"
                    }
                  }
                }
              }
            ]
          }
        ]
      }
    }
  }
}`,
				},
				{
					Title: "Error Handling",
					Content: `Proper error handling ensures Logic Apps workflows are resilient.

**Error Handling Strategies:**
- **Try-Catch**: Catch and handle errors
- **Retry Policies**: Automatic retries
- **Dead Letter**: Store failed messages
- **Notifications**: Alert on failures

**Retry Policies:**
- **Exponential Backoff**: Increasing delays
- **Fixed Interval**: Fixed delay
- **Retry Count**: Maximum retry attempts
- **Timeout**: Operation timeout

**Best Practices:**
- Implement try-catch blocks
- Configure retry policies
- Log errors for debugging
- Notify on critical failures
- Use dead letter for failed messages`,
					CodeExamples: `# Configure retry policy
{
  "retryPolicy": {
    "type": "exponential",
    "count": 3,
    "interval": "PT10S",
    "maximumInterval": "PT1H",
    "minimumInterval": "PT5S"
  }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          428,
			Title:       "Azure API Management",
			Description: "Learn API Management: API gateway, versioning, rate limiting, and API lifecycle management.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "API Management Fundamentals",
					Content: `Azure API Management provides a unified API gateway for managing APIs.

**Components:**
- **API Gateway**: Single entry point for APIs
- **Developer Portal**: Self-service developer experience
- **Publisher Portal**: API management interface
- **Analytics**: API usage and performance insights

**Tiers:**
- **Consumption**: Serverless, pay-per-execution
- **Developer**: Development and testing
- **Basic**: Small-scale production
- **Standard**: Production workloads
- **Premium**: High-scale and multi-region

**Policies:**
- **Inbound Policies**: Request transformation
- **Outbound Policies**: Response transformation
- **Backend Policies**: Backend communication
- **On-error Policies**: Error handling

**API Lifecycle:**
- **Design**: Create API definitions
- **Develop**: Implement backend services
- **Publish**: Make APIs available
- **Monitor**: Track usage and performance
- **Version**: Manage API versions

**Best Practices:**
- Use API versioning
- Implement rate limiting
- Use OAuth 2.0 for authentication
- Monitor API usage
- Implement caching strategies`,
					CodeExamples: `# Create API Management instance
az apim create \\
    --resource-group myResourceGroup \\
    --name myAPIM \\
    --location eastus \\
    --publisher-name "My Company" \\
    --publisher-email "admin@mycompany.com" \\
    --sku-name Developer

# Import API
az apim api import \\
    --resource-group myResourceGroup \\
    --service-name myAPIM \\
    --path "myapi" \\
    --specification-format OpenApi \\
    --specification-url https://example.com/openapi.json

# Create API policy
az apim api policy set \\
    --resource-group myResourceGroup \\
    --service-name myAPIM \\
    --api-id myapi \\
    --policy-format xml \\
    --value @policy.xml

# Example policy (policy.xml)
<policies>
  <inbound>
    <rate-limit calls="100" renewal-period="60" />
    <base />
  </inbound>
  <backend>
    <base />
  </backend>
  <outbound>
    <base />
  </outbound>
</policies>`,
				},
				{
					Title: "API Versioning",
					Content: `API versioning allows multiple API versions to coexist.

**Versioning Strategies:**
- **URL Path**: /api/v1/, /api/v2/
- **Query String**: ?api-version=1.0
- **Header**: Custom header for version
- **Version Sets**: Group related versions

**Version Management:**
- **Version Sets**: Organize versions
- **Version Policies**: Apply policies to versions
- **Deprecation**: Mark versions as deprecated
- **Migration**: Guide users to new versions

**Best Practices:**
- Use consistent versioning strategy
- Document version changes
- Deprecate old versions gracefully
- Provide migration guides
- Monitor version usage`,
					CodeExamples: `# Create API version set
az apim api versionset create \\
    --resource-group myResourceGroup \\
    --service-name myAPIM \\
    --display-name "My API Versions" \\
    --versioning-scheme Segment \\
    --versionset-id myVersionset`,
				},
				{
					Title: "Rate Limiting",
					Content: `Rate limiting protects APIs from abuse and ensures fair usage.

**Rate Limiting Types:**
- **Call Rate**: Limit calls per time period
- **Bandwidth**: Limit data transfer
- **Concurrent Calls**: Limit simultaneous calls
- **Quota**: Limit total usage

**Rate Limit Policies:**
- **Per Key**: Limit per subscription key
- **Per IP**: Limit per IP address
- **Per Operation**: Limit per API operation
- **Global**: Limit across all calls

**Best Practices:**
- Set appropriate rate limits
- Use different limits for different tiers
- Monitor rate limit violations
- Provide clear error messages
- Implement retry logic in clients`,
					CodeExamples: `# Rate limit policy
<policies>
  <inbound>
    <rate-limit calls="100" renewal-period="60" />
    <quota calls="10000" renewal-period="3600" />
  </inbound>
</policies>`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          429,
			Title:       "Azure Arc",
			Description: "Learn Azure Arc: hybrid cloud management, multi-cloud governance, and unified operations.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Arc Fundamentals",
					Content: `Azure Arc extends Azure management and services to any infrastructure.

**Supported Resources:**
- **Servers**: Windows and Linux servers
- **Kubernetes Clusters**: Any Kubernetes cluster
- **Data Services**: SQL Server, PostgreSQL
- **Azure Services**: Run Azure services anywhere

**Management Capabilities:**
- **Unified Management**: Manage resources from Azure Portal
- **Policy and Compliance**: Apply Azure Policy
- **Security**: Azure Security Center integration
- **Monitoring**: Azure Monitor integration
- **Updates**: Azure Update Management

**Use Cases:**
- **Hybrid Cloud**: On-premises and cloud resources
- **Multi-Cloud**: Resources across cloud providers
- **Edge Computing**: Edge locations and devices
- **Compliance**: Meet data residency requirements

**Arc-enabled Services:**
- **Arc-enabled Servers**: Manage servers
- **Arc-enabled Kubernetes**: Manage Kubernetes clusters
- **Arc-enabled Data Services**: Run data services anywhere
- **Arc-enabled Machine Learning**: ML workloads anywhere

**Best Practices:**
- Use Azure Arc for hybrid scenarios
- Implement Azure Policy for governance
- Enable security monitoring
- Use tags for organization
- Monitor Arc resource health`,
					CodeExamples: `# Connect server to Arc
az connectedmachine create \\
    --resource-group myResourceGroup \\
    --name myServer \\
    --location eastus \\
    --kind "onpremises"

# Connect Kubernetes cluster
az connectedk8s connect \\
    --resource-group myResourceGroup \\
    --name myK8sCluster \\
    --location eastus

# Enable Arc-enabled SQL Server
az sql server-arc create \\
    --name myArcSQLServer \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --connectivity-mode indirect \\
    --k8s-namespace arc

# Apply Azure Policy to Arc resources
az policy assignment create \\
    --name "arc-policy" \\
    --display-name "Arc Policy" \\
    --policy "require-tag" \\
    --scope /subscriptions/<subscription-id>/resourceGroups/myResourceGroup`,
				},
				{
					Title: "Arc-enabled Servers",
					Content: `Arc-enabled servers extend Azure management to on-premises and other cloud servers.

**Arc-enabled Server Benefits:**
- **Unified Management**: Manage from Azure Portal
- **Azure Policies**: Apply policies to servers
- **Update Management**: Patch management
- **Monitoring**: Azure Monitor integration

**Server Requirements:**
- **Windows**: Windows Server 2012 R2 or later
- **Linux**: Supported Linux distributions
- **Network**: Internet connectivity or proxy
- **Permissions**: Appropriate Azure permissions

**Best Practices:**
- Use for hybrid scenarios
- Enable Update Management
- Apply Azure Policies
- Monitor server health
- Use tags for organization`,
					CodeExamples: `# Connect server to Arc
az connectedmachine create \\
    --resource-group myResourceGroup \\
    --name myServer \\
    --location eastus \\
    --kind "onpremises"`,
				},
				{
					Title: "Arc-enabled Kubernetes",
					Content: `Arc-enabled Kubernetes extends Azure management to any Kubernetes cluster.

**Arc Kubernetes Benefits:**
- **Unified Management**: Manage from Azure
- **GitOps**: Deploy using GitOps
- **Azure Policies**: Apply policies to clusters
- **Monitoring**: Azure Monitor integration

**Supported Clusters:**
- **AKS**: Azure Kubernetes Service
- **On-Premises**: On-premises Kubernetes
- **Other Clouds**: AWS EKS, GKE, etc.
- **Edge**: Edge Kubernetes clusters

**Best Practices:**
- Use for multi-cloud scenarios
- Enable GitOps for deployments
- Apply Azure Policies
- Monitor cluster health
- Use for compliance requirements`,
					CodeExamples: `# Connect Kubernetes cluster
az connectedk8s connect \\
    --resource-group myResourceGroup \\
    --name myK8sCluster \\
    --location eastus`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
