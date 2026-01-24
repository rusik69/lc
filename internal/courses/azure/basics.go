package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          400,
			Title:       "Azure Fundamentals",
			Description: "Introduction to Azure: cloud computing basics, Azure global infrastructure, and core services.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Microsoft Azure?",
					Content: `Microsoft Azure is the world's second-largest cloud platform, providing on-demand computing power, storage, and networking. It is the go-to choice for enterprises deeply integrated with the Microsoft ecosystem.

**In simpler terms:**
Instead of managing your own servers and data centers, you rent them from Microsoft.

**Why Azure?**
*   **The Enterprise Standard:** If your company uses Windows, Active Directory, or Office 365, Azure is the natural fit.
*   **Hybrid Power:** Superior tools for connecting on-premises data centers to the cloud (Azure Arc).
*   **Services:** Over 200+ services. Everything from Virtual Machines to AI (OpenAI models).

**Global Infrastructure:**
Azure uses **Regions** and **Availability Zones** to keep your apps running.

` + "```" + `
Region (e.g., East US)
│
├── Availability Zone 1 ───────────┐
│   [ Data Center ]                │  <-- High-Speed Link
├── Availability Zone 2 ───────────┤      (< 2ms)
│   [ Data Center ]                │
└── Availability Zone 3 ───────────┘
    [ Data Center ]
` + "```" + `

*   **Region:** A geographic area (e.g., UK South, East US) containing one or more datacenters.
*   **Availability Zone (AZ):** Physically separate locations within a region with independent power and cooling.
*   **Geographies:** Higher-level boundaries (usually countries) for data residency and compliance.

**Key takeaway:** To avoid downtime, always deploy your app across multiple **Availability Zones**. If one data center has a power cut, your app stays up in the others.`,
					CodeExamples: `# Install Azure CLI
# https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

# Login to your account
$ az login

# Set your default subscription (if you have multiple)
$ az account set --subscription "My Enterprise Subscription"

# Test it out: List all resource groups
$ az group list --output table`,
				},
				{
					Title: "Azure Global Infrastructure",
					Content: `Azure has 60+ regions worldwide—the most of any provider. It’s organized into four layers:

**1. Geographies (The "Compliance")**
Usually a country (e.g., United Kingdom, Germany) or a large area.
*   **Purpose:** Ensures data residency. Data doesn't leave the geography without your permission.

**2. Regions (The "Where")**
A set of datacenters connected through a dedicated low-latency network.
*   **Examples:** UK South (London), East US (Virginia).
*   **Regional Pairs:** Every region is "paired" with another at least 300 miles away (e.g., East US and West US). If a massive disaster hits one, your data is safe in the other.

**3. Availability Zones (The "Resilience")**
Individual datacenters inside a region.
*   **What it is:** A location with its own power, cooling, and network.
*   **SLA:** 99.99% for VMs across two or more AZs.

**4. Edge Locations (The "Speed")**
*   **Purpose:** Not for hosting servers, but for caching content (CDN) and global load balancing (Front Door).
*   **Count:** 100+ points-of-presence (PoPs).

**Decision Tree:**
*   Need speed for local users? -> Choose the closest **Region**.
*   Need data to stay in a country? -> Choose the right **Geography**.
*   Need 24/7 uptime? -> Use multiple **Availability Zones**.`,
					CodeExamples: `# Find the closest region to you (Ping)
# Standard Azure endpoints
$ ping eastus.api.azure.com
$ ping uksouth.api.azure.com

# List all available locations
$ az account list-locations --output table`,
				},
				{
					Title: "Core Azure Services",
					Content: `Azure has hundreds of services, but these 6 are the foundations of almost every architecture.

**1. Compute (The "Brains")**
*   **Virtual Machines (IaaS):** Total control. You manage the OS.
*   **Azure Functions (FaaS):** Serverless. Upload code, it runs on demand.
*   **App Service (PaaS):** Managed web hosting for Java, .NET, Python, etc.

**2. Networking (The "Cables")**
*   **VNet (Virtual Network):** Your private space in Azure.
*   **Load Balancer:** Distributes traffic between servers.

**3. Storage & DB (The "Memory")**
*   **Blob Storage:** Unlimited "bucket" for files (photos, backups). Equivalent to AWS S3.
*   **Azure SQL Database:** Managed SQL Server.
*   **Cosmos DB:** Global NoSQL database (like DynamoDB but Multi-Model).

**Visual Architecture:**
` + "```" + `
[ User ] -> [ App Service ] -> [ VNet ] -> [ SQL Database ]
                              |
                     [ Blob Storage (Photos) ]
` + "```" + ``,
					CodeExamples: `# Storage: Create a storage account
$ az storage account create -n mystore -g myRG -l eastus --sku Standard_LRS

# Compute: Create a basic Linux VM
$ az vm create -n myVM -g myRG --image UbuntuLTS --generate-ssh-keys`,
				},

				{
					Title: "Azure Pricing and Billing",
					Content: `Understanding Azure pricing and billing is crucial for managing costs and optimizing cloud spending.

**Azure Pricing Model:**
- **Pay-as-you-go**: Pay only for what you use, no upfront costs
- **Reserved Instances**: 1-3 year commitments for up to 72% savings
- **Spot VMs**: Use unused capacity at up to 90% discount
- **Azure Hybrid Benefit**: Use existing Windows Server and SQL Server licenses
- **Free Tier**: 12 months free for new accounts, always-free services

**Billing Components:**
- **Compute**: Virtual machines, App Service, Functions
- **Storage**: Blob Storage, File Storage, Disk Storage
- **Networking**: Data transfer, Load Balancer, VPN Gateway
- **Databases**: SQL Database, Cosmos DB, managed databases
- **Services**: Monitoring, backup, security services

**Cost Management Tools:**
- **Cost Management + Billing**: Centralized cost tracking and analysis
- **Budgets**: Set spending limits and alerts
- **Cost Analysis**: Detailed cost breakdowns by resource, service, tag
- **Cost Alerts**: Notifications when spending thresholds are reached
- **Reserved Instance Recommendations**: Identify savings opportunities

**Cost Optimization Strategies:**
- Use Reserved Instances for predictable workloads
- Right-size VMs based on actual usage
- Use Spot VMs for fault-tolerant workloads
- Enable auto-shutdown for dev/test VMs
- Use tags for cost allocation and tracking
- Review and remove unused resources regularly
- Leverage Azure Hybrid Benefit for Windows/SQL workloads

**Best Practices:**
- Set up budgets and alerts early
- Use cost allocation tags consistently
- Review Cost Analysis reports monthly
- Consider Reserved Instances for production workloads
- Monitor and optimize storage costs
- Use Azure Advisor for cost recommendations`,
					CodeExamples: `# View current subscription costs
az consumption usage list \\
    --start-date 2024-01-01 \\
    --end-date 2024-01-31

# Create budget
az consumption budget create \\
    --budget-name MonthlyBudget \\
    --amount 1000 \\
    --time-grain Monthly \\
    --start-date 2024-01-01 \\
    --end-date 2024-12-31 \\
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
    --sku Standard_B1s

# Tag resources for cost tracking
az resource tag \\
    --tags Environment=Production Project=WebApp \\
    --ids /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Compute/virtualMachines/myVM

# Using PowerShell
# Get cost for resource group
Get-AzConsumptionUsageSummary \\
    -StartDate (Get-Date).AddDays(-30) \\
    -EndDate (Get-Date) \\
    -Granularity Daily

# Create budget
New-AzConsumptionBudget \\
    -Name MonthlyBudget \\
    -Amount 1000 \\
    -Category Cost \\
    -TimeGrain Monthly \\
    -StartDate (Get-Date) \\
    -EndDate (Get-Date).AddYears(1)`,
				},
				{
					Title: "Azure Resource Manager Basics",
					Content: `Azure Resource Manager (ARM) is the deployment and management service for Azure resources.

**Resource Manager Concepts:**
- **Resource Groups**: Logical containers for related resources
- **Resources**: Individual Azure services (VMs, storage accounts, etc.)
- **Subscriptions**: Billing and access boundaries
- **Management Groups**: Organize subscriptions hierarchically
- **Tags**: Key-value pairs for resource organization

**Resource Group Benefits:**
- Group related resources together
- Manage resources as a single unit
- Apply policies and access control at group level
- Delete entire group to remove all resources
- Organize by application, environment, or project

**Resource Organization:**
- **By Application**: All resources for one app in one group
- **By Environment**: Separate groups for dev, test, prod
- **By Lifecycle**: Group resources with same lifecycle
- **By Department**: Organize by business unit

**Resource Manager Features:**
- **Declarative Templates**: Define desired state in JSON/Bicep
- **Idempotent Operations**: Safe to run multiple times
- **Dependency Management**: Automatically handles resource dependencies
- **Rollback Support**: Undo failed deployments
- **Access Control**: RBAC at resource group level

**ARM Templates:**
- **JSON Templates**: Declarative infrastructure as code
- **Template Specs**: Reusable template libraries
- **Bicep**: Simplified template language (compiles to ARM JSON)
- **Parameters**: Input values for templates
- **Variables**: Reusable values within templates
- **Outputs**: Returned values from deployments

**Best Practices:**
- Use resource groups for logical grouping
- Apply consistent naming conventions
- Use tags for organization and cost tracking
- Implement resource group policies
- Use ARM templates for repeatable deployments
- Store templates in version control
- Test templates in non-production first`,
					CodeExamples: `# Create resource group
az group create \\
    --name myResourceGroup \\
    --location eastus

# List resource groups
az group list --output table

# Show resource group details
az group show --name myResourceGroup

# List resources in group
az resource list --resource-group myResourceGroup --output table

# Move resource to another group
az resource move \\
    --destination-group newResourceGroup \\
    --ids /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Compute/virtualMachines/myVM

# Delete resource group (and all resources)
az group delete --name myResourceGroup --yes

# Add tags to resource group
az group update \\
    --name myResourceGroup \\
    --tags Environment=Production Project=WebApp

# Deploy ARM template
az deployment group create \\
    --resource-group myResourceGroup \\
    --template-file template.json \\
    --parameters @parameters.json

# Example ARM template (template.json)
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "storageAccountName": {
      "type": "string"
    }
  },
  "resources": [
    {
      "type": "Microsoft.Storage/storageAccounts",
      "apiVersion": "2021-04-01",
      "name": "[parameters('storageAccountName')]",
      "location": "[resourceGroup().location]",
      "sku": {
        "name": "Standard_LRS"
      },
      "kind": "StorageV2"
    }
  ]
}

# Using PowerShell
# Create resource group
New-AzResourceGroup -Name myResourceGroup -Location eastus

# Get resource group
Get-AzResourceGroup -Name myResourceGroup

# Deploy template
New-AzResourceGroupDeployment \\
    -ResourceGroupName myResourceGroup \\
    -TemplateFile template.json \\
    -TemplateParameterFile parameters.json`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          401,
			Title:       "Azure Active Directory",
			Description: "Learn Azure AD: identity management, users, groups, roles, and security best practices.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Azure AD Fundamentals",
					Content: `Azure Active Directory (Azure AD) is Microsoft's cloud-based identity and access management service.

**Azure AD Concepts:**
- **Tenant**: Organization's Azure AD instance
- **Users**: Individual people or service principals
- **Groups**: Collection of users with shared permissions
- **Roles**: Built-in or custom roles for permissions
- **Applications**: Apps registered in Azure AD
- **Subscriptions**: Billing and resource containers

**Azure AD Editions:**
- **Free**: Basic identity management
- **Office 365**: Included with Office 365
- **Premium P1**: Advanced identity protection
- **Premium P2**: All P1 features plus Privileged Identity Management

**Identity Types:**
- **Cloud identities**: Created in Azure AD
- **Synchronized identities**: Synced from on-premises AD
- **Guest users**: External users invited to tenant
- **Service principals**: Applications and services

**Best Practices:**
- Enable Multi-Factor Authentication (MFA)
- Use Conditional Access policies
- Follow principle of least privilege
- Regular access reviews
- Monitor sign-in logs
- Use Privileged Identity Management for admin roles`,
					CodeExamples: `# Create Azure AD user
az ad user create \\
    --display-name "John Doe" \\
    --user-principal-name john.doe@contoso.com \\
    --password "SecurePassword123!"

# Create security group
az ad group create \\
    --display-name "Developers" \\
    --mail-nickname "developers"

# Add user to group
az ad group member add \\
    --group "Developers" \\
    --member-id <user-object-id>

# List all users
az ad user list --output table

# Assign role to user
az role assignment create \\
    --assignee john.doe@contoso.com \\
    --role "Contributor" \\
    --scope /subscriptions/<subscription-id>`,
				},
				{
					Title: "Conditional Access Policies",
					Content: `Conditional Access is Azure AD's policy engine that allows you to create access controls based on conditions.

**Conditional Access Components:**
- **Assignments**: Who the policy applies to (users, groups, roles)
- **Cloud Apps**: Which applications are protected
- **Conditions**: When the policy applies (location, device, risk)
- **Access Controls**: What happens when conditions are met (grant, block, require MFA)

**Common Policy Scenarios:**
- **Require MFA for admins**: All administrative roles must use MFA
- **Block legacy authentication**: Prevent basic authentication protocols
- **Require compliant devices**: Only managed devices can access
- **Location-based access**: Block access from untrusted locations
- **Risk-based policies**: Require additional verification for risky sign-ins

**Policy Conditions:**
- **User and Group**: Target specific users or groups
- **Cloud Apps**: Apply to specific applications or all apps
- **Sign-in Risk**: Based on Azure AD Identity Protection risk
- **Device Platform**: iOS, Android, Windows, macOS
- **Location**: Based on IP address or named locations
- **Client Apps**: Browser, mobile apps, desktop clients

**Access Controls:**
- **Grant**: Allow access (with optional requirements)
- **Block**: Deny access completely
- **Session**: Control session behavior (app-enforced restrictions)

**Best Practices:**
- Start with report-only mode to test policies
- Use named locations for trusted IP ranges
- Require MFA for all administrative access
- Block legacy authentication protocols
- Use risk-based policies for additional security
- Document all policies and their purposes
- Test policies in non-production first`,
					CodeExamples: `# Create conditional access policy (using Microsoft Graph API)
# Note: Conditional Access policies are typically managed via Azure Portal or Microsoft Graph API

# Example policy JSON structure
{
  "displayName": "Require MFA for admins",
  "state": "enabled",
  "conditions": {
    "users": {
      "includeUsers": ["admin@contoso.com"],
      "includeRoles": ["62e90394-69f5-4237-9190-012177145e10"]
    },
    "applications": {
      "includeApplications": ["All"]
    },
    "locations": {
      "includeLocations": ["All"]
    }
  },
  "grantControls": {
    "operator": "AND",
    "builtInControls": ["mfa"]
  }
}

# Enable MFA for user
az ad user update \\
    --id john.doe@contoso.com \\
    --account-enabled true

# Using PowerShell
# Install Microsoft.Graph module
Install-Module Microsoft.Graph

# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess"

# Create conditional access policy
$params = @{
    displayName = "Require MFA for admins"
    state = "enabled"
    conditions = @{
        users = @{
            includeUsers = @("admin@contoso.com")
            includeRoles = @("62e90394-69f5-4237-9190-012177145e10")
        }
        applications = @{
            includeApplications = @("All")
        }
    }
    grantControls = @{
        operator = "AND"
        builtInControls = @("mfa")
    }
}
New-MgIdentityConditionalAccessPolicy -BodyParameter $params`,
				},
				{
					Title: "Azure AD B2B and B2C",
					Content: `Azure AD B2B (Business-to-Business) and B2C (Business-to-Consumer) extend identity management to external users.

**Azure AD B2B:**
- **Purpose**: Collaborate with external partners, vendors, contractors
- **Guest Users**: External users invited to your Azure AD tenant
- **Self-Service**: Users can redeem invitations and access resources
- **No Separate Directory**: Guests use their own credentials
- **Access Control**: Same conditional access and RBAC as internal users

**B2B Collaboration Features:**
- **Invitation Flow**: Email invitation with redemption link
- **Just-In-Time (JIT) Provisioning**: Users created automatically on first sign-in
- **Guest User Limits**: Up to 5 guest users per internal user (can be increased)
- **Access Reviews**: Regular reviews of guest access
- **B2B Direct Connect**: Trust relationship with partner organizations

**Azure AD B2C:**
- **Purpose**: Customer-facing applications and identity management
- **Consumer Identity**: Manage customer identities and authentication
- **Social Identity Providers**: Facebook, Google, Microsoft, etc.
- **Custom Policies**: Advanced identity user flows
- **Separate Tenant**: Dedicated B2C tenant separate from corporate Azure AD

**B2C User Flows:**
- **Sign-up and Sign-in**: Combined registration and login
- **Password Reset**: Self-service password recovery
- **Profile Editing**: Users can update their profiles
- **Custom Policies**: Advanced scenarios with Identity Experience Framework

**Use Cases:**
- **B2B**: Partner portals, vendor access, contractor collaboration
- **B2C**: E-commerce sites, mobile apps, customer portals

**Best Practices:**
- Use B2B for business partners and contractors
- Use B2C for customer-facing applications
- Implement access reviews for B2B guests
- Use conditional access for B2B users
- Limit guest permissions to minimum required
- Monitor guest user activity
- Set expiration dates for guest access`,
					CodeExamples: `# Invite B2B guest user
az ad user create \\
    --display-name "Partner User" \\
    --user-principal-name partner@externaldomain.com \\
    --user-type Guest \\
    --mail-nickname partner

# Or use invitation API
az rest \\
    --method POST \\
    --uri "https://graph.microsoft.com/v1.0/invitations" \\
    --headers "Content-Type=application/json" \\
    --body '{
        "invitedUserEmailAddress": "partner@externaldomain.com",
        "invitedUserDisplayName": "Partner User",
        "inviteRedirectUrl": "https://myapp.azurewebsites.net",
        "sendInvitationMessage": true
    }'

# List guest users
az ad user list \\
    --filter "userType eq 'Guest'" \\
    --output table

# Remove guest user
az ad user delete --id partner@externaldomain.com

# Using PowerShell
# Install Microsoft.Graph module
Install-Module Microsoft.Graph

# Connect to Microsoft Graph
Connect-MgGraph -Scopes "User.Invite.All"

# Invite guest user
$params = @{
    invitedUserEmailAddress = "partner@externaldomain.com"
    invitedUserDisplayName = "Partner User"
    inviteRedirectUrl = "https://myapp.azurewebsites.net"
    sendInvitationMessage = $true
}
New-MgInvitation -BodyParameter $params

# For B2C, use Azure Portal or Azure AD B2C APIs
# B2C configuration is typically done through Azure Portal UI`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          402,
			Title:       "Azure Virtual Machines",
			Description: "Master Azure VMs: instances, images, network security groups, and VM management.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "VM Fundamentals",
					Content: `Azure Virtual Machines provide on-demand, scalable computing resources.

**VM Concepts:**
- **Instance**: Virtual server in Azure
- **Image**: Template for VMs (Windows, Linux, custom)
- **VM Size**: CPU, memory, storage, networking capacity
- **Network Security Group**: Virtual firewall
- **Managed Disks**: Persistent block storage
- **Availability Set**: Group VMs for redundancy

**VM Series:**
- **General Purpose**: Balanced compute, memory, networking (B, D, DS)
- **Compute Optimized**: High-performance processors (F, FS)
- **Memory Optimized**: Large memory for databases (E, ES, M, MS)
- **Storage Optimized**: High I/O for databases (L, LS)
- **GPU**: GPU instances (NC, ND, NV)

**Pricing Models:**
- **Pay-as-you-go**: Pay per hour, no commitment
- **Reserved Instances**: 1-3 year commitment, up to 72% savings
- **Spot VMs**: Bid on unused capacity, up to 90% savings
- **Azure Hybrid Benefit**: Use existing licenses

**Best Practices:**
- Choose right VM size for workload
- Use Managed Disks for better reliability
- Use Availability Sets or Availability Zones
- Enable boot diagnostics
- Use tags for organization
- Monitor VM metrics`,
					CodeExamples: `# Create VM
az vm create \\
    --resource-group myResourceGroup \\
    --name myVM \\
    --image UbuntuLTS \\
    --size Standard_B2s \\
    --admin-username azureuser \\
    --generate-ssh-keys \\
    --public-ip-sku Standard

# List all VMs
az vm list --output table

# Get VM details
az vm show --resource-group myResourceGroup --name myVM

# Start VM
az vm start --resource-group myResourceGroup --name myVM

# Stop VM
az vm deallocate --resource-group myResourceGroup --name myVM

# Create VM with custom image
az vm create \\
    --resource-group myResourceGroup \\
    --name myVM \\
    --image myCustomImage \\
    --size Standard_D2s_v3`,
				},
				{
					Title: "VM Scaling and Availability Sets",
					Content: `Azure provides multiple options for scaling VMs and ensuring high availability.

**Availability Sets:**
- **Purpose**: Protect VMs from hardware failures and planned maintenance
- **Fault Domains**: Group VMs across different physical servers
- **Update Domains**: Group VMs for staged updates
- **SLA**: 99.95% uptime for 2+ VMs in availability set
- **Placement**: VMs distributed across fault and update domains

**Availability Zones:**
- **Purpose**: Protect VMs from datacenter-level failures
- **Zones**: Physically separate datacenters within a region
- **SLA**: 99.99% uptime for VMs across zones
- **Zone-redundant**: Deploy across multiple zones
- **Zone-pinned**: Deploy to specific zone

**VM Scale Sets:**
- **Purpose**: Create and manage identical VMs automatically
- **Auto-scaling**: Scale in/out based on metrics
- **Load Balancing**: Distribute traffic across instances
- **Automatic Updates**: Rolling updates with zero downtime
- **Cost-effective**: Pay only for running instances

**Scaling Options:**
- **Manual Scaling**: Manually adjust instance count
- **Scheduled Scaling**: Scale based on time schedule
- **Metric-based Scaling**: Scale based on CPU, memory, custom metrics
- **Scale-in Protection**: Prevent specific instances from being removed

**Scaling Rules:**
- **Scale-out**: Add instances when metric exceeds threshold
- **Scale-in**: Remove instances when metric below threshold
- **Cooldown Period**: Wait time between scaling actions
- **Minimum/Maximum**: Set bounds for instance count

**Best Practices:**
- Use Availability Sets for VMs in same datacenter
- Use Availability Zones for higher availability requirements
- Use VM Scale Sets for stateless workloads
- Configure auto-scaling based on actual usage patterns
- Set appropriate minimum and maximum instance counts
- Use scale-in protection for critical instances
- Monitor scaling events and adjust rules as needed`,
					CodeExamples: `# Create availability set
az vm availability-set create \\
    --resource-group myResourceGroup \\
    --name myAvailabilitySet \\
    --platform-fault-domain-count 2 \\
    --platform-update-domain-count 2

# Create VM in availability set
az vm create \\
    --resource-group myResourceGroup \\
    --name myVM \\
    --availability-set myAvailabilitySet \\
    --image UbuntuLTS \\
    --size Standard_B2s \\
    --admin-username azureuser \\
    --generate-ssh-keys

# Create VM in specific availability zone
az vm create \\
    --resource-group myResourceGroup \\
    --name myVM \\
    --image UbuntuLTS \\
    --size Standard_B2s \\
    --zone 1 \\
    --admin-username azureuser \\
    --generate-ssh-keys

# Create VM Scale Set
az vmss create \\
    --resource-group myResourceGroup \\
    --name myScaleSet \\
    --image UbuntuLTS \\
    --instance-count 2 \\
    --vm-sku Standard_B2s \\
    --upgrade-policy-mode Automatic \\
    --admin-username azureuser \\
    --generate-ssh-keys

# Configure auto-scaling
az monitor autoscale create \\
    --resource-group myResourceGroup \\
    --resource /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Compute/virtualMachineScaleSets/myScaleSet \\
    --min-count 2 \\
    --max-count 10 \\
    --count 2

# Add scale-out rule
az monitor autoscale rule create \\
    --resource-group myResourceGroup \\
    --autoscale-name myScaleSet \\
    --condition "Percentage CPU > 70 avg 5m" \\
    --scale out 1

# Add scale-in rule
az monitor autoscale rule create \\
    --resource-group myResourceGroup \\
    --autoscale-name myScaleSet \\
    --condition "Percentage CPU < 30 avg 5m" \\
    --scale in 1

# Using PowerShell
# Create availability set
New-AzAvailabilitySet \\
    -ResourceGroupName myResourceGroup \\
    -Name myAvailabilitySet \\
    -Location eastus \\
    -PlatformFaultDomainCount 2 \\
    -PlatformUpdateDomainCount 2

# Create VM Scale Set
New-AzVmss \\
    -ResourceGroupName myResourceGroup \\
    -Name myScaleSet \\
    -ImageName UbuntuLTS \\
    -InstanceCount 2 \\
    -VmSize Standard_B2s \\
    -UpgradePolicyMode Automatic`,
				},
				{
					Title: "VM Backup and Recovery",
					Content: `Azure Backup provides comprehensive backup and recovery solutions for virtual machines.

**Azure Backup Features:**
- **Automated Backups**: Scheduled backups without manual intervention
- **Application-Consistent**: Backups include application data consistency
- **Incremental Backups**: Only changed data is backed up
- **Long-term Retention**: Store backups for years
- **Cross-Region Backup**: Backup to different region for disaster recovery

**Recovery Services Vault:**
- **Purpose**: Centralized backup management
- **Storage Redundancy**: LRS, GRS, or ZRS options
- **Backup Policies**: Define backup schedule and retention
- **Soft Delete**: Protection against accidental deletion
- **Cross-Region Restore**: Restore to different region

**Backup Types:**
- **Full Backup**: Complete VM snapshot
- **Incremental Backup**: Only changes since last backup
- **Log Backup**: Transaction logs for databases

**Recovery Options:**
- **Restore VM**: Create new VM from backup
- **Restore Disks**: Restore disks and create VM manually
- **File-level Restore**: Restore individual files
- **Point-in-Time Restore**: Restore to specific point in time

**Backup Policies:**
- **Backup Frequency**: Daily or weekly backups
- **Retention**: How long to keep backups
- **Retention Rules**: Different retention for daily, weekly, monthly, yearly
- **Instant Restore**: Keep snapshots for quick restore

**Best Practices:**
- Enable backup immediately after VM creation
- Use GRS for cross-region protection
- Test restore procedures regularly
- Enable soft delete for protection
- Use application-consistent backups for databases
- Document recovery procedures
- Monitor backup jobs and failures
- Set appropriate retention policies`,
					CodeExamples: `# Create Recovery Services vault
az backup vault create \\
    --resource-group myResourceGroup \\
    --name myBackupVault \\
    --location eastus

# Enable backup for VM
az backup protection enable-for-vm \\
    --resource-group myResourceGroup \\
    --vault-name myBackupVault \\
    --vm myVM \\
    --policy-name DefaultPolicy

# Create backup policy
az backup policy create \\
    --resource-group myResourceGroup \\
    --vault-name myBackupVault \\
    --name DailyBackupPolicy \\
    --backup-management-type AzureIaasVM \\
    --policy '{
        "properties": {
            "backupManagementType": "AzureIaasVM",
            "schedulePolicy": {
                "schedulePolicyType": "SimpleSchedulePolicy",
                "scheduleRunFrequency": "Daily",
                "scheduleRunTimes": ["2024-01-01T02:00:00Z"]
            },
            "retentionPolicy": {
                "retentionPolicyType": "LongTermRetentionPolicy",
                "dailySchedule": {
                    "retentionDuration": {
                        "count": 30,
                        "durationType": "Days"
                    }
                }
            }
        }
    }'

# Trigger backup
az backup protection backup-now \\
    --resource-group myResourceGroup \\
    --vault-name myBackupVault \\
    --container-name myVM \\
    --item-name myVM \\
    --backup-type Full

# List recovery points
az backup recoverypoint list \\
    --resource-group myResourceGroup \\
    --vault-name myBackupVault \\
    --container-name myVM \\
    --item-name myVM

# Restore VM
az backup restore restore-disks \\
    --resource-group myResourceGroup \\
    --vault-name myBackupVault \\
    --container-name myVM \\
    --item-name myVM \\
    --rp-name <recovery-point-name> \\
    --storage-account mystorageaccount \\
    --target-resource-group myResourceGroup

# Using PowerShell
# Create Recovery Services vault
New-AzRecoveryServicesVault \\
    -ResourceGroupName myResourceGroup \\
    -Name myBackupVault \\
    -Location eastus

# Set vault context
Set-AzRecoveryServicesVaultContext \\
    -Vault $vault

# Enable backup
Enable-AzRecoveryServicesBackupProtection \\
    -ResourceGroupName myResourceGroup \\
    -Name myVM \\
    -Policy $policy`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          403,
			Title:       "Azure Storage",
			Description: "Learn Azure Storage: Blob Storage, File Storage, Queue Storage, and Table Storage.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Storage Account Fundamentals",
					Content: `Azure Storage provides scalable cloud storage for data, files, and applications.

**Storage Account Types:**
- **General-purpose v2**: Latest generation, supports all storage types
- **General-purpose v1**: Legacy account type
- **Blob Storage**: Optimized for blob storage only

**Storage Services:**
- **Blob Storage**: Object storage for unstructured data
- **File Storage**: Managed file shares (SMB/NFS)
- **Queue Storage**: Message queuing for application communication
- **Table Storage**: NoSQL key-value store

**Performance Tiers:**
- **Standard**: Cost-effective for most workloads
- **Premium**: High-performance SSD-backed storage

**Redundancy Options:**
- **LRS (Locally Redundant Storage)**: 3 copies in one datacenter
- **ZRS (Zone-Redundant Storage)**: 3 copies across availability zones
- **GRS (Geo-Redundant Storage)**: 6 copies across regions
- **GZRS (Geo-Zone-Redundant Storage)**: Combines ZRS and GRS

**Best Practices:**
- Use General-purpose v2 accounts
- Choose appropriate redundancy level
- Enable soft delete for blob storage
- Use lifecycle management policies
- Enable versioning for critical data`,
					CodeExamples: `# Create storage account
az storage account create \\
    --name mystorageaccount \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --sku Standard_LRS

# Create blob container
az storage container create \\
    --name mycontainer \\
    --account-name mystorageaccount \\
    --auth-mode login

# Upload file to blob
az storage blob upload \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --name myfile.txt \\
    --file ./myfile.txt \\
    --auth-mode login

# List blobs
az storage blob list \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --auth-mode login

# Create file share
az storage share create \\
    --name myshare \\
    --account-name mystorageaccount \\
    --auth-mode login`,
				},
				{
					Title: "Blob Storage Advanced Features",
					Content: `Azure Blob Storage offers advanced features for managing object storage efficiently.

**Blob Types:**
- **Block Blobs**: Optimized for streaming and storing text/binary data (up to 4.75 TB)
- **Append Blobs**: Optimized for append operations (logging scenarios)
- **Page Blobs**: Optimized for random read/write (VHD files, up to 8 TB)

**Access Tiers:**
- **Hot Tier**: Frequently accessed data, lowest access cost
- **Cool Tier**: Infrequently accessed data, lower storage cost
- **Archive Tier**: Rarely accessed data, lowest storage cost, retrieval delay
- **Auto-tiering**: Automatically move blobs between tiers

**Blob Lifecycle Management:**
- **Rules**: Automatically transition blobs between tiers
- **Expiration**: Automatically delete blobs after specified time
- **Cost Optimization**: Move old data to cheaper tiers
- **Policy-based**: Define rules based on age, prefix, tags

**Blob Versioning:**
- **Version History**: Keep previous versions of blobs
- **Point-in-Time Restore**: Restore to specific version
- **Protection**: Prevent accidental deletion or overwrite
- **Cost**: Pay for all versions stored

**Blob Snapshots:**
- **Read-only Copy**: Point-in-time copy of blob
- **Incremental**: Only changed blocks stored
- **Manual Creation**: Create snapshots on demand
- **Restore**: Restore blob from snapshot

**Blob Replication:**
- **Geo-replication**: Automatic replication to secondary region
- **Read Access**: Read from secondary region
- **Failover**: Manual failover to secondary region

**Best Practices:**
- Use appropriate access tier for data access patterns
- Enable lifecycle management for cost optimization
- Enable versioning for critical data
- Use snapshots for backup scenarios
- Configure soft delete for protection
- Use blob tags for organization
- Monitor blob metrics and costs`,
					CodeExamples: `# Create blob with specific tier
az storage blob upload \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --name myfile.txt \\
    --file ./myfile.txt \\
    --tier Cool \\
    --auth-mode login

# Change blob tier
az storage blob set-tier \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --name myfile.txt \\
    --tier Archive \\
    --auth-mode login

# Enable versioning
az storage account blob-service-properties update \\
    --account-name mystorageaccount \\
    --enable-versioning true

# Create blob snapshot
az storage blob snapshot \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --name myfile.txt \\
    --auth-mode login

# List blob versions
az storage blob list \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --include versions \\
    --auth-mode login

# Create lifecycle management policy
az storage account management-policy create \\
    --account-name mystorageaccount \\
    --resource-group myResourceGroup \\
    --policy @policy.json

# policy.json example
{
  "rules": [
    {
      "name": "MoveToCool",
      "enabled": true,
      "type": "Lifecycle",
      "definition": {
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["logs/"]
        },
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 30
            }
          }
        }
      }
    }
  ]
}

# Enable soft delete
az storage blob service-properties delete-policy update \\
    --account-name mystorageaccount \\
    --enable-delete-retention true \\
    --delete-retention-days 7

# Using PowerShell
# Upload blob with tier
Set-AzStorageBlobContent \\
    -Container mycontainer \\
    -Blob myfile.txt \\
    -File ./myfile.txt \\
    -StandardBlobTier Cool \\
    -Context $ctx

# Create snapshot
$blob = Get-AzStorageBlob -Container mycontainer -Blob myfile.txt -Context $ctx
$blob.ICloudBlob.CreateSnapshot()`,
				},
				{
					Title: "Storage Security and Encryption",
					Content: `Azure Storage provides multiple layers of security to protect your data.

**Encryption at Rest:**
- **Azure-managed Keys**: Automatic encryption with Microsoft-managed keys
- **Customer-managed Keys**: Use your own keys stored in Key Vault
- **Encryption Scope**: Different encryption keys per container
- **Always Encrypted**: Encryption enabled by default, cannot be disabled

**Encryption in Transit:**
- **HTTPS Required**: All access must use HTTPS
- **TLS Encryption**: All connections use TLS 1.2 or higher
- **SMB 3.0**: Encrypted connections for Azure Files
- **Secure Transfer**: Enable secure transfer required setting

**Access Control:**
- **Shared Access Signatures (SAS)**: Time-limited, scoped access tokens
- **Stored Access Policies**: Manage SAS permissions centrally
- **Azure AD Integration**: Use Azure AD for authentication
- **Storage Account Keys**: Full access keys (rotate regularly)

**SAS Types:**
- **Account SAS**: Access to multiple services in storage account
- **Service SAS**: Access to specific service (Blob, File, Queue, Table)
- **User Delegation SAS**: Uses Azure AD credentials (Blob only)
- **SAS Tokens**: Include permissions, expiry, IP restrictions

**Network Security:**
- **Firewall Rules**: Restrict access by IP address or VNet
- **Private Endpoints**: Private IP addresses for storage
- **Service Endpoints**: Secure connectivity from VNet
- **Trusted Services**: Allow Azure services to access

**Threat Protection:**
- **Advanced Threat Protection**: Detect anomalous access patterns
- **Alerts**: Notifications for suspicious activities
- **Logging**: Detailed access logs for auditing
- **Monitoring**: Track access patterns and anomalies

**Best Practices:**
- Enable encryption at rest with customer-managed keys
- Require HTTPS for all access
- Use Azure AD authentication when possible
- Rotate storage account keys regularly
- Use SAS tokens with minimum required permissions
- Set expiration dates on SAS tokens
- Use private endpoints for sensitive data
- Enable Advanced Threat Protection
- Monitor access logs regularly`,
					CodeExamples: `# Enable secure transfer
az storage account update \\
    --name mystorageaccount \\
    --resource-group myResourceGroup \\
    --https-only true

# Configure customer-managed key
az storage account encryption-scope create \\
    --account-name mystorageaccount \\
    --resource-group myResourceGroup \\
    --encryption-scope-name myEncryptionScope \\
    --key-source Microsoft.KeyVault \\
    --key-uri https://mykeyvault.vault.azure.net/keys/mykey

# Create container with encryption scope
az storage container create \\
    --account-name mystorageaccount \\
    --name mycontainer \\
    --default-encryption-scope myEncryptionScope \\
    --prevent-encryption-scope-override false \\
    --auth-mode login

# Generate SAS token
az storage blob generate-sas \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --name myfile.txt \\
    --permissions r \\
    --expiry 2024-12-31T23:59:59Z \\
    --auth-mode login

# Create stored access policy
az storage container policy create \\
    --account-name mystorageaccount \\
    --container-name mycontainer \\
    --name myPolicy \\
    --permissions r \\
    --expiry 2024-12-31T23:59:59Z \\
    --auth-mode login

# Configure firewall rules
az storage account network-rule add \\
    --account-name mystorageaccount \\
    --resource-group myResourceGroup \\
    --ip-address 1.2.3.4

# Add VNet rule
az storage account network-rule add \\
    --account-name mystorageaccount \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --subnet mySubnet

# Create private endpoint
az network private-endpoint create \\
    --resource-group myResourceGroup \\
    --name myPrivateEndpoint \\
    --vnet-name myVNet \\
    --subnet mySubnet \\
    --private-connection-resource-id /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Storage/storageAccounts/mystorageaccount \\
    --group-id blob \\
    --connection-name myConnection

# Rotate storage account keys
az storage account keys renew \\
    --account-name mystorageaccount \\
    --resource-group myResourceGroup \\
    --key secondary

# Using PowerShell
# Enable secure transfer
Set-AzStorageAccount \\
    -ResourceGroupName myResourceGroup \\
    -Name mystorageaccount \\
    -EnableHttpsTrafficOnly $true

# Generate SAS token
$ctx = New-AzStorageContext -StorageAccountName mystorageaccount -StorageAccountKey $key
New-AzStorageBlobSASToken \\
    -Container mycontainer \\
    -Blob myfile.txt \\
    -Permission r \\
    -ExpiryTime (Get-Date).AddDays(30) \\
    -Context $ctx`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          404,
			Title:       "Azure App Service",
			Description: "Learn App Service: web apps, API apps, deployment slots, and scaling.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "App Service Fundamentals",
					Content: `Azure App Service is a fully managed platform for building, deploying, and scaling web apps.

**App Service Types:**
- **Web Apps**: Host web applications
- **API Apps**: Host RESTful APIs
- **Mobile Apps**: Backend for mobile applications
- **Function Apps**: Serverless functions (Azure Functions)

**Supported Platforms:**
- .NET, .NET Core
- Java
- Node.js
- Python
- PHP
- Ruby

**App Service Plans:**
- **Free/Shared**: Development and testing
- **Basic**: Production workloads, manual scaling
- **Standard**: Production workloads, auto-scaling, deployment slots
- **Premium**: High-performance, VNet integration
- **Isolated**: Dedicated environment, highest security

**Features:**
- Continuous deployment from Git, GitHub, Azure DevOps
- Deployment slots for staging
- Auto-scaling based on metrics
- Custom domains and SSL certificates
- Authentication and authorization
- Application Insights integration

**Best Practices:**
- Use deployment slots for zero-downtime deployments
- Enable auto-scaling for variable workloads
- Use Application Insights for monitoring
- Configure custom domains with SSL
- Enable backup and restore`,
					CodeExamples: `# Create App Service plan
az appservice plan create \\
    --name myAppServicePlan \\
    --resource-group myResourceGroup \\
    --sku B1

# Create web app
az webapp create \\
    --resource-group myResourceGroup \\
    --plan myAppServicePlan \\
    --name mywebapp \\
    --runtime "node|14-lts"

# Deploy from Git
az webapp deployment source config \\
    --name mywebapp \\
    --resource-group myResourceGroup \\
    --repo-url https://github.com/user/repo.git \\
    --branch master \\
    --manual-integration

# Create deployment slot
az webapp deployment slot create \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging

# Swap slots
az webapp deployment slot swap \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging \\
    --target-slot production`,
				},
				{
					Title: "Deployment Strategies",
					Content: `App Service supports multiple deployment strategies for different scenarios and requirements.

**Deployment Slots:**
- **Production Slot**: Default slot for live application
- **Staging Slots**: Additional slots for testing and staging
- **Slot Swapping**: Zero-downtime deployments
- **Warm-up**: Pre-warm slots before swapping
- **Traffic Routing**: Route percentage of traffic to slots

**Deployment Methods:**
- **Git Deployment**: Direct push to App Service Git repository
- **GitHub Actions**: Automated CI/CD with GitHub
- **Azure DevOps**: Full CI/CD pipeline integration
- **FTP/FTPS**: Manual file upload
- **Zip Deploy**: Deploy application package
- **Container Deployment**: Deploy from container registry

**Deployment Slots Benefits:**
- **Zero-Downtime**: Swap slots without downtime
- **Testing**: Test in production-like environment
- **Rollback**: Quick rollback by swapping back
- **A/B Testing**: Route traffic between slots
- **Blue-Green Deployment**: Deploy to new slot, swap when ready

**Slot Configuration:**
- **App Settings**: Can differ between slots
- **Connection Strings**: Environment-specific values
- **Sticky Settings**: Settings that don't swap
- **Swap with Preview**: Validate before completing swap

**Best Practices:**
- Use deployment slots for production apps
- Test in staging slot before swapping
- Use slot-specific app settings
- Enable auto-swap for non-critical apps
- Monitor deployment logs
- Use deployment center for CI/CD
- Implement health checks before swap
- Document deployment procedures`,
					CodeExamples: `# Create deployment slot
az webapp deployment slot create \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging

# Deploy to slot
az webapp deployment source config-zip \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging \\
    --src app.zip

# Swap slots with preview
az webapp deployment slot swap \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging \\
    --target-slot production \\
    --preview

# Complete swap
az webapp deployment slot swap \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging \\
    --target-slot production \\
    --action swap

# Configure slot-specific app setting
az webapp config appsettings set \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging \\
    --settings ENVIRONMENT=Staging

# Enable auto-swap
az webapp deployment slot auto-swap \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --slot staging \\
    --auto-swap-slot production

# Deploy from GitHub
az webapp deployment source config \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --repo-url https://github.com/user/repo.git \\
    --branch master \\
    --manual-integration

# Using PowerShell
# Create slot
New-AzWebAppSlot \\
    -ResourceGroupName myResourceGroup \\
    -Name mywebapp \\
    -Slot staging

# Swap slots
Switch-AzWebAppSlot \\
    -ResourceGroupName myResourceGroup \\
    -Name mywebapp \\
    -SourceSlotName staging \\
    -DestinationSlotName production`,
				},
				{
					Title: "App Service Authentication",
					Content: `App Service provides built-in authentication and authorization without writing code.

**Authentication Providers:**
- **Azure Active Directory**: Enterprise identity provider
- **Microsoft Account**: Personal Microsoft accounts
- **Facebook**: Facebook authentication
- **Google**: Google accounts
- **Twitter**: Twitter accounts
- **GitHub**: GitHub accounts
- **OpenID Connect**: Custom OpenID Connect providers

**Authentication Flow:**
- **Redirect-based**: Users redirected to provider
- **Token-based**: Provider returns tokens
- **Session Management**: App Service manages sessions
- **Token Storage**: Tokens stored in app settings

**Authorization:**
- **Unauthenticated Requests**: Allow or require authentication
- **Action**: Redirect to login or return 401
- **Allowed Redirect URLs**: Configure redirect after login
- **Token Store**: Enable token storage for app access

**Easy Auth Features:**
- **No Code Required**: Configure in portal or CLI
- **Automatic Token Injection**: Tokens available in headers
- **Session Management**: Automatic session handling
- **Multi-provider**: Support multiple providers

**Token Access:**
- **Headers**: Access tokens in request headers
- **X-MS-CLIENT-PRINCIPAL**: User principal information
- **X-MS-TOKEN-***: Provider-specific tokens
- **Environment Variables**: Tokens in app settings

**Best Practices:**
- Use Azure AD for enterprise apps
- Enable token store for app access
- Configure allowed redirect URLs
- Use HTTPS for all authentication
- Monitor authentication logs
- Implement proper error handling
- Test authentication flows thoroughly
- Document authentication configuration`,
					CodeExamples: `# Enable authentication with Azure AD
az webapp auth update \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --enabled true \\
    --action LoginWithAzureActiveDirectory \\
    --aad-client-id <client-id> \\
    --aad-client-secret <client-secret> \\
    --aad-token-issuer-url https://sts.windows.net/<tenant-id>/

# Configure authentication settings
az webapp auth update \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --unauthenticated-client-action RedirectToLoginPage \\
    --token-store true

# Add Google provider
az webapp auth google update \\
    --resource-group myResourceGroup \\
    --name mywebapp \\
    --client-id <google-client-id> \\
    --client-secret <google-client-secret> \\
    --scopes openid email profile

# View authentication settings
az webapp auth show \\
    --resource-group myResourceGroup \\
    --name mywebapp

# Using PowerShell
# Enable authentication
Set-AzWebApp \\
    -ResourceGroupName myResourceGroup \\
    -Name mywebapp \\
    -IdentityType SystemAssigned

# Note: Authentication configuration is typically done via Azure Portal
# or ARM templates for complex scenarios`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          405,
			Title:       "Azure SQL Database",
			Description: "Learn Azure SQL Database: managed SQL databases, elastic pools, and high availability.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "SQL Database Fundamentals",
					Content: `Azure SQL Database is a fully managed relational database service.

**Deployment Options:**
- **Single Database**: Isolated database with dedicated resources
- **Elastic Pool**: Shared resources across multiple databases
- **Managed Instance**: Full SQL Server instance compatibility

**Service Tiers:**
- **General Purpose**: Balanced compute and storage
- **Business Critical**: High availability with read replicas
- **Hyperscale**: Auto-scaling storage up to 100TB

**Compute Tiers:**
- **Serverless**: Auto-pause, pay per second
- **Provisioned**: Fixed compute capacity

**Features:**
- Automated backups (point-in-time restore)
- High availability (99.995% SLA)
- Geo-replication for disaster recovery
- Advanced threat protection
- Automatic tuning
- Built-in intelligence

**Best Practices:**
- Use elastic pools for multiple databases
- Enable geo-replication for DR
- Use automated tuning
- Monitor performance metrics
- Enable Advanced Data Security
- Regular backup testing`,
					CodeExamples: `# Create SQL server
az sql server create \\
    --name myserver \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --admin-user sqladmin \\
    --admin-password SecurePassword123!

# Create SQL database
az sql db create \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --service-objective S0

# Create elastic pool
az sql elastic-pool create \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mypool \\
    --edition GeneralPurpose \\
    --capacity 2 \\
    --db-dtu-min 0 \\
    --db-dtu-max 100

# Add database to pool
az sql db update \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --elastic-pool mypool`,
				},
				{
					Title: "Performance Tuning",
					Content: `Azure SQL Database provides multiple tools and features for optimizing database performance.

**Performance Monitoring:**
- **Query Performance Insight**: Identify top resource-consuming queries
- **Automatic Tuning**: AI-powered performance optimization
- **DMVs**: Dynamic Management Views for performance analysis
- **Extended Events**: Detailed event tracking
- **Azure Monitor**: Comprehensive monitoring and alerting

**Automatic Tuning:**
- **FORCE_LAST_GOOD_PLAN**: Automatically fix query plan regressions
- **CREATE_INDEX**: Automatically create missing indexes
- **DROP_INDEX**: Remove unused indexes
- **Auto-apply**: Automatically apply recommendations
- **Review**: Manual review before applying

**Index Management:**
- **Missing Indexes**: Identify and create missing indexes
- **Unused Indexes**: Remove indexes that aren't used
- **Index Fragmentation**: Monitor and rebuild indexes
- **Columnstore Indexes**: For analytical workloads

**Query Optimization:**
- **Query Store**: Track query performance over time
- **Plan Forcing**: Force specific execution plans
- **Parameter Sniffing**: Optimize parameterized queries
- **Statistics**: Keep statistics up to date

**Resource Scaling:**
- **DTU Model**: Scale compute and storage together
- **vCore Model**: Scale compute and storage independently
- **Serverless**: Auto-pause and auto-scale
- **Hyperscale**: Auto-scale storage up to 100TB

**Best Practices:**
- Enable automatic tuning
- Monitor Query Performance Insight regularly
- Review and apply tuning recommendations
- Use appropriate service tier for workload
- Keep statistics updated
- Use parameterized queries
- Monitor resource utilization
- Scale resources proactively`,
					CodeExamples: `# Enable automatic tuning
az sql db automatic-tuning update \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --state Enabled \\
    --options createIndex=Enabled dropIndex=Enabled forceLastGoodPlan=Enabled

# View automatic tuning recommendations
az sql db automatic-tuning show \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase

# Scale database (DTU model)
az sql db update \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --service-objective S2

# Scale database (vCore model)
az sql db update \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --capacity 4 \\
    --tier GeneralPurpose

# Enable Query Store
az sql db query-store update \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --status Enabled \\
    --query-capture-mode All

# View top queries
az sql db query-performance-insights show \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase

# Using PowerShell
# Enable automatic tuning
Set-AzSqlDatabaseAutomaticTuning \\
    -ResourceGroupName myResourceGroup \\
    -ServerName myserver \\
    -DatabaseName mydatabase \\
    -State Enabled \\
    -CreateIndex Enabled \\
    -DropIndex Enabled \\
    -ForceLastGoodPlan Enabled`,
				},
				{
					Title: "Disaster Recovery",
					Content: `Azure SQL Database provides comprehensive disaster recovery options to protect your data.

**Geo-Replication:**
- **Active Geo-Replication**: Up to 4 readable secondary databases
- **Automatic Failover Groups**: Automatic failover for multiple databases
- **Readable Secondaries**: Offload read workloads to secondary
- **Near-zero RPO**: Minimal data loss on failover
- **Cross-region**: Replicate to different Azure regions

**Backup and Restore:**
- **Automated Backups**: Point-in-time restore up to 35 days
- **Long-term Retention**: Store backups for up to 10 years
- **Geo-redundant Backups**: Backups stored in paired region
- **Restore Options**: Restore to new database or existing

**Failover Groups:**
- **Automatic Failover**: Failover without data loss
- **Read-write Listener**: Single endpoint for applications
- **Multiple Databases**: Group databases for coordinated failover
- **Failover Policy**: Automatic or manual failover

**Business Continuity:**
- **RPO (Recovery Point Objective)**: Amount of data loss acceptable
- **RTO (Recovery Time Objective)**: Time to restore service
- **High Availability**: Built-in redundancy within region
- **Zone Redundancy**: Replicate across availability zones

**Best Practices:**
- Use geo-replication for critical databases
- Configure failover groups for automatic failover
- Test failover procedures regularly
- Monitor replication lag
- Use readable secondaries for reporting
- Enable long-term backup retention
- Document recovery procedures
- Regular backup testing`,
					CodeExamples: `# Create active geo-replica
az sql db replica create \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --partner-server mysecondaryserver \\
    --partner-resource-group myResourceGroup \\
    --partner-database mydatabase-secondary

# Create failover group
az sql failover-group create \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name myFailoverGroup \\
    --partner-server mysecondaryserver \\
    --partner-resource-group myResourceGroup \\
    --failover-policy Automatic \\
    --grace-period 1

# Add database to failover group
az sql failover-group update \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name myFailoverGroup \\
    --add-db mydatabase

# Manual failover
az sql failover-group set-primary \\
    --resource-group myResourceGroup \\
    --server mysecondaryserver \\
    --name myFailoverGroup

# Restore database to point in time
az sql db restore \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase-restored \\
    --dest-name mydatabase \\
    --time "2024-01-15T10:00:00Z"

# Configure long-term backup retention
az sql db ltr-policy set \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name mydatabase \\
    --weekly-retention "P4W" \\
    --monthly-retention "P12M" \\
    --yearly-retention "P5Y" \\
    --week-of-year 26

# Using PowerShell
# Create geo-replica
New-AzSqlDatabaseSecondary \\
    -ResourceGroupName myResourceGroup \\
    -ServerName myserver \\
    -DatabaseName mydatabase \\
    -PartnerResourceGroupName myResourceGroup \\
    -PartnerServerName mysecondaryserver`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          406,
			Title:       "Azure Virtual Network",
			Description: "Learn Virtual Networks: subnets, network security groups, and connectivity options.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Virtual Network Fundamentals",
					Content: `Azure Virtual Network (VNet) enables Azure resources to securely communicate with each other and the internet.

**VNet Concepts:**
- **Virtual Network**: Isolated network in Azure
- **Subnet**: Segment of VNet address space
- **Network Security Group**: Virtual firewall rules
- **Route Table**: Custom routing rules
- **Service Endpoints**: Private connectivity to Azure services

**Address Spaces:**
- Use private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Plan address space to avoid conflicts
- Use CIDR notation for subnets

**Connectivity Options:**
- **VNet Peering**: Connect VNets in same or different regions
- **VPN Gateway**: Site-to-site or point-to-site VPN
- **ExpressRoute**: Private connection to Azure
- **Virtual WAN**: Centralized network management

**Network Security Groups:**
- Inbound and outbound security rules
- Source and destination IP ranges
- Port and protocol filtering
- Priority-based rule evaluation

**Best Practices:**
- Plan address space carefully
- Use Network Security Groups for security
- Implement hub-spoke architecture for large deployments
- Use service endpoints for private connectivity
- Monitor network traffic`,
					CodeExamples: `# Create virtual network
az network vnet create \\
    --resource-group myResourceGroup \\
    --name myVNet \\
    --address-prefix 10.0.0.0/16 \\
    --subnet-name mySubnet \\
    --subnet-prefix 10.0.1.0/24

# Create subnet
az network vnet subnet create \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --name mySubnet2 \\
    --address-prefix 10.0.2.0/24

# Create network security group
az network nsg create \\
    --resource-group myResourceGroup \\
    --name myNSG

# Add NSG rule
az network nsg rule create \\
    --resource-group myResourceGroup \\
    --nsg-name myNSG \\
    --name AllowSSH \\
    --priority 1000 \\
    --protocol Tcp \\
    --destination-port-ranges 22 \\
    --access Allow

# Associate NSG with subnet
az network vnet subnet update \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --name mySubnet \\
    --network-security-group myNSG`,
				},
				{
					Title: "Network Security Groups",
					Content: `Network Security Groups (NSGs) provide network-level access control for Azure resources.

**NSG Concepts:**
- **Security Rules**: Allow or deny traffic based on rules
- **Priority**: Rules evaluated in priority order (lower number = higher priority)
- **Direction**: Inbound or outbound rules
- **Source/Destination**: IP addresses, service tags, or application security groups
- **Port and Protocol**: TCP, UDP, ICMP, or Any

**Default Rules:**
- **Allow VNet Inbound**: Allow all inbound traffic from VNet
- **Allow Azure Load Balancer Inbound**: Allow Azure load balancer traffic
- **Deny All Inbound**: Deny all other inbound traffic
- **Allow All Outbound**: Allow all outbound traffic

**Service Tags:**
- **VirtualNetwork**: All VNet addresses
- **AzureLoadBalancer**: Azure load balancer
- **Internet**: Public internet
- **AzureCloud**: Azure public IP addresses
- **Storage**: Azure Storage service
- **Sql**: Azure SQL Database

**Application Security Groups:**
- **Purpose**: Group VMs for rule management
- **Simplified Rules**: Reference groups instead of IPs
- **Dynamic**: Rules apply to all VMs in group
- **Scalable**: Add VMs without changing rules

**Rule Evaluation:**
- **Process Order**: Rules evaluated by priority
- **First Match**: First matching rule applies
- **Default Deny**: If no rule matches, traffic denied
- **Explicit Allow**: Must have allow rule for traffic

**Best Practices:**
- Use service tags when possible
- Use application security groups for VM groups
- Deny by default, allow explicitly
- Document rule purposes
- Use descriptive rule names
- Review and audit rules regularly
- Test rules in non-production first
- Use NSG flow logs for troubleshooting`,
					CodeExamples: `# Create NSG
az network nsg create \\
    --resource-group myResourceGroup \\
    --name myNSG \\
    --location eastus

# Add inbound rule for SSH
az network nsg rule create \\
    --resource-group myResourceGroup \\
    --nsg-name myNSG \\
    --name AllowSSH \\
    --priority 1000 \\
    --protocol Tcp \\
    --source-address-prefixes Internet \\
    --source-port-ranges * \\
    --destination-address-prefixes * \\
    --destination-port-ranges 22 \\
    --access Allow

# Add inbound rule for HTTP
az network nsg rule create \\
    --resource-group myResourceGroup \\
    --nsg-name myNSG \\
    --name AllowHTTP \\
    --priority 1010 \\
    --protocol Tcp \\
    --source-address-prefixes Internet \\
    --source-port-ranges * \\
    --destination-address-prefixes * \\
    --destination-port-ranges 80 \\
    --access Allow

# Add rule using service tag
az network nsg rule create \\
    --resource-group myResourceGroup \\
    --nsg-name myNSG \\
    --name AllowStorage \\
    --priority 1020 \\
    --protocol Tcp \\
    --source-address-prefixes VirtualNetwork \\
    --source-port-ranges * \\
    --destination-address-prefixes Storage \\
    --destination-port-ranges 443 \\
    --access Allow

# Create application security group
az network asg create \\
    --resource-group myResourceGroup \\
    --name myASG \\
    --location eastus

# Add VM to application security group
az vm nic update \\
    --resource-group myResourceGroup \\
    --vm-name myVM \\
    --nic-name myNIC \\
    --ip-configurations ipconfig1 \\
    --application-security-groups myASG

# Add rule using application security group
az network nsg rule create \\
    --resource-group myResourceGroup \\
    --nsg-name myNSG \\
    --name AllowASG \\
    --priority 1030 \\
    --protocol Tcp \\
    --source-address-prefixes VirtualNetwork \\
    --source-port-ranges * \\
    --destination-application-security-groups myASG \\
    --destination-port-ranges 443 \\
    --access Allow

# Using PowerShell
# Create NSG
New-AzNetworkSecurityGroup \\
    -ResourceGroupName myResourceGroup \\
    -Name myNSG \\
    -Location eastus

# Add rule
$nsg = Get-AzNetworkSecurityGroup -ResourceGroupName myResourceGroup -Name myNSG
Add-AzNetworkSecurityRuleConfig \\
    -NetworkSecurityGroup $nsg \\
    -Name AllowSSH \\
    -Priority 1000 \\
    -Protocol Tcp \\
    -Direction Inbound \\
    -Access Allow \\
    -SourceAddressPrefix Internet \\
    -SourcePortRange * \\
    -DestinationAddressPrefix * \\
    -DestinationPortRange 22
Set-AzNetworkSecurityGroup -NetworkSecurityGroup $nsg`,
				},
				{
					Title: "Service Endpoints",
					Content: `Service Endpoints provide secure, direct connectivity to Azure services from your virtual network.

**Service Endpoint Concepts:**
- **Private Connectivity**: Traffic stays on Azure backbone
- **No Public IPs**: Services accessed via private IP addresses
- **VNet Integration**: Extend VNet identity to Azure services
- **Service Tags**: Simplified firewall rules

**Supported Services:**
- **Azure Storage**: Blob, File, Queue, Table storage
- **Azure SQL Database**: SQL Database and SQL Data Warehouse
- **Azure Cosmos DB**: Cosmos DB accounts
- **Azure Key Vault**: Key Vault service
- **Azure Service Bus**: Service Bus namespaces
- **Azure Event Hubs**: Event Hubs namespaces

**Benefits:**
- **Security**: Traffic doesn't traverse internet
- **Performance**: Optimized routing on Azure backbone
- **Cost**: No data transfer charges for service endpoints
- **Simplified**: No need for public IPs or NAT gateways

**Configuration:**
- **Subnet Level**: Enable on specific subnets
- **Service Level**: Enable for specific Azure services
- **Firewall Rules**: Configure service firewall to allow VNet
- **Policies**: Use Azure Policy to enforce endpoints

**Best Practices:**
- Enable service endpoints for all supported services
- Use service tags in NSG rules
- Configure service firewalls to allow VNet
- Monitor service endpoint status
- Use private endpoints for additional security
- Document endpoint configurations
- Test connectivity after enabling
- Review service endpoint policies regularly`,
					CodeExamples: `# Enable service endpoint for Storage on subnet
az network vnet subnet update \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --name mySubnet \\
    --service-endpoints Microsoft.Storage

# Enable service endpoint for SQL Database
az network vnet subnet update \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --name mySubnet \\
    --service-endpoints Microsoft.Sql

# Configure storage account firewall to allow VNet
az storage account network-rule add \\
    --account-name mystorageaccount \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --subnet mySubnet

# Configure SQL server firewall to allow VNet
az sql server firewall-rule create \\
    --resource-group myResourceGroup \\
    --server myserver \\
    --name AllowVNet \\
    --subnet mySubnet \\
    --vnet-name myVNet

# List service endpoints on subnet
az network vnet subnet show \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --name mySubnet \\
    --query "serviceEndpoints"

# Disable service endpoint
az network vnet subnet update \\
    --resource-group myResourceGroup \\
    --vnet-name myVNet \\
    --name mySubnet \\
    --service-endpoints ""

# Using PowerShell
# Enable service endpoint
$subnet = Get-AzVirtualNetworkSubnetConfig \\
    -VirtualNetwork $vnet \\
    -Name mySubnet
$subnet.ServiceEndpoints = @(
    @{
        Service = "Microsoft.Storage"
    },
    @{
        Service = "Microsoft.Sql"
    }
)
Set-AzVirtualNetworkSubnetConfig \\
    -VirtualNetwork $vnet \\
    -Name mySubnet \\
    -AddressPrefix $subnet.AddressPrefix \\
    -ServiceEndpoint $subnet.ServiceEndpoints
Set-AzVirtualNetwork -VirtualNetwork $vnet`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          407,
			Title:       "Azure Load Balancer",
			Description: "Learn Load Balancer: distribution algorithms, health probes, and high availability.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Load Balancer Fundamentals",
					Content: `Azure Load Balancer distributes incoming traffic across multiple backend resources.

**Load Balancer Types:**
- **Public Load Balancer**: Internet-facing load balancing
- **Internal Load Balancer**: Internal-only load balancing

**Distribution Algorithms:**
- **5-tuple hash**: Source IP, source port, destination IP, destination port, protocol
- **3-tuple hash**: Source IP, destination IP, protocol
- **Source IP affinity**: Sticky sessions based on source IP

**Health Probes:**
- **HTTP/HTTPS probe**: Check HTTP response
- **TCP probe**: Check TCP connection
- Configure probe interval and timeout
- Unhealthy threshold determines removal

**Load Balancing Rules:**
- Frontend IP and port
- Backend pool
- Health probe
- Distribution algorithm
- Idle timeout

**Best Practices:**
- Use health probes for accurate health checks
- Configure appropriate idle timeout
- Use multiple backend instances for high availability
- Monitor load balancer metrics
- Use Standard SKU for production`,
					CodeExamples: `# Create public IP
az network public-ip create \\
    --resource-group myResourceGroup \\
    --name myPublicIP \\
    --sku Standard \\
    --allocation-method Static

# Create load balancer
az network lb create \\
    --resource-group myResourceGroup \\
    --name myLoadBalancer \\
    --sku Standard \\
    --public-ip-address myPublicIP \\
    --frontend-ip-name myFrontEnd \\
    --backend-pool-name myBackEndPool

# Create health probe
az network lb probe create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHealthProbe \\
    --protocol Http \\
    --port 80 \\
    --path /

# Create load balancing rule
az network lb rule create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHTTPRule \\
    --protocol Tcp \\
    --frontend-port 80 \\
    --backend-port 80 \\
    --frontend-ip-name myFrontEnd \\
    --backend-pool-name myBackEndPool \\
    --probe-name myHealthProbe`,
				},
				{
					Title: "Health Probes Configuration",
					Content: `Health probes are critical for ensuring load balancer only routes traffic to healthy backend instances.

**Probe Types:**
- **HTTP Probe**: Checks HTTP response code (200-399 considered healthy)
- **HTTPS Probe**: Secure HTTP probe with TLS
- **TCP Probe**: Checks TCP connection success

**Probe Configuration:**
- **Interval**: How often to check (5-3600 seconds)
- **Timeout**: Time to wait for response (5-120 seconds)
- **Unhealthy Threshold**: Consecutive failures before marking unhealthy (1-20)
- **Path**: HTTP path to check (for HTTP/HTTPS probes)
- **Port**: Port to check (can differ from load balancing rule port)

**Health States:**
- **Healthy**: Instance receives traffic
- **Unhealthy**: Instance removed from rotation
- **Probe Down**: Probe itself is failing

**Best Practices:**
- Use HTTP probes for application health checks
- Create dedicated health check endpoint (/health)
- Set appropriate interval (15-30 seconds typical)
- Configure timeout less than interval
- Use unhealthy threshold of 2-3
- Monitor probe success rates
- Test probe behavior during deployments`,
					CodeExamples: `# Create HTTP health probe
az network lb probe create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHealthProbe \\
    --protocol Http \\
    --port 80 \\
    --path /health \\
    --interval 15 \\
    --threshold 2

# Create HTTPS probe
az network lb probe create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHTTPSProbe \\
    --protocol Https \\
    --port 443 \\
    --path /api/health \\
    --interval 30 \\
    --threshold 3

# Create TCP probe
az network lb probe create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myTCPProbe \\
    --protocol Tcp \\
    --port 22 \\
    --interval 10 \\
    --threshold 2

# Update probe configuration
az network lb probe update \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHealthProbe \\
    --interval 20 \\
    --threshold 3

# View probe status
az network lb probe show \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHealthProbe`,
				},
				{
					Title: "Load Balancing Rules",
					Content: `Load balancing rules define how traffic is distributed from frontend to backend resources.

**Rule Components:**
- **Frontend IP**: Public or private IP address
- **Frontend Port**: Port on load balancer
- **Backend Port**: Port on backend instances
- **Backend Pool**: Group of backend instances
- **Health Probe**: Probe to check backend health
- **Protocol**: TCP or UDP
- **Idle Timeout**: Connection idle timeout (4-30 minutes)

**Distribution Methods:**
- **5-tuple Hash**: Source IP, source port, destination IP, destination port, protocol
- **3-tuple Hash**: Source IP, destination IP, protocol
- **Source IP Affinity**: Sticky sessions based on source IP

**Port Mapping:**
- **Same Port**: Frontend and backend use same port
- **Port Translation**: Different frontend and backend ports
- **Floating IP**: Direct server return (DSR) mode

**High Availability Ports:**
- **All Ports**: Single rule for all ports (UDP and TCP)
- **Internal Load Balancer**: For network virtual appliances
- **Port Range**: Specify range of ports

**Best Practices:**
- Use health probes with all rules
- Configure appropriate idle timeout
- Use 5-tuple hash for stateless applications
- Use source IP affinity for stateful applications
- Monitor rule metrics
- Test failover scenarios
- Document rule purposes`,
					CodeExamples: `# Create load balancing rule
az network lb rule create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHTTPRule \\
    --protocol Tcp \\
    --frontend-port 80 \\
    --backend-port 8080 \\
    --frontend-ip-name myFrontEnd \\
    --backend-pool-name myBackEndPool \\
    --probe-name myHealthProbe \\
    --idle-timeout 15

# Create rule with port translation
az network lb rule create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHTTPSRule \\
    --protocol Tcp \\
    --frontend-port 443 \\
    --backend-port 8443 \\
    --frontend-ip-name myFrontEnd \\
    --backend-pool-name myBackEndPool \\
    --probe-name myHTTPSProbe

# Create rule with source IP affinity
az network lb rule create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myStickyRule \\
    --protocol Tcp \\
    --frontend-port 8080 \\
    --backend-port 8080 \\
    --frontend-ip-name myFrontEnd \\
    --backend-pool-name myBackEndPool \\
    --probe-name myHealthProbe \\
    --load-distribution SourceIP

# Create high availability ports rule
az network lb rule create \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHAPortsRule \\
    --protocol All \\
    --frontend-port 0 \\
    --backend-port 0 \\
    --frontend-ip-name myFrontEnd \\
    --backend-pool-name myBackEndPool \\
    --probe-name myHealthProbe

# Update rule
az network lb rule update \\
    --resource-group myResourceGroup \\
    --lb-name myLoadBalancer \\
    --name myHTTPRule \\
    --idle-timeout 20`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          408,
			Title:       "Azure Key Vault",
			Description: "Learn Key Vault: secrets, keys, certificates, and secure access management.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Key Vault Fundamentals",
					Content: `Azure Key Vault provides secure storage for secrets, keys, and certificates.

**Key Vault Objects:**
- **Secrets**: Passwords, connection strings, API keys
- **Keys**: Cryptographic keys for encryption
- **Certificates**: SSL/TLS certificates

**Access Control:**
- **Access Policies**: Legacy RBAC model
- **Role-Based Access Control**: Modern Azure RBAC
- **Network Access**: Firewall rules and private endpoints

**Key Vault Features:**
- Hardware Security Modules (HSM) support
- Soft delete and purge protection
- Versioning for secrets and keys
- Access logging and monitoring
- Integration with Azure services

**Best Practices:**
- Use Managed Identity for access
- Enable soft delete and purge protection
- Use private endpoints for secure access
- Rotate secrets regularly
- Monitor access logs
- Use separate vaults for different environments`,
					CodeExamples: `# Create Key Vault
az keyvault create \\
    --name myKeyVault \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --enable-soft-delete true \\
    --enable-purge-protection true

# Store secret
az keyvault secret set \\
    --vault-name myKeyVault \\
    --name mySecret \\
    --value "MySecretValue"

# Retrieve secret
az keyvault secret show \\
    --vault-name myKeyVault \\
    --name mySecret \\
    --query value \\
    --output tsv

# Create key
az keyvault key create \\
    --vault-name myKeyVault \\
    --name myKey \\
    --protection software

# Import certificate
az keyvault certificate import \\
    --vault-name myKeyVault \\
    --name myCert \\
    --file certificate.pfx \\
    --password "CertPassword"`,
				},
				{
					Title: "Access Policies",
					Content: `Key Vault access policies control who can access secrets, keys, and certificates.

**Access Policy Model:**
- **Legacy Model**: Access policies attached to Key Vault
- **Permissions**: Get, List, Set, Delete, Backup, Restore, etc.
- **Principal**: User, group, or service principal
- **Resource Types**: Secrets, Keys, Certificates

**RBAC Model:**
- **Modern Approach**: Azure RBAC roles
- **Built-in Roles**: Key Vault Contributor, Key Vault Secrets Officer, etc.
- **Custom Roles**: Define custom permissions
- **Scope**: Subscription, resource group, or Key Vault level

**Permission Types:**
- **Secrets**: Get, List, Set, Delete, Backup, Restore, Recover, Purge
- **Keys**: Get, List, Create, Update, Delete, Backup, Restore, Recover, Purge, Encrypt, Decrypt, Sign, Verify, Wrap, Unwrap
- **Certificates**: Get, List, Create, Update, Delete, ManageContacts, ManageIssuers, GetIssuers, ListIssuers, SetIssuers, DeleteIssuers, Purge

**Best Practices:**
- Use RBAC model for new deployments
- Follow principle of least privilege
- Use managed identities when possible
- Regularly review access policies
- Remove unused access policies
- Use separate vaults for different environments
- Document access policy purposes
- Monitor access logs`,
					CodeExamples: `# Create access policy (legacy model)
az keyvault set-policy \\
    --name myKeyVault \\
    --resource-group myResourceGroup \\
    --upn user@contoso.com \\
    --secret-permissions get list set \\
    --key-permissions get list \\
    --certificate-permissions get list

# Add access policy for service principal
az keyvault set-policy \\
    --name myKeyVault \\
    --resource-group myResourceGroup \\
    --spn <service-principal-id> \\
    --secret-permissions get list

# Remove access policy
az keyvault delete-policy \\
    --name myKeyVault \\
    --resource-group myResourceGroup \\
    --upn user@contoso.com

# Assign RBAC role
az role assignment create \\
    --role "Key Vault Secrets Officer" \\
    --assignee user@contoso.com \\
    --scope /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.KeyVault/vaults/myKeyVault

# List access policies
az keyvault show \\
    --name myKeyVault \\
    --resource-group myResourceGroup \\
    --query "properties.accessPolicies"

# Using PowerShell
# Set access policy
Set-AzKeyVaultAccessPolicy \\
    -VaultName myKeyVault \\
    -ResourceGroupName myResourceGroup \\
    -UserPrincipalName user@contoso.com \\
    -PermissionsToSecrets get,list,set \\
    -PermissionsToKeys get,list`,
				},
				{
					Title: "Certificate Management",
					Content: `Key Vault provides comprehensive certificate lifecycle management.

**Certificate Types:**
- **Self-signed**: Generated by Key Vault
- **CA-signed**: Issued by certificate authority
- **Import**: Import existing certificates
- **Managed**: Automatically renewed by Key Vault

**Certificate Lifecycle:**
- **Creation**: Generate or import certificate
- **Activation**: Certificate becomes valid
- **Renewal**: Automatic or manual renewal
- **Expiration**: Certificate expires
- **Revocation**: Revoke before expiration

**Certificate Policies:**
- **Issuer**: Certificate authority (DigiCert, GlobalSign, etc.)
- **Validity Period**: Certificate lifetime
- **Key Properties**: Key type, key size, exportable
- **Subject Name**: Certificate subject
- **SANs**: Subject Alternative Names

**Auto-renewal:**
- **Managed Certificates**: Automatic renewal by Key Vault
- **Email Notifications**: Alerts before expiration
- **Renewal Threshold**: Percentage of lifetime remaining
- **CA Integration**: Direct integration with CAs

**Best Practices:**
- Use managed certificates for automatic renewal
- Set up email notifications for expiration
- Monitor certificate expiration dates
- Use appropriate validity periods
- Store certificates in Key Vault for security
- Document certificate purposes
- Test certificate renewal process
- Have backup certificates ready`,
					CodeExamples: `# Create self-signed certificate
az keyvault certificate create \\
    --vault-name myKeyVault \\
    --name myCert \\
    --policy '{
        "x509CertificateProperties": {
            "subject": "CN=www.contoso.com",
            "validityInMonths": 12
        },
        "issuerParameters": {
            "name": "Self"
        },
        "keyProperties": {
            "exportable": true,
            "keySize": 2048,
            "keyType": "RSA"
        }
    }'

# Import certificate
az keyvault certificate import \\
    --vault-name myKeyVault \\
    --name myCert \\
    --file certificate.pfx \\
    --password "CertPassword"

# Create managed certificate
az keyvault certificate create \\
    --vault-name myKeyVault \\
    --name myManagedCert \\
    --policy '{
        "x509CertificateProperties": {
            "subject": "CN=www.contoso.com",
            "validityInMonths": 12
        },
        "issuerParameters": {
            "name": "DigiCert",
            "cty": "OV-SSL"
        },
        "keyProperties": {
            "exportable": true,
            "keySize": 2048,
            "keyType": "RSA"
        }
    }'

# Get certificate
az keyvault certificate show \\
    --vault-name myKeyVault \\
    --name myCert

# Download certificate
az keyvault certificate download \\
    --vault-name myKeyVault \\
    --name myCert \\
    --file certificate.cer

# List certificates
az keyvault certificate list \\
    --vault-name myKeyVault \\
    --output table

# Using PowerShell
# Create certificate
$policy = New-AzKeyVaultCertificatePolicy \\
    -SubjectName "CN=www.contoso.com" \\
    -ValidityInMonths 12 \\
    -IssuerName Self
Add-AzKeyVaultCertificate \\
    -VaultName myKeyVault \\
    -Name myCert \\
    -CertificatePolicy $policy`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          409,
			Title:       "Azure CLI and PowerShell",
			Description: "Learn Azure CLI and PowerShell: command-line tools for Azure management.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Azure CLI Fundamentals",
					Content: `Azure CLI is a cross-platform command-line tool for managing Azure resources.

**Azure CLI Features:**
- Command-line interface
- Supports all Azure services
- Scriptable and automatable
- Cross-platform (Windows, macOS, Linux)
- Interactive mode available

**Configuration:**
- Login: az login
- Set subscription: az account set
- Show current account: az account show
- List subscriptions: az account list

**Common Commands:**
- az login: Authenticate to Azure
- az group create: Create resource group
- az vm create: Create virtual machine
- az storage account create: Create storage account
- az webapp create: Create web app

**Best Practices:**
- Use service principals for automation
- Use Azure Cloud Shell for quick access
- Use --output table for readable output
- Use --query for filtering results
- Store credentials securely`,
					CodeExamples: `# Login to Azure
az login

# Set subscription
az account set --subscription "My Subscription"

# Show current account
az account show

# List all resource groups
az group list --output table

# Create resource group
az group create --name myRG --location eastus

# List VMs with query
az vm list --query "[].{Name:name, Location:location}" --output table

# Using Azure PowerShell
# Install module
Install-Module -Name Az -AllowClobber

# Login
Connect-AzAccount

# Set context
Set-AzContext -Subscription "My Subscription"

# Create resource group
New-AzResourceGroup -Name myRG -Location eastus

# List VMs
Get-AzVM | Select-Object Name, Location, ResourceGroupName`,
				},
				{
					Title: "ARM Templates",
					Content: `Azure Resource Manager (ARM) templates enable declarative infrastructure as code.

**Template Structure:**
- **$schema**: Template schema version
- **contentVersion**: Template version
- **parameters**: Input values
- **variables**: Reusable values
- **resources**: Resources to deploy
- **outputs**: Returned values

**Template Functions:**
- **Resource Functions**: resourceGroup(), subscription(), etc.
- **String Functions**: concat(), substring(), etc.
- **Numeric Functions**: add(), mul(), etc.
- **Array Functions**: array(), length(), etc.
- **Date Functions**: utcNow(), addDays(), etc.

**Template Deployment:**
- **Incremental Mode**: Add/update resources (default)
- **Complete Mode**: Replace entire resource group
- **Validation**: Validate before deployment
- **What-if**: Preview changes before deployment

**Linked Templates:**
- **Nested Templates**: Templates within templates
- **Linked Templates**: External template files
- **Template Specs**: Published template libraries
- **Modularity**: Reusable template components

**Best Practices:**
- Use parameters for configurable values
- Use variables for complex expressions
- Validate templates before deployment
- Use what-if for change preview
- Version control templates
- Test in non-production first
- Document template parameters
- Use template specs for reusability`,
					CodeExamples: `# Deploy ARM template
az deployment group create \\
    --resource-group myResourceGroup \\
    --template-file template.json \\
    --parameters @parameters.json

# Validate template
az deployment group validate \\
    --resource-group myResourceGroup \\
    --template-file template.json \\
    --parameters @parameters.json

# What-if preview
az deployment group what-if \\
    --resource-group myResourceGroup \\
    --template-file template.json \\
    --parameters @parameters.json

# Example template.json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "storageAccountName": {
      "type": "string",
      "metadata": {
        "description": "Storage account name"
      }
    },
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    }
  },
  "variables": {
    "storageAccountType": "Standard_LRS"
  },
  "resources": [
    {
      "type": "Microsoft.Storage/storageAccounts",
      "apiVersion": "2021-04-01",
      "name": "[parameters('storageAccountName')]",
      "location": "[parameters('location')]",
      "sku": {
        "name": "[variables('storageAccountType')]"
      },
      "kind": "StorageV2"
    }
  ],
  "outputs": {
    "storageAccountId": {
      "type": "string",
      "value": "[resourceId('Microsoft.Storage/storageAccounts', parameters('storageAccountName'))]"
    }
  }
}

# Using PowerShell
# Deploy template
New-AzResourceGroupDeployment \\
    -ResourceGroupName myResourceGroup \\
    -TemplateFile template.json \\
    -TemplateParameterFile parameters.json

# Validate template
Test-AzResourceGroupDeployment \\
    -ResourceGroupName myResourceGroup \\
    -TemplateFile template.json`,
				},
				{
					Title: "Bicep Basics",
					Content: `Bicep is a domain-specific language that compiles to ARM JSON templates.

**Bicep Advantages:**
- **Simpler Syntax**: More readable than JSON
- **Type Safety**: Compile-time type checking
- **IntelliSense**: Better IDE support
- **Modularity**: Easy to create reusable modules
- **Transpiles to ARM**: Compiles to standard ARM JSON

**Bicep Structure:**
- **Parameters**: Input values with types
- **Variables**: Reusable values
- **Resources**: Resource declarations
- **Outputs**: Returned values
- **Modules**: Reusable Bicep files

**Data Types:**
- **string**: Text values
- **int**: Integer numbers
- **bool**: Boolean values
- **array**: Collections
- **object**: Key-value pairs

**Resource Declaration:**
- **resource keyword**: Declare resource
- **Symbolic name**: Reference name in template
- **Type**: Full resource type
- **Properties**: Resource properties

**Modules:**
- **Reusability**: Create reusable components
- **Parameters**: Pass values to modules
- **Outputs**: Return values from modules
- **Scope**: Deploy to different scopes

**Best Practices:**
- Use Bicep for new deployments
- Create modules for reusable components
- Use strong typing for parameters
- Leverage IntelliSense in VS Code
- Compile to ARM JSON for validation
- Version control Bicep files
- Document module purposes
- Test modules independently`,
					CodeExamples: `# Compile Bicep to ARM JSON
az bicep build --file main.bicep

# Deploy Bicep file
az deployment group create \\
    --resource-group myResourceGroup \\
    --template-file main.bicep \\
    --parameters storageAccountName=mystorageaccount

# Example main.bicep
@description('Storage account name')
param storageAccountName string

@description('Location for resources')
param location string = resourceGroup().location

var storageAccountType = 'Standard_LRS'

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: storageAccountType
  }
  kind: 'StorageV2'
}

output storageAccountId string = storageAccount.id

# Example module (storage.bicep)
@description('Storage account name')
param storageAccountName string

@description('Location')
param location string

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
}

output storageAccountId string = storageAccount.id

# Using module in main.bicep
module storageModule 'storage.bicep' = {
  name: 'storageDeployment'
  params: {
    storageAccountName: storageAccountName
    location: location
  }
}

# Using PowerShell
# Note: Bicep deployment uses same commands as ARM templates
# Bicep files are compiled to ARM JSON automatically`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
