package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1271,
			Title:       "Azure Security and Compliance",
			Description: "Master Azure security services including Microsoft Defender for Cloud, Azure Key Vault, Managed Identities, Conditional Access, and compliance frameworks.",
			Order:       71,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Identity and Access Security",
					Content: `Azure provides comprehensive identity and access management through Microsoft Entra ID and supporting services.

**Microsoft Entra ID (Azure AD):**
` + "```" + `
Core concepts:
  Tenant:         Organization's Entra ID instance
  User:           Identity (member or guest)
  Group:          Collection of users
  Application:    Registered app identity
  Service Principal: App identity in specific tenant
  Managed Identity:  Azure-managed identity for resources

User management:
  # Create user
  az ad user create \
    --display-name "John Doe" \
    --user-principal-name john@contoso.com \
    --password 'TempP@ss123' \
    --force-change-password-next-sign-in true
  
  # List users
  az ad user list --query "[].{name:displayName, upn:userPrincipalName}" -o table
  
  # Create group
  az ad group create \
    --display-name "Developers" \
    --mail-nickname developers
  
  # Add member to group
  az ad group member add \
    --group "Developers" \
    --member-id $(az ad user show --id john@contoso.com --query id -o tsv)

Managed Identities:
  System-assigned:
    - Lifecycle tied to resource
    - One per resource
    - Auto-created and deleted
    
    az vm identity assign -n myVM -g myRG
    az webapp identity assign -n myApp -g myRG
    az functionapp identity assign -n myFunc -g myRG
  
  User-assigned:
    - Independent lifecycle
    - Can assign to multiple resources
    - Reusable across resources
    
    az identity create -n myIdentity -g myRG
    
    IDENTITY_ID=$(az identity show -n myIdentity -g myRG --query id -o tsv)
    az vm identity assign -n myVM -g myRG --identities "$IDENTITY_ID"

  Using managed identity:
    # Assign RBAC role
    PRINCIPAL_ID=$(az webapp identity show -n myApp -g myRG --query principalId -o tsv)
    
    az role assignment create \
      --assignee "$PRINCIPAL_ID" \
      --role "Storage Blob Data Reader" \
      --scope "/subscriptions/.../storageAccounts/mystorageacct"
    
    # Key Vault access
    az keyvault set-policy \
      --name mykeyvault \
      --object-id "$PRINCIPAL_ID" \
      --secret-permissions get list

Conditional Access:
  Policies control access based on:
    - User/group
    - Cloud app or action
    - Conditions: location, device, risk level
    - Grant: require MFA, compliant device, approved app
    - Session: limited app functionality
  
  Common policies:
    1. Require MFA for all users
    2. Require MFA for admin roles
    3. Block legacy authentication
    4. Require compliant device for sensitive apps
    5. Require MFA from untrusted locations
    6. Block access from risky locations

Privileged Identity Management (PIM):
  - Just-in-time privileged access
  - Time-bound role assignments
  - Require approval for activation
  - Multi-factor authentication at activation
  - Audit trail for role assignments
  
  Eligible assignments:
    User has permission to activate a role
    Must request activation (justification, MFA)
    Time-limited active period (e.g., 8 hours)

RBAC (Role-Based Access Control):
  Built-in roles:
    Owner:                  Full access + assign roles
    Contributor:            Full access, no role assignment
    Reader:                 View only
    User Access Admin:      Manage role assignments
    
    Specialized:
    Virtual Machine Contributor
    Network Contributor
    Storage Blob Data Reader
    Key Vault Secrets User
    AKS Cluster Admin
    SQL DB Contributor
    Cosmos DB Operator
  
  Custom roles:
    az role definition create --role-definition '{
      "Name": "VM Operator",
      "Description": "Start/stop VMs",
      "Actions": [
        "Microsoft.Compute/virtualMachines/start/action",
        "Microsoft.Compute/virtualMachines/deallocate/action",
        "Microsoft.Compute/virtualMachines/restart/action",
        "Microsoft.Compute/virtualMachines/read"
      ],
      "NotActions": [],
      "AssignableScopes": ["/subscriptions/<sub-id>"]
    }'
  
  # Assign role
  az role assignment create \
    --assignee john@contoso.com \
    --role "VM Operator" \
    --scope "/subscriptions/<sub-id>/resourceGroups/myRG"
` + "```" + ``,
					CodeExamples: `# Azure identity security scripts

# 1. RBAC audit
#!/bin/bash
echo "=== RBAC Audit Report ==="

SUB_ID=$(az account show --query id -o tsv)

# Owner role assignments
echo "--- Owner Role Assignments ---"
az role assignment list --role "Owner" \
    --query "[].{principal:principalName, type:principalType, scope:scope}" \
    -o table 2>/dev/null

# Contributor at subscription level
echo ""
echo "--- Subscription-Level Contributors ---"
az role assignment list --scope "/subscriptions/$SUB_ID" \
    --query "[?roleDefinitionName=='Contributor'].{
        principal:principalName, type:principalType
    }" -o table 2>/dev/null

# Custom roles
echo ""
echo "--- Custom Roles ---"
az role definition list --custom-role-only true \
    --query "[].{name:roleName, description:description}" \
    -o table 2>/dev/null

# Service principals with high privileges
echo ""
echo "--- Service Principals with Owner/Contributor ---"
az role assignment list \
    --query "[?principalType=='ServicePrincipal' && (roleDefinitionName=='Owner' || roleDefinitionName=='Contributor')].{
        principal:principalName, role:roleDefinitionName, scope:scope
    }" -o table 2>/dev/null

# 2. Managed identity audit
#!/bin/bash
echo "=== Managed Identity Audit ==="

# User-assigned identities
echo "--- User-Assigned Managed Identities ---"
az identity list \
    --query "[].{name:name, rg:resourceGroup, clientId:clientId}" \
    -o table 2>/dev/null

# System-assigned identities on VMs
echo ""
echo "--- VMs with System-Assigned Identity ---"
for rg in $(az group list --query "[].name" -o tsv 2>/dev/null); do
    az vm list -g "$rg" --query "[?identity.type!='None' && identity.type!=null].{
        name:name, type:identity.type, principalId:identity.principalId
    }" -o table 2>/dev/null
done

# App Services with identity
echo ""
echo "--- App Services with Managed Identity ---"
az webapp list --query "[?identity!=null].{
    name:name, rg:resourceGroup, type:identity.type
}" -o table 2>/dev/null

# 3. Security posture check
#!/bin/bash
echo "=== Identity Security Posture ==="

# Users without MFA (requires Graph API)
echo "--- Users Count ---"
TOTAL_USERS=$(az ad user list --query "length(@)" -o tsv 2>/dev/null)
echo "  Total users: $TOTAL_USERS"

# Guest users
GUEST_USERS=$(az ad user list --query "[?userType=='Guest'] | length(@)" -o tsv 2>/dev/null)
echo "  Guest users: $GUEST_USERS"

# Groups
TOTAL_GROUPS=$(az ad group list --query "length(@)" -o tsv 2>/dev/null)
echo "  Total groups: $TOTAL_GROUPS"

# App registrations
APP_COUNT=$(az ad app list --query "length(@)" -o tsv 2>/dev/null)
echo "  App registrations: $APP_COUNT"

# Apps with expiring credentials
echo ""
echo "--- Apps with Expiring Credentials (next 30 days) ---"
CUTOFF=$(date -u -d '+30 days' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v+30d +%Y-%m-%dT%H:%M:%SZ)
az ad app list --query "[].{
    name:displayName, appId:appId,
    secrets:passwordCredentials[?endDateTime<'$CUTOFF'].endDateTime
}" -o json 2>/dev/null | jq '.[] | select(.secrets | length > 0)'`,
				},
				{
					Title: "Azure Security Services and Key Vault",
					Content: `Azure provides defense-in-depth security services for protecting workloads, data, and infrastructure.

**Microsoft Defender for Cloud:**
` + "```" + `
Defender for Cloud (formerly Azure Security Center):
  Free tier:
    - Security posture assessment
    - Secure Score
    - Basic recommendations
  
  Enhanced (per resource):
    Defender for Servers:
      - Vulnerability assessment
      - Just-in-time VM access
      - Adaptive application controls
      - File integrity monitoring
    
    Defender for Containers:
      - Image vulnerability scanning
      - Runtime protection
      - AKS threat detection
    
    Defender for Storage:
      - Malware scanning
      - Sensitive data detection
      - Activity anomaly detection
    
    Defender for SQL:
      - Vulnerability assessment
      - Advanced Threat Protection
      - SQL injection detection
    
    Defender for Key Vault:
      - Access anomaly detection
      - Unusual operation patterns
    
    Defender for App Service:
      - Web attack detection
      - Dangling DNS detection

Enable Defender:
  # Enable for subscription
  az security pricing create --name VirtualMachines --tier Standard
  az security pricing create --name SqlServers --tier Standard
  az security pricing create --name StorageAccounts --tier Standard
  az security pricing create --name Containers --tier Standard
  az security pricing create --name KeyVaults --tier Standard
  az security pricing create --name AppServices --tier Standard

Secure Score:
  - 0-100% score
  - Based on recommendation compliance
  - Weighted by impact and effort
  - Categories: identity, networking, data, compute, apps
  
  az security secure-score-controls list \
    --query "[].{name:displayName, current:currentScore, max:maxScore}" \
    -o table

Recommendations:
  az security assessment list \
    --query "[?status.code=='Unhealthy'].{
        name:displayName,
        severity:metadata.severity,
        resource:resourceDetails.id
    }" -o table | head -20

Alerts:
  az security alert list \
    --query "[].{name:alertDisplayName, severity:severity, status:status}" \
    -o table | head -20
` + "```" + `

**Azure Key Vault:**
` + "```" + `
Centralized secrets management.

Types of objects:
  Secrets:       Connection strings, API keys, passwords
  Keys:          Cryptographic keys (RSA, EC)
  Certificates:  X.509 certificates with auto-renewal

Create Key Vault:
  az keyvault create \
    --name mykeyvault -g myRG \
    --location eastus \
    --sku premium \
    --enable-rbac-authorization true \
    --enable-purge-protection true \
    --retention-days 90

Access models:
  Vault access policy (legacy):
    az keyvault set-policy \
      --name mykeyvault \
      --upn john@contoso.com \
      --secret-permissions get list set delete \
      --key-permissions get list create delete \
      --certificate-permissions get list create delete
  
  RBAC (recommended):
    az role assignment create \
      --assignee john@contoso.com \
      --role "Key Vault Secrets Officer" \
      --scope $(az keyvault show -n mykeyvault --query id -o tsv)
    
    Roles:
      Key Vault Administrator:          Full access
      Key Vault Secrets Officer:         Secrets CRUD
      Key Vault Secrets User:            Secrets read
      Key Vault Crypto Officer:          Keys CRUD
      Key Vault Crypto User:             Keys use (encrypt/decrypt)
      Key Vault Certificates Officer:    Certs CRUD
      Key Vault Reader:                  Read metadata only

Secrets:
  # Create secret
  az keyvault secret set \
    --vault-name mykeyvault --name dbPassword \
    --value 'MyP@ssw0rd123!'
  
  # Get secret
  az keyvault secret show \
    --vault-name mykeyvault --name dbPassword \
    --query value -o tsv
  
  # List secrets
  az keyvault secret list --vault-name mykeyvault \
    --query "[].{name:name, enabled:attributes.enabled}" -o table
  
  # Soft delete and recovery
  az keyvault secret delete --vault-name mykeyvault --name dbPassword
  az keyvault secret recover --vault-name mykeyvault --name dbPassword
  
  # Purge (permanent delete, requires purge protection disabled)
  az keyvault secret purge --vault-name mykeyvault --name dbPassword

Keys:
  # Create RSA key
  az keyvault key create \
    --vault-name mykeyvault --name mykey \
    --kty RSA --size 2048
  
  # Create EC key
  az keyvault key create \
    --vault-name mykeyvault --name myeckey \
    --kty EC --curve P-256
  
  # Import key
  az keyvault key import \
    --vault-name mykeyvault --name importedkey \
    --pem-file key.pem

Certificates:
  # Create self-signed certificate
  az keyvault certificate create \
    --vault-name mykeyvault --name mycert \
    --policy "$(az keyvault certificate get-default-policy)"
  
  # Import certificate
  az keyvault certificate import \
    --vault-name mykeyvault --name mycert \
    --file cert.pfx --password 'certP@ss'

Private endpoint for Key Vault:
  az network private-endpoint create \
    --name kv-pe -g myRG \
    --vnet-name myVNet --subnet private-endpoints \
    --private-connection-resource-id $(az keyvault show -n mykeyvault --query id -o tsv) \
    --group-id vault \
    --connection-name kv-connection
  
  # Disable public access
  az keyvault update \
    --name mykeyvault \
    --public-network-access Disabled

Key Vault references:
  App Service/Functions can reference Key Vault secrets directly:
  
  Setting value: @Microsoft.KeyVault(VaultName=mykeyvault;SecretName=dbPassword)
  
  az webapp config appsettings set -n myApp -g myRG \
    --settings "DB_PASSWORD=@Microsoft.KeyVault(VaultName=mykeyvault;SecretName=dbPassword)"
  
  Requires managed identity with Key Vault access.

Rotation:
  Event Grid integration for auto-rotation:
  1. Secret nearing expiry → Event Grid notification
  2. Azure Function triggered
  3. Function rotates credential in external service
  4. Function updates Key Vault secret
` + "```" + `

**Azure Network Security:**
` + "```" + `
Microsoft Sentinel (SIEM/SOAR):
  - Cloud-native SIEM
  - AI-powered threat detection
  - Automated response (playbooks)
  - Data connectors (Azure, AWS, M365, firewalls)
  - KQL-based hunting queries
  - Workbooks for visualization

Azure Bastion:
  - Secure RDP/SSH without public IP
  - Browser-based access
  - No agent required
  - Protected against port scanning
  
  az network bastion create \
    --name myBastion -g myRG \
    --vnet-name myVNet \
    --public-ip-address bastion-pip \
    --sku Standard

Just-in-Time VM Access:
  - Opens NSG ports only when needed
  - Time-limited (max 24 hours)
  - Requires approval
  - Audit trail

Azure Confidential Computing:
  - Data encrypted at rest, in transit, AND in use
  - Hardware-based Trusted Execution Environments (TEE)
  - Intel SGX and AMD SEV-SNP
  - Confidential VMs, containers, and services

Encryption:
  At rest:
    Storage Service Encryption (SSE):    All storage (auto)
    Transparent Data Encryption (TDE):   SQL (auto)
    Azure Disk Encryption:               VM OS/data disks
    Customer-managed keys (CMK):         Via Key Vault
  
  In transit:
    TLS 1.2+ required
    HTTPS enforcement
    VPN encryption (IPsec/IKE)
    ExpressRoute encryption (MACsec)
` + "```" + ``,
					CodeExamples: `# Azure security scripts

# 1. Security posture dashboard
#!/bin/bash
echo "=== Azure Security Dashboard ==="

# Secure Score
echo "--- Secure Score ---"
az security secure-score list \
    --query "[0].{current:currentScore, max:maxScore, percentage:percentage}" \
    -o json 2>/dev/null | jq .

# Secure Score by control
echo ""
echo "--- Score by Control ---"
az security secure-score-controls list \
    --query "[?currentScore<maxScore].{
        control:displayName,
        current:currentScore,
        max:maxScore,
        unhealthy:unhealthyResourceCount
    }" -o table 2>/dev/null | head -15

# Active alerts
echo ""
echo "--- Security Alerts ---"
az security alert list \
    --query "[?status!='Dismissed'].{
        alert:alertDisplayName,
        severity:severity,
        status:status,
        time:timeGeneratedUtc
    }" -o table 2>/dev/null | head -15

# Defender plans
echo ""
echo "--- Defender Plans ---"
az security pricing list \
    --query "[].{plan:name, tier:pricingTier}" \
    -o table 2>/dev/null

# 2. Key Vault security audit
#!/bin/bash
echo "=== Key Vault Security Audit ==="

for vault in $(az keyvault list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az keyvault list --query "[?name=='$vault'].resourceGroup" -o tsv | head -1)
    echo "Vault: $vault ($RG)"
    
    # Check configuration
    az keyvault show -n "$vault" -g "$RG" \
        --query "{
            rbacAuth:properties.enableRbacAuthorization,
            softDelete:properties.enableSoftDelete,
            purgeProtection:properties.enablePurgeProtection,
            publicAccess:properties.publicNetworkAccess,
            sku:properties.sku.name
        }" -o json 2>/dev/null | jq .
    
    # Expiring secrets
    echo "  Secrets:"
    CUTOFF=$(date -u -d '+30 days' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v+30d +%Y-%m-%dT%H:%M:%SZ)
    az keyvault secret list --vault-name "$vault" \
        --query "[].{name:name, enabled:attributes.enabled, expires:attributes.expires}" \
        -o table 2>/dev/null | head -10
    
    # Expiring certificates
    echo "  Certificates:"
    az keyvault certificate list --vault-name "$vault" \
        --query "[].{name:name, expires:attributes.expires}" \
        -o table 2>/dev/null | head -10
    
    # Private endpoint
    PE=$(az network private-endpoint list \
        --query "[?contains(privateLinkServiceConnections[0].privateLinkServiceId, '$vault')].name" \
        -o tsv 2>/dev/null | head -1)
    if [ -n "$PE" ]; then
        echo "  Private Endpoint: $PE"
    else
        echo "  [WARNING] No private endpoint configured"
    fi
    echo ""
done

# 3. Compliance overview
#!/bin/bash
echo "=== Compliance Overview ==="

# Regulatory compliance
echo "--- Regulatory Compliance Standards ---"
az security regulatory-compliance-standards list \
    --query "[].{standard:name, state:state, passed:passedControls, failed:failedControls}" \
    -o table 2>/dev/null

# Policy compliance
echo ""
echo "--- Policy Compliance Summary ---"
az policy state summarize \
    --query "{
        totalResources:results.totalResources,
        nonCompliant:results.nonCompliantResources,
        totalPolicies:results.policyAssignments | length(@)
    }" -o json 2>/dev/null | jq .

# Top non-compliant policies
echo ""
echo "--- Top Non-Compliant Policies ---"
az policy state summarize \
    --query "policyAssignments[?results.nonCompliantResources>0] | sort_by(@, &results.nonCompliantResources) | reverse(@)[:10].{
        policy:policyAssignmentId | split('/') | [-1],
        nonCompliant:results.nonCompliantResources,
        total:results.totalResources
    }" -o table 2>/dev/null`,
				},
			},
		},
	})
}
