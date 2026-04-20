package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1245,
			Title:       "Azure Identity and Access Management",
			Description: "Master Azure Active Directory (Entra ID), RBAC, managed identities, conditional access, and security best practices for Azure resources.",
			Order:       45,
			Lessons: []problems.Lesson{
				{
					Title: "Azure AD and RBAC",
					Content: `Azure Active Directory (now Microsoft Entra ID) is the identity platform for Azure. Role-Based Access Control (RBAC) governs who can do what on Azure resources.

**Azure AD (Entra ID) Fundamentals:**
` + "```" + `
Azure AD concepts:
  Tenant:          An organization's instance of Azure AD
  Directory:       Container for users, groups, apps
  Subscription:    Billing container, trusts one Azure AD tenant
  Management Group: Hierarchy above subscriptions for governance

Hierarchy:
  Azure AD Tenant
  └── Management Group (Root)
      ├── Management Group (Production)
      │   ├── Subscription (Prod-App1)
      │   │   ├── Resource Group (app1-rg)
      │   │   │   ├── VM
      │   │   │   ├── Storage Account
      │   │   │   └── Database
      │   │   └── Resource Group (app1-network-rg)
      │   └── Subscription (Prod-App2)
      └── Management Group (Non-Production)
          ├── Subscription (Dev)
          └── Subscription (Staging)

User types:
  Member:   Full users created in directory
  Guest:    External users (B2B collaboration)
  
Group types:
  Security:     Access management
  Microsoft 365: Collaboration (mailbox, SharePoint)
  
  Membership:
    Assigned:   Manually added
    Dynamic:    Rule-based (user.department -eq "Engineering")

Service Principal:
  - Identity for applications/services
  - Used for automation, CI/CD
  
  # Create service principal
  az ad sp create-for-rbac \
    --name myapp-sp \
    --role Contributor \
    --scopes /subscriptions/<sub-id>/resourceGroups/myRG
` + "```" + `

**RBAC (Role-Based Access Control):**
` + "```" + `
RBAC components:
  Security Principal:  Who (user, group, service principal, managed identity)
  Role Definition:     What they can do (set of permissions)
  Scope:               Where (management group, subscription, resource group, resource)

Built-in roles:
  Owner:                Full access + assign roles
  Contributor:          Full access, no role assignment
  Reader:               View only
  User Access Admin:    Manage user access to resources
  
  Resource-specific:
  Virtual Machine Contributor
  Storage Blob Data Contributor
  Network Contributor
  Key Vault Administrator
  AKS Cluster Admin Role
  SQL DB Contributor

Assign role:
  # At resource group scope
  az role assignment create \
    --assignee user@example.com \
    --role "Contributor" \
    --scope /subscriptions/<sub-id>/resourceGroups/myRG
  
  # At subscription scope
  az role assignment create \
    --assignee-object-id <sp-object-id> \
    --assignee-principal-type ServicePrincipal \
    --role "Reader" \
    --scope /subscriptions/<sub-id>
  
  # List assignments
  az role assignment list --resource-group myRG --output table
  
  # List assignments for a user
  az role assignment list --assignee user@example.com --all --output table

Custom roles:
  {
    "Name": "VM Operator",
    "Description": "Can start, restart, and stop VMs",
    "Actions": [
      "Microsoft.Compute/virtualMachines/start/action",
      "Microsoft.Compute/virtualMachines/restart/action",
      "Microsoft.Compute/virtualMachines/deallocate/action",
      "Microsoft.Compute/virtualMachines/read",
      "Microsoft.Compute/virtualMachines/instanceView/read"
    ],
    "NotActions": [],
    "AssignableScopes": [
      "/subscriptions/<sub-id>"
    ]
  }
  
  az role definition create --role-definition @custom-role.json

RBAC best practices:
  1. Least privilege: assign minimum needed permissions
  2. Use groups: assign roles to groups, add users to groups
  3. Use built-in roles when possible
  4. Scope as narrowly as possible
  5. Use Privileged Identity Management (PIM) for just-in-time access
  6. Regular access reviews
  7. Avoid Owner role; use Contributor + User Access Admin separately
` + "```" + `

**Managed Identities:**
` + "```" + `
Managed identities eliminate the need for credentials in code.

Types:
  System-assigned:
    - Tied to a specific resource
    - Created and deleted with the resource
    - Cannot be shared
    - Use for single-resource scenarios
  
  User-assigned:
    - Standalone Azure resource
    - Can be assigned to multiple resources
    - Independent lifecycle
    - Use for shared identity scenarios

Enable system-assigned:
  az vm identity assign \
    --resource-group myRG \
    --name myVM
  
  az webapp identity assign \
    --resource-group myRG \
    --name myWebApp

Create user-assigned:
  az identity create \
    --resource-group myRG \
    --name myIdentity
  
  az vm identity assign \
    --resource-group myRG \
    --name myVM \
    --identities myIdentity

Grant access:
  # Get managed identity principal ID
  PRINCIPAL_ID=$(az vm identity show \
    --resource-group myRG --name myVM \
    --query "principalId" -o tsv)
  
  # Assign role
  az role assignment create \
    --assignee-object-id "$PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --role "Storage Blob Data Contributor" \
    --scope /subscriptions/<sub-id>/resourceGroups/myRG/providers/Microsoft.Storage/storageAccounts/mystorageacct

Usage in code (no credentials needed):
  # Python example
  from azure.identity import DefaultAzureCredential
  from azure.storage.blob import BlobServiceClient
  
  credential = DefaultAzureCredential()
  blob_client = BlobServiceClient(
      "https://mystorageacct.blob.core.windows.net",
      credential=credential
  )
  
  # Go example
  cred, _ := azidentity.NewDefaultAzureCredential(nil)
  client, _ := azblob.NewClient(
      "https://mystorageacct.blob.core.windows.net",
      cred, nil)
` + "```" + ``,
					CodeExamples: `# Azure IAM management scripts

# 1. RBAC audit script
#!/bin/bash
echo "=== Azure RBAC Audit ==="

SUB_ID=$(az account show --query "id" -o tsv)

echo "--- Owner Role Assignments ---"
az role assignment list --role "Owner" --all --query "[].{
    principal:principalName, type:principalType,
    scope:scope
}" -o table

echo ""
echo "--- Service Principals with Contributor+ ---"
az role assignment list --all \
    --query "[?principalType=='ServicePrincipal' && (roleDefinitionName=='Owner' || roleDefinitionName=='Contributor')].{
        principal:principalName, role:roleDefinitionName, scope:scope
    }" -o table

echo ""
echo "--- Custom Roles ---"
az role definition list --custom-role-only --query "[].{
    name:roleName, description:description,
    scopes:assignableScopes[0]
}" -o table

echo ""
echo "--- Subscription-Level Assignments ---"
az role assignment list --scope "/subscriptions/$SUB_ID" --query "[].{
    principal:principalName, type:principalType,
    role:roleDefinitionName
}" -o table

# 2. Managed identity inventory
#!/bin/bash
echo "=== Managed Identity Inventory ==="

echo "--- User-Assigned Identities ---"
az identity list --query "[].{
    name:name, rg:resourceGroup, principalId:principalId
}" -o table

echo ""
echo "--- VMs with Managed Identity ---"
for rg in $(az group list --query "[].name" -o tsv); do
    az vm list -g "$rg" --query "[?identity!=null].{
        name:name, type:identity.type,
        principalId:identity.principalId
    }" -o table 2>/dev/null
done

echo ""
echo "--- Web Apps with Managed Identity ---"
for rg in $(az group list --query "[].name" -o tsv); do
    az webapp list -g "$rg" --query "[?identity!=null].{
        name:name, type:identity.type,
        principalId:identity.principalId
    }" -o table 2>/dev/null
done

# 3. Service principal credential checker
#!/bin/bash
echo "=== Service Principal Credential Audit ==="

for sp in $(az ad sp list --all --query "[?servicePrincipalType=='Application'].appId" -o tsv 2>/dev/null | head -50); do
    APP_NAME=$(az ad sp show --id "$sp" --query "displayName" -o tsv 2>/dev/null)
    
    # Check credentials
    CREDS=$(az ad app credential list --id "$sp" --query "[].{
        keyId:keyId, endDate:endDateTime
    }" -o json 2>/dev/null)
    
    if [ "$(echo "$CREDS" | jq length)" -gt 0 ]; then
        echo ""
        echo "SP: $APP_NAME ($sp)"
        echo "$CREDS" | jq -r '.[] | "  Key: \(.keyId) Expires: \(.endDate)"'
        
        # Check for expiring credentials
        echo "$CREDS" | jq -r '.[].endDateTime' | while read -r expiry; do
            if [ -n "$expiry" ]; then
                EXPIRY_EPOCH=$(date -d "$expiry" +%s 2>/dev/null || echo 0)
                NOW_EPOCH=$(date +%s)
                DAYS_LEFT=$(( (EXPIRY_EPOCH - NOW_EPOCH) / 86400 ))
                
                if [ "$DAYS_LEFT" -lt 30 ] && [ "$DAYS_LEFT" -gt 0 ]; then
                    echo "  WARNING: Credential expires in $DAYS_LEFT days"
                elif [ "$DAYS_LEFT" -le 0 ]; then
                    echo "  CRITICAL: Credential has expired!"
                fi
            fi
        done
    fi
done`,
				},
				{
					Title: "Azure Key Vault and Security Best Practices",
					Content: `Azure Key Vault centralizes secrets management, key management, and certificate management. It's essential for secure application deployment.

**Azure Key Vault:**
` + "```" + `
Key Vault manages:
  Secrets:       Connection strings, passwords, API keys
  Keys:          Encryption keys (RSA, EC)
  Certificates:  SSL/TLS certificates

Tiers:
  Standard:  Software-protected keys
  Premium:   HSM (Hardware Security Module) protected keys

Create Key Vault:
  az keyvault create \
    --resource-group myRG \
    --name myKeyVault \
    --location eastus \
    --sku Standard \
    --enable-rbac-authorization true \
    --enable-soft-delete true \
    --soft-delete-retention-days 90 \
    --enable-purge-protection true

Secrets:
  # Set secret
  az keyvault secret set \
    --vault-name myKeyVault \
    --name DatabasePassword \
    --value "MySecret123!"
  
  # Get secret
  az keyvault secret show \
    --vault-name myKeyVault \
    --name DatabasePassword \
    --query "value" -o tsv
  
  # List secrets
  az keyvault secret list --vault-name myKeyVault -o table
  
  # Set secret with expiry
  az keyvault secret set \
    --vault-name myKeyVault \
    --name ApiKey \
    --value "key123" \
    --expires "2025-12-31T23:59:59Z"
  
  # Versions
  az keyvault secret list-versions \
    --vault-name myKeyVault \
    --name DatabasePassword

Keys:
  # Create key
  az keyvault key create \
    --vault-name myKeyVault \
    --name myEncryptionKey \
    --kty RSA --size 2048
  
  # Import key
  az keyvault key import \
    --vault-name myKeyVault \
    --name imported-key \
    --pem-file key.pem

Certificates:
  # Create self-signed
  az keyvault certificate create \
    --vault-name myKeyVault \
    --name myCert \
    --policy "$(az keyvault certificate get-default-policy)"
  
  # Import certificate
  az keyvault certificate import \
    --vault-name myKeyVault \
    --name imported-cert \
    --file cert.pfx \
    --password "certpass"

Access with RBAC:
  # Grant secret reader
  az role assignment create \
    --assignee <managed-identity-id> \
    --role "Key Vault Secrets User" \
    --scope /subscriptions/.../vaults/myKeyVault
  
  RBAC roles:
    Key Vault Administrator:       Full access
    Key Vault Secrets Officer:     Manage secrets
    Key Vault Secrets User:        Read secrets
    Key Vault Crypto Officer:      Manage keys
    Key Vault Crypto User:         Use keys
    Key Vault Certificates Officer: Manage certs

Network security:
  # Private endpoint
  az network private-endpoint create \
    --resource-group myRG --name kv-pe \
    --vnet-name myVNet --subnet app-subnet \
    --private-connection-resource-id /subscriptions/.../vaults/myKeyVault \
    --group-ids vault --connection-name kv-connection
  
  # Firewall rules
  az keyvault network-rule add \
    --name myKeyVault \
    --vnet-name myVNet --subnet app-subnet
  
  az keyvault update \
    --name myKeyVault \
    --default-action Deny
` + "```" + `

**Security Best Practices:**
` + "```" + `
1. Identity:
   - Use Azure AD for all authentication
   - Enable MFA for all users
   - Use Conditional Access policies
   - Implement Privileged Identity Management (PIM)
   - Regular access reviews
   - Disable legacy authentication protocols

2. Network:
   - Use private endpoints for PaaS services
   - Implement hub-spoke topology with Azure Firewall
   - Enable NSG flow logs
   - Use Azure DDoS Protection Standard
   - No public IPs unless necessary

3. Data:
   - Enable encryption at rest (default)
   - Use customer-managed keys for sensitive data
   - Enable Azure Defender for Storage
   - Implement data classification
   - Use Azure Purview for governance

4. Compute:
   - Use managed identities (no passwords in code)
   - Enable Azure Defender for Servers
   - Keep OS and applications updated
   - Use Azure Bastion instead of public SSH/RDP
   - Implement Just-In-Time VM access

5. Monitoring:
   - Enable Azure Monitor and Log Analytics
   - Configure Azure Sentinel (SIEM)
   - Set up alerts for suspicious activity
   - Enable diagnostic logging on all resources
   - Implement Azure Policy for compliance

6. Azure Policy:
   - Enforce tagging standards
   - Restrict allowed VM sizes
   - Require encryption
   - Deny public access to storage
   - Enforce naming conventions
   
   # Assign policy
   az policy assignment create \
     --name "require-tag" \
     --policy "/providers/Microsoft.Authorization/policyDefinitions/..." \
     --scope "/subscriptions/<sub-id>" \
     --params '{"tagName": {"value": "Environment"}}'
` + "```" + ``,
					CodeExamples: `# Azure security management

# 1. Key Vault secret rotation
#!/bin/bash
VAULT="${1:?Usage: $0 <vault-name>}"
MAX_AGE_DAYS=90

echo "=== Key Vault Secret Audit: $VAULT ==="

az keyvault secret list --vault-name "$VAULT" --query "[].{
    name:name, enabled:attributes.enabled, expires:attributes.expires
}" -o json | jq -r '.[] | select(.enabled==true)' | jq -s '.' | \
jq -r '.[] | "\(.name)|\(.expires // "never")"' | while IFS='|' read -r name expires; do
    
    # Check age of current version
    CREATED=$(az keyvault secret show --vault-name "$VAULT" --name "$name" \
        --query "attributes.created" -o tsv 2>/dev/null)
    
    if [ -n "$CREATED" ]; then
        CREATED_EPOCH=$(date -d "$CREATED" +%s 2>/dev/null || echo 0)
        NOW_EPOCH=$(date +%s)
        AGE_DAYS=$(( (NOW_EPOCH - CREATED_EPOCH) / 86400 ))
        
        STATUS="OK"
        if [ "$AGE_DAYS" -gt "$MAX_AGE_DAYS" ]; then
            STATUS="ROTATE (${AGE_DAYS} days old)"
        fi
        
        printf "  %-30s  Age: %3d days  Expires: %-12s  [%s]\n" \
            "$name" "$AGE_DAYS" "$expires" "$STATUS"
    fi
done

# 2. Azure security posture check
#!/bin/bash
echo "=== Azure Security Posture Check ==="
SUB_ID=$(az account show --query "id" -o tsv)

# Check storage accounts
echo ""
echo "--- Storage Security ---"
az storage account list --query "[].{
    name:name, https:enableHttpsTrafficOnly, tls:minimumTlsVersion,
    publicAccess:allowBlobPublicAccess, networkDefault:networkRuleSet.defaultAction
}" -o table

# Check for public IPs
echo ""
echo "--- Public IP Addresses ---"
az network public-ip list --query "[].{
    name:name, rg:resourceGroup, ip:ipAddress, associated:ipConfiguration.id
}" -o table

# Check NSGs without rules
echo ""
echo "--- NSGs Check ---"
for rg in $(az group list --query "[].name" -o tsv); do
    for nsg in $(az network nsg list -g "$rg" --query "[].name" -o tsv 2>/dev/null); do
        CUSTOM_RULES=$(az network nsg rule list -g "$rg" --nsg-name "$nsg" \
            --query "length(@)" -o tsv 2>/dev/null)
        if [ "$CUSTOM_RULES" = "0" ]; then
            echo "  WARNING: NSG $nsg ($rg) has no custom rules"
        fi
    done
done

# Check Key Vaults
echo ""
echo "--- Key Vault Security ---"
az keyvault list --query "[].{
    name:name, softDelete:properties.enableSoftDelete,
    purgeProtection:properties.enablePurgeProtection,
    rbac:properties.enableRbacAuthorization
}" -o table

# 3. Compliance report generator
#!/bin/bash
echo "=== Azure Compliance Report ==="
echo "Date: $(date)"
echo "Subscription: $(az account show --query 'name' -o tsv)"
echo ""

PASS=0
FAIL=0

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "true" ] || [ "$result" = "PASS" ]; then
        echo "  [PASS] $desc"
        ((PASS++))
    else
        echo "  [FAIL] $desc"
        ((FAIL++))
    fi
}

# Check: All storage accounts require HTTPS
ALL_HTTPS=$(az storage account list --query "[?!enableHttpsTrafficOnly].name" -o tsv)
check "All storage accounts require HTTPS" "$([ -z "$ALL_HTTPS" ] && echo true || echo false)"

# Check: All storage accounts use TLS 1.2+
OLD_TLS=$(az storage account list --query "[?minimumTlsVersion!='TLS1_2'].name" -o tsv)
check "All storage accounts use TLS 1.2" "$([ -z "$OLD_TLS" ] && echo true || echo false)"

# Check: No storage accounts with public blob access
PUBLIC_BLOB=$(az storage account list --query "[?allowBlobPublicAccess==true].name" -o tsv)
check "No public blob access on storage" "$([ -z "$PUBLIC_BLOB" ] && echo true || echo false)"

echo ""
echo "Results: $PASS passed, $FAIL failed"`,
				},
			},
		},
	})
}
