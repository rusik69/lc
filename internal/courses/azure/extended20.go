package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1273,
			Title:       "Azure Storage Advanced and Migration",
			Description: "Advanced Azure Storage features including lifecycle management, object replication, data migration strategies, and Azure Migrate for cloud adoption.",
			Order:       73,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Storage Advanced Features",
					Content: `Azure Storage provides advanced features for data management, lifecycle automation, and high availability.

**Storage Lifecycle Management:**
` + "```" + `
Lifecycle policies:
  Automatically transition or delete blobs based on rules.
  
  az storage account management-policy create \
    --account-name mystorageacct -g myRG \
    --policy '{
      "rules": [
        {
          "enabled": true,
          "name": "move-to-cool",
          "type": "Lifecycle",
          "definition": {
            "filters": {
              "blobTypes": ["blockBlob"],
              "prefixMatch": ["logs/", "backups/"]
            },
            "actions": {
              "baseBlob": {
                "tierToCool": {"daysAfterModificationGreaterThan": 30},
                "tierToArchive": {"daysAfterModificationGreaterThan": 90},
                "delete": {"daysAfterModificationGreaterThan": 365}
              },
              "snapshot": {
                "delete": {"daysAfterCreationGreaterThan": 90}
              }
            }
          }
        }
      ]
    }'
  
  Tier pricing (per GB/month, East US approximate):
    Hot:       ~$0.018
    Cool:      ~$0.010  (30-day min retention)
    Cold:      ~$0.0036 (90-day min retention)
    Archive:   ~$0.002  (180-day min retention)
  
  Access cost (per 10K operations):
    Hot:       $0.004 read, $0.05 write
    Cool:      $0.01 read, $0.10 write
    Archive:   $5.00 read, $0.10 write

Object replication:
  Asynchronous replication between storage accounts.
  
  Rules:
    - Source and destination accounts
    - Container-level filtering
    - Prefix filtering
    - Copy metadata and blob tags
  
  az storage account or-policy create \
    --account-name src-storage -g myRG \
    --destination-account dst-storage \
    --destination-account-resource-group myRG2 \
    --rules '[{
      "sourceContainer": "source-container",
      "destinationContainer": "dest-container",
      "filters": {
        "prefixMatch": ["important/"],
        "minCreationTime": "2024-01-01T00:00:00Z"
      }
    }]'

Immutable storage:
  WORM (Write Once, Read Many) compliance.
  
  Time-based retention:
    az storage container immutability-policy create \
      --account-name mystorageacct -g myRG \
      --container-name compliance-data \
      --period 365
  
  Legal hold:
    az storage container legal-hold set \
      --account-name mystorageacct -g myRG \
      --container-name legal-docs \
      --tags "case123" "investigation456"
  
  Version-level immutability:
    Individual blob version immutability policies.

Soft delete:
  # Blob soft delete
  az storage blob service-properties delete-policy update \
    --account-name mystorageacct -g myRG \
    --enable true --days-retained 14
  
  # Container soft delete
  az storage account blob-service-properties update \
    --account-name mystorageacct -g myRG \
    --enable-container-delete-retention true \
    --container-delete-retention-days 7

Versioning:
  az storage account blob-service-properties update \
    --account-name mystorageacct -g myRG \
    --enable-versioning true
  
  Benefits:
    - Automatic version on every write
    - Restore previous versions
    - No performance impact
    - Works with lifecycle management

Change feed:
  az storage account blob-service-properties update \
    --account-name mystorageacct -g myRG \
    --enable-change-feed true \
    --change-feed-retention-days 7
  
  Use cases:
    - Audit trail for blob operations
    - Trigger downstream processing
    - Sync with other systems
    - Compliance logging

Static website hosting:
  az storage blob service-properties update \
    --account-name mystorageacct -g myRG \
    --static-website --index-document index.html \
    --404-document error.html
  
  # Upload files
  az storage blob upload-batch \
    -d '$web' --account-name mystorageacct \
    -s ./dist --overwrite
  
  # Custom domain + CDN
  az cdn endpoint create \
    --name mycdn -g myRG --profile-name mycdnprofile \
    --origin mystorageacct.z13.web.core.windows.net \
    --origin-host-header mystorageacct.z13.web.core.windows.net

Azure Files:
  # Create file share
  az storage share create \
    --name myshare --account-name mystorageacct \
    --quota 100
  
  # SMB mount (Windows)
  net use Z: \\\\mystorageacct.file.core.windows.net\\myshare /u:mystorageacct <key>
  
  # SMB mount (Linux)
  mount -t cifs //mystorageacct.file.core.windows.net/myshare /mnt/myshare \
    -o vers=3.0,username=mystorageacct,password=<key>,dir_mode=0777,file_mode=0777
  
  # Azure File Sync
  Sync on-premises file servers with Azure Files
  Cloud tiering: keep hot files local, cold in Azure
  Multi-site sync: share data across offices
` + "```" + ``,
					CodeExamples: `# Azure Storage advanced scripts

# 1. Storage lifecycle and cost optimization
#!/bin/bash
echo "=== Storage Optimization Report ==="

for acct in $(az storage account list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az storage account list --query "[?name=='$acct'].resourceGroup" -o tsv | head -1)
    echo "Account: $acct ($RG)"
    
    # Account info
    SKU=$(az storage account show -n "$acct" -g "$RG" --query "sku.name" -o tsv 2>/dev/null)
    KIND=$(az storage account show -n "$acct" -g "$RG" --query "kind" -o tsv 2>/dev/null)
    ACCESS=$(az storage account show -n "$acct" -g "$RG" --query "accessTier" -o tsv 2>/dev/null)
    echo "  SKU: $SKU, Kind: $KIND, Access Tier: $ACCESS"
    
    # Lifecycle policy
    POLICY=$(az storage account management-policy show \
        --account-name "$acct" -g "$RG" \
        --query "policy.rules | length(@)" -o tsv 2>/dev/null)
    
    if [ -n "$POLICY" ] && [ "$POLICY" != "0" ]; then
        echo "  Lifecycle rules: $POLICY"
    else
        echo "  [INFO] No lifecycle policy"
    fi
    
    # Versioning status
    VERSIONING=$(az storage account blob-service-properties show \
        --account-name "$acct" -g "$RG" \
        --query "isVersioningEnabled" -o tsv 2>/dev/null)
    echo "  Versioning: $VERSIONING"
    
    # Soft delete
    SOFT_DELETE=$(az storage blob service-properties delete-policy show \
        --account-name "$acct" -g "$RG" \
        --query "enabled" -o tsv 2>/dev/null)
    echo "  Soft delete: $SOFT_DELETE"
    echo ""
done

# 2. Storage security audit
#!/bin/bash
echo "=== Storage Security Audit ==="

for acct in $(az storage account list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az storage account list --query "[?name=='$acct'].resourceGroup" -o tsv | head -1)
    echo "Account: $acct"
    
    # Security settings
    az storage account show -n "$acct" -g "$RG" \
        --query "{
            httpsOnly:enableHttpsTrafficOnly,
            minTlsVersion:minimumTlsVersion,
            publicBlobAccess:allowBlobPublicAccess,
            sharedKeyAccess:allowSharedKeyAccess,
            publicNetworkAccess:publicNetworkAccess,
            encryption:encryption.keySource
        }" -o json 2>/dev/null | jq .
    
    # Network rules
    DEFAULT_ACTION=$(az storage account show -n "$acct" -g "$RG" \
        --query "networkRuleSet.defaultAction" -o tsv 2>/dev/null)
    
    if [ "$DEFAULT_ACTION" = "Allow" ]; then
        echo "  [WARNING] Public network access allowed"
    else
        echo "  [OK] Network rules restrict access"
    fi
    
    # Private endpoints
    PE_COUNT=$(az network private-endpoint list \
        --query "[?contains(privateLinkServiceConnections[0].privateLinkServiceId, '$acct')] | length(@)" \
        -o tsv 2>/dev/null)
    echo "  Private endpoints: $PE_COUNT"
    echo ""
done

# 3. Blob inventory
#!/bin/bash
echo "=== Blob Inventory ==="

ACCOUNT="${1:-mystorageacct}"
CONTAINER="${2:-data}"

echo "Account: $ACCOUNT, Container: $CONTAINER"

# Count blobs by tier
echo "--- Blobs by Access Tier ---"
for tier in Hot Cool Cold Archive; do
    COUNT=$(az storage blob list \
        --account-name "$ACCOUNT" -c "$CONTAINER" \
        --query "[?properties.blobTier=='$tier'] | length(@)" \
        -o tsv --auth-mode login 2>/dev/null)
    [ -n "$COUNT" ] && [ "$COUNT" != "0" ] && echo "  $tier: $COUNT blobs"
done

# Total size
echo ""
echo "--- Container Size ---"
az storage blob list \
    --account-name "$ACCOUNT" -c "$CONTAINER" \
    --query "[].properties.contentLength" -o tsv \
    --auth-mode login 2>/dev/null | \
    awk '{sum+=$1} END {printf "  Total: %.2f GB\n", sum/1024/1024/1024}'`,
				},
				{
					Title: "Azure Migration and Cloud Adoption",
					Content: `Azure provides tools and frameworks for migrating workloads from on-premises or other clouds to Azure.

**Azure Migrate:**
` + "```" + `
Azure Migrate hub:
  - Discover and assess on-premises workloads
  - Right-size Azure recommendations
  - Cost estimation
  - Dependency mapping
  - Migration tools

Discovery and assessment:
  az extension add -n import-export
  
  # Create Azure Migrate project
  az migrate project create \
    --name myMigrateProject -g myRG \
    --location eastus
  
  Assessment types:
    Azure VM assessment:
      - VM sizing recommendations
      - Cost estimates
      - Readiness analysis
    
    Azure SQL assessment:
      - SQL Server to Azure SQL
      - Migration target (SQL DB, MI, VM)
      - Feature compatibility
    
    Azure App Service assessment:
      - Web apps to App Service
      - Compatibility analysis
    
    AVS (Azure VMware Solution):
      - VMware workloads to AVS

Migration tools:
  Server Migration:
    Agentless (VMware):
      - No agent installation
      - Replication via vCenter
      - Minimal impact
    
    Agent-based:
      - Install mobility agent
      - Supports physical, Hyper-V, AWS, GCP
      - Continuous replication
    
  Database Migration:
    Azure Database Migration Service (DMS):
      - SQL Server → Azure SQL DB / MI
      - MySQL → Azure MySQL
      - PostgreSQL → Azure PostgreSQL
      - MongoDB → Cosmos DB
      - Online (minimal downtime) migration
    
    az dms create \
      --name myDMS -g myRG \
      --sku-name Premium_4vCores \
      --location eastus
  
  Web App Migration:
    Azure App Service Migration Assistant
    - Assess .NET/Java web apps
    - Migrate to App Service
    - Configuration migration

Migration strategies (6 Rs):
  Rehost (Lift and Shift):
    Move as-is to Azure VMs
    Fast, minimal changes
    Use Azure Migrate Server Migration
  
  Replatform:
    Minor modifications for Azure PaaS
    e.g., SQL Server → Azure SQL MI
    App → App Service (with small changes)
  
  Refactor:
    Rearchitect for cloud-native
    Microservices, containers, serverless
    Most effort, most benefit
  
  Repurchase:
    Replace with SaaS solution
    e.g., On-prem Exchange → Microsoft 365
  
  Retire:
    Decommission unused workloads
    Reduce portfolio before migration
  
  Retain:
    Keep on-premises (for now)
    Compliance, latency, or dependency reasons
` + "```" + `

**Azure Data Migration:**
` + "```" + `
Data Box family:
  Data Box Disk:     Up to 40 TB (SSD disks)
  Data Box:          Up to 80 TB (rugged appliance)
  Data Box Heavy:    Up to 1 PB (heavy appliance)
  
  Process:
    1. Order Data Box from Azure portal
    2. Receive appliance
    3. Copy data via SMB/NFS/REST
    4. Ship back to Azure datacenter
    5. Data uploaded to Azure Storage
    6. Appliance securely wiped

AzCopy:
  # Copy local to blob
  azcopy copy '/data/*' \
    'https://mystorageacct.blob.core.windows.net/data?<SAS>' \
    --recursive
  
  # Copy between storage accounts
  azcopy copy \
    'https://source.blob.core.windows.net/data/*?<SAS>' \
    'https://dest.blob.core.windows.net/data?<SAS>' \
    --recursive
  
  # Sync (only copy changed files)
  azcopy sync '/data' \
    'https://mystorageacct.blob.core.windows.net/data?<SAS>'

Storage Mover:
  - Managed migration service
  - Agent-based, runs on-premises
  - NFS/SMB to Azure Files/Blob
  - Incremental copy support
  - Migration project management

Azure File Sync for migration:
  1. Deploy Azure File Sync agent on-premises
  2. Register server with Storage Sync Service
  3. Create sync group (Azure Files ↔ on-premises)
  4. Initial sync (can take days for large datasets)
  5. Enable cloud tiering
  6. Cut over: point apps to Azure Files
  7. Decommission on-premises server

Import/Export service:
  - Ship your own hard drives
  - Up to 10 drives per job
  - BitLocker encrypted
  - Import to Blob/Files
  - Export from Blob
  
  az import-export create \
    --name myImportJob -g myRG \
    --location eastus \
    --type Import \
    --storage-account mystorageacct
` + "```" + `

**Cloud Adoption Framework (CAF):**
` + "```" + `
Microsoft Cloud Adoption Framework:
  
  Strategy:
    - Define business justification
    - Identify motivations
    - Define business outcomes
    - Build business case (ROI)
  
  Plan:
    - Digital estate assessment
    - Rationalize workloads (5 Rs)
    - Migration backlog
    - Skills readiness
    - Cloud adoption plan (ADO)
  
  Ready:
    - Azure landing zone
    - Management groups hierarchy
    - Subscription design
    - Network topology (hub-spoke)
    - Identity model
    - Governance baseline
  
  Adopt:
    Migrate:
      - Assessment → Migration → Optimization
      - Iterative approach (waves)
    
    Innovate:
      - Build cloud-native applications
      - AI/ML integration
      - IoT solutions
  
  Govern:
    - Cost Management discipline
    - Security Baseline
    - Identity Baseline
    - Resource Consistency
    - Deployment Acceleration
  
  Manage:
    - Operations baseline
    - Business alignment
    - Platform operations
    - Workload operations

Landing Zone:
  Enterprise-scale architecture:
    
    Root Management Group
    ├── Platform
    │   ├── Management (Log Analytics, Automation)
    │   ├── Identity (AD DS, Azure AD Connect)
    │   └── Connectivity (Hub VNet, Firewall, VPN/ER)
    ├── Landing Zones
    │   ├── Corp (internal apps, VNet connected)
    │   └── Online (internet-facing, public endpoints)
    ├── Sandbox (experimentation)
    └── Decommissioned
  
  Deploy:
    Azure Landing Zone accelerator (portal wizard)
    Terraform module: Azure/caf-enterprise-scale
    Bicep: ALZ-Bicep repo
` + "```" + ``,
					CodeExamples: `# Azure migration scripts

# 1. Migration readiness assessment
#!/bin/bash
echo "=== Migration Readiness Assessment ==="

# Azure Migrate projects
echo "--- Azure Migrate Projects ---"
az resource list --resource-type "Microsoft.Migrate/migrateProjects" \
    --query "[].{name:name, rg:resourceGroup, location:location}" \
    -o table 2>/dev/null

# Current resource inventory
echo ""
echo "--- Resource Inventory ---"
echo "  VMs:"
az vm list --query "[].{name:name, size:hardwareProfile.vmSize, os:storageProfile.osDisk.osType}" \
    -o table 2>/dev/null | head -10

echo ""
echo "  SQL Databases:"
az sql db list --query "[?name!='master'].{name:name, server:serverName, sku:currentSku.name}" \
    -o table 2>/dev/null | head -10

echo ""
echo "  Storage Accounts:"
az storage account list --query "[].{name:name, kind:kind, sku:sku.name}" \
    -o table 2>/dev/null | head -10

# Resource count by type
echo ""
echo "--- Resource Count by Type ---"
az resource list \
    --query "[].type" -o tsv 2>/dev/null | sort | uniq -c | sort -rn | head -15

# 2. Landing zone checklist
#!/bin/bash
echo "=== Landing Zone Checklist ==="

# Management groups
echo "--- Management Groups ---"
az account management-group list \
    --query "[].{name:name, displayName:displayName}" \
    -o table 2>/dev/null

# Subscriptions
echo ""
echo "--- Subscriptions ---"
az account list \
    --query "[].{name:name, id:id, state:state}" \
    -o table 2>/dev/null

# Policy assignments
echo ""
echo "--- Policy Assignments ---"
POLICY_COUNT=$(az policy assignment list --query "length(@)" -o tsv 2>/dev/null)
echo "  Total policy assignments: $POLICY_COUNT"

# Hub VNet
echo ""
echo "--- Hub Network ---"
az network vnet list \
    --query "[?tags.role=='hub'].{name:name, rg:resourceGroup, space:addressSpace.addressPrefixes[0]}" \
    -o table 2>/dev/null

# Firewall
echo ""
echo "--- Azure Firewall ---"
az network firewall list \
    --query "[].{name:name, rg:resourceGroup, state:provisioningState}" \
    -o table 2>/dev/null

# Log Analytics
echo ""
echo "--- Central Logging ---"
az monitor log-analytics workspace list \
    --query "[].{name:name, rg:resourceGroup, retention:retentionInDays}" \
    -o table 2>/dev/null

# 3. Data migration progress tracker
#!/bin/bash
echo "=== Data Migration Progress ==="

# AzCopy jobs
echo "--- AzCopy Active Jobs ---"
azcopy jobs list 2>/dev/null | head -20

# Storage account data
echo ""
echo "--- Storage Accounts with Data ---"
for acct in $(az storage account list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az storage account list --query "[?name=='$acct'].resourceGroup" -o tsv | head -1)
    
    # Used capacity metric
    USED=$(az monitor metrics list \
        --resource "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$acct" \
        --metric "UsedCapacity" \
        --interval PT1H \
        --query "value[0].timeseries[0].data[-1].average" \
        -o tsv 2>/dev/null)
    
    if [ -n "$USED" ] && [ "$USED" != "None" ]; then
        USED_GB=$(echo "scale=2; $USED / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "N/A")
        echo "  $acct: ${USED_GB} GB"
    fi
done`,
				},
			},
		},
	})
}
