package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1243,
			Title:       "Azure Storage Services",
			Description: "Master Azure Storage including Blob, File, Queue, Table storage, data redundancy options, access tiers, lifecycle management, and security.",
			Order:       43,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Blob Storage and Data Management",
					Content: `Azure Storage provides massively scalable cloud storage for unstructured data, files, messages, and NoSQL data.

**Storage Account Types:**
` + "```" + `
Account types:
  Standard general-purpose v2:
    - Most common, recommended
    - Blob, File, Queue, Table
    - LRS, GRS, ZRS, GZRS redundancy
    - Hot, Cool, Cold, Archive tiers
  
  Premium block blobs:
    - High transaction rates
    - Low latency
    - LRS, ZRS only
  
  Premium file shares:
    - Enterprise file shares
    - SMB and NFS protocols
    - LRS, ZRS only
  
  Premium page blobs:
    - VM disks (unmanaged)
    - LRS only

Redundancy options:
  LRS  (Locally Redundant):
    3 copies within single datacenter
    11 nines durability
    Cheapest
  
  ZRS  (Zone Redundant):
    3 copies across 3 availability zones
    12 nines durability
    Higher availability
  
  GRS  (Geo-Redundant):
    LRS locally + LRS in paired region
    16 nines durability
    Regional disaster recovery
  
  GZRS (Geo-Zone Redundant):
    ZRS locally + LRS in paired region
    16 nines durability
    Best protection
  
  RA-GRS / RA-GZRS:
    Read access to secondary region
    Higher read availability

Access tiers:
  Hot:      Frequent access, highest storage cost, lowest access cost
  Cool:     Infrequent (30+ days), lower storage, higher access
  Cold:     Rarely accessed (90+ days), even lower storage cost
  Archive:  Long-term (180+ days), lowest storage, highest access
            Offline tier, requires rehydration (hours)
` + "```" + `

**Blob Storage:**
` + "```" + `
Blob types:
  Block blobs:   Most common, up to 190.7 TB, optimized for streaming/upload
  Append blobs:  Optimized for append operations (logs)
  Page blobs:    Random read/write, up to 8 TB (VM disks)

Container and blob hierarchy:
  Storage Account → Container → Blob
  
  URL: https://<account>.blob.core.windows.net/<container>/<blob>

Azure CLI:
  # Create storage account
  az storage account create \
    --resource-group myRG \
    --name mystorageacct \
    --location eastus \
    --sku Standard_ZRS \
    --kind StorageV2 \
    --access-tier Hot \
    --min-tls-version TLS1_2
  
  # Create container
  az storage container create \
    --account-name mystorageacct \
    --name mycontainer \
    --public-access off
  
  # Upload blob
  az storage blob upload \
    --account-name mystorageacct \
    --container-name mycontainer \
    --name myfile.txt \
    --file ./localfile.txt \
    --tier Hot
  
  # Download blob
  az storage blob download \
    --account-name mystorageacct \
    --container-name mycontainer \
    --name myfile.txt \
    --file ./downloaded.txt
  
  # List blobs
  az storage blob list \
    --account-name mystorageacct \
    --container-name mycontainer \
    --output table
  
  # Set tier
  az storage blob set-tier \
    --account-name mystorageacct \
    --container-name mycontainer \
    --name myfile.txt \
    --tier Cool
  
  # Delete blob
  az storage blob delete \
    --account-name mystorageacct \
    --container-name mycontainer \
    --name myfile.txt

AzCopy (fast bulk transfer):
  # Login
  azcopy login
  
  # Upload directory
  azcopy copy './local-dir/*' \
    'https://mystorageacct.blob.core.windows.net/mycontainer/' \
    --recursive
  
  # Copy between accounts
  azcopy copy \
    'https://source.blob.core.windows.net/container/*' \
    'https://dest.blob.core.windows.net/container/' \
    --recursive
  
  # Sync (like rsync)
  azcopy sync './local-dir' \
    'https://mystorageacct.blob.core.windows.net/mycontainer' \
    --recursive --delete-destination true
` + "```" + `

**Lifecycle Management:**
` + "```" + `
Automatically transition or delete blobs based on age.

Policy example:
  {
    "rules": [
      {
        "name": "moveToCoool",
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
              },
              "tierToCold": {
                "daysAfterModificationGreaterThan": 90
              },
              "tierToArchive": {
                "daysAfterModificationGreaterThan": 180
              },
              "delete": {
                "daysAfterModificationGreaterThan": 365
              }
            },
            "snapshot": {
              "delete": {
                "daysAfterCreationGreaterThan": 90
              }
            }
          }
        }
      }
    ]
  }

Azure CLI:
  az storage account management-policy create \
    --account-name mystorageacct \
    --resource-group myRG \
    --policy @lifecycle-policy.json
` + "```" + `

**Security:**
` + "```" + `
Authentication methods:
  1. Azure AD (recommended):
     - RBAC roles: Storage Blob Data Reader/Contributor/Owner
     - Managed Identity support
  
  2. Shared Key (account key):
     - Full access to account
     - Should be rotated regularly
     - Use Azure Key Vault to store keys
  
  3. Shared Access Signature (SAS):
     - Delegated access with time limits
     - Account SAS, Service SAS, User Delegation SAS
     
     # Generate SAS token
     az storage container generate-sas \
       --account-name mystorageacct \
       --name mycontainer \
       --permissions rl \
       --expiry 2024-12-31 \
       --https-only
  
  4. Stored Access Policy:
     - Named policies on containers
     - Can be revoked without regenerating keys

Encryption:
  - All data encrypted at rest (AES-256)
  - Microsoft-managed keys (default)
  - Customer-managed keys (Azure Key Vault)
  - Infrastructure encryption (double encryption)

Network security:
  # Restrict to specific VNet
  az storage account network-rule add \
    --account-name mystorageacct \
    --vnet-name myVNet --subnet app-subnet
  
  # Default deny
  az storage account update \
    --name mystorageacct \
    --default-action Deny
  
  # Private endpoint
  az network private-endpoint create \
    --resource-group myRG --name storage-pe \
    --vnet-name myVNet --subnet app-subnet \
    --private-connection-resource-id "/subscriptions/.../storageAccounts/mystorageacct" \
    --group-ids blob --connection-name storage-connection
` + "```" + ``,
					CodeExamples: `# Azure Storage management scripts

# 1. Storage account audit
#!/bin/bash
echo "=== Azure Storage Audit ==="

az storage account list --query "[].{
    name:name, rg:resourceGroup, kind:kind,
    sku:sku.name, tier:accessTier, tls:minimumTlsVersion,
    https:enableHttpsTrafficOnly, public:allowBlobPublicAccess
}" -o table

echo ""
echo "--- Public Access Warning ---"
az storage account list --query "[?allowBlobPublicAccess==true].{
    name:name, rg:resourceGroup
}" -o table 2>/dev/null

echo ""
echo "--- Network Rules ---"
for acct in $(az storage account list --query "[].name" -o tsv); do
    DEFAULT=$(az storage account show -n "$acct" \
        --query "networkRuleSet.defaultAction" -o tsv 2>/dev/null)
    if [ "$DEFAULT" = "Allow" ]; then
        echo "  WARNING: $acct allows all network access"
    fi
done

# 2. Blob storage cost estimator
#!/bin/bash
ACCOUNT="${1:?Usage: $0 <account-name>}"

echo "=== Storage Cost Analysis: $ACCOUNT ==="

for container in $(az storage container list --account-name "$ACCOUNT" \
    --query "[].name" -o tsv 2>/dev/null); do
    
    echo ""
    echo "Container: $container"
    
    # Count blobs by tier
    for tier in Hot Cool Cold Archive; do
        COUNT=$(az storage blob list --account-name "$ACCOUNT" \
            --container-name "$container" \
            --query "[?properties.blobTier=='$tier'] | length(@)" \
            -o tsv 2>/dev/null)
        
        if [ "$COUNT" -gt 0 ]; then
            SIZE=$(az storage blob list --account-name "$ACCOUNT" \
                --container-name "$container" \
                --query "[?properties.blobTier=='$tier'].properties.contentLength | sum(@)" \
                -o tsv 2>/dev/null)
            SIZE_GB=$(echo "scale=2; $SIZE / 1073741824" | bc)
            echo "  $tier: $COUNT blobs, ${SIZE_GB}GB"
        fi
    done
done

# 3. Storage backup script
#!/bin/bash
SOURCE_ACCOUNT="${1:?Usage: $0 <source-account> <dest-account>}"
DEST_ACCOUNT="${2:?Usage: $0 <source-account> <dest-account>}"

echo "=== Syncing Storage: $SOURCE_ACCOUNT → $DEST_ACCOUNT ==="

# Get source containers
for container in $(az storage container list --account-name "$SOURCE_ACCOUNT" \
    --query "[].name" -o tsv 2>/dev/null); do
    
    echo "Syncing container: $container"
    
    # Create dest container if not exists
    az storage container create \
        --account-name "$DEST_ACCOUNT" \
        --name "$container" 2>/dev/null
    
    # Sync using azcopy
    azcopy sync \
        "https://$SOURCE_ACCOUNT.blob.core.windows.net/$container" \
        "https://$DEST_ACCOUNT.blob.core.windows.net/$container" \
        --recursive 2>&1 | tail -5
done

echo "Sync complete."`,
				},
				{
					Title: "Azure Files, Queues, and Table Storage",
					Content: `Azure Storage includes file shares, message queues, and NoSQL table storage alongside blob storage.

**Azure Files:**
` + "```" + `
Fully managed file shares in the cloud accessible via SMB or NFS.

Features:
  - SMB 3.0 and NFS 4.1 protocols
  - Mount on Windows, Linux, macOS
  - Azure File Sync for hybrid scenarios
  - Snapshots and soft delete
  - AD DS and Azure AD DS authentication

Create and mount:
  # Create file share
  az storage share-rm create \
    --resource-group myRG \
    --storage-account mystorageacct \
    --name myshare \
    --quota 100 \
    --access-tier Hot
  
  # Get connection string
  CONN=$(az storage account show-connection-string \
    --name mystorageacct -o tsv)
  
  # Mount on Linux (SMB)
  STORAGE_KEY=$(az storage account keys list \
    --account-name mystorageacct --query "[0].value" -o tsv)
  
  mkdir -p /mnt/azure/myshare
  mount -t cifs //mystorageacct.file.core.windows.net/myshare \
    /mnt/azure/myshare \
    -o vers=3.0,username=mystorageacct,password=$STORAGE_KEY,serverino
  
  # Persistent mount (/etc/fstab)
  # //mystorageacct.file.core.windows.net/myshare /mnt/azure/myshare cifs nofail,credentials=/etc/smbcredentials/mystorageacct.cred,serverino 0 0
  
  # Mount on Windows
  # net use Z: \\mystorageacct.file.core.windows.net\myshare /u:mystorageacct $STORAGE_KEY

Azure File Sync:
  - Sync on-premises file servers with Azure Files
  - Cloud tiering: frequently accessed files local, rest in Azure
  - Multi-site sync: same share on multiple servers
  - Rapid DR: restore a new server from cloud quickly
  
  Components:
    Storage Sync Service → Sync Group → Cloud Endpoint + Server Endpoints
` + "```" + `

**Azure Queue Storage:**
` + "```" + `
Simple message queue for asynchronous communication.

Characteristics:
  - Up to 64KB per message
  - Up to 500TB total queue size
  - 7-day default message TTL (configurable)
  - At-least-once delivery
  - FIFO not guaranteed (use Service Bus for strict FIFO)

Operations:
  # Create queue
  az storage queue create \
    --account-name mystorageacct \
    --name myqueue
  
  # Send message
  az storage message put \
    --account-name mystorageacct \
    --queue-name myqueue \
    --content "$(echo -n 'Hello World' | base64)"
  
  # Peek messages (don't dequeue)
  az storage message peek \
    --account-name mystorageacct \
    --queue-name myqueue \
    --num-messages 5
  
  # Get messages (dequeue with visibility timeout)
  az storage message get \
    --account-name mystorageacct \
    --queue-name myqueue \
    --num-messages 1 \
    --visibility-timeout 30
  
  # Delete message
  az storage message delete \
    --account-name mystorageacct \
    --queue-name myqueue \
    --id <message-id> \
    --pop-receipt <pop-receipt>

Queue vs Service Bus:
  Queue Storage:
    + Simple, cheap (pay per message)
    + Large queues (500TB)
    + Audit trail (all messages logged)
    - No guaranteed FIFO
    - At-least-once delivery
  
  Service Bus:
    + Guaranteed FIFO
    + Exactly-once delivery
    + Topics/subscriptions (pub/sub)
    + Sessions (message grouping)
    + Dead-letter queue
    + Transaction support
    - More expensive
    - Smaller (80GB per queue/topic)
` + "```" + `

**Azure Table Storage:**
` + "```" + `
NoSQL key-value store for semi-structured data.

Characteristics:
  - Schemaless design
  - Partitioned by PartitionKey
  - Ordered by RowKey within partition
  - Cheap, highly available
  - No joins, no secondary indexes

Entity structure:
  PartitionKey:  String, required (determines partition)
  RowKey:        String, required (unique within partition)
  Timestamp:     Auto-generated
  Properties:    Up to 252 custom properties, 1MB per entity

Query patterns:
  Point query:      PartitionKey + RowKey (fastest)
  Partition scan:   PartitionKey only (medium)
  Table scan:       No partition key (slowest, avoid)

Azure CLI:
  # Create table
  az storage table create \
    --account-name mystorageacct \
    --name mytable
  
  # Insert entity
  az storage entity insert \
    --account-name mystorageacct \
    --table-name mytable \
    --entity PartitionKey=users RowKey=user001 \
      Name=John Email=john@example.com

Table Storage vs Cosmos DB Table API:
  Table Storage:
    + Very cheap
    + Simple setup
    - Limited throughput
    - Single region (without GRS)
    - No global distribution
  
  Cosmos DB Table API:
    + Global distribution
    + Guaranteed latency (<10ms)
    + Automatic indexing
    + 5 consistency levels
    - More expensive
    - Same API compatibility
` + "```" + ``,
					CodeExamples: `# Azure storage service scripts

# 1. File share backup with snapshots
#!/bin/bash
ACCOUNT="${1:?Usage: $0 <account-name> <share-name>}"
SHARE="${2:?Usage: $0 <account-name> <share-name>}"
MAX_SNAPSHOTS=30

echo "=== File Share Snapshot: $SHARE ==="

# Create snapshot
SNAPSHOT=$(az storage share snapshot \
    --account-name "$ACCOUNT" \
    --name "$SHARE" \
    --query "snapshot" -o tsv 2>/dev/null)

if [ -n "$SNAPSHOT" ]; then
    echo "Created snapshot: $SNAPSHOT"
else
    echo "ERROR: Failed to create snapshot"
    exit 1
fi

# List snapshots
SNAPSHOTS=$(az storage share list \
    --account-name "$ACCOUNT" \
    --include-snapshots \
    --query "[?name=='$SHARE' && snapshot!=null].snapshot" \
    -o tsv 2>/dev/null | sort)

COUNT=$(echo "$SNAPSHOTS" | wc -l)
echo "Total snapshots: $COUNT"

# Cleanup old snapshots
if [ "$COUNT" -gt "$MAX_SNAPSHOTS" ]; then
    REMOVE=$((COUNT - MAX_SNAPSHOTS))
    echo "Removing $REMOVE old snapshots..."
    
    echo "$SNAPSHOTS" | head -n "$REMOVE" | while read -r snap; do
        echo "  Deleting: $snap"
        az storage share delete \
            --account-name "$ACCOUNT" \
            --name "$SHARE" \
            --snapshot "$snap" 2>/dev/null
    done
fi

# 2. Queue monitoring script
#!/bin/bash
ACCOUNT="${1:?Usage: $0 <account-name>}"

echo "=== Queue Status: $ACCOUNT ==="

for queue in $(az storage queue list --account-name "$ACCOUNT" \
    --query "[].name" -o tsv 2>/dev/null); do
    
    # Get approximate message count
    META=$(az storage queue metadata show \
        --account-name "$ACCOUNT" \
        --queue-name "$queue" -o json 2>/dev/null)
    
    MSG_COUNT=$(echo "$META" | jq -r '.approximateMessageCount // "0"')
    
    STATUS="OK"
    if [ "$MSG_COUNT" -gt 1000 ]; then
        STATUS="WARNING: Queue depth high"
    elif [ "$MSG_COUNT" -gt 10000 ]; then
        STATUS="CRITICAL: Queue backed up"
    fi
    
    printf "  %-30s  Messages: %6s  [%s]\n" "$queue" "$MSG_COUNT" "$STATUS"
done

# 3. Table storage data export
#!/bin/bash
ACCOUNT="${1:?Usage: $0 <account-name> <table-name>}"
TABLE="${2:?Usage: $0 <account-name> <table-name>}"
OUTPUT="${3:-${TABLE}-export.json}"

echo "Exporting table: $TABLE → $OUTPUT"

az storage entity query \
    --account-name "$ACCOUNT" \
    --table-name "$TABLE" \
    --num-results 5000 \
    -o json > "$OUTPUT"

ENTITIES=$(jq '.items | length' "$OUTPUT")
echo "Exported $ENTITIES entities to $OUTPUT"`,
				},
			},
		},
	})
}
