package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          430,
			Title:       "Azure Storage Advanced",
			Description: "Advanced Azure Storage: blob lifecycle management, Azure Files, and Azure Data Lake Storage.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "Blob Lifecycle Management",
					Content: `Blob lifecycle management automatically transitions blobs between access tiers and deletes old blobs.

**Lifecycle Policies:**
- **Rules**: Define actions based on conditions
- **Actions**: Transition tiers or delete blobs
- **Conditions**: Age, prefix, blob type
- **Automatic**: Policies run automatically

**Policy Actions:**
- **TierToCool**: Move from Hot to Cool tier
- **TierToArchive**: Move to Archive tier
- **Delete**: Delete blob or snapshot
- **DeleteBlob**: Delete blob versions

**Policy Conditions:**
- **Age**: Days since last modification
- **Prefix**: Blob name prefix
- **Blob Types**: Block blobs, append blobs
- **Base Blob**: Apply to base blob
- **Snapshots**: Apply to snapshots
- **Versions**: Apply to versions

**Best Practices:**
- Enable lifecycle management for cost optimization
- Set appropriate age thresholds
- Use prefixes for organization
- Test policies before applying
- Monitor policy execution
- Review and adjust policies regularly`,
					CodeExamples: `# Create lifecycle management policy
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
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 90
            },
            "delete": {
              "daysAfterModificationGreaterThan": 2555
            }
          }
        }
      }
    }
  ]
}

# View lifecycle policy
az storage account management-policy show \\
    --account-name mystorageaccount \\
    --resource-group myResourceGroup`,
				},
				{
					Title: "Azure Files",
					Content: `Azure Files provides fully managed file shares in the cloud.

**File Share Features:**
- **SMB Protocol**: Standard SMB 2.1, 3.0, 3.1.1
- **NFS Protocol**: NFS 4.1 for Linux
- **Concurrent Access**: Multiple clients can access simultaneously
- **Snapshot Support**: Point-in-time backups

**File Share Types:**
- **Standard**: HDD-backed, cost-effective
- **Premium**: SSD-backed, high performance
- **Large File Shares**: Up to 100 TiB

**Access Methods:**
- **SMB Mount**: Mount as network drive
- **REST API**: Programmatic access
- **Azure File Sync**: Sync to on-premises

**Best Practices:**
- Use Premium for performance-critical workloads
- Enable snapshots for backup
- Use Azure File Sync for hybrid scenarios
- Secure with private endpoints
- Monitor file share usage`,
					CodeExamples: `# Create file share
az storage share create \\
    --name myshare \\
    --account-name mystorageaccount \\
    --quota 100 \\
    --auth-mode login

# Mount file share (Linux)
sudo mount -t cifs //mystorageaccount.file.core.windows.net/myshare /mnt/myfiles \\
    -o vers=3.0,username=mystorageaccount,password=<account-key>,dir_mode=0777,file_mode=0777

# Create snapshot
az storage share snapshot \\
    --name myshare \\
    --account-name mystorageaccount \\
    --auth-mode login`,
				},
				{
					Title: "Azure Data Lake Storage",
					Content: `Azure Data Lake Storage Gen2 provides big data analytics storage.

**Data Lake Storage Features:**
- **Hierarchical Namespace**: Directory structure
- **HDFS Compatibility**: Compatible with Hadoop
- **Massive Scale**: Exabyte-scale storage
- **Multi-protocol Access**: Blob API and Data Lake API

**Use Cases:**
- **Big Data Analytics**: Analytics workloads
- **Machine Learning**: ML data storage
- **Data Warehousing**: Data warehouse storage
- **Streaming Analytics**: Real-time analytics

**Best Practices:**
- Use hierarchical namespace
- Organize data by date/service
- Use appropriate access tiers
- Implement proper security
- Monitor data lake usage`,
					CodeExamples: `# Create storage account with Data Lake
az storage account create \\
    --name mydatalake \\
    --resource-group myResourceGroup \\
    --location eastus \\
    --sku Standard_LRS \\
    --kind StorageV2 \\
    --enable-hierarchical-namespace true

# Create filesystem (container)
az storage fs create \\
    --name myfilesystem \\
    --account-name mydatalake \\
    --auth-mode login`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          431,
			Title:       "Azure Database Services",
			Description: "Additional Azure database services: MySQL and Redis Cache.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Database for MySQL",
					Content: `Azure Database for MySQL is a fully managed MySQL database service.

**Deployment Options:**
- **Single Server**: Traditional deployment
- **Flexible Server**: More control and flexibility

**Service Tiers:**
- **Basic**: Development and testing
- **General Purpose**: Balanced compute and storage
- **Memory Optimized**: High-performance workloads

**Features:**
- Automated backups
- High availability options
- Read replicas
- Advanced threat protection
- MySQL extensions support

**Best Practices:**
- Use Flexible Server for production
- Enable high availability
- Use read replicas for read scaling
- Monitor performance metrics`,
					CodeExamples: `# Create MySQL Flexible Server
az mysql flexible-server create \\
    --resource-group myResourceGroup \\
    --name myserver \\
    --location eastus \\
    --admin-user myadmin \\
    --admin-password SecurePassword123! \\
    --sku-name Standard_B1ms \\
    --tier GeneralPurpose \\
    --storage-size 32 \\
    --version 8.0.21`,
				},
				{
					Title: "Azure Cache for Redis",
					Content: `Azure Cache for Redis provides in-memory data store for fast data access.

**Redis Features:**
- **In-Memory Storage**: Ultra-fast data access
- **Data Structures**: Strings, hashes, lists, sets, sorted sets
- **Persistence**: Optional data persistence
- **Pub/Sub**: Publish-subscribe messaging

**Cache Tiers:**
- **Basic**: Single node, development
- **Standard**: Two nodes, replication
- **Premium**: Advanced features, clustering

**Use Cases:**
- **Caching**: Application caching
- **Session Store**: Web session storage
- **Real-time Analytics**: Real-time data processing
- **Message Broker**: Pub/sub messaging

**Best Practices:**
- Use Standard or Premium for production
- Implement cache-aside pattern
- Monitor cache hit ratio
- Set appropriate expiration
- Use Redis persistence for critical data`,
					CodeExamples: `# Create Redis cache
az redis create \\
    --resource-group myResourceGroup \\
    --name myredis \\
    --location eastus \\
    --sku Basic \\
    --vm-size c0

# Get Redis connection string
az redis list-keys \\
    --resource-group myResourceGroup \\
    --name myredis`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          432,
			Title:       "Azure Networking Advanced",
			Description: "Advanced networking services: Azure Firewall, Network Watcher, and Private Link.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Firewall",
					Content: `Azure Firewall is a managed, cloud-based network security service.

**Firewall Features:**
- **Network Rules**: Allow/deny network traffic
- **Application Rules**: Allow/deny application traffic
- **NAT Rules**: Destination NAT
- **Threat Intelligence**: Known malicious IPs and domains

**Firewall Policies:**
- **Rule Collections**: Group related rules
- **Priority**: Rule evaluation order
- **Action**: Allow or Deny

**Best Practices:**
- Use firewall policies for centralized management
- Implement proper rule priorities
- Enable threat intelligence
- Monitor firewall logs
- Test rules before applying`,
					CodeExamples: `# Create firewall
az network firewall create \\
    --resource-group myResourceGroup \\
    --name myFirewall \\
    --location eastus

# Create firewall policy
az network firewall policy create \\
    --resource-group myResourceGroup \\
    --name myFirewallPolicy \\
    --location eastus

# Add network rule
az network firewall policy rule-collection-group collection add-filter-collection \\
    --resource-group myResourceGroup \\
    --policy-name myFirewallPolicy \\
    --rule-collection-group-name DefaultNetworkRuleCollectionGroup \\
    --name myNetworkRule \\
    --rule-type NetworkRule \\
    --priority 100 \\
    --action Allow \\
    --rule-name AllowHTTPS \\
    --source-addresses 10.0.0.0/8 \\
    --destination-addresses * \\
    --destination-ports 443 \\
    --protocols TCP`,
				},
				{
					Title: "Network Watcher",
					Content: `Network Watcher provides tools to monitor, diagnose, and troubleshoot networking issues.

**Network Watcher Features:**
- **Packet Capture**: Capture network packets
- **Connection Monitor**: Monitor connectivity
- **IP Flow Verify**: Verify IP flow rules
- **Next Hop**: Determine next hop for traffic
- **VPN Troubleshoot**: Troubleshoot VPN connections

**Use Cases:**
- **Troubleshooting**: Diagnose network issues
- **Monitoring**: Monitor network connectivity
- **Security**: Analyze network traffic
- **Performance**: Identify performance issues

**Best Practices:**
- Enable Network Watcher in all regions
- Use Connection Monitor for ongoing monitoring
- Capture packets for detailed analysis
- Document troubleshooting procedures`,
					CodeExamples: `# Enable Network Watcher
az network watcher configure \\
    --resource-group NetworkWatcherRG \\
    --locations eastus westus \\
    --enabled true

# Create connection monitor
az network watcher connection-monitor create \\
    --resource-group myResourceGroup \\
    --name myConnectionMonitor \\
    --source-resource myVM \\
    --monitor-interval 60`,
				},
				{
					Title: "Private Link",
					Content: `Private Link provides private connectivity to Azure services.

**Private Link Benefits:**
- **Private IP**: Access services via private IP
- **VNet Integration**: Extend VNet to services
- **No Public Exposure**: Services not exposed to internet
- **Simplified Networking**: No NAT or gateway needed

**Private Endpoint:**
- **Private IP**: Private IP address in VNet
- **DNS Integration**: Automatic DNS resolution
- **Network Interface**: Network interface in VNet

**Supported Services:**
- **Storage**: Blob, File, Queue, Table
- **SQL Database**: SQL Database and SQL Data Warehouse
- **Cosmos DB**: Cosmos DB accounts
- **Key Vault**: Key Vault service
- **Many More**: 100+ Azure services

**Best Practices:**
- Use private endpoints for sensitive data
- Configure DNS properly
- Use Network Security Groups
- Monitor private endpoint connections`,
					CodeExamples: `# Create private endpoint
az network private-endpoint create \\
    --resource-group myResourceGroup \\
    --name myPrivateEndpoint \\
    --vnet-name myVNet \\
    --subnet mySubnet \\
    --private-connection-resource-id /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Storage/storageAccounts/mystorageaccount \\
    --group-id blob \\
    --connection-name myConnection`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          433,
			Title:       "Azure Security Advanced",
			Description: "Advanced security services: Azure Sentinel, Azure Defender, and Identity Protection.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Sentinel",
					Content: `Azure Sentinel is a cloud-native SIEM (Security Information and Event Management) solution.

**Sentinel Features:**
- **Security Analytics**: AI-powered security analytics
- **Threat Detection**: Detect threats across environment
- **Incident Management**: Security incident management
- **Hunting**: Proactive threat hunting
- **Automation**: Automated response playbooks

**Data Sources:**
- **Azure Services**: Native Azure service integration
- **On-Premises**: Connect on-premises systems
- **Third-Party**: Connect third-party security tools
- **Custom**: Custom data connectors

**Best Practices:**
- Connect all relevant data sources
- Create custom detection rules
- Use playbooks for automation
- Regular threat hunting
- Monitor Sentinel health`,
					CodeExamples: `# Create Sentinel workspace
az sentinel workspace create \\
    --resource-group myResourceGroup \\
    --workspace-name mySentinelWorkspace \\
    --location eastus`,
				},
				{
					Title: "Azure Defender",
					Content: `Azure Defender provides advanced threat protection for Azure resources.

**Defender Plans:**
- **Defender for Servers**: VM and server protection
- **Defender for App Service**: App Service protection
- **Defender for Storage**: Storage account protection
- **Defender for SQL**: SQL Database protection
- **Defender for Key Vault**: Key Vault protection

**Protection Features:**
- **Threat Detection**: Detect threats and attacks
- **Vulnerability Assessment**: Identify vulnerabilities
- **Just-in-Time Access**: Reduce attack surface
- **Adaptive Application Controls**: Whitelist applications

**Best Practices:**
- Enable Defender for all resources
- Review security recommendations
- Implement Just-in-Time access
- Use adaptive application controls
- Monitor security alerts`,
					CodeExamples: `# Enable Defender for Servers
az security pricing create \\
    --name "VirtualMachines" \\
    --tier "Standard" \\
    --resource-group myResourceGroup`,
				},
				{
					Title: "Identity Protection",
					Content: `Azure AD Identity Protection detects and responds to identity-based risks.

**Risk Detection:**
- **Sign-in Risk**: Risky sign-in attempts
- **User Risk**: Compromised user accounts
- **Risk Levels**: Low, Medium, High
- **Real-time Detection**: Immediate risk detection

**Risk Policies:**
- **User Risk Policy**: Respond to user risk
- **Sign-in Risk Policy**: Respond to sign-in risk
- **MFA Registration**: Require MFA registration

**Best Practices:**
- Enable Identity Protection
- Configure risk policies
- Require MFA for high-risk sign-ins
- Review risk detections regularly
- Investigate high-risk users`,
					CodeExamples: `# Enable Identity Protection (via Azure Portal or Microsoft Graph API)
# Identity Protection is typically configured via Azure Portal UI`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          434,
			Title:       "Azure AI/ML Services",
			Description: "Azure AI and Machine Learning services: Machine Learning, Cognitive Services, and Azure OpenAI.",
			Order:       34,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Machine Learning",
					Content: `Azure Machine Learning provides cloud-based ML platform for training and deploying models.

**ML Features:**
- **Automated ML**: Automated model training
- **ML Designer**: Visual model designer
- **Notebooks**: Jupyter notebooks integration
- **MLOps**: ML operations and deployment
- **Model Registry**: Model versioning and management

**Compute Options:**
- **Compute Instances**: Development workstations
- **Compute Clusters**: Training clusters
- **Inference Clusters**: Model deployment
- **Attached Compute**: Attach existing compute

**Best Practices:**
- Use automated ML for quick results
- Version control experiments
- Implement MLOps practices
- Monitor model performance
- Retrain models regularly`,
					CodeExamples: `# Create ML workspace
az ml workspace create \\
    --resource-group myResourceGroup \\
    --name myMLWorkspace \\
    --location eastus

# Create compute instance
az ml compute create \\
    --resource-group myResourceGroup \\
    --workspace-name myMLWorkspace \\
    --name myCompute \\
    --type ComputeInstance \\
    --size Standard_DS2_v2`,
				},
				{
					Title: "Cognitive Services",
					Content: `Azure Cognitive Services provide AI capabilities via APIs.

**Cognitive Service Categories:**
- **Vision**: Image and video analysis
- **Speech**: Speech recognition and synthesis
- **Language**: Text analysis and translation
- **Decision**: Content moderation and personalization
- **Search**: Bing Search integration

**Popular Services:**
- **Computer Vision**: Image analysis
- **Text Analytics**: Sentiment analysis
- **Translator**: Text translation
- **Speech Services**: Speech-to-text, text-to-speech
- **Form Recognizer**: Document processing

**Best Practices:**
- Use appropriate service for use case
- Implement proper error handling
- Monitor API usage and costs
- Secure API keys
- Cache results when possible`,
					CodeExamples: `# Create Cognitive Services account
az cognitiveservices account create \\
    --resource-group myResourceGroup \\
    --name myCognitiveService \\
    --kind ComputerVision \\
    --sku S1 \\
    --location eastus

# Get API key
az cognitiveservices account keys list \\
    --resource-group myResourceGroup \\
    --name myCognitiveService`,
				},
				{
					Title: "Azure OpenAI",
					Content: `Azure OpenAI provides access to OpenAI's language models through Azure.

**OpenAI Models:**
- **GPT Models**: Text generation and completion
- **Embeddings**: Text embeddings
- **Code Models**: Code generation
- **Image Models**: Image generation

**Features:**
- **Enterprise Security**: Azure security and compliance
- **Private Endpoints**: Private connectivity
- **Content Filtering**: Built-in content filtering
- **Regional Availability**: Deploy in specific regions

**Use Cases:**
- **Content Generation**: Generate text content
- **Code Generation**: Generate code
- **Conversational AI**: Chatbots and assistants
- **Summarization**: Summarize documents

**Best Practices:**
- Use appropriate model for task
- Implement content filtering
- Monitor token usage
- Use private endpoints for security
- Implement proper error handling`,
					CodeExamples: `# Create OpenAI resource (requires approval)
az cognitiveservices account create \\
    --resource-group myResourceGroup \\
    --name myOpenAI \\
    --kind OpenAI \\
    --sku S0 \\
    --location eastus

# Note: Azure OpenAI requires approval and may not be available in all regions`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
