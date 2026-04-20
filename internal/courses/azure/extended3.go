package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1239,
			Title:       "Azure Virtual Machines and Compute",
			Description: "Deep dive into Azure Virtual Machines including VM sizes, availability sets, scale sets, custom images, and compute optimization strategies.",
			Order:       39,
			Lessons: []problems.Lesson{
				{
					Title: "Azure VM Configuration and Management",
					Content: `Azure Virtual Machines provide on-demand, scalable computing resources. Understanding VM types, sizes, and configuration options is essential for cost-effective cloud architecture.

**VM Series and Sizes:**
` + "```" + `
Azure VM families:

General Purpose (balanced CPU/memory):
  B-series:   Burstable, variable workloads (B1s, B2s, B4ms)
  D-series:   Production workloads (D2s_v5, D4s_v5, D8s_v5)
  A-series:   Entry-level, dev/test
  
Compute Optimized (high CPU-to-memory):
  F-series:   CPU-intensive (F2s_v2, F4s_v2, F8s_v2)
  FX-series:  High frequency (up to 4.0 GHz)
  
Memory Optimized (high memory-to-CPU):
  E-series:   In-memory databases (E2s_v5, E4s_v5)
  M-series:   Very large memory (up to 4TB RAM)
  
Storage Optimized:
  L-series:   High disk throughput (L8s_v3, L16s_v3)
  
GPU:
  NC-series:  CUDA compute
  ND-series:  Deep learning training
  NV-series:  Visualization/rendering

Naming convention:
  [Family][Sub-family][vCPUs][Additive features]_[Accelerator type][Version]
  
  Example: Standard_D8s_v5
  D = General purpose
  8 = 8 vCPUs
  s = Premium storage capable
  v5 = Version 5
  
  Features:
  a = AMD processor
  d = Local temp disk
  i = Isolated
  l = Low memory
  m = Memory intensive
  p = ARM processor
  s = Premium storage
  t = Tiny (constrained vCPUs)
` + "```" + `

**Creating and Managing VMs:**
` + "```" + `
Azure CLI:
  # Create VM
  az vm create \
    --resource-group myRG \
    --name myVM \
    --image Ubuntu2204 \
    --size Standard_D2s_v5 \
    --admin-username azureuser \
    --ssh-key-values ~/.ssh/id_rsa.pub \
    --nsg-rule SSH \
    --public-ip-sku Standard \
    --zone 1

  # List available sizes in a region
  az vm list-sizes --location eastus --output table
  
  # List available images
  az vm image list --output table
  az vm image list --publisher Canonical --all --output table
  
  # Resize VM
  az vm resize --resource-group myRG --name myVM --size Standard_D4s_v5
  
  # Start/Stop/Deallocate
  az vm start --resource-group myRG --name myVM
  az vm stop --resource-group myRG --name myVM
  az vm deallocate --resource-group myRG --name myVM  # Stops billing
  
  # Show VM details
  az vm show --resource-group myRG --name myVM
  az vm get-instance-view --resource-group myRG --name myVM

Terraform:
  resource "azurerm_linux_virtual_machine" "example" {
    name                  = "myVM"
    resource_group_name   = azurerm_resource_group.example.name
    location              = azurerm_resource_group.example.location
    size                  = "Standard_D2s_v5"
    admin_username        = "azureuser"
    zone                  = "1"
    
    admin_ssh_key {
      username   = "azureuser"
      public_key = file("~/.ssh/id_rsa.pub")
    }
    
    network_interface_ids = [
      azurerm_network_interface.example.id,
    ]
    
    os_disk {
      caching              = "ReadWrite"
      storage_account_type = "Premium_LRS"
      disk_size_gb         = 128
    }
    
    source_image_reference {
      publisher = "Canonical"
      offer     = "0001-com-ubuntu-server-jammy"
      sku       = "22_04-lts-gen2"
      version   = "latest"
    }
    
    boot_diagnostics {
      storage_account_uri = azurerm_storage_account.diag.primary_blob_endpoint
    }
    
    tags = {
      Environment = "Production"
      Team        = "Platform"
    }
  }
` + "```" + `

**Disks and Storage:**
` + "```" + `
Managed disk types:
  Ultra Disk:     Highest performance, sub-ms latency
                  Up to 160,000 IOPS, 4,000 MB/s
                  Use: SAP HANA, top-tier databases
  
  Premium SSD v2: Flexible performance tuning
                  Up to 80,000 IOPS, 1,200 MB/s
                  Use: Production databases, enterprise apps
  
  Premium SSD:    High-performance SSD
                  Up to 20,000 IOPS, 900 MB/s
                  Use: Production workloads
  
  Standard SSD:   Cost-effective SSD
                  Up to 6,000 IOPS, 750 MB/s
                  Use: Web servers, light databases
  
  Standard HDD:   Lowest cost
                  Up to 2,000 IOPS, 500 MB/s
                  Use: Backup, archive, dev/test

Disk operations:
  # Create managed disk
  az disk create \
    --resource-group myRG \
    --name myDataDisk \
    --size-gb 256 \
    --sku Premium_LRS \
    --zone 1
  
  # Attach to VM
  az vm disk attach \
    --resource-group myRG \
    --vm-name myVM \
    --name myDataDisk
  
  # Detach
  az vm disk detach \
    --resource-group myRG \
    --vm-name myVM \
    --name myDataDisk
  
  # Snapshot
  az snapshot create \
    --resource-group myRG \
    --name mySnapshot \
    --source myDataDisk
  
  # Create disk from snapshot
  az disk create \
    --resource-group myRG \
    --name newDisk \
    --source mySnapshot
  
  # Disk encryption
  az vm encryption enable \
    --resource-group myRG \
    --name myVM \
    --disk-encryption-keyvault myKeyVault \
    --volume-type All
` + "```" + ``,
					CodeExamples: `# Azure VM management

# 1. VM provisioning script
#!/bin/bash
set -euo pipefail

RG="myapp-prod-rg"
LOCATION="eastus"
VNET="myapp-vnet"
SUBNET="app-subnet"
VM_PREFIX="web"
VM_COUNT=3
VM_SIZE="Standard_D2s_v5"
IMAGE="Ubuntu2204"

echo "=== Provisioning $VM_COUNT VMs ==="

# Create resource group
az group create --name "$RG" --location "$LOCATION"

# Create VNet and subnet
az network vnet create \
    --resource-group "$RG" \
    --name "$VNET" \
    --address-prefix 10.0.0.0/16 \
    --subnet-name "$SUBNET" \
    --subnet-prefix 10.0.1.0/24

# Create NSG
az network nsg create --resource-group "$RG" --name "${VM_PREFIX}-nsg"
az network nsg rule create \
    --resource-group "$RG" --nsg-name "${VM_PREFIX}-nsg" \
    --name AllowHTTP --priority 100 --direction Inbound \
    --access Allow --protocol Tcp --destination-port-ranges 80 443

# Create VMs
for i in $(seq 1 "$VM_COUNT"); do
    VM_NAME="${VM_PREFIX}-${i}"
    echo "Creating $VM_NAME..."
    
    az vm create \
        --resource-group "$RG" \
        --name "$VM_NAME" \
        --image "$IMAGE" \
        --size "$VM_SIZE" \
        --vnet-name "$VNET" \
        --subnet "$SUBNET" \
        --nsg "${VM_PREFIX}-nsg" \
        --admin-username azureuser \
        --ssh-key-values ~/.ssh/id_rsa.pub \
        --zone "$i" \
        --no-wait
done

echo "Waiting for VMs to be created..."
az vm wait --created --ids $(az vm list -g "$RG" --query "[].id" -o tsv)

echo "=== VMs Created ==="
az vm list -g "$RG" --show-details --output table

# 2. VM cost analysis
#!/bin/bash
echo "=== Azure VM Cost Analysis ==="

for rg in $(az group list --query "[].name" -o tsv); do
    VMS=$(az vm list -g "$rg" --show-details --query "[].{name:name, size:hardwareProfile.vmSize, state:powerState}" -o json 2>/dev/null)
    
    if [ "$(echo "$VMS" | jq length)" -gt 0 ]; then
        echo ""
        echo "Resource Group: $rg"
        echo "$VMS" | jq -r '.[] | "  \(.name)\t\(.size)\t\(.state)"'
        
        # Count deallocated (not billing compute)
        RUNNING=$(echo "$VMS" | jq '[.[] | select(.state=="VM running")] | length')
        STOPPED=$(echo "$VMS" | jq '[.[] | select(.state!="VM running")] | length')
        echo "  Running: $RUNNING, Stopped: $STOPPED"
    fi
done

# 3. VM health check
#!/bin/bash
RG="${1:?Usage: $0 <resource-group>}"

echo "=== VM Health Check ==="
az vm list -g "$RG" --show-details -o json | jq -r '.[] | 
    "VM: \(.name)",
    "  Status: \(.powerState)",
    "  Size: \(.hardwareProfile.vmSize)",
    "  OS: \(.storageProfile.imageReference.offer // "custom")",
    "  Private IP: \(.privateIps)",
    "  Public IP: \(.publicIps // "none")",
    ""'`,
				},
				{
					Title: "Availability Sets, Zones, and Scale Sets",
					Content: `Azure provides multiple mechanisms for VM high availability: Availability Sets for rack-level protection, Availability Zones for datacenter-level, and Scale Sets for automatic scaling.

**Availability Sets:**
` + "```" + `
Availability Sets distribute VMs across fault domains and update domains.

Fault Domains (FD):
  - Physical server racks in a datacenter
  - Separate power and network
  - Up to 3 FDs per availability set
  
Update Domains (UD):
  - Logical groups for planned maintenance
  - Up to 20 UDs per availability set
  - Azure updates one UD at a time

  Example with 2 FDs and 5 UDs:
    FD 0:  VM1(UD0), VM3(UD2), VM5(UD4)
    FD 1:  VM2(UD1), VM4(UD3)
  
  SLA: 99.95% for 2+ VMs in an availability set

Create:
  az vm availability-set create \
    --resource-group myRG \
    --name myAvailSet \
    --platform-fault-domain-count 3 \
    --platform-update-domain-count 5
  
  az vm create \
    --resource-group myRG \
    --name myVM1 \
    --availability-set myAvailSet \
    --image Ubuntu2204 \
    --size Standard_D2s_v5
` + "```" + `

**Availability Zones:**
` + "```" + `
Availability Zones are physically separate datacenters within a region.

  Region (e.g., East US)
  ├── Zone 1  (Datacenter A)
  ├── Zone 2  (Datacenter B)
  └── Zone 3  (Datacenter C)
  
  Properties:
  - Independent power, cooling, networking
  - Connected by high-speed, low-latency network
  - Not all regions have zones
  
  SLA: 99.99% for VMs across 2+ zones

  Zonal deployment (pinned to specific zone):
    az vm create --zone 1 ...
  
  Zone-redundant (distributed automatically):
    Used by services like Standard LB, Zone-redundant storage

  Choosing between Availability Sets and Zones:
    Availability Sets:
      + Available in all regions
      + No cross-zone latency
      - Single datacenter
      
    Availability Zones:
      + Datacenter-level resilience
      + Higher SLA (99.99%)
      - Small cross-zone latency
      - Not all regions support zones
` + "```" + `

**Virtual Machine Scale Sets (VMSS):**
` + "```" + `
VMSS automatically creates and manages identical VMs.

Features:
  - Auto-scaling (metric-based or schedule)
  - Automatic OS updates
  - Integration with Load Balancer and Application Gateway
  - Up to 1,000 VMs per scale set
  - Supports Availability Zones (zone-spreading)

Create VMSS:
  az vmss create \
    --resource-group myRG \
    --name myScaleSet \
    --image Ubuntu2204 \
    --instance-count 3 \
    --vm-sku Standard_D2s_v5 \
    --admin-username azureuser \
    --ssh-key-values ~/.ssh/id_rsa.pub \
    --zones 1 2 3 \
    --load-balancer myLB \
    --upgrade-policy-mode Automatic

Auto-scale rules:
  # Scale out when CPU > 75% for 10 min
  az monitor autoscale create \
    --resource-group myRG \
    --resource myScaleSet \
    --resource-type Microsoft.Compute/virtualMachineScaleSets \
    --name autoscale-config \
    --min-count 2 \
    --max-count 10 \
    --count 3
  
  az monitor autoscale rule create \
    --resource-group myRG \
    --autoscale-name autoscale-config \
    --condition "Percentage CPU > 75 avg 10m" \
    --scale out 2
  
  # Scale in when CPU < 25% for 10 min
  az monitor autoscale rule create \
    --resource-group myRG \
    --autoscale-name autoscale-config \
    --condition "Percentage CPU < 25 avg 10m" \
    --scale in 1

  # Schedule-based scaling
  az monitor autoscale profile create \
    --resource-group myRG \
    --autoscale-name autoscale-config \
    --name business-hours \
    --min-count 5 --max-count 20 --count 10 \
    --recurrence week Mon Tue Wed Thu Fri \
    --start 08:00 --end 18:00 --timezone "Eastern Standard Time"

Custom Script Extension:
  az vmss extension set \
    --resource-group myRG \
    --vmss-name myScaleSet \
    --name customScript \
    --publisher Microsoft.Azure.Extensions \
    --settings '{
      "commandToExecute": "apt-get update && apt-get install -y nginx"
    }'

Terraform VMSS:
  resource "azurerm_linux_virtual_machine_scale_set" "example" {
    name                = "myScaleSet"
    resource_group_name = azurerm_resource_group.example.name
    location            = azurerm_resource_group.example.location
    sku                 = "Standard_D2s_v5"
    instances           = 3
    admin_username      = "azureuser"
    zones               = ["1", "2", "3"]
    
    admin_ssh_key {
      username   = "azureuser"
      public_key = file("~/.ssh/id_rsa.pub")
    }
    
    source_image_reference {
      publisher = "Canonical"
      offer     = "0001-com-ubuntu-server-jammy"
      sku       = "22_04-lts-gen2"
      version   = "latest"
    }
    
    os_disk {
      storage_account_type = "Premium_LRS"
      caching              = "ReadWrite"
    }
    
    network_interface {
      name    = "nic"
      primary = true
      
      ip_configuration {
        name                                   = "internal"
        primary                                = true
        subnet_id                              = azurerm_subnet.example.id
        load_balancer_backend_address_pool_ids = [azurerm_lb_backend_address_pool.example.id]
      }
    }
    
    automatic_os_upgrade_policy {
      disable_automatic_rollback  = false
      enable_automatic_os_upgrade = true
    }
    
    rolling_upgrade_policy {
      max_batch_instance_percent              = 20
      max_unhealthy_instance_percent          = 20
      max_unhealthy_upgraded_instance_percent = 5
      pause_time_between_batches              = "PT0S"
    }
  }
` + "```" + ``,
					CodeExamples: `# Azure Scale Set management

# 1. VMSS operations script
#!/bin/bash
RG="${1:?Usage: $0 <resource-group> <action> [args]}"
ACTION="${2:?Usage: $0 <resource-group> <action> [args]}"
VMSS="${3:-}"

case "$ACTION" in
    list)
        echo "=== Scale Sets in $RG ==="
        az vmss list -g "$RG" -o table
        ;;
    status)
        VMSS="${VMSS:?Usage: $0 $RG status <vmss-name>}"
        echo "=== VMSS Status: $VMSS ==="
        az vmss list-instances -g "$RG" --name "$VMSS" -o table
        echo ""
        echo "--- Autoscale Config ---"
        az monitor autoscale list -g "$RG" -o table 2>/dev/null
        ;;
    scale)
        VMSS="${VMSS:?Usage: $0 $RG scale <vmss-name> <count>}"
        COUNT="${4:?Usage: $0 $RG scale <vmss-name> <count>}"
        echo "Scaling $VMSS to $COUNT instances..."
        az vmss scale -g "$RG" --name "$VMSS" --new-capacity "$COUNT"
        echo "Done."
        ;;
    update-image)
        VMSS="${VMSS:?Usage: $0 $RG update-image <vmss-name>}"
        echo "Updating instances to latest model..."
        az vmss update-instances -g "$RG" --name "$VMSS" --instance-ids "*"
        echo "Done."
        ;;
    reimage)
        VMSS="${VMSS:?Usage: $0 $RG reimage <vmss-name>}"
        echo "Reimaging all instances..."
        az vmss reimage -g "$RG" --name "$VMSS"
        echo "Done."
        ;;
    *)
        echo "Actions: list, status, scale, update-image, reimage"
        ;;
esac

# 2. VMSS health monitoring
#!/bin/bash
echo "=== VMSS Health Report ==="

for rg in $(az group list --query "[].name" -o tsv); do
    VMSS_LIST=$(az vmss list -g "$rg" --query "[].name" -o tsv 2>/dev/null)
    
    for vmss in $VMSS_LIST; do
        echo ""
        echo "--- $vmss (RG: $rg) ---"
        
        INSTANCES=$(az vmss list-instances -g "$rg" --name "$vmss" \
            --query "[].{id:instanceId, state:provisioningState}" -o json 2>/dev/null)
        
        TOTAL=$(echo "$INSTANCES" | jq length)
        HEALTHY=$(echo "$INSTANCES" | jq '[.[] | select(.state=="Succeeded")] | length')
        
        echo "  Total: $TOTAL, Healthy: $HEALTHY"
        
        if [ "$HEALTHY" -lt "$TOTAL" ]; then
            echo "  WARNING: Some instances unhealthy!"
            echo "$INSTANCES" | jq -r '.[] | select(.state!="Succeeded") | "    Instance \(.id): \(.state)"'
        fi
    done
done`,
				},
			},
		},
	})
}
