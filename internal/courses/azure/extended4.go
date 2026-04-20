package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1241,
			Title:       "Azure Networking Deep Dive",
			Description: "Master Azure networking including Virtual Networks, subnets, Network Security Groups, Azure Firewall, VPN Gateway, ExpressRoute, and traffic management.",
			Order:       41,
			Lessons: []problems.Lesson{
				{
					Title: "Virtual Networks and Network Security",
					Content: `Azure Virtual Network (VNet) is the fundamental building block for private networks in Azure. Understanding networking is critical for secure, performant architectures.

**VNet Architecture:**
` + "```" + `
VNet concepts:
  Address space:   CIDR block (e.g., 10.0.0.0/16)
  Subnets:         Subdivisions of VNet address space
  Region-bound:    VNets exist in a single region
  Subscription:    VNets belong to one subscription
  
  Typical architecture:
  VNet: 10.0.0.0/16
  ├── Web Subnet:      10.0.1.0/24  (public-facing)
  ├── App Subnet:      10.0.2.0/24  (application tier)
  ├── DB Subnet:       10.0.3.0/24  (database tier)
  ├── Gateway Subnet:  10.0.255.0/27 (VPN/ExpressRoute)
  └── AzureFirewall:   10.0.254.0/26 (firewall subnet)

Reserved IPs per subnet:
  .0    Network address
  .1    Default gateway
  .2    Azure DNS mapping
  .3    Azure DNS mapping
  .255  Broadcast
  Total: 5 reserved per subnet

Create VNet:
  az network vnet create \
    --resource-group myRG \
    --name myVNet \
    --address-prefix 10.0.0.0/16 \
    --subnet-name web-subnet \
    --subnet-prefix 10.0.1.0/24

  az network vnet subnet create \
    --resource-group myRG \
    --vnet-name myVNet \
    --name app-subnet \
    --address-prefix 10.0.2.0/24
  
  az network vnet subnet create \
    --resource-group myRG \
    --vnet-name myVNet \
    --name db-subnet \
    --address-prefix 10.0.3.0/24
` + "```" + `

**Network Security Groups (NSG):**
` + "```" + `
NSGs filter traffic with allow/deny rules.

Rule properties:
  Priority:     100-4096 (lower = higher priority)
  Direction:    Inbound or Outbound
  Action:       Allow or Deny
  Protocol:     TCP, UDP, ICMP, or *
  Source/Dest:  IP, CIDR, Service Tag, or ASG

Default rules (cannot be deleted):
  Inbound:
    65000: Allow VNet to VNet
    65001: Allow Azure Load Balancer
    65500: Deny all
  
  Outbound:
    65000: Allow VNet to VNet
    65001: Allow to Internet
    65500: Deny all

Service Tags (predefined groups):
  Internet           All public IPs
  VirtualNetwork     VNet + peered + on-premises
  AzureLoadBalancer  Azure LB health probes
  Storage            Azure Storage IPs
  Sql                Azure SQL IPs
  AzureMonitor       Monitoring endpoints
  ApiManagement      APIM management
  
Create NSG with rules:
  az network nsg create --resource-group myRG --name web-nsg
  
  # Allow HTTP/HTTPS from internet
  az network nsg rule create \
    --resource-group myRG --nsg-name web-nsg \
    --name AllowHTTP \
    --priority 100 --direction Inbound \
    --access Allow --protocol Tcp \
    --source-address-prefixes "*" \
    --destination-port-ranges 80 443
  
  # Allow SSH from specific IP
  az network nsg rule create \
    --resource-group myRG --nsg-name web-nsg \
    --name AllowSSH \
    --priority 110 --direction Inbound \
    --access Allow --protocol Tcp \
    --source-address-prefixes "203.0.113.0/24" \
    --destination-port-ranges 22
  
  # Associate with subnet
  az network vnet subnet update \
    --resource-group myRG --vnet-name myVNet \
    --name web-subnet \
    --network-security-group web-nsg

Application Security Groups (ASG):
  # Group VMs logically instead of by IP
  az network asg create --resource-group myRG --name web-servers
  az network asg create --resource-group myRG --name db-servers
  
  # NSG rule using ASGs
  az network nsg rule create \
    --resource-group myRG --nsg-name app-nsg \
    --name AllowWebToDb \
    --priority 100 --direction Inbound \
    --access Allow --protocol Tcp \
    --source-asgs web-servers \
    --destination-asgs db-servers \
    --destination-port-ranges 5432
` + "```" + `

**VNet Peering:**
` + "```" + `
Connect VNets with low-latency, high-bandwidth private connection.

Types:
  VNet Peering:         Same region
  Global VNet Peering:  Cross-region

Properties:
  - Non-transitive (A↔B and B↔C doesn't mean A↔C)
  - No overlapping address spaces
  - Traffic stays on Microsoft backbone
  - Can peer across subscriptions and tenants
  
Create peering:
  # VNet A → VNet B
  az network vnet peering create \
    --resource-group rgA --name AtoB \
    --vnet-name vnetA \
    --remote-vnet /subscriptions/.../vnetB \
    --allow-vnet-access
  
  # VNet B → VNet A (must create both sides)
  az network vnet peering create \
    --resource-group rgB --name BtoA \
    --vnet-name vnetB \
    --remote-vnet /subscriptions/.../vnetA \
    --allow-vnet-access

Hub-spoke topology:
  Hub VNet:
    - Shared services (firewall, VPN gateway)
    - Peered with all spoke VNets
    
  Spoke VNets:
    - Application workloads
    - Peered with hub only
    - Route traffic through hub firewall
    
  # Enable gateway transit in hub
  az network vnet peering update \
    --resource-group hubRG --name hubToSpoke \
    --vnet-name hubVNet \
    --set allowGatewayTransit=true
  
  # Use remote gateway in spoke
  az network vnet peering update \
    --resource-group spokeRG --name spokeToHub \
    --vnet-name spokeVNet \
    --set useRemoteGateways=true
` + "```" + ``,
					CodeExamples: `# Azure networking scripts

# 1. Network security audit
#!/bin/bash
echo "=== Azure Network Security Audit ==="

for rg in $(az group list --query "[].name" -o tsv); do
    NSGS=$(az network nsg list -g "$rg" --query "[].name" -o tsv 2>/dev/null)
    
    for nsg in $NSGS; do
        echo ""
        echo "--- NSG: $nsg (RG: $rg) ---"
        
        # Check for overly permissive rules
        az network nsg rule list -g "$rg" --nsg-name "$nsg" \
            --query "[?access=='Allow' && direction=='Inbound'].{
                name:name, priority:priority,
                srcAddr:sourceAddressPrefix, destPort:destinationPortRange
            }" -o table 2>/dev/null
        
        # Flag: allow all from any
        WIDE_OPEN=$(az network nsg rule list -g "$rg" --nsg-name "$nsg" \
            --query "[?access=='Allow' && direction=='Inbound' && sourceAddressPrefix=='*'].name" \
            -o tsv 2>/dev/null)
        
        if [ -n "$WIDE_OPEN" ]; then
            echo "  WARNING: Rules open to all sources: $WIDE_OPEN"
        fi
    done
done

# 2. VNet topology mapper
#!/bin/bash
echo "=== Azure VNet Topology ==="

for rg in $(az group list --query "[].name" -o tsv); do
    VNETS=$(az network vnet list -g "$rg" --query "[].name" -o tsv 2>/dev/null)
    
    for vnet in $VNETS; do
        echo ""
        echo "VNet: $vnet (RG: $rg)"
        
        # Address space
        ADDR=$(az network vnet show -g "$rg" -n "$vnet" \
            --query "addressSpace.addressPrefixes[]" -o tsv)
        echo "  Address: $ADDR"
        
        # Subnets
        echo "  Subnets:"
        az network vnet subnet list -g "$rg" --vnet-name "$vnet" \
            --query "[].{name:name, prefix:addressPrefix, nsg:networkSecurityGroup.id}" \
            -o json 2>/dev/null | jq -r '.[] | 
            "    \(.name): \(.prefix) NSG: \(.nsg // "none" | split("/") | last)"'
        
        # Peerings
        PEERINGS=$(az network vnet peering list -g "$rg" --vnet-name "$vnet" \
            --query "[].{name:name, state:peeringState, remote:remoteVirtualNetwork.id}" \
            -o json 2>/dev/null)
        
        if [ "$(echo "$PEERINGS" | jq length)" -gt 0 ]; then
            echo "  Peerings:"
            echo "$PEERINGS" | jq -r '.[] | 
                "    \(.name): \(.state) → \(.remote | split("/") | last)"'
        fi
    done
done

# 3. Network connectivity test
#!/bin/bash
RG="${1:?Usage: $0 <resource-group> <vm-name>}"
VM="${2:?Usage: $0 <resource-group> <vm-name>}"

echo "=== Network Connectivity Test for $VM ==="

# Get VM's NIC and NSG
NIC_ID=$(az vm show -g "$RG" -n "$VM" --query "networkProfile.networkInterfaces[0].id" -o tsv)
NIC_NAME=$(basename "$NIC_ID")

echo "NIC: $NIC_NAME"

# Effective security rules
echo ""
echo "--- Effective Security Rules ---"
az network nic list-effective-nsg -g "$RG" -n "$NIC_NAME" \
    --query "value[].effectiveSecurityRules[?direction=='Inbound'].{
        priority:priority, access:access, protocol:protocol,
        srcPrefix:sourceAddressPrefix, destPort:destinationPortRange
    }" -o table 2>/dev/null | head -20

# Effective routes
echo ""
echo "--- Effective Routes ---"
az network nic show-effective-route-table -g "$RG" -n "$NIC_NAME" \
    --query "value[].{source:source, prefix:addressPrefix[0], nextHop:nextHopType, nextHopIP:nextHopIpAddress[0]}" \
    -o table 2>/dev/null`,
				},
				{
					Title: "Azure Firewall, VPN, and ExpressRoute",
					Content: `Enterprise networking requires centralized security with Azure Firewall, hybrid connectivity through VPN Gateway, and dedicated connections via ExpressRoute.

**Azure Firewall:**
` + "```" + `
Azure Firewall is a managed, cloud-based network security service.

Tiers:
  Standard:  L3-L7 filtering, threat intelligence
  Premium:   + TLS inspection, IDPS, URL filtering, web categories

Features:
  - Stateful firewall as a service
  - Built-in high availability
  - Unrestricted cloud scalability
  - FQDN filtering (DNS-based)
  - Threat intelligence-based filtering
  - SNAT/DNAT support

Deploy Azure Firewall:
  # Create firewall subnet (must be named AzureFirewallSubnet)
  az network vnet subnet create \
    --resource-group myRG --vnet-name hubVNet \
    --name AzureFirewallSubnet \
    --address-prefix 10.0.254.0/26
  
  # Create public IP
  az network public-ip create \
    --resource-group myRG --name fw-pip \
    --sku Standard --allocation-method Static
  
  # Create firewall
  az network firewall create \
    --resource-group myRG --name myFirewall \
    --location eastus \
    --vnet-name hubVNet \
    --sku AZFW_VNet \
    --tier Standard
  
  # Configure IP
  az network firewall ip-config create \
    --resource-group myRG --firewall-name myFirewall \
    --name FW-config --public-ip-address fw-pip \
    --vnet-name hubVNet

Network rules:
  az network firewall network-rule create \
    --resource-group myRG --firewall-name myFirewall \
    --collection-name AllowDNS --priority 200 --action Allow \
    --name dns --protocols UDP \
    --source-addresses "10.0.0.0/16" \
    --destination-addresses "168.63.129.16" \
    --destination-ports 53

Application rules (FQDN):
  az network firewall application-rule create \
    --resource-group myRG --firewall-name myFirewall \
    --collection-name AllowWeb --priority 300 --action Allow \
    --name allow-github \
    --source-addresses "10.0.0.0/16" \
    --protocols Https=443 \
    --target-fqdns "github.com" "*.github.com"

DNAT rules:
  az network firewall nat-rule create \
    --resource-group myRG --firewall-name myFirewall \
    --collection-name InboundDNAT --priority 100 --action Dnat \
    --name ssh-to-web \
    --source-addresses "*" --protocols TCP \
    --destination-addresses "fw-public-ip" \
    --destination-ports 2222 \
    --translated-address 10.0.1.10 \
    --translated-port 22

Route traffic through firewall:
  FW_PRIVATE_IP=$(az network firewall show -g myRG -n myFirewall \
    --query "ipConfigurations[0].privateIPAddress" -o tsv)
  
  az network route-table create -g myRG --name spoke-rt
  az network route-table route create \
    -g myRG --route-table-name spoke-rt \
    --name to-internet --address-prefix 0.0.0.0/0 \
    --next-hop-type VirtualAppliance \
    --next-hop-ip-address "$FW_PRIVATE_IP"
  
  az network vnet subnet update \
    -g myRG --vnet-name spokeVNet --name app-subnet \
    --route-table spoke-rt
` + "```" + `

**VPN Gateway:**
` + "```" + `
VPN Gateway connects on-premises networks to Azure over encrypted tunnels.

Types:
  Site-to-Site (S2S):     On-premises ↔ Azure (IPsec/IKE)
  Point-to-Site (P2S):    Individual client ↔ Azure
  VNet-to-VNet:           Azure VNet ↔ Azure VNet

SKUs:
  VpnGw1:   650 Mbps,  30 S2S tunnels,  250 P2S
  VpnGw2:   1 Gbps,    30 S2S tunnels,  500 P2S
  VpnGw3:   1.25 Gbps, 30 S2S tunnels,  1000 P2S
  VpnGw4:   5 Gbps,    100 S2S tunnels, 5000 P2S
  VpnGw5:   10 Gbps,   100 S2S tunnels, 10000 P2S
  (AZ variants for zone redundancy)

Site-to-Site VPN:
  # Create gateway subnet
  az network vnet subnet create \
    --resource-group myRG --vnet-name hubVNet \
    --name GatewaySubnet \
    --address-prefix 10.0.255.0/27
  
  # Create public IP for gateway
  az network public-ip create \
    --resource-group myRG --name vpn-gw-pip \
    --sku Standard --allocation-method Static
  
  # Create VPN gateway (~30 minutes)
  az network vnet-gateway create \
    --resource-group myRG --name myVpnGw \
    --vnet hubVNet \
    --gateway-type Vpn \
    --vpn-type RouteBased \
    --sku VpnGw2 \
    --public-ip-addresses vpn-gw-pip
  
  # Create local network gateway (on-premises)
  az network local-gateway create \
    --resource-group myRG --name onprem-gw \
    --gateway-ip-address 203.0.113.1 \
    --local-address-prefixes 192.168.0.0/16
  
  # Create connection
  az network vpn-connection create \
    --resource-group myRG --name s2s-connection \
    --vnet-gateway1 myVpnGw \
    --local-gateway2 onprem-gw \
    --shared-key "MySecureSharedKey123!"
` + "```" + `

**ExpressRoute:**
` + "```" + `
ExpressRoute provides dedicated, private connectivity to Azure.

Characteristics:
  - Private connection (not over public internet)
  - Provided by connectivity partner (Equinix, AT&T, etc.)
  - Higher reliability, lower latency
  - Bandwidth: 50 Mbps to 100 Gbps
  - SLA: 99.95%

Peering types:
  Azure Private Peering:   VNets (IaaS, PaaS with VNet integration)
  Microsoft Peering:       Microsoft 365, Dynamics 365, Azure PaaS

SKUs:
  Local:     Connect to nearby Azure region only
  Standard:  Connect to all Azure regions in same geopolitical area
  Premium:   Connect to all Azure regions globally

Create circuit:
  az network express-route create \
    --resource-group myRG \
    --name myExpressRoute \
    --bandwidth 1000 \
    --peering-location "Silicon Valley" \
    --provider "Equinix" \
    --sku-family MeteredData \
    --sku-tier Standard

ExpressRoute vs VPN:
  Feature         ExpressRoute        VPN Gateway
  Connection      Private             Public Internet
  Bandwidth       Up to 100 Gbps      Up to 10 Gbps
  Latency         Predictable         Variable
  Cost            Higher              Lower
  SLA             99.95%              99.95%
  Setup time      Weeks-months        Minutes-hours
  Encryption      Optional (MACsec)   Always (IPsec)
  
  Recommendation:
  - Production/mission-critical: ExpressRoute
  - Dev/test or backup: VPN
  - Best practice: ExpressRoute + VPN as failover
` + "```" + ``,
					CodeExamples: `# Azure networking management

# 1. Azure Firewall log analyzer
#!/bin/bash
RG="${1:?Usage: $0 <resource-group>}"
FW_NAME="${2:?Usage: $0 <resource-group> <firewall-name>}"

echo "=== Azure Firewall Status: $FW_NAME ==="

# Firewall status
STATE=$(az network firewall show -g "$RG" -n "$FW_NAME" \
    --query "provisioningState" -o tsv)
echo "Provisioning State: $STATE"

# Get rules summary
echo ""
echo "--- Network Rule Collections ---"
az network firewall network-rule collection list \
    -g "$RG" --firewall-name "$FW_NAME" \
    --query "[].{name:name, priority:priority, action:action.type, rules:rules|length(@)}" \
    -o table 2>/dev/null

echo ""
echo "--- Application Rule Collections ---"
az network firewall application-rule collection list \
    -g "$RG" --firewall-name "$FW_NAME" \
    --query "[].{name:name, priority:priority, action:action.type, rules:rules|length(@)}" \
    -o table 2>/dev/null

echo ""
echo "--- NAT Rule Collections ---"
az network firewall nat-rule collection list \
    -g "$RG" --firewall-name "$FW_NAME" \
    --query "[].{name:name, priority:priority, rules:rules|length(@)}" \
    -o table 2>/dev/null

# 2. VPN connection monitor
#!/bin/bash
echo "=== VPN Gateway Status ==="

for rg in $(az group list --query "[].name" -o tsv); do
    GWS=$(az network vnet-gateway list -g "$rg" --query "[].name" -o tsv 2>/dev/null)
    
    for gw in $GWS; do
        echo ""
        echo "--- Gateway: $gw (RG: $rg) ---"
        
        az network vnet-gateway show -g "$rg" -n "$gw" \
            --query "{sku:sku.name, type:gatewayType, vpnType:vpnType, active:activeActive}" \
            -o json 2>/dev/null | jq .
        
        # List connections
        echo "  Connections:"
        az network vpn-connection list -g "$rg" \
            --query "[?virtualNetworkGateway1.id.contains(@,'$gw')].{
                name:name, status:connectionStatus, type:connectionType
            }" -o table 2>/dev/null
    done
done

# 3. Hybrid network connectivity test
#!/bin/bash
echo "=== Hybrid Network Status ==="

# Check ExpressRoute circuits
echo "--- ExpressRoute Circuits ---"
az network express-route list --query "[].{
    name:name, rg:resourceGroup, bandwidth:bandwidthInMbps,
    status:circuitProvisioningState, provider:serviceProviderProperties.serviceProviderName
}" -o table 2>/dev/null

# Check VPN connections
echo ""
echo "--- VPN Connections ---"
for rg in $(az group list --query "[].name" -o tsv); do
    az network vpn-connection list -g "$rg" --query "[].{
        name:name, status:connectionStatus, type:connectionType,
        inBytes:ingressBytesTransferred, outBytes:egressBytesTransferred
    }" -o table 2>/dev/null
done`,
				},
			},
		},
	})
}
