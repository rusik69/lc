package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1267,
			Title:       "Azure Networking Advanced Topics",
			Description: "Deep dive into Azure networking including VPN Gateway, ExpressRoute, Azure Firewall, Private Link, and network security patterns.",
			Order:       67,
			Lessons: []problems.Lesson{
				{
					Title: "Azure Hybrid Networking and Connectivity",
					Content: `Azure provides multiple connectivity options for hybrid architectures connecting on-premises networks to Azure and Azure networks to each other.

**VPN Gateway:**
` + "```" + `
VPN Gateway types:
  Site-to-Site (S2S):
    - IPsec/IKE tunnel
    - On-premises to Azure VNet
    - Requires VPN device on-premises
    - Up to 10 Gbps (VpnGw5AZ)
  
  Point-to-Site (P2S):
    - Individual client to Azure VNet
    - OpenVPN, SSTP, IKEv2
    - Azure AD authentication
    - Certificate authentication
  
  VNet-to-VNet:
    - Connect Azure VNets
    - Cross-region, cross-subscription
    - IPsec/IKE tunnel

Create VPN Gateway:
  # Gateway subnet (must be named GatewaySubnet)
  az network vnet subnet create \
    --vnet-name myVNet -g myRG \
    --name GatewaySubnet \
    --address-prefix 10.0.255.0/27
  
  # Public IP
  az network public-ip create \
    --name vpn-pip -g myRG \
    --allocation-method Static --sku Standard
  
  # VPN Gateway (takes 30-45 minutes)
  az network vnet-gateway create \
    --name myVPNGW -g myRG \
    --vnet myVNet \
    --gateway-type Vpn \
    --vpn-type RouteBased \
    --sku VpnGw2AZ \
    --public-ip-addresses vpn-pip \
    --no-wait
  
  # Local network gateway (your on-premises)
  az network local-gateway create \
    --name onPremGW -g myRG \
    --gateway-ip-address 203.0.113.1 \
    --local-address-prefixes 192.168.0.0/16
  
  # Create S2S connection
  az network vpn-connection create \
    --name toOnPrem -g myRG \
    --vnet-gateway1 myVPNGW \
    --local-gateway2 onPremGW \
    --shared-key 'MySharedKey123!'

Gateway SKUs:
  VpnGw1:    650 Mbps,  30 S2S,  250 P2S
  VpnGw2:    1 Gbps,    30 S2S,  500 P2S
  VpnGw3:    1.25 Gbps, 30 S2S,  1000 P2S
  VpnGw4:    5 Gbps,    100 S2S, 5000 P2S
  VpnGw5:    10 Gbps,   100 S2S, 10000 P2S
  (AZ suffix = zone-redundant)

P2S configuration:
  # Azure AD auth
  az network vnet-gateway update \
    --name myVPNGW -g myRG \
    --client-protocol OpenVPN \
    --address-prefixes 172.16.0.0/24 \
    --aad-tenant "https://login.microsoftonline.com/<tenant-id>" \
    --aad-audience "41b23e61-6c1e-4545-b367-cd054e0ed4b4" \
    --aad-issuer "https://sts.windows.net/<tenant-id>/"
` + "```" + `

**ExpressRoute:**
` + "```" + `
Private dedicated connection to Azure.

Connection models:
  CloudExchange Co-location:
    - Layer 2 or Layer 3 at exchange provider
    - Example: Equinix, Megaport
  
  Point-to-Point Ethernet:
    - Direct fiber from your datacenter
  
  Any-to-Any (IPVPN):
    - WAN integration with Azure
    - SD-WAN connectivity

SKUs:
  Local:     Access to 1-2 Azure regions near peering
  Standard:  Access to all regions in same geopolitical area
  Premium:   Global access to all Azure regions

Create ExpressRoute:
  az network express-route create \
    --name myER -g myRG \
    --bandwidth 1000 --provider "Equinix" \
    --peering-location "Silicon Valley" \
    --sku-tier Premium --sku-family MeteredData
  
  # Peering configuration
  az network express-route peering create \
    --circuit-name myER -g myRG \
    --peering-type AzurePrivatePeering \
    --peer-asn 65100 \
    --primary-peer-subnet 10.0.0.0/30 \
    --secondary-peer-subnet 10.0.0.4/30 \
    --vlan-id 200
  
  # Link to VNet via ExpressRoute Gateway
  az network vnet-gateway create \
    --name myERGW -g myRG --vnet myVNet \
    --gateway-type ExpressRoute --sku Standard
  
  az network vpn-connection create \
    --name erConnection -g myRG \
    --vnet-gateway1 myERGW \
    --express-route-circuit2 "/subscriptions/.../expressRouteCircuits/myER" \
    --routing-weight 0

ExpressRoute features:
  Global Reach:      Connect on-premises sites through Azure backbone
  FastPath:          Bypass gateway for data path (Ultra/ErGw3AZ)
  Private peering:   Access Azure VNets
  Microsoft peering: Access Microsoft 365, Dynamics, Azure PaaS
  Direct:            100 Gbps connection to Microsoft edge

ExpressRoute + VPN (coexistence):
  - ExpressRoute as primary (private, high throughput)
  - VPN as failover (encrypted, internet-based)
  - Both connected to same VNet
` + "```" + `

**VNet Peering and Virtual WAN:**
` + "```" + `
VNet Peering:
  - Connect VNets within/across regions
  - Uses Azure backbone (private)
  - No gateway required
  - Low latency, high bandwidth
  - Non-transitive (A↔B, B↔C ≠ A↔C)
  
  az network vnet peering create \
    --name vnet1-to-vnet2 -g myRG \
    --vnet-name vnet1 \
    --remote-vnet "/subscriptions/.../virtualNetworks/vnet2" \
    --allow-vnet-access true \
    --allow-forwarded-traffic true
  
  # Must create peering from both sides
  az network vnet peering create \
    --name vnet2-to-vnet1 -g myRG2 \
    --vnet-name vnet2 \
    --remote-vnet "/subscriptions/.../virtualNetworks/vnet1" \
    --allow-vnet-access true

Hub-spoke topology:
  Spoke VNet 1 ↔ Hub VNet ↔ Spoke VNet 2
                    |
              [VPN/ER Gateway]
                    |
              On-Premises
  
  Hub VNet:
    - Shared services (firewall, DNS, AD)
    - VPN/ExpressRoute gateway
    - Azure Firewall or NVA
  
  Spoke VNets:
    - Workload isolation
    - Peered to hub
    - Route through hub for on-prem/internet

Azure Virtual WAN:
  - Microsoft-managed hub-spoke
  - Automated connectivity
  - Transit routing between all connections
  
  Components:
    Virtual WAN:  Top-level resource
    Hub:          Regional virtual hub
    Connections:  VPN, ExpressRoute, VNet, P2S
  
  az network vwan create \
    --name myVWAN -g myRG --type Standard
  
  az network vhub create \
    --name hub-eastus -g myRG \
    --vwan myVWAN \
    --address-prefix 10.10.0.0/24 \
    --location eastus \
    --sku Standard
  
  # Connect VNet to hub
  az network vhub connection create \
    --name vnet1-connection -g myRG \
    --vhub-name hub-eastus \
    --remote-vnet "/subscriptions/.../virtualNetworks/vnet1" \
    --internet-security true
  
  Routing:
    - Automatic route propagation
    - Custom route tables
    - Route policies (routing intent)
    - BGP support
` + "```" + ``,
					CodeExamples: `# Azure networking scripts

# 1. Hybrid connectivity status
#!/bin/bash
echo "=== Hybrid Connectivity Status ==="

# VPN Gateways
echo "--- VPN Gateways ---"
for gw in $(az network vnet-gateway list --query "[?gatewayType=='Vpn'].name" -o tsv 2>/dev/null); do
    RG=$(az network vnet-gateway list \
        --query "[?name=='$gw'].resourceGroup" -o tsv | head -1)
    
    SKU=$(az network vnet-gateway show -n "$gw" -g "$RG" \
        --query "sku.name" -o tsv 2>/dev/null)
    STATE=$(az network vnet-gateway show -n "$gw" -g "$RG" \
        --query "provisioningState" -o tsv 2>/dev/null)
    
    echo "  $gw ($SKU) - $STATE"
    
    # Connections
    az network vpn-connection list -g "$RG" \
        --query "[?contains(virtualNetworkGateway1.id, '$gw')].{
            name:name, status:connectionStatus, type:connectionType
        }" -o table 2>/dev/null
done

# ExpressRoute circuits
echo ""
echo "--- ExpressRoute Circuits ---"
for circuit in $(az network express-route list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az network express-route list \
        --query "[?name=='$circuit'].resourceGroup" -o tsv | head -1)
    
    echo "  Circuit: $circuit ($RG)"
    az network express-route show -n "$circuit" -g "$RG" \
        --query "{
            provider:serviceProviderProperties.serviceProviderName,
            location:serviceProviderProperties.peeringLocation,
            bandwidth:serviceProviderProperties.bandwidthInMbps,
            state:circuitProvisioningState,
            serviceProviderState:serviceProviderProvisioningState
        }" -o json 2>/dev/null | jq .
done

# VNet peerings
echo ""
echo "--- VNet Peerings ---"
for vnet in $(az network vnet list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az network vnet list --query "[?name=='$vnet'].resourceGroup" -o tsv | head -1)
    
    PEERINGS=$(az network vnet peering list -g "$RG" --vnet-name "$vnet" \
        --query "[].{remote:remoteVirtualNetwork.id | split('/') | [-1], state:peeringState}" \
        -o json 2>/dev/null)
    
    if [ "$PEERINGS" != "[]" ] && [ -n "$PEERINGS" ]; then
        echo "  $vnet ($RG):"
        echo "$PEERINGS" | jq -r '.[] | "    → \(.remote): \(.state)"' 2>/dev/null
    fi
done

# 2. Network connectivity test
#!/bin/bash
echo "=== Network Connectivity Test ==="

# Check VPN connection health
for conn in $(az network vpn-connection list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az network vpn-connection list \
        --query "[?name=='$conn'].resourceGroup" -o tsv | head -1)
    
    echo "Connection: $conn ($RG)"
    az network vpn-connection show -n "$conn" -g "$RG" \
        --query "{
            status:connectionStatus,
            inBytes:ingressBytesTransferred,
            outBytes:egressBytesTransferred,
            protocol:connectionProtocol,
            routingWeight:routingWeight
        }" -o json 2>/dev/null | jq .
    
    # IKE SA info
    echo "  Tunnel status:"
    az network vpn-connection show -n "$conn" -g "$RG" \
        --query "tunnelConnectionStatus[].{
            tunnel:tunnel, status:connectionStatus,
            inBytes:ingressBytesTransferred, outBytes:egressBytesTransferred
        }" -o table 2>/dev/null
done

# 3. Virtual WAN topology
#!/bin/bash
echo "=== Virtual WAN Topology ==="

for vwan in $(az network vwan list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az network vwan list --query "[?name=='$vwan'].resourceGroup" -o tsv | head -1)
    echo "Virtual WAN: $vwan ($RG)"
    
    # Hubs
    for hub in $(az network vhub list --query "[?virtualWan.id | contains(@, '$vwan')].name" -o tsv 2>/dev/null); do
        HUB_RG=$(az network vhub list --query "[?name=='$hub'].resourceGroup" -o tsv | head -1)
        LOCATION=$(az network vhub show -n "$hub" -g "$HUB_RG" --query "location" -o tsv 2>/dev/null)
        echo "  Hub: $hub ($LOCATION)"
        
        # Connected VNets
        az network vhub connection list -g "$HUB_RG" --vhub-name "$hub" \
            --query "[].{name:name, vnet:remoteVirtualNetwork.id | split('/') | [-1], status:provisioningState}" \
            -o table 2>/dev/null
    done
done`,
				},
				{
					Title: "Azure Firewall and Network Security",
					Content: `Azure Firewall and network security services provide layered protection for Azure workloads.

**Azure Firewall:**
` + "```" + `
Azure Firewall:
  - Cloud-native managed firewall
  - Built-in high availability
  - Unrestricted scalability
  - FQDN filtering
  - Threat intelligence
  - TLS inspection (Premium)
  
  SKUs:
    Standard:
      - Network/application rules
      - FQDN filtering
      - Threat intelligence
      - NAT rules (DNAT)
    
    Premium:
      - All Standard features
      - TLS inspection
      - IDPS (intrusion detection)
      - URL filtering
      - Web categories
    
    Basic:
      - For SMB workloads
      - Fixed scale (2 instances)
      - No threat intelligence

Create Azure Firewall:
  # Firewall subnet (must be named AzureFirewallSubnet)
  az network vnet subnet create \
    --vnet-name hubVNet -g myRG \
    --name AzureFirewallSubnet \
    --address-prefix 10.0.1.0/26
  
  # Public IP
  az network public-ip create \
    --name fw-pip -g myRG --sku Standard
  
  # Create firewall
  az network firewall create \
    --name myFirewall -g myRG \
    --location eastus \
    --sku AZFW_VNet --tier Premium
  
  # IP configuration
  az network firewall ip-config create \
    --firewall-name myFirewall -g myRG \
    --name fw-config \
    --public-ip-address fw-pip \
    --vnet-name hubVNet
  
  # Get private IP (for routing)
  FW_PRIVATE_IP=$(az network firewall show \
    --name myFirewall -g myRG \
    --query "ipConfigurations[0].privateIPAddress" -o tsv)

Firewall rules:
  # Network rules (L3/L4)
  az network firewall network-rule create \
    --firewall-name myFirewall -g myRG \
    --collection-name allow-dns \
    --priority 200 --action Allow \
    --name dns \
    --protocols UDP \
    --source-addresses "10.0.0.0/8" \
    --destination-addresses "168.63.129.16" \
    --destination-ports 53
  
  # Application rules (L7)
  az network firewall application-rule create \
    --firewall-name myFirewall -g myRG \
    --collection-name allow-web \
    --priority 300 --action Allow \
    --name allow-microsoft \
    --protocols Https=443 \
    --source-addresses "10.0.0.0/8" \
    --fqdn-tags "WindowsUpdate" "AzureBackup"
  
  az network firewall application-rule create \
    --firewall-name myFirewall -g myRG \
    --collection-name allow-web \
    --priority 300 --action Allow \
    --name allow-github \
    --protocols Https=443 \
    --source-addresses "10.0.0.0/8" \
    --target-fqdns "*.github.com" "github.com"
  
  # NAT rules (DNAT)
  az network firewall nat-rule create \
    --firewall-name myFirewall -g myRG \
    --collection-name inbound-nat \
    --priority 100 --action Dnat \
    --name rdp-to-vm \
    --protocols TCP \
    --source-addresses "*" \
    --destination-addresses "$FW_PUBLIC_IP" \
    --destination-ports 3389 \
    --translated-address 10.0.2.4 \
    --translated-port 3389

Firewall Policy:
  # Centralized policy management
  az network firewall policy create \
    --name myPolicy -g myRG \
    --sku Premium \
    --threat-intel-mode Alert \
    --idps-mode Alert
  
  # Rule collection groups
  az network firewall policy rule-collection-group create \
    --policy-name myPolicy -g myRG \
    --name DefaultNetworkRuleGroup --priority 200
  
  # Associate with firewall
  az network firewall update \
    --name myFirewall -g myRG \
    --firewall-policy "/subscriptions/.../firewallPolicies/myPolicy"

Route traffic through firewall:
  # UDR (User Defined Route)
  az network route-table create \
    --name spoke-rt -g myRG
  
  az network route-table route create \
    --route-table-name spoke-rt -g myRG \
    --name to-internet \
    --address-prefix 0.0.0.0/0 \
    --next-hop-type VirtualAppliance \
    --next-hop-ip-address "$FW_PRIVATE_IP"
  
  az network vnet subnet update \
    --vnet-name spokeVNet -g myRG \
    --name workload-subnet \
    --route-table spoke-rt
` + "```" + `

**Private Link and Private Endpoints:**
` + "```" + `
Azure Private Link:
  - Access Azure PaaS over private IP
  - Traffic stays on Microsoft backbone
  - No public internet exposure
  - Supports 50+ Azure services

Create Private Endpoint:
  # For Storage Account
  az network private-endpoint create \
    --name storage-pe -g myRG \
    --vnet-name myVNet --subnet private-endpoints \
    --private-connection-resource-id $(az storage account show -n mystorageacct -g myRG --query id -o tsv) \
    --group-id blob \
    --connection-name storage-connection
  
  # Private DNS Zone
  az network private-dns zone create \
    --name "privatelink.blob.core.windows.net" -g myRG
  
  az network private-dns link vnet create \
    --zone-name "privatelink.blob.core.windows.net" -g myRG \
    --name storage-dns-link \
    --virtual-network myVNet \
    --registration-enabled false
  
  # DNS zone group (auto-register DNS)
  az network private-endpoint dns-zone-group create \
    --endpoint-name storage-pe -g myRG \
    --name storage-zone-group \
    --private-dns-zone "privatelink.blob.core.windows.net" \
    --zone-name blob

Private DNS zones for common services:
  Storage Blob:    privatelink.blob.core.windows.net
  Storage File:    privatelink.file.core.windows.net
  SQL Database:    privatelink.database.windows.net
  Cosmos DB:       privatelink.documents.azure.com
  Key Vault:       privatelink.vaultcore.azure.net
  ACR:             privatelink.azurecr.io
  Event Hub:       privatelink.servicebus.windows.net
  App Service:     privatelink.azurewebsites.net

Azure DDoS Protection:
  Standard plan:
    - Automatic attack mitigation
    - Adaptive tuning
    - Attack analytics
    - Cost protection (scale-out credits)
    - 100 public IPs included
  
  az network ddos-protection create \
    --name myDDoS -g myRG
  
  az network vnet update \
    --name myVNet -g myRG \
    --ddos-protection-plan myDDoS \
    --ddos-protection true

Web Application Firewall (WAF):
  - OWASP Core Rule Set
  - Custom rules
  - Bot protection
  - Rate limiting
  - Geo-filtering
  
  Available on:
    Application Gateway WAF v2
    Azure Front Door
    Azure CDN
` + "```" + ``,
					CodeExamples: `# Azure network security scripts

# 1. Firewall rules audit
#!/bin/bash
echo "=== Azure Firewall Audit ==="

for fw in $(az network firewall list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az network firewall list --query "[?name=='$fw'].resourceGroup" -o tsv | head -1)
    echo "Firewall: $fw ($RG)"
    
    SKU=$(az network firewall show -n "$fw" -g "$RG" --query "sku.tier" -o tsv 2>/dev/null)
    STATE=$(az network firewall show -n "$fw" -g "$RG" --query "provisioningState" -o tsv 2>/dev/null)
    echo "  SKU: $SKU, State: $STATE"
    
    # Network rules
    echo "  Network Rule Collections:"
    az network firewall network-rule collection list \
        --firewall-name "$fw" -g "$RG" \
        --query "[].{name:name, priority:priority, action:action.type, rules:rules | length(@)}" \
        -o table 2>/dev/null
    
    # Application rules
    echo "  Application Rule Collections:"
    az network firewall application-rule collection list \
        --firewall-name "$fw" -g "$RG" \
        --query "[].{name:name, priority:priority, action:action.type, rules:rules | length(@)}" \
        -o table 2>/dev/null
    
    # NAT rules
    echo "  NAT Rule Collections:"
    az network firewall nat-rule collection list \
        --firewall-name "$fw" -g "$RG" \
        --query "[].{name:name, priority:priority, rules:rules | length(@)}" \
        -o table 2>/dev/null
done

# 2. Private endpoint inventory
#!/bin/bash
echo "=== Private Endpoint Inventory ==="

for pe in $(az network private-endpoint list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az network private-endpoint list \
        --query "[?name=='$pe'].resourceGroup" -o tsv | head -1)
    
    TARGET=$(az network private-endpoint show -n "$pe" -g "$RG" \
        --query "privateLinkServiceConnections[0].privateLinkServiceId | split('/') | [-1]" \
        -o tsv 2>/dev/null)
    
    GROUP=$(az network private-endpoint show -n "$pe" -g "$RG" \
        --query "privateLinkServiceConnections[0].groupIds[0]" \
        -o tsv 2>/dev/null)
    
    STATUS=$(az network private-endpoint show -n "$pe" -g "$RG" \
        --query "privateLinkServiceConnections[0].privateLinkServiceConnectionState.status" \
        -o tsv 2>/dev/null)
    
    IP=$(az network private-endpoint show -n "$pe" -g "$RG" \
        --query "customDnsConfigs[0].ipAddresses[0]" -o tsv 2>/dev/null)
    
    echo "  $pe → $TARGET ($GROUP) - $STATUS [$IP]"
done

# Private DNS zones
echo ""
echo "--- Private DNS Zones ---"
az network private-dns zone list \
    --query "[].{zone:name, records:numberOfRecordSets, vnets:numberOfVirtualNetworkLinks}" \
    -o table 2>/dev/null

# 3. NSG effective rules check
#!/bin/bash
echo "=== NSG Analysis ==="

for nsg in $(az network nsg list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az network nsg list --query "[?name=='$nsg'].resourceGroup" -o tsv | head -1)
    echo "NSG: $nsg ($RG)"
    
    # Count rules
    INBOUND=$(az network nsg rule list --nsg-name "$nsg" -g "$RG" \
        --query "[?direction=='Inbound'] | length(@)" -o tsv 2>/dev/null)
    OUTBOUND=$(az network nsg rule list --nsg-name "$nsg" -g "$RG" \
        --query "[?direction=='Outbound'] | length(@)" -o tsv 2>/dev/null)
    echo "  Rules: $INBOUND inbound, $OUTBOUND outbound"
    
    # Check for overly permissive rules
    ALLOW_ALL=$(az network nsg rule list --nsg-name "$nsg" -g "$RG" \
        --query "[?access=='Allow' && sourceAddressPrefix=='*' && destinationPortRange=='*'].name" \
        -o tsv 2>/dev/null)
    
    if [ -n "$ALLOW_ALL" ]; then
        echo "  [WARNING] Overly permissive rules: $ALLOW_ALL"
    fi
    
    # Check for open management ports
    OPEN_MGMT=$(az network nsg rule list --nsg-name "$nsg" -g "$RG" \
        --query "[?access=='Allow' && sourceAddressPrefix=='*' && (destinationPortRange=='22' || destinationPortRange=='3389')].name" \
        -o tsv 2>/dev/null)
    
    if [ -n "$OPEN_MGMT" ]; then
        echo "  [WARNING] Open management ports: $OPEN_MGMT"
    fi
done`,
				},
			},
		},
	})
}
