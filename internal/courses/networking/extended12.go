package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2632,
			Title:       "Network Design Patterns and Enterprise Architecture",
			Description: "Learn campus network design, data center fabrics, WAN architecture, high availability patterns, and enterprise network planning.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "Enterprise Network Architecture and Design",
					Content: `Enterprise network design involves structured approaches to building reliable, scalable, and secure networks for organizations.

**Campus Network Design:**

Three-Tier Model (Cisco):
  Core Layer:
    High-speed backbone
    Fast packet switching
    No packet manipulation
    Redundant links
    Layer 3 routing
    
  Distribution Layer:
    Policy enforcement
    Inter-VLAN routing
    Access control (ACLs)
    QoS classification
    Route summarization
    Redundant uplinks to core
    
  Access Layer:
    End-user connectivity
    Port security
    802.1X authentication
    VLAN assignment
    PoE for phones/APs
    Spanning tree edge ports

Collapsed Core (Two-Tier):
  Combined core and distribution
  Suitable for small-medium campuses
  Lower cost, simpler management
  Access layer connects directly

Spine-Leaf (Modern Campus):
  Flat architecture
  Every leaf connects to every spine
  Equal-cost multipath (ECMP)
  Low latency
  Easy horizontal scaling

**Data Center Network Design:**

Traditional Three-Tier:
  Core -> Aggregation -> Access (ToR)
  Oversubscription at each tier
  STP domain management
  Limited scalability

Spine-Leaf (Clos Network):
  Every leaf switch connects to every spine
  No STP needed (Layer 3 fabric)
  ECMP for load balancing
  Predictable latency
  Easy to scale (add spines or leaves)
  
  Leaf Switches: Top-of-Rack (ToR)
    Connect servers
    48x 10/25GbE + 4-8x 100/400GbE uplinks
    
  Spine Switches:
    Connect all leaves
    32-64x 100/400GbE ports
    No server connections
    
  Design Rules:
    Servers connect to exactly 2 leaf switches (MLAG)
    Every leaf connects to every spine
    No leaf-to-leaf direct links
    No spine-to-spine direct links

BGP in Data Center (RFC 7938):
  eBGP between every switch pair
  Each switch has unique ASN
  Simple, well-understood protocol
  No need for IGP (OSPF/IS-IS)
  
  ASN Assignment:
    Spine 1: AS 65001
    Spine 2: AS 65002
    Leaf 1: AS 65101
    Leaf 2: AS 65102

VXLAN (Virtual Extensible LAN):
  Overlay network over IP fabric
  16 million segment IDs (vs 4096 VLANs)
  MAC-in-UDP encapsulation
  VTEP: VXLAN Tunnel Endpoint
  
  EVPN (Ethernet VPN):
    BGP-based VXLAN control plane
    MAC/IP advertisement
    ARP suppression
    Multi-tenancy
    Distributed anycast gateway

**WAN Architecture:**

Traditional WAN:
  MPLS (Multiprotocol Label Switching)
  Private, QoS-guaranteed
  Hub-and-spoke or full mesh
  Expensive per-Mbps
  
  MPLS Components:
    PE (Provider Edge): Customer-facing
    P (Provider): Core label switching
    CE (Customer Edge): Customer router
    Labels: Instead of IP lookup, use label
    VRF: Virtual Routing and Forwarding (per-customer)

SD-WAN (Software-Defined WAN):
  Overlay across any transport
  Internet, MPLS, LTE, 5G
  Centralized policy control
  Application-aware routing
  
  Components:
    vEdge/Edge: Branch router
    vSmart Controller: Policy engine
    vManage: Management plane
    vBond Orchestrator: Authentication
    
  Benefits:
    Transport independence
    Dynamic path selection
    Application-aware QoS
    Zero-touch provisioning
    Cost reduction (internet vs MPLS)
    Encryption by default

SASE (Secure Access Service Edge):
  SD-WAN + Security in cloud
  Combines networking and security
  
  Components:
    SD-WAN
    Firewall as a Service (FWaaS)
    Secure Web Gateway (SWG)
    Cloud Access Security Broker (CASB)
    Zero Trust Network Access (ZTNA)

**High Availability Patterns:**

Active-Active:
  Both nodes handle traffic
  Load balanced
  Higher throughput
  More complex state sync
  ECMP, anycast, MLAG

Active-Passive:
  Primary handles all traffic
  Standby takes over on failure
  Simpler configuration
  Wasted capacity in standby
  VRRP, HSRP, CARP

Redundancy Protocols:
  VRRP (Virtual Router Redundancy Protocol):
    Virtual IP shared between routers
    Priority-based election
    Preemption support
    Standard protocol (RFC 5798)
    
  HSRP (Hot Standby Router Protocol):
    Cisco proprietary
    Similar to VRRP
    Active/Standby terminology
    
  MLAG (Multi-Chassis Link Aggregation):
    LAG across two switches
    Appears as single switch to server
    Eliminates STP
    Vendor-specific: vPC (Cisco), MC-LAG

Link Aggregation (802.3ad LACP):
  Bundle multiple physical links
  Increased bandwidth
  Link redundancy
  Hash-based load distribution
    Source/Dest MAC
    Source/Dest IP
    Source/Dest Port

**Network Segmentation:**

VLANs:
  Layer 2 segmentation
  Up to 4094 VLANs
  802.1Q tagging
  Trunk ports carry multiple VLANs
  Access ports single VLAN

VRF (Virtual Routing and Forwarding):
  Layer 3 segmentation
  Separate routing tables per tenant
  VRF-lite: Without MPLS
  Used in multi-tenant environments

Zone-Based Firewall:
  Group interfaces into zones
  Policies between zones
  Default: deny between zones
  Allow within same zone

**IP Address Management (IPAM):**

Planning:
  Hierarchical allocation
  Summarization-friendly CIDR blocks
  Reserve space for growth
  Document all allocations
  
  /16 per region
  /20 per site/VPC
  /24 per subnet/VLAN
  
  Reserved ranges:
    Network address (first)
    Gateway (first usable or last usable)
    Broadcast (last)
    Cloud providers reserve additional (DNS, DHCP)

Dual Stack (IPv4 + IPv6):
  Run both protocols simultaneously
  Gradual migration to IPv6
  DNS returns both A and AAAA records
  Happy Eyeballs (RFC 8305): Try both, use fastest`,
					CodeExamples: `// Network design pattern implementations

package main

import (
    "fmt"
    "math/rand"
    "sort"
    "strings"
    "sync"
    "time"
)

// Spine-Leaf topology builder
type SpineLeafFabric struct {
    spines  []*FabricSwitch
    leaves  []*FabricSwitch
    links   []FabricLink
    mu      sync.RWMutex
}

type FabricSwitch struct {
    ID        string
    Role      string // spine, leaf
    ASN       uint32
    Loopback  string
    BGPPeers  []BGPPeer
    Ports     []SwitchPort
    VTEP      string // VXLAN tunnel endpoint
}

type SwitchPort struct {
    ID      string
    Speed   string // 10G, 25G, 100G, 400G
    Status  string
    Remote  string // Connected switch
}

type FabricLink struct {
    SrcSwitch string
    SrcPort   string
    DstSwitch string
    DstPort   string
    Speed     string
    Status    string
}

type BGPPeer struct {
    Address string
    RemoteASN uint32
    State     string // established, active, idle
    Uptime    time.Duration
}

func NewSpineLeafFabric(numSpines, numLeaves int) *SpineLeafFabric {
    fabric := &SpineLeafFabric{}
    
    // Create spine switches
    for i := 1; i <= numSpines; i++ {
        spine := &FabricSwitch{
            ID:       fmt.Sprintf("spine-%d", i),
            Role:     "spine",
            ASN:      65000 + uint32(i),
            Loopback: fmt.Sprintf("10.0.0.%d", i),
        }
        fabric.spines = append(fabric.spines, spine)
    }
    
    // Create leaf switches
    for i := 1; i <= numLeaves; i++ {
        leaf := &FabricSwitch{
            ID:       fmt.Sprintf("leaf-%d", i),
            Role:     "leaf",
            ASN:      65100 + uint32(i),
            Loopback: fmt.Sprintf("10.0.1.%d", i),
            VTEP:     fmt.Sprintf("10.0.2.%d", i),
        }
        fabric.leaves = append(fabric.leaves, leaf)
    }
    
    // Connect every leaf to every spine
    linkNum := 0
    for _, leaf := range fabric.leaves {
        for _, spine := range fabric.spines {
            linkNum++
            
            leafPort := SwitchPort{
                ID:     fmt.Sprintf("Eth1/%d", linkNum),
                Speed:  "100G",
                Status: "up",
                Remote: spine.ID,
            }
            spinePort := SwitchPort{
                ID:     fmt.Sprintf("Eth1/%d", linkNum),
                Speed:  "100G",
                Status: "up",
                Remote: leaf.ID,
            }
            
            leaf.Ports = append(leaf.Ports, leafPort)
            spine.Ports = append(spine.Ports, spinePort)
            
            // BGP peering
            leafPeerIP := fmt.Sprintf("10.%d.%d.1", linkNum/256, linkNum%256)
            spinePeerIP := fmt.Sprintf("10.%d.%d.2", linkNum/256, linkNum%256)
            
            leaf.BGPPeers = append(leaf.BGPPeers, BGPPeer{
                Address:   spinePeerIP,
                RemoteASN: spine.ASN,
                State:     "established",
            })
            spine.BGPPeers = append(spine.BGPPeers, BGPPeer{
                Address:   leafPeerIP,
                RemoteASN: leaf.ASN,
                State:     "established",
            })
            
            fabric.links = append(fabric.links, FabricLink{
                SrcSwitch: leaf.ID,
                SrcPort:   leafPort.ID,
                DstSwitch: spine.ID,
                DstPort:   spinePort.ID,
                Speed:     "100G",
                Status:    "up",
            })
        }
    }
    
    return fabric
}

func (f *SpineLeafFabric) GetPathCount(srcLeaf, dstLeaf string) int {
    if srcLeaf == dstLeaf {
        return 0 // Same switch
    }
    // Every pair of leaves has paths = number of spines (ECMP)
    return len(f.spines)
}

func (f *SpineLeafFabric) CalculateOversubscription(leaf *FabricSwitch, serverPorts, uplinkPorts int) float64 {
    // Oversubscription ratio
    // e.g., 48 x 25G server ports = 1200G
    //        6 x 100G uplinks = 600G
    // Oversubscription = 1200/600 = 2:1
    serverBW := float64(serverPorts) * 25.0  // Assume 25G server ports
    uplinkBW := float64(uplinkPorts) * 100.0 // Assume 100G uplinks
    
    if uplinkBW == 0 {
        return 0
    }
    return serverBW / uplinkBW
}

// VRRP implementation
type VRRPInstance struct {
    VirtualIP   string
    VirtualMAC  string
    VRID        int
    Priority    int
    State       string // master, backup, init
    Peers       []*VRRPPeer
    PreemptMode bool
    AdvInterval time.Duration
    mu          sync.Mutex
}

type VRRPPeer struct {
    Address   string
    Priority  int
    State     string
    LastSeen  time.Time
}

func NewVRRPInstance(vip string, vrid, priority int) *VRRPInstance {
    return &VRRPInstance{
        VirtualIP:   vip,
        VRID:        vrid,
        Priority:    priority,
        State:       "init",
        PreemptMode: true,
        AdvInterval: time.Second,
        VirtualMAC:  fmt.Sprintf("00:00:5e:00:01:%02x", vrid),
    }
}

func (v *VRRPInstance) Elect() {
    v.mu.Lock()
    defer v.mu.Unlock()
    
    highestPriority := v.Priority
    
    for _, peer := range v.Peers {
        // Consider only recently seen peers
        if time.Since(peer.LastSeen) > 3*v.AdvInterval {
            peer.State = "down"
            continue
        }
        if peer.Priority > highestPriority {
            highestPriority = peer.Priority
        }
    }
    
    if highestPriority == v.Priority {
        v.State = "master"
    } else if v.PreemptMode {
        v.State = "backup"
    }
}

// ECMP path selector
type ECMPRouter struct {
    routes   map[string][]NextHop
    mu       sync.RWMutex
}

type NextHop struct {
    Address  string
    Interface string
    Weight   int
    Active   bool
}

func NewECMPRouter() *ECMPRouter {
    return &ECMPRouter{
        routes: make(map[string][]NextHop),
    }
}

func (r *ECMPRouter) AddRoute(prefix string, hops []NextHop) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.routes[prefix] = hops
}

// Hash-based ECMP selection
func (r *ECMPRouter) SelectNextHop(prefix string, srcIP, dstIP string, srcPort, dstPort uint16) *NextHop {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    hops, exists := r.routes[prefix]
    if !exists {
        return nil
    }
    
    // Get active hops
    activeHops := make([]NextHop, 0)
    for _, hop := range hops {
        if hop.Active {
            activeHops = append(activeHops, hop)
        }
    }
    
    if len(activeHops) == 0 {
        return nil
    }
    
    // 5-tuple hash for consistent ECMP
    hashInput := fmt.Sprintf("%s:%s:%d:%d", srcIP, dstIP, srcPort, dstPort)
    hash := fnvHash(hashInput)
    
    if hasWeights(activeHops) {
        return weightedSelect(activeHops, hash)
    }
    
    idx := hash % uint32(len(activeHops))
    return &activeHops[idx]
}

func fnvHash(s string) uint32 {
    var hash uint32 = 2166136261
    for _, c := range s {
        hash ^= uint32(c)
        hash *= 16777619
    }
    return hash
}

func hasWeights(hops []NextHop) bool {
    for _, h := range hops {
        if h.Weight > 0 {
            return true
        }
    }
    return false
}

func weightedSelect(hops []NextHop, hash uint32) *NextHop {
    totalWeight := 0
    for _, h := range hops {
        totalWeight += h.Weight
    }
    
    target := int(hash) % totalWeight
    cumWeight := 0
    for i := range hops {
        cumWeight += hops[i].Weight
        if target < cumWeight {
            return &hops[i]
        }
    }
    return &hops[len(hops)-1]
}

// IPAM (IP Address Management)
type IPAM struct {
    blocks     map[string]*IPBlock
    allocations map[string]*IPAllocation
    mu          sync.RWMutex
}

type IPBlock struct {
    CIDR        string
    Description string
    Region      string
    Site        string
    Suballocations []*IPAllocation
    Available   bool
}

type IPAllocation struct {
    CIDR        string
    Type        string // vpc, subnet, host
    Description string
    AssignedTo  string
    CreatedAt   time.Time
}

func NewIPAM() *IPAM {
    return &IPAM{
        blocks:      make(map[string]*IPBlock),
        allocations: make(map[string]*IPAllocation),
    }
}

func (ipam *IPAM) AddBlock(cidr, description, region string) error {
    ipam.mu.Lock()
    defer ipam.mu.Unlock()
    
    // Check for overlaps
    for _, block := range ipam.blocks {
        if cidrsOverlapCheck(block.CIDR, cidr) {
            return fmt.Errorf("CIDR %s overlaps with existing block %s", cidr, block.CIDR)
        }
    }
    
    ipam.blocks[cidr] = &IPBlock{
        CIDR:        cidr,
        Description: description,
        Region:      region,
        Available:   true,
    }
    return nil
}

func (ipam *IPAM) Allocate(parentCIDR, cidr, allocType, description, assignee string) error {
    ipam.mu.Lock()
    defer ipam.mu.Unlock()
    
    block, exists := ipam.blocks[parentCIDR]
    if !exists {
        return fmt.Errorf("parent block %s not found", parentCIDR)
    }
    
    // Check overlap with existing allocations
    for _, alloc := range block.Suballocations {
        if cidrsOverlapCheck(alloc.CIDR, cidr) {
            return fmt.Errorf("CIDR %s overlaps with %s (%s)", cidr, alloc.CIDR, alloc.Description)
        }
    }
    
    allocation := &IPAllocation{
        CIDR:        cidr,
        Type:        allocType,
        Description: description,
        AssignedTo:  assignee,
        CreatedAt:   time.Now(),
    }
    
    block.Suballocations = append(block.Suballocations, allocation)
    ipam.allocations[cidr] = allocation
    return nil
}

func (ipam *IPAM) GetUtilization(parentCIDR string) (float64, error) {
    ipam.mu.RLock()
    defer ipam.mu.RUnlock()
    
    block, exists := ipam.blocks[parentCIDR]
    if !exists {
        return 0, fmt.Errorf("block %s not found", parentCIDR)
    }
    
    totalHosts := cidrHostCount(block.CIDR)
    allocatedHosts := 0
    for _, alloc := range block.Suballocations {
        allocatedHosts += cidrHostCount(alloc.CIDR)
    }
    
    if totalHosts == 0 {
        return 0, nil
    }
    return float64(allocatedHosts) / float64(totalHosts) * 100, nil
}

func cidrsOverlapCheck(a, b string) bool {
    // Simplified overlap check
    return strings.HasPrefix(a, strings.Split(b, "/")[0]) || 
           strings.HasPrefix(b, strings.Split(a, "/")[0])
}

func cidrHostCount(cidr string) int {
    parts := strings.Split(cidr, "/")
    if len(parts) != 2 {
        return 0
    }
    var prefix int
    fmt.Sscanf(parts[1], "%d", &prefix)
    if prefix >= 31 {
        return 2
    }
    return 1 << (32 - prefix)
}

// Network health monitor
type NetworkHealthMonitor struct {
    devices  map[string]*NetworkDevice
    checks   []HealthCheck
    alerts   []Alert
    mu       sync.RWMutex
}

type NetworkDevice struct {
    ID         string
    Type       string // router, switch, firewall
    IP         string
    Status     string // up, down, degraded
    LastCheck  time.Time
    Metrics    DeviceMetrics
}

type DeviceMetrics struct {
    CPU       float64
    Memory    float64
    Uptime    time.Duration
    InterfacesUp   int
    InterfacesDown int
    BGPPeersUp     int
    BGPPeersDown   int
    Errors    int64
}

type HealthCheck struct {
    DeviceID  string
    Type      string // ping, snmp, bgp, interface
    Status    string // pass, fail, warn
    Message   string
    Timestamp time.Time
}

type Alert struct {
    Severity  string // critical, warning, info
    DeviceID  string
    Message   string
    Timestamp time.Time
    Resolved  bool
}

func NewNetworkHealthMonitor() *NetworkHealthMonitor {
    return &NetworkHealthMonitor{
        devices: make(map[string]*NetworkDevice),
    }
}

func (m *NetworkHealthMonitor) AddDevice(device *NetworkDevice) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.devices[device.ID] = device
}

func (m *NetworkHealthMonitor) RecordCheck(check HealthCheck) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    m.checks = append(m.checks, check)
    
    if device, exists := m.devices[check.DeviceID]; exists {
        device.LastCheck = check.Timestamp
        
        if check.Status == "fail" {
            device.Status = "down"
            m.alerts = append(m.alerts, Alert{
                Severity:  "critical",
                DeviceID:  check.DeviceID,
                Message:   check.Message,
                Timestamp: check.Timestamp,
            })
        }
    }
}

func (m *NetworkHealthMonitor) GetOverallHealth() string {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    total := len(m.devices)
    down := 0
    degraded := 0
    
    for _, device := range m.devices {
        switch device.Status {
        case "down":
            down++
        case "degraded":
            degraded++
        }
    }
    
    if down > 0 {
        return fmt.Sprintf("Critical: %d/%d devices down", down, total)
    }
    if degraded > 0 {
        return fmt.Sprintf("Warning: %d/%d devices degraded", degraded, total)
    }
    return fmt.Sprintf("Healthy: %d/%d devices up", total, total)
}

func (m *NetworkHealthMonitor) GetActiveAlerts() []Alert {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    var active []Alert
    for _, alert := range m.alerts {
        if !alert.Resolved {
            active = append(active, alert)
        }
    }
    
    sort.Slice(active, func(i, j int) bool {
        return active[i].Timestamp.After(active[j].Timestamp)
    })
    
    return active
}`,
				},
			},
		},
	})
}
