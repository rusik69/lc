package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2621,
			Title:       "Network Address Translation and Routing",
			Description: "Understand NAT types, routing protocols, BGP fundamentals, CIDR subnetting, and network address planning for enterprise and cloud environments.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "NAT, Routing Protocols, and Subnetting",
					Content: `Network Address Translation and routing fundamentals are essential for designing and troubleshooting networks.

**NAT Types:**

SNAT (Source NAT):
  Translates source IP of outgoing packets
  Used for: Private network accessing internet
  Many-to-one: Multiple private IPs share one public IP
  
  Private Host (10.0.0.5) -> NAT Gateway -> Internet (203.0.113.1)
  
  Connection tracking:
    NAT gateway maintains translation table
    Maps (private IP:port) -> (public IP:port)
    Return traffic matched by reverse mapping

DNAT (Destination NAT):
  Translates destination IP of incoming packets
  Used for: Port forwarding, load balancing
  
  Internet (203.0.113.1:80) -> NAT -> Private Server (10.0.0.10:8080)

PAT (Port Address Translation):
  NAT overload - multiple hosts share one public IP
  Distinguished by port numbers
  Most common form of NAT
  
  10.0.0.5:12345 -> 203.0.113.1:40001 (mapped)
  10.0.0.6:54321 -> 203.0.113.1:40002 (mapped)

NAT Traversal:
  Problem: NAT breaks end-to-end connectivity
  Solutions:
    STUN: Discover public IP and port mapping
    TURN: Relay traffic through public server
    ICE: Try direct, then STUN, then TURN
    UPnP: Automatic port forwarding (security risk)
    NAT-PMP/PCP: Apple's port mapping protocol
  
  WebRTC uses ICE for peer-to-peer connections

Carrier-Grade NAT (CGNAT):
  ISP-level NAT (double NAT)
  ISP shares public IPs among customers
  100.64.0.0/10 reserved for CGNAT
  Complicates port forwarding and hosting

**Routing Fundamentals:**

Static Routing:
  Manually configured routes
  Simple, predictable
  No overhead
  Doesn't adapt to failures
  
  ip route add 10.1.0.0/16 via 192.168.1.1
  ip route add default via 192.168.1.1

Dynamic Routing Protocols:

  Distance Vector:
    RIP (Routing Information Protocol)
    Hop count as metric (max 15)
    Periodic full table exchange
    Slow convergence
    Historical, rarely used now

  Link State:
    OSPF (Open Shortest Path First)
    Each router knows full topology
    Dijkstra's algorithm for shortest path
    Fast convergence
    Areas for scalability
    Used within organizations (IGP)
    
    OSPF Areas:
      Area 0 (backbone): Connects all areas
      Other areas: Reduce routing table size
      ABRs (Area Border Routers): Connect areas

  Path Vector:
    BGP (Border Gateway Protocol)
    Routes between autonomous systems
    Policy-based routing
    Foundation of internet routing

**BGP (Border Gateway Protocol):**

  The routing protocol of the internet
  Connects autonomous systems (AS)
  
  Types:
    eBGP (External): Between different AS
    iBGP (Internal): Within same AS
  
  BGP Attributes:
    AS_PATH: List of AS numbers traversed
    NEXT_HOP: Next router IP address
    LOCAL_PREF: Preference for outbound traffic (higher = preferred)
    MED: Multi-Exit Discriminator (suggest entry point to neighbor)
    COMMUNITY: Tag for route policies
  
  BGP Path Selection (simplified):
    1. Highest LOCAL_PREF
    2. Shortest AS_PATH
    3. Lowest ORIGIN type
    4. Lowest MED
    5. eBGP over iBGP
    6. Nearest IGP neighbor
    7. Lowest router ID

  BGP Peering:
    Establish TCP connection (port 179)
    Exchange full routing table (initial)
    Incremental updates afterward
    Keepalive messages every 60 seconds
    Hold timer: 180 seconds (3 missed = peer down)

  BGP Security Issues:
    Route hijacking: Announce someone else's prefix
    Route leaks: Propagate routes incorrectly
    
    Mitigations:
      RPKI (Resource Public Key Infrastructure)
      ROA (Route Origin Authorization)
      IRR filtering (Internet Routing Registry)
      BGP peer authentication (TCP MD5)

  Anycast:
    Same IP announced from multiple locations
    Traffic goes to nearest announcement
    Used for: DNS root servers, CDNs, DDoS mitigation
    BGP naturally routes to nearest

**CIDR and Subnetting:**

CIDR Notation:
  IP/prefix_length
  10.0.0.0/8: 16,777,214 hosts
  172.16.0.0/12: 1,048,574 hosts
  192.168.0.0/16: 65,534 hosts
  10.1.0.0/24: 254 hosts
  10.1.0.0/28: 14 hosts

Subnet Calculation:
  /24: 256 addresses (254 usable)
  /25: 128 addresses (126 usable)
  /26: 64 addresses (62 usable)
  /27: 32 addresses (30 usable)
  /28: 16 addresses (14 usable)
  /29: 8 addresses (6 usable)
  /30: 4 addresses (2 usable, point-to-point)
  /31: 2 addresses (2 usable, RFC 3021)
  /32: 1 address (single host)

VLSM (Variable Length Subnet Masking):
  Different subnet sizes within same network
  Efficient IP address usage
  
  Example: 10.0.0.0/16 divided into:
    10.0.0.0/24: Server subnet (254 hosts)
    10.0.1.0/25: Engineering (126 hosts)
    10.0.1.128/26: Sales (62 hosts)
    10.0.1.192/28: Management (14 hosts)
    10.0.1.208/30: Point-to-point link (2 hosts)

Private Address Ranges (RFC 1918):
  10.0.0.0/8: Class A private
  172.16.0.0/12: Class B private
  192.168.0.0/16: Class C private

Special Addresses:
  127.0.0.0/8: Loopback
  169.254.0.0/16: Link-local (APIPA)
  224.0.0.0/4: Multicast
  0.0.0.0/0: Default route
  255.255.255.255: Broadcast

**Cloud VPC Networking:**

AWS VPC:
  VPC CIDR: /16 to /28
  Subnets: Public (internet gateway) and Private (NAT gateway)
  Route tables per subnet
  Security groups (stateful) + NACLs (stateless)
  VPC peering, Transit Gateway, PrivateLink

GCP VPC:
  Global VPC (spans all regions)
  Subnets are regional
  Auto-mode or custom subnets
  Firewall rules at VPC level
  Shared VPC for multi-project

Azure VNet:
  Regional scope
  Network Security Groups (NSGs)
  VNet peering (global)
  Azure Private Link
  Service endpoints`,
					CodeExamples: `// Network addressing and routing tools

package main

import (
    "encoding/binary"
    "fmt"
    "math"
    "math/bits"
    "net"
    "sort"
    "strings"
)

// Comprehensive subnet calculator
type SubnetCalculator struct{}

type SubnetDetails struct {
    CIDR         string
    NetworkAddr  string
    BroadcastAddr string
    FirstUsable  string
    LastUsable   string
    SubnetMask   string
    WildcardMask string
    TotalHosts   int
    UsableHosts  int
    PrefixLength int
    BinaryMask   string
    IPClass      string
    IsPrivate    bool
}

func (c *SubnetCalculator) Calculate(cidr string) (*SubnetDetails, error) {
    ip, network, err := net.ParseCIDR(cidr)
    if err != nil {
        return nil, fmt.Errorf("invalid CIDR: %w", err)
    }
    
    ones, totalBits := network.Mask.Size()
    hostBits := totalBits - ones
    totalHosts := int(math.Pow(2, float64(hostBits)))
    
    netIP := network.IP.To4()
    if netIP == nil {
        return nil, fmt.Errorf("IPv6 not supported")
    }
    
    // Broadcast
    broadcast := make(net.IP, 4)
    for i := range netIP {
        broadcast[i] = netIP[i] | ^network.Mask[i]
    }
    
    // First and last usable
    first := make(net.IP, 4)
    copy(first, netIP)
    last := make(net.IP, 4)
    copy(last, broadcast)
    
    usable := totalHosts - 2
    if ones >= 31 {
        first = netIP
        last = broadcast
        usable = totalHosts
    } else {
        incrementIP(first)
        decrementIP(last)
    }
    
    // Wildcard mask
    wildcard := make(net.IPMask, 4)
    for i := range network.Mask {
        wildcard[i] = ^network.Mask[i]
    }
    
    // Binary mask representation
    maskBinary := ""
    for _, b := range network.Mask {
        maskBinary += fmt.Sprintf("%08b.", b)
    }
    maskBinary = strings.TrimSuffix(maskBinary, ".")
    
    return &SubnetDetails{
        CIDR:          cidr,
        NetworkAddr:   netIP.String(),
        BroadcastAddr: broadcast.String(),
        FirstUsable:   first.String(),
        LastUsable:    last.String(),
        SubnetMask:    net.IP(network.Mask).String(),
        WildcardMask:  net.IP(wildcard).String(),
        TotalHosts:    totalHosts,
        UsableHosts:   usable,
        PrefixLength:  ones,
        BinaryMask:    maskBinary,
        IPClass:       classifyIP(ip),
        IsPrivate:     isPrivateIP(ip),
    }, nil
}

func incrementIP(ip net.IP) {
    for i := len(ip) - 1; i >= 0; i-- {
        ip[i]++
        if ip[i] != 0 {
            break
        }
    }
}

func decrementIP(ip net.IP) {
    for i := len(ip) - 1; i >= 0; i-- {
        ip[i]--
        if ip[i] != 255 {
            break
        }
    }
}

func classifyIP(ip net.IP) string {
    ip4 := ip.To4()
    if ip4 == nil {
        return "IPv6"
    }
    first := ip4[0]
    switch {
    case first < 128:
        return "A"
    case first < 192:
        return "B"
    case first < 224:
        return "C"
    case first < 240:
        return "D (Multicast)"
    default:
        return "E (Reserved)"
    }
}

func isPrivateIP(ip net.IP) bool {
    privateRanges := []string{
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
    }
    for _, cidr := range privateRanges {
        _, network, _ := net.ParseCIDR(cidr)
        if network.Contains(ip) {
            return true
        }
    }
    return false
}

// VLSM subnet planner
type VLSMPlanner struct{}

type SubnetRequest struct {
    Name        string
    HostsNeeded int
}

type SubnetAllocation struct {
    Name         string
    CIDR         string
    HostsNeeded  int
    HostsProvided int
    NetworkAddr  string
    BroadcastAddr string
}

func (p *VLSMPlanner) Plan(baseCIDR string, requests []SubnetRequest) ([]SubnetAllocation, error) {
    _, baseNet, err := net.ParseCIDR(baseCIDR)
    if err != nil {
        return nil, err
    }
    
    // Sort by hosts needed (largest first for efficient allocation)
    sorted := make([]SubnetRequest, len(requests))
    copy(sorted, requests)
    sort.Slice(sorted, func(i, j int) bool {
        return sorted[i].HostsNeeded > sorted[j].HostsNeeded
    })
    
    baseOnes, totalBits := baseNet.Mask.Size()
    currentIP := ipToUint32(baseNet.IP.To4())
    maxIP := currentIP + uint32(1<<(totalBits-baseOnes))
    
    var allocations []SubnetAllocation
    
    for _, req := range sorted {
        // Calculate required prefix length
        hostBits := int(math.Ceil(math.Log2(float64(req.HostsNeeded + 2))))
        if hostBits < 2 {
            hostBits = 2
        }
        prefixLen := totalBits - hostBits
        subnetSize := uint32(1 << hostBits)
        
        // Align to subnet boundary
        if currentIP%subnetSize != 0 {
            currentIP = ((currentIP / subnetSize) + 1) * subnetSize
        }
        
        if currentIP+subnetSize > maxIP {
            return nil, fmt.Errorf("not enough address space for %s (%d hosts)", req.Name, req.HostsNeeded)
        }
        
        netAddr := uint32ToIP(currentIP)
        bcastAddr := uint32ToIP(currentIP + subnetSize - 1)
        
        allocations = append(allocations, SubnetAllocation{
            Name:          req.Name,
            CIDR:          fmt.Sprintf("%s/%d", netAddr.String(), prefixLen),
            HostsNeeded:   req.HostsNeeded,
            HostsProvided: int(subnetSize) - 2,
            NetworkAddr:   netAddr.String(),
            BroadcastAddr: bcastAddr.String(),
        })
        
        currentIP += subnetSize
    }
    
    return allocations, nil
}

func ipToUint32(ip net.IP) uint32 {
    return binary.BigEndian.Uint32(ip.To4())
}

func uint32ToIP(n uint32) net.IP {
    ip := make(net.IP, 4)
    binary.BigEndian.PutUint32(ip, n)
    return ip
}

// Route table simulator
type RouteTable struct {
    routes []Route
}

type Route struct {
    Destination *net.IPNet
    Gateway     net.IP
    Interface   string
    Metric      int
    Protocol    string
}

func NewRouteTable() *RouteTable {
    return &RouteTable{}
}

func (rt *RouteTable) AddRoute(dest string, gateway string, iface string, metric int) error {
    _, network, err := net.ParseCIDR(dest)
    if err != nil {
        return err
    }
    
    rt.routes = append(rt.routes, Route{
        Destination: network,
        Gateway:     net.ParseIP(gateway),
        Interface:   iface,
        Metric:      metric,
    })
    
    return nil
}

func (rt *RouteTable) Lookup(destIP string) *Route {
    ip := net.ParseIP(destIP)
    if ip == nil {
        return nil
    }
    
    var bestMatch *Route
    bestPrefix := -1
    
    for i := range rt.routes {
        if rt.routes[i].Destination.Contains(ip) {
            ones, _ := rt.routes[i].Destination.Mask.Size()
            if ones > bestPrefix || (ones == bestPrefix && bestMatch != nil && rt.routes[i].Metric < bestMatch.Metric) {
                bestPrefix = ones
                bestMatch = &rt.routes[i]
            }
        }
    }
    
    return bestMatch
}

// IP address conflict detector
func DetectIPConflicts(subnets []string) []IPConflict {
    var conflicts []IPConflict
    
    for i := 0; i < len(subnets); i++ {
        _, netA, err := net.ParseCIDR(subnets[i])
        if err != nil {
            continue
        }
        
        for j := i + 1; j < len(subnets); j++ {
            _, netB, err := net.ParseCIDR(subnets[j])
            if err != nil {
                continue
            }
            
            if netA.Contains(netB.IP) || netB.Contains(netA.IP) {
                conflicts = append(conflicts, IPConflict{
                    SubnetA: subnets[i],
                    SubnetB: subnets[j],
                    Type:    "overlap",
                })
            }
        }
    }
    
    return conflicts
}

type IPConflict struct {
    SubnetA string
    SubnetB string
    Type    string
}`,
				},
			},
		},
	})
}
