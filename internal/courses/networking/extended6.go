package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2622,
			Title:       "Network Protocols Deep Dive",
			Description: "Deep dive into ARP, ICMP, DHCP, IPv6, multicast, and network protocol internals for troubleshooting and design.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Core Network Protocol Internals",
					Content: `Understanding protocol internals is essential for troubleshooting and designing efficient network systems.

**ARP (Address Resolution Protocol):**

  Resolves IP addresses to MAC addresses on local network
  
  ARP Request (broadcast):
    "Who has 192.168.1.10? Tell 192.168.1.5"
    Sent to FF:FF:FF:FF:FF:FF (all hosts on LAN)
  
  ARP Reply (unicast):
    "192.168.1.10 is at AA:BB:CC:DD:EE:FF"
    Sent to requesting host
  
  ARP Cache:
    Stores recent mappings
    Entries timeout (typically 20 minutes)
    View: arp -a / ip neigh show
  
  ARP Attacks:
    ARP Spoofing: Send fake ARP replies
    ARP Cache Poisoning: Redirect traffic (MITM)
    Gratuitous ARP: Announce IP-MAC mapping
    
  Defense:
    Static ARP entries for critical hosts
    Dynamic ARP Inspection (DAI) on switches
    ARP rate limiting
    802.1X port authentication

**ICMP (Internet Control Message Protocol):**

  Error reporting and diagnostic protocol
  Carried inside IP packets
  
  Types:
    Type 0: Echo Reply (ping response)
    Type 3: Destination Unreachable
      Code 0: Network unreachable
      Code 1: Host unreachable
      Code 3: Port unreachable
      Code 4: Fragmentation needed (Path MTU Discovery)
      Code 13: Admin prohibited (firewall)
    Type 8: Echo Request (ping)
    Type 11: Time Exceeded (traceroute uses this)
      Code 0: TTL exceeded in transit
    Type 12: Parameter Problem

  Path MTU Discovery:
    Send packets with DF (Don't Fragment) bit set
    If too large, router returns ICMP "Fragmentation Needed"
    Sender reduces packet size
    Finds maximum MTU along path
    
    Standard MTU: 1500 bytes (Ethernet)
    Jumbo frames: 9000 bytes
    Common tunnel overhead: 1500 - 50 = 1450 (VPN)

  ICMP Security:
    Should not block all ICMP (breaks PMTUD, traceroute)
    Rate-limit ICMP to prevent ICMP flood
    Block ICMP redirect messages
    Allow Type 3 (unreachable) and Type 11 (TTL exceeded)

**DHCP (Dynamic Host Configuration Protocol):**

  Automatic IP address assignment
  
  DORA Process:
    Discover: Client broadcasts "I need an IP" (0.0.0.0 -> 255.255.255.255)
    Offer: Server offers IP address
    Request: Client accepts offer (broadcasts to inform all servers)
    Acknowledge: Server confirms assignment
  
  DHCP Options:
    Option 1: Subnet Mask
    Option 3: Default Gateway
    Option 6: DNS Servers
    Option 51: Lease Time
    Option 53: DHCP Message Type
    Option 121: Classless Static Routes
    
  Lease Lifecycle:
    Assignment: Client gets IP for lease duration
    Renewal (T1): At 50% lease time, unicast to server
    Rebinding (T2): At 87.5% lease time, broadcast
    Expiration: If no renewal, release IP
    
  DHCP Relay:
    Forward DHCP across subnets
    Router acts as relay agent
    Sets giaddr field to identify client subnet
    Server uses giaddr to select correct pool

  DHCP Security:
    DHCP Snooping: Validates DHCP messages
    Only trusted ports can send DHCP offers
    Prevents rogue DHCP servers
    Rate limiting on DHCP requests

**IPv6:**

  128-bit addresses (vs 32-bit IPv4)
  340 undecillion addresses
  
  Address Format:
    2001:0db8:85a3:0000:0000:8a2e:0370:7334
    Shortened: 2001:db8:85a3::8a2e:370:7334
    
  Address Types:
    Global Unicast: 2000::/3 (public addresses)
    Link-Local: fe80::/10 (auto-configured, local link only)
    Unique Local: fc00::/7 (private, like RFC 1918)
    Multicast: ff00::/8
    Loopback: ::1
    
  Key Changes from IPv4:
    No NAT needed (enough addresses)
    No broadcast (multicast instead)
    No ARP (NDP - Neighbor Discovery Protocol instead)
    Mandatory IPSec support
    Simplified header (fixed 40 bytes)
    No fragmentation by routers (source only)
    Auto-configuration (SLAAC)
    
  SLAAC (Stateless Address Auto-Configuration):
    Host generates own address from:
      Network prefix (from Router Advertisement)
      Interface ID (from MAC using EUI-64, or random)
    No DHCP needed for basic connectivity
    DHCPv6 still used for DNS, domain, extra options

  NDP (Neighbor Discovery Protocol):
    Replaces ARP, ICMP Router Discovery
    Router Solicitation/Advertisement
    Neighbor Solicitation/Advertisement
    Redirect messages
    Uses ICMPv6

  Transition Mechanisms:
    Dual Stack: Run IPv4 and IPv6 simultaneously
    Tunneling: Encapsulate IPv6 in IPv4 (6to4, Teredo)
    NAT64/DNS64: Translate between IPv4 and IPv6

**Multicast:**

  One-to-many communication
  Source sends one copy, network duplicates as needed
  
  Addressing:
    IPv4: 224.0.0.0/4
    IPv6: ff00::/8
    
  Well-known groups:
    224.0.0.1: All hosts
    224.0.0.2: All routers
    224.0.0.5: OSPF routers
    224.0.0.251: mDNS
    
  IGMP (Internet Group Management Protocol):
    Hosts join/leave multicast groups
    Routers track group membership
    IGMP snooping on switches (prevents flooding)
    
  Use cases:
    Video streaming (IPTV)
    Software updates to many hosts
    Real-time data distribution (stock tickers)
    Cluster communication (keepalive)

**Network Troubleshooting Methodology:**

OSI Layer Approach (bottom-up):
  Layer 1 (Physical): Cable connected? Link light on?
  Layer 2 (Data Link): ARP resolving? MAC table entry?
  Layer 3 (Network): IP configured? Route exists? Firewall?
  Layer 4 (Transport): Port open? Connection established?
  Layer 7 (Application): DNS resolving? HTTP responding?

Tools per layer:
  L1: ethtool, cable tester
  L2: arp, bridge fdb, tcpdump
  L3: ping, traceroute, ip route
  L4: netstat/ss, telnet, nc
  L7: curl, dig, nslookup, openssl s_client`,
					CodeExamples: `// Network protocol implementations

package main

import (
    "context"
    "encoding/binary"
    "fmt"
    "net"
    "strings"
    "time"
)

// ICMP ping implementation
type Pinger struct {
    target    string
    count     int
    timeout   time.Duration
    interval  time.Duration
}

type PingResult struct {
    Target    string
    Sent      int
    Received  int
    Lost      int
    LossRate  float64
    MinRTT    time.Duration
    MaxRTT    time.Duration
    AvgRTT    time.Duration
    RTTs      []time.Duration
}

func NewPinger(target string, count int) *Pinger {
    return &Pinger{
        target:   target,
        count:    count,
        timeout:  3 * time.Second,
        interval: time.Second,
    }
}

func (p *Pinger) Ping() (*PingResult, error) {
    addr, err := net.ResolveIPAddr("ip4", p.target)
    if err != nil {
        return nil, fmt.Errorf("resolve failed: %w", err)
    }
    
    result := &PingResult{
        Target: addr.String(),
        Sent:   p.count,
    }
    
    for i := 0; i < p.count; i++ {
        rtt, err := p.sendPing(addr)
        if err != nil {
            result.Lost++
            continue
        }
        
        result.Received++
        result.RTTs = append(result.RTTs, rtt)
        
        if rtt < result.MinRTT || result.MinRTT == 0 {
            result.MinRTT = rtt
        }
        if rtt > result.MaxRTT {
            result.MaxRTT = rtt
        }
        
        if i < p.count-1 {
            time.Sleep(p.interval)
        }
    }
    
    if result.Received > 0 {
        var total time.Duration
        for _, rtt := range result.RTTs {
            total += rtt
        }
        result.AvgRTT = total / time.Duration(result.Received)
    }
    
    result.LossRate = float64(result.Lost) / float64(result.Sent) * 100
    
    return result, nil
}

func (p *Pinger) sendPing(addr *net.IPAddr) (time.Duration, error) {
    conn, err := net.DialTimeout("ip4:icmp", addr.String(), p.timeout)
    if err != nil {
        return 0, err
    }
    defer conn.Close()
    
    conn.SetDeadline(time.Now().Add(p.timeout))
    
    // Build ICMP echo request
    msg := buildICMPEchoRequest(0, 0)
    
    start := time.Now()
    _, err = conn.Write(msg)
    if err != nil {
        return 0, err
    }
    
    buf := make([]byte, 1500)
    _, err = conn.Read(buf)
    if err != nil {
        return 0, err
    }
    
    return time.Since(start), nil
}

func buildICMPEchoRequest(id, seq int) []byte {
    msg := make([]byte, 8)
    msg[0] = 8 // Type: Echo Request
    msg[1] = 0 // Code: 0
    binary.BigEndian.PutUint16(msg[4:], uint16(id))
    binary.BigEndian.PutUint16(msg[6:], uint16(seq))
    
    // Checksum
    csum := checksum(msg)
    binary.BigEndian.PutUint16(msg[2:], csum)
    
    return msg
}

func checksum(data []byte) uint16 {
    sum := uint32(0)
    for i := 0; i < len(data)-1; i += 2 {
        sum += uint32(binary.BigEndian.Uint16(data[i:]))
    }
    if len(data)%2 == 1 {
        sum += uint32(data[len(data)-1]) << 8
    }
    sum = (sum >> 16) + (sum & 0xffff)
    sum += sum >> 16
    return ^uint16(sum)
}

// IPv6 address utilities
type IPv6Utils struct{}

func (u *IPv6Utils) ExpandAddress(addr string) (string, error) {
    ip := net.ParseIP(addr)
    if ip == nil {
        return "", fmt.Errorf("invalid IPv6 address: %s", addr)
    }
    
    ip6 := ip.To16()
    if ip6 == nil {
        return "", fmt.Errorf("not an IPv6 address: %s", addr)
    }
    
    parts := make([]string, 8)
    for i := 0; i < 8; i++ {
        parts[i] = fmt.Sprintf("%04x", binary.BigEndian.Uint16(ip6[i*2:]))
    }
    
    return strings.Join(parts, ":"), nil
}

func (u *IPv6Utils) CompressAddress(addr string) string {
    ip := net.ParseIP(addr)
    if ip == nil {
        return addr
    }
    return ip.String()
}

func (u *IPv6Utils) GenerateEUI64(mac net.HardwareAddr) net.IP {
    if len(mac) != 6 {
        return nil
    }
    
    eui64 := make([]byte, 8)
    eui64[0] = mac[0] ^ 0x02 // Flip universal/local bit
    eui64[1] = mac[1]
    eui64[2] = mac[2]
    eui64[3] = 0xff
    eui64[4] = 0xfe
    eui64[5] = mac[3]
    eui64[6] = mac[4]
    eui64[7] = mac[5]
    
    return eui64
}

func (u *IPv6Utils) GenerateLinkLocal(mac net.HardwareAddr) net.IP {
    eui64 := u.GenerateEUI64(mac)
    if eui64 == nil {
        return nil
    }
    
    ip := make(net.IP, 16)
    ip[0] = 0xfe
    ip[1] = 0x80
    // bytes 2-7 are zero
    copy(ip[8:], eui64)
    
    return ip
}

// MTU path discovery simulator
type MTUDiscovery struct {
    target  string
    timeout time.Duration
}

type MTUResult struct {
    Target   string
    PathMTU  int
    Hops     []MTUHop
}

type MTUHop struct {
    IP  string
    MTU int
}

func (d *MTUDiscovery) Discover(ctx context.Context) (*MTUResult, error) {
    result := &MTUResult{
        Target: d.target,
    }
    
    // Binary search for MTU
    low := 68   // Minimum IPv4 MTU
    high := 9000 // Maximum (jumbo frame)
    
    for low < high {
        mid := (low + high + 1) / 2
        
        if d.canSend(ctx, mid) {
            low = mid
        } else {
            high = mid - 1
        }
    }
    
    result.PathMTU = low
    return result, nil
}

func (d *MTUDiscovery) canSend(ctx context.Context, size int) bool {
    conn, err := net.DialTimeout("ip4:icmp", d.target, d.timeout)
    if err != nil {
        return false
    }
    defer conn.Close()
    
    conn.SetDeadline(time.Now().Add(d.timeout))
    
    // Send packet of specified size with DF bit
    payload := make([]byte, size-20) // subtract IP header
    _, err = conn.Write(payload)
    if err != nil {
        return false
    }
    
    buf := make([]byte, 1500)
    _, err = conn.Read(buf)
    return err == nil
}

// Network interface monitor
type InterfaceMonitor struct {
    interfaces map[string]*InterfaceStats
    interval   time.Duration
}

type InterfaceStats struct {
    Name      string
    Addresses []string
    MTU       int
    Flags     net.Flags
    IsUp      bool
    Speed     string
}

func (m *InterfaceMonitor) Snapshot() ([]InterfaceStats, error) {
    ifaces, err := net.Interfaces()
    if err != nil {
        return nil, err
    }
    
    var stats []InterfaceStats
    for _, iface := range ifaces {
        s := InterfaceStats{
            Name:  iface.Name,
            MTU:   iface.MTU,
            Flags: iface.Flags,
            IsUp:  iface.Flags&net.FlagUp != 0,
        }
        
        addrs, err := iface.Addrs()
        if err == nil {
            for _, addr := range addrs {
                s.Addresses = append(s.Addresses, addr.String())
            }
        }
        
        stats = append(stats, s)
    }
    
    return stats, nil
}`,
				},
			},
		},
	})
}
