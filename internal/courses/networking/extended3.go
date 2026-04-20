package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2617,
			Title:       "DNS Architecture and Resolution",
			Description: "Deep dive into DNS architecture, resolution process, record types, DNSSEC, DNS-based load balancing, and modern DNS protocols.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "DNS Resolution and Record Types",
					Content: `The Domain Name System (DNS) is a hierarchical distributed naming system that translates domain names to IP addresses.

**DNS Hierarchy:**

  . (root)
  ├── .com (TLD)
  │   ├── google.com (second-level)
  │   │   ├── www.google.com
  │   │   ├── mail.google.com
  │   │   └── dns.google.com
  │   └── example.com
  ├── .org
  │   └── wikipedia.org
  ├── .net
  └── .io

**DNS Resolution Process (Recursive):**

  Client -> Recursive Resolver -> Root Server
                                -> TLD Server (.com)
                                -> Authoritative Server (example.com)
                                -> Returns IP address
  
  Step by step:
    1. Client queries recursive resolver (ISP or 8.8.8.8)
    2. Resolver checks cache, if miss:
    3. Query root server: "Who handles .com?"
    4. Root returns .com TLD server address
    5. Query TLD server: "Who handles example.com?"
    6. TLD returns authoritative nameserver for example.com
    7. Query authoritative: "What is the A record for www.example.com?"
    8. Authoritative returns IP address
    9. Resolver caches result (respects TTL)
    10. Returns IP to client

**DNS Record Types:**

A Record:
  Maps domain to IPv4 address
  example.com -> 93.184.216.34
  Most common record type

AAAA Record:
  Maps domain to IPv6 address
  example.com -> 2606:2800:220:1:248:1893:25c8:1946

CNAME Record:
  Canonical name (alias)
  www.example.com -> example.com
  Cannot coexist with other records for same name
  Cannot be at zone apex (use ALIAS/ANAME instead)

MX Record:
  Mail exchange servers
  Priority + mail server hostname
  example.com MX 10 mail1.example.com
  example.com MX 20 mail2.example.com (backup)

NS Record:
  Authoritative nameservers for domain
  example.com NS ns1.example.com
  example.com NS ns2.example.com

TXT Record:
  Arbitrary text data
  Used for SPF, DKIM, DMARC (email authentication)
  Domain verification (Google, Let's Encrypt)
  
  SPF: "v=spf1 include:_spf.google.com ~all"
  DKIM: "v=DKIM1; k=rsa; p=MIGf..."

SRV Record:
  Service location
  _service._protocol.name TTL class SRV priority weight port target
  _http._tcp.example.com. 3600 IN SRV 10 60 80 web1.example.com.

PTR Record:
  Reverse DNS lookup (IP to domain)
  34.216.184.93.in-addr.arpa -> example.com

SOA Record:
  Start of Authority
  Primary nameserver, admin email, serial number
  Zone transfer parameters (refresh, retry, expire)

CAA Record:
  Certificate Authority Authorization
  Specifies which CAs can issue certificates
  example.com CAA 0 issue "letsencrypt.org"

**DNS Caching:**

  TTL (Time-to-Live):
    How long a record can be cached
    Short TTL (60s): Frequent changes, faster failover
    Long TTL (86400s): Stable records, reduce DNS load
    
  Caching layers:
    Browser cache (minutes)
    OS resolver cache (varies)
    Local DNS resolver
    Recursive resolver
    ISP resolver

**DNS-Based Load Balancing:**

Round-Robin DNS:
  Multiple A records for same domain
  Returns records in rotating order
  Simple, no health checking
  
  example.com A 1.2.3.4
  example.com A 5.6.7.8
  example.com A 9.10.11.12

Weighted DNS:
  Different weights for different records
  Route more traffic to powerful servers
  AWS Route 53 weighted routing

Geolocation DNS:
  Return nearest server based on client location
  Reduce latency for global users
  AWS Route 53, Cloudflare, NS1

Latency-Based DNS:
  Measure latency to each server
  Return server with lowest latency
  More accurate than geolocation

Health Check DNS:
  Monitor server health
  Remove unhealthy servers from responses
  Failover to backup servers
  AWS Route 53 health checks

**DNSSEC (DNS Security Extensions):**

  Adds authentication and integrity to DNS
  Doesn't encrypt queries (that's DoH/DoT)
  
  How it works:
    Zone Signing Key (ZSK): Signs DNS records
    Key Signing Key (KSK): Signs the ZSK
    DS Record: Parent zone stores hash of child's KSK
    RRSIG: Signature over resource record set
    NSEC/NSEC3: Proves non-existence of records
    
  Chain of trust:
    Root -> .com DS -> example.com DNSKEY -> RRSIG -> Records
    
  Validation:
    Resolver obtains DNSKEY and RRSIG
    Verifies RRSIG using DNSKEY
    Verifies DNSKEY using parent DS record
    Chain up to trusted root key

**Modern DNS Protocols:**

DNS over HTTPS (DoH):
  DNS queries over HTTPS (port 443)
  Encrypted, private DNS lookups
  Bypasses DNS-based filtering
  Looks like normal HTTPS traffic
  Endpoint: https://dns.google/dns-query

DNS over TLS (DoT):
  DNS queries over TLS (port 853)
  Encrypted DNS lookups
  Dedicated port (easier to block/monitor)
  Supported by many public resolvers

DNS over QUIC (DoQ):
  DNS over QUIC protocol
  Lower latency than DoT
  Multiplexed streams
  Connection migration`,
					CodeExamples: `// DNS implementation and tools in Go

package main

import (
    "context"
    "fmt"
    "net"
    "strings"
    "time"
)

// DNS resolver with caching
type DNSResolver struct {
    cache    map[string]*CacheEntry
    upstream string
    timeout  time.Duration
}

type CacheEntry struct {
    Records   []string
    ExpiresAt time.Time
    RecordType string
}

func NewDNSResolver(upstream string) *DNSResolver {
    return &DNSResolver{
        cache:    make(map[string]*CacheEntry),
        upstream: upstream,
        timeout:  5 * time.Second,
    }
}

func (r *DNSResolver) Resolve(ctx context.Context, domain string, recordType string) ([]string, error) {
    // Check cache
    cacheKey := domain + ":" + recordType
    if entry, ok := r.cache[cacheKey]; ok {
        if time.Now().Before(entry.ExpiresAt) {
            return entry.Records, nil
        }
        delete(r.cache, cacheKey)
    }
    
    // Query DNS
    resolver := &net.Resolver{
        PreferGo: true,
        Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
            d := net.Dialer{Timeout: r.timeout}
            return d.DialContext(ctx, "udp", r.upstream+":53")
        },
    }
    
    ctx, cancel := context.WithTimeout(ctx, r.timeout)
    defer cancel()
    
    var records []string
    var err error
    
    switch recordType {
    case "A":
        var ips []net.IP
        ips, err = resolver.LookupIP(ctx, "ip4", domain)
        for _, ip := range ips {
            records = append(records, ip.String())
        }
    case "AAAA":
        var ips []net.IP
        ips, err = resolver.LookupIP(ctx, "ip6", domain)
        for _, ip := range ips {
            records = append(records, ip.String())
        }
    case "CNAME":
        var cname string
        cname, err = resolver.LookupCNAME(ctx, domain)
        if err == nil {
            records = append(records, cname)
        }
    case "MX":
        var mxRecords []*net.MX
        mxRecords, err = resolver.LookupMX(ctx, domain)
        for _, mx := range mxRecords {
            records = append(records, fmt.Sprintf("%d %s", mx.Pref, mx.Host))
        }
    case "NS":
        var nsRecords []*net.NS
        nsRecords, err = resolver.LookupNS(ctx, domain)
        for _, ns := range nsRecords {
            records = append(records, ns.Host)
        }
    case "TXT":
        records, err = resolver.LookupTXT(ctx, domain)
    default:
        return nil, fmt.Errorf("unsupported record type: %s", recordType)
    }
    
    if err != nil {
        return nil, err
    }
    
    // Cache result
    r.cache[cacheKey] = &CacheEntry{
        Records:    records,
        ExpiresAt:  time.Now().Add(5 * time.Minute),
        RecordType: recordType,
    }
    
    return records, nil
}

// DNS health checker for load balancing
type DNSHealthChecker struct {
    targets  []HealthTarget
    interval time.Duration
    healthy  map[string]bool
}

type HealthTarget struct {
    Name     string
    IP       string
    Port     int
    CheckURL string
    Weight   int
}

func NewDNSHealthChecker(targets []HealthTarget, interval time.Duration) *DNSHealthChecker {
    healthy := make(map[string]bool)
    for _, t := range targets {
        healthy[t.IP] = true
    }
    return &DNSHealthChecker{
        targets:  targets,
        interval: interval,
        healthy:  healthy,
    }
}

func (hc *DNSHealthChecker) Start(ctx context.Context) {
    ticker := time.NewTicker(hc.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            hc.checkAll()
        }
    }
}

func (hc *DNSHealthChecker) checkAll() {
    for _, target := range hc.targets {
        addr := fmt.Sprintf("%s:%d", target.IP, target.Port)
        conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
        if err != nil {
            hc.healthy[target.IP] = false
            continue
        }
        conn.Close()
        hc.healthy[target.IP] = true
    }
}

func (hc *DNSHealthChecker) GetHealthyTargets() []HealthTarget {
    var healthy []HealthTarget
    for _, t := range hc.targets {
        if hc.healthy[t.IP] {
            healthy = append(healthy, t)
        }
    }
    return healthy
}

// Domain validation utility
func ValidateDomain(domain string) error {
    if len(domain) > 253 {
        return fmt.Errorf("domain too long: %d characters (max 253)", len(domain))
    }
    
    labels := strings.Split(domain, ".")
    for _, label := range labels {
        if len(label) == 0 {
            return fmt.Errorf("empty label in domain")
        }
        if len(label) > 63 {
            return fmt.Errorf("label too long: %s (%d chars, max 63)", label, len(label))
        }
        
        for i, c := range label {
            if !isValidDNSChar(c) {
                return fmt.Errorf("invalid character '%c' in label %s", c, label)
            }
            if c == '-' && (i == 0 || i == len(label)-1) {
                return fmt.Errorf("label cannot start or end with hyphen: %s", label)
            }
        }
    }
    
    return nil
}

func isValidDNSChar(c rune) bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '-'
}

// Reverse DNS lookup utility
func ReverseDNS(ip string) ([]string, error) {
    names, err := net.LookupAddr(ip)
    if err != nil {
        return nil, fmt.Errorf("reverse DNS lookup failed for %s: %w", ip, err)
    }
    return names, nil
}

// DNS propagation checker
type PropagationChecker struct {
    resolvers []string
    timeout   time.Duration
}

type PropagationResult struct {
    Resolver string
    Records  []string
    Latency  time.Duration
    Error    error
}

func NewPropagationChecker(resolvers []string) *PropagationChecker {
    return &PropagationChecker{
        resolvers: resolvers,
        timeout:   5 * time.Second,
    }
}

func (pc *PropagationChecker) Check(domain string) []PropagationResult {
    results := make([]PropagationResult, len(pc.resolvers))
    
    for i, resolver := range pc.resolvers {
        start := time.Now()
        r := &net.Resolver{
            PreferGo: true,
            Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
                d := net.Dialer{Timeout: pc.timeout}
                return d.DialContext(ctx, "udp", resolver+":53")
            },
        }
        
        ctx, cancel := context.WithTimeout(context.Background(), pc.timeout)
        ips, err := r.LookupHost(ctx, domain)
        cancel()
        
        results[i] = PropagationResult{
            Resolver: resolver,
            Records:  ips,
            Latency:  time.Since(start),
            Error:    err,
        }
    }
    
    return results
}`,
				},
			},
		},
		{
			ID:          2618,
			Title:       "Software-Defined Networking and Network Virtualization",
			Description: "Explore SDN architecture, network overlays, VXLAN, network function virtualization, and container networking models.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "SDN and Network Virtualization",
					Content: `Software-Defined Networking decouples the control plane from the data plane, enabling programmable network management.

**SDN Architecture:**

  Application Layer:
    Network applications (firewall, load balancer, monitoring)
    Use northbound API to communicate with controller
    
  Control Layer:
    SDN Controller (centralized brain)
    Maintains network topology and state
    Programs data plane via southbound API
    Examples: OpenDaylight, ONOS, Ryu, Floodlight
    
  Data Layer:
    Network switches and routers
    Forward packets based on flow tables
    Receive forwarding rules from controller
    OpenFlow protocol for communication

  ┌────────────────────────────────────┐
  │    Applications (Firewall, LB)     │
  ├────────────────────────────────────┤
  │       Northbound API (REST)        │
  ├────────────────────────────────────┤
  │       SDN Controller               │
  ├────────────────────────────────────┤
  │    Southbound API (OpenFlow)       │
  ├────────────────────────────────────┤
  │    Network Devices (Switches)      │
  └────────────────────────────────────┘

**OpenFlow:**

  Protocol between SDN controller and switches
  Flow table entries:
    Match: Packet header fields (src/dst IP, port, VLAN, etc.)
    Action: Forward, drop, modify, send to controller
    Counters: Packet/byte counts
    Priority: Higher priority rules matched first
    Timeout: Idle and hard timeouts

  Packet processing:
    1. Packet arrives at switch
    2. Match against flow table (highest priority first)
    3. If match: Execute action
    4. If no match: Send to controller (packet-in)
    5. Controller decides action, installs flow rule

**Network Overlays:**

VXLAN (Virtual Extensible LAN):
  Encapsulates Layer 2 frames in UDP packets
  24-bit VXLAN Network Identifier (VNI)
  Supports up to 16 million virtual networks (vs 4096 VLANs)
  
  Original Frame -> [VXLAN Header] -> [UDP Header] -> [IP Header] -> [Ethernet]
  
  VTEP (VXLAN Tunnel Endpoint):
    Encapsulates/decapsulates VXLAN packets
    Maps VNI to local VLAN/bridge
    Can be hardware or software

GRE (Generic Routing Encapsulation):
  Simple point-to-point tunneling
  Encapsulates various protocols
  No built-in encryption (pair with IPSec)

GENEVE (Generic Network Virtualization Encapsulation):
  Extensible tunnel format
  Variable-length options for metadata
  Designed to replace VXLAN and GRE
  Used by Open Virtual Network (OVN)

**Network Function Virtualization (NFV):**

  Replace dedicated network hardware with software
  Run on commodity servers
  
  Traditional: Hardware firewall, hardware load balancer
  NFV: Software firewall VM, software LB container
  
  Virtual Network Functions (VNFs):
    Virtual firewall
    Virtual load balancer
    Virtual router
    Virtual IDS/IPS
    Virtual WAN optimizer
  
  Benefits:
    Lower cost (commodity hardware)
    Faster deployment (minutes vs weeks)
    Elastic scaling
    Easier upgrades

**Container Networking:**

Docker Networking:
  Bridge (default): Virtual bridge, NAT to host
  Host: Share host network namespace
  Overlay: Multi-host networking (Docker Swarm)
  Macvlan: Direct MAC assignment, bypasses bridge
  None: No networking

Kubernetes Networking Model:
  Every Pod gets its own IP address
  Pods can communicate without NAT
  Agents on a node can communicate with all pods on that node
  
  CNI (Container Network Interface):
    Standard API for container networking
    Plugins: Calico, Cilium, Flannel, Weave, AWS VPC CNI
    
  Calico:
    BGP-based routing between nodes
    Network policies for microsegmentation
    eBPF dataplane for high performance
    
  Cilium:
    eBPF-based networking
    API-aware network security (L7 policies)
    Transparent encryption (WireGuard/IPSec)
    Service mesh (sidecar-free)
    
  Flannel:
    Simple overlay network
    VXLAN or host-gw backend
    Good for simple setups

Kubernetes Services:
  ClusterIP: Internal virtual IP
  NodePort: External access on node ports (30000-32767)
  LoadBalancer: Cloud provider load balancer
  ExternalName: CNAME to external service

  kube-proxy modes:
    iptables: Default, rule-based routing
    IPVS: High-performance L4 load balancing
    eBPF: Cilium replaces kube-proxy entirely

**Service Discovery:**

DNS-based:
  Kubernetes DNS (CoreDNS)
  service.namespace.svc.cluster.local
  Automatic DNS records for services
  SRV records for port discovery

API-based:
  Consul: Service registration + health checking
  etcd: Key-value store with watch
  ZooKeeper: Coordination service

**Network Policies (Kubernetes):**

  Control traffic flow between pods
  Default: Allow all traffic
  With policy: Default deny, explicit allow
  
  Ingress rules: Control incoming traffic
  Egress rules: Control outgoing traffic
  
  Selectors:
    Pod selector: Match pods by labels
    Namespace selector: Match namespaces
    IP block: Match CIDR ranges
    
  Example concept:
    Allow traffic from frontend pods to backend pods on port 8080
    Deny all other traffic to backend pods`,
					CodeExamples: `// Container networking and SDN concepts in Go

package main

import (
    "context"
    "encoding/binary"
    "fmt"
    "net"
    "sync"
)

// CIDR calculator
type CIDRCalculator struct{}

type SubnetInfo struct {
    Network    net.IP
    Broadcast  net.IP
    FirstHost  net.IP
    LastHost   net.IP
    Netmask    net.IPMask
    TotalHosts int
    UsableHosts int
    CIDR       string
}

func (c *CIDRCalculator) Calculate(cidr string) (*SubnetInfo, error) {
    ip, network, err := net.ParseCIDR(cidr)
    if err != nil {
        return nil, fmt.Errorf("invalid CIDR: %w", err)
    }
    
    ones, bits := network.Mask.Size()
    totalHosts := 1 << (bits - ones)
    
    // Network address
    networkIP := network.IP.To4()
    
    // Broadcast address
    broadcast := make(net.IP, 4)
    for i := range networkIP {
        broadcast[i] = networkIP[i] | ^network.Mask[i]
    }
    
    // First and last usable hosts
    firstHost := make(net.IP, 4)
    copy(firstHost, networkIP)
    firstHost[3]++
    
    lastHost := make(net.IP, 4)
    copy(lastHost, broadcast)
    lastHost[3]--
    
    return &SubnetInfo{
        Network:     networkIP,
        Broadcast:   broadcast,
        FirstHost:   firstHost,
        LastHost:    lastHost,
        Netmask:     network.Mask,
        TotalHosts:  totalHosts,
        UsableHosts: totalHosts - 2,
        CIDR:        fmt.Sprintf("%s/%d", ip.String(), ones),
    }, nil
}

// Subnet allocator for container networking
type SubnetAllocator struct {
    mu        sync.Mutex
    baseNet   *net.IPNet
    subnetBits int
    allocated map[string]string // node -> subnet CIDR
    available []string
}

func NewSubnetAllocator(baseCIDR string, subnetBits int) (*SubnetAllocator, error) {
    _, baseNet, err := net.ParseCIDR(baseCIDR)
    if err != nil {
        return nil, err
    }
    
    alloc := &SubnetAllocator{
        baseNet:    baseNet,
        subnetBits: subnetBits,
        allocated:  make(map[string]string),
    }
    
    // Generate available subnets
    ones, bits := baseNet.Mask.Size()
    subnetCount := 1 << (subnetBits - ones)
    subnetSize := 1 << (bits - subnetBits)
    
    baseIP := ipToUint32(baseNet.IP.To4())
    for i := 0; i < subnetCount; i++ {
        subnetIP := uint32ToIP(baseIP + uint32(i*subnetSize))
        cidr := fmt.Sprintf("%s/%d", subnetIP.String(), subnetBits)
        alloc.available = append(alloc.available, cidr)
    }
    
    return alloc, nil
}

func (a *SubnetAllocator) Allocate(nodeID string) (string, error) {
    a.mu.Lock()
    defer a.mu.Unlock()
    
    // Check if already allocated
    if cidr, ok := a.allocated[nodeID]; ok {
        return cidr, nil
    }
    
    if len(a.available) == 0 {
        return "", fmt.Errorf("no subnets available")
    }
    
    cidr := a.available[0]
    a.available = a.available[1:]
    a.allocated[nodeID] = cidr
    
    return cidr, nil
}

func (a *SubnetAllocator) Release(nodeID string) {
    a.mu.Lock()
    defer a.mu.Unlock()
    
    if cidr, ok := a.allocated[nodeID]; ok {
        a.available = append(a.available, cidr)
        delete(a.allocated, nodeID)
    }
}

func ipToUint32(ip net.IP) uint32 {
    return binary.BigEndian.Uint32(ip.To4())
}

func uint32ToIP(n uint32) net.IP {
    ip := make(net.IP, 4)
    binary.BigEndian.PutUint32(ip, n)
    return ip
}

// Simple service registry for service discovery
type ServiceRegistry struct {
    mu       sync.RWMutex
    services map[string][]ServiceInstance
}

type ServiceInstance struct {
    ID       string
    Name     string
    Address  string
    Port     int
    Tags     []string
    Healthy  bool
    Metadata map[string]string
}

func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string][]ServiceInstance),
    }
}

func (r *ServiceRegistry) Register(instance ServiceInstance) {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    instances := r.services[instance.Name]
    
    // Update if exists
    for i, inst := range instances {
        if inst.ID == instance.ID {
            instances[i] = instance
            return
        }
    }
    
    r.services[instance.Name] = append(instances, instance)
}

func (r *ServiceRegistry) Deregister(name, id string) {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    instances := r.services[name]
    for i, inst := range instances {
        if inst.ID == id {
            r.services[name] = append(instances[:i], instances[i+1:]...)
            return
        }
    }
}

func (r *ServiceRegistry) Lookup(name string) []ServiceInstance {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    var healthy []ServiceInstance
    for _, inst := range r.services[name] {
        if inst.Healthy {
            healthy = append(healthy, inst)
        }
    }
    return healthy
}

// Simple round-robin load balancer
type RoundRobinLB struct {
    mu      sync.Mutex
    counter uint64
    registry *ServiceRegistry
}

func NewRoundRobinLB(registry *ServiceRegistry) *RoundRobinLB {
    return &RoundRobinLB{registry: registry}
}

func (lb *RoundRobinLB) Pick(serviceName string) (*ServiceInstance, error) {
    instances := lb.registry.Lookup(serviceName)
    if len(instances) == 0 {
        return nil, fmt.Errorf("no healthy instances for %s", serviceName)
    }
    
    lb.mu.Lock()
    idx := lb.counter % uint64(len(instances))
    lb.counter++
    lb.mu.Unlock()
    
    return &instances[idx], nil
}

// Network policy evaluator
type NetworkPolicy struct {
    Name          string
    PodSelector   map[string]string
    IngressRules  []IngressRule
    EgressRules   []EgressRule
}

type IngressRule struct {
    FromPodSelector map[string]string
    FromNamespace   string
    FromCIDR        string
    Ports           []PolicyPort
}

type EgressRule struct {
    ToPodSelector map[string]string
    ToNamespace   string
    ToCIDR        string
    Ports         []PolicyPort
}

type PolicyPort struct {
    Protocol string
    Port     int
}

type Pod struct {
    Name      string
    Namespace string
    Labels    map[string]string
    IP        string
}

func EvaluatePolicy(policy *NetworkPolicy, srcPod, dstPod Pod, dstPort int) bool {
    // Check if policy applies to destination pod
    if !matchLabels(dstPod.Labels, policy.PodSelector) {
        return true // Policy doesn't apply
    }
    
    // Check ingress rules
    for _, rule := range policy.IngressRules {
        if rule.FromPodSelector != nil {
            if matchLabels(srcPod.Labels, rule.FromPodSelector) {
                if portAllowed(rule.Ports, dstPort) {
                    return true
                }
            }
        }
        if rule.FromCIDR != "" {
            _, cidr, _ := net.ParseCIDR(rule.FromCIDR)
            if cidr != nil && cidr.Contains(net.ParseIP(srcPod.IP)) {
                if portAllowed(rule.Ports, dstPort) {
                    return true
                }
            }
        }
    }
    
    return false // Denied by default when policy exists
}

func matchLabels(podLabels, selector map[string]string) bool {
    for k, v := range selector {
        if podLabels[k] != v {
            return false
        }
    }
    return true
}

func portAllowed(ports []PolicyPort, targetPort int) bool {
    if len(ports) == 0 {
        return true // All ports allowed
    }
    for _, p := range ports {
        if p.Port == targetPort {
            return true
        }
    }
    return false
}`,
				},
			},
		},
	})
}
