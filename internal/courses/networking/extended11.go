package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2630,
			Title:       "Cloud Networking and Virtual Private Clouds",
			Description: "Master cloud networking concepts including VPCs, peering, transit gateways, PrivateLink, service mesh, and multi-cloud networking.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "Cloud Networking Architecture and Services",
					Content: `Cloud networking abstracts physical infrastructure into software-defined network services. Understanding cloud networking is critical for modern application deployment.

**Virtual Private Cloud (VPC):**

Core Concepts:
  VPC: Isolated network in cloud provider
  Subnet: IP range within a VPC
  Route Table: Routing rules for subnets
  Internet Gateway: VPC internet access
  NAT Gateway: Outbound internet for private subnets
  Security Group: Stateful instance firewall
  Network ACL: Stateless subnet firewall

VPC Design:
  Region: Geographic area (us-east-1, eu-west-1)
  Availability Zone: Isolated data centers within region
  
  CIDR Planning:
    VPC: 10.0.0.0/16 (65,536 addresses)
    Public Subnet AZ-A: 10.0.1.0/24 (254 hosts)
    Public Subnet AZ-B: 10.0.2.0/24 (254 hosts)
    Private Subnet AZ-A: 10.0.10.0/24 (254 hosts)
    Private Subnet AZ-B: 10.0.11.0/24 (254 hosts)
    Database Subnet AZ-A: 10.0.20.0/24 (254 hosts)
    Database Subnet AZ-B: 10.0.21.0/24 (254 hosts)
    
  Best Practices:
    Use /16 or /20 for VPC CIDR
    Leave room for expansion
    Use consistent CIDR scheme across VPCs
    Separate public, private, and data tiers
    Deploy across multiple AZs

**VPC Peering:**

  Direct connection between two VPCs
  Same or different accounts/regions
  Not transitive (A-B and B-C doesn't connect A-C)
  Uses private IP addresses
  
  Limitations:
    No overlapping CIDR ranges
    No transitive routing
    Max peers varies by provider
    Cross-region has bandwidth costs
    
  Route Tables:
    VPC-A route table: 10.1.0.0/16 -> pcx-abc (Peer B)
    VPC-B route table: 10.0.0.0/16 -> pcx-abc (Peer A)

**Transit Gateway:**

  Hub-and-spoke network connectivity
  Centralized routing for multiple VPCs
  
  Benefits:
    Transitive routing (all VPCs can reach each other)
    Centralized traffic management
    VPN and Direct Connect attachment
    Route table segmentation
    Multicast support
    
  Architecture:
    VPC-A --|
    VPC-B --|-- Transit Gateway --|-- On-premises (VPN)
    VPC-C --|                     |-- Direct Connect
    VPC-D --|
    
  Route Domains:
    Shared Services: All VPCs can reach
    Production: Only prod VPCs
    Development: Only dev VPCs
    Each domain has its own route table

**AWS PrivateLink / Azure Private Link / GCP Private Service Connect:**

  Access services without internet exposure
  Traffic stays on cloud backbone
  
  Components:
    Service Provider: Exposes service via endpoint
    Service Consumer: Connects via interface endpoint
    ENI (Elastic Network Interface): In consumer's VPC
    
  Use Cases:
    Access AWS services (S3, DynamoDB) privately
    Expose services to other accounts/VPCs
    Third-party SaaS connectivity
    Zero-trust network access
    
  VPC Endpoints:
    Gateway Endpoints: S3, DynamoDB (route table entry)
    Interface Endpoints: ENI with private IP (PrivateLink)

**Load Balancers in Cloud:**

Application Load Balancer (Layer 7):
  HTTP/HTTPS routing
  Path-based routing (/api -> service A, /web -> service B)
  Host-based routing (api.example.com, www.example.com)
  WebSocket support
  gRPC support
  WAF integration

Network Load Balancer (Layer 4):
  TCP/UDP/TLS routing
  Ultra-low latency (microseconds)
  Static IP / Elastic IP
  Millions of requests per second
  Preserve source IP
  
Gateway Load Balancer (Layer 3):
  Transparent network gateway
  Inline security appliances
  Firewalls, IDS/IPS insertion

**Service Mesh Networking:**

  Manages service-to-service communication
  Sidecar proxy pattern (Envoy, Linkerd-proxy)
  
  Control Plane:
    Istio, Linkerd, Consul Connect
    Service discovery
    Configuration distribution
    Certificate management
    
  Data Plane (Sidecar):
    Traffic routing and load balancing
    Mutual TLS (mTLS)
    Retries and circuit breaking
    Observability (metrics, traces, logs)
    Rate limiting
    
  Traffic Management:
    Virtual Service: Routing rules
    Destination Rule: Load balancing policy
    Gateway: Ingress/egress config
    Service Entry: External services
    
  Features:
    Canary deployments (90/10 traffic split)
    A/B testing
    Traffic mirroring
    Fault injection

**Multi-Cloud Networking:**

  Connecting workloads across cloud providers
  
  Approaches:
    VPN tunnels between clouds
    Dedicated interconnects
    Software-defined WAN (SD-WAN)
    Cloud networking platforms (Aviatrix, Alkira)
    
  Challenges:
    Different networking models
    Overlapping IP spaces
    DNS resolution across clouds
    Consistent security policies
    Cost optimization
    Latency management
    
  DNS Across Clouds:
    Each cloud has DNS service
    Route 53 (AWS), Cloud DNS (GCP), Azure DNS
    External DNS for multi-cloud
    Split-horizon DNS

**Direct Connect / ExpressRoute / Cloud Interconnect:**

  Private dedicated connection to cloud
  Bypasses public internet
  Lower latency, more bandwidth
  
  AWS Direct Connect:
    1 Gbps or 10 Gbps dedicated
    50-500 Mbps hosted connections
    Virtual Interfaces: Public, Private, Transit
    
  Azure ExpressRoute:
    Microsoft peering: Microsoft 365, Azure PaaS
    Private peering: Azure VNets
    Global Reach: Connect on-premises sites
    FastPath: Bypass ExpressRoute gateway
    
  GCP Cloud Interconnect:
    Dedicated: 10/100 Gbps
    Partner: 50 Mbps - 50 Gbps
    Cross-Cloud: Direct to other cloud providers`,
					CodeExamples: `// Cloud networking implementations

package main

import (
    "fmt"
    "math/rand"
    "net"
    "strings"
    "sync"
    "time"
)

// VPC network manager
type VPCManager struct {
    vpcs map[string]*VPC
    peerings map[string]*VPCPeering
    transitGateways map[string]*TransitGateway
    mu sync.RWMutex
}

type VPC struct {
    ID       string
    Name     string
    CIDR     string
    Region   string
    Subnets  []*Subnet
    RouteTables []*RouteTable
    SecurityGroups []*SecurityGroup
    NACLs    []*NetworkACL
}

type Subnet struct {
    ID       string
    Name     string
    CIDR     string
    AZ       string
    Public   bool
    RouteTableID string
}

type RouteTable struct {
    ID     string
    Name   string
    Routes []Route
}

type Route struct {
    Destination string
    Target      string // igw-xxx, nat-xxx, pcx-xxx, tgw-xxx, local
    Status      string
}

type SecurityGroup struct {
    ID          string
    Name        string
    Description string
    IngressRules []SGRule
    EgressRules  []SGRule
}

type SGRule struct {
    Protocol  string // tcp, udp, icmp, -1 (all)
    FromPort  int
    ToPort    int
    Source    string // CIDR or SG ID
}

type NetworkACL struct {
    ID      string
    Rules   []NACLRule
}

type NACLRule struct {
    RuleNumber int
    Protocol   string
    Action     string // allow, deny
    CIDR       string
    FromPort   int
    ToPort     int
    Egress     bool
}

func NewVPCManager() *VPCManager {
    return &VPCManager{
        vpcs:            make(map[string]*VPC),
        peerings:        make(map[string]*VPCPeering),
        transitGateways: make(map[string]*TransitGateway),
    }
}

func (m *VPCManager) CreateVPC(name, cidr, region string) (*VPC, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    // Validate CIDR
    _, _, err := net.ParseCIDR(cidr)
    if err != nil {
        return nil, fmt.Errorf("invalid CIDR: %w", err)
    }
    
    // Check for overlapping CIDRs in same region
    for _, vpc := range m.vpcs {
        if vpc.Region == region && cidrsOverlap(vpc.CIDR, cidr) {
            return nil, fmt.Errorf("CIDR %s overlaps with VPC %s (%s)",
                cidr, vpc.Name, vpc.CIDR)
        }
    }
    
    id := fmt.Sprintf("vpc-%08x", rand.Int31())
    vpc := &VPC{
        ID:     id,
        Name:   name,
        CIDR:   cidr,
        Region: region,
        RouteTables: []*RouteTable{
            {
                ID:   fmt.Sprintf("rtb-%08x", rand.Int31()),
                Name: "main",
                Routes: []Route{
                    {Destination: cidr, Target: "local", Status: "active"},
                },
            },
        },
    }
    
    m.vpcs[id] = vpc
    return vpc, nil
}

func (m *VPCManager) CreateSubnet(vpcID, name, cidr, az string, public bool) (*Subnet, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    vpc, exists := m.vpcs[vpcID]
    if !exists {
        return nil, fmt.Errorf("VPC %s not found", vpcID)
    }
    
    // Verify subnet CIDR is within VPC CIDR
    if !cidrContains(vpc.CIDR, cidr) {
        return nil, fmt.Errorf("subnet CIDR %s not within VPC CIDR %s", cidr, vpc.CIDR)
    }
    
    // Check overlap with existing subnets
    for _, s := range vpc.Subnets {
        if cidrsOverlap(s.CIDR, cidr) {
            return nil, fmt.Errorf("subnet CIDR %s overlaps with %s (%s)",
                cidr, s.Name, s.CIDR)
        }
    }
    
    subnet := &Subnet{
        ID:           fmt.Sprintf("subnet-%08x", rand.Int31()),
        Name:         name,
        CIDR:         cidr,
        AZ:           az,
        Public:       public,
        RouteTableID: vpc.RouteTables[0].ID,
    }
    
    vpc.Subnets = append(vpc.Subnets, subnet)
    return subnet, nil
}

func cidrsOverlap(a, b string) bool {
    _, netA, errA := net.ParseCIDR(a)
    _, netB, errB := net.ParseCIDR(b)
    if errA != nil || errB != nil {
        return false
    }
    return netA.Contains(netB.IP) || netB.Contains(netA.IP)
}

func cidrContains(outer, inner string) bool {
    _, outerNet, errO := net.ParseCIDR(outer)
    innerIP, innerNet, errI := net.ParseCIDR(inner)
    if errO != nil || errI != nil {
        return false
    }
    _ = innerIP
    
    // Check if inner network start is in outer
    if !outerNet.Contains(innerNet.IP) {
        return false
    }
    
    // Check outer mask is smaller (wider)
    outerOnes, _ := outerNet.Mask.Size()
    innerOnes, _ := innerNet.Mask.Size()
    return outerOnes <= innerOnes
}

// VPC Peering
type VPCPeering struct {
    ID       string
    RequesterVPC string
    AccepterVPC  string
    Status       string
}

func (m *VPCManager) CreatePeering(vpcA, vpcB string) (*VPCPeering, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    a, existsA := m.vpcs[vpcA]
    b, existsB := m.vpcs[vpcB]
    if !existsA || !existsB {
        return nil, fmt.Errorf("VPC not found")
    }
    
    // Check CIDR overlap
    if cidrsOverlap(a.CIDR, b.CIDR) {
        return nil, fmt.Errorf("VPCs have overlapping CIDRs")
    }
    
    peering := &VPCPeering{
        ID:           fmt.Sprintf("pcx-%08x", rand.Int31()),
        RequesterVPC: vpcA,
        AccepterVPC:  vpcB,
        Status:       "active",
    }
    
    m.peerings[peering.ID] = peering
    
    // Add routes
    for _, rt := range a.RouteTables {
        rt.Routes = append(rt.Routes, Route{
            Destination: b.CIDR,
            Target:      peering.ID,
            Status:      "active",
        })
    }
    for _, rt := range b.RouteTables {
        rt.Routes = append(rt.Routes, Route{
            Destination: a.CIDR,
            Target:      peering.ID,
            Status:      "active",
        })
    }
    
    return peering, nil
}

// Transit Gateway
type TransitGateway struct {
    ID          string
    Name        string
    Attachments []TGWAttachment
    RouteTables []*TGWRouteTable
}

type TGWAttachment struct {
    ID     string
    Type   string // vpc, vpn, direct-connect, peering
    Target string
    State  string
}

type TGWRouteTable struct {
    ID     string
    Name   string
    Routes []TGWRoute
}

type TGWRoute struct {
    Destination  string
    AttachmentID string
    Type         string // static, propagated
}

func (m *VPCManager) CreateTransitGateway(name string) *TransitGateway {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    tgw := &TransitGateway{
        ID:   fmt.Sprintf("tgw-%08x", rand.Int31()),
        Name: name,
        RouteTables: []*TGWRouteTable{
            {
                ID:   fmt.Sprintf("tgw-rtb-%08x", rand.Int31()),
                Name: "default",
            },
        },
    }
    
    m.transitGateways[tgw.ID] = tgw
    return tgw
}

func (m *VPCManager) AttachVPC(tgwID, vpcID string) (*TGWAttachment, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    tgw, exists := m.transitGateways[tgwID]
    if !exists {
        return nil, fmt.Errorf("TGW %s not found", tgwID)
    }
    
    vpc, exists := m.vpcs[vpcID]
    if !exists {
        return nil, fmt.Errorf("VPC %s not found", vpcID)
    }
    
    attachment := TGWAttachment{
        ID:     fmt.Sprintf("tgw-attach-%08x", rand.Int31()),
        Type:   "vpc",
        Target: vpcID,
        State:  "available",
    }
    
    tgw.Attachments = append(tgw.Attachments, attachment)
    
    // Add route to TGW default route table
    if len(tgw.RouteTables) > 0 {
        tgw.RouteTables[0].Routes = append(tgw.RouteTables[0].Routes, TGWRoute{
            Destination:  vpc.CIDR,
            AttachmentID: attachment.ID,
            Type:         "propagated",
        })
    }
    
    return &attachment, nil
}

// Security group evaluator
type SGEvaluator struct{}

func (e *SGEvaluator) EvaluateIngress(sg *SecurityGroup, protocol string, port int, sourceIP string) bool {
    for _, rule := range sg.IngressRules {
        if !matchProtocol(rule.Protocol, protocol) {
            continue
        }
        if rule.Protocol != "-1" && (port < rule.FromPort || port > rule.ToPort) {
            continue
        }
        if matchSource(rule.Source, sourceIP) {
            return true
        }
    }
    return false
}

func matchProtocol(ruleProto, proto string) bool {
    if ruleProto == "-1" {
        return true // All protocols
    }
    return strings.EqualFold(ruleProto, proto)
}

func matchSource(source, ip string) bool {
    if strings.Contains(source, "/") {
        _, cidr, err := net.ParseCIDR(source)
        if err != nil {
            return false
        }
        return cidr.Contains(net.ParseIP(ip))
    }
    // Security group reference - would check membership
    return source == ip
}

// Service mesh traffic routing
type ServiceMesh struct {
    services map[string]*MeshService
    routes   []*VirtualService
    mu       sync.RWMutex
}

type MeshService struct {
    Name      string
    Endpoints []MeshEndpoint
    Port      int
}

type MeshEndpoint struct {
    Address string
    Port    int
    Weight  int
    Version string
    Healthy bool
}

type VirtualService struct {
    Name     string
    Host     string
    Routes   []TrafficRoute
}

type TrafficRoute struct {
    Match   RouteMatch
    Route   []RouteDestination
}

type RouteMatch struct {
    URI     string
    Headers map[string]string
}

type RouteDestination struct {
    Host    string
    Version string
    Weight  int
    Port    int
}

func NewServiceMesh() *ServiceMesh {
    return &ServiceMesh{
        services: make(map[string]*MeshService),
    }
}

func (m *ServiceMesh) RegisterService(svc *MeshService) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.services[svc.Name] = svc
}

func (m *ServiceMesh) AddVirtualService(vs *VirtualService) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.routes = append(m.routes, vs)
}

func (m *ServiceMesh) Route(host, uri string, headers map[string]string) *MeshEndpoint {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    for _, vs := range m.routes {
        if vs.Host != host {
            continue
        }
        
        for _, route := range vs.Routes {
            if route.Match.URI != "" && !strings.HasPrefix(uri, route.Match.URI) {
                continue
            }
            
            // Weighted routing
            totalWeight := 0
            for _, dest := range route.Route {
                totalWeight += dest.Weight
            }
            
            r := rand.Intn(totalWeight)
            cumWeight := 0
            for _, dest := range route.Route {
                cumWeight += dest.Weight
                if r < cumWeight {
                    return m.selectEndpoint(dest.Host, dest.Version)
                }
            }
        }
    }
    
    // Default: direct to service
    return m.selectEndpoint(host, "")
}

func (m *ServiceMesh) selectEndpoint(host, version string) *MeshEndpoint {
    svc, exists := m.services[host]
    if !exists {
        return nil
    }
    
    var candidates []MeshEndpoint
    for _, ep := range svc.Endpoints {
        if !ep.Healthy {
            continue
        }
        if version != "" && ep.Version != version {
            continue
        }
        candidates = append(candidates, ep)
    }
    
    if len(candidates) == 0 {
        return nil
    }
    
    return &candidates[rand.Intn(len(candidates))]
}`,
				},
			},
		},
		{
			ID:          2631,
			Title:       "Zero Trust Network Architecture",
			Description: "Learn about zero trust security model, microsegmentation, identity-based access, software-defined perimeter, and ZTNA implementations.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "Zero Trust and Modern Network Security",
					Content: `Zero Trust is a security model that requires strict identity verification for every person and device trying to access resources, regardless of their location.

**Zero Trust Principles:**

  "Never trust, always verify"
  
  Core Tenets:
    1. Verify explicitly: Always authenticate and authorize
    2. Use least-privilege access: Just enough access, just in time
    3. Assume breach: Minimize blast radius, segment access
    
  Traditional Perimeter Security (Castle and Moat):
    Internal network = trusted
    External network = untrusted
    Firewall at the border
    VPN for remote access
    Once inside, broad access

  Zero Trust:
    No implicit trust anywhere
    Every access request verified
    Identity is the new perimeter
    Continuous verification
    Microsegmentation

**Zero Trust Architecture Components:**

Identity Provider (IdP):
  Central authentication service
  Multi-factor authentication (MFA)
  Risk-based authentication
  Continuous identity verification
  Examples: Azure AD, Okta, Google Identity

Policy Engine:
  Determines access decisions
  Inputs: Identity, device health, context, risk score
  Outputs: Allow, deny, step-up authentication
  
  Decision Factors:
    User identity and role
    Device compliance and health
    Location and time
    Application sensitivity
    Behavioral analytics
    Threat intelligence

Policy Enforcement Point (PEP):
  Executes access decisions
  Service proxies, API gateways
  Network enforcement points
  Application-level controls

**Microsegmentation:**

  Dividing network into small, isolated segments
  Granular security policies per workload
  East-west traffic control (lateral movement prevention)
  
  Levels:
    Network-level: VLANs, subnets, firewalls
    Host-level: Host-based firewalls
    Process-level: Application-aware policies
    
  Implementation:
    Software-defined: VMware NSX, Illumio
    Cloud-native: Security Groups, NACLs
    Agent-based: Host firewall rules
    Service mesh: mTLS and authorization policies

**Software-Defined Perimeter (SDP):**

  Also called Black Cloud
  Resources invisible until authenticated
  
  SDP Architecture:
    SDP Controller: Authentication and authorization
    SDP Gateway: Access point to protected resources
    SDP Client: User device with SDP agent
    
  Connection Flow:
    1. Client authenticates with SDP Controller
    2. Controller validates identity, device, context
    3. Controller provisions single-packet authorization (SPA)
    4. Client connects to SDP Gateway
    5. Gateway creates encrypted tunnel to resource
    6. Resources remain invisible to unauthorized users

**ZTNA (Zero Trust Network Access):**

  Replaces traditional VPN
  Application-level access control
  
  ZTNA vs VPN:
    VPN: Network-level access, broad permissions
    ZTNA: Application-level access, granular permissions
    
    VPN: Trust device after VPN connects
    ZTNA: Verify continuously, per-session
    
    VPN: All-or-nothing access
    ZTNA: Least-privilege, per-application
    
  ZTNA Models:
    Client-initiated: Agent on device (SDP model)
    Service-initiated: Reverse proxy/broker
    
  Vendors: Zscaler, Cloudflare Access, Google BeyondCorp

**BeyondCorp (Google's Zero Trust):**

  Pioneer zero trust implementation
  No VPN needed
  All access based on user and device trust
  
  Components:
    Device Inventory: Known/managed devices
    Device Trust: Device health assessment
    Access Proxy: All requests routed through
    Access Control Engine: Policy decisions
    Single Sign-On: Centralized authentication
    
  Access Level Factors:
    Device: Encrypted, managed, patched
    User: Authenticated, authorized role
    Context: Location, time, risk score
    Application: Sensitivity classification

**Network Access Control (NAC):**

  802.1X port-based access control
  Endpoints must authenticate before network access
  
  Components:
    Supplicant: Client device
    Authenticator: Network switch/AP
    Authentication Server: RADIUS
    
  Process:
    1. Device connects to port
    2. Switch holds in unauthorized state
    3. EAP authentication exchange
    4. RADIUS server validates credentials
    5. RADIUS returns VLAN assignment
    6. Switch moves port to authorized state
    
  Posture Assessment:
    Antivirus: Installed and updated?
    OS patches: Up to date?
    Firewall: Enabled?
    Encryption: Disk encrypted?
    MDM: Device managed?
    Non-compliant: Quarantine VLAN

**Mutual TLS (mTLS):**

  Both sides present certificates
  Used in service mesh and API security
  
  Standard TLS:
    Server presents certificate
    Client verifies server identity
    
  mTLS:
    Server presents certificate
    Client presents certificate
    Both verify each other
    
  Implementation:
    Certificate Authority (CA): Issues certs
    SPIFFE/SPIRE: Workload identity framework
    Service mesh: Automatic mTLS (Istio, Linkerd)
    API Gateway: Client certificate validation`,
					CodeExamples: `// Zero trust network implementations

package main

import (
    "crypto/rand"
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "math/big"
    "net"
    "strings"
    "sync"
    "time"
)

// Zero Trust Policy Engine
type ZTPolicyEngine struct {
    policies    []ZTPolicy
    identities  map[string]*Identity
    devices     map[string]*DeviceInfo
    mu          sync.RWMutex
}

type ZTPolicy struct {
    ID          string
    Name        string
    Resource    string
    Conditions  []PolicyCondition
    Action      string // allow, deny, mfa
    Priority    int
}

type PolicyCondition struct {
    Type     string // identity, device, location, time, risk
    Operator string // equals, notEquals, in, notIn, greaterThan, lessThan
    Field    string
    Value    interface{}
}

type Identity struct {
    UserID    string
    Email     string
    Roles     []string
    Groups    []string
    MFAVerified bool
    LastAuth  time.Time
    RiskScore int // 0-100
}

type DeviceInfo struct {
    DeviceID     string
    UserID       string
    Managed      bool
    Compliant    bool
    OSVersion    string
    Encrypted    bool
    LastSeen     time.Time
    TrustScore   int // 0-100
}

type AccessRequest struct {
    UserID    string
    DeviceID  string
    Resource  string
    SourceIP  string
    Timestamp time.Time
}

type AccessDecision struct {
    Allowed     bool
    Action      string
    Reason      string
    PolicyID    string
    TrustLevel  int
}

func NewZTPolicyEngine() *ZTPolicyEngine {
    return &ZTPolicyEngine{
        identities: make(map[string]*Identity),
        devices:    make(map[string]*DeviceInfo),
    }
}

func (pe *ZTPolicyEngine) AddPolicy(policy ZTPolicy) {
    pe.mu.Lock()
    defer pe.mu.Unlock()
    pe.policies = append(pe.policies, policy)
}

func (pe *ZTPolicyEngine) RegisterIdentity(identity *Identity) {
    pe.mu.Lock()
    defer pe.mu.Unlock()
    pe.identities[identity.UserID] = identity
}

func (pe *ZTPolicyEngine) RegisterDevice(device *DeviceInfo) {
    pe.mu.Lock()
    defer pe.mu.Unlock()
    pe.devices[device.DeviceID] = device
}

func (pe *ZTPolicyEngine) Evaluate(req *AccessRequest) *AccessDecision {
    pe.mu.RLock()
    defer pe.mu.RUnlock()
    
    identity, hasIdentity := pe.identities[req.UserID]
    device, hasDevice := pe.devices[req.DeviceID]
    
    // Must have valid identity
    if !hasIdentity {
        return &AccessDecision{
            Allowed: false,
            Action:  "deny",
            Reason:  "Unknown identity",
        }
    }
    
    // Check session expiry
    if time.Since(identity.LastAuth) > 8*time.Hour {
        return &AccessDecision{
            Allowed: false,
            Action:  "reauth",
            Reason:  "Session expired",
        }
    }
    
    // Device trust check
    if hasDevice && !device.Compliant {
        return &AccessDecision{
            Allowed: false,
            Action:  "deny",
            Reason:  "Non-compliant device",
        }
    }
    
    // Evaluate policies
    for _, policy := range pe.policies {
        if !matchResource(policy.Resource, req.Resource) {
            continue
        }
        
        allMatch := true
        for _, cond := range policy.Conditions {
            if !pe.evaluateCondition(cond, identity, device, req) {
                allMatch = false
                break
            }
        }
        
        if allMatch {
            return &AccessDecision{
                Allowed:    policy.Action == "allow",
                Action:     policy.Action,
                Reason:     fmt.Sprintf("Policy: %s", policy.Name),
                PolicyID:   policy.ID,
                TrustLevel: pe.calculateTrust(identity, device),
            }
        }
    }
    
    // Default deny
    return &AccessDecision{
        Allowed: false,
        Action:  "deny",
        Reason:  "No matching policy (default deny)",
    }
}

func matchResource(pattern, resource string) bool {
    if pattern == "*" {
        return true
    }
    if strings.HasSuffix(pattern, "/*") {
        prefix := strings.TrimSuffix(pattern, "/*")
        return strings.HasPrefix(resource, prefix)
    }
    return pattern == resource
}

func (pe *ZTPolicyEngine) evaluateCondition(cond PolicyCondition, identity *Identity, device *DeviceInfo, req *AccessRequest) bool {
    switch cond.Type {
    case "identity":
        return pe.evalIdentityCondition(cond, identity)
    case "device":
        return pe.evalDeviceCondition(cond, device)
    case "location":
        return pe.evalLocationCondition(cond, req.SourceIP)
    case "risk":
        return pe.evalRiskCondition(cond, identity)
    default:
        return false
    }
}

func (pe *ZTPolicyEngine) evalIdentityCondition(cond PolicyCondition, identity *Identity) bool {
    switch cond.Field {
    case "role":
        value := cond.Value.(string)
        for _, role := range identity.Roles {
            if role == value {
                return cond.Operator == "equals" || cond.Operator == "in"
            }
        }
        return cond.Operator == "notEquals" || cond.Operator == "notIn"
    case "mfa":
        return identity.MFAVerified == cond.Value.(bool)
    case "group":
        value := cond.Value.(string)
        for _, group := range identity.Groups {
            if group == value {
                return true
            }
        }
        return false
    }
    return false
}

func (pe *ZTPolicyEngine) evalDeviceCondition(cond PolicyCondition, device *DeviceInfo) bool {
    if device == nil {
        return false
    }
    switch cond.Field {
    case "managed":
        return device.Managed == cond.Value.(bool)
    case "compliant":
        return device.Compliant == cond.Value.(bool)
    case "encrypted":
        return device.Encrypted == cond.Value.(bool)
    case "trust_score":
        threshold := cond.Value.(int)
        if cond.Operator == "greaterThan" {
            return device.TrustScore > threshold
        }
        return device.TrustScore >= threshold
    }
    return false
}

func (pe *ZTPolicyEngine) evalLocationCondition(cond PolicyCondition, sourceIP string) bool {
    if cond.Field == "ip_range" {
        _, cidr, err := net.ParseCIDR(cond.Value.(string))
        if err != nil {
            return false
        }
        ip := net.ParseIP(sourceIP)
        result := cidr.Contains(ip)
        if cond.Operator == "notIn" {
            return !result
        }
        return result
    }
    return false
}

func (pe *ZTPolicyEngine) evalRiskCondition(cond PolicyCondition, identity *Identity) bool {
    if cond.Field == "score" {
        threshold := cond.Value.(int)
        if cond.Operator == "lessThan" {
            return identity.RiskScore < threshold
        }
        return identity.RiskScore <= threshold
    }
    return false
}

func (pe *ZTPolicyEngine) calculateTrust(identity *Identity, device *DeviceInfo) int {
    trust := 0
    
    if identity.MFAVerified {
        trust += 30
    }
    if identity.RiskScore < 20 {
        trust += 20
    }
    
    if device != nil {
        if device.Managed {
            trust += 20
        }
        if device.Compliant {
            trust += 15
        }
        if device.Encrypted {
            trust += 15
        }
    }
    
    if trust > 100 {
        trust = 100
    }
    return trust
}

// Single Packet Authorization (SPA)
type SPAServer struct {
    key       []byte
    allowList map[string]time.Time
    mu        sync.RWMutex
}

type SPAPacket struct {
    Timestamp time.Time
    UserID    string
    SourceIP  string
    Resource  string
    Nonce     string
    HMAC      string
}

func NewSPAServer(key []byte) *SPAServer {
    return &SPAServer{
        key:       key,
        allowList: make(map[string]time.Time),
    }
}

func (s *SPAServer) ValidateSPA(pkt *SPAPacket) bool {
    // Check timestamp freshness (30 second window)
    if time.Since(pkt.Timestamp) > 30*time.Second {
        return false
    }
    
    // Verify HMAC
    data := fmt.Sprintf("%s|%s|%s|%s|%s",
        pkt.Timestamp.Format(time.RFC3339),
        pkt.UserID, pkt.SourceIP, pkt.Resource, pkt.Nonce)
    
    h := sha256.New()
    h.Write(s.key)
    h.Write([]byte(data))
    expectedHMAC := hex.EncodeToString(h.Sum(nil))
    
    if pkt.HMAC != expectedHMAC {
        return false
    }
    
    // Add to allow list with expiry
    s.mu.Lock()
    s.allowList[pkt.SourceIP+":"+pkt.Resource] = time.Now().Add(5 * time.Minute)
    s.mu.Unlock()
    
    return true
}

func (s *SPAServer) IsAllowed(sourceIP, resource string) bool {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    key := sourceIP + ":" + resource
    expiry, exists := s.allowList[key]
    if !exists {
        return false
    }
    return time.Now().Before(expiry)
}

// Generate SPA packet
func GenerateSPAPacket(key []byte, userID, sourceIP, resource string) *SPAPacket {
    nonce := make([]byte, 16)
    rand.Read(nonce)
    
    pkt := &SPAPacket{
        Timestamp: time.Now(),
        UserID:    userID,
        SourceIP:  sourceIP,
        Resource:  resource,
        Nonce:     hex.EncodeToString(nonce),
    }
    
    data := fmt.Sprintf("%s|%s|%s|%s|%s",
        pkt.Timestamp.Format(time.RFC3339),
        pkt.UserID, pkt.SourceIP, pkt.Resource, pkt.Nonce)
    
    h := sha256.New()
    h.Write(key)
    h.Write([]byte(data))
    pkt.HMAC = hex.EncodeToString(h.Sum(nil))
    
    return pkt
}

// Microsegmentation policy manager
type MicrosegManager struct {
    workloads map[string]*Workload
    policies  []MicrosegPolicy
    mu        sync.RWMutex
}

type Workload struct {
    ID       string
    Name     string
    Labels   map[string]string
    IP       string
    Ports    []int
    Zone     string
}

type MicrosegPolicy struct {
    Name     string
    Source   LabelSelector
    Dest     LabelSelector
    Ports    []int
    Protocol string
    Action   string // allow, deny
}

type LabelSelector struct {
    MatchLabels map[string]string
}

func NewMicrosegManager() *MicrosegManager {
    return &MicrosegManager{
        workloads: make(map[string]*Workload),
    }
}

func (m *MicrosegManager) RegisterWorkload(w *Workload) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.workloads[w.ID] = w
}

func (m *MicrosegManager) AddPolicy(policy MicrosegPolicy) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.policies = append(m.policies, policy)
}

func (m *MicrosegManager) IsAllowed(srcID, dstID string, port int, protocol string) bool {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    src, srcExists := m.workloads[srcID]
    dst, dstExists := m.workloads[dstID]
    if !srcExists || !dstExists {
        return false // Default deny for unknown workloads
    }
    
    for _, policy := range m.policies {
        if !matchLabels(src.Labels, policy.Source.MatchLabels) {
            continue
        }
        if !matchLabels(dst.Labels, policy.Dest.MatchLabels) {
            continue
        }
        if protocol != "" && policy.Protocol != "" && policy.Protocol != protocol {
            continue
        }
        
        portMatch := len(policy.Ports) == 0 // Empty means all ports
        for _, p := range policy.Ports {
            if p == port {
                portMatch = true
                break
            }
        }
        
        if portMatch {
            return policy.Action == "allow"
        }
    }
    
    return false // Default deny
}

func matchLabels(workloadLabels, selectorLabels map[string]string) bool {
    for key, value := range selectorLabels {
        if workloadLabels[key] != value {
            return false
        }
    }
    return true
}`,
				},
			},
		},
	})
}
