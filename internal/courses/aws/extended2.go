package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2116,
			Title:       "AWS Networking Deep Dive",
			Description: "Master AWS VPC architecture, Transit Gateway, PrivateLink, Direct Connect, Route 53, and advanced networking patterns.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "VPC Advanced Architecture and Connectivity",
					Content: `AWS networking centers around Virtual Private Clouds with a rich set of connectivity options for complex enterprise architectures.

**VPC Architecture:**

VPC Components:
  CIDR Block: Primary and secondary (/16 to /28)
  Subnets: AZ-scoped, public or private
  Route Tables: Per-subnet routing rules
  Internet Gateway (IGW): Public internet access
  NAT Gateway: Outbound for private subnets
  Elastic IP: Static public IPv4 address
  
  VPC CIDR Planning:
    VPC: 10.0.0.0/16 (65,534 usable IPs)
    AWS reserves 5 IPs per subnet:
      .0: Network address
      .1: VPC router
      .2: DNS server
      .3: Future use
      .255: Broadcast (not supported but reserved)
    
    Subnet design (3-tier):
      Public: 10.0.0.0/20, 10.0.16.0/20 (4094 hosts each)
      Private: 10.0.32.0/20, 10.0.48.0/20
      Data: 10.0.64.0/20, 10.0.80.0/20

  Route Table Rules:
    Local route: 10.0.0.0/16 -> local (always present)
    Internet: 0.0.0.0/0 -> igw-xxx (public subnets)
    NAT: 0.0.0.0/0 -> nat-xxx (private subnets)
    Peering: 10.1.0.0/16 -> pcx-xxx
    TGW: 10.0.0.0/8 -> tgw-xxx
    VPN: 192.168.0.0/16 -> vgw-xxx

**Security Groups vs NACLs:**

Security Groups (SG):
  Stateful (return traffic auto-allowed)
  Instance-level
  Allow rules only (no deny)
  Evaluates all rules before deciding
  Can reference other SGs as source/destination
  
  Example SG Rules:
    Inbound:
      Type: HTTPS, Port: 443, Source: 0.0.0.0/0
      Type: SSH, Port: 22, Source: sg-bastion
      Type: Custom TCP, Port: 8080, Source: sg-alb
    Outbound:
      Type: All Traffic, Port: All, Dest: 0.0.0.0/0

Network ACLs (NACL):
  Stateless (must allow both inbound and outbound)
  Subnet-level
  Allow and deny rules
  Processes rules in number order (first match)
  Default NACL allows all traffic
  
  Example NACL:
    Inbound:
      100: Allow TCP 443 from 0.0.0.0/0
      110: Allow TCP 80 from 0.0.0.0/0
      120: Allow TCP 22 from 10.0.0.0/8
      130: Allow TCP 1024-65535 from 0.0.0.0/0 (ephemeral)
      *: Deny All
    Outbound:
      100: Allow TCP 443 to 0.0.0.0/0
      110: Allow TCP 1024-65535 to 0.0.0.0/0 (ephemeral)
      *: Deny All

**VPC Peering:**
  
  Direct network connection between two VPCs
  Can cross regions and accounts
  Not transitive
  No overlapping CIDR
  
  Setup:
    1. Requester creates peering request
    2. Accepter accepts request
    3. Both sides add route table entries
    4. Update security groups to allow traffic

**Transit Gateway (TGW):**

  Hub-and-spoke connectivity
  Connect thousands of VPCs
  Transitive routing
  
  Features:
    Route tables per attachment
    Route propagation
    Inter-region peering
    Multicast support
    Network Manager integration
    Equal-cost multipath (ECMP)
    
  Architecture:
    Production Route Table:
      VPC-Prod-A: 10.1.0.0/16
      VPC-Prod-B: 10.2.0.0/16
      Shared Services: 10.100.0.0/16
      0.0.0.0/0 -> VPN/DX attachment
      
    Development Route Table:
      VPC-Dev-A: 10.10.0.0/16
      VPC-Dev-B: 10.11.0.0/16
      Shared Services: 10.100.0.0/16
      (No route to production)

  Transit Gateway Network Manager:
    Global network visualization
    Topology map
    Event notifications
    Route analysis

**AWS PrivateLink:**

  Access services without internet
  Uses ENI in your VPC
  
  Interface VPC Endpoints:
    Creates ENI with private IP
    Powered by PrivateLink
    Per-AZ pricing
    Supports 100+ AWS services
    
  Gateway VPC Endpoints:
    Route table entry (no ENI)
    Free
    S3 and DynamoDB only
    Regional scope
    
  Endpoint Services (Custom):
    Expose your NLB-backed service
    Others connect via interface endpoint
    Cross-account and cross-region
    No internet exposure

**AWS Direct Connect:**

  Dedicated network link to AWS
  1 Gbps or 10 Gbps dedicated
  50/100/200/300/400/500 Mbps or 1/2/5/10 Gbps hosted
  
  Virtual Interfaces (VIF):
    Private VIF: Access VPC (private IP)
    Public VIF: Access AWS public services
    Transit VIF: Access via Transit Gateway
    
  Components:
    DX Location: Colocation facility
    DX Gateway: Global resource for cross-region
    Router: Customer's BGP router
    LAG: Link Aggregation Group
    
  Resilience:
    Single DX: No redundancy
    Two connections same location: Location redundancy
    Two connections different locations: High resilience
    Maximum resilience: Two locations, two connections each

**Route 53:**

  AWS DNS service
  
  Routing Policies:
    Simple: Single resource
    Weighted: Percentage-based routing (A/B testing)
    Latency: Lowest latency region
    Failover: Active-passive
    Geolocation: Based on user location
    Geoproximity: Based on geographic distance with bias
    Multivalue Answer: Multiple healthy records
    
  Health Checks:
    HTTP/HTTPS/TCP checks
    CloudWatch alarm-based
    Calculated (aggregate multiple checks)
    10 or 30 second intervals
    String matching in response
    
  DNS Firewall:
    Block malicious domains
    Managed domain lists
    Custom allow/deny lists
    Query logging

**VPC Flow Logs:**

  Capture IP traffic metadata
  Subnet, VPC, or ENI level
  Publish to CloudWatch Logs, S3, Kinesis
  
  Flow Log Fields:
    version, account-id, interface-id
    srcaddr, dstaddr, srcport, dstport
    protocol, packets, bytes
    start, end, action (ACCEPT/REJECT), log-status
    
  Analysis:
    Athena queries on S3 data
    CloudWatch Insights
    Security analysis
    Troubleshooting connectivity`,
					CodeExamples: `// AWS networking implementations

package main

import (
    "fmt"
    "net"
    "sort"
    "strings"
    "sync"
    "time"
)

// VPC manager
type AWSVPCManager struct {
    vpcs          map[string]*AWSVPC
    peerings      map[string]*AWSPeering
    transitGW     map[string]*AWSTransitGW
    endpoints     map[string]*VPCEndpoint
    mu            sync.RWMutex
}

type AWSVPC struct {
    ID              string
    Name            string
    CIDR            string
    SecondaryCIDRs  []string
    Region          string
    Subnets         []*AWSSubnet
    RouteTables     []*AWSRouteTable
    SecurityGroups  []*AWSSecurityGroup
    NACLs           []*AWSNetworkACL
    FlowLogsEnabled bool
}

type AWSSubnet struct {
    ID            string
    Name          string
    CIDR          string
    AZ            string
    Public        bool
    RouteTableID  string
    AvailableIPs  int
}

type AWSRouteTable struct {
    ID     string
    Name   string
    Main   bool
    Routes []AWSRoute
}

type AWSRoute struct {
    Destination string
    Target      string
    State       string
    Origin      string // CreateRoute, EnableVpcPeering, etc.
}

type AWSSecurityGroup struct {
    ID          string
    Name        string
    Description string
    VpcID       string
    Ingress     []AWSSecurityGroupRule
    Egress      []AWSSecurityGroupRule
}

type AWSSecurityGroupRule struct {
    Protocol  string
    FromPort  int
    ToPort    int
    CIDR      string
    SGRef     string // Reference to another security group
    Desc      string
}

type AWSNetworkACL struct {
    ID      string
    Default bool
    Rules   []AWSNACLRule
}

type AWSNACLRule struct {
    RuleNumber int
    Protocol   string
    Action     string // allow, deny
    CIDR       string
    FromPort   int
    ToPort     int
    Egress     bool
}

func NewAWSVPCManager() *AWSVPCManager {
    return &AWSVPCManager{
        vpcs:      make(map[string]*AWSVPC),
        peerings:  make(map[string]*AWSPeering),
        transitGW: make(map[string]*AWSTransitGW),
        endpoints: make(map[string]*VPCEndpoint),
    }
}

func (m *AWSVPCManager) CreateVPC(name, cidr, region string) (*AWSVPC, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    _, _, err := net.ParseCIDR(cidr)
    if err != nil {
        return nil, fmt.Errorf("invalid CIDR: %w", err)
    }
    
    vpc := &AWSVPC{
        ID:     fmt.Sprintf("vpc-%s", generateID()),
        Name:   name,
        CIDR:   cidr,
        Region: region,
        RouteTables: []*AWSRouteTable{
            {
                ID:   fmt.Sprintf("rtb-%s", generateID()),
                Name: "main",
                Main: true,
                Routes: []AWSRoute{
                    {Destination: cidr, Target: "local", State: "active", Origin: "CreateRouteTable"},
                },
            },
        },
        SecurityGroups: []*AWSSecurityGroup{
            {
                ID:   fmt.Sprintf("sg-%s", generateID()),
                Name: "default",
                Egress: []AWSSecurityGroupRule{
                    {Protocol: "-1", FromPort: 0, ToPort: 0, CIDR: "0.0.0.0/0", Desc: "Allow all outbound"},
                },
            },
        },
    }
    
    m.vpcs[vpc.ID] = vpc
    return vpc, nil
}

func (m *AWSVPCManager) CreateSubnet(vpcID, name, cidr, az string, public bool) (*AWSSubnet, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    vpc, exists := m.vpcs[vpcID]
    if !exists {
        return nil, fmt.Errorf("VPC %s not found", vpcID)
    }
    
    // Calculate available IPs (total - 5 reserved)
    _, ipNet, _ := net.ParseCIDR(cidr)
    ones, bits := ipNet.Mask.Size()
    totalIPs := 1 << (bits - ones)
    availableIPs := totalIPs - 5
    
    subnet := &AWSSubnet{
        ID:           fmt.Sprintf("subnet-%s", generateID()),
        Name:         name,
        CIDR:         cidr,
        AZ:           az,
        Public:       public,
        RouteTableID: vpc.RouteTables[0].ID,
        AvailableIPs: availableIPs,
    }
    
    vpc.Subnets = append(vpc.Subnets, subnet)
    return subnet, nil
}

// Security group evaluation
func (m *AWSVPCManager) EvaluateIngress(sgID, protocol string, port int, sourceIP string) bool {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    for _, vpc := range m.vpcs {
        for _, sg := range vpc.SecurityGroups {
            if sg.ID != sgID {
                continue
            }
            for _, rule := range sg.Ingress {
                if !matchSGProtocol(rule.Protocol, protocol) {
                    continue
                }
                if rule.Protocol != "-1" && (port < rule.FromPort || port > rule.ToPort) {
                    continue
                }
                if rule.CIDR != "" {
                    _, cidrNet, err := net.ParseCIDR(rule.CIDR)
                    if err != nil {
                        continue
                    }
                    if cidrNet.Contains(net.ParseIP(sourceIP)) {
                        return true
                    }
                }
            }
            return false
        }
    }
    return false
}

func matchSGProtocol(rule, proto string) bool {
    if rule == "-1" {
        return true
    }
    return strings.EqualFold(rule, proto)
}

// NACL evaluation (ordered rules, first match)
func (m *AWSVPCManager) EvaluateNACL(naclID string, protocol string, port int, sourceIP string, egress bool) bool {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    for _, vpc := range m.vpcs {
        for _, nacl := range vpc.NACLs {
            if nacl.ID != naclID {
                continue
            }
            
            var rules []AWSNACLRule
            for _, r := range nacl.Rules {
                if r.Egress == egress {
                    rules = append(rules, r)
                }
            }
            
            sort.Slice(rules, func(i, j int) bool {
                return rules[i].RuleNumber < rules[j].RuleNumber
            })
            
            for _, rule := range rules {
                if rule.Protocol != "-1" && !strings.EqualFold(rule.Protocol, protocol) {
                    continue
                }
                if rule.Protocol != "-1" && (port < rule.FromPort || port > rule.ToPort) {
                    continue
                }
                if rule.CIDR != "" {
                    _, cidrNet, err := net.ParseCIDR(rule.CIDR)
                    if err != nil {
                        continue
                    }
                    if !cidrNet.Contains(net.ParseIP(sourceIP)) {
                        continue
                    }
                }
                return rule.Action == "allow"
            }
            return false // Default deny
        }
    }
    return false
}

// VPC Peering
type AWSPeering struct {
    ID        string
    Requester string
    Accepter  string
    Status    string
    Region    string
}

// Transit Gateway
type AWSTransitGW struct {
    ID           string
    Name         string
    ASN          uint32
    Attachments  []TGWAttach
    RouteTables  []*TGWRouteTab
}

type TGWAttach struct {
    ID     string
    Type   string // vpc, vpn, direct-connect, peering, connect
    Target string
    State  string
}

type TGWRouteTab struct {
    ID     string
    Name   string
    Routes []TGWRt
}

type TGWRt struct {
    Destination  string
    AttachmentID string
    Type         string // static, propagated
    State        string
}

// VPC Endpoint
type VPCEndpoint struct {
    ID          string
    Type        string // Interface, Gateway
    ServiceName string
    VpcID       string
    SubnetIDs   []string
    SGIDs       []string
    PolicyDoc   string
    State       string
}

// Route 53 resolver
type Route53Resolver struct {
    zones     map[string]*HostedZone
    mu        sync.RWMutex
}

type HostedZone struct {
    ID      string
    Name    string
    Private bool
    VpcIDs  []string
    Records []*Route53Record
}

type Route53Record struct {
    Name    string
    Type    string // A, AAAA, CNAME, MX, TXT, NS, etc.
    TTL     int
    Values  []string
    AliasTarget *AliasTarget
    Policy  *RoutingPolicy
    HealthCheckID string
    SetID   string
}

type AliasTarget struct {
    DNSName    string
    HostedZone string
}

type RoutingPolicy struct {
    Type      string // simple, weighted, latency, failover, geolocation, multivalue
    Weight    int
    Region    string
    Failover  string // PRIMARY, SECONDARY
    GeoLocation string
}

func NewRoute53Resolver() *Route53Resolver {
    return &Route53Resolver{
        zones: make(map[string]*HostedZone),
    }
}

func (r *Route53Resolver) CreateZone(name string, private bool) *HostedZone {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    zone := &HostedZone{
        ID:      fmt.Sprintf("Z%s", generateID()),
        Name:    name,
        Private: private,
    }
    r.zones[zone.ID] = zone
    return zone
}

func (r *Route53Resolver) AddRecord(zoneID string, record *Route53Record) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    zone, exists := r.zones[zoneID]
    if !exists {
        return fmt.Errorf("zone %s not found", zoneID)
    }
    
    zone.Records = append(zone.Records, record)
    return nil
}

func (r *Route53Resolver) Resolve(name, recordType string) []string {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    for _, zone := range r.zones {
        if !strings.HasSuffix(name, zone.Name) && name+"." != zone.Name {
            continue
        }
        
        var results []string
        for _, record := range zone.Records {
            if record.Name == name && record.Type == recordType {
                if record.AliasTarget != nil {
                    results = append(results, record.AliasTarget.DNSName)
                } else {
                    results = append(results, record.Values...)
                }
            }
        }
        if len(results) > 0 {
            return results
        }
    }
    return nil
}

// VPC Flow Log analyzer
type FlowLogAnalyzer struct {
    flows []VPCFlowLog
    mu    sync.RWMutex
}

type VPCFlowLog struct {
    Version     int
    AccountID   string
    InterfaceID string
    SrcAddr     string
    DstAddr     string
    SrcPort     int
    DstPort     int
    Protocol    int
    Packets     int64
    Bytes       int64
    Start       time.Time
    End         time.Time
    Action      string // ACCEPT, REJECT
    LogStatus   string
}

func NewFlowLogAnalyzer() *FlowLogAnalyzer {
    return &FlowLogAnalyzer{}
}

func (a *FlowLogAnalyzer) Ingest(flow VPCFlowLog) {
    a.mu.Lock()
    defer a.mu.Unlock()
    a.flows = append(a.flows, flow)
}

func (a *FlowLogAnalyzer) GetRejectedFlows(since time.Time) []VPCFlowLog {
    a.mu.RLock()
    defer a.mu.RUnlock()
    
    var rejected []VPCFlowLog
    for _, f := range a.flows {
        if f.Action == "REJECT" && f.Start.After(since) {
            rejected = append(rejected, f)
        }
    }
    return rejected
}

func (a *FlowLogAnalyzer) GetTopTalkers(n int) []FlowSummary {
    a.mu.RLock()
    defer a.mu.RUnlock()
    
    byIP := make(map[string]int64)
    for _, f := range a.flows {
        byIP[f.SrcAddr] += f.Bytes
    }
    
    summaries := make([]FlowSummary, 0, len(byIP))
    for ip, bytes := range byIP {
        summaries = append(summaries, FlowSummary{IP: ip, TotalBytes: bytes})
    }
    
    sort.Slice(summaries, func(i, j int) bool {
        return summaries[i].TotalBytes > summaries[j].TotalBytes
    })
    
    if n > len(summaries) {
        n = len(summaries)
    }
    return summaries[:n]
}

type FlowSummary struct {
    IP         string
    TotalBytes int64
}

func generateID() string {
    return fmt.Sprintf("%08x", time.Now().UnixNano()%0xFFFFFFFF)
}`,
				},
			},
		},
	})
}
