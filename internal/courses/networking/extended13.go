package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2633,
			Title:       "Network Observability and Monitoring",
			Description: "Learn about SNMP, NetFlow, sFlow, network telemetry, distributed tracing, network metrics, and observability platforms.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "Network Monitoring Telemetry and Observability",
					Content: `Network observability provides visibility into network behavior through metrics, logs, and traces.

**SNMP (Simple Network Management Protocol):**

Versions:
  SNMPv1: Community string auth, plaintext (deprecated)
  SNMPv2c: Community string, bulk operations (common)
  SNMPv3: Username/password, encryption, auth (recommended)

Components:
  Manager: Monitoring station (polls agents)
  Agent: Runs on network device
  MIB: Management Information Base (data structure)
  OID: Object Identifier (unique data point)

Operations:
  GET: Request specific OID value
  GETNEXT: Request next OID in tree
  GETBULK: Request many OIDs at once (v2c+)
  SET: Modify OID value
  TRAP: Unsolicited notification from agent
  INFORM: Acknowledged trap (v2c+)

Common OIDs:
  .1.3.6.1.2.1.1.1.0: sysDescr (system description)
  .1.3.6.1.2.1.1.3.0: sysUptime
  .1.3.6.1.2.1.2.2.1.10: ifInOctets (interface bytes in)
  .1.3.6.1.2.1.2.2.1.16: ifOutOctets (interface bytes out)
  .1.3.6.1.4.1.9.9.109.1.1.1.1.8: cpmCPUTotal5minRev (Cisco CPU)

SNMPv3 Security:
  Authentication: MD5, SHA, SHA-256
  Privacy (Encryption): DES, AES-128, AES-256
  Security Levels:
    noAuthNoPriv: No authentication or encryption
    authNoPriv: Authentication only
    authPriv: Authentication and encryption

**NetFlow / IPFIX:**

  Cisco NetFlow:
    Per-flow traffic statistics
    Exported from routers/switches
    Flow = unique 5-tuple (src IP, dst IP, src port, dst port, protocol)
    
    Versions:
      v5: Fixed format, IPv4 only
      v9: Template-based, flexible, IPv6
    
    Flow Record Contents:
      Source/destination IP and port
      Protocol
      Bytes and packets
      Start/end timestamps
      Input/output interface
      TCP flags
      ToS/DSCP
      AS numbers

  IPFIX (IP Flow Information Export):
    IETF standard based on NetFlow v9
    Template-based
    Variable-length fields
    Enterprise-specific elements
    SCTP, TCP, or UDP transport

  sFlow:
    Sampling-based (not every packet)
    1-in-N packet sampling
    Counter polling
    Lower overhead than NetFlow
    Real-time visibility
    Multi-vendor support

Use Cases:
  Traffic engineering
  Capacity planning
  Security analysis (DDoS detection)
  Billing/accounting
  Application performance monitoring
  Anomaly detection

**Network Telemetry:**

Model-Driven Telemetry (MDT):
  Push-based (device pushes data)
  Structured data (YANG models)
  gRPC or gNMI transport
  Sub-second intervals possible
  Replaces SNMP polling
  
  gNMI (gRPC Network Management Interface):
    Subscribe: Streaming telemetry
    Get: One-time query
    Set: Configuration changes
    Capabilities: Discover supported models
    
  YANG (Yet Another Next Generation):
    Data modeling language
    Defines configuration and state
    Vendor-neutral models (OpenConfig)
    Vendor-specific models (Cisco, Juniper)

Streaming Telemetry Pipeline:
  Network Device -> Collector -> Message Queue -> TSDB -> Dashboard
  
  Components:
    Telegraf: Collector/agent
    Kafka: Message queue/buffer
    InfluxDB/Prometheus: Time-series database
    Grafana: Visualization

**Network Metrics:**

Bandwidth Utilization:
  Bits per second (bps)
  95th percentile billing
  Peak vs average
  Per-interface, per-link

Latency:
  Round-trip time (RTT)
  One-way delay
  TWAMP (Two-Way Active Measurement Protocol)
  SLA monitoring

Packet Loss:
  Loss rate percentage
  Burst vs random
  Forward vs reverse path

Jitter:
  Packet delay variation
  IP Delay Variation (IPDV)
  Mean, max, 99th percentile

Error Rates:
  CRC errors
  Frame errors
  Input/output errors
  Discards (queue overflow)

**Distributed Tracing for Networks:**

  End-to-end request path visualization
  Correlate network and application issues
  
  In-Band Network Telemetry (INT):
    Network devices add metadata to packets
    Hop-by-hop latency
    Queue depth visibility
    Drop reasons
    Path verification
    
  Packet Path Tracing:
    traceroute / tracepath
    MTR (My TraceRoute)
    Paris traceroute (consistent paths)
    
  Service-Level Tracing:
    OpenTelemetry (OTel)
    Zipkin, Jaeger
    Correlate network spans with app spans

**Log Management:**

Syslog:
  Standard log protocol
  UDP port 514, TCP port 514, TLS port 6514
  
  Severity Levels:
    0: Emergency
    1: Alert
    2: Critical
    3: Error
    4: Warning
    5: Notice
    6: Informational
    7: Debug
    
  Facilities: kern, user, daemon, local0-local7
  
  Modern syslog (RFC 5424):
    Structured data
    Message ID
    Timestamp with timezone
    UTF-8 support

Network Log Sources:
  Firewall logs (connections, blocks)
  Authentication logs (802.1X, VPN)
  Routing protocol logs (BGP state changes)
  Interface state changes
  Configuration changes
  DNS query logs
  DHCP lease logs

**Alerting and Incident Response:**

Alert Categories:
  Connectivity: Device unreachable, link down
  Performance: High utilization, latency threshold
  Security: Port scan, DDoS, unauthorized access
  Configuration: Unexpected changes, drift
  Capacity: Approaching limits

Alert Management:
  Deduplication: Group related alerts
  Correlation: Link root cause to symptoms
  Escalation: Time-based escalation
  Suppression: Maintenance windows
  Runbooks: Automated response procedures

SLA Monitoring:
  Availability SLA: 99.99% = 52.56 min/year downtime
  Latency SLA: < 50ms average
  Packet Loss SLA: < 0.1%
  MTBF: Mean Time Between Failures
  MTTR: Mean Time To Repair`,
					CodeExamples: `// Network monitoring and observability implementations

package main

import (
    "fmt"
    "math"
    "sort"
    "strings"
    "sync"
    "time"
)

// SNMP poller
type SNMPPoller struct {
    targets    map[string]*SNMPTarget
    oids       []string
    interval   time.Duration
    results    map[string][]SNMPResult
    mu         sync.RWMutex
}

type SNMPTarget struct {
    Address    string
    Community  string
    Version    int // 2 or 3
    Port       int
    AuthUser   string
    AuthPass   string
    PrivPass   string
    AuthProto  string // MD5, SHA
    PrivProto  string // AES, DES
}

type SNMPResult struct {
    OID       string
    Value     interface{}
    Type      string // integer, string, counter, gauge
    Timestamp time.Time
    Error     string
}

func NewSNMPPoller(interval time.Duration) *SNMPPoller {
    return &SNMPPoller{
        targets:  make(map[string]*SNMPTarget),
        results:  make(map[string][]SNMPResult),
        interval: interval,
    }
}

func (p *SNMPPoller) AddTarget(name string, target *SNMPTarget) {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.targets[name] = target
}

func (p *SNMPPoller) AddOIDs(oids ...string) {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.oids = append(p.oids, oids...)
}

func (p *SNMPPoller) RecordResult(targetName string, result SNMPResult) {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.results[targetName] = append(p.results[targetName], result)
    
    // Keep last 1000 results per target
    if len(p.results[targetName]) > 1000 {
        p.results[targetName] = p.results[targetName][len(p.results[targetName])-1000:]
    }
}

func (p *SNMPPoller) GetLatest(targetName, oid string) *SNMPResult {
    p.mu.RLock()
    defer p.mu.RUnlock()
    
    results := p.results[targetName]
    for i := len(results) - 1; i >= 0; i-- {
        if results[i].OID == oid {
            return &results[i]
        }
    }
    return nil
}

// NetFlow collector
type NetFlowCollector struct {
    flows     map[string]*FlowRecord
    completed []FlowRecord
    mu        sync.RWMutex
    topTalkers *TopTalkersTracker
}

type FlowRecord struct {
    SrcIP      string
    DstIP      string
    SrcPort    uint16
    DstPort    uint16
    Protocol   uint8
    Bytes      uint64
    Packets    uint64
    StartTime  time.Time
    EndTime    time.Time
    TCPFlags   uint8
    InputIface int
    OutputIface int
    ToS        uint8
    SrcAS      uint32
    DstAS      uint32
}

func (f *FlowRecord) Key() string {
    return fmt.Sprintf("%s:%d-%s:%d-%d",
        f.SrcIP, f.SrcPort, f.DstIP, f.DstPort, f.Protocol)
}

func (f *FlowRecord) Duration() time.Duration {
    return f.EndTime.Sub(f.StartTime)
}

func (f *FlowRecord) BitsPerSecond() float64 {
    duration := f.Duration().Seconds()
    if duration == 0 {
        return 0
    }
    return float64(f.Bytes*8) / duration
}

func NewNetFlowCollector() *NetFlowCollector {
    return &NetFlowCollector{
        flows:      make(map[string]*FlowRecord),
        topTalkers: NewTopTalkersTracker(10),
    }
}

func (c *NetFlowCollector) IngestFlow(flow FlowRecord) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    key := flow.Key()
    existing, exists := c.flows[key]
    
    if exists {
        existing.Bytes += flow.Bytes
        existing.Packets += flow.Packets
        existing.EndTime = flow.EndTime
        existing.TCPFlags |= flow.TCPFlags
    } else {
        c.flows[key] = &flow
    }
    
    c.topTalkers.Update(flow.SrcIP, flow.Bytes)
}

func (c *NetFlowCollector) ExpireFlows(maxAge time.Duration) int {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    now := time.Now()
    expired := 0
    
    for key, flow := range c.flows {
        if now.Sub(flow.EndTime) > maxAge {
            c.completed = append(c.completed, *flow)
            delete(c.flows, key)
            expired++
        }
    }
    
    return expired
}

func (c *NetFlowCollector) GetActiveFlowCount() int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return len(c.flows)
}

func (c *NetFlowCollector) GetTopFlows(n int) []FlowRecord {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    flows := make([]FlowRecord, 0, len(c.flows))
    for _, f := range c.flows {
        flows = append(flows, *f)
    }
    
    sort.Slice(flows, func(i, j int) bool {
        return flows[i].Bytes > flows[j].Bytes
    })
    
    if n > len(flows) {
        n = len(flows)
    }
    return flows[:n]
}

// Top talkers tracker
type TopTalkersTracker struct {
    talkers  map[string]uint64
    topN     int
    mu       sync.RWMutex
}

func NewTopTalkersTracker(topN int) *TopTalkersTracker {
    return &TopTalkersTracker{
        talkers: make(map[string]uint64),
        topN:    topN,
    }
}

func (t *TopTalkersTracker) Update(ip string, bytes uint64) {
    t.mu.Lock()
    defer t.mu.Unlock()
    t.talkers[ip] += bytes
}

type TalkerEntry struct {
    IP    string
    Bytes uint64
}

func (t *TopTalkersTracker) GetTop() []TalkerEntry {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    entries := make([]TalkerEntry, 0, len(t.talkers))
    for ip, bytes := range t.talkers {
        entries = append(entries, TalkerEntry{IP: ip, Bytes: bytes})
    }
    
    sort.Slice(entries, func(i, j int) bool {
        return entries[i].Bytes > entries[j].Bytes
    })
    
    if t.topN < len(entries) {
        return entries[:t.topN]
    }
    return entries
}

// Network latency monitor
type LatencyMonitor struct {
    targets   map[string]*LatencyTarget
    mu        sync.RWMutex
}

type LatencyTarget struct {
    Address    string
    Name       string
    Samples    []LatencySample
    MaxSamples int
}

type LatencySample struct {
    RTT       time.Duration
    Loss      bool
    Timestamp time.Time
}

type LatencyStats struct {
    Min     time.Duration
    Max     time.Duration
    Avg     time.Duration
    P50     time.Duration
    P95     time.Duration
    P99     time.Duration
    StdDev  time.Duration
    Loss    float64
    Samples int
}

func NewLatencyMonitor() *LatencyMonitor {
    return &LatencyMonitor{
        targets: make(map[string]*LatencyTarget),
    }
}

func (m *LatencyMonitor) AddTarget(name, address string) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.targets[name] = &LatencyTarget{
        Address:    address,
        Name:       name,
        MaxSamples: 1000,
    }
}

func (m *LatencyMonitor) RecordSample(name string, rtt time.Duration, loss bool) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    target, exists := m.targets[name]
    if !exists {
        return
    }
    
    target.Samples = append(target.Samples, LatencySample{
        RTT:       rtt,
        Loss:      loss,
        Timestamp: time.Now(),
    })
    
    if len(target.Samples) > target.MaxSamples {
        target.Samples = target.Samples[len(target.Samples)-target.MaxSamples:]
    }
}

func (m *LatencyMonitor) GetStats(name string, window time.Duration) *LatencyStats {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    target, exists := m.targets[name]
    if !exists {
        return nil
    }
    
    cutoff := time.Now().Add(-window)
    var rtts []time.Duration
    lossCount := 0
    totalCount := 0
    
    for _, sample := range target.Samples {
        if sample.Timestamp.Before(cutoff) {
            continue
        }
        totalCount++
        if sample.Loss {
            lossCount++
            continue
        }
        rtts = append(rtts, sample.RTT)
    }
    
    if len(rtts) == 0 {
        return &LatencyStats{
            Loss:    float64(lossCount) / float64(totalCount) * 100,
            Samples: totalCount,
        }
    }
    
    sort.Slice(rtts, func(i, j int) bool {
        return rtts[i] < rtts[j]
    })
    
    var sum time.Duration
    for _, rtt := range rtts {
        sum += rtt
    }
    avg := sum / time.Duration(len(rtts))
    
    // Calculate std dev
    var variance float64
    for _, rtt := range rtts {
        diff := float64(rtt - avg)
        variance += diff * diff
    }
    variance /= float64(len(rtts))
    stdDev := time.Duration(math.Sqrt(variance))
    
    return &LatencyStats{
        Min:     rtts[0],
        Max:     rtts[len(rtts)-1],
        Avg:     avg,
        P50:     percentile(rtts, 50),
        P95:     percentile(rtts, 95),
        P99:     percentile(rtts, 99),
        StdDev:  stdDev,
        Loss:    float64(lossCount) / float64(totalCount) * 100,
        Samples: totalCount,
    }
}

func percentile(sorted []time.Duration, p int) time.Duration {
    if len(sorted) == 0 {
        return 0
    }
    idx := int(float64(len(sorted)-1) * float64(p) / 100.0)
    return sorted[idx]
}

// SLA monitor
type SLAMonitor struct {
    slas    map[string]*SLADefinition
    mu      sync.RWMutex
}

type SLADefinition struct {
    Name            string
    TargetAvailability float64 // percentage, e.g., 99.99
    TargetLatency   time.Duration
    TargetLoss      float64 // percentage
    Measurements    []SLAMeasurement
    Violations      []SLAViolation
}

type SLAMeasurement struct {
    Timestamp    time.Time
    Available    bool
    Latency      time.Duration
    PacketLoss   float64
}

type SLAViolation struct {
    Timestamp time.Time
    Type      string // availability, latency, loss
    Value     float64
    Threshold float64
    Duration  time.Duration
}

func NewSLAMonitor() *SLAMonitor {
    return &SLAMonitor{
        slas: make(map[string]*SLADefinition),
    }
}

func (m *SLAMonitor) DefineSLA(sla SLADefinition) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.slas[sla.Name] = &sla
}

func (m *SLAMonitor) RecordMeasurement(slaName string, measurement SLAMeasurement) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    sla, exists := m.slas[slaName]
    if !exists {
        return
    }
    
    sla.Measurements = append(sla.Measurements, measurement)
    
    // Check for violations
    if !measurement.Available {
        sla.Violations = append(sla.Violations, SLAViolation{
            Timestamp: measurement.Timestamp,
            Type:      "availability",
            Value:     0,
            Threshold: sla.TargetAvailability,
        })
    }
    
    if measurement.Latency > sla.TargetLatency {
        sla.Violations = append(sla.Violations, SLAViolation{
            Timestamp: measurement.Timestamp,
            Type:      "latency",
            Value:     float64(measurement.Latency.Milliseconds()),
            Threshold: float64(sla.TargetLatency.Milliseconds()),
        })
    }
    
    if measurement.PacketLoss > sla.TargetLoss {
        sla.Violations = append(sla.Violations, SLAViolation{
            Timestamp: measurement.Timestamp,
            Type:      "loss",
            Value:     measurement.PacketLoss,
            Threshold: sla.TargetLoss,
        })
    }
}

func (m *SLAMonitor) GetAvailability(slaName string, period time.Duration) float64 {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    sla, exists := m.slas[slaName]
    if !exists {
        return 0
    }
    
    cutoff := time.Now().Add(-period)
    total := 0
    available := 0
    
    for _, meas := range sla.Measurements {
        if meas.Timestamp.Before(cutoff) {
            continue
        }
        total++
        if meas.Available {
            available++
        }
    }
    
    if total == 0 {
        return 100.0
    }
    return float64(available) / float64(total) * 100.0
}

func (m *SLAMonitor) GetDowntimeMinutes(slaName string, period time.Duration) float64 {
    availability := m.GetAvailability(slaName, period)
    totalMinutes := period.Minutes()
    return totalMinutes * (1.0 - availability/100.0)
}

// Bandwidth utilization tracker
type BandwidthTracker struct {
    interfaces map[string]*InterfaceCounters
    mu         sync.RWMutex
}

type InterfaceCounters struct {
    Name      string
    Speed     uint64 // bits per second
    Samples   []CounterSample
}

type CounterSample struct {
    Timestamp  time.Time
    InOctets   uint64
    OutOctets  uint64
    InErrors   uint64
    OutErrors  uint64
    InDiscards uint64
}

func NewBandwidthTracker() *BandwidthTracker {
    return &BandwidthTracker{
        interfaces: make(map[string]*InterfaceCounters),
    }
}

func (t *BandwidthTracker) AddInterface(name string, speed uint64) {
    t.mu.Lock()
    defer t.mu.Unlock()
    t.interfaces[name] = &InterfaceCounters{
        Name:  name,
        Speed: speed,
    }
}

func (t *BandwidthTracker) RecordCounters(name string, sample CounterSample) {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    iface, exists := t.interfaces[name]
    if !exists {
        return
    }
    
    iface.Samples = append(iface.Samples, sample)
    if len(iface.Samples) > 2880 { // 24h at 30s intervals
        iface.Samples = iface.Samples[len(iface.Samples)-2880:]
    }
}

func (t *BandwidthTracker) GetUtilization(name string) (inPct, outPct float64) {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    iface, exists := t.interfaces[name]
    if !exists || len(iface.Samples) < 2 {
        return 0, 0
    }
    
    prev := iface.Samples[len(iface.Samples)-2]
    curr := iface.Samples[len(iface.Samples)-1]
    
    duration := curr.Timestamp.Sub(prev.Timestamp).Seconds()
    if duration == 0 {
        return 0, 0
    }
    
    inBPS := float64(curr.InOctets-prev.InOctets) * 8 / duration
    outBPS := float64(curr.OutOctets-prev.OutOctets) * 8 / duration
    
    inPct = inBPS / float64(iface.Speed) * 100
    outPct = outBPS / float64(iface.Speed) * 100
    
    return inPct, outPct
}

func (t *BandwidthTracker) Get95thPercentile(name string) (inBPS, outBPS float64) {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    iface, exists := t.interfaces[name]
    if !exists || len(iface.Samples) < 2 {
        return 0, 0
    }
    
    var inRates, outRates []float64
    
    for i := 1; i < len(iface.Samples); i++ {
        prev := iface.Samples[i-1]
        curr := iface.Samples[i]
        duration := curr.Timestamp.Sub(prev.Timestamp).Seconds()
        if duration == 0 {
            continue
        }
        
        inRates = append(inRates, float64(curr.InOctets-prev.InOctets)*8/duration)
        outRates = append(outRates, float64(curr.OutOctets-prev.OutOctets)*8/duration)
    }
    
    sort.Float64s(inRates)
    sort.Float64s(outRates)
    
    p95Idx := int(float64(len(inRates)-1) * 0.95)
    if p95Idx < len(inRates) {
        inBPS = inRates[p95Idx]
    }
    if p95Idx < len(outRates) {
        outBPS = outRates[p95Idx]
    }
    
    return inBPS, outBPS
}`,
				},
			},
		},
		{
			ID:          2634,
			Title:       "Network Configuration Management and Infrastructure as Code",
			Description: "Learn about network automation with Ansible, Terraform for networking, NETCONF/RESTCONF, configuration management, and GitOps for network infrastructure.",
			Order:       34,
			Lessons: []problems.Lesson{
				{
					Title: "Network Configuration and IaC",
					Content: `Network configuration management applies infrastructure as code principles to network devices and services.

**NETCONF (Network Configuration Protocol):**

  IETF standard for network device management
  Uses SSH transport (port 830)
  XML-encoded data
  RPC-based operations
  
  Protocol Layers:
    Content: Configuration data (YANG models)
    Operations: get, get-config, edit-config, etc.
    Messages: RPC, rpc-reply, notification
    Transport: SSH
    
  Operations:
    <get>: Retrieve running state and config
    <get-config>: Retrieve config only
    <edit-config>: Modify configuration
    <copy-config>: Copy datastores
    <delete-config>: Delete a datastore
    <lock>/<unlock>: Lock datastore for exclusive access
    <commit>: Apply candidate to running (if supported)
    <validate>: Validate candidate config
    
  Datastores:
    running: Active configuration
    candidate: Staged changes (commit to apply)
    startup: Boot configuration

**RESTCONF (RFC 8040):**

  RESTful interface to YANG-modeled data
  HTTP/HTTPS transport
  JSON or XML encoding
  
  Methods:
    GET: Read data
    POST: Create resource
    PUT: Create or replace
    PATCH: Modify resource
    DELETE: Remove resource
    
  URL Structure:
    https://device/restconf/data/ietf-interfaces:interfaces/interface=GigabitEthernet1
    
  Headers:
    Content-Type: application/yang-data+json
    Accept: application/yang-data+json

  RESTCONF vs NETCONF:
    RESTCONF: Simpler, REST-familiar, stateless
    NETCONF: More capabilities, transactions, candidate config

**Ansible for Networking:**

  Agentless automation
  SSH or NETCONF transport
  Modules for major vendors
  
  Inventory:
    [routers]
    router1 ansible_host=10.0.0.1
    router2 ansible_host=10.0.0.2
    
    [routers:vars]
    ansible_network_os=ios
    ansible_connection=network_cli
    
  Common Modules:
    ios_config: Cisco IOS configuration
    nxos_config: Cisco Nexus configuration
    junos_config: Juniper configuration
    eos_config: Arista configuration
    cli_command: Generic CLI command
    netconf_config: NETCONF-based config

  Playbook Example:
    - name: Configure interfaces
      hosts: routers
      gather_facts: false
      tasks:
        - name: Configure interface
          ios_config:
            lines:
              - description WAN Link
              - ip address 10.1.1.1 255.255.255.0
              - no shutdown
            parents: interface GigabitEthernet0/0
            
        - name: Save configuration
          ios_config:
            save_when: modified

**Terraform for Networking:**

  Infrastructure as Code for network resources
  
  Cloud Networking:
    VPCs, Subnets, Route Tables
    Security Groups, NACLs
    Load Balancers
    VPN Gateways
    DNS Records
    
  Network Device Providers:
    PAN-OS (Palo Alto)
    Fortinet (FortiOS)
    Cisco (ACI, ISE)
    F5 (BIG-IP)
    
  State Management:
    Track network resource state
    Plan changes before applying
    Detect drift from desired state

**Configuration Backup and Compliance:**

Configuration Backup:
  Schedule: Daily, hourly, on-change
  Storage: Git repository, S3, NFS
  Retention: Keep N versions
  Comparison: Diff between versions
  
  RANCID (Really Awesome New Cisco Config Differ):
    Login to devices, collect configs
    Store in CVS/SVN/Git
    Detect and email changes
    
  Oxidized:
    Modern RANCID replacement
    REST API
    Git backend
    Web UI
    Model-based (supports many vendors)

Configuration Compliance:
  Define policy rules
  Audit devices against policies
  Report violations
  Auto-remediate (optional)
  
  Policy Examples:
    NTP servers configured correctly
    SSH version 2 only
    No default community strings
    SNMP only from management network
    Banner configured
    Logging to central syslog
    
  Tools:
    Batfish: Config analysis and verification
    Napalm: Network Automation and Programmability
    NetBox: Source of truth / IPAM

**GitOps for Networking:**

  Git as single source of truth
  Changes via pull requests
  Automated testing and validation
  CI/CD pipeline deploys changes
  
  Workflow:
    1. Engineer creates branch
    2. Modifies network config in Git
    3. CI runs validation:
       - Syntax check
       - Batfish analysis
       - Smoke tests
    4. Peer review (Pull Request)
    5. Merge to main
    6. CD pipeline deploys to network
    7. Post-deployment validation
    
  Benefits:
    Audit trail (Git history)
    Rollback (git revert)
    Peer review
    Automated testing
    Consistent processes`,
					CodeExamples: `// Network configuration management implementations

package main

import (
    "encoding/json"
    "encoding/xml"
    "fmt"
    "strings"
    "sync"
    "time"
)

// NETCONF client simulator
type NETCONFClient struct {
    host      string
    port      int
    sessionID int
    datastores map[string]*ConfigDatastore
}

type ConfigDatastore struct {
    Name   string
    Config map[string]interface{}
    Locked bool
    LockOwner string
}

type NETCONFResponse struct {
    XMLName xml.Name
    OK      bool
    Data    interface{}
    Error   *NETCONFError
}

type NETCONFError struct {
    Type     string
    Tag      string
    Severity string
    Message  string
}

func NewNETCONFClient(host string, port int) *NETCONFClient {
    return &NETCONFClient{
        host:      host,
        port:      port,
        sessionID: 1,
        datastores: map[string]*ConfigDatastore{
            "running": {
                Name:   "running",
                Config: make(map[string]interface{}),
            },
            "candidate": {
                Name:   "candidate",
                Config: make(map[string]interface{}),
            },
            "startup": {
                Name:   "startup",
                Config: make(map[string]interface{}),
            },
        },
    }
}

func (c *NETCONFClient) GetConfig(datastore string) (map[string]interface{}, error) {
    ds, exists := c.datastores[datastore]
    if !exists {
        return nil, fmt.Errorf("datastore %s not found", datastore)
    }
    
    // Deep copy
    result := make(map[string]interface{})
    for k, v := range ds.Config {
        result[k] = v
    }
    return result, nil
}

func (c *NETCONFClient) EditConfig(datastore string, config map[string]interface{}) error {
    ds, exists := c.datastores[datastore]
    if !exists {
        return fmt.Errorf("datastore %s not found", datastore)
    }
    
    if ds.Locked && ds.LockOwner != fmt.Sprintf("session-%d", c.sessionID) {
        return fmt.Errorf("datastore %s is locked by another session", datastore)
    }
    
    for key, value := range config {
        ds.Config[key] = value
    }
    return nil
}

func (c *NETCONFClient) Lock(datastore string) error {
    ds, exists := c.datastores[datastore]
    if !exists {
        return fmt.Errorf("datastore %s not found", datastore)
    }
    
    if ds.Locked {
        return fmt.Errorf("datastore %s already locked", datastore)
    }
    
    ds.Locked = true
    ds.LockOwner = fmt.Sprintf("session-%d", c.sessionID)
    return nil
}

func (c *NETCONFClient) Unlock(datastore string) error {
    ds, exists := c.datastores[datastore]
    if !exists {
        return fmt.Errorf("datastore %s not found", datastore)
    }
    
    ds.Locked = false
    ds.LockOwner = ""
    return nil
}

func (c *NETCONFClient) Commit() error {
    candidate := c.datastores["candidate"]
    running := c.datastores["running"]
    
    for key, value := range candidate.Config {
        running.Config[key] = value
    }
    return nil
}

func (c *NETCONFClient) CopyConfig(source, target string) error {
    srcDS, exists := c.datastores[source]
    if !exists {
        return fmt.Errorf("source datastore %s not found", source)
    }
    
    tgtDS, exists := c.datastores[target]
    if !exists {
        return fmt.Errorf("target datastore %s not found", target)
    }
    
    tgtDS.Config = make(map[string]interface{})
    for k, v := range srcDS.Config {
        tgtDS.Config[k] = v
    }
    return nil
}

// RESTCONF handler
type RESTCONFServer struct {
    datastore map[string]interface{}
    mu        sync.RWMutex
}

func NewRESTCONFServer() *RESTCONFServer {
    return &RESTCONFServer{
        datastore: make(map[string]interface{}),
    }
}

func (s *RESTCONFServer) Get(path string) (interface{}, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    parts := strings.Split(strings.Trim(path, "/"), "/")
    current := interface{}(s.datastore)
    
    for _, part := range parts {
        m, ok := current.(map[string]interface{})
        if !ok {
            return nil, fmt.Errorf("path not found: %s", path)
        }
        current, ok = m[part]
        if !ok {
            return nil, fmt.Errorf("path not found: %s", path)
        }
    }
    
    return current, nil
}

func (s *RESTCONFServer) Put(path string, value interface{}) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    parts := strings.Split(strings.Trim(path, "/"), "/")
    return setNestedValue(s.datastore, parts, value)
}

func (s *RESTCONFServer) Patch(path string, value interface{}) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    // Merge with existing
    parts := strings.Split(strings.Trim(path, "/"), "/")
    
    existing := getNestedValue(s.datastore, parts)
    if existingMap, ok := existing.(map[string]interface{}); ok {
        if valueMap, ok := value.(map[string]interface{}); ok {
            for k, v := range valueMap {
                existingMap[k] = v
            }
            return nil
        }
    }
    
    return setNestedValue(s.datastore, parts, value)
}

func (s *RESTCONFServer) Delete(path string) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    parts := strings.Split(strings.Trim(path, "/"), "/")
    if len(parts) == 0 {
        return fmt.Errorf("cannot delete root")
    }
    
    parent := getNestedValue(s.datastore, parts[:len(parts)-1])
    if parentMap, ok := parent.(map[string]interface{}); ok {
        delete(parentMap, parts[len(parts)-1])
        return nil
    }
    
    return fmt.Errorf("parent not found")
}

func setNestedValue(m map[string]interface{}, path []string, value interface{}) error {
    current := m
    for i, part := range path {
        if i == len(path)-1 {
            current[part] = value
            return nil
        }
        
        next, exists := current[part]
        if !exists {
            next = make(map[string]interface{})
            current[part] = next
        }
        
        nextMap, ok := next.(map[string]interface{})
        if !ok {
            return fmt.Errorf("path element %s is not a container", part)
        }
        current = nextMap
    }
    return nil
}

func getNestedValue(m map[string]interface{}, path []string) interface{} {
    current := interface{}(m)
    for _, part := range path {
        cm, ok := current.(map[string]interface{})
        if !ok {
            return nil
        }
        current = cm[part]
    }
    return current
}

// Configuration drift detector
type DriftDetector struct {
    desiredState map[string]map[string]interface{} // device -> config
    currentState map[string]map[string]interface{}
    mu           sync.RWMutex
}

type DriftReport struct {
    Device     string
    Drifts     []ConfigDrift
    Timestamp  time.Time
    InSync     bool
}

type ConfigDrift struct {
    Path     string
    Expected interface{}
    Actual   interface{}
    Type     string // missing, extra, modified
}

func NewDriftDetector() *DriftDetector {
    return &DriftDetector{
        desiredState: make(map[string]map[string]interface{}),
        currentState: make(map[string]map[string]interface{}),
    }
}

func (d *DriftDetector) SetDesired(device string, config map[string]interface{}) {
    d.mu.Lock()
    defer d.mu.Unlock()
    d.desiredState[device] = config
}

func (d *DriftDetector) SetCurrent(device string, config map[string]interface{}) {
    d.mu.Lock()
    defer d.mu.Unlock()
    d.currentState[device] = config
}

func (d *DriftDetector) Detect(device string) *DriftReport {
    d.mu.RLock()
    defer d.mu.RUnlock()
    
    desired := d.desiredState[device]
    current := d.currentState[device]
    
    report := &DriftReport{
        Device:    device,
        Timestamp: time.Now(),
    }
    
    // Check for missing and modified
    for key, expectedVal := range desired {
        actualVal, exists := current[key]
        if !exists {
            report.Drifts = append(report.Drifts, ConfigDrift{
                Path:     key,
                Expected: expectedVal,
                Actual:   nil,
                Type:     "missing",
            })
        } else if fmt.Sprintf("%v", expectedVal) != fmt.Sprintf("%v", actualVal) {
            report.Drifts = append(report.Drifts, ConfigDrift{
                Path:     key,
                Expected: expectedVal,
                Actual:   actualVal,
                Type:     "modified",
            })
        }
    }
    
    // Check for extra
    for key, actualVal := range current {
        if _, exists := desired[key]; !exists {
            report.Drifts = append(report.Drifts, ConfigDrift{
                Path:     key,
                Expected: nil,
                Actual:   actualVal,
                Type:     "extra",
            })
        }
    }
    
    report.InSync = len(report.Drifts) == 0
    return report
}

// Network config backup manager
type ConfigBackupManager struct {
    backups map[string][]ConfigBackup
    mu      sync.RWMutex
}

type ConfigBackup struct {
    Device    string
    Config    string
    Hash      string
    Timestamp time.Time
    Changed   bool
}

func NewConfigBackupManager() *ConfigBackupManager {
    return &ConfigBackupManager{
        backups: make(map[string][]ConfigBackup),
    }
}

func (m *ConfigBackupManager) SaveBackup(device, config string) bool {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    hash := fmt.Sprintf("%x", hashConfig(config))
    
    existing := m.backups[device]
    changed := true
    if len(existing) > 0 {
        lastHash := existing[len(existing)-1].Hash
        changed = lastHash != hash
    }
    
    backup := ConfigBackup{
        Device:    device,
        Config:    config,
        Hash:      hash,
        Timestamp: time.Now(),
        Changed:   changed,
    }
    
    m.backups[device] = append(m.backups[device], backup)
    return changed
}

func (m *ConfigBackupManager) GetLatest(device string) *ConfigBackup {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    backups := m.backups[device]
    if len(backups) == 0 {
        return nil
    }
    return &backups[len(backups)-1]
}

func (m *ConfigBackupManager) GetHistory(device string, n int) []ConfigBackup {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    backups := m.backups[device]
    if n >= len(backups) {
        result := make([]ConfigBackup, len(backups))
        copy(result, backups)
        return result
    }
    
    result := make([]ConfigBackup, n)
    copy(result, backups[len(backups)-n:])
    return result
}

func hashConfig(config string) uint64 {
    var hash uint64 = 14695981039346656037
    for _, b := range []byte(config) {
        hash ^= uint64(b)
        hash *= 1099511628211
    }
    return hash
}`,
				},
			},
		},
	})
}
