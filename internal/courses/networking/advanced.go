package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          220,
			Title:       "Dynamic Routing Protocols",
			Description: "Master dynamic routing: OSPF, BGP, EIGRP, and how routers automatically learn and share routes.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "OSPF (Open Shortest Path First)",
					Content: `OSPF is a link-state routing protocol used in enterprise networks.

**OSPF Characteristics:**
- Link-state protocol
- Fast convergence
- Cost-based metric (bandwidth)
- Hierarchical (areas)
- Open standard

**OSPF Areas:**
- **Area 0**: Backbone area (required)
- **Regular Areas**: Connect to area 0
- Reduces routing table size
- Limits LSA flooding

**OSPF Router Types:**
- **Internal Router**: All interfaces in same area
- **Area Border Router (ABR)**: Connects areas
- **Backbone Router**: In area 0
- **AS Boundary Router (ASBR)**: Connects to other AS

**OSPF Process:**
1. Discover neighbors (Hello packets)
2. Exchange LSAs (Link State Advertisements)
3. Build topology database
4. Calculate shortest path (Dijkstra)
5. Update routing table

**OSPF Advantages:**
- Fast convergence
- Scalable
- No hop count limit
- Supports VLSM
- Load balancing`,
					CodeExamples: `# OSPF configuration (router-specific)

# View OSPF neighbors
# Router-specific commands
# show ip ospf neighbor

# View OSPF database
# show ip ospf database

# View OSPF routes
# show ip route ospf

# OSPF is typically configured on routers
# Not usually on Linux hosts
# But can be done with Quagga/FRR

# FRR (Free Range Routing) on Linux
# sudo apt install frr
# Configure OSPF in /etc/frr/frr.conf

# View OSPF status
# vtysh -c "show ip ospf"`,
				},
				{
					Title: "BGP (Border Gateway Protocol)",
					Content: `BGP is the protocol that routes traffic on the internet.

**BGP Characteristics:**
- Path-vector protocol
- Internet routing protocol
- Policy-based routing
- Very complex
- Slow convergence

**BGP Types:**

**eBGP (External BGP):**
- Between different ASes
- Internet routing
- More strict rules

**iBGP (Internal BGP):**
- Within same AS
- Distributes routes internally
- Full mesh or route reflectors

**AS (Autonomous System):**
- Network under single administration
- Identified by AS number
- Public (1-64511) or private (64512-65535)

**BGP Attributes:**
- **AS Path**: List of ASes traversed
- **Next Hop**: Next router
- **Local Preference**: Preferred path
- **MED**: Multi-Exit Discriminator
- **Origin**: How route was learned

**BGP Path Selection:**
- Complex algorithm
- Multiple attributes considered
- Policy-based
- Very flexible`,
					CodeExamples: `# BGP configuration (router-specific)

# View BGP neighbors
# show ip bgp neighbors

# View BGP table
# show ip bgp

# View BGP routes
# show ip route bgp

# BGP is internet routing protocol
# Usually configured on border routers
# Not on typical Linux hosts

# Can use FRR for BGP on Linux
# sudo apt install frr
# Configure BGP in /etc/frr/frr.conf

# View BGP status
# vtysh -c "show ip bgp summary"`,
				},
				{
					Title: "Routing Protocol Comparison",
					Content: `Understanding differences helps choose the right protocol.

**Protocol Types:**

**Distance Vector:**
- RIP: Simple, slow, hop count
- EIGRP: Cisco, fast, composite metric

**Link State:**
- OSPF: Open, fast, cost-based
- IS-IS: Similar to OSPF, less common

**Path Vector:**
- BGP: Internet routing, policy-based

**Comparison:**

**RIP:**
- Simple
- Slow convergence
- 15 hop limit
- Small networks

**OSPF:**
- Complex
- Fast convergence
- No hop limit
- Enterprise networks

**BGP:**
- Very complex
- Slow convergence
- Internet routing
- Large scale

**Choosing Protocol:**
- Small network: Static or RIP
- Medium network: OSPF
- Large/Internet: BGP

**Protocol Selection Factors:**
- Network size
- Convergence requirements
- Vendor support
- Complexity tolerance
- Policy requirements`,
					CodeExamples: `# Compare routing protocols

# View routing table
ip route show
# Shows routes from all sources

# View route sources
ip route show
# Shows if static, connected, or dynamic

# Test routing
traceroute destination
# Shows path regardless of protocol

# View protocol-specific routes
# Router-specific commands
# show ip route ospf
# show ip route bgp`,
				},
				{
					Title: "OSPF Configuration Deep Dive",
					Content: `Advanced OSPF configuration for complex networks.

**OSPF Areas:**

**Area Types:**
- **Backbone Area (0)**: Required, connects all areas
- **Regular Area**: Standard area, connects to backbone
- **Stub Area**: No external routes, default route only
- **Totally Stubby Area**: No external or inter-area routes
- **NSSA**: Not-So-Stubby Area, allows external routes

**Area Design:**
- Keep areas small (< 50 routers)
- Minimize inter-area traffic
- Use stub areas when possible
- Plan area boundaries carefully

**LSA Types:**

**Type 1 - Router LSA:**
- Describes router's links
- Flooded within area
- Generated by all routers

**Type 2 - Network LSA:**
- Describes network segment
- Generated by DR
- Flooded within area

**Type 3 - Summary LSA:**
- Inter-area routes
- Generated by ABR
- Flooded to other areas

**Type 4/5 - AS External LSAs:**
- External routes
- Generated by ASBR
- Flooded throughout AS

**DR/BDR (Designated Router/Backup DR):**
- Elected on broadcast networks
- Reduces LSA flooding
- DR: Highest priority (or router ID)
- BDR: Backup for DR

**OSPF Configuration Best Practices:**
- Use loopback for router ID
- Set consistent area design
- Configure authentication
- Monitor OSPF adjacencies
- Plan for scalability`,
					CodeExamples: `# OSPF deep dive (router-specific)

# View OSPF areas
# show ip ospf
# Shows area configuration

# View LSAs
# show ip ospf database
# Shows all LSAs

# View specific LSA type
# show ip ospf database router
# Shows Type 1 LSAs

# View DR/BDR
# show ip ospf neighbor
# Shows DR/BDR status

# View OSPF interfaces
# show ip ospf interface
# Shows OSPF interface details

# Configure OSPF area
# router ospf 1
# network 192.168.1.0 0.0.0.255 area 0

# Set router priority (for DR)
# interface eth0
# ip ospf priority 100

# View OSPF routes
# show ip route ospf
# Shows OSPF-learned routes`,
				},
				{
					Title: "BGP Configuration and Policies",
					Content: `BGP policies control route selection and advertisement.

**BGP Route Maps:**
- Control route advertisement
- Filter routes
- Modify attributes
- Match and set conditions
- Very powerful tool

**Route Map Components:**
- **Match**: Conditions to match
- **Set**: Actions to take
- **Permit/Deny**: Allow or block
- **Sequence**: Order matters

**BGP Communities:**
- Tags for routes
- 32-bit value
- Used for policy
- Examples: no-export, no-advertise

**Well-Known Communities:**
- **no-export**: Don't export to eBGP peers
- **no-advertise**: Don't advertise to any peer
- **local-as**: Don't advertise outside local AS
- **internet**: Advertise to internet

**BGP Attributes:**

**AS Path:**
- List of ASes traversed
- Used for loop prevention
- Influences path selection
- Prepend for path manipulation

**Local Preference:**
- iBGP only
- Higher = preferred
- Influences outbound traffic
- Set with route maps

**MED (Multi-Exit Discriminator):**
- eBGP attribute
- Lower = preferred
- Influences inbound traffic
- Used for path selection

**BGP Policy Examples:**
- Prefer certain paths
- Filter specific routes
- Manipulate attributes
- Control advertisement

**BGP Best Practices:**
- Use route maps for control
- Set local preference
- Filter unwanted routes
- Use communities
- Monitor BGP table`,
					CodeExamples: `# BGP configuration (router-specific)

# View BGP table
# show ip bgp
# Shows all BGP routes

# View BGP neighbors
# show ip bgp neighbors
# Shows neighbor status

# View BGP communities
# show ip bgp community
# Shows routes with communities

# Configure route map
# route-map PREFER-PATH permit 10
# match as-path 1
# set local-preference 200

# Apply route map
# router bgp 65000
# neighbor 192.168.1.1 route-map PREFER-PATH in

# Set community
# route-map SET-COMMUNITY permit 10
# set community 65000:100

# View BGP attributes
# show ip bgp 192.168.1.0/24
# Shows route with all attributes

# Filter routes
# ip as-path access-list 1 permit ^65000$
# Filters routes from AS 65000`,
				},
				{
					Title: "EIGRP and IS-IS",
					Content: `Alternative routing protocols for specific scenarios.

**EIGRP (Enhanced Interior Gateway Routing Protocol):**

**Characteristics:**
- Cisco proprietary (now partially open)
- Advanced distance vector
- Fast convergence
- Composite metric
- Supports IPv4 and IPv6

**EIGRP Features:**
- **DUAL**: Diffusing Update Algorithm
- **Feasible Successor**: Backup route
- **Composite Metric**: Bandwidth, delay, reliability, load
- **Partial Updates**: Only changes
- **Bounded Updates**: Only to affected routers

**EIGRP Advantages:**
- Fast convergence
- Low overhead
- Supports VLSM
- Easy configuration
- Good for Cisco networks

**EIGRP Disadvantages:**
- Cisco proprietary (mostly)
- Less common than OSPF
- Vendor lock-in

**IS-IS (Intermediate System to Intermediate System):**

**Characteristics:**
- Link-state protocol
- Similar to OSPF
- OSI protocol (not IP)
- Used by ISPs
- Very scalable

**IS-IS Features:**
- **Levels**: Level 1 (intra-area), Level 2 (inter-area)
- **NET**: Network Entity Title (address)
- **TLVs**: Type-Length-Value (flexible)
- **Fast convergence**
- **Scalable**

**IS-IS Advantages:**
- Very scalable
- Fast convergence
- Flexible (TLVs)
- Used by large ISPs
- Supports IPv4 and IPv6

**IS-IS Disadvantages:**
- Less common than OSPF
- OSI-based (different model)
- Steeper learning curve

**When to Use:**

**EIGRP:**
- Cisco-only network
- Need fast convergence
- Want simplicity
- Mixed protocols

**IS-IS:**
- Very large network
- ISP environment
- Need maximum scalability
- Multi-vendor support`,
					CodeExamples: `# EIGRP (router-specific)

# View EIGRP neighbors
# show ip eigrp neighbors
# Shows EIGRP adjacencies

# View EIGRP topology
# show ip eigrp topology
# Shows EIGRP topology table

# View EIGRP routes
# show ip route eigrp
# Shows EIGRP-learned routes

# Configure EIGRP
# router eigrp 100
# network 192.168.1.0

# IS-IS (router-specific)

# View IS-IS neighbors
# show isis neighbors
# Shows IS-IS adjacencies

# View IS-IS database
# show isis database
# Shows link-state database

# Configure IS-IS
# router isis
# net 49.0001.0000.0000.0001.00
# interface eth0
# ip router isis

# Both protocols are router-specific
# Not typically on Linux hosts
# But concepts important for networking`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          221,
			Title:       "Network Security Advanced",
			Description: "Advanced security topics: intrusion detection, network monitoring, security protocols, and threat mitigation.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Intrusion Detection Systems (IDS)",
					Content: `IDS monitors network for suspicious activity.

**IDS Types:**

**Network IDS (NIDS):**
- Monitors network traffic
- Analyzes packets
- Detects attacks
- Passive monitoring

**Host IDS (HIDS):**
- Monitors individual hosts
- Analyzes logs, files
- Detects host-based attacks

**Detection Methods:**

**Signature-based:**
- Known attack patterns
- Fast detection
- Can't detect new attacks

**Anomaly-based:**
- Baseline normal behavior
- Detects deviations
- Can detect new attacks
- More false positives

**IDS Tools:**
- Snort: Popular NIDS
- Suricata: Modern NIDS
- OSSEC: HIDS
- Fail2ban: Simple IDS`,
					CodeExamples: `# IDS tools

# Snort (NIDS)
sudo snort -A console -q -c /etc/snort/snort.conf -i eth0
# Runs Snort IDS

# Suricata
sudo suricata -c /etc/suricata/suricata.yaml -i eth0
# Runs Suricata IDS

# Fail2ban
sudo fail2ban-client status
# Shows banned IPs
sudo fail2ban-client status sshd
# Shows SSH jail status

# View IDS logs
sudo tail -f /var/log/snort/alert
# Snort alerts
sudo tail -f /var/log/suricata/fast.log
# Suricata alerts`,
				},
				{
					Title: "Network Monitoring",
					Content: `Monitoring networks helps detect issues and attacks.

**Monitoring Tools:**

**SNMP (Simple Network Management Protocol):**
- Standard protocol
- Queries network devices
- Gets statistics
- Port 161

**NetFlow/sFlow:**
- Flow-based monitoring
- Traffic analysis
- Bandwidth monitoring

**Packet Capture:**
- tcpdump: Command-line
- Wireshark: GUI
- Deep packet inspection

**Monitoring Metrics:**
- Bandwidth usage
- Packet loss
- Latency
- Errors
- Connections

**Alerting:**
- Threshold-based
- Anomaly detection
- Real-time notifications`,
					CodeExamples: `# Network monitoring

# SNMP queries
snmpwalk -v 2c -c public localhost
# Queries SNMP device

# NetFlow (if configured)
# Usually router/switch feature
# Collects flow data

# Packet capture
sudo tcpdump -i eth0 -w capture.pcap
# Captures packets to file

# Wireshark (GUI)
wireshark
# Opens GUI packet analyzer

# Monitor bandwidth
iftop
# Real-time bandwidth monitor
nethogs
# Bandwidth by process

# Monitor connections
ss -s
# Shows socket statistics
netstat -s
# Shows protocol statistics`,
				},
				{
					Title: "Security Protocols",
					Content: `Security protocols protect network communications.

**TLS/SSL:**
- Transport Layer Security
- Encrypts data in transit
- Used by HTTPS
- Port 443
- Versions: TLS 1.2, TLS 1.3 (latest)

**IPsec:**
- IP Security
- Encrypts IP packets
- VPN protocol
- AH (Authentication Header) and ESP (Encapsulating Security Payload)
- Modes: Transport, Tunnel

**WPA2/WPA3:**
- WiFi security
- Encrypts wireless traffic
- WPA3 is latest (more secure)
- WPA2 still common

**802.1X:**
- Port-based access control
- Authentication before access
- Used in enterprise WiFi
- EAP (Extensible Authentication Protocol)

**Best Practices:**
- Use encryption
- Strong authentication
- Regular updates
- Monitor for attacks
- Disable weak protocols`,
					CodeExamples: `# Security protocols

# Test TLS/SSL
openssl s_client -connect example.com:443
# Tests SSL/TLS connection

# View SSL certificate
openssl s_client -showcerts -connect example.com:443

# Check TLS version
openssl s_client -connect example.com:443 -tls1_3
# Tests specific TLS version

# Test IPsec
sudo ipsec status
# Shows IPsec connections

# WiFi security
iwconfig
# Shows WiFi security settings

# View security settings
# Check encryption on connections
# Use secure protocols (TLS, IPsec)

# Check certificate validity
openssl x509 -in cert.pem -text -noout
# Shows certificate details`,
				},
				{
					Title: "Network Access Control (NAC)",
					Content: `NAC controls device access to networks.

**What is NAC?**
- Network Access Control
- Controls who can access network
- Authentication before access
- Policy enforcement

**NAC Components:**

**802.1X:**
- Port-based access control
- Authentication before network access
- Three components:
  - **Supplicant**: Client device
  - **Authenticator**: Switch/AP
  - **Authentication Server**: RADIUS server

**EAP Methods:**
- **EAP-TLS**: Certificate-based (most secure)
- **PEAP**: Protected EAP (password-based)
- **EAP-MD5**: Legacy (insecure)
- **EAP-TTLS**: Tunneled TLS

**Port Security:**
- Limits MAC addresses per port
- Prevents unauthorized devices
- Static or dynamic learning
- Violation actions: shutdown, restrict, protect

**NAC Benefits:**
- Prevents unauthorized access
- Enforces policies
- Device visibility
- Compliance

**NAC Implementation:**
- Requires infrastructure
- RADIUS server needed
- Client configuration
- Policy definition

**NAC Best Practices:**
- Use 802.1X for wireless
- Implement port security
- Monitor NAC logs
- Regular policy review`,
					CodeExamples: `# NAC configuration (switch/router-specific)

# 802.1X configuration
# Usually on switch/AP
# Requires RADIUS server

# View 802.1X status
# show dot1x all
# Shows 802.1X sessions

# Port security (switch)
# interface eth0
# switchport port-security
# switchport port-security maximum 2
# switchport port-security violation restrict

# View port security
# show port-security
# Shows port security status

# RADIUS configuration
# Usually requires RADIUS server
# Configured on switch/AP

# Test NAC
# Try connecting unauthorized device
# Verify it's blocked

# Monitor NAC
# Check authentication logs
# Review violations`,
				},
				{
					Title: "Threat Detection and Response",
					Content: `Detecting and responding to network threats is critical.

**Threat Detection:**

**SIEM (Security Information and Event Management):**
- Centralized log collection
- Correlation and analysis
- Real-time monitoring
- Alert generation
- Examples: Splunk, ELK Stack, QRadar

**Log Analysis:**
- Collect logs from all devices
- Analyze for patterns
- Identify anomalies
- Detect attacks

**Threat Intelligence:**
- External threat feeds
- Known bad IPs/domains
- Malware signatures
- Attack patterns

**Detection Methods:**

**Signature-Based:**
- Known attack patterns
- Fast detection
- Can't detect new attacks

**Anomaly-Based:**
- Baseline normal behavior
- Detects deviations
- Can detect new attacks
- More false positives

**Behavioral Analysis:**
- User behavior patterns
- Device behavior
- Network behavior
- Machine learning

**Incident Response:**

**Preparation:**
- Define procedures
- Train team
- Prepare tools
- Document contacts

**Detection:**
- Monitor systems
- Analyze alerts
- Investigate incidents
- Confirm threats

**Containment:**
- Isolate affected systems
- Block malicious traffic
- Preserve evidence
- Limit damage

**Eradication:**
- Remove threats
- Patch vulnerabilities
- Update systems
- Clean systems

**Recovery:**
- Restore services
- Verify systems
- Monitor for recurrence
- Update defenses

**Lessons Learned:**
- Document incident
- Review response
- Improve procedures
- Update defenses`,
					CodeExamples: `# Threat detection and response

# View system logs
sudo journalctl -f
# Shows system logs in real-time

# View authentication logs
sudo tail -f /var/log/auth.log
# Shows authentication attempts

# View firewall logs
sudo tail -f /var/log/iptables.log
# Shows firewall activity

# Analyze logs
grep "Failed password" /var/log/auth.log
# Finds failed login attempts

# Monitor network traffic
sudo tcpdump -i eth0
# Captures suspicious traffic

# Check for suspicious connections
ss -tn | grep ESTAB
# Shows active connections

# Block malicious IPs
sudo iptables -A INPUT -s malicious-ip -j DROP
# Blocks specific IP

# View IDS alerts
sudo tail -f /var/log/snort/alert
# Shows IDS alerts

# Incident response
# 1. Document incident
# 2. Contain threat
# 3. Investigate
# 4. Remediate
# 5. Review`,
				},
				{
					Title: "Network Segmentation Strategies",
					Content: `Network segmentation improves security and performance.

**What is Segmentation?**
- Dividing network into smaller parts
- Logical or physical separation
- Controls traffic flow
- Limits attack surface

**Segmentation Methods:**

**VLANs:**
- Logical segmentation
- Same physical infrastructure
- Easy to implement
- Common method

**Physical Separation:**
- Separate networks physically
- Air-gapped networks
- Most secure
- More expensive

**Firewalls:**
- Segment with firewalls
- Control inter-segment traffic
- Policy enforcement
- Common approach

**Zero Trust:**
- Never trust, always verify
- Verify every connection
- Least privilege access
- Continuous verification

**Zero Trust Principles:**
- Verify explicitly
- Use least privilege
- Assume breach
- Micro-segmentation

**Micro-Segmentation:**
- Fine-grained segmentation
- Per-workload policies
- Dynamic policies
- Software-defined

**Segmentation Strategies:**

**By Function:**
- Servers, workstations, IoT
- Different security levels
- Common approach

**By Department:**
- Sales, HR, Engineering
- Department isolation
- Access control

**By Sensitivity:**
- Public, internal, confidential
- Data classification
- Security levels

**DMZ (Demilitarized Zone):**
- Public-facing services
- Isolated from internal
- Less protected
- Common pattern

**Segmentation Benefits:**
- Limits breach scope
- Improves performance
- Easier compliance
- Better monitoring

**Best Practices:**
- Plan segmentation
- Document policies
- Monitor inter-segment traffic
- Regular review
- Test segmentation`,
					CodeExamples: `# Network segmentation

# View VLANs (logical segmentation)
ip link show type vlan
# Shows VLAN interfaces

# View firewall rules (segmentation)
sudo iptables -L -n -v
# Shows firewall rules between segments

# Test segmentation
ping 192.168.2.1
# From segment 1 to segment 2
# Should be blocked if segmented

# View network segments
ip addr show
# Shows different network segments

# Configure VLAN (segmentation)
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip addr add 192.168.100.1/24 dev eth0.100

# View routing (segmentation control)
ip route show
# Shows routes between segments

# Monitor inter-segment traffic
sudo tcpdump -i eth0 'host 192.168.2.1'
# Monitors traffic to segment 2

# Test zero trust
# Verify authentication required
# Check access policies
# Monitor all connections`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          222,
			Title:       "Network Monitoring & Troubleshooting",
			Description: "Advanced troubleshooting techniques: packet analysis, network diagnostics, performance tuning, and problem resolution.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Packet Analysis",
					Content: `Analyzing packets helps diagnose network issues.

**Packet Capture Tools:**

**tcpdump:**
- Command-line tool
- Powerful filters
- Low overhead
- Linux standard

**Wireshark:**
- GUI tool
- Deep analysis
- Protocol decoding
- Very detailed

**Packet Analysis:**
- Inspect headers
- Check payload
- Identify problems
- Verify protocols

**Common Issues Found:**
- Retransmissions
- Duplicate packets
- Out-of-order packets
- Errors
- Malformed packets`,
					CodeExamples: `# Packet capture and analysis

# Capture packets
sudo tcpdump -i eth0 -w capture.pcap
# Saves to file

# Capture specific traffic
sudo tcpdump -i eth0 port 80
# Only HTTP traffic
sudo tcpdump -i eth0 host 192.168.1.100
# Only from/to specific host

# View packets
sudo tcpdump -i eth0 -n -v
# -n: numeric, -v: verbose

# Analyze saved capture
tcpdump -r capture.pcap
# Reads from file

# Wireshark
wireshark capture.pcap
# Opens in GUI

# Filter in tcpdump
sudo tcpdump -i eth0 'tcp and port 80'
# Complex filters`,
				},
				{
					Title: "Network Diagnostics",
					Content: `Systematic approach to network troubleshooting.

**Troubleshooting Steps:**
1. Identify problem
2. Gather information
3. Test connectivity
4. Check configuration
5. Analyze traffic
6. Fix issue
7. Verify fix

**Diagnostic Tools:**

**ping:**
- Tests connectivity
- Measures latency
- Basic test

**traceroute:**
- Shows path
- Identifies where packets stop
- Measures latency per hop

**mtr:**
- Combines ping and traceroute
- Continuous monitoring
- Better than traceroute

**nslookup/dig:**
- DNS diagnostics
- Tests name resolution

**netstat/ss:**
- Connection information
- Port status
- Process information`,
					CodeExamples: `# Network diagnostics

# Basic connectivity
ping -c 4 google.com
# Tests connectivity

# Path analysis
traceroute google.com
mtr google.com
# Shows path and latency

# DNS test
dig google.com
nslookup google.com
# Tests DNS resolution

# Port test
nc -zv hostname 80
telnet hostname 80
# Tests if port is open

# Connection status
ss -tuln
# Shows listening ports
ss -tn
# Shows active connections

# Interface status
ip link show
# Shows interface status
ethtool eth0
# Shows link details

# Routing test
ip route get 8.8.8.8
# Shows route for destination`,
				},
				{
					Title: "Performance Tuning",
					Content: `Optimizing network performance improves user experience.

**Performance Factors:**
- Bandwidth
- Latency
- Packet loss
- Congestion
- Hardware

**Optimization Techniques:**

**Bandwidth:**
- Increase link speed
- Use faster media
- Reduce overhead

**Latency:**
- Reduce distance
- Optimize routing
- Use faster paths

**Congestion:**
- QoS (Quality of Service)
- Traffic shaping
- Load balancing

**TCP Tuning:**
- Increase buffer sizes
- Enable window scaling
- Optimize algorithms

**Performance Monitoring:**
- Baseline measurements
- Regular monitoring
- Identify bottlenecks
- Measure improvements`,
					CodeExamples: `# Network performance tuning

# View current settings
sysctl net.core.rmem_max
sysctl net.core.wmem_max
# Shows buffer sizes

# Increase buffers
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216

# TCP tuning
sudo sysctl -w net.ipv4.tcp_fin_timeout=30
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# Make permanent
# Add to /etc/sysctl.conf

# Test performance
iperf3 -c server-ip
# Tests bandwidth

# Monitor performance
iftop
# Real-time bandwidth
nethogs
# Bandwidth by process

# Measure latency
ping -c 100 destination | grep "time="
# Shows latency distribution`,
				},
				{
					Title: "Network Monitoring Tools",
					Content: `Comprehensive monitoring requires multiple tools and protocols.

**SNMP (Simple Network Management Protocol):**
- Standard protocol for device management
- Port 161 (UDP)
- Versions: v1, v2c, v3 (secure)
- Queries device statistics
- MIB (Management Information Base)

**SNMP Operations:**
- **GET**: Retrieve value
- **GETNEXT**: Get next value
- **SET**: Configure value
- **TRAP**: Asynchronous notification

**NetFlow:**
- Cisco flow-based monitoring
- Traffic analysis
- Version 5, 9, IPFIX
- Exports flow data
- Analyzes traffic patterns

**sFlow:**
- Sampling-based monitoring
- High-speed networks
- Samples packets
- Lower overhead
- Real-time analysis

**IPFIX (IP Flow Information Export):**
- IETF standard
- Based on NetFlow v9
- Flexible data model
- Vendor-neutral

**Monitoring Tools:**

**Nagios:**
- Infrastructure monitoring
- Alerting
- Plugin-based
- Popular

**Zabbix:**
- Enterprise monitoring
- Real-time monitoring
- Alerting
- Graphing

**Prometheus:**
- Time-series database
- Pull-based
- PromQL query language
- Cloud-native

**Grafana:**
- Visualization
- Works with Prometheus
- Dashboards
- Alerting

**Monitoring Best Practices:**
- Monitor key metrics
- Set appropriate thresholds
- Use multiple tools
- Centralize monitoring
- Regular review`,
					CodeExamples: `# Network monitoring tools

# SNMP queries
snmpwalk -v 2c -c public localhost
# Queries SNMP device
snmpget -v 2c -c public localhost 1.3.6.1.2.1.1.1.0
# Gets specific value

# SNMP v3 (secure)
snmpwalk -v 3 -u username -l authPriv -a SHA -A password localhost

# View SNMP MIBs
snmptranslate -Td iso.org.dod.internet.mgmt.mib-2.system
# Shows MIB structure

# NetFlow (if configured)
# Usually router/switch feature
# Exports to collector
# Analyzed with tools like nfdump

# View NetFlow data
# nfdump -R /path/to/flows -s ip/bytes
# Analyzes NetFlow data

# sFlow (if configured)
# Usually switch feature
# Samples packets
# Analyzed with sflowtool

# View sFlow data
# sflowtool -p 6343
# Receives sFlow data

# Monitoring with tools
# Configure Nagios/Zabbix/Prometheus
# Set up monitoring
# Create dashboards
# Configure alerts`,
				},
				{
					Title: "Troubleshooting Methodology",
					Content: `Systematic troubleshooting approach improves success rate.

**Troubleshooting Process:**

**1. Define Problem:**
- What is the issue?
- When did it start?
- Who is affected?
- Scope of problem

**2. Gather Information:**
- Collect logs
- Interview users
- Check monitoring
- Review changes

**3. Develop Hypothesis:**
- Possible causes
- Most likely first
- Test hypothesis
- Verify or reject

**4. Test Solution:**
- Test in lab if possible
- Apply fix carefully
- Monitor results
- Verify fix

**5. Document:**
- Problem description
- Root cause
- Solution applied
- Prevention

**OSI Layer Troubleshooting:**

**Layer 1 - Physical:**
- Check cables
- Verify link status
- Check power
- Test hardware

**Layer 2 - Data Link:**
- Check MAC addresses
- Verify VLANs
- Check STP
- Test switching

**Layer 3 - Network:**
- Check IP addresses
- Verify routing
- Test connectivity
- Check subnets

**Layer 4 - Transport:**
- Check ports
- Verify connections
- Test protocols
- Check firewalls

**Layer 5-7 - Upper Layers:**
- Check applications
- Verify services
- Test protocols
- Check authentication

**Common Issues by Layer:**

**Physical:**
- Cable problems
- Hardware failure
- Link down

**Data Link:**
- VLAN misconfiguration
- STP loops
- MAC address issues

**Network:**
- Routing problems
- IP conflicts
- Subnet issues

**Transport:**
- Port blocked
- Firewall rules
- Connection issues

**Application:**
- Service down
- Configuration errors
- Authentication failures

**Troubleshooting Tools:**
- ping: Layer 3
- traceroute: Layer 3
- arp: Layer 2
- tcpdump: All layers
- netstat/ss: Layer 4`,
					CodeExamples: `# Troubleshooting methodology

# Layer 1 - Physical
ethtool eth0
# Check link status
ip link show
# Shows interface status

# Layer 2 - Data Link
ip link show
# Check MAC addresses
bridge link show
# Check VLANs, STP

# Layer 3 - Network
ip addr show
# Check IP addresses
ip route show
# Check routing
ping 8.8.8.8
# Test connectivity

# Layer 4 - Transport
ss -tuln
# Check ports
telnet hostname 80
# Test port connectivity

# Layer 7 - Application
curl http://example.com
# Test application
systemctl status service
# Check service status

# Systematic approach
# 1. Start at Layer 1
# 2. Work up layers
# 3. Test at each layer
# 4. Identify problem layer
# 5. Fix at that layer`,
				},
				{
					Title: "Performance Analysis",
					Content: `Analyzing network performance identifies bottlenecks and optimization opportunities.

**Performance Baseline:**
- Establish normal performance
- Measure key metrics
- Document baseline
- Use for comparison

**Key Metrics:**
- **Bandwidth**: Throughput capacity
- **Latency**: Round-trip time
- **Jitter**: Latency variation
- **Packet Loss**: Lost packets percentage
- **Errors**: Error rate
- **Utilization**: Link usage percentage

**Capacity Planning:**
- Current usage analysis
- Growth projections
- Peak usage identification
- Upgrade planning
- Budget considerations

**Bottleneck Identification:**
- Monitor all links
- Identify high utilization
- Check for errors
- Analyze patterns
- Find constraints

**Performance Analysis Tools:**

**Bandwidth Monitoring:**
- iftop: Real-time bandwidth
- nethogs: Per-process bandwidth
- vnstat: Historical statistics
- iperf3: Bandwidth testing

**Latency Analysis:**
- ping: Basic latency
- mtr: Continuous latency
- traceroute: Path latency
- Wireshark: Detailed analysis

**Traffic Analysis:**
- tcpdump: Packet capture
- Wireshark: Deep analysis
- NetFlow: Flow analysis
- sFlow: Sampling analysis

**Performance Optimization:**
- Identify bottlenecks
- Optimize routing
- Implement QoS
- Upgrade hardware
- Optimize protocols

**Performance Reporting:**
- Regular reports
- Trend analysis
- Capacity planning
- Recommendations
- Documentation`,
					CodeExamples: `# Performance analysis

# Establish baseline
iperf3 -c server-ip -t 60
# Tests bandwidth baseline
ping -c 100 destination | grep "time=" | awk '{print $7}' | cut -d= -f2 | sort -n
# Measures latency baseline

# Monitor bandwidth
iftop -i eth0
# Real-time bandwidth monitoring
vnstat -i eth0
# Historical bandwidth statistics

# Analyze latency
mtr destination
# Continuous latency monitoring
ping -c 100 destination | grep "time="
# Latency distribution

# Traffic analysis
sudo tcpdump -i eth0 -w capture.pcap
# Captures traffic for analysis
tcpdump -r capture.pcap | head -100
# Analyzes captured traffic

# View interface statistics
ip -s link show eth0
# Shows detailed statistics
ethtool -S eth0
# Shows Ethernet statistics

# Identify bottlenecks
# Monitor all interfaces
# Find high utilization
# Check for errors
# Analyze patterns

# Performance reporting
# Collect metrics
# Analyze trends
# Create reports
# Make recommendations`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          223,
			Title:       "Software-Defined Networking (SDN)",
			Description: "Introduction to SDN: concepts, OpenFlow, network virtualization, and programmable networks.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "SDN Concepts",
					Content: `SDN separates control plane from data plane.

**Traditional Networking:**
- Control and data plane together
- Each device makes decisions
- Hard to manage
- Vendor-specific

**SDN Architecture:**
- **Control Plane**: Centralized controller
- **Data Plane**: Simple forwarding devices
- **Management Plane**: Applications

**SDN Benefits:**
- Centralized management
- Programmable
- Flexible
- Vendor-agnostic
- Easier automation

**SDN Components:**

**Controller:**
- Central brain
- Makes routing decisions
- Programs switches
- Examples: OpenDaylight, ONOS

**Southbound API:**
- Controller to switches
- OpenFlow protocol
- Programs forwarding

**Northbound API:**
- Applications to controller
- REST APIs
- Applications control network`,
					CodeExamples: `# SDN tools

# OpenFlow (if using SDN)
# Usually requires SDN-capable switches
# And SDN controller

# Mininet (SDN emulator)
sudo mn --topo linear,3
# Creates linear topology
# Can test SDN concepts

# OpenDaylight controller
# Java-based SDN controller
# REST APIs for control

# ONOS controller
# Another SDN controller option

# SDN is typically enterprise/cloud
# Not common on typical Linux hosts
# But concepts important`,
				},
				{
					Title: "Network Virtualization",
					Content: `Network virtualization abstracts physical networks.

**Virtual Networks:**
- Logical networks
- Independent of physical
- Multiple on same infrastructure
- Isolated

**Overlay Networks:**
- Virtual network over physical
- Tunneling protocols
- VXLAN, GRE, Geneve
- Used in clouds

**VXLAN:**
- Virtual eXtensible LAN
- Extends Layer 2 over Layer 3
- 24-bit VNI (VXLAN Network Identifier)
- Used in data centers

**Benefits:**
- Multi-tenancy
- Scalability
- Flexibility
- Isolation

**Virtualization Technologies:**
- **VXLAN**: Layer 2 over Layer 3
- **GRE**: Generic Routing Encapsulation
- **Geneve**: Generic Network Virtualization Encapsulation
- **NVGRE**: Network Virtualization using GRE`,
					CodeExamples: `# Network virtualization

# VXLAN (if supported)
sudo ip link add vxlan0 type vxlan id 100 remote 192.168.1.100 dstport 4789 dev eth0
sudo ip addr add 10.0.0.1/24 dev vxlan0
sudo ip link set vxlan0 up

# GRE tunnel
sudo ip tunnel add gre0 mode gre remote 192.168.1.100 local 192.168.1.1
sudo ip addr add 10.0.0.1/24 dev gre0
sudo ip link set gre0 up

# View virtual interfaces
ip link show type vxlan
ip link show type gre

# Network virtualization is common in:
# - Cloud platforms
# - Containers (Docker, Kubernetes)
# - Virtualization platforms

# Test virtual network
ping 10.0.0.2
# Tests connectivity over virtual network`,
				},
				{
					Title: "OpenFlow Protocol",
					Content: `OpenFlow is the standard protocol for SDN communication.

**What is OpenFlow?**
- Communication protocol
- Controller to switch
- Programs forwarding
- Standardized by ONF

**OpenFlow Components:**

**Flow Tables:**
- Match-action tables
- Match packets
- Apply actions
- Multiple tables possible

**Flow Entries:**
- **Match Fields**: Packet characteristics
- **Actions**: What to do
- **Counters**: Statistics
- **Priority**: Rule priority

**Match Fields:**
- Source/destination IP
- Source/destination port
- Protocol
- VLAN ID
- Many more fields

**Actions:**
- **Forward**: Send to port
- **Drop**: Discard packet
- **Modify**: Change fields
- **Send to Controller**: Send to controller

**OpenFlow Messages:**

**Controller to Switch:**
- **Flow Mod**: Add/modify/delete flows
- **Packet Out**: Send packet
- **Modify State**: Change switch state

**Switch to Controller:**
- **Packet In**: Send packet to controller
- **Flow Removed**: Flow expired
- **Port Status**: Port changed
- **Error**: Error message

**OpenFlow Versions:**
- **1.0**: Original version
- **1.3**: Most common
- **1.5**: Latest features
- Backward compatibility

**OpenFlow Benefits:**
- Standardized
- Programmable
- Flexible
- Vendor-agnostic`,
					CodeExamples: `# OpenFlow (requires SDN setup)

# OpenFlow is typically used with:
# - SDN controllers (OpenDaylight, ONOS)
# - OpenFlow switches
# - Mininet for testing

# Mininet (SDN emulator)
sudo mn --topo linear,3 --controller=remote,ip=127.0.0.1
# Creates topology with remote controller

# View OpenFlow flows
# Usually via controller or switch CLI
# Shows flow table entries

# Add flow (example)
# ovs-ofctl add-flow br0 "priority=100,ip,nw_dst=10.0.0.1,actions=output:1"
# Adds flow to Open vSwitch

# View flows
# ovs-ofctl dump-flows br0
# Shows all flows

# OpenFlow requires:
# - SDN-capable switch
# - SDN controller
# - OpenFlow protocol support`,
				},
				{
					Title: "SDN Controllers",
					Content: `SDN controllers are the brains of SDN networks.

**What is SDN Controller?**
- Centralized control
- Makes routing decisions
- Programs switches
- Manages network

**Controller Functions:**
- Network topology discovery
- Path calculation
- Flow installation
- Network management
- Policy enforcement

**Popular Controllers:**

**OpenDaylight:**
- Linux Foundation project
- Java-based
- Modular architecture
- REST APIs
- Very popular

**ONOS (Open Network Operating System):**
- ON.Lab project
- Java-based
- High performance
- Carrier-grade
- Distributed

**Ryu:**
- Python-based
- Lightweight
- Easy to learn
- Good for learning
- REST APIs

**Floodlight:**
- Java-based
- Simple
- Good for small networks
- REST APIs

**Controller Architecture:**

**Northbound API:**
- Applications to controller
- REST APIs
- Applications control network
- Business logic

**Southbound API:**
- Controller to switches
- OpenFlow protocol
- Programs forwarding
- Data plane control

**Controller Features:**
- Topology management
- Path computation
- Flow management
- Network virtualization
- Policy enforcement

**Controller Selection:**
- **OpenDaylight**: Enterprise, features
- **ONOS**: Carrier-grade, performance
- **Ryu**: Learning, simplicity
- **Floodlight**: Small networks`,
					CodeExamples: `# SDN controllers

# OpenDaylight
# Download and run
# Access web UI at http://localhost:8181
# REST APIs available

# ONOS
# Download and run
# Access web UI
# REST APIs available

# Ryu
# Install: pip install ryu
# Run: ryu-manager simple_switch.py
# REST APIs available

# Controller REST API example
curl http://localhost:8080/stats/switches
# Gets switch list (Ryu example)

# View topology
curl http://localhost:8080/v1.0/topology/switches
# Gets topology (example)

# Controllers typically require:
# - Java (OpenDaylight, ONOS)
# - Python (Ryu)
# - Network with OpenFlow switches
# - Configuration`,
				},
				{
					Title: "Network Function Virtualization (NFV)",
					Content: `NFV virtualizes network functions traditionally done by hardware.

**What is NFV?**
- Network Function Virtualization
- Software-based network functions
- Runs on standard servers
- Replaces dedicated hardware

**Traditional Networking:**
- Dedicated hardware appliances
- Router, firewall, load balancer
- Each function = separate box
- Expensive, inflexible

**NFV Approach:**
- Virtual network functions (VNFs)
- Software on servers
- Flexible deployment
- Cost-effective

**NFV Components:**

**VNF (Virtual Network Function):**
- Software version of network function
- Router, firewall, load balancer
- Runs in VM or container
- Examples: vRouter, vFirewall

**NFVI (NFV Infrastructure):**
- Compute, storage, network
- Virtualization layer
- Provides resources
- Cloud infrastructure

**MANO (Management and Orchestration):**
- NFV-MANO framework
- Manages VNFs
- Orchestrates resources
- Lifecycle management

**NFV Use Cases:**

**vRouter:**
- Virtual router
- Software-based routing
- Flexible, scalable

**vFirewall:**
- Virtual firewall
- Software-based filtering
- Easy to deploy

**vLoad Balancer:**
- Virtual load balancer
- Software-based balancing
- Scalable

**vWAN:**
- Virtual WAN
- Software-defined WAN
- Flexible connectivity

**NFV Benefits:**
- Cost reduction
- Flexibility
- Scalability
- Faster deployment
- Vendor independence

**NFV vs SDN:**
- **NFV**: Virtualizes functions
- **SDN**: Separates control/data plane
- Often used together
- Complementary technologies`,
					CodeExamples: `# NFV examples

# Virtual router (VNF)
# Deploy router VM/container
# Configure routing
# Replace hardware router

# Virtual firewall (VNF)
# Deploy firewall VM/container
# Configure rules
# Replace hardware firewall

# NFV platforms:
# - OpenStack (NFV infrastructure)
# - Kubernetes (container-based)
# - VMware (virtualization)

# Deploy VNF
# Usually via orchestration platform
# OpenStack Heat, Kubernetes, etc.

# View VNF status
# Check VM/container status
# Monitor performance
# Manage lifecycle

# NFV requires:
# - Virtualization platform
# - VNF images
# - Orchestration
# - Management tools`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          224,
			Title:       "Network Automation",
			Description: "Automate network tasks: configuration management, network programmability, APIs, and infrastructure as code.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Network Automation Basics",
					Content: `Automation reduces manual work and errors.

**Why Automate?**
- Consistency
- Speed
- Reduced errors
- Scalability
- Documentation

**Automation Tools:**

**Ansible:**
- Configuration management
- Agentless
- YAML playbooks
- Very popular

**Python:**
- Scripting language
- Network libraries
- netmiko, paramiko
- Flexible

**APIs:**
- REST APIs
- NETCONF/YANG
- gRPC
- Programmatic control

**Infrastructure as Code:**
- Version control
- Repeatable
- Testable
- Documented`,
					CodeExamples: `# Network automation examples

# Ansible playbook example
# - name: Configure switch
#   hosts: switches
#   tasks:
#     - name: Configure VLAN
#       ios_vlan:
#         vlan_id: 100
#         name: Sales

# Python with netmiko
# from netmiko import ConnectHandler
# device = {
#     'device_type': 'cisco_ios',
#     'host': '192.168.1.1',
#     'username': 'admin',
#     'password': 'password',
# }
# net_connect = ConnectHandler(**device)
# output = net_connect.send_command('show ip route')

# REST API example
# curl -X GET https://api.example.com/api/v1/devices
# Gets device list

# Automation is key for:
# - Large networks
# - Frequent changes
# - Consistency
# - DevOps practices`,
				},
				{
					Title: "Network APIs",
					Content: `APIs enable programmatic network control.

**API Types:**

**REST APIs:**
- HTTP-based
- JSON data
- Easy to use
- Common in modern devices
- Stateless

**NETCONF:**
- Network configuration protocol
- XML-based
- YANG data models
- Standard protocol
- Transactional

**gRPC:**
- High performance
- Protocol buffers
- Streaming
- Modern approach
- Binary protocol

**SNMP:**
- Legacy but still used
- Simple queries
- Limited functionality
- Still common

**Using APIs:**
- Automate configuration
- Monitor networks
- Integrate systems
- Build applications

**API Authentication:**
- API keys
- OAuth
- Certificates
- Tokens`,
					CodeExamples: `# Network API examples

# REST API (curl)
curl -X GET https://api.example.com/api/v1/interfaces
# Gets interface list

curl -X POST https://api.example.com/api/v1/vlans \
  -H "Content-Type: application/json" \
  -d '{"vlan_id": 100, "name": "Sales"}'
# Creates VLAN

# Python requests
# import requests
# response = requests.get('https://api.example.com/api/v1/devices')
# devices = response.json()

# NETCONF (if supported)
# Usually requires NETCONF client library
# More complex than REST

# gRPC (if supported)
# Requires gRPC client
# Protocol buffers
# High performance

# APIs enable:
# - Automation
# - Integration
# - Custom tools
# - Modern network management`,
				},
				{
					Title: "Network Configuration Management",
					Content: `Configuration management ensures consistent network configurations.

**Configuration Management Tools:**

**Ansible:**
- Agentless
- YAML playbooks
- Idempotent
- Very popular
- Network modules

**Ansible Network Modules:**
- **ios_config**: Cisco IOS
- **nxos_config**: Cisco NX-OS
- **junos_config**: Juniper
- **eos_config**: Arista
- Many more

**Ansible Playbook Example:**
- Define tasks
- Target devices
- Configuration changes
- Idempotent execution

**Puppet:**
- Agent-based or agentless
- Declarative
- Good for large networks
- Network modules available

**Chef:**
- Agent-based
- Ruby-based
- Flexible
- Network support

**SaltStack:**
- Fast
- Scalable
- Python-based
- Network support

**Configuration Management Benefits:**
- Consistency
- Version control
- Rollback capability
- Documentation
- Reduced errors

**Best Practices:**
- Use version control
- Test in lab first
- Document changes
- Use templates
- Regular backups`,
					CodeExamples: `# Ansible network automation

# Ansible playbook example
# ---
# - name: Configure VLAN
#   hosts: switches
#   tasks:
#     - name: Create VLAN
#       ios_vlan:
#         vlan_id: 100
#         name: Sales
#         state: present
#
#     - name: Configure interface
#       ios_interface:
#         name: GigabitEthernet0/1
#         description: Sales VLAN
#         mode: access
#         access_vlan: 100

# Run playbook
# ansible-playbook configure_vlan.yml

# Ansible ad-hoc commands
# ansible switches -m ios_command -a "show vlan"

# View Ansible modules
# ansible-doc -l | grep network
# Lists network modules

# Test playbook
# ansible-playbook --check configure_vlan.yml
# Dry run (no changes)

# Configuration management enables:
# - Consistent configurations
# - Version control
# - Automation
# - Reduced errors`,
				},
				{
					Title: "Network Testing and Validation",
					Content: `Testing ensures network changes work correctly.

**Testing Types:**

**Unit Testing:**
- Test individual components
- Isolated testing
- Fast execution
- Early detection

**Integration Testing:**
- Test components together
- End-to-end testing
- Realistic scenarios
- Catch integration issues

**Regression Testing:**
- Test after changes
- Ensure nothing broke
- Automated testing
- Continuous validation

**Network Simulation:**

**Mininet:**
- SDN emulator
- Virtual network
- Test topologies
- OpenFlow testing

**GNS3:**
- Network simulator
- Real device images
- Complex topologies
- Realistic testing

**Cisco Packet Tracer:**
- Educational tool
- Cisco devices
- Learning tool
- Limited features

**Testing Tools:**

**pyATS:**
- Python testing framework
- Cisco devices
- Automated testing
- Validation

**Robot Framework:**
- Keyword-driven
- Network libraries
- Test automation
- Reporting

**pytest:**
- Python testing
- Flexible
- Plugins
- Network testing

**Validation:**
- Verify configuration
- Test connectivity
- Check performance
- Validate policies

**Test Automation:**
- Automated test execution
- Continuous testing
- CI/CD integration
- Faster feedback`,
					CodeExamples: `# Network testing

# Mininet (SDN testing)
sudo mn --topo linear,3 --controller=remote,ip=127.0.0.1
# Creates test topology
pingall
# Tests connectivity in Mininet

# pyATS testing
# from pyats import aetest
# class TestConnectivity(aetest.Testcase):
#     def test_ping(self):
#         result = device.ping('192.168.1.1')
#         assert result['success']

# Robot Framework
# *** Test Cases ***
# Test Connectivity
#     Ping    192.168.1.1

# pytest
# def test_vlan_configuration():
#     result = configure_vlan(100, "Sales")
#     assert result['success']

# Network validation
# Verify configuration applied
# Test connectivity
# Check performance
# Validate policies

# Automated testing
# Run tests automatically
# Integrate with CI/CD
# Continuous validation`,
				},
				{
					Title: "Infrastructure as Code for Networks",
					Content: `Infrastructure as Code (IaC) manages networks like software.

**What is IaC?**
- Define infrastructure in code
- Version controlled
- Repeatable
- Testable
- Documented

**IaC Benefits:**
- Version control
- Consistency
- Repeatability
- Testing
- Documentation
- Collaboration

**IaC Tools:**

**Terraform:**
- HashiCorp tool
- Declarative
- Multi-cloud
- State management
- Network providers

**Terraform Network Providers:**
- AWS, Azure, GCP
- Cisco, Juniper
- Many vendors
- Extensible

**Pulumi:**
- General purpose languages
- Python, Go, TypeScript
- Real programming
- More flexible

**Ansible:**
- Configuration management
- Can be used for IaC
- YAML-based
- Agentless

**Terraform Example:**
- Define resources
- Provider configuration
- State management
- Apply changes

**IaC Workflow:**
1. Write code
2. Review code
3. Test code
4. Apply changes
5. Verify results

**IaC Best Practices:**
- Use version control
- Modularize code
- Document code
- Test changes
- Review before apply
- Use state management

**Network IaC Use Cases:**
- Cloud networks (VPCs)
- Network policies
- Firewall rules
- Load balancers
- VPNs`,
					CodeExamples: `# Infrastructure as Code

# Terraform example
# provider "aws" {
#   region = "us-east-1"
# }
#
# resource "aws_vpc" "main" {
#   cidr_block = "10.0.0.0/16"
#   tags = {
#     Name = "main-vpc"
#   }
# }
#
# resource "aws_subnet" "public" {
#   vpc_id     = aws_vpc.main.id
#   cidr_block = "10.0.1.0/24"
# }

# Terraform commands
# terraform init
# terraform plan
# terraform apply
# terraform destroy

# Pulumi example (Python)
# import pulumi
# import pulumi_aws as aws
#
# vpc = aws.ec2.Vpc("main-vpc",
#     cidr_block="10.0.0.0/16"
# )

# Pulumi commands
# pulumi up
# pulumi preview
# pulumi destroy

# IaC enables:
# - Version control
# - Repeatability
# - Testing
# - Collaboration
# - Documentation`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
