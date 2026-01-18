package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          210,
			Title:       "Routing Fundamentals",
			Description: "Learn routing concepts: static vs dynamic routing, routing tables, default gateways, and how packets are routed through networks.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Routing Basics",
					Content: `Routing is how packets move from source to destination across networks.

**What is Routing?**
- Process of selecting path for traffic
- Routers make routing decisions
- Based on routing tables
- Uses IP addresses (Layer 3)

**Routing Table:**
- Database of routes
- Lists networks and next hops
- Used to make forwarding decisions
- Can be static or dynamic

**Route Components:**
- **Destination**: Network or host
- **Gateway**: Next hop router
- **Interface**: Outgoing interface
- **Metric**: Cost/distance

**Default Gateway:**
- Route for unknown destinations
- Usually points to internet router
- 0.0.0.0/0 route
- Essential for internet access

**Routing Process:**
1. Router receives packet
2. Checks destination IP
3. Looks up in routing table
4. Finds best match (longest prefix)
5. Forwards to next hop
6. Repeats until destination

**Types of Routes:**

**Directly Connected:**
- Networks directly attached to router
- No gateway needed
- Automatically added

**Static Routes:**
- Manually configured
- Simple, predictable
- Doesn't adapt

**Dynamic Routes:**
- Learned from routing protocols
- Adapts to changes
- OSPF, BGP, RIP`,
					CodeExamples: `# View routing table
ip route show
route -n
# Shows all routes

# View default route
ip route show default
# Shows default gateway

# Add static route
sudo ip route add 192.168.2.0/24 via 192.168.1.1
# Routes network through gateway

# Add default route
sudo ip route add default via 192.168.1.1
# Sets default gateway

# Delete route
sudo ip route del 192.168.2.0/24

# View route for specific destination
ip route get 8.8.8.8
# Shows which route will be used

# View routing table with metrics
ip route show
# Shows routes with priorities

# Test routing
traceroute google.com
# Shows path packets take

# View connected routes
ip route show
# Directly connected networks shown`,
				},
				{
					Title: "Static vs Dynamic Routing",
					Content: `Understanding when to use static vs dynamic routing is crucial.

**Static Routing:**
- Manually configured routes
- Administrator sets each route
- Simple, predictable
- No protocol overhead
- Doesn't adapt to changes

**When to Use Static:**
- Small networks
- Simple topologies
- Predictable traffic
- Security-sensitive paths
- Default routes

**Dynamic Routing:**
- Routes learned automatically
- Routing protocols exchange information
- Adapts to network changes
- More complex
- Protocol overhead

**When to Use Dynamic:**
- Large networks
- Complex topologies
- Multiple paths
- Need redundancy
- Frequent changes

**Routing Protocols:**

**Distance Vector:**
- RIP (Routing Information Protocol)
- Simple, slow convergence
- Hop count metric

**Link State:**
- OSPF (Open Shortest Path First)
- Fast convergence
- Cost-based metric
- Popular in enterprises

**Path Vector:**
- BGP (Border Gateway Protocol)
- Internet routing
- Policy-based
- Very complex

**Hybrid:**
- EIGRP (Cisco proprietary)
- Combines features`,
					CodeExamples: `# Static routing examples

# Add static route
sudo ip route add 10.0.0.0/8 via 192.168.1.1
sudo ip route add 172.16.0.0/12 via 192.168.1.1

# Add route with metric
sudo ip route add 192.168.2.0/24 via 192.168.1.1 metric 100

# Make route persistent
# Add to /etc/network/interfaces or use netplan/systemd-networkd

# View static routes
ip route show
# Shows all routes (static and dynamic)

# Dynamic routing
# Usually configured on routers
# OSPF, BGP configuration is router-specific

# Test routing
ping 192.168.2.1
# Tests if route works

# View routing protocol (if dynamic)
# Router-specific commands`,
				},
				{
					Title: "Routing Metrics and Path Selection",
					Content: `Routers use metrics to choose the best path.

**Routing Metrics:**
- Values used to compare routes
- Lower is usually better
- Different protocols use different metrics

**Common Metrics:**

**Hop Count:**
- Number of routers to destination
- Used by RIP
- Simple but not always best

**Bandwidth:**
- Link speed
- Higher bandwidth = better
- Used by OSPF

**Delay:**
- Time to reach destination
- Lower delay = better
- Used by EIGRP

**Cost:**
- Calculated value
- Combines multiple factors
- Used by OSPF

**Path Selection:**
- Router compares routes
- Chooses lowest metric
- If tie, uses other criteria
- Load balancing possible

**Administrative Distance:**
- Trustworthiness of route source
- Lower = more trusted
- Static: 1, OSPF: 110, BGP: 20

**Load Balancing:**
- Multiple paths with same metric
- Distributes traffic across paths
- Improves performance
- Redundancy`,
					CodeExamples: `# View route metrics
ip route show
# Shows routes with metrics/priorities

# Add route with metric
sudo ip route add 192.168.2.0/24 via 192.168.1.1 metric 100

# View route details
ip route show 192.168.2.0/24
# Shows specific route with metric

# Test path selection
traceroute destination
# Shows actual path taken

# View multiple routes to same destination
ip route show 8.8.8.8
# Shows which route is used

# View route priorities
ip route show table all
# Shows all routing tables with priorities`,
				},
				{
					Title: "Route Lookup Process",
					Content: `Understanding how routers find the best route is essential.

**Longest Prefix Match:**
- Router matches longest subnet mask first
- More specific routes preferred
- Example: /24 route chosen over /16 route
- Most specific match wins

**Route Lookup Steps:**
1. Extract destination IP from packet
2. Search routing table
3. Find longest matching prefix
4. If multiple matches, use lowest metric
5. Forward to next hop
6. If no match, use default route
7. If no default, drop packet

**Route Matching Example:**
- Destination: 192.168.1.100
- Routes:
  - 192.168.0.0/16 via 10.0.0.1
  - 192.168.1.0/24 via 10.0.0.2
- Result: /24 route chosen (longer prefix)

**Routing Decision Factors:**
- Prefix length (longest match)
- Administrative distance (trust)
- Metric (cost)
- Route type (connected > static > dynamic)

**Default Route:**
- 0.0.0.0/0 matches everything
- Used when no specific route
- Usually points to internet gateway
- Lowest priority (shortest prefix)

**Route Caching:**
- Routers cache lookup results
- Faster subsequent lookups
- Cache invalidated on changes
- Improves performance`,
					CodeExamples: `# View route lookup

# Get route for specific destination
ip route get 192.168.1.100
# Shows which route will be used

# Test longest prefix match
ip route add 192.168.0.0/16 via 10.0.0.1
ip route add 192.168.1.0/24 via 10.0.0.2
ip route get 192.168.1.100
# Shows /24 route is chosen

# View routing table
ip route show
# Shows all routes with prefixes

# View default route
ip route show default
# Shows 0.0.0.0/0 route

# Test routing decision
traceroute 192.168.1.100
# Shows path taken

# View route cache (if available)
# Router-specific commands

# Test route selection
ping 192.168.1.100
# Uses route lookup process`,
				},
				{
					Title: "Route Redistribution",
					Content: `Route redistribution shares routes between different routing protocols.

**What is Redistribution?**
- Sharing routes between protocols
- OSPF routes → BGP
- Static routes → OSPF
- Allows protocol interoperability
- Common in complex networks

**Redistribution Scenarios:**
- **OSPF to BGP**: Internal routes to internet
- **BGP to OSPF**: Internet routes to internal
- **Static to Dynamic**: Manual routes in protocol
- **RIP to OSPF**: Legacy to modern protocol

**Redistribution Challenges:**
- **Metric Conversion**: Different metrics
- **Loop Prevention**: Avoid routing loops
- **Filtering**: Control what's redistributed
- **Administrative Distance**: Manage preferences

**Redistribution Configuration:**
- Define source protocol
- Define destination protocol
- Set metric/seed metric
- Apply filters (route maps)
- Set administrative distance

**Route Maps:**
- Control redistribution
- Filter routes
- Modify attributes
- Match and set conditions
- Powerful tool

**Redistribution Best Practices:**
- Use route maps for filtering
- Set appropriate metrics
- Avoid loops (use tags)
- Document redistribution points
- Test carefully

**Common Redistribution:**
- OSPF → BGP (internal to external)
- BGP → OSPF (external to internal)
- Static → OSPF (manual routes)
- EIGRP → OSPF (Cisco networks)`,
					CodeExamples: `# Route redistribution (router-specific)

# View routing protocols
# Router-specific commands
# show ip route
# show ip ospf route
# show ip bgp

# Redistribution configuration
# Usually router CLI commands
# Example (Cisco-style):
# router ospf 1
# redistribute static metric 100
# redistribute bgp subnets

# View redistributed routes
# Router-specific commands

# Test redistribution
# Routes from one protocol appear in another
# Verify with routing table

# Filter redistribution
# Use route maps
# Match and set conditions

# Monitor redistribution
# Check routing tables
# Verify routes appear correctly`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          211,
			Title:       "Switching & VLANs",
			Description: "Master switching concepts: how switches work, VLANs (Virtual LANs), trunking, and network segmentation.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Switch Operation",
					Content: `Switches are Layer 2 devices that forward frames based on MAC addresses.

**How Switches Work:**
1. **Learning**: Observes source MAC addresses
2. **Forwarding**: Sends frames to correct port
3. **Filtering**: Doesn't forward unnecessary traffic
4. **Flooding**: Broadcasts unknown destinations

**MAC Address Table:**
- Maps MAC addresses to ports
- Learned from incoming frames
- Ages out unused entries (usually 5 minutes)
- Used for forwarding decisions

**Switch Forwarding:**
- **Known MAC**: Forward to specific port
- **Unknown MAC**: Flood to all ports (except source)
- **Broadcast**: Flood to all ports
- **Same port**: Filter (don't forward)

**Collision Domains:**
- Each switch port is separate collision domain
- Switches eliminate collisions
- Full duplex on each port

**Broadcast Domains:**
- Switches forward broadcasts
- All ports in same broadcast domain
- VLANs create separate broadcast domains`,
					CodeExamples: `# View switch/bridge information

# View MAC address table (Linux bridge)
bridge fdb show
# Shows learned MAC addresses

# View bridge ports
bridge link show
# Shows switch ports

# View bridge status
bridge link
# Shows port status

# Monitor switch traffic
sudo tcpdump -i eth0
# Captures traffic on switch port

# View ARP table (shows devices)
arp -a
# Shows MAC addresses on network

# View switch statistics
ethtool -S eth0
# Shows interface statistics

# View port status
ip link show
# Shows interface status (up/down)`,
				},
				{
					Title: "VLANs (Virtual LANs)",
					Content: `VLANs logically segment networks without physical separation.

**What are VLANs?**
- Virtual LANs
- Logical segmentation
- Multiple VLANs on one switch
- Isolates broadcast domains

**VLAN Benefits:**
- **Security**: Isolate sensitive traffic
- **Performance**: Reduce broadcast traffic
- **Flexibility**: Logical grouping
- **Cost**: No need for separate switches

**VLAN Types:**

**Port-based VLAN:**
- Port assigned to VLAN
- Most common
- Simple configuration

**MAC-based VLAN:**
- Device MAC determines VLAN
- Device can move ports
- Less common

**Protocol-based VLAN:**
- Protocol type determines VLAN
- Rarely used

**VLAN Ranges:**
- **Normal Range**: 1-1005
- **Extended Range**: 1006-4094
- VLAN 1: Default (usually not used)

**VLAN Tagging:**
- 802.1Q standard
- Adds VLAN ID to frame
- 4-byte tag
- Used on trunk links`,
					CodeExamples: `# VLAN configuration (Linux)

# Create VLAN interface
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip addr add 192.168.100.1/24 dev eth0.100
sudo ip link set eth0.100 up

# View VLAN interfaces
ip addr show
# Shows VLAN interfaces

# View VLAN information
ip -d link show eth0.100
# Shows VLAN details

# Remove VLAN
sudo ip link del eth0.100

# View VLAN on switch
# Usually via switch management interface

# Test VLAN connectivity
ping 192.168.100.1
# Tests VLAN communication`,
				},
				{
					Title: "Trunking and VLAN Tagging",
					Content: `Trunking carries multiple VLANs over single link.

**Trunk Links:**
- Carry multiple VLANs
- Use VLAN tagging (802.1Q)
- Connect switches
- Connect switch to router

**Access Links:**
- Single VLAN
- No tagging
- Connect end devices
- Most common

**802.1Q Tagging:**
- Adds 4-byte tag to Ethernet frame
- Contains VLAN ID (12 bits)
- 0-4094 VLAN IDs
- Preserves original frame

**Native VLAN:**
- Untagged VLAN on trunk
- Usually VLAN 1
- For backward compatibility

**Trunk Configuration:**
- Both ends must agree
- Same native VLAN
- Same allowed VLANs
- Tagging enabled

**Trunk Protocols:**
- **802.1Q**: Standard trunking
- **ISL**: Cisco proprietary (deprecated)
- **DTP**: Dynamic Trunking Protocol`,
					CodeExamples: `# Trunk configuration (Linux)

# Create trunk interface
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip link add link eth0 name eth0.200 type vlan id 200

# View trunk interfaces
ip link show type vlan
# Shows all VLAN interfaces

# Configure trunk on switch
# Usually via switch CLI or web interface

# Test trunk connectivity
# Ping devices on different VLANs through trunk

# View VLAN tags
sudo tcpdump -i eth0 -e
# Shows VLAN tags in frames`,
				},
				{
					Title: "Spanning Tree Protocol (STP)",
					Content: `STP prevents loops in switched networks.

**Why STP?**
- Switches create redundant paths
- Redundancy causes loops
- Loops cause broadcast storms
- STP blocks redundant paths

**How STP Works:**
- Elects root bridge
- Finds best path to root
- Blocks redundant paths
- Maintains tree topology

**STP States:**
- **Blocking**: Port blocked (no forwarding)
- **Listening**: Preparing to forward
- **Learning**: Learning MAC addresses
- **Forwarding**: Active forwarding
- **Disabled**: Administratively down

**Root Bridge Election:**
- Lowest bridge ID wins
- Bridge ID = Priority + MAC
- Usually configured manually
- Critical for performance

**Port Roles:**
- **Root Port**: Best path to root
- **Designated Port**: Best path on segment
- **Blocked Port**: Redundant path (blocked)

**STP Variants:**

**RSTP (Rapid STP):**
- Faster convergence (seconds vs minutes)
- Additional port states
- Backward compatible
- 802.1w standard

**MSTP (Multiple STP):**
- Multiple spanning trees
- One per VLAN group
- Better load balancing
- 802.1s standard

**STP Best Practices:**
- Configure root bridge manually
- Use RSTP for faster convergence
- Use MSTP for multiple VLANs
- Monitor STP status`,
					CodeExamples: `# STP configuration (switch-specific)

# View STP status
# Switch CLI: show spanning-tree
# Shows root bridge, port states

# Configure root bridge
# Switch CLI: spanning-tree root primary
# Makes switch root bridge

# View STP on Linux bridge
bridge link show
# Shows STP status

# Enable STP on bridge
sudo ip link set dev br0 type bridge stp_state 1

# View STP statistics
# Switch-specific commands

# Test STP
# Create loop, verify STP blocks it
# Remove link, verify STP unblocks

# Monitor STP convergence
# Watch port state changes`,
				},
				{
					Title: "Inter-VLAN Routing",
					Content: `Inter-VLAN routing allows communication between VLANs.

**Why Inter-VLAN Routing?**
- VLANs isolate traffic
- Devices in different VLANs can't communicate
- Need routing for inter-VLAN traffic
- Router or Layer 3 switch required

**Router-on-a-Stick:**
- Single router interface
- Multiple sub-interfaces (one per VLAN)
- 802.1Q trunk to switch
- Each sub-interface in different VLAN

**Router-on-a-Stick Configuration:**
- Router: Create sub-interfaces with VLAN tags
- Switch: Configure trunk port
- Router sub-interfaces have IPs in each VLAN
- Router routes between VLANs

**Layer 3 Switch:**
- Switch with routing capability
- Routes between VLANs internally
- Faster than router-on-a-stick
- No external router needed

**Layer 3 Switch Benefits:**
- Faster (hardware routing)
- Lower latency
- More ports
- Integrated solution

**Inter-VLAN Routing Process:**
1. Host in VLAN 10 sends to VLAN 20
2. Packet goes to default gateway (router/L3 switch)
3. Router/L3 switch routes to VLAN 20
4. Packet delivered to destination

**Configuration Example:**
- VLAN 10: 192.168.10.0/24
- VLAN 20: 192.168.20.0/24
- Router: 192.168.10.1 (VLAN 10), 192.168.20.1 (VLAN 20)
- Hosts use router as gateway`,
					CodeExamples: `# Inter-VLAN routing (Linux)

# Router-on-a-stick setup
# Create VLAN sub-interfaces
sudo ip link add link eth0 name eth0.10 type vlan id 10
sudo ip link add link eth0 name eth0.20 type vlan id 20

# Assign IPs to sub-interfaces
sudo ip addr add 192.168.10.1/24 dev eth0.10
sudo ip addr add 192.168.20.1/24 dev eth0.20

# Enable interfaces
sudo ip link set eth0.10 up
sudo ip link set eth0.20 up

# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1

# Test inter-VLAN routing
ping 192.168.20.100
# From VLAN 10 to VLAN 20

# View routing table
ip route show
# Shows routes to VLANs

# View VLAN interfaces
ip addr show type vlan
# Shows all VLAN sub-interfaces`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          212,
			Title:       "NAT & Port Forwarding",
			Description: "Understand Network Address Translation (NAT): how it works, types of NAT, and port forwarding for services.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "NAT Basics",
					Content: `NAT translates private IPs to public IPs.

**Why NAT?**
- IPv4 address shortage
- Private IPs not routable on internet
- Security (hides internal network)
- Cost savings

**NAT Types:**

**Static NAT:**
- One-to-one mapping
- Public IP per private IP
- Always same mapping
- Used for servers

**Dynamic NAT:**
- Pool of public IPs
- Assigned as needed
- Many-to-many mapping
- Less common

**PAT (Port Address Translation):**
- Many-to-one mapping
- Uses ports to distinguish
- Most common (home routers)
- Also called NAT overload

**NAT Process:**
1. Internal host sends packet
2. NAT router translates source IP/port
3. Packet sent to internet
4. Response comes back
5. NAT router translates back
6. Packet forwarded to internal host

**NAT Table:**
- Tracks translations
- Maps internal IP:port to external IP:port
- Times out inactive entries`,
					CodeExamples: `# View NAT (if router)

# Check if behind NAT
curl ifconfig.me
# Shows public IP
ip addr show
# Shows private IP
# If different, behind NAT

# View NAT table (on router)
# Usually router-specific commands

# Test NAT
# From internal network, access external
# External sees router's public IP

# View NAT statistics
# Router-specific commands

# Port forwarding test
# Configure port forward on router
# Test from external network`,
				},
				{
					Title: "Port Forwarding",
					Content: `Port forwarding maps external ports to internal services.

**What is Port Forwarding?**
- Maps external port to internal IP:port
- Allows external access to internal services
- Common for: web servers, game servers, remote access

**Port Forwarding Example:**
- External: Router public IP:8080
- Internal: 192.168.1.100:80
- External access to 8080 forwards to internal 80

**Common Uses:**
- Web servers
- Game servers
- Remote desktop
- File servers
- VPN servers

**Security Considerations:**
- Exposes internal services
- Use strong passwords
- Keep services updated
- Consider VPN instead

**DMZ (Demilitarized Zone):**
- Special network segment
- Less protected
- For public-facing servers
- Isolated from internal network`,
					CodeExamples: `# Port forwarding configuration
# Usually done on router web interface

# Test port forwarding
# From external network:
telnet router-public-ip 8080
# Should connect to internal service

# Test from internal network
curl localhost:80
# Tests service directly

# View port forwarding rules
# Router-specific commands

# Common port forwards:
# 80 -> web server
# 443 -> HTTPS server
# 22 -> SSH server
# 3389 -> RDP server
# 25565 -> Minecraft server`,
				},
				{
					Title: "NAT Types and Variations",
					Content: `Understanding different NAT types and their characteristics.

**Static NAT:**
- One-to-one mapping
- Public IP per private IP
- Always same mapping
- Used for: Servers, predictable access
- Example: 203.0.113.1 → 192.168.1.10

**Dynamic NAT:**
- Pool of public IPs
- Assigned from pool as needed
- Many-to-many mapping
- Less common than PAT
- Used for: Multiple devices, limited public IPs

**PAT (Port Address Translation):**
- Many-to-one mapping
- Uses ports to distinguish
- Most common (home routers)
- Also called NAT overload
- Example: Multiple devices → One public IP

**NAT64:**
- Translates IPv6 to IPv4
- Allows IPv6-only networks
- Access IPv4 internet
- Used in IPv6 transition

**NAT66:**
- IPv6 to IPv6 NAT
- Less common (IPv6 has enough addresses)
- Used for: Privacy, renumbering

**NAT Characteristics:**

**Full Cone NAT:**
- Same external port for all connections
- Any external host can connect
- Least restrictive

**Restricted Cone NAT:**
- Same external port
- Only previously contacted hosts
- More restrictive

**Port Restricted Cone NAT:**
- Same external port
- Only previously contacted host:port
- More restrictive

**Symmetric NAT:**
- Different external port per connection
- Most restrictive
- Hardest for P2P applications`,
					CodeExamples: `# Check NAT type

# View public IP
curl ifconfig.me
# Shows public IP

# View private IP
ip addr show
# Shows private IP

# Compare IPs
# If different, behind NAT

# Test NAT behavior
# Multiple connections from same host
# Check if external port changes

# View NAT table (router)
# Router-specific commands
# show ip nat translations

# Test NAT traversal
# P2P applications test NAT type
# Determines connectivity method

# Check NAT type programmatically
# Use STUN server
# Determines NAT characteristics`,
				},
				{
					Title: "NAT Troubleshooting",
					Content: `Common NAT issues and how to resolve them.

**Common NAT Problems:**

**No Internet Access:**
- Check NAT table
- Verify default route
- Check firewall rules
- Verify ISP connectivity

**Intermittent Connectivity:**
- NAT table exhaustion
- Timeout issues
- Connection limits
- Check NAT table size

**Port Forwarding Not Working:**
- Verify port forward rules
- Check firewall allows traffic
- Verify service is running
- Test from internal network first

**P2P Applications Failing:**
- NAT type too restrictive
- UPnP not enabled
- Port forwarding needed
- Consider VPN

**Troubleshooting Steps:**

**1. Verify NAT is Working:**
- Check public vs private IP
- Test internet connectivity
- View NAT translations

**2. Check NAT Table:**
- View active translations
- Check for exhaustion
- Monitor table size

**3. Verify Configuration:**
- Check NAT rules
- Verify port forwards
- Check firewall rules

**4. Test Connectivity:**
- From internal network
- From external network
- Different protocols

**5. Check Logs:**
- Router logs
- Firewall logs
- Application logs`,
					CodeExamples: `# NAT troubleshooting

# Check if behind NAT
curl ifconfig.me
ip addr show
# Compare public vs private IP

# Test internet connectivity
ping 8.8.8.8
# Tests basic connectivity

# View NAT translations (router)
# Router-specific: show ip nat translations
# Shows active NAT mappings

# Check NAT statistics
# Router-specific commands
# Shows NAT usage, errors

# Test port forwarding
# From external:
telnet public-ip port
# Tests port forward

# From internal:
telnet internal-ip port
# Tests service directly

# View connection tracking (Linux)
sudo conntrack -L
# Shows NAT connections

# Clear NAT table (if needed)
# Router-specific commands
# Clears all translations

# Monitor NAT activity
# Watch NAT table
# Identify issues`,
				},
				{
					Title: "Advanced Port Forwarding",
					Content: `Advanced port forwarding techniques for complex scenarios.

**Port Ranges:**
- Forward range of ports
- Useful for: Multiple services, games
- Example: 8000-8010 → 192.168.1.100:8000-8010
- Maps external range to internal range

**Multiple Services:**
- Forward multiple ports to same host
- Different services on same server
- Example: 80, 443, 22 → 192.168.1.100
- Each port maps to different service

**Port Mapping:**
- Different external and internal ports
- External 8080 → Internal 80
- Useful for: Hiding service, port conflicts
- Common for web servers

**UPnP (Universal Plug and Play):**
- Automatic port forwarding
- Applications request port opens
- Dynamic configuration
- Convenient but security risk

**DMZ Configuration:**
- Forward all ports to one host
- Less secure
- Useful for: Testing, single server
- Isolate from internal network

**Port Forwarding Best Practices:**
- Only forward needed ports
- Use strong authentication
- Keep services updated
- Monitor forwarded ports
- Consider VPN alternative

**Load Balancing with Port Forwarding:**
- Forward to multiple hosts
- Distribute load
- Requires load balancer
- High availability`,
					CodeExamples: `# Advanced port forwarding

# Port range forwarding
# Router config: 8000-8010 → 192.168.1.100:8000-8010
# Forwards port range

# Multiple port forwards
# Router config:
# 80 → 192.168.1.100:80
# 443 → 192.168.1.100:443
# 22 → 192.168.1.100:22

# Port mapping
# External 8080 → Internal 80
# Hides actual service port

# Test port ranges
for port in {8000..8010}; do
  telnet public-ip $port
done
# Tests port range

# UPnP (if enabled)
# Applications automatically configure
# Check router UPnP status

# DMZ configuration
# Router: Forward all → 192.168.1.100
# Less secure, use carefully

# View all port forwards
# Router-specific commands
# Lists all configured forwards

# Test forwarded ports
nmap -p 80,443,8080 public-ip
# Scans forwarded ports`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          213,
			Title:       "Firewalls & Security",
			Description: "Learn firewall concepts: packet filtering, stateful firewalls, security zones, and basic firewall configuration.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Firewall Basics",
					Content: `Firewalls control network traffic based on rules.

**What is a Firewall?**
- Network security device
- Filters traffic by rules
- Protects network from threats
- Can be hardware or software

**Firewall Types:**

**Packet Filtering:**
- Examines each packet
- Based on: IP, port, protocol
- Fast, simple
- Stateless

**Stateful Firewall:**
- Tracks connections
- Understands context
- More secure
- Allows return traffic automatically

**Application Firewall:**
- Examines application data
- Deep packet inspection
- Most secure
- More complex

**Firewall Rules:**
- **Allow**: Permit traffic
- **Deny**: Block traffic
- **Reject**: Block and notify
- Evaluated in order
- First match wins

**Default Policies:**
- **Deny All**: Block everything, allow specific
- **Allow All**: Allow everything, block specific
- Deny all is more secure`,
					CodeExamples: `# Linux firewall (iptables)

# View firewall rules
sudo iptables -L -n -v
# -L: list, -n: numeric, -v: verbose

# View rules by chain
sudo iptables -L INPUT -n -v
sudo iptables -L OUTPUT -n -v
sudo iptables -L FORWARD -n -v

# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Deny all other input
sudo iptables -A INPUT -j DROP

# Save rules
sudo iptables-save > /etc/iptables/rules.v4

# Restore rules
sudo iptables-restore < /etc/iptables/rules.v4`,
				},
				{
					Title: "firewalld (Red Hat/Fedora)",
					Content: `firewalld is a modern firewall management tool.

**firewalld Features:**
- Zone-based configuration
- Runtime and permanent rules
- D-Bus interface
- Easy service management

**Zones:**
- **public**: Untrusted networks
- **internal**: Trusted internal networks
- **external**: External networks with masquerading
- **dmz**: DMZ networks
- **work**: Work networks
- **home**: Home networks
- **block**: All incoming blocked
- **drop**: All incoming dropped

**Common Commands:**
- firewall-cmd --list-all: Show all rules
- firewall-cmd --add-service: Add service
- firewall-cmd --add-port: Add port
- firewall-cmd --reload: Reload rules`,
					CodeExamples: `# firewalld commands

# View all rules
sudo firewall-cmd --list-all

# Add service
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --add-service=https --permanent

# Add port
sudo firewall-cmd --add-port=8080/tcp --permanent

# Remove service
sudo firewall-cmd --remove-service=http --permanent

# Reload
sudo firewall-cmd --reload

# View zones
sudo firewall-cmd --get-zones

# Set default zone
sudo firewall-cmd --set-default-zone=public

# View active zones
sudo firewall-cmd --get-active-zones`,
				},
				{
					Title: "ufw (Ubuntu)",
					Content: `ufw (Uncomplicated Firewall) simplifies firewall management.

**ufw Features:**
- Simple command-line interface
- Wrapper around iptables
- Easy to use
- Good defaults

**Common Commands:**
- ufw enable: Enable firewall
- ufw disable: Disable firewall
- ufw allow: Allow traffic
- ufw deny: Deny traffic
- ufw status: Show status

**ufw Profiles:**
- Predefined rules for services
- Easy service management
- Common services included`,
					CodeExamples: `# ufw commands

# Enable firewall
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow from specific IP
sudo ufw allow from 192.168.1.100

# Deny port
sudo ufw deny 8080/tcp

# View status
sudo ufw status
sudo ufw status verbose
sudo ufw status numbered

# Delete rule
sudo ufw delete allow 8080/tcp
# Or by number: sudo ufw delete 3

# Reset firewall
sudo ufw reset

# Allow service by name
sudo ufw allow ssh
sudo ufw allow http`,
				},
				{
					Title: "Firewall Rule Design",
					Content: `Designing effective firewall rules requires planning and best practices.

**Rule Ordering:**
- Rules evaluated top to bottom
- First match wins
- Order matters critically
- Most specific rules first

**Best Practices:**

**1. Default Deny:**
- Block everything by default
- Allow only what's needed
- More secure approach
- Principle of least privilege

**2. Specific Rules First:**
- Most specific rules at top
- General rules at bottom
- Prevents over-permissive rules
- Better security

**3. Group Related Rules:**
- Group by service
- Group by source
- Easier to manage
- Clearer logic

**4. Document Rules:**
- Add comments
- Explain purpose
- Note dates
- Track changes

**5. Regular Review:**
- Remove unused rules
- Update outdated rules
- Test rule changes
- Monitor effectiveness

**Common Rule Patterns:**

**Allow Specific Service:**
- Allow port from anywhere
- Common for: Web servers

**Allow from Network:**
- Allow from specific subnet
- More restrictive
- Better security

**Allow Established Connections:**
- Allow return traffic
- Stateful firewall feature
- Essential for connections`,
					CodeExamples: `# Firewall rule design examples

# Default deny policy
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH from specific network
sudo iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Deny everything else (implicit with default policy)

# View rules in order
sudo iptables -L -n -v --line-numbers
# Shows rules with line numbers

# Insert rule at specific position
sudo iptables -I INPUT 3 -p tcp --dport 8080 -j ACCEPT
# Inserts at position 3

# Delete rule by number
sudo iptables -D INPUT 3
# Deletes rule at position 3`,
				},
				{
					Title: "Application Layer Firewalls",
					Content: `Application layer firewalls provide deep packet inspection and advanced security.

**What is Application Layer Firewall?**
- Examines application data
- Layer 7 (Application) inspection
- Deep packet inspection (DPI)
- Understands application protocols

**Application Firewall Types:**

**Proxy Firewall:**
- Intercepts all traffic
- Acts as intermediary
- Terminates connections
- Inspects application data
- Examples: Squid, Blue Coat

**Next-Generation Firewall (NGFW):**
- Combines traditional + application firewall
- Deep packet inspection
- Intrusion prevention
- Application awareness
- Examples: Palo Alto, Fortinet

**Web Application Firewall (WAF):**
- Protects web applications
- Filters HTTP/HTTPS traffic
- Protects against: SQL injection, XSS
- Examples: ModSecurity, Cloudflare WAF

**Application Firewall Features:**
- **Protocol Analysis**: Understands protocols
- **Content Filtering**: Blocks specific content
- **SSL Inspection**: Decrypts and inspects HTTPS
- **Application Control**: Controls applications
- **Threat Detection**: Identifies threats

**Deep Packet Inspection (DPI):**
- Examines packet payload
- Not just headers
- Identifies applications
- Detects threats
- More resource intensive

**Application Firewall Benefits:**
- Better security
- Application awareness
- Content filtering
- Threat detection
- Granular control

**Considerations:**
- Higher cost
- More complex
- Performance impact
- Privacy concerns (SSL inspection)`,
					CodeExamples: `# Application firewall examples

# ModSecurity (WAF)
# Apache module for web application protection
# Filters HTTP requests
# Blocks attacks

# Squid proxy firewall
# Intercepts web traffic
# Filters content
# Caches content

# View proxy logs
# Application firewalls log detailed information
# Shows application activity

# Configure application rules
# Define allowed applications
# Block specific applications
# Set application policies

# SSL inspection
# Decrypts HTTPS traffic
# Inspects encrypted content
# Requires certificates

# Application identification
# Identifies applications by traffic patterns
# Not just port numbers
# More accurate

# Content filtering
# Blocks specific content types
# URL filtering
# Malware scanning`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          214,
			Title:       "VPN & Remote Access",
			Description: "Learn about VPNs: types, protocols (OpenVPN, WireGuard, IPsec), and remote access solutions.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "VPN Concepts",
					Content: `VPN creates secure tunnel over public network.

**What is VPN?**
- Virtual Private Network
- Encrypted connection
- Appears as private network
- Over public internet

**VPN Uses:**
- Remote access to corporate network
- Secure public WiFi
- Bypass geo-restrictions
- Connect branch offices

**VPN Types:**

**Site-to-Site VPN:**
- Connects entire networks
- Router-to-router
- Always on
- For branch offices

**Remote Access VPN:**
- Individual users
- Client-to-server
- On-demand
- For remote workers

**VPN Protocols:**

**OpenVPN:**
- Open source
- Flexible
- Uses SSL/TLS
- Very popular

**IPsec:**
- Standard protocol
- Built into many systems
- Complex configuration
- Very secure

**WireGuard:**
- Modern, fast
- Simple configuration
- Built into Linux kernel
- Growing popularity

**PPTP:**
- Legacy protocol
- Insecure
- Not recommended`,
					CodeExamples: `# VPN client commands

# OpenVPN
sudo openvpn --config client.ovpn
# Connects using config file

# WireGuard
sudo wg-quick up wg0
# Brings up WireGuard interface
sudo wg-quick down wg0
# Brings down interface

# View WireGuard status
sudo wg show
# Shows connection status

# IPsec (strongSwan)
sudo ipsec up connection-name
sudo ipsec down connection-name
sudo ipsec status

# Test VPN connectivity
ping vpn-server-ip
# Tests if VPN is working

# View VPN routes
ip route show
# Shows VPN routes added

# View VPN interface
ip addr show
# Shows VPN interface and IP`,
				},
				{
					Title: "Remote Access Methods",
					Content: `Various methods for remote access to systems.

**SSH (Secure Shell):**
- Encrypted terminal access
- Port 22
- Most common Linux remote access
- Key-based authentication

**RDP (Remote Desktop Protocol):**
- Windows remote desktop
- Port 3389
- GUI access
- Can be secured with VPN

**VNC (Virtual Network Computing):**
- Cross-platform remote desktop
- Less secure
- Usually over VPN

**VPN:**
- Secure tunnel
- Access entire network
- Most secure method
- Required for many services

**Security Best Practices:**
- Use strong authentication
- Encrypt connections
- Limit access
- Monitor connections
- Use VPN when possible

**Remote Access Comparison:**
- **SSH**: Terminal, secure, Linux
- **RDP**: GUI, Windows, secure
- **VNC**: GUI, cross-platform, less secure
- **VPN**: Network access, most secure`,
					CodeExamples: `# SSH remote access

# Connect via SSH
ssh user@hostname
ssh user@192.168.1.100

# SSH with key
ssh -i ~/.ssh/keyfile user@hostname

# SSH port forwarding
ssh -L 8080:localhost:80 user@hostname
# Forwards local 8080 to remote 80

# RDP (Windows)
# Usually via GUI or:
xfreerdp /v:hostname /u:username

# VNC
vncviewer hostname:1
# Connects to VNC server

# Test remote access
ping remote-host
# Tests connectivity before connecting`,
				},
				{
					Title: "VPN Protocols Comparison",
					Content: `Comparing VPN protocols helps choose the right solution.

**OpenVPN:**
- Open source
- Very flexible
- Uses SSL/TLS
- Cross-platform
- Highly configurable
- Good performance
- **Best for**: General purpose, flexibility needed

**WireGuard:**
- Modern, fast
- Simple configuration
- Built into Linux kernel
- Low overhead
- Modern cryptography
- **Best for**: Performance, simplicity, Linux

**IPsec:**
- Standard protocol
- Built into many systems
- Very secure
- Complex configuration
- Widely supported
- **Best for**: Enterprise, interoperability

**PPTP:**
- Legacy protocol
- Insecure
- Not recommended
- **Avoid**: Security vulnerabilities

**SSTP:**
- Microsoft protocol
- Uses SSL/TLS
- Windows native
- **Best for**: Windows environments

**L2TP/IPsec:**
- Combines L2TP and IPsec
- More secure than PPTP
- Widely supported
- **Best for**: Compatibility

**Protocol Comparison:**

**Security:**
- WireGuard: Excellent (modern crypto)
- OpenVPN: Excellent (proven)
- IPsec: Excellent (standard)
- PPTP: Poor (avoid)

**Performance:**
- WireGuard: Best (kernel-based)
- OpenVPN: Good
- IPsec: Good
- PPTP: Poor

**Ease of Use:**
- WireGuard: Easiest
- OpenVPN: Moderate
- IPsec: Complex
- PPTP: Easy (but insecure)`,
					CodeExamples: `# VPN protocol comparison

# OpenVPN
sudo openvpn --config client.ovpn
# Flexible, configurable

# WireGuard
sudo wg-quick up wg0
# Simple, fast

# IPsec (strongSwan)
sudo ipsec up connection-name
# Standard, complex

# Test VPN performance
iperf3 -c vpn-server-ip
# Tests throughput

# Compare protocols
# Test each protocol
# Measure: speed, latency, CPU usage

# View VPN status
sudo wg show  # WireGuard
sudo ipsec status  # IPsec
# Shows connection status

# Monitor VPN traffic
sudo tcpdump -i wg0  # WireGuard
sudo tcpdump -i ipsec0  # IPsec
# Shows VPN traffic`,
				},
				{
					Title: "Site-to-Site VPN Configuration",
					Content: `Site-to-site VPN connects entire networks together.

**What is Site-to-Site VPN?**
- Connects two or more networks
- Router-to-router connection
- Always-on tunnel
- Appears as single network

**Site-to-Site VPN Uses:**
- Connect branch offices
- Connect data centers
- Extend corporate network
- Secure inter-site communication

**Site-to-Site VPN Types:**

**IPsec Site-to-Site:**
- Standard IPsec tunnel
- Router-to-router
- Encrypted tunnel
- Common in enterprises

**GRE over IPsec:**
- Generic Routing Encapsulation
- Carries any protocol
- IPsec provides encryption
- More flexible

**DMVPN:**
- Dynamic Multipoint VPN
- Hub-and-spoke or mesh
- Dynamic tunnels
- Cisco technology

**Site-to-Site Configuration:**

**Router A (Site 1):**
- Local network: 192.168.1.0/24
- Remote network: 192.168.2.0/24
- Peer IP: Router B public IP
- Pre-shared key or certificates

**Router B (Site 2):**
- Local network: 192.168.2.0/24
- Remote network: 192.168.1.0/24
- Peer IP: Router A public IP
- Same pre-shared key

**Site-to-Site Benefits:**
- Secure inter-site communication
- Appears as single network
- Always-on connection
- Cost-effective vs leased lines

**Considerations:**
- Requires public IPs
- Router configuration needed
- Bandwidth limitations
- Latency considerations`,
					CodeExamples: `# Site-to-site VPN (router-specific)

# IPsec site-to-site configuration
# Router A:
# crypto ipsec transform-set ESP-AES256-SHA esp-aes 256 esp-sha-hmac
# crypto map VPN-MAP 10 ipsec-isakmp
# set peer Router-B-Public-IP
# set transform-set ESP-AES256-SHA
# match address VPN-ACL

# Router B:
# Similar configuration
# Peer points to Router A

# Test site-to-site VPN
ping 192.168.2.1
# From Site 1 to Site 2
# Should work through VPN

# View VPN tunnel status
# Router-specific: show crypto ipsec sa
# Shows active tunnels

# Monitor site-to-site traffic
# Router-specific commands
# Shows traffic through tunnel

# Troubleshoot site-to-site
# Check tunnel status
# Verify routing
# Test connectivity`,
				},
				{
					Title: "VPN Troubleshooting",
					Content: `Common VPN issues and troubleshooting steps.

**Common VPN Problems:**

**Connection Fails:**
- Check firewall rules
- Verify ports open
- Check credentials
- Verify server reachable

**Slow Performance:**
- Check bandwidth
- Verify encryption settings
- Check server load
- Consider protocol change

**Intermittent Disconnections:**
- Check network stability
- Verify keepalive settings
- Check timeout values
- Monitor logs

**Routing Issues:**
- Verify routes added
- Check route conflicts
- Verify split tunneling
- Check DNS settings

**Troubleshooting Steps:**

**1. Verify Connectivity:**
- Ping VPN server
- Test port connectivity
- Check firewall rules

**2. Check Configuration:**
- Verify credentials
- Check protocol settings
- Verify certificates
- Check IP addresses

**3. Review Logs:**
- Client logs
- Server logs
- System logs
- Look for errors

**4. Test Components:**
- Test authentication
- Test encryption
- Test routing
- Test DNS

**5. Protocol-Specific:**
- OpenVPN: Check config file
- WireGuard: Check keys, peers
- IPsec: Check phase 1/2

**Common Solutions:**
- Update VPN client
- Check firewall rules
- Verify certificates
- Adjust MTU size
- Change protocol`,
					CodeExamples: `# VPN troubleshooting

# Check VPN connectivity
ping vpn-server-ip
# Tests basic connectivity

# Test VPN port
telnet vpn-server-ip 1194  # OpenVPN
telnet vpn-server-ip 51820  # WireGuard
# Tests if port is open

# View VPN status
sudo wg show  # WireGuard
sudo ipsec status  # IPsec
# Shows connection status

# Check VPN routes
ip route show
# Shows routes added by VPN

# View VPN logs
sudo journalctl -u wg-quick@wg0  # WireGuard
sudo tail -f /var/log/openvpn.log  # OpenVPN
# Shows VPN logs

# Test DNS through VPN
dig @vpn-dns-server example.com
# Tests DNS resolution

# Check MTU
ping -M do -s 1472 vpn-server-ip
# Tests MTU size

# View VPN interface
ip addr show wg0  # WireGuard
ip addr show tun0  # OpenVPN
# Shows VPN interface

# Monitor VPN traffic
sudo tcpdump -i wg0
# Shows VPN traffic

# Restart VPN
sudo wg-quick down wg0
sudo wg-quick up wg0
# Restarts WireGuard`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
