package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          200,
			Title:       "Introduction to Networking",
			Description: "Learn the fundamentals of computer networks: what they are, why they matter, and basic networking concepts.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is a Computer Network?",
					Content: `A computer network is a collection of interconnected devices (computers, servers, routers, switches, and other hardware) that can communicate with each other and share resources. Networks enable devices to exchange data, share hardware and software resources, and provide services to users and applications.

**Why Networks Exist:**

**The Problem Networks Solve:**
Before networks, each computer was isolated. To share data, you had to physically transfer files using floppy disks or other storage media. Networks eliminate this limitation, enabling instant communication and resource sharing.

**Historical Context:**
- **1960s**: ARPANET (precursor to internet) developed
- **1970s**: Ethernet protocol developed
- **1980s**: Local Area Networks (LANs) become common
- **1990s**: Internet becomes publicly available
- **2000s**: Wireless networking (WiFi) becomes widespread
- **2010s**: Cloud computing and mobile networks
- **2020s**: 5G, edge computing, IoT networks
- **2024-2025**: 6G research, network automation, AI-powered networking, zero-trust architectures

**Modern Networking Best Practices (2024-2025):**

**1. Network Automation:**
- **Infrastructure as Code**: Define networks programmatically
- **Automation Tools**: Ansible, Terraform, Python scripts
- **API-Driven**: Use APIs instead of manual configuration
- **Benefits**: Consistency, faster deployment, reduced errors

**2. Zero Trust Architecture:**
- **Never Trust, Always Verify**: Verify every connection
- **Micro-segmentation**: Isolate network segments
- **Identity-Based**: Access based on identity, not location
- **Continuous Monitoring**: Monitor all network activity

**3. Software-Defined Networking (SDN):**
- **Centralized Control**: Programmable network control
- **Separation of Control and Data Planes**: Flexible management
- **Network Virtualization**: Virtual networks on physical infrastructure
- **Use Cases**: Data centers, cloud networks, WAN

**4. Network Observability:**
- **Comprehensive Monitoring**: Metrics, logs, traces
- **Real-Time Visibility**: Understand network behavior
- **Proactive Problem Detection**: Identify issues before users notice
- **Tools**: Prometheus, Grafana, network analyzers

**5. Cloud Networking:**
- **Hybrid Cloud**: Connect on-premises and cloud networks
- **Multi-Cloud**: Connect multiple cloud providers
- **Edge Computing**: Deploy services closer to users
- **SD-WAN**: Software-defined wide area networking

**6. Security-First Approach:**
- **Encryption**: Encrypt data in transit (TLS, VPN)
- **Network Segmentation**: Isolate sensitive systems
- **Intrusion Detection**: Monitor for threats
- **Regular Audits**: Assess network security regularly

**Key Concepts:**

**1. Nodes:**
- **Definition**: Any device connected to the network
- **Examples**: Computers, servers, printers, smartphones, IoT devices
- **Role**: Send and receive data
- **Identification**: Each node has unique address (IP address, MAC address)

**2. Links:**
- **Definition**: Connections between nodes
- **Types**: 
  - **Physical**: Ethernet cables, fiber optic cables
  - **Wireless**: Radio waves (WiFi, Bluetooth, cellular)
- **Characteristics**: Bandwidth, latency, reliability

**3. Protocols:**
- **Definition**: Rules and standards for communication
- **Purpose**: Ensure devices can communicate effectively
- **Examples**: TCP/IP, HTTP, HTTPS, FTP, SMTP
- **Layers**: Different protocols at different network layers

**4. Topology:**
- **Definition**: Physical or logical arrangement of network devices
- **Types**: Star, bus, ring, mesh, hybrid
- **Impact**: Affects performance, reliability, cost

**Types of Networks:**

**1. LAN (Local Area Network):**

**Characteristics:**
- **Geographic Scope**: Small area (home, office, building, campus)
- **Speed**: High-speed connections (100 Mbps to 10 Gbps+)
- **Ownership**: Privately owned and managed
- **Cost**: Relatively low cost
- **Technology**: Ethernet, WiFi

**Use Cases:**
- Home networks (connecting computers, phones, smart devices)
- Office networks (connecting workstations, printers, servers)
- School networks (connecting classrooms, labs, administrative systems)
- Small business networks

**Examples:**
- Home WiFi network connecting laptops, phones, smart TVs
- Office Ethernet network connecting workstations and servers
- School network connecting classrooms and computer labs

**Advantages:**
- High speed and low latency
- Easy to manage and secure
- Cost-effective for small areas
- Reliable (controlled environment)

**2. WAN (Wide Area Network):**

**Characteristics:**
- **Geographic Scope**: Large area (cities, countries, continents)
- **Speed**: Variable (depends on connection type)
- **Ownership**: Can be private or public (internet)
- **Cost**: Higher cost (leased lines, internet connections)
- **Technology**: Leased lines, MPLS, internet, satellite

**Use Cases:**
- Corporate networks connecting multiple offices
- Internet (largest WAN)
- Banking networks
- Government networks

**Examples:**
- Corporate network connecting offices in different cities
- Internet connecting billions of devices worldwide
- Bank network connecting branches nationwide

**Advantages:**
- Connects distant locations
- Enables remote access
- Supports global operations
- Scalable to large size

**3. MAN (Metropolitan Area Network):**

**Characteristics:**
- **Geographic Scope**: City or metropolitan area
- **Speed**: High-speed (fiber optic typically)
- **Ownership**: Usually service provider or municipality
- **Cost**: Moderate cost
- **Technology**: Fiber optic, wireless (WiMAX)

**Use Cases:**
- City-wide WiFi networks
- Cable TV networks
- Municipal networks
- University campus networks spanning city

**Examples:**
- City-wide public WiFi
- Cable internet provider network
- University network connecting multiple campuses

**4. PAN (Personal Area Network):**

**Characteristics:**
- **Geographic Scope**: Very small (personal space, few meters)
- **Speed**: Low to moderate
- **Ownership**: Personal
- **Cost**: Very low
- **Technology**: Bluetooth, USB, infrared

**Use Cases:**
- Connecting personal devices
- Wireless headphones to phone
- Smartwatch to smartphone
- File transfer between nearby devices

**Examples:**
- Bluetooth connection between phone and headphones
- USB connection between computer and external drive
- Wireless mouse/keyboard connection

**5. Other Network Types:**

**CAN (Campus Area Network):**
- Connects multiple LANs within a campus
- Example: University campus with multiple buildings

**SAN (Storage Area Network):**
- Dedicated network for storage devices
- High-speed, specialized for storage access
- Example: Data center storage network

**VPN (Virtual Private Network):**
- Secure connection over public network
- Creates private network over internet
- Example: Remote worker accessing company network

**Why Networks Matter:**

**1. Resource Sharing:**
- **Files**: Share files between devices without physical transfer
- **Printers**: Multiple users share same printer
- **Storage**: Network-attached storage (NAS) for shared storage
- **Internet**: Share internet connection among multiple devices
- **Applications**: Access applications running on servers
- **Real-world**: Office network allows all employees to share printers and files

**2. Communication:**
- **Email**: Send messages instantly across the globe
- **Messaging**: Real-time messaging (Slack, Teams, WhatsApp)
- **Video Calls**: Face-to-face communication (Zoom, Teams)
- **VoIP**: Voice over IP phone calls
- **Collaboration**: Real-time collaboration on documents
- **Real-world**: Teams collaborate on projects remotely via network

**3. Centralized Management:**
- **User Accounts**: Centralized user management (Active Directory)
- **Software Updates**: Deploy updates to all devices
- **Backup**: Centralized backup systems
- **Monitoring**: Monitor all devices from central location
- **Security**: Centralized security policies and enforcement
- **Real-world**: IT department manages hundreds of computers from central console

**4. Cost Efficiency:**
- **Shared Resources**: Share expensive hardware (printers, servers)
- **Bulk Purchasing**: Better pricing for network licenses
- **Reduced Duplication**: Avoid buying same software/hardware multiple times
- **Cloud Services**: Access cloud services instead of buying hardware
- **Real-world**: Small business shares one high-end printer instead of buying multiple

**5. Scalability:**
- **Add Devices**: Easily add new devices to network
- **Expand Coverage**: Extend network to new locations
- **Scale Services**: Scale services as business grows
- **Cloud Integration**: Integrate with cloud services
- **Real-world**: Company adds new offices and connects them to corporate network

**6. Reliability and Redundancy:**
- **Backup Connections**: Multiple paths for reliability
- **Load Distribution**: Distribute load across multiple servers
- **Failover**: Automatic failover if one component fails
- **Disaster Recovery**: Backup systems in different locations
- **Real-world**: Data center uses redundant network paths for high availability

**Network Components:**

**1. NIC (Network Interface Card):**
- **What it is**: Hardware connecting device to network
- **Types**: Ethernet NIC, WiFi adapter, fiber optic NIC
- **Function**: Converts data between computer and network format
- **MAC Address**: Unique hardware address assigned to NIC
- **Example**: Ethernet card in computer, WiFi adapter in laptop

**2. Cables:**
- **Ethernet Cable**: Twisted pair copper cable (Cat5e, Cat6, Cat7)
  - Common in offices and homes
  - Speeds: 100 Mbps to 10 Gbps
  - Maximum length: ~100 meters
- **Fiber Optic Cable**: Glass or plastic fibers transmitting light
  - Very high speed (10 Gbps to 100 Gbps+)
  - Long distance (kilometers)
  - Used in data centers and long-distance connections
- **Coaxial Cable**: Used for cable internet and TV
  - Moderate speed
  - Used by cable providers

**3. Wireless:**
- **WiFi (802.11)**: Wireless local area networking
  - Standards: 802.11n, 802.11ac (WiFi 5), 802.11ax (WiFi 6)
  - Range: ~30-100 meters indoors
  - Speeds: 150 Mbps to 9.6 Gbps (WiFi 6E)
- **Bluetooth**: Short-range wireless (PAN)
  - Range: ~10 meters
  - Used for personal devices
- **Cellular**: Mobile networks (3G, 4G, 5G)
  - Wide area coverage
  - Used for mobile internet

**4. Switches:**
- **What it is**: Network device connecting devices on same network
- **Function**: Forwards data to correct destination based on MAC address
- **Layer**: Operates at Layer 2 (Data Link)
- **Types**: 
  - **Unmanaged**: Plug and play, basic switching
  - **Managed**: Configurable, advanced features (VLANs, QoS)
- **Example**: Office switch connecting all workstations

**5. Routers:**
- **What it is**: Network device connecting different networks
- **Function**: Routes data between networks based on IP address
- **Layer**: Operates at Layer 3 (Network)
- **Features**: 
  - **NAT (Network Address Translation)**: Share one public IP
  - **Firewall**: Security features
  - **DHCP**: Assign IP addresses automatically
- **Example**: Home router connecting home network to internet

**6. Servers:**
- **What it is**: Computer providing services to network clients
- **Types**:
  - **File Server**: Stores and shares files
  - **Web Server**: Hosts websites
  - **Database Server**: Stores and manages databases
  - **Mail Server**: Handles email
  - **DNS Server**: Resolves domain names to IP addresses
- **Example**: Web server hosting company website

**7. Other Components:**
- **Access Points**: Extend WiFi coverage
- **Modems**: Convert between digital and analog signals
- **Gateways**: Connect networks with different protocols
- **Bridges**: Connect network segments
- **Repeaters**: Amplify signals for longer distances

**Network Architecture:**

**Client-Server Model:**
- **Clients**: Request services (browsers, email clients)
- **Servers**: Provide services (web servers, email servers)
- **Centralized**: Services centralized on servers
- **Example**: Web browser (client) requests webpage from web server

**Peer-to-Peer (P2P) Model:**
- **Peers**: Devices act as both clients and servers
- **Decentralized**: No central server
- **Examples**: File sharing (BitTorrent), blockchain networks

**Hybrid Model:**
- **Combination**: Mix of client-server and P2P
- **Example**: Some services centralized, others distributed

**Real-World Network Examples:**

**Home Network:**
- Router connects to internet
- WiFi provides wireless access
- Devices: Laptops, phones, smart TVs, IoT devices
- Shared: Internet connection, printer, network storage

**Office Network:**
- Switches connect workstations
- Servers provide file sharing, email, applications
- Router connects to internet
- Firewall provides security
- WiFi for mobile devices

**Data Center Network:**
- High-speed switches (10 Gbps to 100 Gbps+)
- Redundant paths for reliability
- Load balancers distribute traffic
- Firewalls for security
- Multiple servers for scalability

**Cloud Network:**
- Virtual networks (VPCs)
- Software-defined networking
- Global distribution
- Auto-scaling
- Multiple availability zones

**Conclusion:**

Computer networks are the foundation of modern computing and communication. They enable everything from simple file sharing to global internet communication, from local office networks to worldwide cloud infrastructure.

Understanding networks is essential for:
- **System Administrators**: Managing network infrastructure
- **Network Engineers**: Designing and maintaining networks
- **DevOps Engineers**: Understanding cloud networking
- **Software Developers**: Building network applications
- **Cybersecurity Professionals**: Securing network infrastructure
- **Anyone in IT**: Networks are fundamental to all IT work

Key takeaways:
- **Networks enable communication** and resource sharing
- **Different network types** serve different purposes
- **Network components** work together to enable communication
- **Protocols** ensure devices can communicate
- **Topology** affects performance and reliability

As you learn networking, remember that networks are about enabling communication and sharing resources efficiently and securely. Whether you're setting up a home network or designing a global cloud infrastructure, understanding these fundamentals is essential for success.`,
					CodeExamples: `# Check network interfaces (Linux)
ip addr show
# or
ifconfig

# View network connections
netstat -i
# Shows interface statistics

# Check network status
ip link show
# Shows all network interfaces and their status

# View routing table
ip route
# Shows how packets are routed

# Test connectivity
ping google.com
# Tests if you can reach a host

# View network statistics
ss -s
# Shows socket statistics`,
				},
				{
					Title: "Network Models: OSI and TCP/IP",
					Content: `Network models provide a framework for understanding how networks work. Two main models are OSI and TCP/IP.

**OSI Model (Open Systems Interconnection):**
7-layer model describing network functions:

**Layer 7 - Application:**
- User-facing applications
- Examples: HTTP, FTP, SMTP, DNS
- Provides services to end users

**Layer 6 - Presentation:**
- Data translation, encryption, compression
- Converts data between application and network formats
- Examples: SSL/TLS, JPEG, MPEG

**Layer 5 - Session:**
- Manages sessions between applications
- Establishes, maintains, terminates connections
- Examples: NetBIOS, RPC

**Layer 4 - Transport:**
- End-to-end data delivery
- Error recovery, flow control
- Examples: TCP, UDP
- TCP: Reliable, connection-oriented
- UDP: Fast, connectionless

**Layer 3 - Network:**
- Routing packets across networks
- Logical addressing (IP addresses)
- Examples: IP, ICMP, ARP
- Routers operate here

**Layer 2 - Data Link:**
- Frame delivery on same network
- Physical addressing (MAC addresses)
- Error detection
- Examples: Ethernet, WiFi
- Switches operate here

**Layer 1 - Physical:**
- Physical transmission of bits
- Cables, signals, hardware
- Examples: Ethernet cables, fiber optic, radio waves

**TCP/IP Model (4 layers):**
Simpler, more practical model:

**Application Layer (OSI 5-7):**
- Applications and protocols
- HTTP, FTP, SMTP, DNS

**Transport Layer (OSI 4):**
- TCP, UDP
- Port numbers

**Internet Layer (OSI 3):**
- IP, ICMP
- IP addresses

**Network Access Layer (OSI 1-2):**
- Physical and data link
- Ethernet, WiFi

**Key Differences:**
- OSI: Theoretical, 7 layers
- TCP/IP: Practical, 4 layers
- TCP/IP is what's actually used`,
					CodeExamples: `# View network layers in action

# Layer 7 - Application (HTTP)
curl http://example.com
# HTTP request/response

# Layer 4 - Transport (TCP)
netstat -tuln
# Shows TCP/UDP ports
# -t: TCP, -u: UDP, -l: listening, -n: numeric

# Layer 3 - Network (IP)
ip addr show
# Shows IP addresses
ping 8.8.8.8
# ICMP (part of network layer)

# Layer 2 - Data Link (MAC)
ip link show
# Shows MAC addresses
arp -a
# Shows ARP table (IP to MAC mapping)

# Layer 1 - Physical
ethtool eth0
# Shows physical link status

# View all layers together
tcpdump -i eth0
# Captures packets at all layers`,
				},
				{
					Title: "Network Topologies",
					Content: `Network topology describes how devices are arranged and connected in a network.

**Physical Topologies:**

**Bus Topology:**
- All devices connected to single cable (backbone)
- Simple, inexpensive
- Single point of failure
- Limited cable length
- Rarely used today

**Star Topology:**
- All devices connected to central hub/switch
- Most common today
- Easy to troubleshoot
- Hub/switch is single point of failure
- Used in most LANs

**Ring Topology:**
- Devices connected in circular fashion
- Data travels in one direction
- Token Ring (obsolete)
- FDDI (Fiber Distributed Data Interface)

**Mesh Topology:**
- Every device connected to every other device
- Fully connected: n(n-1)/2 connections
- Highly redundant
- Expensive
- Used in critical networks

**Tree Topology:**
- Hierarchical structure
- Root node connects to multiple branches
- Scalable
- Used in large networks

**Logical Topologies:**

**Broadcast:**
- All devices receive all traffic
- Ethernet uses this
- Devices filter what they need

**Token Passing:**
- Token circulates, device with token can transmit
- Prevents collisions
- Used in Token Ring, FDDI

**Modern Networks:**
- Physical: Usually star topology
- Logical: Usually broadcast (Ethernet)
- Switches create virtual point-to-point connections`,
					CodeExamples: `# View network topology information

# Show network interfaces
ip link show
# Shows physical connections

# View switch/bridge information
bridge link show
# Shows bridge connections

# View network connections
netstat -an
# Shows active connections

# View routing table (logical topology)
ip route show
# Shows how network is logically organized

# View ARP table (shows local network)
arp -a
# Shows devices on same network segment

# Network discovery
nmap -sn 192.168.1.0/24
# Scans network to discover devices

# View network graph (if tools available)
# Most networks use star topology physically
# Switches create logical point-to-point connections`,
				},
				{
					Title: "Network Protocols Overview",
					Content: `Protocols are rules and standards that govern how devices communicate on a network.

**Protocol Characteristics:**
- **Syntax**: Format of data
- **Semantics**: Meaning of data
- **Timing**: When to send/receive

**Common Protocols:**

**Application Layer Protocols:**

**HTTP (HyperText Transfer Protocol):**
- Web browsing
- Port 80 (HTTP), 443 (HTTPS)
- Request/response model
- Stateless

**HTTPS (HTTP Secure):**
- Encrypted HTTP
- Uses SSL/TLS
- Port 443
- Secure web browsing

**FTP (File Transfer Protocol):**
- File transfer
- Port 21 (control), 20 (data)
- Two modes: active, passive

**SMTP (Simple Mail Transfer Protocol):**
- Email sending
- Port 25
- Used by mail servers

**POP3/IMAP:**
- Email retrieval
- POP3: Port 110
- IMAP: Port 143

**DNS (Domain Name System):**
- Resolves domain names to IP addresses
- Port 53
- Critical internet service

**SSH (Secure Shell):**
- Secure remote access
- Port 22
- Encrypted terminal access

**Transport Layer Protocols:**

**TCP (Transmission Control Protocol):**
- Connection-oriented
- Reliable delivery
- Flow control
- Error checking
- Used for: Web, email, file transfer

**UDP (User Datagram Protocol):**
- Connectionless
- Fast, low overhead
- No guarantee of delivery
- Used for: DNS, video streaming, gaming

**Network Layer Protocols:**

**IP (Internet Protocol):**
- Routes packets across networks
- IPv4: 32-bit addresses
- IPv6: 128-bit addresses
- Best-effort delivery

**ICMP (Internet Control Message Protocol):**
- Network diagnostics
- Ping uses ICMP
- Error reporting

**ARP (Address Resolution Protocol):**
- Maps IP to MAC addresses
- Local network only

**Data Link Protocols:**

**Ethernet:**
- Most common LAN protocol
- CSMA/CD (Carrier Sense Multiple Access/Collision Detection)
- Fast Ethernet: 100 Mbps
- Gigabit Ethernet: 1 Gbps

**WiFi (802.11):**
- Wireless LAN protocol
- Multiple standards: a, b, g, n, ac, ax`,
					CodeExamples: `# Test various protocols

# HTTP/HTTPS (Application Layer)
curl http://example.com
curl https://example.com

# DNS (Application Layer)
nslookup google.com
dig google.com
host google.com

# SSH (Application Layer)
ssh user@hostname

# TCP connection test
telnet google.com 80
# Tests TCP connection to port 80

# UDP test (DNS)
dig @8.8.8.8 google.com
# Uses UDP port 53

# ICMP (Network Layer)
ping google.com
# Uses ICMP echo request/reply

# View protocol statistics
netstat -s
# Shows statistics for TCP, UDP, ICMP

# View active connections by protocol
netstat -an | grep tcp
netstat -an | grep udp

# Capture packets (see all protocols)
sudo tcpdump -i eth0
# Shows packets at all layers`,
				},
				{
					Title: "Network Performance Metrics",
					Content: `Understanding network performance metrics helps diagnose issues and optimize networks.

**Bandwidth:**
- Maximum data transfer rate
- Measured in bits per second (bps)
- Common units: Mbps, Gbps
- Example: 100 Mbps connection can transfer 100 million bits/second
- Not the same as speed (latency)

**Throughput:**
- Actual data transfer rate achieved
- Usually less than bandwidth
- Affected by: congestion, protocol overhead, errors
- Real-world performance metric

**Latency (Delay):**
- Time for data to travel from source to destination
- Measured in milliseconds (ms)
- Components:
  - **Propagation delay**: Time through medium
  - **Transmission delay**: Time to put data on wire
  - **Processing delay**: Time to process packet
  - **Queuing delay**: Time waiting in queues

**Jitter:**
- Variation in latency
- Important for real-time applications
- Voice/video need low jitter
- Measured in milliseconds

**Packet Loss:**
- Percentage of packets that don't arrive
- Caused by: congestion, errors, hardware issues
- TCP retransmits lost packets
- UDP doesn't (application must handle)

**Error Rate:**
- Percentage of packets with errors
- Detected by checksums/CRCs
- Errors cause retransmissions

**Measuring Performance:**

**Speed Test:**
- Measures bandwidth/throughput
- Tools: speedtest.net, fast.com
- Tests download/upload speeds

**Ping:**
- Measures latency
- Round-trip time (RTT)
- ping hostname

**Traceroute:**
- Shows path packets take
- Measures latency to each hop
- traceroute hostname

**Factors Affecting Performance:**
- **Distance**: Longer = more latency
- **Medium**: Fiber > copper > wireless
- **Congestion**: More traffic = slower
- **Hardware**: Better equipment = better performance
- **Protocol**: TCP overhead vs UDP speed

**Performance Optimization:**
- **QoS (Quality of Service)**: Prioritize important traffic
- **Traffic Shaping**: Control bandwidth usage
- **Load Balancing**: Distribute traffic across paths
- **Caching**: Reduce redundant transfers
- **Compression**: Reduce data size`,
					CodeExamples: `# Measure bandwidth
speedtest-cli
# Tests internet speed

# Measure latency
ping -c 10 google.com
# -c: count, sends 10 packets
# Shows min/avg/max latency

# Measure jitter
ping -c 100 google.com | grep "time="
# Multiple pings to see variation

# Test throughput
iperf3 -c server-ip
# Client connects to server
# Server: iperf3 -s

# View packet loss
ping -c 100 google.com
# Shows packet loss percentage

# Traceroute (shows path and latency)
traceroute google.com
mtr google.com
# mtr combines ping and traceroute

# View network statistics
ifconfig eth0
# Shows packets sent/received, errors, drops

# Monitor network in real-time
iftop
# Shows bandwidth usage by connection

# View detailed interface stats
ethtool -S eth0
# Shows detailed Ethernet statistics

# Test with different packet sizes
ping -s 1472 google.com
# -s: packet size (MTU testing)

# Measure bandwidth between hosts
iperf3 -c server-ip -t 60
# Tests for 60 seconds

# Measure latency distribution
ping -c 100 -i 0.1 google.com | awk '/time=/ {print $7}' | cut -d= -f2 | sort -n
# Shows latency distribution`,
				},
				{
					Title: "Network Architectures",
					Content: `Network architecture defines how devices are organized and communicate in a network.

**Client-Server Architecture:**
- Centralized model
- **Server**: Provides services/resources
- **Client**: Requests services from server
- Most common architecture
- Examples: Web servers, email servers, file servers

**Client-Server Characteristics:**
- Centralized management
- Scalable (add more clients)
- Server is single point of failure
- Requires server maintenance
- Good for: Large networks, centralized resources

**Peer-to-Peer (P2P) Architecture:**
- Decentralized model
- All devices are equal (peers)
- Each device can be client and server
- No central server needed
- Examples: File sharing, blockchain, mesh networks

**P2P Characteristics:**
- No single point of failure
- Distributed resources
- Harder to manage
- Security challenges
- Good for: Small networks, distributed applications

**Hybrid Architecture:**
- Combines client-server and P2P
- Some centralized services
- Some peer-to-peer communication
- Examples: Modern cloud services, hybrid cloud

**Hybrid Characteristics:**
- Flexibility
- Best of both worlds
- More complex
- Common in enterprise networks

**Network Architecture Selection:**
- **Client-Server**: Centralized control needed, many clients
- **P2P**: Decentralized, equal peers, resilience needed
- **Hybrid**: Complex requirements, mixed needs

**Cloud Architectures:**
- **Public Cloud**: Shared infrastructure (AWS, Azure, GCP)
- **Private Cloud**: Dedicated infrastructure
- **Hybrid Cloud**: Combination of public and private
- **Multi-Cloud**: Multiple cloud providers`,
					CodeExamples: `# Client-Server Examples

# Web server (client-server)
curl http://example.com
# Client requests from server

# Check server status
systemctl status nginx
# Shows web server status

# View server connections
ss -tn | grep :80
# Shows clients connected to server

# P2P Examples

# BitTorrent (P2P file sharing)
# Multiple peers share files directly

# Blockchain (P2P network)
# Nodes communicate peer-to-peer

# Mesh Network (P2P)
# Devices connect directly to each other

# Hybrid Examples

# Cloud services
# Some centralized (authentication)
# Some distributed (content delivery)

# View network architecture
# Client-server: Central server visible
# P2P: Multiple equal connections
# Hybrid: Mix of both patterns`,
				},
				{
					Title: "Network Standards Organizations",
					Content: `Standards organizations create and maintain networking standards that ensure interoperability.

**IEEE (Institute of Electrical and Electronics Engineers):**
- Develops networking standards
- **802.11**: WiFi standards (a, b, g, n, ac, ax)
- **802.3**: Ethernet standards
- **802.1**: Bridging and management
- **802.1Q**: VLAN tagging
- **802.1X**: Port-based access control

**IETF (Internet Engineering Task Force):**
- Develops internet standards
- Creates RFCs (Request for Comments)
- Protocols: TCP/IP, HTTP, SMTP, DNS, BGP
- Open process, consensus-based
- Examples: RFC 791 (IP), RFC 793 (TCP), RFC 2616 (HTTP/1.1)

**ISO (International Organization for Standardization):**
- International standards body
- **OSI Model**: 7-layer reference model
- Works with ITU-T on networking standards
- Global standards for interoperability

**ITU-T (International Telecommunication Union - Telecommunication Standardization Sector):**
- Telecommunications standards
- Works with ISO on OSI model
- Standards for: Telephony, data networks, video
- Global reach

**IANA (Internet Assigned Numbers Authority):**
- Manages internet resources
- **IP Address Allocation**: Assigns IP blocks to RIRs
- **Port Numbers**: Assigns well-known ports
- **Protocol Numbers**: Assigns protocol identifiers
- **Domain Names**: Manages root DNS

**RIRs (Regional Internet Registries):**
- Distribute IP addresses regionally
- **ARIN**: North America
- **RIPE NCC**: Europe, Middle East, Central Asia
- **APNIC**: Asia-Pacific
- **LACNIC**: Latin America
- **AFRINIC**: Africa

**W3C (World Wide Web Consortium):**
- Web standards
- HTML, CSS, HTTP, Web APIs
- Ensures web interoperability

**Why Standards Matter:**
- **Interoperability**: Devices from different vendors work together
- **Innovation**: Common foundation enables new technologies
- **Cost**: Standards reduce development costs
- **Quality**: Standards ensure reliability`,
					CodeExamples: `# View standards in action

# IEEE 802.3 (Ethernet)
ethtool eth0
# Shows Ethernet standard compliance

# IEEE 802.11 (WiFi)
iwconfig
# Shows WiFi standard (a/b/g/n/ac/ax)

# IETF Standards (TCP/IP)
ip addr show
# Uses IETF IP standard (RFC 791)

# IANA Port Assignments
cat /etc/services
# Shows IANA-assigned port numbers

# View protocol numbers
cat /etc/protocols
# Shows IANA protocol numbers

# DNS (IETF standard)
dig google.com
# Uses IETF DNS standards (RFC 1035)

# HTTP (IETF/W3C standard)
curl -v http://example.com
# Shows HTTP headers (IETF RFC 2616)

# Check standards compliance
# Most network tools follow standards
# Ensures interoperability`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          201,
			Title:       "OSI Model & TCP/IP Deep Dive",
			Description: "Master the OSI 7-layer model and TCP/IP stack: understand each layer's function, protocols, and how data flows through networks.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "OSI Model Layer by Layer",
					Content: `Understanding each OSI layer in detail is crucial for networking.

**Layer 7 - Application Layer:**
- Closest to end user
- Provides network services to applications
- User interacts with this layer
- Examples: Web browsers, email clients, file transfer programs
- Protocols: HTTP, HTTPS, FTP, SMTP, POP3, IMAP, DNS, SSH, Telnet

**Layer 6 - Presentation Layer:**
- Data translation and formatting
- Encryption/decryption
- Compression/decompression
- Ensures data is readable by receiving application
- Examples: SSL/TLS encryption, JPEG/PNG images, ASCII/EBCDIC text

**Layer 5 - Session Layer:**
- Establishes, manages, terminates sessions
- Synchronization and dialog control
- Can resume broken sessions
- Examples: NetBIOS, RPC (Remote Procedure Call), PPTP

**Layer 4 - Transport Layer:**
- End-to-end data delivery
- Segmentation and reassembly
- Error recovery and flow control
- Port numbers identify applications
- Protocols: TCP, UDP, SCTP

**TCP (Transmission Control Protocol):**
- Connection-oriented (3-way handshake)
- Reliable (acknowledgments, retransmissions)
- Flow control (sliding window)
- Congestion control
- Used for: Web, email, file transfer

**UDP (User Datagram Protocol):**
- Connectionless
- Fast, low overhead
- No reliability guarantees
- Used for: DNS, video streaming, gaming

**Layer 3 - Network Layer:**
- Logical addressing (IP addresses)
- Routing packets across networks
- Path determination
- Fragmentation
- Protocols: IP, ICMP, ARP, IGMP

**IP (Internet Protocol):**
- IPv4: 32-bit addresses (4.3 billion addresses)
- IPv6: 128-bit addresses (340 undecillion addresses)
- Best-effort delivery (no guarantees)
- Handles routing

**Layer 2 - Data Link Layer:**
- Physical addressing (MAC addresses)
- Frame delivery on same network segment
- Error detection (CRC)
- Media access control
- Protocols: Ethernet, WiFi (802.11), PPP, Frame Relay

**Ethernet:**
- Most common LAN protocol
- CSMA/CD for collision detection
- MAC addresses: 48 bits (6 bytes)
- Frame format: Preamble, Destination MAC, Source MAC, Type, Data, CRC

**Layer 1 - Physical Layer:**
- Physical transmission of bits
- Electrical, optical, or radio signals
- Cables, connectors, hubs
- Bit synchronization
- Examples: Ethernet cables, fiber optic, radio waves, voltages`,
					CodeExamples: `# View each layer in action

# Layer 7 - Application
curl -v http://example.com
# Shows HTTP headers (application layer)

# Layer 6 - Presentation
openssl s_client -connect example.com:443
# Shows SSL/TLS handshake (presentation layer)

# Layer 4 - Transport
netstat -tuln
# Shows TCP/UDP ports (transport layer)
ss -tuln
# Modern alternative

# View TCP connections
ss -tn
# Shows TCP connections with IPs and ports

# Layer 3 - Network
ip addr show
# Shows IP addresses (network layer)
ip route show
# Shows routing table

# Layer 2 - Data Link
ip link show
# Shows MAC addresses (data link layer)
arp -a
# Shows IP to MAC mapping

# Layer 1 - Physical
ethtool eth0
# Shows physical link status
# Speed, duplex, link status

# Capture packets (all layers)
sudo tcpdump -i eth0 -v
# -v: verbose, shows more layer details

# View packet details
sudo tcpdump -i eth0 -X
# -X: shows hex and ASCII of packets`,
				},
				{
					Title: "TCP/IP Model Details",
					Content: `TCP/IP is the practical model used in real networks. Understanding it is essential.

**Application Layer:**
- Combines OSI layers 5, 6, 7
- Applications and user processes
- Protocols: HTTP, FTP, SMTP, DNS, SSH, Telnet
- Uses ports to identify services

**Well-Known Ports (0-1023):**
- 20, 21: FTP
- 22: SSH
- 23: Telnet
- 25: SMTP
- 53: DNS
- 80: HTTP
- 110: POP3
- 143: IMAP
- 443: HTTPS

**Transport Layer:**
- End-to-end communication
- TCP and UDP
- Port numbers identify applications
- Segmentation and reassembly

**TCP Features:**
- **Connection-oriented**: 3-way handshake
- **Reliable**: Acknowledgments, retransmissions
- **Flow control**: Sliding window
- **Congestion control**: Adjusts sending rate
- **Full-duplex**: Both directions simultaneously

**UDP Features:**
- **Connectionless**: No handshake
- **Fast**: Low overhead
- **Unreliable**: No guarantees
- **Simple**: Minimal features

**Internet Layer:**
- Routing and addressing
- IP protocol
- ICMP for diagnostics
- ARP for address resolution

**IP Addresses:**
- IPv4: 32 bits, dotted decimal (192.168.1.1)
- IPv6: 128 bits, hexadecimal (2001:0db8::1)
- Network portion and host portion
- Subnet masks define network boundaries

**Network Access Layer:**
- Combines OSI layers 1 and 2
- Physical transmission
- Frame formatting
- MAC addressing
- Ethernet, WiFi

**Data Encapsulation:**
1. Application data
2. Add TCP/UDP header (segment)
3. Add IP header (packet)
4. Add Ethernet header (frame)
5. Transmit as bits`,
					CodeExamples: `# TCP/IP in action

# Application Layer - HTTP
curl -v http://example.com
# Shows HTTP headers and data

# Transport Layer - TCP
telnet google.com 80
# Tests TCP connection
# Type: GET / HTTP/1.1

# View TCP connections
ss -tn
# Shows all TCP connections
ss -tun
# Shows TCP and UDP

# View listening ports
ss -tuln
# Shows what's listening on ports

# Internet Layer - IP
ip addr show
# Shows IP addresses
ping 8.8.8.8
# ICMP (part of internet layer)

# View routing
ip route show
# Shows how IP routes packets

# Network Access Layer
ip link show
# Shows MAC addresses
ethtool eth0
# Shows physical layer info

# View ARP table
arp -a
# Shows IP to MAC mappings

# Capture and analyze
sudo tcpdump -i eth0 -n
# -n: don't resolve names (faster)
# Shows all layers

# View specific protocol
sudo tcpdump -i eth0 tcp
# Only TCP packets
sudo tcpdump -i eth0 udp
# Only UDP packets
sudo tcpdump -i eth0 icmp
# Only ICMP packets`,
				},
				{
					Title: "Data Encapsulation and De-encapsulation",
					Content: `Data encapsulation is how data moves through network layers. Understanding this is fundamental.

**Encapsulation Process (Sending):**

**Step 1: Application Layer**
- User creates data (e.g., HTTP request)
- Application adds application header
- Result: Application data

**Step 2: Transport Layer**
- Adds TCP or UDP header
- Includes source/destination ports
- TCP: sequence numbers, acknowledgments
- Result: Segment (TCP) or Datagram (UDP)

**Step 3: Network Layer**
- Adds IP header
- Includes source/destination IP addresses
- Includes protocol type (TCP/UDP)
- May fragment if too large
- Result: Packet

**Step 4: Data Link Layer**
- Adds Ethernet header
- Includes source/destination MAC addresses
- Adds trailer with CRC for error detection
- Result: Frame

**Step 5: Physical Layer**
- Converts frame to bits
- Transmits as electrical/optical/radio signals
- Result: Bits on wire

**De-encapsulation Process (Receiving):**

**Step 1: Physical Layer**
- Receives bits
- Converts to frame
- Passes to data link layer

**Step 2: Data Link Layer**
- Checks CRC for errors
- Removes Ethernet header/trailer
- Checks destination MAC (accept or discard)
- Passes packet to network layer

**Step 3: Network Layer**
- Checks destination IP
- May reassemble fragments
- Removes IP header
- Passes segment to transport layer

**Step 4: Transport Layer**
- Removes TCP/UDP header
- Reassembles segments if needed
- Passes data to application layer

**Step 5: Application Layer**
- Processes application data
- Presents to user

**PDU (Protocol Data Unit) Names:**
- Application: Data
- Transport: Segment (TCP) or Datagram (UDP)
- Network: Packet
- Data Link: Frame
- Physical: Bits`,
					CodeExamples: `# View encapsulation in action

# Capture packets to see all layers
sudo tcpdump -i eth0 -X -v
# -X: hex and ASCII
# -v: verbose (shows all headers)

# View Ethernet frame (Layer 2)
sudo tcpdump -i eth0 -e
# -e: shows Ethernet header (MAC addresses)

# View IP header (Layer 3)
sudo tcpdump -i eth0 -n
# -n: shows IP addresses (doesn't resolve names)

# View TCP/UDP headers (Layer 4)
sudo tcpdump -i eth0 -v tcp
# Shows TCP header details

# View application data (Layer 7)
sudo tcpdump -i eth0 -A
# -A: shows ASCII (application data)

# Capture HTTP traffic
sudo tcpdump -i eth0 -A port 80
# Shows HTTP requests/responses

# View packet size at each layer
sudo tcpdump -i eth0 -c 1 -v
# Shows total frame size vs payload

# Analyze specific packet
# Use Wireshark for GUI analysis
# Or tcpdump with detailed output

# View MTU (Maximum Transmission Unit)
ip link show eth0
# Shows MTU (usually 1500 for Ethernet)

# Test fragmentation
ping -s 2000 google.com
# -s: packet size
# If > MTU, will fragment`,
				},
				{
					Title: "Ports and Sockets",
					Content: `Ports and sockets are essential for application communication.

**Ports:**
- 16-bit numbers (0-65535)
- Identify applications/services
- Allow multiple services on one IP address
- Transport layer concept (TCP/UDP)

**Port Ranges:**

**Well-Known Ports (0-1023):**
- Reserved for system services
- Require root/admin to bind
- Examples: 22 (SSH), 80 (HTTP), 443 (HTTPS)

**Registered Ports (1024-49151):**
- Assigned by IANA
- Used by applications
- Examples: 3306 (MySQL), 5432 (PostgreSQL)

**Dynamic/Private Ports (49152-65535):**
- Ephemeral ports
- Used by clients for connections
- Assigned dynamically

**Sockets:**
- Combination of IP address + port
- Identifies endpoint of communication
- Format: IP:Port (e.g., 192.168.1.1:80)

**Socket Types:**

**TCP Socket:**
- Connection-oriented
- Reliable
- Example: Web server (192.168.1.1:80)

**UDP Socket:**
- Connectionless
- Fast
- Example: DNS server (8.8.8.8:53)

**Common Ports:**
- 20, 21: FTP
- 22: SSH
- 23: Telnet
- 25: SMTP
- 53: DNS
- 80: HTTP
- 110: POP3
- 143: IMAP
- 443: HTTPS
- 3306: MySQL
- 5432: PostgreSQL
- 8080: HTTP alternate

**Socket States (TCP):**
- **LISTEN**: Waiting for connections
- **ESTABLISHED**: Active connection
- **TIME_WAIT**: Connection closing
- **CLOSE_WAIT**: Remote closed, local closing`,
					CodeExamples: `# View ports and sockets

# List listening ports
netstat -tuln
# -t: TCP, -u: UDP, -l: listening, -n: numeric
ss -tuln
# Modern alternative

# View all connections
netstat -an
ss -an

# View connections by protocol
ss -tn  # TCP
ss -un  # UDP

# Find process using port
sudo lsof -i :80
sudo ss -tlnp | grep :80
# Shows process ID and name

# View ports for specific process
sudo lsof -i -P -n | grep process-name

# Test port connectivity
telnet hostname 80
nc -zv hostname 80
# Tests if port is open

# Scan ports
nmap -p 80,443 hostname
nmap -p 1-1000 hostname
# Scans specified ports

# View port statistics
netstat -s
# Shows statistics by protocol

# View established connections
ss -tn | grep ESTAB
# Shows active TCP connections

# View listening sockets
ss -tln
# Shows what's listening for connections

# Find what's using port 22
sudo lsof -i :22
# Shows SSH or other service

# View socket states
ss -tn state established
# Shows established connections only`,
				},
				{
					Title: "Layer-by-Layer Protocol Examples",
					Content: `Understanding how protocols work at each layer through practical examples.

**Complete Protocol Stack Example: HTTP Request**

**Layer 7 - Application (HTTP):**
- HTTP GET request
- Headers: Host, User-Agent, Accept
- Request line: GET /index.html HTTP/1.1
- Application data

**Layer 6 - Presentation (TLS/SSL):**
- Encrypts HTTP data
- Handshake, encryption, compression
- Converts to secure format
- HTTPS uses this layer

**Layer 5 - Session:**
- Manages HTTP session
- Cookies maintain session state
- Session timeout handling
- Often combined with application layer

**Layer 4 - Transport (TCP):**
- TCP header added
- Source port: 49152 (ephemeral)
- Destination port: 80 (HTTP)
- Sequence numbers, acknowledgments
- Flow control, error recovery

**Layer 3 - Network (IP):**
- IP header added
- Source IP: 192.168.1.100
- Destination IP: 93.184.216.34
- Protocol: TCP (6)
- Routing information

**Layer 2 - Data Link (Ethernet):**
- Ethernet header added
- Source MAC: aa:bb:cc:dd:ee:ff
- Destination MAC: Router MAC
- Type: IPv4 (0x0800)
- CRC for error detection

**Layer 1 - Physical:**
- Converts to electrical signals
- Transmits on wire
- Bit-level transmission

**Packet Capture Analysis:**
- Each layer adds its header
- Total packet size increases
- Headers contain layer-specific information
- Payload contains upper layer data`,
					CodeExamples: `# Capture and analyze packets layer by layer

# Capture HTTP traffic
sudo tcpdump -i eth0 -A port 80
# -A: ASCII output (shows application data)

# View Ethernet frame (Layer 2)
sudo tcpdump -i eth0 -e
# -e: Shows Ethernet header (MAC addresses)

# View IP header (Layer 3)
sudo tcpdump -i eth0 -n -v
# -n: Numeric (IPs), -v: Verbose (shows IP header)

# View TCP header (Layer 4)
sudo tcpdump -i eth0 -v tcp
# Shows TCP header details (ports, flags, sequence)

# View all layers together
sudo tcpdump -i eth0 -X -v
# -X: Hex and ASCII, -v: Verbose

# Capture to file for analysis
sudo tcpdump -i eth0 -w capture.pcap port 80
# Saves to file

# Analyze saved capture
tcpdump -r capture.pcap -X -v
# Reads and displays with details

# Filter by layer
sudo tcpdump -i eth0 'tcp port 80'  # Transport layer
sudo tcpdump -i eth0 'host 192.168.1.1'  # Network layer
sudo tcpdump -i eth0 'ether host aa:bb:cc:dd:ee:ff'  # Data link layer

# View packet size at each layer
sudo tcpdump -i eth0 -c 1 -v
# Shows total size vs payload size

# HTTP request analysis
curl -v http://example.com
# Shows HTTP headers (Layer 7)

# View TCP connection establishment
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0'
# Shows SYN packets (3-way handshake)`,
				},
				{
					Title: "OSI vs TCP/IP Comparison",
					Content: `Understanding when and why to use OSI vs TCP/IP model.

**OSI Model:**
- 7 layers
- Theoretical reference model
- Detailed layer separation
- Used for teaching and documentation
- Not directly implemented

**TCP/IP Model:**
- 4 layers
- Practical, implemented model
- What's actually used
- Internet standard
- Simplified structure

**Layer Mapping:**

**OSI → TCP/IP:**
- Layers 5-7 → Application Layer
- Layer 4 → Transport Layer
- Layer 3 → Internet Layer
- Layers 1-2 → Network Access Layer

**When to Use OSI Model:**
- Learning networking concepts
- Troubleshooting (layer-by-layer)
- Documentation
- Teaching
- Understanding protocol relationships

**When to Use TCP/IP Model:**
- Actual implementation
- Internet protocols
- Real-world networking
- Configuration
- Troubleshooting internet issues

**Advantages of Each:**

**OSI Advantages:**
- Detailed layer separation
- Clear responsibilities
- Good for learning
- Comprehensive coverage

**TCP/IP Advantages:**
- Simpler, practical
- Actually implemented
- Internet standard
- Real-world focus

**Practical Usage:**
- **Troubleshooting**: Use OSI (layer-by-layer approach)
- **Configuration**: Use TCP/IP (actual protocols)
- **Learning**: Start with OSI, apply TCP/IP
- **Documentation**: OSI for concepts, TCP/IP for implementation`,
					CodeExamples: `# OSI Model Troubleshooting

# Layer 1 - Physical
ethtool eth0
# Check physical link status

# Layer 2 - Data Link
ip link show
# Check MAC addresses, interface status

# Layer 3 - Network
ip addr show
ping 8.8.8.8
# Check IP configuration and routing

# Layer 4 - Transport
ss -tuln
# Check ports and connections

# Layer 7 - Application
curl -v http://example.com
# Check application protocol

# TCP/IP Model Implementation

# Application Layer
curl http://example.com
dig google.com
ssh user@host

# Transport Layer
ss -tn
# Shows TCP connections

# Internet Layer
ip route show
# Shows IP routing

# Network Access Layer
ip link show
# Shows physical interfaces

# Troubleshooting Approach
# 1. Start at Layer 1 (Physical)
# 2. Work up through layers
# 3. Identify where problem occurs
# 4. Fix at that layer`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          202,
			Title:       "IP Addressing & Subnetting",
			Description: "Master IP addressing: IPv4 and IPv6 addresses, subnetting, CIDR notation, and network calculations.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "IPv4 Addressing",
					Content: `IPv4 addresses are 32-bit numbers used to identify devices on networks.

**IPv4 Address Format:**
- 32 bits divided into 4 octets (8 bits each)
- Written in dotted decimal notation
- Range: 0.0.0.0 to 255.255.255.255
- Example: 192.168.1.1

**Address Structure:**
- **Network Portion**: Identifies the network
- **Host Portion**: Identifies the device on network
- Subnet mask determines the split

**IP Address Classes (Historical):**

**Class A:**
- Range: 1.0.0.0 to 126.255.255.255
- First octet: 1-126
- Default mask: 255.0.0.0 (/8)
- 16 million hosts per network
- Used for large organizations

**Class B:**
- Range: 128.0.0.0 to 191.255.255.255
- First octet: 128-191
- Default mask: 255.255.0.0 (/16)
- 65,534 hosts per network
- Used for medium organizations

**Class C:**
- Range: 192.0.0.0 to 223.255.255.255
- First octet: 192-223
- Default mask: 255.255.255.0 (/24)
- 254 hosts per network
- Used for small networks

**Class D:**
- Range: 224.0.0.0 to 239.255.255.255
- Multicast addresses

**Class E:**
- Range: 240.0.0.0 to 255.255.255.255
- Reserved/experimental

**Private IP Addresses (RFC 1918):**
- Not routable on internet
- Used for internal networks
- Ranges:
  - 10.0.0.0/8 (10.0.0.0 to 10.255.255.255)
  - 172.16.0.0/12 (172.16.0.0 to 172.31.255.255)
  - 192.168.0.0/16 (192.168.0.0 to 192.168.255.255)

**Special Addresses:**
- 127.0.0.1: Loopback (localhost)
- 0.0.0.0: Default route/any address
- 255.255.255.255: Broadcast address`,
					CodeExamples: `# View IP addresses
ip addr show
ifconfig
# Shows all IP addresses on system

# View specific interface
ip addr show eth0
ifconfig eth0

# Add IP address
sudo ip addr add 192.168.1.100/24 dev eth0

# Remove IP address
sudo ip addr del 192.168.1.100/24 dev eth0

# Check if IP is private
# 10.x.x.x, 172.16-31.x.x, 192.168.x.x are private

# Test connectivity
ping 192.168.1.1
# Tests if gateway is reachable

# View routing table
ip route show
# Shows how IPs are routed

# Check IP configuration
hostname -I
# Shows all IP addresses

# View network information
ip -4 addr show
# Shows only IPv4 addresses

# Calculate network from IP and mask
# 192.168.1.100/24 = network 192.168.1.0/24
# Hosts: 192.168.1.1 to 192.168.1.254`,
				},
				{
					Title: "Subnet Masks and CIDR",
					Content: `Subnet masks define which part of IP address is network and which is host.

**Subnet Mask:**
- 32-bit number like IP address
- 1s = network portion
- 0s = host portion
- Written in dotted decimal or CIDR notation

**CIDR Notation:**
- Classless Inter-Domain Routing
- Format: IP/prefix-length
- Prefix length = number of network bits
- Example: 192.168.1.0/24

**Common Subnet Masks:**

**/24 (255.255.255.0):**
- 24 network bits, 8 host bits
- 256 addresses (254 usable hosts)
- Common for small networks

**/16 (255.255.0.0):**
- 16 network bits, 16 host bits
- 65,536 addresses (65,534 usable hosts)
- Common for medium networks

**/8 (255.0.0.0):**
- 8 network bits, 24 host bits
- 16,777,216 addresses
- Large networks

**/25 (255.255.255.128):**
- 25 network bits, 7 host bits
- 128 addresses (126 usable hosts)
- Subnetting example

**Subnetting:**
- Dividing network into smaller subnets
- More efficient address usage
- Better security (isolate segments)
- Reduces broadcast domains

**Subnetting Example:**
- Network: 192.168.1.0/24
- Subnet into 4 subnets:
  - 192.168.1.0/26 (64 addresses each)
  - 192.168.1.64/26
  - 192.168.1.128/26
  - 192.168.1.192/26

**Calculating Subnets:**
- 2^n = number of subnets (where n = borrowed bits)
- 2^h - 2 = usable hosts (where h = host bits)
- -2 because network and broadcast addresses`,
					CodeExamples: `# View subnet mask
ip addr show
# Shows IP with CIDR notation (e.g., /24)

# Calculate network
# IP: 192.168.1.100, Mask: /24
# Network: 192.168.1.0/24
# Broadcast: 192.168.1.255
# Hosts: 192.168.1.1 to 192.168.1.254

# View routing table (shows networks)
ip route show
# Shows networks with CIDR notation

# Add route with subnet
sudo ip route add 192.168.2.0/24 via 192.168.1.1

# Test subnet connectivity
ping 192.168.1.1
# Tests if gateway in subnet is reachable

# View network information
ipcalc 192.168.1.100/24
# If ipcalc installed, shows network details

# Manual calculation:
# /24 = 255.255.255.0
# Network: IP AND Mask
# Broadcast: Network OR (NOT Mask)
# First host: Network + 1
# Last host: Broadcast - 1

# View all IPs in subnet
nmap -sn 192.168.1.0/24
# Scans all hosts in subnet

# Check if IPs are in same subnet
# 192.168.1.100/24 and 192.168.1.200/24 = same subnet
# 192.168.1.100/24 and 192.168.2.100/24 = different subnets`,
				},
				{
					Title: "IPv6 Addressing",
					Content: `IPv6 is the next-generation IP protocol with 128-bit addresses.

**IPv6 Address Format:**
- 128 bits = 8 groups of 16 bits (hexadecimal)
- Written as 8 groups of 4 hex digits
- Separated by colons
- Example: 2001:0db8:85a3:0000:0000:8a2e:0370:7334

**IPv6 Address Shortening:**
- Leading zeros can be omitted
- Consecutive groups of zeros can be replaced with ::
- Only once per address
- Example: 2001:db8:85a3::8a2e:370:7334

**IPv6 Address Types:**

**Unicast:**
- Single interface address
- Global unicast: Internet-routable (2000::/3)
- Link-local: Same network segment only (fe80::/10)
- Unique local: Private networks (fc00::/7)
- Loopback: ::1 (like 127.0.0.1)

**Multicast:**
- Multiple interfaces
- Replaces IPv4 broadcast
- Starts with ff00::/8
- Examples: ff02::1 (all nodes), ff02::2 (all routers)

**Anycast:**
- Multiple interfaces, same address
- Routed to nearest
- Uses unicast address format
- Used for: DNS, load balancing

**IPv6 Address Scopes:**
- **Global**: Internet-routable (2000::/3)
- **Link-local**: Same network only (fe80::/10)
- **Site-local**: Deprecated (fec0::/10)
- **Unique Local**: Private (fc00::/7)

**IPv6 Configuration Methods:**

**SLAAC (Stateless Address Autoconfiguration):**
- Router advertises prefix
- Host generates address automatically
- Uses EUI-64 for interface ID
- No DHCP needed

**DHCPv6:**
- Stateful configuration
- Server assigns addresses
- More control
- Can provide other config

**Manual Configuration:**
- Static assignment
- Full control
- Requires manual management

**IPv6 vs IPv4:**
- IPv6: 128 bits, 340 undecillion addresses
- IPv4: 32 bits, 4.3 billion addresses
- IPv6: Built-in security (IPsec)
- IPv6: Better QoS support
- IPv6: Simplified header
- IPv6: No NAT needed (usually)

**Dual Stack:**
- Running IPv4 and IPv6 simultaneously
- Common transition strategy
- Both protocols active
- Applications choose protocol`,
					CodeExamples: `# View IPv6 addresses
ip -6 addr show
# Shows IPv6 addresses

# View all addresses (IPv4 and IPv6)
ip addr show
# Shows both

# Test IPv6 connectivity
ping6 ipv6.google.com
ping6 2001:4860:4860::8888
# Google's IPv6 DNS

# View IPv6 routing
ip -6 route show

# Add IPv6 address
sudo ip -6 addr add 2001:db8::1/64 dev eth0

# Add IPv6 address with scope
sudo ip -6 addr add 2001:db8::1/64 dev eth0 scope global

# View IPv6 neighbors (like ARP)
ip -6 neigh show

# Test IPv6 DNS
dig AAAA google.com
# Returns IPv6 address

# View IPv6 statistics
ip -s -6 addr show

# Check IPv6 connectivity
curl -6 http://ipv6.google.com
# Tests IPv6 HTTP connection

# View IPv6 link-local address
ip -6 addr show dev eth0
# fe80:: addresses are link-local

# View IPv6 multicast groups
ip -6 maddr show
# Shows multicast addresses

# Test IPv6 from link-local
ping6 fe80::1%eth0
# %eth0 specifies interface

# View IPv6 default route
ip -6 route show default

# Enable IPv6 forwarding
sudo sysctl -w net.ipv6.conf.all.forwarding=1`,
				},
				{
					Title: "IPv6 Advanced Configuration",
					Content: `Advanced IPv6 configuration includes address assignment, routing, and transition mechanisms.

**IPv6 Address Assignment:**

**EUI-64:**
- Extended Unique Identifier
- Generates interface ID from MAC
- Takes MAC, inserts FF:FE in middle
- Flips universal/local bit
- Example: MAC aa:bb:cc:dd:ee:ff → Interface ID aabb:ccff:fedd:eeff

**Privacy Extensions:**
- Temporary addresses
- Changes periodically
- Protects privacy
- RFC 4941

**IPv6 Routing:**
- Similar to IPv4 routing
- Uses routing table
- Supports static and dynamic routing
- OSPFv3, BGP for IPv6

**IPv6 Transition Mechanisms:**

**Dual Stack:**
- Run IPv4 and IPv6 together
- Most common approach
- Applications choose protocol

**Tunneling:**
- Encapsulate IPv6 in IPv4
- 6to4, Teredo, ISATAP
- Allows IPv6 over IPv4 networks

**Translation:**
- NAT64, 464XLAT
- Translate between IPv4 and IPv6
- Allows IPv6-only networks to access IPv4

**IPv6 Security:**
- IPsec built-in (optional in IPv4)
- Better authentication
- Protection against scanning (huge address space)
- Still need firewalls

**IPv6 Best Practices:**
- Use /64 for subnets (SLAAC requirement)
- Enable privacy extensions
- Use unique local for private networks
- Plan address allocation carefully`,
					CodeExamples: `# IPv6 advanced configuration

# View IPv6 address details
ip -6 addr show dev eth0
# Shows all IPv6 addresses with scopes

# Add IPv6 address with EUI-64
# Usually automatic with SLAAC

# Configure IPv6 privacy extensions
sudo sysctl -w net.ipv6.conf.eth0.use_tempaddr=2
# 2 = prefer temporary addresses

# View IPv6 routing table
ip -6 route show
ip -6 route show table all

# Add IPv6 default route
sudo ip -6 route add default via 2001:db8::1

# View IPv6 neighbor discovery
ip -6 neigh show
# Shows IPv6 neighbors (like ARP)

# Test IPv6 connectivity
ping6 -c 4 2001:4860:4860::8888
# Tests Google IPv6 DNS

# View IPv6 multicast
ip -6 maddr show
# Shows multicast group memberships

# IPv6 forwarding
sudo sysctl -w net.ipv6.conf.all.forwarding=1
# Enables IPv6 routing

# View IPv6 statistics
cat /proc/net/snmp6
# Shows IPv6 protocol statistics

# Test IPv6 from specific interface
ping6 -I eth0 2001:db8::1
# Uses eth0 interface

# View IPv6 address assignment
# SLAAC: Automatic
# DHCPv6: Check DHCPv6 client
# Manual: ip -6 addr add`,
				},
				{
					Title: "Subnetting Calculations",
					Content: `Mastering subnetting calculations is essential for network design.

**Key Concepts:**
- **Network Address**: First address (all host bits 0)
- **Broadcast Address**: Last address (all host bits 1)
- **Usable Hosts**: Network + 1 to Broadcast - 1
- **Subnet Mask**: Defines network/host boundary

**Calculation Steps:**

**1. Determine Network Bits:**
- From CIDR notation (/24 = 24 network bits)
- Or from subnet mask (255.255.255.0 = 24 bits)

**2. Determine Host Bits:**
- Total bits (32 for IPv4) - network bits
- /24 = 32 - 24 = 8 host bits

**3. Calculate Number of Hosts:**
- 2^host_bits = total addresses
- 2^host_bits - 2 = usable hosts
- -2 for network and broadcast

**4. Calculate Subnet Size:**
- 2^host_bits = addresses per subnet

**Example: 192.168.1.0/24**
- Network bits: 24
- Host bits: 8
- Total addresses: 2^8 = 256
- Usable hosts: 256 - 2 = 254
- Network: 192.168.1.0
- Broadcast: 192.168.1.255
- Hosts: 192.168.1.1 to 192.168.1.254

**Subnetting a Network:**

**Example: Divide 192.168.1.0/24 into 4 subnets**
- Need 4 subnets = 2^2 (borrow 2 bits)
- New prefix: /26 (24 + 2)
- Host bits: 6 (32 - 26)
- Hosts per subnet: 2^6 - 2 = 62

**Subnets:**
- 192.168.1.0/26 (192.168.1.0 to 192.168.1.63)
- 192.168.1.64/26 (192.168.1.64 to 192.168.1.127)
- 192.168.1.128/26 (192.168.1.128 to 192.168.1.191)
- 192.168.1.192/26 (192.168.1.192 to 192.168.1.255)

**Subnetting Tips:**
- Always reserve network and broadcast addresses
- Plan for future growth
- Use powers of 2 for subnet counts
- Document your subnet plan`,
					CodeExamples: `# Calculate subnet information
# Manual calculation or use tools

# Using ipcalc (if installed)
ipcalc 192.168.1.0/24
# Shows network, broadcast, hosts

# Using Python
python3 -c "
import ipaddress
net = ipaddress.ip_network('192.168.1.0/24')
print(f'Network: {net.network_address}')
print(f'Broadcast: {net.broadcast_address}')
print(f'Hosts: {net.num_addresses - 2}')
print(f'First host: {list(net.hosts())[0]}')
print(f'Last host: {list(net.hosts())[-1]}')
"

# List all subnets when dividing
python3 -c "
import ipaddress
net = ipaddress.ip_network('192.168.1.0/24')
subnets = list(net.subnets(prefixlen_diff=2))
for s in subnets:
    print(f'{s}')
"

# View current network
ip addr show eth0
# Shows IP and CIDR notation

# Calculate from IP and mask
# IP: 192.168.1.100/24
# Network: 192.168.1.0
# Broadcast: 192.168.1.255
# Range: 192.168.1.1 - 192.168.1.254

# Calculate subnet mask from CIDR
# /24 = 255.255.255.0
# /25 = 255.255.255.128
# /26 = 255.255.255.192`,
				},
				{
					Title: "VLSM (Variable Length Subnet Masking)",
					Content: `VLSM allows subnets of different sizes within the same network, optimizing address space.

**What is VLSM?**
- Variable Length Subnet Masking
- Different subnet sizes in same network
- More efficient than fixed-size subnets
- Allows optimal address allocation

**VLSM Benefits:**
- **Efficiency**: Use only needed addresses
- **Flexibility**: Different sizes for different needs
- **Optimization**: Reduce wasted addresses
- **Scalability**: Better address utilization

**VLSM Example:**

**Network: 192.168.1.0/24**

**Requirements:**
- 50 hosts needed (use /26 = 64 addresses)
- 25 hosts needed (use /27 = 32 addresses)
- 10 hosts needed (use /28 = 16 addresses)
- 5 hosts needed (use /28 = 16 addresses)

**VLSM Allocation:**
- 192.168.1.0/26 (64 addresses): 192.168.1.0-63
- 192.168.1.64/27 (32 addresses): 192.168.1.64-95
- 192.168.1.96/28 (16 addresses): 192.168.1.96-111
- 192.168.1.112/28 (16 addresses): 192.168.1.112-127
- Remaining: 192.168.1.128-255 (for future use)

**VLSM Process:**
1. List requirements (host counts)
2. Determine subnet sizes needed
3. Allocate largest subnets first
4. Fill gaps with smaller subnets
5. Document allocation

**VLSM vs Fixed Subnetting:**
- **Fixed**: All subnets same size (wasteful)
- **VLSM**: Subnets sized to needs (efficient)
- VLSM saves addresses
- VLSM requires careful planning`,
					CodeExamples: `# VLSM calculation

# Using Python for VLSM
python3 -c "
import ipaddress

# Start with /24 network
network = ipaddress.ip_network('192.168.1.0/24')

# Requirements: 50, 25, 10, 5 hosts
# Calculate subnet sizes needed
subnets = [
    ('50 hosts', 26),  # /26 = 64 addresses (62 usable)
    ('25 hosts', 27),  # /27 = 32 addresses (30 usable)
    ('10 hosts', 28),  # /28 = 16 addresses (14 usable)
    ('5 hosts', 28),  # /28 = 16 addresses (14 usable)
]

current = network.network_address
for name, prefix_len in subnets:
    subnet = ipaddress.ip_network(f'{current}/{prefix_len}', strict=False)
    print(f'{name}: {subnet}')
    current = subnet.broadcast_address + 1
"

# Manual VLSM planning
# 1. List host requirements
# 2. Calculate subnet sizes
# 3. Allocate sequentially
# 4. Document allocations

# View VLSM in routing table
ip route show
# Shows routes with different subnet masks (VLSM)

# Test VLSM connectivity
ping 192.168.1.1  # From /26 subnet
ping 192.168.1.65  # From /27 subnet
# Tests connectivity across VLSM subnets`,
				},
				{
					Title: "Supernetting and CIDR Aggregation",
					Content: `Supernetting combines multiple networks into larger networks, reducing routing table size.

**What is Supernetting?**
- Combining multiple networks into one
- Also called route aggregation
- Uses shorter subnet mask
- Reduces routing table entries

**CIDR Aggregation:**
- Classless Inter-Domain Routing
- Aggregates routes
- Reduces routing table size
- Used by ISPs and large networks

**Supernetting Example:**

**Multiple Networks:**
- 192.168.0.0/24
- 192.168.1.0/24
- 192.168.2.0/24
- 192.168.3.0/24

**Aggregated to:**
- 192.168.0.0/22 (covers all 4 networks)

**Benefits:**
- Smaller routing tables
- Faster routing lookups
- Less memory usage
- Simpler configuration

**Requirements for Aggregation:**
- Networks must be contiguous
- Must align on bit boundaries
- Common network portion
- No gaps in address space

**Aggregation Rules:**
- Networks must be sequential
- Must be powers of 2
- Must share common prefix
- Can't skip addresses

**Example Aggregation:**
- 10.0.0.0/24, 10.0.1.0/24 → 10.0.0.0/23
- 172.16.0.0/24 through 172.16.3.0/24 → 172.16.0.0/22
- 192.168.0.0/24 through 192.168.7.0/24 → 192.168.0.0/21`,
					CodeExamples: `# Supernetting calculation

# Using Python
python3 -c "
import ipaddress

# Multiple /24 networks
networks = [
    ipaddress.ip_network('192.168.0.0/24'),
    ipaddress.ip_network('192.168.1.0/24'),
    ipaddress.ip_network('192.168.2.0/24'),
    ipaddress.ip_network('192.168.3.0/24'),
]

# Aggregate to /22
supernet = ipaddress.ip_network('192.168.0.0/22')
print(f'Aggregated network: {supernet}')
print(f'Covers: {supernet.num_addresses} addresses')
print(f'Original networks: {len(networks)} networks')

# Verify all networks are in supernet
for net in networks:
    print(f'{net} in {supernet}: {net.subnet_of(supernet)}')
"

# View aggregated routes
ip route show
# Shows aggregated routes (shorter masks)

# Test aggregation
# Single route for 192.168.0.0/22
# Instead of 4 routes for /24 networks

# Aggregation in routing protocols
# OSPF, BGP support route aggregation
# Reduces routing table size`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          203,
			Title:       "DNS & DHCP",
			Description: "Understand Domain Name System (DNS) and Dynamic Host Configuration Protocol (DHCP): how they work and how to configure them.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "DNS (Domain Name System)",
					Content: `DNS translates human-readable domain names to IP addresses.

**How DNS Works:**
1. User types domain name (e.g., google.com)
2. Computer queries DNS server
3. DNS server returns IP address
4. Computer connects using IP

**DNS Hierarchy:**
- **Root Servers**: Top level (.)
- **TLD Servers**: Top-level domains (.com, .org, .net)
- **Authoritative Servers**: Own domain records
- **Recursive Resolvers**: Query on behalf of clients

**DNS Record Types:**

**A Record:**
- Maps domain to IPv4 address
- Example: google.com → 142.250.191.14

**AAAA Record:**
- Maps domain to IPv6 address
- Example: google.com → 2607:f8b0:4004:c1b::65

**CNAME Record:**
- Canonical name (alias)
- Points one domain to another
- Example: www → example.com

**MX Record:**
- Mail exchange (email servers)
- Priority and server name
- Example: mail.example.com (priority 10)

**NS Record:**
- Name server (authoritative DNS servers)
- Example: ns1.example.com

**TXT Record:**
- Text information
- Used for verification, SPF, DKIM

**PTR Record:**
- Reverse DNS (IP to domain)
- Used in reverse lookups

**DNS Query Process:**
1. Check local cache
2. Query recursive resolver
3. Resolver queries root server
4. Root refers to TLD server
5. TLD refers to authoritative server
6. Authoritative returns answer
7. Answer cached and returned`,
					CodeExamples: `# DNS queries

# Query A record
nslookup google.com
dig google.com
host google.com

# Query specific record type
dig A google.com
dig AAAA google.com
dig MX google.com
dig NS google.com
dig TXT google.com

# Query specific DNS server
dig @8.8.8.8 google.com
# Uses Google DNS

# Reverse DNS lookup
dig -x 8.8.8.8
nslookup 8.8.8.8
# Finds domain from IP

# View DNS cache (Linux)
systemd-resolve --status
# Shows DNS servers and cache

# Flush DNS cache
sudo systemd-resolve --flush-caches
# Clears local DNS cache

# Test DNS resolution
getent hosts google.com
# Uses system resolver

# View DNS configuration
cat /etc/resolv.conf
# Shows DNS servers

# Query with verbose output
dig +trace google.com
# Shows complete resolution path

# Test DNS server
dig @8.8.8.8 google.com +short
# +short: brief output`,
				},
				{
					Title: "DHCP (Dynamic Host Configuration Protocol)",
					Content: `DHCP automatically assigns IP addresses and network configuration to devices.

**DHCP Benefits:**
- Automatic IP assignment
- Centralized configuration
- Prevents IP conflicts
- Easy network changes
- Reduces administration

**DHCP Process (DORA):**

**1. Discover:**
- Client broadcasts DHCP Discover
- "I need an IP address"

**2. Offer:**
- DHCP server responds with Offer
- "Here's an IP address for you"

**3. Request:**
- Client requests the offered IP
- "I'll take that IP"

**4. Acknowledge:**
- Server acknowledges and assigns
- "IP assigned, here's configuration"

**DHCP Information Provided:**
- IP address
- Subnet mask
- Default gateway
- DNS servers
- Lease time
- Domain name

**DHCP Lease:**
- Temporary assignment
- Must be renewed
- Default: 24 hours (configurable)
- Renewal at 50% of lease time

**DHCP Server Components:**
- **DHCP Server**: Assigns addresses
- **DHCP Client**: Requests configuration
- **DHCP Relay**: Forwards requests across networks

**DHCP Scopes:**
- Range of IP addresses to assign
- Example: 192.168.1.100 to 192.168.1.200
- Exclusions: Reserved addresses
- Reservations: Fixed IP for MAC address`,
					CodeExamples: `# View DHCP configuration

# View current IP (from DHCP)
ip addr show
# Shows IP assigned by DHCP

# View DHCP lease information
cat /var/lib/dhcp/dhclient.leases
# Shows lease details

# Release DHCP lease
sudo dhclient -r eth0
# Releases current lease

# Request new lease
sudo dhclient eth0
# Requests new IP from DHCP

# View DHCP server (if running)
sudo systemctl status isc-dhcp-server
# Checks if DHCP server is running

# View DHCP client status
systemctl status dhcpcd
# Or NetworkManager for DHCP

# Manual DHCP request
sudo dhclient -v eth0
# -v: verbose, shows process

# View network configuration
nmcli device show eth0
# Shows DHCP-assigned settings

# Test DHCP server
dhclient -d eth0
# -d: debug mode, shows details

# View DNS from DHCP
cat /etc/resolv.conf
# Shows DNS servers (may be from DHCP)`,
				},
				{
					Title: "DNS Configuration",
					Content: `Configuring DNS properly ensures reliable name resolution.

**DNS Configuration Files:**

**/etc/resolv.conf:**
- Lists DNS servers
- Format: nameserver IP
- Example:
  nameserver 8.8.8.8
  nameserver 8.8.4.4

**/etc/hosts:**
- Local hostname to IP mapping
- Checked before DNS
- Format: IP hostname aliases
- Example:
  127.0.0.1 localhost
  192.168.1.100 server1

**DNS Server Options:**

**Public DNS Servers:**
- Google: 8.8.8.8, 8.8.4.4
- Cloudflare: 1.1.1.1, 1.0.0.1
- OpenDNS: 208.67.222.222, 208.67.220.220
- Quad9: 9.9.9.9

**Local DNS:**
- Router: Usually 192.168.1.1 or 192.168.0.1
- Internal DNS servers
- Faster for local names

**DNS Configuration Methods:**

**Static Configuration:**
- Edit /etc/resolv.conf directly
- May be overwritten by network manager

**NetworkManager:**
- GUI or nmcli
- Persistent configuration
- Preferred method

**systemd-resolved:**
- Modern Linux DNS resolver
- Caches DNS queries
- Integrated with systemd

**DNS Search Domains:**
- Automatically appends domain to queries
- Example: search example.com
- Query "server" becomes "server.example.com"`,
					CodeExamples: `# Configure DNS

# View current DNS
cat /etc/resolv.conf
# Shows current DNS servers

# Edit DNS (temporary)
sudo nano /etc/resolv.conf
# Add: nameserver 8.8.8.8

# Using NetworkManager
sudo nmcli connection modify "Connection Name" ipv4.dns "8.8.8.8 8.8.4.4"
sudo nmcli connection up "Connection Name"

# Using systemd-resolved
sudo systemd-resolve --set-dns=8.8.8.8 --interface=eth0

# View systemd-resolved status
systemd-resolve --status
# Shows DNS configuration

# Test DNS resolution
dig @8.8.8.8 google.com
# Tests specific DNS server

# View DNS cache
systemd-resolve --statistics
# Shows cache statistics

# Flush DNS cache
sudo systemd-resolve --flush-caches

# Add to /etc/hosts
echo "192.168.1.100 myserver" | sudo tee -a /etc/hosts
# Adds local hostname mapping

# Test hosts file
ping myserver
# Should resolve to 192.168.1.100

# Configure search domain
echo "search example.com" | sudo tee -a /etc/resolv.conf`,
				},
				{
					Title: "DNS Record Types Deep Dive",
					Content: `Understanding all DNS record types is essential for DNS management.

**Common DNS Record Types:**

**A Record:**
- Maps domain to IPv4 address
- Most common record type
- Example: example.com → 93.184.216.34
- Used for: Web servers, mail servers

**AAAA Record:**
- Maps domain to IPv6 address
- IPv6 equivalent of A record
- Example: example.com → 2001:db8::1
- Used for: IPv6-enabled services

**CNAME Record:**
- Canonical name (alias)
- Points one domain to another
- Example: www → example.com
- Cannot use for root domain
- Used for: Aliases, subdomains

**MX Record:**
- Mail exchange (email servers)
- Priority and server name
- Lower priority number = higher priority
- Example: mail.example.com (priority 10)
- Used for: Email routing

**NS Record:**
- Name server (authoritative DNS servers)
- Specifies DNS servers for domain
- Example: ns1.example.com
- Required for domain delegation
- Used for: DNS server identification

**TXT Record:**
- Text information
- Used for: Verification, SPF, DKIM, DMARC
- Example: "v=spf1 include:_spf.google.com ~all"
- Used for: Email authentication, domain verification

**PTR Record:**
- Reverse DNS (IP to domain)
- Used in reverse lookups
- Example: 34.216.184.93.in-addr.arpa → example.com
- Used for: Reverse lookups, email validation

**SRV Record:**
- Service locator
- Specifies service location
- Format: _service._protocol.domain
- Example: _http._tcp.example.com
- Used for: Service discovery

**SOA Record:**
- Start of Authority
- Domain authority information
- Primary name server, email, serial, refresh
- One per zone
- Used for: Zone management

**Other Record Types:**
- **CAA**: Certificate Authority Authorization
- **DNSKEY**: DNSSEC public key
- **DS**: Delegation Signer (DNSSEC)
- **NAPTR**: Name Authority Pointer`,
					CodeExamples: `# Query different DNS record types

# A record (IPv4)
dig A example.com
dig +short A example.com

# AAAA record (IPv6)
dig AAAA example.com
dig +short AAAA example.com

# CNAME record
dig CNAME www.example.com
dig +short CNAME www.example.com

# MX record (mail servers)
dig MX example.com
dig +short MX example.com

# NS record (name servers)
dig NS example.com
dig +short NS example.com

# TXT record
dig TXT example.com
dig TXT _dmarc.example.com
# Shows SPF, DKIM, DMARC records

# PTR record (reverse DNS)
dig -x 8.8.8.8
dig +short -x 93.184.216.34

# SRV record
dig SRV _http._tcp.example.com

# SOA record
dig SOA example.com
# Shows zone authority information

# Query all record types
dig ANY example.com
# Shows all available records

# View specific record details
dig +noall +answer MX example.com
# Shows only MX records`,
				},
				{
					Title: "DHCP Relay and Advanced Configuration",
					Content: `DHCP relay extends DHCP across network segments, enabling centralized DHCP servers.

**DHCP Relay (DHCP Helper):**
- Forwards DHCP messages across routers
- Allows single DHCP server for multiple networks
- Router acts as relay agent
- Extends DHCP to remote networks

**How DHCP Relay Works:**
1. Client broadcasts DHCP Discover
2. Relay agent receives on client network
3. Relay forwards to DHCP server (unicast)
4. Server responds to relay
5. Relay forwards response to client network
6. Client receives IP configuration

**DHCP Relay Configuration:**
- Configure on router/switch
- Specify DHCP server IP
- Enable on interfaces
- Handles multiple networks

**DHCP Options:**
- Additional configuration parameters
- Sent with IP address
- Examples: DNS servers, domain name, NTP servers

**Common DHCP Options:**
- **Option 3**: Default gateway (router)
- **Option 6**: DNS servers
- **Option 15**: Domain name
- **Option 42**: NTP servers
- **Option 51**: Lease time
- **Option 66**: TFTP server (for PXE boot)

**DHCP Scopes:**
- Range of IP addresses to assign
- Example: 192.168.1.100 to 192.168.1.200
- Exclusions: Reserved addresses
- Reservations: Fixed IP for MAC address

**DHCP Reservations:**
- Static IP for specific MAC
- Always same IP for device
- Useful for: Servers, printers, network devices
- Based on MAC address

**DHCP Lease Management:**
- Temporary assignment
- Must be renewed
- Default: 24 hours (configurable)
- Renewal at 50% of lease time
- Release: Client gives up lease
- Renew: Client extends lease`,
					CodeExamples: `# DHCP relay configuration
# Usually on router/switch
# Not typically on Linux hosts

# View DHCP lease information
cat /var/lib/dhcp/dhclient.leases
# Shows current lease details

# View DHCP options received
cat /var/lib/dhcp/dhclient.leases
# Shows all DHCP options

# Release DHCP lease
sudo dhclient -r eth0
# Releases current lease

# Request new lease
sudo dhclient eth0
# Requests new IP from DHCP

# Request with specific options
sudo dhclient -v eth0
# -v: verbose, shows DHCP process

# View DHCP server (if running)
sudo systemctl status isc-dhcp-server
# Checks if DHCP server is running

# DHCP server configuration
# /etc/dhcp/dhcpd.conf
# subnet 192.168.1.0 netmask 255.255.255.0 {
#   range 192.168.1.100 192.168.1.200;
#   option routers 192.168.1.1;
#   option domain-name-servers 8.8.8.8;
# }

# View DHCP client status
systemctl status dhcpcd
# Or NetworkManager for DHCP

# Test DHCP server
dhclient -d eth0
# -d: debug mode, shows details

# View network configuration from DHCP
nmcli device show eth0
# Shows DHCP-assigned settings

# Check DNS from DHCP
cat /etc/resolv.conf
# Shows DNS servers (may be from DHCP)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          204,
			Title:       "Network Devices & Topologies",
			Description: "Learn about network hardware: hubs, switches, routers, access points, and how they create network topologies.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Network Devices Overview",
					Content: `Understanding network devices is essential for building and troubleshooting networks.

**Hub:**
- Layer 1 device (Physical)
- Broadcasts to all ports
- No intelligence
- Obsolete (replaced by switches)
- Creates collision domain

**Switch:**
- Layer 2 device (Data Link)
- Learns MAC addresses
- Forwards frames to specific port
- Creates separate collision domains
- Most common LAN device

**Router:**
- Layer 3 device (Network)
- Routes packets between networks
- Uses IP addresses
- Connects different networks
- Can perform NAT

**Access Point (AP):**
- Wireless hub/switch
- Connects wireless to wired network
- Extends network wirelessly
- 802.11 standards (WiFi)

**Bridge:**
- Connects network segments
- Layer 2 device
- Filters by MAC address
- Less common (switches replaced)

**Modem:**
- Modulates/demodulates signals
- Converts digital to analog
- Used for internet connections
- DSL, cable modems

**Firewall:**
- Security device
- Filters traffic by rules
- Can be hardware or software
- Protects network from threats

**Load Balancer:**
- Distributes traffic
- Multiple servers
- High availability
- Application or network layer

**Device Selection:**
- **Hub**: Never (obsolete)
- **Switch**: LAN connectivity
- **Router**: Inter-network routing
- **AP**: Wireless access
- **Firewall**: Security`,
					CodeExamples: `# View network devices

# View network interfaces (devices)
ip link show
# Shows all network interfaces

# View switch/bridge information
bridge link show
# Shows bridge connections

# View router (default gateway)
ip route show default
# Shows default gateway (router)

# View ARP table (shows devices on network)
arp -a
# Shows MAC addresses (switch-learned)

# View network topology
# Physical inspection or network mapping tools

# Test connectivity through devices
ping 192.168.1.1
# Tests router connectivity

# View device information
ethtool eth0
# Shows physical link information

# View switch ports (if managed switch)
# Usually via web interface or SNMP

# Monitor network traffic
sudo tcpdump -i eth0
# Shows traffic through interface

# View device capabilities
ethtool eth0
# Shows interface capabilities`,
				},
				{
					Title: "Network Cabling and Media",
					Content: `Understanding network cabling and media types is essential for network design.

**Copper Cabling:**

**Twisted Pair:**
- Most common LAN cable
- Pairs twisted to reduce interference
- Categories: Cat5, Cat5e, Cat6, Cat6a, Cat7
- RJ-45 connectors

**Cable Categories:**
- **Cat5**: 100 Mbps, 100m
- **Cat5e**: 1 Gbps, 100m (enhanced)
- **Cat6**: 1 Gbps, 100m (better crosstalk)
- **Cat6a**: 10 Gbps, 100m
- **Cat7**: 10 Gbps, 100m (shielded)

**Coaxial Cable:**
- Used for: Cable internet, older networks
- Single conductor
- Less common in LANs
- Used in cable modems

**Fiber Optic:**

**Single-Mode Fiber:**
- Single light path
- Long distances (kilometers)
- Higher bandwidth
- Used for: WAN, long-haul

**Multi-Mode Fiber:**
- Multiple light paths
- Shorter distances (hundreds of meters)
- Lower cost
- Used for: LAN, data centers

**Fiber Advantages:**
- High bandwidth (10+ Gbps)
- Long distances
- Immune to EMI
- Secure (hard to tap)

**Wireless Media:**

**WiFi Standards (802.11):**
- **802.11a**: 5 GHz, 54 Mbps
- **802.11b**: 2.4 GHz, 11 Mbps
- **802.11g**: 2.4 GHz, 54 Mbps
- **802.11n**: 2.4/5 GHz, 600 Mbps
- **802.11ac**: 5 GHz, 1.3+ Gbps
- **802.11ax (WiFi 6)**: 2.4/5 GHz, 10+ Gbps

**Wireless Frequencies:**
- **2.4 GHz**: Longer range, more interference
- **5 GHz**: Shorter range, less interference
- **6 GHz**: WiFi 6E, newest band

**Cable Selection:**
- **Distance**: Fiber for long, copper for short
- **Speed**: Higher category for more speed
- **Environment**: Shielded for EMI
- **Cost**: Copper cheaper, fiber more expensive`,
					CodeExamples: `# View cable/interface information

# View interface details
ethtool eth0
# Shows speed, duplex, cable info

# View link status
ip link show eth0
# Shows if cable is connected

# View interface speed
ethtool eth0 | grep Speed
# Shows negotiated speed

# View cable type (if supported)
ethtool -m eth0
# Shows cable diagnostics (SFP modules)

# Test cable connectivity
ping 192.168.1.1
# Tests if cable works

# View wireless standards
iwconfig
# Shows WiFi standard (802.11a/b/g/n/ac/ax)

# View wireless frequency
iwconfig wlan0 | grep Frequency
# Shows 2.4 or 5 GHz

# View wireless link quality
iwconfig wlan0
# Shows signal strength

# View interface statistics
ip -s link show eth0
# Shows errors, drops (cable issues)

# Check for cable errors
ethtool -S eth0 | grep -i error
# Shows error statistics

# View interface capabilities
ethtool eth0 | grep -A 10 "Supported"
# Shows supported speeds, duplex`,
				},
				{
					Title: "Network Design Principles",
					Content: `Good network design follows principles for scalability, reliability, and performance.

**Hierarchical Design:**

**Three-Layer Model:**
- **Core Layer**: High-speed backbone, redundancy
- **Distribution Layer**: Policy enforcement, routing
- **Access Layer**: User connectivity, switching

**Benefits:**
- Scalable
- Easier to manage
- Clear separation of concerns
- Predictable performance

**Redundancy:**

**Link Redundancy:**
- Multiple paths between devices
- Prevents single point of failure
- Load balancing
- Failover capability

**Device Redundancy:**
- Multiple routers/switches
- High availability
- Failover mechanisms
- Prevents downtime

**Redundancy Protocols:**
- Spanning Tree Protocol (STP)
- First Hop Redundancy Protocol (FHRP)
- Link Aggregation (LACP)

**Scalability:**

**Modular Design:**
- Add modules as needed
- Don't redesign entire network
- Plan for growth
- Standardized components

**Address Planning:**
- Use VLSM for efficiency
- Reserve space for growth
- Document allocations
- Plan subnet sizes

**Performance:**

**Bandwidth Planning:**
- Estimate traffic needs
- Plan for peak usage
- Monitor and adjust
- Upgrade paths

**QoS (Quality of Service):**
- Prioritize important traffic
- Voice/video priority
- Bandwidth allocation
- Traffic shaping

**Security:**

**Defense in Depth:**
- Multiple security layers
- Firewalls at boundaries
- Network segmentation
- Access control

**Network Segmentation:**
- Separate networks by function
- VLANs for logical separation
- DMZ for public services
- Isolate sensitive systems

**Design Best Practices:**
- Start with requirements
- Plan for growth
- Document everything
- Test before deployment
- Monitor and optimize`,
					CodeExamples: `# Network design verification

# View network topology
ip route show
# Shows network structure

# View VLANs (segmentation)
ip link show type vlan
# Shows VLAN interfaces

# View link aggregation (redundancy)
ip link show type bond
# Shows bonded interfaces

# View routing redundancy
ip route show
# Shows multiple paths (if configured)

# Test redundancy
ping -c 10 192.168.1.1
# Tests connectivity reliability

# View network performance
iftop
# Shows bandwidth usage

# View interface statistics
ip -s link show
# Shows performance metrics

# View routing table size
ip route show | wc -l
# Shows number of routes (scalability)

# View network segments
ip addr show
# Shows different network segments

# Test network design
traceroute 8.8.8.8
# Shows path through network

# View QoS (if configured)
tc qdisc show
# Shows traffic control (QoS)

# Monitor network health
# Use monitoring tools: Nagios, Zabbix, Prometheus`,
				},
				{
					Title: "Switches Deep Dive",
					Content: `Switches are the foundation of modern LANs.

**Switch Functions:**
- **Learning**: Builds MAC address table
- **Forwarding**: Sends frames to correct port
- **Filtering**: Doesn't forward unnecessary traffic
- **Loop Prevention**: STP (Spanning Tree Protocol)

**MAC Address Table:**
- Maps MAC addresses to ports
- Learned by observing traffic
- Ages out unused entries
- Used for forwarding decisions

**Switch Types:**

**Unmanaged Switch:**
- Plug and play
- No configuration
- Basic switching
- Home/small office

**Managed Switch:**
- Configurable
- VLAN support
- Port management
- Enterprise networks

**Switch vs Hub:**
- Hub: Broadcasts to all ports
- Switch: Forwards to specific port
- Switch: Better performance
- Switch: Separate collision domains

**VLAN Support:**
- Virtual LANs
- Logical segmentation
- Managed switches support
- Isolates traffic`,
					CodeExamples: `# View switch information

# View MAC address table (Linux bridge)
bridge fdb show
# Shows MAC addresses learned

# View bridge/switch status
bridge link show
# Shows switch ports

# View switch statistics
ethtool -S eth0
# Shows interface statistics

# Monitor switch traffic
sudo tcpdump -i eth0
# Captures traffic on switch port

# View switch configuration (if managed)
# Usually via web interface or CLI

# Test switch connectivity
ping devices-on-same-switch
# Tests switch forwarding

# View network segments
ip route show
# Shows how switch segments network`,
				},
				{
					Title: "Routers Deep Dive",
					Content: `Routers connect different networks and route traffic.

**Router Functions:**
- **Routing**: Determines best path
- **Forwarding**: Sends packets to next hop
- **NAT**: Network Address Translation
- **Firewall**: Basic packet filtering
- **DHCP**: Often provides DHCP service

**Routing Table:**
- Lists networks and next hops
- Used to make routing decisions
- Can be static or dynamic
- Longest prefix match

**Router Types:**

**Home Router:**
- Combines router, switch, AP, firewall
- NAT for internet sharing
- WiFi access point
- Simple configuration

**Enterprise Router:**
- High performance
- Advanced routing protocols
- Multiple interfaces
- Complex configuration

**Routing Methods:**

**Static Routing:**
- Manually configured routes
- Simple, predictable
- Doesn't adapt to changes

**Dynamic Routing:**
- Routes learned automatically
- Adapts to network changes
- Protocols: OSPF, BGP, RIP`,
					CodeExamples: `# View router information

# View routing table
ip route show
# Shows all routes

# View default gateway (router)
ip route show default
route -n
# Shows default route

# Add static route
sudo ip route add 192.168.2.0/24 via 192.168.1.1
# Routes 192.168.2.0/24 through router

# Delete route
sudo ip route del 192.168.2.0/24

# Test router connectivity
ping 192.168.1.1
# Tests if router is reachable

# View router through traceroute
traceroute google.com
# Shows routers along path

# View routing protocol (if dynamic)
# Usually router-specific commands

# Test NAT (if router does NAT)
# Internal IP vs external IP
curl ifconfig.me
# Shows public IP (after NAT)`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
