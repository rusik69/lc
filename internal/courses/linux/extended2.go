package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1824,
			Title:       "Linux Networking",
			Description: "Master Linux networking: interfaces, routing, iptables/nftables, DNS resolution, network namespaces, and advanced networking tools.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Network Interfaces and Routing",
					Content: `Linux networking begins with interfaces and routing tables. Understanding these fundamentals is essential for system administration and troubleshooting.

**Network Interfaces:**
` + "```" + `
Listing interfaces:
  ip link show                    # All interfaces with state
  ip addr show                    # Interfaces with IP addresses
  ip -br addr show                # Brief format (clean output)
  ip -4 addr show                 # IPv4 only
  ip -6 addr show                 # IPv6 only

Interface types:
  lo        Loopback (127.0.0.1)
  eth0      Physical Ethernet
  ens33     Predictable naming (en=ethernet, s=slot)
  enp0s3    Predictable (en=ethernet, p=PCI bus, s=slot)
  wlan0     Wireless
  wlp2s0    Predictable wireless naming
  bond0     Bonded interface (aggregation)
  br0       Bridge
  veth*     Virtual ethernet pair
  docker0   Docker bridge
  virbr0    libvirt bridge
  tun0      TUN device (VPN)
  tap0      TAP device (VPN)

Interface management:
  # Bring interface up/down
  ip link set eth0 up
  ip link set eth0 down
  
  # Add IP address
  ip addr add 192.168.1.100/24 dev eth0
  ip addr add 10.0.0.1/24 dev eth0 label eth0:1  # Alias
  
  # Remove IP address
  ip addr del 192.168.1.100/24 dev eth0
  
  # Set MTU
  ip link set eth0 mtu 9000  # Jumbo frames
  
  # Set MAC address
  ip link set eth0 down
  ip link set eth0 address 00:11:22:33:44:55
  ip link set eth0 up

Persistent configuration (systemd-networkd):
  /etc/systemd/network/10-eth0.network:
    [Match]
    Name=eth0
    
    [Network]
    Address=192.168.1.100/24
    Gateway=192.168.1.1
    DNS=8.8.8.8
    DNS=8.8.4.4
    
    [Route]
    Gateway=192.168.1.1
    Metric=100

Persistent configuration (Netplan - Ubuntu):
  /etc/netplan/01-config.yaml:
    network:
      version: 2
      renderer: networkd
      ethernets:
        eth0:
          addresses:
            - 192.168.1.100/24
          routes:
            - to: default
              via: 192.168.1.1
          nameservers:
            addresses:
              - 8.8.8.8
              - 8.8.4.4
  
  netplan apply
` + "```" + `

**Routing:**
` + "```" + `
View routing table:
  ip route show                   # All routes
  ip route show table main        # Main table
  ip route show table local       # Local table
  ip -6 route show                # IPv6 routes
  ip route get 8.8.8.8            # Which route for destination

Route types:
  default via 192.168.1.1 dev eth0           # Default gateway
  192.168.1.0/24 dev eth0 scope link         # Directly connected
  10.0.0.0/8 via 192.168.1.254 dev eth0      # Static route
  blackhole 192.168.99.0/24                  # Drop traffic
  unreachable 10.10.0.0/16                   # ICMP unreachable

Adding routes:
  # Default route
  ip route add default via 192.168.1.1 dev eth0
  
  # Specific network
  ip route add 10.0.0.0/8 via 192.168.1.254
  
  # Route with metric (lower = preferred)
  ip route add default via 192.168.1.1 metric 100
  ip route add default via 10.0.0.1 metric 200
  
  # Source-based routing (policy routing)
  ip rule add from 10.0.0.0/24 table 100
  ip route add default via 10.0.0.1 table 100

Deleting routes:
  ip route del default
  ip route del 10.0.0.0/8 via 192.168.1.254

Enable IP forwarding:
  # Temporary
  echo 1 > /proc/sys/net/ipv4/ip_forward
  sysctl -w net.ipv4.ip_forward=1
  
  # Permanent
  echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.d/99-forward.conf
  sysctl -p /etc/sysctl.d/99-forward.conf

ARP:
  ip neigh show                   # ARP table
  ip neigh add 192.168.1.50 lladdr 00:11:22:33:44:55 dev eth0
  ip neigh del 192.168.1.50 dev eth0
  ip neigh flush dev eth0         # Clear ARP cache
` + "```" + `

**Virtual Interfaces:**
` + "```" + `
VLAN:
  ip link add link eth0 name eth0.100 type vlan id 100
  ip addr add 10.100.0.1/24 dev eth0.100
  ip link set eth0.100 up

Bridge:
  ip link add br0 type bridge
  ip link set eth0 master br0
  ip link set eth1 master br0
  ip link set br0 up
  
  # Show bridge
  bridge link show
  bridge fdb show

Bond (NIC aggregation):
  ip link add bond0 type bond mode 802.3ad
  ip link set eth0 master bond0
  ip link set eth1 master bond0
  ip link set bond0 up
  
  Bond modes:
    0: balance-rr (round-robin)
    1: active-backup (failover)
    2: balance-xor (hash-based)
    4: 802.3ad (LACP - most common for servers)
    5: balance-tlb (adaptive transmit)
    6: balance-alb (adaptive load balancing)

VXLAN:
  ip link add vxlan100 type vxlan id 100 \
    dstport 4789 \
    local 192.168.1.10 \
    group 239.1.1.1 \
    dev eth0
  ip addr add 10.200.0.1/24 dev vxlan100
  ip link set vxlan100 up

Macvlan (container networking):
  ip link add macvlan0 link eth0 type macvlan mode bridge
  ip addr add 192.168.1.200/24 dev macvlan0
  ip link set macvlan0 up
` + "```" + ``,
					CodeExamples: `# Network interface configuration examples

# 1. Full systemd-networkd setup
# /etc/systemd/network/10-eth0.network
[Match]
Name=eth0

[Network]
Address=192.168.1.100/24
Gateway=192.168.1.1
DNS=8.8.8.8
DNS=1.1.1.1
Domains=example.com
DNSSEC=allow-downgrade
LLDP=yes
EmitLLDP=nearest-bridge

[Route]
Gateway=192.168.1.1
Metric=100

[Route]
Destination=10.0.0.0/8
Gateway=192.168.1.254

# 2. Netplan with bonding and VLANs
# /etc/netplan/01-bonded.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    ens3:
      dhcp4: no
    ens4:
      dhcp4: no
  bonds:
    bond0:
      interfaces: [ens3, ens4]
      parameters:
        mode: 802.3ad
        lacp-rate: fast
        mii-monitor-interval: 100
  vlans:
    bond0.100:
      id: 100
      link: bond0
      addresses: [10.100.0.1/24]
    bond0.200:
      id: 200
      link: bond0
      addresses: [10.200.0.1/24]

# 3. Policy routing script
#!/bin/bash
# Route traffic from 10.0.1.0/24 via ISP1
# Route traffic from 10.0.2.0/24 via ISP2

# Create routing tables
echo "100 isp1" >> /etc/iproute2/rt_tables
echo "200 isp2" >> /etc/iproute2/rt_tables

# ISP1 routes
ip route add default via 203.0.113.1 table isp1
ip rule add from 10.0.1.0/24 table isp1

# ISP2 routes
ip route add default via 198.51.100.1 table isp2
ip rule add from 10.0.2.0/24 table isp2

# Default route via ISP1
ip route add default via 203.0.113.1 metric 100
ip route add default via 198.51.100.1 metric 200

# 4. Network troubleshooting commands
# Trace route
traceroute -n 8.8.8.8
mtr -n 8.8.8.8

# Check connectivity  
ping -c 4 8.8.8.8
ping6 -c 4 2001:4860:4860::8888

# Check port connectivity
nc -zv example.com 443
ss -tlnp | grep :80

# DNS lookup
dig example.com
dig +short example.com A
host example.com
nslookup example.com`,
				},
				{
					Title: "Firewalling with iptables and nftables",
					Content: `Linux firewalling controls network traffic flow. iptables (legacy) and nftables (modern replacement) are the primary tools.

**iptables Fundamentals:**
` + "```" + `
Tables:
  filter   Default table for packet filtering
  nat      Network address translation
  mangle   Packet header modification
  raw      Bypass connection tracking

Chains (filter table):
  INPUT     Packets destined for the local machine
  OUTPUT    Packets originating from the local machine
  FORWARD   Packets routed through the machine

Chains (nat table):
  PREROUTING   Before routing decision (DNAT)
  POSTROUTING  After routing decision (SNAT/masquerade)
  OUTPUT       Locally generated traffic

Packet flow:
  Incoming → PREROUTING → routing → INPUT → local process
  Outgoing → local process → OUTPUT → POSTROUTING
  Forwarded → PREROUTING → routing → FORWARD → POSTROUTING

Targets:
  ACCEPT    Allow the packet
  DROP      Silently discard
  REJECT    Discard with ICMP error
  LOG       Log and continue processing
  DNAT      Destination NAT
  SNAT      Source NAT
  MASQUERADE  Dynamic SNAT (for dynamic IPs)
  RETURN    Return from chain
` + "```" + `

**iptables Rules:**
` + "```" + `
List rules:
  iptables -L -n -v                    # Verbose with counters
  iptables -L -n --line-numbers        # With rule numbers
  iptables -t nat -L -n -v             # NAT table
  iptables -S                          # Rules in command format

Basic rules:
  # Allow established connections
  iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
  
  # Allow loopback
  iptables -A INPUT -i lo -j ACCEPT
  
  # Allow SSH
  iptables -A INPUT -p tcp --dport 22 -j ACCEPT
  
  # Allow HTTP/HTTPS
  iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT
  
  # Allow ICMP (ping)
  iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT
  
  # Drop everything else
  iptables -A INPUT -j DROP
  
  # Default policy
  iptables -P INPUT DROP
  iptables -P FORWARD DROP
  iptables -P OUTPUT ACCEPT

Advanced rules:
  # Rate limiting (anti-DDoS)
  iptables -A INPUT -p tcp --dport 22 -m connlimit --connlimit-above 3 -j REJECT
  iptables -A INPUT -p tcp --dport 80 -m limit --limit 100/s --limit-burst 200 -j ACCEPT
  
  # Source IP filtering
  iptables -A INPUT -s 192.168.1.0/24 -j ACCEPT
  iptables -A INPUT -s 10.0.0.0/8 -j DROP
  
  # Logging
  iptables -A INPUT -j LOG --log-prefix "IPTables-Drop: " --log-level 4
  iptables -A INPUT -j DROP

NAT rules:
  # SNAT (outbound masquerade)
  iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
  
  # DNAT (port forwarding)
  iptables -t nat -A PREROUTING -p tcp --dport 8080 \
    -j DNAT --to-destination 192.168.1.100:80
  iptables -A FORWARD -p tcp -d 192.168.1.100 --dport 80 -j ACCEPT

Save/restore:
  iptables-save > /etc/iptables/rules.v4
  iptables-restore < /etc/iptables/rules.v4
  
  # With iptables-persistent (Debian/Ubuntu):
  netfilter-persistent save
  netfilter-persistent reload
` + "```" + `

**nftables (Modern Replacement):**
` + "```" + `
Why nftables:
  - Unified framework (replaces iptables, ip6tables, arptables, ebtables)
  - Better performance (single rule evaluation)
  - Atomic rule replacement
  - Better syntax
  - Sets and maps (native support)
  - No fixed tables/chains (user-defined)

Basic usage:
  # List ruleset
  nft list ruleset
  
  # Create table
  nft add table inet filter
  
  # Create chain
  nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
  nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
  nft add chain inet filter output { type filter hook output priority 0 \; policy accept \; }
  
  # Add rules
  nft add rule inet filter input ct state established,related accept
  nft add rule inet filter input iif lo accept
  nft add rule inet filter input tcp dport 22 accept
  nft add rule inet filter input tcp dport { 80, 443 } accept
  nft add rule inet filter input icmp type echo-request accept
  nft add rule inet filter input counter drop

Sets:
  # Named set
  nft add set inet filter allowed_ips { type ipv4_addr \; }
  nft add element inet filter allowed_ips { 192.168.1.0/24, 10.0.0.0/8 }
  nft add rule inet filter input ip saddr @allowed_ips accept
  
  # Dynamic set (auto-populated)
  nft add set inet filter blacklist { type ipv4_addr \; flags timeout \; timeout 1h \; }
  nft add rule inet filter input tcp dport 22 \
    ct state new meter ssh_meter { ip saddr limit rate 3/minute } accept
  
Maps:
  # Port redirect map
  nft add map inet filter portmap { type inet_service : inet_service \; }
  nft add element inet filter portmap { 8080 : 80, 8443 : 443 }
  nft add rule inet nat prerouting dnat to tcp dport map @portmap

nftables configuration file:
  /etc/nftables.conf:
    #!/usr/sbin/nft -f
    flush ruleset
    
    table inet filter {
      set allowed_ssh {
        type ipv4_addr
        elements = { 192.168.1.0/24, 10.0.0.0/8 }
      }
    
      chain input {
        type filter hook input priority 0; policy drop;
        ct state established,related accept
        iif lo accept
        ip saddr @allowed_ssh tcp dport 22 accept
        tcp dport { 80, 443 } accept
        icmp type echo-request accept
        counter drop
      }
      
      chain forward {
        type filter hook forward priority 0; policy drop;
      }
      
      chain output {
        type filter hook output priority 0; policy accept;
      }
    }
  
  systemctl enable nftables
  systemctl start nftables
` + "```" + ``,
					CodeExamples: `# Comprehensive firewall configuration

# 1. Production iptables ruleset
#!/bin/bash
# /etc/iptables/setup.sh

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t mangle -F

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Anti-spoofing
iptables -A INPUT -s 127.0.0.0/8 ! -i lo -j DROP
iptables -A INPUT -s 0.0.0.0/8 -j DROP
iptables -A INPUT -s 169.254.0.0/16 -j DROP
iptables -A INPUT -s 224.0.0.0/4 -j DROP
iptables -A INPUT -s 240.0.0.0/4 -j DROP

# ICMP (rate limited)
iptables -A INPUT -p icmp --icmp-type echo-request \
  -m limit --limit 5/s --limit-burst 10 -j ACCEPT

# SSH (rate limited)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW \
  -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW \
  -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Web services
iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT

# Log dropped packets (rate limited)
iptables -A INPUT -m limit --limit 5/min -j LOG \
  --log-prefix "iptables-drop: " --log-level 4
iptables -A INPUT -j DROP

# Save
iptables-save > /etc/iptables/rules.v4

# 2. Production nftables configuration
# /etc/nftables.conf
#!/usr/sbin/nft -f
flush ruleset

table inet firewall {
  # Trusted networks
  set trusted_nets {
    type ipv4_addr
    flags interval
    elements = { 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16 }
  }

  # Blocked IPs (dynamic with timeout)
  set blocklist {
    type ipv4_addr
    flags timeout
    timeout 24h
  }

  # Rate limit meter
  set ssh_meter {
    type ipv4_addr
    flags dynamic,timeout
    timeout 5m
  }

  chain input {
    type filter hook input priority 0; policy drop;
    
    # Connection tracking
    ct state established,related accept
    ct state invalid drop
    
    # Loopback
    iif lo accept
    
    # Blocklist
    ip saddr @blocklist drop
    
    # ICMP
    icmp type { echo-request, destination-unreachable, time-exceeded } \
      limit rate 10/second accept
    
    # ICMPv6
    icmpv6 type { echo-request, nd-neighbor-solicit, nd-router-advert, 
                   nd-neighbor-advert } accept
    
    # SSH with rate limiting
    tcp dport 22 ct state new \
      add @ssh_meter { ip saddr limit rate 3/minute burst 5 packets } accept
    tcp dport 22 ct state new \
      add @blocklist { ip saddr } drop
    
    # HTTP/HTTPS
    tcp dport { 80, 443 } accept
    
    # Internal services (trusted networks only)
    ip saddr @trusted_nets tcp dport { 9090, 9100, 3000 } accept
    
    # Log and drop
    limit rate 5/minute log prefix "nft-drop: " counter drop
  }
  
  chain forward {
    type filter hook forward priority 0; policy drop;
    ct state established,related accept
  }
  
  chain output {
    type filter hook output priority 0; policy accept;
  }
}

# NAT table
table ip nat {
  chain prerouting {
    type nat hook prerouting priority -100;
    tcp dport 8080 dnat to 192.168.1.100:80
  }
  
  chain postrouting {
    type nat hook postrouting priority 100;
    oifname "eth0" masquerade
  }
}`,
				},
				{
					Title: "DNS and Network Troubleshooting",
					Content: `DNS resolution and network troubleshooting are critical skills for Linux administrators. Understanding how name resolution works and having a systematic troubleshooting approach saves hours.

**DNS Resolution in Linux:**
` + "```" + `
Resolution order (nsswitch.conf):
  /etc/nsswitch.conf:
    hosts: files dns myhostname
    
  1. files → /etc/hosts (local overrides)
  2. dns → resolv.conf (DNS servers)
  3. myhostname → hostname fallback

/etc/hosts:
  127.0.0.1   localhost
  ::1         localhost
  192.168.1.100  myserver.example.com myserver
  # Checked before DNS queries

/etc/resolv.conf:
  nameserver 8.8.8.8        # Primary DNS
  nameserver 8.8.4.4        # Secondary DNS
  search example.com        # Search domain
  options timeout:2 attempts:3 rotate
  
  # systemd-resolved manages this on modern systems
  # Real config: /etc/systemd/resolved.conf
  # resolv.conf is a symlink to stub resolver (127.0.0.53)

systemd-resolved:
  /etc/systemd/resolved.conf:
    [Resolve]
    DNS=8.8.8.8 8.8.4.4
    FallbackDNS=1.1.1.1 1.0.0.1
    Domains=example.com
    DNSSEC=allow-downgrade
    DNSOverTLS=opportunistic
    Cache=yes
    DNSStubListener=yes
  
  resolvectl status           # Show resolution config
  resolvectl query example.com  # Query with details
  resolvectl statistics       # Cache stats
  resolvectl flush-caches     # Flush DNS cache
` + "```" + `

**DNS Tools:**
` + "```" + `
dig (most powerful):
  dig example.com               # A record
  dig example.com MX            # MX records
  dig example.com ANY           # All records
  dig +short example.com        # Short output
  dig @8.8.8.8 example.com     # Query specific server
  dig +trace example.com        # Full delegation trace
  dig -x 8.8.8.8               # Reverse lookup (PTR)
  dig +noall +answer example.com  # Clean output
  
  # Output sections:
  # QUESTION: what was asked
  # ANSWER: the resolved records
  # AUTHORITY: nameservers for the zone
  # ADDITIONAL: extra info (glue records)

host (simple):
  host example.com
  host -t MX example.com
  host 8.8.8.8                 # Reverse

nslookup (interactive):
  nslookup example.com
  nslookup -type=mx example.com
  nslookup example.com 8.8.8.8

Common DNS record types:
  A       IPv4 address
  AAAA    IPv6 address
  CNAME   Alias to another name
  MX      Mail server
  NS      Nameserver
  PTR     Reverse lookup (IP → name)
  SOA     Start of authority
  SRV     Service locator
  TXT     Text (SPF, DKIM, verification)
  CAA     Certificate authority authorization
` + "```" + `

**Network Troubleshooting Methodology:**
` + "```" + `
Layer-by-layer approach (bottom up):

Layer 1 - Physical:
  ip link show                   # Interface up?
  ethtool eth0                   # Link speed/duplex/status
  ethtool -S eth0                # NIC statistics (errors, drops)
  dmesg | grep -i eth            # Driver messages

Layer 2 - Data Link:
  ip neigh show                  # ARP table
  arping -I eth0 192.168.1.1    # ARP ping
  bridge fdb show                # Bridge forwarding table
  tcpdump -i eth0 arp           # Watch ARP traffic

Layer 3 - Network:
  ip addr show                   # IP configured?
  ip route show                  # Routes correct?
  ip route get 10.0.0.1         # Which route for destination?
  ping -c 4 gateway_ip          # Gateway reachable?
  ping -c 4 8.8.8.8             # Internet reachable?
  traceroute -n 8.8.8.8         # Path to destination
  mtr -n 8.8.8.8                # Real-time traceroute

Layer 4 - Transport:
  ss -tlnp                      # Listening TCP ports
  ss -ulnp                      # Listening UDP ports
  ss -s                         # Socket statistics summary
  nc -zv host 443               # Test TCP connection
  nc -uzv host 53               # Test UDP connection
  nmap -sT host                 # Port scan

Layer 7 - Application:
  curl -v https://example.com   # HTTP verbose
  curl -I https://example.com   # Headers only
  openssl s_client -connect host:443  # TLS debug
  wget --spider https://example.com   # Check URL

Connection debugging:
  ss -tnp                       # Active TCP connections
  ss -tnp state established     # Only established
  ss -tnp state time-wait       # TIME_WAIT connections
  ss -tnp '( dport = :443 )'   # Filter by port
  
  # Connection states:
  # ESTABLISHED: active connection
  # TIME_WAIT: closed, waiting for stale packets
  # CLOSE_WAIT: remote closed, local hasn't
  # SYN_SENT: connection attempt
  # LISTEN: waiting for connections
` + "```" + `

**Packet Capture:**
` + "```" + `
tcpdump:
  # Capture on interface
  tcpdump -i eth0
  
  # Common filters
  tcpdump -i eth0 port 80
  tcpdump -i eth0 host 192.168.1.100
  tcpdump -i eth0 src 10.0.0.1
  tcpdump -i eth0 dst port 443
  tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn) != 0'  # SYN packets
  tcpdump -i eth0 icmp
  
  # Useful flags
  tcpdump -i eth0 -n              # Don't resolve names
  tcpdump -i eth0 -nn             # Don't resolve names or ports
  tcpdump -i eth0 -v              # Verbose
  tcpdump -i eth0 -X              # Hex and ASCII
  tcpdump -i eth0 -A              # ASCII only
  tcpdump -i eth0 -c 100          # Capture 100 packets
  tcpdump -i eth0 -w capture.pcap # Write to file
  tcpdump -r capture.pcap         # Read from file
  
  # Complex filters
  tcpdump -i eth0 'host 10.0.0.1 and (port 80 or port 443)'
  tcpdump -i eth0 'net 192.168.1.0/24 and not port 22'

Performance tools:
  iperf3 -s                      # Server mode
  iperf3 -c server_ip -t 30      # Client: 30sec TCP test
  iperf3 -c server_ip -u -b 1G   # UDP bandwidth test
  
  # Check bandwidth
  speedtest-cli
  
  # Network latency
  hping3 -S -p 80 example.com    # TCP SYN ping
  hping3 --traceroute -V -1 host # ICMP traceroute
` + "```" + ``,
					CodeExamples: `# Network troubleshooting scripts

# 1. Comprehensive network diagnostic script
#!/bin/bash
echo "=== Network Diagnostic Report ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

echo "=== Interfaces ==="
ip -br addr show
echo ""

echo "=== Routes ==="
ip route show
echo ""

echo "=== DNS Configuration ==="
cat /etc/resolv.conf
echo ""

echo "=== DNS Resolution Test ==="
for domain in google.com example.com; do
    echo -n "$domain: "
    dig +short "$domain" A | head -1
done
echo ""

echo "=== Gateway Connectivity ==="
GW=$(ip route | awk '/default/ {print $3}')
ping -c 2 -W 2 "$GW" 2>/dev/null && echo "Gateway OK" || echo "Gateway FAIL"
echo ""

echo "=== Internet Connectivity ==="
ping -c 2 -W 2 8.8.8.8 2>/dev/null && echo "Internet OK" || echo "Internet FAIL"
echo ""

echo "=== Listening Ports ==="
ss -tlnp 2>/dev/null | head -20
echo ""

echo "=== Active Connections ==="
ss -tnp state established 2>/dev/null | head -20

# 2. Port connectivity tester
#!/bin/bash
# Usage: ./portcheck.sh hostname port1 port2 ...
HOST=$1
shift
for PORT in "$@"; do
    if nc -z -w2 "$HOST" "$PORT" 2>/dev/null; then
        echo "$HOST:$PORT - OPEN"
    else
        echo "$HOST:$PORT - CLOSED/FILTERED"
    fi
done

# 3. Monitor network connections
#!/bin/bash
# Watch connection states in real-time
watch -n 1 'echo "=== Connection States ===" && \
  ss -s && echo "" && \
  echo "=== Top Connections by State ===" && \
  ss -tn | awk "{print \$1}" | sort | uniq -c | sort -rn && \
  echo "" && \
  echo "=== Top Remote IPs ===" && \
  ss -tn state established | awk "{print \$5}" | \
  cut -d: -f1 | sort | uniq -c | sort -rn | head -10'

# 4. DNS zone check
#!/bin/bash
DOMAIN=$1
echo "=== DNS Records for $DOMAIN ==="
for TYPE in A AAAA MX NS TXT SOA CAA; do
    RESULT=$(dig +short "$DOMAIN" "$TYPE" 2>/dev/null)
    if [ -n "$RESULT" ]; then
        echo "--- $TYPE ---"
        echo "$RESULT"
    fi
done`,
				},
			},
		},
	})
}
