package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1838,
			Title:       "Linux DNS, DHCP, and Network Services",
			Description: "Configure and manage essential network services including DNS servers, DHCP, NTP, and network file sharing on Linux.",
			Order:       38,
			Lessons: []problems.Lesson{
				{
					Title: "DNS Server Configuration",
					Content: `DNS is one of the most critical network services. Understanding how to configure and troubleshoot DNS servers is essential for Linux administration.

**BIND (Berkeley Internet Name Domain):**
` + "```" + `
BIND is the most widely used DNS server software.

Installation:
  apt install bind9 bind9-utils     # Debian/Ubuntu
  dnf install bind bind-utils       # RHEL/Fedora

Main configuration files:
  /etc/bind/named.conf              # Main config
  /etc/bind/named.conf.options      # Server options
  /etc/bind/named.conf.local        # Zone definitions
  /var/lib/bind/                    # Zone data files
  /var/cache/bind/                  # Cache directory

Basic options (/etc/bind/named.conf.options):
  options {
      directory "/var/cache/bind";
      
      // Listen on
      listen-on { 127.0.0.1; 10.0.0.1; };
      listen-on-v6 { ::1; };
      
      // Allow queries from
      allow-query { localhost; 10.0.0.0/24; };
      
      // Forwarding
      forwarders {
          8.8.8.8;
          8.8.4.4;
      };
      forward only;  // or "first"
      
      // Security
      recursion yes;                      // Allow recursive queries
      allow-recursion { 10.0.0.0/24; };  // Only from internal
      
      dnssec-validation auto;
      
      // Rate limiting
      rate-limit {
          responses-per-second 10;
          window 5;
      };
  };

Forward zone (/etc/bind/named.conf.local):
  zone "example.com" {
      type master;
      file "/var/lib/bind/db.example.com";
      allow-transfer { 10.0.0.2; };     // Slave DNS
      also-notify { 10.0.0.2; };
  };

Zone file (/var/lib/bind/db.example.com):
  $TTL    86400
  @       IN      SOA     ns1.example.com. admin.example.com. (
                          2024010101  ; Serial (YYYYMMDDNN)
                          3600        ; Refresh (1 hour)
                          900         ; Retry (15 minutes)
                          604800      ; Expire (1 week)
                          86400       ; Minimum TTL (1 day)
  )
  
  ; Name servers
  @       IN      NS      ns1.example.com.
  @       IN      NS      ns2.example.com.
  
  ; A records
  ns1     IN      A       10.0.0.1
  ns2     IN      A       10.0.0.2
  @       IN      A       10.0.0.10
  www     IN      A       10.0.0.10
  mail    IN      A       10.0.0.20
  db      IN      A       10.0.0.30
  
  ; CNAME records
  ftp     IN      CNAME   www.example.com.
  cdn     IN      CNAME   www.example.com.
  
  ; MX records
  @       IN      MX      10 mail.example.com.
  @       IN      MX      20 mail2.example.com.
  
  ; TXT records
  @       IN      TXT     "v=spf1 mx -all"
  
  ; SRV records
  _http._tcp  IN  SRV     0 5 80 www.example.com.

Reverse zone:
  zone "0.0.10.in-addr.arpa" {
      type master;
      file "/var/lib/bind/db.10.0.0";
  };
  
  ; Zone file (/var/lib/bind/db.10.0.0)
  $TTL    86400
  @       IN      SOA     ns1.example.com. admin.example.com. (
                          2024010101 3600 900 604800 86400 )
  @       IN      NS      ns1.example.com.
  @       IN      NS      ns2.example.com.
  1       IN      PTR     ns1.example.com.
  2       IN      PTR     ns2.example.com.
  10      IN      PTR     www.example.com.
  20      IN      PTR     mail.example.com.
` + "```" + `

**Unbound (Recursive DNS):**
` + "```" + `
Unbound is a validating, recursive, caching DNS resolver.

Installation:
  apt install unbound      # Debian/Ubuntu
  dnf install unbound      # RHEL/Fedora

Configuration (/etc/unbound/unbound.conf):
  server:
      interface: 0.0.0.0
      port: 53
      
      access-control: 10.0.0.0/24 allow
      access-control: 127.0.0.0/8 allow
      
      # Performance
      num-threads: 4
      msg-cache-slabs: 8
      rrset-cache-slabs: 8
      infra-cache-slabs: 8
      key-cache-slabs: 8
      msg-cache-size: 256m
      rrset-cache-size: 512m
      
      # Security
      hide-identity: yes
      hide-version: yes
      harden-glue: yes
      harden-dnssec-stripped: yes
      
      # DNSSEC
      auto-trust-anchor-file: "/var/lib/unbound/root.key"
      
      # Local records
      local-zone: "example.com." static
      local-data: "server1.example.com. A 10.0.0.10"
      local-data: "server2.example.com. A 10.0.0.20"
      
      # Block ads/malware (Pi-hole style)
      local-zone: "ads.example.com" refuse
      include: /etc/unbound/blocklist.conf
  
  forward-zone:
      name: "."
      forward-addr: 8.8.8.8
      forward-addr: 8.8.4.4

Management:
  unbound-control status
  unbound-control stats_noreset
  unbound-control dump_cache > cache.txt
  unbound-control load_cache < cache.txt
  unbound-control flush example.com
  unbound-control flush_zone example.com
` + "```" + `

**DNS Troubleshooting:**
` + "```" + `
dig (DNS lookup utility):
  dig example.com                     # A record
  dig example.com AAAA                # IPv6
  dig example.com MX                  # Mail records
  dig example.com NS                  # Name servers
  dig example.com ANY                 # All records
  dig @8.8.8.8 example.com           # Query specific server
  dig +trace example.com             # Full resolution trace
  dig +short example.com             # Brief output
  dig -x 10.0.0.1                    # Reverse lookup
  
  # Check SOA serial
  dig @ns1.example.com example.com SOA +short
  dig @ns2.example.com example.com SOA +short
  # Serials should match (zone transfer working)

nslookup:
  nslookup example.com
  nslookup -type=MX example.com
  nslookup example.com 8.8.8.8

host:
  host example.com
  host -t MX example.com
  host 10.0.0.1

named-checkconf:
  named-checkconf                              # Check config syntax
  named-checkzone example.com db.example.com   # Check zone file

Common issues:
  1. SERVFAIL: DNSSEC validation failure or upstream issue
     - Check: dig +dnssec example.com
     - Fix: Verify DNSSEC chain or disable validation
  
  2. REFUSED: Access control blocking query
     - Check: allow-query / access-control settings
  
  3. Zone transfer failure: Slave not updating
     - Check serial numbers match after update
     - Check allow-transfer ACL
     - Check connectivity on port 53 TCP
` + "```" + ``,
					CodeExamples: `# DNS and network service configurations

# 1. DNS health check script
#!/bin/bash
DNS_SERVERS=("10.0.0.1" "10.0.0.2" "8.8.8.8")
TEST_DOMAINS=("example.com" "google.com" "github.com")

echo "=== DNS Health Check ==="

for server in "${DNS_SERVERS[@]}"; do
    echo ""
    echo "--- Server: $server ---"
    
    # Response time
    for domain in "${TEST_DOMAINS[@]}"; do
        result=$(dig @"$server" "$domain" +noall +stats +answer 2>/dev/null)
        
        if echo "$result" | grep -q "ANSWER SECTION"; then
            time_ms=$(echo "$result" | grep "Query time" | awk '{print $4}')
            ip=$(echo "$result" | grep -A1 "ANSWER" | tail -1 | awk '{print $NF}')
            printf "  %-20s  %4s ms  %s\n" "$domain" "$time_ms" "$ip"
        else
            printf "  %-20s  FAILED\n" "$domain"
        fi
    done
    
    # Check DNSSEC
    dnssec_result=$(dig @"$server" com. DNSKEY +dnssec +short 2>/dev/null)
    if [ -n "$dnssec_result" ]; then
        echo "  DNSSEC: supported"
    else
        echo "  DNSSEC: not available"
    fi
done

# 2. Zone file generator
#!/bin/bash
# Generate DNS zone file from a simple hosts list
DOMAIN="${1:?Usage: $0 <domain>}"
HOSTS_FILE="${2:?Usage: $0 <domain> <hosts-file>}"
# hosts-file format: hostname ip

SERIAL=$(date +%Y%m%d01)

cat << EOF
\$TTL    86400
@       IN      SOA     ns1.${DOMAIN}. admin.${DOMAIN}. (
                        ${SERIAL}   ; Serial
                        3600        ; Refresh
                        900         ; Retry
                        604800      ; Expire
                        86400       ; Minimum TTL
)

; Name servers
@       IN      NS      ns1.${DOMAIN}.
@       IN      NS      ns2.${DOMAIN}.

; Host records
EOF

while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    
    hostname=$(echo "$line" | awk '{print $1}')
    ip=$(echo "$line" | awk '{print $2}')
    record_type="A"
    
    # Check if IPv6
    if [[ "$ip" == *":"* ]]; then
        record_type="AAAA"
    fi
    
    printf "%-16s IN      %-6s  %s\n" "$hostname" "$record_type" "$ip"
done < "$HOSTS_FILE"

# 3. DHCP server configuration (ISC DHCP)
# /etc/dhcp/dhcpd.conf
# Global options
# option domain-name "example.com";
# option domain-name-servers 10.0.0.1, 10.0.0.2;
# default-lease-time 3600;
# max-lease-time 86400;
# authoritative;
# log-facility local7;
#
# # Subnet definition
# subnet 10.0.0.0 netmask 255.255.255.0 {
#     range 10.0.0.100 10.0.0.200;
#     option routers 10.0.0.1;
#     option subnet-mask 255.255.255.0;
#     option broadcast-address 10.0.0.255;
#     option ntp-servers 10.0.0.1;
# }
#
# # Static assignments
# host webserver {
#     hardware ethernet 00:11:22:33:44:55;
#     fixed-address 10.0.0.10;
#     option host-name "web1";
# }`,
				},
				{
					Title: "NFS, Samba, and Network File Systems",
					Content: `Network file sharing allows multiple systems to access shared storage. NFS is standard for Linux-to-Linux, Samba for Windows interoperability.

**NFS (Network File System):**
` + "```" + `
NFS v4 is the current standard for Linux file sharing.

Server setup:
  # Install
  apt install nfs-kernel-server     # Debian/Ubuntu
  dnf install nfs-utils             # RHEL/Fedora
  
  # Create export directory
  mkdir -p /srv/nfs/shared
  chown nobody:nogroup /srv/nfs/shared
  chmod 755 /srv/nfs/shared
  
  # Configure exports (/etc/exports)
  /srv/nfs/shared    10.0.0.0/24(rw,sync,no_subtree_check,no_root_squash)
  /srv/nfs/readonly  10.0.0.0/24(ro,sync,no_subtree_check)
  /srv/nfs/home      10.0.0.0/24(rw,sync,no_subtree_check,root_squash)
  
  Export options:
    rw / ro               Read-write / read-only
    sync / async          Write to disk before reply / buffer writes
    no_subtree_check      Don't verify file is in exported tree
    root_squash           Map root to nobody (default, secure)
    no_root_squash        Allow root access (use carefully)
    all_squash            Map all users to nobody
    anonuid=1000          Anonymous user UID
    anongid=1000          Anonymous group GID
  
  # Apply changes
  exportfs -ra              # Re-export all
  exportfs -v               # Show current exports
  
  # Start/enable service
  systemctl enable --now nfs-server

Client setup:
  # Install
  apt install nfs-common         # Debian/Ubuntu
  dnf install nfs-utils          # RHEL/Fedora
  
  # Show server exports
  showmount -e server.example.com
  
  # Mount
  mount -t nfs4 server:/srv/nfs/shared /mnt/shared
  
  # /etc/fstab entry
  server:/srv/nfs/shared  /mnt/shared  nfs4  defaults,_netdev  0  0
  
  # AutoFS (mount on demand)
  apt install autofs
  
  # /etc/auto.master
  /mnt/nfs  /etc/auto.nfs  --timeout=300
  
  # /etc/auto.nfs
  shared  -rw,soft,intr  server:/srv/nfs/shared
  home    -rw,soft,intr  server:/srv/nfs/home
  
  systemctl enable --now autofs

NFS performance tuning:
  # Server: increase threads
  # /etc/default/nfs-kernel-server
  RPCNFSDCOUNT=16
  
  # Client: mount options
  mount -t nfs4 -o rsize=1048576,wsize=1048576,hard,timeo=600 \
    server:/export /mnt/nfs
  
  Options:
    rsize/wsize    Read/write block size (max 1MB for NFSv4)
    hard/soft      Hard: retry forever / Soft: timeout and error
    timeo          Timeout in deciseconds (for soft mounts)
    retrans        Number of retries
    nconnect=8     Multiple TCP connections (kernel 5.3+)

NFS troubleshooting:
  # Check NFS statistics
  nfsstat -s        # Server stats
  nfsstat -c        # Client stats
  nfsiostat          # I/O statistics
  
  # Check RPC services
  rpcinfo -p server
  
  # Monitor NFS traffic
  rpcdebug -m nfsd -s all       # Enable server debug
  rpcdebug -m nfs -s all        # Enable client debug
  dmesg | grep -i nfs           # Check kernel messages
` + "```" + `

**Samba (SMB/CIFS):**
` + "```" + `
Samba provides file/print sharing compatible with Windows.

Server setup:
  # Install
  apt install samba              # Debian/Ubuntu
  dnf install samba              # RHEL/Fedora
  
  # Configuration (/etc/samba/smb.conf)
  [global]
      workgroup = WORKGROUP
      server string = File Server
      security = user
      map to guest = Bad User
      
      # Performance
      socket options = TCP_NODELAY SO_RCVBUF=131072 SO_SNDBUF=131072
      read raw = yes
      write raw = yes
      max xmit = 65535
      dead time = 15
      
      # Logging
      log file = /var/log/samba/log.%m
      max log size = 1000
      log level = 1
  
  [shared]
      comment = Shared Files
      path = /srv/samba/shared
      browseable = yes
      read only = no
      valid users = @smbgroup
      create mask = 0664
      directory mask = 0775
      force group = smbgroup
  
  [public]
      comment = Public Files
      path = /srv/samba/public
      browseable = yes
      read only = yes
      guest ok = yes
  
  [homes]
      comment = Home Directories
      browseable = no
      read only = no
      valid users = %S

  # Create Samba user
  useradd -M -s /sbin/nologin smbuser
  smbpasswd -a smbuser
  
  # Create group and directory
  groupadd smbgroup
  usermod -aG smbgroup smbuser
  mkdir -p /srv/samba/shared
  chgrp smbgroup /srv/samba/shared
  chmod 2775 /srv/samba/shared
  
  # Test config
  testparm
  
  # Start
  systemctl enable --now smbd nmbd

Client (Linux):
  # Install
  apt install cifs-utils smbclient
  
  # List shares
  smbclient -L //server -U user
  
  # Interactive access
  smbclient //server/shared -U user
  
  # Mount
  mount -t cifs //server/shared /mnt/samba -o username=user,password=pass
  
  # /etc/fstab (use credentials file for security)
  //server/shared  /mnt/samba  cifs  credentials=/root/.smbcreds,uid=1000,gid=1000  0  0
  
  # /root/.smbcreds (chmod 600)
  username=smbuser
  password=secretpass
  domain=WORKGROUP
` + "```" + `

**iSCSI (Block Storage over Network):**
` + "```" + `
iSCSI provides block-level storage access over TCP/IP.

Target (server) setup:
  # Install targetcli
  apt install targetcli-fb         # Debian/Ubuntu
  dnf install targetcli            # RHEL/Fedora
  
  # Configure with targetcli
  targetcli
  
  # Create backing store
  /backstores/block create disk0 /dev/sdb
  # Or file-based:
  /backstores/fileio create disk1 /srv/iscsi/disk1.img 10G
  
  # Create target
  /iscsi create iqn.2024.com.example:storage
  
  # Create LUN
  /iscsi/iqn.2024.com.example:storage/tpg1/luns create /backstores/block/disk0
  
  # Set ACL (initiator IQN)
  /iscsi/iqn.2024.com.example:storage/tpg1/acls create iqn.2024.com.example:client1
  
  # Save and exit
  saveconfig
  exit

Initiator (client) setup:
  # Install
  apt install open-iscsi           # Debian/Ubuntu
  dnf install iscsi-initiator-utils # RHEL/Fedora
  
  # Set initiator name
  echo "InitiatorName=iqn.2024.com.example:client1" > /etc/iscsi/initiatorname.iscsi
  
  # Discover targets
  iscsiadm -m discovery -t sendtargets -p target-server:3260
  
  # Login
  iscsiadm -m node --targetname iqn.2024.com.example:storage -p target-server:3260 --login
  
  # Auto-login on boot
  iscsiadm -m node --targetname iqn.2024.com.example:storage -p target-server:3260 -o update -n node.startup -v automatic
  
  # New block device appears (e.g., /dev/sdc)
  lsblk
  
  # Format and mount
  mkfs.ext4 /dev/sdc
  mount /dev/sdc /mnt/iscsi
` + "```" + ``,
					CodeExamples: `# Network services management

# 1. NFS server monitoring script
#!/bin/bash
echo "=== NFS Server Status ==="

# Active exports
echo "--- Exports ---"
exportfs -v 2>/dev/null | while read -r line; do
    echo "  $line"
done

# Connected clients
echo ""
echo "--- Connected Clients ---"
if command -v ss > /dev/null 2>&1; then
    ss -tn state established '( dport = :2049 )' | \
    awk 'NR>1 {print $4}' | sort -u | while read -r client; do
        echo "  $client"
    done
fi

# NFS statistics
echo ""
echo "--- NFS Statistics ---"
if [ -f /proc/net/rpc/nfsd ]; then
    # Total operations
    OPS=$(awk '/proc4ops/ {sum=0; for(i=2;i<=NF;i++) sum+=$i; print sum}' \
        /proc/net/rpc/nfsd 2>/dev/null)
    echo "  Total NFSv4 operations: ${OPS:-0}"
    
    # Thread utilization
    THREADS=$(grep "th " /proc/net/rpc/nfsd 2>/dev/null | awk '{print $2}')
    echo "  NFS threads: ${THREADS:-unknown}"
fi

# Mount points using NFS space
echo ""
echo "--- NFS Space Usage ---"
df -h --type=nfs4 2>/dev/null || echo "  No NFS mounts (server side)"

# 2. Samba user management script
#!/bin/bash
case "${1:-help}" in
    add)
        USERNAME="${2:?Usage: $0 add <username>}"
        # Create system user if not exists
        if ! id "$USERNAME" > /dev/null 2>&1; then
            useradd -M -s /sbin/nologin "$USERNAME"
            echo "System user created: $USERNAME"
        fi
        # Set Samba password
        smbpasswd -a "$USERNAME"
        smbpasswd -e "$USERNAME"
        echo "Samba user added: $USERNAME"
        ;;
    disable)
        USERNAME="${2:?Usage: $0 disable <username>}"
        smbpasswd -d "$USERNAME"
        echo "Samba user disabled: $USERNAME"
        ;;
    enable)
        USERNAME="${2:?Usage: $0 enable <username>}"
        smbpasswd -e "$USERNAME"
        echo "Samba user enabled: $USERNAME"
        ;;
    list)
        pdbedit -L -v 2>/dev/null | grep -E "^Unix username:|^Account Flags:" | \
        paste - - | awk -F'[:\t]+' '{
            user=$2; flags=$4;
            gsub(/^ +| +$/, "", user);
            gsub(/^ +| +$/, "", flags);
            printf "  %-20s %s\n", user, flags
        }'
        ;;
    status)
        echo "=== Samba Status ==="
        smbstatus --brief 2>/dev/null
        echo ""
        echo "--- Open Files ---"
        smbstatus --shares 2>/dev/null
        ;;
    *)
        echo "Usage: $0 {add|disable|enable|list|status} [username]"
        ;;
esac

# 3. Network file share backup script
#!/bin/bash
# Backup all NFS and CIFS mounts
BACKUP_DIR="/backup/network-shares"
DATE=$(date +%Y-%m-%d)

echo "=== Network Share Backup ==="

# Find all network mounts
mount | grep -E "type (nfs|cifs)" | while read -r line; do
    MOUNTPOINT=$(echo "$line" | awk '{print $3}')
    FSTYPE=$(echo "$line" | awk '{print $5}')
    SOURCE=$(echo "$line" | awk '{print $1}')
    
    # Create safe directory name
    SAFE_NAME=$(echo "$MOUNTPOINT" | tr '/' '_' | sed 's/^_//')
    DEST="$BACKUP_DIR/$DATE/$SAFE_NAME"
    
    echo "Backing up: $SOURCE ($FSTYPE) -> $DEST"
    mkdir -p "$DEST"
    
    rsync -av --delete \
        --exclude='*.tmp' \
        --exclude='.Trash*' \
        "$MOUNTPOINT/" \
        "$DEST/" 2>&1 | tail -3
    
    echo "  Done: $(du -sh "$DEST" | awk '{print $1}')"
done

echo "Total backup size: $(du -sh "$BACKUP_DIR/$DATE" 2>/dev/null | awk '{print $1}')"`,
				},
			},
		},
	})
}
