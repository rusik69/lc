package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1832,
			Title:       "Linux High Availability and Clustering",
			Description: "Implement high availability on Linux: Pacemaker/Corosync clustering, DRBD replication, keepalived, load balancing, and failover strategies.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "Pacemaker and Corosync Clustering",
					Content: `Linux HA clustering provides automatic failover for critical services. Pacemaker (resource manager) and Corosync (cluster communication) are the standard stack.

**Cluster Architecture:**
` + "```" + `
Components:
  Corosync:  Cluster communication layer
             - Node membership and quorum
             - Reliable messaging between nodes
             - Failure detection
  
  Pacemaker: Cluster resource manager
             - Decides where resources run
             - Handles failover
             - Manages resource dependencies
             - Enforces constraints

Resource types:
  Primitive:  Single instance of a service
  Clone:      Run on multiple nodes (active/active)
  Multi-state: Master/slave (one primary, others secondary)
  Group:      Set of resources managed together
  Bundle:     Container-based resources

Resource agents:
  ocf:       Open Cluster Framework (most complete)
  systemd:   Systemd services
  lsb:       Linux Standard Base init scripts
  service:   Generic service management
  stonith:   Fencing devices

  # List available agents
  pcs resource agents ocf
  pcs resource agents systemd
  pcs resource describe ocf:heartbeat:IPaddr2
` + "```" + `

**Cluster Setup:**
` + "```" + `
Installation (RHEL/CentOS):
  dnf install pacemaker corosync pcs fence-agents-all
  
  # Enable and start pcsd
  systemctl enable --now pcsd
  
  # Set hacluster password (same on all nodes)
  passwd hacluster
  
  # Authenticate nodes
  pcs host auth node1 node2 node3 -u hacluster -p password
  
  # Create cluster
  pcs cluster setup mycluster node1 node2 node3
  
  # Start cluster
  pcs cluster start --all
  pcs cluster enable --all

Quorum:
  # Quorum = majority of nodes must agree
  # 2 nodes: needs special handling (no majority possible)
  # 3 nodes: can lose 1
  # 5 nodes: can lose 2
  
  # For 2-node cluster:
  pcs quorum update two-node=1
  # Or use quorum device (qdevice)
  
  # Check quorum
  pcs quorum status
  corosync-quorumtool

Fencing (STONITH):
  # STONITH = Shoot The Other Node In The Head
  # Required for production clusters!
  # Ensures failed nodes don't corrupt shared data
  
  # Types: IPMI, iLO/iDRAC, cloud APIs, SBD
  
  # IPMI fencing
  pcs stonith create ipmi-node1 fence_ipmilan \
    ipaddr=192.168.1.101 login=admin passwd=secret \
    pcmk_host_list=node1
  
  # Cloud fencing (AWS)
  pcs stonith create fence-aws fence_aws \
    region=us-east-1 \
    pcmk_host_map="node1:i-1234;node2:i-5678"
  
  # Test fencing
  pcs stonith fence node1
` + "```" + `

**Resource Configuration:**
` + "```" + `
Cluster management (pcs):
  pcs status                       # Cluster status
  pcs resource status              # Resource status
  pcs resource create              # Create resource
  pcs resource delete              # Delete resource
  pcs resource move                # Move resource
  pcs resource ban                 # Ban from node
  pcs resource clear               # Clear constraints
  pcs resource cleanup             # Clear error state
  pcs constraint                   # Manage constraints

Example: Virtual IP + Web Server:
  # Create virtual IP
  pcs resource create VirtualIP ocf:heartbeat:IPaddr2 \
    ip=192.168.1.100 cidr_netmask=24 \
    op monitor interval=30s
  
  # Create web server resource
  pcs resource create WebServer systemd:nginx \
    op start timeout=60s \
    op stop timeout=60s \
    op monitor interval=30s timeout=30s
  
  # Group them (colocate and order)
  pcs resource group add WebGroup VirtualIP WebServer
  
  # Or use constraints separately:
  # Colocation: run on same node
  pcs constraint colocation add WebServer with VirtualIP
  
  # Order: start VirtualIP before WebServer
  pcs constraint order VirtualIP then WebServer
  
  # Location: prefer node1
  pcs constraint location WebServer prefers node1=100

Resource operations:
  op start    timeout=60s          # Max time to start
  op stop     timeout=60s          # Max time to stop
  op monitor  interval=30s timeout=30s  # Health check

Migration:
  # Manual move
  pcs resource move WebGroup node2
  
  # Automatic failback
  pcs resource meta WebGroup migration-threshold=3
  pcs resource meta WebGroup failure-timeout=60s
  # After 3 failures within 60s → migrate
` + "```" + ``,
					CodeExamples: `# HA Cluster configuration examples

# 1. Two-node HA cluster setup script
#!/bin/bash
set -e

NODE1="node1"
NODE2="node2"
CLUSTER_NAME="prod-cluster"
VIP="192.168.1.100"
PASSWD="hacluster_password"

echo "=== Setting up HA Cluster ==="

# Install packages
dnf install -y pacemaker corosync pcs fence-agents-all

# Enable and start pcsd
systemctl enable --now pcsd

# Set hacluster password
echo "$PASSWD" | passwd --stdin hacluster

# Authenticate
pcs host auth "$NODE1" "$NODE2" -u hacluster -p "$PASSWD"

# Create cluster
pcs cluster setup "$CLUSTER_NAME" "$NODE1" "$NODE2"

# Start cluster
pcs cluster start --all
pcs cluster enable --all

# Wait for cluster
sleep 10

# Configure for 2-node cluster
pcs property set stonith-enabled=false  # ONLY for testing!
pcs property set no-quorum-policy=ignore

# Create virtual IP
pcs resource create VirtualIP ocf:heartbeat:IPaddr2 \
    ip="$VIP" cidr_netmask=24 \
    op monitor interval=10s

# Create Nginx resource
pcs resource create WebServer systemd:nginx \
    op start timeout=30s \
    op stop timeout=30s \
    op monitor interval=10s timeout=10s on-fail=restart

# Group resources
pcs resource group add WebGroup VirtualIP WebServer

# Set preferred node
pcs constraint location WebGroup prefers "$NODE1"=100

# Verify
pcs status

# 2. DRBD + Pacemaker configuration
# Create DRBD resource
pcs resource create DRBD ocf:linbit:drbd \
    drbd_resource=r0 \
    op monitor interval=15s role=Master \
    op monitor interval=30s role=Slave

# Make it a multi-state (master/slave) resource
pcs resource promotable DRBD \
    promoted-max=1 \
    promoted-node-max=1 \
    clone-max=2 \
    clone-node-max=1

# Create filesystem on DRBD
pcs resource create DRBDfs ocf:heartbeat:Filesystem \
    device=/dev/drbd0 \
    directory=/data \
    fstype=ext4

# Colocation and ordering
pcs constraint colocation add DRBDfs with DRBD-clone INFINITY with-rsc-role=Master
pcs constraint order promote DRBD-clone then start DRBDfs

# 3. Cluster monitoring script
#!/bin/bash
echo "=== Cluster Health Check ==="

# Check if cluster is running
if ! pcs cluster status > /dev/null 2>&1; then
    echo "CRITICAL: Cluster is not running!"
    exit 2
fi

# Check node status
OFFLINE=$(pcs status nodes | grep -c "Offline:" || true)
if [ "$OFFLINE" -gt 0 ]; then
    echo "WARNING: Offline nodes detected"
    pcs status nodes
fi

# Check resource status
STOPPED=$(pcs resource status 2>/dev/null | grep -c "Stopped" || true)
FAILED=$(pcs resource failcount show 2>/dev/null | grep -v "^$" | wc -l || true)

if [ "$STOPPED" -gt 0 ]; then
    echo "WARNING: $STOPPED stopped resources"
fi
if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: Resources with failures detected"
fi

# Check quorum
if ! corosync-quorumtool -s 2>/dev/null | grep -q "Quorate:.*Yes"; then
    echo "CRITICAL: Cluster has lost quorum!"
    exit 2
fi

echo "Cluster status: OK"
pcs status --brief`,
				},
				{
					Title: "keepalived and Load Balancing",
					Content: `keepalived provides simple HA via VRRP (Virtual Router Redundancy Protocol) and integrates with IPVS for load balancing. It's simpler than Pacemaker for basic VIP failover.

**keepalived Architecture:**
` + "```" + `
Components:
  VRRP: Virtual Router Redundancy Protocol
        - One master, one or more backups
        - Master holds virtual IP
        - Backup takes over on master failure
        - Priority-based election
  
  IPVS: IP Virtual Server (Linux kernel LB)
        - Layer 4 load balancing in kernel
        - NAT, Direct Routing, or IP Tunneling
        - Health checking for backend servers

Installation:
  apt install keepalived        # Debian/Ubuntu
  dnf install keepalived        # RHEL/CentOS
` + "```" + `

**VRRP Configuration:**
` + "```" + `
/etc/keepalived/keepalived.conf:

  # Global
  global_defs {
    router_id LB_MASTER
    enable_script_security
    script_user keepalived_script
  }

  # Health check script
  vrrp_script check_nginx {
    script "/usr/local/bin/check_nginx.sh"
    interval 2        # Check every 2 seconds
    weight -20         # Reduce priority by 20 on failure
    fall 3             # 3 failures to mark down
    rise 2             # 2 successes to mark up
  }

  # VRRP instance
  vrrp_instance VI_1 {
    state MASTER           # MASTER on primary, BACKUP on secondary
    interface eth0
    virtual_router_id 51   # Must be same on both nodes (0-255)
    priority 100           # Higher = more preferred (MASTER: 100, BACKUP: 90)
    advert_int 1           # Advertisement interval (seconds)
    
    # Authentication
    authentication {
      auth_type PASS
      auth_pass secretpass
    }
    
    # Virtual IPs
    virtual_ipaddress {
      192.168.1.100/24
    }
    
    # Track scripts
    track_script {
      check_nginx
    }
    
    # Track interfaces
    track_interface {
      eth0
    }
    
    # Notification scripts
    notify_master "/usr/local/bin/notify.sh master"
    notify_backup "/usr/local/bin/notify.sh backup"
    notify_fault  "/usr/local/bin/notify.sh fault"
  }

Backup node configuration:
  vrrp_instance VI_1 {
    state BACKUP
    interface eth0
    virtual_router_id 51
    priority 90              # Lower than master
    advert_int 1
    authentication {
      auth_type PASS
      auth_pass secretpass
    }
    virtual_ipaddress {
      192.168.1.100/24
    }
    track_script {
      check_nginx
    }
  }
` + "```" + `

**IPVS Load Balancing:**
` + "```" + `
IPVS modes:
  NAT:     Load balancer modifies destination IP
           Real servers use LB as default gateway
           Works with any protocol
           LB is bottleneck (all traffic flows through)
  
  DR:      Direct Routing (most common for production)
           LB modifies MAC address only
           Real servers respond directly to client
           Best performance (response bypasses LB)
           Real servers must be on same L2 network
  
  TUN:     IP-in-IP tunneling
           Like DR but works across L3 networks
           Higher overhead than DR

IPVS commands (ipvsadm):
  # View rules
  ipvsadm -Ln
  
  # Add virtual service
  ipvsadm -A -t 192.168.1.100:80 -s rr
  
  # Add real servers
  ipvsadm -a -t 192.168.1.100:80 -r 192.168.1.10:80 -g  # -g = DR
  ipvsadm -a -t 192.168.1.100:80 -r 192.168.1.11:80 -g
  
  # Scheduling algorithms:
  rr:     Round Robin
  wrr:    Weighted Round Robin
  lc:     Least Connections
  wlc:    Weighted Least Connections (default)
  sh:     Source Hashing (session persistence)
  dh:     Destination Hashing

keepalived IPVS configuration:
  virtual_server 192.168.1.100 80 {
    delay_loop 5
    lb_algo wlc
    lb_kind DR
    persistence_timeout 300
    protocol TCP
    
    real_server 192.168.1.10 80 {
      weight 100
      TCP_CHECK {
        connect_timeout 3
        connect_port 80
      }
    }
    
    real_server 192.168.1.11 80 {
      weight 100
      HTTP_GET {
        url {
          path /healthz
          status_code 200
        }
        connect_timeout 3
        retry 3
        delay_before_retry 2
      }
    }
  }
` + "```" + `

**HAProxy (Alternative Load Balancer):**
` + "```" + `
HAProxy: software load balancer
  - Layer 4 and Layer 7 load balancing
  - SSL termination
  - Health checking
  - Connection draining
  - Rate limiting
  - Stick tables (session affinity)

/etc/haproxy/haproxy.cfg:
  global
    maxconn 50000
    log /dev/log local0
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    user haproxy
    group haproxy
    daemon
    ssl-default-bind-ciphersuites TLS_AES_128_GCM_SHA256
    ssl-default-bind-options no-sslv3 no-tlsv10 no-tlsv11
  
  defaults
    mode http
    log global
    option httplog
    option dontlognull
    option http-server-close
    option redispatch
    timeout connect 5s
    timeout client 30s
    timeout server 30s
    retries 3
    
  frontend http_front
    bind *:80
    bind *:443 ssl crt /etc/haproxy/certs/
    redirect scheme https code 301 if !{ ssl_fc }
    default_backend web_servers
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny deny_status 429 if { sc_http_req_rate(0) gt 100 }
  
  backend web_servers
    balance roundrobin
    option httpchk GET /healthz
    http-check expect status 200
    server web1 192.168.1.10:8080 check inter 5s fall 3 rise 2
    server web2 192.168.1.11:8080 check inter 5s fall 3 rise 2
    server web3 192.168.1.12:8080 check inter 5s fall 3 rise 2 backup
  
  listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
    stats admin if LOCALHOST
` + "```" + ``,
					CodeExamples: `# HA and Load Balancing configuration

# 1. keepalived health check script
#!/bin/bash
# /usr/local/bin/check_nginx.sh
# Exit 0 = healthy, Exit 1 = unhealthy

# Check if nginx process exists
if ! pidof nginx > /dev/null 2>&1; then
    exit 1
fi

# Check if nginx responds
if ! curl -sf -o /dev/null -w "%{http_code}" http://localhost/healthz | grep -q "200"; then
    exit 1
fi

exit 0

# 2. keepalived notification script
#!/bin/bash
# /usr/local/bin/notify.sh
TYPE=$1
NAME=$2
ENDSTATE=$3

case "$ENDSTATE" in
    MASTER)
        logger "keepalived: Became MASTER for $NAME"
        # Start the service or update DNS
        systemctl start nginx 2>/dev/null
        ;;
    BACKUP)
        logger "keepalived: Became BACKUP for $NAME"
        ;;
    FAULT)
        logger "keepalived: FAULT state for $NAME"
        # Alert
        ;;
esac

# 3. Complete keepalived + IPVS configuration
# /etc/keepalived/keepalived.conf (Master)
global_defs {
    router_id LB_MASTER
    enable_script_security
}

vrrp_script check_haproxy {
    script "/usr/bin/killall -0 haproxy"
    interval 2
    weight -20
    fall 3
    rise 2
}

vrrp_instance VI_WEB {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass mypass123
    }
    virtual_ipaddress {
        192.168.1.100/24
    }
    track_script {
        check_haproxy
    }
}

virtual_server 192.168.1.100 80 {
    delay_loop 5
    lb_algo wlc
    lb_kind DR
    persistence_timeout 300
    protocol TCP
    
    real_server 192.168.1.10 80 {
        weight 100
        HTTP_GET {
            url {
                path /healthz
                status_code 200
            }
            connect_timeout 3
            retry 3
        }
    }
    
    real_server 192.168.1.11 80 {
        weight 100
        HTTP_GET {
            url {
                path /healthz
                status_code 200
            }
            connect_timeout 3
            retry 3
        }
    }
}

virtual_server 192.168.1.100 443 {
    delay_loop 5
    lb_algo wlc
    lb_kind DR
    persistence_timeout 300
    protocol TCP
    
    real_server 192.168.1.10 443 {
        weight 100
        TCP_CHECK {
            connect_timeout 3
            connect_port 443
        }
    }
    
    real_server 192.168.1.11 443 {
        weight 100
        TCP_CHECK {
            connect_timeout 3
            connect_port 443
        }
    }
}

# 4. DR mode real server setup script
#!/bin/bash
# Run on each real server for Direct Routing mode
VIP="192.168.1.100"

# Configure loopback alias for VIP
ip addr add "$VIP"/32 dev lo

# Disable ARP for VIP on real servers
echo 1 > /proc/sys/net/ipv4/conf/lo/arp_ignore
echo 2 > /proc/sys/net/ipv4/conf/lo/arp_announce
echo 1 > /proc/sys/net/ipv4/conf/all/arp_ignore
echo 2 > /proc/sys/net/ipv4/conf/all/arp_announce

echo "DR mode configured for VIP $VIP"`,
				},
			},
		},
	})
}
