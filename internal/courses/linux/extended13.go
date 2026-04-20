package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1835,
			Title:       "Linux Package Management and Build Systems",
			Description: "Master Linux package management across distributions, building packages, managing repositories, and understanding the software supply chain.",
			Order:       35,
			Lessons: []problems.Lesson{
				{
					Title: "Package Management Deep Dive",
					Content: `Package management is fundamental to Linux administration. Understanding how packages work across distributions enables efficient system management.

**Debian/Ubuntu (APT):**
` + "```" + `
Package format: .deb

Tools:
  dpkg     Low-level package manager
  apt      High-level (recommended)
  apt-get  Traditional high-level
  apt-cache  Package metadata

dpkg (low level):
  dpkg -i package.deb              # Install
  dpkg -r package                  # Remove
  dpkg -P package                  # Purge (remove + config)
  dpkg -l                          # List installed
  dpkg -l 'nginx*'                 # List matching
  dpkg -s package                  # Package status/info
  dpkg -L package                  # List files in package
  dpkg -S /path/to/file            # Which package owns file
  dpkg --configure -a              # Fix broken packages

apt:
  apt update                       # Update package lists
  apt upgrade                      # Upgrade installed packages
  apt full-upgrade                 # Upgrade with dependency changes
  apt install nginx                # Install
  apt install nginx=1.22.0-1       # Install specific version
  apt remove nginx                 # Remove
  apt purge nginx                  # Remove + config files
  apt autoremove                   # Remove unused dependencies
  apt search keyword               # Search packages
  apt show nginx                   # Package details
  apt list --installed             # List installed
  apt list --upgradable            # List upgradable
  apt policy nginx                 # Version and repository info

Repository management:
  # Add repository
  add-apt-repository ppa:user/repo       # Ubuntu PPA
  
  # Manual repo (modern way)
  # 1. Add GPG key
  curl -fsSL https://example.com/key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/example.gpg
  
  # 2. Add repository
  echo "deb [signed-by=/usr/share/keyrings/example.gpg] \
    https://packages.example.com/apt stable main" > \
    /etc/apt/sources.list.d/example.list
  
  # 3. Update and install
  apt update
  apt install example-package

  # Pin package version (prevent upgrade)
  cat > /etc/apt/preferences.d/nginx << EOF
  Package: nginx
  Pin: version 1.22.0*
  Pin-Priority: 1001
  EOF
` + "```" + `

**RHEL/CentOS/Fedora (DNF/YUM):**
` + "```" + `
Package format: .rpm

Tools:
  rpm      Low-level package manager
  dnf      High-level (modern, replaced yum)
  yum      Traditional high-level (still available)

rpm (low level):
  rpm -ivh package.rpm             # Install
  rpm -Uvh package.rpm             # Upgrade (or install)
  rpm -e package                   # Remove
  rpm -qa                          # List all installed
  rpm -qi package                  # Package info
  rpm -ql package                  # List files
  rpm -qf /path/to/file            # Which package owns file
  rpm -V package                   # Verify (check modified files)

dnf:
  dnf check-update                 # Check for updates
  dnf update                       # Update all
  dnf install nginx                # Install
  dnf install nginx-1.22.0         # Specific version
  dnf remove nginx                 # Remove
  dnf autoremove                   # Remove unused deps
  dnf search keyword               # Search
  dnf info nginx                   # Package details
  dnf list installed               # List installed
  dnf list available               # List available
  dnf history                      # Transaction history
  dnf history undo <id>            # Undo transaction
  dnf provides /path/to/file       # Which package provides file
  dnf group list                   # List package groups
  dnf group install "Development Tools"
  
  # Repository management
  dnf repolist                     # List repos
  dnf config-manager --add-repo https://example.com/repo
  dnf config-manager --set-enabled repo-name
  dnf config-manager --set-disabled repo-name

  # Version locking
  dnf install python3-dnf-plugin-versionlock
  dnf versionlock add nginx
  dnf versionlock list
  dnf versionlock delete nginx
` + "```" + `

**Building Packages:**
` + "```" + `
Building .deb packages:
  # Simple approach with dpkg-deb
  mkdir -p mypackage/DEBIAN
  mkdir -p mypackage/usr/local/bin
  
  # Copy binary
  cp myapp mypackage/usr/local/bin/
  
  # Create control file
  cat > mypackage/DEBIAN/control << EOF
  Package: myapp
  Version: 1.0.0
  Architecture: amd64
  Maintainer: Name <email@example.com>
  Description: My Application
  Depends: libc6 (>= 2.17)
  EOF
  
  # Build
  dpkg-deb --build mypackage myapp_1.0.0_amd64.deb

Building .rpm packages:
  # Install build tools
  dnf install rpmdevtools rpmlint
  
  # Create build tree
  rpmdev-setuptree
  # Creates: ~/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
  
  # Create spec file
  cat > ~/rpmbuild/SPECS/myapp.spec << 'EOF'
  Name:    myapp
  Version: 1.0.0
  Release: 1%{?dist}
  Summary: My Application
  License: MIT
  Source0: myapp-1.0.0.tar.gz
  
  %description
  My Application description
  
  %prep
  %setup -q
  
  %build
  make
  
  %install
  install -D -m 755 myapp %{buildroot}/usr/local/bin/myapp
  
  %files
  /usr/local/bin/myapp
  EOF
  
  # Build
  rpmbuild -ba ~/rpmbuild/SPECS/myapp.spec
` + "```" + `

**Container vs System Packages:**
` + "```" + `
Modern deployment options:
  1. System packages (deb/rpm)
     + Native to OS
     + Dependency resolution
     + Security updates via OS
     - Version conflicts possible
     - Hard to isolate
  
  2. Containers (Docker/Podman)
     + Complete isolation
     + Reproducible
     + Easy rollback
     + Language agnostic
     - Overhead (disk, memory)
     - Orchestration needed
  
  3. Flatpak/Snap (desktop apps)
     + Sandboxed
     + Cross-distribution
     - Large size
     - Desktop-focused
  
  4. AppImage
     + Single file, portable
     + No installation
     - No auto-updates
     - Desktop-focused

Best practices:
  - System services: system packages or containers
  - Development: containers (reproducible environments)
  - Production: containers + orchestration (Kubernetes)
  - Desktop: Flatpak or native packages
` + "```" + ``,
					CodeExamples: `# Package management scripts

# 1. Cross-distro package installer
#!/bin/bash
install_package() {
    local pkg="$1"
    
    if command -v apt-get > /dev/null 2>&1; then
        apt-get install -y "$pkg"
    elif command -v dnf > /dev/null 2>&1; then
        dnf install -y "$pkg"
    elif command -v yum > /dev/null 2>&1; then
        yum install -y "$pkg"
    elif command -v pacman > /dev/null 2>&1; then
        pacman -S --noconfirm "$pkg"
    else
        echo "No supported package manager found"
        return 1
    fi
}

# Map common package names across distros
install_common() {
    local name="$1"
    
    if command -v apt-get > /dev/null 2>&1; then
        case "$name" in
            httpd) install_package "apache2" ;;
            vim)   install_package "vim" ;;
            *)     install_package "$name" ;;
        esac
    elif command -v dnf > /dev/null 2>&1; then
        case "$name" in
            apache2) install_package "httpd" ;;
            *)       install_package "$name" ;;
        esac
    fi
}

# 2. Security update checker
#!/bin/bash
echo "=== Security Update Check ==="

if command -v apt-get > /dev/null 2>&1; then
    apt-get update -qq
    SECURITY=$(apt-get -s upgrade 2>/dev/null | grep -i security | wc -l)
    TOTAL=$(apt list --upgradable 2>/dev/null | tail -n+2 | wc -l)
    echo "Total upgradable: $TOTAL"
    echo "Security updates: $SECURITY"
    
    if [ "$SECURITY" -gt 0 ]; then
        echo ""
        echo "Security packages:"
        apt-get -s upgrade 2>/dev/null | grep -i security
    fi
    
elif command -v dnf > /dev/null 2>&1; then
    SECURITY=$(dnf updateinfo list security 2>/dev/null | wc -l)
    TOTAL=$(dnf check-update 2>/dev/null | grep -c "^\S")
    echo "Total upgradable: $TOTAL"
    echo "Security updates: $SECURITY"
fi

# 3. Package audit script
#!/bin/bash
echo "=== Package Audit ==="

if command -v dpkg > /dev/null 2>&1; then
    TOTAL=$(dpkg -l | grep '^ii' | wc -l)
    echo "Installed packages: $TOTAL"
    
    echo ""
    echo "--- Manually installed ---"
    apt-mark showmanual 2>/dev/null | wc -l
    
    echo ""
    echo "--- Packages not from official repos ---"
    apt list --installed 2>/dev/null | grep -v "ubuntu\|debian" | tail -n+2 | head -20
    
    echo ""
    echo "--- Modified config files ---"
    dpkg --verify 2>/dev/null | grep "^..5" | head -20
    
elif command -v rpm > /dev/null 2>&1; then
    TOTAL=$(rpm -qa | wc -l)
    echo "Installed packages: $TOTAL"
    
    echo ""
    echo "--- Packages not from official repos ---"
    rpm -qa --qf '%{NAME} %{VENDOR}\n' | grep -v "Red Hat\|CentOS\|Fedora" | head -20
    
    echo ""
    echo "--- Modified files ---"
    rpm -Va 2>/dev/null | grep "^..5" | head -20
fi

# 4. Automatic security updates setup
#!/bin/bash
if command -v apt-get > /dev/null 2>&1; then
    # Debian/Ubuntu: unattended-upgrades
    apt-get install -y unattended-upgrades apt-listchanges
    
    cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

    cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF
    
    echo "Unattended security upgrades configured"

elif command -v dnf > /dev/null 2>&1; then
    # RHEL/CentOS: dnf-automatic
    dnf install -y dnf-automatic
    
    sed -i 's/apply_updates = no/apply_updates = yes/' \
        /etc/dnf/automatic.conf
    sed -i 's/upgrade_type = default/upgrade_type = security/' \
        /etc/dnf/automatic.conf
    
    systemctl enable --now dnf-automatic.timer
    echo "DNF automatic security updates configured"
fi`,
				},
				{
					Title: "System Monitoring and Observability",
					Content: `Comprehensive monitoring ensures you detect and respond to issues before they impact users. The observability stack on Linux includes metrics, logs, and traces.

**Prometheus Node Exporter:**
` + "```" + `
Node Exporter exposes Linux system metrics for Prometheus.

Installation:
  # Download and install
  wget https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz
  tar xzf node_exporter-*.tar.gz
  cp node_exporter-*/node_exporter /usr/local/bin/
  
  # Create systemd service
  useradd -r -s /sbin/nologin node_exporter
  
  # /etc/systemd/system/node_exporter.service
  [Unit]
  Description=Node Exporter
  After=network.target
  
  [Service]
  Type=simple
  User=node_exporter
  ExecStart=/usr/local/bin/node_exporter \
    --collector.systemd \
    --collector.processes \
    --web.listen-address=:9100
  Restart=on-failure
  
  [Install]
  WantedBy=multi-user.target

Key metrics:
  CPU:
    node_cpu_seconds_total          Per-CPU, per-mode time
    node_load1, node_load5          Load averages
    
  Memory:
    node_memory_MemTotal_bytes
    node_memory_MemAvailable_bytes
    node_memory_SwapTotal_bytes
    node_memory_SwapFree_bytes
    
  Disk:
    node_disk_read_bytes_total
    node_disk_written_bytes_total
    node_disk_io_time_seconds_total
    node_filesystem_avail_bytes
    node_filesystem_size_bytes
    
  Network:
    node_network_receive_bytes_total
    node_network_transmit_bytes_total
    node_network_receive_errs_total

Useful PromQL queries:
  # CPU utilization
  100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
  
  # Memory usage
  (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100
  
  # Disk usage
  (1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100
  
  # Network throughput
  rate(node_network_receive_bytes_total{device="eth0"}[5m])
` + "```" + `

**Log Aggregation:**
` + "```" + `
Common stacks:
  ELK:   Elasticsearch + Logstash + Kibana
  EFK:   Elasticsearch + Fluentd + Kibana
  PLG:   Promtail + Loki + Grafana (lighter weight)

Promtail + Loki (recommended for Kubernetes/Linux):
  - Loki: log aggregation (like Prometheus for logs)
  - Promtail: log collector (tail files, send to Loki)
  - Only indexes labels, not full text (efficient storage)

  Promtail config (/etc/promtail/config.yml):
    server:
      http_listen_port: 9080
    
    positions:
      filename: /tmp/positions.yaml
    
    clients:
      - url: http://loki:3100/loki/api/v1/push
    
    scrape_configs:
      - job_name: syslog
        static_configs:
        - targets: [localhost]
          labels:
            job: syslog
            host: myserver
            __path__: /var/log/syslog
      
      - job_name: journal
        journal:
          labels:
            job: journal
        relabel_configs:
        - source_labels: ['__journal__systemd_unit']
          target_label: 'unit'

rsyslog forwarding:
  # Forward logs to central server
  /etc/rsyslog.d/50-remote.conf:
    *.* @@logserver.example.com:514   # TCP
    *.* @logserver.example.com:514    # UDP
    
  # TLS forwarding
  module(load="omrelp")
  action(type="omrelp"
         target="logserver.example.com"
         port="2514"
         tls="on"
         tls.caCert="/etc/rsyslog.d/ca.pem"
         tls.myCert="/etc/rsyslog.d/client-cert.pem"
         tls.myPrivKey="/etc/rsyslog.d/client-key.pem")
` + "```" + `

**Alerting:**
` + "```" + `
Essential alerts for Linux servers:

  Critical (page):
    - Host down (no metrics for 5 min)
    - Disk > 90% full
    - OOM kills detected
    - RAID degraded
    - Swap usage > 50%
    - systemd service failed
    
  Warning (ticket):
    - CPU > 80% for 15 min
    - Memory > 85% for 15 min
    - Disk > 80% full
    - High disk I/O latency (> 100ms)
    - Network errors increasing
    - NTP out of sync
    - SSL certificate expiring < 30 days
    - Security updates pending

Alertmanager rules (Prometheus):
  groups:
  - name: linux
    rules:
    - alert: HostDown
      expr: up{job="node"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Host {{ $labels.instance }} is down"
    
    - alert: DiskFull
      expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Disk nearly full on {{ $labels.instance }}"
    
    - alert: HighMemory
      expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 85
      for: 15m
      labels:
        severity: warning
` + "```" + ``,
					CodeExamples: `# Monitoring configuration

# 1. Prometheus scrape config for Linux hosts
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts/*.yml"

alerting:
  alertmanagers:
  - static_configs:
    - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'node'
    static_configs:
    - targets:
      - 'web1:9100'
      - 'web2:9100'
      - 'db1:9100'
    relabel_configs:
    - source_labels: [__address__]
      regex: '(.+):.*'
      target_label: instance

  - job_name: 'node-discovery'
    file_sd_configs:
    - files:
      - '/etc/prometheus/targets/*.json'
      refresh_interval: 30s

# 2. Grafana dashboard JSON (key panels)
# Dashboard: Linux Server Overview
# Panels:
#  - CPU Usage: 100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
#  - Memory Usage: (1 - node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes) * 100
#  - Disk Usage: (1 - node_filesystem_avail_bytes/node_filesystem_size_bytes) * 100
#  - Network I/O: rate(node_network_receive_bytes_total[5m])
#  - Disk I/O: rate(node_disk_read_bytes_total[5m])
#  - Load Average: node_load1
#  - Uptime: time() - node_boot_time_seconds

# 3. Custom metrics exporter
#!/bin/bash
# /usr/local/bin/custom_exporter.sh
# Run via cron or systemd timer, writes to textfile collector dir

TEXTFILE_DIR="/var/lib/node_exporter/textfile_collector"
mkdir -p "$TEXTFILE_DIR"

# Custom application metrics
{
    # Active users
    USERS=$(who | wc -l)
    echo "system_active_users $USERS"
    
    # Pending security updates
    if command -v apt-get > /dev/null 2>&1; then
        UPDATES=$(apt-get -s upgrade 2>/dev/null | grep -c "^Inst.*security")
        echo "system_security_updates_pending $UPDATES"
    fi
    
    # Connection count
    CONNS=$(ss -tn state established | wc -l)
    echo "system_established_connections $CONNS"
    
    # Zombie processes
    ZOMBIES=$(ps aux | awk '$8 ~ /Z/' | wc -l)
    echo "system_zombie_processes $ZOMBIES"
    
    # Failed systemd services
    FAILED=$(systemctl --failed --no-legend 2>/dev/null | wc -l)
    echo "system_failed_services $FAILED"
    
} > "$TEXTFILE_DIR/custom.prom.$$"
mv "$TEXTFILE_DIR/custom.prom.$$" "$TEXTFILE_DIR/custom.prom"

# 4. Log monitoring with journald
#!/bin/bash
# Monitor for critical events in journal
journalctl -f -p err --no-pager -o json | while read -r line; do
    MESSAGE=$(echo "$line" | jq -r '.MESSAGE // empty')
    UNIT=$(echo "$line" | jq -r '._SYSTEMD_UNIT // "unknown"')
    
    # Simple alerting
    case "$MESSAGE" in
        *"Out of memory"*)
            echo "CRITICAL: OOM event - $MESSAGE"
            # Send alert
            ;;
        *"segfault"*|*"SIGSEGV"*)
            echo "CRITICAL: Segfault in $UNIT - $MESSAGE"
            ;;
        *"failed"*|*"error"*)
            echo "ERROR: $UNIT - $MESSAGE"
            ;;
    esac
done`,
				},
			},
		},
	})
}
