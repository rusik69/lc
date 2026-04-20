package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1834,
			Title:       "Linux Troubleshooting Methodology",
			Description: "Systematic Linux troubleshooting: boot failures, service issues, performance problems, network debugging, and production incident response.",
			Order:       34,
			Lessons: []problems.Lesson{
				{
					Title: "Boot and Service Troubleshooting",
					Content: `Systematic troubleshooting requires understanding the boot process, service management, and common failure modes. A methodical approach saves time.

**Boot Troubleshooting:**
` + "```" + `
Boot failure stages and symptoms:

  BIOS/UEFI problems:
    Symptom: No POST, beep codes, no display
    Fix: Check hardware, reset BIOS, check boot order
  
  GRUB problems:
    Symptom: "GRUB rescue>" prompt, "error: no such partition"
    Fix:
      # From GRUB rescue
      ls                           # List partitions
      ls (hd0,gpt2)/               # Check for /boot contents
      set root=(hd0,gpt2)
      set prefix=(hd0,gpt2)/boot/grub
      insmod normal
      normal
      # Then from running system: grub-install /dev/sda && update-grub
  
  Kernel problems:
    Symptom: Kernel panic, blank screen after GRUB
    Fix:
      - Boot old kernel from GRUB menu
      - Add "nomodeset" to kernel parameters
      - Boot with "init=/bin/bash" for rescue shell
  
  initramfs problems:
    Symptom: "Unable to mount root fs", dracut emergency shell
    Fix:
      - Check root= parameter in GRUB
      - Regenerate initramfs:
        mkinitramfs -o /boot/initrd.img-$(uname -r)
        # or: dracut -f
  
  Filesystem problems:
    Symptom: "fsck failed", read-only mount, filesystem errors
    Fix:
      - Boot to rescue mode
      - fsck /dev/sda1
      - Check /etc/fstab for errors
      - Temporarily comment out problematic mounts
  
  systemd problems:
    Symptom: Stuck at boot, specific services failing
    Fix:
      - Boot with systemd.unit=rescue.target
      - or systemd.unit=emergency.target
      - journalctl -xb for current boot logs
      - systemctl --failed

GRUB editing at boot:
  1. At GRUB menu, press 'e' to edit
  2. Find the 'linux' line
  3. Add parameters:
     - single or 1: single-user mode
     - systemd.unit=rescue.target: rescue mode
     - systemd.unit=emergency.target: minimal shell
     - init=/bin/bash: bypass init entirely
     - rd.break: break into initramfs
     - nomodeset: disable kernel mode setting
  4. Ctrl+X or F10 to boot
` + "```" + `

**Service Troubleshooting:**
` + "```" + `
Service won't start:
  1. Check status
     systemctl status service-name
     # Look for: Active status, Main PID, error messages
  
  2. Check logs
     journalctl -u service-name -n 50 --no-pager
     journalctl -u service-name --since "5 minutes ago"
  
  3. Check configuration
     systemctl cat service-name      # View unit file
     service-name --check-config     # If supported
     nginx -t                        # nginx config test
     named-checkconf                 # BIND config test
     apachectl configtest            # Apache config test
  
  4. Check dependencies
     systemctl list-dependencies service-name
     systemctl is-active dependency-name
  
  5. Check resources
     # Port in use?
     ss -tlnp | grep :port
     # File permissions?
     ls -la /path/to/config
     ls -la /path/to/data
     # SELinux?
     ausearch -m avc -ts recent
     # Disk space?
     df -h
  
  6. Run manually
     # Find ExecStart command
     systemctl cat service-name | grep ExecStart
     # Run it manually for detailed error output
     /usr/bin/service-binary --config /etc/service.conf
  
  7. Check for resource exhaustion
     Too many open files:
       cat /proc/sys/fs/file-nr
       ulimit -n
     Out of memory:
       dmesg | grep -i oom
       journalctl -k | grep -i oom
     PID limit:
       cat /proc/sys/kernel/pid_max

Service keeps crashing:
  systemctl status service-name
  # Check: restart count, since when
  
  journalctl -u service-name -p err
  # Check: error patterns, stack traces
  
  # Rate limit restarts
  systemctl show service-name -p RestartUSec
  systemctl show service-name -p StartLimitIntervalUSec
  systemctl show service-name -p StartLimitBurst
  
  # Common causes:
  # - Segfault (check core dump)
  # - OOM killed (check dmesg)
  # - Configuration error on reload
  # - External dependency unavailable
  # - Port already in use after fast restart
` + "```" + `

**Filesystem Troubleshooting:**
` + "```" + `
Disk full:
  df -h                            # Check usage
  df -i                            # Check inodes (can be full even if space available)
  
  # Find large files
  find / -xdev -type f -size +100M -exec ls -lh {} \; 2>/dev/null | sort -k5 -h
  
  # Find large directories
  du -h --max-depth=1 / 2>/dev/null | sort -h | tail -20
  
  # Find deleted but open files (space not freed)
  lsof +L1 | grep deleted
  # Fix: restart the process holding the file
  
  # Quick wins:
  journalctl --vacuum-size=500M    # Trim logs
  apt clean                        # Package cache
  find /tmp -type f -atime +7 -delete  # Old temp files
  docker system prune -af          # Docker garbage

Read-only filesystem:
  mount | grep "ro,"                # Check mount options
  dmesg | tail -50                  # Check for I/O errors
  
  # Remount read-write
  mount -o remount,rw /
  
  # If hardware error:
  smartctl -a /dev/sda             # SMART data
  smartctl -t short /dev/sda       # Run test
  
  # Filesystem repair
  # Must unmount first (or boot from live USB)
  umount /dev/sda1
  fsck -y /dev/sda1
  
  # For XFS:
  xfs_repair /dev/sda1
  # If that fails:
  xfs_repair -L /dev/sda1          # Lost+found (data loss possible)
` + "```" + ``,
					CodeExamples: `# Troubleshooting scripts

# 1. Comprehensive system health check
#!/bin/bash
ISSUES=0

check() {
    local name="$1"
    local status="$2"
    if [ "$status" = "OK" ]; then
        printf "  %-30s [\e[32mOK\e[0m]\n" "$name"
    else
        printf "  %-30s [\e[31mFAIL\e[0m] %s\n" "$name" "$status"
        ISSUES=$((ISSUES + 1))
    fi
}

echo "=== System Health Check ==="
echo ""

# Load
LOAD=$(awk '{print $1}' /proc/loadavg)
CPUS=$(nproc)
if (( $(echo "$LOAD > $CPUS * 2" | bc -l) )); then
    check "CPU Load" "High: $LOAD (${CPUS} CPUs)"
else
    check "CPU Load" "OK"
fi

# Memory
MEM_PCT=$(free | awk '/Mem:/ {printf "%d", ($3/$2)*100}')
if [ "$MEM_PCT" -gt 90 ]; then
    check "Memory" "High: ${MEM_PCT}% used"
else
    check "Memory" "OK"
fi

# Swap
SWAP_USED=$(free | awk '/Swap:/ {print $3}')
if [ "${SWAP_USED:-0}" -gt 0 ]; then
    check "Swap" "In use: $(free -h | awk '/Swap:/ {print $3}')"
else
    check "Swap" "OK"
fi

# Disk space
while read -r usage mount; do
    usage_num=${usage%%%}
    if [ "$usage_num" -gt 85 ]; then
        check "Disk $mount" "${usage} used"
    fi
done < <(df -h --output=pcent,target -x tmpfs -x devtmpfs | tail -n+2)
[ "$ISSUES" -eq 0 ] && check "Disk Space" "OK"

# Failed services
FAILED=$(systemctl --failed --no-legend 2>/dev/null | wc -l)
if [ "$FAILED" -gt 0 ]; then
    check "Services" "$FAILED failed"
    systemctl --failed --no-legend 2>/dev/null | while read -r line; do
        echo "    $line"
    done
else
    check "Services" "OK"
fi

# OOM kills
OOM=$(dmesg 2>/dev/null | grep -c "Out of memory" || echo 0)
if [ "$OOM" -gt 0 ]; then
    check "OOM Kills" "$OOM detected"
else
    check "OOM Kills" "OK"
fi

# DNS resolution
if host google.com > /dev/null 2>&1; then
    check "DNS Resolution" "OK"
else
    check "DNS Resolution" "Failed"
fi

# NTP sync
if timedatectl show --property=NTPSynchronized 2>/dev/null | grep -q "yes"; then
    check "NTP Sync" "OK"
else
    check "NTP Sync" "Not synchronized"
fi

echo ""
echo "Issues found: $ISSUES"
exit $ISSUES

# 2. Service recovery script
#!/bin/bash
SERVICE="${1:?Usage: $0 <service-name>}"

echo "=== Diagnosing $SERVICE ==="

# Current status
echo "--- Status ---"
systemctl status "$SERVICE" --no-pager 2>&1 | head -15

# Recent logs
echo ""
echo "--- Recent Logs ---"
journalctl -u "$SERVICE" -n 20 --no-pager -p warning 2>/dev/null

# Check if port is in use
PORT=$(systemctl cat "$SERVICE" 2>/dev/null | grep -oP '(?<=port=)\d+|(?<=Port=)\d+' | head -1)
if [ -n "$PORT" ]; then
    echo ""
    echo "--- Port $PORT ---"
    HOLDER=$(ss -tlnp | grep ":$PORT " | awk '{print $7}')
    if [ -n "$HOLDER" ]; then
        echo "Port $PORT held by: $HOLDER"
    else
        echo "Port $PORT is free"
    fi
fi

# Check dependencies
echo ""
echo "--- Dependencies ---"
systemctl list-dependencies "$SERVICE" --no-pager 2>/dev/null | head -10

# Attempt recovery
echo ""
echo "--- Attempting Recovery ---"
systemctl reset-failed "$SERVICE" 2>/dev/null
systemctl restart "$SERVICE"
sleep 2
if systemctl is-active "$SERVICE" > /dev/null 2>&1; then
    echo "Service recovered successfully"
else
    echo "Service still failed, check logs above"
fi

# 3. Emergency disk cleanup
#!/bin/bash
echo "=== Emergency Disk Cleanup ==="
echo "Current usage:"
df -h / /var /tmp 2>/dev/null

echo ""
echo "Cleaning..."

# Clean journals (keep 500MB)
journalctl --vacuum-size=500M 2>/dev/null
echo "  Journals trimmed"

# Clean package cache
apt-get clean 2>/dev/null || dnf clean all 2>/dev/null
echo "  Package cache cleaned"

# Clean old kernels (keep current + 1 previous)
# dpkg -l linux-image-* | awk '/^ii.*linux-image-[0-9]/ {print $2}' | \
#   head -n -2 | xargs apt-get -y purge

# Clean tmp
find /tmp -type f -atime +3 -delete 2>/dev/null
find /var/tmp -type f -atime +7 -delete 2>/dev/null
echo "  Temp files cleaned"

# Clean old logs
find /var/log -name "*.gz" -mtime +30 -delete 2>/dev/null
find /var/log -name "*.[0-9]" -mtime +30 -delete 2>/dev/null
echo "  Old logs cleaned"

echo ""
echo "After cleanup:"
df -h / /var /tmp 2>/dev/null`,
				},
				{
					Title: "Performance and Network Incident Response",
					Content: `Production incidents require fast, methodical troubleshooting. Having runbooks and systematic approaches reduces mean time to resolution.

**Performance Incident Response:**
` + "```" + `
60-second analysis checklist:
  1. uptime         → Load averages (CPU pressure?)
  2. dmesg -T | tail → Recent kernel errors?
  3. vmstat 1 5     → CPU, memory, I/O overview
  4. mpstat -P ALL 1 3 → Per-CPU breakdown
  5. pidstat 1 3    → Per-process CPU usage
  6. iostat -xz 1 3 → Disk I/O latency/throughput
  7. free -h        → Memory (available, swap)
  8. sar -n DEV 1 3 → Network throughput
  9. ss -s          → Socket stats
  10. top -b -n 1   → Top consumers

CPU spike:
  1. Identify: top/htop → which process?
  2. Is it expected? (cron job, batch process)
  3. Profile: perf top -p <pid>
  4. Check for runaway loops or infinite recursion
  5. Options: kill, nice, cpulimit

Memory pressure:
  1. free -h → available and swap
  2. vmstat 1 → si/so columns (swap activity)
  3. Top memory consumers: ps aux --sort=-rss | head
  4. Check for memory leaks: /proc/<pid>/status VmRSS over time
  5. dmesg | grep -i oom → OOM kills?
  6. Options:
     - Restart leaking process
     - Increase limits
     - Add swap (temporary)
     - Scale up

Disk I/O bottleneck:
  1. iostat -xz 1 → %util, await times
  2. iotop → which process doing I/O?
  3. Check: is it expected? (backup, migration)
  4. Options:
     - ionice for offending process
     - Move I/O heavy work to off-peak
     - Use faster storage
     - Add caching layer
` + "```" + `

**Network Troubleshooting Runbook:**
` + "```" + `
Connection timeout:
  1. ping destination           → Is it reachable?
  2. traceroute destination     → Where does it fail?
  3. dig destination            → DNS resolving correctly?
  4. ss -tnp | grep :port      → Local connection state?
  5. iptables -L -n -v          → Firewall blocking?
  6. tcpdump -i eth0 host dest  → Packets going out? Coming back?

Connection refused:
  1. Is the service running?
     ss -tlnp | grep :port
  2. Is it listening on right interface?
     : 0.0.0.0 = all, 127.0.0.1 = localhost only
  3. Firewall?
     iptables -L -n | grep port
     nft list ruleset | grep port
  4. SELinux/AppArmor?
     ausearch -m avc -ts recent

Slow network:
  1. Bandwidth: iperf3 between hosts
  2. Latency: ping -c 100 (check avg, stddev)
  3. Packet loss: mtr -n destination
  4. TCP issues: ss -ti → retransmits, cwnd
  5. Interface errors: ip -s link show
  6. NIC statistics: ethtool -S eth0 | grep error
  7. Conntrack: sysctl net.netfilter.nf_conntrack_count

DNS issues:
  1. Can resolve? dig domain @server
  2. Correct answer? dig +trace domain
  3. Cache issue? resolvectl flush-caches
  4. /etc/resolv.conf correct?
  5. systemd-resolved running?
  6. Network allows UDP/TCP 53?
` + "```" + `

**Incident Response Workflow:**
` + "```" + `
Phase 1: Detect and Triage (0-5 minutes)
  - What's the impact? (users affected, services down)
  - What changed recently? (deploy, config change, traffic spike)
  - Is it getting worse or stable?
  - Quick mitigations (rollback, restart, scale up)

Phase 2: Investigate (5-30 minutes)
  - Gather data (logs, metrics, traces)
  - Form hypotheses
  - Test hypotheses systematically
  - Narrow down root cause

Phase 3: Mitigate (varies)
  - Apply fix (config change, code fix, scaling)
  - Verify fix works
  - Monitor for recurrence

Phase 4: Postmortem (next day)
  - Timeline of events
  - Root cause analysis
  - Impact assessment
  - Action items to prevent recurrence
  - Share learnings

Communication template:
  Status: [Investigating | Identified | Monitoring | Resolved]
  Impact: [description of user impact]
  Summary: [what's happening]
  Current actions: [what we're doing]
  ETA: [if known]
  Next update: [when]
` + "```" + `

**Common Production Issues:**
` + "```" + `
1. Connection pool exhaustion:
   Symptom: Application errors, "too many connections"
   Check: ss -s, ss -tnp state established | wc -l
   Fix: Increase pool size, check for connection leaks
        Close idle connections, add connection timeouts

2. File descriptor leak:
   Symptom: "Too many open files"
   Check: ls /proc/<pid>/fd | wc -l
          cat /proc/sys/fs/file-nr
   Fix: Find leak source (lsof -p <pid>), restart process
        Increase limits: ulimit -n, LimitNOFILE=

3. TIME_WAIT accumulation:
   Symptom: New connections slow/failing
   Check: ss -s | grep TIME-WAIT
   Fix: sysctl net.ipv4.tcp_tw_reuse=1
        sysctl net.ipv4.tcp_fin_timeout=15
        Connection pooling in application

4. Zombie process accumulation:
   Symptom: Process table full, can't fork
   Check: ps aux | awk '$8 ~ /Z/'
   Fix: Kill parent process, fix parent to properly wait()

5. Log disk full:
   Symptom: Services crash, can't write
   Check: df -h /var/log
   Fix: Emergency cleanup (see script above)
        Set up log rotation, monitoring
` + "```" + ``,
					CodeExamples: `# Incident response scripts

# 1. First-responder diagnostic script
#!/bin/bash
echo "============================================"
echo "INCIDENT DIAGNOSTIC REPORT"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "============================================"
echo ""

echo "=== QUICK STATUS ==="
uptime
echo ""

echo "=== RECENT KERNEL MESSAGES ==="
dmesg -T 2>/dev/null | tail -10
echo ""

echo "=== SYSTEM OVERVIEW ==="
vmstat 1 3
echo ""

echo "=== MEMORY ==="
free -h
echo ""

echo "=== DISK ==="
df -h | grep -v tmpfs | grep -v devtmpfs
echo ""
echo "Inode usage:"
df -i | awk '$5+0 > 80 {print $0}'
echo ""

echo "=== FAILED SERVICES ==="
systemctl --failed --no-legend 2>/dev/null || echo "None"
echo ""

echo "=== TOP CPU ==="
ps -eo pid,user,%cpu,%mem,comm --sort=-%cpu | head -6
echo ""

echo "=== TOP MEMORY ==="
ps -eo pid,user,%cpu,%mem,rss,comm --sort=-rss | head -6
echo ""

echo "=== NETWORK ==="
ss -s
echo ""
echo "Connections by state:"
ss -tan | awk 'NR>1 {print $1}' | sort | uniq -c | sort -rn
echo ""

echo "=== RECENT AUTH FAILURES ==="
journalctl -t sshd -p warning --since "1 hour ago" --no-pager 2>/dev/null | tail -5
echo ""

echo "=== OOM EVENTS ==="
dmesg 2>/dev/null | grep -i "out of memory" | tail -3
journalctl -k --since "1 hour ago" 2>/dev/null | grep -i oom | tail -3
echo ""

echo "============================================"
echo "Report complete"

# 2. Automated incident reporter
#!/bin/bash
INCIDENT_DIR="/var/log/incidents"
DATE=$(date +%Y%m%d-%H%M%S)
REPORT="$INCIDENT_DIR/incident-$DATE"

mkdir -p "$REPORT"

# Capture everything
uptime > "$REPORT/uptime.txt"
free -h > "$REPORT/memory.txt"
df -h > "$REPORT/disk.txt"
ps auxf > "$REPORT/processes.txt"
ss -tnp > "$REPORT/connections.txt"
ss -s > "$REPORT/socket-stats.txt"
ip addr > "$REPORT/network.txt"
ip route > "$REPORT/routes.txt"
dmesg -T > "$REPORT/dmesg.txt" 2>/dev/null
journalctl --since "1 hour ago" --no-pager > "$REPORT/journal.txt" 2>/dev/null
systemctl --failed --no-pager > "$REPORT/failed-services.txt" 2>/dev/null
vmstat 1 10 > "$REPORT/vmstat.txt" 2>/dev/null &
iostat -xz 1 10 > "$REPORT/iostat.txt" 2>/dev/null &

wait

# Create tarball
tar czf "${REPORT}.tar.gz" -C "$INCIDENT_DIR" "$(basename "$REPORT")"
rm -rf "$REPORT"

echo "Incident data captured: ${REPORT}.tar.gz"

# 3. Connection troubleshooter
#!/bin/bash
TARGET="${1:?Usage: $0 <host:port>}"
HOST=$(echo "$TARGET" | cut -d: -f1)
PORT=$(echo "$TARGET" | cut -d: -f2)

echo "=== Troubleshooting connection to $HOST:$PORT ==="

# DNS
echo ""
echo "--- DNS Resolution ---"
if dig +short "$HOST" A | head -3; then
    echo "DNS: OK"
else
    echo "DNS: FAILED - check /etc/resolv.conf"
fi

# Ping
echo ""
echo "--- ICMP Ping ---"
if ping -c 2 -W 2 "$HOST" > /dev/null 2>&1; then
    echo "Ping: OK"
else
    echo "Ping: FAILED (may be filtered)"
fi

# TCP connection
echo ""
echo "--- TCP Connection ---"
if nc -zw3 "$HOST" "$PORT" 2>/dev/null; then
    echo "TCP $PORT: OPEN"
else
    echo "TCP $PORT: CLOSED/FILTERED"
fi

# Route
echo ""
echo "--- Route ---"
ip route get "$(dig +short "$HOST" | head -1)" 2>/dev/null | head -1

# Local firewalling
echo ""
echo "--- Local Firewall ---"
iptables -L OUTPUT -n 2>/dev/null | grep -i "$PORT\|drop\|reject" | head -5
echo "(empty = no relevant rules)"`,
				},
			},
		},
	})
}
