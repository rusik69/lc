package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1829,
			Title:       "Linux Performance Analysis",
			Description: "Master Linux performance analysis: CPU, memory, I/O, and network profiling tools, benchmarking, tracing with perf, eBPF, and systematic performance methodology.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "CPU and Memory Performance",
					Content: `Performance analysis requires understanding system resources and using the right tools at the right time. The USE Method (Utilization, Saturation, Errors) provides a systematic approach.

**CPU Analysis:**
` + "```" + `
Key metrics:
  Utilization:  % of time CPU is busy
  Saturation:   Queue length (run queue)
  Errors:       CPU-related errors (rare)

Tools:
  # Quick overview
  uptime                          # Load averages (1, 5, 15 min)
  
  # Load average interpretation:
  # Load 1.0 on 1-CPU = 100% utilized
  # Load 2.0 on 1-CPU = overloaded (processes waiting)
  # Load 4.0 on 4-CPU = 100% utilized
  # Rule: load > num_CPUs = potentially overloaded
  
  # CPU utilization breakdown
  mpstat -P ALL 1                 # Per-CPU stats every 1 sec
  
  # Fields:
  # %usr:  user-space processes
  # %sys:  kernel
  # %iowait: waiting for I/O (misleading indicator)
  # %irq:  hardware interrupts
  # %soft: software interrupts
  # %steal: hypervisor stealing (VM)
  # %idle: idle

  vmstat 1                        # Virtual memory stats
  # Fields: r (run queue), b (blocked), us, sy, id, wa
  # r >> num_CPUs = CPU saturation
  
  # Per-process CPU usage
  pidstat 1                       # CPU per process
  pidstat -t 1                    # Per thread
  
  # CPU frequency and governor
  cat /proc/cpuinfo | grep "MHz"
  cpupower frequency-info
  cpupower frequency-set -g performance

  # Context switches
  vmstat 1                        # cs column
  pidstat -w 1                    # Per-process context switches
  # voluntary: process yields (I/O wait)
  # involuntary: preempted by scheduler
  
  # CPU cache and hardware events
  perf stat -a sleep 5            # Hardware counters
  # Instructions, cycles, IPC, cache misses, branch misses

/proc/stat:
  cpu  user nice system idle iowait irq softirq steal
  # Cumulative tick counts since boot
  # Calculate rates between two samples
` + "```" + `

**Memory Analysis:**
` + "```" + `
Key metrics:
  Utilization: % memory in use
  Saturation: swapping, OOM kills
  Errors: allocation failures

Tools:
  free -h                         # Memory overview
  # total    Total physical RAM
  # used     Used (includes buffers/cache)
  # free     Truly free (not being used at all)
  # shared   tmpfs and shared memory
  # buff/cache  Disk buffers + page cache
  # available  Estimated available for new apps
  
  # "available" is the key metric (not "free")
  # Linux uses free RAM for cache → free looks low but available is fine

  vmstat 1                        # Memory columns: swpd, free, buff, cache
  # si/so: swap in/out (should be 0 in normal operation)
  
  # Per-process memory
  ps aux --sort=-rss | head        # Sort by RSS
  pmap -x <pid>                    # Detailed memory map
  smem                             # Proportional set size (accurate per-process)
  
  /proc/meminfo:
    MemTotal, MemFree, MemAvailable
    Buffers, Cached, SwapCached
    SwapTotal, SwapFree
    Active, Inactive
    Dirty, Writeback
    Slab, SReclaimable, SUnreclaim
    Mapped, AnonPages
    HugePages_Total, HugePages_Free

  # NUMA memory
  numastat
  numactl --hardware
  # NUMA imbalance can cause performance issues
  # Memory allocated on remote NUMA node = slower access

Page cache:
  # Page cache uses free RAM to cache disk data
  # Speeds up reads enormously
  # Released automatically when applications need RAM
  
  # View page cache usage
  cat /proc/meminfo | grep -E "Cached|Buffers"
  
  # Drop page cache (testing only!)
  sync; echo 3 > /proc/sys/vm/drop_caches
  # 1 = page cache, 2 = dentries/inodes, 3 = both

Swap analysis:
  swapon --show                   # Swap devices
  cat /proc/swaps
  vmstat 1                        # si/so columns
  
  # Per-process swap usage
  for pid in /proc/[0-9]*; do
    swap=$(awk '/VmSwap/ {print $2}' "$pid/status" 2>/dev/null)
    name=$(cat "$pid/comm" 2>/dev/null)
    [ "${swap:-0}" -gt 0 ] && echo "$swap KB $name (${pid##*/})"
  done | sort -rn | head
  
  # Swappiness (0-200, default 60)
  cat /proc/sys/vm/swappiness
  # Lower = prefer evicting file cache over anonymous pages
  # Server recommendation: 10
  sysctl vm.swappiness=10
` + "```" + ``,
					CodeExamples: `# Performance analysis scripts

# 1. System performance snapshot
#!/bin/bash
echo "=== Performance Snapshot $(date) ==="

echo ""
echo "--- CPU ---"
echo "Load averages: $(uptime | awk -F'load average:' '{print $2}')"
echo "CPU count: $(nproc)"
echo ""
mpstat 1 1 | tail -1 | awk '{printf "User: %s%% System: %s%% IOWait: %s%% Idle: %s%%\n", $3, $5, $6, $12}'

echo ""
echo "--- Memory ---"
free -h | awk '
/Mem:/ { printf "Total: %s  Used: %s  Available: %s\n", $2, $3, $7 }
/Swap:/ { printf "Swap Total: %s  Swap Used: %s\n", $2, $3 }
'

echo ""
echo "--- Top CPU Processes ---"
ps -eo pid,user,%cpu,%mem,comm --sort=-%cpu | head -6

echo ""
echo "--- Top Memory Processes ---"
ps -eo pid,user,%cpu,%mem,rss,comm --sort=-rss | head -6

echo ""
echo "--- Disk I/O ---"
iostat -xz 1 1 2>/dev/null | awk '
/^[a-z]/ && NR > 3 { printf "%-10s r/s: %6s  w/s: %6s  await: %6s  %%util: %s%%\n", $1, $4, $5, $10, $NF }
'

echo ""
echo "--- Network ---"
ss -s | head -4

echo ""
echo "--- Open Files ---"
echo "Kernel max: $(cat /proc/sys/fs/file-max)"
echo "Currently open: $(cat /proc/sys/fs/file-nr | awk '{print $1}')"

# 2. Memory leak detector
#!/bin/bash
PID=$1
INTERVAL=${2:-60}
ITERATIONS=${3:-60}

if [ -z "$PID" ]; then
    echo "Usage: $0 <PID> [interval_sec] [iterations]"
    exit 1
fi

echo "Monitoring PID $PID every ${INTERVAL}s for $ITERATIONS iterations"
echo "Time,RSS_KB,VSZ_KB,Swap_KB"

for i in $(seq 1 "$ITERATIONS"); do
    if [ ! -d "/proc/$PID" ]; then
        echo "Process $PID no longer exists"
        exit 1
    fi
    
    RSS=$(awk '/VmRSS/ {print $2}' "/proc/$PID/status" 2>/dev/null)
    VSZ=$(awk '/VmSize/ {print $2}' "/proc/$PID/status" 2>/dev/null)
    SWAP=$(awk '/VmSwap/ {print $2}' "/proc/$PID/status" 2>/dev/null)
    
    echo "$(date +%H:%M:%S),${RSS:-0},${VSZ:-0},${SWAP:-0}"
    sleep "$INTERVAL"
done

# 3. CPU saturation check
#!/bin/bash
CPUS=$(nproc)
THRESHOLD=$((CPUS * 2))

LOAD1=$(uptime | awk -F'load average:' '{print $2}' | awk -F, '{print $1}' | xargs)
LOAD_INT=$(echo "$LOAD1" | cut -d. -f1)

if [ "${LOAD_INT:-0}" -gt "$THRESHOLD" ]; then
    echo "CRITICAL: Load ($LOAD1) exceeds ${THRESHOLD} (${CPUS} CPUs x 2)"
    echo ""
    echo "Top CPU consumers:"
    ps -eo pid,user,%cpu,comm --sort=-%cpu | head -10
    echo ""
    echo "Run queue depth:"
    vmstat 1 3 | tail -1 | awk '{print "run queue:", $1, "blocked:", $2}'
elif [ "${LOAD_INT:-0}" -gt "$CPUS" ]; then
    echo "WARNING: Load ($LOAD1) exceeds CPU count ($CPUS)"
else
    echo "OK: Load ($LOAD1) within normal range for $CPUS CPUs"
fi`,
				},
				{
					Title: "I/O and Network Performance",
					Content: `Disk I/O and network performance are often the bottlenecks in production systems. Understanding how to measure and optimize them is critical.

**Disk I/O Analysis:**
` + "```" + `
Key metrics:
  IOPS:        I/O operations per second
  Throughput:  MB/s read or written
  Latency:     Time per I/O operation (ms)
  Queue depth: Number of outstanding I/Os
  Utilization: % time device is busy

Tools:
  iostat -xz 1                    # Extended disk stats
  # Key columns:
  # r/s, w/s:    Read/write IOPS
  # rkB/s, wkB/s: Read/write throughput
  # await:        Average I/O latency (ms)
  # r_await, w_await: Read/write latency separately
  # avgqu-sz:    Average queue depth
  # %util:       Device utilization
  #   %util 100% doesn't mean saturated for SSDs
  #   (SSDs handle parallel I/O; %util measures busy time)
  
  iotop                           # Per-process I/O (interactive)
  iotop -b -o -d 5               # Batch mode, only active
  
  pidstat -d 1                    # Per-process disk stats
  # kB_rd/s, kB_wr/s: Read/write throughput per process
  
  # Block device stats
  cat /proc/diskstats
  cat /sys/block/sda/stat
  
  # Queue depth and scheduler
  cat /sys/block/sda/queue/scheduler
  cat /sys/block/sda/queue/nr_requests
  
  # I/O scheduler tuning
  echo "mq-deadline" > /sys/block/sda/queue/scheduler  # SSD
  echo "bfq" > /sys/block/sda/queue/scheduler          # HDD

Benchmarking:
  # fio - flexible I/O tester
  # Sequential read
  fio --name=seqread --rw=read --bs=1M --size=1G \
      --numjobs=1 --runtime=30 --direct=1
  
  # Random read (4K, common for databases)
  fio --name=randread --rw=randread --bs=4k --size=1G \
      --numjobs=4 --iodepth=32 --runtime=30 --direct=1
  
  # Mixed workload
  fio --name=mixed --rw=randrw --rwmixread=70 --bs=4k \
      --size=1G --numjobs=4 --iodepth=16 --runtime=30
  
  # dd (simple but less accurate)
  dd if=/dev/zero of=/tmp/test bs=1M count=1024 oflag=direct
  dd if=/tmp/test of=/dev/null bs=1M iflag=direct
` + "```" + `

**Network Performance:**
` + "```" + `
Key metrics:
  Bandwidth:    Data transfer rate
  Latency:      Round-trip time
  Packet loss:  % packets dropped
  Retransmits:  TCP retransmissions
  Connections:  Active/new connections per second

Tools:
  # Interface statistics
  ip -s link show eth0            # Byte/packet counts, errors, drops
  
  # Real-time bandwidth
  iftop -i eth0                   # Interactive bandwidth monitor
  nload eth0                      # Simple bandwidth graph
  vnstat -l                       # Live traffic stats
  
  # Per-process network
  nethogs eth0                    # Per-process bandwidth
  ss -tip                         # TCP connections with info
  
  # TCP statistics
  ss -s                           # Socket summary
  netstat -s                      # Protocol statistics
  nstat                           # Network statistics
  
  # Key TCP metrics:
  cat /proc/net/tcp               # Raw TCP connection table
  ss -ti                          # Per-connection details
  # Look for: retransmits, rto, cwnd, ssthresh

  # Packet drops and errors
  ip -s link show eth0 | grep -E "drops|errors"
  ethtool -S eth0 | grep -E "drop|error|miss"
  cat /proc/net/softnet_stat      # Per-CPU packet processing
  # Column 2: dropped (backlog overflow)
  # Column 3: time_squeeze (CPU too busy)

  # Connection tracking
  conntrack -C                    # Current count
  sysctl net.netfilter.nf_conntrack_count
  sysctl net.netfilter.nf_conntrack_max
  # If count approaches max, connections get dropped!

Benchmarking:
  # iperf3 (bandwidth test)
  iperf3 -s                       # Server
  iperf3 -c server_ip -t 30       # Client: 30s TCP test
  iperf3 -c server_ip -u -b 10G  # UDP test
  iperf3 -c server_ip -P 4       # 4 parallel streams
  
  # Latency test
  ping -c 100 server_ip
  # Look at: avg, stddev, % loss
  
  # HTTP benchmarking
  ab -n 10000 -c 100 http://server/  # Apache bench
  wrk -t4 -c100 -d30s http://server/ # wrk (better)
` + "```" + `

**perf - Linux Profiler:**
` + "```" + `
perf is the standard Linux profiler for CPU and event analysis.

  # Record CPU samples
  perf record -g -p <pid> sleep 30
  perf report                     # Interactive report
  
  # System-wide recording
  perf record -ag sleep 10
  
  # Count events
  perf stat -p <pid> sleep 5
  perf stat -a sleep 5            # System-wide
  
  # perf stat output:
  # instructions, cycles, IPC
  # cache-references, cache-misses
  # branch-instructions, branch-misses
  # context-switches, cpu-migrations
  
  # Flame graphs (visualization)
  perf record -g -p <pid> sleep 30
  perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
  
  # perf trace (strace alternative, lower overhead)
  perf trace -p <pid>
  perf trace -e open,read,write -p <pid>
  
  # Hardware events
  perf stat -e cache-misses,cache-references,instructions,cycles \
    -p <pid> sleep 10

eBPF tools (bcc/bpftrace):
  # High-level performance tools
  execsnoop       # Trace new processes
  opensnoop       # Trace file opens
  biolatency      # Block I/O latency histogram
  tcplife         # TCP connection lifecycle
  tcpretrans      # TCP retransmissions
  runqlat         # Run queue latency
  cachetop        # Page cache hit/miss by process
  funccount       # Count function calls
  
  # bpftrace one-liners
  bpftrace -e 'tracepoint:syscalls:sys_enter_open { printf("%s %s\n", comm, str(args->filename)); }'
  bpftrace -e 'tracepoint:block:block_rq_complete { @[args->rwbs] = hist(args->nr_sector); }'
` + "```" + `

**Systematic Performance Methodology:**
` + "```" + `
USE Method (for resources):
  For each resource (CPU, memory, disk, network):
    1. Utilization: % time busy or % capacity used
    2. Saturation: extra work queued (queue length)
    3. Errors: error events

  CPU:
    U: mpstat (%usr + %sys)
    S: vmstat (r column), runqlat
    E: perf stat (machine check exceptions - rare)
  
  Memory:
    U: free (used/total)
    S: vmstat (si/so - swapping), dmesg | grep oom
    E: dmesg (allocation failures)
  
  Disk:
    U: iostat (%util)
    S: iostat (avgqu-sz), await
    E: /sys/block/*/stat, smartctl
  
  Network:
    U: ip -s link, sar -n DEV
    S: ifconfig (overruns), /proc/net/softnet_stat
    E: ip -s link (errors), ethtool -S

RED Method (for services):
  Rate:     Requests per second
  Errors:   Failed requests per second  
  Duration: Request latency (p50, p95, p99)
` + "```" + ``,
					CodeExamples: `# Performance analysis tools and scripts

# 1. Comprehensive performance report
#!/bin/bash
DURATION=5

echo "=== System Performance Report ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | awk '/Mem:/ {print $2}')"
echo ""

echo "=== USE Method Analysis ==="
echo ""
echo "--- CPU ---"
echo "Utilization:"
mpstat 1 "$DURATION" | tail -1 | awk '{printf "  User: %s%%  System: %s%%  IOWait: %s%%  Idle: %s%%\n", $3, $5, $6, $12}'
echo "Saturation:"
echo "  Run queue: $(vmstat 1 2 | tail -1 | awk '{print $1}')"
echo "  Load avg: $(uptime | awk -F'load average:' '{print $2}')"

echo ""
echo "--- Memory ---"
echo "Utilization:"
free | awk '/Mem:/ {printf "  Used: %.1f%%  Available: %.1f%%\n", ($3/$2)*100, ($7/$2)*100}'
echo "Saturation:"
echo "  Swap used: $(free -h | awk '/Swap:/ {print $3}')"
SWAP_IO=$(vmstat 1 "$DURATION" | tail -1 | awk '{print "si=" $7 " so=" $8}')
echo "  Swap I/O: $SWAP_IO"

echo ""
echo "--- Disk ---"
echo "Utilization and Saturation:"
iostat -xz 1 "$DURATION" 2>/dev/null | awk '
/^[a-z]/ && !/Linux/ && !/^avg/ {
    printf "  %-10s util: %5s%%  avg-queue: %s  await: %sms\n", $1, $NF, $9, $10
}'

echo ""
echo "--- Network ---"
echo "Errors:"
for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep -v lo); do
    ERRORS=$(ip -s link show "$iface" 2>/dev/null | awk '/errors/ {sum += $3} END {print sum+0}')
    DROPS=$(ip -s link show "$iface" 2>/dev/null | awk '/dropped/ {sum += $4} END {print sum+0}')
    echo "  $iface: errors=$ERRORS drops=$DROPS"
done

# 2. I/O latency histogram (simple version)
#!/bin/bash
echo "=== I/O Latency Distribution ==="
iostat -x 1 60 | awk '
/^[a-z]/ && NR > 3 {
    dev = $1
    await = $10 + 0
    if (await < 1) bucket["<1ms"]++
    else if (await < 4) bucket["1-4ms"]++
    else if (await < 8) bucket["4-8ms"]++
    else if (await < 16) bucket["8-16ms"]++
    else if (await < 32) bucket["16-32ms"]++
    else if (await < 64) bucket["32-64ms"]++
    else bucket[">64ms"]++
    total++
}
END {
    if (total > 0) {
        for (b in bucket) {
            bar = ""
            pct = (bucket[b] / total) * 100
            for (i = 0; i < pct; i++) bar = bar "#"
            printf "%-10s %4d (%5.1f%%) %s\n", b, bucket[b], pct, bar
        }
    }
}'

# 3. Network connection analyzer
#!/bin/bash
echo "=== Network Connection Analysis ==="

echo ""
echo "--- Connection States ---"
ss -tan | awk 'NR>1 {print $1}' | sort | uniq -c | sort -rn

echo ""
echo "--- Top Remote IPs (by connection count) ---"
ss -tn state established | awk 'NR>1 {print $5}' | \
    rev | cut -d: -f2- | rev | sort | uniq -c | sort -rn | head -10

echo ""
echo "--- Listening Services ---"
ss -tlnp | awk 'NR>1 {printf "%-6s %-25s %s\n", $1, $4, $7}'

echo ""
echo "--- TCP Retransmit Stats ---"
nstat -az TcpRetransSegs TcpInSegs TcpOutSegs 2>/dev/null | \
    awk 'NR>1 {printf "%-30s %s\n", $1, $2}'

# 4. sysctl performance tuning
#!/bin/bash
# Apply performance tuning sysctls
cat > /etc/sysctl.d/99-performance.conf << 'SYSCTL'
# Network tuning
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 262144
net.core.wmem_default = 262144
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 5

# Memory tuning
vm.swappiness = 10
vm.dirty_ratio = 20
vm.dirty_background_ratio = 5
vm.overcommit_memory = 0
vm.min_free_kbytes = 65536

# File descriptors
fs.file-max = 2097152
fs.nr_open = 2097152

# Connection tracking
net.netfilter.nf_conntrack_max = 1048576
SYSCTL

sysctl -p /etc/sysctl.d/99-performance.conf`,
				},
			},
		},
	})
}
