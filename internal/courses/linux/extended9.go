package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1831,
			Title:       "Linux Kernel and System Internals",
			Description: "Understand Linux kernel internals: kernel modules, procfs and sysfs, kernel parameters, system calls, and kernel compilation.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "Kernel Modules and Parameters",
					Content: `The Linux kernel is modular — functionality can be loaded and unloaded at runtime through kernel modules. Understanding kernel parameters and module management is essential for system tuning.

**Kernel Modules:**
` + "```" + `
Module management:
  lsmod                           # List loaded modules
  modinfo <module>                # Module information
  modprobe <module>               # Load module (resolves dependencies)
  modprobe -r <module>            # Remove module
  insmod /path/to/module.ko       # Load module (no dependency resolution)
  rmmod <module>                  # Remove module
  
  # Module dependencies
  depmod -a                       # Rebuild module dependency database
  modprobe --show-depends ext4    # Show what would be loaded

Module location:
  /lib/modules/$(uname -r)/       # Module files (.ko, .ko.xz, .ko.zst)
  /lib/modules/$(uname -r)/modules.dep  # Dependency database

Persistent module loading:
  # Load at boot
  echo "br_netfilter" > /etc/modules-load.d/bridge.conf
  
  # Blacklist module (prevent loading)
  echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
  echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
  
  # Module options
  echo "options snd-hda-intel model=generic" > /etc/modprobe.d/sound.conf
  
  # View current module options
  systool -v -m <module>
  cat /sys/module/<module>/parameters/*
` + "```" + `

**Kernel Parameters (sysctl):**
` + "```" + `
Viewing parameters:
  sysctl -a                       # All parameters
  sysctl net.ipv4.ip_forward      # Specific parameter
  cat /proc/sys/net/ipv4/ip_forward  # Same via procfs

Setting parameters:
  # Temporary (until reboot)
  sysctl -w net.ipv4.ip_forward=1
  echo 1 > /proc/sys/net/ipv4/ip_forward
  
  # Permanent
  echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.d/99-custom.conf
  sysctl -p /etc/sysctl.d/99-custom.conf

Important parameters by category:

  Networking:
    net.ipv4.ip_forward = 1              # Enable IP forwarding
    net.ipv4.conf.all.rp_filter = 1      # Reverse path filtering
    net.ipv4.conf.all.accept_redirects = 0  # Ignore ICMP redirects
    net.ipv4.conf.all.send_redirects = 0
    net.ipv4.icmp_echo_ignore_broadcasts = 1
    net.ipv4.tcp_syncookies = 1          # SYN flood protection
    net.ipv4.tcp_max_syn_backlog = 65535
    net.core.somaxconn = 65535           # Listen backlog
    net.core.netdev_max_backlog = 65535
    net.ipv4.tcp_fin_timeout = 15        # FIN timeout (default 60)
    net.ipv4.tcp_tw_reuse = 1            # Reuse TIME_WAIT sockets
    net.ipv4.tcp_keepalive_time = 600
    
  Memory:
    vm.swappiness = 10                   # Preference for swapping (0-200)
    vm.dirty_ratio = 20                  # % memory for dirty pages before sync
    vm.dirty_background_ratio = 5        # Background writeback threshold
    vm.overcommit_memory = 0             # 0=heuristic, 1=always, 2=never
    vm.overcommit_ratio = 80             # With overcommit_memory=2
    vm.min_free_kbytes = 65536           # Keep free for emergencies
    vm.vfs_cache_pressure = 100          # Inode/dentry cache reclaim
    
  Filesystem:
    fs.file-max = 2097152                # System-wide file descriptor limit
    fs.inotify.max_user_watches = 524288 # inotify watches
    fs.aio-max-nr = 1048576              # Async I/O events

  Kernel:
    kernel.pid_max = 4194304             # Max PID value
    kernel.threads-max = 256000          # Max threads
    kernel.panic = 10                    # Reboot 10s after panic
    kernel.sysrq = 1                     # Enable SysRq keys
    kernel.core_pattern = /tmp/core-%e-%p # Core dump location
` + "```" + `

**procfs (/proc):**
` + "```" + `
Key files:
  /proc/cpuinfo       CPU information
  /proc/meminfo       Memory statistics
  /proc/vmstat        Virtual memory statistics
  /proc/diskstats     Disk I/O statistics
  /proc/net/          Network statistics
  /proc/interrupts    Hardware interrupts
  /proc/loadavg       Load averages
  /proc/uptime        Uptime in seconds
  /proc/version       Kernel version
  /proc/cmdline       Kernel boot command line
  /proc/mounts        Mounted filesystems
  /proc/filesystems   Supported filesystem types
  /proc/partitions    Partition information
  /proc/swaps         Swap areas
  /proc/sys/          Tunable kernel parameters

Per-process:
  /proc/<pid>/status    Process status
  /proc/<pid>/stat      Raw process statistics
  /proc/<pid>/cmdline   Command line
  /proc/<pid>/environ   Environment variables (null-separated)
  /proc/<pid>/fd/       Open file descriptors
  /proc/<pid>/maps      Memory mappings
  /proc/<pid>/smaps     Detailed memory mappings
  /proc/<pid>/limits    Resource limits
  /proc/<pid>/cgroup    Cgroup membership
  /proc/<pid>/io        I/O statistics
  /proc/<pid>/net/      Network info (per-namespace)
  /proc/<pid>/ns/       Namespaces
  /proc/<pid>/oom_score OOM killer score
  /proc/<pid>/oom_score_adj  OOM score adjustment
` + "```" + `

**sysfs (/sys):**
` + "```" + `
Structure:
  /sys/block/         Block devices
  /sys/bus/           Bus types (pci, usb, etc.)
  /sys/class/         Device classes (net, block, tty)
  /sys/devices/       Device hierarchy
  /sys/firmware/      Firmware interfaces (ACPI, EFI)
  /sys/fs/            Filesystem-specific info
  /sys/kernel/        Kernel info
  /sys/module/        Loaded modules
  /sys/power/         Power management

Useful examples:
  # CPU info
  ls /sys/devices/system/cpu/cpu0/
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  
  # Block device info
  cat /sys/block/sda/queue/scheduler
  cat /sys/block/sda/queue/rotational    # 0=SSD, 1=HDD
  cat /sys/block/sda/queue/nr_requests
  
  # Network
  cat /sys/class/net/eth0/speed
  cat /sys/class/net/eth0/operstate
  cat /sys/class/net/eth0/address
  
  # Power management
  cat /sys/power/state                   # Available states
  echo mem > /sys/power/state            # Suspend to RAM
  
  # Module parameters
  cat /sys/module/tcp_cubic/parameters/beta
` + "```" + ``,
					CodeExamples: `# Kernel management scripts

# 1. System information collector
#!/bin/bash
echo "=== Kernel Information ==="
echo "Version: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Boot parameters: $(cat /proc/cmdline)"
echo ""

echo "=== CPU ==="
CORES=$(nproc)
MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
echo "Model: $MODEL"
echo "Cores: $CORES"
echo "Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
echo ""

echo "=== Memory ==="
awk '/MemTotal|MemAvailable|SwapTotal|HugePages_Total/ {printf "%-20s %s %s\n", $1, $2, $3}' /proc/meminfo

echo ""
echo "=== Block Devices ==="
for DEV in /sys/block/sd* /sys/block/nvme* /sys/block/vd* 2>/dev/null; do
    [ -d "$DEV" ] || continue
    NAME=$(basename "$DEV")
    SIZE=$(cat "$DEV/size" 2>/dev/null)
    SIZE_GB=$((SIZE * 512 / 1024 / 1024 / 1024))
    SCHED=$(cat "$DEV/queue/scheduler" 2>/dev/null)
    ROT=$(cat "$DEV/queue/rotational" 2>/dev/null)
    TYPE="SSD"
    [ "$ROT" = "1" ] && TYPE="HDD"
    echo "$NAME: ${SIZE_GB}GB $TYPE scheduler=[$SCHED]"
done

echo ""
echo "=== Loaded Modules (top 10 by size) ==="
lsmod | sort -k2 -rn | head -10

echo ""
echo "=== Key Kernel Parameters ==="
for PARAM in net.ipv4.ip_forward vm.swappiness vm.overcommit_memory \
             fs.file-max net.core.somaxconn kernel.pid_max; do
    VALUE=$(sysctl -n "$PARAM" 2>/dev/null)
    printf "%-40s %s\n" "$PARAM" "$VALUE"
done

# 2. Module dependency checker
#!/bin/bash
MODULE="${1:?Usage: $0 <module-name>}"

echo "=== Module: $MODULE ==="
modinfo "$MODULE" 2>/dev/null | grep -E "^(description|author|license|depends|vermagic):"

echo ""
echo "=== Dependencies ==="
modprobe --show-depends "$MODULE" 2>/dev/null

echo ""
echo "=== Current Status ==="
if lsmod | grep -q "^${MODULE}[[:space:]]"; then
    echo "LOADED"
    echo "Used by: $(lsmod | grep "^${MODULE}" | awk '{print $4}')"
    echo ""
    echo "Parameters:"
    for P in /sys/module/"$MODULE"/parameters/*; do
        [ -f "$P" ] && echo "  $(basename "$P") = $(cat "$P" 2>/dev/null)"
    done
else
    echo "NOT LOADED"
fi

# 3. Kernel parameter hardening
#!/bin/bash
cat > /etc/sysctl.d/99-hardening.conf << 'EOF'
# Network security
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Kernel hardening
kernel.randomize_va_space = 2
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
kernel.yama.ptrace_scope = 2
kernel.unprivileged_bpf_disabled = 1
net.core.bpf_jit_harden = 2
kernel.perf_event_paranoid = 3

# Filesystem
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.suid_dumpable = 0
EOF

sysctl -p /etc/sysctl.d/99-hardening.conf
echo "Hardening parameters applied"

# 4. OOM score management
#!/bin/bash
echo "=== OOM Scores (highest risk first) ==="
for pid in /proc/[0-9]*; do
    [ -f "$pid/oom_score" ] || continue
    score=$(cat "$pid/oom_score" 2>/dev/null)
    name=$(cat "$pid/comm" 2>/dev/null)
    adj=$(cat "$pid/oom_score_adj" 2>/dev/null)
    [ "${score:-0}" -gt 0 ] && echo "$score $adj ${pid##*/} $name"
done | sort -rn | head -20 | \
    awk '{printf "Score: %-5s Adj: %-5s PID: %-7s %s\n", $1, $2, $3, $4}'

# Protect critical process from OOM killer
# echo -1000 > /proc/<pid>/oom_score_adj`,
				},
				{
					Title: "System Calls and Tracing",
					Content: `System calls are the interface between user space and the kernel. Understanding and tracing them is essential for debugging and performance analysis.

**System Calls Overview:**
` + "```" + `
Common system calls:
  File I/O:
    open/openat     Open a file
    read            Read from file descriptor
    write           Write to file descriptor
    close           Close file descriptor
    lseek           Seek in file
    stat/fstat      File information
    unlink          Delete file
    rename          Rename file
    mkdir/rmdir     Create/remove directory
    
  Process:
    fork/clone      Create child process
    execve          Execute program
    exit_group      Terminate process
    wait4           Wait for child
    getpid/getppid  Get process/parent ID
    kill            Send signal
    
  Memory:
    mmap/munmap     Map/unmap memory
    brk             Change data segment size
    mprotect        Set memory protection
    
  Network:
    socket          Create socket
    bind            Bind to address
    listen          Listen for connections
    accept          Accept connection
    connect         Connect to server
    send/recv       Data transfer
    
  System:
    ioctl           Device control
    epoll_wait      I/O event notification
    futex           Fast userspace mutex
    clock_gettime   Get time
` + "```" + `

**strace (System Call Tracer):**
` + "```" + `
Basic usage:
  strace command                   # Trace command
  strace -p <pid>                  # Attach to running process
  strace -f command                # Follow forks (child processes)
  strace -ff -o output command     # Separate file per child
  
  # Common options
  strace -c command                # Summary statistics only
  strace -e trace=open,read,write command  # Filter syscalls
  strace -e trace=network command  # Only network syscalls
  strace -e trace=file command     # Only file syscalls
  strace -e trace=process command  # Only process syscalls
  strace -e trace=memory command   # Only memory syscalls
  
  strace -t command                # Timestamp per line
  strace -tt command               # Microsecond timestamps
  strace -T command                # Time spent in each syscall
  strace -s 256 command            # Show more string data (default: 32)
  strace -y command                # Show file paths for fd numbers
  strace -yy command               # Show socket details too

Practical examples:
  # Debug permission denied
  strace -e trace=open,openat,access command 2>&1 | grep EACCES
  
  # Find which config files are read
  strace -e trace=openat -y command 2>&1 | grep -v ENOENT
  
  # Debug slow startup
  strace -T -e trace=openat,connect,stat command 2>&1 | sort -t= -k2 -rn | head
  
  # Find which DNS servers are contacted
  strace -e trace=connect command 2>&1 | grep ":53"
  
  # Debug hanging process
  strace -p <pid> -e trace=futex,nanosleep,poll,epoll_wait
  
  # Count syscalls
  strace -c ls /tmp
  # Shows: % time, seconds, usecs/call, calls, errors, syscall
` + "```" + `

**ltrace (Library Call Tracer):**
` + "```" + `
  ltrace command                   # Trace library calls
  ltrace -p <pid>                  # Attach to process
  ltrace -c command                # Summary
  ltrace -e malloc+free command    # Filter functions
  ltrace -s 200 command            # Show more string data
  
  # Useful for:
  # - Debugging dynamic library issues
  # - Understanding library usage patterns
  # - Finding which library functions are slow
` + "```" + `

**Advanced Tracing:**
` + "```" + `
ftrace (kernel function tracer):
  # Available tracers
  cat /sys/kernel/debug/tracing/available_tracers
  
  # Trace a specific function
  echo function > /sys/kernel/debug/tracing/current_tracer
  echo do_sys_openat2 > /sys/kernel/debug/tracing/set_ftrace_filter
  echo 1 > /sys/kernel/debug/tracing/tracing_on
  cat /sys/kernel/debug/tracing/trace_pipe
  # Ctrl+C to stop
  echo 0 > /sys/kernel/debug/tracing/tracing_on
  echo nop > /sys/kernel/debug/tracing/current_tracer

perf trace (modern strace alternative):
  perf trace command               # Lower overhead than strace
  perf trace -p <pid>
  perf trace -e write command      # Filter
  perf trace --duration 10 -p <pid>  # Trace for 10 seconds
  
  # Advantages over strace:
  # - Much lower overhead (ring buffer vs ptrace)
  # - Can trace scheduler events
  # - System-wide tracing
  # - Better for production use

bpftrace:
  # Trace file opens
  bpftrace -e 'tracepoint:syscalls:sys_enter_openat {
    printf("%s %s\n", comm, str(args->filename));
  }'
  
  # Histogram of read sizes
  bpftrace -e 'tracepoint:syscalls:sys_exit_read /args->ret > 0/ {
    @size = hist(args->ret);
  }'
  
  # Track process execution
  bpftrace -e 'tracepoint:syscalls:sys_enter_execve {
    printf("%d %s %s\n", pid, comm, str(args->filename));
  }'
` + "```" + ``,
					CodeExamples: `# Tracing and debugging scripts

# 1. Process debugging toolkit
#!/bin/bash
PID="${1:?Usage: $0 <pid>}"

if [ ! -d "/proc/$PID" ]; then
    echo "Process $PID does not exist"
    exit 1
fi

echo "=== Process $PID Debug Info ==="
echo ""
echo "--- Basic Info ---"
cat "/proc/$PID/status" | grep -E "^(Name|State|Pid|PPid|Uid|Gid|Threads|VmSize|VmRSS|VmSwap):"

echo ""
echo "--- Command Line ---"
tr '\0' ' ' < "/proc/$PID/cmdline"
echo ""

echo ""
echo "--- Open File Descriptors ---"
ls -la "/proc/$PID/fd/" 2>/dev/null | head -20
FD_COUNT=$(ls "/proc/$PID/fd/" 2>/dev/null | wc -l)
echo "Total FDs: $FD_COUNT"

echo ""
echo "--- Open Files (by type) ---"
ls -la "/proc/$PID/fd/" 2>/dev/null | awk '{print $NF}' | \
    awk -F/ '{
        if (/socket/) type="socket"
        else if (/pipe/) type="pipe"
        else if (/anon_inode/) type="eventfd/epoll"
        else type="file"
        count[type]++
    } END {
        for (t in count) printf "  %-15s %d\n", t, count[t]
    }'

echo ""
echo "--- Network Connections ---"
ss -tnp 2>/dev/null | grep "pid=$PID" | head -10

echo ""
echo "--- Memory Maps (summary) ---"
awk '{
    size = (strtonum("0x" $2) - strtonum("0x" $1)) / 1024
    if ($NF ~ /\.so/) type = "shared_lib"
    else if ($NF == "[heap]") type = "heap"
    else if ($NF == "[stack]") type = "stack"
    else if ($NF == "") type = "anonymous"
    else type = "other"
    total[type] += size
} END {
    for (t in total) printf "  %-15s %.1f MB\n", t, total[t]/1024
}' "/proc/$PID/maps" 2>/dev/null

echo ""
echo "--- Resource Limits ---"
cat "/proc/$PID/limits" 2>/dev/null | head -15

# 2. Syscall profiler wrapper
#!/bin/bash
CMD="$@"
echo "Profiling: $CMD"
echo ""

strace -c -f $CMD 2> /tmp/strace_summary.txt
echo ""
echo "=== System Call Summary ==="
cat /tmp/strace_summary.txt
rm -f /tmp/strace_summary.txt

# 3. File access audit
#!/bin/bash
# Track all file accesses by a command
CMD="$@"
LOGFILE="/tmp/file-audit-$$.log"

echo "Auditing file access for: $CMD"
strace -f -e trace=openat,open,stat,access -y -o "$LOGFILE" $CMD

echo ""
echo "=== Files Accessed ==="
grep -oP '(?<=openat\(AT_FDCWD, ")[^"]+' "$LOGFILE" | sort -u

echo ""
echo "=== Files Not Found ==="
grep "ENOENT" "$LOGFILE" | grep -oP '(?<=")[^"]+(?=")' | sort -u

echo ""
echo "=== Permission Denied ==="
grep "EACCES" "$LOGFILE" | grep -oP '(?<=")[^"]+(?=")' | sort -u

rm -f "$LOGFILE"`,
				},
			},
		},
	})
}
