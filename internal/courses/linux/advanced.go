package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1810,
			Title:       "Advanced Process Management",
			Description: "Deep dive into signals, process groups, daemons, cron jobs, systemd services, and process monitoring.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Signals & Process Groups",
					Content: `Signals are software interrupts sent to processes. Understanding signals is crucial for process control.

**Common Signals:**
- SIGTERM (15): Terminate gracefully
- SIGKILL (9): Force kill
- SIGHUP (1): Hang up (reload config)
- SIGINT (2): Interrupt (Ctrl+C)
- SIGSTOP (19): Stop process
- SIGCONT (18): Continue stopped process

**Process Groups:**
- Processes organized into groups
- Job control uses process groups
- ps -o pid,pgid,comm shows process groups`,
					CodeExamples: `# Send signal
kill -SIGTERM 12345
kill -15 12345  # Same as above

# List all signals
kill -l

# Process groups
ps -o pid,pgid,comm

# Send signal to process group
kill -TERM -12345  # Negative PID = process group`,
				},
				{
					Title: "Cron Jobs",
					Content: `Cron schedules commands to run automatically at specified times.

**Cron Syntax:**
- Format: minute hour day month weekday command
- Minute: 0-59
- Hour: 0-23
- Day of month: 1-31
- Month: 1-12
- Day of week: 0-7 (0 or 7 = Sunday)

**Crontab:**
- crontab -e: Edit crontab
- crontab -l: List crontab
- crontab -r: Remove crontab`,
					CodeExamples: `# Edit crontab
crontab -e

# Examples:
# Run every minute
* * * * * /path/to/script.sh

# Run every hour
0 * * * * /path/to/script.sh

# Run daily at 2 AM
0 2 * * * /path/to/script.sh

# Run weekly on Monday at 3 AM
0 3 * * 1 /path/to/script.sh

# Run monthly on 1st at midnight
0 0 1 * * /path/to/script.sh

# List crontab
crontab -l

# Remove crontab
crontab -r`,
				},
				{
					Title: "Daemons & Background Services",
					Content: `Daemons are background processes that run continuously, providing services to the system.

**What are Daemons?**
- Background processes (no controlling terminal)
- Provide system services
- Start at boot or on demand
- Managed by init system (systemd)

**Traditional Daemons:**
- Fork and detach from terminal
- Double-fork to become orphan
- Write PID to file
- Redirect I/O to /dev/null

**systemd Services:**
- Modern way to manage daemons
- Service files define behavior
- Automatic dependency management
- Better logging and monitoring

**Creating a Simple Daemon:**
1. Fork process
2. Exit parent
3. Create new session (setsid)
4. Fork again
5. Exit parent
6. Change directory
7. Set umask
8. Redirect I/O

**Daemon Best Practices:**
- Log to syslog or journald
- Create PID file
- Handle signals (SIGHUP for reload)
- Don't hold file descriptors
- Use appropriate user/group`,
					CodeExamples: `# View running daemons
systemctl list-units --type=service --state=running

# View all services
systemctl list-units --type=service

# Check if service is daemon
systemctl status service-name

# Traditional daemon script
#!/bin/bash
# Simple daemon example
DAEMON_NAME="mydaemon"
PIDFILE="/var/run/${DAEMON_NAME}.pid"

start_daemon() {
    if [ -f "$PIDFILE" ]; then
        echo "Daemon already running"
        return 1
    fi
    
    nohup "$0" run > /dev/null 2>&1 &
    echo $! > "$PIDFILE"
    echo "Daemon started"
}

stop_daemon() {
    if [ ! -f "$PIDFILE" ]; then
        echo "Daemon not running"
        return 1
    fi
    
    PID=$(cat "$PIDFILE")
    kill "$PID"
    rm "$PIDFILE"
    echo "Daemon stopped"
}

run() {
    # Daemon main loop
    while true; do
        # Do work here
        sleep 60
    done
}

case "$1" in
    start) start_daemon ;;
    stop) stop_daemon ;;
    run) run ;;
    *) echo "Usage: $0 {start|stop}" ;;
esac

# systemd service file example
# /etc/systemd/system/myservice.service
# [Unit]
# Description=My Service
# After=network.target
#
# [Service]
# Type=simple
# ExecStart=/usr/bin/myservice
# Restart=always
# User=myservice
#
# [Install]
# WantedBy=multi-user.target`,
				},
				{
					Title: "systemd Services Deep Dive",
					Content: `systemd is the modern init system and service manager. Understanding it deeply is essential for Linux administration.

**Service File Structure:**

**Unit Section:**
- Description: Service description
- After: Start after these units
- Requires: Hard dependency
- Wants: Soft dependency
- Before: Start before these units

**Service Section:**
- Type: Service type (simple, forking, oneshot, notify)
- ExecStart: Command to start service
- ExecStop: Command to stop service
- ExecReload: Command to reload config
- Restart: Restart policy
- User/Group: Run as user/group
- WorkingDirectory: Working directory

**Install Section:**
- WantedBy: Target that wants this service
- RequiredBy: Target that requires this service

**Service Types:**
- **simple**: Main process (default)
- **forking**: Process forks, parent exits
- **oneshot**: Run once and exit
- **notify**: Service sends notification when ready
- **idle**: Wait for other jobs to finish

**Timers:**
- systemd's cron replacement
- More flexible than cron
- Can trigger on calendar or after events
- Service file + timer file`,
					CodeExamples: `# Create service file
sudo nano /etc/systemd/system/myservice.service

# Example service file
[Unit]
Description=My Custom Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/myservice
Restart=always
RestartSec=10
User=myuser
Group=mygroup

[Install]
WantedBy=multi-user.target

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable myservice

# Start service
sudo systemctl start myservice

# Check status
sudo systemctl status myservice

# View service logs
journalctl -u myservice -f

# Create timer
sudo nano /etc/systemd/system/mytimer.timer
# [Unit]
# Description=Run my service daily
#
# [Timer]
# OnCalendar=daily
# Persistent=true
#
# [Install]
# WantedBy=timers.target

# Enable timer
sudo systemctl enable mytimer.timer
sudo systemctl start mytimer.timer

# List timers
systemctl list-timers

# Service dependencies
systemctl list-dependencies myservice

# Mask service (prevent starting)
sudo systemctl mask service-name

# Unmask service
sudo systemctl unmask service-name`,
				},
				{
					Title: "Process Monitoring & Debugging",
					Content: `Monitoring and debugging processes is crucial for troubleshooting and performance optimization.

**strace:**
- Trace system calls and signals
- Shows what system calls process makes
- strace command: Trace command
- strace -p PID: Attach to running process
- strace -e trace=open,read: Trace specific calls

**ltrace:**
- Trace library calls
- Shows function calls to shared libraries
- Similar to strace but for libraries

**Debugging Techniques:**
- **strace**: System call tracing
- **ltrace**: Library call tracing
- **gdb**: GNU Debugger (advanced)
- **core dumps**: Analyze crashed processes
- **/proc/PID**: Process information

**Process Information:**
- /proc/PID/: Process directory
- /proc/PID/cmdline: Command line
- /proc/PID/environ: Environment variables
- /proc/PID/fd/: File descriptors
- /proc/PID/status: Process status

**Monitoring Tools:**
- strace: System call tracer
- ltrace: Library call tracer
- lsof: List open files
- fuser: Identify processes using files`,
					CodeExamples: `# Trace system calls
strace ls -l

# Trace specific system calls
strace -e trace=open,read,write command

# Attach to running process
strace -p 12345

# Save trace to file
strace -o trace.log command

# Trace library calls
ltrace command

# Trace library calls for running process
ltrace -p 12345

# List open files for process
lsof -p 12345

# Find process using file
lsof /path/to/file
fuser /path/to/file

# View process information
cat /proc/12345/cmdline
cat /proc/12345/environ
ls -l /proc/12345/fd/

# Process status
cat /proc/12345/status

# Memory map
cat /proc/12345/maps

# Network connections
lsof -i -P -n | grep 12345

# Debug with gdb
gdb -p 12345
# (gdb) bt  # Backtrace
# (gdb) info registers
# (gdb) quit

# Enable core dumps
ulimit -c unlimited
echo "/tmp/core.%e.%p" > /proc/sys/kernel/core_pattern

# Analyze core dump
gdb program core-file`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1811,
			Title:       "File System Advanced",
			Description: "Learn mount points, fstab, disk management, filesystem types, LVM basics, and RAID concepts.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Mount Points & fstab",
					Content: `Mounting makes filesystems accessible. /etc/fstab defines filesystems to mount at boot.

**Mount Commands:**
- mount: Mount filesystem
- umount: Unmount filesystem
- mount -a: Mount all in fstab
- df: Show filesystem usage
- du: Show directory usage

**fstab Format:**
device mountpoint fstype options dump pass`,
					CodeExamples: `# Mount filesystem
sudo mount /dev/sda1 /mnt

# Unmount
sudo umount /mnt

# View mounted filesystems
mount
df -h

# Edit fstab
sudo nano /etc/fstab

# Example fstab entry
# /dev/sda1 /mnt ext4 defaults 0 2

# Test fstab
sudo mount -a`,
				},
				{
					Title: "Disk Management",
					Content: `Managing disks and partitions is essential for system administration.

**Disk Tools:**
- fdisk: Partition table manipulator
- parted: Advanced partitioning
- lsblk: List block devices
- blkid: Show block device attributes

**Filesystem Types:**
- ext4: Default Linux filesystem
- xfs: High-performance filesystem
- btrfs: Advanced features (snapshots, compression)`,
					CodeExamples: `# List block devices
lsblk

# Show disk usage
df -h

# Show directory usage
du -h /path
du -sh /path  # Summary

# Partition disk
sudo fdisk /dev/sda

# Format partition
sudo mkfs.ext4 /dev/sda1

# Check filesystem
sudo fsck /dev/sda1`,
				},
				{
					Title: "LVM (Logical Volume Manager)",
					Content: `LVM provides flexible disk management by abstracting physical storage into logical volumes.

**LVM Concepts:**
- **Physical Volume (PV)**: Physical disk or partition
- **Volume Group (VG)**: Collection of physical volumes
- **Logical Volume (LV)**: Virtual partition created from volume group
- **Physical Extent (PE)**: Smallest unit in LVM

**LVM Advantages:**
- Resize volumes without unmounting (usually)
- Combine multiple disks into one volume
- Create snapshots
- Move data between physical disks
- Striping and mirroring

**Common LVM Commands:**
- pvcreate: Create physical volume
- vgcreate: Create volume group
- lvcreate: Create logical volume
- pvdisplay, vgdisplay, lvdisplay: Show information
- lvextend: Extend logical volume
- resize2fs: Resize filesystem

**LVM Workflow:**
1. Create physical volume: pvcreate /dev/sdb
2. Create volume group: vgcreate vgname /dev/sdb
3. Create logical volume: lvcreate -L 10G -n lvname vgname
4. Create filesystem: mkfs.ext4 /dev/vgname/lvname
5. Mount: mount /dev/vgname/lvname /mnt`,
					CodeExamples: `# Create physical volume
sudo pvcreate /dev/sdb
sudo pvcreate /dev/sdb1 /dev/sdb2

# Display physical volumes
sudo pvdisplay
sudo pvs

# Create volume group
sudo vgcreate myvg /dev/sdb

# Add physical volume to volume group
sudo vgextend myvg /dev/sdc

# Display volume groups
sudo vgdisplay
sudo vgs

# Create logical volume
sudo lvcreate -L 10G -n mylv myvg
sudo lvcreate -l 100%FREE -n mylv myvg  # Use all space

# Display logical volumes
sudo lvdisplay
sudo lvs

# Create filesystem on logical volume
sudo mkfs.ext4 /dev/myvg/mylv

# Mount logical volume
sudo mount /dev/myvg/mylv /mnt

# Extend logical volume
sudo lvextend -L +5G /dev/myvg/mylv
sudo resize2fs /dev/myvg/mylv  # Resize filesystem

# Reduce logical volume (requires unmount)
sudo umount /mnt
sudo e2fsck -f /dev/myvg/mylv
sudo resize2fs /dev/myvg/mylv 10G
sudo lvreduce -L 10G /dev/myvg/mylv
sudo mount /dev/myvg/mylv /mnt

# Create snapshot
sudo lvcreate -L 1G -s -n mysnap /dev/myvg/mylv

# Remove snapshot
sudo lvremove /dev/myvg/mysnap

# Remove logical volume
sudo umount /mnt
sudo lvremove /dev/myvg/mylv

# Remove volume group
sudo vgremove myvg

# Remove physical volume
sudo pvremove /dev/sdb`,
				},
				{
					Title: "RAID Configuration",
					Content: `RAID (Redundant Array of Independent Disks) provides redundancy and/or performance improvements.

**Software RAID (mdadm):**
- Implemented in kernel
- No special hardware needed
- Flexible configuration
- Managed with mdadm command

**RAID Levels:**

**RAID 0 (Striping):**
- Splits data across disks
- No redundancy
- Fastest performance
- Requires minimum 2 disks

**RAID 1 (Mirroring):**
- Duplicates data on all disks
- Excellent redundancy
- Read performance improved
- Requires minimum 2 disks

**RAID 5:**
- Striping with parity
- Can survive 1 disk failure
- Good balance of performance/redundancy
- Requires minimum 3 disks

**RAID 6:**
- Like RAID 5 but dual parity
- Can survive 2 disk failures
- Requires minimum 4 disks

**RAID 10:**
- Combination of RAID 1 and RAID 0
- Mirrored stripes
- Excellent performance and redundancy
- Requires minimum 4 disks (even number)

**mdadm Commands:**
- mdadm --create: Create RAID array
- mdadm --detail: Show array details
- mdadm --manage: Manage array (add/remove disks)
- /proc/mdstat: Show RAID status`,
					CodeExamples: `# Create RAID 1 (mirroring)
sudo mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb /dev/sdc

# Create RAID 5
sudo mdadm --create /dev/md0 --level=5 --raid-devices=3 /dev/sdb /dev/sdc /dev/sdd

# Create RAID 10
sudo mdadm --create /dev/md0 --level=10 --raid-devices=4 /dev/sdb /dev/sdc /dev/sdd /dev/sde

# View RAID status
cat /proc/mdstat
sudo mdadm --detail /dev/md0

# Create filesystem on RAID
sudo mkfs.ext4 /dev/md0

# Mount RAID
sudo mount /dev/md0 /mnt

# Add spare disk
sudo mdadm --add /dev/md0 /dev/sde

# Remove failed disk
sudo mdadm --remove /dev/md0 /dev/sdb

# Stop RAID array
sudo mdadm --stop /dev/md0

# Assemble existing array
sudo mdadm --assemble /dev/md0 /dev/sdb /dev/sdc

# Save configuration
sudo mdadm --detail --scan >> /etc/mdadm/mdadm.conf

# Monitor RAID
watch cat /proc/mdstat`,
				},
				{
					Title: "Filesystem Tuning",
					Content: `Filesystem tuning optimizes performance and behavior for specific workloads.

**Mount Options:**
- noatime: Don't update access times (faster)
- nodiratime: Don't update directory access times
- relatime: Update access times relative to modify/change
- barrier: Write barriers (safety vs performance)
- data=ordered: Data ordering mode (ext3/ext4)

**Tuning Parameters:**

**ext4 Tuning:**
- tune2fs: Tune ext2/ext3/ext4 filesystems
- reserved-blocks-percent: Reserve space for root
- max-mount-counts: Force fsck after N mounts
- check-interval: Force fsck after time period

**XFS Tuning:**
- xfs_info: Show XFS information
- xfs_growfs: Grow XFS filesystem
- mkfs.xfs options: inode size, block size

**Performance Considerations:**
- **Block size**: Larger = better for large files
- **Inode count**: More inodes = more small files
- **Journaling**: Safety vs performance trade-off
- **Write barriers**: Safety vs performance

**Monitoring:**
- iostat: I/O statistics
- iotop: I/O by process
- dstat: Combined statistics`,
					CodeExamples: `# Mount with performance options
sudo mount -o noatime,nodiratime /dev/sda1 /mnt

# Add to /etc/fstab
# /dev/sda1 /mnt ext4 noatime,nodiratime 0 2

# Tune ext4 filesystem
sudo tune2fs -l /dev/sda1  # List current settings

# Set reserved blocks (default 5%)
sudo tune2fs -m 1 /dev/sda1  # Set to 1%

# Disable time-based checking
sudo tune2fs -i 0 /dev/sda1

# Disable mount-count checking
sudo tune2fs -c 0 /dev/sda1

# Set maximum mount count
sudo tune2fs -c 30 /dev/sda1

# XFS information
xfs_info /mnt

# XFS grow
sudo xfs_growfs /mnt

# Create XFS with specific options
sudo mkfs.xfs -d agcount=4 /dev/sda1  # 4 allocation groups

# I/O statistics
iostat -x 1

# I/O by process
sudo iotop

# Combined statistics
dstat

# Check filesystem usage
df -h
du -sh /path

# Find large files
find / -type f -size +100M 2>/dev/null`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1812,
			Title:       "System Monitoring & Performance",
			Description: "Master monitoring tools, performance tuning, resource limits, memory management, and CPU scheduling.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Monitoring Tools",
					Content: `Monitoring is the practice of continuously observing system behavior to detect anomalies, identify bottlenecks, and plan capacity before problems escalate into outages. Think of it like the instrument panel in a car — without gauges for speed, fuel, and engine temperature, you are driving blind. In Linux, a rich ecosystem of command-line monitoring tools gives you real-time visibility into CPU, memory, disk I/O, and network activity, each offering a different lens on system health.

**1. Process Monitoring: top and htop**

The ` + "`" + `top` + "`" + ` command provides a real-time, continuously updating view of running processes sorted by CPU or memory usage. It shows system-wide metrics (load average, uptime, total memory) alongside per-process details (PID, user, CPU%, memory%, command). Its interactive successor, ` + "`" + `htop` + "`" + `, adds color-coded CPU and memory bars, mouse support, tree-view of process hierarchies, and the ability to search, filter, and kill processes without memorizing PID numbers. In practice, ` + "`" + `htop` + "`" + ` is what most administrators reach for first during an incident — it immediately shows whether the problem is CPU-bound, memory-bound, or caused by a specific runaway process.

**2. I/O Monitoring: iotop and iostat**

When a system feels sluggish but CPU and memory look fine, the bottleneck is often disk I/O. The ` + "`" + `iotop` + "`" + ` command shows I/O usage per process in real time — which process is reading or writing to disk and how much. The ` + "`" + `iostat` + "`" + ` command provides device-level statistics: reads/writes per second, average queue length, and utilization percentage for each disk. The extended mode (` + "`" + `iostat -x` + "`" + `) reveals whether a disk is saturated (look for ` + "`" + `%util` + "`" + ` near 100%) or if requests are queuing up (high ` + "`" + `avgqu-sz` + "`" + `). Together, these tools help you pinpoint whether a slow application is waiting on disk.

**3. Memory and Virtual Memory: free and vmstat**

The ` + "`" + `free -h` + "`" + ` command gives a snapshot of total, used, and available memory in human-readable format. Understanding the "available" column is critical — it accounts for reclaimable cache and buffers, which the kernel uses to speed up disk access. The ` + "`" + `vmstat` + "`" + ` command provides a broader view: it shows memory, swap, I/O, and CPU statistics at configurable intervals. A high ` + "`" + `si` + "`" + `/` + "`" + `so` + "`" + ` (swap in/out) value means the system is actively swapping to disk, which devastates performance. A sustained high value in the ` + "`" + `wa` + "`" + ` (I/O wait) column means processes are blocked waiting for disk operations.

**4. Network Monitoring: nethogs**

While ` + "`" + `iftop` + "`" + ` shows bandwidth by connection, ` + "`" + `nethogs` + "`" + ` groups network traffic by process — showing exactly which program is consuming your bandwidth. This is invaluable for debugging situations where network throughput unexpectedly drops or an unknown process is saturating the link. Combined with ` + "`" + `ss` + "`" + ` (socket statistics) for connection-level detail, these tools give complete visibility into network behavior.`,
					CodeExamples: `# Memory usage
free -h

# CPU and memory stats
vmstat 1

# I/O statistics
iostat -x 1

# I/O by process
sudo iotop

# Network by process
sudo nethogs

# Disk I/O
iostat -d 1`,
				},
				{
					Title: "Resource Limits",
					Content: `Resource limits are the guardrails that prevent a single misbehaving process from consuming all system resources and crashing the entire machine. Imagine a shared office building where one tenant could use all the electricity, water, and parking spaces — without limits, one bad actor takes everyone down. In Linux, the ` + "`" + `ulimit` + "`" + ` command and the ` + "`" + `/etc/security/limits.conf` + "`" + ` file control per-process and per-user resource caps, forming a critical defense layer in production systems.

**1. Understanding Soft vs. Hard Limits**

Every resource limit has two values: a soft limit and a hard limit. The soft limit is the currently enforced ceiling — a process can raise its own soft limit up to the hard limit without requiring root privileges. The hard limit is the absolute maximum, and only root can increase it. This two-tier system allows applications to self-tune (e.g., a database raising its file descriptor limit on startup) while still preventing unprivileged users from consuming unlimited resources. Running ` + "`" + `ulimit -a` + "`" + ` shows all current soft limits; ` + "`" + `ulimit -aH` + "`" + ` shows the hard limits.

**2. File Descriptors (` + "`" + `ulimit -n` + "`" + `): The Most Common Bottleneck**

Every open file, socket, and pipe consumes a file descriptor. The default limit (often 1024) is far too low for servers handling thousands of concurrent connections. A web server hitting this limit will suddenly refuse new connections with "too many open files" errors. Database servers, load balancers, and message brokers all need this limit raised — often to 65536 or higher. This is the single most common resource limit problem in production deployments.

**3. Process Limits (` + "`" + `ulimit -u` + "`" + `) and Memory Limits (` + "`" + `ulimit -v` + "`" + `)**

The process limit caps how many processes a user can spawn, preventing fork bombs (a malicious or buggy program that recursively spawns processes until the system crashes). Virtual memory limits cap how much address space a process can request. In containerized environments, these limits are often supplemented by cgroups, which provide more granular control over CPU, memory, and I/O at the container or process-group level.

**4. Making Limits Persistent**

The ` + "`" + `ulimit` + "`" + ` command only affects the current shell session. To make limits survive reboots, you edit ` + "`" + `/etc/security/limits.conf` + "`" + ` (for PAM-based systems) or create a file in ` + "`" + `/etc/security/limits.d/` + "`" + `. For systemd services, limits are set in the service unit file using directives like ` + "`" + `LimitNOFILE=65536` + "`" + `. Getting this right is a rite of passage for every production systems engineer.`,
					CodeExamples: `# Show limits
ulimit -a

# Set limit
ulimit -n 4096

# Make permanent (edit /etc/security/limits.conf)`,
				},
				{
					Title: "Performance Tuning Techniques",
					Content: `Performance tuning optimizes system behavior for specific workloads and requirements.

**Kernel Parameters:**
- /proc/sys/: Runtime kernel parameters
- sysctl: View and modify kernel parameters
- /etc/sysctl.conf: Persistent kernel parameters

**Common Tuning Areas:**
- **Network**: TCP buffer sizes, connection limits
- **Memory**: Swappiness, overcommit settings
- **File system**: Inode cache, dentry cache
- **Process**: Process limits, scheduling

**sysctl Parameters:**
- vm.swappiness: Swap usage tendency (0-100)
- net.ipv4.tcp_fin_timeout: TCP FIN timeout
- fs.file-max: Maximum open files
- kernel.pid_max: Maximum process ID

**Tuning Strategy:**
1. Identify bottleneck
2. Research relevant parameters
3. Test changes incrementally
4. Monitor results
5. Document changes`,
					CodeExamples: `# View all kernel parameters
sysctl -a

# View specific parameter
sysctl vm.swappiness

# Set parameter temporarily
sudo sysctl -w vm.swappiness=10

# Set multiple parameters
sudo sysctl -w vm.swappiness=10
sudo sysctl -w net.ipv4.tcp_fin_timeout=30

# Make permanent (edit /etc/sysctl.conf)
sudo nano /etc/sysctl.conf
# Add:
# vm.swappiness=10
# net.ipv4.tcp_fin_timeout=30

# Apply sysctl.conf
sudo sysctl -p

# Network tuning
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# File descriptor limits
sudo sysctl -w fs.file-max=1000000

# Process limits
sudo sysctl -w kernel.pid_max=4194304

# TCP tuning
sudo sysctl -w net.ipv4.tcp_fin_timeout=30
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=8192

# Memory overcommit
sudo sysctl -w vm.overcommit_memory=1

# View current values
cat /proc/sys/vm/swappiness
cat /proc/sys/fs/file-max`,
				},
				{
					Title: "CPU Scheduling & Priorities",
					Content: `Understanding CPU scheduling helps optimize process performance and system responsiveness.

**CFS (Completely Fair Scheduler):**
- Default Linux scheduler
- Fair time allocation
- Uses virtual runtime
- Supports nice values

**CPU Affinity:**
- Bind process to specific CPUs
- taskset: Set CPU affinity
- Useful for NUMA systems
- Can improve cache performance

**Scheduling Policies:**
- **SCHED_NORMAL**: Default (CFS)
- **SCHED_FIFO**: First In First Out (real-time)
- **SCHED_RR**: Round Robin (real-time)
- **SCHED_BATCH**: Batch processing
- **SCHED_IDLE**: Idle priority

**chrt Command:**
- Set real-time scheduling
- chrt -f priority command: FIFO scheduling
- chrt -r priority command: Round Robin
- chrt -p PID: Show policy

**Performance Considerations:**
- Real-time processes can starve others
- CPU affinity can help with cache locality
- NUMA awareness important on multi-socket systems`,
					CodeExamples: `# Set CPU affinity
taskset -c 0,1 command
taskset -c 0-3 command  # CPUs 0,1,2,3

# Set affinity for running process
taskset -cp 0,1 12345

# Show CPU affinity
taskset -p 12345

# Real-time scheduling (FIFO)
sudo chrt -f 50 important_process

# Real-time scheduling (Round Robin)
sudo chrt -r 50 important_process

# Show scheduling policy
chrt -p 12345

# Set nice value (affects CFS)
nice -n -10 command
renice -10 12345

# View CPU info
lscpu
cat /proc/cpuinfo

# CPU statistics
mpstat -P ALL 1

# Per-CPU load
top  # Press '1' to show per-CPU

# NUMA info
numactl --hardware
numactl --show

# Run on specific NUMA node
numactl --membind=0 --cpunodebind=0 command`,
				},
				{
					Title: "Memory Management Deep Dive",
					Content: `Understanding memory management is crucial for performance tuning and troubleshooting.

**Memory Types:**
- **Physical RAM**: Actual memory
- **Swap**: Disk-based virtual memory
- **Virtual Memory**: Process address space
- **Cache**: Page cache, buffer cache

**Memory Statistics:**
- free: Show memory usage
- /proc/meminfo: Detailed memory info
- vmstat: Virtual memory statistics
- top/htop: Process memory usage

**Swap Management:**
- **Swappiness**: Tendency to swap (0-100)
- Lower = prefer RAM, higher = prefer swap
- swapoff: Disable swap
- swapon: Enable swap

**OOM Killer:**
- Out-of-Memory killer
- Kills processes when memory exhausted
- /proc/sys/vm/oom_kill_allocating_task: Control behavior
- dmesg: View OOM kills

**Memory Tuning:**
- Adjust swappiness
- Add/remove swap
- Tune overcommit settings
- Monitor memory pressure`,
					CodeExamples: `# Memory usage
free -h
cat /proc/meminfo

# Virtual memory stats
vmstat 1

# Swap usage
swapon --show
cat /proc/swaps

# Add swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Add to /etc/fstab: /swapfile none swap sw 0 0

# Remove swap
sudo swapoff /swapfile
sudo rm /swapfile

# Swappiness
cat /proc/sys/vm/swappiness
sudo sysctl -w vm.swappiness=10

# Memory overcommit
cat /proc/sys/vm/overcommit_memory
# 0 = heuristic, 1 = always, 2 = never

# OOM killer
dmesg | grep -i oom
cat /proc/sys/vm/oom_kill_allocating_task

# Process memory
ps aux --sort=-%mem | head
top  # Press 'M' to sort by memory

# Clear page cache (dangerous!)
sudo sync
sudo echo 3 > /proc/sys/vm/drop_caches
# 1 = page cache, 2 = dentries/inodes, 3 = all`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1813,
			Title:       "Security & Hardening",
			Description: "Learn security best practices, SSH hardening, firewall configuration, SELinux/AppArmor, encryption, and security auditing.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "SSH Hardening",
					Content: `SSH (Secure Shell) is the front door to virtually every Linux server on the internet. Because it provides remote command-line access, it is also the number one target for automated attacks — botnets scan the entire IPv4 address space trying default passwords against SSH on port 22. Hardening SSH is not optional; it is the single most impactful security measure you can take on any Linux server. Think of it as upgrading your front door from a screen door to a steel vault.

**1. Disable Root Login and Password Authentication**

The most critical hardening step is disabling direct root login (` + "`" + `PermitRootLogin no` + "`" + ` in ` + "`" + `sshd_config` + "`" + `) and switching to key-based authentication exclusively (` + "`" + `PasswordAuthentication no` + "`" + `). Root login gives an attacker immediate superuser access if they guess the password. Password authentication is inherently vulnerable to brute-force attacks — no matter how strong your password is, an attacker can try millions of combinations. SSH keys use asymmetric cryptography (typically RSA-4096 or Ed25519), making brute force computationally impossible. With keys, even if an attacker knows your username, they cannot authenticate without your private key file.

**2. Change the Default Port and Limit Access**

Moving SSH to a non-standard port (e.g., 2222 instead of 22) does not provide real security against a determined attacker, but it eliminates the vast majority of automated scans. Combined with ` + "`" + `AllowUsers` + "`" + ` or ` + "`" + `AllowGroups` + "`" + ` directives (which whitelist specific users/groups), you dramatically reduce the attack surface. For even stronger protection, use ` + "`" + `ListenAddress` + "`" + ` to bind SSH to a specific network interface (e.g., a management VLAN) rather than all interfaces.

**3. fail2ban: Automated Intrusion Prevention**

Even with key-only authentication, brute-force attempts consume resources and clutter logs. ` + "`" + `fail2ban` + "`" + ` monitors log files for repeated failed login attempts and automatically adds firewall rules to block the offending IP addresses. It is like a bouncer that throws out anyone who tries the wrong key too many times. The default configuration bans an IP after 5 failed attempts for 10 minutes, but production servers often use stricter settings (3 attempts, 1-hour ban). fail2ban is not limited to SSH — it can protect any service that logs authentication failures.

**4. Additional Hardening: Timeouts, Ciphers, and Two-Factor**

Set ` + "`" + `ClientAliveInterval` + "`" + ` and ` + "`" + `ClientAliveCountMax` + "`" + ` to drop idle sessions. Restrict allowed ciphers and key exchange algorithms to modern, secure options (disable anything with CBC mode or SHA-1). For maximum security, enable two-factor authentication (2FA) using PAM modules like Google Authenticator, requiring both a key and a time-based one-time password. Each layer makes the attacker's job exponentially harder.`,
					CodeExamples: `# Edit SSH config
sudo nano /etc/ssh/sshd_config

# Key settings:
# PermitRootLogin no
# PasswordAuthentication no
# Port 2222
# PubkeyAuthentication yes

# Restart SSH
sudo systemctl restart sshd`,
				},
				{
					Title: "Firewall Configuration",
					Content: `A firewall is the gatekeeper between your server and the outside world, deciding which network packets are allowed in, out, or forwarded based on rules you define. Without a properly configured firewall, every listening service on your server is exposed to the internet — and attackers will find them. Linux has two main firewall management tools: ` + "`" + `iptables` + "`" + ` (the traditional, low-level tool) and ` + "`" + `firewalld` + "`" + ` (the modern, zone-based frontend). Both ultimately configure the kernel's Netfilter framework.

**1. iptables: The Low-Level Powerhouse**

` + "`" + `iptables` + "`" + ` works by defining chains of rules organized into tables. The most commonly used table is ` + "`" + `filter` + "`" + `, which has three built-in chains: ` + "`" + `INPUT` + "`" + ` (packets destined for the local machine), ` + "`" + `OUTPUT` + "`" + ` (packets originating from the local machine), and ` + "`" + `FORWARD` + "`" + ` (packets being routed through the machine). Each rule specifies match criteria (source IP, destination port, protocol) and an action (ACCEPT, DROP, REJECT, LOG). Rules are evaluated in order — the first match wins. A best practice is to set the default policy to DROP and then explicitly allow only the traffic you need (a "whitelist" approach).

**2. firewalld: Zone-Based Management**

` + "`" + `firewalld` + "`" + ` provides a higher-level abstraction using zones and services. A zone (like ` + "`" + `public` + "`" + `, ` + "`" + `trusted` + "`" + `, ` + "`" + `dmz` + "`" + `) is a set of rules applied to a network interface. Services (like ` + "`" + `http` + "`" + `, ` + "`" + `ssh` + "`" + `, ` + "`" + `dns` + "`" + `) are predefined port/protocol combinations. Instead of memorizing port numbers, you can say ` + "`" + `firewall-cmd --add-service=http` + "`" + `. The ` + "`" + `--permanent` + "`" + ` flag persists rules across reboots, while rules without it are temporary (useful for testing). The ` + "`" + `--reload` + "`" + ` command applies permanent changes without dropping existing connections.

**3. Best Practices for Production Firewalls**

Start with a deny-all default policy and explicitly open only the ports your services need. Log dropped packets (at least initially) to understand what traffic is being blocked. Use connection tracking (` + "`" + `-m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT` + "`" + `) to allow return traffic for established connections without opening ports permanently. Separate management traffic (SSH) from application traffic using different interfaces or zones. Always test firewall changes on a non-production system first — a misconfigured rule can lock you out of your own server.`,
					CodeExamples: `# List rules
sudo firewall-cmd --list-all

# Allow service
sudo firewall-cmd --add-service=http --permanent

# Allow port
sudo firewall-cmd --add-port=8080/tcp --permanent

# Reload
sudo firewall-cmd --reload`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1814,
			Title:       "Shell Scripting Advanced",
			Description: "Master advanced bash features, regular expressions, advanced text processing, script optimization, and error handling patterns.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Bash Features",
					Content: `Beyond basic loops and conditionals, Bash offers a set of advanced features that transform it from a simple command runner into a genuinely powerful scripting language. These features — parameter expansion, process substitution, here documents, and advanced redirection — are what separate quick-and-dirty scripts from robust, production-quality automation. Mastering them lets you write scripts that are shorter, faster, and more readable.

**1. Parameter Expansion: String Manipulation Without External Tools**

Parameter expansion lets you manipulate variable values directly in Bash without calling external programs like ` + "`" + `sed` + "`" + ` or ` + "`" + `awk` + "`" + `. The syntax ` + "`" + `${VAR:-default}` + "`" + ` provides a fallback value if the variable is unset (perfect for configuration defaults). ` + "`" + `${VAR:=default}` + "`" + ` goes further — it actually sets the variable to the default. ` + "`" + `${VAR:+alternate}` + "`" + ` returns the alternate value only if VAR is already set (useful for conditional flags). You can also extract substrings (` + "`" + `${VAR:offset:length}` + "`" + `), replace patterns (` + "`" + `${VAR/pattern/replacement}` + "`" + `), and strip prefixes or suffixes (` + "`" + `${VAR%.ext}` + "`" + ` removes the extension from a filename). These operations run in the shell itself, making them dramatically faster than piping to external commands in tight loops.

**2. Process Substitution: Treating Command Output as Files**

Process substitution (` + "`" + `<(command)` + "`" + ` and ` + "`" + `>(command)` + "`" + `) creates a temporary file-like object that contains the output of a command. This is incredibly useful when a program requires file arguments but you want to feed it dynamic data. The classic example is ` + "`" + `diff <(command1) <(command2)` + "`" + `, which compares the output of two commands without creating temporary files. Another common use is ` + "`" + `while read line; do ...; done < <(find . -name "*.log")` + "`" + `, which avoids the subshell problem that occurs with piping into ` + "`" + `while` + "`" + `.

**3. Here Documents and Here Strings**

A here document (` + "`" + `<<EOF ... EOF` + "`" + `) lets you embed multi-line input directly in a script, which is far more readable than chaining ` + "`" + `echo` + "`" + ` statements. It is commonly used to generate configuration files, SQL queries, or email bodies inline. Using ` + "`" + `<<-EOF` + "`" + ` allows indentation with tabs (keeping the script visually clean), and quoting the delimiter (` + "`" + `<<'EOF'` + "`" + `) prevents variable expansion inside the block. Here strings (` + "`" + `<<<` + "`" + `) provide a one-line version for feeding a single string to a command's stdin.

**4. Advanced I/O Redirection**

Beyond basic ` + "`" + `>` + "`" + ` and ` + "`" + `>>` + "`" + `, Bash supports redirecting specific file descriptors (` + "`" + `2>errors.log` + "`" + ` for stderr), combining streams (` + "`" + `2>&1` + "`" + ` merges stderr into stdout), and opening custom file descriptors (` + "`" + `exec 3>logfile` + "`" + `). The pattern ` + "`" + `command > /dev/null 2>&1` + "`" + ` silences all output and is one of the most commonly used idioms in cron jobs and background services.`,
					CodeExamples: `# Parameter expansion
${VAR:-default}  # Use default if VAR unset
${VAR:=default}  # Set and use default
${VAR:+value}    # Use value if VAR set

# Process substitution
diff <(command1) <(command2)

# Here document
cat << EOF
Multi-line
text
EOF`,
				},
				{
					Title: "Regular Expressions in Bash",
					Content: `Regular expressions enable powerful pattern matching in bash scripts.

**Regex Types:**
- **BRE (Basic Regular Expression)**: Default for grep, sed
- **ERE (Extended Regular Expression)**: grep -E, awk
- **PCRE (Perl Compatible)**: More powerful, grep -P

**Basic Patterns:**
- .: Any character
- *: Zero or more of previous
- +: One or more (ERE)
- ?: Zero or one (ERE)
- ^: Start of line
- $: End of line
- []: Character class
- |: OR (ERE)

**Character Classes:**
- [0-9]: Digits
- [a-z]: Lowercase letters
- [A-Z]: Uppercase letters
- [[:digit:]]: POSIX digit class
- [[:alpha:]]: Letters
- [[:alnum:]]: Alphanumeric

**Bash Regex Matching:**
- [[ string =~ regex ]]: Match regex
- Capture groups in BASH_REMATCH array
- =~ operator uses ERE`,
					CodeExamples: `# Basic regex with grep
grep "^Error" file.txt
grep "error$" file.txt
grep "[0-9]" file.txt

# Extended regex
grep -E "error|warning" file.txt
grep -E "[0-9]+" file.txt

# Regex in bash
if [[ "hello123" =~ [0-9]+ ]]; then
    echo "Contains numbers"
fi

# Capture groups
if [[ "hello123world" =~ ([0-9]+) ]]; then
    echo "Number: ${BASH_REMATCH[1]}"
fi

# Email validation
email="user@example.com"
if [[ $email =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "Valid email"
fi

# IP address validation
ip="192.168.1.1"
if [[ $ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "Valid IP format"
fi

# Extract numbers
text="Price: $123.45"
if [[ $text =~ \$([0-9]+\.[0-9]+) ]]; then
    echo "Price: ${BASH_REMATCH[1]}"
fi

# sed with regex
sed 's/[0-9]\+/NUMBER/g' file.txt
sed -E 's/([0-9]+)/\1 numbers/g' file.txt

# awk with regex
awk '/^[0-9]+$/ {print}' file.txt
awk '$1 ~ /^[A-Z]/ {print}' file.txt`,
				},
				{
					Title: "Advanced Text Processing",
					Content: `Mastering advanced text processing enables complex data manipulation and extraction.

**sed Advanced:**
- Multiple commands: sed -e 'cmd1' -e 'cmd2'
- Script files: sed -f script.sed
- In-place editing: sed -i
- Address ranges: sed '10,20d'
- Hold space: Advanced pattern space manipulation

**awk Advanced:**
- Arrays: Associative arrays
- Functions: Built-in and user-defined
- Pattern ranges: awk '/start/,/end/ {print}'
- Field manipulation: $1=$1 (recalculate)
- Output formatting: printf

**Combining Tools:**
- Pipe multiple tools
- Process substitution
- Command substitution
- Complex pipelines`,
					CodeExamples: `# sed advanced
# Multiple substitutions
sed -e 's/old1/new1/g' -e 's/old2/new2/g' file.txt

# In-place editing
sed -i 's/old/new/g' file.txt

# Delete lines matching pattern
sed '/pattern/d' file.txt

# Print lines 10-20
sed -n '10,20p' file.txt

# Replace only in specific lines
sed '5,10s/old/new/g' file.txt

# awk arrays
awk '{count[$1]++} END {for (i in count) print i, count[i]}' file.txt

# awk functions
awk 'function square(x) {return x*x} {print square($1)}' file.txt

# awk pattern ranges
awk '/START/,/END/ {print}' file.txt

# Complex awk processing
awk 'BEGIN {FS=","; OFS="\t"} {print $1, $2, $3*2}' data.csv

# Combine sed and awk
sed 's/old/new/g' file.txt | awk '{print $1, $2}'

# Process substitution
diff <(sort file1.txt) <(sort file2.txt)

# Extract and process
grep "error" log.txt | sed 's/.*\[\([0-9]*\)\].*/\1/' | sort -n | uniq

# Multi-step text processing
cat file.txt | \
    sed 's/[^a-zA-Z0-9 ]//g' | \
    tr '[:upper:]' '[:lower:]' | \
    awk '{for(i=1;i<=NF;i++) count[$i]++} END {for(w in count) print w, count[w]}' | \
    sort -k2 -rn | head -10`,
				},
				{
					Title: "Script Optimization",
					Content: `Optimizing scripts improves performance and resource usage.

**Performance Tips:**
- Avoid unnecessary subshells
- Use built-in commands over external
- Minimize external command calls
- Use arrays instead of repeated commands
- Cache command output

**Best Practices:**
- Use [[ ]] instead of [ ] (faster)
- Avoid cat file | command, use command < file
- Use $(command) instead of backticks
- Quote variables properly
- Use local in functions

**Profiling:**
- time: Measure execution time
- bash -x: Trace execution
- set -x: Enable tracing
- Profile with time command

**Common Optimizations:**
- Reduce external commands
- Use here documents instead of echo
- Minimize file I/O
- Use efficient algorithms`,
					CodeExamples: `# Time command execution
time ./script.sh

# Profile with detailed timing
/usr/bin/time -v ./script.sh

# Avoid unnecessary subshells
# Slow:
result=$(command1 | command2)
# Faster:
command1 | command2 > result

# Use built-ins
# Slow:
[ "$var" = "value" ]
# Faster:
[[ $var == value ]]

# Avoid cat
# Slow:
cat file.txt | grep pattern
# Faster:
grep pattern file.txt

# Cache command output
if [ -z "$CACHED_OUTPUT" ]; then
    CACHED_OUTPUT=$(expensive_command)
fi

# Use arrays efficiently
files=(*.txt)
for file in "${files[@]}"; do
    process "$file"
done

# Minimize external commands in loops
# Slow:
for i in {1..1000}; do
    echo "$i" >> file.txt
done
# Faster:
for i in {1..1000}; do
    echo "$i"
done > file.txt

# Use here documents
cat > config.txt << EOF
setting1=value1
setting2=value2
EOF

# Optimize string operations
# Use parameter expansion instead of external commands
length=${#string}
substring=${string:0:10}`,
				},
				{
					Title: "Debugging & Testing Scripts",
					Content: `Proper debugging and testing ensures script reliability and correctness.

**Debugging Techniques:**
- bash -x: Trace execution
- set -x: Enable tracing
- set -euo pipefail: Strict mode
- trap: Error handling
- echo: Print debugging info

**Testing Approaches:**
- **Unit testing**: Test functions individually
- **Integration testing**: Test script as whole
- **Edge cases**: Test boundary conditions
- **Error cases**: Test error handling

**Testing Tools:**
- shellcheck: Static analysis
- bats: Bash Automated Testing System
- Manual testing with various inputs
- Test with different shells

**Best Practices:**
- Test incrementally
- Test edge cases
- Test error conditions
- Document expected behavior
- Use version control`,
					CodeExamples: `#!/bin/bash
set -euo pipefail  # Strict mode

# Debug function
debug() {
    [ "${DEBUG:-0}" -eq 1 ] && echo "DEBUG: $@" >&2
}

# Test function
test_function() {
    local input="$1"
    local expected="$2"
    local result
    
    result=$(process "$input")
    
    if [ "$result" = "$expected" ]; then
        echo "PASS: $input -> $result"
    else
        echo "FAIL: $input -> $result (expected $expected)"
        return 1
    fi
}

# Run tests
test_function "input1" "output1"
test_function "input2" "output2"

# Using shellcheck
shellcheck script.sh

# Using bats (Bash Automated Testing System)
# test.bats:
# @test "test name" {
#     run ./script.sh
#     [ "$status" -eq 0 ]
#     [ "$output" = "expected" ]
# }

# Debug mode
DEBUG=1 ./script.sh

# Trace specific section
set -x
# Your code
set +x

# Error handling
trap 'echo "Error on line $LINENO"' ERR
trap 'cleanup' EXIT

# Test with different inputs
for input in "test1" "test2" "test3"; do
    echo "Testing: $input"
    ./script.sh "$input"
done`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1815,
			Title:       "System Programming Basics",
			Description: "Introduction to system calls, file I/O programming, process creation (fork/exec), signals programming, and IPC basics.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "System Calls Overview",
					Content: `System calls are the fundamental interface between user-space programs and the Linux kernel. Every time your program opens a file, creates a process, sends data over a network, or allocates memory, it ultimately makes a system call — a controlled entry point into kernel code that executes with elevated privileges. Think of system calls as the API of the operating system itself. Without them, user programs would have no way to interact with hardware, the filesystem, or other processes. Understanding system calls is what separates someone who uses Linux from someone who truly understands it.

**1. How System Calls Work: The User-Kernel Boundary**

When a program calls a function like ` + "`" + `open()` + "`" + ` from the C standard library, the library translates this into a system call by placing the system call number in a CPU register and triggering a software interrupt (on x86-64, this is the ` + "`" + `syscall` + "`" + ` instruction). The CPU switches from user mode to kernel mode, the kernel looks up the system call number in a dispatch table, executes the corresponding kernel function, and returns the result back to user space. This mode switch is relatively expensive (hundreds of nanoseconds), which is why high-performance applications try to minimize system call frequency through techniques like buffered I/O and memory-mapped files.

**2. File Operations: open, read, write, close**

These four system calls form the foundation of all I/O in Linux. ` + "`" + `open()` + "`" + ` returns a file descriptor — a small integer that serves as a handle to the opened file. ` + "`" + `read()` + "`" + ` and ` + "`" + `write()` + "`" + ` transfer data between user-space buffers and the kernel. ` + "`" + `close()` + "`" + ` releases the file descriptor. The beauty of the Unix philosophy is that this same interface works for regular files, pipes, sockets, and device files — everything is a file descriptor.

**3. Process Management: fork, exec, wait**

` + "`" + `fork()` + "`" + ` creates a new process by duplicating the calling process. The child gets an exact copy of the parent's memory space (using copy-on-write for efficiency). ` + "`" + `exec()` + "`" + ` replaces the current process image with a new program — this is how shells launch commands. The combination of fork-then-exec is the standard Unix process creation pattern: the parent forks, the child calls exec to become a new program, and the parent calls ` + "`" + `wait()` + "`" + ` to collect the child's exit status. ` + "`" + `pipe()` + "`" + ` creates a unidirectional data channel between related processes, forming the backbone of shell pipelines (` + "`" + `ls | grep | sort` + "`" + `).

**4. Tracing System Calls: strace**

The ` + "`" + `strace` + "`" + ` tool lets you observe every system call a program makes in real time. This is arguably the most powerful debugging tool in Linux — when a program fails mysteriously, ` + "`" + `strace` + "`" + ` reveals exactly which system call failed and why (e.g., "file not found," "permission denied," "connection refused"). Running ` + "`" + `strace -e trace=open,read,write ./myprogram` + "`" + ` filters to just I/O-related calls, making the output manageable.`,
					CodeExamples: `# C example (conceptual)
#include <unistd.h>
#include <sys/types.h>

pid_t pid = fork();
if (pid == 0) {
    // Child process
    execl("/bin/ls", "ls", NULL);
} else {
    // Parent process
    wait(NULL);
}`,
				},
				{
					Title: "File I/O Programming",
					Content: `File I/O system calls provide low-level file operations.

**System Calls:**
- open: Open file, returns file descriptor
- read: Read from file descriptor
- write: Write to file descriptor
- close: Close file descriptor
- lseek: Move file pointer

**File Descriptors:**
- Small integer representing open file
- 0: stdin, 1: stdout, 2: stderr
- 3+: User files

**Error Handling:**
- System calls return -1 on error
- errno contains error code
- Check return values always

**File Operations:**
- Open modes: O_RDONLY, O_WRONLY, O_RDWR
- Flags: O_CREAT, O_APPEND, O_TRUNC
- Permissions: Set with O_CREAT`,
					CodeExamples: `# C example
#include <fcntl.h>
#include <unistd.h>

int fd = open("file.txt", O_RDONLY);
if (fd == -1) {
    perror("open");
    return 1;
}

char buffer[1024];
ssize_t n = read(fd, buffer, sizeof(buffer) - 1);
if (n == -1) {
    perror("read");
    close(fd);
    return 1;
}
buffer[n] = '\0';

close(fd);

# Write example
fd = open("output.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
write(fd, "Hello", 5);
close(fd);

# Shell equivalent
exec 3< file.txt
read -u 3 line
exec 3<&-

exec 4> output.txt
echo "Hello" >&4
exec 4>&-`,
				},
				{
					Title: "Process Creation Details",
					Content: `Understanding fork and exec is essential for process management.

**fork():**
- Creates child process (copy of parent)
- Returns 0 in child, PID in parent
- Both processes continue from fork()
- Child gets copy of parent's memory

**exec Family:**
- Replaces current process image
- execl, execv, execle, etc.
- Different argument formats
- Process continues with new program

**wait():**
- Parent waits for child to exit
- wait(NULL): Wait for any child
- waitpid(pid, &status, 0): Wait for specific child
- Returns child's exit status

**Process Tree:**
- Parent-child relationships
- pstree: Show process tree
- Orphan processes adopted by init
- Zombie processes (exited but not waited)`,
					CodeExamples: `# C fork/exec example
#include <unistd.h>
#include <sys/wait.h>

pid_t pid = fork();
if (pid == 0) {
    // Child process
    execl("/bin/ls", "ls", "-l", NULL);
    perror("execl");  // Only if exec fails
    exit(1);
} else if (pid > 0) {
    // Parent process
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        printf("Child exited with %d\n", WEXITSTATUS(status));
    }
} else {
    perror("fork");
    return 1;
}

# Shell equivalent
(ls -l) &  # Background process
wait      # Wait for background jobs

# Process tree
pstree
pstree -p  # With PIDs

# View process hierarchy
ps -ef --forest`,
				},
				{
					Title: "Signals Programming",
					Content: `Signals are software interrupts. Programming with signals requires careful handling.

**Signal Handling:**
- signal(): Simple signal handler (deprecated)
- sigaction(): Advanced signal handling (preferred)
- Handler function receives signal number
- Some signals cannot be caught (SIGKILL, SIGSTOP)

**Common Signals:**
- SIGTERM: Termination request
- SIGINT: Interrupt (Ctrl+C)
- SIGHUP: Hang up
- SIGUSR1, SIGUSR2: User-defined
- SIGCHLD: Child process exited

**Signal Masking:**
- Block signals during critical sections
- sigprocmask(): Set signal mask
- Prevent signal delivery temporarily

**Best Practices:**
- Use sigaction instead of signal
- Reinstall handler in handler
- Be careful with async-signal-safe functions
- Handle SIGCHLD to prevent zombies`,
					CodeExamples: `# C signal handling
#include <signal.h>
#include <unistd.h>

void handler(int sig) {
    write(STDOUT_FILENO, "Signal received\n", 15);
}

int main() {
    struct sigaction sa;
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    sigaction(SIGINT, &sa, NULL);
    
    while(1) {
        pause();  // Wait for signal
    }
}

# Block signals
sigset_t set;
sigemptyset(&set);
sigaddset(&set, SIGINT);
sigprocmask(SIG_BLOCK, &set, NULL);
// Critical section
sigprocmask(SIG_UNBLOCK, &set, NULL);

# Shell signal handling
trap 'echo "SIGINT received"' INT
trap 'cleanup' EXIT

# Ignore signal
trap '' INT

# Reset to default
trap - INT`,
				},
				{
					Title: "Inter-Process Communication",
					Content: `IPC mechanisms allow processes to communicate and synchronize.

**IPC Methods:**

**Pipes:**
- Unidirectional communication
- pipe() creates pipe
- Parent-child communication
- Data flows one direction

**FIFOs (Named Pipes):**
- mkfifo() creates named pipe
- Unrelated processes can use
- Appears as file in filesystem
- Blocking I/O

**Shared Memory:**
- Fastest IPC method
- shmget(), shmat(), shmdt()
- Multiple processes access same memory
- Requires synchronization

**Message Queues:**
- msgget(), msgsnd(), msgrcv()
- Structured messages
- Type-based message selection

**Sockets:**
- Network or Unix domain
- Bidirectional communication
- Most flexible`,
					CodeExamples: `# C pipe example
#include <unistd.h>

int pipefd[2];
pipe(pipefd);

if (fork() == 0) {
    // Child: write to pipe
    close(pipefd[0]);
    write(pipefd[1], "Hello", 5);
    close(pipefd[1]);
} else {
    // Parent: read from pipe
    close(pipefd[1]);
    char buf[10];
    read(pipefd[0], buf, 10);
    close(pipefd[0]);
}

# Shell pipe
command1 | command2

# Named pipe (FIFO)
mkfifo mypipe
echo "Hello" > mypipe &
cat < mypipe

# Shared memory (conceptual)
# shmget() - create/get segment
# shmat() - attach to process
# shmdt() - detach
# shmctl() - control

# Message queue (conceptual)
# msgget() - create/get queue
# msgsnd() - send message
# msgrcv() - receive message
# msgctl() - control`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1816,
			Title:       "Kernel & System Internals",
			Description: "Explore kernel architecture, kernel modules, /proc and /sys filesystems, boot process, and init systems.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Kernel Modules",
					Content: `Kernel modules are pieces of code that can be loaded into and unloaded from the Linux kernel at runtime, extending its functionality without requiring a reboot or recompilation. They are the reason Linux can support an enormous range of hardware and filesystems without having a monstrously large kernel binary. Think of modules as plugins for the operating system: when you plug in a USB device, the kernel automatically loads the appropriate driver module; when you mount an NFS filesystem, the NFS module gets loaded on demand.

**1. Why Modules Exist: Monolithic vs. Modular Kernels**

Linux uses a monolithic kernel architecture (all kernel code runs in the same address space for performance), but with a modular twist. Instead of compiling every possible driver and feature into the kernel, most are built as loadable modules (` + "`" + `.ko` + "`" + ` files stored in ` + "`" + `/lib/modules/` + "`" + `). This keeps the base kernel small and fast while allowing virtually unlimited extensibility. The kernel can load modules on demand (triggered by hardware events or manual commands) and unload them when no longer needed, freeing memory.

**2. Module Management Commands**

` + "`" + `lsmod` + "`" + ` lists all currently loaded modules along with their sizes and dependency counts. ` + "`" + `modinfo module_name` + "`" + ` reveals detailed information about a module: its description, author, license, parameters it accepts, and what hardware it supports. ` + "`" + `modprobe` + "`" + ` is the smart module loader — it automatically resolves and loads dependencies. For example, loading a WiFi driver that depends on the ` + "`" + `cfg80211` + "`" + ` module will cause ` + "`" + `modprobe` + "`" + ` to load ` + "`" + `cfg80211` + "`" + ` first. ` + "`" + `modprobe -r` + "`" + ` safely removes a module and its unused dependencies. The lower-level ` + "`" + `insmod` + "`" + ` and ` + "`" + `rmmod` + "`" + ` commands load/unload modules without dependency resolution — use them only when ` + "`" + `modprobe` + "`" + ` is not available.

**3. Blacklisting and Persistent Configuration**

Sometimes you need to prevent a module from loading — for example, to disable a problematic driver or to force a specific driver when two modules compete for the same hardware. Adding ` + "`" + `blacklist module_name` + "`" + ` to a file in ` + "`" + `/etc/modprobe.d/` + "`" + ` prevents automatic loading. Module parameters can be made persistent in the same directory (e.g., ` + "`" + `options snd_hda_intel power_save=1` + "`" + `). For modules that must load at boot, list them in ` + "`" + `/etc/modules` + "`" + ` or create a ` + "`" + `.conf` + "`" + ` file in ` + "`" + `/etc/modules-load.d/` + "`" + `.

**4. Security Implications**

Kernel modules run with full kernel privileges — a malicious module can do anything: install rootkits, intercept system calls, exfiltrate data. This is why production systems often use Secure Boot and module signing to ensure only trusted modules can load. The ` + "`" + `CONFIG_MODULE_SIG_FORCE` + "`" + ` kernel option rejects unsigned modules entirely.`,
					CodeExamples: `# List modules
lsmod

# Module info
modinfo module_name

# Load module
sudo modprobe module_name

# Remove module
sudo modprobe -r module_name`,
				},
				{
					Title: "/proc and /sys",
					Content: `Virtual filesystems providing kernel and system information.

**/proc:**
- Process and kernel information
- /proc/cpuinfo: CPU information
- /proc/meminfo: Memory information
- /proc/version: Kernel version

**/sys:**
- Kernel and device information
- /sys/class: Device classes
- /sys/devices: Physical devices`,
					CodeExamples: `# CPU info
cat /proc/cpuinfo

# Memory info
cat /proc/meminfo

# Kernel version
cat /proc/version

# Process info
ls /proc/PID/

# System info
ls /sys/class/`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1817,
			Title:       "DevOps Tools & Automation",
			Description: "Learn Git basics, CI/CD concepts, configuration management (Ansible), container basics (Docker), and automation scripting.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Git Basics",
					Content: `Git is the universal version control system that underpins virtually all modern software development and DevOps workflows. Created by Linus Torvalds in 2005 to manage the Linux kernel source code, Git tracks every change to every file in a repository, allowing teams to collaborate without overwriting each other's work, roll back to any previous state, and maintain parallel branches of development. For a Linux system administrator, Git is not optional — it is how you version infrastructure code, configuration files, automation scripts, and documentation.

**1. The Core Mental Model: Snapshots, Not Diffs**

Unlike older version control systems that store differences between file versions, Git stores complete snapshots of your project at each commit. When a file has not changed, Git stores a reference to the previous identical file rather than a duplicate. This design makes branching and merging extremely fast — switching branches is essentially just swapping pointers. Every commit has a unique SHA-1 hash that serves as both its identifier and its integrity check. If even one bit changes in the repository history, the hashes change, making tampering detectable.

**2. The Three Stages: Working Directory, Staging Area, Repository**

Understanding Git's three-stage workflow is essential. Your working directory contains the actual files you edit. The staging area (also called the index) is a holding area where you prepare the next commit — you choose exactly which changes to include by running ` + "`" + `git add` + "`" + `. The repository (` + "`" + `.git` + "`" + ` directory) stores the committed history. This three-stage design gives you precise control: you can modify ten files but commit only three of them, keeping unrelated changes separate. ` + "`" + `git status` + "`" + ` shows exactly where every change sits in this pipeline.

**3. Branching and Collaboration**

Branches in Git are lightweight — they are just pointers to commits, not copies of files. Creating a branch (` + "`" + `git checkout -b feature-x` + "`" + `) is instant. This encourages the practice of creating a branch for every feature or fix, keeping the main branch stable. ` + "`" + `git clone` + "`" + ` downloads a complete copy of a remote repository, including all history and branches. ` + "`" + `git pull` + "`" + ` fetches remote changes and merges them, while ` + "`" + `git push` + "`" + ` uploads your local commits. For infrastructure teams, this workflow enables code review for configuration changes, audit trails for compliance, and reproducible environments through version-controlled Infrastructure as Code.

**4. Git for System Administration**

Beyond code, Git is invaluable for tracking ` + "`" + `/etc` + "`" + ` configuration files (tools like ` + "`" + `etckeeper` + "`" + ` automate this), maintaining Ansible playbooks and Terraform modules, documenting runbooks, and managing Dockerfiles and Kubernetes manifests. The ability to ` + "`" + `git blame` + "`" + ` a configuration file to see who changed what and when has saved countless hours of incident investigation.`,
					CodeExamples: `# Initialize repo
git init

# Add files
git add file.txt
git add .

# Commit
git commit -m "Initial commit"

# Status
git status

# History
git log

# Clone
git clone https://github.com/user/repo.git`,
				},
				{
					Title: "Docker Basics",
					Content: `Docker revolutionized software deployment by solving the "it works on my machine" problem. A Docker container packages an application together with all its dependencies — libraries, runtime, configuration files, even the operating system's user-space tools — into a single, portable unit that runs identically on a developer's laptop, a CI server, and a production cluster. For Linux administrators, Docker has become as fundamental as package management: it is how modern applications are built, shipped, and run.

**1. The Container Mental Model: Lightweight Isolation**

A container is not a virtual machine. It does not run its own kernel or emulate hardware. Instead, it uses Linux kernel features — namespaces for isolation (each container has its own view of the filesystem, network, and process tree) and cgroups for resource limits (CPU, memory, I/O). This makes containers extremely lightweight: they start in milliseconds (not minutes like VMs), use minimal memory overhead, and can run hundreds of instances on a single host. Think of containers as isolated apartments in a building (the host) — they share the building's foundation (the kernel) but have their own walls, plumbing, and addresses.

**2. Images vs. Containers: The Blueprint and the Instance**

A Docker image is a read-only template — a layered filesystem built from a Dockerfile. Each instruction in the Dockerfile (` + "`" + `FROM` + "`" + `, ` + "`" + `RUN` + "`" + `, ` + "`" + `COPY` + "`" + `) creates a new layer that is cached and reused. A container is a running instance of an image with a thin read-write layer on top. You can create many containers from the same image, just like printing many documents from the same template. ` + "`" + `docker images` + "`" + ` lists your local images, ` + "`" + `docker ps` + "`" + ` lists running containers (add ` + "`" + `-a` + "`" + ` to include stopped ones), and ` + "`" + `docker build -t myapp .` + "`" + ` builds an image from the Dockerfile in the current directory.

**3. Essential Docker Commands for Daily Work**

` + "`" + `docker run` + "`" + ` creates and starts a container. The ` + "`" + `-d` + "`" + ` flag runs it in the background (detached), ` + "`" + `-p 8080:80` + "`" + ` maps a host port to a container port, ` + "`" + `-v /host/path:/container/path` + "`" + ` mounts a volume, and ` + "`" + `--name` + "`" + ` gives it a human-readable name. ` + "`" + `docker exec -it container_name bash` + "`" + ` opens an interactive shell inside a running container — indispensable for debugging. ` + "`" + `docker logs container_name` + "`" + ` shows stdout/stderr output. ` + "`" + `docker stop` + "`" + ` gracefully stops a container (sends SIGTERM), while ` + "`" + `docker rm` + "`" + ` removes a stopped container.

**4. Docker in Production: Compose and Beyond**

For multi-container applications, Docker Compose lets you define services, networks, and volumes in a single YAML file and manage them with ` + "`" + `docker compose up` + "`" + `. In production, containers are typically orchestrated by Kubernetes or Docker Swarm, which handle scaling, load balancing, rolling updates, and self-healing across clusters of machines. Understanding Docker fundamentals is the prerequisite for all of these.`,
					CodeExamples: `# Run container
docker run -d nginx

# List containers
docker ps
docker ps -a

# List images
docker images

# Build image
docker build -t myapp .

# Execute in container
docker exec -it container_name bash`,
				},
				{
					Title: "Ansible Basics",
					Content: `Ansible is a configuration management and automation tool.

**Ansible Concepts:**
- **Playbooks**: YAML files defining tasks
- **Inventory**: List of hosts to manage
- **Modules**: Units of work (copy, service, etc.)
- **Tasks**: Individual operations
- **Handlers**: Tasks triggered by events

**Ansible Commands:**
- ansible: Run ad-hoc commands
- ansible-playbook: Run playbooks
- ansible-vault: Encrypt sensitive data
- ansible-galaxy: Install roles

**Ad-hoc Commands:**
- Quick one-off tasks
- No playbook needed
- Useful for testing

**Playbooks:**
- Declarative configuration
- Idempotent (safe to run multiple times)
- YAML format
- Reusable and shareable`,
					CodeExamples: `# Ad-hoc command
ansible all -m ping
ansible webservers -m shell -a "uptime"
ansible all -m copy -a "src=/etc/hosts dest=/tmp/hosts"

# Playbook example (playbook.yml)
# ---
# - hosts: webservers
#   tasks:
#     - name: Install nginx
#       apt:
#         name: nginx
#         state: present
#     - name: Start nginx
#       service:
#         name: nginx
#         state: started

# Run playbook
ansible-playbook playbook.yml

# With inventory
ansible-playbook -i inventory playbook.yml

# Check mode (dry-run)
ansible-playbook --check playbook.yml

# Limit to specific hosts
ansible-playbook --limit webservers playbook.yml

# Inventory file (inventory)
# [webservers]
# web1.example.com
# web2.example.com
#
# [dbservers]
# db1.example.com

# Common modules
ansible all -m apt -a "name=nginx state=installed"
ansible all -m service -a "name=nginx state=started"
ansible all -m copy -a "src=file.txt dest=/tmp/file.txt"
ansible all -m file -a "path=/tmp/dir state=directory"`,
				},
				{
					Title: "CI/CD Pipelines",
					Content: `CI/CD (Continuous Integration/Continuous Deployment) automates software delivery.

**CI/CD Concepts:**
- **CI**: Automatically test code changes
- **CD**: Automatically deploy tested code
- **Pipeline**: Automated workflow
- **Stages**: Steps in pipeline (build, test, deploy)

**Common CI/CD Tools:**
- **Jenkins**: Self-hosted CI/CD server
- **GitLab CI**: Integrated with GitLab
- **GitHub Actions**: Integrated with GitHub
- **GitLab Runner**: Executes GitLab CI jobs

**Pipeline Stages:**
1. **Build**: Compile/package application
2. **Test**: Run automated tests
3. **Deploy**: Deploy to environment
4. **Notify**: Send notifications

**Best Practices:**
- Automate everything
- Test before deploy
- Use version control
- Document pipelines
- Monitor deployments`,
					CodeExamples: `# Jenkins pipeline (Jenkinsfile)
# pipeline {
#     agent any
#     stages {
#         stage('Build') {
#             steps {
#                 sh 'make build'
#             }
#         }
#         stage('Test') {
#             steps {
#                 sh 'make test'
#             }
#         }
#         stage('Deploy') {
#             steps {
#                 sh 'make deploy'
#             }
#         }
#     }
# }

# GitLab CI (.gitlab-ci.yml)
# stages:
#   - build
#   - test
#   - deploy
#
# build:
#   stage: build
#   script:
#     - make build
#
# test:
#   stage: test
#   script:
#     - make test
#
# deploy:
#   stage: deploy
#   script:
#     - make deploy

# GitHub Actions (.github/workflows/ci.yml)
# name: CI
# on: [push]
# jobs:
#   build:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v2
#       - run: make build
#       - run: make test

# Simple deployment script
#!/bin/bash
set -e
git pull
make build
make test
make deploy`,
				},
				{
					Title: "Automation Patterns",
					Content: `Understanding automation patterns helps create reliable and maintainable automation.

**Idempotency:**
- Operations safe to run multiple times
- Same result regardless of current state
- Essential for configuration management
- Example: "Ensure package is installed"

**Idempotent Operations:**
- Install package (if not installed)
- Create file (if doesn't exist)
- Start service (if not running)
- Set configuration (if different)

**Configuration Management Principles:**
- **Declarative**: Describe desired state
- **Idempotent**: Safe to rerun
- **Convergent**: Moves toward desired state
- **Testable**: Can verify state

**Common Patterns:**
- **Check-then-act**: Check state, act if needed
- **Template-based**: Generate configs from templates
- **State machine**: Track and transition states
- **Event-driven**: React to events

**Error Handling:**
- Fail fast on errors
- Retry transient failures
- Log all operations
- Notify on failures`,
					CodeExamples: `# Idempotent script
if ! command -v nginx >/dev/null 2>&1; then
    apt-get install -y nginx
fi

# Check state before acting
if [ ! -f /etc/nginx/nginx.conf.backup ]; then
    cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup
fi

# Idempotent service management
systemctl is-active --quiet nginx || systemctl start nginx
systemctl is-enabled --quiet nginx || systemctl enable nginx

# Template-based configuration
cat > /etc/nginx/site.conf << EOF
server {
    listen ${PORT:-80};
    server_name ${SERVER_NAME:-localhost};
    root ${DOCUMENT_ROOT:-/var/www/html};
}
EOF

# State checking
check_service() {
    if systemctl is-active --quiet "$1"; then
        echo "Service $1 is running"
        return 0
    else
        echo "Service $1 is not running"
        return 1
    fi
}

# Retry pattern
retry() {
    local max_attempts=3
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        fi
        echo "Attempt $attempt failed, retrying..."
        sleep 2
        attempt=$((attempt + 1))
    done
    return 1
}

retry apt-get update`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1818,
			Title:       "Advanced Networking",
			Description: "Master advanced network troubleshooting, routing, VPN basics, network namespaces, advanced firewall rules, and network performance tuning.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Network Troubleshooting",
					Content: `When a service goes down or network performance degrades, the ability to systematically diagnose network problems is what separates a capable system administrator from someone who restarts services and hopes for the best. Network troubleshooting is detective work: you start with symptoms (timeouts, packet loss, connection refused), form hypotheses, and use specialized tools to gather evidence at every layer of the network stack. The tools described here are your magnifying glass, fingerprint kit, and forensic lab.

**1. tcpdump: The Packet-Level Microscope**

` + "`" + `tcpdump` + "`" + ` captures network packets in real time, letting you see exactly what is happening on the wire. It is the single most powerful network debugging tool — if a packet reaches the machine, ` + "`" + `tcpdump` + "`" + ` will show it. Common usage patterns include filtering by host (` + "`" + `tcpdump host 10.0.0.1` + "`" + `), port (` + "`" + `tcpdump port 443` + "`" + `), or protocol (` + "`" + `tcpdump tcp` + "`" + `). The ` + "`" + `-w file.pcap` + "`" + ` flag writes captures to a file for later analysis in Wireshark (a GUI tool that provides protocol-level dissection, color-coded streams, and statistical analysis). When debugging "the application cannot connect to the database," ` + "`" + `tcpdump` + "`" + ` on the database server immediately reveals whether packets are arriving, whether the TCP handshake completes, or whether the connection is being reset.

**2. ss and netstat: Connection State at a Glance**

` + "`" + `ss` + "`" + ` (socket statistics, the modern replacement for ` + "`" + `netstat` + "`" + `) shows all active network connections, listening ports, and socket states. Running ` + "`" + `ss -tuln` + "`" + ` quickly answers the question "which ports are open on this machine?" — the ` + "`" + `-t` + "`" + ` flag shows TCP, ` + "`" + `-u` + "`" + ` shows UDP, ` + "`" + `-l` + "`" + ` shows only listening sockets, and ` + "`" + `-n` + "`" + ` shows numeric ports instead of names. Adding ` + "`" + `-p` + "`" + ` reveals which process owns each socket. This is your first diagnostic step for "why can I not connect to port 8080?" — either the port is not listening, or it is bound to localhost instead of all interfaces.

**3. traceroute and mtr: Path Analysis**

` + "`" + `traceroute` + "`" + ` maps the network path from your machine to a destination by sending packets with incrementally increasing TTL values and recording which routers respond. This reveals where packets are being dropped or delayed along the route. ` + "`" + `mtr` + "`" + ` (My Traceroute) combines traceroute and ping into a continuously updating display that shows packet loss and latency at every hop. It is the go-to tool for diagnosing intermittent connectivity issues — a high packet-loss percentage at a specific hop points to the problematic link or router.

**4. A Systematic Troubleshooting Methodology**

Start from the bottom of the OSI model and work up: (1) Physical/Link layer — is the interface up? (` + "`" + `ip link show` + "`" + `). (2) Network layer — do you have an IP address? Can you ping the gateway? (3) Transport layer — is the port listening? Can you connect with ` + "`" + `nc` + "`" + ` or ` + "`" + `telnet` + "`" + `? (4) Application layer — does the service respond correctly? Each layer builds on the previous one, so testing in order quickly isolates where the problem lies.`,
					CodeExamples: `# Packet capture
sudo tcpdump -i eth0

# Show connections
ss -tuln
netstat -tuln

# Trace route
traceroute google.com
mtr google.com`,
				},
				{
					Title: "Routing Configuration",
					Content: `Advanced routing enables complex network topologies and traffic control.

**Routing Concepts:**
- **Default route**: Route for unknown destinations
- **Static routes**: Manually configured routes
- **Dynamic routes**: Learned via routing protocols
- **Routing table**: List of routes
- **Policy routing**: Route based on rules

**Routing Commands:**
- ip route: Show/manage routes
- route: Legacy routing commands
- ip rule: Policy routing rules
- ip route show table: Show specific table

**Route Types:**
- **Host route**: Route to specific host
- **Network route**: Route to network
- **Default route**: 0.0.0.0/0

**Policy Routing:**
- Route based on source IP, packet mark, etc.
- Multiple routing tables
- Rules determine which table to use`,
					CodeExamples: `# View routing table
ip route
ip route show
route -n

# Add static route
sudo ip route add 192.168.2.0/24 via 192.168.1.1
sudo route add -net 192.168.2.0/24 gw 192.168.1.1

# Add default route
sudo ip route add default via 192.168.1.1
sudo route add default gw 192.168.1.1

# Delete route
sudo ip route del 192.168.2.0/24
sudo route del -net 192.168.2.0/24

# Route to specific host
sudo ip route add 10.0.0.1 via 192.168.1.1

# Policy routing - add rule
sudo ip rule add from 192.168.1.0/24 table 100
sudo ip route add default via 192.168.1.1 table 100

# View routing rules
ip rule show

# View specific routing table
ip route show table 100

# Make routes persistent
# Add to /etc/network/interfaces or use netplan/systemd-networkd`,
				},
				{
					Title: "VPN Setup",
					Content: `VPNs (Virtual Private Networks) create secure tunnels over public networks.

**VPN Types:**
- **Site-to-site**: Connect networks
- **Remote access**: Connect individual users
- **Point-to-point**: Connect two endpoints

**VPN Protocols:**
- **OpenVPN**: Flexible, widely supported
- **WireGuard**: Modern, fast, simple
- **IPsec**: Standard, complex
- **PPTP**: Legacy, insecure

**OpenVPN:**
- Uses TLS/SSL
- Flexible configuration
- Works over TCP/UDP
- Requires certificates

**WireGuard:**
- Modern VPN protocol
- Fast and efficient
- Simple configuration
- Built into Linux kernel`,
					CodeExamples: `# OpenVPN server setup
# Install
sudo apt-get install openvpn easy-rsa

# Generate certificates
make-cadir ~/openvpn-ca
cd ~/openvpn-ca
./easyrsa init-pki
./easyrsa build-ca
./easyrsa build-server-full server nopass
./easyrsa build-client-full client1 nopass

# Server config (/etc/openvpn/server.conf)
# port 1194
# proto udp
# dev tun
# ca ca.crt
# cert server.crt
# key server.key
# dh dh.pem
# server 10.8.0.0 255.255.255.0
# push "route 192.168.1.0 255.255.255.0"

# Start OpenVPN
sudo systemctl start openvpn@server
sudo systemctl enable openvpn@server

# WireGuard setup
# Install
sudo apt-get install wireguard

# Generate keys
wg genkey | tee privatekey | wg pubkey > publickey

# Server config (/etc/wireguard/wg0.conf)
# [Interface]
# PrivateKey = <server-private-key>
# Address = 10.0.0.1/24
# ListenPort = 51820
#
# [Peer]
# PublicKey = <client-public-key>
# AllowedIPs = 10.0.0.2/32

# Start WireGuard
sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0

# Check status
sudo wg show`,
				},
				{
					Title: "Network Namespaces",
					Content: `Network namespaces provide network isolation, enabling multiple network stacks.

**Namespace Concepts:**
- Isolated network stack
- Own interfaces, routing tables, firewall rules
- Useful for containers, virtualization
- Created with ip netns

**Namespace Operations:**
- ip netns add: Create namespace
- ip netns list: List namespaces
- ip netns exec: Execute command in namespace
- ip netns delete: Delete namespace

**veth Pairs:**
- Virtual Ethernet pairs
- Connect namespaces
- Like a virtual cable
- Created with ip link add type veth

**Use Cases:**
- Container networking
- Network testing
- Network isolation
- Complex topologies`,
					CodeExamples: `# Create network namespace
sudo ip netns add ns1
sudo ip netns add ns2

# List namespaces
ip netns list

# Execute command in namespace
sudo ip netns exec ns1 ip addr
sudo ip netns exec ns1 bash

# Create veth pair
sudo ip link add veth0 type veth peer name veth1

# Move veth1 to namespace
sudo ip link set veth1 netns ns1

# Configure interfaces
sudo ip addr add 10.0.1.1/24 dev veth0
sudo ip link set veth0 up

sudo ip netns exec ns1 ip addr add 10.0.1.2/24 dev veth1
sudo ip netns exec ns1 ip link set veth1 up

# Test connectivity
sudo ip netns exec ns1 ping 10.0.1.1

# Add default route in namespace
sudo ip netns exec ns1 ip route add default via 10.0.1.1

# Delete namespace
sudo ip netns delete ns1

# Bridge namespaces
sudo ip link add br0 type bridge
sudo ip link set br0 up
sudo ip link set veth0 master br0`,
				},
				{
					Title: "Advanced Firewall Rules",
					Content: `Advanced firewall rules provide fine-grained network traffic control.

**iptables Concepts:**
- **Tables**: filter, nat, mangle, raw
- **Chains**: INPUT, OUTPUT, FORWARD, PREROUTING, POSTROUTING
- **Rules**: Match conditions and actions
- **Targets**: ACCEPT, DROP, REJECT, LOG

**Common Rules:**
- Allow/deny by IP
- Port forwarding
- NAT (Network Address Translation)
- Connection tracking
- Rate limiting

**firewalld (Red Hat):**
- Higher-level firewall management
- Zones for different trust levels
- Services and ports
- Runtime and permanent rules

**ufw (Ubuntu):**
- Simple firewall frontend
- Easy to use
- Wrapper around iptables`,
					CodeExamples: `# iptables - allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Drop all other input
sudo iptables -A INPUT -j DROP

# NAT (masquerade)
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# Port forwarding
sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 192.168.1.100:80
sudo iptables -A FORWARD -p tcp -d 192.168.1.100 --dport 80 -j ACCEPT

# Rate limiting
sudo iptables -A INPUT -p tcp --dport 22 -m limit --limit 5/min -j ACCEPT

# Connection tracking
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Save rules
sudo iptables-save > /etc/iptables/rules.v4

# firewalld
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --reload

# ufw
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw enable`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1819,
			Title:       "Linux in Production",
			Description: "Learn server administration, backup strategies, disaster recovery, high availability, monitoring and alerting, and production best practices.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Backup Strategies",
					Content: `Reliable backups are critical for production systems.

**Backup Types:**
- Full backup: Complete system backup
- Incremental: Only changed files
- Differential: Changes since last full

**Backup Tools:**
- tar: Archive files
- rsync: Synchronize files
- dd: Disk cloning
- borg: Deduplicating backup`,
					CodeExamples: `# Create backup
tar -czf backup.tar.gz /path/to/backup

# Restore backup
tar -xzf backup.tar.gz

# Rsync backup
rsync -avz /source/ /backup/

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup"
SOURCE_DIR="/data"
DATE=$(date +%Y%m%d)
tar -czf $BACKUP_DIR/backup-$DATE.tar.gz $SOURCE_DIR`,
				},
				{
					Title: "Production Best Practices",
					Content: `Running Linux in production is fundamentally different from running it in development or on your personal laptop. Production systems serve real users, handle real money, and any downtime has real consequences — lost revenue, broken trust, and sometimes regulatory penalties. The best practices for production are not about any single tool; they are about building a culture and a system of habits that prevent failures and enable rapid recovery when failures inevitably occur.

**1. Patch Management: The Foundation of Security**

Unpatched software is the number one cause of security breaches. Establish a regular patching cadence — weekly for non-critical systems, immediately for critical security vulnerabilities (CVEs). Use tools like ` + "`" + `unattended-upgrades` + "`" + ` (Debian/Ubuntu) or ` + "`" + `dnf-automatic` + "`" + ` (RHEL/Fedora) to automate security updates. Always test patches in staging before applying to production. Maintain an inventory of all software versions so you can quickly assess exposure when a new vulnerability is announced. The goal is not zero risk — it is rapid, controlled response.

**2. Monitoring, Alerting, and Log Management**

You cannot fix what you cannot see. Every production system needs three layers of observability: metrics (CPU, memory, disk, network — collected by Prometheus or similar), logs (centralized with the ELK stack or Loki, not scattered across individual servers), and traces (for understanding request flow through microservices). Alerting should be actionable — every alert should require human response. If you are ignoring alerts, they are too noisy and need tuning. Use ` + "`" + `journalctl` + "`" + ` for local log investigation, but always ship logs to a central system where they survive server failures and can be searched across the fleet.

**3. Change Management and Documentation**

In production, every change is a potential incident. Implement a change management process: document what you are changing, why, what the rollback plan is, and who approved it. Use Git to version all infrastructure code and configuration. Maintain runbooks — step-by-step guides for common operations and incident response. Document the "why" behind every non-obvious configuration choice. Future-you (or the on-call engineer at 3 AM) will be grateful for a comment explaining why ` + "`" + `net.core.somaxconn` + "`" + ` was set to 65535.

**4. Disaster Recovery and Backup Verification**

Having backups is not enough — you must regularly test restoring from them. A backup that cannot be restored is not a backup; it is a false sense of security. Document your Recovery Time Objective (RTO: how long can you be down?) and Recovery Point Objective (RPO: how much data can you afford to lose?). Practice disaster recovery scenarios: what happens if the database server dies? What if the entire data center is unavailable? The organizations that recover fastest are the ones that practiced failure before it happened.

**5. Security Hardening as a Continuous Process**

Harden every system at deployment: disable unnecessary services, configure firewalls, enforce key-based SSH authentication, implement least-privilege access. Run periodic security audits with tools like ` + "`" + `lynis` + "`" + ` or ` + "`" + `oscap` + "`" + `. Enable audit logging (` + "`" + `auditd` + "`" + `) to track privileged operations. Security is not a one-time checklist — it is an ongoing practice of reducing attack surface, detecting anomalies, and responding to threats.`,
					CodeExamples: `# System updates
sudo apt update && sudo apt upgrade

# Monitor logs
sudo journalctl -f

# Check disk space
df -h

# Monitor system
htop

# Security audit
sudo apt install auditd
sudo auditctl -l`,
				},
				{
					Title: "High Availability Setup",
					Content: `High Availability (HA) ensures services remain available despite failures.

**HA Concepts:**
- **Redundancy**: Multiple components
- **Failover**: Automatic switching to backup
- **Load balancing**: Distribute load
- **Clustering**: Group of systems working together

**HA Components:**
- **Pacemaker**: Cluster resource manager
- **Corosync**: Cluster communication
- **HAProxy**: Load balancer
- **keepalived**: Virtual IP failover

**HAProxy:**
- TCP/HTTP load balancer
- Health checks
- Multiple algorithms (round-robin, leastconn)
- SSL termination

**keepalived:**
- Provides virtual IP (VIP)
- VRRP protocol
- Automatic failover
- Health checking

**Pacemaker/Corosync:**
- Cluster resource management
- Resource monitoring
- Automatic failover
- Fencing (STONITH)`,
					CodeExamples: `# HAProxy configuration (/etc/haproxy/haproxy.cfg)
# global
#     daemon
#
# defaults
#     mode http
#     timeout connect 5000ms
#     timeout client 50000ms
#     timeout server 50000ms
#
# frontend http_front
#     bind *:80
#     default_backend http_back
#
# backend http_back
#     balance roundrobin
#     server web1 192.168.1.10:80 check
#     server web2 192.168.1.11:80 check

# Start HAProxy
sudo systemctl start haproxy
sudo systemctl enable haproxy

# keepalived configuration (/etc/keepalived/keepalived.conf)
# vrrp_script chk_haproxy {
#     script "/usr/bin/killall -0 haproxy"
#     interval 2
# }
#
# vrrp_instance VI_1 {
#     state MASTER
#     interface eth0
#     virtual_router_id 51
#     priority 100
#     virtual_ipaddress {
#         192.168.1.100
#     }
#     track_script {
#         chk_haproxy
#     }
# }

# Start keepalived
sudo systemctl start keepalived
sudo systemctl enable keepalived

# Pacemaker commands
sudo pcs cluster setup --name mycluster node1 node2
sudo pcs cluster start --all
sudo pcs resource create webserver ocf:heartbeat:apache
sudo pcs resource create vip IPaddr2 ip=192.168.1.100`,
				},
				{
					Title: "Monitoring & Alerting",
					Content: `Monitoring and alerting are essential for production systems.

**Monitoring Tools:**

**Prometheus:**
- Time-series database
- Pull-based metrics
- PromQL query language
- Service discovery

**Grafana:**
- Visualization and dashboards
- Connects to Prometheus
- Alerting rules
- Beautiful graphs

**Nagios:**
- Classic monitoring system
- Plugin-based
- Alerting and notifications
- Service and host monitoring

**Monitoring Metrics:**
- **System**: CPU, memory, disk, network
- **Application**: Response time, errors, throughput
- **Business**: Transactions, users, revenue

**Alerting:**
- Threshold-based alerts
- Alert channels (email, Slack, PagerDuty)
- Alert escalation
- Alert suppression`,
					CodeExamples: `# Prometheus configuration (prometheus.yml)
# global:
#   scrape_interval: 15s
#
# scrape_configs:
#   - job_name: 'node'
#     static_configs:
#       - targets: ['localhost:9100']

# Start Prometheus
./prometheus --config.file=prometheus.yml

# Node exporter (system metrics)
./node_exporter

# Grafana
# Install and start Grafana
sudo systemctl start grafana-server
# Access at http://localhost:3000
# Default login: admin/admin

# Nagios configuration
# Define host
# define host {
#     host_name web1
#     address 192.168.1.10
#     check_command check-host-alive
# }
#
# Define service
# define service {
#     host_name web1
#     service_description HTTP
#     check_command check_http
# }

# Simple monitoring script
#!/bin/bash
CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
if (( $(echo "$CPU > 80" | bc -l) )); then
    echo "High CPU: $CPU%" | mail -s "Alert" admin@example.com
fi

# Prometheus alerting rules
# groups:
#   - name: alerts
#     rules:
#       - alert: HighCPU
#         expr: cpu_usage > 80
#         for: 5m
#         annotations:
#           summary: "High CPU usage"`,
				},
				{
					Title: "Log Aggregation",
					Content: `Centralized log aggregation enables efficient log management and analysis.

**ELK Stack:**
- **Elasticsearch**: Search and analytics engine
- **Logstash**: Log processing pipeline
- **Kibana**: Visualization and dashboards

**Alternative Stack:**
- **Fluentd**: Log collector
- **Elasticsearch**: Storage
- **Kibana**: Visualization

**rsyslog:**
- System logging daemon
- Can forward logs
- Filtering and processing
- Configuration: /etc/rsyslog.conf

**Centralized Logging:**
- Collect logs from multiple sources
- Central storage
- Search and analysis
- Alerting on patterns

**Log Shipping:**
- Agents on servers send logs
- Central server receives
- Process and store
- Make searchable`,
					CodeExamples: `# rsyslog configuration (/etc/rsyslog.conf)
# Forward logs to central server
# *.* @192.168.1.100:514

# Restart rsyslog
sudo systemctl restart rsyslog

# Logstash configuration (logstash.conf)
# input {
#     syslog {
#         port => 514
#     }
# }
# filter {
#     grok {
#         match => { "message" => "%{SYSLOGLINE}" }
#     }
# }
# output {
#     elasticsearch {
#         hosts => ["localhost:9200"]
#     }
# }

# Start Logstash
./logstash -f logstash.conf

# Elasticsearch
# Start Elasticsearch
./elasticsearch

# Query Elasticsearch
curl 'localhost:9200/_search?q=error'

# Fluentd configuration
# <source>
#   @type syslog
#   port 514
# </source>
# <match **>
#   @type elasticsearch
#   host localhost
#   port 9200
# </match>

# Centralized logging script
#!/bin/bash
# Send log to central server
logger -n 192.168.1.100 -P 514 "Application log message"

# Log aggregation with journalctl
journalctl --since "1 hour ago" | \
    grep -i error | \
    ssh central-server "cat >> /var/log/aggregated.log"`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
