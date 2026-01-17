package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          130,
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
- `ps -o pid,pgid,comm` shows process groups`,
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
\`\`\`
* * * * * command
│ │ │ │ │
│ │ │ │ └── Day of week (0-7, 0 or 7 = Sunday)
│ │ │ └──── Month (1-12)
│ │ └────── Day of month (1-31)
│ └──────── Hour (0-23)
└────────── Minute (0-59)
\`\`\`

**Crontab:**
- `crontab -e`: Edit crontab
- `crontab -l`: List crontab
- `crontab -r`: Remove crontab`,
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
			},
			ProblemIDs: []int{},
		},
		{
			ID:          131,
			Title:       "File System Advanced",
			Description: "Learn mount points, fstab, disk management, filesystem types, LVM basics, and RAID concepts.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Mount Points & fstab",
					Content: `Mounting makes filesystems accessible. /etc/fstab defines filesystems to mount at boot.

**Mount Commands:**
- `mount`: Mount filesystem
- `umount`: Unmount filesystem
- `mount -a`: Mount all in fstab
- `df`: Show filesystem usage
- `du`: Show directory usage

**fstab Format:**
\`\`\`
device mountpoint fstype options dump pass
\`\`\``,
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
- `fdisk`: Partition table manipulator
- `parted`: Advanced partitioning
- `lsblk`: List block devices
- `blkid`: Show block device attributes

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
			},
			ProblemIDs: []int{},
		},
		{
			ID:          132,
			Title:       "System Monitoring & Performance",
			Description: "Master monitoring tools, performance tuning, resource limits, memory management, and CPU scheduling.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Monitoring Tools",
					Content: `Monitoring system performance helps identify bottlenecks and issues.

**Monitoring Commands:**
- `top`/`htop`: Process monitor
- `iotop`: I/O monitoring
- `nethogs`: Network usage by process
- `free`: Memory usage
- `vmstat`: Virtual memory statistics
- `iostat`: I/O statistics`,
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
					Content: `ulimit controls resource limits for processes.

**Common Limits:**
- `ulimit -a`: Show all limits
- `ulimit -n`: File descriptors
- `ulimit -u`: Processes
- `ulimit -v`: Virtual memory`,
					CodeExamples: `# Show limits
ulimit -a

# Set limit
ulimit -n 4096

# Make permanent (edit /etc/security/limits.conf)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          133,
			Title:       "Security & Hardening",
			Description: "Learn security best practices, SSH hardening, firewall configuration, SELinux/AppArmor, encryption, and security auditing.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "SSH Hardening",
					Content: `Hardening SSH improves security significantly.

**SSH Hardening Steps:**
- Disable root login
- Use key-based authentication
- Change default port
- Disable password authentication
- Use fail2ban`,
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
					Content: `Firewalls control network traffic. iptables and firewalld are common tools.

**firewalld (Red Hat/Fedora):**
- `firewall-cmd`: Configure firewall
- `firewall-cmd --list-all`: Show rules
- `firewall-cmd --add-service`: Add service
- `firewall-cmd --reload`: Reload rules`,
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
			ID:          134,
			Title:       "Shell Scripting Advanced",
			Description: "Master advanced bash features, regular expressions, advanced text processing, script optimization, and error handling patterns.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Bash Features",
					Content: `Advanced bash features make scripts more powerful and efficient.

**Features:**
- Parameter expansion
- Process substitution
- Here documents
- Regular expressions
- Advanced I/O redirection`,
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
			},
			ProblemIDs: []int{},
		},
		{
			ID:          135,
			Title:       "System Programming Basics",
			Description: "Introduction to system calls, file I/O programming, process creation (fork/exec), signals programming, and IPC basics.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "System Calls Overview",
					Content: `System calls are interfaces between user programs and the kernel.

**Common System Calls:**
- `open`, `read`, `write`, `close`: File operations
- `fork`, `exec`: Process creation
- `wait`: Wait for child process
- `kill`: Send signal
- `pipe`: Create pipe`,
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
			},
			ProblemIDs: []int{},
		},
		{
			ID:          136,
			Title:       "Kernel & System Internals",
			Description: "Explore kernel architecture, kernel modules, /proc and /sys filesystems, boot process, and init systems.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Kernel Modules",
					Content: `Kernel modules extend kernel functionality without rebooting.

**Module Commands:**
- `lsmod`: List loaded modules
- `modinfo`: Show module info
- `modprobe`: Load/unload modules
- `insmod`/`rmmod`: Low-level module operations`,
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
- `/proc/cpuinfo`: CPU information
- `/proc/meminfo`: Memory information
- `/proc/version`: Kernel version

**/sys:**
- Kernel and device information
- `/sys/class`: Device classes
- `/sys/devices`: Physical devices`,
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
			ID:          137,
			Title:       "DevOps Tools & Automation",
			Description: "Learn Git basics, CI/CD concepts, configuration management (Ansible), container basics (Docker), and automation scripting.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Git Basics",
					Content: `Git is essential for version control and DevOps workflows.

**Basic Git Commands:**
- `git init`: Initialize repository
- `git add`: Stage changes
- `git commit`: Commit changes
- `git status`: Show status
- `git log`: Show history
- `git clone`: Clone repository`,
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
					Content: `Docker containers package applications with dependencies.

**Docker Commands:**
- `docker run`: Run container
- `docker ps`: List containers
- `docker images`: List images
- `docker build`: Build image
- `docker exec`: Execute in container`,
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
			},
			ProblemIDs: []int{},
		},
		{
			ID:          138,
			Title:       "Advanced Networking",
			Description: "Master advanced network troubleshooting, routing, VPN basics, network namespaces, advanced firewall rules, and network performance tuning.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Network Troubleshooting",
					Content: `Advanced network troubleshooting skills are essential for system administrators.

**Troubleshooting Tools:**
- `tcpdump`: Packet capture
- `wireshark`: GUI packet analyzer
- `netstat`/`ss`: Connection info
- `traceroute`: Trace network path
- `mtr`: Network diagnostic tool`,
					CodeExamples: `# Packet capture
sudo tcpdump -i eth0

# Show connections
ss -tuln
netstat -tuln

# Trace route
traceroute google.com
mtr google.com`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          139,
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
- `tar`: Archive files
- `rsync`: Synchronize files
- `dd`: Disk cloning
- `borg`: Deduplicating backup`,
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
					Content: `Production systems require careful planning and maintenance.

**Best Practices:**
- Regular updates and patches
- Monitoring and alerting
- Log management
- Security hardening
- Documentation
- Disaster recovery plan
- Change management`,
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
			},
			ProblemIDs: []int{},
		},
	})
}
