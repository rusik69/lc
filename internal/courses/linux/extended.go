package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          140,
			Title:       "Boot Process & System Initialization",
			Description: "Understand the Linux boot process from BIOS/UEFI through GRUB to systemd, kernel parameters, initramfs, and boot troubleshooting.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "The Linux Boot Sequence",
					Content: `Understanding the boot process is essential for troubleshooting systems that fail to start. The sequence has distinct stages.

**1. Firmware (BIOS/UEFI):**
*   **BIOS (Legacy):** Reads the first 512 bytes of the boot device (MBR). Limited to 2TB disks and 4 primary partitions.
*   **UEFI (Modern):** Reads the EFI System Partition (ESP, typically /boot/efi). Supports GPT, Secure Boot, and larger disks.
*   Firmware performs POST (Power-On Self-Test), then hands off to the bootloader.

**2. Bootloader (GRUB2):**
*   GRUB2 is the standard Linux bootloader.
*   Configuration: ` + "`" + `/boot/grub/grub.cfg` + "`" + ` (auto-generated from ` + "`" + `/etc/default/grub` + "`" + `).
*   Presents a menu of available kernels. Default timeout is configurable.
*   Loads the kernel and initramfs into memory.

**3. Kernel Initialization:**
*   Kernel decompresses itself into memory.
*   Initializes hardware, mounts the initramfs as a temporary root filesystem.
*   initramfs contains essential drivers (disk, filesystem, LVM, RAID) needed to mount the real root filesystem.
*   Once the real root is mounted, ` + "`" + `/sbin/init` + "`" + ` (systemd) is launched as PID 1.

**4. systemd (PID 1):**
*   Reads the default target (usually ` + "`" + `graphical.target` + "`" + ` or ` + "`" + `multi-user.target` + "`" + `).
*   Starts services in parallel based on dependency graphs.
*   Mounts filesystems from ` + "`" + `/etc/fstab` + "`" + `.
*   Brings up networking, logging, and user services.`,
					CodeExamples: `# View current boot target
systemctl get-default
# multi-user.target

# Change default target
sudo systemctl set-default multi-user.target  # No GUI
sudo systemctl set-default graphical.target    # GUI

# View GRUB configuration
cat /etc/default/grub
# GRUB_DEFAULT=0
# GRUB_TIMEOUT=5
# GRUB_CMDLINE_LINUX="quiet splash"

# Add kernel parameters (e.g., disable IPv6)
sudo vi /etc/default/grub
# GRUB_CMDLINE_LINUX="quiet splash ipv6.disable=1"
sudo update-grub   # Debian/Ubuntu
sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # RHEL/CentOS

# View kernel boot parameters
cat /proc/cmdline
# BOOT_IMAGE=/vmlinuz-5.15.0 root=/dev/sda1 ro quiet splash

# Regenerate initramfs
sudo update-initramfs -u       # Debian/Ubuntu
sudo dracut --force            # RHEL/CentOS

# View boot log
journalctl -b       # Current boot
journalctl -b -1    # Previous boot
journalctl --list-boots  # List all recorded boots`,
				},
				{
					Title: "systemd Service Management",
					Content: `systemd is the init system and service manager for modern Linux distributions. It manages the lifecycle of all system services (daemons).

**1. Units:**
*   Everything in systemd is a "unit." Types include:
    *   ` + "`" + `.service` + "`" + ` -- Daemons (nginx, sshd, postgresql).
    *   ` + "`" + `.socket` + "`" + ` -- Socket-based activation.
    *   ` + "`" + `.timer` + "`" + ` -- Scheduled tasks (replaces cron).
    *   ` + "`" + `.mount` + "`" + ` / ` + "`" + `.automount` + "`" + ` -- Filesystem mounts.
    *   ` + "`" + `.target` + "`" + ` -- Groups of units (like runlevels).
*   Unit files location: ` + "`" + `/usr/lib/systemd/system/` + "`" + ` (package defaults), ` + "`" + `/etc/systemd/system/` + "`" + ` (admin overrides).

**2. Service Commands:**
*   ` + "`" + `systemctl start|stop|restart|reload <unit>` + "`" + ` -- Control service state.
*   ` + "`" + `systemctl enable|disable <unit>` + "`" + ` -- Control auto-start on boot.
*   ` + "`" + `systemctl status <unit>` + "`" + ` -- View current state, recent logs, PID.
*   ` + "`" + `systemctl is-active|is-enabled <unit>` + "`" + ` -- Scripting-friendly checks.

**3. Writing Custom Services:**
*   ` + "`" + `[Unit]` + "`" + ` section: Description, dependencies (After=, Requires=, Wants=).
*   ` + "`" + `[Service]` + "`" + ` section: ExecStart, Restart policy, User, WorkingDirectory.
*   ` + "`" + `[Install]` + "`" + ` section: WantedBy (which target activates this service).

**4. Restart Policies:**
*   ` + "`" + `Restart=always` + "`" + ` -- Restart on any exit (including success). Good for daemons.
*   ` + "`" + `Restart=on-failure` + "`" + ` -- Only restart on non-zero exit code.
*   ` + "`" + `RestartSec=5` + "`" + ` -- Wait 5 seconds between restarts.
*   ` + "`" + `StartLimitBurst=5` + "`" + ` / ` + "`" + `StartLimitIntervalSec=60` + "`" + ` -- Max 5 restarts per minute.`,
					CodeExamples: `# Custom service file: /etc/systemd/system/myapp.service
[Unit]
Description=My Application Server
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=myapp
Group=myapp
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/server --config /etc/myapp/config.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5
StartLimitBurst=5
StartLimitIntervalSec=60
Environment=GOMAXPROCS=4
EnvironmentFile=/etc/myapp/env

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/myapp /var/log/myapp

[Install]
WantedBy=multi-user.target

# Deploy and manage the service
sudo systemctl daemon-reload
sudo systemctl enable --now myapp.service
sudo systemctl status myapp

# View logs for the service
journalctl -u myapp.service -f         # Follow logs
journalctl -u myapp.service --since "1 hour ago"

# Override specific settings without editing the unit file
sudo systemctl edit myapp.service
# Creates /etc/systemd/system/myapp.service.d/override.conf`,
				},
				{
					Title: "Boot Troubleshooting and Recovery",
					Content: `When a Linux system fails to boot, you need to diagnose which stage is failing and use the appropriate recovery method.

**1. GRUB Recovery:**
*   If GRUB menu appears: Press ` + "`" + `e` + "`" + ` to edit boot parameters temporarily.
*   Add ` + "`" + `single` + "`" + ` or ` + "`" + `init=/bin/bash` + "`" + ` to kernel line for single-user mode.
*   If GRUB is broken: Boot from live USB, mount root partition, ` + "`" + `chroot` + "`" + `, reinstall GRUB.

**2. Emergency and Rescue Modes:**
*   ` + "`" + `systemctl rescue` + "`" + ` or add ` + "`" + `systemd.unit=rescue.target` + "`" + ` to kernel args -- minimal single-user mode with root filesystem mounted.
*   ` + "`" + `systemctl emergency` + "`" + ` or add ` + "`" + `systemd.unit=emergency.target` + "`" + ` -- even more minimal, root mounted read-only.

**3. Common Boot Failures:**
*   **Kernel panic:** Missing driver, corrupted initramfs. Fix: Boot older kernel from GRUB, regenerate initramfs.
*   **fsck errors:** Filesystem corruption. Fix: Boot to rescue mode, run ` + "`" + `fsck` + "`" + ` manually.
*   **fstab errors:** Invalid entry prevents mounting. Fix: Boot with ` + "`" + `init=/bin/bash` + "`" + `, fix ` + "`" + `/etc/fstab` + "`" + `.
*   **Service loops:** A service crashes and restarts repeatedly. Fix: ` + "`" + `systemctl mask <service>` + "`" + ` from rescue mode.

**4. Live USB Recovery:**
1. Boot from live USB/ISO.
2. Mount the root partition: ` + "`" + `mount /dev/sda2 /mnt` + "`" + `.
3. Mount boot: ` + "`" + `mount /dev/sda1 /mnt/boot` + "`" + `.
4. Mount pseudo-filesystems: ` + "`" + `mount --bind /dev /mnt/dev` + "`" + `, etc.
5. ` + "`" + `chroot /mnt` + "`" + ` -- Now you're "inside" the broken system.
6. Fix the problem (reinstall GRUB, fix fstab, update initramfs).`,
					CodeExamples: `# GRUB: Edit kernel parameters at boot
# Press 'e' at GRUB menu, find the line starting with 'linux'
# Add one of these to the end:
#   single            -- Single-user mode
#   init=/bin/bash    -- Drop to bash (no networking)
#   rd.break          -- Break into initramfs (before root mount)

# Live USB recovery workflow
sudo mount /dev/sda2 /mnt
sudo mount /dev/sda1 /mnt/boot
sudo mount --bind /dev /mnt/dev
sudo mount --bind /proc /mnt/proc
sudo mount --bind /sys /mnt/sys
sudo chroot /mnt

# Inside chroot: fix GRUB
grub-install /dev/sda
update-grub

# Inside chroot: fix fstab
vi /etc/fstab  # Fix the bad entry

# Inside chroot: regenerate initramfs
update-initramfs -u

# Exit chroot and reboot
exit
sudo umount -R /mnt
sudo reboot

# Check for failed services after boot
systemctl --failed

# Mask a problematic service (prevents it from starting)
sudo systemctl mask problematic-service.service

# View boot timing (find slow services)
systemd-analyze
# Startup finished in 3.5s (kernel) + 8.2s (userspace) = 11.7s

systemd-analyze blame   # List services by startup time
systemd-analyze critical-chain  # Show critical path`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          141,
			Title:       "Log Management & Monitoring",
			Description: "Master Linux logging with journald, syslog, logrotate, and centralized log collection for production systems.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "journald and journalctl",
					Content: `systemd-journald is the default logging system on modern Linux distributions. It collects logs from the kernel, services, and applications into a structured binary format.

**1. Journal Storage:**
*   Binary format stored in ` + "`" + `/var/log/journal/` + "`" + ` (persistent) or ` + "`" + `/run/log/journal/` + "`" + ` (volatile, lost on reboot).
*   To make persistent: ` + "`" + `sudo mkdir -p /var/log/journal` + "`" + ` and restart journald.
*   Configuration: ` + "`" + `/etc/systemd/journald.conf` + "`" + `.
*   Maximum size: ` + "`" + `SystemMaxUse=` + "`" + ` (default: 10% of filesystem, max 4GB).

**2. journalctl Querying:**
*   ` + "`" + `journalctl` + "`" + ` -- All logs.
*   ` + "`" + `journalctl -u <service>` + "`" + ` -- Logs for a specific service.
*   ` + "`" + `journalctl -f` + "`" + ` -- Follow (like tail -f).
*   ` + "`" + `journalctl -b` + "`" + ` -- Current boot only.
*   ` + "`" + `journalctl -p err` + "`" + ` -- Only errors and above (emerg, alert, crit, err).
*   ` + "`" + `journalctl --since "2024-01-01" --until "2024-01-02"` + "`" + ` -- Time range.
*   ` + "`" + `journalctl _UID=1000` + "`" + ` -- Logs from a specific user.
*   ` + "`" + `journalctl -o json-pretty` + "`" + ` -- JSON output for parsing.

**3. Priority Levels (syslog standard):**
*   0 = emerg, 1 = alert, 2 = crit, 3 = err, 4 = warning, 5 = notice, 6 = info, 7 = debug.
*   ` + "`" + `journalctl -p 0..3` + "`" + ` -- Show only emerg through err.

**4. Journal Maintenance:**
*   ` + "`" + `journalctl --disk-usage` + "`" + ` -- Check journal size.
*   ` + "`" + `journalctl --vacuum-size=500M` + "`" + ` -- Reduce to 500MB.
*   ` + "`" + `journalctl --vacuum-time=30d` + "`" + ` -- Remove entries older than 30 days.`,
					CodeExamples: `# Basic queries
journalctl -u nginx.service          # Nginx logs only
journalctl -u nginx -u php-fpm       # Multiple services
journalctl -f -u myapp               # Follow in real-time
journalctl -b -1 -p err              # Errors from previous boot

# Time-based filtering
journalctl --since "2024-01-15 09:00" --until "2024-01-15 18:00"
journalctl --since "1 hour ago"
journalctl --since today

# Kernel messages
journalctl -k                        # Kernel messages (like dmesg)
journalctl -k -p err                 # Kernel errors only

# Output formats
journalctl -u nginx -o json-pretty   # JSON format
journalctl -u nginx -o short-iso     # ISO timestamps
journalctl -u nginx --no-pager       # Don't pipe through pager

# Count messages by priority
journalctl -p err --no-pager | wc -l

# Disk usage and maintenance
journalctl --disk-usage
# Archived and active journals take up 1.2G on disk
sudo journalctl --vacuum-size=500M
sudo journalctl --vacuum-time=14d

# Configuration: /etc/systemd/journald.conf
# [Journal]
# Storage=persistent
# SystemMaxUse=2G
# SystemMaxFileSize=100M
# MaxRetentionSec=30day
# Compress=yes`,
				},
				{
					Title: "Traditional Syslog and Log Files",
					Content: `While journald is the primary logging system, many applications and tools still use traditional log files in ` + "`" + `/var/log/` + "`" + `. Understanding both systems is essential.

**1. Key Log Files:**
*   ` + "`" + `/var/log/syslog` + "`" + ` (Debian) or ` + "`" + `/var/log/messages` + "`" + ` (RHEL) -- General system messages.
*   ` + "`" + `/var/log/auth.log` + "`" + ` or ` + "`" + `/var/log/secure` + "`" + ` -- Authentication logs (SSH, sudo, PAM).
*   ` + "`" + `/var/log/kern.log` + "`" + ` -- Kernel messages.
*   ` + "`" + `/var/log/dmesg` + "`" + ` -- Hardware detection at boot.
*   ` + "`" + `/var/log/apt/` + "`" + ` or ` + "`" + `/var/log/dnf.log` + "`" + ` -- Package manager logs.
*   ` + "`" + `/var/log/nginx/` + "`" + ` -- Application-specific logs.

**2. rsyslog:**
*   Forwards journald messages to traditional log files.
*   Configuration: ` + "`" + `/etc/rsyslog.conf` + "`" + ` and ` + "`" + `/etc/rsyslog.d/` + "`" + `.
*   Can forward logs to remote syslog servers.
*   Uses facility.priority notation: ` + "`" + `auth.warning /var/log/auth.log` + "`" + `.

**3. Facilities:**
*   ` + "`" + `auth` + "`" + ` -- Authentication. ` + "`" + `cron` + "`" + ` -- Cron jobs. ` + "`" + `daemon` + "`" + ` -- System daemons.
*   ` + "`" + `kern` + "`" + ` -- Kernel. ` + "`" + `mail` + "`" + ` -- Mail system. ` + "`" + `local0-local7` + "`" + ` -- Custom use.

**4. Analyzing Logs:**
*   ` + "`" + `grep "error" /var/log/syslog` + "`" + ` -- Search for errors.
*   ` + "`" + `tail -f /var/log/auth.log` + "`" + ` -- Watch authentication events.
*   ` + "`" + `zgrep "pattern" /var/log/syslog.1.gz` + "`" + ` -- Search compressed old logs.
*   ` + "`" + `last` + "`" + ` -- Show recent logins. ` + "`" + `lastb` + "`" + ` -- Show failed logins.`,
					CodeExamples: `# Check important log files
tail -100 /var/log/syslog          # Last 100 lines of syslog
tail -f /var/log/auth.log          # Watch auth events in real-time
grep "Failed password" /var/log/auth.log | tail -20  # Failed SSH logins

# Search across all log files
grep -r "error" /var/log/ --include="*.log" 2>/dev/null | head -50

# Recent logins
last -10                           # Last 10 logins
lastb -10                          # Last 10 failed logins
who                                # Currently logged-in users

# rsyslog configuration example
# /etc/rsyslog.d/50-myapp.conf
# Send myapp logs to a dedicated file
if $programname == 'myapp' then /var/log/myapp.log
& stop

# Forward to remote syslog server
# *.* @@logserver.example.com:514  # TCP
# *.* @logserver.example.com:514   # UDP

# Check which processes have log files open
lsof +D /var/log/ 2>/dev/null | head -20

# Compressed log search
zgrep "kernel panic" /var/log/syslog.*.gz
zcat /var/log/syslog.1.gz | grep "error"`,
				},
				{
					Title: "logrotate and Log Maintenance",
					Content: `logrotate prevents log files from consuming all disk space by rotating, compressing, and deleting old logs automatically.

**1. How It Works:**
*   Runs daily via cron or systemd timer (` + "`" + `/etc/cron.daily/logrotate` + "`" + `).
*   Main config: ` + "`" + `/etc/logrotate.conf` + "`" + `.
*   Application-specific configs: ` + "`" + `/etc/logrotate.d/` + "`" + ` (one file per application).

**2. Key Directives:**
*   ` + "`" + `daily` + "`" + ` / ` + "`" + `weekly` + "`" + ` / ` + "`" + `monthly` + "`" + ` -- Rotation frequency.
*   ` + "`" + `rotate 7` + "`" + ` -- Keep 7 old files before deleting.
*   ` + "`" + `compress` + "`" + ` -- gzip compress old logs.
*   ` + "`" + `delaycompress` + "`" + ` -- Don't compress the most recent rotated file (some apps keep writing to it briefly).
*   ` + "`" + `maxsize 100M` + "`" + ` -- Rotate when file exceeds 100MB regardless of time.
*   ` + "`" + `missingok` + "`" + ` -- Don't error if the log file doesn't exist.
*   ` + "`" + `notifempty` + "`" + ` -- Don't rotate if the file is empty.
*   ` + "`" + `copytruncate` + "`" + ` -- Copy the file, then truncate the original (for apps that can't reopen files).
*   ` + "`" + `postrotate/endscript` + "`" + ` -- Run a command after rotation (e.g., reload nginx).

**3. Rotation Strategies:**
*   **Create and rename:** Default. Renames ` + "`" + `app.log` + "`" + ` to ` + "`" + `app.log.1` + "`" + `, creates new ` + "`" + `app.log` + "`" + `. Requires app to handle SIGHUP.
*   **copytruncate:** Copies the file, then truncates. No signal needed but may lose lines written during copy.

**4. Testing:**
*   ` + "`" + `logrotate -d /etc/logrotate.d/myapp` + "`" + ` -- Dry run (shows what would happen).
*   ` + "`" + `logrotate -f /etc/logrotate.d/myapp` + "`" + ` -- Force rotation now.`,
					CodeExamples: `# Custom logrotate config: /etc/logrotate.d/myapp
/var/log/myapp/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    maxsize 100M
    create 0640 myapp myapp
    postrotate
        systemctl reload myapp > /dev/null 2>&1 || true
    endscript
}

# Nginx logrotate config (typical)
/var/log/nginx/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 $(cat /var/run/nginx.pid)
    endscript
}

# Test configuration (dry run)
sudo logrotate -d /etc/logrotate.d/myapp
# Shows what would happen without making changes

# Force immediate rotation
sudo logrotate -f /etc/logrotate.d/myapp

# Check logrotate state (last rotation timestamps)
cat /var/lib/logrotate/status`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          142,
			Title:       "Memory Management & Performance Tuning",
			Description: "Understand Linux memory management, swap, the OOM killer, hugepages, and performance tuning with sysctl and kernel parameters.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Memory Architecture and Tools",
					Content: `Linux memory management is a complex subsystem that manages physical RAM, virtual memory, caching, and swap. Understanding it is crucial for diagnosing performance issues.

**1. Virtual Memory:**
*   Every process has its own virtual address space. The kernel maps virtual addresses to physical RAM pages (typically 4KB).
*   This provides isolation (processes can't access each other's memory) and allows more memory than physically available (via swap).

**2. Memory Types:**
*   **Used:** Memory actively used by applications.
*   **Buffers:** Metadata cache for filesystems (inode tables, directory listings).
*   **Cached (Page Cache):** File data cached in RAM for faster access. This is FREE memory that the kernel will release when applications need it.
*   **Available:** Memory that can be given to applications (free + reclaimable cache). This is the number you should monitor, not "free."

**3. Key Tools:**
*   ` + "`" + `free -h` + "`" + ` -- Quick overview of memory usage.
*   ` + "`" + `/proc/meminfo` + "`" + ` -- Detailed memory statistics (30+ fields).
*   ` + "`" + `vmstat 1` + "`" + ` -- Virtual memory statistics, updated every second.
*   ` + "`" + `top` + "`" + ` / ` + "`" + `htop` + "`" + ` -- Per-process memory usage (RES = physical RAM, VIRT = virtual).
*   ` + "`" + `smem` + "`" + ` -- Proportional set size (accounts for shared memory correctly).

**4. Common Misconception:**
*   A system showing "low free memory" is NOT necessarily running out of memory.
*   Linux aggressively caches disk data in unused RAM. This is good -- it speeds up disk access.
*   The "available" column in ` + "`" + `free` + "`" + ` is the correct metric to check.`,
					CodeExamples: `# Quick memory overview
free -h
#               total        used        free      shared  buff/cache   available
# Mem:           15Gi       4.2Gi       1.1Gi       350Mi       9.9Gi       10Gi
# Swap:          4.0Gi          0B       4.0Gi

# Detailed memory info
cat /proc/meminfo | head -15
# MemTotal:       16384000 kB
# MemFree:         1126400 kB
# MemAvailable:   10547200 kB   <-- THIS is what matters
# Buffers:          204800 kB
# Cached:          9420800 kB

# Virtual memory statistics (1 second intervals)
vmstat 1 5
# procs  memory        swap    io       system  cpu
# r b   swpd  free   si so   bi  bo    in  cs  us sy id
# 1 0    0  1126400  0  0   12   8   150 300  5  2 93

# Per-process memory usage (sorted by RSS)
ps aux --sort=-rss | head -10

# Check specific process memory map
pmap -x $(pidof nginx)

# System memory pressure
cat /proc/pressure/memory
# some avg10=0.00 avg60=0.00 avg300=0.00 total=0`,
				},
				{
					Title: "Swap and the OOM Killer",
					Content: `Swap extends virtual memory to disk. The OOM (Out of Memory) killer is the kernel's last resort when the system truly runs out of memory.

**1. Swap Types:**
*   **Swap partition:** Dedicated partition, slightly faster.
*   **Swap file:** A file on the filesystem, more flexible (can resize easily).
*   Check: ` + "`" + `swapon --show` + "`" + `, ` + "`" + `free -h` + "`" + `.

**2. Swappiness:**
*   ` + "`" + `vm.swappiness` + "`" + ` (0-200, default 60) controls how aggressively the kernel moves memory pages to swap.
*   0 = Only swap when absolutely necessary (avoid for desktops).
*   10-30 = Recommended for databases and latency-sensitive workloads (keep more data in RAM).
*   60 = Default (balanced).
*   100+ = Aggressively swap (useful when swap is on fast NVMe).

**3. The OOM Killer:**
*   When the system is critically low on memory AND swap, the kernel invokes the OOM killer.
*   It selects a process to kill based on ` + "`" + `oom_score` + "`" + ` (higher = more likely to be killed).
*   Factors: Memory usage, CPU time, nice value, whether it's root.
*   ` + "`" + `/proc/<pid>/oom_score_adj` + "`" + ` -- Adjust OOM score (-1000 to 1000). -1000 = never kill. 1000 = always kill first.

**4. Monitoring OOM Events:**
*   ` + "`" + `dmesg | grep -i "out of memory"` + "`" + ` -- Check for OOM kills.
*   ` + "`" + `journalctl -k | grep -i oom` + "`" + ` -- Same via journald.
*   OOM kills indicate the system needs more RAM, better resource limits, or memory leak fixes.

**5. Best Practices:**
*   Always have some swap (even on servers with lots of RAM). It prevents immediate OOM kills.
*   For databases: Set ` + "`" + `swappiness=10` + "`" + ` and pin working set in RAM.
*   Containers: Set memory limits (` + "`" + `--memory` + "`" + ` in Docker, ` + "`" + `resources.limits.memory` + "`" + ` in Kubernetes) to prevent one container from OOM-killing others.`,
					CodeExamples: `# Check swap status
swapon --show
free -h

# Create a swap file (4GB)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent (add to /etc/fstab)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Adjust swappiness
cat /proc/sys/vm/swappiness  # Current value
sudo sysctl vm.swappiness=10  # Temporary (until reboot)
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.d/99-tuning.conf  # Persistent

# OOM killer scores
# Check a process's OOM score
cat /proc/$(pidof postgres)/oom_score
cat /proc/$(pidof postgres)/oom_score_adj

# Protect critical processes from OOM
echo -1000 > /proc/$(pidof postgres)/oom_score_adj

# Check for recent OOM kills
dmesg | grep -i "out of memory"
dmesg | grep -i "killed process"

# Watch memory pressure in real-time
watch -n 1 'free -h && echo "---" && cat /proc/pressure/memory'`,
				},
				{
					Title: "Performance Tuning with sysctl",
					Content: `sysctl is the interface for tuning kernel parameters at runtime. Combined with other tools, it allows fine-grained optimization of system behavior.

**1. sysctl Basics:**
*   ` + "`" + `sysctl -a` + "`" + ` -- List all tunable parameters (1000+).
*   ` + "`" + `sysctl <key>` + "`" + ` -- Read a parameter.
*   ` + "`" + `sysctl -w <key>=<value>` + "`" + ` -- Set temporarily (until reboot).
*   Persistent: Add to ` + "`" + `/etc/sysctl.d/99-custom.conf` + "`" + ` and run ` + "`" + `sysctl -p` + "`" + `.

**2. Network Tuning:**
*   ` + "`" + `net.core.somaxconn` + "`" + ` -- Max connection backlog (default 4096, increase for high-traffic servers).
*   ` + "`" + `net.ipv4.tcp_max_syn_backlog` + "`" + ` -- SYN queue size (default 1024).
*   ` + "`" + `net.ipv4.tcp_fin_timeout` + "`" + ` -- TIME_WAIT timeout (default 60, reduce to 15-30 for busy servers).
*   ` + "`" + `net.ipv4.ip_local_port_range` + "`" + ` -- Ephemeral port range (default 32768-60999, expand for many connections).

**3. File System Tuning:**
*   ` + "`" + `fs.file-max` + "`" + ` -- System-wide max open files (increase for database servers).
*   ` + "`" + `fs.inotify.max_user_watches` + "`" + ` -- Max inotify watches (increase for file watchers, IDEs).
*   ` + "`" + `vm.dirty_ratio` + "`" + ` / ` + "`" + `vm.dirty_background_ratio` + "`" + ` -- Control when dirty pages are flushed to disk.

**4. Memory Tuning:**
*   ` + "`" + `vm.swappiness` + "`" + ` -- Swap tendency (discussed above).
*   ` + "`" + `vm.overcommit_memory` + "`" + ` -- 0 (heuristic, default), 1 (always allow), 2 (strict, no overcommit).
*   ` + "`" + `vm.min_free_kbytes` + "`" + ` -- Minimum free memory reserved for kernel.

**5. ulimits (Per-Process):**
*   ` + "`" + `ulimit -n` + "`" + ` -- Max open file descriptors per process.
*   ` + "`" + `/etc/security/limits.conf` + "`" + ` -- Persistent limits per user/group.
*   ` + "`" + `LimitNOFILE=` + "`" + ` in systemd service files -- Per-service limits.`,
					CodeExamples: `# View current settings
sysctl vm.swappiness
sysctl net.core.somaxconn

# Apply tuning for a web server: /etc/sysctl.d/99-webserver.conf
# Network
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 15
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1

# Memory
vm.swappiness = 10
vm.overcommit_memory = 0
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5

# File system
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# Apply immediately
sudo sysctl -p /etc/sysctl.d/99-webserver.conf

# ulimits for a user: /etc/security/limits.conf
# <user>  <type>  <item>    <value>
# myapp   soft    nofile    65536
# myapp   hard    nofile    65536
# myapp   soft    nproc     32768

# Check current limits
ulimit -a

# systemd service file limits
# [Service]
# LimitNOFILE=65536
# LimitNPROC=32768

# Check effective limits for a running process
cat /proc/$(pidof nginx)/limits`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          143,
			Title:       "Containerization Fundamentals",
			Description: "Understand Linux kernel features that enable containers: namespaces, cgroups, overlay filesystems, and how container runtimes work.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Linux Namespaces",
					Content: `Namespaces are a Linux kernel feature that provides process isolation. They are the foundation of container technology -- each container runs in its own set of namespaces.

**1. Available Namespaces:**
*   **PID:** Process ID isolation. PID 1 inside the container is NOT PID 1 on the host. Container processes only see their own PIDs.
*   **NET:** Network isolation. Each namespace gets its own network interfaces, routing tables, and IP addresses.
*   **MNT:** Mount isolation. Each namespace has its own filesystem mount points. The container sees its own root filesystem.
*   **UTS:** Hostname isolation. Container can have its own hostname.
*   **IPC:** Inter-Process Communication isolation. Separate shared memory, semaphores, and message queues.
*   **USER:** User ID mapping. UID 0 (root) inside the container can map to an unprivileged user on the host (rootless containers).
*   **CGROUP:** Cgroup isolation. Container sees its own cgroup hierarchy.
*   **TIME (v5.6+):** Time namespace. Allows different boot time offsets.

**2. How Containers Use Namespaces:**
*   ` + "`" + `docker run` + "`" + ` creates a new set of namespaces for the container.
*   ` + "`" + `unshare` + "`" + ` system call creates new namespaces.
*   ` + "`" + `nsenter` + "`" + ` allows entering an existing namespace (used by ` + "`" + `docker exec` + "`" + `).

**3. Viewing Namespaces:**
*   ` + "`" + `lsns` + "`" + ` -- List all namespaces on the system.
*   ` + "`" + `/proc/<pid>/ns/` + "`" + ` -- Namespace links for a process.
*   Processes in the same namespace share the same kernel resources.`,
					CodeExamples: `# List all namespaces
lsns
# NS TYPE   NPROCS PID  USER  COMMAND
# ... mnt        1  100  root  /usr/bin/containerd-shim
# ... pid        1  100  root  /usr/bin/containerd-shim

# View namespaces for a process
ls -la /proc/1/ns/
# lrwxrwxrwx 1 root root 0 net -> 'net:[4026531840]'
# lrwxrwxrwx 1 root root 0 pid -> 'pid:[4026531836]'

# Create a new PID namespace (manual container!)
sudo unshare --pid --mount-proc --fork /bin/bash
# Inside: ps aux shows only our bash process (PID 1)

# Create a new network namespace
sudo ip netns add mynet
sudo ip netns exec mynet ip addr
# Only shows loopback interface

# Enter a container's namespace
PID=$(docker inspect --format '{{.State.Pid}}' mycontainer)
sudo nsenter --target $PID --mount --pid --net --ipc /bin/bash
# Now inside the container's namespace from the host

# Create isolated environment with unshare
sudo unshare --pid --net --mount --uts --ipc --fork /bin/bash
hostname container-demo  # Only affects this UTS namespace`,
				},
				{
					Title: "Control Groups (cgroups)",
					Content: `cgroups (control groups) limit, account for, and isolate resource usage (CPU, memory, I/O, network) of process groups. They are the resource control mechanism behind containers.

**1. cgroup v2 (Unified Hierarchy):**
*   Modern Linux uses cgroup v2 (unified hierarchy at ` + "`" + `/sys/fs/cgroup/` + "`" + `).
*   Each cgroup is a directory. Files in the directory configure limits.
*   Key controllers: ` + "`" + `cpu` + "`" + `, ` + "`" + `memory` + "`" + `, ` + "`" + `io` + "`" + `, ` + "`" + `pids` + "`" + `.

**2. CPU Control:**
*   ` + "`" + `cpu.max` + "`" + ` -- CPU bandwidth limit. Format: ` + "`" + `quota period` + "`" + ` (microseconds). ` + "`" + `50000 100000` + "`" + ` = 50% of one CPU.
*   ` + "`" + `cpu.weight` + "`" + ` -- Relative CPU shares (1-10000, default 100). Used when CPUs are contended.

**3. Memory Control:**
*   ` + "`" + `memory.max` + "`" + ` -- Hard memory limit. Processes are OOM-killed if exceeded.
*   ` + "`" + `memory.high` + "`" + ` -- Soft limit. Processes are throttled (slowed) when exceeded.
*   ` + "`" + `memory.current` + "`" + ` -- Current memory usage.
*   ` + "`" + `memory.swap.max` + "`" + ` -- Swap limit (0 to disable swap for this cgroup).

**4. I/O Control:**
*   ` + "`" + `io.max` + "`" + ` -- Per-device I/O limits (IOPS and bandwidth).
*   ` + "`" + `io.weight` + "`" + ` -- Relative I/O priority.

**5. PID Control:**
*   ` + "`" + `pids.max` + "`" + ` -- Maximum number of processes in this cgroup (prevents fork bombs).

**6. Container Mapping:**
*   Docker ` + "`" + `--memory=512m` + "`" + ` → writes ` + "`" + `memory.max = 536870912` + "`" + ` to the cgroup.
*   Docker ` + "`" + `--cpus=2` + "`" + ` → writes ` + "`" + `cpu.max = 200000 100000` + "`" + ` to the cgroup.
*   Kubernetes resource limits map directly to cgroup settings.`,
					CodeExamples: `# Check cgroup version
stat -fc %T /sys/fs/cgroup/
# cgroup2fs = v2, tmpfs = v1

# View cgroup hierarchy
ls /sys/fs/cgroup/
# cgroup.controllers  cpu.stat  memory.current  ...

# View a container's cgroup
docker inspect --format '{{.HostConfig.CgroupParent}}' mycontainer
cat /sys/fs/cgroup/system.slice/docker-<id>.scope/memory.max

# Create a custom cgroup (manual)
sudo mkdir /sys/fs/cgroup/mygroup
echo "256000000" | sudo tee /sys/fs/cgroup/mygroup/memory.max  # 256MB limit
echo "50000 100000" | sudo tee /sys/fs/cgroup/mygroup/cpu.max  # 50% CPU
echo "100" | sudo tee /sys/fs/cgroup/mygroup/pids.max          # Max 100 processes

# Move a process into the cgroup
echo $$ | sudo tee /sys/fs/cgroup/mygroup/cgroup.procs

# Monitor resource usage
cat /sys/fs/cgroup/mygroup/memory.current
cat /sys/fs/cgroup/mygroup/cpu.stat

# Docker resource limits (maps to cgroups)
docker run --memory=512m --cpus=1.5 --pids-limit=100 nginx

# systemd slice (cgroup integration)
systemd-cgtop  # Top-like view of cgroup resource usage

# View cgroup of a process
cat /proc/$(pidof nginx)/cgroup`,
				},
				{
					Title: "Container Runtimes and OCI",
					Content: `Container runtimes are responsible for actually creating and running containers using kernel primitives (namespaces, cgroups). Understanding the runtime stack is essential for debugging container issues.

**1. The Container Runtime Stack:**
*   **High-level runtime (CRI):** Docker, containerd, CRI-O -- manages image pulling, storage, networking, and lifecycle.
*   **Low-level runtime (OCI):** runc, crun, kata-containers -- performs the actual namespace/cgroup/filesystem setup.
*   In Kubernetes: kubelet → CRI → containerd → runc.

**2. OCI (Open Container Initiative):**
*   **OCI Image Spec:** Standard format for container images (layers, manifests, configs).
*   **OCI Runtime Spec:** Standard for how to run a container (config.json with namespaces, mounts, cgroups).
*   **OCI Distribution Spec:** Standard for pushing/pulling images from registries.

**3. containerd:**
*   The industry-standard container runtime (used by Docker and Kubernetes).
*   ` + "`" + `ctr` + "`" + ` -- Low-level containerd CLI.
*   ` + "`" + `crictl` + "`" + ` -- CRI-compatible CLI (used for Kubernetes debugging).
*   ` + "`" + `nerdctl` + "`" + ` -- Docker-compatible CLI for containerd.

**4. Container Filesystems:**
*   **OverlayFS:** Union filesystem that layers multiple directories.
    *   Lower layers: Read-only image layers.
    *   Upper layer: Writable layer for container modifications.
    *   Merged: The union view seen inside the container.
*   ` + "`" + `docker diff <container>` + "`" + ` -- Shows filesystem changes (A=added, C=changed, D=deleted).

**5. Container Networking:**
*   ` + "`" + `bridge` + "`" + ` -- Default. Containers get a veth pair connecting to docker0 bridge.
*   ` + "`" + `host` + "`" + ` -- Container shares host network namespace (no isolation).
*   ` + "`" + `none` + "`" + ` -- No networking.
*   CNI (Container Network Interface) -- Kubernetes uses CNI plugins (Calico, Cilium, Flannel) for pod networking.`,
					CodeExamples: `# Check container runtime
docker info | grep "Server Version"
crictl info | grep runtime

# Inspect container at the runtime level
crictl ps                     # List containers (Kubernetes nodes)
crictl inspect <container-id> # Detailed container info
crictl logs <container-id>    # Container logs

# containerd native CLI
sudo ctr images list          # List images
sudo ctr containers list      # List containers

# nerdctl (Docker-compatible containerd CLI)
nerdctl run -it --rm alpine sh

# Examine OverlayFS layers
docker inspect --format '{{.GraphDriver.Data}}' mycontainer
# map[LowerDir:/var/lib/docker/overlay2/abc/diff
#     MergedDir:/var/lib/docker/overlay2/xyz/merged
#     UpperDir:/var/lib/docker/overlay2/xyz/diff
#     WorkDir:/var/lib/docker/overlay2/xyz/work]

# View filesystem changes in a running container
docker diff mycontainer
# C /var/log
# A /var/log/app.log
# C /tmp

# Manual OverlayFS mount (to understand the concept)
mkdir -p /tmp/overlay/{lower,upper,work,merged}
echo "base file" > /tmp/overlay/lower/file.txt
mount -t overlay overlay \
  -o lowerdir=/tmp/overlay/lower,upperdir=/tmp/overlay/upper,workdir=/tmp/overlay/work \
  /tmp/overlay/merged

# OCI bundle inspection
runc spec  # Generate default OCI config.json
cat config.json | jq '.linux.namespaces'`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
