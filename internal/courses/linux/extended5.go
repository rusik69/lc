package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1827,
			Title:       "Process Management and Systemd",
			Description: "Deep dive into Linux process management, signals, job control, systemd services, timers, targets, and advanced systemd features.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Process Management Deep Dive",
					Content: `Every program running on Linux is a process. Understanding process lifecycle, states, and management is essential for system administration.

**Process Fundamentals:**
` + "```" + `
Process information:
  /proc/<pid>/        Directory for each process
  /proc/<pid>/status  Process status (name, state, memory, threads)
  /proc/<pid>/cmdline Full command line
  /proc/<pid>/environ Environment variables
  /proc/<pid>/fd/     Open file descriptors
  /proc/<pid>/maps    Memory mappings
  /proc/<pid>/limits  Resource limits
  /proc/<pid>/cgroup  Cgroup membership
  /proc/<pid>/net/    Network statistics
  /proc/<pid>/io      I/O statistics

Process states:
  R  Running or runnable (in run queue)
  S  Sleeping (interruptible, waiting for event)
  D  Disk sleep (uninterruptible, waiting for I/O)
  T  Stopped (by signal or debugger)
  Z  Zombie (terminated but not reaped by parent)
  I  Idle kernel thread

Process hierarchy:
  PID 1: init/systemd (parent of all)
  Every process has a parent (PPID)
  fork() creates child (copy of parent)
  exec() replaces process image
  wait() reaps child (collects exit status)
  Orphaned processes are re-parented to PID 1

Viewing processes:
  ps aux                          # All processes (BSD style)
  ps -ef                          # All processes (POSIX style)
  ps -eo pid,ppid,user,%cpu,%mem,stat,cmd --sort=-%mem | head
  ps -eo pid,comm,wchan           # What processes are waiting on
  pstree -p                       # Process tree with PIDs
  pstree -u                       # With user switches
  
  top                             # Interactive process viewer
  htop                            # Better interactive viewer
  atop                            # Advanced top with disk/net
  
  # Real-time per-process I/O
  iotop
  pidstat -d 1                    # I/O per process
  pidstat -r 1                    # Memory per process
  pidstat -u 1                    # CPU per process
` + "```" + `

**Signals:**
` + "```" + `
Common signals:
  Signal    Number  Default   Description
  SIGHUP    1       Terminate Hangup (reload config)
  SIGINT    2       Terminate Interrupt (Ctrl+C)
  SIGQUIT   3       Core dump Quit (Ctrl+\)
  SIGKILL   9       Terminate Kill (cannot be caught!)
  SIGTERM   15      Terminate Graceful termination (default kill)
  SIGSTOP   19      Stop      Pause (cannot be caught!)
  SIGCONT   18      Continue  Resume stopped process
  SIGUSR1   10      Terminate User-defined signal 1
  SIGUSR2   12      Terminate User-defined signal 2
  SIGCHLD   17      Ignore    Child terminated

Sending signals:
  kill <pid>                      # SIGTERM (default)
  kill -9 <pid>                   # SIGKILL (last resort)
  kill -HUP <pid>                 # SIGHUP (reload config)
  kill -STOP <pid>                # Pause process
  kill -CONT <pid>                # Resume process
  killall nginx                   # By name
  pkill -f "python script.py"    # By pattern
  
  # Kill all of user's processes
  pkill -u username
  
  # Kill process group
  kill -TERM -<pgid>

Signal handling in scripts:
  #!/bin/bash
  cleanup() {
    echo "Caught signal, cleaning up..."
    rm -f /tmp/pidfile
    exit 0
  }
  trap cleanup SIGTERM SIGINT SIGHUP
  trap '' SIGPIPE  # Ignore SIGPIPE
  
  # Run logic
  while true; do
    do_work
    sleep 1
  done
` + "```" + `

**Job Control:**
` + "```" + `
Background/foreground:
  command &                       # Run in background
  Ctrl+Z                         # Suspend foreground process
  bg                              # Resume suspended in background
  fg                              # Bring to foreground
  jobs                            # List background jobs
  fg %1                           # Bring job 1 to foreground
  kill %1                         # Kill job 1

nohup and disown:
  nohup long-command &            # Survive terminal close
  disown %1                       # Detach job from shell
  
  # Run command immune to hangup
  nohup ./script.sh > output.log 2>&1 &

Screen and tmux:
  # tmux (preferred)
  tmux new -s session-name        # New session
  tmux attach -t session-name     # Attach to session
  tmux ls                         # List sessions
  # Ctrl+B, D: detach
  # Ctrl+B, C: new window
  # Ctrl+B, N: next window
  
  # screen
  screen -S name                  # New session
  screen -r name                  # Reattach
  # Ctrl+A, D: detach

Process priority:
  nice -n 10 command              # Start with nice value 10
  nice -n -5 command              # Negative (higher priority, needs root)
  renice -n 10 -p <pid>          # Change running process
  
  # Nice values: -20 (highest) to 19 (lowest)
  # Default: 0
  
  # ionice (I/O priority)
  ionice -c 2 -n 0 command       # Best effort, highest
  ionice -c 3 command             # Idle (only when no other I/O)
  ionice -c 1 -n 0 command       # Real-time I/O (root only)
` + "```" + `

**Zombie and Orphan Processes:**
` + "```" + `
Zombie process:
  - Has terminated but parent hasn't called wait()
  - Shows as Z in ps
  - Takes no resources (just an entry in process table)
  - Cannot be killed (already dead!)
  
  Find zombies:
    ps aux | awk '$8 ~ /Z/'
    ps -eo ppid,pid,stat,cmd | grep Z
  
  Fix:
    - Kill the parent process (zombies are then reaped)
    - Or send SIGCHLD to parent: kill -CHLD <ppid>
    - May indicate a buggy parent process

Orphan process:
  - Parent has terminated before child
  - Re-parented to PID 1 (systemd/init)
  - Normal behavior, handled automatically
  
  # Find orphans (PPID=1 but not system processes)
  ps -eo ppid,pid,user,cmd | awk '$1 == 1 && $3 != "root"'
` + "```" + ``,
					CodeExamples: `# Process management scripts

# 1. Process monitoring script
#!/bin/bash
# Monitor a process and restart if it dies
PROCESS="myapp"
COMMAND="/usr/local/bin/myapp --config /etc/myapp.conf"
PIDFILE="/var/run/myapp.pid"
LOGFILE="/var/log/myapp-monitor.log"

check_process() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_process() {
    echo "$(date): Starting $PROCESS" >> "$LOGFILE"
    $COMMAND &
    echo $! > "$PIDFILE"
}

# Trap for cleanup
trap 'echo "$(date): Monitor stopped" >> "$LOGFILE"; exit 0' SIGTERM SIGINT

# Main loop
while true; do
    if ! check_process; then
        echo "$(date): $PROCESS not running, restarting..." >> "$LOGFILE"
        start_process
    fi
    sleep 10
done

# 2. Resource usage report
#!/bin/bash
echo "=== Top 10 CPU Consumers ==="
ps -eo pid,user,%cpu,%mem,comm --sort=-%cpu | head -11

echo ""
echo "=== Top 10 Memory Consumers ==="
ps -eo pid,user,%cpu,%mem,rss,comm --sort=-rss | head -11

echo ""
echo "=== Process Count by User ==="
ps -eo user= | sort | uniq -c | sort -rn | head -10

echo ""
echo "=== Process States ==="
ps -eo stat= | cut -c1 | sort | uniq -c | sort -rn

echo ""
echo "=== Zombie Processes ==="
ZOMBIES=$(ps aux | awk '$8 ~ /Z/ {print $2, $11}')
if [ -n "$ZOMBIES" ]; then
    echo "$ZOMBIES"
else
    echo "None"
fi

echo ""
echo "=== Open File Descriptors (top 10) ==="
for pid in $(ls /proc/ | grep -E '^[0-9]+$' | head -100); do
    if [ -d "/proc/$pid/fd" ]; then
        count=$(ls -1 "/proc/$pid/fd" 2>/dev/null | wc -l)
        name=$(cat "/proc/$pid/comm" 2>/dev/null)
        echo "$count $pid $name"
    fi
done | sort -rn | head -10

# 3. Graceful shutdown pattern
#!/bin/bash
SHUTDOWN=false

shutdown_handler() {
    echo "Received shutdown signal, finishing current work..."
    SHUTDOWN=true
}

trap shutdown_handler SIGTERM SIGINT

while [ "$SHUTDOWN" = false ]; do
    # Do work
    echo "Processing batch at $(date)"
    # Simulate work
    for i in $(seq 1 5); do
        if [ "$SHUTDOWN" = true ]; then
            echo "Interrupted, saving state..."
            break
        fi
        sleep 1
    done
done

echo "Graceful shutdown complete"
exit 0`,
				},
				{
					Title: "Systemd In Depth",
					Content: `systemd is the init system and service manager for modern Linux. It manages services, targets, timers, mounts, and more.

**Unit Types:**
` + "```" + `
Unit types:
  .service   System services (daemons)
  .socket    Socket-based activation
  .timer     Timer-based activation (cron replacement)
  .mount     Mount points
  .automount Auto-mount on access
  .target    Groups of units (like runlevels)
  .path      Path-based activation
  .slice     Resource management (cgroups)
  .scope     Externally created processes
  .device    Device units

Unit file locations:
  /usr/lib/systemd/system/   Package-provided (don't edit)
  /etc/systemd/system/       Admin customizations (highest priority)
  /run/systemd/system/       Runtime units

Common commands:
  systemctl start nginx            # Start service
  systemctl stop nginx             # Stop service
  systemctl restart nginx          # Restart
  systemctl reload nginx           # Reload config (no restart)
  systemctl enable nginx           # Start on boot
  systemctl disable nginx          # Don't start on boot
  systemctl enable --now nginx     # Enable and start
  systemctl status nginx           # Show status
  systemctl is-active nginx        # Check if running
  systemctl is-enabled nginx       # Check if enabled
  
  systemctl list-units             # All loaded units
  systemctl list-units --failed    # Failed units
  systemctl list-unit-files        # All unit files
  systemctl list-dependencies nginx  # Dependency tree
  
  systemctl daemon-reload          # Reload unit files (after editing)
  systemctl cat nginx              # View unit file
  systemctl edit nginx             # Create override (drop-in)
  systemctl edit --full nginx      # Edit full unit file
  systemctl show nginx             # All properties
  systemctl mask nginx             # Prevent starting entirely
  systemctl unmask nginx           # Undo mask
` + "```" + `

**Service Unit Configuration:**
` + "```" + `
[Unit]
  Description=       Human-readable description
  Documentation=     man pages or URLs
  After=             Start after these units
  Before=            Start before these units
  Requires=          Hard dependency (fails if dep fails)
  Wants=             Soft dependency (doesn't fail if dep fails)
  BindsTo=           Stronger Requires (stops if dep stops)
  Conflicts=         Cannot run with these units
  ConditionPathExists=  Only start if path exists

[Service]
  Type=              Service type
    simple           Default. Process started is the main process
    forking          Process forks (traditional daemon)
    oneshot          Short-lived (run once and exit)
    notify           Like simple, but notifies systemd when ready
    exec             Like simple, but waits for exec() to complete
    dbus             Like simple, activated via D-Bus
    idle             Like simple, delayed until other jobs finish
    
  ExecStartPre=      Commands before main process
  ExecStart=         Main process command
  ExecStartPost=     Commands after main process starts
  ExecReload=        Reload command
  ExecStop=          Stop command
  ExecStopPost=      Commands after stop
  
  Restart=           When to restart
    no               Don't restart (default)
    always           Always restart
    on-failure       Restart on non-zero exit
    on-abnormal      Restart on signal, timeout, watchdog
    on-abort         Restart on signal
    on-success       Restart on clean exit
    
  RestartSec=        Delay before restart
  TimeoutStartSec=   Startup timeout
  TimeoutStopSec=    Shutdown timeout
  WatchdogSec=       Watchdog timeout
  
  User=              Run as user
  Group=             Run as group
  WorkingDirectory=  Working directory
  Environment=       Environment variables
  EnvironmentFile=   File with environment variables
  
  # Resource limits
  LimitNOFILE=65536
  LimitNPROC=4096
  LimitMEMLOCK=infinity
  
  # Process management
  KillMode=          How to stop (control-group/process/mixed/none)
  KillSignal=        Signal to send (SIGTERM default)
  SendSIGKILL=       Send SIGKILL after timeout (yes/no)
  
[Install]
  WantedBy=          Target to install into
  RequiredBy=        Harder dependency target
  Alias=             Alternative names
` + "```" + `

**Systemd Timers:**
` + "```" + `
Timer units replace cron with:
  - Dependency on other services
  - Logging integration (journald)
  - Resource control (cgroups)
  - Randomized delay (avoid thundering herd)

Timer file (/etc/systemd/system/backup.timer):
  [Unit]
  Description=Daily backup timer
  
  [Timer]
  OnCalendar=*-*-* 02:00:00     # Daily at 2 AM
  RandomizedDelaySec=1800        # Random 0-30 min delay
  Persistent=true                # Run missed if system was off
  
  [Install]
  WantedBy=timers.target

Service file (/etc/systemd/system/backup.service):
  [Unit]
  Description=Daily backup
  
  [Service]
  Type=oneshot
  ExecStart=/usr/local/bin/backup.sh
  User=backup

OnCalendar examples:
  *-*-* 00:00:00         Daily at midnight
  Mon *-*-* 00:00:00     Every Monday at midnight
  *-*-01 00:00:00        First of every month
  *-*-* *:00:00          Every hour
  *-*-* *:*:00           Every minute
  *-*-* *:00/15:00       Every 15 minutes
  Mon..Fri *-*-* 09:00   Weekdays at 9 AM

Monotonic timers:
  OnBootSec=15min        After boot
  OnUnitActiveSec=1h     After last activation
  OnStartupSec=5min      After systemd start

Timer commands:
  systemctl start backup.timer
  systemctl enable backup.timer
  systemctl list-timers             # List all timers
  systemctl list-timers --all       # Include inactive
  
  # Test calendar expressions:
  systemd-analyze calendar "Mon *-*-* 09:00"
  systemd-analyze calendar --iterations=5 "*-*-* *:00/15:00"
` + "```" + `

**Targets and Boot:**
` + "```" + `
Targets (like runlevels):
  poweroff.target     Halt (runlevel 0)
  rescue.target       Single user (runlevel 1)
  multi-user.target   Multi-user, no GUI (runlevel 3)
  graphical.target    GUI (runlevel 5)
  reboot.target       Reboot (runlevel 6)
  emergency.target    Emergency shell (minimal)

Commands:
  systemctl get-default                    # Current default target
  systemctl set-default multi-user.target  # Set default
  systemctl isolate rescue.target          # Switch to rescue mode
  systemctl rescue                         # Shortcut for rescue
  systemctl emergency                      # Emergency mode

Boot analysis:
  systemd-analyze                     # Boot time
  systemd-analyze blame               # Time per service
  systemd-analyze critical-chain      # Critical path
  systemd-analyze plot > boot.svg     # Visual timeline
  systemd-analyze verify foo.service  # Validate unit file
` + "```" + `

**journald (Logging):**
` + "```" + `
journalctl:
  journalctl                     # All logs
  journalctl -u nginx            # Specific unit
  journalctl -f                  # Follow (like tail -f)
  journalctl -f -u nginx        # Follow specific unit
  journalctl --since today       # Today's logs
  journalctl --since "2024-01-01 00:00:00"
  journalctl --since "1 hour ago"
  journalctl -p err              # Priority: emerg, alert, crit, err, warning, notice, info, debug
  journalctl -k                  # Kernel messages (dmesg)
  journalctl -b                  # Current boot
  journalctl -b -1               # Previous boot
  journalctl --list-boots        # List boot IDs
  journalctl --disk-usage        # Storage used
  journalctl --vacuum-size=500M  # Trim to 500M
  journalctl --vacuum-time=7d    # Keep only 7 days
  journalctl -o json-pretty      # JSON output
  journalctl _UID=1000           # By user ID
  journalctl _PID=1234           # By process ID

Configuration (/etc/systemd/journald.conf):
  [Journal]
  Storage=persistent              # persistent, volatile, auto, none
  SystemMaxUse=500M               # Max disk usage
  SystemKeepFree=1G               # Min free space
  MaxRetentionSec=1month          # Max retention
  MaxFileSec=1week                # Max per-file retention
  Compress=yes                    # Compress stored journals
  ForwardToSyslog=no              # Don't duplicate to syslog
  RateLimitIntervalSec=30s
  RateLimitBurst=10000
` + "```" + ``,
					CodeExamples: `# Systemd configuration examples

# 1. Production web application service
# /etc/systemd/system/webapp.service
[Unit]
Description=Production Web Application
Documentation=https://docs.example.com/webapp
After=network-online.target postgresql.service redis.service
Wants=network-online.target
Requires=postgresql.service

[Service]
Type=notify
User=webapp
Group=webapp
WorkingDirectory=/opt/webapp
EnvironmentFile=/etc/webapp/environment
ExecStartPre=/opt/webapp/bin/migrate
ExecStart=/opt/webapp/bin/server --config /etc/webapp/config.yaml
ExecReload=/bin/kill -HUP $MAINPID
ExecStop=/bin/kill -TERM $MAINPID
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=on-failure
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=3

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/webapp /var/log/webapp
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=webapp

[Install]
WantedBy=multi-user.target

---
# 2. Cleanup timer
# /etc/systemd/system/cleanup.timer
[Unit]
Description=Weekly cleanup timer

[Timer]
OnCalendar=Sun *-*-* 03:00:00
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target

---
# /etc/systemd/system/cleanup.service
[Unit]
Description=Weekly system cleanup

[Service]
Type=oneshot
ExecStart=/usr/local/bin/system-cleanup.sh
User=root

# Security
ProtectHome=yes
NoNewPrivileges=yes

---
# 3. Socket-activated service
# /etc/systemd/system/myapi.socket
[Unit]
Description=My API Socket

[Socket]
ListenStream=8080
Accept=no
# Or for Unix socket:
# ListenStream=/run/myapi.sock
# SocketMode=0660
# SocketUser=www-data

[Install]
WantedBy=sockets.target

---
# /etc/systemd/system/myapi.service
[Unit]
Description=My API Service
Requires=myapi.socket
After=myapi.socket

[Service]
Type=simple
ExecStart=/opt/myapi/bin/server
User=myapi
Group=myapi
NonBlocking=true

# Activated by socket, no Install needed

---
# 4. Systemd override (drop-in)
# systemctl edit nginx
# Creates /etc/systemd/system/nginx.service.d/override.conf
[Service]
LimitNOFILE=131072
RestartSec=10

---
# 5. Resource control with slices
# /etc/systemd/system/apps.slice
[Unit]
Description=Application Services Slice

[Slice]
CPUQuota=70%
MemoryMax=4G
MemoryHigh=3G
IOWeight=50
TasksMax=1024

# Then in service:
# [Service]
# Slice=apps.slice`,
				},
			},
		},
	})
}
