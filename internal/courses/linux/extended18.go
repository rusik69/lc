package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1840,
			Title:       "Linux Time Synchronization and Scheduling",
			Description: "Configure NTP time synchronization, manage scheduled tasks with cron and systemd timers, and understand Linux timekeeping internals.",
			Order:       40,
			Lessons: []problems.Lesson{
				{
					Title: "Time Synchronization and Scheduling",
					Content: `Accurate time is critical for logging, authentication (Kerberos), distributed systems, and certificates. Cron and systemd timers automate recurring tasks.

**NTP and Chrony:**
` + "```" + `
Chrony is the modern NTP implementation (replaced ntpd).

Installation:
  apt install chrony              # Debian/Ubuntu
  dnf install chrony              # RHEL/Fedora

Configuration (/etc/chrony/chrony.conf):
  # NTP servers
  server 0.pool.ntp.org iburst
  server 1.pool.ntp.org iburst
  server 2.pool.ntp.org iburst
  server 3.pool.ntp.org iburst
  
  # Or use pool directive
  pool pool.ntp.org iburst maxsources 4
  
  # Allow clients from local network
  allow 10.0.0.0/24
  
  # Serve time even if not synced
  local stratum 10
  
  # Record rate of drift
  driftfile /var/lib/chrony/drift
  
  # RTC (hardware clock)
  rtcsync
  
  # Step clock if offset > 1 second (first 3 updates)
  makestep 1.0 3
  
  # Log
  logdir /var/log/chrony

Commands:
  chronyc tracking           # Current synchronization status
  chronyc sources -v         # NTP sources with details
  chronyc sourcestats        # Source statistics
  chronyc activity           # Number of sources online/offline
  chronyc makestep           # Force immediate correction
  
  # Check if synced
  chronyc tracking | grep "Leap status"
  # Normal = synced

  timedatectl                # System time/date/timezone
  timedatectl set-timezone America/New_York
  timedatectl set-ntp true   # Enable NTP synchronization
  timedatectl timesync-status  # systemd-timesyncd status
` + "```" + `

**Cron:**
` + "```" + `
Cron is the traditional Unix job scheduler.

Crontab format:
  # m  h  dom  mon  dow  command
  # *  *   *    *    *   command
  # |  |   |    |    |
  # |  |   |    |    +-- Day of week (0-7, Sun=0 or 7)
  # |  |   |    +------- Month (1-12)
  # |  |   +------------ Day of month (1-31)
  # |  +---------------- Hour (0-23)
  # +------------------- Minute (0-59)

Examples:
  0 * * * *         Every hour
  */15 * * * *      Every 15 minutes
  0 2 * * *         Daily at 2:00 AM
  0 2 * * 1         Mondays at 2:00 AM
  0 0 1 * *         First of each month at midnight
  0 2 * * 1-5       Weekdays at 2:00 AM
  0 9,18 * * *      At 9:00 AM and 6:00 PM

Special strings:
  @reboot            Run once at startup
  @yearly            0 0 1 1 *
  @monthly           0 0 1 * *
  @weekly            0 0 * * 0
  @daily             0 0 * * *
  @hourly            0 * * * *

Management:
  crontab -e                 # Edit user crontab
  crontab -l                 # List user crontab
  crontab -r                 # Remove user crontab
  crontab -u user -l         # List other user's crontab (root)

System cron:
  /etc/crontab               # System crontab
  /etc/cron.d/               # System cron fragments
  /etc/cron.daily/           # Daily scripts
  /etc/cron.hourly/          # Hourly scripts
  /etc/cron.weekly/          # Weekly scripts
  /etc/cron.monthly/         # Monthly scripts

Best practices:
  # Always set PATH
  PATH=/usr/local/bin:/usr/bin:/bin
  
  # Use lock to prevent overlap
  * * * * * flock -n /tmp/myjob.lock /path/to/script.sh
  
  # Redirect output
  0 2 * * * /path/to/backup.sh >> /var/log/backup.log 2>&1
  
  # Send output to /dev/null (suppress email)
  0 * * * * /path/to/script.sh > /dev/null 2>&1
  
  # Random delay (avoid thundering herd)
  0 2 * * * sleep $((RANDOM \% 900)) && /path/to/script.sh
` + "```" + `

**Systemd Timers:**
` + "```" + `
Systemd timers are the modern alternative to cron.

Advantages over cron:
  - Dependency on other services
  - Logging via journal
  - Resource control (cgroups)
  - Calendar expressions
  - Randomized delay
  - Persistent (run missed jobs)

Timer unit (/etc/systemd/system/backup.timer):
  [Unit]
  Description=Daily Backup Timer
  
  [Timer]
  OnCalendar=*-*-* 02:00:00     # Daily at 2 AM
  RandomizedDelaySec=900         # Random delay up to 15 min
  Persistent=true                # Run if missed while off
  
  [Install]
  WantedBy=timers.target

Service unit (/etc/systemd/system/backup.service):
  [Unit]
  Description=Daily Backup
  After=network.target
  
  [Service]
  Type=oneshot
  ExecStart=/usr/local/bin/backup.sh
  User=backup
  
  # Resource limits
  MemoryMax=256M
  CPUQuota=50%

Calendar expressions:
  OnCalendar=hourly             # Every hour
  OnCalendar=daily              # Every day at midnight
  OnCalendar=weekly             # Every Monday at midnight
  OnCalendar=monthly            # First of month
  OnCalendar=*-*-* 02:00:00    # Daily at 2 AM
  OnCalendar=Mon *-*-* 09:00   # Mondays at 9 AM
  OnCalendar=*-*-1,15 00:00:00 # 1st and 15th of month
  OnCalendar=*:0/15             # Every 15 minutes

Monotonic timers (relative):
  OnBootSec=15min               # 15 min after boot
  OnUnitActiveSec=1h            # 1 hour after last activation
  OnStartupSec=30s              # 30 sec after systemd start

Commands:
  systemctl enable --now backup.timer
  systemctl list-timers --all
  systemctl status backup.timer
  systemctl status backup.service    # Last run status
  journalctl -u backup.service       # Logs
  
  # Test timer expression
  systemd-analyze calendar "*-*-* 02:00:00"
  systemd-analyze calendar "Mon *-*-* 09:00"
  
  # Run immediately (test)
  systemctl start backup.service
` + "```" + ``,
					CodeExamples: `# Time and scheduling management

# 1. Cron job audit script
#!/bin/bash
echo "=== Cron Job Audit ==="

# System crontab
echo "--- /etc/crontab ---"
grep -v '^#\|^$' /etc/crontab 2>/dev/null | while read -r line; do
    echo "  $line"
done

# User crontabs
echo ""
echo "--- User Crontabs ---"
for user in $(cut -d: -f1 /etc/passwd); do
    crontab_content=$(crontab -u "$user" -l 2>/dev/null | grep -v '^#\|^$')
    if [ -n "$crontab_content" ]; then
        echo "  User: $user"
        echo "$crontab_content" | while read -r line; do
            echo "    $line"
        done
    fi
done

# /etc/cron.d/
echo ""
echo "--- /etc/cron.d/ ---"
for f in /etc/cron.d/*; do
    [ -f "$f" ] || continue
    echo "  $(basename "$f"):"
    grep -v '^#\|^$' "$f" | while read -r line; do
        echo "    $line"
    done
done

# Systemd timers
echo ""
echo "--- Systemd Timers ---"
systemctl list-timers --no-pager 2>/dev/null | head -20

# 2. NTP monitoring script
#!/bin/bash
echo "=== NTP Status ==="

if command -v chronyc > /dev/null 2>&1; then
    echo "--- Chrony ---"
    
    # Synchronization status
    SYNC=$(chronyc tracking 2>/dev/null)
    SOURCE=$(echo "$SYNC" | grep "Reference ID" | awk -F: '{print $2}')
    STRATUM=$(echo "$SYNC" | grep "Stratum" | awk -F: '{print $2}')
    OFFSET=$(echo "$SYNC" | grep "System time" | awk -F: '{print $2}')
    LEAP=$(echo "$SYNC" | grep "Leap status" | awk -F: '{print $2}')
    
    echo "  Source:  $SOURCE"
    echo "  Stratum:$STRATUM"
    echo "  Offset: $OFFSET"
    echo "  Leap:   $LEAP"
    
    echo ""
    echo "  Sources:"
    chronyc sources 2>/dev/null | tail -n+3 | while read -r line; do
        echo "    $line"
    done
    
elif command -v ntpq > /dev/null 2>&1; then
    echo "--- NTPd ---"
    ntpq -p 2>/dev/null
fi

# Check if time is reasonable
echo ""
echo "--- Time Verification ---"
LOCAL=$(date +%s)
# Compare with HTTP date header
REMOTE=$(curl -sI http://google.com 2>/dev/null | \
    grep -i "^date:" | sed 's/date: //i')

if [ -n "$REMOTE" ]; then
    REMOTE_EPOCH=$(date -d "$REMOTE" +%s 2>/dev/null || echo 0)
    if [ "$REMOTE_EPOCH" -gt 0 ]; then
        DIFF=$((LOCAL - REMOTE_EPOCH))
        ABS_DIFF=${DIFF#-}
        if [ "$ABS_DIFF" -gt 5 ]; then
            echo "  WARNING: Clock offset: ${DIFF}s"
        else
            echo "  OK: Clock is accurate (offset: ${DIFF}s)"
        fi
    fi
fi`,
				},
			},
		},
	})
}
