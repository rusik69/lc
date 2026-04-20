package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1837,
			Title:       "Linux Backup, Recovery, and Disaster Planning",
			Description: "Comprehensive strategies for data protection, backup automation, disaster recovery planning, and system restoration on Linux systems.",
			Order:       37,
			Lessons: []problems.Lesson{
				{
					Title: "Backup Strategies and Tools",
					Content: `A solid backup strategy is the last line of defense against data loss. The 3-2-1 rule is the foundation: 3 copies, 2 different media types, 1 offsite.

**Backup Types:**
` + "```" + `
Full backup:
  - Complete copy of all data
  - Longest time, most storage
  - Simplest restore
  
Incremental backup:
  - Only changes since last backup (any type)
  - Fastest backup, least storage
  - Restore requires full + ALL incrementals in order
  
Differential backup:
  - Changes since last FULL backup
  - Medium time and storage
  - Restore requires full + latest differential
  
  Timeline example:
  Sun: Full (10GB base)
  Mon: Incr: +500MB | Diff: +500MB
  Tue: Incr: +300MB | Diff: +800MB
  Wed: Incr: +400MB | Diff: +1.2GB
  Thu: Incr: +200MB | Diff: +1.4GB
  
  To restore Wednesday's state:
  Incremental: Full + Mon-inc + Tue-inc + Wed-inc
  Differential: Full + Wed-diff
` + "```" + `

**rsync:**
` + "```" + `
rsync is the Swiss army knife of file synchronization.

Basic usage:
  rsync -av /source/ /destination/      # Local sync
  rsync -av /source/ user@host:/dest/   # Remote sync
  rsync -av user@host:/source/ /dest/   # Pull from remote

Key options:
  -a   Archive mode (preserves permissions, times, links, etc.)
  -v   Verbose
  -z   Compress during transfer
  -P   Show progress + resume partial
  --delete            Delete extra files in destination
  --exclude='*.log'   Exclude patterns
  --include='*.conf'  Include patterns
  --dry-run           Show what would happen
  --bwlimit=5000      Limit bandwidth (KB/s)
  --backup            Keep old versions
  --backup-dir=/path  Where to put old versions
  --link-dest=/prev   Hard link unchanged files (space efficient)

Incremental backup with hard links:
  # Creates space-efficient snapshots
  BACKUP_DIR="/backup"
  DATE=$(date +%Y-%m-%d_%H%M)
  LATEST="$BACKUP_DIR/latest"
  
  rsync -av --delete \
    --link-dest="$LATEST" \
    /data/ \
    "$BACKUP_DIR/$DATE/"
  
  # Update latest symlink
  ln -snf "$BACKUP_DIR/$DATE" "$LATEST"
  
  # Result: each backup only stores changed files
  # Unchanged files are hard-linked to previous backup
  # Each directory looks like a full backup
  # But uses disk space of incremental

Over SSH with options:
  rsync -avzP \
    -e 'ssh -p 2222 -i /path/to/key' \
    --exclude='.git' \
    --exclude='node_modules' \
    /project/ \
    user@backup-server:/backups/project/
` + "```" + `

**Borg Backup:**
` + "```" + `
Borg is a deduplicating backup program with encryption.

Installation:
  apt install borgbackup        # Debian/Ubuntu
  dnf install borgbackup        # RHEL/Fedora

Initialize repository:
  # Local repository
  borg init --encryption=repokey /backup/borg-repo
  
  # Remote repository
  borg init --encryption=repokey ssh://user@backup-server/path/to/repo
  
  # Encryption modes:
  #   none       No encryption
  #   repokey    Key in repo (password protects it)
  #   keyfile    Key in ~/.config/borg/keys/
  #   repokey-blake2   Faster hash

Create backup:
  borg create \
    --stats --progress \
    --compression zstd,3 \
    --exclude '/home/*/.cache' \
    --exclude '/home/*/Downloads' \
    --exclude '*.tmp' \
    /backup/borg-repo::'{hostname}-{now:%Y-%m-%d_%H%M}' \
    /home /etc /var/log

List archives:
  borg list /backup/borg-repo
  borg list /backup/borg-repo::archive-name   # Files in archive

Info:
  borg info /backup/borg-repo
  borg info /backup/borg-repo::archive-name

Restore:
  # Extract full archive
  cd /tmp/restore
  borg extract /backup/borg-repo::archive-name
  
  # Extract specific files
  borg extract /backup/borg-repo::archive-name home/user/documents
  
  # Mount as FUSE filesystem
  mkdir /tmp/borg-mount
  borg mount /backup/borg-repo::archive-name /tmp/borg-mount
  # Browse files normally
  ls /tmp/borg-mount
  # Unmount when done
  borg umount /tmp/borg-mount

Prune old archives:
  borg prune \
    --keep-daily 7 \
    --keep-weekly 4 \
    --keep-monthly 6 \
    --keep-yearly 2 \
    /backup/borg-repo
  
  # Compact freed space
  borg compact /backup/borg-repo
` + "```" + `

**restic:**
` + "```" + `
restic is another modern backup tool with deduplication.

Supports multiple backends:
  - Local filesystem
  - SFTP
  - AWS S3 / S3-compatible
  - Azure Blob Storage
  - Google Cloud Storage
  - Backblaze B2

Installation:
  apt install restic        # Debian/Ubuntu
  dnf install restic        # RHEL/Fedora

Initialize:
  # Local
  restic init --repo /backup/restic-repo
  
  # S3
  export AWS_ACCESS_KEY_ID=key
  export AWS_SECRET_ACCESS_KEY=secret
  restic init --repo s3:s3.amazonaws.com/bucket-name
  
  # SFTP
  restic init --repo sftp:user@host:/path/to/repo

Backup:
  restic backup \
    --repo /backup/restic-repo \
    --exclude='*.cache' \
    --exclude-file=.backupignore \
    --tag production \
    /data /etc

Snapshots:
  restic snapshots --repo /backup/restic-repo
  restic snapshots --repo /backup/restic-repo --tag production

Restore:
  restic restore latest --repo /backup/restic-repo --target /tmp/restore
  restic restore latest --repo /backup/restic-repo --target / --include /etc

Prune:
  restic forget \
    --keep-last 10 \
    --keep-daily 7 \
    --keep-weekly 4 \
    --keep-monthly 12 \
    --prune \
    --repo /backup/restic-repo
` + "```" + ``,
					CodeExamples: `# Backup automation

# 1. Comprehensive backup script with Borg
#!/bin/bash
set -euo pipefail

# Configuration
BORG_REPO="/backup/borg-repo"
BACKUP_PATHS="/home /etc /var/www /var/lib/postgresql"
EXCLUDE_PATTERNS=(
    '*.pyc'
    '__pycache__'
    '.cache'
    'node_modules'
    '.npm'
    '*.tmp'
    '*.swp'
    '/home/*/.local/share/Trash'
)

LOG_FILE="/var/log/backup.log"
LOCK_FILE="/tmp/backup.lock"

# Logging
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"; }

# Lock to prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    log "ERROR: Backup already running (lock file exists)"
    exit 1
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

log "=== Backup started ==="

# Build exclude arguments
EXCLUDE_ARGS=""
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude '$pattern'"
done

# Pre-backup: dump databases
log "Dumping databases..."
DUMP_DIR="/tmp/db-dumps"
mkdir -p "$DUMP_DIR"
pg_dumpall > "$DUMP_DIR/postgresql-all.sql" 2>/dev/null || true
mysqldump --all-databases > "$DUMP_DIR/mysql-all.sql" 2>/dev/null || true

# Create archive
ARCHIVE_NAME="{hostname}-{now:%Y-%m-%d_%H%M%S}"
log "Creating archive: $ARCHIVE_NAME"

eval borg create \
    --verbose --stats \
    --compression zstd,3 \
    --checkpoint-interval 600 \
    $EXCLUDE_ARGS \
    "$BORG_REPO::$ARCHIVE_NAME" \
    $BACKUP_PATHS "$DUMP_DIR" 2>&1 | tee -a "$LOG_FILE"

# Cleanup database dumps
rm -rf "$DUMP_DIR"

# Prune old archives
log "Pruning old archives..."
borg prune \
    --keep-daily 7 \
    --keep-weekly 4 \
    --keep-monthly 6 \
    --keep-yearly 2 \
    "$BORG_REPO" 2>&1 | tee -a "$LOG_FILE"

# Compact
borg compact "$BORG_REPO" 2>&1 | tee -a "$LOG_FILE"

# Verify
log "Verifying latest archive..."
borg check --last 1 "$BORG_REPO" 2>&1 | tee -a "$LOG_FILE"

# Report
REPO_SIZE=$(borg info "$BORG_REPO" 2>/dev/null | grep "All archives" | head -1)
log "Repository: $REPO_SIZE"
log "=== Backup completed ==="

# 2. Rsync incremental backup with rotation
#!/bin/bash
BACKUP_BASE="/backup/rsync"
SOURCE="/data"
MAX_BACKUPS=30

DATE=$(date +%Y-%m-%d_%H%M)
CURRENT="$BACKUP_BASE/$DATE"
LATEST="$BACKUP_BASE/latest"

echo "Starting incremental backup: $DATE"

# Create backup with hard links to previous
if [ -L "$LATEST" ]; then
    rsync -av --delete \
        --link-dest="$LATEST" \
        "$SOURCE/" \
        "$CURRENT/"
else
    rsync -av --delete \
        "$SOURCE/" \
        "$CURRENT/"
fi

# Update latest pointer
ln -snf "$CURRENT" "$LATEST"

# Rotate old backups
BACKUP_COUNT=$(find "$BACKUP_BASE" -maxdepth 1 -type d -name '20*' | wc -l)
if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
    REMOVE_COUNT=$((BACKUP_COUNT - MAX_BACKUPS))
    find "$BACKUP_BASE" -maxdepth 1 -type d -name '20*' | \
        sort | head -n "$REMOVE_COUNT" | while read -r dir; do
        echo "Removing old backup: $dir"
        rm -rf "$dir"
    done
fi

echo "Backup complete. Total backups: $(find "$BACKUP_BASE" -maxdepth 1 -type d -name '20*' | wc -l)"

# 3. Backup verification script
#!/bin/bash
echo "=== Backup Verification ==="

check_borg() {
    local repo="$1"
    echo "--- Borg: $repo ---"
    
    if ! borg info "$repo" > /dev/null 2>&1; then
        echo "  FAIL: Cannot access repository"
        return 1
    fi
    
    LATEST=$(borg list "$repo" --last 1 --format '{name}' 2>/dev/null)
    AGE=$(borg info "$repo::$LATEST" --json 2>/dev/null | \
        python3 -c "import sys,json,datetime; d=json.load(sys.stdin); \
        t=datetime.datetime.fromisoformat(d['archives'][0]['start'].replace('Z','+00:00')); \
        print((datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()/3600)")
    
    echo "  Latest: $LATEST"
    printf "  Age: %.1f hours\n" "$AGE"
    
    if (( $(echo "$AGE > 25" | bc -l) )); then
        echo "  WARNING: Backup is older than 25 hours!"
    else
        echo "  OK: Backup is recent"
    fi
    
    # Quick integrity check
    if borg check --last 1 "$repo" 2>/dev/null; then
        echo "  Integrity: PASS"
    else
        echo "  Integrity: FAIL"
    fi
}

check_borg "/backup/borg-repo"`,
				},
				{
					Title: "Disaster Recovery and System Restoration",
					Content: `Disaster recovery (DR) planning ensures you can restore operations after catastrophic failures. The key metrics are RTO (Recovery Time Objective) and RPO (Recovery Point Objective).

**System Image Backup:**
` + "```" + `
Full system image capture for bare-metal recovery.

dd (raw disk image):
  # Full disk backup
  dd if=/dev/sda of=/backup/disk.img bs=4M status=progress
  
  # Compressed
  dd if=/dev/sda bs=4M status=progress | gzip > /backup/disk.img.gz
  
  # Restore
  gunzip -c /backup/disk.img.gz | dd of=/dev/sda bs=4M status=progress
  
  # Partition only
  dd if=/dev/sda1 of=/backup/root.img bs=4M status=progress
  
  WARNING: dd copies everything including free space
  Use for small disks or when exact sector copy needed

Clonezilla (intelligent imaging):
  # Disk to image
  clonezilla -d local_dev -z1p \
    savedisk myimage sda
  
  # Image to disk
  clonezilla -d local_dev \
    restoredisk myimage sda
  
  # Only copies used blocks (much faster than dd)
  # Supports compression, encryption
  # Can clone to remote server

Partclone (used by Clonezilla internally):
  # Backup ext4 partition
  partclone.ext4 -c -d -s /dev/sda1 -o /backup/sda1.img
  
  # Restore
  partclone.ext4 -r -d -s /backup/sda1.img -o /dev/sda1

Rear (Relax-and-Recover):
  # Enterprise-grade disaster recovery
  apt install rear      # or dnf install rear
  
  # Configuration /etc/rear/local.conf:
  OUTPUT=ISO                          # Boot media type
  OUTPUT_URL=nfs://server/backup      # Where to store boot media
  BACKUP=NETFS                        # Backup method
  BACKUP_URL=nfs://server/backup      # Where to store backup
  BACKUP_PROG_EXCLUDE=('/tmp/*' '/dev/shm/*')
  
  # Create rescue media + backup
  rear -v mkbackup
  
  # To recover: boot from rescue ISO
  # At prompt: rear recover
  # Recreates partitions, restores data, fixes bootloader
` + "```" + `

**GRUB Recovery:**
` + "```" + `
GRUB bootloader recovery procedures.

GRUB rescue mode:
  # If you see grub rescue> prompt
  
  # Find bootable partition
  ls                      # List disks/partitions
  ls (hd0,1)/             # Check contents
  ls (hd0,1)/boot/grub    # Look for grub files
  
  # Set boot partition
  set root=(hd0,1)
  set prefix=(hd0,1)/boot/grub
  insmod normal
  normal
  
  # This should boot to normal GRUB menu

From live USB/CD:
  # Mount system partitions
  mount /dev/sda2 /mnt           # Root
  mount /dev/sda1 /mnt/boot/efi  # EFI (if UEFI)
  
  # Chroot
  mount --bind /dev /mnt/dev
  mount --bind /dev/pts /mnt/dev/pts
  mount --bind /proc /mnt/proc
  mount --bind /sys /mnt/sys
  chroot /mnt
  
  # Reinstall GRUB
  # BIOS:
  grub-install /dev/sda
  update-grub
  
  # UEFI:
  grub-install --target=x86_64-efi --efi-directory=/boot/efi
  update-grub
  
  # Exit and reboot
  exit
  umount -R /mnt
  reboot

Filesystem recovery:
  # Check and repair ext4
  e2fsck -f /dev/sda1              # Force check
  e2fsck -y /dev/sda1              # Auto-fix
  
  # XFS repair
  xfs_repair /dev/sda1
  xfs_repair -L /dev/sda1          # Zero corrupt log

  # Btrfs
  btrfs check /dev/sda1
  btrfs check --repair /dev/sda1   # Attempt repair

  # Recover deleted files
  extundelete /dev/sda1 --restore-all
  photorec /dev/sda1               # Data carving
` + "```" + `

**Recovery Procedures:**
` + "```" + `
Disaster recovery runbook template:

1. Assess situation:
   - What failed? (disk, controller, OS, datacenter)
   - What's the blast radius?
   - What's the RPO/RTO?

2. Communication:
   - Notify stakeholders
   - Set up war room / incident channel
   - Assign roles (IC, comms, tech leads)

3. Recovery steps by scenario:

   Single disk failure (RAID):
     a. Identify failed disk: cat /proc/mdstat
     b. Remove from array: mdadm --manage /dev/md0 --remove /dev/sdb1
     c. Replace physical disk
     d. Partition new disk to match
     e. Add to array: mdadm --manage /dev/md0 --add /dev/sdb1
     f. Monitor rebuild: watch cat /proc/mdstat
   
   OS corruption:
     a. Boot from rescue media
     b. Mount filesystems
     c. Check/repair with fsck
     d. If unrepairable: restore from latest backup
     e. Restore bootloader
     f. Verify boot
   
   Full server loss:
     a. Provision new hardware/VM
     b. Boot Rear rescue media (or install minimal OS)
     c. Restore from Borg/restic backup
     d. Restore database dumps
     e. Update DNS/load balancer
     f. Verify services
     g. Monitor closely
   
   Datacenter/Cloud region failure:
     a. Activate DR site
     b. Update DNS to DR site
     c. Restore from offsite backups
     d. Verify all services
     e. Plan failback

4. Verification:
   - All services responding
   - Data integrity verified
   - No security compromise
   - Monitoring active

5. Post-incident:
   - Timeline documentation
   - Root cause analysis
   - Update runbooks
   - Test backup restoration
` + "```" + ``,
					CodeExamples: `# Disaster recovery scripts

# 1. System state capture for DR
#!/bin/bash
# Capture system configuration for DR planning
DR_DIR="/backup/dr-info/$(hostname)-$(date +%Y%m%d)"
mkdir -p "$DR_DIR"

echo "Capturing system state for DR planning..."

# Partition layout
fdisk -l > "$DR_DIR/partitions.txt" 2>/dev/null
blkid > "$DR_DIR/blkid.txt" 2>/dev/null
lsblk -f > "$DR_DIR/lsblk.txt" 2>/dev/null
cat /etc/fstab > "$DR_DIR/fstab.txt"
df -h > "$DR_DIR/disk-usage.txt"

# RAID
if [ -f /proc/mdstat ]; then
    cat /proc/mdstat > "$DR_DIR/mdstat.txt"
    mdadm --detail --scan > "$DR_DIR/mdadm-scan.txt" 2>/dev/null
fi

# LVM
if command -v pvs > /dev/null 2>&1; then
    pvs > "$DR_DIR/pvs.txt" 2>/dev/null
    vgs > "$DR_DIR/vgs.txt" 2>/dev/null
    lvs > "$DR_DIR/lvs.txt" 2>/dev/null
    vgcfgbackup -f "$DR_DIR/vg-backup-%s" 2>/dev/null
fi

# Network
ip addr > "$DR_DIR/ip-addr.txt"
ip route > "$DR_DIR/ip-route.txt"
cat /etc/resolv.conf > "$DR_DIR/resolv.conf"
ss -tlnp > "$DR_DIR/listening-ports.txt"

# Services
systemctl list-units --type=service --state=running > "$DR_DIR/running-services.txt"
systemctl list-unit-files --state=enabled > "$DR_DIR/enabled-services.txt"

# Packages
if command -v dpkg > /dev/null 2>&1; then
    dpkg --get-selections > "$DR_DIR/packages-deb.txt"
fi
if command -v rpm > /dev/null 2>&1; then
    rpm -qa --qf '%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}\n' | sort > "$DR_DIR/packages-rpm.txt"
fi

# Users and groups
getent passwd > "$DR_DIR/passwd.txt"
getent group > "$DR_DIR/group.txt"
getent shadow > "$DR_DIR/shadow.txt" 2>/dev/null

# Crontabs
for user in $(cut -d: -f1 /etc/passwd); do
    crontab -u "$user" -l > "$DR_DIR/crontab-$user.txt" 2>/dev/null
done

# Firewall
iptables-save > "$DR_DIR/iptables.txt" 2>/dev/null
nft list ruleset > "$DR_DIR/nftables.txt" 2>/dev/null

# Kernel
uname -a > "$DR_DIR/kernel.txt"
sysctl -a > "$DR_DIR/sysctl.txt" 2>/dev/null

# Create archive
tar czf "${DR_DIR}.tar.gz" -C "$(dirname "$DR_DIR")" "$(basename "$DR_DIR")"
echo "DR info saved to ${DR_DIR}.tar.gz"

# 2. Backup restoration test script
#!/bin/bash
# Automated backup restore verification
set -e

BORG_REPO="/backup/borg-repo"
RESTORE_DIR="/tmp/restore-test-$(date +%s)"
REPORT="/var/log/restore-test.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$REPORT"; }

log "=== Automated Restore Test ==="
log "Restore directory: $RESTORE_DIR"

# Get latest archive
ARCHIVE=$(borg list "$BORG_REPO" --last 1 --format '{name}' 2>/dev/null)
log "Testing archive: $ARCHIVE"

# Extract to temp directory
mkdir -p "$RESTORE_DIR"
cd "$RESTORE_DIR"
borg extract "$BORG_REPO::$ARCHIVE" 2>&1 | tee -a "$REPORT"

# Verify key files exist
PASS=0
FAIL=0

check_file() {
    local file="$1"
    if [ -e "$RESTORE_DIR$file" ]; then
        log "  PASS: $file exists"
        ((PASS++))
    else
        log "  FAIL: $file missing"
        ((FAIL++))
    fi
}

log "Checking critical files..."
check_file "/etc/passwd"
check_file "/etc/shadow"
check_file "/etc/fstab"
check_file "/etc/ssh/sshd_config"

# Check database dumps
if ls "$RESTORE_DIR"/tmp/db-dumps/*.sql 1> /dev/null 2>&1; then
    for dump in "$RESTORE_DIR"/tmp/db-dumps/*.sql; do
        size=$(stat -f%z "$dump" 2>/dev/null || stat -c%s "$dump" 2>/dev/null || echo 0)
        if [ "$size" -gt 100 ]; then
            log "  PASS: $(basename "$dump") ($size bytes)"
            ((PASS++))
        else
            log "  FAIL: $(basename "$dump") is too small ($size bytes)"
            ((FAIL++))
        fi
    done
fi

# Cleanup
rm -rf "$RESTORE_DIR"

log "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    log "WARNING: Some restore checks failed!"
    exit 1
fi
log "All restore checks passed."`,
				},
			},
		},
	})
}
