package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1826,
			Title:       "Storage and Filesystem Management",
			Description: "Master Linux storage: disk partitioning, LVM, RAID, filesystem types, mount options, and storage performance tuning.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Disk Partitioning and Filesystems",
					Content: `Understanding disk partitioning, filesystem types, and mount management is fundamental to Linux system administration.

**Block Devices and Partitioning:**
` + "```" + `
Block device listing:
  lsblk                          # Tree view of block devices
  lsblk -f                       # With filesystem info
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT
  fdisk -l                       # Detailed partition info
  blkid                          # UUID and filesystem type
  cat /proc/partitions            # Kernel view

Device naming:
  /dev/sda     First SCSI/SATA disk
  /dev/sdb     Second SCSI/SATA disk
  /dev/sda1    First partition on sda
  /dev/nvme0n1 First NVMe disk
  /dev/nvme0n1p1  First partition on NVMe
  /dev/vda     First virtio disk (VM)
  /dev/xvda    First Xen virtual disk

Partition schemes:
  MBR (Master Boot Record):
    - Max 4 primary partitions (or 3 primary + 1 extended)
    - Max disk size: 2 TB
    - Legacy BIOS boot
    
  GPT (GUID Partition Table):
    - Up to 128 partitions (default)
    - Max disk size: 8 ZB (practically unlimited)
    - Required for UEFI boot
    - Always use GPT for new systems

Partitioning tools:
  fdisk /dev/sda                 # MBR (also supports GPT)
  gdisk /dev/sda                 # GPT
  parted /dev/sda                # Both MBR and GPT
  
  # parted examples:
  parted /dev/sda mklabel gpt
  parted /dev/sda mkpart primary ext4 1MiB 100GiB
  parted /dev/sda mkpart primary xfs 100GiB 100%
  parted /dev/sda print
  
  # Inform kernel of partition changes
  partprobe /dev/sda
` + "```" + `

**Filesystem Types:**
` + "```" + `
ext4 (Default for many distros):
  - Journaling filesystem
  - Max file size: 16 TB
  - Max volume: 1 EB
  - Good general-purpose FS
  - Mature, stable, well-tested
  
  mkfs.ext4 /dev/sda1
  mkfs.ext4 -L mydata /dev/sda1           # With label
  tune2fs -l /dev/sda1                     # View parameters
  tune2fs -c 30 /dev/sda1                  # Check every 30 mounts
  resize2fs /dev/sda1                      # Resize (after partition resize)

xfs (Default for RHEL/CentOS):
  - High performance for large files
  - Excellent parallel I/O
  - Max file size: 8 EB
  - Can grow but NOT shrink
  - Best for large storage, databases
  
  mkfs.xfs /dev/sda1
  mkfs.xfs -L mydata /dev/sda1
  xfs_info /dev/sda1
  xfs_growfs /mountpoint                   # Grow filesystem
  xfs_repair /dev/sda1                     # Repair

btrfs (Modern copy-on-write):
  - Snapshots, compression, checksums
  - RAID support built-in
  - Subvolumes
  - Online defrag, balance
  - Good for: desktop, NAS, containers
  
  mkfs.btrfs /dev/sda1
  btrfs subvolume create /mnt/@home
  btrfs subvolume snapshot /mnt/@home /mnt/@home-snap
  btrfs filesystem show
  btrfs balance start --full-balance /mnt

Special filesystems:
  tmpfs    Memory-backed (fast, volatile)
  proc     Process information (/proc)
  sysfs    Device/driver info (/sys)
  devtmpfs Device nodes (/dev)
  overlay  Union mount (containers use this)
  nfs      Network File System
  cifs     SMB/CIFS shares
` + "```" + `

**Mounting:**
` + "```" + `
Mount commands:
  mount /dev/sda1 /mnt                     # Basic mount
  mount -t ext4 /dev/sda1 /mnt            # Specify type
  mount -o rw,noatime /dev/sda1 /mnt      # With options
  mount UUID=xxxx /mnt                      # By UUID (preferred)
  umount /mnt                               # Unmount
  umount -l /mnt                           # Lazy unmount (busy FS)

Common mount options:
  defaults    rw,suid,dev,exec,auto,nouser,async
  rw/ro       Read-write / Read-only
  noatime     Don't update access time (performance)
  relatime    Update atime only if older than mtime
  nosuid      Ignore SUID/SGID bits
  noexec      Don't allow execution
  nodev       Ignore device files
  sync        Synchronous I/O
  errors=remount-ro  Remount read-only on errors
  discard     Enable TRIM for SSDs
  compress=zstd  Enable compression (btrfs)

/etc/fstab (persistent mounts):
  # <device>         <mount>   <type> <options>           <dump> <pass>
  UUID=xxx-yyy       /         ext4   defaults            0      1
  UUID=aaa-bbb       /home     ext4   defaults,noatime    0      2
  UUID=ccc-ddd       /data     xfs    defaults,noatime    0      2
  tmpfs              /tmp      tmpfs  defaults,noatime,size=4G  0  0
  /dev/sdb1          /backup   ext4   defaults,noauto     0      0
  
  # NFS mount
  nfs-server:/share  /nfs      nfs    defaults,_netdev    0      0
  
  # pass values: 0=skip fsck, 1=root, 2=other
  
  # Test fstab without reboot:
  mount -a
  findmnt --verify

systemd mount units:
  /etc/systemd/system/data.mount:
    [Unit]
    Description=Data Partition
    
    [Mount]
    What=/dev/sdb1
    Where=/data
    Type=ext4
    Options=defaults,noatime
    
    [Install]
    WantedBy=multi-user.target
  
  systemctl enable data.mount
  systemctl start data.mount
` + "```" + ``,
					CodeExamples: `# Storage management examples

# 1. Disk setup script
#!/bin/bash
DISK="/dev/sdb"

# Create GPT partition table
parted -s "$DISK" mklabel gpt

# Create partitions
parted -s "$DISK" mkpart primary ext4 1MiB 50%
parted -s "$DISK" mkpart primary xfs 50% 100%

# Inform kernel
partprobe "$DISK"
sleep 1

# Create filesystems
mkfs.ext4 -L data1 "${DISK}1"
mkfs.xfs -L data2 "${DISK}2"

# Create mount points
mkdir -p /data1 /data2

# Get UUIDs
UUID1=$(blkid -s UUID -o value "${DISK}1")
UUID2=$(blkid -s UUID -o value "${DISK}2")

# Add to fstab
cat >> /etc/fstab << EOF
UUID=$UUID1  /data1  ext4  defaults,noatime  0  2
UUID=$UUID2  /data2  xfs   defaults,noatime  0  2
EOF

# Mount
mount -a

echo "Disk setup complete"
lsblk "$DISK"

# 2. Filesystem health check
#!/bin/bash
echo "=== Filesystem Usage ==="
df -hT | grep -v tmpfs | grep -v devtmpfs

echo ""
echo "=== Inode Usage ==="
df -i | grep -v tmpfs | grep -v devtmpfs

echo ""
echo "=== Large Files (>100MB) ==="
find / -xdev -type f -size +100M -exec ls -lh {} \; 2>/dev/null | \
  sort -k5 -h | tail -20

echo ""
echo "=== Disk I/O Statistics ==="
iostat -xz 1 3 2>/dev/null || echo "iostat not available (install sysstat)"

echo ""
echo "=== Mount Options ==="
findmnt -t ext4,xfs,btrfs -o TARGET,SOURCE,FSTYPE,OPTIONS

# 3. SSD optimization
#!/bin/bash
# Enable TRIM timer
systemctl enable fstrim.timer
systemctl start fstrim.timer

# Set I/O scheduler for SSDs
for DEV in /sys/block/sd* /sys/block/nvme*; do
    if [ -f "$DEV/queue/rotational" ]; then
        ROTATIONAL=$(cat "$DEV/queue/rotational")
        DEVNAME=$(basename "$DEV")
        if [ "$ROTATIONAL" = "0" ]; then
            echo "mq-deadline" > "$DEV/queue/scheduler" 2>/dev/null
            echo "$DEVNAME: SSD detected, scheduler set to mq-deadline"
        else
            echo "bfq" > "$DEV/queue/scheduler" 2>/dev/null
            echo "$DEVNAME: HDD detected, scheduler set to bfq"
        fi
    fi
done

# Recommended fstab options for SSD:
# UUID=xxx /  ext4  defaults,noatime,discard  0  1
# Or use fstrim.timer instead of discard mount option

# 4. btrfs snapshot management
#!/bin/bash
SNAP_DIR="/mnt/@snapshots"
SUBVOL="/mnt/@root"
DATE=$(date +%Y%m%d-%H%M%S)

# Create snapshot
btrfs subvolume snapshot -r "$SUBVOL" "$SNAP_DIR/$DATE"

# List snapshots
btrfs subvolume list -s /mnt

# Delete old snapshots (keep last 7)
SNAPS=$(ls -1d "$SNAP_DIR"/* 2>/dev/null | sort -r)
COUNT=0
for SNAP in $SNAPS; do
    COUNT=$((COUNT + 1))
    if [ $COUNT -gt 7 ]; then
        echo "Deleting old snapshot: $SNAP"
        btrfs subvolume delete "$SNAP"
    fi
done`,
				},
				{
					Title: "LVM and RAID",
					Content: `LVM (Logical Volume Manager) provides flexible storage management, while RAID ensures data redundancy and performance.

**LVM Architecture:**
` + "```" + `
LVM layers:
  Physical Volume (PV) → Volume Group (VG) → Logical Volume (LV)
  
  Physical Volume: actual disk or partition
  Volume Group: pool of storage from one or more PVs
  Logical Volume: virtual partition carved from VG

Advantages:
  - Resize volumes without unmounting (online with ext4/xfs)
  - Snapshots for backups
  - Span multiple disks
  - Move data between disks (pvmove)
  - Thin provisioning (overcommit)

Commands:
  Physical Volumes:
    pvcreate /dev/sdb /dev/sdc     # Initialize PVs
    pvs                             # List PVs (brief)
    pvdisplay                       # Detailed PV info
    pvremove /dev/sdc               # Remove PV
  
  Volume Groups:
    vgcreate datavg /dev/sdb /dev/sdc  # Create VG
    vgs                             # List VGs
    vgdisplay datavg                # Detailed VG info
    vgextend datavg /dev/sdd        # Add disk to VG
    vgreduce datavg /dev/sdc        # Remove disk from VG
  
  Logical Volumes:
    lvcreate -L 100G -n data datavg        # Fixed size
    lvcreate -l 100%FREE -n data datavg    # Use all free space
    lvcreate -l 50%VG -n data datavg       # 50% of VG
    lvs                                     # List LVs
    lvdisplay /dev/datavg/data             # Detailed info
    
  Resize:
    # Extend LV and filesystem
    lvextend -L +50G /dev/datavg/data      # Add 50G
    lvextend -l +100%FREE /dev/datavg/data # Use all free
    resize2fs /dev/datavg/data             # Grow ext4
    xfs_growfs /mountpoint                 # Grow xfs
    
    # Or extend both at once:
    lvextend -r -L +50G /dev/datavg/data   # -r resizes FS too
    
  Reduce (ext4 only, not xfs):
    umount /dev/datavg/data
    e2fsck -f /dev/datavg/data
    resize2fs /dev/datavg/data 50G
    lvreduce -L 50G /dev/datavg/data

LVM Snapshots:
  # Create snapshot (needs free space in VG)
  lvcreate -L 10G -s -n data-snap /dev/datavg/data
  
  # Mount snapshot (read-only)
  mount -o ro /dev/datavg/data-snap /mnt/snapshot
  
  # Restore from snapshot
  lvconvert --merge /dev/datavg/data-snap
  # Requires unmount + reactivation
  
  # Delete snapshot
  lvremove /dev/datavg/data-snap

LVM Thin Provisioning:
  # Create thin pool (100G actual, can provision more)
  lvcreate -L 100G --thinpool thinpool datavg
  
  # Create thin volumes (overprovisioned)
  lvcreate -V 50G --thin -n vm1 datavg/thinpool
  lvcreate -V 50G --thin -n vm2 datavg/thinpool
  lvcreate -V 50G --thin -n vm3 datavg/thinpool
  # 150G provisioned on 100G pool!
  
  # Monitor usage
  lvs -a datavg
  # Watch data% and metadata% columns
` + "```" + `

**Software RAID (mdadm):**
` + "```" + `
RAID levels:
  RAID 0 (stripe):   N disks, Nx performance, 0 redundancy
  RAID 1 (mirror):   2+ disks, 1x capacity, N-1 disk failures
  RAID 5 (parity):   3+ disks, (N-1)x capacity, 1 disk failure
  RAID 6 (dual parity): 4+ disks, (N-2)x capacity, 2 disk failures
  RAID 10 (mirror+stripe): 4+ disks, N/2 capacity, fast + redundant

Creating RAID:
  # RAID 1 (mirror)
  mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb /dev/sdc
  
  # RAID 5
  mdadm --create /dev/md0 --level=5 --raid-devices=3 /dev/sdb /dev/sdc /dev/sdd
  
  # RAID 10
  mdadm --create /dev/md0 --level=10 --raid-devices=4 \
    /dev/sdb /dev/sdc /dev/sdd /dev/sde
  
  # With spare drive
  mdadm --create /dev/md0 --level=5 --raid-devices=3 --spare-devices=1 \
    /dev/sdb /dev/sdc /dev/sdd /dev/sde

Managing RAID:
  # Status
  cat /proc/mdstat
  mdadm --detail /dev/md0
  
  # Save config
  mdadm --detail --scan >> /etc/mdadm/mdadm.conf
  
  # Replace failed disk
  mdadm --manage /dev/md0 --fail /dev/sdc      # Mark as failed
  mdadm --manage /dev/md0 --remove /dev/sdc    # Remove
  # (physically replace disk)
  mdadm --manage /dev/md0 --add /dev/sdc       # Add new disk
  # Rebuild starts automatically
  
  # Watch rebuild progress
  watch cat /proc/mdstat
  
  # Add spare
  mdadm --manage /dev/md0 --add-spare /dev/sde
  
  # Grow array (add disk to RAID 5)
  mdadm --grow /dev/md0 --raid-devices=4 --add /dev/sde

Putting it together (LVM on RAID):
  1. Create RAID array: mdadm --create /dev/md0 ...
  2. Create PV: pvcreate /dev/md0
  3. Create VG: vgcreate datavg /dev/md0
  4. Create LV: lvcreate -l 100%FREE -n data datavg
  5. Format: mkfs.ext4 /dev/datavg/data
  6. Mount: mount /dev/datavg/data /data
  → RAID for redundancy, LVM for flexibility
` + "```" + ``,
					CodeExamples: `# LVM and RAID management scripts

# 1. Complete LVM setup
#!/bin/bash
set -e

DISKS="/dev/sdb /dev/sdc"
VG_NAME="datavg"
LV_NAME="appdata"
MOUNT="/data"

echo "Setting up LVM on $DISKS..."

# Create physical volumes
for disk in $DISKS; do
    pvcreate "$disk"
done

# Create volume group
vgcreate "$VG_NAME" $DISKS

# Create logical volume (90% of space, leave room for snapshots)
lvcreate -l 90%VG -n "$LV_NAME" "$VG_NAME"

# Create filesystem
mkfs.ext4 -L "$LV_NAME" "/dev/$VG_NAME/$LV_NAME"

# Mount
mkdir -p "$MOUNT"
UUID=$(blkid -s UUID -o value "/dev/$VG_NAME/$LV_NAME")
echo "UUID=$UUID  $MOUNT  ext4  defaults,noatime  0  2" >> /etc/fstab
mount "$MOUNT"

echo "LVM setup complete:"
lvs "$VG_NAME"
df -h "$MOUNT"

# 2. LVM monitoring and alerting
#!/bin/bash
THRESHOLD=85

for VG in $(vgs --noheadings -o vg_name 2>/dev/null); do
    VG_FREE_PCT=$(vgs --noheadings -o vg_free_count,vg_extent_count "$VG" | \
      awk '{printf "%d\n", ($1/$2)*100}')
    VG_USED=$((100 - VG_FREE_PCT))
    
    if [ "$VG_USED" -gt "$THRESHOLD" ]; then
        echo "WARNING: VG $VG is ${VG_USED}% full"
    fi
done

for LV_PATH in $(lvs --noheadings -o lv_path 2>/dev/null); do
    if findmnt -n "$LV_PATH" > /dev/null 2>&1; then
        MOUNT=$(findmnt -n -o TARGET "$LV_PATH")
        USED=$(df --output=pcent "$MOUNT" | tail -1 | tr -d '% ')
        
        if [ "$USED" -gt "$THRESHOLD" ]; then
            echo "WARNING: $LV_PATH ($MOUNT) is ${USED}% full"
        fi
    fi
done

# 3. RAID health monitoring
#!/bin/bash
# RAID health check
echo "=== RAID Status ==="
cat /proc/mdstat

echo ""
for MD in /dev/md*; do
    if [ -b "$MD" ]; then
        STATE=$(mdadm --detail "$MD" 2>/dev/null | grep "State :" | awk -F: '{print $2}' | xargs)
        DEGRADED=$(mdadm --detail "$MD" 2>/dev/null | grep "Degraded" | awk '{print $NF}')
        
        if [ "$STATE" != "clean" ] && [ "$STATE" != "active" ]; then
            echo "ALERT: $MD state is: $STATE"
        fi
        if [ "$DEGRADED" != "0" ] && [ -n "$DEGRADED" ]; then
            echo "ALERT: $MD is DEGRADED (missing $DEGRADED device(s))"
        fi
    fi
done

# 4. Disk replacement procedure
#!/bin/bash
# Replace failed disk in RAID 1
# Usage: ./replace-disk.sh /dev/md0 /dev/sdc /dev/sde
ARRAY=$1
OLD_DISK=$2
NEW_DISK=$3

echo "Replacing $OLD_DISK with $NEW_DISK in $ARRAY"

# Mark as failed and remove
mdadm --manage "$ARRAY" --fail "$OLD_DISK"
mdadm --manage "$ARRAY" --remove "$OLD_DISK"

# Copy partition table (GPT)
sgdisk -R "$NEW_DISK" "$OLD_DISK"
sgdisk -G "$NEW_DISK"

# Add new disk
mdadm --manage "$ARRAY" --add "$NEW_DISK"

echo "Rebuild started. Monitor with: watch cat /proc/mdstat"
cat /proc/mdstat`,
				},
			},
		},
	})
}
