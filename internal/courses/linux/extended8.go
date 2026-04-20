package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1830,
			Title:       "Linux Virtualization and Cloud",
			Description: "Understand Linux virtualization technologies: KVM/QEMU, libvirt, cloud-init, Vagrant, and preparing Linux systems for cloud environments.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "KVM and QEMU Virtualization",
					Content: `KVM (Kernel-based Virtual Machine) is the standard Linux hypervisor. Combined with QEMU for hardware emulation, it provides full virtualization with near-native performance.

**KVM Architecture:**
` + "```" + `
Components:
  KVM:   Kernel module that turns Linux into a type-1 hypervisor
         Provides CPU and memory virtualization
         Uses hardware extensions (Intel VT-x, AMD-V)
  
  QEMU:  Userspace emulator providing device emulation
         Disk, network, USB, display controllers
         Uses KVM for CPU acceleration
  
  libvirt: Management API and tools
           virsh CLI, virt-manager GUI
           Abstraction layer over KVM/QEMU

Check KVM support:
  # CPU virtualization support
  grep -cE '(vmx|svm)' /proc/cpuinfo
  # vmx = Intel VT-x, svm = AMD-V
  
  # KVM module loaded
  lsmod | grep kvm
  # kvm_intel or kvm_amd should be present
  
  # Install on Ubuntu/Debian
  apt install qemu-kvm libvirt-daemon-system virtinst virt-viewer
  
  # Install on RHEL/CentOS
  dnf install qemu-kvm libvirt virt-install virt-viewer
  
  # Verify
  virsh list --all
  systemctl status libvirtd
` + "```" + `

**Virtual Machine Management:**
` + "```" + `
Creating VMs:
  # virt-install (command line)
  virt-install \
    --name ubuntu-server \
    --ram 4096 \
    --vcpus 2 \
    --disk path=/var/lib/libvirt/images/ubuntu.qcow2,size=40 \
    --os-variant ubuntu22.04 \
    --network bridge=br0 \
    --graphics vnc,listen=0.0.0.0 \
    --cdrom /var/lib/libvirt/boot/ubuntu-22.04-server.iso \
    --boot uefi

  # From cloud image (no install needed)
  virt-install \
    --name test-vm \
    --memory 2048 \
    --vcpus 2 \
    --import \
    --disk /var/lib/libvirt/images/test.qcow2 \
    --os-variant ubuntu22.04 \
    --cloud-init root-password-generate=on \
    --network network=default

virsh commands:
  # Lifecycle
  virsh list --all                 # List all VMs
  virsh start vm-name              # Start VM
  virsh shutdown vm-name           # Graceful shutdown
  virsh destroy vm-name            # Force stop
  virsh reboot vm-name             # Reboot
  virsh suspend vm-name            # Pause
  virsh resume vm-name             # Resume
  virsh undefine vm-name           # Delete definition
  virsh undefine vm-name --remove-all-storage  # Delete + disks
  
  # Configuration
  virsh edit vm-name               # Edit XML config
  virsh dumpxml vm-name            # Show XML config
  virsh dominfo vm-name            # Brief info
  virsh vcpuinfo vm-name           # CPU info
  virsh domblklist vm-name         # Disk list
  virsh domiflist vm-name          # Network interfaces
  
  # Resources (live change)
  virsh setvcpus vm-name 4 --live  # Change CPU count
  virsh setmem vm-name 4G --live   # Change memory
  
  # Console
  virsh console vm-name            # Serial console
  virt-viewer vm-name              # GUI console
  
  # Snapshots
  virsh snapshot-create-as vm-name snap1 "Snapshot 1"
  virsh snapshot-list vm-name
  virsh snapshot-revert vm-name snap1
  virsh snapshot-delete vm-name snap1
  
  # Autostart
  virsh autostart vm-name
  virsh autostart --disable vm-name
` + "```" + `

**Disk Images:**
` + "```" + `
Image formats:
  raw:    Fixed or sparse raw disk image
          + Best performance (no overhead)
          - No features (no snapshots, large size)
          
  qcow2:  QEMU Copy-On-Write v2
          + Snapshots, compression, encryption
          + Thin provisioning (grows on demand)
          + Backing files (linked clones)
          - Small performance overhead

qemu-img commands:
  # Create image
  qemu-img create -f qcow2 disk.qcow2 40G
  
  # Info
  qemu-img info disk.qcow2
  
  # Convert formats
  qemu-img convert -f raw -O qcow2 disk.raw disk.qcow2
  qemu-img convert -f qcow2 -O raw disk.qcow2 disk.raw
  
  # Resize
  qemu-img resize disk.qcow2 +20G
  
  # Create linked clone (fast, uses backing file)
  qemu-img create -f qcow2 -b base.qcow2 -F qcow2 clone.qcow2
  
  # Compress
  qemu-img convert -O qcow2 -c original.qcow2 compressed.qcow2
  
  # Snapshot management
  qemu-img snapshot -c snap1 disk.qcow2
  qemu-img snapshot -l disk.qcow2
  qemu-img snapshot -a snap1 disk.qcow2  # Revert
  qemu-img snapshot -d snap1 disk.qcow2  # Delete
` + "```" + `

**Networking:**
` + "```" + `
Network modes:
  NAT (default):
    - VMs share host IP via NAT
    - VMs can reach internet
    - Host can reach VMs (via port forwarding)
    - Other machines can't reach VMs directly
    
  Bridged:
    - VMs get IPs on host's network
    - Full network access like physical machines
    - Best for servers
    
  Isolated:
    - VMs can only talk to each other
    - No external access
    
  Macvtap:
    - Direct attachment to physical NIC
    - Each VM gets unique MAC
    - Fast, but host can't communicate with VMs

Bridge setup:
  # Create bridge
  ip link add br0 type bridge
  ip link set eth0 master br0
  ip link set br0 up
  dhclient br0
  
  # Or via Netplan
  network:
    version: 2
    ethernets:
      eth0:
        dhcp4: no
    bridges:
      br0:
        interfaces: [eth0]
        dhcp4: yes

  # Define in libvirt
  virsh net-define bridge.xml
  virsh net-start bridge
  virsh net-autostart bridge
` + "```" + ``,
					CodeExamples: `# Virtualization management scripts

# 1. VM provisioning script
#!/bin/bash
set -e

VM_NAME="${1:?Usage: $0 <name> [memory_mb] [vcpus] [disk_gb]}"
MEMORY="${2:-2048}"
VCPUS="${3:-2}"
DISK_GB="${4:-20}"

IMAGE_DIR="/var/lib/libvirt/images"
BASE_IMAGE="$IMAGE_DIR/ubuntu-22.04-server-cloudimg-amd64.img"

if [ ! -f "$BASE_IMAGE" ]; then
    echo "Downloading base image..."
    wget -O "$BASE_IMAGE" \
      "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img"
fi

# Create disk from base image
echo "Creating disk image..."
qemu-img create -f qcow2 -b "$BASE_IMAGE" -F qcow2 \
  "$IMAGE_DIR/${VM_NAME}.qcow2" "${DISK_GB}G"

# Create cloud-init config
TMPDIR=$(mktemp -d)
cat > "$TMPDIR/meta-data" << EOF
instance-id: $VM_NAME
local-hostname: $VM_NAME
EOF

cat > "$TMPDIR/user-data" << EOF
#cloud-config
users:
  - name: admin
    groups: sudo
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "ssh-ed25519 PLACEHOLDER")
packages:
  - qemu-guest-agent
  - curl
runcmd:
  - systemctl enable --now qemu-guest-agent
EOF

# Create cloud-init ISO
genisoimage -output "$IMAGE_DIR/${VM_NAME}-cidata.iso" \
  -volid cidata -joliet -rock "$TMPDIR/user-data" "$TMPDIR/meta-data"
rm -rf "$TMPDIR"

# Create VM
echo "Creating VM: $VM_NAME (${MEMORY}MB RAM, ${VCPUS} vCPUs, ${DISK_GB}GB disk)"
virt-install \
    --name "$VM_NAME" \
    --memory "$MEMORY" \
    --vcpus "$VCPUS" \
    --import \
    --disk "$IMAGE_DIR/${VM_NAME}.qcow2" \
    --disk "$IMAGE_DIR/${VM_NAME}-cidata.iso,device=cdrom" \
    --os-variant ubuntu22.04 \
    --network network=default \
    --graphics none \
    --noautoconsole \
    --channel unix,target.type=virtio,target.name=org.qemu.guest_agent.0

echo "VM $VM_NAME created. Getting IP..."
sleep 15
virsh domifaddr "$VM_NAME" 2>/dev/null || echo "IP not yet available"

# 2. VM backup script
#!/bin/bash
VM_NAME="${1:?Usage: $0 <vm-name>}"
BACKUP_DIR="/backup/vms"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p "$BACKUP_DIR/$VM_NAME"

echo "Creating snapshot for backup..."
virsh snapshot-create-as "$VM_NAME" "backup-$DATE" "Backup snapshot"

echo "Dumping XML configuration..."
virsh dumpxml "$VM_NAME" > "$BACKUP_DIR/$VM_NAME/config-$DATE.xml"

echo "Copying disk images..."
for DISK in $(virsh domblklist "$VM_NAME" --details | awk '/disk/ {print $4}'); do
    BASENAME=$(basename "$DISK")
    echo "  Backing up $BASENAME..."
    cp "$DISK" "$BACKUP_DIR/$VM_NAME/${BASENAME}-${DATE}"
done

echo "Removing backup snapshot..."
virsh snapshot-delete "$VM_NAME" "backup-$DATE"

echo "Backup complete: $BACKUP_DIR/$VM_NAME/"
ls -lh "$BACKUP_DIR/$VM_NAME/"

# 3. VM inventory
#!/bin/bash
echo "=== Virtual Machine Inventory ==="
printf "%-20s %-10s %-6s %-8s %-15s\n" "NAME" "STATE" "vCPUs" "MEMORY" "IP"
echo "-----------------------------------------------------------"

for VM in $(virsh list --all --name); do
    [ -z "$VM" ] && continue
    STATE=$(virsh domstate "$VM" 2>/dev/null)
    VCPUS=$(virsh dominfo "$VM" 2>/dev/null | awk '/CPU/ {print $2; exit}')
    MEM=$(virsh dominfo "$VM" 2>/dev/null | awk '/Max memory/ {printf "%.0fMB", $3/1024}')
    if [ "$STATE" = "running" ]; then
        IP=$(virsh domifaddr "$VM" 2>/dev/null | awk '/ipv4/ {split($4, a, "/"); print a[1]}')
    else
        IP="N/A"
    fi
    printf "%-20s %-10s %-6s %-8s %-15s\n" "$VM" "$STATE" "${VCPUS:-?}" "${MEM:-?}" "${IP:-N/A}"
done`,
				},
				{
					Title: "Cloud-Init and Cloud Preparation",
					Content: `cloud-init is the industry standard for initializing cloud instances. Understanding it is essential for automating Linux deployments in any cloud or virtualization platform.

**cloud-init Overview:**
` + "```" + `
What cloud-init does:
  - Set hostname
  - Configure networking
  - Create users and SSH keys
  - Install packages
  - Run commands on first boot
  - Mount disks
  - Configure DNS
  - Grow filesystem to fill disk
  - Write files

Data sources:
  - Cloud metadata service (AWS, GCP, Azure)
  - Config drive (ISO, partition)
  - NoCloud (local files or ISO)
  - MAAS
  - VMware customization

Boot stages:
  1. Generator: detect data source
  2. Local: apply networking config (before network is up)
  3. Network: fetch remote config, setup users/SSH
  4. Config: run configuration modules
  5. Final: run user scripts, install packages

Configuration files:
  /etc/cloud/cloud.cfg           Main config
  /etc/cloud/cloud.cfg.d/*.cfg   Override configs
  /var/lib/cloud/                State directory
  /var/log/cloud-init.log        Log file
  /var/log/cloud-init-output.log Script output
` + "```" + `

**cloud-config YAML:**
` + "```" + `yaml
#cloud-config

# Hostname
hostname: web-server-01
fqdn: web-server-01.example.com
manage_etc_hosts: true

# Users
users:
  - default
  - name: deploy
    gecos: Deploy User
    groups: [sudo, docker]
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ssh-ed25519 AAAA... user@host
    lock_passwd: true

# SSH
ssh_pwauth: false
disable_root: true
ssh_deletekeys: true
ssh_genkeytypes: [ed25519, rsa]

# Packages
package_update: true
package_upgrade: true
packages:
  - curl
  - wget
  - vim
  - htop
  - docker.io
  - fail2ban

# Write files
write_files:
  - path: /etc/sysctl.d/99-custom.conf
    content: |
      net.core.somaxconn = 65535
      vm.swappiness = 10
    owner: root:root
    permissions: '0644'
  - path: /etc/motd
    content: |
      Production Server - Authorized Access Only
    permissions: '0644'

# Disk setup
disk_setup:
  /dev/sdb:
    table_type: gpt
    layout: true
    overwrite: false

fs_setup:
  - filesystem: ext4
    device: /dev/sdb1
    label: data
    overwrite: false

mounts:
  - [/dev/sdb1, /data, ext4, "defaults,noatime", "0", "2"]

# Run commands (run once on first boot)
runcmd:
  - systemctl enable docker
  - systemctl start docker
  - usermod -aG docker deploy
  - sysctl -p /etc/sysctl.d/99-custom.conf
  - ufw allow 22/tcp
  - ufw allow 80/tcp
  - ufw allow 443/tcp
  - ufw --force enable

# Boot commands (run every boot, before networking)
bootcmd:
  - echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf

# Final message
final_message: "Cloud-init complete. Uptime: $UPTIME seconds"

# Phone home (notify orchestration system)
phone_home:
  url: https://orchestrator.example.com/api/ready
  post: [instance_id, hostname]
  tries: 3
` + "```" + `

**Preparing Linux for Cloud:**
` + "```" + `
Image preparation checklist:
  1. Install cloud-init
     apt install cloud-init
  
  2. Configure cloud-init datasources
     /etc/cloud/cloud.cfg:
       datasource_list: [NoCloud, ConfigDrive, None]
  
  3. Clean SSH host keys (regenerated on first boot)
     rm -f /etc/ssh/ssh_host_*
  
  4. Clean machine-id
     truncate -s0 /etc/machine-id
     rm -f /var/lib/dbus/machine-id
  
  5. Remove persistent network rules
     rm -f /etc/udev/rules.d/70-persistent-net.rules
  
  6. Clean logs
     find /var/log -type f -exec truncate -s0 {} \;
  
  7. Clean tmp
     rm -rf /tmp/* /var/tmp/*
  
  8. Clean package cache
     apt clean
  
  9. Clean cloud-init state
     cloud-init clean
  
  10. Clean bash history
     unset HISTFILE
     rm -f /root/.bash_history
     rm -f /home/*/.bash_history

  # One-liner for image cleanup:
  virt-sysprep -d vm-name
  # Or:
  virt-sysprep -a disk.qcow2

Packer (automated image building):
  Packer automates creating machine images:
  - Define template (HCL or JSON)
  - Build on various platforms (AWS, GCP, Azure, QEMU)
  - Provision with shell scripts, Ansible, etc.
  - Output: AMI, qcow2, VMDK, etc.
  
  # Integrate with existing automation
  packer build template.pkr.hcl

Vagrant (development environments):
  # Vagrantfile
  Vagrant.configure("2") do |config|
    config.vm.box = "ubuntu/jammy64"
    config.vm.hostname = "dev-server"
    config.vm.network "private_network", ip: "192.168.56.10"
    config.vm.provider "libvirt" do |v|
      v.memory = 4096
      v.cpus = 2
    end
    config.vm.provision "shell", inline: <<-SHELL
      apt-get update
      apt-get install -y docker.io
    SHELL
  end
  
  vagrant up
  vagrant ssh
  vagrant halt
  vagrant destroy
` + "```" + ``,
					CodeExamples: `# Cloud preparation and automation

# 1. Image preparation script
#!/bin/bash
set -e

echo "Preparing system for cloud image..."

# Stop services
systemctl stop rsyslog 2>/dev/null || true
systemctl stop cloud-init 2>/dev/null || true

# Clean cloud-init
cloud-init clean --logs 2>/dev/null || true
rm -rf /var/lib/cloud/*

# Clean SSH host keys
rm -f /etc/ssh/ssh_host_*

# Clean machine-id
truncate -s0 /etc/machine-id
rm -f /var/lib/dbus/machine-id

# Clean network
rm -f /etc/udev/rules.d/70-persistent-net.rules
rm -f /etc/netplan/50-cloud-init.yaml

# Clean package cache
apt-get -y autoremove
apt-get -y clean

# Clean logs
find /var/log -type f -exec truncate -s0 {} \;
rm -f /var/log/*.gz
rm -f /var/log/*.[0-9]

# Clean tmp
rm -rf /tmp/* /var/tmp/*

# Clean bash history
unset HISTFILE
rm -f /root/.bash_history
rm -f /home/*/.bash_history
history -c

# Zero free space for better compression
dd if=/dev/zero of=/EMPTY bs=1M 2>/dev/null || true
rm -f /EMPTY
sync

echo "Image preparation complete"

# 2. Cloud-init debugging
#!/bin/bash
echo "=== Cloud-Init Status ==="
cloud-init status --long

echo ""
echo "=== Cloud-Init Errors ==="
grep -i "error\|warn\|fail" /var/log/cloud-init.log | tail -20

echo ""
echo "=== Data Source ==="
cloud-init query ds | head -5 2>/dev/null

echo ""
echo "=== Instance Metadata ==="
cloud-init query instance_id 2>/dev/null
cloud-init query local_hostname 2>/dev/null
cloud-init query region 2>/dev/null

echo ""
echo "=== User Data ==="
cloud-init query userdata 2>/dev/null | head -20

echo ""
echo "=== Network Config ==="
cat /var/lib/cloud/instance/network-config.json 2>/dev/null | head -20

# Re-run cloud-init (testing)
# cloud-init clean && cloud-init init && cloud-init modules --mode config && cloud-init modules --mode final

# 3. Packer template for QEMU
# template.pkr.hcl
# packer {
#   required_plugins {
#     qemu = {
#       version = "~> 1"
#       source  = "github.com/hashicorp/qemu"
#     }
#   }
# }
#
# source "qemu" "ubuntu" {
#   iso_url          = "https://releases.ubuntu.com/22.04/ubuntu-22.04.3-live-server-amd64.iso"
#   iso_checksum     = "sha256:..."
#   output_directory = "output"
#   vm_name          = "ubuntu-base.qcow2"
#   format           = "qcow2"
#   disk_size        = "20G"
#   memory           = 2048
#   cpus             = 2
#   headless         = true
#   ssh_username     = "packer"
#   ssh_password     = "packer"
#   ssh_timeout      = "30m"
#   boot_command     = ["...autoinstall..."]
# }
#
# build {
#   sources = ["source.qemu.ubuntu"]
#   
#   provisioner "shell" {
#     scripts = ["scripts/base.sh", "scripts/cleanup.sh"]
#   }
# }

# 4. Vagrantfile for multi-VM environment
# -*- mode: ruby -*-
# Vagrant.configure("2") do |config|
#   config.vm.box = "generic/ubuntu2204"
#   
#   # Web servers
#   (1..3).each do |i|
#     config.vm.define "web#{i}" do |web|
#       web.vm.hostname = "web#{i}"
#       web.vm.network "private_network", ip: "192.168.56.#{10+i}"
#       web.vm.provider "libvirt" do |v|
#         v.memory = 1024
#         v.cpus = 1
#       end
#     end
#   end
#   
#   # Database
#   config.vm.define "db" do |db|
#     db.vm.hostname = "db"
#     db.vm.network "private_network", ip: "192.168.56.20"
#     db.vm.provider "libvirt" do |v|
#       v.memory = 2048
#       v.cpus = 2
#     end
#   end
# end`,
				},
			},
		},
	})
}
