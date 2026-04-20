package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1836,
			Title:       "Linux Containers and Namespaces",
			Description: "Understand the Linux kernel features that enable containers: namespaces, cgroups, overlay filesystems, and how container runtimes leverage them.",
			Order:       36,
			Lessons: []problems.Lesson{
				{
					Title: "Container Fundamentals: Namespaces and Cgroups",
					Content: `Containers are not a kernel feature but a combination of kernel features: namespaces for isolation, cgroups for resource limits, and overlay filesystems for layered images.

**Linux Namespaces:**
` + "```" + `
Namespaces isolate system resources per process group:

Type        Isolates                        Flag
------      -----------                     -------
Mount       Filesystem mount points         CLONE_NEWNS
UTS         Hostname and domain name        CLONE_NEWUTS
IPC         Inter-process communication     CLONE_NEWIPC
Network     Network devices, stacks, ports  CLONE_NEWNET
PID         Process IDs                     CLONE_NEWPID
User        User and group IDs              CLONE_NEWUSER
Cgroup      Cgroup root directory           CLONE_NEWCGROUP
Time        Boot and monotonic clocks       CLONE_NEWTIME

Viewing namespaces:
  lsns                              # List all namespaces
  lsns -t net                       # List network namespaces
  ls -la /proc/$$/ns/               # Current process namespaces
  
  # Each namespace is a file under /proc/<pid>/ns/
  ls /proc/1/ns/
  cgroup  ipc  mnt  net  pid  pid_for_children  user  uts

Creating namespaces manually:
  # New UTS namespace (own hostname)
  unshare --uts /bin/bash
  hostname container-test            # Only affects this namespace
  exit
  hostname                           # Original hostname unchanged
  
  # New PID namespace
  unshare --pid --fork --mount-proc /bin/bash
  ps aux                             # Only see processes in this ns
  # PID 1 is our bash, completely isolated
  
  # New network namespace
  unshare --net /bin/bash
  ip link                            # Only loopback
  ip addr                            # No external interfaces
  
  # Combine multiple namespaces
  unshare --pid --net --uts --mount --ipc --fork /bin/bash
  # Almost a container!

Network namespace with connectivity:
  # Create named network namespace
  ip netns add myns
  
  # Create veth pair
  ip link add veth0 type veth peer name veth1
  
  # Move one end into namespace
  ip link set veth1 netns myns
  
  # Configure host side
  ip addr add 10.200.1.1/24 dev veth0
  ip link set veth0 up
  
  # Configure namespace side
  ip netns exec myns ip addr add 10.200.1.2/24 dev veth1
  ip netns exec myns ip link set veth1 up
  ip netns exec myns ip link set lo up
  
  # Enable routing
  ip netns exec myns ip route add default via 10.200.1.1
  sysctl -w net.ipv4.ip_forward=1
  iptables -t nat -A POSTROUTING -s 10.200.1.0/24 -j MASQUERADE
  
  # Test
  ip netns exec myns ping 10.200.1.1
  ip netns exec myns curl example.com
  
  # Run process in namespace
  ip netns exec myns python3 -m http.server 8080
  
  # List namespaces
  ip netns list
  
  # Delete
  ip netns delete myns
` + "```" + `

**Cgroups v2:**
` + "```" + `
Cgroups control resource allocation and accounting.

Cgroup v2 unified hierarchy (modern):
  mount | grep cgroup
  # cgroup2 on /sys/fs/cgroup type cgroup2
  
  # Hierarchy
  /sys/fs/cgroup/
  ├── cgroup.controllers       # Available controllers
  ├── cgroup.subtree_control   # Active controllers for children
  ├── system.slice/            # System services
  │   ├── sshd.service/
  │   └── nginx.service/
  ├── user.slice/              # User sessions
  │   └── user-1000.slice/
  └── mygroup/                 # Custom cgroup

Controllers:
  cpu      CPU time allocation
  cpuset   CPU and memory node pinning
  memory   Memory limits and accounting
  io       Block I/O limits
  pids     Process count limits
  rdma     RDMA resource limits

Creating cgroups:
  # Enable controllers
  echo "+cpu +memory +io +pids" > /sys/fs/cgroup/cgroup.subtree_control
  
  # Create cgroup
  mkdir /sys/fs/cgroup/myapp
  
  # Set memory limit (100MB)
  echo 104857600 > /sys/fs/cgroup/myapp/memory.max
  echo 83886080 > /sys/fs/cgroup/myapp/memory.high  # Soft limit (80MB)
  
  # Set CPU limit (50% of one core)
  echo "50000 100000" > /sys/fs/cgroup/myapp/cpu.max
  # Means: 50000 microseconds per 100000 microsecond period
  
  # Set CPU weight (relative, default 100)
  echo 50 > /sys/fs/cgroup/myapp/cpu.weight
  
  # Limit IO (1MB/s write to device 8:0)
  echo "8:0 wbps=1048576" > /sys/fs/cgroup/myapp/io.max
  
  # Limit process count
  echo 100 > /sys/fs/cgroup/myapp/pids.max
  
  # Add current process
  echo $$ > /sys/fs/cgroup/myapp/cgroup.procs
  
  # Check current usage
  cat /sys/fs/cgroup/myapp/memory.current
  cat /sys/fs/cgroup/myapp/cpu.stat
  cat /sys/fs/cgroup/myapp/pids.current

Systemd cgroup integration:
  # View cgroup tree
  systemd-cgls
  
  # View resource usage
  systemd-cgtop
  
  # Set resource limits for service
  systemctl set-property nginx.service MemoryMax=512M
  systemctl set-property nginx.service CPUQuota=200%  # 2 cores
  systemctl set-property nginx.service TasksMax=256
  
  # In unit file
  [Service]
  MemoryMax=512M
  MemoryHigh=400M
  CPUQuota=200%
  CPUWeight=50
  TasksMax=256
  IOWeight=50
` + "```" + `

**Overlay Filesystem:**
` + "```" + `
OverlayFS enables container image layering:

  # Layer structure
  Lower layers:  Read-only (base image, intermediate layers)
  Upper layer:   Read-write (container changes)
  Work dir:      Temporary workspace for atomic operations
  Merged:        Union view (what the container sees)
  
  # Mount overlay
  mount -t overlay overlay \
    -o lowerdir=/lower1:/lower2,upperdir=/upper,workdir=/work \
    /merged
  
  # Example: create container-like environment
  # Setup directories
  mkdir -p /container/{lower,upper,work,merged}
  
  # Create base "image" (use debootstrap or copy)
  debootstrap focal /container/lower http://archive.ubuntu.com/ubuntu
  
  # Mount overlay
  mount -t overlay overlay \
    -o lowerdir=/container/lower,upperdir=/container/upper,workdir=/container/work \
    /container/merged
  
  # Changes in /container/merged go to /container/upper
  touch /container/merged/newfile
  ls /container/upper/newfile    # File is in upper layer
  
  # Lower layer unchanged
  ls /container/lower/newfile    # Not found!
  
  # Deleted files: whiteout files in upper layer
  rm /container/merged/etc/hostname
  ls -la /container/upper/etc/hostname
  # Shows as character device 0,0 (whiteout)
` + "```" + ``,
					CodeExamples: `# Container primitives

# 1. Minimal container from scratch
#!/bin/bash
# Build a minimal container using namespaces + cgroups + chroot
set -e

CONTAINER_ROOT="/tmp/container_root"
CONTAINER_NAME="mycontainer"
CGROUP_PATH="/sys/fs/cgroup/$CONTAINER_NAME"

# Create filesystem (minimal)
mkdir -p "$CONTAINER_ROOT"/{bin,lib,lib64,proc,sys,dev,etc,tmp}

# Copy busybox for basic commands
cp /bin/busybox "$CONTAINER_ROOT/bin/"
# Create symlinks for common commands
for cmd in sh ls cat echo ps mount mkdir; do
    ln -sf busybox "$CONTAINER_ROOT/bin/$cmd"
done

# Copy required libraries
for lib in $(ldd /bin/busybox | grep -oP '/lib\S+'); do
    dir=$(dirname "$lib")
    mkdir -p "$CONTAINER_ROOT$dir"
    cp "$lib" "$CONTAINER_ROOT$lib" 2>/dev/null || true
done

# Create cgroup
mkdir -p "$CGROUP_PATH"
echo "52428800" > "$CGROUP_PATH/memory.max"  # 50MB
echo "100" > "$CGROUP_PATH/pids.max"

# Setup resolv.conf
echo "nameserver 8.8.8.8" > "$CONTAINER_ROOT/etc/resolv.conf"

# Run in new namespaces
unshare --pid --net --uts --mount --ipc --fork \
    --map-root-user --map-current-user \
    /bin/sh -c "
    # Set hostname
    hostname $CONTAINER_NAME
    
    # Mount proc and sys
    mount -t proc proc $CONTAINER_ROOT/proc
    mount -t sysfs sys $CONTAINER_ROOT/sys
    mount -t tmpfs tmpfs $CONTAINER_ROOT/tmp
    
    # Create minimal dev nodes
    mount -t tmpfs tmpfs $CONTAINER_ROOT/dev
    mknod -m 666 $CONTAINER_ROOT/dev/null c 1 3
    mknod -m 666 $CONTAINER_ROOT/dev/zero c 1 5
    mknod -m 666 $CONTAINER_ROOT/dev/random c 1 8
    mknod -m 666 $CONTAINER_ROOT/dev/urandom c 1 9
    
    # Pivot root
    mkdir -p $CONTAINER_ROOT/.old_root
    pivot_root $CONTAINER_ROOT $CONTAINER_ROOT/.old_root
    umount -l /.old_root 2>/dev/null || true
    rmdir /.old_root 2>/dev/null || true
    
    cd /
    exec /bin/sh
"

# Cleanup
rm -rf "$CONTAINER_ROOT"
rmdir "$CGROUP_PATH" 2>/dev/null || true

# 2. Cgroup resource monitor
#!/bin/bash
# Monitor cgroup resource usage
CGROUP="/sys/fs/cgroup"

echo "=== Cgroup Resource Usage ==="
echo ""

for service_dir in "$CGROUP"/system.slice/*.service; do
    [ -d "$service_dir" ] || continue
    service=$(basename "$service_dir")
    
    # Memory
    mem_current=$(cat "$service_dir/memory.current" 2>/dev/null || echo 0)
    mem_max=$(cat "$service_dir/memory.max" 2>/dev/null || echo "max")
    mem_mb=$((mem_current / 1048576))
    
    # PIDs
    pids=$(cat "$service_dir/pids.current" 2>/dev/null || echo 0)
    
    # CPU
    cpu_usage=$(awk '/usage_usec/ {print $2}' "$service_dir/cpu.stat" 2>/dev/null || echo 0)
    cpu_sec=$((cpu_usage / 1000000))
    
    if [ "$mem_mb" -gt 0 ] || [ "$pids" -gt 0 ]; then
        printf "%-40s  MEM: %6dMB  PIDs: %4s  CPU: %ds\n" \
            "$service" "$mem_mb" "$pids" "$cpu_sec"
    fi
done

# 3. Namespace inspector
#!/bin/bash
# Inspect namespaces of running containers/processes
echo "=== Namespace Inspector ==="

for pid in /proc/[0-9]*; do
    pid_num=$(basename "$pid")
    [ -d "$pid/ns" ] || continue
    
    # Get process name
    comm=$(cat "$pid/comm" 2>/dev/null || continue)
    
    # Check if in non-default namespaces
    default_pid_ns=$(readlink /proc/1/ns/pid 2>/dev/null)
    proc_pid_ns=$(readlink "$pid/ns/pid" 2>/dev/null)
    
    if [ "$proc_pid_ns" != "$default_pid_ns" ] 2>/dev/null; then
        echo ""
        echo "PID $pid_num ($comm) - containerized:"
        echo "  PID NS:  $proc_pid_ns"
        echo "  NET NS:  $(readlink $pid/ns/net 2>/dev/null)"
        echo "  MNT NS:  $(readlink $pid/ns/mnt 2>/dev/null)"
        echo "  UTS NS:  $(readlink $pid/ns/uts 2>/dev/null)"
        
        # Check cgroup
        cgroup=$(cat "$pid/cgroup" 2>/dev/null | head -1)
        echo "  CGROUP:  $cgroup"
    fi
done`,
				},
				{
					Title: "Container Runtimes and Rootless Containers",
					Content: `Container runtimes translate high-level container operations into Linux kernel operations. Understanding the runtime landscape helps make informed decisions.

**Container Runtime Architecture:**
` + "```" + `
Container runtime layers:

High-level (manage images, networking, storage):
  Docker Engine (dockerd)
  Podman (daemonless)
  containerd
  CRI-O (Kubernetes-focused)

Low-level (actually create containers):
  runc (OCI reference implementation)
  crun (C implementation, lighter)
  kata-containers (VM-based isolation)
  gVisor (runsc, application kernel)

OCI (Open Container Initiative) Standards:
  Image Spec:    How container images are structured
  Runtime Spec:  How containers are created/run
  Distribution:  How images are distributed (registries)

Docker architecture:
  docker CLI → dockerd (daemon) → containerd → runc → container
  
Podman architecture:
  podman CLI → conmon → runc → container  (no daemon!)

CRI-O (Kubernetes):
  kubelet → CRI → CRI-O → runc → container
` + "```" + `

**Podman (Daemonless Containers):**
` + "```" + `
Podman is a Docker-compatible container engine that runs without a daemon.

Installation:
  # Debian/Ubuntu
  apt install podman
  
  # RHEL/Fedora
  dnf install podman

Commands (Docker-compatible):
  podman pull docker.io/library/nginx:latest
  podman images
  podman run -d -p 8080:80 --name web nginx
  podman ps
  podman logs web
  podman exec -it web /bin/bash
  podman stop web
  podman rm web
  podman build -t myapp .
  podman push myapp docker.io/user/myapp

Key differences from Docker:
  1. Daemonless: No background service
     - Each container is a child process
     - Uses fork/exec model
     - Containers survive reboot via systemd
  
  2. Rootless by default
     - Run containers without root
     - Uses user namespaces
     - Improved security
  
  3. Pod support  
     - Group containers (like Kubernetes pods)
     - podman pod create --name mypod
     - podman run --pod mypod container1
     - podman run --pod mypod container2
     - podman generate kube mypod > pod.yaml
  
  4. Systemd integration
     # Generate systemd service for container
     podman generate systemd --new --name web > \
       ~/.config/systemd/user/container-web.service
     systemctl --user enable --now container-web.service

Podman compose:
  # Install
  pip3 install podman-compose
  
  # Use existing docker-compose.yml
  podman-compose up -d
  podman-compose down
  
  # Or use podman with Docker compose
  export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
  podman system service --time=0 &
  docker-compose up -d
` + "```" + `

**Rootless Containers:**
` + "```" + `
Running containers without root using user namespaces.

Prerequisites:
  # Check for user namespace support
  sysctl kernel.unprivileged_userns_clone
  # Should be 1 (or not present = enabled)
  
  # Ensure subuid/subgid are configured
  cat /etc/subuid
  # user:100000:65536   (user mapped to UIDs 100000-165535)
  
  cat /etc/subgid
  # user:100000:65536
  
  # Add if missing
  usermod --add-subuids 100000-165535 --add-subgids 100000-165535 username

How rootless works:
  Host                    Container
  UID 1000 (user)    →    UID 0 (root)
  UID 100000         →    UID 1
  UID 100001         →    UID 2
  ...                     ...
  
  - Container root (UID 0) maps to unprivileged user on host
  - Cannot access host files owned by real root
  - Cannot bind to privileged ports (<1024) by default

Rootless Docker:
  # Install rootless Docker
  curl -fsSL https://get.docker.com/rootless | sh
  
  # Set environment
  export PATH=$HOME/bin:$PATH
  export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
  
  # Systemd user service
  systemctl --user start docker
  systemctl --user enable docker
  loginctl enable-linger $(whoami)

Rootless Podman (default):
  # Just run as regular user
  podman run -d nginx
  
  # Check rootless
  podman info | grep rootless
  # rootless: true

Rootless limitations and workarounds:
  # Port < 1024
  # Option 1: Use high port
  podman run -p 8080:80 nginx
  
  # Option 2: Allow low ports
  sysctl net.ipv4.ip_unprivileged_port_start=80
  
  # Networking: rootless uses slirp4netns or pasta
  # Limited: no bridge networking by default
  podman run --network slirp4netns:port_handler=slirp4netns nginx
  podman run --network pasta nginx  # Newer, better performance
  
  # Storage: uses fuse-overlayfs
  podman info | grep graphDriverName
  # overlay (with fuse-overlayfs for rootless)
  
  # Volumes: UID mapping
  podman run -v ./data:/data:Z nginx
  # :Z applies SELinux label
  # Files owned by mapped UID on host
` + "```" + `

**Container Security:**
` + "```" + `
Security layers for containers:

1. Capabilities (drop unnecessary):
   podman run --cap-drop=ALL --cap-add=NET_BIND_SERVICE nginx
   
   Key capabilities:
     CAP_NET_BIND_SERVICE   Bind to ports < 1024
     CAP_SYS_ADMIN          Many admin operations
     CAP_NET_RAW            Raw sockets (ping)
     CAP_DAC_OVERRIDE       Bypass file permissions
     CAP_SETUID             Change UID
   
   Best practice: drop ALL, add only needed

2. Seccomp (syscall filtering):
   # Default profile blocks ~44 dangerous syscalls
   podman run --security-opt seccomp=profile.json nginx
   
   # Disable (not recommended):
   podman run --security-opt seccomp=unconfined nginx

3. SELinux/AppArmor:
   # Container SELinux labels
   podman run --security-opt label=type:container_t nginx
   
   # Disable (not recommended):
   podman run --security-opt label=disable nginx

4. Read-only filesystem:
   podman run --read-only --tmpfs /tmp nginx

5. No new privileges:
   podman run --security-opt no-new-privileges:true nginx

6. Resource limits:
   podman run --memory 512m --cpus 2 --pids-limit 100 nginx

7. Image scanning:
   # Trivy
   trivy image nginx:latest
   
   # Podman
   podman image scan nginx:latest

Comprehensive secure run:
   podman run -d \
     --name secure-nginx \
     --read-only \
     --tmpfs /tmp:rw,noexec,nosuid \
     --tmpfs /var/cache/nginx:rw \
     --tmpfs /var/run:rw \
     --cap-drop ALL \
     --cap-add NET_BIND_SERVICE \
     --security-opt no-new-privileges:true \
     --security-opt seccomp=default \
     --memory 256m \
     --cpus 1 \
     --pids-limit 50 \
     --user 101:101 \
     -p 8080:80 \
     nginx:latest
` + "```" + ``,
					CodeExamples: `# Container runtime operations

# 1. Container inspection tool
#!/bin/bash
# Inspect running containers and their isolation
echo "=== Container Security Audit ==="

for pid in $(podman ps -q 2>/dev/null); do
    name=$(podman inspect "$pid" --format '{{.Name}}' 2>/dev/null)
    image=$(podman inspect "$pid" --format '{{.ImageName}}' 2>/dev/null)
    
    echo ""
    echo "--- Container: $name ($image) ---"
    
    # Check capabilities
    caps=$(podman inspect "$pid" --format '{{.HostConfig.CapAdd}}')
    dropped=$(podman inspect "$pid" --format '{{.HostConfig.CapDrop}}')
    echo "  Added caps:   $caps"
    echo "  Dropped caps: $dropped"
    
    # Check read-only
    readonly=$(podman inspect "$pid" --format '{{.HostConfig.ReadonlyRootfs}}')
    echo "  Read-only FS: $readonly"
    
    # Check resource limits
    mem_limit=$(podman inspect "$pid" --format '{{.HostConfig.Memory}}')
    cpu_quota=$(podman inspect "$pid" --format '{{.HostConfig.CpuQuota}}')
    pids_limit=$(podman inspect "$pid" --format '{{.HostConfig.PidsLimit}}')
    echo "  Memory limit: $mem_limit"
    echo "  CPU quota:    $cpu_quota"
    echo "  PIDs limit:   $pids_limit"
    
    # Check user
    user=$(podman inspect "$pid" --format '{{.Config.User}}')
    echo "  User:         ${user:-root (!)}"
    
    # Check privileged
    priv=$(podman inspect "$pid" --format '{{.HostConfig.Privileged}}')
    echo "  Privileged:   $priv"
    
    # Warnings
    if [ "$priv" = "true" ]; then
        echo "  WARNING: Running privileged!"
    fi
    if [ -z "$user" ] || [ "$user" = "0" ] || [ "$user" = "root" ]; then
        echo "  WARNING: Running as root!"
    fi
    if [ "$readonly" != "true" ]; then
        echo "  WARNING: Filesystem is writable"
    fi
    if [ "$mem_limit" = "0" ]; then
        echo "  WARNING: No memory limit set"
    fi
done

# 2. Rootless container setup script
#!/bin/bash
# Setup rootless container environment for a user
USERNAME="${1:?Usage: $0 <username>}"

echo "Setting up rootless containers for $USERNAME"

# Check subuid/subgid
if ! grep -q "^$USERNAME:" /etc/subuid; then
    # Find next available range
    LAST_UID=$(awk -F: '{print $2 + $3}' /etc/subuid | sort -n | tail -1)
    NEXT_UID=${LAST_UID:-100000}
    
    echo "$USERNAME:$NEXT_UID:65536" >> /etc/subuid
    echo "$USERNAME:$NEXT_UID:65536" >> /etc/subgid
    echo "  Added subuid/subgid range: $NEXT_UID-$((NEXT_UID+65535))"
fi

# Enable linger (keep user services after logout)
loginctl enable-linger "$USERNAME"

# Create XDG_RUNTIME_DIR if needed
RUNTIME_DIR="/run/user/$(id -u "$USERNAME")"
if [ ! -d "$RUNTIME_DIR" ]; then
    mkdir -p "$RUNTIME_DIR"
    chown "$USERNAME:$USERNAME" "$RUNTIME_DIR"
    chmod 700 "$RUNTIME_DIR"
fi

# Install required packages
echo "Installing container tools..."
if command -v apt-get > /dev/null 2>&1; then
    apt-get install -y podman slirp4netns fuse-overlayfs uidmap
elif command -v dnf > /dev/null 2>&1; then
    dnf install -y podman slirp4netns fuse-overlayfs shadow-utils
fi

echo "Setup complete. User $USERNAME can now run rootless containers."
echo "  Run: su - $USERNAME -c 'podman run hello-world'"

# 3. Container-to-systemd converter
#!/bin/bash
# Convert running containers to systemd services
EXPORT_DIR="$HOME/.config/systemd/user"
mkdir -p "$EXPORT_DIR"

for cid in $(podman ps -q 2>/dev/null); do
    name=$(podman inspect "$cid" --format '{{.Name}}')
    echo "Generating systemd service for: $name"
    
    podman generate systemd \
        --new \
        --name "$name" \
        --restart-policy=on-failure \
        --restart-sec=10 \
        > "$EXPORT_DIR/container-$name.service"
    
    echo "  Created: $EXPORT_DIR/container-$name.service"
done

echo ""
echo "Reload and enable:"
echo "  systemctl --user daemon-reload"
for cid in $(podman ps -q 2>/dev/null); do
    name=$(podman inspect "$cid" --format '{{.Name}}')
    echo "  systemctl --user enable container-$name.service"
done`,
				},
			},
		},
	})
}
