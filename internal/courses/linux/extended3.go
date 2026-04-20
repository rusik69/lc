package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1825,
			Title:       "Linux Security Hardening",
			Description: "Comprehensive Linux security: user management, PAM, SELinux/AppArmor, SSH hardening, audit framework, and security best practices.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "User Security and PAM",
					Content: `Linux security begins with proper user management and authentication. PAM (Pluggable Authentication Modules) provides a flexible framework for authentication.

**User and Group Management:**
` + "```" + `
User files:
  /etc/passwd   User accounts (name:x:uid:gid:info:home:shell)
  /etc/shadow   Password hashes (name:hash:lastchange:min:max:warn:...)
  /etc/group    Group definitions (name:x:gid:members)
  /etc/gshadow  Group passwords (rarely used)

Security practices:
  # Lock user account
  usermod -L username
  passwd -l username
  
  # Disable login shell
  usermod -s /usr/sbin/nologin username
  chsh -s /bin/false username
  
  # Set password expiration
  chage -M 90 username          # Max 90 days
  chage -m 7 username           # Min 7 days between changes
  chage -W 14 username          # Warn 14 days before expiry
  chage -E 2025-12-31 username  # Account expiration date
  chage -l username             # View password aging info
  
  # Password policies (/etc/login.defs)
  PASS_MAX_DAYS   90
  PASS_MIN_DAYS   7
  PASS_MIN_LEN    12
  PASS_WARN_AGE   14
  
  # Find users with no password
  awk -F: '$2 == "" {print $1}' /etc/shadow
  
  # Find UID 0 users (besides root)
  awk -F: '$3 == 0 && $1 != "root" {print $1}' /etc/passwd
  
  # Find users with login shells
  grep -v 'nologin\|false' /etc/passwd | cut -d: -f1

Sudo configuration:
  visudo                         # Safe editor for sudoers
  
  /etc/sudoers:
    # Full access
    admin ALL=(ALL:ALL) ALL
    
    # Specific commands without password
    deploy ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart myapp
    
    # Group-based
    %wheel ALL=(ALL:ALL) ALL
    %devops ALL=(ALL) NOPASSWD: /usr/bin/docker
    
    # Logging
    Defaults  logfile="/var/log/sudo.log"
    Defaults  log_input, log_output
    Defaults  !rootpw
    Defaults  timestamp_timeout=15
    
  # Use /etc/sudoers.d/ for modular config
  echo "deploy ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart myapp" > /etc/sudoers.d/deploy
  chmod 440 /etc/sudoers.d/deploy
` + "```" + `

**PAM (Pluggable Authentication Modules):**
` + "```" + `
PAM configuration: /etc/pam.d/

Module types:
  auth       Authentication (verify identity)
  account    Account management (access control, expiration)
  password   Password management (change password)
  session    Session management (logging, resource limits)

Control flags:
  required    Must succeed, continue checking
  requisite   Must succeed, fail immediately if not
  sufficient  If succeeds, skip remaining (unless required failed)
  optional    Result only matters if it's the only module
  include     Include another PAM config file

Common PAM modules:
  pam_unix.so       Traditional password authentication
  pam_pwquality.so  Password quality checking
  pam_faillock.so   Account lockout after failures
  pam_limits.so     Resource limits (/etc/security/limits.conf)
  pam_access.so     Access control (/etc/security/access.conf)
  pam_time.so       Time-based access control
  pam_google_authenticator.so  TOTP 2FA
  pam_sss.so        SSSD (LDAP/AD integration)

Example: /etc/pam.d/common-auth
  auth  required  pam_faillock.so preauth silent deny=5 unlock_time=900
  auth  required  pam_unix.so
  auth  required  pam_faillock.so authfail deny=5 unlock_time=900
  auth  optional  pam_permit.so

Password quality (/etc/security/pwquality.conf):
  minlen = 12
  dcredit = -1       # At least 1 digit
  ucredit = -1       # At least 1 uppercase
  lcredit = -1       # At least 1 lowercase
  ocredit = -1       # At least 1 special char
  maxrepeat = 3      # Max 3 consecutive same chars
  difok = 5          # Min 5 chars different from old password
  enforce_for_root   # Apply to root too
  dictcheck = 1      # Check against dictionary

Account lockout:
  # /etc/pam.d/system-auth
  auth required pam_faillock.so preauth silent deny=5 unlock_time=900
  auth required pam_unix.so
  auth [default=die] pam_faillock.so authfail deny=5 unlock_time=900
  
  # Check locked accounts
  faillock --user username
  
  # Unlock account
  faillock --user username --reset
` + "```" + `

**Resource Limits:**
` + "```" + `
/etc/security/limits.conf:
  # <domain> <type> <item> <value>
  *          soft   nofile    65536
  *          hard   nofile    131072
  *          soft   nproc     4096
  *          hard   nproc     8192
  @devs      hard   maxlogins 2
  www-data   soft   nofile    65536
  root       soft   nofile    unlimited

Items:
  nofile    Max open file descriptors
  nproc     Max number of processes
  maxlogins Max simultaneous logins
  memlock   Max locked memory (KB)
  as        Max address space (KB)
  cpu       Max CPU time (minutes)
  fsize     Max file size (KB)
  stack     Max stack size (KB)
  core      Max core file size (KB)

Check limits:
  ulimit -a                      # Current limits
  ulimit -n                      # Open files limit
  cat /proc/<pid>/limits         # Limits for a process
` + "```" + ``,
					CodeExamples: `# Security hardening scripts

# 1. User audit script
#!/bin/bash
echo "=== Security User Audit ==="

echo "--- Users with UID 0 (root-equivalent) ---"
awk -F: '$3 == 0' /etc/passwd

echo ""
echo "--- Users with empty passwords ---"
awk -F: '$2 == "" || $2 == "!" || $2 == "*"' /etc/shadow 2>/dev/null

echo ""
echo "--- Users with login shells ---"
grep -v -E 'nologin|false|sync|shutdown|halt' /etc/passwd | \
  awk -F: '{print $1, $7}'

echo ""
echo "--- Users with no password expiry ---"
awk -F: '$5 == "" || $5 == "99999"' /etc/shadow 2>/dev/null | \
  cut -d: -f1

echo ""
echo "--- World-writable directories ---"
find / -type d -perm -002 -not -path "/proc/*" -not -path "/sys/*" \
  2>/dev/null | head -20

echo ""
echo "--- SUID files ---"
find / -type f -perm -4000 -not -path "/proc/*" 2>/dev/null

echo ""
echo "--- SGID files ---"
find / -type f -perm -2000 -not -path "/proc/*" 2>/dev/null

echo ""
echo "--- Files with no owner ---"
find / -nouser -o -nogroup 2>/dev/null | head -20

# 2. PAM configuration for password quality
# /etc/pam.d/common-password
password  requisite   pam_pwquality.so retry=3 minlen=12 \
  dcredit=-1 ucredit=-1 lcredit=-1 ocredit=-1 \
  maxrepeat=3 difok=5 reject_username enforce_for_root
password  [success=1 default=ignore] pam_unix.so obscure use_authtok \
  try_first_pass sha512 remember=12 rounds=65536
password  requisite   pam_deny.so
password  required    pam_permit.so

# 3. Access control configuration
# /etc/security/access.conf
+ : root : LOCAL
+ : @admins : ALL
+ : @devops : 192.168.1.0/24
- : ALL : ALL

# 4. Systemd service hardening
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=myapp
Group=myapp
ExecStart=/usr/local/bin/myapp
Restart=on-failure
RestartSec=5

# Security hardening
NoNewPrivileges=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/myapp
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
RestrictNamespaces=yes
RestrictSUIDSGID=yes
MemoryDenyWriteExecute=yes
LockPersonality=yes
SystemCallFilter=@system-service
SystemCallArchitectures=native
CapabilityBoundingSet=
AmbientCapabilities=

[Install]
WantedBy=multi-user.target`,
				},
				{
					Title: "SELinux and AppArmor",
					Content: `Mandatory Access Control (MAC) systems like SELinux and AppArmor provide additional security beyond traditional discretionary access control (file permissions).

**SELinux (Security-Enhanced Linux):**
` + "```" + `
Modes:
  Enforcing   Policies are enforced (denies and logs)
  Permissive  Policies are not enforced (logs only)
  Disabled    SELinux is off

Check/set mode:
  getenforce                     # Current mode
  sestatus                       # Detailed status
  setenforce 0                   # Set permissive (temporary)
  setenforce 1                   # Set enforcing (temporary)
  
  # Permanent: /etc/selinux/config
  SELINUX=enforcing
  SELINUXTYPE=targeted

Security contexts:
  Format: user:role:type:level
  Example: system_u:system_r:httpd_t:s0
  
  # View file contexts
  ls -Z /var/www/html/
  # -rw-r--r--. root root system_u:object_r:httpd_sys_content_t:s0 index.html
  
  # View process contexts
  ps -auxZ | grep httpd
  # system_u:system_r:httpd_t:s0  root  httpd

Types (most important part):
  httpd_t              Apache/Nginx process type
  httpd_sys_content_t  Web content (read-only)
  httpd_sys_rw_content_t  Writable web content
  sshd_t               SSH daemon type
  user_home_t          User home directory

Managing file contexts:
  # Restore default context
  restorecon -Rv /var/www/html/
  
  # Set context on file
  chcon -t httpd_sys_content_t /var/www/html/newfile
  
  # Permanent context rule
  semanage fcontext -a -t httpd_sys_content_t "/srv/web(/.*)?"
  restorecon -Rv /srv/web/

Booleans (feature toggles):
  # List all booleans
  getsebool -a
  getsebool -a | grep httpd
  
  # Set boolean
  setsebool -P httpd_can_network_connect on
  setsebool -P httpd_can_sendmail on
  setsebool -P httpd_use_nfs on
  
  # Common booleans:
  httpd_can_network_connect    # Apache can make network connections
  httpd_can_sendmail           # Apache can send email
  httpd_enable_homedirs        # Apache can serve user home dirs
  samba_enable_home_dirs       # Samba can access home dirs
  ftpd_full_access             # FTP full filesystem access

Ports:
  semanage port -l | grep http   # List port labels
  semanage port -a -t http_port_t -p tcp 8080  # Add custom port
  semanage port -d -t http_port_t -p tcp 8080  # Remove custom port

Troubleshooting:
  # View denials
  ausearch -m avc -ts recent
  audit2why < /var/log/audit/audit.log
  
  # Generate policy from denials
  audit2allow -a -M mypolicy
  semodule -i mypolicy.pp
  
  # sealert (setroubleshoot)
  sealert -a /var/log/audit/audit.log
` + "```" + `

**AppArmor (Alternative to SELinux):**
` + "```" + `
Used by: Ubuntu, SUSE, Debian
Concept: Path-based access control (easier than SELinux)

Modes:
  Enforce    Policy is enforced
  Complain   Log violations but don't enforce

Commands:
  aa-status                      # Show all profiles and their mode
  aa-enforce /etc/apparmor.d/usr.sbin.nginx   # Set enforce
  aa-complain /etc/apparmor.d/usr.sbin.nginx  # Set complain
  aa-disable /etc/apparmor.d/usr.sbin.nginx   # Disable profile
  
  # Reload profiles
  apparmor_parser -r /etc/apparmor.d/usr.sbin.nginx

Profile structure:
  /etc/apparmor.d/usr.sbin.nginx:
    #include <tunables/global>
    
    /usr/sbin/nginx {
      #include <abstractions/base>
      #include <abstractions/nameservice>
      
      # Capabilities
      capability net_bind_service,
      capability setuid,
      capability setgid,
      
      # Network access
      network inet stream,
      network inet6 stream,
      
      # File access
      /usr/sbin/nginx mr,
      /etc/nginx/** r,
      /var/log/nginx/** w,
      /var/www/** r,
      /run/nginx.pid rw,
      /var/lib/nginx/** rw,
      
      # Deny access
      deny /etc/shadow r,
      deny /root/** rwx,
    }

Permission flags:
  r    Read
  w    Write
  a    Append
  x    Execute
  m    Memory map executable
  k    File locking
  l    Create links
  ix   Inherit execute (child gets parent profile)
  px   Profile execute (child gets its own profile)
  Ux   Unconfined execute (child runs unconfined)
  cx   Child profile execute

Generate profile:
  aa-genprof /usr/sbin/myapp     # Interactive profile generator
  aa-logprof                     # Update profiles from logs
` + "```" + `

**SSH Hardening:**
` + "```" + `
/etc/ssh/sshd_config:
  # Authentication
  PermitRootLogin no              # Disable root login
  PasswordAuthentication no       # Keys only
  PubkeyAuthentication yes
  AuthenticationMethods publickey # Only public key
  MaxAuthTries 3
  LoginGraceTime 30
  
  # Network
  Port 22                        # Consider non-standard port
  ListenAddress 0.0.0.0
  Protocol 2                     # SSH2 only
  
  # Security
  PermitEmptyPasswords no
  X11Forwarding no
  AllowTcpForwarding no
  GatewayPorts no
  PermitTunnel no
  
  # Users/Groups
  AllowUsers admin deploy
  AllowGroups sshusers
  DenyUsers guest
  
  # Timeouts
  ClientAliveInterval 300
  ClientAliveCountMax 2
  
  # Ciphers (strong only)
  Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com
  MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
  KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org
  HostKeyAlgorithms ssh-ed25519,rsa-sha2-512

  # Logging
  LogLevel VERBOSE
  
  # Banner
  Banner /etc/ssh/banner

After changes:
  sshd -t                        # Test config
  systemctl restart sshd
` + "```" + `

**Linux Audit Framework:**
` + "```" + `
auditd: kernel-level syscall auditing

Configuration: /etc/audit/auditd.conf
  log_file = /var/log/audit/audit.log
  max_log_file = 50
  num_logs = 5
  max_log_file_action = ROTATE

Audit rules: /etc/audit/rules.d/

Rule types:
  -w  Watch file/directory for changes
  -a  System call auditing
  -k  Key for searching (tag)

Common rules:
  # Watch authentication files
  -w /etc/passwd -p wa -k identity
  -w /etc/shadow -p wa -k identity
  -w /etc/group -p wa -k identity
  -w /etc/sudoers -p wa -k sudoers
  -w /etc/sudoers.d/ -p wa -k sudoers
  
  # Watch SSH config
  -w /etc/ssh/sshd_config -p wa -k sshd_config
  
  # Watch cron
  -w /etc/crontab -p wa -k cron
  -w /var/spool/cron/ -p wa -k cron
  
  # Watch system time changes
  -a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
  -a always,exit -F arch=b64 -S clock_settime -k time-change
  
  # Watch for unauthorized access attempts
  -a always,exit -F arch=b64 -S open -F exit=-EACCES -k access
  -a always,exit -F arch=b64 -S open -F exit=-EPERM -k access
  
  # Watch privilege escalation
  -a always,exit -F arch=b64 -S execve -F euid=0 -F auid>=1000 -k privilege_esc

Search audit logs:
  ausearch -k identity -ts today
  ausearch -m USER_LOGIN -ts recent
  ausearch -ua username
  aureport -au                   # Authentication report
  aureport -l                    # Login report
  aureport --summary             # Summary report
` + "```" + ``,
					CodeExamples: `# Security hardening configuration

# 1. Comprehensive audit rules
# /etc/audit/rules.d/99-security.rules

# Delete all existing rules
-D

# Set buffer size
-b 8192

# Failure mode (1=print, 2=panic)
-f 1

# Identity changes
-w /etc/passwd -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/gshadow -p wa -k identity
-w /etc/security/ -p wa -k security_conf

# Sudo and privilege escalation
-w /etc/sudoers -p wa -k sudoers
-w /etc/sudoers.d/ -p wa -k sudoers
-w /var/log/sudo.log -p wa -k sudo_log

# SSH
-w /etc/ssh/sshd_config -p wa -k sshd
-w /root/.ssh/ -p wa -k ssh_keys

# System startup
-w /etc/systemd/ -p wa -k systemd
-w /usr/lib/systemd/ -p wa -k systemd

# Kernel modules
-w /sbin/insmod -p x -k modules
-w /sbin/rmmod -p x -k modules
-w /sbin/modprobe -p x -k modules
-a always,exit -F arch=b64 -S init_module -S delete_module -k modules

# Time changes
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time
-a always,exit -F arch=b64 -S clock_settime -k time
-w /etc/localtime -p wa -k time

# Network configuration
-w /etc/hosts -p wa -k network
-w /etc/sysconfig/network -p wa -k network
-w /etc/sysconfig/network-scripts/ -p wa -k network

# Login/logout
-w /var/log/lastlog -p wa -k logins
-w /var/run/faillock/ -p wa -k logins

# File deletion
-a always,exit -F arch=b64 -S unlink -S unlinkat -S rename \
  -S renameat -F auid>=1000 -F auid!=4294967295 -k delete

# Make config immutable (must be last rule)
-e 2

# 2. SSH key management
#!/bin/bash
# Generate strong SSH key
ssh-keygen -t ed25519 -C "user@host" -f ~/.ssh/id_ed25519

# Deploy key to remote host
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@remote

# SSH config for convenience and security
# ~/.ssh/config
Host production
    HostName 10.0.1.100
    User deploy
    IdentityFile ~/.ssh/id_ed25519_prod
    Port 22
    ForwardAgent no
    StrictHostKeyChecking yes
    UserKnownHostsFile ~/.ssh/known_hosts

Host bastion
    HostName bastion.example.com
    User admin
    IdentityFile ~/.ssh/id_ed25519
    DynamicForward 1080

Host internal-*
    ProxyJump bastion
    User deploy
    IdentityFile ~/.ssh/id_ed25519_internal

# Fix SSH permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/authorized_keys
chmod 644 ~/.ssh/known_hosts
chmod 600 ~/.ssh/config

# 3. SELinux custom policy for web application
#!/bin/bash
# Create policy for custom app serving on port 8080
semanage port -a -t http_port_t -p tcp 8080
semanage fcontext -a -t httpd_sys_content_t "/opt/myapp/static(/.*)?"
semanage fcontext -a -t httpd_sys_rw_content_t "/opt/myapp/data(/.*)?"
semanage fcontext -a -t httpd_log_t "/opt/myapp/logs(/.*)?"
restorecon -Rv /opt/myapp/
setsebool -P httpd_can_network_connect 1`,
				},
			},
		},
	})
}
