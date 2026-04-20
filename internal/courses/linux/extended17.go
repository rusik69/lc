package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1839,
			Title:       "Linux Email, Web, and Application Servers",
			Description: "Configure and manage production web servers (Nginx, Apache), mail servers (Postfix), reverse proxies, load balancers, and application deployment on Linux.",
			Order:       39,
			Lessons: []problems.Lesson{
				{
					Title: "Web Server Administration",
					Content: `Web servers are the backbone of internet-facing Linux infrastructure. Nginx and Apache are the dominant choices, each with different architectures and strengths.

**Nginx Configuration:**
` + "```" + `
Architecture:
  - Event-driven, asynchronous
  - Master process + worker processes
  - Single-threaded workers handle thousands of connections
  - Low memory footprint
  - Excellent for static content and reverse proxying

Installation:
  apt install nginx              # Debian/Ubuntu
  dnf install nginx              # RHEL/Fedora

Main config structure:
  /etc/nginx/
  ├── nginx.conf                 # Main configuration
  ├── conf.d/                    # Additional configs (*.conf)
  ├── sites-available/           # Virtual host configs (Debian)
  ├── sites-enabled/             # Enabled vhosts (symlinks)
  ├── modules-available/         # Available modules
  └── snippets/                  # Reusable config snippets

Global settings (/etc/nginx/nginx.conf):
  user www-data;
  worker_processes auto;          # Match CPU cores
  worker_rlimit_nofile 65535;
  pid /run/nginx.pid;
  
  events {
      worker_connections 4096;    # Per worker
      multi_accept on;
      use epoll;
  }
  
  http {
      # Basic settings
      sendfile on;
      tcp_nopush on;
      tcp_nodelay on;
      keepalive_timeout 65;
      types_hash_max_size 2048;
      server_tokens off;          # Hide version
      
      # MIME types
      include /etc/nginx/mime.types;
      default_type application/octet-stream;
      
      # Logging
      access_log /var/log/nginx/access.log;
      error_log /var/log/nginx/error.log;
      
      # Gzip compression
      gzip on;
      gzip_vary on;
      gzip_min_length 1000;
      gzip_proxied any;
      gzip_comp_level 6;
      gzip_types text/plain text/css application/json
                 application/javascript text/xml application/xml
                 application/xml+rss text/javascript;
      
      # Rate limiting
      limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
      limit_conn_zone $binary_remote_addr zone=addr:10m;
      
      # Include virtual hosts
      include /etc/nginx/conf.d/*.conf;
      include /etc/nginx/sites-enabled/*;
  }

Virtual host with SSL:
  server {
      listen 80;
      server_name example.com www.example.com;
      return 301 https://$server_name$request_uri;
  }
  
  server {
      listen 443 ssl http2;
      server_name example.com www.example.com;
      root /var/www/example.com;
      index index.html;
      
      # SSL
      ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
      ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
      ssl_protocols TLSv1.2 TLSv1.3;
      ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
      ssl_prefer_server_ciphers off;
      ssl_session_cache shared:SSL:10m;
      ssl_session_timeout 10m;
      ssl_stapling on;
      ssl_stapling_verify on;
      
      # Security headers
      add_header X-Frame-Options "SAMEORIGIN" always;
      add_header X-Content-Type-Options "nosniff" always;
      add_header X-XSS-Protection "1; mode=block" always;
      add_header Strict-Transport-Security "max-age=31536000" always;
      add_header Content-Security-Policy "default-src 'self'" always;
      
      # Locations
      location / {
          try_files $uri $uri/ =404;
      }
      
      location /api/ {
          limit_req zone=api burst=20 nodelay;
          proxy_pass http://backend;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }
      
      # Static file caching
      location ~* \.(jpg|jpeg|png|gif|ico|css|js|woff2)$ {
          expires 30d;
          add_header Cache-Control "public, immutable";
      }
      
      # Deny hidden files
      location ~ /\. {
          deny all;
      }
  }

Load balancer / reverse proxy:
  upstream backend {
      least_conn;                    # Load balancing method
      server 10.0.0.10:8080 weight=3;
      server 10.0.0.11:8080 weight=2;
      server 10.0.0.12:8080 backup;
      
      keepalive 32;                  # Connection pooling
  }
  
  # Health checks (Nginx Plus or OpenResty)
  # upstream backend {
  #     zone backend 64k;
  #     server 10.0.0.10:8080;
  #     server 10.0.0.11:8080;
  #     health_check interval=10 fails=3 passes=2;
  # }

  # Load balancing methods:
  #   round_robin    Default, equal distribution
  #   least_conn     Fewest active connections
  #   ip_hash        Sticky sessions by client IP
  #   hash $key      Custom hash (e.g., $request_uri)
` + "```" + `

**Apache Configuration:**
` + "```" + `
Architecture:
  - Process/thread-based (MPM modules)
  - prefork: One process per connection (stable, memory heavy)
  - worker:  Threads within processes (hybrid)
  - event:   Like worker but handles keep-alive efficiently

Installation:
  apt install apache2             # Debian/Ubuntu
  dnf install httpd               # RHEL/Fedora

Config structure (Debian):
  /etc/apache2/
  ├── apache2.conf                # Main config
  ├── ports.conf                  # Listen ports
  ├── mods-available/             # Available modules
  ├── mods-enabled/               # Enabled modules (symlinks)
  ├── sites-available/            # Virtual hosts
  ├── sites-enabled/              # Enabled vhosts (symlinks)
  └── conf-available/             # Configuration fragments

Module management:
  a2enmod rewrite                 # Enable mod_rewrite
  a2enmod ssl                     # Enable SSL
  a2enmod proxy proxy_http        # Enable reverse proxy
  a2enmod headers                 # Enable header manipulation
  a2dismod autoindex              # Disable directory listing
  a2ensite mysite.conf            # Enable virtual host
  a2dissite 000-default.conf      # Disable default

Virtual host:
  <VirtualHost *:443>
      ServerName example.com
      ServerAlias www.example.com
      DocumentRoot /var/www/example.com
      
      SSLEngine on
      SSLCertificateFile /etc/letsencrypt/live/example.com/fullchain.pem
      SSLCertificateKeyFile /etc/letsencrypt/live/example.com/privkey.pem
      
      <Directory /var/www/example.com>
          AllowOverride All
          Require all granted
      </Directory>
      
      # Reverse proxy
      ProxyPreserveHost On
      ProxyPass /api http://localhost:8080/
      ProxyPassReverse /api http://localhost:8080/
      
      # Security
      Header always set X-Frame-Options "SAMEORIGIN"
      Header always set X-Content-Type-Options "nosniff"
      
      # Logging
      ErrorLog ${APACHE_LOG_DIR}/example-error.log
      CustomLog ${APACHE_LOG_DIR}/example-access.log combined
  </VirtualHost>
` + "```" + `

**Let's Encrypt / Certbot:**
` + "```" + `
Free TLS certificates with automatic renewal.

Installation:
  apt install certbot python3-certbot-nginx     # For Nginx
  apt install certbot python3-certbot-apache     # For Apache

Obtain certificate:
  # Nginx (automatic config)
  certbot --nginx -d example.com -d www.example.com
  
  # Apache
  certbot --apache -d example.com -d www.example.com
  
  # Standalone (no web server)
  certbot certonly --standalone -d example.com
  
  # Webroot (existing web server)
  certbot certonly --webroot -w /var/www/html -d example.com
  
  # DNS challenge (wildcard)
  certbot certonly --manual --preferred-challenges dns -d '*.example.com'

Renewal:
  # Test renewal
  certbot renew --dry-run
  
  # Automatic renewal (usually via systemd timer or cron)
  systemctl list-timers | grep certbot
  
  # Manual cron
  0 0,12 * * * certbot renew --quiet --post-hook "systemctl reload nginx"

Certificate locations:
  /etc/letsencrypt/live/example.com/
  ├── fullchain.pem     # Certificate + intermediate
  ├── privkey.pem       # Private key
  ├── cert.pem          # Certificate only
  └── chain.pem         # Intermediate certificates
` + "```" + ``,
					CodeExamples: `# Web server management scripts

# 1. Nginx configuration tester and deployer
#!/bin/bash
SITES_AVAILABLE="/etc/nginx/sites-available"
SITES_ENABLED="/etc/nginx/sites-enabled"

case "${1:-help}" in
    enable)
        SITE="${2:?Usage: $0 enable <site>}"
        if [ ! -f "$SITES_AVAILABLE/$SITE" ]; then
            echo "Site config not found: $SITES_AVAILABLE/$SITE"
            exit 1
        fi
        ln -sf "$SITES_AVAILABLE/$SITE" "$SITES_ENABLED/$SITE"
        nginx -t && systemctl reload nginx
        echo "Site enabled: $SITE"
        ;;
    disable)
        SITE="${2:?Usage: $0 disable <site>}"
        rm -f "$SITES_ENABLED/$SITE"
        nginx -t && systemctl reload nginx
        echo "Site disabled: $SITE"
        ;;
    test)
        echo "Testing nginx configuration..."
        nginx -t 2>&1
        ;;
    list)
        echo "Available sites:"
        for site in "$SITES_AVAILABLE"/*; do
            name=$(basename "$site")
            if [ -L "$SITES_ENABLED/$name" ]; then
                echo "  [ENABLED]  $name"
            else
                echo "  [DISABLED] $name"
            fi
        done
        ;;
    status)
        echo "=== Nginx Status ==="
        systemctl status nginx --no-pager | head -10
        echo ""
        echo "--- Active Connections ---"
        if curl -s http://localhost/nginx_status 2>/dev/null; then
            true
        else
            echo "  stub_status not configured"
        fi
        echo ""
        echo "--- Worker Processes ---"
        ps aux | grep "nginx: worker" | grep -v grep | \
            awk '{printf "  PID: %s  CPU: %s%%  MEM: %s%%  RSS: %s KB\n", $2, $3, $4, $6}'
        ;;
    *)
        echo "Usage: $0 {enable|disable|test|list|status} [site]"
        ;;
esac

# 2. SSL certificate monitor
#!/bin/bash
echo "=== SSL Certificate Status ==="

if [ -d /etc/letsencrypt/live ]; then
    for cert_dir in /etc/letsencrypt/live/*/; do
        domain=$(basename "$cert_dir")
        cert="$cert_dir/fullchain.pem"
        
        if [ ! -f "$cert" ]; then
            continue
        fi
        
        # Get expiry date
        expiry=$(openssl x509 -enddate -noout -in "$cert" | cut -d= -f2)
        expiry_epoch=$(date -d "$expiry" +%s 2>/dev/null || date -jf "%b %d %T %Y %Z" "$expiry" +%s 2>/dev/null)
        now_epoch=$(date +%s)
        days_left=$(( (expiry_epoch - now_epoch) / 86400 ))
        
        # Status
        if [ "$days_left" -lt 7 ]; then
            status="CRITICAL"
        elif [ "$days_left" -lt 30 ]; then
            status="WARNING"
        else
            status="OK"
        fi
        
        printf "  %-30s  Expires: %s  Days: %3d  [%s]\n" \
            "$domain" "$(date -d "$expiry" +%Y-%m-%d 2>/dev/null || echo "$expiry")" \
            "$days_left" "$status"
    done
fi

# Also check any custom certificates
for cert in /etc/ssl/certs/local-*.pem /etc/nginx/ssl/*.pem; do
    [ -f "$cert" ] || continue
    
    expiry=$(openssl x509 -enddate -noout -in "$cert" 2>/dev/null | cut -d= -f2)
    [ -z "$expiry" ] && continue
    
    cn=$(openssl x509 -subject -noout -in "$cert" 2>/dev/null | sed 's/.*CN = //')
    expiry_epoch=$(date -d "$expiry" +%s 2>/dev/null || echo 0)
    now_epoch=$(date +%s)
    days_left=$(( (expiry_epoch - now_epoch) / 86400 ))
    
    printf "  %-30s  Expires: %-12s  Days: %3d\n" "$cn" "$expiry" "$days_left"
done

# 3. Access log analyzer
#!/bin/bash
# Quick nginx/apache access log analysis
LOG="${1:-/var/log/nginx/access.log}"

if [ ! -f "$LOG" ]; then
    echo "Log file not found: $LOG"
    exit 1
fi

echo "=== Access Log Analysis: $(basename "$LOG") ==="
TOTAL=$(wc -l < "$LOG")
echo "Total requests: $TOTAL"

echo ""
echo "--- Top 10 IPs ---"
awk '{print $1}' "$LOG" | sort | uniq -c | sort -rn | head -10 | \
    while read -r count ip; do
        printf "  %6d  %s\n" "$count" "$ip"
    done

echo ""
echo "--- HTTP Status Codes ---"
awk '{print $9}' "$LOG" | sort | uniq -c | sort -rn | head -10 | \
    while read -r count code; do
        case "$code" in
            2*) color="OK" ;;
            3*) color="REDIRECT" ;;
            4*) color="CLIENT_ERR" ;;
            5*) color="SERVER_ERR" ;;
            *)  color="OTHER" ;;
        esac
        printf "  %6d  %s (%s)\n" "$count" "$code" "$color"
    done

echo ""
echo "--- Top 10 Requested URLs ---"
awk '{print $7}' "$LOG" | sort | uniq -c | sort -rn | head -10 | \
    while read -r count url; do
        printf "  %6d  %s\n" "$count" "$url"
    done

echo ""
echo "--- Top 10 User Agents ---"
awk -F'"' '{print $6}' "$LOG" | sort | uniq -c | sort -rn | head -10 | \
    while read -r count ua; do
        printf "  %6d  %.60s\n" "$count" "$ua"
    done

echo ""
echo "--- Requests per Hour (last 24h) ---"
awk '{print $4}' "$LOG" | cut -d: -f2 | sort | uniq -c | sort -k2 -n | \
    while read -r count hour; do
        bar=$(printf '%*s' "$((count / 10))" '' | tr ' ' '#')
        printf "  %s:00  %5d  %s\n" "$hour" "$count" "$bar"
    done

# 4. Web server hardening check
#!/bin/bash
echo "=== Web Server Security Check ==="

# Check if server header leaks version
for url in "http://localhost" "https://localhost"; do
    headers=$(curl -sI "$url" 2>/dev/null)
    if [ -n "$headers" ]; then
        server=$(echo "$headers" | grep -i "^Server:" | head -1)
        if echo "$server" | grep -qiE "nginx/|apache/|[0-9]+\.[0-9]+"; then
            echo "  WARNING: Server version exposed: $server"
        else
            echo "  OK: Server version hidden"
        fi
        
        # Check security headers
        for header in "X-Frame-Options" "X-Content-Type-Options" \
                      "Strict-Transport-Security" "Content-Security-Policy"; do
            if echo "$headers" | grep -qi "^$header:"; then
                echo "  OK: $header present"
            else
                echo "  MISSING: $header"
            fi
        done
    fi
done

# Check SSL configuration
if command -v openssl > /dev/null 2>&1; then
    echo ""
    echo "--- SSL Check ---"
    
    # Test for weak protocols
    for proto in ssl3 tls1 tls1_1; do
        if echo | openssl s_client -connect localhost:443 -"$proto" 2>/dev/null | \
            grep -q "CONNECTED"; then
            echo "  WARNING: $proto is enabled (should be disabled)"
        else
            echo "  OK: $proto is disabled"
        fi
    done
fi`,
				},
				{
					Title: "Mail Server and Application Deployment",
					Content: `Setting up a mail server requires careful configuration for deliverability, security, and spam prevention. Application deployment on Linux uses systemd, reverse proxies, and process managers.

**Postfix Mail Server:**
` + "```" + `
Postfix is a fast, secure, and widely used MTA (Mail Transfer Agent).

Installation:
  apt install postfix              # Debian/Ubuntu
  dnf install postfix              # RHEL/Fedora

Main config (/etc/postfix/main.cf):
  # Basic settings
  myhostname = mail.example.com
  mydomain = example.com
  myorigin = $mydomain
  inet_interfaces = all
  inet_protocols = ipv4
  mydestination = $myhostname, localhost.$mydomain, localhost, $mydomain
  
  # Network
  mynetworks = 127.0.0.0/8 [::ffff:127.0.0.0]/104 [::1]/128 10.0.0.0/24
  
  # Mailbox
  home_mailbox = Maildir/
  # Or: mailbox_command = /usr/lib/dovecot/deliver
  
  # Size limits
  message_size_limit = 52428800    # 50MB
  mailbox_size_limit = 1073741824  # 1GB
  
  # TLS (outbound)
  smtp_tls_security_level = may
  smtp_tls_loglevel = 1
  
  # TLS (inbound)
  smtpd_tls_security_level = may
  smtpd_tls_cert_file = /etc/letsencrypt/live/mail.example.com/fullchain.pem
  smtpd_tls_key_file = /etc/letsencrypt/live/mail.example.com/privkey.pem
  smtpd_tls_protocols = !SSLv2, !SSLv3, !TLSv1, !TLSv1.1
  
  # SMTP authentication (via Dovecot SASL)
  smtpd_sasl_type = dovecot
  smtpd_sasl_path = private/auth
  smtpd_sasl_auth_enable = yes
  
  # Restrictions (anti-spam)
  smtpd_helo_required = yes
  smtpd_helo_restrictions =
      permit_mynetworks,
      reject_non_fqdn_helo_hostname,
      reject_invalid_helo_hostname
  
  smtpd_sender_restrictions =
      permit_mynetworks,
      reject_non_fqdn_sender,
      reject_unknown_sender_domain
  
  smtpd_recipient_restrictions =
      permit_mynetworks,
      permit_sasl_authenticated,
      reject_unauth_destination,
      reject_rbl_client zen.spamhaus.org,
      reject_rbl_client bl.spamcop.net

DNS records needed:
  ; MX record
  example.com.     IN  MX   10 mail.example.com.
  
  ; A record for mail server
  mail.example.com. IN  A    203.0.113.10
  
  ; SPF (Sender Policy Framework)
  example.com.     IN  TXT  "v=spf1 mx ip4:203.0.113.10 -all"
  
  ; DKIM (DomainKeys Identified Mail)
  ; Generated by opendkim
  default._domainkey.example.com. IN TXT "v=DKIM1; k=rsa; p=MIIBIj..."
  
  ; DMARC
  _dmarc.example.com. IN TXT "v=DMARC1; p=reject; rua=mailto:dmarc@example.com"
  
  ; Reverse DNS (PTR)
  10.113.0.203.in-addr.arpa. IN PTR mail.example.com.
` + "```" + `

**Application Deployment with Systemd:**
` + "```" + `
Deploy applications as systemd services for process management.

Web application service:
  # /etc/systemd/system/myapp.service
  [Unit]
  Description=My Web Application
  After=network.target postgresql.service
  Wants=postgresql.service
  
  [Service]
  Type=simple
  User=myapp
  Group=myapp
  WorkingDirectory=/opt/myapp
  ExecStart=/opt/myapp/bin/server
  ExecReload=/bin/kill -HUP $MAINPID
  Restart=on-failure
  RestartSec=5
  
  # Environment
  EnvironmentFile=/etc/myapp/env
  Environment=GO_ENV=production
  
  # Security hardening
  NoNewPrivileges=true
  PrivateTmp=true
  ProtectSystem=strict
  ProtectHome=true
  ReadWritePaths=/var/lib/myapp /var/log/myapp
  
  # Resource limits
  MemoryMax=512M
  CPUQuota=200%
  TasksMax=256
  LimitNOFILE=65535
  
  # Logging
  StandardOutput=journal
  StandardError=journal
  SyslogIdentifier=myapp
  
  [Install]
  WantedBy=multi-user.target

Zero-downtime deployment:
  # Blue-green with systemd
  # 1. Deploy new version
  cp -r /opt/myapp /opt/myapp-new
  # Update binaries in /opt/myapp-new
  
  # 2. Test new version
  /opt/myapp-new/bin/server --check-config
  
  # 3. Switch
  systemctl stop myapp
  mv /opt/myapp /opt/myapp-old
  mv /opt/myapp-new /opt/myapp
  systemctl start myapp
  
  # 4. Verify
  curl -f http://localhost:8080/health
  
  # 5. Cleanup
  rm -rf /opt/myapp-old

Socket activation (start on first connection):
  # /etc/systemd/system/myapp.socket
  [Unit]
  Description=My App Socket
  
  [Socket]
  ListenStream=8080
  
  [Install]
  WantedBy=sockets.target
  
  # Service uses socket
  # /etc/systemd/system/myapp.service
  [Service]
  ExecStart=/opt/myapp/bin/server
  # Application receives fd from systemd
` + "```" + `

**Process Managers:**
` + "```" + `
For applications that need more sophisticated process management.

Supervisor:
  apt install supervisor
  
  # /etc/supervisor/conf.d/myapp.conf
  [program:myapp]
  command=/opt/myapp/bin/server
  directory=/opt/myapp
  user=myapp
  autostart=true
  autorestart=true
  redirect_stderr=true
  stdout_logfile=/var/log/myapp/stdout.log
  stderr_logfile=/var/log/myapp/stderr.log
  environment=GO_ENV=production,PORT=8080
  numprocs=1
  startsecs=10
  startretries=3
  
  # Control
  supervisorctl status
  supervisorctl restart myapp
  supervisorctl tail -f myapp stdout

PM2 (Node.js and others):
  npm install -g pm2
  
  # Start application
  pm2 start app.js --name myapp -i max
  
  # Ecosystem file (ecosystem.config.js)
  module.exports = {
    apps: [{
      name: 'myapp',
      script: './server.js',
      instances: 'max',
      exec_mode: 'cluster',
      env: { NODE_ENV: 'production', PORT: 3000 }
    }]
  };
  
  pm2 start ecosystem.config.js
  pm2 save
  pm2 startup                      # Generate systemd service

Gunicorn (Python WSGI):
  pip install gunicorn
  
  # Direct run
  gunicorn --workers 4 --bind 0.0.0.0:8000 myapp:app
  
  # With systemd
  [Service]
  ExecStart=/usr/local/bin/gunicorn \
      --workers 4 \
      --bind unix:/run/myapp/gunicorn.sock \
      --access-logfile /var/log/myapp/access.log \
      myapp.wsgi:application
` + "```" + ``,
					CodeExamples: `# Application deployment automation

# 1. Application deployment script
#!/bin/bash
set -euo pipefail

APP_NAME="${1:?Usage: $0 <app-name> <version>}"
VERSION="${2:?Usage: $0 <app-name> <version>}"
APP_DIR="/opt/$APP_NAME"
DEPLOY_DIR="$APP_DIR/releases/$VERSION"
CURRENT_LINK="$APP_DIR/current"
SHARED_DIR="$APP_DIR/shared"
MAX_RELEASES=5

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "Deploying $APP_NAME v$VERSION"

# Create directories
mkdir -p "$DEPLOY_DIR" "$SHARED_DIR"/{log,tmp,config}

# Download/copy release
log "Fetching release..."
# Example: download from artifact store
# curl -o "$DEPLOY_DIR/app.tar.gz" "https://releases.example.com/$APP_NAME/$VERSION.tar.gz"
# tar xzf "$DEPLOY_DIR/app.tar.gz" -C "$DEPLOY_DIR"

# Link shared resources
log "Linking shared resources..."
ln -sf "$SHARED_DIR/log" "$DEPLOY_DIR/log"
ln -sf "$SHARED_DIR/config/env" "$DEPLOY_DIR/.env"

# Run migrations (if applicable)
if [ -f "$DEPLOY_DIR/bin/migrate" ]; then
    log "Running migrations..."
    "$DEPLOY_DIR/bin/migrate" up
fi

# Health check on new version
if [ -f "$DEPLOY_DIR/bin/server" ]; then
    log "Running config check..."
    "$DEPLOY_DIR/bin/server" --check-config || {
        log "ERROR: Config check failed"
        rm -rf "$DEPLOY_DIR"
        exit 1
    }
fi

# Switch current symlink
log "Switching to new version..."
ln -sfn "$DEPLOY_DIR" "$CURRENT_LINK"

# Restart service
log "Restarting service..."
systemctl restart "$APP_NAME" 2>/dev/null || true

# Wait for health
log "Waiting for health check..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        log "Health check passed!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        log "ERROR: Health check failed after 30 seconds"
        # Rollback
        PREV=$(ls -t "$APP_DIR/releases/" | sed -n '2p')
        if [ -n "$PREV" ]; then
            log "Rolling back to $PREV"
            ln -sfn "$APP_DIR/releases/$PREV" "$CURRENT_LINK"
            systemctl restart "$APP_NAME"
        fi
        exit 1
    fi
    sleep 1
done

# Cleanup old releases
log "Cleaning old releases..."
ls -t "$APP_DIR/releases/" | tail -n +$((MAX_RELEASES + 1)) | while read -r old; do
    log "  Removing: $old"
    rm -rf "$APP_DIR/releases/$old"
done

log "Deployment complete: $APP_NAME v$VERSION"

# 2. Service health monitor
#!/bin/bash
# Monitor application services and restart if needed
SERVICES=("myapp" "nginx" "postgresql")
HEALTH_ENDPOINTS=("http://localhost:8080/health" "" "")
ALERT_EMAIL="admin@example.com"

for i in "${!SERVICES[@]}"; do
    service="${SERVICES[$i]}"
    health="${HEALTH_ENDPOINTS[$i]}"
    
    # Check systemd status
    if ! systemctl is-active --quiet "$service"; then
        echo "$(date): $service is down, restarting..."
        systemctl restart "$service"
        
        sleep 5
        if systemctl is-active --quiet "$service"; then
            echo "  $service restarted successfully"
        else
            echo "  CRITICAL: $service failed to restart"
        fi
        continue
    fi
    
    # Check health endpoint if defined
    if [ -n "$health" ]; then
        if ! curl -sf --max-time 5 "$health" > /dev/null 2>&1; then
            echo "$(date): $service health check failed"
            
            # Check response code
            code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$health" 2>/dev/null)
            echo "  HTTP status: $code"
            
            # Restart if unhealthy
            systemctl restart "$service"
            echo "  Service restarted"
        fi
    fi
done

# 3. Log rotation configuration
# /etc/logrotate.d/myapp
# /var/log/myapp/*.log {
#     daily
#     missingok
#     rotate 30
#     compress
#     delaycompress
#     notifempty
#     create 0640 myapp myapp
#     sharedscripts
#     postrotate
#         systemctl reload myapp > /dev/null 2>/dev/null || true
#     endscript
# }`,
				},
			},
		},
	})
}
