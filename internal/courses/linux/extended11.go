package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1833,
			Title:       "Configuration Management and Automation",
			Description: "Automate Linux administration with Ansible, infrastructure as code principles, idempotent configuration, and large-scale system management.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "Ansible Fundamentals",
					Content: `Ansible is an agentless automation tool that uses SSH to manage Linux systems. It's the de facto standard for Linux configuration management.

**Ansible Architecture:**
` + "```" + `
Components:
  Control Node:  Machine running Ansible (your laptop/CI server)
  Managed Nodes: Target machines (no agent needed, just SSH + Python)
  Inventory:     List of managed hosts
  Playbooks:     YAML files describing desired state
  Modules:       Units of work (apt, yum, file, template, etc.)
  Roles:         Reusable collection of tasks, files, templates
  Facts:         System information gathered from managed nodes

How it works:
  1. Read inventory (which hosts)
  2. Read playbook (what to do)
  3. SSH to managed nodes
  4. Copy module code to remote (as Python scripts)
  5. Execute modules
  6. Collect results
  7. Clean up temporary files

Installation:
  pip install ansible
  # Or:
  apt install ansible    # Debian/Ubuntu
  dnf install ansible    # RHEL/CentOS

Configuration: ansible.cfg
  [defaults]
  inventory = ./inventory
  remote_user = deploy
  private_key_file = ~/.ssh/id_ed25519
  host_key_checking = False
  forks = 20
  timeout = 30
  retry_files_enabled = False
  
  [privilege_escalation]
  become = True
  become_method = sudo
  become_user = root
  become_ask_pass = False
` + "```" + `

**Inventory:**
` + "```" + `
INI format (inventory/hosts):
  [webservers]
  web1.example.com
  web2.example.com ansible_host=10.0.1.11
  
  [dbservers]
  db1.example.com ansible_host=10.0.2.10
  db2.example.com ansible_host=10.0.2.11
  
  [production:children]
  webservers
  dbservers
  
  [webservers:vars]
  http_port=80
  app_version=2.0.0
  
  [all:vars]
  ansible_user=deploy
  ansible_python_interpreter=/usr/bin/python3

YAML format (inventory/hosts.yml):
  all:
    children:
      webservers:
        hosts:
          web1:
            ansible_host: 10.0.1.10
          web2:
            ansible_host: 10.0.1.11
        vars:
          http_port: 80
      dbservers:
        hosts:
          db1:
            ansible_host: 10.0.2.10
    vars:
      ansible_user: deploy

Dynamic inventory:
  # Script that outputs JSON inventory
  # AWS EC2:
  ansible-inventory -i aws_ec2.yml --list
  # aws_ec2.yml:
  plugin: amazon.aws.aws_ec2
  regions:
    - us-east-1
  filters:
    tag:Environment: production
  keyed_groups:
    - key: tags.Role
      prefix: role
` + "```" + `

**Ad-hoc Commands:**
` + "```" + `
  # Ping all hosts
  ansible all -m ping
  
  # Run command
  ansible webservers -m command -a "uptime"
  ansible webservers -a "free -h"  # command is default module
  
  # Shell (supports pipes, redirects)
  ansible webservers -m shell -a "ps aux | grep nginx | wc -l"
  
  # Package management
  ansible webservers -m apt -a "name=nginx state=present" -b
  
  # Service management
  ansible webservers -m systemd -a "name=nginx state=restarted" -b
  
  # Copy file
  ansible webservers -m copy -a "src=index.html dest=/var/www/html/"
  
  # Gather facts
  ansible web1 -m setup
  ansible web1 -m setup -a "filter=ansible_distribution*"
` + "```" + `

**Playbooks:**
` + "```" + `yaml
---
# site.yml - Main playbook
- name: Configure web servers
  hosts: webservers
  become: yes
  vars:
    app_name: mywebapp
    app_port: 8080
    
  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted
        
    - name: reload nginx
      systemd:
        name: nginx
        state: reloaded
  
  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
      
    - name: Install packages
      apt:
        name:
          - nginx
          - curl
          - htop
        state: present
    
    - name: Create app directory
      file:
        path: "/opt/{{ app_name }}"
        state: directory
        owner: www-data
        group: www-data
        mode: '0755'
    
    - name: Deploy nginx config
      template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/sites-available/default
        validate: "nginx -t -c %s"  # Won't deploy if invalid
      notify: reload nginx
    
    - name: Ensure nginx is running
      systemd:
        name: nginx
        state: started
        enabled: yes
    
    - name: Check application health
      uri:
        url: "http://localhost:{{ app_port }}/healthz"
        status_code: 200
      register: health
      retries: 5
      delay: 3
      until: health.status == 200
` + "```" + `

**Jinja2 Templates:**
` + "```" + `
templates/nginx.conf.j2:
  server {
      listen {{ http_port | default(80) }};
      server_name {{ ansible_fqdn }};
      
      location / {
          proxy_pass http://127.0.0.1:{{ app_port }};
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
      }
      
      {% if enable_ssl | default(false) %}
      listen 443 ssl;
      ssl_certificate /etc/ssl/certs/{{ ansible_hostname }}.crt;
      ssl_certificate_key /etc/ssl/private/{{ ansible_hostname }}.key;
      {% endif %}
      
      {% for location in extra_locations | default([]) %}
      location {{ location.path }} {
          {{ location.directive }};
      }
      {% endfor %}
  }

Template features:
  {{ variable }}                   Variable substitution
  {% if condition %} ... {% endif %}  Conditional
  {% for item in list %} ... {% endfor %}  Loop
  {{ var | default('value') }}     Default filter
  {{ var | upper }}                String filter
  {{ list | join(', ') }}         Join filter
  {{ var | regex_replace('^(.*)$', '\\1_suffix') }}
` + "```" + ``,
					CodeExamples: `# Ansible configuration examples

# 1. Ansible role structure
# roles/webserver/
#   tasks/main.yml
#   handlers/main.yml
#   templates/nginx.conf.j2
#   files/
#   vars/main.yml
#   defaults/main.yml
#   meta/main.yml

# roles/webserver/defaults/main.yml
---
nginx_worker_processes: auto
nginx_worker_connections: 1024
nginx_keepalive_timeout: 65
nginx_client_max_body_size: 10m
app_port: 8080
enable_ssl: false

# roles/webserver/tasks/main.yml
---
- name: Install nginx
  apt:
    name: nginx
    state: present
    
- name: Deploy nginx configuration
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
    owner: root
    group: root
    mode: '0644'
    validate: "nginx -t -c %s"
  notify: restart nginx

- name: Deploy site configuration
  template:
    src: site.conf.j2
    dest: "/etc/nginx/sites-available/{{ app_name }}"
  notify: reload nginx

- name: Enable site
  file:
    src: "/etc/nginx/sites-available/{{ app_name }}"
    dest: "/etc/nginx/sites-enabled/{{ app_name }}"
    state: link
  notify: reload nginx

- name: Remove default site
  file:
    path: /etc/nginx/sites-enabled/default
    state: absent
  notify: reload nginx

- name: Ensure nginx is running
  systemd:
    name: nginx
    state: started
    enabled: yes

# roles/webserver/handlers/main.yml
---
- name: restart nginx
  systemd:
    name: nginx
    state: restarted

- name: reload nginx
  systemd:
    name: nginx
    state: reloaded

# 2. Playbook using role
# site.yml
---
- name: Configure web servers
  hosts: webservers
  become: yes
  roles:
    - role: common
    - role: webserver
      vars:
        app_name: production-app
        enable_ssl: true

- name: Configure databases
  hosts: dbservers
  become: yes
  roles:
    - role: common
    - role: postgresql
      vars:
        pg_version: "15"
        pg_max_connections: 200

# 3. Common role tasks
# roles/common/tasks/main.yml
---
- name: Set timezone
  timezone:
    name: UTC

- name: Install common packages
  apt:
    name:
      - vim
      - curl
      - wget
      - htop
      - iotop
      - net-tools
      - unzip
      - jq
    state: present

- name: Configure sysctl
  sysctl:
    name: "{{ item.key }}"
    value: "{{ item.value }}"
    state: present
    reload: yes
  loop:
    - { key: "net.core.somaxconn", value: "65535" }
    - { key: "vm.swappiness", value: "10" }
    - { key: "fs.file-max", value: "2097152" }

- name: Set file descriptor limits
  pam_limits:
    domain: '*'
    limit_type: "{{ item.type }}"
    limit_item: nofile
    value: "{{ item.value }}"
  loop:
    - { type: soft, value: "65536" }
    - { type: hard, value: "131072" }

- name: Configure NTP
  systemd:
    name: systemd-timesyncd
    state: started
    enabled: yes`,
				},
				{
					Title: "Advanced Ansible and Infrastructure as Code",
					Content: `Advanced Ansible features enable managing complex infrastructure at scale with proper error handling, secrets management, and testing.

**Advanced Playbook Features:**
` + "```" + `
Conditionals:
  - name: Install on Debian
    apt:
      name: nginx
    when: ansible_os_family == "Debian"
  
  - name: Install on RedHat
    dnf:
      name: nginx
    when: ansible_os_family == "RedHat"
  
  - name: Only if variable defined
    debug:
      msg: "{{ custom_var }}"
    when: custom_var is defined

Loops:
  - name: Create users
    user:
      name: "{{ item.name }}"
      groups: "{{ item.groups }}"
      shell: /bin/bash
    loop:
      - { name: alice, groups: "sudo,docker" }
      - { name: bob, groups: "docker" }
  
  - name: Install from list
    apt:
      name: "{{ item }}"
    loop: "{{ packages }}"
    
  - name: Template multiple configs
    template:
      src: "{{ item.src }}"
      dest: "{{ item.dest }}"
    loop:
      - { src: app.conf.j2, dest: /etc/app/app.conf }
      - { src: logging.conf.j2, dest: /etc/app/logging.conf }

Blocks (error handling):
  - name: Deploy with rollback
    block:
      - name: Deploy new version
        copy:
          src: "app-{{ version }}.tar.gz"
          dest: /opt/app/
      
      - name: Extract
        unarchive:
          src: "/opt/app/app-{{ version }}.tar.gz"
          dest: /opt/app/
          remote_src: yes
      
      - name: Restart service
        systemd:
          name: myapp
          state: restarted
          
    rescue:
      - name: Rollback to previous version
        copy:
          src: /opt/app/backup/
          dest: /opt/app/current/
          remote_src: yes
      
      - name: Restart with old version
        systemd:
          name: myapp
          state: restarted
    
    always:
      - name: Clean up temp files
        file:
          path: /opt/app/tmp/
          state: absent

Delegation:
  - name: Remove from load balancer
    uri:
      url: "http://lb.example.com/api/deregister/{{ inventory_hostname }}"
      method: POST
    delegate_to: localhost
  
  - name: Wait for connections to drain
    wait_for:
      host: "{{ inventory_hostname }}"
      port: 80
      state: drained
      timeout: 60
    delegate_to: localhost

Serial execution (rolling deploy):
  - hosts: webservers
    serial: "25%"       # Deploy to 25% of hosts at a time
    max_fail_percentage: 10
    tasks:
      - name: Deploy application
        # ...
` + "```" + `

**Ansible Vault (Secrets):**
` + "```" + `
  # Create encrypted file
  ansible-vault create secrets.yml
  
  # Edit encrypted file
  ansible-vault edit secrets.yml
  
  # Encrypt existing file
  ansible-vault encrypt vars/production.yml
  
  # Decrypt
  ansible-vault decrypt vars/production.yml
  
  # View without decrypting
  ansible-vault view secrets.yml
  
  # Run playbook with vault
  ansible-playbook site.yml --ask-vault-pass
  ansible-playbook site.yml --vault-password-file ~/.vault_pass
  
  # Encrypt single variable
  ansible-vault encrypt_string 'mysecretpassword' --name 'db_password'
  # Output:
  # db_password: !vault |
  #   $ANSIBLE_VAULT;1.1;AES256
  #   ...

Using in playbooks:
  vars_files:
    - vars/common.yml
    - vars/secrets.yml   # Encrypted
    
  # Or inline encrypted variables in any YAML file
  db_password: !vault |
    $ANSIBLE_VAULT;1.1;AES256
    6637...encrypted...data
` + "```" + `

**Testing and CI/CD:**
` + "```" + `
Ansible Lint:
  ansible-lint playbook.yml
  ansible-lint roles/webserver/

Molecule (role testing):
  # Install
  pip install molecule molecule-docker
  
  # Initialize
  cd roles/webserver
  molecule init scenario -d docker
  
  # Test lifecycle
  molecule create       # Create test instance
  molecule converge     # Run playbook
  molecule idempotence  # Run again (should have no changes)
  molecule verify       # Run tests (Testinfra)
  molecule destroy      # Clean up
  molecule test         # Full lifecycle

Testinfra (infrastructure testing):
  # molecule/default/tests/test_default.py
  def test_nginx_installed(host):
      nginx = host.package("nginx")
      assert nginx.is_installed
  
  def test_nginx_running(host):
      nginx = host.service("nginx")
      assert nginx.is_running
      assert nginx.is_enabled
  
  def test_nginx_listening(host):
      socket = host.socket("tcp://0.0.0.0:80")
      assert socket.is_listening

Dry run (check mode):
  ansible-playbook site.yml --check --diff
  # Shows what WOULD change without changing anything
  # --diff shows file diffs
` + "```" + ``,
					CodeExamples: `# Advanced Ansible patterns

# 1. Rolling deployment playbook
---
- name: Rolling deploy to web servers
  hosts: webservers
  serial: 1                    # One host at a time
  max_fail_percentage: 0       # Stop on any failure
  become: yes
  
  vars:
    app_version: "{{ version }}"
    health_check_url: "http://localhost:8080/healthz"
    lb_api: "http://lb.example.com/api"
  
  pre_tasks:
    - name: Deregister from load balancer
      uri:
        url: "{{ lb_api }}/deregister"
        method: POST
        body_format: json
        body:
          host: "{{ inventory_hostname }}"
      delegate_to: localhost
    
    - name: Wait for connections to drain
      pause:
        seconds: 30
  
  tasks:
    - name: Stop application
      systemd:
        name: myapp
        state: stopped
    
    - name: Deploy new version
      unarchive:
        src: "releases/myapp-{{ app_version }}.tar.gz"
        dest: /opt/myapp/
    
    - name: Update symlink
      file:
        src: "/opt/myapp/myapp-{{ app_version }}"
        dest: /opt/myapp/current
        state: link
    
    - name: Start application
      systemd:
        name: myapp
        state: started
    
    - name: Wait for health check
      uri:
        url: "{{ health_check_url }}"
        status_code: 200
      register: health
      retries: 10
      delay: 5
      until: health.status == 200
  
  post_tasks:
    - name: Register with load balancer
      uri:
        url: "{{ lb_api }}/register"
        method: POST
        body_format: json
        body:
          host: "{{ inventory_hostname }}"
      delegate_to: localhost

# 2. Dynamic host provisioning
---
- name: Provision and configure new servers
  hosts: localhost
  gather_facts: no
  
  tasks:
    - name: Create VMs
      community.libvirt.virt:
        name: "web-{{ item }}"
        state: running
        xml: "{{ lookup('template', 'vm.xml.j2') }}"
      loop: "{{ range(1, server_count + 1) | list }}"
      register: created_vms
    
    - name: Wait for SSH
      wait_for:
        host: "{{ item }}"
        port: 22
        delay: 10
        timeout: 300
      loop: "{{ created_vms.results | map(attribute='item') | list }}"
    
    - name: Add to dynamic inventory
      add_host:
        name: "web-{{ item }}"
        groups: new_webservers
      loop: "{{ range(1, server_count + 1) | list }}"

- name: Configure new servers
  hosts: new_webservers
  become: yes
  roles:
    - common
    - webserver

# 3. Ansible callback for Slack notifications
# ansible.cfg:
# [defaults]
# callback_whitelist = community.general.slack
#
# [callback_slack]
# webhook_url = https://hooks.slack.com/services/xxx
# channel = #deployments
# username = Ansible

# 4. Inventory management script
#!/bin/bash
# Generate Ansible inventory from infrastructure
echo "[webservers]"
for i in $(seq 1 3); do
    echo "web${i} ansible_host=10.0.1.${i}0"
done

echo ""
echo "[dbservers]"
echo "db1 ansible_host=10.0.2.10"

echo ""
echo "[all:vars]"
echo "ansible_user=deploy"
echo "ansible_python_interpreter=/usr/bin/python3"`,
				},
			},
		},
	})
}
