package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1460,
			Title:       "GitOps and Configuration Management",
			Description: "Implement GitOps workflows with ArgoCD and Flux, and manage configuration with Ansible, Salt, and modern configuration management approaches.",
			Order:       60,
			Lessons: []problems.Lesson{
				{
					Title: "GitOps with ArgoCD and Flux",
					Content: `GitOps uses Git as the single source of truth for declarative infrastructure and application configuration.

**GitOps Principles:**
` + "```" + `
Core principles:
  1. Declarative:
     Entire system described declaratively
     Kubernetes manifests, Helm charts, Kustomize
  
  2. Versioned and Immutable:
     Desired state stored in Git
     Complete audit trail
     Rollback = git revert
  
  3. Pulled Automatically:
     Agents pull desired state from Git
     No push-based access needed
     No CI system needs cluster credentials
  
  4. Continuously Reconciled:
     Agents continuously compare actual vs desired
     Auto-correct drift
     Self-healing infrastructure

GitOps workflow:
  Developer → Git Push → Git Repository
                              ↓
                     GitOps Agent (ArgoCD/Flux)
                              ↓
                     Kubernetes Cluster
                              ↓
                     Reconciliation Loop
                     (actual == desired?)

Repository patterns:
  Mono-repo:
    gitops-repo/
    ├── apps/
    │   ├── app1/
    │   │   ├── base/
    │   │   └── overlays/
    │   │       ├── dev/
    │   │       ├── staging/
    │   │       └── prod/
    │   └── app2/
    └── infrastructure/
        ├── cert-manager/
        ├── ingress-nginx/
        └── monitoring/
  
  Multi-repo:
    app-source-repo    (application code + Dockerfile)
    app-config-repo    (Kubernetes manifests)
    infra-config-repo  (cluster-level config)
` + "```" + `

**ArgoCD:**
` + "```" + `
Installation:
  kubectl create namespace argocd
  kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
  
  # CLI access
  argocd login <argocd-server> --username admin --password <password>
  
  # Get initial password
  kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

Application definition:
  apiVersion: argoproj.io/v1alpha1
  kind: Application
  metadata:
    name: myapp
    namespace: argocd
  spec:
    project: default
    source:
      repoURL: https://github.com/myorg/gitops-repo.git
      targetRevision: main
      path: apps/myapp/overlays/production
    destination:
      server: https://kubernetes.default.svc
      namespace: production
    syncPolicy:
      automated:
        prune: true
        selfHeal: true
        allowEmpty: false
      syncOptions:
        - CreateNamespace=true
        - PrunePropagationPolicy=foreground
      retry:
        limit: 5
        backoff:
          duration: 5s
          factor: 2
          maxDuration: 3m

ApplicationSet (manage multiple apps):
  apiVersion: argoproj.io/v1alpha1
  kind: ApplicationSet
  metadata:
    name: myapp-set
    namespace: argocd
  spec:
    generators:
      - git:
          repoURL: https://github.com/myorg/gitops-repo.git
          revision: main
          directories:
            - path: apps/*
      - list:
          elements:
            - cluster: dev
              url: https://dev-cluster
            - cluster: staging
              url: https://staging-cluster
            - cluster: prod
              url: https://prod-cluster
    template:
      metadata:
        name: '{{path.basename}}-{{cluster}}'
      spec:
        project: default
        source:
          repoURL: https://github.com/myorg/gitops-repo.git
          targetRevision: main
          path: '{{path}}/overlays/{{cluster}}'
        destination:
          server: '{{url}}'
          namespace: '{{path.basename}}'

Sync waves and hooks:
  # Pre-sync hook (run before sync)
  apiVersion: batch/v1
  kind: Job
  metadata:
    name: db-migrate
    annotations:
      argocd.argoproj.io/hook: PreSync
      argocd.argoproj.io/hook-delete-policy: HookSucceeded
  spec:
    template:
      spec:
        containers:
          - name: migrate
            image: myapp:latest
            command: ["./migrate", "up"]
        restartPolicy: Never
  
  # Sync wave ordering
  metadata:
    annotations:
      argocd.argoproj.io/sync-wave: "1"  # Lower = earlier

Multi-cluster:
  # Add cluster
  argocd cluster add my-context --name production
  
  # Application targeting specific cluster
  spec:
    destination:
      server: https://production-cluster:6443
      namespace: default

Image Updater:
  # Automatically update image tags
  apiVersion: argoproj.io/v1alpha1
  kind: Application
  metadata:
    annotations:
      argocd-image-updater.argoproj.io/image-list: myapp=myregistry/myapp
      argocd-image-updater.argoproj.io/myapp.update-strategy: semver
      argocd-image-updater.argoproj.io/myapp.allow-tags: regexp:^v[0-9]+\.[0-9]+\.[0-9]+$
` + "```" + `

**Flux CD:**
` + "```" + `
Installation:
  flux bootstrap github \
    --owner=myorg \
    --repository=fleet-infra \
    --branch=main \
    --path=clusters/production \
    --personal

Source (Git repository):
  apiVersion: source.toolkit.fluxcd.io/v1
  kind: GitRepository
  metadata:
    name: myapp
    namespace: flux-system
  spec:
    interval: 1m
    url: https://github.com/myorg/myapp-config
    ref:
      branch: main
    secretRef:
      name: git-credentials

Kustomization (reconciliation):
  apiVersion: kustomize.toolkit.fluxcd.io/v1
  kind: Kustomization
  metadata:
    name: myapp
    namespace: flux-system
  spec:
    interval: 5m
    path: ./overlays/production
    prune: true
    sourceRef:
      kind: GitRepository
      name: myapp
    healthChecks:
      - apiVersion: apps/v1
        kind: Deployment
        name: myapp
        namespace: production
    timeout: 3m

HelmRelease:
  apiVersion: helm.toolkit.fluxcd.io/v2beta2
  kind: HelmRelease
  metadata:
    name: nginx-ingress
    namespace: flux-system
  spec:
    interval: 5m
    chart:
      spec:
        chart: ingress-nginx
        version: ">=4.0.0 <5.0.0"
        sourceRef:
          kind: HelmRepository
          name: ingress-nginx
    values:
      controller:
        replicas: 2
        resources:
          requests:
            cpu: 100m
            memory: 128Mi

Image automation:
  apiVersion: image.toolkit.fluxcd.io/v1beta2
  kind: ImageRepository
  metadata:
    name: myapp
  spec:
    image: myregistry/myapp
    interval: 1m
  ---
  apiVersion: image.toolkit.fluxcd.io/v1beta2
  kind: ImagePolicy
  metadata:
    name: myapp
  spec:
    imageRepositoryRef:
      name: myapp
    policy:
      semver:
        range: ">=1.0.0"
  ---
  apiVersion: image.toolkit.fluxcd.io/v1beta1
  kind: ImageUpdateAutomation
  metadata:
    name: myapp
  spec:
    interval: 1m
    sourceRef:
      kind: GitRepository
      name: myapp
    git:
      checkout:
        ref:
          branch: main
      commit:
        author:
          email: flux@example.com
        messageTemplate: "Update image to {{.NewTag}}"
      push:
        branch: main
    update:
      path: ./overlays/production
` + "```" + ``,
					CodeExamples: `# GitOps management scripts

# 1. ArgoCD status dashboard
#!/bin/bash
echo "=== ArgoCD Status ==="

# Applications
echo "--- Applications ---"
argocd app list 2>/dev/null || \
    kubectl get applications -n argocd \
        -o custom-columns='NAME:.metadata.name,SYNC:.status.sync.status,HEALTH:.status.health.status,REPO:.spec.source.repoURL' \
        2>/dev/null

# Out of sync apps
echo ""
echo "--- Out of Sync ---"
kubectl get applications -n argocd -o json 2>/dev/null | \
    jq -r '.items[] | select(.status.sync.status != "Synced") | "\(.metadata.name): \(.status.sync.status)"' 2>/dev/null

# Unhealthy apps
echo ""
echo "--- Unhealthy ---"
kubectl get applications -n argocd -o json 2>/dev/null | \
    jq -r '.items[] | select(.status.health.status != "Healthy") | "\(.metadata.name): \(.status.health.status)"' 2>/dev/null

# Recent sync history
echo ""
echo "--- Recent Syncs ---"
for app in $(kubectl get applications -n argocd -o name 2>/dev/null | head -5); do
    NAME=$(echo "$app" | cut -d'/' -f2)
    echo "  $NAME:"
    kubectl get "$app" -n argocd -o json 2>/dev/null | \
        jq -r '.status.history[-3:][] | "    \(.deployedAt): \(.revision[:8])"' 2>/dev/null
done

# 2. Flux status checker
#!/bin/bash
echo "=== Flux Status ==="

# Sources
echo "--- Git Sources ---"
flux get sources git 2>/dev/null || \
    kubectl get gitrepositories -n flux-system 2>/dev/null

# Kustomizations
echo ""
echo "--- Kustomizations ---"
flux get kustomizations 2>/dev/null || \
    kubectl get kustomizations -n flux-system 2>/dev/null

# Helm releases
echo ""
echo "--- Helm Releases ---"
flux get helmreleases --all-namespaces 2>/dev/null || \
    kubectl get helmreleases --all-namespaces 2>/dev/null

# Image policies
echo ""
echo "--- Image Policies ---"
flux get image policy --all-namespaces 2>/dev/null

# 3. GitOps drift detector
#!/bin/bash
echo "=== GitOps Drift Detection ==="

# ArgoCD diff
echo "--- ArgoCD Diffs ---"
for app in $(argocd app list -o name 2>/dev/null); do
    DIFF=$(argocd app diff "$app" 2>/dev/null)
    if [ -n "$DIFF" ]; then
        echo "DRIFT in $app:"
        echo "$DIFF" | head -10
        echo "..."
    fi
done

# Flux suspended resources
echo ""
echo "--- Suspended Flux Resources ---"
kubectl get kustomizations,helmreleases --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.suspend == true) | "\(.kind)/\(.metadata.name) in \(.metadata.namespace)"' 2>/dev/null`,
				},
				{
					Title: "Configuration Management with Ansible",
					Content: `Ansible provides agentless configuration management for automating server provisioning, application deployment, and orchestration.

**Ansible Fundamentals:**
` + "```" + `
Architecture:
  Control node:    Machine running Ansible
  Managed nodes:   Target servers (SSH/WinRM)
  Inventory:       List of managed nodes
  Playbook:        YAML automation scripts
  Roles:           Reusable automation packages
  Collections:     Distribution format for content

Inventory:
  # inventory/hosts.yml
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
          max_clients: 200
      
      databases:
        hosts:
          db1:
            ansible_host: 10.0.2.10
          db2:
            ansible_host: 10.0.2.11
        vars:
          db_port: 5432
    
    vars:
      ansible_user: deploy
      ansible_ssh_private_key_file: ~/.ssh/deploy_key

Playbook structure:
  ---
  - name: Configure web servers
    hosts: webservers
    become: true
    vars:
      app_version: "1.5.0"
    
    pre_tasks:
      - name: Update apt cache
        apt:
          update_cache: true
          cache_valid_time: 3600
    
    tasks:
      - name: Install nginx
        apt:
          name: nginx
          state: present
      
      - name: Configure nginx
        template:
          src: templates/nginx.conf.j2
          dest: /etc/nginx/nginx.conf
        notify: Restart nginx
      
      - name: Deploy application
        copy:
          src: "files/app-{{ app_version }}.tar.gz"
          dest: /opt/app/
        notify: Restart application
      
      - name: Ensure services running
        service:
          name: "{{ item }}"
          state: started
          enabled: true
        loop:
          - nginx
          - app
    
    handlers:
      - name: Restart nginx
        service:
          name: nginx
          state: restarted
      
      - name: Restart application
        service:
          name: app
          state: restarted

Roles:
  roles/
  └── webserver/
      ├── defaults/
      │   └── main.yml    (default variables)
      ├── files/           (static files)
      ├── handlers/
      │   └── main.yml    (handlers)
      ├── meta/
      │   └── main.yml    (role metadata/dependencies)
      ├── tasks/
      │   └── main.yml    (tasks)
      ├── templates/       (Jinja2 templates)
      └── vars/
          └── main.yml    (role variables)
  
  # Using role in playbook
  - hosts: webservers
    roles:
      - role: webserver
        vars:
          http_port: 8080
      - role: monitoring
        tags: [monitoring]
` + "```" + `

**Advanced Ansible:**
` + "```" + `
Jinja2 templates:
  # templates/nginx.conf.j2
  server {
      listen {{ http_port }};
      server_name {{ ansible_hostname }};
      
      {% for location in app_locations %}
      location {{ location.path }} {
          proxy_pass http://{{ location.upstream }};
      }
      {% endfor %}
      
      {% if ssl_enabled %}
      listen 443 ssl;
      ssl_certificate /etc/ssl/{{ domain }}.crt;
      ssl_certificate_key /etc/ssl/{{ domain }}.key;
      {% endif %}
  }

Conditionals and loops:
  - name: Install packages per OS
    apt:
      name: "{{ item }}"
    loop: "{{ debian_packages }}"
    when: ansible_os_family == "Debian"
  
  - name: Configure users
    user:
      name: "{{ item.name }}"
      groups: "{{ item.groups | join(',') }}"
      state: "{{ item.state | default('present') }}"
    loop: "{{ users }}"
    when: item.state != 'absent'
  
  # Block with error handling
  - block:
      - name: Deploy application
        copy:
          src: app.tar.gz
          dest: /opt/app/
      - name: Start application
        service:
          name: app
          state: started
    rescue:
      - name: Rollback
        copy:
          src: app-previous.tar.gz
          dest: /opt/app/
      - name: Start previous version
        service:
          name: app
          state: started
    always:
      - name: Send notification
        slack:
          token: "{{ slack_token }}"
          msg: "Deployment {{ 'succeeded' if not ansible_failed_task else 'failed' }}"

Vault (secrets):
  # Encrypt file
  ansible-vault encrypt group_vars/production/secrets.yml
  
  # Decrypt
  ansible-vault decrypt group_vars/production/secrets.yml
  
  # Edit encrypted file
  ansible-vault edit group_vars/production/secrets.yml
  
  # Run playbook with vault
  ansible-playbook site.yml --ask-vault-pass
  ansible-playbook site.yml --vault-password-file .vault_pass

  # Encrypt single variable
  ansible-vault encrypt_string 'mysecret' --name 'db_password'
  
  # In vars file:
  db_password: !vault |
    $ANSIBLE_VAULT;1.1;AES256
    36623962353263...

Dynamic inventory:
  # AWS EC2 plugin
  # inventory/aws_ec2.yml
  plugin: amazon.aws.aws_ec2
  regions:
    - us-east-1
  keyed_groups:
    - key: tags.Environment
      prefix: env
    - key: instance_type
      prefix: type
  filters:
    tag:ManagedBy: ansible
  compose:
    ansible_host: public_ip_address

Testing with Molecule:
  # Initialize
  molecule init scenario -d docker
  
  # Test lifecycle
  molecule create     # Create instances
  molecule converge   # Run playbook
  molecule verify     # Run tests
  molecule destroy    # Cleanup
  molecule test       # Full cycle
  
  # molecule/default/molecule.yml
  driver:
    name: docker
  platforms:
    - name: ubuntu
      image: ubuntu:22.04
    - name: centos
      image: centos:stream9
  provisioner:
    name: ansible
  verifier:
    name: ansible
` + "```" + ``,
					CodeExamples: `# Configuration management scripts

# 1. Ansible inventory reporter
#!/bin/bash
echo "=== Ansible Inventory Report ==="

INVENTORY="${1:-inventory}"

# List hosts by group
echo "--- Hosts by Group ---"
ansible-inventory -i "$INVENTORY" --list 2>/dev/null | \
    jq -r 'to_entries[] | select(.value | type == "object" and has("hosts")) | "\(.key): \(.value.hosts | join(", "))"' 2>/dev/null

# Host count
echo ""
echo "--- Host Count ---"
TOTAL=$(ansible-inventory -i "$INVENTORY" --list 2>/dev/null | jq -r '._meta.hostvars | keys | length' 2>/dev/null)
echo "  Total hosts: $TOTAL"

# Variables preview
echo ""
echo "--- Group Variables ---"
for group in $(ansible-inventory -i "$INVENTORY" --list 2>/dev/null | jq -r 'to_entries[] | select(.value | type == "object" and has("vars")) | .key' 2>/dev/null); do
    echo "  $group:"
    ansible-inventory -i "$INVENTORY" --list 2>/dev/null | jq -r ".\"$group\".vars | to_entries[] | \"    \(.key): \(.value)\"" 2>/dev/null
done

# 2. Playbook dry-run checker
#!/bin/bash
echo "=== Playbook Dry Run ==="

PLAYBOOK="${1:-site.yml}"
INVENTORY="${2:-inventory}"

echo "Playbook: $PLAYBOOK"
echo "Inventory: $INVENTORY"
echo ""

# Syntax check
echo "--- Syntax Check ---"
ansible-playbook "$PLAYBOOK" -i "$INVENTORY" --syntax-check 2>&1

# List tasks
echo ""
echo "--- Task List ---"
ansible-playbook "$PLAYBOOK" -i "$INVENTORY" --list-tasks 2>&1

# List hosts
echo ""
echo "--- Target Hosts ---"
ansible-playbook "$PLAYBOOK" -i "$INVENTORY" --list-hosts 2>&1

# Check mode (dry run)
echo ""
echo "--- Check Mode ---"
ansible-playbook "$PLAYBOOK" -i "$INVENTORY" --check --diff 2>&1 | head -50

# 3. Configuration drift detector
#!/bin/bash
echo "=== Configuration Drift Detection ==="

PLAYBOOK="${1:-site.yml}"
INVENTORY="${2:-inventory}"

echo "Running check mode to detect drift..."
echo ""

OUTPUT=$(ansible-playbook "$PLAYBOOK" -i "$INVENTORY" --check --diff 2>&1)

CHANGED=$(echo "$OUTPUT" | grep -c "changed=")
OK=$(echo "$OUTPUT" | grep -c "ok=")

echo "Summary:"
echo "  Tasks OK (no drift): $OK"
echo "  Tasks Changed (drift): $CHANGED"

if [ "$CHANGED" -gt 0 ]; then
    echo ""
    echo "--- Drifted Tasks ---"
    echo "$OUTPUT" | grep "changed:" | head -20
fi`,
				},
			},
		},
	})
}
