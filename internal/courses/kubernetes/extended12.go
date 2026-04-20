package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1730,
			Title:       "Kubernetes Configuration Management",
			Description: "Master configuration patterns: ConfigMaps, Secrets, external secret management, environment variables, and configuration best practices.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "ConfigMaps and Secrets Management",
					Content: `ConfigMaps and Secrets are the primary ways to provide configuration to containers. Understanding their patterns, limitations, and security implications is essential.

**ConfigMap:**
` + "```" + `yaml
# From literal values
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
data:
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"
  FEATURE_FLAGS: "email=true,sms=false"
  
  # Multi-line config file
  app.yaml: |
    server:
      port: 8080
      read_timeout: 30s
      write_timeout: 30s
    database:
      max_open_conns: 25
      max_idle_conns: 5
      conn_max_lifetime: 5m
    cache:
      ttl: 300s
      max_size: 1000

  # nginx.conf
  nginx.conf: |
    worker_processes auto;
    events {
        worker_connections 1024;
    }
    http {
        server {
            listen 80;
            location / {
                proxy_pass http://localhost:8080;
            }
            location /health {
                return 200 'OK';
            }
        }
    }
` + "```" + `

**Using ConfigMaps:**
` + "```" + `yaml
# As environment variables
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: app
    image: myapp:v1
    # Individual keys
    env:
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: LOG_LEVEL
    # All keys as env vars
    envFrom:
    - configMapRef:
        name: app-config
        prefix: APP_    # Optional prefix: APP_LOG_LEVEL, etc.

---
# As mounted files
apiVersion: v1
kind: Pod
metadata:
  name: app-with-files
spec:
  containers:
  - name: app
    image: myapp:v1
    volumeMounts:
    - name: config
      mountPath: /etc/app
      readOnly: true
    - name: nginx-config
      mountPath: /etc/nginx/nginx.conf
      subPath: nginx.conf    # Mount single file without overwriting dir
  volumes:
  - name: config
    configMap:
      name: app-config
      items:             # Select specific keys
      - key: app.yaml
        path: config.yaml  # Rename in mount
  - name: nginx-config
    configMap:
      name: app-config

# Mounted ConfigMaps auto-update (kubelet sync period ~1 min)
# Environment variables do NOT update (requires pod restart)
# subPath mounts do NOT auto-update
` + "```" + `

**Secrets:**
` + "```" + `yaml
# Secret types:
#   Opaque           → arbitrary data (default)
#   kubernetes.io/tls → TLS cert + key
#   kubernetes.io/dockerconfigjson → image pull credentials
#   kubernetes.io/basic-auth → user/password
#   kubernetes.io/ssh-auth → SSH key

apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
  namespace: production
type: Opaque
data:
  # Base64 encoded (NOT encrypted!)
  username: cG9zdGdyZXM=     # echo -n "postgres" | base64
  password: c3VwZXJzZWNyZXQ=  # echo -n "supersecret" | base64
stringData:
  # Plain text (converted to base64 on create)
  connection-string: "postgresql://postgres:supersecret@db:5432/myapp"

---
# TLS Secret
apiVersion: v1
kind: Secret
metadata:
  name: app-tls
  namespace: production
type: kubernetes.io/tls
data:
  tls.crt: <base64-cert>
  tls.key: <base64-key>

---
# Image pull secret
apiVersion: v1
kind: Secret
metadata:
  name: registry-credentials
  namespace: production
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: <base64-docker-config>
` + "```" + `

**Secret Security:**
` + "```" + `
Kubernetes Secrets limitations:
  ✗ Base64 is encoding, NOT encryption
  ✗ Stored as plaintext in etcd (unless encryption at rest enabled)
  ✗ Anyone with RBAC access to secrets can read them
  ✗ Mounted secrets visible in /proc/mounts

Hardening:
  ✓ Enable encryption at rest for etcd:
      apiServer:
        encryptionConfiguration:
          resources:
          - resources: [secrets]
            providers:
            - aescbc:
                keys:
                - name: key1
                  secret: <base64-encryption-key>
            - identity: {}
  
  ✓ Use RBAC to restrict secret access
  ✓ Use external secret managers (Vault, AWS SM, etc.)
  ✓ Enable audit logging for secret access
  ✓ Rotate secrets regularly
  ✓ Use short-lived tokens (ServiceAccount token projection)
  ✓ Don't use secrets for large data (use ConfigMaps)
` + "```" + ``,
					CodeExamples: `# Configuration Management Examples

# 1. External Secrets Operator (ESO)
# Syncs secrets from external providers to Kubernetes secrets

apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secretsmanager
  namespace: production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: db-credentials
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: production/database
      property: username
  - secretKey: password
    remoteRef:
      key: production/database
      property: password
  - secretKey: connection-string
    remoteRef:
      key: production/database
      property: connection_string

---
# ClusterSecretStore for shared access
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: vault
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: api-keys
  namespace: production
spec:
  refreshInterval: 30m
  secretStoreRef:
    name: vault
    kind: ClusterSecretStore
  target:
    name: api-keys
  data:
  - secretKey: stripe-key
    remoteRef:
      key: secret/data/production/stripe
      property: api_key
  - secretKey: sendgrid-key
    remoteRef:
      key: secret/data/production/sendgrid
      property: api_key

---
# 2. Sealed Secrets (Bitnami)
# Encrypt secrets for safe storage in Git
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: db-credentials
  namespace: production
spec:
  encryptedData:
    username: AgA1H2... # Encrypted by kubeseal
    password: AgB2K9...
  template:
    metadata:
      name: db-credentials
      namespace: production

---
# 3. ConfigMap with immutable flag
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-v2
  namespace: production
immutable: true  # Cannot be updated, only deleted and recreated
data:
  config.yaml: |
    version: 2
    feature_flags:
      new_ui: true
      dark_mode: true

---
# 4. Projected volumes (combine multiple sources)
apiVersion: v1
kind: Pod
metadata:
  name: app-projected
  namespace: production
spec:
  containers:
  - name: app
    image: myapp:v1
    volumeMounts:
    - name: all-config
      mountPath: /etc/app
      readOnly: true
  volumes:
  - name: all-config
    projected:
      sources:
      - configMap:
          name: app-config
          items:
          - key: app.yaml
            path: config.yaml
      - secret:
          name: db-credentials
          items:
          - key: password
            path: db-password
      - serviceAccountToken:
          path: token
          expirationSeconds: 3600
          audience: vault
      - downwardAPI:
          items:
          - path: labels
            fieldRef:
              fieldPath: metadata.labels
          - path: cpu-limit
            resourceFieldRef:
              containerName: app
              resource: limits.cpu

---
# 5. Reloader — auto restart on config change
# https://github.com/stakater/Reloader
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
  annotations:
    reloader.stakater.com/auto: "true"
    # OR specific:
    # configmap.reloader.stakater.com/reload: "app-config"
    # secret.reloader.stakater.com/reload: "db-credentials"
spec:
  template:
    spec:
      containers:
      - name: app
        image: myapp:v1
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: db-credentials

---
# 6. HashiCorp Vault Agent Injector
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "myapp"
        vault.hashicorp.com/agent-inject-secret-db: "secret/data/production/database"
        vault.hashicorp.com/agent-inject-template-db: |
          {{- with secret "secret/data/production/database" -}}
          export DB_HOST="{{ .Data.data.host }}"
          export DB_USER="{{ .Data.data.username }}"
          export DB_PASS="{{ .Data.data.password }}"
          {{- end -}}
    spec:
      serviceAccountName: myapp
      containers:
      - name: app
        image: myapp:v1
        command: ["sh", "-c", "source /vault/secrets/db && ./myapp"]`,
				},
				{
					Title: "Advanced Configuration Patterns",
					Content: `Beyond basic ConfigMaps and Secrets, Kubernetes offers sophisticated patterns for managing configuration across environments and at scale.

**Environment-Specific Configuration:**
` + "```" + `
Pattern 1: Kustomize overlays
  base/
    configmap.yaml    # Common config
    deployment.yaml
    kustomization.yaml
  overlays/
    dev/
      configmap-patch.yaml   # Dev overrides
      kustomization.yaml
    production/
      configmap-patch.yaml   # Production overrides
      kustomization.yaml

Pattern 2: Helm values per environment
  helm install myapp ./chart -f values-production.yaml
  helm install myapp ./chart -f values-staging.yaml
  # values.yaml (defaults) + values-<env>.yaml (overrides)

Pattern 3: ConfigMap per version (immutable)
  app-config-v1 (immutable: true)
  app-config-v2 (immutable: true)
  → Deployment references specific version
  → Rollback = reference old ConfigMap version
  → Clean up: delete old ConfigMaps
` + "```" + `

**Downward API:**
` + "```" + `yaml
# Inject pod metadata into containers
apiVersion: v1
kind: Pod
metadata:
  name: app
  labels:
    app: myapp
    version: v2
  annotations:
    team: platform
spec:
  containers:
  - name: app
    image: myapp:v1
    env:
    # Pod metadata as env vars
    - name: POD_NAME
      valueFrom:
        fieldRef:
          fieldPath: metadata.name
    - name: POD_NAMESPACE
      valueFrom:
        fieldRef:
          fieldPath: metadata.namespace
    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
    - name: NODE_NAME
      valueFrom:
        fieldRef:
          fieldPath: spec.nodeName
    - name: NODE_IP
      valueFrom:
        fieldRef:
          fieldPath: status.hostIP
    # Resource limits
    - name: CPU_LIMIT
      valueFrom:
        resourceFieldRef:
          containerName: app
          resource: limits.cpu
    - name: MEMORY_LIMIT
      valueFrom:
        resourceFieldRef:
          containerName: app
          resource: limits.memory
    # ServiceAccount name
    - name: SERVICE_ACCOUNT
      valueFrom:
        fieldRef:
          fieldPath: spec.serviceAccountName
    resources:
      limits:
        cpu: "2"
        memory: 1Gi
` + "```" + `

**Configuration Validation:**
` + "```" + `
Approaches to validate configuration:

1. Init container validation:
   initContainers:
   - name: config-validator
     image: myapp:v1
     command: ["./myapp", "--validate-config", "/etc/app/config.yaml"]
     volumeMounts:
     - name: config
       mountPath: /etc/app

2. Admission webhook:
   - ValidatingWebhookConfiguration watches ConfigMaps
   - Validates structure and values before creation
   - Blocks invalid configs from being applied

3. CI/CD validation:
   - Lint YAML syntax
   - Schema validation (JSON Schema)
   - Dry-run: kubectl apply --dry-run=server
   - Policy checking: OPA/Kyverno

4. Helm schema validation:
   - values.schema.json validates values before rendering
   - helm template --debug catches rendering errors
` + "```" + `

**ConfigMap and Secret Limits:**
` + "```" + `
Limits:
  - ConfigMap/Secret max size: 1 MiB (etcd limit)
  - etcd max value size: 1.5 MiB by default
  - Environment variable total size: platform dependent (~32KB Linux)
  - Number of ConfigMaps/Secrets: limited by etcd storage

Performance impact:
  - Many Secrets mounted as files → kubelet watches/syncs
  - Projected volumes with many sources → slower mount
  - Large ConfigMaps → slow API server responses

Best practices:
  ✓ Keep ConfigMaps small and focused
  ✓ Use immutable ConfigMaps when possible (reduces kubelet load)
  ✓ Don't store binary data in ConfigMaps (use binaryData field if needed)
  ✓ Use external configuration services for large configs
  ✓ Label and organize ConfigMaps for easy management
  ✓ Include version in ConfigMap name for easy rollback
  ✓ Use checksum annotation on Deployments for automatic rollout:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
` + "```" + ``,
					CodeExamples: `# Advanced Configuration Patterns

# 1. Multi-environment ConfigMap with Kustomize
# base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"
  CACHE_TTL: "300"
  MAX_RETRIES: "3"
  config.yaml: |
    server:
      port: 8080
      graceful_shutdown: 30s
    observability:
      metrics: true
      tracing: true

# overlays/production/configmap-patch.yaml
# apiVersion: v1
# kind: ConfigMap
# metadata:
#   name: app-config
# data:
#   LOG_LEVEL: "warn"
#   CACHE_TTL: "3600"
#   MAX_RETRIES: "5"

---
# 2. Config reload without restart (sidecar pattern)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-reload
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:v1
        ports:
        - containerPort: 8080
        - containerPort: 8081  # Admin/reload port
        volumeMounts:
        - name: config
          mountPath: /etc/app
          readOnly: true
        # App watches /etc/app for changes and reloads
      - name: config-reloader
        image: jimmidyson/configmap-reload:v0.9.0
        args:
        - --volume-dir=/etc/app
        - --webhook-url=http://localhost:8081/-/reload
        - --webhook-method=POST
        volumeMounts:
        - name: config
          mountPath: /etc/app
          readOnly: true
        resources:
          limits:
            cpu: 50m
            memory: 32Mi
      volumes:
      - name: config
        configMap:
          name: app-config

---
# 3. Feature flags via ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: feature-flags
  namespace: production
data:
  flags.json: |
    {
      "new_checkout_flow": {
        "enabled": true,
        "rollout_percentage": 25
      },
      "dark_mode": {
        "enabled": true,
        "rollout_percentage": 100
      },
      "ai_recommendations": {
        "enabled": false,
        "rollout_percentage": 0
      },
      "two_factor_auth": {
        "enabled": true,
        "rollout_percentage": 100
      }
    }

---
# 4. Secret rotation pattern
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rotate-db-password
  namespace: production
spec:
  schedule: "0 0 1 * *"  # Monthly
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator
          containers:
          - name: rotator
            image: myregistry/secret-rotator:v1
            env:
            - name: SECRET_NAME
              value: db-credentials
            - name: NAMESPACE
              value: production
            - name: DB_HOST
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: host
            # Script:
            # 1. Generate new password
            # 2. Update database password
            # 3. Update Kubernetes secret
            # 4. Rolling restart deployment
            command:
            - /bin/sh
            - -c
            - |
              NEW_PASS=$(openssl rand -base64 32)
              # Update DB password (via SQL)
              # Update K8s secret
              kubectl create secret generic db-credentials \
                --from-literal=password="$NEW_PASS" \
                --dry-run=client -o yaml | kubectl apply -f -
              # Trigger rolling restart
              kubectl rollout restart deployment/myapp -n production
          restartPolicy: OnFailure

---
# 5. ConfigMap generator with checksum (Kustomize)
# kustomization.yaml
# configMapGenerator:
# - name: app-config
#   files:
#   - config.yaml
#   - nginx.conf
#   options:
#     disableNameSuffixHash: false  # Adds hash suffix
# # Result: app-config-h5k2m (unique per content)
# # Deployment auto-updated to reference new ConfigMap name

---
# 6. Multi-source environment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: comprehensive-config
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:v1
        env:
        # From Downward API
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        # From ConfigMap
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: LOG_LEVEL
        # From Secret
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        # Static
        - name: APP_VERSION
          value: "2.0.0"
        # All ConfigMap keys
        envFrom:
        - configMapRef:
            name: feature-flags-env
            optional: true
        volumeMounts:
        - name: config-files
          mountPath: /etc/app
          readOnly: true
        - name: secrets
          mountPath: /etc/secrets
          readOnly: true
        - name: tls
          mountPath: /etc/tls
          readOnly: true
      volumes:
      - name: config-files
        configMap:
          name: app-config
      - name: secrets
        secret:
          secretName: db-credentials
          defaultMode: 0400
      - name: tls
        secret:
          secretName: app-tls
          defaultMode: 0400`,
				},
			},
		},
	})
}
