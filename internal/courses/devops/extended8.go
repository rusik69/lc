package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1464,
			Title:       "Container Security and Supply Chain",
			Description: "Implement container security best practices including image hardening, vulnerability scanning, runtime security, and software supply chain integrity.",
			Order:       64,
			Lessons: []problems.Lesson{
				{
					Title: "Container Image Security",
					Content: `Container security starts with building secure images and establishing a trusted supply chain for container artifacts.

**Dockerfile Security Best Practices:**
` + "```" + `
Base image selection:
  Use minimal base images:
    Distroless:  gcr.io/distroless/static-debian12
    Alpine:      alpine:3.19
    Scratch:     scratch (for static binaries)
    Chainguard:  cgr.dev/chainguard/static
  
  Avoid:
    ubuntu:latest (large, many CVEs)
    :latest tags (unpinned)
    Unknown registries

Multi-stage builds:
  # Build stage
  FROM golang:1.22-alpine AS builder
  WORKDIR /app
  COPY go.mod go.sum ./
  RUN go mod download
  COPY . .
  RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /app/server ./cmd/server
  
  # Runtime stage
  FROM gcr.io/distroless/static-debian12:nonroot
  COPY --from=builder /app/server /server
  USER nonroot:nonroot
  EXPOSE 8080
  ENTRYPOINT ["/server"]

Security hardening:
  # Don't run as root
  RUN addgroup -S appgroup && adduser -S appuser -G appgroup
  USER appuser
  
  # Read-only filesystem
  RUN chmod 555 /app/server
  
  # No shell
  FROM scratch
  COPY --from=builder /app/server /server
  ENTRYPOINT ["/server"]
  
  # Minimize layers
  RUN apt-get update && \
      apt-get install -y --no-install-recommends package && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/*
  
  # Use .dockerignore
  .git
  .env
  *.md
  test/
  node_modules/
  
  # Pin versions
  FROM node:20.11.1-alpine3.19
  RUN npm ci --only=production
  
  # No secrets in image
  # Bad: COPY .env /app/.env
  # Bad: ENV DB_PASSWORD=secret123
  # Good: Read from environment at runtime
  
  # HEALTHCHECK
  HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1

Kubernetes security context:
  securityContext:
    runAsNonRoot: true
    runAsUser: 65534
    readOnlyRootFilesystem: true
    allowPrivilegeEscalation: false
    capabilities:
      drop: ["ALL"]
    seccompProfile:
      type: RuntimeDefault
  
  # Pod security standards
  apiVersion: v1
  kind: Namespace
  metadata:
    labels:
      pod-security.kubernetes.io/enforce: restricted
      pod-security.kubernetes.io/audit: restricted
      pod-security.kubernetes.io/warn: restricted
` + "```" + `

**Image Scanning and Signing:**
` + "```" + `
Vulnerability scanning tools:
  Trivy (Aqua Security):
    trivy image myapp:1.0.0
    trivy image --severity HIGH,CRITICAL myapp:1.0.0
    trivy image --format json myapp:1.0.0
    trivy fs --security-checks vuln,config .
    trivy k8s --report summary cluster
  
  Grype (Anchore):
    grype myapp:1.0.0
    grype myapp:1.0.0 --only-fixed
    grype myapp:1.0.0 --fail-on critical
  
  Docker Scout:
    docker scout cves myapp:1.0.0
    docker scout recommendations myapp:1.0.0
  
  Snyk:
    snyk container test myapp:1.0.0
    snyk container monitor myapp:1.0.0

CI/CD integration:
  # GitHub Actions
  - name: Scan image
    uses: aquasecurity/trivy-action@master
    with:
      image-ref: myregistry/myapp:${{ github.sha }}
      format: 'sarif'
      output: 'trivy-results.sarif'
      severity: 'CRITICAL,HIGH'
      exit-code: '1'
  
  - name: Upload results
    uses: github/codeql-action/upload-sarif@v3
    with:
      sarif_file: 'trivy-results.sarif'

Image signing with cosign:
  # Generate key pair
  cosign generate-key-pair
  
  # Sign image
  cosign sign --key cosign.key myregistry/myapp:1.0.0
  
  # Verify signature
  cosign verify --key cosign.pub myregistry/myapp:1.0.0
  
  # Keyless signing (with OIDC)
  cosign sign myregistry/myapp:1.0.0  # Uses Sigstore/Fulcio
  cosign verify --certificate-identity user@example.com \
    --certificate-oidc-issuer https://accounts.google.com \
    myregistry/myapp:1.0.0

Admission policies:
  # Kyverno: Allow only signed images
  apiVersion: kyverno.io/v1
  kind: ClusterPolicy
  metadata:
    name: verify-image
  spec:
    validationFailureAction: Enforce
    rules:
      - name: verify-signature
        match:
          any:
            - resources:
                kinds: [Pod]
        verifyImages:
          - imageReferences: ["myregistry/*"]
            attestors:
              - entries:
                  - keys:
                      publicKeys: |-
                        -----BEGIN PUBLIC KEY-----
                        MFkwEwYHKoZIzj0...
                        -----END PUBLIC KEY-----
  
  # OPA/Gatekeeper: Block privileged containers
  apiVersion: constraints.gatekeeper.sh/v1beta1
  kind: K8sPSPPrivilegedContainer
  metadata:
    name: no-privileged
  spec:
    match:
      kinds:
        - apiGroups: [""]
          kinds: ["Pod"]
    parameters:
      exemptImages: ["gcr.io/kube-system/*"]

SBOM (Software Bill of Materials):
  Generate:
    syft myregistry/myapp:1.0.0 -o spdx-json > sbom.json
    trivy image --format spdx-json myapp:1.0.0 > sbom.json
    docker sbom myapp:1.0.0
  
  Attach to image:
    cosign attach sbom --sbom sbom.json myregistry/myapp:1.0.0
  
  Scan SBOM for vulnerabilities:
    grype sbom:sbom.json
    trivy sbom sbom.json
` + "```" + ``,
					CodeExamples: `# Container security scripts

# 1. Container image security audit
#!/bin/bash
echo "=== Container Image Security Audit ==="

# Scan all running images
echo "--- Scanning Running Images ---"
IMAGES=$(kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[].spec.containers[].image' | sort -u)

for image in $IMAGES; do
    echo ""
    echo "Image: $image"
    
    if command -v trivy &>/dev/null; then
        CRITICAL=$(trivy image --quiet --severity CRITICAL "$image" 2>/dev/null | grep -c "CRITICAL" || echo "0")
        HIGH=$(trivy image --quiet --severity HIGH "$image" 2>/dev/null | grep -c "HIGH" || echo "0")
        echo "  Critical: $CRITICAL, High: $HIGH"
    fi
done

# Check for latest tags
echo ""
echo "--- Images Using :latest Tag ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.containers[].image | test(":latest$") or (test(":") | not)) |
    "\(.metadata.namespace)/\(.metadata.name): \(.spec.containers[].image)"' 2>/dev/null

# Check for root containers
echo ""
echo "--- Containers Running as Root ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(
        .spec.securityContext.runAsNonRoot != true and
        (.spec.containers[].securityContext.runAsNonRoot != true)
    ) | "\(.metadata.namespace)/\(.metadata.name)"' 2>/dev/null | head -10

# 2. Kubernetes security posture
#!/bin/bash
echo "=== Kubernetes Security Posture ==="

# Pod Security Standards
echo "--- Namespace Security Labels ---"
kubectl get namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.metadata.labels | keys[] | test("pod-security")) |
    "\(.metadata.name): \(.metadata.labels | to_entries[] | select(.key | test("pod-security")) | "\(.key)=\(.value)")"' 2>/dev/null

# Privileged pods
echo ""
echo "--- Privileged Pods ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.containers[].securityContext.privileged == true) |
    "\(.metadata.namespace)/\(.metadata.name)"' 2>/dev/null

# Host network pods
echo ""
echo "--- Host Network Pods ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.hostNetwork == true) |
    "\(.metadata.namespace)/\(.metadata.name)"' 2>/dev/null

# Containers without resource limits
echo ""
echo "--- Containers Without Limits ---"
kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[] | select(.spec.containers[].resources.limits == null) |
    "\(.metadata.namespace)/\(.metadata.name)"' 2>/dev/null | head -10

# 3. Image signing verifier
#!/bin/bash
echo "=== Image Signature Verification ==="

if ! command -v cosign &>/dev/null; then
    echo "cosign not installed"
    exit 1
fi

KEY="${1}"
if [ -z "$KEY" ]; then
    echo "Usage: $0 <public-key-file>"
    exit 1
fi

echo "Verifying running images..."
IMAGES=$(kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[].spec.containers[].image' | sort -u)

for image in $IMAGES; do
    RESULT=$(cosign verify --key "$KEY" "$image" 2>&1)
    if [ $? -eq 0 ]; then
        echo "  [SIGNED]   $image"
    else
        echo "  [UNSIGNED] $image"
    fi
done`,
				},
				{
					Title: "Runtime Security and Network Policies",
					Content: `Runtime security protects containers during execution with monitoring, network policies, and threat detection.

**Kubernetes Network Policies:**
` + "```" + `
Network policy basics:
  Default: All pods can communicate with all pods.
  NetworkPolicy: Restrict traffic at L3/L4.
  
  Requires CNI plugin:
    Calico, Cilium, Weave Net, Kube-router

Default deny all ingress:
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata:
    name: default-deny-ingress
    namespace: production
  spec:
    podSelector: {}
    policyTypes:
      - Ingress

Default deny all egress:
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata:
    name: default-deny-egress
    namespace: production
  spec:
    podSelector: {}
    policyTypes:
      - Egress

Allow specific traffic:
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata:
    name: allow-frontend-to-backend
    namespace: production
  spec:
    podSelector:
      matchLabels:
        app: backend
    policyTypes:
      - Ingress
    ingress:
      - from:
          - podSelector:
              matchLabels:
                app: frontend
        ports:
          - protocol: TCP
            port: 8080

Complex policy:
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata:
    name: api-policy
    namespace: production
  spec:
    podSelector:
      matchLabels:
        app: api
    policyTypes:
      - Ingress
      - Egress
    ingress:
      - from:
          - namespaceSelector:
              matchLabels:
                name: ingress-nginx
          - podSelector:
              matchLabels:
                app: frontend
        ports:
          - protocol: TCP
            port: 8080
    egress:
      - to:
          - podSelector:
              matchLabels:
                app: database
        ports:
          - protocol: TCP
            port: 5432
      - to:
          - namespaceSelector: {}
        ports:
          - protocol: UDP
            port: 53  # DNS
      - to:
          - ipBlock:
              cidr: 10.0.0.0/8  # Internal services

Cilium network policies (L7):
  apiVersion: cilium.io/v2
  kind: CiliumNetworkPolicy
  metadata:
    name: l7-policy
  spec:
    endpointSelector:
      matchLabels:
        app: api
    ingress:
      - fromEndpoints:
          - matchLabels:
              app: frontend
        toPorts:
          - ports:
              - port: "8080"
                protocol: TCP
            rules:
              http:
                - method: "GET"
                  path: "/api/v1/.*"
                - method: "POST"
                  path: "/api/v1/orders"
` + "```" + `

**Runtime Security:**
` + "```" + `
Falco (runtime threat detection):
  Monitors syscalls at kernel level.
  Detects unexpected behavior.
  
  Rules:
    - rule: Terminal shell in container
      desc: Detect shell spawning in container
      condition: >
        spawned_process and container and shell_procs
        and not user_expected_terminal_shell_in_container
      output: >
        Shell spawned in container
        (user=%user.name container=%container.name
         shell=%proc.name parent=%proc.pname)
      priority: WARNING
    
    - rule: Read sensitive file
      desc: Detect reading of sensitive files
      condition: >
        open_read and container and
        (fd.name startswith /etc/shadow or
         fd.name startswith /etc/passwd)
      output: >
        Sensitive file opened
        (file=%fd.name user=%user.name container=%container.name)
      priority: WARNING
    
    - rule: Write below /etc
      desc: Unexpected write to /etc
      condition: >
        write_etc_common and container and
        not user_known_write_etc_programs
      output: >
        File written below /etc
        (file=%fd.name user=%user.name)
      priority: ERROR

  Install:
    helm install falco falcosecurity/falco \
      --namespace falco --create-namespace \
      --set tty=true

Seccomp profiles:
  apiVersion: v1
  kind: Pod
  spec:
    securityContext:
      seccompProfile:
        type: RuntimeDefault  # or Localhost
   
  # Custom profile
  {
    "defaultAction": "SCMP_ACT_ERRNO",
    "architectures": ["SCMP_ARCH_X86_64"],
    "syscalls": [
      {
        "names": ["read", "write", "open", "close",
                  "stat", "fstat", "mmap", "brk",
                  "rt_sigaction", "access", "getpid",
                  "socket", "connect", "accept",
                  "sendto", "recvfrom", "bind",
                  "listen", "epoll_create", "epoll_wait",
                  "epoll_ctl", "futex", "exit_group"],
        "action": "SCMP_ACT_ALLOW"
      }
    ]
  }

AppArmor profiles:
  annotations:
    container.apparmor.security.beta.kubernetes.io/mycontainer: runtime/default
  
  # Custom:
  container.apparmor.security.beta.kubernetes.io/mycontainer: localhost/my-profile

Policy engines:
  Kyverno:
    - Kubernetes-native policy engine
    - Validate, mutate, generate resources
    - No new language (YAML)
  
  OPA/Gatekeeper:
    - General-purpose policy engine
    - Rego language
    - Constraint templates
  
  Kubewarden:
    - WebAssembly-based policies
    - Write in any language that compiles to WASM

Secrets management:
  External Secrets Operator:
    apiVersion: external-secrets.io/v1beta1
    kind: ExternalSecret
    metadata:
      name: db-credentials
    spec:
      refreshInterval: 1h
      secretStoreRef:
        kind: SecretStore
        name: vault-backend
      target:
        name: db-credentials
      data:
        - secretKey: password
          remoteRef:
            key: secret/data/database
            property: password
  
  Sealed Secrets:
    # Encrypt secret
    kubeseal --format=yaml < secret.yaml > sealed-secret.yaml
    
    # Only controller can decrypt
    kubectl apply -f sealed-secret.yaml
` + "```" + ``,
					CodeExamples: `# Runtime security scripts

# 1. Network policy audit
#!/bin/bash
echo "=== Network Policy Audit ==="

# Namespaces without network policies
echo "--- Namespaces Without Network Policies ---"
for ns in $(kubectl get ns -o name 2>/dev/null | cut -d'/' -f2); do
    NP_COUNT=$(kubectl get networkpolicies -n "$ns" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NP_COUNT" = "0" ]; then
        POD_COUNT=$(kubectl get pods -n "$ns" --no-headers 2>/dev/null | wc -l | tr -d ' ')
        if [ "$POD_COUNT" -gt 0 ]; then
            echo "  $ns ($POD_COUNT pods, no policies)"
        fi
    fi
done

# List all network policies
echo ""
echo "--- Network Policies ---"
kubectl get networkpolicies --all-namespaces 2>/dev/null

# Default deny check
echo ""
echo "--- Namespaces with Default Deny ---"
for ns in $(kubectl get ns -o name 2>/dev/null | cut -d'/' -f2); do
    DENY=$(kubectl get networkpolicies -n "$ns" -o json 2>/dev/null | \
        jq -r '.items[] | select(.spec.podSelector == {} or .spec.podSelector.matchLabels == null) | .metadata.name' 2>/dev/null)
    if [ -n "$DENY" ]; then
        echo "  $ns: $DENY"
    fi
done

# 2. Falco alert analyzer
#!/bin/bash
echo "=== Falco Security Alerts ==="

# Recent alerts from Falco
echo "--- Recent Alerts ---"
kubectl logs -n falco -l app.kubernetes.io/name=falco --tail=50 2>/dev/null | \
    grep -E "Warning|Error|Critical" | tail -20

# Alert summary
echo ""
echo "--- Alert Summary ---"
kubectl logs -n falco -l app.kubernetes.io/name=falco --tail=1000 2>/dev/null | \
    grep -oP '(?<=rule=)[^)]+' | sort | uniq -c | sort -rn | head -10

# 3. Security compliance checker
#!/bin/bash
echo "=== Security Compliance Check ==="

PASS=0
FAIL=0
WARN=0

check() {
    local name="$1" result="$2"
    case "$result" in
        PASS) ((PASS++)); echo "  [PASS] $name" ;;
        FAIL) ((FAIL++)); echo "  [FAIL] $name" ;;
        WARN) ((WARN++)); echo "  [WARN] $name" ;;
    esac
}

# Check PSA labels
PSA=$(kubectl get ns -l 'pod-security.kubernetes.io/enforce' --no-headers 2>/dev/null | wc -l | tr -d ' ')
check "Pod Security Admission labels" "$([ "$PSA" -gt 0 ] && echo PASS || echo WARN)"

# Check network policies exist
NP=$(kubectl get networkpolicies --all-namespaces --no-headers 2>/dev/null | wc -l | tr -d ' ')
check "Network policies configured" "$([ "$NP" -gt 0 ] && echo PASS || echo FAIL)"

# Check RBAC enabled
RBAC=$(kubectl api-versions 2>/dev/null | grep "rbac.authorization.k8s.io")
check "RBAC enabled" "$([ -n "$RBAC" ] && echo PASS || echo FAIL)"

# Check no default service account tokens
DEFAULT_SA=$(kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq '[.items[] | select(.spec.serviceAccountName == "default" and .spec.automountServiceAccountToken != false)] | length' 2>/dev/null)
check "No default SA token mounts" "$([ "$DEFAULT_SA" = "0" ] && echo PASS || echo WARN)"

echo ""
echo "Results: PASS=$PASS FAIL=$FAIL WARN=$WARN"`,
				},
			},
		},
	})
}
