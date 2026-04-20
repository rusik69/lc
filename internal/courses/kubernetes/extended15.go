package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1733,
			Title:       "Kubernetes API and Extensibility",
			Description: "Deep dive into the Kubernetes API machinery, admission controllers, API aggregation, and extending Kubernetes with custom API servers and webhooks.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "Kubernetes API Deep Dive",
					Content: `The Kubernetes API is the foundation of the platform. Understanding its structure, versioning, and mechanics is essential for advanced usage and extension.

**API Structure:**
` + "```" + `
API URL pattern:
  /apis/{group}/{version}/namespaces/{namespace}/{resource}/{name}
  /api/v1/namespaces/{namespace}/{resource}/{name}   (core group)

Examples:
  GET  /api/v1/namespaces/default/pods            → list pods
  GET  /api/v1/namespaces/default/pods/mypod       → get pod
  POST /api/v1/namespaces/default/pods             → create pod
  PUT  /api/v1/namespaces/default/pods/mypod       → update pod
  PATCH /api/v1/namespaces/default/pods/mypod      → patch pod
  DELETE /api/v1/namespaces/default/pods/mypod     → delete pod
  
  /apis/apps/v1/namespaces/default/deployments     → deployments
  /apis/batch/v1/namespaces/default/jobs           → jobs

API Groups:
  Core (""): pods, services, configmaps, secrets, nodes, namespaces
  apps:      deployments, statefulsets, daemonsets, replicasets
  batch:     jobs, cronjobs
  networking.k8s.io: ingresses, networkpolicies
  rbac.authorization.k8s.io: roles, clusterroles, bindings
  policy:    poddisruptionbudgets
  autoscaling: horizontalpodautoscalers
  storage.k8s.io: storageclasses, csidriver, csistoragecapacities
  coordination.k8s.io: leases

API Versioning:
  Alpha (v1alpha1): unstable, disabled by default, may be removed
  Beta (v1beta1):   mostly stable, enabled by default, migration path
  GA (v1):          stable, permanent, backward compatible

Version lifecycle:
  v1alpha1 → v1beta1 → v1
  Deprecation rules (KEP):
    GA: 12 months or 3 releases (whichever longer) after deprecation
    Beta: 9 months or 3 releases
    Alpha: 0 releases (can be removed anytime)
` + "```" + `

**API Server Request Flow:**
` + "```" + `
Request → Authentication → Authorization → Admission → Validation → etcd

1. Authentication (who are you?):
   - Client certificates (x509)
   - Bearer tokens
   - OIDC tokens
   - ServiceAccount tokens
   - Webhook token review
   → Result: user info (username, groups, UID)

2. Authorization (are you allowed?):
   - RBAC (most common)
   - ABAC
   - Webhook
   - Node authorizer
   → Result: allow/deny

3. Admission control (should we modify/validate?):
   Mutating admission:
     - DefaultStorageClass (adds default storage class)
     - ServiceAccount (adds default SA token)
     - LimitRanger (applies default limits)
     - PodPreset (inject env vars, volumes)
     - Istio sidecar injection
     → Can MODIFY the request
   
   Validating admission:
     - PodSecurity (enforce PSS)
     - ResourceQuota (check limits)
     - NamespaceLifecycle (prevent ops on terminating NS)
     - OPA Gatekeeper policies
     - Kyverno policies
     → Can only ACCEPT or REJECT

   Order: Mutating webhooks → Schema validation → Validating webhooks
   
4. Validation:
   - OpenAPI schema validation
   - Field types, required fields
   - Resource-specific validation

5. Persistence:
   - Serialize to protobuf/JSON
   - Write to etcd
   - Return response to client
` + "```" + `

**Watch Mechanism:**
` + "```" + `
Watch = long-lived HTTP request for change notifications

GET /api/v1/namespaces/default/pods?watch=true

Response: server-sent events (chunked transfer encoding)
  {"type":"ADDED","object":{...}}
  {"type":"MODIFIED","object":{...}}
  {"type":"DELETED","object":{...}}
  {"type":"BOOKMARK","object":{...}}

Events are ordered by resourceVersion.

Watch types:
  ADDED:    resource created
  MODIFIED: resource updated
  DELETED:  resource deleted
  BOOKMARK: synthetic event to update resourceVersion
  ERROR:    watch error (usually 410 Gone)

ResourceVersion:
  - Opaque string (usually etcd revision number)
  - Used to resume watch after disconnect
  - List then watch pattern:
    1. GET /pods → response includes resourceVersion: "12345"
    2. GET /pods?watch=true&resourceVersion=12345
    → Gets all changes since version 12345

Watch caching:
  - API server caches recent events in memory
  - Default: 100 events per resource type
  - If client's resourceVersion is too old: 410 Gone
  - Client must re-list and restart watch

Informers (client-go):
  - Combination of list + watch
  - Local cache of resources
  - Event handlers: OnAdd, OnUpdate, OnDelete
  - Efficient: only one API connection per resource type
  - Used by all controllers and operators
` + "```" + `

**Server-Side Apply (SSA):**
` + "```" + `
SSA = field ownership tracking
  - Tracks who manages which fields
  - Enables safe multi-actor management
  - Replaces client-side apply (kubectl apply)

Benefits:
  - Conflict detection: two actors modifying same field
  - Clear ownership: who set this field?
  - Better merge behavior
  - Declarative with conflict resolution

Field managers:
  - Each actor has a unique manager name
  - kubectl: "kubectl-client-side-apply" or custom
  - Controllers: controller-specific name
  - SSA: specified in PATCH request

  kubectl apply --server-side --field-manager=my-controller

Conflict resolution:
  - force=false: reject if owned by another manager
  - force=true: take ownership of conflicting fields

# SSA is the preferred approach for GitOps and controllers
# Kubernetes 1.22+: SSA is GA
` + "```" + ``,
					CodeExamples: `# Kubernetes API Examples

# 1. Dynamic admission webhook (validating)
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: pod-validator
webhooks:
- name: pod-validator.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  timeoutSeconds: 5
  failurePolicy: Fail
  matchPolicy: Equivalent
  rules:
  - apiGroups: [""]
    apiVersions: ["v1"]
    operations: ["CREATE", "UPDATE"]
    resources: ["pods"]
    scope: Namespaced
  clientConfig:
    service:
      namespace: webhook-system
      name: pod-validator
      path: /validate
      port: 443
    caBundle: LS0t...  # Base64 encoded CA cert
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values: ["kube-system", "webhook-system"]
  objectSelector:
    matchExpressions:
    - key: skip-validation
      operator: DoesNotExist

---
# 2. Mutating webhook (e.g., inject labels)
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: pod-mutator
webhooks:
- name: pod-mutator.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  timeoutSeconds: 5
  failurePolicy: Ignore  # Don't block on failure
  reinvocationPolicy: IfNeeded
  rules:
  - apiGroups: [""]
    apiVersions: ["v1"]
    operations: ["CREATE"]
    resources: ["pods"]
    scope: Namespaced
  clientConfig:
    service:
      namespace: webhook-system
      name: pod-mutator
      path: /mutate
      port: 443
    caBundle: LS0t...

---
# 3. API aggregation layer
apiVersion: apiregistration.k8s.io/v1
kind: APIService
metadata:
  name: v1beta1.metrics.k8s.io
spec:
  service:
    name: metrics-server
    namespace: kube-system
  group: metrics.k8s.io
  version: v1beta1
  insecureSkipTLSVerify: true
  groupPriorityMinimum: 100
  versionPriority: 100

---
# 4. Custom Metrics API (for HPA)
apiVersion: apiregistration.k8s.io/v1
kind: APIService
metadata:
  name: v1beta1.custom.metrics.k8s.io
spec:
  service:
    name: prometheus-adapter
    namespace: monitoring
  group: custom.metrics.k8s.io
  version: v1beta1
  insecureSkipTLSVerify: false
  caBundle: LS0t...
  groupPriorityMinimum: 100
  versionPriority: 100

---
# 5. Webhook service and deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pod-validator
  namespace: webhook-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pod-validator
  template:
    metadata:
      labels:
        app: pod-validator
    spec:
      serviceAccountName: pod-validator
      containers:
      - name: webhook
        image: myregistry/pod-validator:v1.0.0
        ports:
        - containerPort: 8443
          name: https
        volumeMounts:
        - name: tls
          mountPath: /etc/webhook/tls
          readOnly: true
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            memory: 256Mi
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 5
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8443
            scheme: HTTPS
      volumes:
      - name: tls
        secret:
          secretName: pod-validator-tls
---
apiVersion: v1
kind: Service
metadata:
  name: pod-validator
  namespace: webhook-system
spec:
  selector:
    app: pod-validator
  ports:
  - port: 443
    targetPort: 8443
    protocol: TCP

---
# 6. cert-manager for webhook certificates
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: pod-validator-tls
  namespace: webhook-system
spec:
  secretName: pod-validator-tls
  dnsNames:
  - pod-validator.webhook-system.svc
  - pod-validator.webhook-system.svc.cluster.local
  issuerRef:
    name: webhook-ca-issuer
    kind: ClusterIssuer
  duration: 8760h    # 1 year
  renewBefore: 720h  # Renew 30 days before

---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: webhook-ca-issuer
spec:
  selfSigned: {}`,
				},
				{
					Title: "Extending Kubernetes with Admission Controllers and Finalizers",
					Content: `Admission controllers and finalizers provide powerful mechanisms to enforce policies and manage resource lifecycle in Kubernetes clusters.

**Admission Controller Webhooks:**
` + "```" + `
Admission webhooks are HTTP callbacks that receive admission requests.

AdmissionReview request:
  {
    "apiVersion": "admission.k8s.io/v1",
    "kind": "AdmissionReview",
    "request": {
      "uid": "unique-id",
      "kind": {"group":"","version":"v1","kind":"Pod"},
      "resource": {"group":"","version":"v1","resource":"pods"},
      "namespace": "default",
      "operation": "CREATE",
      "object": { ... the pod spec ... },
      "oldObject": null,  // set on UPDATE
      "userInfo": {
        "username": "admin",
        "groups": ["system:masters"]
      }
    }
  }

AdmissionReview response (validating):
  {
    "apiVersion": "admission.k8s.io/v1",
    "kind": "AdmissionReview",
    "response": {
      "uid": "same-uid-as-request",
      "allowed": true,
      // or if denied:
      "allowed": false,
      "status": {
        "code": 403,
        "message": "Pod must have resource limits"
      }
    }
  }

AdmissionReview response (mutating):
  {
    "apiVersion": "admission.k8s.io/v1",
    "kind": "AdmissionReview",
    "response": {
      "uid": "same-uid-as-request",
      "allowed": true,
      "patchType": "JSONPatch",
      "patch": "base64-encoded-json-patch"
    }
  }

JSON Patch format:
  [
    {"op":"add","path":"/metadata/labels/injected","value":"true"},
    {"op":"replace","path":"/spec/containers/0/resources/limits/memory","value":"512Mi"},
    {"op":"remove","path":"/metadata/annotations/temporary"}
  ]
` + "```" + `

**Webhook Best Practices:**
` + "```" + `
Performance:
  - Keep webhook fast (< 1 second)
  - Timeout: 5-10 seconds max
  - Multiple replicas for HA
  - failurePolicy: Fail for security, Ignore for optional

Reliability:
  - Exclude kube-system from webhook scope
  - Monitor webhook latency and errors
  - Use reinvocationPolicy: IfNeeded for mutating
  - Handle dry-run requests
  - idempotent operations

Security:
  - Use TLS (required)
  - Validate AdmissionReview version
  - Check operation type before acting
  - Don't log sensitive fields (secrets)
  - cert-manager for certificate lifecycle

Debugging:
  - Check API server logs for webhook errors
  - kubectl get mutatingwebhookconfigurations
  - kubectl get validatingwebhookconfigurations
  - Temporarily set failurePolicy: Ignore to unblock
` + "```" + `

**Finalizers:**
` + "```" + `
Finalizers = pre-delete hooks for cleanup.

How they work:
  1. Add finalizer to resource metadata.finalizers[]
  2. When resource is deleted → deletionTimestamp is set
  3. Resource enters "Terminating" state
  4. Controller performs cleanup
  5. Controller removes finalizer
  6. When all finalizers removed → resource is actually deleted

Use cases:
  - Delete external resources (cloud resources, DNS records)
  - Cascade delete to related resources
  - Backup before delete
  - Audit logging

Implementation steps:
  1. When creating: add finalizer
     metadata:
       finalizers:
       - mycontroller.example.com/cleanup
  
  2. In reconcile loop: check if deleting
     if object.DeletionTimestamp != nil {
       // Perform cleanup
       // Remove finalizer
       // Update resource
     }
  
  3. Only remove finalizer after cleanup succeeds
     → If cleanup fails, resource stays in Terminating
     → Retry until cleanup succeeds

Common issues:
  - Stuck Terminating: finalizer controller not running
  - Fix: remove finalizer manually (emergency only)
    kubectl patch resource/name --type=merge \
      -p '{"metadata":{"finalizers":null}}'
  - Prevention: monitor finalizer queue length
` + "```" + `

**Custom Controllers Pattern:**
` + "```" + `
Controller = reconciliation loop

Core pattern:
  1. Watch resources (informer)
  2. Queue changes (workqueue)
  3. Process queue items (reconcile)
  4. Update status
  5. Re-queue if needed

Reconciliation principles:
  - Level-triggered (not edge-triggered)
    - Don't react to "what changed"
    - Compute "what should be" and make it so
    - Idempotent: running twice = same result
  
  - Eventual consistency
    - May take multiple reconcile loops
    - Handle partial failures gracefully
    - Status reflects actual state

  - Owner references
    - Set ownerReferences for child resources
    - Automatic garbage collection
    - When parent deleted → children deleted

Error handling:
  - Transient: re-queue with backoff
  - Permanent: update status with error, don't re-queue
  - Rate limit: use workqueue rate limiter
  - Exponential backoff: 1s → 2s → 4s → ... → 16m (max)

Status management:
  - .status.conditions[]: standardized condition array
  - Types: Ready, Available, Progressing, Degraded
  - Status: True, False, Unknown
  - LastTransitionTime, Reason, Message
  - Update status subresource (doesn't trigger reconcile)
` + "```" + ``,
					CodeExamples: `# Admission Controller and Finalizer Examples

# 1. Validating admission policy (Kubernetes 1.26+)
# CEL-based validation (no webhook needed!)
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-non-root
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE", "UPDATE"]
      resources: ["pods"]
  validations:
  - expression: >
      object.spec.containers.all(c,
        has(c.securityContext) &&
        has(c.securityContext.runAsNonRoot) &&
        c.securityContext.runAsNonRoot == true
      )
    message: "All containers must run as non-root"
  - expression: >
      object.spec.containers.all(c,
        has(c.resources) &&
        has(c.resources.requests) &&
        has(c.resources.requests.memory) &&
        has(c.resources.requests.cpu)
      )
    message: "All containers must have resource requests"

---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: require-non-root-binding
spec:
  policyName: require-non-root
  validationActions:
  - Deny
  matchResources:
    namespaceSelector:
      matchLabels:
        enforce-security: "true"

---
# 2. Conversion webhook (for CRD version upgrade)
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  names:
    kind: Database
    listKind: DatabaseList
    plural: databases
    singular: database
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              engine:
                type: string
                enum: ["postgres", "mysql", "mariadb"]
              version:
                type: string
              replicas:
                type: integer
                minimum: 1
                maximum: 10
              storage:
                type: object
                properties:
                  size:
                    type: string
                  storageClass:
                    type: string
            required: ["engine", "version", "replicas"]
          status:
            type: object
            properties:
              phase:
                type: string
              conditions:
                type: array
                items:
                  type: object
                  properties:
                    type:
                      type: string
                    status:
                      type: string
                    lastTransitionTime:
                      type: string
                      format: date-time
                    reason:
                      type: string
                    message:
                      type: string
    subresources:
      status: {}
    additionalPrinterColumns:
    - name: Engine
      type: string
      jsonPath: .spec.engine
    - name: Version
      type: string
      jsonPath: .spec.version
    - name: Replicas
      type: integer
      jsonPath: .spec.replicas
    - name: Phase
      type: string
      jsonPath: .status.phase
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
  - name: v1alpha1
    served: true
    storage: false
    deprecated: true
    deprecationWarning: "example.com/v1alpha1 Database is deprecated; use v1"
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              engine:
                type: string
              replicas:
                type: integer
  conversion:
    strategy: Webhook
    webhook:
      conversionReviewVersions: ["v1"]
      clientConfig:
        service:
          namespace: database-operator
          name: database-webhook
          path: /convert

---
# 3. Custom resource with finalizer
apiVersion: example.com/v1
kind: Database
metadata:
  name: mydb
  namespace: production
  finalizers:
  - databases.example.com/cleanup
spec:
  engine: postgres
  version: "15.4"
  replicas: 3
  storage:
    size: 100Gi
    storageClass: fast-ssd

---
# 4. Leader election for controller HA
apiVersion: coordination.k8s.io/v1
kind: Lease
metadata:
  name: database-controller
  namespace: database-operator
# Lease is automatically managed by controller-runtime
# leader election. Controller pods compete for the lease.
# Only the leader runs reconciliation.

---
# 5. RBAC for controller
apiVersion: v1
kind: ServiceAccount
metadata:
  name: database-controller
  namespace: database-operator

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: database-controller
rules:
# CRD management
- apiGroups: ["example.com"]
  resources: ["databases"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["example.com"]
  resources: ["databases/status"]
  verbs: ["get", "update", "patch"]
- apiGroups: ["example.com"]
  resources: ["databases/finalizers"]
  verbs: ["update"]
# Managed resources
- apiGroups: ["apps"]
  resources: ["statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["services", "configmaps", "secrets", "persistentvolumeclaims"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch"]
# Leader election
- apiGroups: ["coordination.k8s.io"]
  resources: ["leases"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: database-controller
subjects:
- kind: ServiceAccount
  name: database-controller
  namespace: database-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: database-controller`,
				},
			},
		},
	})
}
