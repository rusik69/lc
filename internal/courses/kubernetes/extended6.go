package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1724,
			Title:       "Custom Resources and Operators",
			Description: "Master Kubernetes extensibility: Custom Resource Definitions (CRDs), controllers, operator pattern, and operator frameworks.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Custom Resource Definitions",
					Content: `CRDs extend the Kubernetes API with custom resource types. They're the foundation for operators and platform extensions.

**CRD Basics:**
` + "```" + `
What CRDs provide:
  - New API endpoint: /apis/<group>/<version>/<resource>
  - CRUD operations via kubectl
  - Watch/list with labels and field selectors
  - Schema validation (OpenAPI v3)
  - Status subresource
  - Scale subresource (for HPA)
  - Versioning with conversion webhooks

Use cases:
  - Database provisioning (PostgreSQL, MySQL, Redis)
  - Certificate management (cert-manager)
  - CI/CD pipelines (Tekton)
  - Infrastructure (Crossplane)
  - Networking (Istio VirtualService)
  - Security policies (Kyverno, OPA)
` + "```" + `

**CRD Definition:**
` + "```" + `yaml
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
    shortNames:
    - db
    categories:
    - all
  scope: Namespaced
  versions:
  - name: v1alpha1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        required: ["spec"]
        properties:
          spec:
            type: object
            required: ["engine", "version", "storage"]
            properties:
              engine:
                type: string
                enum: ["postgresql", "mysql", "redis"]
              version:
                type: string
                pattern: "^[0-9]+\\.[0-9]+$"
              replicas:
                type: integer
                minimum: 1
                maximum: 10
                default: 1
              storage:
                type: object
                required: ["size"]
                properties:
                  size:
                    type: string
                    pattern: "^[0-9]+(Gi|Ti)$"
                  storageClassName:
                    type: string
              resources:
                type: object
                properties:
                  cpu:
                    type: string
                  memory:
                    type: string
              backup:
                type: object
                properties:
                  enabled:
                    type: boolean
                    default: false
                  schedule:
                    type: string
                  retention:
                    type: string
                    default: "7d"
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Pending", "Creating", "Running", "Failed", "Deleting"]
              ready:
                type: boolean
              endpoint:
                type: string
              replicas:
                type: integer
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
      scale:
        specReplicasPath: .spec.replicas
        statusReplicasPath: .status.replicas
    additionalPrinterColumns:
    - name: Engine
      type: string
      jsonPath: .spec.engine
    - name: Version
      type: string
      jsonPath: .spec.version
    - name: Phase
      type: string
      jsonPath: .status.phase
    - name: Ready
      type: boolean
      jsonPath: .status.ready
    - name: Endpoint
      type: string
      jsonPath: .status.endpoint
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
` + "```" + `

**Custom Resource Instance:**
` + "```" + `yaml
apiVersion: example.com/v1alpha1
kind: Database
metadata:
  name: orders-db
  namespace: production
spec:
  engine: postgresql
  version: "16.1"
  replicas: 3
  storage:
    size: 100Gi
    storageClassName: fast-ssd
  resources:
    cpu: "2"
    memory: 4Gi
  backup:
    enabled: true
    schedule: "0 2 * * *"
    retention: "30d"
` + "```" + `

**kubectl Usage:**
` + "```" + `
# After applying CRD:
kubectl get databases -n production
kubectl get db -n production          # Short name
kubectl describe db orders-db -n production
kubectl delete db orders-db -n production

# Scale (with scale subresource)
kubectl scale db orders-db --replicas=5

# Output:
NAME       ENGINE      VERSION  PHASE    READY  ENDPOINT                  AGE
orders-db  postgresql  16.1     Running  true   orders-db.production:5432  2d
` + "```" + `

**CRD Versioning:**
` + "```" + `yaml
# Multiple versions with conversion
spec:
  versions:
  - name: v1alpha1
    served: true
    storage: false   # Not stored, converted from v1beta1
  - name: v1beta1
    served: true
    storage: true    # Stored version
  conversion:
    strategy: Webhook
    webhook:
      conversionReviewVersions: ["v1"]
      clientConfig:
        service:
          name: db-operator-webhook
          namespace: system
          path: /convert
` + "```" + ``,
					CodeExamples: `# CRD Examples

# 1. Certificate CRD (cert-manager style)
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: certificates.tls.example.com
spec:
  group: tls.example.com
  names:
    kind: Certificate
    plural: certificates
    singular: certificate
    shortNames: [cert]
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        required: ["spec"]
        properties:
          spec:
            type: object
            required: ["dnsNames", "issuerRef"]
            properties:
              secretName:
                type: string
              dnsNames:
                type: array
                items:
                  type: string
              issuerRef:
                type: object
                required: ["name", "kind"]
                properties:
                  name:
                    type: string
                  kind:
                    type: string
                    enum: ["Issuer", "ClusterIssuer"]
              duration:
                type: string
                default: "2160h"
              renewBefore:
                type: string
                default: "360h"
          status:
            type: object
            properties:
              ready:
                type: boolean
              expirationDate:
                type: string
              renewalTime:
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
                    reason:
                      type: string
                    message:
                      type: string
    subresources:
      status: {}
    additionalPrinterColumns:
    - name: Ready
      type: boolean
      jsonPath: .status.ready
    - name: Expiration
      type: string
      jsonPath: .status.expirationDate
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
---
# Usage
apiVersion: tls.example.com/v1
kind: Certificate
metadata:
  name: web-cert
  namespace: production
spec:
  secretName: web-tls
  dnsNames:
  - "app.example.com"
  - "*.app.example.com"
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  duration: "2160h"
  renewBefore: "720h"

---
# 2. Queue CRD
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: queues.messaging.example.com
spec:
  group: messaging.example.com
  names:
    kind: Queue
    plural: queues
    singular: queue
    shortNames: [q]
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
            required: ["type"]
            properties:
              type:
                type: string
                enum: ["standard", "fifo", "priority"]
              maxSize:
                type: integer
                default: 10000
              retentionPeriod:
                type: string
                default: "24h"
              deadLetterQueue:
                type: object
                properties:
                  enabled:
                    type: boolean
                    default: true
                  maxReceiveCount:
                    type: integer
                    default: 3
          status:
            type: object
            properties:
              messageCount:
                type: integer
              consumerCount:
                type: integer
              endpoint:
                type: string
              phase:
                type: string
    subresources:
      status: {}
    additionalPrinterColumns:
    - name: Type
      type: string
      jsonPath: .spec.type
    - name: Messages
      type: integer
      jsonPath: .status.messageCount
    - name: Consumers
      type: integer
      jsonPath: .status.consumerCount
    - name: Phase
      type: string
      jsonPath: .status.phase`,
				},
				{
					Title: "Operator Pattern and Frameworks",
					Content: `Operators are Kubernetes controllers that encode operational knowledge for managing complex applications. They watch CRDs and reconcile actual state to desired state.

**Operator Concepts:**
` + "```" + `
Control loop:
  1. Watch: Observe custom resources (and related resources)
  2. Analyze: Compare desired state (spec) vs actual state
  3. Act: Create/update/delete Kubernetes resources
  4. Report: Update custom resource status

Operator maturity levels (OperatorHub):
  Level 1: Basic Install     → automated installation
  Level 2: Seamless Upgrades → automated upgrades, version migration
  Level 3: Full Lifecycle    → backup, restore, failure recovery
  Level 4: Deep Insights     → metrics, alerts, log analysis
  Level 5: Auto Pilot        → auto-scaling, auto-tuning, anomaly detection

Frameworks:
  Kubebuilder      → most popular, Kubernetes SIG project
  Operator SDK     → Red Hat, wraps Kubebuilder + Ansible/Helm
  controller-runtime → underlying library used by both
  KUDO            → declarative operators
  Metacontroller  → webhook-based lightweight controllers
` + "```" + `

**Reconciliation Pattern (controller-runtime):**
` + "```" + `
// Reconcile function — called for every create/update/delete event
func (r *DatabaseReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    // 1. Fetch the custom resource
    var db examplev1.Database
    if err := r.Get(ctx, req.NamespacedName, &db); err != nil {
        if apierrors.IsNotFound(err) {
            return ctrl.Result{}, nil // Deleted, nothing to do
        }
        return ctrl.Result{}, err
    }

    // 2. Handle deletion (finalizers)
    if !db.DeletionTimestamp.IsZero() {
        return r.handleDeletion(ctx, &db)
    }

    // 3. Ensure finalizer
    if !controllerutil.ContainsFinalizer(&db, finalizerName) {
        controllerutil.AddFinalizer(&db, finalizerName)
        if err := r.Update(ctx, &db); err != nil {
            return ctrl.Result{}, err
        }
    }

    // 4. Reconcile child resources
    // Create StatefulSet
    if err := r.reconcileStatefulSet(ctx, &db); err != nil {
        return ctrl.Result{}, err
    }

    // Create Service
    if err := r.reconcileService(ctx, &db); err != nil {
        return ctrl.Result{}, err
    }

    // Create ConfigMap
    if err := r.reconcileConfigMap(ctx, &db); err != nil {
        return ctrl.Result{}, err
    }

    // 5. Update status
    db.Status.Phase = "Running"
    db.Status.Ready = true
    if err := r.Status().Update(ctx, &db); err != nil {
        return ctrl.Result{}, err
    }

    // 6. Requeue after interval (for periodic checks)
    return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil
}

// Setup — which resources to watch
func (r *DatabaseReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&examplev1.Database{}).             // Watch our CRD
        Owns(&appsv1.StatefulSet{}).            // Watch owned StatefulSets
        Owns(&corev1.Service{}).                // Watch owned Services
        Owns(&corev1.ConfigMap{}).              // Watch owned ConfigMaps
        WithOptions(controller.Options{
            MaxConcurrentReconciles: 3,
        }).
        Complete(r)
}
` + "```" + `

**Owner References:**
` + "```" + `
When a controller creates child resources, it sets ownerReferences:
  - Child is garbage collected when parent is deleted
  - Changes to child trigger reconciliation of parent

controllerutil.SetControllerReference(&db, &sts, r.Scheme)
// Sets:
// ownerReferences:
// - apiVersion: example.com/v1alpha1
//   kind: Database
//   name: orders-db
//   uid: xxx
//   controller: true
//   blockOwnerDeletion: true
` + "```" + `

**Finalizers:**
` + "```" + `
Finalizers prevent resource deletion until cleanup is done:

Flow:
  1. User deletes resource (kubectl delete db orders-db)
  2. Kubernetes sets deletionTimestamp (but doesn't delete)
  3. Controller sees deletionTimestamp, performs cleanup
  4. Controller removes finalizer
  5. Kubernetes deletes the resource

Use cases:
  - Delete external resources (cloud databases, DNS records)
  - Graceful shutdown of stateful workloads
  - Cleanup of non-owned Kubernetes resources
  
Implementation:
  // Add finalizer on create
  controllerutil.AddFinalizer(&db, "example.com/cleanup")
  r.Update(ctx, &db)
  
  // On delete (deletionTimestamp set):
  if controllerutil.ContainsFinalizer(&db, "example.com/cleanup") {
      // Perform cleanup
      if err := r.deleteExternalDB(ctx, &db); err != nil {
          return ctrl.Result{}, err // Retry
      }
      // Remove finalizer
      controllerutil.RemoveFinalizer(&db, "example.com/cleanup")
      r.Update(ctx, &db)
  }
` + "```" + `

**Webhooks:**
` + "```" + `
Admission webhooks validate or mutate resources before persistence:

Validating webhook:
  - Reject invalid configurations
  - Enforce business rules
  - Read-only (can't modify)

Mutating webhook:
  - Set defaults
  - Inject sidecars
  - Add labels/annotations
  - Runs BEFORE validating webhooks

Implementation (Kubebuilder):
  // +kubebuilder:webhook:path=/mutate-v1-database,mutating=true,sideEffects=None
  func (r *Database) Default() {
      if r.Spec.Replicas == 0 {
          r.Spec.Replicas = 1
      }
      if r.Spec.Resources.Memory == "" {
          r.Spec.Resources.Memory = "1Gi"
      }
  }
  
  // +kubebuilder:webhook:path=/validate-v1-database,mutating=false,sideEffects=None
  func (r *Database) ValidateCreate() (admission.Warnings, error) {
      if r.Spec.Engine == "postgresql" && r.Spec.Replicas%2 == 0 {
          return nil, field.Invalid(
              field.NewPath("spec").Child("replicas"),
              r.Spec.Replicas,
              "PostgreSQL requires odd number of replicas for quorum",
          )
      }
      return nil, nil
  }
` + "```" + ``,
					CodeExamples: `# Operator Deployment

# 1. Operator Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: database-operator
  namespace: database-system
  labels:
    app: database-operator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: database-operator
  template:
    metadata:
      labels:
        app: database-operator
    spec:
      serviceAccountName: database-operator
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: manager
        image: myregistry/database-operator:v1.0.0
        args:
        - --leader-elect
        - --metrics-bind-address=:8080
        - --health-probe-bind-address=:8081
        ports:
        - containerPort: 8080
          name: metrics
        - containerPort: 8081
          name: health
        - containerPort: 9443
          name: webhook
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        resources:
          limits:
            cpu: 500m
            memory: 256Mi
          requests:
            cpu: 100m
            memory: 128Mi
        livenessProbe:
          httpGet:
            path: /healthz
            port: health
          initialDelaySeconds: 15
        readinessProbe:
          httpGet:
            path: /readyz
            port: health
          initialDelaySeconds: 5
        volumeMounts:
        - name: webhook-certs
          mountPath: /tmp/k8s-webhook-server/serving-certs
          readOnly: true
      volumes:
      - name: webhook-certs
        secret:
          secretName: database-operator-webhook-cert

---
# 2. Operator RBAC
apiVersion: v1
kind: ServiceAccount
metadata:
  name: database-operator
  namespace: database-system

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: database-operator-manager
rules:
# Custom resources
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
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
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
  name: database-operator-manager
subjects:
- kind: ServiceAccount
  name: database-operator
  namespace: database-system
roleRef:
  kind: ClusterRole
  name: database-operator-manager
  apiGroup: rbac.authorization.k8s.io

---
# 3. Webhook Configuration
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: database-validating
webhooks:
- name: vdatabase.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  clientConfig:
    service:
      name: database-operator-webhook
      namespace: database-system
      path: /validate-example-com-v1alpha1-database
  rules:
  - apiGroups: ["example.com"]
    apiVersions: ["v1alpha1"]
    operations: ["CREATE", "UPDATE"]
    resources: ["databases"]
  failurePolicy: Fail

---
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: database-mutating
webhooks:
- name: mdatabase.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  clientConfig:
    service:
      name: database-operator-webhook
      namespace: database-system
      path: /mutate-example-com-v1alpha1-database
  rules:
  - apiGroups: ["example.com"]
    apiVersions: ["v1alpha1"]
    operations: ["CREATE", "UPDATE"]
    resources: ["databases"]
  failurePolicy: Fail

---
# 4. ServiceMonitor for operator metrics
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: database-operator
  namespace: database-system
spec:
  selector:
    matchLabels:
      app: database-operator
  endpoints:
  - port: metrics
    interval: 30s`,
				},
			},
		},
	})
}
