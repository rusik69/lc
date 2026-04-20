package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1725,
			Title:       "CI/CD and GitOps",
			Description: "Master Kubernetes deployment automation: GitOps with Flux and ArgoCD, progressive delivery, and CI/CD pipeline integration.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "GitOps with Flux and ArgoCD",
					Content: `GitOps uses Git as the single source of truth for infrastructure and application deployment. Changes are made via pull requests and automatically reconciled to the cluster.

**GitOps Principles:**
` + "```" + `
Four principles:
  1. Declarative:    Desired state is expressed declaratively
  2. Versioned:      State is stored in Git (versioned, auditable)
  3. Automated:      Approved changes auto-applied to cluster
  4. Reconciled:     Agents ensure cluster matches Git state

Benefits:
  ✓ Audit trail: every change is a Git commit
  ✓ Rollback: git revert = cluster rollback
  ✓ Security: no direct kubectl access needed
  ✓ Consistency: cluster always matches Git
  ✓ Developer experience: PRs for infrastructure

Two approaches:
  Push-based: CI pipeline pushes to cluster (traditional)
    GitHub Actions → kubectl apply → cluster
    
  Pull-based: Agent in cluster pulls from Git (GitOps)
    Git repo ← Agent polls → reconciles cluster
    More secure (no external cluster access needed)
` + "```" + `

**Flux CD:**
` + "```" + `yaml
# Bootstrap Flux
# flux bootstrap github \
#   --owner=myorg --repository=fleet \
#   --branch=main --path=clusters/production

# GitRepository source
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: app-repo
  namespace: flux-system
spec:
  interval: 1m  # Poll interval
  url: https://github.com/myorg/app-manifests
  ref:
    branch: main
  secretRef:
    name: git-credentials

---
# Kustomization (Flux applies kustomize overlays)
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: app-production
  namespace: flux-system
spec:
  interval: 5m
  path: ./overlays/production
  prune: true                    # Delete resources removed from Git
  sourceRef:
    kind: GitRepository
    name: app-repo
  targetNamespace: production
  healthChecks:
  - apiVersion: apps/v1
    kind: Deployment
    name: myapp
    namespace: production
  timeout: 3m
  wait: true

---
# HelmRelease (Flux manages Helm charts)
apiVersion: helm.toolkit.fluxcd.io/v2beta2
kind: HelmRelease
metadata:
  name: myapp
  namespace: production
spec:
  interval: 5m
  chart:
    spec:
      chart: myapp
      version: "1.x"
      sourceRef:
        kind: HelmRepository
        name: myrepo
        namespace: flux-system
  values:
    replicaCount: 3
    image:
      tag: v2.0.0
  upgrade:
    remediation:
      retries: 3
  rollback:
    enable: true
` + "```" + `

**ArgoCD:**
` + "```" + `yaml
# Application CRD
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
  finalizers:
  - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/app-manifests
    targetRevision: main
    path: overlays/production
    kustomize:
      images:
      - myregistry/myapp:v2.0.0
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true        # Delete removed resources
      selfHeal: true     # Revert manual changes
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

---
# ApplicationSet (generate apps for multiple environments)
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: myapp-environments
  namespace: argocd
spec:
  generators:
  - list:
      elements:
      - cluster: production
        url: https://prod-cluster.example.com
        revision: main
      - cluster: staging
        url: https://staging-cluster.example.com
        revision: develop
      - cluster: dev
        url: https://dev-cluster.example.com
        revision: develop
  template:
    metadata:
      name: "myapp-{{cluster}}"
    spec:
      project: default
      source:
        repoURL: https://github.com/myorg/app-manifests
        targetRevision: "{{revision}}"
        path: "overlays/{{cluster}}"
      destination:
        server: "{{url}}"
        namespace: myapp
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
` + "```" + `

**Git Repository Structure:**
` + "```" + `
app-manifests/
  base/
    deployment.yaml
    service.yaml
    kustomization.yaml
  overlays/
    dev/
      kustomization.yaml
      patches/
        replicas.yaml
    staging/
      kustomization.yaml
      patches/
        replicas.yaml
    production/
      kustomization.yaml
      patches/
        replicas.yaml
        resources.yaml
        hpa.yaml

# base/kustomization.yaml
resources:
- deployment.yaml
- service.yaml

# overlays/production/kustomization.yaml
resources:
- ../../base
patches:
- path: patches/replicas.yaml
- path: patches/resources.yaml
- path: patches/hpa.yaml
images:
- name: myapp
  newName: myregistry/myapp
  newTag: v2.0.0
namespace: production
` + "```" + ``,
					CodeExamples: `# GitOps Configuration Examples

# 1. Flux multi-tenancy setup
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: infrastructure
  namespace: flux-system
spec:
  interval: 10m
  url: https://github.com/myorg/infrastructure
  ref:
    branch: main
  secretRef:
    name: git-credentials
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: infrastructure
  namespace: flux-system
spec:
  interval: 10m
  path: ./infrastructure
  prune: true
  sourceRef:
    kind: GitRepository
    name: infrastructure
  dependsOn: []
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: apps
  namespace: flux-system
spec:
  interval: 5m
  path: ./apps/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: infrastructure
  dependsOn:
  - name: infrastructure
  healthChecks:
  - apiVersion: apps/v1
    kind: Deployment
    name: myapp
    namespace: production

---
# 2. Flux Image Automation
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageRepository
metadata:
  name: myapp
  namespace: flux-system
spec:
  image: myregistry/myapp
  interval: 5m
  secretRef:
    name: registry-credentials
---
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImagePolicy
metadata:
  name: myapp
  namespace: flux-system
spec:
  imageRepositoryRef:
    name: myapp
  policy:
    semver:
      range: ">=1.0.0"
---
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageUpdateAutomation
metadata:
  name: flux-system
  namespace: flux-system
spec:
  interval: 30m
  sourceRef:
    kind: GitRepository
    name: infrastructure
  git:
    checkout:
      ref:
        branch: main
    commit:
      author:
        email: flux@example.com
        name: Flux
      messageTemplate: "Automated image update"
    push:
      branch: main
  update:
    path: ./apps
    strategy: Setters

---
# 3. ArgoCD with Helm values from Git
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: monitoring-stack
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://prometheus-community.github.io/helm-charts
    chart: kube-prometheus-stack
    targetRevision: "55.0.0"
    helm:
      releaseName: monitoring
      valueFiles:
      - $values/monitoring/production-values.yaml
  sources:
  - repoURL: https://prometheus-community.github.io/helm-charts
    chart: kube-prometheus-stack
    targetRevision: "55.0.0"
    helm:
      valueFiles:
      - $values/monitoring/production-values.yaml
  - repoURL: https://github.com/myorg/helm-values
    targetRevision: main
    ref: values
  destination:
    server: https://kubernetes.default.svc
    namespace: monitoring
  syncPolicy:
    automated:
      prune: true
    syncOptions:
    - ServerSideApply=true

---
# 4. ArgoCD Project for team isolation
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: team-alpha
  namespace: argocd
spec:
  description: "Team Alpha applications"
  sourceRepos:
  - "https://github.com/myorg/team-alpha-*"
  destinations:
  - namespace: "team-alpha-*"
    server: "https://kubernetes.default.svc"
  clusterResourceWhitelist:
  - group: ""
    kind: Namespace
  namespaceResourceBlacklist:
  - group: ""
    kind: ResourceQuota
  - group: ""
    kind: LimitRange
  roles:
  - name: developer
    description: "Team Alpha developer"
    policies:
    - p, proj:team-alpha:developer, applications, get, team-alpha/*, allow
    - p, proj:team-alpha:developer, applications, sync, team-alpha/*, allow
    groups:
    - team-alpha-devs

---
# 5. Notification configuration
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-prod
  namespace: argocd
  annotations:
    notifications.argoproj.io/subscribe.on-sync-succeeded.slack: deployments
    notifications.argoproj.io/subscribe.on-sync-failed.slack: alerts
    notifications.argoproj.io/subscribe.on-health-degraded.slack: alerts
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/app-manifests
    path: production
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: production`,
				},
				{
					Title: "Progressive Delivery",
					Content: `Progressive delivery gradually rolls out changes to reduce risk. Key strategies include canary deployments, blue-green, and A/B testing.

**Deployment Strategies:**
` + "```" + `
Rolling Update (Kubernetes default):
  - Gradually replaces old pods with new
  - Controlled by maxSurge and maxUnavailable
  - Simple, no extra infrastructure
  - Limited traffic control

Blue-Green:
  - Two identical environments (blue = current, green = new)
  - Switch traffic all at once via service selector
  - Instant rollback (switch back)
  - Requires 2x resources during deploy

Canary:
  - Route small % of traffic to new version
  - Gradually increase if metrics look good
  - Automated promotion/rollback based on metrics
  - Requires traffic splitting (Istio, Nginx, etc.)

A/B Testing:
  - Route based on headers, cookies, user attributes
  - Different users see different versions
  - Business metric driven (conversion, engagement)
` + "```" + `

**Argo Rollouts:**
` + "```" + `yaml
# Canary deployment with Argo Rollouts
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
  namespace: production
spec:
  replicas: 10
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
        image: myregistry/myapp:v2.0.0
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: "1"
            memory: 512Mi
  strategy:
    canary:
      canaryService: myapp-canary
      stableService: myapp-stable
      trafficRouting:
        nginx:
          stableIngress: myapp-ingress
          annotationPrefix: nginx.ingress.kubernetes.io
      steps:
      # Step 1: 5% traffic to canary
      - setWeight: 5
      - pause: {duration: 5m}
      # Step 2: Run analysis
      - analysis:
          templates:
          - templateName: success-rate
          - templateName: latency
          args:
          - name: service-name
            value: myapp-canary
      # Step 3: Increase to 25%
      - setWeight: 25
      - pause: {duration: 10m}
      # Step 4: Run analysis again
      - analysis:
          templates:
          - templateName: success-rate
      # Step 5: 50%
      - setWeight: 50
      - pause: {duration: 10m}
      # Step 6: 100% (promotion)
      - setWeight: 100
      
      # Automatic rollback on failure
      abortScaleDownDelaySeconds: 30
      dynamicStableScale: true

---
# AnalysisTemplate — Prometheus-based
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: production
spec:
  args:
  - name: service-name
  metrics:
  - name: success-rate
    interval: 1m
    count: 5
    successCondition: result[0] >= 0.95
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: |
          sum(rate(http_requests_total{service="{{args.service-name}}",status!~"5.."}[5m]))
          /
          sum(rate(http_requests_total{service="{{args.service-name}}"}[5m]))

---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: latency
  namespace: production
spec:
  args:
  - name: service-name
  metrics:
  - name: p99-latency
    interval: 1m
    count: 5
    successCondition: result[0] < 0.5
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket{service="{{args.service-name}}"}[5m])) by (le)
          )
` + "```" + `

**Blue-Green with Argo Rollouts:**
` + "```" + `yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp-bluegreen
  namespace: production
spec:
  replicas: 5
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
        image: myregistry/myapp:v2.0.0
        ports:
        - containerPort: 8080
  strategy:
    blueGreen:
      activeService: myapp-active
      previewService: myapp-preview
      autoPromotionEnabled: false
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: myapp-preview
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: myapp-active
      scaleDownDelaySeconds: 300  # Keep old version for 5 min
      abortScaleDownDelaySeconds: 30
` + "```" + `

**Flagger (Progressive Delivery Operator):**
` + "```" + `yaml
# Flagger Canary with Istio
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: myapp
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  service:
    port: 8080
    targetPort: 8080
    gateways:
    - public-gateway.istio-system.svc.cluster.local
    hosts:
    - app.example.com
  analysis:
    interval: 1m
    threshold: 5             # Max failed checks before rollback
    maxWeight: 50            # Max canary traffic %
    stepWeight: 10           # Traffic increase per step
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
    webhooks:
    - name: load-test
      url: http://flagger-loadtester.test/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://myapp-canary.production:8080/"
` + "```" + ``,
					CodeExamples: `# Progressive Delivery Examples

# 1. Services for canary routing
apiVersion: v1
kind: Service
metadata:
  name: myapp-stable
  namespace: production
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: myapp-canary
  namespace: production
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080

---
# 2. Ingress for canary with nginx
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "0"
spec:
  ingressClassName: nginx
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: myapp-stable
            port:
              number: 80

---
# 3. Istio VirtualService for traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: myapp
  namespace: production
spec:
  hosts:
  - app.example.com
  gateways:
  - public-gateway
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: myapp-canary
        port:
          number: 80
  - route:
    - destination:
        host: myapp-stable
        port:
          number: 80
      weight: 90
    - destination:
        host: myapp-canary
        port:
          number: 80
      weight: 10

---
# 4. AnalysisRun for manual trigger
apiVersion: argoproj.io/v1alpha1
kind: AnalysisRun
metadata:
  name: myapp-canary-analysis-1
  namespace: production
spec:
  metrics:
  - name: success-rate
    interval: 30s
    count: 10
    successCondition: result[0] >= 0.99
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: |
          sum(rate(http_requests_total{service="myapp-canary",status!~"5.."}[2m]))
          /
          sum(rate(http_requests_total{service="myapp-canary"}[2m]))
  - name: error-count
    interval: 30s
    count: 10
    failureCondition: result[0] > 10
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: |
          sum(increase(http_requests_total{service="myapp-canary",status=~"5.."}[2m]))
  - name: latency-p99
    interval: 30s
    count: 10
    successCondition: result[0] < 0.5
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket{service="myapp-canary"}[2m])) by (le)
          )

---
# 5. Experiment (A/B testing)
apiVersion: argoproj.io/v1alpha1
kind: Experiment
metadata:
  name: myapp-ab-test
  namespace: production
spec:
  duration: 1h
  templates:
  - name: baseline
    replicas: 2
    selector:
      matchLabels:
        app: myapp
        variant: baseline
    template:
      metadata:
        labels:
          app: myapp
          variant: baseline
      spec:
        containers:
        - name: app
          image: myregistry/myapp:v1.0.0
  - name: canary
    replicas: 2
    selector:
      matchLabels:
        app: myapp
        variant: canary
    template:
      metadata:
        labels:
          app: myapp
          variant: canary
      spec:
        containers:
        - name: app
          image: myregistry/myapp:v2.0.0
  analyses:
  - name: compare-success-rates
    templateName: success-rate-comparison
    args:
    - name: baseline-hash
      value: baseline
    - name: canary-hash
      value: canary`,
				},
			},
		},
	})
}
