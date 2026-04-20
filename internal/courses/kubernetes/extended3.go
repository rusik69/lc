package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1721,
			Title:       "Helm and Package Management",
			Description: "Master Helm charts: templating, values, hooks, dependencies, chart repositories, and OCI-based distribution.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Helm Charts Deep Dive",
					Content: `Helm is the de facto package manager for Kubernetes. It uses charts (packages of templated YAML) to deploy applications.

**Chart Structure:**
` + "```" + `
mychart/
  Chart.yaml           # Metadata: name, version, dependencies
  values.yaml          # Default configuration values
  charts/              # Dependency charts
  templates/           # Kubernetes manifests (Go templates)
    deployment.yaml
    service.yaml
    ingress.yaml
    hpa.yaml
    configmap.yaml
    secret.yaml
    serviceaccount.yaml
    _helpers.tpl       # Template helpers/partials
    NOTES.txt          # Post-install instructions
    tests/
      test-connection.yaml
  .helmignore          # Files to exclude from packaging
` + "```" + `

**Chart.yaml:**
` + "```" + `yaml
apiVersion: v2
name: myapp
description: A Helm chart for MyApp
type: application      # application or library
version: 1.2.3         # Chart version (SemVer)
appVersion: "2.0.0"    # Application version

keywords:
  - web
  - api
maintainers:
  - name: team
    email: team@example.com

dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
  - name: redis
    version: "17.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
    alias: cache
` + "```" + `

**Go Template Basics:**
` + "```" + `
Values access:
  {{ .Values.image.repository }}
  {{ .Values.image.tag | default .Chart.AppVersion }}
  {{ .Release.Name }}
  {{ .Release.Namespace }}
  {{ .Chart.Name }}

Control flow:
  {{- if .Values.ingress.enabled }}
  ... ingress manifest ...
  {{- end }}
  
  {{- range .Values.extraEnv }}
  - name: {{ .name }}
    value: {{ .value | quote }}
  {{- end }}

  {{ with .Values.nodeSelector }}
  nodeSelector:
    {{- toYaml . | nindent 4 }}
  {{ end }}

Functions:
  {{ .Values.name | quote }}              # Quote string
  {{ .Values.name | upper }}              # Uppercase
  {{ .Values.name | default "myapp" }}    # Default value
  {{ include "mychart.fullname" . }}      # Include helper
  {{ toYaml .Values.resources | nindent 8 }} # YAML emit
  {{ required "image.tag is required" .Values.image.tag }}
  {{ printf "%s-%s" .Release.Name .Chart.Name }}

Whitespace:
  {{-  → trim left whitespace
  -}}  → trim right whitespace
  Critical for clean YAML output!
` + "```" + `

**_helpers.tpl:**
` + "```" + `
{{/*
Expand the name of the chart.
*/}}
{{- define "mychart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mychart.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mychart.labels" -}}
helm.sh/chart: {{ include "mychart.chart" . }}
{{ include "mychart.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "mychart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mychart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
` + "```" + `

**Helm Commands:**
` + "```" + `
# Install
helm install myrelease ./mychart
helm install myrelease ./mychart -f production-values.yaml
helm install myrelease ./mychart --set image.tag=v2.0.0

# Upgrade
helm upgrade myrelease ./mychart
helm upgrade --install myrelease ./mychart  # Install or upgrade

# Rollback
helm rollback myrelease 1    # Rollback to revision 1
helm history myrelease       # Show revision history

# Template (dry-run, show rendered manifests)
helm template myrelease ./mychart
helm template myrelease ./mychart --debug  # Show errors

# Dependencies
helm dependency update ./mychart
helm dependency build ./mychart

# Package and push
helm package ./mychart
helm push mychart-1.2.3.tgz oci://registry.example.com/charts

# Test
helm test myrelease
` + "```" + ``,
					CodeExamples: `# Complete Helm Chart Example

# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "mychart.fullname" . }}
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "mychart.selectorLabels" . | nindent 6 }}
  strategy:
    type: {{ .Values.strategy.type | default "RollingUpdate" }}
    {{- if eq .Values.strategy.type "RollingUpdate" }}
    rollingUpdate:
      maxSurge: {{ .Values.strategy.maxSurge | default "25%" }}
      maxUnavailable: {{ .Values.strategy.maxUnavailable | default "25%" }}
    {{- end }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "mychart.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "mychart.serviceAccountName" . }}
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        ports:
        - name: http
          containerPort: {{ .Values.containerPort | default 8080 }}
          protocol: TCP
        {{- if .Values.probes.liveness.enabled }}
        livenessProbe:
          httpGet:
            path: {{ .Values.probes.liveness.path | default "/healthz" }}
            port: http
          initialDelaySeconds: {{ .Values.probes.liveness.initialDelaySeconds | default 10 }}
          periodSeconds: {{ .Values.probes.liveness.periodSeconds | default 15 }}
        {{- end }}
        {{- if .Values.probes.readiness.enabled }}
        readinessProbe:
          httpGet:
            path: {{ .Values.probes.readiness.path | default "/readyz" }}
            port: http
          initialDelaySeconds: {{ .Values.probes.readiness.initialDelaySeconds | default 5 }}
          periodSeconds: {{ .Values.probes.readiness.periodSeconds | default 10 }}
        {{- end }}
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        {{- with .Values.env }}
        env:
          {{- range $key, $value := . }}
          - name: {{ $key }}
            value: {{ $value | quote }}
          {{- end }}
        {{- end }}
        {{- with .Values.envFrom }}
        envFrom:
          {{- toYaml . | nindent 10 }}
        {{- end }}
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        {{- with .Values.extraVolumeMounts }}
          {{- toYaml . | nindent 8 }}
        {{- end }}
      volumes:
      - name: tmp
        emptyDir: {}
      {{- with .Values.extraVolumes }}
        {{- toYaml . | nindent 6 }}
      {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.topologySpreadConstraints }}
      topologySpreadConstraints:
        {{- toYaml . | nindent 8 }}
      {{- end }}

---
# templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "mychart.fullname" . }}
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type | default "ClusterIP" }}
  ports:
  - port: {{ .Values.service.port | default 80 }}
    targetPort: http
    protocol: TCP
    name: http
  selector:
    {{- include "mychart.selectorLabels" . | nindent 4 }}

---
# templates/ingress.yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "mychart.fullname" . }}
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType | default "Prefix" }}
            backend:
              service:
                name: {{ include "mychart.fullname" $ }}
                port:
                  number: {{ $.Values.service.port | default 80 }}
          {{- end }}
    {{- end }}
{{- end }}

---
# templates/hpa.yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "mychart.fullname" . }}
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "mychart.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
  {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
  {{- end }}
  {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
  {{- end }}
{{- end }}`,
				},
				{
					Title: "Helm Hooks and Testing",
					Content: `Helm hooks allow you to run operations at specific points in the release lifecycle. They're used for pre/post install, upgrade, delete, and rollback actions.

**Hook Types:**
` + "```" + `
Annotations:
  helm.sh/hook: <event>
  helm.sh/hook-weight: "<number>"    # Order (ascending)
  helm.sh/hook-delete-policy: <policy>

Events:
  pre-install      → before any resources are installed
  post-install     → after all resources are installed
  pre-delete       → before deletion begins
  post-delete      → after deletion completes
  pre-upgrade      → before upgrade begins
  post-upgrade     → after upgrade completes
  pre-rollback     → before rollback begins
  post-rollback    → after rollback completes
  test             → run on "helm test"

Delete policies:
  before-hook-creation → delete previous hook before new one
  hook-succeeded       → delete after hook succeeds
  hook-failed          → delete after hook fails
` + "```" + `

**Database Migration Hook:**
` + "```" + `yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "mychart.fullname" . }}-migrate
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": pre-upgrade,pre-install
    "helm.sh/hook-weight": "-5"
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 300
  template:
    metadata:
      labels:
        app: {{ include "mychart.name" . }}-migrate
    spec:
      restartPolicy: Never
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: migrate
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        command: ["./migrate", "--direction", "up"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {{ include "mychart.fullname" . }}-db
              key: url
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
` + "```" + `

**Helm Tests:**
` + "```" + `yaml
# templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "mychart.fullname" . }}-test-connection"
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  restartPolicy: Never
  containers:
  - name: wget
    image: busybox:1.36
    command: ['wget']
    args: ['{{ include "mychart.fullname" . }}:{{ .Values.service.port }}/healthz']

---
# templates/tests/test-api.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "mychart.fullname" . }}-test-api"
  annotations:
    "helm.sh/hook": test
    "helm.sh/hook-weight": "5"
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  restartPolicy: Never
  containers:
  - name: test
    image: curlimages/curl:8.4.0
    command: ['sh', '-c']
    args:
    - |
      set -e
      echo "Testing health endpoint..."
      curl -sf http://{{ include "mychart.fullname" . }}:{{ .Values.service.port }}/healthz
      echo "Testing API endpoint..."
      curl -sf http://{{ include "mychart.fullname" . }}:{{ .Values.service.port }}/api/v1/status
      echo "All tests passed!"
` + "```" + `

**Values Schema Validation:**
` + "```" + `json
// values.schema.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["image"],
  "properties": {
    "replicaCount": {
      "type": "integer",
      "minimum": 1,
      "default": 1
    },
    "image": {
      "type": "object",
      "required": ["repository"],
      "properties": {
        "repository": {
          "type": "string",
          "pattern": "^[a-z0-9/.:-]+$"
        },
        "tag": {
          "type": "string"
        },
        "pullPolicy": {
          "type": "string",
          "enum": ["Always", "IfNotPresent", "Never"]
        }
      }
    },
    "service": {
      "type": "object",
      "properties": {
        "type": {
          "type": "string",
          "enum": ["ClusterIP", "NodePort", "LoadBalancer"]
        },
        "port": {
          "type": "integer",
          "minimum": 1,
          "maximum": 65535
        }
      }
    }
  }
}
` + "```" + `

**Best Practices:**
` + "```" + `
Chart design:
  ✓ Use _helpers.tpl for reusable template snippets
  ✓ Include checksum/config annotation for ConfigMap changes
  ✓ Support nameOverride and fullnameOverride
  ✓ Use default values for everything
  ✓ Add values.schema.json for validation
  ✓ Include NOTES.txt with useful post-install info
  
Values design:
  ✓ Flat, predictable structure
  ✓ Document every value in values.yaml with comments
  ✓ Use .Values.global for cross-chart values
  ✓ Provide sane defaults (chart works with zero overrides)
  
Versioning:
  ✓ Chart version (version:) → chart template/structure changes
  ✓ App version (appVersion:) → application version
  ✓ Both follow SemVer
  ✓ Bump chart version on any template change

Security:
  ✓ Pin image tags (never use :latest)
  ✓ Use digest pinning for critical charts
  ✓ Sign charts with helm provenance
  ✓ Store in OCI registry (helm push to oci://)
` + "```" + ``,
					CodeExamples: `# values.yaml — Complete example with documentation

# -- Number of replicas
replicaCount: 2

image:
  # -- Container image repository
  repository: myregistry/myapp
  # -- Image pull policy
  pullPolicy: IfNotPresent
  # -- Overrides the image tag (default: chart appVersion)
  tag: ""

# -- Image pull secrets
imagePullSecrets: []
# -- Override the chart name
nameOverride: ""
# -- Override the full release name
fullnameOverride: ""

serviceAccount:
  # -- Create a ServiceAccount
  create: true
  # -- Annotations for the ServiceAccount
  annotations: {}
  # -- ServiceAccount name (generated if not set)
  name: ""

# -- Pod annotations
podAnnotations: {}

# -- Container port
containerPort: 8080

service:
  # -- Service type
  type: ClusterIP
  # -- Service port
  port: 80

ingress:
  # -- Enable ingress
  enabled: false
  # -- Ingress class name
  className: nginx
  # -- Ingress annotations
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  # -- Ingress hosts
  hosts:
    - host: myapp.example.com
      paths:
        - path: /
          pathType: Prefix
  # -- Ingress TLS configuration
  tls:
    - secretName: myapp-tls
      hosts:
        - myapp.example.com

# -- Resource requests and limits
resources:
  limits:
    cpu: "1"
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 256Mi

autoscaling:
  # -- Enable HPA
  enabled: true
  # -- Minimum replicas
  minReplicas: 2
  # -- Maximum replicas
  maxReplicas: 10
  # -- Target CPU utilization
  targetCPUUtilizationPercentage: 70
  # -- Target memory utilization
  targetMemoryUtilizationPercentage: 80

# -- Node selector
nodeSelector: {}

# -- Tolerations
tolerations: []

# -- Affinity rules
affinity: {}

# -- Topology spread constraints
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: DoNotSchedule
    labelSelector:
      matchLabels:
        app.kubernetes.io/name: myapp

probes:
  liveness:
    enabled: true
    path: /healthz
    initialDelaySeconds: 10
    periodSeconds: 15
  readiness:
    enabled: true
    path: /readyz
    initialDelaySeconds: 5
    periodSeconds: 10

# -- Environment variables (key: value)
env:
  LOG_LEVEL: info
  APP_ENV: production

# -- Environment variable sources
envFrom: []

strategy:
  type: RollingUpdate
  maxSurge: "25%"
  maxUnavailable: "0"

# -- Extra volume mounts
extraVolumeMounts: []
# -- Extra volumes
extraVolumes: []

# Database dependency
postgresql:
  enabled: true
  auth:
    postgresPassword: changeme
    database: myapp
  primary:
    persistence:
      size: 10Gi

# Cache dependency
redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: false

---
# production-values.yaml — Production overrides
replicaCount: 5

image:
  tag: "2.0.0"

resources:
  limits:
    cpu: "2"
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 50
  targetCPUUtilizationPercentage: 60

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: app-tls
      hosts:
        - app.example.com

env:
  LOG_LEVEL: warn
  APP_ENV: production
  GOMAXPROCS: "2"

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - myapp
        topologyKey: kubernetes.io/hostname

postgresql:
  enabled: true
  primary:
    persistence:
      size: 100Gi
    resources:
      limits:
        cpu: "4"
        memory: 8Gi`,
				},
			},
		},
	})
}
