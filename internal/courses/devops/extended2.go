package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1452,
			Title:       "Advanced CI/CD Pipeline Patterns",
			Description: "Master advanced continuous integration and delivery patterns including trunk-based development, pipeline as code, multi-environment deployments, and release management strategies.",
			Order:       52,
			Lessons: []problems.Lesson{
				{
					Title: "CI/CD Pipeline Architecture and Patterns",
					Content: `Advanced CI/CD pipeline patterns enable teams to deliver software rapidly and reliably across complex environments.

**Trunk-Based Development:**
` + "```" + `
Trunk-based development model:
  Main branch (trunk):
    - Single source of truth
    - Always deployable
    - Small, frequent commits
    - Feature flags for incomplete work
  
  Short-lived feature branches:
    - Live < 1-2 days
    - Small scope (< 400 lines)
    - Merge to trunk frequently
    - Delete after merge
  
  Release branches (optional):
    - Cut from trunk for releases
    - Cherry-pick fixes only
    - No new features
    - Short-lived (< 2 weeks)

  Benefits:
    - Continuous integration (real CI)
    - Fewer merge conflicts
    - Faster feedback loops
    - Simpler branch management
    - Better code review efficiency

  vs GitFlow:
    GitFlow:
      develop → feature → develop → release → main → hotfix
      Complex, long-lived branches, infrequent integration
    
    Trunk-based:
      feature → main (with feature flags)
      Simple, short-lived branches, continuous integration

Feature flags:
  Types:
    Release flags:     Hide incomplete features
    Experiment flags:  A/B testing
    Ops flags:         Circuit breakers, kill switches
    Permission flags:  Entitlements, beta access
  
  Implementation:
    # LaunchDarkly, Unleash, Flagsmith, or custom
    if feature_enabled("new-checkout"):
        return new_checkout_flow()
    else:
        return legacy_checkout_flow()
  
  Lifecycle:
    1. Create flag (default: off)
    2. Develop behind flag
    3. Test with flag on (specific environments/users)
    4. Gradually roll out (10% → 50% → 100%)
    5. Remove flag and dead code
  
  Best practices:
    - Flag naming convention (team-feature-date)
    - Regular flag cleanup (< 30 days for release flags)
    - Flag ownership (team/person responsible)
    - Avoid nested flags
    - Monitor flag evaluations
` + "```" + `

**Pipeline as Code:**
` + "```" + `
GitHub Actions:
  name: CI/CD Pipeline
  on:
    push:
      branches: [main]
    pull_request:
  
  concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: true
  
  jobs:
    lint:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-go@v5
          with:
            go-version: '1.22'
        - run: golangci-lint run ./...
    
    test:
      runs-on: ubuntu-latest
      strategy:
        matrix:
          go-version: ['1.21', '1.22']
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-go@v5
          with:
            go-version: ${{ matrix.go-version }}
        - run: go test -race -coverprofile=coverage.out ./...
        - uses: codecov/codecov-action@v4
    
    build:
      needs: [lint, test]
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: docker/setup-buildx-action@v3
        - uses: docker/login-action@v3
          with:
            registry: ghcr.io
            username: ${{ github.actor }}
            password: ${{ secrets.GITHUB_TOKEN }}
        - uses: docker/build-push-action@v5
          with:
            push: true
            tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
            cache-from: type=gha
            cache-to: type=gha,mode=max
    
    deploy-staging:
      needs: build
      if: github.ref == 'refs/heads/main'
      environment: staging
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: |
            kubectl set image deployment/app \
              app=ghcr.io/${{ github.repository }}:${{ github.sha }}
    
    deploy-production:
      needs: deploy-staging
      if: github.ref == 'refs/heads/main'
      environment:
        name: production
        url: https://myapp.example.com
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: |
            kubectl set image deployment/app \
              app=ghcr.io/${{ github.repository }}:${{ github.sha }}

GitLab CI:
  stages:
    - lint
    - test
    - build
    - deploy
  
  variables:
    DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  
  lint:
    stage: lint
    image: golangci/golangci-lint
    script:
      - golangci-lint run ./...
  
  test:
    stage: test
    image: golang:1.22
    script:
      - go test -race -coverprofile=coverage.out ./...
    coverage: '/^total:.*\s(\d+\.\d+)%/'
    artifacts:
      reports:
        coverage_report:
          coverage_format: cobertura
          path: coverage.xml
  
  build:
    stage: build
    image: docker:24
    services:
      - docker:24-dind
    script:
      - docker build -t $DOCKER_IMAGE .
      - docker push $DOCKER_IMAGE
  
  deploy_staging:
    stage: deploy
    environment:
      name: staging
    script:
      - kubectl set image deployment/app app=$DOCKER_IMAGE
    only:
      - main
  
  deploy_production:
    stage: deploy
    environment:
      name: production
    script:
      - kubectl set image deployment/app app=$DOCKER_IMAGE
    when: manual
    only:
      - main

Jenkins Pipeline (Declarative):
  pipeline {
    agent any
    
    environment {
      DOCKER_IMAGE = "myregistry/myapp:${env.BUILD_NUMBER}"
    }
    
    stages {
      stage('Test') {
        steps {
          sh 'go test -race ./...'
        }
      }
      stage('Build') {
        steps {
          sh "docker build -t ${DOCKER_IMAGE} ."
          sh "docker push ${DOCKER_IMAGE}"
        }
      }
      stage('Deploy Staging') {
        when { branch 'main' }
        steps {
          sh "kubectl set image deployment/app app=${DOCKER_IMAGE}"
        }
      }
      stage('Deploy Production') {
        when { branch 'main' }
        input { message "Deploy to production?" }
        steps {
          sh "kubectl set image deployment/app app=${DOCKER_IMAGE}"
        }
      }
    }
  }
` + "```" + `

**Deployment Strategies:**
` + "```" + `
Rolling update:
  - Replace instances gradually
  - Zero downtime
  - Rollback by reversing
  - Default Kubernetes strategy
  
  spec:
    strategy:
      type: RollingUpdate
      rollingUpdate:
        maxSurge: 25%
        maxUnavailable: 25%

Blue-Green deployment:
  - Two identical environments (blue/green)
  - Deploy to inactive environment
  - Switch traffic instantly
  - Instant rollback (switch back)
  
  Process:
    1. Blue (v1) serving traffic
    2. Deploy v2 to Green
    3. Test Green
    4. Switch load balancer to Green
    5. Green (v2) serving traffic
    6. Blue becomes standby

Canary deployment:
  - Route small % of traffic to new version
  - Monitor metrics (errors, latency)
  - Gradually increase traffic
  - Automated rollback on anomaly
  
  Stages:
    1. Deploy canary (v2) alongside stable (v1)
    2. Route 5% traffic to canary
    3. Monitor for 15 minutes
    4. Route 25% → 50% → 100%
    5. Remove old version
  
  Argo Rollouts:
    spec:
      strategy:
        canary:
          steps:
            - setWeight: 5
            - pause: { duration: 10m }
            - setWeight: 25
            - pause: { duration: 10m }
            - setWeight: 50
            - pause: { duration: 10m }
          canaryMetadata:
            labels:
              role: canary
          analysis:
            templates:
              - templateName: success-rate
            startingStep: 2

A/B testing:
  - Route based on user attributes
  - Header, cookie, or user segment
  - Measure business metrics
  - Data-driven decisions

Shadow deployment:
  - Mirror production traffic to new version
  - New version processes but doesn't respond
  - Compare results
  - No user impact
` + "```" + ``,
					CodeExamples: `# CI/CD pipeline management scripts

# 1. Deployment status checker
#!/bin/bash
echo "=== Deployment Status ==="

# Kubernetes rollout status
for deploy in $(kubectl get deployments -o name 2>/dev/null); do
    NAME=$(echo "$deploy" | cut -d'/' -f2)
    STATUS=$(kubectl rollout status "$deploy" --timeout=5s 2>&1)
    IMAGE=$(kubectl get "$deploy" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null)
    REPLICAS=$(kubectl get "$deploy" -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null)
    
    echo "  $NAME: $REPLICAS ready, image=$IMAGE"
done

# Recent deployments
echo ""
echo "--- Recent Rollout History ---"
for deploy in $(kubectl get deployments -o name 2>/dev/null); do
    echo "$deploy:"
    kubectl rollout history "$deploy" 2>/dev/null | tail -5
done

# 2. Pipeline metrics
#!/bin/bash
echo "=== Pipeline Metrics ==="

# GitHub Actions (requires gh CLI)
echo "--- Recent Workflow Runs ---"
gh run list --limit 10 --json name,status,conclusion,createdAt \
    --jq '.[] | "\(.name): \(.conclusion // .status) (\(.createdAt))"' 2>/dev/null

echo ""
echo "--- Success Rate (last 50 runs) ---"
TOTAL=$(gh run list --limit 50 --json conclusion --jq 'length' 2>/dev/null)
SUCCESS=$(gh run list --limit 50 --json conclusion --jq '[.[] | select(.conclusion=="success")] | length' 2>/dev/null)
if [ -n "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
    RATE=$(echo "scale=1; $SUCCESS * 100 / $TOTAL" | bc 2>/dev/null)
    echo "  Success: $SUCCESS/$TOTAL ($RATE%)"
fi

# 3. Canary analysis
#!/bin/bash
echo "=== Canary Analysis ==="

CANARY_POD=$(kubectl get pods -l role=canary -o name 2>/dev/null | head -1)
STABLE_POD=$(kubectl get pods -l role=stable -o name 2>/dev/null | head -1)

if [ -n "$CANARY_POD" ]; then
    echo "Canary pod: $CANARY_POD"
    kubectl top "$CANARY_POD" 2>/dev/null
    
    echo ""
    echo "Stable pod: $STABLE_POD"
    kubectl top "$STABLE_POD" 2>/dev/null
fi`,
				},
				{
					Title: "Release Management and Artifact Versioning",
					Content: `Release management ensures consistent, traceable, and repeatable software delivery across environments.

**Semantic Versioning:**
` + "```" + `
SemVer format: MAJOR.MINOR.PATCH
  MAJOR: Breaking changes (incompatible API)
  MINOR: New features (backward compatible)
  PATCH: Bug fixes (backward compatible)
  
  Pre-release: 1.0.0-alpha.1, 1.0.0-beta.2, 1.0.0-rc.1
  Build metadata: 1.0.0+build.123

Automated versioning:
  Conventional Commits:
    feat: new feature             → minor bump
    fix: bug fix                  → patch bump
    feat!: breaking change        → major bump
    BREAKING CHANGE: in footer    → major bump
    
    feat(auth): add OAuth2 support
    fix(api): handle null response
    feat!: redesign user API
    chore: update dependencies (no version bump)
  
  Tools:
    semantic-release (Node.js)
    goreleaser (Go)
    release-please (Google)
    conventional-changelog

Container image tagging:
  Best practices:
    - Use commit SHA: myapp:abc123f
    - Use SemVer: myapp:1.2.3
    - Use branch: myapp:main (mutable, for dev)
    - Never use :latest in production
    - Immutable tags for production
  
  Multi-tag strategy:
    docker build -t myapp:1.2.3 \
                 -t myapp:1.2 \
                 -t myapp:1 \
                 -t myapp:abc123f .
` + "```" + `

**Artifact Management:**
` + "```" + `
Container registries:
  Docker Hub:       Public/private, rate limits
  GitHub (ghcr.io): Integrated with GitHub
  AWS ECR:          Integrated with AWS
  GCP Artifact Reg: Integrated with GCP
  Azure ACR:        Integrated with Azure
  Harbor:           Self-hosted, CNCF
  
  Vulnerability scanning:
    docker scout cve myapp:1.2.3
    trivy image myapp:1.2.3
    grype myapp:1.2.3
  
  Image signing:
    cosign sign --key cosign.key myregistry/myapp:1.2.3
    cosign verify --key cosign.pub myregistry/myapp:1.2.3

Helm chart repositories:
  # Package chart
  helm package ./my-chart --version 1.2.3
  
  # Push to OCI registry
  helm push my-chart-1.2.3.tgz oci://myregistry/charts
  
  # Push to ChartMuseum
  curl --data-binary "@my-chart-1.2.3.tgz" \
    https://chartmuseum.example.com/api/charts

Binary artifacts:
  Go: goreleaser
    # .goreleaser.yml
    builds:
      - env: [CGO_ENABLED=0]
        goos: [linux, darwin, windows]
        goarch: [amd64, arm64]
    
    archives:
      - format: tar.gz
        format_overrides:
          - goos: windows
            format: zip
    
    release:
      github:
        owner: myorg
        name: myapp
    
    # Run
    goreleaser release --clean

SBOM (Software Bill of Materials):
  Generate:
    syft myapp:1.2.3 -o spdx-json > sbom.json
    trivy sbom myapp:1.2.3 --format spdx-json > sbom.json
  
  Attach to image:
    cosign attach sbom --sbom sbom.json myregistry/myapp:1.2.3
  
  Verify:
    cosign verify-attestation myregistry/myapp:1.2.3

Supply chain security:
  SLSA framework (Supply chain Levels for Software Artifacts):
    Level 1: Documentation of build process
    Level 2: Tamper resistance of build service
    Level 3: Hardened against tampering
    Level 4: Two-person reviewed, hermetic builds
  
  Sigstore:
    cosign:    Container signing
    rekor:     Transparency log
    fulcio:    Certificate authority
` + "```" + `

**Environment Promotion:**
` + "```" + `
Promotion pipeline:
  dev → staging → production
  
  Each environment:
    - Separate namespace/cluster
    - Environment-specific config
    - Promotion gates (tests, approvals)
    - Audit trail

GitOps promotion:
  Repository structure:
    environments/
    ├── dev/
    │   ├── kustomization.yaml
    │   └── patches/
    ├── staging/
    │   ├── kustomization.yaml
    │   └── patches/
    └── production/
        ├── kustomization.yaml
        └── patches/
  
  Promote by updating image tag:
    # dev/kustomization.yaml
    images:
      - name: myapp
        newTag: abc123f  # commit SHA
    
    # Promote to staging: copy tag to staging/kustomization.yaml
    # Promote to production: copy tag to production/kustomization.yaml

Promotion gates:
  Automated:
    - All tests pass
    - Security scan clean
    - Performance regression check
    - Compliance check
    - SLA metrics within threshold
  
  Manual:
    - Change advisory board approval
    - Product owner sign-off
    - Security team review
  
  Progressive:
    - Canary metrics healthy for N minutes
    - Error rate below threshold
    - Latency within SLA
    - No increase in support tickets

Rollback strategies:
  Kubernetes:
    kubectl rollout undo deployment/myapp
    kubectl rollout undo deployment/myapp --to-revision=3
  
  GitOps:
    git revert HEAD  # Revert last commit
    # ArgoCD/Flux will auto-sync
  
  Database:
    Forward-only migrations (recommended)
    - Write backward-compatible migrations
    - Deploy migration before code change
    - Code handles both old and new schema
    
    Rollback migrations (risky):
    - Separate up/down scripts
    - Test rollback in staging
    - May lose data
` + "```" + ``,
					CodeExamples: `# Release management scripts

# 1. Semantic version bumper
#!/bin/bash
echo "=== Version Bump ==="

CURRENT=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
echo "Current: $CURRENT"

# Parse version
MAJOR=$(echo "$CURRENT" | sed 's/v//' | cut -d. -f1)
MINOR=$(echo "$CURRENT" | sed 's/v//' | cut -d. -f2)
PATCH=$(echo "$CURRENT" | sed 's/v//' | cut -d. -f3)

# Check conventional commits since last tag
BREAKING=$(git log "$CURRENT"..HEAD --pretty=format:"%s" 2>/dev/null | grep -c "BREAKING CHANGE\|!:" || true)
FEATURES=$(git log "$CURRENT"..HEAD --pretty=format:"%s" 2>/dev/null | grep -c "^feat" || true)
FIXES=$(git log "$CURRENT"..HEAD --pretty=format:"%s" 2>/dev/null | grep -c "^fix" || true)

echo "Since $CURRENT: $BREAKING breaking, $FEATURES features, $FIXES fixes"

if [ "$BREAKING" -gt 0 ]; then
    MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0
    BUMP="major"
elif [ "$FEATURES" -gt 0 ]; then
    MINOR=$((MINOR + 1)); PATCH=0
    BUMP="minor"
elif [ "$FIXES" -gt 0 ]; then
    PATCH=$((PATCH + 1))
    BUMP="patch"
else
    echo "No version bump needed"
    exit 0
fi

NEW_VERSION="v$MAJOR.$MINOR.$PATCH"
echo "Bump: $BUMP → $NEW_VERSION"

# 2. Release notes generator
#!/bin/bash
echo "=== Release Notes ==="

LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null)
echo "## Changes since $LAST_TAG"

echo ""
echo "### Features"
git log "$LAST_TAG"..HEAD --pretty=format:"- %s (%h)" 2>/dev/null | grep "^- feat" || echo "None"

echo ""
echo "### Bug Fixes"
git log "$LAST_TAG"..HEAD --pretty=format:"- %s (%h)" 2>/dev/null | grep "^- fix" || echo "None"

echo ""
echo "### Other"
git log "$LAST_TAG"..HEAD --pretty=format:"- %s (%h)" 2>/dev/null | grep -v "^- feat\|^- fix" || echo "None"

# 3. Artifact inventory
#!/bin/bash
echo "=== Artifact Inventory ==="

# Container images
echo "--- Container Images ---"
for deploy in $(kubectl get deployments -o name 2>/dev/null); do
    NAME=$(echo "$deploy" | cut -d'/' -f2)
    IMAGE=$(kubectl get "$deploy" -o jsonpath='{.spec.template.spec.containers[0].image}')
    echo "  $NAME: $IMAGE"
done

# Helm releases
echo ""
echo "--- Helm Releases ---"
helm list --all-namespaces 2>/dev/null | head -15`,
				},
			},
		},
	})
}
