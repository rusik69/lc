package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          250,
			Title:       "DevOps Fundamentals",
			Description: "Introduction to DevOps: culture, principles, practices, and the DevOps lifecycle.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "DevOps Fundamentals",
					Content: `DevOps is the union of people, process, and technology to continually provide value to customers. It’s a culture shift that removes the high-wall between Development and Operations.

**1. The Pillars of DevOps (C.A.L.M.S.)**
*   **Culture:** Shared responsibility and trust.
*   **Automation:** If you do it twice, automate it.
*   **Lean:** Small batch sizes, fast feedback loops.
*   **Measurement:** Use data (DORA metrics) to improve.
*   **Sharing:** Open communication and shared goals.

**2. The CI/CD Lifecycle**
` + "```" + `
[ Plan ] -> [ Code ] -> [ Build ] -> [ Test ] ──┐
                                               │
[ Monitor ] <- [ Operate ] <- [ Deploy ] <─────┘
` + "```" + `

**3. Key Practices**
*   **Continuous Integration (CI):** Automate builds/tests on every commit.
*   **Continuous Delivery (CD):** Code is always deployable.
*   **Infrastructure as Code (IaC):** Manage servers using code (Terraform/Ansible).
*   **Observability:** Logs, Metrics, and Traces to understand system health.

**4. DORA Metrics (Success Indicators)**
1.  **Deployment Frequency:** How often you ship code.
2.  **Lead Time for Changes:** Time from commit to production.
3.  **Change Failure Rate:** % of deployments that cause a fail.
4.  **MTTR:** Mean Time to Recovery from a failure.`,
					CodeExamples: `# 1. Automated CI (GitHub Actions)
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm install && npm test

# 2. Infrastructure as Code (Terraform)
resource "aws_instance" "web" {
  ami           = "ami-xyz"
  instance_type = "t2.micro"
}`,
				},
				{
					Title: "Historical Evolution",
					Content: `**The Old Way (Silos):**
*   **Dev:** "I built it, now it's Ops' problem."
*   **Ops:** "I won't deploy this, it might break my stable server."
*   **Result:** Slow releases (months), manual errors, and friction.

**The Agile Era:**
Development got faster, but the "Wall of Confusion" remained between Dev and Ops.

**The DevOps Era (Modern):**
*   **Platform Engineering:** Self-service for developers.
*   **GitOps:** Managing infrastructure via PRs.
*   **AIOps:** Using AI to predict issues.`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          251,
			Title:       "Git and Version Control",
			Description: "Master Git workflows, branching strategies, and version control best practices for DevOps.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Git Fundamentals for DevOps",
					Content: `Git is essential for DevOps workflows, enabling version control and collaboration.

**Git Basics:**
- Repository: Project storage
- Commit: Snapshot of changes
- Branch: Independent line of development
- Merge: Combine branches
- Remote: External repository

**Essential Commands:**
- git init: Initialize repository
- git clone: Copy repository
- git add: Stage changes
- git commit: Save changes
- git push: Upload changes
- git pull: Download changes
- git branch: Manage branches
- git merge: Merge branches

**Git Workflows:**
- Centralized: Single main branch
- Feature Branch: Feature branches
- Git Flow: Structured branching
- GitHub Flow: Simple workflow
- GitLab Flow: Environment branches

**Best Practices:**
- Commit often with meaningful messages
- Use branches for features
- Review code before merging
- Keep main branch stable
- Use tags for releases`,
					CodeExamples: `# Initialize repository
git init
git remote add origin https://github.com/user/repo.git

# Basic workflow
git add .
git commit -m "Add feature X"
git push origin main

# Branching
git checkout -b feature/new-feature
# Make changes
git add .
git commit -m "Implement new feature"
git push origin feature/new-feature

# Merge
git checkout main
git merge feature/new-feature
git push origin main

# Git Flow example
git checkout -b develop
git checkout -b feature/user-auth
# Work on feature
git checkout develop
git merge feature/user-auth
git checkout -b release/1.0.0
# Prepare release
git checkout main
git merge release/1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"

# GitHub Flow (simpler)
git checkout -b feature
# Make changes
git push origin feature
# Create PR, merge via GitHub
git checkout main
git pull origin main`,
				},
				{
					Title: "Advanced Git for DevOps",
					Content: `Advanced Git features for complex DevOps scenarios.

**Advanced Features:**
- Rebase: Rewrite commit history
- Cherry-pick: Apply specific commits
- Stash: Temporarily save changes
- Hooks: Automate actions
- Submodules: Nested repositories
- Tags: Mark releases

**Git Hooks:**
- pre-commit: Before commit
- post-commit: After commit
- pre-push: Before push
- post-receive: After receive (server)

**Rebase vs Merge:**
- Rebase: Linear history, cleaner
- Merge: Preserves history, simpler

**Best Practices:**
- Use hooks for automation
- Tag releases
- Use .gitignore properly
- Handle large files with LFS
- Use signed commits for security`,
					CodeExamples: `# Rebase
git checkout feature
git rebase main
# Resolve conflicts if any
git rebase --continue

# Cherry-pick
git cherry-pick <commit-hash>

# Stash
git stash
# Work on other things
git stash pop

# Git hooks
# .git/hooks/pre-commit
#!/bin/sh
npm run lint
npm test

# Tags
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0

# .gitignore
node_modules/
.env
*.log
dist/
build/`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          252,
			Title:       "Linux for DevOps",
			Description: "Essential Linux commands and concepts for DevOps engineers.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Essential Linux Commands",
					Content: `Linux command-line skills are essential for DevOps work.

**File Operations:**
- ls: List files
- cd: Change directory
- pwd: Print working directory
- cp: Copy files
- mv: Move/rename files
- rm: Remove files
- mkdir: Create directory
- find: Search files

**Text Processing:**
- cat: Display file
- less/more: View file page by page
- grep: Search text
- sed: Stream editor
- awk: Text processing
- head/tail: View file parts

**System Information:**
- ps: Process status
- top/htop: Process monitor
- df: Disk space
- free: Memory usage
- uname: System information
- whoami: Current user

**Permissions:**
- chmod: Change permissions
- chown: Change ownership
- sudo: Execute as superuser`,
					CodeExamples: `# File operations
ls -la                    # List all files with details
cd /var/log              # Change directory
pwd                      # Show current directory
cp file.txt backup/      # Copy file
mv old.txt new.txt       # Rename file
rm -rf directory/        # Remove directory recursively
mkdir -p path/to/dir     # Create directory tree

# Text processing
cat file.txt             # Display file
grep "error" log.txt     # Search for "error"
sed 's/old/new/g' file    # Replace text
awk '{print $1}' file    # Print first column
head -n 20 file.txt      # First 20 lines
tail -f log.txt          # Follow log file

# System information
ps aux                   # All processes
top                      # Process monitor
df -h                    # Disk space human-readable
free -h                  # Memory usage
uname -a                 # System info

# Permissions
chmod 755 script.sh      # rwxr-xr-x
chmod +x script.sh       # Add execute permission
chown user:group file    # Change ownership
sudo command             # Run as root`,
				},
				{
					Title: "Linux System Administration",
					Content: `System administration tasks common in DevOps.

**Process Management:**
- Background processes
- Process signals
- Job control
- Service management

**Package Management:**
- apt/yum: Package installers
- Update system packages
- Install software
- Remove packages

**Service Management:**
- systemd: Modern init system
- systemctl: Service control
- journalctl: Log viewing

**Network:**
- netstat/ss: Network connections
- ifconfig/ip: Network configuration
- curl/wget: Download files
- ping: Test connectivity

**File System:**
- mount: Mount filesystems
- fdisk: Disk partitioning
- du: Disk usage
- tar: Archive files`,
					CodeExamples: `# Process management
nohup command &          # Run in background
jobs                     # List background jobs
fg %1                    # Bring job to foreground
kill -9 PID              # Force kill process
killall process_name     # Kill all instances

# Package management (Debian/Ubuntu)
sudo apt update          # Update package list
sudo apt upgrade         # Upgrade packages
sudo apt install nginx   # Install package
sudo apt remove nginx    # Remove package

# Package management (RHEL/CentOS)
sudo yum update          # Update packages
sudo yum install nginx   # Install package
sudo yum remove nginx    # Remove package

# Service management
sudo systemctl start nginx    # Start service
sudo systemctl stop nginx     # Stop service
sudo systemctl restart nginx  # Restart service
sudo systemctl status nginx   # Check status
sudo systemctl enable nginx   # Enable on boot
journalctl -u nginx -f        # View logs

# Network
netstat -tulpn           # Network connections
ss -tulpn                # Modern netstat
curl https://api.com     # HTTP request
wget https://file.zip     # Download file
ping google.com          # Test connectivity

# File system
df -h                    # Disk space
du -sh /path             # Directory size
tar -czf archive.tar.gz dir/  # Create archive
tar -xzf archive.tar.gz       # Extract archive`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          253,
			Title:       "Shell Scripting",
			Description: "Master Bash scripting: automation, error handling, and DevOps automation scripts.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Bash Scripting Basics",
					Content: `Bash scripting automates DevOps tasks and workflows.

**Script Basics:**
- Shebang: #!/bin/bash
- Variables: name="value"
- Arguments: $1, $2, $@
- Exit codes: 0 = success, non-zero = failure
- Comments: # comment

**Control Flow:**
- if/else: Conditional execution
- for/while: Loops
- case: Switch statement
- Functions: Reusable code

**Common Patterns:**
- Error handling: set -e, trap
- Logging: echo, logger
- Input validation
- Configuration files
- Environment variables

**Best Practices:**
- Use set -euo pipefail
- Quote variables
- Check command existence
- Handle errors gracefully
- Use functions for reusability`,
					CodeExamples: `#!/bin/bash
# Basic script
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Variables
NAME="DevOps"
VERSION=1.0

# Functions
function greet() {
    local name=$1
    echo "Hello, $name"
}

greet "World"

# Conditional
if [ "$1" == "deploy" ]; then
    echo "Deploying..."
elif [ "$1" == "test" ]; then
    echo "Testing..."
else
    echo "Usage: $0 [deploy|test]"
    exit 1
fi

# Loop
for file in *.log; do
    echo "Processing $file"
done

# While loop
while true; do
    echo "Running..."
    sleep 5
done

# Error handling
set -e
trap 'echo "Error on line $LINENO"' ERR

# Check command exists
if ! command -v docker &> /dev/null; then
    echo "Docker not found"
    exit 1
fi`,
				},
				{
					Title: "Advanced Shell Scripting",
					Content: `Advanced Bash features for complex automation tasks.

**Advanced Features:**
- Arrays: Multiple values
- Associative arrays: Key-value pairs
- Here documents: Multi-line input
- Process substitution: <(command)
- Command substitution: $(command)
- Parameter expansion: ${var}

**Useful Patterns:**
- Logging functions
- Configuration parsing
- Parallel execution
- Retry logic
- Lock files
- Signal handling

**Best Practices:**
- Use arrays for lists
- Validate input
- Use lock files for mutual exclusion
- Implement retry logic
- Proper logging
- Cleanup on exit`,
					CodeExamples: `#!/bin/bash
set -euo pipefail

# Arrays
FILES=("file1.txt" "file2.txt" "file3.txt")
echo "${FILES[@]}"           # All elements
echo "${#FILES[@]}"          # Length

# Associative arrays
declare -A CONFIG
CONFIG["host"]="localhost"
CONFIG["port"]="8080"
echo "${CONFIG[host]}"

# Here document
cat <<EOF > config.txt
host=localhost
port=8080
EOF

# Command substitution
DATE=$(date +%Y-%m-%d)
echo "Today is $DATE"

# Parameter expansion
FILE="backup.tar.gz"
echo "${FILE%.*}"           # backup.tar
echo "${FILE%%.*}"          # backup
echo "${FILE#*.}"            # tar.gz

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a app.log
}

log "Starting deployment"

# Retry logic
retry() {
    local max_attempts=$1
    shift
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        fi
        echo "Attempt $attempt failed, retrying..."
        attempt=$((attempt + 1))
        sleep 2
    done
    return 1
}

retry 3 curl -f https://api.example.com

# Lock file
LOCK_FILE="/tmp/deploy.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "Deployment already running"
    exit 1
fi
trap "rm -f $LOCK_FILE" EXIT
touch "$LOCK_FILE"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          254,
			Title:       "Docker Fundamentals",
			Description: "Learn Docker: containers, images, Dockerfile, and containerization basics.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Docker Introduction",
					Content: `Docker enables containerization for consistent application deployment.

**What is Docker?**
- Containerization platform
- Package applications with dependencies
- Isolated environments
- Consistent across environments
- Lightweight virtualization

**Key Concepts:**
- **Image**: Read-only template
- **Container**: Running instance of image
- **Dockerfile**: Build instructions
- **Registry**: Image storage (Docker Hub)
- **Volume**: Persistent data storage
- **Network**: Container networking

**Docker vs Virtual Machines:**
- Containers: Share OS kernel, lighter
- VMs: Full OS, heavier, more isolated
- Containers: Faster startup, less overhead
- VMs: Better isolation, more resources

**Benefits:**
- Consistency across environments
- Fast deployment
- Resource efficiency
- Easy scaling
- Version control for environments`,
					CodeExamples: `# Docker commands
docker --version          # Check version
docker info               # System information
docker ps                 # Running containers
docker ps -a              # All containers
docker images             # List images

# Pull and run image
docker pull nginx:latest
docker run -d -p 80:80 nginx

# Build image
docker build -t myapp:1.0 .

# Container management
docker start container_id
docker stop container_id
docker restart container_id
docker rm container_id    # Remove container
docker rmi image_id       # Remove image

# View logs
docker logs container_id
docker logs -f container_id  # Follow logs

# Execute commands in container
docker exec -it container_id /bin/bash
docker exec container_id ls /app

# Inspect
docker inspect container_id
docker stats              # Resource usage`,
				},
				{
					Title: "Dockerfile and Image Building",
					Content: `Dockerfile defines how to build Docker images.

**Dockerfile Instructions:**
- FROM: Base image
- RUN: Execute commands
- COPY/ADD: Copy files
- WORKDIR: Set working directory
- ENV: Environment variables
- EXPOSE: Document ports
- CMD: Default command
- ENTRYPOINT: Entry point

**Best Practices:**
- Use specific base image tags
- Minimize layers
- Use .dockerignore
- Order instructions by change frequency
- Use multi-stage builds
- Don't run as root
- Use health checks

**Multi-stage Builds:**
- Build stage: Compile application
- Runtime stage: Minimal runtime image
- Smaller final image
- Better security`,
					CodeExamples: `# Basic Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

USER node

CMD ["node", "server.js"]

# Multi-stage build
# Stage 1: Build
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Runtime
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./
EXPOSE 3000
CMD ["node", "dist/server.js"]

# .dockerignore
node_modules/
.git/
.env
*.log
dist/

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:3000/health || exit 1`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          255,
			Title:       "Docker Compose",
			Description: "Orchestrate multi-container applications with Docker Compose.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Docker Compose Fundamentals",
					Content: `Docker Compose orchestrates multi-container Docker applications.

**What is Docker Compose?**
- Define multi-container apps
- YAML configuration file
- Single command to start/stop
- Service dependencies
- Networking between containers

**Key Concepts:**
- **Service**: Container definition
- **Network**: Container networking
- **Volume**: Shared storage
- **Environment**: Variables
- **Dependencies**: Service ordering

**Common Use Cases:**
- Development environments
- Microservices applications
- Database + application
- Full stack applications
- Testing environments

**Benefits:**
- Simple configuration
- Easy local development
- Reproducible environments
- Service orchestration`,
					CodeExamples: `# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      - db
    volumes:
      - ./app:/app
    networks:
      - app-network

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - app-network

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    networks:
      - app-network

volumes:
  postgres-data:

networks:
  app-network:
    driver: bridge

# Commands
docker-compose up              # Start services
docker-compose up -d           # Start in background
docker-compose down            # Stop and remove
docker-compose ps              # List services
docker-compose logs            # View logs
docker-compose logs -f web     # Follow logs
docker-compose exec web bash   # Execute command
docker-compose build           # Build images
docker-compose restart web     # Restart service`,
				},
				{
					Title: "Advanced Docker Compose",
					Content: `Advanced Docker Compose features and production patterns.

**Advanced Features:**
- **Override Files**: Use multiple compose files (docker-compose.override.yml)
- **Environment Files**: Externalize configuration (.env files)
- **Profiles**: Conditional service activation
- **Health Checks**: Service health monitoring
- **Resource Limits**: CPU and memory constraints
- **Restart Policies**: Container restart behavior
- **Secrets Management**: Secure credential handling

**Production Patterns:**
- Separate dev/prod configurations
- Use environment-specific override files
- Implement health checks for all services
- Set resource limits
- Use secrets for sensitive data
- Configure logging drivers
- Use external networks and volumes

**Best Practices:**
- Version control compose files
- Use .env files for environment variables
- Separate concerns with profiles
- Implement proper health checks
- Set appropriate resource limits
- Use named volumes for persistence
- Document service dependencies

**Common Pitfalls:**
- Hardcoding values in compose files
- Not using health checks
- Missing resource limits
- Exposing unnecessary ports
- Not using secrets for sensitive data
- Mixing dev and prod configurations`,
					CodeExamples: `# docker-compose.yml (base)
version: '3.8'

services:
  web:
    build: .
    ports:
      - "${WEB_PORT:-3000}:3000"
    environment:
      - NODE_ENV=${NODE_ENV:-development}
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  db:
    image: postgres:14
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1G
    secrets:
      - db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres-data:
    driver: local

# docker-compose.prod.yml (production override)
version: '3.8'

services:
  web:
    environment:
      - NODE_ENV=production
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        max_attempts: 3
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://logs.example.com:514"

# docker-compose.dev.yml (development override)
version: '3.8'

services:
  web:
    volumes:
      - ./src:/app/src
      - ./node_modules:/app/node_modules
    environment:
      - DEBUG=true
    command: npm run dev

  db:
    ports:
      - "5432:5432"

# Using profiles
services:
  web:
    # ... base config
  worker:
    profiles:
      - worker
    # ... worker config
  redis:
    profiles:
      - cache
    # ... redis config

# Run with profiles
docker-compose --profile worker --profile cache up

# .env file example
WEB_PORT=3000
NODE_ENV=production
DB_NAME=myapp
DB_USER=admin
DB_PASSWORD=secure_password

# Commands with override files
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
docker-compose --env-file .env.production up`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          256,
			Title:       "CI/CD Basics",
			Description: "Introduction to Continuous Integration and Continuous Deployment: concepts, pipelines, and automation.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "CI/CD Concepts",
					Content: `CI/CD automates the software delivery process.

**Continuous Integration (CI):**
- Merge code frequently
- Automated testing
- Early bug detection
- Build automation
- Code quality checks

**Continuous Deployment (CD):**
- Automated deployment
- Deploy to production automatically
- Manual approval gates (Continuous Delivery)
- Rollback capabilities
- Environment promotion

**CI/CD Pipeline Stages:**
1. **Source**: Code repository
2. **Build**: Compile and package
3. **Test**: Run automated tests
4. **Deploy**: Deploy to environments
5. **Monitor**: Monitor deployment

**Benefits:**
- Faster releases
- Higher quality
- Reduced manual work
- Consistent deployments
- Faster feedback

**Common Tools:**
- Jenkins: Self-hosted CI/CD
- GitHub Actions: GitHub-native
- GitLab CI: GitLab-native
- CircleCI: Cloud CI/CD
- Travis CI: Cloud CI/CD`,
					CodeExamples: `# GitHub Actions example
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test
      
      - name: Build
        run: npm run build
      
      - name: Deploy
        if: github.ref == 'refs/heads/main'
        run: npm run deploy

# GitLab CI example
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - npm install
    - npm run build
  artifacts:
    paths:
      - dist/

test:
  stage: test
  script:
    - npm test

deploy:
  stage: deploy
  script:
    - npm run deploy
  only:
    - main`,
				},
				{
					Title: "Building CI/CD Pipelines",
					Content: `Design effective CI/CD pipelines for your applications.

**Pipeline Design:**
- Fast feedback: Fail fast
- Parallel execution: Speed up
- Caching: Reuse dependencies
- Artifacts: Pass between stages
- Environments: Dev, staging, prod

**Best Practices:**
- Keep pipelines fast
- Test before deploy
- Use environment variables
- Implement rollback
- Monitor deployments
- Security scanning

**Common Patterns:**
- Feature branch pipelines
- Pull request pipelines
- Main branch deployment
- Blue-green deployment
- Canary deployments`,
					CodeExamples: `# Advanced GitHub Actions
name: Full CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:

env:
  NODE_VERSION: '18'
  REGISTRY: ghcr.io

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
      - uses: actions/cache@v3
        with:
          path: ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
      - run: npm ci
      - run: npm test
      - run: npm run lint

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t ${{ env.REGISTRY }}/app:${{ github.sha }} .
      - name: Push to registry
        run: docker push ${{ env.REGISTRY }}/app:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/app app=${{ env.REGISTRY }}/app:${{ github.sha }}
          kubectl rollout status deployment/app`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          257,
			Title:       "Build Tools and Package Managers",
			Description: "Master build tools and package managers: npm, Maven, Gradle, and build automation.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Package Managers Overview",
					Content: `Package managers handle dependencies and builds for different languages.

**Language-Specific Package Managers:**
- **npm/yarn/pnpm**: JavaScript/Node.js
- **Maven/Gradle**: Java
- **pip/poetry**: Python
- **Composer**: PHP
- **NuGet**: .NET
- **Cargo**: Rust
- **Go modules**: Go

**Key Concepts:**
- Dependencies: External packages
- Lock files: Version pinning
- Registries: Package repositories
- Semantic versioning: Version numbers
- Dependency resolution: Conflict handling

**Build Tools:**
- Compile source code
- Run tests
- Package artifacts
- Generate documentation
- Publish packages

**Best Practices:**
- Use lock files
- Pin dependency versions
- Regular updates
- Security scanning
- Reproducible builds`,
					CodeExamples: `# npm (Node.js)
npm init                 # Initialize project
npm install express      # Install package
npm install -D jest      # Install dev dependency
npm install --save-exact # Exact version
npm update               # Update packages
npm audit                # Security audit
npm audit fix            # Fix vulnerabilities

# package.json
{
    "name": "myapp",
    "version": "1.0.0",
    "scripts": {
        "start": "node server.js",
        "test": "jest",
        "build": "webpack"
    },
    "dependencies": {
        "express": "^4.18.0"
    },
    "devDependencies": {
        "jest": "^29.0.0"
    }
}

# Maven (Java)
# pom.xml
<project>
    <dependencies>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-core</artifactId>
            <version>5.3.21</version>
        </dependency>
    </dependencies>
</project>

mvn clean install        # Build and install
mvn test                 # Run tests
mvn package              # Create JAR

# Gradle (Java/Kotlin)
# build.gradle
plugins {
    id 'java'
}

dependencies {
    implementation 'org.springframework:spring-core:5.3.21'
    testImplementation 'junit:junit:4.13.2'
}

gradle build             # Build
gradle test              # Test
gradle publish           # Publish`,
				},
				{
					Title: "Advanced Build Automation",
					Content: `Advanced build automation techniques and CI/CD integration.

**Build Automation Concepts:**
- **Incremental Builds**: Only rebuild changed components
- **Parallel Builds**: Build multiple modules simultaneously
- **Caching**: Cache dependencies and build artifacts
- **Build Artifacts**: Store and version build outputs
- **Multi-stage Builds**: Separate build and runtime environments
- **Build Scripts**: Automate complex build processes

**CI/CD Integration:**
- Trigger builds on code changes
- Run tests automatically
- Generate and publish artifacts
- Version artifacts with build numbers
- Deploy artifacts to repositories
- Notify on build failures

**Build Optimization:**
- Use build caches
- Parallel execution
- Incremental compilation
- Dependency caching
- Artifact caching
- Build time monitoring

**Best Practices:**
- Use build tools consistently
- Cache dependencies
- Version artifacts properly
- Store artifacts in repositories
- Monitor build times
- Fail fast on errors
- Generate build reports

**Common Pitfalls:**
- Not caching dependencies
- Building unnecessary components
- Not versioning artifacts
- Missing build verification
- Slow build processes
- Inconsistent build environments`,
					CodeExamples: `# npm build script with caching
# .npmrc
cache=/tmp/.npm
package-lock=true

# package.json scripts
{
  "scripts": {
    "prebuild": "npm run lint",
    "build": "webpack --mode production",
    "postbuild": "npm run test:build",
    "build:ci": "npm ci && npm run build",
    "build:cache": "npm ci --cache /tmp/.npm && npm run build"
  }
}

# Maven with caching and parallel builds
# .mvn/maven.config
-T 4
-Dmaven.build.cache.enabled=true

# pom.xml with build profiles
<profiles>
  <profile>
    <id>ci</id>
    <properties>
      <skipTests>false</skipTests>
    </properties>
  </profile>
  <profile>
    <id>fast</id>
    <properties>
      <maven.test.skip>true</maven.test.skip>
    </properties>
  </profile>
</profiles>

# Build with profile
mvn clean install -Pci -T 4

# Gradle build with caching
# gradle.properties
org.gradle.caching=true
org.gradle.parallel=true
org.gradle.configureondemand=true

# build.gradle with build cache
buildCache {
    local {
        enabled = true
        directory = new File(rootDir, '.gradle/build-cache')
    }
}

# Multi-stage Docker build (build automation)
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
CMD ["node", "dist/server.js"]

# Build script example (Bash)
#!/bin/bash
set -e

BUILD_DIR="build"
ARTIFACT_DIR="artifacts"
VERSION=$(git describe --tags --always)

echo "Building version $VERSION"

# Clean previous builds
rm -rf $BUILD_DIR $ARTIFACT_DIR
mkdir -p $BUILD_DIR $ARTIFACT_DIR

# Install dependencies
npm ci

# Run tests
npm test

# Build
npm run build

# Package artifacts
tar -czf $ARTIFACT_DIR/app-$VERSION.tar.gz -C $BUILD_DIR .

# Upload to artifact repository
# aws s3 cp $ARTIFACT_DIR/app-$VERSION.tar.gz s3://artifacts/myapp/

echo "Build complete: $ARTIFACT_DIR/app-$VERSION.tar.gz"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          258,
			Title:       "Configuration Management Basics",
			Description: "Introduction to configuration management: Ansible, Puppet, Chef basics.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Configuration Management Concepts",
					Content: `Configuration management automates server and application configuration.

**What is Configuration Management?**
- Automate server setup
- Ensure consistency
- Version control infrastructure
- Idempotent operations
- Repeatable deployments

**Key Concepts:**
- **Idempotency**: Same result every time
- **Declarative**: Describe desired state
- **Agent vs Agentless**: How tools connect
- **Push vs Pull**: Configuration delivery

**Popular Tools:**
- **Ansible**: Agentless, YAML-based
- **Puppet**: Agent-based, declarative
- **Chef**: Agent-based, Ruby DSL
- **SaltStack**: Agent-based, Python

**Use Cases:**
- Server provisioning
- Application deployment
- Configuration updates
- Compliance enforcement
- Disaster recovery`,
					CodeExamples: `# Ansible playbook
# playbook.yml
- hosts: webservers
  become: yes
  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
    
    - name: Start nginx
      systemd:
        name: nginx
        state: started
        enabled: yes
    
    - name: Copy config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: restart nginx
  
  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted

# Run playbook
ansible-playbook playbook.yml

# Ansible inventory
# inventory.ini
[webservers]
web1 ansible_host=192.168.1.10
web2 ansible_host=192.168.1.11

[databases]
db1 ansible_host=192.168.1.20`,
				},
				{
					Title: "Ansible Basics",
					Content: `Get started with Ansible for configuration management.

**Ansible Overview:**
- Agentless architecture (uses SSH)
- YAML-based playbooks
- Idempotent operations
- Simple and powerful
- Large module ecosystem

**Core Components:**
- **Inventory**: List of managed hosts
- **Playbooks**: Automation scripts (YAML)
- **Tasks**: Individual actions
- **Modules**: Reusable units of work
- **Handlers**: Tasks triggered by events
- **Variables**: Dynamic values
- **Templates**: Configuration files with variables

**Basic Concepts:**
- **Ad-hoc Commands**: Quick one-liners
- **Playbooks**: Reusable automation
- **Roles**: Reusable playbook components
- **Variables**: Dynamic configuration
- **Facts**: System information

**Common Modules:**
- **apt/yum**: Package management
- **systemd**: Service management
- **copy/template**: File operations
- **user/group**: User management
- **file**: File/directory management
- **command/shell**: Execute commands

**Best Practices:**
- Use roles for reusability
- Keep playbooks simple
- Use variables for flexibility
- Test playbooks with --check
- Use handlers for notifications
- Document your playbooks

**Common Pitfalls:**
- Not using idempotent modules
- Hardcoding values
- Not testing playbooks
- Ignoring errors
- Not using roles`,
					CodeExamples: `# Ad-hoc commands
ansible all -m ping
ansible webservers -m apt -a "name=nginx state=present" --become
ansible all -a "uptime"
ansible webservers -m copy -a "src=/local/file dest=/remote/file"

# Basic playbook
# playbook.yml
- name: Setup web server
  hosts: webservers
  become: yes
  vars:
    nginx_port: 80
    app_user: www-data
  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
    
    - name: Install nginx
      apt:
        name: nginx
        state: present
    
    - name: Create app directory
      file:
        path: /var/www/myapp
        state: directory
        owner: "{{ app_user }}"
        group: "{{ app_user }}"
        mode: '0755'
    
    - name: Copy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/myapp
        owner: root
        group: root
        mode: '0644'
      notify: restart nginx
    
    - name: Enable site
      file:
        src: /etc/nginx/sites-available/myapp
        dest: /etc/nginx/sites-enabled/myapp
        state: link
      notify: restart nginx
    
    - name: Start and enable nginx
      systemd:
        name: nginx
        state: started
        enabled: yes
  
  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted

# Template file (nginx.conf.j2)
server {
    listen {{ nginx_port }};
    server_name {{ server_name }};
    root /var/www/myapp;
    
    location / {
        try_files $uri $uri/ =404;
    }
}

# Inventory with variables
# inventory.ini
[webservers]
web1 ansible_host=192.168.1.10 nginx_port=80
web2 ansible_host=192.168.1.11 nginx_port=8080

[webservers:vars]
server_name=example.com

# Using variables
- name: Display variable
  debug:
    msg: "Server port is {{ nginx_port }}"

# Conditional tasks
- name: Install package on Debian
  apt:
    name: nginx
  when: ansible_os_family == "Debian"

- name: Install package on RedHat
  yum:
    name: nginx
  when: ansible_os_family == "RedHat"

# Loops
- name: Install packages
  apt:
    name: "{{ item }}"
  loop:
    - nginx
    - mysql-server
    - redis-server

# Error handling
- name: Try to start service
  systemd:
    name: myservice
    state: started
  ignore_errors: yes
  register: result

- name: Check if service started
  debug:
    msg: "Service started successfully"
  when: result.failed == false

# Run playbook
ansible-playbook playbook.yml -i inventory.ini
ansible-playbook playbook.yml --check  # Dry run
ansible-playbook playbook.yml --limit webservers`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          259,
			Title:       "Monitoring Fundamentals",
			Description: "Introduction to monitoring: logs, metrics, alerts, and observability basics.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Monitoring Basics",
					Content: `Monitoring provides visibility into system and application health.

**Monitoring Types:**
- **Logs**: Event records
- **Metrics**: Numerical measurements
- **Traces**: Request journeys
- **Alerts**: Notifications

**Key Metrics:**
- **Availability**: Uptime percentage
- **Latency**: Response time
- **Throughput**: Requests per second
- **Error Rate**: Failed requests
- **Resource Usage**: CPU, memory, disk

**Monitoring Stack:**
- **Collection**: Agents, exporters
- **Storage**: Time-series databases
- **Visualization**: Dashboards (Grafana)
- **Alerting**: Alert managers

**Best Practices:**
- Monitor everything
- Set meaningful alerts
- Use dashboards
- Log structured data
- Track business metrics`,
					CodeExamples: `# Application logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("User logged in", extra={"user_id": 123})

# Prometheus metrics
from prometheus_client import Counter, Histogram

requests_total = Counter('http_requests_total', 'Total requests')
request_duration = Histogram('http_request_duration_seconds', 'Request duration')

@request_duration.time()
def handle_request():
    requests_total.inc()
    # Handle request

# Health check endpoint
@app.route('/health')
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# System metrics (Linux)
top                    # CPU and memory
iostat -x 1           # Disk I/O
netstat -i            # Network interfaces
free -h                # Memory usage`,
				},
				{
					Title: "Logging and Metrics Collection",
					Content: `Effective logging and metrics collection strategies.

**Logging Best Practices:**
- Structured logging (JSON)
- Log levels (DEBUG, INFO, WARN, ERROR)
- Include context (request ID, user ID)
- Don't log sensitive data
- Centralized logging

**Metrics Collection:**
- Application metrics
- Infrastructure metrics
- Business metrics
- Custom metrics

**Tools:**
- **Logging**: ELK Stack, Loki, Splunk
- **Metrics**: Prometheus, Datadog, New Relic
- **Tracing**: Jaeger, Zipkin
- **Dashboards**: Grafana

**Common Patterns:**
- Log aggregation
- Metric scraping
- Alert rules
- Dashboard creation`,
					CodeExamples: `# Structured logging
{
    "timestamp": "2024-01-17T10:00:00Z",
    "level": "INFO",
    "service": "user-service",
    "request_id": "abc123",
    "message": "User created",
    "user_id": 123,
    "duration_ms": 45
}

# Prometheus exporter
from prometheus_client import start_http_server, Gauge

cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')

def collect_metrics():
    cpu_usage.set(get_cpu_usage())
    # Collect other metrics

start_http_server(8000)
while True:
    collect_metrics()
    time.sleep(10)

# Grafana dashboard query
rate(http_requests_total[5m])

# Alert rule (Prometheus)
groups:
  - name: alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
