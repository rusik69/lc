.PHONY: build run dev dev-local test clean help docker-build docker-up docker-down docker-restart docker-logs docker-sandbox

# Build the application
build:
	@echo "Building application..."
	@go mod download
	@mkdir -p bin
	@go build -o bin/lc ./cmd/lc

# Run the application
run: build
	@echo "Running application..."
	@./bin/lc

# Development mode: full Docker clean rebuild and restart
dev: docker-down docker-build-no-cache docker-up
	@echo ""
	@echo "=== Verifying containers are running ==="
	@sleep 2
	@if docker ps --format "{{.Names}}" | grep -qE "(lc_app|lc-sandbox)"; then \
		echo "✓ Containers are running"; \
		echo ""; \
		docker ps --filter "name=lc" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
		echo ""; \
		echo "Application should be available at http://localhost:8080"; \
		echo "View logs with: make docker-logs"; \
	else \
		echo "✗ Error: Containers failed to start"; \
		echo "Checking logs..."; \
		if command -v docker-compose > /dev/null 2>&1; then \
			docker-compose logs --tail=50; \
		elif docker compose version > /dev/null 2>&1; then \
			docker compose logs --tail=50; \
		fi; \
		exit 1; \
	fi

# Development mode: run locally with sandbox container
dev-local: docker-sandbox
	@echo "Building application..."
	@go build -o bin/lc ./cmd/lc
	@echo "Starting application on http://localhost:8080"
	@SANDBOX_CONTAINER=lc-sandbox ./bin/lc

# Ensure sandbox container is running (for local development)
docker-sandbox:
	@echo "=== Checking sandbox container ==="
	@if ! docker info > /dev/null 2>&1; then \
		echo "Error: Docker is not running"; \
		exit 1; \
	fi
	@if ! docker ps | grep -q lc-sandbox; then \
		echo "Starting sandbox container..."; \
		docker rm -f lc-sandbox 2>/dev/null || true; \
		docker run -d \
			--name lc-sandbox \
			--memory 512m \
			--cpus 1.5 \
			--pids-limit 100 \
			--security-opt no-new-privileges:true \
			--cap-drop ALL \
			--tmpfs /tmp:rw,exec,size=200m \
			--tmpfs /go:rw,exec,size=300m \
			golang:1.23-alpine \
			sleep infinity; \
		echo "✓ Sandbox container started"; \
	else \
		echo "✓ Sandbox container already running"; \
	fi

# Run tests in docker-compose
test:
	@echo "Running tests in docker-compose..."
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose run --rm --profile test test; \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose run --rm --profile test test; \
	else \
		echo "✗ Error: docker-compose not found"; \
		exit 1; \
	fi

# Run only unit tests (fast) in docker-compose
test-unit:
	@echo "Running unit tests in docker-compose..."
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose run --rm --profile test test go test -v -short ./tests/... ./internal/...; \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose run --rm --profile test test go test -v -short ./tests/... ./internal/...; \
	else \
		echo "✗ Error: docker-compose not found"; \
		exit 1; \
	fi

# Run e2e tests (requires Docker) in docker-compose
test-e2e:
	@echo "Running e2e tests in docker-compose..."
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose run --rm --profile test test go test -v ./tests/... -run E2E; \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose run --rm --profile test test go test -v ./tests/... -run E2E; \
	else \
		echo "✗ Error: docker-compose not found"; \
		exit 1; \
	fi

# Run tests with coverage report
test-coverage: test
	@echo "Generating coverage report..."
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose run --rm --profile test test go tool cover -html=coverage.out -o coverage.html; \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose run --rm --profile test test go tool cover -html=coverage.out -o coverage.html; \
	else \
		go tool cover -html=coverage.out -o coverage.html; \
	fi
	@echo "Coverage report generated: coverage.html"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin/ coverage.out coverage.html

# Run linter
lint:
	@echo "Running linter..."
	@golangci-lint run || echo "Install golangci-lint: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...

# Vet code
vet:
	@echo "Running go vet..."
	@go vet ./...

# Run all checks (fmt, vet, lint, test)
check: fmt vet test

# Docker: Build image
docker-build:
	@echo "Building Docker image..."
	@echo "Ensuring Go 1.23 base image is available..."
	@docker pull golang:1.23-alpine || true
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose build || (echo "✗ Docker build failed" && exit 1); \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose build || (echo "✗ Docker build failed" && exit 1); \
	else \
		echo "✗ Error: docker-compose not found. Install docker-compose or Docker Desktop."; \
		exit 1; \
	fi
	@echo "✓ Docker image built successfully"

# Docker: Build image without cache (for troubleshooting)
docker-build-no-cache:
	@echo "Building Docker image without cache..."
	@docker pull golang:1.23-alpine
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose build --no-cache || (echo "✗ Docker build failed" && exit 1); \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose build --no-cache || (echo "✗ Docker build failed" && exit 1); \
	else \
		echo "✗ Error: docker-compose not found"; \
		exit 1; \
	fi
	@echo "✓ Docker image built successfully"

# Docker: Start containers
docker-up:
	@echo "Cleaning up any conflicting containers..."
	@docker rm -f lc-sandbox 2>/dev/null || true
	@echo "Starting containers..."
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose up -d || (echo "✗ Failed to start containers" && exit 1); \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose up -d || (echo "✗ Failed to start containers" && exit 1); \
	else \
		echo "✗ Error: docker-compose not found. Install docker-compose or Docker Desktop."; \
		exit 1; \
	fi
	@echo "✓ Containers started"
	@echo "Application running at http://localhost:8080"

# Docker: Stop containers
docker-down:
	@echo "Stopping containers..."
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose down; \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose down; \
	else \
		echo "✗ Error: docker-compose not found"; \
		exit 1; \
	fi
	@echo "Cleaning up standalone sandbox container if exists..."
	@docker rm -f lc-sandbox 2>/dev/null || true

# Docker: Restart containers
docker-restart:
	@echo "Restarting containers..."
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose restart; \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose restart; \
	else \
		echo "✗ Error: docker-compose not found"; \
		exit 1; \
	fi

# Docker: View logs
docker-logs:
	@if command -v docker-compose > /dev/null 2>&1; then \
		docker-compose logs -f; \
	elif docker compose version > /dev/null 2>&1; then \
		docker compose logs -f; \
	else \
		echo "✗ Error: docker-compose not found"; \
		exit 1; \
	fi

# Docker: Full rebuild and restart (fast, uses cache)
docker-rebuild: docker-down docker-build docker-up

# Docker: Full clean rebuild and restart (no cache)
docker-rebuild-clean: docker-down docker-build-no-cache docker-up

# Deployment: Deploy to production server via SSH
deploy:
	@echo "=== Deploying to production server ==="
	@if [ ! -f .prod.env ]; then \
		echo "✗ Error: .prod.env file not found"; \
		echo "Create .prod.env with ADMIN_PASSWORD=your_password"; \
		exit 1; \
	fi
	@echo "Copying files to server..."
	@ssh root@msk.govno2.cloud 'mkdir -p /root/lc'
	@FILES="Dockerfile docker-compose.yml .prod.env go.mod cmd/ internal/ web/"; \
	if [ -f go.sum ]; then FILES="$$FILES go.sum"; fi; \
	tar czf /tmp/lc-deploy.tar.gz $$FILES
	@scp /tmp/lc-deploy.tar.gz root@msk.govno2.cloud:/root/lc/
	@ssh root@msk.govno2.cloud 'cd /root/lc && tar xzf lc-deploy.tar.gz && rm lc-deploy.tar.gz'
	@rm -f /tmp/lc-deploy.tar.gz
	@echo "Building and deploying on server..."
	@ssh root@msk.govno2.cloud 'cd /root/lc && \
		echo "Stopping existing containers..." && \
		if command -v docker-compose > /dev/null 2>&1; then \
			docker-compose down 2>/dev/null || true; \
		elif docker compose version > /dev/null 2>&1; then \
			docker compose down 2>/dev/null || true; \
		fi && \
		echo "Cleaning up Docker resources..." && \
		docker container prune -f 2>/dev/null || true && \
		docker image prune -af 2>/dev/null || true && \
		docker volume prune -f 2>/dev/null || true && \
		docker system prune -af --volumes 2>/dev/null || true && \
		docker builder prune -af 2>/dev/null || true && \
		if command -v docker-compose > /dev/null 2>&1; then \
			echo "Building Docker image..." && \
			docker-compose build && \
			echo "Starting containers..." && \
			docker-compose up -d; \
		elif docker compose version > /dev/null 2>&1; then \
			echo "Building Docker image..." && \
			docker compose build && \
			echo "Starting containers..." && \
			docker compose up -d; \
		else \
			echo "Error: docker-compose not found"; \
			exit 1; \
		fi'
	@echo "✓ Deployment complete!"
	@echo "Application should be available at http://msk.govno2.cloud:8080"

# Help target
help:
	@echo "Available targets:"
	@echo "  build            - Build the application (local)"
	@echo "  run              - Build and run the application (local)"
	@echo "  dev              - Full Docker clean rebuild and restart (no cache)"
	@echo "  dev-local        - Run locally with sandbox container"
	@echo "  test             - Run tests"
	@echo "  test-unit        - Run unit tests only (fast)"
	@echo "  test-e2e         - Run e2e tests (requires Docker)"
	@echo "  test-coverage    - Run tests and generate coverage report"
	@echo "  lint             - Run linter"
	@echo "  fmt              - Format code"
	@echo "  vet              - Run go vet"
	@echo "  check            - Run all checks (fmt, vet, test)"
	@echo "  clean            - Clean build artifacts"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build        - Build Docker image"
	@echo "  docker-build-no-cache - Build Docker image without cache"
	@echo "  docker-up           - Start containers"
	@echo "  docker-down         - Stop containers"
	@echo "  docker-restart      - Restart containers (without rebuild)"
	@echo "  docker-logs         - View container logs"
	@echo "  docker-rebuild      - Full rebuild and restart (fast, uses cache)"
	@echo "  docker-rebuild-clean - Full clean rebuild and restart (no cache)"
	@echo "  docker-sandbox      - Ensure sandbox container is running"
	@echo ""
	@echo "Deployment targets:"
	@echo "  deploy             - Deploy to production server via SSH"
	@echo ""
	@echo "  help             - Show this help message"
