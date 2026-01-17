# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Install dependencies (including docker CLI for tests)
RUN apk update && apk add --no-cache git make docker-cli

# Copy go mod files
COPY go.mod go.sum* ./
RUN go mod download || true

# Copy source code
COPY . .

# Build the application
ENV GOTOOLCHAIN=auto
RUN go build -o /app/bin/lc ./cmd/lc

# Runtime stage
FROM golang:1.23-alpine

WORKDIR /app

# Install docker CLI and Python 3 for executing code in sandbox
# Ensure community repository is available and update package index
RUN apk update && apk add --no-cache docker-cli ca-certificates python3 py3-pip

# Copy built binary
COPY --from=builder /app/bin/lc /app/lc

# Copy web assets (templates and static files)
COPY web/ /app/web/

# Expose port
EXPOSE 8080

# Create directory for temporary code execution
RUN mkdir -p /tmp/code

# Run the application
CMD ["/app/lc"]
