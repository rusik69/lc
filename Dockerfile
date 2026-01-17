# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Install dependencies
RUN apk add --no-cache git make

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

# Install docker CLI for executing code in sandbox
RUN apk add --no-cache docker-cli ca-certificates

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
