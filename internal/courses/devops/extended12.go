package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1472,
			Title:       "Performance Engineering and Load Testing",
			Description: "Design and execute performance tests, analyze results, and implement optimizations for system throughput, latency, and scalability.",
			Order:       72,
			Lessons: []problems.Lesson{
				{
					Title: "Load Testing Strategies and Tools",
					Content: `Performance engineering ensures systems meet performance requirements under expected and peak loads.

**Load Testing Types:**
` + "```" + `
Testing types:
  Smoke Test:
    Minimal load to verify system works.
    1-5 virtual users for a few minutes.
    Baseline validation.
  
  Load Test:
    Expected normal and peak load.
    Verify SLAs under typical conditions.
    Gradually ramp up to target load.
  
  Stress Test:
    Push beyond normal capacity.
    Find breaking points.
    Identify failure modes.
  
  Spike Test:
    Sudden large increases in load.
    Test auto-scaling response.
    Verify recovery after spike.
  
  Soak Test (Endurance):
    Normal load for extended period.
    Detect memory leaks, connection leaks.
    Verify stability over hours/days.
  
  Breakpoint Test:
    Incrementally increase load.
    Find maximum capacity.
    Identify bottlenecks.

k6 (Grafana):
  Modern load testing tool.
  JavaScript-based test scripts.
  
  Basic test:
    import http from 'k6/http';
    import { check, sleep } from 'k6';
    
    export const options = {
      stages: [
        { duration: '2m', target: 100 },  // Ramp up
        { duration: '5m', target: 100 },  // Stay
        { duration: '2m', target: 200 },  // Stress
        { duration: '5m', target: 200 },  // Stay
        { duration: '2m', target: 0 },    // Ramp down
      ],
      thresholds: {
        http_req_duration: ['p(95)<500'],  // 95th percentile < 500ms
        http_req_failed: ['rate<0.01'],    // Error rate < 1%
        http_reqs: ['rate>100'],           // Throughput > 100 RPS
      },
    };
    
    export default function() {
      const res = http.get('https://api.example.com/items');
      check(res, {
        'status is 200': (r) => r.status === 200,
        'response time < 500ms': (r) => r.timings.duration < 500,
      });
      sleep(1);
    }
  
  Advanced scenarios:
    import http from 'k6/http';
    import { check, group, sleep } from 'k6';
    import { SharedArray } from 'k6/data';
    
    const users = new SharedArray('users', function() {
      return JSON.parse(open('./users.json'));
    });
    
    export default function() {
      const user = users[__VU % users.length];
      
      group('login', function() {
        const loginRes = http.post('https://api.example.com/login', 
          JSON.stringify({ email: user.email, password: user.password }),
          { headers: { 'Content-Type': 'application/json' } }
        );
        check(loginRes, { 'login success': (r) => r.status === 200 });
        
        const token = loginRes.json('token');
        
        group('browse items', function() {
          const items = http.get('https://api.example.com/items', {
            headers: { 'Authorization': 'Bearer ' + token },
          });
          check(items, { 'items loaded': (r) => r.status === 200 });
        });
        
        group('create order', function() {
          const order = http.post('https://api.example.com/orders',
            JSON.stringify({ item_id: 1, quantity: 1 }),
            { headers: { 
              'Content-Type': 'application/json',
              'Authorization': 'Bearer ' + token,
            }}
          );
          check(order, { 'order created': (r) => r.status === 201 });
        });
      });
      
      sleep(Math.random() * 3 + 1);
    }
  
  Run:
    k6 run test.js
    k6 run --vus 100 --duration 5m test.js
    k6 run --out influxdb=http://localhost:8086/k6 test.js
    k6 run --out cloud test.js  # Grafana Cloud k6

Locust (Python):
  from locust import HttpUser, task, between
  
  class WebUser(HttpUser):
      wait_time = between(1, 5)
      
      @task(3)
      def view_items(self):
          self.client.get("/items")
      
      @task(1)
      def create_order(self):
          self.client.post("/orders",
              json={"item_id": 1, "quantity": 1})
      
      def on_start(self):
          self.client.post("/login",
              json={"email": "test@example.com", "password": "test"})
  
  Run:
    locust -f locustfile.py --host=https://api.example.com
    locust -f locustfile.py --headless --users 100 --spawn-rate 10

Vegeta (Go):
  echo "GET https://api.example.com/items" | \
    vegeta attack -duration=60s -rate=100/s | \
    vegeta report
  
  echo "GET https://api.example.com/items" | \
    vegeta attack -duration=60s -rate=100/s | \
    vegeta plot > results.html

wrk:
  wrk -t12 -c400 -d30s https://api.example.com/items
  wrk -t12 -c400 -d30s -s script.lua https://api.example.com
` + "```" + ``,
					CodeExamples: `# Performance testing scripts

# 1. Load test runner with reporting
#!/bin/bash
set -e

TARGET="${1:?Usage: $0 <target-url> [test-type]}"
TEST_TYPE="${2:-load}"
RESULTS_DIR="./perf-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== Performance Test ==="
echo "Target: $TARGET"
echo "Type: $TEST_TYPE"
echo "Results: $RESULTS_DIR"

# Check tools
if ! command -v k6 &>/dev/null; then
    echo "k6 not found. Install: brew install k6"
    exit 1
fi

# Create test script based on type
case "$TEST_TYPE" in
    smoke)
        cat > "$RESULTS_DIR/test.js" << 'EOF'
import http from 'k6/http';
import { check } from 'k6';
export const options = {
  vus: 3,
  duration: '1m',
  thresholds: {
    http_req_duration: ['p(99)<1000'],
    http_req_failed: ['rate<0.01'],
  },
};
export default function() {
  const res = http.get(__ENV.TARGET_URL);
  check(res, { 'status 200': (r) => r.status === 200 });
}
EOF
        ;;
    load)
        cat > "$RESULTS_DIR/test.js" << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
export const options = {
  stages: [
    { duration: '2m', target: 50 },
    { duration: '5m', target: 50 },
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
  },
};
export default function() {
  const res = http.get(__ENV.TARGET_URL);
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(1);
}
EOF
        ;;
    stress)
        cat > "$RESULTS_DIR/test.js" << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
export const options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 300 },
    { duration: '5m', target: 300 },
    { duration: '5m', target: 0 },
  ],
};
export default function() {
  const res = http.get(__ENV.TARGET_URL);
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(0.5);
}
EOF
        ;;
esac

# Run test
echo ""
echo "Running $TEST_TYPE test..."
k6 run --out json="$RESULTS_DIR/results.json" \
    --env TARGET_URL="$TARGET" \
    "$RESULTS_DIR/test.js" 2>&1 | tee "$RESULTS_DIR/output.txt"

echo ""
echo "Results saved to $RESULTS_DIR"

# 2. Quick HTTP benchmark
#!/bin/bash
URL="${1:?Usage: $0 <url> [requests] [concurrency]}"
REQUESTS="${2:-1000}"
CONCURRENCY="${3:-50}"

echo "=== HTTP Benchmark ==="
echo "URL: $URL"
echo "Requests: $REQUESTS"
echo "Concurrency: $CONCURRENCY"

if command -v hey &>/dev/null; then
    hey -n "$REQUESTS" -c "$CONCURRENCY" "$URL"
elif command -v ab &>/dev/null; then
    ab -n "$REQUESTS" -c "$CONCURRENCY" "$URL"
elif command -v wrk &>/dev/null; then
    wrk -t4 -c"$CONCURRENCY" -d30s "$URL"
else
    echo "No benchmark tool found. Install: brew install hey"
    exit 1
fi`,
				},
				{
					Title: "Performance Analysis and Optimization",
					Content: `Analyzing performance test results and implementing optimizations to meet SLOs.

**Performance Analysis:**
` + "```" + `
Key metrics:
  Latency:
    p50: Median response time (typical experience)
    p95: 95th percentile (worst case for most users)
    p99: 99th percentile (tail latency)
    p999: 99.9th percentile (extreme cases)
  
  Throughput:
    Requests per second (RPS/QPS)
    Transactions per second (TPS)
    Bytes per second
  
  Error rate:
    HTTP error percentage
    Timeout percentage
    Connection error percentage
  
  Saturation:
    CPU utilization
    Memory usage
    Disk I/O
    Network bandwidth
    Connection pool usage

Bottleneck identification:
  USE Method (Brendan Gregg):
    For each resource:
      Utilization: % time resource is busy
      Saturation: Amount of queued work
      Errors: Error events
    
    Resources: CPU, Memory, Disk, Network
    
    CPU saturation: Load average > CPU count
    Memory saturation: Swap usage, OOM kills
    Disk saturation: I/O queue depth
    Network saturation: Dropped packets, retransmits

  RED Method (Tom Wilkie):
    For each service:
      Rate: Requests per second
      Errors: Failed requests per second
      Duration: Distribution of latencies
    
    Prometheus queries:
      Rate: rate(http_requests_total[5m])
      Errors: rate(http_requests_total{status=~"5.."}[5m])
      Duration: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

Profiling tools:
  Go:
    pprof:
      import _ "net/http/pprof"
      go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
      go tool pprof http://localhost:6060/debug/pprof/heap
      go tool pprof http://localhost:6060/debug/pprof/goroutine
    
    trace:
      curl -o trace.out http://localhost:6060/debug/pprof/trace?seconds=5
      go tool trace trace.out
    
    Continuous profiling:
      Pyroscope: Attach to running services
      Parca: Open-source continuous profiling
  
  System:
    BPF tools:
      bpftrace, BCC tools
      tcplife, biolatency, profile
    
    perf:
      perf stat ./myapp
      perf record -g ./myapp
      perf report
    
    strace:
      strace -c ./myapp  # Syscall summary
      strace -e trace=network ./myapp
` + "```" + `

**Optimization Techniques:**
` + "```" + `
Application optimization:
  Caching:
    CDN: Static assets, API responses
    Application cache: Redis/Memcached
    Database cache: Query result cache
    
    Cache strategies:
      Cache-aside: App checks cache, falls back to DB
      Write-through: Write to cache and DB simultaneously
      Write-behind: Write to cache, async write to DB
      Read-through: Cache handles DB reads
    
    Redis caching patterns:
      # Cache with TTL
      SET user:123 '{"name":"John"}' EX 3600
      
      # Cache-aside pattern
      result = redis.get(key)
      if result is None:
          result = db.query(...)
          redis.setex(key, ttl, result)
      return result
  
  Connection pooling:
    Database: PgBouncer, ProxySQL
    HTTP: Keep-alive, HTTP/2 multiplexing
    gRPC: Persistent connections
  
  Async processing:
    Message queues: RabbitMQ, Kafka, SQS
    Background jobs: Celery, Sidekiq, Temporal
    Event-driven: Webhooks, WebSockets
  
  Database optimization:
    Query optimization (EXPLAIN ANALYZE)
    Index tuning
    Connection pooling
    Read replicas for read-heavy workloads
    Denormalization for read performance
    Partitioning for large tables

Infrastructure optimization:
  Auto-scaling:
    HPA (Kubernetes):
      apiVersion: autoscaling/v2
      kind: HorizontalPodAutoscaler
      metadata:
        name: myapp
      spec:
        scaleTargetRef:
          apiVersion: apps/v1
          kind: Deployment
          name: myapp
        minReplicas: 2
        maxReplicas: 20
        metrics:
          - type: Resource
            resource:
              name: cpu
              target:
                type: Utilization
                averageUtilization: 70
          - type: Pods
            pods:
              metric:
                name: requests_per_second
              target:
                type: AverageValue
                averageValue: "100"
        behavior:
          scaleUp:
            stabilizationWindowSeconds: 60
            policies:
              - type: Pods
                value: 4
                periodSeconds: 60
          scaleDown:
            stabilizationWindowSeconds: 300
    
    KEDA (event-driven autoscaling):
      apiVersion: keda.sh/v1alpha1
      kind: ScaledObject
      metadata:
        name: myapp
      spec:
        scaleTargetRef:
          name: myapp
        minReplicaCount: 1
        maxReplicaCount: 50
        triggers:
          - type: prometheus
            metadata:
              serverAddress: http://prometheus:9090
              metricName: http_requests_total
              query: sum(rate(http_requests_total{app="myapp"}[2m]))
              threshold: "100"
  
  CDN optimization:
    Cache headers:
      Cache-Control: public, max-age=31536000, immutable
      Cache-Control: private, no-cache
      ETag: "abc123"
      Vary: Accept-Encoding
    
    Edge computing:
      Cloudflare Workers
      AWS CloudFront Functions
      Vercel Edge Functions
  
  HTTP optimization:
    Compression: gzip, brotli
    HTTP/2: Multiplexing, header compression
    HTTP/3: QUIC, 0-RTT
    Keep-alive connections
    Request batching
` + "```" + ``,
					CodeExamples: `# Performance analysis scripts

# 1. System performance monitor
#!/bin/bash
echo "=== System Performance Monitor ==="
echo "Timestamp: $(date)"

# CPU
echo ""
echo "--- CPU ---"
CPU_USAGE=$(top -l 1 -n 0 2>/dev/null | grep "CPU usage" || \
    grep 'cpu ' /proc/stat 2>/dev/null | awk '{printf "%.1f%%", ($2+$4)*100/($2+$4+$5)}')
echo "Usage: $CPU_USAGE"
LOAD=$(uptime | awk -F'load averages?:' '{print $2}' | tr -d ' ')
echo "Load Average: $LOAD"
CPU_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "?")
echo "CPU Count: $CPU_COUNT"

# Memory
echo ""
echo "--- Memory ---"
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024/1024}')
    echo "Total: $TOTAL_MEM"
    vm_stat 2>/dev/null | head -5
else
    free -h 2>/dev/null
fi

# Disk
echo ""
echo "--- Disk ---"
df -h / 2>/dev/null | tail -1

# Network connections
echo ""
echo "--- Connections ---"
if command -v ss &>/dev/null; then
    echo "ESTABLISHED: $(ss -t state established 2>/dev/null | wc -l | tr -d ' ')"
    echo "TIME_WAIT: $(ss -t state time-wait 2>/dev/null | wc -l | tr -d ' ')"
    echo "CLOSE_WAIT: $(ss -t state close-wait 2>/dev/null | wc -l | tr -d ' ')"
elif command -v netstat &>/dev/null; then
    echo "ESTABLISHED: $(netstat -an 2>/dev/null | grep ESTABLISHED | wc -l | tr -d ' ')"
    echo "TIME_WAIT: $(netstat -an 2>/dev/null | grep TIME_WAIT | wc -l | tr -d ' ')"
fi

# Kubernetes pods (if available)
echo ""
echo "--- Top Pods by CPU ---"
kubectl top pods --all-namespaces --sort-by=cpu --no-headers 2>/dev/null | head -5

echo ""
echo "--- Top Pods by Memory ---"
kubectl top pods --all-namespaces --sort-by=memory --no-headers 2>/dev/null | head -5

# 2. Latency analyzer
#!/bin/bash
URL="${1:?Usage: $0 <url> [samples]}"
SAMPLES="${2:-100}"

echo "=== Latency Analysis ==="
echo "URL: $URL"
echo "Samples: $SAMPLES"

TIMES=()
ERRORS=0

for i in $(seq 1 "$SAMPLES"); do
    TIME=$(curl -s -o /dev/null -w "%{time_total}" --max-time 10 "$URL" 2>/dev/null)
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        TIMES+=("$TIME")
    else
        ((ERRORS++))
    fi
    
    # Progress
    if (( i % 10 == 0 )); then echo -n "."; fi
done

echo ""

if [ ${#TIMES[@]} -eq 0 ]; then
    echo "No successful requests"
    exit 1
fi

# Calculate percentiles using sort
SORTED=$(printf '%s\n' "${TIMES[@]}" | sort -n)
COUNT=${#TIMES[@]}

P50_IDX=$((COUNT * 50 / 100))
P95_IDX=$((COUNT * 95 / 100))
P99_IDX=$((COUNT * 99 / 100))

P50=$(echo "$SORTED" | sed -n "${P50_IDX}p")
P95=$(echo "$SORTED" | sed -n "${P95_IDX}p")
P99=$(echo "$SORTED" | sed -n "${P99_IDX}p")
MIN=$(echo "$SORTED" | head -1)
MAX=$(echo "$SORTED" | tail -1)
AVG=$(printf '%s\n' "${TIMES[@]}" | awk '{sum+=$1} END {printf "%.3f", sum/NR}')

echo ""
echo "--- Results ---"
echo "Samples: $COUNT successful, $ERRORS errors"
echo "Min:     ${MIN}s"
echo "Avg:     ${AVG}s"
echo "P50:     ${P50}s"
echo "P95:     ${P95}s"
echo "P99:     ${P99}s"
echo "Max:     ${MAX}s"
echo "Error:   $(echo "scale=1; $ERRORS * 100 / $SAMPLES" | bc 2>/dev/null || echo "N/A")%"

# 3. Resource trend logger
#!/bin/bash
echo "=== Resource Trend Logger ==="
INTERVAL="${1:-10}"
DURATION="${2:-300}"
OUTPUT="${3:-/tmp/resource-trend.csv}"

echo "Logging every ${INTERVAL}s for ${DURATION}s to $OUTPUT"
echo "timestamp,cpu_percent,memory_percent,disk_percent,connections" > "$OUTPUT"

END=$(($(date +%s) + DURATION))

while [ "$(date +%s)" -lt "$END" ]; do
    TS=$(date +%Y-%m-%dT%H:%M:%S)
    
    # CPU (simplified)
    CPU=$(top -l 1 -n 0 2>/dev/null | grep "CPU usage" | awk '{print $3}' | tr -d '%' || echo "0")
    
    # Memory
    if [[ "$OSTYPE" == "darwin"* ]]; then
        MEM=$(memory_pressure 2>/dev/null | grep "System-wide" | awk '{print $4}' | tr -d '%' || echo "0")
    else
        MEM=$(free 2>/dev/null | awk '/Mem:/{printf "%.1f", $3/$2*100}' || echo "0")
    fi
    
    # Disk
    DISK=$(df / 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
    
    # Connections
    CONNS=$(netstat -an 2>/dev/null | grep ESTABLISHED | wc -l | tr -d ' ')
    
    echo "$TS,$CPU,$MEM,$DISK,$CONNS" >> "$OUTPUT"
    
    sleep "$INTERVAL"
done

echo "Done. Results in $OUTPUT"
echo "Lines: $(wc -l < "$OUTPUT")"`,
				},
			},
		},
		{
			ID:          1473,
			Title:       "Windows DevOps and Cross-Platform Operations",
			Description: "Manage Windows infrastructure with PowerShell, Windows containers, IIS administration, and cross-platform CI/CD pipelines.",
			Order:       73,
			Lessons: []problems.Lesson{
				{
					Title: "PowerShell Automation and Windows Administration",
					Content: `Windows DevOps requires proficiency in PowerShell, Windows Server administration, and cross-platform tooling.

**PowerShell Fundamentals:**
` + "```" + `
PowerShell basics:
  Variables and types:
    $name = "Server01"
    [int]$port = 8080
    [string[]]$servers = @("srv1", "srv2", "srv3")
    [hashtable]$config = @{
      Environment = "Production"
      Region = "US-East"
      MaxRetries = 3
    }
  
  Pipeline and filtering:
    Get-Process | Where-Object CPU -gt 100 | Sort-Object CPU -Descending
    Get-Service | Where-Object {$_.Status -eq "Running"} | Select-Object Name, Status
    Get-ChildItem -Recurse -Filter *.log | Measure-Object -Property Length -Sum
  
  Functions:
    function Get-ServerHealth {
      param(
        [Parameter(Mandatory)][string]$ServerName,
        [int]$Timeout = 30
      )
      
      $result = @{
        Server = $ServerName
        Timestamp = Get-Date
        Reachable = Test-Connection $ServerName -Count 1 -Quiet
      }
      
      if ($result.Reachable) {
        $os = Get-CimInstance Win32_OperatingSystem -ComputerName $ServerName
        $result.FreeMemoryGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        $result.Uptime = (Get-Date) - $os.LastBootUpTime
      }
      
      [PSCustomObject]$result
    }
  
  Error handling:
    try {
      $service = Get-Service -Name "MyService" -ErrorAction Stop
      Restart-Service $service -Force
      Write-Output "Service restarted successfully"
    }
    catch [Microsoft.PowerShell.Commands.ServiceCommandException] {
      Write-Error "Service not found: $_"
    }
    catch {
      Write-Error "Unexpected error: $($_.Exception.Message)"
    }
    finally {
      Write-Output "Cleanup complete"
    }

Windows Server management:
  IIS:
    Import-Module WebAdministration
    
    # Create website
    New-WebSite -Name "MyApp" -Port 443 -HostHeader "myapp.example.com" \
      -PhysicalPath "C:\inetpub\myapp" -Ssl
    
    # Application pool
    New-WebAppPool -Name "MyAppPool"
    Set-ItemProperty "IIS:\AppPools\MyAppPool" -Name processModel.identityType -Value ApplicationPoolIdentity
    Set-ItemProperty "IIS:\AppPools\MyAppPool" -Name recycling.periodicRestart.time -Value "02:00:00"
    
    # Health check
    Get-WebSite | Select-Object Name, State, Bindings
    Get-WebAppPoolState -Name "MyAppPool"
  
  Windows Services:
    # Create service
    New-Service -Name "MyApp" -BinaryPathName "C:\apps\myapp.exe" \
      -DisplayName "My Application" -StartupType Automatic \
      -Description "My application service"
    
    # Service management
    Get-Service MyApp | Select-Object Status, StartType
    Start-Service MyApp
    Stop-Service MyApp -Force
    Restart-Service MyApp
    
    # NSSM (Non-Sucking Service Manager)
    nssm install MyApp "C:\apps\myapp.exe"
    nssm set MyApp AppDirectory "C:\apps"
    nssm set MyApp AppStdout "C:\logs\myapp-stdout.log"
    nssm set MyApp AppStderr "C:\logs\myapp-stderr.log"
  
  Windows Firewall:
    New-NetFirewallRule -DisplayName "Allow HTTP" \
      -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow
    
    New-NetFirewallRule -DisplayName "Allow HTTPS" \
      -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow
    
    Get-NetFirewallRule | Where-Object Enabled -eq True | \
      Select-Object DisplayName, Direction, Action

DSC (Desired State Configuration):
  Configuration WebServerConfig {
    Import-DscResource -ModuleName PSDscResources
    Import-DscResource -ModuleName xWebAdministration
    
    Node "WebServer" {
      WindowsFeature IIS {
        Name = "Web-Server"
        Ensure = "Present"
      }
      
      WindowsFeature ASPNet {
        Name = "Web-Asp-Net45"
        Ensure = "Present"
        DependsOn = "[WindowsFeature]IIS"
      }
      
      xWebsite DefaultSite {
        Name = "Default Web Site"
        Ensure = "Absent"
        DependsOn = "[WindowsFeature]IIS"
      }
      
      xWebsite MyApp {
        Name = "MyApp"
        State = "Started"
        PhysicalPath = "C:\inetpub\myapp"
        BindingInfo = @(
          MSFT_xWebBindingInformation {
            Protocol = "HTTPS"
            Port = 443
            HostName = "myapp.example.com"
            CertificateThumbprint = "ABC123..."
            CertificateStoreName = "My"
          }
        )
        DependsOn = "[WindowsFeature]IIS"
      }
    }
  }
` + "```" + ``,
					CodeExamples: `# PowerShell automation scripts

# 1. Server health report
# Get-ServerReport.ps1
function Get-ServerReport {
    param(
        [string[]]$Servers = @("localhost"),
        [string]$OutputPath = ".\server-report.html"
    )
    
    $results = foreach ($server in $Servers) {
        try {
            $os = Get-CimInstance Win32_OperatingSystem -ComputerName $server -ErrorAction Stop
            $cpu = Get-CimInstance Win32_Processor -ComputerName $server | 
                Measure-Object -Property LoadPercentage -Average
            $disk = Get-CimInstance Win32_LogicalDisk -ComputerName $server -Filter "DriveType=3"
            
            [PSCustomObject]@{
                Server = $server
                Status = "Online"
                OS = $os.Caption
                Uptime = "{0:dd}d {0:hh}h" -f ((Get-Date) - $os.LastBootUpTime)
                CPUPercent = "$($cpu.Average)%"
                FreeMemoryGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
                TotalMemoryGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
                DiskFreeGB = ($disk | ForEach-Object { 
                    "$($_.DeviceID) $(([math]::Round($_.FreeSpace/1GB,1)))GB"
                }) -join ", "
            }
        }
        catch {
            [PSCustomObject]@{
                Server = $server
                Status = "Offline"
                OS = "N/A"
                Uptime = "N/A"
                CPUPercent = "N/A"
                FreeMemoryGB = "N/A"
                TotalMemoryGB = "N/A"
                DiskFreeGB = "N/A"
            }
        }
    }
    
    $results | Format-Table -AutoSize
    $results | ConvertTo-Html -Title "Server Report $(Get-Date)" | Out-File $OutputPath
    Write-Output "Report saved: $OutputPath"
}

# 2. IIS deployment script
# Deploy-WebApp.ps1
function Deploy-WebApp {
    param(
        [Parameter(Mandatory)][string]$AppName,
        [Parameter(Mandatory)][string]$PackagePath,
        [string]$SitePath = "C:\inetpub",
        [int]$Port = 443
    )
    
    Import-Module WebAdministration
    
    $appPath = Join-Path $SitePath $AppName
    $backupPath = "${appPath}_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    
    Write-Output "Deploying $AppName..."
    
    # Stop app pool
    $pool = "$AppName-Pool"
    if (Test-Path "IIS:\AppPools\$pool") {
        Write-Output "Stopping app pool: $pool"
        Stop-WebAppPool -Name $pool
        Start-Sleep -Seconds 5
    }
    
    # Backup current version
    if (Test-Path $appPath) {
        Write-Output "Backing up to: $backupPath"
        Copy-Item -Path $appPath -Destination $backupPath -Recurse
    }
    
    # Deploy new version
    Write-Output "Deploying from: $PackagePath"
    if ($PackagePath.EndsWith(".zip")) {
        Expand-Archive -Path $PackagePath -DestinationPath $appPath -Force
    } else {
        Copy-Item -Path "$PackagePath\*" -Destination $appPath -Recurse -Force
    }
    
    # Create app pool if needed
    if (-not (Test-Path "IIS:\AppPools\$pool")) {
        New-WebAppPool -Name $pool
        Set-ItemProperty "IIS:\AppPools\$pool" processModel.identityType ApplicationPoolIdentity
    }
    
    # Start app pool
    Start-WebAppPool -Name $pool
    
    # Health check
    Start-Sleep -Seconds 5
    $health = Invoke-WebRequest -Uri "https://localhost:$Port/health" -UseBasicParsing -SkipCertificateCheck
    if ($health.StatusCode -eq 200) {
        Write-Output "Deployment successful - health check passed"
        # Cleanup old backups (keep last 3)
        Get-ChildItem "${appPath}_backup_*" -Directory | 
            Sort-Object CreationTime -Descending | 
            Select-Object -Skip 3 | 
            Remove-Item -Recurse -Force
    } else {
        Write-Warning "Health check failed (HTTP $($health.StatusCode))"
        Write-Warning "Rolling back..."
        if (Test-Path $backupPath) {
            Stop-WebAppPool -Name $pool
            Remove-Item $appPath -Recurse -Force
            Move-Item $backupPath $appPath
            Start-WebAppPool -Name $pool
            Write-Output "Rollback complete"
        }
    }
}

# 3. Windows service monitor
# Monitor-Services.ps1
function Monitor-Services {
    param(
        [string[]]$ServiceNames = @("W3SVC", "MSSQLSERVER"),
        [int]$IntervalSeconds = 30,
        [string]$LogPath = ".\service-monitor.log"
    )
    
    function Write-Log {
        param([string]$Message)
        $entry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $Message"
        Add-Content -Path $LogPath -Value $entry
        Write-Output $entry
    }
    
    Write-Log "Starting service monitor for: $($ServiceNames -join ', ')"
    
    while ($true) {
        foreach ($name in $ServiceNames) {
            try {
                $svc = Get-Service -Name $name -ErrorAction Stop
                if ($svc.Status -ne "Running") {
                    Write-Log "WARNING: $name is $($svc.Status)"
                    Write-Log "Attempting restart of $name..."
                    try {
                        Start-Service -Name $name -ErrorAction Stop
                        Start-Sleep -Seconds 5
                        $svc = Get-Service -Name $name
                        Write-Log "Service $name is now $($svc.Status)"
                    }
                    catch {
                        Write-Log "ERROR: Failed to restart $name - $($_.Exception.Message)"
                    }
                }
            }
            catch {
                Write-Log "ERROR: Cannot check $name - $($_.Exception.Message)"
            }
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
}`,
				},
				{
					Title: "Windows Containers and Cross-Platform CI/CD",
					Content: `Windows containers enable containerizing Windows workloads, while cross-platform pipelines handle mixed environments.

**Windows Containers:**
` + "```" + `
Container types:
  Windows Server containers:
    Process isolation (shared kernel).
    Similar to Linux containers.
    Requires matching host OS version.
  
  Hyper-V containers:
    VM-level isolation.
    Each container has own kernel.
    Better security isolation.
    Can run different OS versions.

Windows container images:
  Base images:
    mcr.microsoft.com/windows/servercore:ltsc2022 (5+ GB)
    mcr.microsoft.com/windows/nanoserver:ltsc2022 (300+ MB)
    mcr.microsoft.com/dotnet/aspnet:8.0-nanoserver-ltsc2022
    mcr.microsoft.com/dotnet/sdk:8.0-windowsservercore-ltsc2022
  
  Dockerfile for .NET:
    FROM mcr.microsoft.com/dotnet/sdk:8.0-windowsservercore-ltsc2022 AS build
    WORKDIR /src
    COPY *.csproj .
    RUN dotnet restore
    COPY . .
    RUN dotnet publish -c Release -o /app/publish
    
    FROM mcr.microsoft.com/dotnet/aspnet:8.0-nanoserver-ltsc2022
    WORKDIR /app
    COPY --from=build /app/publish .
    USER ContainerUser
    EXPOSE 8080
    ENTRYPOINT ["dotnet", "MyApp.dll"]
  
  Dockerfile for IIS:
    FROM mcr.microsoft.com/windows/servercore/iis:windowsservercore-ltsc2022
    SHELL ["powershell", "-Command"]
    
    # Install URL Rewrite module
    RUN Invoke-WebRequest -Uri https://example.com/urlrewrite.msi -OutFile urlrewrite.msi; \
        Start-Process msiexec -ArgumentList '/i', 'urlrewrite.msi', '/quiet' -Wait; \
        Remove-Item urlrewrite.msi
    
    # Deploy application
    COPY ./publish /inetpub/wwwroot
    
    # Configure IIS
    RUN Remove-WebSite -Name 'Default Web Site'; \
        New-WebSite -Name 'MyApp' -Port 80 -PhysicalPath 'C:\inetpub\wwwroot'
    
    EXPOSE 80
    ENTRYPOINT ["C:\\ServiceMonitor.exe", "w3svc"]

Kubernetes Windows nodes:
  # Mixed cluster (Linux + Windows)
  Node selector for Windows:
    nodeSelector:
      kubernetes.io/os: windows
  
  Tolerations:
    tolerations:
      - key: "os"
        operator: "Equal"
        value: "windows"
        effect: "NoSchedule"
  
  Deployment:
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: iis-app
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: iis-app
      template:
        metadata:
          labels:
            app: iis-app
        spec:
          nodeSelector:
            kubernetes.io/os: windows
          containers:
            - name: iis
              image: myregistry/iis-app:latest
              ports:
                - containerPort: 80
              resources:
                limits:
                  cpu: "1"
                  memory: 2Gi
` + "```" + `

**Cross-Platform CI/CD:**
` + "```" + `
GitHub Actions multi-platform:
  name: Cross-Platform Build
  on: [push]
  
  jobs:
    build-linux:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-go@v5
          with:
            go-version: '1.22'
        - run: |
            GOOS=linux GOARCH=amd64 go build -o bin/app-linux-amd64 ./cmd/app
            GOOS=linux GOARCH=arm64 go build -o bin/app-linux-arm64 ./cmd/app
        - uses: actions/upload-artifact@v4
          with:
            name: linux-binaries
            path: bin/
    
    build-windows:
      runs-on: windows-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-go@v5
          with:
            go-version: '1.22'
        - run: go build -o bin/app.exe ./cmd/app
        - uses: actions/upload-artifact@v4
          with:
            name: windows-binary
            path: bin/
    
    build-container:
      runs-on: ubuntu-latest
      needs: [build-linux]
      steps:
        - uses: actions/checkout@v4
        - uses: docker/setup-buildx-action@v3
        - uses: docker/build-push-action@v5
          with:
            push: true
            platforms: linux/amd64,linux/arm64
            tags: myregistry/app:latest
    
    test-linux:
      runs-on: ubuntu-latest
      needs: [build-linux]
      steps:
        - uses: actions/checkout@v4
        - run: go test ./... -v
    
    test-windows:
      runs-on: windows-latest
      needs: [build-windows]
      steps:
        - uses: actions/checkout@v4
        - run: go test ./... -v

Azure DevOps multi-platform:
  trigger:
    - main
  
  stages:
    - stage: Build
      jobs:
        - job: Linux
          pool:
            vmImage: 'ubuntu-latest'
          steps:
            - task: GoTool@0
              inputs:
                version: '1.22'
            - script: go build -o $(Build.ArtifactStagingDirectory)/app ./cmd/app
            - publish: $(Build.ArtifactStagingDirectory)
              artifact: linux-app
        
        - job: Windows
          pool:
            vmImage: 'windows-latest'
          steps:
            - task: GoTool@0
              inputs:
                version: '1.22'
            - script: go build -o $(Build.ArtifactStagingDirectory)\app.exe .\cmd\app
            - publish: $(Build.ArtifactStagingDirectory)
              artifact: windows-app
    
    - stage: Deploy
      dependsOn: Build
      jobs:
        - deployment: DeployLinux
          environment: production-linux
          strategy:
            runOnce:
              deploy:
                steps:
                  - download: current
                    artifact: linux-app
        
        - deployment: DeployWindows
          environment: production-windows
          strategy:
            runOnce:
              deploy:
                steps:
                  - download: current
                    artifact: windows-app  

Cross-platform tools:
  Ansible (Windows support):
    ansible.cfg:
      [defaults]
      transport = winrm
    
    Inventory:
      [windows]
      win01 ansible_host=10.0.1.10
      
      [windows:vars]
      ansible_user=admin
      ansible_password={{ vault_win_password }}
      ansible_connection=winrm
      ansible_winrm_server_cert_validation=ignore
    
    Playbook:
      - hosts: windows
        tasks:
          - name: Install IIS
            win_feature:
              name: Web-Server
              state: present
          
          - name: Deploy application
            win_copy:
              src: ./publish/
              dest: C:\inetpub\myapp\
          
          - name: Start website
            win_iis_website:
              name: MyApp
              state: started
              physical_path: C:\inetpub\myapp
              port: 443
` + "```" + ``,
					CodeExamples: `# Cross-platform operations scripts

# 1. Multi-platform build script
#!/bin/bash
set -e

echo "=== Cross-Platform Build ==="

APP_NAME="${1:-myapp}"
VERSION="${2:-$(git describe --tags --always 2>/dev/null || echo 'dev')}"
OUTPUT_DIR="./dist"

mkdir -p "$OUTPUT_DIR"

PLATFORMS=(
    "linux/amd64"
    "linux/arm64"
    "darwin/amd64"
    "darwin/arm64"
    "windows/amd64"
)

LDFLAGS="-s -w -X main.version=$VERSION -X main.buildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for PLATFORM in "${PLATFORMS[@]}"; do
    OS="${PLATFORM%/*}"
    ARCH="${PLATFORM#*/}"
    OUTPUT="${OUTPUT_DIR}/${APP_NAME}-${OS}-${ARCH}"
    
    if [ "$OS" = "windows" ]; then
        OUTPUT="${OUTPUT}.exe"
    fi
    
    echo "Building: $OS/$ARCH → $OUTPUT"
    GOOS=$OS GOARCH=$ARCH CGO_ENABLED=0 \
        go build -ldflags="$LDFLAGS" -o "$OUTPUT" ./cmd/"$APP_NAME" 2>/dev/null
done

echo ""
echo "Build artifacts:"
ls -lh "$OUTPUT_DIR"/

# Create checksums
echo ""
echo "Generating checksums..."
cd "$OUTPUT_DIR"
shasum -a 256 * > checksums.txt
cat checksums.txt
cd ..

echo ""
echo "Build complete: $VERSION"

# 2. Container multi-arch build
#!/bin/bash
set -e

IMAGE="${1:?Usage: $0 <image-name> [tag]}"
TAG="${2:-latest}"

echo "=== Multi-Architecture Container Build ==="
echo "Image: $IMAGE:$TAG"

# Ensure buildx is available
docker buildx create --use --name multiarch 2>/dev/null || true

# Build and push
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "$IMAGE:$TAG" \
    --push \
    .

echo ""
echo "Image pushed: $IMAGE:$TAG"
echo "Platforms: linux/amd64, linux/arm64"

# Verify
docker manifest inspect "$IMAGE:$TAG" 2>/dev/null | \
    jq '.manifests[] | {platform: .platform, digest: .digest}' 2>/dev/null

# 3. Environment sync checker
#!/bin/bash
echo "=== Cross-Platform Environment Check ==="

check_tool() {
    local tool="$1"
    if command -v "$tool" &>/dev/null; then
        VERSION=$($tool --version 2>&1 | head -1)
        echo "  [OK] $tool: $VERSION"
    else
        echo "  [MISSING] $tool"
    fi
}

echo "--- Core Tools ---"
check_tool "git"
check_tool "go"
check_tool "python3"
check_tool "node"
check_tool "docker"
check_tool "kubectl"

echo ""
echo "--- Build Tools ---"
check_tool "make"
check_tool "gcc"

echo ""
echo "--- DevOps Tools ---"
check_tool "terraform"
check_tool "ansible"
check_tool "helm"
check_tool "k6"
check_tool "trivy"

echo ""
echo "--- Platform ---"
echo "  OS: $(uname -s)"
echo "  Arch: $(uname -m)"
echo "  Shell: $SHELL"`,
				},
			},
		},
	})
}
