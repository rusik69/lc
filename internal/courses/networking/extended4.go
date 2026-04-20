package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2619,
			Title:       "Load Balancing and Traffic Management",
			Description: "Master load balancing algorithms, health checking, session persistence, global server load balancing, and traffic shaping techniques.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Load Balancing Algorithms and Architectures",
					Content: `Load balancers distribute incoming traffic across multiple servers to improve reliability, scalability, and performance.

**Load Balancing Layers:**

Layer 4 (Transport):
  Operates at TCP/UDP level
  Forwards based on IP and port
  No inspection of application data
  High performance, low latency
  Examples: HAProxy (TCP mode), AWS NLB, IPVS
  
  Techniques:
    NAT (Network Address Translation)
    DSR (Direct Server Return)
    IP tunneling

Layer 7 (Application):
  Operates at HTTP/HTTPS level
  Can inspect headers, URLs, cookies
  Content-based routing
  SSL termination
  Examples: HAProxy (HTTP mode), Nginx, AWS ALB, Envoy
  
  Features:
    Path-based routing (/api -> backend, /static -> CDN)
    Header-based routing (mobile vs desktop)
    Cookie-based session affinity
    Request modification (add/remove headers)
    Rate limiting per endpoint

**Load Balancing Algorithms:**

Round Robin:
  Distribute requests sequentially across servers
  Simple, equal distribution
  Doesn't account for server capacity or load
  
  Server 1 -> Server 2 -> Server 3 -> Server 1 -> ...

Weighted Round Robin:
  Assign weights based on server capacity
  Server A (weight 3): Gets 3x more requests
  Server B (weight 1): Gets 1x requests
  
  A -> A -> A -> B -> A -> A -> A -> B -> ...

Least Connections:
  Route to server with fewest active connections
  Better than round robin for variable request durations
  Accounts for slow requests naturally
  
  Server A: 5 connections  <- new request goes here
  Server B: 8 connections
  Server C: 12 connections

Weighted Least Connections:
  Combine weights with connection count
  Score = connections / weight
  Route to lowest score
  
  Server A (weight 3, 6 connections): Score = 2
  Server B (weight 1, 3 connections): Score = 3
  Route to A (lower score)

IP Hash:
  Hash client IP to determine server
  Same client always goes to same server
  Provides session persistence without cookies
  Problem: Uneven distribution with NAT/proxies

Consistent Hashing:
  Hash key maps to ring position
  Find next server clockwise on ring
  Adding/removing server only moves fraction of keys
  Used in: Nginx upstream, Envoy, Maglev
  
  Better than IP hash for:
    Cache locality
    Minimal redistribution on server changes
    Virtual nodes for even distribution

Random:
  Random server selection
  Surprisingly effective with many servers
  Power of Two Choices: Pick 2 random, choose less loaded
  Simple to implement, no state needed

Least Response Time:
  Route to server with fastest response + fewest connections
  Requires monitoring response times
  Adaptive to actual server performance

**Health Checking:**

Active Health Checks:
  Load balancer sends periodic probe to servers
  TCP connect (port open?)
  HTTP request (returns 200?)
  Custom script (application-specific)
  
  Parameters:
    Interval: How often to check (5-30 seconds)
    Timeout: Max wait for response (2-5 seconds)
    Threshold: Failures before marking unhealthy (2-3)
    Recovery: Successes before marking healthy (2-3)

Passive Health Checks:
  Monitor actual traffic for failures
  Track error rates, response times
  Mark unhealthy based on real traffic patterns
  No additional probe traffic

**Session Persistence (Sticky Sessions):**

Cookie-based:
  Load balancer sets a cookie with server ID
  Subsequent requests routed to same server
  Most flexible, works through proxies/NAT
  
  Set-Cookie: SERVERID=server1; Path=/

Source IP:
  Hash source IP to determine server
  Simple but breaks with NAT
  
Application-level:
  Store sessions in shared store (Redis, Memcached)
  Any server can handle any request
  Best approach for scalability

**Session Draining (Graceful Removal):**
  Stop sending new connections to server being removed
  Allow existing connections to complete
  Timeout for lingering connections
  Used during deployments and maintenance

**Global Server Load Balancing (GSLB):**

  Distribute traffic across geographic regions
  DNS-based routing to nearest/healthiest region
  
  Methods:
    Geographic routing: Route based on client location
    Latency-based: Route to lowest latency endpoint
    Failover: Active-passive between regions
    Weighted: Split traffic by percentage
  
  Services: AWS Route 53, Cloudflare, NS1, Akamai GTM

**Traffic Shaping and QoS:**

Rate Limiting:
  Limit requests per time window
  Algorithms: Token bucket, leaky bucket, sliding window
  Per client, per API key, per endpoint

Traffic Prioritization:
  Differentiate between traffic types
  Critical API calls > Background jobs > Monitoring
  QoS marking (DSCP values)

Connection Limits:
  Max connections per server
  Max connections per client
  Queuing for excess connections

Bandwidth Throttling:
  Limit bandwidth per client or service
  Prevent single client from saturating network
  Fair sharing of resources`,
					CodeExamples: `// Load balancing implementations in Go

package main

import (
    "fmt"
    "hash/fnv"
    "math/rand"
    "net/http"
    "net/http/httputil"
    "net/url"
    "sort"
    "sync"
    "sync/atomic"
    "time"
)

// Backend server
type Backend struct {
    URL          *url.URL
    Weight       int
    Alive        bool
    Connections  int64
    ResponseTime time.Duration
    mu           sync.RWMutex
}

func (b *Backend) IsAlive() bool {
    b.mu.RLock()
    defer b.mu.RUnlock()
    return b.Alive
}

func (b *Backend) SetAlive(alive bool) {
    b.mu.Lock()
    defer b.mu.Unlock()
    b.Alive = alive
}

// Round robin load balancer
type RoundRobinBalancer struct {
    backends []*Backend
    current  uint64
}

func (lb *RoundRobinBalancer) Next() *Backend {
    for i := 0; i < len(lb.backends); i++ {
        idx := atomic.AddUint64(&lb.current, 1) % uint64(len(lb.backends))
        if lb.backends[idx].IsAlive() {
            return lb.backends[idx]
        }
    }
    return nil
}

// Weighted round robin
type WeightedRRBalancer struct {
    backends        []*Backend
    currentWeight   int
    currentIndex    int
    maxWeight       int
    gcdWeight       int
    mu              sync.Mutex
}

func NewWeightedRRBalancer(backends []*Backend) *WeightedRRBalancer {
    maxW := 0
    weights := make([]int, len(backends))
    for i, b := range backends {
        if b.Weight > maxW {
            maxW = b.Weight
        }
        weights[i] = b.Weight
    }
    
    return &WeightedRRBalancer{
        backends:      backends,
        currentIndex:  -1,
        maxWeight:     maxW,
        gcdWeight:     gcd(weights),
    }
}

func (lb *WeightedRRBalancer) Next() *Backend {
    lb.mu.Lock()
    defer lb.mu.Unlock()
    
    for {
        lb.currentIndex = (lb.currentIndex + 1) % len(lb.backends)
        if lb.currentIndex == 0 {
            lb.currentWeight -= lb.gcdWeight
            if lb.currentWeight <= 0 {
                lb.currentWeight = lb.maxWeight
            }
        }
        
        b := lb.backends[lb.currentIndex]
        if b.IsAlive() && b.Weight >= lb.currentWeight {
            return b
        }
    }
}

func gcd(values []int) int {
    result := values[0]
    for _, v := range values[1:] {
        result = gcdTwo(result, v)
    }
    return result
}

func gcdTwo(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

// Least connections load balancer
type LeastConnBalancer struct {
    backends []*Backend
    mu       sync.Mutex
}

func (lb *LeastConnBalancer) Next() *Backend {
    lb.mu.Lock()
    defer lb.mu.Unlock()
    
    var best *Backend
    var minConn int64 = 1<<63 - 1
    
    for _, b := range lb.backends {
        if !b.IsAlive() {
            continue
        }
        conn := atomic.LoadInt64(&b.Connections)
        if conn < minConn {
            minConn = conn
            best = b
        }
    }
    
    return best
}

// Consistent hashing load balancer
type ConsistentHashBalancer struct {
    ring     []hashEntry
    replicas int
    backends map[string]*Backend
    mu       sync.RWMutex
}

type hashEntry struct {
    hash    uint32
    backend string
}

func NewConsistentHashBalancer(replicas int) *ConsistentHashBalancer {
    return &ConsistentHashBalancer{
        replicas: replicas,
        backends: make(map[string]*Backend),
    }
}

func (lb *ConsistentHashBalancer) Add(backend *Backend) {
    lb.mu.Lock()
    defer lb.mu.Unlock()
    
    key := backend.URL.String()
    lb.backends[key] = backend
    
    for i := 0; i < lb.replicas; i++ {
        hash := hashKey(fmt.Sprintf("%s-%d", key, i))
        lb.ring = append(lb.ring, hashEntry{hash: hash, backend: key})
    }
    
    sort.Slice(lb.ring, func(i, j int) bool {
        return lb.ring[i].hash < lb.ring[j].hash
    })
}

func (lb *ConsistentHashBalancer) Get(key string) *Backend {
    lb.mu.RLock()
    defer lb.mu.RUnlock()
    
    if len(lb.ring) == 0 {
        return nil
    }
    
    hash := hashKey(key)
    idx := sort.Search(len(lb.ring), func(i int) bool {
        return lb.ring[i].hash >= hash
    })
    
    if idx == len(lb.ring) {
        idx = 0
    }
    
    // Find first alive backend
    for i := 0; i < len(lb.ring); i++ {
        entry := lb.ring[(idx+i)%len(lb.ring)]
        backend := lb.backends[entry.backend]
        if backend != nil && backend.IsAlive() {
            return backend
        }
    }
    
    return nil
}

func hashKey(key string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(key))
    return h.Sum32()
}

// Power of Two Choices
type P2CBalancer struct {
    backends []*Backend
}

func (lb *P2CBalancer) Next() *Backend {
    alive := make([]*Backend, 0)
    for _, b := range lb.backends {
        if b.IsAlive() {
            alive = append(alive, b)
        }
    }
    
    if len(alive) == 0 {
        return nil
    }
    if len(alive) == 1 {
        return alive[0]
    }
    
    // Pick two random backends
    i := rand.Intn(len(alive))
    j := rand.Intn(len(alive))
    for j == i {
        j = rand.Intn(len(alive))
    }
    
    // Choose the one with fewer connections
    if atomic.LoadInt64(&alive[i].Connections) <= atomic.LoadInt64(&alive[j].Connections) {
        return alive[i]
    }
    return alive[j]
}

// Health checker
type HealthChecker struct {
    backends []*Backend
    interval time.Duration
    timeout  time.Duration
    threshold int
    failures map[string]int
    mu       sync.Mutex
}

func NewHealthChecker(backends []*Backend, interval time.Duration) *HealthChecker {
    return &HealthChecker{
        backends:  backends,
        interval:  interval,
        timeout:   3 * time.Second,
        threshold: 3,
        failures:  make(map[string]int),
    }
}

func (hc *HealthChecker) Start(ctx context.Context) {
    ticker := time.NewTicker(hc.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            hc.checkAll()
        }
    }
}

func (hc *HealthChecker) checkAll() {
    for _, backend := range hc.backends {
        go hc.check(backend)
    }
}

func (hc *HealthChecker) check(backend *Backend) {
    client := &http.Client{Timeout: hc.timeout}
    resp, err := client.Get(backend.URL.String() + "/health")
    
    hc.mu.Lock()
    defer hc.mu.Unlock()
    
    key := backend.URL.String()
    
    if err != nil || resp.StatusCode != 200 {
        hc.failures[key]++
        if hc.failures[key] >= hc.threshold {
            backend.SetAlive(false)
        }
    } else {
        hc.failures[key] = 0
        backend.SetAlive(true)
    }
    
    if resp != nil {
        resp.Body.Close()
    }
}

// Reverse proxy with load balancing
type LoadBalancedProxy struct {
    balancer interface{ Next() *Backend }
}

func (p *LoadBalancedProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    backend := p.balancer.Next()
    if backend == nil {
        http.Error(w, "no backends available", http.StatusServiceUnavailable)
        return
    }
    
    atomic.AddInt64(&backend.Connections, 1)
    defer atomic.AddInt64(&backend.Connections, -1)
    
    proxy := httputil.NewSingleHostReverseProxy(backend.URL)
    proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
        backend.SetAlive(false)
        http.Error(w, "backend error", http.StatusBadGateway)
    }
    
    proxy.ServeHTTP(w, r)
}`,
				},
			},
		},
		{
			ID:          2620,
			Title:       "Network Performance and Optimization",
			Description: "Optimize network performance with TCP tuning, congestion control, bandwidth management, latency optimization, and network monitoring strategies.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "TCP Tuning and Network Optimization",
					Content: `Network performance optimization requires understanding TCP internals, congestion control, and system tuning.

**TCP Connection Lifecycle:**

Three-Way Handshake:
  Client -> SYN -> Server
  Client <- SYN-ACK <- Server
  Client -> ACK -> Server
  
  Cost: 1 RTT before data can be sent
  Mitigation: TCP Fast Open (TFO) - send data in SYN

Four-Way Teardown:
  Client -> FIN -> Server
  Client <- ACK <- Server
  Client <- FIN <- Server
  Client -> ACK -> Server
  
  TIME_WAIT: Client waits 2*MSL (60-120 seconds)
  Issue: Can exhaust port range with many short connections
  Fix: Connection pooling, SO_REUSEADDR/SO_REUSEPORT

**TCP Windows:**

Receive Window:
  How much data receiver can accept
  Advertised in TCP header
  Window scaling: Up to 1GB (RFC 7323)
  Auto-tuned by modern OS

Congestion Window (cwnd):
  How much data sender can have in flight
  Controlled by congestion algorithm
  Not visible in TCP header
  
  Bandwidth-Delay Product (BDP):
    BDP = Bandwidth × RTT
    Optimal window size = BDP
    
    Example: 100 Mbps link, 50ms RTT
    BDP = 100,000,000 * 0.050 / 8 = 625,000 bytes
    Need 625 KB window to fill the pipe

**TCP Congestion Control Algorithms:**

Slow Start:
  cwnd starts at initcwnd (typically 10 segments)
  Double cwnd each RTT (exponential growth)
  Until ssthresh or packet loss
  
  RTT 0: cwnd = 10
  RTT 1: cwnd = 20
  RTT 2: cwnd = 40
  RTT 3: cwnd = 80

Congestion Avoidance:
  After reaching ssthresh
  Increase cwnd by 1 segment per RTT (linear)
  Conservative growth to avoid congestion

CUBIC (Linux default):
  Cubic function for window growth
  Aggressive after loss recovery
  Good for high-bandwidth, long-RTT networks
  Most widely deployed algorithm

BBR (Bottleneck Bandwidth and Round-trip propagation time):
  Google's congestion control
  Model-based, not loss-based
  Measures bandwidth and RTT
  Much better performance on lossy networks
  
  Phases:
    Startup: Exponential bandwidth probing
    Drain: Reduce inflight to measured BDP
    ProbeBW: Cycle through bandwidth probes
    ProbeRTT: Periodically reduce cwnd to measure RTT

**TCP Tuning Parameters:**

Socket Buffer Sizes:
  net.core.rmem_max = 16777216 (16 MB)
  net.core.wmem_max = 16777216
  net.ipv4.tcp_rmem = "4096 131072 16777216" (min default max)
  net.ipv4.tcp_wmem = "4096 131072 16777216"

Connection Handling:
  net.core.somaxconn = 65535  (listen backlog)
  net.ipv4.tcp_max_syn_backlog = 65535
  net.core.netdev_max_backlog = 65535

Time Wait:
  net.ipv4.tcp_tw_reuse = 1  (reuse TIME_WAIT sockets)
  net.ipv4.tcp_fin_timeout = 15

Keepalive:
  net.ipv4.tcp_keepalive_time = 600
  net.ipv4.tcp_keepalive_intvl = 60
  net.ipv4.tcp_keepalive_probes = 3

Initial Window:
  net.ipv4.tcp_init_cwnd = 10 (modern default)
  Larger initial window reduces page load time

**Latency Optimization:**

Sources of Latency:
  Propagation delay: Speed of light (~5ms per 1000km)
  Transmission delay: Data size / bandwidth
  Processing delay: Router/switch processing
  Queuing delay: Waiting in buffers
  Serialization delay: Converting data to wire format

Reducing Latency:
  Use CDNs for static content
  Deploy close to users (edge computing)
  Connection pooling (avoid handshake latency)
  HTTP/2 multiplexing (avoid head-of-line blocking)
  QUIC/HTTP/3 (0-RTT connection establishment)
  DNS prefetching and preconnection
  TCP Fast Open for repeated connections

**Nagle's Algorithm and TCP_NODELAY:**

  Nagle: Buffer small packets, send when ACK received or buffer full
  Good for: Reducing small packet overhead (telnet, SSH typing)
  Bad for: Real-time applications, interactive protocols
  
  TCP_NODELAY: Disable Nagle, send immediately
  Use for: Gaming, real-time, interactive applications
  
  Delayed ACK interaction:
    Nagle waits for ACK, delayed ACK waits for more data
    Together they cause ~200ms delay
    TCP_NODELAY fixes this

**Network Monitoring:**

Tools:
  tcpdump: Packet capture and analysis
  Wireshark: GUI packet analyzer
  netstat/ss: Connection statistics
  iftop: Real-time bandwidth monitoring
  iperf3: Network throughput testing
  mtr: Combined ping and traceroute
  nmap: Network scanning and auditing

Key Metrics:
  Throughput: Actual data transfer rate
  Latency: Round-trip time (p50, p95, p99)
  Packet loss: Percentage of lost packets
  Jitter: Variation in latency
  Retransmission rate: TCP retransmits
  Connection time: Time to establish connection
  Error rate: TCP errors, resets`,
					CodeExamples: `// Network performance tools in Go

package main

import (
    "context"
    "fmt"
    "math"
    "net"
    "sort"
    "sync"
    "time"
)

// Latency measurer with statistics
type LatencyMeasurer struct {
    target  string
    samples []time.Duration
    mu      sync.Mutex
}

func NewLatencyMeasurer(target string) *LatencyMeasurer {
    return &LatencyMeasurer{
        target:  target,
        samples: make([]time.Duration, 0),
    }
}

func (m *LatencyMeasurer) Measure(count int, interval time.Duration) *LatencyStats {
    for i := 0; i < count; i++ {
        start := time.Now()
        conn, err := net.DialTimeout("tcp", m.target, 5*time.Second)
        if err == nil {
            latency := time.Since(start)
            m.mu.Lock()
            m.samples = append(m.samples, latency)
            m.mu.Unlock()
            conn.Close()
        }
        if i < count-1 {
            time.Sleep(interval)
        }
    }
    
    return m.Stats()
}

type LatencyStats struct {
    Min     time.Duration
    Max     time.Duration
    Mean    time.Duration
    Median  time.Duration
    P95     time.Duration
    P99     time.Duration
    StdDev  time.Duration
    Jitter  time.Duration
    Samples int
    Lost    int
}

func (m *LatencyMeasurer) Stats() *LatencyStats {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if len(m.samples) == 0 {
        return &LatencyStats{}
    }
    
    sorted := make([]time.Duration, len(m.samples))
    copy(sorted, m.samples)
    sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
    
    stats := &LatencyStats{
        Min:     sorted[0],
        Max:     sorted[len(sorted)-1],
        Median:  sorted[len(sorted)/2],
        P95:     sorted[int(float64(len(sorted))*0.95)],
        P99:     sorted[int(float64(len(sorted))*0.99)],
        Samples: len(sorted),
    }
    
    // Mean
    var total time.Duration
    for _, s := range sorted {
        total += s
    }
    stats.Mean = total / time.Duration(len(sorted))
    
    // Standard deviation
    var variance float64
    meanFloat := float64(stats.Mean)
    for _, s := range sorted {
        diff := float64(s) - meanFloat
        variance += diff * diff
    }
    variance /= float64(len(sorted))
    stats.StdDev = time.Duration(math.Sqrt(variance))
    
    // Jitter (mean absolute difference between consecutive samples)
    if len(m.samples) > 1 {
        var jitterSum time.Duration
        for i := 1; i < len(m.samples); i++ {
            diff := m.samples[i] - m.samples[i-1]
            if diff < 0 {
                diff = -diff
            }
            jitterSum += diff
        }
        stats.Jitter = jitterSum / time.Duration(len(m.samples)-1)
    }
    
    return stats
}

// Bandwidth estimator
type BandwidthEstimator struct {
    target string
    port   int
}

type BandwidthResult struct {
    BytesSent    int64
    Duration     time.Duration
    Throughput   float64 // Mbps
    RTT          time.Duration
    BDP          int64 // Bandwidth-Delay Product in bytes
}

func (e *BandwidthEstimator) Estimate(ctx context.Context, duration time.Duration) (*BandwidthResult, error) {
    addr := fmt.Sprintf("%s:%d", e.target, e.port)
    conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
    if err != nil {
        return nil, fmt.Errorf("connect failed: %w", err)
    }
    defer conn.Close()
    
    // Measure RTT
    rttStart := time.Now()
    conn.Write([]byte("PING"))
    buf := make([]byte, 4)
    conn.Read(buf)
    rtt := time.Since(rttStart)
    
    // Send data for specified duration
    data := make([]byte, 65536) // 64KB chunks
    var totalBytes int64
    start := time.Now()
    deadline := start.Add(duration)
    
    conn.SetWriteDeadline(deadline)
    for time.Now().Before(deadline) {
        n, err := conn.Write(data)
        if err != nil {
            break
        }
        totalBytes += int64(n)
    }
    
    elapsed := time.Since(start)
    throughputMbps := float64(totalBytes*8) / elapsed.Seconds() / 1_000_000
    bdp := int64(throughputMbps * 1_000_000 / 8 * rtt.Seconds())
    
    return &BandwidthResult{
        BytesSent:  totalBytes,
        Duration:   elapsed,
        Throughput: throughputMbps,
        RTT:        rtt,
        BDP:        bdp,
    }, nil
}

// Traceroute implementation
type TracerouteHop struct {
    TTL     int
    Address string
    RTTs    []time.Duration
    Timeout bool
}

func Traceroute(host string, maxHops int, probes int) ([]TracerouteHop, error) {
    destIP, err := net.ResolveIPAddr("ip4", host)
    if err != nil {
        return nil, fmt.Errorf("resolve failed: %w", err)
    }
    
    var hops []TracerouteHop
    
    for ttl := 1; ttl <= maxHops; ttl++ {
        hop := TracerouteHop{TTL: ttl}
        
        for probe := 0; probe < probes; probe++ {
            conn, err := net.DialTimeout("ip4:icmp", destIP.String(), 3*time.Second)
            if err != nil {
                hop.Timeout = true
                continue
            }
            
            // Set TTL
            rawConn, err := conn.(*net.IPConn).SyscallConn()
            if err != nil {
                conn.Close()
                continue
            }
            rawConn.Control(func(fd uintptr) {
                // syscall.SetsockoptInt(int(fd), syscall.IPPROTO_IP, syscall.IP_TTL, ttl)
            })
            
            start := time.Now()
            // Send ICMP echo and wait for reply or TTL exceeded
            conn.SetDeadline(time.Now().Add(3 * time.Second))
            
            rtt := time.Since(start)
            hop.RTTs = append(hop.RTTs, rtt)
            hop.Address = conn.RemoteAddr().String()
            
            conn.Close()
        }
        
        hops = append(hops, hop)
        
        if hop.Address == destIP.String() {
            break
        }
    }
    
    return hops, nil
}

// Network connection pool with metrics
type ConnPoolMetrics struct {
    ActiveConns   int64
    IdleConns     int64
    TotalCreated  int64
    TotalClosed   int64
    WaitCount     int64
    WaitDuration  time.Duration
    MaxLifetime   time.Duration
    mu            sync.Mutex
}

type ManagedConnPool struct {
    idle     chan net.Conn
    active   int64
    maxSize  int
    factory  func() (net.Conn, error)
    metrics  *ConnPoolMetrics
    mu       sync.Mutex
}

func NewManagedConnPool(maxSize int, factory func() (net.Conn, error)) *ManagedConnPool {
    return &ManagedConnPool{
        idle:    make(chan net.Conn, maxSize),
        maxSize: maxSize,
        factory: factory,
        metrics: &ConnPoolMetrics{},
    }
}

func (p *ManagedConnPool) Get(ctx context.Context) (net.Conn, error) {
    // Try idle connection first
    select {
    case conn := <-p.idle:
        atomic.AddInt64(&p.metrics.IdleConns, -1)
        atomic.AddInt64(&p.metrics.ActiveConns, 1)
        return conn, nil
    default:
    }
    
    // Create new if under limit
    p.mu.Lock()
    total := atomic.LoadInt64(&p.metrics.ActiveConns) + atomic.LoadInt64(&p.metrics.IdleConns)
    if int(total) < p.maxSize {
        p.mu.Unlock()
        conn, err := p.factory()
        if err != nil {
            return nil, err
        }
        atomic.AddInt64(&p.metrics.ActiveConns, 1)
        atomic.AddInt64(&p.metrics.TotalCreated, 1)
        return conn, nil
    }
    p.mu.Unlock()
    
    // Wait for idle connection
    atomic.AddInt64(&p.metrics.WaitCount, 1)
    waitStart := time.Now()
    
    select {
    case conn := <-p.idle:
        p.metrics.mu.Lock()
        p.metrics.WaitDuration += time.Since(waitStart)
        p.metrics.mu.Unlock()
        atomic.AddInt64(&p.metrics.IdleConns, -1)
        atomic.AddInt64(&p.metrics.ActiveConns, 1)
        return conn, nil
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

func (p *ManagedConnPool) Put(conn net.Conn) {
    atomic.AddInt64(&p.metrics.ActiveConns, -1)
    
    select {
    case p.idle <- conn:
        atomic.AddInt64(&p.metrics.IdleConns, 1)
    default:
        conn.Close()
        atomic.AddInt64(&p.metrics.TotalClosed, 1)
    }
}`,
				},
			},
		},
	})
}
