package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2623,
			Title:       "Proxy Servers and Content Delivery",
			Description: "Understand forward and reverse proxies, CDN architecture, edge computing, content caching, and traffic optimization strategies.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Proxies, CDNs, and Content Delivery",
					Content: `Proxy servers and CDNs are fundamental to modern internet architecture for performance, security, and scalability.

**Forward Proxy:**

  Client -> Forward Proxy -> Internet -> Server
  
  Client knows about the proxy
  Server doesn't know about the client
  
  Use cases:
    Corporate internet filtering
    Anonymous browsing
    Geo-restriction bypass
    Caching for bandwidth savings
    Content filtering and monitoring
  
  Types:
    HTTP Proxy: Handles HTTP/HTTPS traffic
    SOCKS Proxy: Protocol-agnostic (SOCKS4, SOCKS5)
    Transparent Proxy: Interceptrts without client configuration
    
  CONNECT method (for HTTPS):
    Client: CONNECT example.com:443 HTTP/1.1
    Proxy: 200 Connection Established
    Client then establishes TLS directly with server
    Proxy cannot inspect encrypted content

**Reverse Proxy:**

  Client -> Reverse Proxy -> Backend Server(s)
  
  Client doesn't know about the backend
  Server is protected behind the proxy
  
  Capabilities:
    Load balancing across backends
    SSL/TLS termination
    Caching static content
    Compression (gzip, brotli)
    HTTP/2 and HTTP/3 support
    Request/response modification
    Rate limiting and WAF
    Authentication and authorization
    
  Popular reverse proxies:
    Nginx: High-performance, event-driven
    HAProxy: TCP/HTTP load balancing
    Envoy: Cloud-native proxy (service mesh)
    Caddy: Automatic HTTPS
    Traefik: Dynamic configuration, Docker/K8s native

**SSL/TLS Termination:**

  Options:
    At load balancer: Simple, offloads crypto from backends
      Client --(HTTPS)--> LB --(HTTP)--> Backend
      
    At backend: End-to-end encryption
      Client --(HTTPS)--> LB --(HTTPS)--> Backend
      
    Re-encryption: Terminate and re-encrypt
      Client --(HTTPS)--> LB --(new HTTPS)--> Backend
      
    Pass-through: LB doesn't inspect traffic (L4)
      Client --(HTTPS)--> LB --(HTTPS)--> Backend (same TLS session)

**CDN (Content Delivery Network):**

  Architecture:
    Origin Server: Has the original content
    Edge Servers: Cache copies close to users
    PoPs (Points of Presence): Data centers worldwide
    
    User -> Nearest Edge -> [Cache Hit] -> Serve content
                         -> [Cache Miss] -> Origin -> Cache -> Serve
  
  Content Types:
    Static: Images, CSS, JS, fonts, videos
    Dynamic: API responses, personalized content
    Streaming: Video/audio (HLS, DASH)
    
  Cache Control Headers:
    Cache-Control: public, max-age=31536000
    Cache-Control: private, no-cache
    Cache-Control: no-store
    ETag: "abc123" (conditional requests)
    Last-Modified: Wed, 01 Jan 2025 00:00:00 GMT
    Vary: Accept-Encoding (cache per encoding)
    
  Cache Invalidation:
    TTL expiration: Automatic after timeout
    Purge API: Explicitly remove cached content
    Versioned URLs: style.v2.css (never invalidate, new URL)
    Surrogate Keys: Tag content, purge by tag

  CDN Features:
    Edge compute: Run code at edge (Cloudflare Workers, Lambda@Edge)
    Image optimization: Resize, compress, format conversion
    Video transcoding: Multiple bitrates, codecs
    DDoS protection: Absorb attack traffic
    WAF: Web Application Firewall at edge
    Bot management: Detect and block bots
    Real User Monitoring (RUM): Performance metrics

  Major CDN Providers:
    Cloudflare: Global CDN, DNS, security
    AWS CloudFront: Integration with AWS services
    Akamai: Largest CDN, enterprise focus
    Fastly: Real-time purging, edge compute (VCL/Wasm)
    Google Cloud CDN: Integration with GCP

**Edge Computing:**

  Process data close to where it's generated
  Reduce latency and bandwidth
  
  Edge Tiers:
    Far Edge: IoT devices, sensors
    Near Edge: Cell towers, micro data centers
    Regional Edge: CDN PoPs, cloud regions
    
  Use cases:
    Real-time video processing
    IoT data aggregation
    AR/VR rendering
    Gaming (low-latency)
    Content personalization

**HTTP Caching Strategy:**

  Immutable Content (cache forever):
    Static assets with content hash in filename
    Cache-Control: public, max-age=31536000, immutable
    style.a1b2c3.css, bundle.d4e5f6.js
    
  Short-lived Content (revalidate):
    API responses, HTML pages
    Cache-Control: public, max-age=60, stale-while-revalidate=300
    ETag for conditional requests (If-None-Match)
    
  Private Content (no shared cache):
    User-specific data
    Cache-Control: private, max-age=0, must-revalidate
    
  No Cache:
    Sensitive data, real-time info
    Cache-Control: no-store
    
Stale-While-Revalidate:
  Serve stale content while fetching fresh in background
  User gets fast response (cache hit)
  Cache refreshed asynchronously
  Best for content that changes but doesn't need to be instant

**Compression:**

  gzip: Widely supported, good compression
  Brotli (br): Better compression ratio, slower
  zstd: Facebook's algorithm, great for streaming
  
  Content-Encoding: gzip
  Accept-Encoding: gzip, deflate, br
  
  Best practices:
    Compress text (HTML, CSS, JS, JSON, XML)
    Don't compress already compressed (images, video, zip)
    Pre-compress static files at build time
    Real-time compression for dynamic content
    Minimum size threshold (~1KB)`,
					CodeExamples: `// Proxy and CDN implementation patterns

package main

import (
    "compress/gzip"
    "crypto/sha256"
    "fmt"
    "io"
    "net/http"
    "net/http/httputil"
    "net/url"
    "strings"
    "sync"
    "time"
)

// Simple reverse proxy with caching
type CachingReverseProxy struct {
    target   *url.URL
    proxy    *httputil.ReverseProxy
    cache    *ContentCache
    logger   func(string, ...interface{})
}

func NewCachingReverseProxy(target string) (*CachingReverseProxy, error) {
    u, err := url.Parse(target)
    if err != nil {
        return nil, err
    }
    
    p := &CachingReverseProxy{
        target: u,
        proxy:  httputil.NewSingleHostReverseProxy(u),
        cache:  NewContentCache(1000),
    }
    
    return p, nil
}

func (p *CachingReverseProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Only cache GET requests
    if r.Method != "GET" {
        p.proxy.ServeHTTP(w, r)
        return
    }
    
    cacheKey := r.URL.String()
    
    // Check cache
    if entry, ok := p.cache.Get(cacheKey); ok {
        w.Header().Set("X-Cache", "HIT")
        w.Header().Set("Content-Type", entry.ContentType)
        w.Header().Set("ETag", entry.ETag)
        
        // Conditional request
        if r.Header.Get("If-None-Match") == entry.ETag {
            w.WriteHeader(http.StatusNotModified)
            return
        }
        
        w.Write(entry.Body)
        return
    }
    
    // Cache miss - proxy to backend
    w.Header().Set("X-Cache", "MISS")
    recorder := &responseRecorder{ResponseWriter: w, body: &strings.Builder{}}
    p.proxy.ServeHTTP(recorder, r)
    
    // Cache the response if cacheable
    if recorder.statusCode == 200 && isCacheable(recorder.Header()) {
        body := []byte(recorder.body.String())
        hash := sha256.Sum256(body)
        etag := fmt.Sprintf("\"%%x\"", hash)
        
        p.cache.Set(cacheKey, &CacheEntry{
            Body:        body,
            ContentType: recorder.Header().Get("Content-Type"),
            ETag:        etag,
            CachedAt:    time.Now(),
            TTL:         parseCacheControl(recorder.Header().Get("Cache-Control")),
        })
    }
}

type responseRecorder struct {
    http.ResponseWriter
    statusCode int
    body       *strings.Builder
}

func (r *responseRecorder) WriteHeader(code int) {
    r.statusCode = code
    r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(b []byte) (int, error) {
    r.body.Write(b)
    return r.ResponseWriter.Write(b)
}

// Content cache with TTL
type ContentCache struct {
    mu       sync.RWMutex
    entries  map[string]*CacheEntry
    maxSize  int
}

type CacheEntry struct {
    Body        []byte
    ContentType string
    ETag        string
    CachedAt    time.Time
    TTL         time.Duration
}

func NewContentCache(maxSize int) *ContentCache {
    return &ContentCache{
        entries: make(map[string]*CacheEntry),
        maxSize: maxSize,
    }
}

func (c *ContentCache) Get(key string) (*CacheEntry, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    entry, ok := c.entries[key]
    if !ok {
        return nil, false
    }
    
    if time.Since(entry.CachedAt) > entry.TTL {
        return nil, false
    }
    
    return entry, true
}

func (c *ContentCache) Set(key string, entry *CacheEntry) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if len(c.entries) >= c.maxSize {
        c.evictOldest()
    }
    
    c.entries[key] = entry
}

func (c *ContentCache) evictOldest() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, entry := range c.entries {
        if oldestKey == "" || entry.CachedAt.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.CachedAt
        }
    }
    
    if oldestKey != "" {
        delete(c.entries, oldestKey)
    }
}

func (c *ContentCache) Purge(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    delete(c.entries, key)
}

func (c *ContentCache) PurgeAll() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.entries = make(map[string]*CacheEntry)
}

func isCacheable(headers http.Header) bool {
    cc := headers.Get("Cache-Control")
    if strings.Contains(cc, "no-store") || strings.Contains(cc, "private") {
        return false
    }
    return true
}

func parseCacheControl(cc string) time.Duration {
    parts := strings.Split(cc, ",")
    for _, part := range parts {
        part = strings.TrimSpace(part)
        if strings.HasPrefix(part, "max-age=") {
            var seconds int
            fmt.Sscanf(part, "max-age=%d", &seconds)
            return time.Duration(seconds) * time.Second
        }
    }
    return 5 * time.Minute // Default TTL
}

// Gzip compression middleware
type GzipMiddleware struct {
    level     int
    minSize   int
    mimeTypes map[string]bool
}

func NewGzipMiddleware() *GzipMiddleware {
    return &GzipMiddleware{
        level:   gzip.DefaultCompression,
        minSize: 1024,
        mimeTypes: map[string]bool{
            "text/html":               true,
            "text/css":                true,
            "text/javascript":         true,
            "application/javascript":  true,
            "application/json":        true,
            "application/xml":         true,
            "text/xml":                true,
            "image/svg+xml":           true,
        },
    }
}

func (m *GzipMiddleware) Handler(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
            next.ServeHTTP(w, r)
            return
        }
        
        gz, err := gzip.NewWriterLevel(w, m.level)
        if err != nil {
            next.ServeHTTP(w, r)
            return
        }
        defer gz.Close()
        
        w.Header().Set("Content-Encoding", "gzip")
        w.Header().Del("Content-Length")
        w.Header().Set("Vary", "Accept-Encoding")
        
        gzw := &gzipResponseWriter{
            ResponseWriter: w,
            Writer:         gz,
        }
        
        next.ServeHTTP(gzw, r)
    })
}

type gzipResponseWriter struct {
    http.ResponseWriter
    Writer io.Writer
}

func (w *gzipResponseWriter) Write(b []byte) (int, error) {
    return w.Writer.Write(b)
}

// Cache warming utility
type CacheWarmer struct {
    urls    []string
    client  *http.Client
    workers int
}

func NewCacheWarmer(urls []string, workers int) *CacheWarmer {
    return &CacheWarmer{
        urls:    urls,
        client:  &http.Client{Timeout: 10 * time.Second},
        workers: workers,
    }
}

func (cw *CacheWarmer) Warm() []WarmResult {
    results := make([]WarmResult, len(cw.urls))
    urlChan := make(chan int, cw.workers)
    var wg sync.WaitGroup
    
    for i := 0; i < cw.workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for idx := range urlChan {
                start := time.Now()
                resp, err := cw.client.Get(cw.urls[idx])
                duration := time.Since(start)
                
                r := WarmResult{
                    URL:     cw.urls[idx],
                    Latency: duration,
                }
                
                if err != nil {
                    r.Error = err.Error()
                } else {
                    r.StatusCode = resp.StatusCode
                    r.CacheStatus = resp.Header.Get("X-Cache")
                    resp.Body.Close()
                }
                
                results[idx] = r
            }
        }()
    }
    
    for i := range cw.urls {
        urlChan <- i
    }
    close(urlChan)
    wg.Wait()
    
    return results
}

type WarmResult struct {
    URL         string
    StatusCode  int
    Latency     time.Duration
    CacheStatus string
    Error       string
}`,
				},
			},
		},
		{
			ID:          2624,
			Title:       "Network Automation and Programmability",
			Description: "Automate network operations with network APIs, configuration management, NETCONF/YANG, programmable switches, and network testing frameworks.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Network Automation and APIs",
					Content: `Network automation replaces manual configuration with programmable, repeatable processes.

**Network Configuration Management:**

Traditional (manual):
  SSH into device -> Enter commands -> Hope for no typos
  Problems: Inconsistency, human error, no audit trail, slow

Automated:
  Define desired state -> Tool applies changes -> Verify
  Benefits: Consistency, audit trail, speed, reproducibility

Configuration Management Tools:
  Ansible (agentless):
    SSH-based, YAML playbooks
    Network modules for Cisco, Juniper, Arista
    Idempotent operations
    
  Terraform (infrastructure as code):
    Manages cloud network resources
    VPCs, subnets, security groups, load balancers
    State management and drift detection
    
  Nornir (Python framework):
    Python-native automation framework
    Inventory management
    Task-based execution
    Plugin system

**Network APIs:**

REST APIs:
  Most modern network devices expose REST
  CRUD operations on network configuration
  JSON request/response
  
  Examples:
    GET /api/v1/interfaces: List interfaces
    PUT /api/v1/interfaces/eth0: Configure interface
    POST /api/v1/vlans: Create VLAN
    DELETE /api/v1/acls/rule/5: Delete ACL rule

gNMI (gRPC Network Management Interface):
  Google's network management protocol
  Based on gRPC (uses Protocol Buffers)
  Subscribe to streaming telemetry
  Set/Get configuration
  
  Operations:
    Capabilities: What does the device support?
    Get: Read configuration or state
    Set: Modify configuration
    Subscribe: Stream telemetry data

NETCONF (Network Configuration Protocol):
  XML-based protocol over SSH
  YANG data models define configuration schema
  Transaction-based (commit/rollback)
  
  Operations:
    get-config: Read configuration
    edit-config: Modify configuration
    copy-config: Copy between datastores
    lock/unlock: Prevent concurrent changes
    commit: Apply candidate to running
    
  Datastores:
    running: Active configuration
    candidate: Staged changes (not yet applied)
    startup: Loaded at boot

YANG Data Models:
  Defines structure of network configuration
  Modules describe devices, interfaces, routing
  Standard models: OpenConfig, IETF
  Vendor-specific models: Cisco, Juniper
  
  Example YANG structure:
    module openconfig-interfaces {
      container interfaces {
        list interface {
          key "name";
          leaf name { type string; }
          container config {
            leaf enabled { type boolean; }
            leaf mtu { type uint16; }
          }
          container state {
            leaf oper-status { type enumeration; }
            leaf counters { ... }
          }
        }
      }
    }

**Network Testing:**

Configuration Validation:
  Syntax checking before deployment
  Policy compliance verification
  Simulate configuration changes
  Tools: Batfish, CI/CD pipelines

Connectivity Testing:
  Automated ping/traceroute after changes
  Verify expected paths and latencies
  Test failover scenarios
  
Network Simulation:
  GNS3: Network topology simulation
  EVE-NG: Multi-vendor network emulation
  Containerlab: Container-based network labs
  Mininet: SDN experimentation

Chaos Testing for Networks:
  Inject latency, packet loss, DNS failures
  tc (traffic control) for Linux
  Toxiproxy: Programmable network proxy
  
  tc qdisc add dev eth0 root netem delay 100ms 20ms
  tc qdisc add dev eth0 root netem loss 5%
  tc qdisc add dev eth0 root netem corrupt 1%

**Intent-Based Networking (IBN):**

  Describe WHAT you want, not HOW to configure
  System translates intent to device configuration
  Continuous verification of intent
  
  Example intent:
    "Web servers in VLAN 10 can talk to database servers in VLAN 20 on port 5432"
    
  System generates:
    VLAN configurations on switches
    ACL rules on firewalls
    Security group rules in cloud
    Routing entries as needed

**Network as Code:**

  Version-controlled network configuration
  CI/CD pipeline for network changes
  
  Pipeline:
    1. Developer creates branch with config changes
    2. Linting and syntax validation
    3. Simulation/dry-run (Batfish)
    4. Peer review and approval
    5. Deploy to staging/lab environment
    6. Automated testing
    7. Deploy to production
    8. Post-deployment verification

  GitOps for networking:
    Git as single source of truth
    Automated reconciliation
    Drift detection and correction
    Audit trail through Git history`,
					CodeExamples: `// Network automation patterns in Go

package main

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "net"
    "net/http"
    "sync"
    "time"
)

// Network device client
type NetworkDevice struct {
    Address  string
    Username string
    Password string
    client   *http.Client
}

type Interface struct {
    Name        string "json:\"name\""
    Description string "json:\"description\""
    Enabled     bool   "json:\"enabled\""
    MTU         int    "json:\"mtu\""
    IPAddress   string "json:\"ip_address,omitempty\""
    Speed       string "json:\"speed,omitempty\""
    Duplex      string "json:\"duplex,omitempty\""
    OperStatus  string "json:\"oper_status,omitempty\""
}

func NewNetworkDevice(address, username, password string) *NetworkDevice {
    return &NetworkDevice{
        Address:  address,
        Username: username,
        Password: password,
        client: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (d *NetworkDevice) GetInterfaces(ctx context.Context) ([]Interface, error) {
    req, err := http.NewRequestWithContext(ctx, "GET",
        fmt.Sprintf("https://%s/api/v1/interfaces", d.Address), nil)
    if err != nil {
        return nil, err
    }
    req.SetBasicAuth(d.Username, d.Password)
    
    resp, err := d.client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 {
        return nil, fmt.Errorf("unexpected status: %d", resp.StatusCode)
    }
    
    var interfaces []Interface
    if err := json.NewDecoder(resp.Body).Decode(&interfaces); err != nil {
        return nil, fmt.Errorf("decode failed: %w", err)
    }
    
    return interfaces, nil
}

func (d *NetworkDevice) ConfigureInterface(ctx context.Context, iface Interface) error {
    body, err := json.Marshal(iface)
    if err != nil {
        return err
    }
    
    req, err := http.NewRequestWithContext(ctx, "PUT",
        fmt.Sprintf("https://%s/api/v1/interfaces/%s", d.Address, iface.Name),
        bytes.NewReader(body))
    if err != nil {
        return err
    }
    req.SetBasicAuth(d.Username, d.Password)
    req.Header.Set("Content-Type", "application/json")
    
    resp, err := d.client.Do(req)
    if err != nil {
        return fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 && resp.StatusCode != 204 {
        return fmt.Errorf("configure failed with status: %d", resp.StatusCode)
    }
    
    return nil
}

// Network inventory and bulk operations
type NetworkInventory struct {
    Devices []DeviceInfo
}

type DeviceInfo struct {
    Hostname string
    Address  string
    Role     string
    Site     string
    OS       string
    Username string
    Password string
}

type BulkResult struct {
    Device  string
    Success bool
    Error   string
    Output  string
}

func (inv *NetworkInventory) BulkConfigure(ctx context.Context, configFn func(*NetworkDevice) error) []BulkResult {
    results := make([]BulkResult, len(inv.Devices))
    var wg sync.WaitGroup
    sem := make(chan struct{}, 10) // Max 10 concurrent
    
    for i, device := range inv.Devices {
        wg.Add(1)
        go func(idx int, dev DeviceInfo) {
            defer wg.Done()
            sem <- struct{}{}
            defer func() { <-sem }()
            
            client := NewNetworkDevice(dev.Address, dev.Username, dev.Password)
            err := configFn(client)
            
            result := BulkResult{
                Device:  dev.Hostname,
                Success: err == nil,
            }
            if err != nil {
                result.Error = err.Error()
            }
            results[idx] = result
        }(i, device)
    }
    
    wg.Wait()
    return results
}

// Connectivity test framework
type ConnectivityTest struct {
    Name     string
    Source   string
    Target   string
    Port     int
    Protocol string
    Expected bool
    Timeout  time.Duration
}

type TestResult struct {
    Test    ConnectivityTest
    Passed  bool
    Latency time.Duration
    Error   string
}

func RunConnectivityTests(tests []ConnectivityTest) []TestResult {
    results := make([]TestResult, len(tests))
    
    var wg sync.WaitGroup
    for i, test := range tests {
        wg.Add(1)
        go func(idx int, t ConnectivityTest) {
            defer wg.Done()
            results[idx] = runSingleTest(t)
        }(i, test)
    }
    
    wg.Wait()
    return results
}

func runSingleTest(test ConnectivityTest) TestResult {
    result := TestResult{Test: test}
    
    timeout := test.Timeout
    if timeout == 0 {
        timeout = 5 * time.Second
    }
    
    addr := fmt.Sprintf("%s:%d", test.Target, test.Port)
    start := time.Now()
    
    conn, err := net.DialTimeout(test.Protocol, addr, timeout)
    result.Latency = time.Since(start)
    
    connected := err == nil
    if conn != nil {
        conn.Close()
    }
    
    result.Passed = connected == test.Expected
    if err != nil && test.Expected {
        result.Error = err.Error()
    }
    if err == nil && !test.Expected {
        result.Error = "connection succeeded but expected failure"
    }
    
    return result
}

// Network change validator
type ChangeValidator struct {
    preChecks  []ConnectivityTest
    postChecks []ConnectivityTest
}

type ChangeValidation struct {
    PreResults  []TestResult
    PostResults []TestResult
    AllPassed   bool
    Regressions []string
}

func (v *ChangeValidator) ValidateChange(changeFn func() error) (*ChangeValidation, error) {
    validation := &ChangeValidation{}
    
    // Run pre-change checks
    validation.PreResults = RunConnectivityTests(v.preChecks)
    
    // Apply change
    if err := changeFn(); err != nil {
        return validation, fmt.Errorf("change failed: %w", err)
    }
    
    // Wait for convergence
    time.Sleep(5 * time.Second)
    
    // Run post-change checks
    validation.PostResults = RunConnectivityTests(v.postChecks)
    
    // Check for regressions
    validation.AllPassed = true
    for _, result := range validation.PostResults {
        if !result.Passed {
            validation.AllPassed = false
            validation.Regressions = append(validation.Regressions,
                fmt.Sprintf("%s: %s -> %s:%d (%s)",
                    result.Test.Name, result.Test.Source,
                    result.Test.Target, result.Test.Port,
                    result.Error))
        }
    }
    
    return validation, nil
}`,
				},
			},
		},
	})
}
