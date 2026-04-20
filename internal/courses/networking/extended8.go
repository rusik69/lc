package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2625,
			Title:       "Network Troubleshooting and Analysis",
			Description: "Master network troubleshooting with packet capture analysis, traffic analysis, network forensics, and systematic debugging approaches.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Packet Analysis and Network Debugging",
					Content: `Effective network troubleshooting combines systematic methodology with deep protocol knowledge.

**Packet Capture Tools:**

tcpdump:
  Command-line packet capture and analysis
  
  Common filters:
    tcpdump -i eth0: Capture on specific interface
    tcpdump host 10.0.0.1: Traffic to/from host
    tcpdump port 80: HTTP traffic
    tcpdump -n src 10.0.0.1 and dst port 443: Specific flow
    tcpdump -w capture.pcap: Write to file
    tcpdump -r capture.pcap: Read from file
    tcpdump -c 100: Capture 100 packets then stop
    tcpdump -A: Print packet content as ASCII
    tcpdump -X: Print in hex and ASCII
    
  Advanced filters:
    tcpdump 'tcp[tcpflags] & (tcp-syn) != 0': SYN packets
    tcpdump 'tcp[tcpflags] & (tcp-rst) != 0': RST packets
    tcpdump 'tcp[13] & 0x02 != 0': SYN via bit mask
    tcpdump 'ip[2:2] > 576': Large packets
    tcpdump 'tcp[20:2] = 0x4745': HTTP GET requests

Wireshark/tshark:
  GUI (Wireshark) and CLI (tshark) packet analyzer
  Deep protocol analysis
  Follow TCP streams
  Protocol statistics
  Display filters (different from capture filters)
  
  Display filters:
    http.request.method == "GET"
    tcp.analysis.retransmission
    dns.query.name contains "example"
    frame.time_delta > 0.5
    tcp.stream eq 5

**TCP Connection Analysis:**

Healthy Connection:
  Clean 3-way handshake (SYN, SYN-ACK, ACK)
  Reasonable RTT (consistent with distance)
  No retransmissions
  Clean shutdown (FIN, FIN-ACK, ACK)

Common TCP Problems:

SYN Flood:
  Many SYN packets, no ACK
  Sign: Half-open connections filling backlog
  Diagnosis: tcpdump 'tcp[tcpflags] == tcp-syn' | count
  Fix: SYN cookies, increase backlog, rate limiting

Retransmissions:
  Same data sent multiple times
  Sign of packet loss or receive buffer issues
  Diagnosis: tcp.analysis.retransmission in Wireshark
  Metrics: Retransmission rate should be < 1%
  Causes: Congestion, faulty hardware, buffer overflow

Window Size Issues:
  Zero window: Receiver buffer full
  Window scaling problems: Middleboxes stripping options
  Diagnosis: tcp.window_size_value == 0
  Fix: Tune buffer sizes, fix middlebox issues

RST Packets:
  Connection reset unexpectedly
  Causes: Firewall, port not listening, application crash
  Diagnosis: Count RST packets per destination

Duplicate ACKs:
  Three duplicate ACKs trigger fast retransmit
  Sign of packet loss (gap in sequence numbers)
  Diagnosis: tcp.analysis.duplicate_ack

**DNS Troubleshooting:**

Tools:
  dig: DNS query tool (recommended)
    dig example.com A
    dig @8.8.8.8 example.com MX
    dig example.com +trace (full resolution path)
    dig example.com +short (concise output)
    
  nslookup: Cross-platform DNS query
    nslookup example.com
    nslookup -type=MX example.com
    
  host: Simple DNS lookups
    host example.com
    host -t NS example.com

Common DNS Issues:
  NXDOMAIN: Domain doesn't exist
  SERVFAIL: Server error (misconfigured zone, DNSSEC failure)
  REFUSED: Server refused query (ACL)
  Timeout: Server unreachable or too slow
  Cached stale data: Wait for TTL or flush cache

**HTTP Troubleshooting:**

curl for HTTP debugging:
  curl -v: Verbose (headers, TLS handshake)
  curl -I: HEAD request (headers only)
  curl -L: Follow redirects
  curl -k: Skip TLS verification (debugging only)
  curl --resolve host:port:ip: Override DNS
  curl -w '@format.txt': Custom output format
  curl --connect-timeout 5 --max-time 30: Timeouts
  
  Timing info:
    curl -w "DNS: %{time_namelookup}s\nConnect: %{time_connect}s\nTLS: %{time_appconnect}s\nFirst Byte: %{time_starttransfer}s\nTotal: %{time_total}s\n"

Common HTTP Issues:
  301/302 loops: Redirect chains
  403: Authentication/authorization failure
  404: Wrong URL or path
  499: Client closed connection (Nginx)
  500: Internal server error (check server logs)
  502: Bad Gateway (upstream down)
  503: Service Unavailable (overloaded)
  504: Gateway Timeout (upstream too slow)

**Network Performance Debugging:**

ss (Socket Statistics):
  ss -tuln: Listening TCP/UDP sockets
  ss -tn state established: Active connections
  ss -s: Summary statistics
  ss -tnp: Show process names
  ss -i: Internal TCP info (cwnd, rtt, retrans)

iperf3 (Bandwidth Testing):
  Server: iperf3 -s
  Client: iperf3 -c server-ip
  UDP test: iperf3 -c server-ip -u -b 100M
  Bidirectional: iperf3 -c server-ip --bidir
  Parallel streams: iperf3 -c server-ip -P 4

mtr (My Traceroute):
  Combines ping and traceroute
  Continuous monitoring
  Shows packet loss per hop
  mtr -n --report server-ip

**Systematic Troubleshooting:**

Step 1: Define the Problem
  What exactly isn't working?
  When did it start?
  What changed recently?
  Who/what is affected?

Step 2: Gather Information
  Network topology
  Error messages
  Recent changes
  Affected and unaffected components

Step 3: Isolate the Cause
  Top-down (application -> network -> physical)
  Bottom-up (physical -> network -> application)
  Divide and conquer (test middle point)
  Follow the packet path

Step 4: Test Theory
  Make one change at a time
  Verify each change
  Document what you tried

Step 5: Verify Resolution
  Confirm the original problem is fixed
  Check for side effects
  Document the solution

**Common Network Debugging Scenarios:**

"Can't connect to service":
  1. DNS resolution working? (dig, nslookup)
  2. IP reachable? (ping, traceroute)
  3. Port open? (telnet, nc, ss)
  4. Firewall rules? (iptables -L, nft list)
  5. Service running? (systemctl status, docker ps)
  6. Application logs? (journalctl, docker logs)

"Slow application":
  1. Is it DNS? (measure lookup time)
  2. Is it network? (ping, mtr for latency/loss)
  3. Is it TLS? (measure handshake time)
  4. Is it server? (time to first byte)
  5. Is it payload? (content size, compression)
  6. Is it bandwidth? (iperf3)

"Intermittent connectivity":
  1. Packet loss pattern? (mtr, continuous ping)
  2. Specific hop? (traceroute analysis)
  3. CPU/memory on network devices?
  4. Interface errors? (ethtool -S, ip -s link)
  5. Cable/hardware issues? (error counters)
  6. MTU issues? (test with different packet sizes)`,
					CodeExamples: `// Network troubleshooting tools in Go

package main

import (
    "context"
    "fmt"
    "net"
    "net/http"
    "strings"
    "sync"
    "time"
)

// Comprehensive connectivity checker
type ConnectivityChecker struct {
    timeout time.Duration
}

type ConnCheckResult struct {
    Target        string
    DNSResolve    *StepResult
    TCPConnect    *StepResult
    TLSHandshake  *StepResult
    HTTPRequest   *StepResult
    OverallStatus string
}

type StepResult struct {
    Success bool
    Latency time.Duration
    Error   string
    Details map[string]string
}

func NewConnectivityChecker(timeout time.Duration) *ConnectivityChecker {
    return &ConnectivityChecker{timeout: timeout}
}

func (c *ConnectivityChecker) Check(ctx context.Context, target string, port int) *ConnCheckResult {
    result := &ConnCheckResult{Target: target}
    
    // Step 1: DNS Resolution
    dnsStart := time.Now()
    ips, err := net.DefaultResolver.LookupHost(ctx, target)
    result.DNSResolve = &StepResult{
        Success: err == nil,
        Latency: time.Since(dnsStart),
        Details: map[string]string{},
    }
    if err != nil {
        result.DNSResolve.Error = err.Error()
        result.OverallStatus = "DNS_FAILURE"
        return result
    }
    result.DNSResolve.Details["resolved_ips"] = strings.Join(ips, ", ")
    
    // Step 2: TCP Connection
    addr := fmt.Sprintf("%s:%d", ips[0], port)
    tcpStart := time.Now()
    conn, err := net.DialTimeout("tcp", addr, c.timeout)
    result.TCPConnect = &StepResult{
        Success: err == nil,
        Latency: time.Since(tcpStart),
        Details: map[string]string{"address": addr},
    }
    if err != nil {
        result.TCPConnect.Error = err.Error()
        result.OverallStatus = "TCP_FAILURE"
        return result
    }
    conn.Close()
    
    // Step 3: HTTP Request (if port 80 or 443)
    if port == 80 || port == 443 {
        scheme := "http"
        if port == 443 {
            scheme = "https"
        }
        
        httpStart := time.Now()
        client := &http.Client{Timeout: c.timeout}
        resp, err := client.Get(fmt.Sprintf("%s://%s", scheme, target))
        result.HTTPRequest = &StepResult{
            Success: err == nil,
            Latency: time.Since(httpStart),
            Details: map[string]string{},
        }
        if err != nil {
            result.HTTPRequest.Error = err.Error()
            result.OverallStatus = "HTTP_FAILURE"
            return result
        }
        resp.Body.Close()
        result.HTTPRequest.Details["status"] = fmt.Sprintf("%d", resp.StatusCode)
        result.HTTPRequest.Details["server"] = resp.Header.Get("Server")
    }
    
    result.OverallStatus = "OK"
    return result
}

// Port range scanner
type PortRangeScanner struct {
    target     string
    concurrent int
    timeout    time.Duration
}

type PortResult struct {
    Port    int
    State   string // open, closed, filtered
    Service string
    Banner  string
}

func NewPortRangeScanner(target string) *PortRangeScanner {
    return &PortRangeScanner{
        target:     target,
        concurrent: 200,
        timeout:    2 * time.Second,
    }
}

func (s *PortRangeScanner) ScanRange(start, end int) []PortResult {
    var results []PortResult
    var mu sync.Mutex
    
    sem := make(chan struct{}, s.concurrent)
    var wg sync.WaitGroup
    
    for port := start; port <= end; port++ {
        wg.Add(1)
        sem <- struct{}{}
        
        go func(p int) {
            defer wg.Done()
            defer func() { <-sem }()
            
            result := s.scanPort(p)
            if result.State == "open" {
                mu.Lock()
                results = append(results, result)
                mu.Unlock()
            }
        }(port)
    }
    
    wg.Wait()
    return results
}

func (s *PortRangeScanner) scanPort(port int) PortResult {
    addr := fmt.Sprintf("%s:%d", s.target, port)
    conn, err := net.DialTimeout("tcp", addr, s.timeout)
    if err != nil {
        if strings.Contains(err.Error(), "refused") {
            return PortResult{Port: port, State: "closed"}
        }
        return PortResult{Port: port, State: "filtered"}
    }
    defer conn.Close()
    
    result := PortResult{
        Port:    port,
        State:   "open",
        Service: wellKnownPort(port),
    }
    
    // Try banner grab
    conn.SetReadDeadline(time.Now().Add(time.Second))
    buf := make([]byte, 1024)
    n, _ := conn.Read(buf)
    if n > 0 {
        result.Banner = strings.TrimSpace(string(buf[:n]))
    }
    
    return result
}

func wellKnownPort(port int) string {
    services := map[int]string{
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
        53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP",
        443: "HTTPS", 993: "IMAPS", 995: "POP3S",
        3306: "MySQL", 5432: "PostgreSQL", 6379: "Redis",
        8080: "HTTP-ALT", 8443: "HTTPS-ALT", 27017: "MongoDB",
    }
    if svc, ok := services[port]; ok {
        return svc
    }
    return ""
}

// Network path analyzer
type PathAnalyzer struct {
    target   string
    maxHops  int
    probes   int
    timeout  time.Duration
}

type PathHop struct {
    Hop      int
    IP       string
    Hostname string
    RTTs     []time.Duration
    AvgRTT   time.Duration
    Loss     float64
    ASN      string
}

type PathAnalysis struct {
    Target      string
    Hops        []PathHop
    TotalHops   int
    Bottleneck  *PathHop
    PacketLoss  float64
}

func (a *PathAnalyzer) Analyze(hops []PathHop) *PathAnalysis {
    analysis := &PathAnalysis{
        Target:    a.target,
        Hops:      hops,
        TotalHops: len(hops),
    }
    
    // Find bottleneck (highest RTT increase)
    var maxIncrease time.Duration
    for i := 1; i < len(hops); i++ {
        if hops[i].AvgRTT == 0 || hops[i-1].AvgRTT == 0 {
            continue
        }
        increase := hops[i].AvgRTT - hops[i-1].AvgRTT
        if increase > maxIncrease {
            maxIncrease = increase
            analysis.Bottleneck = &hops[i]
        }
    }
    
    // Calculate overall loss
    if len(hops) > 0 {
        lastHop := hops[len(hops)-1]
        analysis.PacketLoss = lastHop.Loss
    }
    
    return analysis
}

// HTTP timing breakdown
type HTTPTimings struct {
    DNSLookup    time.Duration "json:\"dns_lookup\""
    TCPConnect   time.Duration "json:\"tcp_connect\""
    TLSHandshake time.Duration "json:\"tls_handshake\""
    ServerWait   time.Duration "json:\"server_wait\""
    ContentTransfer time.Duration "json:\"content_transfer\""
    Total        time.Duration "json:\"total\""
    StatusCode   int           "json:\"status_code\""
    ContentLength int64        "json:\"content_length\""
}

func MeasureHTTPTimings(targetURL string) (*HTTPTimings, error) {
    timings := &HTTPTimings{}
    
    totalStart := time.Now()
    
    // Measure DNS
    u, _ := url.Parse(targetURL)
    dnsStart := time.Now()
    ips, err := net.LookupHost(u.Hostname())
    timings.DNSLookup = time.Since(dnsStart)
    if err != nil {
        return timings, fmt.Errorf("DNS lookup failed: %w", err)
    }
    _ = ips
    
    // Measure TCP + TLS + HTTP
    client := &http.Client{
        Timeout: 30 * time.Second,
    }
    
    connectStart := time.Now()
    resp, err := client.Get(targetURL)
    if err != nil {
        return timings, err
    }
    defer resp.Body.Close()
    
    timings.StatusCode = resp.StatusCode
    timings.ContentLength = resp.ContentLength
    
    // Read body
    transferStart := time.Now()
    body := make([]byte, 0)
    buf := make([]byte, 32*1024)
    for {
        n, err := resp.Body.Read(buf)
        body = append(body, buf[:n]...)
        if err != nil {
            break
        }
    }
    timings.ContentTransfer = time.Since(transferStart)
    
    timings.Total = time.Since(totalStart)
    timings.TCPConnect = time.Since(connectStart) - timings.ContentTransfer
    
    return timings, nil
}

// Continuous network monitor
type NetworkMonitor struct {
    targets  []MonitorTarget
    interval time.Duration
    results  chan MonitorResult
}

type MonitorTarget struct {
    Name     string
    Host     string
    Port     int
    Protocol string
}

type MonitorResult struct {
    Target    MonitorTarget
    Timestamp time.Time
    Reachable bool
    Latency   time.Duration
    Error     string
}

func NewNetworkMonitor(targets []MonitorTarget, interval time.Duration) *NetworkMonitor {
    return &NetworkMonitor{
        targets:  targets,
        interval: interval,
        results:  make(chan MonitorResult, 100),
    }
}

func (m *NetworkMonitor) Start(ctx context.Context) <-chan MonitorResult {
    go func() {
        ticker := time.NewTicker(m.interval)
        defer ticker.Stop()
        defer close(m.results)
        
        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                for _, target := range m.targets {
                    go m.probe(target)
                }
            }
        }
    }()
    
    return m.results
}

func (m *NetworkMonitor) probe(target MonitorTarget) {
    addr := fmt.Sprintf("%s:%d", target.Host, target.Port)
    start := time.Now()
    
    conn, err := net.DialTimeout(target.Protocol, addr, 5*time.Second)
    latency := time.Since(start)
    
    result := MonitorResult{
        Target:    target,
        Timestamp: time.Now(),
        Reachable: err == nil,
        Latency:   latency,
    }
    
    if err != nil {
        result.Error = err.Error()
    } else {
        conn.Close()
    }
    
    select {
    case m.results <- result:
    default:
        // Drop if channel full
    }
}`,
				},
			},
		},
	})
}
