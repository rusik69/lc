package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2616,
			Title:       "Network Security Fundamentals",
			Description: "Understand network security with TLS/SSL, certificate management, VPNs, firewalls, intrusion detection systems, and common network attacks and defenses.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "TLS/SSL and Encryption",
					Content: `Transport Layer Security (TLS) is the foundation of secure communication on the internet.

**TLS Handshake (TLS 1.3):**

  Client                          Server
    |                                |
    |  ClientHello                   |
    |  (supported versions,          |
    |   cipher suites,               |
    |   key share)                   |
    | -----------------------------> |
    |                                |
    |  ServerHello                   |
    |  (selected version,            |
    |   cipher suite,                |
    |   key share)                   |
    |  EncryptedExtensions           |
    |  Certificate                   |
    |  CertificateVerify             |
    |  Finished                      |
    | <----------------------------- |
    |                                |
    |  Finished                      |
    | -----------------------------> |
    |                                |
    |  Application Data (encrypted)  |
    | <===========================>  |

TLS 1.3 improvements over 1.2:
  1-RTT handshake (was 2-RTT in TLS 1.2)
  0-RTT resumption for returning clients
  Removed insecure algorithms (RSA key exchange, CBC, RC4, SHA-1)
  Forward secrecy mandatory (ephemeral Diffie-Hellman only)
  Simplified cipher suites

**Certificate Chain:**

  Root CA (self-signed, in trust store)
    |
    ├── Intermediate CA (signed by Root)
    |     |
    |     └── Server Certificate (signed by Intermediate)
    |           Contains: Domain name, public key, validity dates
    |
    └── Intermediate CA 2
          └── Another Server Certificate

Certificate Validation:
  1. Check certificate chain up to trusted root
  2. Verify each signature in the chain
  3. Check expiration dates
  4. Check revocation (CRL or OCSP)
  5. Verify domain name matches (CN or SAN)

Certificate Types:
  DV (Domain Validation): Proves domain ownership
  OV (Organization Validation): Verifies organization identity
  EV (Extended Validation): Strictest verification

Let's Encrypt:
  Free, automated DV certificates
  ACME protocol for automation
  90-day validity, auto-renewal
  HTTP-01 or DNS-01 challenge for validation

**mTLS (Mutual TLS):**

  Both client and server present certificates
  Used for service-to-service authentication
  
  Standard TLS: Server authenticates to client
  mTLS: Both parties authenticate to each other
  
  Use cases:
    Microservice communication
    API authentication for partners
    Zero-trust network architecture
    IoT device authentication

**Cipher Suites:**

  TLS 1.3 cipher suites:
    TLS_AES_256_GCM_SHA384
    TLS_AES_128_GCM_SHA256
    TLS_CHACHA20_POLY1305_SHA256

  Components:
    Key Exchange: ECDHE (Elliptic Curve Diffie-Hellman Ephemeral)
    Authentication: RSA or ECDSA (certificate signature)
    Encryption: AES-GCM or ChaCha20-Poly1305
    Hash: SHA-256 or SHA-384

**Certificate Pinning:**
  
  Pin expected certificate or public key
  Prevents MITM with rogue certificates
  Hard to manage (rotation requires app updates)
  
  Alternatives:
    Certificate Transparency (CT) logs
    CAA DNS records
    DANE (DNS-based Authentication of Named Entities)

**Forward Secrecy:**

  Compromise of long-term key doesn't compromise past sessions
  Each session uses ephemeral keys
  If server private key is leaked:
    Without FS: All recorded traffic can be decrypted
    With FS: Only future sessions compromised
  
  Achieved with:
    ECDHE (Elliptic Curve Diffie-Hellman Ephemeral)
    DHE (Diffie-Hellman Ephemeral)

**HSTS (HTTP Strict Transport Security):**

  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
  
  Forces HTTPS for all future requests
  Prevents SSL stripping attacks
  HSTS preload list in browsers`,
					CodeExamples: `// Network security implementations in Go

package main

import (
    "crypto/tls"
    "crypto/x509"
    "fmt"
    "io"
    "net/http"
    "os"
    "time"
)

// TLS server configuration
func createTLSConfig() *tls.Config {
    return &tls.Config{
        MinVersion: tls.VersionTLS13,
        CurvePreferences: []tls.CurveID{
            tls.X25519,
            tls.CurveP256,
        },
        CipherSuites: []uint16{
            tls.TLS_AES_256_GCM_SHA384,
            tls.TLS_AES_128_GCM_SHA256,
            tls.TLS_CHACHA20_POLY1305_SHA256,
        },
    }
}

// HTTPS server with TLS 1.3
func startTLSServer(certFile, keyFile string) error {
    mux := http.NewServeMux()
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        fmt.Fprintf(w, "Hello, TLS!")
    })

    server := &http.Server{
        Addr:      ":443",
        Handler:   mux,
        TLSConfig: createTLSConfig(),
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
    }

    return server.ListenAndServeTLS(certFile, keyFile)
}

// mTLS server - requires client certificates
func startMTLSServer(certFile, keyFile, caFile string) error {
    caCert, err := os.ReadFile(caFile)
    if err != nil {
        return fmt.Errorf("read CA cert: %w", err)
    }

    caCertPool := x509.NewCertPool()
    if !caCertPool.AppendCertsFromPEM(caCert) {
        return fmt.Errorf("failed to parse CA certificate")
    }

    tlsConfig := &tls.Config{
        MinVersion: tls.VersionTLS13,
        ClientAuth: tls.RequireAndVerifyClientCert,
        ClientCAs:  caCertPool,
    }

    mux := http.NewServeMux()
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        if len(r.TLS.PeerCertificates) > 0 {
            clientCN := r.TLS.PeerCertificates[0].Subject.CommonName
            fmt.Fprintf(w, "Hello, %s (authenticated via mTLS)", clientCN)
        }
    })

    server := &http.Server{
        Addr:      ":8443",
        Handler:   mux,
        TLSConfig: tlsConfig,
    }

    return server.ListenAndServeTLS(certFile, keyFile)
}

// mTLS client
func createMTLSClient(clientCert, clientKey, caCert string) (*http.Client, error) {
    cert, err := tls.LoadX509KeyPair(clientCert, clientKey)
    if err != nil {
        return nil, fmt.Errorf("load client cert: %w", err)
    }

    caPEM, err := os.ReadFile(caCert)
    if err != nil {
        return nil, fmt.Errorf("read CA cert: %w", err)
    }

    caPool := x509.NewCertPool()
    if !caPool.AppendCertsFromPEM(caPEM) {
        return nil, fmt.Errorf("failed to parse CA certificate")
    }

    return &http.Client{
        Transport: &http.Transport{
            TLSClientConfig: &tls.Config{
                Certificates: []tls.Certificate{cert},
                RootCAs:      caPool,
                MinVersion:   tls.VersionTLS13,
            },
        },
        Timeout: 30 * time.Second,
    }, nil
}

// Certificate information extractor
func getCertificateInfo(addr string) (*CertInfo, error) {
    conn, err := tls.Dial("tcp", addr, &tls.Config{
        InsecureSkipVerify: false,
    })
    if err != nil {
        return nil, fmt.Errorf("TLS dial failed: %w", err)
    }
    defer conn.Close()

    state := conn.ConnectionState()
    if len(state.PeerCertificates) == 0 {
        return nil, fmt.Errorf("no peer certificates")
    }

    cert := state.PeerCertificates[0]
    info := &CertInfo{
        Subject:     cert.Subject.CommonName,
        Issuer:      cert.Issuer.CommonName,
        NotBefore:   cert.NotBefore,
        NotAfter:    cert.NotAfter,
        DNSNames:    cert.DNSNames,
        TLSVersion:  tlsVersionString(state.Version),
        CipherSuite: tls.CipherSuiteName(state.CipherSuite),
        ChainLength: len(state.PeerCertificates),
    }

    return info, nil
}

type CertInfo struct {
    Subject     string
    Issuer      string
    NotBefore   time.Time
    NotAfter    time.Time
    DNSNames    []string
    TLSVersion  string
    CipherSuite string
    ChainLength int
}

func tlsVersionString(v uint16) string {
    switch v {
    case tls.VersionTLS10:
        return "TLS 1.0"
    case tls.VersionTLS11:
        return "TLS 1.1"
    case tls.VersionTLS12:
        return "TLS 1.2"
    case tls.VersionTLS13:
        return "TLS 1.3"
    default:
        return fmt.Sprintf("Unknown (0x%04x)", v)
    }
}`,
				},
				{
					Title: "Firewalls, VPNs, and Network Defense",
					Content: `Network security architecture uses multiple layers of defense to protect against threats.

**Firewall Types:**

Packet Filter (Stateless):
  Inspects individual packets
  Checks source/destination IP, port, protocol
  No awareness of connection state
  Fast but limited protection
  Example: iptables basic rules

Stateful Firewall:
  Tracks connection state (NEW, ESTABLISHED, RELATED)
  Allows return traffic for established connections
  Better security than stateless
  Example: iptables with conntrack, nftables

Application Layer Firewall (WAF):
  Inspects application-layer data (HTTP, DNS, etc.)
  Can block SQL injection, XSS, path traversal
  Protocol-aware filtering
  Example: ModSecurity, AWS WAF, Cloudflare WAF

Next-Generation Firewall (NGFW):
  Deep packet inspection
  Application awareness
  Intrusion prevention
  TLS inspection
  Example: Palo Alto, Fortinet

**iptables / nftables:**

iptables chains:
  INPUT: Packets destined for the host
  OUTPUT: Packets originating from the host
  FORWARD: Packets being routed through the host

Common rules:
  Allow established connections:
    iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
  
  Allow SSH from specific subnet:
    iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT
  
  Allow HTTP/HTTPS:
    iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT
  
  Drop all other inbound:
    iptables -A INPUT -j DROP
  
  Rate limit connections:
    iptables -A INPUT -p tcp --dport 22 -m connlimit --connlimit-above 3 -j REJECT

nftables (modern replacement):
  Unified framework for packet filtering
  Better performance than iptables
  Atomic rule replacements
  Sets and maps for efficient matching

**VPN Technologies:**

IPSec VPN:
  Network-layer encryption
  Site-to-site or remote access
  IKE (Internet Key Exchange) for key management
  ESP (Encapsulating Security Payload) for encryption
  
  Modes:
    Transport: Encrypts payload only
    Tunnel: Encrypts entire original packet

WireGuard:
  Modern, simple VPN protocol
  Uses state-of-the-art cryptography
  Minimal attack surface (~4000 lines of code)
  Built into Linux kernel
  Fast connection establishment
  
  Key features:
    Noise protocol framework
    Curve25519 key exchange
    ChaCha20 encryption
    Poly1305 MAC
    BLAKE2s hash

OpenVPN:
  SSL/TLS-based VPN
  Runs over UDP or TCP
  Supports various authentication methods
  Cross-platform compatibility
  More complex but highly configurable

**Network Segmentation:**

VLANs (Virtual LANs):
  Logical network separation on same physical infrastructure
  IEEE 802.1Q tagging
  Isolate broadcast domains
  
  Example:
    VLAN 10: Employee workstations
    VLAN 20: Servers
    VLAN 30: IoT devices
    VLAN 40: Guest WiFi

Microsegmentation:
  Fine-grained security at workload level
  Zero-trust network approach
  Service mesh provides microsegmentation in Kubernetes
  
  Software-defined: NSX, Calico, Cilium
  Each workload has its own security policy
  East-west traffic inspection

DMZ (Demilitarized Zone):
  Buffer zone between external and internal networks
  
  Internet -> [Firewall] -> DMZ -> [Firewall] -> Internal
                             |
                          Web servers
                          Load balancers
                          Reverse proxies

**IDS/IPS (Intrusion Detection/Prevention Systems):**

IDS (Detection):
  Monitors traffic for suspicious activity
  Generates alerts, does not block
  Passive monitoring (mirror port)
  
IPS (Prevention):
  Monitors AND blocks suspicious traffic
  Inline deployment (traffic flows through)
  Can drop malicious packets in real-time

Detection Methods:
  Signature-based: Match known attack patterns
  Anomaly-based: Detect deviations from baseline
  Heuristic: Rule-based analysis of behavior
  ML-based: Machine learning for pattern detection

Network-based (NIDS/NIPS):
  Monitor network traffic
  Deployed at network boundaries
  Tools: Snort, Suricata, Zeek

Host-based (HIDS/HIPS):
  Monitor individual hosts
  File integrity monitoring
  System call monitoring
  Tools: OSSEC, Wazuh, Tripwire

**Common Network Attacks:**

DDoS (Distributed Denial of Service):
  Volumetric: Flood bandwidth (UDP flood, amplification)
  Protocol: Exploit protocol weaknesses (SYN flood)
  Application: Target application layer (HTTP flood)
  
  Mitigation:
    CDN/DDoS protection (Cloudflare, AWS Shield)
    Rate limiting
    SYN cookies
    Anycast routing
    Traffic scrubbing centers

Man-in-the-Middle (MITM):
  ARP spoofing: Poison ARP cache to redirect traffic
  DNS spoofing: Return false DNS responses
  SSL stripping: Downgrade HTTPS to HTTP
  
  Mitigation:
    TLS everywhere
    HSTS headers
    Certificate pinning
    DNSSEC

DNS Attacks:
  DNS amplification: Small query, large response
  DNS tunneling: Encode data in DNS queries
  DNS cache poisoning: Insert false records
  
  Mitigation:
    DNSSEC (DNS Security Extensions)
    DNS over HTTPS (DoH) / DNS over TLS (DoT)
    Response rate limiting
    Query monitoring`,
					CodeExamples: `// Network defense tools in Go

package main

import (
    "fmt"
    "net"
    "sync"
    "time"
)

// Simple port scanner for network auditing
type PortScanner struct {
    target     string
    startPort  int
    endPort    int
    timeout    time.Duration
    concurrent int
}

func NewPortScanner(target string, startPort, endPort int) *PortScanner {
    return &PortScanner{
        target:     target,
        startPort:  startPort,
        endPort:    endPort,
        timeout:    2 * time.Second,
        concurrent: 100,
    }
}

type ScanResult struct {
    Port   int
    Open   bool
    Banner string
}

func (s *PortScanner) Scan() []ScanResult {
    var results []ScanResult
    var mu sync.Mutex
    
    ports := make(chan int, s.concurrent)
    var wg sync.WaitGroup
    
    // Workers
    for i := 0; i < s.concurrent; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for port := range ports {
                result := s.scanPort(port)
                if result.Open {
                    mu.Lock()
                    results = append(results, result)
                    mu.Unlock()
                }
            }
        }()
    }
    
    // Feed ports
    for p := s.startPort; p <= s.endPort; p++ {
        ports <- p
    }
    close(ports)
    
    wg.Wait()
    return results
}

func (s *PortScanner) scanPort(port int) ScanResult {
    addr := fmt.Sprintf("%s:%d", s.target, port)
    conn, err := net.DialTimeout("tcp", addr, s.timeout)
    if err != nil {
        return ScanResult{Port: port, Open: false}
    }
    defer conn.Close()
    
    // Try to grab banner
    banner := ""
    conn.SetReadDeadline(time.Now().Add(2 * time.Second))
    buf := make([]byte, 1024)
    n, err := conn.Read(buf)
    if err == nil && n > 0 {
        banner = string(buf[:n])
    }
    
    return ScanResult{Port: port, Open: true, Banner: banner}
}

// Rate limiter for network traffic
type TokenBucketLimiter struct {
    mu       sync.Mutex
    tokens   float64
    capacity float64
    rate     float64 // tokens per second
    lastTime time.Time
}

func NewTokenBucketLimiter(rate float64, capacity float64) *TokenBucketLimiter {
    return &TokenBucketLimiter{
        tokens:   capacity,
        capacity: capacity,
        rate:     rate,
        lastTime: time.Now(),
    }
}

func (l *TokenBucketLimiter) Allow() bool {
    l.mu.Lock()
    defer l.mu.Unlock()
    
    now := time.Now()
    elapsed := now.Sub(l.lastTime).Seconds()
    l.lastTime = now
    
    l.tokens += elapsed * l.rate
    if l.tokens > l.capacity {
        l.tokens = l.capacity
    }
    
    if l.tokens < 1 {
        return false
    }
    
    l.tokens--
    return true
}

// Connection tracker (simplified stateful firewall concept)
type ConnectionTracker struct {
    mu          sync.RWMutex
    connections map[string]*ConnState
    timeout     time.Duration
}

type ConnState struct {
    SrcIP     string
    DstIP     string
    SrcPort   int
    DstPort   int
    Protocol  string
    State     string // NEW, ESTABLISHED, RELATED, CLOSING
    CreatedAt time.Time
    LastSeen  time.Time
    BytesSent int64
    BytesRecv int64
}

func NewConnectionTracker(timeout time.Duration) *ConnectionTracker {
    ct := &ConnectionTracker{
        connections: make(map[string]*ConnState),
        timeout:     timeout,
    }
    go ct.cleanup()
    return ct
}

func (ct *ConnectionTracker) Track(srcIP string, srcPort int, dstIP string, dstPort int, protocol string) *ConnState {
    key := fmt.Sprintf("%s:%d->%s:%d/%s", srcIP, srcPort, dstIP, dstPort, protocol)
    reverseKey := fmt.Sprintf("%s:%d->%s:%d/%s", dstIP, dstPort, srcIP, srcPort, protocol)
    
    ct.mu.Lock()
    defer ct.mu.Unlock()
    
    // Check for existing connection
    if conn, exists := ct.connections[key]; exists {
        conn.LastSeen = time.Now()
        return conn
    }
    
    // Check for reverse connection (return traffic)
    if conn, exists := ct.connections[reverseKey]; exists {
        conn.State = "ESTABLISHED"
        conn.LastSeen = time.Now()
        return conn
    }
    
    // New connection
    conn := &ConnState{
        SrcIP:     srcIP,
        DstIP:     dstIP,
        SrcPort:   srcPort,
        DstPort:   dstPort,
        Protocol:  protocol,
        State:     "NEW",
        CreatedAt: time.Now(),
        LastSeen:  time.Now(),
    }
    ct.connections[key] = conn
    return conn
}

func (ct *ConnectionTracker) cleanup() {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        ct.mu.Lock()
        now := time.Now()
        for key, conn := range ct.connections {
            if now.Sub(conn.LastSeen) > ct.timeout {
                delete(ct.connections, key)
            }
        }
        ct.mu.Unlock()
    }
}

// Network traffic analyzer
type TrafficAnalyzer struct {
    mu       sync.Mutex
    stats    map[string]*HostStats
    alerts   chan Alert
}

type HostStats struct {
    IP             string
    PacketCount    int64
    ByteCount      int64
    ConnectionCount int
    UniqueDestPorts map[int]bool
    FirstSeen      time.Time
    LastSeen       time.Time
}

type Alert struct {
    Timestamp time.Time
    Severity  string
    Source    string
    Message  string
}

func (a *TrafficAnalyzer) Analyze(srcIP string, dstPort int, bytes int64) {
    a.mu.Lock()
    defer a.mu.Unlock()
    
    stats, ok := a.stats[srcIP]
    if !ok {
        stats = &HostStats{
            IP:             srcIP,
            UniqueDestPorts: make(map[int]bool),
            FirstSeen:      time.Now(),
        }
        a.stats[srcIP] = stats
    }
    
    stats.PacketCount++
    stats.ByteCount += bytes
    stats.UniqueDestPorts[dstPort] = true
    stats.LastSeen = time.Now()
    
    // Port scan detection
    if len(stats.UniqueDestPorts) > 100 {
        elapsed := time.Since(stats.FirstSeen)
        if elapsed < 60*time.Second {
            a.alerts <- Alert{
                Timestamp: time.Now(),
                Severity:  "high",
                Source:    srcIP,
                Message:   fmt.Sprintf("Possible port scan: %d unique ports in %v", len(stats.UniqueDestPorts), elapsed),
            }
        }
    }
    
    // DDoS detection (high packet rate)
    elapsed := time.Since(stats.FirstSeen).Seconds()
    if elapsed > 0 {
        pps := float64(stats.PacketCount) / elapsed
        if pps > 10000 {
            a.alerts <- Alert{
                Timestamp: time.Now(),
                Severity:  "critical",
                Source:    srcIP,
                Message:   fmt.Sprintf("Possible DDoS: %.0f packets/sec", pps),
            }
        }
    }
}`,
				},
			},
		},
	})
}
