package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2626,
			Title:       "Email Protocols and Messaging Systems",
			Description: "Understand email delivery with SMTP, IMAP, POP3, email authentication (SPF, DKIM, DMARC), and messaging protocols for modern applications.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Email Protocols and Authentication",
					Content: `Email remains a critical communication protocol. Understanding its architecture is essential for deliverability and security.

**SMTP (Simple Mail Transfer Protocol):**

  Protocol for sending email
  Port 25 (relay), 587 (submission), 465 (SMTPS)
  
  Email Delivery Flow:
    Sender MUA -> Sender MTA -> DNS MX -> Recipient MTA -> Recipient MDA -> Recipient MUA
    
    MUA: Mail User Agent (email client: Outlook, Thunderbird)
    MTA: Mail Transfer Agent (server: Postfix, Exchange)
    MDA: Mail Delivery Agent (delivers to mailbox)
    
  SMTP Session:
    EHLO client.example.com
    AUTH LOGIN (base64 encoded credentials)
    MAIL FROM:<sender@example.com>
    RCPT TO:<recipient@example.net>
    DATA
    Subject: Test Email
    From: sender@example.com
    To: recipient@example.net
    
    Hello, this is a test.
    .
    QUIT
    
  SMTP Response Codes:
    220: Service ready
    250: OK, completed
    354: Start mail input (DATA accepted)
    421: Service not available
    450: Mailbox unavailable (temporary)
    550: Mailbox unavailable (permanent)
    554: Transaction failed

  STARTTLS:
    Upgrade plain-text connection to TLS
    Opportunistic encryption (falls back to plain)
    Port 587 with STARTTLS is the modern standard for submission

**IMAP (Internet Message Access Protocol):**

  Protocol for reading email
  Port 143 (IMAP), 993 (IMAPS)
  
  Key features:
    Server-side storage (messages stay on server)
    Folder management
    Multiple device access
    Partial message fetch
    Search capabilities
    
  IMAP vs POP3:
    IMAP: Server-side, multi-device, folders
    POP3: Download and delete, single device, simple

**Email Authentication:**

SPF (Sender Policy Framework):
  DNS TXT record listing authorized sending servers
  Receiver checks if sending IP is authorized
  
  example.com TXT "v=spf1 ip4:203.0.113.0/24 include:_spf.google.com -all"
  
  Qualifiers:
    +: Pass (default)
    -: Hard fail (reject)
    ~: Soft fail (accept but mark)
    ?: Neutral
  
  Mechanisms:
    ip4/ip6: Specific IP ranges
    include: Include another domain's SPF
    a: Domain's A record IP
    mx: Domain's MX record IPs
    all: Match everything (usually at end)

DKIM (DomainKeys Identified Mail):
  Cryptographic signature on email headers and body
  Public key published in DNS
  Proves email hasn't been modified in transit
  
  Signing process:
    1. Sender signs message with private key
    2. Signature added as DKIM-Signature header
    3. Receiver gets public key from DNS TXT record
    4. Verifies signature matches content
    
  DNS record:
    selector._domainkey.example.com TXT "v=DKIM1; k=rsa; p=MIGf..."
  
  Header:
    DKIM-Signature: v=1; a=rsa-sha256; d=example.com; s=selector;
      h=from:to:subject:date; b=<signature>; bh=<body hash>

DMARC (Domain-based Message Authentication, Reporting & Conformance):
  Policy on how to handle SPF/DKIM failures
  Builds on SPF and DKIM
  
  DNS record:
    _dmarc.example.com TXT "v=DMARC1; p=reject; rua=mailto:dmarc@example.com; pct=100"
  
  Policies:
    p=none: Monitor only (reporting)
    p=quarantine: Mark as spam
    p=reject: Reject message
  
  Alignment:
    SPF alignment: MAIL FROM domain matches From header
    DKIM alignment: DKIM d= domain matches From header
    At least one must pass and align

**Messaging Protocols:**

AMQP (Advanced Message Queuing Protocol):
  Standard messaging protocol
  Producer -> Exchange -> Queue -> Consumer
  Exchange types: Direct, Topic, Fanout, Headers
  Acknowledgments, persistence, routing
  Implementation: RabbitMQ

MQTT (Message Queuing Telemetry Transport):
  Lightweight publish/subscribe
  Designed for IoT and constrained devices
  QoS levels: 0 (at most once), 1 (at least once), 2 (exactly once)
  Retained messages and last will
  Implementation: Mosquitto, HiveMQ

XMPP (Extensible Messaging and Presence Protocol):
  Real-time messaging
  Presence information
  XML-based
  Federated (like email)
  Used by: Jabber, chat systems

WebSocket Messaging:
  Full-duplex communication over HTTP
  Low latency, real-time
  Binary and text frames
  Used by: Chat apps, live updates, gaming

**Email Deliverability Best Practices:**

Technical Setup:
  Valid SPF, DKIM, and DMARC records
  Reverse DNS (PTR) matches sending hostname
  Consistent From address
  Valid HELO/EHLO hostname
  TLS encryption for sending

Content:
  Avoid spam trigger words
  Text-to-image ratio
  Valid HTML
  Unsubscribe link (CAN-SPAM, GDPR)
  List-Unsubscribe header

Infrastructure:
  Dedicated IP for bulk sending
  IP warmup for new senders
  Bounce handling (remove invalid addresses)
  Complaint feedback loops
  Rate limiting to ISPs`,
					CodeExamples: `// Email and messaging protocol implementations

package main

import (
    "crypto/tls"
    "encoding/base64"
    "fmt"
    "net"
    "net/smtp"
    "strings"
    "time"
)

// SMTP client with authentication
type EmailSender struct {
    host     string
    port     int
    username string
    password string
    from     string
    useTLS   bool
}

type EmailMessage struct {
    To      []string
    CC      []string
    Subject string
    Body    string
    HTML    string
    Headers map[string]string
}

func NewEmailSender(host string, port int, username, password, from string) *EmailSender {
    return &EmailSender{
        host:     host,
        port:     port,
        username: username,
        password: password,
        from:     from,
        useTLS:   port == 465 || port == 587,
    }
}

func (s *EmailSender) Send(msg *EmailMessage) error {
    addr := fmt.Sprintf("%s:%d", s.host, s.port)
    
    auth := smtp.PlainAuth("", s.username, s.password, s.host)
    
    // Build message
    var builder strings.Builder
    builder.WriteString(fmt.Sprintf("From: %s\r\n", s.from))
    builder.WriteString(fmt.Sprintf("To: %s\r\n", strings.Join(msg.To, ", ")))
    if len(msg.CC) > 0 {
        builder.WriteString(fmt.Sprintf("Cc: %s\r\n", strings.Join(msg.CC, ", ")))
    }
    builder.WriteString(fmt.Sprintf("Subject: %s\r\n", msg.Subject))
    builder.WriteString("MIME-Version: 1.0\r\n")
    
    for key, value := range msg.Headers {
        builder.WriteString(fmt.Sprintf("%s: %s\r\n", key, value))
    }
    
    if msg.HTML != "" {
        builder.WriteString("Content-Type: text/html; charset=UTF-8\r\n")
        builder.WriteString("\r\n")
        builder.WriteString(msg.HTML)
    } else {
        builder.WriteString("Content-Type: text/plain; charset=UTF-8\r\n")
        builder.WriteString("\r\n")
        builder.WriteString(msg.Body)
    }
    
    recipients := append(msg.To, msg.CC...)
    
    return smtp.SendMail(addr, auth, s.from, recipients, []byte(builder.String()))
}

// SPF record validator
type SPFValidator struct{}

type SPFResult struct {
    Domain  string
    Record  string
    Pass    bool
    Details string
}

func (v *SPFValidator) Validate(domain string, senderIP string) (*SPFResult, error) {
    records, err := net.LookupTXT(domain)
    if err != nil {
        return nil, fmt.Errorf("TXT lookup failed: %w", err)
    }
    
    var spfRecord string
    for _, record := range records {
        if strings.HasPrefix(record, "v=spf1") {
            spfRecord = record
            break
        }
    }
    
    if spfRecord == "" {
        return &SPFResult{
            Domain:  domain,
            Pass:    false,
            Details: "No SPF record found",
        }, nil
    }
    
    result := &SPFResult{
        Domain: domain,
        Record: spfRecord,
    }
    
    ip := net.ParseIP(senderIP)
    mechanisms := strings.Fields(spfRecord)
    
    for _, mech := range mechanisms[1:] { // Skip v=spf1
        qualifier := "+"
        if mech[0] == '+' || mech[0] == '-' || mech[0] == '~' || mech[0] == '?' {
            qualifier = string(mech[0])
            mech = mech[1:]
        }
        
        if strings.HasPrefix(mech, "ip4:") {
            cidr := strings.TrimPrefix(mech, "ip4:")
            if !strings.Contains(cidr, "/") {
                cidr += "/32"
            }
            _, network, err := net.ParseCIDR(cidr)
            if err != nil {
                continue
            }
            if network.Contains(ip) {
                result.Pass = qualifier == "+"
                result.Details = fmt.Sprintf("Matched ip4:%s (qualifier: %s)", cidr, qualifier)
                return result, nil
            }
        }
        
        if mech == "all" {
            result.Pass = qualifier == "+"
            result.Details = fmt.Sprintf("Matched 'all' (qualifier: %s)", qualifier)
            return result, nil
        }
    }
    
    result.Pass = false
    result.Details = "No mechanism matched"
    return result, nil
}

// DMARC record parser
type DMARCRecord struct {
    Version  string
    Policy   string // none, quarantine, reject
    SubPolicy string
    Percent  int
    ReportURI []string
    Alignment string
}

func ParseDMARCRecord(domain string) (*DMARCRecord, error) {
    records, err := net.LookupTXT("_dmarc." + domain)
    if err != nil {
        return nil, fmt.Errorf("DMARC lookup failed: %w", err)
    }
    
    for _, record := range records {
        if !strings.HasPrefix(record, "v=DMARC1") {
            continue
        }
        
        dmarc := &DMARCRecord{Percent: 100}
        parts := strings.Split(record, ";")
        
        for _, part := range parts {
            part = strings.TrimSpace(part)
            kv := strings.SplitN(part, "=", 2)
            if len(kv) != 2 {
                continue
            }
            
            switch strings.TrimSpace(kv[0]) {
            case "v":
                dmarc.Version = strings.TrimSpace(kv[1])
            case "p":
                dmarc.Policy = strings.TrimSpace(kv[1])
            case "sp":
                dmarc.SubPolicy = strings.TrimSpace(kv[1])
            case "pct":
                fmt.Sscanf(strings.TrimSpace(kv[1]), "%d", &dmarc.Percent)
            case "rua":
                dmarc.ReportURI = strings.Split(strings.TrimSpace(kv[1]), ",")
            }
        }
        
        return dmarc, nil
    }
    
    return nil, fmt.Errorf("no DMARC record found for %s", domain)
}

// MX record checker
type MXChecker struct {
    timeout time.Duration
}

type MXResult struct {
    Domain   string
    Records  []MXRecord
    Warnings []string
}

type MXRecord struct {
    Host     string
    Priority uint16
    IPs      []string
    TLSSupport bool
    Reachable  bool
}

func NewMXChecker() *MXChecker {
    return &MXChecker{timeout: 10 * time.Second}
}

func (c *MXChecker) Check(domain string) (*MXResult, error) {
    mxRecords, err := net.LookupMX(domain)
    if err != nil {
        return nil, fmt.Errorf("MX lookup failed: %w", err)
    }
    
    result := &MXResult{Domain: domain}
    
    for _, mx := range mxRecords {
        record := MXRecord{
            Host:     mx.Host,
            Priority: mx.Pref,
        }
        
        // Resolve MX host
        ips, err := net.LookupHost(mx.Host)
        if err == nil {
            record.IPs = ips
        }
        
        // Check connectivity (port 25)
        conn, err := net.DialTimeout("tcp", mx.Host+":25", c.timeout)
        if err == nil {
            record.Reachable = true
            conn.Close()
            
            // Check STARTTLS support
            record.TLSSupport = c.checkSTARTTLS(mx.Host)
        }
        
        result.Records = append(result.Records, record)
    }
    
    // Warnings
    if len(result.Records) < 2 {
        result.Warnings = append(result.Warnings, "Only one MX record - no redundancy")
    }
    
    for _, rec := range result.Records {
        if !rec.TLSSupport {
            result.Warnings = append(result.Warnings,
                fmt.Sprintf("MX %s does not support STARTTLS", rec.Host))
        }
    }
    
    return result, nil
}

func (c *MXChecker) checkSTARTTLS(host string) bool {
    conn, err := net.DialTimeout("tcp", host+":25", c.timeout)
    if err != nil {
        return false
    }
    defer conn.Close()
    
    conn.SetDeadline(time.Now().Add(c.timeout))
    
    // Read banner
    buf := make([]byte, 1024)
    conn.Read(buf)
    
    // Send EHLO
    fmt.Fprintf(conn, "EHLO checker.local\r\n")
    n, err := conn.Read(buf)
    if err != nil {
        return false
    }
    
    response := string(buf[:n])
    return strings.Contains(response, "STARTTLS")
}

// Simple pub/sub message broker
type MessageBroker struct {
    topics      map[string][]chan Message
    mu          sync.RWMutex
}

type Message struct {
    Topic     string
    Payload   []byte
    Timestamp time.Time
    Headers   map[string]string
}

func NewMessageBroker() *MessageBroker {
    return &MessageBroker{
        topics: make(map[string][]chan Message),
    }
}

func (b *MessageBroker) Subscribe(topic string, bufSize int) <-chan Message {
    ch := make(chan Message, bufSize)
    
    b.mu.Lock()
    b.topics[topic] = append(b.topics[topic], ch)
    b.mu.Unlock()
    
    return ch
}

func (b *MessageBroker) Publish(topic string, payload []byte) {
    msg := Message{
        Topic:     topic,
        Payload:   payload,
        Timestamp: time.Now(),
    }
    
    b.mu.RLock()
    subscribers := b.topics[topic]
    b.mu.RUnlock()
    
    for _, ch := range subscribers {
        select {
        case ch <- msg:
        default:
            // Drop if subscriber is slow
        }
    }
}

func (b *MessageBroker) Unsubscribe(topic string, ch <-chan Message) {
    b.mu.Lock()
    defer b.mu.Unlock()
    
    subs := b.topics[topic]
    for i, sub := range subs {
        if sub == ch {
            b.topics[topic] = append(subs[:i], subs[i+1:]...)
            close(sub)
            break
        }
    }
}`,
				},
			},
		},
		{
			ID:          2627,
			Title:       "Wireless Networking and Mobile Networks",
			Description: "Understand WiFi standards, cellular networks, Bluetooth, network mobility, and wireless security protocols.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Wireless and Mobile Network Technologies",
					Content: `Wireless networking encompasses WiFi, cellular, Bluetooth, and emerging technologies that enable mobile connectivity.

**WiFi Standards (IEEE 802.11):**

802.11a (1999):
  Frequency: 5 GHz
  Max speed: 54 Mbps
  Range: Short (less interference)

802.11b (1999):
  Frequency: 2.4 GHz
  Max speed: 11 Mbps
  Range: Medium

802.11g (2003):
  Frequency: 2.4 GHz
  Max speed: 54 Mbps
  Backward compatible with 802.11b

802.11n / WiFi 4 (2009):
  Frequency: 2.4 GHz and 5 GHz
  Max speed: 600 Mbps
  MIMO (Multiple Input Multiple Output)
  Channel bonding (20/40 MHz)

802.11ac / WiFi 5 (2013):
  Frequency: 5 GHz only
  Max speed: 6.93 Gbps (theoretical)
  MU-MIMO (Multi-User MIMO)
  Beamforming
  80/160 MHz channels

802.11ax / WiFi 6 (2019):
  Frequency: 2.4 GHz and 5 GHz
  Max speed: 9.6 Gbps (theoretical)
  OFDMA (multi-user per channel)
  Target Wake Time (battery saving for IoT)
  BSS Coloring (reduced interference)
  1024-QAM modulation

802.11be / WiFi 7 (2024):
  Frequency: 2.4 GHz, 5 GHz, and 6 GHz
  Max speed: 46 Gbps (theoretical)
  320 MHz channels
  Multi-Link Operation (MLO)
  4096-QAM modulation

**WiFi Security:**

WEP (Wired Equivalent Privacy):
  Deprecated, easily cracked
  RC4 encryption with static keys
  Never use

WPA (WiFi Protected Access):
  Temporary fix for WEP
  TKIP encryption
  Deprecated

WPA2 (2004):
  AES-CCMP encryption
  Personal mode: Pre-shared key (PSK)
  Enterprise mode: 802.1X / RADIUS authentication
  Vulnerable to KRACK attack (patched)

WPA3 (2018):
  SAE (Simultaneous Authentication of Equals)
  Replaces PSK handshake
  Forward secrecy
  Protected Management Frames mandatory
  Enhanced Open (OWE) for open networks
  192-bit security suite for enterprise

802.1X Authentication:
  Port-based access control
  Supplicant (client) -> Authenticator (AP) -> Auth Server (RADIUS)
  EAP methods:
    EAP-TLS: Certificate-based (most secure)
    PEAP: Password-based with TLS tunnel
    EAP-TTLS: Similar to PEAP

**Cellular Networks:**

2G (GSM):
  Circuit-switched voice
  GPRS/EDGE for data (up to 384 Kbps)
  Being decommissioned globally

3G (UMTS/CDMA):
  HSPA+: Up to 42 Mbps
  Introduced mobile internet
  Being phased out

4G (LTE):
  All-IP network (no circuit switching)
  Download: 100-300 Mbps typical
  Latency: 30-50 ms
  OFDMA (downlink), SC-FDMA (uplink)
  VoLTE for voice over data

5G:
  Sub-6 GHz: Wider coverage, 100-900 Mbps
  mmWave (24-100 GHz): Short range, 1-10 Gbps
  Latency: 1-10 ms
  Massive MIMO (64+ antennas)
  Network slicing (virtual dedicated networks)
  
  Use cases:
    Enhanced Mobile Broadband (eMBB)
    Ultra-Reliable Low-Latency (URLLC)
    Massive Machine-Type Communications (mMTC)

**Bluetooth:**

Classic Bluetooth:
  Short-range (10-100m)
  1-3 Mbps
  Audio, file transfer
  Profiles: A2DP (audio), HFP (hands-free), SPP (serial)

Bluetooth Low Energy (BLE):
  Extremely low power
  1-2 Mbps
  IoT sensors, beacons, wearables
  GATT protocol (services and characteristics)
  Advertising (broadcast mode)

Bluetooth Mesh:
  Many-to-many communication
  Extends range through relay nodes
  Smart lighting, building automation
  Up to 32,767 devices

**Network Mobility:**

Mobile IP:
  Home Agent: Router at home network
  Foreign Agent: Router at visited network
  Care-of Address: Temporary address at foreign network
  Triangular routing or tunnel

Handoff/Handover:
  Hard handoff: Break before make (interruption)
  Soft handoff: Make before break (seamless)
  Vertical handoff: Between network types (WiFi -> cellular)

WiFi Roaming:
  802.11r (Fast BSS Transition): Pre-authenticate with next AP
  802.11k (Radio Resource Measurement): AP provides neighbor list
  802.11v (BSS Transition Management): AP suggests move
  
  Enterprise roaming:
    Same SSID across all APs
    Controller-based: Centralized management
    Mesh: APs communicate for seamless handoff

**IoT Networking Protocols:**

LoRaWAN:
  Long range (10+ km rural)
  Very low power, low data rate
  Unlicensed spectrum (ISM band)
  Star topology with gateways

Zigbee (IEEE 802.15.4):
  Short range (10-100m)
  Low power mesh network
  Smart home devices
  Up to 250 Kbps

Thread:
  IPv6-based mesh networking
  Low power, secure
  No hub required
  Matter smart home standard

NB-IoT (Narrowband IoT):
  Uses cellular infrastructure
  Deep indoor penetration
  Very low power
  Small data volumes`,
					CodeExamples: `// Wireless network utilities

package main

import (
    "fmt"
    "math"
    "strings"
)

// WiFi channel planner
type WiFiChannelPlanner struct{}

type ChannelPlan struct {
    Band        string
    Channels    []ChannelInfo
    Recommended []int
    Conflicts   []ChannelConflict
}

type ChannelInfo struct {
    Number     int
    Frequency  float64 // MHz
    Width      int     // MHz
    Band       string
    DFS        bool    // Dynamic Frequency Selection required
}

type ChannelConflict struct {
    Channel1 int
    Channel2 int
    Overlap  float64 // MHz
}

func (p *WiFiChannelPlanner) Get24GHzChannels() []ChannelInfo {
    channels := make([]ChannelInfo, 0)
    for i := 1; i <= 13; i++ {
        freq := 2412.0 + float64(i-1)*5
        channels = append(channels, ChannelInfo{
            Number:    i,
            Frequency: freq,
            Width:     20,
            Band:      "2.4 GHz",
            DFS:       false,
        })
    }
    return channels
}

func (p *WiFiChannelPlanner) Get5GHzChannels() []ChannelInfo {
    uniiChannels := []struct {
        number int
        freq   float64
        dfs    bool
    }{
        {36, 5180, false}, {40, 5200, false}, {44, 5220, false}, {48, 5240, false},
        {52, 5260, true}, {56, 5280, true}, {60, 5300, true}, {64, 5320, true},
        {100, 5500, true}, {104, 5520, true}, {108, 5540, true}, {112, 5560, true},
        {116, 5580, true}, {120, 5600, true}, {124, 5620, true}, {128, 5640, true},
        {132, 5660, true}, {136, 5680, true}, {140, 5700, true}, {144, 5720, true},
        {149, 5745, false}, {153, 5765, false}, {157, 5785, false}, {161, 5805, false},
        {165, 5825, false},
    }
    
    channels := make([]ChannelInfo, len(uniiChannels))
    for i, ch := range uniiChannels {
        channels[i] = ChannelInfo{
            Number:    ch.number,
            Frequency: ch.freq,
            Width:     20,
            Band:      "5 GHz",
            DFS:       ch.dfs,
        }
    }
    return channels
}

func (p *WiFiChannelPlanner) RecommendNonOverlapping24() []int {
    return []int{1, 6, 11}
}

// WiFi signal strength calculator
type SignalCalculator struct{}

// Free Space Path Loss (FSPL) in dB
func (c *SignalCalculator) FreeSpacePathLoss(frequencyMHz float64, distanceM float64) float64 {
    return 20*math.Log10(distanceM) + 20*math.Log10(frequencyMHz) - 27.55
}

// Estimate distance from RSSI
func (c *SignalCalculator) EstimateDistance(rssiDBm float64, txPowerDBm float64, frequencyMHz float64) float64 {
    pathLoss := txPowerDBm - rssiDBm
    exponent := (pathLoss - 20*math.Log10(frequencyMHz) + 27.55) / 20.0
    return math.Pow(10, exponent)
}

// Link budget calculation
type LinkBudget struct {
    TxPower        float64 // dBm
    TxAntennaGain  float64 // dBi
    TxCableLoss    float64 // dB
    PathLoss       float64 // dB
    RxAntennaGain  float64 // dBi
    RxCableLoss    float64 // dB
    FadeMargin     float64 // dB
}

func (lb *LinkBudget) ReceivedPower() float64 {
    return lb.TxPower + lb.TxAntennaGain - lb.TxCableLoss -
        lb.PathLoss + lb.RxAntennaGain - lb.RxCableLoss - lb.FadeMargin
}

func (lb *LinkBudget) IsLinkViable(rxSensitivity float64) bool {
    return lb.ReceivedPower() >= rxSensitivity
}

// Network type selector for mobile devices
type NetworkSelector struct {
    availableNetworks []AvailableNetwork
}

type AvailableNetwork struct {
    Type      string  // WiFi, 4G, 5G
    SSID      string  // For WiFi
    Signal    float64 // dBm
    Bandwidth float64 // Mbps estimated
    Latency   float64 // ms estimated
    Cost      float64 // 0 = free, higher = more expensive
    Secure    bool
}

type NetworkPreference struct {
    PreferWiFi    bool
    MinSignal     float64
    MaxLatency    float64
    AvoidMetered  bool
}

func (s *NetworkSelector) SelectBest(prefs NetworkPreference) *AvailableNetwork {
    var best *AvailableNetwork
    bestScore := -math.MaxFloat64
    
    for i := range s.availableNetworks {
        net := &s.availableNetworks[i]
        
        // Filter
        if net.Signal < prefs.MinSignal {
            continue
        }
        if net.Latency > prefs.MaxLatency && prefs.MaxLatency > 0 {
            continue
        }
        if prefs.AvoidMetered && net.Cost > 0 {
            continue
        }
        
        // Score
        score := 0.0
        score += net.Bandwidth * 0.3   // Bandwidth weight
        score += (net.Signal + 100) * 0.2 // Signal normalized
        score -= net.Latency * 0.2      // Lower latency better
        score -= net.Cost * 10          // Cost penalty
        
        if prefs.PreferWiFi && net.Type == "WiFi" {
            score += 20
        }
        if net.Secure {
            score += 5
        }
        
        if score > bestScore {
            bestScore = score
            best = net
        }
    }
    
    return best
}

// Bluetooth device scanner simulator
type BLEScanner struct {
    devices []BLEDevice
}

type BLEDevice struct {
    Address    string
    Name       string
    RSSI       int
    Services   []string
    Connectable bool
    TxPower    int
    Distance   float64
}

func (s *BLEScanner) EstimateDistances() {
    for i := range s.devices {
        dev := &s.devices[i]
        if dev.TxPower != 0 {
            // Simple distance estimation from RSSI
            ratio := float64(dev.TxPower-dev.RSSI) / 20.0
            dev.Distance = math.Pow(10, ratio)
        }
    }
}

func (s *BLEScanner) FilterByService(serviceUUID string) []BLEDevice {
    var filtered []BLEDevice
    for _, dev := range s.devices {
        for _, svc := range dev.Services {
            if strings.EqualFold(svc, serviceUUID) {
                filtered = append(filtered, dev)
                break
            }
        }
    }
    return filtered
}

func (s *BLEScanner) NearestDevice() *BLEDevice {
    if len(s.devices) == 0 {
        return nil
    }
    
    best := &s.devices[0]
    for i := 1; i < len(s.devices); i++ {
        if s.devices[i].RSSI > best.RSSI {
            best = &s.devices[i]
        }
    }
    return best
}`,
				},
			},
		},
	})
}
