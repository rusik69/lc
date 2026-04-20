package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2628,
			Title:       "VoIP and Real-Time Communication Protocols",
			Description: "Learn about Voice over IP, SIP signaling, RTP media transport, WebRTC, and quality of service for real-time communications.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "VoIP SIP RTP and Real-Time Communication",
					Content: `Real-time communication protocols enable voice, video, and interactive media over IP networks.

**VoIP Architecture:**

  Traditional telephony (PSTN):
    Circuit-switched, dedicated path
    64 Kbps per call (G.711)
    Reliable, consistent quality
    
  VoIP:
    Packet-switched, shared network
    Variable bandwidth (codecs)
    Cost-effective, flexible
    Quality depends on network
    
  VoIP Components:
    IP Phone / Softphone (endpoint)
    Call Server / PBX (Asterisk, FreeSWITCH)
    Media Gateway (PSTN interconnect)
    SBC (Session Border Controller)
    Registrar / Location Server

**SIP (Session Initiation Protocol):**

  Signaling protocol for initiating, managing, terminating sessions
  Text-based (like HTTP)
  Port 5060 (UDP/TCP), 5061 (TLS)
  
  SIP Methods:
    INVITE: Start session
    ACK: Confirm INVITE response
    BYE: End session
    CANCEL: Cancel pending INVITE
    REGISTER: Register contact with registrar
    OPTIONS: Query capabilities
    REFER: Transfer call
    SUBSCRIBE/NOTIFY: Event notification
    INFO: Mid-session signaling
    UPDATE: Modify session
    MESSAGE: Instant messaging
    PRACK: Provisional acknowledgment
    
  SIP Response Codes:
    1xx: Provisional (100 Trying, 180 Ringing, 183 Progress)
    2xx: Success (200 OK)
    3xx: Redirection (301 Moved, 302 Temporary)
    4xx: Client Error (401 Unauthorized, 403 Forbidden, 404 Not Found, 408 Timeout, 486 Busy)
    5xx: Server Error (500 Internal, 503 Unavailable)
    6xx: Global Failure (600 Busy Everywhere, 603 Decline)
    
  SIP Call Flow (Basic):
    Alice                  Proxy                  Bob
      |--INVITE------------>|                      |
      |<--100 Trying--------|                      |
      |                     |--INVITE------------>|
      |                     |<--180 Ringing-------|
      |<--180 Ringing-------|                      |
      |                     |<--200 OK------------|
      |<--200 OK------------|                      |
      |--ACK-------------------------------------->|
      |<=========RTP Media========================>|
      |--BYE-------------------------------------->|
      |<--200 OK----------------------------------|
      
  SIP Headers:
    Via: Path of request
    From: Caller identity
    To: Callee identity
    Call-ID: Unique dialog identifier
    CSeq: Sequence number and method
    Contact: Direct reachability
    Max-Forwards: Hop limit
    Content-Type: Body type (application/sdp)

**SDP (Session Description Protocol):**

  Describes multimedia sessions
  Used in SIP INVITE body
  
  SDP Fields:
    v=0 (version)
    o=alice 12345 12345 IN IP4 192.168.1.100 (origin)
    s=Call (session name)
    c=IN IP4 192.168.1.100 (connection info)
    t=0 0 (timing - 0 0 means permanent)
    m=audio 49170 RTP/AVP 0 8 101 (media line)
    a=rtpmap:0 PCMU/8000 (attribute - codec mapping)
    a=rtpmap:8 PCMA/8000
    a=rtpmap:101 telephone-event/8000
    a=sendrecv (direction)
    
  Codec Negotiation:
    Offer contains supported codecs in priority order
    Answer selects subset of offered codecs
    Both sides must agree on at least one codec

**RTP (Real-time Transport Protocol):**

  Carries media (audio/video)
  UDP-based (port range 16384-32767 typical)
  No guaranteed delivery (accepts packet loss)
  
  RTP Header:
    Version (2)
    Padding, Extension, CSRC Count
    Marker bit (frame boundary)
    Payload Type (codec identifier)
    Sequence Number (detect loss, reorder)
    Timestamp (synchronization)
    SSRC (synchronization source)
    
  RTCP (RTP Control Protocol):
    Companion to RTP
    Quality feedback and statistics
    
    Report Types:
      SR (Sender Report): Sender statistics
      RR (Receiver Report): Receiver statistics
      SDES (Source Description): Participant info
      BYE: Leave session
      APP: Application-specific
    
    RTCP Statistics:
      Packets sent/received
      Bytes sent/received
      Packet loss rate
      Jitter (variation in packet arrival)
      Round-trip time

**Audio Codecs:**

G.711 (PCM):
  Bitrate: 64 Kbps
  Quality: Toll quality (PSTN equivalent)
  Delay: Very low
  uLaw (North America), aLaw (Europe)

G.729:
  Bitrate: 8 Kbps
  Quality: Good
  Patented (was, now expired)
  CELP-based compression

G.722 (HD Voice):
  Bitrate: 64 Kbps
  Wideband: 50-7000 Hz
  Better quality than G.711

Opus:
  Bitrate: 6-510 Kbps (variable)
  Wideband to fullband
  Open source, royalty free
  Used by WebRTC, Discord, Zoom
  Low latency, adaptive

**WebRTC (Web Real-Time Communication):**

  Browser-to-browser audio/video/data
  No plugins required
  End-to-end encrypted
  
  Architecture:
    Peer A <-- Signaling Server --> Peer B
    Peer A <-- STUN/TURN --> Peer B (media)
    
  Components:
    MediaStream API: Access camera/microphone
    RTCPeerConnection: P2P connection
    RTCDataChannel: Arbitrary data
    
  ICE (Interactive Connectivity Establishment):
    Finds best path between peers
    Gathers candidates:
      Host: Local IP addresses
      Server Reflexive: Public IP (via STUN)
      Relay: TURN server (fallback)
    
    Candidate Pairing:
      Pairs all local with remote candidates
      Connectivity checks (STUN binding)
      Selects best working pair
    
  STUN (Session Traversal Utilities for NAT):
    Discovers public IP and port
    Simple request/response
    Not always sufficient (symmetric NAT)
    
  TURN (Traversal Using Relays around NAT):
    Relay server as fallback
    All media passes through TURN
    Higher latency but always works
    Bandwidth-intensive for provider

  SRTP (Secure RTP):
    Encrypted RTP
    DTLS-SRTP key exchange (WebRTC mandatory)
    AES-128 encryption
    HMAC-SHA1 authentication

**Quality of Service for VoIP:**

Jitter:
  Variation in packet arrival times
  Causes choppy audio
  Jitter buffer absorbs variation (20-60ms typical)
  Adaptive vs fixed jitter buffer

Latency:
  One-way delay targets:
    < 150ms: Excellent
    150-300ms: Acceptable  
    > 300ms: Poor (noticeable delay)
    > 500ms: Unusable for conversation
    
  Contributing factors:
    Codec delay (5-30ms)
    Packetization delay (20ms typical)
    Network transmission delay
    Jitter buffer delay
    Processing delay

Packet Loss:
  VoIP tolerates 1-3% loss
  > 5% loss: Noticeable quality degradation
  PLC (Packet Loss Concealment):
    Interpolation from adjacent packets
    Comfort noise generation
    Silence substitution

MOS (Mean Opinion Score):
  Quality rating 1-5
  5: Excellent
  4: Good
  3: Fair
  2: Poor
  1: Bad
  G.711 achieves ~4.3 MOS
  Opus at 32 Kbps achieves ~4.5 MOS

QoS Mechanisms:
  DSCP marking: EF (Expedited Forwarding) for voice
  802.1p: Layer 2 priority
  Traffic shaping: Guarantee bandwidth
  Priority queuing: Voice before data
  
  DSCP Values:
    EF (46): Voice media
    AF41 (34): Video media
    CS3 (24): Voice signaling
    AF21 (18): Video signaling
    BE (0): Best effort (data)`,
					CodeExamples: `// VoIP and real-time communication implementations

package main

import (
    "encoding/binary"
    "fmt"
    "math"
    "net"
    "strings"
    "sync"
    "time"
)

// SIP message parser
type SIPMessage struct {
    Method     string
    RequestURI string
    StatusCode int
    StatusText string
    Headers    map[string][]string
    Body       string
    IsRequest  bool
}

func ParseSIPMessage(data string) *SIPMessage {
    msg := &SIPMessage{
        Headers: make(map[string][]string),
    }
    
    lines := strings.Split(data, "\r\n")
    if len(lines) == 0 {
        return nil
    }
    
    // Parse start line
    startLine := lines[0]
    if strings.HasPrefix(startLine, "SIP/2.0") {
        // Response
        msg.IsRequest = false
        parts := strings.SplitN(startLine, " ", 3)
        if len(parts) >= 3 {
            fmt.Sscanf(parts[1], "%d", &msg.StatusCode)
            msg.StatusText = parts[2]
        }
    } else {
        // Request
        msg.IsRequest = true
        parts := strings.SplitN(startLine, " ", 3)
        if len(parts) >= 2 {
            msg.Method = parts[0]
            msg.RequestURI = parts[1]
        }
    }
    
    // Parse headers
    bodyStart := -1
    for i := 1; i < len(lines); i++ {
        line := lines[i]
        if line == "" {
            bodyStart = i + 1
            break
        }
        
        colonIdx := strings.Index(line, ":")
        if colonIdx > 0 {
            name := strings.TrimSpace(line[:colonIdx])
            value := strings.TrimSpace(line[colonIdx+1:])
            msg.Headers[name] = append(msg.Headers[name], value)
        }
    }
    
    // Parse body
    if bodyStart > 0 && bodyStart < len(lines) {
        msg.Body = strings.Join(lines[bodyStart:], "\r\n")
    }
    
    return msg
}

func (m *SIPMessage) GetHeader(name string) string {
    if vals, ok := m.Headers[name]; ok && len(vals) > 0 {
        return vals[0]
    }
    return ""
}

func (m *SIPMessage) GetCallID() string {
    return m.GetHeader("Call-ID")
}

func (m *SIPMessage) GetFrom() string {
    return m.GetHeader("From")
}

func (m *SIPMessage) GetTo() string {
    return m.GetHeader("To")
}

// SDP parser
type SDPSession struct {
    Version    int
    Origin     SDPOrigin
    Name       string
    Connection SDPConnection
    Media      []SDPMedia
}

type SDPOrigin struct {
    Username  string
    SessionID string
    Version   string
    NetType   string
    AddrType  string
    Address   string
}

type SDPConnection struct {
    NetType  string
    AddrType string
    Address  string
}

type SDPMedia struct {
    Type     string // audio, video
    Port     int
    Protocol string // RTP/AVP
    Formats  []int  // Payload types
    Codecs   map[int]string
    Direction string // sendrecv, sendonly, recvonly, inactive
}

func ParseSDP(data string) *SDPSession {
    session := &SDPSession{}
    var currentMedia *SDPMedia
    
    for _, line := range strings.Split(data, "\r\n") {
        if len(line) < 2 || line[1] != '=' {
            continue
        }
        
        field := line[0]
        value := line[2:]
        
        switch field {
        case 'v':
            fmt.Sscanf(value, "%d", &session.Version)
        case 'o':
            parts := strings.Fields(value)
            if len(parts) >= 6 {
                session.Origin = SDPOrigin{
                    Username:  parts[0],
                    SessionID: parts[1],
                    Version:   parts[2],
                    NetType:   parts[3],
                    AddrType:  parts[4],
                    Address:   parts[5],
                }
            }
        case 's':
            session.Name = value
        case 'c':
            parts := strings.Fields(value)
            if len(parts) >= 3 {
                session.Connection = SDPConnection{
                    NetType:  parts[0],
                    AddrType: parts[1],
                    Address:  parts[2],
                }
            }
        case 'm':
            parts := strings.Fields(value)
            if len(parts) >= 3 {
                media := SDPMedia{
                    Type:     parts[0],
                    Protocol: parts[2],
                    Codecs:   make(map[int]string),
                    Direction: "sendrecv",
                }
                fmt.Sscanf(parts[1], "%d", &media.Port)
                for _, pt := range parts[3:] {
                    var payloadType int
                    fmt.Sscanf(pt, "%d", &payloadType)
                    media.Formats = append(media.Formats, payloadType)
                }
                session.Media = append(session.Media, media)
                currentMedia = &session.Media[len(session.Media)-1]
            }
        case 'a':
            if currentMedia != nil {
                if strings.HasPrefix(value, "rtpmap:") {
                    // a=rtpmap:0 PCMU/8000
                    rest := strings.TrimPrefix(value, "rtpmap:")
                    parts := strings.SplitN(rest, " ", 2)
                    if len(parts) == 2 {
                        var pt int
                        fmt.Sscanf(parts[0], "%d", &pt)
                        currentMedia.Codecs[pt] = parts[1]
                    }
                } else if value == "sendrecv" || value == "sendonly" ||
                    value == "recvonly" || value == "inactive" {
                    currentMedia.Direction = value
                }
            }
        }
    }
    
    return session
}

// RTP packet handler
type RTPHeader struct {
    Version    uint8
    Padding    bool
    Extension  bool
    CSRCCount  uint8
    Marker     bool
    PayloadType uint8
    SeqNumber  uint16
    Timestamp  uint32
    SSRC       uint32
}

func ParseRTPHeader(data []byte) *RTPHeader {
    if len(data) < 12 {
        return nil
    }
    
    header := &RTPHeader{
        Version:    (data[0] >> 6) & 0x03,
        Padding:    (data[0] & 0x20) != 0,
        Extension:  (data[0] & 0x10) != 0,
        CSRCCount:  data[0] & 0x0F,
        Marker:     (data[1] & 0x80) != 0,
        PayloadType: data[1] & 0x7F,
        SeqNumber:  binary.BigEndian.Uint16(data[2:4]),
        Timestamp:  binary.BigEndian.Uint32(data[4:8]),
        SSRC:       binary.BigEndian.Uint32(data[8:12]),
    }
    
    return header
}

func SerializeRTPHeader(h *RTPHeader) []byte {
    data := make([]byte, 12)
    
    data[0] = (h.Version << 6)
    if h.Padding {
        data[0] |= 0x20
    }
    if h.Extension {
        data[0] |= 0x10
    }
    data[0] |= h.CSRCCount & 0x0F
    
    data[1] = h.PayloadType & 0x7F
    if h.Marker {
        data[1] |= 0x80
    }
    
    binary.BigEndian.PutUint16(data[2:4], h.SeqNumber)
    binary.BigEndian.PutUint32(data[4:8], h.Timestamp)
    binary.BigEndian.PutUint32(data[8:12], h.SSRC)
    
    return data
}

// Jitter buffer
type JitterBuffer struct {
    mu        sync.Mutex
    buffer    map[uint16]*RTPPacket
    minDelay  time.Duration
    maxDelay  time.Duration
    nextSeq   uint16
    ready     chan *RTPPacket
    started   bool
}

type RTPPacket struct {
    Header    RTPHeader
    Payload   []byte
    Received  time.Time
}

func NewJitterBuffer(minDelay, maxDelay time.Duration) *JitterBuffer {
    return &JitterBuffer{
        buffer:   make(map[uint16]*RTPPacket),
        minDelay: minDelay,
        maxDelay: maxDelay,
        ready:    make(chan *RTPPacket, 100),
    }
}

func (jb *JitterBuffer) Insert(pkt *RTPPacket) {
    jb.mu.Lock()
    defer jb.mu.Unlock()
    
    jb.buffer[pkt.Header.SeqNumber] = pkt
    
    if !jb.started {
        jb.nextSeq = pkt.Header.SeqNumber
        jb.started = true
    }
}

func (jb *JitterBuffer) Get() *RTPPacket {
    jb.mu.Lock()
    defer jb.mu.Unlock()
    
    pkt, exists := jb.buffer[jb.nextSeq]
    if exists {
        delete(jb.buffer, jb.nextSeq)
        jb.nextSeq++
        return pkt
    }
    
    // Packet missing - advance sequence
    jb.nextSeq++
    return nil // Signal packet loss
}

// VoIP quality monitor
type VoIPQualityMonitor struct {
    mu           sync.Mutex
    packetsRecv  uint64
    packetsLost  uint64
    jitterSamples []float64
    latencySamples []float64
    lastSeq      uint16
    lastArrival  time.Time
    lastJitter   float64
}

func NewVoIPQualityMonitor() *VoIPQualityMonitor {
    return &VoIPQualityMonitor{}
}

func (m *VoIPQualityMonitor) RecordPacket(seq uint16, arrivalTime time.Time) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    m.packetsRecv++
    
    if m.packetsRecv > 1 {
        // Check for lost packets
        expected := m.lastSeq + 1
        if seq > expected {
            m.packetsLost += uint64(seq - expected)
        }
        
        // Calculate jitter (RFC 3550)
        transitDiff := arrivalTime.Sub(m.lastArrival).Seconds()
        m.lastJitter += (math.Abs(transitDiff) - m.lastJitter) / 16.0
        m.jitterSamples = append(m.jitterSamples, m.lastJitter)
    }
    
    m.lastSeq = seq
    m.lastArrival = arrivalTime
}

func (m *VoIPQualityMonitor) GetPacketLossRate() float64 {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    total := m.packetsRecv + m.packetsLost
    if total == 0 {
        return 0
    }
    return float64(m.packetsLost) / float64(total)
}

func (m *VoIPQualityMonitor) GetJitter() float64 {
    m.mu.Lock()
    defer m.mu.Unlock()
    return m.lastJitter
}

// Calculate R-factor (E-model, ITU-T G.107)
func (m *VoIPQualityMonitor) GetRFactor() float64 {
    lossRate := m.GetPacketLossRate() * 100
    jitter := m.GetJitter() * 1000 // ms
    
    // Simplified R-factor calculation
    r := 93.2 - lossRate*2.5 - jitter*0.1
    
    if r < 0 {
        r = 0
    }
    if r > 100 {
        r = 100
    }
    
    return r
}

// Convert R-factor to MOS
func (m *VoIPQualityMonitor) GetMOS() float64 {
    r := m.GetRFactor()
    
    if r < 0 {
        return 1.0
    }
    if r > 100 {
        return 4.5
    }
    
    // ITU-T G.107 R to MOS conversion
    mos := 1.0 + 0.035*r + r*(r-60)*(100-r)*7e-6
    
    if mos < 1.0 {
        mos = 1.0
    }
    if mos > 5.0 {
        mos = 5.0
    }
    
    return mos
}

func (m *VoIPQualityMonitor) QualityRating() string {
    mos := m.GetMOS()
    switch {
    case mos >= 4.3:
        return "Excellent"
    case mos >= 4.0:
        return "Good"
    case mos >= 3.6:
        return "Fair"
    case mos >= 3.1:
        return "Poor"
    default:
        return "Bad"
    }
}

// ICE candidate gatherer
type ICECandidate struct {
    Type       string // host, srflx, relay
    Foundation string
    Component  int
    Protocol   string // udp, tcp
    Priority   uint32
    Address    string
    Port       int
    RelAddr    string
    RelPort    int
}

type ICEGatherer struct {
    stunServer string
    turnServer string
    candidates []ICECandidate
}

func NewICEGatherer(stunServer, turnServer string) *ICEGatherer {
    return &ICEGatherer{
        stunServer: stunServer,
        turnServer: turnServer,
    }
}

func (g *ICEGatherer) GatherHostCandidates() []ICECandidate {
    var candidates []ICECandidate
    
    addrs, err := net.InterfaceAddrs()
    if err != nil {
        return candidates
    }
    
    for _, addr := range addrs {
        ipNet, ok := addr.(*net.IPNet)
        if !ok || ipNet.IP.IsLoopback() {
            continue
        }
        
        ip := ipNet.IP.To4()
        if ip == nil {
            continue // Skip IPv6 for simplicity
        }
        
        candidate := ICECandidate{
            Type:      "host",
            Component: 1,
            Protocol:  "udp",
            Priority:  calculatePriority("host", 1),
            Address:   ip.String(),
            Port:      0, // Would bind to random port
        }
        candidates = append(candidates, candidate)
    }
    
    g.candidates = append(g.candidates, candidates...)
    return candidates
}

func calculatePriority(candidateType string, component int) uint32 {
    var typePreference uint32
    switch candidateType {
    case "host":
        typePreference = 126
    case "srflx":
        typePreference = 100
    case "relay":
        typePreference = 0
    }
    
    localPreference := uint32(65535)
    return (typePreference << 24) | (localPreference << 8) | uint32(256-component)
}`,
				},
			},
		},
		{
			ID:          2629,
			Title:       "Network Storage Protocols",
			Description: "Learn about iSCSI, NFS, SMB/CIFS, Fibre Channel, storage area networks, and network-attached storage architectures.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "Network Storage and SAN Technologies",
					Content: `Network storage protocols enable shared storage access over networks, from file-level NFS/SMB to block-level iSCSI and Fibre Channel.

**Storage Architectures:**

DAS (Direct Attached Storage):
  Storage directly connected to server
  SATA, SAS, NVMe interfaces
  Simple, fast, single-server access
  No network overhead

NAS (Network Attached Storage):
  File-level storage over network
  NFS (Unix/Linux) or SMB (Windows)
  Easy sharing across clients
  Managed via file system

SAN (Storage Area Network):
  Block-level storage over dedicated network
  Fibre Channel or iSCSI
  High performance, enterprise grade
  Appears as local disk to server
  
  SAN Components:
    Initiator: Server requesting storage
    Target: Storage device/array
    Fabric: Network connecting them (FC switches)
    LUN: Logical Unit Number (storage unit)
    Zone: FC switch access control
    Masking: LUN-level access control

**NFS (Network File System):**

NFSv3:
  Stateless protocol
  UDP or TCP
  File locking via NLM (separate protocol)
  AUTH_SYS authentication
  Up to 64KB read/write
  
NFSv4:
  Stateful protocol
  TCP only (port 2049)
  Built-in file locking
  Kerberos authentication (RPCSEC_GSS)
  Compound operations (reduce round trips)
  Namespace federation (referrals)
  
NFSv4.1/4.2:
  pNFS (parallel NFS):
    Client communicates directly with storage devices
    Metadata server provides layout maps
    Layouts: Files, Blocks, Objects
    Massive parallel performance
  
  Server-side copy (v4.2):
    Copy data without sending through client
    Reduces network bandwidth
  
  Space reservation:
    Allocate space without writing data
  
NFS Performance Tuning:
  rsize/wsize: Read/write block size (optimal: 1MB)
  async vs sync: Async for performance
  noatime: Don't update access time
  nconnect: Multiple TCP connections
  MTU: Use jumbo frames (9000)

**SMB/CIFS:**

SMB (Server Message Block):
  Microsoft protocol for file/printer sharing
  Port 445 (TCP)
  
  SMB Versions:
    SMBv1/CIFS: Legacy, insecure (disable!)
    SMBv2: Vista/Server 2008
    SMBv2.1: Windows 7/2008 R2 (large MTU, leasing)
    SMBv3.0: Windows 8/2012 (encryption, multichannel)
    SMBv3.0.2: Windows 8.1/2012 R2
    SMBv3.1.1: Windows 10/2016 (pre-auth integrity, AES-128-GCM)
    
  SMBv3 Features:
    SMB Encryption: AES-128-CCM/GCM
    SMB Multichannel: Multiple NICs, load balance
    SMB Direct: RDMA support (low latency)
    Continuous Availability: Transparent failover
    Directory leasing: Cached directory listings
    
  SMB Security:
    NTLM authentication (legacy)
    Kerberos authentication (preferred)
    Signing: Protects against tampering
    Encryption: End-to-end (SMBv3+)

**iSCSI Protocol:**

  Block storage over TCP/IP
  Port 3260
  
  iSCSI Components:
    Initiator: Client software/hardware
    Target: Storage device
    IQN: iSCSI Qualified Name (unique identifier)
      iqn.2024-01.com.example:storage.target1
    
  iSCSI Session:
    Discovery: Find available targets
    Login: Authenticate and negotiate
    Full Feature Phase: SCSI commands over TCP
    Logout: End session
    
  iSCSI Security:
    CHAP authentication (bidirectional)
    IPsec encryption
    iSNS (iSCSI Name Service) for discovery
    ACLs based on initiator IQN
    
  iSCSI Multipathing (MPIO):
    Multiple paths to same storage
    Active/Active or Active/Passive
    Failover on path failure
    Load balancing across paths

**Fibre Channel:**

  Dedicated high-speed network for storage
  Speeds: 8/16/32/64 Gbps
  Low latency, lossless
  
  FC Protocol Layers:
    FC-4: Protocol mapping (SCSI, IP)
    FC-3: Common services (multicast)
    FC-2: Framing, flow control, QoS
    FC-1: Encoding (8B/10B, 64B/66B)
    FC-0: Physical (fiber optic, copper)
    
  FC Topologies:
    Point-to-Point: Direct connection
    Arbitrated Loop (FC-AL): Shared loop (legacy)
    Switched Fabric: Full fabric with FC switches
    
  FC Addressing:
    WWNN (World Wide Node Name): Device identifier
    WWPN (World Wide Port Name): Port identifier
    FC-ID: Switch-assigned address (24-bit)
    
  Zoning:
    Hard zoning: Switch port-based
    Soft zoning: WWPN-based
    Smart zoning: Limits initiator-to-initiator traffic
    
  FCoE (Fibre Channel over Ethernet):
    FC frames encapsulated in Ethernet
    Requires lossless Ethernet (DCB/PFC)
    Converged Network Adapter (CNA)
    Reduces cable/switch count

**NVMe over Fabrics (NVMe-oF):**

  NVMe protocol over network
  Very low latency
  
  Transports:
    NVMe/TCP: Standard TCP/IP
    NVMe/RDMA: RoCE or InfiniBand
    NVMe/FC: Over Fibre Channel
    
  Benefits:
    Native NVMe performance over network
    Multipath and namespace sharing
    Scalable to thousands of namespaces
    
  Compared to iSCSI:
    Lower latency (microseconds vs milliseconds)
    Higher IOPS (millions vs hundreds of thousands)
    More CPU efficient (less protocol overhead)`,
					CodeExamples: `// Network storage protocol implementations

package main

import (
    "encoding/binary"
    "fmt"
    "io"
    "net"
    "strings"
    "sync"
    "time"
)

// iSCSI target simulator
type ISCSITarget struct {
    iqn       string
    luns      map[int]*StorageLUN
    mu        sync.RWMutex
    sessions  map[string]*ISCSISession
    authDatabase map[string]string // initiator IQN -> CHAP secret
}

type StorageLUN struct {
    ID       int
    Size     int64  // bytes
    Data     []byte
    ReadOnly bool
}

type ISCSISession struct {
    ISID         string
    InitiatorIQN string
    TargetIQN    string
    State        string
    MaxBurstLen  int
    FirstBurstLen int
    MaxRecvDataSeg int
    CmdSN        uint32
    ExpStatSN    uint32
}

func NewISCSITarget(iqn string) *ISCSITarget {
    return &ISCSITarget{
        iqn:          iqn,
        luns:         make(map[int]*StorageLUN),
        sessions:     make(map[string]*ISCSISession),
        authDatabase: make(map[string]string),
    }
}

func (t *ISCSITarget) AddLUN(id int, sizeBytes int64) {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    t.luns[id] = &StorageLUN{
        ID:   id,
        Size: sizeBytes,
        Data: make([]byte, sizeBytes),
    }
}

func (t *ISCSITarget) AddInitiator(iqn, chapSecret string) {
    t.mu.Lock()
    defer t.mu.Unlock()
    t.authDatabase[iqn] = chapSecret
}

func (t *ISCSITarget) Authenticate(initiatorIQN, secret string) bool {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    expected, exists := t.authDatabase[initiatorIQN]
    if !exists {
        return false
    }
    return expected == secret
}

func (t *ISCSITarget) Read(lunID int, offset int64, length int) ([]byte, error) {
    t.mu.RLock()
    defer t.mu.RUnlock()
    
    lun, exists := t.luns[lunID]
    if !exists {
        return nil, fmt.Errorf("LUN %d not found", lunID)
    }
    
    if offset+int64(length) > lun.Size {
        return nil, fmt.Errorf("read beyond LUN size")
    }
    
    data := make([]byte, length)
    copy(data, lun.Data[offset:offset+int64(length)])
    return data, nil
}

func (t *ISCSITarget) Write(lunID int, offset int64, data []byte) error {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    lun, exists := t.luns[lunID]
    if !exists {
        return fmt.Errorf("LUN %d not found", lunID)
    }
    
    if lun.ReadOnly {
        return fmt.Errorf("LUN %d is read-only", lunID)
    }
    
    if offset+int64(len(data)) > lun.Size {
        return fmt.Errorf("write beyond LUN size")
    }
    
    copy(lun.Data[offset:], data)
    return nil
}

// NFS RPC call simulator
type NFSClient struct {
    serverAddr string
    version    int
    auth       NFSAuth
    rootFH     []byte
    conn       net.Conn
}

type NFSAuth struct {
    Flavor int    // 1=AUTH_SYS, 6=RPCSEC_GSS
    UID    uint32
    GID    uint32
    Groups []uint32
}

type NFSFileAttr struct {
    Type     string // regular, directory, symlink
    Mode     uint32
    UID      uint32
    GID      uint32
    Size     int64
    Atime    time.Time
    Mtime    time.Time
    Ctime    time.Time
}

type NFSExport struct {
    Path       string
    Clients    []string // Allowed client patterns
    Options    ExportOptions
}

type ExportOptions struct {
    ReadWrite    bool
    Sync         bool
    NoRootSquash bool
    AllSquash    bool
    AnonUID      uint32
    AnonGID      uint32
    SecFlavors   []string // sys, krb5, krb5i, krb5p
}

type NFSServer struct {
    exports []NFSExport
    mu      sync.RWMutex
}

func NewNFSServer() *NFSServer {
    return &NFSServer{}
}

func (s *NFSServer) AddExport(export NFSExport) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.exports = append(s.exports, export)
}

func (s *NFSServer) CheckAccess(clientIP string, exportPath string) (*NFSExport, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    for _, export := range s.exports {
        if export.Path != exportPath {
            continue
        }
        
        for _, pattern := range export.Clients {
            if matchClient(clientIP, pattern) {
                return &export, nil
            }
        }
        
        return nil, fmt.Errorf("client %s not authorized for %s", clientIP, exportPath)
    }
    
    return nil, fmt.Errorf("export %s not found", exportPath)
}

func matchClient(ip, pattern string) bool {
    if pattern == "*" {
        return true
    }
    
    // Simple subnet match
    if strings.Contains(pattern, "/") {
        _, cidr, err := net.ParseCIDR(pattern)
        if err != nil {
            return false
        }
        return cidr.Contains(net.ParseIP(ip))
    }
    
    return ip == pattern
}

// SMB share manager
type SMBServer struct {
    shares    map[string]*SMBShare
    mu        sync.RWMutex
    sessions  map[string]*SMBSession
}

type SMBShare struct {
    Name        string
    Path        string
    Description string
    MaxUsers    int
    ReadOnly    bool
    ACL         []SMBAccessRule
}

type SMBAccessRule struct {
    Principal  string // user or group
    Permission string // full, read, change, none
}

type SMBSession struct {
    SessionID  uint64
    User       string
    ClientIP   string
    Dialect    string // SMB 2.0.2, 3.1.1, etc.
    SigningReq bool
    Encrypted  bool
    Connected  time.Time
    TreeConns  map[uint32]string // tree ID -> share name
}

func NewSMBServer() *SMBServer {
    return &SMBServer{
        shares:   make(map[string]*SMBShare),
        sessions: make(map[string]*SMBSession),
    }
}

func (s *SMBServer) CreateShare(share SMBShare) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.shares[share.Name] = &share
}

func (s *SMBServer) CheckShareAccess(shareName, user string) (string, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    share, exists := s.shares[shareName]
    if !exists {
        return "", fmt.Errorf("share %s not found", shareName)
    }
    
    for _, rule := range share.ACL {
        if rule.Principal == user || rule.Principal == "Everyone" {
            if rule.Permission == "none" {
                return "", fmt.Errorf("access denied for %s", user)
            }
            return rule.Permission, nil
        }
    }
    
    return "", fmt.Errorf("no ACL entry for %s", user)
}

// Multipath I/O manager
type MPIOManager struct {
    pathGroups map[string]*PathGroup
    mu         sync.RWMutex
    policy     string // round-robin, least-pending, failover
}

type PathGroup struct {
    TargetIQN string
    LunID     int
    Paths     []*StoragePath
    ActiveIdx int
}

type StoragePath struct {
    ID         string
    TargetAddr string
    State      string // active, standby, failed
    Latency    time.Duration
    IOCount    uint64
    PendingIO  int32
    LastCheck  time.Time
}

func NewMPIOManager(policy string) *MPIOManager {
    return &MPIOManager{
        pathGroups: make(map[string]*PathGroup),
        policy:     policy,
    }
}

func (m *MPIOManager) AddPath(targetIQN string, lunID int, path *StoragePath) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    key := fmt.Sprintf("%s:lun%d", targetIQN, lunID)
    group, exists := m.pathGroups[key]
    if !exists {
        group = &PathGroup{
            TargetIQN: targetIQN,
            LunID:     lunID,
        }
        m.pathGroups[key] = group
    }
    
    group.Paths = append(group.Paths, path)
}

func (m *MPIOManager) SelectPath(targetIQN string, lunID int) (*StoragePath, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    key := fmt.Sprintf("%s:lun%d", targetIQN, lunID)
    group, exists := m.pathGroups[key]
    if !exists {
        return nil, fmt.Errorf("no paths for %s", key)
    }
    
    activePaths := make([]*StoragePath, 0)
    for _, p := range group.Paths {
        if p.State == "active" {
            activePaths = append(activePaths, p)
        }
    }
    
    if len(activePaths) == 0 {
        return nil, fmt.Errorf("no active paths for %s", key)
    }
    
    switch m.policy {
    case "round-robin":
        idx := group.ActiveIdx % len(activePaths)
        group.ActiveIdx++
        return activePaths[idx], nil
        
    case "least-pending":
        best := activePaths[0]
        for _, p := range activePaths[1:] {
            if p.PendingIO < best.PendingIO {
                best = p
            }
        }
        return best, nil
        
    case "failover":
        return activePaths[0], nil
        
    default:
        return activePaths[0], nil
    }
}

func (m *MPIOManager) MarkFailed(pathID string) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    for _, group := range m.pathGroups {
        for _, p := range group.Paths {
            if p.ID == pathID {
                p.State = "failed"
                p.LastCheck = time.Now()
                return
            }
        }
    }
}`,
				},
			},
		},
	})
}
