package networking

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterNetworkingModules([]problems.CourseModule{
		{
			ID:          2615,
			Title:       "Modern Networking Protocols",
			Description: "Learn modern networking protocols and concepts: HTTP/2, HTTP/3, gRPC, WebSockets, and API protocols that power today's internet infrastructure.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "HTTP/2 and HTTP/3",
					Content: `The evolution of HTTP has been driven by the need to reduce latency and improve performance.

**HTTP/1.1 Problems:**
- **Head-of-line blocking**: One slow request blocks all subsequent requests on the same connection
- **Single request per connection**: Browsers open 6-8 connections per domain as a workaround
- **Uncompressed headers**: Headers sent in plain text, often 500+ bytes per request
- **No prioritization**: All requests treated equally

**HTTP/2 (2015):**

**Key Features:**
1. **Multiplexing**: Multiple requests/responses on a single TCP connection, interleaved
2. **Header compression (HPACK)**: Dramatic reduction in header overhead
3. **Server push**: Server can proactively send resources before client asks
4. **Stream prioritization**: Assign priorities to requests
5. **Binary protocol**: More efficient parsing than text-based HTTP/1.1

` + "```" + `
HTTP/1.1:
Connection 1: [Request A]────────[Response A]
Connection 2: [Request B]────[Response B]
Connection 3: [Request C]──────────[Response C]

HTTP/2 (single connection):
Stream 1: [Request A]──[Response A chunks]
Stream 2: [Request B]──[Response B chunks]
Stream 3: [Request C]──[Response C chunks]
(All interleaved on ONE TCP connection)
` + "```" + `

**HTTP/2 Limitation**: Still uses TCP, which has its own head-of-line blocking at the transport layer. If a TCP packet is lost, ALL streams are blocked until retransmission.

**HTTP/3 (2022):**

Built on QUIC (UDP-based transport protocol developed by Google).

**Key Improvements over HTTP/2:**
1. **No TCP head-of-line blocking**: Streams are independent at transport level
2. **Faster connection setup**: 0-RTT or 1-RTT (vs TCP's 3-way handshake + TLS)
3. **Connection migration**: Survives IP changes (mobile WiFi → cellular)
4. **Built-in encryption**: TLS 1.3 mandatory, integrated into transport

` + "```" + `
Connection Setup Comparison:
  HTTP/1.1 + TLS 1.2: TCP SYN → SYN-ACK → ACK → TLS (2 RTT) = 3 RTT
  HTTP/2 + TLS 1.3:   TCP SYN → SYN-ACK → ACK → TLS (1 RTT) = 2 RTT
  HTTP/3 + QUIC:      QUIC handshake = 1 RTT (0-RTT on reconnect!)
` + "```" + `

**When to Use Each:**
- **HTTP/1.1**: Legacy systems, simple APIs, not performance-critical
- **HTTP/2**: Default for most web applications today
- **HTTP/3**: Mobile apps, latency-sensitive apps, lossy networks (cellular)

**Adoption (2024):**
- HTTP/2: ~35% of all websites, ~95% of browsers support
- HTTP/3: ~30% of all websites (Cloudflare, Google serve HTTP/3 by default)`,
					CodeExamples: `# Nginx: Enable HTTP/2
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    # HTTP/2 specific settings
    http2_max_concurrent_streams 128;
    http2_idle_timeout 5m;
}

# Nginx: Enable HTTP/3 (QUIC)
server {
    listen 443 ssl;
    listen 443 quic;  # UDP port for QUIC
    http2 on;
    http3 on;

    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    # Advertise HTTP/3 support
    add_header Alt-Svc 'h3=":443"; ma=86400';
}

# Caddy: Automatic HTTP/3 (zero config!)
# Caddy enables HTTP/2 and HTTP/3 by default
example.com {
    respond "Hello, HTTP/3!"
}

# Go: HTTP/2 client
package main

import (
    "crypto/tls"
    "fmt"
    "net/http"
    "golang.org/x/net/http2"
)

func main() {
    // Go's default HTTP client supports HTTP/2 automatically
    resp, err := http.Get("https://example.com")
    fmt.Println(resp.Proto) // "HTTP/2.0"

    // Force HTTP/2 with custom transport
    transport := &http2.Transport{
        TLSClientConfig: &tls.Config{},
    }
    client := &http.Client{Transport: transport}
    resp, _ = client.Get("https://example.com")
}

# curl: Test HTTP versions
curl -v --http2 https://example.com    # Force HTTP/2
curl -v --http3 https://example.com    # Force HTTP/3 (needs curl 7.66+)

# Check protocol negotiation
curl -w '%{http_version}\n' -o /dev/null -s https://example.com`,
				},
				{
					Title: "WebSockets and Real-Time Communication",
					Content: `WebSockets provide full-duplex communication channels over a single TCP connection, enabling real-time data exchange between client and server.

**Why WebSockets?**
HTTP is request-response: client asks, server answers. But for real-time apps (chat, live updates, games), you need the server to push data without a client request.

**Alternatives to WebSockets:**

| Technique | Direction | Latency | Complexity | Use Case |
|-----------|-----------|---------|-----------|----------|
| Polling | Client → Server | High (interval) | Low | Legacy systems |
| Long Polling | Server → Client | Medium | Medium | Server-push, simple |
| SSE (Server-Sent Events) | Server → Client only | Low | Low | Live feeds, notifications |
| WebSocket | Bidirectional | Very Low | Medium | Chat, games, trading |

**WebSocket Protocol:**

` + "```" + `
1. HTTP Upgrade handshake:
   Client: GET /ws HTTP/1.1
           Upgrade: websocket
           Connection: Upgrade
           Sec-WebSocket-Key: dGhlIHNhbXBsZQ==

   Server: HTTP/1.1 101 Switching Protocols
           Upgrade: websocket
           Connection: Upgrade
           Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

2. Bidirectional communication over single TCP connection
3. Message framing (text or binary)
4. Ping/pong for keepalive
5. Close handshake
` + "```" + `

**Server-Sent Events (SSE):**
Simpler alternative when you only need server → client push:
- Uses regular HTTP (works with load balancers, CDNs)
- Auto-reconnection built into browser API
- Text-only (no binary)
- One-directional (server to client)
- Great for: notifications, live feeds, dashboards

**Real-Time Architecture Patterns:**

**1. Hub/Room Pattern (Chat):**
` + "```" + `
Client A ──→ Hub ──→ Room "general" ──→ [Client A, Client B, Client C]
Client B ──→     ──→ Room "team-1"  ──→ [Client A, Client D]
` + "```" + `

**2. Pub/Sub with Message Broker:**
` + "```" + `
Clients ←→ WebSocket Servers ←→ Redis Pub/Sub ←→ WebSocket Servers ←→ Clients
` + "```" + `
Scales horizontally: multiple WebSocket servers share state via Redis.

**3. Event Sourcing + WebSocket:**
` + "```" + `
Write: Client → API → Event Store → Publish Event
Read:  Event Stream → WebSocket Server → Subscribed Clients
` + "```" + `

**Scaling WebSockets:**
- WebSockets are stateful (connection per client)
- Can't use simple round-robin load balancing
- Solutions: sticky sessions, Redis Pub/Sub, or dedicated WebSocket service
- Each server can handle ~10K-100K concurrent connections (depending on hardware and message rate)

**When NOT to Use WebSockets:**
- Server → client only (use SSE instead, simpler)
- Infrequent updates (use polling, less complexity)
- RESTful CRUD (use standard HTTP)
- When you need HTTP caching (WebSocket bypasses HTTP cache)`,
					CodeExamples: `// Go: WebSocket server using gorilla/websocket
package main

import (
    "log"
    "net/http"
    "sync"
    "github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
    CheckOrigin: func(r *http.Request) bool { return true },
}

// Hub manages all connected clients
type Hub struct {
    mu      sync.RWMutex
    clients map[*websocket.Conn]bool
}

func (h *Hub) Broadcast(message []byte) {
    h.mu.RLock()
    defer h.mu.RUnlock()
    for conn := range h.clients {
        conn.WriteMessage(websocket.TextMessage, message)
    }
}

func (h *Hub) Register(conn *websocket.Conn) {
    h.mu.Lock()
    h.clients[conn] = true
    h.mu.Unlock()
}

func (h *Hub) Unregister(conn *websocket.Conn) {
    h.mu.Lock()
    delete(h.clients, conn)
    h.mu.Unlock()
    conn.Close()
}

var hub = &Hub{clients: make(map[*websocket.Conn]bool)}

func wsHandler(w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Println("Upgrade error:", err)
        return
    }

    hub.Register(conn)
    defer hub.Unregister(conn)

    for {
        _, message, err := conn.ReadMessage()
        if err != nil {
            break
        }
        hub.Broadcast(message)
    }
}

func main() {
    http.HandleFunc("/ws", wsHandler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}

# JavaScript: WebSocket client
const ws = new WebSocket('wss://example.com/ws');

ws.onopen = () => {
    console.log('Connected');
    ws.send(JSON.stringify({ type: 'join', room: 'general' }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

ws.onclose = () => {
    console.log('Disconnected');
    // Implement reconnection logic
    setTimeout(() => connectWebSocket(), 1000);
};

# Server-Sent Events (simpler alternative)
# Go server:
func sseHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    flusher, _ := w.(http.Flusher)

    for {
        fmt.Fprintf(w, "data: %s\n\n", time.Now().String())
        flusher.Flush()
        time.Sleep(1 * time.Second)
    }
}

# JavaScript client (built-in API):
const source = new EventSource('/events');
source.onmessage = (event) => {
    console.log(event.data);
};`,
				},
				{
					Title: "gRPC and Protocol Buffers",
					Content: `gRPC is a high-performance, open-source RPC framework developed by Google. It uses Protocol Buffers (protobuf) for serialization and HTTP/2 for transport.

**Why gRPC?**

| Feature | REST/JSON | gRPC/Protobuf |
|---------|-----------|---------------|
| Serialization | JSON (text, ~5-10KB) | Protobuf (binary, ~1-2KB) |
| Speed | ~10ms parsing | ~1ms parsing |
| Schema | Optional (OpenAPI) | Required (.proto) |
| Streaming | Limited (SSE, WebSocket) | Built-in bidirectional |
| Code generation | Optional | Built-in |
| Browser support | Full | Limited (grpc-web) |
| Human readable | Yes | No (binary) |

**When to Use gRPC:**
- Service-to-service communication (microservices)
- High-throughput, low-latency requirements
- Streaming data (real-time, logs, metrics)
- Polyglot environments (strong cross-language support)
- Need strict API contracts (schema enforcement)

**When to Use REST:**
- Public APIs (browser clients)
- Simple CRUD operations
- Human debugging (readable JSON)
- Broad ecosystem compatibility
- Wide cache support

**gRPC Communication Patterns:**

1. **Unary**: Client sends one request, server sends one response (like REST)
2. **Server streaming**: Client sends one request, server streams multiple responses
3. **Client streaming**: Client streams multiple requests, server sends one response
4. **Bidirectional streaming**: Both client and server stream messages

**Protocol Buffers:**
Binary serialization format. Define messages and services in .proto files, generate code in any language.

**Benefits:**
- 3-10x smaller than JSON
- 20-100x faster to parse than JSON
- Backward/forward compatible (add fields without breaking)
- Type-safe (compile-time checks)

**gRPC in Production:**
- Used by: Google, Netflix, Square, Cisco, Docker
- Service mesh integration: Istio, Linkerd support gRPC natively
- Load balancing: L7 load balancers (Envoy) understand gRPC
- Health checking: gRPC has a standard health checking protocol`,
					CodeExamples: `# Protocol Buffer definition (.proto)
syntax = "proto3";
package userservice;

option go_package = "github.com/example/user/pb";

// Message definitions
message User {
    string id = 1;
    string name = 2;
    string email = 3;
    int32 age = 4;
    repeated string roles = 5;
}

message GetUserRequest {
    string id = 1;
}

message ListUsersRequest {
    int32 page_size = 1;
    string page_token = 2;
}

message ListUsersResponse {
    repeated User users = 1;
    string next_page_token = 2;
}

// Service definition
service UserService {
    // Unary RPC
    rpc GetUser(GetUserRequest) returns (User);
    
    // Server streaming
    rpc ListUsers(ListUsersRequest) returns (stream User);
    
    // Client streaming
    rpc UploadUsers(stream User) returns (UploadResponse);
    
    // Bidirectional streaming
    rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

// Go: gRPC server implementation
type userServer struct {
    pb.UnimplementedUserServiceServer
    users map[string]*pb.User
}

func (s *userServer) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    user, ok := s.users[req.Id]
    if !ok {
        return nil, status.Errorf(codes.NotFound, "user %s not found", req.Id)
    }
    return user, nil
}

// Server streaming
func (s *userServer) ListUsers(req *pb.ListUsersRequest, stream pb.UserService_ListUsersServer) error {
    for _, user := range s.users {
        if err := stream.Send(user); err != nil {
            return err
        }
    }
    return nil
}

func main() {
    lis, _ := net.Listen("tcp", ":50051")
    grpcServer := grpc.NewServer()
    pb.RegisterUserServiceServer(grpcServer, &userServer{
        users: make(map[string]*pb.User),
    })
    grpcServer.Serve(lis)
}

// Go: gRPC client
func main() {
    conn, _ := grpc.Dial("localhost:50051", grpc.WithInsecure())
    defer conn.Close()
    client := pb.NewUserServiceClient(conn)

    // Unary call
    user, err := client.GetUser(context.Background(), &pb.GetUserRequest{Id: "123"})

    // Server streaming
    stream, _ := client.ListUsers(context.Background(), &pb.ListUsersRequest{})
    for {
        user, err := stream.Recv()
        if err == io.EOF { break }
        fmt.Println(user.Name)
    }
}`,
				},
			},
		},
	})
}
