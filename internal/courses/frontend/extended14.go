package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1573,
			Title:       "Real-time Communication and WebSockets",
			Description: "Implement real-time features with WebSockets, Server-Sent Events, WebRTC, and real-time synchronization patterns for collaborative applications.",
			Order:       73,
			Lessons: []problems.Lesson{
				{
					Title: "WebSocket Architecture and Patterns",
					Content: `WebSockets provide full-duplex communication channels over a single TCP connection for real-time data exchange.

**WebSocket Client Implementation:**
` + "```" + `javascript
// Robust WebSocket client with reconnection
class WebSocketClient {
  constructor(url, options = {}) {
    this.url = url;
    this.options = {
      reconnectInterval: 1000,
      maxReconnectInterval: 30000,
      reconnectDecay: 1.5,
      maxReconnectAttempts: Infinity,
      pingInterval: 30000,
      ...options,
    };
    
    this.ws = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.isConnecting = false;
    this.messageQueue = [];
    this.pingTimer = null;
    
    this.connect();
  }

  connect() {
    if (this.isConnecting) return;
    this.isConnecting = true;

    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      this.handleClose();
      return;
    }

    this.ws.onopen = () => {
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      
      // Flush queued messages
      while (this.messageQueue.length > 0) {
        const msg = this.messageQueue.shift();
        this.ws.send(msg);
      }
      
      // Start ping/pong
      this.startPing();
      this.emit('connected');
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'pong') return;
        
        // Emit typed event
        this.emit(data.type, data.payload);
        this.emit('message', data);
      } catch {
        this.emit('message', event.data);
      }
    };

    this.ws.onclose = (event) => {
      this.isConnecting = false;
      this.stopPing();
      
      if (!event.wasClean) {
        this.handleReconnect();
      }
      
      this.emit('disconnected', { code: event.code, reason: event.reason });
    };

    this.ws.onerror = (error) => {
      this.emit('error', error);
    };
  }

  handleReconnect() {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      this.emit('max_reconnect_reached');
      return;
    }

    const interval = Math.min(
      this.options.reconnectInterval * 
        Math.pow(this.options.reconnectDecay, this.reconnectAttempts),
      this.options.maxReconnectInterval
    );

    this.reconnectAttempts++;
    this.emit('reconnecting', { attempt: this.reconnectAttempts, delay: interval });
    
    setTimeout(() => this.connect(), interval);
  }

  send(type, payload) {
    const message = JSON.stringify({ type, payload, timestamp: Date.now() });
    
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    } else {
      this.messageQueue.push(message);
    }
  }

  startPing() {
    this.pingTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, this.options.pingInterval);
  }

  stopPing() {
    clearInterval(this.pingTimer);
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.listeners.get(event).delete(callback);
  }

  emit(event, data) {
    this.listeners.get(event)?.forEach((cb) => cb(data));
  }

  close() {
    this.options.maxReconnectAttempts = 0;
    this.stopPing();
    this.ws?.close(1000, 'Client closing');
  }
}

// React hook for WebSocket
function useWebSocket(url, options = {}) {
  const [status, setStatus] = useState('disconnected');
  const [lastMessage, setLastMessage] = useState(null);
  const clientRef = useRef(null);

  useEffect(() => {
    const client = new WebSocketClient(url, options);
    clientRef.current = client;

    client.on('connected', () => setStatus('connected'));
    client.on('disconnected', () => setStatus('disconnected'));
    client.on('reconnecting', () => setStatus('reconnecting'));
    client.on('message', (msg) => setLastMessage(msg));

    return () => client.close();
  }, [url]);

  const send = useCallback((type, payload) => {
    clientRef.current?.send(type, payload);
  }, []);

  return { status, lastMessage, send };
}
` + "```" + `

**Server-Sent Events (SSE):**
` + "```" + `javascript
// SSE client - simpler than WebSocket for server-to-client streaming
function useEventSource(url, options = {}) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState('connecting');

  useEffect(() => {
    const source = new EventSource(url, {
      withCredentials: options.withCredentials,
    });

    source.onopen = () => setStatus('connected');
    source.onerror = () => {
      setStatus('error');
      // EventSource auto-reconnects
    };

    // Default message handler
    source.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };

    // Named event handlers
    if (options.events) {
      for (const [eventName, handler] of Object.entries(options.events)) {
        source.addEventListener(eventName, (event) => {
          handler(JSON.parse(event.data));
        });
      }
    }

    return () => source.close();
  }, [url]);

  return { data, status };
}

// Usage - Live notifications
function NotificationFeed({ userId }) {
  const { data } = useEventSource('/api/notifications/stream?userId=' + userId, {
    events: {
      notification: (data) => {
        showToast(data.message);
        playNotificationSound();
      },
      heartbeat: () => {}, // Keep connection alive
    },
  });

  return data ? <NotificationList items={data.notifications} /> : null;
}

// Server endpoint (Node.js/Express)
// app.get('/api/notifications/stream', (req, res) => {
//   res.writeHead(200, {
//     'Content-Type': 'text/event-stream',
//     'Cache-Control': 'no-cache',
//     Connection: 'keep-alive',
//   });
//
//   const send = (event, data) => {
//     res.write('event: ' + event + '\n');
//     res.write('data: ' + JSON.stringify(data) + '\n\n');
//   };
//
//   // Send heartbeat every 30s
//   const heartbeat = setInterval(() => send('heartbeat', {}), 30000);
//
//   // Listen for new notifications
//   const unsub = notificationService.subscribe(req.user.id, (notification) => {
//     send('notification', notification);
//   });
//
//   req.on('close', () => {
//     clearInterval(heartbeat);
//     unsub();
//   });
// });
` + "```" + `

**Collaborative Editing with CRDT:**
` + "```" + `javascript
// Simplified CRDT for collaborative text editing
// Using Yjs for production collaborative editing

import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';

function useCollaborativeEditor(roomId) {
  const [doc] = useState(() => new Y.Doc());
  const [provider, setProvider] = useState(null);
  const [awareness, setAwareness] = useState(null);
  const [peers, setPeers] = useState([]);

  useEffect(() => {
    const wsProvider = new WebsocketProvider(
      'wss://collaboration.example.com',
      roomId,
      doc,
      { connect: true }
    );

    const awarenessInstance = wsProvider.awareness;
    
    // Set local user info
    awarenessInstance.setLocalStateField('user', {
      name: currentUser.name,
      color: generateColor(currentUser.id),
      cursor: null,
    });

    // Track other users' presence
    awarenessInstance.on('change', () => {
      const states = Array.from(awarenessInstance.getStates().entries())
        .filter(([clientId]) => clientId !== doc.clientID)
        .map(([, state]) => state.user)
        .filter(Boolean);
      setPeers(states);
    });

    setProvider(wsProvider);
    setAwareness(awarenessInstance);

    return () => {
      wsProvider.disconnect();
      doc.destroy();
    };
  }, [roomId]);

  // Get shared text type
  const text = doc.getText('content');

  return { doc, text, provider, awareness, peers };
}

// Usage with rich text editor (TipTap)
function CollaborativeEditor({ roomId }) {
  const { doc, provider, awareness, peers } = useCollaborativeEditor(roomId);
  
  const editor = useEditor({
    extensions: [
      StarterKit.configure({ history: false }),
      Collaboration.configure({ document: doc }),
      CollaborationCursor.configure({
        provider,
        user: { name: currentUser.name, color: '#3b82f6' },
      }),
    ],
  });

  return (
    <div>
      <div className="presence-bar">
        {peers.map((peer, i) => (
          <span key={i} style={{ color: peer.color }}>
            {peer.name}
          </span>
        ))}
      </div>
      <EditorContent editor={editor} />
    </div>
  );
}

// Optimistic updates with conflict resolution
class OptimisticStore {
  constructor(wsClient) {
    this.state = {};
    this.pendingOps = [];
    this.version = 0;
    this.ws = wsClient;
    
    this.ws.on('state_update', (data) => {
      this.handleServerUpdate(data);
    });
  }

  dispatch(operation) {
    // Apply optimistically
    const opId = crypto.randomUUID();
    const op = { ...operation, id: opId, clientVersion: this.version };
    
    this.state = this.applyOp(this.state, op);
    this.pendingOps.push(op);
    
    // Send to server
    this.ws.send('operation', op);
    
    return opId;
  }

  handleServerUpdate({ state, version, acknowledgedOps }) {
    // Remove acknowledged operations
    this.pendingOps = this.pendingOps.filter(
      (op) => !acknowledgedOps.includes(op.id)
    );
    
    // Rebase: apply pending ops on top of server state
    let newState = state;
    for (const op of this.pendingOps) {
      newState = this.applyOp(newState, op);
    }
    
    this.state = newState;
    this.version = version;
    this.notify();
  }

  applyOp(state, op) {
    switch (op.type) {
      case 'set':
        return { ...state, [op.key]: op.value };
      case 'delete':
        const next = { ...state };
        delete next[op.key];
        return next;
      case 'increment':
        return { ...state, [op.key]: (state[op.key] || 0) + op.amount };
      default:
        return state;
    }
  }
}
` + "```" + ``,
					CodeExamples: `// WebRTC for peer-to-peer communication

// 1. Simple WebRTC data channel
class PeerConnection {
  constructor(signaling) {
    this.signaling = signaling;
    this.pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
      ],
    });
    
    this.dataChannel = null;
    this.onMessage = null;
    
    this.pc.onicecandidate = (event) => {
      if (event.candidate) {
        signaling.send('ice-candidate', event.candidate);
      }
    };
    
    this.pc.ondatachannel = (event) => {
      this.setupDataChannel(event.channel);
    };
    
    signaling.on('offer', (offer) => this.handleOffer(offer));
    signaling.on('answer', (answer) => this.handleAnswer(answer));
    signaling.on('ice-candidate', (candidate) => {
      this.pc.addIceCandidate(new RTCIceCandidate(candidate));
    });
  }

  async createOffer() {
    this.dataChannel = this.pc.createDataChannel('data');
    this.setupDataChannel(this.dataChannel);
    
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    this.signaling.send('offer', offer);
  }

  async handleOffer(offer) {
    await this.pc.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await this.pc.createAnswer();
    await this.pc.setLocalDescription(answer);
    this.signaling.send('answer', answer);
  }

  async handleAnswer(answer) {
    await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
  }

  setupDataChannel(channel) {
    this.dataChannel = channel;
    channel.onopen = () => console.log('Data channel open');
    channel.onclose = () => console.log('Data channel closed');
    channel.onmessage = (event) => {
      this.onMessage?.(JSON.parse(event.data));
    };
  }

  send(data) {
    if (this.dataChannel?.readyState === 'open') {
      this.dataChannel.send(JSON.stringify(data));
    }
  }

  close() {
    this.dataChannel?.close();
    this.pc.close();
  }
}

// 2. React hook for real-time presence
function usePresence(channelName) {
  const [users, setUsers] = useState(new Map());
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocketClient('/ws/presence/' + channelName);
    
    ws.current.on('presence_state', (state) => {
      setUsers(new Map(Object.entries(state)));
    });
    
    ws.current.on('presence_join', ({ userId, data }) => {
      setUsers(prev => new Map(prev).set(userId, data));
    });
    
    ws.current.on('presence_leave', ({ userId }) => {
      setUsers(prev => {
        const next = new Map(prev);
        next.delete(userId);
        return next;
      });
    });
    
    ws.current.on('presence_update', ({ userId, data }) => {
      setUsers(prev => new Map(prev).set(userId, { ...prev.get(userId), ...data }));
    });

    // Announce presence
    ws.current.send('join', {
      userId: currentUser.id,
      name: currentUser.name,
      status: 'online',
    });

    // Track cursor position for collaborative features
    const handleMouseMove = throttle((e) => {
      ws.current.send('update', {
        cursor: { x: e.clientX, y: e.clientY },
      });
    }, 50);
    
    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      ws.current.send('leave', { userId: currentUser.id });
      ws.current.close();
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, [channelName]);

  return { users: Array.from(users.values()) };
}

// Usage - show live cursors
function CollaborativeCanvas() {
  const { users } = usePresence('design-room-1');
  
  return (
    <div style={{ position: 'relative' }}>
      <Canvas />
      {users.map(user => user.cursor && (
        <div
          key={user.userId}
          style={{
            position: 'absolute',
            left: user.cursor.x,
            top: user.cursor.y,
            pointerEvents: 'none',
            transition: 'all 0.1s linear',
          }}
        >
          <CursorIcon color={user.color} />
          <span style={{ fontSize: 12 }}>{user.name}</span>
        </div>
      ))}
    </div>
  );
}`,
				},
			},
		},
	})
}
