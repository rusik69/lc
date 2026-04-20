package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2223,
			Title:       "Python Networking and Security",
			Description: "Master socket programming, HTTP clients, SSL/TLS, cryptography, secure coding practices, and network protocols in Python.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "Sockets HTTP Clients SSL Cryptography and Security",
					Content: `Python provides comprehensive networking and security libraries for building robust, secure applications.

**Socket Programming:**

Socket Basics:
  socket.socket(AF_INET, SOCK_STREAM) — TCP
  socket.socket(AF_INET, SOCK_DGRAM) — UDP
  socket.socket(AF_INET6, SOCK_STREAM) — TCP IPv6
  
  Address families:
    AF_INET: IPv4
    AF_INET6: IPv6
    AF_UNIX: Unix domain sockets
    
  Socket types:
    SOCK_STREAM: TCP (reliable, ordered)
    SOCK_DGRAM: UDP (unreliable, unordered)
    SOCK_RAW: Raw socket access

TCP Server:
  sock.bind((host, port))
  sock.listen(backlog)
  conn, addr = sock.accept()
  data = conn.recv(buffer_size)
  conn.send(data)
  conn.close()

TCP Client:
  sock.connect((host, port))
  sock.send(data)
  data = sock.recv(buffer_size)
  sock.close()

Socket Options:
  sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
  sock.settimeout(seconds)
  sock.setblocking(False)
  
  SO_REUSEADDR: Reuse address
  SO_KEEPALIVE: Keep connection alive
  TCP_NODELAY: Disable Nagle's algorithm
  SO_RCVBUF/SO_SNDBUF: Buffer sizes

Select/Poll:
  select.select(rlist, wlist, xlist, timeout)
  select.poll() — poll-based I/O multiplexing
  selectors module — high-level I/O multiplexing
  
  selectors.DefaultSelector()
  sel.register(sock, EVENT_READ, callback)
  events = sel.select(timeout)

socketserver module:
  TCPServer, UDPServer
  ThreadingMixIn, ForkingMixIn
  BaseRequestHandler: handle() method

**HTTP Clients:**

urllib:
  urllib.request.urlopen(url)
  urllib.parse.urlencode(params)
  urllib.parse.urlparse(url)
  
  Request object:
    req = urllib.request.Request(url, headers={...})
    
requests library:
  requests.get(url, params={...})
  requests.post(url, json={...})
  requests.put(url, data={...})
  requests.delete(url)
  requests.patch(url)
  
  Response:
    response.status_code
    response.json()
    response.text
    response.headers
    response.cookies
    response.content (bytes)
    response.raise_for_status()
    
  Sessions:
    session = requests.Session()
    session.headers.update({...})
    session.cookies.set(name, value)
    
  Authentication:
    requests.get(url, auth=('user', 'pass'))
    HTTPDigestAuth
    
  Timeouts:
    requests.get(url, timeout=(connect, read))
    
  Retries:
    HTTPAdapter + Retry
    
httpx (async HTTP):
  async with httpx.AsyncClient() as client:
      response = await client.get(url)
  
  HTTP/2 support
  Connection pooling
  Async/sync API
  
  httpx.Client() — sync client
  httpx.AsyncClient() — async client

aiohttp (async HTTP):
  async with aiohttp.ClientSession() as session:
      async with session.get(url) as response:
          data = await response.json()
  
  Server:
    app = aiohttp.web.Application()
    app.router.add_get('/', handler)
    aiohttp.web.run_app(app)

**SSL/TLS:**

ssl module:
  ssl.create_default_context() — recommended
  context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
  
  Server:
    context.load_cert_chain(certfile, keyfile)
    
  Client:
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_verify_locations(cafile)
    
  Wrapping sockets:
    ssl_sock = context.wrap_socket(sock, server_hostname=host)
    
  Best practices:
    Always verify certificates
    Use latest TLS version
    Don't disable hostname checking
    Use strong cipher suites

Certificate handling:
  ssl.get_server_certificate((host, port))
  ssl.PEM_cert_to_DER_cert(pem)
  
  OpenSSL integration:
    pyOpenSSL for advanced certificate manipulation

**Cryptography:**

hashlib module:
  hashlib.sha256(data).hexdigest()
  hashlib.sha512(data).digest()
  hashlib.blake2b(data).hexdigest()
  hashlib.pbkdf2_hmac('sha256', password, salt, 100000)
  hashlib.scrypt(password, salt=salt, n=16384, r=8, p=1)
  
  PBKDF2: Password-based key derivation
  scrypt: Memory-hard password hashing
  
hmac module:
  hmac.new(key, msg, hashlib.sha256).hexdigest()
  hmac.compare_digest(a, b) — timing-safe comparison

secrets module:
  secrets.token_bytes(32) — random bytes
  secrets.token_hex(32) — random hex string
  secrets.token_urlsafe(32) — URL-safe token
  secrets.randbelow(n) — random int below n
  secrets.choice(seq) — random choice

cryptography library:
  Fernet (symmetric encryption):
    key = Fernet.generate_key()
    f = Fernet(key)
    token = f.encrypt(data)
    original = f.decrypt(token)
    
  AES (Advanced Encryption Standard):
    AES-GCM (authenticated)
    AES-CBC (with HMAC for auth)
    AES-CTR
    
  RSA:
    Key generation
    Signing/verification
    Encryption/decryption
    
  ECDSA:
    Elliptic curve digital signatures
    Smaller keys, faster operations
    
  X.509 Certificates:
    Certificate creation
    Certificate parsing
    Chain validation
    
  Key derivation:
    PBKDF2HMAC
    Scrypt
    HKDF
    
  Password hashing:
    bcrypt (via bcrypt library)
    argon2 (via argon2-cffi)

**Secure Coding Practices:**

Input Validation:
  Validate all external input
  Whitelist over blacklist
  Parameterized queries (prevent SQL injection)
  HTML escaping (prevent XSS)
  Path traversal prevention

Secrets Management:
  Never hardcode secrets
  Use environment variables or secret manager
  python-dotenv for .env files
  
  os.environ.get('SECRET_KEY')
  
Authentication:
  Use bcrypt or argon2 for passwords
  Constant-time comparison for tokens
  JWT with proper validation
  Session management best practices

OWASP Top 10:
  A01: Broken Access Control
  A02: Cryptographic Failures
  A03: Injection
  A04: Insecure Design
  A05: Security Misconfiguration
  A06: Vulnerable Components
  A07: Authentication Failures
  A08: Data Integrity Failures
  A09: Logging Failures
  A10: SSRF

Security Libraries:
  bandit: Security linter
  safety: Dependency vulnerability scanner
  pip-audit: Audit pip dependencies
  pyup: Dependency updates

**Network Protocols:**

DNS:
  socket.getaddrinfo(host, port)
  socket.gethostbyname(host)
  dnspython library for advanced DNS

SMTP:
  smtplib.SMTP(host, port)
  smtplib.SMTP_SSL(host, 465)
  server.starttls()
  server.login(user, password)
  server.sendmail(from_addr, to_addrs, msg)

FTP:
  ftplib.FTP(host)
  ftplib.FTP_TLS(host) — encrypted
  ftp.login(user, passwd)
  ftp.retrbinary('RETR file', callback)
  ftp.storbinary('STOR file', file)

SSH:
  paramiko library
  SSHClient, SFTPClient
  Key-based authentication
  Port forwarding
  
  Fabric: High-level SSH automation

WebSockets:
  websockets library (asyncio)
  websocket-client library (sync)
  
  Server:
    async with websockets.serve(handler, host, port):
        await asyncio.Future()
  
  Client:
    async with websockets.connect(uri) as ws:
        await ws.send(data)
        response = await ws.recv()

gRPC:
  Protocol Buffers (.proto files)
  grpcio library
  grpcio-tools for code generation
  Unary, server streaming, client streaming, bidirectional`,
					CodeExamples: `# Python networking and security examples

import hashlib
import hmac
import os
import secrets
import socket
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# Socket Programming
# ============================================================

class TCPServer:
    """Simple TCP server with connection handling."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8080):
        self.host = host
        self.port = port
        self._handlers: Dict[str, Callable] = {}
        self._running = False
    
    def route(self, path: str):
        def decorator(handler):
            self._handlers[path] = handler
            return handler
        return decorator
    
    def handle_connection(self, conn: socket.socket, addr: tuple):
        """Handle a single client connection."""
        try:
            data = conn.recv(4096)
            if not data:
                return
            
            request = self._parse_request(data.decode('utf-8', errors='replace'))
            
            handler = self._handlers.get(request.get('path', '/'))
            if handler:
                response = handler(request)
            else:
                response = {'status': 404, 'body': 'Not Found'}
            
            response_bytes = self._format_response(response)
            conn.sendall(response_bytes)
        except Exception as e:
            error_response = self._format_response({
                'status': 500,
                'body': f'Internal Server Error: {str(e)}'
            })
            try:
                conn.sendall(error_response)
            except Exception:
                pass
        finally:
            conn.close()
    
    def _parse_request(self, raw: str) -> dict:
        lines = raw.split('\r\n')
        if not lines:
            return {}
        
        parts = lines[0].split(' ')
        request = {
            'method': parts[0] if parts else 'GET',
            'path': parts[1] if len(parts) > 1 else '/',
            'version': parts[2] if len(parts) > 2 else 'HTTP/1.1',
            'headers': {},
        }
        
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                request['headers'][key.strip()] = value.strip()
            elif line == '':
                break
        
        # Body after empty line
        body_start = raw.find('\r\n\r\n')
        if body_start >= 0:
            request['body'] = raw[body_start + 4:]
        
        return request
    
    def _format_response(self, response: dict) -> bytes:
        status = response.get('status', 200)
        body = response.get('body', '')
        content_type = response.get('content_type', 'text/plain')
        
        status_text = {
            200: 'OK', 201: 'Created', 204: 'No Content',
            301: 'Moved Permanently', 302: 'Found',
            400: 'Bad Request', 401: 'Unauthorized',
            403: 'Forbidden', 404: 'Not Found',
            500: 'Internal Server Error',
        }.get(status, 'Unknown')
        
        body_bytes = body.encode('utf-8') if isinstance(body, str) else body
        
        headers = [
            f'HTTP/1.1 {status} {status_text}',
            f'Content-Type: {content_type}',
            f'Content-Length: {len(body_bytes)}',
            'Connection: close',
            '',
            '',
        ]
        
        return '\r\n'.join(headers).encode('utf-8') + body_bytes


class MessageProtocol:
    """Length-prefixed message protocol for TCP."""
    
    HEADER_FORMAT = '!I'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    @staticmethod
    def send_message(sock: socket.socket, data: bytes):
        """Send a length-prefixed message."""
        header = struct.pack(MessageProtocol.HEADER_FORMAT, len(data))
        sock.sendall(header + data)
    
    @staticmethod
    def recv_message(sock: socket.socket) -> Optional[bytes]:
        """Receive a length-prefixed message."""
        header = MessageProtocol._recv_exact(sock, MessageProtocol.HEADER_SIZE)
        if not header:
            return None
        
        msg_len = struct.unpack(MessageProtocol.HEADER_FORMAT, header)[0]
        return MessageProtocol._recv_exact(sock, msg_len)
    
    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)


class ConnectionPool:
    """TCP connection pool."""
    
    def __init__(self, host: str, port: int, max_connections: int = 10):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self._available: List[socket.socket] = []
        self._in_use: int = 0
    
    def acquire(self) -> socket.socket:
        if self._available:
            self._in_use += 1
            return self._available.pop()
        
        if self._in_use >= self.max_connections:
            raise RuntimeError("Connection pool exhausted")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        self._in_use += 1
        return sock
    
    def release(self, sock: socket.socket):
        self._in_use -= 1
        try:
            # Check if connection is still alive
            sock.setblocking(False)
            try:
                data = sock.recv(1, socket.MSG_PEEK)
                if data:
                    self._available.append(sock)
                else:
                    sock.close()
            except BlockingIOError:
                self._available.append(sock)
            finally:
                sock.setblocking(True)
        except Exception:
            try:
                sock.close()
            except Exception:
                pass
    
    def close_all(self):
        for sock in self._available:
            try:
                sock.close()
            except Exception:
                pass
        self._available.clear()


# ============================================================
# HTTP Client
# ============================================================

class HTTPClient:
    """Simple HTTP client."""
    
    def __init__(self, base_url: str = '', timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.headers: Dict[str, str] = {
            'User-Agent': 'PythonHTTPClient/1.0',
        }
        self._session_cookies: Dict[str, str] = {}
    
    def get(self, path: str, params: dict = None, 
            headers: dict = None) -> 'HTTPResponse':
        return self._request('GET', path, params=params, headers=headers)
    
    def post(self, path: str, data: Any = None,
             json_data: dict = None, headers: dict = None) -> 'HTTPResponse':
        return self._request('POST', path, data=data, 
                           json_data=json_data, headers=headers)
    
    def put(self, path: str, data: Any = None,
            json_data: dict = None, headers: dict = None) -> 'HTTPResponse':
        return self._request('PUT', path, data=data,
                           json_data=json_data, headers=headers)
    
    def delete(self, path: str, headers: dict = None) -> 'HTTPResponse':
        return self._request('DELETE', path, headers=headers)
    
    def _request(self, method: str, path: str, params: dict = None,
                 data: Any = None, json_data: dict = None,
                 headers: dict = None) -> 'HTTPResponse':
        url = f"{self.base_url}{path}"
        
        if params:
            query = '&'.join(f"{k}={v}" for k, v in params.items())
            url += f"?{query}"
        
        all_headers = {**self.headers}
        if headers:
            all_headers.update(headers)
        
        body = ''
        if json_data is not None:
            import json
            body = json.dumps(json_data)
            all_headers['Content-Type'] = 'application/json'
        elif data is not None:
            body = str(data)
        
        # Build raw HTTP request
        request_line = f"{method} {path} HTTP/1.1\r\n"
        header_lines = ''.join(f"{k}: {v}\r\n" for k, v in all_headers.items())
        
        if body:
            header_lines += f"Content-Length: {len(body)}\r\n"
        
        raw_request = f"{request_line}{header_lines}\r\n{body}"
        
        # Simulate response
        return HTTPResponse(
            status_code=200,
            headers={'Content-Type': 'application/json'},
            body=f'{{"method": "{method}", "path": "{path}"}}'
        )


@dataclass 
class HTTPResponse:
    status_code: int
    headers: Dict[str, str]
    body: str
    
    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300
    
    def json(self) -> dict:
        import json
        return json.loads(self.body)
    
    @property
    def text(self) -> str:
        return self.body
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(self.status_code, self.body)


class HTTPError(Exception):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class RetryClient:
    """HTTP client with retry logic."""
    
    def __init__(self, client: HTTPClient, max_retries: int = 3,
                 backoff_factor: float = 0.5,
                 retry_statuses: tuple = (429, 500, 502, 503, 504)):
        self._client = client
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._retry_statuses = retry_statuses
    
    def get(self, path: str, **kwargs) -> HTTPResponse:
        return self._retry('get', path, **kwargs)
    
    def post(self, path: str, **kwargs) -> HTTPResponse:
        return self._retry('post', path, **kwargs)
    
    def _retry(self, method: str, path: str, **kwargs) -> HTTPResponse:
        last_response = None
        
        for attempt in range(self._max_retries + 1):
            response = getattr(self._client, method)(path, **kwargs)
            last_response = response
            
            if response.status_code not in self._retry_statuses:
                return response
            
            if attempt < self._max_retries:
                wait = self._backoff_factor * (2 ** attempt)
                time.sleep(wait)
        
        return last_response


# ============================================================
# Cryptography
# ============================================================

class PasswordManager:
    """Secure password hashing and verification."""
    
    HASH_ITERATIONS = 100000
    SALT_LENGTH = 32
    HASH_ALGORITHM = 'sha256'
    
    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_bytes(PasswordManager.SALT_LENGTH)
        hash_bytes = hashlib.pbkdf2_hmac(
            PasswordManager.HASH_ALGORITHM,
            password.encode('utf-8'),
            salt,
            PasswordManager.HASH_ITERATIONS
        )
        return f"{salt.hex()}${hash_bytes.hex()}"
    
    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        try:
            salt_hex, hash_hex = stored_hash.split('$')
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)
            
            actual_hash = hashlib.pbkdf2_hmac(
                PasswordManager.HASH_ALGORITHM,
                password.encode('utf-8'),
                salt,
                PasswordManager.HASH_ITERATIONS
            )
            
            return hmac.compare_digest(actual_hash, expected_hash)
        except (ValueError, IndexError):
            return False


class TokenManager:
    """Secure token generation and validation."""
    
    def __init__(self, secret_key: str):
        self._secret_key = secret_key.encode('utf-8')
    
    def generate_token(self, data: str, ttl: int = 3600) -> str:
        timestamp = str(int(time.time()))
        expires = str(int(time.time()) + ttl)
        
        payload = f"{data}|{timestamp}|{expires}"
        signature = hmac.new(
            self._secret_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"{payload}|{signature}"
    
    def verify_token(self, token: str) -> Optional[str]:
        try:
            parts = token.rsplit('|', 1)
            if len(parts) != 2:
                return None
            
            payload, received_sig = parts
            
            expected_sig = hmac.new(
                self._secret_key,
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(received_sig, expected_sig):
                return None
            
            data_parts = payload.split('|')
            if len(data_parts) != 3:
                return None
            
            data, timestamp, expires = data_parts
            
            if int(expires) < int(time.time()):
                return None
            
            return data
        except (ValueError, IndexError):
            return None


class SimpleEncryption:
    """Simple XOR encryption (for educational purposes only)."""
    
    def __init__(self, key: bytes):
        self._key = key
    
    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(
            p ^ self._key[i % len(self._key)]
            for i, p in enumerate(plaintext)
        )
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        return self.encrypt(ciphertext)  # XOR is symmetric


class CSRFProtection:
    """CSRF token generation and validation."""
    
    def __init__(self, secret_key: str):
        self._secret_key = secret_key.encode('utf-8')
        self._tokens: Dict[str, float] = {}
        self._token_ttl = 3600
    
    def generate_token(self, session_id: str) -> str:
        token = secrets.token_hex(32)
        
        sig = hmac.new(
            self._secret_key,
            f"{session_id}:{token}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        full_token = f"{token}.{sig}"
        self._tokens[full_token] = time.time()
        
        return full_token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        if token not in self._tokens:
            return False
        
        if time.time() - self._tokens[token] > self._token_ttl:
            del self._tokens[token]
            return False
        
        parts = token.split('.')
        if len(parts) != 2:
            return False
        
        raw_token, received_sig = parts
        
        expected_sig = hmac.new(
            self._secret_key,
            f"{session_id}:{raw_token}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(received_sig, expected_sig)
    
    def cleanup(self):
        now = time.time()
        expired = [t for t, ts in self._tokens.items()
                   if now - ts > self._token_ttl]
        for t in expired:
            del self._tokens[t]


class InputSanitizer:
    """Input validation and sanitization."""
    
    DANGEROUS_CHARS = '<>&"' + "'"
    SQL_KEYWORDS = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP',
                    'UNION', 'OR', 'AND', '--', ';'}
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#x27;',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        upper = text.upper()
        for keyword in InputSanitizer.SQL_KEYWORDS:
            if keyword in upper:
                return True
        return False
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        # Prevent path traversal
        path = path.replace('..', '')
        path = path.replace('//', '/')
        if path.startswith('/'):
            path = path[1:]
        
        # Only allow alphanumeric, dots, hyphens, underscores, slashes
        return ''.join(
            c for c in path
            if c.isalnum() or c in '.-_/'
        )
    
    @staticmethod
    def validate_email(email: str) -> bool:
        if '@' not in email or '.' not in email:
            return False
        local, domain = email.rsplit('@', 1)
        if not local or not domain:
            return False
        if '..' in domain:
            return False
        parts = domain.split('.')
        return all(part.isalnum() or '-' in part for part in parts)`,
				},
			},
		},
	})
}
