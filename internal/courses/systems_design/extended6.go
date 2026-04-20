package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          2424,
			Title:       "Security Architecture and Zero Trust",
			Description: "Master authentication patterns, authorization models, zero trust architecture, API security, encryption at rest and in transit, and security monitoring.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Authentication Authorization Encryption and Security Monitoring",
					Content: `Security architecture is a critical aspect of system design. Understanding authentication, authorization, encryption, and monitoring patterns is essential for building secure distributed systems.

**Authentication Patterns:**

Session-Based:
  User logs in → server creates session → stores session ID in cookie
  Server stores session data (memory, Redis, DB)
  
  Pros: Simple, server controls sessions
  Cons: Stateful, session storage scaling, CSRF vulnerability
  
Token-Based (JWT):
  User logs in → server creates signed JWT → client stores token
  Client sends token in Authorization header
  Server verifies signature without state
  
  JWT Structure:
    Header: Algorithm, token type
    Payload: Claims (sub, exp, iat, iss, custom)
    Signature: HMAC or RSA/ECDSA
    
  Access tokens:
    Short-lived (15 min - 1 hour)
    Contains user identity and permissions
    
  Refresh tokens:
    Long-lived (days - weeks)
    Used to obtain new access tokens
    Stored securely, can be revoked
    
  Token rotation:
    Issue new refresh token with each use
    Detect reuse → revoke all tokens (token theft detection)

OAuth 2.0 Flows:
  Authorization Code: Server-side apps (most secure)
  PKCE: Mobile/SPA apps (no client secret)
  Client Credentials: Service-to-service (M2M)
  Device Code: Input-constrained devices (TV, CLI)
  
  Never use Implicit flow (deprecated, insecure)

Multi-Factor Authentication (MFA):
  Something you know: Password, PIN
  Something you have: Phone, hardware key (FIDO2/WebAuthn)
  Something you are: Biometrics
  
  TOTP: Time-based one-time password (Google Authenticator)
  SMS: Less secure (SIM swapping)
  Push notification: Approve/deny on device
  Hardware keys: U2F/FIDO2 (strongest)

Passkeys (WebAuthn/FIDO2):
  Public key cryptography
  No passwords to phish or leak
  Device-bound or synced across devices
  Strong phishing resistance

**Authorization Models:**

RBAC (Role-Based Access Control):
  Users → Roles → Permissions
  Simple, well-understood
  Coarse-grained
  Examples: admin, editor, viewer

ABAC (Attribute-Based Access Control):
  Policies based on attributes
  Subject attributes: Role, department, clearance
  Resource attributes: Type, owner, classification
  Environment: Time, location, IP
  Fine-grained, flexible, complex

ReBAC (Relationship-Based Access Control):
  Access based on relationships between entities
  User is owner/editor/viewer of resource
  Google Zanzibar model (used in Google Drive)
  Examples: SpiceDB, Authzed, OpenFGA

Policy-as-Code:
  OPA (Open Policy Agent): Rego language
  Cedar (AWS): Permit/forbid policies
  Casbin: Multiple models support
  
  Benefits:
    Version controlled
    Testable
    Auditable
    Decoupled from application

**Zero Trust Architecture:**

Principles:
  Never trust, always verify
  Assume breach
  Verify explicitly
  Least privilege access
  Micro-segmentation

Components:
  Identity verification: Strong authentication for every request
  Device validation: Check device health and compliance
  Network segmentation: Micro-perimeters around resources
  Least privilege: Minimal access for each identity
  Continuous monitoring: Real-time threat assessment
  Encryption everywhere: TLS for all communication

Implementation:
  Service mesh with mTLS (Istio, Linkerd)
  Identity-aware proxy (Google BeyondCorp, Cloudflare Access)
  Network policies (Kubernetes NetworkPolicy, Calico)
  Secret management (HashiCorp Vault, AWS Secrets Manager)

**API Security:**

Authentication:
  API keys: Simple, rotate regularly
  OAuth 2.0 tokens: Standard, granular scopes
  Mutual TLS: Certificate-based, strong identity

Rate Limiting:
  Per-user, per-IP, per-API key
  Token bucket or sliding window
  Return 429 with Retry-After

Input Validation:
  Validate all input at boundary
  Whitelist over blacklist
  SQL injection: Parameterized queries
  XSS: Output encoding, CSP headers
  SSRF: Validate URLs, blocklist internal IPs
  
Request Signing:
  HMAC-based request signing
  Include timestamp to prevent replay
  AWS Signature V4 pattern
  
  Steps:
    Create canonical request
    Create string to sign (canonical hash + timestamp)
    Calculate signature (HMAC with secret key)
    Add signature to Authorization header

API Versioning Security:
  Deprecate old versions with known vulnerabilities
  Force upgrade timeline
  Monitor usage of deprecated versions

**Encryption:**

At Rest:
  AES-256: Block cipher, standard for data at rest
  Envelope encryption: Data key encrypted by master key
  KMS (Key Management Service): Manage encryption keys
  
  Database encryption:
    TDE (Transparent Data Encryption): Encrypts data files
    Column-level: Encrypt specific sensitive columns
    Application-level: Encrypt before writing

In Transit:
  TLS 1.3: Latest standard
    Fewer round trips (1-RTT, 0-RTT)
    Removed weak ciphers
    Forward secrecy mandatory
    
  Certificate management:
    CA-signed certificates
    Auto-renewal (Let's Encrypt, cert-manager)
    Certificate pinning (mobile apps)
    mTLS for service-to-service

Key Management:
  Separation of duties
  Key rotation schedule
  HSM (Hardware Security Module) for master keys
  Never store keys with data
  Audit key access

**Security Monitoring:**

SIEM (Security Information and Event Management):
  Collect logs from all systems
  Correlate events across services
  Detect anomalies and threats
  Alert on security incidents

Logging for Security:
  Authentication events (login, logout, failures)
  Authorization decisions (allow, deny)
  Data access (who accessed what)
  Configuration changes
  API calls with metadata
  
  Never log: Passwords, tokens, PII, credit cards

Intrusion Detection:
  Network-based (NIDS): Monitor traffic patterns
  Host-based (HIDS): Monitor system calls, file integrity
  Application-based: WAF, RASP

Incident Response:
  Detection → Containment → Eradication → Recovery → Lessons
  
  Runbooks for common incidents
  On-call rotation
  Post-incident review (blameless)
  
  Communication:
    Internal: Status page, incident channel
    External: Customer notification, regulatory reporting`,
					CodeExamples: `# Security Architecture Implementation Examples

import time
import hashlib
import hmac
import secrets
import base64
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod

# ============================================================
# JWT Implementation (Simplified HMAC-based)
# ============================================================

class JWT:
    """Simplified JWT implementation (HMAC-SHA256)."""
    
    @staticmethod
    def encode(payload: Dict[str, Any], secret: str) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()).decode().rstrip('=')
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()).decode().rstrip('=')
        
        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            secret.encode(),
            signing_input.encode(),
            hashlib.sha256
        ).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f"{header_b64}.{payload_b64}.{sig_b64}"
    
    @staticmethod
    def decode(token: str, secret: str) -> Optional[Dict[str, Any]]:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        header_b64, payload_b64, sig_b64 = parts
        
        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            secret.encode(),
            signing_input.encode(),
            hashlib.sha256
        ).digest()
        expected_b64 = base64.urlsafe_b64encode(
            expected_sig).decode().rstrip('=')
        
        if not hmac.compare_digest(sig_b64, expected_b64):
            return None
        
        # Decode payload
        padding = 4 - len(payload_b64) % 4
        payload_b64 += '=' * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        
        # Check expiration
        if 'exp' in payload and payload['exp'] < time.time():
            return None
        
        return payload


class TokenManager:
    """Manage access and refresh tokens."""
    
    def __init__(self, secret: str, access_ttl: int = 900,
                 refresh_ttl: int = 604800):
        self.secret = secret
        self.access_ttl = access_ttl
        self.refresh_ttl = refresh_ttl
        self._revoked_tokens: Set[str] = set()
        self._refresh_tokens: Dict[str, Dict] = {}
    
    def create_tokens(self, user_id: str,
                      roles: List[str]) -> Tuple[str, str]:
        now = time.time()
        
        access_payload = {
            "sub": user_id,
            "roles": roles,
            "type": "access",
            "iat": now,
            "exp": now + self.access_ttl,
            "jti": secrets.token_hex(16),
        }
        access_token = JWT.encode(access_payload, self.secret)
        
        refresh_id = secrets.token_hex(32)
        self._refresh_tokens[refresh_id] = {
            "user_id": user_id,
            "roles": roles,
            "created_at": now,
            "expires_at": now + self.refresh_ttl,
        }
        
        return access_token, refresh_id
    
    def verify_access(self, token: str) -> Optional[Dict]:
        payload = JWT.decode(token, self.secret)
        if payload is None:
            return None
        
        jti = payload.get("jti")
        if jti in self._revoked_tokens:
            return None
        
        return payload
    
    def refresh(self, refresh_id: str) -> Optional[Tuple[str, str]]:
        data = self._refresh_tokens.get(refresh_id)
        if data is None:
            return None
        
        if time.time() > data["expires_at"]:
            del self._refresh_tokens[refresh_id]
            return None
        
        # Rotate refresh token
        del self._refresh_tokens[refresh_id]
        return self.create_tokens(data["user_id"], data["roles"])
    
    def revoke(self, token: str):
        payload = JWT.decode(token, self.secret)
        if payload and "jti" in payload:
            self._revoked_tokens.add(payload["jti"])


# ============================================================
# RBAC Implementation
# ============================================================

@dataclass
class Permission:
    resource: str
    action: str
    
    def __hash__(self):
        return hash((self.resource, self.action))
    
    def __eq__(self, other):
        return self.resource == other.resource and self.action == other.action


@dataclass
class Role:
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)


class RBAC:
    """Role-Based Access Control."""
    
    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, Set[str]] = defaultdict(set)
    
    def create_role(self, name: str, permissions: List[Tuple[str, str]] = None,
                    parent_roles: List[str] = None):
        perms = {Permission(r, a) for r, a in (permissions or [])}
        self._roles[name] = Role(
            name=name,
            permissions=perms,
            parent_roles=set(parent_roles or []),
        )
    
    def assign_role(self, user_id: str, role_name: str):
        if role_name not in self._roles:
            raise ValueError(f"Role {role_name} not found")
        self._user_roles[user_id].add(role_name)
    
    def revoke_role(self, user_id: str, role_name: str):
        self._user_roles[user_id].discard(role_name)
    
    def check_permission(self, user_id: str, resource: str,
                        action: str) -> bool:
        required = Permission(resource, action)
        
        for role_name in self._user_roles.get(user_id, set()):
            if self._role_has_permission(role_name, required, set()):
                return True
        
        return False
    
    def _role_has_permission(self, role_name: str,
                            permission: Permission,
                            visited: Set[str]) -> bool:
        if role_name in visited:
            return False
        visited.add(role_name)
        
        role = self._roles.get(role_name)
        if role is None:
            return False
        
        if permission in role.permissions:
            return True
        
        # Check wildcard
        wildcard = Permission(permission.resource, "*")
        if wildcard in role.permissions:
            return True
        
        # Check parent roles
        for parent in role.parent_roles:
            if self._role_has_permission(parent, permission, visited):
                return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        permissions = set()
        for role_name in self._user_roles.get(user_id, set()):
            self._collect_permissions(role_name, permissions, set())
        return permissions
    
    def _collect_permissions(self, role_name: str,
                            permissions: Set[Permission],
                            visited: Set[str]):
        if role_name in visited:
            return
        visited.add(role_name)
        
        role = self._roles.get(role_name)
        if role is None:
            return
        
        permissions.update(role.permissions)
        for parent in role.parent_roles:
            self._collect_permissions(parent, permissions, visited)


# ============================================================
# HMAC Request Signing
# ============================================================

class RequestSigner:
    """HMAC-based request signing."""
    
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
    
    def sign(self, method: str, path: str, headers: Dict[str, str],
             body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time()))
        
        # Create canonical request
        canonical = self._canonical_request(
            method, path, headers, body, timestamp)
        
        # Create string to sign
        string_to_sign = f"HMAC-SHA256\n{timestamp}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
        
        # Calculate signature
        signature = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        signed_headers = dict(headers)
        signed_headers["X-Timestamp"] = timestamp
        signed_headers["Authorization"] = (
            f"HMAC-SHA256 Credential={self.access_key}, "
            f"Signature={signature}")
        
        return signed_headers
    
    def verify(self, method: str, path: str, headers: Dict[str, str],
               body: str = "", max_age: int = 300) -> bool:
        timestamp = headers.get("X-Timestamp")
        auth = headers.get("Authorization", "")
        
        if not timestamp or not auth:
            return False
        
        # Check timestamp freshness
        try:
            ts = int(timestamp)
            if abs(time.time() - ts) > max_age:
                return False
        except ValueError:
            return False
        
        # Extract signature from header
        if not auth.startswith("HMAC-SHA256"):
            return False
        
        parts = auth.split("Signature=")
        if len(parts) != 2:
            return False
        provided_sig = parts[1]
        
        # Recalculate
        verify_headers = {k: v for k, v in headers.items()
                         if k not in ("Authorization",)}
        
        canonical = self._canonical_request(
            method, path, verify_headers, body, timestamp)
        
        string_to_sign = f"HMAC-SHA256\n{timestamp}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
        
        expected_sig = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(provided_sig, expected_sig)
    
    def _canonical_request(self, method: str, path: str,
                          headers: Dict[str, str], body: str,
                          timestamp: str) -> str:
        sorted_headers = '\n'.join(
            f"{k.lower()}:{v}" for k, v in sorted(headers.items()))
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        return f"{method}\n{path}\n{sorted_headers}\n{timestamp}\n{body_hash}"


# ============================================================
# Audit Logger
# ============================================================

class AuditEventType(Enum):
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    CONFIG_CHANGE = "config.change"
    ADMIN_ACTION = "admin.action"


@dataclass
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    timestamp: float
    user_id: str
    resource: str
    action: str
    result: str
    source_ip: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Security audit logging."""
    
    def __init__(self):
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._alerts: List[Callable] = []
    
    def log(self, event_type: AuditEventType, user_id: str,
            resource: str, action: str, result: str,
            source_ip: str = "", **metadata):
        event = AuditEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            source_ip=source_ip,
            metadata=metadata,
        )
        
        with self._lock:
            self._events.append(event)
        
        # Check alert rules
        for alert_fn in self._alerts:
            alert_fn(event)
    
    def add_alert_rule(self, rule: Callable[[AuditEvent], None]):
        self._alerts.append(rule)
    
    def query(self, user_id: str = None,
              event_type: AuditEventType = None,
              start_time: float = None,
              end_time: float = None,
              limit: int = 100) -> List[AuditEvent]:
        with self._lock:
            events = self._events
            
            if user_id:
                events = [e for e in events if e.user_id == user_id]
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            
            return events[-limit:]
    
    def failed_login_count(self, user_id: str,
                          window_seconds: float = 300) -> int:
        cutoff = time.time() - window_seconds
        with self._lock:
            return sum(1 for e in self._events
                      if e.user_id == user_id
                      and e.event_type == AuditEventType.AUTH_FAILED
                      and e.timestamp >= cutoff)`,
				},
			},
		},
	})
}
