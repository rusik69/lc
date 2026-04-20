package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1564,
			Title:       "Web Security for Frontend Applications",
			Description: "Protect web applications against XSS, CSRF, injection attacks, and implement Content Security Policy, authentication, and secure coding practices.",
			Order:       64,
			Lessons: []problems.Lesson{
				{
					Title: "XSS Prevention and Content Security Policy",
					Content: `Cross-Site Scripting (XSS) allows attackers to inject malicious scripts into web pages viewed by other users.

**Types of XSS:**
` + "```" + `
Reflected XSS:
  Attack URL: https://site.com/search?q=<script>steal(document.cookie)</script>
  Server reflects input directly into HTML response
  Prevention: Encode output, validate input

Stored XSS:
  Attacker stores malicious content in database
  Example: Forum post with <img onerror="steal()" src="x">
  Every user who views the post gets attacked
  Prevention: Sanitize on input AND encode on output

DOM-based XSS:
  Vulnerable JavaScript uses untrusted data in DOM manipulation
  Example: document.innerHTML = location.hash.slice(1)
  Prevention: Use textContent instead of innerHTML, sanitize

Mutation XSS (mXSS):
  Exploits browser HTML parser quirks
  Sanitizer sees safe HTML, browser reparses into dangerous HTML
  Example: <svg><style><img src=x onerror=alert(1)//</style></svg>
  Prevention: Use trusted sanitizer libraries (DOMPurify)
` + "```" + `

**XSS Prevention Techniques:**
` + "```" + `javascript
// 1. NEVER use innerHTML with untrusted data
// BAD
element.innerHTML = userInput;
document.write(userInput);

// GOOD - use textContent for text
element.textContent = userInput;

// 2. React auto-escapes by default (safe)
function Comment({ text }) {
  return <p>{text}</p>; // Auto-escaped
}

// DANGEROUS - dangerouslySetInnerHTML
function Comment({ html }) {
  // Only use with sanitized content!
  return <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html) }} />;
}

// 3. DOMPurify for HTML sanitization
import DOMPurify from 'dompurify';

// Basic sanitization
const clean = DOMPurify.sanitize(dirty);

// Allow specific tags/attributes
const clean = DOMPurify.sanitize(dirty, {
  ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br', 'ul', 'ol', 'li'],
  ALLOWED_ATTR: ['href', 'target', 'rel'],
  ALLOW_DATA_ATTR: false,
});

// Remove all HTML (text only)
const textOnly = DOMPurify.sanitize(dirty, { ALLOWED_TAGS: [] });

// Hook to modify sanitization
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
});

// 4. URL sanitization
function sanitizeUrl(url) {
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
      return '#';
    }
    return parsed.href;
  } catch {
    return '#';
  }
}

// BAD: javascript: protocol XSS
// <a href="javascript:alert('xss')">Click</a>

// GOOD: Validate URL protocol
<a href={sanitizeUrl(userProvidedUrl)}>Link</a>

// 5. Template literal injection
// BAD
const html = '<div class="' + userInput + '">' + content + '</div>';

// GOOD: use tagged template literals or framework escaping
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
` + "```" + `

**Content Security Policy (CSP):**
` + "```" + `
HTTP Header:
  Content-Security-Policy: 
    default-src 'self';
    script-src 'self' 'nonce-abc123';
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https:;
    font-src 'self' https://fonts.googleapis.com;
    connect-src 'self' https://api.example.com;
    frame-src 'none';
    object-src 'none';
    base-uri 'self';
    form-action 'self';
    upgrade-insecure-requests;

Meta tag (limited - no frame-ancestors or report-uri):
  <meta http-equiv="Content-Security-Policy" 
    content="default-src 'self'; script-src 'self'">

Nonce-based CSP (recommended for inline scripts):
  Content-Security-Policy: script-src 'nonce-randomValue123'
  
  <script nonce="randomValue123">
    // This inline script is allowed
  </script>

Report-only mode (for testing):
  Content-Security-Policy-Report-Only: 
    default-src 'self';
    report-uri /csp-report;

Directives:
  default-src:    Fallback for all resource types
  script-src:     JavaScript sources
  style-src:      CSS sources
  img-src:        Image sources
  font-src:       Font sources
  connect-src:    Fetch/XHR/WebSocket sources
  frame-src:      iframe sources
  media-src:      Audio/video sources
  object-src:     Plugin sources (Flash, Java)
  base-uri:       Restrict <base> element
  form-action:    Restrict form submission targets
  frame-ancestors: Which pages can embed this page (replaces X-Frame-Options)

Source values:
  'self':           Same origin
  'none':           Block all
  'unsafe-inline':  Allow inline scripts/styles (avoid!)
  'unsafe-eval':    Allow eval() (avoid!)
  'nonce-value':    Allow specific inline with nonce
  'strict-dynamic': Trust scripts loaded by trusted scripts
  https::           Any HTTPS source
  data::            Data URIs
  *.example.com:    Any subdomain of example.com
` + "```" + ``,
					CodeExamples: `// Frontend security patterns

// 1. CSRF Protection
// Server sets CSRF token in cookie and response
// Client includes token in requests

function getCsrfToken() {
  // From meta tag
  return document.querySelector('meta[name="csrf-token"]')?.content;
  // Or from cookie
  // return document.cookie.match(/csrf=([^;]+)/)?.[1];
}

// Axios interceptor for CSRF token
axios.interceptors.request.use((config) => {
  const token = getCsrfToken();
  if (token) {
    config.headers['X-CSRF-Token'] = token;
  }
  return config;
});

// 2. Secure cookie handling
// Server should set secure cookie attributes:
// Set-Cookie: session=abc123; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=3600

// JavaScript cannot access HttpOnly cookies (prevents XSS cookie theft)
// SameSite=Strict prevents CSRF by not sending cookie in cross-site requests

// 3. Subresource Integrity (SRI)
// Verify CDN resources haven't been tampered with
// <script src="https://cdn.example.com/lib.js"
//   integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC"
//   crossorigin="anonymous">
// </script>

// Generate SRI hash:
// openssl dgst -sha384 -binary lib.js | openssl base64 -A

// 4. Secure headers configuration (Express example)
const helmet = require('helmet');

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", (req, res) => "'nonce-" + res.locals.nonce + "'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https://api.example.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      objectSrc: ["'none'"],
      frameSrc: ["'none'"],
      baseUri: ["'self'"],
      formAction: ["'self'"],
    },
  },
  crossOriginEmbedderPolicy: true,
  crossOriginOpenerPolicy: true,
  crossOriginResourcePolicy: { policy: 'same-site' },
  dnsPrefetchControl: true,
  frameguard: { action: 'deny' },
  hsts: { maxAge: 31536000, includeSubDomains: true, preload: true },
  noSniff: true,
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
  xssProtection: true,
}));

// 5. Secure JWT handling
// Store access token in memory (not localStorage)
let accessToken = null;

function setAccessToken(token) {
  accessToken = token;
  // Set expiry timer to refresh before expiration
  const payload = JSON.parse(atob(token.split('.')[1]));
  const expiresIn = payload.exp * 1000 - Date.now();
  setTimeout(refreshToken, expiresIn - 60000); // Refresh 1 min before
}

// Refresh token in httpOnly cookie (set by server)
async function refreshToken() {
  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include',  // Send httpOnly cookie
    });
    const { accessToken: newToken } = await res.json();
    setAccessToken(newToken);
  } catch (err) {
    // Redirect to login
    window.location.href = '/login';
  }
}

// 6. Secure postMessage handling
window.addEventListener('message', (event) => {
  // ALWAYS verify origin
  if (event.origin !== 'https://trusted-origin.com') return;
  
  // Validate message structure
  if (typeof event.data !== 'object') return;
  if (!['action1', 'action2'].includes(event.data.type)) return;
  
  // Process trusted message
  handleMessage(event.data);
});

// Sending messages - specify exact target origin
iframe.contentWindow.postMessage(
  { type: 'action', data: 'value' },
  'https://specific-origin.com' // NEVER use '*'
);`,
				},
				{
					Title: "Authentication Patterns and Secure Storage",
					Content: `Implement secure authentication flows including OAuth, JWT, session management, and credential storage.

**Authentication Flows:**
` + "```" + `javascript
// OAuth 2.0 / OIDC Authorization Code Flow with PKCE
// (Recommended for SPAs)

class AuthService {
  constructor(config) {
    this.config = config;
    this.accessToken = null;
    this.refreshTimeoutId = null;
  }

  // Step 1: Generate PKCE challenge
  async generatePKCE() {
    const verifier = this.randomString(128);
    const encoder = new TextEncoder();
    const data = encoder.encode(verifier);
    const hash = await crypto.subtle.digest('SHA-256', data);
    const challenge = this.base64UrlEncode(hash);
    
    sessionStorage.setItem('pkce_verifier', verifier);
    return { verifier, challenge };
  }

  randomString(length) {
    const array = crypto.getRandomValues(new Uint8Array(length));
    return Array.from(array, b => b.toString(36)).join('').slice(0, length);
  }

  base64UrlEncode(buffer) {
    return btoa(String.fromCharCode(...new Uint8Array(buffer)))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');
  }

  // Step 2: Redirect to authorization server
  async login() {
    const { challenge } = await this.generatePKCE();
    const state = this.randomString(32);
    sessionStorage.setItem('oauth_state', state);

    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.config.clientId,
      redirect_uri: this.config.redirectUri,
      scope: this.config.scope,
      state,
      code_challenge: challenge,
      code_challenge_method: 'S256',
    });

    window.location.href = 
      this.config.authorizationEndpoint + '?' + params.toString();
  }

  // Step 3: Handle callback
  async handleCallback(searchParams) {
    const code = searchParams.get('code');
    const state = searchParams.get('state');
    const savedState = sessionStorage.getItem('oauth_state');
    const verifier = sessionStorage.getItem('pkce_verifier');

    // Validate state to prevent CSRF
    if (state !== savedState) {
      throw new Error('Invalid state parameter');
    }

    // Exchange code for tokens
    const response = await fetch(this.config.tokenEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: this.config.clientId,
        code,
        redirect_uri: this.config.redirectUri,
        code_verifier: verifier,
      }),
    });

    const tokens = await response.json();
    
    // Store access token in memory only
    this.setAccessToken(tokens.access_token);
    
    // Refresh token handled via httpOnly cookie or stored securely
    
    // Clean up
    sessionStorage.removeItem('pkce_verifier');
    sessionStorage.removeItem('oauth_state');

    return tokens;
  }

  setAccessToken(token) {
    this.accessToken = token;
    // Schedule refresh
    const payload = JSON.parse(atob(token.split('.')[1]));
    const expiresIn = payload.exp * 1000 - Date.now();
    clearTimeout(this.refreshTimeoutId);
    this.refreshTimeoutId = setTimeout(
      () => this.refreshAccessToken(), 
      expiresIn - 60000
    );
  }

  async refreshAccessToken() {
    try {
      const response = await fetch(this.config.tokenEndpoint, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          grant_type: 'refresh_token',
          client_id: this.config.clientId,
        }),
      });
      const tokens = await response.json();
      this.setAccessToken(tokens.access_token);
    } catch {
      this.logout();
    }
  }

  getAccessToken() { return this.accessToken; }

  logout() {
    this.accessToken = null;
    clearTimeout(this.refreshTimeoutId);
    // Clear server-side session
    fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
    window.location.href = '/login';
  }
}

// React Auth Context
const AuthContext = React.createContext(null);

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const authService = useRef(new AuthService(config));

  useEffect(() => {
    // Try to restore session on mount
    authService.current.refreshAccessToken()
      .then(() => fetchUser())
      .catch(() => setLoading(false));
  }, []);

  async function fetchUser() {
    try {
      const token = authService.current.getAccessToken();
      const res = await fetch('/api/me', {
        headers: { Authorization: 'Bearer ' + token },
      });
      setUser(await res.json());
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }

  const value = {
    user,
    loading,
    login: () => authService.current.login(),
    logout: () => authService.current.logout(),
    getToken: () => authService.current.getAccessToken(),
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// Protected route
function ProtectedRoute({ children }) {
  const { user, loading } = useContext(AuthContext);
  const location = useLocation();

  if (loading) return <LoadingSpinner />;
  if (!user) return <Navigate to="/login" state={{ from: location }} />;
  return children;
}
` + "```" + ``,
					CodeExamples: `// Secure storage and input validation

// 1. Secure storage wrapper
class SecureStorage {
  // Use for non-sensitive data only
  // Never store tokens, passwords, or PII in localStorage
  
  static set(key, value) {
    try {
      const serialized = JSON.stringify(value);
      localStorage.setItem(key, serialized);
    } catch (err) {
      console.error('Storage error:', err);
    }
  }

  static get(key) {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch {
      return null;
    }
  }

  static remove(key) {
    localStorage.removeItem(key);
  }

  // Encrypted storage (Web Crypto API)
  static async encrypt(data, key) {
    const encoder = new TextEncoder();
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      key,
      encoder.encode(JSON.stringify(data))
    );
    return {
      iv: Array.from(iv),
      data: Array.from(new Uint8Array(encrypted)),
    };
  }

  static async decrypt(encrypted, key) {
    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: new Uint8Array(encrypted.iv) },
      key,
      new Uint8Array(encrypted.data)
    );
    const decoder = new TextDecoder();
    return JSON.parse(decoder.decode(decrypted));
  }

  static async deriveKey(password, salt) {
    const encoder = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      encoder.encode(password),
      'PBKDF2',
      false,
      ['deriveKey']
    );
    return crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    );
  }
}

// 2. Input validation (client-side)
// Always validate on server too!
import { z } from 'zod';

const signupSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Must contain an uppercase letter')
    .regex(/[0-9]/, 'Must contain a number')
    .regex(/[^A-Za-z0-9]/, 'Must contain a special character'),
  name: z.string()
    .min(2, 'Name must be at least 2 characters')
    .max(100, 'Name must be less than 100 characters')
    .regex(/^[a-zA-Z\s'-]+$/, 'Name contains invalid characters'),
  age: z.number().int().min(13).max(150).optional(),
});

function validateForm(data) {
  const result = signupSchema.safeParse(data);
  if (!result.success) {
    return {
      valid: false,
      errors: result.error.flatten().fieldErrors,
    };
  }
  return { valid: true, data: result.data };
}

// 3. Rate limiting on client side
class RateLimiter {
  constructor(maxRequests, windowMs) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.requests = [];
  }

  canMakeRequest() {
    const now = Date.now();
    this.requests = this.requests.filter(t => now - t < this.windowMs);
    if (this.requests.length >= this.maxRequests) return false;
    this.requests.push(now);
    return true;
  }
}

const apiLimiter = new RateLimiter(10, 60000); // 10 requests per minute

async function makeApiCall(url, options) {
  if (!apiLimiter.canMakeRequest()) {
    throw new Error('Too many requests. Please wait.');
  }
  return fetch(url, options);
}

// 4. Content Security - Trusted Types API
// Prevents DOM XSS by requiring typed objects for dangerous sinks
if (window.trustedTypes?.createPolicy) {
  const policy = trustedTypes.createPolicy('default', {
    createHTML: (input) => DOMPurify.sanitize(input),
    createScriptURL: (input) => {
      const url = new URL(input, document.baseURI);
      if (url.origin === location.origin) return url.href;
      throw new TypeError('Untrusted URL: ' + input);
    },
  });
}

// 5. Clickjacking prevention
// Server header: X-Frame-Options: DENY
// Or CSP: frame-ancestors 'none'

// Client-side framebusting (defense in depth)
if (window.top !== window.self) {
  window.top.location = window.self.location;
}`,
				},
			},
		},
		{
			ID:          1565,
			Title:       "Design Systems and Component Libraries",
			Description: "Build scalable design systems with component primitives, design tokens, theming, documentation, and multi-platform component architectures.",
			Order:       65,
			Lessons: []problems.Lesson{
				{
					Title: "Design Tokens and Theming Architecture",
					Content: `Design tokens are the visual design atoms of a design system - the values needed to construct and maintain a design system.

**Design Token Structure:**
` + "```" + `javascript
// tokens/colors.js
export const colors = {
  // Primitive tokens (raw values)
  blue: {
    50: '#eff6ff',
    100: '#dbeafe',
    200: '#bfdbfe',
    300: '#93c5fd',
    400: '#60a5fa',
    500: '#3b82f6',
    600: '#2563eb',
    700: '#1d4ed8',
    800: '#1e40af',
    900: '#1e3a8a',
  },
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  },
  // ... red, green, yellow, etc.
};

// Semantic tokens (purpose-driven, reference primitive tokens)
export const semanticColors = {
  light: {
    text: {
      primary: colors.gray[900],
      secondary: colors.gray[600],
      tertiary: colors.gray[400],
      inverse: '#ffffff',
      link: colors.blue[600],
      success: '#059669',
      error: '#dc2626',
      warning: '#d97706',
    },
    background: {
      primary: '#ffffff',
      secondary: colors.gray[50],
      tertiary: colors.gray[100],
      inverse: colors.gray[900],
      overlay: 'rgba(0, 0, 0, 0.5)',
    },
    border: {
      primary: colors.gray[200],
      secondary: colors.gray[100],
      focus: colors.blue[500],
      error: '#dc2626',
    },
    interactive: {
      primary: colors.blue[600],
      primaryHover: colors.blue[700],
      primaryActive: colors.blue[800],
      secondary: colors.gray[100],
      secondaryHover: colors.gray[200],
      disabled: colors.gray[100],
      disabledText: colors.gray[400],
    },
  },
  dark: {
    text: {
      primary: colors.gray[50],
      secondary: colors.gray[400],
      tertiary: colors.gray[500],
      inverse: colors.gray[900],
      link: colors.blue[400],
      success: '#34d399',
      error: '#f87171',
      warning: '#fbbf24',
    },
    background: {
      primary: colors.gray[900],
      secondary: colors.gray[800],
      tertiary: colors.gray[700],
      inverse: '#ffffff',
      overlay: 'rgba(0, 0, 0, 0.7)',
    },
    border: {
      primary: colors.gray[700],
      secondary: colors.gray[800],
      focus: colors.blue[400],
      error: '#f87171',
    },
    interactive: {
      primary: colors.blue[500],
      primaryHover: colors.blue[400],
      primaryActive: colors.blue[300],
      secondary: colors.gray[800],
      secondaryHover: colors.gray[700],
      disabled: colors.gray[800],
      disabledText: colors.gray[600],
    },
  },
};

// Spacing scale
export const spacing = {
  0: '0',
  0.5: '0.125rem',  // 2px
  1: '0.25rem',      // 4px
  1.5: '0.375rem',   // 6px
  2: '0.5rem',       // 8px
  3: '0.75rem',      // 12px
  4: '1rem',         // 16px
  5: '1.25rem',      // 20px
  6: '1.5rem',       // 24px
  8: '2rem',         // 32px
  10: '2.5rem',      // 40px
  12: '3rem',        // 48px
  16: '4rem',        // 64px
  20: '5rem',        // 80px
  24: '6rem',        // 96px
};

// Typography scale
export const typography = {
  fontFamily: {
    sans: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    mono: "'JetBrains Mono', 'Fira Code', monospace",
  },
  fontSize: {
    xs: ['0.75rem', { lineHeight: '1rem' }],
    sm: ['0.875rem', { lineHeight: '1.25rem' }],
    base: ['1rem', { lineHeight: '1.5rem' }],
    lg: ['1.125rem', { lineHeight: '1.75rem' }],
    xl: ['1.25rem', { lineHeight: '1.75rem' }],
    '2xl': ['1.5rem', { lineHeight: '2rem' }],
    '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
    '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
  },
  fontWeight: {
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
  },
};

// Shadows
export const shadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
};

// Breakpoints
export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
};
` + "```" + `

**CSS Variables Theming:**
` + "```" + `css
:root {
  /* Convert tokens to CSS custom properties */
  --color-text-primary: #111827;
  --color-text-secondary: #4b5563;
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f9fafb;
  --color-border-primary: #e5e7eb;
  --color-interactive-primary: #2563eb;
  --color-interactive-primary-hover: #1d4ed8;
  
  --spacing-1: 0.25rem;
  --spacing-2: 0.5rem;
  --spacing-4: 1rem;
  
  --radius-sm: 0.25rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-full: 9999px;
  
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

[data-theme="dark"] {
  --color-text-primary: #f9fafb;
  --color-text-secondary: #9ca3af;
  --color-bg-primary: #111827;
  --color-bg-secondary: #1f2937;
  --color-border-primary: #374151;
  --color-interactive-primary: #3b82f6;
  --color-interactive-primary-hover: #60a5fa;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --color-text-primary: #f9fafb;
    --color-text-secondary: #9ca3af;
    --color-bg-primary: #111827;
    --color-bg-secondary: #1f2937;
    --color-border-primary: #374151;
  }
}

/* Component using tokens */
.card {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-4);
  box-shadow: var(--shadow-sm);
  color: var(--color-text-primary);
}
` + "```" + ``,
					CodeExamples: `// Component library architecture

// 1. Polymorphic component pattern
function Box({ as: Component = 'div', className, style, ...props }) {
  return <Component className={className} style={style} {...props} />;
}

// Usage: <Box as="section" className="card">...</Box>
// Usage: <Box as={Link} to="/about">Go to about</Box>

// 2. Compound component pattern for complex components
const SelectContext = createContext(null);

function Select({ value, onChange, children }) {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef(null);

  return (
    <SelectContext.Provider value={{ value, onChange, isOpen, setIsOpen, triggerRef }}>
      <div className="select-root" style={{ position: 'relative' }}>
        {children}
      </div>
    </SelectContext.Provider>
  );
}

Select.Trigger = function SelectTrigger({ children, placeholder }) {
  const { value, isOpen, setIsOpen, triggerRef } = useContext(SelectContext);
  return (
    <button
      ref={triggerRef}
      role="combobox"
      aria-expanded={isOpen}
      aria-haspopup="listbox"
      onClick={() => setIsOpen(!isOpen)}
      className="select-trigger"
    >
      {value || placeholder}
      <ChevronDown />
    </button>
  );
};

Select.Content = function SelectContent({ children }) {
  const { isOpen } = useContext(SelectContext);
  if (!isOpen) return null;
  return (
    <ul role="listbox" className="select-content">
      {children}
    </ul>
  );
};

Select.Item = function SelectItem({ value, children }) {
  const { value: selected, onChange, setIsOpen } = useContext(SelectContext);
  return (
    <li
      role="option"
      aria-selected={value === selected}
      onClick={() => { onChange(value); setIsOpen(false); }}
      className="select-item"
    >
      {children}
      {value === selected && <Check />}
    </li>
  );
};

// Usage
// <Select value={country} onChange={setCountry}>
//   <Select.Trigger placeholder="Select country" />
//   <Select.Content>
//     <Select.Item value="us">United States</Select.Item>
//     <Select.Item value="uk">United Kingdom</Select.Item>
//   </Select.Content>
// </Select>

// 3. Theme provider with CSS variables
function ThemeProvider({ theme = 'light', children }) {
  const tokens = theme === 'dark' ? semanticColors.dark : semanticColors.light;

  const cssVars = useMemo(() => {
    const vars = {};
    function flatten(obj, prefix = '') {
      for (const [key, value] of Object.entries(obj)) {
        const varName = prefix ? prefix + '-' + key : key;
        if (typeof value === 'object') {
          flatten(value, varName);
        } else {
          vars['--color-' + varName] = value;
        }
      }
    }
    flatten(tokens);
    return vars;
  }, [tokens]);

  return (
    <div style={cssVars} data-theme={theme}>
      {children}
    </div>
  );
}

// 4. Responsive variant system (like Stitches/CVA)
import { cva } from 'class-variance-authority';

const button = cva(
  'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2',
  {
    variants: {
      variant: {
        primary: 'bg-blue-600 text-white hover:bg-blue-700',
        secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200',
        ghost: 'hover:bg-gray-100',
        destructive: 'bg-red-600 text-white hover:bg-red-700',
      },
      size: {
        sm: 'h-8 px-3 text-sm',
        md: 'h-10 px-4 text-sm',
        lg: 'h-12 px-6 text-base',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
);

function Button({ variant, size, className, ...props }) {
  return (
    <button className={button({ variant, size, className })} {...props} />
  );
}`,
				},
			},
		},
	})
}
