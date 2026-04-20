package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2218,
			Title:       "Python Web Development and APIs",
			Description: "Master Flask, FastAPI, Django, REST API design, authentication, database integration, and web application patterns.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Flask FastAPI Django REST APIs and Web Patterns",
					Content: `Python is one of the most popular languages for web development, with mature frameworks for every use case.

**Flask — Micro Framework:**

Core Concepts:
  WSGI application
  Minimal core, extensible with blueprints/extensions
  Jinja2 templating
  Werkzeug WSGI toolkit
  
Application Factory Pattern:
  Create app in a function (not module level)
  Register blueprints, extensions, config
  Enables testing with different configs
  
Routing:
  @app.route('/path') — decorator-based
  @app.route('/user/<int:id>') — URL parameters
  Methods: GET, POST, PUT, DELETE, PATCH
  url_for('function_name') — URL generation
  
Request/Response:
  request.args — query parameters
  request.form — form data  
  request.json — JSON body
  request.files — uploaded files
  request.headers — HTTP headers
  request.method — HTTP method
  
  return jsonify(data), status_code
  make_response(data, status_code)
  redirect(url_for('function'))
  abort(404)

Blueprints:
  Modular application components
  Own routes, templates, static files
  Registered with app.register_blueprint()
  URL prefix per blueprint

Extensions:
  Flask-SQLAlchemy: ORM integration
  Flask-Migrate: Database migrations (Alembic)
  Flask-Login: User sessions
  Flask-WTF: Form validation
  Flask-CORS: Cross-origin requests
  Flask-Caching: Response caching
  Flask-Limiter: Rate limiting
  Flask-Mail: Email sending

Error Handling:
  @app.errorhandler(404) — custom error pages
  @app.errorhandler(Exception) — catch-all
  abort(status_code, description)

Middleware:
  @app.before_request — before each request
  @app.after_request — after each request  
  @app.teardown_request — cleanup
  @app.before_first_request — one-time setup

Context:
  Application context: current_app, g
  Request context: request, session
  
  g: Per-request global (store db connection, user, etc.)
  session: Signed cookie-based session

**FastAPI — Modern Async Framework:**

Core Concepts:
  ASGI application (async support)
  Built on Starlette + Pydantic
  Automatic OpenAPI/Swagger docs
  Type hint-driven validation
  
Routing:
  @app.get('/path')
  @app.post('/path')
  @app.put('/path/{id}')
  @app.delete('/path/{id}')
  
  Path parameters: automatic type conversion
  Query parameters: optional with defaults
  Request body: Pydantic models

Pydantic Models:
  Data validation and serialization
  BaseModel subclass
  Field types with constraints
  field_validator for custom validation
  model_validator for cross-field validation
  Automatic JSON Schema generation

Dependency Injection:
  Depends() — declare dependencies
  Sub-dependencies (nested)
  Yield dependencies (with cleanup)
  Background tasks
  
  Use cases:
    Database sessions
    Authentication
    Rate limiting
    Logging
    Caching

Security:
  OAuth2PasswordBearer — token auth
  HTTPBearer — Bearer token
  HTTPBasic — Basic auth
  API key (header, query, cookie)
  
  OAuth2 with JWT flow:
    Login → validate → create JWT → return token
    Subsequent requests → verify JWT → get user

Async Support:
  async def endpoint — async handler
  await for async operations
  BackgroundTasks — post-response processing
  Streaming responses
  WebSocket support

Middleware:
  @app.middleware("http") — custom middleware
  CORSMiddleware — CORS configuration
  TrustedHostMiddleware — host validation
  GZipMiddleware — response compression

Response Models:
  response_model=Model — output schema
  response_model_exclude — hide fields
  status_code=201 — response status
  Automatic serialization

**Django — Full-Stack Framework:**

Core Concepts:
  MTV pattern (Model-Template-View)
  Batteries-included
  ORM, admin, auth, forms, template engine
  
Project Structure:
  manage.py — CLI tool
  settings.py — configuration
  urls.py — URL routing
  apps/ — Django applications
  
Models (ORM):
  class MyModel(models.Model)
  CharField, IntegerField, DateTimeField, etc.
  ForeignKey, ManyToManyField, OneToOneField
  Meta class: ordering, constraints, indexes
  
  Managers: Model.objects (default manager)
  QuerySets: filter(), exclude(), annotate(), aggregate()
  Q objects: complex queries (OR, AND, NOT)
  F objects: field references
  
  Migrations:
    makemigrations — generate migration files
    migrate — apply migrations
    showmigrations — list migrations

Views:
  Function-based views (FBV)
  Class-based views (CBV)
    ListView, DetailView, CreateView, UpdateView, DeleteView
    TemplateView, RedirectView
    Mixins: LoginRequiredMixin, etc.
  
  Django REST Framework (DRF):
    Serializers: ModelSerializer, HyperlinkedModelSerializer
    ViewSets: ModelViewSet, ReadOnlyModelViewSet
    Routers: DefaultRouter (auto URL generation)
    Permissions: IsAuthenticated, IsAdminUser, custom
    Throttling: AnonRateThrottle, UserRateThrottle
    Filtering: DjangoFilterBackend, SearchFilter, OrderingFilter
    Pagination: PageNumberPagination, LimitOffsetPagination
    Versioning: URLPathVersioning, AcceptHeaderVersioning

Admin:
  Automatic admin interface
  ModelAdmin customization
  List display, filters, search
  Inline editing
  Actions (bulk operations)

Authentication:
  Built-in User model
  Custom User model (AbstractUser, AbstractBaseUser)
  Login, logout, password reset
  Groups and permissions
  Social auth (django-allauth)
  JWT (djangorestframework-simplejwt)

Middleware:
  SecurityMiddleware
  SessionMiddleware
  CommonMiddleware
  CsrfViewMiddleware
  AuthenticationMiddleware
  Custom middleware classes

Signals:
  pre_save, post_save
  pre_delete, post_delete
  request_started, request_finished
  
Caching:
  Per-view caching
  Template fragment caching
  Low-level cache API
  Backends: Memcached, Redis, database, file

Channels:
  WebSocket support
  ASGI deployment
  Channel layers (Redis)

**REST API Design Best Practices:**

Resources:
  Nouns, not verbs: /users, not /getUsers
  Plural: /users, not /user
  Nested: /users/123/orders
  
HTTP Methods:
  GET: Read (safe, idempotent)
  POST: Create
  PUT: Full update (idempotent)
  PATCH: Partial update
  DELETE: Remove (idempotent)
  
Status Codes:
  200 OK, 201 Created, 204 No Content
  301 Moved, 304 Not Modified
  400 Bad Request, 401 Unauthorized, 403 Forbidden
  404 Not Found, 409 Conflict, 422 Unprocessable
  429 Too Many Requests
  500 Internal Server Error, 503 Service Unavailable
  
Pagination:
  Cursor-based (recommended for large datasets)
  Offset-based (page + per_page)
  Link headers (RFC 5988)
  
Filtering:
  Query parameters: ?status=active&sort=-created
  Dedicated filter syntax: ?filter[status]=active
  
Versioning:
  URL path: /api/v1/users
  Accept header: Accept: application/vnd.api.v1+json
  Custom header: X-API-Version: 1
  
Error Format:
  Consistent error response structure
  Error code, message, details
  Validation errors per field

**Authentication Patterns:**

Session-based:
  Server stores session state
  Cookie-based session ID
  Good for server-rendered apps
  
Token-based (JWT):
  Stateless authentication
  Access token (short-lived) + Refresh token (long-lived)
  Stored in HTTP-only cookie or Authorization header
  
API Key:
  Simple authentication
  Header: X-API-Key or query parameter
  Good for service-to-service
  
OAuth 2.0:
  Authorization Code flow (web apps)
  Client Credentials flow (service-to-service)
  PKCE extension (mobile/SPA)
  
**Database Integration:**

SQLAlchemy:
  Python SQL toolkit + ORM
  Engine: Connection pool
  Session: Unit of work pattern
  Models: Declarative base
  Query: Chainable API
  Alembic: Database migrations
  
  Async: AsyncSession, create_async_engine

Tortoise ORM:
  Async ORM for Python
  Django-inspired API
  Works with asyncio frameworks

MongoDB:
  Motor: Async MongoDB driver
  PyMongo: Sync driver
  MongoEngine: ODM (Object-Document Mapper)
  Beanie: Async ODM with Pydantic

Redis:
  redis-py: Sync client
  aioredis: Async client
  Use: Caching, sessions, pub/sub, queues`,
					CodeExamples: `# Python web development examples

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import hashlib
import hmac
import json
import secrets
import time
from enum import Enum

# ============================================================
# Flask-style Application
# ============================================================

class Route:
    """Route registration for WSGI app."""
    
    def __init__(self, path: str, methods: list, handler, name: str = None):
        self.path = path
        self.methods = methods
        self.handler = handler
        self.name = name or handler.__name__
        self._param_names = self._extract_params(path)
    
    def _extract_params(self, path: str) -> list:
        params = []
        for part in path.split('/'):
            if part.startswith('<') and part.endswith('>'):
                params.append(part[1:-1])
        return params
    
    def match(self, path: str, method: str) -> Optional[dict]:
        if method not in self.methods:
            return None
        
        route_parts = self.path.split('/')
        path_parts = path.split('/')
        
        if len(route_parts) != len(path_parts):
            return None
        
        params = {}
        for rp, pp in zip(route_parts, path_parts):
            if rp.startswith('<') and rp.endswith('>'):
                param_name = rp[1:-1]
                if ':' in param_name:
                    type_name, param_name = param_name.split(':')
                    if type_name == 'int':
                        try:
                            pp = int(pp)
                        except ValueError:
                            return None
                params[param_name] = pp
            elif rp != pp:
                return None
        
        return params


class Request:
    """HTTP request wrapper."""
    
    def __init__(self, method: str, path: str, headers: dict = None,
                 query_params: dict = None, body: bytes = None):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.body = body
        self._json = None
    
    @property
    def json(self) -> dict:
        if self._json is None and self.body:
            self._json = json.loads(self.body)
        return self._json
    
    @property
    def content_type(self) -> str:
        return self.headers.get('Content-Type', '')
    
    @property
    def authorization(self) -> Optional[str]:
        auth = self.headers.get('Authorization', '')
        if auth.startswith('Bearer '):
            return auth[7:]
        return None


class Response:
    """HTTP response wrapper."""
    
    def __init__(self, body: Any = None, status_code: int = 200,
                 headers: dict = None, content_type: str = 'application/json'):
        self.status_code = status_code
        self.headers = headers or {}
        self.headers['Content-Type'] = content_type
        
        if isinstance(body, (dict, list)):
            self.body = json.dumps(body)
        elif body is None:
            self.body = ''
        else:
            self.body = str(body)


class MicroFramework:
    """Minimal web framework (Flask-like)."""
    
    def __init__(self, name: str):
        self.name = name
        self._routes: List[Route] = []
        self._middlewares = []
        self._error_handlers = {}
        self._before_request = []
        self._after_request = []
    
    def route(self, path: str, methods: list = None):
        def decorator(handler):
            route = Route(path, methods or ['GET'], handler)
            self._routes.append(route)
            return handler
        return decorator
    
    def get(self, path: str):
        return self.route(path, ['GET'])
    
    def post(self, path: str):
        return self.route(path, ['POST'])
    
    def put(self, path: str):
        return self.route(path, ['PUT'])
    
    def delete(self, path: str):
        return self.route(path, ['DELETE'])
    
    def before_request(self, func):
        self._before_request.append(func)
        return func
    
    def after_request(self, func):
        self._after_request.append(func)
        return func
    
    def errorhandler(self, status_code: int):
        def decorator(handler):
            self._error_handlers[status_code] = handler
            return handler
        return decorator
    
    def handle_request(self, request: Request) -> Response:
        # Run before_request hooks
        for hook in self._before_request:
            result = hook(request)
            if result is not None:
                return result
        
        # Find matching route
        for route in self._routes:
            params = route.match(request.path, request.method)
            if params is not None:
                try:
                    result = route.handler(request, **params)
                    if isinstance(result, Response):
                        response = result
                    elif isinstance(result, tuple):
                        response = Response(body=result[0], status_code=result[1])
                    else:
                        response = Response(body=result)
                except Exception as e:
                    response = Response(
                        body={"error": str(e)},
                        status_code=500
                    )
                
                # Run after_request hooks
                for hook in self._after_request:
                    response = hook(request, response) or response
                
                return response
        
        # No route found
        if 404 in self._error_handlers:
            return self._error_handlers[404](request)
        return Response(body={"error": "Not Found"}, status_code=404)


class Blueprint:
    """Modular route collection (Flask Blueprint-like)."""
    
    def __init__(self, name: str, url_prefix: str = ''):
        self.name = name
        self.url_prefix = url_prefix
        self._routes: List[Route] = []
    
    def route(self, path: str, methods: list = None):
        def decorator(handler):
            full_path = self.url_prefix + path
            route = Route(full_path, methods or ['GET'], handler)
            self._routes.append(route)
            return handler
        return decorator
    
    def register(self, app: MicroFramework):
        app._routes.extend(self._routes)


# ============================================================
# REST API Components
# ============================================================

class APIResource:
    """RESTful resource handler."""
    
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, dict] = {}
        self._next_id = 1
    
    def list(self, page: int = 1, per_page: int = 20,
             filters: dict = None, sort_by: str = None) -> dict:
        items = list(self._items.values())
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                items = [i for i in items if i.get(key) == value]
        
        # Sort
        if sort_by:
            reverse = sort_by.startswith('-')
            field = sort_by.lstrip('-')
            items.sort(key=lambda x: x.get(field, ''), reverse=reverse)
        
        # Paginate
        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        items = items[start:end]
        
        return {
            "data": items,
            "meta": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        }
    
    def get(self, item_id: str) -> Optional[dict]:
        return self._items.get(item_id)
    
    def create(self, data: dict) -> dict:
        item_id = str(self._next_id)
        self._next_id += 1
        
        data['id'] = item_id
        data['created_at'] = datetime.utcnow().isoformat()
        data['updated_at'] = data['created_at']
        
        self._items[item_id] = data
        return data
    
    def update(self, item_id: str, data: dict) -> Optional[dict]:
        if item_id not in self._items:
            return None
        
        item = self._items[item_id]
        item.update(data)
        item['updated_at'] = datetime.utcnow().isoformat()
        return item
    
    def delete(self, item_id: str) -> bool:
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False


# ============================================================
# Authentication
# ============================================================

class JWTAuth:
    """Simple JWT authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256',
                 access_ttl: int = 3600, refresh_ttl: int = 86400 * 7):
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_ttl = access_ttl
        self._refresh_ttl = refresh_ttl
        self._revoked_tokens = set()
    
    def create_access_token(self, user_id: str, claims: dict = None) -> str:
        payload = {
            'sub': user_id,
            'iat': int(time.time()),
            'exp': int(time.time()) + self._access_ttl,
            'type': 'access',
            'jti': secrets.token_hex(16),
        }
        if claims:
            payload.update(claims)
        return self._encode(payload)
    
    def create_refresh_token(self, user_id: str) -> str:
        payload = {
            'sub': user_id,
            'iat': int(time.time()),
            'exp': int(time.time()) + self._refresh_ttl,
            'type': 'refresh',
            'jti': secrets.token_hex(16),
        }
        return self._encode(payload)
    
    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = self._decode(token)
            
            if payload.get('jti') in self._revoked_tokens:
                return None
            
            if payload.get('exp', 0) < time.time():
                return None
            
            return payload
        except Exception:
            return None
    
    def revoke_token(self, token: str):
        payload = self._decode(token)
        if payload and 'jti' in payload:
            self._revoked_tokens.add(payload['jti'])
    
    def _encode(self, payload: dict) -> str:
        header = json.dumps({"alg": self._algorithm, "typ": "JWT"})
        payload_json = json.dumps(payload)
        
        import base64
        h = base64.urlsafe_b64encode(header.encode()).rstrip(b'=').decode()
        p = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b'=').decode()
        
        signing_input = f"{h}.{p}"
        signature = hmac.new(
            self._secret_key.encode(),
            signing_input.encode(),
            hashlib.sha256
        ).digest()
        s = base64.urlsafe_b64encode(signature).rstrip(b'=').decode()
        
        return f"{h}.{p}.{s}"
    
    def _decode(self, token: str) -> Optional[dict]:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        import base64
        # Verify signature
        signing_input = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(
            self._secret_key.encode(),
            signing_input.encode(),
            hashlib.sha256
        ).digest()
        
        actual_sig = base64.urlsafe_b64decode(parts[2] + '==')
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        
        payload = base64.urlsafe_b64decode(parts[1] + '==')
        return json.loads(payload)


class PasswordHasher:
    """Secure password hashing."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        hash_val = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            iterations=100000
        ).hex()
        return f"{salt}${hash_val}"
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        salt, hash_val = hashed.split('$')
        test_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            iterations=100000
        ).hex()
        return hmac.compare_digest(test_hash, hash_val)


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int):
        self._rate = rate
        self._burst = burst
        self._buckets: Dict[str, dict] = {}
    
    def allow(self, key: str) -> bool:
        now = time.time()
        
        if key not in self._buckets:
            self._buckets[key] = {
                'tokens': self._burst - 1,
                'last_refill': now
            }
            return True
        
        bucket = self._buckets[key]
        elapsed = now - bucket['last_refill']
        bucket['tokens'] = min(
            self._burst,
            bucket['tokens'] + elapsed * self._rate
        )
        bucket['last_refill'] = now
        
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    def remaining(self, key: str) -> int:
        if key not in self._buckets:
            return self._burst
        return int(self._buckets[key]['tokens'])


# ============================================================
# ORM-like Database Layer
# ============================================================

class QuerySet:
    """Django-like queryset for filtering."""
    
    def __init__(self, items: list):
        self._items = items
        self._filters = []
        self._order_by = None
        self._limit = None
        self._offset = None
    
    def filter(self, **kwargs) -> 'QuerySet':
        qs = QuerySet(self._items)
        qs._filters = self._filters + [kwargs]
        qs._order_by = self._order_by
        qs._limit = self._limit
        qs._offset = self._offset
        return qs
    
    def exclude(self, **kwargs) -> 'QuerySet':
        qs = QuerySet(self._items)
        qs._filters = self._filters + [{'__exclude__': kwargs}]
        qs._order_by = self._order_by
        return qs
    
    def order_by(self, field: str) -> 'QuerySet':
        qs = QuerySet(self._items)
        qs._filters = self._filters
        qs._order_by = field
        return qs
    
    def limit(self, n: int) -> 'QuerySet':
        qs = QuerySet(self._items)
        qs._filters = self._filters
        qs._order_by = self._order_by
        qs._limit = n
        qs._offset = self._offset
        return qs
    
    def offset(self, n: int) -> 'QuerySet':
        qs = QuerySet(self._items)
        qs._filters = self._filters
        qs._order_by = self._order_by
        qs._limit = self._limit
        qs._offset = n
        return qs
    
    def _apply_filters(self) -> list:
        result = self._items[:]
        
        for f in self._filters:
            if '__exclude__' in f:
                exclude = f['__exclude__']
                result = [
                    item for item in result
                    if not all(item.get(k) == v for k, v in exclude.items())
                ]
            else:
                filtered = []
                for item in result:
                    match = True
                    for k, v in f.items():
                        if '__' in k:
                            field, op = k.rsplit('__', 1)
                            item_val = item.get(field)
                            if op == 'gt' and not (item_val and item_val > v):
                                match = False
                            elif op == 'lt' and not (item_val and item_val < v):
                                match = False
                            elif op == 'gte' and not (item_val and item_val >= v):
                                match = False
                            elif op == 'lte' and not (item_val and item_val <= v):
                                match = False
                            elif op == 'contains' and not (item_val and v in item_val):
                                match = False
                            elif op == 'in' and item_val not in v:
                                match = False
                        elif item.get(k) != v:
                            match = False
                    if match:
                        filtered.append(item)
                result = filtered
        
        return result
    
    def all(self) -> list:
        result = self._apply_filters()
        
        if self._order_by:
            reverse = self._order_by.startswith('-')
            field = self._order_by.lstrip('-')
            result.sort(key=lambda x: x.get(field, ''), reverse=reverse)
        
        if self._offset:
            result = result[self._offset:]
        
        if self._limit:
            result = result[:self._limit]
        
        return result
    
    def first(self):
        result = self.all()
        return result[0] if result else None
    
    def count(self) -> int:
        return len(self._apply_filters())
    
    def exists(self) -> bool:
        return self.count() > 0
    
    def values(self, *fields) -> list:
        result = self.all()
        if fields:
            return [{k: item.get(k) for k in fields} for item in result]
        return result


class ModelManager:
    """Simple ORM-like model manager."""
    
    def __init__(self):
        self._storage: Dict[str, dict] = {}
        self._next_id = 1
        self._schema: Dict[str, type] = {}
    
    def define_field(self, name: str, field_type: type, required: bool = True):
        self._schema[name] = field_type
    
    def create(self, **kwargs) -> dict:
        obj = {'id': self._next_id}
        self._next_id += 1
        
        for name, field_type in self._schema.items():
            if name in kwargs:
                obj[name] = kwargs[name]
        
        obj['created_at'] = datetime.utcnow().isoformat()
        obj['updated_at'] = obj['created_at']
        
        self._storage[str(obj['id'])] = obj
        return obj
    
    def get(self, pk) -> Optional[dict]:
        return self._storage.get(str(pk))
    
    def filter(self, **kwargs) -> QuerySet:
        return QuerySet(list(self._storage.values())).filter(**kwargs)
    
    def all(self) -> QuerySet:
        return QuerySet(list(self._storage.values()))
    
    def update(self, pk, **kwargs) -> Optional[dict]:
        obj = self._storage.get(str(pk))
        if obj:
            obj.update(kwargs)
            obj['updated_at'] = datetime.utcnow().isoformat()
        return obj
    
    def delete(self, pk) -> bool:
        return self._storage.pop(str(pk), None) is not None
    
    def count(self) -> int:
        return len(self._storage)


# ============================================================
# Middleware Chain
# ============================================================

class MiddlewareChain:
    """WSGI-style middleware chain."""
    
    def __init__(self):
        self._middlewares = []
    
    def add(self, middleware):
        self._middlewares.append(middleware)
    
    def execute(self, request: Request) -> Response:
        def make_handler(index):
            if index >= len(self._middlewares):
                return lambda req: Response(body={"error": "No handler"}, status_code=404)
            
            middleware = self._middlewares[index]
            next_handler = make_handler(index + 1)
            return lambda req: middleware(req, next_handler)
        
        handler = make_handler(0)
        return handler(request)


def cors_middleware(request: Request, next_handler):
    """CORS middleware."""
    response = next_handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    if request.method == 'OPTIONS':
        return Response(status_code=204)
    
    return response


def logging_middleware(request: Request, next_handler):
    """Request logging middleware."""
    start = time.time()
    response = next_handler(request)
    duration = time.time() - start
    
    print(f"{request.method} {request.path} -> {response.status_code} ({duration:.3f}s)")
    return response


def auth_middleware(jwt_auth: JWTAuth):
    """Authentication middleware factory."""
    def middleware(request: Request, next_handler):
        # Skip auth for public endpoints
        public_paths = ['/api/login', '/api/register', '/api/health']
        if request.path in public_paths:
            return next_handler(request)
        
        token = request.authorization
        if not token:
            return Response(
                body={"error": "Authentication required"},
                status_code=401
            )
        
        payload = jwt_auth.verify_token(token)
        if not payload:
            return Response(
                body={"error": "Invalid or expired token"},
                status_code=401
            )
        
        # Attach user info to request
        request.user_id = payload.get('sub')
        return next_handler(request)
    
    return middleware`,
				},
			},
		},
	})
}
