package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2215,
			Title:       "Advanced Python Patterns",
			Description: "Master advanced Python patterns including metaclasses, descriptors, concurrency, type system, and production-grade Python techniques.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Metaclasses and Class Creation",
					Content: `In Python, classes are objects too. A metaclass is the "class of a class" — it controls how classes are created and behaves as a factory for classes.

**The Class Creation Chain:**
- ` + "`" + `type` + "`" + ` is the default metaclass
- When you write ` + "`" + `class Foo:` + "`" + `, Python calls ` + "`" + `type('Foo', (object,), {...})` + "`" + `
- Custom metaclasses let you intercept and modify class creation

**When to Use Metaclasses:**
- Framework/library authors (Django ORM, SQLAlchemy)
- Enforcing coding conventions across classes
- Automatic registration of classes (plugin systems)
- Abstract base classes with enforcement

**Tim Peters' Rule:** "Metaclasses are deeper magic than 99% of users should ever worry about. If you wonder whether you need them, you don't."

**The ` + "`" + `__init_subclass__` + "`" + ` Alternative (Python 3.6+):**
For most metaclass use cases, ` + "`" + `__init_subclass__` + "`" + ` is simpler and sufficient. It's called when a class is subclassed.

**Practical Uses:**
1. **Singleton Pattern**: Ensure only one instance exists
2. **Auto-registration**: Classes register themselves on creation
3. **Validation**: Ensure subclasses have required attributes/methods
4. **ORM Field Collection**: Gather field definitions from class body (like Django models)`,
					CodeExamples: `# Basic metaclass
class Meta(type):
    def __new__(cls, name, bases, namespace):
        # Called when class is created
        print(f"Creating class: {name}")
        # Enforce convention: all methods must have docstrings
        for key, value in namespace.items():
            if callable(value) and not key.startswith('_'):
                if not value.__doc__:
                    raise TypeError(f"Method {key} in {name} must have a docstring")
        return super().__new__(cls, name, bases, namespace)

class MyService(metaclass=Meta):
    def process(self):
        """Process data."""  # Required!
        pass

# __init_subclass__ — simpler alternative (Python 3.6+)
class Plugin:
    registry = {}

    def __init_subclass__(cls, plugin_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = plugin_name or cls.__name__.lower()
        Plugin.registry[name] = cls
        print(f"Registered plugin: {name}")

class JSONPlugin(Plugin, plugin_name="json"):
    def process(self, data):
        return json.dumps(data)

class XMLPlugin(Plugin, plugin_name="xml"):
    def process(self, data):
        return to_xml(data)

# Plugin.registry == {"json": JSONPlugin, "xml": XMLPlugin}
# Plugins register themselves just by being defined!

# Singleton using metaclass
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = create_connection()

# Always returns same instance
db1 = Database()
db2 = Database()
assert db1 is db2  # True`,
				},
				{
					Title: "Descriptors and Properties Deep Dive",
					Content: `Descriptors are the mechanism behind Python's properties, methods, static methods, class methods, and slots. Understanding them unlocks Python's data model.

**What is a Descriptor?**
Any object that defines ` + "`" + `__get__` + "`" + `, ` + "`" + `__set__` + "`" + `, or ` + "`" + `__delete__` + "`" + ` methods. When accessed as a class attribute, descriptors intercept attribute access.

**Types:**
- **Data Descriptor**: Defines both ` + "`" + `__get__` + "`" + ` and ` + "`" + `__set__` + "`" + ` (higher priority than instance dict)
- **Non-Data Descriptor**: Defines only ` + "`" + `__get__` + "`" + ` (lower priority than instance dict)

**Attribute Lookup Order:**
1. Data descriptors on the class
2. Instance ` + "`" + `__dict__` + "`" + `
3. Non-data descriptors on the class

**Practical Uses:**
- **Validated attributes**: Type checking, range validation
- **Lazy computation**: Compute on first access, cache result
- **ORM fields**: Define database column properties (SQLAlchemy, Django)
- **Logging**: Track attribute access and changes

**` + "`" + `@property` + "`" + ` is a Descriptor:**
` + "`" + `@property` + "`" + ` is just a convenient way to create a data descriptor. Under the hood, it uses ` + "`" + `__get__` + "`" + `, ` + "`" + `__set__` + "`" + `, and ` + "`" + `__delete__` + "`" + `.

**` + "`" + `__slots__` + "`" + `:**
Normally, Python objects store attributes in a ` + "`" + `__dict__` + "`" + ` (a dictionary). ` + "`" + `__slots__` + "`" + ` uses descriptors to store attributes in a fixed-size array instead:
- **Memory savings**: 40-50% less memory per instance
- **Speed**: Slightly faster attribute access
- **Restriction**: Can't add arbitrary attributes
- **Use when**: Creating millions of instances of the same class`,
					CodeExamples: `# Custom descriptor for validated attributes
class Validated:
    def __init__(self, validator, default=None):
        self.validator = validator
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name  # Python 3.6+

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)

    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}: {value}")
        obj.__dict__[self.name] = value

# Usage: reusable validation
class User:
    name = Validated(lambda x: isinstance(x, str) and len(x) > 0)
    age = Validated(lambda x: isinstance(x, int) and 0 < x < 150)
    email = Validated(lambda x: isinstance(x, str) and "@" in x)

user = User()
user.name = "Alice"   # OK
user.age = 30         # OK
# user.age = -5       # ValueError: Invalid value for age: -5
# user.email = "bad"  # ValueError: Invalid value for email: bad

# Lazy descriptor — compute once, cache forever
class LazyProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.func(obj)
        # Replace descriptor with computed value in instance dict
        setattr(obj, self.name, value)
        return value

class DataProcessor:
    def __init__(self, filename):
        self.filename = filename

    @LazyProperty
    def data(self):
        """Expensive computation — only done once."""
        print("Loading data...")
        with open(self.filename) as f:
            return f.read()

p = DataProcessor("big_file.csv")
# First access: "Loading data..." printed, data loaded
print(p.data)
# Second access: no loading, returns cached value
print(p.data)

# __slots__ for memory efficiency
class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 1 million Points with __slots__: ~80 MB
# 1 million Points without __slots__: ~170 MB
# Almost 50% memory savings!`,
				},
				{
					Title: "Concurrency: asyncio, Threading, and Multiprocessing",
					Content: `Python offers three concurrency models. Choosing the right one depends on your workload type.

**The GIL (Global Interpreter Lock):**
CPython has a GIL that allows only one thread to execute Python bytecode at a time. This means:
- **CPU-bound threads**: GIL prevents true parallelism (use multiprocessing)
- **I/O-bound threads**: GIL is released during I/O (threading works well)
- **asyncio**: Single-threaded, cooperative multitasking (best for I/O)

**Decision Guide:**

` + "```" + `
What type of work?
  │
  ├─ I/O-bound (network, disk, database)?
  │    ├─ Many connections (1000+)? → asyncio
  │    └─ Few connections? → threading or asyncio
  │
  └─ CPU-bound (computation, data processing)?
       └─ → multiprocessing
` + "```" + `

**1. asyncio (Async/Await):**
- Single-threaded, event-loop-based
- Best for: HTTP clients/servers, database queries, file I/O
- Strengths: Handles thousands of concurrent connections, low overhead
- Weaknesses: Entire ecosystem must be async ("async all the way down")
- Libraries: aiohttp, asyncpg, httpx, FastAPI

**2. threading:**
- OS threads, preemptive multitasking
- Best for: Simple I/O parallelism, legacy code
- Strengths: Easy to understand, works with existing synchronous code
- Weaknesses: GIL limits CPU parallelism, race conditions, deadlocks
- Use: concurrent.futures.ThreadPoolExecutor

**3. multiprocessing:**
- Separate OS processes, true parallelism
- Best for: CPU-intensive work (data processing, ML training)
- Strengths: Bypasses GIL, true parallel execution
- Weaknesses: Higher memory overhead, IPC complexity, no shared memory (easily)
- Use: concurrent.futures.ProcessPoolExecutor

**Python 3.12+ Free Threading (PEP 703):**
Experimental GIL-free mode available with ` + "`" + `python --disable-gil` + "`" + `. Not yet stable but a game-changer for CPU-bound Python threading.

**Common Pitfalls:**
1. Using threading for CPU-bound work (GIL blocks parallelism)
2. Not handling exceptions in async code (silent failures)
3. Mixing async and sync code without bridges
4. Shared mutable state without locks (race conditions)
5. Creating too many processes (memory overhead)`,
					CodeExamples: `# asyncio: Fetch multiple URLs concurrently
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Fetch 100 URLs concurrently (single thread!)
urls = [f"https://api.example.com/item/{i}" for i in range(100)]
results = asyncio.run(fetch_all(urls))

# Threading: Simple I/O parallelism
from concurrent.futures import ThreadPoolExecutor
import requests

def fetch_sync(url):
    return requests.get(url).text

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fetch_sync, urls))

# Multiprocessing: CPU-bound work
from concurrent.futures import ProcessPoolExecutor
import math

def cpu_intensive(n):
    """Find prime numbers up to n."""
    primes = []
    for i in range(2, n):
        if all(i % j != 0 for j in range(2, int(math.sqrt(i)) + 1)):
            primes.append(i)
    return primes

# Uses ALL CPU cores
with ProcessPoolExecutor() as executor:
    ranges = [100000, 200000, 300000, 400000]
    results = list(executor.map(cpu_intensive, ranges))

# asyncio producer-consumer pattern
async def producer(queue):
    for i in range(100):
        await queue.put(f"item_{i}")
        await asyncio.sleep(0.01)
    await queue.put(None)  # Sentinel

async def consumer(queue, name):
    while True:
        item = await queue.get()
        if item is None:
            queue.put_nowait(None)  # Pass sentinel to next consumer
            break
        print(f"{name} processed {item}")
        await asyncio.sleep(0.05)

async def main():
    queue = asyncio.Queue(maxsize=10)
    await asyncio.gather(
        producer(queue),
        consumer(queue, "worker-1"),
        consumer(queue, "worker-2"),
        consumer(queue, "worker-3"),
    )

asyncio.run(main())`,
				},
				{
					Title: "Type Hints and Static Analysis",
					Content: `Python's type system has evolved dramatically since Python 3.5. Modern Python uses type hints extensively for code quality, documentation, and IDE support.

**Why Type Hints?**
- Catch bugs before runtime (with mypy, pyright)
- Better IDE autocomplete and refactoring
- Self-documenting code
- Enable runtime validation (pydantic, FastAPI)

**Basic Types (3.9+):**
Since Python 3.9, you can use built-in types directly instead of importing from typing:
- ` + "`" + `list[int]` + "`" + ` instead of ` + "`" + `List[int]` + "`" + `
- ` + "`" + `dict[str, int]` + "`" + ` instead of ` + "`" + `Dict[str, int]` + "`" + `
- ` + "`" + `tuple[str, int]` + "`" + ` instead of ` + "`" + `Tuple[str, int]` + "`" + `

**Union Types (3.10+):**
` + "`" + `str | int` + "`" + ` instead of ` + "`" + `Union[str, int]` + "`" + `

**Key typing Module Features:**
- ` + "`" + `Optional[X]` + "`" + ` = ` + "`" + `X | None` + "`" + `
- ` + "`" + `TypeVar` + "`" + `: Generic type parameters
- ` + "`" + `Protocol` + "`" + `: Structural subtyping (like Go interfaces)
- ` + "`" + `TypedDict` + "`" + `: Dictionary with known key types
- ` + "`" + `Literal` + "`" + `: Exact value types
- ` + "`" + `TypeAlias` + "`" + `: Named type aliases
- ` + "`" + `ParamSpec` + "`" + `: Capture function parameter types (for decorators)

**Protocols (Structural Typing):**
Like Go interfaces — any class that has the right methods satisfies the protocol, without explicit inheritance.

**Pydantic (Runtime Validation):**
Type hints + runtime validation + serialization. The backbone of FastAPI.

**Best Practices:**
1. Always type function signatures (parameters and return)
2. Use ` + "`" + `mypy --strict` + "`" + ` in CI
3. Avoid ` + "`" + `Any` + "`" + ` — it defeats the purpose
4. Use ` + "`" + `Protocol` + "`" + ` for structural typing
5. Use ` + "`" + `dataclasses` + "`" + ` or ` + "`" + `pydantic.BaseModel` + "`" + ` for data classes`,
					CodeExamples: `from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass

# Basic type hints
def greet(name: str, times: int = 1) -> str:
    return f"Hello, {name}! " * times

# Generic function
T = TypeVar('T')

def first(items: list[T]) -> T | None:
    return items[0] if items else None

# Protocol (structural typing — like Go interfaces)
class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

class Square:
    def draw(self) -> str:
        return "Drawing square"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # Works! Circle has draw() method
render(Square())  # Works! Square has draw() method

# Generic class
class Repository(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def save(self, id: str, item: T) -> None:
        self._items[id] = item

    def get(self, id: str) -> T | None:
        return self._items.get(id)

@dataclass
class User:
    name: str
    email: str

user_repo = Repository[User]()
user_repo.save("1", User("Alice", "alice@example.com"))

# TypedDict for structured dictionaries
from typing import TypedDict

class APIResponse(TypedDict):
    status: int
    data: dict[str, str]
    error: str | None

def parse_response(resp: APIResponse) -> str:
    if resp["error"]:
        return f"Error: {resp['error']}"
    return f"Status {resp['status']}: {resp['data']}"

# Pydantic for runtime validation
from pydantic import BaseModel, EmailStr, field_validator

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    age: int

    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 0 or v > 150:
            raise ValueError('age must be between 0 and 150')
        return v

# Validates at runtime
user = CreateUserRequest(name="Alice", email="alice@example.com", age=30)
# Raises ValidationError for invalid data`,
				},
				{
					Title: "Python Performance and Profiling",
					Content: `Writing performant Python requires understanding CPython's internals, choosing the right data structures, and knowing when to reach for optimization tools.

**Python Performance Rules:**
1. **Measure first** — don't optimize without profiling
2. **Algorithm > micro-optimization** — O(n) beats optimized O(n²) always
3. **Use built-in operations** — they're implemented in C
4. **Avoid unnecessary copies** — generators instead of lists
5. **Use the right data structure** — set lookup O(1) vs list O(n)

**Common Performance Wins:**

**1. Use Sets for Membership Testing:**
` + "```" + `
# Bad: O(n) lookup
if item in large_list:  # Scans entire list

# Good: O(1) lookup
if item in large_set:   # Hash-based lookup
` + "```" + `

**2. Use Generators for Large Data:**
` + "```" + `
# Bad: Creates entire list in memory
squares = [x**2 for x in range(10_000_000)]  # ~80 MB

# Good: Generates values on demand
squares = (x**2 for x in range(10_000_000))  # ~0 MB
` + "```" + `

**3. Use Local Variables:**
Local variable access is faster than global/attribute access in CPython.

**4. Use ` + "`" + `collections.defaultdict` + "`" + ` and ` + "`" + `Counter` + "`" + `:**
Optimized C implementations beat manual dictionary logic.

**5. String Concatenation:**
` + "```" + `
# Bad: O(n²) — creates new string each iteration
result = ""
for s in strings:
    result += s

# Good: O(n) — join is implemented in C
result = "".join(strings)
` + "```" + `

**Profiling Tools:**

1. **cProfile**: Built-in profiler, function-level timing
2. **line_profiler**: Line-by-line timing (` + "`" + `@profile` + "`" + `)
3. **memory_profiler**: Track memory usage per line
4. **py-spy**: Sampling profiler (no code changes, works in production)
5. **scalene**: CPU + memory profiler with GPU support

**When Python Isn't Enough:**
- **Cython**: Write C extensions with Python-like syntax
- **NumPy/Pandas**: Vectorized operations in C
- **ctypes/cffi**: Call C libraries directly
- **Rust bindings (PyO3)**: Write performance-critical code in Rust
- **Numba**: JIT compilation for numerical code`,
					CodeExamples: `# Profiling with cProfile
import cProfile

def slow_function():
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

cProfile.run('slow_function()')
# Output shows time per function call

# Using timeit for micro-benchmarks
import timeit

# Compare list vs set membership testing
setup = "data = list(range(100000)); data_set = set(data)"

list_time = timeit.timeit("99999 in data", setup=setup, number=1000)
set_time = timeit.timeit("99999 in data_set", setup=setup, number=1000)

print(f"List: {list_time:.4f}s")  # ~2.5s
print(f"Set: {set_time:.4f}s")    # ~0.0001s
# Set is ~25,000x faster!

# Memory-efficient processing with generators
def process_large_file(filename):
    """Process file line by line without loading into memory."""
    with open(filename) as f:
        for line in f:  # Generator — one line at a time
            yield line.strip().split(',')

# Pipeline of generators (lazy evaluation)
def read_csv(filename):
    for row in process_large_file(filename):
        yield row

def filter_active(rows):
    for row in rows:
        if row[2] == 'active':
            yield row

def extract_emails(rows):
    for row in rows:
        yield row[1]

# Nothing executes until we iterate!
pipeline = extract_emails(filter_active(read_csv("users.csv")))
for email in pipeline:  # Processes one row at a time
    send_email(email)

# Using __slots__ for memory efficiency
class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

# vs regular class (uses __dict__)
class PointDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys
p1 = Point(1, 2)
p2 = PointDict(1, 2)
print(sys.getsizeof(p1))  # ~56 bytes
print(sys.getsizeof(p2))  # ~48 bytes + __dict__ (~104 bytes)`,
				},
			},
		},
		{
			ID:          2216,
			Title:       "Python Project Patterns",
			Description: "Learn production-grade Python project organization, dependency management, packaging, and deployment patterns.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Modern Python Project Structure",
					Content: `A well-organized Python project follows modern conventions using pyproject.toml, src layout, and proper tooling.

**The Modern Stack (2024+):**

**Package Management:**
- **uv**: Fastest Python package installer (replaces pip + virtualenv)
- **poetry**: Full project management (dependencies, packaging, publishing)
- **pip + venv**: Standard library (always available)

**Project Configuration:**
- **pyproject.toml**: Single file for all project config (replaces setup.py, setup.cfg, tox.ini, etc.)

**Linting & Formatting:**
- **ruff**: Ultra-fast linter + formatter (replaces flake8, isort, black)
- **mypy**: Static type checker

**Testing:**
- **pytest**: De facto standard testing framework
- **pytest-cov**: Coverage reporting
- **pytest-asyncio**: Testing async code

**Recommended Project Layout:**

` + "```" + `
my-project/
├── pyproject.toml          # All configuration in one file
├── README.md
├── src/
│   └── my_project/         # Source code (src layout)
│       ├── __init__.py
│       ├── main.py
│       ├── models.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── user_service.py
│       └── api/
│           ├── __init__.py
│           └── routes.py
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── test_models.py
│   └── test_services/
│       └── test_user_service.py
├── docs/
├── scripts/
├── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml
` + "```" + `

**Why src Layout?**
Prevents accidentally importing the local package instead of the installed one during testing. The ` + "`" + `src/` + "`" + ` directory isn't on ` + "`" + `sys.path` + "`" + `, so tests always use the installed version.

**pyproject.toml Best Practices:**
- Pin direct dependencies to compatible ranges (` + "`" + `~=1.2` + "`" + ` or ` + "`" + `>=1.2,<2` + "`" + `)
- Use lock files for reproducible installs (uv.lock, poetry.lock)
- Group dev dependencies separately
- Configure all tools in pyproject.toml (ruff, mypy, pytest)`,
					CodeExamples: `# pyproject.toml — complete modern Python project config
[project]
name = "my-project"
version = "1.0.0"
description = "A modern Python project"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.100",
    "sqlalchemy>=2.0",
    "pydantic>=2.0",
    "httpx>=0.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "mypy>=1.5",
    "ruff>=0.1",
]

[project.scripts]
my-app = "my_project.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py311"
line-length = 88
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]

[tool.mypy]
strict = true
python_version = "3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/my_project --cov-report=term-missing"

# ─────────────────────────────────────────

# Dockerfile for Python project
FROM python:3.12-slim AS base
WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies first (Docker layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Run
CMD ["uv", "run", "uvicorn", "my_project.main:app", "--host", "0.0.0.0"]

# ─────────────────────────────────────────

# conftest.py — shared test fixtures
import pytest
from my_project.database import Database

@pytest.fixture
def db():
    """In-memory database for testing."""
    database = Database(":memory:")
    database.create_tables()
    yield database
    database.close()

@pytest.fixture
def client(db):
    """Test client with injected database."""
    from my_project.main import create_app
    app = create_app(database=db)
    from fastapi.testclient import TestClient
    return TestClient(app)`,
				},
			},
		},
	})
}
