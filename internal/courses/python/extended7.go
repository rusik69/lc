package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2222,
			Title:       "Python Advanced Topics and Internals",
			Description: "Master metaclasses, descriptors, generators, memory management, GC internals, C extensions, and performance optimization.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Metaclasses Descriptors Generators Memory and Performance",
					Content: `Understanding Python internals and advanced features enables writing high-performance, idiomatic code.

**Metaclasses:**

What is a metaclass:
  A class whose instances are classes
  type is the default metaclass
  Classes are objects (instances of their metaclass)
  
  type(name, bases, namespace) → creates a class
  
Metaclass creation process:
  1. Python collects class body into namespace dict
  2. Calls metaclass.__new__(mcs, name, bases, namespace)
  3. Calls metaclass.__init__(cls, name, bases, namespace)
  4. Returns new class object
  
Custom metaclass:
  class Meta(type):
      def __new__(mcs, name, bases, namespace):
          # Modify class before creation
          return super().__new__(mcs, name, bases, namespace)
      
      def __init__(cls, name, bases, namespace):
          # Configure class after creation
          super().__init__(name, bases, namespace)
      
      def __call__(cls, *args, **kwargs):
          # Control instance creation
          return super().__call__(*args, **kwargs)

__init_subclass__:
  Simpler alternative to metaclasses
  Called when a class is subclassed
  class Base:
      def __init_subclass__(cls, **kwargs):
          super().__init_subclass__(**kwargs)
          # Register, validate, etc.

__class_getitem__:
  Support MyClass[T] syntax
  Generic class subscripting
  
Use cases:
  ORM field registration
  API endpoint registration
  Plugin systems
  Interface enforcement
  Singleton pattern

**Descriptors:**

Descriptor protocol:
  __get__(self, obj, objtype=None)
  __set__(self, obj, value)
  __delete__(self, obj)
  __set_name__(self, owner, name)
  
Data descriptor: Defines __set__ and/or __delete__
  Takes priority over instance __dict__
  
Non-data descriptor: Only __get__
  Instance __dict__ takes priority
  
Lookup order:
  1. Data descriptors (from class)
  2. Instance __dict__
  3. Non-data descriptors (from class)
  4. __getattr__ (fallback)

Built-in descriptors:
  property: Data descriptor
  classmethod: Non-data descriptor
  staticmethod: Non-data descriptor
  
  functions: Non-data descriptors (__get__ returns bound method)

Custom descriptors:
  Validated fields (type checking, range checking)
  Lazy computed properties
  Cached properties (functools.cached_property)
  Logging attribute access
  Read-only attributes

__slots__:
  Restrict instance attributes
  Saves memory (no __dict__)
  Faster attribute access
  Descriptors under the hood
  
  class Point:
      __slots__ = ('x', 'y')

**Generators and Coroutines:**

Generator functions:
  yield — produce value, suspend execution
  yield from — delegate to sub-generator
  send(value) — resume with value
  throw(exc) — raise exception at yield
  close() — raise GeneratorExit
  
Generator expressions:
  (expr for x in iterable if cond)
  Lazy evaluation, memory efficient
  
Iterator protocol:
  __iter__() — return self
  __next__() — return next value or raise StopIteration

itertools:
  count(start, step) — infinite counter
  cycle(iterable) — infinite cycle
  repeat(elem, n) — repeat element
  
  chain(*iterables) — concatenate
  compress(data, selectors) — filter by selectors
  islice(iterable, start, stop, step) — slice
  
  product(*iterables) — Cartesian product
  permutations(iterable, r) — permutations
  combinations(iterable, r) — combinations
  combinations_with_replacement(iterable, r)
  
  groupby(iterable, key) — group consecutive
  accumulate(iterable, func) — running total
  starmap(func, iterable) — unpack arguments
  
  takewhile(pred, iterable) — take while true
  dropwhile(pred, iterable) — drop while true
  filterfalse(pred, iterable) — opposite of filter
  
  zip_longest(*iterables, fillvalue=None)
  tee(iterable, n) — n independent iterators
  batched(iterable, n) — batch into chunks (3.12+)

Coroutines (async/await):
  async def — define coroutine
  await — suspend until result ready
  async for — async iteration
  async with — async context manager
  
  AsyncIterator:
    __aiter__() → self
    __anext__() → await next value
    
  AsyncContextManager:
    __aenter__()
    __aexit__()
    
  AsyncGenerator:
    async def gen():
        yield value

**Memory Management:**

Reference counting:
  Primary mechanism
  Immediate cleanup when refcount hits 0
  sys.getrefcount(obj)
  
  Pros: Deterministic, immediate
  Cons: Can't handle cycles

Cycle detection (gc module):
  Generational garbage collector
  3 generations (0, 1, 2)
  New objects in generation 0
  Surviving objects promoted
  
  gc.collect() — force collection
  gc.get_count() — object counts per generation
  gc.get_threshold() — collection thresholds
  gc.disable() — disable GC (careful!)
  gc.set_debug(gc.DEBUG_LEAK) — debug mode

weakref module:
  References that don't prevent GC
  weakref.ref(obj, callback=None)
  weakref.proxy(obj) — transparent proxy
  WeakValueDictionary — values are weak refs
  WeakSet — elements are weak refs
  
  Use cases: Caches, observer patterns, parent references

Memory layout:
  Everything is an object
  Each object: refcount + type pointer + value
  int: 28 bytes (small ints cached: -5 to 256)
  float: 24 bytes
  str: 49+ bytes (compact: ASCII uses 1 byte/char)
  list: 56 + 8n bytes (pointer array)
  dict: 232+ bytes (hash table)
  
  sys.getsizeof(obj) — shallow size
  
  Small object allocator:
    Objects < 512 bytes
    Pools of fixed-size blocks
    Arena (256 KB) → Pool (4 KB) → Block (8-512 bytes)

Memory optimization:
  __slots__: No __dict__ (40-50% savings per instance)
  Interning: Reuse identical strings
  tuple > list for immutable sequences
  array.array for homogeneous numeric data
  memoryview for zero-copy slicing
  
  Collections:
    collections.deque > list for queue operations
    frozenset > set when immutable
    bytes > str for binary data

**Performance Optimization:**

Profiling:
  cProfile: C extension profiler
    python -m cProfile -s cumtime script.py
    
  profile: Pure Python profiler
  
  line_profiler: Line-by-line profiling
    @profile decorator
    
  memory_profiler: Memory usage profiling
    @profile for memory
    
  tracemalloc: Memory allocation tracking
    tracemalloc.start()
    snapshot = tracemalloc.take_snapshot()
    
  py-spy: Sampling profiler (no overhead)
    py-spy top -- python script.py

timeit:
  python -m timeit "expression"
  timeit.timeit(stmt, number=1000000)
  
  IPython: %timeit and %%timeit magic

Common optimizations:
  1. Use built-in functions (map, filter, sum, min, max)
  2. List comprehensions > loops
  3. Local variables > global variables
  4. str.join() > string concatenation
  5. collections.defaultdict > setdefault
  6. dict.get() > try/except KeyError
  7. Set operations for membership testing
  8. Avoid unnecessary object creation
  9. Use __slots__ for data classes
  10. Lazy evaluation with generators

Caching:
  functools.lru_cache — in-memory LRU cache
  functools.cache — unlimited cache (3.9+)
  cachetools — extensible caching library
  
  @lru_cache(maxsize=128)
  @cache — simple unbounded cache

C Extensions:
  ctypes: Call C libraries directly
  cffi: C Foreign Function Interface
  Cython: C extensions from Python-like code
  pybind11: C++ bindings
  
  Writing C extension:
    PyObject for Python objects
    Py_INCREF/Py_DECREF for refcounting
    PyArg_ParseTuple for argument parsing

Alternative implementations:
  PyPy: JIT-compiled Python (2-10x faster)
  Cython: Static compilation
  Nuitka: Python-to-C compiler
  mypyc: Compile type-annotated Python

**Type System Advanced:**

Generic Types:
  TypeVar: T = TypeVar('T')
  Generic[T]: Base for generic classes
  ParamSpec: P = ParamSpec('P')
  TypeVarTuple: Ts = TypeVarTuple('Ts')

Protocol:
  Structural subtyping
  @runtime_checkable
  No inheritance needed
  Duck typing with type safety

Union Types:
  X | Y (Python 3.10+)
  Optional[X] = X | None

Type Guards:
  TypeGuard[T]: Narrow type in conditional
  TypeIs[T]: More precise narrowing (3.12+)
  assert_type(val, Type): Static assertion

Literal Types:
  Literal['GET', 'POST']
  Restrict to specific values

TypedDict:
  Dict with specific key types
  total=False for optional keys
  Required/NotRequired (3.11+)

Annotated:
  Annotated[int, Gt(0)] — metadata
  Used by Pydantic, attrs for validation

Self:
  from typing import Self
  Method return type for fluent APIs

overload:
  @overload for multiple signatures
  Actual implementation without @overload`,
					CodeExamples: `# Python advanced topics and internals examples

import sys
import gc
import weakref
import time
import functools
from abc import ABC, abstractmethod
from typing import (
    Any, Callable, Dict, Generic, Iterator, List,
    Optional, Protocol, TypeVar, Type, runtime_checkable
)
from dataclasses import dataclass

T = TypeVar('T')

# ============================================================
# Metaclass Examples
# ============================================================

class RegistryMeta(type):
    """Metaclass that automatically registers subclasses."""
    
    _registry: Dict[str, type] = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases:  # Don't register the base class itself
            mcs._registry[name] = cls
        return cls
    
    @classmethod
    def get_registry(mcs):
        return dict(mcs._registry)
    
    @classmethod
    def create(mcs, name, *args, **kwargs):
        cls = mcs._registry.get(name)
        if cls is None:
            raise ValueError(f"Unknown class: {name}")
        return cls(*args, **kwargs)


class Plugin(metaclass=RegistryMeta):
    """Base class for auto-registered plugins."""
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        pass


class ValidationMeta(type):
    """Metaclass that validates class definitions."""
    
    def __new__(mcs, name, bases, namespace):
        # Ensure all abstract methods are implemented
        abstract = set()
        for base in bases:
            for attr_name in dir(base):
                attr = getattr(base, attr_name, None)
                if getattr(attr, '__isabstractmethod__', False):
                    abstract.add(attr_name)
        
        for method_name in abstract:
            if method_name not in namespace:
                # Check if any base provides it
                found = False
                for base in bases:
                    if method_name in base.__dict__:
                        if not getattr(base.__dict__[method_name], 
                                      '__isabstractmethod__', False):
                            found = True
                            break
                if not found and not namespace.get(method_name):
                    pass  # Allow abstract classes
        
        # Enforce naming conventions
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and not attr_name.startswith('_'):
                if not attr_name.islower():
                    raise TypeError(
                        f"Method '{attr_name}' in '{name}' must be lowercase")
        
        return super().__new__(mcs, name, bases, namespace)


class InterfaceEnforcer(type):
    """Metaclass that enforces interface contracts."""
    
    required_methods: List[str] = []
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        if bases and hasattr(mcs, 'required_methods'):
            for method in mcs.required_methods:
                if method not in namespace:
                    raise TypeError(
                        f"Class '{name}' must implement '{method}'")
        
        return cls


# ============================================================
# Descriptor Examples
# ============================================================

class Validated:
    """Descriptor that validates on set."""
    
    def __init__(self, validator=None, default=None):
        self._validator = validator
        self._default = default
        self._name = None
    
    def __set_name__(self, owner, name):
        self._name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name, self._default)
    
    def __set__(self, obj, value):
        if self._validator:
            self._validator(value)
        setattr(obj, self._name, value)
    
    def __delete__(self, obj):
        delattr(obj, self._name)


class TypeChecked:
    """Descriptor that enforces type."""
    
    def __init__(self, expected_type: type):
        self._expected_type = expected_type
        self._name = None
    
    def __set_name__(self, owner, name):
        self._name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name, None)
    
    def __set__(self, obj, value):
        if not isinstance(value, self._expected_type):
            raise TypeError(
                f"Expected {self._expected_type.__name__}, "
                f"got {type(value).__name__}")
        setattr(obj, self._name, value)


class Bounded:
    """Descriptor that enforces numeric bounds."""
    
    def __init__(self, min_val=None, max_val=None):
        self._min = min_val
        self._max = max_val
        self._name = None
    
    def __set_name__(self, owner, name):
        self._name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name, None)
    
    def __set__(self, obj, value):
        if self._min is not None and value < self._min:
            raise ValueError(f"Value must be >= {self._min}")
        if self._max is not None and value > self._max:
            raise ValueError(f"Value must be <= {self._max}")
        setattr(obj, self._name, value)


class LazyProperty:
    """Descriptor for lazy-computed properties (computed once)."""
    
    def __init__(self, func):
        self._func = func
        self._name = None
    
    def __set_name__(self, owner, name):
        self._name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self._func(obj)
        # Store in instance dict to bypass descriptor on next access
        setattr(obj, self._name, value)
        return value


class Observable:
    """Descriptor that notifies on changes."""
    
    def __init__(self, default=None):
        self._default = default
        self._name = None
        self._callbacks = []
    
    def __set_name__(self, owner, name):
        self._name = f'_{name}'
        self._attr_name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name, self._default)
    
    def __set__(self, obj, value):
        old_value = getattr(obj, self._name, self._default)
        setattr(obj, self._name, value)
        for callback in self._callbacks:
            callback(obj, self._attr_name, old_value, value)
    
    def on_change(self, callback):
        self._callbacks.append(callback)
        return callback


# ============================================================
# Generator Patterns
# ============================================================

def chunked(iterable, size: int):
    """Yield successive chunks of specified size."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def windowed(iterable, size: int, step: int = 1):
    """Sliding window over iterable."""
    buffer = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) == size:
            yield tuple(buffer)
            buffer = buffer[step:]


def flatten(iterable, depth: int = -1):
    """Flatten nested iterables to specified depth."""
    for item in iterable:
        if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
            if depth != 0:
                yield from flatten(item, depth - 1)
            else:
                yield item
        else:
            yield item


def interleave(*iterables):
    """Interleave multiple iterables."""
    iterators = [iter(it) for it in iterables]
    while iterators:
        next_iterators = []
        for it in iterators:
            try:
                yield next(it)
                next_iterators.append(it)
            except StopIteration:
                pass
        iterators = next_iterators


def take(n: int, iterable):
    """Take first n items from iterable."""
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item


def pairwise(iterable):
    """Yield consecutive pairs."""
    prev = None
    first = True
    for item in iterable:
        if first:
            prev = item
            first = False
        else:
            yield prev, item
            prev = item


class Pipeline:
    """Lazy pipeline using generators."""
    
    def __init__(self, source):
        self._source = source
        self._transforms = []
    
    def map(self, func):
        self._transforms.append(('map', func))
        return self
    
    def filter(self, predicate):
        self._transforms.append(('filter', predicate))
        return self
    
    def take(self, n):
        self._transforms.append(('take', n))
        return self
    
    def skip(self, n):
        self._transforms.append(('skip', n))
        return self
    
    def __iter__(self):
        result = iter(self._source)
        
        for op, arg in self._transforms:
            if op == 'map':
                result = (arg(x) for x in result)
            elif op == 'filter':
                result = (x for x in result if arg(x))
            elif op == 'take':
                result = take(arg, result)
            elif op == 'skip':
                result = _skip(arg, result)
        
        return result
    
    def collect(self) -> list:
        return list(self)
    
    def reduce(self, func, initial=None):
        return functools.reduce(func, self, initial)


def _skip(n, iterable):
    for i, item in enumerate(iterable):
        if i >= n:
            yield item


# ============================================================
# Memory Management
# ============================================================

class MemoryTracker:
    """Track object memory usage."""
    
    @staticmethod
    def sizeof(obj, seen=None) -> int:
        """Deep sizeof including referenced objects."""
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(
                MemoryTracker.sizeof(k, seen) + MemoryTracker.sizeof(v, seen)
                for k, v in obj.items()
            )
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(MemoryTracker.sizeof(item, seen) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += MemoryTracker.sizeof(obj.__dict__, seen)
        elif hasattr(obj, '__slots__'):
            size += sum(
                MemoryTracker.sizeof(getattr(obj, slot, None), seen)
                for slot in obj.__slots__
                if hasattr(obj, slot)
            )
        
        return size
    
    @staticmethod
    def object_count_by_type() -> Dict[str, int]:
        """Count objects by type."""
        counts: Dict[str, int] = {}
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            counts[type_name] = counts.get(type_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:20])


class WeakCache(Generic[T]):
    """Cache using weak references."""
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
    
    def get(self, key: str) -> Optional[T]:
        ref = self._cache.get(key)
        if ref is not None:
            value = ref()
            if value is not None:
                return value
            del self._cache[key]
        return None
    
    def put(self, key: str, value: T):
        def on_finalize(ref):
            self._cache.pop(key, None)
        self._cache[key] = weakref.ref(value, on_finalize)
    
    def __len__(self):
        # Clean up dead references
        dead = [k for k, v in self._cache.items() if v() is None]
        for k in dead:
            del self._cache[k]
        return len(self._cache)


class ObjectPool(Generic[T]):
    """Object pool to reduce allocation overhead."""
    
    def __init__(self, factory: Callable[[], T], max_size: int = 10):
        self._factory = factory
        self._max_size = max_size
        self._available: List[T] = []
        self._in_use: int = 0
    
    def acquire(self) -> T:
        if self._available:
            self._in_use += 1
            return self._available.pop()
        self._in_use += 1
        return self._factory()
    
    def release(self, obj: T):
        self._in_use -= 1
        if len(self._available) < self._max_size:
            self._available.append(obj)
    
    @property
    def stats(self) -> dict:
        return {
            'available': len(self._available),
            'in_use': self._in_use,
            'max_size': self._max_size,
        }


# ============================================================
# Performance Optimization
# ============================================================

class Profiler:
    """Simple code profiler."""
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
    
    def time(self, name: str = None):
        """Decorator/context manager for timing."""
        if callable(name):
            # Used as decorator without arguments
            func = name
            return self._wrap(func, func.__name__)
        
        # Used as decorator with name argument
        def decorator(func):
            return self._wrap(func, name or func.__name__)
        return decorator
    
    def _wrap(self, func, name):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if name not in self._timings:
                    self._timings[name] = []
                self._timings[name].append(elapsed)
        return wrapper
    
    def report(self) -> str:
        lines = [f"{'Function':<30} {'Calls':>6} {'Total':>10} {'Mean':>10} {'Min':>10} {'Max':>10}"]
        lines.append('-' * 80)
        
        for name, times in sorted(self._timings.items()):
            total = sum(times)
            mean = total / len(times)
            lines.append(
                f"{name:<30} {len(times):>6} {total:>10.4f} "
                f"{mean:>10.4f} {min(times):>10.4f} {max(times):>10.4f}"
            )
        
        return '\n'.join(lines)
    
    def reset(self):
        self._timings.clear()


def memoize(func):
    """Memoization decorator with cache stats."""
    cache = {}
    hits = 0
    misses = 0
    
    @functools.wraps(func)
    def wrapper(*args):
        nonlocal hits, misses
        if args in cache:
            hits += 1
            return cache[args]
        misses += 1
        result = func(*args)
        cache[args] = result
        return result
    
    wrapper.cache = cache
    wrapper.cache_info = lambda: {'hits': hits, 'misses': misses, 'size': len(cache)}
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


class LRUCache(Generic[T]):
    """LRU cache implementation."""
    
    def __init__(self, maxsize: int = 128):
        self._maxsize = maxsize
        self._cache: Dict[Any, T] = {}
        self._order: List[Any] = []
        self._hits = 0
        self._misses = 0
    
    def get(self, key) -> Optional[T]:
        if key in self._cache:
            self._hits += 1
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        self._misses += 1
        return None
    
    def put(self, key, value: T):
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._maxsize:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = value
        self._order.append(key)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def __len__(self):
        return len(self._cache)
    
    def __contains__(self, key):
        return key in self._cache`,
				},
			},
		},
	})
}
