package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2221,
			Title:       "Python Design Patterns and Architecture",
			Description: "Master creational, structural, and behavioral design patterns, SOLID principles, and application architecture in Python.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Design Patterns SOLID Principles and Application Architecture",
					Content: `Python's flexibility and dynamic nature make design patterns both powerful and idiomatic. Understanding patterns helps build maintainable, extensible applications.

**SOLID Principles in Python:**

Single Responsibility Principle (SRP):
  A class should have only one reason to change
  Separate concerns into different classes
  Python: Use modules and functions freely
  
Open/Closed Principle (OCP):
  Open for extension, closed for modification
  Use abstract base classes and protocols
  Python: Duck typing + protocols enable extension
  
Liskov Substitution Principle (LSP):
  Subtypes must be substitutable for base types
  Don't violate the interface contract
  Python: Follow Protocol/ABC contracts
  
Interface Segregation Principle (ISP):
  Many specific interfaces > one general interface
  Python: Use Protocol classes (structural subtyping)
  Clients shouldn't depend on unused methods
  
Dependency Inversion Principle (DIP):
  Depend on abstractions, not concretions
  Python: Dependency injection via constructor
  Use Protocol/ABC for type hints

**Creational Patterns:**

Singleton:
  Ensure only one instance exists
  Python approaches:
    Module-level variable (simplest)
    __new__ override
    Metaclass
    Decorator
  
  Use cases: Configuration, logging, connection pool
  Caution: Global state, testing difficulty

Factory Method:
  Create objects without specifying exact class
  Subclasses decide which class to instantiate
  Python: Often just a function
  
  Variations:
    Simple Factory: Function returns instances
    Factory Method: Override in subclass
    Abstract Factory: Family of related objects

Builder:
  Construct complex objects step by step
  Separate construction from representation
  Python: Method chaining with fluent interface
  
  Use cases: Complex configurations, query builders

Prototype:
  Create objects by copying existing ones
  Python: copy.copy() / copy.deepcopy()
  __copy__ and __deepcopy__ for customization

**Structural Patterns:**

Adapter:
  Convert interface to expected interface
  Python: Wrapper class or function
  Multiple inheritance for class adapter
  
  Use cases: Third-party library integration

Decorator (Pattern, not @decorator):
  Add behavior without modifying original
  Python: @decorator syntax is built-in
  Composition over inheritance
  
  functools.wraps: Preserve original metadata

Facade:
  Simplified interface to complex subsystem
  Python: Module or class wrapping complex APIs
  Hide implementation details

Proxy:
  Placeholder controlling access to object
  Types: Virtual proxy, protection proxy, caching proxy
  Python: __getattr__ for transparent proxying

Composite:
  Tree structures with uniform interface
  Both leaves and composites implement same interface
  Python: Recursive data structures

Bridge:
  Separate abstraction from implementation
  Both can vary independently
  Python: Inject implementation via constructor

Flyweight:
  Share objects to reduce memory
  Python: __slots__, interning, caching
  weakref for optional caching

**Behavioral Patterns:**

Strategy:
  Interchangeable algorithms
  Python: Pass functions as arguments (first-class functions)
  Or use Protocol/ABC for complex strategies
  
  Use cases: Sorting, validation, pricing rules

Observer:
  Notify dependents of state changes
  Python: Callback lists, signals
  EventEmitter pattern
  weakref for avoiding memory leaks

Command:
  Encapsulate request as object
  Support undo, queuing, logging
  Python: Callable objects (__call__)

Template Method:
  Define algorithm skeleton, defer steps to subclasses
  Python: Abstract methods with ABC
  Or use hook methods with default implementations

Iterator:
  Sequential access to collection elements
  Python: __iter__ and __next__ (built-in support)
  Generator functions (yield)
  itertools module

State:
  Object behavior changes based on state
  State classes with associated behavior
  Python: Dynamic method dispatch or state objects

Chain of Responsibility:
  Pass request along chain until handled
  Python: Middleware chains, exception handling

Mediator:
  Centralized communication between objects
  Reduces coupling between components
  Python: Event bus, message broker pattern

Memento:
  Capture and restore object state
  Python: copy.deepcopy for state snapshots
  Or custom __getstate__/__setstate__

Visitor:
  Add operations to objects without changing them
  Python: Single dispatch (@singledispatch)
  Or use isinstance checks (more Pythonic)

**Application Architecture Patterns:**

Layered Architecture:
  Presentation → Business Logic → Data Access
  Each layer depends only on layer below
  
Hexagonal (Ports and Adapters):
  Core domain in center
  Ports: Interfaces for external communication
  Adapters: Implementations of ports
  Easy to test (mock adapters)
  
Clean Architecture:
  Entities → Use Cases → Interface Adapters → Frameworks
  Dependencies point inward
  
Repository Pattern:
  Abstraction over data storage
  In-memory, SQL, NoSQL implementations
  Easy to swap storage backends
  
Unit of Work:
  Track changes to objects
  Commit/rollback as transaction
  
CQRS (Command Query Responsibility Segregation):
  Separate read and write models
  Different optimization for each
  
Event Sourcing:
  Store events instead of current state
  Rebuild state by replaying events
  Full audit trail

**Python-Specific Patterns:**

Context Manager:
  __enter__ / __exit__ protocol
  @contextmanager decorator
  Resource acquisition/release
  
Descriptor:
  __get__, __set__, __delete__
  Property, classmethod, staticmethod
  Custom validators, lazy attributes
  
Metaclass:
  Classes that create classes
  __new__, __init__, __call__
  Registration, validation, ORM magic
  
Protocol (Structural Subtyping):
  Define interface without inheritance
  typing.Protocol
  Runtime: @runtime_checkable`,
					CodeExamples: `# Python design patterns and architecture examples

from abc import ABC, abstractmethod
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, 
    Optional, Protocol, TypeVar, runtime_checkable
)
from dataclasses import dataclass, field
from functools import wraps, singledispatch
from contextlib import contextmanager
from weakref import ref
import copy
import time

T = TypeVar('T')

# ============================================================
# Creational Patterns
# ============================================================

# --- Singleton ---

class SingletonMeta(type):
    """Metaclass-based singleton."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DatabaseConfig(metaclass=SingletonMeta):
    def __init__(self, host="localhost", port=5432):
        self.host = host
        self.port = port
        self.pool_size = 10


# --- Factory ---

@runtime_checkable
class Serializer(Protocol):
    def serialize(self, data: dict) -> str: ...
    def deserialize(self, raw: str) -> dict: ...


class JSONSerializer:
    def serialize(self, data: dict) -> str:
        import json
        return json.dumps(data)
    
    def deserialize(self, raw: str) -> dict:
        import json
        return json.loads(raw)


class XMLSerializer:
    def serialize(self, data: dict) -> str:
        parts = ['<data>']
        for key, value in data.items():
            parts.append(f'  <{key}>{value}</{key}>')
        parts.append('</data>')
        return '\n'.join(parts)
    
    def deserialize(self, raw: str) -> dict:
        result = {}
        for line in raw.split('\n'):
            line = line.strip()
            if line.startswith('<') and not line.startswith('<data') and not line.startswith('</'):
                tag_end = line.index('>')
                tag = line[1:tag_end]
                value_end = line.index('<', tag_end)
                value = line[tag_end + 1:value_end]
                result[tag] = value
        return result


class CSVSerializer:
    def serialize(self, data: dict) -> str:
        keys = ','.join(data.keys())
        values = ','.join(str(v) for v in data.values())
        return f"{keys}\n{values}"
    
    def deserialize(self, raw: str) -> dict:
        lines = raw.strip().split('\n')
        keys = lines[0].split(',')
        values = lines[1].split(',')
        return dict(zip(keys, values))


def serializer_factory(format_type: str) -> Serializer:
    """Factory function for serializers."""
    serializers = {
        'json': JSONSerializer,
        'xml': XMLSerializer,
        'csv': CSVSerializer,
    }
    cls = serializers.get(format_type)
    if cls is None:
        raise ValueError(f"Unknown format: {format_type}")
    return cls()


# --- Builder ---

class QueryBuilder:
    """SQL query builder with fluent interface."""
    
    def __init__(self):
        self._select = []
        self._from = ""
        self._where = []
        self._order_by = []
        self._limit = None
        self._offset = None
        self._joins = []
        self._group_by = []
        self._having = []
    
    def select(self, *columns: str) -> 'QueryBuilder':
        self._select.extend(columns)
        return self
    
    def from_table(self, table: str) -> 'QueryBuilder':
        self._from = table
        return self
    
    def where(self, condition: str) -> 'QueryBuilder':
        self._where.append(condition)
        return self
    
    def join(self, table: str, on: str, join_type: str = 'INNER') -> 'QueryBuilder':
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self
    
    def order_by(self, column: str, desc: bool = False) -> 'QueryBuilder':
        direction = "DESC" if desc else "ASC"
        self._order_by.append(f"{column} {direction}")
        return self
    
    def group_by(self, *columns: str) -> 'QueryBuilder':
        self._group_by.extend(columns)
        return self
    
    def having(self, condition: str) -> 'QueryBuilder':
        self._having.append(condition)
        return self
    
    def limit(self, n: int) -> 'QueryBuilder':
        self._limit = n
        return self
    
    def offset(self, n: int) -> 'QueryBuilder':
        self._offset = n
        return self
    
    def build(self) -> str:
        parts = []
        
        columns = ', '.join(self._select) if self._select else '*'
        parts.append(f"SELECT {columns}")
        
        if self._from:
            parts.append(f"FROM {self._from}")
        
        for join in self._joins:
            parts.append(join)
        
        if self._where:
            parts.append(f"WHERE {' AND '.join(self._where)}")
        
        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")
        
        if self._having:
            parts.append(f"HAVING {' AND '.join(self._having)}")
        
        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")
        
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")
        
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")
        
        return ' '.join(parts)


# ============================================================
# Structural Patterns
# ============================================================

# --- Adapter ---

class OldPaymentGateway:
    """Legacy payment system with different interface."""
    
    def make_payment(self, amount_cents: int, card_token: str) -> dict:
        return {
            'transaction_id': f'txn_{amount_cents}',
            'status': 'approved',
            'amount_cents': amount_cents,
        }


class PaymentProcessor(Protocol):
    def charge(self, amount: float, payment_method: str) -> bool: ...
    def refund(self, transaction_id: str) -> bool: ...


class PaymentAdapter:
    """Adapts OldPaymentGateway to PaymentProcessor interface."""
    
    def __init__(self, gateway: OldPaymentGateway):
        self._gateway = gateway
    
    def charge(self, amount: float, payment_method: str) -> bool:
        result = self._gateway.make_payment(
            int(amount * 100), payment_method
        )
        return result['status'] == 'approved'
    
    def refund(self, transaction_id: str) -> bool:
        return True


# --- Decorator Pattern ---

def retry(max_attempts: int = 3, delay: float = 1.0, 
          exceptions: tuple = (Exception,)):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait = delay * (2 ** attempt)
                        time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator


def cache(ttl: float = 60.0):
    """Caching decorator with TTL."""
    def decorator(func):
        _cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            
            if key in _cache:
                value, timestamp = _cache[key]
                if time.time() - timestamp < ttl:
                    return value
            
            result = func(*args, **kwargs)
            _cache[key] = (result, time.time())
            return result
        
        wrapper.clear_cache = lambda: _cache.clear()
        return wrapper
    return decorator


def log_calls(func):
    """Log function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper


# --- Proxy ---

class LazyProxy(Generic[T]):
    """Lazy loading proxy - loads object only when accessed."""
    
    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._instance: Optional[T] = None
    
    def _load(self):
        if self._instance is None:
            self._instance = self._factory()
    
    def __getattr__(self, name):
        self._load()
        return getattr(self._instance, name)


class CachingProxy:
    """Proxy that caches method results."""
    
    def __init__(self, target, ttl: float = 60.0):
        self._target = target
        self._cache = {}
        self._ttl = ttl
    
    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr
        
        def cached_method(*args, **kwargs):
            key = (name, args, tuple(sorted(kwargs.items())))
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return value
            result = attr(*args, **kwargs)
            self._cache[key] = (result, time.time())
            return result
        
        return cached_method


# ============================================================
# Behavioral Patterns
# ============================================================

# --- Strategy ---

class SortStrategy(Protocol):
    def sort(self, data: list) -> list: ...


class BubbleSort:
    def sort(self, data: list) -> list:
        arr = data[:]
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


class QuickSort:
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class MergeSort:
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self._merge(left, right)
    
    def _merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result


class Sorter:
    """Context that uses a sorting strategy."""
    
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    @property
    def strategy(self):
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort(self, data: list) -> list:
        return self._strategy.sort(data)


# --- Observer ---

class EventEmitter:
    """Observer pattern implementation."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, listener: Callable):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)
    
    def off(self, event: str, listener: Callable):
        if event in self._listeners:
            self._listeners[event].remove(listener)
    
    def once(self, event: str, listener: Callable):
        def wrapper(*args, **kwargs):
            self.off(event, wrapper)
            return listener(*args, **kwargs)
        self.on(event, wrapper)
    
    def emit(self, event: str, *args, **kwargs):
        for listener in self._listeners.get(event, []):
            listener(*args, **kwargs)
    
    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(event, []))


# --- Command ---

class Command(ABC):
    @abstractmethod
    def execute(self) -> Any: ...
    
    @abstractmethod
    def undo(self) -> Any: ...


@dataclass
class TextEditor:
    """Receiver: the object commands act upon."""
    content: str = ""


class InsertCommand(Command):
    def __init__(self, editor: TextEditor, text: str, position: int):
        self.editor = editor
        self.text = text
        self.position = position
    
    def execute(self):
        self.editor.content = (
            self.editor.content[:self.position] +
            self.text +
            self.editor.content[self.position:]
        )
    
    def undo(self):
        self.editor.content = (
            self.editor.content[:self.position] +
            self.editor.content[self.position + len(self.text):]
        )


class DeleteCommand(Command):
    def __init__(self, editor: TextEditor, position: int, length: int):
        self.editor = editor
        self.position = position
        self.length = length
        self.deleted_text = ""
    
    def execute(self):
        self.deleted_text = self.editor.content[
            self.position:self.position + self.length
        ]
        self.editor.content = (
            self.editor.content[:self.position] +
            self.editor.content[self.position + self.length:]
        )
    
    def undo(self):
        self.editor.content = (
            self.editor.content[:self.position] +
            self.deleted_text +
            self.editor.content[self.position:]
        )


class CommandHistory:
    """Invoker: stores and executes commands."""
    
    def __init__(self):
        self._history: List[Command] = []
        self._redo_stack: List[Command] = []
    
    def execute(self, command: Command):
        command.execute()
        self._history.append(command)
        self._redo_stack.clear()
    
    def undo(self):
        if self._history:
            command = self._history.pop()
            command.undo()
            self._redo_stack.append(command)
    
    def redo(self):
        if self._redo_stack:
            command = self._redo_stack.pop()
            command.execute()
            self._history.append(command)


# --- Chain of Responsibility ---

class Handler(ABC):
    def __init__(self):
        self._next: Optional['Handler'] = None
    
    def set_next(self, handler: 'Handler') -> 'Handler':
        self._next = handler
        return handler
    
    def handle(self, request: dict) -> Optional[dict]:
        if self._next:
            return self._next.handle(request)
        return None


class AuthHandler(Handler):
    def handle(self, request: dict) -> Optional[dict]:
        if 'token' not in request:
            return {'error': 'Authentication required', 'status': 401}
        return super().handle(request)


class RateLimitHandler(Handler):
    def __init__(self):
        super().__init__()
        self._requests = {}
    
    def handle(self, request: dict) -> Optional[dict]:
        ip = request.get('ip', 'unknown')
        now = time.time()
        
        if ip in self._requests:
            count, window_start = self._requests[ip]
            if now - window_start < 60:
                if count >= 100:
                    return {'error': 'Rate limit exceeded', 'status': 429}
                self._requests[ip] = (count + 1, window_start)
            else:
                self._requests[ip] = (1, now)
        else:
            self._requests[ip] = (1, now)
        
        return super().handle(request)


class ValidationHandler(Handler):
    def handle(self, request: dict) -> Optional[dict]:
        if 'body' not in request:
            return {'error': 'Request body required', 'status': 400}
        return super().handle(request)


# --- Repository Pattern ---

class Repository(Generic[T], ABC):
    """Abstract repository interface."""
    
    @abstractmethod
    def get(self, id: str) -> Optional[T]: ...
    
    @abstractmethod
    def list(self, **filters) -> List[T]: ...
    
    @abstractmethod
    def add(self, entity: T) -> T: ...
    
    @abstractmethod
    def update(self, entity: T) -> T: ...
    
    @abstractmethod
    def delete(self, id: str) -> bool: ...


@dataclass
class User:
    id: str
    name: str
    email: str
    active: bool = True


class InMemoryUserRepository(Repository[User]):
    """In-memory implementation of user repository."""
    
    def __init__(self):
        self._storage: Dict[str, User] = {}
    
    def get(self, id: str) -> Optional[User]:
        return self._storage.get(id)
    
    def list(self, **filters) -> List[User]:
        result = list(self._storage.values())
        for key, value in filters.items():
            result = [u for u in result if getattr(u, key, None) == value]
        return result
    
    def add(self, entity: User) -> User:
        self._storage[entity.id] = entity
        return entity
    
    def update(self, entity: User) -> User:
        self._storage[entity.id] = entity
        return entity
    
    def delete(self, id: str) -> bool:
        return self._storage.pop(id, None) is not None


# --- Unit of Work ---

class UnitOfWork:
    """Track changes and commit as transaction."""
    
    def __init__(self, repository: Repository):
        self._repository = repository
        self._new: List[Any] = []
        self._dirty: List[Any] = []
        self._removed: List[str] = []
    
    def register_new(self, entity):
        self._new.append(entity)
    
    def register_dirty(self, entity):
        self._dirty.append(entity)
    
    def register_removed(self, id: str):
        self._removed.append(id)
    
    def commit(self):
        for entity in self._new:
            self._repository.add(entity)
        for entity in self._dirty:
            self._repository.update(entity)
        for id in self._removed:
            self._repository.delete(id)
        self._clear()
    
    def rollback(self):
        self._clear()
    
    def _clear(self):
        self._new.clear()
        self._dirty.clear()
        self._removed.clear()


# --- Event Sourcing ---

@dataclass
class Event:
    event_type: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    version: int = 0


class EventStore:
    """Append-only event store."""
    
    def __init__(self):
        self._events: Dict[str, List[Event]] = {}
    
    def append(self, aggregate_id: str, event: Event):
        if aggregate_id not in self._events:
            self._events[aggregate_id] = []
        event.version = len(self._events[aggregate_id]) + 1
        self._events[aggregate_id].append(event)
    
    def get_events(self, aggregate_id: str, 
                   after_version: int = 0) -> List[Event]:
        events = self._events.get(aggregate_id, [])
        return [e for e in events if e.version > after_version]
    
    def get_all_events(self) -> List[Event]:
        all_events = []
        for events in self._events.values():
            all_events.extend(events)
        all_events.sort(key=lambda e: e.timestamp)
        return all_events


class Aggregate(ABC):
    """Base aggregate with event sourcing."""
    
    def __init__(self, id: str):
        self.id = id
        self.version = 0
        self._pending_events: List[Event] = []
    
    def apply_event(self, event: Event):
        handler = getattr(self, f'_apply_{event.event_type}', None)
        if handler:
            handler(event.data)
        self.version = event.version
    
    def raise_event(self, event_type: str, data: dict):
        event = Event(event_type=event_type, data=data)
        self._pending_events.append(event)
        self.apply_event(event)
    
    def get_pending_events(self) -> List[Event]:
        events = self._pending_events[:]
        self._pending_events.clear()
        return events
    
    @classmethod
    def from_events(cls, id: str, events: List[Event]):
        aggregate = cls(id)
        for event in events:
            aggregate.apply_event(event)
        return aggregate


class BankAccount(Aggregate):
    """Example aggregate with event sourcing."""
    
    def __init__(self, id: str):
        super().__init__(id)
        self.balance = 0.0
        self.owner = ""
        self.active = False
    
    def open(self, owner: str, initial_deposit: float):
        if self.active:
            raise ValueError("Account already open")
        self.raise_event('account_opened', {
            'owner': owner,
            'initial_deposit': initial_deposit,
        })
    
    def deposit(self, amount: float):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self.raise_event('money_deposited', {'amount': amount})
    
    def withdraw(self, amount: float):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.raise_event('money_withdrawn', {'amount': amount})
    
    def _apply_account_opened(self, data: dict):
        self.owner = data['owner']
        self.balance = data['initial_deposit']
        self.active = True
    
    def _apply_money_deposited(self, data: dict):
        self.balance += data['amount']
    
    def _apply_money_withdrawn(self, data: dict):
        self.balance -= data['amount']`,
				},
			},
		},
	})
}
