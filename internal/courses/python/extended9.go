package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2224,
			Title:       "Python Database and ORM Patterns",
			Description: "Master SQLAlchemy, database migrations, connection pooling, query optimization, NoSQL integration, and data access patterns.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "SQLAlchemy Migrations Connection Pooling and Data Access",
					Content: `Python's database ecosystem provides powerful tools for relational and NoSQL databases, with SQLAlchemy being the most comprehensive ORM.

**SQLAlchemy Core:**

Engine:
  create_engine(url) — connection factory
  Connection pooling built-in
  
  URL formats:
    sqlite:///path/to/db.sqlite
    postgresql://user:pass@host:5432/db
    mysql+pymysql://user:pass@host:3306/db
    
  Engine options:
    echo=True — log SQL
    pool_size=5 — connection pool size
    max_overflow=10 — extra connections beyond pool_size
    pool_timeout=30 — wait time for connection
    pool_recycle=3600 — recycle connections after N seconds
    pool_pre_ping=True — verify connections before use

Connection:
  with engine.connect() as conn:
      result = conn.execute(text("SELECT ..."))
      conn.commit()

SQL Expression Language:
  Table, Column, MetaData
  select(), insert(), update(), delete()
  join(), outerjoin()
  and_(), or_(), not_()
  func.count(), func.sum(), func.max()
  
  Type system:
    Integer, String, Float, Boolean
    DateTime, Date, Time
    Text, LargeBinary
    JSON, ARRAY (PostgreSQL)
    Enum

**SQLAlchemy ORM:**

Declarative Base:
  from sqlalchemy.orm import DeclarativeBase
  class Base(DeclarativeBase): pass
  
  Mapped columns:
    name: Mapped[str] = mapped_column(String(100))
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(String(200))

Relationships:
  One-to-Many:
    posts: Mapped[List["Post"]] = relationship(back_populates="author")
    
  Many-to-One:
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    author: Mapped["User"] = relationship(back_populates="posts")
    
  Many-to-Many:
    Association table + relationship with secondary=
    
  One-to-One:
    uselist=False on relationship

Session:
  from sqlalchemy.orm import Session
  with Session(engine) as session:
      session.add(obj)
      session.commit()
      
  Session lifecycle:
    new → add → pending → flush → persistent → commit
    expunge → detached
    delete → deleted → flush → detached
    
  Patterns:
    session.get(Model, id) — by primary key
    session.query(Model).filter_by(...) — legacy query
    select(Model).where(Model.col == val) — 2.0 style
    session.scalars(stmt).all() — execute and get results

Query Examples:
  # Select
  stmt = select(User).where(User.name == "Alice")
  user = session.scalars(stmt).first()
  
  # Join
  stmt = select(User, Post).join(Post)
  
  # Aggregation
  stmt = select(func.count(User.id)).where(User.active == True)
  
  # Subquery
  subq = select(func.count(Post.id)).where(Post.author_id == User.id).scalar_subquery()
  stmt = select(User.name, subq.label("post_count"))
  
  # Eager loading
  stmt = select(User).options(selectinload(User.posts))
  stmt = select(User).options(joinedload(User.profile))
  
  # Pagination
  stmt = select(User).offset(20).limit(10)
  
  # Ordering
  stmt = select(User).order_by(User.name.asc())

Loading Strategies:
  lazy="select" — N+1 queries (default)
  selectinload — SELECT IN (...) for collections
  joinedload — LEFT JOIN
  subqueryload — separate subquery
  raiseload — raise error if accessed (prevent N+1)
  
  Best practices:
    Use selectinload for collections
    Use joinedload for single objects
    Use raiseload in performance-critical paths

Events:
  @event.listens_for(Session, "before_commit")
  @event.listens_for(User, "before_insert")
  @event.listens_for(engine, "before_cursor_execute")

Hybrid Properties:
  @hybrid_property — works at instance and SQL level
  @hybrid_method — method version
  Custom SQL expressions

**Async SQLAlchemy:**

  create_async_engine(url)
  async_sessionmaker(engine)
  AsyncSession
  
  async with async_session() as session:
      result = await session.execute(stmt)
      await session.commit()

**Alembic (Migrations):**

Setup:
  alembic init alembic
  Configure alembic.ini and env.py
  
Commands:
  alembic revision --autogenerate -m "message"
  alembic upgrade head
  alembic downgrade -1
  alembic history
  alembic current
  alembic stamp head — mark as current
  
Migration file:
  def upgrade():
      op.create_table(...)
      op.add_column(...)
      op.create_index(...)
      op.alter_column(...)
      
  def downgrade():
      op.drop_table(...)
      op.drop_column(...)
      
  Operations:
    create_table, drop_table
    add_column, drop_column, alter_column
    create_index, drop_index
    create_foreign_key, drop_constraint
    execute (raw SQL)
    bulk_insert
    
  Data migrations:
    Use op.execute() for data changes
    Or use bind.execute() with ORM

**Connection Pooling:**

Pool types:
  QueuePool (default): FIFO queue
  StaticPool: Single connection (testing)
  NullPool: No pooling (new connection each time)
  AsyncAdaptedQueuePool: Async adapter
  
Configuration:
  pool_size: Permanent connections
  max_overflow: Temporary connections above pool_size
  pool_timeout: Wait time for available connection
  pool_recycle: Max connection age
  pool_pre_ping: Check connection before use
  
  Best practices:
    pool_size = number of concurrent workers
    pool_recycle < database wait_timeout
    pool_pre_ping = True for reliability
    max_overflow = pool_size for burst capacity

**Query Optimization:**

Indexes:
  Single column: Index('idx_name', User.name)
  Composite: Index('idx_name_email', User.name, User.email)
  Unique: Index('idx_email', User.email, unique=True)
  Partial: Index('idx_active', User.name, postgresql_where=(User.active == True))
  
  When to index:
    Frequently filtered columns
    Join columns
    Order by columns
    High selectivity columns
    
  When NOT to index:
    Small tables
    Frequently updated columns
    Low selectivity (boolean)

Query analysis:
  EXPLAIN ANALYZE — execution plan
  session.execute(text("EXPLAIN ANALYZE ..."))
  
N+1 Query Problem:
  Problem: Loading related objects in a loop
  Solution: Eager loading (selectinload, joinedload)
  Detection: SQL logging, raiseload

Batch operations:
  session.bulk_save_objects(objects) — legacy bulk
  session.execute(insert(Model).values([...])) — 2.0 bulk
  session.execute(update(Model).where(...).values(...)) — bulk update
  
  Returning:
    insert(Model).returning(Model) — get inserted rows

**NoSQL Integration:**

MongoDB (Motor/PyMongo):
  client = AsyncIOMotorClient(url)
  db = client.database
  collection = db.collection
  
  CRUD:
    await collection.insert_one(doc)
    await collection.find(query).to_list(limit)
    await collection.update_one(filter, update)
    await collection.delete_one(filter)
    
  Aggregation pipeline:
    collection.aggregate([
      {"$match": {...}},
      {"$group": {"_id": "$field", "count": {"$sum": 1}}},
      {"$sort": {"count": -1}}
    ])

Redis:
  import redis
  r = redis.Redis(host='localhost', port=6379, db=0)
  
  Strings: r.set(key, value), r.get(key)
  Hashes: r.hset(name, key, value), r.hgetall(name)
  Lists: r.lpush(name, *values), r.lrange(name, 0, -1)
  Sets: r.sadd(name, *values), r.smembers(name)
  Sorted sets: r.zadd(name, {member: score})
  
  Async: aioredis
  
  Use cases:
    Caching (with TTL)
    Session storage
    Rate limiting
    Pub/sub messaging
    Task queues
    Leaderboards

**Data Access Patterns:**

Repository Pattern:
  Abstract data access behind interface
  Swap implementations (SQL, NoSQL, in-memory)
  
Unit of Work:
  Track changes, batch commits
  Transaction management
  
Active Record:
  Model methods for persistence
  Django ORM style
  
Data Mapper:
  Separate domain objects from persistence
  SQLAlchemy ORM style
  
CQRS:
  Separate read/write models
  Different optimization for each
  Read: Denormalized views
  Write: Normalized tables
  
Specification Pattern:
  Encapsulate query criteria
  Composable with and/or/not
  Reusable across repositories`,
					CodeExamples: `# Python database and ORM pattern examples

import time
import hashlib
import threading
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager

T = TypeVar('T')
ID = TypeVar('ID')

# ============================================================
# Connection Pool Implementation
# ============================================================

class Connection:
    """Simulated database connection."""
    
    _next_id = 0
    
    def __init__(self, dsn: str):
        Connection._next_id += 1
        self.id = Connection._next_id
        self.dsn = dsn
        self.created_at = time.time()
        self.last_used = time.time()
        self._closed = False
        self._in_transaction = False
    
    def execute(self, sql: str, params: tuple = None) -> list:
        if self._closed:
            raise RuntimeError("Connection is closed")
        self.last_used = time.time()
        return []
    
    def begin(self):
        self._in_transaction = True
    
    def commit(self):
        self._in_transaction = False
    
    def rollback(self):
        self._in_transaction = False
    
    def close(self):
        self._closed = True
    
    @property
    def is_closed(self) -> bool:
        return self._closed
    
    def ping(self) -> bool:
        return not self._closed
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at


class ConnectionPool:
    """Thread-safe connection pool."""
    
    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10,
                 max_overflow: int = 5, timeout: float = 30.0,
                 recycle: float = 3600.0, pre_ping: bool = True):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.recycle = recycle
        self.pre_ping = pre_ping
        
        self._pool: List[Connection] = []
        self._overflow_count = 0
        self._lock = threading.Lock()
        self._stats = PoolStats()
        
        # Pre-fill pool
        for _ in range(min_size):
            self._pool.append(self._create_connection())
    
    def _create_connection(self) -> Connection:
        conn = Connection(self.dsn)
        self._stats.connections_created += 1
        return conn
    
    def acquire(self) -> Connection:
        with self._lock:
            # Try to get from pool
            while self._pool:
                conn = self._pool.pop(0)
                
                # Check if connection should be recycled
                if conn.age > self.recycle:
                    conn.close()
                    self._stats.connections_recycled += 1
                    continue
                
                # Ping check
                if self.pre_ping and not conn.ping():
                    conn.close()
                    self._stats.connections_invalidated += 1
                    continue
                
                self._stats.checkouts += 1
                return conn
            
            # Pool empty, try overflow
            total = len(self._pool) + self._overflow_count
            if total < self.max_size + self.max_overflow:
                conn = self._create_connection()
                self._overflow_count += 1
                self._stats.checkouts += 1
                self._stats.overflow_count = max(
                    self._stats.overflow_count, self._overflow_count)
                return conn
            
            raise TimeoutError("Connection pool exhausted")
    
    def release(self, conn: Connection):
        with self._lock:
            if conn.is_closed:
                self._overflow_count = max(0, self._overflow_count - 1)
                return
            
            if len(self._pool) < self.max_size:
                self._pool.append(conn)
                self._stats.checkins += 1
            else:
                conn.close()
                self._overflow_count = max(0, self._overflow_count - 1)
    
    @contextmanager
    def connection(self):
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)
    
    @contextmanager
    def transaction(self):
        conn = self.acquire()
        conn.begin()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release(conn)
    
    def close_all(self):
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
    
    @property
    def stats(self) -> 'PoolStats':
        with self._lock:
            self._stats.pool_size = len(self._pool)
            return self._stats


@dataclass
class PoolStats:
    pool_size: int = 0
    checkouts: int = 0
    checkins: int = 0
    connections_created: int = 0
    connections_recycled: int = 0
    connections_invalidated: int = 0
    overflow_count: int = 0


# ============================================================
# ORM Implementation
# ============================================================

class Field:
    """Base field descriptor for ORM models."""
    
    def __init__(self, field_type: type = str, primary_key: bool = False,
                 nullable: bool = True, default: Any = None,
                 unique: bool = False, index: bool = False):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.unique = unique
        self.index = index
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)
    
    def __set__(self, obj, value):
        if value is not None and not isinstance(value, self.field_type):
            try:
                value = self.field_type(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Expected {self.field_type.__name__} for {self.name}, "
                    f"got {type(value).__name__}")
        if not self.nullable and value is None:
            raise ValueError(f"{self.name} cannot be null")
        obj.__dict__[self.name] = value


class IntegerField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type=int, **kwargs)


class StringField(Field):
    def __init__(self, max_length: int = 255, **kwargs):
        super().__init__(field_type=str, **kwargs)
        self.max_length = max_length
    
    def __set__(self, obj, value):
        if value is not None and len(str(value)) > self.max_length:
            raise ValueError(
                f"{self.name} exceeds max length {self.max_length}")
        super().__set__(obj, value)


class FloatField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type=float, **kwargs)


class BooleanField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type=bool, **kwargs)


class DateTimeField(Field):
    def __init__(self, auto_now: bool = False, auto_now_add: bool = False, **kwargs):
        super().__init__(field_type=float, **kwargs)  # Store as timestamp
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add


class ForeignKey(Field):
    def __init__(self, reference: str, **kwargs):
        super().__init__(field_type=int, **kwargs)
        self.reference = reference


class ModelMeta(type):
    """Metaclass for ORM models."""
    
    _registry: Dict[str, type] = {}
    
    def __new__(mcs, name, bases, namespace):
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                fields[key] = value
        
        namespace['_fields'] = fields
        namespace['_table_name'] = namespace.get(
            '_table_name', name.lower() + 's')
        
        cls = super().__new__(mcs, name, bases, namespace)
        
        if bases:  # Don't register base Model
            mcs._registry[name] = cls
        
        return cls


class Model(metaclass=ModelMeta):
    """Base ORM model."""
    
    _fields: Dict[str, Field] = {}
    _table_name: str = ""
    
    def __init__(self, **kwargs):
        for name, field_obj in self._fields.items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif field_obj.default is not None:
                setattr(self, name, field_obj.default)
    
    def to_dict(self) -> dict:
        return {
            name: getattr(self, name, None)
            for name in self._fields
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Model':
        return cls(**{k: v for k, v in data.items() if k in cls._fields})
    
    def __repr__(self):
        fields = ', '.join(
            f"{name}={getattr(self, name, None)!r}"
            for name in list(self._fields)[:3]
        )
        return f"{self.__class__.__name__}({fields})"


# ============================================================
# Repository Pattern
# ============================================================

class Specification(ABC, Generic[T]):
    """Specification pattern for composable queries."""
    
    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        pass
    
    def __and__(self, other: 'Specification[T]') -> 'Specification[T]':
        return AndSpecification(self, other)
    
    def __or__(self, other: 'Specification[T]') -> 'Specification[T]':
        return OrSpecification(self, other)
    
    def __invert__(self) -> 'Specification[T]':
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    def __init__(self, left: Specification[T], right: Specification[T]):
        self._left = left
        self._right = right
    
    def is_satisfied_by(self, entity: T) -> bool:
        return (self._left.is_satisfied_by(entity) and
                self._right.is_satisfied_by(entity))


class OrSpecification(Specification[T]):
    def __init__(self, left: Specification[T], right: Specification[T]):
        self._left = left
        self._right = right
    
    def is_satisfied_by(self, entity: T) -> bool:
        return (self._left.is_satisfied_by(entity) or
                self._right.is_satisfied_by(entity))


class NotSpecification(Specification[T]):
    def __init__(self, spec: Specification[T]):
        self._spec = spec
    
    def is_satisfied_by(self, entity: T) -> bool:
        return not self._spec.is_satisfied_by(entity)


class Repository(ABC, Generic[T]):
    """Abstract repository."""
    
    @abstractmethod
    def get(self, id: Any) -> Optional[T]:
        pass
    
    @abstractmethod
    def list(self, spec: Specification[T] = None,
             order_by: str = None, limit: int = None) -> List[T]:
        pass
    
    @abstractmethod
    def add(self, entity: T) -> T:
        pass
    
    @abstractmethod
    def update(self, entity: T) -> T:
        pass
    
    @abstractmethod
    def remove(self, id: Any) -> bool:
        pass
    
    @abstractmethod
    def count(self, spec: Specification[T] = None) -> int:
        pass


class InMemoryRepository(Repository[T]):
    """In-memory repository implementation."""
    
    def __init__(self, id_field: str = 'id'):
        self._storage: Dict[Any, T] = {}
        self._id_field = id_field
        self._next_id = 1
    
    def _get_id(self, entity: T) -> Any:
        return getattr(entity, self._id_field, None)
    
    def _set_id(self, entity: T, id_val: Any):
        setattr(entity, self._id_field, id_val)
    
    def get(self, id: Any) -> Optional[T]:
        return self._storage.get(id)
    
    def list(self, spec: Specification[T] = None,
             order_by: str = None, limit: int = None) -> List[T]:
        items = list(self._storage.values())
        
        if spec:
            items = [item for item in items if spec.is_satisfied_by(item)]
        
        if order_by:
            reverse = order_by.startswith('-')
            field_name = order_by.lstrip('-')
            items.sort(
                key=lambda x: getattr(x, field_name, ''),
                reverse=reverse
            )
        
        if limit:
            items = items[:limit]
        
        return items
    
    def add(self, entity: T) -> T:
        if self._get_id(entity) is None:
            self._set_id(entity, self._next_id)
            self._next_id += 1
        self._storage[self._get_id(entity)] = entity
        return entity
    
    def update(self, entity: T) -> T:
        id_val = self._get_id(entity)
        if id_val not in self._storage:
            raise ValueError(f"Entity with id {id_val} not found")
        self._storage[id_val] = entity
        return entity
    
    def remove(self, id: Any) -> bool:
        return self._storage.pop(id, None) is not None
    
    def count(self, spec: Specification[T] = None) -> int:
        if spec is None:
            return len(self._storage)
        return len([e for e in self._storage.values()
                    if spec.is_satisfied_by(e)])


# ============================================================
# Unit of Work
# ============================================================

class UnitOfWork:
    """Tracks changes and commits as transaction."""
    
    def __init__(self):
        self._new: List[Any] = []
        self._dirty: List[Any] = []
        self._removed: List[Any] = []
        self._identity_map: Dict[tuple, Any] = {}
    
    def register_new(self, entity):
        key = self._entity_key(entity)
        if key not in self._identity_map:
            self._new.append(entity)
            self._identity_map[key] = entity
    
    def register_dirty(self, entity):
        key = self._entity_key(entity)
        if entity not in self._new:
            self._dirty.append(entity)
        self._identity_map[key] = entity
    
    def register_removed(self, entity):
        key = self._entity_key(entity)
        if entity in self._new:
            self._new.remove(entity)
        elif entity in self._dirty:
            self._dirty.remove(entity)
        self._removed.append(entity)
        self._identity_map.pop(key, None)
    
    def get(self, entity_type: type, id: Any) -> Optional[Any]:
        return self._identity_map.get((entity_type.__name__, id))
    
    def commit(self, repositories: Dict[type, Repository]):
        try:
            for entity in self._new:
                repo = repositories.get(type(entity))
                if repo:
                    repo.add(entity)
            
            for entity in self._dirty:
                repo = repositories.get(type(entity))
                if repo:
                    repo.update(entity)
            
            for entity in self._removed:
                repo = repositories.get(type(entity))
                if repo:
                    repo.remove(getattr(entity, 'id', None))
            
            self._clear()
        except Exception:
            self.rollback()
            raise
    
    def rollback(self):
        self._clear()
    
    def _clear(self):
        self._new.clear()
        self._dirty.clear()
        self._removed.clear()
    
    def _entity_key(self, entity) -> tuple:
        return (type(entity).__name__, getattr(entity, 'id', id(entity)))
    
    @property  
    def pending_count(self) -> int:
        return len(self._new) + len(self._dirty) + len(self._removed)


# ============================================================
# Query Builder
# ============================================================

class SQLBuilder:
    """Type-safe SQL query builder."""
    
    def __init__(self, table: str):
        self._table = table
        self._select_cols = ['*']
        self._where_clauses = []
        self._params = []
        self._order_by = []
        self._limit = None
        self._offset = None
        self._joins = []
        self._group_by = []
        self._having = []
    
    def select(self, *columns: str) -> 'SQLBuilder':
        self._select_cols = list(columns)
        return self
    
    def where(self, clause: str, *params) -> 'SQLBuilder':
        self._where_clauses.append(clause)
        self._params.extend(params)
        return self
    
    def where_eq(self, column: str, value) -> 'SQLBuilder':
        self._where_clauses.append(f"{column} = ?")
        self._params.append(value)
        return self
    
    def where_in(self, column: str, values: list) -> 'SQLBuilder':
        placeholders = ', '.join(['?'] * len(values))
        self._where_clauses.append(f"{column} IN ({placeholders})")
        self._params.extend(values)
        return self
    
    def where_like(self, column: str, pattern: str) -> 'SQLBuilder':
        self._where_clauses.append(f"{column} LIKE ?")
        self._params.append(pattern)
        return self
    
    def where_between(self, column: str, low, high) -> 'SQLBuilder':
        self._where_clauses.append(f"{column} BETWEEN ? AND ?")
        self._params.extend([low, high])
        return self
    
    def where_null(self, column: str) -> 'SQLBuilder':
        self._where_clauses.append(f"{column} IS NULL")
        return self
    
    def where_not_null(self, column: str) -> 'SQLBuilder':
        self._where_clauses.append(f"{column} IS NOT NULL")
        return self
    
    def join(self, table: str, on: str,
             join_type: str = 'INNER') -> 'SQLBuilder':
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self
    
    def left_join(self, table: str, on: str) -> 'SQLBuilder':
        return self.join(table, on, 'LEFT')
    
    def order_by(self, column: str, desc: bool = False) -> 'SQLBuilder':
        direction = 'DESC' if desc else 'ASC'
        self._order_by.append(f"{column} {direction}")
        return self
    
    def group_by(self, *columns: str) -> 'SQLBuilder':
        self._group_by.extend(columns)
        return self
    
    def having(self, clause: str, *params) -> 'SQLBuilder':
        self._having.append(clause)
        self._params.extend(params)
        return self
    
    def limit(self, n: int) -> 'SQLBuilder':
        self._limit = n
        return self
    
    def offset(self, n: int) -> 'SQLBuilder':
        self._offset = n
        return self
    
    def build_select(self) -> tuple:
        parts = [f"SELECT {', '.join(self._select_cols)}"]
        parts.append(f"FROM {self._table}")
        
        for join in self._joins:
            parts.append(join)
        
        if self._where_clauses:
            parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
        
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
        
        return ' '.join(parts), tuple(self._params)
    
    def build_insert(self, data: dict) -> tuple:
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        sql = f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders})"
        return sql, tuple(data.values())
    
    def build_update(self, data: dict) -> tuple:
        set_clause = ', '.join(f"{k} = ?" for k in data.keys())
        params = list(data.values()) + self._params
        
        sql = f"UPDATE {self._table} SET {set_clause}"
        if self._where_clauses:
            sql += f" WHERE {' AND '.join(self._where_clauses)}"
        
        return sql, tuple(params)
    
    def build_delete(self) -> tuple:
        sql = f"DELETE FROM {self._table}"
        if self._where_clauses:
            sql += f" WHERE {' AND '.join(self._where_clauses)}"
        return sql, tuple(self._params)


# ============================================================
# Migration System
# ============================================================

@dataclass
class Migration:
    version: str
    name: str
    upgrade_sql: List[str]
    downgrade_sql: List[str]
    applied_at: Optional[float] = None


class MigrationManager:
    """Database migration management."""
    
    def __init__(self):
        self._migrations: List[Migration] = []
        self._applied: Dict[str, float] = {}
    
    def add_migration(self, version: str, name: str,
                      upgrade: List[str], downgrade: List[str]):
        self._migrations.append(Migration(
            version=version,
            name=name,
            upgrade_sql=upgrade,
            downgrade_sql=downgrade,
        ))
        self._migrations.sort(key=lambda m: m.version)
    
    def get_pending(self) -> List[Migration]:
        return [m for m in self._migrations
                if m.version not in self._applied]
    
    def upgrade(self, target: str = None) -> List[str]:
        applied = []
        for migration in self._migrations:
            if migration.version in self._applied:
                continue
            if target and migration.version > target:
                break
            
            self._applied[migration.version] = time.time()
            migration.applied_at = self._applied[migration.version]
            applied.append(migration.version)
        
        return applied
    
    def downgrade(self, target: str = None) -> List[str]:
        rolled_back = []
        for migration in reversed(self._migrations):
            if migration.version not in self._applied:
                continue
            if target and migration.version <= target:
                break
            
            del self._applied[migration.version]
            migration.applied_at = None
            rolled_back.append(migration.version)
        
        return rolled_back
    
    def current_version(self) -> Optional[str]:
        if not self._applied:
            return None
        return max(self._applied.keys())
    
    def history(self) -> List[dict]:
        return [
            {
                'version': m.version,
                'name': m.name,
                'applied': m.version in self._applied,
                'applied_at': self._applied.get(m.version),
            }
            for m in self._migrations
        ]`,
				},
			},
		},
	})
}
