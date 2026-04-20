package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          2420,
			Title:       "Messaging Systems and Event-Driven Architecture",
			Description: "Master message queues, event streaming, pub/sub patterns, event sourcing, CQRS, and building reliable event-driven systems.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Message Queues Event Streaming and Event Sourcing",
					Content: `Event-driven architecture decouples producers and consumers, enabling scalable, resilient, and loosely coupled systems.

**Message Queue vs Event Stream:**

Message Queue (Point-to-Point):
  Producer → Queue → Consumer
  Message consumed by exactly one consumer
  Message removed after consumption
  Ordering: FIFO within queue
  Examples: RabbitMQ, Amazon SQS, ActiveMQ
  
  Use cases:
    Task distribution across workers
    Background job processing
    Request buffering during spikes
    Decoupling microservices

Event Stream (Pub/Sub):
  Producer → Topic → Multiple Consumers
  Events retained for a period (or forever)
  Multiple consumer groups read independently
  Ordering: Per partition
  Examples: Apache Kafka, Amazon Kinesis, Pulsar
  
  Use cases:
    Event sourcing
    Real-time analytics
    Log aggregation
    Change data capture (CDC)
    Stream processing

**Message Queue Patterns:**

Work Queue:
  Multiple workers consume from one queue
  Load distribution across workers
  Acknowledgment ensures at-most-once or at-least-once delivery
  Dead letter queue for failed messages
  
  Retry strategies:
    Fixed delay: Wait N seconds before retry
    Exponential backoff: 1s, 2s, 4s, 8s...
    Exponential with jitter: Backoff + random offset
    Max retries: Move to DLQ after N failures

Request-Reply:
  Request queue + Reply queue
  Correlation ID links request to response
  Timeout handling for missing replies
  Alternative: RPC over messaging

Priority Queue:
  Messages with different priorities
  High priority processed first
  Separate queues per priority level
  Or priority field with queue ordering

Routing:
  Direct: Route to specific queue by routing key
  Topic: Pattern matching on routing key
  Fanout: Broadcast to all bound queues
  Headers: Route based on message headers

**Apache Kafka Architecture:**

Core concepts:
  Topic: Named feed of messages
  Partition: Ordered, immutable sequence of records
  Offset: Unique position in partition
  Broker: Kafka server
  Producer: Writes records to topics
  Consumer: Reads records from topics
  Consumer Group: Set of consumers sharing topic partitions
  
  Cluster layout:
    Multiple brokers
    ZooKeeper/KRaft for coordination
    Replication across brokers
    Leader/follower per partition

Partitioning:
  Round-robin: Even distribution
  Key-based: Hash(key) mod partition_count
  Custom partitioner: Application-specific logic
  
  Ordering guarantees:
    Within partition: Strict ordering
    Across partitions: No ordering guarantee
    Key-based: Same key → same partition → ordered

Replication:
  Replication factor: Number of copies per partition
  ISR (In-Sync Replicas): Replicas caught up with leader
  Leader handles all reads/writes
  Followers replicate from leader
  
  acks configuration:
    acks=0: Fire and forget (fastest, may lose data)
    acks=1: Leader acknowledges (balanced)
    acks=all: All ISR acknowledge (safest, slowest)
  
  min.insync.replicas: Minimum ISR for acks=all
    With RF=3 and min.insync=2:
    Tolerate 1 broker failure with no data loss

Consumer Groups:
  Each partition assigned to one consumer in group
  Rebalancing when consumers join/leave
  Exactly-once semantics with transactions
  
  Offset management:
    Auto-commit: Periodic offset commits
    Manual commit: Application controls
    Earliest: Start from beginning
    Latest: Start from newest

Kafka Streams:
  Stream processing library
  Stateful and stateless operations
  KTables: Changelog stream as table
  Windowing: Tumbling, hopping, sliding, session

**Event Sourcing:**

Principles:
  Store events, not current state
  Events are immutable facts
  Current state derived by replaying events
  Complete audit trail
  
  Event store:
    Append-only log
    Stream per aggregate
    Global ordering (or per-stream)
    Subscriptions for projections

Benefits:
  Complete audit trail of all changes
  Temporal queries (state at any point in time)
  Event replay for debugging
  Natural fit for CQRS
  Easy to add new projections

Challenges:
  Event schema evolution
  Eventual consistency of projections
  Snapshot optimization for long streams
  Complexity of event handling

Patterns:
  Aggregate: Consistency boundary
  Event handler: Process events
  Projection: Read model from events
  Snapshot: Periodic state capture
  Saga: Long-running process across aggregates

**CQRS (Command Query Responsibility Segregation):**

Principles:
  Separate write model (commands) from read model (queries)
  Commands: Change state, return success/failure
  Queries: Return data, no side effects
  
  Architecture:
    Command side: Domain model, business rules, event store
    Query side: Denormalized read models, optimized for queries
    Sync: Events from command side update read models

Benefits:
  Independent scaling of reads and writes
  Optimized data models for each use case
  Different storage technologies per side
  Better separation of concerns

Read model projections:
  Event handler transforms events into read models
  Multiple projections for different query patterns
  Rebuild by replaying all events
  Eventually consistent with write model

**Delivery Guarantees:**

At-most-once:
  Fire and forget
  No retries
  Messages may be lost
  Fastest, simplest
  Use: Metrics, logs where loss is acceptable

At-least-once:
  Acknowledge after processing
  Retry on failure
  Messages may be duplicated
  Consumers must be idempotent
  Use: Most business operations

Exactly-once:
  Hardest to achieve
  Techniques:
    Idempotent consumers + at-least-once
    Transactional outbox + deduplication
    Kafka transactions (producer and consumer)
  Use: Financial transactions, inventory

Idempotency:
  Same operation applied multiple times = same result
  Implementation:
    Idempotency key in message
    Deduplication table
    Conditional updates (optimistic locking)
    Natural idempotency (SET vs INCREMENT)

**Outbox Pattern:**

Problem: Atomically write to DB and publish event
Solution:
  1. Write entity + event to DB in same transaction
  2. Background process reads outbox table
  3. Publishes events to message broker
  4. Marks events as published
  
  Alternative: Change Data Capture (CDC)
    Debezium reads DB transaction log
    Publishes changes to Kafka
    No application code changes needed

**Dead Letter Queue (DLQ):**

  Messages that cannot be processed after N retries
  Moved to separate DLQ for investigation
  Manual or automated retry from DLQ
  Alerting on DLQ growth
  
  Best practices:
    Set reasonable retry limits (3-5)
    Include original error context
    Monitor DLQ depth
    Regular DLQ review process`,
					CodeExamples: `# Event-Driven Architecture Implementation Examples

import time
import json
import hashlib
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque

# ============================================================
# Event Store Implementation
# ============================================================

@dataclass
class Event:
    event_id: str
    stream_id: str
    event_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    version: int
    timestamp: float
    
    def to_dict(self) -> dict:
        return {
            'event_id': self.event_id,
            'stream_id': self.stream_id,
            'event_type': self.event_type,
            'data': self.data,
            'metadata': self.metadata,
            'version': self.version,
            'timestamp': self.timestamp,
        }


class EventStore:
    """In-memory event store."""
    
    def __init__(self):
        self._streams: Dict[str, List[Event]] = defaultdict(list)
        self._global_log: List[Event] = []
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def append(self, stream_id: str, event_type: str,
               data: Dict[str, Any], metadata: Dict[str, Any] = None,
               expected_version: int = None) -> Event:
        with self._lock:
            stream = self._streams[stream_id]
            current_version = len(stream)
            
            if expected_version is not None and current_version != expected_version:
                raise ConcurrencyError(
                    f"Expected version {expected_version}, "
                    f"got {current_version}")
            
            event = Event(
                event_id=str(uuid.uuid4()),
                stream_id=stream_id,
                event_type=event_type,
                data=data,
                metadata=metadata or {},
                version=current_version,
                timestamp=time.time(),
            )
            
            stream.append(event)
            self._global_log.append(event)
            
            # Notify subscribers
            for handler in self._subscribers.get(event_type, []):
                handler(event)
            for handler in self._subscribers.get('*', []):
                handler(event)
            
            return event
    
    def read_stream(self, stream_id: str,
                    start_version: int = 0,
                    max_count: int = None) -> List[Event]:
        with self._lock:
            stream = self._streams.get(stream_id, [])
            events = stream[start_version:]
            if max_count:
                events = events[:max_count]
            return events
    
    def read_all(self, start_position: int = 0,
                 max_count: int = None) -> List[Event]:
        with self._lock:
            events = self._global_log[start_position:]
            if max_count:
                events = events[:max_count]
            return events
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        self._subscribers[event_type].append(handler)
    
    def stream_version(self, stream_id: str) -> int:
        return len(self._streams.get(stream_id, []))


class ConcurrencyError(Exception):
    pass


# ============================================================
# Aggregate with Event Sourcing
# ============================================================

class Aggregate(ABC):
    """Base aggregate with event sourcing."""
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self._pending_events: List[Event] = []
    
    def apply_event(self, event: Event):
        handler_name = f"_apply_{event.event_type}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event.data)
        self.version = event.version + 1
    
    def raise_event(self, event_type: str, data: Dict[str, Any]):
        event = Event(
            event_id=str(uuid.uuid4()),
            stream_id=self.aggregate_id,
            event_type=event_type,
            data=data,
            metadata={},
            version=self.version,
            timestamp=time.time(),
        )
        self._pending_events.append(event)
        self.apply_event(event)
    
    @classmethod
    def load(cls, aggregate_id: str, events: List[Event]) -> 'Aggregate':
        obj = cls(aggregate_id)
        for event in events:
            obj.apply_event(event)
        return obj
    
    @property
    def pending_events(self) -> List[Event]:
        return self._pending_events
    
    def clear_pending(self):
        self._pending_events.clear()


class OrderAggregate(Aggregate):
    """Example: Order aggregate with event sourcing."""
    
    def __init__(self, aggregate_id: str):
        super().__init__(aggregate_id)
        self.status = "new"
        self.items: List[Dict] = []
        self.total = 0.0
        self.customer_id = ""
    
    def create(self, customer_id: str, items: List[Dict]):
        if self.status != "new":
            raise ValueError("Order already created")
        self.raise_event("OrderCreated", {
            "customer_id": customer_id,
            "items": items,
        })
    
    def confirm(self):
        if self.status != "created":
            raise ValueError(f"Cannot confirm order in {self.status}")
        self.raise_event("OrderConfirmed", {})
    
    def ship(self, tracking_number: str):
        if self.status != "confirmed":
            raise ValueError(f"Cannot ship order in {self.status}")
        self.raise_event("OrderShipped", {
            "tracking_number": tracking_number,
        })
    
    def cancel(self, reason: str):
        if self.status in ("shipped", "cancelled"):
            raise ValueError(f"Cannot cancel order in {self.status}")
        self.raise_event("OrderCancelled", {"reason": reason})
    
    def _apply_OrderCreated(self, data: Dict):
        self.status = "created"
        self.customer_id = data["customer_id"]
        self.items = data["items"]
        self.total = sum(i.get("price", 0) * i.get("qty", 1)
                        for i in self.items)
    
    def _apply_OrderConfirmed(self, data: Dict):
        self.status = "confirmed"
    
    def _apply_OrderShipped(self, data: Dict):
        self.status = "shipped"
    
    def _apply_OrderCancelled(self, data: Dict):
        self.status = "cancelled"


# ============================================================
# CQRS Read Model Projection
# ============================================================

class ReadModelProjection(ABC):
    """Base class for read model projections."""
    
    @abstractmethod
    def handle(self, event: Event):
        pass


class OrderSummaryProjection(ReadModelProjection):
    """Projects order events to summary read model."""
    
    def __init__(self):
        self.summaries: Dict[str, Dict] = {}
        self.customer_orders: Dict[str, List[str]] = defaultdict(list)
    
    def handle(self, event: Event):
        handler = getattr(self, f"_on_{event.event_type}", None)
        if handler:
            handler(event)
    
    def _on_OrderCreated(self, event: Event):
        self.summaries[event.stream_id] = {
            "order_id": event.stream_id,
            "customer_id": event.data["customer_id"],
            "item_count": len(event.data["items"]),
            "total": sum(i.get("price", 0) * i.get("qty", 1)
                        for i in event.data["items"]),
            "status": "created",
            "created_at": event.timestamp,
            "updated_at": event.timestamp,
        }
        self.customer_orders[event.data["customer_id"]].append(
            event.stream_id)
    
    def _on_OrderConfirmed(self, event: Event):
        if event.stream_id in self.summaries:
            self.summaries[event.stream_id]["status"] = "confirmed"
            self.summaries[event.stream_id]["updated_at"] = event.timestamp
    
    def _on_OrderShipped(self, event: Event):
        if event.stream_id in self.summaries:
            self.summaries[event.stream_id]["status"] = "shipped"
            self.summaries[event.stream_id]["updated_at"] = event.timestamp
            self.summaries[event.stream_id]["tracking"] = event.data.get(
                "tracking_number")
    
    def _on_OrderCancelled(self, event: Event):
        if event.stream_id in self.summaries:
            self.summaries[event.stream_id]["status"] = "cancelled"
            self.summaries[event.stream_id]["updated_at"] = event.timestamp
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        return self.summaries.get(order_id)
    
    def get_customer_orders(self, customer_id: str) -> List[Dict]:
        order_ids = self.customer_orders.get(customer_id, [])
        return [self.summaries[oid] for oid in order_ids
                if oid in self.summaries]
    
    def get_orders_by_status(self, status: str) -> List[Dict]:
        return [s for s in self.summaries.values()
                if s["status"] == status]


# ============================================================
# Message Queue Implementation
# ============================================================

@dataclass
class Message:
    id: str
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = 0
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0
    correlation_id: Optional[str] = None


class MessageQueue:
    """In-memory message queue with DLQ."""
    
    def __init__(self, name: str, max_retries: int = 3):
        self.name = name
        self.max_retries = max_retries
        self._queue: deque = deque()
        self._dlq: deque = deque()
        self._in_flight: Dict[str, Message] = {}
        self._lock = threading.Lock()
        self._stats = QueueStats()
    
    def publish(self, body: Any, headers: Dict[str, str] = None,
                priority: int = 0, correlation_id: str = None) -> str:
        msg = Message(
            id=str(uuid.uuid4()),
            body=body,
            headers=headers or {},
            timestamp=time.time(),
            max_retries=self.max_retries,
            priority=priority,
            correlation_id=correlation_id,
        )
        
        with self._lock:
            self._queue.append(msg)
            self._stats.published += 1
        
        return msg.id
    
    def consume(self) -> Optional[Message]:
        with self._lock:
            if not self._queue:
                return None
            
            msg = self._queue.popleft()
            self._in_flight[msg.id] = msg
            self._stats.consumed += 1
            return msg
    
    def acknowledge(self, message_id: str):
        with self._lock:
            self._in_flight.pop(message_id, None)
            self._stats.acknowledged += 1
    
    def reject(self, message_id: str, requeue: bool = True):
        with self._lock:
            msg = self._in_flight.pop(message_id, None)
            if msg is None:
                return
            
            msg.retry_count += 1
            
            if requeue and msg.retry_count < msg.max_retries:
                self._queue.append(msg)
                self._stats.requeued += 1
            else:
                self._dlq.append(msg)
                self._stats.dead_lettered += 1
    
    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._queue)
    
    @property
    def dlq_depth(self) -> int:
        with self._lock:
            return len(self._dlq)
    
    def drain_dlq(self) -> List[Message]:
        with self._lock:
            messages = list(self._dlq)
            self._dlq.clear()
            return messages


@dataclass
class QueueStats:
    published: int = 0
    consumed: int = 0
    acknowledged: int = 0
    requeued: int = 0
    dead_lettered: int = 0


# ============================================================
# Pub/Sub with Topic Partitions
# ============================================================

@dataclass
class Record:
    key: Optional[str]
    value: Any
    partition: int = 0
    offset: int = 0
    timestamp: float = 0


class Partition:
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.records: List[Record] = []
        self._lock = threading.Lock()
    
    def append(self, record: Record) -> int:
        with self._lock:
            record.offset = len(self.records)
            record.partition = self.partition_id
            record.timestamp = time.time()
            self.records.append(record)
            return record.offset
    
    def read(self, offset: int, max_records: int = 100) -> List[Record]:
        with self._lock:
            return self.records[offset:offset + max_records]
    
    @property
    def latest_offset(self) -> int:
        return len(self.records)


class Topic:
    """Kafka-like topic with partitions."""
    
    def __init__(self, name: str, num_partitions: int = 3,
                 replication_factor: int = 1):
        self.name = name
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.partitions = [Partition(i) for i in range(num_partitions)]
    
    def _partition_for_key(self, key: Optional[str]) -> int:
        if key is None:
            return hash(time.time()) % self.num_partitions
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_partitions
    
    def produce(self, key: Optional[str], value: Any) -> Tuple[int, int]:
        partition_id = self._partition_for_key(key)
        record = Record(key=key, value=value)
        offset = self.partitions[partition_id].append(record)
        return partition_id, offset
    
    def consume(self, partition_id: int, offset: int,
                max_records: int = 100) -> List[Record]:
        if 0 <= partition_id < self.num_partitions:
            return self.partitions[partition_id].read(offset, max_records)
        return []


class ConsumerGroup:
    """Consumer group with partition assignment."""
    
    def __init__(self, group_id: str, topic: Topic):
        self.group_id = group_id
        self.topic = topic
        self.consumers: List[str] = []
        self.assignments: Dict[str, List[int]] = {}
        self.offsets: Dict[int, int] = {
            i: 0 for i in range(topic.num_partitions)}
    
    def join(self, consumer_id: str):
        if consumer_id not in self.consumers:
            self.consumers.append(consumer_id)
            self._rebalance()
    
    def leave(self, consumer_id: str):
        if consumer_id in self.consumers:
            self.consumers.remove(consumer_id)
            self._rebalance()
    
    def _rebalance(self):
        self.assignments.clear()
        if not self.consumers:
            return
        
        partitions = list(range(self.topic.num_partitions))
        for i, partition in enumerate(partitions):
            consumer = self.consumers[i % len(self.consumers)]
            if consumer not in self.assignments:
                self.assignments[consumer] = []
            self.assignments[consumer].append(partition)
    
    def poll(self, consumer_id: str,
             max_records: int = 100) -> List[Record]:
        assigned = self.assignments.get(consumer_id, [])
        records = []
        
        for partition_id in assigned:
            offset = self.offsets.get(partition_id, 0)
            new_records = self.topic.consume(
                partition_id, offset, max_records)
            records.extend(new_records)
            
            if new_records:
                self.offsets[partition_id] = (
                    new_records[-1].offset + 1)
        
        return records
    
    def commit(self, partition_id: int, offset: int):
        self.offsets[partition_id] = offset


# ============================================================
# Outbox Pattern
# ============================================================

class OutboxEntry:
    def __init__(self, aggregate_id: str, event_type: str,
                 payload: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.aggregate_id = aggregate_id
        self.event_type = event_type
        self.payload = payload
        self.created_at = time.time()
        self.published = False
        self.published_at: Optional[float] = None


class TransactionalOutbox:
    """Outbox pattern for reliable event publishing."""
    
    def __init__(self):
        self._entries: List[OutboxEntry] = []
        self._lock = threading.Lock()
    
    def save_with_event(self, aggregate_id: str, event_type: str,
                        payload: Dict[str, Any]) -> OutboxEntry:
        entry = OutboxEntry(aggregate_id, event_type, payload)
        with self._lock:
            self._entries.append(entry)
        return entry
    
    def get_unpublished(self, batch_size: int = 100) -> List[OutboxEntry]:
        with self._lock:
            unpublished = [e for e in self._entries if not e.published]
            return unpublished[:batch_size]
    
    def mark_published(self, entry_ids: List[str]):
        with self._lock:
            now = time.time()
            for entry in self._entries:
                if entry.id in entry_ids:
                    entry.published = True
                    entry.published_at = now
    
    def cleanup(self, older_than: float):
        with self._lock:
            cutoff = time.time() - older_than
            self._entries = [
                e for e in self._entries
                if not e.published or e.published_at > cutoff
            ]


# ============================================================
# Saga Orchestrator with Compensation
# ============================================================

class SagaStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


@dataclass
class SagaStepDef:
    name: str
    action: Callable[[Dict], Dict]
    compensation: Callable[[Dict], None]
    status: str = "pending"
    result: Optional[Dict] = None
    error: Optional[str] = None


class SagaManager:
    """Manages saga execution with compensating transactions."""
    
    def __init__(self, saga_id: str):
        self.saga_id = saga_id
        self.steps: List[SagaStepDef] = []
        self.context: Dict[str, Any] = {}
        self.status = SagaStatus.RUNNING
        self.history: List[Dict] = []
    
    def step(self, name: str, action: Callable, compensation: Callable):
        self.steps.append(SagaStepDef(
            name=name, action=action, compensation=compensation))
        return self
    
    def execute(self) -> bool:
        completed = []
        
        for step_def in self.steps:
            step_def.status = "running"
            self._log(step_def.name, "started")
            
            try:
                result = step_def.action(self.context)
                step_def.result = result or {}
                step_def.status = "completed"
                self.context.update(step_def.result)
                completed.append(step_def)
                self._log(step_def.name, "completed")
                
            except Exception as e:
                step_def.status = "failed"
                step_def.error = str(e)
                self._log(step_def.name, f"failed: {e}")
                
                self.status = SagaStatus.COMPENSATING
                self._compensate(completed)
                return False
        
        self.status = SagaStatus.COMPLETED
        return True
    
    def _compensate(self, completed: List[SagaStepDef]):
        for step_def in reversed(completed):
            try:
                step_def.compensation(self.context)
                step_def.status = "compensated"
                self._log(step_def.name, "compensated")
            except Exception as e:
                step_def.status = "compensation_failed"
                self._log(step_def.name, f"compensation failed: {e}")
                self.status = SagaStatus.FAILED
                return
        
        self.status = SagaStatus.COMPENSATED
    
    def _log(self, step: str, action: str):
        self.history.append({
            "saga_id": self.saga_id,
            "step": step,
            "action": action,
            "timestamp": time.time(),
        })`,
				},
			},
		},
		{
			ID:          2421,
			Title:       "System Design Interview Patterns",
			Description: "Master URL shortener, Twitter feed, chat system, search engine, notification system, and other common system design problems.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Common System Design Problems and Solutions",
					Content: `System design interviews test the ability to design large-scale distributed systems. Here are common problems and their solution approaches.

**URL Shortener (TinyURL):**

Requirements:
  Functional: Shorten URL, redirect, custom alias, expiry
  Non-functional: Low latency (~100ms), high availability, ~100M URL/day

Estimation:
  Write: 100M/day = ~1200 URL/s
  Read: 10:1 ratio = 12000 redirect/s
  Storage: 100M × 365 × 5 years × 500 bytes = ~91 TB
  
Key generation:
  Base62 encoding: [a-zA-Z0-9] = 62^7 = 3.5 trillion combinations
  Counter-based: Sequential, globally unique (coordination needed)
  Hash-based: MD5/SHA256 → take first 7 chars (collision handling needed)
  Pre-generated keys: Background service generates keys, store in DB
  
Architecture:
  Client → Load Balancer → API Servers → Cache (Redis) → Database
  
  Write path:
    Generate short URL
    Store (short_url, long_url, created, expiry) in DB
    
  Read path:
    Look up cache first
    If miss, query DB
    Cache long URL
    Return 301 (permanent) or 302 (temporary) redirect
  
  Database:
    Short URL as primary key
    Range-based sharding on short URL prefix
    Read replicas for redirect performance

**News Feed / Timeline:**

Requirements:
  Post creation: 500 posts/s
  Feed generation: 50,000 feed reads/s
  Rich media (images, videos, links)
  Ranking and personalization

Feed Generation Approaches:

  Fan-out on write (Push model):
    On new post → write to all followers' feeds
    Pros: Fast read, feed pre-computed
    Cons: Slow write for users with many followers
    Use for: Users with < 5K followers
    
  Fan-out on read (Pull model):
    On feed request → query all followed users' posts
    Pros: Fast write, no wasted work
    Cons: Slow read, heavy on-demand processing
    Use for: Celebrity users (millions of followers)
    
  Hybrid:
    Push for regular users
    Pull for celebrities
    Best of both approaches

Architecture:
  Post Service: Store posts, handle media
  Fan-out Service: Push posts to follower feeds
  Feed Service: Merge and rank feed items
  Cache: Pre-computed feeds in Redis
  
  Feed cache:
    User ID → sorted set of (post_id, score)
    Keep last 200-500 posts per user
    Merge with pull results for celebrities

Ranking:
  Recency: Time decay function
  Engagement: Likes, comments, shares
  Relationship: Close friends scored higher
  Content type: User preferences
  ML model: Combine features for relevance score

**Chat Application (WhatsApp/Slack):**

Requirements:
  1-on-1 messaging
  Group chat (up to 100K members)
  Online/offline status
  Read receipts
  Media sharing
  50M daily active users

Architecture:
  Chat Server: WebSocket connections
  Message Service: Store and deliver messages
  Presence Service: Online/offline status
  Group Service: Manage group membership
  Media Service: Upload/download media
  Push Notification: Offline message delivery

Message flow:
  1. Client A sends message via WebSocket
  2. Chat server stores in Message Service
  3. Message Service checks if recipient online
  4. If online: Route via WebSocket to recipient's chat server
  5. If offline: Queue for push notification
  6. Recipient acknowledges receipt

Data model:
  Messages table: message_id, channel_id, sender_id, content, 
                  created_at, type
  Channels table: channel_id, type (direct/group), created_at
  Channel Members: channel_id, user_id, role, last_read_at

Message storage:
  Recent messages: Key-value store (Redis/Cassandra)
  Historical messages: Columnar store
  Message ID: Snowflake ID (time-sortable, unique)
  Partition: By channel_id for locality

Group messaging:
  Small groups (< 100): Fan-out on write
  Large groups: Fan-out on read with caching
  Channel per group in message broker

**Search Engine:**

Components:
  Web Crawler: Discover and fetch pages
  Indexer: Parse and build inverted index
  Query Processor: Parse and execute queries
  Ranker: Score and order results
  Result Composer: Format and return results

Inverted Index:
  Term → List of (document_id, positions, frequency)
  Compressed posting lists
  Term dictionaries with trie/B-tree
  
  Building:
    Tokenize documents
    Normalize (lowercase, stemming, lemmatizing)
    Build term-document mapping
    Sort and compress posting lists

Ranking (simplified PageRank):
  Content relevance: TF-IDF, BM25
  Link analysis: PageRank, HITS
  User signals: Click-through rate
  Freshness: Recent content bonus
  Quality: Domain authority

**Notification System:**

Types:
  Push notification (mobile)
  Email
  SMS
  In-app notification
  Webhook

Architecture:
  Notification Service: Receive and route notifications
  Template Service: Generate content from templates
  Priority Queue: Handle different urgency levels
  Rate Limiter: Prevent notification spam
  Delivery Service: Platform-specific delivery
  Analytics: Track delivery, open, click rates

Delivery guarantees:
  Store notification in DB before sending
  Mark as sent after successful delivery
  Retry failed deliveries with backoff
  DLQ for permanent failures

User preferences:
  Per-channel opt-in/opt-out
  Quiet hours
  Frequency capping
  Priority thresholds

**Rate Limiter:**

Algorithms:
  Token bucket: Smooth, allows bursts up to capacity
  Leaky bucket: Fixed rate output
  Fixed window: Count per time window (boundary burst issue)
  Sliding window log: Exact, expensive memory
  Sliding window counter: Approximate, memory efficient

Distributed rate limiting:
  Redis + Lua scripts for atomic operations
  Race condition handling with Redis MULTI
  Synchronization across multiple nodes
  Sticky sessions for local rate limiting

Configuration:
  Per-user, per-API, per-IP limits
  Different tiers (free, paid)
  Response: 429 Too Many Requests
  Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset

**Key Design Principles:**

Estimation framework:
  Users → DAU → Peak QPS → Storage → Bandwidth
  Over-provision by 2-3x for peaks
  Round up for simplicity

Start from requirements:
  Functional: What does the system do?
  Non-functional: Scale, latency, availability, consistency

API first:
  Define core APIs before architecture
  RESTful or gRPC
  Pagination, filtering, sorting

Data model:
  Identify entities and relationships
  Choose SQL vs NoSQL based on access patterns
  Plan for sharding and replication

Trade-offs:
  Consistency vs availability
  Latency vs throughput
  Cost vs performance
  Simplicity vs flexibility`,
					CodeExamples: `# System Design Pattern Implementation Examples

import time
import hashlib
import random
import string
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod

# ============================================================
# URL Shortener
# ============================================================

class URLShortener:
    """URL shortening service."""
    
    BASE62 = string.ascii_lowercase + string.ascii_uppercase + string.digits
    
    def __init__(self):
        self._url_map: Dict[str, URLEntry] = {}
        self._reverse_map: Dict[str, str] = {}
        self._counter = 1000000
        self._lock = threading.Lock()
    
    def shorten(self, long_url: str, custom_alias: str = None,
                ttl_seconds: float = None) -> str:
        with self._lock:
            # Check if already shortened
            if long_url in self._reverse_map:
                return self._reverse_map[long_url]
            
            if custom_alias:
                if custom_alias in self._url_map:
                    raise ValueError("Alias already taken")
                short_code = custom_alias
            else:
                short_code = self._generate_code()
            
            expiry = None
            if ttl_seconds:
                expiry = time.time() + ttl_seconds
            
            entry = URLEntry(
                short_code=short_code,
                long_url=long_url,
                created_at=time.time(),
                expires_at=expiry,
            )
            
            self._url_map[short_code] = entry
            self._reverse_map[long_url] = short_code
            return short_code
    
    def resolve(self, short_code: str) -> Optional[str]:
        entry = self._url_map.get(short_code)
        if entry is None:
            return None
        
        if entry.expires_at and time.time() > entry.expires_at:
            del self._url_map[short_code]
            self._reverse_map.pop(entry.long_url, None)
            return None
        
        entry.click_count += 1
        entry.last_accessed = time.time()
        return entry.long_url
    
    def _generate_code(self) -> str:
        self._counter += 1
        return self._to_base62(self._counter)
    
    def _to_base62(self, num: int) -> str:
        if num == 0:
            return self.BASE62[0]
        result = []
        while num > 0:
            result.append(self.BASE62[num % 62])
            num //= 62
        return ''.join(reversed(result))
    
    def stats(self, short_code: str) -> Optional[Dict]:
        entry = self._url_map.get(short_code)
        if entry is None:
            return None
        return {
            "short_code": entry.short_code,
            "long_url": entry.long_url,
            "click_count": entry.click_count,
            "created_at": entry.created_at,
            "last_accessed": entry.last_accessed,
        }


@dataclass
class URLEntry:
    short_code: str
    long_url: str
    created_at: float
    expires_at: Optional[float] = None
    click_count: int = 0
    last_accessed: Optional[float] = None


# ============================================================
# News Feed System
# ============================================================

@dataclass
class Post:
    post_id: str
    author_id: str
    content: str
    created_at: float
    likes: int = 0
    comments: int = 0
    media_urls: List[str] = field(default_factory=list)


class NewsFeedService:
    """News feed with hybrid push/pull."""
    
    CELEBRITY_THRESHOLD = 5000
    FEED_CACHE_SIZE = 200
    
    def __init__(self):
        self._posts: Dict[str, Post] = {}
        self._user_posts: Dict[str, List[str]] = defaultdict(list)
        self._followers: Dict[str, Set[str]] = defaultdict(set)
        self._following: Dict[str, Set[str]] = defaultdict(set)
        self._feed_cache: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.FEED_CACHE_SIZE))
    
    def follow(self, follower_id: str, followee_id: str):
        self._followers[followee_id].add(follower_id)
        self._following[follower_id].add(followee_id)
    
    def unfollow(self, follower_id: str, followee_id: str):
        self._followers[followee_id].discard(follower_id)
        self._following[follower_id].discard(followee_id)
    
    def create_post(self, author_id: str, content: str,
                    media_urls: List[str] = None) -> Post:
        post = Post(
            post_id=str(uuid.uuid4()),
            author_id=author_id,
            content=content,
            created_at=time.time(),
            media_urls=media_urls or [],
        )
        
        self._posts[post.post_id] = post
        self._user_posts[author_id].append(post.post_id)
        
        # Fan-out on write for non-celebrities
        follower_count = len(self._followers.get(author_id, set()))
        if follower_count < self.CELEBRITY_THRESHOLD:
            self._fan_out_write(post)
        
        return post
    
    def _fan_out_write(self, post: Post):
        for follower_id in self._followers.get(post.author_id, set()):
            self._feed_cache[follower_id].appendleft(post.post_id)
    
    def get_feed(self, user_id: str, count: int = 20) -> List[Post]:
        # Get cached feed (push model results)
        cached_ids = list(self._feed_cache.get(user_id, deque()))
        
        # Fan-out on read for celebrities
        celebrity_ids = []
        for followee_id in self._following.get(user_id, set()):
            if len(self._followers.get(followee_id, set())) >= self.CELEBRITY_THRESHOLD:
                celebrity_ids.extend(
                    self._user_posts.get(followee_id, [])[-20:])
        
        # Merge and deduplicate
        all_ids = list(dict.fromkeys(cached_ids + celebrity_ids))
        
        # Get posts and sort by time
        posts = []
        for pid in all_ids:
            post = self._posts.get(pid)
            if post:
                posts.append(post)
        
        posts.sort(key=lambda p: p.created_at, reverse=True)
        return posts[:count]
    
    def _rank_posts(self, posts: List[Post], user_id: str) -> List[Post]:
        def score(post: Post) -> float:
            age_hours = (time.time() - post.created_at) / 3600
            recency = 1.0 / (1.0 + age_hours)
            engagement = (post.likes * 1.0 + post.comments * 2.0)
            is_close = 1.5 if post.author_id in self._following.get(
                user_id, set()) else 1.0
            return recency * 100 + engagement * is_close
        
        return sorted(posts, key=score, reverse=True)


# ============================================================
# Chat Service
# ============================================================

@dataclass
class ChatMessage:
    message_id: str
    channel_id: str
    sender_id: str
    content: str
    timestamp: float
    msg_type: str = "text"
    read_by: Set[str] = field(default_factory=set)


@dataclass
class Channel:
    channel_id: str
    channel_type: str  # "direct" or "group"
    members: Set[str]
    name: Optional[str] = None
    created_at: float = 0


class ChatService:
    """Simple chat service."""
    
    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        self._messages: Dict[str, List[ChatMessage]] = defaultdict(list)
        self._user_channels: Dict[str, Set[str]] = defaultdict(set)
        self._online_users: Set[str] = set()
        self._message_handlers: Dict[str, Callable] = {}
    
    def create_direct_channel(self, user1: str, user2: str) -> Channel:
        channel_id = f"dm:{min(user1, user2)}:{max(user1, user2)}"
        
        if channel_id in self._channels:
            return self._channels[channel_id]
        
        channel = Channel(
            channel_id=channel_id,
            channel_type="direct",
            members={user1, user2},
            created_at=time.time(),
        )
        
        self._channels[channel_id] = channel
        self._user_channels[user1].add(channel_id)
        self._user_channels[user2].add(channel_id)
        return channel
    
    def create_group_channel(self, name: str,
                            members: Set[str]) -> Channel:
        channel = Channel(
            channel_id=str(uuid.uuid4()),
            channel_type="group",
            members=members,
            name=name,
            created_at=time.time(),
        )
        
        self._channels[channel.channel_id] = channel
        for member in members:
            self._user_channels[member].add(channel.channel_id)
        
        return channel
    
    def send_message(self, channel_id: str, sender_id: str,
                     content: str, msg_type: str = "text") -> ChatMessage:
        channel = self._channels.get(channel_id)
        if channel is None:
            raise ValueError("Channel not found")
        
        if sender_id not in channel.members:
            raise ValueError("Not a member of this channel")
        
        message = ChatMessage(
            message_id=str(uuid.uuid4()),
            channel_id=channel_id,
            sender_id=sender_id,
            content=content,
            timestamp=time.time(),
            msg_type=msg_type,
        )
        message.read_by.add(sender_id)
        
        self._messages[channel_id].append(message)
        
        # Deliver to online members
        for member in channel.members:
            if member != sender_id and member in self._online_users:
                handler = self._message_handlers.get(member)
                if handler:
                    handler(message)
        
        return message
    
    def get_messages(self, channel_id: str,
                     before: float = None,
                     limit: int = 50) -> List[ChatMessage]:
        messages = self._messages.get(channel_id, [])
        if before:
            messages = [m for m in messages if m.timestamp < before]
        return messages[-limit:]
    
    def mark_read(self, channel_id: str, user_id: str,
                  up_to_message_id: str):
        messages = self._messages.get(channel_id, [])
        for msg in messages:
            msg.read_by.add(user_id)
            if msg.message_id == up_to_message_id:
                break
    
    def set_online(self, user_id: str):
        self._online_users.add(user_id)
    
    def set_offline(self, user_id: str):
        self._online_users.discard(user_id)
    
    def get_unread_count(self, user_id: str,
                        channel_id: str) -> int:
        messages = self._messages.get(channel_id, [])
        return sum(1 for m in messages if user_id not in m.read_by)


# ============================================================
# Snowflake ID Generator
# ============================================================

class SnowflakeIDGenerator:
    """Twitter Snowflake-like distributed ID generator."""
    
    EPOCH = 1609459200000  # 2021-01-01 00:00:00 UTC in ms
    WORKER_BITS = 10
    SEQUENCE_BITS = 12
    
    MAX_WORKER_ID = (1 << WORKER_BITS) - 1
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1
    
    def __init__(self, worker_id: int):
        if worker_id < 0 or worker_id > self.MAX_WORKER_ID:
            raise ValueError(f"Worker ID must be 0-{self.MAX_WORKER_ID}")
        
        self.worker_id = worker_id
        self._sequence = 0
        self._last_timestamp = -1
        self._lock = threading.Lock()
    
    def _current_millis(self) -> int:
        return int(time.time() * 1000)
    
    def generate(self) -> int:
        with self._lock:
            timestamp = self._current_millis()
            
            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & self.MAX_SEQUENCE
                if self._sequence == 0:
                    # Wait for next millisecond
                    while timestamp <= self._last_timestamp:
                        timestamp = self._current_millis()
            else:
                self._sequence = 0
            
            self._last_timestamp = timestamp
            
            return (
                ((timestamp - self.EPOCH) << (self.WORKER_BITS + self.SEQUENCE_BITS)) |
                (self.worker_id << self.SEQUENCE_BITS) |
                self._sequence
            )
    
    @classmethod
    def extract_timestamp(cls, snowflake_id: int) -> float:
        timestamp_ms = (snowflake_id >> (cls.WORKER_BITS + cls.SEQUENCE_BITS)) + cls.EPOCH
        return timestamp_ms / 1000.0
    
    @classmethod
    def extract_worker_id(cls, snowflake_id: int) -> int:
        return (snowflake_id >> cls.SEQUENCE_BITS) & cls.MAX_WORKER_ID`,
				},
			},
		},
	})
}
