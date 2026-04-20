package systemsdesign

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSystemsDesignModules([]problems.CourseModule{
		{
			ID:          2423,
			Title:       "Storage Systems and Data Infrastructure",
			Description: "Master storage engines, database internals, distributed file systems, object storage, data lakes, and data pipeline architectures.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "Storage Engines Distributed Storage and Data Pipelines",
					Content: `Understanding storage systems is fundamental to system design. The choice of storage engine, consistency model, and data pipeline architecture directly impacts system performance and reliability.

**Storage Engine Internals:**

B-Tree Based (Traditional RDBMS):
  In-place updates
  Optimized for reads
  Page-based storage (4KB-16KB)
  Write amplification from page rewrites
  Examples: PostgreSQL, MySQL InnoDB, SQLite
  
  Structure:
    Root → Internal Nodes → Leaf Nodes
    Leaf nodes contain actual data or pointers
    Balanced: O(log N) for reads and writes
    Fan-out: Typically 100-500 keys per node
    
  Write-Ahead Log (WAL):
    All writes recorded in WAL first
    WAL flushed to disk (fsync)
    In case of crash, replay WAL
    Checkpoint: Flush dirty pages, truncate WAL

LSM-Tree Based (NoSQL):
  Append-only writes
  Optimized for writes
  Compaction in background
  Read amplification from multiple levels
  Examples: RocksDB, LevelDB, Cassandra, HBase
  
  Write path:
    Write to in-memory buffer (memtable)
    When full, flush to disk as SSTable (sorted)
    Background compaction merges SSTables
    
  Read path:
    Check memtable
    Check bloom filters for each SSTable level
    Binary search in relevant SSTables
    
  Compaction:
    Size-tiered: Merge similar-size SSTables
    Leveled: Each level is a sorted run
    FIFO: Time-based, oldest deleted
    
  Bloom Filters:
    Probabilistic data structure
    No false negatives, possible false positives
    Check if key might exist in SSTable
    Avoid unnecessary disk reads

**Column-Family Stores:**

Wide-column model:
  Row key → Column families → Columns
  Each column has timestamp
  Sparse: Only store columns that exist
  
  Examples: Cassandra, HBase, Bigtable
  
  Cassandra data model:
    Keyspace → Table → Partition → Row → Column
    Partition key: Determines which node stores data
    Clustering key: Orders rows within partition
    
  Access patterns:
    Efficient: Query by partition key
    Efficient: Range query on clustering key within partition
    Inefficient: Full table scan
    Design table per query pattern

**Document Stores:**

Flexible schema:
  JSON/BSON documents
  Nested objects and arrays
  Schema-on-read
  
  Examples: MongoDB, CouchDB, DynamoDB
  
  MongoDB internals:
    WiredTiger storage engine
    B-tree indexes
    Document-level locking
    Journaling (WAL equivalent)
    Oplog: Replication log
    
  Indexing:
    Single field, compound, multikey (arrays)
    Text index for full-text search
    Geospatial indexes (2d, 2dsphere)
    Hashed index for sharding
    TTL index for auto-expiry

**Distributed File Systems:**

HDFS (Hadoop Distributed File System):
  NameNode: Metadata management
  DataNode: Block storage
  Block size: 128MB default
  Replication: 3 copies across racks
  
  Write: Client → NameNode (metadata) → DataNodes (pipeline)
  Read: Client → NameNode (block locations) → nearest DataNode
  
  Limitations:
    Single NameNode (SPOF, solved by HA)
    Small files problem
    Not suitable for random writes

Object Storage:
  S3-compatible: AWS S3, MinIO, GCS
  Flat namespace with bucket/key
  Immutable objects (versioning optional)
  Eventual consistency (or strong in newer S3)
  
  Architecture:
    Metadata service: Maps keys to object locations
    Data service: Stores object data
    Erasure coding: Fault tolerance with less overhead than replication
    
  Erasure Coding:
    Split data into k fragments
    Generate m parity fragments
    Tolerate m failures
    Less storage overhead than 3x replication
    Example: (10,4) = 10 data + 4 parity = tolerate 4 failures

**Data Pipeline Architectures:**

Lambda Architecture:
  Batch layer: Process all data (MapReduce, Spark)
  Speed layer: Process recent data (Storm, Flink)
  Serving layer: Merge batch + speed results
  
  Pros: Accurate with batch, low-latency with speed
  Cons: Dual code paths, complexity

Kappa Architecture:
  Single stream processing pipeline
  Reprocess by replaying from beginning
  Simpler than Lambda
  
  Pros: Single code path, simpler
  Cons: Reprocessing can be slow

ETL (Extract, Transform, Load):
  Extract: Pull data from sources
  Transform: Clean, normalize, aggregate
  Load: Write to destination (data warehouse)
  
  Tools: Airflow, dbt, Spark, AWS Glue

ELT (Extract, Load, Transform):
  Load raw data into warehouse first
  Transform using warehouse compute
  
  Benefits:
    Faster ingestion
    Transform on powerful warehouse hardware
    Schema-on-read flexibility
    
  Tools: Fivetran, Stitch, Snowflake, BigQuery

Change Data Capture (CDC):
  Capture database changes as events
  Source: Transaction log (binlog, WAL, oplog)
  Destination: Kafka, S3, data warehouse
  
  Tools: Debezium, Maxwell, AWS DMS
  
  Use cases:
    Real-time analytics
    Cache invalidation
    Search index updates
    Microservice data sync

**Data Lake vs Data Warehouse:**

Data Warehouse:
  Structured data
  Schema-on-write
  Optimized for analytics
  SQL interface
  Examples: Snowflake, Redshift, BigQuery

Data Lake:
  Raw data: structured + semi-structured + unstructured
  Schema-on-read
  Cheap storage (S3/HDFS)
  Multiple processing engines
  
  Formats:
    Parquet: Columnar, compressed, efficient analytics
    Avro: Row-based, schema evolution
    ORC: Columnar, Hive optimized
    Delta Lake: ACID on data lake (Parquet + log)
    Iceberg: Open table format (metadata + data)

Data Lakehouse:
  Combines data lake + warehouse
  ACID transactions on data lake
  SQL + ML on same data
  Examples: Delta Lake, Apache Iceberg, Hudi

**Caching Strategies:**

Cache-Aside (Lazy Loading):
  Read: Check cache → miss → read DB → write cache
  Write: Write DB → invalidate cache
  Pros: Only caches what's needed
  Cons: Cache miss penalty, stale data possible

Write-Through:
  Write: Write cache + DB together
  Read: Always from cache
  Pros: Cache always up-to-date
  Cons: Write latency, cache all data

Write-Behind (Write-Back):
  Write: Write to cache only
  Background: Async flush to DB
  Pros: Fast writes
  Cons: Data loss risk, complexity

Read-Through:
  Cache sits between app and DB
  Cache handles DB reads on miss
  Pros: Simplified app code
  Cons: Cache complexity

Refresh-Ahead:
  Predict which entries will be accessed
  Refresh before expiry
  Pros: Low latency on read
  Cons: Wasteful if prediction wrong

Cache eviction:
  LRU: Least Recently Used
  LFU: Least Frequently Used
  TTL: Time To Live
  Random: Simple, decent performance
  ARC: Adaptive Replacement Cache

Distributed caching:
  Client-side: In-process cache (Caffeine, Guava)
  Sidecar: Co-located cache (Redis sidecar)
  Dedicated: Centralized cluster (Redis, Memcached)
  CDN: Edge caching for static content
  
  Consistent hashing for node distribution
  Replication for fault tolerance
  Gossip protocol for membership`,
					CodeExamples: `# Storage Systems and Data Infrastructure Examples

import time
import hashlib
import bisect
import struct
import threading
import os
from typing import Any, Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from enum import Enum

# ============================================================
# LSM-Tree Implementation
# ============================================================

class MemTable:
    """In-memory sorted buffer for LSM tree."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._data: Dict[str, Tuple[str, float, bool]] = {}  # key → (value, timestamp, deleted)
        self._size = 0
    
    def put(self, key: str, value: str):
        self._data[key] = (value, time.time(), False)
        self._size += 1
    
    def get(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, _, deleted = entry
        if deleted:
            return None
        return value
    
    def delete(self, key: str):
        self._data[key] = ("", time.time(), True)
        self._size += 1
    
    @property
    def is_full(self) -> bool:
        return self._size >= self.max_size
    
    def sorted_entries(self) -> List[Tuple[str, str, float, bool]]:
        return sorted(
            [(k, v, ts, d) for k, (v, ts, d) in self._data.items()],
            key=lambda x: x[0])
    
    def clear(self):
        self._data.clear()
        self._size = 0


class SSTable:
    """Sorted String Table (immutable on-disk structure)."""
    
    def __init__(self, table_id: int, level: int = 0):
        self.table_id = table_id
        self.level = level
        self.created_at = time.time()
        self._entries: List[Tuple[str, str, float, bool]] = []
        self._index: Dict[str, int] = {}
        self._bloom: set = set()
        self.min_key: str = ""
        self.max_key: str = ""
        self.size = 0
    
    @classmethod
    def from_memtable(cls, table_id: int, memtable: MemTable) -> 'SSTable':
        table = cls(table_id)
        table._entries = memtable.sorted_entries()
        table._build_index()
        return table
    
    @classmethod
    def from_entries(cls, table_id: int, entries: List[Tuple], level: int = 0) -> 'SSTable':
        table = cls(table_id, level)
        table._entries = sorted(entries, key=lambda x: x[0])
        table._build_index()
        return table
    
    def _build_index(self):
        self._index.clear()
        self._bloom.clear()
        
        for i, (key, _, _, _) in enumerate(self._entries):
            if i % 16 == 0:  # Sparse index every 16 entries
                self._index[key] = i
            self._bloom.add(key)
        
        if self._entries:
            self.min_key = self._entries[0][0]
            self.max_key = self._entries[-1][0]
        self.size = len(self._entries)
    
    def might_contain(self, key: str) -> bool:
        return key in self._bloom
    
    def get(self, key: str) -> Optional[Tuple[str, float, bool]]:
        if not self.might_contain(key):
            return None
        
        # Binary search
        lo, hi = 0, len(self._entries) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._entries[mid][0] == key:
                _, value, ts, deleted = self._entries[mid]
                return value, ts, deleted
            elif self._entries[mid][0] < key:
                lo = mid + 1
            else:
                hi = mid - 1
        
        return None
    
    def scan(self, start_key: str = "", end_key: str = "") -> Iterator:
        for key, value, ts, deleted in self._entries:
            if start_key and key < start_key:
                continue
            if end_key and key > end_key:
                break
            if not deleted:
                yield key, value


class LSMTree:
    """Log-Structured Merge Tree."""
    
    def __init__(self, memtable_size: int = 1000,
                 level0_max: int = 4):
        self._memtable = MemTable(memtable_size)
        self._immutable: Optional[MemTable] = None
        self._levels: List[List[SSTable]] = [[] for _ in range(7)]
        self._next_id = 0
        self._lock = threading.Lock()
        self.level0_max = level0_max
    
    def put(self, key: str, value: str):
        with self._lock:
            self._memtable.put(key, value)
            if self._memtable.is_full:
                self._flush()
    
    def get(self, key: str) -> Optional[str]:
        with self._lock:
            # Check memtable
            result = self._memtable.get(key)
            if result is not None:
                return result
            
            # Check immutable memtable
            if self._immutable:
                result = self._immutable.get(key)
                if result is not None:
                    return result
            
            # Check each level (newest first)
            for level in self._levels:
                for table in reversed(level):
                    entry = table.get(key)
                    if entry is not None:
                        value, _, deleted = entry
                        if deleted:
                            return None
                        return value
            
            return None
    
    def delete(self, key: str):
        with self._lock:
            self._memtable.delete(key)
            if self._memtable.is_full:
                self._flush()
    
    def _flush(self):
        self._immutable = self._memtable
        self._memtable = MemTable(self._memtable.max_size)
        
        self._next_id += 1
        table = SSTable.from_memtable(self._next_id, self._immutable)
        self._levels[0].append(table)
        self._immutable = None
        
        # Trigger compaction if needed
        if len(self._levels[0]) >= self.level0_max:
            self._compact(0)
    
    def _compact(self, level: int):
        if level >= len(self._levels) - 1:
            return
        
        # Merge all tables in current level
        all_entries: Dict[str, Tuple[str, str, float, bool]] = {}
        
        for table in self._levels[level]:
            for entry in table._entries:
                key = entry[0]
                if key not in all_entries or entry[2] > all_entries[key][2]:
                    all_entries[key] = entry
        
        # Add existing entries from next level
        for table in self._levels[level + 1]:
            for entry in table._entries:
                key = entry[0]
                if key not in all_entries or entry[2] > all_entries[key][2]:
                    all_entries[key] = entry
        
        # Create new SSTable at next level
        self._next_id += 1
        entries = sorted(all_entries.values(), key=lambda x: x[0])
        
        # Remove deleted entries during compaction
        entries = [e for e in entries if not e[3]]
        
        new_table = SSTable.from_entries(
            self._next_id, entries, level + 1)
        
        self._levels[level].clear()
        self._levels[level + 1] = [new_table]
    
    @property
    def stats(self) -> Dict[str, Any]:
        level_stats = []
        for i, level in enumerate(self._levels):
            if level:
                level_stats.append({
                    "level": i,
                    "tables": len(level),
                    "total_entries": sum(t.size for t in level),
                })
        return {
            "memtable_size": self._memtable._size,
            "levels": level_stats,
        }


# ============================================================
# LRU Cache
# ============================================================

class LRUCache:
    """Least Recently Used cache."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: float = None):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
                self._cache[key] = value
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)
    
    @property
    def size(self) -> int:
        return len(self._cache)


class LFUCache:
    """Least Frequently Used cache."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: Dict[str, Any] = {}
        self._freq: Dict[str, int] = {}
        self._freq_map: Dict[int, OrderedDict] = defaultdict(OrderedDict)
        self._min_freq = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        freq = self._freq[key]
        del self._freq_map[freq][key]
        if not self._freq_map[freq]:
            del self._freq_map[freq]
            if self._min_freq == freq:
                self._min_freq += 1
        
        self._freq[key] = freq + 1
        self._freq_map[freq + 1][key] = True
        
        return self._cache[key]
    
    def put(self, key: str, value: Any):
        if self.capacity <= 0:
            return
        
        if key in self._cache:
            self._cache[key] = value
            self.get(key)
            return
        
        if len(self._cache) >= self.capacity:
            # Evict least frequent
            evict_key = next(iter(self._freq_map[self._min_freq]))
            del self._freq_map[self._min_freq][evict_key]
            if not self._freq_map[self._min_freq]:
                del self._freq_map[self._min_freq]
            del self._cache[evict_key]
            del self._freq[evict_key]
        
        self._cache[key] = value
        self._freq[key] = 1
        self._freq_map[1][key] = True
        self._min_freq = 1


# ============================================================
# Write-Ahead Log
# ============================================================

@dataclass
class WALEntry:
    sequence: int
    operation: str  # "PUT" or "DELETE"
    key: str
    value: str
    timestamp: float
    checksum: str


class WriteAheadLog:
    """Write-ahead log for crash recovery."""
    
    def __init__(self):
        self._entries: List[WALEntry] = []
        self._sequence = 0
        self._checkpoint_seq = 0
        self._lock = threading.Lock()
    
    def append(self, operation: str, key: str, value: str = "") -> int:
        with self._lock:
            self._sequence += 1
            
            data = f"{self._sequence}:{operation}:{key}:{value}"
            checksum = hashlib.md5(data.encode()).hexdigest()
            
            entry = WALEntry(
                sequence=self._sequence,
                operation=operation,
                key=key,
                value=value,
                timestamp=time.time(),
                checksum=checksum,
            )
            
            self._entries.append(entry)
            return self._sequence
    
    def read_from(self, sequence: int) -> List[WALEntry]:
        with self._lock:
            return [e for e in self._entries if e.sequence > sequence]
    
    def checkpoint(self, sequence: int):
        with self._lock:
            self._checkpoint_seq = sequence
            self._entries = [e for e in self._entries
                           if e.sequence > sequence]
    
    def recover(self) -> List[WALEntry]:
        return self.read_from(self._checkpoint_seq)
    
    def verify_integrity(self) -> bool:
        for entry in self._entries:
            data = f"{entry.sequence}:{entry.operation}:{entry.key}:{entry.value}"
            expected = hashlib.md5(data.encode()).hexdigest()
            if entry.checksum != expected:
                return False
        return True
    
    @property
    def size(self) -> int:
        return len(self._entries)


# ============================================================
# Bloom Filter
# ============================================================

class BloomFilter:
    """Probabilistic set membership test."""
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        import math
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        self._bits = [False] * self.size
        self._count = 0
    
    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        import math
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        import math
        return max(1, int(m / n * math.log(2)))
    
    def _hashes(self, key: str) -> List[int]:
        h1 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]
    
    def add(self, key: str):
        for pos in self._hashes(key):
            self._bits[pos] = True
        self._count += 1
    
    def might_contain(self, key: str) -> bool:
        return all(self._bits[pos] for pos in self._hashes(key))
    
    @property
    def estimated_false_positive_rate(self) -> float:
        import math
        if self._count == 0:
            return 0.0
        return (1 - math.exp(-self.num_hashes * self._count / self.size)) ** self.num_hashes`,
				},
			},
		},
	})
}
