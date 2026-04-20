package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2217,
			Title:       "Python Concurrency and Parallelism",
			Description: "Master threading, multiprocessing, asyncio, concurrent.futures, and advanced concurrency patterns in Python.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Threading Multiprocessing AsyncIO and Concurrency Patterns",
					Content: `Python offers multiple approaches to concurrency: threading for I/O-bound tasks, multiprocessing for CPU-bound tasks, and asyncio for high-concurrency I/O.

**The Global Interpreter Lock (GIL):**

CPython's GIL allows only one thread to execute Python bytecode at a time. This means:
  - Threading does NOT provide true parallelism for CPU-bound tasks
  - Threading IS useful for I/O-bound tasks (network, file, database)
  - Multiprocessing bypasses the GIL (separate processes)
  - asyncio uses cooperative multitasking (single thread)
  
Note: Python 3.13+ includes experimental free-threaded mode (no GIL).

**Threading Module:**

Thread Creation:
  threading.Thread(target=func, args=())
  thread.start() — begins execution
  thread.join() — wait for completion
  thread.daemon = True — dies when main thread exits
  thread.is_alive() — check if running

Thread Synchronization:
  Lock: Mutual exclusion
    lock = threading.Lock()
    lock.acquire() / lock.release()
    with lock: (context manager)
    
  RLock: Reentrant lock (same thread can acquire multiple times)
    rlock = threading.RLock()
    
  Semaphore: Limit concurrent access
    sem = threading.Semaphore(value=5)
    with sem: (allows up to 5 concurrent)
    
  BoundedSemaphore: Error if released more than acquired
    
  Event: Thread signaling
    event = threading.Event()
    event.set() — signal
    event.wait() — block until signaled
    event.clear() — reset
    event.is_set() — check status
    
  Condition: Wait for complex conditions
    cond = threading.Condition()
    cond.wait() — wait for notification
    cond.notify() — wake one waiter
    cond.notify_all() — wake all waiters
    
  Barrier: Synchronize N threads
    barrier = threading.Barrier(parties=3)
    barrier.wait() — block until all parties arrive

Thread-Safe Data Structures:
  queue.Queue: FIFO (thread-safe)
  queue.LifoQueue: LIFO
  queue.PriorityQueue: Priority-based
  
  Methods:
    q.put(item) — add item (blocks if full)
    q.get() — remove item (blocks if empty)
    q.task_done() — signal completion
    q.join() — wait for all tasks
    q.empty(), q.full(), q.qsize()

Thread Pool:
  concurrent.futures.ThreadPoolExecutor
  Manages a pool of worker threads
  submit() — returns Future
  map() — parallel map
  shutdown() — clean up

**Multiprocessing Module:**

Process Creation:
  multiprocessing.Process(target=func, args=())
  process.start() / process.join()
  process.terminate() / process.kill()
  process.exitcode — return code
  
Inter-Process Communication:
  Queue: multiprocessing.Queue()
    Process-safe FIFO queue
    q.put() / q.get()
    
  Pipe: multiprocessing.Pipe()
    Two-way communication
    conn1, conn2 = Pipe()
    conn.send() / conn.recv()
    
  Value/Array: Shared memory
    val = multiprocessing.Value('i', 0)
    arr = multiprocessing.Array('d', [1.0, 2.0])
    
  Manager: Shared objects
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_list = manager.list()

Process Pool:
  multiprocessing.Pool(processes=4)
  pool.map(func, iterable)
  pool.starmap(func, [(arg1, arg2), ...])
  pool.apply_async(func, args)
  pool.close() / pool.join()
  
  concurrent.futures.ProcessPoolExecutor
  Same API as ThreadPoolExecutor

Shared Memory (Python 3.8+):
  multiprocessing.shared_memory.SharedMemory
  Name-based shared memory blocks
  Useful for large data (NumPy arrays)

**asyncio — Asynchronous I/O:**

Core Concepts:
  Coroutine: async def function
  Task: Scheduled coroutine
  Event Loop: Runs coroutines
  Future: Placeholder for eventual result
  
Running:
  asyncio.run(main()) — entry point
  await coroutine — wait for result
  asyncio.create_task(coro) — schedule task
  asyncio.gather(*coros) — run concurrently
  asyncio.wait(tasks) — flexible waiting
  
  asyncio.sleep(seconds) — non-blocking sleep
  asyncio.shield(coro) — protect from cancellation
  asyncio.wait_for(coro, timeout) — with timeout
  
Task Groups (Python 3.11+):
  async with asyncio.TaskGroup() as tg:
      task1 = tg.create_task(coro1())
      task2 = tg.create_task(coro2())
  # Both tasks complete or all cancelled on error
  
Synchronization (asyncio versions):
  asyncio.Lock()
  asyncio.Event()
  asyncio.Semaphore(value)
  asyncio.Condition()
  asyncio.Queue()
  asyncio.Barrier(parties)
  
Streams:
  reader, writer = await asyncio.open_connection(host, port)
  data = await reader.read(1024)
  writer.write(data)
  await writer.drain()
  
  server = await asyncio.start_server(handler, host, port)
  
Subprocess:
  proc = await asyncio.create_subprocess_exec(...)
  proc = await asyncio.create_subprocess_shell(...)
  stdout, stderr = await proc.communicate()

**concurrent.futures:**

Common API:
  executor.submit(fn, *args) → Future
  executor.map(fn, *iterables)
  executor.shutdown(wait=True)
  
Future Methods:
  future.result(timeout=None) — get result
  future.exception() — get exception
  future.done() — check if completed
  future.cancel() — attempt to cancel
  future.add_done_callback(fn) — callback on completion
  
Waiting:
  concurrent.futures.wait(futures, return_when=FIRST_COMPLETED)
  concurrent.futures.as_completed(futures) — iterate as done

**Advanced Patterns:**

Producer-Consumer:
  Producers add items to queue
  Consumers take items from queue
  Thread-safe with queue.Queue
  Async with asyncio.Queue

Worker Pool:
  Fixed number of workers
  Tasks distributed from queue
  Graceful shutdown with sentinel values
  
Pipeline:
  Stage 1 → Queue → Stage 2 → Queue → Stage 3
  Each stage runs in separate thread/process
  Back-pressure with bounded queues
  
Fan-Out/Fan-In:
  Single source → multiple workers → aggregate results
  
Rate Limiter:
  Token bucket algorithm
  Limit concurrent operations
  asyncio.Semaphore for async

Map-Reduce:
  Map: Apply function in parallel
  Reduce: Aggregate results
  multiprocessing.Pool.map for parallel map`,
					CodeExamples: `# Python concurrency and parallelism examples

import threading
import multiprocessing
import asyncio
import concurrent.futures
import queue
import time
from typing import Any, Callable, List, Optional

# ============================================================
# Threading Examples
# ============================================================

class ThreadSafeCounter:
    """Thread-safe counter using Lock."""
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def decrement(self):
        with self._lock:
            self._value -= 1
    
    @property
    def value(self):
        with self._lock:
            return self._value


class ReadWriteLock:
    """Readers-writer lock implementation."""
    
    def __init__(self):
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._writer_lock = threading.Lock()
    
    def acquire_read(self):
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._writer_lock.acquire()
    
    def release_read(self):
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._writer_lock.release()
    
    def acquire_write(self):
        self._writer_lock.acquire()
    
    def release_write(self):
        self._writer_lock.release()


class BoundedBlockingQueue:
    """Thread-safe bounded blocking queue."""
    
    def __init__(self, capacity: int):
        self._queue = []
        self._capacity = capacity
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
    
    def put(self, item):
        with self._not_full:
            while len(self._queue) >= self._capacity:
                self._not_full.wait()
            self._queue.append(item)
            self._not_empty.notify()
    
    def get(self):
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            item = self._queue.pop(0)
            self._not_full.notify()
            return item
    
    def size(self):
        with self._lock:
            return len(self._queue)


class WorkerPool:
    """Thread pool with graceful shutdown."""
    
    def __init__(self, num_workers: int):
        self._task_queue = queue.Queue()
        self._results = queue.Queue()
        self._workers = []
        self._shutdown = False
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"worker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self):
        while True:
            task = self._task_queue.get()
            if task is None:  # Sentinel for shutdown
                self._task_queue.task_done()
                break
            
            func, args, kwargs = task
            try:
                result = func(*args, **kwargs)
                self._results.put(("success", result))
            except Exception as e:
                self._results.put(("error", str(e)))
            finally:
                self._task_queue.task_done()
    
    def submit(self, func, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("Pool is shut down")
        self._task_queue.put((func, args, kwargs))
    
    def shutdown(self, wait=True):
        self._shutdown = True
        for _ in self._workers:
            self._task_queue.put(None)
        if wait:
            for worker in self._workers:
                worker.join()
    
    def get_results(self):
        results = []
        while not self._results.empty():
            results.append(self._results.get())
        return results


class PeriodicTimer:
    """Repeating timer that runs a function periodically."""
    
    def __init__(self, interval: float, func: Callable, *args, **kwargs):
        self._interval = interval
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._timer = None
        self._running = False
    
    def _run(self):
        self._running = False
        self.start()
        self._func(*self._args, **self._kwargs)
    
    def start(self):
        if not self._running:
            self._timer = threading.Timer(self._interval, self._run)
            self._timer.daemon = True
            self._timer.start()
            self._running = True
    
    def stop(self):
        if self._timer:
            self._timer.cancel()
        self._running = False


class ThreadSafeCache:
    """Thread-safe LRU cache with expiry."""
    
    def __init__(self, max_size: int = 100, ttl: float = 60.0):
        self._cache = {}
        self._access_order = []
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.RLock()
    
    def get(self, key):
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    del self._cache[key]
                    self._access_order.remove(key)
            return None
    
    def put(self, key, value):
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self._max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            
            self._cache[key] = (value, time.time())
            self._access_order.append(key)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


# ============================================================
# Producer-Consumer Pattern
# ============================================================

class Pipeline:
    """Multi-stage processing pipeline using threads."""
    
    def __init__(self):
        self._stages = []
        self._queues = []
    
    def add_stage(self, func: Callable, workers: int = 1):
        q = queue.Queue(maxsize=100)
        self._queues.append(q)
        self._stages.append((func, workers))
    
    def run(self, items: list):
        threads = []
        
        # Feed input
        input_q = self._queues[0]
        for item in items:
            input_q.put(item)
        for _ in range(self._stages[0][1]):
            input_q.put(None)  # Sentinel
        
        # Start stage threads
        for i, (func, workers) in enumerate(self._stages):
            in_q = self._queues[i]
            out_q = self._queues[i + 1] if i + 1 < len(self._queues) else None
            
            for w in range(workers):
                t = threading.Thread(
                    target=self._stage_worker,
                    args=(func, in_q, out_q, i == len(self._stages) - 1)
                )
                t.start()
                threads.append(t)
        
        for t in threads:
            t.join()
        
        # Collect results from last queue
        if self._queues:
            results = []
            last_q = self._queues[-1]
            while not last_q.empty():
                item = last_q.get()
                if item is not None:
                    results.append(item)
            return results
        return []
    
    def _stage_worker(self, func, in_q, out_q, is_last):
        while True:
            item = in_q.get()
            if item is None:
                if out_q:
                    out_q.put(None)
                in_q.task_done()
                break
            
            result = func(item)
            if out_q:
                out_q.put(result)
            in_q.task_done()


# ============================================================
# Multiprocessing Examples  
# ============================================================

def parallel_map_reduce(data, map_func, reduce_func, num_workers=None):
    """Parallel map-reduce using multiprocessing."""
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # Split data into chunks
    chunk_size = max(1, len(data) // num_workers)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Map phase
    with multiprocessing.Pool(num_workers) as pool:
        mapped = pool.map(map_func, chunks)
    
    # Reduce phase
    result = mapped[0]
    for m in mapped[1:]:
        result = reduce_func(result, m)
    
    return result


class SharedMemoryArray:
    """Shared memory wrapper for numeric arrays across processes."""
    
    def __init__(self, typecode: str, size: int):
        self._array = multiprocessing.Array(typecode, size)
        self._size = size
    
    def __getitem__(self, index):
        return self._array[index]
    
    def __setitem__(self, index, value):
        self._array[index] = value
    
    def __len__(self):
        return self._size
    
    def to_list(self):
        return list(self._array)


def process_with_timeout(func, args=(), timeout=30):
    """Run a function in a separate process with timeout."""
    result_queue = multiprocessing.Queue()
    
    def wrapper(q, fn, a):
        try:
            result = fn(*a)
            q.put(("success", result))
        except Exception as e:
            q.put(("error", str(e)))
    
    proc = multiprocessing.Process(target=wrapper, args=(result_queue, func, args))
    proc.start()
    proc.join(timeout)
    
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, "timeout"
    
    if not result_queue.empty():
        status, value = result_queue.get()
        return value, status
    
    return None, "no result"


# ============================================================
# AsyncIO Examples
# ============================================================

class AsyncRateLimiter:
    """Token bucket rate limiter for asyncio."""
    
    def __init__(self, rate: float, burst: int = 1):
        self._rate = rate  # tokens per second
        self._burst = burst
        self._tokens = burst
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._burst,
                self._tokens + elapsed * self._rate
            )
            self._last_refill = now
            
            if self._tokens >= 1:
                self._tokens -= 1
                return
            
            wait_time = (1 - self._tokens) / self._rate
            await asyncio.sleep(wait_time)
            self._tokens = 0
            self._last_refill = time.monotonic()


class AsyncBatcher:
    """Batches async requests together for efficiency."""
    
    def __init__(self, batch_func, max_size: int = 10, max_wait: float = 0.1):
        self._batch_func = batch_func
        self._max_size = max_size
        self._max_wait = max_wait
        self._pending = []
        self._lock = asyncio.Lock()
        self._timer_task = None
    
    async def submit(self, item):
        future = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            self._pending.append((item, future))
            
            if len(self._pending) >= self._max_size:
                await self._flush()
            elif self._timer_task is None:
                self._timer_task = asyncio.create_task(self._timer())
        
        return await future
    
    async def _timer(self):
        await asyncio.sleep(self._max_wait)
        async with self._lock:
            if self._pending:
                await self._flush()
            self._timer_task = None
    
    async def _flush(self):
        if not self._pending:
            return
        
        batch = self._pending[:]
        self._pending.clear()
        
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]
        
        try:
            results = await self._batch_func(items)
            for future, result in zip(futures, results):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


class AsyncWorkerPool:
    """Async worker pool with concurrency limit."""
    
    def __init__(self, max_workers: int):
        self._semaphore = asyncio.Semaphore(max_workers)
        self._tasks = set()
    
    async def submit(self, coro):
        async with self._semaphore:
            task = asyncio.create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return await task
    
    async def map(self, func, items):
        tasks = []
        for item in items:
            task = asyncio.create_task(self._bounded_call(func, item))
            tasks.append(task)
        return await asyncio.gather(*tasks)
    
    async def _bounded_call(self, func, item):
        async with self._semaphore:
            return await func(item)
    
    async def shutdown(self):
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def async_retry(coro_func, max_retries=3, backoff=1.0, exceptions=(Exception,)):
    """Retry an async function with exponential backoff."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await coro_func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait = backoff * (2 ** attempt)
                await asyncio.sleep(wait)
    
    raise last_exception


async def async_timeout_with_fallback(coro, timeout, fallback):
    """Run coroutine with timeout and fallback value."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return fallback


class AsyncEventBus:
    """Async publish-subscribe event bus."""
    
    def __init__(self):
        self._subscribers = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, event_type: str, handler):
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, data=None):
        async with self._lock:
            handlers = self._subscribers.get(event_type, [])[:]
        
        tasks = [handler(data) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def unsubscribe(self, event_type: str, handler):
        async with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(handler)


# ============================================================
# concurrent.futures Examples
# ============================================================

def parallel_download(urls: list, max_workers: int = 10):
    """Download multiple URLs concurrently."""
    results = {}
    
    def download_one(url):
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_one, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                results[url] = ("success", len(data))
            except Exception as e:
                results[url] = ("error", str(e))
    
    return results


def parallel_compute(items: list, func: Callable, max_workers: int = None):
    """CPU-bound parallel computation."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.map(func, items))
    return futures


class FutureChain:
    """Chain multiple futures together."""
    
    def __init__(self, executor):
        self._executor = executor
        self._chain = []
    
    def then(self, func):
        self._chain.append(func)
        return self
    
    def execute(self, initial_value):
        future = self._executor.submit(lambda x: x, initial_value)
        
        for func in self._chain:
            future = self._executor.submit(
                lambda f, fn: fn(f.result()),
                future, func
            )
        
        return future`,
				},
			},
		},
	})
}
