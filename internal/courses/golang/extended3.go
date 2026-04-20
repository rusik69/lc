package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1622,
			Title:       "Concurrency Deep Dive",
			Description: "Master Go's concurrency primitives at a deeper level: goroutine scheduling, channel internals, sync primitives, and advanced patterns for building concurrent systems.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Goroutine Scheduler Internals",
					Content: `Go's runtime uses an M:N scheduler — it multiplexes M goroutines onto N OS threads. Understanding the scheduler helps you write efficient concurrent programs and debug performance issues.

**The GMP Model:**
` + "```" + `
G = Goroutine (lightweight, ~2-8 KB stack)
M = Machine (OS thread)
P = Processor (logical CPU, scheduling context)

GOMAXPROCS controls the number of Ps (default: number of CPU cores)

    P0           P1           P2           P3
   ┌────┐      ┌────┐      ┌────┐      ┌────┐
   │LRQ │      │LRQ │      │LRQ │      │LRQ │  Local Run Queues
   │G G G│      │G G │      │G G G│      │G   │
   └──┬─┘      └──┬─┘      └──┬─┘      └──┬─┘
      │           │           │           │
   ┌──┴─┐      ┌──┴─┐      ┌──┴─┐      ┌──┴─┐
   │ M0 │      │ M1 │      │ M2 │      │ M3 │  OS Threads
   └────┘      └────┘      └────┘      └────┘
   
   ┌────────────────────────────────────────────┐
   │              Global Run Queue               │  Overflow
   │                G  G  G  G                   │  goroutines
   └────────────────────────────────────────────┘

How scheduling works:
1. Each P has a local run queue (LRQ) holding runnable Gs
2. Each M must acquire a P to execute goroutines
3. M takes the next G from P's LRQ
4. If LRQ is empty → steal from GRQ (global queue)
5. If GRQ is empty → steal from another P's LRQ (work stealing)
` + "```" + `

**Goroutine States:**
` + "```" + `
States:
  _Gidle:     Just allocated, not yet used
  _Grunnable: In a run queue, waiting to be scheduled
  _Grunning:  Currently executing on an M/P
  _Gsyscall:  Executing a system call (M blocked)
  _Gwaiting:  Blocked (channel op, mutex, I/O, sleep, select)
  _Gdead:     Finished execution, may be reused

Scheduling decisions happen at:
  - Channel operations (send/receive on blocking channel)
  - go statement (new goroutine created)
  - Blocking system calls (file I/O, network)
  - Garbage collection (STW pauses, mark assists)
  - sync.Mutex lock contention
  - time.Sleep / time.After
  - runtime.Gosched() (voluntary yield)

Preemption (since Go 1.14):
  Before 1.14: Only cooperative preemption (at function calls)
    Problem: tight loop without function calls → no preemption → stalls
  
  Since 1.14: Asynchronous preemption via signals
    Runtime sends signal to preempt long-running goroutines (~10ms quantum)
    Based on sysmon monitoring goroutine execution time
    Uses SIGURG signal on Unix systems
` + "```" + `

**System Call Handling:**
` + "```" + `
When a goroutine makes a blocking syscall:

Before:  P0 ──── M0 (G running)
         LRQ: [G1, G2, G3]

During syscall:
  M0 is blocked in kernel → can't run other goroutines
  Runtime detaches P0 from M0 ("handoff")
  
  P0 ──── M1 (new/idle M, runs G1 from LRQ)
  M0 (blocked in syscall with G)
  
  When syscall completes:
  - If P0 is idle → G goes back to P0's LRQ
  - If no P idle → G goes to Global Run Queue
  - M0 may be put back in idle M pool

Network I/O is special:
  Go uses netpoller (epoll/kqueue/io_uring)
  Network syscalls don't block the M!
  Goroutine is parked, M continues running other Gs
  When data arrives → netpoller wakes the goroutine
  
  This is why Go handles 100K+ concurrent connections efficiently
  Each connection = one goroutine (cheap)
  But only GOMAXPROCS OS threads doing the actual work
` + "```" + `

**Work Stealing:**
` + "```" + `
When a P's local run queue is empty:

1. Check global run queue (take batch of Gs, up to len(GRQ)/GOMAXPROCS+1)
2. Check netpoller for ready network goroutines
3. Steal from another P's LRQ (take half their queue)

Stealing algorithm:
  victim := random P (not self)
  n := len(victim.LRQ) / 2  // steal half
  for i := 0; i < n; i++ {
      g := victim.LRQ.popTail()
      self.LRQ.pushHead(g)
  }

This ensures even load distribution across all Ps.
Without stealing, one P could be overloaded while others idle.

Spinning threads:
  An M without work to do "spins" briefly before sleeping
  This avoids the overhead of sleep/wake for new work arriving quickly
  At most GOMAXPROCS spinning Ms at any time → bounded CPU waste
` + "```" + ``,
					CodeExamples: `// Concurrency patterns demonstrating scheduler behavior
package main

import (
    "fmt"
    "runtime"
    "sync"
    "sync/atomic"
    "time"
)

// Demonstrate GOMAXPROCS effect
func testGOMAXPROCS() {
    fmt.Println("=== GOMAXPROCS Effect ===")
    
    work := func() int64 {
        var sum int64
        for i := 0; i < 1_000_000; i++ {
            sum += int64(i)
        }
        return sum
    }

    for _, procs := range []int{1, 2, 4, runtime.NumCPU()} {
        runtime.GOMAXPROCS(procs)
        
        start := time.Now()
        var wg sync.WaitGroup
        numGoroutines := 8
        wg.Add(numGoroutines)
        
        for i := 0; i < numGoroutines; i++ {
            go func() {
                defer wg.Done()
                work()
            }()
        }
        wg.Wait()
        elapsed := time.Since(start)
        
        fmt.Printf("  GOMAXPROCS=%d: %v (8 goroutines)\n", procs, elapsed)
    }
    runtime.GOMAXPROCS(runtime.NumCPU())
}

// Demonstrate goroutine stack growth
func stackGrowthDemo() {
    fmt.Println("\n=== Goroutine Stack Growth ===")
    
    var memBefore, memAfter runtime.MemStats
    runtime.GC()
    runtime.ReadMemStats(&memBefore)
    
    const N = 10000
    var wg sync.WaitGroup
    wg.Add(N)
    
    done := make(chan struct{})
    for i := 0; i < N; i++ {
        go func() {
            wg.Done()
            <-done // Block until released
        }()
    }
    wg.Wait()
    
    runtime.ReadMemStats(&memAfter)
    perGoroutine := (memAfter.Sys - memBefore.Sys) / N
    fmt.Printf("  Created %d goroutines\n", N)
    fmt.Printf("  Memory per goroutine: ~%d bytes\n", perGoroutine)
    fmt.Printf("  Goroutines active: %d\n", runtime.NumGoroutine())
    
    close(done) // Release all goroutines
    time.Sleep(10 * time.Millisecond) // Let them finish
    fmt.Printf("  Goroutines after cleanup: %d\n", runtime.NumGoroutine())
}

// Demonstrate work stealing via goroutine distribution
func workStealingDemo() {
    fmt.Println("\n=== Work Stealing Demonstration ===")
    
    runtime.GOMAXPROCS(4)
    
    var counts [4]int64 // Track which P runs each goroutine
    var wg sync.WaitGroup
    
    const N = 1000
    wg.Add(N)
    
    for i := 0; i < N; i++ {
        go func() {
            defer wg.Done()
            // Record which P we're running on
            // (approximate - P can change during execution)
            pid := runtime_procPin()
            runtime_procUnpin()
            atomic.AddInt64(&counts[pid%4], 1)
            
            // Do some work
            sum := 0
            for j := 0; j < 10000; j++ {
                sum += j
            }
            _ = sum
        }()
    }
    wg.Wait()
    
    total := int64(0)
    for i, c := range counts {
        total += c
        fmt.Printf("  P%d handled: %d goroutines (%.1f%%)\n",
            i, c, float64(c)/float64(N)*100)
    }
}

// We can't actually call runtime_procPin directly, so simulate it
func runtime_procPin() int {
    // In real code, this is a runtime internal
    // Here we just use NumCPU as a stand-in
    return int(time.Now().UnixNano()) % runtime.GOMAXPROCS(0)
}

func runtime_procUnpin() {}

// Demonstrate preemption behavior
func preemptionDemo() {
    fmt.Println("\n=== Asynchronous Preemption (Go 1.14+) ===")
    
    runtime.GOMAXPROCS(1)
    
    done := make(chan bool)
    
    // This tight loop would never yield in Go < 1.14
    go func() {
        for i := 0; i < 1_000_000_000; i++ {
            // No function calls = no cooperative preemption points
            // But Go 1.14+ uses SIGURG for async preemption
        }
        done <- true
    }()
    
    // This goroutine should still be able to run thanks to preemption
    go func() {
        time.Sleep(10 * time.Millisecond) // Will this ever execute?
        fmt.Println("  Second goroutine executed (preemption works!)")
        done <- true
    }()
    
    <-done
    <-done
    runtime.GOMAXPROCS(runtime.NumCPU())
}

// Fan-out/fan-in pattern
func fanOutFanIn() {
    fmt.Println("\n=== Fan-Out/Fan-In Pattern ===")
    
    // Generate work items
    gen := func(nums ...int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for _, n := range nums {
                out <- n
            }
        }()
        return out
    }
    
    // Square numbers (worker)
    sq := func(in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for n := range in {
                out <- n * n
            }
        }()
        return out
    }
    
    // Merge multiple channels into one
    merge := func(channels ...<-chan int) <-chan int {
        var wg sync.WaitGroup
        out := make(chan int)
        
        wg.Add(len(channels))
        for _, ch := range channels {
            go func(c <-chan int) {
                defer wg.Done()
                for n := range c {
                    out <- n
                }
            }(ch)
        }
        
        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
    
    // Fan-out: distribute work across multiple goroutines
    input := gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    
    // Fan-out to 3 workers
    w1 := sq(input) // Workers share the input channel
    w2 := sq(input)
    w3 := sq(input)
    
    // Fan-in: merge results
    results := merge(w1, w2, w3)
    
    sum := 0
    count := 0
    for r := range results {
        sum += r
        count++
    }
    fmt.Printf("  Processed %d items, sum of squares: %d\n", count, sum)
}

func main() {
    fmt.Printf("Go version: %s\n", runtime.Version())
    fmt.Printf("CPUs: %d, GOMAXPROCS: %d\n", runtime.NumCPU(), runtime.GOMAXPROCS(0))
    
    testGOMAXPROCS()
    stackGrowthDemo()
    fanOutFanIn()
}`,
				},
				{
					Title: "Channel Internals and Patterns",
					Content: `Channels are Go's primary synchronization mechanism. Understanding their internal implementation helps you use them efficiently and choose the right pattern.

**Channel Internal Structure:**
` + "```" + `
type hchan struct {
    qcount   uint   // Number of elements currently in buffer
    dataqsiz uint   // Size of circular buffer (cap)
    buf      *byte  // Points to circular buffer of dataqsiz elements
    elemsize uint16 // Size of each element
    closed   uint32 // Is channel closed?
    sendx    uint   // Send index in circular buffer
    recvx    uint   // Receive index in circular buffer
    recvq    waitq  // List of goroutines waiting to receive
    sendq    waitq  // List of goroutines waiting to send
    lock     mutex  // Protects all fields
}

Unbuffered channel (make(chan int)):
  dataqsiz = 0, buf = nil
  Every send blocks until a receiver is ready (and vice versa)
  Direct copy from sender's stack to receiver's stack
  → Zero allocation for the data transfer!

Buffered channel (make(chan int, 10)):
  dataqsiz = 10, buf = [10]int
  Implemented as a lock-free ring buffer (with mutex)
  Send blocks only when buffer is full
  Receive blocks only when buffer is empty
  
  buf: [_, _, D, D, D, _, _, _, _, _]
            ^recvx      ^sendx
` + "```" + `

**Channel Operations and Blocking:**
` + "```" + `
Send (ch <- val):
  1. Acquire lock
  2. If there's a waiting receiver in recvq:
     → Copy val directly to receiver's stack (skip buffer)
     → Wake receiver goroutine
  3. Else if buffer has space:
     → Copy val to buf[sendx]
     → Increment sendx (mod dataqsiz)
  4. Else (buffer full or unbuffered):
     → Park sender goroutine in sendq
     → Release lock, yield to scheduler
  5. Release lock

Receive (val = <-ch):
  1. Acquire lock
  2. If there's a waiting sender in sendq:
     → If buffered: copy buf[recvx] to val, copy sender's val to buf
     → If unbuffered: copy sender's val directly to val
     → Wake sender goroutine
  3. Else if buffer has data:
     → Copy buf[recvx] to val
     → Increment recvx (mod dataqsiz)
  4. Else (buffer empty):
     → Park receiver goroutine in recvq
  5. Release lock

Close (close(ch)):
  1. Acquire lock
  2. Set closed = 1
  3. Wake ALL waiting receivers (they get zero value, ok=false)
  4. Wake ALL waiting senders (they panic!)
  5. Release lock

Key rules:
  - Send to closed channel → panic
  - Receive from closed channel → zero value, ok=false (after drain)
  - Close nil channel → panic
  - Send to nil channel → block forever
  - Receive from nil channel → block forever
  - Close is idempotent only if channel not already closed
` + "```" + `

**Select Statement:**
` + "```" + `
select evaluates all cases simultaneously:

select {
case msg := <-ch1:
    // ch1 is ready
case ch2 <- val:
    // ch2 accepted val
case <-time.After(5 * time.Second):
    // timeout
default:
    // non-blocking: none ready
}

Internal implementation:
1. Lock all channels involved (in address order to avoid deadlock)
2. Check if any case is immediately ready
3. If multiple ready → pick one randomly (uniform distribution)
4. If none ready and no default → park goroutine on all channels
5. When woken → determine which case, remove from other wait queues

The random selection prevents starvation:
  Without randomness, the first case would always win
  This could starve later cases
  Go explicitly uses a pseudo-random order for fairness

Performance tip:
  Non-blocking select (with default) is very fast
  Blocking select has overhead from locking multiple channels
  For hot paths, consider sync.Mutex or atomic operations instead
` + "```" + `

**Common Channel Patterns:**
` + "```" + `
1. Done Channel (cancellation):
   done := make(chan struct{})
   go worker(done)
   close(done) // Signal all workers to stop
   
   // In worker:
   select {
   case <-done: return
   case item := <-work: process(item)
   }

2. Pipeline:
   gen → filter → transform → output
   Each stage is a goroutine, connected by channels

3. Semaphore (bounded concurrency):
   sem := make(chan struct{}, maxConcurrent)
   sem <- struct{}{} // Acquire
   <-sem              // Release

4. Or-Done Channel:
   Wrap a channel to respect cancellation:
   func orDone(done <-chan struct{}, c <-chan int) <-chan int {
       out := make(chan int)
       go func() {
           defer close(out)
           for {
               select {
               case <-done: return
               case v, ok := <-c:
                   if !ok { return }
                   select {
                   case out <- v:
                   case <-done: return
                   }
               }
           }
       }()
       return out
   }

5. Tee Channel:
   Split one channel into two (both get every item):
   func tee(done <-chan struct{}, in <-chan int) (<-chan int, <-chan int)

6. Bridge Channel:
   Flatten a channel of channels into a single channel:
   func bridge(done <-chan struct{}, chanCh <-chan <-chan int) <-chan int
` + "```" + ``,
					CodeExamples: `// Advanced channel patterns
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Pipeline pattern
func generator(ctx context.Context, nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            select {
            case out <- n:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func multiply(ctx context.Context, in <-chan int, factor int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            select {
            case out <- n * factor:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func add(ctx context.Context, in <-chan int, addend int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            select {
            case out <- n + addend:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

// Semaphore pattern - bounded concurrency
type Semaphore struct {
    ch chan struct{}
}

func NewSemaphore(maxConcurrency int) *Semaphore {
    return &Semaphore{ch: make(chan struct{}, maxConcurrency)}
}

func (s *Semaphore) Acquire() { s.ch <- struct{}{} }
func (s *Semaphore) Release() { <-s.ch }

// Or channel - returns when ANY of the channels closes
func or(channels ...<-chan struct{}) <-chan struct{} {
    switch len(channels) {
    case 0:
        return nil
    case 1:
        return channels[0]
    }

    orDone := make(chan struct{})
    go func() {
        defer close(orDone)
        switch len(channels) {
        case 2:
            select {
            case <-channels[0]:
            case <-channels[1]:
            }
        default:
            select {
            case <-channels[0]:
            case <-channels[1]:
            case <-channels[2]:
            case <-or(append(channels[3:], orDone)...):
            }
        }
    }()
    return orDone
}

// Rate limiter using channels
type RateLimiter struct {
    tokens chan struct{}
    done   chan struct{}
}

func NewRateLimiter(rate int, interval time.Duration) *RateLimiter {
    rl := &RateLimiter{
        tokens: make(chan struct{}, rate),
        done:   make(chan struct{}),
    }
    
    // Fill initial tokens
    for i := 0; i < rate; i++ {
        rl.tokens <- struct{}{}
    }
    
    // Refill tokens periodically
    go func() {
        ticker := time.NewTicker(interval / time.Duration(rate))
        defer ticker.Stop()
        for {
            select {
            case <-ticker.C:
                select {
                case rl.tokens <- struct{}{}:
                default: // Bucket full
                }
            case <-rl.done:
                return
            }
        }
    }()
    
    return rl
}

func (rl *RateLimiter) Wait() { <-rl.tokens }
func (rl *RateLimiter) Stop() { close(rl.done) }

// Worker pool
func workerPool(ctx context.Context, numWorkers int, jobs <-chan int) <-chan int {
    results := make(chan int, numWorkers)
    var wg sync.WaitGroup
    
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for job := range jobs {
                select {
                case <-ctx.Done():
                    return
                default:
                    // Simulate work
                    result := job * job
                    results <- result
                }
            }
        }(i)
    }
    
    go func() {
        wg.Wait()
        close(results)
    }()
    
    return results
}

func main() {
    // Pipeline demo
    fmt.Println("=== Pipeline Pattern ===")
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    // Pipeline: generate → multiply by 2 → add 1
    pipeline := add(ctx, multiply(ctx, generator(ctx, 1, 2, 3, 4, 5), 2), 1)
    
    for v := range pipeline {
        fmt.Printf("  %d", v) // 3, 5, 7, 9, 11
    }
    fmt.Println()
    
    // Semaphore demo
    fmt.Println("\n=== Bounded Concurrency (Semaphore) ===")
    sem := NewSemaphore(3) // Max 3 concurrent
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            sem.Acquire()
            defer sem.Release()
            fmt.Printf("  Worker %d running\n", id)
            time.Sleep(50 * time.Millisecond)
        }(i)
    }
    wg.Wait()
    
    // Or channel - first one wins
    fmt.Println("\n=== Or Channel (First Signal Wins) ===")
    sig := func(after time.Duration) <-chan struct{} {
        ch := make(chan struct{})
        go func() {
            defer close(ch)
            time.Sleep(after)
        }()
        return ch
    }
    
    start := time.Now()
    <-or(
        sig(2*time.Hour),
        sig(5*time.Minute),
        sig(100*time.Millisecond), // This wins
        sig(1*time.Hour),
    )
    fmt.Printf("  Done after %v (shortest timer)\n", time.Since(start).Round(time.Millisecond))
    
    // Worker pool demo
    fmt.Println("\n=== Worker Pool ===")
    ctx2, cancel2 := context.WithCancel(context.Background())
    defer cancel2()
    
    jobs := make(chan int, 20)
    for i := 1; i <= 20; i++ {
        jobs <- i
    }
    close(jobs)
    
    results := workerPool(ctx2, 4, jobs)
    sum := 0
    count := 0
    for r := range results {
        sum += r
        count++
    }
    fmt.Printf("  Processed %d jobs, sum of squares: %d\n", count, sum)
    
    // Rate limiter demo
    fmt.Println("\n=== Rate Limiter ===")
    rl := NewRateLimiter(5, time.Second) // 5 per second
    defer rl.Stop()
    
    start = time.Now()
    for i := 0; i < 10; i++ {
        rl.Wait()
        fmt.Printf("  Request %d at %v\n", i, time.Since(start).Round(time.Millisecond))
    }
}`,
				},
				{
					Title: "sync Package Deep Dive",
					Content: `The sync package provides low-level synchronization primitives. While channels are idiomatic, sync primitives are essential for shared-memory concurrent access.

**sync.Mutex and sync.RWMutex:**
` + "```" + `
sync.Mutex:
  Lock() / Unlock()
  Only one goroutine can hold the lock at a time
  Zero value is an unlocked mutex (no initialization needed!)
  
  // ALWAYS use defer for Unlock to prevent deadlocks
  mu.Lock()
  defer mu.Unlock()
  // critical section

sync.RWMutex:
  RLock() / RUnlock() - Multiple readers can hold simultaneously
  Lock() / Unlock()   - Writer gets exclusive access
  
  Read-heavy workloads benefit from RWMutex:
    100 goroutines reading, 1 writing → 100x read throughput vs Mutex
    
  BUT: RWMutex has overhead (tracks reader count)
  For simple cases with equal read/write, regular Mutex may be faster

Internal implementation (Go 1.19+):
  Mutex uses two modes:
    Normal mode: FIFO order, but new arrivals can steal the lock
      → Better throughput (reduces context switches)
    Starvation mode: Strict FIFO after 1ms of waiting
      → Prevents tail latency for long-waiting goroutines
      
  Transition: Normal → Starvation when a waiter waits > 1ms
  Transition: Starvation → Normal when waiter queue is empty or < 1ms wait
` + "```" + `

**sync.WaitGroup:**
` + "```" + `
Usage pattern:
  var wg sync.WaitGroup
  
  for i := 0; i < n; i++ {
      wg.Add(1)
      go func() {
          defer wg.Done()  // Decrement when goroutine finishes
          // do work
      }()
  }
  wg.Wait() // Block until counter reaches 0

Common mistakes:
  1. Calling wg.Add(1) INSIDE the goroutine
     → Race condition: Wait() might see 0 before Add(1) executes
  
  2. Forgetting wg.Done() in error paths
     → Wait() blocks forever (goroutine leak + deadlock)
  
  3. Reusing WaitGroup before Wait() returns
     → Undefined behavior

Internal: int64 counter + semaphore
  Add(n) adds n to counter
  Done() is Add(-1)
  Wait() blocks when counter > 0, wakes when counter reaches 0
` + "```" + `

**sync.Once and sync.OnceFunc:**
` + "```" + `
sync.Once:
  Guarantees a function runs exactly once, even from multiple goroutines
  
  var once sync.Once
  var instance *DB
  
  func GetDB() *DB {
      once.Do(func() {
          instance = connectToDatabase() // Runs exactly once
      })
      return instance
  }
  
  If the function panics, Once is still considered "done"
  → Won't retry! Use sync.OnceFunc for retry behavior

sync.OnceFunc (Go 1.21+):
  f := sync.OnceFunc(func() {
      fmt.Println("computed once")
  })
  f() // Runs
  f() // No-op
  
sync.OnceValue (Go 1.21+):
  getExpensiveValue := sync.OnceValue(func() int {
      return computeExpensiveValue()
  })
  v := getExpensiveValue() // Computes once, returns cached value
  v = getExpensiveValue()  // Returns cached value
` + "```" + `

**sync.Pool:**
` + "```" + `
sync.Pool: Cache of temporary objects that can be reused
  Reduces GC pressure by recycling allocated objects
  Objects may be collected at any GC cycle (no guarantees!)

  var bufPool = sync.Pool{
      New: func() any {
          return new(bytes.Buffer)
      },
  }
  
  // Get from pool (or allocate new)
  buf := bufPool.Get().(*bytes.Buffer)
  buf.Reset() // ALWAYS reset before use!
  
  // Use buffer...
  buf.WriteString("hello")
  
  // Return to pool
  bufPool.Put(buf)

When to use sync.Pool:
  ✓ Frequently allocated/freed objects (buffers, structs)
  ✓ Objects with expensive initialization
  ✓ High-throughput systems (reduce GC pauses)
  ✗ Long-lived objects (pool drains on GC)
  ✗ Objects that MUST persist (no guarantees!)

Performance impact:
  Without pool: allocate+GC every request → GC pressure
  With pool: reuse existing → fewer allocations
  Real-world: 30-70% reduction in allocs for HTTP handlers
` + "```" + `

**sync.Map:**
` + "```" + `
sync.Map: Concurrent map (no external locking needed)

  var m sync.Map
  m.Store("key", "value")
  v, ok := m.Load("key")
  m.Delete("key")
  m.LoadOrStore("key", "default") // Load if exists, store if not
  m.Range(func(key, value any) bool {
      fmt.Println(key, value)
      return true // Continue iteration
  })

When to use sync.Map vs map+Mutex:
  sync.Map is better when:
    - Keys are stable (mostly reads, few writes)
    - Many goroutines read disjoint sets of keys
    
  map+Mutex is better when:
    - Frequent writes to the same keys
    - need to clear the map atomically  
    - Need typed access without assertions

sync.Map internal: Two maps
  read map: Lock-free reads (atomic pointer)
  dirty map: Writes go here first (with mutex)
  Promotes dirty → read after enough misses
  → Optimized for read-heavy/append-only patterns
` + "```" + `

**atomic Package:**
` + "```" + `
For simple shared variables, atomic operations avoid locks entirely:

  var counter atomic.Int64
  counter.Add(1)
  counter.Store(42)
  v := counter.Load()
  
  // Compare-and-swap (CAS):
  old := counter.Load()
  swapped := counter.CompareAndSwap(old, old+1)
  
  // atomic.Value (any type, but same type always):
  var config atomic.Value
  config.Store(Config{...})
  cfg := config.Load().(Config)

Go 1.19+ typed atomics (preferred):
  atomic.Int32, atomic.Int64, atomic.Uint32, atomic.Uint64
  atomic.Bool
  atomic.Pointer[T]
  
  // Old style (still works):
  var val int64
  atomic.AddInt64(&val, 1)
  
  // New style (cleaner):
  var val atomic.Int64
  val.Add(1)

When to use atomics:
  ✓ Simple counters, flags, status variables
  ✓ Lock-free data structures
  ✗ Complex operations on multiple variables (use mutex)
  ✗ When you need to protect a multi-step operation
` + "```" + ``,
					CodeExamples: `// sync package patterns
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

// Safe counter with mutex
type SafeCounter struct {
    mu sync.RWMutex
    v  map[string]int
}

func (c *SafeCounter) Inc(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.v[key]++
}

func (c *SafeCounter) Get(key string) int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.v[key]
}

// Connection pool using sync.Pool
type Connection struct {
    ID     int
    Active bool
}

var connID atomic.Int64
var connPool = sync.Pool{
    New: func() any {
        id := connID.Add(1)
        return &Connection{ID: int(id)}
    },
}

func withConnection(fn func(*Connection)) {
    conn := connPool.Get().(*Connection)
    conn.Active = true
    defer func() {
        conn.Active = false
        connPool.Put(conn)
    }()
    fn(conn)
}

// Lazy singleton with sync.Once
type Database struct {
    connected bool
    name      string
}

var dbOnce sync.Once
var dbInstance *Database

func GetDatabase() *Database {
    dbOnce.Do(func() {
        fmt.Println("  Connecting to database (happens once)")
        dbInstance = &Database{connected: true, name: "postgres"}
    })
    return dbInstance
}

// Concurrent map benchmark comparison
func benchmarkMaps() {
    const numOps = 10000
    const numGoroutines = 10
    
    // sync.Map
    start := time.Now()
    var sm sync.Map
    var wg1 sync.WaitGroup
    for g := 0; g < numGoroutines; g++ {
        wg1.Add(1)
        go func(id int) {
            defer wg1.Done()
            for i := 0; i < numOps; i++ {
                key := fmt.Sprintf("%d-%d", id, i)
                sm.Store(key, i)
                sm.Load(key)
            }
        }(g)
    }
    wg1.Wait()
    syncMapDur := time.Since(start)
    
    // map + RWMutex
    start = time.Now()
    m := make(map[string]int)
    var mu sync.RWMutex
    var wg2 sync.WaitGroup
    for g := 0; g < numGoroutines; g++ {
        wg2.Add(1)
        go func(id int) {
            defer wg2.Done()
            for i := 0; i < numOps; i++ {
                key := fmt.Sprintf("%d-%d", id, i)
                mu.Lock()
                m[key] = i
                mu.Unlock()
                mu.RLock()
                _ = m[key]
                mu.RUnlock()
            }
        }(g)
    }
    wg2.Wait()
    mutexMapDur := time.Since(start)
    
    fmt.Printf("  sync.Map: %v\n", syncMapDur)
    fmt.Printf("  map+RWMutex: %v\n", mutexMapDur)
}

// Barrier using WaitGroup
func parallelComputation() {
    fmt.Println("  Phase 1: All workers compute independently")
    
    const workers = 4
    results := make([]int, workers)
    var phase1 sync.WaitGroup
    
    phase1.Add(workers)
    for i := 0; i < workers; i++ {
        go func(id int) {
            defer phase1.Done()
            results[id] = (id + 1) * 100
            fmt.Printf("    Worker %d: computed %d\n", id, results[id])
        }(i)
    }
    phase1.Wait()
    
    fmt.Println("  Phase 2: Combine results (barrier passed)")
    total := 0
    for _, r := range results {
        total += r
    }
    fmt.Printf("    Total: %d\n", total)
}

func main() {
    // SafeCounter
    fmt.Println("=== Safe Counter (RWMutex) ===")
    c := SafeCounter{v: make(map[string]int)}
    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            c.Inc("hits")
        }()
    }
    wg.Wait()
    fmt.Printf("  Total hits: %d\n", c.Get("hits"))
    
    // sync.Pool
    fmt.Println("\n=== Connection Pool (sync.Pool) ===")
    for i := 0; i < 5; i++ {
        withConnection(func(conn *Connection) {
            fmt.Printf("  Using connection ID=%d\n", conn.ID)
        })
    }
    
    // sync.Once
    fmt.Println("\n=== Singleton (sync.Once) ===")
    var wg2 sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg2.Add(1)
        go func() {
            defer wg2.Done()
            db := GetDatabase()
            _ = db
        }()
    }
    wg2.Wait()
    fmt.Printf("  DB connected: %v, name: %s\n",
        GetDatabase().connected, GetDatabase().name)
    
    // Map benchmark
    fmt.Println("\n=== Map Concurrency Comparison ===")
    benchmarkMaps()
    
    // Barrier pattern
    fmt.Println("\n=== Barrier Pattern (WaitGroup) ===")
    parallelComputation()
    
    // Atomic operations
    fmt.Println("\n=== Atomic Operations ===")
    var counter atomic.Int64
    var wg3 sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg3.Add(1)
        go func() {
            defer wg3.Done()
            counter.Add(1)
        }()
    }
    wg3.Wait()
    fmt.Printf("  Atomic counter: %d (expected: 1000)\n", counter.Load())
}`,
				},
			},
		},
	})
}
