package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1624,
			Title:       "Database Patterns in Go",
			Description: "Master database access in Go: SQL, connection management, transactions, migrations, ORMs, and NoSQL integration.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "SQL with database/sql",
					Content: `The database/sql package provides a generic interface around SQL databases. It handles connection pooling, prepared statements, and transactions automatically.

**Architecture:**
` + "```" + `
database/sql architecture:

  Your Code
      │
      ▼
  sql.DB ───────────── Connection Pool
      │                   │  │  │  ...
      ▼                   ▼  ▼  ▼
  database/sql/driver    driver.Conn instances
      │
      ▼
  Driver (lib/pq, pgx, mysql, sqlite3)
      │
      ▼
  Database Server

sql.DB is NOT a single connection. It is:
  - A connection pool manager
  - Thread-safe (goroutine-safe)
  - Lazy (connects on first use)
  - Long-lived (create once, share everywhere)
  - Handles reconnection automatically

NEVER create sql.DB per request!
  ✗ func handler(w http.ResponseWriter, r *http.Request) {
        db, _ := sql.Open("postgres", dsn) // WRONG
        defer db.Close()
    }
  ✓ var db *sql.DB // Package-level, initialized once
` + "```" + `

**Connection Pool Configuration:**
` + "```" + `
db, err := sql.Open("postgres", dsn)

// Pool settings
db.SetMaxOpenConns(25)    // Max concurrent connections
db.SetMaxIdleConns(5)     // Max idle connections in pool
db.SetConnMaxLifetime(5 * time.Minute)  // Max time a conn can be reused
db.SetConnMaxIdleTime(1 * time.Minute)  // Max time a conn can be idle

Sizing guidelines:
  MaxOpenConns:
    - Too low → queries queue up, high latency
    - Too high → database overloaded, diminishing returns
    - Rule of thumb: 2-4x CPU cores of database server
    - PostgreSQL default max: 100 connections
    
  MaxIdleConns:
    - Too low → frequent connection creation/teardown
    - Too high → waste memory holding idle connections
    - Usually: MaxOpenConns / 2 to MaxOpenConns
    
  ConnMaxLifetime:
    - Prevents using stale connections
    - Must be < database server's connection timeout
    - 5-30 minutes typical
    
  ConnMaxIdleTime:
    - Reclaims idle connections faster
    - Useful for bursty workloads

Verify connection:
  ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
  defer cancel()
  err := db.PingContext(ctx)
` + "```" + `

**Queries and Scanning:**
` + "```" + `
Single row:
  var user User
  err := db.QueryRowContext(ctx,
      "SELECT id, name, email FROM users WHERE id = $1", id,
  ).Scan(&user.ID, &user.Name, &user.Email)
  
  if err == sql.ErrNoRows {
      // Not found (not an error in most cases)
  } else if err != nil {
      // Actual error
  }

Multiple rows:
  rows, err := db.QueryContext(ctx,
      "SELECT id, name, email FROM users WHERE active = $1", true)
  if err != nil { return err }
  defer rows.Close() // CRITICAL: always close rows!
  
  var users []User
  for rows.Next() {
      var u User
      if err := rows.Scan(&u.ID, &u.Name, &u.Email); err != nil {
          return err
      }
      users = append(users, u)
  }
  // Check for iteration errors
  if err := rows.Err(); err != nil { return err }

Exec (INSERT, UPDATE, DELETE):
  result, err := db.ExecContext(ctx,
      "INSERT INTO users (name, email) VALUES ($1, $2)",
      "Alice", "alice@example.com",
  )
  id, _ := result.LastInsertId()     // Not supported by all drivers
  affected, _ := result.RowsAffected()

IMPORTANT: Use positional parameters ($1, $2 for PostgreSQL; ? for MySQL)
  NEVER concatenate user input into queries → SQL injection!
  ✗ db.Query("SELECT * FROM users WHERE name = '" + name + "'")
  ✓ db.QueryContext(ctx, "SELECT * FROM users WHERE name = $1", name)
` + "```" + `

**Prepared Statements:**
` + "```" + `
Use prepared statements when executing the same query repeatedly:

  stmt, err := db.PrepareContext(ctx,
      "INSERT INTO events (type, data) VALUES ($1, $2)")
  if err != nil { return err }
  defer stmt.Close()
  
  for _, event := range events {
      _, err := stmt.ExecContext(ctx, event.Type, event.Data)
      if err != nil { return err }
  }

How prepared statements work:
  1. db.Prepare() → server parses and plans query ONCE
  2. stmt.Exec() → server executes with parameters (fast)
  3. Reuses the plan across multiple calls
  
  Without prepare:
    Each db.Query → parse → plan → execute
  With prepare:
    First: parse → plan (save plan)
    Subsequent: execute with plan (skip parse+plan)
  
Performance gain: ~10-30% for repeated queries
  Mainly useful in loops; for one-off queries, no benefit

Caveat: prepared statements are per-connection.
  database/sql re-prepares transparently on new connections.
  This means the pool manages the complexity for you.
` + "```" + ``,
					CodeExamples: `// Database patterns in Go
package main

import (
    "context"
    "database/sql"
    "errors"
    "fmt"
    "sync"
    "time"
)

// Repository pattern
type User struct {
    ID        int64
    Name      string
    Email     string
    CreatedAt time.Time
}

type UserRepository interface {
    Create(ctx context.Context, user *User) error
    GetByID(ctx context.Context, id int64) (*User, error)
    List(ctx context.Context, limit, offset int) ([]User, error)
    Update(ctx context.Context, user *User) error
    Delete(ctx context.Context, id int64) error
}

// PostgreSQL implementation
type pgUserRepo struct {
    db *sql.DB
}

func NewPGUserRepo(db *sql.DB) UserRepository {
    return &pgUserRepo{db: db}
}

func (r *pgUserRepo) Create(ctx context.Context, user *User) error {
    return r.db.QueryRowContext(ctx,
        "INSERT INTO users (name, email, created_at) VALUES ($1, $2, $3) RETURNING id",
        user.Name, user.Email, time.Now(),
    ).Scan(&user.ID)
}

func (r *pgUserRepo) GetByID(ctx context.Context, id int64) (*User, error) {
    var u User
    err := r.db.QueryRowContext(ctx,
        "SELECT id, name, email, created_at FROM users WHERE id = $1", id,
    ).Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt)
    
    if errors.Is(err, sql.ErrNoRows) {
        return nil, fmt.Errorf("user %d not found", id)
    }
    if err != nil {
        return nil, fmt.Errorf("query user: %w", err)
    }
    return &u, nil
}

func (r *pgUserRepo) List(ctx context.Context, limit, offset int) ([]User, error) {
    rows, err := r.db.QueryContext(ctx,
        "SELECT id, name, email, created_at FROM users ORDER BY id LIMIT $1 OFFSET $2",
        limit, offset,
    )
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt); err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    return users, rows.Err()
}

func (r *pgUserRepo) Update(ctx context.Context, user *User) error {
    result, err := r.db.ExecContext(ctx,
        "UPDATE users SET name = $1, email = $2 WHERE id = $3",
        user.Name, user.Email, user.ID,
    )
    if err != nil {
        return err
    }
    affected, _ := result.RowsAffected()
    if affected == 0 {
        return fmt.Errorf("user %d not found", user.ID)
    }
    return nil
}

func (r *pgUserRepo) Delete(ctx context.Context, id int64) error {
    result, err := r.db.ExecContext(ctx,
        "DELETE FROM users WHERE id = $1", id,
    )
    if err != nil {
        return err
    }
    affected, _ := result.RowsAffected()
    if affected == 0 {
        return fmt.Errorf("user %d not found", id)
    }
    return nil
}

// In-memory implementation for testing
type memUserRepo struct {
    mu     sync.RWMutex
    users  map[int64]User
    nextID int64
}

func NewMemUserRepo() UserRepository {
    return &memUserRepo{
        users:  make(map[int64]User),
        nextID: 1,
    }
}

func (r *memUserRepo) Create(ctx context.Context, user *User) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    user.ID = r.nextID
    user.CreatedAt = time.Now()
    r.users[user.ID] = *user
    r.nextID++
    return nil
}

func (r *memUserRepo) GetByID(ctx context.Context, id int64) (*User, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    u, ok := r.users[id]
    if !ok {
        return nil, fmt.Errorf("user %d not found", id)
    }
    return &u, nil
}

func (r *memUserRepo) List(ctx context.Context, limit, offset int) ([]User, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    users := make([]User, 0, len(r.users))
    for _, u := range r.users {
        users = append(users, u)
    }
    
    if offset >= len(users) {
        return nil, nil
    }
    end := offset + limit
    if end > len(users) {
        end = len(users)
    }
    return users[offset:end], nil
}

func (r *memUserRepo) Update(ctx context.Context, user *User) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    if _, ok := r.users[user.ID]; !ok {
        return fmt.Errorf("user %d not found", user.ID)
    }
    r.users[user.ID] = *user
    return nil
}

func (r *memUserRepo) Delete(ctx context.Context, id int64) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    if _, ok := r.users[id]; !ok {
        return fmt.Errorf("user %d not found", id)
    }
    delete(r.users, id)
    return nil
}

// Service layer using repository
type UserService struct {
    repo UserRepository
}

func (s *UserService) CreateUser(ctx context.Context, name, email string) (*User, error) {
    if name == "" || email == "" {
        return nil, errors.New("name and email are required")
    }
    user := &User{Name: name, Email: email}
    if err := s.repo.Create(ctx, user); err != nil {
        return nil, fmt.Errorf("create user: %w", err)
    }
    return user, nil
}

func main() {
    repo := NewMemUserRepo()
    svc := &UserService{repo: repo}
    ctx := context.Background()
    
    // Create users
    alice, _ := svc.CreateUser(ctx, "Alice", "alice@example.com")
    fmt.Printf("Created: %+v\n", alice)
    
    bob, _ := svc.CreateUser(ctx, "Bob", "bob@example.com")
    fmt.Printf("Created: %+v\n", bob)
    
    // Get user
    user, _ := repo.GetByID(ctx, 1)
    fmt.Printf("Got: %+v\n", user)
    
    // List users
    users, _ := repo.List(ctx, 10, 0)
    fmt.Printf("Listed %d users\n", len(users))
    
    // Update user
    alice.Name = "Alice Updated"
    _ = repo.Update(ctx, alice)
    updated, _ := repo.GetByID(ctx, 1)
    fmt.Printf("Updated: %+v\n", updated)
    
    // Delete user
    _ = repo.Delete(ctx, 2)
    remaining, _ := repo.List(ctx, 10, 0)
    fmt.Printf("After delete: %d users\n", len(remaining))
}`,
				},
				{
					Title: "Transactions and Migrations",
					Content: `Transactions ensure atomicity of multi-statement operations. Migrations manage schema evolution. Both are essential for production database work.

**Transactions:**
` + "```" + `
Basic transaction:
  tx, err := db.BeginTx(ctx, &sql.TxOptions{
      Isolation: sql.LevelSerializable, // Or LevelDefault
      ReadOnly:  false,
  })
  if err != nil { return err }
  
  // ALWAYS ensure rollback on error (no-op if committed)
  defer tx.Rollback()
  
  // Use tx instead of db for all operations
  _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, fromID)
  if err != nil { return err }
  
  _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, toID)
  if err != nil { return err }
  
  // Commit (makes changes permanent)
  return tx.Commit()

Isolation levels:
  sql.LevelDefault         → Database default (usually ReadCommitted)
  sql.LevelReadUncommitted → Dirty reads possible (fastest, weakest)
  sql.LevelReadCommitted   → No dirty reads
  sql.LevelRepeatableRead  → Same read in tx returns same result
  sql.LevelSerializable    → Full isolation (slowest, strongest)

  Tradeoff: higher isolation → more locking → lower throughput

Transaction helper pattern:
  func WithTx(ctx context.Context, db *sql.DB, fn func(tx *sql.Tx) error) error {
      tx, err := db.BeginTx(ctx, nil)
      if err != nil { return err }
      defer tx.Rollback()
      
      if err := fn(tx); err != nil {
          return err
      }
      return tx.Commit()
  }
  
  // Usage:
  err := WithTx(ctx, db, func(tx *sql.Tx) error {
      _, err := tx.Exec(...)
      if err != nil { return err }
      _, err = tx.Exec(...)
      return err
  })
` + "```" + `

**Avoiding Transaction Pitfalls:**
` + "```" + `
1. Long-running transactions:
   - Hold locks, block other queries
   - Risk of deadlock
   - Keep transactions SHORT
   - Do computation outside tx, only db ops inside
   
   ✗ tx.Begin()
     result := expensiveComputation()  // 5 seconds of CPU
     tx.Exec(... result ...)
     tx.Commit()
     
   ✓ result := expensiveComputation()  // Outside tx
     tx.Begin()
     tx.Exec(... result ...)
     tx.Commit()                       // Tx held for milliseconds

2. Connection leak:
   tx holds a connection from the pool until Commit/Rollback
   If you forget to commit/rollback → connection never returned!
   ALWAYS: defer tx.Rollback()

3. Deadlocks:
   Tx A: UPDATE users SET ... WHERE id = 1  (locks row 1)
   Tx B: UPDATE users SET ... WHERE id = 2  (locks row 2)
   Tx A: UPDATE users SET ... WHERE id = 2  (waits for Tx B)
   Tx B: UPDATE users SET ... WHERE id = 1  (waits for Tx A → DEADLOCK!)
   
   Prevention: always access rows in consistent order
   Detection: database detects and aborts one transaction
   Handling: retry the aborted transaction
   
4. Passing tx vs db:
   Functions that work with both can use an interface:
   type DBTX interface {
       ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
       QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
       QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
   }
   // Both *sql.DB and *sql.Tx satisfy this interface
` + "```" + `

**Migrations:**
` + "```" + `
Migrations are version-controlled schema changes:

  001_create_users.up.sql:
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
  
  001_create_users.down.sql:
    DROP TABLE users;
  
  002_add_user_avatar.up.sql:
    ALTER TABLE users ADD COLUMN avatar_url TEXT;
  
  002_add_user_avatar.down.sql:
    ALTER TABLE users DROP COLUMN avatar_url;

Tools:
  - golang-migrate/migrate: Most popular, CLI + library
  - goose: Supports Go and SQL migrations
  - atlas: Schema-as-code, declarative

golang-migrate usage:
  migrate -database "postgres://..." -path ./migrations up
  migrate -database "postgres://..." -path ./migrations down 1
  
  Programmatic:
  m, err := migrate.New("file://migrations", databaseURL)
  m.Up()     // Run all pending
  m.Down()   // Rollback last
  m.Steps(2) // Run next 2

Migration best practices:
  - Always write both up AND down
  - Never modify a deployed migration (create new one)
  - Use transactions for DDL when possible
  - Test migrations on production-like data
  - Include data migrations when schema changes require it
  - Keep migrations small and focused
  - Name descriptively: 003_add_index_users_email.sql
  - Run in CI to catch issues early
` + "```" + ``,
					CodeExamples: `// Transaction patterns and migration concepts
package main

import (
    "context"
    "errors"
    "fmt"
    "sync"
)

// DBTX interface for testable database code
type DBTX interface {
    Exec(ctx context.Context, query string, args ...any) error
    Query(ctx context.Context, query string, args ...any) ([]map[string]any, error)
}

// In-memory database simulation
type MemDB struct {
    mu       sync.RWMutex
    accounts map[int64]float64
    txLog    []string
}

func NewMemDB() *MemDB {
    return &MemDB{
        accounts: map[int64]float64{
            1: 1000.0, // Alice
            2: 500.0,  // Bob
            3: 750.0,  // Charlie
        },
        txLog: make([]string, 0),
    }
}

// Transaction simulation
type MemTx struct {
    db       *MemDB
    ops      []func() error
    committed bool
    rolledBack bool
}

func (db *MemDB) Begin() *MemTx {
    return &MemTx{db: db, ops: make([]func() error, 0)}
}

func (tx *MemTx) Transfer(fromID, toID int64, amount float64) error {
    if tx.committed || tx.rolledBack {
        return errors.New("transaction already finished")
    }
    tx.ops = append(tx.ops, func() error {
        from, ok := tx.db.accounts[fromID]
        if !ok {
            return fmt.Errorf("account %d not found", fromID)
        }
        if from < amount {
            return fmt.Errorf("insufficient funds: have %.2f, need %.2f", from, amount)
        }
        if _, ok := tx.db.accounts[toID]; !ok {
            return fmt.Errorf("account %d not found", toID)
        }
        tx.db.accounts[fromID] -= amount
        tx.db.accounts[toID] += amount
        tx.db.txLog = append(tx.db.txLog, fmt.Sprintf("transfer %.2f from %d to %d", amount, fromID, toID))
        return nil
    })
    return nil
}

func (tx *MemTx) Commit() error {
    if tx.committed {
        return errors.New("already committed")
    }
    if tx.rolledBack {
        return errors.New("already rolled back")
    }
    tx.db.mu.Lock()
    defer tx.db.mu.Unlock()
    
    // Take snapshot for rollback
    snapshot := make(map[int64]float64)
    for k, v := range tx.db.accounts {
        snapshot[k] = v
    }
    
    for _, op := range tx.ops {
        if err := op(); err != nil {
            // Rollback on any error
            tx.db.accounts = snapshot
            tx.rolledBack = true
            return err
        }
    }
    tx.committed = true
    return nil
}

func (tx *MemTx) Rollback() {
    if !tx.committed && !tx.rolledBack {
        tx.rolledBack = true
    }
}

func (db *MemDB) PrintBalances() {
    db.mu.RLock()
    defer db.mu.RUnlock()
    fmt.Println("  Balances:")
    for id, balance := range db.accounts {
        fmt.Printf("    Account %d: $%.2f\n", id, balance)
    }
}

// Migration system
type Migration struct {
    Version     int
    Description string
    Up          func() error
    Down        func() error
}

type Migrator struct {
    migrations []Migration
    current    int
    log        []string
}

func NewMigrator() *Migrator {
    return &Migrator{current: 0}
}

func (m *Migrator) Add(migration Migration) {
    m.migrations = append(m.migrations, migration)
}

func (m *Migrator) Up() error {
    for m.current < len(m.migrations) {
        mg := m.migrations[m.current]
        fmt.Printf("  Applying migration %d: %s\n", mg.Version, mg.Description)
        if err := mg.Up(); err != nil {
            return fmt.Errorf("migration %d failed: %w", mg.Version, err)
        }
        m.current++
        m.log = append(m.log, fmt.Sprintf("applied: %d %s", mg.Version, mg.Description))
    }
    return nil
}

func (m *Migrator) Down(steps int) error {
    for i := 0; i < steps && m.current > 0; i++ {
        m.current--
        mg := m.migrations[m.current]
        fmt.Printf("  Rolling back migration %d: %s\n", mg.Version, mg.Description)
        if err := mg.Down(); err != nil {
            m.current++ // Restore position
            return fmt.Errorf("rollback %d failed: %w", mg.Version, err)
        }
        m.log = append(m.log, fmt.Sprintf("rolled back: %d %s", mg.Version, mg.Description))
    }
    return nil
}

func (m *Migrator) Status() {
    fmt.Printf("  Migration status: %d/%d applied\n", m.current, len(m.migrations))
    for i, mg := range m.migrations {
        status := "pending"
        if i < m.current {
            status = "applied"
        }
        fmt.Printf("    [%s] %d: %s\n", status, mg.Version, mg.Description)
    }
}

// WithTx helper pattern
func WithTx(db *MemDB, fn func(tx *MemTx) error) error {
    tx := db.Begin()
    defer tx.Rollback() // No-op if committed

    if err := fn(tx); err != nil {
        return err
    }
    return tx.Commit()
}

func main() {
    db := NewMemDB()
    
    // === Transactions ===
    fmt.Println("=== Transaction: Successful Transfer ===")
    db.PrintBalances()
    
    err := WithTx(db, func(tx *MemTx) error {
        return tx.Transfer(1, 2, 200.0) // Alice → Bob $200
    })
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    }
    db.PrintBalances()
    
    fmt.Println("\n=== Transaction: Failed Transfer (Insufficient Funds) ===")
    err = WithTx(db, func(tx *MemTx) error {
        return tx.Transfer(2, 3, 9999.0) // Bob → Charlie $9999 (too much!)
    })
    fmt.Printf("  Error: %v\n", err)
    db.PrintBalances() // Balances unchanged
    
    // === Migrations ===
    fmt.Println("\n=== Migration System ===")
    schema := make(map[string]bool)
    
    migrator := NewMigrator()
    migrator.Add(Migration{
        Version: 1, Description: "create users table",
        Up:   func() error { schema["users"] = true; return nil },
        Down: func() error { delete(schema, "users"); return nil },
    })
    migrator.Add(Migration{
        Version: 2, Description: "create orders table",
        Up:   func() error { schema["orders"] = true; return nil },
        Down: func() error { delete(schema, "orders"); return nil },
    })
    migrator.Add(Migration{
        Version: 3, Description: "add index on users.email",
        Up:   func() error { schema["idx_users_email"] = true; return nil },
        Down: func() error { delete(schema, "idx_users_email"); return nil },
    })
    
    fmt.Println("\n  Before migrations:")
    migrator.Status()
    
    fmt.Println("\n  Running all migrations:")
    migrator.Up()
    migrator.Status()
    fmt.Printf("  Schema: %v\n", schema)
    
    fmt.Println("\n  Rolling back 1 migration:")
    migrator.Down(1)
    migrator.Status()
    fmt.Printf("  Schema: %v\n", schema)
}`,
				},
				{
					Title: "NoSQL and Caching Patterns",
					Content: `Go has excellent support for NoSQL databases and caching layers. Understanding the patterns for Redis, MongoDB, and in-memory caching is essential for modern applications.

**Redis Patterns:**
` + "```" + `
Redis is the most common cache/queue in Go microservices.

Client setup (go-redis):
  rdb := redis.NewClient(&redis.Options{
      Addr:         "localhost:6379",
      Password:     "",
      DB:           0,
      PoolSize:     10,           // Connection pool size
      MinIdleConns: 5,            // Keep idle connections ready
      DialTimeout:  5 * time.Second,
      ReadTimeout:  3 * time.Second,
      WriteTimeout: 3 * time.Second,
  })

Basic operations:
  // String
  rdb.Set(ctx, "key", "value", 10*time.Minute)  // With TTL
  val, err := rdb.Get(ctx, "key").Result()
  if err == redis.Nil { /* key doesn't exist */ }
  
  // Hash
  rdb.HSet(ctx, "user:1", "name", "Alice", "email", "alice@example.com")
  name, _ := rdb.HGet(ctx, "user:1", "name").Result()
  
  // List (queue)
  rdb.LPush(ctx, "queue", "task1", "task2")
  task, _ := rdb.RPop(ctx, "queue").Result()  // FIFO
  
  // Set
  rdb.SAdd(ctx, "online_users", "user1", "user2")
  count, _ := rdb.SCard(ctx, "online_users").Result()
  
  // Sorted Set (leaderboard)
  rdb.ZAdd(ctx, "scores", redis.Z{Score: 100, Member: "player1"})
  top, _ := rdb.ZRevRangeWithScores(ctx, "scores", 0, 9).Result()

Cache-aside pattern:
  func GetUser(ctx context.Context, id string) (*User, error) {
      // Try cache first
      cached, err := rdb.Get(ctx, "user:"+id).Bytes()
      if err == nil {
          var user User
          json.Unmarshal(cached, &user)
          return &user, nil
      }
      
      // Cache miss → hit database
      user, err := db.GetUser(ctx, id)
      if err != nil { return nil, err }
      
      // Write to cache
      data, _ := json.Marshal(user)
      rdb.Set(ctx, "user:"+id, data, 5*time.Minute)
      
      return user, nil
  }

Distributed locking:
  lock := rdb.SetNX(ctx, "lock:resource", "holder_id", 30*time.Second)
  if lock.Val() {
      defer rdb.Del(ctx, "lock:resource")
      // Do exclusive work
  }
  // For production: use Redlock algorithm (multiple Redis instances)
` + "```" + `

**In-Memory Caching:**
` + "```" + `
sync.Map (built-in, no TTL):
  var cache sync.Map
  cache.Store("key", value)
  val, ok := cache.Load("key")
  
  Best for: read-heavy workloads with stable key set
  Not good for: TTL, size limits, eviction

Manual cache with TTL:
  type Cache struct {
      mu    sync.RWMutex
      items map[string]cacheItem
  }
  
  type cacheItem struct {
      value     any
      expiresAt time.Time
  }
  
  func (c *Cache) Get(key string) (any, bool) {
      c.mu.RLock()
      defer c.mu.RUnlock()
      item, ok := c.items[key]
      if !ok || time.Now().After(item.expiresAt) {
          return nil, false
      }
      return item.value, true
  }

LRU Cache:
  Keep track of access order
  Evict least-recently-used when capacity reached
  Use container/list (doubly linked list) + map
  
  Libraries: hashicorp/golang-lru, dgraph-io/ristretto

Cache invalidation strategies:
  1. TTL: Set expiry, tolerate staleness
  2. Write-through: Update cache on every write
  3. Write-behind: Queue cache updates (async)
  4. Event-driven: Database change events trigger invalidation
  5. Version tag: Include version in cache key ("user:1:v5")

"There are only two hard things in CS:
 cache invalidation and naming things."
   — Phil Karlton
` + "```" + ``,
					CodeExamples: `// In-memory cache with TTL and LRU eviction
package main

import (
    "container/list"
    "context"
    "fmt"
    "sync"
    "time"
)

// TTL Cache
type TTLCache struct {
    mu       sync.RWMutex
    items    map[string]ttlItem
    ticker   *time.Ticker
    stopCh   chan struct{}
}

type ttlItem struct {
    value     any
    expiresAt time.Time
}

func NewTTLCache(cleanupInterval time.Duration) *TTLCache {
    c := &TTLCache{
        items:  make(map[string]ttlItem),
        ticker: time.NewTicker(cleanupInterval),
        stopCh: make(chan struct{}),
    }
    go c.cleanup()
    return c
}

func (c *TTLCache) Set(key string, value any, ttl time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.items[key] = ttlItem{
        value:     value,
        expiresAt: time.Now().Add(ttl),
    }
}

func (c *TTLCache) Get(key string) (any, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    item, ok := c.items[key]
    if !ok || time.Now().After(item.expiresAt) {
        return nil, false
    }
    return item.value, true
}

func (c *TTLCache) Delete(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    delete(c.items, key)
}

func (c *TTLCache) cleanup() {
    for {
        select {
        case <-c.ticker.C:
            c.mu.Lock()
            now := time.Now()
            for key, item := range c.items {
                if now.After(item.expiresAt) {
                    delete(c.items, key)
                }
            }
            c.mu.Unlock()
        case <-c.stopCh:
            c.ticker.Stop()
            return
        }
    }
}

func (c *TTLCache) Close() {
    close(c.stopCh)
}

func (c *TTLCache) Len() int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    count := 0
    now := time.Now()
    for _, item := range c.items {
        if !now.After(item.expiresAt) {
            count++
        }
    }
    return count
}

// LRU Cache
type LRUCache struct {
    mu       sync.Mutex
    capacity int
    items    map[string]*list.Element
    order    *list.List
}

type lruEntry struct {
    key   string
    value any
}

func NewLRUCache(capacity int) *LRUCache {
    return &LRUCache{
        capacity: capacity,
        items:    make(map[string]*list.Element),
        order:    list.New(),
    }
}

func (c *LRUCache) Get(key string) (any, bool) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    elem, ok := c.items[key]
    if !ok {
        return nil, false
    }
    // Move to front (most recently used)
    c.order.MoveToFront(elem)
    return elem.Value.(*lruEntry).value, true
}

func (c *LRUCache) Set(key string, value any) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    // Update existing
    if elem, ok := c.items[key]; ok {
        c.order.MoveToFront(elem)
        elem.Value.(*lruEntry).value = value
        return
    }
    
    // Evict oldest if at capacity
    if c.order.Len() >= c.capacity {
        oldest := c.order.Back()
        if oldest != nil {
            c.order.Remove(oldest)
            delete(c.items, oldest.Value.(*lruEntry).key)
        }
    }
    
    // Add new
    entry := &lruEntry{key: key, value: value}
    elem := c.order.PushFront(entry)
    c.items[key] = elem
}

func (c *LRUCache) Len() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.order.Len()
}

// Cache-aside pattern
type CacheAside struct {
    cache *TTLCache
    ttl   time.Duration
}

func NewCacheAside(ttl time.Duration) *CacheAside {
    return &CacheAside{
        cache: NewTTLCache(time.Minute),
        ttl:   ttl,
    }
}

func (ca *CacheAside) GetOrLoad(ctx context.Context, key string, loader func(context.Context, string) (any, error)) (any, error) {
    // Try cache
    if val, ok := ca.cache.Get(key); ok {
        fmt.Printf("  Cache HIT: %s\n", key)
        return val, nil
    }
    
    // Cache miss - load from source
    fmt.Printf("  Cache MISS: %s (loading from source)\n", key)
    val, err := loader(ctx, key)
    if err != nil {
        return nil, err
    }
    
    // Store in cache
    ca.cache.Set(key, val, ca.ttl)
    return val, nil
}

func (ca *CacheAside) Invalidate(key string) {
    ca.cache.Delete(key)
}

func (ca *CacheAside) Close() {
    ca.cache.Close()
}

func main() {
    // TTL Cache demo
    fmt.Println("=== TTL Cache ===")
    cache := NewTTLCache(100 * time.Millisecond)
    defer cache.Close()
    
    cache.Set("session:abc", "user-data", 200*time.Millisecond)
    cache.Set("session:def", "other-data", 500*time.Millisecond)
    
    if val, ok := cache.Get("session:abc"); ok {
        fmt.Printf("  Got: %v\n", val)
    }
    fmt.Printf("  Cache size: %d\n", cache.Len())
    
    time.Sleep(300 * time.Millisecond) // Wait for session:abc to expire
    
    _, ok := cache.Get("session:abc")
    fmt.Printf("  After 300ms, session:abc exists: %v\n", ok)
    fmt.Printf("  Cache size: %d\n", cache.Len())
    
    // LRU Cache demo
    fmt.Println("\n=== LRU Cache (capacity=3) ===")
    lru := NewLRUCache(3)
    
    lru.Set("a", 1)
    lru.Set("b", 2)
    lru.Set("c", 3)
    fmt.Printf("  After a,b,c: size=%d\n", lru.Len())
    
    // Access "a" to make it recently used
    lru.Get("a")
    
    // Add "d" → should evict "b" (least recently used)
    lru.Set("d", 4)
    
    _, hasA := lru.Get("a")
    _, hasB := lru.Get("b")
    _, hasC := lru.Get("c")
    _, hasD := lru.Get("d")
    fmt.Printf("  After adding d: a=%v b=%v c=%v d=%v\n", hasA, hasB, hasC, hasD)
    
    // Cache-aside pattern
    fmt.Println("\n=== Cache-Aside Pattern ===")
    ca := NewCacheAside(5 * time.Second)
    defer ca.Close()
    
    ctx := context.Background()
    
    // Simulated database loader
    loader := func(ctx context.Context, key string) (any, error) {
        time.Sleep(10 * time.Millisecond) // Simulate DB latency
        return fmt.Sprintf("data-for-%s", key), nil
    }
    
    // First access: cache miss
    val, _ := ca.GetOrLoad(ctx, "user:1", loader)
    fmt.Printf("  Result: %v\n", val)
    
    // Second access: cache hit
    val, _ = ca.GetOrLoad(ctx, "user:1", loader)
    fmt.Printf("  Result: %v\n", val)
    
    // Invalidate and re-fetch
    ca.Invalidate("user:1")
    val, _ = ca.GetOrLoad(ctx, "user:1", loader)
    fmt.Printf("  Result: %v\n", val)
}`,
				},
			},
		},
	})
}
