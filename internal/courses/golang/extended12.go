package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          1631,
			Title:       "Go Generics and Modern Idioms",
			Description: "Master Go generics (type parameters), iterators, modern error handling, and Go 1.21-1.23 features.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "Generics Deep Dive",
					Content: `Go 1.18 introduced generics (type parameters). They enable type-safe, reusable code without reflection or code generation.

**Type Parameters:**
` + "```" + `
Basic syntax:
  func Map[T, U any](s []T, f func(T) U) []U {
      result := make([]U, len(s))
      for i, v := range s {
          result[i] = f(v)
      }
      return result
  }
  
  // Usage (type inference):
  strs := Map([]int{1, 2, 3}, strconv.Itoa)     // []string
  doubled := Map([]int{1, 2, 3}, func(n int) int { return n * 2 })

Type constraints:
  any           → any type (alias for interface{})
  comparable    → supports == and != (maps keys, etc.)
  ~int          → underlying type is int (includes type aliases)
  
  Custom constraint:
  type Number interface {
      ~int | ~int8 | ~int16 | ~int32 | ~int64 |
      ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
      ~float32 | ~float64
  }
  
  func Sum[T Number](nums []T) T {
      var total T
      for _, n := range nums {
          total += n
      }
      return total
  }
  
  Sum([]int{1, 2, 3})           // 6
  Sum([]float64{1.5, 2.5, 3.0}) // 7.0

cmp.Ordered (stdlib):
  import "cmp"
  
  // cmp.Ordered: any type that supports < > <= >=
  func Max[T cmp.Ordered](a, b T) T {
      if a > b { return a }
      return b
  }

Generic types:
  type Stack[T any] struct {
      items []T
  }
  
  func (s *Stack[T]) Push(item T) {
      s.items = append(s.items, item)
  }
  
  func (s *Stack[T]) Pop() (T, bool) {
      var zero T
      if len(s.items) == 0 {
          return zero, false
      }
      item := s.items[len(s.items)-1]
      s.items = s.items[:len(s.items)-1]
      return item, true
  }
  
  intStack := &Stack[int]{}
  strStack := &Stack[string]{}
` + "```" + `

**When to Use Generics:**
` + "```" + `
USE generics for:
  ✓ Container types (Stack, Queue, Set, Tree)
  ✓ Collection operations (Map, Filter, Reduce)
  ✓ Type-safe APIs that work across types
  ✓ Eliminating interface{}/any + type assertions
  
DON'T use generics for:
  ✗ When a simple interface works
  ✗ When the implementation differs per type
  ✗ Just to be "generic" (YAGNI)
  ✗ When it makes code harder to read

Rule of thumb:
  If you're writing the same function for multiple types
  and the LOGIC is identical → use generics
  
  If the logic differs per type → use interfaces
  
  Example:
    Generic: func Contains[T comparable](s []T, v T) bool
    Interface: func Sort(s sort.Interface) — behavior differs per type

Performance:
  Go generics use "GC shape stenciling":
    - One copy of code per GC shape (pointer, int, float, etc.)
    - NOT one copy per type (unlike C++ templates)
    - Slight overhead for non-pointer types (dictionary passing)
    - For most code: negligible performance impact
` + "```" + ``,
					CodeExamples: `// Generics patterns in Go
package main

import (
    "cmp"
    "fmt"
    "strings"
)

// Generic data structures

// Set
type Set[T comparable] struct {
    items map[T]struct{}
}

func NewSet[T comparable](items ...T) *Set[T] {
    s := &Set[T]{items: make(map[T]struct{})}
    for _, item := range items {
        s.Add(item)
    }
    return s
}

func (s *Set[T]) Add(item T) { s.items[item] = struct{}{} }

func (s *Set[T]) Remove(item T) { delete(s.items, item) }

func (s *Set[T]) Contains(item T) bool {
    _, ok := s.items[item]
    return ok
}

func (s *Set[T]) Len() int { return len(s.items) }

func (s *Set[T]) Union(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        result.Add(item)
    }
    for item := range other.items {
        result.Add(item)
    }
    return result
}

func (s *Set[T]) Intersection(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        if other.Contains(item) {
            result.Add(item)
        }
    }
    return result
}

func (s *Set[T]) Difference(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        if !other.Contains(item) {
            result.Add(item)
        }
    }
    return result
}

func (s *Set[T]) ToSlice() []T {
    result := make([]T, 0, len(s.items))
    for item := range s.items {
        result = append(result, item)
    }
    return result
}

// Pair
type Pair[T, U any] struct {
    First  T
    Second U
}

func NewPair[T, U any](first T, second U) Pair[T, U] {
    return Pair[T, U]{First: first, Second: second}
}

// Result (Either/Result pattern)
type Result[T any] struct {
    value T
    err   error
}

func Ok[T any](value T) Result[T] {
    return Result[T]{value: value}
}

func Err[T any](err error) Result[T] {
    return Result[T]{err: err}
}

func (r Result[T]) Unwrap() (T, error) {
    return r.value, r.err
}

func (r Result[T]) IsOk() bool {
    return r.err == nil
}

func (r Result[T]) Map(fn func(T) T) Result[T] {
    if r.err != nil {
        return r
    }
    return Ok(fn(r.value))
}

// Generic collection operations

func Map[T, U any](s []T, f func(T) U) []U {
    result := make([]U, len(s))
    for i, v := range s {
        result[i] = f(v)
    }
    return result
}

func Filter[T any](s []T, pred func(T) bool) []T {
    var result []T
    for _, v := range s {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

func Reduce[T, U any](s []T, initial U, fn func(U, T) U) U {
    result := initial
    for _, v := range s {
        result = fn(result, v)
    }
    return result
}

func Contains[T comparable](s []T, target T) bool {
    for _, v := range s {
        if v == target {
            return true
        }
    }
    return false
}

func Unique[T comparable](s []T) []T {
    seen := make(map[T]struct{})
    var result []T
    for _, v := range s {
        if _, ok := seen[v]; !ok {
            seen[v] = struct{}{}
            result = append(result, v)
        }
    }
    return result
}

func GroupBy[T any, K comparable](s []T, keyFn func(T) K) map[K][]T {
    result := make(map[K][]T)
    for _, v := range s {
        key := keyFn(v)
        result[key] = append(result[key], v)
    }
    return result
}

func MinMax[T cmp.Ordered](s []T) (T, T) {
    if len(s) == 0 {
        var zero T
        return zero, zero
    }
    min, max := s[0], s[0]
    for _, v := range s[1:] {
        if v < min {
            min = v
        }
        if v > max {
            max = v
        }
    }
    return min, max
}

func Chunk[T any](s []T, size int) [][]T {
    var result [][]T
    for i := 0; i < len(s); i += size {
        end := i + size
        if end > len(s) {
            end = len(s)
        }
        result = append(result, s[i:end])
    }
    return result
}

func Zip[T, U any](a []T, b []U) []Pair[T, U] {
    minLen := len(a)
    if len(b) < minLen {
        minLen = len(b)
    }
    result := make([]Pair[T, U], minLen)
    for i := 0; i < minLen; i++ {
        result[i] = NewPair(a[i], b[i])
    }
    return result
}

func main() {
    // Generic Set
    fmt.Println("=== Generic Set ===")
    
    s1 := NewSet(1, 2, 3, 4, 5)
    s2 := NewSet(3, 4, 5, 6, 7)
    
    fmt.Printf("  s1: %v\n", s1.ToSlice())
    fmt.Printf("  s2: %v\n", s2.ToSlice())
    fmt.Printf("  Union: %v\n", s1.Union(s2).ToSlice())
    fmt.Printf("  Intersection: %v\n", s1.Intersection(s2).ToSlice())
    fmt.Printf("  Difference: %v\n", s1.Difference(s2).ToSlice())
    
    // String set
    tags := NewSet("go", "rust", "python", "go")
    fmt.Printf("  String set (deduped): %v (%d items)\n", tags.ToSlice(), tags.Len())
    
    // Collection operations
    fmt.Println("\n=== Generic Collections ===")
    
    nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    
    // Map
    doubled := Map(nums, func(n int) int { return n * 2 })
    fmt.Printf("  Doubled: %v\n", doubled)
    
    strs := Map(nums, func(n int) string { return fmt.Sprintf("#%d", n) })
    fmt.Printf("  Strings: %v\n", strs)
    
    // Filter
    evens := Filter(nums, func(n int) bool { return n%2 == 0 })
    fmt.Printf("  Evens: %v\n", evens)
    
    // Reduce
    sum := Reduce(nums, 0, func(acc, n int) int { return acc + n })
    fmt.Printf("  Sum: %d\n", sum)
    
    joined := Reduce(strs, "", func(acc, s string) string {
        if acc == "" { return s }
        return acc + ", " + s
    })
    fmt.Printf("  Joined: %s\n", joined)
    
    // Contains
    fmt.Printf("  Contains 5: %v\n", Contains(nums, 5))
    fmt.Printf("  Contains 11: %v\n", Contains(nums, 11))
    
    // Unique
    withDups := []int{1, 2, 2, 3, 3, 3, 4, 4, 5}
    fmt.Printf("  Unique: %v\n", Unique(withDups))
    
    // GroupBy
    words := []string{"apple", "banana", "avocado", "blueberry", "cherry", "apricot"}
    grouped := GroupBy(words, func(s string) string { return string(s[0]) })
    fmt.Println("  GroupBy first letter:")
    for k, v := range grouped {
        fmt.Printf("    %s: %v\n", k, v)
    }
    
    // MinMax
    min, max := MinMax(nums)
    fmt.Printf("  Min: %d, Max: %d\n", min, max)
    
    minS, maxS := MinMax([]string{"banana", "apple", "cherry"})
    fmt.Printf("  Min: %s, Max: %s\n", minS, maxS)
    
    // Chunk
    chunks := Chunk(nums, 3)
    fmt.Printf("  Chunks(3): %v\n", chunks)
    
    // Zip
    names := []string{"Alice", "Bob", "Charlie"}
    ages := []int{30, 25, 35}
    pairs := Zip(names, ages)
    fmt.Println("  Zip:")
    for _, p := range pairs {
        fmt.Printf("    %s: %d\n", p.First, p.Second)
    }
    
    // Result type
    fmt.Println("\n=== Result Type ===")
    
    r1 := Ok(42)
    r2 := r1.Map(func(n int) int { return n * 2 })
    val, err := r2.Unwrap()
    fmt.Printf("  Ok(42).Map(*2) = %d, err=%v\n", val, err)
    
    r3 := Err[int](fmt.Errorf("something failed"))
    r4 := r3.Map(func(n int) int { return n * 2 }) // Map skipped for errors
    val, err = r4.Unwrap()
    fmt.Printf("  Err.Map(*2) = %d, err=%v\n", val, err)
    
    // Pair usage
    fmt.Println("\n=== Pair ===")
    coords := []Pair[float64, float64]{
        NewPair(1.0, 2.0),
        NewPair(3.0, 4.0),
        NewPair(5.0, 6.0),
    }
    for _, c := range coords {
        fmt.Printf("  (%.1f, %.1f)\n", c.First, c.Second)
    }
    
    // String operations
    fmt.Println("\n=== String Generic Ops ===")
    wordList := []string{"hello", "world", "foo", "bar", "hello", "baz", "foo"}
    upper := Map(wordList, strings.ToUpper)
    fmt.Printf("  Upper: %v\n", upper)
    
    long := Filter(wordList, func(s string) bool { return len(s) > 3 })
    fmt.Printf("  Long (>3): %v\n", long)
    
    uniqueWords := Unique(wordList)
    fmt.Printf("  Unique: %v\n", uniqueWords)
}`,
				},
				{
					Title: "Modern Go Features (1.21-1.23)",
					Content: `Go evolves with each release. Understanding the latest features helps you write more idiomatic and efficient code.

**Go 1.21 Features:**
` + "```" + `
slices package (standard library):
  import "slices"
  
  slices.Sort(s)                          // Sort in place
  slices.SortFunc(s, func(a, b T) int)   // Custom sort
  slices.Contains(s, val)                 // Check if exists
  slices.Index(s, val)                    // Find index (-1 if not found)  
  slices.Equal(a, b)                      // Compare slices
  slices.Compact(s)                       // Remove consecutive duplicates
  slices.Clone(s)                         // Shallow copy
  slices.Reverse(s)                       // Reverse in place
  slices.Min(s), slices.Max(s)            // Min/max
  slices.BinarySearch(s, val)             // Binary search (sorted)
  slices.Delete(s, i, j)                  // Delete range [i,j)
  slices.Insert(s, i, vals...)            // Insert at position
  slices.Replace(s, i, j, vals...)        // Replace range

maps package:
  import "maps"
  
  maps.Equal(a, b)       // Compare maps
  maps.Clone(m)          // Shallow copy
  maps.Copy(dst, src)    // Copy entries
  maps.DeleteFunc(m, fn) // Delete matching entries
  maps.Keys(m)           // Get all keys
  maps.Values(m)         // Get all values

Built-in functions:
  min(a, b)   // Built-in min (variadic)
  max(a, b)   // Built-in max (variadic)
  clear(m)    // Clear map or zero slice elements
  
  // No more sort.Min or custom min functions!
  smallest := min(1, 2, 3)     // 1
  largest := max("a", "b", "c") // "c"

log/slog (structured logging):
  slog.Info("request handled",
      "method", "GET",
      "path", "/api/users",
      "status", 200,
      "duration", 42*time.Millisecond,
  )

sync.OnceFunc, sync.OnceValue, sync.OnceValues:
  load := sync.OnceFunc(func() {
      config = loadExpensiveConfig()
  })
  load() // First call executes
  load() // Subsequent calls are no-ops
  
  getConfig := sync.OnceValue(func() *Config {
      return loadConfig()
  })
  cfg := getConfig() // Loaded once, cached
` + "```" + `

**Go 1.22 Features:**
` + "```" + `
Range over integers:
  for i := range 10 {
      fmt.Println(i) // 0, 1, 2, ..., 9
  }
  
  // Replaces:
  for i := 0; i < 10; i++ { ... }

For loop variable fix:
  // Before Go 1.22 (BUG!):
  for _, v := range values {
      go func() {
          fmt.Println(v) // All print same value!
      }()
  }
  
  // Go 1.22+: each iteration gets its own variable
  for _, v := range values {
      go func() {
          fmt.Println(v) // Each prints correct value
      }()
  }

Enhanced ServeMux:
  mux.HandleFunc("GET /api/users", listUsers)
  mux.HandleFunc("POST /api/users", createUser)
  mux.HandleFunc("GET /api/users/{id}", getUser)
  
  id := r.PathValue("id") // Extract path parameter

math/rand/v2:
  import "math/rand/v2"
  
  rand.IntN(100)           // [0, 100)
  rand.Float64()           // [0.0, 1.0)
  rand.N(10 * time.Second) // Random duration [0, 10s)
  rand.Shuffle(n, swap)    // Shuffle
  // Auto-seeded, no need for rand.Seed()!
` + "```" + `

**Go 1.23 Features:**
` + "```" + `
Range over function (iterators):
  // Iterator function signature:
  func(yield func(V) bool)           // Single value
  func(yield func(K, V) bool)        // Key-value pair
  
  // Example: iterate over lines
  func Lines(s string) iter.Seq[string] {
      return func(yield func(string) bool) {
          for _, line := range strings.Split(s, "\n") {
              if !yield(line) { return }
          }
      }
  }
  
  for line := range Lines("hello\nworld\nfoo") {
      fmt.Println(line)
  }

  // Pair iterator:
  func Enumerate[T any](s []T) iter.Seq2[int, T] {
      return func(yield func(int, T) bool) {
          for i, v := range s {
              if !yield(i, v) { return }
          }
      }
  }
  
  for i, v := range Enumerate([]string{"a", "b", "c"}) {
      fmt.Printf("%d: %s\n", i, v)
  }

  slices package iterator support:
    slices.All(s)         → iter.Seq2[int, T]
    slices.Values(s)      → iter.Seq[T]
    slices.Backward(s)    → iter.Seq2[int, T]
    slices.Collect(seq)   → []T
    slices.Sorted(seq)    → []T
    
  maps package:
    maps.All(m)           → iter.Seq2[K, V]
    maps.Keys(m)          → iter.Seq[K]
    maps.Values(m)        → iter.Seq[V]
    maps.Collect(seq)     → map[K]V

unique package (string/value interning):
  import "unique"
  
  h1 := unique.Make("hello")
  h2 := unique.Make("hello")
  h1 == h2 // true (same handle, O(1) comparison)
  // Reduces memory for repeated string values

structs package (experimental):
  structs.HostLayout → ensures struct layout matches C ABI
` + "```" + ``,
					CodeExamples: `// Modern Go features demonstration
package main

import (
    "cmp"
    "fmt"
    "maps"
    "slices"
    "strings"
)

// Iterator patterns (Go 1.23 style, using function types)

// Generic iterator type (simplified iter.Seq)
type Seq[V any] func(yield func(V) bool)
type Seq2[K, V any] func(yield func(K, V) bool)

// Iterator constructors
func FromSlice[T any](s []T) Seq[T] {
    return func(yield func(T) bool) {
        for _, v := range s {
            if !yield(v) { return }
        }
    }
}

func Enumerate[T any](s []T) Seq2[int, T] {
    return func(yield func(int, T) bool) {
        for i, v := range s {
            if !yield(i, v) { return }
        }
    }
}

func Range(start, end int) Seq[int] {
    return func(yield func(int) bool) {
        for i := start; i < end; i++ {
            if !yield(i) { return }
        }
    }
}

func Lines(s string) Seq[string] {
    return func(yield func(string) bool) {
        for _, line := range strings.Split(s, "\n") {
            if !yield(line) { return }
        }
    }
}

// Iterator combinators
func MapIter[T, U any](seq Seq[T], fn func(T) U) Seq[U] {
    return func(yield func(U) bool) {
        seq(func(v T) bool {
            return yield(fn(v))
        })
    }
}

func FilterIter[T any](seq Seq[T], pred func(T) bool) Seq[T] {
    return func(yield func(T) bool) {
        seq(func(v T) bool {
            if pred(v) {
                return yield(v)
            }
            return true
        })
    }
}

func TakeIter[T any](seq Seq[T], n int) Seq[T] {
    return func(yield func(T) bool) {
        count := 0
        seq(func(v T) bool {
            if count >= n { return false }
            count++
            return yield(v)
        })
    }
}

// Collect iterator into slice
func Collect[T any](seq Seq[T]) []T {
    var result []T
    seq(func(v T) bool {
        result = append(result, v)
        return true
    })
    return result
}

// ForEach
func ForEach[T any](seq Seq[T], fn func(T)) {
    seq(func(v T) bool {
        fn(v)
        return true
    })
}

// Count
func Count[T any](seq Seq[T]) int {
    n := 0
    seq(func(T) bool {
        n++
        return true
    })
    return n
}

func main() {
    // slices package
    fmt.Println("=== slices package ===")
    
    nums := []int{5, 3, 8, 1, 9, 2, 7, 4, 6}
    sorted := slices.Clone(nums)
    slices.Sort(sorted)
    fmt.Printf("  Original: %v\n", nums)
    fmt.Printf("  Sorted:   %v\n", sorted)
    
    fmt.Printf("  Contains 5: %v\n", slices.Contains(nums, 5))
    fmt.Printf("  Index of 8: %d\n", slices.Index(nums, 8))
    fmt.Printf("  Min: %d, Max: %d\n", slices.Min(nums), slices.Max(nums))
    
    // Binary search on sorted slice
    idx, found := slices.BinarySearch(sorted, 7)
    fmt.Printf("  BinarySearch(7): idx=%d found=%v\n", idx, found)
    
    // Compact (remove consecutive duplicates)
    dups := []int{1, 1, 2, 2, 2, 3, 3, 1, 1}
    compacted := slices.Compact(slices.Clone(dups))
    fmt.Printf("  Compact(%v) = %v\n", dups, compacted)
    
    // SortFunc (custom)
    words := []string{"banana", "apple", "cherry", "date"}
    slices.SortFunc(words, func(a, b string) int {
        return cmp.Compare(len(a), len(b)) // Sort by length
    })
    fmt.Printf("  Sort by length: %v\n", words)
    
    // maps package
    fmt.Println("\n=== maps package ===")
    
    m1 := map[string]int{"a": 1, "b": 2, "c": 3}
    m2 := maps.Clone(m1)
    m2["d"] = 4
    
    fmt.Printf("  m1: %v\n", m1)
    fmt.Printf("  m2 (cloned + d): %v\n", m2)
    fmt.Printf("  Equal: %v\n", maps.Equal(m1, m1))
    
    // Delete entries matching predicate
    evens := maps.Clone(m1)
    maps.DeleteFunc(evens, func(k string, v int) bool {
        return v%2 != 0 // Delete odd values
    })
    fmt.Printf("  After DeleteFunc(odd): %v\n", evens)
    
    // Built-in min/max
    fmt.Println("\n=== Built-in min/max ===")
    fmt.Printf("  min(3, 1, 4, 1, 5): %d\n", min(3, 1, 4, 1, 5))
    fmt.Printf("  max(3, 1, 4, 1, 5): %d\n", max(3, 1, 4, 1, 5))
    fmt.Printf("  min(\"banana\", \"apple\"): %s\n", min("banana", "apple"))
    
    // Range over integers (Go 1.22)
    fmt.Println("\n=== Range over integers ===")
    fmt.Print("  range 5: ")
    for i := range 5 {
        fmt.Printf("%d ", i)
    }
    fmt.Println()
    
    // Iterator patterns
    fmt.Println("\n=== Iterator Patterns ===")
    
    // Basic iteration
    seq := FromSlice([]string{"hello", "world", "foo", "bar"})
    fmt.Print("  FromSlice: ")
    ForEach(seq, func(s string) { fmt.Printf("%s ", s) })
    fmt.Println()
    
    // Map + Filter + Collect
    numbers := FromSlice([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
    
    pipeline := MapIter(
        FilterIter(numbers, func(n int) bool { return n%2 == 0 }),
        func(n int) int { return n * n },
    )
    result := Collect(pipeline)
    fmt.Printf("  Evens squared: %v\n", result)
    
    // Take first N
    first3 := Collect(TakeIter(FromSlice(sorted), 3))
    fmt.Printf("  First 3 sorted: %v\n", first3)
    
    // Lines iterator
    text := "Hello World\nFoo Bar\nBaz Qux"
    fmt.Println("  Lines:")
    ForEach(Lines(text), func(line string) {
        fmt.Printf("    > %s\n", line)
    })
    
    // Range iterator
    fmt.Print("  Range(5,10): ")
    ForEach(Range(5, 10), func(n int) { fmt.Printf("%d ", n) })
    fmt.Println()
    
    // Count
    evenCount := Count(FilterIter(FromSlice(nums), func(n int) bool { return n%2 == 0 }))
    fmt.Printf("  Even count in %v: %d\n", nums, evenCount)
    
    // Enumerate
    fmt.Println("\n  Enumerate:")
    Enumerate([]string{"go", "rust", "python"})(func(i int, v string) bool {
        fmt.Printf("    [%d] %s\n", i, v)
        return true
    })
    
    // Chained operations
    fmt.Println("\n=== Chained Operations ===")
    
    data := []string{"apple", "banana", "avocado", "blueberry", "cherry", "apricot", "blackberry"}
    
    // Filter words starting with 'b', map to uppercase, take first 2
    bWords := Collect(TakeIter(
        MapIter(
            FilterIter(FromSlice(data), func(s string) bool {
                return strings.HasPrefix(s, "b")
            }),
            strings.ToUpper,
        ),
        2,
    ))
    fmt.Printf("  First 2 'b' words (upper): %v\n", bWords)
    
    // clear() built-in
    fmt.Println("\n=== clear() built-in ===")
    clearMap := map[string]int{"a": 1, "b": 2}
    fmt.Printf("  Before clear: %v\n", clearMap)
    clear(clearMap)
    fmt.Printf("  After clear:  %v (len=%d)\n", clearMap, len(clearMap))
    
    clearSlice := []int{1, 2, 3, 4, 5}
    fmt.Printf("  Slice before clear: %v\n", clearSlice)
    clear(clearSlice)
    fmt.Printf("  Slice after clear:  %v (len=%d)\n", clearSlice, len(clearSlice))
}`,
				},
			},
		},
	})
}
