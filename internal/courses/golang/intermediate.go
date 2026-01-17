package golang

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterGolangModules([]problems.CourseModule{
		{
			ID:          34,
			Title:       "Pointers & Memory",
			Description: "Understand pointers, memory management, and pass by value vs reference.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Understanding Pointers",
					Content: `**Pointers:**
- Store memory address of a value
- Type: *Type (pointer to Type)
- Zero value is nil
- Use & to get address
- Use * to dereference

**Why Pointers?**
- Modify function arguments
- Avoid copying large structs
- Share data between functions
- Represent optional values`,
					CodeExamples: `package main

import "fmt"

func main() {
    x := 42
    p := &x  // p is pointer to x
    
    fmt.Println(x)   // 42
    fmt.Println(p)   // Memory address
    fmt.Println(*p)  // 42 (dereference)
    
    *p = 21  // Modify through pointer
    fmt.Println(x)   // 21
}`,
				},
				{
					Title: "Pointer Operations",
					Content: `**Pointer Operations:**
- Address operator: &variable
- Dereference operator: *pointer
- Pointer comparison: ==, !=
- nil pointer check

**Pointer Arithmetic:**
- Go does NOT support pointer arithmetic
- This is intentional for safety`,
					CodeExamples: `package main

import "fmt"

func main() {
    a := 10
    b := 20
    
    p1 := &a
    p2 := &b
    p3 := &a
    
    // Dereference
    fmt.Println(*p1)  // 10
    
    // Comparison
    fmt.Println(p1 == p2)  // false
    fmt.Println(p1 == p3)  // true
    
    // nil check
    var p4 *int
    if p4 == nil {
        fmt.Println("p4 is nil")
    }
}`,
				},
				{
					Title: "Pass by Value vs Reference",
					Content: `**Pass by Value:**
- Go passes arguments by value (copies them)
- Changes to parameters don't affect original
- For large structs, this can be inefficient

**Pass by Reference (Pointer):**
- Pass pointer to allow modification
- More efficient for large structs
- Can represent optional values with nil`,
					CodeExamples: `package main

import "fmt"

// Pass by value
func modifyValue(x int) {
    x = 100
}

// Pass by reference
func modifyPointer(x *int) {
    *x = 100
}

func main() {
    a := 10
    modifyValue(a)
    fmt.Println(a)  // 10 (unchanged)
    
    modifyPointer(&a)
    fmt.Println(a)  // 100 (changed)
}`,
				},
				{
					Title: "Nil Pointers",
					Content: `**Nil Pointers:**
- Zero value for pointers is nil
- Dereferencing nil pointer causes panic
- Always check for nil before dereferencing
- Common pattern: return nil for "not found" cases`,
					CodeExamples: `package main

import "fmt"

func findValue(slice []int, target int) *int {
    for i, v := range slice {
        if v == target {
            return &slice[i]
        }
    }
    return nil  // Not found
}

func main() {
    nums := []int{1, 2, 3, 4, 5}
    
    result := findValue(nums, 3)
    if result != nil {
        fmt.Println("Found:", *result)
    } else {
        fmt.Println("Not found")
    }
}`,
				},
				{
					Title: "Memory Management",
					Content: `**Garbage Collection:**
- Go has automatic garbage collection
- No manual memory management needed
- GC runs automatically in background
- Tries to minimize pause times

**Memory Safety:**
- No pointer arithmetic
- No dangling pointers (GC handles)
- Compile-time type checking
- Runtime bounds checking

**Best Practices:**
- Let GC handle memory
- Use pointers only when needed
- Avoid unnecessary allocations
- Profile if performance is critical`,
					CodeExamples: `package main

import "fmt"

// GC handles cleanup automatically
func createSlice() []int {
    s := make([]int, 1000)
    // s will be garbage collected when no longer referenced
    return s
}

func main() {
    s := createSlice()
    fmt.Println(len(s))
    // s can be garbage collected after main returns
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          35,
			Title:       "Arrays, Slices & Maps",
			Description: "Master arrays, slices, and maps - Go's collection types.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Arrays",
					Content: `**Arrays:**
- Fixed-size sequence of elements
- Size is part of type: [5]int vs [10]int
- Value type (copied when assigned)
- Rarely used directly (slices are preferred)

**Array Operations:**
- Indexing: arr[i]
- Length: len(arr)
- Iteration: for i, v := range arr`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Array declaration
    var arr1 [5]int
    arr2 := [5]int{1, 2, 3, 4, 5}
    arr3 := [...]int{1, 2, 3}  // Compiler determines size
    
    // Accessing elements
    fmt.Println(arr2[0])  // 1
    
    // Length
    fmt.Println(len(arr2))  // 5
    
    // Iteration
    for i, v := range arr2 {
        fmt.Printf("Index %d: %d\n", i, v)
    }
}`,
				},
				{
					Title: "Slices",
					Content: `**Slices:**
- Dynamic arrays (growable)
- Reference type (points to underlying array)
- More commonly used than arrays
- Created with make() or slice literal

**Slice Operations:**
- Creation: make([]Type, length, capacity)
- Append: append(slice, elements...)
- Copy: copy(dest, src)
- Slicing: slice[start:end]`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Slice creation
    var s1 []int
    s2 := []int{1, 2, 3}
    s3 := make([]int, 5)        // length 5, capacity 5
    s4 := make([]int, 0, 10)     // length 0, capacity 10
    
    // Append
    s1 = append(s1, 1, 2, 3)
    s1 = append(s1, 4, 5)
    
    // Slicing
    s5 := s1[1:3]  // [2, 3]
    s6 := s1[:3]   // [1, 2, 3]
    s7 := s1[2:]   // [3, 4, 5]
    
    // Length and capacity
    fmt.Println(len(s1), cap(s1))
    
    fmt.Println(s1, s2, s3, s4, s5, s6, s7)
}`,
				},
				{
					Title: "Slice Internals",
					Content: `**Slice Structure:**
A slice is a struct containing:
- Pointer to underlying array
- Length (current elements)
- Capacity (total allocated space)

**Important Behaviors:**
- Slicing creates new slice but shares underlying array
- Modifying slice can affect other slices sharing same array
- Append may create new array if capacity exceeded
- Use copy() to create independent slice`,
					CodeExamples: `package main

import "fmt"

func main() {
    original := []int{1, 2, 3, 4, 5}
    
    // Slice shares underlying array
    slice1 := original[1:4]  // [2, 3, 4]
    slice1[0] = 99
    fmt.Println(original)  // [1, 99, 3, 4, 5] - modified!
    
    // Independent copy
    slice2 := make([]int, len(original))
    copy(slice2, original)
    slice2[0] = 100
    fmt.Println(original)  // [1, 99, 3, 4, 5] - unchanged
    fmt.Println(slice2)    // [100, 99, 3, 4, 5]
}`,
				},
				{
					Title: "Maps",
					Content: `**Maps:**
- Key-value pairs (hash table/dictionary)
- Reference type
- Zero value is nil
- Keys must be comparable types

**Map Operations:**
- Creation: make(map[KeyType]ValueType)
- Access: m[key]
- Assignment: m[key] = value
- Delete: delete(m, key)
- Check existence: value, ok := m[key]`,
					CodeExamples: `package main

import "fmt"

func main() {
    // Map creation
    var m1 map[string]int
    m2 := make(map[string]int)
    m3 := map[string]int{
        "apple":  5,
        "banana": 3,
    }
    
    // Operations
    m2["one"] = 1
    m2["two"] = 2
    
    // Access
    value := m2["one"]
    fmt.Println(value)
    
    // Check existence
    v, ok := m2["three"]
    if ok {
        fmt.Println("Found:", v)
    } else {
        fmt.Println("Not found")
    }
    
    // Delete
    delete(m2, "one")
    
    // Iteration
    for key, value := range m3 {
        fmt.Printf("%s: %d\n", key, value)
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          36,
			Title:       "Structs & Methods",
			Description: "Learn structs, methods, receivers, and embedded types.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Struct Definition",
					Content: `**Structs:**
- Collection of fields (like classes without inheritance)
- Value type
- Can be initialized in multiple ways
- Fields can have tags for metadata

**Struct Syntax:**
type StructName struct {
    Field1 Type1
    Field2 Type2
}`,
					CodeExamples: `package main

import "fmt"

type Person struct {
    Name string
    Age  int
    City string
}

func main() {
    // Zero value
    var p1 Person
    
    // Literal initialization
    p2 := Person{"Alice", 30, "NYC"}
    p3 := Person{Name: "Bob", Age: 25}
    p4 := Person{
        Name: "Charlie",
        Age:  35,
        City: "LA",
    }
    
    // Access fields
    fmt.Println(p2.Name)
    p2.Age = 31
    
    fmt.Println(p1, p2, p3, p4)
}`,
				},
				{
					Title: "Methods and Receivers",
					Content: `**Methods:**
- Functions with receiver argument
- Receiver can be value or pointer
- Value receiver: works on copy
- Pointer receiver: works on original

**Method Syntax:**
func (receiver ReceiverType) MethodName() ReturnType {
    // body
}`,
					CodeExamples: `package main

import "fmt"

type Rectangle struct {
    Width  float64
    Height float64
}

// Value receiver
func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

// Pointer receiver (can modify)
func (r *Rectangle) Scale(factor float64) {
    r.Width *= factor
    r.Height *= factor
}

func main() {
    rect := Rectangle{Width: 10, Height: 5}
    fmt.Println(rect.Area())  // 50
    
    rect.Scale(2)
    fmt.Println(rect.Width, rect.Height)  // 20, 10
}`,
				},
				{
					Title: "Method Sets",
					Content: `**Method Sets:**
- Value receiver: works with both value and pointer
- Pointer receiver: works only with pointer
- Interface satisfaction depends on method set

**Best Practice:**
- Use pointer receivers when method modifies struct
- Use pointer receivers for large structs (efficiency)
- Use value receivers for small, immutable structs`,
					CodeExamples: `package main

import "fmt"

type Counter struct {
    value int
}

// Pointer receiver
func (c *Counter) Increment() {
    c.value++
}

// Value receiver
func (c Counter) GetValue() int {
    return c.value
}

func main() {
    c := Counter{value: 0}
    
    // Both work
    c.Increment()      // Go automatically converts
    (&c).Increment()  // Explicit pointer
    
    fmt.Println(c.GetValue())  // 2
}`,
				},
				{
					Title: "Embedded Structs",
					Content: `**Embedding:**
- Struct can embed another struct
- Fields of embedded struct are promoted
- Similar to composition (not inheritance)
- Can override methods

**Embedding Syntax:**
type Outer struct {
    Inner  // Embedded (no field name)
    Field  // Regular field
}`,
					CodeExamples: `package main

import "fmt"

type Engine struct {
    Power int
}

func (e Engine) Start() {
    fmt.Println("Engine started")
}

type Car struct {
    Engine  // Embedded
    Brand   string
}

func main() {
    car := Car{
        Engine: Engine{Power: 200},
        Brand:  "Toyota",
    }
    
    // Promoted fields and methods
    fmt.Println(car.Power)  // Direct access
    car.Start()             // Direct access
}`,
				},
				{
					Title: "Field Tags",
					Content: `**Field Tags:**
- Metadata attached to struct fields
- Used by encoding packages (JSON, XML, etc.)
- Format: ` + "`" + `tag:"value"` + "`" + `
- Accessed via reflection

**Common Uses:**
- JSON encoding/decoding
- Database ORM mapping
- Validation`,
					CodeExamples: `package main

import (
    "encoding/json"
    "fmt"
)

type User struct {
    ID       int    ` + "`" + `json:"id"` + "`" + `
    Username string ` + "`" + `json:"username"` + "`" + `
    Email    string ` + "`" + `json:"email,omitempty"` + "`" + `
    Password string ` + "`" + `json:"-"` + "`" + `  // Ignored
}

func main() {
    u := User{
        ID:       1,
        Username: "alice",
        Email:    "alice@example.com",
        Password: "secret",
    }
    
    data, _ := json.Marshal(u)
    fmt.Println(string(data))
    // {"id":1,"username":"alice","email":"alice@example.com"}
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          37,
			Title:       "Interfaces",
			Description: "Master interfaces, type assertions, and polymorphism in Go.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Interface Definition",
					Content: `**Interfaces:**
- Define set of methods
- Implicit satisfaction (no explicit declaration)
- Type satisfies interface if it implements all methods
- Can be empty (interface{})

**Interface Syntax:**
type InterfaceName interface {
    Method1() ReturnType
    Method2(param Type) ReturnType
}`,
					CodeExamples: `package main

import "fmt"

type Shape interface {
    Area() float64
    Perimeter() float64
}

type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func main() {
    var s Shape = Rectangle{Width: 10, Height: 5}
    fmt.Println(s.Area())
}`,
				},
				{
					Title: "Implicit Satisfaction",
					Content: `**Implicit Interface Satisfaction:**
- No "implements" keyword
- Type automatically satisfies interface if methods match
- Enables decoupling and flexibility
- Interfaces are satisfied implicitly at compile time`,
					CodeExamples: `package main

import "fmt"

type Writer interface {
    Write([]byte) (int, error)
}

type File struct {
    name string
}

func (f File) Write(data []byte) (int, error) {
    fmt.Printf("Writing to %s: %s\n", f.name, string(data))
    return len(data), nil
}

// File automatically satisfies Writer interface
func save(w Writer, data []byte) {
    w.Write(data)
}

func main() {
    f := File{name: "output.txt"}
    save(f, []byte("Hello"))
}`,
				},
				{
					Title: "Empty Interface",
					Content: `**Empty Interface (interface{}):**
- Can hold any type
- Used for generic-like behavior (before Go 1.18)
- Type assertions needed to use
- Common in standard library`,
					CodeExamples: `package main

import "fmt"

func printValue(v interface{}) {
    // Type assertion
    switch val := v.(type) {
    case int:
        fmt.Printf("Integer: %d\n", val)
    case string:
        fmt.Printf("String: %s\n", val)
    case bool:
        fmt.Printf("Boolean: %v\n", val)
    default:
        fmt.Printf("Unknown type: %v\n", val)
    }
}

func main() {
    printValue(42)
    printValue("hello")
    printValue(true)
}`,
				},
				{
					Title: "Type Assertions",
					Content: `**Type Assertions:**
- Extract concrete type from interface
- Syntax: value, ok := interface.(Type)
- Returns value and boolean indicating success
- Panics if type doesn't match (without ok check)

**Type Switches:**
- Switch on type of interface value
- More convenient than multiple assertions`,
					CodeExamples: `package main

import "fmt"

func processValue(v interface{}) {
    // Type assertion with ok check
    if str, ok := v.(string); ok {
        fmt.Printf("String: %s\n", str)
    } else if num, ok := v.(int); ok {
        fmt.Printf("Number: %d\n", num)
    }
    
    // Type switch
    switch val := v.(type) {
    case string:
        fmt.Printf("String: %s\n", val)
    case int:
        fmt.Printf("Int: %d\n", val)
    case float64:
        fmt.Printf("Float: %f\n", val)
    default:
        fmt.Printf("Unknown: %v\n", val)
    }
}

func main() {
    processValue("hello")
    processValue(42)
    processValue(3.14)
}`,
				},
				{
					Title: "Interface Composition",
					Content: `**Interface Composition:**
- Interfaces can embed other interfaces
- Creates larger interface from smaller ones
- Type must implement all embedded methods
- Promotes code reuse and flexibility`,
					CodeExamples: `package main

import "fmt"

type Reader interface {
    Read([]byte) (int, error)
}

type Writer interface {
    Write([]byte) (int, error)
}

// Composed interface
type ReadWriter interface {
    Reader
    Writer
}

type File struct {
    data []byte
}

func (f *File) Read(b []byte) (int, error) {
    copy(b, f.data)
    return len(f.data), nil
}

func (f *File) Write(b []byte) (int, error) {
    f.data = append(f.data, b...)
    return len(b), nil
}

func main() {
    var rw ReadWriter = &File{}
    rw.Write([]byte("Hello"))
    
    buf := make([]byte, 10)
    rw.Read(buf)
    fmt.Println(string(buf))
}`,
				},
				{
					Title: "Common Interfaces",
					Content: `**Standard Library Interfaces:**

- **io.Reader**: Read data
- **io.Writer**: Write data
- **io.Closer**: Close resource
- **fmt.Stringer**: String representation
- **error**: Error values
- **sort.Interface**: Sorting

**Implementing Standard Interfaces:**
Makes your types work with standard library functions.`,
					CodeExamples: `package main

import (
    "fmt"
    "io"
    "os"
)

type Counter struct {
    count int
}

func (c *Counter) Write(p []byte) (int, error) {
    c.count += len(p)
    return len(p), nil
}

func main() {
    var w io.Writer = &Counter{}
    fmt.Fprintf(w, "Hello")
    
    // Counter implements io.Writer
    fmt.Fprintf(os.Stdout, "World")
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
