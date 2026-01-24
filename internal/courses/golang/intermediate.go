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
					Content: `Pointers are just variables that hold the memory address of another variable. They are crucial in Go for sharing data and efficient memory usage.

**The "&" and "*" Operators:**
*   **& (Address Of):** "Where is this variable stored?" -> Returns memory address.
*   **(*) (Dereference):** "What value is at this address?" -> Returns the value.

**Stack vs Heap (Simplified):**
*   **Stack:** Fast, local memory. Variables created here are cleaned up when the function returns.
*   **Heap:** Persistent memory. Variables created here stay until the Garbage Collector (GC) decides they are no longer needed.
*   *Note:* You don't manage this manually, but knowing it helps performance. If you return a pointer to a local variable, Go "escapes" it to the heap automatically.

**When to use Pointers:**
1.  **Modifying Data:** If a function needs to change a variable passed to it.
2.  **Efficiency:** To avoid copying a massive struct (value mostly read-only).
3.  **Consistency:** If some methods on a struct need a pointer receiver, use pointer receivers for all of them.`,
					CodeExamples: `package main

import "fmt"

func main() {
    count := 10
    
    // Pass by value (copy)
    increment(count)
    fmt.Println(count) // Still 10!

    // Pass by reference (pointer)
    incrementPtr(&count)
    fmt.Println(count) // Now 11!
}

func increment(x int) {
    x++ // Only modifies the copy
}

func incrementPtr(x *int) {
    *x++ // Goes to the address and modifies the value
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
				{
					Title: "Pointer Methods",
					Content: `Methods can have pointer receivers, allowing them to modify the receiver and work efficiently with large structs.

**Pointer Receiver Methods:**
- Syntax: func (p *Type) MethodName()
- Can modify receiver
- More efficient for large structs
- Required when method needs to modify state

**Value vs Pointer Receivers:**
- Value receiver: Works on copy, cannot modify
- Pointer receiver: Works on original, can modify
- Go automatically converts: value.Method() works with pointer receiver

**When to Use Pointer Receivers:**
- Method modifies receiver
- Struct is large (avoid copying)
- Consistency: use same receiver type for all methods
- Need nil checks

**Nil Receivers:**
- Pointer receivers can be nil
- Check for nil before use
- Common pattern: return early if nil`,
					CodeExamples: `package main

import "fmt"

type Counter struct {
    value int
}

// Pointer receiver - can modify
func (c *Counter) Increment() {
    c.value++
}

// Pointer receiver - can be nil
func (c *Counter) GetValue() int {
    if c == nil {
        return 0  // Handle nil receiver
    }
    return c.value
}

// Value receiver - cannot modify
func (c Counter) Display() {
    fmt.Printf("Counter value: %d\n", c.value)
}

func main() {
    c := Counter{value: 0}
    
    // Both work (Go converts automatically)
    c.Increment()      // Converts to (&c).Increment()
    (&c).Increment()  // Explicit
    
    fmt.Println(c.GetValue())  // 2
    
    // Nil receiver
    var nilCounter *Counter
    fmt.Println(nilCounter.GetValue())  // 0 (handled)
}`,
				},
				{
					Title: "Nil Pointer Safety",
					Content: `Nil pointers are a common source of panics in Go. Understanding how to handle them safely is crucial.

**Nil Pointer Behavior:**
- Dereferencing nil pointer causes panic
- Methods on nil receivers can be called (if handled)
- Always check for nil before dereferencing
- Use defensive programming

**Nil Checks:**
- if ptr == nil { return }
- if ptr != nil { use ptr }
- Common pattern: return early on nil

**Nil Receiver Methods:**
- Methods can handle nil receivers
- Check for nil inside method
- Return sensible default values
- Common in standard library

**Best Practices:**
- Check for nil before dereferencing
- Return early on nil
- Handle nil in methods
- Use nil to represent "not found" or "empty"
- Document nil behavior

**Common Patterns:**
- Return nil for "not found"
- Check nil before use
- Provide nil-safe methods
- Use nil as sentinel value`,
					CodeExamples: `package main

import "fmt"

type Node struct {
    Value int
    Next  *Node
}

// Safe nil handling
func (n *Node) GetValue() int {
    if n == nil {
        return 0  // Default value
    }
    return n.Value
}

// Nil check before use
func traverse(head *Node) {
    current := head
    for current != nil {
        fmt.Println(current.Value)
        current = current.Next
    }
}

// Return nil for "not found"
func findNode(head *Node, value int) *Node {
    current := head
    for current != nil {
        if current.Value == value {
            return current
        }
        current = current.Next
    }
    return nil  // Not found
}

func main() {
    var head *Node  // nil
    
    // Safe to call method (handles nil)
    fmt.Println(head.GetValue())  // 0
    
    // Check before dereferencing
    if head != nil {
        fmt.Println(head.Value)  // Won't execute
    }
    
    // Find node
    node := findNode(head, 5)
    if node != nil {
        fmt.Println("Found:", node.Value)
    } else {
        fmt.Println("Not found")
    }
}`,
				},
				{
					Title: "Pointer Best Practices",
					Content: `Following best practices for pointers helps write safe, efficient, and maintainable Go code.

**When to Use Pointers:**

**1. Modify Function Arguments:**
- Pass pointer when function needs to modify value
- More efficient than returning new value
- Clear intent

**2. Large Structs:**
- Avoid copying large structs
- Use pointer to pass efficiently
- Pointer is 8 bytes (64-bit), struct might be much larger

**3. Optional Values:**
- Use nil to represent "not set"
- Common pattern: *int for optional integer
- Check for nil before use

**4. Sharing State:**
- Multiple functions need same data
- Pass pointer to share
- Be careful with concurrent access

**When NOT to Use Pointers:**

**1. Small Values:**
- int, string, bool: pass by value
- Copying is cheap
- Simpler code

**2. Immutable Data:**
- If value shouldn't change
- Pass by value prevents modification
- Type safety

**3. Slices and Maps:**
- Already reference types
- Don't need pointer (usually)
- Exception: when modifying slice itself (append)

**Best Practices:**
- Use pointers for large structs
- Use pointers when modification needed
- Always check for nil
- Document nil behavior
- Be consistent in API design`,
					CodeExamples: `package main

import "fmt"

type User struct {
    ID    int
    Name  string
    Email string
}

// Good: Pointer for large struct
func updateUser(u *User, name string) {
    u.Name = name  // Modifies original
}

// Good: Pointer for optional
func setOptionalID(id *int) {
    if id != nil {
        *id = 42
    }
}

// Good: Value for small
func add(a, b int) int {  // No pointer needed
    return a + b
}

// Bad: Unnecessary pointer
func addBad(a, b *int) int {  // Overcomplicated
    return *a + *b
}

func main() {
    user := User{ID: 1, Name: "Alice"}
    updateUser(&user, "Bob")
    fmt.Println(user.Name)  // Bob
    
    var optionalID *int
    setOptionalID(optionalID)  // Safe, does nothing
    
    id := 0
    setOptionalID(&id)
    fmt.Println(id)  // 42
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
					Title: "Slice Internals: Length vs Capacity",
					Content: `Slices are not arrays. They are a "view" into an underlying array. Understanding this distinction is key to avoiding bugs and performance issues.

**Anatomy of a Slice:**
A slice is a tiny struct with three fields:
1.  **Pointer:** Points to the start of the data in the underlying array.
2.  **Length (len):** How many elements are in the slice *right now*.
3.  **Capacity (cap):** How many elements *can* be in the slice before it needs to grow (reallocate).

**Visualizing Append:**
When you 'append()' to a slice:
1.  If 'len < cap', it just places the new item in the array and increases 'len'. Fast!
2.  If 'len == cap', the array is full. Go creates a *new*, bigger array (usually 2x size), copies everything over, updates the pointer, and then adds the item. Slower!

**Gotcha sharing:**
Slicing 'b := a[1:3]' does NOT copy data. 'b' points to the same array as 'a'. Changing 'b[0]' changes 'a[1]'. Use 'copy()' if you need independent data.`,
					CodeExamples: `package main

import "fmt"

func main() {
    // initialize: len=0, cap=5
    numbers := make([]int, 0, 5) 
    
    for i := 0; i < 8; i++ {
        numbers = append(numbers, i)
        // Watch capacity jump: 5 -> 10 when we hit 6th element
        fmt.Printf("Len: %d, Cap: %d\n", len(numbers), cap(numbers))
    }
    
    // Slice Trap
    original := []int{1, 2, 3}
    view := original[:]
    view[0] = 999
    fmt.Println(original[0]) // Prints 999!
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
				{
					Title: "Embedding",
					Content: `Struct embedding allows one struct to include another struct's fields and methods, promoting composition over inheritance.

**Embedding Syntax:**
- Embed struct without field name
- Fields and methods are promoted
- Can access embedded fields directly
- Can override methods

**Promotion:**
- Embedded struct's fields become accessible
- Embedded struct's methods become callable
- No need to qualify with embedded struct name
- Can still access via embedded struct name

**Method Promotion:**
- Methods of embedded struct are promoted
- Can call directly on outer struct
- Can override by defining method on outer struct
- Outer struct's method takes precedence

**Use Cases:**
- Composition over inheritance
- Code reuse
- Extending functionality
- Mixin pattern

**Best Practices:**
- Use embedding for "is-a" relationships
- Prefer composition over inheritance
- Be explicit about what you're embedding
- Document embedded types`,
					CodeExamples: `package main

import "fmt"

type Engine struct {
    Power int
}

func (e Engine) Start() {
    fmt.Println("Engine started")
}

type Car struct {
    Engine  // Embedded (no field name)
    Brand   string
}

// Method promotion
func (c Car) Drive() {
    c.Start()  // Can call embedded method
    fmt.Printf("Driving %s car\n", c.Brand)
}

// Override method
func (c Car) Start() {
    fmt.Println("Car engine starting...")
    c.Engine.Start()  // Call embedded method explicitly
}

func main() {
    car := Car{
        Engine: Engine{Power: 200},
        Brand:  "Toyota",
    }
    
    // Promoted fields
    fmt.Println(car.Power)  // Direct access
    fmt.Println(car.Engine.Power)  // Explicit access
    
    // Promoted methods
    car.Start()  // Uses Car's Start (overridden)
    car.Drive()
}`,
				},
				{
					Title: "Method Promotion",
					Content: `When a struct embeds another struct, the embedded struct's methods are promoted and can be called directly on the outer struct.

**Promotion Rules:**
- Embedded struct's methods become available
- Can call directly: outer.Method()
- Can still call explicitly: outer.Embedded.Method()
- Outer struct can override methods

**Method Resolution:**
- Outer struct's methods take precedence
- If not found, check embedded structs
- Ambiguity causes compile error
- Must be explicit if ambiguous

**Use Cases:**
- Extending functionality
- Code reuse
- Interface satisfaction
- Mixin pattern

**Best Practices:**
- Use promotion for convenience
- Override when behavior needs to change
- Be explicit when ambiguous
- Document promoted methods`,
					CodeExamples: `package main

import "fmt"

type Reader struct{}

func (r Reader) Read() {
    fmt.Println("Reading...")
}

type Writer struct{}

func (w Writer) Write() {
    fmt.Println("Writing...")
}

// Multiple embedding
type ReadWriter struct {
    Reader  // Promotes Read()
    Writer  // Promotes Write()
}

// Override Read
func (rw ReadWriter) Read() {
    fmt.Println("ReadWriter reading...")
    rw.Reader.Read()  // Call embedded method
}

func main() {
    rw := ReadWriter{}
    
    // Promoted methods
    rw.Read()   // Uses ReadWriter's Read (overridden)
    rw.Write()  // Uses Writer's Write (promoted)
    
    // Explicit access
    rw.Reader.Read()  // Uses Reader's Read directly
}`,
				},
				{
					Title: "Struct Tags",
					Content: `Struct tags provide metadata for struct fields, used by encoding packages and validation libraries.

**Tag Syntax:**
- Format: ` + "`" + `key:"value" key2:"value2"` + "`" + `
- Multiple tags separated by spaces
- Values can have options (comma-separated)
- Accessed via reflection

**Common Tag Types:**

**1. JSON Tags:**
- json:"field_name"
- json:"field_name,omitempty" (omit if zero)
- json:"-" (ignore field)
- json:",omitempty" (use Go field name)

**2. XML Tags:**
- xml:"field_name"
- xml:"field_name,attr" (attribute)
- xml:",chardata" (character data)

**3. Database Tags:**
- db:"column_name"
- gorm:"column:name" (GORM)
- sql:"column_name" (database/sql)

**4. Validation Tags:**
- validate:"required"
- validate:"min=1,max=100"
- validate:"email"

**Tag Options:**
- omitempty: Omit if zero value
- -: Ignore field
- attr: XML attribute
- chardata: XML character data

**Accessing Tags:**
- Use reflect package
- Common in encoding/json, encoding/xml
- Used by ORMs and validators`,
					CodeExamples: `package main

import (
    "encoding/json"
    "encoding/xml"
    "fmt"
    "reflect"
)

type User struct {
    ID       int    ` + "`" + `json:"id" xml:"id,attr" db:"user_id" validate:"required"` + "`" + `
    Username string ` + "`" + `json:"username" xml:"username" db:"username" validate:"required,min=3"` + "`" + `
    Email    string ` + "`" + `json:"email,omitempty" xml:"email" db:"email" validate:"email"` + "`" + `
    Password string ` + "`" + `json:"-" xml:"-" db:"password"` + "`" + `
}

// Access tags via reflection
func printTags(u User) {
    t := reflect.TypeOf(u)
    for i := 0; i < t.NumField(); i++ {
        field := t.Field(i)
        fmt.Printf("Field: %s\n", field.Name)
        fmt.Printf("  JSON: %s\n", field.Tag.Get("json"))
        fmt.Printf("  XML: %s\n", field.Tag.Get("xml"))
        fmt.Printf("  DB: %s\n", field.Tag.Get("db"))
        fmt.Printf("  Validate: %s\n", field.Tag.Get("validate"))
    }
}

func main() {
    u := User{
        ID:       1,
        Username: "alice",
        Email:    "alice@example.com",
        Password: "secret",
    }
    
    // JSON
    jsonData, _ := json.Marshal(u)
    fmt.Println("JSON:", string(jsonData))
    
    // XML
    xmlData, _ := xml.Marshal(u)
    fmt.Println("XML:", string(xmlData))
    
    // Tags
    printTags(u)
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
				{
					Title: "Interface Composition",
					Content: `Interfaces can embed other interfaces, creating larger interfaces through composition.

**Composition Syntax:**
- Embed interface without method name
- All embedded methods become part of interface
- Type must implement all methods
- Promotes code reuse

**Benefits:**
- Code reuse
- Flexible design
- Clear intent
- Standard library pattern

**Common Compositions:**
- io.ReadWriter = io.Reader + io.Writer
- io.ReadWriteCloser = io.Reader + io.Writer + io.Closer
- http.ResponseWriter = io.Writer + http.Flusher + http.Hijacker

**Best Practices:**
- Compose small, focused interfaces
- Follow standard library patterns
- Document composed interfaces
- Use for code reuse`,
					CodeExamples: `package main

import (
    "fmt"
    "io"
)

type Reader interface {
    Read([]byte) (int, error)
}

type Writer interface {
    Write([]byte) (int, error)
}

type Closer interface {
    Close() error
}

// Composed interfaces
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

type File struct {
    data []byte
    pos  int
}

func (f *File) Read(b []byte) (int, error) {
    if f.pos >= len(f.data) {
        return 0, io.EOF
    }
    n := copy(b, f.data[f.pos:])
    f.pos += n
    return n, nil
}

func (f *File) Write(b []byte) (int, error) {
    f.data = append(f.data, b...)
    return len(b), nil
}

func (f *File) Close() error {
    f.data = nil
    return nil
}

func main() {
    var rwc ReadWriteCloser = &File{}
    
    rwc.Write([]byte("Hello"))
    
    buf := make([]byte, 10)
    rwc.Read(buf)
    fmt.Println(string(buf))
    
    rwc.Close()
}`,
				},
				{
					Title: "Empty Interface",
					Content: `The empty interface (interface{}) represents any type, enabling generic-like behavior before Go 1.18.

**Empty Interface:**
- interface{} is alias for any (Go 1.18+)
- Can hold any value
- Zero value is nil
- Used extensively before generics

**Common Uses:**
- Accept any type as parameter
- Store heterogeneous collections
- JSON unmarshaling
- Reflection operations
- Function parameters (fmt.Printf, etc.)

**Working with Empty Interface:**
- Type assertions to extract type
- Type switches for multiple types
- Reflection for inspection
- No type safety (use generics when possible)

**Best Practices:**
- Use sparingly (prefer generics in Go 1.18+)
- Document expected types
- Use type assertions/switches
- Consider generics for new code
- Prefer specific interfaces when possible`,
					CodeExamples: `package main

import "fmt"

// Accept any type
func printValue(v interface{}) {
    // Type switch
    switch val := v.(type) {
    case int:
        fmt.Printf("Integer: %d\n", val)
    case string:
        fmt.Printf("String: %s\n", val)
    case bool:
        fmt.Printf("Boolean: %v\n", val)
    default:
        fmt.Printf("Unknown: %v (type: %T)\n", val, val)
    }
}

// Store different types
func storeMixed() {
    var values []interface{}
    values = append(values, 42)
    values = append(values, "hello")
    values = append(values, true)
    values = append(values, 3.14)
    
    for _, v := range values {
        printValue(v)
    }
}

// Type assertion
func extractString(v interface{}) (string, bool) {
    str, ok := v.(string)
    return str, ok
}

func main() {
    printValue(42)
    printValue("hello")
    printValue(true)
    
    storeMixed()
    
    if str, ok := extractString("test"); ok {
        fmt.Println("Extracted:", str)
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
