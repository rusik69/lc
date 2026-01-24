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
					Content: "Pointers are a fundamental concept in Go that allow you to work with the memory addresses of variables. They are essential for sharing data efficiently and controlling memory usage.\n\n" +
						"**1. The Mechanics:**\n" +
						"- **Address Operator (`&`)**: Used to find where a variable is located in memory (e.g., `&x`).\n" +
						"- **Dereference Operator (`*`)**: Used to access the value stored at a memory address (e.g., `*ptr`).\n\n" +
						"**2. The \"Pass by Value\" Rule:**\n" +
						"Go is strictly **pass by value**. When you pass a variable to a function, Go creates a *copy*. If the variable is a pointer, Go copies the *address*, allowing you to modify the original data. This is why pointers are often called \"reference types\" in other languages, though Go is more explicit.\n\n" +
						"**3. Stack vs. Heap:**\n" +
						"Go automatically decides whether a variable should live on the **Stack** (fast, local) or the **Heap** (slower, persistent). If a function returns a pointer to a local variable, Go \"escapes\" that variable to the heap so it remains valid after the function finishes.",
					CodeExamples: `# BASIC REDIRECTION
x := 42
ptr := &x      // ptr stores the address of x
fmt.Println(ptr) // e.g., 0xc0000120b0
fmt.Println(*ptr) // 42 (dereferencing)

# SHARING DATA (Function)
func double(n *int) {
    *n = *n * 2 // Modifies the original value
}

val := 10
double(&val)
fmt.Println(val) // 20`,
				},

				{
					Title: "Pointer Operations",
					Content: "Go simplifies pointers by removing many of the dangerous operations found in languages like C. This makes Go safer while still providing the performance benefits of direct memory access.\n\n" +
						"**1. Allowed Operations:**\n" +
						"- **Comparison**: You can compare two pointers using `==` or `!=`. They are equal if they point to the exact same memory address or if both are `nil`.\n" +
						"- **Nil Check**: Always check if a pointer is `nil` before dereferencing it to avoid a runtime panic.\n\n" +
						"**2. The Big Restriction: No Pointer Arithmetic:**\n" +
						"Unlike C/C++, Go does **not** allow you to add or subtract from a pointer (e.g., `ptr++` is a compile error). You cannot move a pointer to the next memory address manually. This prevents buffer overflows and many common security vulnerabilities.",
					CodeExamples: `# NIL CHECK (Crucial)
var ptr *int
if ptr != nil {
    fmt.Println(*ptr)
} else {
    fmt.Println("Pointer is nil, skipping dereference")
}

# POINTER COMPARISON
x, y := 10, 10
p1 := &x
p2 := &x
p3 := &y
fmt.Println(p1 == p2) // true
fmt.Println(p1 == p3) // false (different addresses)`,
				},

				{
					Title: "Pass by Value vs Reference",
					Content: "One of the most important concepts to master in Go is how data is passed between functions. This affects both the correctness and the performance of your application.\n\n" +
						"**1. Everything is a Value**: When you pass a variable to a function, Go always creates a **copy** of that variable. If you pass an `int`, the function gets a copy of the number. If you pass a `struct`, the function gets a copy of the entire struct.\n\n" +
						"**2. Passing Pointers**: When you pass a pointer, you are still passing a value—but that value is a **memory address**. The function gets a copy of the address, allowing it to modify the original data at that address.\n\n" +
						"**3. Performance Considerations**: For small values like `int` or `bool`, copying is extremely fast. For large `structs`, copying all the data can be slow and memory-intensive; in those cases, passing a pointer is much more efficient.",
					CodeExamples: `# PASS BY VALUE (Copy)
type LargeStruct struct {
    data [1000]int
}

func doWork(s LargeStruct) {
    // 's' is a copy of the original 1000 integers!
}

# PASS BY REFERENCE (Pointer)
func doWorkFaster(s *LargeStruct) {
    // 's' is just a pointer (8 bytes on 64-bit)
}

# THE "CHANGING THE COPY" TRAP
func rename(s string) {
    s = "New Name" // Only changes the local copy
}

val := "Old Name"
rename(val)
fmt.Println(val) // Still "Old Name"`,
				},

				{
					Title: "Nil Pointers",
					Content: "A pointer that does not point to any valid memory address is said to be `nil`. In Go, the zero value for any pointer type is `nil`.\n\n" +
						"**1. The Danger of Panic**: Dereferencing a `nil` pointer (e.g., trying to access `*ptr` when `ptr` is `nil`) will cause your program to crash immediately with a **runtime panic**. Unlike languages like Java which might throw a `NullPointerException`, Go's panics are non-recoverable by default.\n\n" +
						"**2. Safe Handling**: Defensive programming is necessary. You should always check for `nil` before using a pointer that was returned from a function or passed as an optional argument.\n\n" +
						"**3. Representing \"Nothing\"**: `nil` is frequently used to represent the absence of a value, such as a \"not found\" result from a database search or an optional configuration flag.",
					CodeExamples: `# DEFENSIVE CHECK
func printValue(n *int) {
    if n == nil {
        fmt.Println("No value provided")
        return
    }
    fmt.Println("Value is:", *n)
}

# OPTIONAL RETURN
func findUser(id string) *User {
    if userExists(id) {
        return &foundUser
    }
    return nil // Explicitly saying "Nothing found"
}`,
				},

				{
					Title: "Memory Management",
					Content: "Go provides a sophisticated, automatic memory management system that frees developers from the burden of manual memory allocation and deallocation (like `malloc` and `free` in C).\n\n" +
						"**1. Escape Analysis**: When you create a variable, the Go compiler uses \"Escape Analysis\" to decide if it can live on the fast **Stack** or needs to be \"escaped\" to the **Heap**. If a variable is accessible after its parent function returns (e.g., you return its address), it MUST go to the heap.\n\n" +
						"**2. The Garbage Collector (GC)**: Go uses a concurrent, tri-color, mark-and-sweep garbage collector. It works in the background to identify memory that is no longer reachable and reclaim it. Go's GC is optimized for **low latency**, attempting to minimize the \"Stop The World\" pauses where your program's execution is frozen.\n\n" +
						"**3. Performance Pro-Tip**: While GC is automatic, creating too many short-lived objects on the heap can increase GC pressure and slow down your app. Reusing objects (e.g., via `sync.Pool`) can help in high-performance scenarios.",
					CodeExamples: `# STACK vs HEAP DATA
func onStack() {
    x := 42 // 'x' lives on the stack; fast and local
}

func onHeap() *int {
    y := 10 // 'y' would normally be on the stack...
    return &y // ...but because we return its address, it "escapes" to the heap
}

# CONTROLLING ALLOCATION
# Use 'make' for slices/maps/channels to pre-size them and avoid repeated heap allocations.
s := make([]int, 0, 100) // Initial capacity of 100 avoids resizing`,
				},

				{
					Title: "Pointer Methods",
					Content: "In Go, methods can be defined with either a **Value Receiver** or a **Pointer Receiver**. Understanding the choice between them is critical for both functional correctness and system performance.\n\n" +
						"**1. Value Receivers (`func (v T)`)**: The method receives a copy of the struct. Any changes made to the struct's fields during the method only affect the copy, not the original.\n\n" +
						"**2. Pointer Receivers (`func (p *T)`)**: The method receives the memory address of the struct. This allows the method to **modify** the original struct and is significantly more efficient for large data structures, as it avoids copying.\n\n" +
						"**3. Automatic Conversion**: Go's compiler is smart. If you have a value but call a method that requires a pointer receiver, Go will automatically take the address for you (as long as the value is addressable).",
					CodeExamples: `# RECTANGLE EXAMPLE
type Rect struct { W, H int }

# 1. VALUE RECEIVER (Read-only)
func (r Rect) Area() int { return r.W * r.H }

# 2. POINTER RECEIVER (Modifiers)
func (r *Rect) Scale(f int) {
    r.W *= f
    r.H *= f
}

# AUTOMATIC DEREFERENCE
myRect := Rect{10, 5}
myRect.Scale(2) // Go does (&myRect).Scale(2) automatically!`,
				},

				{
					Title: "Nil Pointer Safety",
					Content: "In Go, `nil` is a value that represents the absence of an address. Handling it correctly is the difference between a stable application and one that crashes frequently.\n\n" +
						"**1. The \"Method on Nil\" Feature**: Unlike many languages where calling a method on `null` is an immediate crash, Go actually allows you to call a method on a `nil` pointer. The crash only happens if the method tries to **access** a field of the struct without checking for `nil` first.\n\n" +
						"**2. Defensive Methods**: It is good practice to write methods that handle a `nil` receiver gracefully (e.g., by returning a default value or an error), especially for data structures like Linked Lists or Trees.\n\n" +
						"**3. Built-in Safety**: Go's type system and strict compiler prevent many basic pointer errors, but the developer is still responsible for logical `nil` checks.",
					CodeExamples: `# SAFE METHOD ON NIL
type Node struct { Value int }

func (n *Node) SafeValue() int {
    if n == nil {
        return 0 // Handle the "empty" case gracefully
    }
    return n.Value
}

var myNode *Node // nil
fmt.Println(myNode.SafeValue()) // 0 (No crash!)

# DANGEROUS METHOD
func (n *Node) UnsafeValue() int {
    return n.Value // CRASH if n is nil
}`,
				},

				{
					Title: "Pointer Best Practices",
					Content: "Using pointers correctly is about balancing performance with code simplicity. Here are the three gold rules of pointers in Go.\n\n" +
						"**Rule 1: Use for large structs**. If your struct has many fields or large arrays, pass by pointer to avoid the cost of copying memory on every function call.\n\n" +
						"**Rule 2: Use for consistency**. If some methods on a type require a pointer receiver (to modify state), it's best practice to use pointer receivers for **all** methods on that type, even the read-only ones. This makes the interface consistent.\n\n" +
						"**Rule 3: Avoid for small data**. Types like `int`, `float64`, `bool`, and `string` are small and efficient to copy. Using a pointer for these (e.g., `*int`) adds unnecessary complexity and can actually be slower due to pointer chasing.",
					CodeExamples: `# THE CONSISTENCY RULE
type Player struct { Health int }

# Good: Pointer for both modifier and reader
func (p *Player) TakeDamage(n int) { p.Health -= n }
func (p *Player) IsAlive() bool   { return p.Health > 0 }

# BAD: POINTER FOR SMALL DATA
func increment(n *int) { // Unnecessary pointer!
    *n++
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
					Content: "In Go, an array is a numbered sequence of elements of a single type with a **fixed length**. While you will use slices more often in practice, understanding arrays is crucial because slices are built on top of them.\n\n" +
						"**1. Fixed Size**: Once an array is declared, its size cannot change. The size is actually part of the array's type; for example, `[5]int` and `[10]int` are distinct, incompatible types.\n\n" +
						"**2. Value Type Behavior**: Unlike languages where arrays are references (like Java or JS), Go arrays are **value types**. If you assign an array to a new variable or pass it to a function, Go creates a **complete copy** of the entire array. This makes them expensive to pass if they are large.\n\n" +
						"**3. Zero Initialization**: When you declare an array without a literal, all its elements are automatically initialized to their zero value (e.g., `0` for `int`, `false` for `bool`).",
					CodeExamples: `# ARRAY DECLARATION
var a [5]int           // [0 0 0 0 0]
b := [3]string{"a", "b"} 
c := [...]int{1, 2, 3} // Compiler counts elements

# THE VALUE TYPE TRAP
arr1 := [3]int{1, 1, 1}
arr2 := arr1      // arr2 is a NEW COPY
arr2[0] = 999     // Does NOT affect arr1
fmt.Println(arr1) // [1 1 1]

# MULTI-DIMENSIONAL
var matrix [2][2]int
matrix[0][1] = 5`,
				},

				{
					Title: "Slices",
					Content: "Slices are the most common and powerful way to work with collections of data in Go. They provide a dynamic, flexible interface to an underlying array.\n\n" +
						"**1. Reference Semantics**: Unlike arrays, slices are **reference types**. A slice does not store the elements itself; it stores a pointer to an underlying array, a length, and a capacity. If you copy a slice, both copies point to the **same data**.\n\n" +
						"**2. Creating Slices**: You can create a slice using a literal, by slicing an existing array/slice, or using the `make()` function. `make()` is preferred when you know the required size or want to pre-allocate capacity.\n\n" +
						"**3. The Empty Slice**: A slice that hasn't been initialized is `nil` and has a length and capacity of 0. However, you can still `append` to a `nil` slice safely.",
					CodeExamples: `# SLICE DECLARATION
s := []int{1, 2, 3} // Literal
a := make([]int, 5)  // Len 5, Cap 5
b := make([]int, 0, 5) // Len 0, Cap 5 (Optimized for append)

# SLICING OPERATOR
primes := [6]int{2, 3, 5, 7, 11, 13}
subset := primes[1:4] // [3, 5, 7]

# REFERENCE BEHAVIOR
copy_s := s
copy_s[0] = 999
fmt.Println(s[0]) // Prints 999!`,
				},

				{
					Title: "Slice Internals: Length vs Capacity",
					Content: "Slices are not just dynamic arrays; they are small data structures (headers) that point to a segment of an underlying array. Understanding how they grow is key to writing high-performance Go code.\n\n" +
						"**1. The Three Fields**: A slice header contains a pointer to the data, a **Length** (the number of elements it currently contains), and a **Capacity** (the number of elements it *can* hold before being reallocated).\n\n" +
						"**2. How 'append' Grows**: When you use `append()` and the length exceeds the capacity, Go must allocate a **new, larger array**. It typically doubles the capacity (for small slices) to minimize the number of reallocations, which are expensive because the entire dataset must be copied.\n\n" +
						"**3. Performance Tip**: If you know how many elements your slice will eventually hold, providing an initial capacity to `make()` (e.g., `make([]int, 0, 100)`) can drastically improve performance by avoiding repeated reallocations.",
					CodeExamples: `# VISUALIZING CAPACITY
s := make([]int, 0, 2)
s = append(s, 1)    // Len: 1, Cap: 2
s = append(s, 2)    // Len: 2, Cap: 2
s = append(s, 3)    // Len: 3, Cap: 4 (Doubled!)

# SLICING MEMORY TRAP
hugeArray := make([]int, 1000000)
smallPart := hugeArray[0:10]
// Even though smallPart only has 10 elements,
// it keeps the entire hugeArray alive in memory!
# SOLUTION: Use copy() to free the large array
snippet := make([]int, 10)
copy(snippet, hugeArray[0:10])`,
				},

				{
					Title: "Maps",
					Content: "Maps are Go's built-in associative data type (sometimes called hash tables or dictionaries in other languages). They map unique keys to specific values.\n\n" +
						"**1. Keys and Values**: Map keys can be of any type that is comparable (e.g., `string`, `int`, `bool`, or even pointers). Slices and maps cannot be used as keys because they are not comparable.\n\n" +
						"**2. The Comma Ok Idiom**: When you access a key that doesn't exist, Go returns the zero value for that type. To distinguish between a \"zero value\" and a \"missing key,\" use the two-value assignment: `val, ok := m[key]`.\n\n" +
						"**3. Random Iteration**: Go intentionally randomizes the order of map iteration. You should never rely on maps maintaining a specific order; if you need order, you must maintain a separate slice of keys.",
					CodeExamples: `# MAP INITIALIZATION
m := make(map[string]int)
m["Apple"] = 5
delete(m, "Apple")

# THE "COMMA OK" IDIOM
age, exists := m["Alice"]
if !exists {
    fmt.Println("User not found")
}

# MAP LITERAL
scores := map[string]int{
    "Red":   10,
    "Blue":  20,
}

# ZERO VALUE TRAP
m2 := make(map[string]int)
fmt.Println(m2["Missing"]) // Prints 0! Use 'ok' to be sure.`,
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
					Content: "A struct (short for structure) is a typed collection of fields. They are Go's way of grouping related data together into a single unit, similar to classes in other languages but without the complexity of inheritance.\n\n" +
						"**1. Field Characteristics**: Each field in a struct has a name and a type. Fields starting with an uppercase letter are **exported** (public), while those starting with a lowercase letter are **unexported** (private to the package).\n\n" +
						"**2. Zero Value Initialization**: When a struct is declared without an explicit value, all its fields are automatically initialized to their respective zero values.\n\n" +
						"**3. Anonymous Structs**: You can also define structs without giving them a name. These are useful for one-time data structures, such as when parsing a specific JSON response in a single function.",
					CodeExamples: `# NAMED STRUCT
type User struct {
    ID   int
    Name string
}

# COMPOSITE LITERALS
u1 := User{ID: 1, Name: "Bob"} // Field names (Preferred)
u2 := User{2, "Alice"}          // Positional (Risky if order changes)

# ANONYMOUS STRUCT
config := struct {
    Port int
    Env  string
}{
    Port: 8080,
    Env:  "production",
}`,
				},

				{
					Title: "Methods and Receivers",
					Content: "Go does not have classes, but you can define **Methods** on types. A method is simply a function with a special **receiver** argument that appears between the `func` keyword and the method name.\n\n" +
						"**1. Value Receivers (`(t T)`)**: The method operates on a **copy** of the original value. Any modifications to fields inside the method are lost when the method returns. Use this for small, immutable types.\n\n" +
						"**2. Pointer Receivers (`(t *T)`)**: The method operates on the **original** value in memory. This allows the method to modify the struct's fields and is more efficient for large structs because it avoids copying.\n\n" +
						"**3. Receiver Types**: While most methods are defined on structs, Go allows you to define methods on **any type** defined in the same package (except for basic types like `int` or `string` directly).",
					CodeExamples: `# VALUE vs POINTER RECEIVER
type User struct { Name string }

# 1. VALUE (No change)
func (u User) UpdateNameValue(n string) {
    u.Name = n
}

# 2. POINTER (Works!)
func (u *User) UpdateNamePtr(n string) {
    u.Name = n
}

user := User{"Alice"}
user.UpdateNameValue("Bob") // user.Name is still "Alice"
user.UpdateNamePtr("Bob")   // user.Name is now "Bob"`,
				},

				{
					Title: "Method Sets",
					Content: "The **Method Set** of a type determines the interfaces that the type implements and the methods that can be called on its values and pointers. The rules are strict and crucial for understanding Go's polymorphism.\n\n" +
						"**1. Value Receivers (`T`)**: Methods defined with a value receiver are part of the method set of **both** the value type `T` and the pointer type `*T`.\n\n" +
						"**2. Pointer Receivers (`*T`)**: Methods defined with a pointer receiver are part of the method set **only** of the pointer type `*T`. They are NOT part of the method set of the value type `T`.\n\n" +
						"**3. Interface Satisfaction**: If an interface requires a method that you've defined with a pointer receiver, only a **pointer** to your struct can satisfy that interface; passing a value will result in a compile error.",
					CodeExamples: `# THE SATISFACTION RULE
type Shaper interface { Area() int }
type Square struct { side int }

# Pointer Receiver
func (s *Square) Area() int { return s.side * s.side }

func main() {
    sq := Square{5}
    
    # var s1 Shaper = sq  // ERROR! Square does not implement Shaper (Area has pointer receiver)
    var s2 Shaper = &sq // OK! *Square implements Shaper
}`,
				},

				{
					Title: "Embedded Structs",
					Content: "Go does not support type inheritance in the traditional sense (no `extends` keyword). Instead, it uses **Composition** through a feature called \"Embedded Structs\" or anonymous fields.\n\n" +
						"**1. Field and Method Promotion**: When you embed a struct (by giving it a type name without a field name), all the fields and methods of the embedded struct are \"promoted\" to the outer struct. You can access them as if they were defined directly on the outer struct.\n\n" +
						"**2. Shadowing and Collisions**: If the outer struct has a field or method with the same name as an embedded one, the outer one \"shadows\" the inner one. You can still access the inner one by using the type name of the embedded struct.\n\n" +
						"**3. Composition over Inheritance**: This approach is more flexible than inheritance and avoids many of the deep hierarchy problems found in other languages.",
					CodeExamples: `# EMBEDDING PATTERN
type Engine struct { HP int }
func (e Engine) Start() { fmt.Println("Vroom!") }

type Car struct {
    Engine // Embedded (no field name)
    Brand string
}

func main() {
    c := Car{Brand: "Toyota"}
    c.HP = 200 // Promoted field
    c.Start()  // Promoted method
}

# SHADOWING EXAMPLE
type SuperCar struct {
    Car
}
func (s SuperCar) Start() { fmt.Println("SILENT Vroom!") }
# s.Start() calls SuperCar's method, shadowing Car's.`,
				},

				{
					Title: "Field Tags",
					Content: "Field Tags are strings of metadata attached to struct fields. They are ignored by the Go compiler but can be read at runtime using the `reflect` package. They are most commonly used for encoding (JSON, XML) and database mappings (ORM).\n\n" +
						"**1. Syntax**: Tags are written after the field type, enclosed in backticks (e.g., `` `json:\"name\"` ``). The standard format is `key:\"value\"`.\n\n" +
						"**2. JSON Mapping**: Using the `json` tag, you can control how your struct fields are named when converted to JSON. For example, you can make an uppercase field (exported) map to a lowercase key in JSON.\n\n" +
						"**3. The 'omitempty' Option**: A common JSON tag option is `omitempty`, which tells the encoder to skip the field if it has its zero value (e.g., `0`, `\"\"`, or `nil`).",
					CodeExamples: `# JSON MAPPING
type User struct {
    FirstName string ` + "`" + `json:"first_name"` + "`" + `
    LastName  string ` + "`" + `json:"last_name"` + "`" + `
    Password  string ` + "`" + `json:"-"` + "`" + `          // Always hide in JSON
    Age       int    ` + "`" + `json:"age,omitempty"` + "`" + ` // Hide if 0
}

# ORM / DATABASE TAGS
type Product struct {
    ID    int    ` + "`" + `gorm:"primaryKey"` + "`" + `
    Price float64 ` + "`" + `validate:"min=0"` + "`" + `
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
					Content: "An interface in Go is a type that defines a **set of method signatures**. It specifies what an object can *do*, not what it *is*. In this way, interfaces are the key to building decoupled, modular, and testable code in Go.\n\n" +
						"**1. Behavior Focus**: Unlike structs which define data, interfaces define behavior. Any type that provides the methods listed in an interface is said to \"satisfy\" or \"implement\" that interface.\n\n" +
						"**2. Zero Value**: The zero value of an interface is `nil`. A `nil` interface holds neither a value nor a concrete type.\n\n" +
						"**3. Standard Library Examples**: Many core features of Go rely on simple interfaces, such as `fmt.Stringer` (for printing) or `io.Reader` (for reading data streams).",
					CodeExamples: `# INTERFACE DECLARATION
type Stringer interface {
    String() string
}

# io.Writer (Core Go Interface)
type Writer interface {
    Write(p []byte) (n int, err error)
}

# INTERFACE AS A CONTRACT
func PrintDetails(s Stringer) {
    fmt.Println("Details:", s.String())
}`,
				},

				{
					Title: "Implicit Satisfaction",
					Content: "In many languages (like Java), a class must explicitly state which interfaces it implements using a keyword like `implements`. In Go, this is done **implicitly**. This is a form of structural typing often referred to as \"Duck Typing\": *if it walks like a duck and quacks like a duck, it is a duck.*\n\n" +
						"**1. No Declaration Needed**: If your struct has all the methods required by an interface, it automagically implements that interface. You don't need to import the interface or declare any connection to it.\n\n" +
						"**2. Decoupling**: This allows you to define interfaces for types you don't even own (like types from the standard library or third-party packages) and use them in your functions.\n\n" +
						"**3. Compile-time Safety**: Even though the connection is implicit, Go's compiler still checks it strictly. If a type is missing a method required by an interface, your code will fail to compile.",
					CodeExamples: `# THE DUCK TEST
type Flyer interface { Fly() }

type Bird struct{}
func (b Bird) Fly() { fmt.Println("Flap flap") }

# Bird AUTOMATICALLY implements Flyer.

# THIRD-PARTY INTEGRATION
# You can define an interface in your package that matches methods 
# in a library you use, allowing you to mock that library for testing.
type Saver interface {
    Save(data []byte) error
}`,
				},

				{
					Title: "Empty Interface",
					Content: "The **Empty Interface** is defined as `interface{}`. Because it specifies zero methods, absolutely **every type** in Go satisfies the empty interface. It is the closest thing Go has to a \"top type\" (like `Object` in Java or `any` in TypeScript).\n\n" +
						"**1. The 'any' Alias**: Since Go 1.18, `any` is a built-in alias for `interface{}`. They are exactly identical, and `any` is now the preferred way to write it for cleaner code.\n\n" +
						"**2. When to Use**: Use the empty interface when you need to handle values of unknown types, such as in logging, JSON unmarshaling, or when building generic data containers (though modern Go prefers Generics for this).\n\n" +
						"**3. The Catch**: An `interface{}` value doesn't have any methods of its own. To perform operations on the underlying value (like math on an `int`), you must first convert it back to its concrete type using **Type Assertions**.",
					CodeExamples: `# THE VERSATILE 'any'
var i any

i = 42
i = "hello"
i = true

# PASSING ANY TYPE
func Log(v any) {
    fmt.Printf("Logging: %v\n", v)
}

# SLICE OF ANY
mixed := []any{1, "two", 3.0}`,
				},

				{
					Title: "Type Assertions & Switches",
					Content: "Since an interface variable can hold any type that satisfies it, you often need to \"unpack\" that value to use its concrete fields or methods. Go provides two ways to do this: Type Assertions and Type Switches.\n\n" +
						"**1. Type Assertion**: Used to access the underlying concrete value. Syntax: `v, ok := i.(T)`. If `ok` is true, the assertion succeeded. If you omit `ok` (e.g., `v := i.(T)`) and the type is wrong, your program will **panic**.\n\n" +
						"**2. Type Switch**: A cleaner way to handle multiple possible types for an interface variable. It uses a modified `switch` syntax: `switch v := i.(type)`.\n\n" +
						"**3. Defensive Coding**: Always use the \"comma-ok\" idiom for assertions or use a type switch to avoid unexpected runtime crashes.",
					CodeExamples: `# THE COMMA-OK IDIOM
var i any = "hello"

# Safe Assertion
s, ok := i.(string)
if ok {
    fmt.Println(s) // "hello"
}

# THE TYPE SWITCH
func handle(i any) {
    switch v := i.(type) {
    case int:
        fmt.Printf("Twice %d is %d\n", v, v*2)
    case string:
        fmt.Printf("Length of %q is %d\n", v, len(v))
    default:
        fmt.Printf("Unknown type %T\n", v)
    }
}`,
				},

				{
					Title: "Interface Composition",
					Content: "In Go, large interfaces are often built by combining smaller ones. This is called **Interface Composition** (or embedding). It is a key reason why Go interfaces are typically very small (often just one or two methods).\n\n" +
						"**1. Embedding Syntax**: You can embed an interface in another just by listing its name (no field name needed). The new interface then includes all the methods of the embedded one.\n\n" +
						"**2. Implementation**: To satisfy a composed interface, a type must implement **every method** from all the embedded interfaces.\n\n" +
						"**3. Standard Examples**: The `io` package provides perfect examples: `io.ReadWriter` is composed of `io.Reader` and `io.Writer`, and `io.ReadWriteCloser` adds `io.Closer` to that mix.",
					CodeExamples: `# REUSABLE BLOCKS
type Reader interface { Read(p []byte) (n int, err error) }
type Writer interface { Write(p []byte) (n int, err error) }

# INTERFACE COMPOSITION
type ReadWriter interface {
    Reader
    Writer
}

# MULTIPLE EMBEDDING
type FileHandler interface {
    ReadWriter
    Close() error
}`,
				},

				{
					Title: "Common Interfaces",
					Content: "Go's power lies in a few extremely simple, pervasive interfaces found in the standard library. By implementing these, your types can plug into a vast ecosystem of existing Go code.\n\n" +
						"**1. fmt.Stringer**: The most common interface. If your type has a `String() string` method, `fmt.Printf` and others will use it automatically for printing.\n\n" +
						"**2. error**: Built-in interface for error handling. Any type with an `Error() string` method is an error.\n\n" +
						"**3. io.Reader & io.Writer**: These are the backbone of Go's I/O system. They abstract away the details of where data is coming from or going, whether it's a file, a network socket, or an in-memory buffer.",
					CodeExamples: `# fmt.Stringer
func (u User) String() string {
    return fmt.Sprintf("%s <%s>", u.Name, u.Email)
}

# error (Custom Error)
type MyError struct { Msg string }
func (e MyError) Error() string { return e.Msg }

# io.Writer (Backbone)
func Greet(w io.Writer, name string) {
    fmt.Fprintf(w, "Hello, %s\n", name)
}
# Can pass os.Stdout, a file, or any buffer!`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
