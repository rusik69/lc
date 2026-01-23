package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          48,
			Title:       "Data Structures",
			Description: "Master lists, tuples, sets, dictionaries, and comprehensions.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Lists",
					Content: `**Lists in Python:**

**Characteristics:**
- **Ordered**: Elements maintain insertion order
- **Mutable**: Can modify after creation (add, remove, change)
- **Heterogeneous**: Can contain different types (int, str, list, etc.)
- **Indexed**: Access elements by position (0-based indexing)
- **Dynamic**: Grow and shrink as needed

**Creating Lists:**
- Square brackets: [1, 2, 3]
- List constructor: list(iterable)
- List comprehension: [x for x in range(5)]
- Empty list: []

**Indexing:**
- Positive indices: 0 to len-1 (left to right)
- Negative indices: -1 to -len (right to left, -1 is last)
- IndexError: Raised when index out of range

**Slicing:**
- Syntax: list[start:end:step]
- Returns new list (doesn't modify original)
- start: inclusive, end: exclusive
- step: skip elements (can be negative for reverse)

**Common Operations:**
- **Add**: append(), extend(), insert()
- **Remove**: remove(), pop(), del, clear()
- **Search**: index(), count(), in operator
- **Modify**: Direct assignment, slice assignment
- **Sort**: sort(), sorted() (returns new list)

**List Methods:**
- **Modification**: append(), extend(), insert(), remove(), pop(), clear()
- **Search**: index(), count()
- **Ordering**: sort(), reverse()
- **Copying**: copy() (shallow copy)

**Performance Considerations:**
- **append()**: O(1) amortized - very fast
- **insert()**: O(n) - slow, shifts elements
- **remove()**: O(n) - slow, searches then removes
- **in operator**: O(n) - linear search
- **index()**: O(n) - linear search

**Common Pitfalls:**
- **Shallow vs Deep Copy**: copy() only copies references
- **Modifying while iterating**: Can cause unexpected behavior
- **List as default argument**: Mutable default argument trap`,
					CodeExamples: `# Create lists
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, [1, 2, 3]]  # Can nest lists
empty = []

# Indexing
print(fruits[0])     # "apple" (first)
print(fruits[-1])    # "cherry" (last)
print(fruits[-2])    # "banana" (second to last)

# Slicing
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[1:4])      # [1, 2, 3] (indices 1, 2, 3)
print(numbers[:3])      # [0, 1, 2] (first 3)
print(numbers[3:])      # [3, 4, 5, 6, 7, 8, 9] (from index 3)
print(numbers[::2])     # [0, 2, 4, 6, 8] (every 2nd)
print(numbers[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (reverse)

# Adding elements
fruits.append("orange")           # Add to end: O(1)
fruits.extend(["grape", "kiwi"]) # Add multiple: O(k)
fruits.insert(1, "mango")         # Insert at index: O(n)

# Removing elements
fruits.remove("banana")  # Remove first occurrence: O(n)
last = fruits.pop()      # Remove and return last: O(1)
first = fruits.pop(0)    # Remove and return first: O(n)
del fruits[1]            # Delete by index
fruits.clear()           # Remove all elements

# Searching
if "apple" in fruits:     # Membership test: O(n)
    index = fruits.index("apple")  # Find index: O(n)
count = fruits.count("apple")     # Count occurrences: O(n)

# Modifying elements
fruits[0] = "pineapple"           # Direct assignment
fruits[1:3] = ["grape", "kiwi"]   # Slice assignment

# Sorting
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort()                    # In-place sort
sorted_nums = sorted(numbers)     # Returns new sorted list
numbers.sort(reverse=True)        # Descending order

# Copying (IMPORTANT!)
original = [1, 2, [3, 4]]
shallow = original.copy()         # Shallow copy
deep = [x.copy() if isinstance(x, list) else x for x in original]  # Deep copy

shallow[2].append(5)
print(original)  # [1, 2, [3, 4, 5]] - Modified! (shallow copy shares nested list)

# List comprehension (more Pythonic than loops)
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
nested = [[i*j for j in range(3)] for i in range(3)]

# Common patterns
# Filter
positive = [x for x in numbers if x > 0]

# Transform
doubled = [x * 2 for x in numbers]

# Combine
pairs = [(x, y) for x in [1, 2] for y in [3, 4]]
# [(1, 3), (1, 4), (2, 3), (2, 4)]`,
				},
				{
					Title: "Tuples",
					Content: `**Tuples in Python:**

**Characteristics:**
- **Ordered**: Elements maintain order
- **Immutable**: Cannot modify after creation (no add, remove, change)
- **Hashable**: Can be used as dictionary keys (if all elements are hashable)
- **Faster**: Slightly faster than lists for iteration and access
- **Memory efficient**: Less overhead than lists

**Creating Tuples:**
- Parentheses: (1, 2, 3)
- Comma-separated: 1, 2, 3 (parentheses optional)
- Single element: (42,) or 42, (comma required!)
- Tuple constructor: tuple(iterable)
- Empty tuple: ()

**Key Differences from Lists:**
- **Immutable**: Cannot modify after creation
- **Hashable**: Can be dictionary keys (if elements are hashable)
- **Faster**: Slightly better performance
- **Smaller**: Less memory overhead

**When to Use Tuples:**
- **Fixed data**: Data that shouldn't change
- **Dictionary keys**: Must be hashable (tuples are, lists are not)
- **Function returns**: Return multiple values
- **Records**: Group related data (coordinates, RGB, etc.)
- **Performance**: When immutability is acceptable

**When NOT to Use Tuples:**
- Need to modify elements
- Need list methods (append, extend, etc.)
- Dynamic collections

**Common Operations:**
- **Indexing**: Same as lists (0-based, negative indices)
- **Slicing**: Returns new tuple
- **Concatenation**: + operator creates new tuple
- **Repetition**: * operator
- **Unpacking**: Assign to multiple variables
- **Membership**: in operator

**Performance:**
- **Creation**: Faster than lists
- **Access**: Same speed as lists
- **Memory**: Less overhead than lists`,
					CodeExamples: `# Create tuples
point = (10, 20)
rgb = (255, 0, 0)
single = (42,)  # Note comma - required for single element!
empty = ()

# Without parentheses (tuple packing)
coords = 10, 20  # Same as (10, 20)
x, y = 5, 10     # Tuple packing

# Tuple unpacking
x, y = point
print(x, y)  # 10 20

r, g, b = rgb
print(r, g, b)  # 255 0 0

# Extended unpacking (Python 3)
first, *middle, last = (1, 2, 3, 4, 5)
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5

# Indexing (same as lists)
print(point[0])   # 10
print(point[-1])  # 20

# Slicing (returns new tuple)
print(point[0:1])  # (10,)
print(point[:1])   # (10,)

# Concatenation (creates new tuple)
combined = point + (30,)
print(combined)  # (10, 20, 30)

# Repetition
repeated = (1, 2) * 3
print(repeated)  # (1, 2, 1, 2, 1, 2)

# Membership
if 10 in point:
    print("Found!")

# As dictionary keys (must be hashable)
locations = {
    (0, 0): "origin",
    (1, 1): "diagonal",
    (10, 20): "point"
}

# Function return values
def get_name():
    return "Alice", "Smith"  # Returns tuple

first, last = get_name()

# Multiple return values
def divide(a, b):
    return a // b, a % b

quotient, remainder = divide(10, 3)

# Swapping variables (tuple unpacking)
a, b = 10, 20
a, b = b, a  # Swap without temp variable
print(a, b)  # 20 10

# Nested tuples
matrix = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
print(matrix[0][1])  # 2

# Tuple with mutable elements (be careful!)
mutable_tuple = ([1, 2], [3, 4])
mutable_tuple[0].append(3)  # OK! Modifying list inside tuple
print(mutable_tuple)  # ([1, 2, 3], [3, 4])

# But cannot reassign
# mutable_tuple[0] = [5, 6]  # ERROR! Cannot modify tuple

# Converting between list and tuple
numbers_list = [1, 2, 3]
numbers_tuple = tuple(numbers_list)
back_to_list = list(numbers_tuple)

# Tuple methods (limited compared to lists)
t = (1, 2, 3, 2, 1)
print(t.count(2))    # 2 (count occurrences)
print(t.index(3))    # 2 (find index)`,
				},
				{
					Title: "Sets",
					Content: `**Sets in Python:**

**Characteristics:**
- **Unordered**: No guaranteed order (Python 3.7+ maintains insertion order as implementation detail)
- **Unique elements**: No duplicates allowed
- **Mutable**: Can add/remove elements
- **Hashable elements**: Elements must be immutable (hashable)
- **Fast membership**: O(1) average case for in operator

**Creating Sets:**
- Curly braces: {1, 2, 3}
- Set constructor: set(iterable)
- Set comprehension: {x for x in range(5)}
- Empty set: set() (NOT {} - that's empty dict!)

**Key Advantages:**
- **Fast membership testing**: O(1) vs O(n) for lists
- **Automatic deduplication**: Removes duplicates automatically
- **Set operations**: Union, intersection, difference, etc.
- **Efficient**: Optimized for membership and set operations

**When to Use Sets:**
- **Remove duplicates**: Convert list to set and back
- **Fast membership testing**: Checking if item exists
- **Set operations**: Union, intersection, difference
- **Unique collections**: When duplicates don't make sense

**Set Operations:**
- **Union**: All elements in either set (| or union())
- **Intersection**: Elements in both sets (& or intersection())
- **Difference**: Elements in first but not second (- or difference())
- **Symmetric Difference**: Elements in either but not both (^ or symmetric_difference())
- **Subset/Superset**: Check if one set contains another

**Performance:**
- **add()**: O(1) average
- **remove()**: O(1) average
- **in operator**: O(1) average
- **Union/Intersection**: O(n) where n is size of smaller set

**Limitations:**
- **Unordered**: Cannot index or slice
- **Hashable only**: Cannot contain mutable elements (lists, dicts, sets)
- **No duplicates**: Automatically removes duplicates`,
					CodeExamples: `# Create sets
fruits = {"apple", "banana", "cherry"}
numbers = set([1, 2, 3, 4])
mixed = {1, "hello", 3.14, (1, 2)}  # Can mix types (if hashable)

# Empty set (IMPORTANT: use set(), not {})
empty = set()
# {} creates empty dict, not empty set!

# Adding elements
fruits.add("orange")           # Add single element
fruits.update(["grape", "kiwi"])  # Add multiple elements

# Removing elements
fruits.remove("banana")        # Remove (raises KeyError if missing)
fruits.discard("grape")        # Remove (no error if missing)
popped = fruits.pop()          # Remove and return arbitrary element
fruits.clear()                 # Remove all elements

# Set operations
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union (all elements in either set)
union = set1 | set2                    # {1, 2, 3, 4, 5, 6, 7, 8}
union = set1.union(set2)               # Same as above

# Intersection (elements in both sets)
intersection = set1 & set2             # {4, 5}
intersection = set1.intersection(set2)  # Same

# Difference (elements in set1 but not set2)
difference = set1 - set2                # {1, 2, 3}
difference = set1.difference(set2)      # Same

# Symmetric difference (elements in either but not both)
symmetric = set1 ^ set2                 # {1, 2, 3, 6, 7, 8}
symmetric = set1.symmetric_difference(set2)  # Same

# Membership testing (FAST - O(1))
if "apple" in fruits:
    print("Found!")

# Subset and superset
set_a = {1, 2, 3}
set_b = {1, 2, 3, 4, 5}
print(set_a.issubset(set_b))    # True
print(set_b.issuperset(set_a))  # True
print(set_a <= set_b)           # True (subset)
print(set_b >= set_a)           # True (superset)

# Remove duplicates from list
numbers = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(numbers))     # [1, 2, 3, 4] (order may vary)

# Set comprehension
squares = {x**2 for x in range(10)}
evens = {x for x in range(20) if x % 2 == 0}

# Frozen sets (immutable sets)
frozen = frozenset([1, 2, 3])
# frozen.add(4)  # ERROR! Frozen sets are immutable
# Can be used as dictionary keys
dict_with_frozenset = {frozen: "value"}

# Common use case: Finding common elements
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
common = set(list1) & set(list2)  # {4, 5}`,
				},
				{
					Title: "Dictionaries",
					Content: `**Dictionaries in Python:**

**Characteristics:**
- **Key-value pairs**: Mapping from keys to values
- **Unordered**: No guaranteed order (Python 3.7+ maintains insertion order)
- **Mutable**: Can add, modify, remove key-value pairs
- **Keys must be hashable**: Immutable types only (str, int, tuple, frozenset)
- **Fast lookup**: O(1) average case for access by key
- **Dynamic**: Grow and shrink as needed

**Creating Dictionaries:**
- Curly braces: {"key": "value"}
- Dict constructor: dict(key1=value1, key2=value2)
- From pairs: dict([("key1", "value1"), ("key2", "value2")])
- Dict comprehension: {k: v for k, v in items}
- Empty dict: {} or dict()

**Key Features:**
- **Fast access**: O(1) average case lookup
- **Flexible values**: Values can be any type
- **Hashable keys**: Keys must be immutable
- **No duplicate keys**: Later assignment overwrites previous

**Common Operations:**
- **Access**: dict[key] or dict.get(key, default)
- **Set/Update**: dict[key] = value
- **Delete**: del dict[key] or dict.pop(key)
- **Iterate**: keys(), values(), items()
- **Check membership**: key in dict

**Dictionary Methods:**
- **Access**: get(), setdefault()
- **Modification**: update(), pop(), popitem(), clear()
- **Views**: keys(), values(), items() (return view objects)
- **Copying**: copy() (shallow copy)
- **Creation**: fromkeys() (create from iterable)

**Performance:**
- **Access**: O(1) average, O(n) worst case
- **Insertion**: O(1) average
- **Deletion**: O(1) average
- **Membership**: O(1) average

**Common Use Cases:**
- **Lookup tables**: Fast value lookup by key
- **Counters**: Count occurrences (or use collections.Counter)
- **Caching**: Store computed values
- **Configuration**: Store settings
- **JSON-like data**: Represent structured data`,
					CodeExamples: `# Create dictionaries
person = {"name": "Alice", "age": 30}
scores = dict(math=95, science=87)
empty = {}

# Dict from list of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
mapping = dict(pairs)  # {"a": 1, "b": 2, "c": 3}

# Accessing values
print(person["name"])              # "Alice" (raises KeyError if missing)
print(person.get("age"))           # 30 (returns None if missing)
print(person.get("city", "NYC"))   # "NYC" (default value)

# Safe access pattern
city = person.get("city") or "Unknown"

# Modifying dictionaries
person["age"] = 31                 # Update existing
person["city"] = "NYC"             # Add new key-value pair
person.update({"email": "alice@example.com", "phone": "123-456-7890"})

# Deleting
del person["city"]                 # Delete key
age = person.pop("age")            # Delete and return value
person.pop("nonexistent", "default")  # Return default if key missing
key, value = person.popitem()      # Remove and return last item (Python 3.7+)
person.clear()                     # Remove all items

# Iterating dictionaries
person = {"name": "Alice", "age": 30, "city": "NYC"}

# Iterate keys
for key in person:
    print(key, person[key])

# Iterate keys explicitly
for key in person.keys():
    print(key)

# Iterate values
for value in person.values():
    print(value)

# Iterate key-value pairs (most common)
for key, value in person.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
evens = {k: v for k, v in squares.items() if k % 2 == 0}

# Nested dictionaries
users = {
    "alice": {"age": 30, "city": "NYC"},
    "bob": {"age": 25, "city": "LA"}
}
print(users["alice"]["city"])  # "NYC"

# Dictionary methods
person = {"name": "Alice", "age": 30}

# get() - safe access
name = person.get("name", "Unknown")

# setdefault() - set if not exists, return value
city = person.setdefault("city", "NYC")  # Sets "city": "NYC" if not exists

# update() - merge dictionaries
person.update({"email": "alice@example.com"})

# copy() - shallow copy
person_copy = person.copy()
person_copy["age"] = 31  # Doesn't affect original

# fromkeys() - create from iterable
keys = ["name", "age", "city"]
defaults = dict.fromkeys(keys, None)  # {"name": None, "age": None, "city": None}

# Dictionary views (Python 3)
keys_view = person.keys()      # dict_keys object
values_view = person.values()   # dict_values object
items_view = person.items()     # dict_items object

# Views are dynamic - reflect changes to dict
person["new_key"] = "new_value"
print("new_key" in keys_view)  # True

# Common patterns
# Counting occurrences
text = "hello world"
counts = {}
for char in text:
    counts[char] = counts.get(char, 0) + 1
# Or use collections.Counter

# Grouping
students = [("Alice", "A"), ("Bob", "B"), ("Charlie", "A")]
by_grade = {}
for name, grade in students:
    if grade not in by_grade:
        by_grade[grade] = []
    by_grade[grade].append(name)
# Or use collections.defaultdict`,
				},
				{
					Title: "List Comprehensions",
					Content: `**List Comprehensions - Pythonic Way to Create Lists:**

**What are Comprehensions:**
- **Concise syntax**: Create lists in single line
- **More Pythonic**: Preferred over loops for simple transformations
- **Faster**: Often faster than equivalent loops
- **Readable**: Express intent clearly

**Basic Syntax:**
[expression for item in iterable]
[expression for item in iterable if condition]

**Components:**
1. **Expression**: What to include in list
2. **for clause**: Iterate over iterable
3. **if clause**: Optional filter condition

**Advantages:**
- **Concise**: Less code than equivalent loop
- **Readable**: Clear intent
- **Fast**: Optimized by Python interpreter
- **Functional style**: Declarative rather than imperative

**When to Use:**
- **Simple transformations**: One-line operations
- **Filtering**: Select elements based on condition
- **Creating new lists**: From existing data
- **Readable code**: When comprehension is clearer than loop

**When NOT to Use:**
- **Complex logic**: Use regular loop for readability
- **Side effects**: Comprehensions should be pure (no side effects)
- **Multiple statements**: Use loop if need multiple operations

**Nested Comprehensions:**
- Can nest multiple for loops
- Creates cartesian product
- Can become hard to read - use carefully

**Performance:**
- Generally faster than equivalent loops
- More memory efficient (no intermediate list creation)
- Optimized by Python interpreter`,
					CodeExamples: `# Basic comprehension
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Equivalent loop (more verbose)
squares = []
for x in range(10):
    squares.append(x**2)

# With condition (filtering)
evens = [x for x in range(10) if x % 2 == 0]
# [0, 2, 4, 6, 8]

# Transform elements
names = ["alice", "bob", "charlie"]
capitalized = [name.capitalize() for name in names]
# ["Alice", "Bob", "Charlie"]

# Multiple conditions
numbers = [x for x in range(20) if x % 2 == 0 if x > 10]
# [12, 14, 16, 18] (even numbers greater than 10)

# Conditional expression (ternary in comprehension)
numbers = [x if x % 2 == 0 else x * 2 for x in range(5)]
# [0, 2, 4, 6, 8] (even as-is, odd doubled)

# Nested comprehensions
matrix = [[i*j for j in range(3)] for i in range(3)]
# [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# Flattening nested lists
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [item for sublist in nested for item in sublist]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Cartesian product
colors = ["red", "green"]
sizes = ["S", "M", "L"]
products = [(color, size) for color in colors for size in sizes]
# [('red', 'S'), ('red', 'M'), ('red', 'L'), ('green', 'S'), ...]

# Processing with function
def process(x):
    return x * 2 if x > 5 else x

processed = [process(x) for x in range(10)]

# Filtering and transforming
words = ["hello", "world", "python", "code"]
lengths = [len(word) for word in words if len(word) > 4]
# [5, 5, 6] (lengths of words longer than 4 chars)

# Common patterns
# Extract attributes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person("Alice", 30), Person("Bob", 25)]
names = [p.name for p in people]  # ["Alice", "Bob"]

# Remove duplicates while preserving order
numbers = [1, 2, 2, 3, 3, 3, 4]
unique = []
[unique.append(x) for x in numbers if x not in unique]
# Better: list(dict.fromkeys(numbers))`,
				},
				{
					Title: "Dictionary Comprehensions",
					Content: `**Dictionary Comprehensions - Create Dicts Concisely:**

**What are Dictionary Comprehensions:**
- **Concise syntax**: Create dictionaries in single line
- **Similar to list comprehensions**: But produces key-value pairs
- **Pythonic**: Preferred way to create dictionaries from iterables
- **Flexible**: Can transform keys, values, or both

**Basic Syntax:**
{key_expression: value_expression for item in iterable}
{key: value for item in iterable if condition}

**Components:**
1. **Key expression**: What becomes the key
2. **Value expression**: What becomes the value
3. **for clause**: Iterate over iterable
4. **if clause**: Optional filter condition

**Common Use Cases:**
- **Transform existing dictionaries**: Change keys/values
- **Create from lists**: Convert list to dictionary
- **Filter dictionaries**: Select items based on condition
- **Invert dictionaries**: Swap keys and values
- **Group data**: Create lookup structures

**Advantages:**
- **Concise**: Less code than equivalent loops
- **Readable**: Clear intent
- **Fast**: Optimized by Python
- **Functional**: Declarative style

**When to Use:**
- **Simple transformations**: One-line key/value operations
- **Creating from iterables**: When you have data to convert
- **Filtering dictionaries**: Select items based on condition

**Performance:**
- Generally faster than equivalent loops
- More memory efficient`,
					CodeExamples: `# Basic dictionary comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Equivalent loop
squares = {}
for x in range(5):
    squares[x] = x**2

# From list (create lookup)
names = ["Alice", "Bob", "Charlie"]
name_lengths = {name: len(name) for name in names}
# {"Alice": 5, "Bob": 3, "Charlie": 7}

# With condition (filtering)
evens = {x: x**2 for x in range(10) if x % 2 == 0}
# {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Transform existing dictionary
person = {"name": "Alice", "age": 30, "city": "NYC"}
uppercase_keys = {k.upper(): v for k, v in person.items()}
# {"NAME": "Alice", "AGE": 30, "CITY": "NYC"}

# Transform values
doubled_ages = {k: v * 2 if isinstance(v, int) else v 
                for k, v in person.items()}
# {"name": "Alice", "age": 60, "city": "NYC"}

# Swap keys and values (invert dictionary)
reverse = {v: k for k, v in person.items()}
# {"Alice": "name", 30: "age", "NYC": "city"}
# Warning: Only works if values are unique and hashable!

# Filter dictionary items
scores = {"Alice": 95, "Bob": 87, "Charlie": 92}
high_scores = {name: score for name, score in scores.items() if score >= 90}
# {"Alice": 95, "Charlie": 92}

# Create from two lists (using zip)
keys = ["a", "b", "c"]
values = [1, 2, 3]
mapping = {k: v for k, v in zip(keys, values)}
# {"a": 1, "b": 2, "c": 3}

# Nested dictionary comprehension
matrix = {i: {j: i*j for j in range(3)} for i in range(3)}
# {0: {0: 0, 1: 0, 2: 0}, 1: {0: 0, 1: 1, 2: 2}, ...}

# Group data
students = [("Alice", "A"), ("Bob", "B"), ("Charlie", "A"), ("David", "B")]
by_grade = {}
for name, grade in students:
    by_grade.setdefault(grade, []).append(name)
# Or with defaultdict:
from collections import defaultdict
by_grade = defaultdict(list)
for name, grade in students:
    by_grade[grade].append(name)

# Count occurrences (better to use Counter, but example)
text = "hello"
char_counts = {char: text.count(char) for char in set(text)}
# {'h': 1, 'e': 1, 'l': 2, 'o': 1}`,
				},
				{
					Title: "Choosing the Right Data Structure",
					Content: `**Data Structure Selection Guide:**

Choosing the right data structure is crucial for writing efficient Python code. Each structure has specific use cases and performance characteristics.

**When to Use Each Data Structure:**

**Lists:**
- **Use when**: You need ordered, mutable sequence
- **Best for**: 
  - Storing items in order
  - Frequent appending/removing from end
  - Index-based access
  - Iterating through items
  - Duplicates allowed
- **Avoid when**: 
  - Need fast membership testing (use set)
  - Need unique elements (use set)
  - Need key-value pairs (use dict)
  - Data shouldn't change (use tuple)

**Tuples:**
- **Use when**: You need ordered, immutable sequence
- **Best for**:
  - Fixed data that shouldn't change
  - Dictionary keys (must be hashable)
  - Function return values (multiple values)
  - Records/coordinates (x, y), (r, g, b)
  - Performance-critical code (slightly faster than lists)
- **Avoid when**:
  - Need to modify elements
  - Need list methods (append, extend, etc.)
  - Dynamic collections

**Sets:**
- **Use when**: You need unique elements and fast membership testing
- **Best for**:
  - Removing duplicates
  - Fast membership testing (O(1) vs O(n) for lists)
  - Set operations (union, intersection, difference)
  - Tracking unique items
  - Mathematical set operations
- **Avoid when**:
  - Need ordered elements (Python 3.7+ maintains order, but not guaranteed)
  - Need to index or slice
  - Need duplicates
  - Need mutable elements (sets require hashable elements)

**Dictionaries:**
- **Use when**: You need key-value mappings
- **Best for**:
  - Fast lookup by key (O(1) average)
  - Storing related data together
  - Counting occurrences
  - Caching/memoization
  - Configuration/settings
  - Grouping data
- **Avoid when**:
  - Need ordered sequence (use list)
  - Need duplicate keys (not possible)
  - Simple list of items (use list)
  - Need set operations (use set)

**Performance Comparison:**

**Membership Testing (in operator):**
- **List**: O(n) - linear search through all elements
- **Tuple**: O(n) - linear search through all elements
- **Set**: O(1) - hash table lookup (average case)
- **Dict**: O(1) - hash table lookup (average case)
- **Winner**: Set/Dict for large collections

**Adding Elements:**
- **List.append()**: O(1) amortized - very fast
- **List.insert()**: O(n) - slow, shifts elements
- **Set.add()**: O(1) average - fast
- **Dict[key] = value**: O(1) average - fast
- **Winner**: Set/Dict for adding, List.append() for end insertion

**Removing Elements:**
- **List.remove()**: O(n) - searches then removes
- **List.pop()**: O(1) from end, O(n) from beginning
- **Set.remove()**: O(1) average - fast
- **Dict.pop()**: O(1) average - fast
- **Winner**: Set/Dict for removal

**Iteration:**
- **List**: O(n) - sequential access, very fast
- **Tuple**: O(n) - sequential access, slightly faster than list
- **Set**: O(n) - iteration over all elements
- **Dict**: O(n) - iteration over keys/values/items
- **Winner**: All similar, but tuple slightly faster

**Common Patterns:**

**Pattern 1: Remove Duplicates**
- **List**: O(n²) - check each element against all others
- **Set**: O(n) - convert to set and back: list(set(items))
- **Winner**: Set

**Pattern 2: Count Occurrences**
- **List**: O(n²) - count() called for each unique element
- **Dict**: O(n) - single pass: {item: items.count(item) for item in set(items)}
- **Counter**: O(n) - from collections import Counter; Counter(items)
- **Winner**: Counter or Dict

**Pattern 3: Fast Lookup**
- **List**: O(n) - linear search
- **Set/Dict**: O(1) - hash table lookup
- **Winner**: Set/Dict

**Pattern 4: Ordered Collection**
- **List**: O(1) append, maintains order
- **Dict** (Python 3.7+): O(1) insert, maintains insertion order
- **OrderedDict**: Explicit ordering, O(1) operations
- **Winner**: List for sequences, Dict for key-value with order

**Real-World Examples:**

**Example 1: Tracking Unique Visitors**
- **Use Set**: Fast membership testing, automatic deduplication
- visitors = set()
- visitors.add(user_id)  # O(1)
- if user_id in visitors:  # O(1)

**Example 2: Counting Word Frequencies**
- **Use Dict**: Fast lookup and update
- word_count = {}
- word_count[word] = word_count.get(word, 0) + 1
- Or: from collections import Counter; Counter(words)

**Example 3: Storing User Preferences**
- **Use Dict**: Key-value pairs, fast lookup
- preferences = {"theme": "dark", "language": "en"}
- theme = preferences.get("theme", "light")

**Example 4: Processing Ordered Data**
- **Use List**: Maintains order, supports indexing
- items = [process(item) for item in data]
- first_item = items[0]

**Example 5: Fixed Configuration**
- **Use Tuple**: Immutable, can be dictionary key
- config = ("localhost", 8080, True)
- server_configs = {config: "production"}

**Best Practices:**
- **Profile first**: Don't optimize prematurely
- **Use appropriate structure**: Match structure to use case
- **Consider alternatives**: collections module has specialized structures
- **Document choices**: Explain why you chose a structure
- **Test performance**: Measure if performance is critical
- **Use type hints**: Clarify what structure you're using`,
					CodeExamples: `# Performance comparison: Membership testing
import time

# Large list
large_list = list(range(1000000))
large_set = set(large_list)
large_dict = {i: i for i in range(1000000)}

# Test membership
target = 999999

# List (slow - O(n))
start = time.time()
result = target in large_list
list_time = time.time() - start
print(f"List: {list_time:.6f}s")  # ~0.01s

# Set (fast - O(1))
start = time.time()
result = target in large_set
set_time = time.time() - start
print(f"Set: {set_time:.6f}s")  # ~0.000001s (much faster!)

# Dict (fast - O(1))
start = time.time()
result = target in large_dict
dict_time = time.time() - start
print(f"Dict: {dict_time:.6f}s")  # ~0.000001s

# Pattern 1: Remove duplicates
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

# Using list (slow - O(n²))
unique_list = []
for num in numbers:
    if num not in unique_list:  # O(n) for each element
        unique_list.append(num)

# Using set (fast - O(n))
unique_set = list(set(numbers))  # O(n) - much faster!

# Pattern 2: Count occurrences
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]

# Using list.count() (slow - O(n²))
counts_list = {word: words.count(word) for word in set(words)}

# Using dict (fast - O(n))
counts_dict = {}
for word in words:
    counts_dict[word] = counts_dict.get(word, 0) + 1

# Using Counter (fastest and cleanest - O(n))
from collections import Counter
counts_counter = Counter(words)

# Pattern 3: Fast lookup table
# Use dict for O(1) lookup instead of list with O(n) search

# Slow: List lookup
def find_user_list(users, user_id):
    for user in users:  # O(n)
        if user.id == user_id:
            return user
    return None

# Fast: Dict lookup
user_dict = {user.id: user for user in users}
def find_user_dict(user_id):
    return user_dict.get(user_id)  # O(1)

# Pattern 4: Tracking seen items
# Use set for O(1) membership testing

seen = set()
for item in large_dataset:
    if item not in seen:  # O(1) check
        process(item)
        seen.add(item)  # O(1) add

# Pattern 5: Grouping data
# Use dict with list values

students = [("Alice", "A"), ("Bob", "B"), ("Charlie", "A")]

# Group by grade
by_grade = {}
for name, grade in students:
    if grade not in by_grade:
        by_grade[grade] = []
    by_grade[grade].append(name)

# Or use defaultdict
from collections import defaultdict
by_grade = defaultdict(list)
for name, grade in students:
    by_grade[grade].append(name)

# Real-world example: Cache with expiration
from time import time

class Cache:
    def __init__(self, ttl=3600):
        self.cache = {}  # Dict for O(1) lookup
        self.timestamps = {}  # Track when items were added
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            if time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = value
        self.timestamps[key] = time()

# Real-world example: Event log (ordered)
events = []  # List maintains order
events.append(("2024-01-01", "user_login", "alice"))
events.append(("2024-01-01", "page_view", "home"))
events.append(("2024-01-02", "user_logout", "alice"))

# Process in order
for timestamp, event_type, user in events:
    process_event(timestamp, event_type, user)

# Real-world example: Configuration (immutable)
# Use tuple for fixed configuration that can be dict key
DATABASE_CONFIG = ("localhost", 5432, "mydb", "user", "pass")
configs = {
    DATABASE_CONFIG: "production",
    ("localhost", 3306, "testdb", "user", "pass"): "development"
}

# Real-world example: Unique tags
tags = set()  # Set for unique tags
tags.add("python")
tags.add("programming")
tags.add("python")  # Duplicate ignored
print(tags)  # {"python", "programming"}

# Choosing based on use case
# Need fast membership? → Set
# Need key-value pairs? → Dict
# Need ordered sequence? → List
# Need immutable sequence? → Tuple
# Need to count items? → Counter (from collections)
# Need ordered dict? → OrderedDict (Python < 3.7) or dict (Python 3.7+)
# Need queue? → deque (from collections)
# Need default values? → defaultdict (from collections)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          49,
			Title:       "Strings & Text Processing",
			Description: "Learn string operations, formatting, and regular expressions.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "String Operations",
					Content: `**Strings in Python:**

**Fundamental Characteristics:**
- **Immutable sequences**: Once created, strings cannot be modified in place
- **Unicode support**: Full Unicode character support (Python 3+)
- **Sequence operations**: Support indexing, slicing, and iteration
- **Rich method set**: Extensive built-in methods for manipulation

**String Creation:**
- Single quotes: 'text'
- Double quotes: "text"
- Triple quotes: """multi-line text""" or '''multi-line text'''
- Raw strings: r"path\\to\\file" (backslashes literal)
- f-strings: f"Hello {name}" (formatted strings, Python 3.6+)

**String Immutability:**
- Strings cannot be changed after creation
- Operations return new strings
- Efficient for sharing and memory management
- Use string methods or slicing to create new strings

**String Methods by Category:**

**Case Conversion:**
- **upper()**: Convert to uppercase
- **lower()**: Convert to lowercase
- **capitalize()**: First character uppercase, rest lowercase
- **title()**: Title case (each word capitalized)
- **swapcase()**: Swap case of all characters
- **casefold()**: Aggressive lowercase (for case-insensitive comparison)

**Whitespace Handling:**
- **strip()**: Remove leading and trailing whitespace
- **lstrip()**: Remove leading whitespace only
- **rstrip()**: Remove trailing whitespace only
- **strip(chars)**: Remove specific characters from ends

**Splitting and Joining:**
- **split(sep, maxsplit)**: Split into list by separator
- **rsplit(sep, maxsplit)**: Split from right
- **splitlines(keepends)**: Split by line breaks
- **join(iterable)**: Join iterable with string as separator

**Searching and Finding:**
- **find(sub, start, end)**: Find first occurrence, returns -1 if not found
- **rfind(sub, start, end)**: Find from right
- **index(sub, start, end)**: Find first occurrence, raises ValueError if not found
- **rindex(sub, start, end)**: Find from right
- **count(sub, start, end)**: Count occurrences
- **startswith(prefix, start, end)**: Check if starts with prefix
- **endswith(suffix, start, end)**: Check if ends with suffix

**Replacement:**
- **replace(old, new, count)**: Replace occurrences (count limits replacements)
- **translate(table)**: Map characters using translation table
- **maketrans(x, y, z)**: Create translation table

**Validation:**
- **isdigit()**: Check if all characters are digits
- **isalpha()**: Check if all characters are alphabetic
- **isalnum()**: Check if all characters are alphanumeric
- **isspace()**: Check if all characters are whitespace
- **isupper()**: Check if all characters are uppercase
- **islower()**: Check if all characters are lowercase
- **istitle()**: Check if string is title case

**Padding and Alignment:**
- **center(width, fillchar)**: Center string in width
- **ljust(width, fillchar)**: Left-justify string
- **rjust(width, fillchar)**: Right-justify string
- **zfill(width)**: Pad with zeros on left

**String Operators:**
- **+**: Concatenation (creates new string)
- **\***: Repetition (e.g., "hi" * 3 = "hihihi")
- **in**: Membership testing (substring check)
- **not in**: Non-membership testing
- **[]**: Indexing and slicing

**Performance Considerations:**
- **String concatenation**: Use join() for multiple strings (faster than +)
- **String building**: Use list and join() for many concatenations
- **String formatting**: f-strings are fastest (Python 3.6+)
- **Immutability**: Operations create new strings (consider memory for large strings)

**Common Patterns:**
- **Text cleaning**: strip(), lower(), replace()
- **Parsing**: split(), find(), slicing
- **Validation**: startswith(), endswith(), isdigit()
- **Formatting**: f-strings, format(), % formatting
- **Search**: find(), index(), in operator`,
					CodeExamples: `# String creation - multiple ways
single = 'Single quotes'
double = "Double quotes"
triple = """Multi-line
string with
multiple lines"""
raw = r"C:\\Users\\Name"  # Raw string - backslashes literal
f_string = f"Hello {name}"  # Formatted string

# Case conversion
text = "hello world"
print(text.upper())        # "HELLO WORLD"
print(text.lower())        # "hello world"
print(text.capitalize())   # "Hello world"
print(text.title())        # "Hello World"
print("HeLLo".swapcase())  # "hEllO"

# Whitespace handling
text = "  hello world  "
print(text.strip())        # "hello world"
print(text.lstrip())       # "hello world  "
print(text.rstrip())       # "  hello world"

# Remove specific characters
text = "!!!hello!!!"
print(text.strip("!"))     # "hello"

# Splitting
text = "apple,banana,cherry"
words = text.split(",")    # ["apple", "banana", "cherry"]
words = text.split(",", 1) # ["apple", "banana,cherry"] (max 1 split)

# Joining (efficient for multiple strings)
words = ["apple", "banana", "cherry"]
result = "-".join(words)   # "apple-banana-cherry"
result = "".join(words)    # "applebananacherry"

# Searching
text = "hello world"
print(text.find("world"))      # 6 (index of first occurrence)
print(text.find("python"))     # -1 (not found)
print(text.index("world"))      # 6
# print(text.index("python")) # ValueError: substring not found

print(text.rfind("l"))         # 9 (find from right)
print(text.count("l"))         # 3 (count occurrences)

# Checking prefixes and suffixes
filename = "document.pdf"
print(filename.endswith(".pdf"))    # True
print(filename.startswith("doc"))   # True

# Replacement
text = "hello world world"
print(text.replace("world", "Python"))           # "hello Python Python"
print(text.replace("world", "Python", 1))        # "hello Python world" (replace once)

# Validation
print("123".isdigit())        # True
print("abc".isalpha())        # True
print("abc123".isalnum())     # True
print("   ".isspace())        # True
print("HELLO".isupper())      # True

# Padding and alignment
text = "hello"
print(text.center(11, "-"))   # "---hello---"
print(text.ljust(10, "*"))    # "hello*****"
print(text.rjust(10, "*"))    # "*****hello"
print("42".zfill(5))          # "00042"

# String operators
greeting = "Hello" + " " + "World"  # Concatenation
line = "-" * 40                      # Repetition
if "world" in "hello world":         # Membership
    print("Found!")

# Indexing and slicing (same as lists)
text = "Python"
print(text[0])       # "P"
print(text[-1])      # "n" (last character)
print(text[1:4])     # "yth" (slice)
print(text[::-1])    # "nohtyP" (reverse)

# Performance: Use join() for multiple strings
# SLOW (creates many intermediate strings):
result = ""
for word in words:
    result += word  # Inefficient!

# FAST (creates one string):
result = "".join(words)  # Efficient!

# String building pattern
parts = []
parts.append("Hello")
parts.append("World")
result = " ".join(parts)  # "Hello World"

# Text cleaning example
def clean_text(text):
    """Clean and normalize text."""
    return text.strip().lower().replace("  ", " ")

# Parsing example
def parse_email(email):
    """Extract username and domain from email."""
    if "@" in email:
        username, domain = email.split("@", 1)
        return username, domain
    return None, None

# Validation example
def is_valid_phone(phone):
    """Check if phone number contains only digits and dashes."""
    cleaned = phone.replace("-", "").replace(" ", "")
    return cleaned.isdigit() and len(cleaned) >= 10

# Formatting comparison
name = "Alice"
age = 30

# f-strings (fastest, Python 3.6+)
message = f"{name} is {age} years old"

# .format() method
message = "{} is {} years old".format(name, age)
message = "{name} is {age} years old".format(name=name, age=age)

# % formatting (old style, not recommended)
message = "%s is %d years old" % (name, age)`,
				},
				{
					Title: "String Formatting",
					Content: `**String Formatting Methods:**

1. **f-strings** (Python 3.6+): f"text {variable}"
   - Most modern and recommended
   - Fast and readable
   - Can include expressions

2. **.format()**: "text {}".format(variable)
   - Flexible and powerful
   - Supports positional and named placeholders

3. **% formatting**: "text %s" % variable
   - Old style (like C printf)
   - Still works but not recommended

**Format Specifiers:**
- :d for integers
- :f for floats
- :.2f for 2 decimal places
- :s for strings`,
					CodeExamples: `# f-strings (recommended)
name = "Alice"
age = 30
message = f"{name} is {age} years old"
print(message)  # "Alice is 30 years old"

# Expressions in f-strings
print(f"Next year: {age + 1}")

# Format specifiers
pi = 3.14159
print(f"Pi: {pi:.2f}")  # "Pi: 3.14"

# .format() method
template = "{} is {} years old"
print(template.format(name, age))

# Named placeholders
template = "{name} is {age} years old"
print(template.format(name="Alice", age=30))

# % formatting (old style)
print("%s is %d years old" % (name, age))`,
				},
				{
					Title: "Regular Expressions",
					Content: `**Regular Expressions (re module):**
- Pattern matching for strings
- Powerful text processing tool
- Import: import re

**Common Functions:**
- re.search(): Find first match
- re.findall(): Find all matches
- re.match(): Match at start of string
- re.sub(): Replace matches
- re.split(): Split by pattern

**Pattern Syntax:**
- . matches any character
- \\d matches digit
- \\w matches word character
- \\s matches whitespace
- * zero or more, + one or more, ? zero or one
- [] character class
- ^ start, $ end`,
					CodeExamples: `import re

# Search
text = "Contact: alice@example.com"
match = re.search(r'\\w+@\\w+\\.\\w+', text)
if match:
    print(match.group())  # "alice@example.com"

# Find all
text = "Prices: $10, $20, $30"
prices = re.findall(r'\\$\\d+', text)
print(prices)  # ['$10', '$20', '$30']

# Replace
text = "Hello World"
new_text = re.sub(r'World', 'Python', text)
print(new_text)  # "Hello Python"

# Split
text = "apple,banana;cherry"
items = re.split(r'[,;]', text)
print(items)  # ['apple', 'banana', 'cherry']

# Pattern with groups
text = "Date: 2024-01-15"
match = re.search(r'(\\d{4})-(\\d{2})-(\\d{2})', text)
if match:
    year, month, day = match.groups()`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          50,
			Title:       "Object-Oriented Programming",
			Description: "Learn classes, objects, inheritance, and OOP concepts in Python.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Classes and Objects",
					Content: `**Classes:**
- Blueprint for creating objects
- Defined with class keyword
- Contain attributes (data) and methods (functions)
- Follow PascalCase naming convention

**Objects:**
- Instances of a class
- Created by calling class like a function
- Each object has its own attributes

**Class Definition:**
class ClassName:
    """Docstring"""
    def __init__(self, ...):
        # Constructor
        pass
    
    def method(self, ...):
        # Method
        pass`,
					CodeExamples: `# Define class
class Dog:
    """A simple Dog class"""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says woof!"
    
    def get_info(self):
        return f"{self.name} is {self.age} years old"

# Create objects
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

# Access attributes
print(dog1.name)  # "Buddy"
print(dog1.age)   # 3

# Call methods
print(dog1.bark())  # "Buddy says woof!"
print(dog2.get_info())  # "Max is 5 years old"`,
				},
				{
					Title: "Inheritance",
					Content: `**Inheritance:**
- Create new class based on existing class
- Inherit attributes and methods
- Can override parent methods
- Supports multiple inheritance

**Syntax:**
class ChildClass(ParentClass):
    def __init__(self, ...):
        super().__init__(...)
        # Child-specific initialization

**Method Overriding:**
- Define method with same name in child class
- Child method replaces parent method
- Can call parent method with super()`,
					CodeExamples: `# Parent class
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"
    
    def move(self):
        return f"{self.name} moves"

# Child class
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
    
    def speak(self):  # Override
        return f"{self.name} says woof!"

# Another child
class Cat(Animal):
    def speak(self):  # Override
        return f"{self.name} says meow!"

# Usage
dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())  # "Buddy says woof!"
print(dog.move())   # "Buddy moves" (inherited)`,
				},
				{
					Title: "Special Methods",
					Content: `**Special Methods (Dunder Methods):**

**What are Special Methods:**
- Methods with double underscores (__method__)
- Define how objects behave with built-in operations
- Called automatically by Python interpreter
- Enable operator overloading and custom object behavior
- Make your classes work seamlessly with Python's built-in functions

**Naming Convention:**
- Always surrounded by double underscores
- Called "dunder" methods (double underscore)
- Not meant to be called directly (though you can)
- Python calls them automatically based on context

**Categories of Special Methods:**

**1. Object Creation and Representation:**
- **__init__(self, ...)**: Constructor - called when object is created
- **__new__(cls, ...)**: Actually creates the instance (rarely overridden)
- **__del__(self)**: Destructor - called when object is deleted
- **__str__(self)**: User-friendly string representation (str(), print())
- **__repr__(self)**: Developer-friendly representation (repr(), debugging)
- **__format__(self, format_spec)**: Custom formatting (format() function)
- **__bytes__(self)**: Byte representation (bytes() function)

**2. Comparison Operators:**
- **__eq__(self, other)**: Equality (==)
- **__ne__(self, other)**: Inequality (!=)
- **__lt__(self, other)**: Less than (<)
- **__le__(self, other)**: Less than or equal (<=)
- **__gt__(self, other)**: Greater than (>)
- **__ge__(self, other)**: Greater than or equal (>=)
- **__hash__(self)**: Hash value (for use in sets/dicts as keys)
- **__bool__(self)**: Truth value (bool(), if statements)

**3. Arithmetic Operators:**
- **__add__(self, other)**: Addition (+)
- **__sub__(self, other)**: Subtraction (-)
- **__mul__(self, other)**: Multiplication (*)
- **__truediv__(self, other)**: True division (/)
- **__floordiv__(self, other)**: Floor division (//)
- **__mod__(self, other)**: Modulo (%)
- **__pow__(self, other)**: Power (**)
- **__abs__(self)**: Absolute value (abs())
- **__neg__(self)**: Unary negation (-)
- **__pos__(self)**: Unary plus (+)

**4. In-place Operators:**
- **__iadd__(self, other)**: In-place addition (+=)
- **__isub__(self, other)**: In-place subtraction (-=)
- **__imul__(self, other)**: In-place multiplication (*=)
- **__itruediv__(self, other)**: In-place division (/=)
- **__ifloordiv__(self, other)**: In-place floor division (//=)
- **__imod__(self, other)**: In-place modulo (%=)
- **__ipow__(self, other)**: In-place power (**=)

**5. Container/Sequence Methods:**
- **__len__(self)**: Length (len())
- **__getitem__(self, key)**: Get item (obj[key])
- **__setitem__(self, key, value)**: Set item (obj[key] = value)
- **__delitem__(self, key)**: Delete item (del obj[key])
- **__contains__(self, item)**: Membership test (in operator)
- **__iter__(self)**: Return iterator (iter(), for loops)
- **__next__(self)**: Next item in iterator
- **__reversed__(self)**: Reversed iteration (reversed())

**6. Context Managers:**
- **__enter__(self)**: Enter context (with statement)
- **__exit__(self, exc_type, exc_val, exc_tb)**: Exit context
- Enables use with 'with' statement for resource management

**7. Callable Objects:**
- **__call__(self, ...)**: Make instance callable (obj())
- Allows objects to be called like functions

**8. Attribute Access:**
- **__getattr__(self, name)**: Get attribute (when not found)
- **__setattr__(self, name, value)**: Set attribute
- **__delattr__(self, name)**: Delete attribute
- **__getattribute__(self, name)**: Get attribute (always called)
- **__dir__(self)**: Directory listing (dir())

**Best Practices:**
- **__repr__** should be unambiguous and ideally recreateable
- **__str__** should be user-friendly
- **__eq__** and **__hash__** should be consistent (equal objects have same hash)
- Use **functools.total_ordering** decorator for comparison methods
- Implement **__iter__** and **__next__** for iterable objects
- Use **__getitem__** and **__len__** to make sequence-like objects

**Common Patterns:**
- Implement **__repr__** to return code that recreates object
- Use **super()** in **__init__** for inheritance
- Return **NotImplemented** for unsupported operations
- Raise **TypeError** for invalid operations
- Use **@functools.total_ordering** to reduce comparison boilerplate`,
					CodeExamples: `# Comprehensive Point class with many special methods
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # String representations
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"  # Should be recreable
    
    # Comparison operators
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)
    
    def __hash__(self):
        return hash((self.x, self.y))  # Required if __eq__ is defined
    
    # Arithmetic operators
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, (int, float)):
            return Point(self.x + other, self.y + other)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    def __abs__(self):
        return (self.x**2 + self.y**2)**0.5
    
    # Container-like behavior
    def __len__(self):
        return 2  # Point has 2 coordinates
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Point index out of range")
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __contains__(self, value):
        return value in (self.x, self.y)

# Usage examples
p1 = Point(1, 2)
p2 = Point(3, 4)

# String representation
print(str(p1))      # "Point(1, 2)"
print(repr(p1))     # "Point(1, 2)"

# Comparison
print(p1 == p2)     # False
print(p1 < p2)      # True
print(p1 != p2)     # True (uses __ne__ automatically)

# Arithmetic
p3 = p1 + p2        # Point(4, 6)
p4 = p1 * 2         # Point(2, 4)
distance = abs(p1)  # 2.236...

# Container-like
print(len(p1))      # 2
print(p1[0])        # 1 (x coordinate)
print(p1[1])        # 2 (y coordinate)
for coord in p1:    # Iteration
    print(coord)
print(2 in p1)      # True

# Using in sets/dicts (requires __hash__)
points = {p1, p2}   # Set of points
point_dict = {p1: "first", p2: "second"}

# Vector class with more operators
class Vector:
    def __init__(self, *components):
        self.components = list(components)
    
    def __len__(self):
        return len(self.components)
    
    def __getitem__(self, index):
        return self.components[index]
    
    def __setitem__(self, index, value):
        self.components[index] = value
    
    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("Vectors must have same length")
        return Vector(*[a + b for a, b in zip(self, other)])
    
    def __mul__(self, scalar):
        return Vector(*[x * scalar for x in self])
    
    def __str__(self):
        return f"Vector{tuple(self.components)}"

v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)
v3 = v1 + v2        # Vector(5, 7, 9)
v4 = v1 * 2         # Vector(2, 4, 6)

# Callable object
class Counter:
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())    # 1
print(counter())    # 2
print(counter())    # 3

# Context manager
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False  # Don't suppress exceptions

# Usage with 'with' statement
with FileManager("data.txt", "w") as f:
    f.write("Hello")

# Using functools.total_ordering for comparison methods
from functools import total_ordering

@total_ordering
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __eq__(self, other):
        return self.age == other.age
    
    def __lt__(self, other):
        return self.age < other.age
    # Automatically gets __le__, __gt__, __ge__ from total_ordering

p1 = Person("Alice", 30)
p2 = Person("Bob", 25)
print(p1 > p2)      # True (uses __lt__ automatically)
print(p1 <= p2)     # False (uses __lt__ and __eq__ automatically)`,
				},
				{
					Title: "Property Decorators and Descriptors",
					Content: `**Property Decorators:**

**What are Properties:**
- Properties allow you to use methods like attributes
- Provide controlled access to instance variables
- Enable getters, setters, and deleters
- Maintain backward compatibility while adding validation

**@property Decorator:**
- Converts method to read-only property
- Access like attribute: obj.property (not obj.property())
- Can be overridden with setter and deleter

**@property.setter:**
- Defines setter for property
- Called when property is assigned: obj.property = value
- Enables validation and computed properties

**@property.deleter:**
- Defines deleter for property
- Called when property is deleted: del obj.property
- Useful for cleanup operations

**Benefits of Properties:**
- **Encapsulation**: Control access to internal state
- **Validation**: Validate values before setting
- **Computed Properties**: Calculate values on-the-fly
- **Backward Compatibility**: Change implementation without breaking API
- **Lazy Evaluation**: Compute expensive values only when needed

**When to Use Properties:**
- Need validation when setting values
- Computed values based on other attributes
- Maintaining backward compatibility
- Lazy evaluation of expensive computations
- Read-only attributes that should look like variables

**Descriptors:**

**What are Descriptors:**
- Descriptors are objects that define how attribute access works
- More powerful than properties
- Used by properties internally
- Enable reusable attribute access patterns

**Descriptor Protocol:**
- **__get__(self, instance, owner)**: Get attribute value
- **__set__(self, instance, value)**: Set attribute value
- **__delete__(self, instance)**: Delete attribute
- **__set_name__(self, owner, name)**: Called when descriptor is assigned

**Types of Descriptors:**
- **Data Descriptors**: Implement __set__ or __delete__
- **Non-data Descriptors**: Only implement __get__
- Data descriptors take precedence over instance dictionaries

**Common Use Cases:**
- **Type Validation**: Ensure attributes are correct type
- **Range Validation**: Ensure values are in valid range
- **Cached Properties**: Cache expensive computations
- **Lazy Properties**: Compute only when accessed
- **Reusable Patterns**: Share validation logic across classes

**Best Practices:**
- Use **@property** for simple getters/setters
- Use **descriptors** for reusable validation patterns
- Keep property logic simple
- Document property behavior
- Consider performance implications
- Use **__set_name__** for better error messages`,
					CodeExamples: `# Basic property example
class Circle:
    def __init__(self, radius):
        self._radius = radius  # Private attribute (convention)
    
    @property
    def radius(self):
        """Get the radius."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set the radius with validation."""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def diameter(self):
        """Computed property - diameter is always 2 * radius."""
        return 2 * self._radius
    
    @property
    def area(self):
        """Computed property - area calculated from radius."""
        import math
        return math.pi * self._radius ** 2

# Usage
circle = Circle(5)
print(circle.radius)    # 5 (access like attribute)
print(circle.diameter)  # 10 (computed)
print(circle.area)     # 78.54... (computed)

circle.radius = 10     # Uses setter (validates)
# circle.radius = -5   # Raises ValueError

# Property with deleter
class Person:
    def __init__(self, name):
        self._name = name
        self._name_history = [name]
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        if not value:
            raise ValueError("Name cannot be empty")
        self._name_history.append(value)
        self._name = value
    
    @name.deleter
    def name(self):
        print(f"Deleting name: {self._name}")
        self._name = None

person = Person("Alice")
person.name = "Bob"     # Uses setter
del person.name         # Uses deleter

# Cached property (lazy evaluation)
class DataProcessor:
    def __init__(self, data):
        self.data = data
        self._processed = None
    
    @property
    def processed(self):
        """Process data only once, cache result."""
        if self._processed is None:
            print("Processing data...")
            self._processed = [x * 2 for x in self.data]
        return self._processed

processor = DataProcessor([1, 2, 3])
print(processor.processed)  # Processes and caches
print(processor.processed)   # Uses cached value (no processing)

# Read-only property
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @property
    def fahrenheit(self):
        """Read-only computed property."""
        return self._celsius * 9/5 + 32

temp = Temperature(25)
print(temp.fahrenheit)  # 77.0
# temp.fahrenheit = 80  # AttributeError: can't set attribute

# Descriptor for type validation
class TypedProperty:
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, None)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Expected {self.expected_type.__name__}, got {type(value).__name__}")
        setattr(instance, self.name, value)

class Person:
    name = TypedProperty(str)
    age = TypedProperty(int)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 30)
# person.age = "thirty"  # TypeError: Expected int, got str

# Descriptor for range validation
class BoundedProperty:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name)
    
    def __set__(self, instance, value):
        if not (self.min_value <= value <= self.max_value):
            raise ValueError(f"Value must be between {self.min_value} and {self.max_value}")
        setattr(instance, self.name, value)

class Score:
    percentage = BoundedProperty(0, 100)
    
    def __init__(self, percentage):
        self.percentage = percentage

score = Score(85)
# score.percentage = 150  # ValueError: Value must be between 0 and 100

# Non-data descriptor (read-only)
class ReadOnly:
    def __init__(self, value):
        self.value = value
    
    def __get__(self, instance, owner):
        return self.value

class Config:
    version = ReadOnly("1.0.0")
    api_key = ReadOnly("secret-key")

config = Config()
print(config.version)  # "1.0.0"
# config.version = "2.0.0"  # Works (non-data descriptor can be overridden)

# Data descriptor (takes precedence)
class ReadOnlyData:
    def __init__(self, value):
        self.value = value
    
    def __get__(self, instance, owner):
        return self.value
    
    def __set__(self, instance, value):
        raise AttributeError("Cannot set read-only attribute")

class Config2:
    version = ReadOnlyData("1.0.0")

config2 = Config2()
print(config2.version)  # "1.0.0"
# config2.version = "2.0.0"  # AttributeError: Cannot set read-only attribute`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
