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
					Content: `**Strings:**
- Immutable sequences of characters
- Created with single, double, or triple quotes
- Support indexing and slicing
- Many built-in methods

**String Methods:**
- upper(), lower(), capitalize(), title()
- strip(), lstrip(), rstrip()
- split(), join()
- replace(), find(), index()
- startswith(), endswith()
- isdigit(), isalpha(), isalnum()

**String Operators:**
- + for concatenation
- * for repetition
- in for membership testing`,
					CodeExamples: `# String creation
s1 = "Hello"
s2 = 'World'
s3 = """Multi-line
string"""

# Concatenation
greeting = s1 + " " + s2  # "Hello World"

# Repetition
line = "-" * 40

# Methods
text = "  hello world  "
print(text.strip())        # "hello world"
print(text.upper())        # "  HELLO WORLD  "
print(text.replace("world", "Python"))

# Split and join
words = "apple,banana,cherry".split(",")
result = "-".join(words)  # "apple-banana-cherry"

# Membership
if "hello" in text:
    print("Found!")`,
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
- Methods with double underscores
- Define how objects behave with built-in operations
- Called automatically by Python

**Common Special Methods:**
- __init__: Constructor
- __str__: String representation (user-friendly)
- __repr__: String representation (developer-friendly)
- __len__: Length of object
- __eq__: Equality comparison
- __lt__, __gt__: Comparison operators
- __add__, __sub__: Arithmetic operators

**Operator Overloading:**
- Define custom behavior for operators
- Makes objects work with built-in operators`,
					CodeExamples: `class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __len__(self):
        return 2  # Point has 2 coordinates

# Usage
p1 = Point(1, 2)
p2 = Point(3, 4)

print(p1)           # "Point(1, 2)" (uses __str__)
print(p1 == p2)     # False (uses __eq__)
p3 = p1 + p2        # Point(4, 6) (uses __add__)
print(len(p1))      # 2 (uses __len__)`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
