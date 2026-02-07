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
					Content: `**Lists in Python**

**1. What Are Lists and Why Do They Matter?**

Lists are the most commonly used data structure in Python, and for good reason. Think of a list as a flexible container — like a shelf where you can place items in a specific order, rearrange them, add new ones, or remove old ones at any time. In programming, you will constantly need to store collections of related data: a sequence of user names, a series of sensor readings, the lines of a file, or the results of a database query. Lists are the go-to tool for all of these situations.

What makes lists so powerful is their combination of characteristics. They are **ordered**, meaning every element has a definite position and the order you put items in is the order they stay in. They are **mutable**, so unlike strings or tuples, you can change a list after you create it — adding elements, removing them, or modifying them in place. They are **heterogeneous**, which means a single list can hold integers, strings, other lists, or even a mix of completely different types. They support **indexing**, so you can jump directly to any element by its position (starting from 0). And they are **dynamic**, growing and shrinking automatically as you add or remove elements — you never need to declare a size up front.

**2. Creating Lists**

Python gives you several ways to create lists, each suited to different situations. The most common approach is using square brackets: [1, 2, 3] creates a list with three integers. You can also use the list() constructor to convert any iterable (like a string, tuple, or range) into a list — for example, list("hello") produces ["h", "e", "l", "l", "o"]. List comprehensions offer a concise, Pythonic way to build lists from expressions: [x for x in range(5)] generates [0, 1, 2, 3, 4]. And of course, an empty list is simply []. Choosing the right creation method matters: comprehensions are preferred for transformations and filtering, while the constructor is handy when converting between data types.

**3. Indexing — Accessing Elements by Position**

Every element in a list has a numerical index. Positive indices count from the left, starting at 0 — so the first element is at index 0, the second at index 1, and so on up to len(list) - 1. Python also supports negative indices, which count backwards from the end: -1 refers to the last element, -2 to the second-to-last, and so forth. This is incredibly convenient when you want the last item without needing to know the list's length. If you try to access an index that doesn't exist, Python raises an IndexError — a common beginner mistake that's easy to avoid by checking the list length first or using try/except.

**4. Slicing — Extracting Sublists**

Slicing lets you extract a portion of a list using the syntax list[start:end:step]. The start index is inclusive, the end index is exclusive, and the step controls how many elements to skip. For example, numbers[1:4] returns elements at indices 1, 2, and 3. Omitting start means "from the beginning," omitting end means "to the end," and a negative step reverses the direction — numbers[::-1] is the classic Python idiom for reversing a list. An important detail: slicing always returns a new list rather than modifying the original, which makes it safe to use without worrying about side effects.

**5. Common Operations — Adding, Removing, Searching, and Sorting**

Lists come with a rich set of built-in methods. For **adding elements**, append() adds a single item to the end (very fast, O(1) amortized), extend() adds all items from another iterable, and insert() places an item at a specific index (slower, O(n), because it must shift subsequent elements). For **removing elements**, remove() finds and deletes the first occurrence of a value, pop() removes and returns an element by index (O(1) from the end, O(n) from the beginning), del removes by index without returning the value, and clear() empties the entire list. For **searching**, the in operator checks membership (O(n) linear scan), index() finds the position of a value, and count() tallies how many times a value appears. For **sorting**, sort() sorts the list in place while sorted() returns a new sorted list, leaving the original untouched — an important distinction when you need to preserve the original order.

**6. Performance Considerations**

Understanding the time complexity of list operations helps you write efficient code. Appending to the end is O(1) amortized, making lists excellent as stacks (last-in, first-out). However, inserting at the beginning or middle is O(n) because every subsequent element must shift. Similarly, removing by value with remove() is O(n) since Python must search for the element first. The in operator and index() are both O(n) because they perform a linear scan. If you find yourself frequently checking membership on large lists, consider using a set instead, which offers O(1) lookups.

**7. Common Pitfalls to Watch Out For**

Three traps catch even experienced developers. First, **shallow vs. deep copy**: calling copy() or using slicing (list[:]) creates a shallow copy, meaning nested objects (like lists within lists) are still shared references. Modifying a nested list in the copy will also change the original. Use copy.deepcopy() from the copy module when you need a fully independent copy. Second, **modifying a list while iterating** over it can skip elements or cause unexpected behavior — iterate over a copy of the list instead, or build a new list with a comprehension. Third, the **mutable default argument trap**: defining a function with a default argument like def f(items=[]) means all calls share the same list object. Use None as the default and create a new list inside the function instead.`,
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
					Content: `**Tuples in Python**

**1. What Are Tuples and Why Do They Exist?**

If lists are like a whiteboard where you can erase and rewrite at will, tuples are like text carved in stone — once created, they cannot be changed. A tuple is an ordered, immutable sequence of elements. At first glance, this might seem like a limitation: why would you want a data structure you cannot modify? The answer lies in the guarantees that immutability provides. When data should not change — a set of coordinates, an RGB color value, a database record — using a tuple makes your intent crystal clear to anyone reading your code. It says "this data is fixed and should not be altered."

Beyond communicating intent, tuples offer practical advantages. Because Python knows a tuple will never change, it can optimize memory allocation and access speed. Tuples use less memory than lists and are slightly faster to create and iterate over. Most importantly, tuples are **hashable** (as long as all their elements are also hashable), which means they can serve as dictionary keys or be placed into sets — something lists simply cannot do.

**2. Creating Tuples**

There are several ways to create tuples in Python. The most common uses parentheses: (1, 2, 3). However, it is actually the commas that define a tuple, not the parentheses — writing 1, 2, 3 without any parentheses also creates a tuple (this is called "tuple packing"). The parentheses are just there for clarity and to avoid ambiguity in complex expressions. One critical gotcha: to create a single-element tuple, you must include a trailing comma — (42,) or simply 42,. Without the comma, (42) is just the integer 42 wrapped in parentheses, not a tuple at all. You can also use the tuple() constructor to convert any iterable into a tuple, and an empty tuple is simply ().

**3. How Tuples Differ from Lists**

The fundamental difference is immutability. Once a tuple is created, you cannot add, remove, or change its elements. This means tuples lack most of the methods that lists have — there is no append(), extend(), insert(), remove(), or sort(). Tuples only support two methods: count() to tally occurrences of a value, and index() to find the position of a value. This minimalism is a feature, not a bug. It means that when you pass a tuple to a function, you can be confident the function cannot accidentally modify your data. Tuples are also slightly faster than lists for iteration and element access, and they consume less memory because Python can store them more compactly.

**4. When to Use Tuples — and When Not To**

Reach for a tuple when you have **fixed data** that should not change: geographic coordinates (latitude, longitude), color values (r, g, b), or configuration constants. Tuples are the natural choice for **function return values** when you need to return multiple pieces of data — Python's return a, b syntax implicitly creates a tuple. They are essential when you need a **dictionary key** that contains multiple values, since lists are unhashable and cannot serve as keys. They also work well as lightweight **records** to group related data, especially when the meaning of each position is well understood.

Avoid tuples when you need a **dynamic collection** that will grow, shrink, or change over time. If you find yourself wanting to call append() or modify elements, a list is the right choice. The rule of thumb is: if the collection is a fixed snapshot of data, use a tuple; if it is a living, evolving collection, use a list.

**5. Common Operations**

Even though tuples are immutable, they support all the read-only sequence operations you know from lists. **Indexing** works identically — positive indices count from the left starting at 0, and negative indices count backwards from the end. **Slicing** extracts sub-tuples and always returns a new tuple. **Concatenation** with the + operator creates a new tuple combining two existing ones. **Repetition** with the * operator repeats a tuple's contents. **Unpacking** — one of Python's most elegant features — lets you assign a tuple's elements to individual variables in a single statement, like x, y = point. Python 3 extends this with the star operator for extended unpacking: first, *middle, last = my_tuple captures the first and last elements while collecting everything in between into a list. The **in** operator checks for membership, just as with lists.

**6. Performance Characteristics**

Tuples are created faster than lists because Python can allocate them as a single fixed-size block of memory. Element access is the same speed as lists (O(1) for indexing), but iteration is marginally faster. Most significantly, tuples use noticeably less memory than equivalent lists — an important consideration when you are working with millions of records. Python also caches small tuples internally (a process called "interning"), so creating the same small tuple repeatedly may not even allocate new memory. If performance or memory usage is a concern and your data does not need to change, tuples are the clear winner over lists.`,
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
					Content: `**Sets in Python**

**1. What Are Sets and Why Are They Special?**

A set in Python is an unordered collection of unique elements. If you have ever studied sets in mathematics, the concept is identical: a set is simply a group of distinct items with no regard for order or repetition. Think of a set like a bag of colored marbles where you can never have two marbles of the same color — if you try to add a duplicate, the bag simply ignores it.

Sets solve two extremely common programming problems better than any other data structure. First, they provide **blazing-fast membership testing**: checking whether an item exists in a set is O(1) on average (constant time), compared to O(n) for a list where Python must scan through every element. If you have a million items and need to check whether a value exists, a set will give you the answer almost instantly, while a list might take a noticeable amount of time. Second, sets **automatically eliminate duplicates**, making them the simplest way to remove repeated values from any collection.

**2. Creating Sets**

You can create a set using curly braces: {1, 2, 3}. You can also use the set() constructor to convert any iterable into a set — set([1, 2, 2, 3, 3]) produces {1, 2, 3} with duplicates automatically removed. Set comprehensions work just like list comprehensions but with curly braces: {x for x in range(5)} generates {0, 1, 2, 3, 4}. There is one critical gotcha that trips up many beginners: an empty set must be created with set(), not with {}. Using empty curly braces {} creates an empty dictionary, not an empty set. This is a historical quirk of Python's syntax that you simply need to memorize.

**3. Set Operations — The Mathematical Power**

One of the most compelling reasons to use sets is their support for mathematical set operations, which let you compare and combine collections with elegant, readable syntax. The **union** (| operator or union() method) combines all elements from both sets. The **intersection** (& operator or intersection() method) finds elements that appear in both sets. The **difference** (- operator or difference() method) finds elements in the first set that are not in the second. The **symmetric difference** (^ operator or symmetric_difference() method) finds elements that are in either set but not in both. You can also test relationships: issubset() checks if one set is entirely contained within another, and issuperset() checks the reverse. These operations are not just academic — they are incredibly useful in real-world scenarios like finding common users between two platforms, identifying missing items, or computing which permissions a user has versus which they need.

**4. Performance — Why Sets Are So Fast**

Sets are implemented using hash tables under the hood, which is the same technology that powers dictionaries. This means that adding an element with add(), removing with remove() or discard(), and checking membership with the in operator are all O(1) average-case operations. Union and intersection operations run in O(min(n, m)) to O(n + m) time depending on the sizes of the sets involved. This performance makes sets the ideal choice whenever you need fast lookups or need to frequently check whether items exist in a collection. A common pattern is converting a list to a set before performing repeated membership checks — the upfront cost of building the set (O(n)) is quickly repaid by the speed of subsequent O(1) lookups.

**5. Limitations to Keep in Mind**

Sets come with important constraints. Because they are **unordered**, you cannot access elements by index or use slicing — there is no concept of "the third element" in a set. While CPython 3.7+ happens to maintain insertion order as an implementation detail, this is not part of the language specification and should not be relied upon. All elements must be **hashable** (immutable), which means you cannot put lists, dictionaries, or other sets inside a set. If you need a set that contains set-like elements, use frozenset — an immutable variant of set that is itself hashable. Finally, because sets cannot contain duplicates, they are not appropriate when you need to track how many times something appears (use a Counter or dictionary for that). Despite these limitations, sets are an indispensable tool in every Python programmer's toolkit.`,
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
					Content: `**Dictionaries in Python**

**1. What Are Dictionaries and Why Are They Everywhere?**

A dictionary is Python's implementation of a **key-value mapping** — a data structure that associates unique keys with corresponding values, much like a real-world dictionary maps words to their definitions. Imagine a phone book: you look up a person's name (the key) and instantly find their phone number (the value). Dictionaries work the same way, and they are arguably the most important data structure in Python. Internally, Python itself uses dictionaries everywhere — to store object attributes, module namespaces, function keyword arguments, and class definitions. Understanding dictionaries deeply is not optional; it is essential to becoming a proficient Python programmer.

Dictionaries are **mutable**, so you can add, modify, and remove key-value pairs after creation. Keys must be **hashable** (immutable types like strings, integers, tuples, or frozensets), but values can be absolutely anything — numbers, strings, lists, other dictionaries, or even functions. Since Python 3.7, dictionaries officially maintain **insertion order**, meaning items come out in the same order you put them in. And most crucially, dictionaries offer **O(1) average-case lookup** — accessing a value by its key is nearly instantaneous regardless of how many items the dictionary contains.

**2. Creating Dictionaries**

Python provides multiple ways to create dictionaries, each optimized for different situations. The most common is curly brace syntax: {"name": "Alice", "age": 30}. The dict() constructor accepts keyword arguments — dict(name="Alice", age=30) — which can look cleaner when keys are simple strings. You can build a dictionary from a list of key-value pairs: dict([("a", 1), ("b", 2)]). Dictionary comprehensions offer concise creation from iterables: {x: x**2 for x in range(5)}. An empty dictionary is either {} or dict(). The fromkeys() class method creates a dictionary from a sequence of keys, all initialized to the same default value.

**3. Accessing and Modifying Values**

There are two primary ways to access dictionary values, and choosing between them matters. Using square brackets — dict["key"] — returns the value if the key exists, but raises a KeyError if it does not. The get() method — dict.get("key", default) — returns the value if found, or a default value (None if not specified) if the key is missing. In practice, get() is safer and should be your default choice whenever a key might not exist. For setting values, simple assignment dict["key"] = value creates or updates a key-value pair. The update() method merges another dictionary or iterable of pairs into the current one. For deletion, del dict["key"] removes a key (raising KeyError if missing), pop("key") removes and returns the value (with an optional default), and popitem() removes and returns the last inserted pair (useful for processing items one at a time).

**4. Iterating Over Dictionaries**

Dictionaries offer three view objects for iteration. Calling keys() returns a view of all keys, values() returns a view of all values, and items() returns a view of (key, value) tuples. The most common pattern is iterating over items(): for key, value in person.items() gives you both pieces of data in each iteration. Simply writing for key in dict iterates over keys by default. An important detail: these views are **dynamic** — they reflect changes to the dictionary in real time. If you add a key to the dictionary, it immediately appears in the keys view without creating a new view object.

**5. Dictionary Methods — Your Essential Toolkit**

Beyond basic access and modification, dictionaries offer several powerful methods. The get() method provides safe access with defaults. The setdefault() method is a clever two-in-one: it returns the value for a key if it exists, but if the key is missing, it sets it to the provided default and returns that default — perfect for building dictionaries of lists. The update() method merges data from another dictionary or key-value pairs. The copy() method creates a shallow copy. The fromkeys() class method creates a new dictionary from a sequence of keys with a uniform default value.

**6. Performance and Use Cases**

Dictionaries are built on hash tables, giving them O(1) average-case performance for access, insertion, deletion, and membership testing. The worst case is O(n) due to hash collisions, but Python's hash table implementation is highly optimized and worst-case scenarios are extremely rare in practice. This makes dictionaries the ideal choice for **lookup tables** (mapping IDs to objects), **counters** (tallying occurrences), **caches** (storing previously computed results to avoid redundant work), **configuration stores** (holding application settings), and **JSON-like structured data** (nested dictionaries mirror JSON objects perfectly). Whenever you need to associate one piece of data with another and retrieve it quickly, a dictionary is almost certainly the right tool.`,
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
					Content: `**List Comprehensions — The Pythonic Way to Create Lists**

**1. What Are List Comprehensions and Why Should You Care?**

List comprehensions are one of Python's most distinctive and beloved features. They provide a concise, readable, and efficient way to create new lists by transforming or filtering existing iterables — all in a single line of code. Instead of writing a multi-line for loop that initializes an empty list, iterates over data, and appends results one at a time, a list comprehension lets you express the same logic as a compact, declarative statement. Think of it as telling Python "give me a list of X for each Y in Z" rather than giving step-by-step instructions.

For example, where a traditional loop might take four lines — create an empty list, loop over a range, compute a square, and append it — a comprehension does the same thing in one: [x**2 for x in range(10)]. This is not just about saving keystrokes. List comprehensions are considered more "Pythonic" because they express intent clearly and directly. Experienced Python developers expect to see them, and code reviews will often suggest converting simple loops into comprehensions for clarity.

**2. The Syntax — How Comprehensions Are Structured**

The basic syntax is: [expression for item in iterable]. Here, "expression" is the value that will be placed into the resulting list — it can be a simple variable, a calculation, a function call, or any valid Python expression. The "for item in iterable" part works just like a regular for loop, iterating over each element. You can add an optional filter with an if clause: [expression for item in iterable if condition]. Only items where the condition evaluates to True will be included in the result. You can also include a conditional expression (ternary operator) in the expression itself: [x if x > 0 else 0 for x in numbers], which transforms every element but does not filter any out.

**3. Why Comprehensions Are Faster Than Loops**

List comprehensions are not just syntactic sugar — they are actually faster than equivalent for loops in most cases. This is because the Python interpreter optimizes comprehension execution internally. In a regular loop, each call to list.append() involves a method lookup and function call overhead. A comprehension avoids this by building the list directly in optimized C code behind the scenes. For large data sets, this speed difference can be significant. Benchmarks typically show comprehensions running 10-30% faster than equivalent loops, and sometimes even more for simple transformations.

**4. When to Use Comprehensions — and When to Avoid Them**

List comprehensions shine when you need to perform **simple transformations** (like converting all strings to uppercase), **filtering** (like extracting even numbers), or **creating new lists from existing data** (like extracting attributes from objects). The key word is "simple." A comprehension should be easy to read at a glance. If you find yourself writing a comprehension that stretches across multiple lines or requires complex nested logic, that is a strong signal to switch back to a regular for loop. Readability always trumps brevity.

You should also avoid using comprehensions for **side effects** — actions that modify external state rather than producing a value. Writing [print(x) for x in items] technically works, but it creates a useless list of None values and confuses readers who expect comprehensions to produce meaningful results. Use a regular for loop when the goal is to perform actions rather than build a new collection.

**5. Nested Comprehensions — Power with Caution**

You can nest multiple for clauses inside a single comprehension. The clauses are evaluated left to right, just like nested for loops written out longhand. This is useful for flattening nested lists — [item for sublist in nested for item in sublist] — or generating cartesian products — [(x, y) for x in [1, 2] for y in [3, 4]]. However, deeply nested comprehensions quickly become unreadable. As a general rule, if your comprehension has more than two for clauses or combines multiple conditions, break it out into a regular loop or a helper function. The goal of a comprehension is to make code clearer, and an incomprehensible comprehension defeats its own purpose.

**6. Performance and Memory Considerations**

Beyond raw speed, comprehensions are memory-efficient because they avoid the overhead of repeatedly calling append() and resizing the internal array. For very large data sets where you do not need all results in memory at once, consider using a generator expression instead (same syntax but with parentheses instead of brackets). A generator expression produces values lazily, one at a time, which can dramatically reduce memory usage when processing millions of items.`,
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
					Content: `**Dictionary Comprehensions — Creating Dictionaries with Elegance**

**1. What Are Dictionary Comprehensions?**

Dictionary comprehensions are the dictionary counterpart to list comprehensions. Just as a list comprehension lets you build a list in a single expressive line, a dictionary comprehension lets you build a dictionary — a collection of key-value pairs — with the same concise, declarative style. If you have ever found yourself writing a loop that creates an empty dictionary, iterates over some data, and assigns key-value pairs one at a time, a dictionary comprehension can almost certainly replace that pattern with a single, readable statement.

The idea is simple but powerful: instead of describing the steps to build a dictionary, you describe what the dictionary should contain. This declarative approach is considered more Pythonic and is preferred by the Python community for straightforward transformations.

**2. The Syntax — How They Work**

The basic syntax is: {key_expression: value_expression for item in iterable}. The key_expression defines what each key will be, the value_expression defines the corresponding value, and the for clause iterates over the source data. You can add an optional if clause for filtering: {key: value for item in iterable if condition}. Only items where the condition is True will produce entries in the resulting dictionary.

The four components work together like an assembly line. The for clause feeds items from the source. The optional if clause acts as a quality inspector, letting only qualifying items through. The key and value expressions then transform each surviving item into the final key-value pair that ends up in the dictionary.

**3. Common Use Cases — Where Dictionary Comprehensions Shine**

Dictionary comprehensions are incredibly versatile. One of the most common uses is **transforming an existing dictionary** — for example, converting all keys to uppercase: {k.upper(): v for k, v in person.items()}. Another frequent pattern is **creating a lookup dictionary from a list**: {name: len(name) for name in names} builds a mapping from names to their lengths. **Filtering a dictionary** is equally elegant: {k: v for k, v in scores.items() if v >= 90} keeps only entries where the value meets a threshold. **Inverting a dictionary** — swapping keys and values — is a one-liner: {v: k for k, v in original.items()} (though you must be careful that values are unique and hashable). You can even build dictionaries from two parallel lists using zip: {k: v for k, v in zip(keys, values)}.

**4. When to Use Them and When to Reach for Something Else**

Dictionary comprehensions are ideal for **simple, one-step transformations** where the relationship between input and output is clear. If you can describe the dictionary you want in plain English in one sentence — "a dictionary mapping each word to its length" — a comprehension is probably the right tool. However, if the logic for computing keys or values requires multiple steps, conditional branches, or error handling, a regular for loop will be more readable and maintainable. Complex dictionary-building logic stuffed into a comprehension becomes a dense, hard-to-debug puzzle.

For grouping operations — like building a dictionary where each key maps to a list of items — a plain loop with setdefault() or a collections.defaultdict is usually clearer than trying to force the logic into a comprehension. The goal is always readability first, cleverness second.

**5. Performance and Efficiency**

Like list comprehensions, dictionary comprehensions are optimized by the Python interpreter and generally run faster than equivalent loop-based code. The performance gain comes from avoiding repeated method-call overhead (no dict.__setitem__() calls per iteration) and from the interpreter's ability to optimize the comprehension as a single expression. For most practical purposes, the speed difference is modest, but for large-scale data transformations — processing thousands or millions of records — it can add up meaningfully. The real benefit, however, is in code clarity: a well-written comprehension communicates intent immediately, whereas a multi-line loop requires the reader to mentally trace the execution to understand the result.`,
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
					Content: `**Choosing the Right Data Structure**

**1. Why Data Structure Selection Is One of the Most Important Decisions You Make**

Choosing the right data structure is not just a matter of style — it is one of the most impactful decisions in software design. The wrong choice can turn an algorithm that should run in milliseconds into one that takes minutes, or make code that should be straightforward into a tangled mess of workarounds. The right choice, on the other hand, makes your code faster, clearer, and easier to maintain. Think of data structures like tools in a toolbox: a hammer is perfect for nails but terrible for screws. Each Python data structure — list, tuple, set, and dictionary — is optimized for specific patterns of access and modification, and understanding these patterns is what separates proficient programmers from beginners.

**2. Lists — Your Default Ordered Collection**

Reach for a list when you need an **ordered, mutable sequence**. Lists are ideal for storing items that have a meaningful order — a chronological log of events, a ranked leaderboard, or the lines of a file. They excel when you frequently append to or remove from the end (both O(1) operations), need to access elements by numerical index, or need to iterate through all items in order. Lists happily allow duplicate values, which makes them suitable for data where repetition is meaningful (like a list of transactions).

However, lists are a poor choice when you need fast membership testing (checking "is X in my collection?" is O(n) for lists but O(1) for sets), when you need unique elements (use a set), when you need key-value associations (use a dictionary), or when the data should be immutable (use a tuple). If you find yourself calling "if item in my_list" inside a loop that processes thousands of items, switching to a set can improve performance by orders of magnitude.

**3. Tuples — When Immutability Is a Feature, Not a Limitation**

Tuples are the right choice when your data is **fixed and should not change**. They communicate intent — anyone reading your code immediately understands that a tuple's contents are meant to be permanent. Use tuples for geographic coordinates (latitude, longitude), color values (r, g, b), database records, function return values, and dictionary keys (since tuples are hashable and lists are not). In performance-critical inner loops, tuples offer a small but measurable speed advantage over lists because Python can optimize their memory layout.

Avoid tuples when you need to modify the collection after creation. If you catch yourself wanting to call append() or reassign elements, a list is the right tool. The decision between list and tuple often comes down to a single question: "Will this collection ever change?" If yes, use a list. If no, use a tuple.

**4. Sets — When Uniqueness and Speed Matter**

Sets are purpose-built for two scenarios: **ensuring uniqueness** and **fast membership testing**. Converting a list with duplicates to a set and back (list(set(items))) is the standard Python idiom for deduplication. Checking whether an item exists in a set is O(1) on average, making sets invaluable when you need to track which items you have already seen, which users are online, or which permissions are granted. Sets also support mathematical operations — union, intersection, difference, and symmetric difference — which are useful for comparing groups (e.g., "which users are in both Group A and Group B?").

Sets are not appropriate when you need ordered access (you cannot index or slice a set), when you need duplicate elements, or when your elements are mutable (lists and dictionaries cannot be added to sets because they are unhashable). If you need an immutable set that can itself be stored in another set or used as a dictionary key, use frozenset.

**5. Dictionaries — The Swiss Army Knife of Python**

Dictionaries are the right choice whenever you need to **associate keys with values** and retrieve values quickly. They offer O(1) average-case lookup, insertion, and deletion, making them ideal for lookup tables (mapping user IDs to user objects), counters (tallying occurrences of words), caches (storing expensive computation results), configuration stores, and any situation where you need to answer the question "given this key, what is the associated value?" quickly.

Since Python 3.7, dictionaries maintain insertion order, which makes them even more versatile — you get both fast lookup and predictable ordering. Avoid dictionaries when you just need a simple sequence (use a list), when you need set operations like union and intersection (use a set), or when duplicate keys are required (not possible in a dictionary — each key must be unique).

**6. Performance at a Glance**

The critical performance differences boil down to a few key operations. For **membership testing** (the in operator), lists and tuples are O(n) — they must scan every element — while sets and dictionaries are O(1) thanks to hash tables. For **adding elements**, list.append() and set.add() are both O(1), but list.insert() at an arbitrary position is O(n). For **removing elements**, list.remove() is O(n) while set.remove() and dict.pop() are O(1). For **iteration**, all four structures are O(n), with tuples having a slight edge in raw speed.

These differences may seem small, but they compound. If you check membership inside a loop that runs n times against a collection of size n, using a list gives you O(n²) total time while using a set gives you O(n). For n = 1,000,000, that is the difference between one million operations and one trillion operations — the difference between a fraction of a second and potentially hours.

**7. Real-World Decision-Making Patterns**

In practice, choosing a data structure follows a simple decision tree. Ask yourself: Do I need key-value pairs? Use a **dictionary**. Do I need uniqueness or fast membership checks? Use a **set**. Do I need an ordered collection that might change? Use a **list**. Do I need an ordered collection that should never change? Use a **tuple**. Do I need to count things? Use **collections.Counter** (a specialized dictionary). Do I need a queue? Use **collections.deque**. Do I need default values for missing keys? Use **collections.defaultdict**.

The collections module in Python's standard library provides several specialized data structures that extend the basic four. Learning about Counter, defaultdict, OrderedDict, deque, and namedtuple will give you even more precise tools for common patterns. The best practice is to start with the simplest structure that meets your needs, profile your code if performance is a concern, and only switch to a more complex structure when measurements show it is necessary.`,
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
					Content: `**Strings in Python**

**1. What Are Strings and Why Are They Fundamental?**

Strings are one of the most frequently used data types in any programming language, and Python treats them as first-class citizens with an exceptionally rich set of built-in operations. A string is an **immutable sequence of Unicode characters**. The immutability part is crucial: once a string is created, it cannot be changed in place. Every operation that appears to modify a string — replacing characters, converting case, stripping whitespace — actually creates and returns a brand-new string, leaving the original untouched. This design choice makes strings safe to share between different parts of your program without worrying about unexpected mutations, and it allows Python to intern (cache and reuse) common strings for memory efficiency.

Python 3 uses Unicode by default, which means strings can contain characters from virtually any writing system on Earth — Latin, Cyrillic, Chinese, Arabic, emoji, and more — without any special configuration. This is a significant improvement over Python 2, where the distinction between byte strings and Unicode strings was a constant source of bugs.

**2. Creating Strings — Multiple Approaches for Different Needs**

Python offers several ways to define strings, each suited to different situations. Single quotes ('text') and double quotes ("text") are functionally identical — use whichever allows you to avoid escaping internal quotes. Triple quotes (three single or double quotes) create multi-line strings that preserve line breaks, making them ideal for docstrings and long text blocks. Raw strings (prefixed with r, like r"C:\\Users\\Name") treat backslashes as literal characters rather than escape sequences, which is invaluable for file paths and regular expression patterns. And f-strings (prefixed with f, like f"Hello {name}"), introduced in Python 3.6, embed expressions directly inside strings — they are the most readable and fastest formatting option available.

**3. Immutability — The Most Important Thing to Understand**

The fact that strings are immutable has profound practical implications. You cannot do text[0] = "H" to change the first character — this raises a TypeError. Instead, you must create a new string: text = "H" + text[1:]. This means that building a string by repeatedly concatenating with += inside a loop is inefficient: each concatenation creates an entirely new string object, copies all the existing characters, and then discards the old object. For building strings from many pieces, the idiomatic and performant approach is to collect pieces in a list and then call "".join(pieces) at the end. The join() method is optimized internally to calculate the total length needed, allocate memory once, and copy all pieces in a single pass.

**4. String Methods — A Comprehensive Toolkit**

Python strings come with a remarkably complete set of built-in methods, organized by purpose. **Case conversion** methods (upper(), lower(), capitalize(), title(), swapcase(), casefold()) let you normalize text for display or comparison — casefold() is particularly important for case-insensitive comparisons because it handles edge cases in non-English alphabets that lower() misses. **Whitespace handling** methods (strip(), lstrip(), rstrip()) remove leading and trailing whitespace or specified characters — essential for cleaning user input or data read from files.

**Splitting and joining** are among the most commonly used operations. split() breaks a string into a list of substrings based on a delimiter, while join() does the reverse — it assembles a list of strings into a single string with a separator between each element. These two methods together are the backbone of text parsing and generation in Python.

**Searching methods** (find(), rfind(), index(), rindex(), count(), startswith(), endswith()) let you locate substrings, count occurrences, and check prefixes and suffixes. The difference between find() and index() is how they handle missing substrings: find() returns -1, while index() raises a ValueError. Choose based on whether a missing substring is an expected case or an error condition.

**Validation methods** (isdigit(), isalpha(), isalnum(), isspace(), isupper(), islower(), istitle()) return True or False based on the character content of the string. These are invaluable for input validation — checking whether a user entered a valid numeric string, for example, before attempting to convert it to an integer.

**5. String Operators and Sequence Behavior**

Strings support the same sequence operations as lists and tuples. The + operator concatenates two strings, the * operator repeats a string, the in operator checks for substring presence, and square brackets provide indexing and slicing. Indexing follows the same rules as lists: index 0 is the first character, -1 is the last, and slicing with [start:end:step] extracts substrings. The classic Python idiom for reversing a string is text[::-1].

**6. Performance Best Practices**

Three performance guidelines will serve you well. First, when joining many strings, always use "separator".join(list_of_strings) rather than concatenating with += in a loop. Second, for string formatting, prefer f-strings — they are both the most readable and the fastest option. Third, when performing many searches against the same text, consider whether a compiled regular expression or a set-based lookup would be more efficient than repeated calls to find() or the in operator. Strings are incredibly versatile and well-optimized in Python, but understanding their immutable nature is the key to using them efficiently.`,
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
					Content: `**String Formatting — Turning Data into Readable Text**

**1. Why String Formatting Matters**

Almost every program needs to combine variables with text to produce meaningful output — whether it is a log message, a user-facing notification, an API response, or a report. String formatting is the mechanism that lets you embed variable values, calculations, and expressions inside text strings in a clean, readable way. Python has evolved through three generations of formatting approaches, and understanding all three is important because you will encounter each of them in real-world codebases, even though modern Python strongly favors one in particular.

**2. f-strings — The Modern Standard (Python 3.6+)**

f-strings (formatted string literals) are the most powerful, readable, and performant formatting method in Python. You create one by prefixing a string with the letter f and then embedding expressions directly inside curly braces: f"Hello, {name}! You are {age} years old." The expressions inside the braces are evaluated at runtime, so you can include not just variables but any valid Python expression — arithmetic (f"{price * 1.1:.2f}"), method calls (f"{name.upper()}"), conditional expressions (f"{'even' if x % 2 == 0 else 'odd'}"), and even function calls.

What makes f-strings the recommended choice is their combination of readability and speed. The variable names appear right where their values will be inserted, making the string's intent immediately clear. They are also the fastest formatting option because the Python interpreter compiles them into efficient bytecode that avoids the overhead of method calls.

**3. The .format() Method — Flexible and Still Widely Used**

Before f-strings existed, the .format() method was the standard approach. It uses curly braces as placeholders: "Hello, {}! You are {} years old.".format(name, age). Placeholders can be positional ({0}, {1}), named ({name}, {age}), or a mix. Named placeholders are particularly useful when the same value appears multiple times in a template or when the format string is stored separately from the data (such as in a configuration file or translation table). While .format() is slightly more verbose than f-strings, it remains valuable in situations where the template string needs to be defined separately from the data that fills it — something f-strings cannot do because they are evaluated immediately at the point of definition.

**4. %-formatting — The Legacy Approach**

The oldest formatting method uses the % operator with format specifiers borrowed from C's printf function: "%s is %d years old" % (name, age). Here, %s is a placeholder for a string, %d for an integer, %f for a floating-point number, and so on. While this syntax still works and you will see it in older codebases, it is generally not recommended for new code because it is less readable than the alternatives, harder to use with complex formatting, and does not support keyword arguments natively. It is worth knowing because legacy code, logging format strings, and some third-party libraries still use it.

**5. Format Specifiers — Controlling How Values Appear**

All three formatting methods support format specifiers that control the appearance of inserted values. These go after a colon inside the placeholder. The most commonly used specifiers include :d for formatting integers, :f for floating-point numbers, :.2f for exactly two decimal places, :, for thousands separators (e.g., f"{1000000:,}" produces "1,000,000"), :> or :< for right or left alignment within a fixed width, and :0 for zero-padding. For example, f"{pi:.4f}" formats pi to four decimal places, f"{name:>20}" right-aligns a name within a 20-character field, and f"{number:08d}" pads an integer with leading zeros to 8 digits. Mastering format specifiers allows you to produce professionally formatted output — clean tables, aligned columns, and precise numerical displays — without resorting to manual string manipulation.

**6. Choosing the Right Approach**

For new Python 3.6+ code, use f-strings as your default. They are the most readable, the fastest, and the most Pythonic. Use .format() when you need to store a template string and fill it in later, or when you are working with code that must support Python versions older than 3.6. Use %-formatting only when maintaining legacy code that already uses it, or when working with logging module format strings (which conventionally use % style). Regardless of which method you choose, the goal is always the same: produce clear, correctly formatted text that makes your program's output professional and easy to understand.`,
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
					Content: `**Regular Expressions — Powerful Pattern Matching for Text**

**1. What Are Regular Expressions and Why Learn Them?**

Regular expressions (often abbreviated as "regex" or "regexp") are a specialized mini-language for describing patterns in text. Think of them as a supercharged search-and-replace tool. While basic string methods like find() and replace() let you search for exact, fixed substrings, regular expressions let you search for patterns — "any sequence of digits," "a word followed by an @ sign followed by another word," "any line that starts with a date." This makes them indispensable for tasks like validating user input (is this a valid email address?), extracting structured data from unstructured text (pull all phone numbers from a document), cleaning and transforming text (normalize date formats), and parsing log files or configuration files.

Python's re module provides full regular expression support. You import it with "import re" and then use its functions to compile patterns, search for matches, and perform substitutions. While regular expressions have a reputation for being cryptic, learning even the basics unlocks enormous productivity for text processing tasks that would otherwise require dozens of lines of manual string manipulation.

**2. Core Functions — Your Main Tools**

The re module provides several key functions, each designed for a different use case. **re.search(pattern, string)** scans through the entire string looking for the first location where the pattern matches, and returns a Match object (or None if no match is found). This is your go-to function for checking whether a pattern exists anywhere in a string. **re.match(pattern, string)** is similar but only checks for a match at the very beginning of the string — it will not find a pattern that starts in the middle. **re.findall(pattern, string)** returns a list of all non-overlapping matches, which is incredibly useful for extracting all occurrences of a pattern (like all email addresses or all numbers in a document). **re.sub(pattern, replacement, string)** replaces all matches with a replacement string, acting as a pattern-aware version of str.replace(). **re.split(pattern, string)** splits a string at every point where the pattern matches, which is more powerful than str.split() because you can split on complex patterns (like "any combination of commas, semicolons, or whitespace").

**3. Pattern Syntax — The Building Blocks**

Regular expression patterns are built from a combination of literal characters and special metacharacters. A **dot (.)** matches any single character except a newline. **\\d** matches any digit (0-9), **\\w** matches any "word character" (letters, digits, and underscores), and **\\s** matches any whitespace character (spaces, tabs, newlines). The uppercase versions (\\D, \\W, \\S) match the opposite — non-digits, non-word characters, and non-whitespace, respectively.

**Quantifiers** control how many times a pattern element can repeat: **\*** means "zero or more times," **+** means "one or more times," and **?** means "zero or one time" (making something optional). So \\d+ matches one or more digits, and \\w* matches zero or more word characters. **Square brackets []** define a character class — [aeiou] matches any vowel, [0-9] matches any digit, and [^abc] matches any character except a, b, or c. The **caret ^** anchors a pattern to the start of the string, and the **dollar sign $** anchors it to the end.

**4. Groups — Capturing Parts of a Match**

One of the most powerful features of regular expressions is **groups**, created with parentheses (). Groups let you extract specific portions of a match. For example, the pattern (\\d{4})-(\\d{2})-(\\d{2}) matches a date like "2024-01-15" and captures the year, month, and day as separate groups that you can access individually through the Match object's group() and groups() methods. Groups are essential for parsing structured text — extracting the username and domain from an email, pulling the protocol, host, and path from a URL, or capturing the key and value from a configuration line.

**5. Practical Tips and Best Practices**

Always use **raw strings** (prefixed with r) for regex patterns: r"\\d+" instead of "\\\\d+". Raw strings treat backslashes literally, preventing conflicts between Python's string escaping and regex escaping. For patterns you use repeatedly, **compile** them with re.compile() — this pre-processes the pattern once and reuses the compiled version, which is faster when the same pattern is applied to many strings. Start simple and build up — regular expressions can become very complex, so it is better to write a slightly longer but readable pattern than a dense one-liner that nobody (including your future self) can understand. Finally, be aware that regular expressions are not always the right tool. For simple substring searches, str.find() or the in operator are faster and clearer. Regex shines when the pattern you are looking for has variability — "any phone number" rather than "this specific phone number."`,
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
					Content: `**Classes and Objects — The Foundation of Object-Oriented Python**

**1. What Are Classes and Why Do They Exist?**

A class is a **blueprint for creating objects** — a template that defines what data an object will hold (its attributes) and what actions it can perform (its methods). Think of a class like an architectural blueprint for a house: the blueprint itself is not a house, but it describes exactly how to build one. You can use the same blueprint to build many houses, each with its own address, paint color, and occupants. In the same way, a class defines the structure, and each object (instance) created from that class carries its own specific data.

Object-oriented programming (OOP) exists because it mirrors how humans naturally think about the world. We instinctively categorize things into types — dogs, cars, bank accounts, users — and each type has characteristics (attributes) and behaviors (methods). Classes let you organize your code around these natural categories, grouping related data and functionality together into cohesive units. This makes programs easier to understand, maintain, and extend, especially as they grow in complexity.

**2. Defining a Class**

In Python, you define a class using the class keyword, followed by the class name (which by convention uses PascalCase — each word capitalized, no underscores) and a colon. Inside the class body, you define methods — which are simply functions that belong to the class. The most important method is __init__, the **constructor** (or initializer). Python calls __init__ automatically every time you create a new instance of the class. Its job is to set up the initial state of the object by assigning values to instance attributes. The first parameter of every instance method is always self, which refers to the specific object the method is being called on. Through self, methods can access and modify the object's own data.

**3. Creating and Using Objects**

You create an object (an instance of a class) by calling the class name as if it were a function, passing any arguments that __init__ expects (excluding self, which Python provides automatically). For example, if you have a Dog class whose __init__ takes name and age, you would write dog1 = Dog("Buddy", 3). This creates a new Dog object with its own name and age attributes. You can create as many instances as you need, each with different data. Accessing an object's attributes uses dot notation: dog1.name returns "Buddy". Calling a method also uses dot notation: dog1.bark() executes the bark method on that specific dog.

**4. Instance Attributes vs. Class Attributes**

Instance attributes are defined inside __init__ using self (like self.name = name) and belong to a specific object — each instance has its own copy. Class attributes, on the other hand, are defined directly in the class body (outside any method) and are shared by all instances. Class attributes are useful for constants or default values that apply to every instance, while instance attributes hold the unique state of each individual object.

**5. Why This Matters — The Bigger Picture**

Classes are the foundation upon which all of Python's more advanced OOP features are built: inheritance, polymorphism, encapsulation, special methods, decorators, and more. Even if you are not writing your own classes every day, you are constantly using objects created from classes — every string, list, dictionary, file handle, and exception in Python is an object. Understanding how classes work gives you a deeper understanding of Python itself, and it equips you to model complex real-world problems in code that is organized, reusable, and maintainable. The convention of adding a descriptive docstring immediately after the class definition documents the class's purpose and is considered essential for professional code.`,
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
					Content: `**Inheritance — Building on What Already Exists**

**1. What Is Inheritance and Why Is It Powerful?**

Inheritance is one of the core pillars of object-oriented programming, and it addresses a fundamental problem in software development: **code reuse without duplication**. Imagine you have a working Animal class with methods for eating, sleeping, and moving. Now you need Dog, Cat, and Bird classes. Without inheritance, you would have to copy all the shared animal behavior into each new class — tripling the code and tripling the maintenance burden. With inheritance, you simply declare that Dog, Cat, and Bird are types of Animal. They automatically inherit all of the parent's attributes and methods, and you only write additional code for what makes each one unique. A Dog might add a bark() method; a Bird might override the move() method to say "flies" instead of "walks." Everything else comes for free from the parent.

This mirrors how we naturally categorize things in the real world. A golden retriever is a dog, a dog is an animal, and an animal is a living thing. Each level of this hierarchy adds specificity while inheriting the general properties of the levels above it. Inheritance lets you model these hierarchies in code, creating a tree of increasingly specialized classes.

**2. How to Define an Inherited Class**

In Python, you indicate inheritance by placing the parent class name in parentheses after the child class name: class Dog(Animal):. This single declaration tells Python that Dog inherits everything from Animal. Inside the child class, you can define new methods and attributes that are unique to dogs, and you can override (replace) any method inherited from the parent by defining a method with the same name.

The __init__ method of the child class typically needs to call the parent's __init__ to ensure the parent's setup logic runs. This is done using the super() function: super().__init__(name) calls the parent's constructor, passing along any arguments the parent needs. After that call, you can add any child-specific initialization. The super() function is not limited to __init__ — you can use it inside any method to call the parent's version of that method, which is useful when you want to extend rather than completely replace the parent's behavior.

**3. Method Overriding — Specializing Behavior**

When a child class defines a method with the same name as a method in the parent class, the child's version takes precedence. This is called **method overriding**, and it is how you specialize behavior for different types. For example, if Animal.speak() returns "Some sound," Dog.speak() can override it to return "Woof!" and Cat.speak() can return "Meow!" When you call speak() on a Dog object, Python finds the Dog version and uses it. When you call speak() on a plain Animal object, Python uses the Animal version. This ability for different objects to respond differently to the same method call is called **polymorphism** — you can write code that works with any Animal, and each specific type will behave appropriately.

**4. The Power of super() — Extending Rather Than Replacing**

Sometimes you do not want to completely replace a parent method — you want to add to it. For example, a child's __init__ might need to do everything the parent does plus a few extra steps. By calling super().__init__(...) first and then adding your own logic, you extend the parent's behavior rather than discarding it. This pattern keeps your code DRY (Don't Repeat Yourself) and ensures that changes to the parent's __init__ automatically flow down to all children.

**5. Multiple Inheritance and the Method Resolution Order**

Python supports **multiple inheritance**, where a class can inherit from more than one parent: class FlyingFish(Fish, Bird). This is powerful but comes with complexity — what happens if both Fish and Bird define a swim() method? Python resolves this using the **Method Resolution Order (MRO)**, which follows the C3 linearization algorithm. In simple terms, Python searches for methods in the child first, then in each parent left to right, then in the grandparents, and so on, following a predictable and consistent order. You can inspect this order with ClassName.__mro__ or ClassName.mro(). While multiple inheritance is available, many Python programmers prefer composition (having an object contain other objects) over deep multiple inheritance hierarchies, as composition tends to produce simpler, more maintainable code.

**6. Inheritance in the Real World**

Inheritance is used extensively in Python's own standard library and in popular frameworks. Django's class-based views, for example, use inheritance to let you build on pre-built view behaviors. Exception handling relies on an inheritance hierarchy — all exceptions inherit from BaseException, and most user-facing exceptions inherit from Exception. Understanding inheritance means understanding how these frameworks are designed, which makes you a more effective user of them and better equipped to design your own class hierarchies when the need arises.`,
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
					Content: `**Special Methods (Dunder Methods) — Making Your Objects Feel Like Built-in Types**

**1. What Are Special Methods and Why Are They the Secret Sauce of Python?**

Special methods — often called "dunder methods" because their names are surrounded by double underscores (like __init__, __str__, __add__) — are the mechanism that makes Python's object model so elegant and powerful. They define how your objects interact with Python's built-in operations and syntax. When you write len(my_object), Python calls my_object.__len__(). When you write obj_a + obj_b, Python calls obj_a.__add__(obj_b). When you write print(my_object), Python calls my_object.__str__(). Special methods are the hooks that let your custom classes participate seamlessly in Python's syntax — making them feel as natural and intuitive as built-in types like int, str, or list.

You almost never call special methods directly (though you can). Instead, Python calls them for you when you use operators, built-in functions, or language constructs. This is what people mean when they say Python has a "data model" — the special methods are the protocol that your objects can implement to integrate with every part of the language.

**2. Object Creation and Representation**

The most fundamental special methods deal with creating objects and converting them to strings. **__init__** is the constructor you already know — it initializes a new instance with the data you provide. (Technically, __new__ creates the instance first, but you rarely need to override it.) **__str__** defines the "user-friendly" string representation that print() and str() produce — this should be readable and informative for end users. **__repr__** defines the "developer-friendly" representation used in debugging and the interactive interpreter — ideally, it should return a string that could recreate the object, like "Point(3, 4)". A good rule of thumb: always implement __repr__ (it is used as a fallback when __str__ is not defined), and add __str__ when you want a different, more polished output for users.

**3. Comparison Operators — Making Objects Comparable**

By implementing comparison special methods, you can make your objects sortable, comparable, and usable in conditional expressions. **__eq__** defines equality (==), **__lt__** defines less-than (<), **__le__** defines less-than-or-equal (<=), and so on for __gt__ and __ge__. If you define __eq__, you should also define **__hash__** to ensure your objects work correctly in sets and as dictionary keys — the rule is that objects that compare as equal must produce the same hash value. The **__bool__** method controls what happens when your object is used in a boolean context (like an if statement), letting you define when an object should be considered "truthy" or "falsy."

A practical tip: rather than implementing all six comparison methods manually, use the **@functools.total_ordering** decorator. You only need to define __eq__ and one ordering method (like __lt__), and the decorator automatically generates the remaining four.

**4. Arithmetic Operators — Custom Math for Custom Types**

Special methods let you define what +, -, *, /, and other operators mean for your objects. **__add__** handles +, **__sub__** handles -, **__mul__** handles *, **__truediv__** handles /, **__floordiv__** handles //, **__mod__** handles %, and **__pow__** handles **. There are also unary operators: **__neg__** for negation (-obj), **__pos__** for unary plus (+obj), and **__abs__** for abs(obj). This is incredibly powerful for mathematical objects like vectors, matrices, complex numbers, or monetary amounts. A Vector class with __add__ defined lets you write v3 = v1 + v2, which reads as naturally as adding two numbers.

For each arithmetic method, there is also an in-place version (like **__iadd__** for +=) and a reflected version (like **__radd__** that handles cases where the left operand does not support the operation). When an operation is not supported, return the special singleton **NotImplemented** (not raise NotImplementedError) — this tells Python to try the reflected method on the other operand instead.

**5. Container and Sequence Methods — Making Objects Behave Like Collections**

If your class represents a collection of items, you can make it behave like a built-in collection by implementing container special methods. **__len__** lets len() work on your objects. **__getitem__** enables indexing (obj[key]) and, if you use integer indices, also enables iteration and slicing automatically. **__setitem__** enables item assignment (obj[key] = value), and **__delitem__** enables deletion (del obj[key]). **__contains__** customizes the in operator for membership testing. **__iter__** returns an iterator object (enabling for loops), and **__next__** produces the next value from that iterator.

Implementing even a subset of these methods can make your custom class feel remarkably natural. A Deck class with __len__ and __getitem__ can be used with len(deck), deck[0], for card in deck, and random.choice(deck) — all without any additional work.

**6. Context Managers, Callables, and Attribute Access**

Three more categories of special methods deserve attention. **Context manager methods** (__enter__ and __exit__) let your objects work with the with statement, which is Python's elegant pattern for resource management — ensuring files are closed, locks are released, or database transactions are committed, even if an exception occurs. **The __call__ method** makes instances callable like functions — writing obj() invokes obj.__call__(). This is useful for creating function-like objects that maintain state, like counters, decorators, or strategy objects. **Attribute access methods** (__getattr__, __setattr__, __delattr__) give you fine-grained control over what happens when attributes are accessed, set, or deleted, enabling patterns like lazy loading, proxying, and attribute validation.

**7. Best Practices for Special Methods**

Always implement __repr__ — it is invaluable for debugging. Make __str__ user-friendly and __repr__ developer-friendly. Keep __eq__ and __hash__ consistent: objects that compare as equal must hash identically. Use @functools.total_ordering to avoid boilerplate in comparison methods. Return NotImplemented (do not raise an exception) when an operation does not make sense for a given type combination — this gives Python the chance to try the other operand's method. And remember: the goal of special methods is to make your objects feel like natural, built-in parts of the language. When used well, they make your code more readable, more intuitive, and more Pythonic.`,
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
					Content: `**Property Decorators and Descriptors — Controlled Attribute Access**

**1. The Problem Properties Solve**

In many programming languages, the standard practice is to make all attributes private and access them through explicit getter and setter methods: get_name(), set_name(value). This is verbose and makes code cluttered with method calls. Python takes a different philosophical approach: start by using simple public attributes (self.name = name), and if you later need to add validation, computation, or access control, use **properties** to add that behavior without changing the interface. Code that reads obj.name and writes obj.name = "Alice" continues to work exactly the same — it just silently routes through your property methods behind the scenes.

This is one of Python's most elegant design patterns. Properties let you start simple and add complexity only when needed, without breaking any code that already uses your class. Think of properties as invisible guardrails: from the outside, attributes look and feel like plain variables, but internally, custom logic runs whenever they are accessed or modified.

**2. The @property Decorator — Getters Made Invisible**

The @property decorator converts a method into a read-only attribute. When you decorate a method with @property, accessing the attribute calls the method automatically — but without parentheses. So circle.radius looks like accessing a plain attribute, but it actually calls the radius() method under the hood. This is useful for **computed properties** — attributes whose values are derived from other data. A Circle class might store only the radius internally but provide diameter and area as properties that are calculated on-the-fly from the radius. From the user's perspective, circle.diameter looks just like any other attribute, but it always returns the correct value because it is recomputed each time it is accessed.

**3. Setters and Deleters — Full Control Over Attribute Assignment**

To make a property writable, you add a **setter** using the @property_name.setter decorator. The setter method receives the value being assigned and can validate it, transform it, or trigger side effects before storing it. For example, a radius setter could reject negative values by raising a ValueError, or a name setter could strip whitespace and record the change in a history log. You can also define a **deleter** using @property_name.deleter, which runs when someone writes del obj.property — useful for cleanup operations like resetting state or releasing resources.

The combination of getter, setter, and deleter gives you complete control over an attribute's lifecycle while maintaining the clean, simple syntax of plain attribute access. Users of your class never need to know whether they are interacting with a simple attribute or a sophisticated property — the interface is identical.

**4. Real-World Benefits of Properties**

Properties serve several important purposes in practice. **Validation** ensures data integrity — a property setter can enforce that an age is positive, a percentage is between 0 and 100, or an email address contains an @ sign. **Computed properties** derive values from other attributes, keeping your data model consistent — when the radius changes, the diameter and area update automatically. **Lazy evaluation** delays expensive computations until they are actually needed, and can cache the result for subsequent accesses. **Backward compatibility** is perhaps the most strategically important benefit: if you initially exposed a plain attribute and later need to add validation or computation, converting it to a property requires zero changes to existing code that uses your class.

**5. Descriptors — The Engine Behind Properties**

Descriptors are a more advanced and powerful mechanism that properties are actually built on top of. A descriptor is any object that implements at least one of the special methods __get__, __set__, or __delete__. When Python accesses an attribute on an object, it checks whether the attribute is a descriptor, and if so, calls the appropriate descriptor method instead of performing a normal attribute lookup.

The key insight is that descriptors are **reusable**. A property is defined per-attribute within a single class, but a descriptor is a separate class that can be used across many classes. For example, you could write a TypedProperty descriptor that validates the type of a value, and then use it in dozens of classes: name = TypedProperty(str), age = TypedProperty(int), price = TypedProperty(float). Each declaration creates a new descriptor instance with its own validation rules, but all share the same underlying logic. The __set_name__ method (Python 3.6+) makes descriptors even more convenient by automatically telling each descriptor instance what attribute name it was assigned to, enabling better error messages.

**6. Data vs. Non-Data Descriptors**

Descriptors come in two flavors with an important behavioral difference. **Data descriptors** implement __set__ or __delete__ (in addition to __get__) and take precedence over instance dictionaries — if an instance has both a data descriptor and an instance attribute with the same name, the descriptor wins. **Non-data descriptors** only implement __get__ and can be overridden by instance attributes. This distinction matters because it determines the attribute lookup order. Properties are data descriptors (they implement __set__), which is why you cannot accidentally bypass a property by assigning to the instance dictionary.

**7. When to Use Properties vs. Descriptors**

Use **@property** when the validation or computation logic is specific to one class and one attribute — it is simpler, more readable, and requires less boilerplate. Use **descriptors** when you have a reusable pattern of attribute access that you want to apply across multiple attributes or multiple classes — type checking, range validation, lazy loading, or auditing, for example. In both cases, keep the logic as simple as possible. Properties and descriptors that do too much become hard to debug because the behavior is hidden behind what looks like simple attribute access. Document their behavior clearly so that users of your class understand that reading or writing an attribute may trigger side effects.`,
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
