package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          51,
			Title:       "Modules & Packages",
			Description: "Understand Python modules, packages, and dependency management.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Importing Modules",
					Content: `**Modules:**
- Files containing Python code
- Can be imported and used in other files
- Organize code into reusable units

**Import Methods:**
- import module: Import entire module
- from module import name: Import specific name
- from module import *: Import all (not recommended)
- import module as alias: Import with alias

**Standard Library:**
- Built-in modules included with Python
- No installation needed
- Examples: os, sys, math, datetime, json`,
					CodeExamples: `# Import entire module
import math
print(math.pi)
print(math.sqrt(16))

# Import specific
from math import pi, sqrt
print(pi)
print(sqrt(16))

# Import with alias
import datetime as dt
now = dt.datetime.now()

# Import all (not recommended)
from math import *
print(pi)  # No math. prefix needed

# Multiple imports
import os, sys, json

# Conditional import
try:
    import optional_module
except ImportError:
    optional_module = None`,
				},
				{
					Title: "Creating Modules",
					Content: `**Creating Modules:**
- Any .py file is a module
- Module name is filename (without .py)
- Can be imported by other files

**Module Structure:**
- Imports at top
- Module-level code
- Functions and classes
- if __name__ == "__main__": for executable code

**Module Search Path:**
- Current directory
- PYTHONPATH environment variable
- Standard library directories
- Site-packages directory`,
					CodeExamples: `# mymodule.py
"""A simple module example"""

# Module-level variable
MODULE_VERSION = "1.0"

# Function
def greet(name):
    return f"Hello, {name}!"

# Class
class Calculator:
    def add(self, a, b):
        return a + b

# Executable code (runs when script is executed directly)
if __name__ == "__main__":
    print("Running module directly")
    print(greet("World"))

# In another file:
# import mymodule
# print(mymodule.greet("Alice"))
# calc = mymodule.Calculator()`,
				},
				{
					Title: "Packages",
					Content: `**Packages:**
- Directories containing modules
- Must have __init__.py file (Python 3.3+ can be empty)
- Organize related modules
- Can have subpackages

**Package Structure:**
mypackage/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        module3.py

**Importing from Packages:**
- import package.module
- from package import module
- from package.module import name`,
					CodeExamples: `# Package structure:
# mypackage/
#     __init__.py
#     math_utils.py
#     string_utils.py

# __init__.py can be empty or contain:
# from .math_utils import add, multiply
# from .string_utils import capitalize

# Importing
import mypackage.math_utils
result = mypackage.math_utils.add(1, 2)

from mypackage import math_utils
result = math_utils.add(1, 2)

from mypackage.math_utils import add
result = add(1, 2)

# If __init__.py imports:
from mypackage import add  # Direct import`,
				},
				{
					Title: "Virtual Environments",
					Content: `**Virtual Environments:**
- Isolated Python environments
- Separate package installations per project
- Avoid dependency conflicts
- Standard tool: venv (Python 3.3+)

**Creating venv:**
python -m venv venv_name

**Activating:**
- Linux/Mac: source venv/bin/activate
- Windows: venv\\Scripts\\activate

**Deactivating:**
deactivate

**Benefits:**
- Project-specific dependencies
- Reproducible environments
- Easy to share (requirements.txt)`,
					CodeExamples: `# Create virtual environment
python -m venv myenv

# Activate (Linux/Mac)
source myenv/bin/activate

# Activate (Windows)
myenv\\Scripts\\activate

# Install packages
pip install requests
pip install flask==2.0.0

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          52,
			Title:       "File I/O & Exception Handling",
			Description: "Learn file operations, context managers, and error handling.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Reading and Writing Files",
					Content: `**File Operations:**
- Open files with open() function
- Read, write, or append modes
- Always close files (or use context manager)

**File Modes:**
- 'r': Read (default)
- 'w': Write (overwrites)
- 'a': Append
- 'x': Exclusive creation
- 'b': Binary mode
- 't': Text mode (default)
- '+': Read and write

**Reading Methods:**
- read(): Read entire file
- readline(): Read one line
- readlines(): Read all lines as list
- Iterate with for loop`,
					CodeExamples: `# Reading file
file = open("data.txt", "r")
content = file.read()
file.close()

# Better: context manager
with open("data.txt", "r") as file:
    content = file.read()

# Read line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())

# Read all lines
with open("data.txt", "r") as file:
    lines = file.readlines()

# Writing file
with open("output.txt", "w") as file:
    file.write("Hello, World!\\n")
    file.write("Second line\\n")

# Appending
with open("log.txt", "a") as file:
    file.write("New entry\\n")`,
				},
				{
					Title: "Exception Handling",
					Content: `**Exceptions:**
- Errors that occur during execution
- Can be caught and handled
- Prevents program from crashing

**Try/Except:**
- try: Code that might raise exception
- except: Handle specific exceptions
- else: Execute if no exception
- finally: Always execute

**Exception Types:**
- ValueError: Invalid value
- TypeError: Wrong type
- FileNotFoundError: File not found
- KeyError: Dictionary key not found
- IndexError: List index out of range
- ZeroDivisionError: Division by zero`,
					CodeExamples: `# Basic exception handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Multiple exceptions
try:
    value = int(input("Enter number: "))
    result = 10 / value
except ValueError:
    print("Invalid number")
except ZeroDivisionError:
    print("Cannot divide by zero")

# Catch all exceptions
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")

# Else and finally
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Error")
else:
    print(f"Result: {result}")
finally:
    print("Always executes")`,
				},
				{
					Title: "Raising Exceptions",
					Content: `**Raising Exceptions:**
- Use raise keyword
- Can raise built-in or custom exceptions
- Include error message

**Custom Exceptions:**
- Create class inheriting from Exception
- Can add custom attributes
- More specific error handling

**When to Raise:**
- Invalid input
- Precondition not met
- Unrecoverable error
- To signal error to caller`,
					CodeExamples: `# Raise built-in exception
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Custom exception
class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        message = f"Insufficient funds. Balance: {balance}, Required: {amount}"
        super().__init__(message)

# Using custom exception
def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

# Catching custom exception
try:
    withdraw(100, 200)
except InsufficientFundsError as e:
    print(f"Error: {e}")
    print(f"Balance: {e.balance}, Amount: {e.amount}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          53,
			Title:       "Advanced Features",
			Description: "Explore decorators, generators, context managers, and advanced Python features.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Decorators",
					Content: `**Decorators:**
- Functions that modify other functions
- Use @ syntax
- Common for logging, timing, authentication
- Functions are first-class objects

**How Decorators Work:**
- Take function as argument
- Return modified function
- Can wrap original function
- Preserve function metadata with functools.wraps

**Common Uses:**
- Timing functions
- Caching results
- Authentication checks
- Logging`,
					CodeExamples: `# Simple decorator
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Prints 3 times

# Built-in decorators
@property
@staticmethod
@classmethod`,
				},
				{
					Title: "Generators",
					Content: `**Generators:**
- Functions that yield values instead of return
- Lazy evaluation (values generated on demand)
- Memory efficient for large sequences
- Use yield keyword

**Generator Functions:**
- Defined like functions
- Use yield instead of return
- Can have multiple yield statements
- Returns generator object

**Generator Expressions:**
- Similar to list comprehensions
- Use () instead of []
- More memory efficient`,
					CodeExamples: `# Generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Use generator
for num in countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# Generator expression
squares = (x**2 for x in range(10))
print(list(squares))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Memory efficient
def read_large_file(filename):
    with open(filename) as file:
        for line in file:
            yield line.strip()

# Process one line at a time
for line in read_large_file("huge.txt"):
    process(line)`,
				},
				{
					Title: "Context Managers",
					Content: `**Context Managers:**
- Objects used with with statement
- Ensure proper setup and cleanup
- Implement __enter__ and __exit__ methods

**With Statement:**
- Automatically handles resource cleanup
- Even if exception occurs
- Common for files, locks, connections

**Creating Context Managers:**
- Class-based: Implement __enter__ and __exit__
- Function-based: Use @contextmanager decorator`,
					CodeExamples: `# Built-in context manager (files)
with open("data.txt", "r") as file:
    content = file.read()
# File automatically closed

# Custom context manager
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        print(f"Elapsed: {time.time() - self.start:.2f}s")

with Timer():
    # Do something
    time.sleep(1)

# Using contextlib
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    print(f"Elapsed: {time.time() - start:.2f}s")

with timer():
    # Do something
    pass`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          54,
			Title:       "Standard Library",
			Description: "Explore Python's rich standard library: os, sys, datetime, collections, and more.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "os and sys Modules",
					Content: `**os Module:**
- Operating system interface
- File and directory operations
- Environment variables
- Process management

**sys Module:**
- System-specific parameters
- Command-line arguments
- Standard input/output
- Python interpreter info

**Common Functions:**
- os.getcwd(): Current directory
- os.listdir(): List directory contents
- os.path.join(): Join paths
- sys.argv: Command-line arguments
- sys.exit(): Exit program`,
					CodeExamples: `import os
import sys

# Current directory
print(os.getcwd())

# List files
files = os.listdir(".")

# Path operations
path = os.path.join("folder", "file.txt")
exists = os.path.exists(path)

# Environment variables
home = os.getenv("HOME")
path_var = os.environ.get("PATH")

# Command-line arguments
if len(sys.argv) > 1:
    print(f"Argument: {sys.argv[1]}")

# Exit program
sys.exit(0)`,
				},
				{
					Title: "datetime and time",
					Content: `**datetime Module:**
- Date and time operations
- datetime, date, time, timedelta classes
- Formatting and parsing

**time Module:**
- Time-related functions
- Sleep, time measurement
- Time conversions

**Common Operations:**
- Get current time
- Format dates
- Parse strings to dates
- Calculate differences
- Sleep/delay`,
					CodeExamples: `from datetime import datetime, timedelta
import time

# Current time
now = datetime.now()
print(now)

# Format date
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted)  # "2024-01-15 14:30:00"

# Parse string
date_str = "2024-01-15"
parsed = datetime.strptime(date_str, "%Y-%m-%d")

# Time differences
future = now + timedelta(days=7)
diff = future - now
print(diff.days)  # 7

# Sleep
time.sleep(1)  # Sleep 1 second

# Measure time
start = time.time()
# Do something
elapsed = time.time() - start`,
				},
				{
					Title: "collections Module",
					Content: `**collections Module:**
- Specialized container datatypes
- Extend built-in containers

**Common Types:**
- **deque**: Double-ended queue (fast append/pop from both ends)
- **Counter**: Count hashable objects
- **defaultdict**: Dictionary with default factory
- **OrderedDict**: Dictionary that remembers insertion order
- **namedtuple**: Tuple with named fields

**When to Use:**
- deque: When you need fast queue operations
- Counter: Counting items
- defaultdict: Avoid KeyError for missing keys`,
					CodeExamples: `from collections import deque, Counter, defaultdict, namedtuple

# Deque
queue = deque([1, 2, 3])
queue.append(4)      # Add to right
queue.appendleft(0)  # Add to left
queue.pop()         # Remove from right
queue.popleft()     # Remove from left

# Counter
words = ["apple", "banana", "apple", "cherry"]
counter = Counter(words)
print(counter)  # Counter({'apple': 2, 'banana': 1, 'cherry': 1})
print(counter.most_common(2))  # [('apple', 2), ('banana', 1)]

# defaultdict
dd = defaultdict(int)
dd["a"] += 1  # No KeyError, defaults to 0
print(dd["a"])  # 1

# namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(1, 2)
print(p.x, p.y)  # 1 2`,
				},
				{
					Title: "json and csv Modules",
					Content: `**json Module:**
- JSON (JavaScript Object Notation) support
- Serialize Python objects to JSON
- Deserialize JSON to Python objects

**csv Module:**
- Read and write CSV files
- Handle comma-separated values
- DictReader/DictWriter for dictionary-based access

**Common Operations:**
- json.dumps(): Python to JSON string
- json.loads(): JSON string to Python
- json.dump(): Python to JSON file
- json.load(): JSON file to Python`,
					CodeExamples: `import json
import csv

# JSON
data = {"name": "Alice", "age": 30}
json_str = json.dumps(data)
print(json_str)  # '{"name": "Alice", "age": 30}'

parsed = json.loads(json_str)
print(parsed["name"])  # "Alice"

# JSON file
with open("data.json", "w") as f:
    json.dump(data, f)

with open("data.json", "r") as f:
    loaded = json.load(f)

# CSV
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 30])
    writer.writerow(["Bob", 25])

# Read CSV
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["Name"], row["Age"])`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          55,
			Title:       "Testing & Debugging",
			Description: "Learn unittest, pytest, mocking, and debugging techniques.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "unittest Framework",
					Content: `**unittest:**
- Built-in testing framework
- Inspired by JUnit (Java)
- Organize tests into classes

**Test Structure:**
- Inherit from unittest.TestCase
- Test methods start with test_
- Use assert methods for checks
- setUp() and tearDown() for fixtures

**Assert Methods:**
- assertEqual(), assertNotEqual()
- assertTrue(), assertFalse()
- assertIn(), assertNotIn()
- assertRaises()`,
					CodeExamples: `import unittest

def add(a, b):
    return a + b

class TestMath(unittest.TestCase):
    def setUp(self):
        # Run before each test
        self.num1 = 5
        self.num2 = 3
    
    def test_add_positive(self):
        self.assertEqual(add(5, 3), 8)
    
    def test_add_negative(self):
        self.assertEqual(add(-1, -2), -3)
    
    def test_add_zero(self):
        self.assertEqual(add(0, 5), 5)
    
    def tearDown(self):
        # Run after each test
        pass

if __name__ == "__main__":
    unittest.main()`,
				},
				{
					Title: "pytest Basics",
					Content: `**pytest:**
- Popular third-party testing framework
- Simpler syntax than unittest
- Better fixtures and plugins
- Install: pip install pytest

**Features:**
- No need for classes
- Simple assert statements
- Automatic test discovery
- Rich assertion introspection

**Running Tests:**
- pytest test_file.py
- pytest (discovers all tests)
- pytest -v (verbose)
- pytest -k pattern (run matching tests)`,
					CodeExamples: `# test_math.py
def add(a, b):
    return a + b

def test_add_positive():
    assert add(5, 3) == 8

def test_add_negative():
    assert add(-1, -2) == -3

def test_add_zero():
    assert add(0, 5) == 5

# Run: pytest test_math.py

# Fixtures
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15`,
				},
				{
					Title: "Debugging Techniques",
					Content: `**Debugging Methods:**

1. **print()**: Simple debugging output
2. **pdb**: Python debugger (interactive)
3. **IDE debuggers**: Visual debugging
4. **logging**: Structured logging

**pdb Commands:**
- n (next): Execute next line
- s (step): Step into function
- c (continue): Continue execution
- l (list): Show code
- p variable: Print variable
- q (quit): Exit debugger`,
					CodeExamples: `# Using print
def calculate(x, y):
    print(f"DEBUG: x={x}, y={y}")
    result = x + y
    print(f"DEBUG: result={result}")
    return result

# Using pdb
import pdb

def calculate(x, y):
    pdb.set_trace()  # Breakpoint
    result = x + y
    return result

# Using logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate(x, y):
    logger.debug(f"Calculating with x={x}, y={y}")
    result = x + y
    logger.debug(f"Result: {result}")
    return result`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          56,
			Title:       "Web Development Basics",
			Description: "Introduction to Flask, Django, and web development with Python.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Flask Basics",
					Content: `**Flask:**
- Lightweight web framework
- Minimal and flexible
- Great for APIs and small apps
- Install: pip install flask

**Basic Flask App:**
- Create Flask instance
- Define routes with @app.route()
- Return HTML or JSON
- Run with app.run()

**Routes:**
- Map URLs to functions
- Support HTTP methods (GET, POST, etc.)
- Can have URL parameters`,
					CodeExamples: `from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello, Flask!</h1>"

@app.route("/api/users/<int:user_id>")
def get_user(user_id):
    return jsonify({"id": user_id, "name": "Alice"})

@app.route("/api/data", methods=["POST"])
def create_data():
    data = request.json
    return jsonify({"status": "created", "data": data}), 201

if __name__ == "__main__":
    app.run(debug=True)`,
				},
				{
					Title: "Django Overview",
					Content: `**Django:**
- Full-featured web framework
- "Batteries included" philosophy
- ORM, admin panel, authentication
- Install: pip install django

**Django Structure:**
- Project: Collection of apps
- App: Self-contained module
- Models: Database schema
- Views: Request handlers
- Templates: HTML rendering
- URLs: URL routing

**Django Commands:**
- django-admin startproject: Create project
- python manage.py startapp: Create app
- python manage.py runserver: Run server
- python manage.py migrate: Apply migrations`,
					CodeExamples: `# Django project structure:
# myproject/
#     manage.py
#     myproject/
#         settings.py
#         urls.py
#         wsgi.py
#     myapp/
#         models.py
#         views.py
#         urls.py

# models.py
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()

# views.py
from django.http import JsonResponse
from .models import User

def user_list(request):
    users = User.objects.all()
    return JsonResponse({"users": list(users.values())})`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          57,
			Title:       "Best Practices & Idioms",
			Description: "Learn PEP 8, Pythonic code patterns, and common pitfalls.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "PEP 8 Style Guide",
					Content: `**PEP 8:**
- Python Enhancement Proposal 8
- Official style guide for Python
- Improves code readability

**Key Rules:**
- Use 4 spaces for indentation
- Maximum 79 characters per line
- Use blank lines to separate functions/classes
- Import statements at top
- Use snake_case for functions/variables
- Use PascalCase for classes
- Use UPPER_CASE for constants

**Tools:**
- autopep8: Auto-format code
- flake8: Check style
- black: Opinionated formatter`,
					CodeExamples: `# Good PEP 8 style
import os
import sys

MAX_SIZE = 100

class UserManager:
    """Manages user operations."""
    
    def __init__(self):
        self.users = []
    
    def add_user(self, name, email):
        """Add a new user."""
        user = {"name": name, "email": email}
        self.users.append(user)

# Bad style
import os,sys
MAX_SIZE=100
class userManager:
    def __init__(self):
        self.users=[]
    def add_user(self,name,email):
        user={"name":name,"email":email}
        self.users.append(user)`,
				},
				{
					Title: "Pythonic Code Patterns",
					Content: `**Pythonic Patterns:**
- Code that follows Python conventions
- Idiomatic and readable
- Leverages Python features

**Common Patterns:**
- List comprehensions over loops
- Context managers for resources
- Generator expressions for memory efficiency
- Unpacking for multiple assignment
- enumerate() instead of range(len())
- Use in operator for membership
- Use zip() for parallel iteration`,
					CodeExamples: `# Pythonic: List comprehension
squares = [x**2 for x in range(10)]

# Not Pythonic: Loop
squares = []
for x in range(10):
    squares.append(x**2)

# Pythonic: enumerate
for index, value in enumerate(items):
    print(f"{index}: {value}")

# Not Pythonic: range(len())
for i in range(len(items)):
    print(f"{i}: {items[i]}")

# Pythonic: Unpacking
first, *rest = [1, 2, 3, 4]

# Pythonic: Context manager
with open("file.txt") as f:
    content = f.read()

# Pythonic: Generator
large_squares = (x**2 for x in range(1000000))`,
				},
				{
					Title: "Common Pitfalls",
					Content: `**Common Mistakes:**

1. **Mutable Default Arguments:**
   - Don't use mutable objects as default values
   - Use None and check inside function

2. **Modifying List While Iterating:**
   - Create copy or iterate backwards

3. **== vs is:**
   - == compares values
   - is compares identity (same object)

4. **Variable Scope:**
   - Understand local vs global
   - Use global/nonlocal when needed

5. **String Concatenation:**
   - Use join() for multiple strings
   - Avoid + in loops`,
					CodeExamples: `# Mutable default argument (WRONG)
def add_item(item, items=[]):
    items.append(item)
    return items

# Correct
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

# Modifying while iterating (WRONG)
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # Problem!

# Correct
numbers = [num for num in numbers if num % 2 != 0]

# == vs is
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True (same values)
print(a is b)  # False (different objects)

# String concatenation (WRONG)
result = ""
for word in words:
    result += word  # Inefficient

# Correct
result = "".join(words)`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
