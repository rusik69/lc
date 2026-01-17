package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          44,
			Title:       "Introduction to Python",
			Description: "Learn the fundamentals of Python: history, installation, and your first program.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Python?",
					Content: `Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Named after Monty Python's Flying Circus, Python emphasizes code readability and simplicity.

**Design Philosophy (PEP 20 - The Zen of Python):**
- Beautiful is better than ugly
- Explicit is better than implicit
- Simple is better than complex
- Complex is better than complicated
- Readability counts
- There should be one obvious way to do it (but it may not be obvious at first)

**Key Features:**
- **Interpreted**: No compilation step, runs directly from source code
- **Dynamically Typed**: Types determined at runtime, flexible but requires careful testing
- **Multi-paradigm**: Supports OOP, functional, procedural, and imperative styles
- **Extensive Standard Library**: "Batteries included" - rich built-in modules
- **Cross-platform**: Write once, run on Windows, macOS, Linux, and more
- **Open Source**: Free to use, modify, and distribute

**Python Versions:**
- **Python 2.x**: Legacy version, end of life January 1, 2020
- **Python 3.x**: Current version (3.8+ recommended, 3.11+ for latest features)
- **Always use Python 3** for new projects - Python 2 is deprecated

**Performance Characteristics:**
- Slower than compiled languages (C, C++, Go) for CPU-intensive tasks
- Fast enough for most applications (web, data processing, automation)
- Can integrate with C/C++ for performance-critical sections
- Excellent for I/O-bound tasks (web requests, file operations, databases)

**Real-World Applications:**
- **Web Development**: Django, Flask, FastAPI frameworks
- **Data Science**: NumPy, Pandas, Matplotlib, Jupyter
- **Machine Learning**: TensorFlow, PyTorch, Scikit-learn
- **Automation**: Scripting, system administration, DevOps
- **Scientific Computing**: Research, simulations, data analysis
- **Game Development**: Pygame, Panda3D
- **Desktop Applications**: Tkinter, PyQt, Kivy

**Companies Using Python:**
- Google (YouTube, Google Search infrastructure)
- Netflix (content delivery, recommendation algorithms)
- Instagram (web backend, originally Django)
- Spotify (data analysis, backend services)
- Dropbox (desktop client, server infrastructure)
- Reddit (web framework)
- NASA (scientific computing, data analysis)
- Facebook (infrastructure tools, data processing)`,
					CodeExamples: `# Hello World - Python's simplicity
print("Hello, World!")

# Readable code - self-documenting
def calculate_total_price(items, tax_rate=0.08):
    """Calculate total price with tax."""
    subtotal = sum(item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax

# Multiple paradigms in one language
# Functional style
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))

# List comprehension (Pythonic)
squares = [x**2 for x in numbers]

# Object-oriented style
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

# Dynamic typing example
value = 42        # integer
value = "hello"   # now string
value = [1, 2, 3] # now list
# Same variable, different types`,
				},
				{
					Title: "Installation and Setup",
					Content: `**Installing Python:**

**Windows:**
1. Download installer from https://www.python.org/downloads/
2. **Important**: Check "Add Python to PATH" during installation
3. Choose "Install for all users" if you have admin rights
4. Verify: Open Command Prompt and run: python --version

**macOS:**
- Option 1: Download from python.org
- Option 2: Use Homebrew: brew install python3
- Option 3: Use pyenv for version management: brew install pyenv

**Linux:**
- Ubuntu/Debian: sudo apt-get install python3 python3-pip
- Fedora: sudo dnf install python3 python3-pip
- Arch: sudo pacman -S python python-pip

**Python Shell (REPL):**
- **REPL**: Read-Eval-Print Loop
- Interactive interpreter: python or python3
- Try code immediately without creating files
- Great for testing, debugging, and learning
- Exit with exit() or Ctrl+D (Linux/Mac) / Ctrl+Z+Enter (Windows)

**IDEs and Editors:**

**Beginner-Friendly:**
- **IDLE**: Built-in with Python, simple and lightweight
- **Thonny**: Designed for beginners, great for learning

**Professional IDEs:**
- **PyCharm**: Full-featured IDE (Community Edition is free)
  - Excellent debugging, refactoring, and code completion
  - Built-in terminal, database tools, version control
- **VS Code**: Popular lightweight editor with Python extension
  - Free, extensible, great for web development
  - Excellent Git integration and extensions marketplace
- **Spyder**: Scientific Python IDE, great for data science

**Notebooks:**
- **Jupyter Notebook**: Interactive notebooks for data science
- **JupyterLab**: Next-generation notebook interface
- **Google Colab**: Free cloud-based notebooks

**Virtual Environments - Why They Matter:**
- **Isolation**: Each project has its own dependencies
- **Version Control**: Different projects can use different package versions
- **Reproducibility**: Share exact environment with requirements.txt
- **Clean System**: Avoid polluting system Python installation

**Virtual Environment Best Practices:**
- Always use virtual environments for projects
- Never install packages globally (except tools like pip, setuptools)
- Include venv/ in .gitignore (don't commit virtual environments)
- Use requirements.txt to document dependencies
- Consider using pipenv or poetry for more advanced dependency management`,
					CodeExamples: `# Check Python version
python --version
# or
python3 --version

# Check pip version
pip --version
pip3 --version

# Create virtual environment
python -m venv myenv
# or
python3 -m venv myenv

# Activate (Linux/Mac)
source myenv/bin/activate
# Prompt changes to show (myenv)

# Activate (Windows Command Prompt)
myenv\\Scripts\\activate.bat

# Activate (Windows PowerShell)
myenv\\Scripts\\Activate.ps1
# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install packages
pip install requests
pip install flask==2.0.0  # Specific version
pip install "django>=3.0,<4.0"  # Version range

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Upgrade pip
python -m pip install --upgrade pip

# Deactivate virtual environment
deactivate

# Using venv with different Python version
python3.11 -m venv myenv311

# Virtual environment structure
# myenv/
#   bin/          # (Scripts/ on Windows) - activation scripts
#   include/       # Header files
#   lib/          # Installed packages
#   pyvenv.cfg    # Configuration file`,
				},
				{
					Title: "Your First Python Program",
					Content: `**Running Python Code:**

**Three Main Ways:**

1. **Interactive Shell (REPL)**
   - Type python or python3 in terminal
   - Execute code line by line
   - See results immediately
   - Perfect for experimentation and quick tests

2. **Script Files**
   - Save code in .py files
   - Run with: python script.py or python3 script.py
   - Full programs and applications
   - Can be imported as modules

3. **Jupyter Notebooks**
   - Interactive cells for data science
   - Mix code, markdown, and visualizations
   - Great for exploratory analysis

**Basic Program Structure:**
- **No semicolons**: Python uses newlines to end statements
- **Indentation is syntax**: 4 spaces standard (PEP 8), tabs also work but spaces preferred
- **Comments**: # for single-line, triple quotes for multi-line
- **Docstrings**: Triple-quoted strings at module/function/class start
- **Main guard**: if __name__ == "__main__": allows script to be run or imported

**Indentation Rules:**
- Python uses indentation to define code blocks (unlike braces in C/Java)
- Consistent indentation is required (mixing tabs and spaces causes errors)
- Standard is 4 spaces per indentation level
- Indentation level determines scope

**Common Pitfalls:**
- **IndentationError**: Mixing tabs and spaces, inconsistent indentation
- **SyntaxError**: Missing colons after if/for/def statements
- **NameError**: Using variable before assignment
- **ImportError**: Module not found (check PYTHONPATH, virtual environment)`,
					CodeExamples: `# hello.py - Your first program
"""
Module docstring - describes what this module does.
This is a simple hello world program.
"""

# Import statements at the top
import sys
from datetime import datetime

# Module-level constants
GREETING = "Hello, Python!"

# Function definitions
def greet(name="World"):
    """Greet someone by name.
    
    Args:
        name: Name to greet (default: "World")
    
    Returns:
        Greeting message string
    """
    return f"{GREETING} {name}!"

def main():
    """Main entry point of the program."""
    print(greet("Alice"))
    print("Python uses indentation for blocks")
    
    # Control structures use indentation
    if True:
        print("This is indented 4 spaces")
        print("So is this")
        if True:
            print("This is indented 8 spaces (nested)")
    
    # Lists and loops
    fruits = ["apple", "banana", "cherry"]
    for fruit in fruits:
        print(f"  - {fruit}")  # Indented inside loop

# Main guard - allows script to be run directly or imported
if __name__ == "__main__":
    main()
    # When run directly: python hello.py
    # When imported: import hello (main() won't run)

# Running from command line:
# python hello.py
# python3 hello.py
# ./hello.py (if shebang line: #!/usr/bin/env python3)

# Interactive shell example:
# $ python3
# >>> print("Hello!")
# Hello!
# >>> x = 10
# >>> x * 2
# 20
# >>> exit()`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          45,
			Title:       "Basic Syntax & Types",
			Description: "Master Python's basic types, variables, and type system.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Variables and Naming",
					Content: `**Variables in Python:**

**Key Characteristics:**
- **No declaration needed**: Variables are created when first assigned
- **Dynamically typed**: Type is inferred from value, not declared
- **Type can change**: Variable can be reassigned to different types
- **References, not values**: Variables hold references to objects

**Variable Assignment:**
- Simple: name = "Alice"
- Multiple: x, y, z = 1, 2, 3 (tuple unpacking)
- Chained: x = y = z = 0 (all reference same object - be careful with mutable types!)

**Naming Rules:**
- Must start with letter (a-z, A-Z) or underscore (_)
- Can contain letters, digits (0-9), and underscores
- **Case-sensitive**: name, Name, NAME are different variables
- Cannot use Python keywords (if, for, def, class, etc.)
- Cannot use special characters (@, #, $, %, etc.)

**Python Keywords (cannot use as variable names):**
False, None, True, and, as, assert, async, await, break, class, continue, def, del, elif, else, except, finally, for, from, global, if, import, in, is, lambda, nonlocal, not, or, pass, raise, return, try, while, with, yield

**Naming Conventions (PEP 8):**
- **snake_case**: Variables, functions, methods, modules, packages
  - Example: user_name, calculate_total, my_module
- **UPPER_CASE**: Constants (convention, Python has no true constants)
  - Example: MAX_SIZE, DEFAULT_TIMEOUT, PI
- **PascalCase**: Classes
  - Example: UserAccount, DatabaseConnection
- **Single leading underscore (_name)**: Internal use, "protected"
  - Signals "internal implementation, don't use directly"
- **Double leading underscore (__name)**: Name mangling (rarely used)
  - Python mangles the name to avoid conflicts in inheritance
- **Trailing underscore (name_)**: Avoid conflict with keywords
  - Example: class_ (if class is a keyword conflict)

**Best Practices:**
- Use descriptive names: user_count not uc
- Avoid single letters except for loop counters: for i in range(10)
- Use verbs for functions: calculate_total(), get_user()
- Use nouns for variables: user_name, total_price
- Be consistent with naming style throughout project`,
					CodeExamples: `# Basic variable assignment
name = "Alice"
age = 30
height = 5.6
is_student = True

# Dynamic typing - type can change
x = 10        # x is int
print(type(x))  # <class 'int'>

x = "hello"   # x is now str
print(type(x))  # <class 'str'>

x = [1, 2, 3] # x is now list
print(type(x))  # <class 'list'>

# Multiple assignment (tuple unpacking)
a, b, c = 1, 2, 3
print(a, b, c)  # 1 2 3

# Swap variables easily
x, y = 10, 20
x, y = y, x  # Swap without temp variable
print(x, y)  # 20 10

# Chained assignment (be careful!)
x = y = z = []  # All reference same list!
x.append(1)
print(y, z)  # [1] [1] - all affected!

# Better for mutable types:
x = []
y = []
z = []

# Naming examples following PEP 8
user_name = "alice"           # snake_case for variables
MAX_SIZE = 100                # UPPER_CASE for constants
_is_internal = True           # Leading underscore for internal
class_ = "Python 101"         # Trailing underscore to avoid keyword

# Good naming
def calculate_total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total

# Bad naming (too short, unclear)
def calc(items):
    t = 0
    for i in items:
        t += i.p
    return t

# Check if name is keyword
import keyword
print(keyword.iskeyword("if"))    # True
print(keyword.iskeyword("name"))   # False`,
				},
				{
					Title: "Basic Types",
					Content: `**Python's Built-in Types:**

**Numeric Types:**

**int (Integer):**
- Unlimited precision in Python 3 (no overflow!)
- Can be arbitrarily large: 2**1000 works perfectly
- Supports binary (0b), octal (0o), hexadecimal (0x) literals
- Underscores for readability: 1_000_000

**float (Floating-Point):**
- Double precision (64-bit) by default
- Approximations - be careful with equality comparisons!
- Scientific notation: 1.5e10 = 15000000000.0
- Special values: float('inf'), float('-inf'), float('nan')

**complex (Complex Numbers):**
- Format: 3 + 4j or complex(3, 4)
- Used in scientific computing, signal processing
- Access real/imaginary parts: .real, .imag

**Text Type:**

**str (String):**
- Immutable sequence of Unicode characters
- Can use single quotes, double quotes, or triple quotes
- Triple quotes preserve newlines and formatting
- Raw strings: r"C:\\path" (backslashes not escaped)
- f-strings: f"Hello {name}" (Python 3.6+)

**Boolean Type:**

**bool:**
- Only two values: True and False (capitalized!)
- Subclass of int: True == 1, False == 0
- Truthiness: All values can be evaluated as True/False
- Falsy values: False, None, 0, 0.0, empty string, empty list, empty dict, empty tuple, empty set

**None Type:**

**None:**
- Represents absence of value (like null in other languages)
- Only one None object exists (singleton)
- Common default return value for functions
- Use is None or is not None for comparison (not ==)

**Type Checking:**

**type() function:**
- Returns the exact type of an object
- Returns a type object: <class 'int'>

**isinstance() function (preferred):**
- Checks if object is instance of type or subclass
- More flexible: isinstance(x, (int, float)) checks multiple types
- Handles inheritance correctly

**Type Hints (Python 3.5+):**
- Optional annotations for better code documentation
- Use typing module for complex types
- Tools like mypy can check types statically`,
					CodeExamples: `# Integers - unlimited precision!
small = 42
large = 999999999999999999999999999999999999999999999999999999999999
huge = 2**1000  # Works perfectly in Python 3!

# Different bases
binary = 0b1010      # 10 in decimal
octal = 0o755        # 493 in decimal
hexadecimal = 0xFF   # 255 in decimal

# Underscores for readability
million = 1_000_000
credit_card = 1234_5678_9012_3456

# Floats - be careful with precision!
pi = 3.14159
scientific = 1.5e10  # 15000000000.0
small = 1e-5         # 0.00001

# Float precision issues
print(0.1 + 0.2)  # 0.30000000000000004 (not exactly 0.3!)
# Use decimal.Decimal for exact decimal arithmetic

# Special float values
infinity = float('inf')
negative_inf = float('-inf')
not_a_number = float('nan')

# Complex numbers
c1 = 3 + 4j
c2 = complex(3, 4)  # Same as above
print(c1.real)     # 3.0
print(c1.imag)     # 4.0
print(abs(c1))     # 5.0 (magnitude)

# Strings - multiple ways to create
single = 'Single quotes'
double = "Double quotes"
triple = """Triple quotes
can span
multiple lines"""

# Raw strings (no escape processing)
path = r"C:\\Users\\Name\\Documents"  # Backslashes literal
regex = r"\d+\.\d+"  # Regex pattern

# f-strings (formatted string literals)
name = "Alice"
age = 30
message = f"{name} is {age} years old"
expression = f"Next year: {age + 1}"

# Boolean
is_active = True
is_deleted = False

# Truthiness evaluation
print(bool(1))        # True
print(bool(0))        # False
print(bool(""))       # False
print(bool("hello"))  # True
print(bool([]))       # False
print(bool([1, 2]))   # True

# None
value = None
if value is None:  # Use 'is', not '=='
    print("No value assigned")

# Type checking
x = 42
print(type(x))              # <class 'int'>
print(type(x) == int)       # True
print(isinstance(x, int))   # True (preferred)

# isinstance handles inheritance
class Animal:
    pass

class Dog(Animal):
    pass

d = Dog()
print(isinstance(d, Dog))    # True
print(isinstance(d, Animal)) # True (inheritance)
print(type(d) == Animal)      # False (exact type only)

# Type hints (Python 3.5+)
from typing import List, Dict, Optional

def process_numbers(numbers: List[int]) -> int:
    return sum(numbers)

def get_user(id: int) -> Optional[Dict[str, str]]:
    if id > 0:
        return {"id": str(id), "name": "Alice"}
    return None`,
				},
				{
					Title: "Type Conversion",
					Content: `**Type Conversion:**
- Use built-in functions: int(), float(), str(), bool()
- Some conversions may raise ValueError

**Common Conversions:**
- int() converts to integer
- float() converts to float
- str() converts to string
- bool() converts to boolean (truthiness)

**Implicit Conversion:**
- Python automatically converts types in some operations
- Example: int + float = float`,
					CodeExamples: `# Explicit conversion
x = int(3.14)      # 3
y = float(42)     # 42.0
z = str(100)       # "100"
b = bool(1)        # True

# String to number
age = int("25")
price = float("19.99")

# Number to string
message = "Age: " + str(25)

# Boolean conversion
print(bool(1))     # True
print(bool(0))     # False
print(bool(""))    # False
print(bool("hi"))  # True

# Implicit conversion
result = 5 + 3.14  # 8.14 (float)`,
				},
				{
					Title: "Comments and Docstrings",
					Content: `**Comments:**
- Single-line: # comment
- Multi-line: Use multiple # lines
- Comments are ignored by interpreter

**Docstrings:**
- Triple-quoted strings used for documentation
- First statement in module, function, or class
- Accessible via __doc__ attribute
- Follow PEP 257 conventions

**When to Use:**
- Comments: Explain why or how
- Docstrings: Document what functions/classes do`,
					CodeExamples: `# Single-line comment
x = 10  # This is a comment

# Multi-line comment
# Line 1
# Line 2
# Line 3

def calculate_total(items):
    """
    Calculate the total price of items.
    
    Args:
        items: List of item prices
        
    Returns:
        Total price as float
    """
    return sum(items)

# Access docstring
print(calculate_total.__doc__)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          46,
			Title:       "Control Structures",
			Description: "Learn if/else, loops, and flow control in Python.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "If/Elif/Else Statements",
					Content: `**Conditional Statements:**

**If Statement:**
- Tests a condition (expression that evaluates to True/False)
- Executes block if condition is True
- Uses indentation (4 spaces standard) to define block scope
- No parentheses needed around condition (unlike C/Java)

**Elif (Else If):**
- Short for "else if"
- Tests additional conditions sequentially
- Can have multiple elif clauses
- Only first True condition executes (short-circuit evaluation)
- More efficient than multiple if statements

**Else:**
- Executes if all previous conditions are False
- Optional clause
- Catches all remaining cases

**Truthiness in Python:**
Python evaluates "truthiness" - any value can be used in boolean context:

**Falsy Values (evaluate to False):**
- False
- None
- 0 (zero of any numeric type)
- 0.0, 0j (complex zero)
- "" (empty string)
- [] (empty list)
- {} (empty dict)
- () (empty tuple)
- set() (empty set)

**Truthy Values (evaluate to True):**
- Everything else!
- Non-empty strings, lists, dicts, tuples, sets
- Non-zero numbers
- Custom objects (unless they define __bool__() or __len__())

**Common Patterns:**
- Guard clauses: Early returns for invalid conditions
- Ternary operator: value if condition else other_value
- Chained comparisons: 0 < x < 10 (Python-specific!)
- Membership testing: if item in collection

**Best Practices:**
- Use explicit comparisons for clarity: if x == 0: vs if not x:
- Avoid deep nesting (use early returns/guard clauses)
- Use elif instead of multiple if statements when appropriate
- Be careful with mutable defaults in function parameters`,
					CodeExamples: `# Simple if
age = 18
if age >= 18:
    print("Adult")

# If-else
if age >= 18:
    print("Adult")
else:
    print("Minor")

# If-elif-else chain
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Truthiness examples
name = "Alice"
if name:  # True - non-empty string
    print(f"Hello, {name}")

items = []
if not items:  # True - empty list is falsy
    print("No items")

# Guard clause pattern (early return)
def process_user(user):
    if user is None:
        return None  # Early exit
    if not user.is_active:
        return None
    # Process active user...
    return user.process()

# Ternary operator (conditional expression)
age = 20
status = "Adult" if age >= 18 else "Minor"

# Chained comparisons (Python-specific!)
x = 5
if 0 < x < 10:  # Equivalent to: 0 < x and x < 10
    print("x is between 0 and 10")

# Membership testing
fruits = ["apple", "banana", "cherry"]
if "apple" in fruits:
    print("Found apple!")

# Multiple conditions
age = 25
has_license = True
if age >= 18 and has_license:
    print("Can drive")

# Using 'or' for defaults
name = user_input or "Guest"  # Use "Guest" if user_input is falsy

# Checking for None explicitly
value = None
if value is None:  # Use 'is', not '=='
    print("No value")

# Negation patterns
if not user.is_banned:  # More readable than: if user.is_banned == False
    print("User can post")

# Complex conditions with parentheses
if (age >= 18 and has_license) or is_admin:
    print("Can drive")

# Short-circuit evaluation
if items and items[0] == "apple":  # Won't error if items is empty
    print("First item is apple")`,
				},
				{
					Title: "For Loops",
					Content: `**For Loops:**
- Iterate over sequences (lists, strings, tuples, etc.)
- More Pythonic than while loops for iteration
- Use range() for numeric loops

**Range Function:**
- range(stop): 0 to stop-1
- range(start, stop): start to stop-1
- range(start, stop, step): with step size

**Enumerate:**
- Get both index and value
- enumerate(sequence) returns (index, value) pairs

**Loop Control:**
- break: Exit loop immediately
- continue: Skip to next iteration
- else: Executes if loop completes normally (no break)`,
					CodeExamples: `# Iterate over list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Range loop
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for i in range(1, 6):
    print(i)  # 1, 2, 3, 4, 5

for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# Enumerate
for index, value in enumerate(fruits):
    print(f"{index}: {value}")

# Loop with else
for i in range(5):
    if i == 10:
        break
else:
    print("Loop completed normally")`,
				},
				{
					Title: "While Loops",
					Content: `**While Loops:**
- Execute while condition is True
- Check condition before each iteration
- Can create infinite loops if condition never becomes False

**Common Patterns:**
- Counter-based loops
- Condition-based loops
- Infinite loops with break

**Best Practices:**
- Ensure condition eventually becomes False
- Use break for early exit
- Avoid infinite loops unless intentional`,
					CodeExamples: `# Basic while loop
count = 0
while count < 5:
    print(count)
    count += 1

# Condition-based
user_input = ""
while user_input != "quit":
    user_input = input("Enter command: ")
    print(f"You entered: {user_input}")

# Infinite loop with break
while True:
    response = input("Enter 'q' to quit: ")
    if response == 'q':
        break
    print("Continuing...")

# While-else
count = 0
while count < 3:
    print(count)
    count += 1
else:
    print("Loop finished")`,
				},
				{
					Title: "Break, Continue, and Pass",
					Content: `**Break:**
- Exits the innermost loop immediately
- Skips remaining iterations
- Can break out of nested loops with labels (Python 3.10+)

**Continue:**
- Skips current iteration
- Continues with next iteration
- Useful for filtering or skipping certain cases

**Pass:**
- Placeholder statement
- Does nothing
- Used when syntax requires a statement but no action needed
- Common in empty functions, classes, or exception handlers`,
					CodeExamples: `# Break example
for i in range(10):
    if i == 5:
        break
    print(i)  # Prints 0-4

# Continue example
for i in range(10):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)  # Prints 1, 3, 5, 7, 9

# Pass example
def placeholder_function():
    pass  # To be implemented later

class EmptyClass:
    pass  # To be implemented later

try:
    risky_operation()
except:
    pass  # Ignore errors (usually not recommended)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          47,
			Title:       "Functions",
			Description: "Master function definition, parameters, return values, and scope.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Function Definition",
					Content: `**Functions in Python:**

**Key Concepts:**
- **Defined with def keyword**: Functions are first-class objects
- **Reusable blocks**: Encapsulate logic for reuse
- **Parameters and return values**: Can take input and produce output
- **First-class objects**: Can be assigned, passed as arguments, returned, stored in data structures

**Function Syntax:**
def function_name(parameters):
    """Docstring - describes what function does"""
    # Function body
    return value  # Optional

**Function Components:**
1. **def keyword**: Marks function definition
2. **Function name**: Follows naming conventions (snake_case)
3. **Parameters**: Input values (can be zero or more)
4. **Docstring**: Documentation (triple-quoted string)
5. **Body**: Indented code block
6. **return statement**: Optional, exits function and returns value

**Calling Functions:**
- Use function name followed by parentheses: function_name()
- Pass arguments for parameters: function_name(arg1, arg2)
- Can capture return value: result = function_name()
- Functions without return statement return None

**Function vs Method:**
- **Function**: Standalone, defined at module level
- **Method**: Function defined inside a class, called on object

**Why Use Functions:**
- **DRY Principle**: Don't Repeat Yourself
- **Modularity**: Break code into logical pieces
- **Testability**: Easier to test isolated functions
- **Readability**: Self-documenting code with good names
- **Maintainability**: Change in one place affects all calls`,
					CodeExamples: `# Simple function (no parameters, no return)
def greet():
    print("Hello!")

greet()  # Call function
# Output: Hello!

# Function with parameters
def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")
# Output: Hello, Alice!

# Function with return value
def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # 8

# Function as first-class object
def multiply(x, y):
    return x * y

# Assign function to variable
operation = multiply
print(operation(4, 5))  # 20

# Pass function as argument
def apply_operation(func, a, b):
    return func(a, b)

result = apply_operation(multiply, 3, 4)
print(result)  # 12

# Function stored in data structure
operations = {
    'add': lambda x, y: x + y,
    'multiply': multiply,
    'subtract': lambda x, y: x - y
}

print(operations['add'](5, 3))  # 8

# Function without return (returns None)
def print_sum(a, b):
    print(a + b)

result = print_sum(2, 3)  # Prints: 5
print(result)  # None`,
				},
				{
					Title: "Parameters and Arguments",
					Content: `**Understanding Parameters and Arguments:**

**Terminology:**
- **Parameter**: Variable in function definition
- **Argument**: Value passed to function when calling it

**Parameter Types:**

1. **Positional Parameters:**
   - Passed in order based on position
   - Must be provided (unless has default)
   - Most common type

2. **Keyword Arguments:**
   - Passed by parameter name
   - Order doesn't matter
   - More readable for functions with many parameters
   - Can mix with positional (positional first)

3. **Default Parameters:**
   - Have default values assigned
   - Optional to provide when calling
   - Must come after non-default parameters
   - **Warning**: Use immutable defaults! (common pitfall)

4. ***args (Variable Positional):**
   - Collects extra positional arguments as tuple
   - Name 'args' is convention, can be any name
   - Useful for functions accepting variable number of arguments

5. ****kwargs (Variable Keyword):**
   - Collects extra keyword arguments as dictionary
   - Name 'kwargs' is convention, can be any name
   - Useful for passing configuration options

**Parameter Order (when mixing):**
1. Positional parameters (no defaults)
2. Positional parameters (with defaults)
3. *args
4. Keyword-only parameters (after *args)
5. **kwargs

**Common Pitfalls:**
- **Mutable default arguments**: Default values evaluated once at definition time
- **Modifying mutable defaults**: Changes persist across calls
- **Solution**: Use None as default, create new object inside function`,
					CodeExamples: `# Positional arguments
def greet(first, last):
    print(f"{first} {last}")

greet("Alice", "Smith")  # Alice Smith
greet("Smith", "Alice")  # Smith Alice (order matters!)

# Keyword arguments
greet(last="Smith", first="Alice")  # Alice Smith (order doesn't matter)
greet(first="Alice", last="Smith")  # Same result

# Mixing positional and keyword
def create_user(name, age, email, active=True):
    return {"name": name, "age": age, "email": email, "active": active}

# Positional first, then keyword
user1 = create_user("Alice", 30, "alice@example.com")
user2 = create_user("Bob", 25, "bob@example.com", active=False)
user3 = create_user("Charlie", email="charlie@example.com", age=28)

# Default parameters
def greet(name, greeting="Hello", punctuation="!"):
    print(f"{greeting}, {name}{punctuation}")

greet("Alice")  # Hello, Alice!
greet("Bob", "Hi")  # Hi, Bob!
greet("Charlie", "Hey", ".")  # Hey, Charlie.

# *args - variable positional arguments
def sum_all(*args):
    print(f"Received {len(args)} arguments")
    return sum(args)

print(sum_all(1, 2, 3))  # 6
print(sum_all(1, 2, 3, 4, 5))  # 15
print(sum_all())  # 0 (empty tuple)

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="NYC")
# name: Alice
# age: 30
# city: NYC

# Combining all types
def complex_function(required, optional="default", *args, **kwargs):
    print(f"Required: {required}")
    print(f"Optional: {optional}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

complex_function(1, "opt", 2, 3, key1="value1", key2="value2")

# MUTABLE DEFAULT ARGUMENT PITFALL (WRONG!)
def add_item(item, items=[]):  # BAD!
    items.append(item)
    return items

list1 = add_item("apple")
list2 = add_item("banana")
print(list1)  # ['apple', 'banana'] - WRONG! Should be ['apple']
print(list2)  # ['apple', 'banana'] - WRONG! Should be ['banana']

# CORRECT way - use None
def add_item_correct(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

list1 = add_item_correct("apple")
list2 = add_item_correct("banana")
print(list1)  # ['apple'] - Correct!
print(list2)  # ['banana'] - Correct!`,
				},
				{
					Title: "Return Values",
					Content: `**Return Statement:**

**How Return Works:**
- **Exits function immediately**: No code after return executes
- **Can return any type**: int, str, list, dict, object, None, etc.
- **Returns None by default**: If no return statement or return without value
- **Can return multiple values**: Actually returns a tuple (can unpack)

**Return Patterns:**

1. **Single Value Return:**
   - Most common pattern
   - Returns one value

2. **Multiple Values Return:**
   - Returns tuple (comma-separated values)
   - Can unpack into multiple variables
   - Useful for functions that compute multiple related values

3. **Early Returns:**
   - Return early for error cases or special conditions
   - Reduces nesting (guard clauses)
   - More readable code

4. **Conditional Returns:**
   - Different values based on conditions
   - Can have multiple return statements

**Return Types:**
- Python is dynamically typed - no explicit return type required
- **Type Hints (Python 3.5+)**: Optional annotations for documentation
- Use typing module for complex types: List[int], Dict[str, int], Optional[str]
- Tools like mypy can check types statically

**Best Practices:**
- Return early for error cases (guard clauses)
- Be consistent with return types
- Document return values in docstring
- Use type hints for better IDE support and documentation`,
					CodeExamples: `# Single return value
def square(x):
    return x * x

result = square(5)  # 25

# Multiple returns (actually returns tuple)
def divide(a, b):
    if b == 0:
        return None, None  # Error case
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide(10, 3)
print(f"10 / 3 = {q} remainder {r}")  # 10 / 3 = 3 remainder 1

# Unpacking multiple returns
def get_name():
    return "Alice", "Smith"

first, last = get_name()
print(first, last)  # Alice Smith

# Or keep as tuple
name = get_name()
print(name)  # ('Alice', 'Smith')

# Early returns (guard clauses)
def process_user(user):
    if user is None:
        return None  # Early exit
    if not user.is_active:
        return None  # Early exit
    if user.is_banned:
        return None  # Early exit
    
    # Process active, non-banned user
    return user.process()

# Conditional returns
def get_status(score):
    if score >= 90:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Pass"
    else:
        return "Fail"

# No return statement (returns None)
def do_something():
    print("Doing something")
    # No return statement - implicitly returns None

result = do_something()  # Prints: Doing something
print(result)  # None

# Explicit None return
def find_item(items, target):
    for item in items:
        if item == target:
            return item
    return None  # Explicit None return

# Type hints for return values
from typing import List, Dict, Optional, Tuple

def get_user_data(user_id: int) -> Optional[Dict[str, str]]:
    """Get user data by ID. Returns None if not found."""
    if user_id > 0:
        return {"id": str(user_id), "name": "Alice"}
    return None

def process_numbers(numbers: List[int]) -> Tuple[int, float]:
    """Process numbers and return sum and average."""
    total = sum(numbers)
    average = total / len(numbers) if numbers else 0.0
    return total, average

total, avg = process_numbers([1, 2, 3, 4, 5])
print(f"Total: {total}, Average: {avg}")  # Total: 15, Average: 3.0`,
				},
				{
					Title: "Lambda Functions",
					Content: `**Lambda Functions (Anonymous Functions):**

**What are Lambdas:**
- **Anonymous functions**: Functions without a name
- **Defined with lambda keyword**: Shorter syntax than def
- **Single expression only**: Can only contain one expression, not statements
- **First-class objects**: Can be assigned, passed, returned like regular functions

**Syntax:**
lambda parameters: expression

**When to Use Lambdas:**
- **Short, simple operations**: One-liner functions
- **As arguments**: Pass to higher-order functions (map, filter, sorted)
- **Temporary functions**: Don't need to name them
- **Functional programming style**: When using functional patterns

**When NOT to Use Lambdas:**
- **Complex logic**: Use regular function instead
- **Multiple statements**: Lambdas can't contain statements
- **Need docstrings**: Lambdas can't have docstrings
- **Debugging**: Harder to debug (no name in stack traces)

**Common Use Cases:**
- **map()**: Transform each element
- **filter()**: Select elements based on condition
- **sorted()**: Custom sorting key
- **reduce()**: Accumulate values
- **Event handlers**: Short callback functions

**Limitations:**
- Cannot contain statements (print, return, pass, if, for, etc.)
- Single expression only
- No docstrings
- Less readable for complex logic
- Harder to test and debug`,
					CodeExamples: `# Basic lambda
square = lambda x: x * x
print(square(5))  # 25

# Equivalent regular function
def square_func(x):
    return x * x

# Lambda with multiple parameters
add = lambda a, b: a + b
print(add(3, 4))  # 7

# Lambda with no parameters
get_pi = lambda: 3.14159
print(get_pi())  # 3.14159

# With map() - transform each element
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# With filter() - select elements
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# With sorted() - custom sorting
people = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
sorted_by_age = sorted(people, key=lambda p: p[1])
print(sorted_by_age)  # [('Bob', 25), ('Alice', 30), ('Charlie', 35)]

# Lambda in list comprehension (usually better to use comprehension)
squares = [(lambda x: x**2)(x) for x in range(5)]
# But this is better:
squares = [x**2 for x in range(5)]

# Lambda with default arguments
multiply = lambda x, y=2: x * y
print(multiply(5))    # 10 (uses default y=2)
print(multiply(5, 3))  # 15

# Lambda with conditional expression
max_value = lambda a, b: a if a > b else b
print(max_value(10, 5))  # 10

# Lambda in higher-order function
def apply_operation(operation, a, b):
    return operation(a, b)

result = apply_operation(lambda x, y: x + y, 5, 3)
print(result)  # 8

# Lambda vs regular function - when to use which
# Use lambda for simple, one-line operations
numbers = [1, 2, 3, 4]
doubled = list(map(lambda x: x * 2, numbers))

# Use regular function for complex logic
def process_number(x):
    """Process number with complex logic."""
    if x < 0:
        return 0
    result = x * 2
    if result > 10:
        result = 10
    return result

processed = [process_number(x) for x in numbers]`,
				},
				{
					Title: "Scope and Namespaces",
					Content: `**Understanding Scope in Python:**

**Scope Definition:**
- **Scope**: Region where a variable is accessible
- **Namespace**: Mapping from names to objects
- Python uses **function scope**, not block scope (unlike C/Java)

**Scope Levels (LEGB Rule):**
1. **Local (L)**: Inside current function
2. **Enclosing (E)**: In enclosing (non-local) functions
3. **Global (G)**: At module level
4. **Built-in (B)**: Python built-in names (print, len, etc.)

**Variable Lookup Order:**
Python searches for names in this order:
1. Local scope (current function)
2. Enclosing scopes (outer functions)
3. Global scope (module level)
4. Built-in scope (Python built-ins)

**Key Rules:**
- **Reading**: Can read variables from outer scopes
- **Assignment**: Creates local variable unless declared otherwise
- **global keyword**: Declares variable as global (modify global)
- **nonlocal keyword**: Declares variable as nonlocal (modify enclosing)

**Common Pitfalls:**
- **UnboundLocalError**: Trying to modify global without declaring it
- **Shadowing**: Local variable hides global with same name
- **Mutable vs Immutable**: Modifying mutable objects doesn't require global

**Best Practices:**
- Avoid global variables when possible
- Use function parameters instead of globals
- Be explicit with global/nonlocal when needed
- Use descriptive names to avoid shadowing`,
					CodeExamples: `# Global variable
x = 10  # Global scope

def func():
    # Local variable
    y = 20  # Local scope
    print(x)  # Can read global (10)
    print(y)  # Local (20)

func()

# Reading vs Writing - Important distinction!
count = 0  # Global

def increment():
    # This would cause UnboundLocalError:
    # count += 1  # ERROR! Tries to create local 'count'
    
    # Solution: declare as global
    global count
    count += 1  # Now modifies global

increment()
print(count)  # 1

# Modifying mutable objects (no global needed)
items = []  # Global list

def add_item(item):
    items.append(item)  # OK! Modifying mutable object
    # No global needed because we're not reassigning

add_item("apple")
print(items)  # ['apple']

# But reassignment requires global
def reset_items():
    # items = []  # ERROR! Would create local 'items'
    global items
    items = []  # Now modifies global

reset_items()
print(items)  # []

# Enclosing scope (nonlocal)
def outer():
    x = 10  # Enclosing scope
    
    def inner():
        # Can read enclosing variable
        print(x)  # 10
    
    inner()

outer()

# Modifying enclosing scope
def outer():
    x = 10
    
    def inner():
        # x = 20  # ERROR! Creates local 'x', doesn't modify enclosing
        nonlocal x  # Declare as nonlocal
        x = 20  # Now modifies enclosing
    
    inner()
    print(x)  # 20

outer()

# Nested scopes
x = "global"

def outer():
    x = "outer"
    
    def inner():
        x = "inner"
        print(x)  # inner (local)
    
    inner()
    print(x)  # outer (outer's local)

outer()
print(x)  # global

# Built-in scope
def func():
    # Can use built-in names
    length = len([1, 2, 3])
    print(length)  # 3
    
    # But don't shadow them!
    # len = 5  # BAD! Shadows built-in len()

func()

# Scope resolution example
x = "global"

def outer():
    x = "outer"
    
    def inner():
        # LEGB lookup: Local → Enclosing → Global → Built-in
        print(x)  # Finds "outer" in enclosing scope
    
    inner()

outer()  # Prints: outer`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
