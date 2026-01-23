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
					Content: `**Understanding Python Modules:**

**What are Modules:**
- Files containing Python code (.py files)
- Can be imported and used in other files
- Organize code into reusable units
- Provide namespace isolation
- Enable code organization and reuse

**Module Benefits:**
- **Code Organization**: Group related functionality
- **Reusability**: Write once, use many times
- **Namespace Management**: Avoid naming conflicts
- **Maintainability**: Easier to maintain and test
- **Modularity**: Break large programs into smaller pieces

**Import Statement Types:**

**1. Import Entire Module:**
- Syntax: import module_name
- Access: module_name.function_name
- Namespace preserved (prevents conflicts)
- Most explicit and safe

**2. Import Specific Names:**
- Syntax: from module import name1, name2
- Access: name1 directly (no module prefix)
- Brings names into current namespace
- Can cause name conflicts

**3. Import with Alias:**
- Syntax: import module as alias
- Access: alias.function_name
- Useful for long module names
- Avoids naming conflicts

**4. Import All (Not Recommended):**
- Syntax: from module import *
- Access: All names directly
- Pollutes namespace
- Makes code harder to read
- Can cause unexpected conflicts

**Import Best Practices:**

**1. Import Order (PEP 8):**
- Standard library imports
- Related third-party imports
- Local application/library imports
- Separate groups with blank line

**2. Import Style:**
- One import per line (preferred)
- Group related imports
- Use absolute imports (preferred)
- Use relative imports within packages (from . import module)

**3. Avoid:**
- Circular imports (A imports B, B imports A)
- Import * (pollutes namespace)
- Unused imports (remove them)
- Importing inside functions (unless needed for performance)

**4. Conditional Imports:**
- Use try/except for optional dependencies
- Provide fallback behavior
- Document optional dependencies

**Standard Library Modules:**
- Built-in modules included with Python
- No installation needed
- Comprehensive functionality
- Well-documented and tested
- Examples: os, sys, math, datetime, json, collections, itertools, functools

**Common Import Patterns:**
- **Standard library first**: import os, sys
- **Third-party second**: import requests, flask
- **Local imports last**: from mypackage import mymodule
- **Group related**: import os, sys together
- **Use aliases**: import numpy as np, pandas as pd

**Import Errors:**
- **ModuleNotFoundError**: Module not found in search path
- **ImportError**: Error during import (rare)
- **SyntaxError**: Syntax error in imported module
- **AttributeError**: Attribute not found in module

**Module Search Path:**
Python searches for modules in this order:
1. Current directory
2. Directories in PYTHONPATH environment variable
3. Standard library directories
4. Site-packages directory (third-party packages)
5. Any .pth files in site-packages

**Checking Module Search Path:**
- import sys; print(sys.path)
- Shows all directories Python searches
- Can modify sys.path programmatically (not recommended)`,
					CodeExamples: `# Import entire module (recommended for clarity)
import math
print(math.pi)           # 3.14159...
print(math.sqrt(16))     # 4.0
print(math.cos(0))       # 1.0

# Import specific names
from math import pi, sqrt, cos
print(pi)                # 3.14159... (no math. prefix)
print(sqrt(16))          # 4.0
print(cos(0))            # 1.0

# Import with alias (useful for long names)
import datetime as dt
now = dt.datetime.now()
print(now)

# Common aliases
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all (NOT RECOMMENDED - pollutes namespace)
from math import *
print(pi)  # Works, but where did pi come from? Unclear!

# Multiple imports (one per line preferred)
import os
import sys
import json

# Grouped imports (PEP 8 style)
# Standard library
import os
import sys
import json

# Third-party
import requests
import numpy as np

# Local
from mypackage import mymodule

# Conditional import (for optional dependencies)
try:
    import optional_module
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False
    optional_module = None

# Use conditional import
if HAS_OPTIONAL:
    optional_module.do_something()

# Import from submodules
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from itertools import chain, combinations

# Relative imports (within package)
# In mypackage/subpackage/module.py:
from . import sibling_module      # Same level
from .. import parent_module      # Parent level
from ..sibling import other_module  # Sibling package

# Import with error handling
try:
    import expensive_module
except ImportError as e:
    print(f"Could not import expensive_module: {e}")
    # Fallback behavior
    expensive_module = None

# Checking if module is available
import importlib.util

def module_exists(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None

if module_exists("numpy"):
    import numpy as np
    print("NumPy available")
else:
    print("NumPy not installed")

# Import and inspect module
import math
print(dir(math))              # List all names in module
print(math.__file__)          # Path to module file
print(math.__name__)          # Module name
print(help(math.sqrt))        # Help for function

# Dynamic import
module_name = "math"
math_module = __import__(module_name)
print(math_module.pi)

# Or use importlib (preferred)
import importlib
math_module = importlib.import_module("math")
print(math_module.pi)

# Import with __all__ (controls import *)
# In mymodule.py:
__all__ = ["public_function", "PublicClass"]  # Only these exported

def public_function():
    pass

def _private_function():  # Not in __all__
    pass

# When importing: from mymodule import *
# Only public_function and PublicClass are imported`,
				},
				{
					Title: "Creating Modules",
					Content: `**Creating Your Own Modules:**

**Module Basics:**
- Any .py file is a module
- Module name is filename (without .py extension)
- Can be imported by other files
- Provides namespace for code organization

**Module Structure (Best Practices):**

**1. Module Docstring:**
- First statement should be docstring
- Describes what the module does
- Accessible via __doc__ attribute

**2. Imports:**
- All imports at the top
- Follow PEP 8 import order
- Standard library → Third-party → Local

**3. Module-Level Constants:**
- UPPER_CASE naming convention
- Constants that don't change
- Accessible via module.CONSTANT

**4. Functions and Classes:**
- Main functionality of module
- Well-documented with docstrings
- Follow naming conventions

**5. if __name__ == "__main__":**
- Code that runs when executed directly
- Not executed when imported
- Useful for testing and demos

**Module Naming:**
- Use lowercase with underscores: my_module.py
- Avoid hyphens, spaces, special characters
- Keep names descriptive but concise
- Don't conflict with standard library names

**Module-Level Code:**
- Executes when module is imported
- Should be minimal (avoid side effects)
- Can initialize module state
- Can register things globally

**The __name__ Variable:**
- **__main__**: When script is run directly
- **module_name**: When imported as module
- Allows same file to be script or module
- Essential for testing and demos

**Module Attributes:**
- **__name__**: Module name
- **__file__**: Path to module file
- **__doc__**: Module docstring
- **__package__**: Package name (None for top-level)
- **__all__**: List of exported names (for import *)

**Best Practices:**
- **Keep modules focused**: One module, one purpose
- **Document everything**: Docstrings for module, functions, classes
- **Avoid side effects**: Don't execute code on import unless necessary
- **Use __all__**: Control what gets exported
- **Test modules**: Include test code in __main__ block
- **Handle errors**: Don't let import errors crash program

**Common Patterns:**

**Pattern 1: Configuration Module**
- Store configuration constants
- Can be imported by other modules
- Easy to modify settings

**Pattern 2: Utility Module**
- Collection of utility functions
- No shared state
- Pure functions preferred

**Pattern 3: Class Module**
- Define classes for reuse
- Can be instantiated by other modules
- Encapsulates related functionality

**Pattern 4: Initialization Module**
- Sets up environment
- Registers handlers
- Initializes global state`,
					CodeExamples: `# mymodule.py - Well-structured module
"""
My Module - A simple example module.

This module provides utility functions for string manipulation
and a Calculator class for basic arithmetic operations.

Author: Your Name
Version: 1.0.0
"""

# Module-level constants
MODULE_VERSION = "1.0.0"
DEFAULT_GREETING = "Hello"

# Control what gets exported with import *
__all__ = ["greet", "Calculator", "MODULE_VERSION"]

# Imports (standard library first)
import os
import sys
from datetime import datetime

# Third-party imports (if any)
# import requests

# Local imports (if any)
# from .submodule import helper

# Module-level function
def greet(name, greeting=DEFAULT_GREETING):
    """Greet someone by name.
    
    Args:
        name: Name to greet
        greeting: Greeting message (default: "Hello")
    
    Returns:
        Greeting message string
    
    Example:
        >>> greet("Alice")
        'Hello, Alice!'
    """
    return f"{greeting}, {name}!"

# Module-level class
class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        """Initialize calculator with empty history."""
        self.history = []
    
    def add(self, a, b):
        """Add two numbers.
        
        Args:
            a: First number
            b: Second number
        
        Returns:
            Sum of a and b
        """
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history

# Module-level code (executes on import - use sparingly)
# This runs every time module is imported
_initialized = False

def _initialize():
    """Private function to initialize module."""
    global _initialized
    if not _initialized:
        # Setup code here
        _initialized = True

_initialize()  # Call on import

# Executable code (only runs when script is executed directly)
if __name__ == "__main__":
    # Test code or demo
    print(f"MyModule version {MODULE_VERSION}")
    print(greet("World"))
    
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.get_history())
    
    # Module information
    print(f"Module name: {__name__}")
    print(f"Module file: {__file__}")
    print(f"Module doc: {__doc__}")

# Using the module in another file:
# import mymodule
# print(mymodule.greet("Alice"))
# calc = mymodule.Calculator()
# print(mymodule.MODULE_VERSION)

# Configuration module pattern
# config.py
DATABASE_URL = "postgresql://localhost/mydb"
API_KEY = "secret-key"
DEBUG = False
MAX_RETRIES = 3

# Using config:
# from config import DATABASE_URL, API_KEY

# Utility module pattern
# utils.py
"""Utility functions for common operations."""

def format_currency(amount, currency="USD"):
    """Format amount as currency."""
    return f"{currency} {amount:,.2f}"

def validate_email(email):
    """Validate email format."""
    return "@" in email and "." in email.split("@")[1]

# Class module pattern
# models.py
"""Data models for the application."""

class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

# Initialization module pattern
# __init__.py or setup.py
"""Initialize application state."""

_registered_handlers = []

def register_handler(handler):
    """Register an event handler."""
    _registered_handlers.append(handler)

def get_handlers():
    """Get all registered handlers."""
    return _registered_handlers.copy()`,
				},
				{
					Title: "Packages",
					Content: `**Understanding Python Packages:**

**What are Packages:**
- Directories containing Python modules
- Organize related modules together
- Create hierarchical namespace
- Enable better code organization
- Can contain subpackages (nested packages)

**Package Requirements:**
- **__init__.py file**: Makes directory a package
  - Python 3.3+: Can be empty (namespace packages)
  - Python < 3.3: Must exist (can be empty)
  - Often contains package initialization code
  - Can control what gets imported

**Package Structure:**
mypackage/
    __init__.py          # Package initialization
    module1.py           # Module in package
    module2.py           # Another module
    subpackage/          # Subpackage
        __init__.py      # Subpackage init
        module3.py       # Module in subpackage
    tests/               # Test package (optional)
        __init__.py
        test_module1.py

**Package vs Module:**
- **Module**: Single .py file
- **Package**: Directory with __init__.py containing modules
- **Subpackage**: Package inside another package

**__init__.py Patterns:**

**1. Empty __init__.py:**
- Simplest approach
- Package exists but doesn't export anything
- Import modules explicitly

**2. Import Control:**
- Import commonly used names
- Make package API cleaner
- Control what's available

**3. Package Initialization:**
- Set up package state
- Register things globally
- Initialize resources

**4. Version and Metadata:**
- Store package version
- Package-level constants
- Package information

**Import Patterns:**

**Absolute Imports (Recommended):**
- Full path from package root
- Clear and explicit
- Works from anywhere
- Example: from mypackage.subpackage import module

**Relative Imports:**
- Relative to current package
- Use . for current, .. for parent
- Only work within packages
- Example: from . import sibling_module

**Import Best Practices:**
- **Use absolute imports**: Clearer and more reliable
- **Import from __init__.py**: Cleaner API
- **Avoid circular imports**: Design packages carefully
- **Document package structure**: README or docstrings
- **Use __all__**: Control package exports

**Package Organization:**

**Flat Structure:**
- All modules at package root
- Good for small packages
- Simple imports

**Nested Structure:**
- Organize by functionality
- Subpackages for features
- Better for large packages

**Common Package Patterns:**

**Pattern 1: Library Package**
- Collection of related modules
- Clear public API
- Well-documented

**Pattern 2: Application Package**
- Organized by feature/component
- Clear separation of concerns
- Easy to navigate

**Pattern 3: Plugin Package**
- Extensible architecture
- Plugin discovery
- Dynamic loading

**Namespace Packages (Python 3.3+):**
- Packages without __init__.py
- Can span multiple directories
- Useful for plugin systems
- More flexible structure`,
					CodeExamples: `# Package structure example:
# mypackage/
#     __init__.py
#     math_utils.py
#     string_utils.py
#     data/
#         __init__.py
#         processors.py
#         validators.py

# __init__.py - Empty (simple)
# (file exists but is empty)

# __init__.py - Import control (common pattern)
# mypackage/__init__.py
"""
MyPackage - A collection of utility modules.

This package provides mathematical and string utilities.
"""

from .math_utils import add, multiply, divide
from .string_utils import capitalize, reverse

__version__ = "1.0.0"
__all__ = ["add", "multiply", "divide", "capitalize", "reverse"]

# math_utils.py
"""Mathematical utility functions."""

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# string_utils.py
"""String utility functions."""

def capitalize(text):
    return text.capitalize()

def reverse(text):
    return text[::-1]

# Importing from package

# Method 1: Import entire module
import mypackage.math_utils
result = mypackage.math_utils.add(1, 2)

# Method 2: Import module from package
from mypackage import math_utils
result = math_utils.add(1, 2)

# Method 3: Import specific function
from mypackage.math_utils import add
result = add(1, 2)

# Method 4: Import from __init__.py (cleaner API)
from mypackage import add, multiply
result = add(1, 2)

# Subpackage imports
from mypackage.data import processors
from mypackage.data.processors import process_data

# Or if data/__init__.py exports:
from mypackage.data import process_data

# Relative imports (within package)
# In mypackage/data/processors.py:
from ..math_utils import add  # Import from parent package
from .validators import validate  # Import from same package

# Package with initialization
# mypackage/__init__.py
_registry = {}

def register(name, func):
    """Register a function."""
    _registry[name] = func

def get(name):
    """Get registered function."""
    return _registry.get(name)

# Initialize package
register("default", lambda x: x)

# Using the package
import mypackage
mypackage.register("custom", my_function)
func = mypackage.get("custom")

# Namespace package (Python 3.3+)
# No __init__.py needed
# Can have package spread across directories
# plugin1/
#     mypackage/
#         module1.py
# plugin2/
#     mypackage/
#         module2.py
# Both contribute to mypackage namespace

# Package with version and metadata
# mypackage/__init__.py
__version__ = "1.2.3"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = ["version", "author"]

def version():
    """Get package version."""
    return __version__

def author():
    """Get package author."""
    return __author__

# Package structure best practices
# myproject/
#     README.md
#     setup.py
#     requirements.txt
#     mypackage/
#         __init__.py
#         core/
#             __init__.py
#             models.py
#             views.py
#         utils/
#             __init__.py
#             helpers.py
#         tests/
#             __init__.py
#             test_core.py
#             test_utils.py

# Importing from nested structure
from mypackage.core import models
from mypackage.utils.helpers import helper_function

# Package discovery pattern
import pkgutil
import mypackage

# Find all modules in package
for importer, modname, ispkg in pkgutil.walk_packages(
    mypackage.__path__, mypackage.__name__ + "."):
    print(f"Found: {modname} (package: {ispkg})")`,
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
					Content: `**File Operations in Python:**

**Opening Files:**
- Use **open()** function to open files
- Returns file object (file handle)
- Must close file when done (or use context manager)
- File operations are I/O bound (can be slow)

**File Modes:**

**Text Modes:**
- **'r'**: Read mode (default for text)
- **'w'**: Write mode (overwrites existing file)
- **'a'**: Append mode (adds to end of file)
- **'x'**: Exclusive creation (fails if file exists)
- **'r+'**: Read and write (file must exist)
- **'w+'**: Read and write (creates/overwrites)
- **'a+'**: Read and append (creates if doesn't exist)

**Binary Modes:**
- **'rb'**: Read binary
- **'wb'**: Write binary
- **'ab'**: Append binary
- **'rb+'**, **'wb+'**, **'ab+'**: Binary read/write modes

**Mode Combinations:**
- **'t'**: Text mode (default, can omit)
- **'b'**: Binary mode
- **'+'**: Read and write access

**Reading Methods:**

**1. read(size):**
- Read entire file or specified bytes
- Returns string (text) or bytes (binary)
- Can specify size to read in chunks
- Loads entire file into memory

**2. readline():**
- Read single line
- Includes newline character
- Returns empty string at EOF
- Memory efficient for large files

**3. readlines():**
- Read all lines into list
- Each line includes newline
- Loads entire file into memory
- Use for small files

**4. Iteration:**
- File objects are iterable
- Most memory efficient
- Best for large files
- Automatically handles newlines

**Writing Methods:**

**1. write(string):**
- Write string to file
- Returns number of characters written
- Doesn't add newline automatically
- Use \\n for newlines

**2. writelines(sequence):**
- Write sequence of strings
- Doesn't add newlines between items
- More efficient than multiple write() calls

**3. flush():**
- Force write buffer to disk
- Usually automatic, but can force
- Useful for real-time logging

**File Object Attributes:**
- **name**: File name/path
- **mode**: File mode
- **closed**: Whether file is closed
- **encoding**: Text encoding (text mode)

**Best Practices:**
- **Always use context manager**: with statement ensures file is closed
- **Handle encoding**: Specify encoding for text files (UTF-8 recommended)
- **Use binary mode**: For non-text files (images, executables)
- **Handle errors**: File operations can fail (permissions, missing file)
- **Check file exists**: Before reading (or handle FileNotFoundError)
- **Use pathlib**: Modern path handling (Python 3.4+)

**Common File Operations:**

**Reading Entire File:**
- Use read() for small files
- Use iteration for large files
- Consider memory usage

**Processing Line by Line:**
- Use iteration (most efficient)
- Use readline() for more control
- Strip newlines with strip()

**Writing Multiple Lines:**
- Use writelines() with list
- Add \\n manually
- Or use print() with file parameter

**Working with Different File Formats:**
- **Text files**: Use text mode, handle encoding
- **CSV files**: Use csv module
- **JSON files**: Use json module
- **Binary files**: Use binary mode
- **Large files**: Process in chunks`,
					CodeExamples: `# Reading entire file (small files)
with open("data.txt", "r") as file:
    content = file.read()  # Reads entire file
    print(content)

# Reading with encoding (important for non-ASCII)
with open("data.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Reading line by line (memory efficient)
with open("large_file.txt", "r") as file:
    for line in file:
        process(line.strip())  # strip() removes \\n

# Reading specific number of characters
with open("data.txt", "r") as file:
    chunk = file.read(100)  # Read first 100 characters
    next_chunk = file.read(100)  # Read next 100

# Reading all lines into list
with open("data.txt", "r") as file:
    lines = file.readlines()  # List of lines (includes \\n)

# Reading without newlines
with open("data.txt", "r") as file:
    lines = [line.strip() for line in file]

# Writing to file
with open("output.txt", "w") as file:
    file.write("Hello, World!\\n")
    file.write("Second line\\n")

# Writing multiple lines
lines = ["Line 1\\n", "Line 2\\n", "Line 3\\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)

# Or using print (adds newline automatically)
with open("output.txt", "w") as file:
    print("Line 1", file=file)
    print("Line 2", file=file)

# Appending to file
with open("log.txt", "a") as file:
    file.write(f"{datetime.now()}: Event occurred\\n")

# Reading and writing (r+ mode)
with open("data.txt", "r+") as file:
    content = file.read()
    file.write("\\nAppended line")

# Binary file operations
with open("image.jpg", "rb") as file:
    image_data = file.read()  # Read as bytes

with open("copy.jpg", "wb") as file:
    file.write(image_data)  # Write bytes

# Processing large files in chunks
def process_large_file(filename, chunk_size=1024):
    """Process large file in chunks."""
    with open(filename, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            process_chunk(chunk)

# Reading CSV-like data
with open("data.csv", "r") as file:
    for line in file:
        fields = line.strip().split(",")
        process_fields(fields)

# Writing formatted data
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]

with open("output.txt", "w") as file:
    for record in data:
        file.write(f"{record['name']}: {record['age']}\\n")

# File object attributes
with open("data.txt", "r") as file:
    print(file.name)      # File path
    print(file.mode)     # 'r'
    print(file.encoding) # 'utf-8' (default)
    print(file.closed)   # False
print(file.closed)       # True (after context exits)

# Error handling
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
except IOError as e:
    print(f"I/O error: {e}")

# Check if file exists before reading
import os
if os.path.exists("data.txt"):
    with open("data.txt", "r") as file:
        content = file.read()
else:
    print("File does not exist")

# Reading with error handling
def safe_read_file(filename):
    """Safely read file with error handling."""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        # Try different encoding
        with open(filename, "r", encoding="latin-1") as file:
            return file.read()`,
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
				{
					Title: "Path Handling with pathlib",
					Content: `**pathlib Module (Python 3.4+):**

**What is pathlib:**
- Object-oriented filesystem paths
- Modern alternative to os.path
- Cross-platform path handling
- More intuitive and Pythonic
- Recommended for new code

**Why Use pathlib:**
- **Cross-platform**: Handles Windows/Unix differences automatically
- **Object-oriented**: Paths are objects with methods
- **More readable**: path / "subdir" / "file.txt" vs os.path.join()
- **Type safety**: Path objects vs strings
- **Rich API**: Many convenient methods

**Path Types:**
- **Path**: Concrete path (works on current OS)
- **PurePath**: Abstract path (no filesystem access)
- **PosixPath**: Unix-style paths
- **WindowsPath**: Windows-style paths

**Creating Paths:**

**1. From String:**
- Path("file.txt")
- Path("/absolute/path")
- Path("relative/path")

**2. Current Directory:**
- Path.cwd() - current working directory
- Path.home() - user home directory

**3. Joining Paths:**
- Use / operator: Path("dir") / "file.txt"
- More readable than os.path.join()

**Path Operations:**

**1. Path Components:**
- **name**: Filename with extension
- **stem**: Filename without extension
- **suffix**: File extension
- **suffixes**: All extensions (for .tar.gz)
- **parent**: Parent directory
- **parts**: Tuple of path components
- **anchor**: Drive/root

**2. Path Queries:**
- **exists()**: Check if path exists
- **is_file()**: Check if is file
- **is_dir()**: Check if is directory
- **is_symlink()**: Check if is symlink
- **stat()**: Get file stats

**3. Path Modifications:**
- **with_name()**: Change filename
- **with_suffix()**: Change extension
- **with_stem()**: Change stem (name without extension)
- **joinpath()**: Join paths
- **resolve()**: Resolve to absolute path

**4. Directory Operations:**
- **mkdir()**: Create directory
- **rmdir()**: Remove directory
- **iterdir()**: Iterate directory contents
- **glob()**: Find files matching pattern
- **rglob()**: Recursive glob

**5. File Operations:**
- **read_text()**: Read file as text
- **write_text()**: Write text to file
- **read_bytes()**: Read file as bytes
- **write_bytes()**: Write bytes to file
- **open()**: Open file (works with context manager)

**Best Practices:**
- **Use Path objects**: Instead of strings for paths
- **Use / operator**: For joining paths
- **Use read_text/write_text**: For simple text files
- **Handle PathNotFoundError**: When paths don't exist
- **Use resolve()**: To get absolute paths
- **Use glob()**: For finding files

**Comparison with os.path:**
- **os.path.join()** → Path / "subdir" / "file.txt"
- **os.path.exists()** → Path.exists()
- **os.path.isdir()** → Path.is_dir()
- **os.path.isfile()** → Path.is_file()
- **os.path.basename()** → Path.name
- **os.path.dirname()** → Path.parent

**Common Patterns:**
- **Find files**: Use glob() or rglob()
- **Process directory**: Use iterdir()
- **Read config**: Use read_text()
- **Create directories**: Use mkdir(parents=True)
- **Path validation**: Use exists(), is_file(), is_dir()`,
					CodeExamples: `from pathlib import Path

# Creating paths
file_path = Path("file.txt")
dir_path = Path("mydir")
abs_path = Path("/absolute/path/to/file.txt")

# Current and home directories
current = Path.cwd()      # Current working directory
home = Path.home()        # User home directory

# Joining paths (use / operator)
data_dir = Path("data")
file_path = data_dir / "subdir" / "file.txt"
# More readable than: os.path.join("data", "subdir", "file.txt")

# Path components
path = Path("/home/user/data/file.txt")
print(path.name)      # "file.txt"
print(path.stem)      # "file"
print(path.suffix)    # ".txt"
print(path.parent)    # Path("/home/user/data")
print(path.parts)     # ("/", "home", "user", "data", "file.txt")
print(path.anchor)    # "/" (root)

# Multiple extensions
archive = Path("archive.tar.gz")
print(archive.suffix)    # ".gz"
print(archive.suffixes) # [".tar", ".gz"]
print(archive.stem)     # "archive.tar"

# Path queries
path = Path("file.txt")
print(path.exists())     # True if file exists
print(path.is_file())    # True if is file
print(path.is_dir())     # True if is directory

# Check before operations
if path.exists() and path.is_file():
    content = path.read_text()

# Path modifications
path = Path("old_name.txt")
new_path = path.with_name("new_name.txt")
new_ext = path.with_suffix(".csv")
new_stem = path.with_stem("different")

# Resolve to absolute path
relative = Path("file.txt")
absolute = relative.resolve()  # Full absolute path

# Directory operations
# Create directory
data_dir = Path("data")
data_dir.mkdir()  # Create if doesn't exist
data_dir.mkdir(exist_ok=True)  # Don't error if exists

# Create nested directories
nested = Path("a/b/c")
nested.mkdir(parents=True, exist_ok=True)  # Create parents

# Remove directory (must be empty)
empty_dir = Path("empty")
empty_dir.rmdir()

# Iterate directory
for item in Path(".").iterdir():
    if item.is_file():
        print(f"File: {item.name}")
    elif item.is_dir():
        print(f"Directory: {item.name}")

# Find files with glob
# Find all .txt files
for txt_file in Path(".").glob("*.txt"):
    print(txt_file)

# Find recursively
for py_file in Path(".").rglob("*.py"):
    print(py_file)

# Find in subdirectory
for file in Path("data").glob("*.csv"):
    process_csv(file)

# File operations
# Read text file
content = Path("data.txt").read_text(encoding="utf-8")

# Write text file
Path("output.txt").write_text("Hello, World!", encoding="utf-8")

# Read bytes
image_data = Path("image.jpg").read_bytes()

# Write bytes
Path("copy.jpg").write_bytes(image_data)

# Open file (for more control)
with Path("data.txt").open("r") as file:
    content = file.read()

# Common patterns

# Pattern 1: Process all files in directory
def process_directory(directory):
    """Process all files in directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"{directory} is not a directory")
    
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            process_file(file_path)

# Pattern 2: Find and process specific files
def find_and_process(pattern):
    """Find files matching pattern and process them."""
    for file_path in Path(".").rglob(pattern):
        if file_path.is_file():
            process_file(file_path)

# Pattern 3: Create directory structure
def setup_project_structure():
    """Create project directory structure."""
    base = Path("myproject")
    base.mkdir(exist_ok=True)
    
    (base / "src").mkdir(exist_ok=True)
    (base / "tests").mkdir(exist_ok=True)
    (base / "docs").mkdir(exist_ok=True)
    
    # Create files
    (base / "README.md").write_text("# My Project\\n")
    (base / "requirements.txt").touch()

# Pattern 4: Read configuration file
def load_config(config_file="config.json"):
    """Load configuration from file."""
    config_path = Path(config_file)
    if config_path.exists():
        import json
        return json.loads(config_path.read_text())
    return {}

# Pattern 5: Backup file
def backup_file(file_path):
    """Create backup of file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    
    backup_path = path.with_suffix(f"{path.suffix}.bak")
    backup_path.write_bytes(path.read_bytes())
    return backup_path

# Pattern 6: Get file size
def get_file_size(file_path):
    """Get file size in bytes."""
    path = Path(file_path)
    if path.exists() and path.is_file():
        return path.stat().st_size
    return 0

# Pattern 7: List files by extension
def list_files_by_ext(directory, extension):
    """List all files with given extension."""
    dir_path = Path(directory)
    return [f for f in dir_path.iterdir() 
            if f.is_file() and f.suffix == extension]

# Pattern 8: Safe file writing
def safe_write(file_path, content):
    """Safely write file (creates directory if needed)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

# Comparison: os.path vs pathlib
import os

# Old way (os.path)
old_path = os.path.join("data", "subdir", "file.txt")
if os.path.exists(old_path) and os.path.isfile(old_path):
    with open(old_path, "r") as f:
        content = f.read()

# New way (pathlib)
new_path = Path("data") / "subdir" / "file.txt"
if new_path.exists() and new_path.is_file():
    content = new_path.read_text()

# More readable and Pythonic!`,
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
					Content: `**Understanding Decorators:**

**What are Decorators:**
- Functions that modify or enhance other functions
- Use @ syntax for convenience
- Common pattern in Python
- Functions are first-class objects (can be passed around)
- Enable code reuse and separation of concerns

**How Decorators Work:**
- Decorator is a function that takes a function as argument
- Returns a new function (usually wrapper)
- Wrapper function calls original function
- Can add behavior before/after function call
- Can modify arguments or return value

**Decorator Syntax:**
- **@decorator**: Applied to function below
- Equivalent to: func = decorator(func)
- Can stack multiple decorators
- Applied bottom to top

**Types of Decorators:**

**1. Function Decorators:**
- Simplest form
- Takes function, returns function
- Most common type

**2. Decorator Factories:**
- Returns decorator function
- Allows decorator to take arguments
- More flexible

**3. Class Decorators:**
- Classes that act as decorators
- Implement __call__ method
- Can maintain state

**4. Method Decorators:**
- Applied to methods
- Can access self
- Common: @property, @staticmethod, @classmethod

**Preserving Metadata:**
- Decorators can hide original function metadata
- Use **functools.wraps** to preserve
- Important for debugging and introspection
- Preserves __name__, __doc__, __module__, etc.

**Common Decorator Patterns:**

**1. Timing Decorator:**
- Measure execution time
- Useful for profiling
- Non-intrusive performance measurement

**2. Caching Decorator:**
- Cache function results
- Avoid recomputation
- Use functools.lru_cache

**3. Logging Decorator:**
- Log function calls
- Track arguments and results
- Debugging aid

**4. Authentication Decorator:**
- Check permissions before execution
- Common in web frameworks
- Enforce access control

**5. Retry Decorator:**
- Retry on failure
- Handle transient errors
- Configurable retry logic

**6. Validation Decorator:**
- Validate arguments
- Type checking
- Input sanitization

**7. Rate Limiting Decorator:**
- Limit function call frequency
- Prevent abuse
- Throttle requests

**Best Practices:**
- **Use functools.wraps**: Preserve function metadata
- **Document decorators**: Explain what they do
- **Keep decorators simple**: Don't overcomplicate
- **Test decorators**: Ensure they work correctly
- **Consider performance**: Decorators add overhead
- **Use class decorators**: When you need state

**Advanced Patterns:**
- **Decorator chaining**: Multiple decorators
- **Conditional decorators**: Apply based on conditions
- **Decorator with arguments**: Factory pattern
- **Class-based decorators**: Stateful decorators`,
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
# Output:
# Before function
# Hello!
# After function

# Equivalent to:
def say_hello():
    print("Hello!")
say_hello = my_decorator(say_hello)

# Decorator with functools.wraps (preserves metadata)
from functools import wraps

def timing_decorator(func):
    @wraps(func)  # Preserves func's metadata
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    import time
    time.sleep(1)
    return "Done"

# Function metadata preserved
print(slow_function.__name__)  # "slow_function" (not "wrapper")

# Decorator with arguments (factory pattern)
def repeat(times):
    """Decorator factory - returns a decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Prints 3 times

# Caching decorator
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    """Compute Fibonacci number (cached)."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# First call computes, subsequent calls use cache
print(fibonacci(30))  # Computed
print(fibonacci(30))  # From cache (instant)

# Logging decorator
def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

add(3, 5)
# Output:
# Calling add with args=(3, 5), kwargs={}
# add returned 8

# Authentication decorator
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user or not user.is_authenticated:
            raise PermissionError("Authentication required")
        return func(*args, **kwargs)
    return wrapper

@require_auth
def delete_user(user_id):
    # Only authenticated users can call this
    pass

# Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success"

# Validation decorator
def validate_types(**types):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate args
            for i, (arg, expected_type) in enumerate(zip(args, types.values())):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {i} must be {expected_type.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(a=int, b=int)
def multiply(a, b):
    return a * b

multiply(3, 5)  # OK
# multiply(3, "5")  # TypeError

# Rate limiting decorator
from time import time
from collections import defaultdict

call_times = defaultdict(list)

def rate_limit(max_calls=5, period=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            func_name = func.__name__
            
            # Remove old calls outside period
            call_times[func_name] = [
                t for t in call_times[func_name] 
                if now - t < period
            ]
            
            if len(call_times[func_name]) >= max_calls:
                raise RuntimeError(f"Rate limit exceeded for {func_name}")
            
            call_times[func_name].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=3, period=60)
def api_call():
    return "API response"

# Class-based decorator (stateful)
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()  # say_hello called 1 times
say_hello()  # say_hello called 2 times

# Multiple decorators (stacked)
@timing_decorator
@log_calls
@require_auth
def sensitive_operation():
    return "Result"

# Applied bottom to top:
# 1. require_auth wraps sensitive_operation
# 2. log_calls wraps result
# 3. timing_decorator wraps result

# Conditional decorator
def conditional_decorator(condition):
    def decorator(func):
        if condition:
            return timing_decorator(func)
        return func
    return decorator

DEBUG = True

@conditional_decorator(DEBUG)
def debug_function():
    pass

# Built-in decorators
class MyClass:
    @property
    def value(self):
        return self._value
    
    @staticmethod
    def static_method():
        return "Static"
    
    @classmethod
    def class_method(cls):
        return f"Class: {cls.__name__}"`,
				},
				{
					Title: "Generators",
					Content: `**Understanding Generators:**

**What are Generators:**
- Functions that yield values instead of returning them
- Lazy evaluation - values generated on demand
- Memory efficient for large sequences
- Use **yield** keyword instead of **return**
- Return generator objects (iterators)

**Why Use Generators:**
- **Memory Efficiency**: Don't store entire sequence in memory
- **Lazy Evaluation**: Compute values only when needed
- **Performance**: Faster for large datasets
- **Infinite Sequences**: Can represent infinite sequences
- **Pipeline Processing**: Chain generators together

**Generator Functions:**
- Defined like regular functions
- Use **yield** instead of **return**
- Can have multiple **yield** statements
- Execution pauses at **yield**, resumes on next call
- Returns generator object (not values directly)
- State is preserved between calls

**Generator Expressions:**
- Similar syntax to list comprehensions
- Use **()** instead of **[]**
- More memory efficient than list comprehensions
- Lazy evaluation
- Can be passed directly to functions

**Generator vs List Comprehension:**
- **List**: [x**2 for x in range(10)] - Creates list immediately
- **Generator**: (x**2 for x in range(10)) - Creates generator, lazy
- **Memory**: Generator uses O(1) memory, list uses O(n)
- **Speed**: List faster for small data, generator faster for large

**Generator Methods:**
- **next()**: Get next value (raises StopIteration when done)
- **send(value)**: Send value to generator (advanced)
- **throw(exception)**: Raise exception in generator
- **close()**: Close generator

**Common Patterns:**

**1. Infinite Sequences:**
- Generate infinite sequences
- Useful for streams, counters
- Can be limited with itertools.islice

**2. File Processing:**
- Process large files line by line
- Don't load entire file into memory
- Memory efficient for huge files

**3. Data Pipelines:**
- Chain generators together
- Each generator processes previous output
- Efficient data transformation

**4. Filtering and Mapping:**
- Filter and transform data lazily
- Combine with other generators
- Efficient for large datasets

**5. Stateful Generators:**
- Generators maintain state
- Useful for state machines
- Can be reset or reused

**Best Practices:**
- **Use for large data**: When memory is concern
- **Use for infinite sequences**: When sequence is unbounded
- **Don't convert to list**: Unless you need all values
- **Chain generators**: For efficient pipelines
- **Use itertools**: For common generator patterns
- **Handle StopIteration**: When using next() directly

**Generator vs Iterator:**
- **Generator**: Function with yield (simpler)
- **Iterator**: Class with __iter__ and __next__ (more control)
- **Generator**: Easier to write, less flexible
- **Iterator**: More flexible, more code

**Advanced Features:**
- **yield from**: Delegate to another generator
- **Generator.send()**: Send values to generator
- **Generator.throw()**: Raise exceptions in generator
- **Coroutines**: Generators used for async programming`,
					CodeExamples: `# Basic generator function
def countdown(n):
    """Count down from n to 1."""
    while n > 0:
        yield n
        n -= 1

# Use generator
for num in countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# Generator object
gen = countdown(5)
print(type(gen))  # <class 'generator'>

# Manual iteration
gen = countdown(3)
print(next(gen))  # 3
print(next(gen))  # 2
print(next(gen))  # 1
# print(next(gen))  # StopIteration

# Generator expression
squares = (x**2 for x in range(10))
print(list(squares))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Memory comparison
import sys

# List comprehension (stores all values)
list_comp = [x**2 for x in range(1000000)]
print(sys.getsizeof(list_comp))  # ~8MB

# Generator expression (stores nothing)
gen_expr = (x**2 for x in range(1000000))
print(sys.getsizeof(gen_expr))  # ~200 bytes (much smaller!)

# Process large file (memory efficient)
def read_large_file(filename):
    """Read file line by line without loading entire file."""
    with open(filename) as file:
        for line in file:
            yield line.strip()

# Process one line at a time
for line in read_large_file("huge.txt"):
    process(line)  # Only one line in memory at a time

# Infinite sequence
def fibonacci():
    """Generate Fibonacci numbers infinitely."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Use with itertools.islice to limit
from itertools import islice
first_10 = list(islice(fibonacci(), 10))
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Generator pipeline
def numbers():
    """Generate numbers."""
    for i in range(10):
        yield i

def squares(seq):
    """Square each number."""
    for n in seq:
        yield n ** 2

def evens(seq):
    """Filter even numbers."""
    for n in seq:
        if n % 2 == 0:
            yield n

# Chain generators
pipeline = evens(squares(numbers()))
print(list(pipeline))  # [0, 4, 16, 36, 64]

# Using yield from (delegation)
def chain_generators(*generators):
    """Chain multiple generators together."""
    for gen in generators:
        yield from gen

gen1 = (x for x in range(3))
gen2 = (x for x in range(3, 6))
chained = chain_generators(gen1, gen2)
print(list(chained))  # [0, 1, 2, 3, 4, 5]

# Stateful generator
def counter(start=0, step=1):
    """Counter generator with state."""
    current = start
    while True:
        yield current
        current += step

count = counter(10, 2)
print(next(count))  # 10
print(next(count))  # 12
print(next(count))  # 14

# Generator with send() (coroutine-like)
def accumulator():
    """Accumulator that can receive values."""
    total = 0
    while True:
        value = yield total
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)  # Initialize generator
print(acc.send(10))  # 10
print(acc.send(20))  # 30
print(acc.send(5))   # 35

# Filtering with generator
def filter_evens(numbers):
    """Filter even numbers."""
    for n in numbers:
        if n % 2 == 0:
            yield n

numbers = range(10)
evens = filter_evens(numbers)
print(list(evens))  # [0, 2, 4, 6, 8]

# Mapping with generator
def map_squares(numbers):
    """Square each number."""
    for n in numbers:
        yield n ** 2

numbers = range(5)
squares = map_squares(numbers)
print(list(squares))  # [0, 1, 4, 9, 16]

# Batch processing
def batch(iterable, batch_size):
    """Process items in batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

data = range(10)
for batch in batch(data, 3):
    print(batch)  # [0, 1, 2], [3, 4, 5], [6, 7, 8], [9]

# Reading CSV efficiently
def read_csv(filename):
    """Read CSV file line by line."""
    with open(filename) as file:
        header = next(file).strip().split(",")
        for line in file:
            values = line.strip().split(",")
            yield dict(zip(header, values))

# Process one row at a time
for row in read_csv("large.csv"):
    process_row(row)

# Generator with error handling
def safe_read_file(filename):
    """Safely read file with error handling."""
    try:
        with open(filename) as file:
            for line in file:
                yield line.strip()
    except FileNotFoundError:
        yield f"Error: File {filename} not found"
    except Exception as e:
        yield f"Error: {e}"

# Generator for pagination
def paginate(items, page_size):
    """Paginate items."""
    page = []
    for item in items:
        page.append(item)
        if len(page) == page_size:
            yield page
            page = []
    if page:
        yield page

items = range(20)
for page in paginate(items, 5):
    print(page)  # [0,1,2,3,4], [5,6,7,8,9], ...

# Combining generators
def combine(*generators):
    """Combine multiple generators."""
    for gen in generators:
        yield from gen

gen1 = (x for x in range(3))
gen2 = (x for x in range(3, 6))
combined = combine(gen1, gen2)
print(list(combined))  # [0, 1, 2, 3, 4, 5]

# Generator vs list comprehension timing
import time

# List comprehension (eager)
start = time.time()
result = [x**2 for x in range(10000000)]
list_time = time.time() - start

# Generator expression (lazy)
start = time.time()
gen = (x**2 for x in range(10000000))
gen_time = time.time() - start

print(f"List: {list_time:.4f}s")  # Slower (creates list)
print(f"Gen: {gen_time:.4f}s")    # Faster (creates generator)`,
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
				{
					Title: "Async/Await Basics",
					Content: `**Asynchronous Programming:**

**What is Async/Await:**
- Asynchronous programming in Python
- Non-blocking I/O operations
- Concurrent execution without threads
- Introduced in Python 3.5
- Built on generators and coroutines

**Why Use Async:**
- **I/O-bound tasks**: Network requests, file operations
- **Concurrency**: Handle many operations simultaneously
- **Performance**: Better than threads for I/O-bound work
- **Scalability**: Handle thousands of connections
- **Non-blocking**: Don't wait for slow operations

**Key Concepts:**

**1. Coroutines:**
- Functions defined with **async def**
- Return coroutine objects (not values directly)
- Must be awaited or run in event loop
- Can be paused and resumed

**2. await Keyword:**
- Pause execution until coroutine completes
- Only usable inside async functions
- Yields control back to event loop
- Allows other coroutines to run

**3. Event Loop:**
- Manages execution of coroutines
- Schedules async operations
- Handles I/O events
- Created with **asyncio.run()** or **asyncio.get_event_loop()**

**4. Tasks:**
- Wrappers around coroutines
- Can be scheduled concurrently
- Allow cancellation
- Track execution state

**Basic Syntax:**
- **async def**: Define async function
- **await**: Wait for async operation
- **asyncio.run()**: Run async function
- **asyncio.create_task()**: Create task

**Common Patterns:**
- **Sequential execution**: await one after another
- **Concurrent execution**: Use asyncio.gather()
- **Timeout**: Use asyncio.wait_for()
- **Background tasks**: Use asyncio.create_task()

**When to Use:**
- **Network I/O**: HTTP requests, WebSockets
- **File I/O**: Reading/writing files (with aiofiles)
- **Database operations**: Async database drivers
- **Web scraping**: Multiple URLs concurrently
- **APIs**: Multiple API calls

**When NOT to Use:**
- **CPU-bound tasks**: Use multiprocessing instead
- **Simple scripts**: Overhead not worth it
- **Synchronous code**: If no I/O operations

**Best Practices:**
- **Use async/await**: For I/O-bound operations
- **Avoid blocking**: Don't use blocking calls in async code
- **Use asyncio.gather()**: For concurrent operations
- **Handle errors**: Use try/except in async functions
- **Close resources**: Use async context managers

**Common Modules:**
- **asyncio**: Core async functionality
- **aiohttp**: Async HTTP client/server
- **aiofiles**: Async file operations
- **aiodns**: Async DNS resolution`,
					CodeExamples: `import asyncio

# Basic async function
async def hello():
    """Simple async function."""
    print("Hello")
    await asyncio.sleep(1)  # Non-blocking sleep
    print("World")

# Run async function
asyncio.run(hello())

# Async function with return
async def fetch_data():
    """Simulate fetching data."""
    await asyncio.sleep(1)  # Simulate network delay
    return {"data": "result"}

# Await result
async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())

# Sequential execution
async def sequential():
    """Run operations sequentially."""
    result1 = await fetch_data()
    result2 = await fetch_data()
    return result1, result2

# Concurrent execution with gather
async def concurrent():
    """Run operations concurrently."""
    results = await asyncio.gather(
        fetch_data(),
        fetch_data(),
        fetch_data()
    )
    return results

# Much faster - runs in parallel!

# Create tasks
async def create_tasks():
    """Create and manage tasks."""
    task1 = asyncio.create_task(fetch_data())
    task2 = asyncio.create_task(fetch_data())
    
    # Do other work while tasks run
    await asyncio.sleep(0.5)
    
    # Wait for tasks
    result1 = await task1
    result2 = await task2
    return result1, result2

# Timeout
async def with_timeout():
    """Execute with timeout."""
    try:
        result = await asyncio.wait_for(
            fetch_data(),
            timeout=0.5  # 0.5 seconds
        )
        return result
    except asyncio.TimeoutError:
        return "Timeout!"

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        print("Opening resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, *args):
        print("Closing resource")
        await asyncio.sleep(0.1)

async def use_resource():
    async with AsyncResource() as resource:
        await asyncio.sleep(1)
        print("Using resource")

# Practical example: Fetch multiple URLs
async def fetch_url(url):
    """Simulate fetching URL."""
    await asyncio.sleep(1)  # Simulate network delay
    return f"Data from {url}"

async def fetch_multiple_urls():
    """Fetch multiple URLs concurrently."""
    urls = ["url1", "url2", "url3", "url4", "url5"]
    
    # Sequential (slow - 5 seconds)
    # results = []
    # for url in urls:
    #     result = await fetch_url(url)
    #     results.append(result)
    
    # Concurrent (fast - ~1 second)
    results = await asyncio.gather(*[fetch_url(url) for url in urls])
    return results

# Error handling
async def fetch_with_error_handling():
    """Fetch with error handling."""
    try:
        result = await fetch_url("url")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# Cancel tasks
async def cancellable_task():
    """Task that can be cancelled."""
    try:
        await asyncio.sleep(10)
        return "Done"
    except asyncio.CancelledError:
        print("Task cancelled")
        raise

async def cancel_example():
    """Example of cancelling task."""
    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled")

# Running in background
async def background_task():
    """Background task."""
    while True:
        print("Background work")
        await asyncio.sleep(1)

async def main_with_background():
    """Main function with background task."""
    # Start background task
    bg_task = asyncio.create_task(background_task())
    
    # Do main work
    await asyncio.sleep(5)
    
    # Cancel background task
    bg_task.cancel()
    try:
        await bg_task
    except asyncio.CancelledError:
        pass

# Event loop (advanced)
async def event_loop_example():
    """Example using event loop directly."""
    loop = asyncio.get_event_loop()
    
    # Schedule coroutine
    future = asyncio.ensure_future(fetch_data())
    
    # Do other work
    await asyncio.sleep(0.5)
    
    # Wait for result
    result = await future
    return result

# Note: For real async I/O, use libraries like:
# - aiohttp for HTTP requests
# - aiofiles for file operations
# - asyncpg for PostgreSQL
# - aiomysql for MySQL`,
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
				{
					Title: "itertools Module",
					Content: `**itertools - Iterator Tools:**

**What is itertools:**
- Collection of tools for working with iterators
- Efficient, memory-friendly functions
- Common iterator patterns
- Part of standard library

**Infinite Iterators:**
- **count(start, step)**: Count from start infinitely
- **cycle(iterable)**: Cycle through iterable infinitely
- **repeat(value, times)**: Repeat value (or infinitely)

**Finite Iterators:**
- **chain(*iterables)**: Chain iterables together
- **compress(data, selectors)**: Filter using boolean mask
- **dropwhile(predicate, iterable)**: Drop while predicate true
- **takewhile(predicate, iterable)**: Take while predicate true
- **filterfalse(predicate, iterable)**: Filter where predicate false
- **islice(iterable, start, stop, step)**: Slice iterator

**Combinatoric Iterators:**
- **product(*iterables, repeat)**: Cartesian product
- **permutations(iterable, r)**: Permutations
- **combinations(iterable, r)**: Combinations
- **combinations_with_replacement()**: Combinations with replacement

**Common Use Cases:**
- **Chaining iterables**: Combine multiple sequences
- **Filtering**: Advanced filtering patterns
- **Grouping**: Group consecutive elements
- **Combinatorics**: Generate permutations/combinations
- **Windowing**: Sliding window over data

**Best Practices:**
- **Use for large data**: Memory efficient
- **Combine with generators**: For pipelines
- **Use itertools.chain**: Instead of list concatenation
- **Use itertools.islice**: For pagination
- **Use itertools.groupby**: For grouping`,
					CodeExamples: `from itertools import *

# Infinite iterators
# count - count from start
counter = count(10, 2)  # 10, 12, 14, 16, ...
print(list(islice(counter, 5)))  # [10, 12, 14, 16, 18]

# cycle - cycle through iterable
cycler = cycle([1, 2, 3])  # 1, 2, 3, 1, 2, 3, ...
print(list(islice(cycler, 7)))  # [1, 2, 3, 1, 2, 3, 1]

# repeat - repeat value
repeater = repeat("hello", 3)  # hello, hello, hello
print(list(repeater))  # ['hello', 'hello', 'hello']

# Finite iterators
# chain - chain iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
chained = chain(list1, list2)
print(list(chained))  # [1, 2, 3, 4, 5, 6]

# chain.from_iterable - chain from iterable of iterables
nested = [[1, 2], [3, 4], [5, 6]]
flattened = chain.from_iterable(nested)
print(list(flattened))  # [1, 2, 3, 4, 5, 6]

# compress - filter using boolean mask
data = [1, 2, 3, 4, 5]
selectors = [True, False, True, False, True]
compressed = compress(data, selectors)
print(list(compressed))  # [1, 3, 5]

# dropwhile - drop while predicate true
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
dropped = dropwhile(lambda x: x < 5, numbers)
print(list(dropped))  # [5, 6, 7, 8]

# takewhile - take while predicate true
taken = takewhile(lambda x: x < 5, numbers)
print(list(taken))  # [1, 2, 3, 4]

# filterfalse - filter where predicate false
evens = filterfalse(lambda x: x % 2, range(10))
print(list(evens))  # [0, 2, 4, 6, 8]

# islice - slice iterator
numbers = range(20)
sliced = islice(numbers, 5, 15, 2)  # Start=5, Stop=15, Step=2
print(list(sliced))  # [5, 7, 9, 11, 13]

# Combinatoric iterators
# product - Cartesian product
product_result = product([1, 2], [3, 4])
print(list(product_result))  # [(1, 3), (1, 4), (2, 3), (2, 4)]

# With repeat
dice = product([1, 2, 3, 4, 5, 6], repeat=2)
print(list(islice(dice, 6)))  # First 6 combinations

# permutations - all permutations
perms = permutations([1, 2, 3], 2)
print(list(perms))  # [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# combinations - combinations (order doesn't matter)
combs = combinations([1, 2, 3, 4], 2)
print(list(combs))  # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# combinations_with_replacement - with replacement
combs_wr = combinations_with_replacement([1, 2, 3], 2)
print(list(combs_wr))  # [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

# groupby - group consecutive elements
data = [1, 1, 1, 2, 2, 3, 3, 3, 3]
grouped = groupby(data)
for key, group in grouped:
    print(f"{key}: {list(group)}")
# 1: [1, 1, 1]
# 2: [2, 2]
# 3: [3, 3, 3, 3]

# Group by key function
words = ["apple", "banana", "cherry", "date"]
grouped = groupby(words, key=len)
for length, group in grouped:
    print(f"{length}: {list(group)}")
# 5: ['apple']
# 6: ['banana', 'cherry']
# 4: ['date']

# accumulate - cumulative operations
numbers = [1, 2, 3, 4, 5]
cumsum = accumulate(numbers)
print(list(cumsum))  # [1, 3, 6, 10, 15]

# With custom function
cumprod = accumulate(numbers, lambda x, y: x * y)
print(list(cumprod))  # [1, 2, 6, 24, 120]

# pairwise - pairs of consecutive elements (Python 3.10+)
# pairs = pairwise([1, 2, 3, 4])
# print(list(pairs))  # [(1, 2), (2, 3), (3, 4)]

# Practical examples

# Example 1: Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(chain.from_iterable(nested))

# Example 2: Pagination
def paginate(items, page_size):
    """Paginate items using islice."""
    it = iter(items)
    while True:
        page = list(islice(it, page_size))
        if not page:
            break
        yield page

# Example 3: Sliding window
def sliding_window(iterable, size):
    """Create sliding window of size."""
    it = iter(iterable)
    window = list(islice(it, size))
    if len(window) == size:
        yield tuple(window)
    for item in it:
        window = window[1:] + [item]
        yield tuple(window)

# Example 4: Generate all pairs
items = [1, 2, 3, 4]
pairs = list(combinations(items, 2))
print(pairs)  # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# Example 5: Batch processing
def batch(iterable, batch_size):
    """Process items in batches."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch`,
				},
				{
					Title: "functools Module",
					Content: `**functools - Higher-Order Functions:**

**What is functools:**
- Tools for working with functions
- Higher-order functions and operations
- Function decorators and utilities
- Part of standard library

**Key Functions:**

**1. functools.lru_cache:**
- Least Recently Used cache decorator
- Memoize function results
- Automatic cache management
- Configurable cache size

**2. functools.partial:**
- Partial function application
- Fix some arguments
- Create specialized functions
- Useful for callbacks

**3. functools.wraps:**
- Preserve function metadata
- Used in decorators
- Maintains __name__, __doc__, etc.
- Important for debugging

**4. functools.reduce:**
- Apply function cumulatively
- Reduce iterable to single value
- Functional programming tool
- Python 3: moved to functools

**5. functools.total_ordering:**
- Generate comparison methods
- Only need __eq__ and one comparison
- Automatically generates others
- Reduces boilerplate

**6. functools.singledispatch:**
- Single-dispatch generic functions
- Overload functions by type
- Polymorphic functions
- Type-based dispatch

**Common Use Cases:**
- **Caching**: Memoize expensive computations
- **Partial application**: Create specialized functions
- **Decorators**: Preserve metadata
- **Functional programming**: Reduce, map patterns
- **Type dispatch**: Polymorphic functions

**Best Practices:**
- **Use lru_cache**: For expensive pure functions
- **Use partial**: For callbacks and specialization
- **Use wraps**: Always in decorators
- **Use total_ordering**: For comparison classes
- **Use singledispatch**: For type-based dispatch`,
					CodeExamples: `from functools import *

# lru_cache - memoization
@lru_cache(maxsize=128)
def fibonacci(n):
    """Compute Fibonacci number (cached)."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# First call computes
print(fibonacci(30))  # Computed
print(fibonacci(30))  # From cache (instant)

# Cache info
print(fibonacci.cache_info())  # CacheInfo(hits=29, misses=31, maxsize=128, currsize=31)

# Clear cache
fibonacci.cache_clear()

# partial - partial function application
def multiply(x, y):
    return x * y

# Create specialized functions
double = partial(multiply, 2)  # Fix first argument
print(double(5))  # 10 (2 * 5)

triple = partial(multiply, 3)  # Fix first argument
print(triple(5))  # 15 (3 * 5)

# With keyword arguments
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Useful for callbacks
def process_data(data, callback):
    result = callback(data)
    return result

# Create specialized processors
process_numbers = partial(process_data, callback=lambda x: x * 2)
process_strings = partial(process_data, callback=lambda x: x.upper())

# wraps - preserve metadata in decorators
def timing_decorator(func):
    @wraps(func)  # Preserves func's metadata
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timing_decorator
def slow_function():
    """A slow function."""
    import time
    time.sleep(1)
    return "Done"

# Metadata preserved
print(slow_function.__name__)  # "slow_function" (not "wrapper")
print(slow_function.__doc__)   # "A slow function."

# reduce - cumulative operation
numbers = [1, 2, 3, 4, 5]

# Sum
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# Product
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120

# Maximum
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(maximum)  # 5

# With initial value
total = reduce(lambda x, y: x + y, numbers, 10)
print(total)  # 25 (10 + 15)

# total_ordering - generate comparison methods
@total_ordering
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __eq__(self, other):
        return self.age == other.age
    
    def __lt__(self, other):
        return self.age < other.age
    # Automatically gets __le__, __gt__, __ge__

p1 = Person("Alice", 30)
p2 = Person("Bob", 25)

print(p1 > p2)   # True (uses __lt__ automatically)
print(p1 <= p2)  # False (uses __lt__ and __eq__)

# singledispatch - type-based dispatch
@singledispatch
def process(value):
    """Default processor."""
    return f"Processing {type(value).__name__}: {value}"

@process.register(int)
def _(value):
    """Process integer."""
    return f"Integer: {value * 2}"

@process.register(str)
def _(value):
    """Process string."""
    return f"String: {value.upper()}"

@process.register(list)
def _(value):
    """Process list."""
    return f"List: {sum(value)}"

print(process(10))      # "Integer: 20"
print(process("hello")) # "String: HELLO"
print(process([1, 2, 3])) # "List: 6"
print(process(3.14))    # "Processing float: 3.14" (default)

# Practical examples

# Example 1: Caching expensive computation
@lru_cache(maxsize=None)  # Unlimited cache
def expensive_computation(n):
    """Expensive computation."""
    result = sum(i**2 for i in range(n))
    return result

# Example 2: Specialized functions
def send_email(to, subject, body):
    """Send email."""
    print(f"To: {to}, Subject: {subject}, Body: {body}")

# Create specialized senders
send_notification = partial(send_email, subject="Notification")
send_alert = partial(send_email, subject="Alert", to="admin@example.com")

send_notification("user@example.com", "Your order is ready")
send_alert("System error occurred")

# Example 3: Functional pipeline
def pipeline(*functions):
    """Chain functions together."""
    return lambda x: reduce(lambda acc, f: f(acc), functions, x)

# Create pipeline
process = pipeline(
    lambda x: x * 2,
    lambda x: x + 10,
    lambda x: x ** 2
)

print(process(5))  # ((5 * 2) + 10) ** 2 = 400`,
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
		{
			ID:          58,
			Title:       "Data Science Essentials",
			Description: "Introduction to NumPy, Pandas, Matplotlib, and data analysis with Python.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "NumPy Basics",
					Content: `**NumPy (Numerical Python):**
- Fundamental package for scientific computing
- Provides N-dimensional array object
- Fast mathematical operations
- Install: pip install numpy

**Key Features:**
- **ndarray**: N-dimensional array object
- **Vectorized operations**: Fast element-wise operations
- **Broadcasting**: Operations on arrays of different shapes
- **Linear algebra**: Matrix operations
- **Random number generation**: Statistical distributions

**Why NumPy:**
- Much faster than Python lists for numerical operations
- Memory efficient
- Rich mathematical functions
- Foundation for Pandas, SciPy, scikit-learn

**Common Operations:**
- Array creation: np.array(), np.zeros(), np.ones(), np.arange()
- Array manipulation: reshape(), transpose(), concatenate()
- Mathematical operations: +, -, *, /, np.sum(), np.mean()
- Indexing and slicing: Similar to Python lists but multi-dimensional`,
					CodeExamples: `import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
arr3 = np.zeros((3, 3))  # 3x3 array of zeros
arr4 = np.ones((2, 4))   # 2x4 array of ones
arr5 = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Array properties
print(arr2.shape)   # (2, 3) - dimensions
print(arr2.ndim)    # 2 - number of dimensions
print(arr2.size)    # 6 - total elements
print(arr2.dtype)   # int64 - data type

# Array operations (vectorized)
arr = np.array([1, 2, 3, 4, 5])
print(arr * 2)           # [2, 4, 6, 8, 10] - element-wise
print(arr + 10)          # [11, 12, 13, 14, 15]
print(arr ** 2)          # [1, 4, 9, 16, 25]

# Array operations between arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)             # [5, 7, 9]
print(a * b)             # [4, 10, 18]

# Mathematical functions
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))       # 15
print(np.mean(arr))      # 3.0
print(np.std(arr))       # Standard deviation
print(np.max(arr))       # 5
print(np.min(arr))       # 1

# Indexing and slicing
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr[0, 1])         # 2 - element at row 0, col 1
print(arr[0, :])         # [1, 2, 3] - first row
print(arr[:, 1])         # [2, 5, 8] - second column
print(arr[0:2, 1:3])     # [[2, 3], [5, 6]] - subarray

# Reshaping
arr = np.arange(12)
reshaped = arr.reshape(3, 4)  # 3x4 array

# Random numbers
random_arr = np.random.rand(3, 3)  # 3x3 random array [0, 1)
random_int = np.random.randint(0, 10, size=(3, 3))  # Random integers`,
				},
				{
					Title: "Pandas DataFrames",
					Content: `**Pandas:**
- Powerful data manipulation and analysis library
- Built on NumPy
- Provides DataFrame (2D) and Series (1D) objects
- Install: pip install pandas

**Key Features:**
- **DataFrame**: 2D labeled data structure (like Excel spreadsheet)
- **Series**: 1D labeled array
- **Data cleaning**: Handle missing data, duplicates
- **Data transformation**: Filter, group, merge, pivot
- **I/O**: Read/write CSV, Excel, JSON, SQL, etc.

**Common Operations:**
- Reading data: pd.read_csv(), pd.read_excel(), pd.read_json()
- Data inspection: head(), tail(), info(), describe()
- Selection: loc[], iloc[], column selection
- Filtering: Boolean indexing
- Grouping: groupby()
- Merging: merge(), join()

**Why Pandas:**
- Handles missing data gracefully
- Flexible data manipulation
- Time series support
- SQL-like operations
- Integration with other libraries`,
					CodeExamples: `import pandas as pd
import numpy as np

# Create DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'NYC', 'Chicago'],
    'salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)

# Read from CSV
df = pd.read_csv('data.csv')

# Basic inspection
print(df.head())        # First 5 rows
print(df.tail())        # Last 5 rows
print(df.info())        # Data types and non-null counts
print(df.describe())    # Statistical summary
print(df.shape)         # (rows, columns)

# Column selection
print(df['name'])       # Single column (Series)
print(df[['name', 'age']])  # Multiple columns (DataFrame)

# Row selection
print(df.loc[0])       # Row by label
print(df.iloc[0])       # Row by integer position
print(df.loc[0:2])      # Rows 0-2

# Filtering
young = df[df['age'] < 30]
nyc_residents = df[df['city'] == 'NYC']
high_salary = df[df['salary'] > 60000]

# Multiple conditions
filtered = df[(df['age'] < 30) & (df['salary'] > 50000)]

# Adding columns
df['bonus'] = df['salary'] * 0.1
df['total'] = df['salary'] + df['bonus']

# Grouping
by_city = df.groupby('city')
print(by_city['salary'].mean())  # Average salary by city
print(by_city.size())            # Count by city

# Aggregations
print(df.groupby('city').agg({
    'salary': ['mean', 'sum', 'count'],
    'age': 'mean'
}))

# Missing data
df['new_col'] = np.nan  # Add column with NaN
print(df.isnull())      # Check for missing values
df_clean = df.dropna()  # Remove rows with NaN
df_filled = df.fillna(0)  # Fill NaN with 0

# Sorting
df_sorted = df.sort_values('salary', ascending=False)

# Merging
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'id': [1, 2, 4], 'age': [25, 30, 35]})
merged = pd.merge(df1, df2, on='id', how='inner')`,
				},
				{
					Title: "Data Visualization with Matplotlib",
					Content: `**Matplotlib:**
- Comprehensive plotting library
- Create static, animated, and interactive visualizations
- Install: pip install matplotlib

**Key Features:**
- **Line plots**: Plot data over time
- **Bar charts**: Compare categories
- **Histograms**: Distribution of data
- **Scatter plots**: Relationship between variables
- **Subplots**: Multiple plots in one figure
- **Customization**: Colors, labels, legends, styles

**Common Plot Types:**
- plot(): Line plot
- scatter(): Scatter plot
- bar(): Bar chart
- hist(): Histogram
- boxplot(): Box plot
- pie(): Pie chart

**Best Practices:**
- Always label axes
- Add title and legend
- Use appropriate plot type
- Keep plots simple and clear
- Save figures with appropriate format`,
					CodeExamples: `import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Basic line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Sine Wave')
plt.grid(True)
plt.show()

# Multiple lines
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.show()

# Scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()

# Bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values)
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart')
plt.show()

# Histogram
data = np.random.normal(100, 15, 1000)
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].bar(categories, values)
axes[1, 1].hist(data, bins=30)
plt.tight_layout()
plt.show()

# Pandas integration
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
df.plot.scatter(x='x', y='y')
plt.show()`,
				},
				{
					Title: "Data Cleaning and Preprocessing",
					Content: `**Data Cleaning:**
- Critical step in data science workflow
- Real-world data is messy
- Handle missing values, duplicates, outliers
- Normalize and transform data

**Common Tasks:**
- **Handle missing data**: Remove or impute
- **Remove duplicates**: Drop duplicate rows
- **Handle outliers**: Detect and treat outliers
- **Data types**: Convert to appropriate types
- **Normalization**: Scale numerical features
- **Encoding**: Convert categorical to numerical

**Pandas Methods:**
- dropna(): Remove missing values
- fillna(): Fill missing values
- drop_duplicates(): Remove duplicates
- replace(): Replace values
- astype(): Convert data types
- normalize(): Normalize data`,
					CodeExamples: `import pandas as pd
import numpy as np

# Create sample data with issues
data = {
    'name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
    'age': [25, 30, 25, None, 28],
    'salary': [50000, 60000, 50000, 70000, 55000],
    'city': ['NYC', 'LA', 'NYC', 'NYC', 'Chicago']
}
df = pd.DataFrame(data)

# Check for missing values
print(df.isnull().sum())
print(df.isnull().any())

# Handle missing values
# Option 1: Remove rows with any missing value
df_clean = df.dropna()

# Option 2: Remove rows where all values are missing
df_clean = df.dropna(how='all')

# Option 3: Fill missing values
df_filled = df.fillna(0)  # Fill with 0
df_filled = df.fillna(df.mean())  # Fill with mean
df_filled = df['age'].fillna(df['age'].mean())  # Fill specific column

# Remove duplicates
df_no_dup = df.drop_duplicates()
df_no_dup = df.drop_duplicates(subset=['name'])  # Based on specific columns

# Handle outliers
# Using IQR method
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]

# Data type conversion
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(float)

# String operations
df['name_upper'] = df['name'].str.upper()
df['name_length'] = df['name'].str.len()

# Replace values
df['city'] = df['city'].replace('NYC', 'New York')

# Normalization (min-max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['salary_normalized'] = scaler.fit_transform(df[['salary']])

# One-hot encoding (categorical to numerical)
df_encoded = pd.get_dummies(df, columns=['city'])`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
