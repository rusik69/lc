package python

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterPythonModules([]problems.CourseModule{
		{
			ID:          2207,
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
					Content: `**Understanding Virtual Environments in Python:**

**1. What Are Virtual Environments and Why They Matter:**

A virtual environment is an isolated, self-contained directory tree that contains a Python installation along with its own set of installed packages. Think of it like having separate toolboxes for each construction project you work on — each toolbox has exactly the tools (packages) needed for that specific job, and tools in one box never interfere with tools in another. Without virtual environments, every Python project on your machine would share the same global set of packages. This might sound convenient at first, but it quickly becomes a nightmare: Project A might need version 1.0 of a library, while Project B requires version 2.0 of the same library. Installing one would break the other. Virtual environments solve this "dependency hell" problem entirely by giving each project its own isolated sandbox.

**2. How Virtual Environments Work Under the Hood:**

When you create a virtual environment, Python copies (or symlinks) its interpreter into a new directory and sets up a clean site-packages folder where packages will be installed. When you "activate" the environment, your shell's PATH variable is modified so that the Python interpreter and pip inside the virtual environment take precedence over the system-wide ones. This means any package you install with pip goes into the virtual environment's directory, not the global one. When you deactivate, your PATH reverts to normal, and you are back to using the system Python. The standard tool for creating virtual environments since Python 3.3 is the built-in venv module, which requires no additional installation. There are also third-party tools like virtualenv (which predates venv and offers more features), conda (popular in data science for managing both Python packages and system-level dependencies), and poetry (which combines dependency management with virtual environments).

**3. Creating and Using Virtual Environments:**

To create a virtual environment, run the command: python -m venv venv_name, where venv_name is the directory name you choose (commonly "venv" or ".venv"). This creates a directory structure containing the Python interpreter, pip, and an empty site-packages folder. To activate the environment on Linux or macOS, run: source venv/bin/activate. On Windows, use: venv\Scripts\activate. You will notice your terminal prompt changes to show the environment name in parentheses, confirming it is active. To deactivate, simply type "deactivate" at the command line, which restores your shell to its previous state.

**4. Managing Dependencies with requirements.txt:**

One of the most powerful aspects of virtual environments is reproducibility. Once you have installed all the packages your project needs, you can freeze the exact versions into a requirements.txt file by running: pip freeze > requirements.txt. This file lists every installed package and its precise version number. When a colleague (or your future self, or a CI/CD pipeline) needs to recreate the environment, they simply create a new virtual environment, activate it, and run: pip install -r requirements.txt. This guarantees that everyone working on the project uses the exact same dependency versions, eliminating "it works on my machine" problems. Think of requirements.txt as a recipe that lets anyone bake the exact same cake.

**5. Best Practices for Virtual Environments:**

Always create a virtual environment for every new project — even small scripts benefit from isolation. Add your virtual environment directory (e.g., "venv/") to your .gitignore file so it is not committed to version control; only the requirements.txt should be tracked. Use descriptive names for your environments or stick with the convention of ".venv" (which many IDEs like VS Code and PyCharm automatically detect). Keep your requirements.txt up to date whenever you add or remove packages. For more complex projects, consider using pip-tools, Poetry, or Pipenv, which offer advanced features like dependency resolution, lock files, and automatic virtual environment management.`,
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
			ID:          2208,
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
					Content: `**Understanding Exception Handling in Python:**

**1. What Are Exceptions and Why They Matter:**

Exceptions are events that occur during program execution that disrupt the normal flow of instructions. Think of them as alarm bells that go off when something unexpected happens — a file you tried to open does not exist, a user entered text where a number was expected, or you accidentally tried to divide by zero. Without exception handling, any such error would immediately crash your entire program, potentially losing data or leaving resources in an inconsistent state. Exception handling gives you a structured, elegant way to anticipate potential problems and respond gracefully. Instead of your program abruptly dying with a cryptic traceback, you can catch the error, display a friendly message to the user, log the problem for debugging, attempt an alternative approach, or clean up resources before exiting. In production software, robust exception handling is the difference between a professional application and a fragile script.

**2. The Try/Except Block — Your Safety Net:**

The try/except block is Python's primary mechanism for handling exceptions. The try clause contains the code that might raise an exception — this is the "risky" code you want to protect. If an exception occurs within the try block, Python immediately stops executing the remaining code in that block and jumps to the matching except clause. The except clause specifies which exception type(s) to catch and what to do when they occur. You can have multiple except clauses to handle different exception types differently, much like having different emergency procedures for different types of emergencies (fire vs. earthquake vs. medical). The else clause, which is optional, runs only if no exception was raised in the try block — it is the "everything went fine" path. The finally clause, also optional, runs no matter what — whether an exception occurred or not, whether it was caught or not. This makes finally perfect for cleanup operations like closing files, releasing locks, or disconnecting from databases.

**3. Common Exception Types You Will Encounter:**

Python has a rich hierarchy of built-in exceptions, each representing a specific category of error. **ValueError** is raised when a function receives an argument of the right type but an inappropriate value, such as passing a negative number to a function that expects positive values. **TypeError** occurs when an operation is applied to an object of inappropriate type, like trying to add a string to an integer. **FileNotFoundError** is raised when you attempt to open a file that does not exist on disk. **KeyError** occurs when you try to access a dictionary key that does not exist. **IndexError** is raised when you try to access a list element with an index that is out of range. **ZeroDivisionError** happens when you divide or modulo by zero. **AttributeError** occurs when you try to access an attribute or method that an object does not have. **PermissionError** is raised when you lack the necessary permissions for a file operation. Understanding which exceptions can be raised by different operations allows you to write targeted, precise error handling rather than catching everything blindly with a generic except clause.

**4. Best Practices for Exception Handling:**

The golden rule is to be specific: always catch the most specific exception type possible rather than using a bare "except" or catching the base Exception class. Catching too broadly can mask bugs by silently swallowing errors you did not anticipate. Keep your try blocks small — only wrap the specific lines of code that might raise the exception, not entire functions. Use the else clause for code that should only run when no exception occurred, keeping it separate from the protected code. Use the finally clause for cleanup that must happen regardless of success or failure. When catching an exception, use the "as" keyword to capture the exception object (e.g., except ValueError as e) so you can inspect or log the error message. Never silently swallow exceptions with an empty except block — at minimum, log the error so you have a trail for debugging.`,
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
					Content: `**Raising and Creating Custom Exceptions in Python:**

**1. Why Raise Exceptions — Being an Active Participant in Error Handling:**

While the previous lesson covered catching exceptions that Python raises for you, this lesson is about the other side of the coin: deliberately raising exceptions in your own code. When you write a function, you are making a contract with the caller — "give me valid inputs, and I will give you a correct result." But what happens when the caller breaks that contract? Rather than silently producing wrong results (which is far worse than crashing), you should raise an exception that clearly communicates what went wrong. Think of raising an exception like a referee blowing a whistle: the game stops, and everyone knows exactly what rule was violated. The raise keyword is how you blow that whistle. You can raise any exception type, and you should always include a descriptive error message that explains not just what happened, but ideally why it is a problem and what the caller should do differently.

**2. Using the raise Keyword Effectively:**

The basic syntax is straightforward: "raise ExceptionType("descriptive message")". You can raise any built-in exception type — ValueError for invalid arguments, TypeError for wrong types, RuntimeError for general runtime problems, and so on. Choose the exception type that most accurately describes the nature of the error. For example, if a function receives a negative number where only positive numbers make sense, raise ValueError — the type is correct (it is a number), but the value is invalid. If a function receives a string where it expected a number, raise TypeError. When re-raising an exception after partial handling (say, after logging it), you can use a bare "raise" statement without arguments inside an except block, which preserves the original traceback. You can also chain exceptions using "raise NewException() from original_exception", which creates a clear causal chain that helps with debugging.

**3. Creating Custom Exception Classes — Speaking Your Domain's Language:**

While Python's built-in exceptions cover generic error categories, real-world applications have domain-specific errors that deserve their own exception types. Creating a custom exception is as simple as defining a class that inherits from Exception (or from a more specific built-in exception). For example, a banking application might define InsufficientFundsError, AccountLockedError, and TransactionLimitExceededError. Each of these communicates a specific business rule violation far more clearly than a generic ValueError ever could. Custom exceptions can carry additional attributes — an InsufficientFundsError might include the current balance and the attempted withdrawal amount, giving the caller all the information needed to display a helpful error message or take corrective action. You can also create exception hierarchies by having your custom exceptions inherit from a common base exception specific to your application (e.g., class BankingError(Exception) as the parent), which allows callers to catch broad categories of errors or specific ones as needed.

**4. When and Where to Raise Exceptions — Guidelines for Defensive Programming:**

Raise exceptions at the boundaries of your code — at the beginning of functions to validate inputs (known as "guard clauses" or "precondition checks"), at integration points where external data enters your system, and whenever an operation cannot complete its contract. The principle is "fail fast and fail loud": detect problems as early as possible and report them clearly, rather than letting invalid data propagate through your system and cause mysterious failures far from the original mistake. However, do not overuse exceptions for normal control flow — exceptions should represent exceptional, unexpected situations, not routine branching logic. For example, checking whether a dictionary has a key before accessing it is better than catching KeyError in normal code. Reserve exception raising for genuine error conditions: invalid inputs that violate function contracts, preconditions that are not met, states that should be impossible if the rest of the code is correct, and situations where continuing would produce incorrect or dangerous results.`,
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
			ID:          2209,
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
					Content: `**Understanding Context Managers in Python:**

**1. What Are Context Managers and Why They Are Essential:**

A context manager is a Python object that defines a runtime context — a controlled environment where setup happens automatically before your code runs, and cleanup happens automatically after your code finishes, no matter what. The most familiar example is file handling: when you open a file using the "with" statement, the file is automatically closed when you exit the block, even if an exception occurs midway through. Think of a context manager like a responsible host at a dinner party — they set the table before guests arrive (setup), and they clean up after everyone leaves (teardown), regardless of whether the party went smoothly or someone accidentally knocked over a vase (an exception). Without context managers, you would need to manually handle cleanup with try/finally blocks everywhere, which is verbose, error-prone, and easy to forget. Context managers encode the cleanup logic once and guarantee it runs every time.

**2. The "with" Statement — Python's Resource Management Superpower:**

The "with" statement is the syntactic sugar that makes context managers so pleasant to use. When Python encounters "with open('file.txt') as f:", it calls the context manager's __enter__ method (which opens the file and returns the file object), binds the result to the variable f, executes your code block, and then calls the __exit__ method (which closes the file) when the block ends. The crucial insight is that __exit__ is called no matter how the block ends — whether it completes normally, encounters a return statement, or raises an exception. This guarantee is what makes context managers invaluable for any resource that needs deterministic cleanup: file handles, database connections, network sockets, thread locks, temporary directories, and even things like changing and restoring the working directory or modifying and reverting global state. You can also nest multiple context managers using "with A() as a, B() as b:" for managing several resources simultaneously.

**3. Creating Your Own Context Managers — Two Approaches:**

There are two ways to create context managers. The class-based approach requires you to define a class with two special methods: __enter__ (called when entering the "with" block, its return value is bound to the "as" variable) and __exit__ (called when leaving the block, receiving exception information as arguments). The __exit__ method receives three arguments — the exception type, exception value, and traceback — allowing it to inspect and optionally suppress exceptions by returning True. This approach is ideal when your context manager needs to maintain state or when you want a reusable, object-oriented design. The function-based approach uses the @contextmanager decorator from the contextlib module, which lets you write a generator function where everything before the "yield" is the setup phase, and everything after is the cleanup phase. This is more concise and often more readable for simple context managers. The contextlib module also provides other helpful utilities like suppress() for selectively ignoring exceptions, redirect_stdout() for capturing output, and ExitStack for managing a dynamic collection of context managers.

**4. Real-World Use Cases and Best Practices:**

Context managers shine in any situation where you have a pair of operations that must happen together — open/close, acquire/release, start/stop, connect/disconnect, enter/exit. Database transactions are a perfect example: you begin a transaction, do some work, and then either commit (if everything succeeded) or rollback (if something failed). A context manager can automate this pattern perfectly. Thread locks are another classic case: acquire the lock on enter, release it on exit, preventing deadlocks even when exceptions occur. Timing blocks of code, temporarily changing environment variables, creating and cleaning up temporary files, and managing mock objects in tests are all common uses. The best practice is to use context managers whenever resource cleanup is required — they make your intentions clear, eliminate cleanup bugs, and make your code significantly more robust and readable.`,
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
			ID:          2210,
			Title:       "Standard Library",
			Description: "Explore Python's rich standard library: os, sys, datetime, collections, and more.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "os and sys Modules",
					Content: `**Understanding the os and sys Modules — Python's Interface to the Operating System:**

**1. The os Module — Your Bridge to the Operating System:**

The os module is one of Python's most important standard library modules, providing a portable way to interact with the operating system. Think of it as a universal translator between your Python code and whatever operating system your program happens to be running on — whether that is Windows, macOS, or Linux. It abstracts away the differences between operating systems so you can write code that works everywhere. The os module gives you the power to navigate and manipulate the filesystem (creating, renaming, and deleting files and directories), inspect and modify environment variables (which control system-wide configuration), execute system commands, manage processes, and query information about the operating system itself. For instance, os.getcwd() tells you the current working directory (where your script is "standing" in the filesystem), os.listdir() lists the contents of a directory (like the "ls" command on Unix or "dir" on Windows), and os.path.join() intelligently combines path components using the correct separator for the current OS (forward slash on Unix, backslash on Windows). The os.environ dictionary gives you access to all environment variables, which are commonly used to store configuration like API keys, database URLs, and feature flags. Note that for modern Python code, the pathlib module (covered in a previous lesson) is often preferred for path operations, but os remains essential for environment variables, process management, and operations that pathlib does not cover.

**2. The sys Module — Peeking Inside the Python Interpreter:**

While os connects you to the operating system, the sys module connects you to the Python interpreter itself. It provides access to variables and functions that are tightly coupled with the Python runtime environment. The most commonly used feature is sys.argv, a list containing the command-line arguments passed to your script — sys.argv[0] is the script name, and subsequent elements are the arguments. This is essential for building command-line tools and scripts that accept user input. sys.exit() allows you to terminate your program with a specific exit code (0 for success, non-zero for errors), which is important for scripts that are called by other programs or shell scripts that need to check whether your program succeeded. sys.path is a list of directories that Python searches when you use an import statement — understanding and sometimes modifying this list is crucial for debugging import errors. sys.stdin, sys.stdout, and sys.stderr give you direct access to the standard input, output, and error streams, allowing you to redirect output, read piped input, or write error messages to the appropriate stream. sys.version and sys.platform tell you which Python version and operating system you are running on, which is useful for writing code that adapts to its environment.

**3. Practical Patterns and When to Use Each Module:**

In practice, you will often use os and sys together. A typical command-line script might use sys.argv to parse arguments, os.path to construct file paths, os.environ to read configuration, and sys.exit to return an appropriate exit code. When building cross-platform applications, os.path.join() and os.sep (the OS-specific path separator) ensure your paths work correctly everywhere. os.getenv() is the safe way to read environment variables — it returns None (or a default you specify) if the variable is not set, unlike os.environ[] which raises KeyError. For environment management, you can set variables with os.environ["KEY"] = "value", though these changes only affect the current process and its children. The os module also provides os.makedirs() for creating nested directory structures (similar to "mkdir -p" on Unix), os.walk() for recursively traversing directory trees, and os.remove()/os.rmdir() for deleting files and directories.

**4. Best Practices and Modern Alternatives:**

While os and sys are fundamental, modern Python offers alternatives for some tasks. Use pathlib instead of os.path for path manipulation — it is more readable and Pythonic. Use argparse or click instead of raw sys.argv for parsing command-line arguments — they provide help messages, type checking, and validation automatically. Use subprocess instead of os.system() for running external commands — it is more secure and gives you better control over input/output. Use logging instead of writing directly to sys.stderr — it provides configurable log levels, formatting, and output destinations. However, os.environ, sys.exit(), os.getcwd(), and many other features of these modules remain the standard and recommended approach for their respective tasks.`,
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
					Content: `**Understanding the datetime and time Modules — Working with Dates and Time in Python:**

**1. The datetime Module — Your Complete Toolkit for Dates and Times:**

Working with dates and times is one of those tasks that seems simple on the surface but hides enormous complexity underneath — time zones, daylight saving time, leap years, varying month lengths, and different calendar systems all conspire to make temporal programming surprisingly tricky. Python's datetime module provides a robust, well-designed set of classes that handle most of this complexity for you. The module offers four core classes: datetime.date for calendar dates (year, month, day), datetime.time for times of day (hour, minute, second, microsecond), datetime.datetime for combined date and time (the most commonly used class), and datetime.timedelta for representing durations or differences between points in time. Think of a datetime object as a precise pin on the timeline, and a timedelta as the distance between two pins. The datetime.now() method gives you the current local date and time, while datetime.utcnow() gives you the current UTC time. You can create specific dates with the constructor: datetime(2024, 1, 15, 14, 30, 0) represents January 15, 2024 at 2:30 PM.

**2. Formatting and Parsing — Converting Between Strings and Dates:**

One of the most common tasks is converting between datetime objects and human-readable strings. The strftime() method (string format time) converts a datetime object into a formatted string using format codes: %Y for four-digit year, %m for two-digit month, %d for two-digit day, %H for hour (24-hour), %M for minute, %S for second, and many more. For example, now.strftime("%Y-%m-%d %H:%M:%S") produces a string like "2024-01-15 14:30:00". The reverse operation — parsing a string into a datetime object — uses the strptime() method (string parse time) with the same format codes. This is essential when reading dates from files, user input, or APIs. A common gotcha is that strptime is strict about the format: the string must exactly match the pattern you provide, or it raises ValueError. For more flexible parsing, the third-party dateutil library provides a parse() function that can intelligently handle many date formats automatically.

**3. Time Arithmetic with timedelta — Calculating Durations and Differences:**

The timedelta class represents a duration — a span of time — and is what you get when you subtract one datetime from another. You can create timedelta objects directly: timedelta(days=7) represents one week, timedelta(hours=3, minutes=30) represents three and a half hours. Adding a timedelta to a datetime gives you a new datetime in the future; subtracting gives you one in the past. This makes it easy to answer questions like "what date is 30 days from now?" or "how many days between these two dates?" Subtracting two datetimes gives you a timedelta, and you can access its total_seconds() method to get the exact duration in seconds. This is incredibly useful for deadline calculations, scheduling, age computations, billing period calculations, and any domain where time intervals matter.

**4. The time Module — Low-Level Time Operations:**

While the datetime module deals with calendar dates and clock times, the time module provides lower-level, system-oriented time functions. time.time() returns the current time as a floating-point number of seconds since the "epoch" (January 1, 1970 on Unix systems) — this is often called a "Unix timestamp" and is useful for measuring elapsed time or storing timestamps as simple numbers. time.sleep(seconds) pauses your program for the specified number of seconds, which is useful for rate limiting, polling loops, or adding delays between operations. For precise performance measurement, time.perf_counter() provides the highest resolution timer available on your system and is the recommended way to benchmark code. time.monotonic() provides a clock that never goes backwards (unlike the system clock, which can be adjusted), making it ideal for measuring timeouts and intervals.

**5. Best Practices for Working with Time:**

Always store and transmit times in UTC, converting to local time only for display to users. Use datetime objects rather than strings for all internal computations — strings are for humans, datetime objects are for code. When comparing or sorting dates, make sure all your datetime objects are either all "naive" (no timezone info) or all "aware" (with timezone info); mixing them will raise errors. For timezone-aware datetimes in modern Python (3.9+), use the zoneinfo module from the standard library. For serious date/time work, consider the third-party libraries arrow, pendulum, or dateutil, which provide more intuitive APIs and handle edge cases that the standard library does not.`,
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
					Content: `**Understanding the collections Module — Specialized Container Datatypes:**

**1. Why the collections Module Exists — Beyond Lists and Dicts:**

Python's built-in containers — lists, tuples, dictionaries, and sets — are incredibly versatile and handle most situations well. However, certain programming patterns come up so frequently that using basic containers for them results in awkward, verbose, or inefficient code. The collections module provides specialized container types that are optimized for these common patterns, making your code cleaner, faster, and more expressive. Think of it like having a well-stocked kitchen: a chef's knife handles most cutting tasks, but a bread knife, paring knife, and cleaver each excel at their specific jobs. The collections module gives you those specialized tools for data structures.

**2. deque — The Double-Ended Queue for Fast Operations on Both Ends:**

A deque (pronounced "deck", short for "double-ended queue") is like a list, but with a superpower: adding and removing elements from both the left and right ends is O(1) — constant time — whereas a regular list's insert(0, item) and pop(0) operations are O(n) because they require shifting every other element. This makes deque perfect for implementing queues (FIFO — first in, first out), stacks, or any situation where you need fast access to both ends of a sequence. Deques also support a maxlen parameter that creates a fixed-size buffer: when it is full and you add a new item to one end, an item is automatically discarded from the other end. This is ideal for implementing sliding windows, keeping the most recent N items, or building circular buffers. Real-world uses include BFS (breadth-first search) in graph algorithms, maintaining a history of recent actions with a fixed memory budget, and implementing producer-consumer patterns.

**3. Counter — Effortless Counting and Frequency Analysis:**

Counter is a dictionary subclass specifically designed for counting hashable objects. Instead of writing a loop with a regular dictionary to count word frequencies, you simply pass your iterable to Counter and get back an object that maps each element to its count. It provides the most_common(n) method, which returns the n most frequent elements and their counts — extremely useful for frequency analysis, finding top-N items, and statistical summaries. Counters also support mathematical operations: you can add two Counters together (combining counts), subtract them (finding differences), and use intersection and union operations. This makes Counter invaluable for tasks like analyzing text (word frequency), processing logs (error frequency), counting votes, inventory management, and any situation where you need to tally occurrences of items.

**4. defaultdict — Dictionaries That Never Raise KeyError:**

A defaultdict is a dictionary that automatically creates a default value when you access a key that does not exist, eliminating the need for tedious "check if key exists, if not initialize it" patterns. You configure it with a factory function that produces the default value: defaultdict(int) defaults to 0, defaultdict(list) defaults to empty lists, and defaultdict(set) defaults to empty sets. This is transformative for building grouping and aggregation patterns. For example, grouping items by category — instead of checking if each category key exists and creating a list if it does not, you simply append to defaultdict(list) and it just works. The code becomes dramatically cleaner and more readable. Common uses include building indexes, creating adjacency lists for graphs, aggregating data by keys, and any pattern where you accumulate values under dictionary keys.

**5. namedtuple — Tuples with Meaningful Names:**

A namedtuple is a lightweight, immutable data structure that gives names to each position in a tuple. Instead of accessing elements by mysterious numeric indices (point[0], point[1]), you access them by descriptive names (point.x, point.y). This dramatically improves code readability without the overhead of defining a full class. Named tuples use the same memory as regular tuples and are just as fast, making them ideal for representing simple data records like points, database rows, RGB colors, or any small collection of related values where immutability is appropriate. They also work as dictionary keys (since they are hashable), support iteration and unpacking, and provide a readable __repr__. For mutable alternatives with more features, consider dataclasses (Python 3.7+), which provide a similar improvement in readability but allow modification after creation.`,
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
					Content: `**Understanding the json and csv Modules — Working with Common Data Formats:**

**1. The json Module — Python's Gateway to the Web's Favorite Data Format:**

JSON (JavaScript Object Notation) has become the de facto standard for data exchange on the internet. Every web API you interact with, every configuration file in modern tools, and most inter-service communication in distributed systems uses JSON. Python's built-in json module makes it trivially easy to convert between Python data structures and JSON format. The process of converting a Python object (dictionaries, lists, strings, numbers, booleans, and None) into a JSON string is called serialization, and the reverse — converting a JSON string back into Python objects — is called deserialization. Think of it like translation between two languages: your Python code speaks Python (with dicts, lists, and tuples), while the outside world speaks JSON (with objects and arrays). The json module is the translator that converts fluently between the two. The four core functions are json.dumps() (dump to string — serializes a Python object to a JSON-formatted string), json.loads() (load from string — deserializes a JSON string to a Python object), json.dump() (dump to file — writes JSON directly to a file object), and json.load() (load from file — reads JSON directly from a file object). The 's' at the end of dumps/loads stands for "string" — these variants work with strings, while dump/load work with file objects.

**2. JSON Best Practices and Common Pitfalls:**

When serializing to JSON, remember that JSON has a more limited type system than Python. Dictionaries become JSON objects, lists become JSON arrays, strings remain strings, integers and floats become numbers, True/False become true/false, and None becomes null. However, tuples are silently converted to arrays (and will come back as lists when deserialized), sets cannot be serialized at all (you must convert them to lists first), and datetime objects require custom handling. For pretty-printing JSON (useful for configuration files or debugging), use the indent parameter: json.dumps(data, indent=2) produces nicely formatted, human-readable output. For non-ASCII characters, set ensure_ascii=False to preserve Unicode characters in the output. When working with custom objects, you can provide a custom encoder by subclassing json.JSONEncoder or by passing a default function that converts your objects to serializable types.

**3. The csv Module — Reading and Writing Tabular Data:**

CSV (Comma-Separated Values) is one of the oldest and most universal formats for tabular data — every spreadsheet program, database tool, and data analysis platform can read and write CSV files. Python's csv module provides a clean, efficient way to work with this format, handling the many edge cases that make CSV trickier than it appears (fields containing commas, fields with embedded newlines, fields wrapped in quotes, different delimiter characters, and varying line endings across operating systems). The csv.reader reads rows as lists of strings, while csv.DictReader reads rows as dictionaries with the header row's values as keys — the latter is almost always more readable and less error-prone since you access fields by name rather than numeric index. Similarly, csv.writer writes lists, while csv.DictWriter writes dictionaries. When opening CSV files, always pass newline="" to the open() function — this prevents the csv module from mishandling line endings on different platforms.

**4. Choosing Between JSON and CSV — and When to Use Each:**

JSON and CSV serve different purposes and have different strengths. CSV excels at flat, tabular data — rows and columns, like a spreadsheet. It is simple, compact, and universally supported. JSON excels at hierarchical, nested data — it can represent complex structures like objects within objects, arrays within arrays, and mixed types. If your data is naturally a table (database exports, spreadsheet data, log entries), CSV is usually the right choice. If your data has nested structures (API responses, configuration files, document stores), JSON is the way to go. Both formats are human-readable text, making them easy to inspect and debug. For very large datasets, consider the csv module's memory-efficient row-by-row processing, which allows you to handle files larger than your available memory.`,
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
			ID:          2211,
			Title:       "Testing & Debugging",
			Description: "Learn unittest, pytest, mocking, and debugging techniques.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "unittest Framework",
					Content: `**Understanding the unittest Framework — Python's Built-In Testing Toolkit:**

**1. Why Testing Matters and What unittest Provides:**

Testing is the practice of writing code that verifies your other code works correctly. It might seem like extra work at first, but it is one of the most valuable investments you can make in a software project. Think of tests as a safety net for acrobats — they allow you to make bold, confident changes to your code knowing that if something breaks, your tests will catch it immediately. Without tests, every change to your codebase is a gamble: you might fix one thing and unknowingly break three others. Python's unittest module is the built-in testing framework that ships with every Python installation, requiring no additional packages. It was inspired by JUnit from the Java world and follows the xUnit family of testing frameworks, so if you have experience with testing in other languages, the concepts will feel familiar. While third-party frameworks like pytest (covered in the next lesson) are more popular for new projects due to their simpler syntax, unittest remains important to understand because many existing codebases use it, and it forms the foundation that other testing tools build upon.

**2. Structuring Tests with TestCase Classes:**

In unittest, tests are organized into classes that inherit from unittest.TestCase. Each test method within the class must start with "test_" — this naming convention is how unittest discovers which methods are tests and which are helper methods. The setUp() method, if defined, runs before each test method, setting up any state or resources your tests need (like creating objects, opening database connections, or preparing test data). Similarly, tearDown() runs after each test method, cleaning up resources. This setup/teardown pattern ensures each test starts with a fresh, known state, preventing one test's side effects from contaminating another — a critical property called test isolation. Think of it like a restaurant kitchen: before each meal service (test), the kitchen is cleaned and prep is done (setUp), and after service, everything is cleaned again (tearDown). For expensive resources shared across all tests in a class, you can use setUpClass() and tearDownClass() class methods, which run once for the entire class rather than before each individual test.

**3. Assert Methods — The Vocabulary of Verification:**

The TestCase class provides a rich set of assertion methods that verify expected behavior. assertEqual(a, b) checks that two values are equal and provides a clear failure message showing both values if they differ. assertNotEqual(a, b) verifies two values are not the same. assertTrue(x) and assertFalse(x) check boolean conditions. assertIn(a, b) verifies that a is a member of container b — useful for checking if a string contains a substring or a list contains an element. assertRaises(ExceptionType) verifies that a specific exception is raised — this is how you test error handling code. assertAlmostEqual(a, b) handles floating-point comparison (since 0.1 + 0.2 does not exactly equal 0.3 in floating-point math). Each assertion method produces a descriptive failure message when the check fails, making it easy to understand what went wrong without adding print statements everywhere. You can also provide custom failure messages as an additional argument to any assertion method.

**4. Running Tests and Best Practices:**

To run your tests, you can use the unittest.main() call at the bottom of your test file (which runs when the file is executed directly), or use the command-line discovery: python -m unittest discover searches for and runs all test files in your project. Write tests that are small, focused, and independent — each test should verify one specific behavior. Name your test methods descriptively: test_add_positive_numbers, test_divide_by_zero_raises_error — the name should read like a specification of what behavior is being verified. Aim for tests that are fast (so you run them frequently), deterministic (same result every time), and independent (order does not matter). A good rule of thumb is to write tests for every function that has any logic, every bug you fix (to prevent regression), and every edge case you can think of. Tests are not just about finding bugs — they serve as living documentation of how your code is supposed to behave.`,
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
					Content: `**Understanding pytest — The Modern Python Testing Framework:**

**1. Why pytest Has Become the Standard for Python Testing:**

While unittest is Python's built-in testing framework, pytest has become the overwhelming favorite in the Python community for new projects — and for good reason. pytest takes a radically simpler approach to testing: instead of requiring you to organize tests into classes, inherit from a base class, and use special assertion methods, pytest lets you write plain functions with plain assert statements. This dramatic reduction in ceremony means you spend more time thinking about what to test and less time wrestling with framework boilerplate. Think of the difference like writing a letter by hand versus filling out a bureaucratic form — both communicate information, but one is far more natural and pleasant. pytest is a third-party package (install it with pip install pytest), but it is so widely adopted that it is effectively the standard testing tool in the Python ecosystem. Major projects like Django, Flask, SQLAlchemy, and thousands of others use pytest for their test suites.

**2. The Power of Simple Assert Statements:**

One of pytest's most beloved features is its handling of assert statements. In unittest, you must choose from dozens of specialized assertion methods (assertEqual, assertIn, assertRaises, etc.), and if you use the wrong one, the failure message might be unhelpful. With pytest, you just write "assert expression" using plain Python, and pytest uses "assertion introspection" to automatically generate detailed, informative failure messages. When "assert result == expected" fails, pytest shows you both the actual and expected values, the difference between them, and even highlights which parts differ in long strings or lists. This means you get better error messages with simpler code — truly the best of both worlds. You can still add a custom message with "assert condition, 'explanation'" when needed.

**3. Fixtures — pytest's Revolutionary Approach to Test Setup:**

Fixtures are pytest's replacement for unittest's setUp/tearDown, and they are far more powerful and flexible. A fixture is a function decorated with @pytest.fixture that provides data or resources to tests. Tests request fixtures simply by including the fixture name as a function parameter — pytest's dependency injection system automatically calls the fixture and passes its return value to the test. Fixtures can use "yield" to separate setup and cleanup code: everything before the yield is setup, and everything after is cleanup. Fixtures can depend on other fixtures, creating composable chains. They can be scoped to different lifetimes: "function" (default, runs once per test), "class" (once per test class), "module" (once per test file), or "session" (once for the entire test run). This granular control means expensive resources like database connections can be shared efficiently while cheap resources are recreated for each test.

**4. Test Discovery, Running, and Advanced Features:**

pytest automatically discovers tests by looking for files named test_*.py or *_test.py, and within those files, functions named test_* and classes named Test*. Running "pytest" with no arguments searches the current directory recursively. Use "pytest -v" for verbose output showing each test name and result. Use "pytest -k pattern" to run only tests whose names match a pattern. Use "pytest -x" to stop at the first failure (useful for fixing one bug at a time). Use "pytest --tb=short" for condensed tracebacks. pytest also supports parametrized tests (running the same test with multiple inputs using @pytest.mark.parametrize), test marking (tagging tests with categories like "slow" or "integration" using @pytest.mark), and a vast ecosystem of plugins for everything from code coverage (pytest-cov) to parallel execution (pytest-xdist) to async testing (pytest-asyncio). The combination of simplicity for basic use and power for advanced scenarios is what makes pytest the preferred choice for Python testing.`,
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
					Content: `**Mastering Debugging Techniques in Python:**

**1. The Art of Debugging — More Than Just Finding Bugs:**

Debugging is the systematic process of identifying, isolating, and fixing defects in your code. It is a skill that separates novice programmers from experienced ones, and it is something you will spend a surprising amount of your career doing — by some estimates, developers spend 30-50% of their time debugging. The key insight is that debugging is not about randomly changing code and hoping for the best; it is about forming hypotheses ("I think the bug is caused by X"), designing experiments to test those hypotheses (adding observations, running specific inputs), and narrowing down the problem space until you find the root cause. Good debugging starts with reproducing the bug reliably — if you cannot make it happen consistently, you cannot verify that you have fixed it. Then you systematically narrow the scope: which module? Which function? Which line? What is the state of the variables at that point? Python provides several tools at different levels of sophistication for this process.

**2. print() Debugging — Simple, Universal, and Surprisingly Effective:**

The humble print() function is the most basic debugging tool, and despite its simplicity, it remains useful even for experienced developers. The idea is simple: insert print statements at strategic points in your code to observe variable values, execution flow, and intermediate results. While it lacks the sophistication of more advanced tools, print debugging has advantages: it works everywhere (no setup needed), it is easy to understand, and it can trace execution through complex control flow. The key to effective print debugging is being strategic about what and where you print. Use f-strings for clarity: print(f"DEBUG: x={x}, type={type(x)}, len={len(x)}"). Print at function entry and exit to trace call flow. Print loop variables to understand iteration behavior. Always remove or comment out your debug prints when you are done — or better yet, use the logging module instead, which lets you leave the statements in place but turn them on and off with a configuration change.

**3. pdb — Python's Interactive Debugger:**

pdb (Python Debugger) is Python's built-in interactive debugger, and learning to use it is a significant level-up for any Python developer. It allows you to pause your program's execution at any point, inspect the state of all variables, step through code line by line, evaluate arbitrary expressions, and even modify variables on the fly. To start the debugger, insert pdb.set_trace() (or in Python 3.7+, simply breakpoint()) at the point where you want to pause. When execution reaches that line, you are dropped into an interactive prompt. The most important commands are: "n" (next) to execute the next line without stepping into function calls, "s" (step) to step into a function call, "c" (continue) to resume normal execution until the next breakpoint, "l" (list) to show the source code around your current position, "p variable" to print a variable's value, "pp expression" to pretty-print a complex expression, "w" (where) to show the call stack, and "q" (quit) to exit the debugger. Think of pdb as having a freeze-frame button for your running program — you can stop time, look around, understand the state, and then resume.

**4. The logging Module — Production-Grade Debugging:**

While print() is great for quick investigations and pdb is perfect for deep interactive debugging, the logging module is the right tool for long-term, production-quality instrumentation. The logging module provides configurable severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) that let you control how much detail is recorded. During development, you set the level to DEBUG to see everything; in production, you raise it to WARNING or ERROR to capture only important events. Log messages can be directed to multiple destinations simultaneously — the console, log files, remote logging services, or email alerts — using handlers. Each log message automatically includes a timestamp, the logger name, the severity level, and your message, creating a structured record that is easy to search and analyze. The key advantage over print() is that logging statements can remain permanently in your code: they serve as documentation of important events, provide visibility into production behavior, and can be activated or silenced without changing code. Always use logging.getLogger(__name__) to create module-specific loggers, and configure them centrally using logging.basicConfig() or a configuration dictionary for complex setups.`,
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
			ID:          2212,
			Title:       "Web Development Basics",
			Description: "Introduction to Flask, Django, and web development with Python.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Flask Basics",
					Content: `**Understanding Flask — Python's Lightweight Web Framework:**

**1. What is Flask and Why Choose It:**

Flask is a lightweight, flexible web framework for Python that makes it remarkably easy to build web applications and APIs. It is often called a "micro-framework" — not because it is limited in capability, but because it keeps its core small and simple, letting you choose the tools and libraries you want rather than imposing decisions on you. Think of Flask as a LEGO baseplate: it gives you a solid foundation, and you snap on exactly the pieces you need. This contrasts with full-featured frameworks like Django that come with everything pre-built (more on Django in the next lesson). Flask is the go-to choice for building REST APIs, microservices, prototypes, small to medium web applications, and any project where you want maximum control over your technology choices. It powers parts of Pinterest, LinkedIn, and many startups. To get started, install it with pip install flask, and you can have a working web application in literally five lines of code.

**2. How Flask Works — Routes, Views, and the Request-Response Cycle:**

At its heart, Flask follows a simple pattern: URLs are mapped to Python functions (called "view functions" or "route handlers"), and when a user visits a URL, Flask calls the corresponding function and sends its return value back to the browser. You define these mappings using the @app.route() decorator, which is one of the most elegant uses of decorators in the Python ecosystem. For example, @app.route("/") maps the root URL to a function, @app.route("/about") maps the /about URL, and @app.route("/users/<int:user_id>") maps URLs with dynamic parameters that are passed as arguments to your function. Routes can be restricted to specific HTTP methods (GET, POST, PUT, DELETE) using the methods parameter, which is essential for building RESTful APIs. When a request comes in, Flask creates a request object containing all the information about the incoming request (headers, form data, JSON body, query parameters, cookies), and your view function returns a response — either a simple string (rendered as HTML), a tuple of (body, status_code), or a Response object for more control. The jsonify() function makes it easy to return JSON responses for API endpoints.

**3. Flask Application Structure and Configuration:**

A Flask application starts with creating a Flask instance: app = Flask(__name__). The __name__ argument tells Flask where to find templates and static files relative to your module. For small applications, a single Python file works fine. For larger projects, Flask supports a modular structure using Blueprints — reusable components that group related routes, templates, and static files. A typical Flask project structure includes a templates/ folder for HTML templates (using the Jinja2 templating engine), a static/ folder for CSS, JavaScript, and images, and your Python application file(s). Configuration is handled through app.config, which can load settings from Python files, environment variables, or dictionaries. Running app.run(debug=True) starts a development server with auto-reload (it restarts when you change code) and a debugger that appears in the browser when errors occur — incredibly useful during development but must be disabled in production.

**4. When to Use Flask and the Flask Ecosystem:**

Flask is ideal when you want simplicity and flexibility: REST APIs and microservices, single-page application backends, prototypes and MVPs, projects where you want to choose your own ORM, authentication system, and other components. Flask has a rich ecosystem of extensions that add common functionality: Flask-SQLAlchemy for database integration, Flask-Login for user authentication, Flask-RESTful or Flask-RESTX for building RESTful APIs with automatic documentation, Flask-CORS for handling cross-origin requests, Flask-Migrate for database migrations, and many more. For production deployment, do not use the built-in development server — use a proper WSGI server like Gunicorn or uWSGI behind a reverse proxy like Nginx. Flask also integrates well with modern async Python through frameworks like Quart (an async-compatible Flask API).`,
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
					Content: `**Understanding Django — Python's Full-Featured Web Framework:**

**1. What is Django and Its "Batteries Included" Philosophy:**

Django is Python's most popular full-featured web framework, designed to help developers build complex, database-driven websites quickly and with clean, pragmatic design. Unlike Flask's minimalist approach, Django follows a "batteries included" philosophy — it ships with everything you need to build a production-ready web application out of the box: an ORM (Object-Relational Mapper) for database operations, a powerful automatic admin interface, user authentication and authorization, form handling and validation, template engine, URL routing, middleware support, caching, internationalization, and much more. Think of Django as a fully furnished apartment versus Flask's empty loft: with Django, you move in and start living (building features) immediately, while with Flask, you get to choose every piece of furniture yourself. Django powers some of the internet's biggest sites, including Instagram, Mozilla, Pinterest, Disqus, and The Washington Post. It is the go-to choice for content management systems, e-commerce platforms, social networks, and any application with complex data models and business logic.

**2. Django's Architecture — The MTV Pattern:**

Django follows the Model-Template-View (MTV) pattern, which is its version of the well-known Model-View-Controller (MVC) pattern. Models define your data structure and map to database tables — you write Python classes, and Django automatically creates the SQL schema, handles migrations, and provides a rich query API. Templates are HTML files with Django's template language for dynamic content rendering — they handle presentation and keep logic out of your HTML. Views are Python functions (or classes) that receive web requests and return web responses — they contain the business logic that determines what data to fetch, how to process it, and what to display. URLs are configured in urls.py files that map URL patterns to views, similar to Flask's routes but organized centrally. This separation of concerns keeps your code organized: data logic in models, presentation in templates, and business logic in views. Django also has a concept of "apps" — self-contained modules within a project that encapsulate related functionality (like a "blog" app, a "users" app, or a "payments" app), making code reusable across projects.

**3. Django's Killer Features — ORM and Admin:**

Django's ORM is one of its most powerful features. Instead of writing raw SQL, you define model classes with fields, and Django handles all the database operations. You can query with Python: User.objects.filter(age__gt=25).order_by('-name') instead of writing SQL. The ORM supports relationships (ForeignKey, ManyToMany, OneToOne), aggregations, annotations, complex lookups, and transactions. When you change your models, Django's migration system automatically generates migration files that update the database schema — no manual SQL required. The Django admin is another standout feature: with just a few lines of configuration, you get a complete web-based interface for managing your application's data. Create, read, update, and delete records, search, filter, and sort — all generated automatically from your models. This is invaluable for internal tools, content management, and rapid prototyping, often saving weeks of development time.

**4. Getting Started and Django Commands:**

Django projects are managed through the django-admin command and the manage.py script. To start a new project, run "django-admin startproject myproject", which creates the project skeleton with settings, URL configuration, and WSGI/ASGI entry points. To add functionality, create apps with "python manage.py startapp myapp", then register the app in settings.py. "python manage.py runserver" starts the development server. "python manage.py makemigrations" detects model changes and creates migration files, while "python manage.py migrate" applies those migrations to the database. "python manage.py createsuperuser" creates an admin account. Django is best suited for medium to large applications where development speed, security, and scalability are priorities — it handles much of the security hardening (CSRF protection, SQL injection prevention, XSS protection) automatically, letting you focus on building features rather than worrying about vulnerabilities.`,
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
			ID:          2213,
			Title:       "Best Practices & Idioms",
			Description: "Learn PEP 8, Pythonic code patterns, and common pitfalls.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "PEP 8 Style Guide",
					Content: `**Understanding PEP 8 — The Official Python Style Guide:**

**1. What is PEP 8 and Why Consistent Style Matters:**

PEP 8 is Python Enhancement Proposal 8, the official style guide for Python code written by Guido van Rossum (Python's creator) and other core developers. It defines conventions for how Python code should be formatted and organized to maximize readability. But why does style matter so much? Because code is read far more often than it is written — by your colleagues, by your future self, and by contributors to open source projects. Consistent style reduces cognitive load: when all the code in a project looks the same, you can focus on what it does rather than how it is formatted. Think of PEP 8 like grammar rules for a natural language — you can technically be understood without them, but proper grammar makes communication smoother and more professional. PEP 8 is not enforced by the Python interpreter (your code runs fine with any formatting), but following it is considered a mark of professionalism and is expected in most Python teams and open source projects. The guide itself says "a foolish consistency is the hobgoblin of little minds" — there are times when breaking the rules makes sense, but you should know the rules before you break them.

**2. Key Formatting Rules — The Foundation of Readable Python:**

The most important PEP 8 rules cover indentation, line length, spacing, and blank lines. Use 4 spaces per indentation level — not tabs, not 2 spaces, but exactly 4 spaces. This is deeply embedded in Python culture and virtually all Python code follows this convention. Limit lines to 79 characters for code and 72 for comments and docstrings — while modern screens can show more, shorter lines make side-by-side diffs easier and prevent horizontal scrolling. Use blank lines strategically: two blank lines before and after top-level function and class definitions, one blank line between methods inside a class, and single blank lines within functions to separate logical sections. Put all import statements at the top of the file, organized into three groups separated by blank lines: standard library imports first, then third-party library imports, then local application imports. Surround binary operators with single spaces (x = a + b, not x=a+b), but do not use spaces inside brackets (call(arg), not call( arg )). Use trailing commas in multi-line structures to make diffs cleaner.

**3. Naming Conventions — Communicating Intent Through Names:**

PEP 8 defines clear naming conventions that communicate the type and purpose of each identifier. Use snake_case (lowercase with underscores) for functions, methods, variables, and module names: calculate_total_price, user_count, my_module.py. Use PascalCase (CapitalizedWords) for class names: UserAccount, HTTPConnection, DatabaseManager. Use UPPER_CASE_WITH_UNDERSCORES for constants: MAX_RETRIES, DEFAULT_TIMEOUT, API_BASE_URL. Names starting with a single underscore (_private_method) indicate internal use — a convention that signals "this is an implementation detail, not part of the public API." Names starting with double underscores (__mangled) trigger Python's name mangling mechanism, which is rarely needed. Names with leading and trailing double underscores (__init__, __str__) are reserved for Python's special methods. Avoid single-character names except for short, obvious contexts like loop variables (i, j, k) or mathematical formulas. Choose descriptive, meaningful names that reveal intent — the goal is to write code that reads almost like English prose.

**4. Tools for Enforcing PEP 8 — Let Machines Do the Tedious Work:**

Manually checking PEP 8 compliance is tedious and error-prone, so the Python ecosystem has excellent tools for automating style enforcement. Black is an "opinionated" code formatter that automatically reformats your code to be PEP 8 compliant (with a default line length of 88 characters) — you do not make choices, it just makes everything consistent. Many teams adopt Black because it eliminates style debates entirely: "just run Black" is the answer to every formatting question. Flake8 is a linting tool that checks your code for style violations and common errors without modifying it — useful as a pre-commit hook or CI check. autopep8 is another formatter that is less opinionated than Black, making only the changes needed for PEP 8 compliance. isort specifically handles import sorting and grouping. Ruff is a newer, extremely fast linter and formatter written in Rust that combines the functionality of flake8, isort, and more. Most modern editors (VS Code, PyCharm) can be configured to run these tools automatically on save, making PEP 8 compliance effortless. The investment in setting up these tools pays off immediately in cleaner, more consistent code.`,
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
					Content: `**Understanding Pythonic Code Patterns — Writing Code the Python Way:**

**1. What "Pythonic" Means and Why It Matters:**

"Pythonic" code is code that uses Python's features and idioms in the way they were intended to be used. Every programming language has its own culture and conventions — code that would be perfectly natural in Java or C++ might look awkward and verbose in Python, and vice versa. Writing Pythonic code means embracing Python's philosophy of readability, simplicity, and expressiveness. The Zen of Python (try running "import this" in a Python shell) encapsulates this philosophy: "Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex. Readability counts." Pythonic code is not just an aesthetic preference — it is often more efficient, less buggy, and easier to maintain. When experienced Python developers review your code, they will immediately notice whether it follows Pythonic patterns or reads like "Java written in Python." Learning these patterns is like learning the idioms of a natural language — they make you fluent rather than just grammatically correct.

**2. List Comprehensions and Generator Expressions — Python's Power Tools:**

List comprehensions are one of Python's most distinctive and beloved features. Instead of writing a multi-line loop to build a list (create empty list, iterate, append), you express the entire operation in a single, readable line: [x**2 for x in range(10)] creates a list of squares, [name.upper() for name in names if len(name) > 3] filters and transforms in one expression. This is not just shorter — it clearly communicates intent: "I am creating a list by transforming/filtering another sequence." Similarly, dictionary comprehensions ({k: v for k, v in pairs}) and set comprehensions ({x for x in items}) follow the same pattern. For large datasets where you do not need the entire list in memory at once, generator expressions (using parentheses instead of brackets) provide lazy evaluation: (x**2 for x in range(10000000)) generates values one at a time, using O(1) memory regardless of the input size. The rule of thumb is: use list comprehensions for small to medium transformations, generator expressions for large data or when you only need to iterate once.

**3. Enumeration, Unpacking, and Iteration Patterns:**

Python provides elegant built-in functions for common iteration patterns that eliminate the need for manual indexing and temporary variables. The enumerate() function gives you both the index and the value when iterating, so instead of the clunky "for i in range(len(items)): item = items[i]", you write "for i, item in enumerate(items):" — cleaner, less error-prone, and immediately obvious in intent. The zip() function lets you iterate over multiple sequences in parallel: "for name, age in zip(names, ages):" pairs corresponding elements together, like a zipper joining two sides. Tuple unpacking allows you to assign multiple values at once: "first, *rest = my_list" grabs the first element and collects everything else into "rest." Swapping variables is a single line: "a, b = b, a" — no temporary variable needed. The "in" operator provides clean membership testing: "if item in collection:" works for lists, sets, dictionaries, strings, and any iterable, replacing verbose loop-based searches.

**4. Context Managers, EAFP, and Other Pythonic Principles:**

Always use context managers (the "with" statement) for resources that need cleanup — files, database connections, locks, network sockets. This is not just cleaner, it is safer, because cleanup happens even when exceptions occur. Python follows the EAFP principle (Easier to Ask Forgiveness than Permission): instead of checking whether an operation will succeed before trying it, just try it and handle the exception if it fails. For example, instead of "if key in dictionary: value = dictionary[key]", Pythonic code uses "try: value = dictionary[key] except KeyError: handle_missing()." Use the ternary expression for simple conditional assignments: "result = value_if_true if condition else value_if_false." Use the walrus operator (:=) in Python 3.8+ for assignments within expressions: "if (n := len(items)) > 10: print(f'Too many: {n}')." Use f-strings for string formatting — they are the most readable and performant option. Prefer str.join() over concatenation in loops — "".join(parts) is both cleaner and dramatically faster for building strings from many pieces. These patterns, taken together, transform your code from merely functional to truly Pythonic.`,
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
					Content: `**Common Python Pitfalls — Mistakes Every Developer Should Know:**

**1. The Mutable Default Argument Trap — Python's Most Famous Gotcha:**

This is arguably the most common Python pitfall, and it trips up developers of all experience levels. When you define a function with a mutable default argument like def add_item(item, items=[]), the default list is created once when the function is defined, not each time it is called. This means every call to the function that uses the default shares the same list object. So calling add_item("apple") adds "apple" to the shared list, and the next call add_item("banana") adds "banana" to the same list — which now contains both items! This behavior is deeply counterintuitive because you expect each call to start fresh. The fix is simple but important: use None as the default and create the mutable object inside the function: def add_item(item, items=None): if items is None: items = []. This ensures a new list is created on every call. This same trap applies to dictionaries, sets, and any other mutable object used as a default argument. The underlying reason is that function default values are evaluated at function definition time (when Python first reads the def statement), not at call time.

**2. Modifying a Collection While Iterating Over It:**

Modifying a list (or dictionary, or set) while iterating over it is a recipe for subtle, hard-to-debug errors. When you remove items from a list inside a for loop, the iteration index gets confused because the list is shrinking underneath it — some items get skipped, and you may get unexpected results or IndexError exceptions. For example, if you loop through a list and remove even numbers, you might miss some because removing an element shifts all subsequent elements left by one position. The solution depends on the situation: for filtering, use a list comprehension to create a new list (numbers = [n for n in numbers if n % 2 != 0]), which is both correct and Pythonic. If you must modify in place, iterate over a copy (for item in list(original):) or iterate backwards (for i in range(len(items) - 1, -1, -1):). For dictionaries, you can iterate over list(dict.keys()) to avoid the "dictionary changed size during iteration" RuntimeError. The fundamental lesson is: never modify the thing you are iterating over — either create a new collection or work with a copy.

**3. The Difference Between == and is — Value vs Identity:**

This confusion causes subtle bugs that can be maddening to track down. The == operator compares values (do these two objects contain the same data?), while the "is" operator compares identity (are these literally the same object in memory?). Two lists [1, 2, 3] and [1, 2, 3] are equal (==) but are not the same object (not "is"). Python caches small integers (-5 to 256) and short strings, so "is" sometimes works for them by accident — but relying on this is a bug waiting to happen. Always use == for comparing values, and reserve "is" for two specific cases: checking for None (if x is None:) and checking for sentinel values. Using "is" instead of == for value comparison is a particularly insidious bug because it might appear to work in testing with small numbers, then fail in production with larger values.

**4. Variable Scope Surprises and the UnboundLocalError:**

Python's variable scoping rules follow the LEGB rule (Local, Enclosing, Global, Built-in), but they have a surprising twist that catches many developers. If you assign to a variable anywhere in a function, Python treats that variable as local to the entire function — even before the assignment executes. This means if a global variable x = 10 exists and your function tries to print(x) before doing x = 20, you get an UnboundLocalError, not 10. The fix is to use the "global" keyword if you genuinely need to modify a global variable (global x), or better yet, avoid global variables entirely by passing values as arguments and returning results. Similarly, in nested functions, use "nonlocal" to modify variables from the enclosing scope. The deeper lesson is that Python functions should prefer explicit parameter passing over implicit global state — it makes code more predictable, testable, and easier to reason about.

**5. String Concatenation Performance — The Hidden Quadratic:**

Building a string by repeatedly concatenating with the + operator inside a loop is a classic performance trap. Because Python strings are immutable, each concatenation creates a brand new string object and copies all the characters from both operands. In a loop that runs N times, this results in O(N squared) time complexity — concatenating 100,000 short strings can take noticeably longer than expected. The Pythonic solution is to collect all the pieces in a list and join them at the end: result = "".join(parts). This is O(N) because join calculates the total length once, allocates a single string of that size, and copies each piece exactly once. For building strings with formatting, f-strings are both readable and efficient. For writing incremental output, consider writing directly to a StringIO buffer. This is one of those cases where the Pythonic way is not just more readable but also dramatically more performant.`,
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
			ID:          2214,
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
					Content: `**Understanding Data Cleaning and Preprocessing — The Foundation of Good Data Science:**

**1. Why Data Cleaning Is the Most Important Step in Data Science:**

There is a famous saying in data science: "garbage in, garbage out." No matter how sophisticated your machine learning model or how beautiful your visualization, if the underlying data is dirty, your results will be unreliable or outright wrong. Real-world data is almost always messy — it arrives with missing values (sensors that failed, users who skipped form fields), duplicate records (systems that recorded the same event twice), inconsistent formatting (dates written as "01/15/2024", "January 15, 2024", and "2024-01-15" in the same column), outliers (a salary recorded as $1 instead of $100,000 due to a data entry error), and incorrect data types (numbers stored as strings, dates stored as plain text). Data cleaning — also called data wrangling or data munging — is the process of detecting and fixing these issues to produce a clean, consistent dataset that you can trust. By most estimates, data scientists spend 60-80% of their time on data cleaning and preprocessing, making it the most time-consuming (and arguably most valuable) part of the data science workflow.

**2. Handling Missing Data — Deciding What to Do with the Gaps:**

Missing data is perhaps the most common data quality issue, and how you handle it can significantly impact your analysis. Pandas represents missing values as NaN (Not a Number) for numeric data and None for object data. The first step is understanding the extent of the problem: df.isnull().sum() shows how many missing values exist in each column, and df.isnull().mean() shows the percentage. The simplest approach is to remove rows or columns with missing data using dropna(), but this can throw away a lot of valuable information — if 30% of your rows have a missing value in one column, dropping all of them loses nearly a third of your data. A better approach is often imputation — replacing missing values with reasonable estimates. Common strategies include filling with the column's mean or median (for numeric data), filling with the mode (for categorical data), forward-filling or backward-filling (for time series, where the previous or next value is a reasonable estimate), or using more sophisticated techniques like K-nearest neighbors imputation. The choice depends on why the data is missing and how it relates to other variables in your dataset.

**3. Handling Duplicates, Outliers, and Data Type Issues:**

Duplicate records can artificially inflate counts, skew averages, and bias machine learning models. Use df.duplicated() to identify them and df.drop_duplicates() to remove them, optionally specifying subset columns to define what constitutes a "duplicate." Outliers — data points that are dramatically different from the rest — require careful consideration. Some outliers are genuine extreme values that should be kept (a billionaire in a salary dataset), while others are errors that should be corrected or removed (a negative age). The IQR (Interquartile Range) method identifies outliers as values more than 1.5 times the IQR below Q1 or above Q3. Z-score methods flag values more than 2-3 standard deviations from the mean. Data type issues are another common headache: a column of numbers might be stored as strings because one row contains "N/A" as text, preventing mathematical operations. Use df.astype() to convert types, pd.to_numeric() with errors='coerce' to convert strings to numbers (turning unparseable values to NaN), and pd.to_datetime() to parse date strings into proper datetime objects.

**4. Feature Normalization and Categorical Encoding — Preparing Data for Models:**

Many machine learning algorithms are sensitive to the scale of input features — a feature measured in thousands (like salary) can dominate one measured in single digits (like years of experience), even if the smaller-scale feature is more important. Normalization addresses this by putting all features on a comparable scale. Min-Max scaling transforms values to a 0-1 range using (value - min) / (max - min). Standard scaling (z-score normalization) transforms values to have zero mean and unit variance. Pandas can perform these transformations directly, or you can use scikit-learn's MinMaxScaler and StandardScaler for a more formal pipeline. Categorical encoding converts non-numeric categories (like city names or product types) into numbers that machine learning algorithms can process. One-hot encoding (pd.get_dummies()) creates a binary column for each category — the most common approach that avoids implying any ordinal relationship between categories. Label encoding assigns an integer to each category — simpler but can mislead algorithms into thinking "Chicago=0 < NYC=1 < LA=2" implies an ordering. Choosing the right encoding depends on your algorithm and the nature of your categorical variables.`,
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
