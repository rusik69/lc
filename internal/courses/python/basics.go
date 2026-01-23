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
   - Use exit() or Ctrl+D to quit
   - History available (up arrow)
   - IPython provides enhanced REPL with autocomplete, syntax highlighting

2. **Script Files**
   - Save code in .py files
   - Run with: python script.py or python3 script.py
   - Full programs and applications
   - Can be imported as modules
   - Use shebang line: #!/usr/bin/env python3 for direct execution
   - Make executable: chmod +x script.py

3. **Jupyter Notebooks**
   - Interactive cells for data science
   - Mix code, markdown, and visualizations
   - Great for exploratory analysis
   - Install: pip install jupyter
   - Run: jupyter notebook or jupyter lab
   - Save as .ipynb files

**Basic Program Structure:**
- **No semicolons**: Python uses newlines to end statements
- **Indentation is syntax**: 4 spaces standard (PEP 8), tabs also work but spaces preferred
- **Comments**: # for single-line, triple quotes for multi-line
- **Docstrings**: Triple-quoted strings at module/function/class start
- **Main guard**: if __name__ == "__main__": allows script to be run or imported

**Indentation Rules:**
- Python uses indentation to define code blocks (unlike braces in C/Java)
- Consistent indentation is required (mixing tabs and spaces causes errors)
- Standard is 4 spaces per indentation level (PEP 8)
- Indentation level determines scope
- Same indentation level = same scope
- Increased indentation = new scope (function, class, if, for, etc.)

**Why Indentation Matters:**
- Makes code more readable
- Enforces consistent style
- Reduces need for braces/brackets
- Pythonic way of structuring code

**Common Pitfalls:**

**1. IndentationError:**
- Mixing tabs and spaces (Python 3 disallows this)
- Inconsistent indentation levels
- Solution: Use 4 spaces consistently, configure editor to show whitespace
- Use: python -tt script.py to detect mixed tabs/spaces

**2. SyntaxError:**
- Missing colons after if/for/def/while/class statements
- Unclosed parentheses, brackets, or quotes
- Solution: Check for colons, match brackets/quotes

**3. NameError:**
- Using variable before assignment
- Typo in variable name
- Variable not in scope
- Solution: Check variable names, ensure assignment before use

**4. ImportError:**
- Module not found
- Wrong PYTHONPATH
- Virtual environment not activated
- Solution: Check PYTHONPATH, activate venv, install package

**5. Indentation Confusion:**
- Not understanding scope
- Wrong indentation level
- Solution: Use consistent 4-space indentation, understand scope rules

**Best Practices:**

**1. Use Consistent Indentation:**
- Always use 4 spaces (PEP 8)
- Configure editor to insert spaces, not tabs
- Use python -tt to detect mixed indentation

**2. Use Main Guard:**
- Allows script to be imported or run
- Prevents code execution on import
- Enables testing and reuse

**3. Write Docstrings:**
- Document modules, functions, classes
- Use triple-quoted strings
- Follow PEP 257 conventions

**4. Follow PEP 8:**
- Python style guide
- Use linters (flake8, pylint, black)
- Format code consistently

**5. Use Type Hints (Python 3.5+):**
- Improves code clarity
- Enables static type checking
- Better IDE support

**Real-World Examples:**

**Simple Script:**
- Hello world program
- Command-line utility
- Data processing script

**Module Design:**
- Reusable code
- Can be imported
- Contains functions/classes

**Package Structure:**
- Multiple modules
- __init__.py file
- Organized codebase`,
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
					Content: `Variables are fundamental to programming - they allow us to store and manipulate data. Python's approach to variables is unique compared to many languages, emphasizing simplicity and flexibility. Understanding how variables work in Python, including their reference semantics and naming conventions, is crucial for writing effective Python code.

**Understanding Python Variables:**

**Python's Unique Approach:**

Unlike statically-typed languages like C++ or Java, Python variables don't require type declarations. This makes Python code more concise and flexible, but also requires understanding how Python handles variable assignment and references.

**Key Characteristics:**

**1. No Declaration Needed:**
- Variables are created automatically when first assigned
- No need to declare type or allocate memory explicitly
- Assignment creates the variable: x = 10 creates variable x

**2. Dynamically Typed:**
- Type is determined by the value assigned, not declared
- Python infers type from the value
- Type checking happens at runtime, not compile time

**3. Type Can Change:**
- Same variable can hold different types over time
- x = 10 (int), then x = "hello" (str) is perfectly valid
- This flexibility is powerful but requires discipline

**4. References, Not Values:**
- Variables hold references to objects, not the objects themselves
- This is crucial for understanding mutable vs immutable types
- Multiple variables can reference the same object

**Variable Assignment:**

**Simple Assignment:**
- Basic syntax: variable_name = value
- Creates variable if it doesn't exist
- Updates reference if variable exists
- Example: name = "Alice", age = 30, is_active = True

**Multiple Assignment (Tuple Unpacking):**
- Assign multiple values in one statement
- Syntax: x, y, z = 1, 2, 3
- Right side is tuple (parentheses optional)
- Number of variables must match number of values

**Advanced Unpacking:**
- Extended unpacking: x, *rest = [1, 2, 3, 4] (x=1, rest=[2,3,4])
- Ignore values: x, _, z = (1, 2, 3) (ignore middle value)
- Nested unpacking: (x, y), z = ((1, 2), 3)

**Chained Assignment:**
- Syntax: x = y = z = 0
- All variables reference the same object
- **Critical**: For mutable types, this shares the object!
- Example: x = y = z = [] means all three reference the same list

**Why References Matter:**

**Immutable Types (int, str, tuple):**
- Chained assignment is safe: x = y = 10
- Changing x doesn't affect y (creates new object)
- Each variable can be modified independently

**Mutable Types (list, dict, set):**
- Chained assignment shares object: x = y = []
- Modifying x affects y (same object)
- Solution: Create separate objects: x = [], y = []

**Naming Rules:**

**1. Must Start with Letter or Underscore:**
- Valid: name, _internal, user123
- Invalid: 123name, @variable, $price
- Underscore is considered a letter for naming

**2. Can Contain Letters, Digits, Underscores:**
- Valid: user_name, item2, MAX_SIZE
- Invalid: user-name, item.price, user@name
- No spaces allowed (use underscores)

**3. Case-Sensitive:**
- name, Name, NAME are three different variables
- Python distinguishes uppercase and lowercase
- Be consistent to avoid confusion

**4. Cannot Use Keywords:**
- Keywords are reserved words with special meaning
- Cannot use: if, for, def, class, import, etc.
- Use keyword module to check: import keyword; keyword.iskeyword('if')

**5. Cannot Use Special Characters:**
- No @, #, $, %, &, *, etc.
- Underscore (_) is the only special character allowed
- Use underscores to separate words

**Python Keywords (Complete List):**

**Control Flow:** if, elif, else, for, while, break, continue, pass

**Functions:** def, return, lambda, yield

**Classes:** class

**Modules:** import, from, as

**Exceptions:** try, except, finally, raise, assert

**Scope:** global, nonlocal

**Operators:** and, or, not, in, is

**Async:** async, await

**Constants:** True, False, None

**Other:** del, with, try

**Naming Conventions (PEP 8):**

**1. snake_case (Variables, Functions, Methods, Modules, Packages):**

**When to Use:**
- Variable names: user_name, total_count, is_valid
- Function names: calculate_total(), get_user_data()
- Method names: process_payment(), validate_input()
- Module names: user_management.py, data_processor.py
- Package names: my_package (directory name)

**Examples:**
- Good: calculate_total_price(), user_account_balance
- Bad: CalculateTotalPrice(), userAccountBalance

**2. UPPER_CASE (Constants):**

**When to Use:**
- Constants (values that shouldn't change)
- Module-level constants
- Configuration values

**Note:** Python doesn't enforce constants - this is convention only. Values can still be changed, but UPPER_CASE signals intent.

**Examples:**
- MAX_RETRIES = 3
- DEFAULT_TIMEOUT = 30
- PI = 3.14159
- API_BASE_URL = "https://api.example.com"

**3. PascalCase (Classes):**

**When to Use:**
- Class names only
- Exception classes
- Named tuples

**Examples:**
- class UserAccount:
- class DatabaseConnection:
- class ValidationError(Exception):

**4. Single Leading Underscore (_name):**

**Meaning:** "Internal use" - signals that this is implementation detail

**When to Use:**
- Internal methods/attributes not part of public API
- Signals "use at your own risk"
- Not imported by "from module import *"

**Examples:**
- _internal_method()
- _cache_data
- _validate_input()

**5. Double Leading Underscore (__name):**

**Meaning:** Name mangling - Python mangles the name

**When to Use:**
- Rarely needed
- Prevents accidental overriding in inheritance
- Python changes __name to _Class__name

**Examples:**
- __private_attribute (becomes _MyClass__private_attribute)
- Usually avoid unless you understand name mangling

**6. Trailing Underscore (name_):**

**When to Use:**
- Avoid conflict with keywords
- When keyword would be perfect name

**Examples:**
- class_ (if 'class' conflicts with keyword)
- type_ (if 'type' conflicts with built-in)
- import_ (if 'import' conflicts)

**Best Practices:**

**1. Use Descriptive Names:**
- Good: user_count, calculate_total_price, is_valid_email
- Bad: uc, calc, flag
- Names should explain purpose, not require comments

**2. Avoid Abbreviations:**
- Good: user_account, database_connection
- Bad: usr_acct, db_conn
- Exceptions: Common abbreviations (id, url, api) are acceptable

**3. Use Verbs for Functions:**
- Good: calculate_total(), process_payment(), validate_input()
- Bad: total(), payment(), input()
- Functions do things - names should reflect actions

**4. Use Nouns for Variables:**
- Good: user_name, total_price, item_count
- Bad: get_user_name, calculate_total_price, count_items()
- Variables hold values - names should reflect what they store

**5. Be Consistent:**
- Use same naming style throughout project
- Follow team/project conventions
- Consistency is more important than perfection

**6. Avoid Single Letters (Except Loop Counters):**
- Good: for i in range(10), for item in items
- Bad: x = calculate_total(), d = get_data()
- Single letters are acceptable for: loop counters (i, j, k), mathematical variables (x, y, z), generic iterators

**7. Use Boolean Naming Conventions:**
- Prefix with is_, has_, can_, should_
- Good: is_valid, has_permission, can_edit, should_retry
- Bad: valid, permission, edit, retry
- Makes boolean nature clear

**Common Naming Mistakes:**

**1. Using Keywords:**
- Bad: class = "Python 101" (class is keyword)
- Good: class_name = "Python 101" or class_ = "Python 101"

**2. Inconsistent Casing:**
- Bad: userName, user_name, UserName mixed
- Good: user_name consistently (following snake_case)

**3. Too Short:**
- Bad: x, y, z for meaningful variables
- Good: x, y, z only for mathematical/coordinate variables

**4. Too Long:**
- Bad: number_of_users_in_the_current_active_session
- Good: active_session_user_count

**5. Misleading Names:**
- Bad: list = [1, 2, 3] (shadows built-in list)
- Good: numbers = [1, 2, 3] or items = [1, 2, 3]

**6. Hungarian Notation:**
- Bad: strName, intCount, bIsValid (type in name)
- Good: name, count, is_valid (Python doesn't need type hints in names)

**Variable Scope:**

**Understanding Scope:**
- Variables have scope (where they're accessible)
- Local scope: Inside function
- Global scope: Module level
- Enclosing scope: Nested functions

**Global Variables:**
- Defined at module level
- Accessible throughout module
- Can be modified with global keyword

**Local Variables:**
- Defined inside function
- Only accessible within function
- Shadows global variables with same name

**Nonlocal Variables:**
- Used in nested functions
- Refers to variable in enclosing (non-global) scope
- Modified with nonlocal keyword

**Memory and Performance:**

**Variable Storage:**
- Variables are stored in namespaces (dictionaries)
- Local namespace: function's local variables
- Global namespace: module's global variables
- Built-in namespace: Python's built-in functions/types

**Memory Considerations:**
- Variables hold references (small, fixed size)
- Actual objects stored separately
- Multiple variables can reference same object (saves memory)

**Performance Tips:**
- Local variable access is faster than global
- Avoid creating unnecessary variables
- Reuse variables when appropriate (but don't sacrifice readability)

**Type Hints (Python 3.5+):**

**What Are Type Hints:**
- Optional annotations indicating expected types
- Don't affect runtime (Python ignores them)
- Tools like mypy can check types statically

**Syntax:**
- Basic: name: str = "Alice"
- Functions: def process(data: list[int]) -> bool:
- Variables: count: int = 0

**Benefits:**
- Better IDE support (autocomplete, error detection)
- Documentation (types are self-documenting)
- Static type checking with mypy
- Easier refactoring

**Example:**
Python code example:
def calculate_total(items: list[float], tax_rate: float = 0.08) -> float:
    """Calculate total with tax."""
    subtotal = sum(items)
    return subtotal * (1 + tax_rate)


**Real-World Examples:**

**1. Configuration Variables:**
- Use UPPER_CASE for configuration
- Example: DATABASE_URL, API_KEY, MAX_RETRIES
- Makes configuration obvious

**2. Function Parameters:**
- Use descriptive names
- Example: def create_user(username: str, email: str, is_admin: bool)
- Parameter names are part of API

**3. Loop Variables:**
- Use meaningful names when possible
- Example: for user in users, for item in shopping_cart
- Single letters acceptable for simple loops: for i in range(10)

**4. Boolean Variables:**
- Use is_/has_/can_ prefixes
- Example: is_authenticated, has_permission, can_edit
- Makes boolean nature immediately clear

**5. Temporary Variables:**
- Can use shorter names for temporary values
- Example: temp, tmp, val (but still be clear)
- Prefer descriptive names when variable is used multiple times

Understanding variables and naming in Python deeply helps you write more maintainable, readable code. Good naming is one of the most important aspects of clean code - it makes your code self-documenting and easier to understand.`,
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
					Content: `Python provides a rich set of built-in data types that form the foundation of all Python programming. Understanding these types deeply - their characteristics, behaviors, and when to use each - is essential for writing effective Python code. Python's type system is dynamic but well-designed, offering both flexibility and power.

**Python's Type System:**

**Dynamic Typing:**
- Types are determined at runtime, not compile time
- Variables don't have fixed types - they reference typed objects
- Type checking happens when operations are performed
- This flexibility enables rapid development but requires discipline

**Everything is an Object:**
- In Python, everything is an object
- Even types are objects (classes)
- This unified model simplifies the language
- Enables powerful metaprogramming capabilities

**Numeric Types:**

**1. int (Integer):**

**Unlimited Precision:**
- Python 3 integers have unlimited precision
- No overflow errors (unlike C/C++/Java)
- Can represent arbitrarily large numbers
- Only limited by available memory

**Examples:**
- Small: 42, -10, 0
- Large: 999999999999999999999999999999999
- Huge: 2**1000, 10**1000 (works perfectly!)

**Why Unlimited Precision Matters:**
- Cryptography: Need large integers for encryption
- Scientific computing: Arbitrary precision calculations
- No need to worry about integer overflow
- Simplifies many algorithms

**Integer Literals:**

**Decimal (Base 10):**
- Standard: 42, 100, -5
- With underscores: 1_000_000 (improves readability)
- Underscores ignored by Python: 1_000_000 == 1000000

**Binary (Base 2):**
- Prefix: 0b or 0B
- Example: 0b1010 = 10, 0b1111 = 15
- Useful for: Bit manipulation, flags, binary data

**Octal (Base 8):**
- Prefix: 0o or 0O
- Example: 0o755 = 493
- Useful for: File permissions (Unix), legacy systems

**Hexadecimal (Base 16):**
- Prefix: 0x or 0X
- Example: 0xFF = 255, 0x1A = 26
- Useful for: Memory addresses, color codes, low-level programming

**Integer Operations:**
- Arithmetic: +, -, *, / (returns float), // (floor division), % (modulo), ** (exponentiation)
- Bitwise: &, |, ^, ~, <<, >>
- Comparisons: ==, !=, <, >, <=, >=

**2. float (Floating-Point):**

**IEEE 754 Double Precision:**
- 64-bit floating-point numbers
- Approximately 15-17 decimal digits of precision
- Range: ~1.7e-308 to ~1.7e+308

**Precision Limitations:**
- Floats are approximations, not exact
- 0.1 + 0.2 ≠ 0.3 (due to binary representation)
- Cannot represent all decimal numbers exactly
- Important for financial calculations (use Decimal instead)

**Float Literals:**
- Standard: 3.14, -0.5, 100.0
- Scientific notation: 1.5e10 = 15000000000.0, 1e-5 = 0.00001
- Can omit leading zero: .5 = 0.5
- Can omit trailing zero: 5. = 5.0

**Special Float Values:**
- Infinity: float('inf'), float('-inf')
- Not a Number: float('nan')
- Check with: math.isinf(), math.isnan()

**Float Comparison:**
- Never use == for float comparison!
- Use: abs(a - b) < epsilon (small tolerance)
- Or: math.isclose(a, b)

**When to Use Decimal:**
- Financial calculations (money)
- When exact decimal representation needed
- Use decimal.Decimal instead of float

**3. complex (Complex Numbers):**

**Format:**
- a + bj or complex(a, b)
- j (or J) is imaginary unit (not i like in mathematics)
- Example: 3 + 4j, complex(3, 4)

**Accessing Parts:**
- .real: Real part (float)
- .imag: Imaginary part (float)
- abs(): Magnitude (distance from origin)

**Operations:**
- Arithmetic: +, -, *, /, **
- Conjugate: .conjugate()
- Magnitude: abs() or math.sqrt(real² + imag²)

**Use Cases:**
- Scientific computing (physics, engineering)
- Signal processing (Fourier transforms)
- Graphics (2D rotations)
- Quantum computing simulations

**Text Type:**

**str (String):**

**Immutable Unicode Sequences:**
- Strings are immutable (cannot be modified in place)
- Sequence of Unicode characters (not bytes)
- Supports all Unicode characters (emojis, Chinese, Arabic, etc.)

**String Literals:**

**Single Quotes:**
- 'Hello, World!'
- Use when string contains double quotes
- Example: 'He said "Hello"'

**Double Quotes:**
- "Hello, World!"
- Use when string contains single quotes
- Example: "It's a beautiful day"

**Triple Quotes (Multi-line):**
- """Multi-line string""" or '''Multi-line string'''
- Preserves newlines and formatting
- Useful for: Docstrings, multi-line text, SQL queries

**Raw Strings:**
- r"C:\\path" or r'C:\path'
- Backslashes are literal (not escape characters)
- Useful for: File paths, regex patterns, Windows paths

**f-strings (Formatted String Literals, Python 3.6+):**
- f"Hello {name}, you are {age} years old"
- Evaluates expressions inside {}
- Fastest string formatting method
- Most readable and recommended

**String Operations:**
- Concatenation: "hello" + "world"
- Repetition: "hello" * 3 = "hellohellohello"
- Indexing: s[0], s[-1] (first, last character)
- Slicing: s[1:4], s[:5], s[3:]
- Membership: 'a' in "hello", 'x' not in "hello"

**String Methods:**
- .upper(), .lower(), .title(), .capitalize()
- .strip(), .lstrip(), .rstrip()
- .split(), .join()
- .find(), .index(), .replace()
- .startswith(), .endswith()
- .isalpha(), .isdigit(), .isalnum()

**Boolean Type:**

**bool:**

**Only Two Values:**
- True (capitalized!)
- False (capitalized!)
- Not true/false (lowercase doesn't work)

**Subclass of int:**
- bool is subclass of int
- True == 1, False == 0
- Can use in arithmetic: True + True = 2
- But use booleans for boolean logic, not arithmetic

**Truthiness:**

**All Values Are Truthy or Falsy:**
- Every value can be evaluated as True or False
- Used in if statements, while loops, boolean operations

**Falsy Values (Evaluate to False):**
- False itself
- None
- Zero: 0, 0.0, 0j
- Empty sequences: "", [], (), {}
- Empty sets: set()
- Custom classes can define __bool__() or __len__()

**Truthy Values (Evaluate to True):**
- Everything else!
- Non-empty strings, lists, dicts, etc.
- Non-zero numbers
- Most objects

**Common Mistakes:**
- if x == True: (redundant, use if x:)
- if x is True: (checks identity, usually not needed)
- if x: (preferred - uses truthiness)

**None Type:**

**None:**

**Absence of Value:**
- Represents "no value" or "null"
- Only one None object exists (singleton)
- All None references point to same object

**Use Cases:**
- Default return value for functions that don't return anything
- Default parameter values (when None means "not provided")
- Placeholder for optional values
- Signal "no result" or "not found"

**Comparison:**
- Use is None or is not None (not ==)
- None is None: True (identity check)
- None == None: True (but use is preferred)
- Why? is checks identity, == checks equality

**Type Checking:**

**type() Function:**
- Returns exact type of object
- Returns type object: <class 'int'>
- Less flexible (doesn't handle inheritance)

**isinstance() Function (Preferred):**
- Checks if object is instance of type or subclass
- More flexible: isinstance(x, (int, float))
- Handles inheritance correctly
- Recommended for type checking

**Example:**
- type(x) == int: Checks exact type
- isinstance(x, int): Checks type or subclass (better)

**Type Hints (Python 3.5+):**

**What Are Type Hints:**
- Optional annotations indicating expected types
- Don't affect runtime behavior
- Tools like mypy can check types statically
- Improve IDE support and documentation

**Basic Syntax:**
- Variables: name: str = "Alice"
- Functions: def process(data: list[int]) -> bool:
- Multiple types: value: int | float = 0 (Python 3.10+)

**Benefits:**
- Better IDE autocomplete and error detection
- Self-documenting code
- Static type checking with mypy
- Easier refactoring and maintenance

**Real-World Type Usage:**

**1. Choosing Numeric Types:**
- int: Whole numbers, counts, indices
- float: Measurements, calculations, scientific data
- Decimal: Money, exact decimal arithmetic
- complex: Scientific computing, signal processing

**2. String Best Practices:**
- Use f-strings for formatting (Python 3.6+)
- Use raw strings for paths and regex
- Use triple quotes for multi-line text
- Be aware of Unicode vs bytes distinction

**3. Boolean Logic:**
- Use truthiness naturally: if items: (not if len(items) > 0:)
- Be explicit when needed: if x is not None:
- Use boolean names: is_valid, has_permission

**4. None Handling:**
- Check with is None (not == None)
- Use None as default for optional parameters
- Return None to indicate "no result"

**Performance Considerations:**

**Integer Operations:**
- Very fast (hardware-accelerated)
- Unlimited precision has some overhead for large numbers
- Still fast enough for most use cases

**Float Operations:**
- Fast but beware precision issues
- Use Decimal for exact decimal arithmetic
- Consider performance vs precision tradeoff

**String Operations:**
- Immutability means operations create new strings
- String concatenation can be slow (use join() for many strings)
- f-strings are fastest formatting method

**Type Conversion Overhead:**
- Converting between types has cost
- Avoid unnecessary conversions
- Cache converted values when reused

Understanding Python's basic types deeply helps you write more efficient, correct code. Each type has its strengths and use cases - choosing the right type for your data is an important skill.`,
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
					Content: `Type conversion (also called type casting) is the process of converting a value from one data type to another. Python provides built-in functions for explicit type conversion and performs implicit conversions automatically in certain situations. Understanding type conversion is crucial for working with user input, data processing, and ensuring your code handles different data types correctly.

**Why Type Conversion Matters:**

**1. User Input Handling:**
- User input from input() is always a string
- Need to convert to numbers for calculations
- Example: age = int(input("Enter age: "))

**2. Data Processing:**
- APIs return strings, need numbers for calculations
- Database values may be strings, need conversion
- File reading returns strings, need type conversion

**3. Type Safety:**
- Ensure operations work with correct types
- Prevent type-related errors
- Make code more robust and predictable

**4. Interoperability:**
- Convert between Python types and external formats (JSON, XML)
- Interface with C libraries (ctypes)
- Work with databases (SQL types)

**Explicit Type Conversion Functions:**

**1. int() - Convert to Integer:**

**Basic Usage:**
- Converts numbers (float, complex) to integer
- Truncates decimal part (doesn't round)
- Converts strings containing valid integers
- Can specify base for string conversion (binary, octal, hex)

**Examples:**
- int(3.14) → 3 (truncates, not rounds)
- int(3.9) → 3 (still truncates)
- int("42") → 42 (string to int)
- int("1010", 2) → 10 (binary string to int)
- int("FF", 16) → 255 (hex string to int)

**When It Fails:**
- ValueError: Invalid string (e.g., int("hello"))
- ValueError: Float string without conversion (e.g., int("3.14"))
- TypeError: Cannot convert complex numbers directly

**Real-World Use Cases:**
- Converting user input: age = int(input("Age: "))
- Parsing command-line arguments: port = int(sys.argv[1])
- Converting database IDs: user_id = int(db_result)
- Processing numeric strings from files/APIs

**2. float() - Convert to Float:**

**Basic Usage:**
- Converts integers to floats
- Converts strings containing valid float representations
- Handles scientific notation
- Returns float even from integer input

**Examples:**
- float(42) → 42.0
- float("3.14") → 3.14
- float("1e-3") → 0.001 (scientific notation)
- float("inf") → inf (infinity)
- float("nan") → nan (not a number)

**When It Fails:**
- ValueError: Invalid string format
- TypeError: Cannot convert complex numbers directly

**Real-World Use Cases:**
- Converting prices: price = float(input("Price: "))
- Parsing scientific data: value = float(data_line.split()[1])
- Converting percentages: rate = float("12.5")
- Processing measurements: weight = float(measurement_string)

**3. str() - Convert to String:**

**Basic Usage:**
- Converts any Python object to string representation
- Uses object's __str__() method
- Always succeeds (everything has string representation)
- Creates human-readable representation

**Examples:**
- str(42) → "42"
- str(3.14) → "3.14"
- str([1, 2, 3]) → "[1, 2, 3]"
- str(None) → "None"
- str(True) → "True"

**Real-World Use Cases:**
- Formatting output: print("Value: " + str(value))
- Logging: logger.info("Processing " + str(count) + " items")
- Building messages: message = "User " + str(user_id) + " logged in"
- File writing: file.write(str(data))

**4. bool() - Convert to Boolean:**

**Basic Usage:**
- Converts any value to True or False
- Uses Python's truthiness rules
- Only False, None, 0, empty collections are False
- Everything else is True

**Truthiness Rules:**
- False: False, None, 0, 0.0, 0j, "", (), [], {}, set()
- True: Everything else (including negative numbers, non-empty strings, etc.)

**Examples:**
- bool(1) → True
- bool(0) → False
- bool("") → False
- bool("hello") → True
- bool([]) → False
- bool([1, 2]) → True
- bool(None) → False

**Real-World Use Cases:**
- Validating input: if bool(user_input): process()
- Checking if list is empty: if bool(items): process_items()
- Converting to boolean for APIs: send_data({"active": bool(status)})

**5. Complex Type Conversions:**

**Converting to Complex:**
- complex(3, 4) → (3+4j)
- complex("3+4j") → (3+4j)
- Used in scientific computing, signal processing

**Converting Collections:**
- list("hello") → ['h', 'e', 'l', 'l', 'o']
- tuple([1, 2, 3]) → (1, 2, 3)
- set([1, 2, 2, 3]) → {1, 2, 3}
- dict([('a', 1), ('b', 2)]) → {'a': 1, 'b': 2}

**Implicit Type Conversion (Coercion):**

**When Python Converts Automatically:**
- Arithmetic operations: int + float = float
- Comparisons: Can compare int and float directly
- Mixed numeric types in expressions

**Examples:**
- 5 + 3.14 → 8.14 (int promoted to float)
- 10 / 2 → 5.0 (division always returns float in Python 3)
- 10 // 2 → 5 (floor division returns int)
- 2 ** 3.0 → 8.0 (exponentiation promotes to float)

**Important Notes:**
- Python 3: / always returns float (use // for integer division)
- Python 2: / returns int for integer operands (different behavior)
- Mixed arithmetic promotes to "higher" type (int → float → complex)

**Common Conversion Patterns:**

**1. String to Number:**
Python code example:
# Safe conversion with error handling
def safe_int(s, default=0):
    try:
        return int(s)
    except ValueError:
        return default

age = safe_int(input("Age: "), 0)

**2. Number to String Formatting:**
Python code example:
# Using f-strings (Python 3.6+)
value = 42
message = f"Value: {value}"

# Using format()
message = "Value: {}".format(value)

# Using % formatting (older style)
message = "Value: %d" % value

**3. Handling Invalid Conversions:**
Python code example:
# Always handle conversion errors
try:
    number = int(user_input)
except ValueError:
    print("Invalid number!")
    number = None

**Best Practices:**

**1. Always Validate Input:**
- Don't assume conversion will succeed
- Use try-except for user input
- Provide meaningful error messages

**2. Be Explicit:**
- Prefer explicit conversion over relying on implicit
- Makes code more readable and predictable
- Easier to debug type-related issues

**3. Handle Edge Cases:**
- Empty strings: int("") raises ValueError
- None values: int(None) raises TypeError
- Whitespace: int("  42  ") works (strips whitespace)
- Scientific notation: float("1e10") works

**4. Use Appropriate Types:**
- Don't convert unnecessarily
- Use int for whole numbers, float for decimals
- Consider using Decimal for financial calculations (precision)

**5. Performance Considerations:**
- Type conversion has overhead
- Avoid converting in tight loops if possible
- Cache converted values when reused

**Common Pitfalls:**

**1. Assuming Rounding:**
- int(3.9) is 3, not 4 (truncates, doesn't round)
- Use round() first if rounding needed: int(round(3.9)) → 4

**2. String to Float Issues:**
- int("3.14") raises ValueError (use float() first)
- Need two-step: int(float("3.14")) → 3

**3. Truthiness Confusion:**
- bool("False") is True (non-empty string)
- bool(0.0) is False (zero is falsy)
- Check actual values, not just truthiness

**4. Implicit Conversion Surprises:**
- 10 / 2 is 5.0 in Python 3 (not 5)
- Use // for integer division: 10 // 2 → 5

**5. None Handling:**
- int(None) raises TypeError
- Check for None before converting
- Use: value = int(x) if x is not None else 0

**Advanced Techniques:**

**1. Custom Conversion Functions:**
Python code example:
def to_int_safe(value, default=0):
    """Safely convert to int with default."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

**2. Type Checking Before Conversion:**
Python code example:
if isinstance(value, str) and value.isdigit():
    number = int(value)
else:
    number = None

**3. Converting with Validation:**
Python code example:
def convert_percentage(s):
    """Convert string percentage to float."""
    s = s.strip().rstrip('%')
    value = float(s)
    if 0 <= value <= 100:
        return value / 100
    raise ValueError("Percentage out of range")

**Real-World Examples:**

**1. Processing CSV Data:**
Python code example:
# CSV values are strings, need conversion
for row in csv_reader:
    name = row[0]  # Already string
    age = int(row[1])  # Convert to int
    salary = float(row[2])  # Convert to float

**2. API Response Processing:**
Python code example:
# JSON returns strings for numbers sometimes
response = api.get_data()
user_id = int(response['id'])  # Convert string to int
price = float(response['price'])  # Convert to float

**3. Command-Line Arguments:**
Python code example:
import sys
# sys.argv are always strings
port = int(sys.argv[1])  # Convert to int
timeout = float(sys.argv[2])  # Convert to float

**4. Database Value Handling:**
Python code example:
# Database may return strings or numbers
result = db.fetchone()
user_id = int(result[0]) if isinstance(result[0], str) else result[0]
created_at = datetime.fromisoformat(str(result[1]))

**Performance Considerations:**

**Conversion Speed:**
- int() and float() are fast for valid inputs
- String parsing has overhead (especially for invalid strings)
- Type checking before conversion can be slower than try-except
- Cache conversions when values are reused

**Memory Considerations:**
- Converting creates new objects (Python objects are immutable for numbers)
- String conversion creates new string objects
- Consider if conversion is necessary or if you can work with original type

**Type Conversion in Different Contexts:**

**1. Mathematical Operations:**
- Automatic promotion in arithmetic
- Be aware of precision loss (int → float)
- Use Decimal for financial calculations

**2. String Formatting:**
- f-strings automatically convert: f"{number}"
- format() handles conversion
- % formatting requires matching types

**3. JSON Serialization:**
- JSON only supports: str, int, float, bool, None, list, dict
- Custom objects need conversion
- Use json.dumps() which handles conversion

**4. Database Operations:**
- ORMs handle conversion automatically
- Raw SQL may require explicit conversion
- Database drivers convert Python types to SQL types

Understanding type conversion deeply helps you write more robust Python code that handles different data sources, user input, and external APIs correctly. Always validate and handle conversion errors gracefully.`,
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
					Content: `Comments and docstrings are essential tools for writing maintainable, understandable Python code. While both serve to document code, they have different purposes and conventions. Understanding when and how to use each effectively is a hallmark of professional Python development.

**The Philosophy of Documentation:**

Good documentation makes code self-explanatory. As the famous saying goes: "Code is read more often than it's written." Well-documented code reduces cognitive load, helps new team members understand the codebase quickly, and prevents bugs by clarifying intent.

**Comments - Explaining the "Why" and "How":**

**What Are Comments?**
Comments are lines in your code that are ignored by the Python interpreter. They exist solely for human readers to understand the code's purpose, logic, or implementation details.

**Types of Comments:**

**1. Single-Line Comments:**
- Start with # symbol
- Everything after # on that line is ignored
- Can be on its own line or at end of code line
- Most common type of comment

**2. Inline Comments:**
- Comments on the same line as code
- Should be used sparingly
- Keep them short and relevant
- Place at least 2 spaces after code, then #

**3. Block Comments:**
- Multiple lines, each starting with #
- Used for longer explanations
- Should explain the "why" behind code
- Can describe algorithms or complex logic

**When to Use Comments:**

**1. Explain Complex Logic:**
- Algorithms that aren't immediately obvious
- Business rules that might seem counterintuitive
- Workarounds for bugs or limitations
- Performance optimizations that sacrifice readability

**2. Document "Why" Not "What":**
- Don't state the obvious: # Increment counter (bad)
- Do explain reasoning: # Use binary search because data is sorted (good)
- Explain business decisions: # Round down to prevent overcharging
- Document non-obvious choices: # Use list instead of set to preserve order

**3. Mark TODOs and FIXMEs:**
- Mark incomplete work: # TODO: Add error handling
- Note known issues: # FIXME: This breaks with negative numbers
- Reference tickets: # See JIRA-1234 for context
- Note temporary solutions: # HACK: Remove after API v2 is stable

**4. Section Headers:**
- Organize large files: # ==== Data Processing ====
- Mark logical sections: # Authentication Logic
- Help navigate complex files

**Comment Best Practices:**

**1. Keep Comments Up-to-Date:**
- Outdated comments are worse than no comments
- Update comments when code changes
- Remove comments for code that no longer exists
- Review comments during code reviews

**2. Write Clear, Concise Comments:**
- Use proper grammar and spelling
- Be concise but complete
- Avoid abbreviations unless standard
- Write in the same language as codebase (usually English)

**3. Don't Over-Comment:**
- Self-documenting code is better than excessive comments
- If code needs many comments, consider refactoring
- Comments shouldn't repeat what code obviously does
- Let code speak for itself when possible

**4. Use Comments for Context:**
- Explain historical context: # This was added for GDPR compliance
- Note dependencies: # Requires numpy >= 1.20.0
- Document assumptions: # Assumes input is already validated
- Explain performance characteristics: # O(n log n) due to sorting

**Docstrings - Documenting the "What":**

**What Are Docstrings?**
Docstrings are triple-quoted strings (""" or ''') placed as the first statement in a module, function, class, or method. They serve as official documentation that can be accessed programmatically and are used by documentation generators and IDEs.

**Why Docstrings Matter:**
- Accessible via __doc__ attribute
- Used by help() function
- Integrated into IDEs for autocomplete hints
- Can be extracted by documentation tools (Sphinx, pydoc)
- Follow standard conventions (PEP 257)

**Types of Docstrings:**

**1. One-Line Docstrings:**
- Brief summary on single line
- Should fit on one line (72-79 characters)
- No period needed if it's a phrase
- Use for simple functions

**2. Multi-Line Docstrings:**
- Summary line (like one-liner)
- Blank line
- Detailed description
- Sections for Args, Returns, Raises, Examples
- Follow Google, NumPy, or reStructuredText style

**Docstring Conventions (PEP 257):**

**1. Summary Line:**
- First line should be a concise summary
- Stand alone as a description
- Use imperative mood: "Return" not "Returns"
- End with period

**2. Detailed Description:**
- Blank line after summary
- More detailed explanation
- Explain behavior, edge cases, usage

**3. Sections (for functions/methods):**
- **Args/Parameters**: List each parameter with type and description
- **Returns**: Describe return value and type
- **Raises**: List exceptions that might be raised
- **Examples**: Show usage examples
- **Note**: Additional important information
- **See Also**: Related functions/classes

**Docstring Styles:**

**1. Google Style (Popular):**
Python code example:
def calculate_total(items, tax_rate=0.08):
    """Calculate total price including tax.
    
    Args:
        items: List of item prices (floats).
        tax_rate: Tax rate as decimal (default 0.08).
    
    Returns:
        Total price as float.
    
    Raises:
        ValueError: If tax_rate is negative.
    
    Example:
        >>> calculate_total([10.0, 20.0], 0.1)
        33.0
    """


**2. NumPy Style:**
Python code example:
def calculate_total(items, tax_rate=0.08):
    """Calculate total price including tax.
    
    Parameters
    ----------
    items : list of float
        List of item prices.
    tax_rate : float, optional
        Tax rate as decimal (default is 0.08).
    
    Returns
    -------
    float
        Total price including tax.
    
    Raises
    ------
    ValueError
        If tax_rate is negative.
    """


**3. reStructuredText Style:**
Python code example:
def calculate_total(items, tax_rate=0.08):
    """Calculate total price including tax.
    
    :param items: List of item prices
    :type items: list of float
    :param tax_rate: Tax rate as decimal
    :type tax_rate: float
    :returns: Total price including tax
    :rtype: float
    :raises: ValueError if tax_rate is negative
    """


**When to Use Docstrings:**

**1. All Public Functions:**
- Functions meant to be used by others
- Part of your API/interface
- Should have complete docstrings

**2. All Classes:**
- Document class purpose and usage
- Document public methods
- Document class-level attributes if significant

**3. Modules:**
- Module-level docstring explains module purpose
- Lists main classes/functions
- Provides usage examples

**4. Complex Private Functions:**
- Even private functions benefit from docstrings
- Helps maintainers understand code
- Can be less formal than public API docs

**Docstring Best Practices:**

**1. Write for Users, Not Implementers:**
- Focus on how to use, not how it's implemented
- Describe behavior, not implementation details
- Provide examples of common use cases

**2. Keep Docstrings Current:**
- Update when function signature changes
- Update when behavior changes
- Remove outdated information

**3. Include Examples:**
- Show typical usage
- Show edge cases
- Use doctest format for testable examples

**4. Be Consistent:**
- Choose one style and stick with it
- Follow project conventions
- Use same format throughout codebase

**5. Document Edge Cases:**
- What happens with None?
- What about empty inputs?
- Behavior with invalid inputs?
- Side effects or state changes?

**Accessing Documentation:**

**1. __doc__ Attribute:**
Python code example:
print(calculate_total.__doc__)


**2. help() Function:**
Python code example:
help(calculate_total)


**3. IDE Integration:**
- Hover over function name
- Autocomplete shows docstring
- Documentation panels

**4. Documentation Generators:**
- Sphinx: Generates HTML/PDF docs
- pydoc: Command-line documentation
- AutoAPI: Auto-generate from docstrings

**Common Patterns:**

**1. Function Docstrings:**
Python code example:
def process_data(data, validate=True):
    """Process input data with optional validation.
    
    This function takes raw data and processes it according to
    business rules. Validation can be skipped for performance
    when data is known to be valid.
    
    Args:
        data: Raw data dictionary.
        validate: Whether to validate data (default True).
    
    Returns:
        Processed data dictionary.
    
    Raises:
        ValueError: If validation fails and validate=True.
        TypeError: If data is not a dictionary.
    
    Example:
        >>> data = {"name": "John", "age": 30}
        >>> process_data(data)
        {"name": "John", "age": 30, "processed": True}
    """


**2. Class Docstrings:**
Python code example:
class UserManager:
    """Manages user operations and authentication.
    
    This class handles user creation, authentication, and
    session management. It provides a high-level interface
    for user-related operations.
    
    Attributes:
        users: Dictionary mapping usernames to User objects.
        active_sessions: Set of active session IDs.
    
    Example:
        >>> manager = UserManager()
        >>> manager.create_user("alice", "password123")
        >>> manager.authenticate("alice", "password123")
        True
    """


**3. Module Docstrings:**
Python code example:
"""User management module.

This module provides classes and functions for managing users,
authentication, and authorization. It integrates with the
database layer and provides a clean API for user operations.

Classes:
    User: Represents a single user.
    UserManager: Manages user operations.

Functions:
    hash_password: Hash password securely.
    validate_email: Validate email format.

Example:
    >>> from user_management import UserManager
    >>> manager = UserManager()
    >>> manager.create_user("alice", "password")
"""


**Comments vs Docstrings - When to Use What:**

**Use Comments For:**
- Explaining complex algorithms or logic
- Documenting "why" decisions were made
- Marking TODOs and temporary code
- Explaining non-obvious implementation details
- Providing context for future maintainers

**Use Docstrings For:**
- Documenting what functions/classes do
- Describing parameters and return values
- Providing usage examples
- Creating API documentation
- Explaining public interfaces

**Anti-Patterns to Avoid:**

**1. Commenting Obvious Code:**
Python code example:
# Bad
x = x + 1  # Increment x by 1

# Good
x += 1  # Account for off-by-one in indexing


**2. Outdated Comments:**
Python code example:
# Bad - comment doesn't match code
# Sort the list
data.reverse()  # Actually reverses now!

# Good - update or remove
data.reverse()  # Reverse to process in LIFO order


**3. Missing Docstrings:**
Python code example:
# Bad
def calculate(x, y):
    return x * y + 10

# Good
def calculate(x, y):
    """Calculate value using formula: x * y + 10.
    
    Args:
        x: First operand.
        y: Second operand.
    
    Returns:
        Calculated result.
    """
    return x * y + 10


**4. Docstrings That Repeat Code:**
Python code example:
# Bad
def add(a, b):
    """Add a and b."""
    return a + b

# Good
def add(a, b):
    """Add two numbers, handling integer overflow.
    
    This function adds two numbers and automatically
    promotes to float if result exceeds integer range.
    """
    return a + b


**Tools and Linting:**

**1. pylint:**
- Checks for missing docstrings
- Enforces docstring conventions
- Configurable rules

**2. pydocstyle (pep257):**
- Checks PEP 257 compliance
- Validates docstring format
- Can be integrated into CI/CD

**3. Sphinx:**
- Generates documentation from docstrings
- Supports multiple docstring styles
- Creates professional documentation

**Real-World Examples:**

**1. Well-Documented Function:**
Python code example:
def binary_search(arr, target):
    """Search for target in sorted array using binary search.
    
    This function implements the classic binary search algorithm
    to find the index of target in a sorted array. The array
    must be sorted in ascending order for correct results.
    
    Args:
        arr: Sorted list of comparable elements.
        target: Element to search for.
    
    Returns:
        Index of target if found, -1 otherwise.
    
    Raises:
        ValueError: If array is not sorted.
    
    Time Complexity:
        O(log n) where n is array length.
    
    Example:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
        >>> binary_search([1, 3, 5, 7, 9], 4)
        -1
    """


**2. Well-Commented Algorithm:**
Python code example:
def quicksort(arr):
    """Sort array using quicksort algorithm."""
    # Base case: arrays with 0 or 1 elements are already sorted
    if len(arr) <= 1:
        return arr
    
    # Choose pivot (median of first, middle, last for better performance)
    # This helps avoid worst-case O(n²) performance on sorted arrays
    pivot_idx = len(arr) // 2
    pivot = arr[pivot_idx]
    
    # Partition: elements < pivot go left, > pivot go right
    # This is the key step that makes quicksort efficient
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Recursively sort partitions and combine
    # The recursion depth is O(log n) on average
    return quicksort(left) + middle + quicksort(right)


Effective use of comments and docstrings makes your code more maintainable, easier to understand, and more professional. Invest time in good documentation - it pays dividends in reduced bugs and faster onboarding of new team members.`,
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
					Content: `Python provides three control flow statements that give you fine-grained control over loop execution and code structure: break, continue, and pass. Understanding when and how to use these statements effectively is crucial for writing clean, efficient Python code.

**Break - Exiting Loops Early:**

**What Break Does:**
The break statement immediately terminates the innermost loop (for or while) and transfers control to the statement immediately following the loop. It's Python's way of saying "I'm done with this loop, exit now."

**Basic Usage:**
- Exits only the innermost loop
- Skips all remaining iterations
- Control passes to code after the loop
- Commonly used with conditional statements

**When to Use Break:**

**1. Search Operations:**
- Stop searching once target is found
- Avoid unnecessary iterations
- Improve performance by early exit
- Example: Finding first matching item

**2. Input Validation:**
- Exit loop when valid input received
- Common in interactive programs
- Example: Retry until valid input

**3. Error Conditions:**
- Exit when error detected
- Prevent further processing
- Example: Stop processing on invalid data

**4. Performance Optimization:**
- Exit early when condition met
- Avoid processing unnecessary items
- Example: Stop when threshold reached

**Break Examples:**

**1. Finding First Match:**
Python code example:
# Search for first even number
numbers = [1, 3, 5, 7, 8, 9, 11]
first_even = None
for num in numbers:
    if num % 2 == 0:
        first_even = num
        break  # Found it, no need to continue
print(first_even)  # 8


**2. Input Validation Loop:**
Python code example:
# Keep asking until valid input
while True:
    age = input("Enter your age: ")
    try:
        age = int(age)
        if 0 <= age <= 120:
            break  # Valid input, exit loop
        else:
            print("Age must be between 0 and 120")
    except ValueError:
        print("Please enter a valid number")
print(f"Your age is {age}")


**3. Processing Until Condition:**
Python code example:
# Process items until sum exceeds threshold
items = [10, 20, 30, 40, 50]
total = 0
processed = []
for item in items:
    total += item
    processed.append(item)
    if total > 50:
        break  # Stop when threshold exceeded
print(processed)  # [10, 20, 30]


**4. Nested Loops:**
Python code example:
# Break only exits innermost loop
for i in range(3):
    for j in range(5):
        if j == 2:
            break  # Only breaks inner loop
        print(f"i={i}, j={j}")
# Output: i=0,j=0 i=0,j=1 i=1,j=0 i=1,j=1 i=2,j=0 i=2,j=1


**Breaking Out of Nested Loops:**

**Method 1: Using Flags (Traditional):**
Python code example:
found = False
for i in range(3):
    for j in range(3):
        if some_condition(i, j):
            found = True
            break
    if found:
        break


**Method 2: Using Exception (Less Common):**
Python code example:
class BreakOuterLoop(Exception):
    pass

try:
    for i in range(3):
        for j in range(3):
            if some_condition(i, j):
                raise BreakOuterLoop
except BreakOuterLoop:
    pass


**Method 3: Using Functions (Cleanest):**
Python code example:
def find_item():
    for i in range(3):
        for j in range(3):
            if some_condition(i, j):
                return i, j  # Exits all loops
    return None

result = find_item()


**Method 4: Using Labels (Python 3.10+):**
Python code example:
# Python 3.10+ supports breaking to labels
for i in range(3):
    for j in range(3):
        if some_condition(i, j):
            break  # Still only breaks inner loop
    else:
        continue  # Only executed if inner loop didn't break
    break  # Break outer loop


**Continue - Skipping Iterations:**

**What Continue Does:**
The continue statement skips the rest of the current loop iteration and immediately starts the next iteration. It's Python's way of saying "skip this one, move to the next."

**Basic Usage:**
- Skips remaining code in current iteration
- Starts next iteration immediately
- Only affects innermost loop
- Commonly used for filtering

**When to Use Continue:**

**1. Filtering Data:**
- Skip items that don't meet criteria
- Process only valid items
- Cleaner than nested if statements
- Example: Process only non-empty strings

**2. Error Handling:**
- Skip items that cause errors
- Continue processing remaining items
- Log errors but don't stop
- Example: Process valid files, skip invalid

**3. Early Validation:**
- Skip items that fail validation
- Avoid deep nesting
- Improve readability
- Example: Skip invalid user input

**4. Performance:**
- Skip expensive operations when not needed
- Early exit from iteration logic
- Example: Skip processing for already-processed items

**Continue Examples:**

**1. Filtering Invalid Data:**
Python code example:
# Process only positive numbers
numbers = [5, -2, 8, -1, 10, -3]
positive_sum = 0
for num in numbers:
    if num <= 0:
        continue  # Skip negative numbers
    positive_sum += num
print(positive_sum)  # 23


**2. Processing Valid Items:**
Python code example:
# Process only valid email addresses
emails = ["alice@example.com", "invalid", "bob@test.com", "bad@"]
valid_emails = []
for email in emails:
    if "@" not in email or "." not in email.split("@")[1]:
        continue  # Skip invalid emails
    valid_emails.append(email)
print(valid_emails)  # ['alice@example.com', 'bob@test.com']


**3. Error Handling in Loops:**
Python code example:
# Process files, skip ones that fail
files = ["file1.txt", "file2.txt", "file3.txt"]
results = []
for filename in files:
    try:
        with open(filename) as f:
            data = f.read()
            results.append(process(data))
    except FileNotFoundError:
        continue  # Skip missing files, continue with others
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue  # Log error but continue


**4. Avoiding Deep Nesting:**
Python code example:
# Without continue (nested)
for item in items:
    if item.is_valid():
        if item.is_ready():
            if item.has_permission():
                process(item)

# With continue (flatter)
for item in items:
    if not item.is_valid():
        continue
    if not item.is_ready():
        continue
    if not item.has_permission():
        continue
    process(item)  # Only reached if all checks pass


**Continue vs. If-Else:**

**When Continue is Better:**
- Multiple conditions to skip
- Reduces nesting levels
- Makes "happy path" clearer
- Early validation checks

**When If-Else is Better:**
- Simple single condition
- Both branches need processing
- More readable for simple cases

**Pass - The Placeholder Statement:**

**What Pass Does:**
The pass statement is a null operation - it does nothing when executed. It's a placeholder that satisfies Python's syntax requirement that certain code blocks cannot be empty.

**Basic Usage:**
- Syntactically required but does nothing
- Used where code is required but action isn't
- Common in stubs, placeholders, minimal implementations
- Allows code to run without errors

**When to Use Pass:**

**1. Placeholder Functions:**
- Function signature defined but not implemented
- To be implemented later
- Allows code to run without errors
- Example: Stub functions for testing

**2. Empty Classes:**
- Class structure defined but no methods yet
- Minimal class definition
- Placeholder for future implementation
- Example: Exception classes, data classes

**3. Exception Handlers:**
- Catch exception but take no action
- Suppress specific errors intentionally
- Logging handled elsewhere
- Example: Ignoring expected errors

**4. Conditional Blocks:**
- Empty if/else blocks
- Placeholder for future logic
- Maintains code structure
- Example: Future feature placeholders

**Pass Examples:**

**1. Placeholder Functions:**
Python code example:
# Function to be implemented later
def process_data(data):
    """Process data - implementation pending."""
    pass  # TODO: Implement data processing

# Allows code to run
result = process_data(my_data)  # Returns None but doesn't error


**2. Empty Classes:**
Python code example:
# Base class with no implementation
class BaseProcessor:
    """Base class for data processors."""
    pass  # To be subclassed

# Custom exception
class ValidationError(Exception):
    """Raised when validation fails."""
    pass  # Inherits Exception behavior, no custom logic needed


**3. Exception Handling:**
Python code example:
# Suppress specific expected errors
try:
    result = risky_operation()
except ExpectedError:
    pass  # Ignore this specific error, continue execution
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise  # Re-raise unexpected errors


**4. Conditional Placeholders:**
Python code example:
# Placeholder for future feature
if feature_enabled:
    # TODO: Implement new feature
    pass
else:
    use_old_feature()


**Pass vs. Other Approaches:**

**Pass vs. Comment:**
Python code example:
# Pass - syntactically valid
def function():
    pass

# Comment - syntax error!
def function():
    # To be implemented


**Pass vs. Return None:**
Python code example:
# Pass - implicit None return
def function():
    pass  # Returns None

# Explicit None - clearer intent
def function():
    return None  # More explicit


**Pass vs. Ellipsis:**
Python code example:
# Pass - does nothing
def function():
    pass

# Ellipsis - also valid placeholder
def function():
    ...  # Also does nothing, sometimes used for same purpose


**Best Practices:**

**1. Use Break Sparingly:**
- Consider if loop structure can be improved
- Sometimes refactoring is better than break
- Document why break is necessary
- Prefer for early exit patterns

**2. Use Continue for Clarity:**
- Reduces nesting
- Makes "happy path" obvious
- Better than deeply nested if statements
- Use when filtering is clearer than processing

**3. Use Pass Intentionally:**
- Not a substitute for implementation
- Mark with TODO comments
- Use in stubs and placeholders
- Remove when implementing

**4. Avoid Excessive Use:**
- Too many breaks/continues can indicate design issues
- Consider refactoring complex loops
- Extract logic into functions
- Use list comprehensions when filtering

**Common Patterns:**

**1. Search Pattern (Break):**
Python code example:
def find_first_match(items, predicate):
    for item in items:
        if predicate(item):
            return item  # Early return instead of break
    return None


**2. Filter Pattern (Continue):**
Python code example:
# Using continue
valid_items = []
for item in items:
    if not is_valid(item):
        continue
    valid_items.append(process(item))

# Using list comprehension (often better)
valid_items = [process(item) for item in items if is_valid(item)]


**3. Stub Pattern (Pass):**
Python code example:
class API:
    def authenticate(self, credentials):
        """Authenticate user - to be implemented."""
        pass  # TODO: Implement OAuth2 flow
    
    def get_data(self):
        """Get data - to be implemented."""
        pass  # TODO: Implement API call


**Anti-Patterns:**

**1. Break in Finally Blocks:**
Python code example:
# Bad - break in finally doesn't work as expected
for item in items:
    try:
        process(item)
    finally:
        break  # This breaks, but finally always runs


**2. Continue After Break:**
Python code example:
# Bad - unreachable code
for item in items:
    if condition:
        break
    continue  # Never reached if break happens


**3. Pass as Implementation:**
Python code example:
# Bad - pass is not implementation
def critical_function():
    pass  # This doesn't do anything!

# Good - at least raise NotImplementedError
def critical_function():
    raise NotImplementedError("Must be implemented")


**4. Excessive Break/Continue:**
Python code example:
# Bad - too many control flow statements
for item in items:
    if condition1:
        continue
    if condition2:
        break
    if condition3:
        continue
    # Hard to follow logic

# Better - refactor into function
def should_process(item):
    return condition1 and not condition2 and condition3

for item in items:
    if should_process(item):
        process(item)


Understanding break, continue, and pass gives you fine-grained control over loop execution and code structure. Use them judiciously to write clearer, more maintainable code. Remember: the best code often uses these statements sparingly, preferring clear structure and well-designed loops.`,
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
