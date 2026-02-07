package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          120,
			Title:       "Shell Scripting Fundamentals",
			Description: "Learn bash scripting basics: variables, conditionals, loops, functions, and script execution.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Bash Scripting Basics",
					Content: `Shell scripting is one of the most powerful skills a Linux user or system administrator can develop. At its core, a shell script is simply a plain text file containing a sequence of commands that the shell interpreter executes one after another. Think of it like writing a recipe: instead of typing each command manually every time you need to perform a task, you write them all down once, and then the shell follows your instructions automatically. Bash (Bourne Again SHell) is by far the most common shell scripting language on Linux systems, and it comes pre-installed on virtually every Linux distribution.

**1. What is a Shell Script?**

A shell script is a text file that contains a series of commands that the shell can execute. When you run a script, the shell reads each line and executes it as if you had typed it directly into the terminal. This is incredibly useful because it allows you to automate repetitive tasks, chain together complex sequences of operations, and create reusable tools that you or your team can run with a single command. Imagine you need to back up several directories, compress them, and upload them to a remote server every night — instead of remembering and typing dozens of commands, you write a script once and let it handle everything.

**2. Script Structure**

Every well-written shell script follows a predictable structure. The first line is the "shebang" (#!/bin/bash), which tells the operating system which interpreter to use when running the script. Think of it as a label on a recipe that says "use this oven." After the shebang, you typically add comments (lines starting with #) that explain what the script does — this is crucial for maintainability, because a script you wrote six months ago might be completely mysterious without comments. The rest of the script is made up of actual shell commands, variables, conditionals, loops, and functions — all the building blocks that let you create sophisticated automation.

**3. Creating and Running a Script**

Creating a script is straightforward: open a text editor (such as nano, vim, or even a graphical editor), create a new file with a .sh extension (like myscript.sh), and start writing your commands. The first line should always be #!/bin/bash. Once you have written your script, you need to make it executable by running "chmod +x myscript.sh" — this tells the operating system that this file is allowed to be run as a program. You can then execute it with "./myscript.sh" (the ./ tells the shell to look in the current directory). Alternatively, you can run it with "bash myscript.sh", which does not require the file to be executable, because you are explicitly telling bash to interpret it.

**4. Why Write Shell Scripts?**

Shell scripting is invaluable for several reasons. First, it eliminates human error from repetitive tasks — a script will execute the same steps in the same order every time. Second, scripts serve as living documentation of complex procedures; instead of writing instructions in a wiki that might become outdated, the script itself is the instructions. Third, scripts can be scheduled to run automatically using tools like cron, enabling you to automate backups, system maintenance, monitoring, and reporting without any manual intervention. Finally, shell scripts are the backbone of system administration, DevOps pipelines, and deployment automation across the industry. Learning to write them is an investment that pays dividends throughout your career.`,
					CodeExamples: `#!/bin/bash
# This is a comment
# Simple script example

echo "Hello, World!"
echo "Current date: $(date)"
echo "Current user: $USER"

# Save as hello.sh
# Make executable: chmod +x hello.sh
# Run: ./hello.sh

# Or run without making executable
bash hello.sh`,
				},
				{
					Title: "Variables",
					Content: `Variables are the fundamental building blocks of any programming or scripting language, and bash is no exception. A variable is essentially a named container that holds a piece of data — a string of text, a number, or even the output of a command. Understanding how variables work in bash is critical because nearly every script you write will use them to store configuration values, track state, process user input, or pass data between different parts of your script.

**1. Variable Assignment**

In bash, you assign a value to a variable with the syntax VARIABLE=value. One of the most common mistakes beginners make is adding spaces around the equals sign — bash interprets "VARIABLE = value" as trying to run a command called "VARIABLE" with "=" and "value" as arguments. When you assign a variable, you do not use the dollar sign; the dollar sign is only used when you want to retrieve (reference) the value. So you write NAME="John" to assign, and then echo $NAME to use it. This distinction between assignment and reference is one of the first things to internalize when learning bash scripting.

**2. Variable Types**

Bash variables are untyped in the traditional sense — everything is technically a string — but you can use them to hold different kinds of data. Strings are assigned with quotes (NAME="John Doe"), numbers are assigned directly (COUNT=10), and you can even capture the output of a command into a variable using command substitution with the syntax RESULT=$(command). For example, CURRENT_DATE=$(date) stores the current date and time as a string. This ability to capture command output makes bash scripting extraordinarily flexible, because you can dynamically generate variable values based on the current state of the system.

**3. Special Variables**

Bash provides a set of built-in special variables that give you access to important information about the script and its execution environment. The variable $0 holds the name of the script itself, while $1, $2, $3, and so on hold the positional arguments passed to the script on the command line. The variable $# tells you how many arguments were passed, and $@ gives you all arguments as a list. Perhaps the most frequently used special variable is $?, which contains the exit status of the last command that was executed — 0 means success, and any non-zero value indicates an error. The variable $$ gives you the process ID of the current script, which is useful for creating unique temporary file names. Mastering these special variables is essential for writing scripts that can interact with their environment and respond to user input.

**4. Variable Best Practices**

Following consistent naming conventions makes your scripts much more readable and maintainable. The widely accepted convention is to use UPPERCASE names for constants and environment variables (like PATH, HOME, or your own CONFIG_DIR), and lowercase names for local script variables (like filename, counter, or result). Always quote variable references that might contain spaces — "$NAME" instead of $NAME — to prevent word splitting, which can cause subtle and hard-to-debug errors. Using curly braces for clarity, like ${VARIABLE}, is especially important when you need to concatenate a variable with other text (for example, "${FILE}name" correctly produces "testname", whereas "$FILEname" would look for a variable called FILEname that probably does not exist). These habits will save you hours of debugging as your scripts grow in complexity.`,
					CodeExamples: `#!/bin/bash
# Variable assignment
NAME="John"
AGE=30
CITY="New York"

# Using variables
echo "Name: $NAME"
echo "Age: $AGE"
echo "City: $CITY"

# Command substitution
CURRENT_DATE=$(date)
echo "Today: $CURRENT_DATE"

# Alternative syntax
CURRENT_TIME=date +%H:%M:%S
echo "Time: $CURRENT_TIME"

# Command line arguments
echo "Script name: $0"
echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"
echo "Number of arguments: $#"

# Exit status
ls /nonexistent
echo "Exit status: $?"

# Using braces for clarity
FILE="test"
echo "${FILE}name"  # testname
echo "$FILEnam"     # empty (FILEnam doesn't exist)

# Read-only variable
readonly PI=3.14159
# PI=4  # Error: readonly variable`,
				},
				{
					Title: "Conditionals",
					Content: `Conditionals are what transform a simple list of commands into an intelligent program that can make decisions. Without conditionals, a script would blindly execute every line regardless of context — it would try to delete a file even if it does not exist, or restart a service that is already running. Conditionals give your scripts the ability to examine the current state of the system, evaluate expressions, and choose different paths of execution based on the results. This is the foundation of all logic in shell scripting.

**1. The if Statement**

The most basic conditional in bash is the "if" statement. Its syntax is: if [ condition ]; then ... fi. The square brackets are actually a shorthand for the "test" command — when you write [ -f myfile.txt ], you are really running "test -f myfile.txt", which checks whether the file exists. The "then" keyword marks the beginning of the code block that runs if the condition is true, and "fi" (which is "if" spelled backwards) marks the end. You can extend this with "else" to provide an alternative path when the condition is false, and with "elif" (else-if) to chain multiple conditions together. Think of it like a decision tree: the script evaluates each condition in order and takes the first branch that matches.

**2. Test Conditions for Files**

One of the most common uses of conditionals in shell scripts is checking the state of files and directories. The test operators for files are incredibly useful: [ -f file ] checks whether a regular file exists, [ -d dir ] checks for a directory, [ -r file ] checks if the file is readable by the current user, [ -w file ] checks if it is writable, and [ -x file ] checks if it is executable. These checks are essential for writing robust scripts — for example, before reading a configuration file, you should always verify it exists and is readable, and before writing to a log file, you should check that you have write permission. Without these guards, your scripts will crash with cryptic error messages when they encounter unexpected conditions.

**3. String and Numeric Comparisons**

Bash conditionals also support comparing strings and numbers, but the syntax differs between the two, which is a common source of confusion. For strings, you use [ str1 = str2 ] for equality and [ str1 != str2 ] for inequality. You can also check if a string is empty with [ -z string ] or non-empty with [ -n string ]. For numbers, however, you must use special operators: [ num1 -eq num2 ] for equality, [ num1 -lt num2 ] for less than, [ num1 -gt num2 ] for greater than, [ num1 -le num2 ] for less than or equal, and [ num1 -ge num2 ] for greater than or equal. The reason for this distinction is that bash's single-bracket test command treats everything as strings by default, so the -eq, -lt, and -gt operators explicitly tell it to perform numeric comparison.

**4. Combining Conditions and the Case Statement**

You can combine multiple conditions using logical AND (&&) and logical OR (||) operators. For example, [ -f "$FILE" ] && [ -r "$FILE" ] checks that a file both exists and is readable. For situations where you need to compare a single variable against many possible values, the "case" statement is more readable than a long chain of elif blocks. The case statement works like a switch statement in other languages: it matches the value of a variable against a series of patterns and executes the corresponding block of code. This is especially useful for parsing command-line arguments (like start, stop, restart) or handling menu selections in interactive scripts.`,
					CodeExamples: `#!/bin/bash
# Simple if
if [ -f "file.txt" ]; then
    echo "File exists"
fi

# if-else
if [ -d "/tmp" ]; then
    echo "Directory exists"
else
    echo "Directory not found"
fi

# String comparison
NAME="John"
if [ "$NAME" = "John" ]; then
    echo "Hello John"
fi

# Number comparison
AGE=25
if [ $AGE -ge 18 ]; then
    echo "Adult"
else
    echo "Minor"
fi

# File checks
FILE="script.sh"
if [ -f "$FILE" ]; then
    echo "$FILE is a regular file"
elif [ -d "$FILE" ]; then
    echo "$FILE is a directory"
else
    echo "$FILE does not exist"
fi

# Multiple conditions
if [ -f "$FILE" ] && [ -r "$FILE" ]; then
    echo "File exists and is readable"
fi

# Using test command (alternative syntax)
if test -f "$FILE"; then
    echo "File exists"
fi

# Case statement
case "$1" in
    start)
        echo "Starting..."
        ;;
    stop)
        echo "Stopping..."
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        ;;
esac`,
				},
				{
					Title: "Loops",
					Content: `Loops are one of the most powerful constructs in any programming language, and in shell scripting they are absolutely indispensable. A loop allows you to repeat a block of commands multiple times — whether that means processing every file in a directory, iterating through a list of servers to check their status, or reading a data file line by line. Without loops, you would have to manually duplicate commands for each item you want to process, which is error-prone, tedious, and completely impractical when dealing with hundreds or thousands of items.

**1. The for Loop**

The for loop is the most commonly used loop in bash scripting. Its basic syntax is: for variable in list; do ... done. On each iteration, the variable takes on the next value from the list, and the commands inside the loop body execute with that value. The "list" can be almost anything: a literal list of words (for name in Alice Bob Charlie), a range of numbers (for i in {1..100}), the output of a command (for file in $(ls *.log)), or a glob pattern that expands to matching filenames (for file in /var/log/*.log). Bash also supports a C-style for loop — for ((i=0; i<10; i++)) — which is familiar to programmers coming from C, Java, or similar languages and is particularly useful when you need precise control over a numeric counter.

**2. The while and until Loops**

The while loop repeats its body as long as a condition remains true. Its syntax is: while [ condition ]; do ... done. This is ideal for situations where you do not know in advance how many iterations you need — for example, reading a file line by line until you reach the end, or waiting for a process to finish. The until loop is the mirror image of while: it repeats as long as the condition is false, stopping when it becomes true. While "until" is less commonly used, it can make certain logic more readable — for instance, "until [ -f /tmp/ready.flag ]; do sleep 1; done" clearly expresses "keep waiting until this file appears."

**3. Loop Control: break, continue, and exit**

Bash provides three keywords for controlling loop execution. The "break" statement immediately exits the loop, skipping any remaining iterations — this is useful when you have found what you are looking for and do not need to continue searching. The "continue" statement skips the rest of the current iteration and jumps to the next one — for example, if you are processing files and want to skip files that do not match a certain pattern. The "exit" statement is more drastic: it terminates the entire script, not just the loop. Understanding when to use each of these is key to writing efficient loops that do not waste time on unnecessary iterations.

**4. Common Loop Patterns**

Several loop patterns come up again and again in real-world scripting. Iterating over files with a glob (for file in *.txt; do ... done) is perhaps the most common — you might use it to rename files, process log data, or convert image formats. Reading a file line by line with a while loop (while read line; do ... done < file.txt) is essential for processing structured data like CSV files or configuration files. Nested loops (a loop inside another loop) let you work with combinations — for example, generating a multiplication table or testing connectivity between multiple hosts and ports. As you gain experience, you will develop an instinct for which loop pattern best fits each situation.`,
					CodeExamples: `#!/bin/bash
# For loop with list
for name in Alice Bob Charlie; do
    echo "Hello, $name"
done

# For loop with files
for file in *.txt; do
    echo "Processing: $file"
done

# For loop with range
for i in {1..5}; do
    echo "Number: $i"
done

# C-style for loop
for ((i=1; i<=5; i++)); do
    echo "Count: $i"
done

# While loop
COUNT=1
while [ $COUNT -le 5 ]; do
    echo "Count: $COUNT"
    COUNT=$((COUNT + 1))
done

# Until loop
COUNT=1
until [ $COUNT -gt 5 ]; do
    echo "Count: $COUNT"
    COUNT=$((COUNT + 1))
done

# Read file line by line
while read line; do
    echo "Line: $line"
done < file.txt

# Loop with break
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        break
    fi
    echo $i
done

# Loop with continue
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        continue
    fi
    echo $i
done

# Nested loops
for i in {1..3}; do
    for j in {1..3}; do
        echo "$i x $j = $((i * j))"
    done
done`,
				},
				{
					Title: "Functions",
					Content: `Functions are the key to writing shell scripts that are organized, reusable, and maintainable. Without functions, a complex script quickly becomes an unreadable wall of commands that is difficult to debug, modify, or extend. A function is essentially a named block of commands that you can call by name from anywhere in your script — think of it like creating your own custom command. Just as you might refactor a large essay into sections with clear headings, functions let you break a monolithic script into logical, self-contained pieces that each handle one specific task.

**1. Function Syntax and Declaration**

Bash supports two syntaxes for declaring functions. The most common is function_name() { commands; }, where you write the function name followed by parentheses and then the function body enclosed in curly braces. The alternative syntax uses the "function" keyword: function function_name { commands; }. Both are functionally identical, but the first form is more portable across different shells. A function must be defined before it is called — bash reads scripts from top to bottom, so if you try to call a function before its definition, you will get a "command not found" error. Many scripters adopt the practice of defining all functions at the top of the script and then calling them at the bottom, creating a clear separation between definitions and execution.

**2. Arguments and Return Values**

Functions accept arguments the same way scripts do — through positional parameters $1, $2, $3, and so on. When you call "greet Alice", the value "Alice" is accessible inside the function as $1. This is important: the positional parameters inside a function are local to that function call, not the script's command-line arguments. Functions communicate their success or failure through the "return" statement, which sets an exit status (0 for success, 1-255 for various failure conditions). However, "return" only sets a numeric status — if you need to return actual data (like a computed string or number), the common pattern is to echo the result and capture it with command substitution: RESULT=$(my_function args). This distinction between exit status and output is a crucial concept that trips up many beginners.

**3. Local Variables and Scope**

By default, all variables in bash are global — if you set a variable inside a function, it is visible everywhere in the script, which can lead to accidental name collisions and subtle bugs. The "local" keyword solves this problem: "local count=0" creates a variable that exists only within the function and disappears when the function returns. Using local variables is one of the most important best practices in bash scripting. Without them, two functions that both use a variable named "result" would silently overwrite each other's data, creating bugs that are extremely difficult to track down. Always declare your function variables as local unless you explicitly intend them to be global.

**4. Best Practices for Functions**

Write functions that follow the single responsibility principle — each function should do one thing and do it well. Give functions descriptive names that clearly communicate what they do (like "backup_database" or "validate_input" rather than "func1" or "do_stuff"). Always use local variables inside functions to avoid polluting the global namespace. Return meaningful exit codes so that callers can check whether the function succeeded or failed. Document complex functions with a comment block that explains the purpose, expected arguments, and return values. Following these practices transforms your scripts from fragile, one-off hacks into reliable, professional tools that others (including your future self) can understand and maintain.`,
					CodeExamples: `#!/bin/bash
# Simple function
greet() {
    echo "Hello, World!"
}

# Call function
greet

# Function with arguments
greet_person() {
    echo "Hello, $1!"
}

greet_person "Alice"
greet_person "Bob"

# Function with return value
is_even() {
    if [ $(($1 % 2)) -eq 0 ]; then
        return 0  # Success (true)
    else
        return 1  # Failure (false)
    fi
}

if is_even 4; then
    echo "4 is even"
fi

# Function with local variable
counter() {
    local count=0
    count=$((count + 1))
    echo "Count: $count"
}

counter  # Count: 1
counter  # Count: 1 (local, resets each call)

# Function returning value via echo
add() {
    echo $(($1 + $2))
}

RESULT=$(add 5 3)
echo "Sum: $RESULT"

# Function checking file
file_exists() {
    if [ -f "$1" ]; then
        echo "File $1 exists"
        return 0
    else
        echo "File $1 not found"
        return 1
    fi
}

file_exists "script.sh"`,
				},
				{
					Title: "Script Debugging",
					Content: `Debugging is an unavoidable part of writing shell scripts, and developing strong debugging skills will save you countless hours of frustration. Even experienced scripters make mistakes — a misplaced quote, a misspelled variable name, or a subtle logic error can cause a script to behave in completely unexpected ways. The good news is that bash provides a rich set of built-in debugging tools and techniques that, once mastered, make it relatively straightforward to track down and fix problems. Think of debugging like being a detective: you gather clues (error messages, unexpected output), form hypotheses, and test them until you find the culprit.

**1. Debug Mode with bash -x and set -x**

The single most powerful debugging tool in bash is the execution trace mode, activated with "bash -x script.sh" or by adding "set -x" inside your script. When trace mode is enabled, bash prints each command to stderr before executing it, with a "+" prefix. Crucially, it shows the command after variable expansion, so you can see exactly what values your variables held at each step. This is like having a play-by-play commentary of your script's execution. You can enable tracing for just a specific section of your script by placing "set -x" before the problematic area and "set +x" after it, keeping the debug output focused and manageable rather than overwhelming.

**2. Strict Mode with set -euo pipefail**

One of the best things you can do for script reliability is to enable strict mode at the top of every script with "set -euo pipefail". The -e flag causes the script to exit immediately if any command fails (returns a non-zero exit status), rather than silently continuing with potentially corrupted state. The -u flag treats references to undefined variables as errors, catching typos like $NAEM instead of $NAME that would otherwise silently expand to an empty string. The -o pipefail flag ensures that a pipeline (command1 | command2) fails if any command in the pipeline fails, not just the last one. Together, these three flags catch the vast majority of common scripting errors early, before they cascade into larger problems. It is widely considered a best practice to include this line in every production script.

**3. Common Error Categories**

Script errors generally fall into four categories, each requiring a different debugging approach. Syntax errors (missing quotes, unmatched brackets, forgotten semicolons) are caught by the shell before execution and produce error messages with line numbers — you can also check for them without running the script using "bash -n script.sh". Variable errors (unset variables, wrong variable names, missing quotes) are often silent in default mode but are caught by "set -u". Logic errors (wrong conditions, infinite loops, off-by-one errors) are the trickiest because the script runs without errors but produces incorrect results — trace mode and strategic echo statements are your best tools here. Permission errors (script not executable, cannot write to file) are usually straightforward once you recognize the "Permission denied" message pattern.

**4. Debugging Tools and Best Practices**

Beyond the built-in bash tools, the shellcheck utility is an invaluable static analysis tool that examines your script without running it and identifies potential issues ranging from common pitfalls to stylistic problems — install it and run it on every script you write. For runtime debugging, strategically placed echo statements (like "echo DEBUG: variable=$variable") let you inspect variable values at specific points in execution. Always test scripts incrementally: write a few lines, test them, then add more. When debugging a complex script, isolate the problematic section into a smaller test script that you can run and modify quickly. Check exit codes after important commands ($? or using && and ||), and validate all inputs at the beginning of your script. These disciplined practices transform debugging from a frustrating guessing game into a systematic, manageable process.`,
					CodeExamples: `#!/bin/bash
# Debug mode - trace execution
set -x

NAME="John"
echo "Hello, $NAME"

set +x  # Disable debug mode

# Syntax check (no execution)
bash -n script.sh

# Run with debug output
bash -x script.sh

# Enable strict mode
set -euo pipefail
# -e: Exit on error
# -u: Exit on undefined variable
# -o pipefail: Exit on pipe failure

# Debug specific section
set -x
# Your code here
set +x

# Print variable values
echo "DEBUG: NAME=$NAME"
echo "DEBUG: COUNT=$COUNT"

# Check command exit status
command
if [ $? -eq 0 ]; then
    echo "Success"
else
    echo "Failed with exit code: $?"
fi

# Better: Use && and ||
command && echo "Success" || echo "Failed"

# Debug function
debug_function() {
    echo "DEBUG: Function called with: $@"
    # Function code
}

# Using shellcheck (install first: apt install shellcheck)
shellcheck script.sh
# Shows potential issues and suggestions

# Common debugging patterns
# 1. Check if variable is set
if [ -z "${VAR:-}" ]; then
    echo "ERROR: VAR is not set"
    exit 1
fi

# 2. Check if file exists before using
if [ ! -f "$FILE" ]; then
    echo "ERROR: File $FILE not found"
    exit 1
fi

# 3. Validate input
if [ $# -eq 0 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

# 4. Debug with function
debug() {
    [ "${DEBUG:-0}" -eq 1 ] && echo "DEBUG: $@" >&2
}

debug "Processing file: $FILE"
# Only prints if DEBUG=1`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          121,
			Title:       "Advanced Shell Scripting",
			Description: "Master advanced bash features: arrays, string manipulation, file I/O, error handling, and debugging.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Arrays",
					Content: `Arrays are a fundamental data structure that allow you to store and manipulate collections of values under a single variable name. While regular bash variables hold a single value, an array can hold dozens, hundreds, or even thousands of values, each accessible by a numeric index. Think of an array like a numbered list or a row of labeled mailboxes — each slot has a position (starting from 0), and you can put a value in any slot, read it back, or iterate through all the slots. Arrays are essential when your script needs to work with lists of files, server names, configuration options, or any collection of related data.

**1. Declaring and Accessing Arrays**

You create an array in bash by enclosing a space-separated list of values in parentheses: FRUITS=("apple" "banana" "cherry"). You can also build an array one element at a time with ARRAY[0]="first" and ARRAY[1]="second". To access a specific element, you use the syntax ${ARRAY[index]} — for example, ${FRUITS[0]} returns "apple" (remember, arrays are zero-indexed, so the first element is at position 0, not 1). To access all elements at once, use ${ARRAY[@]} or ${ARRAY[*]}, which expands to every element in the array. To find out how many elements are in the array, use ${#ARRAY[@]}. These operations form the foundation of all array manipulation in bash, and you will use them constantly when working with lists of data.

**2. Array Operations: Adding, Slicing, and Iterating**

Bash provides several operators for manipulating arrays. To add an element to the end of an existing array, use the += operator: FRUITS+=("date") appends "date" to the FRUITS array. To extract a subset of elements (a "slice"), use ${ARRAY[@]:start:length} — for example, ${FRUITS[@]:1:2} returns the second and third elements. To get a list of all valid indices in an array, use ${!ARRAY[@]}, which is useful for sparse arrays where some indices might be unset. Iterating over an array with a for loop is one of the most common patterns: for item in "${ARRAY[@]}"; do echo "$item"; done processes each element one at a time. Always quote the array expansion ("${ARRAY[@]}") to correctly handle elements that contain spaces.

**3. Practical Array Use Cases**

In real-world scripting, arrays shine in several scenarios. You might store a list of server hostnames and loop through them to check their status. You might read the lines of a configuration file into an array for processing. You might collect the output of a command (like the list of running containers) into an array and then operate on each item. Arrays also make it easy to build up a list of arguments to pass to a command — instead of concatenating strings, you append to an array and then expand it. While bash arrays are not as feature-rich as arrays in languages like Python or JavaScript (for example, there is no built-in sort or search), they are more than sufficient for most system administration and automation tasks, and mastering them will significantly expand what you can accomplish in shell scripts.`,
					CodeExamples: `#!/bin/bash
# Declare array
FRUITS=("apple" "banana" "cherry")

# Access elements
echo ${FRUITS[0]}  # apple
echo ${FRUITS[1]}  # banana
echo ${FRUITS[2]}  # cherry

# All elements
echo ${FRUITS[@]}  # apple banana cherry

# Array length
echo ${#FRUITS[@]}  # 3

# Loop through array
for fruit in "${FRUITS[@]}"; do
    echo "Fruit: $fruit"
done

# Add element
FRUITS+=("date")
echo ${FRUITS[@]}

# Array slice
echo ${FRUITS[@]:1:2}  # banana cherry

# Indices
echo ${!FRUITS[@]}  # 0 1 2 3`,
				},
				{
					Title: "String Manipulation",
					Content: `String manipulation is one of the most frequently needed skills in shell scripting, because so much of what scripts do involves processing text — parsing filenames, extracting data from command output, transforming configuration values, and formatting output for reports. While you could use external tools like sed, awk, or cut for string operations, bash has a rich set of built-in string manipulation features that are faster (no subprocess spawned) and often more readable. Mastering these built-in operations will make your scripts both more efficient and more elegant.

**1. String Length and Substrings**

To find the length of a string, use ${#STRING} — for example, if STRING="Hello World", then ${#STRING} evaluates to 11. This is useful for validation (checking that input meets a minimum length requirement) or for formatting output with consistent column widths. To extract a portion of a string, use the substring syntax ${STRING:start:length}, where "start" is the zero-based starting position and "length" is the number of characters to extract. For example, ${STRING:0:5} returns "Hello" and ${STRING:6} returns everything from position 6 onward ("World"). This is analogous to the substring or slice operations in other programming languages.

**2. Search and Replace**

Bash provides two forms of in-string replacement. The single-slash form ${STRING/old/new} replaces only the first occurrence of "old" with "new", while the double-slash form ${STRING//old/new} replaces all occurrences. For example, if PATH_VAR="/usr/local/bin:/usr/bin:/bin", you could replace all colons with newlines using ${PATH_VAR//:/\\n} to display one path per line. These replacements are performed using glob patterns (not regular expressions), so you can use wildcards like * and ? in the search pattern. This capability is extremely useful for tasks like sanitizing user input, transforming file paths, or modifying configuration strings without resorting to external tools.

**3. Prefix and Suffix Removal**

The # and % operators remove patterns from the beginning or end of a string, respectively. The single # (${STRING#pattern}) removes the shortest matching prefix, while the double ## (${STRING##pattern}) removes the longest matching prefix. Similarly, % removes the shortest matching suffix and %% removes the longest. The classic use case is extracting filename components: if FILE="archive.tar.gz", then ${FILE#*.} gives "tar.gz" (shortest prefix up to first dot removed), ${FILE##*.} gives "gz" (longest prefix removed, leaving only the final extension), ${FILE%.*} gives "archive.tar" (shortest suffix removed), and ${FILE%%.*} gives "archive" (longest suffix removed). These operators are invaluable for parsing file paths, extracting extensions, and decomposing structured strings.

**4. Case Conversion**

Bash 4.0 and later provide built-in case conversion operators. The double-caret ${STRING^^} converts the entire string to uppercase, while the double-comma ${STRING,,} converts it to lowercase. You can also convert just the first character with a single operator: ${STRING^} capitalizes the first letter, and ${STRING,} lowercases it. These are convenient for normalizing user input (for example, converting "yes", "YES", and "Yes" all to the same form for comparison), formatting output, or generating standardized identifiers. Before bash 4.0, you would have needed to pipe through tr or use other external commands to achieve the same result, making these built-in operators a welcome addition.`,
					CodeExamples: `#!/bin/bash
STRING="Hello World"

# Length
echo ${#STRING}  # 11

# Substring
echo ${STRING:0:5}  # Hello
echo ${STRING:6}    # World

# Replace
echo ${STRING/World/Unix}  # Hello Unix
echo ${STRING//l/L}        # HeLLo WorLd

# Remove prefix
FILE="file.txt"
echo ${FILE#*.}     # txt
echo ${FILE##*.}    # txt

# Remove suffix
echo ${FILE%.*}     # file
echo ${FILE%%.*}    # file

# Case conversion
TEXT="Hello World"
echo ${TEXT^^}      # HELLO WORLD
echo ${TEXT,,}      # hello world`,
				},
				{
					Title: "Error Handling",
					Content: `Error handling is what separates amateur scripts from production-quality automation. A script without error handling is like a car without brakes — it works fine as long as everything goes perfectly, but the moment something unexpected happens, the results can be catastrophic. In a production environment, network connections drop, disks fill up, files get moved or deleted, and services crash. A well-written script anticipates these failures, detects them when they occur, and responds gracefully — whether that means retrying, falling back to an alternative, logging the error, or cleanly shutting down.

**1. Understanding Exit Codes**

Every command in Linux returns an exit code when it finishes: 0 means success, and any non-zero value (1 through 255) indicates some kind of failure. This convention is the foundation of all error handling in shell scripting. You can check the exit code of the last command using the special variable $?. For example, after running "grep pattern file.txt", $? will be 0 if the pattern was found, 1 if it was not found, and 2 if there was an error (like the file not existing). Many commands use different non-zero codes to indicate different types of failures, so consulting a command's man page to understand its exit codes can be very informative. Your own scripts and functions should also follow this convention — return 0 for success and non-zero for failure — so they integrate seamlessly into the larger ecosystem of shell tools.

**2. Strict Mode: set -euo pipefail**

The most impactful error handling technique is enabling bash's strict mode with "set -euo pipefail" at the top of your script. The -e flag (errexit) causes the script to immediately exit if any command returns a non-zero exit code, preventing the script from blindly continuing with invalid state. The -u flag (nounset) treats any reference to an undefined variable as an error, catching typos and missing configurations before they cause subtle downstream problems. The -o pipefail flag changes how pipeline errors are reported: normally, a pipeline like "cat file | grep pattern | wc -l" reports only the exit code of the last command (wc), even if cat or grep failed. With pipefail, the pipeline fails if any component fails. Together, these three flags catch the vast majority of common scripting errors and should be the first line of every production script after the shebang.

**3. The trap Command for Cleanup and Signal Handling**

The "trap" command allows you to register a function or command that will be automatically executed when certain signals or events occur. The most common use is cleanup on exit: trap 'rm -f /tmp/tempfile; echo "Cleaned up"' EXIT ensures that temporary files are removed no matter how the script exits — whether it completes normally, hits an error, or is interrupted by the user pressing Ctrl+C. You can also trap specific signals like ERR (triggered when a command fails) to implement custom error logging: trap 'echo "Error on line $LINENO" >&2' ERR. The trap mechanism is the bash equivalent of try/finally blocks in other languages, and it is essential for writing scripts that do not leave behind messy temporary files, dangling processes, or half-completed operations when something goes wrong.

**4. Conditional Error Handling with && and ||**

For inline error handling, bash provides the && (AND) and || (OR) operators. The command "mkdir /tmp/mydir && echo 'Directory created'" only runs the echo if mkdir succeeds. Conversely, "mkdir /tmp/mydir || echo 'Failed to create directory'" runs the echo only if mkdir fails. You can combine these for a compact if-then-else pattern: "command && handle_success || handle_failure". This pattern is especially useful in scripts where you want to handle errors on a per-command basis rather than with the blanket "exit on error" approach. For more complex error handling, you can wrap sequences in functions and use if statements to check their return codes, giving you full control over how your script responds to each possible failure scenario.`,
					CodeExamples: `#!/bin/bash
# Exit on error
set -e

# Exit on undefined variable
set -u

# Exit on pipe failure
set -o pipefail

# Or combine
set -euo pipefail

# Check exit status
command || echo "Command failed"

# Execute if success
command && echo "Command succeeded"

# Trap errors
trap 'echo "Error on line $LINENO"' ERR

# Trap exit
trap 'echo "Script exiting"' EXIT

# Cleanup on exit
cleanup() {
    echo "Cleaning up..."
    rm -f /tmp/tempfile
}
trap cleanup EXIT`,
				},
				{
					Title: "File I/O in Scripts",
					Content: `Reading from and writing to files is at the heart of what most shell scripts do. Whether you are processing log files, generating reports, parsing configuration data, or transforming CSV files, you need to understand how bash handles file input and output. The Unix philosophy of "everything is a file" means that the same techniques you use to read a text file also work for reading from devices, pipes, and network sockets. Mastering file I/O in bash will unlock your ability to build powerful data processing pipelines entirely in shell script.

**1. Reading Files with the while read Loop**

The most common and robust way to read a file line by line in bash is the "while read" loop with input redirection: while IFS= read -r line; do ... done < filename. This pattern reads one line at a time, making it memory-efficient even for very large files (unlike approaches that load the entire file into memory). The -r flag tells read not to interpret backslash characters as escape sequences, preserving the literal content of each line. Setting IFS= (Internal Field Separator to empty) before read prevents leading and trailing whitespace from being stripped. You can also read multiple fields per line by providing multiple variable names: "while IFS=, read -r name age city; do ... done < data.csv" splits each line at commas and assigns the parts to separate variables. This is how you parse structured data like CSV or TSV files in bash.

**2. The read Command for Interactive Input**

Beyond file reading, the "read" command is also your primary tool for getting interactive input from users. The basic "read variable" waits for the user to type a line and stores it in the variable. Adding -p provides a prompt ("read -p 'Enter your name: ' name"), -s hides the input for passwords ("read -s -p 'Password: ' pass"), -t sets a timeout in seconds ("read -t 10 response"), and -a reads a line into an array ("read -a words <<< 'one two three'"). These options make read versatile enough for building interactive menus, confirmation prompts, and configuration wizards within your scripts. The here-string operator (<<<) is a convenient way to feed a string directly to read or any command that expects stdin, without needing a separate echo pipe.

**3. File Redirection and Output**

Bash's file redirection operators are the mechanism for directing data into and out of files. The > operator redirects output to a file, creating it if it does not exist or overwriting it if it does — "echo 'Hello' > greeting.txt" writes "Hello" to the file. The >> operator appends instead of overwriting, which is essential for building up log files or accumulating results. Input redirection with < feeds a file's contents into a command's stdin. You can redirect stderr separately with 2> (for example, "command 2> errors.log" captures only error messages), or merge stderr into stdout with 2>&1. For writing multi-line content, here documents (<<EOF ... EOF) let you embed a block of text directly in your script with variable expansion, which is perfect for generating configuration files, email bodies, or SQL queries.

**4. File Descriptors and Advanced I/O**

Under the hood, all I/O in Linux works through file descriptors — numbered handles that represent open files or streams. The three standard descriptors are 0 (stdin), 1 (stdout), and 2 (stderr), but you can open additional custom descriptors (3 through 9) using the exec command. For example, "exec 3> output.txt" opens file descriptor 3 for writing to output.txt, and then "echo 'data' >&3" writes to it. This is useful when you need to read from one file and write to another simultaneously, or when you need to separate different types of output. You close a file descriptor with "exec 3>&-". While custom file descriptors are an advanced technique, understanding them gives you complete control over how data flows through your scripts and is essential for building sophisticated data processing tools that handle multiple input and output streams.`,
					CodeExamples: `#!/bin/bash
# Read file line by line
while read line; do
    echo "Line: $line"
done < file.txt

# Read with IFS (Internal Field Separator)
while IFS= read -r line; do
    echo "$line"
done < file.txt

# Read into variables
while read name age city; do
    echo "Name: $name, Age: $age, City: $city"
done < data.txt

# Read into array
read -a array <<< "one two three"
echo "${array[0]}"  # one

# Write to file (overwrite)
echo "Hello World" > output.txt

# Append to file
echo "New line" >> output.txt

# Write multiple lines
cat > config.txt << EOF
server=example.com
port=8080
timeout=30
EOF

# Append multiple lines
cat >> log.txt << EOF
$(date): Event occurred
User: $USER
EOF

# Read and process
while read -r line; do
    if [[ $line =~ error ]]; then
        echo "ERROR: $line" >> error.log
    fi
done < application.log

# Here string
grep "pattern" <<< "text with pattern in it"

# Read from command output
while read -r line; do
    echo "Process: $line"
done < <(ps aux | grep python)

# File descriptors
exec 3> output.txt
echo "This goes to file" >&3
exec 3>&-  # Close

# Read and write simultaneously
exec 3< input.txt
exec 4> output.txt
while read -u 3 line; do
    echo "Processed: $line" >&4
done
exec 3<&-
exec 4>&-

# Process CSV file
while IFS=, read -r col1 col2 col3; do
    echo "Column 1: $col1"
    echo "Column 2: $col2"
done < data.csv

# Read with timeout
if read -t 5 -r line; then
    echo "You entered: $line"
else
    echo "Timeout!"
fi

# Read password (hidden)
read -s -p "Password: " password
echo

# Read with prompt
read -p "Enter name: " name
echo "Hello, $name"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          122,
			Title:       "System Administration",
			Description: "Learn user management, group management, sudo, system services (systemd), and log management.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "User Management",
					Content: `User management is one of the most fundamental responsibilities of a Linux system administrator. Every person (or automated service) that interacts with a Linux system does so through a user account, and managing these accounts — creating them, configuring their permissions, modifying their properties, and removing them when no longer needed — is essential for maintaining both the functionality and security of your systems. Think of user accounts as keys to a building: each person gets a key that opens the specific doors they need access to, and as an administrator, you are the locksmith who creates, manages, and revokes those keys.

**1. User Management Commands**

Linux provides a suite of commands for managing user accounts. The "useradd" command creates a new user account, typically with the -m flag to also create a home directory (useradd -m -s /bin/bash newuser). The "passwd" command sets or changes a user's password. The "usermod" command modifies an existing account — you can change the user's shell, home directory, group memberships, or even lock the account. The "userdel" command removes a user account, optionally with -r to also delete their home directory and mail spool. For quick information, "id" shows a user's UID, GID, and group memberships, "whoami" displays the current user's name, and "who" or "w" lists all currently logged-in users. These commands form the daily toolkit of any system administrator.

**2. User Account Files**

Behind the scenes, Linux stores user account information in three critical files. The /etc/passwd file contains one line per user account with fields for username, UID, GID, home directory, and login shell — despite its name, it no longer stores actual passwords. The /etc/shadow file stores the encrypted (hashed) passwords along with password aging information (expiration dates, minimum age, etc.) — this file is readable only by root for security. The /etc/group file maps group names to GIDs and lists group members. Understanding these files is important because sometimes you need to troubleshoot login issues, audit user accounts, or write scripts that process user information. While you should generally use the management commands rather than editing these files directly, knowing their structure helps you understand what the commands are doing behind the scenes.

**3. Common User Operations and Best Practices**

In day-to-day administration, the most frequent operations are creating users for new team members, adding users to groups for access control (usermod -aG docker username adds the user to the docker group), and locking accounts when employees leave (usermod -L username). When creating a user, always specify a shell (-s /bin/bash) and create a home directory (-m). When removing a user, consider whether to preserve or delete their home directory depending on data retention policies. For security, enforce strong password policies, regularly audit user accounts to remove stale ones, and follow the principle of least privilege — give each user only the minimum access they need to do their job. Many organizations also use centralized authentication systems like LDAP or Active Directory, but understanding local user management is essential as a foundation.`,
					CodeExamples: `# Create user
sudo useradd -m -s /bin/bash newuser

# Set password
sudo passwd newuser

# Add user to group
sudo usermod -aG sudo newuser
sudo usermod -aG docker newuser

# Modify user
sudo usermod -s /bin/zsh newuser  # Change shell
sudo usermod -d /new/home newuser  # Change home directory

# Delete user
sudo userdel newuser
sudo userdel -r newuser  # Also remove home directory

# View user info
id username
id  # Current user

# View all users
cat /etc/passwd
getent passwd

# View logged in users
who
w

# Lock/unlock account
sudo usermod -L username  # Lock
sudo usermod -U username  # Unlock`,
				},
				{
					Title: "System Services (systemd)",
					Content: `systemd is the init system and service manager that has become the standard on virtually all modern Linux distributions, including Ubuntu, Debian, Fedora, CentOS, and Arch Linux. It is responsible for bootstrapping the entire system during boot, managing all running services (daemons), handling system logging, and much more. Understanding systemd is not optional for modern Linux administration — it is the control plane through which you start, stop, monitor, and configure nearly everything that runs on your system. Think of systemd as the orchestra conductor: it decides which instruments (services) play, when they start, and what happens if one of them hits a wrong note (fails).

**1. Managing Services with systemctl**

The primary command for interacting with systemd is "systemctl", which provides a unified interface for controlling all system services. The most commonly used subcommands are: "systemctl start service" to start a stopped service, "systemctl stop service" to stop a running one, "systemctl restart service" to stop and start it again (useful after configuration changes), and "systemctl reload service" to tell a running service to re-read its configuration without fully restarting (not all services support this). The "systemctl status service" command is invaluable for troubleshooting — it shows whether the service is running, its PID, memory usage, and the most recent log entries, all in a single concise view. For example, "systemctl status nginx" instantly tells you if your web server is running, when it started, and whether it has logged any errors.

**2. Boot-time Configuration: enable and disable**

Beyond starting and stopping services in the current session, systemd manages which services start automatically when the system boots. "systemctl enable service" creates symbolic links in the appropriate systemd directories so that the service is started during the boot process, while "systemctl disable service" removes those links. It is important to understand that "enable" does not start the service immediately — it only configures it to start on the next boot. Similarly, "disable" does not stop a currently running service. To both enable and start a service in one command, you can use "systemctl enable --now service". This distinction between runtime state (started/stopped) and boot configuration (enabled/disabled) is crucial: a service can be running but disabled (it will not survive a reboot) or stopped but enabled (it will start automatically on the next boot).

**3. Understanding Service States and Troubleshooting**

systemd tracks services through several states that tell you exactly what is happening. An "active (running)" service is currently executing. An "inactive (dead)" service is stopped. A "failed" service attempted to start but encountered an error — this is your cue to investigate using "systemctl status service" and "journalctl -u service" to read the detailed logs. The "list-units" command shows all currently loaded units and their states, while "list-unit-files" shows all installed unit files and whether they are enabled or disabled. When a service fails, the troubleshooting workflow is: check the status for the high-level error, read the journal logs for detailed messages, fix the configuration or dependency issue, and then try starting the service again. Mastering this workflow is essential because you will encounter service failures regularly in any production environment.

**4. Service Unit Files**

Every service managed by systemd is defined by a unit file, typically located in /etc/systemd/system/ (for custom or overridden configurations) or /lib/systemd/system/ (for distribution-provided defaults). A unit file specifies how to start the service, what user to run it as, what dependencies it has, and what to do if it fails. Understanding the structure of unit files enables you to create custom services for your own applications, modify the behavior of existing services, and debug startup issues. After modifying a unit file, you must run "systemctl daemon-reload" to tell systemd to re-read the configuration. While creating custom unit files is an advanced topic, even a basic understanding helps you grasp why services behave the way they do and how to configure them to meet your needs.`,
					CodeExamples: `# Check service status
sudo systemctl status nginx

# Start service
sudo systemctl start nginx

# Stop service
sudo systemctl stop nginx

# Restart service
sudo systemctl restart nginx

# Reload configuration
sudo systemctl reload nginx

# Enable at boot
sudo systemctl enable nginx

# Disable at boot
sudo systemctl disable nginx

# List all services
systemctl list-units --type=service

# List enabled services
systemctl list-unit-files --type=service --state=enabled

# View service logs
sudo journalctl -u nginx
sudo journalctl -u nginx -f  # Follow logs`,
				},
				{
					Title: "Log Management",
					Content: `Log management is one of the most critical skills for any system administrator, because logs are your primary window into what is actually happening on your system. When a service crashes, when a security breach occurs, when performance degrades, or when users report mysterious errors, the logs are where you go to find answers. Logs record everything: login attempts, service startups and failures, kernel messages, application errors, network events, and more. Without effective log management, troubleshooting becomes guesswork, security auditing is impossible, and compliance requirements cannot be met. Think of logs as the black box recorder on an airplane — they capture a detailed record of events that you can replay and analyze when something goes wrong.

**1. The systemd Journal (journalctl)**

On modern Linux systems running systemd, the primary logging system is the systemd journal, queried with the "journalctl" command. Unlike traditional text-based log files, the journal stores entries in a binary, indexed format that enables powerful querying capabilities. Running "journalctl" without arguments shows all log entries, but its real power lies in filtering: "journalctl -u nginx" shows only entries from the nginx service, "journalctl --since '1 hour ago'" filters by time, "journalctl -p err" shows only error-level and above messages, and "journalctl -b" shows entries from the current boot. The -f flag follows the log in real time (like tail -f), and you can combine multiple filters for precise results like "journalctl -u nginx --since today -p warning". The journal also captures both stdout and stderr from services, making it a one-stop shop for debugging.

**2. Traditional syslog and Log File Locations**

Before systemd, and still in parallel on many systems, the traditional syslog daemon (rsyslog or syslog-ng) writes log messages to plain text files in /var/log/. On Debian and Ubuntu systems, general system messages go to /var/log/syslog, while on Red Hat and CentOS they go to /var/log/messages. Authentication events (login attempts, sudo usage, SSH connections) are logged to /var/log/auth.log, and kernel messages go to /var/log/kern.log. Application-specific logs typically live in their own subdirectories — /var/log/nginx/ for Nginx access and error logs, /var/log/apache2/ for Apache, /var/log/mysql/ for MySQL, and so on. Knowing which file to check for which type of event is essential for efficient troubleshooting. When you are investigating an issue, always start by identifying the relevant log file and then use tools like tail, grep, and less to search through it.

**3. Log Rotation with logrotate**

Without management, log files would grow indefinitely until they fill the disk — a situation that can bring an entire system to a halt. The logrotate utility prevents this by automatically rotating, compressing, and eventually deleting old log files according to configurable policies. The main configuration file is /etc/logrotate.conf, and individual service configurations live in /etc/logrotate.d/. A typical logrotate configuration might specify that logs should be rotated weekly, kept for 4 weeks, compressed after rotation, and that the service should be notified to reopen its log files after rotation. Understanding logrotate is important because misconfigured rotation can lead to either lost log data (rotated too aggressively) or full disks (not rotated often enough). In production environments, you may also want to ship logs to a centralized log server or log management platform for long-term retention and analysis.

**4. Log Analysis Techniques**

Reading raw logs is only the beginning — real power comes from analysis. The grep command is your go-to tool for finding specific patterns: "grep 'error' /var/log/syslog" finds all error messages, and "grep -c 'Failed password' /var/log/auth.log" counts failed login attempts (potentially indicating a brute-force attack). The awk command excels at structured log parsing: "awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn" extracts, counts, and ranks IP addresses by frequency. For real-time monitoring, "tail -f" follows a log file as new entries are written, and you can combine it with grep to filter live output: "tail -f /var/log/syslog | grep error". In production environments, automated log analysis through tools like fail2ban (which automatically bans IPs with too many failed login attempts) or centralized platforms like the ELK stack (Elasticsearch, Logstash, Kibana) transforms passive log files into active security and monitoring infrastructure.`,
					CodeExamples: `# View all system logs
journalctl

# View logs for specific service
journalctl -u nginx
journalctl -u ssh

# Follow logs (real-time)
journalctl -f
journalctl -u nginx -f

# View logs since specific time
journalctl --since "2024-01-15 10:00:00"
journalctl --since "1 hour ago"
journalctl --since yesterday
journalctl --since "2024-01-01" --until "2024-01-31"

# View logs by priority
journalctl -p err    # Errors and above
journalctl -p warning
journalctl -p info

# View kernel logs
journalctl -k
dmesg  # Alternative

# View boot logs
journalctl -b
journalctl -b -0  # Current boot
journalctl -b -1  # Previous boot

# View logs for specific user
journalctl _UID=1000

# View logs for specific executable
journalctl /usr/bin/nginx

# Search logs
journalctl | grep error
journalctl -u nginx | grep "404"

# Export logs
journalctl -u nginx > nginx.log

# Traditional syslog
tail -f /var/log/syslog
tail -f /var/log/auth.log

# View last 100 lines
tail -n 100 /var/log/syslog

# View and follow
tail -f /var/log/nginx/access.log

# Log rotation
cat /etc/logrotate.conf
cat /etc/logrotate.d/nginx

# Manual log rotation
sudo logrotate -f /etc/logrotate.conf

# Count log entries
grep -c "error" /var/log/syslog

# Find recent errors
grep "error" /var/log/syslog | tail -20

# Parse log entries with awk
awk '/error/ {print $1, $2, $5}' /var/log/syslog

# Monitor multiple log files
tail -f /var/log/syslog /var/log/auth.log

# Filter by date
grep "Jan 15" /var/log/syslog

# Count unique IPs from access log
awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          123,
			Title:       "Package Management",
			Description: "Master package management: apt/yum/dnf, installing software, updating systems, and managing repositories.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "APT (Debian/Ubuntu)",
					Content: `APT (Advanced Package Tool) is the package management system used on Debian-based Linux distributions, which include Debian itself, Ubuntu, Linux Mint, and many others. It is one of the most important tools a Linux administrator or developer needs to master, because it is the primary way you install, update, and remove software on these systems. APT works by maintaining a local database of available packages sourced from remote repositories (essentially online software libraries), and it automatically handles dependency resolution — meaning that when you install a package, APT figures out what other packages it needs and installs those too. Think of APT as an app store for the command line, but far more powerful and flexible.

**1. Updating and Upgrading**

The first step in any package management workflow is running "apt update", which downloads the latest package lists from all configured repositories. This does not install or upgrade anything — it simply refreshes your system's knowledge of what packages are available and what versions exist. After updating, "apt upgrade" compares the versions of your installed packages against the repository lists and upgrades any that have newer versions available. It is considered best practice to always run "apt update" before "apt upgrade" or "apt install" to ensure you are working with current information. For security, regularly upgrading your packages is critical because updates often include patches for vulnerabilities that could be exploited by attackers.

**2. Installing, Removing, and Searching for Packages**

To install new software, use "apt install package-name" — APT will download the package and all its dependencies, verify their integrity, and install them. To remove software you no longer need, "apt remove package-name" uninstalls the package but leaves its configuration files behind (in case you want to reinstall later with the same settings), while "apt purge package-name" removes both the package and its configuration files for a completely clean removal. To find packages, "apt search keyword" searches package names and descriptions for the keyword, and "apt show package-name" displays detailed information about a specific package including its version, size, dependencies, and description. The "apt list --installed" command shows all currently installed packages, which is useful for auditing what software is on your system.

**3. dpkg: The Lower-Level Package Tool**

Under the hood, APT relies on dpkg, the Debian package manager, to actually install and manage individual .deb package files. While APT handles repository access, dependency resolution, and download, dpkg handles the physical installation and removal of packages on your system. You interact with dpkg directly when you need to install a .deb file that you downloaded manually (dpkg -i package.deb), list installed packages (dpkg -l), or query which package owns a specific file (dpkg -S /path/to/file). One common workflow is: download a .deb file from a vendor's website, install it with "dpkg -i", and if dependency errors occur, run "apt install -f" to have APT automatically download and install the missing dependencies. Understanding the relationship between APT and dpkg helps you troubleshoot package installation issues and gives you more control over your system's software.`,
					CodeExamples: `# Update package lists
sudo apt update

# Upgrade all packages
sudo apt upgrade

# Install package
sudo apt install nginx

# Remove package
sudo apt remove nginx

# Remove with config files
sudo apt purge nginx

# Search packages
apt search nginx

# Show package info
apt show nginx

# List installed packages
apt list --installed

# Install .deb file
sudo dpkg -i package.deb
sudo apt install -f  # Fix dependencies`,
				},
				{
					Title: "YUM/DNF (Red Hat/Fedora)",
					Content: `YUM (Yellowdog Updater Modified) and its modern successor DNF (Dandified YUM) are the package management tools used on Red Hat-based Linux distributions, including Red Hat Enterprise Linux (RHEL), CentOS, Fedora, Rocky Linux, and AlmaLinux. These distributions dominate the enterprise server market, so understanding their package management system is essential for anyone working in a professional Linux environment. DNF is the default package manager starting with Fedora 22 and RHEL 8, and it offers improved performance, better dependency resolution, and a cleaner codebase compared to the older YUM. In most cases, the commands are nearly identical — you can mentally substitute "dnf" wherever you previously used "yum."

**1. Updating and Installing Packages**

The command "dnf update" (or "dnf upgrade", which is functionally identical) refreshes the repository metadata and upgrades all installed packages to their latest available versions in a single operation — unlike APT, which separates the "update metadata" and "upgrade packages" steps. To install new software, "dnf install package-name" downloads and installs the package along with all its dependencies. DNF displays a summary of what will be installed, upgraded, or removed and asks for confirmation before proceeding, giving you a chance to review the changes. For removing software, "dnf remove package-name" uninstalls the package and, by default, also removes any dependencies that were installed solely for that package and are no longer needed by anything else — a feature called "autoremove" that helps keep your system clean.

**2. Searching and Inspecting Packages**

Before installing a package, you often need to find it or learn more about it. The "dnf search keyword" command searches package names and descriptions, while "dnf info package-name" displays detailed metadata including version, release, architecture, size, repository, and a full description. The "dnf list installed" command shows everything currently installed on the system, and "dnf list available" shows what is available in the configured repositories but not yet installed. DNF also supports "dnf provides /path/to/file", which tells you which package owns a specific file — incredibly useful when you need a command or library that is not installed and you do not know which package provides it.

**3. RPM: The Lower-Level Package Tool**

Just as APT builds on dpkg, DNF and YUM build on RPM (Red Hat Package Manager), which handles the actual installation and management of individual .rpm package files. You use RPM directly when installing a locally downloaded .rpm file with "rpm -i package.rpm" (or "rpm -Uvh" for a more informative upgrade), querying the RPM database with "rpm -qa" (list all installed packages) or "rpm -qi package" (detailed info), and verifying package integrity with "rpm -V package". A common workflow for third-party software is to download the .rpm file, install it with "dnf localinstall package.rpm" (which is preferred over "rpm -i" because it also resolves dependencies from your configured repositories), and then manage it going forward with standard dnf commands. Understanding the RPM layer is also valuable for troubleshooting package conflicts and verifying that installed files have not been tampered with.`,
					CodeExamples: `# Update packages
sudo dnf update

# Install package
sudo dnf install nginx

# Remove package
sudo dnf remove nginx

# Search packages
dnf search nginx

# Show package info
dnf info nginx

# List installed packages
dnf list installed

# Install .rpm file
sudo rpm -i package.rpm`,
				},
				{
					Title: "Repository Management",
					Content: `Software repositories are the centralized servers from which your package manager downloads software, and managing them properly is essential for both getting the software you need and keeping your system secure. Think of repositories like trusted suppliers for a store: your package manager only installs software from suppliers it knows and trusts, and you (as the administrator) control which suppliers are on the approved list. Adding the wrong repository or failing to verify its authenticity can expose your system to tampered or malicious packages, so understanding repository management is as much a security skill as it is a convenience skill.

**1. APT Repositories on Debian/Ubuntu**

On Debian-based systems, repository sources are configured in two places: the file /etc/apt/sources.list and individual files in the /etc/apt/sources.list.d/ directory. Each entry follows the format "deb [options] URL distribution components" — for example, "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe" specifies the URL, the distribution codename (jammy for Ubuntu 22.04), and which components to enable. Ubuntu organizes software into four components: "main" contains officially supported free software, "restricted" contains supported proprietary software (like certain drivers), "universe" contains community-maintained free software, and "multiverse" contains software with restrictive licensing. Understanding these components helps you make informed decisions about what software sources to enable based on your needs and your organization's policies regarding proprietary software.

**2. GPG Keys and Repository Authentication**

Every legitimate repository signs its packages with a GPG (GNU Privacy Guard) key, and your package manager verifies these signatures before installing anything. This cryptographic verification ensures that the packages you download have not been tampered with in transit and actually come from the claimed source. On Debian/Ubuntu, you add a repository's GPG key before adding the repository itself — the modern approach uses the "signed-by" option in the sources.list entry to specify which keyring file to use for verification (for example, [signed-by=/usr/share/keyrings/example-keyring.gpg]). On Red Hat systems, GPG keys are specified in the repository configuration file with the "gpgkey" directive and verified when "gpgcheck=1" is set. Never disable GPG checking in a production environment — while it might seem like a quick fix for a key error, it opens the door to installing compromised packages.

**3. YUM/DNF Repositories on Red Hat/Fedora**

On Red Hat-based systems, repositories are configured through .repo files in the /etc/yum.repos.d/ directory. Each .repo file contains one or more repository sections, identified by a section header like [repository-id], with options including "name" (human-readable description), "baseurl" (the repository URL), "enabled" (1 or 0), "gpgcheck" (1 or 0), and "gpgkey" (URL to the GPG key). You can enable or disable repositories per-transaction using "dnf --enablerepo=repo-name install package" or "dnf --disablerepo=repo-name update", which is useful when a third-party repository conflicts with the base repositories. The "dnf repolist" command shows all configured repositories and their status, while "dnf repoinfo" provides detailed information about a specific repository.

**4. Best Practices for Repository Management**

When managing repositories, follow several important best practices. First, only add repositories from trusted sources — an untrusted repository can install malware or overwrite critical system packages with compromised versions. Second, always verify and install GPG keys before adding a repository. Third, be cautious with third-party repositories that might contain packages that conflict with or override your distribution's official packages — repository priority settings can help manage conflicts. Fourth, after adding any new repository, run "apt update" or "dnf makecache" to refresh your local metadata. Fifth, periodically audit your configured repositories to remove any that are no longer needed or maintained, as stale repositories can cause update failures. Finally, in enterprise environments, consider setting up a local mirror or proxy repository to provide a consistent, controlled source of packages across all your systems.`,
					CodeExamples: `# View APT sources
cat /etc/apt/sources.list
ls /etc/apt/sources.list.d/

# Add repository (Ubuntu)
sudo add-apt-repository "deb http://archive.ubuntu.com/ubuntu jammy main"
sudo add-apt-repository ppa:user/ppa-name

# Remove repository
sudo add-apt-repository --remove "deb ..."

# Update package lists after adding repo
sudo apt update

# Add repository manually
echo "deb http://example.com/repo stable main" | sudo tee /etc/apt/sources.list.d/example.list

# Add GPG key
wget -qO - https://example.com/key.gpg | sudo apt-key add -
# Or use signed-by in sources.list:
# deb [signed-by=/usr/share/keyrings/example-keyring.gpg] http://example.com/repo stable main

# List GPG keys
apt-key list

# View repository information
apt-cache policy package-name

# YUM/DNF repository file
sudo nano /etc/yum.repos.d/example.repo
# Content:
# [example]
# name=Example Repository
# baseurl=http://example.com/repo
# enabled=1
# gpgcheck=1
# gpgkey=http://example.com/repo/RPM-GPG-KEY

# Enable/disable repository
sudo yum-config-manager --enable example
sudo yum-config-manager --disable example

# List repositories
yum repolist
dnf repolist

# View repository info
yum repoinfo example
dnf repoinfo example

# Add repository from URL
sudo yum-config-manager --add-repo http://example.com/repo.repo

# Install GPG key
sudo rpm --import http://example.com/RPM-GPG-KEY`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          124,
			Title:       "Networking Fundamentals",
			Description: "Learn network configuration, SSH, file transfer (SCP/rsync), firewall basics, and network troubleshooting.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Network Configuration",
					Content: `Network configuration is a foundational skill for any Linux system administrator, because nearly every modern system depends on network connectivity to function. Whether you are setting up a web server that needs to listen on a specific IP address, configuring a development machine to communicate with other services, or troubleshooting why a server has lost connectivity, you need to understand how Linux handles networking at the command line. Unlike graphical operating systems where network settings are tucked away in a settings panel, Linux gives you direct, powerful control over every aspect of your network configuration through command-line tools.

**1. The ip Command: Modern Network Configuration**

The "ip" command is the modern, comprehensive tool for network configuration on Linux, replacing the older "ifconfig" and "route" commands. It is part of the iproute2 package and is the recommended tool on all current Linux distributions. The "ip" command is organized around subcommands: "ip addr" (or "ip a") shows all IP addresses assigned to your network interfaces, "ip link" shows the state of network interfaces (whether they are up or down, their MAC addresses, and MTU settings), and "ip route" (or "ip r") displays the routing table that determines how packets are forwarded to their destinations. You can also use "ip" to modify configuration: "ip addr add 192.168.1.100/24 dev eth0" assigns an IP address to an interface, "ip link set eth0 up" brings an interface online, and "ip route add" creates new routing entries. These changes are temporary (lost on reboot) — for permanent configuration, you need to edit the appropriate configuration files for your distribution's network manager (like Netplan on Ubuntu or NetworkManager on Fedora).

**2. Testing Connectivity and DNS**

The most basic network troubleshooting tool is "ping", which sends ICMP echo request packets to a destination and reports whether it responds. Running "ping google.com" tests both DNS resolution and network connectivity in one command — if it fails, "ping 8.8.8.8" (Google's DNS server by IP) helps determine whether the problem is DNS or routing. For more targeted connectivity testing, "curl" and "wget" can test HTTP/HTTPS connections to specific URLs and ports. Understanding the distinction between DNS resolution failures, routing failures, and firewall blocks is essential for efficient troubleshooting.

**3. Monitoring Connections with ss and netstat**

To see what network connections are active on your system and what ports are being listened on, use the "ss" command (the modern replacement for the older "netstat"). Running "ss -tuln" shows all TCP and UDP listening sockets with their port numbers — this is invaluable for verifying that services are running on the expected ports. Adding "-p" shows which process owns each socket ("ss -tulnp"), which helps you identify what program is using a specific port. The "ss -s" command provides a summary of socket statistics. While "netstat" is still available on many systems, "ss" is faster, provides more information, and is the tool you should learn for modern systems. These monitoring tools are essential for security auditing (finding unexpected open ports), troubleshooting (verifying a service is listening), and capacity planning (monitoring connection counts).`,
					CodeExamples: `# Show IP addresses
ip addr
ip a

# Show network interfaces
ip link
ip link show

# Show routing table
ip route
ip r

# Add IP address
sudo ip addr add 192.168.1.100/24 dev eth0

# Bring interface up/down
sudo ip link set eth0 up
sudo ip link set eth0 down

# Legacy ifconfig
ifconfig
ifconfig eth0 up
ifconfig eth0 down

# Test connectivity
ping google.com
ping -c 4 8.8.8.8  # 4 packets

# Show network connections
netstat -tuln
ss -tuln  # Modern alternative`,
				},
				{
					Title: "SSH & Remote Access",
					Content: `SSH (Secure Shell) is the standard protocol for securely accessing remote Linux systems over a network. It encrypts all communication between your local machine and the remote server, protecting your credentials, commands, and data from eavesdroppers. In the real world, SSH is how system administrators manage servers in data centers they may never physically visit, how developers deploy code to production servers, and how DevOps engineers automate infrastructure management across hundreds of machines. Before SSH, remote administration was done with protocols like Telnet and rsh, which transmitted everything — including passwords — in plain text. SSH made all of those obsolete by providing strong encryption by default.

**1. Connecting with SSH**

The basic SSH connection command is "ssh user@hostname", where you specify the username on the remote system and either a hostname or IP address. If the remote SSH server runs on a non-standard port (anything other than the default port 22), you add the -p flag: "ssh -p 2222 user@host". The first time you connect to a new server, SSH displays the server's fingerprint and asks you to confirm it — this is a security measure to prevent man-in-the-middle attacks, and in a professional environment, you should verify this fingerprint through an out-of-band channel (like a secure wiki or a colleague). Once connected, you have a full interactive shell on the remote system and can run commands as if you were sitting at the server's keyboard. You can also run a single command without entering an interactive session: "ssh user@host 'df -h'" executes the "df -h" command on the remote server and returns the output.

**2. SSH Key-Based Authentication**

While SSH supports password authentication, key-based authentication is far more secure and convenient. It works using a pair of cryptographic keys: a private key (which stays on your local machine and must be kept secret) and a public key (which you place on every server you want to access). When you connect, the server challenges your client to prove it possesses the private key without actually transmitting it — this is mathematically secure and immune to brute-force password attacks. To set up key authentication, first generate a key pair with "ssh-keygen -t ed25519" (or "ssh-keygen -t rsa -b 4096" for RSA), then copy your public key to the server with "ssh-copy-id user@host". After this one-time setup, you can log in without typing a password. For additional security, protect your private key with a passphrase and use ssh-agent to cache the unlocked key in memory so you only need to enter the passphrase once per session.

**3. File Transfer with SCP and Rsync**

SSH is not just for interactive sessions — it also provides the foundation for secure file transfer. SCP (Secure Copy) uses the SSH protocol to copy files between your local machine and a remote server: "scp file.txt user@host:/remote/path/" copies a local file to the server, "scp user@host:/remote/file.txt ." copies a remote file locally, and the -r flag enables recursive copying of entire directories. For more sophisticated file synchronization, rsync over SSH is the preferred tool: "rsync -avz directory/ user@host:/remote/path/" synchronizes a local directory to a remote location, intelligently transferring only the differences between source and destination. Rsync is dramatically more efficient than SCP for repeated transfers of large directory trees because it only sends changed portions of files. These tools are the backbone of deployment scripts, backup systems, and any workflow that moves data between machines.`,
					CodeExamples: `# Connect to remote host
ssh user@192.168.1.100
ssh user@example.com

# Connect on custom port
ssh -p 2222 user@host

# Generate SSH key
ssh-keygen -t rsa -b 4096

# Copy public key to server
ssh-copy-id user@host

# SSH with key file
ssh -i ~/.ssh/keyfile user@host

# SCP (Secure Copy)
scp file.txt user@host:/path/
scp -r directory/ user@host:/path/

# Rsync over SSH
rsync -avz file.txt user@host:/path/
rsync -avz directory/ user@host:/path/`,
				},
				{
					Title: "Network Troubleshooting Basics",
					Content: `Network troubleshooting is one of the most important and frequently exercised skills for any Linux system administrator. Network issues can manifest in countless ways — a web application that suddenly stops responding, a database connection that times out, a server that cannot be reached via SSH, or an intermittent slowness that frustrates users. The key to effective troubleshooting is having a systematic approach rather than randomly trying things. Think of it like a doctor diagnosing a patient: you start with broad observations, progressively narrow down the possible causes, and apply targeted tests until you identify the root cause. The tools and techniques in this lesson form your diagnostic toolkit for network problems.

**1. Check Connectivity with ping and traceroute**

The first step in any network troubleshooting session is to determine whether basic connectivity exists. The "ping" command sends ICMP echo request packets to a destination and reports if they are returned, giving you round-trip time and packet loss statistics. Start by pinging the remote host by IP address (to eliminate DNS as a variable), then by hostname (to test DNS resolution). If ping fails, use "traceroute" (or "mtr" for an interactive, continuously-updating version) to trace the path your packets take through the network — this reveals exactly where along the route communication breaks down. For example, if traceroute shows packets reaching your gateway but dying at the next hop, the problem is likely with your ISP or upstream network, not your local configuration. Use "ping -c 4 host" to send exactly 4 packets (otherwise ping runs indefinitely on Linux), and note that some hosts block ICMP, so a lack of ping response does not always mean the host is unreachable.

**2. Diagnose DNS with nslookup, dig, and host**

DNS resolution issues are among the most common network problems and can mimic total connectivity failure. If you can ping an IP address but not a hostname, the problem is almost certainly DNS. The "nslookup" command performs a basic DNS lookup and shows which server answered the query. The "dig" command provides much more detailed output, including the full DNS response with all record types, TTL values, and the authority chain — it is the preferred tool for serious DNS debugging. The simpler "host" command gives a concise answer suitable for quick checks. Your system's DNS configuration lives in /etc/resolv.conf, which lists the DNS servers your system queries. Common DNS problems include: misconfigured /etc/resolv.conf (wrong server addresses), a local DNS cache serving stale entries, or the upstream DNS server being unreachable.

**3. Inspect Network Configuration and Routing**

If connectivity issues persist, examine your local network configuration. Use "ip addr show" to verify that your network interfaces have the correct IP addresses assigned and that the interfaces are in the "UP" state. Use "ip route" to check your routing table — make sure there is a default gateway configured (the "default via" entry) that points to your router. A missing or incorrect default route is a frequent cause of "no internet" issues even when the local network works fine. The "ip route get 8.8.8.8" command is particularly useful — it shows exactly which interface and route would be used to reach a specific destination, which helps diagnose routing issues. For interface-level diagnostics, "ethtool eth0" shows physical link status, speed, and duplex settings, which can reveal hardware problems like a disconnected cable or a speed mismatch.

**4. Check Ports, Connections, and Firewalls**

When a specific service is unreachable even though general connectivity works, the problem usually lies at the port or firewall level. Use "ss -tuln" to verify that the service is listening on the expected port — if it is not listed, the service may have failed to start or may be configured to listen on a different port or interface. Use "sudo lsof -i :80" to see which process is bound to a specific port. If the service is listening correctly, check the firewall: "sudo iptables -L -n -v" shows iptables rules, and "sudo firewall-cmd --list-all" shows firewalld configuration. For testing connectivity to a specific port from another machine, "curl -I http://host:port" tests HTTP, while "nc -zv host port" (netcat) tests arbitrary TCP ports. For comprehensive port scanning, "nmap host" reveals all open ports on a target. When troubleshooting, always work through the layers systematically: physical link, IP configuration, routing, DNS, firewall, and finally application — this methodical approach ensures you do not waste time investigating the wrong layer.`,
					CodeExamples: `# Test connectivity
ping google.com
ping -c 4 8.8.8.8

# Test IPv6
ping6 ipv6.google.com

# DNS lookup
nslookup google.com
dig google.com
host google.com

# Check DNS configuration
cat /etc/resolv.conf

# Test DNS resolution
getent hosts google.com

# Check network interface
ip addr show
ip a
ifconfig

# Check interface status
ip link show
ethtool eth0  # Interface details

# Bring interface up/down
sudo ip link set eth0 up
sudo ip link set eth0 down

# Check routing table
ip route
ip r
route -n

# Add route
sudo ip route add 192.168.1.0/24 via 192.168.0.1

# Delete route
sudo ip route del 192.168.1.0/24

# Trace network path
traceroute google.com
mtr google.com  # Interactive traceroute

# Check listening ports
netstat -tuln
ss -tuln  # Modern alternative

# Check connections
netstat -an | grep ESTABLISHED
ss -tun | grep ESTABLISHED

# Check which process uses port
sudo lsof -i :80
sudo ss -tlnp | grep :80

# Scan ports
nmap localhost
nmap 192.168.1.1

# Check firewall
sudo iptables -L
sudo firewall-cmd --list-all

# Test HTTP connectivity
curl -I http://example.com
wget --spider http://example.com

# Check network statistics
netstat -s
ss -s

# Monitor network traffic
sudo tcpdump -i eth0
sudo tcpdump -i eth0 port 80

# Common troubleshooting sequence
# 1. Check if interface is up
ip link show eth0

# 2. Check IP configuration
ip addr show eth0

# 3. Test local connectivity
ping -c 4 $(ip route | grep default | awk '{print $3}')

# 4. Test DNS
nslookup google.com

# 5. Test external connectivity
ping -c 4 google.com

# 6. Check routing
ip route get 8.8.8.8

# 7. Check firewall
sudo iptables -L -n -v`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
