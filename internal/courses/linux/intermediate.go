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
					Content: `Shell scripting automates tasks and combines commands. Bash is the most common shell scripting language.

**What is a Shell Script?**
- Text file containing commands
- Executed by shell interpreter
- Automates repetitive tasks
- Combines multiple commands

**Script Structure:**
1. Shebang: `#!/bin/bash`
2. Comments: `# This is a comment`
3. Commands: Actual shell commands
4. Variables, conditionals, loops

**Creating a Script:**
1. Create file: `nano script.sh`
2. Add shebang: `#!/bin/bash`
3. Add commands
4. Make executable: `chmod +x script.sh`
5. Run: `./script.sh` or `bash script.sh`

**Why Script?**
- Automate repetitive tasks
- Combine multiple commands
- Create reusable tools
- Schedule tasks (cron)
- System administration`,
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
					Content: `Variables store data in scripts. Understanding variables is fundamental to scripting.

**Variable Assignment:**
- `VARIABLE=value` (no spaces around =)
- No $ when assigning
- Use $ when referencing: `$VARIABLE`

**Variable Types:**
- **Strings**: `NAME="John"`
- **Numbers**: `COUNT=10`
- **Command output**: `DATE=$(date)` or `DATE=`date``

**Special Variables:**
- `$0`: Script name
- `$1, $2, ...`: Command line arguments
- `$#`: Number of arguments
- `$@`: All arguments
- `$?`: Exit status of last command
- `$$`: Process ID

**Variable Best Practices:**
- Use uppercase for constants
- Use lowercase for variables
- Quote strings with spaces
- Use `{}` for clarity: `${VARIABLE}``,
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
CURRENT_TIME=`date +%H:%M:%S`
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
					Content: `Conditionals allow scripts to make decisions based on conditions.

**if Statement:**
\`\`\`bash
if [ condition ]; then
    commands
fi
\`\`\`

**if-else:**
\`\`\`bash
if [ condition ]; then
    commands
else
    commands
fi
\`\`\`

**if-elif-else:**
\`\`\`bash
if [ condition1 ]; then
    commands
elif [ condition2 ]; then
    commands
else
    commands
fi
\`\`\`

**Test Conditions:**
- `[ -f file ]`: File exists and is regular file
- `[ -d dir ]`: Directory exists
- `[ -r file ]`: File is readable
- `[ -w file ]`: File is writable
- `[ -x file ]`: File is executable
- `[ -z string ]`: String is empty
- `[ -n string ]`: String is not empty
- `[ str1 = str2 ]`: Strings are equal
- `[ str1 != str2 ]`: Strings are not equal
- `[ num1 -eq num2 ]`: Numbers are equal
- `[ num1 -lt num2 ]`: num1 less than num2
- `[ num1 -gt num2 ]`: num1 greater than num2`,
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
					Content: `Loops repeat commands multiple times. Essential for processing lists and automating repetitive tasks.

**for Loop:**
\`\`\`bash
for variable in list; do
    commands
done
\`\`\`

**while Loop:**
\`\`\`bash
while [ condition ]; do
    commands
done
\`\`\`

**until Loop:**
\`\`\`bash
until [ condition ]; do
    commands
done
\`\`\`

**Loop Control:**
- `break`: Exit loop
- `continue`: Skip to next iteration
- `exit`: Exit script

**Common Patterns:**
- Iterate over files: `for file in *.txt; do`
- Iterate over numbers: `for i in {1..10}; do`
- Read file line by line: `while read line; do``,
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
					Content: `Functions group commands for reuse. They make scripts modular and maintainable.

**Function Syntax:**
\`\`\`bash
function_name() {
    commands
    return value
}
\`\`\`

**Alternative Syntax:**
\`\`\`bash
function function_name {
    commands
    return value
}
\`\`\`

**Function Features:**
- Accept arguments: `$1, $2, ...`
- Return exit status: `return 0` (success) or `return 1` (failure)
- Local variables: `local var=value`
- Can call other functions

**Best Practices:**
- Use descriptive names
- Keep functions focused (single responsibility)
- Use local variables
- Return meaningful exit codes`,
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
					Content: `Debugging scripts is essential for creating reliable automation. Learn techniques to find and fix errors.

**Common Debugging Techniques:**

**bash -x (Debug Mode):**
- Shows each command before execution
- Displays variable expansions
- Traces script execution
- `bash -x script.sh` or `set -x` in script

**set Options:**
- `set -x`: Enable debug mode (trace execution)
- `set -e`: Exit on error
- `set -u`: Exit on undefined variable
- `set -o pipefail`: Exit on pipe failure
- `set +x`: Disable debug mode

**Common Errors:**
- **Syntax errors**: Missing quotes, brackets, semicolons
- **Variable errors**: Unset variables, wrong variable names
- **Logic errors**: Wrong conditions, infinite loops
- **Permission errors**: Script not executable, file permissions

**Debugging Tools:**
- **bash -x**: Trace execution
- **bash -n**: Syntax check (no execution)
- **shellcheck**: Static analysis tool
- **echo**: Print variable values
- **set -x**: Enable tracing

**Best Practices:**
- Test scripts incrementally
- Use `set -euo pipefail` for strict mode
- Add echo statements for debugging
- Check exit codes
- Validate input`,
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
					Content: `Arrays store multiple values. Useful for processing lists of items.

**Array Declaration:**
- `ARRAY=(item1 item2 item3)`
- `ARRAY[0]=item1`
- Access: `${ARRAY[0]}`
- All elements: `${ARRAY[@]}` or `${ARRAY[*]}`
- Length: `${#ARRAY[@]}`

**Array Operations:**
- Add element: `ARRAY+=("new")`
- Slice: `${ARRAY[@]:start:length}`
- Indices: `${!ARRAY[@]}``,
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
					Content: `Bash provides powerful string manipulation capabilities.

**String Operations:**
- Length: `${#STRING}`
- Substring: `${STRING:start:length}`
- Replace: `${STRING/old/new}` (first) or `${STRING//old/new}` (all)
- Remove prefix: `${STRING#pattern}` or `${STRING##pattern}`
- Remove suffix: `${STRING%pattern}` or `${STRING%%pattern}`
- Uppercase: `${STRING^^}`
- Lowercase: `${STRING,,}``,
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
					Content: `Proper error handling makes scripts robust and reliable.

**Exit Codes:**
- `0`: Success
- Non-zero: Failure
- Check with `$?`

**Error Handling Options:**
- `set -e`: Exit on error
- `set -u`: Exit on undefined variable
- `set -o pipefail`: Exit on pipe failure
- `trap`: Handle errors and signals
- `||`: Execute if command fails
- `&&`: Execute if command succeeds`,
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
					Content: `Reading and writing files is fundamental to shell scripting. Master file operations for data processing and automation.

**Reading Files:**

**while read Loop:**
- Read file line by line
- Handles large files efficiently
- Preserves whitespace with IFS

**read Command:**
- Read input into variables
- `read var`: Read into variable
- `read -r`: Don't interpret backslashes
- `read -a`: Read into array

**File Redirection:**
- `< file`: Input redirection
- `> file`: Output redirection (overwrite)
- `>> file`: Append redirection
- `<<`: Here document

**Writing Files:**
- `echo > file`: Write (overwrite)
- `echo >> file`: Append
- `cat > file`: Write from stdin
- `cat >> file`: Append from stdin

**Here Documents:**
- Multi-line input
- `<< EOF`: Start here document
- `EOF`: End marker (can be any word)

**Here Strings:**
- `<<< string`: Pass string as input
- Shorthand for echo | command

**File Descriptors:**
- `0`: stdin
- `1`: stdout
- `2`: stderr
- `3+`: Custom file descriptors`,
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
					Content: `Managing users is fundamental to Linux administration.

**User Management Commands:**
- `useradd`: Create user
- `usermod`: Modify user
- `userdel`: Delete user
- `passwd`: Change password
- `id`: Show user ID and groups
- `whoami`: Show current user
- `w` or `who`: Show logged in users

**User Files:**
- `/etc/passwd`: User accounts
- `/etc/shadow`: Encrypted passwords
- `/etc/group`: Groups

**Common Operations:**
- Create user: `useradd -m username`
- Set password: `passwd username`
- Add to group: `usermod -aG groupname username`
- Lock account: `usermod -L username`
- Unlock account: `usermod -U username``,
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
					Content: `systemd manages system services. Understanding it is essential for modern Linux administration.

**systemctl Commands:**
- `systemctl start service`: Start service
- `systemctl stop service`: Stop service
- `systemctl restart service`: Restart service
- `systemctl reload service`: Reload configuration
- `systemctl status service`: Check status
- `systemctl enable service`: Enable at boot
- `systemctl disable service`: Disable at boot
- `systemctl list-units`: List all units

**Service States:**
- **active**: Running
- **inactive**: Stopped
- **failed**: Failed to start
- **enabled**: Will start at boot
- **disabled**: Won't start at boot`,
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
					Content: `Managing logs is crucial for system administration, troubleshooting, and security auditing.

**System Logs:**

**journalctl (systemd):**
- Query systemd journal
- Centralized logging (systemd systems)
- Binary format, indexed
- `journalctl`: View logs
- `journalctl -u service`: Service-specific logs
- `journalctl -f`: Follow logs (like tail -f)
- `journalctl --since "1 hour ago"`: Time-based filtering

**syslog (Traditional):**
- `/var/log/syslog`: System messages (Debian/Ubuntu)
- `/var/log/messages`: System messages (Red Hat)
- `/var/log/auth.log`: Authentication logs
- `/var/log/kern.log`: Kernel messages

**Common Log Files:**
- `/var/log/auth.log`: Authentication attempts
- `/var/log/syslog`: General system logs
- `/var/log/apache2/`: Apache web server logs
- `/var/log/nginx/`: Nginx web server logs
- `/var/log/mail.log`: Mail server logs
- `/var/log/cron.log`: Cron job logs

**Log Rotation:**
- Prevents logs from filling disk
- Managed by `logrotate`
- Configuration: `/etc/logrotate.conf`
- Service configs: `/etc/logrotate.d/`

**Log Analysis:**
- `grep`: Search logs
- `tail -f`: Follow log in real-time
- `less`: View logs page by page
- `awk`: Process log entries
- `journalctl`: Query systemd logs`,
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
					Content: `APT (Advanced Package Tool) manages packages on Debian-based systems.

**Common apt Commands:**
- `apt update`: Update package lists
- `apt upgrade`: Upgrade installed packages
- `apt install package`: Install package
- `apt remove package`: Remove package
- `apt search keyword`: Search packages
- `apt show package`: Show package info
- `apt list --installed`: List installed packages

**dpkg:**
- Lower-level package tool
- `dpkg -i package.deb`: Install .deb file
- `dpkg -l`: List installed packages`,
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
					Content: `YUM/DNF manages packages on Red Hat-based systems. DNF is the newer version.

**Common dnf/yum Commands:**
- `dnf update`: Update packages
- `dnf install package`: Install package
- `dnf remove package`: Remove package
- `dnf search keyword`: Search packages
- `dnf info package`: Show package info
- `dnf list installed`: List installed packages

**rpm:**
- Lower-level package tool
- `rpm -i package.rpm`: Install .rpm file
- `rpm -qa`: List installed packages`,
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
					Content: `Managing software repositories is essential for package management and system security.

**APT Repositories (Debian/Ubuntu):**

**Repository Sources:**
- `/etc/apt/sources.list`: Main repository list
- `/etc/apt/sources.list.d/`: Additional repository files
- Format: `deb [options] URL distribution components`

**Repository Components:**
- **main**: Officially supported software
- **restricted**: Supported but not free
- **universe**: Community-maintained
- **multiverse**: Not free software

**Adding Repositories:**
- `add-apt-repository`: Add repository (Ubuntu)
- Edit `/etc/apt/sources.list` manually
- Add `.list` file to `/etc/apt/sources.list.d/`

**GPG Keys:**
- Repositories are signed with GPG keys
- Keys stored in `/etc/apt/trusted.gpg.d/`
- `apt-key`: Manage keys (deprecated, use signed-by)

**YUM/DNF Repositories (Red Hat/Fedora):**

**Repository Files:**
- `/etc/yum.repos.d/`: Repository configuration files
- `.repo` files define repositories
- Format: `[repository-id]` with options

**Repository Options:**
- `baseurl`: Repository URL
- `gpgcheck`: Verify GPG signatures
- `enabled`: Enable/disable repository
- `gpgkey`: GPG key URL

**Managing Repositories:**
- Enable/disable repositories
- Add third-party repositories
- Configure repository priorities
- Handle GPG keys`,
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
					Content: `Configuring network interfaces is essential for system administration.

**Network Commands:**
- `ip`: Modern network configuration tool
- `ifconfig`: Legacy tool (may need installation)
- `ping`: Test connectivity
- `netstat`: Network connections
- `ss`: Modern netstat replacement

**ip Command:**
- `ip addr`: Show IP addresses
- `ip link`: Show network interfaces
- `ip route`: Show routing table
- `ip addr add`: Add IP address`,
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
					Content: `SSH (Secure Shell) provides secure remote access to Linux systems.

**SSH Basics:**
- `ssh user@host`: Connect to remote host
- `ssh -p port user@host`: Connect on specific port
- `ssh-keygen`: Generate SSH keys
- `ssh-copy-id`: Copy public key to remote host

**SSH Keys:**
- More secure than passwords
- Public key on server, private key on client
- No password needed after setup`,
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
					Content: `Network troubleshooting is essential for system administrators. Learn fundamental techniques to diagnose and fix network issues.

**Basic Troubleshooting Steps:**

**1. Check Connectivity:**
- `ping`: Test network connectivity
- `ping -c 4 host`: Send 4 packets
- `ping6`: IPv6 ping

**2. Check DNS Resolution:**
- `nslookup`: Query DNS
- `dig`: DNS lookup tool
- `host`: Simple DNS lookup
- `/etc/resolv.conf`: DNS configuration

**3. Check Network Configuration:**
- `ip addr`: Show IP addresses
- `ip link`: Show interfaces
- `ifconfig`: Legacy tool
- `hostname -I`: Show IP addresses

**4. Check Routing:**
- `ip route`: Show routing table
- `route -n`: Legacy routing table
- `traceroute`: Trace network path
- `mtr`: Network diagnostic tool

**5. Check Ports and Connections:**
- `netstat`: Network connections
- `ss`: Modern netstat replacement
- `lsof`: List open files/ports
- `nmap`: Network scanner

**Common Issues:**
- **No connectivity**: Check interface, routing, firewall
- **DNS not working**: Check resolv.conf, DNS server
- **Port not accessible**: Check firewall, service status
- **Slow connection**: Check bandwidth, latency, routing`,
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
