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
			},
			ProblemIDs: []int{},
		},
	})
}
