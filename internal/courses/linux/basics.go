package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          110,
			Title:       "Introduction to Linux",
			Description: "Learn what Linux is, its history, distributions, and why it's essential for modern computing.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is Linux?",
					Content: `Linux is a free and open-source Unix-like operating system kernel originally created by Linus Torvalds in 1991. While the term "Linux" technically refers only to the kernel, it is commonly used to describe the entire operating system formed by combining the kernel with GNU utilities, system libraries, and various applications.

**Technical Definition:**
At its core, Linux is a **monolithic kernel**. This means the entire operating system is working in kernel space and is alone in supervisor mode. It handles process management, memory management, hardware device drivers, and system calls. However, it is modular, allowing drivers and extensions to be loaded and unloaded at runtime.

**The GNU/Linux Relationship:**
The software we call "Linux" is usually a combination of:
1. **The Linux Kernel:** Manages the hardware.
2. **GNU Utilities:** The shell (bash), core utilities (ls, cp, rm), and compilers (gcc) provided by the Free Software Foundation.
3. **System Libraries:** Like glibc, which provide the interface for applications to talk to the kernel.
4. **Desktop Environments/Applications:** GNOME, KDE, Web Browsers, etc.

**Historical Context:**

**The Birth of Linux:**
In 1991, Linus Torvalds, a 21-year-old student at the University of Helsinki, announced he was working on a free operating system kernel as a hobby. He famously stated it "won't be big and professional like gnu". Paradoxically, the combination of his kernel with the GNU project's tools created the professional-grade system we use today.

**Key Milestones:**
- **1991**: Linus Torvalds releases Linux kernel version 0.01 (10,000 lines of code).
- **1992**: Linux kernel licensed under GPL v2, ensuring it remains free forever.
- **1993**: Slackware and Debian are founded, bringing Linux to the masses.
- **1996**: Version 2.0 released; Linux supports multiple processors (SMP).
- **2001**: Version 2.4 released; support for USB, PC Card, and better journaling filesystems.
- **2011**: Linus releases version 3.0 to mark the 20th anniversary.
- **2020s**: Linux powers 100% of the world's top 500 supercomputers and nearly all cloud infrastructure.

**What Makes Linux Special:**

**1. Open Source (GPL):**
- **Transparency**: Anyone can inspect the code for backdoors or vulnerabilities.
- **Collaboration**: Thousands of developers from competing companies (Intel, AMD, Google, Microsoft) work together on the kernel.
- **Freedom**: You are never locked into a single vendor.

**2. The Unix Philosophy:**
Linux is built on the idea that "everything is a file" and "small programs should do one thing well and work together."
- **Pipes (|)**: The power to connect the output of one tool to the input of another is the secret to Linux's CLI efficiency.
- **Text Streams**: Standard protocols for communication between tools.

**3. Stability and Reliability:**
- **Kernel/User Space Separation**: A crashing application (user space) cannot easily bring down the entire system (kernel space).
- **Zero-Downtime Updates**: Many kernel updates can now be applied without rebooting using technologies like kpatch or kgraft.

**4. Security Architecture:**
- **DAC (Discretionary Access Control)**: Traditional owner/group/other permissions.
- **MAC (Mandatory Access Control)**: Advanced security layers like SELinux or AppArmor that restrict even the root user's potential impact.

**Linux vs Other Operating Systems:**

**Linux:**
- **Kernel Type**: Monolithic (modular).
- **License**: Open Source (GPL).
- **Filesystem**: Case-sensitive (ext4, XFS, Btrfs).
- **Performance**: Extremely high efficiency, especially under heavy network load.

**Windows:**
- **Kernel Type**: Hybrid (NT kernel).
- **License**: Proprietary.
- **Filesystem**: Case-insensitive (NTFS).
- **Design Focus**: Consistency and user-friendliness for desktop users.

**macOS:**
- **Kernel Type**: Hybrid (XNU - Mach + BSD).
- **License**: Proprietary (though based on open-source Darwin).
- **Design Focus**: Seamless hardware/software integration and premium user experience.

**Why Linux Matters Today:**

**1. Infrastructure & Cloud:**
- **AWS, GCP, Azure**: These "Clouds" are essentially massive Linux data centers.
- **Hypervisors**: KVM (Kernel-based Virtual Machine) turns Linux into a world-class platform for virtualization.

**2. Containerization:**
Docker and Kubernetes rely on Linux-specific features like **Cgroups** (resource limiting) and **Namespaces** (isolation). Without Linux, modern microservices wouldn't exist in their current form.

**3. Embedded & Android:**
- **Smartphones**: Android is built on top of the Linux kernel.
- **IoT**: From smart fridges to space rovers (NASA's Perseverance rover runs Linux), it is the go-to OS for reliability.

**4. Development Environment:**
Most modern languages (Python, Go, Node.js, Rust) are "Linux-first." Developing on Linux provides the closest environment to production for most backend developers.`,

					CodeExamples: `# Check Linux version
uname -a
# Output: Linux hostname 5.15.0-generic #1-Ubuntu SMP ...

# Check distribution
cat /etc/os-release
# Output shows distribution name, version, etc.

# Check kernel version
uname -r
# Output: 5.15.0-generic

# System information
hostnamectl
# Shows hostname, OS, kernel, architecture

# Check if running Linux
uname -s
# Output: Linux`,
				},
				{
					Title: "Linux Distributions",
					Content: `A Linux distribution (often called a "distro") is an operating system built from a software collection that includes the Linux kernel, system libraries, applications, and package management tools. While Linux itself is just the kernel, distributions package everything needed to create a complete, usable operating system.

**Categorization by Release Model:**

**1. Fixed Release (Point Release):**
- **How it works**: Versions are released at fixed intervals (e.g., every 6 months or 2 years). Each version is supported for a specific time.
- **Pros**: Predictability, extreme stability, tested component harmony.
- **Cons**: Software can become outdated. Upgrading between major versions can be risky.
- **Examples**: Ubuntu LTS, Debian Stable, RHEL.

**2. Rolling Release:**
- **How it works**: There are no "versions" of the OS. You install once and update continuously. Packages are updated as soon as they are stable.
- **Pros**: Always have the latest software, no major OS upgrade re-installs.
- **Cons**: Higher risk of breakage after an update. Requires more user intervention.
- **Examples**: Arch Linux, Gentoo, openSUSE Tumbleweed.

**Major Distribution Families:**

**1. The Debian Family (The "People's" OS):**
- **Debian**: Known for its strict adherence to free software and stability. It uses the '.deb' package format and 'apt' manager.
- **Ubuntu**: Built on Debian, it made Linux accessible to non-technical users. It introduced "LTS" (Long Term Support) versions, which are the industry standard for production servers.
- **Kali Linux**: A specialized Debian derivative pre-loaded with hundreds of security and penetration testing tools.

**2. The Red Hat Family (The Enterprise Titan):**
- **RHEL (Red Hat Enterprise Linux)**: The gold standard for enterprise. Requires a paid subscription for support and updates. Uses '.rpm' packages and 'dnf'.
- **Fedora**: The upstream for RHEL. It's where new features (like Wayland or Systemd) are tested before they reach the enterprise.
- **Rocky Linux / AlmaLinux**: Community-driven, bug-for-bug compatible clones of RHEL that emerged after CentOS changed its model.

**3. The Arch Family (The DIY Choice):**
- **Arch Linux**: Follows the KISS (Keep It Simple, Stupid) principle. It provides a minimal base, and the user must manually configure everything.
- **Manjaro**: An Arch-based distro that adds a GUI and a more user-friendly layer while retaining access to the massive **'AUR' (Arch User Repository)**.

**Package Management Internals:**

**What is a Package?**
A package is a compressed archive ('.deb' or '.rpm') containing:
- **Binaries**: The compiled program code.
- **Metadata**: Name, version, description, and dependencies.
- **Scripts**: Instructions for tasks to run before or after installation (e.g., creating a system user).

**How Repositories Work:**
Repositories are central servers hosting thousands of packages.
1. **Index Update**: When you run `apt update`, you download a list of available packages and versions.
2. **Dependency Resolution**: When installing `vlc`, the manager checks if `libavcodec` is installed. If not, it fetches it automatically.
3. **Verification**: Packages are digitally signed. Your system checks the signature against a trusted public key to ensure the software hasn't been tampered with.

**Universal Package Formats:**
To solve the "one binary for all distros" problem, new formats have emerged:
- **Snap (Canonical)**: Bundles dependencies, runs in a sandbox.
- **Flatpak (Community)**: Focused on desktop apps, excellent isolation.
- **AppImage**: A single file that "just runs" without installation.`,
					CodeExamples: `# Package Manager Identity
# Debian/Ubuntu uses APT
# Red Hat/Fedora uses DNF
# Arch uses PACMAN

# SEARCHING for a package
# Ubuntu: apt search nginx
# Fedora: dnf search nginx
# Arch: pacman -Ss nginx

# INSTALLING a package
# Ubuntu: sudo apt install htop
# Fedora: sudo dnf install htop
# Arch: sudo pacman -S htop

# REMOVING a package (and its config)
# Ubuntu: sudo apt purge htop
# Fedora: sudo dnf remove htop
# Arch: sudo pacman -Rns htop

# SYSTEM UPGRADE
# Ubuntu: sudo apt update && sudo apt upgrade
# Fedora: sudo dnf upgrade
# Arch: sudo pacman -Syu`,
				},

				{
					Title: "Getting Started with Linux",
					Content: `Starting your Linux journey requires understanding the basics of accessing and using the system.

**Ways to Use Linux:**

**1. Virtual Machine:**
- Install VirtualBox or VMware
- Download Linux ISO
- Create VM and install Linux
- Safe way to learn without affecting main OS

**2. Dual Boot:**
- Install Linux alongside Windows/macOS
- Choose OS at boot time
- Requires partitioning disk

**3. WSL (Windows Subsystem for Linux):**
- Run Linux on Windows
- Good for development
- Easy to set up

**4. Cloud Instance:**
- AWS EC2, Google Cloud, DigitalOcean
- Access via SSH
- Real Linux server experience

**5. Live USB:**
- Boot from USB without installing
- Test Linux before installing
- Portable Linux environment

**First Steps:**
1. **Open Terminal**: Access command line interface
2. **Learn Basic Commands**: Start with ls, cd, pwd
3. **Understand File System**: Navigate directories
4. **Practice Regularly**: Hands-on experience is key

**Terminal vs GUI:**
- **Terminal**: Powerful, efficient, scriptable
- **GUI**: Visual, intuitive, good for beginners
- Most Linux work happens in terminal

**Getting Help:**
- **man pages**: Manual for commands (man ls)
- **--help flag**: Quick help (ls --help)
- **info**: Detailed documentation (info command)
- **Online**: Stack Overflow, Arch Wiki, Ubuntu Forums`,
					CodeExamples: `# Open terminal (usually Ctrl+Alt+T or search "Terminal")

# Check current user
whoami
# Output: username

# Check current directory
pwd
# Output: /home/username

# List files
ls

# List files with details
ls -l

# List all files (including hidden)
ls -la

# Get help for a command
man ls
# Press 'q' to quit

# Quick help
ls --help

# Clear terminal
clear
# Or Ctrl+L

# Exit terminal
exit
# Or Ctrl+D`,
				},
				{
					Title: "Linux History & Evolution",
					Content: `Understanding Linux's history helps appreciate its design philosophy and current state.

**Unix Origins:**
- **1969**: Unix created at Bell Labs by Ken Thompson and Dennis Ritchie
- **1970s**: Unix spread to universities and commercial systems
- **1980s**: Commercial Unix variants (AIX, Solaris, HP-UX)
- **1983**: GNU Project started by Richard Stallman (free Unix-like OS)

**Linux Creation:**
- **1991**: Linus Torvalds, a Finnish student, created Linux kernel
- **1992**: Linux kernel released under GPL (GNU General Public License)
- **1993**: First distributions (Slackware, Debian)
- **1990s**: Rapid growth, adoption by developers and companies

**Key Milestones:**
- **1994**: Linux 1.0 released
- **1996**: Linux mascot Tux created
- **1998**: Major companies (IBM, Oracle) support Linux
- **2000s**: Enterprise adoption, Android (Linux-based) launched
- **2010s**: Cloud computing boom (Linux-powered)
- **2020s**: Dominance in cloud, containers, supercomputers

**GNU/Linux:**
- Linux is just the kernel
- GNU provides userland tools (bash, coreutils, etc.)
- Together: GNU/Linux operating system
- Often just called "Linux" (kernel name became OS name)

**Linux Philosophy:**
- **Open Source**: Source code freely available
- **Community-Driven**: Developed by volunteers and companies
- **Modular**: Kernel + userland tools
- **Portable**: Runs on many architectures
- **Free**: Free as in freedom (and often free as in cost)

**Why Linux Succeeded:**
- Free and open-source
- Runs on commodity hardware
- Strong community support
- Corporate backing (Red Hat, Canonical, etc.)
- Excellent for servers and development
- Powers the internet and cloud`,
					CodeExamples: `# Check kernel version and release date
uname -r
# Output: 5.15.0-generic

# Kernel release info
cat /proc/version
# Output: Linux version 5.15.0-91-generic (buildd@...) #102-Ubuntu SMP ...

# Distribution history
cat /etc/os-release
# Shows distribution name, version, release date

# Check GNU tools version
bash --version
# Output: GNU bash, version 5.1.16(1)-release

# Kernel compilation date
dmesg | head -1
# Shows kernel boot messages with timestamp

# System uptime (shows when kernel was booted)
uptime
# Output: up X days, Y hours

# Kernel modules (show kernel extensibility)
lsmod | head
# Lists loaded kernel modules`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          111,
			Title:       "Command Line Fundamentals",
			Description: "Master essential command line operations: navigation, file operations, and basic commands.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Terminal Basics",
					Content: `The terminal is your direct interface to the power of Linux. To master it, you must understand the distinction between three related but different concepts:

**1. The Terminal Emulator**: This is the GUI application you run (like GNOME Terminal, Alacritty, or VS Code's integrated terminal). It mimics the behavior of old physical hardware terminals.
**2. The Shell**: This is the "brain" or the interpreter. It takes your text input, understands it, and tells the OS what to do. The most common shell is **Bash**.
**3. The TTY (Teletype)**: This refers to the underlying device driver that handles the communication. You can access "pure" TTYs (no GUI) on most Linux systems by pressing Ctrl+Alt+F2 through F6.

**Shell Features to Know:**
- **Globbing**: Using wildcards like ` + "`" + `*` + "`" + ` (match anything) or ` + "`" + `?` + "`" + ` (match one char).
- **Expansion**: The shell "expands" your commands before running them. For example, ` + "`" + `echo {1..3}` + "`" + ` becomes ` + "`" + `1 2 3` + "`" + `.
- **Redirection**: Using ` + "`" + `>` + "`" + `, ` + "`" + `>>` + "`" + `, and ` + "`" + `<` + "`" + ` to move data between files and commands.

**Standard Streams:**
Every command has three streams:
1. **stdin (0)**: Standard Input (usually your keyboard).
2. **stdout (1)**: Standard Output (usually your screen).
3. **stderr (2)**: Standard Error (where error messages go).`,
					CodeExamples: `# WHAT SHELL AM I USING?
echo $0
# or
ps -p $$

# EXPANSION EXAMPLES
# Brace expansion
echo photo_{2021,2022,2023}.jpg
# Output: photo_2021.jpg photo_2022.jpg photo_2023.jpg

# GLOBBING EXAMPLES
# List all .png files
ls *.png
# List files with a single character as name
ls ?.txt

# REDIRECON BASICS
# Save output to a file (overwrites)
echo "Hello" > file.txt
# Append to a file
echo "World" >> file.txt

# SYSTEM PROMPT SYMBOLS
# $ = Regular User
# # = Root (Superuser) - use with CAUTION!`,
				},

				{
					Title: "Navigation Commands",
					Content: `The Linux directory structure is a tree, starting from the **Root (/)**. Navigating it efficiently is the hallmark of a power user.

**Path Types Explained:**
- **Absolute Path**: Starts from the root (e.g., ` + "`" + `/var/log/nginx` + "`" + `). It works no matter where you currently are.
- **Relative Path**: Starts from your current location (e.g., ` + "`" + `../config` + "`" + `).
- **The Home (~)**: Your personal space. ` + "`" + `cd ~` + "`" + ` always takes you back home.

**Advanced Navigation:**
- **cd -**: A "back" button for the terminal. It returns you to the immediately previous directory.
- **pushd / popd**: A "stack" for directories. ` + "`" + `pushd` + "`" + ` saves your current spot and moves you, while ` + "`" + `popd` + "`" + ` brings you back.
- **ls -F**: Adds a trailing symbol to names to show their type (` + "`" + `/` + "`" + ` for directory, ` + "`" + `*` + "`" + ` for executable).`,
					CodeExamples: `# WHERE AM I?
pwd

# THE QUICK "BACK" BUTTON
cd /etc/nginx
cd /var/log
cd -
# Now you are back in /etc/nginx!

# USING THE DIRECTORY STACK
pushd /tmp
# Do some work in /tmp...
popd
# You are back exactly where you were.

# LISTING SECRETS
# -a shows hidden files (starting with .)
# -h shows "Human Readable" sizes (KB, MB, GB)
ls -lah

# RECURSIVE LISTING (The tree view)
ls -R
# or if installed:
tree`,
				},

				{
					Title: "File Operations",
					Content: `Managing files is the core of system administration. Understanding the nuances of these commands prevents data loss and improves productivity.

**1. Copying (cp):**
- **Recursive (` + "`" + `-r` + "`" + `)**: Mandatory for copying directories.
- **Archive (` + "`" + `-a` + "`" + `)**: The preferred way to copy. It preserves permissions, ownership, and links.
- **Safety (` + "`" + `-i` + "`" + `)**: Always use this to avoid accidentally overwriting a file with the same name.

**2. Moving and Renaming (mv):**
In Linux, renaming is just moving a file to a new name in the same directory.
- **Tip**: You can move an entire directory just like a file. No ` + "`" + `-r` + "`" + ` is needed for ` + "`" + `mv` + "`" + `.

**3. The Danger of Deletion (rm):**
**WARNING**: There is no "Trash" or "Recycle Bin" in the terminal. Once deleted with ` + "`" + `rm` + "`" + `, a file is gone.
- **The -rf Flag**: ` + "`" + `r` + "`" + ` (recursive) and ` + "`" + `f` + "`" + ` (force). This is extremely powerful and dangerous. Use with extreme caution.
- **Best Practice**: Before running ` + "`" + `rm -rf` + "`" + ` on a wildcard (like ` + "`" + `rm -rf *` + "`" + `), run ` + "`" + `ls *` + "`" + ` first to see exactly what will be deleted.

**4. Creating Files and Directories:**
- **mkdir -p**: Creates nested structures in one go (e.g., ` + "`" + `mkdir -p project/src/utils` + "`" + `).
- **touch**: Not just for creating empty files; its primary purpose is to update the access/modification time of a file.`,
					CodeExamples: `# THE SAFE WAY TO COPY
cp -ai source_folder backup_folder

# RENAMING A DIRECTORY
mv old_name new_name

# PREVIEW BEFORE DELETE
# Use ls to see what your wildcard matches
ls logs/*.tmp
# If it looks correct, then delete:
rm logs/*.tmp

# CREATE NESTED DIRS IN ONE GO
mkdir -p app/{src,bin,docs,test}
# This creates 4 directories inside "app" using brace expansion!

# DELETE AN EMPTY DIRECTORY
rmdir empty_folder`,
				},

				{
					Title: "Command History & Shortcuts",
					Content: `The "Readline" library powers the input of most Linux shells, providing deep history and cursor manipulation features that make you feel like you have superpowers.

**1. The Magic of History Expansion:**
- **!!**: Repeat the entire last command. (Useful with sudo: ` + "`" + `sudo !!` + "`" + `).
- **!$**: Represents the last argument of the previous command. (e.g., if you ran ` + "`" + `mkdir my_long_dir_name` + "`" + `, follow up with ` + "`" + `cd !$` + "`" + `).
- **!n**: Run the n-th command from your history list.
- **Ctrl+R**: The recursive search. Type a part of an old command to find it instantly.

**2. Lightning-Fast Cursor Movement:**
- **Ctrl+A / Ctrl+E**: Jump to the beginning or end of the line.
- **Alt+F / Alt+B**: Move forward or backward by one whole word.
- **Ctrl+XX**: Toggle between the start of the line and your current cursor position.

**3. Deletion and Yanking:**
- **Ctrl+U**: "Cut" everything from the cursor back to the start.
- **Ctrl+K**: "Cut" everything from the cursor to the end.
- **Ctrl+Y**: "Paste" (yank) what you just cut with the above commands.
- **Ctrl+W**: Delete the word immediately before the cursor.

**4. The Essential Tab:**
Tab completion is not just for laziness; it prevents typos.
- **Single Tab**: Completes the name if unique.
- **Double Tab**: Lists all possible completions.`,
					CodeExamples: `# REPEAT AS ROOT
# You forgot sudo? No problem:
$ apt update
E: Could not open lock file...
$ sudo !!

# REUSING ARGUMENTS
$ touch very_long_filename_with_details.txt
$ mv !$ backup/

# SEARCHING HISTORY
# Press Ctrl+R, then type "nginx"
(reverse-i-search)'ngi': systemctl restart nginx

# EDITING LONG PATHS
# Cursor is at the end:
$ ls -l /var/www/html/site/assets/img/logo.png
# Press Alt+B several times to jump back word by word.`,
				},

				{
					Title: "Command Aliases & Customization",
					Content: `The true power of Linux is discovered when you stop using it "out of the box" and start tailoring it to your workflow.

**1. What are Aliases?**
An alias is essentially a nickname for a long command.
- **Safety**: ` + "`" + `alias rm='rm -i'` + "`" + ` (prompts before deleting).
- **Correctness**: ` + "`" + `alias grep='grep --color=auto'` + "`" + ` (makes finding things easier).
- **Speed**: ` + "`" + `alias ..='cd ..'` + "`" + `.

**2. Persistent Configuration (.bashrc vs .bash_profile):**
To make your aliases work every time you open a terminal, you must put them in a configuration file:
- **.bashrc**: Executed for every new **interactive non-login** shell (like when you open a terminal window).
- **.bash_profile**: Executed once for **login** shells (like when you first SSH into a server).
- **Best Practice**: Put your aliases in ` + "`" + `.bashrc` + "`" + ` or a separate ` + "`" + `.bash_aliases` + "`" + ` file that is called by ` + "`" + `.bashrc` + "`" + `.

**3. Environment Variables:**
Variables like ` + "`" + `$PATH` + "`" + ` or ` + "`" + `$EDITOR` + "`" + ` global settings that applications look at.
- **$PATH**: The list of directories the shell searches for commands. If a command isn't in your PATH, you have to type the full path (e.g., ` + "`" + `/home/user/my_app` + "`" + `).
- **$EDITOR**: Tells Git or other tools which text editor you prefer (e.g., ` + "`" + `export EDITOR=vim` + "`" + `).

**4. Shell Functions (Pro Level):**
While aliases are great for simple shortcuts, **functions** allow you to use logic and parameters. For example, a function can create a directory and immediately move into it—something a simple alias cannot do easily.`,
					CodeExamples: `# VIEW ALL CURRENT ALIASES
alias

# USEFUL PRODUCTION ALIASES
alias myip="curl -s https://ifconfig.me"
alias ports="sudo netstat -tulanp"
alias psg="ps aux | grep -v grep | grep -i"

# PERSISTENCE (Adding to .bashrc)
echo "alias ll='ls -lah'" >> ~/.bashrc
source ~/.bashrc  # Apply changes immediately

# THE "MKCD" FUNCTION
# Add this to your ~/.bashrc:
function mkcd() {
    mkdir -p "$1" && cd "$1"
}

# THE POWER OF $PATH
# Temporary addition:
export PATH=$PATH:/home/user/my_custom_tools
# Now you can run tools in that folder just by name!`,
				},

			},
			ProblemIDs: []int{},
		},
		{
			ID:          112,
			Title:       "File System & Permissions",
			Description: "Understand Linux file system structure, file types, permissions, ownership, and links.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Linux File System Structure",
					Content: `The Linux directory structure is not random; it follows the **Filesystem Hierarchy Standard (FHS)**. Unlike Windows, which uses drive letters (C:, D:), Linux uses a single unified tree starting from the **Root (/)**.

**Key Directories and Their Roles:**
- **/bin & /sbin**: Essential binaries. ` + "`" + `/bin` + "`" + ` is for all users, while ` + "`" + `/sbin` + "`" + ` is for system administration (commands like ` + "`" + `iptables` + "`" + ` or ` + "`" + `reboot` + "`" + `).
- **/etc**: The "Control Center." Almost all system-wide configuration files (like ` + "`" + `/etc/hosts` + "`" + ` or ` + "`" + `/etc/passwd` + "`" + `) live here.
- **/home**: User-specific data. Every user has a subfolder here (e.g., ` + "`" + `/home/alice` + "`" + `) for their private files.
- **/root**: The home directory for the root user. Note that it is *not* inside ` + "`" + `/home` + "`" + `.
- **/var**: Variable data. This is where logs (` + "`" + `/var/log` + "`" + `), databases, and mail spools are stored. If your disk is full, check here first!
- **/proc & /sys**: These are **Pseudo Filesystems**. They don't actually exist on the disk; they are interfaces to the Linux kernel. Reading a file in ` + "`" + `/proc` + "`" + ` (like ` + "`" + `/proc/meminfo` + "`" + `) gives you real-time data from the kernel.
- **/tmp**: Temporary files. On many modern systems, this is stored in RAM (tmpfs) and is wiped on reboot.

**Everything is a File:**
In Linux, hardware devices are represented as files in ** /dev**.
- ` + "`" + `/dev/sda` + "`" + `: Your hard drive.
- ` + "`" + `/dev/null` + "`" + `: The "Black Hole." Anything you send here disappears.
- ` + "`" + `/dev/random` + "`" + `: A source of high-quality random numbers.`,
					CodeExamples: `# WHERE IS A COMMAND LOCATED?
which ls
# Output: /usr/bin/ls

# CHECK DISK USAGE BY DIRECTORY
df -h
du -sh /var/log

# PEEKING INTO THE KERNEL
cat /proc/cpuinfo
cat /proc/uptime

# THE BLACK HOLE
# Discard error messages from a command:
ls /nonexistent_folder 2> /dev/null

# LISTING WITH TYPE INDICATORS
ls -F /
# / = directory
# @ = symlink
# * = executable`,
				},

				{
					Title: "File Permissions",
					Content: `Linux is a multi-user system, so security is managed through a granular permission system. Every file and directory has an owner and belongs to a specific group.

**1. Basic Permission Bits:**
- **Read (r / 4)**: For files, you can read content. For directories, you can list filenames (but not access their metadata without 'x').
- **Write (w / 2)**: For files, you can modify or truncate them. For directories, you can create, rename, or delete files *within* them.
- **Execute (x / 1)**: For files, you can run them as programs. For directories, you can "enter" them (cd) and access files inside.

**2. Special Permissions (The 4th Digit):**
Sometimes you'll see permissions like ` + "`" + `s` + "`" + ` or ` + "`" + `t` + "`" + ` instead of ` + "`" + `x` + "`" + `.
- **SUID (Set User ID / 4000)**: When a file is executed, it runs with the permissions of the *owner* (often root). Example: ` + "`" + `/usr/bin/passwd` + "`" + `.
- **SGID (Set Group ID / 2000)**: Files run with the permissions of the *group*. On directories, new files created inside inherit the parent's group.
- **Sticky Bit (1000)**: Only the owner of a file (or root) can delete it, even if others have write access to the directory. Common on ` + "`" + `/tmp` + "`" + `.

**3. Umask (The Default Permission):**
When you create a new file, it doesn't have 777. The ` + "`" + `umask` + "`" + ` value "masks" or subtracts permissions from the default (usually 666 for files, 777 for dirs).`,
					CodeExamples: `# VIEWING PERMISSIONS
ls -l /etc/shadow
# Output: -rw-r----- 1 root shadow ... 
# ONLY root can write, others cannot even read.

# ADDING THE STICKY BIT
chmod +t my_shared_folder/
# Now others can't delete each other's files.

# THE SUID BIT EXAMPLE
ls -l /usr/bin/passwd
# Output: -rwsr-xr-x ...
# Note the 's' instead of 'x'.

# CHANGING PERMISSIONS (Octal)
# 755 = User(rwx), Group(rx), Others(rx)
chmod 755 script.sh

# THE UMASK VALUE
umask
# If output is 0022, new files get 644 (666-022) and dirs get 755 (777-022).`,
				},

				{
					Title: "Ownership & chown",
					Content: `Ownership is the bedrock of Linux security. Every file is owned by exactly one **User** and one **Group**.

**1. The Root User (The Superuser):**
Root is the "God mode" of Linux. It has UID 0 and can bypass all permission checks.
- **DANGER**: Running as root all the time is risky. One typo in ` + "`" + `rm` + "`" + ` can destroy the entire system.
- **Sudo (SuperUser DO)**: The standard way to run a single command with root privileges without logging in as root.

**2. Changing Owners (chown):**
Only root (or a user with sudo) can change the owner of a file.
- **Syntax**: ` + "`" + `chown user:group filename` + "`" + `.
- **Cleanup**: If you copy files from a USB drive, they often end up with the wrong owner. Use ` + "`" + `chown` + "`" + ` to fix them.

**3. Changing Groups (chgrp):**
A simpler version of chown that only affects the group.

**4. Identity Commands:**
- **id**: Shows your UID (User ID) and GID (Group ID).
- **whoami**: A quick way to see which user the current shell is running as.`,
					CodeExamples: `# WHO AM I?
id
# Output: uid=1000(alice) gid=1000(alice) groups=1000(alice),27(sudo)

# TAKING OWNERSHIP OF A WEB FOLDER
sudo chown -R www-data:www-data /var/www/html

# GIVING YOURSELF PERMISSION FOR A FILE
sudo chown $USER:$USER my_config.conf

# RUNNING A COMMAND AS ROOT
sudo apt update

# SWITCHING TO THE ROOT USER (Use with caution!)
sudo -i`,
				},

				{
					Title: "Links (Hard & Soft)",
					Content: `In Linux, a "link" is essentially a pointer to data on your disk. Understanding the two types of links is key to understanding how the filesystem manages data.

**1. Symbolic Links (Soft Links / Symlinks):**
These are like "Shortcuts" in Windows or "Aliases" in macOS.
- **How they work**: A symlink is a special file that contains the *path* to another file.
- **Deleting**: If you delete the original file, the symlink becomes **dangling** (broken).
- **Flexibility**: They can link to directories and can point to files on different disks or partitions.

**2. Hard Links:**
These are much more foundational and tied to the **Inode** system.
- **How they work**: A hard link is just another name for the data on the disk. They share the same Inode number.
- **Deleting**: If you delete the "original" file, the data is *not* deleted as long as at least one hard link still exists. The data is only freed when the link count reaches zero.
- **Restrictions**: They cannot link to directories and cannot cross different disk partitions.

**3. What is an Inode?**
An Inode (Index Node) is a data structure that stores everything about a file *except* its name and its actual data. It stores permissions, owner, size, and timestamps.
- Each file has a unique Inode number.
- You can see this number with ` + "`" + `ls -i` + "`" + `.`,
					CodeExamples: `# CREATING A SYMLINK (Most common)
ln -s /etc/nginx/nginx.conf my_config.conf

# CREATING A HARD LINK
ln target_file.txt link_name.txt

# SEEING INODE NUMBERS
ls -li
# Output: 123456 -rw-r--r-- 2 alice ...
# The '2' indicates there are two hard links for this inode.

# FINDING THE TARGET OF A SYMLINK
readlink my_config.conf
# or simply
ls -l my_config.conf

# DANGER: BROKEN SYMLINKS
# If you move the original file, symlinks pointing to its old name will break!`,
				},

				{
					Title: "Special File Types & Inodes",
					Content: `Understanding special file types and inodes is crucial for advanced Linux administration.

**Inodes (Index Nodes):**
- Data structure storing file metadata
- Each file/directory has unique inode number
- Contains: permissions, owner, size, timestamps, location on disk
- Limited number per filesystem (inode limit)
- View with ls -i or stat command

**Inode Information:**
- **Inode Number**: Unique identifier
- **File Type**: Regular, directory, link, device, etc.
- **Permissions**: Read, write, execute
- **Owner & Group**: User and group IDs
- **Size**: File size in bytes
- **Timestamps**: Access, modify, change times
- **Links**: Number of hard links
- **Blocks**: Disk blocks used

**Special File Types:**

**Device Files:**
- Represent hardware devices
- **Block devices** (/dev/sda): Storage devices (b in ls -l)
- **Character devices** (/dev/tty): Serial devices (c in ls -l)
- Created with mknod (usually automatic)

**Sockets:**
- Inter-process communication
- Unix domain sockets (local)
- Network sockets (TCP/UDP)
- Created by applications
- Shown as 's' in ls -l

**Named Pipes (FIFOs):**
- First In, First Out
- One-way communication
- Created with mkfifo
- Shown as 'p' in ls -l

**stat Command:**
- Shows detailed inode information
- More comprehensive than ls -l
- Includes all metadata`,
					CodeExamples: `# View inode numbers
ls -i file.txt
# Output: 1234567 file.txt

# Detailed inode information
stat file.txt
# Output:
#   File: file.txt
#   Size: 1024        Blocks: 8          IO Block: 4096   regular file
# Device: 803h/2051d  Inode: 1234567     Links: 1
# Access: (0644/-rw-r--r--)  Uid: ( 1000/   user)   Gid: ( 1000/   user)
# Access: 2024-01-15 10:30:00.000000000 +0000
# Modify: 2024-01-15 10:25:00.000000000 +0000
# Change: 2024-01-15 10:25:00.000000000 +0000

# View inode for directory
stat /home
# Shows directory inode info

# Check inode usage
df -i
# Shows inode usage per filesystem

# Find files by inode number
find / -inum 1234567 2>/dev/null

# View device files
ls -l /dev/sda*
# Output shows 'b' for block device

ls -l /dev/tty*
# Output shows 'c' for character device

# Create named pipe
mkfifo mypipe
ls -l mypipe
# Output shows 'p' for pipe

# Use pipe
echo "Hello" > mypipe &
cat < mypipe
# Output: Hello

# View socket files
ls -l /var/run/*.sock
# Shows 's' for socket

# Check file type
file /dev/sda1
# Output: /dev/sda1: block special

file /dev/tty1
# Output: /dev/tty1: character special

# Inode limits
tune2fs -l /dev/sda1 | grep -i inode
# Shows inode count and usage

# Find files with same inode (hard links)
find / -inum $(stat -c %i file.txt) 2>/dev/null`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          113,
			Title:       "Text Processing & Filters",
			Description: "Master text processing tools: cat, grep, sed, awk, sort, and other filters for manipulating text data.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Viewing Files",
					Content: `Linux provides many tools to view file contents. Each has its use case.

**cat (concatenate):**
- Display entire file
- Simple, fast
- Good for small files
- cat file.txt
- cat file1 file2: Concatenate multiple files

**less/more:**
- View file page by page
- **less**: More features (can scroll backward)
- **more**: Simpler, older
- Navigate with arrow keys, space, 'q' to quit
- Better for large files

**head:**
- Show first N lines (default: 10)
- head -n 20 file.txt
- head -20 file.txt

**tail:**
- Show last N lines (default: 10)
- tail -n 20 file.txt
- tail -f file.txt: Follow file (watch for new lines) - great for logs!

**wc (word count):**
- Count lines, words, characters
- wc file.txt: Shows lines, words, bytes
- wc -l: Lines only
- wc -w: Words only
- wc -c: Bytes only`,
					CodeExamples: `# View entire file
cat file.txt

# View multiple files
cat file1.txt file2.txt

# View with line numbers
cat -n file.txt

# View file page by page
less file.txt
# Navigate: Space=next page, b=previous, q=quit, /search=find

more file.txt
# Navigate: Space=next page, q=quit

# View first 10 lines
head file.txt

# View first 20 lines
head -n 20 file.txt
head -20 file.txt

# View last 10 lines
tail file.txt

# View last 20 lines
tail -n 20 file.txt
tail -20 file.txt

# Follow log file (watch updates)
tail -f /var/log/syslog
# Press Ctrl+C to stop

# Count lines, words, bytes
wc file.txt
# Output: 100 500 5000 file.txt
#         lines words bytes

# Count lines only
wc -l file.txt

# Count words only
wc -w file.txt

# Count bytes only
wc -c file.txt

# Count characters
wc -m file.txt`,
				},
				{
					Title: "grep - Pattern Searching",
					Content: `grep is one of the most powerful text search tools. It finds lines matching a pattern.

**Basic Usage:**
- grep pattern file
- Searches for pattern in file
- Prints matching lines

**Common Options:**
- -i: Case-insensitive search
- -v: Invert match (show non-matching lines)
- -r: Recursive (search in directories)
- -n: Show line numbers
- -c: Count matches
- -l: Show only filenames with matches
- -E: Extended regex (or use egrep)
- -w: Match whole words only
- --color: Highlight matches

**Pattern Types:**
- **Literal**: grep "error" file.txt
- **Regex**: grep "^Error" file.txt (lines starting with Error)
- **Character classes**: grep "[0-9]" file.txt (lines with digits)

**Common Patterns:**
- ^: Start of line
- $: End of line
- .: Any character
- *: Zero or more of previous
- +: One or more of previous
- []: Character class
- |: OR (with -E)`,
					CodeExamples: `# Basic search
grep "error" logfile.txt

# Case-insensitive
grep -i "error" logfile.txt

# Show line numbers
grep -n "error" logfile.txt

# Count matches
grep -c "error" logfile.txt

# Show only filenames
grep -l "error" *.txt

# Invert match (show lines without pattern)
grep -v "error" logfile.txt

# Recursive search
grep -r "error" /var/log

# Whole word match
grep -w "error" logfile.txt
# Matches "error" but not "errors" or "terror"

# Extended regex
grep -E "error|warning" logfile.txt
# Matches lines with "error" OR "warning"

# Multiple patterns
grep -e "error" -e "warning" logfile.txt

# Highlight matches
grep --color "error" logfile.txt

# Search in multiple files
grep "pattern" file1.txt file2.txt

# Common patterns
grep "^Error" file.txt        # Lines starting with Error
grep "error$" file.txt       # Lines ending with error
grep "[0-9]" file.txt        # Lines with digits
grep "^$" file.txt           # Empty lines
grep "error.*warning" file.txt # error followed by warning

# Combine with pipes
cat file.txt | grep "error" | grep -v "debug"`,
				},
				{
					Title: "sed - Stream Editor",
					Content: `sed is a stream editor for filtering and transforming text. It's powerful for batch editing.

**Basic Usage:**
- sed 's/old/new/' file: Substitute first occurrence per line
- sed 's/old/new/g' file: Substitute all occurrences (global)
- sed -i: Edit file in-place (modify file)

**Common Operations:**
- **Substitute**: s/pattern/replacement/
- **Delete**: d (delete lines)
- **Print**: p (print lines)
- **Insert**: i (insert before line)
- **Append**: a (append after line)

**Addresses:**
- Line numbers: sed '5d' file (delete line 5)
- Ranges: sed '1,5d' file (delete lines 1-5)
- Patterns: sed '/pattern/d' file (delete matching lines)
- Last line: $

**Common Options:**
- -i: In-place editing
- -n: Suppress default output
- -e: Multiple expressions`,
					CodeExamples: `# Substitute first occurrence per line
sed 's/old/new/' file.txt

# Substitute all occurrences
sed 's/old/new/g' file.txt

# Case-insensitive substitution
sed 's/old/new/gi' file.txt

# Edit file in-place (modify file)
sed -i 's/old/new/g' file.txt

# Delete line 5
sed '5d' file.txt

# Delete lines 1-5
sed '1,5d' file.txt

# Delete lines matching pattern
sed '/error/d' file.txt

# Delete empty lines
sed '/^$/d' file.txt

# Print specific lines
sed -n '5p' file.txt        # Print line 5
sed -n '1,5p' file.txt      # Print lines 1-5

# Print lines matching pattern
sed -n '/error/p' file.txt

# Substitute and save to new file
sed 's/old/new/g' file.txt > newfile.txt

# Multiple substitutions
sed -e 's/old1/new1/g' -e 's/old2/new2/g' file.txt

# Common transformations
sed 's/^/prefix /' file.txt     # Add prefix to each line
sed 's/$/ suffix/' file.txt     # Add suffix to each line
sed 's/[0-9]//g' file.txt       # Remove all digits
sed 's/  */ /g' file.txt        # Replace multiple spaces with single space`,
				},
				{
					Title: "awk - Pattern Processing",
					Content: `awk is a powerful programming language for text processing. It's excellent for column-based data.

**Basic Usage:**
- awk 'pattern {action}' file
- Processes file line by line
- Splits lines into fields (columns)
- Fields accessed as $1, $2, $3, etc.
- $0 = entire line

**Built-in Variables:**
- NR: Current record (line) number
- NF: Number of fields in current line
- FS: Field separator (default: space)
- OFS: Output field separator

**Common Patterns:**
- /pattern/: Lines matching pattern
- NR==1: First line
- $1=="value": First field equals value
- $NF: Last field

**Common Actions:**
- print: Print line or fields
- print $1: Print first field
- print $NF: Print last field`,
					CodeExamples: `# Print entire file
awk '{print}' file.txt

# Print first field (column)
awk '{print $1}' file.txt

# Print first and second fields
awk '{print $1, $2}' file.txt

# Print last field
awk '{print $NF}' file.txt

# Print lines matching pattern
awk '/error/ {print}' file.txt

# Print specific lines
awk 'NR==1 {print}' file.txt        # First line
awk 'NR==5 {print}' file.txt        # Line 5
awk 'NR>=5 && NR<=10 {print}' file.txt  # Lines 5-10

# Print with line numbers
awk '{print NR, $0}' file.txt

# Print number of fields per line
awk '{print NF, $0}' file.txt

# Custom field separator
awk -F: '{print $1}' /etc/passwd
# Uses ':' as separator (like /etc/passwd)

# Print lines where field matches
awk '$1=="error" {print}' file.txt
awk '$3>100 {print}' file.txt

# Calculations
awk '{sum+=$1} END {print sum}' file.txt
# Sum first column

# Count lines
awk 'END {print NR}' file.txt

# Common patterns
awk '{print $1, $NF}' file.txt           # First and last field
awk 'NF>5 {print}' file.txt              # Lines with more than 5 fields
awk '$1~/pattern/ {print}' file.txt      # First field matches pattern`,
				},
				{
					Title: "Other Useful Filters",
					Content: `Linux provides many other text processing tools for specific tasks.

**sort:**
- Sort lines
- sort file.txt: Alphabetical sort
- sort -n: Numeric sort
- sort -r: Reverse sort
- sort -u: Unique lines only
- sort -k2: Sort by second field

**uniq:**
- Remove duplicate consecutive lines
- uniq file.txt
- uniq -c: Count occurrences
- uniq -d: Show only duplicates
- Usually used with sort: sort file.txt | uniq

**cut:**
- Extract columns/fields
- cut -d: -f1 file.txt: Field 1, delimiter ':'
- cut -c1-5 file.txt: Characters 1-5
- cut -f1,3 file.txt: Fields 1 and 3

**paste:**
- Merge lines from files
- paste file1.txt file2.txt
- paste -d: file1 file2: Custom delimiter

**tr (translate):**
- Translate or delete characters
- tr 'a-z' 'A-Z': Convert to uppercase
- tr -d ' ': Delete spaces
- tr -s ' ': Squeeze multiple spaces`,
					CodeExamples: `# Sort alphabetically
sort file.txt

# Sort numerically
sort -n numbers.txt

# Reverse sort
sort -r file.txt

# Sort by second field
sort -k2 file.txt

# Remove duplicates (requires sorted input)
sort file.txt | uniq

# Count occurrences
sort file.txt | uniq -c

# Show only duplicates
sort file.txt | uniq -d

# Extract first field (tab-delimited)
cut -f1 file.txt

# Extract first field (colon-delimited)
cut -d: -f1 /etc/passwd

# Extract characters 1-10
cut -c1-10 file.txt

# Extract multiple fields
cut -d: -f1,3,5 /etc/passwd

# Merge files side by side
paste file1.txt file2.txt

# Merge with delimiter
paste -d: file1.txt file2.txt

# Convert to uppercase
tr 'a-z' 'A-Z' < file.txt

# Convert to lowercase
tr 'A-Z' 'a-z' < file.txt

# Delete characters
tr -d ' ' < file.txt

# Squeeze multiple spaces
tr -s ' ' < file.txt

# Replace characters
tr ' ' '_' < file.txt

# Combine tools
cat file.txt | grep "error" | sort | uniq -c
# Find errors, sort, count unique`,
				},
				{
					Title: "Combining Text Processing Tools",
					Content: `The real power of Linux text tools comes from combining them with pipes. Mastering command chaining is essential.

**Pipes (|):**
- Connects output of one command to input of another
- Left-to-right processing
- Enables complex data transformations
- Each command processes stream sequentially

**Command Chaining:**
- **Pipe (|)**: Pass output to next command
- **Semicolon (;)**: Run commands sequentially (independent)
- **Ampersand (&)**: Run command in background
- **&&**: Run next command only if previous succeeds
- **||**: Run next command only if previous fails

**Common Patterns:**

**Filter and Process:**
- command | grep | sort | uniq
- Filter, then organize results

**Extract and Transform:**
- command | cut | sed | awk
- Extract fields, transform, process

**Count and Summarize:**
- command | grep | wc
- Filter then count

**Multi-step Processing:**
- Chain multiple filters
- Each step refines the data
- Final output is processed result

**Best Practices:**
- Start with data source
- Apply filters progressively
- Use appropriate tools for each step
- Test each step independently
- Consider performance (some tools are faster)`,
					CodeExamples: `# Basic pipe
ls -l | grep "\.txt$"
# List files, filter .txt files

# Multiple pipes
ps aux | grep python | awk '{print $2}' | xargs kill
# Find Python processes, extract PIDs, kill them

# Filter and count
grep "error" /var/log/syslog | wc -l
# Count error lines

# Extract and process
cat /etc/passwd | cut -d: -f1,7 | grep "/bin/bash"
# Extract username and shell, filter bash users

# Complex processing
netstat -tuln | grep LISTEN | awk '{print $4}' | cut -d: -f2 | sort -n | uniq
# List listening ports, extract port numbers, sort, remove duplicates

# Process log file
tail -f /var/log/apache2/access.log | grep "404" | awk '{print $1}' | sort | uniq -c | sort -rn
# Monitor log, find 404 errors, extract IPs, count unique IPs, sort by count

# Find large files
find /home -type f -size +100M | xargs ls -lh | awk '{print $5, $9}' | sort -h
# Find large files, list with sizes, extract size and name, sort by size

# Process CSV data
cat data.csv | cut -d, -f2,3 | sort | uniq | wc -l
# Extract columns 2 and 3, sort, find unique, count

# Multi-step text processing
cat file.txt | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | tr -s ' ' | sort | uniq -c | sort -rn | head -10
# Convert to lowercase, remove punctuation, squeeze spaces, sort, count, show top 10

# Process system information
ps aux | awk '{if($3>50) print $2, $3, $11}' | sort -k2 -rn | head -5
# Find processes using >50% CPU, show PID/CPU/command, sort by CPU, show top 5

# Command chaining with &&
mkdir newdir && cd newdir && touch file.txt
# Create directory, change to it, create file (only if each step succeeds)

# Command chaining with ||
command || echo "Command failed"
# Run command, or print error message

# Complex pipeline
journalctl -u nginx --since "1 hour ago" | grep -i error | awk '{print $1, $2, $5}' | sort | uniq -c
# Get nginx logs from last hour, find errors, extract time and message, count occurrences`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          114,
			Title:       "Process Management Basics",
			Description: "Learn to view, control, and manage processes: ps, top, kill, jobs, and process states.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Understanding Processes",
					Content: `A process is a running instance of a program. Understanding processes is essential for system administration.

**What is a Process?**
- Running program with its own memory space
- Has Process ID (PID)
- Has parent process (PPID)
- Has user owner
- Consumes resources (CPU, memory)

**Process States:**
- **Running (R)**: Currently executing
- **Sleeping (S)**: Waiting for event
- **Stopped (T)**: Suspended (Ctrl+Z)
- **Zombie (Z)**: Terminated but not cleaned up
- **Dead (X)**: Terminated

**Process Types:**
- **Foreground**: Interactive, blocks terminal
- **Background**: Runs without blocking terminal
- **Daemon**: Background service, detached from terminal

**Process Hierarchy:**
- All processes descend from init (PID 1) or systemd
- Parent-child relationships
- Process tree structure`,
					CodeExamples: `# View process tree
pstree
# Shows hierarchical process tree

# View with PIDs
pstree -p

# View process hierarchy
ps auxf
# 'f' shows forest (tree) view

# Find process ID
pgrep firefox
# Output: 12345

# Check if process is running
pidof firefox
# Output: 12345

# View process information
ps -p 12345

# Process states in ps output
ps aux
# STAT column shows state:
# R = Running
# S = Sleeping
# T = Stopped
# Z = Zombie
# D = Uninterruptible sleep`,
				},
				{
					Title: "ps - Process Status",
					Content: `ps displays information about running processes. It's one of the most important process management tools.

**Common ps Commands:**
- ps: Show current terminal's processes
- ps aux: Show all processes (BSD style)
- ps -ef: Show all processes (Unix style)
- ps -u username: Show user's processes
- ps -p PID: Show specific process

**ps aux Output Columns:**
- **USER**: Process owner
- **PID**: Process ID
- **%CPU**: CPU usage percentage
- **%MEM**: Memory usage percentage
- **VSZ**: Virtual memory size
- **RSS**: Resident set size (physical memory)
- **TTY**: Terminal
- **STAT**: Process state
- **START**: Start time
- **TIME**: CPU time used
- **COMMAND**: Command line

**Useful Options:**
- -f: Full format (more details)
- -l: Long format
- -e: All processes
- -u: User-specific
- --sort: Sort output`,
					CodeExamples: `# Show current terminal processes
ps

# Show all processes
ps aux

# Show all processes (alternative)
ps -ef

# Show processes for specific user
ps -u username
ps aux | grep username

# Show specific process
ps -p 12345

# Show process tree
ps auxf

# Show full command line
ps auxww
# 'ww' shows unlimited width

# Sort by CPU usage
ps aux --sort=-%cpu | head

# Sort by memory usage
ps aux --sort=-%mem | head

# Show only running processes
ps aux | grep -v "sleep"

# Count processes
ps aux | wc -l

# Find process by name
ps aux | grep firefox

# Show parent-child relationships
ps -ef | grep -E "PID|firefox"

# Custom output format
ps -eo pid,user,comm,etime
# Shows: PID, user, command, elapsed time`,
				},
				{
					Title: "top & htop",
					Content: `top and htop provide real-time, interactive views of running processes. They're essential for monitoring system performance.

**top:**
- Built-in process monitor
- Updates continuously
- Shows CPU, memory usage
- Interactive commands
- Available on all Linux systems

**htop:**
- Enhanced version of top
- Better interface (colors, bars)
- Easier navigation
- May need installation: apt install htop

**top Interactive Commands:**
- **q**: Quit
- **k**: Kill process (enter PID)
- **r**: Renice process (change priority)
- **Space**: Refresh immediately
- **M**: Sort by memory
- **P**: Sort by CPU
- **T**: Sort by time
- **u**: Filter by user
- **/**: Search

**Key Information Displayed:**
- System uptime
- Load average
- CPU usage
- Memory usage
- Process list with details`,
					CodeExamples: `# Start top
top

# Start htop (if installed)
htop

# Run top in batch mode (non-interactive)
top -b -n 1
# Useful for scripting

# Run top for specific user
top -u username

# Update interval (seconds)
top -d 5
# Updates every 5 seconds

# Show specific number of processes
top -n 20

# Save top output to file
top -b -n 1 > top_output.txt

# While in top:
# Press 'M' to sort by memory
# Press 'P' to sort by CPU
# Press 'k' then enter PID to kill
# Press 'r' then enter PID to renice
# Press 'u' then enter username to filter
# Press 'q' to quit

# Alternative: Use htop (more user-friendly)
htop
# Navigate with arrow keys
# F9 to kill process
# F10 to quit`,
				},
				{
					Title: "kill & Process Control",
					Content: `Controlling processes is essential for system management. Learn to stop, restart, and manage processes.

**kill:**
- Send signals to processes
- kill PID: Send TERM signal (graceful termination)
- kill -9 PID: Send KILL signal (force kill)
- kill -15 PID: Send TERM signal (default)

**Common Signals:**
- **SIGTERM (15)**: Terminate gracefully (default)
- **SIGKILL (9)**: Force kill (cannot be ignored)
- **SIGHUP (1)**: Hang up (often used to reload config)
- **SIGINT (2)**: Interrupt (Ctrl+C)
- **SIGSTOP (19)**: Stop process (Ctrl+Z)
- **SIGCONT (18)**: Continue stopped process

**killall:**
- Kill processes by name
- killall processname
- killall -9 processname: Force kill

**pkill:**
- Kill processes by pattern
- pkill firefox
- pkill -9 -u username processname

**jobs, fg, bg:**
- Manage background jobs
- jobs: List background jobs
- fg %1: Bring job 1 to foreground
- bg %1: Run job 1 in background
- &: Run command in background`,
					CodeExamples: `# Kill process gracefully
kill 12345

# Force kill process
kill -9 12345
kill -KILL 12345

# Send HUP signal (reload config)
kill -HUP 12345
kill -1 12345

# Kill by name
killall firefox

# Force kill by name
killall -9 firefox

# Kill by pattern
pkill firefox

# Kill user's processes
pkill -u username firefox

# List all signals
kill -l

# Run command in background
sleep 100 &
# Output: [1] 12345
#         [1] = job number, 12345 = PID

# List background jobs
jobs
# Output: [1]+ Running sleep 100 &

# Bring job to foreground
fg %1
# or
fg 1

# Send job to background
bg %1

# Stop process (Ctrl+Z)
# Process goes to stopped state

# Continue stopped process
fg
# or
bg  # if you want it in background

# Kill background job
kill %1

# No hangup (continue after logout)
nohup command &
# Output goes to nohup.out

# Disown job (remove from shell's job table)
disown %1`,
				},
				{
					Title: "Process Priorities & Scheduling",
					Content: `Linux uses priorities to determine which processes get CPU time. Understanding priorities helps manage system performance.

**Nice Values:**
- Range: -20 (highest priority) to +19 (lowest priority)
- Default: 0
- Lower nice = higher priority = more CPU time
- Only root can set negative nice values

**Priority Levels:**
- **Real-time**: Highest priority (SCHED_FIFO, SCHED_RR)
- **High priority**: Negative nice values (-20 to -1)
- **Normal**: Nice 0 (default)
- **Low priority**: Positive nice values (+1 to +19)

**nice Command:**
- Run command with specified nice value
- nice -n 10 command: Run with nice +10
- nice -n -10 command: Run with nice -10 (requires root)

**renice Command:**
- Change nice value of running process
- renice +10 PID: Increase nice (lower priority)
- renice -10 PID: Decrease nice (higher priority, requires root)

**chrt Command:**
- Set real-time scheduling policy
- chrt -f 50 command: FIFO scheduling, priority 50
- chrt -r 50 command: Round-robin scheduling

**Scheduling Policies:**
- **SCHED_NORMAL (CFS)**: Default, fair scheduling
- **SCHED_FIFO**: First In First Out (real-time)
- **SCHED_RR**: Round Robin (real-time)
- **SCHED_BATCH**: Batch processing (lower priority)
- **SCHED_IDLE**: Idle priority (lowest)

**When to Adjust Priorities:**
- **Increase priority**: Important processes, interactive apps
- **Decrease priority**: Background tasks, batch jobs
- **Real-time**: Time-critical applications (audio, video)`,
					CodeExamples: `# Check current nice value
nice
# Output: 0

# Run command with low priority
nice -n 10 long_running_task.sh
# Nice value: +10 (lower priority)

# Run command with high priority (requires root)
sudo nice -n -10 important_task.sh
# Nice value: -10 (higher priority)

# Change priority of running process
renice +10 12345
# Increase nice value (lower priority)

# Change priority to high (requires root)
sudo renice -10 12345
# Decrease nice value (higher priority)

# Change priority for all processes of a user
sudo renice +5 -u username

# Change priority for all processes of a group
sudo renice +5 -g groupname

# View process priorities
ps -eo pid,ni,comm | head
# Shows PID, nice value, command

# Sort by priority
ps -eo pid,ni,comm --sort=ni | head
# Processes sorted by nice value

# Set real-time scheduling (FIFO)
sudo chrt -f 50 important_process
# Priority 50, FIFO scheduling

# Set real-time scheduling (Round Robin)
sudo chrt -r 50 important_process
# Priority 50, Round Robin scheduling

# Check scheduling policy
chrt -p 12345
# Shows current scheduling policy and priority

# Set batch scheduling
chrt -b 0 batch_job.sh
# Batch scheduling (lower priority)

# Set idle scheduling
chrt -i 0 background_task.sh
# Idle scheduling (lowest priority)

# taskset - CPU affinity
taskset -c 0,1 process_name
# Run process only on CPUs 0 and 1

# Check CPU affinity
taskset -p 12345
# Shows which CPUs process can run on

# Common use cases
nice -n 19 backup.sh          # Low priority backup
nice -n -5 compile.sh         # High priority compilation (root)
renice +15 $(pgrep backup)    # Lower priority for all backup processes`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
