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
					Content: `Linux is a free and open-source Unix-like operating system kernel originally created by Linus Torvalds in 1991. What started as a hobby project has grown into the most widely used operating system in the world, powering everything from smartphones and embedded devices to the world's most powerful supercomputers and cloud infrastructure.

**Historical Context:**

**The Birth of Linux:**
In 1991, Linus Torvalds, a Finnish computer science student, announced he was working on a free operating system kernel as a hobby. This kernel, combined with GNU software and other open-source components, became what we know today as Linux.

**Key Milestones:**
- **1991**: Linus Torvalds releases Linux kernel version 0.01
- **1992**: Linux kernel licensed under GPL (GNU General Public License)
- **1993**: First Linux distributions emerge (Debian, Slackware)
- **1994**: Linux 1.0 released, Red Hat founded
- **1996**: Linux gains enterprise attention
- **2000s**: Linux becomes dominant on servers
- **2010s**: Linux powers cloud computing, Android, IoT
- **2020s**: Linux is everywhere - cloud, edge, embedded, supercomputers

**What Makes Linux Special:**

**1. Open Source:**
- **Free**: No licensing costs
- **Transparent**: Source code is visible and auditable
- **Modifiable**: Can customize for specific needs
- **Community-Driven**: Developed by thousands of contributors worldwide
- **Security**: Open source allows security audits and fixes

**2. Unix Philosophy:**
Linux follows the Unix philosophy, which emphasizes:
- **Simplicity**: Do one thing and do it well
- **Composability**: Small programs that work together
- **Text-based**: Text files for configuration and data
- **Modularity**: Build complex systems from simple components
- **Clarity**: Clear, readable code and interfaces

**3. Stability and Reliability:**
- **Uptime**: Linux servers can run for years without rebooting
- **Crash Resistance**: Isolated processes prevent system-wide crashes
- **Memory Management**: Efficient memory handling
- **Process Management**: Robust process scheduling and management
- **Real-world example**: Many Linux servers have uptime measured in years

**4. Security:**
- **User Permissions**: Granular access control
- **Principle of Least Privilege**: Users have minimum necessary permissions
- **Regular Updates**: Security patches available quickly
- **Audit Trail**: Comprehensive logging capabilities
- **Isolation**: Processes and users are isolated from each other

**5. Performance:**
- **Efficient**: Low overhead, high performance
- **Scalable**: Handles everything from embedded devices to supercomputers
- **Resource Management**: Excellent resource allocation
- **Optimization**: Highly optimized for various hardware architectures

**Linux vs Other Operating Systems:**

**Linux:**
- **Cost**: Free and open-source
- **Customization**: Highly customizable (kernel, distributions, desktop environments)
- **Use Cases**: Servers, cloud, embedded, development, supercomputers
- **Interface**: Primarily command-line, GUI available
- **Software**: Vast repository of free software
- **Community**: Large, active open-source community
- **Learning Curve**: Steeper initially, powerful once learned
- **Hardware Support**: Excellent (especially servers and embedded)

**Windows:**
- **Cost**: Proprietary, requires license (expensive for servers)
- **Customization**: Limited (closed source)
- **Use Cases**: Desktop, enterprise servers, gaming
- **Interface**: GUI-focused, command-line available (PowerShell)
- **Software**: Commercial software ecosystem
- **Community**: Microsoft support, commercial ecosystem
- **Learning Curve**: Easier for beginners (GUI-focused)
- **Hardware Support**: Excellent for desktop hardware

**macOS:**
- **Cost**: Proprietary, requires Apple hardware
- **Customization**: Limited (closed source, but Unix-based)
- **Use Cases**: Desktop, development, creative work
- **Interface**: GUI-focused, Unix terminal available
- **Software**: Apple ecosystem, some open-source
- **Community**: Apple support, developer community
- **Learning Curve**: Moderate (Unix-like, but GUI-focused)
- **Hardware Support**: Limited to Apple hardware

**Why Linux Matters:**

**1. Servers:**
- **Web Servers**: Powers majority of web servers worldwide
  - Apache HTTP Server: Most popular web server
  - Nginx: High-performance web server and reverse proxy
  - Both run primarily on Linux
- **Database Servers**: MySQL, PostgreSQL, MongoDB run on Linux
- **Application Servers**: Java, Python, Node.js applications
- **File Servers**: Samba, NFS for file sharing
- **Real-world**: Over 90% of cloud instances run Linux

**2. Cloud Computing:**
- **Foundation of Cloud**: AWS, Google Cloud, Azure run on Linux
- **Containers**: Docker and Kubernetes are Linux-native
- **Serverless**: Many serverless platforms run on Linux
- **Microservices**: Container orchestration platforms
- **Real-world**: All major cloud providers use Linux extensively

**3. Containers and Orchestration:**
- **Docker**: Container platform built on Linux
- **Kubernetes**: Container orchestration (originally for Linux)
- **Container Images**: Most container images are Linux-based
- **Microservices**: Enable microservices architecture
- **DevOps**: Essential for modern DevOps practices

**4. Embedded Systems and IoT:**
- **Routers**: Most network routers run Linux
- **Smart TVs**: Many smart TVs use Linux
- **IoT Devices**: Raspberry Pi, embedded devices
- **Automotive**: Car infotainment systems
- **Real-world**: Linux powers billions of embedded devices

**5. Supercomputers:**
- **Top 500**: Linux powers all top 500 supercomputers
- **High-Performance Computing**: Scientific computing, research
- **Parallel Processing**: Excellent for parallel workloads
- **Real-world**: 100% of top 500 supercomputers run Linux

**6. Mobile Devices:**
- **Android**: Based on Linux kernel
- **Billions of Devices**: Android powers billions of smartphones
- **Tablets**: Many tablets run Android/Linux
- **Real-world**: Linux (via Android) is the most widely used OS

**7. Development:**
- **Preferred by Developers**: Many developers prefer Linux
- **Development Tools**: Excellent development tooling
- **Package Management**: Easy software installation
- **Scripting**: Powerful shell scripting capabilities

**Linux Foundation Training Path (2024-2025 Best Practices):**

**Foundation Level:**
- **Introduction to Linux (LFS101)**: Linux fundamentals, command-line basics
- **Linux Essentials**: Basic Linux knowledge for beginners
- **Prerequisites**: None, suitable for beginners

**System Administration:**
- **Linux System Administration Essentials (LFS207)**: Intermediate-level foundational course
- **Linux System Administration**: Comprehensive system administration skills
- **Prerequisites**: Basic Linux knowledge

**Certification Paths:**
- **Linux Foundation Certified System Administrator (LFCS)**: Entry-level certification
- **Linux Foundation Certified Engineer (LFCE)**: Advanced certification
- **Kubernetes Certifications**: CKA, CKAD, CKS (container orchestration)
- **Cloud Native Certifications**: KCNA, KCSA (cloud-native technologies)

**Key Skills for 2024-2025:**
- **Containerization**: Docker, Podman, container management
- **Orchestration**: Kubernetes, container orchestration
- **Automation**: Ansible, Terraform, Infrastructure as Code
- **Cloud Integration**: AWS, Azure, GCP Linux administration
- **Security**: SELinux, firewalls, security hardening
- **Monitoring**: System monitoring, log management, observability
- **Real-world**: Most software development happens on Linux

**8. Enterprise:**
- **Enterprise Servers**: Red Hat Enterprise Linux, SUSE Linux Enterprise
- **Mission-Critical**: Powers mission-critical enterprise applications
- **Cost Savings**: Significant cost savings vs proprietary OS
- **Support**: Enterprise support available from vendors
- **Real-world**: Major enterprises rely on Linux for critical workloads

**Linux Philosophy:**

The Unix/Linux philosophy consists of several key principles:

**1. "Everything is a File":**
- Devices, processes, network connections are represented as files
- Unified interface for interacting with system components
- Simplifies system interaction and programming
- Example: /dev/sda (disk), /proc/cpuinfo (CPU info), /dev/tty (terminal)

**2. "Small, Focused Programs":**
- Each program does one thing well
- Programs are simple and focused
- Easier to understand, maintain, and debug
- Example: grep (search), wc (word count), sort (sorting)

**3. "Chain Programs Together":**
- Programs can be combined using pipes (|)
- Output of one program becomes input of another
- Build complex operations from simple tools
- Example: cat file.txt | grep "error" | sort | uniq

**4. "Avoid Captive User Interfaces":**
- Prefer command-line interfaces
- Scriptable and automatable
- Can be combined with other programs
- Example: Command-line tools vs GUI-only applications

**5. "Store Data in Flat Text Files":**
- Configuration files are plain text
- Human-readable and editable
- Version control friendly
- Example: /etc/hosts, /etc/passwd, configuration files

**6. "Make Each Program Do One Thing Well":**
- Single responsibility principle
- Programs are tools, not solutions
- Combine tools to solve problems
- Example: Unix tools (ls, cat, grep, awk, sed)

**7. "Worse is Better":**
- Simplicity over perfection
- Good enough is better than perfect
- Practical solutions over theoretical perfection
- Example: Simple, working solutions preferred

**Linux Architecture:**

**Kernel:**
- Core of the operating system
- Manages hardware resources
- Provides system calls to applications
- Handles process scheduling, memory management, device drivers

**System Libraries:**
- GNU C Library (glibc) and other libraries
- Provide functions for applications
- Interface between applications and kernel

**System Tools:**
- Essential utilities (ls, cp, mv, etc.)
- Usually part of GNU coreutils
- Provide basic system functionality

**Shell:**
- Command-line interface
- Bash, Zsh, Fish, etc.
- Interprets commands and executes programs

**Desktop Environment (Optional):**
- GUI for desktop use
- GNOME, KDE, XFCE, etc.
- Not needed for servers

**Applications:**
- User applications
- Web browsers, text editors, development tools
- Can be GUI or command-line

**Getting Started with Linux:**

**Why Learn Linux:**

**1. Career Opportunities:**
- **DevOps**: Essential for DevOps roles
- **Cloud Computing**: Required for cloud platforms
- **System Administration**: Core skill for sysadmins
- **Software Development**: Many developers use Linux
- **Cybersecurity**: Important for security professionals

**2. Practical Benefits:**
- **Cost**: Free to use and learn
- **Skills**: Transferable across many technologies
- **Understanding**: Deep understanding of how computers work
- **Automation**: Powerful scripting and automation capabilities
- **Control**: Complete control over your system

**3. Technology Foundation:**
- **Cloud**: Understanding cloud requires Linux knowledge
- **Containers**: Docker and Kubernetes are Linux-based
- **DevOps**: Modern DevOps relies heavily on Linux
- **Networking**: Many network devices run Linux
- **Embedded**: IoT and embedded systems use Linux

**Common Linux Distributions:**

**For Beginners:**
- **Ubuntu**: Most popular, user-friendly, great documentation
- **Linux Mint**: Even more user-friendly, Windows-like interface
- **Fedora**: Modern, cutting-edge, good for learning

**For Servers:**
- **Ubuntu Server**: Popular, well-supported, easy to use
- **CentOS/Rocky Linux**: Enterprise-standard, RHEL-compatible
- **Debian**: Stable, reliable, widely used
- **Red Hat Enterprise Linux**: Commercial, enterprise support

**For Advanced Users:**
- **Arch Linux**: Rolling release, minimal, learn everything
- **Gentoo**: Source-based, maximum customization
- **Slackware**: Traditional, oldest surviving distribution

**Conclusion:**

Linux is not just an operating system—it's a fundamental technology that powers the modern digital world. From the servers hosting websites to the smartphones in our pockets, from cloud infrastructure to supercomputers, Linux is everywhere.

Understanding Linux is essential for:
- **Developers**: Most development happens on Linux
- **DevOps Engineers**: Containers, cloud, automation all use Linux
- **System Administrators**: Core skill for managing servers
- **Cloud Engineers**: Cloud platforms are Linux-based
- **Anyone in Tech**: Linux knowledge is increasingly valuable

The key to learning Linux is hands-on practice. Start with a beginner-friendly distribution like Ubuntu, learn basic commands, understand the file system, practice regularly, and gradually build your skills. Linux rewards those who invest time in learning it with powerful capabilities and deep understanding of how computers work.

Remember: Linux is about freedom—freedom to use, modify, and distribute. This freedom has enabled Linux to become the foundation of modern computing infrastructure. Whether you're running a web server, developing software, or managing cloud infrastructure, Linux skills are essential for success in modern technology.`,
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

**What Makes a Distribution:**

**Core Components:**
- **Linux Kernel**: The core operating system kernel
- **System Libraries**: GNU C Library (glibc) and other essential libraries
- **System Tools**: Basic utilities (ls, cp, mv, grep, etc.)
- **Shell**: Command-line interface (Bash, Zsh, etc.)
- **Package Manager**: Tool for installing and managing software
- **Desktop Environment** (optional): GUI for desktop use
- **Applications**: Software packages and applications

**Distribution Philosophy:**

Each distribution has its own philosophy and goals:
- **Stability vs Cutting-Edge**: Some prioritize stability, others latest features
- **Ease of Use vs Control**: Some are user-friendly, others give full control
- **Commercial vs Community**: Some are commercial, others community-driven
- **Package Management**: Different package managers and philosophies

**Major Distribution Families:**

**1. Debian-based Distributions:**

**Debian:**
- **Philosophy**: Stability, free software, community-driven
- **Release Cycle**: Stable releases every 2-3 years
- **Package Manager**: apt (Advanced Package Tool)
- **Use Cases**: Servers, stable desktop systems
- **Strengths**: 
  - Extremely stable and reliable
  - Huge software repository (60,000+ packages)
  - Strong commitment to free software
  - Excellent for servers
- **Weaknesses**: 
  - Software can be outdated
  - Slower release cycle
  - Less user-friendly than Ubuntu
- **Real-world**: Powers many web servers and cloud instances

**Ubuntu:**
- **Philosophy**: User-friendly, "Linux for human beings"
- **Release Cycle**: Regular releases every 6 months, LTS every 2 years
- **Package Manager**: apt (inherited from Debian)
- **Use Cases**: Desktop, servers, cloud, IoT
- **Strengths**: 
  - Most popular Linux distribution
  - Excellent documentation and community support
  - User-friendly, easy to install
  - Large software repository
  - Strong commercial support (Canonical)
- **Weaknesses**: 
  - Some proprietary software included
  - Can be resource-heavy
  - Some customization removed for simplicity
- **Variants**: 
  - **Ubuntu Desktop**: Full desktop experience
  - **Ubuntu Server**: Server-focused, no GUI
  - **Ubuntu Core**: Minimal, IoT-focused
  - **Kubuntu**: KDE desktop environment
  - **Xubuntu**: XFCE desktop (lightweight)
  - **Lubuntu**: LXQt desktop (very lightweight)
- **Real-world**: Most popular Linux distribution, used by millions

**Linux Mint:**
- **Philosophy**: User-friendly, Windows-like experience
- **Base**: Based on Ubuntu (or Debian)
- **Package Manager**: apt
- **Use Cases**: Desktop, especially for Windows users switching to Linux
- **Strengths**: 
  - Very user-friendly
  - Windows-like interface (Cinnamon desktop)
  - Includes multimedia codecs by default
  - Good for beginners
- **Weaknesses**: 
  - Less popular than Ubuntu
  - Smaller community
  - Based on Ubuntu, so inherits Ubuntu's issues
- **Real-world**: Popular choice for Windows users trying Linux

**Kali Linux:**
- **Philosophy**: Security-focused, penetration testing
- **Base**: Based on Debian
- **Package Manager**: apt
- **Use Cases**: Security testing, penetration testing, forensics
- **Strengths**: 
  - Pre-installed security tools (hundreds of tools)
  - Regularly updated security tools
  - Excellent for security professionals
- **Weaknesses**: 
  - Not for general use
  - Requires security knowledge
  - Can be misused if not careful
- **Real-world**: Standard tool for cybersecurity professionals

**2. Red Hat-based Distributions:**

**Red Hat Enterprise Linux (RHEL):**
- **Philosophy**: Enterprise-grade, commercial, supported
- **Release Cycle**: Major releases every 3-4 years, minor updates regularly
- **Package Manager**: yum (older) or dnf (newer)
- **Use Cases**: Enterprise servers, mission-critical systems
- **Strengths**: 
  - Enterprise-grade stability and support
  - Long support lifecycle (10+ years)
  - Excellent documentation
  - Strong security features
  - Commercial support available
- **Weaknesses**: 
  - Requires paid subscription
  - Software can be conservative/outdated
  - Less cutting-edge features
- **Real-world**: Powers many enterprise servers and data centers

**CentOS (Community Enterprise Operating System):**
- **Philosophy**: Free RHEL clone, community-driven
- **Status**: Now CentOS Stream (rolling release, upstream of RHEL)
- **Package Manager**: yum/dnf
- **Use Cases**: Servers, when RHEL compatibility needed without cost
- **Strengths**: 
  - Binary compatible with RHEL
  - Free and open-source
  - Stable and reliable
  - Good for learning RHEL
- **Weaknesses**: 
  - CentOS 8 ended, now CentOS Stream (different model)
  - Less support than RHEL
- **Note**: Traditional CentOS ended, replaced by CentOS Stream
- **Real-world**: Was very popular for servers, now transitioning

**Rocky Linux:**
- **Philosophy**: RHEL-compatible, community-driven replacement for CentOS
- **Release Cycle**: Follows RHEL release cycle
- **Package Manager**: dnf
- **Use Cases**: Servers, RHEL-compatible systems
- **Strengths**: 
  - Binary compatible with RHEL
  - Free and open-source
  - Community-driven
  - Fills gap left by CentOS
- **Weaknesses**: 
  - Relatively new (founded 2020)
  - Smaller community than CentOS had
- **Real-world**: Growing adoption as CentOS replacement

**Fedora:**
- **Philosophy**: Cutting-edge, innovation, community-driven
- **Release Cycle**: New release every 6 months, supported for ~13 months
- **Package Manager**: dnf
- **Use Cases**: Desktop, development, testing new technologies
- **Strengths**: 
  - Latest software and features
  - Strong security (SELinux enabled by default)
  - Good for developers
  - Sponsored by Red Hat (but community-driven)
- **Weaknesses**: 
  - Faster release cycle (more updates needed)
  - Can be less stable than RHEL
  - Shorter support lifecycle
- **Real-world**: Popular with developers, used by Linus Torvalds

**3. Arch-based Distributions:**

**Arch Linux:**
- **Philosophy**: Simplicity, user-centric, do-it-yourself
- **Release Cycle**: Rolling release (continuous updates)
- **Package Manager**: pacman
- **Use Cases**: Advanced users, learning Linux deeply, customization
- **Strengths**: 
  - Rolling release (always up-to-date)
  - Minimal base installation (build what you need)
  - Excellent documentation (Arch Wiki)
  - Huge software repository (AUR - Arch User Repository)
  - Maximum control and customization
- **Weaknesses**: 
  - Steep learning curve
  - Requires manual configuration
  - Can break if not careful with updates
  - Not for beginners
- **Real-world**: Popular with advanced Linux users and developers

**Manjaro:**
- **Philosophy**: User-friendly Arch Linux
- **Base**: Based on Arch Linux
- **Package Manager**: pacman
- **Use Cases**: Desktop, users who want Arch benefits with easier setup
- **Strengths**: 
  - Easier installation than Arch
  - Rolling release benefits
  - Access to AUR
  - Multiple desktop environments available
- **Weaknesses**: 
  - Still more complex than Ubuntu
  - Can have compatibility issues (delayed Arch updates)
- **Real-world**: Popular Arch-based distribution for desktop users

**4. Other Notable Distributions:**

**openSUSE:**
- **Philosophy**: Stable, reliable, enterprise and desktop
- **Release Cycle**: Regular releases, Leap (stable) and Tumbleweed (rolling)
- **Package Manager**: zypper
- **Use Cases**: Desktop, servers, enterprise
- **Strengths**: 
  - Very stable and reliable
  - Good for both desktop and server
  - YaST (configuration tool) is user-friendly
  - Strong in Europe
- **Weaknesses**: 
  - Less popular globally
  - Smaller community than Ubuntu/Fedora
- **Real-world**: Popular in Europe, used by some enterprises

**Slackware:**
- **Philosophy**: Traditional Unix-like, simplicity, stability
- **Release Cycle**: Very slow (years between releases)
- **Package Manager**: pkgtool, slackpkg
- **Use Cases**: Learning Unix, servers, embedded systems
- **Strengths**: 
  - Oldest surviving Linux distribution
  - Very stable
  - Traditional Unix feel
  - Minimal and simple
- **Weaknesses**: 
  - Very slow release cycle
  - Manual package management
  - Steep learning curve
  - Not for beginners
- **Real-world**: Used by Unix purists and some embedded systems

**Gentoo:**
- **Philosophy**: Source-based, maximum optimization and customization
- **Release Cycle**: Rolling release
- **Package Manager**: Portage
- **Use Cases**: Maximum performance, learning, customization
- **Strengths**: 
  - Compile from source (optimized for your hardware)
  - Maximum customization
  - Excellent documentation
  - Very flexible
- **Weaknesses**: 
  - Very time-consuming (compile everything)
  - Steep learning curve
  - Not practical for most users
- **Real-world**: Used by performance enthusiasts and learning purposes

**Choosing a Distribution:**

**Decision Factors:**

**1. Experience Level:**
- **Beginner**: Ubuntu, Linux Mint, Fedora
- **Intermediate**: Debian, openSUSE, Manjaro
- **Advanced**: Arch Linux, Gentoo, Slackware

**2. Use Case:**
- **Desktop**: Ubuntu, Linux Mint, Fedora, Manjaro
- **Server**: Ubuntu Server, Debian, RHEL/Rocky Linux, CentOS Stream
- **Development**: Fedora, Ubuntu, Arch Linux
- **Security**: Kali Linux, Parrot OS
- **Enterprise**: RHEL, SUSE Linux Enterprise, Ubuntu Server

**3. Hardware:**
- **Modern Hardware**: Any distribution
- **Old Hardware**: Lubuntu, Xubuntu, Debian (lightweight)
- **ARM/Raspberry Pi**: Raspberry Pi OS, Ubuntu ARM
- **Servers**: Ubuntu Server, Debian, RHEL

**4. Support Needs:**
- **Commercial Support**: RHEL, SUSE Linux Enterprise, Ubuntu (Canonical)
- **Community Support**: Ubuntu, Fedora, Arch Linux (excellent communities)
- **Documentation**: Arch Linux (Arch Wiki), Ubuntu, Fedora

**5. Software Requirements:**
- **Latest Software**: Fedora, Arch Linux, openSUSE Tumbleweed
- **Stable Software**: Debian, RHEL, Ubuntu LTS
- **Specific Software**: Check if software is available in distribution's repository

**Package Managers:**

**Understanding Package Managers:**

Package managers are tools that automate the process of installing, updating, configuring, and removing software. They handle dependencies automatically and ensure software compatibility.

**apt (Advanced Package Tool) - Debian/Ubuntu:**
- **Commands**: apt-get, apt-cache, apt (newer unified command)
- **Repositories**: Debian/Ubuntu repositories
- **Strengths**: 
  - Easy to use
  - Large repository
  - Good dependency resolution
  - Well-documented
- **Common Commands**:
  - apt update: Update package lists
  - apt upgrade: Upgrade installed packages
  - apt install package: Install package
  - apt remove package: Remove package
  - apt search keyword: Search for packages

**yum/dnf (Yellowdog Updater Modified/Dandified YUM) - Red Hat/Fedora:**
- **Commands**: yum (older), dnf (newer, default in Fedora)
- **Repositories**: RPM repositories
- **Strengths**: 
  - Good dependency resolution
  - Transaction support (can rollback)
  - Plugin system
- **Common Commands**:
  - dnf update: Update packages
  - dnf install package: Install package
  - dnf remove package: Remove package
  - dnf search keyword: Search for packages

**pacman (Package Manager) - Arch Linux:**
- **Commands**: pacman
- **Repositories**: Official repositories + AUR (Arch User Repository)
- **Strengths**: 
  - Fast and efficient
  - Simple syntax
  - Access to AUR (huge software collection)
- **Common Commands**:
  - pacman -Syu: Update system
  - pacman -S package: Install package
  - pacman -R package: Remove package
  - pacman -Ss keyword: Search for packages

**zypper - openSUSE:**
- **Commands**: zypper
- **Repositories**: openSUSE repositories
- **Strengths**: 
  - Good dependency resolution
  - Transaction support
  - Repository management
- **Common Commands**:
  - zypper update: Update packages
  - zypper install package: Install package
  - zypper remove package: Remove package
  - zypper search keyword: Search for packages

**Distribution Comparison:**

**Stability Ranking:**
1. Debian Stable (most stable)
2. RHEL/Rocky Linux
3. Ubuntu LTS
4. openSUSE Leap
5. Fedora
6. Arch Linux (rolling, can be less stable)

**Ease of Use Ranking:**
1. Linux Mint (easiest)
2. Ubuntu
3. Fedora
4. openSUSE
5. Debian
6. Arch Linux (most challenging)

**Software Freshness Ranking:**
1. Arch Linux (always latest)
2. Fedora (cutting-edge)
3. openSUSE Tumbleweed (rolling)
4. Ubuntu (regular releases)
5. Debian (stable, older software)
6. RHEL (enterprise, conservative)

**Best Practices:**

**1. Start with Beginner-Friendly:**
- Begin with Ubuntu or Linux Mint
- Learn basics before switching
- Don't start with Arch or Gentoo
- **Example**: New users should start with Ubuntu

**2. Match Distribution to Use Case:**
- Desktop: Ubuntu, Linux Mint, Fedora
- Server: Ubuntu Server, Debian, RHEL
- Learning: Ubuntu, Fedora, Arch Linux
- **Example**: Use Ubuntu Server for web servers

**3. Consider Support:**
- Commercial support: RHEL, SUSE, Ubuntu (Canonical)
- Community support: Ubuntu, Fedora, Arch Linux
- **Example**: Enterprise needs commercial support (RHEL)

**4. Test Before Committing:**
- Try distributions in virtual machines
- Test with live USB
- Don't switch production systems without testing
- **Example**: Test new distribution in VM before installing

**5. Learn Package Manager:**
- Master your distribution's package manager
- Understand repository management
- Learn dependency resolution
- **Example**: Learn apt commands for Ubuntu

**Common Mistakes:**

**1. Choosing Wrong Distribution:**
- **Problem**: Choosing Arch Linux as beginner
- **Impact**: Frustration, giving up on Linux
- **Solution**: Start with Ubuntu or Linux Mint
- **Example**: Beginner choosing Arch Linux and struggling

**2. Not Updating Regularly:**
- **Problem**: Not updating system regularly
- **Impact**: Security vulnerabilities, outdated software
- **Solution**: Regular updates (daily/weekly)
- **Example**: System compromised due to unpatched vulnerabilities

**3. Mixing Repositories:**
- **Problem**: Adding incompatible repositories
- **Impact**: Dependency conflicts, broken system
- **Solution**: Use official repositories, be careful with third-party repos
- **Example**: Adding incompatible PPA breaking Ubuntu system

**4. Ignoring Documentation:**
- **Problem**: Not reading distribution documentation
- **Impact**: Missing features, incorrect usage
- **Solution**: Read official documentation
- **Example**: Not understanding Arch installation process

**Real-World Distribution Usage:**

**Web Servers:**
- **Ubuntu Server**: Most popular for web servers
- **Debian**: Stable, reliable choice
- **RHEL/Rocky Linux**: Enterprise web servers
- **Real-world**: Majority of web servers run Ubuntu or Debian

**Cloud Computing:**
- **Ubuntu**: Most popular on AWS, Google Cloud, Azure
- **Amazon Linux**: AWS-optimized distribution
- **Container Images**: Alpine Linux (minimal), Ubuntu, Debian
- **Real-world**: Most cloud instances run Ubuntu

**Enterprise Servers:**
- **RHEL**: Enterprise standard
- **SUSE Linux Enterprise**: Enterprise alternative
- **Ubuntu Server**: Growing enterprise adoption
- **Real-world**: Large enterprises use RHEL or SUSE

**Desktop:**
- **Ubuntu**: Most popular desktop Linux
- **Linux Mint**: Popular for Windows users
- **Fedora**: Popular with developers
- **Real-world**: Ubuntu dominates desktop Linux market

**Development:**
- **Fedora**: Cutting-edge, good for development
- **Ubuntu**: Popular development platform
- **Arch Linux**: Maximum control for developers
- **Real-world**: Many developers use Fedora or Ubuntu

**Conclusion:**

Choosing the right Linux distribution is important for your success and satisfaction. Each distribution has its strengths and is suited for different use cases. The key is to:

- **Start simple**: Begin with beginner-friendly distributions
- **Match to use case**: Choose distribution suited for your needs
- **Learn the package manager**: Essential skill for any distribution
- **Don't be afraid to try**: Test distributions before committing
- **Consider support**: Commercial vs community support

Remember: The best distribution is the one that works for you. Don't get caught up in distribution wars - focus on learning Linux fundamentals, which transfer across distributions. Whether you choose Ubuntu, Fedora, Arch, or any other distribution, the core Linux knowledge you gain will be valuable regardless of which distribution you use.

Most importantly: **Start using Linux**. Don't spend too much time choosing - pick Ubuntu or Linux Mint, install it, and start learning. You can always switch distributions later as you gain experience and understand your needs better.`,
					CodeExamples: `# Ubuntu/Debian - Check distribution
lsb_release -a
# Output:
# Distributor ID: Ubuntu
# Description:    Ubuntu 22.04.3 LTS
# Release:        22.04
# Codename:       jammy

# Red Hat/CentOS - Check distribution
cat /etc/redhat-release
# Output: CentOS Linux release 8.5.2111

# Arch Linux - Check distribution
cat /etc/arch-release
# Output: Arch Linux

# Package manager examples
# Ubuntu/Debian
apt update && apt upgrade
apt install package-name

# Red Hat/Fedora
dnf update
dnf install package-name

# Arch
pacman -Syu
pacman -S package-name`,
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
					Content: `The terminal (command line) is the most powerful way to interact with Linux. Understanding it is essential for Linux proficiency.

**What is a Terminal?**
- Text-based interface to the operating system
- Also called: shell, command line, CLI (Command Line Interface)
- More powerful than GUI for many tasks
- Faster and more efficient for experienced users

**Common Shells:**
- **bash**: Most common default shell (Bourne Again Shell)
- **zsh**: Enhanced shell, default on macOS
- **fish**: User-friendly with syntax highlighting
- **sh**: Basic shell (POSIX compliant)

**Terminal Components:**
- **Prompt**: Shows current directory and user (e.g., user@host:~$)
- **Command**: What you type
- **Arguments**: Options and parameters
- **Output**: Results displayed

**Command Structure:**

command [options] [arguments]


Example: ls -l /home
- ls: command
- -l: option (long format)
- /home: argument (directory)`,
					CodeExamples: `# Check current shell
echo $SHELL
# Output: /bin/bash

# List available shells
cat /etc/shells

# Change shell (temporarily)
zsh

# Change shell (permanently)
chsh -s /bin/zsh

# Command structure examples
ls                    # Command only
ls -l                 # Command with option
ls -l /home          # Command with option and argument
ls -lah /home        # Multiple options combined

# Understanding the prompt
# user@hostname:~/directory$ 
# user: current username
# hostname: computer name
# ~/directory: current directory (~ = home)
# $: regular user prompt (# = root user)`,
				},
				{
					Title: "Navigation Commands",
					Content: `Navigating the file system is fundamental to using Linux. These commands help you move around directories.

**Essential Navigation Commands:**

**pwd (Print Working Directory):**
- Shows current directory path
- Always know where you are

**cd (Change Directory):**
- Move to different directories
- cd or cd ~: Go to home directory
- cd ..: Go up one level
- cd -: Go to previous directory
- cd /: Go to root directory

**ls (List):**
- List files and directories
- ls: Basic listing
- ls -l: Long format (detailed)
- ls -a: Show hidden files
- ls -h: Human-readable sizes
- ls -R: Recursive (show subdirectories)

**Path Types:**
- **Absolute**: Full path from root (/home/user/documents)
- **Relative**: Path relative to current directory (./documents or ../parent)
- **~**: Shortcut for home directory
- **.**: Current directory
- **..**: Parent directory`,
					CodeExamples: `# Print current directory
pwd
# Output: /home/username

# Navigate to home directory
cd ~
# or
cd

# Navigate to specific directory
cd /home/username/documents

# Go up one level
cd ..

# Go to root directory
cd /

# Go to previous directory
cd -

# List current directory contents
ls

# List with details
ls -l
# Output shows: permissions, owner, size, date, name

# List all files (including hidden)
ls -a
# Hidden files start with '.'

# List with human-readable sizes
ls -lh

# Combine options
ls -lah
# -l: long format
# -a: all files
# -h: human-readable

# List specific directory
ls /etc

# List with pattern matching
ls *.txt
# Shows only .txt files

# Navigate using relative paths
cd ./documents      # Current directory's documents folder
cd ../parent        # Parent directory's parent folder
cd ~/documents      # Home directory's documents folder`,
				},
				{
					Title: "File Operations",
					Content: `Working with files is a core Linux skill. Learn to create, copy, move, rename, and delete files.

**File Operations:**

**cp (Copy):**
- Copy files and directories
- cp source destination
- cp -r: Copy directories recursively
- cp -i: Interactive (prompt before overwrite)
- cp -v: Verbose (show what's being copied)

**mv (Move/Rename):**
- Move files to different location
- Rename files
- mv source destination
- Can move multiple files

**rm (Remove):**
- Delete files and directories
- **Warning**: Permanent deletion (no trash)
- rm file: Delete file
- rm -r: Delete directory recursively
- rm -f: Force (no prompts)
- rm -i: Interactive (safer)

**mkdir (Make Directory):**
- Create new directories
- mkdir dirname
- mkdir -p: Create parent directories if needed

**touch:**
- Create empty file
- Update file timestamp
- touch filename

**File Naming:**
- Case-sensitive (File.txt ≠ file.txt)
- Avoid spaces (use underscores or hyphens)
- Avoid special characters`,
					CodeExamples: `# Create a file
touch myfile.txt

# Create multiple files
touch file1.txt file2.txt file3.txt

# Create a directory
mkdir mydir

# Create nested directories
mkdir -p parent/child/grandchild

# Copy a file
cp source.txt destination.txt

# Copy to directory
cp file.txt /home/user/documents/

# Copy directory
cp -r sourcedir destdir

# Copy with confirmation
cp -i source.txt dest.txt

# Move/rename a file
mv oldname.txt newname.txt

# Move file to directory
mv file.txt /home/user/documents/

# Move multiple files
mv file1.txt file2.txt /home/user/documents/

# Delete a file
rm file.txt

# Delete directory
rm -r directory

# Delete with confirmation (safer)
rm -ri directory

# Force delete (dangerous!)
rm -rf directory

# List files to verify operations
ls -l`,
				},
				{
					Title: "Command History & Shortcuts",
					Content: `Linux provides powerful shortcuts and history features to make command line work more efficient.

**Command History:**
- **history**: View command history
- **Up/Down arrows**: Navigate through history
- **Ctrl+R**: Search history interactively
- **!n**: Execute command number n from history
- **!!**: Repeat last command
- **!string**: Execute last command starting with string

**Keyboard Shortcuts:**
- **Ctrl+C**: Cancel/interrupt current command
- **Ctrl+D**: Exit shell or end input
- **Ctrl+L**: Clear screen (same as clear)
- **Ctrl+A**: Move cursor to beginning of line
- **Ctrl+E**: Move cursor to end of line
- **Ctrl+U**: Delete from cursor to beginning of line
- **Ctrl+K**: Delete from cursor to end of line
- **Ctrl+W**: Delete word before cursor
- **Tab**: Auto-complete commands and filenames
- **Double Tab**: Show completion options

**Command Aliases:**
- Create shortcuts for long commands
- alias ll='ls -lah'
- alias ..='cd ..'
- View aliases: alias
- Remove alias: unalias name

**Environment Variables:**
- $HOME: Home directory path
- $USER: Current username
- $PATH: Directories searched for commands
- $PWD: Current directory`,
					CodeExamples: `# View command history
history

# View last 10 commands
history | tail -10

# Execute command from history by number
!42

# Repeat last command
!!

# Execute last command starting with 'ls'
!ls

# Search history interactively
# Press Ctrl+R, type search term

# Clear history
history -c

# Keyboard shortcuts in action
# Type: ls -l /very/long/path
# Press Ctrl+A: cursor moves to beginning
# Press Ctrl+E: cursor moves to end
# Press Ctrl+U: deletes entire line

# Tab completion
# Type: ls /usr/b
# Press Tab: completes to /usr/bin

# Create aliases
alias ll='ls -lah'
alias ..='cd ..'
alias grep='grep --color=auto'

# Use alias
ll  # Executes: ls -lah

# View all aliases
alias

# Remove alias
unalias ll

# View environment variables
echo $HOME
echo $USER
echo $PATH

# Add to PATH (temporarily)
export PATH=$PATH:/new/directory

# Make alias permanent (add to ~/.bashrc)
echo "alias ll='ls -lah'" >> ~/.bashrc
source ~/.bashrc`,
				},
				{
					Title: "Command Aliases & Customization",
					Content: `Customizing your shell environment makes Linux more efficient and personalized.

**What are Aliases?**
- Shortcuts for long commands
- Save typing time
- Make commands more memorable
- Can include options and arguments

**Creating Aliases:**
- alias name='command': Create alias
- alias: List all aliases
- unalias name: Remove alias
- Temporary: Lost when session ends
- Permanent: Add to shell config file

**Common Aliases:**
- alias ll='ls -lah': Long listing with details
- alias la='ls -A': List all including hidden
- alias ..='cd ..': Go up one directory
- alias ...='cd ../..': Go up two directories
- alias grep='grep --color=auto': Colorized grep
- alias vi='vim': Use vim instead of vi

**Shell Configuration Files:**
- ~/.bashrc: Bash config (runs for interactive shells)
- ~/.bash_profile: Login shell config
- ~/.profile: General profile (all shells)
- ~/.bash_aliases: Dedicated alias file (sourced by .bashrc)

**Shell Customization:**
- **Prompt (PS1)**: Customize command prompt
- **Environment Variables**: PATH, EDITOR, etc.
- **Functions**: More powerful than aliases
- **Completion**: Tab completion customization

**Best Practices:**
- Use descriptive alias names
- Don't override essential commands
- Document aliases with comments
- Keep aliases simple
- Use functions for complex logic`,
					CodeExamples: `# Create temporary alias
alias ll='ls -lah'
ll  # Executes: ls -lah

# Create alias with options
alias grep='grep --color=auto'
grep "error" file.txt  # Now colorized

# Create alias for directory navigation
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# Create alias for common operations
alias rm='rm -i'  # Interactive delete (safer)
alias cp='cp -i'  # Interactive copy
alias mv='mv -i'  # Interactive move

# List all aliases
alias

# Remove alias
unalias ll

# Make aliases permanent
# Edit ~/.bashrc
nano ~/.bashrc

# Add aliases:
alias ll='ls -lah'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'

# Reload configuration
source ~/.bashrc
# or
. ~/.bashrc

# Custom prompt (PS1)
export PS1='[\u@\h \W]\$ '
# \u = username, \h = hostname, \W = current directory

# More advanced prompt
export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
# Green username@host, blue directory

# Environment variables
export EDITOR=vim
export VISUAL=vim
export PAGER=less

# Add to PATH
export PATH=$PATH:/usr/local/bin

# Shell functions (more powerful than aliases)
function mkcd() {
    mkdir -p "$1" && cd "$1"
}
# Usage: mkcd newdir

function extract() {
    if [ -f "$1" ]; then
        case "$1" in
            *.tar.bz2) tar xjf "$1" ;;
            *.tar.gz) tar xzf "$1" ;;
            *.bz2) bunzip2 "$1" ;;
            *.rar) unrar x "$1" ;;
            *.gz) gunzip "$1" ;;
            *.tar) tar xf "$1" ;;
            *.tbz2) tar xjf "$1" ;;
            *.tgz) tar xzf "$1" ;;
            *.zip) unzip "$1" ;;
            *.Z) uncompress "$1" ;;
            *.7z) 7z x "$1" ;;
            *) echo "'$1' cannot be extracted" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}
# Usage: extract archive.tar.gz`,
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
					Content: `Linux organizes everything in a hierarchical tree structure starting from root (/). Understanding this structure is crucial.

**Root Directory (/):**
- Top-level directory
- Everything branches from here
- Similar to C:\ on Windows

**Important Directories:**

**/bin**: Essential binary executables (ls, cp, mv)
**/boot**: Boot loader files and kernel
**/dev**: Device files (hardware representation)
**/etc**: System configuration files
**/home**: User home directories
**/lib**: Shared libraries
**/media**: Removable media mount points
**/mnt**: Temporary mount points
**/opt**: Optional/third-party software
**/proc**: Virtual filesystem (process/kernel info)
**/root**: Root user's home directory
**/run**: Runtime data (since boot)
**/sbin**: System binaries (admin commands)
**/srv**: Service data
**/sys**: Virtual filesystem (kernel/sys info)
**/tmp**: Temporary files
**/usr**: User programs and data
**/var**: Variable data (logs, cache, spool)

**File Types:**
- **Regular files**: Text, images, executables
- **Directories**: Folders containing files
- **Symbolic links**: Shortcuts to files/directories
- **Device files**: Represent hardware
- **Sockets**: Inter-process communication
- **Named pipes**: Process communication`,
					CodeExamples: `# View root directory structure
ls /

# View directory tree
tree /
# (if tree not installed: apt install tree)

# View specific directory
ls /etc
ls /home
ls /usr/bin

# Check file type
file /bin/ls
# Output: /bin/ls: ELF 64-bit LSB executable

file /etc/passwd
# Output: /etc/passwd: ASCII text

# List with file types
ls -F
# Shows: / for directories, * for executables, @ for links

# Detailed listing showing file types
ls -l
# First character shows type:
# - = regular file
# d = directory
# l = symbolic link
# c = character device
# b = block device
# s = socket
# p = named pipe

# Navigate important directories
cd /etc        # Configuration files
cd /var/log    # Log files
cd /tmp        # Temporary files
cd /usr/bin    # User executables`,
				},
				{
					Title: "File Permissions",
					Content: `Linux uses a permission system to control access to files and directories. Understanding permissions is essential for security.

**Permission Types:**
- **Read (r)**: View file contents or list directory
- **Write (w)**: Modify file or create/delete in directory
- **Execute (x)**: Run file as program or enter directory

**Permission Groups:**
- **Owner (u)**: File owner
- **Group (g)**: Group members
- **Others (o)**: Everyone else

**Permission Representation:**
- **Symbolic**: rwxrwxrwx (read, write, execute for owner/group/others)
- **Octal**: 755 (4+2+1 for each group)
  - 4 = read
  - 2 = write
  - 1 = execute
  - 7 = 4+2+1 (read+write+execute)

**Common Permissions:**
- **755**: rwxr-xr-x (owner: all, others: read+execute)
- **644**: rw-r--r-- (owner: read+write, others: read)
- **600**: rw------- (owner only)
- **777**: rwxrwxrwx (all permissions - dangerous!)

**Viewing Permissions:**

ls -l filename
-rwxr-xr-x 1 user group 1234 Jan 1 12:00 filename

- First 10 characters: permissions and type
- Next: number of links, owner, group, size, date, name`,
					CodeExamples: `# View file permissions
ls -l myfile.txt
# Output: -rw-r--r-- 1 user group 1024 Jan 1 12:00 myfile.txt

# Breakdown:
# - = regular file
# rw- = owner can read/write
# r-- = group can read
# r-- = others can read

# View directory permissions
ls -ld mydir
# Output: drwxr-xr-x 2 user group 4096 Jan 1 12:00 mydir

# Change permissions (symbolic)
chmod u+x file.txt      # Add execute for owner
chmod g-w file.txt      # Remove write for group
chmod o+r file.txt      # Add read for others
chmod a+x file.txt      # Add execute for all (a = all)

# Change permissions (octal)
chmod 755 script.sh     # rwxr-xr-x
chmod 644 file.txt      # rw-r--r--
chmod 600 secret.txt    # rw------- (owner only)
chmod 777 dangerous.sh  # rwxrwxrwx (all - not recommended!)

# Recursive permission change
chmod -R 755 directory/

# Common permission patterns
chmod 755 script.sh      # Executable script
chmod 644 document.txt   # Regular document
chmod 600 .ssh/id_rsa    # Private key (owner only)
chmod 700 .ssh/          # SSH directory (owner only)`,
				},
				{
					Title: "Ownership & chown",
					Content: `File ownership determines who can change permissions and control access. Understanding ownership is crucial for system administration.

**Ownership Concepts:**
- **Owner**: User who created the file
- **Group**: Group associated with the file
- **Root**: Superuser with all permissions

**chown (Change Owner):**
- Change file/directory owner
- Usually requires root privileges
- chown user file
- chown user:group file
- chown -R: Recursive (for directories)

**chgrp (Change Group):**
- Change group ownership
- chgrp groupname file

**whoami**: Show current user
**id**: Show user and group IDs
**groups**: Show groups user belongs to

**sudo:**
- Execute command as another user (usually root)
- sudo command
- Requires password (unless configured otherwise)
- Use sparingly and carefully!`,
					CodeExamples: `# Check current user
whoami
# Output: username

# Check user and group IDs
id
# Output: uid=1000(username) gid=1000(username) groups=1000(username),27(sudo)

# Check groups
groups
# Output: username sudo

# View file ownership
ls -l file.txt
# Output: -rw-r--r-- 1 user group 1024 Jan 1 12:00 file.txt
#         owner: user, group: group

# Change owner (requires sudo)
sudo chown newuser file.txt

# Change owner and group
sudo chown newuser:newgroup file.txt

# Change group only
sudo chgrp newgroup file.txt
# or
sudo chown :newgroup file.txt

# Recursive ownership change
sudo chown -R user:group directory/

# Common ownership operations
sudo chown www-data:www-data /var/www/html  # Web server files
sudo chown user:user ~/documents            # User's documents

# Change ownership of current user's files
chown $USER:$USER file.txt
# (may not require sudo for own files)

# View all files owned by user
find / -user username 2>/dev/null

# View all files owned by group
find / -group groupname 2>/dev/null`,
				},
				{
					Title: "Links (Hard & Soft)",
					Content: `Links provide ways to reference files by multiple names or locations. Linux supports two types of links.

**Hard Links:**
- Multiple directory entries pointing to same file data
- Same inode number
- Cannot link directories
- Cannot cross filesystems
- All links are equal (no "original")
- Deleting one link doesn't delete data (until last link removed)

**Symbolic Links (Soft Links):**
- Special file containing path to target
- Different inode from target
- Can link directories
- Can cross filesystems
- Points to name, not data
- Broken if target deleted (dangling link)
- Shows with -> in ls -l

**When to Use:**
- **Hard links**: Same filesystem, want multiple names
- **Symbolic links**: Cross filesystem, directories, or want to see target clearly

**Creating Links:**
- ln target linkname: Create hard link
- ln -s target linkname: Create symbolic link`,
					CodeExamples: `# Create a file
echo "Hello World" > original.txt

# Create hard link
ln original.txt hardlink.txt

# Create symbolic link
ln -s original.txt symlink.txt

# View links
ls -li
# Output shows inode numbers:
# 1234567 -rw-r--r-- 2 user group 12 Jan 1 12:00 original.txt
# 1234567 -rw-r--r-- 2 user group 12 Jan 1 12:00 hardlink.txt
# 1234568 lrwxrwxrwx 1 user group 12 Jan 1 12:00 symlink.txt -> original.txt
# Note: hard links have same inode (1234567), symlink has different (1234568)

# Count links
ls -l original.txt
# Shows: -rw-r--r-- 2 user group ...
#         ^ number of hard links

# Edit through link (both work)
echo "Modified" >> hardlink.txt
cat original.txt  # Shows modification

echo "Modified2" >> symlink.txt
cat original.txt  # Shows modification

# Delete original
rm original.txt

# Hard link still works
cat hardlink.txt  # Still shows content

# Symbolic link is broken
cat symlink.txt   # Error: No such file or directory
ls -l symlink.txt # Shows: symlink.txt -> original.txt (red/broken)

# Create symbolic link to directory
ln -s /usr/bin bin_link
cd bin_link  # Works like /usr/bin

# Find all symbolic links
find /usr -type l

# Find broken symbolic links
find /usr -type l ! -exec test -e {} \;`,
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
