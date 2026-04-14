package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1400,
			Title:       "DevOps Fundamentals",
			Description: "Introduction to DevOps: culture, principles, practices, and the DevOps lifecycle.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "DevOps Fundamentals",
					Content: `DevOps is the union of people, process, and technology to continually provide value to customers. It’s a culture shift that removes the high-wall between Development and Operations.

**1. The Pillars of DevOps (C.A.L.M.S.)**
*   **Culture:** Shared responsibility and trust.
*   **Automation:** If you do it twice, automate it.
*   **Lean:** Small batch sizes, fast feedback loops.
*   **Measurement:** Use data (DORA metrics) to improve.
*   **Sharing:** Open communication and shared goals.

**2. The CI/CD Lifecycle**
` + "```" + `
[ Plan ] -> [ Code ] -> [ Build ] -> [ Test ] ──┐
                                               │
[ Monitor ] <- [ Operate ] <- [ Deploy ] <─────┘
` + "```" + `

**3. Key Practices**
*   **Continuous Integration (CI):** Automate builds/tests on every commit.
*   **Continuous Delivery (CD):** Code is always deployable.
*   **Infrastructure as Code (IaC):** Manage servers using code (Terraform/Ansible).
*   **Observability:** Logs, Metrics, and Traces to understand system health.

**4. DORA Metrics (Success Indicators)**
1.  **Deployment Frequency:** How often you ship code.
2.  **Lead Time for Changes:** Time from commit to production.
3.  **Change Failure Rate:** % of deployments that cause a fail.
4.  **MTTR:** Mean Time to Recovery from a failure.`,
					CodeExamples: `# 1. Automated CI (GitHub Actions)
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm install && npm test

# 2. Infrastructure as Code (Terraform)
resource "aws_instance" "web" {
  ami           = "ami-xyz"
  instance_type = "t2.micro"
}`,
				},
				{
				Title: "Historical Evolution",
				Content: `The story of DevOps is really the story of how the software industry learned — often painfully — that building software and running software are not two separate problems. They are one continuous challenge, and solving it requires tearing down walls, both organizational and technical. Understanding this history is not just trivia: it explains why DevOps practices exist and why ignoring them leads teams back into the same traps their predecessors fell into.

**1. The Waterfall Era: Sequential and Siloed**

In the early days of enterprise software (1970s through the early 2000s), most organizations followed a Waterfall methodology. Requirements were gathered up front, often in massive documents that took months to write. Then development would begin, sometimes lasting a year or more before anything was tested or deployed. Once developers were done, they would "throw the code over the wall" to a completely separate Operations team. The Ops team had never seen the code, had no input on its design, and was suddenly responsible for making it run reliably in production.

The result was predictable: deployments were terrifying events. They happened quarterly or even annually, often on weekends, and frequently failed. When something broke, Dev blamed Ops ("you deployed it wrong") and Ops blamed Dev ("you wrote buggy code"). This adversarial relationship — sometimes called "The Wall of Confusion" — was the defining dysfunction of the era. Releases were slow (often months apart), manual errors were rampant, and the friction between teams meant that fixes took weeks to reach customers.

**2. The Agile Revolution: Faster Development, Same Wall**

The Agile Manifesto of 2001 transformed how software was developed. Teams adopted Scrum, Kanban, and iterative development. Instead of year-long cycles, developers shipped working software every two weeks. Code quality improved because feedback loops were shorter — you could show a feature to stakeholders and course-correct quickly.

However, Agile solved only half the problem. Development got dramatically faster, but Operations was still working the old way. Ops teams were still manually provisioning servers, hand-editing configuration files, and doing deployments as stressful all-hands events. You had a Ferrari engine (Agile development) connected to horse-cart wheels (manual operations). The bottleneck simply moved from "we can not build fast enough" to "we can not deploy fast enough." The Wall of Confusion was still standing — it just became more painful because the pile of code waiting to be deployed grew faster.

**3. The Birth of DevOps: Tearing Down the Wall**

DevOps emerged around 2008-2009, inspired by talks like Patrick Debois' "Agile Infrastructure" and the famous "10+ Deploys Per Day" presentation by John Allspaw and Paul Hammond at Flickr. The core insight was revolutionary yet simple: the people who build the software and the people who run the software need to be the same team, or at the very least, need to collaborate deeply and share responsibility.

This was not just a tooling change — it was a cultural revolution. DevOps introduced shared ownership: developers were now responsible for their code in production (you build it, you run it). Operations engineers started writing code (Infrastructure as Code). Automation replaced manual processes. Continuous Integration and Continuous Deployment (CI/CD) pipelines meant that code could go from a developer's laptop to production in minutes, not months. The Wall of Confusion was replaced by shared dashboards, shared on-call rotations, and shared goals.

**4. The Modern DevOps Landscape**

Today, DevOps has evolved into a rich ecosystem of practices and specializations:

*   **Platform Engineering** has emerged as a discipline focused on building internal developer platforms — self-service portals where developers can provision infrastructure, deploy applications, and monitor their services without filing tickets or waiting for another team. The goal is to make doing the right thing the easy thing.
*   **GitOps** extends Infrastructure as Code by using Git as the single source of truth for both application code and infrastructure configuration. Want to change a firewall rule? Open a pull request. Want to scale up your cluster? Merge a PR. Every change is versioned, reviewed, and auditable.
*   **AIOps** applies machine learning and artificial intelligence to operations data, helping teams predict failures before they happen, automatically correlate alerts to reduce noise, and identify root causes faster than any human could by sifting through millions of log lines.
*   **Site Reliability Engineering (SRE)**, pioneered by Google, formalized many DevOps principles with concepts like error budgets, SLOs (Service Level Objectives), and toil reduction — giving teams a mathematical framework for balancing reliability with feature velocity.

The journey from Waterfall to modern DevOps is a story of the industry learning that speed and stability are not opposites — they are allies. The practices that make you faster (automation, CI/CD, IaC) are the same practices that make you more reliable. Understanding this history helps you appreciate why each DevOps practice exists and what problem it was designed to solve.`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1401,
			Title:       "Git and Version Control",
			Description: "Master Git workflows, branching strategies, and version control best practices for DevOps.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
				Title: "Git Fundamentals for DevOps",
				Content: `Version control is the absolute foundation of every DevOps practice. Without it, there is no CI/CD, no Infrastructure as Code, no GitOps, no code review — nothing. Git is not just a tool for tracking code changes; it is the collaboration backbone that makes modern software delivery possible. If DevOps were a house, Git would be the foundation slab everything else is built on.

**1. Why Version Control Matters for DevOps**

Imagine a team of five engineers all editing the same server configuration file by SSHing into a production machine. Who changed what? When? Why? If something breaks, how do you roll back? This was the reality before version control, and it was chaos. Version control solves this by providing a complete, auditable history of every change ever made. In a DevOps context, this goes far beyond just source code — teams version-control their infrastructure definitions (Terraform files), their CI/CD pipeline configurations (GitHub Actions YAML), their monitoring dashboards, their documentation, even their database migrations. The principle is simple: if it can be a text file, it should be in Git.

**2. How Git Works Conceptually: The Distributed Model**

Unlike older version control systems (CVS, Subversion) that relied on a central server, Git is distributed. Every developer has a complete copy of the entire repository history on their local machine. This design decision, made by Linus Torvalds when he created Git in 2005 to manage Linux kernel development, has profound implications. You can commit, branch, view history, and even merge entirely offline. If the central server goes down, every developer's machine is effectively a backup. Under the hood, Git stores data as a series of snapshots (not diffs). Each commit is a snapshot of your entire project at that moment, identified by a SHA-1 hash. Branches are simply lightweight pointers to specific commits, which is why creating and switching branches in Git is nearly instantaneous — it is not copying files, just moving a pointer.

The key concepts to internalize are: a **repository** is the complete history of your project; a **commit** is an immutable snapshot with a message explaining why the change was made; a **branch** is a movable pointer that lets you work on features in isolation; a **merge** combines the work from two branches; and a **remote** is a copy of the repository hosted elsewhere (like GitHub or GitLab) that enables team collaboration.

**3. Essential Commands and What They Actually Do**

The core Git workflow follows a predictable pattern: you make changes, stage them, commit them, and share them. Understanding each step matters. When you run "git add", you are moving changes into a staging area (also called the index) — think of it as preparing a package before shipping. This staging area is powerful because it lets you commit only some of your changes, keeping commits focused and meaningful. "git commit" takes everything in the staging area and creates a permanent snapshot. "git push" uploads your local commits to a remote, and "git pull" downloads and integrates remote changes. The commands "git branch" and "git merge" let you work on parallel lines of development and recombine them.

**4. Git Workflows: Choosing the Right Strategy**

Different teams need different branching strategies, and choosing the right one is a critical DevOps decision:

*   **Centralized Workflow** is the simplest — everyone commits to a single main branch. This works for very small teams or simple projects, but it creates bottlenecks and merge conflicts quickly as the team grows.
*   **Feature Branch Workflow** is the most common starting point. Every new feature or bug fix gets its own branch. Developers work in isolation, then open a pull request (PR) to merge back. This enables code review, CI checks on the branch, and clean separation of work.
*   **Git Flow** is a more structured model with dedicated branches for features, releases, hotfixes, and development. It was designed for projects with scheduled release cycles (like desktop software with versioned releases). It provides excellent control but adds complexity — many modern web teams find it heavyweight.
*   **GitHub Flow** is a deliberately simplified workflow: there is only "main" and feature branches. You branch off main, do your work, open a PR, get it reviewed, and merge. Main is always deployable. This pairs perfectly with continuous deployment — every merge to main triggers a production deploy.
*   **GitLab Flow** extends GitHub Flow by adding environment branches (staging, production). Code flows from feature branches to main, then to staging, then to production. This is ideal for teams that need approval gates between environments.

**5. Best Practices That Compound Over Time**

The discipline of committing often with meaningful messages pays enormous dividends over time. Six months from now, when a production bug leads you to "git blame" a particular line, the commit message is your only context for why that change was made. Write messages that explain the "why," not the "what" — the diff already shows what changed. Use branches for every piece of work, no matter how small, because this enables CI to validate each change independently. Always review code before merging — code review catches bugs, spreads knowledge across the team, and maintains code quality. Keep the main branch stable and deployable at all times, because a broken main branch blocks the entire team. Finally, use tags to mark releases so you can always identify exactly which commit is running in production.`,
					CodeExamples: `# Initialize repository
git init
git remote add origin https://github.com/user/repo.git

# Basic workflow
git add .
git commit -m "Add feature X"
git push origin main

# Branching
git checkout -b feature/new-feature
# Make changes
git add .
git commit -m "Implement new feature"
git push origin feature/new-feature

# Merge
git checkout main
git merge feature/new-feature
git push origin main

# Git Flow example
git checkout -b develop
git checkout -b feature/user-auth
# Work on feature
git checkout develop
git merge feature/user-auth
git checkout -b release/1.0.0
# Prepare release
git checkout main
git merge release/1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"

# GitHub Flow (simpler)
git checkout -b feature
# Make changes
git push origin feature
# Create PR, merge via GitHub
git checkout main
git pull origin main`,
				},
				{
				Title: "Advanced Git for DevOps",
				Content: `Once you have mastered the basics of Git, a set of advanced features becomes essential for professional DevOps work. These are not obscure power-user tricks — they are everyday tools that experienced engineers use to maintain clean histories, automate quality checks, and secure their delivery pipelines. Understanding these features conceptually, not just knowing the commands, is what separates a Git user from a Git practitioner.

**1. Rebase vs Merge: The Great Debate**

This is one of the most important decisions a team makes about their Git workflow. Both rebase and merge combine work from two branches, but they do it in fundamentally different ways, and the choice affects how your project history reads.

**Merge** creates a new "merge commit" that ties two branch histories together. It preserves the complete, true history of how development happened — you can see exactly when a branch was created, what commits happened in parallel, and when they were combined. The downside is that with many active branches, the history graph becomes a tangled web of merge commits that can be hard to follow.

**Rebase** takes a different approach: it replays your branch's commits on top of the target branch, as if you had started your work from the latest commit on main. The result is a perfectly linear history — clean, easy to read, easy to bisect when hunting bugs. The trade-off is that rebase rewrites commit hashes, which means you should never rebase commits that have already been pushed to a shared branch (this rewrites history that other people are building on, causing confusion and potential data loss).

The practical DevOps recommendation: use rebase for local feature branches before merging (to keep history clean), and use merge commits for the actual integration into main (to preserve the record of when features landed). Many teams enforce this with branch protection rules.

**2. Cherry-Pick: Surgical Precision**

Sometimes you need exactly one specific commit from another branch — perhaps a critical bug fix that was made on a feature branch but needs to go into a hotfix release immediately. Cherry-pick copies a single commit and applies it to your current branch. It is a surgical tool, not an everyday workflow. In DevOps, cherry-pick is most commonly used for emergency hotfixes: the fix is developed and tested on a feature branch, then cherry-picked into the release branch to avoid pulling in unrelated, untested changes.

**3. Stash: Context Switching Without Losing Work**

DevOps engineers are frequently interrupted — a production alert fires while you are in the middle of writing a Terraform module. Git stash lets you save your uncommitted changes to a temporary stack and restore a clean working directory. You can then switch branches, investigate the issue, and later pop your stashed changes to resume exactly where you left off. Think of it as a bookmark for your in-progress work.

**4. Git Hooks: Automation at the Source**

Git hooks are scripts that run automatically at specific points in the Git workflow, and they are one of the most powerful yet underused DevOps tools. They transform Git from a passive version control system into an active quality gate.

*   **pre-commit** hooks run before a commit is created. Teams use them to run linters, formatters, and security scanners. If the hook fails (returns a non-zero exit code), the commit is rejected. This means code style violations and obvious bugs never even make it into the repository. Tools like pre-commit and husky make managing these hooks easy.
*   **pre-push** hooks run before code is pushed to a remote. Teams use these for running unit tests — if tests fail, the push is blocked. This catches broken code before it reaches the CI server.
*   **post-receive** hooks run on the server after code is received. These are used to trigger deployments, send notifications, or update dashboards. In GitOps workflows, a post-receive hook might automatically update a Kubernetes deployment.

The key insight is that hooks shift quality checks left — catching problems earlier when they are cheaper and faster to fix. A bug caught by a pre-commit hook costs seconds to fix; the same bug caught in production costs hours and potentially customer trust.

**5. Signed Commits: Trust and Verification**

In a world of supply chain attacks and compromised accounts, knowing that a commit was actually authored by who it claims to be is a security concern. GPG-signed commits provide cryptographic proof of authorship. When you sign a commit, you are attaching your GPG signature, and anyone can verify that the commit was not tampered with and was genuinely created by you. Many organizations now require signed commits on protected branches, especially for infrastructure code where a malicious change could compromise an entire production environment.

**6. Submodules and Large File Storage (LFS)**

Submodules let you embed one Git repository inside another — useful when your project depends on a shared library that has its own release cycle. However, submodules add complexity and are often replaced by package managers in modern workflows. Git LFS (Large File Storage) solves a different problem: Git was designed for text files and struggles with large binary files (machine learning models, compiled artifacts, media files). LFS stores these files externally while keeping lightweight pointers in the repository, keeping your clone fast while still versioning large assets.`,
					CodeExamples: `# Rebase
git checkout feature
git rebase main
# Resolve conflicts if any
git rebase --continue

# Cherry-pick
git cherry-pick <commit-hash>

# Stash
git stash
# Work on other things
git stash pop

# Git hooks
# .git/hooks/pre-commit
#!/bin/sh
npm run lint
npm test

# Tags
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0

# .gitignore
node_modules/
.env
*.log
dist/
build/`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1402,
			Title:       "Linux for DevOps",
			Description: "Essential Linux commands and concepts for DevOps engineers.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
				Title: "Essential Linux Commands",
				Content: `The Linux command line is the primary workspace for DevOps engineers. While graphical tools and web consoles exist, the CLI remains the most powerful, scriptable, and universal interface for managing servers, debugging issues, and automating workflows. Almost every production server in the world runs Linux, and the vast majority of them have no graphical interface at all — your only way to interact with them is through a terminal. Mastering the CLI is not optional for DevOps; it is the prerequisite for everything else.

**1. The Unix Philosophy: Why the CLI Works This Way**

Linux inherits the Unix philosophy that has guided its design since the 1970s: each tool should do one thing and do it well, and tools should be composable through pipes and redirection. Instead of one massive program that does everything, Unix gives you dozens of small, focused tools that you can chain together. The "grep" command searches text. The "sort" command sorts lines. The "uniq" command removes duplicates. Individually, they are simple. But piped together — "grep ERROR app.log | sort | uniq -c | sort -rn" — they become a powerful log analysis engine built from Lego-like pieces. This composability is why the CLI has survived for 50 years while countless GUIs have come and gone.

**2. File Operations: Navigating and Manipulating the Filesystem**

At the most basic level, you need to navigate the Linux filesystem and manipulate files. The commands "ls" (list files), "cd" (change directory), and "pwd" (print working directory) are your navigation tools — they are to the filesystem what a map and compass are to a hiker. The file manipulation commands "cp" (copy), "mv" (move or rename), "rm" (remove), and "mkdir" (make directory) let you organize files. The "find" command is particularly powerful — it recursively searches directory trees based on name patterns, file size, modification time, permissions, and dozens of other criteria. In DevOps, you will use "find" constantly: finding log files older than 30 days for cleanup, locating configuration files with specific permissions, or searching for recently modified files during incident investigation.

**3. Text Processing: The DevOps Superpower**

Text processing commands are arguably the most important tools in a DevOps engineer's arsenal. In the DevOps world, almost everything is a text file — logs, configuration files, API responses, YAML manifests, CSV data. Being able to search, filter, transform, and analyze text from the command line is what lets you diagnose production issues in seconds rather than minutes.

"cat" displays file contents, "less" and "more" let you page through large files interactively, and "head" and "tail" show you the beginning or end of a file. The "tail -f" command is particularly essential — it follows a log file in real-time, showing you new lines as they are written, which is invaluable during deployments and incident response.

"grep" is the Swiss Army knife of text search — it finds lines matching a pattern (including full regular expressions) across one or many files. DevOps engineers use grep constantly: searching logs for error messages, finding configuration values, validating that a deployment changed the right files. "sed" (stream editor) performs text transformations — replacing strings, deleting lines, inserting content — and is essential for scripting configuration changes. "awk" is a complete text-processing language that excels at columnar data — extracting specific fields from structured output like "ps aux" or CSV files.

**4. System Information: Understanding What Your Server Is Doing**

Monitoring and troubleshooting require knowing what is happening on a system. "ps" shows running processes, and "ps aux" gives you the full picture — every process, who started it, how much CPU and memory it is using. "top" (and its prettier cousin "htop") give you a real-time, continuously updating view of system resource usage — essential for spotting runaway processes or memory leaks. "df" shows disk space usage (always use "df -h" for human-readable sizes), "free" shows memory usage, and "uname" tells you about the kernel and operating system version. These commands are the first things you run when SSHing into a server to investigate a problem.

**5. Permissions: Security at the Filesystem Level**

Linux's permission model is fundamental to security. Every file and directory has three sets of permissions (read, write, execute) for three categories (owner, group, others). "chmod" changes these permissions — understanding octal notation (755 means owner can read/write/execute, group and others can read/execute) is essential. "chown" changes file ownership, which matters when applications run as specific users. "sudo" lets you execute commands as the superuser (root), and understanding when and why to use sudo — and more importantly, when not to — is critical for security. In DevOps, you will constantly deal with permissions: making scripts executable, ensuring configuration files are readable only by the right services, and setting up correct ownership for application directories.`,
					CodeExamples: `# File operations
ls -la                    # List all files with details
cd /var/log              # Change directory
pwd                      # Show current directory
cp file.txt backup/      # Copy file
mv old.txt new.txt       # Rename file
rm -rf directory/        # Remove directory recursively
mkdir -p path/to/dir     # Create directory tree

# Text processing
cat file.txt             # Display file
grep "error" log.txt     # Search for "error"
sed 's/old/new/g' file    # Replace text
awk '{print $1}' file    # Print first column
head -n 20 file.txt      # First 20 lines
tail -f log.txt          # Follow log file

# System information
ps aux                   # All processes
top                      # Process monitor
df -h                    # Disk space human-readable
free -h                  # Memory usage
uname -a                 # System info

# Permissions
chmod 755 script.sh      # rwxr-xr-x
chmod +x script.sh       # Add execute permission
chown user:group file    # Change ownership
sudo command             # Run as root`,
				},
				{
				Title: "Linux System Administration",
				Content: `System administration is the operational backbone of DevOps. While modern tools like Kubernetes and Terraform abstract away much of the infrastructure, understanding what happens underneath — how services start, how packages are managed, how networking works at the OS level — is what separates an engineer who can debug production issues from one who is helpless when abstractions leak. And abstractions always leak eventually.

**1. Process Management: Controlling What Runs on Your System**

Every program running on a Linux system is a process, identified by a unique Process ID (PID). Understanding process management is essential because in DevOps, you are constantly starting, stopping, monitoring, and troubleshooting processes — whether they are application servers, database engines, or background workers.

Processes can run in the foreground (attached to your terminal) or the background. When you run a long command and need your terminal back, you can append "&" to run it in the background, or use "nohup" to ensure it continues running even after you disconnect your SSH session. The "jobs" command lists background processes in your current shell, "fg" brings one back to the foreground, and "bg" resumes a suspended process in the background.

Process signals are the inter-process communication mechanism in Linux. When you press Ctrl+C, you are sending SIGINT (signal interrupt) to the foreground process. "kill -15" (SIGTERM) politely asks a process to shut down, giving it time to clean up. "kill -9" (SIGKILL) forcefully terminates a process without giving it a chance to clean up — use this only as a last resort, because it can leave corrupted files, zombie database connections, or orphaned lock files. Understanding the difference between graceful and forceful termination is critical for operating production services safely.

**2. Package Management: Installing and Maintaining Software**

Package managers are how you install, update, and remove software on Linux systems. They handle not just the software itself but also its dependencies — the libraries and tools that software needs to function. Debian-based systems (Ubuntu, Debian) use "apt," while Red Hat-based systems (CentOS, RHEL, Fedora) use "yum" or "dnf."

The workflow is straightforward but important to do correctly: first update the package list ("apt update" fetches the latest catalog of available packages from remote repositories), then upgrade installed packages ("apt upgrade" installs newer versions), and install new software ("apt install nginx"). Always update before installing — otherwise you might install an outdated version with known security vulnerabilities. In DevOps, package management is often automated through configuration management tools like Ansible, ensuring that every server in a fleet has exactly the same software at the same version.

**3. Service Management with systemd**

systemd is the init system and service manager used by virtually all modern Linux distributions. It is the first process that starts when your system boots (PID 1), and it is responsible for starting all other services in the correct order, managing their lifecycle, and restarting them if they crash.

The "systemctl" command is your primary interface to systemd. "systemctl start nginx" starts a service, "systemctl stop nginx" stops it, "systemctl restart nginx" stops and starts it, and "systemctl status nginx" shows you whether it is running, its PID, and recent log output. Crucially, "systemctl enable nginx" configures a service to start automatically at boot — forgetting this step is a common mistake that leads to services disappearing after a server reboot. "journalctl" is the companion command for viewing systemd logs. "journalctl -u nginx -f" follows the nginx logs in real-time, and "journalctl --since '1 hour ago'" shows you logs from the past hour. These logging capabilities are invaluable for debugging deployment issues.

**4. Networking: Connecting and Diagnosing**

DevOps engineers spend a surprising amount of time debugging networking issues. "netstat" (or its modern replacement "ss") shows you active network connections — which ports are open, which processes are listening, and who is connected. This is the first tool you reach for when an application "is not working" — often the issue is simply that the process is not listening on the expected port, or a firewall is blocking traffic.

"curl" and "wget" are HTTP clients used for testing APIs, downloading files, and verifying that web services are responding correctly. "curl -v" is particularly useful because it shows you the full HTTP conversation including headers, which helps debug issues with redirects, authentication, and SSL certificates. "ping" tests basic network connectivity, though in modern cloud environments it is often blocked by security groups, so "curl" is usually more reliable for testing service reachability.

**5. Filesystem Operations: Storage and Archives**

Understanding the filesystem is essential for capacity management and data handling. "mount" attaches filesystems to the directory tree — when you add a new disk to a server, you need to format it and mount it. "df -h" shows you how much space is available on each mounted filesystem (and is often the first command you run when an application reports "disk full"). "du -sh /path" shows you how much space a specific directory is using, helping you identify what is consuming all the disk space. The "tar" command creates and extracts archive files — it is the standard way to package and transfer collections of files in Linux, and you will encounter tar archives constantly when deploying applications, creating backups, and downloading software.`,
					CodeExamples: `# Process management
nohup command &          # Run in background
jobs                     # List background jobs
fg %1                    # Bring job to foreground
kill -9 PID              # Force kill process
killall process_name     # Kill all instances

# Package management (Debian/Ubuntu)
sudo apt update          # Update package list
sudo apt upgrade         # Upgrade packages
sudo apt install nginx   # Install package
sudo apt remove nginx    # Remove package

# Package management (RHEL/CentOS)
sudo yum update          # Update packages
sudo yum install nginx   # Install package
sudo yum remove nginx    # Remove package

# Service management
sudo systemctl start nginx    # Start service
sudo systemctl stop nginx     # Stop service
sudo systemctl restart nginx  # Restart service
sudo systemctl status nginx   # Check status
sudo systemctl enable nginx   # Enable on boot
journalctl -u nginx -f        # View logs

# Network
netstat -tulpn           # Network connections
ss -tulpn                # Modern netstat
curl https://api.com     # HTTP request
wget https://file.zip     # Download file
ping google.com          # Test connectivity

# File system
df -h                    # Disk space
du -sh /path             # Directory size
tar -czf archive.tar.gz dir/  # Create archive
tar -xzf archive.tar.gz       # Extract archive`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1403,
			Title:       "Shell Scripting",
			Description: "Master Bash scripting: automation, error handling, and DevOps automation scripts.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
				Title: "Bash Scripting Basics",
				Content: `Shell scripting is arguably the most important practical skill in the DevOps toolkit. It is the glue that holds automation together — the language you use to wire up tools, automate repetitive tasks, build deployment scripts, create health checks, and respond to incidents. Every DevOps engineer, regardless of what other programming languages they know, writes shell scripts regularly. Learning Bash is not about becoming a programmer; it is about giving yourself a superpower that turns hours of manual work into seconds of automated execution.

**1. Script Fundamentals: The Building Blocks**

Every Bash script begins with a shebang line (#!/bin/bash) which tells the operating system which interpreter to use. This seemingly trivial line prevents a class of subtle bugs where scripts behave differently because they were accidentally interpreted by a different shell (sh, dash, zsh).

Variables in Bash are assigned without spaces around the equals sign (NAME="value" — note that "NAME = value" would fail). This is one of many Bash quirks that trip up newcomers. To use a variable, prefix it with a dollar sign ($NAME). Arguments are accessed through positional parameters: $1 is the first argument, $2 is the second, $@ represents all arguments, and $# gives you the count. These make your scripts flexible — instead of hardcoding server names or file paths, you pass them as arguments.

Exit codes are the communication protocol of the Unix world. Every command returns an exit code when it finishes: 0 means success, and any non-zero value means failure. This convention is what makes it possible to chain commands with "&&" (run the next command only if the previous one succeeded) and "||" (run the next command only if the previous one failed). Understanding exit codes is essential because they are the foundation of error handling in shell scripts, CI/CD pipelines, and process management.

**2. Control Flow: Making Decisions and Repeating Actions**

Bash provides the control flow constructs you would expect from any programming language, though with its own distinctive syntax. "if/else" blocks let you make decisions based on conditions — checking if a file exists, if a service is running, or if a command succeeded. "for" loops iterate over lists (files in a directory, servers in an inventory, lines in a file), while "while" loops continue until a condition changes. "case" statements handle multiple conditions cleanly, and they are particularly useful for parsing command-line arguments in your scripts.

Functions let you organize your scripts into reusable, named blocks. Experienced scripters structure their scripts with functions for each logical operation — a "deploy" function, a "rollback" function, a "healthcheck" function — making the script readable and maintainable. Functions can accept arguments (just like scripts) and return exit codes, enabling clean composition.

**3. The Sacred Incantation: set -euo pipefail**

If you learn one thing about Bash scripting, make it this: always start your scripts with "set -euo pipefail". This single line transforms Bash from a dangerously permissive language into a reasonably safe one. Here is what each flag does:

*   **set -e** (exit on error): Without this, Bash happily continues executing after a command fails. Imagine a deployment script where "docker pull" fails but the script continues and "docker run" uses the old, broken image. With "set -e", the script stops immediately when any command returns a non-zero exit code.
*   **set -u** (error on undefined variables): Without this, referencing an undefined variable silently expands to an empty string. The command "rm -rf $DIERCTORY/" (note the typo) would become "rm -rf /" and destroy your entire filesystem. With "set -u", Bash raises an error when you reference a variable that has not been defined.
*   **set -o pipefail**: By default, the exit code of a pipeline (commands connected with |) is the exit code of the last command only. So "failing_command | grep something" would report success if grep succeeds, hiding the failure of the first command. "pipefail" makes the pipeline return the exit code of the last command that failed.

Together, these flags catch the vast majority of scripting bugs at the earliest possible moment. Experienced DevOps engineers consider a script without this line to be a ticking time bomb.

**4. Error Handling: The Trap Mechanism**

The "trap" command lets you register functions that run when specific signals are received. The most common use is cleanup: "trap 'rm -f /tmp/lockfile' EXIT" ensures that a temporary lock file is deleted when the script exits, whether it exits normally, due to an error, or because of a Ctrl+C interrupt. This is Bash's equivalent of a "finally" block in other languages, and it is essential for writing robust scripts that do not leave behind mess when they fail.

**5. Real-World Automation Patterns**

In practice, DevOps shell scripts follow recognizable patterns. Deployment scripts typically validate prerequisites (is Docker installed? do I have the right permissions?), perform the deployment steps, run health checks to verify success, and either report success or trigger a rollback. Health check scripts poll an endpoint in a loop, waiting for a service to become ready. Backup scripts use date-stamped filenames, rotate old backups, and report to monitoring systems. Log rotation scripts archive old log files and compress them to save space. Understanding these patterns — and seeing them in the code examples — is more valuable than memorizing syntax, because the patterns transfer to any automation task you encounter.`,
					CodeExamples: `#!/bin/bash
# Basic script
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Variables
NAME="DevOps"
VERSION=1.0

# Functions
function greet() {
    local name=$1
    echo "Hello, $name"
}

greet "World"

# Conditional
if [ "$1" == "deploy" ]; then
    echo "Deploying..."
elif [ "$1" == "test" ]; then
    echo "Testing..."
else
    echo "Usage: $0 [deploy|test]"
    exit 1
fi

# Loop
for file in *.log; do
    echo "Processing $file"
done

# While loop
while true; do
    echo "Running..."
    sleep 5
done

# Error handling
set -e
trap 'echo "Error on line $LINENO"' ERR

# Check command exists
if ! command -v docker &> /dev/null; then
    echo "Docker not found"
    exit 1
fi`,
				},
				{
				Title: "Advanced Shell Scripting",
				Content: `Once you have mastered the basics of Bash scripting, a set of advanced features unlocks the ability to write sophisticated, production-grade automation. These features — arrays, parameter expansion, process substitution, and robust patterns like retry logic and lock files — are what transform simple scripts into reliable tools that teams depend on daily. Understanding them deeply, with practical DevOps context, is what separates a "quick hack" script from one you would trust to run unattended at 3 AM.

**1. Arrays: Managing Collections of Data**

Bash supports indexed arrays (ordered lists) and associative arrays (key-value maps), and both are essential for real-world scripting. Consider a deployment script that needs to deploy to multiple servers: instead of hardcoding server names, you store them in an array and iterate. Arrays make scripts flexible and data-driven.

Indexed arrays are created with parentheses: SERVERS=("web1" "web2" "web3"). You access elements with ${SERVERS[0]}, get all elements with ${SERVERS[@]}, and get the length with ${#SERVERS[@]}. Associative arrays (declared with "declare -A") let you create dictionaries: CONFIG["host"]="localhost", CONFIG["port"]="8080". In DevOps, you might use an associative array to map environment names to their respective configuration files, or service names to their ports.

**2. Parameter Expansion: String Surgery Without External Tools**

Bash's parameter expansion features let you manipulate strings without calling external commands like "sed" or "awk," making your scripts faster and more portable. These look cryptic at first but become second nature quickly.

*   ${var%pattern} removes the shortest match of "pattern" from the end of the variable. So if FILE="backup.tar.gz", then ${FILE%.*} gives you "backup.tar" — useful for stripping file extensions.
*   ${var%%pattern} removes the longest match from the end. ${FILE%%.*} gives you "backup" — stripping everything after the first dot.
*   ${var#pattern} and ${var##pattern} do the same from the beginning.
*   ${var:-default} provides a default value if the variable is unset or empty — essential for making scripts configurable with sensible defaults (PORT=${PORT:-8080}).
*   ${var:?error message} causes the script to exit with an error if the variable is unset — a powerful validation tool.

In DevOps, you use parameter expansion constantly: extracting version numbers from filenames, constructing paths dynamically, providing defaults for configuration values, and validating required environment variables.

**3. Here Documents and Here Strings: Multi-line Input**

Here documents (heredocs) let you embed multi-line text directly in your script, which is invaluable for generating configuration files, sending multi-line input to commands, or creating templates. The syntax "cat <<EOF ... EOF" writes everything between the markers to stdout. If you use "<<'EOF'" (with quotes), variables are not expanded — useful for generating scripts or configuration that should contain literal dollar signs.

In DevOps, heredocs are used to generate Nginx configuration files, write systemd unit files, create Kubernetes manifests, or compose email notifications — any situation where you need to produce structured multi-line output from a script.

**4. Process Substitution and Command Substitution**

Command substitution — $(command) — captures the output of a command into a variable. DATE=$(date +%Y-%m-%d) stores today's date. COMMIT=$(git rev-parse HEAD) stores the current commit hash. You will use this in virtually every DevOps script for generating dynamic values.

Process substitution — <(command) — is more advanced and less well-known. It presents the output of a command as if it were a file, which lets you use commands that expect file arguments with dynamic data. For example, "diff <(ssh server1 cat /etc/config) <(ssh server2 cat /etc/config)" compares configuration files between two remote servers without creating temporary files. This is tremendously useful for comparing outputs, feeding data to commands that only accept files, and composing complex data pipelines.

**5. Production Patterns: Retry Logic, Lock Files, and Signal Handling**

Production scripts need to handle the messy reality of distributed systems where things fail intermittently. A retry function wraps a command and re-executes it with exponential backoff if it fails — essential for operations like downloading from flaky repositories, connecting to services that are still starting up, or pushing to APIs with rate limits. The pattern is: try the command, if it fails, sleep for an increasing amount of time, and try again, up to a maximum number of attempts.

Lock files prevent multiple instances of the same script from running simultaneously. If your backup script takes 2 hours and cron triggers it every hour, you need a lock file to prevent overlapping runs that could corrupt data. The pattern is: check if the lock file exists (if so, exit), create the lock file, register a trap to delete it on exit, and proceed.

Signal handling with "trap" goes beyond simple cleanup. You can trap SIGTERM (sent by process managers during shutdown) to perform graceful cleanup, SIGHUP (traditionally used to reload configuration) to re-read config files without restarting, and SIGUSR1/SIGUSR2 for custom behaviors like dumping status information. Well-written DevOps scripts handle all relevant signals to behave predictably in automated environments.

**6. Logging: Making Scripts Observable**

Production scripts must produce clear, structured logs. A logging function that prepends timestamps and log levels — log "INFO" "Deployment started" producing "[2024-01-15 14:30:22] INFO: Deployment started" — transforms debugging from archaeology into reading. Use "tee -a" to write to both stdout and a log file simultaneously, and direct stderr to a separate error log for easy filtering. Good logging is the difference between a script that works and a script that works and can be debugged when it does not.`,
					CodeExamples: `#!/bin/bash
set -euo pipefail

# Arrays
FILES=("file1.txt" "file2.txt" "file3.txt")
echo "${FILES[@]}"           # All elements
echo "${#FILES[@]}"          # Length

# Associative arrays
declare -A CONFIG
CONFIG["host"]="localhost"
CONFIG["port"]="8080"
echo "${CONFIG[host]}"

# Here document
cat <<EOF > config.txt
host=localhost
port=8080
EOF

# Command substitution
DATE=$(date +%Y-%m-%d)
echo "Today is $DATE"

# Parameter expansion
FILE="backup.tar.gz"
echo "${FILE%.*}"           # backup.tar
echo "${FILE%%.*}"          # backup
echo "${FILE#*.}"            # tar.gz

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a app.log
}

log "Starting deployment"

# Retry logic
retry() {
    local max_attempts=$1
    shift
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        fi
        echo "Attempt $attempt failed, retrying..."
        attempt=$((attempt + 1))
        sleep 2
    done
    return 1
}

retry 3 curl -f https://api.example.com

# Lock file
LOCK_FILE="/tmp/deploy.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "Deployment already running"
    exit 1
fi
trap "rm -f $LOCK_FILE" EXIT
touch "$LOCK_FILE"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1404,
			Title:       "Docker Fundamentals",
			Description: "Learn Docker: containers, images, Dockerfile, and containerization basics.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
				Title: "Docker Introduction",
				Content: `Docker fundamentally changed how software is deployed, and understanding why requires going beyond "it is like a lightweight VM." Docker is a containerization platform that packages an application together with everything it needs to run — its code, runtime, libraries, system tools, and configuration — into a single, portable unit called a container. This solves one of the most persistent problems in software: the "it works on my machine" syndrome.

**1. The Shipping Container Analogy**

Before standardized shipping containers were invented in the 1950s, loading cargo onto ships was a nightmare. Every item — barrels, crates, bags, machinery — had a different shape and size. Loading a ship required specialized labor, took days, and cargo was frequently damaged or lost. Then Malcolm McLean invented the standardized shipping container: a uniform metal box that could be packed at a factory, transported by truck, loaded onto a ship by crane, and delivered to its destination without anyone ever touching the contents. It did not matter whether the container held electronics or bananas — the infrastructure for moving it was identical.

Docker containers are the same concept applied to software. It does not matter whether your container holds a Node.js web app, a Python ML model, or a Java enterprise application — the infrastructure for building, shipping, and running it is identical. Your laptop, your CI server, your staging environment, and your production cluster all run containers the same way. This standardization is what made Docker revolutionary.

**2. How Docker Works Under the Hood: Namespaces and Cgroups**

Docker is not a virtual machine — it does not emulate hardware or run a separate operating system. Instead, it uses two Linux kernel features to create isolated environments that share the host's kernel:

*   **Namespaces** provide isolation. Each container gets its own view of the system: its own process tree (PID namespace), its own network stack (network namespace), its own filesystem (mount namespace), its own hostname (UTS namespace), and its own user IDs (user namespace). A process inside the container cannot see processes in other containers or on the host. It thinks it is running on its own dedicated machine.
*   **Cgroups** (control groups) provide resource limits. They control how much CPU, memory, disk I/O, and network bandwidth a container can use. This prevents a runaway container from consuming all the host's resources and crashing everything else — a critical property for running multiple services on the same machine.

Together, namespaces and cgroups give you isolation and resource control without the overhead of running a full virtual machine. This is why containers start in milliseconds (no OS to boot) and use a fraction of the resources a VM would require.

**3. Key Concepts Every DevOps Engineer Must Know**

*   **Image**: A read-only template that contains your application and all its dependencies. Think of it as a class in object-oriented programming — it defines what something is. Images are built in layers, with each layer representing a filesystem change. This layered architecture enables efficient storage and sharing — if ten images all use the same Ubuntu base layer, that layer is stored only once.
*   **Container**: A running instance of an image. Think of it as an object created from a class. You can run multiple containers from the same image, each with its own isolated state. Containers are ephemeral by design — when they stop, any changes made inside them are lost (unless you use volumes).
*   **Dockerfile**: A text file containing the instructions for building an image. It is the recipe — starting from a base image, it specifies what to install, what files to copy, what ports to expose, and what command to run when the container starts. Dockerfiles are version-controlled, making your build process reproducible and auditable.
*   **Registry**: A repository for storing and distributing images. Docker Hub is the public registry (like npm for containers), but organizations typically run private registries for proprietary software. Your CI/CD pipeline builds images and pushes them to a registry; your deployment process pulls images from the registry and runs them.
*   **Volume**: Docker's mechanism for persistent data. Since containers are ephemeral, anything written inside a container disappears when it stops. Volumes are directories that exist outside the container's filesystem and persist across container restarts — essential for databases, uploaded files, and logs.
*   **Network**: Docker's networking system lets containers communicate with each other and with the outside world. By default, Docker creates an isolated network for each set of containers, and you can configure which ports are exposed to the host.

**4. Docker vs Virtual Machines: Understanding the Trade-offs**

Containers and VMs solve similar problems but at different levels. VMs virtualize hardware — each VM runs its own complete operating system on emulated hardware, managed by a hypervisor. This provides strong isolation (a kernel bug in one VM cannot affect another) but at the cost of significant resource overhead (each VM needs its own OS, consuming gigabytes of memory and taking minutes to boot). Containers virtualize the operating system — they share the host kernel but are isolated through namespaces. This makes them dramatically lighter (megabytes instead of gigabytes), faster to start (milliseconds instead of minutes), and more efficient (you can run many more containers than VMs on the same hardware). The trade-off is weaker isolation — since containers share a kernel, a kernel vulnerability could theoretically allow a container escape. In practice, most organizations use containers for application workloads and VMs for strong multi-tenant isolation.

**5. Why Docker Revolutionized Deployment**

Before Docker, deploying an application meant installing its dependencies on a server, configuring the runtime, setting environment variables, and hoping that nothing conflicted with the other applications on the same machine. Different applications might need different versions of the same library, leading to "dependency hell." Docker eliminated this entirely: each application runs in its own container with its own dependencies, completely isolated from everything else. This made microservices architectically practical — you could run dozens of services on the same host without conflicts. It made CI/CD pipelines reliable — the same image tested in CI is the exact same image deployed to production. And it made scaling trivial — need more capacity? Just run more containers.`,
					CodeExamples: `# Docker commands
docker --version          # Check version
docker info               # System information
docker ps                 # Running containers
docker ps -a              # All containers
docker images             # List images

# Pull and run image
docker pull nginx:latest
docker run -d -p 80:80 nginx

# Build image
docker build -t myapp:1.0 .

# Container management
docker start container_id
docker stop container_id
docker restart container_id
docker rm container_id    # Remove container
docker rmi image_id       # Remove image

# View logs
docker logs container_id
docker logs -f container_id  # Follow logs

# Execute commands in container
docker exec -it container_id /bin/bash
docker exec container_id ls /app

# Inspect
docker inspect container_id
docker stats              # Resource usage`,
				},
				{
				Title: "Dockerfile and Image Building",
				Content: `A Dockerfile is the recipe for building a Docker image, and writing an effective Dockerfile is one of the most important skills in containerized DevOps. A poorly written Dockerfile produces bloated images that are slow to build, slow to push and pull, contain security vulnerabilities, and waste storage. A well-written Dockerfile produces minimal, secure, fast-building images that are a joy to work with. The difference comes down to understanding how Docker's layer caching works and applying a set of proven best practices.

**1. Dockerfile Instructions: What Each One Does and Why**

Each instruction in a Dockerfile creates a layer in the image, and understanding layers is key to writing efficient Dockerfiles.

*   **FROM** specifies the base image — the starting point for your image. Everything else builds on top of it. Always use a specific tag (FROM node:18-alpine, not FROM node) because "latest" is a moving target that can break your builds without warning. Alpine-based images are much smaller than Debian-based ones (5MB vs 100MB+ for the base OS), which means faster pulls and a smaller attack surface.
*   **RUN** executes commands during the build process — installing packages, compiling code, creating directories. Each RUN instruction creates a new layer. Combine related commands with "&&" to reduce layers: "RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*" is one layer instead of three, and it removes the apt cache in the same layer so it never gets stored.
*   **COPY** copies files from your local machine into the image. **ADD** does the same but also handles URLs and auto-extracts tar archives — however, COPY is preferred because its behavior is more transparent and predictable.
*   **WORKDIR** sets the working directory for subsequent instructions. Always use WORKDIR instead of "RUN cd /app" because WORKDIR persists across instructions while cd does not.
*   **ENV** sets environment variables that are available both during build and at runtime. Use it for configuration that should be baked into the image (like NODE_ENV=production).
*   **EXPOSE** documents which ports the container listens on. It does not actually publish the port — it is metadata that helps other engineers and tools understand how the container is meant to be used.
*   **CMD** specifies the default command to run when the container starts. It can be overridden at runtime. **ENTRYPOINT** is similar but is not overridden — it defines the container's primary purpose. A common pattern is to use ENTRYPOINT for the application binary and CMD for default arguments.

**2. Layer Caching: Why Instruction Order Matters**

Docker builds images layer by layer, from top to bottom, and it caches each layer. If nothing has changed in a layer's inputs (the instruction and any files it references), Docker reuses the cached version instead of rebuilding it. This caching is immensely powerful — a build that takes 5 minutes the first time might take 10 seconds on subsequent runs. But caching is invalidated from the point of the first change downward: if layer 3 changes, layers 4, 5, 6, and all subsequent layers must be rebuilt even if they have not changed.

This is why instruction ordering matters critically. Put things that change infrequently at the top (installing system packages, setting up the base environment) and things that change frequently at the bottom (copying your application code). The most important optimization in any Node.js Dockerfile is separating "COPY package*.json ./" and "RUN npm install" from "COPY . ." — this way, your dependencies are only reinstalled when package.json changes, not every time you edit a source file. This single pattern can reduce rebuild times from minutes to seconds.

**3. Multi-stage Builds: Smaller, Safer Images**

Multi-stage builds are one of Docker's most powerful features, and they solve two problems simultaneously: image size and security. The idea is simple — you use one stage to build your application (with all the build tools, compilers, and dev dependencies) and a separate stage for the final runtime image (with only the compiled output and runtime dependencies).

Consider a Go application: the build stage needs the Go compiler, build tools, and source code (hundreds of megabytes), but the runtime only needs the compiled binary (a few megabytes). With a multi-stage build, the final image contains only the binary — not the compiler, not the source code, not the build cache. This makes the image dramatically smaller (faster to push, pull, and start) and dramatically more secure (fewer installed tools means fewer potential vulnerabilities and less for an attacker to work with if they compromise the container).

The same principle applies to Node.js (build stage compiles TypeScript and bundles assets, runtime stage runs the Node.js output), Java (build stage compiles with Maven, runtime stage runs the JAR), and virtually any compiled language.

**4. Security Best Practices**

Never run your application as root inside a container. If an attacker compromises your application, running as root gives them full control of the container (and potentially the host, depending on the security configuration). Always create a non-root user and switch to it with the USER instruction.

Use a .dockerignore file to exclude files that should never be in your image — node_modules, .git directories, .env files containing secrets, log files, and local development artifacts. This reduces image size and prevents accidental exposure of sensitive data.

Add HEALTHCHECK instructions so Docker (and orchestrators like Kubernetes) can verify that your application is actually working, not just that the process is running. A common pattern is to check an HTTP health endpoint: "HEALTHCHECK --interval=30s CMD curl -f http://localhost:3000/health || exit 1". This enables automatic restart of unhealthy containers and reliable load balancing.`,
					CodeExamples: `# Basic Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

USER node

CMD ["node", "server.js"]

# Multi-stage build
# Stage 1: Build
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Runtime
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./
EXPOSE 3000
CMD ["node", "dist/server.js"]

# .dockerignore
node_modules/
.git/
.env
*.log
dist/

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:3000/health || exit 1`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1405,
			Title:       "Docker Compose",
			Description: "Orchestrate multi-container applications with Docker Compose.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
				Title: "Docker Compose Fundamentals",
				Content: `Docker Compose takes the concept of containerization and extends it to entire application stacks. While Docker alone runs individual containers, most real-world applications consist of multiple services that need to work together — a web server, a database, a cache, a message queue, perhaps a background worker. Docker Compose lets you define all of these services, their configuration, their networking, and their storage in a single YAML file, and then start or stop the entire stack with one command. It is Infrastructure as Code applied to your local development environment, and it is one of the most immediately useful tools in the DevOps ecosystem.

**1. The Problem Compose Solves: Defining Your Infrastructure as Code**

Without Compose, setting up a development environment that includes a web app, a PostgreSQL database, and a Redis cache would require running three separate docker commands with the right flags, ports, volumes, and environment variables. If a new developer joins the team, they would need to know all these commands and run them in the right order. If the configuration changes, everyone would need to update their commands manually. This is error-prone, undocumented, and violates the core DevOps principle that infrastructure should be defined as code.

Docker Compose solves this by letting you describe your entire application stack declaratively in a docker-compose.yml file. This file is version-controlled alongside your application code, so it evolves with your project. Every developer on the team runs "docker-compose up" and gets an identical environment. There is no "it works on my machine" because everyone's machine runs the exact same configuration. New developers can go from cloning the repo to running the full application in minutes instead of hours.

**2. Key Concepts: Services, Networks, and Volumes**

*   **Services** are the heart of Compose. Each service definition describes one container — what image to use (or how to build it), what ports to expose, what environment variables to set, what volumes to mount, and what other services it depends on. A service called "web" might build from your local Dockerfile, expose port 3000, and connect to a database service.
*   **Networks** in Compose handle service-to-service communication. By default, Compose creates a single network for your entire application, and every service is reachable by its service name. This means your web application can connect to PostgreSQL using the hostname "db" (the service name) instead of an IP address — Compose's built-in DNS handles the resolution automatically. This mimics how service discovery works in production orchestrators like Kubernetes, making the transition from development to production smoother.
*   **Volumes** provide persistent storage that survives container restarts. When you run "docker-compose down" and then "docker-compose up" again, data stored in volumes (like your database data) is preserved. Without volumes, your database would start empty every time — useful for testing but terrible for development. Compose distinguishes between named volumes (managed by Docker, ideal for databases) and bind mounts (mapping a host directory into the container, ideal for live code reloading during development).
*   **Dependencies** control startup order. "depends_on: db" tells Compose that the web service should start after the database service. However, it is important to understand that "depends_on" only waits for the container to start, not for the service inside to be ready. A PostgreSQL container starts in milliseconds, but the database engine inside it takes seconds to initialize. Production-grade Compose files use health checks with "condition: service_healthy" to wait for actual readiness.

**3. Common Use Cases in the DevOps Workflow**

Compose shines in several scenarios. For **local development**, it gives every developer a consistent, isolated environment with all dependencies running locally — no need for shared development databases or cloud resources. For **integration testing**, your CI/CD pipeline can spin up the entire application stack with Compose, run tests against it, and tear it down — ensuring tests run against realistic infrastructure. For **microservices development**, Compose lets you run the specific services you are working on locally while connecting to mocked or minimal versions of the rest. For **demos and prototyping**, you can share a Compose file that lets anyone run your entire application with a single command.

**4. Why Compose Matters for DevOps Culture**

Docker Compose embodies several core DevOps principles. It treats infrastructure configuration as code that is version-controlled, reviewed, and tested. It eliminates environmental drift between developers. It reduces onboarding time from days to minutes. It makes it possible to reproduce bugs locally by running the exact same stack. And it provides a stepping stone to production orchestration — the concepts of services, networks, volumes, and health checks in Compose map directly to their equivalents in Kubernetes, making the learning curve gentler. For many teams, Compose is where they first experience the power of declarative infrastructure, and that experience shapes how they think about operations at every scale.`,
					CodeExamples: `# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      - db
    volumes:
      - ./app:/app
    networks:
      - app-network

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - app-network

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    networks:
      - app-network

volumes:
  postgres-data:

networks:
  app-network:
    driver: bridge

# Commands
docker-compose up              # Start services
docker-compose up -d           # Start in background
docker-compose down            # Stop and remove
docker-compose ps              # List services
docker-compose logs            # View logs
docker-compose logs -f web     # Follow logs
docker-compose exec web bash   # Execute command
docker-compose build           # Build images
docker-compose restart web     # Restart service`,
				},
				{
					Title: "Advanced Docker Compose",
					Content: `Advanced Docker Compose features and production patterns.

**Advanced Features:**
- **Override Files**: Use multiple compose files (docker-compose.override.yml)
- **Environment Files**: Externalize configuration (.env files)
- **Profiles**: Conditional service activation
- **Health Checks**: Service health monitoring
- **Resource Limits**: CPU and memory constraints
- **Restart Policies**: Container restart behavior
- **Secrets Management**: Secure credential handling

**Production Patterns:**
- Separate dev/prod configurations
- Use environment-specific override files
- Implement health checks for all services
- Set resource limits
- Use secrets for sensitive data
- Configure logging drivers
- Use external networks and volumes

**Best Practices:**
- Version control compose files
- Use .env files for environment variables
- Separate concerns with profiles
- Implement proper health checks
- Set appropriate resource limits
- Use named volumes for persistence
- Document service dependencies

**Common Pitfalls:**
- Hardcoding values in compose files
- Not using health checks
- Missing resource limits
- Exposing unnecessary ports
- Not using secrets for sensitive data
- Mixing dev and prod configurations`,
					CodeExamples: `# docker-compose.yml (base)
version: '3.8'

services:
  web:
    build: .
    ports:
      - "${WEB_PORT:-3000}:3000"
    environment:
      - NODE_ENV=${NODE_ENV:-development}
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  db:
    image: postgres:14
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1G
    secrets:
      - db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres-data:
    driver: local

# docker-compose.prod.yml (production override)
version: '3.8'

services:
  web:
    environment:
      - NODE_ENV=production
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        max_attempts: 3
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://logs.example.com:514"

# docker-compose.dev.yml (development override)
version: '3.8'

services:
  web:
    volumes:
      - ./src:/app/src
      - ./node_modules:/app/node_modules
    environment:
      - DEBUG=true
    command: npm run dev

  db:
    ports:
      - "5432:5432"

# Using profiles
services:
  web:
    # ... base config
  worker:
    profiles:
      - worker
    # ... worker config
  redis:
    profiles:
      - cache
    # ... redis config

# Run with profiles
docker-compose --profile worker --profile cache up

# .env file example
WEB_PORT=3000
NODE_ENV=production
DB_NAME=myapp
DB_USER=admin
DB_PASSWORD=secure_password

# Commands with override files
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
docker-compose --env-file .env.production up`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1406,
			Title:       "CI/CD Basics",
			Description: "Introduction to Continuous Integration and Continuous Deployment: concepts, pipelines, and automation.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
				Title: "CI/CD Concepts",
				Content: `Continuous Integration and Continuous Deployment (CI/CD) is the beating heart of DevOps. It is the practice of automating every step between a developer writing code and that code running in production. Before CI/CD, software releases were stressful, manual, error-prone events that happened infrequently. With CI/CD, releases become routine, automated, and boring — and in operations, boring is exactly what you want.

**1. The Philosophy: Why Automation Changes Everything**

Consider the cost of a manual deployment. A developer finishes a feature and packages it up. An operations engineer reviews a deployment checklist, SSHs into servers, runs commands in the right order, updates configuration files, restarts services, and verifies everything is working. This process might take hours. It is different every time because humans make mistakes. It happens infrequently because it is painful, which means each deployment contains weeks or months of changes, making it riskier and harder to debug when something goes wrong. If something does go wrong, rolling back is another manual process.

CI/CD replaces all of this with automated pipelines. Code is tested automatically on every commit. Builds are created automatically. Deployments happen automatically. The entire process takes minutes instead of hours, runs identically every time, and happens dozens of times per day instead of once per quarter. The paradoxical insight of CI/CD is that deploying more often is safer than deploying less often — small, frequent changes are easier to test, easier to understand, and easier to roll back than massive, infrequent releases.

**2. Continuous Integration (CI): Catching Problems Early**

Continuous Integration is the practice of frequently merging code changes into a shared repository and automatically verifying each change with builds and tests. The key word is "continuously" — not once a week or once a sprint, but every commit.

When a developer pushes code, the CI system automatically checks it out, installs dependencies, compiles the code, runs the test suite, performs code quality checks (linting, static analysis), scans for security vulnerabilities, and reports the results. If any step fails, the team is notified immediately, and the broken commit can be fixed before it compounds with other changes.

The value of CI compounds over time. Without it, bugs hide in the codebase for days or weeks, interacting with other changes in unpredictable ways. Finding and fixing them becomes an archaeological expedition. With CI, a bug is caught within minutes of being introduced, when the developer still has the full context in their head. The cost of fixing a bug caught by CI is orders of magnitude lower than one caught in production.

**3. Continuous Delivery vs Continuous Deployment**

These terms sound similar but differ in one crucial way. **Continuous Delivery** means that your code is always in a deployable state — every change that passes the automated pipeline could be released to production at any time. However, the actual deployment to production requires a manual approval step (a human clicks a button). This is appropriate for organizations that need regulatory compliance, change management boards, or coordination with marketing launches.

**Continuous Deployment** goes one step further: every change that passes the automated pipeline is automatically deployed to production with no human intervention. This is the gold standard of DevOps — it requires extremely high confidence in your automated tests and monitoring, but it provides the fastest possible feedback loop. Companies like Netflix, Amazon, and Etsy deploy hundreds or thousands of times per day using continuous deployment.

**4. Anatomy of a CI/CD Pipeline**

A pipeline is a series of stages that code passes through on its way to production. Each stage acts as a quality gate — if a stage fails, the pipeline stops and the change does not proceed:

1.  **Source Stage**: Triggered by a code change — a push to a branch, a pull request, or a merge to main. The pipeline checks out the code and begins.
2.  **Build Stage**: Compiles the code, resolves dependencies, and creates build artifacts (compiled binaries, Docker images, packaged archives). If the code does not compile, the pipeline fails fast.
3.  **Test Stage**: Runs automated tests — unit tests, integration tests, end-to-end tests, security scans, and code quality checks. This is where most bugs are caught.
4.  **Deploy Stage**: Deploys to target environments, typically progressing through staging (for final validation) and then production. Deployment strategies like blue-green or canary minimize risk.
5.  **Monitor Stage**: After deployment, automated checks verify that the application is healthy. Metrics like error rates, latency, and resource usage are compared to baselines. If something looks wrong, an automatic rollback can be triggered.

**5. Common CI/CD Tools and Their Trade-offs**

*   **Jenkins** is the veteran — self-hosted, infinitely customizable through plugins, and used by thousands of organizations. It offers maximum control but requires significant maintenance and infrastructure.
*   **GitHub Actions** is GitHub-native, making it the natural choice for teams using GitHub. Workflows are defined in YAML files alongside your code, and the marketplace provides thousands of pre-built actions for common tasks.
*   **GitLab CI** is deeply integrated with GitLab, providing a seamless experience from code to deployment. Its pipeline visualization and environment management are particularly strong.
*   **CircleCI** and **Travis CI** are cloud-hosted services that minimize infrastructure overhead — you define your pipeline and they run it. They are ideal for teams that do not want to manage CI infrastructure.

The best tool depends on your team's ecosystem, scale, and needs, but the principles are identical across all of them.`,
					CodeExamples: `# GitHub Actions example
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test
      
      - name: Build
        run: npm run build
      
      - name: Deploy
        if: github.ref == 'refs/heads/main'
        run: npm run deploy

# GitLab CI example
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - npm install
    - npm run build
  artifacts:
    paths:
      - dist/

test:
  stage: test
  script:
    - npm test

deploy:
  stage: deploy
  script:
    - npm run deploy
  only:
    - main`,
				},
				{
				Title: "Building CI/CD Pipelines",
				Content: `Knowing the theory of CI/CD is one thing; designing and building effective pipelines is another. A well-designed pipeline is fast, reliable, secure, and provides clear feedback to developers. A poorly designed one is slow, flaky, and becomes a bottleneck that the team works around rather than with. The principles in this lesson are the difference between a pipeline that accelerates your team and one that frustrates them.

**1. The Fail-Fast Principle: Catching Problems at the Cheapest Point**

The most important principle in pipeline design is "fail fast" — run the cheapest, fastest checks first, and progressively move to slower, more expensive checks. If your code has a syntax error, there is no point running a 20-minute integration test suite to discover it. A well-designed pipeline might look like: lint check (5 seconds) -> unit tests (30 seconds) -> build (1 minute) -> integration tests (5 minutes) -> security scan (2 minutes) -> deploy to staging (2 minutes) -> end-to-end tests (10 minutes) -> deploy to production.

If the lint check fails, the developer gets feedback in 5 seconds and the pipeline stops — no compute time is wasted on tests that would also fail. This ordering is not just about speed; it is about developer experience. A developer who gets feedback in seconds stays in flow. A developer who waits 20 minutes for feedback context-switches to something else, and the context switch has a real cognitive cost.

**2. Parallel Execution: Doing More Without Waiting Longer**

Many pipeline stages can run simultaneously. Unit tests, linting, security scanning, and type checking do not depend on each other, so they should run in parallel. Testing across multiple versions (Node 16, 18, and 20) or multiple operating systems (Ubuntu, Alpine) are naturally parallel via matrix strategies. The goal is to minimize wall-clock time — the time a developer waits for results — even if total compute time increases.

**3. Caching: The Biggest Performance Win**

Dependency installation is often the slowest part of a pipeline. Downloading and installing node_modules, Python packages, or Maven dependencies from scratch on every build wastes minutes of time and significant bandwidth. CI/CD caching solves this by storing these dependencies between runs and only reinstalling when the lock file changes.

The pattern is universal across CI systems: compute a hash of the lock file (package-lock.json, poetry.lock, go.sum), use that hash as a cache key, and restore the cached dependencies if the key matches. This single optimization can reduce pipeline times by 50-80%. Understanding how caching works — including cache invalidation (changing the lock file busts the cache) and cache scope (branch-specific vs shared caches) — is essential for pipeline performance.

**4. Artifacts: Passing Work Between Stages**

Artifacts are the outputs of pipeline stages — compiled binaries, Docker images, test reports, code coverage data. The build stage produces an artifact (a Docker image, a JAR file), stores it in a registry or artifact storage, and subsequent stages (test, deploy) consume it. This ensures that exactly the same artifact that was tested is the one deployed to production — no rebuilding, no risk of inconsistency.

Versioning artifacts is critical. Common strategies include using the Git commit SHA (guaranteeing uniqueness and traceability), semantic version tags for releases, and build numbers for internal tracking. The ability to trace a production artifact back to the exact commit that produced it is invaluable for debugging.

**5. Deployment Strategies: Minimizing Risk**

How you deploy to production is as important as what you deploy. Several strategies exist, each with different trade-offs:

*   **Rolling deployment** gradually replaces old instances with new ones. It is simple and requires minimal extra resources, but a bad deployment affects real users as new instances come online.
*   **Blue-green deployment** runs two identical environments: "blue" (current production) and "green" (new version). Traffic is switched from blue to green all at once. If something goes wrong, you switch back to blue instantly. The trade-off is that you need double the infrastructure during deployment.
*   **Canary deployment** routes a small percentage of traffic (say 5%) to the new version first. You monitor error rates and latency, and if everything looks good, gradually increase the percentage. If something is wrong, you roll back having affected only a tiny fraction of users. This is the safest strategy but requires sophisticated traffic routing and monitoring.

**6. Security in the Pipeline**

Modern pipelines include security scanning as a first-class stage: static application security testing (SAST) analyzes source code for vulnerabilities, dependency scanning checks for known vulnerabilities in your libraries, container image scanning checks for vulnerabilities in your Docker images, and secrets detection ensures that API keys and passwords have not been accidentally committed. Shifting security left — running these checks early and automatically — catches vulnerabilities when they are cheapest to fix, rather than discovering them in a penetration test or, worse, in production.`,
					CodeExamples: `# Advanced GitHub Actions
name: Full CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:

env:
  NODE_VERSION: '18'
  REGISTRY: ghcr.io

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
      - uses: actions/cache@v3
        with:
          path: ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
      - run: npm ci
      - run: npm test
      - run: npm run lint

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t ${{ env.REGISTRY }}/app:${{ github.sha }} .
      - name: Push to registry
        run: docker push ${{ env.REGISTRY }}/app:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/app app=${{ env.REGISTRY }}/app:${{ github.sha }}
          kubectl rollout status deployment/app`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1407,
			Title:       "Build Tools and Package Managers",
			Description: "Master build tools and package managers: npm, Maven, Gradle, and build automation.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
				Title: "Package Managers Overview",
				Content: `Package managers are one of those tools that modern developers take for granted, but they solve an incredibly difficult problem: dependency management. Every non-trivial application depends on external libraries — a web framework, a database driver, a logging library, a JSON parser. Those libraries in turn depend on other libraries, creating a tree of transitive dependencies that can grow into hundreds or thousands of packages. Managing this manually would be impossible. Package managers automate the process of finding, downloading, installing, updating, and removing these dependencies, and in the DevOps context, they are critical to achieving reproducible, reliable builds.

**1. The Dependency Management Problem**

Imagine building a house and needing to source every individual component yourself — finding the right bolts, ensuring the lumber meets specifications, verifying that the electrical wire gauge matches the breaker capacity. Package managers are like a general contractor who handles all of this for you. You specify what you need (the web framework, the database driver), and the package manager figures out what else is required (the framework's dependencies, their dependencies, and so on), resolves any conflicts (library A needs version 2 of something, but library B needs version 3), and installs everything in the right order.

Different programming ecosystems have developed their own package managers, each with different strengths:
*   **npm, yarn, and pnpm** serve the JavaScript/Node.js ecosystem. npm is the default, yarn introduced deterministic installs, and pnpm uses hard links for dramatically faster and more space-efficient installations.
*   **Maven and Gradle** serve the Java ecosystem. Maven uses XML configuration and convention-over-configuration, while Gradle uses a Groovy or Kotlin DSL and offers more flexibility with incremental builds.
*   **pip and Poetry** serve Python. pip is the basic installer, while Poetry adds dependency resolution, virtual environment management, and lock file support — addressing many of pip's limitations.
*   **Cargo** (Rust), **Go modules** (Go), **NuGet** (.NET), and **Composer** (PHP) each serve their respective ecosystems with similar core concepts.

**2. Lock Files: The Key to Reproducible Builds**

Lock files are arguably the most important concept in dependency management for DevOps, and they are frequently misunderstood. Your package manifest (package.json, requirements.txt, go.mod) specifies what dependencies you want and what version ranges are acceptable. A lock file (package-lock.json, poetry.lock, go.sum) records the exact version of every dependency — including transitive dependencies — that was resolved during installation.

Why does this matter? Without a lock file, running "npm install" on two different machines at two different times might install different versions of dependencies (because a new patch version was published in between), leading to subtly different behavior. This violates the fundamental DevOps principle that builds should be reproducible. With a lock file committed to version control, "npm ci" installs exactly the same versions every time, on every machine, in every environment. The lock file is what makes "it works on my machine" become "it works everywhere."

Always commit your lock files to version control. Always use the "clean install" variant of your package manager in CI/CD (npm ci instead of npm install, pip install --require-hashes, etc.) to ensure the lock file is respected exactly.

**3. Semantic Versioning: A Communication Protocol**

Semantic versioning (SemVer) is the convention that most package ecosystems follow: versions have three numbers — MAJOR.MINOR.PATCH (e.g., 4.18.2). Each number communicates intent:
*   **PATCH** (4.18.2 -> 4.18.3): Bug fixes that should not break anything. Safe to update.
*   **MINOR** (4.18.0 -> 4.19.0): New features added, but backwards compatible. Usually safe to update.
*   **MAJOR** (4.0.0 -> 5.0.0): Breaking changes that may require code modifications. Must be updated carefully.

Version ranges in package manifests (like ^4.18.0 in npm, meaning "any version >= 4.18.0 and < 5.0.0") use SemVer to express how much flexibility the package manager has. Understanding SemVer helps you make informed decisions about how tightly to pin your dependencies. Pinning too loosely means unexpected breaking changes; pinning too tightly means missing important security patches.

**4. Security: Dependencies as Attack Surface**

Your dependencies are part of your attack surface. Supply chain attacks — where malicious code is injected into popular packages — have become a significant threat. Package managers provide tools to combat this: "npm audit" and its equivalents scan your dependency tree for known vulnerabilities. Automated tools like Dependabot and Renovate create pull requests to update vulnerable dependencies. In a DevOps pipeline, dependency scanning should be an automated stage that blocks deployments when critical vulnerabilities are found.

**5. Build Tools: From Source to Artifact**

Package managers often double as build tools, providing a standardized way to compile source code, run tests, package artifacts, and publish releases. The "scripts" section of a package.json, Maven lifecycle phases, and Gradle tasks all provide hooks for automating the build process. In DevOps, the build tool configuration is as important as the application code — it defines how your software is assembled, tested, and prepared for deployment. Treat it with the same care you would give to production code: version control it, review changes to it, and test it in CI.`,
					CodeExamples: `# npm (Node.js)
npm init                 # Initialize project
npm install express      # Install package
npm install -D jest      # Install dev dependency
npm install --save-exact # Exact version
npm update               # Update packages
npm audit                # Security audit
npm audit fix            # Fix vulnerabilities

# package.json
{
    "name": "myapp",
    "version": "1.0.0",
    "scripts": {
        "start": "node server.js",
        "test": "jest",
        "build": "webpack"
    },
    "dependencies": {
        "express": "^4.18.0"
    },
    "devDependencies": {
        "jest": "^29.0.0"
    }
}

# Maven (Java)
# pom.xml
<project>
    <dependencies>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-core</artifactId>
            <version>5.3.21</version>
        </dependency>
    </dependencies>
</project>

mvn clean install        # Build and install
mvn test                 # Run tests
mvn package              # Create JAR

# Gradle (Java/Kotlin)
# build.gradle
plugins {
    id 'java'
}

dependencies {
    implementation 'org.springframework:spring-core:5.3.21'
    testImplementation 'junit:junit:4.13.2'
}

gradle build             # Build
gradle test              # Test
gradle publish           # Publish`,
				},
				{
				Title: "Advanced Build Automation",
				Content: `As projects grow in size and complexity, naive build processes become a bottleneck that drags down the entire development team. A build that takes 20 minutes means developers wait 20 minutes for CI feedback, 20 minutes to verify a fix, 20 minutes before a deployment can proceed. Advanced build automation techniques — incremental builds, caching strategies, parallel execution, and artifact management — can reduce this to minutes or even seconds, directly accelerating team velocity and deployment frequency.

**1. Incremental Builds: Only Rebuild What Changed**

The most effective way to speed up a build is to avoid doing unnecessary work. An incremental build system tracks what inputs changed since the last build and only rebuilds the components affected by those changes. If you modified one file in a project with 1,000 source files, an incremental build recompiles only that file (and anything that depends on it), not all 1,000.

Modern build tools implement this at various levels. The Go compiler inherently does incremental compilation. Gradle tracks inputs and outputs of every task and skips tasks whose inputs have not changed. Webpack's watch mode recompiles only modified modules. Docker's layer caching (discussed in the Docker lessons) is another form of incremental building — unchanged layers are reused from cache.

The key to making incremental builds work is declaring dependencies correctly. The build tool needs to know the complete graph of what depends on what. If this graph is incorrect or incomplete, incremental builds can produce stale outputs — one of the most confusing categories of bugs to debug.

**2. Parallel Builds: Leveraging Multiple Cores**

Modern CI servers have multiple CPU cores, and sequential builds waste most of them. Parallel build execution runs independent build tasks simultaneously — if module A and module B do not depend on each other, they can be compiled at the same time. Maven's "-T" flag enables parallel module building ("mvn -T 4" uses 4 threads). Gradle parallelizes by default when configured ("org.gradle.parallel=true"). In CI/CD, matrix strategies run tests across multiple versions or platforms simultaneously.

The speedup from parallelization depends on the dependency graph. A project with many independent modules benefits enormously. A project where everything depends on everything else in a long chain benefits less. Understanding your project's dependency graph helps you optimize for parallelism — sometimes restructuring modules to reduce inter-dependencies pays for itself in build time savings alone.

**3. Caching Strategies: Persistence Across Builds**

Caching operates at multiple levels, and understanding each level is essential for optimization:

*   **Dependency caching** stores downloaded packages (node_modules, Maven's .m2 repository, pip's package cache) between builds. Since dependencies change infrequently compared to source code, this avoids re-downloading gigabytes of packages on every build. In CI, this is typically implemented with cache keys based on lock file hashes.
*   **Build output caching** stores compiled artifacts and intermediate results. Gradle's build cache, Bazel's remote cache, and Turbo's remote caching all store the outputs of build tasks keyed by their inputs. If the inputs have not changed (even across different branches or machines), the cached output is reused. Remote caching takes this further by sharing the cache across the entire team and CI infrastructure — a build task that one developer already computed is reused by everyone else.
*   **Docker layer caching** in CI deserves special attention. Without layer caching, every Docker build in CI starts from scratch, downloading base images and reinstalling dependencies. Configuring your CI to cache Docker layers (using registry-based caching or volume-mounted caches) can reduce Docker build times from minutes to seconds.

**4. Artifact Management: From Build to Deployment**

Build artifacts — the outputs of your build process — are the bridge between CI and CD. A Docker image, a compiled binary, a JAR file, or a deployment package are all artifacts. Managing them properly is essential for reliable deployments.

Artifacts should be immutable and versioned. Once built, an artifact is never modified — if you need a change, you build a new artifact with a new version. Artifact repositories (Artifactory, Nexus, GitHub Packages, Docker registries) store these artifacts and provide a reliable source for deployment pipelines. The version should be traceable back to the exact source code that produced it — typically through the Git commit SHA or a release tag.

The golden rule is "build once, deploy everywhere." The same artifact that was tested in CI is promoted to staging, and the same artifact from staging is deployed to production. Never rebuild for a different environment — change only the configuration (via environment variables or configuration files), not the artifact. This eliminates an entire class of "but it worked in staging" bugs caused by subtle differences in build environments.

**5. Build Monitoring and Optimization**

You cannot improve what you do not measure. Track your build times over time — total duration, time per stage, cache hit rates, and flaky test frequency. Most CI systems provide analytics for this. When build times creep up (and they always do as projects grow), the data tells you where to focus optimization efforts. A common pattern is to set a build time budget (e.g., "CI must complete in under 10 minutes") and treat violations as bugs to be fixed, just like performance regressions in your application.`,
					CodeExamples: `# npm build script with caching
# .npmrc
cache=/tmp/.npm
package-lock=true

# package.json scripts
{
  "scripts": {
    "prebuild": "npm run lint",
    "build": "webpack --mode production",
    "postbuild": "npm run test:build",
    "build:ci": "npm ci && npm run build",
    "build:cache": "npm ci --cache /tmp/.npm && npm run build"
  }
}

# Maven with caching and parallel builds
# .mvn/maven.config
-T 4
-Dmaven.build.cache.enabled=true

# pom.xml with build profiles
<profiles>
  <profile>
    <id>ci</id>
    <properties>
      <skipTests>false</skipTests>
    </properties>
  </profile>
  <profile>
    <id>fast</id>
    <properties>
      <maven.test.skip>true</maven.test.skip>
    </properties>
  </profile>
</profiles>

# Build with profile
mvn clean install -Pci -T 4

# Gradle build with caching
# gradle.properties
org.gradle.caching=true
org.gradle.parallel=true
org.gradle.configureondemand=true

# build.gradle with build cache
buildCache {
    local {
        enabled = true
        directory = new File(rootDir, '.gradle/build-cache')
    }
}

# Multi-stage Docker build (build automation)
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
CMD ["node", "dist/server.js"]

# Build script example (Bash)
#!/bin/bash
set -e

BUILD_DIR="build"
ARTIFACT_DIR="artifacts"
VERSION=$(git describe --tags --always)

echo "Building version $VERSION"

# Clean previous builds
rm -rf $BUILD_DIR $ARTIFACT_DIR
mkdir -p $BUILD_DIR $ARTIFACT_DIR

# Install dependencies
npm ci

# Run tests
npm test

# Build
npm run build

# Package artifacts
tar -czf $ARTIFACT_DIR/app-$VERSION.tar.gz -C $BUILD_DIR .

# Upload to artifact repository
# aws s3 cp $ARTIFACT_DIR/app-$VERSION.tar.gz s3://artifacts/myapp/

echo "Build complete: $ARTIFACT_DIR/app-$VERSION.tar.gz"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1408,
			Title:       "Configuration Management Basics",
			Description: "Introduction to configuration management: Ansible, Puppet, Chef basics.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
				Title: "Configuration Management Concepts",
				Content: `Configuration management is the discipline of automating the setup, maintenance, and enforcement of server and application configurations across your entire infrastructure. It is the answer to a question that every operations team eventually faces: "How do we manage hundreds or thousands of servers without going insane?" The manual approach — SSHing into each server, running commands, editing configuration files — does not scale. It is slow, error-prone, impossible to audit, and guarantees that servers drift out of sync over time, creating a snowflake environment where every machine is slightly different in ways no one fully understands.

**1. The Problem of Infrastructure Drift**

Infrastructure drift is what happens when servers that are supposed to be identical gradually become different. One server got a hotfix that was never applied to the others. A developer SSH'd into a production machine and changed a config file "temporarily" but never changed it back. An OS update was applied to half the fleet but interrupted on the other half. Over time, you end up with an environment where no two servers are quite the same, and nobody is entirely sure what state any given server is in.

This drift is not just untidy — it is dangerous. Bugs appear on some servers but not others, making them nearly impossible to reproduce. Deployments succeed on some machines and fail on others. Security patches are inconsistently applied, leaving holes in your defenses. Configuration management solves drift by continuously enforcing the desired state: you declare what each server should look like, and the tool ensures it matches, correcting any drift automatically.

**2. Idempotency: The Most Important Property**

Idempotency means that applying a configuration multiple times produces the same result as applying it once. If you say "ensure nginx is installed," and nginx is already installed, the tool does nothing. If nginx is not installed, the tool installs it. This is a foundational property that makes configuration management safe and reliable.

Without idempotency, running a configuration script twice might install duplicate packages, append duplicate lines to configuration files, or restart services unnecessarily. Idempotent operations are safe to re-run at any time — you can run your configuration management tool every 30 minutes as a cron job, and it will correct any drift without causing side effects. This transforms configuration from a one-time setup task into a continuous enforcement mechanism.

**3. Declarative vs Imperative: Describing the "What" Not the "How"**

Configuration management tools generally follow a declarative model: you describe the desired state of your infrastructure ("nginx should be installed, running, and configured with this file"), and the tool figures out how to achieve that state. This is fundamentally different from imperative scripting ("apt install nginx, then cp this file, then systemctl start nginx"), where you specify the exact steps.

The declarative approach has several advantages. It is self-documenting — the configuration file IS the documentation of your infrastructure. It handles edge cases automatically — the tool knows how to install a package whether it is absent, outdated, or already present. And it is naturally idempotent — declaring a desired state is inherently a repeatable operation.

**4. Push vs Pull Models**

Configuration management tools deliver configurations in one of two ways:

*   **Push model** (used by Ansible): A central machine connects to target servers (typically via SSH) and pushes the desired configuration to them. The administrator initiates the process. This is simpler to set up and reason about — you run a command, and things happen. But it requires the central machine to have network access to all targets.
*   **Pull model** (used by Puppet, Chef): An agent runs on each target server and periodically contacts a central server to pull its configuration. This is more autonomous — even if the central server is unreachable, the agent continues enforcing the last known good configuration. It also scales better because the work is distributed, but it adds complexity (agents need to be installed and maintained on every server).

Each model has trade-offs, and the best choice depends on your infrastructure size, network topology, and operational style.

**5. Popular Tools and Their Philosophies**

*   **Ansible** is agentless and uses YAML-based playbooks. It connects via SSH and requires no software on target machines (only Python). Its simplicity and low barrier to entry have made it the most widely adopted tool for teams getting started with configuration management.
*   **Puppet** is agent-based with its own declarative language. It excels at large-scale infrastructure where continuous enforcement is critical. Puppet agents run every 30 minutes by default, constantly correcting drift.
*   **Chef** is agent-based and uses Ruby as its configuration language. It appeals to teams who want the full power of a programming language for complex configuration logic.
*   **SaltStack** (now part of VMware) is agent-based, uses Python, and is known for its speed — it can execute commands across thousands of servers in seconds using a message bus rather than SSH.

**6. Configuration Management in the Modern DevOps Stack**

Configuration management complements other DevOps tools. Terraform provisions infrastructure (creates VMs, networks, load balancers), and then Ansible configures it (installs software, deploys applications, manages configurations). In a Kubernetes world, much of what traditional configuration management did is handled by container images and Kubernetes manifests, but Ansible and its peers remain essential for managing the underlying infrastructure — the Kubernetes nodes themselves, bare-metal servers, network devices, and any systems that do not run in containers.`,
					CodeExamples: `# Ansible playbook
# playbook.yml
- hosts: webservers
  become: yes
  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
    
    - name: Start nginx
      systemd:
        name: nginx
        state: started
        enabled: yes
    
    - name: Copy config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: restart nginx
  
  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted

# Run playbook
ansible-playbook playbook.yml

# Ansible inventory
# inventory.ini
[webservers]
web1 ansible_host=192.168.1.10
web2 ansible_host=192.168.1.11

[databases]
db1 ansible_host=192.168.1.20`,
				},
				{
				Title: "Ansible Basics",
				Content: `Ansible has become the most popular configuration management tool in the DevOps world, largely because of its elegant simplicity: it requires no agent software on managed servers, uses human-readable YAML for its configuration language, and connects over standard SSH — a protocol that is already available on virtually every Linux server. This low barrier to entry means you can go from zero to managing a fleet of servers in hours, not days or weeks.

**1. The Agentless Architecture: Why It Matters**

Most configuration management tools (Puppet, Chef, SaltStack) require you to install and maintain agent software on every server you manage. This creates a chicken-and-egg problem: how do you manage the agent before the agent is installed? It also means more software to maintain, more security surface, and more things that can break.

Ansible takes a fundamentally different approach. It connects to target servers over SSH (or WinRM for Windows), executes the necessary commands, and disconnects. Nothing is installed on the managed servers except Python (which is pre-installed on virtually all Linux distributions). This means you can start using Ansible against existing servers immediately — no preparation required. The control machine (your laptop or a CI server) is the only place Ansible needs to be installed.

Under the hood, when you run an Ansible playbook, Ansible generates small Python scripts, copies them to the target servers via SSH, executes them, collects the results, and removes the scripts. This "push and execute" model is simple, transparent, and easy to debug — if something goes wrong, you can SSH into the server yourself and see exactly what happened.

**2. Playbooks as Runbooks: Codifying Operational Knowledge**

A playbook in Ansible is a YAML file that describes a set of tasks to execute on a set of hosts. Think of it as a codified runbook — the kind of document that operations teams traditionally write to describe how to set up a server, deploy an application, or respond to an incident, except that instead of a human reading and executing the steps, Ansible executes them automatically.

This codification is incredibly valuable. Traditional runbooks are text documents that go stale quickly — the actual procedure drifts from what is documented, and the next person who follows the runbook discovers (usually at the worst possible moment) that step 5 no longer works. An Ansible playbook is a living, executable document. If it runs successfully, the procedure works. If the infrastructure changes, the playbook is updated to match. It is documentation that cannot lie, because it is the automation itself.

Playbooks consist of plays (which target a group of hosts) containing tasks (individual actions). Each task uses a module — a pre-built unit of work like installing a package, copying a file, managing a service, or creating a user. Ansible ships with thousands of modules covering virtually every common operation, and the community contributes thousands more through Ansible Galaxy.

**3. Core Components and How They Fit Together**

*   **Inventory** defines the servers Ansible manages. At its simplest, it is a text file listing hostnames grouped by function: [webservers], [databases], [monitoring]. It can also be dynamic — pulling server lists from AWS, Azure, or your CMDB — so your inventory stays up to date automatically as servers are created and destroyed in the cloud.
*   **Tasks** are the individual actions: install this package, copy this file, start this service. Each task is declarative and idempotent — it describes the desired state, and Ansible figures out what (if anything) needs to change to achieve it.
*   **Handlers** are special tasks that only run when triggered by a notification from another task. The classic example: a task copies a new nginx configuration file and notifies the "restart nginx" handler. If the config file did not change (because it was already up to date), the handler does not run — nginx is not restarted unnecessarily. This pattern prevents unnecessary service disruptions.
*   **Variables** make playbooks flexible. Instead of hardcoding "install nginx on port 80," you write "install nginx on port {{ nginx_port }}." Variables can come from the inventory, from variable files, from the command line, or from facts (system information that Ansible automatically gathers from each host, like the OS version, IP address, and available memory).
*   **Templates** use the Jinja2 templating language to generate configuration files dynamically. A template file might contain "listen {{ nginx_port }};" and Ansible fills in the actual value based on the host's variables. This lets you use one template for all your servers while customizing the output for each one.
*   **Roles** are the primary mechanism for organizing and reusing Ansible code. A role packages related tasks, handlers, templates, variables, and files into a self-contained unit (like "webserver," "database," or "monitoring-agent"). Roles can be shared across projects and published to Ansible Galaxy for the community to use.

**4. From Ad-hoc Commands to Structured Automation**

Ansible provides a spectrum from simple to sophisticated. **Ad-hoc commands** let you execute a single task across your fleet without writing a playbook: "ansible webservers -m apt -a 'name=nginx state=present'" installs nginx on all web servers with one command. This is great for quick tasks and exploration.

When tasks become more complex or need to be repeatable, you write **playbooks**. When playbooks grow large or are needed across multiple projects, you refactor them into **roles**. This progression — from ad-hoc to playbook to role — mirrors how automation typically evolves in organizations, starting with quick wins and gradually building a library of reusable automation.

**5. Best Practices for Reliable Automation**

Always test playbooks with "--check" (dry run mode) before applying them to production — this shows you what would change without actually making any changes. Use variables instead of hardcoded values so playbooks work across environments. Structure your playbooks with roles from the start, even if you only have one — the organizational discipline pays off quickly as your automation grows. Always use idempotent modules (apt, systemd, file) instead of raw commands (command, shell) when possible, because idempotent modules can be re-run safely while raw commands might have unintended side effects on subsequent runs. And treat your Ansible code with the same rigor as your application code: version-control it, review changes through pull requests, and test it in a staging environment before applying it to production.`,
					CodeExamples: `# Ad-hoc commands
ansible all -m ping
ansible webservers -m apt -a "name=nginx state=present" --become
ansible all -a "uptime"
ansible webservers -m copy -a "src=/local/file dest=/remote/file"

# Basic playbook
# playbook.yml
- name: Setup web server
  hosts: webservers
  become: yes
  vars:
    nginx_port: 80
    app_user: www-data
  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
    
    - name: Install nginx
      apt:
        name: nginx
        state: present
    
    - name: Create app directory
      file:
        path: /var/www/myapp
        state: directory
        owner: "{{ app_user }}"
        group: "{{ app_user }}"
        mode: '0755'
    
    - name: Copy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/myapp
        owner: root
        group: root
        mode: '0644'
      notify: restart nginx
    
    - name: Enable site
      file:
        src: /etc/nginx/sites-available/myapp
        dest: /etc/nginx/sites-enabled/myapp
        state: link
      notify: restart nginx
    
    - name: Start and enable nginx
      systemd:
        name: nginx
        state: started
        enabled: yes
  
  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted

# Template file (nginx.conf.j2)
server {
    listen {{ nginx_port }};
    server_name {{ server_name }};
    root /var/www/myapp;
    
    location / {
        try_files $uri $uri/ =404;
    }
}

# Inventory with variables
# inventory.ini
[webservers]
web1 ansible_host=192.168.1.10 nginx_port=80
web2 ansible_host=192.168.1.11 nginx_port=8080

[webservers:vars]
server_name=example.com

# Using variables
- name: Display variable
  debug:
    msg: "Server port is {{ nginx_port }}"

# Conditional tasks
- name: Install package on Debian
  apt:
    name: nginx
  when: ansible_os_family == "Debian"

- name: Install package on RedHat
  yum:
    name: nginx
  when: ansible_os_family == "RedHat"

# Loops
- name: Install packages
  apt:
    name: "{{ item }}"
  loop:
    - nginx
    - mysql-server
    - redis-server

# Error handling
- name: Try to start service
  systemd:
    name: myservice
    state: started
  ignore_errors: yes
  register: result

- name: Check if service started
  debug:
    msg: "Service started successfully"
  when: result.failed == false

# Run playbook
ansible-playbook playbook.yml -i inventory.ini
ansible-playbook playbook.yml --check  # Dry run
ansible-playbook playbook.yml --limit webservers`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1409,
			Title:       "Monitoring Fundamentals",
			Description: "Introduction to monitoring: logs, metrics, alerts, and observability basics.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
				Title: "Monitoring Basics",
				Content: `Monitoring is the eyes and ears of your production systems. Without it, you are flying blind — you do not know if your application is healthy, if users are experiencing errors, if a server is running out of disk space, or if a deployment made things better or worse. Monitoring is not an afterthought or a nice-to-have; it is a fundamental requirement for operating reliable systems. The saying in DevOps is: "if you cannot measure it, you cannot manage it." Effective monitoring turns invisible problems into visible ones, enabling teams to respond before users even notice.

**1. The Three Pillars of Observability**

Modern monitoring is built on three complementary data types, often called the "three pillars of observability." Each provides a different lens into your system, and together they give you a complete picture:

*   **Logs** are timestamped records of discrete events — "user 123 logged in at 14:30:22," "database query failed with timeout after 30s," "deployment of version 2.1.0 started." Logs are the most detailed and human-readable form of observability data. They tell you exactly what happened and when. The challenge with logs is volume — a busy application can produce millions of log lines per hour, so effective log management requires centralization, indexing, and search capabilities.
*   **Metrics** are numerical measurements over time — CPU usage at 5-second intervals, request count per minute, average response time over the last hour. Metrics are compact (a single number per data point), efficient to store, and ideal for dashboards, alerting, and trend analysis. They tell you the "what" at a high level: the error rate is 5%, latency is 200ms, memory usage is 85%.
*   **Traces** track the journey of a single request as it flows through your system — from the load balancer, to the web server, to the application code, to the database, to a cache, and back. In a microservices architecture where a single user request might touch ten different services, traces are essential for understanding where time is spent and where failures occur. Without traces, debugging a slow request in a distributed system is like finding a needle in a haystack.

Each pillar answers different questions. Metrics tell you "something is wrong" (error rate spiked). Logs tell you "what specifically went wrong" (stack trace of the error). Traces tell you "where in the system it went wrong" (the database service took 5 seconds to respond). Together, they provide complete observability.

**2. The Golden Signals: What to Monitor**

Google's Site Reliability Engineering book popularized four "golden signals" that every service should monitor. These are not the only metrics that matter, but they are the most important starting point:

*   **Latency**: How long requests take. Monitor both the average and the percentiles (p50, p95, p99). The average can hide problems — if 99% of requests take 50ms but 1% take 10 seconds, the average looks fine but 1% of your users are having a terrible experience. The p99 reveals this.
*   **Traffic**: How much demand is being placed on your system — requests per second, concurrent connections, messages processed per minute. Understanding your traffic patterns helps you plan capacity and identify anomalies (a sudden spike might indicate a DDoS attack or a viral feature).
*   **Errors**: The rate of failed requests. This includes explicit errors (HTTP 500 responses) and implicit errors (HTTP 200 responses that contain wrong data, or responses that are too slow to be useful). A rising error rate is often the first sign of a problem.
*   **Saturation**: How "full" your service is — CPU usage, memory usage, disk I/O, network bandwidth, queue depth. Saturation metrics are leading indicators: if your disk is 95% full, you will have a problem soon even if everything looks fine right now.

**3. The Monitoring Stack: From Collection to Action**

A monitoring system has four layers, each serving a distinct purpose:

*   **Collection** is how data gets from your applications and infrastructure into the monitoring system. Agents (like the Prometheus Node Exporter or Datadog Agent) run on servers and collect system metrics. Application code exports custom metrics through libraries. Log shippers (Filebeat, Fluentd) forward log files to central storage. The key principle is that collection should be lightweight — the monitoring system should not itself become a performance problem.
*   **Storage** holds the collected data. Metrics are stored in time-series databases (Prometheus, InfluxDB, Graphite) optimized for numerical data indexed by time. Logs are stored in search-optimized systems (Elasticsearch, Loki) that support full-text search across millions of records. Traces are stored in specialized backends (Jaeger, Zipkin) that understand the parent-child relationships between spans.
*   **Visualization** turns raw data into understandable dashboards. Grafana has become the de facto standard, connecting to virtually any data source and providing rich, customizable dashboards. Good dashboards are designed for specific audiences and purposes — an executive dashboard showing business KPIs looks very different from an on-call engineer's incident response dashboard.
*   **Alerting** notifies the right people when something goes wrong. Alert rules define conditions ("if error rate exceeds 5% for more than 5 minutes, alert the on-call engineer"), and alert managers (Prometheus Alertmanager, PagerDuty, OpsGenie) handle routing, deduplication, escalation, and notification delivery.

**4. Alerting Philosophy: Meaningful Alerts, Not Noise**

The biggest mistake teams make with monitoring is creating too many alerts. If your on-call engineer receives 50 alerts per day, they learn to ignore them — and when a real incident happens, it gets lost in the noise. This is called "alert fatigue," and it is the most common monitoring anti-pattern.

Effective alerts follow these principles: alert on symptoms, not causes (alert on "high error rate," not "CPU is high" — high CPU might be perfectly normal during peak traffic); set thresholds based on actual impact (users are affected, SLOs are threatened); ensure every alert is actionable (the recipient can do something about it); and include context (which service, what threshold was breached, a link to the relevant dashboard). Every alert should either wake someone up because something is genuinely broken, or it should not exist.`,
					CodeExamples: `# Application logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("User logged in", extra={"user_id": 123})

# Prometheus metrics
from prometheus_client import Counter, Histogram

requests_total = Counter('http_requests_total', 'Total requests')
request_duration = Histogram('http_request_duration_seconds', 'Request duration')

@request_duration.time()
def handle_request():
    requests_total.inc()
    # Handle request

# Health check endpoint
@app.route('/health')
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# System metrics (Linux)
top                    # CPU and memory
iostat -x 1           # Disk I/O
netstat -i            # Network interfaces
free -h                # Memory usage`,
				},
				{
				Title: "Logging and Metrics Collection",
				Content: `Collecting the right data in the right format is the foundation of effective observability. The difference between an organization that resolves incidents in minutes and one that takes hours often comes down to how well their logging and metrics are structured, centralized, and queryable. This lesson covers the practical strategies and tools that make the difference.

**1. Structured vs Unstructured Logging: A Critical Choice**

Traditional logging produces unstructured text: "2024-01-17 10:00:00 ERROR: Failed to connect to database host db-primary on port 5432 for user app_service". This is human-readable, but try writing a query to find all database connection errors for a specific user in the last hour — you would need complex regex patterns that break when the log format changes even slightly.

Structured logging produces machine-parseable records, typically in JSON: {"timestamp":"2024-01-17T10:00:00Z", "level":"ERROR", "message":"Failed to connect to database", "host":"db-primary", "port":5432, "user":"app_service", "error_type":"connection_timeout"}. Every piece of information is a named field that can be filtered, aggregated, and queried reliably. Want all errors for user "app_service"? Query on the "user" field. Want to count connection timeouts per database host? Aggregate on "host" where "error_type" is "connection_timeout."

The investment in structured logging pays for itself the first time you have a production incident. Instead of grep-ing through gigabytes of unstructured text and hoping your regex matches, you run precise queries against indexed fields and get answers in seconds. Every modern logging platform (Elasticsearch, Loki, Splunk, Datadog Logs) is optimized for structured data.

**2. Log Levels: A Severity Contract**

Log levels are a severity classification system that serves two purposes: they let you filter logs by importance, and they let you control logging verbosity in different environments.

*   **DEBUG** is for detailed diagnostic information useful during development but too verbose for production. Method parameters, intermediate calculation results, SQL queries being executed — these help developers trace program flow but would overwhelm production log storage.
*   **INFO** records significant events during normal operation: application started, user logged in, request processed, deployment completed. These form the narrative of what your application is doing.
*   **WARN** indicates something unexpected that is not yet a problem but might become one: disk space below 20%, connection pool nearing capacity, deprecated API endpoint still receiving traffic. Warnings are early indicators that deserve attention before they become errors.
*   **ERROR** means something failed: a request could not be completed, a database query timed out, an external API returned an unexpected response. Errors usually require investigation and may require immediate action.
*   **FATAL/CRITICAL** means the application cannot continue: it lost its database connection entirely, a required configuration file is missing, memory is exhausted. These typically result in process termination.

In practice, production environments typically run at INFO level, with the ability to dynamically switch to DEBUG for specific components during troubleshooting — without restarting the application. This dynamic level control is a feature of modern logging frameworks and is invaluable during incident response.

**3. Context: Making Logs Useful**

A log message that says "Request failed" is nearly useless. A log message that says "Request failed" with a request ID, user ID, endpoint, HTTP method, response time, and error details is immediately actionable. Adding context to every log entry — especially a request ID that follows a request through every service it touches — is what makes distributed system debugging possible.

The request ID pattern is particularly powerful. When a user reports a problem, they provide their request ID (often displayed in error pages). You search for that ID across all your services and see the complete journey of their request — which services it touched, how long each step took, and exactly where it failed. Without request IDs, correlating logs across services in a distributed system is effectively impossible.

Equally important is knowing what NOT to log. Never log passwords, API keys, credit card numbers, social security numbers, or any personally identifiable information (PII). In many jurisdictions, logging PII violates regulations like GDPR. Sanitize sensitive data before logging, and regularly audit your logs to ensure nothing sensitive is leaking.

**4. Time-Series Databases: Purpose-Built for Metrics**

Metrics data — numerical values indexed by time and labels — has unique characteristics that general-purpose databases handle poorly. A single server exporting 100 metrics every 10 seconds generates 864,000 data points per day. Across a fleet of 100 servers, that is 86.4 million data points daily. Time-series databases (TSDBs) like Prometheus, InfluxDB, and TimescaleDB are optimized for this specific workload.

Prometheus, the most popular TSDB in the DevOps ecosystem, uses a pull model: instead of applications pushing metrics to Prometheus, Prometheus scrapes metrics endpoints at regular intervals. Applications expose an HTTP endpoint (typically /metrics) that returns current metric values in a standardized format. This pull model has elegant properties: Prometheus knows immediately if a target is down (the scrape fails), and applications do not need to know about the monitoring system.

Metrics in Prometheus have types: counters (only go up — total requests served), gauges (go up and down — current memory usage), histograms (distribute values into buckets — request durations), and summaries (calculate percentiles). Understanding these types is essential for writing correct queries. PromQL, Prometheus's query language, lets you calculate rates ("requests per second over the last 5 minutes"), ratios ("error rate as a percentage"), and aggregations ("average CPU usage across all servers").

**5. The Alerting Pipeline: From Detection to Response**

Effective alerting connects metrics to human action through a carefully designed pipeline. Alert rules in Prometheus define conditions and durations: "if the error rate exceeds 5% for more than 5 minutes, fire an alert." The duration prevents transient spikes from causing false alarms. Alertmanager receives fired alerts and applies routing rules (route database alerts to the database team, route application alerts to the application team), deduplication (do not send 100 alerts for the same issue), grouping (combine related alerts into one notification), and silencing (suppress known issues during maintenance windows).

The notification itself should include everything the responder needs to start investigating: what service is affected, what threshold was breached, when it started, a link to the relevant Grafana dashboard, and a link to the runbook for this alert. The goal is to minimize the time from "alert received" to "engineer is looking at the right dashboard," because those minutes matter during an incident.

**6. Building an Observability Stack**

The standard open-source observability stack that many organizations adopt combines Prometheus (metrics collection and storage), Grafana (visualization and dashboards), Alertmanager (alert routing and notification), Loki (log aggregation — designed to work with Grafana), and Jaeger or Tempo (distributed tracing). This stack is powerful, cost-effective, and benefits from a massive community. Commercial alternatives (Datadog, New Relic, Splunk) offer similar capabilities with less operational overhead but at higher cost. The choice depends on your team's size, budget, and appetite for managing infrastructure.`,
					CodeExamples: `# Structured logging
{
    "timestamp": "2024-01-17T10:00:00Z",
    "level": "INFO",
    "service": "user-service",
    "request_id": "abc123",
    "message": "User created",
    "user_id": 123,
    "duration_ms": 45
}

# Prometheus exporter
from prometheus_client import start_http_server, Gauge

cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')

def collect_metrics():
    cpu_usage.set(get_cpu_usage())
    # Collect other metrics

start_http_server(8000)
while True:
    collect_metrics()
    time.sleep(10)

# Grafana dashboard query
rate(http_requests_total[5m])

# Alert rule (Prometheus)
groups:
  - name: alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
