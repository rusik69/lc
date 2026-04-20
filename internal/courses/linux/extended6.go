package linux

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterLinuxModules([]problems.CourseModule{
		{
			ID:          1828,
			Title:       "Shell Scripting Advanced",
			Description: "Advanced Bash scripting: arrays, string manipulation, regular expressions, subshells, process substitution, debugging, and scripting best practices.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Bash Features",
					Content: `Beyond basic scripting, Bash provides powerful features for text processing, data structures, and process management.

**Arrays:**
` + "```" + `
Indexed arrays:
  # Declaration
  arr=(one two three four)
  arr[0]="first"
  declare -a arr
  
  # Access
  echo "${arr[0]}"              # First element
  echo "${arr[@]}"              # All elements
  echo "${arr[*]}"              # All as single string
  echo "${#arr[@]}"             # Array length
  echo "${!arr[@]}"             # All indices
  
  # Slice
  echo "${arr[@]:1:2}"          # Elements 1-2 (two three)
  
  # Append
  arr+=("five")
  arr+=(six seven)
  
  # Delete
  unset arr[2]                   # Remove element (leaves gap)
  arr=("${arr[@]}")             # Re-index
  
  # Iterate
  for item in "${arr[@]}"; do
    echo "$item"
  done
  
  # Iterate with index
  for i in "${!arr[@]}"; do
    echo "$i: ${arr[$i]}"
  done

Associative arrays (Bash 4+):
  declare -A config
  config[host]="localhost"
  config[port]="8080"
  config[env]="production"
  
  echo "${config[host]}"        # Access by key
  echo "${!config[@]}"          # All keys
  echo "${config[@]}"           # All values
  echo "${#config[@]}"          # Number of keys
  
  # Check key exists
  if [[ -v config[host] ]]; then
    echo "host is set"
  fi
  
  # Iterate
  for key in "${!config[@]}"; do
    echo "$key = ${config[$key]}"
  done
` + "```" + `

**String Manipulation:**
` + "```" + `
str="Hello World Linux"

# Length
echo "${#str}"                  # 17

# Substring
echo "${str:6}"                 # World Linux
echo "${str:6:5}"               # World

# Find and replace
echo "${str/World/Earth}"       # Hello Earth Linux (first match)
echo "${str//l/L}"              # HeLLo WorLd Linux (all matches)
echo "${str/#Hello/Hi}"         # Hi World Linux (prefix)
echo "${str/%Linux/Unix}"       # Hello World Unix (suffix)

# Remove pattern
file="/path/to/file.tar.gz"
echo "${file#*/}"               # path/to/file.tar.gz (shortest from start)
echo "${file##*/}"              # file.tar.gz (longest from start = basename)
echo "${file%.*}"               # /path/to/file.tar (shortest from end)
echo "${file%%.*}"              # /path/to/file (longest from end)

# Common patterns:
echo "${file##*.}"              # gz (extension)
echo "${file%/*}"               # /path/to (directory)

# Case conversion (Bash 4+)
echo "${str,,}"                 # hello world linux (lowercase)
echo "${str^^}"                 # HELLO WORLD LINUX (uppercase)
echo "${str~}"                  # hELLO WORLD LINUX (toggle first)
echo "${str~~}"                 # hELLO wORLD lINUX (toggle all)

# Default values
echo "${var:-default}"          # Use default if unset/empty
echo "${var:=default}"          # Set to default if unset/empty
echo "${var:+alternate}"        # Use alternate if set
echo "${var:?error message}"    # Exit with error if unset/empty
` + "```" + `

**Process Substitution and Subshells:**
` + "```" + `
Subshells:
  # Parentheses create subshell
  (cd /tmp && ls)               # cd doesn't affect parent
  echo $(pwd)                   # Still in original directory
  
  # Variables in subshells don't propagate
  x=1
  (x=2; echo "inner: $x")      # inner: 2
  echo "outer: $x"             # outer: 1

Command substitution:
  result=$(command)              # Preferred
  result=$(cat file | wc -l)
  today=$(date +%Y-%m-%d)

Process substitution:
  # <(command) creates a file descriptor from command output
  # Useful when command expects a file, not stdin
  
  diff <(ls dir1) <(ls dir2)    # Compare directory listings
  
  # Compare sorted files without temp files
  diff <(sort file1) <(sort file2)
  
  # Read from process
  while read line; do
    echo "Got: $line"
  done < <(find /var/log -name "*.log" -mtime -1)
  
  # Write to process
  tee >(gzip > compressed.gz) >(wc -l > count.txt) < input.txt

Here documents and here strings:
  # Here document (multi-line input)
  cat << EOF
  Hello $USER
  Today is $(date)
  EOF
  
  # No variable expansion
  cat << 'EOF'
  This $var is literal
  EOF
  
  # Here string (single line)
  grep "pattern" <<< "$variable"
  
  # Indent-friendly
  cat <<-EOF
	  Tabs are stripped
	  from the beginning
	EOF
` + "```" + `

**Regular Expressions in Bash:**
` + "```" + `
[[ with =~ operator:
  # Basic regex matching
  if [[ "hello123" =~ ^[a-z]+[0-9]+$ ]]; then
    echo "Match!"
  fi
  
  # Capture groups
  if [[ "2024-01-15" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})$ ]]; then
    year="${BASH_REMATCH[1]}"
    month="${BASH_REMATCH[2]}"
    day="${BASH_REMATCH[3]}"
    echo "$year/$month/$day"
  fi
  
  # IP address validation
  ip_regex='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
  if [[ "$ip" =~ $ip_regex ]]; then
    echo "Valid IP format"
  fi
  
  # Email validation (basic)
  email_regex='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
  if [[ "$email" =~ $email_regex ]]; then
    echo "Valid email format"
  fi

grep with regex:
  grep -E 'pattern1|pattern2' file     # Extended regex (alternation)
  grep -P '\d{3}-\d{4}' file          # Perl-compatible regex
  grep -o '[0-9]\+' file               # Print only matches
  grep -c 'pattern' file               # Count matches
  grep -n 'pattern' file               # Line numbers
  grep -r 'pattern' dir/               # Recursive
  grep -l 'pattern' *.log              # Files with matches

sed:
  sed 's/old/new/' file                 # First match per line
  sed 's/old/new/g' file               # All matches
  sed -i 's/old/new/g' file            # In-place edit
  sed -n '10,20p' file                 # Print lines 10-20
  sed '/pattern/d' file                 # Delete matching lines
  sed '/^$/d' file                      # Delete empty lines
  sed -E 's/([0-9]+)/[\1]/g' file     # Extended regex

awk:
  awk '{print $1}' file                # First field
  awk -F: '{print $1, $3}' /etc/passwd # Custom delimiter
  awk '$3 > 1000' /etc/passwd          # Condition
  awk 'NR >= 10 && NR <= 20' file     # Line range
  awk '{sum += $1} END {print sum}' f  # Sum column
  awk '!seen[$0]++' file               # Remove duplicates (preserve order)
` + "```" + ``,
					CodeExamples: `# Advanced Bash scripting examples

# 1. Configuration file parser
#!/bin/bash
declare -A CONFIG

parse_config() {
    local file="$1"
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        # Remove surrounding quotes
        value="${value%\"}"
        value="${value#\"}"
        CONFIG["$key"]="$value"
    done < "$file"
}

get_config() {
    local key="$1"
    local default="${2:-}"
    echo "${CONFIG[$key]:-$default}"
}

# Usage:
# parse_config /etc/myapp.conf
# DB_HOST=$(get_config "database.host" "localhost")

# 2. Parallel execution helper
#!/bin/bash
MAX_PARALLEL=4
RUNNING=0

run_parallel() {
    local cmd="$1"
    while [ "$RUNNING" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null
        RUNNING=$((RUNNING - 1))
    done
    eval "$cmd" &
    RUNNING=$((RUNNING + 1))
}

wait_all() {
    wait
    RUNNING=0
}

# Usage:
# for host in "${HOSTS[@]}"; do
#     run_parallel "ssh $host 'uptime'"
# done
# wait_all

# 3. Log analyzer
#!/bin/bash
LOGFILE="${1:-/var/log/syslog}"

echo "=== Log Analysis Report ==="
echo "File: $LOGFILE"
echo "Period: $(head -1 "$LOGFILE" | awk '{print $1,$2,$3}') to $(tail -1 "$LOGFILE" | awk '{print $1,$2,$3}')"
echo ""

echo "=== Error Count by Service ==="
grep -i "error\|fail\|crit" "$LOGFILE" | \
    awk '{print $5}' | sort | uniq -c | sort -rn | head -10

echo ""
echo "=== Hourly Message Distribution ==="
awk '{print $3}' "$LOGFILE" | cut -d: -f1 | sort | uniq -c

echo ""
echo "=== Top 10 Most Frequent Messages ==="
awk '{$1=$2=$3=$4=$5=""; print $0}' "$LOGFILE" | \
    sed 's/^[[:space:]]*//' | sort | uniq -c | sort -rn | head -10

# 4. Safe script template
#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Strict mode:
# -e: exit on error
# -u: error on undefined variable
# -o pipefail: pipe returns rightmost non-zero exit code

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }
error() { log "ERROR: $*"; exit 1; }
cleanup() { log "Cleaning up..."; rm -f "$TMPFILE"; }

trap cleanup EXIT

TMPFILE=$(mktemp)

main() {
    log "Starting $SCRIPT_NAME"
    # Script logic here
    log "Complete"
}

main "$@"`,
				},
				{
					Title: "Text Processing and Automation",
					Content: `Linux provides powerful text processing tools for data manipulation, log analysis, and automation tasks.

**Advanced awk:**
` + "```" + `
awk program structure:
  awk 'BEGIN { init } pattern { action } END { cleanup }' file

Built-in variables:
  NR    Current record (line) number (global)
  NF    Number of fields in current record
  FNR   Record number in current file
  FS    Field separator (default: whitespace)
  RS    Record separator (default: newline)
  OFS   Output field separator
  ORS   Output record separator
  FILENAME  Current filename

Examples:
  # CSV processing
  awk -F, '{print $1, $3}' data.csv
  
  # Sum a column
  awk '{sum += $2} END {printf "Total: %.2f\n", sum}' data.txt
  
  # Average
  awk '{sum += $1; count++} END {print sum/count}' data.txt
  
  # Filter and transform
  awk -F: '$3 >= 1000 {printf "%-20s UID=%s\n", $1, $3}' /etc/passwd
  
  # Group by
  awk '{count[$1]++} END {for (k in count) print count[k], k}' access.log | sort -rn
  
  # Multi-file
  awk 'FNR==1 {print "=== " FILENAME " ==="} {print}' file1 file2
  
  # Convert to JSON
  awk -F, 'NR>1 {printf "{\"name\":\"%s\",\"value\":%s}\n", $1, $2}' data.csv
  
  # Running average
  awk '{sum += $1; print NR, $1, sum/NR}' data.txt
  
  # Find duplicates
  awk 'seen[$0]++ == 1' file           # Print first duplicate
  awk 'seen[$0]++' file                # Print all duplicates
  
  # Transpose columns to rows
  awk '{for(i=1;i<=NF;i++) a[i]=a[i]" "$i} END {for(i in a) print a[i]}' file

awk functions:
  # String functions
  length(s)              # String length
  substr(s, start, len)  # Substring
  index(s, target)       # Find position
  split(s, arr, sep)     # Split into array
  sub(regex, repl, s)    # Replace first
  gsub(regex, repl, s)   # Replace all
  match(s, regex)        # Regex match
  tolower(s), toupper(s) # Case conversion
  sprintf(fmt, ...)      # Format string
  
  # Math functions
  int(x)    sin(x)    cos(x)    sqrt(x)
  log(x)    exp(x)    rand()    srand(seed)
` + "```" + `

**Advanced sed:**
` + "```" + `
Multi-line operations:
  # Replace across lines (join continuation lines)
  sed ':a; /\\$/ { N; s/\\\n//; ba }' file
  
  # Delete block between patterns
  sed '/START/,/END/d' file
  
  # Extract block between patterns
  sed -n '/START/,/END/p' file
  
  # Insert before/after pattern
  sed '/pattern/i\New line before' file
  sed '/pattern/a\New line after' file
  
  # Multiple operations
  sed -e 's/foo/bar/g' -e 's/baz/qux/g' file
  sed '
    s/foo/bar/g
    s/baz/qux/g
    /^#/d
  ' file

Practical sed recipes:
  # Remove trailing whitespace
  sed 's/[[:space:]]*$//' file
  
  # Add line numbers
  sed '=' file | sed 'N; s/\n/\t/'
  
  # Double space
  sed 'G' file
  
  # Remove consecutive blank lines
  sed '/^$/N;/^\n$/d' file
  
  # Convert Windows to Unix line endings
  sed 's/\r$//' file
  
  # Extract email addresses
  sed -n 's/.*\([a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]*\.[a-zA-Z]\{2,\}\).*/\1/p' file
` + "```" + `

**Automation with cron and at:**
` + "```" + `
Cron format:
  # m h dom mon dow command
  # * * * * *
  # │ │ │   │   │
  # │ │ │   │   └─ day of week (0-7, 0=Sun)
  # │ │ │   └───── month (1-12)
  # │ │ └───────── day of month (1-31)
  # │ └─────────── hour (0-23)
  # └───────────── minute (0-59)

Examples:
  0 2 * * *     /usr/local/bin/backup.sh      # Daily 2 AM
  */15 * * * *  /usr/local/bin/check.sh        # Every 15 min
  0 0 * * 0     /usr/local/bin/weekly.sh       # Sunday midnight
  0 9-17 * * 1-5 /usr/local/bin/monitor.sh    # Weekdays 9-5 hourly
  0 0 1 * *     /usr/local/bin/monthly.sh      # First of month

Management:
  crontab -l                    # List user's crontab
  crontab -e                    # Edit user's crontab
  crontab -r                    # Remove user's crontab
  crontab -u user -l            # List another user's crontab

System cron directories:
  /etc/cron.d/          Custom cron files
  /etc/cron.daily/      Daily scripts
  /etc/cron.hourly/     Hourly scripts
  /etc/cron.weekly/     Weekly scripts
  /etc/cron.monthly/    Monthly scripts

at (one-time scheduled jobs):
  echo "backup.sh" | at 2:00 AM
  at -f script.sh 10:00 PM tomorrow
  at now + 2 hours
  atq                           # List pending jobs
  atrm 5                        # Remove job 5
` + "```" + ``,
					CodeExamples: `# Text processing and automation examples

# 1. Log rotation and analysis
#!/bin/bash
LOGDIR="/var/log/myapp"
ARCHIVE="/var/log/myapp/archive"
RETENTION_DAYS=30

mkdir -p "$ARCHIVE"

# Rotate current logs
for log in "$LOGDIR"/*.log; do
    [ -f "$log" ] || continue
    BASENAME=$(basename "$log")
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    gzip -c "$log" > "$ARCHIVE/${BASENAME}.${TIMESTAMP}.gz"
    : > "$log"  # Truncate
done

# Delete old archives
find "$ARCHIVE" -name "*.gz" -mtime +"$RETENTION_DAYS" -delete

# Generate daily report
echo "=== Daily Log Report $(date +%Y-%m-%d) ==="
for log in "$LOGDIR"/*.log; do
    [ -f "$log" ] || continue
    echo "--- $(basename "$log") ---"
    echo "Total lines: $(wc -l < "$log")"
    echo "Errors: $(grep -ci 'error' "$log")"
    echo "Warnings: $(grep -ci 'warn' "$log")"
    echo ""
done

# 2. CSV processor
#!/bin/bash
# Process CSV: calculate stats per group
INPUT="$1"

awk -F, '
NR == 1 {
    # Save headers
    for (i = 1; i <= NF; i++) headers[i] = $i
    next
}
{
    group = $1
    value = $2 + 0
    count[group]++
    sum[group] += value
    if (!(group in min) || value < min[group]) min[group] = value
    if (!(group in max) || value > max[group]) max[group] = value
}
END {
    printf "%-15s %8s %8s %8s %8s\n", "Group", "Count", "Sum", "Min", "Max"
    printf "%-15s %8s %8s %8s %8s\n", "-----", "-----", "---", "---", "---"
    for (g in count) {
        printf "%-15s %8d %8.2f %8.2f %8.2f\n", 
               g, count[g], sum[g], min[g], max[g]
    }
}' "$INPUT"

# 3. System maintenance automation
#!/bin/bash
set -euo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Starting system maintenance"

# Update package lists
log "Updating packages..."
apt-get update -qq
UPGRADABLE=$(apt list --upgradable 2>/dev/null | wc -l)
log "$((UPGRADABLE - 1)) packages can be upgraded"

# Clean package cache
log "Cleaning package cache..."
apt-get autoremove -y -qq
apt-get clean -qq

# Clean old journals
log "Trimming journal logs..."
journalctl --vacuum-time=7d --vacuum-size=500M 2>/dev/null

# Clean tmp files older than 7 days
log "Cleaning temp files..."
find /tmp -type f -atime +7 -delete 2>/dev/null

# Check disk usage
log "Disk usage check:"
df -h / /var /home 2>/dev/null | awk 'NR>1 {
    gsub(/%/, "", $5)
    if ($5+0 > 85) printf "WARNING: %s is %s%% full\n", $6, $5
    else printf "OK: %s is %s%% full\n", $6, $5
}'

# Check for failed services
FAILED=$(systemctl --failed --no-legend | wc -l)
if [ "$FAILED" -gt 0 ]; then
    log "WARNING: $FAILED failed services:"
    systemctl --failed --no-legend
fi

log "Maintenance complete"`,
				},
			},
		},
	})
}
