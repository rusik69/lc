package algorithms

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAlgorithmsModules([]problems.CourseModule{
		{
			ID:          5,
			Title:       "Linked Lists",
			Description: "Master linked list operations and common patterns for solving linked list problems.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "Linked Lists Basics",
					Content: `A linked list is a fundamental linear data structure where elements are stored in nodes, each containing data and a reference (pointer) to the next node. Unlike arrays, elements are not stored in contiguous memory locations, which gives linked lists unique properties and trade-offs.

**Understanding Linked Lists:**

**Node Structure:**
- Each node contains: data (value) and next (pointer to next node)
- Last node's next pointer is null/nil (indicates end)
- Head pointer points to first node
- Optional tail pointer points to last node

**Visual Representation:**
` + "```" + `
Singly Linked List:
Head -> [10|next] -> [20|next] -> [30|nil]
         Node 1       Node 2       Node 3
` + "```" + `

**Types of Linked Lists:**

**1. Singly Linked List:**

**Structure:**
- Each node has: data and next pointer
- Can only traverse forward (head to tail)
- Last node's next is null

**Advantages:**
- Less memory overhead (one pointer per node)
- Simple implementation
- Efficient forward traversal

**Disadvantages:**
- Cannot traverse backward
- Must traverse from head to reach any node
- Cannot delete node without previous node reference

**Use Cases:**
- Simple lists where backward traversal not needed
- Implementing stack (LIFO)
- Memory-constrained environments

**2. Doubly Linked List:**

**Structure:**
- Each node has: data, next pointer, and prev pointer
- Can traverse both forward and backward
- First node's prev is null, last node's next is null

**Advantages:**
- Bidirectional traversal
- Can delete node with just node reference (no need for previous)
- More flexible operations

**Disadvantages:**
- More memory overhead (two pointers per node)
- More complex implementation
- Need to update both next and prev pointers

**Use Cases:**
- Need backward traversal
- Frequent deletions by node reference
- Implementing deque (double-ended queue)

**3. Circular Linked List:**

**Structure:**
- Last node points back to first node
- No null pointer at end
- Can be singly or doubly linked

**Advantages:**
- No null pointer to check
- Can start traversal from any node
- Useful for round-robin algorithms

**Disadvantages:**
- Need to track when we've completed a cycle
- More complex termination conditions
- Risk of infinite loops if not careful

**Use Cases:**
- Round-robin scheduling
- Circular buffers
- Problems requiring cyclic processing

**Memory Structure Comparison:**

**Array:**
` + "```" + `
Memory: [10][20][30][40]
        All elements contiguous
        Cache-friendly (spatial locality)
        Fixed size (or expensive resizing)
` + "```" + `

**Linked List:**
` + "```" + `
Memory: [10|ptr] -> [20|ptr] -> [30|ptr] -> [40|nil]
        Nodes scattered in memory
        Cache-unfriendly (random access)
        Dynamic size (easy growth)
` + "```" + `

**Advantages of Linked Lists:**

**1. Dynamic Size:**
- Grow and shrink as needed
- No need to pre-allocate size
- Only allocate memory for elements that exist

**2. Efficient Insertion/Deletion:**
- Insert at head: O(1) - just update head pointer
- Insert at known position: O(1) - just update pointers
- Delete at known position: O(1) - just update pointers
- No shifting of elements needed (unlike arrays)

**3. Memory Efficiency:**
- Only allocate what you need
- No wasted space for unused capacity
- Can free memory when nodes deleted

**4. Flexibility:**
- Easy to rearrange elements
- Can split and merge lists efficiently
- Can insert/delete without affecting other elements

**Disadvantages of Linked Lists:**

**1. No Random Access:**
- Cannot access element by index directly
- Must traverse from head to reach element
- Accessing kth element: O(k) time

**2. Extra Memory Overhead:**
- Need to store pointers (8 bytes per pointer on 64-bit)
- Singly linked: 1 pointer per node
- Doubly linked: 2 pointers per node

**3. Cache Unfriendly:**
- Nodes scattered in memory (not contiguous)
- Poor spatial locality
- Cache misses more frequent than arrays

**4. No Backward Traversal (Singly Linked):**
- Cannot traverse backward
- Need doubly linked list for bidirectional traversal

**Common Operations:**

**1. Insert at Head:**
- Create new node
- Set new node's next to current head
- Update head to new node
- Time: O(1)

**2. Insert at Tail:**
- If tail pointer maintained: O(1)
- Otherwise: traverse to end, then insert: O(n)

**3. Insert at Position:**
- Traverse to position: O(k) where k is position
- Update pointers: O(1)
- Total: O(n) worst case

**4. Delete at Head:**
- Update head to head.next
- Free old head node
- Time: O(1)

**5. Delete at Position:**
- Find node at position: O(k)
- Update pointers: O(1)
- Total: O(n) worst case

**6. Search:**
- Linear search from head
- Time: O(n) worst case
- No binary search possible (no random access)

**Common Patterns:**

**1. Dummy Node:**
- Use dummy head node to simplify edge cases
- Avoids special handling for empty list
- Simplifies insertion/deletion at head

**2. Two Pointers:**
- Fast/slow pointers for cycle detection
- Fast/slow pointers for finding middle
- Two pointers for finding kth from end

**3. Reverse:**
- Iterative reversal: O(n) time, O(1) space
- Recursive reversal: O(n) time, O(n) space
- Partial reversal: reverse sublist

**4. Merge:**
- Merge two sorted lists: O(n+m) time
- Use two pointers, compare and merge
- Efficient for combining sorted data

**When to Use Linked Lists:**

**Good For:**
- Unknown or frequently changing size
- Frequent insertions/deletions in middle
- Don't need random access
- Implementing other data structures (stack, queue)
- Memory allocation constraints

**Not Ideal For:**
- Need random access by index
- Cache performance critical
- Memory overhead is concern
- Frequent searching

**Common Mistakes:**

**1. Losing Reference:**
- Not saving next pointer before updating
- Results in losing rest of list
- Always save next before modifying

**2. Not Handling Empty List:**
- Forgetting to check if head is null
- Accessing null pointer causes crash
- Always check empty case first

**3. Off-by-One Errors:**
- Incorrect loop termination
- Stopping one node too early/late
- Carefully track current and next

**4. Pointer Updates:**
- Forgetting to update pointers correctly
- Updating in wrong order
- Need to update all affected pointers

**5. Memory Leaks:**
- Not freeing deleted nodes (in languages with manual memory)
- Losing reference to allocated memory
- Always free memory when deleting`,
					CodeExamples: `// Go: Comprehensive linked list operations

// ===== BASIC NODE STRUCTURE =====
type ListNode struct {
    Val  int
    Next *ListNode
}

// ===== INSERT OPERATIONS =====
// Insert at head
func insertAtHead(head *ListNode, val int) *ListNode {
    newNode := &ListNode{Val: val, Next: head}
    return newNode  // New head
}

// Insert at tail
func insertAtTail(head *ListNode, val int) *ListNode {
    newNode := &ListNode{Val: val, Next: nil}
    if head == nil {
        return newNode
    }
    curr := head
    for curr.Next != nil {
        curr = curr.Next
    }
    curr.Next = newNode
    return head
}

// Insert at position
func insertAtPosition(head *ListNode, val int, pos int) *ListNode {
    if pos == 0 {
        return &ListNode{Val: val, Next: head}
    }
    curr := head
    for i := 0; i < pos-1 && curr != nil; i++ {
        curr = curr.Next
    }
    if curr == nil {
        return head  // Position out of bounds
    }
    newNode := &ListNode{Val: val, Next: curr.Next}
    curr.Next = newNode
    return head
}

// ===== DELETE OPERATIONS =====
// Delete at head
func deleteAtHead(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    return head.Next
}

// Delete by value
func deleteByValue(head *ListNode, val int) *ListNode {
    if head == nil {
        return nil
    }
    if head.Val == val {
        return head.Next
    }
    curr := head
    for curr.Next != nil {
        if curr.Next.Val == val {
            curr.Next = curr.Next.Next
            return head
        }
        curr = curr.Next
    }
    return head
}

// Delete at position
func deleteAtPosition(head *ListNode, pos int) *ListNode {
    if head == nil {
        return nil
    }
    if pos == 0 {
        return head.Next
    }
    curr := head
    for i := 0; i < pos-1 && curr != nil && curr.Next != nil; i++ {
        curr = curr.Next
    }
    if curr != nil && curr.Next != nil {
        curr.Next = curr.Next.Next
    }
    return head
}

// ===== SEARCH OPERATIONS =====
// Search for value
func search(head *ListNode, val int) bool {
    curr := head
    for curr != nil {
        if curr.Val == val {
            return true
        }
        curr = curr.Next
    }
    return false
}

// Get node at index
func getNodeAt(head *ListNode, index int) *ListNode {
    curr := head
    for i := 0; i < index && curr != nil; i++ {
        curr = curr.Next
    }
    return curr
}

// ===== TRAVERSAL =====
// Print list
func printList(head *ListNode) {
    curr := head
    for curr != nil {
        fmt.Print(curr.Val, " -> ")
        curr = curr.Next
    }
    fmt.Println("nil")
}

// Get length
func getLength(head *ListNode) int {
    length := 0
    curr := head
    for curr != nil {
        length++
        curr = curr.Next
    }
    return length
}

// ===== REVERSE OPERATIONS =====
// Reverse entire list (iterative)
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head
    for curr != nil {
        next := curr.Next  // Save next
        curr.Next = prev   // Reverse link
        prev = curr        // Move prev forward
        curr = next        // Move curr forward
    }
    return prev  // New head
}

// Reverse entire list (recursive)
func reverseListRecursive(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    newHead := reverseListRecursive(head.Next)
    head.Next.Next = head
    head.Next = nil
    return newHead
}

// ===== DUMMY NODE PATTERN =====
// Remove duplicates using dummy node
func removeDuplicates(head *ListNode) *ListNode {
    dummy := &ListNode{Val: 0, Next: head}
    curr := dummy
    
    for curr.Next != nil && curr.Next.Next != nil {
        if curr.Next.Val == curr.Next.Next.Val {
            val := curr.Next.Val
            for curr.Next != nil && curr.Next.Val == val {
                curr.Next = curr.Next.Next
            }
        } else {
            curr = curr.Next
        }
    }
    
    return dummy.Next
}

// ===== TWO POINTERS PATTERN =====
// Find middle node
func findMiddle(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}

// Find kth from end
func findKthFromEnd(head *ListNode, k int) *ListNode {
    fast := head
    for i := 0; i < k && fast != nil; i++ {
        fast = fast.Next
    }
    if fast == nil {
        return nil  // k > length
    }
    slow := head
    for fast != nil {
        slow = slow.Next
        fast = fast.Next
    }
    return slow
}

# Python: Comprehensive linked list operations

# ===== BASIC NODE STRUCTURE =====
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# ===== INSERT OPERATIONS =====
# Insert at head
def insert_at_head(head, val):
    return ListNode(val, head)

# Insert at tail
def insert_at_tail(head, val):
    new_node = ListNode(val)
    if not head:
        return new_node
    curr = head
    while curr.next:
        curr = curr.next
    curr.next = new_node
    return head

# Insert at position
def insert_at_position(head, val, pos):
    if pos == 0:
        return ListNode(val, head)
    curr = head
    for i in range(pos - 1):
        if not curr:
            return head  # Position out of bounds
        curr = curr.next
    if not curr:
        return head
    new_node = ListNode(val, curr.next)
    curr.next = new_node
    return head

# ===== DELETE OPERATIONS =====
# Delete at head
def delete_at_head(head):
    if not head:
        return None
    return head.next

# Delete by value
def delete_by_value(head, val):
    if not head:
        return None
    if head.val == val:
        return head.next
    curr = head
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
            return head
        curr = curr.next
    return head

# Delete at position
def delete_at_position(head, pos):
    if not head:
        return None
    if pos == 0:
        return head.next
    curr = head
    for i in range(pos - 1):
        if not curr or not curr.next:
            return head
        curr = curr.next
    if curr.next:
        curr.next = curr.next.next
    return head

# ===== SEARCH OPERATIONS =====
# Search for value
def search(head, val):
    curr = head
    while curr:
        if curr.val == val:
            return True
        curr = curr.next
    return False

# Get node at index
def get_node_at(head, index):
    curr = head
    for i in range(index):
        if not curr:
            return None
        curr = curr.next
    return curr

# ===== TRAVERSAL =====
# Print list
def print_list(head):
    curr = head
    result = []
    while curr:
        result.append(str(curr.val))
        curr = curr.next
    print(" -> ".join(result) + " -> None")

# Get length
def get_length(head):
    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next
    return length

# ===== REVERSE OPERATIONS =====
# Reverse entire list (iterative)
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next  # Save next
        curr.next = prev       # Reverse link
        prev = curr            # Move prev forward
        curr = next_node       # Move curr forward
    return prev  # New head

# Reverse entire list (recursive)
def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head

# ===== DUMMY NODE PATTERN =====
# Remove duplicates using dummy node
def remove_duplicates(head):
    dummy = ListNode(0, head)
    curr = dummy
    
    while curr.next and curr.next.next:
        if curr.next.val == curr.next.next.val:
            val = curr.next.val
            while curr.next and curr.next.val == val:
                curr.next = curr.next.next
        else:
            curr = curr.next
    
    return dummy.next

# ===== TWO POINTERS PATTERN =====
# Find middle node
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# Find kth from end
def find_kth_from_end(head, k):
    fast = head
    for _ in range(k):
        if not fast:
            return None  # k > length
        fast = fast.next
    
    slow = head
    while fast:
        slow = slow.next
        fast = fast.next
    return slow

# Key insights:
# 1. Always save next pointer before modifying
# 2. Use dummy node to simplify edge cases
# 3. Two pointers (fast/slow) for cycle detection and finding middle
# 4. Check for empty list before operations
# 5. Linked lists are O(1) for insert/delete at known position, O(n) for search`,
				},
				{
					Title: "Reverse Operations",
					Content: `Reversing linked lists is a fundamental operation that appears in many problems. There are multiple ways to reverse: iterative, recursive, and partial reversals.

**Types of Reversals:**

**1. Reverse Entire List:**
- Reverse all nodes from head to tail
- Iterative: Use three pointers (prev, curr, next)
- Recursive: Reverse rest, then connect head

**2. Reverse in Groups:**
- Reverse k nodes at a time
- Example: Reverse nodes in groups of 2: 1->2->3->4 becomes 2->1->4->3
- Requires careful pointer management

**3. Reverse Between Positions:**
- Reverse nodes from position m to n
- Keep nodes before m and after n unchanged
- Requires finding start position and reconnecting

**4. Reverse in Pairs:**
- Swap every two adjacent nodes
- Example: 1->2->3->4 becomes 2->1->4->3
- Can be done iteratively or recursively

**Key Techniques:**
- **Three Pointer Technique**: prev, curr, next for iterative reversal
- **Dummy Node**: Simplify edge cases (empty list, single node)
- **Recursive Approach**: Reverse rest, then connect current node
- **Group Reversal**: Reverse k nodes, then recursively reverse rest

**Common Patterns:**
- Save next pointer before modifying
- Update pointers in correct order
- Handle edge cases: empty list, single node, k > list length
- Return new head after reversal`,
					CodeExamples: `// Go: Reverse entire list (iterative)
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head
    
    for curr != nil {
        next := curr.Next  // Save next
        curr.Next = prev   // Reverse link
        prev = curr        // Move prev forward
        curr = next        // Move curr forward
    }
    
    return prev  // New head
}

// Go: Reverse entire list (recursive)
func reverseListRecursive(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    newHead := reverseListRecursive(head.Next)
    head.Next.Next = head  // Reverse link
    head.Next = nil        // Set tail to nil
    
    return newHead
}

// Go: Reverse in groups of k
func reverseKGroup(head *ListNode, k int) *ListNode {
    curr := head
    count := 0
    
    // Check if k nodes exist
    for curr != nil && count < k {
        curr = curr.Next
        count++
    }
    
    if count == k {
        // Reverse remaining groups first
        curr = reverseKGroup(curr, k)
        
        // Reverse current k nodes
        for count > 0 {
            next := head.Next
            head.Next = curr
            curr = head
            head = next
            count--
        }
        head = curr
    }
    
    return head
}

# Python: Reverse entire list (iterative)
def reverse_list(head):
    prev = None
    curr = head
    
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    
    return prev

# Python: Reverse in groups of k
def reverse_k_group(head, k):
    curr = head
    count = 0
    
    while curr and count < k:
        curr = curr.next
        count += 1
    
    if count == k:
        curr = reverse_k_group(curr, k)
        
        while count > 0:
            next_node = head.next
            head.next = curr
            curr = head
            head = next_node
            count -= 1
        head = curr
    
    return head`,
				},
				{
					Title: "Merge K Sorted Lists",
					Content: `Merging k sorted linked lists is a classic problem that demonstrates divide-and-conquer and heap/priority queue techniques.

**Problem Statement:**
Given k sorted linked lists, merge them into one sorted linked list.

**Approaches:**

**1. Brute Force:**
- Merge lists one by one
- Time: O(k×n) where n is average list length
- Simple but inefficient

**2. Divide and Conquer:**
- Pair up lists and merge pairs
- Repeat until one list remains
- Time: O(n log k) - log k levels, each level processes n nodes
- Space: O(1) excluding recursion stack

**3. Priority Queue/Heap:**
- Add head of each list to min-heap
- Extract minimum, add to result
- Add next node from same list to heap
- Time: O(n log k) - n extractions, each O(log k)
- Space: O(k) - heap stores k nodes

**Best Approach:**
- **Divide and Conquer**: Best time complexity, good space efficiency
- **Priority Queue**: Clean code, good for interview, slightly more space

**Key Insights:**
- Divide and conquer reduces problem size logarithmically
- Priority queue always gives us minimum element efficiently
- Can merge two lists in O(m + n) time
- Handle edge cases: empty lists, single list, null inputs`,
					CodeExamples: `// Go: Merge K Lists using Divide and Conquer
func mergeKLists(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }
    if len(lists) == 1 {
        return lists[0]
    }
    
    mid := len(lists) / 2
    left := mergeKLists(lists[:mid])
    right := mergeKLists(lists[mid:])
    
    return mergeTwoLists(left, right)
}

func mergeTwoLists(l1, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    curr := dummy
    
    for l1 != nil && l2 != nil {
        if l1.Val <= l2.Val {
            curr.Next = l1
            l1 = l1.Next
        } else {
            curr.Next = l2
            l2 = l2.Next
        }
        curr = curr.Next
    }
    
    if l1 != nil {
        curr.Next = l1
    } else {
        curr.Next = l2
    }
    
    return dummy.Next
}

# Python: Merge K Lists using Divide and Conquer
def merge_k_lists(lists):
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    
    mid = len(lists) // 2
    left = merge_k_lists(lists[:mid])
    right = merge_k_lists(lists[mid:])
    
    return merge_two_lists(left, right)

def merge_two_lists(l1, l2):
    dummy = ListNode()
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    return dummy.next`,
				},
				{
					Title: "Floyd's Cycle Detection Details",
					Content: `Floyd's cycle detection algorithm (also called the "tortoise and hare" algorithm) detects cycles in linked lists using two pointers moving at different speeds.

**How It Works:**
1. Use two pointers: slow (moves 1 step) and fast (moves 2 steps)
2. If cycle exists, fast pointer will eventually catch up to slow pointer
3. If no cycle, fast pointer reaches end (nil)

**Mathematical Proof:**
- Let distance from start to cycle start = m
- Let cycle length = n
- When slow enters cycle, fast is m steps ahead
- Fast catches up at rate of 1 step per iteration
- They meet after (n - m) iterations from cycle start
- Meeting point is (n - m) steps from cycle start

**Finding Cycle Start:**
After detecting cycle, to find where cycle starts:
1. Keep one pointer at meeting point
2. Move another pointer from head
3. Move both one step at a time
4. They meet at cycle start

**Why Finding Cycle Start Works:**
- Distance from head to cycle start = m
- Distance from meeting point to cycle start = n - (n - m) = m
- So both pointers travel same distance to cycle start

**Applications:**
- Detect cycles in linked lists
- Find duplicate numbers (array as linked list)
- Detect infinite loops
- Find cycle length
- Find cycle start position

**Time Complexity:** O(n) - linear time
**Space Complexity:** O(1) - constant space`,
					CodeExamples: `// Go: Detect cycle
func hasCycle(head *ListNode) bool {
    if head == nil {
        return false
    }
    
    slow, fast := head, head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            return true  // Cycle detected
        }
    }
    
    return false
}

// Go: Find cycle start
func detectCycle(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    slow, fast := head, head
    
    // Detect cycle
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            break  // Cycle detected
        }
    }
    
    // No cycle
    if fast == nil || fast.Next == nil {
        return nil
    }
    
    // Find cycle start
    slow = head
    for slow != fast {
        slow = slow.Next
        fast = fast.Next
    }
    
    return slow  // Cycle start
}

# Python: Detect cycle
def has_cycle(head):
    if not head:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

# Python: Find cycle start
def detect_cycle(head):
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            break
    
    if not fast or not fast.next:
        return None
    
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow`,
				},
			},
			ProblemIDs: []int{36, 37, 38, 39, 40},
		},
		{
			ID:          6,
			Title:       "Trees",
			Description: "Learn tree data structures, traversal methods, and tree-based algorithms.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Binary Trees",
					Content: `A binary tree is a tree data structure where each node has at most two children: left and right. Trees are fundamental for representing hierarchical relationships.

**Tree Terminology:**
- **Root**: Topmost node (no parent)
- **Leaf**: Node with no children (terminal node)
- **Internal Node**: Node with at least one child
- **Depth**: Distance from root to node (root depth = 0)
- **Height**: Maximum depth of tree (longest path from root to leaf)
- **Subtree**: Tree formed by a node and all its descendants
- **Ancestor**: Any node on path from root to current node
- **Descendant**: Any node in subtree of current node
- **Sibling**: Nodes with same parent

**Tree Properties:**
- **Full Binary Tree**: Every node has 0 or 2 children
- **Complete Binary Tree**: All levels filled except possibly last, filled left to right
- **Perfect Binary Tree**: All internal nodes have 2 children, all leaves at same level
- **Balanced Tree**: Height difference between left and right subtrees ≤ 1

**Traversal Methods:**

1. **Inorder (Left → Root → Right)**:
   - Visits nodes in sorted order for BST
   - Used for: printing sorted values, expression trees

2. **Preorder (Root → Left → Right)**:
   - Root visited before children
   - Used for: copying tree, prefix expressions

3. **Postorder (Left → Right → Root)**:
   - Root visited after children
   - Used for: deleting tree, postfix expressions

4. **Level-order (Breadth-First)**:
   - Visits nodes level by level
   - Uses queue for implementation
   - Used for: finding level-specific information

**Recursive vs Iterative:**
- **Recursive**: Natural for tree problems, uses call stack
- **Iterative**: Uses explicit stack/queue, more control

**Common Patterns:**
- **DFS (Depth-First)**: Preorder, Inorder, Postorder
- **BFS (Breadth-First)**: Level-order traversal
- **Path Problems**: Track path from root to node
- **Tree Construction**: Build tree from traversal sequences

**Time Complexity:** O(n) for all traversals - visit each node once

**Space Complexity:** 
- Recursive: O(h) where h is height (call stack)
- Iterative: O(n) worst case (explicit stack/queue)`,
					CodeExamples: `// Go: TreeNode structure
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Inorder traversal
func inorderTraversal(root *TreeNode) []int {
    var result []int
    if root == nil {
        return result
    }
    result = append(result, inorderTraversal(root.Left)...)
    result = append(result, root.Val)
    result = append(result, inorderTraversal(root.Right)...)
    return result
}

# Python: TreeNode structure
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right`,
				},
				{
					Title: "Binary Search Trees",
					Content: `A Binary Search Tree (BST) is a binary tree with ordering property that enables efficient searching, insertion, and deletion.

**BST Property:**
- **Left subtree**: All nodes have values < root
- **Right subtree**: All nodes have values > root
- **Both subtrees**: Are also valid BSTs
- **No duplicates**: Typically (or handle with count/frequency)

**Key Properties:**
- **Inorder Traversal**: Always gives sorted order (ascending)
- **Search**: O(log n) average (balanced), O(n) worst (skewed)
- **Insert**: O(log n) average, O(n) worst
- **Delete**: O(log n) average, O(n) worst
- **Min/Max**: O(log n) - leftmost/rightmost node

**BST Operations:**

1. **Search**:
   - Compare target with root
   - Go left if smaller, right if larger
   - Repeat until found or null

2. **Insert**:
   - Find correct position (like search)
   - Insert new node as leaf
   - Maintain BST property

3. **Delete** (Three Cases):
   - **No children**: Simply remove node
   - **One child**: Replace with child
   - **Two children**: Replace with inorder successor (or predecessor)

**Balanced BSTs:**
- **AVL Tree**: Self-balancing, height difference ≤ 1
- **Red-Black Tree**: Self-balancing with color properties
- **B-Tree**: Multi-way tree for databases
- **Splay Tree**: Recently accessed nodes move to root

**When BST Degenerates:**
- Inserting sorted sequence creates linked list
- Height becomes O(n) instead of O(log n)
- Operations become O(n) instead of O(log n)

**Common Problems:**
- Validate BST (check ordering property)
- Find kth smallest/largest element
- Range queries (find all values in range)
- Lowest Common Ancestor (LCA)

**Real-World Applications:**
- Database indexing
- Symbol tables in compilers
- Priority queues (with modifications)
- Expression evaluation`,
					CodeExamples: `// Go: BST search
func searchBST(root *TreeNode, val int) *TreeNode {
    if root == nil || root.Val == val {
        return root
    }
    if val < root.Val {
        return searchBST(root.Left, val)
    }
    return searchBST(root.Right, val)
}

# Python: BST search
def searchBST(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return searchBST(root.left, val)
    return searchBST(root.right, val)`,
				},
				{
					Title: "Tree Traversal Techniques",
					Content: `Master different tree traversal methods and when to use each.

**Iterative Traversals:**

All recursive traversals can be converted to iterative using explicit stack:

1. **Preorder Iterative**:
   - Push root to stack
   - While stack not empty: pop, process, push right then left

2. **Inorder Iterative**:
   - Use stack to simulate recursion
   - Go left as far as possible, then process, then go right

3. **Postorder Iterative**:
   - Most complex - need to track if node processed
   - Use two stacks or modified preorder

**Level-order (BFS) Traversal:**
- Use queue instead of stack
- Process nodes level by level
- Natural for: finding level-specific info, shortest path

**Morris Traversal:**
- Inorder traversal with O(1) space (no stack)
- Uses threaded binary tree concept
- Advanced technique for space-constrained scenarios

**Common Tree Problems:**
- Maximum depth/height
- Symmetric tree checking
- Path sum problems
- Lowest Common Ancestor (LCA)
- Serialization/Deserialization`,
					CodeExamples: `// Go: Iterative preorder
func preorderIterative(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }
    stack := []*TreeNode{root}
    result := []int{}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        result = append(result, node.Val)
        
        if node.Right != nil {
            stack = append(stack, node.Right)
        }
        if node.Left != nil {
            stack = append(stack, node.Left)
        }
    }
    return result
}

// Level-order traversal
func levelOrder(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    queue := []*TreeNode{root}
    result := [][]int{}
    
    for len(queue) > 0 {
        levelSize := len(queue)
        level := []int{}
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            level = append(level, node.Val)
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        result = append(result, level)
    }
    return result
}

# Python: Iterative preorder
def preorder_iterative(root):
    if not root:
        return []
    stack = [root]
    result = []
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result`,
				},
				{
					Title: "Tree Construction",
					Content: `Building trees from various inputs is a common problem type. Understanding how to construct trees from traversal sequences is essential.

**Common Construction Problems:**

**1. Build Tree from Preorder and Inorder:**
- Preorder gives root first, then left subtree, then right subtree
- Inorder gives left subtree, then root, then right subtree
- Use preorder to find root, use inorder to split into left/right subtrees
- Recursively build subtrees

**2. Build Tree from Postorder and Inorder:**
- Postorder gives left subtree, then right subtree, then root
- Similar approach: use postorder to find root (from end), use inorder to split
- Process postorder from right to left

**3. Build Tree from Preorder and Postorder:**
- More challenging - need to identify subtree boundaries
- Use preorder to find root, find next root in postorder to identify left subtree end

**4. Build Balanced BST from Sorted Array:**
- Middle element is root
- Recursively build left subtree from left half, right subtree from right half
- Results in balanced BST

**5. Build Tree from Level-order:**
- Use queue to process nodes level by level
- Connect children as you process each node

**Key Techniques:**
- Use hash map to quickly find positions in inorder array
- Recursively build left and right subtrees
- Handle edge cases: empty arrays, single element
- Track indices carefully to avoid out-of-bounds errors`,
					CodeExamples: `// Go: Build tree from preorder and inorder
func buildTree(preorder []int, inorder []int) *TreeNode {
    // Create map for O(1) lookup
    inMap := make(map[int]int)
    for i, val := range inorder {
        inMap[val] = i
    }
    
    preIndex := 0
    return buildHelper(preorder, &preIndex, inorder, 0, len(inorder)-1, inMap)
}

func buildHelper(preorder []int, preIndex *int, inorder []int, inStart, inEnd int, inMap map[int]int) *TreeNode {
    if inStart > inEnd {
        return nil
    }
    
    rootVal := preorder[*preIndex]
    *preIndex++
    root := &TreeNode{Val: rootVal}
    
    inRoot := inMap[rootVal]
    
    root.Left = buildHelper(preorder, preIndex, inorder, inStart, inRoot-1, inMap)
    root.Right = buildHelper(preorder, preIndex, inorder, inRoot+1, inEnd, inMap)
    
    return root
}

// Go: Build balanced BST from sorted array
func sortedArrayToBST(nums []int) *TreeNode {
    if len(nums) == 0 {
        return nil
    }
    
    mid := len(nums) / 2
    root := &TreeNode{Val: nums[mid]}
    root.Left = sortedArrayToBST(nums[:mid])
    root.Right = sortedArrayToBST(nums[mid+1:])
    
    return root
}

# Python: Build tree from preorder and inorder
def build_tree(preorder, inorder):
    in_map = {val: idx for idx, val in enumerate(inorder)}
    pre_idx = [0]  # Use list to modify in recursive calls
    
    def build(in_start, in_end):
        if in_start > in_end:
            return None
        
        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        root = TreeNode(root_val)
        
        in_root = in_map[root_val]
        
        root.left = build(in_start, in_root - 1)
        root.right = build(in_root + 1, in_end)
        
        return root
    
    return build(0, len(inorder) - 1)

# Python: Build balanced BST from sorted array
def sorted_array_to_bst(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    
    return root`,
				},
				{
					Title: "Lowest Common Ancestor",
					Content: `The Lowest Common Ancestor (LCA) of two nodes in a tree is the deepest node that has both nodes as descendants. This is a fundamental tree problem with many applications.

**Problem Types:**

**1. LCA in Binary Tree:**
- No ordering property (not necessarily BST)
- Need to search both subtrees
- Return node if found in both subtrees, or return non-null result

**2. LCA in BST:**
- Can use BST property to navigate
- If both nodes < root, go left
- If both nodes > root, go right
- Otherwise, root is LCA

**3. LCA with Parent Pointers:**
- Each node has pointer to parent
- Find depth of both nodes
- Move deeper node up to same depth
- Move both up until they meet

**4. LCA of Deepest Leaves:**
- Find deepest leaves first
- Then find LCA of all deepest leaves
- Requires two passes: find depth, then find LCA

**Algorithm for Binary Tree:**
1. If root is null or equals p or q, return root
2. Recursively search left and right subtrees
3. If both subtrees return non-null, root is LCA
4. Otherwise, return non-null result (LCA is in that subtree)

**Time Complexity:** O(n) - visit each node once
**Space Complexity:** O(h) - recursion stack depth

**Applications:**
- Find distance between two nodes
- Determine relationship between nodes
- Tree queries and path problems
- Network routing algorithms`,
					CodeExamples: `// Go: LCA in Binary Tree
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil || root == p || root == q {
        return root
    }
    
    left := lowestCommonAncestor(root.Left, p, q)
    right := lowestCommonAncestor(root.Right, p, q)
    
    if left != nil && right != nil {
        return root  // p and q in different subtrees
    }
    
    if left != nil {
        return left  // Both in left subtree
    }
    return right  // Both in right subtree
}

// Go: LCA in BST (optimized)
func lowestCommonAncestorBST(root, p, q *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    // Both nodes in left subtree
    if p.Val < root.Val && q.Val < root.Val {
        return lowestCommonAncestorBST(root.Left, p, q)
    }
    
    // Both nodes in right subtree
    if p.Val > root.Val && q.Val > root.Val {
        return lowestCommonAncestorBST(root.Right, p, q)
    }
    
    // Nodes in different subtrees, root is LCA
    return root
}

// Go: LCA with parent pointers
type NodeWithParent struct {
    Val    int
    Left   *NodeWithParent
    Right  *NodeWithParent
    Parent *NodeWithParent
}

func lowestCommonAncestorWithParent(p, q *NodeWithParent) *NodeWithParent {
    // Find depth of both nodes
    depthP := getDepth(p)
    depthQ := getDepth(q)
    
    // Move deeper node up to same level
    if depthP > depthQ {
        for i := 0; i < depthP-depthQ; i++ {
            p = p.Parent
        }
    } else {
        for i := 0; i < depthQ-depthP; i++ {
            q = q.Parent
        }
    }
    
    // Move both up until they meet
    for p != q {
        p = p.Parent
        q = q.Parent
    }
    
    return p
}

func getDepth(node *NodeWithParent) int {
    depth := 0
    for node.Parent != nil {
        depth++
        node = node.Parent
    }
    return depth
}

# Python: LCA in Binary Tree
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root
    
    return left if left else right

# Python: LCA in BST
def lowest_common_ancestor_bst(root, p, q):
    if not root:
        return None
    
    if p.val < root.val and q.val < root.val:
        return lowest_common_ancestor_bst(root.left, p, q)
    
    if p.val > root.val and q.val > root.val:
        return lowest_common_ancestor_bst(root.right, p, q)
    
    return root`,
				},
				{
					Title: "Tree Serialization",
					Content: `Tree serialization converts a tree into a string representation, and deserialization reconstructs the tree from that string. This is essential for storing trees or transmitting them over networks.

**Serialization Formats:**

**1. Preorder with Null Markers:**
- Use special marker (e.g., "#" or "null") for null nodes
- Preorder traversal: root, left, right
- Example: "1,2,#,#,3,4,#,#,5,#,#"

**2. Level-order Serialization:**
- Serialize level by level
- Include null markers for missing children
- Example: "1,2,3,#,#,4,5"

**3. JSON Format:**
- Use JSON structure with nested objects
- More verbose but human-readable
- Example: {"val": 1, "left": {"val": 2}, "right": {"val": 3}}

**Deserialization:**
- Parse string back into tree structure
- Reconstruct nodes in same order as serialization
- Handle null markers correctly

**Key Considerations:**
- **Uniqueness**: Serialization should uniquely represent tree
- **Efficiency**: Compact representation saves space
- **Robustness**: Handle edge cases (empty tree, single node)
- **Parsing**: Efficient string parsing is important

**Use Cases:**
- Storing trees in databases
- Transmitting trees over network
- Caching tree structures
- Debugging tree algorithms
- Version control for tree data

**Common Problems:**
- Serialize and deserialize binary tree
- Serialize and deserialize BST (can use more efficient format)
- Serialize N-ary tree
- Compare trees using serialization`,
					CodeExamples: `// Go: Serialize tree using preorder
import (
    "strconv"
    "strings"
)

func serialize(root *TreeNode) string {
    var result []string
    serializeHelper(root, &result)
    return strings.Join(result, ",")
}

func serializeHelper(root *TreeNode, result *[]string) {
    if root == nil {
        *result = append(*result, "#")
        return
    }
    
    *result = append(*result, strconv.Itoa(root.Val))
    serializeHelper(root.Left, result)
    serializeHelper(root.Right, result)
}

// Go: Deserialize tree
func deserialize(data string) *TreeNode {
    values := strings.Split(data, ",")
    index := 0
    return deserializeHelper(values, &index)
}

func deserializeHelper(values []string, index *int) *TreeNode {
    if *index >= len(values) || values[*index] == "#" {
        (*index)++
        return nil
    }
    
    val, _ := strconv.Atoi(values[*index])
    (*index)++
    
    root := &TreeNode{Val: val}
    root.Left = deserializeHelper(values, index)
    root.Right = deserializeHelper(values, index)
    
    return root
}

# Python: Serialize tree using preorder
def serialize(root):
    result = []
    
    def serialize_helper(node):
        if not node:
            result.append("#")
            return
        result.append(str(node.val))
        serialize_helper(node.left)
        serialize_helper(node.right)
    
    serialize_helper(root)
    return ",".join(result)

# Python: Deserialize tree
def deserialize(data):
    values = data.split(",")
    index = [0]
    
    def deserialize_helper():
        if index[0] >= len(values) or values[index[0]] == "#":
            index[0] += 1
            return None
        
        val = int(values[index[0]])
        index[0] += 1
        
        root = TreeNode(val)
        root.left = deserialize_helper()
        root.right = deserialize_helper()
        
        return root
    
    return deserialize_helper()`,
				},
			},
			ProblemIDs: []int{41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
		},
		{
			ID:          8,
			Title:       "Strings",
			Description: "Master string manipulation techniques, pattern matching, and string algorithms.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "String Fundamentals",
					Content: `Strings are sequences of characters. Understanding string operations is crucial for many programming problems.

**Key Concepts:**
- Strings are immutable in many languages (Go, Python, Java)
- Character encoding (ASCII, Unicode)
- String vs character array
- Common operations: concatenation, slicing, searching

**Important Operations:**
- **Length**: O(1) in most languages
- **Access**: O(1) by index
- **Substring**: O(n) where n is substring length
- **Search**: O(n) for single character, O(n×m) for pattern matching

**Common Patterns:**
- Two pointers for palindromes
- Sliding window for substrings
- Character frequency counting
- String building (use StringBuilder in Java, list + join in Python)`,
					CodeExamples: `// Go: String operations
s := "hello"
length := len(s)  // O(1)
char := s[0]  // O(1) - byte access
substr := s[1:3]  // O(n) - creates new string

// String building (efficient)
var builder strings.Builder
for i := 0; i < 10; i++ {
    builder.WriteString("a")
}
result := builder.String()  // O(n) total

# Python: String operations
s = "hello"
length = len(s)  # O(1)
char = s[0]  # O(1)
substr = s[1:3]  # O(n) - creates new string

# String building (efficient)
result = ''.join(['a' for _ in range(10)])  # O(n) total`,
				},
				{
					Title: "String Pattern Matching",
					Content: `Pattern matching is finding occurrences of a pattern in a text string.

**Naive Approach:**
- Check every position: O(n×m) where n=text length, m=pattern length
- Simple but inefficient for long strings

**Efficient Algorithms:**
1. **KMP (Knuth-Morris-Pratt)**: O(n + m)
   - Uses failure function to avoid re-checking characters
   - Preprocesses pattern to build skip table

2. **Rabin-Karp**: O(n + m) average, O(n×m) worst
   - Uses rolling hash
   - Good for multiple pattern matching

3. **Boyer-Moore**: O(n/m) best case, O(n×m) worst
   - Skips characters using bad character rule
   - Efficient for long patterns

**When to Use:**
- Simple problems: Naive approach is fine
- Interview problems: Usually expect O(n) solution
- Production: Use built-in functions (strings.Contains, regex)`,
					CodeExamples: `// Go: Naive pattern matching
func containsPattern(text, pattern string) bool {
    for i := 0; i <= len(text)-len(pattern); i++ {
        match := true
        for j := 0; j < len(pattern); j++ {
            if text[i+j] != pattern[j] {
                match = false
                break
            }
        }
        if match {
            return true
        }
    }
    return false
}

# Python: Using built-in (highly optimized)
def contains_pattern(text, pattern):
    return pattern in text  # Uses optimized algorithm

# Manual implementation
def contains_pattern_manual(text, pattern):
    for i in range(len(text) - len(pattern) + 1):
        if text[i:i+len(pattern)] == pattern:
            return True
    return False`,
				},
				{
					Title: "String Manipulation Techniques",
					Content: `Common string manipulation techniques for solving problems:

**1. Character Frequency Counting:**
- Use hash map to count occurrences
- Useful for anagrams, character replacement

**2. Two Pointers:**
- Check palindromes
- Reverse strings
- Remove duplicates

**3. Sliding Window:**
- Longest substring without repeating characters
- Minimum window substring

**4. String Building:**
- Build strings efficiently (avoid concatenation in loops)
- Use StringBuilder or list + join

**5. Character Mapping:**
- Map characters to indices (a→0, b→1, etc.)
- Useful for frequency arrays

**Common Problems:**
- Valid anagrams
- Longest palindromic substring
- String compression
- String matching`,
					CodeExamples: `// Go: Character frequency
func countChars(s string) map[rune]int {
    freq := make(map[rune]int)
    for _, char := range s {
        freq[char]++
    }
    return freq
}

// Two pointers: Check palindrome
func isPalindrome(s string) bool {
    left, right := 0, len(s)-1
    for left < right {
        if s[left] != s[right] {
            return false
        }
        left++
        right--
    }
    return true
}

# Python: Character frequency
def count_chars(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Two pointers: Check palindrome
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True`,
				},
				{
					Title: "String Matching Algorithms",
					Content: `Efficient string matching algorithms are crucial for text processing, search engines, and pattern recognition. Understanding these algorithms helps solve complex string problems.

**Algorithm Comparison:**

**1. Naive Algorithm:**
- Check pattern at every position
- Time: O(n×m) where n=text length, m=pattern length
- Simple but inefficient

**2. KMP (Knuth-Morris-Pratt):**
- Preprocesses pattern to build failure function (LPS array)
- Uses information from previous matches to skip positions
- Time: O(n + m) - linear time!
- Space: O(m) for LPS array

**3. Rabin-Karp:**
- Uses rolling hash to compare substrings
- Hash text substrings and compare with pattern hash
- Time: O(n + m) average, O(n×m) worst (hash collisions)
- Good for multiple pattern matching

**4. Boyer-Moore:**
- Scans pattern from right to left
- Uses bad character rule and good suffix rule to skip
- Time: O(n/m) best case, O(n×m) worst
- Very efficient for long patterns

**5. Z-Algorithm:**
- Builds Z-array: Z[i] = longest substring starting at i that matches prefix
- Used for pattern matching and string problems
- Time: O(n + m)

**When to Use:**
- **Naive**: Short strings, simple problems
- **KMP**: General pattern matching, guaranteed O(n+m)
- **Rabin-Karp**: Multiple patterns, plagiarism detection
- **Boyer-Moore**: Long patterns, text editors
- **Built-in**: Production code (highly optimized)`,
					CodeExamples: `// Go: KMP Algorithm
func kmpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 {
        return []int{}
    }
    
    // Build LPS (Longest Proper Prefix which is also Suffix) array
    lps := buildLPS(pattern)
    
    result := []int{}
    i, j := 0, 0  // i for text, j for pattern
    
    for i < n {
        if text[i] == pattern[j] {
            i++
            j++
        }
        
        if j == m {
            result = append(result, i-j)  // Pattern found
            j = lps[j-1]
        } else if i < n && text[i] != pattern[j] {
            if j != 0 {
                j = lps[j-1]
            } else {
                i++
            }
        }
    }
    
    return result
}

func buildLPS(pattern string) []int {
    m := len(pattern)
    lps := make([]int, m)
    length := 0
    i := 1
    
    for i < m {
        if pattern[i] == pattern[length] {
            length++
            lps[i] = length
            i++
        } else {
            if length != 0 {
                length = lps[length-1]
            } else {
                lps[i] = 0
                i++
            }
        }
    }
    
    return lps
}

// Go: Rabin-Karp Algorithm
func rabinKarp(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 || n < m {
        return []int{}
    }
    
    const base = 256
    const mod = 101  // Prime number
    
    // Calculate hash of pattern and first window
    patternHash := 0
    textHash := 0
    h := 1
    
    for i := 0; i < m-1; i++ {
        h = (h * base) % mod
    }
    
    for i := 0; i < m; i++ {
        patternHash = (base*patternHash + int(pattern[i])) % mod
        textHash = (base*textHash + int(text[i])) % mod
    }
    
    result := []int{}
    
    for i := 0; i <= n-m; i++ {
        if patternHash == textHash {
            // Verify match (hash collision check)
            match := true
            for j := 0; j < m; j++ {
                if text[i+j] != pattern[j] {
                    match = false
                    break
                }
            }
            if match {
                result = append(result, i)
            }
        }
        
        // Calculate hash for next window
        if i < n-m {
            textHash = (base*(textHash-int(text[i])*h) + int(text[i+m])) % mod
            if textHash < 0 {
                textHash += mod
            }
        }
    }
    
    return result
}

# Python: KMP Algorithm
def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    
    lps = build_lps(pattern)
    result = []
    i = j = 0
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result

def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps`,
				},
				{
					Title: "Trie Data Structure",
					Content: `A Trie (prefix tree) is a tree-like data structure that stores strings efficiently and enables fast prefix-based searches. It's essential for autocomplete, spell checkers, and IP routing.

**Trie Properties:**
- Each node represents a character
- Path from root to node represents a string
- Common prefixes share nodes (space efficient)
- Fast prefix search: O(m) where m is prefix length

**Operations:**
- **Insert**: Add string to trie - O(m) where m is string length
- **Search**: Check if string exists - O(m)
- **StartsWith**: Check if prefix exists - O(m)
- **Delete**: Remove string - O(m)

**Node Structure:**
- Character value (or implicit from position)
- Children array/map (one per possible character)
- Boolean flag: isEndOfWord
- Optional: count/frequency for word frequency

**Advantages:**
- Fast prefix searches
- Space efficient for common prefixes
- Easy to implement autocomplete
- Good for dictionary/word problems

**Disadvantages:**
- Can use significant memory (especially with large alphabets)
- Slower than hash table for exact matches
- More complex than simple data structures

**Applications:**
- Autocomplete systems
- Spell checkers
- IP routing (longest prefix matching)
- Word games (Scrabble, Boggle)
- Search engines (prefix matching)
- Contact list search

**Variations:**
- **Compressed Trie**: Merge single-child nodes
- **Suffix Trie**: Store all suffixes (for suffix matching)
- **Ternary Search Trie**: More memory efficient`,
					CodeExamples: `// Go: Trie Implementation
type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
}

type Trie struct {
    root *TrieNode
}

func NewTrie() *Trie {
    return &Trie{root: &TrieNode{children: make(map[rune]*TrieNode)}}
}

func (t *Trie) Insert(word string) {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            node.children[char] = &TrieNode{children: make(map[rune]*TrieNode)}
        }
        node = node.children[char]
    }
    node.isEnd = true
}

func (t *Trie) Search(word string) bool {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return node.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
    node := t.root
    for _, char := range prefix {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return true
}

// Go: Word Search using Trie
func findWords(board [][]byte, words []string) []string {
    trie := NewTrie()
    for _, word := range words {
        trie.Insert(word)
    }
    
    result := []string{}
    visited := make([][]bool, len(board))
    for i := range visited {
        visited[i] = make([]bool, len(board[0]))
    }
    
    var dfs func(int, int, *TrieNode, string)
    dfs = func(i, j int, node *TrieNode, path string) {
        if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || visited[i][j] {
            return
        }
        
        char := rune(board[i][j])
        if node.children[char] == nil {
            return
        }
        
        node = node.children[char]
        path += string(char)
        
        if node.isEnd {
            result = append(result, path)
            node.isEnd = false  // Avoid duplicates
        }
        
        visited[i][j] = true
        dfs(i+1, j, node, path)
        dfs(i-1, j, node, path)
        dfs(i, j+1, node, path)
        dfs(i, j-1, node, path)
        visited[i][j] = false
    }
    
    for i := 0; i < len(board); i++ {
        for j := 0; j < len(board[0]); j++ {
            dfs(i, j, trie.root, "")
        }
    }
    
    return result
}

# Python: Trie Implementation
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True`,
				},
				{
					Title: "Suffix Arrays",
					Content: `Suffix arrays are powerful data structures for string processing that enable efficient substring searches and various string algorithms.

**What is a Suffix Array?**
- Array containing all suffixes of a string in sorted order
- Each element is starting index of a suffix
- Enables binary search for substring matching

**Construction:**
- Generate all suffixes: O(n) suffixes, each O(n) length
- Sort suffixes: O(n log n) comparisons, each O(n) time
- Total: O(n² log n) naive, O(n log n) optimized

**Applications:**

**1. Substring Search:**
- Binary search on suffix array
- Find all occurrences of pattern
- Time: O(m log n) where m=pattern length, n=text length

**2. Longest Common Substring:**
- Compare adjacent suffixes in sorted order
- Find longest common prefix between adjacent suffixes

**3. Longest Repeated Substring:**
- Find longest substring that appears at least twice
- Compare adjacent suffixes

**4. Suffix Array vs Suffix Tree:**
- Suffix Array: Simpler, less memory, slightly slower
- Suffix Tree: Faster queries, more memory, more complex

**Key Operations:**
- **Build**: Construct suffix array from string
- **Search**: Find pattern using binary search
- **LCP Array**: Longest Common Prefix array for optimizations

**LCP Array:**
- LCP[i] = length of longest common prefix between SA[i] and SA[i-1]
- Enables faster substring searches
- Can be computed in O(n) time`,
					CodeExamples: `// Go: Suffix Array Construction
func buildSuffixArray(s string) []int {
    n := len(s)
    suffixes := make([][2]int, n)  // [index, rank]
    
    // Initial ranks (characters)
    for i := 0; i < n; i++ {
        suffixes[i] = [2]int{i, int(s[i])}
    }
    
    // Sort by rank
    sort.Slice(suffixes, func(i, j int) bool {
        return suffixes[i][1] < suffixes[j][1]
    })
    
    // Assign new ranks
    rank := make([]int, n)
    rank[suffixes[0][0]] = 0
    for i := 1; i < n; i++ {
        if suffixes[i][1] == suffixes[i-1][1] {
            rank[suffixes[i][0]] = rank[suffixes[i-1][0]]
        } else {
            rank[suffixes[i][0]] = i
        }
    }
    
    // Build suffix array indices
    sa := make([]int, n)
    for i := 0; i < n; i++ {
        sa[i] = suffixes[i][0]
    }
    
    return sa
}

// Go: Search pattern using suffix array
func searchPattern(text string, sa []int, pattern string) []int {
    n := len(text)
    m := len(pattern)
    result := []int{}
    
    // Binary search for pattern
    left, right := 0, n-1
    start, end := -1, -1
    
    // Find left boundary
    for left <= right {
        mid := (left + right) / 2
        suffix := text[sa[mid]:]
        cmp := comparePrefix(suffix, pattern)
        
        if cmp == 0 {
            start = mid
            right = mid - 1
        } else if cmp < 0 {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    // Find right boundary
    left, right = 0, n-1
    for left <= right {
        mid := (left + right) / 2
        suffix := text[sa[mid]:]
        cmp := comparePrefix(suffix, pattern)
        
        if cmp == 0 {
            end = mid
            left = mid + 1
        } else if cmp < 0 {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    // Collect all occurrences
    if start != -1 && end != -1 {
        for i := start; i <= end; i++ {
            result = append(result, sa[i])
        }
    }
    
    return result
}

func comparePrefix(s, pattern string) int {
    minLen := len(s)
    if len(pattern) < minLen {
        minLen = len(pattern)
    }
    
    for i := 0; i < minLen; i++ {
        if s[i] < pattern[i] {
            return -1
        } else if s[i] > pattern[i] {
            return 1
        }
    }
    
    if len(s) < len(pattern) {
        return -1
    } else if len(s) > len(pattern) {
        return 1
    }
    return 0
}

# Python: Suffix Array Construction
def build_suffix_array(s):
    n = len(s)
    suffixes = [(i, s[i:]) for i in range(n)]
    suffixes.sort(key=lambda x: x[1])
    return [idx for idx, _ in suffixes]

# Python: Search pattern using suffix array
def search_pattern(text, sa, pattern):
    n = len(text)
    result = []
    
    # Binary search for left boundary
    left, right = 0, n - 1
    start = -1
    
    while left <= right:
        mid = (left + right) // 2
        suffix = text[sa[mid]:]
        if suffix.startswith(pattern):
            start = mid
            right = mid - 1
        elif suffix < pattern:
            left = mid + 1
        else:
            right = mid - 1
    
    # Binary search for right boundary
    left, right = 0, n - 1
    end = -1
    
    while left <= right:
        mid = (left + right) // 2
        suffix = text[sa[mid]:]
        if suffix.startswith(pattern):
            end = mid
            left = mid + 1
        elif suffix < pattern:
            left = mid + 1
        else:
            right = mid - 1
    
    if start != -1 and end != -1:
        return [sa[i] for i in range(start, end + 1)]
    return []`,
				},
			},
			ProblemIDs: []int{2, 4, 15, 27, 30},
		},
		{
			ID:          9,
			Title:       "Graphs",
			Description: "Learn graph data structures, traversal algorithms, and graph-based problem solving.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Graph Fundamentals",
					Content: `A graph is a collection of nodes (vertices) connected by edges. Graphs are the most general data structure, modeling any relationship or network.

**Graph Types:**

1. **Directed Graph (Digraph)**:
   - Edges have direction: A→B means edge from A to B
   - A→B ≠ B→A (unless both edges exist)
   - Examples: Web links, dependencies, social media follows

2. **Undirected Graph**:
   - Edges have no direction: A-B means bidirectional connection
   - A-B = B-A
   - Examples: Social networks (friends), road networks

3. **Weighted Graph**:
   - Edges have weights/costs
   - Can represent distance, cost, time, etc.
   - Examples: Maps with distances, network latency

4. **Unweighted Graph**:
   - All edges equal (weight = 1)
   - Simpler, used when only connectivity matters

**Graph Representations:**

1. **Adjacency List** (Most common):
   - List of lists: graph[i] = [neighbors of node i]
   - **Space**: O(V + E) - efficient for sparse graphs
   - **Edge Lookup**: O(degree) - check neighbors
   - **Add Edge**: O(1) - append to list
   - **Best for**: Sparse graphs, most algorithms

2. **Adjacency Matrix**:
   - 2D array: matrix[i][j] = 1 if edge exists, 0 otherwise
   - **Space**: O(V²) - can be wasteful for sparse graphs
   - **Edge Lookup**: O(1) - direct array access
   - **Add Edge**: O(1) - set matrix[i][j] = 1
   - **Best for**: Dense graphs, frequent edge queries

**Key Terminology:**
- **Vertex/Node**: A point in the graph
- **Edge**: Connection between vertices
- **Degree**: Number of edges connected to vertex
  - In-degree: incoming edges (directed)
  - Out-degree: outgoing edges (directed)
- **Path**: Sequence of connected vertices
- **Cycle**: Path that starts and ends at same vertex
- **Connected**: All vertices reachable from each other
- **Component**: Maximal connected subgraph
- **Tree**: Connected graph with no cycles (n-1 edges for n vertices)
- **Forest**: Collection of trees

**Graph Properties:**
- **Sparse**: Few edges relative to vertices (E ≈ V)
- **Dense**: Many edges (E ≈ V²)
- **Acyclic**: No cycles
- **Connected**: All vertices reachable
- **Bipartite**: Can partition vertices into two sets with no edges within sets`,
					CodeExamples: `// Go: Graph representation (adjacency list)
type Graph struct {
    vertices int
    adj      [][]int
}

func NewGraph(vertices int) *Graph {
    return &Graph{
        vertices: vertices,
        adj:      make([][]int, vertices),
    }
}

func (g *Graph) AddEdge(u, v int) {
    g.adj[u] = append(g.adj[u], v)
    // For undirected: g.adj[v] = append(g.adj[v], u)
}

# Python: Graph representation
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v):
        self.adj[u].append(v)
        # For undirected: self.adj[v].append(u)`,
				},
				{
					Title: "Graph Traversal: BFS",
					Content: `Breadth-First Search (BFS) explores a graph level by level, visiting all neighbors before moving to next level. It's like ripples spreading from a stone dropped in water.

**BFS Algorithm:**
1. Start from source node, mark as visited
2. Add source to queue
3. While queue not empty:
   - Dequeue node
   - Process node
   - Add all unvisited neighbors to queue
   - Mark neighbors as visited

**Key Properties:**
- **Shortest Path**: Finds shortest path in unweighted graphs (minimum edges)
- **Level Order**: Visits nodes level by level (distance 0, 1, 2, ...)
- **Queue-based**: Uses FIFO queue to maintain level order
- **Complete**: Visits all reachable nodes from source
- **Time**: O(V + E) - visit each vertex and edge once
- **Space**: O(V) - queue and visited array

**BFS Applications:**

1. **Shortest Path** (Unweighted):
   - Minimum steps to reach target
   - Shortest path length
   - Level of each node from source

2. **Level-order Processing**:
   - Process nodes level by level
   - Find nodes at specific distance
   - Level-order tree traversal

3. **Connected Components**:
   - Find all nodes reachable from source
   - Count connected components
   - Check connectivity

4. **Bipartite Checking**:
   - Color nodes alternately
   - Check if graph is bipartite

**BFS vs DFS:**
- **BFS**: Finds shortest path, uses more memory (queue)
- **DFS**: Uses less memory (stack), doesn't guarantee shortest path`,
					CodeExamples: `// Go: BFS implementation
func BFS(graph [][]int, start int) []int {
    visited := make([]bool, len(graph))
    queue := []int{start}
    visited[start] = true
    result := []int{}
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        result = append(result, node)
        
        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                visited[neighbor] = true
                queue = append(queue, neighbor)
            }
        }
    }
    return result
}

# Python: BFS implementation
from collections import deque

def bfs(graph, start):
    visited = [False] * len(graph)
    queue = deque([start])
    visited[start] = True
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return result`,
				},
				{
					Title: "Graph Traversal: DFS",
					Content: `Depth-First Search (DFS) explores as far as possible along each branch before backtracking. Like exploring a maze - go as deep as possible before trying another path.

**DFS Algorithm:**
1. Start from source node, mark as visited
2. For each unvisited neighbor:
   - Recursively visit neighbor
   - Or use explicit stack
3. Backtrack when no more unvisited neighbors

**Key Properties:**
- **Stack-based**: Uses LIFO stack (naturally recursive)
- **Deep Exploration**: Explores one path completely before others
- **Memory Efficient**: Only stores path from root to current node
- **Time**: O(V + E) - visit each vertex and edge once
- **Space**: O(V) for recursion stack (or explicit stack)

**DFS Variants:**

1. **Pre-order**: Process node before children
2. **In-order**: Process node between children (trees)
3. **Post-order**: Process node after children

**DFS Applications:**

1. **Path Finding**:
   - Find any path from source to target
   - Check if path exists
   - Enumerate all paths

2. **Cycle Detection**:
   - Detect cycles in directed/undirected graphs
   - Use color marking (white/gray/black)

3. **Topological Sort**:
   - Order nodes such that dependencies come first
   - Used in: build systems, course prerequisites

4. **Connected Components**:
   - Find all nodes in component
   - Count components
   - Check connectivity

5. **Strongly Connected Components**:
   - Kosaraju's or Tarjan's algorithm
   - Find SCCs in directed graphs

**DFS vs BFS:**
- **DFS**: Less memory, doesn't guarantee shortest path
- **BFS**: More memory, guarantees shortest path (unweighted)

**Common Patterns:**
- **Backtracking**: Use DFS with state restoration
- **Memoization**: Cache results during DFS
- **Path Tracking**: Maintain path during traversal`,
					CodeExamples: `// Go: DFS recursive
func DFS(graph [][]int, start int) []int {
    visited := make([]bool, len(graph))
    result := []int{}
    dfsHelper(graph, start, visited, &result)
    return result
}

func dfsHelper(graph [][]int, node int, visited []bool, result *[]int) {
    visited[node] = true
    *result = append(*result, node)
    
    for _, neighbor := range graph[node] {
        if !visited[neighbor] {
            dfsHelper(graph, neighbor, visited, result)
        }
    }
}

# Python: DFS recursive
def dfs(graph, start):
    visited = [False] * len(graph)
    result = []
    
    def dfs_helper(node):
        visited[node] = True
        result.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs_helper(neighbor)
    
    dfs_helper(start)
    return result`,
				},
				{
					Title: "Common Graph Algorithms",
					Content: `Essential graph algorithms for problem-solving:

**1. Shortest Path:**
- **BFS**: Unweighted graphs, O(V + E)
- **Dijkstra**: Weighted graphs, non-negative weights, O((V + E) log V)
- **Bellman-Ford**: Handles negative weights, O(V × E)

**2. Minimum Spanning Tree:**
- **Kruskal**: Greedy, O(E log E)
- **Prim**: Greedy, O(E log V)

**3. Topological Sort:**
- Order nodes such that all dependencies come before dependents
- Uses DFS, O(V + E)

**4. Cycle Detection:**
- **Undirected**: DFS with parent tracking
- **Directed**: DFS with color marking (white/gray/black)

**5. Connected Components:**
- Use DFS/BFS to find all connected nodes
- Important for social networks, islands problems`,
					CodeExamples: `// Go: Topological sort
func topologicalSort(graph [][]int) []int {
    visited := make([]bool, len(graph))
    result := []int{}
    
    var dfs func(int)
    dfs = func(node int) {
        visited[node] = true
        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                dfs(neighbor)
            }
        }
        result = append([]int{node}, result...)  // Prepend
    }
    
    for i := 0; i < len(graph); i++ {
        if !visited[i] {
            dfs(i)
        }
    }
    return result
}

# Python: Cycle detection (undirected)
def has_cycle(graph):
    visited = [False] * len(graph)
    
    def dfs(node, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # Back edge found
        return False
    
    for i in range(len(graph)):
        if not visited[i]:
            if dfs(i, -1):
                return True
    return False`,
				},
				{
					Title: "Shortest Path Algorithms",
					Content: `Finding shortest paths in graphs is fundamental to many applications. Different algorithms handle different graph types and constraints.

**Algorithm Types:**

**1. BFS (Unweighted Graphs):**
- Finds shortest path in terms of number of edges
- Time: O(V + E)
- Use when: All edges have same weight (or unweighted)

**2. Dijkstra's Algorithm (Non-negative Weights):**
- Finds shortest path from source to all nodes
- Greedy algorithm using priority queue
- Time: O((V + E) log V) with binary heap
- Use when: Non-negative edge weights

**3. Bellman-Ford Algorithm (Negative Weights):**
- Handles negative edge weights
- Can detect negative cycles
- Time: O(V × E)
- Use when: Negative weights possible, need cycle detection

**4. Floyd-Warshall (All Pairs):**
- Finds shortest paths between all pairs of nodes
- Dynamic programming approach
- Time: O(V³)
- Use when: Need all-pairs shortest paths

**5. A* Algorithm (Heuristic Search):**
- Uses heuristic to guide search
- More efficient than Dijkstra for specific targets
- Time: Depends on heuristic quality

**Key Concepts:**
- **Relaxation**: Update distance if shorter path found
- **Priority Queue**: Always process closest unvisited node
- **Negative Cycles**: Make shortest path undefined
- **Path Reconstruction**: Store parent pointers to reconstruct path`,
					CodeExamples: `// Go: Dijkstra's Algorithm
import "container/heap"

type Item struct {
    node     int
    distance int
}

type MinHeap []Item

func (h MinHeap) Len() int { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h MinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(Item)) }
func (h *MinHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0:n-1]
    return x
}

func dijkstra(graph [][]int, weights [][]int, start int) []int {
    n := len(graph)
    dist := make([]int, n)
    for i := range dist {
        dist[i] = int(^uint(0) >> 1)  // Max int
    }
    dist[start] = 0
    
    h := &MinHeap{Item{start, 0}}
    heap.Init(h)
    visited := make([]bool, n)
    
    for h.Len() > 0 {
        item := heap.Pop(h).(Item)
        u := item.node
        
        if visited[u] {
            continue
        }
        visited[u] = true
        
        for _, v := range graph[u] {
            if !visited[v] {
                newDist := dist[u] + weights[u][v]
                if newDist < dist[v] {
                    dist[v] = newDist
                    heap.Push(h, Item{v, dist[v]})
                }
            }
        }
    }
    
    return dist
}

// Go: Bellman-Ford Algorithm
func bellmanFord(edges [][]int, n int, start int) []int {
    dist := make([]int, n)
    for i := range dist {
        dist[i] = int(^uint(0) >> 1)
    }
    dist[start] = 0
    
    // Relax edges V-1 times
    for i := 0; i < n-1; i++ {
        for _, edge := range edges {
            u, v, w := edge[0], edge[1], edge[2]
            if dist[u] != int(^uint(0)>>1) && dist[u]+w < dist[v] {
                dist[v] = dist[u] + w
            }
        }
    }
    
    // Check for negative cycles
    for _, edge := range edges {
        u, v, w := edge[0], edge[1], edge[2]
        if dist[u] != int(^uint(0)>>1) && dist[u]+w < dist[v] {
            // Negative cycle detected
            return nil
        }
    }
    
    return dist
}

# Python: Dijkstra's Algorithm
import heapq

def dijkstra(graph, weights, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    
    heap = [(0, start)]
    visited = [False] * n
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if visited[u]:
            continue
        visited[u] = True
        
        for v in graph[u]:
            if not visited[v]:
                new_dist = dist[u] + weights[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(heap, (dist[v], v))
    
    return dist

# Python: Bellman-Ford
def bellman_ford(edges, n, start):
    dist = [float('inf')] * n
    dist[start] = 0
    
    # Relax V-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle
    
    return dist`,
				},
				{
					Title: "Union-Find",
					Content: `Union-Find (also called Disjoint Set Union, DSU) is a data structure that efficiently tracks disjoint sets and supports union and find operations.

**Operations:**
- **Find(x)**: Find representative (root) of set containing x
- **Union(x, y)**: Merge sets containing x and y
- **Connected(x, y)**: Check if x and y are in same set

**Data Structure:**
- Each element points to its parent
- Root element points to itself
- Path compression: Make all nodes point directly to root
- Union by rank: Attach smaller tree under larger tree

**Optimizations:**

**1. Path Compression:**
- When finding root, make all nodes on path point directly to root
- Amortizes to nearly O(1) per operation
- Makes tree flat

**2. Union by Rank:**
- Keep track of tree depth (rank)
- Always attach smaller tree to larger tree
- Prevents tree from becoming too deep

**Time Complexity:**
- Without optimizations: O(n) worst case
- With path compression: O(α(n)) amortized (inverse Ackermann, effectively constant)
- With union by rank: O(α(n)) amortized

**Applications:**
- **Connected Components**: Find all connected components in graph
- **Kruskal's Algorithm**: Minimum spanning tree
- **Network Connectivity**: Check if nodes are connected
- **Image Processing**: Connected component labeling
- **Equivalence Relations**: Partition elements into equivalence classes

**Common Problems:**
- Number of islands
- Friend circles
- Redundant connections
- Accounts merge
- Sentence similarity`,
					CodeExamples: `// Go: Union-Find with path compression and union by rank
type UnionFind struct {
    parent []int
    rank   []int
}

func NewUnionFind(n int) *UnionFind {
    parent := make([]int, n)
    rank := make([]int, n)
    for i := range parent {
        parent[i] = i
    }
    return &UnionFind{parent: parent, rank: rank}
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        uf.parent[x] = uf.Find(uf.parent[x])  // Path compression
    }
    return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
    rootX := uf.Find(x)
    rootY := uf.Find(y)
    
    if rootX == rootY {
        return  // Already in same set
    }
    
    // Union by rank
    if uf.rank[rootX] < uf.rank[rootY] {
        uf.parent[rootX] = rootY
    } else if uf.rank[rootX] > uf.rank[rootY] {
        uf.parent[rootY] = rootX
    } else {
        uf.parent[rootY] = rootX
        uf.rank[rootX]++
    }
}

func (uf *UnionFind) Connected(x, y int) bool {
    return uf.Find(x) == uf.Find(y)
}

// Example: Number of connected components
func numConnectedComponents(n int, edges [][]int) int {
    uf := NewUnionFind(n)
    for _, edge := range edges {
        uf.Union(edge[0], edge[1])
    }
    
    components := make(map[int]bool)
    for i := 0; i < n; i++ {
        components[uf.Find(i)] = true
    }
    
    return len(components)
}

# Python: Union-Find
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Example: Number of connected components
def num_connected_components(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    
    components = set()
    for i in range(n):
        components.add(uf.find(i))
    
    return len(components)`,
				},
				{
					Title: "Topological Sort Applications",
					Content: `Topological sort orders nodes in a directed acyclic graph (DAG) such that for every edge (u, v), u appears before v. This has many practical applications.

**Key Properties:**
- Only works on DAGs (Directed Acyclic Graphs)
- Multiple valid topological orders may exist
- If cycle exists, topological sort is impossible

**Algorithms:**

**1. DFS-based (Kahn's Algorithm):**
- Use DFS with post-order processing
- Add node to result after processing all descendants
- Time: O(V + E)

**2. BFS-based (Kahn's Algorithm):**
- Start with nodes having in-degree 0
- Remove edges, add nodes with new in-degree 0
- Time: O(V + E)

**Applications:**

**1. Course Prerequisites:**
- Order courses so prerequisites come first
- Detect circular dependencies

**2. Build Systems:**
- Determine build order for dependencies
- Compile files in correct order

**3. Task Scheduling:**
- Schedule tasks respecting dependencies
- Find valid execution order

**4. Event Ordering:**
- Order events based on dependencies
- Determine chronological order

**5. Package Management:**
- Install packages in dependency order
- Resolve dependency conflicts

**6. Compiler Design:**
- Order symbol definitions
- Resolve forward references

**Common Problems:**
- Course schedule
- Alien dictionary
- Reconstruct itinerary
- Sequence reconstruction
- Minimum height trees`,
					CodeExamples: `// Go: Topological Sort using DFS
func topologicalSortDFS(graph [][]int) []int {
    n := len(graph)
    visited := make([]bool, n)
    result := []int{}
    
    var dfs func(int)
    dfs = func(node int) {
        visited[node] = true
        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                dfs(neighbor)
            }
        }
        result = append([]int{node}, result...)  // Prepend
    }
    
    for i := 0; i < n; i++ {
        if !visited[i] {
            dfs(i)
        }
    }
    
    return result
}

// Go: Topological Sort using BFS (Kahn's Algorithm)
func topologicalSortBFS(graph [][]int) []int {
    n := len(graph)
    inDegree := make([]int, n)
    
    // Calculate in-degrees
    for i := 0; i < n; i++ {
        for _, neighbor := range graph[i] {
            inDegree[neighbor]++
        }
    }
    
    // Start with nodes having in-degree 0
    queue := []int{}
    for i := 0; i < n; i++ {
        if inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }
    
    result := []int{}
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        result = append(result, node)
        
        // Remove edges and update in-degrees
        for _, neighbor := range graph[node] {
            inDegree[neighbor]--
            if inDegree[neighbor] == 0 {
                queue = append(queue, neighbor)
            }
        }
    }
    
    // Check for cycle
    if len(result) != n {
        return nil  // Cycle exists
    }
    
    return result
}

# Python: Topological Sort using DFS
def topological_sort_dfs(graph):
    n = len(graph)
    visited = [False] * n
    result = []
    
    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
        result.insert(0, node)  # Prepend
    
    for i in range(n):
        if not visited[i]:
            dfs(i)
    
    return result

# Python: Topological Sort using BFS (Kahn's)
def topological_sort_bfs(graph):
    n = len(graph)
    in_degree = [0] * n
    
    # Calculate in-degrees
    for i in range(n):
        for neighbor in graph[i]:
            in_degree[neighbor] += 1
    
    # Start with nodes having in-degree 0
    queue = [i for i in range(n) if in_degree[i] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else None  # None if cycle`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
