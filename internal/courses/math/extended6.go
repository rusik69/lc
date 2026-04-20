package math

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMathModules([]problems.CourseModule{
		{
			ID:          2720,
			Title:       "Graph Theory and Discrete Probability",
			Description: "Master graph theory fundamentals, network algorithms, random graphs, Markov chains, and discrete probability applications in computing.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Graph Theory and Network Algorithms",
					Content: `Graph theory provides the mathematical foundation for network analysis, social networks, routing algorithms, and many computational problems.

**Graph Fundamentals:**

Graph G = (V, E):
  V: Set of vertices (nodes)
  E: Set of edges (connections)
  |V| = n, |E| = m

Types:
  Undirected: {u,v} ∈ E (unordered pair)
  Directed (digraph): (u,v) ∈ E (ordered pair)
  Weighted: Each edge has a weight w(e)
  Simple: No self-loops, no multi-edges
  Multigraph: Allows multiple edges between same vertices
  Bipartite: V = V₁ ∪ V₂, edges only between V₁ and V₂

Degree:
  deg(v) = number of edges incident to v
  In directed: in-degree + out-degree
  Handshaking lemma: Σ deg(v) = 2|E|
  Average degree: 2m/n

Representations:
  Adjacency matrix: A[i][j] = 1 if edge (i,j) exists
    Space: O(n²), Edge lookup: O(1)
    Symmetric for undirected graphs
    A^k[i][j] = number of length-k walks from i to j
    
  Adjacency list: List of neighbors for each vertex
    Space: O(n + m), Neighbor iteration: O(deg(v))
    Better for sparse graphs (m << n²)

**Connectivity:**

Path: Sequence of vertices with consecutive edges
Walk: Path allowing repeated vertices
Cycle: Path that starts and ends at same vertex

Connected (undirected): Path between every pair of vertices
  Connected components: Maximal connected subgraphs
  Bridge: Edge whose removal disconnects graph
  Articulation point: Vertex whose removal disconnects

Strongly connected (directed): Path in both directions between all pairs
  SCC: Maximal strongly connected subgraph
  Tarjan's or Kosaraju's algorithm: Find all SCCs in O(n+m)

**Trees:**

Tree: Connected acyclic graph
  n vertices, n-1 edges
  Unique path between any two vertices
  Removing any edge disconnects
  Adding any edge creates cycle

Spanning tree: Subgraph that is tree containing all vertices
  MST: Minimum weight spanning tree
    Kruskal's: Sort edges, add if no cycle (O(m log m))
    Prim's: Grow from vertex, add cheapest edge (O(m log n))

**Planar Graphs:**

Planar: Can be drawn without edge crossings
Euler's formula: V - E + F = 2 (for connected planar)
  F: Number of faces (including outer face)
  Corollary: E ≤ 3V - 6 (for V ≥ 3)

Kuratowski's theorem: G is planar iff it has no K₅ or K₃,₃ subdivision

**Graph Coloring:**

k-coloring: Assign one of k colors to each vertex so adjacent vertices differ
Chromatic number χ(G): Minimum k for valid coloring

Bounds:
  χ(G) ≤ Δ(G) + 1 (Δ = max degree)
  χ(G) ≤ ω(G) (ω = clique number, for perfect graphs)
  Brooks' theorem: χ(G) ≤ Δ(G) unless G is complete or odd cycle

Four Color Theorem: Every planar graph is 4-colorable

Applications:
  Register allocation: Variables to registers
  Scheduling: Conflicts to time slots
  Map coloring

**Matching:**

Matching: Set of edges with no shared vertices
Maximum matching: Largest matching
Perfect matching: Every vertex is matched

Bipartite matching:
  König's theorem: max matching = min vertex cover
  Hall's theorem: Perfect matching exists iff |N(S)| ≥ |S| for all S ⊆ V₁
  Hopcroft-Karp: O(m√n) for bipartite

**Network Flow:**

Flow network: Directed graph with source s, sink t, capacities c(e)
  Flow f(e) ≤ c(e) for all edges
  Conservation: Flow in = flow out (except s, t)

Max-flow min-cut theorem:
  Maximum flow value = minimum cut capacity
  Ford-Fulkerson: O(m × max_flow)
  Edmonds-Karp: O(nm²)
  Dinic's: O(n²m)

Applications:
  Bipartite matching via max flow
  Image segmentation
  Circulation problems

**Spectral Graph Theory:**

Adjacency matrix eigenvalues:
  Spectral radius: Largest |λ|
  Connected iff second largest |λ| < largest |λ|

Laplacian L = D - A:
  L is positive semi-definite
  Smallest eigenvalue = 0 (multiplicity = # components)
  Second smallest (Fiedler value): Algebraic connectivity
  Fiedler vector: Used for spectral clustering/partitioning

Normalized Laplacian:
  L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2)AD^(-1/2)
  Eigenvalues ∈ [0, 2]

**Random Graphs:**

Erdős–Rényi G(n, p):
  n vertices, each edge with probability p
  Expected edges: C(n,2) × p
  
  Thresholds:
    p = 1/n: Giant component emerges
    p = ln(n)/n: Graph becomes connected
    p = 1/2: Typical random graph

Properties:
  Degree distribution: Binomial → approximately Poisson
  Diameter: O(log n)
  Clustering coefficient: p (low for sparse graphs)

Small-world networks:
  High clustering + short path lengths
  Watts-Strogatz model

Scale-free networks:
  Power-law degree distribution: P(k) ~ k^(-α)
  Barabási-Albert preferential attachment
  Hub vulnerability: Robust to random failures, fragile to targeted attacks`,
					CodeExamples: `# Graph Theory and Discrete Probability

import math
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque

# ============================================================
# Graph with Analysis
# ============================================================

class Graph:
    """Undirected graph with analysis methods."""
    
    def __init__(self, n: int = 0):
        self.n = n
        self.adj: Dict[int, Set[int]] = defaultdict(set)
        self.weights: Dict[Tuple[int, int], float] = {}
    
    def add_edge(self, u: int, v: int, weight: float = 1.0):
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.weights[(u, v)] = weight
        self.weights[(v, u)] = weight
        self.n = max(self.n, u + 1, v + 1)
    
    def degree(self, v: int) -> int:
        return len(self.adj[v])
    
    def neighbors(self, v: int) -> Set[int]:
        return self.adj[v]
    
    def num_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self.adj.values()) // 2
    
    def is_connected(self) -> bool:
        if self.n == 0:
            return True
        visited = set()
        start = next(iter(self.adj))
        queue = deque([start])
        visited.add(start)
        
        while queue:
            v = queue.popleft()
            for u in self.adj[v]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        
        return len(visited) == len(self.adj)
    
    def connected_components(self) -> List[Set[int]]:
        visited: Set[int] = set()
        components = []
        
        for v in self.adj:
            if v not in visited:
                component: Set[int] = set()
                queue = deque([v])
                visited.add(v)
                
                while queue:
                    u = queue.popleft()
                    component.add(u)
                    for w in self.adj[u]:
                        if w not in visited:
                            visited.add(w)
                            queue.append(w)
                
                components.append(component)
        
        return components
    
    def shortest_path(self, source: int,
                      target: int) -> Tuple[float, List[int]]:
        """BFS shortest path (unweighted)."""
        if source == target:
            return 0, [source]
        
        visited = {source}
        queue = deque([(source, [source])])
        
        while queue:
            v, path = queue.popleft()
            for u in self.adj[v]:
                if u == target:
                    return len(path), path + [u]
                if u not in visited:
                    visited.add(u)
                    queue.append((u, path + [u]))
        
        return float('inf'), []
    
    def is_bipartite(self) -> bool:
        color: Dict[int, int] = {}
        
        for start in self.adj:
            if start in color:
                continue
            queue = deque([start])
            color[start] = 0
            
            while queue:
                v = queue.popleft()
                for u in self.adj[v]:
                    if u not in color:
                        color[u] = 1 - color[v]
                        queue.append(u)
                    elif color[u] == color[v]:
                        return False
        
        return True
    
    def has_cycle(self) -> bool:
        visited: Set[int] = set()
        
        for start in self.adj:
            if start in visited:
                continue
            stack = [(start, -1)]
            visited.add(start)
            
            while stack:
                v, parent = stack.pop()
                for u in self.adj[v]:
                    if u not in visited:
                        visited.add(u)
                        stack.append((u, v))
                    elif u != parent:
                        return True
        
        return False
    
    def bridges(self) -> List[Tuple[int, int]]:
        """Find all bridges using Tarjan's algorithm."""
        disc: Dict[int, int] = {}
        low: Dict[int, int] = {}
        result: List[Tuple[int, int]] = []
        timer = [0]
        
        def dfs(v: int, parent: int):
            disc[v] = low[v] = timer[0]
            timer[0] += 1
            
            for u in self.adj[v]:
                if u not in disc:
                    dfs(u, v)
                    low[v] = min(low[v], low[u])
                    if low[u] > disc[v]:
                        result.append((v, u))
                elif u != parent:
                    low[v] = min(low[v], disc[u])
        
        for v in self.adj:
            if v not in disc:
                dfs(v, -1)
        
        return result
    
    def clustering_coefficient(self, v: int) -> float:
        """Local clustering coefficient."""
        neighbors = list(self.adj[v])
        k = len(neighbors)
        if k < 2:
            return 0.0
        
        triangles = 0
        for i in range(k):
            for j in range(i + 1, k):
                if neighbors[j] in self.adj[neighbors[i]]:
                    triangles += 1
        
        return 2 * triangles / (k * (k - 1))
    
    def average_clustering(self) -> float:
        vertices = list(self.adj.keys())
        if not vertices:
            return 0.0
        return sum(self.clustering_coefficient(v) for v in vertices) / len(vertices)


# ============================================================
# Graph Coloring
# ============================================================

def greedy_coloring(graph: Graph) -> Dict[int, int]:
    """Greedy graph coloring."""
    colors: Dict[int, int] = {}
    
    # Order vertices by degree (descending)
    vertices = sorted(graph.adj.keys(),
                     key=lambda v: graph.degree(v), reverse=True)
    
    for v in vertices:
        neighbor_colors = {colors[u] for u in graph.adj[v] if u in colors}
        
        color = 0
        while color in neighbor_colors:
            color += 1
        
        colors[v] = color
    
    return colors


# ============================================================
# Bipartite Matching (Hungarian-like)
# ============================================================

def max_bipartite_matching(graph: Graph,
                           left: Set[int],
                           right: Set[int]) -> Dict[int, int]:
    """Maximum bipartite matching using augmenting paths."""
    match_left: Dict[int, int] = {}
    match_right: Dict[int, int] = {}
    
    def augment(u: int, visited: Set[int]) -> bool:
        for v in graph.adj[u]:
            if v not in right or v in visited:
                continue
            visited.add(v)
            
            if v not in match_right or augment(match_right[v], visited):
                match_left[u] = v
                match_right[v] = u
                return True
        
        return False
    
    for u in left:
        augment(u, set())
    
    return match_left


# ============================================================
# Markov Chains
# ============================================================

class MarkovChain:
    """Discrete-time Markov chain."""
    
    def __init__(self, transition: List[List[float]]):
        self.P = transition
        self.n = len(transition)
    
    def step(self, state: int) -> int:
        """Take one step from current state."""
        r = random.random()
        cumsum = 0.0
        for j in range(self.n):
            cumsum += self.P[state][j]
            if r <= cumsum:
                return j
        return self.n - 1
    
    def simulate(self, start: int, steps: int) -> List[int]:
        """Simulate chain for given number of steps."""
        path = [start]
        state = start
        for _ in range(steps):
            state = self.step(state)
            path.append(state)
        return path
    
    def stationary_distribution(self,
                                 max_iters: int = 1000,
                                 tol: float = 1e-8) -> List[float]:
        """Find stationary distribution by power iteration."""
        pi = [1.0 / self.n] * self.n
        
        for _ in range(max_iters):
            new_pi = [0.0] * self.n
            for j in range(self.n):
                for i in range(self.n):
                    new_pi[j] += pi[i] * self.P[i][j]
            
            diff = sum(abs(new_pi[i] - pi[i]) for i in range(self.n))
            pi = new_pi
            
            if diff < tol:
                break
        
        return pi
    
    def expected_hitting_time(self, source: int,
                              target: int,
                              max_iters: int = 10000) -> float:
        """Estimate expected hitting time by simulation."""
        total = 0
        n_sims = 1000
        
        for _ in range(n_sims):
            state = source
            steps = 0
            while state != target and steps < max_iters:
                state = self.step(state)
                steps += 1
            total += steps
        
        return total / n_sims
    
    def is_irreducible(self) -> bool:
        """Check if chain is irreducible (all states communicate)."""
        # BFS from each state to check reachability
        for start in range(self.n):
            visited = set()
            queue = deque([start])
            visited.add(start)
            
            while queue:
                state = queue.popleft()
                for j in range(self.n):
                    if self.P[state][j] > 0 and j not in visited:
                        visited.add(j)
                        queue.append(j)
            
            if len(visited) < self.n:
                return False
        
        return True


# ============================================================
# Random Graph Generation
# ============================================================

def erdos_renyi(n: int, p: float) -> Graph:
    """Generate Erdős-Rényi random graph G(n, p)."""
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                g.add_edge(i, j)
    return g


def barabasi_albert(n: int, m: int) -> Graph:
    """Generate Barabási-Albert preferential attachment graph."""
    g = Graph(n)
    
    # Start with complete graph on m+1 vertices
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            g.add_edge(i, j)
    
    # Add vertices with preferential attachment
    degrees = [m] * (m + 1)  # Initial degrees
    total_degree = sum(degrees)
    
    for v in range(m + 1, n):
        targets: Set[int] = set()
        
        while len(targets) < m:
            # Select target with probability proportional to degree
            r = random.random() * total_degree
            cumsum = 0.0
            for u in range(v):
                cumsum += degrees[u] if u < len(degrees) else 0
                if cumsum >= r:
                    targets.add(u)
                    break
        
        degrees.append(0)
        for u in targets:
            g.add_edge(v, u)
            degrees[v] += 1
            degrees[u] += 1
            total_degree += 2
    
    return g


def watts_strogatz(n: int, k: int, p: float) -> Graph:
    """Generate Watts-Strogatz small-world graph."""
    g = Graph(n)
    
    # Create ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            g.add_edge(i, (i + j) % n)
    
    # Rewire edges
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if random.random() < p:
                target = (i + j) % n
                # Remove edge
                g.adj[i].discard(target)
                g.adj[target].discard(i)
                
                # Add random edge
                new_target = random.randint(0, n - 1)
                while new_target == i or new_target in g.adj[i]:
                    new_target = random.randint(0, n - 1)
                
                g.add_edge(i, new_target)
    
    return g`,
				},
			},
		},
	})
}
