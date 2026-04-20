package softwarearchitecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterSoftwareArchitectureModules([]problems.CourseModule{
		{
			ID:          2325,
			Title:       "Architecture Decision Records and Documentation",
			Description: "Document architectural decisions effectively with ADRs, C4 model diagrams, architecture fitness functions, and living documentation practices.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Documenting Architecture Decisions",
					Content: `Architecture decisions shape the system for years. Documenting them preserves context and rationale for future teams.

**Architecture Decision Records (ADRs):**

An ADR captures a single architectural decision along with its context and consequences.

ADR Template:
  Title: ADR-NNN: Short descriptive title
  Date: When the decision was made
  Status: Proposed | Accepted | Deprecated | Superseded
  Context: What situation prompted the decision?
  Decision: What was decided and why?
  Consequences: What are the trade-offs?
  Alternatives Considered: What other options were evaluated?

Example ADR:
  Title: ADR-001: Use PostgreSQL as Primary Database
  Date: 2024-01-15
  Status: Accepted
  
  Context:
    We need a primary relational database for our e-commerce platform.
    Requirements: ACID compliance, JSON support, full-text search,
    strong community, and proven scalability to 10M+ records.
  
  Decision:
    Use PostgreSQL 16 as our primary relational database.
    Rationale:
    - JSONB support for flexible product attributes
    - Full-text search eliminates need for separate search engine initially
    - Strong ecosystem (pgvector for ML, PostGIS for geo)
    - Proven scalability with read replicas and partitioning
    - Team has existing PostgreSQL expertise
  
  Alternatives Considered:
    MySQL: Lacks JSONB, weaker full-text search
    MongoDB: No ACID transactions across documents (at the time)
    CockroachDB: Higher operational complexity, cost
  
  Consequences:
    Positive:
    - Flexible schema evolution with JSONB
    - Single database for relational + search initially
    - Large talent pool for hiring
    Negative:
    - Scaling writes requires careful partitioning strategy
    - May need dedicated search engine (Elasticsearch) at higher scale
    - Need to manage connection pooling (PgBouncer)

ADR Lifecycle:
  1. Propose: Author writes ADR, shares for review
  2. Discuss: Team reviews, adds comments
  3. Decide: Accept, reject, or defer
  4. Implement: Build what was decided
  5. Evaluate: Review if decision still holds
  6. Supersede: Replace with new ADR if needed

Best Practices:
  Store ADRs in version control (close to code)
  Number sequentially (ADR-001, ADR-002)
  Never delete, mark as superseded
  Link related ADRs together
  Include date and participants
  Keep them concise (1-2 pages max)

**C4 Model for Architecture Diagrams:**

Level 1 - System Context:
  Shows the system as a whole
  Its relationships with users and other systems
  Who uses it? What does it depend on?
  
  [User] -> [E-Commerce System] -> [Payment Gateway]
                                 -> [Email Service]
                                 -> [Shipping Provider]

Level 2 - Container:
  Shows high-level containers inside the system
  Web app, API, database, message queue
  Technology choices
  
  [Web App (React)] -> [API (Go)] -> [PostgreSQL]
                                   -> [Redis Cache]
                                   -> [Kafka]
                    -> [CDN (CloudFront)]

Level 3 - Component:
  Shows components within a container
  Services, controllers, repositories
  
  API Container:
    [Auth Controller] -> [User Service] -> [User Repository]
    [Order Controller] -> [Order Service] -> [Order Repository]
                                          -> [Payment Client]
                                          -> [Event Publisher]

Level 4 - Code:
  UML class diagrams, package diagrams
  Usually auto-generated
  Only for complex, critical areas

**Architecture Fitness Functions:**

  Automated tests that verify architecture goals are met
  Run in CI/CD pipeline
  Alert when architecture degrades
  
  Types:
    Static: Analyze code structure (dependency rules, metrics)
    Dynamic: Test runtime behavior (performance, resilience)
    Manual: Review processes (security audits, chaos days)

Examples:
  Dependency constraint: "Domain layer must not import infrastructure"
  Performance: "p99 latency must be under 200ms"
  Resilience: "System must handle 10% error rate gracefully"
  Coupling: "No circular dependencies between packages"
  Size: "No service handles more than one bounded context"
  Security: "All external inputs must be validated"

**Living Documentation:**

  Documentation generated from code or tests
  Always up-to-date by construction
  
  Approaches:
    API docs from OpenAPI/Swagger specs
    Architecture diagrams from code (Structurizr)
    BDD scenarios as living specs
    README files with tested code examples
    Dependency graphs from build tools

  Trade-offs:
    Auto-generated: Always current, less context
    Hand-written: More context, often stale
    Hybrid: Auto-generated with hand-written supplements

**Technical Debt Documentation:**

  Track and categorize technical debt:
    Deliberate/Prudent: "We know this isn't ideal, but we need to ship"
    Deliberate/Reckless: "We don't have time for design"
    Inadvertent/Prudent: "Now we know how we should have done it"
    Inadvertent/Reckless: "What's layering?"

  Tech Debt Register:
    Description: What is the debt?
    Impact: What does it affect? (performance, maintainability)
    Interest: What ongoing cost does it cause?
    Effort: How much work to pay it off?
    Priority: When should it be addressed?

**Architecture Review Process:**

  Lightweight Architecture Review:
    Regular (weekly/biweekly) team discussion
    Review recent ADRs and design proposals
    15-30 minutes, focused
    
  Architecture Board:
    Cross-team review for significant decisions
    Ensures system-wide consistency
    Monthly or as needed
    
  Design Review Checklist:
    Does it follow our architecture principles?
    Are security concerns addressed?
    How does it handle failure?
    Is it observable (logging, metrics, tracing)?
    Does it introduce unwanted coupling?
    Is the migration path clear?
    What are the operational implications?`,
					CodeExamples: `// Architecture fitness functions and documentation

// Dependency constraint checker (build-time)
type DependencyRule struct {
    Source      string // package that has the rule
    Forbidden   []string // packages it must not import
    Description string
}

func CheckDependencyRules(rules []DependencyRule) []Violation {
    var violations []Violation
    
    for _, rule := range rules {
        imports := getPackageImports(rule.Source)
        for _, imp := range imports {
            for _, forbidden := range rule.Forbidden {
                if strings.Contains(imp, forbidden) {
                    violations = append(violations, Violation{
                        Rule:    rule.Description,
                        Package: rule.Source,
                        Import:  imp,
                        Message: fmt.Sprintf("%s must not import %s", rule.Source, forbidden),
                    })
                }
            }
        }
    }
    
    return violations
}

// Architecture rules for the project
var architectureRules = []DependencyRule{
    {
        Source:    "internal/domain",
        Forbidden: []string{"internal/infrastructure", "internal/api", "database/sql"},
        Description: "Domain layer must not depend on infrastructure",
    },
    {
        Source:    "internal/application",
        Forbidden: []string{"internal/infrastructure", "internal/api"},
        Description: "Application layer must not depend on infrastructure or API",
    },
    {
        Source:    "internal/api",
        Forbidden: []string{"internal/infrastructure"},
        Description: "API layer must not directly access infrastructure",
    },
}

// Fitness function: Cyclomatic complexity check
type ComplexityChecker struct {
    maxComplexity int
}

type ComplexityViolation struct {
    File       string
    Function   string
    Complexity int
    Threshold  int
}

func (c *ComplexityChecker) Check(files []string) []ComplexityViolation {
    var violations []ComplexityViolation
    
    for _, file := range files {
        fset := token.NewFileSet()
        f, err := parser.ParseFile(fset, file, nil, 0)
        if err != nil {
            continue
        }
        
        ast.Inspect(f, func(n ast.Node) bool {
            fn, ok := n.(*ast.FuncDecl)
            if !ok {
                return true
            }
            
            complexity := calculateCyclomaticComplexity(fn)
            if complexity > c.maxComplexity {
                violations = append(violations, ComplexityViolation{
                    File:       file,
                    Function:   fn.Name.Name,
                    Complexity: complexity,
                    Threshold:  c.maxComplexity,
                })
            }
            return true
        })
    }
    
    return violations
}

// Architecture metrics collector
type ArchitectureMetrics struct {
    PackageCount       int
    TotalFiles         int
    TotalLines         int
    MaxPackageSize     int
    CircularDeps       int
    AverageCoupling    float64
    TestCoverage       float64
    LargestFunctions   []FunctionMetric
}

type FunctionMetric struct {
    Package    string
    Name       string
    Lines      int
    Complexity int
    Parameters int
}

func CollectArchitectureMetrics(rootDir string) (*ArchitectureMetrics, error) {
    metrics := &ArchitectureMetrics{}
    
    packages, err := listPackages(rootDir)
    if err != nil {
        return nil, err
    }
    
    metrics.PackageCount = len(packages)
    
    for _, pkg := range packages {
        files := listGoFiles(pkg)
        metrics.TotalFiles += len(files)
        
        for _, file := range files {
            lines := countLines(file)
            metrics.TotalLines += lines
        }
        
        pkgSize := countPackageLines(pkg)
        if pkgSize > metrics.MaxPackageSize {
            metrics.MaxPackageSize = pkgSize
        }
    }
    
    metrics.CircularDeps = detectCircularDependencies(packages)
    metrics.AverageCoupling = calculateAverageCoupling(packages)
    
    return metrics, nil
}

// ADR parser and validator
type ADR struct {
    Number      int
    Title       string
    Date        time.Time
    Status      ADRStatus
    Context     string
    Decision    string
    Consequences string
    SupersededBy *int
}

type ADRStatus string
const (
    ADRProposed   ADRStatus = "Proposed"
    ADRAccepted   ADRStatus = "Accepted"
    ADRDeprecated ADRStatus = "Deprecated"
    ADRSuperseded ADRStatus = "Superseded"
)

type ADRValidator struct {
    adrDir string
}

func (v *ADRValidator) Validate() []ADRIssue {
    var issues []ADRIssue
    
    adrs, err := v.loadADRs()
    if err != nil {
        return []ADRIssue{{Message: "Failed to load ADRs: " + err.Error()}}
    }
    
    // Check sequential numbering
    for i, adr := range adrs {
        expected := i + 1
        if adr.Number != expected {
            issues = append(issues, ADRIssue{
                ADR:     adr.Number,
                Message: fmt.Sprintf("Expected ADR-%03d but found ADR-%03d", expected, adr.Number),
            })
        }
    }
    
    // Check superseded ADRs have references
    for _, adr := range adrs {
        if adr.Status == ADRSuperseded && adr.SupersededBy == nil {
            issues = append(issues, ADRIssue{
                ADR:     adr.Number,
                Message: "Superseded ADR must reference the superseding ADR",
            })
        }
    }
    
    // Check required sections
    for _, adr := range adrs {
        if adr.Context == "" {
            issues = append(issues, ADRIssue{ADR: adr.Number, Message: "Missing context section"})
        }
        if adr.Decision == "" {
            issues = append(issues, ADRIssue{ADR: adr.Number, Message: "Missing decision section"})
        }
        if adr.Consequences == "" {
            issues = append(issues, ADRIssue{ADR: adr.Number, Message: "Missing consequences section"})
        }
    }
    
    return issues
}

type ADRIssue struct {
    ADR     int
    Message string
}

// Health dashboard data
type ArchitectureHealth struct {
    Score        float64                    "json:\"score\""
    Checks       map[string]HealthResult    "json:\"checks\""
    Trends       map[string][]TrendPoint    "json:\"trends\""
    LastUpdated  time.Time                  "json:\"last_updated\""
}

type HealthResult struct {
    Status  string  "json:\"status\""
    Score   float64 "json:\"score\""
    Details string  "json:\"details\""
}

type TrendPoint struct {
    Date  time.Time "json:\"date\""
    Value float64   "json:\"value\""
}

func EvaluateArchitectureHealth(metrics *ArchitectureMetrics, rules []DependencyRule) *ArchitectureHealth {
    health := &ArchitectureHealth{
        Checks:      make(map[string]HealthResult),
        Trends:      make(map[string][]TrendPoint),
        LastUpdated: time.Now(),
    }
    
    // Check dependency violations
    violations := CheckDependencyRules(rules)
    if len(violations) == 0 {
        health.Checks["dependencies"] = HealthResult{Status: "pass", Score: 1.0, Details: "No violations"}
    } else {
        health.Checks["dependencies"] = HealthResult{
            Status:  "fail",
            Score:   0.0,
            Details: fmt.Sprintf("%d dependency violations found", len(violations)),
        }
    }
    
    // Check package sizes
    if metrics.MaxPackageSize < 5000 {
        health.Checks["package_size"] = HealthResult{Status: "pass", Score: 1.0}
    } else if metrics.MaxPackageSize < 10000 {
        health.Checks["package_size"] = HealthResult{Status: "warn", Score: 0.5}
    } else {
        health.Checks["package_size"] = HealthResult{Status: "fail", Score: 0.0}
    }
    
    // Check circular dependencies
    if metrics.CircularDeps == 0 {
        health.Checks["circular_deps"] = HealthResult{Status: "pass", Score: 1.0}
    } else {
        health.Checks["circular_deps"] = HealthResult{
            Status:  "fail",
            Score:   0.0,
            Details: fmt.Sprintf("%d circular dependencies", metrics.CircularDeps),
        }
    }
    
    // Calculate overall score
    total := 0.0
    for _, check := range health.Checks {
        total += check.Score
    }
    health.Score = total / float64(len(health.Checks))
    
    return health
}`,
				},
			},
		},
		{
			ID:          2326,
			Title:       "Team Topologies and Conway's Law",
			Description: "Align team structures with architecture using Conway's Law, team topologies, cognitive load management, and organizational patterns for effective software delivery.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "Organizational Architecture and Team Design",
					Content: `Conway's Law states that organizations design systems that mirror their communication structures. Use this intentionally.

**Conway's Law:**

  "Any organization that designs a system will produce a design whose
  structure is a copy of the organization's communication structure."
  - Melvin Conway, 1967

  Implication:
    If three teams build a compiler, you get a three-pass compiler
    Team structure = Architecture structure
    Want microservices? Organize into small, autonomous teams
    
  Inverse Conway Maneuver:
    Deliberately structure teams to get desired architecture
    Don't fight Conway's Law, use it

**Team Topologies (Matthew Skelton & Manuel Pais):**

Four Fundamental Team Types:

1. Stream-Aligned Team:
   Primary type, aligned to a flow of work
   Delivers value to a customer or user
   Cross-functional (dev, ops, test, UX)
   Owns a product or service end-to-end
   Size: 5-9 people (two-pizza team)
   
   Example: Checkout Team, Search Team, Mobile App Team
   
   Responsibilities:
     Full ownership of their services
     On-call for their systems
     Direct customer feedback loop
     Autonomous decision-making

2. Platform Team:
   Enables stream-aligned teams to deliver faster
   Provides self-service internal products
   Reduces cognitive load on stream teams
   
   Example: Infrastructure Platform, CI/CD Platform, Data Platform
   
   Products (not projects):
     Self-service Kubernetes clusters
     CI/CD pipeline templates
     Observability stack (logging, metrics, tracing)
     Database-as-a-service
     Secret management

3. Enabling Team:
   Helps stream-aligned teams overcome obstacles
   Specialists in a particular domain
   Temporary engagement, knowledge transfer focus
   
   Example: Security Enablement, Performance, Architecture
   
   Activities:
     Research and recommend new approaches
     Pair with stream teams to transfer knowledge
     Create guides and documentation
     Run workshops and training

4. Complicated-Subsystem Team:
   Owns a complex subsystem requiring specialist knowledge
   Reduces cognitive load on stream teams
   Only when truly specialized knowledge needed
   
   Example: ML/AI Engine, Video Processing, Financial Calculations
   
   Criteria for creation:
     Requires deep specialist expertise
     Would overwhelm a stream-aligned team
     Shared by multiple stream teams

**Interaction Modes:**

1. Collaboration:
   Two teams work closely together
   High bandwidth communication
   Temporary, discovery-oriented
   Best for: Exploring new approaches, integration challenges
   
2. X-as-a-Service:
   One team provides, other consumes
   Clear API/contract between teams
   Minimal coordination needed
   Best for: Platform capabilities, stable interfaces

3. Facilitating:
   Enabling team helps stream team
   Knowledge transfer, coaching
   Time-limited engagement
   Best for: Adopting new practices, overcoming obstacles

**Cognitive Load:**

Types:
  Intrinsic: Complexity of the task itself
  Extraneous: Complexity from environment (poor tooling, processes)
  Germane: Complexity from learning (good for growth)

Team Cognitive Load:
  Each team has limited cognitive capacity
  Don't overload teams with too many responsibilities
  Use team topologies to manage cognitive load
  
  Too much cognitive load:
    Team owns too many services
    Too many technologies to master
    Complex inter-team dependencies
    Unclear responsibilities

  Signs of overload:
    Slow delivery
    High defect rate
    Team members context-switching frequently
    Knowledge silos within team

  Reduction strategies:
    Platform team abstractions
    Clear domain boundaries
    Self-service capabilities
    Reducing number of services per team

**Organizational Patterns:**

Domain-Aligned Teams:
  Teams organized around business domains
  Matches DDD bounded contexts
  Each team owns their domain end-to-end
  
  Order Domain Team:
    Order Service, Order Database, Order Events
    Owns order lifecycle from creation to fulfillment

  Product Domain Team:
    Product Catalog, Product Search, Product Media
    Owns product information and discovery

Cross-Functional Teams:
  Include all skills needed to deliver
  Developers, testers, ops, UX designer
  Minimize handoffs between teams
  "You build it, you run it"

Scaling Patterns:
  Small org (1-5 teams): Stream-aligned teams only
  Medium org (5-15 teams): Add platform team
  Large org (15+ teams): Full topology with all team types

**Team APIs:**

Each team should define:
  What services/products they offer
  How other teams can interact with them
  Communication channels (Slack, email, office hours)
  Response time expectations
  Documentation and self-service resources

Team API Example:
  Team: Platform Engineering
  Products: Kubernetes clusters, CI/CD pipelines, monitoring
  Interaction: X-as-a-Service
  Self-Service: Internal developer portal
  Support: #platform-support Slack channel
  Office Hours: Wednesdays 2-4pm
  SLO: New cluster provisioned within 4 hours
  Documentation: https://internal-wiki/platform

**Architecture and Team Evolution:**

  Start simple, evolve as needed:
    1. Small team, monolith (< 5 developers)
    2. Growing team, modular monolith (5-15 developers)
    3. Multiple teams, microservices (15+ developers)
    
  Team splits guide service splits:
    When a team gets too big, split it
    Split the codebase along the same boundary
    Each new team owns their services
    
  Architecture follows organization:
    Don't create microservices without matching teams
    One team per bounded context / service group
    Shared services need clear ownership`,
					CodeExamples: `// Team topology modeling and cognitive load analysis

type TeamType string
const (
    StreamAligned       TeamType = "stream-aligned"
    Platform            TeamType = "platform"
    Enabling            TeamType = "enabling"
    ComplicatedSubsystem TeamType = "complicated-subsystem"
)

type InteractionMode string
const (
    Collaboration   InteractionMode = "collaboration"
    XAsAService     InteractionMode = "x-as-a-service"
    Facilitating    InteractionMode = "facilitating"
)

type Team struct {
    Name        string
    Type        TeamType
    Members     int
    Services    []string
    Technologies []string
    Domains     []string
}

type TeamInteraction struct {
    TeamA       string
    TeamB       string
    Mode        InteractionMode
    Purpose     string
    StartDate   time.Time
    ExpectedEnd *time.Time // nil for ongoing
}

// Cognitive load calculator
type CognitiveLoadAssessment struct {
    Team            string
    IntrinsicLoad   float64  // Domain complexity
    ExtraneousLoad  float64  // Environmental complexity
    GermaneLoad     float64  // Learning complexity
    TotalLoad       float64
    Capacity        float64
    OverloadRisk    string   // low, medium, high
    Recommendations []string
}

func AssessCognitiveLoad(team Team) *CognitiveLoadAssessment {
    assessment := &CognitiveLoadAssessment{
        Team:     team.Name,
        Capacity: float64(team.Members) * 10, // arbitrary units
    }
    
    // Intrinsic load: number and complexity of domains
    assessment.IntrinsicLoad = float64(len(team.Domains)) * 15
    
    // Extraneous load: number of services and technologies
    assessment.ExtraneousLoad = float64(len(team.Services))*5 +
        float64(len(team.Technologies))*3
    
    // Germane load: new technologies or domains
    assessment.GermaneLoad = 10 // baseline for continuous learning
    
    assessment.TotalLoad = assessment.IntrinsicLoad +
        assessment.ExtraneousLoad + assessment.GermaneLoad
    
    // Assess risk
    ratio := assessment.TotalLoad / assessment.Capacity
    switch {
    case ratio < 0.7:
        assessment.OverloadRisk = "low"
    case ratio < 0.9:
        assessment.OverloadRisk = "medium"
        assessment.Recommendations = append(assessment.Recommendations,
            "Consider reducing number of services or technologies")
    default:
        assessment.OverloadRisk = "high"
        assessment.Recommendations = append(assessment.Recommendations,
            "Team is overloaded - consider splitting or offloading to platform team",
            "Reduce number of owned services",
            "Consolidate technologies",
        )
    }
    
    // Specific recommendations
    if len(team.Services) > team.Members {
        assessment.Recommendations = append(assessment.Recommendations,
            fmt.Sprintf("Team owns %d services with %d members - consider service consolidation",
                len(team.Services), team.Members))
    }
    
    if len(team.Technologies) > 5 {
        assessment.Recommendations = append(assessment.Recommendations,
            fmt.Sprintf("Team uses %d technologies - consider standardization",
                len(team.Technologies)))
    }
    
    return assessment
}

// Team topology validator
type TopologyValidator struct {
    teams        []Team
    interactions []TeamInteraction
}

type TopologyIssue struct {
    Severity string
    Message  string
    Teams    []string
}

func (v *TopologyValidator) Validate() []TopologyIssue {
    var issues []TopologyIssue
    
    // Check team sizes
    for _, team := range v.teams {
        if team.Members > 9 {
            issues = append(issues, TopologyIssue{
                Severity: "warning",
                Message:  fmt.Sprintf("Team %s has %d members (recommended max: 9)", team.Name, team.Members),
                Teams:    []string{team.Name},
            })
        }
        if team.Members < 3 {
            issues = append(issues, TopologyIssue{
                Severity: "info",
                Message:  fmt.Sprintf("Team %s has only %d members (may lack resilience)", team.Name, team.Members),
                Teams:    []string{team.Name},
            })
        }
    }
    
    // Check interaction modes match team types
    for _, interaction := range v.interactions {
        teamA := v.findTeam(interaction.TeamA)
        teamB := v.findTeam(interaction.TeamB)
        if teamA == nil || teamB == nil {
            continue
        }
        
        // Platform teams should primarily use x-as-a-service
        if teamA.Type == Platform && interaction.Mode == Collaboration {
            issues = append(issues, TopologyIssue{
                Severity: "warning",
                Message: fmt.Sprintf("Platform team %s is collaborating with %s - should aim for x-as-a-service",
                    teamA.Name, teamB.Name),
                Teams: []string{teamA.Name, teamB.Name},
            })
        }
        
        // Enabling teams should primarily use facilitating mode
        if teamA.Type == Enabling && interaction.Mode != Facilitating {
            issues = append(issues, TopologyIssue{
                Severity: "info",
                Message: fmt.Sprintf("Enabling team %s using %s mode with %s - expected facilitating",
                    teamA.Name, interaction.Mode, teamB.Name),
                Teams: []string{teamA.Name, teamB.Name},
            })
        }
    }
    
    // Check for services without clear ownership
    ownedServices := make(map[string]string)
    for _, team := range v.teams {
        for _, svc := range team.Services {
            if existingTeam, exists := ownedServices[svc]; exists {
                issues = append(issues, TopologyIssue{
                    Severity: "error",
                    Message:  fmt.Sprintf("Service %s owned by both %s and %s", svc, existingTeam, team.Name),
                    Teams:    []string{existingTeam, team.Name},
                })
            }
            ownedServices[svc] = team.Name
        }
    }
    
    // Check cognitive load
    for _, team := range v.teams {
        if team.Type == StreamAligned {
            assessment := AssessCognitiveLoad(team)
            if assessment.OverloadRisk == "high" {
                issues = append(issues, TopologyIssue{
                    Severity: "error",
                    Message:  fmt.Sprintf("Team %s has high cognitive load risk (load: %.0f, capacity: %.0f)", team.Name, assessment.TotalLoad, assessment.Capacity),
                    Teams:    []string{team.Name},
                })
            }
        }
    }
    
    return issues
}

func (v *TopologyValidator) findTeam(name string) *Team {
    for i := range v.teams {
        if v.teams[i].Name == name {
            return &v.teams[i]
        }
    }
    return nil
}

// Dependency matrix between teams
type DependencyMatrix struct {
    teams        []string
    dependencies map[string]map[string]int // team -> team -> dependency count
}

func NewDependencyMatrix(teams []string) *DependencyMatrix {
    deps := make(map[string]map[string]int)
    for _, t := range teams {
        deps[t] = make(map[string]int)
    }
    return &DependencyMatrix{teams: teams, dependencies: deps}
}

func (m *DependencyMatrix) AddDependency(from, to string) {
    m.dependencies[from][to]++
}

func (m *DependencyMatrix) HighCouplingPairs(threshold int) []CouplingPair {
    var pairs []CouplingPair
    for from, deps := range m.dependencies {
        for to, count := range deps {
            if count >= threshold {
                pairs = append(pairs, CouplingPair{
                    TeamA: from,
                    TeamB: to,
                    Count: count,
                })
            }
        }
    }
    return pairs
}

type CouplingPair struct {
    TeamA string
    TeamB string
    Count int
}`,
				},
			},
		},
	})
}
